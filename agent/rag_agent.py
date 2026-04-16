"""
LangGraph RAG Agent for the Atlas Metadata Fabric Semantic Layer.

State machine flow:
  1. classify_intent  – decide retrieval strategy
  2. retrieve         – run hybrid / self-query / full-text search
  3. generate         – synthesise answer with governance context

Integrates:
  • Azure OpenAI as the LLM
  • MongoDB MCP Server (via langchain-mcp-adapters) for Text-to-MQL
  • LangSmith tracing for full observability
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph

from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    MONGODB_URI,
    MONGODB_DATABASE,
    COL_UNIFIED_METADATA,
)
from search.hybrid_search import (
    get_hybrid_retriever,
    get_self_query_retriever,
    get_fulltext_retriever,
)


# ── LLM ───────────────────────────────────────────────────────────────────────

def get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0,
    )


# ── Agent State ────────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """Mutable state that flows through the LangGraph nodes."""
    query: str
    intent: str                            # hybrid | self_query | fulltext | mql
    retrieved_docs: list[dict]
    answer: str
    trace: list[dict]                      # full execution trace
    tool_calls: list[dict]
    latency_ms: float
    mcp_success: bool                      # whether MCP query succeeded


def _make_initial_state(query: str) -> AgentState:
    return AgentState(
        query=query, intent="", retrieved_docs=[],
        answer="", trace=[], tool_calls=[], latency_ms=0.0,
        mcp_success=False,
    )


def _log(state: AgentState, step: str, detail: Any = None) -> None:
    entry = {"step": step, "ts": time.time()}
    if detail is not None:
        entry["detail"] = detail
    state.setdefault("trace", []).append(entry)


# ── Node: Classify Intent ─────────────────────────────────────────────────────

INTENT_SYSTEM = """You are a metadata search router. Given the user's question,
decide the best retrieval strategy. Respond with EXACTLY one word:
  hybrid     – general semantic + keyword question
  self_query – question that filters on specific metadata fields (domain, classification, PII, etc.)
  fulltext   – simple keyword lookup
  mql        – user wants raw data or a structured MongoDB query
"""


def classify_intent(state: AgentState) -> AgentState:
    llm = get_llm()
    t0 = time.time()
    resp = llm.invoke([
        SystemMessage(content=INTENT_SYSTEM),
        HumanMessage(content=state["query"]),
    ])
    intent = resp.content.strip().lower()
    if intent not in ("hybrid", "self_query", "fulltext", "mql"):
        intent = "hybrid"
    state["intent"] = intent
    _log(state, "classify_intent", {"intent": intent})
    state.setdefault("tool_calls", []).append(
        {"tool": "classify_intent", "result": intent,
         "latency_ms": round((time.time() - t0) * 1000, 1)})
    return state



# ── Node: Retrieve ────────────────────────────────────────────────────────────

def _docs_to_dicts(docs: list[Document]) -> list[dict]:
    """Convert LangChain Documents to serialisable dicts."""
    results = []
    for d in docs:
        entry = {"content": d.page_content}
        entry.update(d.metadata)
        results.append(entry)
    return results


def retrieve(state: AgentState) -> AgentState:
    t0 = time.time()
    intent = state.get("intent", "hybrid")
    query = state["query"]
    try:
        if intent == "hybrid":
            retriever = get_hybrid_retriever(k=5)
            docs = retriever.invoke(query)
        elif intent == "self_query":
            llm = get_llm()
            retriever = get_self_query_retriever(llm)
            docs = retriever.invoke(query)
        elif intent == "fulltext":
            retriever = get_fulltext_retriever(k=5)
            docs = retriever.invoke(query)
        else:
            docs = []

        state["retrieved_docs"] = _docs_to_dicts(docs)
        _log(state, "retrieve", {"strategy": intent, "doc_count": len(docs)})
        state.setdefault("tool_calls", []).append({
            "tool": f"retrieve_{intent}",
            "result": f"{len(docs)} docs retrieved",
            "latency_ms": round((time.time() - t0) * 1000, 1),
        })
    except Exception as exc:
        _log(state, "retrieve_error", {"error": str(exc)})
        state.setdefault("tool_calls", []).append({"tool": f"retrieve_{intent}", "error": str(exc)})
        if intent != "hybrid":
            state["intent"] = "hybrid"
            return retrieve(state)
    return state


# ── Node: MCP Query (MongoDB MCP Server — primary NLQ path) ──────────────────

def _run_mcp_query(query: str) -> list[dict]:
    """Run a natural language query via the MongoDB MCP Server (stdio).

    Spawns npx mongodb-mcp-server as a subprocess, sends the query
    as a tool call, and returns the results.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    async def _do():
        async with MultiServerMCPClient({
            "mongodb": {
                "transport": "stdio",
                "command": "npx",
                "args": [
                    "-y", "mongodb-mcp-server@latest",
                    "--connectionString", MONGODB_URI,
                    "--readOnly",
                ],
            }
        }) as client:
            tools = client.get_tools()
            # Find a query / find tool
            query_tool = None
            for t in tools:
                if "find" in t.name.lower() or "query" in t.name.lower():
                    query_tool = t
                    break
            if not query_tool:
                # Use the LLM to pick the right tool and args
                llm = get_llm()
                from langgraph.prebuilt import create_react_agent
                agent = create_react_agent(llm, tools)
                result = await agent.ainvoke(
                    {"messages": [HumanMessage(content=(
                        f"Query the MongoDB database '{MONGODB_DATABASE}', "
                        f"collection '{COL_UNIFIED_METADATA}' to answer: {query}"
                    ))]}
                )
                # Extract content from last AI message
                last_msg = result["messages"][-1]
                content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
                try:
                    return json.loads(content) if content.strip().startswith("[") else [{"mcp_response": content}]
                except (json.JSONDecodeError, TypeError):
                    return [{"mcp_response": content}]
            else:
                # Directly invoke the find tool
                result = await query_tool.ainvoke({
                    "database": MONGODB_DATABASE,
                    "collection": COL_UNIFIED_METADATA,
                    "filter": {},
                })
                if isinstance(result, str):
                    try:
                        return json.loads(result)
                    except (json.JSONDecodeError, TypeError):
                        return [{"mcp_response": result}]
                return result if isinstance(result, list) else [result]

    return asyncio.run(_do())


def mcp_query(state: AgentState) -> AgentState:
    """Primary NLQ path: try MongoDB MCP Server first."""
    t0 = time.time()
    query = state["query"]
    try:
        results = _run_mcp_query(query)
        # Sanitise ObjectIds
        for doc in results:
            if isinstance(doc, dict) and "_id" in doc:
                doc["_id"] = str(doc["_id"])
        state["retrieved_docs"] = results
        state["mcp_success"] = True
        latency = round((time.time() - t0) * 1000, 1)
        _log(state, "mcp_query", {"doc_count": len(results), "latency_ms": latency})
        state.setdefault("tool_calls", []).append({
            "tool": "mcp_query (MongoDB MCP Server)",
            "result": f"{len(results)} docs retrieved via MCP",
            "latency_ms": latency,
        })
    except Exception as exc:
        state["mcp_success"] = False
        _log(state, "mcp_query_fallback", {"error": str(exc)})
        state.setdefault("tool_calls", []).append({
            "tool": "mcp_query (MongoDB MCP Server)",
            "error": f"MCP failed, falling back: {exc}",
            "latency_ms": round((time.time() - t0) * 1000, 1),
        })
    return state


# ── Node: Text-to-MQL (fallback for MQL intent) ─────────────────────────────

MQL_SYSTEM = f"""You are a MongoDB query expert. Given a natural language question
about the metadata fabric, generate a valid MongoDB aggregation pipeline (as JSON)
against the collection '{COL_UNIFIED_METADATA}' in database '{MONGODB_DATABASE}'.
Return ONLY the JSON array of pipeline stages, no explanation."""


def text_to_mql(state: AgentState) -> AgentState:
    """Use the LLM to translate NL → MQL and execute via PyMongo."""
    from utils.mongo_client import get_collection

    llm = get_llm()
    t0 = time.time()
    resp = llm.invoke([
        SystemMessage(content=MQL_SYSTEM),
        HumanMessage(content=state["query"]),
    ])
    mql_text = resp.content.strip()
    if mql_text.startswith("```"):
        mql_text = "\n".join(mql_text.split("\n")[1:])
        if mql_text.endswith("```"):
            mql_text = mql_text[:-3]

    state.setdefault("tool_calls", []).append({
        "tool": "text_to_mql",
        "mql": mql_text,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    })
    _log(state, "text_to_mql", {"mql": mql_text})

    try:
        pipeline = json.loads(mql_text)
        col = get_collection(COL_UNIFIED_METADATA)
        results = list(col.aggregate(pipeline))
        for doc in results:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
        state["retrieved_docs"] = results
        state.setdefault("tool_calls", []).append({"tool": "mql_execute", "result": f"{len(results)} docs"})
    except Exception as exc:
        _log(state, "mql_error", {"error": str(exc)})
        state.setdefault("tool_calls", []).append({"tool": "mql_execute", "error": str(exc)})
        state["intent"] = "hybrid"
        return retrieve(state)

    return state


# ── Node: Generate Answer ────────────────────────────────────────────────────

ANSWER_SYSTEM = """You are the Atlas Metadata Fabric Agent (AMF-Agent).
You answer questions about enterprise metadata using the retrieved context below.
ALWAYS include governance information (PII tags, SIDs, classification, PTB status,
regulatory frameworks) when present — this is the Permit To Build (PTB) gateway.
Be concise but thorough. If the context is insufficient, say so."""


def generate(state: AgentState) -> AgentState:
    llm = get_llm()
    t0 = time.time()
    context = json.dumps(state.get("retrieved_docs", []), indent=2, default=str)[:12000]
    resp = llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=(
            f"Context:\n{context}\n\n"
            f"Question: {state['query']}"
        )),
    ])
    state["answer"] = resp.content
    total = round((time.time() - t0) * 1000, 1)
    _log(state, "generate", {"latency_ms": total})
    state.setdefault("tool_calls", []).append({"tool": "generate_answer", "latency_ms": total})
    state["latency_ms"] = round(sum(
        tc.get("latency_ms", 0) for tc in state.get("tool_calls", [])
    ), 1)
    return state


# ── Router Edges ──────────────────────────────────────────────────────────────

def route_after_mcp(state: AgentState) -> str:
    """After MCP: if it succeeded, go straight to generate; otherwise fallback."""
    if state.get("mcp_success"):
        return "generate"
    # MCP failed — fall back based on intent
    intent = state.get("intent", "hybrid")
    if intent == "mql":
        return "text_to_mql"
    return "retrieve"


# ── Build the Graph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the LangGraph state machine.

    Flow:
      classify_intent → mcp_query (always try MCP first)
        ├─ success → generate
        └─ fail   → retrieve / text_to_mql (based on intent) → generate
    """
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("mcp_query", mcp_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("text_to_mql", text_to_mql)
    graph.add_node("generate", generate)

    # Edges
    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "mcp_query")  # always try MCP first
    graph.add_conditional_edges("mcp_query", route_after_mcp,
                                {"generate": "generate",
                                 "retrieve": "retrieve",
                                 "text_to_mql": "text_to_mql"})
    graph.add_edge("retrieve", "generate")
    graph.add_edge("text_to_mql", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────

_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def ask(query: str) -> dict:
    """Run a query through the full RAG pipeline and return the final state dict."""
    graph = get_graph()
    initial = _make_initial_state(query)
    result = graph.invoke(initial)
    return result