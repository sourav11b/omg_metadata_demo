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
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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

@dataclass
class AgentState:
    """Mutable state that flows through the LangGraph nodes."""
    query: str = ""
    intent: str = ""                       # hybrid | self_query | fulltext | mql
    retrieved_docs: list[dict] = field(default_factory=list)
    answer: str = ""
    trace: list[dict] = field(default_factory=list)  # full execution trace
    tool_calls: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0


def _log(state: AgentState, step: str, detail: Any = None) -> None:
    entry = {"step": step, "ts": time.time()}
    if detail is not None:
        entry["detail"] = detail
    state.trace.append(entry)


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
        HumanMessage(content=state.query),
    ])
    intent = resp.content.strip().lower()
    if intent not in ("hybrid", "self_query", "fulltext", "mql"):
        intent = "hybrid"
    state.intent = intent
    _log(state, "classify_intent", {"intent": intent})
    state.tool_calls.append({"tool": "classify_intent", "result": intent,
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
    intent = state.intent
    try:
        if intent == "hybrid":
            retriever = get_hybrid_retriever(k=5)
            docs = retriever.invoke(state.query)
        elif intent == "self_query":
            llm = get_llm()
            retriever = get_self_query_retriever(llm)
            docs = retriever.invoke(state.query)
        elif intent == "fulltext":
            retriever = get_fulltext_retriever(k=5)
            docs = retriever.invoke(state.query)
        else:
            docs = []

        state.retrieved_docs = _docs_to_dicts(docs)
        _log(state, "retrieve", {"strategy": intent, "doc_count": len(docs)})
        state.tool_calls.append({
            "tool": f"retrieve_{intent}",
            "result": f"{len(docs)} docs retrieved",
            "latency_ms": round((time.time() - t0) * 1000, 1),
        })
    except Exception as exc:
        _log(state, "retrieve_error", {"error": str(exc)})
        state.tool_calls.append({"tool": f"retrieve_{intent}", "error": str(exc)})
        if intent != "hybrid":
            state.intent = "hybrid"
            return retrieve(state)
    return state


# ── Node: Text-to-MQL ────────────────────────────────────────────────────────

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
        HumanMessage(content=state.query),
    ])
    mql_text = resp.content.strip()
    if mql_text.startswith("```"):
        mql_text = "\n".join(mql_text.split("\n")[1:])
        if mql_text.endswith("```"):
            mql_text = mql_text[:-3]

    state.tool_calls.append({
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
        state.retrieved_docs = results
        state.tool_calls.append({"tool": "mql_execute", "result": f"{len(results)} docs"})
    except Exception as exc:
        _log(state, "mql_error", {"error": str(exc)})
        state.tool_calls.append({"tool": "mql_execute", "error": str(exc)})
        state.intent = "hybrid"
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
    context = json.dumps(state.retrieved_docs, indent=2, default=str)[:12000]
    resp = llm.invoke([
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=(
            f"Context:\n{context}\n\n"
            f"Question: {state.query}"
        )),
    ])
    state.answer = resp.content
    total = round((time.time() - t0) * 1000, 1)
    _log(state, "generate", {"latency_ms": total})
    state.tool_calls.append({"tool": "generate_answer", "latency_ms": total})
    # Compute total latency
    state.latency_ms = round(sum(
        tc.get("latency_ms", 0) for tc in state.tool_calls
    ), 1)
    return state


# ── Router Edge ───────────────────────────────────────────────────────────────

def route_after_classify(state: AgentState) -> str:
    """Route to the correct retrieval node based on classified intent."""
    if state.intent == "mql":
        return "text_to_mql"
    return "retrieve"


# ── Build the Graph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the LangGraph state machine."""
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve", retrieve)
    graph.add_node("text_to_mql", text_to_mql)
    graph.add_node("generate", generate)

    # Edges
    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges("classify_intent", route_after_classify,
                                {"retrieve": "retrieve", "text_to_mql": "text_to_mql"})
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


def ask(query: str) -> AgentState:
    """Run a query through the full RAG pipeline and return the final state."""
    graph = get_graph()
    initial = AgentState(query=query)
    result = graph.invoke(initial)
    return result