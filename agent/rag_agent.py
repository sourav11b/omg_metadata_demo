"""
LangGraph RAG Agent for the Atlas Metadata Fabric Semantic Layer.

State machine flow:
  1. classify_intent  – decide retrieval strategy
  2. retrieve         – run hybrid / self-query / full-text search
  3. generate         – synthesise answer with governance context

Integrates:
  • Azure OpenAI as the LLM
  • **Real MongoDB MCP Server** (launched as a subprocess via
    ``langchain-mcp-adapters``) for structured Text-to-MQL queries
  • LangSmith tracing for full observability
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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
    get_vector_retriever,
    get_hybrid_retriever,
    get_self_query_retriever,
    get_fulltext_retriever,
)

log = logging.getLogger(__name__)


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
    schema_context: str                    # collection schema injected at session start
    chat_history: list[dict]               # prior conversation turns [{role, content}]


def _make_initial_state(query: str, schema_context: str = "",
                        chat_history: list[dict] | None = None) -> AgentState:
    return AgentState(
        query=query, intent="", retrieved_docs=[],
        answer="", trace=[], tool_calls=[], latency_ms=0.0,
        mcp_success=False, schema_context=schema_context,
        chat_history=chat_history or [],
    )


def _history_to_messages(state: AgentState) -> list:
    """Convert chat_history dicts to LangChain message objects."""
    msgs = []
    for turn in state.get("chat_history", []):
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        else:
            from langchain_core.messages import AIMessage
            msgs.append(AIMessage(content=content))
    return msgs


def _log(state: AgentState, step: str, detail: Any = None) -> None:
    entry = {"step": step, "ts": time.time()}
    if detail is not None:
        entry["detail"] = detail
    state.setdefault("trace", []).append(entry)


# ── Node: Classify Intent ─────────────────────────────────────────────────────

INTENT_SYSTEM = f"""You are a metadata search router. Given the user's question,
decide the best retrieval strategy. Respond with EXACTLY one word:
  vector     – conceptual / semantic similarity question (e.g. "entities related to …", "similar to …", "what data do we have about …")
  hybrid     – general question that benefits from both semantic + keyword matching
  self_query – question that filters on specific metadata fields (domain, classification, PII, sensitivity, steward, regulation, ptb_status)
  fulltext   – simple keyword / exact term lookup
  mql        – user wants counts, aggregations, comparisons, raw data, or a structured MongoDB query

IMPORTANT: All queries should target the '{COL_UNIFIED_METADATA}' collection by default.
This is the consolidated semantic layer containing all metadata. Only use source
collections (source_logical_models, source_physical_schemas, source_governance_tags)
if the user EXPLICITLY asks to look at raw/source data.
"""


def classify_intent(state: AgentState) -> AgentState:
    llm = get_llm()
    t0 = time.time()
    messages = [SystemMessage(content=INTENT_SYSTEM)]
    # Include recent history so intent classification understands follow-ups
    messages.extend(_history_to_messages(state))
    messages.append(HumanMessage(content=state["query"]))
    resp = llm.invoke(messages)
    intent = resp.content.strip().lower()
    if intent not in ("vector", "hybrid", "self_query", "fulltext", "mql"):
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
        if intent == "vector":
            retriever = get_vector_retriever(k=5)
            docs = retriever.invoke(query)
        elif intent == "hybrid":
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


# ── Node: MCP Query (Real MongoDB MCP Server via langchain-mcp-adapters) ─────

MCP_SYSTEM_PROMPT = f"""You are a MongoDB query expert with access to MongoDB
MCP Server tools. Use the available tools to answer the user's question.

IMPORTANT: Always target database '{MONGODB_DATABASE}' and collection
'{COL_UNIFIED_METADATA}' by default. This is the consolidated semantic layer
with all metadata (logical model, physical schema, governance tags) merged
into one document per entity.

Only use these source collections if the user EXPLICITLY asks for raw data:
  - 'source_logical_models'
  - 'source_physical_schemas'
  - 'source_governance_tags'

When using the find or aggregate tools:
  - Exclude 'embedding' and 'embedding_text' fields from projections (too large)
  - Limit results to 10 documents unless the user asks for more
  - For aggregate, always pass the database and collection arguments

Use the 'find' tool for simple lookups/filters and 'aggregate' for counts,
groupings, comparisons, and complex pipelines."""


async def _run_mcp_query_async(
    query: str,
    schema_context: str = "",
    chat_history: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """Run a natural language query via the **real** MongoDB MCP Server.

    Launches ``mongodb-mcp-server`` as a stdio subprocess using
    ``langchain-mcp-adapters`` and lets the LLM decide which MCP tool to
    call (``find``, ``aggregate``, etc.).  The MCP Server itself handles
    all MongoDB I/O — no PyMongo calls needed here.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    mcp_client = MultiServerMCPClient(
        {
            "mongodb": {
                "command": "npx",
                "args": ["-y", "mongodb-mcp-server", "--readOnly"],
                "transport": "stdio",
                "env": {
                    "MDB_MCP_CONNECTION_STRING": MONGODB_URI,
                },
            }
        }
    )

    async with mcp_client:
        tools = mcp_client.get_tools()

        if not tools:
            raise RuntimeError("MongoDB MCP Server returned no tools")

        llm = get_llm()
        llm_with_tools = llm.bind_tools(tools)

        # Build system message with optional schema context
        system_text = MCP_SYSTEM_PROMPT
        if schema_context:
            system_text += (
                f"\n\nCOLLECTION SCHEMA (use for accurate field paths):\n"
                f"{schema_context}"
            )

        messages: list = [SystemMessage(content=system_text)]
        # Conversation history for follow-up context
        for turn in (chat_history or []):
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=query))

        # First LLM call — generates tool calls
        ai_response = await llm_with_tools.ainvoke(messages)
        messages.append(ai_response)

        tool_call_info: dict = {}
        results: list[dict] = []

        if ai_response.tool_calls:
            # Execute each tool call via the MCP server
            from langchain_core.messages import ToolMessage

            for tc in ai_response.tool_calls:
                tool_call_info = {
                    "tool_name": tc["name"],
                    "tool_args": tc["args"],
                }
                # Find the matching LangChain tool
                matched_tool = next(
                    (t for t in tools if t.name == tc["name"]), None
                )
                if matched_tool is None:
                    raise RuntimeError(
                        f"MCP tool '{tc['name']}' not found in server tools"
                    )
                # Invoke the tool (MCP server handles MongoDB I/O)
                tool_result = await matched_tool.ainvoke(tc["args"])
                messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tc["id"])
                )
                # Parse the tool result into structured docs
                if isinstance(tool_result, str):
                    try:
                        parsed = json.loads(tool_result)
                        if isinstance(parsed, list):
                            results.extend(parsed)
                        elif isinstance(parsed, dict):
                            results.append(parsed)
                    except (json.JSONDecodeError, TypeError):
                        results.append({"raw_result": tool_result})
                elif isinstance(tool_result, list):
                    results.extend(tool_result)
                elif isinstance(tool_result, dict):
                    results.append(tool_result)
                else:
                    results.append({"raw_result": str(tool_result)})
        else:
            # LLM chose not to call any tools — extract text answer
            results.append({"answer": ai_response.content})
            tool_call_info = {"tool_name": "none", "note": "LLM answered directly"}

        # Sanitise ObjectIds and remove embeddings
        for doc in results:
            if isinstance(doc, dict):
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
                doc.pop("embedding", None)
                doc.pop("embedding_text", None)

    return results, tool_call_info


def _run_mcp_query(
    query: str,
    schema_context: str = "",
    chat_history: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """Synchronous wrapper around the async MCP query."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing event loop (e.g. Streamlit / Jupyter)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                _run_mcp_query_async(query, schema_context, chat_history),
            )
            return future.result(timeout=120)
    else:
        return asyncio.run(
            _run_mcp_query_async(query, schema_context, chat_history)
        )


def mcp_query(state: AgentState) -> AgentState:
    """Primary NLQ path: query via the real MongoDB MCP Server subprocess."""
    t0 = time.time()
    query = state["query"]
    try:
        results, tool_call = _run_mcp_query(
            query, state.get("schema_context", ""),
            chat_history=state.get("chat_history", []),
        )
        state["retrieved_docs"] = results
        state["mcp_success"] = True
        latency = round((time.time() - t0) * 1000, 1)
        _log(state, "mcp_query", {
            "tool_call": tool_call,
            "doc_count": len(results),
            "latency_ms": latency,
        })
        state.setdefault("tool_calls", []).append({
            "tool": "mcp_query (MongoDB MCP Server)",
            "mql": json.dumps(tool_call, indent=2, default=str),
            "result": f"{len(results)} docs retrieved via MCP",
            "latency_ms": latency,
        })
    except Exception as exc:
        log.warning("MCP query failed, falling back: %s", exc, exc_info=True)
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
against the database '{MONGODB_DATABASE}'.

IMPORTANT: Always query the collection '{COL_UNIFIED_METADATA}' by default. This is
the consolidated semantic layer containing all metadata (logical attributes, physical
schema, governance tags, PII/SID classifications) merged per entity.

Only query these source collections if the user EXPLICITLY asks for raw/source data:
  - 'source_logical_models'
  - 'source_physical_schemas'
  - 'source_governance_tags'

Return a JSON object with:
  - "collection": the collection name to query (default: '{COL_UNIFIED_METADATA}')
  - "pipeline": the aggregation pipeline array

Return ONLY valid JSON, no explanation."""


def text_to_mql(state: AgentState) -> AgentState:
    """Use the LLM to translate NL → MQL and execute via PyMongo."""
    from utils.mongo_client import get_collection

    llm = get_llm()
    t0 = time.time()
    schema_ctx = state.get("schema_context", "")
    system_prompt = MQL_SYSTEM
    if schema_ctx:
        system_prompt += f"\n\nCOLLECTION SCHEMA:\n{schema_ctx}"
    resp = llm.invoke([
        SystemMessage(content=system_prompt),
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
        parsed = json.loads(mql_text)
        # Support both formats: raw pipeline array or {collection, pipeline} object
        if isinstance(parsed, list):
            pipeline = parsed
            target_col = COL_UNIFIED_METADATA
        else:
            pipeline = parsed.get("pipeline", parsed)
            target_col = parsed.get("collection", COL_UNIFIED_METADATA)
        col = get_collection(target_col)
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

ANSWER_SYSTEM = f"""You are the Atlas Metadata Fabric Agent (AMF-Agent).
You answer questions about enterprise metadata using the retrieved context below.

The data comes from the '{COL_UNIFIED_METADATA}' collection — the consolidated
Semantic Layer that merges logical models, physical schemas, and governance tags
into a single document per entity.

ALWAYS include governance information (PII tags, SIDs, classification, PTB status,
regulatory frameworks) when present — this is the Permit To Build (PTB) gateway.
Be concise but thorough. If the context is insufficient, say so."""


def generate(state: AgentState) -> AgentState:
    llm = get_llm()
    t0 = time.time()
    context = json.dumps(state.get("retrieved_docs", []), indent=2, default=str)[:12000]

    messages = [SystemMessage(content=ANSWER_SYSTEM)]
    # Inject conversation history so LLM can handle follow-ups
    messages.extend(_history_to_messages(state))
    messages.append(HumanMessage(content=(
        f"Context:\n{context}\n\n"
        f"Question: {state['query']}"
    )))
    resp = llm.invoke(messages)
    state["answer"] = resp.content
    total = round((time.time() - t0) * 1000, 1)
    _log(state, "generate", {"latency_ms": total})
    state.setdefault("tool_calls", []).append({"tool": "generate_answer", "latency_ms": total})
    state["latency_ms"] = round(sum(
        tc.get("latency_ms", 0) for tc in state.get("tool_calls", [])
    ), 1)
    return state


# ── Router Edges ──────────────────────────────────────────────────────────────

def route_after_classify(state: AgentState) -> str:
    """Route based on intent after classification.

    vector / hybrid / fulltext → go directly to their native retrievers
    self_query / mql           → try MCP first (structured query via LLM)

    If the user's query contains "skip MCP" (case-insensitive), MCP is
    bypassed entirely and the fallback path is used instead:
      - mql       → text_to_mql
      - self_query → retrieve (self-query retriever)
    """
    intent = state.get("intent", "hybrid")
    query = state.get("query", "")

    # Check for explicit MCP bypass trigger
    skip_mcp = "skip mcp" in query.lower()
    if skip_mcp:
        # Strip the trigger phrase from the query so downstream nodes
        # don't see it as part of the actual question.
        import re
        cleaned = re.sub(r"(?i)\s*skip\s+mcp\s*", " ", query).strip()
        state["query"] = cleaned
        _log(state, "skip_mcp", "User requested MCP bypass")
        if intent == "mql":
            return "text_to_mql"
        return "retrieve"

    if intent in ("self_query", "mql"):
        return "mcp_query"
    return "retrieve"  # vector, hybrid, fulltext all go to retrieve


def route_after_mcp(state: AgentState) -> str:
    """After MCP: if it succeeded, go to generate; otherwise fallback."""
    if state.get("mcp_success"):
        return "generate"
    intent = state.get("intent", "hybrid")
    if intent == "mql":
        return "text_to_mql"
    return "retrieve"  # self_query falls back to retriever


# ── Build the Graph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the LangGraph state machine.

    Flow:
      classify_intent
        ├─ vector    → retrieve (pure cosine similarity) → generate
        ├─ hybrid    → retrieve (vector + BM25 RRF) → generate
        ├─ fulltext  → retrieve (BM25 keyword) → generate
        ├─ self_query → mcp_query first
        │     ├─ success → generate
        │     └─ fail   → retrieve (self-query retriever) → generate
        └─ mql → mcp_query first
              ├─ success → generate
              └─ fail   → text_to_mql → generate

    "skip MCP" in query → bypass MCP entirely:
        ├─ mql       → text_to_mql directly → generate
        └─ self_query → retrieve directly → generate
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
    graph.add_conditional_edges("classify_intent", route_after_classify,
                                {"mcp_query": "mcp_query",
                                 "retrieve": "retrieve",
                                 "text_to_mql": "text_to_mql"})
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


def ask(query: str, schema_context: str = "",
        chat_history: list[dict] | None = None) -> dict:
    """Run a query through the full RAG pipeline and return the final state dict.

    Args:
        query: The user's question.
        schema_context: Collection schema text for MCP/MQL prompts.
        chat_history: Prior conversation turns [{role, content}, ...].
                      Last 10 turns are used to enable follow-up questions.
    """
    # Limit to last 10 turns to avoid token overflow
    history = (chat_history or [])[-10:]
    graph = get_graph()
    initial = _make_initial_state(query, schema_context=schema_context,
                                  chat_history=history)
    result = graph.invoke(initial)
    return result