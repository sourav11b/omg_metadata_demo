"""
AMF-Agent – Atlas Metadata Fabric Semantic Layer Chat

Features:
  • RAG chat with hybrid / self-query / full-text / Text-to-MQL retrieval
  • Full execution trace with tool calls and latency
  • Governance tag visibility (PII, SID, PTB status)
  • Ingestion & consolidation button with live status
  • Data seed / index status dashboard
  • Session memory persisted in MongoDB, with reset
  • LangChain caching via MongoDB

Usage:
    streamlit run app.py
"""

import datetime
import json
import time
import uuid

import streamlit as st

from agent.rag_agent import ask
from config.settings import (
    COL_LOGICAL_MODELS,
    COL_PHYSICAL_SCHEMAS,
    COL_GOVERNANCE_TAGS,
    COL_UNIFIED_METADATA,
    COL_CONVERSATION_HISTORY,
    COL_SESSION_MEMORY,
    MONGODB_DATABASE,
    VECTOR_SEARCH_INDEX,
    FULLTEXT_SEARCH_INDEX,
)
from utils.mongo_client import get_collection, get_database


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AMF-Agent Demo · Real-Time Semantic Layer for Agentic Commerce",
    page_icon="🧠",
    layout="wide",
)

# ── MongoDB helpers ───────────────────────────────────────────────────────────

def _col_count(name: str) -> int:
    try:
        return get_collection(name).count_documents({})
    except Exception:
        return 0


def _save_message(session_id: str, role: str, content: str,
                  trace_data: dict | None = None) -> None:
    """Persist a single chat message to MongoDB."""
    doc = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
    }
    if trace_data:
        doc["trace_data"] = trace_data
    get_collection(COL_CONVERSATION_HISTORY).insert_one(doc)


def _load_session_messages(session_id: str) -> list[dict]:
    """Load conversation history for a session from MongoDB."""
    try:
        docs = list(
            get_collection(COL_CONVERSATION_HISTORY)
            .find({"session_id": session_id}, {"_id": 0})
            .sort("timestamp", 1)
        )
        return [{"role": d["role"], "content": d["content"],
                 **({"trace_data": d["trace_data"]} if "trace_data" in d else {})}
                for d in docs]
    except Exception:
        return []


def _save_semantic_memory(session_id: str, query: str, answer: str,
                          intent: str) -> None:
    """Store a semantic memory entry (query → answer mapping)."""
    get_collection(COL_SESSION_MEMORY).insert_one({
        "session_id": session_id,
        "query": query,
        "answer": answer,
        "intent": intent,
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
    })


def _get_semantic_context(session_id: str, limit: int = 5) -> str:
    """Retrieve recent semantic memory for context injection."""
    try:
        docs = list(
            get_collection(COL_SESSION_MEMORY)
            .find({"session_id": session_id}, {"_id": 0, "query": 1, "answer": 1})
            .sort("timestamp", -1)
            .limit(limit)
        )
        if not docs:
            return ""
        lines = []
        for d in reversed(docs):
            lines.append(f"Q: {d['query']}\nA: {d['answer']}")
        return "\n---\n".join(lines)
    except Exception:
        return ""


def _clear_session(session_id: str) -> None:
    """Delete all conversation and memory data for a session."""
    get_collection(COL_CONVERSATION_HISTORY).delete_many({"session_id": session_id})
    get_collection(COL_SESSION_MEMORY).delete_many({"session_id": session_id})


# ── Session ID management ─────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = _load_session_messages(st.session_state.session_id)

# ── Schema introspection at session start ─────────────────────────────────────
if "schema_context" not in st.session_state:
    from utils.schema_inspector import inspect_collection
    try:
        st.session_state.schema_context = inspect_collection()
    except Exception:
        st.session_state.schema_context = ""


# ── Trace rendering helper ────────────────────────────────────────────────────

def _render_trace(trace_data: dict) -> None:
    """Render the full execution trace inside expandable sections."""
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Retrieval Strategy", str(trace_data.get("intent", "—")).upper())
    with col2:
        st.metric("Total Latency", f"{trace_data.get('latency_ms', 0):.0f} ms")

    with st.expander("🔍 Tool Calls & Execution Trace", expanded=False):
        for i, tc in enumerate(trace_data.get("tool_calls", []), 1):
            tool_name = tc.get("tool", "unknown")
            latency = tc.get("latency_ms", "—")
            st.markdown(f"**Step {i}: `{tool_name}`** — *{latency} ms*")
            if "mql" in tc:
                st.code(tc["mql"], language="json")
            if "result" in tc:
                st.write(f"Result: {tc['result']}")
            if "error" in tc:
                st.error(f"Error: {tc['error']}")
            st.divider()

    with st.expander("📄 Retrieved Documents & Governance Tags", expanded=False):
        docs = trace_data.get("retrieved_docs", [])
        if not docs:
            st.info("No documents retrieved.")
        for i, doc in enumerate(docs, 1):
            st.markdown(f"### Doc {i}: {doc.get('entity_name', doc.get('entity_id', ''))}")
            gov = doc.get("governance", {})
            if gov:
                tags = gov.get("tags", [])
                pii_tags = [t for t in tags if t.get("tag") == "PII"]
                sid_tags = [t for t in tags if t.get("tag") == "SID"]
                if pii_tags:
                    st.warning(f"⚠️ PII Fields: {', '.join(t['field'] for t in pii_tags)}")
                if sid_tags:
                    st.info(f"🔑 SID Fields: {', '.join(t['field'] for t in sid_tags)}")
                st.write(f"**Classification:** {gov.get('classification')}")
                st.write(f"**PTB Status:** {gov.get('ptb_status')}")
                st.write(f"**Regulatory:** {', '.join(gov.get('regulatory_frameworks', []))}")
                st.write(f"**Data Steward:** {gov.get('data_steward')}")
            st.json(doc)
            st.divider()

    with st.expander("🪵 Raw Trace Log", expanded=False):
        st.json(trace_data.get("trace", []))


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 Atlas Metadata Fabric Agent (AMF-Agent) Demo")
st.subheader("Real-Time Semantic Layer for Agentic Commerce")

with st.expander("ℹ️ About this Demo", expanded=False):
    st.markdown("""
**This demonstration showcases the Atlas Metadata Fabric Agent (AMF-Agent)** and its
ability to construct a real-time, unified Semantic Layer. Data from heterogeneous
sources (models, schema, tags) is ingested into MongoDB and consolidated via Change
Streams and Atlas Stream Processing into comprehensive metadata documents. Atlas
Vector Search is used to generate and store embeddings, enabling Retrieval-Augmented
Generation (RAG) and hybrid search for an Agentic Commerce foundation.

The demo features a **low-latency chat interface** where a business user can ask
natural language questions. Hybrid search, combining keyword and semantic search,
retrieves relevant metadata, explicitly displaying governance tags (e.g., PII, SIDs)
to establish the Semantic Layer as the **"Permit to Build" (PTB) gateway** for AI
agents. The application utilizes LangChain/LangGraph, Azure OpenAI, and Voyage AI,
with full tracing via LangSmith, positioning Atlas as the **"operational nervous
system"** for AI.

---

**OMG (One Meta Guard):** Amex's system for standardization, tagging, and governance
of metadata (e.g., PII, SIDs). It is the Permit to Build (PTB) gateway for new
business requirements, ensuring tagging happens correctly and guardrails are applied
before production. OMG is already in production and will consume the Semantic Layer.
The backend of OMG is planned to go to MongoDB.

**AMF-Agent (Atlas Metadata Fabric Agent):** A demonstration application built by
MongoDB. The AMF-Agent Demo showcases how Atlas can construct a real-time, unified
Semantic Layer. This Semantic Layer, created by the AMF-Agent, is critical for OMG
and all Gen AI applications.
    """)

# ── Architecture Diagrams ─────────────────────────────────────────────────────
import streamlit.components.v1 as components

def _render_mermaid(diagram: str, height: int = 520) -> None:
    """Render a Mermaid diagram inside a Streamlit component."""
    # Escape the diagram for safe JS embedding
    escaped = diagram.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    html = (
        '<!DOCTYPE html><html><head>'
        '<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>'
        '</head><body style="margin:0;padding:12px;background:#0e1117;">'
        '<div id="graph"></div>'
        '<script>'
        'mermaid.initialize({startOnLoad:false,theme:"dark",'
        'themeVariables:{'
        'primaryColor:"#1a1a2e",primaryTextColor:"#e0e0e0",'
        'primaryBorderColor:"#4a4a6a",lineColor:"#888",'
        'secondaryColor:"#16213e",tertiaryColor:"#0f3460",'
        'fontSize:"13px",edgeLabelBackground:"#1a1a2e"'
        '}});'
        'var def = `' + escaped + '`;'
        'mermaid.render("mermaid-svg", def).then(function(result){'
        'document.getElementById("graph").innerHTML = result.svg;'
        '});'
        '</script></body></html>'
    )
    components.html(html, height=height)

with st.expander("📊 Data Ingestion & Consolidation Flow", expanded=False):
    st.caption("How heterogeneous metadata flows into MongoDB and gets consolidated into the Semantic Layer")
    _render_mermaid("""
flowchart LR
    subgraph Sources["Heterogeneous Sources"]
        direction TB
        LM["Logical Models<br/>Entity definitions<br/>Attributes and types<br/>Relationships"]
        PS["Physical Schemas<br/>DB / Table / Columns<br/>Data types and formats<br/>Row counts"]
        GT["Governance Tags<br/>PII / SID tags<br/>Classification<br/>PTB status<br/>Regulatory frameworks"]
    end

    subgraph Ingest["MongoDB Atlas Source Collections"]
        direction TB
        C1[("source_logical_models")]
        C2[("source_physical_schemas")]
        C3[("source_governance_tags")]
    end

    subgraph Process["Real-Time Processing"]
        direction TB
        CS["Change Streams Worker<br/>Watches all 3 collections<br/>Merges by entity_id"]
        ASP["Atlas Stream Processing<br/>3 parallel processors<br/>merge into unified"]
    end

    subgraph Semantic["Semantic Layer"]
        direction TB
        UM[("unified_metadata<br/>Consolidated document<br/>All sources merged")]
        VE["Voyage AI Embeddings<br/>voyage-finance-2"]
        VS["Atlas Vector Search Index<br/>cosine similarity"]
        FTS["Atlas Search Full-Text Index<br/>BM25 / Lucene"]
    end

    LM --> C1
    PS --> C2
    GT --> C3

    C1 --> CS
    C2 --> CS
    C3 --> CS
    C1 -.-> ASP
    C2 -.-> ASP
    C3 -.-> ASP

    CS --> UM
    ASP -.-> UM
    UM --> VE
    VE --> VS
    UM --> FTS
    """, height=520)

with st.expander("🤖 RAG Chat Agent — Query Paths", expanded=False):
    st.caption("How the chatbot processes your question through different retrieval strategies")
    _render_mermaid("""
flowchart TD
    Q["User Question"]
    CI["Classify Intent<br/>Azure OpenAI<br/>Determines best strategy"]

    subgraph MCP["Step 1: MCP Query - Always First"]
        direction TB
        MC["MongoDB MCP Server<br/>LLM generates find/aggregate<br/>Executes via PyMongo"]
    end

    MCPOK{"MCP<br/>Success?"}

    subgraph Fallback["Step 2: Fallback if MCP fails"]
        direction TB
        HY["Hybrid Search<br/>MongoDBAtlasHybridSearchRetriever<br/>Vector plus BM25 via RRF"]
        SQ["Self-Query<br/>MongoDBAtlasSelfQueryRetriever<br/>LLM-generated metadata filters"]
        FT["Full-Text<br/>MongoDBAtlasFullTextSearchRetriever<br/>BM25 keyword search"]
        MQL["Text-to-MQL<br/>NL to MongoDB Aggregation<br/>Direct pipeline execution"]
    end

    GEN["Generate Answer<br/>Azure OpenAI gpt-4o<br/>Includes governance context<br/>PII/SID/PTB tags highlighted"]

    TRACE["Execution Trace<br/>Intent classification<br/>Tool calls and latency<br/>Retrieved docs<br/>Governance tags"]

    Q --> CI
    CI --> MC
    MC --> MCPOK
    MCPOK -- "Yes" --> GEN
    MCPOK -- "No" --> HY
    MCPOK -- "No" --> SQ
    MCPOK -- "No" --> FT
    MCPOK -- "No" --> MQL

    HY --> GEN
    SQ --> GEN
    FT --> GEN
    MQL --> GEN

    GEN --> TRACE
    """, height=650)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    # ── Session management ─────────────────────────────────────────────────
    st.subheader("Session")
    st.caption(f"ID: `{st.session_state.session_id[:8]}…`")
    if st.button("🔄 Reset Session", use_container_width=True):
        _clear_session(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        # Re-introspect schema on reset
        from utils.schema_inspector import inspect_collection
        try:
            st.session_state.schema_context = inspect_collection()
        except Exception:
            st.session_state.schema_context = ""
        st.rerun()

    st.divider()

    # ── Ingestion ───────────────────────────────────────────────────────────
    st.subheader("📥 Data Ingestion")
    ingest_count = st.number_input(
        "New entities per click", min_value=10, max_value=5000,
        value=1000, step=100, key="ingest_count",
    )
    if st.button("▶️ Ingest New Data", use_container_width=True):
        from ingestion.ingest import run_ingestion
        with st.spinner(f"Ingesting {ingest_count} new + 50 updates …"):
            stats = run_ingestion(new_count=int(ingest_count), update_count=50)
        st.success(
            f"✅ Inserted: {stats['new']['logical_models']} LM, "
            f"{stats['new']['physical_schemas']} PS, "
            f"{stats['new']['governance_tags']} GT  ·  "
            f"Updated: {stats['updated']['logical_models']} LM, "
            f"{stats['updated']['physical_schemas']} PS, "
            f"{stats['updated']['governance_tags']} GT"
        )

    st.divider()

    # ── Consolidation (background) ─────────────────────────────────────────
    st.subheader("🔄 Consolidation")
    from ingestion.change_stream_worker import (
        start_background_consolidation,
        stop_background_consolidation,
        is_background_running,
        get_consolidation_stats,
    )

    running = is_background_running()
    if running:
        st.success("🟢 Background consolidation running")
        if st.button("⏹️ Stop Consolidation", use_container_width=True, type="secondary"):
            stop_background_consolidation()
            st.rerun()
    else:
        st.info("⚪ Background consolidation stopped")
        if st.button("▶️ Start Consolidation", use_container_width=True, type="primary"):
            start_background_consolidation()
            st.rerun()

    # Show consolidation stats
    c_stats = get_consolidation_stats()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Consolidated", c_stats.get("total_consolidated", 0))
    with c2:
        st.metric("Pending", c_stats.get("pending", 0))
    if c_stats.get("errors", 0) > 0:
        st.warning(f"⚠️ {c_stats['errors']} consolidation errors")
    if c_stats.get("last_consolidated_at"):
        st.caption(f"Last: {c_stats['last_consolidated_at']}")

    st.divider()

    # ── Data & Index Status Dashboard ──────────────────────────────────────
    st.subheader("📊 Status Dashboard")
    if st.button("🔃 Refresh Status", use_container_width=True) or "show_status" not in st.session_state:
        st.session_state.show_status = True

    if st.session_state.get("show_status"):
        st.markdown("**Collection Counts**")
        cols = st.columns(2)
        with cols[0]:
            st.metric("Logical Models", _col_count(COL_LOGICAL_MODELS))
            st.metric("Physical Schemas", _col_count(COL_PHYSICAL_SCHEMAS))
        with cols[1]:
            st.metric("Governance Tags", _col_count(COL_GOVERNANCE_TAGS))
            st.metric("Unified Metadata", _col_count(COL_UNIFIED_METADATA))

        st.markdown("**Session Data**")
        st.metric("Conversation Messages", _col_count(COL_CONVERSATION_HISTORY))
        st.metric("Semantic Memories", _col_count(COL_SESSION_MEMORY))

        # Index status
        st.markdown("**Search Indexes**")
        try:
            col = get_collection(COL_UNIFIED_METADATA)
            indexes = list(col.list_search_indexes())
            for idx in indexes:
                name = idx.get("name", "?")
                status = idx.get("status", "?")
                icon = "✅" if status == "READY" else "⏳"
                st.write(f"{icon} `{name}` — {status}")
            if not indexes:
                st.warning("No search indexes found. Run `python -m indexes.setup_indexes`")
        except Exception as e:
            st.warning(f"Could not list indexes: {e}")

        # Schema context
        if st.session_state.get("schema_context"):
            with st.expander("🔎 Collection Schema (session context)", expanded=False):
                st.code(st.session_state.schema_context, language="text")

    st.divider()

    # ── Sample questions ───────────────────────────────────────────────────
    st.subheader("💡 Sample Questions")
    samples = [
        "What PII fields exist in the Customer entity?",
        "Which entities are subject to PCI-DSS?",
        "Show me the physical schema for transactions",
        "What is the PTB status for the Account entity?",
        "Find all entities with High sensitivity tags",
        "What merchant data do we store and who is the data steward?",
    ]
    for s in samples:
        if st.button(s, key=s):
            st.session_state["prefill"] = s


# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "trace_data" in msg:
            _render_trace(msg["trace_data"])

# ── Chat input ────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input("Ask about your metadata fabric …") or prefill

if user_input:
    # Save & display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    _save_message(st.session_state.session_id, "user", user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Reasoning …"):
            result = ask(user_input, schema_context=st.session_state.get("schema_context", ""))

        answer = result.get("answer", "No answer generated.")
        st.markdown(answer)

        # Build trace data
        trace_data = {
            "intent": result.get("intent", ""),
            "tool_calls": result.get("tool_calls", []),
            "trace": result.get("trace", []),
            "retrieved_docs": result.get("retrieved_docs", []),
            "latency_ms": result.get("latency_ms", 0),
        }
        _render_trace(trace_data)

        # Persist to MongoDB
        _save_message(st.session_state.session_id, "assistant", answer, trace_data)
        _save_semantic_memory(
            st.session_state.session_id, user_input, answer,
            result.get("intent", ""),
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "trace_data": trace_data,
        })