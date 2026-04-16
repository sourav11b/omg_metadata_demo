"""
AMF-Agent – Atlas Metadata Fabric Semantic Layer Chat

Streamlit application providing:
  • RAG chat interface with hybrid search
  • Full execution trace (intent → retrieval → generation)
  • Tool call details with latency metrics
  • Governance tag visibility (PII, SID, PTB status)

Usage:
    streamlit run app.py
"""

import json
import streamlit as st

from agent.rag_agent import ask, AgentState

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AMF-Agent · Metadata Fabric",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Atlas Metadata Fabric Agent")
st.caption(
    "Semantic Layer for Agentic Commerce — powered by MongoDB Atlas, "
    "Voyage AI embeddings, Azure OpenAI, and LangGraph"
)

# ── Sidebar: architecture info ────────────────────────────────────────────────
with st.sidebar:
    st.header("Architecture")
    st.markdown("""
    **Data Flow**
    1. Heterogeneous sources → MongoDB collections
    2. Change Streams / Atlas Stream Processing → consolidation
    3. Voyage AI → vector embeddings
    4. Atlas Vector Search + Atlas Search indexes

    **RAG Pipeline (LangGraph)**
    ```
    classify_intent
        ├─ hybrid (vector + BM25 RRF)
        ├─ self_query (LLM-generated filters)
        ├─ fulltext (BM25 keyword)
        └─ mql (Text-to-MQL)
    → generate answer
    ```

    **Observability**: LangSmith tracing
    """)
    st.divider()
    st.markdown("**Sample Questions**")
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

# ── Trace rendering helper ────────────────────────────────────────────────────

def _render_trace(trace_data: dict) -> None:
    """Render the full execution trace inside expandable sections."""
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Retrieval Strategy", trace_data.get("intent", "—").upper())
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
            # Highlight governance tags
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


# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "trace_data" in msg:
            _render_trace(msg["trace_data"])

# ── Input ─────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input("Ask about your metadata fabric …") or prefill

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Reasoning …"):
            result: AgentState = ask(user_input)

        # ── Answer ────────────────────────────────────────────────────────
        st.markdown(result.answer)

        # ── Trace panel ──────────────────────────────────────────────────
        trace_data = {
            "intent": result.intent,
            "tool_calls": result.tool_calls,
            "trace": result.trace,
            "retrieved_docs": result.retrieved_docs,
            "latency_ms": result.latency_ms,
        }
        _render_trace(trace_data)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result.answer,
            "trace_data": trace_data,
        })
