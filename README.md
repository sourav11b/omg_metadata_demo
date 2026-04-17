# 🧠 Atlas Metadata Fabric Agent (AMF-Agent)

A demo application that simulates metadata ingestion, consolidation, and discovery to create a **Semantic Layer** — providing real-time context for an Agentic Commerce foundation powered by MongoDB Atlas.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Heterogeneous Source Systems                     │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────┐     │
│  │Logical Models│  │Physical Schemas  │  │Governance Tags    │     │
│  │(entities,    │  │(DB/table/columns,│  │(PII, SID, PTB,    │     │
│  │ attributes)  │  │ storage formats) │  │ regulatory)       │     │
│  └──────┬───────┘  └────────┬─────────┘  └─────────┬─────────┘     │
└─────────┼──────────────────┼───────────────────────┼───────────────┘
          │                  │                       │
          ▼                  ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│              MongoDB Atlas (Source Collections)                     │
│  source_logical_models │ source_physical_schemas │ source_gov_tags  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼                               ▼
   ┌─────────────────────┐        ┌──────────────────────┐
   │  Change Streams      │        │  Atlas Stream         │
   │  Worker (Python)     │        │  Processing (API)     │
   └──────────┬──────────┘        └──────────┬───────────┘
              │     Consolidate + Embed       │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Semantic Layer (unified_metadata)                 │
│  ┌────────────────┐  ┌──────────────────┐  ┌────────────────────┐  │
│  │Unified Document│  │Atlas Vector Search│  │Atlas Search (BM25) │  │
│  │(consolidated   │  │(Voyage AI embeds) │  │(full-text index)   │  │
│  │ metadata)      │  │                  │  │                    │  │
│  └────────────────┘  └──────────────────┘  └────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LangGraph RAG Agent                              │
│                                                                     │
│  classify_intent                                                    │
│    ├─ vector    → Vector Search (cosine similarity)                │
│    ├─ hybrid    → Hybrid Search (Vector + BM25 via RRF)            │
│    ├─ fulltext  → Full-Text Search (BM25 keyword)                  │
│    ├─ self_query → MongoDB MCP Server* → fallback: Self-Query      │
│    └─ mql       → MongoDB MCP Server* → fallback: Text-to-MQL     │
│                          │                                          │
│  * MCP Server: LLM generates find/aggregate queries executed via   │
│    PyMongo, following the MCP (Model Context Protocol) pattern     │
│                          │                                          │
│                          ▼                                          │
│                    generate_answer (Azure OpenAI)                   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Streamlit Chat UI                                      │
│  💬 Chat │ 📊 Execution Trace │ 🏷️ Governance Tags │ ⏱️ Latency   │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Heterogeneous Metadata Ingestion** — Logical models, physical schemas, and governance tags ingested into separate MongoDB collections
- **Real-Time Consolidation** — Change Streams worker and Atlas Stream Processing pipelines merge disparate sources into a single unified document
- **Voyage AI Embeddings** — Vector embeddings generated and stored natively in Atlas Vector Search
- **MongoDB MCP Server (Model Context Protocol)** — Primary query path for structured queries (`self_query`, `mql` intents). The LLM generates MongoDB `find`/`aggregate` queries following the MCP tool-call pattern, executed directly via PyMongo against the `unified_metadata` collection. Falls back to native retrievers on failure
- **Five Retrieval Strategies** via `langchain-mongodb`:
  - `Vector Search` — Pure cosine similarity via Voyage AI embeddings
  - `MongoDBAtlasHybridSearchRetriever` — Vector + BM25 via Reciprocal Rank Fusion
  - `MongoDBAtlasFullTextSearchRetriever` — BM25 keyword search
  - `MongoDBAtlasSelfQueryRetriever` — LLM-generated metadata filters (MCP-first, then fallback)
  - `Text-to-MQL` — Natural language translated to MongoDB aggregation pipelines (MCP-first, then fallback)
- **Governance as a First-Class Citizen** — PII tags, SIDs, classification, PTB status, and regulatory frameworks surfaced in every response
- **Conversation Memory** — Session history persisted in MongoDB; LLM receives prior turns for follow-up question support
- **Full Execution Trace** — Intent classification, MCP tool calls, latency metrics, and retrieved documents visible in the UI
- **LangSmith Observability** — End-to-end tracing of the RAG pipeline

## Tech Stack

| Component | Technology |
|---|---|
| Database | MongoDB Atlas |
| Real-Time Processing | Change Streams, Atlas Stream Processing |
| MCP Server | MongoDB MCP Server (LLM-driven find/aggregate via MCP pattern) |
| Embeddings | Voyage AI (`voyage-finance-2`) |
| Vector Search | Atlas Vector Search |
| Full-Text Search | Atlas Search (Lucene BM25) |
| LLM | Azure OpenAI (`gpt-4o`) |
| Orchestration | LangGraph + LangChain |
| Observability | LangSmith |
| UI | Streamlit |

## Project Structure

```
omg_metadata_demo/
├── config/
│   └── settings.py                 # Centralised configuration from .env
├── data/
│   └── seed_data.py                # Programmatic metadata generator (1000+ entities per batch)
├── ingestion/
│   ├── ingest.py                   # Seed data → MongoDB source collections
│   └── change_stream_worker.py     # Real-time consolidation + embedding
├── atlas_streams/
│   └── stream_processing.py        # Atlas Stream Processing Admin API scripts
├── embeddings/
│   └── voyage_embeddings.py        # Voyage AI embedding generation
├── search/
│   └── hybrid_search.py            # Hybrid / SelfQuery / FullText retrievers
├── agent/
│   └── rag_agent.py                # LangGraph state machine (MCP-first RAG pipeline)
├── indexes/
│   └── setup_indexes.py            # Vector Search + Atlas Search index creation
├── utils/
│   ├── mongo_client.py             # MongoDB client singleton
│   ├── atlas_auth.py               # Atlas Admin API auth (OAuth2 / Digest auto-detect)
│   └── schema_inspector.py         # Collection schema introspection for LLM context
├── app.py                          # Streamlit chat application
├── requirements.txt
└── env.example                     # Environment variable template
```


## Prerequisites

- Python 3.11+
- MongoDB Atlas cluster (M10+ for Change Streams; M0 works for basic ingestion)
- Atlas API keys with **Project Owner** or **Project Stream Processing Owner** role
- Voyage AI API key
- Azure OpenAI resource with a `gpt-4o` deployment
- LangSmith API key (optional, for tracing)

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/sourav11b/omg_metadata_demo.git
cd omg_metadata_demo
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp env.example .env
# Edit .env with your credentials
```

### 3. Create Search Indexes

```bash
python -m indexes.setup_indexes
```

This creates both the **Vector Search** index (for Voyage AI embeddings) and the **Atlas Search** index (BM25 full-text) on the `unified_metadata` collection.

### 4. Ingest Seed Data

```bash
python -m ingestion.ingest
```

### 5. Consolidate Metadata

**One-shot** (consolidate all entities once):
```bash
python -m ingestion.change_stream_worker --once
```

**Live** (watch for real-time changes):
```bash
python -m ingestion.change_stream_worker
```

### 6. (Optional) Atlas Stream Processing

Set up the equivalent consolidation pipelines running inside Atlas:
```bash
python -m atlas_streams.stream_processing setup
```

Manage processors:
```bash
python -m atlas_streams.stream_processing list
python -m atlas_streams.stream_processing status
```

### 7. Run the Application

```bash
streamlit run app.py
```

## Sample Questions

| Question | Intent | Path |
|---|---|---|
| *What entities are semantically related to fraud scoring?* | `vector` | Vector Search (cosine) |
| *What PII fields exist in the Customer entity?* | `hybrid` | Hybrid (Vector + BM25 RRF) |
| *Describe the metadata for payment transaction processing* | `hybrid` | Hybrid (Vector + BM25 RRF) |
| *Find entities stored in the RISK_DW database* | `fulltext` | Full-Text (BM25 keyword) |
| *Which entities are subject to PCI-DSS?* | `self_query` | MCP Server → Self-Query fallback |
| *Show all Confidential entities in the Payments domain* | `self_query` | MCP Server → Self-Query fallback |
| *Count how many entities exist per domain* | `mql` | MCP Server → Text-to-MQL fallback |
| *List all distinct data stewards and how many entities each manages* | `mql` | MCP Server → Text-to-MQL fallback |

## LangSmith Tracing

With `LANGCHAIN_TRACING_V2=true` in your `.env`, every RAG invocation is traced in LangSmith, including:

- Intent classification latency
- Retrieval strategy and document count
- Generated MQL (for Text-to-MQL queries)
- Answer generation latency
- Full token usage

## License

This project is a demonstration application. See your organisation's licensing guidelines.