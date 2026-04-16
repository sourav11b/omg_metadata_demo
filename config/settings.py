"""Centralised configuration loaded from environment / .env file."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── MongoDB Atlas ──────────────────────────────────────────────────────────────
MONGODB_URI: str = os.getenv("MONGODB_URI", "")
MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "amf_metadata_fabric")

# Collection names
COL_LOGICAL_MODELS = "source_logical_models"
COL_PHYSICAL_SCHEMAS = "source_physical_schemas"
COL_GOVERNANCE_TAGS = "source_governance_tags"
COL_UNIFIED_METADATA = "unified_metadata"
COL_CONVERSATION_HISTORY = "conversation_history"
COL_SESSION_MEMORY = "session_memory"

# ── Atlas Admin API ────────────────────────────────────────────────────────────
ATLAS_PUBLIC_KEY: str = os.getenv("ATLAS_PUBLIC_KEY", "")
ATLAS_PRIVATE_KEY: str = os.getenv("ATLAS_PRIVATE_KEY", "")
ATLAS_PROJECT_ID: str = os.getenv("ATLAS_PROJECT_ID", "")
ATLAS_CLUSTER_NAME: str = os.getenv("ATLAS_CLUSTER_NAME", "")
ATLAS_STREAM_INSTANCE: str = os.getenv("ATLAS_STREAM_INSTANCE", "")
ATLAS_API_BASE = "https://cloud.mongodb.com/api/atlas/v2"

# ── Voyage AI (Embeddings) ────────────────────────────────────────────────────
VOYAGE_API_KEY: str = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_MODEL: str = os.getenv("VOYAGE_MODEL", "voyage-3.5")

# ── Azure OpenAI (LLM) ────────────────────────────────────────────────────────
AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# ── LangSmith (Observability) ─────────────────────────────────────────────────
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "amf-agent-demo")

# Search index names
VECTOR_SEARCH_INDEX = "vector_index"
FULLTEXT_SEARCH_INDEX = "fulltext_index"

# Embedding dimensions (Voyage 3.5 = 1024)
EMBEDDING_DIMENSIONS = 1024
