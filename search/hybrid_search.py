"""
Hybrid search module for the AMF Semantic Layer.

Wraps three LangChain-MongoDB retrievers so the RAG agent can pick the
best retrieval strategy per query:

  1. MongoDBAtlasHybridSearchRetriever  – combines vector + full-text via RRF
  2. MongoDBAtlasSelfQueryRetriever     – LLM-generated metadata filters
  3. MongoDBAtlasFullTextSearchRetriever – BM25 keyword search
"""

from functools import lru_cache

from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers import (
    MongoDBAtlasFullTextSearchRetriever,
    MongoDBAtlasHybridSearchRetriever,
)
from langchain_mongodb.retrievers.self_querying import MongoDBAtlasSelfQueryRetriever
try:
    from langchain.chains.query_constructor.schema import AttributeInfo
except (ImportError, ModuleNotFoundError):
    try:
        from langchain_community.query_constructors.schema import AttributeInfo  # type: ignore
    except (ImportError, ModuleNotFoundError):
        from dataclasses import dataclass

        @dataclass
        class AttributeInfo:  # type: ignore
            name: str
            description: str
            type: str

from config.settings import (
    COL_UNIFIED_METADATA,
    VECTOR_SEARCH_INDEX,
    FULLTEXT_SEARCH_INDEX,
    EMBEDDING_DIMENSIONS,
)
from embeddings.voyage_embeddings import get_voyage_embeddings
from utils.mongo_client import get_collection


# ── Metadata field descriptions (for SelfQueryRetriever) ──────────────────────

METADATA_FIELD_INFO = [
    AttributeInfo(name="entity_name", description="Name of the data entity (e.g. Customer, Account, Transaction)", type="string"),
    AttributeInfo(name="domain", description="Business domain (e.g. Customer Management, Payments, Marketing)", type="string"),
    AttributeInfo(name="governance.classification", description="Data classification level: Confidential, Restricted, Internal", type="string"),
    AttributeInfo(name="governance.ptb_status", description="Permit To Build status: Approved or Pending", type="string"),
    AttributeInfo(name="governance.data_steward", description="Name of the data steward responsible", type="string"),
    AttributeInfo(name="governance.tags.tag", description="Governance tag type: PII, SID, Financial, Business", type="string"),
    AttributeInfo(name="governance.tags.sensitivity", description="Sensitivity level: High, Medium, Low", type="string"),
    AttributeInfo(name="governance.regulatory_frameworks", description="Applicable regulations: GDPR, CCPA, PCI-DSS, SOX, AML", type="string"),
    AttributeInfo(name="physical.database", description="Physical database name", type="string"),
    AttributeInfo(name="physical.table_name", description="Physical table name", type="string"),
    AttributeInfo(name="physical.storage_format", description="Storage format: Parquet, Delta", type="string"),
]

DOCUMENT_CONTENT_DESCRIPTION = (
    "Unified metadata document describing a data entity in the enterprise, "
    "including its logical model, physical schema, governance tags (PII, SID), "
    "regulatory frameworks, and lineage information."
)


# ── Vector Store (shared by hybrid & self-query retrievers) ───────────────────

@lru_cache(maxsize=1)
def get_vector_store() -> MongoDBAtlasVectorSearch:
    """Return a MongoDBAtlasVectorSearch bound to unified_metadata."""
    return MongoDBAtlasVectorSearch(
        collection=get_collection(COL_UNIFIED_METADATA),
        embedding=get_voyage_embeddings(),
        index_name=VECTOR_SEARCH_INDEX,
        text_key="embedding_text",
        embedding_key="embedding",
    )


# ── 0. Pure Vector Retriever (semantic similarity only) ──────────────────────

def get_vector_retriever(k: int = 5):
    """Return a pure vector search retriever (cosine similarity via Voyage AI)."""
    return get_vector_store().as_retriever(search_kwargs={"k": k})


# ── 1. Hybrid Retriever (Vector + Full-Text via RRF) ─────────────────────────

def get_hybrid_retriever(k: int = 5) -> MongoDBAtlasHybridSearchRetriever:
    """Return a hybrid (vector + BM25) retriever."""
    return MongoDBAtlasHybridSearchRetriever(
        vectorstore=get_vector_store(),
        search_index_name=FULLTEXT_SEARCH_INDEX,
        top_k=k,
        fulltext_penalty=60.0,
        vector_penalty=60.0,
    )


# ── 2. Self-Query Retriever (LLM-generated filters) ──────────────────────────

def get_self_query_retriever(llm) -> MongoDBAtlasSelfQueryRetriever:
    """Return a self-query retriever that translates NL into vector filters."""
    return MongoDBAtlasSelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=get_vector_store(),
        document_contents=DOCUMENT_CONTENT_DESCRIPTION,
        metadata_field_info=METADATA_FIELD_INFO,
        verbose=True,
    )


# ── 3. Full-Text Search Retriever (BM25) ─────────────────────────────────────

def get_fulltext_retriever(k: int = 5) -> MongoDBAtlasFullTextSearchRetriever:
    """Return a pure BM25 full-text search retriever."""
    return MongoDBAtlasFullTextSearchRetriever(
        collection=get_collection(COL_UNIFIED_METADATA),
        search_index_name=FULLTEXT_SEARCH_INDEX,
        search_field="embedding_text",
        top_k=k,
    )
