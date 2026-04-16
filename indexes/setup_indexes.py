"""
Create Atlas Search and Vector Search indexes on the unified_metadata
collection using the Atlas Admin API.

Usage:
    python -m indexes.setup_indexes
"""

import json
import requests
from requests.auth import HTTPDigestAuth

from config.settings import (
    ATLAS_PUBLIC_KEY,
    ATLAS_PRIVATE_KEY,
    ATLAS_PROJECT_ID,
    ATLAS_CLUSTER_NAME,
    ATLAS_API_BASE,
    MONGODB_DATABASE,
    COL_UNIFIED_METADATA,
    VECTOR_SEARCH_INDEX,
    FULLTEXT_SEARCH_INDEX,
    EMBEDDING_DIMENSIONS,
)

_AUTH = HTTPDigestAuth(ATLAS_PUBLIC_KEY, ATLAS_PRIVATE_KEY)
_HEADERS = {
    "Accept": "application/vnd.atlas.2024-05-30+json",
    "Content-Type": "application/vnd.atlas.2024-05-30+json",
}


def _url(path: str) -> str:
    return f"{ATLAS_API_BASE}/groups/{ATLAS_PROJECT_ID}/clusters/{ATLAS_CLUSTER_NAME}{path}"


# ── Vector Search Index ───────────────────────────────────────────────────────

VECTOR_INDEX_DEF = {
    "name": VECTOR_SEARCH_INDEX,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": EMBEDDING_DIMENSIONS,
                "similarity": "cosine",
            },
            {"type": "filter", "path": "entity_name"},
            {"type": "filter", "path": "domain"},
            {"type": "filter", "path": "governance.classification"},
            {"type": "filter", "path": "governance.ptb_status"},
            {"type": "filter", "path": "governance.tags.tag"},
            {"type": "filter", "path": "governance.tags.sensitivity"},
            {"type": "filter", "path": "governance.regulatory_frameworks"},
        ]
    },
}


# ── Full-Text Search Index ────────────────────────────────────────────────────

FULLTEXT_INDEX_DEF = {
    "name": FULLTEXT_SEARCH_INDEX,
    "type": "search",
    "definition": {
        "mappings": {
            "dynamic": False,
            "fields": {
                "embedding_text": {"type": "string", "analyzer": "lucene.standard"},
                "entity_name": {"type": "string", "analyzer": "lucene.standard"},
                "domain": {"type": "string", "analyzer": "lucene.standard"},
                "description": {"type": "string", "analyzer": "lucene.standard"},
                "governance": {
                    "type": "document",
                    "fields": {
                        "classification": {"type": "string"},
                        "ptb_status": {"type": "string"},
                        "data_steward": {"type": "string", "analyzer": "lucene.standard"},
                        "tags": {
                            "type": "document",
                            "fields": {
                                "tag": {"type": "stringFacet"},
                                "field": {"type": "string"},
                                "sensitivity": {"type": "stringFacet"},
                            },
                        },
                    },
                },
            },
        }
    },
}


def create_vector_search_index() -> dict:
    """Create the vector search index via Atlas Admin API."""
    body = {
        "collectionName": COL_UNIFIED_METADATA,
        "database": MONGODB_DATABASE,
        **VECTOR_INDEX_DEF,
    }
    resp = requests.post(
        _url("/search/indexes"), auth=_AUTH, headers=_HEADERS, json=body,
    )
    resp.raise_for_status()
    print(f"[index] Vector search index '{VECTOR_SEARCH_INDEX}' created ✓")
    return resp.json()


def create_fulltext_search_index() -> dict:
    """Create the full-text (Atlas Search) index via Atlas Admin API."""
    body = {
        "collectionName": COL_UNIFIED_METADATA,
        "database": MONGODB_DATABASE,
        **FULLTEXT_INDEX_DEF,
    }
    resp = requests.post(
        _url("/search/indexes"), auth=_AUTH, headers=_HEADERS, json=body,
    )
    resp.raise_for_status()
    print(f"[index] Full-text search index '{FULLTEXT_SEARCH_INDEX}' created ✓")
    return resp.json()


def list_indexes() -> list[dict]:
    """List all search indexes on the cluster."""
    resp = requests.get(
        _url(f"/search/indexes/{MONGODB_DATABASE}/{COL_UNIFIED_METADATA}"),
        auth=_AUTH, headers=_HEADERS,
    )
    resp.raise_for_status()
    indexes = resp.json()
    for idx in indexes:
        print(f"  • {idx.get('name')}  [{idx.get('type')}]  status={idx.get('status')}")
    return indexes


def setup_all_indexes() -> None:
    """Create both search indexes."""
    print("=" * 60)
    print("AMF-Agent  ·  Atlas Search Index Setup")
    print("=" * 60)
    create_vector_search_index()
    create_fulltext_search_index()
    print("\nVerifying …")
    list_indexes()
    print("=" * 60)


if __name__ == "__main__":
    setup_all_indexes()
