"""
Create Atlas Search and Vector Search indexes on the unified_metadata
collection using the Atlas Admin API.

Usage:
    python -m indexes.setup_indexes
"""

import json
import time

import requests

from config.settings import (
    ATLAS_PROJECT_ID,
    ATLAS_CLUSTER_NAME,
    ATLAS_API_BASE,
    MONGODB_DATABASE,
    COL_UNIFIED_METADATA,
    VECTOR_SEARCH_INDEX,
    FULLTEXT_SEARCH_INDEX,
    EMBEDDING_DIMENSIONS,
    AUTO_EMBEDDING_MODEL,
)
from utils.atlas_auth import get_atlas_auth

_AUTH = get_atlas_auth()
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
                "type": "autoEmbed",
                "path": "embedding_text",
                "model": AUTO_EMBEDDING_MODEL,
                "numDimensions": EMBEDDING_DIMENSIONS,
                "similarity": "cosine",
                "modality": "text",
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
    print(f"[index] Creating vector index with body:\n{json.dumps(body, indent=2)}")
    resp = requests.post(
        _url("/search/indexes"), auth=_AUTH, headers=_HEADERS, json=body,
    )
    if not resp.ok:
        print(f"[index] ERROR {resp.status_code}: {resp.text}")
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


def delete_index_by_name(index_name: str) -> bool:
    """Delete a search index by name. Returns True if deleted, False if not found."""
    indexes = list_indexes()
    target = None
    for idx in indexes:
        if idx.get("name") == index_name:
            target = idx
            break

    if not target:
        print(f"[index] Index '{index_name}' not found — nothing to delete")
        return False

    index_id = target.get("indexID") or target.get("id")
    if not index_id:
        print(f"[index] Could not determine ID for index '{index_name}'")
        return False

    resp = requests.delete(
        _url(f"/search/indexes/{index_id}"),
        auth=_AUTH, headers=_HEADERS,
    )
    if resp.status_code == 204 or resp.status_code == 202:
        print(f"[index] Index '{index_name}' (id={index_id}) deleted ✓")
        return True
    resp.raise_for_status()
    return True


def delete_all_indexes() -> None:
    """Delete all search indexes on the unified_metadata collection."""
    print("[index] Deleting existing indexes …")
    for name in (VECTOR_SEARCH_INDEX, FULLTEXT_SEARCH_INDEX):
        delete_index_by_name(name)


def recreate_all_indexes(wait_seconds: int = 10) -> None:
    """Delete existing indexes, wait, then recreate them."""
    print("=" * 60)
    print("AMF-Agent  ·  Atlas Search Index Recreate")
    print("=" * 60)
    delete_all_indexes()
    print(f"\n⏳ Waiting {wait_seconds}s for deletions to propagate …")
    time.sleep(wait_seconds)
    print()
    create_vector_search_index()
    create_fulltext_search_index()
    print("\nVerifying …")
    list_indexes()
    print("=" * 60)


def setup_all_indexes() -> None:
    """Create both search indexes (without deleting first)."""
    print("=" * 60)
    print("AMF-Agent  ·  Atlas Search Index Setup")
    print("=" * 60)
    create_vector_search_index()
    create_fulltext_search_index()
    print("\nVerifying …")
    list_indexes()
    print("=" * 60)


if __name__ == "__main__":
    import sys
    if "--recreate" in sys.argv:
        recreate_all_indexes()
    elif "--delete" in sys.argv:
        delete_all_indexes()
    elif "--list" in sys.argv:
        list_indexes()
    else:
        setup_all_indexes()
