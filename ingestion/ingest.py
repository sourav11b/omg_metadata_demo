"""
Ingest heterogeneous metadata from seed data into separate MongoDB collections.

Each source system writes to its own collection, simulating disparate data
producers.  The Change Stream worker (change_stream_worker.py) or Atlas Stream
Processing pipeline then consolidates them into unified_metadata.

Usage:
    python -m ingestion.ingest
"""

import datetime
from pymongo import ReplaceOne

from config.settings import (
    COL_LOGICAL_MODELS,
    COL_PHYSICAL_SCHEMAS,
    COL_GOVERNANCE_TAGS,
)
from data.seed_data import LOGICAL_MODELS, PHYSICAL_SCHEMAS, GOVERNANCE_TAGS
from utils.mongo_client import get_collection


def _stamp(docs: list[dict]) -> list[dict]:
    """Add ingestion timestamp to each document."""
    now = datetime.datetime.now(datetime.timezone.utc)
    for d in docs:
        d["ingested_at"] = now
    return docs


def ingest_logical_models() -> int:
    """Upsert logical model documents."""
    col = get_collection(COL_LOGICAL_MODELS)
    ops = [
        ReplaceOne({"entity_id": d["entity_id"]}, d, upsert=True)
        for d in _stamp(LOGICAL_MODELS)
    ]
    result = col.bulk_write(ops)
    count = result.upserted_count + result.modified_count
    print(f"[ingest] logical_models: {count} docs upserted/updated")
    return count


def ingest_physical_schemas() -> int:
    """Upsert physical schema documents."""
    col = get_collection(COL_PHYSICAL_SCHEMAS)
    ops = [
        ReplaceOne({"entity_id": d["entity_id"]}, d, upsert=True)
        for d in _stamp(PHYSICAL_SCHEMAS)
    ]
    result = col.bulk_write(ops)
    count = result.upserted_count + result.modified_count
    print(f"[ingest] physical_schemas: {count} docs upserted/updated")
    return count


def ingest_governance_tags() -> int:
    """Upsert governance tag documents."""
    col = get_collection(COL_GOVERNANCE_TAGS)
    ops = [
        ReplaceOne({"entity_id": d["entity_id"]}, d, upsert=True)
        for d in _stamp(GOVERNANCE_TAGS)
    ]
    result = col.bulk_write(ops)
    count = result.upserted_count + result.modified_count
    print(f"[ingest] governance_tags: {count} docs upserted/updated")
    return count


def run_full_ingestion() -> None:
    """Run the complete ingestion pipeline across all three sources."""
    print("=" * 60)
    print("AMF-Agent  ·  Metadata Ingestion Pipeline")
    print("=" * 60)
    ingest_logical_models()
    ingest_physical_schemas()
    ingest_governance_tags()
    print("=" * 60)
    print("Ingestion complete.")


if __name__ == "__main__":
    run_full_ingestion()
