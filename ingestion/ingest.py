"""
Ingest heterogeneous metadata from generated seed data into separate MongoDB
collections.  Each run:
  1. INSERTs *new_count* brand-new entities (default 1000)
  2. UPDATEs *update_count* randomly-chosen existing entities (default 50)

This ensures collection counts grow on every ingestion click.

Usage:
    python -m ingestion.ingest            # 1000 new + 50 updates
    python -m ingestion.ingest --count 500
"""

import datetime
from pymongo import InsertOne, ReplaceOne

from config.settings import (
    COL_LOGICAL_MODELS,
    COL_PHYSICAL_SCHEMAS,
    COL_GOVERNANCE_TAGS,
)
from data.seed_data import generate_batch, generate_updates
from utils.mongo_client import get_collection


def _stamp(docs: list[dict]) -> list[dict]:
    """Add ingestion timestamp to each document."""
    now = datetime.datetime.now(datetime.timezone.utc)
    for d in docs:
        d["ingested_at"] = now
    return docs


def _get_existing_ids() -> list[str]:
    """Return all entity_ids currently in the logical models collection."""
    col = get_collection(COL_LOGICAL_MODELS)
    return [d["entity_id"] for d in col.find({}, {"entity_id": 1, "_id": 0})]


def _bulk_insert(col_name: str, docs: list[dict]) -> int:
    """Insert new documents."""
    if not docs:
        return 0
    col = get_collection(col_name)
    result = col.insert_many(_stamp(docs))
    return len(result.inserted_ids)


def _bulk_upsert(col_name: str, docs: list[dict]) -> int:
    """Upsert (update-or-insert) documents by entity_id."""
    if not docs:
        return 0
    col = get_collection(col_name)
    ops = [ReplaceOne({"entity_id": d["entity_id"]}, d, upsert=True)
           for d in _stamp(docs)]
    result = col.bulk_write(ops)
    return result.upserted_count + result.modified_count


def run_ingestion(new_count: int = 1000, update_count: int = 50) -> dict:
    """Run the complete ingestion pipeline.

    Returns a stats dict with counts for inserts and updates per collection.
    """
    stats: dict = {"new": {}, "updated": {}}

    # ── 1. Generate and INSERT new entities ────────────────────────────────
    lm_new, ps_new, gt_new = generate_batch(new_count)

    stats["new"]["logical_models"] = _bulk_insert(COL_LOGICAL_MODELS, lm_new)
    stats["new"]["physical_schemas"] = _bulk_insert(COL_PHYSICAL_SCHEMAS, ps_new)
    stats["new"]["governance_tags"] = _bulk_insert(COL_GOVERNANCE_TAGS, gt_new)

    print(f"[ingest] Inserted {new_count} new entities across all sources")

    # ── 2. Generate and UPDATE existing entities ───────────────────────────
    existing_ids = _get_existing_ids()
    lm_upd, ps_upd, gt_upd = generate_updates(existing_ids, update_count)

    stats["updated"]["logical_models"] = _bulk_upsert(COL_LOGICAL_MODELS, lm_upd)
    stats["updated"]["physical_schemas"] = _bulk_upsert(COL_PHYSICAL_SCHEMAS, ps_upd)
    stats["updated"]["governance_tags"] = _bulk_upsert(COL_GOVERNANCE_TAGS, gt_upd)

    print(f"[ingest] Updated {update_count} existing entities across all sources")

    return stats


# Legacy aliases for backward compatibility
def ingest_logical_models() -> int:
    stats = run_ingestion(new_count=1000, update_count=50)
    return stats["new"]["logical_models"]

def ingest_physical_schemas() -> int:
    return 0  # handled by run_ingestion

def ingest_governance_tags() -> int:
    return 0  # handled by run_ingestion


def run_full_ingestion() -> None:
    print("=" * 60)
    print("AMF-Agent  ·  Metadata Ingestion Pipeline")
    print("=" * 60)
    stats = run_ingestion()
    print(f"Stats: {stats}")
    print("=" * 60)
    print("Ingestion complete.")


if __name__ == "__main__":
    import sys
    count = 1000
    for arg in sys.argv[1:]:
        if arg.startswith("--count"):
            count = int(sys.argv[sys.argv.index(arg) + 1])
    run_ingestion(new_count=count)
