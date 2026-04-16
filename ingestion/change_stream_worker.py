"""
Real-time Change Stream worker that consolidates heterogeneous metadata
into a single unified document per entity, then generates a Voyage AI
vector embedding and writes it to the unified_metadata collection.

The worker watches *all three* source collections for insert / replace /
update operations, merges attributes by entity_id, and upserts the
consolidated document.

Usage:
    python -m ingestion.change_stream_worker
"""

import datetime
import threading
import time

from config.settings import (
    COL_LOGICAL_MODELS,
    COL_PHYSICAL_SCHEMAS,
    COL_GOVERNANCE_TAGS,
    COL_UNIFIED_METADATA,
)
from embeddings.voyage_embeddings import generate_embedding
from utils.mongo_client import get_collection, get_database


def _build_text_for_embedding(doc: dict) -> str:
    """Flatten the unified document into a single text block for embedding."""
    parts: list[str] = []
    parts.append(f"Entity: {doc.get('entity_name', '')} ({doc.get('entity_id', '')})")
    parts.append(f"Domain: {doc.get('domain', '')}")
    parts.append(f"Description: {doc.get('description', '')}")

    # Logical attributes
    for attr in doc.get("logical_attributes", []):
        parts.append(f"Attribute {attr['name']} ({attr['type']}): {attr['description']}")

    # Physical columns
    if doc.get("physical"):
        p = doc["physical"]
        parts.append(f"Database: {p.get('database')}.{p.get('schema_name')}.{p.get('table_name')}")
        for col in p.get("columns", []):
            parts.append(f"Column {col['name']} {col['data_type']}")

    # Governance
    if doc.get("governance"):
        g = doc["governance"]
        parts.append(f"Classification: {g.get('classification')}")
        parts.append(f"PTB Status: {g.get('ptb_status')}")
        parts.append(f"Data Steward: {g.get('data_steward')}")
        for tag in g.get("tags", []):
            parts.append(f"Tag: {tag['field']} = {tag['tag']} (sensitivity: {tag['sensitivity']})")
        parts.append(f"Regulatory: {', '.join(g.get('regulatory_frameworks', []))}")

    return "\n".join(parts)


def consolidate_entity(entity_id: str) -> None:
    """Merge data from all three sources for *entity_id* and upsert to unified."""
    logical = get_collection(COL_LOGICAL_MODELS).find_one({"entity_id": entity_id})
    physical = get_collection(COL_PHYSICAL_SCHEMAS).find_one({"entity_id": entity_id})
    governance = get_collection(COL_GOVERNANCE_TAGS).find_one({"entity_id": entity_id})

    if not logical:
        return  # nothing to consolidate yet

    unified: dict = {
        "entity_id": entity_id,
        "entity_name": logical.get("entity_name"),
        "domain": logical.get("domain"),
        "description": logical.get("description"),
        "logical_attributes": logical.get("attributes", []),
        "relationships": logical.get("relationships", []),
        "source_systems": ["enterprise_logical_model"],
        "consolidated_at": datetime.datetime.now(datetime.timezone.utc),
    }

    if physical:
        unified["physical"] = {
            "database": physical.get("database"),
            "schema_name": physical.get("schema_name"),
            "table_name": physical.get("table_name"),
            "columns": physical.get("columns", []),
            "storage_format": physical.get("storage_format"),
            "partition_key": physical.get("partition_key"),
            "row_count_approx": physical.get("row_count_approx"),
        }
        unified["source_systems"].append("physical_schema_catalog")

    if governance:
        unified["governance"] = {
            "classification": governance.get("classification"),
            "data_steward": governance.get("data_steward"),
            "retention_policy": governance.get("retention_policy"),
            "tags": governance.get("tags", []),
            "regulatory_frameworks": governance.get("regulatory_frameworks", []),
            "ptb_status": governance.get("ptb_status"),
        }
        unified["source_systems"].append("governance_catalog")

    # Generate embedding
    text = _build_text_for_embedding(unified)
    unified["embedding_text"] = text
    unified["embedding"] = generate_embedding(text)

    get_collection(COL_UNIFIED_METADATA).replace_one(
        {"entity_id": entity_id}, unified, upsert=True
    )
    print(f"[consolidate] {entity_id} → unified_metadata  ✓")


def consolidate_all() -> None:
    """One-shot consolidation of every entity currently in the source collections."""
    entity_ids: set[str] = set()
    for col_name in (COL_LOGICAL_MODELS, COL_PHYSICAL_SCHEMAS, COL_GOVERNANCE_TAGS):
        for doc in get_collection(col_name).find({}, {"entity_id": 1}):
            entity_ids.add(doc["entity_id"])
    print(f"[consolidate_all] Found {len(entity_ids)} entities to consolidate …")
    for eid in sorted(entity_ids):
        consolidate_entity(eid)
    print("[consolidate_all] Done.")


def _watch_collection(col_name: str) -> None:
    """Watch a single collection's change stream and trigger consolidation."""
    col = get_collection(col_name)
    pipeline = [{"$match": {"operationType": {"$in": ["insert", "replace", "update"]}}}]
    print(f"[watch] Listening on {col_name} …")
    try:
        with col.watch(pipeline, full_document="updateLookup") as stream:
            for change in stream:
                entity_id = change.get("fullDocument", {}).get("entity_id")
                if entity_id:
                    print(f"[watch] Change detected in {col_name} for {entity_id}")
                    consolidate_entity(entity_id)
    except Exception as exc:
        print(f"[watch] Error on {col_name}: {exc}")


def run_change_stream_workers() -> None:
    """Spawn a thread per source collection to watch for changes."""
    print("=" * 60)
    print("AMF-Agent  ·  Change Stream Consolidation Workers")
    print("=" * 60)
    collections = [COL_LOGICAL_MODELS, COL_PHYSICAL_SCHEMAS, COL_GOVERNANCE_TAGS]
    threads = []
    for col_name in collections:
        t = threading.Thread(target=_watch_collection, args=(col_name,), daemon=True)
        t.start()
        threads.append(t)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[watch] Shutting down …")


if __name__ == "__main__":
    import sys

    if "--once" in sys.argv:
        consolidate_all()
    else:
        run_change_stream_workers()
