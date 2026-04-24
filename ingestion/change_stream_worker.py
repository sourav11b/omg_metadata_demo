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

import traceback

from config.settings import (
    COL_LOGICAL_MODELS,
    COL_PHYSICAL_SCHEMAS,
    COL_GOVERNANCE_TAGS,
    COL_UNIFIED_METADATA,
    COL_DLQ,
)
from embeddings.voyage_embeddings import generate_embedding
from utils.mongo_client import get_collection, get_database


def _write_to_dlq(entity_id: str, source: str, error: Exception,
                  document: dict | None = None) -> None:
    """Write a failed consolidation event to the dead letter queue collection.

    Each DLQ document captures:
      - entity_id & source collection that triggered the failure
      - error class, message, and full traceback
      - the original document (if available) for replay / inspection
      - timestamp for SLA tracking
    """
    try:
        dlq_doc = {
            "entity_id": entity_id,
            "source_collection": source,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "original_document": document,
            "status": "pending",          # pending | retried | resolved
            "retry_count": 0,
            "created_at": datetime.datetime.now(datetime.timezone.utc),
        }
        get_collection(COL_DLQ).insert_one(dlq_doc)
        print(f"[DLQ] Written failed event for {entity_id} ({source})")
    except Exception as dlq_exc:
        # Last resort — don't let DLQ failures crash the worker
        print(f"[DLQ] CRITICAL: Could not write to DLQ: {dlq_exc}")


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


# ── Consolidation stats (shared across threads / UI reads) ────────────────────

_stats_lock = threading.Lock()
_consolidation_stats = {
    "total_consolidated": 0,
    "pending": 0,
    "last_consolidated_at": None,
    "running": False,
    "errors": 0,
}


def get_consolidation_stats() -> dict:
    with _stats_lock:
        # Recompute pending
        source_ids: set[str] = set()
        for cn in (COL_LOGICAL_MODELS, COL_PHYSICAL_SCHEMAS, COL_GOVERNANCE_TAGS):
            for d in get_collection(cn).find({}, {"entity_id": 1}):
                source_ids.add(d["entity_id"])
        unified_ids: set[str] = set()
        for d in get_collection(COL_UNIFIED_METADATA).find({}, {"entity_id": 1}):
            unified_ids.add(d["entity_id"])
        _consolidation_stats["pending"] = len(source_ids - unified_ids)
        _consolidation_stats["total_consolidated"] = len(unified_ids)
        return dict(_consolidation_stats)


def _increment_stat(key: str, delta: int = 1) -> None:
    with _stats_lock:
        _consolidation_stats[key] = _consolidation_stats.get(key, 0) + delta
        _consolidation_stats["last_consolidated_at"] = (
            datetime.datetime.now(datetime.timezone.utc).isoformat()
        )


def consolidate_all() -> dict:
    """One-shot: consolidate every entity not yet in unified_metadata."""
    source_ids: set[str] = set()
    for cn in (COL_LOGICAL_MODELS, COL_PHYSICAL_SCHEMAS, COL_GOVERNANCE_TAGS):
        for d in get_collection(cn).find({}, {"entity_id": 1}):
            source_ids.add(d["entity_id"])

    unified_ids: set[str] = set()
    for d in get_collection(COL_UNIFIED_METADATA).find({}, {"entity_id": 1}):
        unified_ids.add(d["entity_id"])

    pending = source_ids - unified_ids
    # Also re-consolidate a sample of existing ones (updates)
    existing_sample = set()
    if unified_ids:
        import random
        existing_sample = set(random.sample(
            list(unified_ids), k=min(50, len(unified_ids))
        ))

    to_process = pending | existing_sample
    print(f"[consolidate_all] {len(pending)} new + {len(existing_sample)} updates = {len(to_process)} to process")

    done, errors = 0, 0
    for eid in to_process:
        try:
            consolidate_entity(eid)
            done += 1
            _increment_stat("total_consolidated", 0)  # updates timestamp
        except Exception as exc:
            errors += 1
            _write_to_dlq(eid, "consolidate_all", exc)
            print(f"[consolidate] Error on {eid}: {exc}")

    with _stats_lock:
        _consolidation_stats["errors"] += errors

    return {"processed": done, "errors": errors, "pending_new": len(pending)}


# ── Background continuous consolidation ──────────────────────────────────────

_bg_thread: threading.Thread | None = None
_bg_stop_event = threading.Event()


def _background_consolidation_loop() -> None:
    """Continuously consolidate pending entities every 5 seconds."""
    with _stats_lock:
        _consolidation_stats["running"] = True
    while not _bg_stop_event.is_set():
        try:
            source_ids: set[str] = set()
            for cn in (COL_LOGICAL_MODELS, COL_PHYSICAL_SCHEMAS, COL_GOVERNANCE_TAGS):
                for d in get_collection(cn).find({}, {"entity_id": 1}):
                    source_ids.add(d["entity_id"])

            unified_ids: set[str] = set()
            for d in get_collection(COL_UNIFIED_METADATA).find({}, {"entity_id": 1}):
                unified_ids.add(d["entity_id"])

            pending = source_ids - unified_ids
            if pending:
                print(f"[bg-consolidate] Processing {len(pending)} pending entities …")
                for eid in pending:
                    try:
                        consolidate_entity(eid)
                        _increment_stat("total_consolidated", 0)
                    except Exception as exc:
                        _increment_stat("errors")
                        _write_to_dlq(eid, "bg_consolidate", exc)
                        print(f"[bg-consolidate] Error on {eid}: {exc}")
        except Exception as exc:
            print(f"[bg-consolidate] Loop error: {exc}")

        _bg_stop_event.wait(timeout=5)

    with _stats_lock:
        _consolidation_stats["running"] = False
    print("[bg-consolidate] Background consolidation stopped")


def start_background_consolidation() -> None:
    """Start background consolidation thread (if not already running)."""
    global _bg_thread
    if _bg_thread is not None and _bg_thread.is_alive():
        return  # already running
    _bg_stop_event.clear()
    _bg_thread = threading.Thread(
        target=_background_consolidation_loop, daemon=True, name="bg-consolidate",
    )
    _bg_thread.start()
    print("[bg-consolidate] Background consolidation started")


def stop_background_consolidation() -> None:
    """Signal the background consolidation thread to stop."""
    _bg_stop_event.set()
    with _stats_lock:
        _consolidation_stats["running"] = False
    print("[bg-consolidate] Stop signal sent")


def is_background_running() -> bool:
    return _bg_thread is not None and _bg_thread.is_alive() and not _bg_stop_event.is_set()


# ── Change Stream workers (for Atlas M10+) ───────────────────────────────────

def _watch_collection(col_name: str) -> None:
    col = get_collection(col_name)
    pipeline = [{"$match": {"operationType": {"$in": ["insert", "replace", "update"]}}}]
    print(f"[watch] Listening on {col_name} …")
    try:
        with col.watch(pipeline, full_document="updateLookup") as stream:
            for change in stream:
                entity_id = change.get("fullDocument", {}).get("entity_id")
                if entity_id:
                    try:
                        consolidate_entity(entity_id)
                        _increment_stat("total_consolidated", 0)
                    except Exception as exc:
                        _increment_stat("errors")
                        _write_to_dlq(
                            entity_id, col_name, exc,
                            document=change.get("fullDocument"),
                        )
                        print(f"[watch] Error consolidating {entity_id}: {exc}")
    except Exception as exc:
        print(f"[watch] Stream error on {col_name}: {exc}")


def run_change_stream_workers() -> None:
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
    elif "--bg" in sys.argv:
        start_background_consolidation()
        while True:
            time.sleep(5)
    else:
        run_change_stream_workers()
