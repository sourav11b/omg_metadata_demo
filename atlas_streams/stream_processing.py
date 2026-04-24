"""
Atlas Stream Processing – Admin API helper scripts.

These functions use the Atlas Admin API v2 to:
  1. Create / list Stream Processing Instances (SPIs)
  2. Create connections (source & sink)
  3. Create & manage Stream Processors (consolidation pipelines)

The pipelines replicate the same consolidation logic as the Change Stream
worker but run entirely inside Atlas Stream Processing infrastructure.

Prerequisites:
  - Atlas API keys with Project Owner or Project Stream Processing Owner role
  - A stream processing instance already provisioned (or use create_instance)

Usage:
    python -m atlas_streams.stream_processing
"""

import json
import pprint
from typing import Any

import requests

from config.settings import (
    ATLAS_PROJECT_ID,
    ATLAS_STREAM_INSTANCE,
    ATLAS_API_BASE,
    ATLAS_CLUSTER_NAME,
    MONGODB_DATABASE,
    COL_LOGICAL_MODELS,
    COL_PHYSICAL_SCHEMAS,
    COL_GOVERNANCE_TAGS,
    COL_UNIFIED_METADATA,
    COL_DLQ,
)
from utils.atlas_auth import get_atlas_auth

# ── Helpers ────────────────────────────────────────────────────────────────────

_AUTH = get_atlas_auth()
_HEADERS = {
    "Accept": "application/vnd.atlas.2024-05-30+json",
    "Content-Type": "application/vnd.atlas.2024-05-30+json",
}


def _url(path: str) -> str:
    return f"{ATLAS_API_BASE}/groups/{ATLAS_PROJECT_ID}{path}"


def _post(path: str, body: dict) -> dict:
    resp = requests.post(_url(path), auth=_AUTH, headers=_HEADERS, json=body)
    if not resp.ok:
        print(f"[ASP] POST {path} → {resp.status_code}: {resp.text}")
    resp.raise_for_status()
    return resp.json()


def _get(path: str) -> dict:
    resp = requests.get(_url(path), auth=_AUTH, headers=_HEADERS)
    if not resp.ok:
        print(f"[ASP] GET {path} → {resp.status_code}: {resp.text}")
    resp.raise_for_status()
    return resp.json()


def _delete(path: str) -> int:
    resp = requests.delete(_url(path), auth=_AUTH, headers=_HEADERS)
    if not resp.ok:
        print(f"[ASP] DELETE {path} → {resp.status_code}: {resp.text}")
    resp.raise_for_status()
    return resp.status_code


# ── 1. Stream Processing Instance ─────────────────────────────────────────────

def create_instance(instance_name: str, provider: str = "AWS",
                    region: str = "US_EAST_1") -> dict:
    """Create a new Atlas Stream Processing Instance."""
    body = {
        "name": instance_name,
        "dataProcessRegion": {"cloudProvider": provider, "region": region},
    }
    result = _post("/streams", body)
    print(f"[ASP] Created instance: {result.get('name')}")
    return result


def list_instances() -> list[dict]:
    """List all Stream Processing Instances in the project."""
    result = _get("/streams")
    instances = result.get("results", [])
    print(f"[ASP] {len(instances)} instance(s) found")
    return instances


# ── 2. Connections (source / sink) ─────────────────────────────────────────────

def create_cluster_connection(connection_name: str) -> dict:
    """Create a connection from the stream instance to the Atlas cluster.

    Skips gracefully if the connection already exists.
    """
    # Check if connection already exists
    try:
        existing = list_connections()
        for c in existing:
            if c.get("name") == connection_name:
                print(f"[ASP] Connection '{connection_name}' already exists, skipping")
                return c
    except Exception:
        pass

    body = {
        "name": connection_name,
        "type": "Cluster",
        "clusterName": ATLAS_CLUSTER_NAME,
        "dbRoleToExecute": {
            "role": "readWriteAnyDatabase",
            "type": "BUILT_IN",
        },
    }
    try:
        result = _post(f"/streams/{ATLAS_STREAM_INSTANCE}/connections", body)
        print(f"[ASP] Connection created: {connection_name}")
        return result
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 409:
            print(f"[ASP] Connection '{connection_name}' already exists (409)")
            return {}
        raise



# ── 3. Stream Processors (consolidation pipelines) ────────────────────────────

def _make_source_stage(collection: str, connection: str = "amf_cluster") -> dict:
    """$source stage reading from an Atlas collection."""
    return {
        "$source": {
            "connectionName": connection,
            "db": MONGODB_DATABASE,
            "coll": collection,
        }
    }


def _make_merge_sink(connection: str = "amf_cluster") -> dict:
    """$merge stage writing/upserting into the unified_metadata collection."""
    return {
        "$merge": {
            "into": {
                "connectionName": connection,
                "db": MONGODB_DATABASE,
                "coll": COL_UNIFIED_METADATA,
            },
            "on": "entity_id",
            "whenMatched": "merge",
            "whenNotMatched": "insert",
        }
    }


def create_logical_model_processor(connection: str = "amf_cluster") -> dict:
    """Processor: watches logical models → projects & merges into unified."""
    pipeline = [
        _make_source_stage(COL_LOGICAL_MODELS, connection),
        {
            "$project": {
                "entity_id": 1,
                "entity_name": 1,
                "domain": 1,
                "description": 1,
                "logical_attributes": "$attributes",
                "relationships": 1,
                "source_systems": {"$literal": ["enterprise_logical_model"]},
                "consolidated_at": "$$NOW",
            }
        },
        _make_merge_sink(connection),
    ]
    return _create_processor("proc_logical_models", pipeline, connection)


def create_physical_schema_processor(connection: str = "amf_cluster") -> dict:
    """Processor: watches physical schemas → nests under 'physical' key.
    Works with dynamically generated entity_ids from seed_data generator."""
    pipeline = [
        _make_source_stage(COL_PHYSICAL_SCHEMAS, connection),
        {
            "$project": {
                "entity_id": 1,
                "physical": {
                    "database": "$database",
                    "schema_name": "$schema_name",
                    "table_name": "$table_name",
                    "columns": "$columns",
                    "storage_format": "$storage_format",
                    "partition_key": "$partition_key",
                    "row_count_approx": "$row_count_approx",
                },
            }
        },
        _make_merge_sink(connection),
    ]
    return _create_processor("proc_physical_schemas", pipeline, connection)


def create_governance_tag_processor(connection: str = "amf_cluster") -> dict:
    """Processor: watches governance tags → nests under 'governance' key."""
    pipeline = [
        _make_source_stage(COL_GOVERNANCE_TAGS, connection),
        {
            "$project": {
                "entity_id": 1,
                "governance": {
                    "classification": "$classification",
                    "data_steward": "$data_steward",
                    "retention_policy": "$retention_policy",
                    "tags": "$tags",
                    "regulatory_frameworks": "$regulatory_frameworks",
                    "ptb_status": "$ptb_status",
                },
            }
        },
        _make_merge_sink(connection),
    ]
    return _create_processor("proc_governance_tags", pipeline, connection)


def _make_dlq_options(connection: str = "amf_cluster") -> dict:
    """Build the Dead Letter Queue (DLQ) options for a stream processor.

    Messages that fail processing (validation errors, $merge conflicts,
    malformed documents, etc.) are written to the DLQ collection instead of
    being silently dropped.  This enables post-hoc inspection and replay.
    """
    return {
        "dlq": {
            "coll": COL_DLQ,
            "connectionName": connection,
            "db": MONGODB_DATABASE,
        }
    }


def _create_processor(name: str, pipeline: list[dict],
                      connection: str = "amf_cluster") -> dict:
    """Create a processor with a Dead Letter Queue, skipping if it already exists."""
    # Check existing
    try:
        existing = list_processors()
        for p in existing:
            if p.get("name") == name:
                print(f"[ASP] Processor '{name}' already exists, skipping")
                return p
    except Exception:
        pass

    body: dict[str, Any] = {
        "name": name,
        "pipeline": pipeline,
        "options": _make_dlq_options(connection),
    }
    try:
        result = _post(f"/streams/{ATLAS_STREAM_INSTANCE}/processor", body)
        print(f"[ASP] Processor created: {name} (DLQ → {MONGODB_DATABASE}.{COL_DLQ})")
        return result
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 409:
            print(f"[ASP] Processor '{name}' already exists (409)")
            return {}
        raise


# ── 4. Management helpers ─────────────────────────────────────────────────────

def start_processor(name: str) -> dict:
    return _post(f"/streams/{ATLAS_STREAM_INSTANCE}/processor/{name}:start", {})


def stop_processor(name: str) -> dict:
    return _post(f"/streams/{ATLAS_STREAM_INSTANCE}/processor/{name}:stop", {})


def list_processors() -> list[dict]:
    result = _get(f"/streams/{ATLAS_STREAM_INSTANCE}/processors")
    procs = result.get("results", [])
    print(f"[ASP] {len(procs)} processor(s)")
    for p in procs:
        print(f"       • {p.get('name')}  [{p.get('state')}]")
    return procs


def delete_processor(name: str) -> None:
    _delete(f"/streams/{ATLAS_STREAM_INSTANCE}/processor/{name}")
    print(f"[ASP] Deleted processor: {name}")


# ── 5. Startup health-check / ensure DLQ ─────────────────────────────────────

# Map processor names → their creation functions
_PROCESSOR_CREATORS = {
    "proc_logical_models": create_logical_model_processor,
    "proc_physical_schemas": create_physical_schema_processor,
    "proc_governance_tags": create_governance_tag_processor,
}


def _processor_has_dlq(proc: dict) -> bool:
    """Check whether a processor already has a DLQ configured."""
    options = proc.get("options", {})
    dlq = options.get("dlq", {})
    return bool(dlq.get("coll"))


def ensure_processors_with_dlq(connection: str = "amf_cluster") -> dict:
    """Startup check: ensure all three processors exist with DLQ configured.

    For each expected processor:
      - If it doesn't exist → create it (with DLQ)
      - If it exists but has no DLQ → stop, delete, and recreate it (with DLQ)
      - If it exists and already has DLQ → leave it alone

    Returns a summary dict: {"created": [...], "recreated": [...], "ok": [...]}
    """
    summary: dict[str, list[str]] = {"created": [], "recreated": [], "ok": []}

    # Ensure prerequisites
    _ensure_unique_index()
    create_cluster_connection(connection)

    # Fetch current processors
    try:
        existing = {p["name"]: p for p in list_processors()}
    except Exception as exc:
        print(f"[ASP] Could not list processors: {exc}")
        return summary

    for name, creator_fn in _PROCESSOR_CREATORS.items():
        if name not in existing:
            # Processor missing → create fresh
            print(f"[ASP] Processor '{name}' not found — creating with DLQ …")
            creator_fn(connection)
            try:
                start_processor(name)
                print(f"[ASP]   ✓ {name} started")
            except Exception as exc:
                print(f"[ASP]   ✗ {name} failed to start: {exc}")
            summary["created"].append(name)

        elif not _processor_has_dlq(existing[name]):
            # Processor exists but no DLQ → stop + delete + recreate
            print(f"[ASP] Processor '{name}' exists but has NO DLQ — recreating …")
            try:
                stop_processor(name)
            except Exception:
                pass  # may already be stopped
            try:
                delete_processor(name)
            except Exception as exc:
                print(f"[ASP]   ✗ Could not delete '{name}': {exc}")
                summary["ok"].append(name)  # leave it alone
                continue
            creator_fn(connection)
            try:
                start_processor(name)
                print(f"[ASP]   ✓ {name} recreated with DLQ and started")
            except Exception as exc:
                print(f"[ASP]   ✗ {name} recreated but failed to start: {exc}")
            summary["recreated"].append(name)

        else:
            print(f"[ASP] Processor '{name}' ✓ (DLQ already configured)")
            summary["ok"].append(name)

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def _ensure_unique_index() -> None:
    """Create a unique index on entity_id in unified_metadata.

    $merge requires a unique index on the 'on' field to guarantee
    that join fields are unique.
    """
    from utils.mongo_client import get_collection
    col = get_collection(COL_UNIFIED_METADATA)
    col.create_index("entity_id", unique=True, name="entity_id_unique")
    print("[ASP] Unique index on unified_metadata.entity_id ensured ✓")


def setup_all(connection: str = "amf_cluster") -> None:
    """End-to-end: create index + connection + all three consolidation processors."""
    print("=" * 60)
    print("AMF-Agent  ·  Atlas Stream Processing Setup")
    print("=" * 60)

    # Step 0: ensure unique index for $merge
    _ensure_unique_index()

    # Step 1: create connection
    create_cluster_connection(connection)

    # Step 2: create processors
    create_logical_model_processor(connection)
    create_physical_schema_processor(connection)
    create_governance_tag_processor(connection)

    # Step 3: start processors
    print("\nStarting processors …")
    for name in ("proc_logical_models", "proc_physical_schemas", "proc_governance_tags"):
        try:
            start_processor(name)
            print(f"  ✓ {name} started")
        except Exception as exc:
            print(f"  ✗ {name} failed to start: {exc}")
    print("=" * 60)
    print("Setup complete.")


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "setup"
    if cmd == "setup":
        setup_all()
    elif cmd == "ensure":
        result = ensure_processors_with_dlq()
        print(f"\n[ASP] Ensure result: {result}")
    elif cmd == "list":
        list_processors()
    elif cmd == "status":
        list_instances()
        list_connections()
        list_processors()
    else:
        print(f"Unknown command: {cmd}  (try: setup | ensure | list | status)")
