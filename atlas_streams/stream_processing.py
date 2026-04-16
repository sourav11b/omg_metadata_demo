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
    resp.raise_for_status()
    return resp.json()


def _get(path: str) -> dict:
    resp = requests.get(_url(path), auth=_AUTH, headers=_HEADERS)
    resp.raise_for_status()
    return resp.json()


def _delete(path: str) -> int:
    resp = requests.delete(_url(path), auth=_AUTH, headers=_HEADERS)
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
    """Create a connection from the stream instance to the Atlas cluster."""
    body = {
        "name": connection_name,
        "type": "Cluster",
        "clusterName": ATLAS_CLUSTER_NAME,
    }
    result = _post(f"/streams/{ATLAS_STREAM_INSTANCE}/connections", body)
    print(f"[ASP] Connection created: {connection_name}")
    return result



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
    return _create_processor("proc_logical_models", pipeline)


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
    return _create_processor("proc_physical_schemas", pipeline)


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
    return _create_processor("proc_governance_tags", pipeline)


def _create_processor(name: str, pipeline: list[dict]) -> dict:
    body: dict[str, Any] = {"name": name, "pipeline": pipeline}
    result = _post(f"/streams/{ATLAS_STREAM_INSTANCE}/processor", body)
    print(f"[ASP] Processor created: {name}")
    return result


# ── 4. Management helpers ─────────────────────────────────────────────────────

def start_processor(name: str) -> dict:
    return _post(f"/streams/{ATLAS_STREAM_INSTANCE}/processor/{name}:start", {})


def stop_processor(name: str) -> dict:
    return _post(f"/streams/{ATLAS_STREAM_INSTANCE}/processor/{name}:stop", {})


def list_processors() -> list[dict]:
    result = _get(f"/streams/{ATLAS_STREAM_INSTANCE}/processor")
    procs = result.get("results", [])
    print(f"[ASP] {len(procs)} processor(s)")
    for p in procs:
        print(f"       • {p.get('name')}  [{p.get('state')}]")
    return procs


def delete_processor(name: str) -> None:
    _delete(f"/streams/{ATLAS_STREAM_INSTANCE}/processor/{name}")
    print(f"[ASP] Deleted processor: {name}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def setup_all(connection: str = "amf_cluster") -> None:
    """End-to-end: create connection + all three consolidation processors."""
    print("=" * 60)
    print("AMF-Agent  ·  Atlas Stream Processing Setup")
    print("=" * 60)
    create_cluster_connection(connection)
    create_logical_model_processor(connection)
    create_physical_schema_processor(connection)
    create_governance_tag_processor(connection)
    print("\nStarting processors …")
    for name in ("proc_logical_models", "proc_physical_schemas", "proc_governance_tags"):
        start_processor(name)
    print("=" * 60)
    print("All processors running.")


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "setup"
    if cmd == "setup":
        setup_all()
    elif cmd == "list":
        list_processors()
    elif cmd == "status":
        list_instances()
        list_connections()
        list_processors()
    else:
        print(f"Unknown command: {cmd}  (try: setup | list | status)")
