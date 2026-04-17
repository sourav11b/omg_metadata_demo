"""
Schema introspection for MongoDB collections.

Examines sample documents to build a structural summary of all fields,
types, nested objects, and array shapes.  This schema context is injected
into session memory so the LLM can generate accurate MQL queries.
"""

import json
from typing import Any

from config.settings import COL_UNIFIED_METADATA, MONGODB_DATABASE
from utils.mongo_client import get_collection


def _infer_type(value: Any) -> str:
    """Return a human-readable type string for a BSON value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "double"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        if value:
            inner = _infer_type(value[0])
            return f"array<{inner}>"
        return "array<unknown>"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _build_schema(docs: list[dict], prefix: str = "") -> dict:
    """Merge multiple docs into a single schema map: field_path → type info."""
    schema: dict[str, dict] = {}
    for doc in docs:
        for key, val in doc.items():
            if key == "_id":
                continue
            path = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            t = _infer_type(val)
            if path not in schema:
                schema[path] = {"types": set(), "sample": None, "count": 0}
            schema[path]["types"].add(t)
            schema[path]["count"] += 1
            if schema[path]["sample"] is None and val is not None:
                if isinstance(val, (str, int, float, bool)):
                    schema[path]["sample"] = val
                elif isinstance(val, list) and val and isinstance(val[0], (str, int, float)):
                    schema[path]["sample"] = val[:3]

            # Recurse into nested objects
            if isinstance(val, dict):
                nested = _build_schema([val], prefix=path)
                for nk, nv in nested.items():
                    if nk not in schema:
                        schema[nk] = nv
                    else:
                        schema[nk]["types"] |= nv["types"]
                        schema[nk]["count"] += nv["count"]

            # Recurse into arrays of objects
            if isinstance(val, list) and val and isinstance(val[0], dict):
                nested = _build_schema(val, prefix=f"{path}[]")
                for nk, nv in nested.items():
                    if nk not in schema:
                        schema[nk] = nv
                    else:
                        schema[nk]["types"] |= nv["types"]
                        schema[nk]["count"] += nv["count"]

    return schema


def inspect_collection(collection_name: str = COL_UNIFIED_METADATA,
                       sample_size: int = 20) -> str:
    """Inspect a collection and return a structured schema summary as text.

    Samples up to `sample_size` documents and infers the document structure.
    Returns a formatted string suitable for LLM context injection.
    """
    col = get_collection(collection_name)
    total = col.estimated_document_count()
    # Sample documents (without embedding vectors to save space)
    docs = list(col.find({}, {"embedding": 0}).limit(sample_size))

    if not docs:
        return f"Collection '{collection_name}' in database '{MONGODB_DATABASE}' is empty."

    schema = _build_schema(docs)

    lines = [
        f"=== Schema for '{MONGODB_DATABASE}.{collection_name}' ===",
        f"Estimated document count: {total}",
        f"Sampled: {len(docs)} documents",
        "",
        "Fields:",
    ]

    for path in sorted(schema.keys()):
        info = schema[path]
        types_str = " | ".join(sorted(info["types"]))
        sample = ""
        if info["sample"] is not None:
            s = json.dumps(info["sample"], default=str)
            if len(s) > 80:
                s = s[:77] + "..."
            sample = f"  (e.g. {s})"
        lines.append(f"  {path}: {types_str}{sample}")

    lines.append("")
    lines.append("Sample entity_ids: " + ", ".join(
        d.get("entity_id", "?") for d in docs[:5]
    ))

    # Extract unique values for key filter fields
    for field in ["domain", "governance.classification", "governance.ptb_status"]:
        vals = set()
        for d in docs:
            parts = field.split(".")
            v = d
            for p in parts:
                if isinstance(v, dict):
                    v = v.get(p)
                else:
                    v = None
                    break
            if v:
                vals.add(str(v))
        if vals:
            lines.append(f"Distinct '{field}' values: {', '.join(sorted(vals))}")

    return "\n".join(lines)
