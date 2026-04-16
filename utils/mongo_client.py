"""Singleton MongoDB client and convenience accessors."""

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from config.settings import (
    MONGODB_URI,
    MONGODB_DATABASE,
    COL_LOGICAL_MODELS,
    COL_PHYSICAL_SCHEMAS,
    COL_GOVERNANCE_TAGS,
    COL_UNIFIED_METADATA,
)

_client: MongoClient | None = None


def get_client() -> MongoClient:
    """Return a cached MongoClient instance."""
    global _client
    if _client is None:
        if not MONGODB_URI:
            raise ValueError("MONGODB_URI is not set. Check your .env file.")
        _client = MongoClient(MONGODB_URI)
    return _client


def get_database() -> Database:
    return get_client()[MONGODB_DATABASE]


def get_collection(name: str) -> Collection:
    return get_database()[name]


# Convenience accessors
def logical_models_col() -> Collection:
    return get_collection(COL_LOGICAL_MODELS)


def physical_schemas_col() -> Collection:
    return get_collection(COL_PHYSICAL_SCHEMAS)


def governance_tags_col() -> Collection:
    return get_collection(COL_GOVERNANCE_TAGS)


def unified_metadata_col() -> Collection:
    return get_collection(COL_UNIFIED_METADATA)
