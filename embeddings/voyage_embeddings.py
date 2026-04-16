"""
Voyage AI embedding helper.

Provides a thin wrapper around langchain-voyageai to generate embeddings
that are stored alongside the unified metadata documents and indexed via
Atlas Vector Search.

The same VoyageAIEmbeddings instance is re-used by the vector store and
search retrievers so that query-time embeddings match document-time
embeddings.
"""

from functools import lru_cache

from langchain_voyageai import VoyageAIEmbeddings

from config.settings import VOYAGE_API_KEY, VOYAGE_MODEL


@lru_cache(maxsize=1)
def get_voyage_embeddings() -> VoyageAIEmbeddings:
    """Return a cached VoyageAIEmbeddings instance."""
    if not VOYAGE_API_KEY:
        raise ValueError("VOYAGE_API_KEY is not set. Check your .env file.")
    return VoyageAIEmbeddings(
        voyage_api_key=VOYAGE_API_KEY,
        model=VOYAGE_MODEL,
    )


def generate_embedding(text: str) -> list[float]:
    """Generate a single embedding vector for *text*."""
    model = get_voyage_embeddings()
    return model.embed_query(text)


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Batch-generate embeddings for a list of texts."""
    model = get_voyage_embeddings()
    return model.embed_documents(texts)
