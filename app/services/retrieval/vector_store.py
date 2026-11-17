import logfire
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from app.config import settings


def get_client() -> QdrantClient:
    """Connect to Qdrant from .env settings (local or cloud)."""
    if settings.QDRANT_URL.startswith("https"):
        return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    return QdrantClient(url=settings.QDRANT_URL)


def ensure_collection(client: QdrantClient, dim: int) -> None:
    """(Re)create the collection with the embedding dimension. Wipes old points."""
    name = settings.QDRANT_COLLECTION
    with logfire.span("Ensure collection", collection=name, dim=dim):
        if client.collection_exists(name):
            client.delete_collection(name)
            logfire.info(f"Deleted existing collection '{name}'.")
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logfire.info(f"Created collection '{name}' (dim={dim}, COSINE).")


def upsert_points(client: QdrantClient, points: list[PointStruct]) -> None:
    """Upsert one batch of points."""
    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)


def search(client: QdrantClient, query_vector: list[float], limit: int = 5):
    """Top-k nearest neighbours for a query vector. Returns scored points with payload."""
    results = client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    return results.points
