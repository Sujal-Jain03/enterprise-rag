"""Embed processed_data chunks and upsert them into Qdrant.

Usage:
    python index_data.py                  # index everything (true + noisy)
    python index_data.py --source true    # index true_data only (quick test)
    python index_data.py --resume         # continue a crashed run (skips indexed chunks)

Chunks are embedded + upserted in SUPER_BATCHes of 500, so a crash loses
at most one super-batch of work. Point IDs are each chunk's global index
in the full corpus, assigned before any --source filter, so an ID means
the same chunk in every run and --resume skips exactly the right chunks.
"""
import argparse
import glob
import json
import logfire

logfire.configure(service_name="indexing", send_to_logfire=False)

from app.config import settings
from app.services.retrieval import embeddings as emb
from app.services.retrieval import vector_store as vs
from qdrant_client.models import PointStruct

UPSERT_BATCH = 100
SUPER_BATCH = 500


def load_chunks(source: str):
    """Load (global_id, text, payload) in stable sorted order.

    IDs come from the FULL corpus before filtering by source, so a chunk's
    ID is the same for --source true and --source all.
    """
    items = []
    for path in sorted(glob.glob("processed_data/*/*.json")):
        data = json.load(open(path, encoding="utf-8"))
        for i, text in enumerate(data["chunks"]):
            items.append((len(items), text, {
                "filename": data["filename"],
                "source_type": data["source_type"],
                "chunk_index": i,
                "text": text,
            }))
    if source != "all":
        items = [it for it in items if it[2]["source_type"] == source]
    return items


def existing_ids(client) -> set[int]:
    """Point IDs already in the collection (one scroll fetches all)."""
    points, _ = client.scroll(
        collection_name=settings.QDRANT_COLLECTION,
        limit=10000, with_payload=False, with_vectors=False,
    )
    return {p.id for p in points}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["true", "noisy", "all"], default="all")
    parser.add_argument("--resume", action="store_true",
                        help="skip chunks already in Qdrant; do not recreate the collection")
    args = parser.parse_args()

    items = load_chunks(args.source)
    print(f"Loaded {len(items)} chunks (source={args.source}).")

    dim = emb.get_embedding_dim()
    print(f"Embedding model: {emb._model_type}, dim: {dim}")

    client = vs.get_client()
    done = set()
    if args.resume:
        if not client.collection_exists(settings.QDRANT_COLLECTION):
            raise SystemExit("Cannot resume: collection does not exist. Run without --resume.")
        col_dim = client.get_collection(
            collection_name=settings.QDRANT_COLLECTION
        ).config.params.vectors.size
        if col_dim != dim:
            raise SystemExit(
                f"Collection dim is {col_dim} but active model dim is {dim} — "
                "cannot resume. Re-run without --resume to rebuild."
            )
        done = existing_ids(client)
        print(f"Resume: {len(done)} chunks already indexed, skipping them.")
    else:
        vs.ensure_collection(client, dim)

    todo = [(gid, t, p) for (gid, t, p) in items if gid not in done]
    print(f"Indexing {len(todo)} chunks in super-batches of {SUPER_BATCH}.")
    for start in range(0, len(todo), SUPER_BATCH):
        group = todo[start:start + SUPER_BATCH]
        vectors = emb.embed_texts([t for _, t, _ in group])
        points = [
            PointStruct(id=gid, vector=vec, payload=payload)
            for (gid, _, payload), vec in zip(group, vectors)
        ]
        for j in range(0, len(points), UPSERT_BATCH):
            vs.upsert_points(client, points[j:j + UPSERT_BATCH])
        print(f"Upserted {min(start + SUPER_BATCH, len(todo))}/{len(todo)} chunks.")

    count = client.count(collection_name=settings.QDRANT_COLLECTION)
    print(f"Done. Collection now holds {count.count} points.")


if __name__ == "__main__":
    main()
