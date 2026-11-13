"""Persistent vector store: Chroma (PersistentClient, on-disk) with cosine HNSW.

One collection per chunking strategy (A, B). Strategy C reuses collection B and
only adds a metadata `where` filter, so B and C differ in exactly one factor.
IDs are deterministic, so re-ingesting the same version is an idempotent
upsert; chunks of a re-ingested document that no longer exist are deleted.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from .models import Chunk

FILTERABLE_FIELDS = {"chapter": int, "section_id": str, "document_id": str}
_BATCH = 256


class ChromaStore:
    def __init__(self, path: Path, dimensions: int, embedding_model: str) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.path = Path(path)
        self.dimensions = dimensions
        self.embedding_model = embedding_model
        self.client = chromadb.PersistentClient(path=str(self.path),
                                                settings=Settings(anonymized_telemetry=False))

    @staticmethod
    def collection_name(strategy: str) -> str:
        return f"day6_strategy_{strategy}"

    def collection(self, strategy: str):
        col = self.client.get_or_create_collection(
            name=self.collection_name(strategy),
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": self.embedding_model, "dimensions": self.dimensions},
            embedding_function=None,
        )
        meta = col.metadata or {}
        if meta.get("embedding_model") != self.embedding_model or meta.get("dimensions") != self.dimensions:
            raise RuntimeError(f"Collection {col.name} was built with {meta}; refusing to mix models/dimensions")
        return col

    def reset(self, strategy: str) -> None:
        try:
            self.client.delete_collection(self.collection_name(strategy))
        except Exception:  # collection did not exist
            pass

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> dict[str, int]:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        bad = [c.chunk_id for c, v in zip(chunks, vectors) if len(v) != self.dimensions]
        if bad:
            raise ValueError(f"wrong vector dimension for {bad[:3]}")
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate chunk IDs inside one ingest batch")

        col = self.collection(strategy)
        before = col.count()
        new_ids = set(ids)
        obsolete: list[str] = []
        for doc_id in sorted({c.document_id for c in chunks}):
            existing = col.get(where={"document_id": doc_id}, include=[])["ids"]
            obsolete += [i for i in existing if i not in new_ids]
        if obsolete:
            col.delete(ids=obsolete)

        for i in range(0, len(chunks), _BATCH):
            batch = chunks[i:i + _BATCH]
            col.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=vectors[i:i + _BATCH],
                documents=[c.text for c in batch],
                metadatas=[{"document_id": c.document_id, "source": c.source, "page": c.page, **c.metadata}
                           for c in batch],
            )
        return {"before": before, "after": col.count(), "deleted_obsolete": len(obsolete), "upserted": len(chunks)}

    @staticmethod
    def build_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
        if not filters:
            return None
        clauses = []
        for key, value in filters.items():
            if key not in FILTERABLE_FIELDS:
                raise ValueError(f"filter on '{key}' is not allowed")
            clauses.append({key: {"$eq": FILTERABLE_FIELDS[key](value)}})
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimensions:
            raise ValueError("query vector dimension mismatch")
        col = self.collection(strategy)
        res = col.query(query_embeddings=[query_vector], n_results=top_k, where=self.build_where(filters),
                        include=["documents", "metadatas", "distances"])
        hits = []
        for rank, (cid, text, meta, dist) in enumerate(zip(res["ids"][0], res["documents"][0],
                                                            res["metadatas"][0], res["distances"][0]), start=1):
            hits.append({"rank": rank, "chunk_id": cid, "score": round(1.0 - float(dist), 6),
                         "cosine_distance": round(float(dist), 6), "text": text, "metadata": meta})
        return hits

    def all_records(self, strategy: str) -> dict[str, Any]:
        return self.collection(strategy).get(include=["embeddings", "metadatas"])

    def count(self, strategy: str) -> int:
        return self.collection(strategy).count()
