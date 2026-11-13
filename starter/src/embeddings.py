"""Local embedding model via Sentence Transformers (no API key, no cost).

Model: BAAI/bge-small-en-v1.5 pinned to a Hugging Face revision. Documents are
embedded as-is; queries get the BGE retrieval instruction prefix recommended
by the model card. Vectors are L2-normalized so cosine distance is meaningful.
Errors propagate: there is no fallback to mock/random vectors.
"""
from __future__ import annotations

from typing import Protocol

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, question: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, revision: str = DEFAULT_REVISION,
                 batch_size: int = 32, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.device = device
        self.model = SentenceTransformer(model_name, revision=revision, device=device)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dimensions = int(get_dim())
        self.max_seq_length = int(self.model.max_seq_length)
        self.query_instruction = BGE_QUERY_INSTRUCTION if "bge" in model_name.lower() else ""

    def token_count(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=True)["input_ids"])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                    show_progress_bar=False, convert_to_numpy=True)
        return vectors.tolist()

    def embed_query(self, question: str) -> list[float]:
        vector = self.model.encode([self.query_instruction + question], normalize_embeddings=True,
                                   show_progress_bar=False, convert_to_numpy=True)
        return vector[0].tolist()

    def describe(self) -> dict:
        return {
            "provider": "local / sentence-transformers",
            "model": self.model_name,
            "revision": self.revision,
            "dimensions": self.dimensions,
            "max_seq_length_tokens": self.max_seq_length,
            "normalize_embeddings": True,
            "query_instruction": self.query_instruction,
            "document_processing": "raw chunk text, no prefix",
            "device": self.device,
            "cost_usd": 0.0,
        }
