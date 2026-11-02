"""`sentence-transformers` implementation of `Embedder` for the BGE family (local, CPU-friendly)."""

import logging
import time

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BgeEmbedder:
    """Loads the model once. Documents are embedded as-is; queries get the BGE instruction.

    Vectors are L2-normalized, so cosine similarity == dot product.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 16,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ) -> None:
        started = time.perf_counter()
        self._model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.dimensions: int = int(self._model.get_embedding_dimension() or 0)
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        logger.info(
            "Loaded embedding model %s (%d dims, device=%s) in %.1fs",
            model_name,
            self.dimensions,
            device,
            time.perf_counter() - started,
        )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"{self.query_instruction}{text}"])[0]
