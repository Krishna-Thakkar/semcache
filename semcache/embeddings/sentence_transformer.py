import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedding:
    """Generates L2-normalized embeddings using a SentenceTransformer model.

    All output vectors are unit-normalized so that inner product in the FAISS
    index equals cosine similarity.
    """

    EMBEDDING_DIM = 384

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Input text (should be normalized before calling).

        Returns:
            L2-normalized float32 vector of shape (384,).
        """
        vector = self._model.encode(text, convert_to_numpy=True).astype(np.float32)
        return self._normalize(vector)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of text strings.

        Args:
            texts: List of input strings.

        Returns:
            L2-normalized float32 array of shape (n, 384).
        """
        vectors = self._model.encode(texts, convert_to_numpy=True).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
