from pathlib import Path

import faiss
import numpy as np


class FaissVectorStore:
    """FAISS-backed vector store using IndexIDMap over IndexFlatIP.

    Vectors must be L2-normalized before insertion so that inner product
    equals cosine similarity.
    """

    INDEX_FILENAME = "index.faiss"

    def __init__(self, storage_dir: str = ".semcache", dim: int = 384):
        self._dim = dim
        self._index_path = Path(storage_dir) / self.INDEX_FILENAME
        self._index = self._build_empty_index()

    # ------------------------------------------------------------------
    # Index construction helpers
    # ------------------------------------------------------------------

    def _build_empty_index(self) -> faiss.IndexIDMap:
        base = faiss.IndexFlatIP(self._dim)
        return faiss.IndexIDMap(base)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_vector(self, vector_id: int, vector: np.ndarray) -> None:
        """Add a single vector to the index.

        Args:
            vector_id: Stable integer ID (matches SQLite primary key).
            vector: L2-normalized float32 array of shape (dim,) or (1, dim).
        """
        vec = np.ascontiguousarray(vector, dtype=np.float32).reshape(1, self._dim)
        ids = np.array([vector_id], dtype=np.int64)
        self._index.add_with_ids(vec, ids)

    def add_vectors(self, vector_ids: list[int], vectors: np.ndarray) -> None:
        """Batch-add vectors to the index.

        Args:
            vector_ids: List of stable integer IDs.
            vectors: L2-normalized float32 array of shape (n, dim).
        """
        vecs = np.ascontiguousarray(vectors, dtype=np.float32).reshape(len(vector_ids), self._dim)
        ids = np.array(vector_ids, dtype=np.int64)
        self._index.add_with_ids(vecs, ids)

    def remove_vector(self, vector_id: int) -> None:
        """Remove a single vector from the index by its stable ID.

        Args:
            vector_id: ID to remove.
        """
        ids = faiss.IDSelectorArray(np.array([vector_id], dtype=np.int64))
        self._index.remove_ids(ids)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, vector: np.ndarray, k: int = 5
    ) -> list[tuple[int, float]]:
        """Find the k nearest vectors by cosine similarity.

        Args:
            vector: L2-normalized float32 query vector of shape (dim,).
            k: Number of results to return.

        Returns:
            List of (vector_id, similarity_score) tuples, ordered by
            descending similarity. Empty if the index contains no vectors.
        """
        if self._index.ntotal == 0:
            return []

        k = min(k, self._index.ntotal)
        vec = np.ascontiguousarray(vector, dtype=np.float32).reshape(1, self._dim)
        scores, ids = self._index.search(vec, k)

        results = []
        for score, vid in zip(scores[0], ids[0]):
            if vid != -1:
                results.append((int(vid), float(score)))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self) -> None:
        """Write the FAISS index to disk."""
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))

    def load_index(self) -> None:
        """Load the FAISS index from disk if the file exists.

        The loaded index is re-wrapped with IndexIDMap to preserve stable IDs.
        """
        if not self._index_path.exists():
            return
        loaded = faiss.read_index(str(self._index_path))
        # faiss.read_index restores the full IndexIDMap; assign directly.
        self._index = loaded

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def total(self) -> int:
        """Number of vectors currently in the index."""
        return self._index.ntotal
