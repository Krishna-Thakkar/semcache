import tempfile
from pathlib import Path

import numpy as np
import pytest

from semcache.stores.faiss_store import FaissVectorStore


def random_unit_vector(dim: int = 384) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def store(tmp_path):
    return FaissVectorStore(storage_dir=str(tmp_path))


class TestFaissVectorStore:
    def test_empty_search_returns_empty(self, store):
        query = random_unit_vector()
        assert store.search(query, k=1) == []

    def test_add_and_search_returns_correct_id(self, store):
        vec = random_unit_vector()
        store.add_vector(42, vec)
        results = store.search(vec, k=1)
        assert len(results) == 1
        assert results[0][0] == 42

    def test_search_score_near_one_for_identical_vector(self, store):
        vec = random_unit_vector()
        store.add_vector(1, vec)
        results = store.search(vec, k=1)
        assert abs(results[0][1] - 1.0) < 1e-5

    def test_total_count_increments(self, store):
        assert store.total == 0
        store.add_vector(1, random_unit_vector())
        assert store.total == 1
        store.add_vector(2, random_unit_vector())
        assert store.total == 2

    def test_multiple_vectors_nearest_neighbor(self, store):
        target = random_unit_vector()
        store.add_vector(10, target)
        store.add_vector(20, random_unit_vector())
        store.add_vector(30, random_unit_vector())
        results = store.search(target, k=1)
        assert results[0][0] == 10

    def test_add_vectors_batch(self, store):
        vecs = np.stack([random_unit_vector() for _ in range(3)])
        store.add_vectors([100, 200, 300], vecs)
        assert store.total == 3

    def test_remove_vector(self, store):
        vec = random_unit_vector()
        store.add_vector(99, vec)
        assert store.total == 1
        store.remove_vector(99)
        assert store.total == 0
        results = store.search(vec, k=1)
        assert results == []

    def test_remove_one_of_many(self, store):
        v1 = random_unit_vector()
        v2 = random_unit_vector()
        store.add_vector(1, v1)
        store.add_vector(2, v2)
        store.remove_vector(1)
        assert store.total == 1
        results = store.search(v2, k=1)
        assert results[0][0] == 2

    def test_save_and_load_preserves_index(self, tmp_path):
        store = FaissVectorStore(storage_dir=str(tmp_path))
        vec = random_unit_vector()
        store.add_vector(7, vec)
        store.save_index()

        store2 = FaissVectorStore(storage_dir=str(tmp_path))
        store2.load_index()
        assert store2.total == 1
        results = store2.search(vec, k=1)
        assert results[0][0] == 7
        assert abs(results[0][1] - 1.0) < 1e-5

    def test_load_nonexistent_index_is_noop(self, tmp_path):
        store = FaissVectorStore(storage_dir=str(tmp_path))
        store.load_index()  # should not raise
        assert store.total == 0

    def test_search_k_larger_than_index(self, store):
        store.add_vector(1, random_unit_vector())
        results = store.search(random_unit_vector(), k=10)
        assert len(results) <= 1
