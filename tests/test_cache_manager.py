"""Tests for CacheManager.

A stub embedding engine is used so tests are fast and fully deterministic.
The stub returns a fixed or per-call controlled numpy vector so we can
precisely exercise exact-hit, semantic-hit, and miss paths.
"""
import numpy as np
import pytest

from semcache.core.cache_manager import CacheManager, SIMILARITY_THRESHOLD
from semcache.stores.faiss_store import FaissVectorStore
from semcache.stores.metadata_store import MetadataStore


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def _fixed_vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal(384).astype(np.float32))


class StubEmbedding:
    """Returns vectors from a pre-set queue; repeats last if queue exhausted."""

    def __init__(self, vectors: list[np.ndarray]):
        self._queue = list(vectors)
        self._last = vectors[-1]

    def embed(self, text: str) -> np.ndarray:
        if self._queue:
            self._last = self._queue.pop(0)
        return self._last


def fake_llm(prompt: str) -> str:
    return f"response:{prompt}"


def _make_manager(tmp_path, vectors: list[np.ndarray]) -> CacheManager:
    return CacheManager(
        metadata_store=MetadataStore(db_path=str(tmp_path / "meta.sqlite")),
        vector_store=FaissVectorStore(storage_dir=str(tmp_path)),
        embedding_engine=StubEmbedding(vectors),
    )


# ---------------------------------------------------------------------------
# Exact cache hit
# ---------------------------------------------------------------------------

class TestExactCacheHit:
    def test_llm_not_called_on_exact_hit(self, tmp_path):
        vec = _fixed_vec(0)
        manager = _make_manager(tmp_path, [vec, vec])
        calls = []

        def recording_llm(p):
            calls.append(p)
            return f"response:{p}"

        manager.query("What is AI?", recording_llm)      # miss — LLM called
        manager.query("What is AI?", recording_llm)      # exact hit — LLM NOT called
        assert len(calls) == 1

    def test_exact_hit_returns_cached_response(self, tmp_path):
        vec = _fixed_vec(1)
        manager = _make_manager(tmp_path, [vec, vec])
        r1 = manager.query("Hello world", fake_llm)
        r2 = manager.query("Hello world", fake_llm)
        assert r1 == r2

    def test_exact_hit_increments_hit_count(self, tmp_path):
        vec = _fixed_vec(2)
        manager = _make_manager(tmp_path, [vec, vec])
        manager.query("Hello", fake_llm)
        manager.query("Hello", fake_llm)
        entry = manager.metadata_store.get_entry(1)
        assert entry["hit_count"] == 1  # one access-time update on second query


# ---------------------------------------------------------------------------
# Semantic cache hit
# ---------------------------------------------------------------------------

class TestSemanticCacheHit:
    def test_similar_vector_hits_cache(self, tmp_path):
        base = _fixed_vec(10)
        # near-identical vector: tiny perturbation stays above threshold
        tiny_noise = np.zeros(384, dtype=np.float32)
        tiny_noise[0] = 1e-4
        similar = _unit(base + tiny_noise)

        manager = _make_manager(tmp_path, [base, similar])
        calls = []

        def recording_llm(p):
            calls.append(p)
            return f"response:{p}"

        manager.query("first prompt", recording_llm)     # miss
        manager.query("second prompt", recording_llm)    # semantic hit

        assert len(calls) == 1

    def test_semantic_hit_returns_same_response(self, tmp_path):
        base = _fixed_vec(11)
        tiny_noise = np.zeros(384, dtype=np.float32)
        tiny_noise[0] = 1e-4
        similar = _unit(base + tiny_noise)

        manager = _make_manager(tmp_path, [base, similar])
        r1 = manager.query("first prompt", fake_llm)
        r2 = manager.query("second prompt", fake_llm)
        assert r1 == r2

    def test_dissimilar_vector_causes_miss(self, tmp_path):
        v1 = _fixed_vec(20)
        v2 = _fixed_vec(21)  # independent random vector — will be below threshold
        manager = _make_manager(tmp_path, [v1, v2])
        calls = []

        def recording_llm(p):
            calls.append(p)
            return f"response:{p}"

        manager.query("first prompt", recording_llm)
        manager.query("unrelated prompt", recording_llm)
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# Cache miss
# ---------------------------------------------------------------------------

class TestCacheMiss:
    def test_miss_stores_new_entry(self, tmp_path):
        manager = _make_manager(tmp_path, [_fixed_vec(30)])
        manager.query("brand new prompt", fake_llm)
        assert manager.metadata_store.get_total_entries() == 1

    def test_miss_returns_llm_response(self, tmp_path):
        manager = _make_manager(tmp_path, [_fixed_vec(31)])
        result = manager.query("my prompt", fake_llm)
        assert result == "response:my prompt"

    def test_entry_stored_with_correct_metadata(self, tmp_path):
        manager = _make_manager(tmp_path, [_fixed_vec(32)])
        manager.query("test prompt", fake_llm)
        entry = manager.metadata_store.get_entry(1)
        assert entry["prompt"] == "test prompt"
        assert entry["response"] == "response:test prompt"


# ---------------------------------------------------------------------------
# Multiple queries / hit_count accumulation
# ---------------------------------------------------------------------------

class TestMultipleQueries:
    def test_repeated_exact_queries_increase_hit_count(self, tmp_path):
        vec = _fixed_vec(40)
        manager = _make_manager(tmp_path, [vec] + [vec] * 4)
        manager.query("repeat me", fake_llm)   # miss, stored
        for _ in range(4):
            manager.query("repeat me", fake_llm)  # 4 exact hits
        entry = manager.metadata_store.get_entry(1)
        assert entry["hit_count"] == 4

    def test_independent_prompts_stored_separately(self, tmp_path):
        manager = _make_manager(tmp_path, [_fixed_vec(50), _fixed_vec(51)])
        manager.query("prompt alpha", fake_llm)
        manager.query("prompt beta", fake_llm)
        assert manager.metadata_store.get_total_entries() == 2
