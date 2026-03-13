import numpy as np
import pytest

from semcache.core.cache_manager import CacheManager
from semcache.core.semcache import SemCache
from semcache.stores.faiss_store import FaissVectorStore
from semcache.stores.metadata_store import MetadataStore


def fake_llm(prompt: str) -> str:
    return f"response:{prompt}"


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


class StubEmbedding:
    def __init__(self, vec: np.ndarray):
        self._vec = vec

    def embed(self, text: str) -> np.ndarray:
        return self._vec


def _make_semcache(tmp_path) -> SemCache:
    """Return a SemCache backed by isolated tmp stores."""
    sc = SemCache.__new__(SemCache)
    rng = np.random.default_rng(0)
    vec = _unit(rng.standard_normal(384).astype(np.float32))
    sc.cache_manager = CacheManager(
        metadata_store=MetadataStore(db_path=str(tmp_path / "meta.sqlite")),
        vector_store=FaissVectorStore(storage_dir=str(tmp_path)),
        embedding_engine=StubEmbedding(vec),
    )
    sc.extract_question = False
    return sc


class TestBasicUsage:
    def test_ask_returns_llm_response_on_miss(self, tmp_path):
        sc = _make_semcache(tmp_path)
        result = sc.ask("Explain CNN", fake_llm)
        assert result == "response:Explain CNN"

    def test_ask_returns_string(self, tmp_path):
        sc = _make_semcache(tmp_path)
        result = sc.ask("What is deep learning?", fake_llm)
        assert isinstance(result, str)


class TestCacheReuse:
    def test_second_call_returns_same_response(self, tmp_path):
        sc = _make_semcache(tmp_path)
        r1 = sc.ask("Explain CNN", fake_llm)
        r2 = sc.ask("Explain CNN", fake_llm)
        assert r1 == r2

    def test_llm_called_only_once_on_repeat(self, tmp_path):
        sc = _make_semcache(tmp_path)
        calls = []

        def counting_llm(p):
            calls.append(p)
            return f"response:{p}"

        sc.ask("Explain CNN", counting_llm)
        sc.ask("Explain CNN", counting_llm)
        assert len(calls) == 1


class TestStats:
    def test_stats_returns_dict(self, tmp_path):
        sc = _make_semcache(tmp_path)
        assert isinstance(sc.stats(), dict)

    def test_stats_entries_zero_initially(self, tmp_path):
        sc = _make_semcache(tmp_path)
        assert sc.stats()["entries"] == 0

    def test_stats_entries_increments_after_ask(self, tmp_path):
        sc = _make_semcache(tmp_path)
        sc.ask("Hello", fake_llm)
        assert sc.stats()["entries"] == 1


class TestClear:
    def test_clear_empties_metadata(self, tmp_path):
        sc = _make_semcache(tmp_path)
        sc.ask("Hello", fake_llm)
        assert sc.stats()["entries"] == 1
        sc.clear()
        assert sc.stats()["entries"] == 0

    def test_clear_empties_faiss_index(self, tmp_path):
        sc = _make_semcache(tmp_path)
        sc.ask("Hello", fake_llm)
        sc.clear()
        assert sc.cache_manager.vector_store.total == 0

    def test_clear_empties_prompt_index(self, tmp_path):
        sc = _make_semcache(tmp_path)
        sc.ask("Hello", fake_llm)
        sc.clear()
        assert sc.cache_manager.metadata_store.prompt_index == {}

    def test_llm_called_again_after_clear(self, tmp_path):
        sc = _make_semcache(tmp_path)
        calls = []

        def counting_llm(p):
            calls.append(p)
            return f"response:{p}"

        sc.ask("Hello", counting_llm)
        sc.clear()
        sc.ask("Hello", counting_llm)
        assert len(calls) == 2

    def test_clear_on_empty_cache_is_noop(self, tmp_path):
        sc = _make_semcache(tmp_path)
        sc.clear()  # should not raise
        assert sc.stats()["entries"] == 0


class TestPackageImport:
    def test_semcache_importable_from_package(self):
        from semcache import SemCache as SC
        assert SC is SemCache
