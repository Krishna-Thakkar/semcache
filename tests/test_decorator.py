"""Tests for the @cache() decorator and optional RAG question extraction."""
import numpy as np
import pytest

from semcache.core.cache_manager import CacheManager
from semcache.core.semcache import SemCache
from semcache.decorators import cache
from semcache.stores.faiss_store import FaissVectorStore
from semcache.stores.metadata_store import MetadataStore
from semcache.utils.question_extractor import extract_question


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


class StubEmbedding:
    """Returns vectors from a queue; repeats last entry when queue is empty."""

    def __init__(self, *vecs: np.ndarray):
        self._queue = list(vecs)
        self._last = vecs[-1]

    def embed(self, text: str) -> np.ndarray:
        if self._queue:
            self._last = self._queue.pop(0)
        return self._last


def _fixed_vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal(384).astype(np.float32))


def _make_semcache(tmp_path, extract: bool = False, *vecs) -> SemCache:
    sc = SemCache.__new__(SemCache)
    sc.extract_question = extract
    embedding = StubEmbedding(*vecs) if vecs else StubEmbedding(_fixed_vec(0))
    sc.cache_manager = CacheManager(
        metadata_store=MetadataStore(db_path=str(tmp_path / "meta.sqlite")),
        vector_store=FaissVectorStore(storage_dir=str(tmp_path)),
        embedding_engine=embedding,
    )
    return sc


# ---------------------------------------------------------------------------
# question_extractor unit tests
# ---------------------------------------------------------------------------

class TestExtractQuestion:
    def test_extracts_question_prefix(self):
        prompt = "Context: some text\n\nQuestion: What is CNN?"
        assert extract_question(prompt) == "What is CNN?"

    def test_extracts_q_prefix(self):
        assert extract_question("Q: What is RNN?") == "What is RNN?"

    def test_extracts_user_prefix(self):
        assert extract_question("User: How does attention work?") == "How does attention work?"

    def test_falls_back_to_original_when_no_pattern(self):
        prompt = "Just a plain prompt with no marker."
        assert extract_question(prompt) == prompt

    def test_strips_trailing_whitespace(self):
        assert extract_question("Question:  What is AI?  ") == "What is AI?"

    def test_case_insensitive(self):
        assert extract_question("QUESTION: What is BERT?") == "What is BERT?"

    def test_multiline_context_extracts_correctly(self):
        prompt = (
            "Context:\nLong piece of background text.\n\n"
            "Question: What is the main idea?"
        )
        assert extract_question(prompt) == "What is the main idea?"


# ---------------------------------------------------------------------------
# Decorator tests
# ---------------------------------------------------------------------------

class TestCacheDecorator:
    def test_decorated_function_returns_response(self, tmp_path, monkeypatch):
        sc = _make_semcache(tmp_path)
        calls = []

        # Patch SemCache() inside the decorator to use our isolated instance.
        monkeypatch.setattr(
            "semcache.decorators.cache_decorator.SemCache", lambda: sc
        )

        @cache()
        def my_llm(prompt: str) -> str:
            calls.append(prompt)
            return f"answer:{prompt}"

        result = my_llm("Explain CNN")
        assert result == "answer:Explain CNN"

    def test_decorator_calls_llm_only_once(self, tmp_path, monkeypatch):
        sc = _make_semcache(tmp_path)
        calls = []

        monkeypatch.setattr(
            "semcache.decorators.cache_decorator.SemCache", lambda: sc
        )

        @cache()
        def my_llm(prompt: str) -> str:
            calls.append(prompt)
            return f"answer:{prompt}"

        my_llm("Explain CNN")
        my_llm("Explain CNN")
        assert len(calls) == 1

    def test_decorator_second_call_returns_same_response(self, tmp_path, monkeypatch):
        sc = _make_semcache(tmp_path)

        monkeypatch.setattr(
            "semcache.decorators.cache_decorator.SemCache", lambda: sc
        )

        @cache()
        def my_llm(prompt: str) -> str:
            return f"answer:{prompt}"

        r1 = my_llm("Hello world")
        r2 = my_llm("Hello world")
        assert r1 == r2

    def test_decorator_preserves_function_name(self, tmp_path, monkeypatch):
        sc = _make_semcache(tmp_path)

        monkeypatch.setattr(
            "semcache.decorators.cache_decorator.SemCache", lambda: sc
        )

        @cache()
        def my_named_llm(prompt: str) -> str:
            return prompt

        assert my_named_llm.__name__ == "my_named_llm"


# ---------------------------------------------------------------------------
# extract_question integration with SemCache
# ---------------------------------------------------------------------------

class TestExtractQuestionIntegration:
    def test_rag_prompt_hits_cache_by_extracted_question(self, tmp_path):
        sc = _make_semcache(tmp_path, extract=True)
        calls = []

        def llm(p):
            calls.append(p)
            return f"answer:{p}"

        rag_prompt = "Context: background text.\n\nQuestion: What is CNN?"
        plain_prompt = "What is CNN?"

        sc.ask(rag_prompt, llm)   # stores under extracted "What is CNN?"
        sc.ask(plain_prompt, llm) # exact hit — LLM not called again
        assert len(calls) == 1

    def test_extract_question_flag_false_uses_full_prompt(self, tmp_path):
        # Use distinct orthogonal vectors so semantic search does not hit.
        sc = _make_semcache(tmp_path, False, _fixed_vec(0), _fixed_vec(99))
        calls = []

        def llm(p):
            calls.append(p)
            return f"answer:{p}"

        sc.ask("Context: bg\n\nQuestion: What is CNN?", llm)
        sc.ask("What is CNN?", llm)
        # Different normalized prompts + dissimilar vectors → two LLM calls
        assert len(calls) == 2
