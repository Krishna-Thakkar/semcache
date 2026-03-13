from typing import Callable

from semcache.core.cache_manager import CacheManager
from semcache.utils.prompt_canonicalizer import canonicalize_prompt
from semcache.utils.question_extractor import extract_question


class SemCache:
    """User-facing API for the SemCache semantic caching library."""

    def __init__(
        self,
        extract_question: bool = False,
        canonicalize_prompt: bool = True,
    ):
        self.cache_manager = CacheManager()
        self.extract_question = extract_question
        self.canonicalize_prompt = canonicalize_prompt

    def ask(self, prompt: str, llm_fn: Callable[[str], str]) -> str:
        """Return a cached or freshly generated response for *prompt*.

        Pipeline applied before cache lookup:
        1. extract_question (optional) — strip RAG context, keep question text
        2. canonicalize_prompt (optional) — remove conversational filler
        3. normalize_prompt — handled inside CacheManager

        Args:
            prompt: The user's input string.
            llm_fn: Callable that accepts a prompt and returns a response string.

        Returns:
            Cached response on hit, or the result of llm_fn on miss.
        """
        if self.extract_question:
            prompt = extract_question(prompt)
        if self.canonicalize_prompt:
            prompt = canonicalize_prompt(prompt)
        return self.cache_manager.query(prompt, llm_fn)

    def stats(self) -> dict:
        """Return basic cache statistics.

        Returns:
            Dictionary containing 'entries': total number of cached responses.
        """
        return {
            "entries": self.cache_manager.metadata_store.get_total_entries()
        }

    def clear(self) -> None:
        """Clear all cached data.

        Resets the FAISS index, deletes all SQLite metadata rows,
        and clears the in-memory prompt index.
        """
        ms = self.cache_manager.metadata_store
        vs = self.cache_manager.vector_store

        for entry in ms.get_lru_entries():
            ms.delete_entry(entry["vector_id"])
            vs.remove_vector(entry["vector_id"])

        ms.prompt_index.clear()
