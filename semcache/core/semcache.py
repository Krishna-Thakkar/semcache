# Placeholder for Phase 8 implementation — SemCache public API


class SemCache:
    """Public interface for the SemCache semantic caching library.

    Implementation coming in Phase 8.
    """

    def __init__(
        self,
        storage_dir: str = ".semcache",
        similarity_threshold: float = 0.85,
        max_size: int = 1000,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        raise NotImplementedError("SemCache will be implemented in Phase 8.")

    def ask(self, prompt: str, llm_fn) -> str:
        raise NotImplementedError

    def stats(self) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError
