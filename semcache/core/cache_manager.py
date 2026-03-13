from typing import Callable, Optional

from semcache.embeddings.sentence_transformer import SentenceTransformerEmbedding
from semcache.stores.faiss_store import FaissVectorStore
from semcache.stores.metadata_store import MetadataStore
from semcache.utils.normalize import normalize_prompt

DEFAULT_SIMILARITY_THRESHOLD = 0.9
DEFAULT_TOP_K = 5


class CacheManager:
    """Orchestrates prompt normalization, exact lookup, semantic search,
    and cache insertion across the embedding, FAISS, and metadata layers.
    """

    def __init__(
        self,
        metadata_store: Optional[MetadataStore] = None,
        vector_store: Optional[FaissVectorStore] = None,
        embedding_engine: Optional[SentenceTransformerEmbedding] = None,
        semantic_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.metadata_store = metadata_store or MetadataStore()
        self.vector_store = vector_store or FaissVectorStore()
        self.embedding_engine = embedding_engine or SentenceTransformerEmbedding()
        self.semantic_threshold = semantic_threshold
        self.top_k = top_k

    def query(self, prompt: str, llm_fn: Callable[[str], str]) -> str:
        """Return a cached response for *prompt*, calling *llm_fn* on a miss.

        Lookup order:
        1. Exact match via prompt_index (O(1)).
        2. Semantic match via FAISS (score ≥ SIMILARITY_THRESHOLD).
        3. Full miss — call llm_fn, store result, return response.
        """
        normalized = normalize_prompt(prompt)

        # Step 2 — exact cache lookup
        if normalized in self.metadata_store.prompt_index:
            vector_id = self.metadata_store.prompt_index[normalized]
            entry = self.metadata_store.get_entry(vector_id)
            if entry:
                self.metadata_store.update_access_time(vector_id)
                return entry["response"]

        # Step 3 — generate embedding
        embedding = self.embedding_engine.embed(prompt)

        # Step 4 & 5 — semantic search with threshold (skip if index empty)
        if self.vector_store.total > 0:
            k = min(self.top_k, self.vector_store.total)
            results = self.vector_store.search(embedding, k=k)
        else:
            results = []
        if results:
            best_id, best_score = results[0]
            if best_score >= self.semantic_threshold:
                entry = self.metadata_store.get_entry(best_id)
                if entry:
                    self.metadata_store.update_access_time(best_id)
                    return entry["response"]

        # Step 6 — cache miss: call LLM
        response = llm_fn(prompt)

        # Step 7 — store result
        vector_id = self.metadata_store.get_total_entries() + 1
        self.vector_store.add_vector(vector_id, embedding)
        self.metadata_store.add_entry(vector_id, prompt, normalized, response)

        return response
