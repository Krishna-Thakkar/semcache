"""configurable_threshold.py — Tuning semantic_threshold and top_k.

semantic_threshold controls how similar two prompts must be for the second
to be served from cache.  Lower values increase cache hits; higher values
require a closer match.

top_k controls how many candidate vectors FAISS retrieves.  Increasing it
gives the matcher more options but adds a small search cost.

Run with:
    python examples/configurable_threshold.py
"""

from semcache import SemCache


def fake_llm(prompt: str) -> str:
    print(f"  [LLM called] {prompt!r}")
    return f"Response to: {prompt}"


# --- Strict cache (default 0.90) ---
print("=== strict cache (threshold=0.90) ===")
strict = SemCache(semantic_threshold=0.90, top_k=5)
strict.ask("Explain CNN", fake_llm)
strict.ask("Explain convolutional neural network", fake_llm)  # may miss
print()

# --- Lenient cache ---
print("=== lenient cache (threshold=0.70) ===")
lenient = SemCache(semantic_threshold=0.70, top_k=5)
lenient.ask("Explain CNN", fake_llm)
lenient.ask("Explain convolutional neural network", fake_llm)  # more likely to hit
print()

# --- Narrow top_k ---
print("=== narrow top_k (top_k=1) ===")
narrow = SemCache(semantic_threshold=0.90, top_k=1)
narrow.ask("Explain CNN", fake_llm)
narrow.ask("Explain CNN please", fake_llm)  # exact hit via prompt_index
print()
