"""basic_usage.py — Simplest SemCache workflow.

Run with:
    python examples/basic_usage.py
"""

from semcache import SemCache

cache = SemCache()


def fake_llm(prompt: str) -> str:
    print("  [LLM called]")
    return f"Response to: {prompt}"


print("First call:")
print(cache.ask("Explain CNN", fake_llm))

print("\nSecond call (same prompt — should hit cache, LLM not called):")
print(cache.ask("Explain CNN", fake_llm))

print("\nStats:", cache.stats())
