"""decorator_usage.py — Decorator-based caching.

The @cache() decorator wraps any LLM function so responses are cached
automatically without changing call sites.

Run with:
    python examples/decorator_usage.py
"""

from semcache.decorators import cache


@cache()
def ask_llm(prompt: str) -> str:
    print("  [LLM called]")
    return f"Response to: {prompt}"


print("First call:")
print(ask_llm("Explain CNN"))

print("\nSecond call (same prompt — LLM not called):")
print(ask_llm("Explain CNN"))

print("\nDifferent prompt (LLM called again):")
print(ask_llm("What is backpropagation?"))
