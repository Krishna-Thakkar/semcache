import functools

from semcache.core.semcache import SemCache


def cache():
    """Decorator that wraps an LLM function with SemCache semantic caching.

    A single ``SemCache`` instance is created per decoration and reused
    across all calls, so the cache persists for the lifetime of the process.

    Usage::

        @cache()
        def ask_llm(prompt: str) -> str:
            return openai.chat(prompt)

        response = ask_llm("Explain CNN")   # calls LLM on miss
        response = ask_llm("Explain CNN")   # returned from cache
    """
    semcache = SemCache()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(prompt: str) -> str:
            return semcache.ask(prompt, func)
        return wrapper

    return decorator
