from semcache import SemCache


def fake_llm(prompt):
    return "response"


cache = SemCache()

print(cache.ask("Explain CNN", fake_llm))
