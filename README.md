# SemCache

Semantic caching for LLM calls to reduce cost and latency.

SemCache sits in front of any LLM function and returns a cached response
when an incoming prompt is semantically equivalent to one it has seen
before — even if the wording differs.

---

## Installation

> Until a PyPI release is available, install directly from the repository:

```bash
pip install -e .
```

---

## Quick Start

```python
from semcache import SemCache

cache = SemCache()

def call_llm(prompt: str) -> str:
    # replace with your real LLM call
    return f"Response to: {prompt}"

# First call — LLM is invoked and the response is cached
response = cache.ask("Explain CNN", call_llm)

# Second call — returned instantly from cache, LLM not called
response = cache.ask("Explain CNN", call_llm)
```

---

## Decorator Usage

Wrap any LLM function with `@cache()` to add caching transparently:

```python
from semcache.decorators import cache

@cache()
def call_llm(prompt: str) -> str:
    return f"Response to: {prompt}"

call_llm("Explain CNN")   # LLM invoked, response cached
call_llm("Explain CNN")   # served from cache
```

---

## RAG / Question Extraction

When prompts include a context block followed by a question, enable
`extract_question=True` to cache on the question alone.  Two prompts
with different context but the same question will share a cache entry.

```python
cache = SemCache(extract_question=True)

rag_prompt = """
Context:
CNNs are deep learning models used for image recognition.

Question: What is CNN?
"""

cache.ask(rag_prompt, call_llm)  # cached under "What is CNN?"
```

---

## Configuration

All parameters are optional and have sensible defaults:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `semantic_threshold` | `float` | `0.90` | Minimum cosine similarity to count as a cache hit |
| `top_k` | `int` | `5` | Number of FAISS candidates to retrieve per query |
| `extract_question` | `bool` | `False` | Strip RAG context, cache on question only |
| `canonicalize_prompt` | `bool` | `True` | Remove conversational filler (`please`, `can you`, …) |

```python
cache = SemCache(
    semantic_threshold=0.85,
    top_k=3,
    extract_question=True,
    canonicalize_prompt=True,
)
```

### Threshold guidance

- **0.95+** — very strict; only near-identical phrasings hit the cache
- **0.90** (default) — balanced; handles minor wording variation
- **0.80–0.85** — lenient; catches paraphrases and synonyms
- **< 0.80** — aggressive; risk of false hits on unrelated prompts

---

## How It Works

Every `ask()` call passes through the following pipeline:

```
prompt
  ↓
extract_question    (optional) — keep only the question from a RAG prompt
  ↓
canonicalize_prompt (optional) — strip conversational filler words
  ↓
normalize_prompt    — lowercase, remove punctuation, collapse whitespace
  ↓
exact cache lookup  — O(1) dict check; returns immediately on hit
  ↓
embed prompt        — SentenceTransformer (all-MiniLM-L6-v2, 384-dim)
  ↓
FAISS search        — cosine similarity via IndexFlatIP + IndexIDMap
  ↓
threshold check     — return cached response if score ≥ semantic_threshold
  ↓
LLM call            — invoke llm_fn, store result, return response
```

Cache state is persisted across process restarts:

- **Vector index** — `~/.semcache/index.faiss`
- **Metadata** — `~/.semcache/metadata.sqlite`

---

## Cache Management

```python
# Current entry count
cache.stats()          # {"entries": 42}

# Wipe all cached data
cache.clear()
```

---

## Examples

| File | Description |
|---|---|
| `examples/basic_usage.py` | Minimal end-to-end example |
| `examples/decorator_usage.py` | `@cache()` decorator |
| `examples/rag_usage.py` | Question extraction for RAG prompts |
| `examples/configurable_threshold.py` | Threshold and top_k tuning |

Run any example with:

```bash
python examples/basic_usage.py
```
