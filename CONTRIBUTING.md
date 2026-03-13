# Contributing to SemCache

Thank you for your interest in contributing to SemCache!

---

## Development Environment

**Requirements:** Python 3.10+

```bash
git clone https://github.com/your-org/semcache.git
cd semcache
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

---

## Running Tests

```bash
pytest
```

Run a specific test file:

```bash
pytest tests/test_cache_manager.py -v
```

All tests must pass before submitting a pull request.

---

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use type annotations for all public functions and methods
- Keep functions focused — one responsibility per function
- Prefer `np.ascontiguousarray` over `.astype` when enforcing memory layout

---

## Submitting a Pull Request

1. Fork the repository and create a feature branch:

   ```bash
   git checkout -b feat/my-feature
   ```

2. Make your changes, add tests for new behaviour.

3. Run the full test suite and confirm everything passes:

   ```bash
   pytest
   ```

4. Commit using a semantic commit message:

   ```
   feat(stores): add TTL expiry to metadata store
   fix(core): handle empty prompt correctly
   docs: update README with new config options
   ```

5. Push your branch and open a pull request against `main`.

---

## Commit Message Convention

Format: `<type>(<scope>): <short description>`

| Type | When to use |
|---|---|
| `feat` | New feature |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `refactor` | Code change with no behaviour change |
| `test` | Adding or updating tests |
| `docs` | Documentation only |
| `chore` | Build, packaging, tooling |

---

## Project Structure

```
semcache/
  core/           — CacheManager and SemCache public API
  stores/         — FAISS vector store and SQLite metadata store
  embeddings/     — SentenceTransformer embedding engine
  utils/          — Prompt normalization, canonicalization, hashing
  decorators/     — @cache() decorator
examples/         — Runnable usage examples
tests/            — pytest test suite
```
