from pathlib import Path


def ensure_cache_dir(path: str | None = None) -> Path:
    """Create the .semcache storage directory if it does not exist.

    Args:
        path: Optional custom directory path. Defaults to .semcache in the
              current working directory.

    Returns:
        The resolved Path to the storage directory.
    """
    if path is not None:
        cache_dir = Path(path)
    else:
        cache_dir = Path(".semcache")

    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
