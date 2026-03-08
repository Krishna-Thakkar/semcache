import hashlib


def hash_prompt(normalized_text: str) -> str:
    """Generate a stable SHA-256 hash of a normalized prompt.

    Args:
        normalized_text: A normalized prompt string (output of normalize_prompt).

    Returns:
        64-character lowercase hexadecimal digest.
    """
    return hashlib.sha256(normalized_text.encode()).hexdigest()
