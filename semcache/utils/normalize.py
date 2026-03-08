import re
import string


def normalize_prompt(text: str) -> str:
    """Normalize a prompt for consistent caching.

    Steps:
    1. Lowercase
    2. Strip leading/trailing whitespace
    3. Remove punctuation
    4. Collapse multiple spaces to a single space

    Args:
        text: Raw prompt string.

    Returns:
        Normalized prompt string.
    """
    text = text.lower()
    text = text.strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text
