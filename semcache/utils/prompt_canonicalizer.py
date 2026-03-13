import re

# Ordered longest-first so more specific phrases are tried before shorter ones.
_PREFIX_PATTERN = re.compile(
    r"^(?:"
    r"please\s+"
    r"|can\s+you\s+"
    r"|could\s+you\s+"
    r"|would\s+you\s+"
    r"|tell\s+me\s+about\s+"
    r"|tell\s+me\s+"
    r"|what\s+is\s+"
    r"|what\s+are\s+"
    r"|whats\s+"
    r"|explain\s+"
    r"|describe\s+"
    r")+",
    re.IGNORECASE,
)


def canonicalize_prompt(prompt: str) -> str:
    """Strip common conversational prefixes from the start of a prompt.

    Multiple stacked prefixes are removed in a single pass (e.g.
    ``"please explain ..."`` → ``"..."``, ``"can you tell me about ..."``
    → ``"..."``).

    If no recognised prefix is found the prompt is returned unchanged.

    Args:
        prompt: Input string (should already be lowercased / trimmed if
                coming from the normalization pipeline).

    Returns:
        Canonicalized prompt with leading filler phrases removed.
    """
    result = _PREFIX_PATTERN.sub("", prompt).strip()
    return result if result else prompt
