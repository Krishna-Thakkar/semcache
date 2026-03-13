import re

# Patterns matched in order; the first capture group is the extracted question.
_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(?:Question|Q|User)\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE),
]


def extract_question(prompt: str) -> str:
    """Extract the question portion from a RAG-style prompt.

    Looks for common prefixes such as ``Question:``, ``Q:``, or ``User:``
    and returns the text that follows. If no pattern is matched the original
    prompt is returned unchanged.

    Args:
        prompt: Raw prompt, possibly containing context and a question marker.

    Returns:
        Extracted question string, or *prompt* if no marker is found.
    """
    for pattern in _PATTERNS:
        match = pattern.search(prompt)
        if match:
            return match.group(1).strip()
    return prompt
