import re
from underthesea import word_tokenize


def normalize_text(text: str) -> str:
    """Lowercase, remove duplicate spaces, and strip whitespace."""
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    """Tokenize Vietnamese text using underthesea.word_tokenize."""
    text = normalize_text(text)
    if not text:
        return []
    return word_tokenize(text)
