"""Conservative normalization: retain letters, hamza, diacritics and numbers."""

import re
import unicodedata


def normalize(text):
    if not isinstance(text, str):
        raise ValueError("النص يجب أن يكون سلسلة نصية.")
    text = unicodedata.normalize("NFC", text).replace("ـ", "")
    # Remove bidi formatting and zero-width controls, retaining ordinary whitespace.
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    text = "".join(c for c in text if not unicodedata.category(c).startswith("C") or c.isspace())
    return re.sub(r"\s+", " ", text).strip()


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!؟!?])\s+|\n+", text) if s.strip()]


def word_count(text):
    return len(normalize(text).split())
