"""Conservative extractive fallback; offsets always refer to the original input."""
import re

_FACT = re.compile(r"\d|[٠-٩]|أعلن|أعلنت|بلغ|بلغت|افتتح|وقّع|وقعت|قتل|توفي|أكد|أصدرت|أصدر|سجل|ارتفع|انخفض")
_OPINION = re.compile(r"نتوقع|يتوقع|سيكون|ربما|برأيي|أعتقد|من المتوقع|قد يحدث")


def extract_claims(article: str, limit: int = 3) -> list:
    from .schemas import Claim

    if not isinstance(article, str) or not article.strip() or len(article) > 100_000:
        raise ValueError("أدخل مقالًا نصيًا لا يتجاوز 100000 حرف.")
    claims = []
    seen = set()
    for match in re.finditer(r"[^.!؟?\n]+[.!؟?]?", article):
        raw = match.group()
        sentence = raw.strip()
        if not 6 <= len(sentence.split()) <= 65 or not _FACT.search(sentence):
            continue
        if _OPINION.search(sentence) or sentence in seen:
            continue
        start = match.start() + len(raw) - len(raw.lstrip())
        claims.append(Claim(sentence, start, start + len(sentence)))
        seen.add(sentence)
        if len(claims) >= max(1, min(limit, 3)):
            break
    return claims
