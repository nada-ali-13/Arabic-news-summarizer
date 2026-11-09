"""Conservative lexical relevance, not learned entity linking or semantic retrieval."""
import ipaddress
import re
from difflib import SequenceMatcher
from urllib.parse import urlsplit, urlunsplit


def safe_url(value):
    try:
        p = urlsplit(value)
        if p.scheme != "https" or not p.hostname or p.username or p.password:
            return ""
        if p.port not in (None, 443) or any(ord(c) <= 32 for c in value):
            return ""
        host = p.hostname.lower().rstrip(".")
        if host == "localhost" or "." not in host or host.endswith((".local", ".internal")):
            return ""
        try:
            if not ipaddress.ip_address(host).is_global:
                return ""
        except ValueError:
            pass
        return urlunsplit(("https", p.netloc, p.path or "/", p.query, ""))
    except (ValueError, TypeError):
        return ""


def tokens(text):
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩أإآى", "0123456789اااي"))
    return set(re.findall(r"[\w]+", re.sub(r"[\u064b-\u065fـ]", "", text).lower()))


def relevance(claim, passage):
    a, b = tokens(claim), tokens(passage)
    # Unknown/mismatched quantities and dates fail closed. This can reject real contradictions.
    def numbers(ts):
        return {t for t in ts if any(c.isdigit() for c in t)}
    if numbers(a) != numbers(b):
        return 0.0
    neg = {"لا", "لم", "لن", "ليس", "نفت", "نفى"}
    if bool(a & neg) != bool(b & neg):
        return 0.0
    content = a - {"في", "من", "على", "ان", "الى", "عن", "هذا", "هذه", "تم"}
    overlap = len(content & b) / max(1, len(content))
    return overlap if len(content & b) >= 5 and overlap == 1.0 else 0.0


def deduplicate(records):
    result, urls = [], set()
    for record in records:
        url = safe_url(record.url)
        canonical = urlsplit(url)._replace(query="", fragment="").geturl()
        if not url or canonical in urls:
            continue
        if record.text and any(
            old.text and SequenceMatcher(None, record.text, old.text).ratio() > .85
            for old in result
        ):
            continue
        urls.add(canonical)
        result.append(record)
    return result
