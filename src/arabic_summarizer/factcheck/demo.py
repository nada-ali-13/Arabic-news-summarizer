"""Isolated offline records. Empty until authenticated public records are curated."""
import json
from pathlib import Path

from .evidence import safe_url

LABEL = "وضع العرض — نتائج محفوظة وليست تحققًا مباشرًا"


def load_records(path=None):
    path = path or Path(__file__).with_name("fixtures") / "demo.json"
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Invalid demo fixture")
    required = ("text", "publisher", "rating", "url", "retrieved_at")
    for record in records:
        if not isinstance(record, dict) or not all(
            isinstance(record.get(key), str) and record[key] for key in required
        ) or not safe_url(record["url"]):
            raise ValueError("Unvalidated demo record")
    return records
