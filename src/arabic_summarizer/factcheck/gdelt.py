import re
from datetime import datetime, timezone
from urllib.parse import urlsplit

from .evidence import safe_url
from .http import JsonClient, SourceError
from .schemas import Evidence


def query_for(claim):
    # Strip all GDELT operators supplied by untrusted article text.
    words = re.findall(r"[\u0621-\u064a\u0660-\u0669\d]+", claim)
    if len(words) < 3:
        raise SourceError("no_arabic_query")
    return " ".join(words[:14]) + " sourcelang:arabic"


def parse(payload):
    articles = payload.get("articles", [])
    if not isinstance(articles, list):
        raise SourceError("malformed_payload")
    result = []
    for item in articles[:5]:
        if not isinstance(item, dict) or item.get("language", "").lower() != "arabic":
            continue
        url = safe_url(item.get("url", ""))
        title, date = item.get("title", ""), item.get("seendate", "")
        if url and isinstance(title, str) and isinstance(date, str):
            result.append(Evidence(text="", title=title[:1000], url=url,
                                   publisher=urlsplit(url).hostname, date=date,
                                   retrieved_at=datetime.now(timezone.utc).isoformat()))
    return result


class Gdelt:
    def __init__(self, client=None):
        self.client = client or JsonClient("gdelt")

    def search(self, claim, deadline):
        return parse(self.client.get({"query": query_for(claim), "mode": "artlist",
                                      "format": "json", "maxrecords": 5,
                                      "timespan": "3months", "sort": "datedesc"}, deadline))
