import os
from datetime import datetime, timezone

from .evidence import relevance, safe_url
from .http import JsonClient, SourceError
from .schemas import Evidence


def parse(payload):
    claims = payload.get("claims", [])
    if not isinstance(claims, list):
        raise SourceError("malformed_payload")
    results = []
    for claim in claims[:20]:
        if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
            continue
        reviews = claim.get("claimReview", [])
        if not isinstance(reviews, list):
            continue
        for review in reviews[:5]:
            if not isinstance(review, dict):
                continue
            url = safe_url(review.get("url", ""))
            publisher = review.get("publisher", {})
            if not url or not isinstance(publisher, dict):
                continue
            fields = [publisher.get("name", publisher.get("site", "")),
                      review.get("reviewDate", ""), review.get("textualRating", ""),
                      claim.get("claimDate", ""), review.get("title", "")]
            if not all(isinstance(v, str) for v in fields):
                continue
            results.append(Evidence(
                text=claim["text"][:4000], url=url, publisher=fields[0], date=fields[1],
                kind="factcheck", rating=fields[2], claim_date=fields[3], title=fields[4],
                retrieved_at=datetime.now(timezone.utc).isoformat(),
            ))
    return results


class GoogleFactCheck:
    def __init__(self, client=None):
        self.client = client or JsonClient("google")

    def search(self, claim, deadline):
        key = os.environ.get("FACT_CHECK_API_KEY")
        if not key:
            raise SourceError("missing_FACT_CHECK_API_KEY")
        records = parse(self.client.get({"query": claim[:500], "languageCode": "ar",
                                         "pageSize": 5, "key": key}, deadline))
        return sorted((r for r in records if relevance(claim, r.text)),
                      key=lambda r: relevance(claim, r.text), reverse=True)
