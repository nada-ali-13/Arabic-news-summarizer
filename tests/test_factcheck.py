import json
import time
from types import SimpleNamespace

import pytest

from arabic_summarizer.factcheck.claims import extract_claims
from arabic_summarizer.factcheck.demo import LABEL, load_records
from arabic_summarizer.factcheck.evidence import deduplicate, relevance, safe_url
from arabic_summarizer.factcheck.gdelt import parse as parse_gdelt
from arabic_summarizer.factcheck.gdelt import query_for
from arabic_summarizer.factcheck.google_factcheck import GoogleFactCheck, parse
from arabic_summarizer.factcheck.http import JsonClient, SourceError
from arabic_summarizer.factcheck.nli import Nli, NliUnavailable, label_mapping
from arabic_summarizer.factcheck.pipeline import Pipeline
from arabic_summarizer.factcheck.schemas import Assessment, Claim, Evidence, Verdict
from arabic_summarizer.factcheck.ui import demo_html, render

CLAIM = "أعلنت وزارة الصحة اليمنية افتتاح مركز طبي جديد في صنعاء."


def test_claims_are_bounded_original_spans_and_skip_opinion():
    article = "أعتقد أن هذا القرار سيكون الأفضل.\n " + CLAIM + "\n" + CLAIM
    claims = extract_claims(article)
    assert len(claims) == 1
    assert article[claims[0].start:claims[0].end] == CLAIM
    assert len(extract_claims("\n".join(CLAIM.rstrip(".") + str(i) for i in range(10)))) == 3
    with pytest.raises(ValueError):
        extract_claims(" ")


@pytest.mark.parametrize("other", [
    "أعلنت وزارة التعليم المصرية افتتاح جامعة جديدة في القاهرة.",
    CLAIM.replace("صنعاء", "عدن").replace("اليمنية", "المصرية"),
    CLAIM + " عام 2020", "لم " + CLAIM,
])
def test_irrelevant_context_rejected(other):
    assert relevance(CLAIM, other) == 0


def test_exact_and_numeric_normalization():
    assert relevance(CLAIM, CLAIM) == 1
    assert relevance(CLAIM + " ٢٠٢٦", CLAIM + " 2026") == 1


def test_google_metadata_multiple_reviews_and_bad_payloads():
    payload = {"claims": [{"text": CLAIM, "claimDate": "2026-01-01", "claimReview": [
        {"publisher": {"name": "جهة"}, "url": "https://example.org/review",
         "reviewDate": "2026-01-02", "textualRating": "مضلل جزئيًا", "title": "مراجعة"},
        {"url": "javascript:alert(1)"}]}]}
    records = parse(payload)
    assert len(records) == 1
    assert records[0].rating == "مضلل جزئيًا"
    assert records[0].claim_date == "2026-01-01"
    assert records[0].publisher == "جهة" and records[0].retrieved_at
    assert parse({}) == []
    with pytest.raises(SourceError):
        parse({"claims": {}})


def test_missing_key_never_calls_transport(monkeypatch):
    monkeypatch.delenv("FACT_CHECK_API_KEY", raising=False)
    with pytest.raises(SourceError, match="missing_FACT_CHECK_API_KEY"):
        GoogleFactCheck().search(CLAIM, time.monotonic() + 10)


def test_gdelt_arabic_and_injection():
    query = query_for(CLAIM + ' " OR domain:evil.com')
    assert query.endswith("sourcelang:arabic") and "OR" not in query and '"' not in query
    item = {"url": "https://example.org/a", "title": CLAIM, "language": "Arabic",
            "seendate": "20260101T000000Z"}
    assert parse_gdelt({"articles": [item]})[0].text == ""
    assert parse_gdelt({"articles": [{**item, "language": "English"}]}) == []
    assert parse_gdelt({}) == []
    with pytest.raises(SourceError):
        parse_gdelt({"articles": "error"})


class Response:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self.headers = headers or {"Content-Type": "application/json"}
        self.payload = payload if payload is not None else {}
        self.closed = False

    def iter_content(self, _):
        yield json.dumps(self.payload).encode()

    def close(self):
        self.closed = True


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def transport(responses):
    now = [100.0]
    def sleep(seconds):
        now[0] += seconds
    JsonClient._last.clear()
    session = Session(responses)
    return JsonClient("google", session, clock=lambda: now[0], sleep=sleep), session


@pytest.mark.parametrize("status,code", [(401, "authentication"), (403, "authentication"),
                                         (302, "http_error")])
def test_api_errors_redacted(status, code):
    client, session = transport([Response(status)])
    with pytest.raises(SourceError, match=code) as exc:
        client.get({"key": "SECRET"}, 150)
    assert "SECRET" not in str(exc.value)
    assert session.calls[0][1]["allow_redirects"] is False


def test_backoff_cache_and_quota():
    client, session = transport([Response(429), Response(503), Response(payload={"claims": []})])
    assert client.get({"query": CLAIM, "key": "SECRET"}, 150) == {"claims": []}
    assert len(session.calls) == 3
    client.get({"query": CLAIM, "key": "SECRET"}, 150)
    assert len(session.calls) == 3
    assert "SECRET" not in str(client.cache)
    client, _ = transport([Response(429), Response(429), Response(429)])
    with pytest.raises(SourceError, match="quota"):
        client.get({}, 150)


def test_timeout_non_json_malformed_and_budget():
    client, _ = transport([TimeoutError("SECRET")] * 3)
    with pytest.raises(SourceError, match="network_or_payload_error"):
        client.get({}, 150)
    client, _ = transport([Response(headers={"Content-Type": "text/html"})])
    with pytest.raises(SourceError, match="non_json"):
        client.get({}, 150)
    client, _ = transport([Response(payload=[])])
    with pytest.raises(SourceError, match="malformed"):
        client.get({}, 150)
    with pytest.raises(SourceError, match="time_budget"):
        client.get({}, 100)


def test_label_mapping_is_config_derived_and_disabled(monkeypatch):
    assert label_mapping(SimpleNamespace(id2label={0: "neutral", 1: "contradiction",
                                                  2: "entailment"}))[2] == "entailment"
    with pytest.raises(NliUnavailable):
        label_mapping(SimpleNamespace(id2label={0: "LABEL_0"}))
    monkeypatch.delenv("KHULASA_NLI_ENABLED", raising=False)
    with pytest.raises(NliUnavailable):
        Nli().assess(CLAIM, CLAIM)


class Search:
    def __init__(self, records):
        self.records = records

    def search(self, *args):
        return self.records


def test_no_evidence_and_factchecker_rating_do_not_become_verdict():
    pipeline = Pipeline(google=Search([]), gdelt=Search([]))
    results, _ = pipeline.verify(CLAIM)
    assert results[0].verdict == Verdict.INSUFFICIENT_EVIDENCE
    review = Evidence(CLAIM, "https://example.org/review", "جهة", kind="factcheck", rating="صحيح")
    pipeline.google = Search([review])
    result = pipeline.verify(CLAIM)[0][0]
    assert result.verdict == Verdict.INSUFFICIENT_EVIDENCE and not result.relations


def test_conflicting_evidence_and_headlines(monkeypatch):
    monkeypatch.setenv("KHULASA_GDELT_ENABLED", "1")
    monkeypatch.setenv("KHULASA_EVIDENCE_HOSTS", "example.org,example.net")
    records = [Evidence("", "https://example.org/a", "أ"),
               Evidence("", "https://example.net/b", "ب")]
    # Isolate orchestration from conservative extraction/deduplication for this conflict test.
    monkeypatch.setattr("arabic_summarizer.factcheck.pipeline.relevance", lambda a, b: 1 if a == CLAIM else 0)
    counter = iter(["entailment", "contradiction"])
    nli = SimpleNamespace(assess=lambda *args: next(counter))
    # Empty-text discovery results must survive deduplication by URL.
    pipeline = Pipeline(google=Search([]), gdelt=Search(records), nli=nli,
                        extract=lambda _: CLAIM)
    # Give passages distinct content to represent independent contradictory evidence.
    original = pipeline.extract
    pipeline.extract = lambda url: original(url).rstrip(".") + (" إعلان رسمي." if ".org" in url else " نفي رسمي.")
    results, _ = pipeline.verify(CLAIM)
    assert results[0].verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert any("تعارض" in note for note in results[0].notes)


def test_disabled_and_demo_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("KHULASA_FACTCHECK_ENABLED", "0")
    assert Pipeline().verify(CLAIM)[0] == []
    records = load_records()
    assert len(records) == 1 and LABEL in demo_html()
    assert records[0]["retrieved_at"] == "2026-11-09"
    assert records[0]["publisher"] == "في ميزان فرانس برس"
    assert records[0]["rating"] == "مضلّل"
    assert records[0]["url"] == "https://factcheckarabic.afp.com/doc.afp.com.B3FY2T3"
    assert "ولا تحققًا من النص المُدخل" in demo_html()
    bad = tmp_path / "bad.json"
    bad.write_text('[{"text":"invented"}]')
    with pytest.raises(ValueError):
        load_records(bad)


def test_safe_render_and_syndication():
    a = Evidence(CLAIM, "https://example.org/a", "<script>bad</script>",
                 kind="factcheck", rating="تقييم منشور")
    b = Evidence(CLAIM, "https://example.net/b", "ب")
    assert len(deduplicate([a, b])) == 1
    rendered = render([Assessment(Claim(CLAIM, 0, len(CLAIM)), evidence=[a])], [])
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered
    assert safe_url("http://127.0.0.1/a") == ""
    assert safe_url("https://user:pass@example.org/a") == ""


def test_ui_independent_and_original_summarizer(monkeypatch):
    from app import build_app
    from arabic_summarizer.inference import Summarizer

    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.setenv("KHULASA_FACTCHECK_ENABLED", "0")
    monkeypatch.setattr(Summarizer, "load", lambda _: pytest.fail("Unexpected load"))
    demo = build_app()
    assert "التحقق من الخبر" in str(demo.config)
    handler = next(fn.fn for fn in demo.fns.values() if fn.fn.__name__ == "verify")
    assert "معطّل" in handler(CLAIM, "", False)
    assert LABEL in handler("", "", True)
    demo.close()



def test_google_only_presentation_claim(monkeypatch):
    monkeypatch.delenv("KHULASA_GDELT_ENABLED", raising=False)
    monkeypatch.delenv("KHULASA_NLI_ENABLED", raising=False)
    monkeypatch.setenv("FACT_CHECK_API_KEY", "test-only-key")
    record = load_records()[0]
    payload = {"claims": [{"text": record["text"], "claimReview": [{
        "publisher": {"name": record["publisher"]}, "textualRating": record["rating"],
        "url": record["url"],
    }]}]}
    client, session = transport([Response(payload=payload)])
    # The fake transport clock is isolated; use a real future deadline for API caching.
    google = GoogleFactCheck(client)
    def forbidden(*args):
        pytest.fail("Google-only mode must not call GDELT or NLI")
    pipeline = Pipeline(google=google, gdelt=SimpleNamespace(search=forbidden),
                        nli=SimpleNamespace(assess=forbidden))
    results, notes = pipeline.verify(record["text"])
    assert len(session.calls) == 1 and len(results) == 1
    assert results[0].verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert not results[0].relations
    html = render(results, notes)
    assert "عُثر على تقرير تدقيق مطابق" in html
    assert record["publisher"] in html and record["rating"] in html
    assert f'href="{record["url"]}"' in html
    assert "هذا تقييم جهة التدقيق المنشور" in html
    assert "NLI" not in html and "أدلة غير كافية" not in html
    assert "موضع الجملة" not in html and "علاقة" not in html
    assert 'class="kh-badge"' in html
    assert "عرض تقرير التحقق الأصلي" in html


def test_gdelt_remains_opt_in(monkeypatch):
    monkeypatch.setenv("KHULASA_GDELT_ENABLED", "1")
    called = []
    gdelt = SimpleNamespace(search=lambda *args: called.append(True) or [])
    Pipeline(google=Search([]), gdelt=gdelt).verify(CLAIM)
    assert called == [True]



@pytest.mark.parametrize("evidence", [
    [],
    [Evidence("تقرير لا يتعلق بالادعاء", "https://example.org/a", "جهة",
              kind="factcheck", rating="صحيح")],
    [Evidence(CLAIM, "javascript:alert(1)", "جهة", kind="factcheck", rating="صحيح")],
    [Evidence(CLAIM, "https://example.org/a", "جهة", kind="factcheck", rating="")],
    [Evidence(CLAIM, "https://example.org/a", "جهة", rating="صحيح")],
])
def test_ui_neutral_without_suitable_published_rating(evidence):
    result = Assessment(Claim(CLAIM, 42, 99), evidence=evidence)
    html = render([result], ["Google: missing_FACT_CHECK_API_KEY"])
    assert "لم تتوفر معلومات كافية للتحقق" in html
    assert 'class="kh-badge kh-neutral"' in html
    assert "عرض تقرير التحقق الأصلي" not in html
    assert "missing_FACT_CHECK_API_KEY" not in html
    assert "أدلة غير كافية" not in html


def test_ui_unknown_rating_is_preserved_and_escaped():
    rating = 'غير محسوم <img src=x onerror=alert(1)>'
    review = Evidence(CLAIM, "https://example.org/review", "جهة", kind="factcheck",
                      rating=rating, date="2026-11-09T12:00:00Z")
    html = render([Assessment(Claim(CLAIM, 42, 99), evidence=[review])], [])
    assert 'غير محسوم &lt;img src=x onerror=alert(1)&gt;' in html
    assert "<img" not in html and "2026-11-09T" not in html
    assert html.count('class="kh-link"') == 1
    assert "نتيجة التحقق من الخبر" in html
    assert "لم تتوفر معلومات كافية للتحقق" in render([], [])
