import os
import threading
import time
from dataclasses import replace
from urllib.parse import urlsplit

from arabic_summarizer.articles import extract_article

from .claims import extract_claims
from .evidence import deduplicate, relevance
from .gdelt import Gdelt
from .google_factcheck import GoogleFactCheck
from .http import SourceError
from .nli import Nli, NliUnavailable
from .schemas import Assessment, Verdict


class Pipeline:
    def __init__(self, google=None, gdelt=None, nli=None, extract=None):
        self.google = google or GoogleFactCheck()
        self.gdelt = gdelt or Gdelt()
        self.nli = nli or Nli()
        self.extract = extract or extract_article
        self.lock = threading.Lock()

    def verify(self, article):
        if os.environ.get("KHULASA_FACTCHECK_ENABLED", "1") == "0":
            return [], ["التحقق معطّل؛ يمكنك استخدام التلخيص."]
        if not self.lock.acquire(blocking=False):
            return [], ["التحقق مشغول؛ حاول بعد قليل."]
        try:
            return self._verify(article)
        finally:
            self.lock.release()

    def _verify(self, article):
        deadline = time.monotonic() + 60
        results = []
        trusted = {h.strip().lower() for h in os.environ.get(
            "KHULASA_EVIDENCE_HOSTS", "").split(",") if h.strip()}
        sources = [("Google", self.google)]
        if os.environ.get("KHULASA_GDELT_ENABLED", "0") == "1":
            sources.append(("GDELT", self.gdelt))
        for claim in extract_claims(article):
            result = Assessment(claim)
            records = []
            for source, client in sources:
                if time.monotonic() >= deadline - 1:
                    result.notes.append("انتهت ميزانية البحث.")
                    break
                try:
                    records.extend(client.search(claim.text, deadline))
                except SourceError as exc:
                    result.notes.append(f"{source}: {exc}")
                except Exception:
                    result.notes.append(f"{source}: تعذر البحث.")
            for record in deduplicate(records)[:6]:
                if record.kind == "factcheck":
                    # Original claim + published rating are NOT a factual NLI premise.
                    result.evidence.append(record)
                    continue
                if len(result.evidence) >= 4 or time.monotonic() + 21 >= deadline:
                    continue
                # Explicit exact-host allowlist represents human source selection,
                # not a claim that a publisher is always correct.
                if urlsplit(record.url).hostname not in trusted:
                    result.notes.append("اكتشاف تغطية فقط: المصدر غير معتمد لفحص NLI.")
                    result.evidence.append(record)
                    continue
                try:
                    body = self.extract(record.url)
                    candidates = [c.text for c in extract_claims(body)]
                    passages = [p for p in candidates if relevance(claim.text, p)]
                    if not passages:
                        continue
                    record = replace(record, text=max(passages,
                                     key=lambda p: relevance(claim.text, p)))
                    if any(relevance(record.text, r.text) for r in result.evidence if r.text):
                        continue
                    result.evidence.append(record)
                    if time.monotonic() >= deadline:
                        continue
                    relation = self.nli.assess(record.text, claim.text)
                    result.relations.append((record.url, relation))
                except NliUnavailable as exc:
                    result.notes.append(str(exc))
                except Exception:
                    result.notes.append("تعذر استخراج دليل نصي آمن من المصدر.")
            relations = {relation for _, relation in result.relations}
            if {"entailment", "contradiction"} <= relations:
                result.notes.append("تعارض بين الأدلة؛ يلزم تدقيق بشري.")
            elif "entailment" in relations:
                result.verdict = Verdict.SUPPORTED
            elif "contradiction" in relations:
                result.verdict = Verdict.REFUTED
            if not result.evidence:
                result.notes.append("لم تُسترجع أدلة مناسبة؛ غياب الأدلة لا يعني كذب الادعاء.")
            results.append(result)
        return results, ([] if results else ["لم نجد جملة خبرية قابلة للفحص بالقواعد الحالية."])
