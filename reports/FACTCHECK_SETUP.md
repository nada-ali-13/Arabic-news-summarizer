# خُلاصة: التحقق المستند إلى الأدلة

## Status and scope

This branch adds an optional, conservative evidence-assessment feature to `app.py`.
It is **not a validated fake-news classifier**. No article-level true/fake verdict,
accuracy measurement, or factual confidence percentage is produced. The original
summarizer, weights, training/evaluation notebooks and private Drive files are unchanged.

Live Google access was previously verified by the owner, not in this implementation
session. No key was supplied here. GDELT and Hugging Face HTTPS probes on 2026-11-09
were denied by the environment proxy (CONNECT 403). Consequently GDELT's current
endpoint format, the proposed model's availability/license and real NLI inference
are **not independently validated here**. Keep this PR unmerged until these live
checks and human review are complete. No demo records were fabricated.

## Architecture

`src/arabic_summarizer/factcheck/` is optional and imported on first verification:

- `claims.py`: deterministic extractive Arabic sentence rules, original character
  offsets, maximum three claims. Numbers and reporting verbs are hints. Opinions
  and explicit predictions are excluded heuristically. No learned extraction or
  entity linking is claimed. Compound sentences are not decomposed; source ambiguity
  and relative dates remain limitations.
- `google_factcheck.py`: individual-claim search at the official
  `https://factchecktools.googleapis.com/v1alpha1/claims:search` endpoint;
  `query`, `languageCode=ar`, `pageSize=5`; parses all returned reviews (bounded),
  claim/review dates, original ratings, publisher, title and citation.
- `gdelt.py`: public `https://api.gdeltproject.org/api/v2/doc/doc`, `mode=artlist`,
  `format=json`, `maxrecords=5`, `timespan=3months`, `sort=datedesc`,
  `sourcelang:arabic`. Only explicitly Arabic results are accepted. `seendate`
  is GDELT's observation timestamp, **not guaranteed original publication time**.
  Missing dates remain unknown. Empty/non-JSON/English-only results yield no evidence.
  These documented API conventions still require a live smoke test in this environment.
- `http.py`: fixed-endpoint transport, no redirects, TLS verification, bounded JSON,
  6-second request timeout, at most three attempts, bounded backoff and Retry-After,
  conservative per-process throttles and 128-entry/15-minute caches. No credentials
  in cache keys, messages, or logs. Auth failures are not retried. Cache is in RAM.
- `evidence.py`: conservative normalized lexical coverage, exact numeric/date-token
  matching and negation checks. Every significant claim token must be present;
  this rejects many paraphrases and contradictions intentionally. No semantic
  ranker, Arabic NER or robust event/coreference resolver is implemented. Same-topic
  coverage is insufficient. Google records are ranked only after this filter.
  URL/near-identical text deduplication reduces syndication; it cannot establish
  independence or identify every copied report. Search rankings are not proof.
- `pipeline.py`: 60-second search/extraction scheduling budget, one active check,
  fail-closed fallbacks and explicit conflict handling. Fact-check ratings remain
  citations only, never premises for NLI. Headlines are discovery metadata only.
  Full text uses existing DNS-pinned SSRF-safe article extraction (redirect checks,
  size/content-type limits, timeouts). Only operator-approved exact HTTPS hosts
  can supply NLI premises. Select public sources whose access is permitted; no
  paywall, robots or authentication bypass. Primary-source selection is manual.
- `nli.py`: optional lazy CPU-only local inference and config-derived label mapping;
  model reused, no GPU allocation and no interactive downloads. Unknown label
  mappings fail closed. Failed loading is latched until app restart.
- `ui.py`: escaped RTL HTML, citations, source ratings, evidence relationships and
  limitations; separate button/output, no change to summarization callbacks.
- `demo.py` and `fixtures/demo.json`: isolated offline presentation records.

Google discovers previously published fact checks; GDELT discovers news coverage.
Neither provides an oracle of truth. Google claim text is the **claim being reviewed**,
not the review's explanatory evidence; passing it to NLI would spuriously entail itself.

## Colab setup

Use a fresh runtime and an authorized copy of the original summarizer weights.
Do not retrain or overwrite weights. The existing Gradio notebook embeds its old
interface and pins old helpers; it is deliberately unchanged. To use this feature,
check out this branch and run `app.py` or import `build_app` from this checkout:

```python
# Colab shell cell, after cloning the repository:
%cd /content/Arabic-news-summarizer
!git fetch origin feature/fact-checking
!git checkout feature/fact-checking
!pip install -e .
```

Configure the key in **Colab Secrets**, grant this notebook access, then:

```python
import os
from google.colab import userdata
os.environ['FACT_CHECK_API_KEY'] = userdata.get('FACT_CHECK_API_KEY')
os.environ['ARABIC_SUMMARIZER_MODEL'] = '/content/drive/MyDrive/YOUR_AUTHORIZED_MODEL/model'
# Set only reviewed primary/news hosts; exact names, no wildcards:
os.environ['KHULASA_EVIDENCE_HOSTS'] = ''
```

Never paste the key in chat, notebooks, URLs, commits or screenshots. The code reads
only the environment; a missing key shows a safe diagnostic and GDELT may still run.
Do not alter API restrictions. Key stays restricted to Fact Check Tools API.

### NLI candidate and activation gate

Candidate identifier: `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`.
This family is a compact multilingual MiniLM/XNLI classifier with Arabic among
XNLI languages. License is **unverified here**; confirm it directly on the model card before use.
Do not treat the identifier, license, artifact format or parameter count as verified
by this session. Review the exact model card/config/files first:
https://huggingface.co/MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli

Provision model/tokenizer to the Hugging Face cache separately, only after checking
license, Arabic coverage, safetensors availability and `id2label`. If the repository
lacks safetensors, the current loader intentionally refuses pickle weights. Resolve
that with reviewed safe conversion or select/review another model before activation;
do not silently allow unsafe remote code or change labels. No `trust_remote_code`.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from arabic_summarizer.factcheck.nli import MODEL_ID, label_mapping
AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, use_safetensors=True)
print(label_mapping(model.config))  # must contain entailment, neutral, contradiction
# Run real Arabic entailment/contradiction/neutral examples and human review here.
del model
os.environ['KHULASA_NLI_ENABLED'] = '1'
```

Start with CPU NLI (default), batch 1, sequence limit 384; it does not compete for
AraT5v2 GPU memory. A T4 remains available for the summarizer. Plan conservatively
for 1–2 GB extra host RAM for weights/framework/activations; this is a planning
allowance, **not measured model memory or latency**. Profile in your actual Colab
runtime. The search budget is not a hard cancellation deadline for local model
loading/inference, DNS or a blocking socket read; initial provisioning is separate.
CPU inference can extend latency. Leave NLI disabled until validated.

```python
from app import build_app
build_app().launch(share=True)  # temporary public link only if desired
```

Set approved evidence hosts before running to enable full-text assessment. With
no hosts or NLI disabled, the UI provides discovery/citations and insufficient
evidence, never invented model output. NLI score threshold .9 is a conservative
uncalibrated abstention rule, not an accuracy estimate. An entailment relation is
not proof of real-world truth. Dates, entity identity and claims with no explicit
year require human inspection; temporal relevance is not fully solved.

## Quotas and service errors

The owner's console showed **300 AllRequests/minute**, approximately 0.33% observed
usage. This is neither an unlimited daily quota nor a pricing/free-use guarantee.
Google throttle is one attempt per 2 seconds (at most 30/minute per process);
GDELT uses 6 seconds. Concurrent/too-fast calls abstain rather than wait in a queue.
Up to 3 claims × 2 source queries, each with at most 3 attempts, are scheduled
within the shared budget. No pagination. Article retrieval is additionally bounded.
Multiple app processes share no cache/throttle; use a gateway or shared limiter
before multi-worker deployment. Errors report safe codes (`quota`, `local_rate_limit`,
`authentication_or_access`, `network_or_payload_error`, `non_json_response`,
`time_budget`), not provider bodies or keyed URLs. Adjust limits if quotas change.

## Demo

Toggle **وضع العرض — نتائج محفوظة وليست تحققًا مباشرًا**. It does no API calls,
article fetching or NLI. The fixture is currently an empty list; the UI says so.
To populate it, retrieve real public records in a permitted environment, manually
validate the original publication and rating, then save objects with nonempty
`text`, `publisher`, `rating`, `url`, `retrieved_at` (ISO UTC retrieval date), plus
review/claim dates where available. Preserve original text/rating; remove keys,
request URLs and personal data not necessary to the public claim. Review provenance
before committing. Structural validation cannot establish authenticity by itself.
Do not add synthetic claims or results. Saved examples never verify the entered article.

## Testing and rollback

```bash
python -m pip install -e . -r requirements-dev.txt
ruff check .
pytest -q
```

`tests/test_factcheck.py` uses mocked responses/model relationships, including auth,
quota, retries, malformed/empty results, relevance, conflicts, HTML escaping, demo
integrity, opt-out and real Gradio construction. Existing summarizer tests exercise
mock generation and an offline tiny random T5; neither validates private AraT5v2
weights. See `reports/FACTCHECK_VALIDATION.md` for actual commands and results.

Disable with `KHULASA_FACTCHECK_ENABLED=0`. Summarization does not initialize the
fact-check subsystem. Restore the original application by checking out `main` in
a clean checkout after saving any uncommitted work; no weights need restoration.
Never merge this feature automatically.
