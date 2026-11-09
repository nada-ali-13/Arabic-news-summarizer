# Presentation update — 2026-11-09 (current status)

Existing PR #6 / branch `feature/fact-checking`; no new PR and no changes to main.
Google-only live verification is the default. GDELT requires explicit
`KHULASA_GDELT_ENABLED=1`; NLI remains optional and is not called for Google reviews.

User-reported real Colab results (not rerun by this development environment):
GoogleFactCheck.search returned the AFP record now saved in the offline fixture;
GDELT returned HTTP 429 with a five-second spacing instruction; Hugging Face API
returned 200, MIT license and safetensors availability for the selected model.
The model loaded on CPU with labels 0 entailment / 1 neutral / 2 contradiction.
Three Arabic smoke examples gave 0.9949 entailment, 0.9892 contradiction and
0.8854 neutral. These are smoke tests, not an accuracy benchmark.

The offline AFP fixture preserves the exact claim, publisher, rating and URL
provided by the user. Retrieval date 2026-11-09 is from that real user test;
no review timestamp or new retrieval event is invented. Demo never checks the
input or calls external services. It explicitly identifies its saved provenance.

Executed for this update:

```bash
work/venv/bin/ruff check .
# All checks passed!
GRADIO_ANALYTICS_ENABLED=False HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 \
  work/venv/bin/pytest tests/test_factcheck.py tests/test_inference_app.py tests/test_articles.py -q
# 62 passed in 13.76s; no failures
```

The two small new regression tests verify default Google-only orchestration with
the presentation claim through a mocked Google transport, no NLI/GDELT calls,
separate labels and clickable citation; and explicit GDELT opt-in. Demo integrity
assertions now check the authentic user-provided AFP record. Tests use no live key.
Original summarization UI and article-security tests also passed.

A real local Playwright/Chromium presentation smoke test passed: populated demo,
exact AFP rating/link, retrieval date and saved-record/input disclaimers, original
summarize button and developer credit, and mobile layout without horizontal overflow.
No live external request or NLI download was attempted during this update.
Full production summarization with private weights was not rerun here; its code,
paths, model files and controls remain unchanged.

## Historical initial implementation validation (superseded where noted above)

# Fact-checking validation — 2026-11-09

Base: `main` at `b2dca11513fc455b05cdc2b8d476b85ad6daad49`.
Branch: `feature/fact-checking`. No merge, private weights access or Drive writes.

## Executed checks

Environment: Python 3.12, CPU PyTorch 2.8.0+cpu, Transformers 4.57.1,
Gradio 5.49.1, Ruff 0.14.3. Isolated environment at ignored `work/venv`.

```bash
work/venv/bin/ruff check .
# All checks passed!
GRADIO_ANALYTICS_ENABLED=False HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 work/venv/bin/pytest -q
# 105 passed, 2 warnings in 20.20s (final run; initial full run: 105 passed, 3 warnings in 24.91s)
```

Warnings were existing Transformers Trainer filesystem-mtime fallback warnings.
The full suite includes 20 new fact-check tests and all 85 existing tests.
New tests use synthetic **test-only mocked** API payloads and model relationships;
these fixtures are never presented as live/demo public records. No NLI model was
loaded by these tests. Existing training/numerics tests do execute CPU optimization
on offline tiny random T5 models, not the private production summarizer.

A local Gradio server was started with:

```bash
PYTHONPATH=src:. GRADIO_ANALYTICS_ENABLED=False KHULASA_FACTCHECK_ENABLED=0 \
  work/venv/bin/python -c 'from app import build_app; build_app().launch(server_name="127.0.0.1", server_port=7860)'
```

A real Playwright/Chromium smoke test against `http://127.0.0.1:7860`:
opened the optional accordion, invoked the disabled check, verified the Arabic
unavailable message, switched to demo, verified empty saved-record message,
checked the original summarizer button and developer credit, then checked no
horizontal overflow at 390px mobile width. Passed. Desktop/mobile screenshots
were inspected locally; no model outputs or credentials were captured.
This is a UI smoke test, not a full live fact-check or production summarization run.

`git diff --check` passed. No changes to inference.py, articles.py, notebooks,
training, evaluation reports or weights relative to the base.

## Live probes and unavailable validation

Bounded keyless HTTPS HEAD probes to these hosts were attempted:

- `factchecktools.googleapis.com`: proxy CONNECT 403 (not an API response).
- `api.gdeltproject.org`: proxy CONNECT 403 (not a GDELT response).
- `huggingface.co`: proxy CONNECT 403 (not a model-availability response).

No bypass of network policy was attempted. This runtime has no configured API
secret. The owner's earlier Google HTTP-200/five-record result is user-provided
history, not a test run here. No live API JSON, authentic demo records, NLI model
availability/license/artifact review, Arabic NLI inference, quality metrics or
end-to-end run with the private AraT5v2 weights could be validated here.

## Review gates before production

1. Permit the above services in environment settings and repeat bounded live GET
   requests; inspect actual GDELT language/JSON/date fields and source coverage.
2. Provision the key through Colab Secrets/environment, never chat or source files.
3. Validate the candidate NLI card/license, safetensors and actual config labels;
   test real Arabic premise/hypothesis pairs, latency, memory and abstention.
4. Review approved primary hosts, exact event/date/entity relevance and access terms.
5. Populate demo only from manually validated public records, or retain empty list.
6. Exercise the unchanged production summarizer with authorized local weights.

This is a reviewable implementation with safe fallbacks, not a claim that all
requested production validation has been completed. Conservative lexical matching,
source selection and unresolved temporal/entity semantics limit its recall and
reliability; a more capable entity-aware semantic retrieval system is future work.
