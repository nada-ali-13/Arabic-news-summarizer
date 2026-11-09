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
