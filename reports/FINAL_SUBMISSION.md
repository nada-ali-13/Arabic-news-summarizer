# خُلاصة — final submission record

## Source and audit decision

Prepared from `main` at `d1b24cbeef87aa61e45693fd5f7895a9f8db6f05`, on branch `codex/khulasa-final-submission`, using the supplied `Khulasa_GitHub_Handoff.zip` and `CODEX_TASK.txt`. PR #3 (storage recovery) and PR #4 (earlier 200-step workflow) remain unmerged and were not used as the branch base.

Before edits, the repository tree, imports, README, tests, old notebook/reports, final notebook cells and UI source were reviewed. The following rationale was recorded before changes:

| Decision | Exact paths | Reason |
|---|---|---|
| Keep | `src/arabic_summarizer/{text,data,metrics,inference,articles,numerics,io}.py` | Final notebooks import these helpers; preserve preprocessing, Arabic ROUGE, numerical and URL safeguards. |
| Keep as historical code | `src/arabic_summarizer/{training,checkpoints,config}.py`, `configs/dev.json`, `configs/full.json` | Existing tests and training imports depend on them; no reason to delete functioning code. The final notebooks do not invoke the old training/recovery runner. |
| Label legacy in place | `notebooks/Arabic_News_Summarizer_Training.ipynb`, `reports/{COLAB_RESTART,FULL_TRAINING,NUMERICS_VALIDATION,VALIDATION,QUALITATIVE_REVIEW}.md` | Preserve earlier work, links and bootstrap tests while directing readers to the final workflow. |
| Keep | `tests/`, `scripts/`, `.github/workflows/tests.yml`, `reports/{pytest.xml,pytest-numerics.xml,T4_SMOKE_RESULT.json,COMPATIBILITY.md}` | Regression checks, utilities, historical evidence and source/licensing references remain useful; old XML files are not the current validation report. |
| Update | `app.py`, `README.md`, `requirements.txt`, `pyproject.toml`, `deployment/README.md` | Final product UI, accurate submission narrative, matching dependency pins and optional deployment caveat. |
| Add | Three supplied final notebooks, `reports/metrics_summary.json`, this record, `tests/test_notebooks.py` | Authoritative final artifacts and lightweight validation. |
| Archive/remove | **None** | Dependency audit supports retaining files in place. No code, models, Drive files, Trash or Git history deleted. |

All code and markdown cell sources in the final notebooks match the handoff exactly. The metrics JSON and evaluation/demo notebooks are byte-for-byte copies. Training notebook cell outputs were already empty, but a final metadata audit found 121 stale progress-widget states and Colab output references: only that rendering metadata was cleared. Training parameters and source pins are unchanged. `app.py` adapts `interface/final_gradio_cell.py` into importable `build_app(summarizer=None)`, removes the notebook-global requirement and automatic public launch, adds a small mobile CSS rule, and retains CLI launch, bounded queue, URL extraction and Arabic errors. The root inference loader uses FP32 on CPU/CUDA to match the final demo notebook; no full model was loaded to validate performance or memory.

The old example gallery is removed from the UI. Header, warning footer and separate visible author credit come from the final source. Requirements pin protobuf 6.31.1, safetensors 0.6.2 and urllib3 2.5.0 to the supplied environment. TensorFlow/Flax are not dependencies and are disabled before the app lazily imports Transformers; notebooks retain their own existing environment setup. Keep Colab's CUDA PyTorch installation.

## Evidence and limits

The scores in `metrics_summary.json` are supplied evaluation evidence, not values recomputed during repository preparation. Test uses 300 seeded examples, validation uses a separate 100; test includes the earlier inspected 64-example subset. The same rows are compared for both models. ROUGE is lexical overlap, not accuracy or factual correctness; manual factual labels remain blank. No full article review data or private Drive sharing URLs/tokens were added.

Codex has **not live-tested the final demo on the user's Colab**. Trained weights are private and absent from GitHub; collaborators require permitted access to the complete export and must configure `MODEL_DIR` or `ARABIC_SUMMARIZER_MODEL`. No project retraining, full model load, large model-weight download, Drive access, permanent hosting or PR merge was performed. Temporary Gradio sharing is not deployment.

## Local validation

Validated on 2026-11-07 with Python 3.12.14, PyTorch 2.8.0+cpu, Transformers 4.57.1, Gradio 5.49.1, Datasets 4.3.0, protobuf 6.31.1, pytest 9.2.0, nbformat 5.11.1 and Ruff 0.14.3 in an isolated virtual environment.

| Check actually run | Result |
|---|---|
| Full existing/new lightweight suite, `USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False pytest -q` | **85 passed in 9.80 s**. Includes existing offline tiny random-T5 numerical/training tests, URL security fixtures, mock inference, real Gradio build/default lazy-load smoke, and four notebook schema/code-syntax checks. |
| `ruff check .` | Passed. Only the supplied notebooks have targeted exceptions for import ordering/placement, compact statements and unused imports; other rules remain active. Their cell sources are preserved. |
| `python -m compileall -q src app.py scripts` | Passed. |
| `python -m pip check` | No broken requirements. |
| `python app.py --help` | CLI parsed and displayed the optional temporary-share flag. |
| `git diff --check` | Passed. |
| Handoff source/hash comparisons | All final code/markdown cell sources unchanged; exact metrics JSON confirmed. |
| Local Chromium/Playwright UI smoke with an explicit mock summarizer | Passed header/credit visibility, submission, statistics, clear, RTL, and layouts at 1366 px and 390 px. Mobile cards stack without horizontal overflow; no browser page errors. No screenshot added to the repository or presented as trained-model output. |

The three warnings in pytest are Transformers' existing checkpoint mtime-order fallback warnings in tiny-model tests, not failures. Notebook cells were parsed/compiled, never executed. URL extraction was tested through existing controlled fixtures and the UI's mocked URL handler, not a live news-site request. Browser validation uses mock results and does not establish trained-model quality, CUDA memory/latency, or actual Colab/Drive execution.

### Handoff integrity

All final notebook code and markdown cell sources were compared to the supplied ZIP and are unchanged. Evaluation and Gradio notebooks and metrics JSON also match byte-for-byte. The training notebook differs only in removal of stale widget/output display metadata and JSON serialization; no training settings changed.

| Artifact | SHA-256 committed file |
|---|---|
| `notebooks/Arabic_News_Training_1200.ipynb` | `f593daa03b7a27719596d3311e28889e06843daef27a00491f12ae8c4417ff17` |
| `notebooks/Arabic_News_Evaluation.ipynb` | `27d59d4a12dd71c56debb04a3b05f07aabca3ba7d06066581dff8fcbef50006d` |
| `notebooks/Arabic_News_Gradio.ipynb` | `84d9ec1f74fce68c6630a2f954ee4987a34b4d0906d32df17e11bf8b75581b50` |
| `reports/metrics_summary.json` | `9212e995dc659695d8dac2e9631f009f3d3e913f10346d46e9d6539a3eed51ce` |
