> **سجل تاريخي:** يصف هذا الملف العمل السابق، وليس مسار تسليم خُلاصة النهائي. راجع [README الحالي](../README.md) و[النتائج النهائية](metrics_summary.json).

# Executed validation

Date: **2026-11-06**. Source project inspected before editing: initial commit `98947f10511fec6a0ee6c96129640a06db66da76`, containing only the two-line README. No existing implementation was replaced.

## Executed successfully

This report records the initial build. For the correction's 80 CPU tests and the subsequent user-verified real T4 smoke, see [numerical validation](NUMERICS_VALIDATION.md) and [recorded T4 evidence](T4_SMOKE_RESULT.json). Full-plan training/quality and real Colab interruption recovery remain pending.

- **62 pytest cases passed**. Machine-readable results: [pytest.xml](pytest.xml).
- `ruff check .`: passed, including Python source, tests, scripts and notebook code cells.
- `python -m compileall -q src app.py scripts`: passed.
- Notebook validated against nbformat 4 and every code cell compiled; all execution outputs remain empty.
- Built and installed a Python wheel locally without downloading dependencies, then imported the packaged `arabic_summarizer` successfully.
- **GitHub Actions passed** in a fresh Ubuntu/Python 3.12 runner, including dependency installation, Ruff and pytest, for implementation commit `cd0d89e013ca97116add5fd94237c2769860d285`: [verified run](https://github.com/nada-ali-13/Arabic-news-summarizer/actions/runs/42208195534).

Environment: Python **3.12.14**, PyTorch **2.8.0+cpu**, Transformers **4.57.1**, Accelerate **1.11.0**, Datasets **4.3.0**, Gradio **5.49.1**, SentencePiece **0.2.1**, Trafilatura **2.0.0**, rouge-score **0.1.2**, Ruff **0.14.3**. No CUDA/GPU was used. Test-only dependencies were installed under an external workspace scratch directory, outside the source repository. No large pretrained model checkpoint was downloaded.

## What the tests establish

- Conservative Arabic normalization, word counts, strict config validation and fixed official split handling.
- Empty-row removal and exact ID/source/URL leakage prevention with test priority.
- Real parsing of a controlled v2-style tar/JSONL archive, schema checks, statistics and revision locking, with the network download mocked.
- Unicode Arabic ROUGE gives meaningful nonzero overlap and correct exact-match behavior, rather than discarding Arabic tokens.
- Actual **Seq2SeqTrainer optimization** of a tiny randomly initialized T5 on synthetic Arabic fixtures.
- Actual Adam state saved at step 1, simulated interruption, automatic recovery from the sealed checkpoint and continued Adam state at step 2.
- Skipping the completed smoke experiment, reusing prepared data, preserving baseline results and excluding test evaluation until the plan finishes.
- Full checkpoint manifests; corrupt/incomplete/foreign-run checkpoints rejected; older valid checkpoint fallback; dangling best-checkpoint references rejected.
- Immutable export, best-pointer protection, independent model/tokenizer loading through the inference class and actual CPU generation with that tiny test model.
- Saved evaluation outputs recompute to the same full-reference ROUGE values.
- Long-document generation routing, invalid input and summary statistics with lightweight mocks.
- Actual Gradio Blocks construction and event handler execution without a web server or model download.
- Real Trafilatura extraction from controlled Arabic HTML.
- SSRF tests: blocked schemes/ports/userinfo; local, private, metadata, multicast, mapped and transition IP addresses; mixed DNS responses; pinned target IP, retained Host/SNI/certificate hostname; redirect revalidation and simulated DNS rebinding; response status/type/compression/size limits.

One benign warning from the Trainer in the tiny filesystem test: checkpoint mtimes may be unreliable, so it uses numeric checkpoint ordering. The warning is recorded in pytest output; there were no failing or skipped tests.

## Not yet verified

| Work | Status / required next step |
|---|---|
| Actual public tokenizer/weight/archive downloads from Colab | Upstream availability/config/files inspected via public metadata. Run notebook to exercise downloads. |
| AraT5v2 supervised fine-tuning on T4 | Pending real Colab GPU execution. Tiny CPU test is not evidence of GPU memory, FP16 stability or final quality. |
| Actual Google Drive mount, partial-write behavior and reconnection | Pending user-authorized Colab run. Local full-state recovery was verified. |
| XL-Sum baseline/fine-tuned ROUGE and qualitative review | Pending training and final evaluation. No scores supplied. |
| Full-model CPU latency/RSS | Pending model export and benchmark. No substitute timing from the tiny test. |
| Live public article fetch | Controlled fixtures passed. No direct live article fetch was attempted through the restricted cloud network. Proxy-only infrastructure may require text paste. |
| Browser/device visual presentation | Gradio structure/RTL/event logic tested; visual behavior on target phones/browsers remains unverified. |
| Hugging Face Spaces / Gradio sharing | Prepared, not deployed or launched publicly. Sharing is initiated by the user running the final notebook cell. |
| Model redistribution license | Exact model card has no explicit weight license. Resolve before public weight redistribution. Dataset license is CC BY-NC-SA 4.0. |

## User execution required

Open `notebooks/Arabic_News_Summarizer_Training.ipynb` in Colab from the review branch. Select a T4, run all, authorize Drive, retain the same run directory on restart, then inspect actual reports. A terminated runtime must be reopened manually. Unsaved steps may be lost. Raise budgets only deliberately; use a new directory for changed training settings. The optional app cell loads the best completed export and requests a temporary Gradio share link.
