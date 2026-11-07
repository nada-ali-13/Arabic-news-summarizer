> **سجل تاريخي:** يصف هذا الملف العمل السابق، وليس مسار تسليم خُلاصة النهائي. راجع [README الحالي](../README.md) و[النتائج النهائية](metrics_summary.json).

# Starting full training after the verified T4 smoke

The user verified the corrected FP32 two-step smoke on a real T4 at source commit **52525c54f683a244f1df71a6403c7e3984d5ac49**, schema 2, status `smoke_completed`, loss `12.97121810913086`, smoke ROUGE-Lsum `0.0`. [Evidence](T4_SMOKE_RESULT.json). These smoke metrics do not establish final summarization quality. This guide starts the original bounded full plan; it does not alter the tested training implementation.

## Start a separate full-mode run

1. Open the updated primary notebook in a fresh Google Colab **T4 GPU** runtime. Authorize Google Drive. Preserve the successful development smoke directory, every failed/v1 directory, and all existing source locks.
2. In the notebook's **single Drive/setup cell**, replace its four settings with:

   ```python
   MODE = "full"
   CODE_REF = "52525c54f683a244f1df71a6403c7e3984d5ac49"
   SMOKE_ONLY = False
   RUN_NAME = "full-v2-fp32-final-01"
   ```

   Use a new, empty run directory on the **first** launch. If that name has already been used for another configuration, choose another new name. Do not change `MODE` from dev to full inside the successful dev directory: source/target limits, validation subset and experiment plan differ. A directory named `full-v2-numerics` that pinned pre-fix main must also remain untouched.
3. Keep `configs/full.json` unchanged initially. It explicitly requests `mixed_precision="no"`. The first setup output must show `mode=full`, `smoke_only=False` and the chosen persistent directory. Each Trainer reports `requested=no, effective=no, accelerator=no, native_amp=False, scaler=False`. Its `precision.json` must report float32 model parameters.
4. Check actual Google Drive free quota before starting. The project permits up to **40 GB**, with a **7 GB checkpoint reserve**; the configured cap does not grant storage. Full model/Adam checkpoints are several GB each. Keep enough room for two retained checkpoints per real experiment, immutable exports, dataset cache and prepared data. Colab's virtual filesystem free-space report may not match account quota; check Drive storage too.
5. Run all cells. Full mode runs **its own smoke** with the longer full-mode token limits before the baseline and real candidates. The earlier dev smoke did not verify 1,024-token memory/performance. A pause or out-of-memory error is not a successful full run; preserve its state/diagnostic before changing settings. No hand edits between experiments are needed.

Pinning the verified commit works before and after PR #2 is merged, including if its branch is later deleted. The notebook itself may have newer documentation; the training code is fetched at the explicit verified SHA. For a new run after merge, `CODE_REF="main"` is also supported and freezes the main commit actually fetched. Once a run is started, its existing `source_commit.json` intentionally takes precedence. Do not edit that lock to change versions mid-run.

## What the full plan executes

- Two-step smoke with up to 8 training examples; it is excluded from candidate selection.
- Pretrained baseline on the fixed 256-example validation subset.
- Two independent supervised candidates: learning rates **5e-5** and **1e-4**, up to **1,200 optimizer steps each**, batch size 1, gradient accumulation 16, source limit 1,024 and target limit 128. Early stopping uses validation ROUGE-Lsum, patience 3.
- Evaluation and full-state checkpoint saves every **100 optimizer steps** in real candidates, with at most two checkpoints retained per experiment. Budget stops can request an additional save between scheduled checkpoints.
- Validation-only best-model selection, immutable model/tokenizer exports and atomic `best.json` protection.
- After the complete bounded plan finishes, baseline versus selected-model evaluation on a fixed **512-example test subset**, plus saved qualitative examples. This is not an evaluation of all 4,689 official test rows.

The configured time limits are **3 hours per notebook invocation** and **7 hours cumulative recorded compute**. Evaluation and checkpoint I/O may overrun a boundary. The full plan is not promised to finish within a free Colab allocation or those limits. `status="paused"` preserves recoverable progress; `status="completed"` and `final_report.json` establish plan completion. Actual quality remains to be measured.

## Resume after a pause, disconnection or runtime termination

1. Reopen the same updated notebook in a fresh T4 runtime and mount the same Drive account. The user must reopen a terminated runtime; Codex cannot restart it automatically.
2. Reuse **exactly the same four full-run settings**, including `RUN_NAME` and `SMOKE_ONLY=False`. Keep the same experiment configuration, seed, precision and sequence lengths. Do **not** choose another directory for recovery; first-launch emptiness is not a requirement on resume.
3. Run all. The notebook restores the source SHA from `source_commit.json`; model/dataset revisions are restored from `run_config.json`. Completed experiments are skipped. The interrupted experiment discovers the latest SHA-256-validated, fully saved checkpoint and uses `resume_from_checkpoint` to restore weights, Adam, scheduler, Trainer step/epoch and RNG state.
4. If the newest save is incomplete/corrupt, an older valid save is tried. If none is valid, the code stops and preserves files rather than silently starting over. Work since the latest complete save may be lost. Do not copy dev or v1 checkpoints into the full directory.
5. If a **session** budget pause occurs, reopening/rerunning permits another invocation while respecting the cumulative limit. If the **total** budget is exhausted, decide whether to extend it explicitly in the training cell, for example:

   ```python
   from dataclasses import replace
   config = replace(config, max_hours=10, session_hours=3)
   ```

   Apply that after `load_config` and before `run`; the example is optional, not an automatic extension. Time/storage budget fields may change on restart. Raising storage limits requires actual space. Changed training semantics require a new run directory, not overwritten locks.

Do not run two Colab sessions against the same run directory. Numerical failures remain fatal and produce `experiments/<name>/numerical_failure.json`; inspect it with `precision.json` rather than suppressing checks or silently switching back to FP16.

## Persistent outputs to inspect

All are under `MyDrive/ArabicNewsSummarizer/<RUN_NAME>/`:

- `source_commit.json`, `run_config.json`, `installed_packages.txt`: reproducibility.
- `experiments.json`, `budget.json`: progress, statuses, metrics and budget accounting.
- `experiments/<name>/checkpoint-<step>/complete.json`: validated full-state checkpoint manifests.
- `exports/<name>-<id>/` and `best.json`: complete independent inference artifacts and winner pointer.
- `baseline_validation.json`, `baseline_test.json`, `best_test.json`, `final_report.json`: actual comparisons and qualitative output. They appear as their stages complete, not before.

Inspect factuality, omissions, fluency, repetition and completeness with `QUALITATIVE_REVIEW.md`. A two-step smoke ROUGE of zero is not a reason to tune on the held-out test set. Real Colab interruption recovery and final full-plan quality have not yet been verified by the reported smoke success.
