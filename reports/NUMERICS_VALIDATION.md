# Numerical correction: evidence and validation

Based on latest main `f2da46c` (the original implementation is merged). No existing user experiments or model artifacts were edited.

## Root cause evidence

The user exposed the previously hidden log values in Colab: `global_step=1`, `loss=27.2233`, `grad_norm=nan`, `learning_rate=0.0`, with configuration `mixed_precision="no"`. A read-only check in that same session then returned:

```text
env: fp16
active: fp16
torch: 2.11.0+cu130
```

This establishes that the reported FP32 retry was actually still running FP16. The pinned Transformers TrainingArguments inherits `ACCELERATE_MIXED_PRECISION` when both precision flags are false. Changing a Drive directory or the project's printed config does not reset the inherited setting. This defect was independently reproduced locally with the real Transformers/Accelerate objects, without requiring GPU training.

The original callback also concealed which field was nonfinite and detected gradient failure only after optimizer.step. Zero learning rate is not sufficient protection: Adam can contaminate weights/state with NaN even when LR is zero. A small direct PyTorch regression reproduces that behavior.

The correction explicitly sets/verifies the effective precision; development and full modes default to explicit FP32. No model replacement, architecture change, loss rescaling, metric suppression, NaN replacement or automatic precision retry is used. The first internal FP16 operation producing NaN in the pretrained T4 model is not identified here; we do not claim to have repaired FP16 arithmetic in the upstream model.

## Executed checks

- **80 pytest cases passed** on Python 3.12.14, PyTorch 2.8.0+cpu, Transformers 4.57.1, Accelerate 1.11.0. See [machine-readable results](pytest-numerics.xml).
- Ruff passed for the entire repository, including notebook cells.
- Python source compilation passed.
- Notebook schema validation and compilation of every code cell passed in pytest.
- Existing full-state checkpoint recovery/export/inference integration test passed, now also checking smoke-only mode followed by continuation of the original plan.
- New tests reproduce inherited FP16 selection despite false flags, then verify that the project's FP32 retry uses `accelerator=no`, no AMP and no scaler even with a stale FP16 environment.
- A real two-step FP32 tiny T5 smoke with **gated GELU and an untied output head** passes, with validation and optimizer checkpoint save.
- Injected NaN forward loss stops before backward; injected NaN gradients stop before Adam, with unchanged weights, empty optimizer state and no checkpoint. Diagnostics record the exact metric, step, bad parameter and bounded batch evidence.
- Dynamic `-100` label padding, decoder shifting, all-ignored-target rejection, token range checks, numeric scalar logging and explicit nonfinite JSON representation are covered.
- The actual notebook source-checkout helper is executed against local Git remotes: new runs fetch updated main rather than stale HEAD; existing source locks stay pinned; exact commit checkout works; dirty user edits are preserved.
- Old run configuration schemas are refused without modification.
- Notebook retry settings are defined once in setup. Pre-fix main is rejected before writing a fresh source lock, and existing incompatible source locks are preserved. This covers the user-confirmed `main`/`f2da46c` source-selection failure before training.

Benign Trainer warnings about unreliable filesystem mtimes falling back to numerical checkpoint ordering remain in the tiny tests.

## User-verified real Colab T4 smoke

The user subsequently reported successful execution of commit `52525c54f683a244f1df71a6403c7e3984d5ac49` on a real Google Colab NVIDIA T4:

| Field | Reported result |
|---|---|
| Training schema | 2 |
| Requested / effective / Accelerator precision | no / no / no |
| native_amp / scaler | False / False |
| Final status | smoke_completed |
| Smoke evaluation loss | 12.97121810913086 |
| Smoke ROUGE-Lsum | 0.0 |

[Machine-readable user-reported evidence](T4_SMOKE_RESULT.json). This is a two-step pipeline smoke pass, not a final model quality score. Zero smoke ROUGE-Lsum is uninformative for final quality and is not used to select a real candidate. The assistant did not execute this GPU run; the execution and values were supplied by the user. Follow-up finalization changes only documentation and notebook Markdown, leaving the tested training code/configuration unchanged.

## Still not verified

Completion/quality of the full training plan, full-mode 1,024-token memory/performance, recovery after an actual Colab runtime termination, and FP16 stability remain unverified. Full-state recovery is covered by real local CPU Trainer tests. No GPU is attached to the assistant execution environment; no additional full pretrained weights were downloaded.

Local tests use PyTorch 2.8.0+cpu; the user's CUDA runtime is 2.11.0+cu130. The requirements preserve a supported installed Colab CUDA PyTorch (>=2.6,<3) rather than forcing a CPU wheel or changing its CUDA build. The updated notebook records the actual installed versions and verified precision.

Use [the full training and recovery guide](FULL_TRAINING.md) to start a separate full-mode run. Preserve the successful smoke run, all v1/failed runs and their source pins. If a later numerical failure occurs, retain its exact diagnostic; this smoke pass does not guarantee every longer full-mode batch will fit or remain stable.
