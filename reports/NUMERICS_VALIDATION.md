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

## Not verified

**Corrected AraT5v2/XL-Sum smoke on a real Colab T4 is pending.** No GPU is attached to this execution environment. Direct Hugging Face downloads are also blocked by its network policy, so no additional full pretrained weights were downloaded. The user's observed active precision is real Colab evidence of the defect, not a verified post-fix training result.

Local tests use PyTorch 2.8.0+cpu; the user's CUDA runtime is 2.11.0+cu130. The requirements preserve a supported installed Colab CUDA PyTorch (>=2.6,<3) rather than forcing a CPU wheel or changing its CUDA build. The updated notebook records the actual installed versions and verified precision.

Use [the explicit fresh-runtime/new-directory restart procedure](COLAB_RESTART.md). Preserve all v1/failed runs and their source pins. If a true FP32 attempt still fails, the new failure report names the exact stage and value; retain that report rather than treating this change as proof of T4 success.
