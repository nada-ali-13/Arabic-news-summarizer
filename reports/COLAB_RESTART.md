# Safely retrying the corrected smoke test

## What is established

The user's diagnostic trace is **global step 1, loss 27.2233, grad_norm NaN, learning_rate 0.0**. The original generic callback did not identify the bad field, and ran after the optimizer update. This is a nonfinite-gradient failure with finite loss, not evidence of a NaN training loss. A zero learning rate cannot protect Adam from NaN gradients; that behavior is reproduced in a regression test.

A separate, reproducible configuration defect explains why an apparent FP32 retry can still use FP16. In [Transformers 4.57.1 TrainingArguments](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/training_args.py#L1873), setting both `fp16=False` and `bf16=False` retains `ACCELERATE_MIXED_PRECISION` from the environment. A previous FP16 attempt leaves that variable set to `fp16`. In a real local reproduction, false flags resulted in `Accelerator.mixed_precision == "fp16"`. A new Drive directory does not reset this process state.

The corrected pipeline explicitly sets the selected effective precision before constructing TrainingArguments, then asserts that Accelerator agrees. It writes the requested/effective/actual precision, AMP/scaler status, model parameter dtypes and relevant model/tokenizer configuration to `experiments/smoke/precision.json`. Development mode now explicitly requests FP32 (`mixed_precision="no"`). Full mode also defaults to explicit FP32; FP16 remains an opt-in configuration, not an automatic retry. Unsupported BF16 is rejected on T4 rather than silently falling through.

**Evidence boundary:** the precision-selection bug and gradient-safety defects are locally reproduced and corrected. The exact internal operation that produced NaN gradients in the user's pretrained-model T4 run has not been reproduced here. We have not verified that the corrected pretrained-model smoke succeeds on a real T4. A finite CPU tiny-model smoke is not proof of that. The user confirmed this precise mismatch in the same Colab session after requesting `"no"`: `env: fp16`, `active: fp16`, `torch: 2.11.0+cu130`. Thus the reported FP32 retry was still an FP16 attempt; it is not evidence of an FP32 gradient failure. This confirms the precision-selection defect in that session, but does not identify the first internal FP16 operation producing NaN.

## Fresh Colab retry — preserve all old experiments

1. Open the **updated notebook from this correction PR**, not the earlier saved copy. Before merge, use the correction branch or exact fixed commit shown in the PR. After merge, the permanent default is `CODE_REF="main"`.
2. Use **Runtime → Disconnect and delete runtime** (or a fresh Colab session), then select **T4 GPU**. This resets imported Python modules, monkey patches, Accelerator/AMP state and the ephemeral `/content` checkout. It does **not** delete Google Drive experiments. Do not reuse the earlier temporary `traced_budget` monkey patch in the fresh session.
3. In the Drive/configuration cell set `MODE = "dev"` and use a **new empty directory**, for example:

   ```python
   MODE = "dev"
   RUN_DIR = Path("/content/drive/MyDrive/ArabicNewsSummarizer/dev-v2-numerics-first-retry")
   RUN_DIR.mkdir(parents=True, exist_ok=True)
   ```

   Keep `dev-v1`, `dev-fp32-test`, `dev-log-diagnostic` and every old `source_commit.json`/checkpoint untouched. Check the new directory is empty before its first run. The notebook default has moved to `<mode>-v2-numerics` to avoid reusing v1 directories.
4. In the source cell, before the PR is merged, set:

   ```python
   CODE_REF = "codex/fix-smoke-numerics"  # or the exact fixed commit SHA from the PR
   ```

   The updated helper **fetches that ref even when a code checkout already exists**, checks out the fetched commit, prints its SHA and pins it in the NEW directory's `source_commit.json`. Do not simply change `CODE_REF` inside an old pinned run: an existing lock intentionally wins. Never delete or rewrite an old lock to force different code onto its checkpoints. Dirty checkouts are refused without deleting local edits.
5. In the training cell, keep the new `configs/dev.json` precision (`"no"`) and set:

   ```python
   SMOKE_ONLY = True
   result = run(config, RUN_DIR, smoke_only=SMOKE_ONLY)
   ```

   This executes only the two-step smoke and its validation/save path. It does not run the pretrained baseline, later candidate or held-out evaluation. Expected setup output is `requested=no, effective=no, accelerator=no, native_amp=False, scaler=False`; model parameter dtype should be `torch.float32`. If these differ, stop and retain `precision.json`.
6. Success is explicitly `status: "smoke_completed"`, with `experiments.json` marking smoke completed. This is not a final trained model or project quality result. The optional app cell prints a pending message until a real candidate export exists.
7. If a numerical failure remains, do not switch precision or keep restarting experiments blindly. Preserve and inspect `experiments/smoke/numerical_failure.json` and `precision.json`. The exception names the phase and exact bad value and completed step. The JSON includes attempted optimizer step, latest batch token IDs/label counts, offending parameter gradient summaries, precision flags, versions and GPU identity. Nonfinite values are explicit strings `nan`, `inf` or `-inf`, not zeros, filtered metrics or fabricated valid numbers. Share those two files/log contents for diagnosis.

## Continue or resume after the corrected smoke

After the corrected smoke has actually passed on T4, set `SMOKE_ONLY=False` and continue the same v2 run/configuration. The completed smoke is skipped and the original bounded experiment plan continues. After runtime termination, reopen the same updated notebook and run all against the same v2 path; its frozen source commit and verified checkpoints are restored.

Training schema **2** is part of the run fingerprint. Old v1 run configs/checkpoints are intentionally refused by this pipeline; weights/optimizer state from failed numerical runs must not be silently resumed. New v2 checkpoints retain the existing model/Adam/scheduler/RNG/Trainer state and SHA-256 validation. Raising budgets is still supported. A later code change requires a new run directory or a separately reviewed migration.

## Safety changes

The loss is checked before backward; unscaled gradient norms are checked before clipping/Adam using `error_if_nonfinite=True`; evaluation loss and all logged numeric fields remain checked. Genuine NaN/Infinity stops execution. The implementation does not use `nan_to_num`, suppress the failing field, skip a bad batch, alter model architecture/loss scaling, automatically downgrade precision, or publish successful results for a failed run.

The original tokenization already avoided fixed target padding, and the base collator already used `-100` for dynamic label padding. There is no evidence that bad label masking caused the supplied trace. The corrected collator makes that invariant explicit, rejects empty/all-ignored targets (which produce NaN mean cross-entropy), validates token ranges and source masks, and tests decoder shifting. Official data splits and conservative Arabic normalization are unchanged.
