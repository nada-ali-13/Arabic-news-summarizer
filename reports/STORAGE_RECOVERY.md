# Drive Trash incident and safe full-state recovery

## What consumed the space

The user confirmed that Drive Trash contains approximately **8 GB of training artifacts created during this run**. The account was nearly empty before training (reported 0.17 GB used), then showed 12.74 GB used, entirely in Drive. The active project scan was 4.74 GB. This is not attributed to pre-existing user files.

The old runner saved full Adam state for **both smoke steps**, then called `shutil.rmtree` on their mounted Drive directories. Drive moved those files to Trash, where they still count against quota. Two model-plus-optimizer pairs are approximately `2 × (1.47 + 2.94) = 8.82` decimal GB, or 8.21 GiB. The active checkpoint-10 adds another 4.41 decimal GB; prepared data and reports add about 0.33 GB. These approximate amounts and GB/GiB display differences explain the reported account usage. The read-only audit reconciles actual API bytes, file charged sizes, ancestry and Trash instead of forcing approximate UI values to agree exactly.

A second defect made this worse: the old callback detected its free-space reserve at optimizer step 10 and **requested another large checkpoint write without reserving its actual write size**. Its project limit of 40 GB was not the account quota. Mounted filesystem free space was also not an authoritative account-quota check.

## First action: CPU audit, no further GPU training

Open [the storage recovery notebook](../notebooks/Arabic_News_Summarizer_Storage_Recovery.ipynb) from this PR in a **fresh CPU Colab runtime**. Before merge, use the PR branch `codex/storage-safe-recovery`; for reproducibility set `CODE_REF` to the exact PR head SHA shown by GitHub. After merge, `main` resolves and freezes the merged commit. Keep:

```python
RUN_DIR = Path('/content/drive/MyDrive/ArabicNewsSummarizer/full-v2-fp32-final-01')
ENABLE_TRAINING = False
SEED_LEGACY_WITHOUT_GPU = False
APPLY_SMOKE_TRASH_DELETION = False
APPLY_MANAGED_RETENTION = False
```

Run the bootstrap and read-only audit/preview cells. Do not run the old training notebook pinned at `52525c5` for further training. That commit passed the precision smoke but predates this storage correction.

The new notebook uses a separate code checkout and `storage-v1/runtime_source_commit.json`. It **preserves the original** `source_commit.json`, `run_config.json`, budget, prepared data, experiment records and checkpoint-10. Do not rewrite their locks, change experiment lengths/precision, rename checkpoints or create a fresh training directory for this recovery. Do not run two writers against this run.

The audit reads account `storageQuota.limit`, `usage`, `usageInDrive` and `usageInDriveTrash`. It lists owned active and trashed file IDs, names, charged bytes, parents/ancestors, and checksums. `other_*` means outside this project folder lineage, not pre-existing user files. Missing ancestry, revisions, hidden/other folders, shared storage, units and delayed accounting are reported rather than guessed away. Known Hugging Face cache paths are printed without credentials. Keep the detailed audit private.

## Explicit, narrow cleanup of existing Trash

1. The preview must successfully validate **all files of `experiments/arat5-5e5/checkpoint-10`** against its SHA-256 manifest. If it fails, stop: preserve all files and investigate the manifest before cleanup.
2. Review every proposed candidate ID, name, size, checksum and original location. The automated list accepts only already-trashed, expected generated files with proven project → experiments → smoke → checkpoint-1/2 ancestry. It excludes checkpoint-10. It does not offer entire folders or unrelated Trash contents.
3. Only after reviewing that exact list, set `APPLY_SMOKE_TRASH_DELETION=True`, rerun its cell and type the exact `DELETE OBSOLETE SMOKE …` phrase. Broader Drive authorization for legacy mount-created files is requested only then. The entire batch, ancestry and protected checkpoint are revalidated before the first permanent delete. A receipt is saved under `storage-v1/`.
4. **Never empty the whole Trash.** If the list is empty/incomplete because Drive lost ancestry, do not match files by names alone. Inspect Drive's original-location metadata and select only independently identified obsolete smoke artifacts for a separate, explicit manual confirmation. Preserve ambiguous files and checkpoint-10.
5. Requery quota; wait for actual account accounting to reflect the cleanup. A successful delete request is not treated as immediate free space. If cleanup is interrupted, regenerate the audit/plan for the remaining files rather than replaying stale IDs.

No real Drive files were deleted by Codex. The account-specific candidate list must be reviewed in your Colab session.

## Verify the recoverable step before spending GPU time

After cleanup is reflected, set `SEED_LEGACY_WITHOUT_GPU=True` and execute the adoption cell on CPU. This validates the original, copies it locally, measures lossless compression, writes `storage-v1/storage-plan.json`, checks the **actual account limit**, and reports measured and worst-case lifecycle budgets before training. CPU adoption can verify a recoverable archive even when the full training plan remains blocked. The configured `40 GB` legacy field does not override this check.

If there is sufficient current upload/staging headroom, it uploads an immutable compressed step-10 snapshot, checks server size/MD5 against the local archive, publishes/read-backs the index, then **downloads and decodes it and validates every original file hash**. Check `storage-v1/recovery-verification.json` for `downloaded_decoded_all_file_hashes_verified`, step 10. The original checkpoint-10 stays unchanged, including Adam, scheduler and RNG. This procedure verifies your actual archive roundtrip without executing an optimizer step. It is optional in the API but recommended before this incident's first T4 resume. A block is not successful verification; inspect its exact shortage instead of lowering the margin.

New archives live in a separate visible MyDrive folder, `ArabicSummarizerRecovery-<run-id>`. Keep this folder **and the original checkpoint-10 model**: every lossless model delta references that immutable model, not a previous delta. Deleting the base invalidates future restores. Fresh runs persist a separate immutable pretrained base instead.

## Resume the existing full plan

1. After successful CPU verification, select a fresh **T4** runtime. Reopen the same recovery notebook, same `RUN_DIR` and frozen storage-code source. Leave deletion flags false unless a concrete pending batch needs approval. Set `ENABLE_TRAINING=True`; keep `SEED_LEGACY_WITHOUT_GPU=False` after the verification has succeeded.
2. Run all. Reconfirm `requested=no, effective=no, accelerator=no, native_amp=False, scaler=False`. The copied/restored full checkpoint is passed to `Trainer.train(resume_from_checkpoint=...)`. It restores Adam moments/step, scheduler, RNG, Trainer epoch and data skipping. No optimizer or scheduler reset is introduced. Smoke, existing baseline and completed experiments are skipped.
3. On a cleanup pause, stop GPU work. Inspect `store.cleanup_plan()` in the notebook, check the replacement IDs and exact obsolete IDs, then enable `APPLY_MANAGED_RETENTION` and type its exact `DELETE MANAGED …` phrase. Only old **new-format, app/run-owned full/weights archives** may be permanently removed, and only after replacement checksums and its published index validate. Original legacy artifacts/base archives cannot be retired by this method. No blanket permission is assumed for later batches.
4. Rerun with the same settings after cleanup/accounting refresh. In the same kernel, rerun the quota/cleanup and training cells, not the source-bootstrap cell; a fresh runtime can run all. Full-state state already committed at an experiment's last step is finalized without taking an extra optimizer step. An interrupted upload never authorizes removal of the old checkpoint. Pending index/best publication is completed before training continues.
5. On session time limit, reopen/rerun to continue. On the total time limit, explicitly extend only `max_hours` if desired; the config fingerprint intentionally excludes budgets. Do not change optimizer, LR, max_steps, accumulation, seed or token lengths in this run.

Progress/new reports are under `storage-v1/`; the managed index is in the separate recovery folder. The old top-level metadata remains an incident snapshot. Final selection, baseline comparison, held-out test evaluation and the optional application remain available. Only the global winning model's weights are retained; losing experiments retain validation metrics and progress records, rather than an additional full model export each. The current experiment's full optimizer state is retained until safely replaced by the next one. A completed losing experiment is not automatically rerun.

## Checkpoint and cache policy

- Trainer writes full checkpoints **locally under `/content`**, never through the Drive mount. Local retention is one latest checkpoint. Smoke checkpoints remain local and never enter Drive Trash.
- Model deltas use bytewise XOR against the immutable base followed by streaming Zstandard. Optimizer, scheduler, RNG, configuration and manifests are compressed unchanged. Restoration is byte-exact; this is not quantization or weights-only recovery. Compression ratio is measured, never assumed.
- Remote retention is one global latest **full-state** archive plus one global best **weights/tokenizer** archive. During replacement the old and new coexist. Each delta is independently recoverable from the immutable base, without a chain of superseded checkpoints.
- Uploads have unique IDs. Local manifests, archive SHA-256, server checksums, index readback and restore manifests protect replacement publication. Every cleanup requires explicit ID-bound confirmation. Permanent API deletion of confirmed managed obsolete IDs bypasses Trash. If confirmation is pending, training pauses instead of silently accumulating snapshots.
- Account API quota is checked before GPU work, periodically before optimizer work, and immediately before each upload, including active files and Trash. Worst-case next full-state size, 10% write overhead and a **1 decimal GB margin** are reserved. This replaces the old generic reserve with a measured lifecycle budget and local staging; it does not merely lower a number to force progress. A plan can still be blocked after cleanup if capacity is insufficient.
- HF model binaries, datasets, Arrow transformations, raw checkpoints and compression staging stay on local Colab disk. A pinned allowlist downloads only the upstream PyTorch binary; background safetensors conversion/duplicate format downloads are avoided. Existing prepared data is copied locally and reused. No cache is automatically purged from Drive. Inspect any legacy cache first and preserve unknown artifacts.
- A failed upload/index publication may leave an unreferenced managed object. It is visible to the folder audit and quota checks and is **not automatically deleted**; inspect its ID before a separate explicit cleanup. Do not launch simultaneous sessions. Quota is not an atomic reservation against other account activity; Drive rejection preserves the previous committed state.

## Budget for two 1,200-step experiments

Checkpoint-10 has completed 10 of the first candidate's steps, leaving **1,190 + 1,200 = 2,390** optimizer steps if neither candidate stops early. At accumulation 16, that is up to **38,240 further training microbatches**. There are up to 24 scheduled validation/checkpoint rounds (256 validation examples each), then baseline and winner test generation on 512 examples each. Session-budget pauses may add extra saves. Retention bounds concurrent snapshots; it does not retain 24 full checkpoints.

For the reported legacy state, a useful **illustration, not a measured compression result**, in decimal GB is:

| Concurrent account component | Example GB |
|---|---:|
| Protected original full checkpoint-10 | 4.41 |
| Prepared data/reports/other project bytes | 0.33 |
| Other account bytes reported before training | 0.17 |
| Current lossless full-state archive (must measure) | 3.20 |
| Global best compressed model (must measure) | 0.90 |
| Next replacement, budgeted at uncompressed full size | 4.41 |
| 10% replacement overhead + fixed margin | 1.441 |
| **Peak example** | **14.861** |

This measured-size example is close to a nominal 15 GB limit and is **not a guarantee**. The default training gate also budgets future retained archives at full size: replacing the example 3.20 GB current archive and 0.90 GB best with 4.41 GB and 1.47 GB gives **16.641 decimal GB**. That exceeds 15 decimal GB by 1.641 GB, and 15 GiB by about 0.535 decimal GB. With these reported sizes, **GPU training is blocked even if the current compressed snapshot fits**. CPU archive verification can still proceed with adequate current headroom. This deliberately avoids spending GPU time on a plan whose future capacity depends on an unverified compression ratio. The actual API may report a 15 GiB account limit (16.106 decimal GB); UI labels are not used as byte limits. The first forecast conservatively budgets a full model for best weights until a real archive is measured. A larger optimizer archive, weak delta compression, remaining Trash, revisions, orphan uploads or other new usage can make the plan infeasible. `storage-plan.json` reports both measured-current and uncompressed lifecycle peaks, `full_plan_fits`, and the minimum additional capacity/offload bytes for the worst-case plan; every later replacement is checked again. There is no automatic opt-out that treats current compression as a future guarantee.

The time limit of seven cumulative hours is not a credible throughput estimate by itself. Measure on this T4: seconds per optimizer step (with 16 microbatches), seconds per 256-example generation evaluation, and CPU compression/upload time. Budget approximately `2390 × step_seconds + remaining_evaluations × evaluation_seconds + pack/upload/restore time + final test time`, plus baseline if absent. Compression and explicit cleanup pauses add wall time; free Colab session duration is not guaranteed.

If the measured safe peak exceeds quota, preserve checkpoint-10 and do not train. Reliable alternatives are more quota or a separately verified off-account full-state backup/store with enough replacement headroom. Removing the protected original requires an independently verified replacement base and a separate reviewed migration; this implementation does not do so. Keeping weights alone loses Adam/scheduler/RNG and is a new training trajectory; reduced optimizer precision/LoRA also changes the experiment. None is silently substituted for exact recovery.

## Validation and limits

CPU regression tests exercise an actual tiny gated-GELU T5/Adam Trainer: exact legacy step-10 archive roundtrip, continuation to the scheduled last step, scheduler/Adam progress, runtime reset/base restore, managed smoke isolation, interrupted base publication, quota rejection before model load/upload, corrupt replacement/index failures, stale quota snapshots, explicit retention approval, and scoped Trash deletion. Every original legacy file hash is checked unchanged. The complete CPU suite also preserves the original numerical, inference and preprocessing coverage.

The user's real T4 FP32 smoke at `52525c54f683a244f1df71a6403c7e3984d5ac49` remains valid precision evidence. **This new storage profile, live Drive API authorization/deletion, full-model compression ratio and T4 full-state recovery have not been verified on the user's account by Codex.** Perform the CPU archive verification above before the next GPU resume. No full-plan success or final summarization quality is claimed.
