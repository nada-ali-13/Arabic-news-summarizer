# Storage correction validation

Based on merged main `d1b24cbeef87aa61e45693fd5f7895a9f8db6f05` (PR #2).

Local final checks: **106 passed**, no failures/skips; five benign upstream Trainer warnings about mtime ordering. `ruff check .`, `git diff --check`, both notebook schema validation and compilation of all code cells passed. Machine-readable results: [pytest-storage.xml](pytest-storage.xml).

Tests use CPU PyTorch 2.8.0 and Transformers 4.57.1, offline tiny real gated-GELU T5 optimization, local pretrained/tokenizer fixtures, and fake account/Drive adapters. They do not download AraT5v2's 1.47 GB weights or access a live Drive account.

New coverage includes actual step-10 Adam/scheduler/RNG byte-preserving transfer and restore; continuation to the scheduled final step without an extra update after cleanup; completed work skipping; all original file hashes unchanged; fresh smoke checkpoints staying local; fresh immutable-base recovery after runtime loss; failed base/index publication retry; quota including Trash/GB-vs-GiB, stale quota, insufficient quota before model allocation or upload; worst-case lifecycle rejection despite favorable current compression; corrupt replacement preservation; explicit confirmation tied to cleanup IDs/index; scoped namespace and legacy smoke lineage checks; protected-checkpoint corruption blocking deletion; notebook deletion/training defaults off; persistent precision and training logs; legacy mounted-Drive writes blocked.

The user's earlier real T4 precision smoke remains recorded separately in [T4_SMOKE_RESULT.json](T4_SMOKE_RESULT.json). The user also confirmed the actual ~8 GB Trash cause; [incident evidence](STORAGE_INCIDENT.json) keeps their approximate UI values distinct from exact API bytes.

Pending: live Google Drive authorization, account-specific audit/cleanup list, full-size compression, CPU remote archive decode/hash proof, real T4 full-state recovery and complete 2 × 1,200-step execution. The recovery notebook provides the CPU proof before the GPU gate. No real artifacts have been deleted by Codex and no new T4 success is claimed.
