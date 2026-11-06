"""Quota-aware, copy-validate-publish-retire checkpoint storage.

Legacy files are never removed. Only immutable artifacts created by this store
may be retired, and only after a remote replacement and its index are verified.
All byte budgets use decimal GB; archives are lossless, not optimizer quantization.
"""

import hashlib
import json
import math
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .checkpoint_codec import pack, unpack
from .checkpoints import directory_bytes, seal, sha256, valid
from .io import atomic_json

GB = 1_000_000_000


class StorageBlocked(RuntimeError):
    pass


@dataclass(frozen=True)
class Quota:
    limit: int
    usage: int
    drive_usage: int
    trash_usage: int
    observed_at: float

    @classmethod
    def from_api(cls, data):
        q = data.get("storageQuota", data)
        if "limit" not in q or "usage" not in q:
            raise StorageBlocked("Drive API did not return a finite account limit and usage")
        values = [int(q[k]) for k in ("limit", "usage", "usageInDrive", "usageInDriveTrash")]
        if values[0] <= 0 or any(v < 0 for v in values[1:]):
            raise StorageBlocked("Invalid account quota response")
        return cls(*values, time.time())

    def report(self, project_bytes=0):
        return {
            "account_limit_bytes": self.limit,
            "account_usage_bytes": self.usage,
            "account_free_bytes": max(0, self.limit - self.usage),
            "drive_usage_bytes": self.drive_usage,
            "drive_trash_bytes": self.trash_usage,
            "outside_drive_services_bytes": max(0, self.usage - self.drive_usage),
            "scanned_project_logical_bytes": project_bytes,
            "account_minus_project_bytes": self.usage - project_bytes,
            "observed_at": self.observed_at,
            "note": "Outside-Drive usage is not split into Gmail/Photos by this API. "
            "Logical filesystem bytes are not identical to account-charged bytes.",
        }


@dataclass(frozen=True)
class StoragePolicy:
    account_cap_bytes: int | None = None
    safety_margin_bytes: int = GB
    write_overhead_ratio: float = 0.10

    def __post_init__(self):
        if (
            self.account_cap_bytes is not None and self.account_cap_bytes <= 0
        ) or self.safety_margin_bytes <= 0:
            raise ValueError("Positive account cap and safety margin required")
        if not 0.05 <= self.write_overhead_ratio <= 1:
            raise ValueError("Write overhead must be between 5% and 100%")

    def needed(self, payload_bytes):
        return math.ceil(payload_bytes * (1 + self.write_overhead_ratio)) + self.safety_margin_bytes

    def require(self, quota, payload_bytes):
        # The account-wide API is queried immediately before every actual upload.
        if time.time() - quota.observed_at > 60:
            raise StorageBlocked("Quota snapshot is stale; refresh account usage before writing")
        free = max(0, min(quota.limit, self.account_cap_bytes or quota.limit) - quota.usage)
        required = self.needed(payload_bytes)
        if free < required:
            raise StorageBlocked(
                f"Account quota blocks write: free={free / GB:.2f} GB, "
                f"required={required / GB:.2f} GB including payload, "
                f"10%-or-configured overhead and safety margin. "
                "Existing checkpoints remain untouched."
            )

    def plan(
        self,
        quota,
        checkpoint_bytes,
        model_bytes,
        project_bytes,
        protected_bytes=0,
        managed_bytes=0,
        current_archive_bytes=None,
        best_archive_bytes=None,
    ):
        # One active full state + one replacement + one global best weights archive.
        # Preserve all legacy data by default; no hypothetical deletion is credited.
        outside = max(0, quota.usage - project_bytes)
        fixed = max(0, project_bytes - protected_bytes - managed_bytes)
        managed_peak = (
            checkpoint_bytes
            + (best_archive_bytes or model_bytes)
            + max(current_archive_bytes or checkpoint_bytes, model_bytes)
        )
        minimum = (
            outside
            + fixed
            + protected_bytes
            + managed_peak
            + self.safety_margin_bytes
            + math.ceil(checkpoint_bytes * self.write_overhead_ratio)
        )
        # A measured step-10 delta is not a bound on later weight/Adam entropy.
        # Promise the full plan only when later full-size archives also fit.
        worst_case = (
            outside
            + fixed
            + protected_bytes
            + 2 * checkpoint_bytes
            + model_bytes
            + self.safety_margin_bytes
            + math.ceil(checkpoint_bytes * self.write_overhead_ratio)
        )
        cap = min(quota.limit, self.account_cap_bytes or quota.limit)
        return {
            **quota.report(project_bytes),
            "checkpoint_bytes": checkpoint_bytes,
            "best_model_bytes": model_bytes,
            "protected_legacy_bytes": protected_bytes,
            "peak_managed_payload_bytes": managed_peak,
            "conservative_account_peak_bytes": minimum,
            "uncompressed_lifecycle_account_peak_bytes": worst_case,
            "measured_current_replacement_plan_fits": minimum <= cap,
            "minimum_additional_capacity_or_offload_bytes": max(0, worst_case - cap),
            "next_full_write_required_bytes": self.needed(checkpoint_bytes),
            "next_full_write_shortfall_bytes": max(
                0,
                self.needed(checkpoint_bytes)
                - max(0, min(quota.limit, self.account_cap_bytes or quota.limit) - quota.usage),
            ),
            "full_plan_fits": worst_case <= cap,
            "retention": "one managed full-state latest across experiments; one global best weights",
            "legacy_policy": "protected, never automatically deleted",
        }


def model_export(checkpoint, destination, fingerprint):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    allowed = {
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "added_tokens.json",
        "run_config.json",
        "model.safetensors.index.json",
    }
    for path in Path(checkpoint).iterdir():
        if path.is_file() and (path.name in allowed or path.name.endswith(".safetensors")):
            shutil.copy2(path, destination / path.name)
    seal(destination, fingerprint, checkpoint=False)
    return destination


class ManagedStore:
    """Backend contract: fresh quota, upload/stat/download, index read/write, scoped delete.

    A single writer is required. Uploads have unique IDs. The backend must validate
    folder/run ownership before delete; legacy Drive paths are never given to it.
    """

    def __init__(self, backend, local_root, fingerprint, policy=None, base_dir=None):
        self.backend = backend
        self.local_root = Path(local_root).resolve()
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        self.base_dir = Path(base_dir).resolve() if base_dir else None
        self.policy = policy or StoragePolicy()
        self.index = backend.read_index() or {
            "version": 1,
            "fingerprint": fingerprint,
            "latest": None,
            "best": None,
            "obsolete": [],
            "state": {},
        }
        if self.index["fingerprint"] != fingerprint:
            raise StorageBlocked("Storage index belongs to a different training configuration")

    def verify(self, artifact):
        actual = self.backend.stat(artifact["id"])
        if actual["size"] != artifact["size"] or actual["md5"] != artifact["md5"]:
            raise StorageBlocked("Remote checkpoint size/checksum validation failed")
        return actual

    def publish(self):
        self.index["commit"] = uuid.uuid4().hex
        self.backend.write_index(self.index)
        observed = self.backend.read_index()
        if not observed or observed != self.index:
            raise StorageBlocked("Remote index read-back failed; old artifacts are not retired")

    def cleanup_plan(self):
        current = [self.index[k] for k in ("latest", "best", "base") if self.index.get(k)]
        for artifact in current:
            self.verify(artifact)
        obsolete = self.index.get("obsolete", [])
        plan = {
            "index_commit": self.index.get("commit"),
            "obsolete": obsolete,
            "replacement_ids": [a["id"] for a in current],
            "bytes_to_retire": sum(a["size"] for a in obsolete),
        }
        digest = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16]
        plan["confirmation"] = "DELETE MANAGED " + digest
        return plan

    def cleanup(self, confirmation=None):
        # An obsolete file is eligible only when all currently referenced
        # replacements are independently verified and the committed index agrees.
        observed_index = self.backend.read_index()
        if observed_index is None and not any(
            self.index.get(k) for k in ("latest", "best", "base", "obsolete")
        ):
            return
        if observed_index != self.index:
            raise StorageBlocked("Index changed; refusing cleanup (single writer required)")
        current = [self.index[k] for k in ("latest", "best", "base") if self.index.get(k)]
        for artifact in current:
            self.verify(artifact)
        if not current:
            return
        plan = self.cleanup_plan()
        if self.index.get("obsolete") and confirmation != plan["confirmation"]:
            raise StorageBlocked(
                "Explicit cleanup confirmation required. Review cleanup_plan(): "
                + plan["confirmation"]
            )
        remaining = []
        live_ids = {a["id"] for a in current}
        for old in self.index.get("obsolete", []):
            if old["id"] in live_ids:
                raise StorageBlocked("Refusing to retire a live artifact")
            try:
                self.backend.delete_managed(
                    old, confirmation="DELETE " + old["id"] + " " + old["md5"]
                )
            except Exception:
                remaining.append(old)  # Explicitly retain pending cleanup on failure.
        if remaining != self.index.get("obsolete", []):
            self.index["last_cleanup_receipt"] = {
                "confirmation": confirmation,
                "replacement_ids": plan["replacement_ids"],
                "deleted_ids": [a["id"] for a in self.index["obsolete"] if a not in remaining],
                "observed_at": time.time(),
            }
            self.index["obsolete"] = remaining
            self.publish()
        if remaining:
            raise StorageBlocked(
                "Managed cleanup incomplete; quota must be refreshed before continuing"
            )

    def upload(self, directory, kind, experiment, step):
        if not valid(directory, self.fingerprint):
            raise StorageBlocked("Local checkpoint/export did not pass SHA-256 validation")
        size = directory_bytes(directory)
        if shutil.disk_usage(self.local_root).free < size * 2 + self.policy.safety_margin_bytes:
            raise StorageBlocked("Insufficient local staging space; no Drive artifact was changed")
        archive = self.local_root / (uuid.uuid4().hex + ".tar.zst")
        try:
            pack(directory, archive, self.base_dir)
            print(
                f"Lossless archive: raw={size / GB:.3f} GB, packed={archive.stat().st_size / GB:.3f} GB; checking live account quota"
            )
            self.policy.require(self.backend.quota(), archive.stat().st_size)
            digest = (
                hashlib.md5(archive.read_bytes()).hexdigest()
                if archive.stat().st_size < 1_000_000
                else None
            )
            if digest is None:
                h = hashlib.md5()
                with archive.open("rb") as stream:
                    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        h.update(block)
                digest = h.hexdigest()
            remote = self.backend.upload(archive, kind, experiment, step)
            artifact = {
                **remote,
                "kind": kind,
                "experiment": experiment,
                "step": step,
                "unpacked_bytes": size,
                "sha256": sha256(archive),
                "fingerprint": self.fingerprint,
                "manifest_sha256": sha256(Path(directory) / "complete.json"),
            }
            if artifact["size"] != archive.stat().st_size or artifact["md5"] != digest:
                raise StorageBlocked(
                    "Uploaded artifact differs from local archive; previous state retained"
                )
            self.verify(artifact)
            return artifact
        finally:
            archive.unlink(missing_ok=True)  # Only ephemeral archive created by this call.

    def commit_checkpoint(self, directory, experiment, step, state):
        artifact = self.upload(directory, "full", experiment, step)
        previous = self.index.get("latest")
        self.index["latest"] = artifact
        self.index["state"] = state
        if previous:
            self.index["obsolete"].append(previous)
        self.publish()
        return artifact

    def commit_best(self, directory, experiment, step, metrics):
        score = metrics["eval_rougeLsum"]
        if self.index.get("best") and score <= self.index["best"]["metrics"]["eval_rougeLsum"]:
            return self.index["best"]
        artifact = self.upload(directory, "weights", experiment, step)
        artifact["metrics"] = metrics
        previous = self.index.get("best")
        self.index["best"] = artifact
        if previous:
            self.index["obsolete"].append(previous)
        self.publish()
        return artifact

    def restore(self, artifact, destination):
        self.verify(artifact)
        destination = Path(destination)
        if destination.exists():
            if valid(destination, self.fingerprint):
                # Compare the exact manifest, not merely the run fingerprint.
                if sha256(destination / "complete.json") == artifact.get("manifest_sha256"):
                    return destination
            raise StorageBlocked("Restore destination exists; use an empty local workspace")
        archive = self.local_root / (uuid.uuid4().hex + ".tar.zst")
        staging = destination.with_name(destination.name + ".staging-" + uuid.uuid4().hex)
        try:
            self.backend.download(artifact["id"], archive)
            if archive.stat().st_size != artifact["size"] or sha256(archive) != artifact["sha256"]:
                raise StorageBlocked("Downloaded archive failed validation")
            unpack(archive, staging, artifact["unpacked_bytes"], self.base_dir)
            if not valid(staging, self.fingerprint):
                raise StorageBlocked("Restored checkpoint failed per-file SHA-256 validation")
            # Add a non-payload marker outside the inner complete.json manifest.
            atomic_json(staging / "restored_artifact.json", {"id": artifact["id"]})
            staging.rename(destination)
            return destination
        finally:
            archive.unlink(missing_ok=True)
            if staging.exists():
                shutil.rmtree(staging)  # Only temporary local extraction created here.
