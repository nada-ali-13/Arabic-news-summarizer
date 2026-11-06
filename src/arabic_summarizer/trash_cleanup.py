"""Explicit, ID-bound cleanup of ONLY proven obsolete smoke artifacts in Trash."""

import hashlib
import json
from pathlib import Path

from .checkpoints import sha256, valid
from .drive_storage import FIELDS
from .storage import StorageBlocked

ALLOWED = {
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "complete.json",
    "run_config.json",
    "added_tokens.json",
    "scaler.pt",
}


def preview_smoke_cleanup(audit, protected_checkpoint, fingerprint):
    protected = Path(protected_checkpoint).resolve()
    if protected.name != "checkpoint-10" or not valid(protected, fingerprint):
        raise StorageBlocked("The preserved checkpoint-10 must pass full SHA-256 validation first")
    candidates = []
    for row in audit["largest_files"]:
        names = set(row.get("ancestor_names", []))
        if (
            row.get("category") == "project_trash"
            and row.get("trashed")
            and {"experiments", "smoke"} <= names
            and bool({"checkpoint-1", "checkpoint-2"} & names)
            and row.get("md5")
            and row["name"] in ALLOWED
            and "checkpoint-10" not in names
        ):
            candidates.append(
                {
                    key: row.get(key)
                    for key in ("id", "name", "size_bytes", "charged_bytes", "md5", "parents")
                }
            )
    plan = {
        "project_folder_id": audit.get("project_folder_id"),
        "protected_checkpoint": str(protected),
        "fingerprint": fingerprint,
        "protected_manifest_sha256": sha256(protected / "complete.json"),
        "candidates": sorted(candidates, key=lambda r: r["id"]),
        "candidate_charged_bytes": sum(r["charged_bytes"] for r in candidates),
    }
    token = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16]
    plan["confirmation"] = "DELETE OBSOLETE SMOKE " + token
    return plan


def apply_smoke_cleanup(service, plan, confirmation):
    # Full Drive delete scope may be needed for old mount-created files. Request
    # that authorization only after reviewing this concrete plan, never at audit.
    expected = dict(plan)
    expected.pop("confirmation", None)
    token = (
        "DELETE OBSOLETE SMOKE "
        + hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()[:16]
    )
    if not plan["candidates"] or confirmation != token or plan["confirmation"] != token:
        raise StorageBlocked("No deletion: exact nonempty reviewed plan confirmation required")
    protected = Path(plan["protected_checkpoint"])
    if (
        not valid(protected, plan["fingerprint"])
        or sha256(protected / "complete.json") != plan["protected_manifest_sha256"]
    ):
        raise StorageBlocked("Protected checkpoint changed; no deletion allowed")
    if not plan.get("project_folder_id"):
        raise StorageBlocked("An exact project folder ID is required")

    def check_lineage(item):
        pending, seen, names = list(item.get("parents", [])), set(), set()
        while pending:
            file_id = pending.pop()
            if file_id in seen:
                continue
            seen.add(file_id)
            if len(seen) > 100:
                raise StorageBlocked("Unexpected ancestor graph")
            parent = service.files().get(fileId=file_id, fields="id,name,parents").execute()
            names.add(parent["name"])
            pending.extend(parent.get("parents", []))
        return (
            plan["project_folder_id"] in seen
            and {"experiments", "smoke"} <= names
            and bool({"checkpoint-1", "checkpoint-2"} & names)
            and "checkpoint-10" not in names
        )

    # Validate the complete batch before the first DELETE.
    for row in plan["candidates"]:
        actual = service.files().get(fileId=row["id"], fields=FIELDS).execute()
        if (
            not check_lineage(actual)
            or not actual.get("trashed")
            or actual["name"] != row["name"]
            or actual.get("parents", []) != row["parents"]
            or actual.get("md5Checksum") != row["md5"]
            or int(actual.get("size", 0)) != row["size_bytes"]
        ):
            raise StorageBlocked("Cleanup candidate changed; regenerate the read-only audit")
    deleted = []
    for row in plan["candidates"]:
        service.files().delete(fileId=row["id"]).execute()
        deleted.append(row["id"])
    # Account quota can lag. This is an observation, not a promise space is free.
    return {
        "deleted_ids": deleted,
        "quota_after": service.about().get(fields="storageQuota").execute(),
        "protected_checkpoint": str(protected),
    }
