"""Commit-marker checkpoints: incomplete/corrupt saves are never resumed."""

import hashlib
import re
from pathlib import Path

from .io import atomic_json, read_json


class RecoveryError(RuntimeError):
    pass


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def seal(directory, fingerprint, checkpoint=True):
    directory = Path(directory)
    if checkpoint:
        required = [
            "config.json",
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        ]
        if not all((directory / f).is_file() for f in required):
            raise RecoveryError("Checkpoint missing trainer/optimizer/scheduler/RNG state")
    if not (
        (directory / "model.safetensors").is_file()
        or (directory / "model.safetensors.index.json").is_file()
    ):
        raise RecoveryError("Missing safe model weights")
    if not checkpoint and (
        not (directory / "config.json").is_file()
        or not (directory / "tokenizer_config.json").is_file()
        or not any((directory / name).is_file() for name in ("tokenizer.json", "spiece.model"))
    ):
        raise RecoveryError("Export must include model configuration and a complete tokenizer")
    files = {
        str(p.relative_to(directory)): {"size": p.stat().st_size, "sha256": sha256(p)}
        for p in directory.rglob("*")
        if p.is_file() and p.name != "complete.json"
    }
    atomic_json(directory / "complete.json", {"fingerprint": fingerprint, "files": files})


def valid(directory, fingerprint):
    directory = Path(directory)
    try:
        manifest = read_json(directory / "complete.json")
        if manifest["fingerprint"] != fingerprint or not manifest["files"]:
            return False
        for name, entry in manifest["files"].items():
            p = directory / name
            if not p.resolve().is_relative_to(directory.resolve()):
                return False
            if not p.is_file() or p.stat().st_size != entry["size"] or sha256(p) != entry["sha256"]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def latest_checkpoint(directory, fingerprint):
    directory = Path(directory)
    candidates = sorted(
        (p for p in directory.glob("checkpoint-*") if re.fullmatch(r"checkpoint-\d+", p.name)),
        key=lambda p: int(p.name.split("-")[1]),
        reverse=True,
    )
    for path in candidates:
        if valid(path, fingerprint):
            try:
                state = read_json(path / "trainer_state.json")
                if state["global_step"] != int(path.name.split("-")[1]):
                    continue
                best = state.get("best_model_checkpoint")
                # The Trainer loads these weights at the end. Never resume with
                # a dangling/corrupted best-model reference.
                if best and Path(best) != path and not valid(best, fingerprint):
                    continue
                return str(path)
            except (OSError, ValueError, KeyError, TypeError):
                continue
    if candidates:
        raise RecoveryError(
            "No complete checkpoint survives validation; keep files and inspect, "
            "or start a new run directory. Refusing a silent fresh start."
        )
    return None


def directory_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
