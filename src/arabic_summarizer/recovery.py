"""Opt-in storage profile: immutable legacy base, local Trainer I/O, managed Drive archives."""

import shutil
import time
import uuid
from dataclasses import replace
from pathlib import Path

from .checkpoint_codec import pack
from .checkpoints import directory_bytes, latest_checkpoint, seal, valid
from .config import Config, Experiment
from .io import atomic_json, read_json
from .storage import ManagedStore, StorageBlocked, StoragePolicy, model_export

TOKENIZER_FILES = [
    "config.json",
    "generation_config.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
]


def pinned_snapshot(config, cache_dir, include_weights=False):
    from huggingface_hub import snapshot_download

    return snapshot_download(
        config.model_id,
        revision=config.model_revision,
        cache_dir=str(cache_dir),
        allow_patterns=TOKENIZER_FILES + (["pytorch_model.bin"] if include_weights else []),
    )


def load_run_config(root):
    raw = read_json(Path(root) / "run_config.json")
    raw["experiments"] = [Experiment(**row) for row in raw["experiments"]]
    return Config(**raw)


def existing_checkpoints(root, config):
    found = {}
    for experiment in config.experiments:
        path = latest_checkpoint(Path(root) / "experiments" / experiment.name, config.fingerprint)
        if path:
            found[experiment.name] = Path(path)
    return found


class StorageRecovery:
    def __init__(self, root, backend, local_root, policy=None, config=None):
        self.root = Path(root).resolve()
        self.backend = backend
        self.local_root = Path(local_root).resolve()
        if self.local_root.is_relative_to(self.root) or "/drive/" in str(self.local_root):
            raise ValueError("Local staging/cache must be outside persistent Google Drive")
        self.policy = policy or StoragePolicy()
        self.control_root = self.root / "storage-v1"
        if (self.root / "run_config.json").exists():
            self.config = load_run_config(root)
        elif config is not None:
            quota = self.backend.quota()
            self.policy.require(quota, 6_000_000_000)
            preliminary = self.policy.plan(
                quota, 4_500_000_000, 1_500_000_000, 1_500_000_000, protected_bytes=1_500_000_000
            )
            if not preliminary["full_plan_fits"]:
                raise StorageBlocked(
                    "Fresh-run worst-case quota budget does not fit; no model downloaded"
                )
            from .training import resolve_revisions

            self.root.mkdir(parents=True, exist_ok=True)
            self.config = resolve_revisions(config, self.root)
        else:
            raise StorageBlocked(
                "Existing run_config.json or a fresh-run configuration is required"
            )
        if config is not None:
            requested = replace(
                config,
                model_revision=self.config.model_revision
                if config.model_revision == "main"
                else config.model_revision,
                dataset_revision=self.config.dataset_revision
                if config.dataset_revision == "main"
                else config.dataset_revision,
            )
            if requested.fingerprint != self.config.fingerprint:
                raise StorageBlocked(
                    "Requested configuration differs from the saved run; no resume allowed"
                )
            self.config = replace(
                self.config,
                max_hours=config.max_hours,
                session_hours=config.session_hours,
                max_storage_gb=config.max_storage_gb,
                reserve_gb=config.reserve_gb,
            )
        self.legacy = existing_checkpoints(root, self.config)
        if not self.backend.read_index():
            for path in self.legacy.values():
                if read_json(path / "trainer_state.json").get("best_metric") is not None:
                    raise StorageBlocked(
                        "Legacy checkpoint already has validation-best history. "
                        "This migration is for the reported pre-evaluation step-10 run; "
                        "refusing to discard best-model history."
                    )
        self.base_checkpoint = (
            min(self.legacy.values(), key=lambda p: int(p.name.split("-")[-1]))
            if self.legacy
            else None
        )
        self.checkpoint_bytes = (
            directory_bytes(self.base_checkpoint) if self.base_checkpoint else 4_500_000_000
        )
        self.model_bytes = (
            sum(p.stat().st_size for p in self.base_checkpoint.glob("*.safetensors"))
            if self.base_checkpoint
            else 1_500_000_000
        )
        self.project_bytes = directory_bytes(root)
        self.protected_bytes = sum(directory_bytes(p) for p in self.legacy.values())
        self.state = None
        self.store = None
        self.resumes = {}

    def audit_plan(self):
        """No writes, downloads, model loads or GPU work."""
        quota = self.backend.quota()
        managed = self.backend.managed_bytes()
        return self.policy.plan(
            quota,
            self.checkpoint_bytes,
            self.model_bytes,
            self.project_bytes + managed,
            self.protected_bytes,
            managed,
            current_archive_bytes=(self.backend.read_index() or {}).get("latest", {}).get("size")
            if (self.backend.read_index() or {}).get("latest")
            else None,
        )

    def _local_space(self):
        self.local_root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.local_root).free < 4 * self.checkpoint_bytes + self.model_bytes:
            raise StorageBlocked(
                "Need local space for checkpoint, immutable base, tar staging and optimizer I/O"
            )

    def begin(self, config, root, for_training=True):
        if Path(root).resolve() != self.root or config.fingerprint != self.config.fingerprint:
            raise StorageBlocked("Legacy training semantics changed; refusing checkpoint migration")
        # Fail before copying GBs, downloading a base model, or initializing CUDA.
        self.policy.require(self.backend.quota(), self.checkpoint_bytes)
        self._local_space()
        base = self.local_root / "immutable-base"
        index = self.backend.read_index() or {}
        self.store = ManagedStore(self.backend, self.local_root, config.fingerprint, self.policy)
        if not base.exists():
            if self.base_checkpoint:
                model_export(self.base_checkpoint, base, config.fingerprint)
            elif index.get("base"):
                self.store.restore(index["base"], base)
            else:
                from transformers import AutoTokenizer

                from .training import load_model

                snapshot = pinned_snapshot(
                    config, self.local_root / "hf-cache", include_weights=True
                )
                model = load_model(config, cache_dir=self.local_root / "hf-cache")
                model.save_pretrained(base, safe_serialization=True)
                AutoTokenizer.from_pretrained(
                    snapshot, use_fast=False, local_files_only=True
                ).save_pretrained(base)
                atomic_json(base / "run_config.json", config.to_dict())
                seal(base, config.fingerprint, checkpoint=False)
                del model
                self.model_bytes = sum(p.stat().st_size for p in base.glob("*.safetensors"))
                self.checkpoint_bytes = self.model_bytes * 3 + 50_000_000
        if not valid(base, config.fingerprint):
            raise StorageBlocked("Local immutable compression base is invalid")
        self.model_bytes = sum(p.stat().st_size for p in base.glob("*.safetensors"))
        if self.base_checkpoint is None:
            self.checkpoint_bytes = self.model_bytes * 3 + 50_000_000
            # A prior upload/index publication can fail after the local export
            # succeeds. Never let its mere existence bypass durable base storage.
            if not self.store.index.get("base"):
                artifact = self.store.upload(base, "base", "pretrained", 0)
                self.store.index["base"] = artifact
                self.store.publish()
        self.store.base_dir = base
        self.store.cleanup()
        self.state = self.store.index.get("state") or {
            "experiments": read_json(self.root / "experiments.json")
            if (self.root / "experiments.json").exists()
            else {},
            "selections": {},
            "pending_best": None,
        }
        self.control_root.mkdir(parents=True, exist_ok=True)
        for name in ("experiments.json", "budget.json"):
            target = self.control_root / name
            if not target.exists() and (self.root / name).exists():
                shutil.copy2(self.root / name, target)
        if self.state.get("experiments"):
            atomic_json(self.control_root / "experiments.json", self.state["experiments"])
        latest = self.store.index.get("latest")
        if latest:
            destination = self.output_dir(latest["experiment"]) / f"checkpoint-{latest['step']}"
            self.resumes[latest["experiment"]] = self.store.restore(latest, destination)
        else:
            for name, source in self.legacy.items():
                target = self.output_dir(name) / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copytree(source, target)
                if not valid(target, config.fingerprint):
                    raise StorageBlocked("Copied legacy checkpoint failed validation")
                self.resumes[name] = target
        self.finish_pending_best()
        # Measure lossless compression without GPU training. Never promise a ratio.
        current = self.store.index.get("latest")
        if current:
            measured = current["size"]
        else:
            probe = self.local_root / "capacity-probe.tar.zst"
            try:
                pack(self.base_checkpoint or base, probe, base)
                measured = probe.stat().st_size if self.base_checkpoint else self.checkpoint_bytes
            finally:
                probe.unlink(missing_ok=True)
        managed = self.backend.managed_bytes()
        remote_base = (self.store.index.get("base") or {}).get("size", 0)
        plan = self.policy.plan(
            self.backend.quota(),
            self.checkpoint_bytes,
            self.model_bytes,
            self.project_bytes + managed,
            self.protected_bytes + remote_base,
            managed - remote_base,
            current_archive_bytes=measured,
            best_archive_bytes=(self.store.index.get("best") or {}).get("size"),
        )
        plan["measured_current_archive_bytes"] = measured
        atomic_json(self.control_root / "storage-plan.json", plan)
        if for_training and not plan["full_plan_fits"]:
            raise StorageBlocked(
                "Worst-case full-state lifecycle does not fit actual account quota. "
                "Current compression is not a guarantee for later checkpoints. "
                "No training started; inspect storage-plan.json and preserve the legacy checkpoint."
            )
        self.preflight()

    def preflight(self):
        """Reserve worst-case uncompressed replacement even when deltas compress well."""
        if (
            shutil.disk_usage(self.local_root).free
            < 3 * self.checkpoint_bytes + self.policy.safety_margin_bytes
        ):
            raise StorageBlocked(
                "Local checkpoint staging headroom is insufficient before training"
            )
        quota = self.backend.quota()
        latest = self.store.index.get("latest") if self.store else None
        reclaim = latest["size"] if latest else 0
        # Upload full first, validate/publish, retire only managed old full, then best.
        peak_extra = max(self.checkpoint_bytes, self.checkpoint_bytes - reclaim + self.model_bytes)
        self.policy.require(quota, peak_extra)
        return quota

    def seed_legacy(self):
        """No-GPU adoption and download/restore proof; original stays untouched."""
        self.begin(self.config, self.root, for_training=False)
        if self.store.index.get("latest"):
            artifact = self.store.index["latest"]
        else:
            if self.base_checkpoint is None:
                raise StorageBlocked("No legacy checkpoint exists to seed")
            name = next(name for name, path in self.legacy.items() if path == self.base_checkpoint)
            artifact = self.store.commit_checkpoint(
                self.resumes[name], name, int(self.base_checkpoint.name.split("-")[-1]), self.state
            )
        verification = self.local_root / ("verify-recovery-" + uuid.uuid4().hex)
        try:
            self.store.restore(artifact, verification)
            atomic_json(
                self.control_root / "recovery-verification.json",
                {
                    "artifact_id": artifact["id"],
                    "step": artifact["step"],
                    "archive_sha256": artifact["sha256"],
                    "manifest_sha256": artifact["manifest_sha256"],
                    "status": "downloaded_decoded_all_file_hashes_verified",
                },
            )
        finally:
            if verification.exists():
                shutil.rmtree(verification)  # Owned ephemeral verification copy only.
        return artifact

    def output_dir(self, name):
        return self.local_root / "experiments" / name

    def resume_for(self, name):
        return str(self.resumes[name]) if name in self.resumes else None

    def tokenizer_source(self):
        source = self.local_root / "immutable-base"
        if (source / "tokenizer_config.json").exists():
            return str(source)
        return pinned_snapshot(self.config, self.local_root / "hf-cache")

    def finish_pending_best(self):
        pending = self.state.get("pending_best") if self.state else None
        if not pending:
            return
        name, step, metrics = pending["experiment"], pending["step"], pending["metrics"]
        directory = self.output_dir(name) / f"checkpoint-{step}"
        if not directory.exists():
            directory = self.store.restore(self.store.index["latest"], directory)
        export = self.local_root / ("best-" + uuid.uuid4().hex)
        try:
            model_export(directory, export, self.config.fingerprint)
            self.store.commit_best(export, name, step, metrics)
            self.store.cleanup()
            self.state["pending_best"] = None
            self.store.index["state"] = self.state
            self.store.publish()
        finally:
            if export.exists():
                shutil.rmtree(export)  # Ephemeral, owned temporary export only.

    def callback(self, experiment, ledger, session_start):
        from transformers import TrainerCallback

        from .numerics import check_logs

        controller = self
        config = self.config

        class ManagedCallback(TrainerCallback):
            def __init__(self):
                self.last = time.monotonic()
                self.stopped = None
                self.trainer = None

            def tick(self):
                now = time.monotonic()
                ledger["seconds"] += now - self.last
                self.last = now
                atomic_json(controller.control_root / "budget.json", ledger)

            def on_step_begin(self, args, state, control, **kwargs):
                if state.global_step % 10 == 0:
                    controller.preflight()  # Raises before forward/backward, not after Adam.

            def on_step_end(self, args, state, control, **kwargs):
                self.tick()
                if ledger["seconds"] >= config.max_hours * 3600:
                    self.stopped = "total time budget"
                elif time.monotonic() - session_start >= config.session_hours * 3600:
                    self.stopped = "session time budget"
                try:
                    if state.global_step % 10 == 0:
                        controller.preflight()
                except StorageBlocked as exc:
                    self.stopped = str(exc)
                    # Never force an unsafe large write when quota was just rejected.
                    control.should_save = False
                    control.should_evaluate = False
                    control.should_training_stop = True
                    return control
                if self.stopped:
                    control.should_save = True
                    control.should_training_stop = True
                return control

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                metrics = dict(metrics or {})
                score = metrics["eval_rougeLsum"]
                selected = controller.state["selections"].setdefault(
                    experiment.name, {"metrics": None, "bad_evaluations": 0}
                )
                if selected["metrics"] is None or score > selected["metrics"]["eval_rougeLsum"]:
                    selected.update(metrics=metrics, step=state.global_step, bad_evaluations=0)
                else:
                    selected["bad_evaluations"] += 1
                best = controller.store.index.get("best")
                if not experiment.smoke and (
                    best is None or score > best["metrics"]["eval_rougeLsum"]
                ):
                    controller.state["pending_best"] = {
                        "experiment": experiment.name,
                        "step": state.global_step,
                        "metrics": metrics,
                    }
                if not experiment.smoke and selected["bad_evaluations"] >= 3:
                    control.should_training_stop = True
                return control

            def on_save(self, args, state, control, **kwargs):
                directory = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                atomic_json(directory / "run_config.json", config.to_dict())
                atomic_json(directory / "storage_state.json", controller.state)
                seal(directory, config.fingerprint)
                if not experiment.smoke:
                    controller.state["experiments"] = read_json(
                        controller.control_root / "experiments.json"
                    )
                    controller.store.commit_checkpoint(
                        directory, experiment.name, state.global_step, controller.state
                    )
                    controller.finish_pending_best()
                    controller.store.cleanup()
                self.tick()

            def on_log(self, args, state, control, logs=None, **kwargs):
                check_logs(args.output_dir, state, logs)

        return ManagedCallback()

    def finalize_saved_completion(self, experiment, checkpoint, states):
        if not checkpoint:
            return False
        saved = read_json(Path(checkpoint) / "trainer_state.json")
        selection = self.state["selections"].get(experiment.name, {})
        finished = (
            saved["global_step"] >= experiment.max_steps or selection.get("bad_evaluations", 0) >= 3
        )
        if not finished or not selection.get("metrics"):
            return False
        states[experiment.name] = {
            **states.get(experiment.name, {}),
            "status": "completed",
            "metrics": selection["metrics"],
            "best_step": selection.get("step"),
            "global_step": saved["global_step"],
            "training_loss": saved.get("log_history", []),
            "retention": "metrics retained; only global winner weights and current full state persisted",
        }
        self.state["experiments"] = states
        self.store.index["state"] = self.state
        self.store.publish()
        atomic_json(self.control_root / "experiments.json", states)
        return True

    def completion(self, experiment, trainer, states):
        selection = self.state["selections"].get(experiment.name, {})
        metrics = selection.get("metrics")
        if metrics is None:
            raise StorageBlocked("No validated selection metrics for completed experiment")
        record = {
            **states[experiment.name],
            "status": "completed",
            "metrics": metrics,
            "global_step": trainer.state.global_step,
            "best_step": selection.get("step"),
            "training_loss": trainer.state.log_history,
            "retention": "metrics retained; only global winner weights and current full state persisted",
        }
        states[experiment.name] = record
        self.state["experiments"] = states
        self.store.index["state"] = self.state
        self.store.publish()
        return record

    def persist_diagnostics(self, name):
        for filename in ("precision.json", "numerical_failure.json"):
            source = self.output_dir(name) / filename
            if source.exists():
                target = self.control_root / "diagnostics" / name / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    def release_completed_local(self, name):
        path = self.output_dir(name)
        if not path.resolve().is_relative_to(self.local_root):
            raise StorageBlocked("Unsafe local cleanup path")
        if path.exists():
            shutil.rmtree(
                path
            )  # Ephemeral only; all persistent legacy/managed artifacts are elsewhere.

    def best_path(self):
        best = self.store.index.get("best")
        if not best:
            raise StorageBlocked("No completed validation-selected model exists")
        return str(self.store.restore(best, self.local_root / ("selected-" + best["id"])))
