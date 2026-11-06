"""Actual Adam/scheduler/RNG archive recovery from step 10; no GPU/HF downloads."""

from pathlib import Path

import pytest
import torch
from _storage_helpers import MemoryBackend
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (
    PreTrainedTokenizerFast,
    T5Config,
    T5ForConditionalGeneration,
    TrainerCallback,
)

from arabic_summarizer import training
from arabic_summarizer.checkpoints import seal, sha256
from arabic_summarizer.config import Config, Experiment
from arabic_summarizer.io import atomic_json, read_json
from arabic_summarizer.recovery import StorageRecovery
from arabic_summarizer.storage import ManagedStore, StorageBlocked, StoragePolicy


def prepare_legacy(tmp_path):
    torch.set_num_threads(1)
    root = tmp_path / "legacy-run"
    root.mkdir()
    config = Config(
        model_revision="test-model",
        dataset_revision="test-data",
        source_length=32,
        target_length=8,
        validation_samples=2,
        test_samples=2,
        max_hours=1,
        session_hours=1,
        max_storage_gb=1,
        reserve_gb=0.000001,
        experiments=[
            Experiment("smoke", 2, 2, accumulation=1, eval_steps=1, save_steps=1, smoke=True),
            Experiment("candidate", 12, 4, accumulation=1, eval_steps=12, save_steps=12),
            Experiment("second", 2, 4, accumulation=1, eval_steps=2, save_steps=2),
        ],
    )
    backend = Tokenizer(
        WordLevel(
            {"<pad>": 0, "</s>": 1, "<unk>": 2, "خبر": 3, "مكتبة": 4, "جديدة": 5}, unk_token="<unk>"
        )
    )
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        model_input_names=["input_ids", "attention_mask"],
    )
    model_config = T5Config(
        vocab_size=6,
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_kv=8,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        feed_forward_proj="gated-gelu",
        tie_word_embeddings=False,
    )

    def load(config, path=None, cache_dir=None):
        if path:
            return real_load(config, path)
        m = T5ForConditionalGeneration(model_config)
        m.config.use_cache = False
        m.generation_config.max_length = 8
        m.generation_config.num_beams = 4
        return m

    real_load = training.load_model
    ds = DatasetDict(
        {
            name: Dataset.from_list(
                [
                    {
                        "id": f"{name}-{i}",
                        "url": "",
                        "title": "خبر",
                        "text": "خبر مكتبة جديدة",
                        "summary": "مكتبة جديدة",
                    }
                    for i in range(4)
                ]
            )
            for name in ("train", "validation", "test")
        }
    )
    ds.save_to_disk(str(root / "prepared"))
    atomic_json(root / "prepared/complete.json", {"fingerprint": config.fingerprint})
    atomic_json(root / "run_config.json", config.to_dict())
    atomic_json(root / "source_commit.json", {"commit": "52525c54f683a244f1df71a6403c7e3984d5ac49"})
    atomic_json(
        root / "experiments.json",
        {"smoke": {"status": "completed"}, "candidate": {"status": "paused"}},
    )
    atomic_json(root / "budget.json", {"seconds": 0})
    atomic_json(root / "baseline_validation.json", {"metrics": {"unit_test_fixture": 0}})
    rows = training.tokenize(ds["train"], tokenizer, config)

    class StopAtTen(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 10:
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            atomic_json(checkpoint / "run_config.json", config.to_dict())
            seal(checkpoint, config.fingerprint)

    trainer = training.build_trainer(
        config,
        config.experiments[1],
        root / "experiments/candidate",
        load(config),
        tokenizer,
        rows,
        rows.select([0, 1]),
        [StopAtTen()],
    )
    trainer.train()
    assert trainer.state.global_step == 10
    return root, config, load


def test_existing_step10_is_preserved_and_managed_recovery_continues_without_extra_steps(
    tmp_path, monkeypatch
):
    root, config, loader = prepare_legacy(tmp_path)
    before = {str(p.relative_to(root)): sha256(p) for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(training, "load_model", loader)
    # If this is accidentally called, dataset cache reuse/regression fails.
    monkeypatch.setattr(
        training, "load_official", lambda *a, **kw: pytest.fail("dataset redownload")
    )
    backend = MemoryBackend()
    policy = StoragePolicy(safety_margin_bytes=1000)
    seed_controller = StorageRecovery(root, backend, tmp_path / "local", policy)
    seed = seed_controller.seed_legacy()
    assert seed["step"] == 10
    seeded = seed_controller.store.restore(seed, tmp_path / "seed-roundtrip")
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors"):
        assert (seeded / name).read_bytes() == (
            root / "experiments/candidate/checkpoint-10" / name
        ).read_bytes()
    pauses = []
    result = None
    for _ in range(12):
        controller = StorageRecovery(root, backend, tmp_path / "local", policy)
        try:
            result = training.run(config, root, allow_cpu=True, storage=controller)
        except StorageBlocked as exc:
            pauses.append(str(exc))
        if result and result["status"] == "completed":
            break
        store = ManagedStore(backend, tmp_path / "cleanup-local", config.fingerprint, policy)
        if store.index.get("obsolete"):
            plan = store.cleanup_plan()
            store.cleanup(confirmation=plan["confirmation"])
        elif result:
            pytest.fail(str(result))
    assert result and result["status"] == "completed", pauses
    states = read_json(root / "storage-v1/experiments.json")
    assert states["candidate"]["global_step"] == 12
    assert states["second"]["global_step"] == 2
    assert states["candidate"]["training_loss"]
    assert (root / "storage-v1/diagnostics/candidate/precision.json").exists()
    assert read_json(root / "storage-v1/recovery-verification.json")["step"] == 10
    # Restore the most recent full state: optimizer progress is exact, not a weights-only restart.
    store = ManagedStore(
        backend,
        tmp_path / "readback",
        config.fingerprint,
        policy,
        base_dir=tmp_path / "local/immutable-base",
    )
    checkpoint = store.restore(store.index["latest"], tmp_path / "roundtrip")
    opt = torch.load(checkpoint / "optimizer.pt", weights_only=True)
    assert all(int(s["step"]) == 2 for s in opt["state"].values())
    # Byte-for-byte preservation of every pre-existing artifact, including optimizer10 and all original metadata.
    assert all(
        (root / name).exists() and sha256(root / name) == value for name, value in before.items()
    )
    candidate_archive = next(
        row
        for row in backend.deleted_records.values()
        if row["experiment"] == "candidate" and row["step"] == 12
    )
    from arabic_summarizer.checkpoint_codec import unpack

    packed = tmp_path / "candidate12.tar.zst"
    packed.write_bytes(candidate_archive["data"])
    unpack(packed, tmp_path / "candidate12", 1_000_000, tmp_path / "local/immutable-base")
    candidate_opt = torch.load(tmp_path / "candidate12/optimizer.pt", weights_only=True)
    assert all(int(s["step"]) == 12 for s in candidate_opt["state"].values())
    assert torch.load(tmp_path / "candidate12/scheduler.pt", weights_only=True)["last_epoch"] == 12
    assert backend.deleted  # Only newly owned immutable archives, after explicit confirmations.
    live = [value for value in backend.files.values() if value["kind"] == "full"]
    assert len(live) == 1
    assert len([v for v in backend.files.values() if v["kind"] == "weights"]) == 1
    assert not (
        root / "experiments/second"
    ).exists()  # No new full-state writes through Drive mount.


def test_low_quota_stops_before_model_load_and_does_not_touch_legacy(tmp_path, monkeypatch):
    root, config, _ = prepare_legacy(tmp_path)
    backend = MemoryBackend(limit=100_000, external_usage=99_000)
    controller = StorageRecovery(
        root, backend, tmp_path / "local", StoragePolicy(safety_margin_bytes=1000)
    )
    original = sha256(root / "experiments/candidate/checkpoint-10/optimizer.pt")
    monkeypatch.setattr(
        training, "load_model", lambda *a, **kw: pytest.fail("model loaded despite quota")
    )
    with pytest.raises(StorageBlocked, match="quota blocks"):
        training.run(config, root, allow_cpu=True, storage=controller)
    assert backend.uploads == 0 and backend.deleted == []
    assert sha256(root / "experiments/candidate/checkpoint-10/optimizer.pt") == original


def fresh_fixture(tmp_path, monkeypatch):
    """Real local model/tokenizer stand in for the pinned HF snapshot, not Trainer."""
    import shutil
    from dataclasses import replace

    from arabic_summarizer import recovery

    legacy, config, loader = prepare_legacy(tmp_path)
    config = replace(config, experiments=[config.experiments[0], config.experiments[2]])
    snapshot = tmp_path / "pretrained"
    loader(config).save_pretrained(snapshot, safe_serialization=False)
    for name in recovery.TOKENIZER_FILES:
        source = legacy / "experiments/candidate/checkpoint-10" / name
        if source.exists() and name not in {"config.json", "generation_config.json"}:
            shutil.copy2(source, snapshot / name)
    monkeypatch.setattr(recovery, "pinned_snapshot", lambda *a, **kw: str(snapshot))

    def resolve(config, root):
        atomic_json(root / "run_config.json", config.to_dict())
        return config

    monkeypatch.setattr(training, "resolve_revisions", resolve)
    monkeypatch.setattr(StorageRecovery, "_local_space", lambda self: None)
    root = tmp_path / "fresh"
    root.mkdir()
    shutil.copytree(legacy / "prepared", root / "prepared")
    atomic_json(root / "prepared/complete.json", {"fingerprint": config.fingerprint})
    return root, config


def test_fresh_managed_smoke_stays_local_and_base_survives_runtime_reset(tmp_path, monkeypatch):
    import shutil

    root, config = fresh_fixture(tmp_path, monkeypatch)
    backend = MemoryBackend()
    policy = StoragePolicy(safety_margin_bytes=1000)
    local = tmp_path / "fresh-local"
    result = None
    for _ in range(8):
        controller = StorageRecovery(root, backend, local, policy, config=config)
        try:
            result = training.run(controller.config, root, allow_cpu=True, storage=controller)
        except StorageBlocked:
            pass
        if result and result["status"] == "completed":
            break
        store = ManagedStore(backend, local, config.fingerprint, policy)
        if store.index.get("obsolete"):
            store.cleanup(confirmation=store.cleanup_plan()["confirmation"])
    assert result and result["status"] == "completed"
    all_artifacts = list(backend.files.values()) + list(backend.deleted_records.values())
    assert not any(row["experiment"] == "smoke" for row in all_artifacts)
    assert len([row for row in all_artifacts if row["kind"] == "base"]) == 1
    assert not (root / "experiments").exists()
    shutil.rmtree(local)
    # A runtime loss restores the durable base without fetching model weights again.
    from arabic_summarizer import recovery

    monkeypatch.setattr(recovery, "pinned_snapshot", lambda *a, **kw: pytest.fail("HF redownload"))
    again = StorageRecovery(root, backend, local, policy)
    again.begin(again.config, root)
    assert (local / "immutable-base/model.safetensors").is_file()
    assert again.store.index["latest"]["step"] == 2


def test_base_publication_retry_does_not_trust_only_local_export(tmp_path, monkeypatch):
    root, config = fresh_fixture(tmp_path, monkeypatch)
    backend = MemoryBackend()
    policy = StoragePolicy(safety_margin_bytes=1000)
    local = tmp_path / "retry-local"
    first = StorageRecovery(root, backend, local, policy, config=config)
    backend.fail_index = True
    with pytest.raises(OSError, match="publication"):
        first.begin(first.config, root)
    assert (local / "immutable-base/complete.json").exists()
    assert backend.index is None
    backend.fail_index = False
    again = StorageRecovery(root, backend, local, policy)
    again.begin(again.config, root)
    assert backend.index["base"]
    restored = ManagedStore(backend, local, config.fingerprint, policy).restore(
        backend.index["base"], tmp_path / "base-proof"
    )
    assert (restored / "model.safetensors").is_file()


def test_legacy_direct_drive_writes_are_blocked(tmp_path):
    from arabic_summarizer.config import load_config

    root = tmp_path / "drive/MyDrive/legacy"
    with pytest.raises(StorageBlocked, match="direct-to-Drive"):
        training.run(load_config("configs/dev.json"), root, allow_cpu=True)
    assert not list(root.rglob("*.pt"))


def test_recovery_notebook_defaults_are_non_destructive():
    import ast
    import json

    notebook = json.loads(
        Path("notebooks/Arabic_News_Summarizer_Storage_Recovery.ipynb").read_text()
    )
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assignments = {}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value.value
    for name in (
        "ENABLE_TRAINING",
        "SEED_LEGACY_WITHOUT_GPU",
        "APPLY_SMOKE_TRASH_DELETION",
        "APPLY_MANAGED_RETENTION",
    ):
        assert assignments[name] is False
    assert "runtime_source_commit.json" in code
