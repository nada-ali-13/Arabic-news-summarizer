import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from arabic_summarizer.config import load_config
from arabic_summarizer.data import load_official
from arabic_summarizer.io import read_json
from arabic_summarizer.training import resolve_revisions

ROOT = Path(__file__).resolve().parents[1]


def test_real_official_archive_parser_without_network(tmp_path, monkeypatch):
    import huggingface_hub

    archive = tmp_path / "arabic.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        for split in ("train", "val", "test"):
            row = {
                "id": split,
                "text": "مقال " + split,
                "summary": "ملخص عربي",
                "title": "خبر",
                "url": "",
            }
            body = (json.dumps(row, ensure_ascii=False) + "\n").encode()
            info = tarfile.TarInfo(f"nested/arabic_{split}.jsonl")
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    calls = []

    def download(repo_id, filename, **kwargs):
        calls.append((repo_id, filename, kwargs))
        return str(archive)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    config = load_config(ROOT / "configs/dev.json")
    ds = load_official(config, tmp_path / "cache")
    assert set(ds) == {"train", "validation", "test"}
    assert ds["validation"][0]["id"] == "val"
    assert calls[0][:2] == ("csebuetnlp/xlsum", "data/arabic_XLSum_v2.0.tar.bz2")
    assert calls[0][2]["repo_type"] == "dataset"
    assert read_json(tmp_path / "dataset_statistics.json")["train"]["kept"] == 1


def test_revisions_lock_and_semantic_changes_rejected(tmp_path, monkeypatch):
    from huggingface_hub import HfApi

    calls = []

    def model_info(*args, **kwargs):
        calls.append("model")
        return SimpleNamespace(sha="model-commit")

    monkeypatch.setattr(HfApi, "model_info", model_info)
    monkeypatch.setattr(HfApi, "dataset_info", lambda *a, **kw: SimpleNamespace(sha="data-commit"))
    config = load_config(ROOT / "configs/dev.json")
    first = resolve_revisions(config, tmp_path)
    assert first.model_revision == "model-commit"
    assert first.dataset_revision == "data-commit"
    assert resolve_revisions(replace(config, max_hours=3), tmp_path).max_hours == 3
    assert calls == ["model"]
    with pytest.raises(ValueError, match="Configuration changed"):
        resolve_revisions(replace(config, seed=71), tmp_path)
    with pytest.raises(ValueError, match="revision changed"):
        resolve_revisions(replace(config, model_revision="new-commit"), tmp_path)


@pytest.mark.parametrize(
    "changes",
    [
        {"max_hours": float("nan")},
        {"max_storage_gb": float("inf")},
        {"validation_samples": 1.5},
        {"source_length": 2048},
        {"test_samples": 0},
    ],
)
def test_invalid_configuration(changes):
    with pytest.raises(ValueError):
        replace(load_config(ROOT / "configs/dev.json"), **changes)


def test_pretrained_loader_uses_original_weight_format_and_local_exports_are_safe(monkeypatch):
    from transformers import AutoModelForSeq2SeqLM

    from arabic_summarizer.training import load_model

    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(config=SimpleNamespace(), generation_config=SimpleNamespace())

    monkeypatch.setattr(AutoModelForSeq2SeqLM, "from_pretrained", fake)
    config = load_config(ROOT / "configs/dev.json")
    load_model(config)
    load_model(config, "/export")
    assert calls[0][1]["weights_only"] is True
    assert calls[0][1]["use_safetensors"] is False
    assert calls[0][1]["revision"] == config.model_revision
    assert calls[1][1]["use_safetensors"] is True
