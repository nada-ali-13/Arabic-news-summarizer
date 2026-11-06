import json
from dataclasses import replace
from pathlib import Path

import pytest

from arabic_summarizer.checkpoints import RecoveryError, latest_checkpoint, seal, valid
from arabic_summarizer.config import Experiment, load_config
from arabic_summarizer.data import clean_splits
from arabic_summarizer.io import atomic_json, read_json
from arabic_summarizer.metrics import ArabicTokenizer, rouge
from arabic_summarizer.text import normalize, word_count
from arabic_summarizer.training import update_best

ROOT = Path(__file__).resolve().parents[1]


def row(id_, text="نص عربي", summary="ملخص", url=""):
    return {"id": id_, "text": text, "summary": summary, "url": url, "title": "عنوان"}


def test_normalization_retains_arabic_information():
    assert normalize("  أَخبار\u200f عـربية\n ٢٠٢٦  ") == "أَخبار عربية ٢٠٢٦"
    assert normalize("إلى آفاق مسؤولية ة ى") == "إلى آفاق مسؤولية ة ى"
    assert word_count("ألف  باء\nجيم") == 3
    with pytest.raises(ValueError):
        normalize(None)


def test_schema_empty_and_leakage_test_priority():
    data = {
        "test": [row("test"), row("test-copy")],
        "validation": [row("duplicate", "نص عربي"), row("v", "خبر مستقل")],
        "train": [
            row("test", "تغيير نص"),
            row("duplicate-url", "شيء", url="https://x"),
            row("t", "مقال جديد"),
            row("empty", ""),
        ],
    }
    data["validation"][1]["url"] = "https://x"
    clean, audit = clean_splits(data)
    assert len(clean["test"]) == 2  # Retain usable official test rows.
    assert [r["id"] for r in clean["validation"]] == ["v"]
    assert [r["id"] for r in clean["train"]] == ["t"]
    assert audit["train"]["removed"] == 3
    data["train"][0].pop("summary")
    with pytest.raises(ValueError, match="schema"):
        clean_splits(data)


@pytest.mark.parametrize("path", ["dev.json", "full.json"])
def test_configs(path):
    config = load_config(ROOT / "configs" / path)
    assert config.experiments[0].smoke
    assert replace(config, max_hours=20).fingerprint == config.fingerprint
    assert replace(config, seed=1).fingerprint != config.fingerprint


@pytest.mark.parametrize(
    "changes", [{"name": "../escape"}, {"max_steps": 0}, {"save_steps": 7}, {"learning_rate": -1}]
)
def test_bad_experiment(changes):
    with pytest.raises(ValueError):
        Experiment(**{**dict(name="valid", max_steps=2, train_samples=8), **changes})


def test_arabic_rouge_nonzero_and_not_english_tokenizer():
    assert ArabicTokenizer().tokenize("أخبار، عربية! ٢٠٢٦") == ["أخبار", "عربية", "٢٠٢٦"]
    exact = rouge(["هذا خبر عربي."], ["هذا خبر عربي."])
    assert all(score == 100 for score in exact.values())
    assert rouge(["مكتبة جديدة"], ["مكتبة قديمة"])["rouge1"] == 50
    assert rouge(["جامعة"], ["مطار"])["rouge1"] == 0


def checkpoint(path):
    path.mkdir(parents=True)
    for name in [
        "config.json",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "model.safetensors",
    ]:
        (path / name).write_text("{}")
    atomic_json(path / "trainer_state.json", {"global_step": int(path.name.split("-")[1])})
    seal(path, "run")


def test_recovery_skips_incomplete_corrupt_and_wrong_configuration(tmp_path):
    checkpoint(tmp_path / "checkpoint-1")
    checkpoint(tmp_path / "checkpoint-2")
    (tmp_path / "checkpoint-2" / "optimizer.pt").write_text("tampered")
    (tmp_path / "checkpoint-3").mkdir()
    assert latest_checkpoint(tmp_path, "run").endswith("checkpoint-1")
    assert not valid(tmp_path / "checkpoint-1", "other")
    with pytest.raises(RecoveryError):
        latest_checkpoint(tmp_path, "other")
    assert latest_checkpoint(tmp_path / "new", "run") is None


def test_checkpoint_refuses_partial_state(tmp_path):
    (tmp_path / "model.safetensors").write_text("weights")
    with pytest.raises(RecoveryError):
        seal(tmp_path, "run")


def test_atomic_state_and_best_is_never_downgraded(tmp_path):
    atomic_json(tmp_path / "state.json", {"status": "عربي"})
    assert read_json(tmp_path / "state.json")["status"] == "عربي"
    states = {"good": {"status": "completed", "export": "/good", "metrics": {"eval_rougeLsum": 10}}}
    update_best(tmp_path, states, "run")
    states["bad"] = {"status": "completed", "export": "/bad", "metrics": {"eval_rougeLsum": 5}}
    assert update_best(tmp_path, states, "run")["path"] == "/good"
    assert read_json(tmp_path / "best.json")["path"] == "/good"


def test_notebook_schema_and_all_code_cells_compile():
    import nbformat

    notebook = nbformat.read(ROOT / "notebooks/Arabic_News_Summarizer_Training.ipynb", as_version=4)
    nbformat.validate(notebook)
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"cell-{i}", "exec")
            assert not cell.outputs
    source = json.dumps(notebook)
    assert "resume" in source.lower() and "drive.mount" in source


def test_recovery_rejects_dangling_best_model_reference(tmp_path):
    path = tmp_path / "checkpoint-1"
    checkpoint(path)
    atomic_json(
        path / "trainer_state.json",
        {"global_step": 1, "best_model_checkpoint": str(tmp_path / "missing")},
    )
    seal(path, "run")
    with pytest.raises(RecoveryError):
        latest_checkpoint(tmp_path, "run")
