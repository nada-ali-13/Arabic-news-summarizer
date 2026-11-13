"""Submission notebook helpers run against local fixtures; no Drive/API/model access."""

import ast
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Khulasa_Final_Demo.ipynb"


def cell_with(marker):
    return next(c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells
                if c.cell_type == "code" and marker in c.source)


def helper(name):
    source = cell_with(f"def {name}(")
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"Path": Path, "subprocess": subprocess, "json": json}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook-helper", "exec"), namespace)
    return namespace[name]


def model_files(tmp_path):
    for name, value in {"config.json": '{"model_type":"t5"}',
                        "tokenizer_config.json": '{}', "spiece.model": 'fixture',
                        "model.safetensors": 'fixture, not model weights'}.items():
        (tmp_path / name).write_text(value)
    return tmp_path


def test_final_notebook_schema_and_outputs():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, "final-notebook", "exec")
            assert not cell.outputs and cell.execution_count is None
    source = NOTEBOOK.read_text()
    assert "USE_TF" in source and "USE_FLAX" in source
    assert "KHULASA_GDELT_ENABLED" in source and "getpass" in source
    assert "userdata.get" in source and "AUTHORIZED_PROJECT" not in source


def test_model_validation_requires_tokenizer_and_safe_complete_shards(tmp_path):
    validate = helper("validate_model_dir")
    root = model_files(tmp_path)
    assert validate(root) == root.resolve()
    (root / "spiece.model").unlink()
    with pytest.raises(ValueError):
        validate(root)
    (root / "spiece.model").write_text('fixture')
    index = root / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "part.safetensors"}}))
    with pytest.raises(ValueError):
        validate(root)
    (root / "part.safetensors").write_text('fixture')
    assert validate(root) == root.resolve()
    index.write_text(json.dumps({"weight_map": {"a": "../outside.safetensors"}}))
    with pytest.raises(ValueError):
        validate(root)


def test_notebook_checkout_is_repeatable_and_preserves_dirty_files(tmp_path):
    remote = tmp_path / "remote"
    subprocess.run(["git", "init", "-b", "feature/fact-checking", str(remote)],
                   check=True, capture_output=True)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(remote), *args], text=True).strip()
    git("config", "user.name", "Fixture")
    git("config", "user.email", "test@example.invalid")
    (remote / "file.txt").write_text('first')
    git("add", ".")
    git("commit", "-m", "fixture")
    checkout = helper("prepare_checkout")
    target = tmp_path / "checkout"
    first = checkout(target, str(remote), "feature/fact-checking")
    assert first == checkout(target, str(remote), "feature/fact-checking")
    (remote / "file.txt").write_text('second')
    git("commit", "-am", "advance")
    assert first != checkout(target, str(remote), "feature/fact-checking")
    (target / "file.txt").write_text('local work')
    with pytest.raises(RuntimeError):
        checkout(target, str(remote), "feature/fact-checking")
    assert (target / "file.txt").read_text() == 'local work'


def test_notebook_full_model_loading_precedes_launch(tmp_path, monkeypatch):
    import getpass

    import app
    from arabic_summarizer import inference

    root = model_files(tmp_path)
    events = []
    class Model:
        device = "fixture-cpu"
        def __init__(self, model_path):
            assert Path(model_path) == root
        def load(self):
            events.append("load")
    monkeypatch.setattr(inference, "Summarizer", Model)
    monkeypatch.setattr(getpass, "getpass", lambda _: str(root))
    monkeypatch.setenv("ARABIC_SUMMARIZER_MODEL", "fixture")
    namespace = {"Path": Path, "os": __import__("os"), "FULL_DEMO": True,
                 "SHARE_PUBLIC": False}
    exec(compile(cell_with("def validate_model_dir("), "load-cell", "exec"), namespace)
    def build_app(summarizer):
        assert isinstance(summarizer, Model)
        events.append("build")
        return SimpleNamespace(launch=lambda **kwargs: events.append("launch"))
    monkeypatch.setattr(app, "build_app", build_app)
    exec(compile(cell_with("from app import build_app"), "launch-cell", "exec"), namespace)
    assert events == ["load", "build", "launch"]


def test_installation_preserves_existing_torch_constraint(tmp_path):
    source = cell_with("torch_before =")
    nodes = ast.parse(source).body
    start = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "torch_before" for t in node.targets))
    calls = []
    def install(args, **kwargs):
        constraint = Path(args[args.index("-c") + 1]).read_text()
        assert constraint == "torch==2.8.0+cu126\n"
        calls.append(args)
    namespace = {"version": lambda name: "2.8.0+cu126" if name == "torch" else "5.49.1",
                 "tempfile": __import__("tempfile"), "Path": Path,
                 "subprocess": SimpleNamespace(run=install), "sys": __import__("sys"),
                 "REPO_DIR": tmp_path,
                 "os": SimpleNamespace(chdir=lambda _: None)}
    original_path = list(namespace["sys"].path)
    try:
        exec(compile(ast.Module(body=nodes[start:], type_ignores=[]), "install-cell", "exec"), namespace)
    finally:
        namespace["sys"].path[:] = original_path
    assert len(calls) == 1 and "--upgrade" not in calls[0]


def test_numeric_evaluation_evidence_is_unchanged():
    import hashlib

    current = json.loads((ROOT / "reports/metrics_summary.json").read_text())
    current.pop("models")
    digest = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
    assert digest == "8ca3de68060823746ad1ce7c43c3e2f99e62264d232bb209babc6be11bc3b874"


def test_submission_static_audit():
    from scripts.validate_submission import check_submission

    assert check_submission() == []
