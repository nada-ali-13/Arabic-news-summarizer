"""Execute the actual notebook checkout helper against a local Git remote."""

import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def checkout_source():
    nb = json.loads((ROOT / "notebooks/Arabic_News_Summarizer_Training.ipynb").read_text())
    source = next(
        "".join(cell["source"])
        for cell in nb["cells"]
        if "def checkout_source(" in "".join(cell["source"])
    )
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
    namespace = {"ast": ast, "json": json, "re": re, "Path": Path, "subprocess": subprocess}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "notebook_checkout", "exec"),
        namespace,
    )
    return namespace["checkout_source"]


@pytest.fixture
def remote(tmp_path):
    path = tmp_path / "remote"
    path.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(path)], check=True, capture_output=True)
    git(path, "config", "user.email", "test@example.invalid")
    git(path, "config", "user.name", "Test")
    config = path / "src/arabic_summarizer/config.py"
    config.parent.mkdir(parents=True)
    config.write_text("class Config:\n    training_schema_version: int = 2\n")
    (config.parent / "storage.py").write_text("# storage feature fixture\n")
    (path / "version.txt").write_text("old")
    git(path, "add", ".")
    git(path, "commit", "-m", "old")
    return path


def advance(remote):
    (remote / "version.txt").write_text("fixed")
    git(remote, "add", ".")
    git(remote, "commit", "-m", "fixed")
    return git(remote, "rev-parse", "HEAD")


def test_new_run_fetches_main_even_with_stale_existing_checkout(tmp_path, remote, checkout_source):
    checkout = tmp_path / "checkout"
    old = checkout_source(str(remote), "main", checkout, tmp_path / "old-run")
    fixed = advance(remote)
    assert old != fixed
    result = checkout_source(str(remote), "main", checkout, tmp_path / "new-v2-run")
    assert result == fixed
    assert (checkout / "version.txt").read_text() == "fixed"
    assert json.loads((tmp_path / "old-run/source_commit.json").read_text())["commit"] == old
    assert checkout_source(str(remote), "main", checkout, tmp_path / "old-run") == old
    assert (checkout / "version.txt").read_text() == "old"


def test_exact_fixed_commit_is_frozen_and_dirty_edits_preserved(tmp_path, remote, checkout_source):
    checkout, run = tmp_path / "checkout", tmp_path / "new-run"
    fixed = advance(remote)
    assert checkout_source(str(remote), fixed, checkout, run) == fixed
    (checkout / "version.txt").write_text("local user work")
    with pytest.raises(RuntimeError, match="local edits"):
        checkout_source(str(remote), fixed, checkout, run)
    assert (checkout / "version.txt").read_text() == "local user work"


def test_old_main_is_rejected_before_new_source_lock_is_written(tmp_path, remote, checkout_source):
    config = remote / "src/arabic_summarizer/config.py"
    config.write_text("class Config:\n    seed: int = 42\n")
    git(remote, "add", ".")
    git(remote, "commit", "-m", "simulate pre-fix main")
    checkout, run = tmp_path / "checkout", tmp_path / "fresh-run"
    with pytest.raises(RuntimeError, match="No new source lock was written") as error:
        checkout_source(str(remote), "main", checkout, run)
    assert git(remote, "rev-parse", "HEAD") in str(error.value)
    assert "training_schema=None" in str(error.value)
    assert not (run / "source_commit.json").exists()


def test_incompatible_existing_source_lock_is_preserved(tmp_path, remote, checkout_source):
    from arabic_summarizer.io import atomic_json

    config = remote / "src/arabic_summarizer/config.py"
    config.write_text("class Config:\n    seed: int = 42\n")
    git(remote, "add", ".")
    git(remote, "commit", "-m", "old source")
    old = git(remote, "rev-parse", "HEAD")
    config.write_text("class Config:\n    training_schema_version: int = 2\n")
    git(remote, "add", ".")
    git(remote, "commit", "-m", "fixed source")
    run = tmp_path / "old-run"
    atomic_json(run / "source_commit.json", {"commit": old, "repository": str(remote)})
    before = (run / "source_commit.json").read_bytes()
    with pytest.raises(RuntimeError, match="Existing source lock was preserved"):
        checkout_source(str(remote), "main", tmp_path / "checkout", run)
    assert (run / "source_commit.json").read_bytes() == before


def test_retry_settings_are_assigned_only_in_setup_cell():
    nb = json.loads((ROOT / "notebooks/Arabic_News_Summarizer_Training.ipynb").read_text())
    settings = {"MODE", "CODE_REF", "SMOKE_ONLY", "RUN_NAME", "RUN_DIR"}
    occurrences = {key: [] for key in settings}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in settings:
                        occurrences[target.id].append(i)
    assert all(len(indices) == 1 for indices in occurrences.values())
    assert len({indices[0] for indices in occurrences.values()}) == 1
