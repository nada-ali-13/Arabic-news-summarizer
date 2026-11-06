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
    namespace = {"json": json, "re": re, "Path": Path, "subprocess": subprocess}
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
