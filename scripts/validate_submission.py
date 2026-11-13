"""Static submission checks; never executes notebooks or prints secret values."""

import ast
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = (
    "README.md", "reports/FACTCHECK_SETUP.md", "reports/FACTCHECK_VALIDATION.md",
    "reports/FINAL_SUBMISSION.md", "reports/COMPATIBILITY.md", "deployment/README.md",
)


def check_submission(root=ROOT):
    errors = []
    for name in DOCS:
        path = root / name
        if not path.is_file():
            errors.append(f"missing_document:{name}")
            continue
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            target = target.strip().split(' "')[0].strip("<>")
            if urlsplit(target).scheme or target.startswith("#"):
                continue
            destination = unquote(target.split("#")[0])
            if not (path.parent / destination).exists():
                errors.append(f"broken_relative_link:{name}")
    notebook = json.loads((root / "notebooks/Khulasa_Final_Demo.ipynb").read_text())
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            if cell.get("outputs") or cell.get("execution_count") is not None:
                errors.append(f"saved_notebook_output:cell-{i}")
            try:
                ast.parse("".join(cell["source"]))
            except SyntaxError:
                errors.append(f"notebook_syntax:cell-{i}")
    # Validate local notebook paths in project GitHub and Colab links without network.
    readme = (root / "README.md").read_text()
    for ref in re.findall(r"(?:github\.com|colab\.research\.google\.com/github)/"
                          r"nada-ali-13/Arabic-news-summarizer/blob/([^\s)]+)", readme):
        prefix = "main/"
        if not ref.startswith(prefix) or not (root / ref[len(prefix):]).is_file():
            errors.append("invalid_project_notebook_url")
    # Only report paths, never matching contents.
    paths = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard"],
        text=True,
    ).splitlines()
    patterns = [
        re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
        re.compile(r"(?:gh[pousr]_|github_pat_|hf_)[0-9A-Za-z_]{25,}"),
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(r"/content/drive/" + r"MyDrive/(?!AUTHORIZED_PROJECT)[^\s\"'\\]+"),
    ]
    for name in sorted(set(paths)):
        path = root / name
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            continue
        if any(pattern.search(content) for pattern in patterns):
            errors.append(f"possible_secret_or_private_path:{name}")
    return errors


if __name__ == "__main__":
    issues = check_submission()
    if issues:
        raise SystemExit("\n".join(issues))
    print("Submission checks passed: relative links, notebook URLs/syntax/outputs, secret/path patterns.")
