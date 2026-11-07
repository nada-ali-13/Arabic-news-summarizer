"""Validate notebook artifacts without executing cells, training, or fetching weights."""

from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
FINAL_NAMES = {
    "Arabic_News_Training_1200.ipynb",
    "Arabic_News_Evaluation.ipynb",
    "Arabic_News_Gradio.ipynb",
}


@pytest.mark.parametrize("path", sorted((ROOT / "notebooks").glob("*.ipynb")), ids=lambda p: p.name)
def test_notebook_schema_and_code_syntax(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    if path.name in FINAL_NAMES:
        assert "widgets" not in notebook.metadata
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{path.name}:cell-{index}", "exec")
            if path.name in FINAL_NAMES:
                assert cell.outputs == []
                assert cell.execution_count is None
                assert "outputId" not in cell.metadata
