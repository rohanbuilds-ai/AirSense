"""Automated test verifying that all AirSense notebooks are valid nbformat v4 files."""

from pathlib import Path
import nbformat
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
NOTEBOOK_FILES = [
    "01_dataset_audit.ipynb",
    "02_linear_algebra_lab.ipynb",
    "03_calibration_experiments.ipynb",
    "04_probability_and_statistics_lab.ipynb",
]


@pytest.mark.parametrize("nb_name", NOTEBOOK_FILES)
def test_notebook_schema_and_execution_state(nb_name: str):
    """Verify that notebook file exists, parses as valid JSON, passes v4 schema, and has executed cells."""
    nb_path = NOTEBOOKS_DIR / nb_name
    assert nb_path.exists(), f"Notebook {nb_name} does not exist."

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Validate schema
    nbformat.validate(nb)

    # Check top-level properties
    assert nb.nbformat == 4
    assert len(nb.cells) > 0

    code_cells = [c for c in nb.cells if c.get("cell_type") == "code"]
    assert len(code_cells) > 0

    # Ensure all code cells have execution counts and cell IDs
    for cell in code_cells:
        assert "id" in cell
        assert cell.get("execution_count") is not None
