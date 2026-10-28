"""Execute all AirSense Jupyter notebooks in dependency order and save executed outputs.

Usage:
    python scripts/execute_notebooks.py
"""

import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

NOTEBOOK_ORDER = [
    "01_dataset_audit.ipynb",
    "02_linear_algebra_lab.ipynb",
    "03_calibration_experiments.ipynb",
    "04_probability_and_statistics_lab.ipynb",
]


def execute_notebook(nb_path: Path) -> tuple[bool, str]:
    """Execute a single notebook and save outputs."""
    print(f"\n---> Executing {nb_path.name}...")
    t0 = time.time()

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Normalize cell IDs (nbformat 5.1.4+)
    for idx, cell in enumerate(nb.cells):
        if "id" not in cell:
            cell["id"] = f"cell-{idx:04d}"

    client = NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )

    try:
        client.execute()
        duration = time.time() - t0
        print(f"     [SUCCESS] Completed in {duration:.2f}s")

        # Save executed notebook with outputs
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        return True, f"Completed in {duration:.2f}s"
    except Exception as e:
        duration = time.time() - t0
        err_msg = f"Failed after {duration:.2f}s: {e}"
        print(f"     [ERROR] {err_msg}")
        return False, err_msg


def main() -> int:
    """Execute all notebooks in order."""
    print("=" * 80)
    print("  AIRSENSE JUPYTER NOTEBOOK EXECUTION PIPELINE")
    print("=" * 80)

    results = {}
    all_ok = True

    for nb_name in NOTEBOOK_ORDER:
        nb_path = NOTEBOOKS_DIR / nb_name
        success, msg = execute_notebook(nb_path)
        results[nb_name] = (success, msg)
        if not success:
            all_ok = False

    print("\n" + "=" * 80)
    print("  EXECUTION SUMMARY:")
    for nb_name, (success, msg) in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"  [{status}] {nb_name}: {msg}")
    print("=" * 80)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
