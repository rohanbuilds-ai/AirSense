"""Validate all AirSense Jupyter notebooks using nbformat and schema checks.

Usage:
    python scripts/validate_notebooks.py
"""

import sys
from pathlib import Path
import nbformat

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOK_FILES = [
    "01_dataset_audit.ipynb",
    "02_linear_algebra_lab.ipynb",
    "03_calibration_experiments.ipynb",
    "04_probability_and_statistics_lab.ipynb",
]


def validate_notebook(nb_path: Path) -> dict:
    """Validate a single notebook for JSON, schema, cell structures, and paths."""
    report = {
        "file": nb_path.name,
        "valid_json": False,
        "valid_schema": False,
        "total_cells": 0,
        "code_cells": 0,
        "markdown_cells": 0,
        "executed_cells": 0,
        "unexecuted_cells": 0,
        "hardcoded_absolute_paths": [],
        "errors": [],
    }

    if not nb_path.exists():
        report["errors"].append(f"File not found: {nb_path}")
        return report

    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        report["valid_json"] = True
    except Exception as e:
        report["errors"].append(f"JSON / read error: {e}")
        return report

    try:
        nbformat.validate(nb)
        report["valid_schema"] = True
    except Exception as e:
        report["errors"].append(f"Schema validation error: {e}")

    report["total_cells"] = len(nb.cells)

    for idx, cell in enumerate(nb.cells):
        ctype = cell.get("cell_type", "")
        source = cell.get("source", "")

        if ctype == "markdown":
            report["markdown_cells"] += 1
        elif ctype == "code":
            report["code_cells"] += 1
            exec_count = cell.get("execution_count")
            outputs = cell.get("outputs", [])

            if exec_count is not None or len(outputs) > 0:
                report["executed_cells"] += 1
            else:
                report["unexecuted_cells"] += 1

            # Check for hardcoded absolute paths like C:\ or /home/
            if isinstance(source, str):
                lines = source.splitlines()
            else:
                lines = source

            for line_idx, line in enumerate(lines):
                if any(p in line for p in ["C:\\", "E:\\", "D:\\", "/home/"]):
                    # Allow comments or docstrings if any, but flag for audit
                    report["hardcoded_absolute_paths"].append(
                        f"Cell {idx} (Line {line_idx + 1}): {line.strip()}"
                    )

    return report


def main() -> int:
    """Validate all notebooks and print summary."""
    print("=" * 80)
    print("  AIRSENSE JUPYTER NOTEBOOK AUDIT & VALIDATION")
    print("=" * 80)

    all_passed = True

    for nb_name in NOTEBOOK_FILES:
        nb_path = NOTEBOOKS_DIR / nb_name
        res = validate_notebook(nb_path)

        status = "PASSED" if res["valid_json"] and res["valid_schema"] and len(res["errors"]) == 0 else "FAILED"
        if status == "FAILED":
            all_passed = False

        print(f"\n[{status}] {res['file']}")
        print(f"    - Valid JSON: {res['valid_json']}")
        print(f"    - Valid nbformat v4 Schema: {res['valid_schema']}")
        print(f"    - Total Cells: {res['total_cells']} (Markdown: {res['markdown_cells']}, Code: {res['code_cells']})")
        print(f"    - Executed Cells: {res['executed_cells']} | Unexecuted Cells: {res['unexecuted_cells']}")

        if res["hardcoded_absolute_paths"]:
            print(f"    - WARNING: {len(res['hardcoded_absolute_paths'])} hardcoded absolute paths detected:")
            for p in res["hardcoded_absolute_paths"][:3]:
                print(f"        * {p}")

        if res["errors"]:
            print(f"    - ERRORS: {res['errors']}")

    print("\n" + "=" * 80)
    if all_passed:
        print("  ALL 4 NOTEBOOKS VALIDATED SUCCESSFULLY (Valid JSON & nbformat v4 Schema)")
        print("=" * 80)
        return 0
    else:
        print("  ONE OR MORE NOTEBOOKS FAILED VALIDATION")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
