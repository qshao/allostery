#!/usr/bin/env python3
"""Build examples/kras_wt/demo.ipynb from cell definitions."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    }
    nb["cells"] = _cells()
    out = Path(__file__).parent / "demo.ipynb"
    nbf.write(nb, str(out))
    print(f"Wrote {out}  ({len(nb['cells'])} cells)")


def _cells() -> list[nbf.NotebookNode]:
    return []   # expanded in later tasks


if __name__ == "__main__":
    build()
