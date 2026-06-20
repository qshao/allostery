from __future__ import annotations

import pytest

from allostery.cli import build_parser


def test_help_lists_new_commands_and_flags(capsys) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    text = capsys.readouterr().out
    assert "interpret" in text
    assert "workflow" in text
    assert "--json" in text
    assert "--quiet" in text


def test_readme_documents_interpret_and_workflow() -> None:
    from pathlib import Path
    readme = Path(__file__).resolve().parent.parent / "README.md"
    body = readme.read_text(encoding="utf-8")
    assert "allostery interpret" in body
    assert "allostery workflow" in body


def test_help_lists_validate_command() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    # capsys is unavailable here; re-render help text directly
    text = parser.format_help()
    assert "validate" in text


def test_readme_documents_validate() -> None:
    from pathlib import Path
    readme = Path(__file__).resolve().parent.parent / "README.md"
    body = readme.read_text(encoding="utf-8")
    assert "allostery validate" in body
