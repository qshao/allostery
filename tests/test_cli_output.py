from __future__ import annotations

import json
from pathlib import Path

from allostery.cli_output import Result, format_result


def _ok() -> Result:
    return Result(
        command="interpret",
        summary="interpret candidates={'hubs': 3} json=a.json md=a.md",
        data={"counts": {"hubs": 3}},
        artifacts=[Path("a.json"), Path("a.md")],
    )


def test_default_mode_emits_summary_on_stdout() -> None:
    out, err = format_result(_ok())
    assert out == "interpret candidates={'hubs': 3} json=a.json md=a.md"
    assert err == ""


def test_quiet_mode_emits_only_artifact_paths() -> None:
    out, err = format_result(_ok(), quiet=True)
    assert out == "a.json\na.md"
    assert err == ""


def test_json_mode_emits_parseable_object() -> None:
    out, err = format_result(_ok(), json_mode=True)
    payload = json.loads(out)
    assert payload["status"] == "ok"
    assert payload["command"] == "interpret"
    assert payload["artifacts"] == ["a.json", "a.md"]
    assert payload["data"] == {"counts": {"hubs": 3}}
    assert err == ""


def test_error_result_goes_to_stderr_in_default_mode() -> None:
    result = Result(command="analyze", status="error", error="no such file: x.csv")
    out, err = format_result(result)
    assert out == ""
    assert err == "no such file: x.csv"


def test_error_result_in_json_mode_is_on_stdout() -> None:
    result = Result(command="analyze", status="error", error="boom")
    out, err = format_result(result, json_mode=True)
    payload = json.loads(out)
    assert payload["status"] == "error"
    assert payload["error"] == "boom"
    assert err == ""
