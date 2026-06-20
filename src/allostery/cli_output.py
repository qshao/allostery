from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Result:
    command: str
    status: str = "ok"
    summary: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)
    error: str | None = None


def format_result(
    result: Result,
    *,
    json_mode: bool = False,
    quiet: bool = False,
) -> tuple[str, str]:
    if json_mode:
        payload = {
            "command": result.command,
            "status": result.status,
            "summary": result.summary,
            "data": result.data,
            "artifacts": [str(path) for path in result.artifacts],
            "error": result.error,
        }
        return json.dumps(payload, indent=2), ""

    if result.status == "error":
        return "", result.error or "error"

    if quiet:
        return "\n".join(str(path) for path in result.artifacts), ""

    return result.summary, ""


__all__ = ["Result", "format_result"]
