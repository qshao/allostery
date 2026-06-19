from __future__ import annotations

import copy
from typing import Any

from allostery.interpret.llm import LLMBackend
from allostery.interpret.prompts import RESPONSE_SCHEMA, SYSTEM_PROMPT, build_user_prompt

_REQUIRED = ("summary", "mechanism_hypothesis", "key_residues", "confidence", "parametric", "caveats")


def _is_valid(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if any(key not in obj for key in _REQUIRED):
        return False
    if obj["confidence"] not in ("low", "medium", "high"):
        return False
    if not isinstance(obj["parametric"], bool):
        return False
    if not isinstance(obj["key_residues"], list):
        return False
    return True


def _interpret_item(candidate_type: str, item: dict[str, Any], backend: LLMBackend) -> dict[str, Any]:
    user = build_user_prompt(candidate_type, item)
    last: Any = None
    for _attempt in range(2):
        last = backend.generate_json(SYSTEM_PROMPT, user, RESPONSE_SCHEMA)
        if _is_valid(last):
            return last
    return {"invalid": True, "raw": last}


def interpret_report(report: dict[str, Any], backend: LLMBackend) -> dict[str, Any]:
    enriched = copy.deepcopy(report)
    candidates = enriched["candidates"]
    for candidate_type, items in candidates.items():
        for item in items:
            item["interpretation"] = _interpret_item(candidate_type, item, backend)
    return enriched


__all__ = ["interpret_report"]
