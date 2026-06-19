from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = (
    "You are a structural-biology assistant interpreting the output of a deep-learning "
    "allostery model. You are given a candidate allosteric structure with topological and "
    "structural evidence computed from the protein. Ground every statement in the supplied "
    "evidence. If you assert a functional role from prior knowledge that is not present in the "
    "evidence, set \"parametric\" to true and lower the confidence. Never invent residues or "
    "numbers that are not in the evidence. Respond only with JSON matching the requested schema."
)

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "mechanism_hypothesis": {"type": "string"},
        "key_residues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string"},
                    "role": {"type": "string"},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["label", "role", "evidence_refs"],
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "parametric": {"type": "boolean"},
        "caveats": {"type": "string"},
    },
    "required": [
        "summary", "mechanism_hypothesis", "key_residues",
        "confidence", "parametric", "caveats",
    ],
}


def build_user_prompt(candidate_type: str, item: dict[str, Any]) -> str:
    return (
        f"Candidate type: {candidate_type}\n"
        f"Evidence (JSON):\n{json.dumps(item, indent=2)}\n\n"
        "Interpret this candidate's likely allosteric role. Return JSON only."
    )


__all__ = ["RESPONSE_SCHEMA", "SYSTEM_PROMPT", "build_user_prompt"]
