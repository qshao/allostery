from __future__ import annotations

from allostery.interpret.engine import interpret_report


class _FakeBackend:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate_json(self, system: str, user: str, schema: dict) -> dict:
        self.calls += 1
        return self._responses.pop(0)


def _report() -> dict:
    return {
        "schema_version": 1,
        "source": "scores.csv",
        "parameters": {},
        "structural_context": False,
        "candidates": {
            "communities": [],
            "pathways": [],
            "hubs": [{"type": "hub", "label": "A:5 GLY", "centrality": 0.5,
                      "degree": 3, "evidence": {"label": "A:5 GLY"}}],
            "clusters": [],
        },
    }


def _valid() -> dict:
    return {
        "summary": "key control point", "mechanism_hypothesis": "couples sites",
        "key_residues": [{"label": "A:5 GLY", "role": "hub", "evidence_refs": ["centrality"]}],
        "confidence": "medium", "parametric": True, "caveats": "no external validation",
    }


def test_interpret_report_merges_interpretation() -> None:
    backend = _FakeBackend([_valid()])
    enriched = interpret_report(_report(), backend)
    hub = enriched["candidates"]["hubs"][0]
    assert hub["interpretation"]["confidence"] == "medium"
    assert backend.calls == 1


def test_interpret_report_retries_then_falls_back_on_invalid_json() -> None:
    backend = _FakeBackend([{"bad": "shape"}, {"still": "bad"}])
    enriched = interpret_report(_report(), backend)
    hub = enriched["candidates"]["hubs"][0]
    assert hub["interpretation"]["invalid"] is True
    assert "raw" in hub["interpretation"]
    assert backend.calls == 2
