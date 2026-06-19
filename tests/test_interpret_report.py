from __future__ import annotations

import json
from pathlib import Path

from allostery.network import AllostericNetwork
from allostery.interpret.candidates import extract_candidates
from allostery.interpret.report import build_report, render_markdown, write_report


def _net_and_rows():
    labels = [f"A:{i} GLY" for i in range(4)]
    adj = {i: [] for i in range(4)}
    for u in range(3):
        adj[u].append((u + 1, 1.0))
        adj[u + 1].append((u, 1.0))
    net = AllostericNetwork(node_labels=labels, adjacency=adj)
    rows = [
        {"residue_i_chain": "A", "residue_i_number": str(u), "residue_i_name": "GLY",
         "residue_j_chain": "A", "residue_j_number": str(u + 1), "residue_j_name": "GLY",
         "score": "1.0"}
        for u in range(3)
    ]
    return net, rows


def test_build_report_shape() -> None:
    net, rows = _net_and_rows()
    candidates = extract_candidates(net, rows, top_paths=3, top_hubs=3)
    report = build_report(candidates, None, source="scores.csv", parameters={"top_k": 20})
    assert report["schema_version"] == 1
    assert report["structural_context"] is False
    for key in ("communities", "pathways", "hubs", "clusters"):
        assert key in report["candidates"]
    for hub in report["candidates"]["hubs"]:
        assert "evidence" in hub
        assert "interpretation" not in hub


def test_write_report_emits_json_and_markdown(tmp_path: Path) -> None:
    net, rows = _net_and_rows()
    candidates = extract_candidates(net, rows, top_paths=3, top_hubs=3)
    report = build_report(candidates, None, source="scores.csv", parameters={})
    json_path = tmp_path / "out.json"
    md_path = tmp_path / "out.md"
    write_report(report, json_path, md_path)
    loaded = json.loads(json_path.read_text())
    assert loaded["schema_version"] == 1
    text = md_path.read_text()
    assert "# Allostery Interpretation Report" in text
    assert "## Hub / Bottleneck Residues" in text
