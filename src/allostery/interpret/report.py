from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from allostery.interpret.candidates import CandidateSet
from allostery.interpret.structure import StructuralContext


def _residue_evidence(label: str, context: StructuralContext | None) -> dict[str, Any]:
    if context is None or label not in context.label_to_index:
        return {"label": label}
    features = context.per_residue[context.label_to_index[label]]
    return {"label": label, "rmsf": features.rmsf, "contact_number": features.contact_number}


def build_report(
    candidates: CandidateSet,
    context: StructuralContext | None,
    *,
    source: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    def geom(labels: list[str]) -> dict[str, Any]:
        return context.geometry(labels) if context is not None else {}

    communities = [
        {
            "type": "community",
            "members": c.members,
            "internal_weight": c.internal_weight,
            "evidence": {
                "size": len(c.members),
                "geometry": geom(c.members),
                "residues": [_residue_evidence(m, context) for m in c.members],
            },
        }
        for c in candidates.communities
    ]
    pathways = [
        {
            "type": "pathway",
            "nodes": p.nodes,
            "total_weight": p.total_weight,
            "hop_count": p.hop_count,
            "evidence": {
                "geometry": geom(p.nodes),
                "residues": [_residue_evidence(m, context) for m in p.nodes],
            },
        }
        for p in candidates.pathways
    ]
    hubs = [
        {
            "type": "hub",
            "label": h.label,
            "centrality": h.centrality,
            "degree": h.degree,
            "evidence": _residue_evidence(h.label, context),
        }
        for h in candidates.hubs
    ]
    clusters = [
        {
            "type": "cluster",
            "members": c.members,
            "pair_count": c.pair_count,
            "mean_score": c.mean_score,
            "evidence": {
                "geometry": geom(c.members),
                "residues": [_residue_evidence(m, context) for m in c.members],
            },
        }
        for c in candidates.clusters
    ]
    return {
        "schema_version": 1,
        "source": source,
        "parameters": parameters,
        "structural_context": context is not None,
        "candidates": {
            "communities": communities,
            "pathways": pathways,
            "hubs": hubs,
            "clusters": clusters,
        },
    }


def _render_interpretation(item: dict[str, Any]) -> list[str]:
    interp = item.get("interpretation")
    if not interp:
        return []
    if interp.get("invalid"):
        return ["  - **Interpretation**: LLM response was invalid; see `interpretation.raw` in JSON."]
    lines = [f"  - **Interpretation** (confidence: {interp.get('confidence', 'n/a')}, "
             f"parametric: {interp.get('parametric', 'n/a')}): {interp.get('summary', '')}"]
    if interp.get("mechanism_hypothesis"):
        lines.append(f"    - Mechanism: {interp['mechanism_hypothesis']}")
    if interp.get("caveats"):
        lines.append(f"    - Caveats: {interp['caveats']}")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Allostery Interpretation Report", "", f"Source: `{report['source']}`", ""]
    cand = report["candidates"]

    lines.append("## Communities / Modules")
    for i, item in enumerate(cand["communities"], 1):
        lines.append(f"- **Module {i}** ({len(item['members'])} residues, "
                     f"internal weight {item['internal_weight']:.3f}): "
                     f"{', '.join(item['members'])}")
        lines.extend(_render_interpretation(item))
    lines.append("")

    lines.append("## Candidate Pathways")
    for i, item in enumerate(cand["pathways"], 1):
        lines.append(f"- **Pathway {i}** ({item['hop_count']} hops, "
                     f"weight {item['total_weight']:.3f}): {' -> '.join(item['nodes'])}")
        lines.extend(_render_interpretation(item))
    lines.append("")

    lines.append("## Hub / Bottleneck Residues")
    for i, item in enumerate(cand["hubs"], 1):
        lines.append(f"- **{i}. {item['label']}** "
                     f"(centrality {item['centrality']:.4f}, degree {item['degree']})")
        lines.extend(_render_interpretation(item))
    lines.append("")

    lines.append("## Coupled-Pair Clusters")
    for i, item in enumerate(cand["clusters"], 1):
        lines.append(f"- **Cluster {i}** ({item['pair_count']} pairs, "
                     f"mean score {item['mean_score']:.3f}): {', '.join(item['members'])}")
        lines.extend(_render_interpretation(item))
    lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], json_path: str | Path, md_path: str | Path) -> None:
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")


__all__ = ["build_report", "render_markdown", "write_report"]
