from __future__ import annotations

from pathlib import Path
from typing import Any

from allostery.interpret.candidates import extract_candidates
from allostery.interpret.engine import interpret_report
from allostery.interpret.llm import LLMBackend, make_backend
from allostery.interpret.report import build_report, write_report
from allostery.interpret.structure import compute_structural_context
from allostery.io.trajectory import load_trajectory
from allostery.network import build_graph, read_scores_csv


def run_interpretation(
    scores_csv: str | Path,
    *,
    out_json: str | Path,
    out_md: str | Path,
    pdb_path: str | Path | None = None,
    topology_path: str | Path | None = None,
    top_k: int = 20,
    top_paths: int = 5,
    top_hubs: int = 10,
    llm: str = "none",
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    backend: LLMBackend | None = None,
) -> dict[str, Any]:
    rows = read_scores_csv(scores_csv)
    net = build_graph(rows, top_k=top_k)
    candidates = extract_candidates(net, rows, top_paths=top_paths, top_hubs=top_hubs)

    context = None
    if pdb_path is not None:
        trajectory = load_trajectory(Path(pdb_path), topology_path=topology_path)
        context = compute_structural_context(trajectory)

    parameters = {"top_k": top_k, "top_paths": top_paths, "top_hubs": top_hubs}
    report = build_report(candidates, context, source=str(scores_csv), parameters=parameters)
    write_report(report, out_json, out_md)

    if llm != "none":
        if backend is None:
            backend = make_backend(llm, model=llm_model, base_url=llm_base_url)
        report = interpret_report(report, backend)
        write_report(report, out_json, out_md)

    return report


__all__ = ["run_interpretation"]
