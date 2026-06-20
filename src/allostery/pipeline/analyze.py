from __future__ import annotations

from pathlib import Path

from allostery.network import (
    betweenness_centrality,
    build_graph,
    detect_threshold,
    format_report,
    format_score_histogram,
    read_scores_csv,
    shortest_paths,
)
from allostery.pipeline.pymol_export import write_pymol_script


def run_network_analysis(
    scores_csv: str | Path,
    top_k: int = 20,
    source: str | None = None,
    sink: str | None = None,
    top_paths: int = 5,
    top_hubs: int = 10,
    out_path: str | Path | None = None,
    out_pml: Path | None = None,
    pdb_path: Path | None = None,
) -> str:
    """Read a scores CSV, build the allosteric network, and return a text report."""
    rows = read_scores_csv(scores_csv)
    all_scores = [float(r["score"]) for r in rows]
    top_k_scores = all_scores[:top_k]
    if not top_k_scores:
        raise ValueError(
            "No top-k scores available; increase --top-k or check the scores CSV."
        )
    threshold_score, threshold_rank = detect_threshold(top_k_scores)

    net = build_graph(rows, top_k=top_k)
    if net.num_nodes == 0:
        raise ValueError(
            "No edges in the network after top-k filtering; increase --top-k "
            "or check the scores CSV."
        )

    threshold_line = (
        f"Suggested threshold: {threshold_score:.4f}"
        f" (top {threshold_rank} of {len(top_k_scores)} scored pairs"
        f" — largest gap at rank {threshold_rank})"
    )
    body = format_report(
        net,
        source_label=source,
        sink_label=sink,
        top_hubs=top_hubs,
        top_paths=top_paths,
    )
    histogram = format_score_histogram(all_scores, bins=10, threshold_rank=threshold_rank)
    report = f"{threshold_line}\n\n{body}\n\n{histogram}"

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")

    if out_pml is not None and pdb_path is not None:
        centrality = betweenness_centrality(net)
        sorted_rows = sorted(rows, key=lambda r: float(r["score"]), reverse=True)[:top_k]
        top_pairs = [
            (
                f"{r['residue_i_chain']}:{r['residue_i_number']} {r['residue_i_name']}",
                f"{r['residue_j_chain']}:{r['residue_j_number']} {r['residue_j_name']}",
                float(r["score"]),
            )
            for r in sorted_rows
        ]
        path_edges = None
        if source is not None and sink is not None:
            paths = shortest_paths(net, source, sink, top_n=1)
            if paths:
                path_nodes, _ = paths[0]
                path_edges = list(zip(path_nodes, path_nodes[1:]))
        out_pml.parent.mkdir(parents=True, exist_ok=True)
        write_pymol_script(
            pml_path=out_pml,
            pdb_path=pdb_path,
            node_labels=net.node_labels,
            centrality=centrality,
            top_pairs=top_pairs,
            path_edges=path_edges,
        )

    return report


__all__ = ["run_network_analysis"]
