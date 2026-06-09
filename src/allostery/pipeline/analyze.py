from __future__ import annotations

from pathlib import Path

from allostery.network import build_graph, format_report, read_scores_csv


def run_network_analysis(
    scores_csv: str | Path,
    top_k: int = 20,
    source: str | None = None,
    sink: str | None = None,
    top_paths: int = 5,
    top_hubs: int = 10,
) -> str:
    """Read a scores CSV, build the allosteric network, and return a text report."""
    rows = read_scores_csv(scores_csv)
    net = build_graph(rows, top_k=top_k)
    return format_report(
        net,
        source_label=source,
        sink_label=sink,
        top_hubs=top_hubs,
        top_paths=top_paths,
    )


__all__ = ["run_network_analysis"]
