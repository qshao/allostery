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
    out_path: str | Path | None = None,
) -> str:
    """Read a scores CSV, build the allosteric network, and return a text report."""
    rows = read_scores_csv(scores_csv)
    net = build_graph(rows, top_k=top_k)
    if net.num_nodes == 0:
        raise ValueError(
            "No edges in the network after top-k filtering; increase --top-k "
            "or check the scores CSV."
        )
    report = format_report(
        net,
        source_label=source,
        sink_label=sink,
        top_hubs=top_hubs,
        top_paths=top_paths,
    )
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
    return report


__all__ = ["run_network_analysis"]
