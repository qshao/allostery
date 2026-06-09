from __future__ import annotations

import csv
from pathlib import Path

import pytest

from allostery.network import (
    AllostericNetwork,
    betweenness_centrality,
    build_graph,
    channel_summary,
    connected_components,
    dijkstra,
    format_report,
    hub_summary,
    network_summary,
    read_scores_csv,
    shortest_paths,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_scores_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "rank", "score",
        "residue_i_index", "residue_i_chain", "residue_i_number", "residue_i_name",
        "residue_j_index", "residue_j_chain", "residue_j_number", "residue_j_name",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            writer.writerow({"rank": i, **row})


def _simple_rows() -> list[dict[str, str]]:
    """Three residues A:1, A:2, A:3 with two edges: (1,2)=0.9, (2,3)=0.8."""
    return [
        {
            "score": "0.9",
            "residue_i_chain": "A", "residue_i_number": "1", "residue_i_name": "GLY",
            "residue_j_chain": "A", "residue_j_number": "2", "residue_j_name": "ALA",
        },
        {
            "score": "0.8",
            "residue_i_chain": "A", "residue_i_number": "2", "residue_i_name": "ALA",
            "residue_j_chain": "A", "residue_j_number": "3", "residue_j_name": "SER",
        },
    ]


# ---------------------------------------------------------------------------
# read_scores_csv
# ---------------------------------------------------------------------------

def test_read_scores_csv_returns_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "scores.csv"
    _write_scores_csv(csv_path, _simple_rows())
    rows = read_scores_csv(csv_path)
    assert len(rows) == 2
    assert rows[0]["score"] == "0.9"


def test_read_scores_csv_missing_required_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rank", "score"])
        writer.writeheader()
        writer.writerow({"rank": 1, "score": "0.5"})
    with pytest.raises(ValueError, match="missing required columns"):
        read_scores_csv(csv_path)


def test_read_scores_csv_empty_file_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    fieldnames = ["rank", "score", "residue_i_chain", "residue_i_number", "residue_i_name",
                  "residue_j_chain", "residue_j_number", "residue_j_name", "residue_i_index", "residue_j_index"]
    with open(csv_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fieldnames).writeheader()
    with pytest.raises(ValueError, match="No rows"):
        read_scores_csv(csv_path)


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------

def test_build_graph_node_count(tmp_path: Path) -> None:
    rows = _simple_rows()
    net = build_graph(rows, top_k=2)
    assert net.num_nodes == 3  # A:1, A:2, A:3


def test_build_graph_edge_count(tmp_path: Path) -> None:
    rows = _simple_rows()
    net = build_graph(rows, top_k=2)
    assert net.num_edges == 2


def test_build_graph_top_k_limits_edges() -> None:
    rows = _simple_rows()
    net = build_graph(rows, top_k=1)  # only top-1
    assert net.num_edges == 1
    assert net.num_nodes == 2  # only the two residues from the top edge


def test_build_graph_node_labels_format() -> None:
    rows = _simple_rows()
    net = build_graph(rows, top_k=2)
    assert "A:1 GLY" in net.node_labels
    assert "A:2 ALA" in net.node_labels
    assert "A:3 SER" in net.node_labels


# ---------------------------------------------------------------------------
# connected_components
# ---------------------------------------------------------------------------

def test_connected_components_single_component() -> None:
    rows = _simple_rows()
    net = build_graph(rows, top_k=2)
    comps = connected_components(net)
    assert len(comps) == 1
    assert len(comps[0]) == 3


def test_connected_components_two_components() -> None:
    # Two disconnected edges: (A:1,A:2) and (B:1,B:2)
    rows = [
        {
            "score": "0.9",
            "residue_i_chain": "A", "residue_i_number": "1", "residue_i_name": "GLY",
            "residue_j_chain": "A", "residue_j_number": "2", "residue_j_name": "ALA",
        },
        {
            "score": "0.8",
            "residue_i_chain": "B", "residue_i_number": "1", "residue_i_name": "VAL",
            "residue_j_chain": "B", "residue_j_number": "2", "residue_j_name": "LEU",
        },
    ]
    net = build_graph(rows, top_k=2)
    comps = connected_components(net)
    assert len(comps) == 2


# ---------------------------------------------------------------------------
# dijkstra
# ---------------------------------------------------------------------------

def test_dijkstra_reaches_all_nodes() -> None:
    rows = _simple_rows()
    net = build_graph(rows, top_k=2)
    src = net.node_index("A:1 GLY")
    dist, prev = dijkstra(net, src)
    assert len(dist) == 3
    assert dist[src] == 0.0


def test_dijkstra_shorter_path_preferred() -> None:
    rows = _simple_rows()  # (1,2)=0.9, (2,3)=0.8
    net = build_graph(rows, top_k=2)
    src = net.node_index("A:1 GLY")
    snk = net.node_index("A:3 SER")
    dist, _ = dijkstra(net, src)
    # distance 1→2 = 1/0.9 ≈ 1.111, distance 2→3 = 1/0.8 = 1.25, total ≈ 2.361
    assert dist[snk] == pytest.approx(1 / 0.9 + 1 / 0.8, rel=1e-5)


# ---------------------------------------------------------------------------
# shortest_paths
# ---------------------------------------------------------------------------

def test_shortest_paths_finds_direct_connection() -> None:
    rows = _simple_rows()
    net = build_graph(rows, top_k=2)
    paths = shortest_paths(net, "A:1 GLY", "A:2 ALA", top_n=1)
    assert len(paths) == 1
    path_labels, _ = paths[0]
    assert path_labels == ["A:1 GLY", "A:2 ALA"]


def test_shortest_paths_finds_indirect_connection() -> None:
    rows = _simple_rows()
    net = build_graph(rows, top_k=2)
    paths = shortest_paths(net, "A:1 GLY", "A:3 SER", top_n=1)
    assert len(paths) == 1
    path_labels, _ = paths[0]
    assert path_labels == ["A:1 GLY", "A:2 ALA", "A:3 SER"]


def test_shortest_paths_no_path_returns_empty() -> None:
    rows = [
        {
            "score": "0.9",
            "residue_i_chain": "A", "residue_i_number": "1", "residue_i_name": "GLY",
            "residue_j_chain": "A", "residue_j_number": "2", "residue_j_name": "ALA",
        },
        {
            "score": "0.8",
            "residue_i_chain": "B", "residue_i_number": "1", "residue_i_name": "VAL",
            "residue_j_chain": "B", "residue_j_number": "2", "residue_j_name": "LEU",
        },
    ]
    net = build_graph(rows, top_k=2)
    paths = shortest_paths(net, "A:1 GLY", "B:1 VAL", top_n=1)
    assert paths == []


def test_shortest_paths_unknown_residue_raises() -> None:
    rows = _simple_rows()
    net = build_graph(rows, top_k=2)
    with pytest.raises(ValueError, match="not found"):
        shortest_paths(net, "A:1 GLY", "Z:99 UNK", top_n=1)


# ---------------------------------------------------------------------------
# betweenness_centrality
# ---------------------------------------------------------------------------

def test_betweenness_centrality_middle_node_highest() -> None:
    rows = _simple_rows()  # A:2 is the bridge between A:1 and A:3
    net = build_graph(rows, top_k=2)
    centrality = betweenness_centrality(net)
    idx_a2 = net.node_index("A:2 ALA")
    idx_a1 = net.node_index("A:1 GLY")
    idx_a3 = net.node_index("A:3 SER")
    assert centrality[idx_a2] >= centrality[idx_a1]
    assert centrality[idx_a2] >= centrality[idx_a3]


def test_betweenness_centrality_isolated_node_is_zero() -> None:
    # Single edge — endpoints have centrality 0 (no paths pass through them)
    rows = [
        {
            "score": "0.9",
            "residue_i_chain": "A", "residue_i_number": "1", "residue_i_name": "GLY",
            "residue_j_chain": "A", "residue_j_number": "2", "residue_j_name": "ALA",
        }
    ]
    net = build_graph(rows, top_k=1)
    centrality = betweenness_centrality(net)
    assert all(v == pytest.approx(0.0) for v in centrality.values())


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------

def test_format_report_contains_network_header() -> None:
    rows = _simple_rows()
    net = build_graph(rows, top_k=2)
    report = format_report(net)
    assert "Allosteric Network" in report
    assert "Residues" in report
    assert "Edges" in report


def test_format_report_contains_hub_section() -> None:
    rows = _simple_rows()
    net = build_graph(rows, top_k=2)
    report = format_report(net, top_hubs=3)
    assert "Hub Residues" in report


def test_format_report_with_channel_section() -> None:
    rows = _simple_rows()
    net = build_graph(rows, top_k=2)
    report = format_report(net, source_label="A:1 GLY", sink_label="A:3 SER", top_paths=1)
    assert "Allosteric Channel" in report
    assert "A:1 GLY" in report
    assert "A:3 SER" in report


def test_format_report_no_channel_when_not_requested() -> None:
    rows = _simple_rows()
    net = build_graph(rows, top_k=2)
    report = format_report(net)
    assert "Allosteric Channel" not in report


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def test_cli_analyze_prints_report(tmp_path: Path) -> None:
    from allostery.cli import main
    csv_path = tmp_path / "scores.csv"
    _write_scores_csv(csv_path, _simple_rows())
    ret = main(["analyze", str(csv_path), "--top-k", "2"])
    assert ret == 0


def test_cli_analyze_with_source_sink(tmp_path: Path) -> None:
    from allostery.cli import main
    csv_path = tmp_path / "scores.csv"
    _write_scores_csv(csv_path, _simple_rows())
    ret = main(["analyze", str(csv_path), "--top-k", "2", "--source", "A:1 GLY", "--sink", "A:3 SER"])
    assert ret == 0
