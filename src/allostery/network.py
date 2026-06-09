from __future__ import annotations

import csv
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class AllostericNetwork:
    """Weighted undirected graph of residue-residue allosteric interactions."""

    node_labels: list[str]  # index → "CHAIN:NUM NAME"
    adjacency: dict[int, list[tuple[int, float]]] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return len(self.node_labels)

    @property
    def num_edges(self) -> int:
        return sum(len(neighbors) for neighbors in self.adjacency.values()) // 2

    def node_index(self, label: str) -> int:
        try:
            return self.node_labels.index(label)
        except ValueError:
            raise ValueError(f"Residue {label!r} not found in network. "
                             f"Available: {', '.join(self.node_labels)}")


def _residue_label(chain: str, number: str, name: str) -> str:
    return f"{chain}:{number} {name}"


def read_scores_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a pair scores CSV into a list of row dicts."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {
        "score", "residue_i_chain", "residue_i_number", "residue_i_name",
        "residue_j_chain", "residue_j_number", "residue_j_name",
    }
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"Scores CSV is missing required columns: {sorted(missing)}")
    return rows


def build_graph(rows: list[dict[str, str]], top_k: int = 20) -> AllostericNetwork:
    """Build an undirected weighted graph from the top-k scored pairs."""
    # Sort by score descending and take top_k
    sorted_rows = sorted(rows, key=lambda r: float(r["score"]), reverse=True)
    selected = sorted_rows[:top_k]

    # Collect unique node labels preserving order of appearance
    node_set: dict[str, int] = {}
    for row in selected:
        for prefix in ("i", "j"):
            label = _residue_label(row[f"residue_{prefix}_chain"],
                                   row[f"residue_{prefix}_number"],
                                   row[f"residue_{prefix}_name"])
            if label not in node_set:
                node_set[label] = len(node_set)

    node_labels = list(node_set.keys())
    adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)

    for row in selected:
        label_i = _residue_label(row["residue_i_chain"], row["residue_i_number"],
                                  row["residue_i_name"])
        label_j = _residue_label(row["residue_j_chain"], row["residue_j_number"],
                                  row["residue_j_name"])
        idx_i = node_set[label_i]
        idx_j = node_set[label_j]
        score = float(row["score"])
        adjacency[idx_i].append((idx_j, score))
        adjacency[idx_j].append((idx_i, score))

    net = AllostericNetwork(node_labels=node_labels, adjacency=dict(adjacency))
    return net


def connected_components(net: AllostericNetwork) -> list[list[int]]:
    """Return connected components as lists of node indices."""
    visited: set[int] = set()
    components: list[list[int]] = []
    for start in range(net.num_nodes):
        if start in visited:
            continue
        component: list[int] = []
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbor, _ in net.adjacency.get(node, []):
                if neighbor not in visited:
                    stack.append(neighbor)
        components.append(component)
    return components


def dijkstra(net: AllostericNetwork, source: int) -> tuple[dict[int, float], dict[int, int | None]]:
    """Dijkstra shortest paths from source. Distance = sum of 1/score along path."""
    dist: dict[int, float] = {source: 0.0}
    prev: dict[int, int | None] = {source: None}
    heap: list[tuple[float, int]] = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        for v, score in net.adjacency.get(u, []):
            edge_dist = 1.0 / score if score > 0 else float("inf")
            nd = d + edge_dist
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    return dist, prev


def _reconstruct_path(prev: dict[int, int | None], target: int) -> list[int] | None:
    path: list[int] = []
    current: int | None = target
    while current is not None:
        path.append(current)
        current = prev.get(current)
    path.reverse()
    if not path or path[0] != _find_source(prev):
        return None
    return path


def _find_source(prev: dict[int, int | None]) -> int:
    for node, parent in prev.items():
        if parent is None:
            return node
    return next(iter(prev))


def shortest_paths(
    net: AllostericNetwork,
    source_label: str,
    sink_label: str,
    top_n: int = 5,
) -> list[tuple[list[str], float]]:
    """
    Find up to top_n shortest paths from source to sink using Yen's k-shortest paths.
    Returns list of (path_as_labels, total_distance).
    """
    src = net.node_index(source_label)
    snk = net.node_index(sink_label)

    # Yen's algorithm
    best: list[tuple[float, list[int]]] = []
    candidates: list[tuple[float, list[int]]] = []

    dist, prev = dijkstra(net, src)
    if snk not in dist:
        return []
    first_path = _reconstruct_path_between(prev, src, snk)
    if first_path is None:
        return []
    best.append((dist[snk], first_path))

    for _ in range(1, top_n):
        last_path = best[-1][1]
        for i in range(len(last_path) - 1):
            spur_node = last_path[i]
            root_path = last_path[: i + 1]
            root_dist = sum(
                1.0 / _edge_score(net, root_path[k], root_path[k + 1])
                for k in range(len(root_path) - 1)
            )

            # Temporarily remove edges used by existing best paths with same root
            removed: list[tuple[int, int, float]] = []
            for _, existing in best:
                if existing[: i + 1] == root_path and i + 1 < len(existing):
                    u, v = existing[i], existing[i + 1]
                    removed.extend(_remove_edge(net, u, v))

            # Remove root nodes (except spur) to avoid cycles
            removed_nodes: dict[int, list[tuple[int, float]]] = {}
            for rn in root_path[:-1]:
                if rn in net.adjacency:
                    removed_nodes[rn] = net.adjacency.pop(rn)

            spur_dist, spur_prev = dijkstra(net, spur_node)

            # Restore
            for rn, neighbors in removed_nodes.items():
                net.adjacency[rn] = neighbors
            for u, v, score in removed:
                _restore_edge(net, u, v, score)

            if snk in spur_dist:
                spur_path = _reconstruct_path_between(spur_prev, spur_node, snk)
                if spur_path is not None:
                    full_path = root_path[:-1] + spur_path
                    total_dist = root_dist + spur_dist[snk]
                    if not any(fp == full_path for _, fp in candidates + best):
                        heapq.heappush(candidates, (total_dist, full_path))

        if not candidates:
            break
        d, p = heapq.heappop(candidates)
        best.append((d, p))

    return [
        ([net.node_labels[n] for n in path], dist_val)
        for dist_val, path in best
    ]


def _edge_score(net: AllostericNetwork, u: int, v: int) -> float:
    for neighbor, score in net.adjacency.get(u, []):
        if neighbor == v:
            return score
    return 1.0


def _remove_edge(net: AllostericNetwork, u: int, v: int) -> list[tuple[int, int, float]]:
    removed: list[tuple[int, int, float]] = []
    if u in net.adjacency:
        before = net.adjacency[u]
        new_u = [(n, s) for n, s in before if n != v]
        if len(new_u) < len(before):
            score = next(s for n, s in before if n == v)
            removed.append((u, v, score))
            net.adjacency[u] = new_u
    if v in net.adjacency:
        before = net.adjacency[v]
        new_v = [(n, s) for n, s in before if n != u]
        if len(new_v) < len(before):
            net.adjacency[v] = new_v
    return removed


def _restore_edge(net: AllostericNetwork, u: int, v: int, score: float) -> None:
    net.adjacency.setdefault(u, []).append((v, score))
    net.adjacency.setdefault(v, []).append((u, score))


def _reconstruct_path_between(
    prev: dict[int, int | None], source: int, target: int
) -> list[int] | None:
    path: list[int] = []
    current: int | None = target
    while current is not None:
        path.append(current)
        if current == source:
            break
        current = prev.get(current)
    else:
        return None
    path.reverse()
    if not path or path[0] != source:
        return None
    return path


def betweenness_centrality(net: AllostericNetwork) -> dict[int, float]:
    """Brandes algorithm for betweenness centrality (normalized)."""
    n = net.num_nodes
    centrality: dict[int, float] = {i: 0.0 for i in range(n)}

    for s in range(n):
        stack: list[int] = []
        pred: dict[int, list[int]] = {i: [] for i in range(n)}
        sigma: dict[int, float] = {i: 0.0 for i in range(n)}
        sigma[s] = 1.0
        dist_map: dict[int, float] = {i: float("inf") for i in range(n)}
        dist_map[s] = 0.0
        heap: list[tuple[float, int]] = [(0.0, s)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist_map[u]:
                continue
            stack.append(u)
            for v, score in net.adjacency.get(u, []):
                edge_dist = 1.0 / score if score > 0 else float("inf")
                nd = d + edge_dist
                if nd < dist_map[v]:
                    dist_map[v] = nd
                    pred[v] = [u]
                    sigma[v] = sigma[u]
                    heapq.heappush(heap, (nd, v))
                elif nd == dist_map[v]:
                    pred[v].append(u)
                    sigma[v] += sigma[u]

        delta: dict[int, float] = {i: 0.0 for i in range(n)}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                centrality[w] += delta[w]

    # Normalize by (n-1)(n-2) for undirected graph
    if n > 2:
        norm = (n - 1) * (n - 2)
        for node in centrality:
            centrality[node] /= norm

    return centrality


def network_summary(net: AllostericNetwork) -> str:
    components = connected_components(net)
    lines = [
        "=== Allosteric Network ===",
        f"Residues (nodes):       {net.num_nodes}",
        f"Edges (scored pairs):   {net.num_edges}",
        f"Connected components:   {len(components)}",
    ]
    return "\n".join(lines)


def hub_summary(net: AllostericNetwork, top_n: int = 10) -> str:
    centrality = betweenness_centrality(net)
    ranked = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    lines = ["", f"=== Hub Residues (Top {top_n} by Betweenness Centrality) ==="]
    for rank, (idx, score) in enumerate(ranked, 1):
        lines.append(f"  {rank:>2}.  {net.node_labels[idx]:<20}  {score:.4f}")
    return "\n".join(lines)


def channel_summary(
    net: AllostericNetwork,
    source_label: str,
    sink_label: str,
    top_n: int = 5,
) -> str:
    paths = shortest_paths(net, source_label, sink_label, top_n=top_n)
    lines = ["", f"=== Allosteric Channel: {source_label} → {sink_label} ==="]
    if not paths:
        lines.append("  No path found between these residues in the current network.")
        lines.append("  Try increasing --top-k to include more edges.")
        return "\n".join(lines)
    for rank, (path, dist) in enumerate(paths, 1):
        path_str = " → ".join(path)
        lines.append(f"  Path {rank} (hops {len(path) - 1}):  {path_str}   (dist {dist:.3f})")
    return "\n".join(lines)


def format_report(
    net: AllostericNetwork,
    source_label: str | None = None,
    sink_label: str | None = None,
    top_hubs: int = 10,
    top_paths: int = 5,
) -> str:
    parts = [network_summary(net), hub_summary(net, top_n=top_hubs)]
    if source_label is not None and sink_label is not None:
        parts.append(channel_summary(net, source_label, sink_label, top_n=top_paths))
    return "\n".join(parts)


__all__ = [
    "AllostericNetwork",
    "build_graph",
    "betweenness_centrality",
    "channel_summary",
    "connected_components",
    "dijkstra",
    "format_report",
    "hub_summary",
    "network_summary",
    "read_scores_csv",
    "shortest_paths",
]
