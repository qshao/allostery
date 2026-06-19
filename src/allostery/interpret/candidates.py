from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

from allostery.network import AllostericNetwork, betweenness_centrality, shortest_paths


@dataclass
class Community:
    members: list[str]
    member_indices: list[int]
    internal_weight: float


@dataclass
class Pathway:
    nodes: list[str]
    total_weight: float
    hop_count: int


@dataclass
class Hub:
    label: str
    index: int
    centrality: float
    degree: int


@dataclass
class Cluster:
    members: list[str]
    pair_count: int
    mean_score: float


@dataclass
class CandidateSet:
    communities: list[Community]
    pathways: list[Pathway]
    hubs: list[Hub]
    clusters: list[Cluster]


def _total_weight(net: AllostericNetwork) -> float:
    return 0.5 * sum(w for u in range(net.num_nodes) for _, w in net.adjacency.get(u, []))


def _weighted_degree(net: AllostericNetwork) -> dict[int, float]:
    return {u: sum(w for _, w in net.adjacency.get(u, [])) for u in range(net.num_nodes)}


def _modularity(net: AllostericNetwork, labels: dict[int, int], two_m: float,
                degree: dict[int, float]) -> float:
    if two_m == 0:
        return 0.0
    q = 0.0
    for u in range(net.num_nodes):
        neighbors = {v: w for v, w in net.adjacency.get(u, [])}
        for v in range(net.num_nodes):
            if labels[u] != labels[v]:
                continue
            a_uv = neighbors.get(v, 0.0)
            q += a_uv - (degree[u] * degree[v]) / two_m
    return q / two_m


def extract_communities(net: AllostericNetwork) -> list[Community]:
    if net.num_nodes == 0:
        return []
    two_m = 2.0 * _total_weight(net)
    degree = _weighted_degree(net)
    labels = {i: i for i in range(net.num_nodes)}
    best_q = _modularity(net, labels, two_m, degree)

    improved = True
    while improved:
        improved = False
        best_merge: tuple[dict[int, int], float] | None = None
        for u in range(net.num_nodes):
            for v, _ in net.adjacency.get(u, []):
                if labels[u] == labels[v]:
                    continue
                trial = dict(labels)
                source, target = labels[v], labels[u]
                for node, community in trial.items():
                    if community == source:
                        trial[node] = target
                q = _modularity(net, trial, two_m, degree)
                if q > best_q + 1e-12 and (best_merge is None or q > best_merge[1]):
                    best_merge = (trial, q)
        if best_merge is not None:
            labels, best_q = best_merge
            improved = True

    groups: dict[int, list[int]] = defaultdict(list)
    for node, community in labels.items():
        groups[community].append(node)

    communities: list[Community] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        member_set = set(members)
        seen: set[tuple[int, int]] = set()
        internal = 0.0
        for u in members:
            for v, w in net.adjacency.get(u, []):
                if v in member_set:
                    key = (min(u, v), max(u, v))
                    if key not in seen:
                        seen.add(key)
                        internal += w
        communities.append(Community(
            members=[net.node_labels[i] for i in sorted(members)],
            member_indices=sorted(members),
            internal_weight=internal,
        ))
    communities.sort(key=lambda c: c.internal_weight, reverse=True)
    return communities


def extract_hubs(net: AllostericNetwork, top_hubs: int = 10) -> list[Hub]:
    centrality = betweenness_centrality(net)
    ranked = sorted(centrality.items(), key=lambda kv: kv[1], reverse=True)
    hubs: list[Hub] = []
    for index, score in ranked[:top_hubs]:
        hubs.append(Hub(
            label=net.node_labels[index],
            index=index,
            centrality=score,
            degree=len(net.adjacency.get(index, [])),
        ))
    return hubs


def _pair_label(chain: str, number: str, name: str) -> str:
    return f"{chain}:{number} {name}"


def extract_clusters(rows: list[dict[str, str]], max_pairs: int = 30) -> list[Cluster]:
    ranked = sorted(rows, key=lambda r: float(r["score"]), reverse=True)[:max_pairs]
    parent: dict[str, str] = {}

    def find(label: str) -> str:
        parent.setdefault(label, label)
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    pair_scores: list[tuple[str, str, float]] = []
    for r in ranked:
        a = _pair_label(r["residue_i_chain"], r["residue_i_number"], r["residue_i_name"])
        b = _pair_label(r["residue_j_chain"], r["residue_j_number"], r["residue_j_name"])
        union(a, b)
        pair_scores.append((a, b, float(r["score"])))

    members: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    totals: dict[str, float] = defaultdict(float)
    for a, b, score in pair_scores:
        root = find(a)
        members[root].update((a, b))
        counts[root] += 1
        totals[root] += score

    clusters: list[Cluster] = []
    for root, labels in members.items():
        clusters.append(Cluster(
            members=sorted(labels),
            pair_count=counts[root],
            mean_score=totals[root] / counts[root],
        ))
    clusters.sort(key=lambda c: c.mean_score, reverse=True)
    return clusters


def _edge_score(net: AllostericNetwork, u: int, v: int) -> float:
    for neighbor, score in net.adjacency.get(u, []):
        if neighbor == v:
            return score
    return 0.0


def _path_weight(net: AllostericNetwork, labels: list[str]) -> float:
    indices = [net.node_labels.index(label) for label in labels]
    return sum(_edge_score(net, a, b) for a, b in zip(indices, indices[1:]))


def extract_pathways(net: AllostericNetwork, hubs: list[Hub], top_paths: int = 5) -> list[Pathway]:
    seeds = [h.label for h in hubs[: min(4, len(hubs))]]
    found: dict[tuple[str, str], Pathway] = {}
    for source, sink in combinations(seeds, 2):
        paths = shortest_paths(net, source, sink, top_n=1)
        for labels, _distance in paths:
            if len(labels) < 3:
                continue
            key = (labels[0], labels[-1])
            found[key] = Pathway(
                nodes=labels,
                total_weight=_path_weight(net, labels),
                hop_count=len(labels) - 1,
            )
    pathways = sorted(found.values(), key=lambda p: p.total_weight, reverse=True)
    return pathways[:top_paths]


def extract_candidates(
    net: AllostericNetwork,
    rows: list[dict[str, str]],
    *,
    top_paths: int = 5,
    top_hubs: int = 10,
) -> CandidateSet:
    communities = extract_communities(net)
    hubs = extract_hubs(net, top_hubs=top_hubs)
    pathways = extract_pathways(net, hubs, top_paths=top_paths)
    clusters = extract_clusters(rows)
    return CandidateSet(
        communities=communities,
        pathways=pathways,
        hubs=hubs,
        clusters=clusters,
    )


__all__ = [
    "CandidateSet", "Cluster", "Community", "Hub", "Pathway",
    "extract_candidates", "extract_clusters", "extract_communities",
    "extract_hubs", "extract_pathways",
]
