from __future__ import annotations

from allostery.network import AllostericNetwork
from allostery.interpret.candidates import (
    CandidateSet, Cluster, Community, Hub, Pathway,
    extract_candidates, extract_clusters, extract_communities,
    extract_hubs, extract_pathways,
)


def _two_triangles() -> AllostericNetwork:
    labels = [f"A:{i} GLY" for i in range(6)]
    adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(6)}

    def link(u: int, v: int, w: float) -> None:
        adj[u].append((v, w))
        adj[v].append((u, w))

    for u, v in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        link(u, v, 1.0)
    link(2, 3, 0.05)
    return AllostericNetwork(node_labels=labels, adjacency=adj)


def _path_graph() -> AllostericNetwork:
    labels = [f"A:{i} GLY" for i in range(5)]
    adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(5)}
    for u in range(4):
        adj[u].append((u + 1, 1.0))
        adj[u + 1].append((u, 1.0))
    return AllostericNetwork(node_labels=labels, adjacency=adj)


def test_extract_communities_finds_two_modules() -> None:
    net = _two_triangles()
    communities = extract_communities(net)
    assert all(isinstance(c, Community) for c in communities)
    member_sets = sorted((sorted(c.member_indices) for c in communities))
    assert member_sets == [[0, 1, 2], [3, 4, 5]]


def test_extract_hubs_ranks_bottleneck_first() -> None:
    hubs = extract_hubs(_path_graph(), top_hubs=3)
    assert all(isinstance(h, Hub) for h in hubs)
    assert hubs[0].index == 2
    assert hubs[0].centrality >= hubs[-1].centrality


def test_extract_clusters_groups_shared_residues() -> None:
    def row(ci, ni, cj, nj, score):
        return {
            "residue_i_chain": ci, "residue_i_number": str(ni), "residue_i_name": "GLY",
            "residue_j_chain": cj, "residue_j_number": str(nj), "residue_j_name": "GLY",
            "score": str(score),
        }
    rows = [row("A", 1, "A", 2, 0.9), row("A", 2, "A", 3, 0.8), row("B", 9, "B", 10, 0.7)]
    clusters = extract_clusters(rows, max_pairs=30)
    sizes = sorted(len(c.members) for c in clusters)
    assert sizes == [2, 3]
    assert all(isinstance(c, Cluster) for c in clusters)


def test_extract_pathways_finds_multi_hop_route() -> None:
    net = _path_graph()
    hubs = extract_hubs(net, top_hubs=5)
    pathways = extract_pathways(net, hubs, top_paths=5)
    assert all(isinstance(p, Pathway) for p in pathways)
    longest = max(pathways, key=lambda p: p.hop_count)
    assert longest.hop_count >= 2
    assert longest.nodes[0] != longest.nodes[-1]


def test_extract_candidates_returns_full_set() -> None:
    def row(ci, ni, cj, nj, score):
        return {
            "residue_i_chain": ci, "residue_i_number": str(ni), "residue_i_name": "GLY",
            "residue_j_chain": cj, "residue_j_number": str(nj), "residue_j_name": "GLY",
            "score": str(score),
        }
    rows = [row("A", 0, "A", 1, 0.9), row("A", 1, "A", 2, 0.8), row("A", 2, "A", 3, 0.7)]
    net = _path_graph()
    result = extract_candidates(net, rows, top_paths=3, top_hubs=3)
    assert isinstance(result, CandidateSet)
    assert result.hubs and result.clusters
    assert isinstance(result.communities, list)
    assert isinstance(result.pathways, list)
