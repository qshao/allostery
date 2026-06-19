# Allostery Interpretation Subsystem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a residue-pair scores CSV into candidate allosteric networks plus an optional LLM-composed biological interpretation, emitted as a deterministic JSON + markdown report.

**Architecture:** A new `src/allostery/interpret/` package. Deterministic candidate extraction (communities, pathways, hubs, coupled-pair clusters) builds on the existing `network.py` graph algorithms. CA-derivable structural features ground each candidate. A `ReportBuilder` always emits JSON + markdown; an opt-in `InterpretationEngine` calls a pluggable `LLMBackend` (Ollama / Anthropic / OpenAI) and merges interpretation additively. A new `allostery interpret` CLI subcommand wires it together, mirroring `analyze`.

**Tech Stack:** Python 3.11+, NumPy, the existing `allostery.network` and `allostery.io` modules. Optional, lazily-imported `anthropic` / `openai` SDKs; Ollama via stdlib `urllib`.

## Global Constraints

- `from __future__ import annotations` at the top of every new module (repo convention).
- No new **hard** dependencies. `anthropic` and `openai` are optional and **lazily imported** only when their backend is selected; Ollama uses stdlib `urllib` only.
- No `networkx` — graph algorithms are hand-rolled, reusing `allostery.network` (which already implements Dijkstra/Brandes/Yen).
- LLM API keys come from environment only (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) — never parameters, config, or logs.
- Only honest CA-derivable structural features (RMSF, contact-count burial proxy, CA geometry). No DSSP/SASA.
- The deterministic candidate report is authoritative; the LLM layer adds an `interpretation` field and never mutates candidate data.
- Residue label format is `"CHAIN:NUM NAME"` (matches `allostery.network` and the scores CSV).
- Anthropic backend (per the `claude-api` reference): model `claude-opus-4-8`, `thinking={"type": "adaptive"}`, structured output via `output_config={"format": {"type": "json_schema", "schema": SCHEMA}}`, `max_tokens=16000`.
- Tests use the existing `fixture_path` fixture (`tests/conftest.py`) and `tests/fixtures/tiny_trajectory.pdb`. No real network or model calls in the suite.

---

### Task 1: Candidate dataclasses + community extraction

**Files:**
- Create: `src/allostery/interpret/__init__.py`
- Create: `src/allostery/interpret/candidates.py`
- Test: `tests/test_interpret_candidates.py`

**Interfaces:**
- Consumes: `allostery.network.AllostericNetwork` (`.node_labels: list[str]`, `.adjacency: dict[int, list[tuple[int, float]]]`, `.num_nodes: int`).
- Produces: dataclasses `Community`, `Pathway`, `Hub`, `Cluster`, `CandidateSet`; function `extract_communities(net: AllostericNetwork) -> list[Community]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_interpret_candidates.py
from __future__ import annotations

from allostery.network import AllostericNetwork
from allostery.interpret.candidates import Community, extract_communities


def _two_triangles() -> AllostericNetwork:
    # Two strongly-connected triangles {0,1,2} and {3,4,5} joined by one weak edge 2-3.
    labels = [f"A:{i} GLY" for i in range(6)]
    adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(6)}

    def link(u: int, v: int, w: float) -> None:
        adj[u].append((v, w))
        adj[v].append((u, w))

    for u, v in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]:
        link(u, v, 1.0)
    link(2, 3, 0.05)
    return AllostericNetwork(node_labels=labels, adjacency=adj)


def test_extract_communities_finds_two_modules() -> None:
    net = _two_triangles()
    communities = extract_communities(net)
    assert all(isinstance(c, Community) for c in communities)
    member_sets = sorted((sorted(c.member_indices) for c in communities))
    assert member_sets == [[0, 1, 2], [3, 4, 5]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_interpret_candidates.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'allostery.interpret'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/allostery/interpret/__init__.py
from __future__ import annotations
```

```python
# src/allostery/interpret/candidates.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from allostery.network import AllostericNetwork


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_interpret_candidates.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/allostery/interpret/__init__.py src/allostery/interpret/candidates.py tests/test_interpret_candidates.py
git commit -m "feat: add candidate dataclasses and community extraction"
```

---

### Task 2: Hub and coupled-pair-cluster extraction

**Files:**
- Modify: `src/allostery/interpret/candidates.py`
- Test: `tests/test_interpret_candidates.py`

**Interfaces:**
- Consumes: `AllostericNetwork`, `allostery.network.betweenness_centrality`, the scores rows from `allostery.network.read_scores_csv` (list of dicts with keys `residue_i_chain/number/name`, `residue_j_*`, `score`).
- Produces: `extract_hubs(net: AllostericNetwork, top_hubs: int = 10) -> list[Hub]`; `extract_clusters(rows: list[dict[str, str]], max_pairs: int = 30) -> list[Cluster]`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_interpret_candidates.py
from allostery.interpret.candidates import Cluster, Hub, extract_clusters, extract_hubs


def _path_graph() -> AllostericNetwork:
    # Path 0-1-2-3-4; node 2 is the bottleneck with highest betweenness.
    labels = [f"A:{i} GLY" for i in range(5)]
    adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(5)}
    for u in range(4):
        adj[u].append((u + 1, 1.0))
        adj[u + 1].append((u, 1.0))
    return AllostericNetwork(node_labels=labels, adjacency=adj)


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
    assert sizes == [2, 3]  # {B9,B10} and {A1,A2,A3}
    assert all(isinstance(c, Cluster) for c in clusters)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_interpret_candidates.py -k "hubs or clusters" -v`
Expected: FAIL — `ImportError: cannot import name 'extract_hubs'`

- [ ] **Step 3: Write minimal implementation**

```python
# add imports at top of src/allostery/interpret/candidates.py
from allostery.network import AllostericNetwork, betweenness_centrality
```

```python
# append to src/allostery/interpret/candidates.py

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_interpret_candidates.py -k "hubs or clusters" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/allostery/interpret/candidates.py tests/test_interpret_candidates.py
git commit -m "feat: add hub and coupled-pair-cluster extraction"
```

---

### Task 3: Pathway extraction + `extract_candidates` orchestrator

**Files:**
- Modify: `src/allostery/interpret/candidates.py`
- Test: `tests/test_interpret_candidates.py`

**Interfaces:**
- Consumes: `AllostericNetwork`, `allostery.network.shortest_paths`, plus `extract_communities`/`extract_hubs`/`extract_clusters` from Tasks 1–2.
- Produces: `extract_pathways(net, hubs, top_paths=5) -> list[Pathway]`; `extract_candidates(net, rows, *, top_paths=5, top_hubs=10) -> CandidateSet`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_interpret_candidates.py
from allostery.interpret.candidates import (
    CandidateSet, Pathway, extract_candidates, extract_pathways,
)


def test_extract_pathways_finds_multi_hop_route() -> None:
    net = _path_graph()  # 0-1-2-3-4
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_interpret_candidates.py -k "pathways or full_set" -v`
Expected: FAIL — `ImportError: cannot import name 'extract_pathways'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to imports at top of src/allostery/interpret/candidates.py
from allostery.network import (
    AllostericNetwork, betweenness_centrality, shortest_paths,
)
```

```python
# append to src/allostery/interpret/candidates.py
from itertools import combinations


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
            if len(labels) < 3:  # require at least 2 hops (multi-hop channel)
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
```

Also add an `__all__` at the end of `candidates.py`:

```python
__all__ = [
    "CandidateSet", "Cluster", "Community", "Hub", "Pathway",
    "extract_candidates", "extract_clusters", "extract_communities",
    "extract_hubs", "extract_pathways",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_interpret_candidates.py -v`
Expected: PASS (all candidate tests)

- [ ] **Step 5: Commit**

```bash
git add src/allostery/interpret/candidates.py tests/test_interpret_candidates.py
git commit -m "feat: add pathway extraction and candidate orchestrator"
```

---

### Task 4: Structural context (CA-derivable features)

**Files:**
- Create: `src/allostery/interpret/structure.py`
- Test: `tests/test_interpret_structure.py`

**Interfaces:**
- Consumes: `allostery.io.pdb.Trajectory` (`.residues: tuple[ResidueRecord, ...]`, `.coordinates: np.ndarray` shape `[frames, n_residues, 3]`).
- Produces: dataclasses `ResidueStructuralFeatures`, `StructuralContext`; function `compute_structural_context(trajectory, contact_cutoff=8.0) -> StructuralContext`. `StructuralContext` exposes `.per_residue: dict[int, ResidueStructuralFeatures]`, `.label_to_index: dict[str, int]`, `.mean_coords: np.ndarray`, `.geometry(labels: list[str]) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_interpret_structure.py
from __future__ import annotations

from pathlib import Path

from allostery.io.pdb import load_multimodel_pdb
from allostery.interpret.structure import (
    ResidueStructuralFeatures, StructuralContext, compute_structural_context,
)


def test_structural_context_from_fixture(fixture_path: Path) -> None:
    trajectory = load_multimodel_pdb(fixture_path / "tiny_trajectory.pdb")
    context = compute_structural_context(trajectory)
    assert isinstance(context, StructuralContext)
    n = trajectory.coordinates.shape[1]
    assert len(context.per_residue) == n
    assert all(isinstance(f, ResidueStructuralFeatures) for f in context.per_residue.values())
    assert all(f.rmsf >= 0.0 for f in context.per_residue.values())
    assert all(f.contact_number >= 0 for f in context.per_residue.values())
    # label map matches residue records
    first = trajectory.residues[0]
    label = f"{first.chain_id}:{first.residue_number} {first.name}"
    assert context.label_to_index[label] == 0


def test_geometry_returns_radius_of_gyration(fixture_path: Path) -> None:
    trajectory = load_multimodel_pdb(fixture_path / "tiny_trajectory.pdb")
    context = compute_structural_context(trajectory)
    labels = list(context.label_to_index.keys())[:2]
    geom = context.geometry(labels)
    assert geom["n_resolved"] == 2
    assert geom["radius_of_gyration"] >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_interpret_structure.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'allostery.interpret.structure'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/allostery/interpret/structure.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allostery.io.pdb import Trajectory


@dataclass
class ResidueStructuralFeatures:
    rmsf: float
    contact_number: int


@dataclass
class StructuralContext:
    per_residue: dict[int, ResidueStructuralFeatures]
    mean_coords: np.ndarray
    label_to_index: dict[str, int]
    contact_cutoff: float

    def geometry(self, labels: list[str]) -> dict[str, float]:
        indices = [self.label_to_index[label] for label in labels if label in self.label_to_index]
        if not indices:
            return {"radius_of_gyration": 0.0, "n_resolved": 0}
        points = self.mean_coords[indices]
        centroid = points.mean(axis=0)
        rg = float(np.sqrt(((points - centroid) ** 2).sum(axis=1).mean()))
        return {"radius_of_gyration": rg, "n_resolved": len(indices)}


def compute_structural_context(trajectory: Trajectory, contact_cutoff: float = 8.0) -> StructuralContext:
    coords = np.asarray(trajectory.coordinates, dtype=np.float64)  # [F, N, 3]
    mean = coords.mean(axis=0)  # [N, 3]
    displacement = coords - mean[None, :, :]
    rmsf = np.sqrt((displacement ** 2).sum(axis=2).mean(axis=0))  # [N]

    diff = mean[:, None, :] - mean[None, :, :]
    distance = np.sqrt((diff ** 2).sum(axis=2))
    contacts = (distance < contact_cutoff) & (distance > 0.0)
    contact_number = contacts.sum(axis=1)

    per_residue = {
        i: ResidueStructuralFeatures(rmsf=float(rmsf[i]), contact_number=int(contact_number[i]))
        for i in range(mean.shape[0])
    }
    label_to_index = {
        f"{r.chain_id}:{r.residue_number} {r.name}": i
        for i, r in enumerate(trajectory.residues)
    }
    return StructuralContext(
        per_residue=per_residue,
        mean_coords=mean,
        label_to_index=label_to_index,
        contact_cutoff=contact_cutoff,
    )


__all__ = [
    "ResidueStructuralFeatures", "StructuralContext", "compute_structural_context",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_interpret_structure.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/allostery/interpret/structure.py tests/test_interpret_structure.py
git commit -m "feat: add CA-derivable structural context"
```

---

### Task 5: Report builder (JSON + markdown)

**Files:**
- Create: `src/allostery/interpret/report.py`
- Test: `tests/test_interpret_report.py`

**Interfaces:**
- Consumes: `CandidateSet` (Task 1–3), `StructuralContext | None` (Task 4).
- Produces: `build_report(candidates, context, *, source, parameters) -> dict`; `render_markdown(report: dict) -> str`; `write_report(report: dict, json_path, md_path) -> None`. JSON shape: `{"schema_version": 1, "source", "parameters", "candidates": {"communities", "pathways", "hubs", "clusters"}, "structural_context": bool}`. Each candidate item carries an `"evidence"` dict and no `"interpretation"` key until the engine adds one.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_interpret_report.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_interpret_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'allostery.interpret.report'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/allostery/interpret/report.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_interpret_report.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/allostery/interpret/report.py tests/test_interpret_report.py
git commit -m "feat: add interpretation report builder (json + markdown)"
```

---

### Task 6: LLM backend protocol, factory, and adapters

**Files:**
- Create: `src/allostery/interpret/llm/__init__.py`
- Create: `src/allostery/interpret/llm/ollama.py`
- Create: `src/allostery/interpret/llm/anthropic.py`
- Create: `src/allostery/interpret/llm/openai.py`
- Test: `tests/test_interpret_llm.py`

**Interfaces:**
- Produces: `LLMBackend` Protocol with `generate_json(self, system: str, user: str, schema: dict) -> dict`; `make_backend(name: str, *, model: str | None = None, base_url: str | None = None) -> LLMBackend`. Adapters: `OllamaBackend(model, base_url="http://localhost:11434", *, urlopen=None)`, `AnthropicBackend(model="claude-opus-4-8", *, client=None)`, `OpenAIBackend(model="gpt-4.1", *, client=None)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_interpret_llm.py
from __future__ import annotations

import json

import pytest

from allostery.interpret.llm import make_backend
from allostery.interpret.llm.ollama import OllamaBackend
from allostery.interpret.llm.anthropic import AnthropicBackend
from allostery.interpret.llm.openai import OpenAIBackend


def test_make_backend_dispatch() -> None:
    assert isinstance(make_backend("ollama", model="qwen3"), OllamaBackend)
    assert isinstance(make_backend("anthropic"), AnthropicBackend)
    assert isinstance(make_backend("openai"), OpenAIBackend)
    with pytest.raises(ValueError):
        make_backend("nope")


def test_ollama_backend_parses_response() -> None:
    class _Resp:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload
        def read(self) -> bytes:
            return self._payload
        def __enter__(self): return self
        def __exit__(self, *exc): return False

    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        body = {"message": {"content": json.dumps({"ok": True})}}
        return _Resp(json.dumps(body).encode("utf-8"))

    backend = OllamaBackend(model="qwen3", urlopen=fake_urlopen)
    result = backend.generate_json("sys", "user", {"type": "object"})
    assert result == {"ok": True}
    assert captured["url"].endswith("/api/chat")


def test_anthropic_backend_parses_response() -> None:
    class _Block:
        type = "text"
        text = json.dumps({"summary": "hi"})

    class _Message:
        content = [_Block()]

    class _Messages:
        def create(self, **kwargs):
            assert kwargs["model"] == "claude-opus-4-8"
            assert kwargs["thinking"] == {"type": "adaptive"}
            return _Message()

    class _Client:
        messages = _Messages()

    backend = AnthropicBackend(client=_Client())
    assert backend.generate_json("sys", "user", {"type": "object"}) == {"summary": "hi"}


def test_openai_backend_parses_response() -> None:
    class _Choice:
        class message:  # noqa: N801
            content = json.dumps({"summary": "hi"})

    class _Completions:
        def create(self, **kwargs):
            class _R:
                choices = [_Choice()]
            return _R()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    backend = OpenAIBackend(client=_Client())
    assert backend.generate_json("sys", "user", {"type": "object"}) == {"summary": "hi"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_interpret_llm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'allostery.interpret.llm'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/allostery/interpret/llm/__init__.py
from __future__ import annotations

from typing import Protocol


class LLMBackend(Protocol):
    def generate_json(self, system: str, user: str, schema: dict) -> dict:
        ...


def make_backend(
    name: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> LLMBackend:
    if name == "ollama":
        from allostery.interpret.llm.ollama import OllamaBackend
        return OllamaBackend(
            model=model or "qwen3",
            base_url=base_url or "http://localhost:11434",
        )
    if name == "anthropic":
        from allostery.interpret.llm.anthropic import AnthropicBackend
        return AnthropicBackend(model=model or "claude-opus-4-8")
    if name == "openai":
        from allostery.interpret.llm.openai import OpenAIBackend
        return OpenAIBackend(model=model or "gpt-4.1")
    raise ValueError(f"unknown llm backend {name!r}; expected ollama, anthropic, or openai")


__all__ = ["LLMBackend", "make_backend"]
```

```python
# src/allostery/interpret/llm/ollama.py
from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable


class OllamaBackend:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        *,
        urlopen: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._urlopen = urlopen or urllib.request.urlopen

    def generate_json(self, system: str, user: str, schema: dict) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": schema,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
        return json.loads(body["message"]["content"])


__all__ = ["OllamaBackend"]
```

```python
# src/allostery/interpret/llm/anthropic.py
from __future__ import annotations

import json
from typing import Any


class AnthropicBackend:
    def __init__(self, model: str = "claude-opus-4-8", *, client: Any | None = None) -> None:
        self.model = model
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - exercised via message
                raise ImportError(
                    "the anthropic backend requires the 'anthropic' package: pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client

    def generate_json(self, system: str, user: str, schema: dict) -> dict:
        client = self._ensure_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = next(block.text for block in message.content if block.type == "text")
        return json.loads(text)


__all__ = ["AnthropicBackend"]
```

```python
# src/allostery/interpret/llm/openai.py
from __future__ import annotations

import json
from typing import Any


class OpenAIBackend:
    def __init__(self, model: str = "gpt-4.1", *, client: Any | None = None) -> None:
        self.model = model
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - exercised via message
                raise ImportError(
                    "the openai backend requires the 'openai' package: pip install openai"
                ) from exc
            self._client = openai.OpenAI()
        return self._client

    def generate_json(self, system: str, user: str, schema: dict) -> dict:
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "interpretation", "schema": schema},
            },
        )
        return json.loads(response.choices[0].message.content)


__all__ = ["OpenAIBackend"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_interpret_llm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/allostery/interpret/llm tests/test_interpret_llm.py
git commit -m "feat: add pluggable llm backends (ollama, anthropic, openai)"
```

---

### Task 7: Prompts + interpretation engine

**Files:**
- Create: `src/allostery/interpret/prompts.py`
- Create: `src/allostery/interpret/engine.py`
- Test: `tests/test_interpret_engine.py`

**Interfaces:**
- Consumes: a report dict (Task 5), an `LLMBackend` (Task 6).
- Produces: `prompts.SYSTEM_PROMPT: str`, `prompts.RESPONSE_SCHEMA: dict`, `prompts.build_user_prompt(candidate_type: str, item: dict) -> str`; `engine.interpret_report(report: dict, backend: LLMBackend) -> dict` (returns a new report with an `interpretation` key on each candidate item).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_interpret_engine.py
from __future__ import annotations

from allostery.interpret.engine import interpret_report


class _FakeBackend:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate_json(self, system: str, user: str, schema: dict) -> dict:
        self.calls += 1
        return self._responses.pop(0)


def _report() -> dict:
    return {
        "schema_version": 1,
        "source": "scores.csv",
        "parameters": {},
        "structural_context": False,
        "candidates": {
            "communities": [],
            "pathways": [],
            "hubs": [{"type": "hub", "label": "A:5 GLY", "centrality": 0.5,
                      "degree": 3, "evidence": {"label": "A:5 GLY"}}],
            "clusters": [],
        },
    }


def _valid() -> dict:
    return {
        "summary": "key control point", "mechanism_hypothesis": "couples sites",
        "key_residues": [{"label": "A:5 GLY", "role": "hub", "evidence_refs": ["centrality"]}],
        "confidence": "medium", "parametric": True, "caveats": "no external validation",
    }


def test_interpret_report_merges_interpretation() -> None:
    backend = _FakeBackend([_valid()])
    enriched = interpret_report(_report(), backend)
    hub = enriched["candidates"]["hubs"][0]
    assert hub["interpretation"]["confidence"] == "medium"
    assert backend.calls == 1


def test_interpret_report_retries_then_falls_back_on_invalid_json() -> None:
    backend = _FakeBackend([{"bad": "shape"}, {"still": "bad"}])
    enriched = interpret_report(_report(), backend)
    hub = enriched["candidates"]["hubs"][0]
    assert hub["interpretation"]["invalid"] is True
    assert "raw" in hub["interpretation"]
    assert backend.calls == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_interpret_engine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'allostery.interpret.engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/allostery/interpret/prompts.py
from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = (
    "You are a structural-biology assistant interpreting the output of a deep-learning "
    "allostery model. You are given a candidate allosteric structure with topological and "
    "structural evidence computed from the protein. Ground every statement in the supplied "
    "evidence. If you assert a functional role from prior knowledge that is not present in the "
    "evidence, set \"parametric\" to true and lower the confidence. Never invent residues or "
    "numbers that are not in the evidence. Respond only with JSON matching the requested schema."
)

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "mechanism_hypothesis": {"type": "string"},
        "key_residues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string"},
                    "role": {"type": "string"},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["label", "role", "evidence_refs"],
            },
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "parametric": {"type": "boolean"},
        "caveats": {"type": "string"},
    },
    "required": [
        "summary", "mechanism_hypothesis", "key_residues",
        "confidence", "parametric", "caveats",
    ],
}


def build_user_prompt(candidate_type: str, item: dict[str, Any]) -> str:
    return (
        f"Candidate type: {candidate_type}\n"
        f"Evidence (JSON):\n{json.dumps(item, indent=2)}\n\n"
        "Interpret this candidate's likely allosteric role. Return JSON only."
    )


__all__ = ["RESPONSE_SCHEMA", "SYSTEM_PROMPT", "build_user_prompt"]
```

```python
# src/allostery/interpret/engine.py
from __future__ import annotations

import copy
from typing import Any

from allostery.interpret.llm import LLMBackend
from allostery.interpret.prompts import RESPONSE_SCHEMA, SYSTEM_PROMPT, build_user_prompt

_REQUIRED = ("summary", "mechanism_hypothesis", "key_residues", "confidence", "parametric", "caveats")


def _is_valid(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if any(key not in obj for key in _REQUIRED):
        return False
    if obj["confidence"] not in ("low", "medium", "high"):
        return False
    if not isinstance(obj["parametric"], bool):
        return False
    if not isinstance(obj["key_residues"], list):
        return False
    return True


def _interpret_item(candidate_type: str, item: dict[str, Any], backend: LLMBackend) -> dict[str, Any]:
    user = build_user_prompt(candidate_type, item)
    last: Any = None
    for _attempt in range(2):
        last = backend.generate_json(SYSTEM_PROMPT, user, RESPONSE_SCHEMA)
        if _is_valid(last):
            return last
    return {"invalid": True, "raw": last}


def interpret_report(report: dict[str, Any], backend: LLMBackend) -> dict[str, Any]:
    enriched = copy.deepcopy(report)
    candidates = enriched["candidates"]
    for candidate_type, items in candidates.items():
        for item in items:
            item["interpretation"] = _interpret_item(candidate_type, item, backend)
    return enriched


__all__ = ["interpret_report"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_interpret_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/allostery/interpret/prompts.py src/allostery/interpret/engine.py tests/test_interpret_engine.py
git commit -m "feat: add interpretation prompts and engine"
```

---

### Task 8: Pipeline orchestrator

**Files:**
- Create: `src/allostery/pipeline/interpret.py`
- Test: `tests/test_pipeline_interpret.py`

**Interfaces:**
- Consumes: `allostery.network.read_scores_csv` + `build_graph`; `extract_candidates`; `compute_structural_context`; `build_report`/`write_report`; `make_backend`; `interpret_report`; `allostery.io.trajectory.load_trajectory`.
- Produces: `run_interpretation(scores_csv, *, out_json, out_md, pdb_path=None, topology_path=None, top_k=20, top_paths=5, top_hubs=10, llm="none", llm_model=None, llm_base_url=None, backend=None) -> dict`. The `backend` parameter is for tests/injection; when `None` and `llm != "none"`, it is built via `make_backend`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_interpret.py
from __future__ import annotations

import json
from pathlib import Path

from allostery.pipeline.interpret import run_interpretation


def _write_scores(path: Path) -> None:
    header = ("rank,score,residue_i_index,residue_i_chain,residue_i_number,residue_i_name,"
              "residue_j_index,residue_j_chain,residue_j_number,residue_j_name\n")
    rows = [
        "1,0.9,0,A,1,GLY,1,A,2,GLY",
        "2,0.8,1,A,2,GLY,2,A,3,GLY",
        "3,0.7,2,A,3,GLY,3,A,4,GLY",
    ]
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def test_run_interpretation_without_llm(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    report = run_interpretation(scores, out_json=out_json, out_md=out_md)
    assert out_json.exists() and out_md.exists()
    assert report["structural_context"] is False
    assert "interpretation" not in report["candidates"]["hubs"][0]


def test_run_interpretation_with_injected_backend(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)

    class _FakeBackend:
        def generate_json(self, system, user, schema):
            return {
                "summary": "s", "mechanism_hypothesis": "m",
                "key_residues": [], "confidence": "low",
                "parametric": False, "caveats": "c",
            }

    report = run_interpretation(
        scores, out_json=tmp_path / "o.json", out_md=tmp_path / "o.md",
        llm="ollama", backend=_FakeBackend(),
    )
    assert report["candidates"]["hubs"][0]["interpretation"]["confidence"] == "low"
    loaded = json.loads((tmp_path / "o.json").read_text())
    assert loaded["candidates"]["hubs"][0]["interpretation"]["confidence"] == "low"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_interpret.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'allostery.pipeline.interpret'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/allostery/pipeline/interpret.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_interpret.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/allostery/pipeline/interpret.py tests/test_pipeline_interpret.py
git commit -m "feat: add interpretation pipeline orchestrator"
```

---

### Task 9: CLI `interpret` subcommand

**Files:**
- Modify: `src/allostery/cli.py`
- Test: `tests/test_cli_interpret.py`

**Interfaces:**
- Consumes: `run_interpretation` (Task 8).
- Produces: a new `interpret` subcommand on the existing argparse parser, dispatched in `main`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_interpret.py
from __future__ import annotations

from pathlib import Path

from allostery.cli import main


def _write_scores(path: Path) -> None:
    header = ("rank,score,residue_i_index,residue_i_chain,residue_i_number,residue_i_name,"
              "residue_j_index,residue_j_chain,residue_j_number,residue_j_name\n")
    rows = [
        "1,0.9,0,A,1,GLY,1,A,2,GLY",
        "2,0.8,1,A,2,GLY,2,A,3,GLY",
        "3,0.7,2,A,3,GLY,3,A,4,GLY",
    ]
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def test_cli_interpret_writes_outputs(tmp_path: Path, capsys) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"
    code = main([
        "interpret", str(scores),
        "--out-json", str(out_json), "--out-md", str(out_md),
    ])
    assert code == 0
    assert out_json.exists() and out_md.exists()
    assert "interpret" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_interpret.py -v`
Expected: FAIL — argparse exits non-zero / `SystemExit` because `interpret` is not a known subcommand.

- [ ] **Step 3: Write minimal implementation**

In `src/allostery/cli.py`, add the import near the other pipeline imports (after line 15, `from allostery.pipeline.influence_train import train_influence_model`):

```python
from allostery.pipeline.interpret import run_interpretation
```

Add `'interpret'` to the subcommands frozenset (currently `_SUBCOMMANDS = frozenset({'run', 'analyze', 'check'})`):

```python
_SUBCOMMANDS = frozenset({'run', 'analyze', 'check', 'interpret'})
```

In `build_parser`, after the `analyze_parser` block (before `return parser`), add:

```python
    interpret_parser = subparsers.add_parser(
        'interpret', help='Extract candidate allosteric networks and interpret a scores CSV')
    interpret_parser.add_argument('scores_csv', help='Path to scores CSV produced by a pipeline run')
    interpret_parser.add_argument('--pdb', default=None, help='Reference structure/trajectory for structural context')
    interpret_parser.add_argument('--topology', default=None, help='Topology file for non-PDB trajectories')
    interpret_parser.add_argument('--top-k', type=int, default=20, help='Edges to include when building the graph')
    interpret_parser.add_argument('--top-paths', type=int, default=5, help='Candidate pathways to report')
    interpret_parser.add_argument('--top-hubs', type=int, default=10, help='Hub residues to report')
    interpret_parser.add_argument('--out-json', default=None, help='Output JSON path (default: <scores>.interpret.json)')
    interpret_parser.add_argument('--out-md', default=None, help='Output markdown path (default: <scores>.interpret.md)')
    interpret_parser.add_argument('--llm', default='none', choices=['none', 'ollama', 'anthropic', 'openai'],
                                  help='LLM backend for interpretation (default: none)')
    interpret_parser.add_argument('--llm-model', default=None, help='Model name for the chosen backend')
    interpret_parser.add_argument('--llm-base-url', default=None, help='Base URL (Ollama; default http://localhost:11434)')
```

In `main`, add a dispatch branch after the `analyze` branch (after the block ending with `return 0` for analyze, before the `# Dispatch: subcommand 'run'` comment):

```python
    # Dispatch: subcommand 'interpret'
    if args.command == 'interpret':
        scores_path = Path(args.scores_csv)
        out_json = args.out_json or scores_path.with_suffix('.interpret.json')
        out_md = args.out_md or scores_path.with_suffix('.interpret.md')
        report = run_interpretation(
            scores_path,
            out_json=out_json,
            out_md=out_md,
            pdb_path=args.pdb,
            topology_path=args.topology,
            top_k=args.top_k,
            top_paths=args.top_paths,
            top_hubs=args.top_hubs,
            llm=args.llm,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
        )
        counts = {key: len(value) for key, value in report['candidates'].items()}
        print(f'interpret candidates={counts} json={out_json} md={out_md}')
        return 0
```

(`Path` is already imported at the top of `cli.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_interpret.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

Run: `pytest -q`
Expected: PASS (all existing tests plus the new interpret tests).

```bash
git add src/allostery/cli.py tests/test_cli_interpret.py
git commit -m "feat: add 'interpret' CLI subcommand"
```

---

## Self-Review

**Spec coverage:**
- §3 package layout → Tasks 1–9 create exactly `interpret/{candidates,structure,report,engine,prompts}.py`, `interpret/llm/{__init__,ollama,anthropic,openai}.py`, `pipeline/interpret.py`, CLI.
- §4.1 four candidate types → Task 1 (communities), Task 2 (hubs, clusters), Task 3 (pathways + orchestrator). ✓
- §4.2 structural context (RMSF, contact number, geometry) → Task 4. ✓
- §4.3 ReportBuilder JSON+markdown, interpretation absent until engine → Task 5. ✓
- §4.4 LLMBackend protocol, factory, lazy adapters, env-only keys, Anthropic specifics → Task 6. ✓
- §4.5 engine: per-candidate grounded prompt, validate, retry-once, raw-text fallback, additive merge → Task 7. ✓
- §5 CLI subcommand + flags + default output paths → Task 9. ✓
- §6 data flow (stop at no-LLM) → Task 8. ✓
- §7 error handling reuses `read_scores_csv`/`load_trajectory` errors; backend ImportError messages in Task 6; malformed-JSON fallback in Task 7. ✓
- §8 testing: deterministic candidate tests (1–3), structure fixture (4), report round-trip (5), FakeBackend engine (7), mocked adapters (6), CLI `--llm none` e2e (9). ✓

**Placeholder scan:** No TBD/TODO; every code step contains complete code; tests contain real assertions. ✓

**Type consistency:** `CandidateSet`/`Community`/`Pathway`/`Hub`/`Cluster` defined in Task 1 and used unchanged in Tasks 3/5; `extract_candidates(net, rows, *, top_paths, top_hubs)` signature consistent between Task 3 definition and Task 8 call; `compute_structural_context(trajectory, contact_cutoff=8.0)` consistent (Task 4 def, Task 8 call); `build_report(candidates, context, *, source, parameters)` and `write_report(report, json_path, md_path)` consistent (Task 5 def, Task 8 call); `make_backend(name, *, model, base_url)` consistent (Task 6 def, Task 8 call); `interpret_report(report, backend)` consistent (Task 7 def, Task 8 call); `run_interpretation(...)` signature consistent (Task 8 def, Task 9 call). ✓
