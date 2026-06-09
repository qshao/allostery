# Allosteric Network Analysis Design

**Date:** 2026-06-08  
**Status:** Approved — proceeding to implementation

## Goal

Post-process a scored residue-residue pairs CSV to extract the allosteric network and channels: network summary, hub (bottleneck) residues ranked by betweenness centrality, and optional shortest-path channels between a user-specified source and sink residue.

## Input

A scores CSV produced by any `allostery` pipeline run (relational, influence, or cri). Required columns: `score`, `residue_i_chain`, `residue_i_number`, `residue_i_name`, `residue_j_chain`, `residue_j_number`, `residue_j_name`. Optional directed columns: `influence_i_on_j`, `influence_j_on_i`.

## CLI

```
allostery analyze scores.csv [--top-k 20] [--source A:12] [--sink A:87] [--top-paths 5]
```

- `--top-k N` — include only the top-N scoring pairs as graph edges (default 20)  
- `--source CHAIN:NUM` — source residue for path finding (e.g. `A:12`)  
- `--sink CHAIN:NUM` — sink residue for path finding  
- `--top-paths N` — number of shortest paths to list (default 5)

## New Files

- `src/allostery/network.py` — pure-stdlib graph construction, Dijkstra, betweenness centrality, text report
- `src/allostery/pipeline/analyze.py` — reads CSV, calls network module, prints report
- Modify `src/allostery/cli.py` — add `analyze` subcommand via `argparse` subparsers
- `tests/test_network.py` — unit tests for graph construction, shortest path, betweenness

## Architecture

```
scores.csv
  → read_scores_csv()          # list[dict] with score + residue identity
  → build_graph()              # AllosticNetwork (adjacency dict, node labels)
  → network_summary()          # str: N nodes, M edges, K components
  → hub_residues()             # list[(node_label, centrality)] sorted desc
  → shortest_paths()           # list[list[node_label]] source→sink
  → format_report()            # assembled text report
```

`AllostericNetwork` stores:
- `nodes`: dict mapping `node_id` (str "CHAIN:NUM NAME") → index
- `edges`: list of `(i, j, weight)` where `weight = score` (undirected)
- `adjacency`: `dict[int, list[(int, float)]]` for algorithms

Edge weight = `score` from CSV. Distance metric = `1 / score` (higher score = shorter path).

## Output (stdout)

```
=== Allosteric Network ===
Residues (nodes): 8
Edges (top-20 pairs): 20
Connected components: 1

=== Hub Residues (Betweenness Centrality) ===
 1.  A:3  SER   0.4286
 2.  A:2  ALA   0.2857
...

=== Allosteric Channel: A:1 GLY → A:3 SER ===
Path 1 (length 1):  A:1 GLY → A:3 SER   (score 0.499)
Path 2 (length 2):  A:1 GLY → A:2 ALA → A:3 SER   (score 0.487)
```

## No New Dependencies

All algorithms implemented with stdlib (`heapq`, `collections`). No networkx required.
