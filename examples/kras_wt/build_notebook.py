#!/usr/bin/env python3
"""Build examples/kras_wt/demo.ipynb from cell definitions."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    }
    nb["cells"] = _cells()
    out = Path(__file__).parent / "demo.ipynb"
    nbf.write(nb, str(out))
    print(f"Wrote {out}  ({len(nb['cells'])} cells)")


def _cells() -> list[nbf.NotebookNode]:
    return [
        # ── §1 Introduction ──────────────────────────────────────────────────
        md("""\
# KRAS WT Allosteric Network Analysis
### A worked example using the `allostery` CLI

This notebook demonstrates the complete allosteric network analysis pipeline \
using a 1 µs molecular dynamics simulation of GDP-bound KRAS4B (wild type).

---

## What is an allosteric network?

Allostery describes how a signal at one site in a protein influences a distant \
site. In KRAS, GDP/GTP binding at the nucleotide pocket modulates effector \
binding 15–20 Å away. The **allosteric network** is the set of residue-pair \
couplings that transmit this signal.

The `allostery` influence model learns these couplings from MD trajectories by \
predicting each residue's acceleration as a weighted sum of messages from all \
other residues. Pairs whose motions systematically predict each other's \
accelerations receive high influence scores.

---

## System

| Property | Value |
|----------|-------|
| Protein | KRAS4B GDP-bound (wild type) |
| Residues | 169 (chain A) |
| Trajectory | 1 µs MD, 5001 frames at 200 ps/frame |
| Pre-computed scores | included — training can be skipped |

---

## What this notebook demonstrates

1. How to configure and validate your trajectory input
2. How to train the influence model and score residue pairs
3. How to analyse the resulting allosteric network
4. How different time-window sizes reveal different layers of the mechanism
5. How to visualise hub residues, score distributions, and the network graph\
"""),

        # ── §2 Configuration ─────────────────────────────────────────────────
        md("""\
---
## Step 1: Configure and validate your trajectory

Edit `config` below to point to your own trajectory.
A template with all fields documented is at `examples/kras_wt/template_config.yaml`.

**Key parameters:**

| Parameter | Meaning |
|-----------|---------|
| `pdb_path` | Trajectory file (.trr, .xtc, .dcd, or multi-model .pdb) |
| `topology_path` | Topology file (.gro, .tpr, .psf) — omit for .pdb |
| `window_size` | Frames per window; `window_size × time_step` = window duration (ps) |
| `stride` | Frames between window starts (1 = maximal overlap) |
| `time_step` | Physical time per saved frame in ps |
| `top_k` | Top-scoring pairs used as network edges |\
"""),

        code("""\
from pathlib import Path
import sys, subprocess

# ── Change these for your own protein ────────────────────────────────────────
config = "configs/kras_wt_influence.yaml"
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(".").resolve()
assert (REPO_ROOT / "configs" / "kras_wt_influence.yaml").exists(), (
    "Run Jupyter from the repo root:\\n"
    "  cd /path/to/allostery && jupyter notebook examples/kras_wt/demo.ipynb"
)
sys.path.insert(0, str(REPO_ROOT / "src"))

FIGURES_DIR = REPO_ROOT / "examples" / "kras_wt" / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

print(f"Repo root : {REPO_ROOT}")
print(f"Figures   : {FIGURES_DIR}")
print(f"Config    : {config}")\
"""),

        code("""\
# Validate config and inspect trajectory
result = subprocess.run(
    ["./scripts/preprocess.sh", config],
    capture_output=True, text=True,
)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr, file=sys.stderr)\
"""),

        # ── §3 Train ─────────────────────────────────────────────────────────
        md("""\
---
## Step 2: Train — *skip this section if using pre-computed KRAS WT scores*

`scripts/train.sh` runs `allostery run <config>`, which:
1. Loads the trajectory and extracts sliding windows
2. Trains the influence model (predicts per-residue accelerations)
3. Scores all residue pairs and saves results to the CSV

Training takes ~5–10 minutes on GPU for a 1 µs / 5001-frame trajectory.\
"""),

        code("""\
# SKIP if using pre-computed scores — uncomment to train from scratch
#
# result = subprocess.run(
#     ["./scripts/train.sh", config],
#     capture_output=True, text=True,
# )
# print(result.stdout)

# Verify pre-computed scores exist
scores_w5 = Path("outputs/kras_wt/influence_scores.csv")
assert scores_w5.exists(), (
    f"Scores not found: {scores_w5}\\n"
    "Run training first or check your output.score_csv_path in the config."
)
n_pairs = sum(1 for _ in open(scores_w5)) - 1  # subtract header
print(f"Scores ready : {scores_w5}")
print(f"Pairs scored : {n_pairs:,}")\
"""),

        # ── §4 Analyze ───────────────────────────────────────────────────────
        md("""\
---
## Step 3: Analyse the allosteric network

`run_network_analysis` loads the scores CSV, builds a weighted graph from the \
top-k pairs, and reports:

- **Hub residues** ranked by betweenness centrality — residues that lie on \
many shortest paths in the network
- **Suggested threshold** — the score value where the largest gap occurs within \
the top-k pairs (Kneedle knee detection)
- **Score distribution** — histogram of all scored pairs\
"""),

        code("""\
from allostery.pipeline.analyze import run_network_analysis

report_w5 = run_network_analysis(str(scores_w5), top_k=30)
print(report_w5)\
"""),

        # ── §5 Visualise — figure 1: hub bar chart ───────────────────────────
        md("""\
---
## Step 4: Visualise

### Hub residue centrality\
"""),

        code("""\
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from allostery.network import read_scores_csv, build_graph, betweenness_centrality


def hub_centrality(scores_csv: Path, top_k: int = 30) -> list[tuple[str, float]]:
    rows = read_scores_csv(str(scores_csv))
    net = build_graph(rows, top_k=top_k)
    cent = betweenness_centrality(net)
    pairs = [(net.node_labels[i], v) for i, v in cent.items()]
    return sorted(pairs, key=lambda x: -x[1])[:10]


hubs_w5 = hub_centrality(scores_w5)
labels = [h[0] for h in hubs_w5]
values = [h[1] for h in hubs_w5]

fig, ax = plt.subplots(figsize=(7, 4))
colors = cm.Reds_r(np.linspace(0.1, 0.7, len(values)))
ax.barh(labels[::-1], values[::-1], color=colors[::-1])
ax.set_xlabel("Betweenness centrality")
ax.set_title("Hub residues — window_size=5 (1 ns)")
ax.set_xlim(0, 1)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "hub_centrality_w5.png", dpi=150)
plt.show()
print(f"Saved: {FIGURES_DIR / 'hub_centrality_w5.png'}")\
"""),

        # ── §5 Visualise — figure 2: score distribution ──────────────────────
        md("### Score distribution"),

        code("""\
with open(scores_w5) as f:
    all_scores = [float(r["score"]) for r in csv.DictReader(f)]

top_score = max(all_scores)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(all_scores, bins=50, color="#c0392b", alpha=0.8, edgecolor="white", linewidth=0.3)
ax.axvline(top_score, color="black", linestyle="--", linewidth=1.5,
           label=f"Top score: {top_score:.4f}")
ax.set_yscale("log")
ax.set_xlabel("Influence score")
ax.set_ylabel("Number of pairs (log scale)")
ax.set_title(f"Score distribution — {len(all_scores):,} pairs")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "score_distribution_w5.png", dpi=150)
plt.show()
print(f"Saved: {FIGURES_DIR / 'score_distribution_w5.png'}")\
"""),

        # ── §5 Visualise — figure 3: network graph ───────────────────────────
        md("### Allosteric network graph"),

        code("""\
import networkx as nx

rows = read_scores_csv(str(scores_w5))
G = nx.Graph()
for r in rows[:30]:
    i_label = f"{r['residue_i_chain']}:{r['residue_i_number']} {r['residue_i_name']}"
    j_label = f"{r['residue_j_chain']}:{r['residue_j_number']} {r['residue_j_name']}"
    G.add_edge(i_label, j_label, weight=float(r["score"]))

net = build_graph(rows, top_k=30)
cent = betweenness_centrality(net)
node_sizes = {
    net.node_labels[i]: 200 + 1800 * v
    for i, v in cent.items()
}

pos = nx.spring_layout(G, seed=42, k=2.0)
edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
max_w = max(edge_weights)
node_labels = {n: n.split(":")[1].split()[0] for n in G.nodes()}  # residue number only
sizes = [node_sizes.get(n, 200) for n in G.nodes()]

fig, ax = plt.subplots(figsize=(9, 7))
nx.draw_networkx_edges(
    G, pos,
    edge_color=[w / max_w for w in edge_weights],
    edge_cmap=plt.cm.Reds,
    width=2.5, alpha=0.8, ax=ax,
)
nx.draw_networkx_nodes(G, pos, node_color="#2c3e50", node_size=sizes, ax=ax)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_color="white",
                        font_size=7, ax=ax)
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.Reds,
    norm=plt.Normalize(vmin=min(edge_weights), vmax=max_w),
)
plt.colorbar(sm, ax=ax, label="Influence score", shrink=0.6)
ax.set_title("Allosteric network — top-30 pairs (window_size=5, 1 ns)")
ax.axis("off")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "network_graph_w5.png", dpi=150)
plt.show()
print(f"Saved: {FIGURES_DIR / 'network_graph_w5.png'}")\
"""),

        # ── §6 Multi-timescale comparison ────────────────────────────────────
        md("""\
---
## Step 5: Multi-timescale comparison

The `window_size` parameter controls which **timescale of dynamics** the model \
captures. Pairs of window sizes reveal different layers of the allosteric mechanism.

| Window | Duration | Dynamics captured |
|--------|----------|-------------------|
| w5     | 1 ns     | Fast backbone fluctuations, Switch I jitter |
| w25    | 5 ns     | Slow collective motions, α3 helix breathing |

Run `scripts/train.sh configs/kras_wt_w25.yaml` to train the 5 ns model \
(pre-computed scores are already included).\
"""),

        code("""\
scores_w25 = Path("outputs/kras_wt_w25/influence_scores.csv")
assert scores_w25.exists(), (
    f"w25 scores not found: {scores_w25}\\n"
    "Run: ./scripts/train.sh configs/kras_wt_w25.yaml"
)

hubs_w25 = hub_centrality(scores_w25)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

def _plot_hubs(ax, hubs, title):
    labels = [h[0] for h in hubs]
    values = [h[1] for h in hubs]
    colors = ["#e74c3c" if v > 0.05 else "#bdc3c7" for v in values]
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("Betweenness centrality")
    ax.set_title(title)
    ax.set_xlim(0, 1)

_plot_hubs(ax1, hubs_w5,  "window_size=5  (1 ns)")
_plot_hubs(ax2, hubs_w25, "window_size=25 (5 ns)")
fig.suptitle("Hub residues: fast vs slow dynamics", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "hub_comparison_w5_w25.png", dpi=150)
plt.show()
print(f"Saved: {FIGURES_DIR / 'hub_comparison_w5_w25.png'}")

# Print interpretation
top_w5  = hubs_w5[0][0]
top_w25 = hubs_w25[0][0]
print(f"\\nw5  top hub : {top_w5}  (fast Switch I fluctuations)")
print(f"w25 top hub : {top_w25}  (slow α3-helix collective motion)")
print("\\nInterpretation: the allosteric signal propagates")
print(f"  {top_w5} (Switch I hinge) → {top_w25} (α3 helix) → effector interface")\
"""),

        # ── §7 Next steps ────────────────────────────────────────────────────
        md("""\
---
## Next steps

### Pathway analysis
Find the shortest allosteric path between two residues of interest:

```bash
./scripts/analyze.sh outputs/kras_wt/influence_scores.csv \\
    --source "A:48 GLY" --sink "A:121 PRO"
```

### PyMOL visualisation
Export a `.pml` script to colour residues and edges in PyMOL:

```bash
./scripts/analyze.sh outputs/kras_wt/influence_scores.csv \\
    --pdb /path/to/structure.pdb \\
    --out-pml outputs/kras_wt/network.pml
```

Then in PyMOL: `File → Run Script → network.pml`

### Choosing window_size for your system

- Start with **2–3 window sizes** spanning 1–10× your expected allosteric timescale
- Each window needs ≥ 100 **independent** samples: `trajectory_length / window_size`
- For 10 ns windows you need ≥ 1 µs of trajectory
- Compare hub residues across window sizes — residues that appear consistently \
are the most reliable allosteric hubs

### Running on your own trajectory

1. Copy `examples/kras_wt/template_config.yaml` → `configs/my_protein.yaml`
2. Set `pdb_path`, `topology_path`, `time_step` for your system
3. Run: `./scripts/preprocess.sh configs/my_protein.yaml`
4. Run: `./scripts/train.sh configs/my_protein.yaml`
5. Run: `./scripts/analyze.sh outputs/my_protein/influence_scores.csv`\
"""),
    ]


if __name__ == "__main__":
    build()
