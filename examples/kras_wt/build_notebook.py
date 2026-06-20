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
        "language_info": {"name": "python", "version": "3.11.0"},
    }
    nb["cells"] = _cells()
    out = Path(__file__).parent / "demo.ipynb"
    nbf.write(nb, str(out))
    print(f"Wrote {out}  ({len(nb['cells'])} cells)")


def _cells() -> list[nbf.NotebookNode]:
    return [
        # ── §0 Setup ─────────────────────────────────────────────────────────
        md("""\
# KRAS WT Allosteric Network Analysis
### A worked example using the `allostery` CLI

---

## Setup: create a virtual environment (run once in your terminal)

Before opening this notebook, create and activate a Python 3.11+ virtual \
environment and install the required packages:

```bash
# 1. Clone the repository
git clone https://github.com/qshao/allostery.git
cd allostery

# 2. Create and activate a virtual environment
python3.11 -m venv allostery-env
source allostery-env/bin/activate          # Linux / macOS
# allostery-env\\Scripts\\activate.bat     # Windows — use this line instead

# 3. Install the allostery package (development mode)
pip install -e .

# 4. Install notebook and visualisation dependencies
pip install matplotlib networkx jupyter mdtraj nbformat

# 5. Launch Jupyter from the repo root (important — all paths are relative to here)
jupyter notebook examples/kras_wt/demo.ipynb
```

> **GPU note:** PyTorch is installed as a CPU-only build by default. For GPU \
training add `--index-url https://download.pytorch.org/whl/cu121` \
(adjust for your CUDA version) before the `pip install -e .` step, or \
set `device: cpu` in your config if no GPU is available.

> **Trajectory files:** The 1 µs KRAS WT trajectory is not included in the \
repository (it is 2.1 GB). Pre-computed influence scores **are** included, \
so you can run all analysis and visualisation steps without the raw trajectory.\
"""),

        # ── §1 Introduction ──────────────────────────────────────────────────
        md("""\
---

## Introduction

### What is the `allostery` influence model?

The `allostery` package implements a **data-driven approach to identifying \
allosteric networks in proteins directly from molecular dynamics (MD) \
trajectories**, without requiring prior knowledge of the allosteric mechanism \
or hand-curated residue contacts.

#### The problem with traditional approaches

Classical allostery analysis (mutual information, linear correlation, \
dynamic network analysis) measures how much two residues move together. \
These methods are inherently **symmetric**: if residue A is correlated \
with residue B, B is equally correlated with A. They cannot distinguish \
which residue drives the other, and they conflate direct coupling with \
indirect correlation through a shared third party.

#### What the influence model learns

The influence model frames allostery as a **causal prediction problem**:

> *Can I predict residue i's acceleration at time t+1 from the current \
positions and velocities of all other residues?*

For each residue i, an attention-based message-passing network aggregates \
weighted messages from every other residue j and predicts i's acceleration. \
The **learned attention weight from j to i** is the raw influence score \
— it measures how much knowing j's current state helps predict i's future \
motion.

After training on sliding windows of the MD trajectory, the model scores \
all N×(N−1)/2 residue pairs. High-scoring pairs are edges in the \
**allosteric network**; residues that lie on many short paths through this \
network (high betweenness centrality) are **allosteric hubs**.

#### Key properties

| Property | Detail |
|----------|--------|
| **Asymmetric by design** | Influence of j→i ≠ influence of i→j; scores are then symmetrised |
| **Time-translation invariant** | Uses relative positions + finite differences, not absolute coordinates; any window of the trajectory contributes equally |
| **Timescale-tunable** | `window_size` sets the temporal scale: small windows capture fast local fluctuations, large windows capture slow collective motions |
| **No contact threshold** | Scores all pairs regardless of distance; the allosteric network emerges from the data |

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

1. How to install the package and configure your environment
2. How to configure and validate your trajectory input
3. How to train the influence model and score all residue pairs
4. How to analyse the resulting allosteric network (hub residues, pathways)
5. How different time-window sizes reveal different mechanistic layers
6. How to visualise hub residues, score distributions, and the network graph\
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
with open(scores_w5) as f:
    n_pairs = sum(1 for _ in f) - 1
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

rows = sorted(read_scores_csv(str(scores_w5)), key=lambda r: float(r["score"]), reverse=True)
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

        # ── §5d Visualise — figure 4: residue–residue influence matrix ──────
        md("### Residue–residue influence matrix"),

        code("""\
def build_influence_matrix(scores_csv: Path) -> tuple[np.ndarray, list[str]]:
    \"\"\"Build an N×N symmetric influence matrix from a scores CSV.\"\"\"
    with open(scores_csv) as f:
        rows = list(csv.DictReader(f))
    n = max(
        max(int(r["residue_i_index"]) for r in rows),
        max(int(r["residue_j_index"]) for r in rows),
    ) + 1
    matrix = np.zeros((n, n))
    res_nums: dict[int, str] = {}
    for r in rows:
        i, j = int(r["residue_i_index"]), int(r["residue_j_index"])
        s = float(r["score"])
        matrix[i, j] = s
        matrix[j, i] = s
        res_nums[i] = r["residue_i_number"]
        res_nums[j] = r["residue_j_number"]
    labels = [res_nums[k] for k in range(n)]
    return matrix, labels


matrix_w5, res_labels = build_influence_matrix(scores_w5)
n = len(res_labels)

# KRAS4B structural regions (residue numbers, 1-indexed)
REGIONS = {
    "P-loop":     (10,  17, "#3498db"),
    "Switch I":   (25,  40, "#e74c3c"),
    "Switch II":  (57,  76, "#e67e22"),
    "α3 helix":  (107, 126, "#9b59b6"),
}
# Convert residue numbers to 0-based matrix indices
res_to_idx = {int(v): k for k, v in enumerate(res_labels)}

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(matrix_w5, cmap="Reds", aspect="equal",
               interpolation="nearest", vmin=0)
plt.colorbar(im, ax=ax, label="Influence score", shrink=0.8)

# Axis ticks every 20 residues
tick_step = 20
ticks = list(range(0, n, tick_step))
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels([res_labels[t] for t in ticks], rotation=90, fontsize=8)
ax.set_yticklabels([res_labels[t] for t in ticks], fontsize=8)
ax.set_xlabel("Residue")
ax.set_ylabel("Residue")

# Annotate structural regions with coloured lines along both axes
for name, (start, end, color) in REGIONS.items():
    s_idx = res_to_idx.get(start, start - 1)
    e_idx = res_to_idx.get(end,   end   - 1)
    mid   = (s_idx + e_idx) / 2
    for axis in ("x", "y"):
        ax.axvline(s_idx - 0.5, color=color, linewidth=0.8, alpha=0.6) if axis == "x" else \
        ax.axhline(s_idx - 0.5, color=color, linewidth=0.8, alpha=0.6)
        ax.axvline(e_idx + 0.5, color=color, linewidth=0.8, alpha=0.6) if axis == "x" else \
        ax.axhline(e_idx + 0.5, color=color, linewidth=0.8, alpha=0.6)
    ax.text(mid, -4, name, ha="center", va="bottom", fontsize=6.5,
            color=color, fontweight="bold", rotation=0)

ax.set_title("Residue–residue influence matrix (window_size=5, 1 ns)", pad=20)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "influence_matrix_w5.png", dpi=150)
plt.show()
print(f"Saved: {FIGURES_DIR / 'influence_matrix_w5.png'}")
print(f"Matrix: {n}×{n}  |  score range: {matrix_w5.min():.4f} – {matrix_w5.max():.4f}")\
"""),

        # ── §6 Multi-timescale comparison ────────────────────────────────────
        md("""\
---
## Step 5: Multi-timescale comparison

The `window_size` parameter controls which **timescale of dynamics** the model \
captures. Eight window sizes were trained on KRAS WT, spanning 0.6 ns to 15 ns. \
All pre-computed score files are included in the repository.

| Config | Duration | Ind. samples | Dynamics captured |
|--------|----------|:------------:|-------------------|
| w3     | 0.6 ns   | ~1667        | Sub-ns backbone vibrations |
| w5     | 1 ns     | ~1000        | Fast backbone fluctuations, Switch I jitter |
| w10    | 2 ns     | ~500         | Early cooperative motions |
| w15    | 3 ns     | ~333         | Switch region dynamics |
| w25    | 5 ns     | ~200         | Slow collective motions, α3 helix breathing |
| w35    | 7 ns     | ~142         | Intermediate collective motions |
| w50    | 10 ns    | ~100         | Slow lobe motions (marginal statistics) |
| w75    | 15 ns    | ~66          | Very slow collective dynamics (marginal statistics) |

> **Statistical reliability:** each window needs ≥ 100 independent samples. \
For 5001 frames at 200 ps/frame, w50 (100 samples) is the practical limit; \
w75 (66 samples) should be interpreted with caution.\
"""),

        code("""\
# Known allosteric residues (red); terminal residues (grey = artefact candidates)
ALLOSTERIC = {"A:48 GLY", "A:121 PRO", "A:122 SER", "A:79 LEU"}
TERMINI    = {"A:169 LYS", "A:1 MET"}

windows = [
    ("w3\\n0.6 ns",  Path("outputs/kras_wt_w3/influence_scores.csv")),
    ("w5\\n1 ns",    Path("outputs/kras_wt/influence_scores.csv")),
    ("w10\\n2 ns",   Path("outputs/kras_wt_w10/influence_scores.csv")),
    ("w15\\n3 ns",   Path("outputs/kras_wt_w15/influence_scores.csv")),
    ("w25\\n5 ns",   Path("outputs/kras_wt_w25/influence_scores.csv")),
    ("w35\\n7 ns",   Path("outputs/kras_wt_w35/influence_scores.csv")),
    ("w50\\n10 ns",  Path("outputs/kras_wt_w50/influence_scores.csv")),
    ("w75\\n15 ns",  Path("outputs/kras_wt_w75/influence_scores.csv")),
]

fig, axes = plt.subplots(1, len(windows), figsize=(22, 6), sharey=False)

for ax, (label, path) in zip(axes, windows):
    assert path.exists(), f"Scores not found: {path}\\nRun: ./scripts/train.sh configs/kras_wt_{label.split(chr(10))[0]}.yaml"
    ranked = hub_centrality(path, top_k=30)
    residues = [h[0] for h in ranked]
    values   = [h[1] for h in ranked]
    colors = [
        "#c0392b" if r in ALLOSTERIC else
        "#95a5a6" if r in TERMINI    else
        "#2980b9"
        for r in residues
    ]
    ax.barh(residues[::-1], values[::-1], color=colors[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Centrality", fontsize=8)
    ax.set_title(label, fontsize=9, fontweight="bold")
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=7)

fig.suptitle(
    "Hub residues across timescales — KRAS WT GDP-bound\\n"
    "Red = known allosteric region  |  Grey = terminus (artefact candidate)  |  Blue = other",
    fontsize=11, fontweight="bold",
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "hub_comparison_all_windows.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {FIGURES_DIR / 'hub_comparison_all_windows.png'}")\
"""),

        md("""\
### Interpretation

| Timescale | Top hub | Biology |
|-----------|---------|---------|
| 1–3 ns (w5–w15) | **GLY48** (Switch I) | Fast hinge — backbone flexibility at the nucleotide-binding loop; the fastest allosteric signal carrier |
| 5 ns (w25) | **PRO121** (α3 helix) | Slow collective lobe motion; direct effector-interface coupling residue |
| 15 ns (w75) | **SER122** (α3 helix) | Neighbour of PRO121 — consistent with α3 helix as the slow-timescale hub |
| 0.6 ns (w3), 7 ns (w35) | **LYS169 / MET1** | C- and N-termini — likely free-terminal artefact, not true allosteric coupling |

**Implied pathway:** GLY48 (Switch I hinge) → PRO121/SER122 (α3 helix) → effector interface

The model identifies **two mechanistic layers**:
- A fast (1–3 ns) Switch I flexibility hub that rapidly samples conformational states
- A slow (5–15 ns) α3-helix hub that transmits the accumulated signal to the effector surface\
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
