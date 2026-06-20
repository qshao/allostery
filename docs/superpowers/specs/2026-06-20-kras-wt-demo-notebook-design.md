# KRAS WT Allosteric Network Demo Notebook

## Goal

Produce `examples/kras_wt/demo.ipynb` — a self-contained Jupyter notebook that walks researchers through the full allostery pipeline using KRAS WT GDP-bound MD as a worked example, and gives them the template and guidance to run it on their own trajectory.

---

## Context

The repo already contains:
- `examples/` directory with `influence_example_config.yaml` and `train_and_score.py`
- `scripts/preprocess.sh`, `scripts/train.sh`, `scripts/analyze.sh`
- `configs/kras_wt_influence.yaml` (w5) and `configs/kras_wt_w25.yaml` (w25)
- Pre-computed scores at `outputs/kras_wt/influence_scores.csv` (w5) and `outputs/kras_wt_w25/influence_scores.csv` (w25)
- `networkx` and `matplotlib` installed; `nbformat` and `jupyter` are NOT installed by default

---

## Deliverables

### 1. `examples/kras_wt/demo.ipynb`

Built programmatically with `nbformat` (install at plan-time: `pip install nbformat`). A standard `.ipynb` JSON file that Jupyter can open. Researchers run `jupyter notebook examples/kras_wt/demo.ipynb` or open via JupyterLab.

### 2. `examples/kras_wt/template_config.yaml`

A template YAML config for researchers to fill in for their own system. All trajectory-specific values marked with `# <-- SET THIS`.

---

## Notebook Structure

### §1 Introduction (markdown cell)

- What allosteric networks are and why they matter
- The KRAS WT system: KRAS4B GDP-bound, 169 residues, 1 µs MD, 5001 frames at 200 ps/frame
- What this notebook demonstrates: pipeline walkthrough + multi-timescale comparison

### §2 Configuration (markdown + code cells)

**Markdown:** Explains every YAML field:
- `pdb_path` / `topology_path`: trajectory and topology files
- `window_size`: temporal scale (frames); `window_size × time_step` = window duration in ps
- `stride`: sampling interval (1 = maximum overlap, `window_size` = non-overlapping)
- `time_step`: physical time per frame in ps (must match MD output frequency)
- `top_k`: number of top-scoring pairs used as graph edges

**Code cell:** Validates the config using `allostery check`:
```python
import subprocess, sys
config = "configs/kras_wt_influence.yaml"   # <-- change to your config
result = subprocess.run(["allostery", "check", config], capture_output=True, text=True)
print(result.stdout or result.stderr)
```

Then calls `scripts/preprocess.sh` to show trajectory stats:
```python
result = subprocess.run(["./scripts/preprocess.sh", config], capture_output=True, text=True, cwd="../..")
print(result.stdout)
```

### §3 Train (markdown + code cells)

**Markdown heading:** `## Step 2: Train — skip this section if using pre-computed KRAS WT scores`

**Code cell** (marked with `# SKIP if using pre-computed scores`):
```python
result = subprocess.run(["./scripts/train.sh", config], capture_output=True, text=True, cwd="../..")
print(result.stdout)
```

**Code cell:** Verifies CSV exists:
```python
from pathlib import Path
scores_csv = Path("../../outputs/kras_wt/influence_scores.csv")
assert scores_csv.exists(), f"Scores not found: {scores_csv}"
print(f"Scores ready: {scores_csv}  ({scores_csv.stat().st_size // 1024} KB)")
```

### §4 Analyze — Single Window (markdown + code cells)

**Markdown:** Explains what the analysis produces (hub residues, betweenness centrality, threshold).

**Code cell:** Calls `run_network_analysis` directly (no subprocess):
```python
import sys
sys.path.insert(0, str(Path("../../src")))
from allostery.pipeline.analyze import run_network_analysis

report = run_network_analysis(scores_csv, top_k=30)
print(report)
```

### §5 Visualize (markdown + code cells)

Three matplotlib figures, each in its own code cell.

**Figure 1 — Hub residue centrality bar chart:**
- Horizontal bar chart, top-10 residues, bars colored by centrality value (white→red colormap)
- X-axis: betweenness centrality (0–1), Y-axis: residue labels (`A:48 GLY`)
- Title: `Hub Residues — window_size=5 (1 ns)`

**Figure 2 — Score distribution (log scale):**
- Histogram of all 14 196 pair scores (10 bins)
- Y-axis log scale to show the long tail of near-zero pairs
- Vertical dashed line at the suggested threshold score
- Annotated with `n=14196 pairs`, threshold value

**Figure 3 — Allosteric network graph:**
- `networkx` spring layout
- Nodes: residues (sized by betweenness centrality, minimum size 100)
- Edges: top-30 pairs (colored by score, white→red; width proportional to score)
- Node labels: residue number only (e.g. `48`)
- Colorbar for edge score

All figures saved to `examples/kras_wt/figures/` as PNG (`hub_centrality_w5.png`, `score_distribution_w5.png`, `network_graph_w5.png`).

### §6 Multi-Timescale Comparison (markdown + code cells)

**Markdown:** Explains why window size matters (fast local fluctuations vs slow collective motions), links to the scientific discussion from the session.

**Code cell:** Loads both score CSVs and runs analysis for each:
```python
configs_to_compare = [
    ("w5  (1 ns)",  "../../outputs/kras_wt/influence_scores.csv"),
    ("w25 (5 ns)",  "../../outputs/kras_wt_w25/influence_scores.csv"),
]
results = {}
for label, csv_path in configs_to_compare:
    report = run_network_analysis(csv_path, top_k=30)
    results[label] = report
```

**Figure 4 — Side-by-side hub centrality comparison:**
- Two subplots (1 row × 2 columns), same residue label set on Y-axis
- Left: w5 top-10 hubs; Right: w25 top-10 hubs
- Residues that appear in both highlighted in a distinct color
- Saved as `hub_comparison_w5_w25.png`

**Prose markdown cell** summarising the key finding:
- w5 (1 ns): GLY48 (Switch I) dominates — fast local backbone flexibility
- w25 (5 ns): PRO121 (α3 helix) becomes the hub — slow collective allosteric lobe motion
- Allosteric pathway implied: Switch I (GLY48) → α3 helix (PRO121) → effector interface

### §7 Next Steps (markdown cell)

- **Pathway analysis:** `./scripts/analyze.sh ... --source "A:48 GLY" --sink "A:121 PRO"`
- **PyMOL export:** `./scripts/analyze.sh ... --pdb structure.pdb --out-pml network.pml`
- **Choosing window_size for your system:** rule of thumb — start with 5–10× the autocorrelation time of the slowest motion of interest; compare 2–3 window sizes
- **More data = better slow dynamics:** 100+ independent windows per window_size recommended; for 10 ns windows, need ≥ 1 µs trajectory

---

## Template Config (`examples/kras_wt/template_config.yaml`)

```yaml
mode: run

data:
  pdb_path: /path/to/trajectory.trr    # <-- SET THIS (.trr, .xtc, .dcd, or .pdb)
  topology_path: /path/to/topology.gro # <-- SET THIS (.gro, .tpr, .psf; omit for .pdb)
  window_size: 25     # frames; window_size × time_step = window duration (ps)
  horizon_size: 1
  stride: 1
  time_step: 200.0    # <-- SET THIS: ps per saved frame (check your MD output frequency)
  preprocess: align   # align removes rigid-body rotation/translation (recommended)

model:
  family: influence
  hidden_dim: 64
  residue_layers: 2
  pair_layers: 1
  dropout: 0.1

training:
  epochs: 100
  learning_rate: 0.001
  sparsity_weight: 0.01
  consistency_weight: 0.0
  validation_fraction: 0.1
  patience: 20
  device: cuda        # <-- SET TO cpu if no GPU available

scoring:
  top_k: 30

output:
  model_path: outputs/my_protein/influence_model.pt      # <-- SET THIS
  score_csv_path: outputs/my_protein/influence_scores.csv # <-- SET THIS
```

---

## Global Constraints

- Notebook cells must run top-to-bottom without errors when pre-computed score CSVs are present
- All file paths inside the notebook are relative to `examples/kras_wt/` (notebook's own directory)
- `../../` prefix reaches the repo root from `examples/kras_wt/`
- No new runtime dependencies beyond `networkx` and `matplotlib` (already installed); `nbformat` needed only at build time
- Figures saved as PNG to `examples/kras_wt/figures/` (created if missing)
- No hardcoded absolute paths — all paths derived from `Path(__file__).parent` or equivalent notebook-relative logic
- `from __future__ import annotations` not required in notebook cells (not `.py` files)
