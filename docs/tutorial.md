# Allostery Tutorial

This tutorial walks you through installing the package, preparing your trajectory data, training the allosteric influence model, and interpreting the results.

---

## Table of Contents

1. [What this package does](#1-what-this-package-does)
2. [Requirements](#2-requirements)
3. [Installation](#3-installation)
4. [Preparing your trajectory](#4-preparing-your-trajectory)
5. [Quick start](#5-quick-start)
6. [Understanding the config file](#6-understanding-the-config-file)
7. [Model families](#7-model-families)
8. [Running in three modes](#8-running-in-three-modes)
9. [Interpreting the output](#9-interpreting-the-output)
10. [Allosteric network analysis](#10-allosteric-network-analysis)
11. [Tuning the model](#11-tuning-the-model)
12. [Python API](#12-python-api)

---

## 1. What this package does

`allostery` learns which residues dynamically influence which other residues directly from a molecular dynamics (MD) trajectory. It does this by training a neural network to predict each residue's acceleration from the motions of all other residues. The learned residue-residue influence matrix is the allosteric network.

The primary model (`family: influence`) builds a full N×N directed influence matrix via attention-based message passing — every residue can potentially influence every other, and the model learns to concentrate attention on the pairs that actually matter for explaining the observed dynamics. No spatial contact cutoff is imposed, so long-range allosteric communication is discoverable.

---

## 2. Requirements

- Python 3.11 or newer
- [PyTorch](https://pytorch.org/) (CPU or CUDA)
- NumPy
- PyYAML

GPU is optional. All examples below run on CPU. If you have a CUDA-capable GPU, set `training.device: cuda` in your config.

---

## 3. Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/qshao/allostery.git
cd allostery
```

### Step 2 — Create a virtual environment

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

You should now see `(.venv)` at the start of your shell prompt.

### Step 3 — Install PyTorch

Install PyTorch before the package so you can choose the right variant for your hardware. Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) to get the exact command for your OS and CUDA version. Common examples:

```bash
# CPU only (works everywhere)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.x (Linux/Windows with an NVIDIA GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Step 4 — Install the package

```bash
pip install -e .
```

The `-e` flag installs in editable mode so edits to the source are reflected immediately without reinstalling.

To also install the developer tools (pytest):

```bash
pip install -e ".[dev]"
```

### Step 5 — Verify the installation

```bash
allostery --help
```

Expected output:

```
usage: allostery [-h] {run,analyze} ...

positional arguments:
  {run,analyze}
    run          Run training/scoring pipeline from config YAML
    analyze      Analyze allosteric network from scores CSV
```

Run the bundled tests to confirm everything works:

```bash
pytest -q
```

All tests should pass.

---

## 4. Preparing your trajectory

The package reads **multi-model PDB files**, where each MODEL/ENDMDL block is one frame. This is the standard format produced by most MD analysis tools.

### Converting from common MD formats

**GROMACS** (using MDAnalysis):

```python
import MDAnalysis as mda

u = mda.Universe("topol.tpr", "traj.xtc")
ca = u.select_atoms("name CA")

with mda.Writer("trajectory.pdb", ca.n_atoms) as w:
    for ts in u.trajectory:
        w.write(ca)
```

**AMBER / OpenMM** (using MDTraj):

```python
import mdtraj as md

traj = md.load("trajectory.dcd", top="topology.pdb")
ca = traj.atom_slice(traj.topology.select("name CA"))
ca.save_pdb("trajectory.pdb")
```

**CHARMM / NAMD** (using VMD):

```tcl
mol load psf protein.psf dcd trajectory.dcd
set ca [atomselect top "name CA"]
animate write pdb trajectory.pdb sel $ca waitfor all
```

### Recommended preprocessing

The model works with raw Cartesian coordinates but performs better when the trajectory is preprocessed:

1. **Remove solvent and ions** — keep only protein C-alpha atoms.
2. **Align to a reference frame** — remove overall rotation and translation. You can use the built-in `preprocess: align` option (see [Section 6](#6-understanding-the-config-file)) or align externally before saving.
3. **Wrap periodic boundary conditions** — if your simulation uses PBC, unwrap or image atoms so residues do not jump across box boundaries between frames.
4. **Ensure sequential residue numbering** — residues are identified by their order in the PDB, not by residue number.

### What the fixture looks like

A small 3-frame, 3-residue example lives at `tests/fixtures/tiny_trajectory.pdb`. Open it to see the expected format:

```
MODEL        1
ATOM      1  CA  GLY A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  ALA A   2       3.800   0.000   0.000  1.00  0.00           C
ATOM      3  CA  SER A   3       7.600   0.000   0.000  1.00  0.00           C
ENDMDL
MODEL        2
...
ENDMDL
```

Only the `CA` atom lines are used; all other ATOM records are ignored.

---

## 5. Quick start

Run the full end-to-end pipeline on the bundled 3-residue, 3-frame fixture:

**Step 1 — Train and score:**

```bash
allostery examples/influence_example_config.yaml
```

Output:

```
epoch 1/5  train=5.3806
epoch 2/5  train=4.8756
epoch 3/5  train=4.3966
epoch 4/5  train=3.9450
epoch 5/5  train=3.5217
trained samples=1 checkpoint=outputs/influence_example_model.pt
scored pairs=3 csv=outputs/influence_example_scores.csv top_k=10
completed mode=run
```

The scores CSV lists every residue pair ranked by learned allosteric influence:

```
rank,score,...,influence_i_on_j,influence_j_on_i
1,0.508,1,A,2,ALA,2,A,3,SER,1,0.503,0.513
2,0.503,0,A,1,GLY,2,A,3,SER,1,0.497,0.510
3,0.489,0,A,1,GLY,1,A,2,ALA,1,0.487,0.490
```

**Step 2 — Analyze the allosteric network:**

```bash
allostery analyze outputs/influence_example_scores.csv --top-k 3
```

Output:

```
=== Allosteric Network ===
Residues (nodes):       3
Edges (scored pairs):   3
Connected components:   1

=== Hub Residues (Top 10 by Betweenness Centrality) ===
   1.  A:2 ALA               0.0000
   2.  A:3 SER               0.0000
   3.  A:1 GLY               0.0000
```

**Step 3 — Find allosteric channels:**

```bash
allostery analyze outputs/influence_example_scores.csv \
  --top-k 3 --source "A:1 GLY" --sink "A:3 SER" --top-paths 3
```

Output:

```
=== Allosteric Network ===
Residues (nodes):       3
Edges (scored pairs):   3
Connected components:   1

=== Hub Residues (Top 10 by Betweenness Centrality) ===
   1.  A:2 ALA               0.0000
   2.  A:3 SER               0.0000
   3.  A:1 GLY               0.0000

=== Allosteric Channel: A:1 GLY → A:3 SER ===
  Path 1 (hops 1):  A:1 GLY → A:3 SER   (dist 1.987)
  Path 2 (hops 2):  A:1 GLY → A:2 ALA → A:3 SER   (dist 4.015)
```

For a real protein, replace `pdb_path` with your own trajectory and adjust the hyperparameters as described in [Section 11](#11-tuning-the-model).

---

## 6. Understanding the config file

All pipeline settings live in a single YAML file. Here is an annotated example:

```yaml
mode: run           # train | score | run (train then score)

data:
  pdb_path: path/to/trajectory.pdb   # multi-model PDB trajectory
  window_size: 5      # frames per training window (minimum 3)
  horizon_size: 1     # required but unused by the influence model; set to 1
  stride: 1           # step between consecutive windows
  time_step: 1.0      # time between frames in your trajectory (any consistent unit)
  preprocess: align   # none | center | align — applied before feature extraction

model:
  family: influence   # relational | cri | influence
  hidden_dim: 64      # width of all hidden layers
  residue_layers: 2   # depth of the per-residue encoder MLP
  pair_layers: 1      # unused by influence model; set to 1
  dropout: 0.1        # dropout fraction (0.0 to disable)

training:
  epochs: 50
  learning_rate: 0.001
  consistency_weight: 0.0   # unused by influence model; set to 0
  sparsity_weight: 0.01     # entropy penalty — higher = sparser influence network
  validation_fraction: 0.2  # fraction of windows held out for early stopping
  patience: 10              # stop after this many epochs with no val improvement
  seed: 42
  device: cpu               # cpu | cuda | cuda:0 | cuda:1
  batch_size: 8

scoring:
  top_k: 20           # printed in the completion message; all pairs are written to CSV

output:
  model_path: outputs/model.pt
  score_csv_path: outputs/scores.csv
```

**Path resolution:** all paths in the config are resolved relative to the config file's directory, not the working directory. This means you can run `allostery configs/my_run.yaml` from any directory and paths will resolve correctly.

---

## 7. Model families

The package ships three model families selectable via `model.family`:

| Family | How it works | Best for |
|---|---|---|
| `influence` | Full N×N attention-based influence matrix; no spatial cutoff | Detecting long-range allostery |
| `cri` | Sparse directed contact graph + latent interaction types | When spatial locality is expected |
| `relational` | Pairwise motion-feature encoder; no explicit influence | Baseline / exploratory scoring |

For allosteric network detection the `influence` model is recommended because it does not pre-filter pairs by distance.

---

## 8. Running in three modes

### `mode: train`

Trains the model and saves a checkpoint. Does not produce a score CSV.

```yaml
mode: train
output:
  model_path: outputs/model.pt
  score_csv_path: ~    # not required for train-only mode
```

```bash
allostery configs/train.yaml
```

### `mode: score`

Loads a previously saved checkpoint and scores the trajectory. Requires `model_path` to point to an existing checkpoint.

```yaml
mode: score
output:
  model_path: outputs/model.pt        # must already exist
  score_csv_path: outputs/scores.csv
```

```bash
allostery configs/score.yaml
```

### `mode: run`

Trains then immediately scores. Equivalent to running `train` then `score` with the same checkpoint.

```yaml
mode: run
```

```bash
allostery configs/run.yaml
```

---

## 9. Interpreting the output

### Score CSV columns

| Column | Description |
|---|---|
| `rank` | 1-based rank (1 = strongest inferred allosteric coupling) |
| `score` | Mean of `influence_i_on_j` and `influence_j_on_i` — the undirected coupling strength |
| `residue_i_index` | 0-based residue index in the trajectory |
| `residue_i_chain` | Chain ID from the PDB |
| `residue_i_number` | Residue sequence number from the PDB |
| `residue_i_name` | Three-letter residue name |
| `residue_j_*` | Same fields for the second residue of the pair |
| `influence_i_on_j` | Directed: mean attention weight `A[j, i]` — how strongly i drives j |
| `influence_j_on_i` | Directed: mean attention weight `A[i, j]` — how strongly j drives i |
| `support_count` | Number of trajectory windows that contributed to the score |

### Reading the influence matrix

The influence model outputs a full N×N matrix per trajectory window. Each row `j` is a probability distribution over all other residues: `A[j, i]` is the fraction of residue j's acceleration explained by residue i. A high `A[j, i]` means residue i is the dominant driver of residue j's motion in the learned dynamics.

Because the matrix is directed, you can distinguish:

- **Allosteric source**: a residue with high outgoing influence (large column sums) — it drives many others
- **Allosteric sink**: a residue with high incoming influence (large row entropy close to uniform after training means it has no dominant driver; a peaked row means one residue dominates)
- **Bidirectional coupling**: both `A[j, i]` and `A[i, j]` are large

### What score threshold to use

After a well-trained run, the influence distribution becomes sparse — a small number of pairs have noticeably higher scores than the background. A practical approach:

1. Plot the score distribution (scores are in the CSV, load them with pandas or numpy).
2. Look for a gap or elbow in the sorted score curve — pairs above the gap are the inferred allosteric network.
3. Cross-validate against known allosteric sites or perturbation experiments if available.

---

## 10. Allosteric network analysis

Once you have a scores CSV from any pipeline run, use `allostery analyze` to extract the allosteric network and communication channels. This step requires no model checkpoint and works on any scores CSV produced by the `influence`, `cri`, or `relational` pipeline.

### Basic network summary

```bash
allostery analyze outputs/scores.csv --top-k 20
```

This builds a weighted undirected graph from the top-20 scoring residue pairs and reports:

- **Residues (nodes):** how many unique residues appear in the top-k pairs
- **Edges:** number of graph edges (= `--top-k`)
- **Connected components:** number of disconnected sub-networks (1 = fully connected)
- **Hub residues:** ranked by betweenness centrality — residues that lie on the most shortest paths through the network. High centrality means the residue is a key relay in allosteric communication.

### Finding allosteric channels

Specify a source and sink residue to find the shortest communication paths:

```bash
allostery analyze outputs/scores.csv \
  --top-k 30 \
  --source "A:12 GLY" \
  --sink "A:87 SER" \
  --top-paths 5
```

Residue labels use the format `CHAIN:NUMBER NAME` exactly as they appear in the `residue_i_chain`, `residue_i_number`, `residue_i_name` columns of the CSV. For example, chain A, residue 12, glycine → `"A:12 GLY"`.

Path distance is defined as the sum of `1/score` along the path: higher-scored edges are treated as shorter. Path 1 is always the most direct (highest-scored) route.

### Choosing `--top-k`

`--top-k` controls how many scored pairs are included as graph edges. Too few and the source/sink may not be connected; too many and the network becomes dense and less informative. A practical starting range is 20–50 pairs. If `allostery analyze` reports "No path found", increase `--top-k`.

### Using the analyze command from Python

```python
from allostery.network import build_graph, format_report, read_scores_csv

rows = read_scores_csv("outputs/scores.csv")
net = build_graph(rows, top_k=30)
report = format_report(
    net,
    source_label="A:12 GLY",
    sink_label="A:87 SER",
    top_hubs=10,
    top_paths=5,
)
print(report)
```

You can also access the underlying graph objects directly:

```python
from allostery.network import betweenness_centrality, shortest_paths

centrality = betweenness_centrality(net)
# centrality[i] = normalized betweenness for node i
for i, score in sorted(centrality.items(), key=lambda kv: -kv[1])[:5]:
    print(f"{net.node_labels[i]}: {score:.4f}")

paths = shortest_paths(net, "A:12 GLY", "A:87 SER", top_n=3)
for path_labels, dist in paths:
    print(" → ".join(path_labels), f"(dist {dist:.3f})")
```

---

## 11. Tuning the model

### Trajectory preprocessing

| `preprocess` value | Effect |
|---|---|
| `none` | Raw coordinates (default) |
| `center` | Subtract the mean position each frame |
| `align` | Least-squares align each frame to the first frame of the window |

For trajectories with significant overall protein motion, `align` usually gives the best results because it removes rigid-body motion before computing velocities and accelerations.

### `window_size`

Controls how many frames are stacked into a single training sample. A window of size `W` yields `W - 2` internal frames with central-difference velocities and accelerations. Larger windows capture slower motions; smaller windows are faster to train.

- Minimum: 3 (one internal frame)
- Typical range: 5–20 for ns-scale MD with ps-scale output
- If you have very long trajectories, increase `stride` to avoid redundant windows rather than reducing `window_size`

### `sparsity_weight`

Controls how peaked the influence distribution becomes. Higher values push each residue's influence row toward a single dominant sender, producing a sparse network. Lower values let the influence distribute more evenly.

- `0.0`: no sparsity regularization — the network may be dense
- `0.001–0.01`: mild sparsity — recommended starting range
- `0.1+`: strong sparsity — use if the network seems too diffuse

### `hidden_dim` and `residue_layers`

`hidden_dim` scales model capacity. For a protein with N residues:

| N | Suggested `hidden_dim` | Suggested `residue_layers` |
|---|---|---|
| < 100 | 32–64 | 2 |
| 100–300 | 64–128 | 2–3 |
| 300–500 | 128–256 | 3 |
| > 500 | 256+ | 3–4 |

### `epochs` and early stopping

With `validation_fraction > 0` and `patience > 0`, training stops automatically when validation loss stops improving. Set `epochs` to a large ceiling (e.g., 200) and let early stopping decide when to halt. If you set `validation_fraction: 0.0`, early stopping is disabled and the model trains for exactly `epochs` epochs.

### GPU training

Set `device: cuda` to use the first available GPU:

```yaml
training:
  device: cuda
  batch_size: 32   # increase batch size when GPU memory allows
```

Check available devices with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

---

## 12. Python API

You can drive the full pipeline from Python without a YAML config.

### Training

```python
from allostery.pipeline.influence_train import train_influence_model

result = train_influence_model(
    pdb_path="trajectory.pdb",
    window_size=5,
    stride=1,
    time_step=1.0,
    hidden_dim=64,
    num_encoder_layers=2,
    dropout=0.1,
    epochs=100,
    learning_rate=1e-3,
    sparsity_weight=0.01,
    preprocess="align",
    validation_fraction=0.2,
    patience=10,
    seed=42,
    device="cpu",
    batch_size=8,
    checkpoint_path="outputs/model.pt",   # omit to skip saving
)

print(f"Trained on {result.num_samples} windows")
print(f"Best epoch: {result.best_epoch}, val loss: {result.best_validation_loss:.4f}")
```

### Scoring

```python
from allostery.pipeline.influence_score import score_influence_trajectory

scores = score_influence_trajectory(
    model=result.model,
    pdb_path="trajectory.pdb",
    window_size=5,
    stride=1,
    time_step=1.0,
    preprocess="align",
)

print("Top 5 allosteric pairs:")
for pair in scores[:5]:
    ri = pair["residue_i"]
    rj = pair["residue_j"]
    print(
        f"  {ri['name']}{ri['residue_number']} — {rj['name']}{rj['residue_number']}: "
        f"score={pair['score']:.4f}  "
        f"({ri['name']}{ri['residue_number']}→{rj['name']}{rj['residue_number']} = {pair['influence_i_on_j']:.4f}, "
        f"{rj['name']}{rj['residue_number']}→{ri['name']}{ri['residue_number']} = {pair['influence_j_on_i']:.4f})"
    )
```

### Loading a saved checkpoint

```python
from allostery.pipeline.score import load_scoring_model
from allostery.pipeline.influence_score import score_influence_trajectory

model = load_scoring_model("outputs/model.pt")

scores = score_influence_trajectory(
    model=model,
    pdb_path="new_trajectory.pdb",
    window_size=5,
    stride=1,
    time_step=1.0,
)
```

### Accessing the influence matrix directly

```python
import torch
from allostery.models.influence import AllostericInfluenceModel
from allostery.influence.data import build_influence_samples
from allostery.io.pdb import load_multimodel_pdb

traj = load_multimodel_pdb("trajectory.pdb")
samples = build_influence_samples(traj.coordinates, window_size=5, stride=1, time_step=1.0)

model = AllostericInfluenceModel(state_dim=6, hidden_dim=64, num_encoder_layers=2)
# ... load trained weights if needed ...

model.eval()
with torch.no_grad():
    matrices = []
    for sample in samples:
        state = torch.as_tensor(sample.state_features[None, ...], dtype=torch.float32)
        out = model(state)
        matrices.append(out["influence_matrix"].squeeze(0))  # [N, N]

# Mean influence matrix across all windows
mean_matrix = torch.stack(matrices).mean(dim=0)  # [N, N]
# mean_matrix[j, i] = average influence of residue i on residue j
```

---

---

## Appendix: Config reference

| Key | Type | Default | Description |
|---|---|---|---|
| `mode` | string | — | `train`, `score`, or `run` |
| `data.pdb_path` | path | — | Multi-model PDB trajectory |
| `data.window_size` | int | — | Frames per window (≥ 3) |
| `data.horizon_size` | int | — | Unused by influence/cri; set to 1 |
| `data.stride` | int | — | Step between windows (≥ 1) |
| `data.time_step` | float | 1.0 | Time between frames |
| `data.preprocess` | string | `none` | `none`, `center`, or `align` |
| `data.distance_cutoff` | float | 20.0 | CRI only: neighbor cutoff in Å |
| `data.max_neighbors` | int | 2 | CRI only: max directed neighbors |
| `data.min_sequence_separation` | int | 0 | CRI only: min sequence gap for edges |
| `model.family` | string | `relational` | `relational`, `cri`, or `influence` |
| `model.hidden_dim` | int | — | Hidden layer width |
| `model.residue_layers` | int | — | Per-residue encoder depth |
| `model.pair_layers` | int | — | Pair encoder depth (relational/cri) |
| `model.dropout` | float | — | Dropout rate [0, 1) |
| `model.edge_types` | int | — | CRI only: number of latent edge types (≥ 2) |
| `training.epochs` | int | — | Max training epochs |
| `training.learning_rate` | float | — | Adam learning rate |
| `training.consistency_weight` | float | 0.0 | Relational only |
| `training.entropy_weight` | float | 0.0 | CRI only |
| `training.no_edge_weight` | float | 0.0 | CRI only |
| `training.sparsity_weight` | float | 0.0 | Influence only: row-entropy penalty |
| `training.validation_fraction` | float | 0.2 | Fraction of windows for validation |
| `training.patience` | int | 5 | Early-stopping patience (0 = disabled) |
| `training.seed` | int | 0 | Random seed |
| `training.device` | string | `cpu` | PyTorch device string |
| `training.batch_size` | int | 4 | Samples per gradient update |
| `scoring.top_k` | int | — | Printed in completion message |
| `output.model_path` | path | — | Checkpoint save/load path |
| `output.score_csv_path` | path | — | CSV output path |
