# Allostery

C-alpha relational-network tooling for learning allosteric residue-residue interactions from MD trajectories.

The primary model (`family: influence`) builds a full directed residue-residue influence matrix via attention-based message passing. Every residue can influence every other — no spatial cutoff — and the learned attention weights form the allosteric network. See [docs/tutorial.md](docs/tutorial.md) for a full walkthrough.

## Quick install

```bash
git clone https://github.com/qshao/allostery.git
cd allostery
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
allostery --help
```

## Run the example

```bash
# Step 1 — train and score
allostery examples/influence_example_config.yaml

# Step 2 — analyze the allosteric network
allostery analyze outputs/influence_example_scores.csv --top-k 3

# Step 3 — find allosteric channels between two residues
allostery analyze outputs/influence_example_scores.csv \
  --top-k 3 --source "A:1 GLY" --sink "A:3 SER"
```

`allostery examples/...` trains the influence model on the bundled 3-frame fixture and writes a ranked pair-score CSV to `outputs/`. `allostery analyze` then builds the allosteric network and lists hub residues and communication channels.

## KRAS WT demo notebook

`examples/kras_wt/demo.ipynb` is a full end-to-end walkthrough on a real system: **KRAS4B GDP-bound wild type**, 169 residues, 1 µs MD trajectory. It covers preprocessing, training at eight window sizes (0.6–15 ns), hub-centrality comparison, score distributions, allosteric network graphs, and 169×169 residue–residue influence matrices across all timescales.

Open it with Jupyter:

```bash
pip install jupyter matplotlib networkx
jupyter notebook examples/kras_wt/demo.ipynb
```

The notebook is generated from `examples/kras_wt/build_notebook.py`; run that script to regenerate `demo.ipynb` after any edits to the build script.

## Virtual Environment

Create an isolated environment before installing dependencies or running training and inference:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## Install

Install PyTorch first (choose the variant for your hardware):

```bash
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.x
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then install this package:

```bash
pip install -e .
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
pytest -q
```

## Commands

The CLI has five commands and three global output flags:

```
allostery run <config.yaml>              # train / score / run pipeline from YAML config
allostery check <config.yaml>            # validate config without running anything
allostery analyze <scores.csv> [options] # post-process: network + channels
allostery interpret <scores.csv> [opts]  # candidate allosteric networks + optional LLM interpretation
allostery workflow <config.yaml>         # run -> analyze -> interpret end to end from one config
allostery validate [options]             # measure scorer accuracy vs. synthetic ground truth
allostery --version                      # print version and exit

# Global flags (before the subcommand):
#   --json    emit a single JSON object on stdout (for scripts)
#   --quiet   print only artifact paths
#   --debug   show full tracebacks instead of a clean error message

# Legacy short form (no subcommand) still works:
allostery my_config.yaml
```

Exit codes: `0` success, `1` user/input error, `2` usage error, `3` external/backend error.

### Interpret

Turn a scores CSV into candidate allosteric networks (communities, pathways, hubs,
coupled-pair clusters) plus an optional biological interpretation:

```bash
# Deterministic report only (no LLM)
allostery interpret outputs/scores.csv --pdb path/to/structure.pdb

# With a local Ollama model
allostery interpret outputs/scores.csv --llm ollama --llm-model qwen3

# With a cloud API (key from the environment)
ANTHROPIC_API_KEY=... allostery interpret outputs/scores.csv --llm anthropic
```

### Workflow

Run the whole pipeline from one config file. Add optional `analyze:` and `interpret:`
sections and `allostery workflow` runs train -> score -> analyze -> interpret:

```yaml
mode: run
# ... data / model / training / scoring / output as usual ...
analyze:
  top_k: 20
interpret:
  llm: none          # none | ollama | anthropic | openai
  top_hubs: 10
```

```bash
allostery workflow config.yaml
```

### Validate

Measure how well each scorer recovers *known* residue–residue coupling. The harness
generates synthetic trajectories from a planted coupling graph (exact ground truth),
runs classical baselines (DCCM, mutual information, contact frequency) alongside a
shuffled-trajectory null and the three model families, and reports ranking metrics
(ROC-AUC, PR-AUC, precision@k) averaged over seeds:

```bash
# Baselines only, fast
allostery validate --scorers dccm,mi,contact,null --seeds 5

# Full comparison including the trained model families
allostery validate --n-residues 24 --couplings 8 --seeds 3

# Machine-readable report for scripting / CI
allostery --json validate --out-json outputs/validation.json
```

The report flags whether each model family beats the best classical baseline — a method
is only meaningful if it outperforms trivial correlation on ground truth.

### Pipeline (run)

Use a single YAML file to control the whole pipeline:

```yaml
mode: run

data:
  pdb_path: path/to/trajectory.pdb
  window_size: 5
  horizon_size: 1
  stride: 1
  time_step: 1.0
  preprocess: align

model:
  family: influence
  hidden_dim: 64
  residue_layers: 2
  pair_layers: 1
  dropout: 0.1

training:
  epochs: 50
  learning_rate: 0.001
  consistency_weight: 0.0
  sparsity_weight: 0.01

scoring:
  top_k: 20

output:
  model_path: outputs/model.pt
  score_csv_path: outputs/scores.csv
```

```bash
allostery my_config.yaml
```

If running directly from the source tree without installing:

```bash
PYTHONPATH=src python -m allostery.cli my_config.yaml
```

### Network analysis (analyze)

After scoring, analyze the allosteric network from the scores CSV:

```bash
# Network summary + hub residues
allostery analyze outputs/scores.csv --top-k 20

# Add allosteric channel between two residues
allostery analyze outputs/scores.csv \
  --top-k 20 \
  --source "A:12 GLY" \
  --sink "A:87 SER" \
  --top-paths 5
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--top-k N` | 20 | Top-N pairs used as graph edges |
| `--source "CHAIN:NUM NAME"` | — | Source residue for channel search |
| `--sink "CHAIN:NUM NAME"` | — | Sink residue for channel search |
| `--top-paths N` | 5 | Shortest paths to list |
| `--top-hubs N` | 10 | Hub residues to list |

### Config validation (check)

Validate a config file without running the pipeline — useful to catch typos and missing files before a long training run:

```bash
allostery check my_config.yaml
```

Exits 0 with `Config OK: mode=run, family=influence` on success, or exits 1 and prints the validation errors to stderr.

## Model Families

| Family | Description |
|---|---|
| `influence` | Full N×N attention-based influence matrix; detects long-range allostery |
| `cri` | Sparse directed contact graph with latent interaction types |
| `relational` | Pairwise motion-feature encoder; baseline scoring |

## Config Reference

Top-level sections: `mode`, `data`, `model`, `training`, `scoring`, `output`.

Key parameters for the `influence` model:

- `data.window_size` — frames per training window (minimum 3)
- `data.preprocess` — `none`, `center`, or `align`
- `data.normalize` — remove each frame's centroid from position features for translation invariance (default `true`)
- `model.hidden_dim` — network width (32–256 depending on protein size)
- `model.residue_layers` — encoder depth (2–4)
- `model.residue_chunk_size` — tile the influence aggregation over receivers to bound peak memory on large proteins (default: unset = dense)
- `training.sparsity_weight` — entropy penalty for sparse network (0.001–0.01 typical)
- `training.device` — `cpu` or `cuda`
- `training.mixed_precision` — enable CUDA autocast/GradScaler (default `false`; no-op on CPU)
- `training.grad_clip_norm` — max gradient norm (default `1.0`; set to `null` to disable)
- `training.lr_scheduler` — `none` or `plateau` (default `plateau`)
- `training.deterministic` — set cuDNN deterministic flags for reproducible GPU runs (default `false`)

Full config reference and model description: [docs/tutorial.md](docs/tutorial.md).

## Outputs

- **Checkpoints** include model weights, architecture metadata, and a config snapshot.
- **Score CSVs** rank every residue pair by allosteric influence score and include directed columns `influence_i_on_j` and `influence_j_on_i`.
