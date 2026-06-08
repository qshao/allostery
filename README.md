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
allostery examples/influence_example_config.yaml
```

This trains the influence model on the bundled 3-frame fixture and writes a ranked pair-score CSV to `outputs/`.

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

## Run From YAML

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

Run it with:

```bash
allostery my_config.yaml
```

If running directly from the source tree without installing:

```bash
PYTHONPATH=src python -m allostery.cli my_config.yaml
```

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
- `model.hidden_dim` — network width (32–256 depending on protein size)
- `model.residue_layers` — encoder depth (2–4)
- `training.sparsity_weight` — entropy penalty for sparse network (0.001–0.01 typical)
- `training.device` — `cpu` or `cuda`

Full config reference and model description: [docs/tutorial.md](docs/tutorial.md).

## Outputs

- **Checkpoints** include model weights, architecture metadata, and a config snapshot.
- **Score CSVs** rank every residue pair by allosteric influence score and include directed columns `influence_i_on_j` and `influence_j_on_i`.
