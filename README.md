# Allostery

C-alpha relational-network tooling for residue-residue interaction scoring from multi-model PDB trajectories.

## Virtual Environment

Create an isolated environment before installing dependencies or running training and inference:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

## Install

```bash
pip install -e .
```

This installs the `allostery` console entry point declared in `pyproject.toml`.

## Run From YAML

Use a single YAML file to control the whole pipeline:

```yaml
mode: run

data:
  pdb_path: ../tests/fixtures/tiny_trajectory.pdb
  window_size: 1
  horizon_size: 1
  stride: 1

model:
  hidden_dim: 8
  residue_layers: 3
  pair_layers: 4
  dropout: 0.15

training:
  epochs: 1
  learning_rate: 0.001
  consistency_weight: 0.25

scoring:
  top_k: 5

output:
  model_path: ../outputs/example_model.pt
  score_csv_path: ../outputs/example_scores.csv
```

Run it with:

```bash
allostery examples/example_config.yaml
```

If you are running directly from the source tree without installing the package, use:

```bash
PYTHONPATH=src python -m allostery.cli examples/example_config.yaml
```

## Config Schema

Top-level sections:

- `mode`: `train`, `score`, or `run`
- `data`: `pdb_path`, `window_size`, `horizon_size`, `stride`
- `model`: `hidden_dim`, `residue_layers`, `pair_layers`, `dropout`
- `training`: `epochs`, `learning_rate`, `consistency_weight`
- `scoring`: `top_k`
- `output`: `model_path`, `score_csv_path`

Notes:

- Paths are resolved relative to the YAML file location.
- `training` is required for `train` and `run`.
- `scoring` is required for `score` and `run`.
- `model_path` is required whenever a checkpoint is needed.
- `score_csv_path` is required for `score` and `run`.

## Outputs

- Checkpoints include model weights, architecture metadata, and the serialized config snapshot.
- Score CSVs contain ranked residue-pair scores with residue metadata for downstream analysis.
