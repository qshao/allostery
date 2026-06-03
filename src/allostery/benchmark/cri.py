
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from allostery.pipeline.cri_score import score_cri_trajectory
from allostery.pipeline.cri_train import train_cri_model


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    pdb_path: Path | None
    synthetic: bool
    synthetic_frames: int
    synthetic_residues: int
    window_size: int
    stride: int
    time_step: float
    distance_cutoff: float
    max_neighbors: int
    edge_types: int
    hidden_dim: int
    dropout: float
    epochs: int
    learning_rate: float
    entropy_weight: float
    no_edge_weight: float
    validation_fraction: float
    patience: int
    batch_size: int
    repeat: int
    smoke_max_train_seconds: float | None
    smoke_max_score_seconds: float | None
    output_json: Path | None
    output_csv: Path | None
    seed: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Benchmark the CRI training and scoring path.')
    parser.add_argument('pdb_path', nargs='?', type=Path)
    parser.add_argument('--synthetic', action='store_true', help='Generate a synthetic multi-model PDB benchmark input.')
    parser.add_argument('--synthetic-frames', type=int, default=96)
    parser.add_argument('--synthetic-residues', type=int, default=32)
    parser.add_argument('--window-size', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--time-step', type=float, default=1.0)
    parser.add_argument('--distance-cutoff', type=float, default=20.0)
    parser.add_argument('--max-neighbors', type=int, default=2)
    parser.add_argument('--edge-types', type=int, default=2)
    parser.add_argument('--hidden-dim', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--entropy-weight', type=float, default=0.0)
    parser.add_argument('--no-edge-weight', type=float, default=0.0)
    parser.add_argument('--validation-fraction', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--smoke-max-train-seconds', type=float, default=1.0, help='Fail if mean train time exceeds this many seconds.')
    parser.add_argument('--smoke-max-score-seconds', type=float, default=1.0, help='Fail if mean score time exceeds this many seconds.')
    parser.add_argument('--output-json', type=Path)
    parser.add_argument('--output-csv', type=Path)
    parser.add_argument('--print-meta-only', action='store_true', help='Print benchmark metadata without running the workload.')
    return parser


def _build_benchmark_input(config: BenchmarkConfig) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if config.synthetic:
        tempdir = tempfile.TemporaryDirectory()
        pdb_path = Path(tempdir.name) / 'synthetic_cri_benchmark.pdb'
        _write_synthetic_pdb(pdb_path, config.synthetic_frames, config.synthetic_residues, config.seed)
        return pdb_path, tempdir
    if config.pdb_path is None:
        raise ValueError('pdb_path is required unless --synthetic is set')
    return config.pdb_path, None


def _current_git_sha() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ['git', '-C', str(repo_root), 'rev-parse', 'HEAD'],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 'unknown'
    return result.stdout.strip() or 'unknown'


def _benchmark_config_summary(config: BenchmarkConfig) -> dict[str, object]:
    return {
        'pdb_path': str(config.pdb_path) if config.pdb_path is not None else None,
        'synthetic': int(config.synthetic),
        'synthetic_frames': config.synthetic_frames,
        'synthetic_residues': config.synthetic_residues,
        'window_size': config.window_size,
        'stride': config.stride,
        'time_step': config.time_step,
        'distance_cutoff': config.distance_cutoff,
        'max_neighbors': config.max_neighbors,
        'edge_types': config.edge_types,
        'hidden_dim': config.hidden_dim,
        'dropout': config.dropout,
        'epochs': config.epochs,
        'learning_rate': config.learning_rate,
        'entropy_weight': config.entropy_weight,
        'no_edge_weight': config.no_edge_weight,
        'validation_fraction': config.validation_fraction,
        'patience': config.patience,
        'batch_size': config.batch_size,
        'repeat': config.repeat,
        'smoke_max_train_seconds': config.smoke_max_train_seconds,
        'smoke_max_score_seconds': config.smoke_max_score_seconds,
        'output_json': str(config.output_json) if config.output_json is not None else None,
        'output_csv': str(config.output_csv) if config.output_csv is not None else None,
        'seed': config.seed,
    }


def _write_synthetic_pdb(path: Path, num_frames: int, num_residues: int, seed: int) -> None:
    residues = ['GLY', 'ALA', 'SER', 'THR', 'LEU', 'VAL', 'ASP', 'ASN', 'GLU', 'GLN']
    lines: list[str] = []
    atom_serial = 1
    for frame_index in range(num_frames):
        lines.append(f'MODEL{frame_index + 1:>9}')
        frame_phase = (2.0 * math.pi * frame_index) / max(num_frames, 1)
        for residue_index in range(num_residues):
            residue_name = residues[residue_index % len(residues)]
            base_x = 10.0 + residue_index * 1.45
            base_y = 10.0 + math.sin(residue_index * 0.5)
            base_z = 10.0 + math.cos(residue_index * 0.5)
            oscillation = 0.15 * math.sin(frame_phase + residue_index * 0.35 + seed * 0.01)
            x = base_x + oscillation
            y = base_y + 0.5 * oscillation
            z = base_z - 0.3 * oscillation
            lines.append(
                f'ATOM  {atom_serial:5d}  CA  {residue_name:>3s} A{residue_index + 1:4d}'
                f'{x:11.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C'
            )
            atom_serial += 1
        lines.append('ENDMDL')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_benchmark(config: BenchmarkConfig) -> dict[str, object]:
    pdb_path, tempdir = _build_benchmark_input(config)
    git_sha = _current_git_sha()
    train_times: list[float] = []
    score_times: list[float] = []
    scores: list[dict[str, object]] = []
    last_train_result = None
    try:
        for repeat_index in range(config.repeat):
            train_start = time.perf_counter()
            train_result = train_cri_model(
                pdb_path=pdb_path,
                window_size=config.window_size,
                stride=config.stride,
                time_step=config.time_step,
                distance_cutoff=config.distance_cutoff,
                max_neighbors=config.max_neighbors,
                edge_types=config.edge_types,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                entropy_weight=config.entropy_weight,
                no_edge_weight=config.no_edge_weight,
                validation_fraction=config.validation_fraction,
                patience=config.patience,
                batch_size=config.batch_size,
                seed=config.seed + repeat_index,
            )
            train_times.append(time.perf_counter() - train_start)
            last_train_result = train_result

            score_start = time.perf_counter()
            scores = score_cri_trajectory(
                model=train_result.model,
                pdb_path=pdb_path,
                window_size=config.window_size,
                stride=config.stride,
                time_step=config.time_step,
                distance_cutoff=config.distance_cutoff,
                max_neighbors=config.max_neighbors,
            )
            score_times.append(time.perf_counter() - score_start)
    finally:
        if tempdir is not None:
            tempdir.cleanup()

    summary: dict[str, object] = {
        'repeat': config.repeat,
        'train_seconds_mean': sum(train_times) / len(train_times),
        'train_seconds_min': min(train_times),
        'score_seconds_mean': sum(score_times) / len(score_times),
        'score_seconds_min': min(score_times),
        'num_scores': len(scores),
        'synthetic': int(config.synthetic),
        'synthetic_frames': config.synthetic_frames,
        'synthetic_residues': config.synthetic_residues,
        'git_sha': git_sha,
    }
    if last_train_result is not None:
        summary['train_samples'] = last_train_result.train_samples
        summary['validation_samples'] = last_train_result.validation_samples
        summary['batch_size'] = last_train_result.batch_size
    return summary


def write_summary_files(summary: dict[str, object], output_json: Path | None, output_csv: Path | None) -> None:
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)


def enforce_smoke_thresholds(
    summary: dict[str, object],
    smoke_max_train_seconds: float | None,
    smoke_max_score_seconds: float | None,
) -> None:
    if smoke_max_train_seconds is not None and float(summary['train_seconds_mean']) > smoke_max_train_seconds:
        raise SystemExit(
            f"CRI benchmark train mean {summary['train_seconds_mean']:.6f}s exceeded smoke threshold {smoke_max_train_seconds:.6f}s"
        )
    if smoke_max_score_seconds is not None and float(summary['score_seconds_mean']) > smoke_max_score_seconds:
        raise SystemExit(
            f"CRI benchmark score mean {summary['score_seconds_mean']:.6f}s exceeded smoke threshold {smoke_max_score_seconds:.6f}s"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BenchmarkConfig(
        pdb_path=args.pdb_path,
        synthetic=bool(args.synthetic),
        synthetic_frames=int(args.synthetic_frames),
        synthetic_residues=int(args.synthetic_residues),
        window_size=int(args.window_size),
        stride=int(args.stride),
        time_step=float(args.time_step),
        distance_cutoff=float(args.distance_cutoff),
        max_neighbors=int(args.max_neighbors),
        edge_types=int(args.edge_types),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        entropy_weight=float(args.entropy_weight),
        no_edge_weight=float(args.no_edge_weight),
        validation_fraction=float(args.validation_fraction),
        patience=int(args.patience),
        batch_size=int(args.batch_size),
        repeat=int(args.repeat),
        smoke_max_train_seconds=args.smoke_max_train_seconds,
        smoke_max_score_seconds=args.smoke_max_score_seconds,
        output_json=args.output_json,
        output_csv=args.output_csv,
        seed=int(args.seed),
    )
    if args.print_meta_only:
        meta = {'mode': 'meta', 'git_sha': _current_git_sha(), **_benchmark_config_summary(config)}
        write_summary_files(meta, config.output_json, config.output_csv)
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0
    summary = run_benchmark(config)
    enforce_smoke_thresholds(summary, config.smoke_max_train_seconds, config.smoke_max_score_seconds)
    write_summary_files(summary, config.output_json, config.output_csv)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
