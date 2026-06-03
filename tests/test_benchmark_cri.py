from __future__ import annotations

import csv
import json
from pathlib import Path

from allostery.benchmark import cri as cri_benchmark
from allostery.benchmark.cri import BenchmarkConfig, enforce_smoke_thresholds, run_benchmark, write_summary_files


def test_run_benchmark_on_synthetic_workload_produces_summary() -> None:
    config = BenchmarkConfig(
        pdb_path=None,
        synthetic=True,
        synthetic_frames=12,
        synthetic_residues=6,
        window_size=3,
        stride=1,
        time_step=1.0,
        distance_cutoff=20.0,
        max_neighbors=2,
        edge_types=2,
        hidden_dim=4,
        dropout=0.0,
        epochs=1,
        learning_rate=1e-3,
        entropy_weight=0.0,
        no_edge_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        batch_size=2,
        repeat=1,
        smoke_max_train_seconds=None,
        smoke_max_score_seconds=None,
        output_json=None,
        output_csv=None,
        seed=0,
    )

    summary = run_benchmark(config)

    assert summary['synthetic'] == 1
    assert summary['synthetic_frames'] == 12
    assert summary['synthetic_residues'] == 6
    assert summary['train_samples'] > 1
    assert summary['num_scores'] > 0
    assert summary['train_seconds_mean'] >= 0.0
    assert summary['score_seconds_mean'] >= 0.0
    assert isinstance(summary['git_sha'], str)
    assert summary['git_sha']


def test_write_summary_files_writes_json_and_csv(tmp_path: Path) -> None:
    summary = {
        'repeat': 1,
        'train_seconds_mean': 0.1,
        'train_seconds_min': 0.1,
        'score_seconds_mean': 0.2,
        'score_seconds_min': 0.2,
        'num_scores': 3,
        'synthetic': 1,
        'synthetic_frames': 12,
        'synthetic_residues': 6,
        'train_samples': 4,
        'validation_samples': 0,
        'batch_size': 2,
        'git_sha': 'fdc230a5304a3d9e8b8530f122bdc39def95e7cc',
    }
    json_path = tmp_path / 'benchmark.json'
    csv_path = tmp_path / 'benchmark.csv'

    write_summary_files(summary, json_path, csv_path)

    assert json.loads(json_path.read_text(encoding='utf-8')) == summary
    rows = list(csv.DictReader(csv_path.open(encoding='utf-8')))
    assert len(rows) == 1
    assert rows[0]['synthetic'] == '1'
    assert rows[0]['num_scores'] == '3'
    assert rows[0]['git_sha'] == 'fdc230a5304a3d9e8b8530f122bdc39def95e7cc'


def test_enforce_smoke_thresholds_rejects_slow_runs() -> None:
    summary = {
        'train_seconds_mean': 1.0,
        'score_seconds_mean': 0.01,
    }

    try:
        enforce_smoke_thresholds(summary, smoke_max_train_seconds=0.5, smoke_max_score_seconds=None)
    except SystemExit as exc:
        assert 'exceeded smoke threshold' in str(exc)
    else:
        raise AssertionError('expected smoke threshold failure')


def test_main_print_meta_only_skips_workload_and_writes_metadata(tmp_path: Path, capsys, monkeypatch) -> None:
    json_path = tmp_path / 'meta.json'
    csv_path = tmp_path / 'meta.csv'

    monkeypatch.setattr(cri_benchmark, '_current_git_sha', lambda: 'test-sha')

    def _unexpected_run(*args, **kwargs):
        raise AssertionError('run_benchmark should not be called in meta-only mode')

    monkeypatch.setattr(cri_benchmark, 'run_benchmark', _unexpected_run)

    exit_code = cri_benchmark.main(['--print-meta-only', '--output-json', str(json_path), '--output-csv', str(csv_path)])

    assert exit_code == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout['mode'] == 'meta'
    assert stdout['git_sha'] == 'test-sha'
    assert 'train_seconds_mean' not in stdout
    assert json.loads(json_path.read_text(encoding='utf-8')) == stdout
    rows = list(csv.DictReader(csv_path.open(encoding='utf-8')))
    assert len(rows) == 1
    assert rows[0]['mode'] == 'meta'
    assert rows[0]['git_sha'] == 'test-sha'
    assert rows[0]['synthetic'] == '0'
