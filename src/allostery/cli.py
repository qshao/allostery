from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

from allostery import __version__
from allostery.config import AppConfig, load_config
from allostery.io import write_pair_scores_csv
from allostery.pipeline.analyze import run_network_analysis
from allostery.pipeline.cri_score import score_cri_trajectory
from allostery.pipeline.cri_train import train_cri_model
from allostery.pipeline.influence_score import score_influence_trajectory
from allostery.pipeline.influence_train import train_influence_model
from allostery.pipeline.interpret import run_interpretation
from allostery.io.checkpoint import load_checkpoint
from allostery.pipeline.score import build_scoring_model, load_scoring_model, score_trajectory
from allostery.pipeline.train import TrainResult, train_model


_SUBCOMMANDS = frozenset({'run', 'analyze', 'check', 'interpret'})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='allostery')
    parser.add_argument('--version', action='version', version=f'allostery {__version__}')
    subparsers = parser.add_subparsers(dest='command')

    # Default pipeline command (config YAML)
    pipeline_parser = subparsers.add_parser('run', help='Run training/scoring pipeline from config YAML')
    pipeline_parser.add_argument('config_path', help='Path to YAML config file')

    # Network analysis command
    # Config validation dry-run
    check_parser = subparsers.add_parser('check', help='Validate config without running the pipeline')
    check_parser.add_argument('config_path', help='Path to YAML config file')

    analyze_parser = subparsers.add_parser('analyze', help='Analyze allosteric network from scores CSV')
    analyze_parser.add_argument('scores_csv', help='Path to scores CSV produced by a pipeline run')
    analyze_parser.add_argument('--top-k', type=int, default=20,
                                help='Number of top-scoring pairs to include as graph edges (default 20)')
    analyze_parser.add_argument('--source', default=None,
                                help='Source residue for channel analysis, e.g. "A:12 GLY"')
    analyze_parser.add_argument('--sink', default=None,
                                help='Sink residue for channel analysis, e.g. "A:87 SER"')
    analyze_parser.add_argument('--top-paths', type=int, default=5,
                                help='Number of shortest paths to report (default 5)')
    analyze_parser.add_argument('--top-hubs', type=int, default=10,
                                help='Number of hub residues to report (default 10)')

    interpret_parser = subparsers.add_parser(
        'interpret', help='Extract candidate allosteric networks and interpret a scores CSV')
    interpret_parser.add_argument('scores_csv', help='Path to scores CSV produced by a pipeline run')
    interpret_parser.add_argument('--pdb', default=None, help='Reference structure/trajectory for structural context')
    interpret_parser.add_argument('--topology', default=None, help='Topology file for non-PDB trajectories')
    interpret_parser.add_argument('--top-k', type=int, default=20, help='Edges to include when building the graph')
    interpret_parser.add_argument('--top-paths', type=int, default=5, help='Candidate pathways to report')
    interpret_parser.add_argument('--top-hubs', type=int, default=10, help='Hub residues to report')
    interpret_parser.add_argument('--out-json', default=None, help='Output JSON path (default: <scores>.interpret.json)')
    interpret_parser.add_argument('--out-md', default=None, help='Output markdown path (default: <scores>.interpret.md)')
    interpret_parser.add_argument('--llm', default='none', choices=['none', 'ollama', 'anthropic', 'openai'],
                                  help='LLM backend for interpretation (default: none)')
    interpret_parser.add_argument('--llm-model', default=None, help='Model name for the chosen backend')
    interpret_parser.add_argument('--llm-base-url', default=None, help='Base URL (Ollama; default http://localhost:11434)')

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import sys as _sys
    effective: list[str] = list(argv) if argv is not None else _sys.argv[1:]
    # Legacy: bare config_path with no subcommand prefix → treat as 'run config_path'
    if effective and effective[0] not in _SUBCOMMANDS and not effective[0].startswith('-'):
        effective = ['run'] + effective
    args = build_parser().parse_args(effective)

    # Dispatch: subcommand 'check'
    if args.command == 'check':
        try:
            config = load_config(args.config_path)
            print(f'Config OK: mode={config.mode}, family={config.model.family}')
            return 0
        except Exception as exc:
            print(str(exc), file=_sys.stderr)
            return 1

    # Dispatch: subcommand 'analyze'
    if args.command == 'analyze':
        report = run_network_analysis(
            scores_csv=args.scores_csv,
            top_k=args.top_k,
            source=args.source,
            sink=args.sink,
            top_paths=args.top_paths,
            top_hubs=args.top_hubs,
        )
        print(report)
        return 0

    # Dispatch: subcommand 'interpret'
    if args.command == 'interpret':
        scores_path = Path(args.scores_csv)
        out_json = args.out_json or scores_path.with_suffix('.interpret.json')
        out_md = args.out_md or scores_path.with_suffix('.interpret.md')
        report = run_interpretation(
            scores_path,
            out_json=out_json,
            out_md=out_md,
            pdb_path=args.pdb,
            topology_path=args.topology,
            top_k=args.top_k,
            top_paths=args.top_paths,
            top_hubs=args.top_hubs,
            llm=args.llm,
            llm_model=args.llm_model,
            llm_base_url=args.llm_base_url,
        )
        counts = {key: len(value) for key, value in report['candidates'].items()}
        print(f'interpret candidates={counts} json={out_json} md={out_md}')
        return 0

    # Dispatch: subcommand 'run'
    config_path = getattr(args, 'config_path', None)
    if config_path is None:
        build_parser().print_help()
        return 1

    config = load_config(config_path)

    if config.mode == 'train':
        _run_train(config)
    elif config.mode == 'score':
        _run_score(config)
    else:
        _run_run(config)

    print(f'completed mode={config.mode}')
    return 0


def _run_train(config: AppConfig) -> TrainResult:
    training = config.training
    model_path = config.output.model_path
    if training is None or model_path is None:
        raise ValueError('train mode requires training config and model_path')

    if config.model.family == 'influence':
        inf_result = train_influence_model(
            pdb_path=config.data.pdb_path,
            window_size=config.data.window_size,
            stride=config.data.stride,
            time_step=config.data.time_step,
            preprocess=config.data.preprocess,
            hidden_dim=config.model.hidden_dim,
            num_encoder_layers=config.model.residue_layers,
            dropout=config.model.dropout,
            min_sequence_separation=config.data.min_sequence_separation,
            epochs=training.epochs,
            learning_rate=training.learning_rate,
            sparsity_weight=training.sparsity_weight,
            validation_fraction=training.validation_fraction,
            patience=training.patience,
            seed=training.seed,
            device=training.device,
            batch_size=training.batch_size,
            verbose=training.verbose,
            checkpoint_path=model_path,
            config_snapshot=_serialize_config(config),
            topology_path=config.data.topology_path,
            normalize=config.data.normalize,
            grad_clip_norm=training.grad_clip_norm,
            mixed_precision=training.mixed_precision,
            lr_scheduler=training.lr_scheduler,
            residue_chunk_size=config.model.residue_chunk_size,
            deterministic=training.deterministic,
        )
        print(f'trained samples={inf_result.num_samples} checkpoint={model_path}')
        return inf_result  # type: ignore[return-value]

    if config.model.family == 'cri':
        result = train_cri_model(
            pdb_path=config.data.pdb_path,
            topology_path=config.data.topology_path,
            window_size=config.data.window_size,
            stride=config.data.stride,
            time_step=config.data.time_step,
            distance_cutoff=config.data.distance_cutoff,
            max_neighbors=config.data.max_neighbors,
            min_sequence_separation=config.data.min_sequence_separation,
            preprocess=config.data.preprocess,
            validation_fraction=training.validation_fraction,
            patience=training.patience,
            seed=training.seed,
            device=training.device,
            batch_size=training.batch_size,
            edge_types=int(config.model.edge_types or 0),
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout,
            epochs=training.epochs,
            learning_rate=training.learning_rate,
            entropy_weight=training.entropy_weight,
            no_edge_weight=training.no_edge_weight,
            verbose=training.verbose,
            checkpoint_path=model_path,
            config_snapshot=_serialize_config(config),
        )
        print(f'trained samples={result.num_samples} checkpoint={model_path}')
        return result  # type: ignore[return-value]

    result = train_model(
        pdb_path=config.data.pdb_path,
        topology_path=config.data.topology_path,
        window_size=config.data.window_size,
        horizon_size=config.data.horizon_size,
        stride=config.data.stride,
        hidden_dim=config.model.hidden_dim,
        residue_layers=config.model.residue_layers,
        pair_layers=config.model.pair_layers,
        dropout=config.model.dropout,
        epochs=training.epochs,
        learning_rate=training.learning_rate,
        consistency_weight=training.consistency_weight,
        validation_fraction=training.validation_fraction,
        patience=training.patience,
        seed=training.seed,
        device=training.device,
        batch_size=training.batch_size,
        verbose=training.verbose,
        checkpoint_path=model_path,
        config_snapshot=_serialize_config(config),
    )
    print(f'trained samples={result.num_samples} checkpoint={model_path}')
    return result


def _run_score(config: AppConfig) -> int:
    scoring = config.scoring
    model_path = config.output.model_path
    score_csv_path = config.output.score_csv_path
    if scoring is None or model_path is None or score_csv_path is None:
        raise ValueError('score mode requires scoring config, model_path, and score_csv_path')

    checkpoint = load_checkpoint(model_path)
    model = build_scoring_model(checkpoint)
    if config.model.family == 'influence':
        snapshot = checkpoint.metadata.get('training', {})
        scores = score_influence_trajectory(
            model=model,  # type: ignore[arg-type]
            pdb_path=config.data.pdb_path,
            topology_path=config.data.topology_path,
            window_size=config.data.window_size,
            stride=config.data.stride,
            time_step=config.data.time_step,
            preprocess=config.data.preprocess,
            normalize=bool(snapshot.get('normalize', False)),
            batch_size=config.training.batch_size if config.training else 8,
            device=config.training.device if config.training else 'cpu',
            min_sequence_separation=config.data.min_sequence_separation,
        )
    elif config.model.family == 'cri':
        scores = score_cri_trajectory(
            model=model,  # type: ignore[arg-type]
            pdb_path=config.data.pdb_path,
            topology_path=config.data.topology_path,
            window_size=config.data.window_size,
            stride=config.data.stride,
            time_step=config.data.time_step,
            distance_cutoff=config.data.distance_cutoff,
            max_neighbors=config.data.max_neighbors,
            min_sequence_separation=config.data.min_sequence_separation,
            preprocess=config.data.preprocess,
        )
    else:
        scores = score_trajectory(
            model=model,  # type: ignore[arg-type]
            pdb_path=config.data.pdb_path,
            topology_path=config.data.topology_path,
            window_size=config.data.window_size,
            horizon_size=config.data.horizon_size,
            stride=config.data.stride,
        )
    write_pair_scores_csv(score_csv_path, scores)
    print(f'scored pairs={len(scores)} csv={score_csv_path} top_k={scoring.top_k}')
    return len(scores)


def _run_run(config: AppConfig) -> None:
    _run_train(config)
    _run_score(config)


def _serialize_config(config: AppConfig) -> dict[str, Any]:
    return _serialize_value(asdict(config))


def _serialize_value(value: Any) -> Any:
    if is_dataclass(value):
        return _serialize_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


if __name__ == '__main__':
    raise SystemExit(main())
