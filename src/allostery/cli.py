from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from allostery import __version__
from allostery.config import AppConfig, load_config
from allostery.pipeline.analyze import run_network_analysis
from allostery.pipeline.execute import run_scoring, run_training, serialize_config
from allostery.pipeline.interpret import run_interpretation


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


def _run_train(config: AppConfig):
    result = run_training(config)
    print(f'trained samples={result.num_samples} checkpoint={config.output.model_path}')
    return result


def _run_score(config: AppConfig) -> int:
    count = run_scoring(config)
    scoring = config.scoring
    print(f'scored pairs={count} csv={config.output.score_csv_path} '
          f'top_k={scoring.top_k if scoring else 0}')
    return count


def _run_run(config: AppConfig) -> None:
    _run_train(config)
    _run_score(config)


if __name__ == '__main__':
    raise SystemExit(main())
