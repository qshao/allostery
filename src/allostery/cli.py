from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from allostery import __version__
from allostery.cli_errors import USER_ERROR, exit_code_for
from allostery.cli_output import Result, format_result
from allostery.config import AppConfig, load_config
from allostery.pipeline.analyze import run_network_analysis
from allostery.pipeline.execute import run_scoring, run_training
from allostery.pipeline.interpret import run_interpretation
from allostery.pipeline.progress import TrainingProgress
from allostery.pipeline.workflow import run_workflow
from allostery.validation.harness import (
    ValidationConfig,
    render_validation_table,
    run_validation,
    validation_report_to_dict,
)


_SUBCOMMANDS = frozenset({'run', 'analyze', 'check', 'interpret', 'workflow', 'validate'})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='allostery')
    parser.add_argument('--version', action='version', version=f'allostery {__version__}')
    parser.add_argument('--debug', action='store_true',
                        help='Show full tracebacks instead of a clean error message')
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('--json', action='store_true',
                              help='Emit a single JSON object on stdout (for scripts)')
    output_group.add_argument('--quiet', action='store_true',
                              help='Suppress summaries; print only artifact paths')
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
    analyze_parser.add_argument(
        '--pdb', default=None,
        help='Structure file for PyMOL export (required when --out-pml is given)'
    )
    analyze_parser.add_argument(
        '--out-pml', default=None,
        help='Write a PyMOL .pml script to this path'
    )

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

    workflow_parser = subparsers.add_parser(
        'workflow', help='Run train/score then analyze+interpret end to end from one config')
    workflow_parser.add_argument('config_path', help='Path to YAML config file')

    validate_parser = subparsers.add_parser(
        'validate',
        help='Measure scorer accuracy against synthetic planted-coupling ground truth')
    validate_parser.add_argument(
        '--scorers', default=None,
        help='Comma-separated scorers (default: all). '
             'Choices: dccm,mi,contact,null,influence,cri,relational')
    validate_parser.add_argument('--n-residues', type=int, default=24)
    validate_parser.add_argument('--couplings', type=int, default=8)
    validate_parser.add_argument('--noise', type=float, default=0.05)
    validate_parser.add_argument('--frames', type=int, default=128)
    validate_parser.add_argument('--seeds', type=int, default=3)
    validate_parser.add_argument('--seed', type=int, default=0)
    validate_parser.add_argument('--out-json', default=None, help='Write the full JSON report here')

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import sys as _sys
    effective: list[str] = list(argv) if argv is not None else _sys.argv[1:]
    if effective and effective[0] not in _SUBCOMMANDS and not effective[0].startswith('-'):
        effective = ['run'] + effective
    args = build_parser().parse_args(effective)

    try:
        result = _dispatch(args)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - mapped to a clean message
        if args.debug:
            raise
        code = exit_code_for(exc)
        message = str(exc) if code is not None else 'internal error; rerun with --debug for details'
        if code is None:
            code = USER_ERROR
        result = Result(command=getattr(args, 'command', '') or '', status='error', error=message)
        _emit(result, args)
        return code

    _emit(result, args)
    return 0


def _emit(result: Result, args: argparse.Namespace) -> None:
    import sys as _sys
    stdout_text, stderr_text = format_result(
        result, json_mode=args.json, quiet=args.quiet,
    )
    if stdout_text:
        print(stdout_text)
    if stderr_text:
        print(stderr_text, file=_sys.stderr)


def _dispatch(args: argparse.Namespace) -> Result:
    if args.command == 'validate':
        scorers = None
        if args.scorers:
            scorers = [name.strip() for name in args.scorers.split(',') if name.strip()]
        config = ValidationConfig(
            n_residues=args.n_residues,
            n_couplings=args.couplings,
            noise=args.noise,
            frames=args.frames,
            seeds=args.seeds,
            base_seed=args.seed,
        )
        report = run_validation(config, scorers=scorers)
        data = validation_report_to_dict(report)
        artifacts: list[Path] = []
        if args.out_json:
            out_json = Path(args.out_json)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(json.dumps(data, indent=2), encoding='utf-8')
            artifacts.append(out_json)
        return Result(
            command='validate',
            summary=render_validation_table(report),
            data=data,
            artifacts=artifacts,
        )

    if args.command == 'workflow':
        import sys as _sys
        config = load_config(args.config_path)
        emit_progress = not args.json and not args.quiet
        stage_progress = (lambda stage: print(f'[{stage}] ...', file=_sys.stderr)) if emit_progress else None
        total_epochs = config.training.epochs if config.training and config.mode in {'train', 'run'} else 0
        with TrainingProgress(total_epochs, quiet=not emit_progress) as tp:
            result = run_workflow(config, progress=stage_progress, training_progress_fn=tp.update)
        return result

    if args.command == 'check':
        config = load_config(args.config_path)
        return Result(
            command='check',
            summary=f'Config OK: mode={config.mode}, family={config.model.family}',
            data={'mode': config.mode, 'family': config.model.family},
        )

    if args.command == 'analyze':
        out_pml = Path(args.out_pml) if args.out_pml else None
        pdb_path = Path(args.pdb) if args.pdb else None
        if out_pml is not None and pdb_path is None:
            raise ValueError("--out-pml requires --pdb to specify the structure file")
        report = run_network_analysis(
            scores_csv=args.scores_csv,
            top_k=args.top_k,
            source=args.source,
            sink=args.sink,
            top_paths=args.top_paths,
            top_hubs=args.top_hubs,
            out_pml=out_pml,
            pdb_path=pdb_path,
        )
        artifacts = [out_pml] if out_pml is not None else []
        return Result(command='analyze', summary=report, artifacts=artifacts)

    if args.command == 'interpret':
        scores_path = Path(args.scores_csv)
        out_json = Path(args.out_json) if args.out_json else scores_path.with_suffix('.interpret.json')
        out_md = Path(args.out_md) if args.out_md else scores_path.with_suffix('.interpret.md')
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
        return Result(
            command='interpret',
            summary=f'interpret candidates={counts} json={out_json} md={out_md}',
            data={'counts': counts},
            artifacts=[out_json, out_md],
        )

    # 'run' (and the legacy bare-config form)
    config_path = getattr(args, 'config_path', None)
    if config_path is None:
        build_parser().print_help()
        raise ValueError('no command given')
    config = load_config(config_path)
    lines: list[str] = []
    artifacts: list[Path] = []
    if config.mode in {'train', 'run'}:
        total_epochs = config.training.epochs if config.training else 0
        quiet = args.quiet or args.json
        with TrainingProgress(total_epochs, quiet=quiet) as tp:
            result = run_training(config, progress_fn=tp.update)
        tp.finish(
            best_epoch=getattr(result, 'best_epoch', None),
            best_val_loss=getattr(result, 'best_validation_loss', None),
        )
        lines.append(f'trained samples={result.num_samples} checkpoint={config.output.model_path}')
        if config.output.model_path is not None:
            artifacts.append(config.output.model_path)
    if config.mode in {'score', 'run'}:
        count = run_scoring(config)
        top_k = config.scoring.top_k if config.scoring else 0
        lines.append(f'scored pairs={count} csv={config.output.score_csv_path} top_k={top_k}')
        if config.output.score_csv_path is not None:
            artifacts.append(config.output.score_csv_path)
    lines.append(f'completed mode={config.mode}')
    return Result(command='run', summary='\n'.join(lines), data={'mode': config.mode}, artifacts=artifacts)


if __name__ == '__main__':
    raise SystemExit(main())
