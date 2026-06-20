from __future__ import annotations

from pathlib import Path
from typing import Callable

from allostery.cli_output import Result
from allostery.config import AppConfig
from allostery.pipeline.analyze import run_network_analysis
from allostery.pipeline.execute import run_scoring, run_training
from allostery.pipeline.interpret import run_interpretation


class WorkflowError(RuntimeError):
    def __init__(self, stage: str, artifacts: list[Path], cause: BaseException) -> None:
        paths = ", ".join(str(p) for p in artifacts) or "(none)"
        super().__init__(
            f"workflow failed at stage {stage!r}; artifacts already written: {paths}. "
            f"Cause: {cause}"
        )
        self.stage = stage
        self.artifacts = artifacts


def run_workflow(
    config: AppConfig,
    *,
    backend=None,
    progress: Callable[[str], None] | None = None,
) -> Result:
    stages: list[str] = []
    artifacts: list[Path] = []
    summary: list[str] = []

    def step(name: str) -> None:
        stages.append(name)
        if progress is not None:
            progress(name)

    wants_post = config.analyze is not None or config.interpret is not None
    if wants_post and config.mode == 'train':
        raise ValueError(
            "analyze/interpret stages require a scoring stage; set mode to 'score' or 'run'"
        )

    if config.mode in {'train', 'run'}:
        step('train')
        result = run_training(config)
        summary.append(f'trained samples={result.num_samples} checkpoint={config.output.model_path}')
        if config.output.model_path is not None:
            artifacts.append(config.output.model_path)

    if config.mode in {'score', 'run'}:
        step('score')
        count = run_scoring(config)
        top_k = config.scoring.top_k if config.scoring else 0
        summary.append(f'scored pairs={count} csv={config.output.score_csv_path} top_k={top_k}')
        if config.output.score_csv_path is not None:
            artifacts.append(config.output.score_csv_path)

    score_csv = config.output.score_csv_path

    current = 'analyze'
    try:
        if config.analyze is not None:
            step('analyze')
            a = config.analyze
            out_path = a.out_path or score_csv.with_suffix('.network.txt')
            run_network_analysis(
                score_csv, top_k=a.top_k, source=a.source, sink=a.sink,
                top_paths=a.top_paths, top_hubs=a.top_hubs, out_path=out_path,
            )
            summary.append(f'analyzed network -> {out_path}')
            artifacts.append(out_path)

        if config.interpret is not None:
            current = 'interpret'
            step('interpret')
            it = config.interpret
            out_json = it.out_json or score_csv.with_suffix('.interpret.json')
            out_md = it.out_md or score_csv.with_suffix('.interpret.md')
            run_interpretation(
                score_csv, out_json=out_json, out_md=out_md,
                pdb_path=it.pdb_path or config.data.pdb_path,
                topology_path=config.data.topology_path,
                top_k=it.top_k, top_paths=it.top_paths, top_hubs=it.top_hubs,
                llm=it.llm, llm_model=it.llm_model, llm_base_url=it.llm_base_url,
                backend=backend,
            )
            summary.append(f'interpreted -> {out_json}, {out_md}')
            artifacts.extend([out_json, out_md])
    except Exception as exc:  # noqa: BLE001 - re-raised with context, artifacts preserved
        raise WorkflowError(stage=current, artifacts=artifacts, cause=exc) from exc

    summary.append(f"workflow complete (stages: {', '.join(stages)})")
    return Result(
        command='workflow',
        summary='\n'.join(summary),
        data={'stages': stages, 'mode': config.mode},
        artifacts=artifacts,
    )


__all__ = ['WorkflowError', 'run_workflow']
