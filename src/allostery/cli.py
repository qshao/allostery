from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

from allostery.config import AppConfig, load_config
from allostery.io import write_pair_scores_csv
from allostery.pipeline.cri_score import score_cri_trajectory
from allostery.pipeline.cri_train import train_cri_model
from allostery.pipeline.influence_score import score_influence_trajectory
from allostery.pipeline.influence_train import train_influence_model
from allostery.pipeline.score import load_scoring_model, score_trajectory
from allostery.pipeline.train import TrainResult, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='allostery')
    parser.add_argument('config_path')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config_path)

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
            epochs=training.epochs,
            learning_rate=training.learning_rate,
            sparsity_weight=training.sparsity_weight,
            validation_fraction=training.validation_fraction,
            patience=training.patience,
            seed=training.seed,
            device=training.device,
            batch_size=training.batch_size,
            checkpoint_path=model_path,
            config_snapshot=_serialize_config(config),
        )
        print(f'trained samples={inf_result.num_samples} checkpoint={model_path}')
        return inf_result  # type: ignore[return-value]

    if config.model.family == 'cri':
        result = train_cri_model(
            pdb_path=config.data.pdb_path,
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
            checkpoint_path=model_path,
            config_snapshot=_serialize_config(config),
        )
        print(f'trained samples={result.num_samples} checkpoint={model_path}')
        return result  # type: ignore[return-value]

    result = train_model(
        pdb_path=config.data.pdb_path,
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

    model = load_scoring_model(model_path)
    if config.model.family == 'influence':
        scores = score_influence_trajectory(
            model=model,  # type: ignore[arg-type]
            pdb_path=config.data.pdb_path,
            window_size=config.data.window_size,
            stride=config.data.stride,
            time_step=config.data.time_step,
            preprocess=config.data.preprocess,
        )
    elif config.model.family == 'cri':
        scores = score_cri_trajectory(
            model=model,  # type: ignore[arg-type]
            pdb_path=config.data.pdb_path,
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
