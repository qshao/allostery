from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from allostery.config import AppConfig
from allostery.io import write_pair_scores_csv
from allostery.io.checkpoint import load_checkpoint
from allostery.pipeline.cri_score import score_cri_trajectory
from allostery.pipeline.cri_train import train_cri_model
from allostery.pipeline.influence_score import score_influence_trajectory
from allostery.pipeline.influence_train import train_influence_model
from allostery.pipeline.score import build_scoring_model, score_trajectory
from allostery.pipeline.train import train_model


def run_training(config: AppConfig) -> Any:
    training = config.training
    model_path = config.output.model_path
    if training is None or model_path is None:
        raise ValueError('train mode requires training config and model_path')

    if config.model.family == 'influence':
        return train_influence_model(
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
            config_snapshot=serialize_config(config),
            topology_path=config.data.topology_path,
            normalize=config.data.normalize,
            grad_clip_norm=training.grad_clip_norm,
            mixed_precision=training.mixed_precision,
            lr_scheduler=training.lr_scheduler,
            residue_chunk_size=config.model.residue_chunk_size,
            deterministic=training.deterministic,
        )

    if config.model.family == 'cri':
        return train_cri_model(
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
            config_snapshot=serialize_config(config),
        )

    return train_model(
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
        config_snapshot=serialize_config(config),
    )


def run_scoring(config: AppConfig) -> int:
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
    return len(scores)


def serialize_config(config: AppConfig) -> dict[str, Any]:
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


__all__ = ["run_scoring", "run_training", "serialize_config"]
