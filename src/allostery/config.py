from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Literal


Mode = Literal['train', 'score', 'run']


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DataConfig:
    pdb_path: Path
    window_size: int
    horizon_size: int
    stride: int
    time_step: float = 1.0
    distance_cutoff: float = 20.0
    max_neighbors: int = 2
    min_sequence_separation: int = 0
    preprocess: str = 'none'
    topology_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ModelConfig:
    hidden_dim: int
    residue_layers: int
    pair_layers: int
    dropout: float
    family: str = 'relational'
    edge_types: int | None = None


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int
    learning_rate: float
    consistency_weight: float
    entropy_weight: float = 0.0
    no_edge_weight: float = 0.0
    sparsity_weight: float = 0.0
    validation_fraction: float = 0.2
    patience: int = 5
    seed: int = 0
    device: str = 'cpu'
    batch_size: int = 4
    verbose: bool = True


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    top_k: int


@dataclass(frozen=True, slots=True)
class OutputConfig:
    model_path: Path | None
    score_csv_path: Path | None


@dataclass(frozen=True, slots=True)
class AppConfig:
    mode: Mode
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig | None
    scoring: ScoringConfig | None
    output: OutputConfig


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    raw = _load_yaml_mapping(config_path.read_text(encoding='utf-8'))
    base_dir = config_path.parent
    config_filename = config_path.name

    mode = raw.get('mode')
    if mode not in {'train', 'score', 'run'}:
        raise ConfigError(
            f"{config_filename}: mode must be one of train, score, or run (got {mode!r})"
        )

    data_raw = _require_mapping(raw, 'data')
    model_raw = _require_mapping(raw, 'model')
    output_raw = _require_mapping(raw, 'output')
    training_raw = _require_optional_mapping(raw, 'training')
    scoring_raw = _require_optional_mapping(raw, 'scoring')

    training = None
    if mode in {'train', 'run'}:
        training_raw = _require_mode_mapping(training_raw, 'training', mode)
        training = TrainingConfig(
            epochs=int(_require_value(training_raw, 'epochs')),
            learning_rate=float(_require_value(training_raw, 'learning_rate')),
            consistency_weight=float(_require_value(training_raw, 'consistency_weight')),
            entropy_weight=float(training_raw.get('entropy_weight', 0.0)),
            no_edge_weight=float(training_raw.get('no_edge_weight', 0.0)),
            sparsity_weight=float(training_raw.get('sparsity_weight', 0.0)),
            validation_fraction=float(training_raw.get('validation_fraction', 0.2)),
            patience=int(training_raw.get('patience', 5)),
            seed=int(training_raw.get('seed', 0)),
            device=str(training_raw.get('device', 'cpu')),
            batch_size=int(training_raw.get('batch_size', 4)),
            verbose=bool(training_raw.get('verbose', True)),
        )

    scoring = None
    if mode in {'score', 'run'}:
        scoring_raw = _require_mode_mapping(scoring_raw, 'scoring', mode)
        scoring = ScoringConfig(top_k=int(_require_value(scoring_raw, 'top_k')))

    config = AppConfig(
        mode=mode,
        data=DataConfig(
            pdb_path=_resolve_path(base_dir, _require_value(data_raw, 'pdb_path')),
            window_size=int(_require_value(data_raw, 'window_size')),
            horizon_size=int(_require_value(data_raw, 'horizon_size')),
            stride=int(_require_value(data_raw, 'stride')),
            time_step=float(data_raw.get('time_step', 1.0)),
            distance_cutoff=float(data_raw.get('distance_cutoff', 20.0)),
            max_neighbors=int(data_raw.get('max_neighbors', 2)),
            min_sequence_separation=int(data_raw.get('min_sequence_separation', 0)),
            preprocess=str(data_raw.get('preprocess', 'none')),
            topology_path=_optional_path(base_dir, data_raw.get('topology_path')),
        ),
        model=ModelConfig(
            hidden_dim=int(_require_value(model_raw, 'hidden_dim')),
            residue_layers=int(_require_value(model_raw, 'residue_layers')),
            pair_layers=int(_require_value(model_raw, 'pair_layers')),
            dropout=float(_require_value(model_raw, 'dropout')),
            family=str(model_raw.get('family', 'relational')),
            edge_types=int(model_raw['edge_types']) if model_raw.get('edge_types') is not None else None,
        ),
        training=training,
        scoring=scoring,
        output=OutputConfig(
            model_path=_optional_path(base_dir, output_raw.get('model_path')),
            score_csv_path=_optional_path(base_dir, output_raw.get('score_csv_path')),
        ),
    )
    validate_config(config, config_filename)
    return config


def _load_yaml_mapping(text: str) -> dict[str, Any]:
    try:
        yaml = import_module('yaml')
    except ModuleNotFoundError as exc:
        raise RuntimeError('PyYAML is required to load configuration files') from exc

    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError('config must be a YAML mapping')
    return raw


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f'{key} section is required')
    return value


def _require_optional_mapping(raw: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f'{key} section must be a mapping')
    return value


def _require_mode_mapping(raw: dict[str, Any] | None, key: str, mode: Mode) -> dict[str, Any]:
    if raw is None:
        raise ValueError(f'{key} section is required for {mode} mode')
    return raw


def _require_value(raw: dict[str, Any], key: str) -> Any:
    if key not in raw or raw[key] is None:
        raise ValueError(f'{key} is required')
    return raw[key]


def _resolve_path(base_dir: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _optional_path(base_dir: Path, value: object) -> Path | None:
    if value in {None, ''}:
        return None
    return _resolve_path(base_dir, value)


def validate_config(config: AppConfig, config_file: str = "") -> None:
    errors: list[str] = []

    if not config.data.pdb_path.exists():
        errors.append(f"data.pdb_path does not exist (got {config.data.pdb_path!r})")
    if config.data.window_size <= 0:
        errors.append(f"data.window_size must be > 0 (got {config.data.window_size})")
    if config.data.horizon_size <= 0:
        errors.append(f"data.horizon_size must be > 0 (got {config.data.horizon_size})")
    if config.data.stride <= 0:
        errors.append(f"data.stride must be > 0 (got {config.data.stride})")
    if config.data.time_step <= 0:
        errors.append(f"data.time_step must be > 0 (got {config.data.time_step})")
    if config.data.distance_cutoff <= 0:
        errors.append(f"data.distance_cutoff must be > 0 (got {config.data.distance_cutoff})")
    if config.data.max_neighbors <= 0:
        errors.append(f"data.max_neighbors must be > 0 (got {config.data.max_neighbors})")
    if config.data.min_sequence_separation < 0:
        errors.append(
            f"data.min_sequence_separation must be >= 0 (got {config.data.min_sequence_separation})"
        )
    if config.data.preprocess not in {'none', 'center', 'align'}:
        errors.append(
            f"data.preprocess must be one of none, center, or align (got {config.data.preprocess!r})"
        )
    if config.model.hidden_dim <= 0:
        errors.append(f"model.hidden_dim must be > 0 (got {config.model.hidden_dim})")
    if config.model.residue_layers <= 0:
        errors.append(f"model.residue_layers must be > 0 (got {config.model.residue_layers})")
    if config.model.pair_layers <= 0:
        errors.append(f"model.pair_layers must be > 0 (got {config.model.pair_layers})")
    if config.model.family not in {'relational', 'cri', 'influence'}:
        errors.append(
            f"model.family must be one of relational, cri, or influence (got {config.model.family!r})"
        )
    if config.model.family == 'cri':
        if config.model.edge_types is None:
            errors.append("model.edge_types is required for cri model family")
        elif config.model.edge_types < 2:
            errors.append(f"model.edge_types must be >= 2 (got {config.model.edge_types})")
    if not 0.0 <= config.model.dropout < 1.0:
        errors.append(f"model.dropout must be >= 0.0 and < 1.0 (got {config.model.dropout})")
    if config.mode in {'train', 'run'}:
        if config.training is None:
            errors.append(f"training section is required for {config.mode} mode")
        else:
            if config.training.epochs <= 0:
                errors.append(f"training.epochs must be > 0 (got {config.training.epochs})")
            if config.training.learning_rate <= 0:
                errors.append(
                    f"training.learning_rate must be > 0 (got {config.training.learning_rate})"
                )
            if config.training.entropy_weight < 0:
                errors.append(
                    f"training.entropy_weight must be >= 0 (got {config.training.entropy_weight})"
                )
            if config.training.no_edge_weight < 0:
                errors.append(
                    f"training.no_edge_weight must be >= 0 (got {config.training.no_edge_weight})"
                )
            if config.training.sparsity_weight < 0:
                errors.append(
                    f"training.sparsity_weight must be >= 0 (got {config.training.sparsity_weight})"
                )
            if not 0.0 <= config.training.validation_fraction < 1.0:
                errors.append(
                    f"training.validation_fraction must be >= 0.0 and < 1.0"
                    f" (got {config.training.validation_fraction})"
                )
            if config.training.patience < 0:
                errors.append(
                    f"training.patience must be >= 0 (got {config.training.patience})"
                )
            if not config.training.device:
                errors.append("training.device must not be empty")
            if config.training.batch_size <= 0:
                errors.append(
                    f"training.batch_size must be > 0 (got {config.training.batch_size})"
                )
    if config.mode in {'score', 'run'}:
        if config.scoring is None:
            errors.append(f"scoring section is required for {config.mode} mode")
        else:
            if config.scoring.top_k <= 0:
                errors.append(f"scoring.top_k must be > 0 (got {config.scoring.top_k})")
    if config.mode in {'train', 'score', 'run'} and config.output.model_path is None:
        errors.append("output.model_path is required")
    if config.mode in {'score', 'run'} and config.output.score_csv_path is None:
        errors.append("output.score_csv_path is required")

    if errors:
        joined = "\n  ".join(errors)
        prefix = f"{config_file}:\n  " if config_file else ""
        raise ConfigError(f"{prefix}{joined}")


__all__ = [
    'AppConfig',
    'ConfigError',
    'DataConfig',
    'ModelConfig',
    'Mode',
    'OutputConfig',
    'ScoringConfig',
    'TrainingConfig',
    'load_config',
    'validate_config',
]
