from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Literal


Mode = Literal['train', 'score', 'run']


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
    validation_fraction: float = 0.2
    patience: int = 5
    seed: int = 0
    device: str = 'cpu'
    batch_size: int = 4


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

    mode = raw.get('mode')
    if mode not in {'train', 'score', 'run'}:
        raise ValueError('mode must be one of train, score, or run')

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
            validation_fraction=float(training_raw.get('validation_fraction', 0.2)),
            patience=int(training_raw.get('patience', 5)),
            seed=int(training_raw.get('seed', 0)),
            device=str(training_raw.get('device', 'cpu')),
            batch_size=int(training_raw.get('batch_size', 4)),
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
    validate_config(config)
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


def validate_config(config: AppConfig) -> None:
    if not config.data.pdb_path.exists():
        raise ValueError('pdb_path does not exist')
    if config.data.window_size <= 0:
        raise ValueError('window_size must be greater than zero')
    if config.data.horizon_size <= 0:
        raise ValueError('horizon_size must be greater than zero')
    if config.data.stride <= 0:
        raise ValueError('stride must be greater than zero')
    if config.data.time_step <= 0:
        raise ValueError('time_step must be greater than zero')
    if config.data.distance_cutoff <= 0:
        raise ValueError('distance_cutoff must be greater than zero')
    if config.data.max_neighbors <= 0:
        raise ValueError('max_neighbors must be greater than zero')
    if config.data.min_sequence_separation < 0:
        raise ValueError('min_sequence_separation must be greater than or equal to zero')
    if config.data.preprocess not in {'none', 'center', 'align'}:
        raise ValueError('preprocess must be one of none, center, or align')
    if config.model.hidden_dim <= 0:
        raise ValueError('hidden_dim must be greater than zero')
    if config.model.residue_layers <= 0:
        raise ValueError('residue_layers must be greater than zero')
    if config.model.pair_layers <= 0:
        raise ValueError('pair_layers must be greater than zero')
    if config.model.family not in {'relational', 'cri'}:
        raise ValueError('family must be one of relational or cri')
    if config.model.family == 'cri':
        if config.model.edge_types is None:
            raise ValueError('edge_types is required for cri model family')
        if config.model.edge_types < 2:
            raise ValueError('edge_types must be at least 2')
    if not 0.0 <= config.model.dropout < 1.0:
        raise ValueError('dropout must be greater than or equal to zero and less than one')
    if config.mode in {'train', 'run'}:
        if config.training is None:
            raise ValueError(f'training section is required for {config.mode} mode')
        if config.training.epochs <= 0:
            raise ValueError('epochs must be greater than zero')
        if config.training.learning_rate <= 0:
            raise ValueError('learning_rate must be greater than zero')
        if config.training.entropy_weight < 0:
            raise ValueError('entropy_weight must be greater than or equal to zero')
        if config.training.no_edge_weight < 0:
            raise ValueError('no_edge_weight must be greater than or equal to zero')
        if not 0.0 <= config.training.validation_fraction < 1.0:
            raise ValueError('validation_fraction must be greater than or equal to zero and less than one')
        if config.training.patience < 0:
            raise ValueError('patience must be greater than or equal to zero')
        if not config.training.device:
            raise ValueError('device must not be empty')
        if config.training.batch_size <= 0:
            raise ValueError('batch_size must be greater than zero')
    if config.mode in {'score', 'run'}:
        if config.scoring is None:
            raise ValueError(f'scoring section is required for {config.mode} mode')
        if config.scoring.top_k <= 0:
            raise ValueError('top_k must be greater than zero')
    if config.mode in {'train', 'score', 'run'} and config.output.model_path is None:
        raise ValueError('model_path is required')
    if config.mode in {'score', 'run'} and config.output.score_csv_path is None:
        raise ValueError('score_csv_path is required')


__all__ = [
    'AppConfig',
    'DataConfig',
    'ModelConfig',
    'Mode',
    'OutputConfig',
    'ScoringConfig',
    'TrainingConfig',
    'load_config',
    'validate_config',
]
