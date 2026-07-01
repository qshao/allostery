from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Literal


Mode = Literal['train', 'score', 'run']


class ConfigError(ValueError):
    pass


_DATA_KEYS: frozenset[str] = frozenset({
    'pdb_path', 'window_size', 'horizon_size', 'stride', 'time_step',
    'distance_cutoff', 'max_neighbors', 'min_sequence_separation', 'preprocess', 'topology_path',
    'normalize', 'window_sizes',
})
_MODEL_KEYS: frozenset[str] = frozenset({
    'family', 'hidden_dim', 'residue_layers', 'pair_layers', 'dropout', 'edge_types',
    'residue_chunk_size', 'num_heads',
})
_TRAINING_KEYS: frozenset[str] = frozenset({
    'epochs', 'learning_rate', 'consistency_weight', 'entropy_weight', 'no_edge_weight',
    'sparsity_weight', 'validation_fraction', 'patience', 'seed', 'device', 'batch_size', 'verbose',
    'mixed_precision', 'grad_clip_norm', 'lr_scheduler', 'deterministic',
})
_SCORING_KEYS: frozenset[str] = frozenset({'top_k'})
_OUTPUT_KEYS: frozenset[str] = frozenset({'model_path', 'score_csv_path'})
_ANALYZE_KEYS: frozenset[str] = frozenset({
    'top_k', 'source', 'sink', 'top_paths', 'top_hubs', 'out_path',
})
_INTERPRET_KEYS: frozenset[str] = frozenset({
    'llm', 'llm_model', 'llm_base_url', 'pdb_path',
    'top_k', 'top_paths', 'top_hubs', 'out_json', 'out_md',
})


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
    normalize: bool = True
    window_sizes: list[int] | None = None


@dataclass(frozen=True, slots=True)
class ModelConfig:
    hidden_dim: int
    residue_layers: int
    pair_layers: int
    dropout: float
    family: str = 'relational'
    edge_types: int | None = None
    residue_chunk_size: int | None = None
    num_heads: int = 4


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
    mixed_precision: bool = False
    grad_clip_norm: float | None = 1.0
    lr_scheduler: str = 'plateau'
    deterministic: bool = False


@dataclass(frozen=True, slots=True)
class ScoringConfig:
    top_k: int


@dataclass(frozen=True, slots=True)
class AnalyzeConfig:
    top_k: int = 20
    source: str | None = None
    sink: str | None = None
    top_paths: int = 5
    top_hubs: int = 10
    out_path: Path | None = None


@dataclass(frozen=True, slots=True)
class InterpretConfig:
    llm: str = 'none'
    llm_model: str | None = None
    llm_base_url: str | None = None
    pdb_path: Path | None = None
    top_k: int = 20
    top_paths: int = 5
    top_hubs: int = 10
    out_json: Path | None = None
    out_md: Path | None = None


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
    analyze: AnalyzeConfig | None = None
    interpret: InterpretConfig | None = None


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
    analyze_raw = _require_optional_mapping(raw, 'analyze')
    interpret_raw = _require_optional_mapping(raw, 'interpret')

    _warn_unknown_keys(data_raw, _DATA_KEYS, 'data', config_filename)
    _warn_unknown_keys(model_raw, _MODEL_KEYS, 'model', config_filename)
    if training_raw is not None:
        _warn_unknown_keys(training_raw, _TRAINING_KEYS, 'training', config_filename)
    if scoring_raw is not None:
        _warn_unknown_keys(scoring_raw, _SCORING_KEYS, 'scoring', config_filename)
    _warn_unknown_keys(output_raw, _OUTPUT_KEYS, 'output', config_filename)
    if analyze_raw is not None:
        _warn_unknown_keys(analyze_raw, _ANALYZE_KEYS, 'analyze', config_filename)
    if interpret_raw is not None:
        _warn_unknown_keys(interpret_raw, _INTERPRET_KEYS, 'interpret', config_filename)

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
            mixed_precision=bool(training_raw.get('mixed_precision', False)),
            grad_clip_norm=(
                float(training_raw['grad_clip_norm'])
                if 'grad_clip_norm' in training_raw and training_raw['grad_clip_norm'] is not None
                else (None if 'grad_clip_norm' in training_raw else 1.0)
            ),
            lr_scheduler=str(training_raw.get('lr_scheduler', 'plateau')),
            deterministic=bool(training_raw.get('deterministic', False)),
        )

    scoring = None
    if mode in {'score', 'run'}:
        scoring_raw = _require_mode_mapping(scoring_raw, 'scoring', mode)
        scoring = ScoringConfig(top_k=int(_require_value(scoring_raw, 'top_k')))

    analyze_cfg = None
    if analyze_raw is not None:
        analyze_cfg = AnalyzeConfig(
            top_k=int(analyze_raw.get('top_k', 20)),
            source=(str(analyze_raw['source']) if analyze_raw.get('source') else None),
            sink=(str(analyze_raw['sink']) if analyze_raw.get('sink') else None),
            top_paths=int(analyze_raw.get('top_paths', 5)),
            top_hubs=int(analyze_raw.get('top_hubs', 10)),
            out_path=_optional_path(base_dir, analyze_raw.get('out_path')),
        )

    interpret_cfg = None
    if interpret_raw is not None:
        interpret_cfg = InterpretConfig(
            llm=str(interpret_raw.get('llm', 'none')),
            llm_model=(str(interpret_raw['llm_model']) if interpret_raw.get('llm_model') else None),
            llm_base_url=(str(interpret_raw['llm_base_url']) if interpret_raw.get('llm_base_url') else None),
            pdb_path=_optional_path(base_dir, interpret_raw.get('pdb_path')),
            top_k=int(interpret_raw.get('top_k', 20)),
            top_paths=int(interpret_raw.get('top_paths', 5)),
            top_hubs=int(interpret_raw.get('top_hubs', 10)),
            out_json=_optional_path(base_dir, interpret_raw.get('out_json')),
            out_md=_optional_path(base_dir, interpret_raw.get('out_md')),
        )

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
            normalize=bool(data_raw.get('normalize', True)),
            window_sizes=(
                [int(w) for w in data_raw['window_sizes']]
                if data_raw.get('window_sizes') is not None else None
            ),
        ),
        model=ModelConfig(
            hidden_dim=int(_require_value(model_raw, 'hidden_dim')),
            residue_layers=int(_require_value(model_raw, 'residue_layers')),
            pair_layers=int(_require_value(model_raw, 'pair_layers')),
            dropout=float(_require_value(model_raw, 'dropout')),
            family=str(model_raw.get('family', 'relational')),
            edge_types=int(model_raw['edge_types']) if model_raw.get('edge_types') is not None else None,
            residue_chunk_size=(
                int(model_raw['residue_chunk_size'])
                if model_raw.get('residue_chunk_size') is not None else None
            ),
            num_heads=int(model_raw.get('num_heads', 4)),
        ),
        training=training,
        scoring=scoring,
        output=OutputConfig(
            model_path=_optional_path(base_dir, output_raw.get('model_path')),
            score_csv_path=_optional_path(base_dir, output_raw.get('score_csv_path')),
        ),
        analyze=analyze_cfg,
        interpret=interpret_cfg,
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


def _warn_unknown_keys(
    raw: dict[str, Any],
    known: frozenset[str],
    section: str,
    config_file: str,
) -> None:
    prefix = f"{config_file}: " if config_file else ""
    for key in raw:
        if key not in known:
            print(
                f"warning: {prefix}{section}.{key} is not a recognized config key",
                file=sys.stderr,
            )


def validate_config(config: AppConfig, config_file: str = "") -> None:
    errors: list[str] = []

    if not config.data.pdb_path.exists():
        errors.append(f"data.pdb_path does not exist (got {config.data.pdb_path!r})")
    if (config.data.topology_path is not None
            and not config.data.topology_path.exists()):
        errors.append(
            f"data.topology_path: file not found: {config.data.topology_path}"
        )
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
    if config.model.residue_chunk_size is not None and config.model.residue_chunk_size <= 0:
        errors.append(
            f"model.residue_chunk_size must be > 0 (got {config.model.residue_chunk_size})"
        )
    if config.model.num_heads <= 0:
        errors.append(f"model.num_heads must be > 0 (got {config.model.num_heads})")
    if config.model.hidden_dim % config.model.num_heads != 0:
        errors.append(
            f"model.hidden_dim ({config.model.hidden_dim}) must be divisible by "
            f"model.num_heads ({config.model.num_heads})"
        )
    if config.data.window_sizes is not None:
        for ws in config.data.window_sizes:
            if ws <= 0:
                errors.append(f"data.window_sizes entries must be > 0 (got {ws})")
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
            if config.training.device.startswith('cuda'):
                try:
                    _torch = import_module('torch')
                    if not _torch.cuda.is_available():
                        errors.append(
                            f"training.device is {config.training.device!r} but CUDA is not "
                            f"available on this machine"
                        )
                except ImportError:
                    pass  # torch not yet installed; skip check
            if config.training.batch_size <= 0:
                errors.append(
                    f"training.batch_size must be > 0 (got {config.training.batch_size})"
                )
            if config.training.lr_scheduler not in {'none', 'plateau'}:
                errors.append(
                    f"training.lr_scheduler must be one of none, plateau "
                    f"(got {config.training.lr_scheduler!r})"
                )
            if config.training.grad_clip_norm is not None and config.training.grad_clip_norm <= 0:
                errors.append(
                    f"training.grad_clip_norm must be > 0 (got {config.training.grad_clip_norm})"
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

    if (config.analyze is not None or config.interpret is not None) and config.mode == 'train':
        errors.append(
            "analyze/interpret sections require scoring; set mode to 'score' or 'run', not 'train'"
        )

    if config.analyze is not None:
        a = config.analyze
        if a.top_k <= 0:
            errors.append(f"analyze.top_k must be > 0 (got {a.top_k})")
        if a.top_paths <= 0:
            errors.append(f"analyze.top_paths must be > 0 (got {a.top_paths})")
        if a.top_hubs <= 0:
            errors.append(f"analyze.top_hubs must be > 0 (got {a.top_hubs})")
        if (a.source is None) != (a.sink is None):
            errors.append("analyze.source and analyze.sink must be provided together")
    if config.interpret is not None:
        it = config.interpret
        if it.llm not in {'none', 'ollama', 'anthropic', 'openai'}:
            errors.append(
                f"interpret.llm must be one of none, ollama, anthropic, openai (got {it.llm!r})"
            )
        if it.top_k <= 0:
            errors.append(f"interpret.top_k must be > 0 (got {it.top_k})")
        if it.top_paths <= 0:
            errors.append(f"interpret.top_paths must be > 0 (got {it.top_paths})")
        if it.top_hubs <= 0:
            errors.append(f"interpret.top_hubs must be > 0 (got {it.top_hubs})")

    if errors:
        joined = "\n  ".join(errors)
        prefix = f"{config_file}:\n  " if config_file else ""
        raise ConfigError(f"{prefix}{joined}")


__all__ = [
    'AnalyzeConfig',
    'AppConfig',
    'ConfigError',
    'DataConfig',
    'InterpretConfig',
    'ModelConfig',
    'Mode',
    'OutputConfig',
    'ScoringConfig',
    'TrainingConfig',
    'load_config',
    'validate_config',
]
