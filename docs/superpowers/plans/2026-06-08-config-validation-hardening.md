# Config Validation Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every config error immediately actionable — showing the filename, the offending key, and the received value — and report all problems in a single raise instead of one per run.

**Architecture:** Introduce `ConfigError(ValueError)` so existing `except ValueError` callers are unaffected. Refactor `validate_config` to accumulate errors into a list and raise once. Thread the config filename into all error messages. Add per-section unknown-key detection that prints warnings to stderr.

**Tech Stack:** Python 3.11, pytest. Changes confined to `src/allostery/config.py` and `tests/test_config.py`.

---

## File Structure

- **Modify** `src/allostery/config.py` — add `ConfigError`, refactor `validate_config`, add `_warn_unknown_keys`, add known-key frozensets, thread filename through `load_config`.
- **Modify** `tests/test_config.py` — add tests for filename context, "got X" values, multi-error reporting, unknown-key warnings, and `ConfigError` subclassing.

---

### Task 1: Add `ConfigError` and wire it into `load_config`

**Files:**
- Modify: `src/allostery/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add this test at the bottom of `tests/test_config.py`:

```python
def test_config_error_is_value_error_subclass() -> None:
    from allostery.config import ConfigError
    assert issubclass(ConfigError, ValueError)
```

- [ ] **Step 2: Run the test to see it fail**

```bash
python -m pytest tests/test_config.py::test_config_error_is_value_error_subclass -v
```

Expected: `ImportError: cannot import name 'ConfigError'`.

- [ ] **Step 3: Add `ConfigError` to `config.py`, update `validate_config` signature, and update `load_config`**

In `src/allostery/config.py`, directly after the imports block add:

```python
class ConfigError(ValueError):
    pass
```

Change `validate_config`'s signature to accept an optional filename (body stays the same for now):

```python
def validate_config(config: AppConfig, config_file: str = "") -> None:
```

In `load_config`, add `config_filename = config_path.name` right after `base_dir = config_path.parent` and update the mode check to use `ConfigError`:

```python
config_filename = config_path.name
mode = raw.get('mode')
if mode not in {'train', 'score', 'run'}:
    raise ConfigError(
        f"{config_filename}: mode must be one of train, score, or run (got {mode!r})"
    )
```

Change the `validate_config` call at the end of `load_config` to:

```python
validate_config(config, config_filename)
```

Add `'ConfigError'` to the `__all__` list at the bottom of the file.

- [ ] **Step 4: Run the test to see it pass**

```bash
python -m pytest tests/test_config.py::test_config_error_is_value_error_subclass -v
```

Expected: PASS.

- [ ] **Step 5: Run the full test suite**

```bash
python -m pytest tests/ -q
```

Expected: all tests pass (`ConfigError` is a `ValueError` subclass so existing `pytest.raises(ValueError)` checks still match).

- [ ] **Step 6: Commit**

```bash
git add src/allostery/config.py tests/test_config.py
git commit -m "feat: add ConfigError subclass and thread filename into mode error"
```

---

### Task 2: Refactor `validate_config` to collect all errors and include filename + "got X"

**Files:**
- Modify: `src/allostery/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Add these three tests to `tests/test_config.py`:

```python
def test_config_error_message_includes_filename(tmp_path: Path) -> None:
    from allostery.config import load_config
    bad_config = tmp_path / "bad.yaml"
    _write_config(
        bad_config,
        [
            "mode: run",
            "data:",
            f"  pdb_path: {FIXTURE_PDB}",
            "  window_size: 0",  # invalid
            "  horizon_size: 1",
            "  stride: 1",
            "model:",
            "  hidden_dim: 8",
            "  residue_layers: 2",
            "  pair_layers: 2",
            "  dropout: 0.0",
            "training:",
            "  epochs: 1",
            "  learning_rate: 0.001",
            "  consistency_weight: 0.25",
            "scoring:",
            "  top_k: 5",
            "output:",
            "  model_path: outputs/model.pt",
            "  score_csv_path: outputs/scores.csv",
        ],
    )
    with pytest.raises(ValueError) as exc_info:
        load_config(bad_config)
    assert "bad.yaml" in str(exc_info.value)


def test_config_error_includes_got_value(tmp_path: Path) -> None:
    from allostery.config import load_config
    bad_config = tmp_path / "got.yaml"
    _write_config(
        bad_config,
        [
            "mode: run",
            "data:",
            f"  pdb_path: {FIXTURE_PDB}",
            "  window_size: 0",  # invalid — should mention "got 0"
            "  horizon_size: 1",
            "  stride: 1",
            "model:",
            "  hidden_dim: 8",
            "  residue_layers: 2",
            "  pair_layers: 2",
            "  dropout: 0.0",
            "training:",
            "  epochs: 1",
            "  learning_rate: 0.001",
            "  consistency_weight: 0.25",
            "scoring:",
            "  top_k: 5",
            "output:",
            "  model_path: outputs/model.pt",
            "  score_csv_path: outputs/scores.csv",
        ],
    )
    with pytest.raises(ValueError) as exc_info:
        load_config(bad_config)
    assert "got 0" in str(exc_info.value)


def test_config_error_reports_multiple_errors_at_once(tmp_path: Path) -> None:
    from allostery.config import load_config
    bad_config = tmp_path / "multi.yaml"
    _write_config(
        bad_config,
        [
            "mode: run",
            "data:",
            f"  pdb_path: {FIXTURE_PDB}",
            "  window_size: 0",   # bad
            "  horizon_size: 0",  # also bad
            "  stride: 1",
            "model:",
            "  hidden_dim: 8",
            "  residue_layers: 2",
            "  pair_layers: 2",
            "  dropout: 0.0",
            "training:",
            "  epochs: 1",
            "  learning_rate: 0.001",
            "  consistency_weight: 0.25",
            "scoring:",
            "  top_k: 5",
            "output:",
            "  model_path: outputs/model.pt",
            "  score_csv_path: outputs/scores.csv",
        ],
    )
    with pytest.raises(ValueError) as exc_info:
        load_config(bad_config)
    msg = str(exc_info.value)
    assert "window_size" in msg
    assert "horizon_size" in msg
```

- [ ] **Step 2: Run the failing tests**

```bash
python -m pytest tests/test_config.py::test_config_error_message_includes_filename tests/test_config.py::test_config_error_includes_got_value tests/test_config.py::test_config_error_reports_multiple_errors_at_once -v
```

Expected: all three FAIL (current `validate_config` raises on first error with no filename or "got" context).

- [ ] **Step 3: Replace the body of `validate_config` with the error-accumulating version**

Replace the entire `validate_config` function in `src/allostery/config.py` with:

```python
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
```

- [ ] **Step 4: Run all config tests**

```bash
python -m pytest tests/test_config.py -v
```

Expected: all tests pass, including the three new ones.

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/allostery/config.py tests/test_config.py
git commit -m "feat: collect all config validation errors and include filename and 'got X' context"
```

---

### Task 3: Add unknown-key detection with stderr warnings

**Files:**
- Modify: `src/allostery/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_unknown_key_in_training_prints_warning_to_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from allostery.config import load_config
    config_path = tmp_path / "typo.yaml"
    _write_config(
        config_path,
        [
            "mode: run",
            "data:",
            f"  pdb_path: {FIXTURE_PDB}",
            "  window_size: 1",
            "  horizon_size: 1",
            "  stride: 1",
            "model:",
            "  hidden_dim: 8",
            "  residue_layers: 2",
            "  pair_layers: 2",
            "  dropout: 0.0",
            "training:",
            "  epochs: 1",
            "  learnig_rate: 0.001",  # typo — should warn
            "  learning_rate: 0.001",  # real key present so no crash
            "  consistency_weight: 0.25",
            "scoring:",
            "  top_k: 5",
            "output:",
            "  model_path: outputs/model.pt",
            "  score_csv_path: outputs/scores.csv",
        ],
    )
    load_config(config_path)
    captured = capsys.readouterr()
    assert "learnig_rate" in captured.err
    assert "warning" in captured.err.lower()
```

- [ ] **Step 2: Run the test to see it fail**

```bash
python -m pytest tests/test_config.py::test_unknown_key_in_training_prints_warning_to_stderr -v
```

Expected: FAIL — no warning is currently emitted.

- [ ] **Step 3: Add `import sys`, known-key frozensets, and `_warn_unknown_keys` to `config.py`**

Add `import sys` to the imports at the top of `src/allostery/config.py` (alongside the existing stdlib imports).

Add these module-level constants directly after the `ConfigError` class definition:

```python
_DATA_KEYS: frozenset[str] = frozenset({
    'pdb_path', 'window_size', 'horizon_size', 'stride', 'time_step',
    'distance_cutoff', 'max_neighbors', 'min_sequence_separation', 'preprocess', 'topology_path',
})
_MODEL_KEYS: frozenset[str] = frozenset({
    'family', 'hidden_dim', 'residue_layers', 'pair_layers', 'dropout', 'edge_types',
})
_TRAINING_KEYS: frozenset[str] = frozenset({
    'epochs', 'learning_rate', 'consistency_weight', 'entropy_weight', 'no_edge_weight',
    'sparsity_weight', 'validation_fraction', 'patience', 'seed', 'device', 'batch_size', 'verbose',
})
_SCORING_KEYS: frozenset[str] = frozenset({'top_k'})
_OUTPUT_KEYS: frozenset[str] = frozenset({'model_path', 'score_csv_path'})
```

Add this private function alongside the other helper functions (before `validate_config`):

```python
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
```

- [ ] **Step 4: Call `_warn_unknown_keys` from `load_config`**

In `load_config`, after the five section-extraction calls (`_require_mapping` / `_require_optional_mapping`) and after `config_filename = config_path.name`, add:

```python
_warn_unknown_keys(data_raw, _DATA_KEYS, 'data', config_filename)
_warn_unknown_keys(model_raw, _MODEL_KEYS, 'model', config_filename)
if training_raw is not None:
    _warn_unknown_keys(training_raw, _TRAINING_KEYS, 'training', config_filename)
if scoring_raw is not None:
    _warn_unknown_keys(scoring_raw, _SCORING_KEYS, 'scoring', config_filename)
_warn_unknown_keys(output_raw, _OUTPUT_KEYS, 'output', config_filename)
```

- [ ] **Step 5: Run all config tests**

```bash
python -m pytest tests/test_config.py -v
```

Expected: all tests pass, including the new unknown-key test.

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/allostery/config.py tests/test_config.py
git commit -m "feat: warn on unrecognized config keys to catch typos early"
```
