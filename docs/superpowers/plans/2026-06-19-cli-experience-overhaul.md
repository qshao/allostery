# CLI Experience Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `allostery` CLI behave consistently for humans and scripts — traceback-free errors with documented exit codes, a `Result`-based presentation layer with `--json`/`--quiet`, and a config-driven `workflow` command that runs train → score → analyze → interpret end to end.

**Architecture:** Two new leaf modules (`cli_errors.py`, `cli_output.py`) define the error taxonomy and the renderable `Result`. Pipeline execution helpers move out of `cli.py` into `pipeline/execute.py` so both `cli.py` and the new `pipeline/workflow.py` can reuse them without a circular import. `cli.py` gains global flags and a single dispatch wrapper; `config.py` gains optional `analyze:`/`interpret:` sections consumed only by `workflow`.

**Tech Stack:** Python 3.11+, stdlib `argparse`/`json`/`dataclasses`, the existing `allostery.config`, `allostery.network`, `allostery.pipeline`, and `allostery.interpret` modules. No new dependencies.

## Global Constraints

- `from __future__ import annotations` at the top of every new module (repo convention).
- No new dependencies. `--json` uses stdlib `json`; optional LLM backends stay lazily imported.
- **Default-mode output must stay byte-for-byte identical.** Existing `tests/test_cli.py` asserts exact stdout lines for run/train/score (`trained samples=…`, `scored pairs=…`, `completed mode=…`). The presentation layer and global flags are additive — when neither `--json` nor `--quiet` is passed, stdout is unchanged.
- Exit codes: `0` success; `1` user/input error (`ValueError`/`ConfigError`/`FileNotFoundError`); `2` argparse usage error (left to argparse); `3` external/backend error (`ImportError`, network `OSError`).
- API keys come from environment only (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) — never parameters, config, or logs.
- LLM backend selection in config is one of `none|ollama|anthropic|openai`; the standalone `interpret` command stays CSV-driven; the new config sections are consumed **only** by `workflow`.
- Residue label format is `"CHAIN:NUM NAME"`.
- Tests use the existing `fixture_path` fixture and `tests/fixtures/tiny_trajectory.pdb`. No real network or model calls.

---

### Task 1: Error taxonomy and exit-code mapping

**Files:**
- Create: `src/allostery/cli_errors.py`
- Test: `tests/test_cli_errors.py`

**Interfaces:**
- Produces: `USER_ERROR = 1`, `USAGE_ERROR = 2`, `BACKEND_ERROR = 3`; `exit_code_for(exc: BaseException) -> int | None` (returns `None` for unexpected exceptions; chases `__cause__` one level so wrapped errors map to their cause's code).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_errors.py
from __future__ import annotations

from allostery.cli_errors import BACKEND_ERROR, USER_ERROR, exit_code_for
from allostery.config import ConfigError


def test_value_and_config_and_missing_file_are_user_errors() -> None:
    assert exit_code_for(ValueError("bad")) == USER_ERROR
    assert exit_code_for(ConfigError("bad config")) == USER_ERROR
    assert exit_code_for(FileNotFoundError("nope")) == USER_ERROR


def test_import_and_network_errors_are_backend_errors() -> None:
    assert exit_code_for(ImportError("no anthropic")) == BACKEND_ERROR
    assert exit_code_for(ConnectionError("refused")) == BACKEND_ERROR
    assert exit_code_for(OSError("socket")) == BACKEND_ERROR


def test_unexpected_returns_none() -> None:
    assert exit_code_for(KeyError("x")) is None


def test_chases_cause_one_level() -> None:
    inner = ImportError("no openai")
    outer = RuntimeError("workflow failed")
    outer.__cause__ = inner
    assert exit_code_for(outer) == BACKEND_ERROR
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_errors.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'allostery.cli_errors'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/allostery/cli_errors.py
from __future__ import annotations

USER_ERROR = 1
USAGE_ERROR = 2
BACKEND_ERROR = 3


def _direct_code(exc: BaseException) -> int | None:
    # FileNotFoundError is an OSError subclass but is a user error, so check it first.
    if isinstance(exc, (ValueError, FileNotFoundError)):
        return USER_ERROR
    if isinstance(exc, (ImportError, OSError)):
        return BACKEND_ERROR
    return None


def exit_code_for(exc: BaseException) -> int | None:
    code = _direct_code(exc)
    if code is not None:
        return code
    cause = exc.__cause__
    if cause is not None:
        return _direct_code(cause)
    return None


__all__ = ["BACKEND_ERROR", "USAGE_ERROR", "USER_ERROR", "exit_code_for"]
```

(`ConnectionError`/`TimeoutError` are `OSError` subclasses, so they map to `BACKEND_ERROR` via the `OSError` branch.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_errors.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/allostery/cli_errors.py tests/test_cli_errors.py
git commit -m "feat: add CLI error taxonomy and exit-code mapping"
```

---

### Task 2: Result dataclass and renderers

**Files:**
- Create: `src/allostery/cli_output.py`
- Test: `tests/test_cli_output.py`

**Interfaces:**
- Produces: `Result` dataclass (`command: str`, `status: str = "ok"`, `summary: str = ""`, `data: dict = {}`, `artifacts: list[Path] = []`, `error: str | None = None`); `format_result(result: Result, *, json_mode: bool = False, quiet: bool = False) -> tuple[str, str]` returning `(stdout_text, stderr_text)`, each possibly `""`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_output.py
from __future__ import annotations

import json
from pathlib import Path

from allostery.cli_output import Result, format_result


def _ok() -> Result:
    return Result(
        command="interpret",
        summary="interpret candidates={'hubs': 3} json=a.json md=a.md",
        data={"counts": {"hubs": 3}},
        artifacts=[Path("a.json"), Path("a.md")],
    )


def test_default_mode_emits_summary_on_stdout() -> None:
    out, err = format_result(_ok())
    assert out == "interpret candidates={'hubs': 3} json=a.json md=a.md"
    assert err == ""


def test_quiet_mode_emits_only_artifact_paths() -> None:
    out, err = format_result(_ok(), quiet=True)
    assert out == "a.json\na.md"
    assert err == ""


def test_json_mode_emits_parseable_object() -> None:
    out, err = format_result(_ok(), json_mode=True)
    payload = json.loads(out)
    assert payload["status"] == "ok"
    assert payload["command"] == "interpret"
    assert payload["artifacts"] == ["a.json", "a.md"]
    assert payload["data"] == {"counts": {"hubs": 3}}
    assert err == ""


def test_error_result_goes_to_stderr_in_default_mode() -> None:
    result = Result(command="analyze", status="error", error="no such file: x.csv")
    out, err = format_result(result)
    assert out == ""
    assert err == "no such file: x.csv"


def test_error_result_in_json_mode_is_on_stdout() -> None:
    result = Result(command="analyze", status="error", error="boom")
    out, err = format_result(result, json_mode=True)
    payload = json.loads(out)
    assert payload["status"] == "error"
    assert payload["error"] == "boom"
    assert err == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_output.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'allostery.cli_output'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/allostery/cli_output.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Result:
    command: str
    status: str = "ok"
    summary: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)
    error: str | None = None


def format_result(
    result: Result,
    *,
    json_mode: bool = False,
    quiet: bool = False,
) -> tuple[str, str]:
    if json_mode:
        payload = {
            "command": result.command,
            "status": result.status,
            "summary": result.summary,
            "data": result.data,
            "artifacts": [str(path) for path in result.artifacts],
            "error": result.error,
        }
        return json.dumps(payload, indent=2), ""

    if result.status == "error":
        return "", result.error or "error"

    if quiet:
        return "\n".join(str(path) for path in result.artifacts), ""

    return result.summary, ""


__all__ = ["Result", "format_result"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_output.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/allostery/cli_output.py tests/test_cli_output.py
git commit -m "feat: add Result dataclass and CLI renderers"
```

---

### Task 3: Extract pipeline execution helpers (pure refactor)

**Files:**
- Create: `src/allostery/pipeline/execute.py`
- Modify: `src/allostery/cli.py` (replace the bodies of `_run_train`/`_run_score`/`_serialize_config` with calls into `execute`)
- Test: `tests/test_pipeline_execute.py`

**Interfaces:**
- Produces: `run_training(config: AppConfig) -> Any` (returns the family's train result; every variant has `.num_samples`); `run_scoring(config: AppConfig) -> int` (writes the scores CSV, returns the pair count); `serialize_config(config: AppConfig) -> dict[str, Any]`. These do **not** print.
- Consumes (in `cli.py`): the three functions above.

This is a behavior-preserving move so `workflow` can reuse training/scoring without importing `cli.py`. Output stays identical because `cli.py` keeps printing the same lines around these calls.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_execute.py
from __future__ import annotations

from pathlib import Path

from allostery.config import load_config
from allostery.pipeline.execute import run_scoring, run_training, serialize_config


def _run_config(tmp_path: Path, fixture_path: Path) -> Path:
    checkpoint = tmp_path / "model.pt"
    scores = tmp_path / "scores.csv"
    text = "\n".join([
        "mode: run",
        "data:",
        f"  pdb_path: {fixture_path / 'tiny_trajectory.pdb'}",
        "  window_size: 3",
        "  horizon_size: 1",
        "  stride: 1",
        "model:",
        "  family: influence",
        "  hidden_dim: 8",
        "  residue_layers: 2",
        "  pair_layers: 1",
        "  dropout: 0.0",
        "training:",
        "  epochs: 1",
        "  learning_rate: 0.01",
        "  consistency_weight: 0.0",
        "  verbose: false",
        "scoring:",
        "  top_k: 3",
        "output:",
        f"  model_path: {checkpoint}",
        f"  score_csv_path: {scores}",
    ])
    path = tmp_path / "run.yaml"
    path.write_text(text + "\n", encoding="utf-8")
    return path


def test_run_training_and_scoring(tmp_path: Path, fixture_path: Path) -> None:
    config = load_config(_run_config(tmp_path, fixture_path))
    result = run_training(config)
    assert result.num_samples >= 1
    assert config.output.model_path.exists()
    count = run_scoring(config)
    assert count == 3
    assert config.output.score_csv_path.exists()


def test_serialize_config_is_json_safe(tmp_path: Path, fixture_path: Path) -> None:
    import json
    config = load_config(_run_config(tmp_path, fixture_path))
    json.dumps(serialize_config(config))  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_execute.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'allostery.pipeline.execute'`

- [ ] **Step 3: Write minimal implementation**

Create `src/allostery/pipeline/execute.py` by moving the logic currently in `cli.py`'s `_run_train`, `_run_score`, `_serialize_config`, and `_serialize_value`, **removing the `print(...)` statements** (the CLI will print). Keep every other line identical:

```python
# src/allostery/pipeline/execute.py
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
```

Now update `cli.py` so its helpers delegate (keeping the prints so default output is unchanged). Replace the bodies of `_run_train`, `_run_score`, `_run_run`, `_serialize_config`, and `_serialize_value` (and drop the now-unused pipeline imports they used) with:

```python
# in src/allostery/cli.py — replace the import block lines 10-19 region with the slimmer set
from allostery.config import AppConfig, load_config
from allostery.pipeline.analyze import run_network_analysis
from allostery.pipeline.execute import run_scoring, run_training, serialize_config
from allostery.pipeline.interpret import run_interpretation
```

```python
# in src/allostery/cli.py — replace _run_train/_run_score/_run_run/_serialize_* with:
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
```

Delete `_serialize_config`/`_serialize_value` from `cli.py` (they now live in `execute.py`).

- [ ] **Step 4: Run test to verify it passes, and confirm no regression**

Run: `pytest tests/test_pipeline_execute.py tests/test_cli.py -v`
Expected: PASS (new execute tests plus all existing CLI tests — default output is unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/allostery/pipeline/execute.py src/allostery/cli.py tests/test_pipeline_execute.py
git commit -m "refactor: extract pipeline execution helpers into pipeline/execute"
```

---

### Task 4: Global flags, dispatch wrapper, and presentation routing

**Files:**
- Modify: `src/allostery/cli.py`
- Test: `tests/test_cli_presentation.py`

**Interfaces:**
- Consumes: `exit_code_for` (Task 1), `Result`/`format_result` (Task 2), `run_training`/`run_scoring` (Task 3), existing `run_network_analysis`/`run_interpretation`.
- Produces: `--debug`, `--quiet`, `--json` global flags (`--quiet`/`--json` mutually exclusive); `_dispatch(args) -> Result`; a `main` that wraps `_dispatch` in try/except, maps exceptions via `exit_code_for`, and renders via `format_result`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_presentation.py
from __future__ import annotations

import json
from pathlib import Path

from allostery.cli import main


def _write_scores(path: Path) -> None:
    header = ("rank,score,residue_i_index,residue_i_chain,residue_i_number,residue_i_name,"
              "residue_j_index,residue_j_chain,residue_j_number,residue_j_name\n")
    rows = ["1,0.9,0,A,1,GLY,1,A,2,GLY", "2,0.8,1,A,2,GLY,2,A,3,GLY", "3,0.7,2,A,3,GLY,3,A,4,GLY"]
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def test_analyze_missing_file_is_clean_user_error(tmp_path: Path, capsys) -> None:
    code = main(["analyze", str(tmp_path / "nope.csv")])
    captured = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in captured.err
    assert captured.err.strip() != ""


def test_debug_flag_reraises(tmp_path: Path) -> None:
    import pytest
    with pytest.raises(Exception):
        main(["--debug", "analyze", str(tmp_path / "nope.csv")])


def test_interpret_json_mode_emits_object(tmp_path: Path, capsys) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)
    code = main(["--json", "interpret", str(scores),
                 "--out-json", str(tmp_path / "o.json"), "--out-md", str(tmp_path / "o.md")])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["command"] == "interpret"
    assert payload["status"] == "ok"
    assert str(tmp_path / "o.json") in payload["artifacts"]


def test_interpret_quiet_mode_emits_only_artifacts(tmp_path: Path, capsys) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)
    code = main(["--quiet", "interpret", str(scores),
                 "--out-json", str(tmp_path / "o.json"), "--out-md", str(tmp_path / "o.md")])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.splitlines() == [str(tmp_path / "o.json"), str(tmp_path / "o.md")]


def test_json_and_quiet_are_mutually_exclusive(tmp_path: Path) -> None:
    import pytest
    with pytest.raises(SystemExit) as exc:
        main(["--json", "--quiet", "analyze", str(tmp_path / "x.csv")])
    assert exc.value.code == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_presentation.py -v`
Expected: FAIL — global flags not defined / `analyze` raises an uncaught exception (no wrapper yet).

- [ ] **Step 3: Write minimal implementation**

In `build_parser()` add the global flags right after the `--version` line (line 27):

```python
    parser.add_argument('--debug', action='store_true',
                        help='Show full tracebacks instead of a clean error message')
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('--json', action='store_true',
                              help='Emit a single JSON object on stdout (for scripts)')
    output_group.add_argument('--quiet', action='store_true',
                              help='Suppress summaries; print only artifact paths')
```

Add imports near the top of `cli.py`:

```python
from allostery.cli_errors import USER_ERROR, exit_code_for
from allostery.cli_output import Result, format_result
```

Replace the whole body of `main(...)` (everything after `args = build_parser().parse_args(effective)`) and the per-command branches with a dispatch wrapper plus a `_dispatch` function. The new `main`:

```python
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
    if args.command == 'check':
        config = load_config(args.config_path)
        return Result(
            command='check',
            summary=f'Config OK: mode={config.mode}, family={config.model.family}',
            data={'mode': config.mode, 'family': config.model.family},
        )

    if args.command == 'analyze':
        report = run_network_analysis(
            scores_csv=args.scores_csv,
            top_k=args.top_k,
            source=args.source,
            sink=args.sink,
            top_paths=args.top_paths,
            top_hubs=args.top_hubs,
        )
        return Result(command='analyze', summary=report)

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
        result = run_training(config)
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
```

Delete the now-superseded `_run_train`/`_run_score`/`_run_run` helpers from `cli.py` (their logic now lives in `_dispatch` + `execute`). The empty-help path now raises `ValueError('no command given')` which the wrapper renders cleanly with exit 1.

Note: this changes nothing in default mode — `run`/`train`/`score` still print the same lines (now via `_dispatch`'s summary), and `check`/`analyze`/`interpret` print the same strings as before.

- [ ] **Step 4: Run test to verify it passes, and confirm no regression**

Run: `pytest tests/test_cli_presentation.py tests/test_cli.py tests/test_cli_interpret.py -v`
Expected: PASS (new presentation tests plus all existing CLI tests, whose exact stdout assertions still hold).

- [ ] **Step 5: Commit**

```bash
git add src/allostery/cli.py tests/test_cli_presentation.py
git commit -m "feat: add global flags, dispatch wrapper, and presentation routing"
```

---

### Task 5: Empty-graph guard for analyze and interpret

**Files:**
- Modify: `src/allostery/pipeline/analyze.py`
- Modify: `src/allostery/pipeline/interpret.py`
- Test: `tests/test_empty_graph_guard.py`

**Interfaces:**
- Produces: both `run_network_analysis` and `run_interpretation` raise `ValueError` with an "increase --top-k" message when the built graph has zero nodes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_empty_graph_guard.py
from __future__ import annotations

from pathlib import Path

import pytest

from allostery.pipeline.analyze import run_network_analysis
from allostery.pipeline.interpret import run_interpretation


def _write_scores(path: Path) -> None:
    header = ("rank,score,residue_i_index,residue_i_chain,residue_i_number,residue_i_name,"
              "residue_j_index,residue_j_chain,residue_j_number,residue_j_name\n")
    path.write_text(header + "1,0.9,0,A,1,GLY,1,A,2,GLY\n", encoding="utf-8")


def test_analyze_empty_graph_raises(tmp_path: Path) -> None:
    scores = tmp_path / "s.csv"
    _write_scores(scores)
    with pytest.raises(ValueError, match="top-k"):
        run_network_analysis(scores, top_k=0)


def test_interpret_empty_graph_raises(tmp_path: Path) -> None:
    scores = tmp_path / "s.csv"
    _write_scores(scores)
    with pytest.raises(ValueError, match="top-k"):
        run_interpretation(scores, out_json=tmp_path / "o.json", out_md=tmp_path / "o.md", top_k=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_empty_graph_guard.py -v`
Expected: FAIL — no `ValueError` raised (empty graph currently produces empty output, not an error).

- [ ] **Step 3: Write minimal implementation**

In `src/allostery/pipeline/analyze.py`, after `net = build_graph(rows, top_k=top_k)`:

```python
    if net.num_nodes == 0:
        raise ValueError(
            "No edges in the network after top-k filtering; increase --top-k "
            "or check the scores CSV."
        )
```

In `src/allostery/pipeline/interpret.py`, after `net = build_graph(rows, top_k=top_k)`:

```python
    if net.num_nodes == 0:
        raise ValueError(
            "No edges in the network after top-k filtering; increase --top-k "
            "or check the scores CSV."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_empty_graph_guard.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/allostery/pipeline/analyze.py src/allostery/pipeline/interpret.py tests/test_empty_graph_guard.py
git commit -m "feat: guard analyze/interpret against empty graphs"
```

---

### Task 6: Config `analyze:` and `interpret:` sections

**Files:**
- Modify: `src/allostery/config.py`
- Test: `tests/test_config_workflow_sections.py`

**Interfaces:**
- Produces: `AnalyzeConfig(top_k=20, source=None, sink=None, top_paths=5, top_hubs=10, out_path=None)`; `InterpretConfig(llm='none', llm_model=None, llm_base_url=None, pdb_path=None, top_k=20, top_paths=5, top_hubs=10, out_json=None, out_md=None)`; `AppConfig` gains `analyze: AnalyzeConfig | None = None` and `interpret: InterpretConfig | None = None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_workflow_sections.py
from __future__ import annotations

from pathlib import Path

import pytest

from allostery.config import ConfigError, load_config


def _base(tmp_path: Path, fixture_path: Path, extra: list[str]) -> Path:
    checkpoint = tmp_path / "model.pt"
    scores = tmp_path / "scores.csv"
    lines = [
        "mode: run",
        "data:",
        f"  pdb_path: {fixture_path / 'tiny_trajectory.pdb'}",
        "  window_size: 3",
        "  horizon_size: 1",
        "  stride: 1",
        "model:",
        "  family: influence",
        "  hidden_dim: 8",
        "  residue_layers: 2",
        "  pair_layers: 1",
        "  dropout: 0.0",
        "training:",
        "  epochs: 1",
        "  learning_rate: 0.01",
        "  consistency_weight: 0.0",
        "scoring:",
        "  top_k: 3",
        "output:",
        f"  model_path: {checkpoint}",
        f"  score_csv_path: {scores}",
    ] + extra
    path = tmp_path / "cfg.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_sections_parse(tmp_path: Path, fixture_path: Path) -> None:
    cfg = load_config(_base(tmp_path, fixture_path, [
        "analyze:",
        "  top_k: 10",
        "  top_hubs: 4",
        "interpret:",
        "  llm: none",
        "  top_hubs: 6",
    ]))
    assert cfg.analyze.top_k == 10
    assert cfg.analyze.top_hubs == 4
    assert cfg.interpret.llm == "none"
    assert cfg.interpret.top_hubs == 6


def test_no_sections_default_to_none(tmp_path: Path, fixture_path: Path) -> None:
    cfg = load_config(_base(tmp_path, fixture_path, []))
    assert cfg.analyze is None
    assert cfg.interpret is None


def test_bad_llm_enum_rejected(tmp_path: Path, fixture_path: Path) -> None:
    with pytest.raises(ConfigError, match="interpret.llm"):
        load_config(_base(tmp_path, fixture_path, ["interpret:", "  llm: gpt5"]))


def test_lone_source_rejected(tmp_path: Path, fixture_path: Path) -> None:
    with pytest.raises(ConfigError, match="source"):
        load_config(_base(tmp_path, fixture_path, ["analyze:", "  source: A:1 GLY"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_workflow_sections.py -v`
Expected: FAIL — `AttributeError: 'AppConfig' object has no attribute 'analyze'`

- [ ] **Step 3: Write minimal implementation**

In `src/allostery/config.py`, add the key frozensets near the existing ones:

```python
_ANALYZE_KEYS: frozenset[str] = frozenset({
    'top_k', 'source', 'sink', 'top_paths', 'top_hubs', 'out_path',
})
_INTERPRET_KEYS: frozenset[str] = frozenset({
    'llm', 'llm_model', 'llm_base_url', 'pdb_path',
    'top_k', 'top_paths', 'top_hubs', 'out_json', 'out_md',
})
```

Add the dataclasses (after `ScoringConfig`):

```python
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
```

Extend `AppConfig` with two new defaulted fields (append after `output`):

```python
    analyze: AnalyzeConfig | None = None
    interpret: InterpretConfig | None = None
```

In `load_config`, after the `output_raw` mapping is read, parse the optional sections:

```python
    analyze_raw = _require_optional_mapping(raw, 'analyze')
    interpret_raw = _require_optional_mapping(raw, 'interpret')
    if analyze_raw is not None:
        _warn_unknown_keys(analyze_raw, _ANALYZE_KEYS, 'analyze', config_filename)
    if interpret_raw is not None:
        _warn_unknown_keys(interpret_raw, _INTERPRET_KEYS, 'interpret', config_filename)

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
```

Pass them into the `AppConfig(...)` constructor: add `analyze=analyze_cfg, interpret=interpret_cfg,` after `output=...`.

Add validation rules at the end of `validate_config` (before the `if errors:` block):

```python
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
```

Add `AnalyzeConfig` and `InterpretConfig` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config_workflow_sections.py tests/test_config.py -v`
Expected: PASS (new section tests plus all existing config tests).

- [ ] **Step 5: Commit**

```bash
git add src/allostery/config.py tests/test_config_workflow_sections.py
git commit -m "feat: add analyze/interpret config sections with validation"
```

---

### Task 7: Analyze writes its report to a file

**Files:**
- Modify: `src/allostery/pipeline/analyze.py`
- Test: `tests/test_analyze_write.py`

**Interfaces:**
- Produces: `run_network_analysis(scores_csv, top_k=20, source=None, sink=None, top_paths=5, top_hubs=10, out_path=None) -> str` — when `out_path` is given, writes the report text there (creating parent dirs) and still returns the string.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analyze_write.py
from __future__ import annotations

from pathlib import Path

from allostery.pipeline.analyze import run_network_analysis


def _write_scores(path: Path) -> None:
    header = ("rank,score,residue_i_index,residue_i_chain,residue_i_number,residue_i_name,"
              "residue_j_index,residue_j_chain,residue_j_number,residue_j_name\n")
    rows = ["1,0.9,0,A,1,GLY,1,A,2,GLY", "2,0.8,1,A,2,GLY,2,A,3,GLY"]
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def test_analyze_writes_report_file(tmp_path: Path) -> None:
    scores = tmp_path / "s.csv"
    _write_scores(scores)
    out = tmp_path / "nested" / "network.txt"
    report = run_network_analysis(scores, top_k=5, out_path=out)
    assert out.exists()
    assert out.read_text(encoding="utf-8") == report
    assert "Allosteric Network" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyze_write.py -v`
Expected: FAIL — `TypeError: run_network_analysis() got an unexpected keyword argument 'out_path'`

- [ ] **Step 3: Write minimal implementation**

Replace the body of `run_network_analysis` in `src/allostery/pipeline/analyze.py`:

```python
def run_network_analysis(
    scores_csv: str | Path,
    top_k: int = 20,
    source: str | None = None,
    sink: str | None = None,
    top_paths: int = 5,
    top_hubs: int = 10,
    out_path: str | Path | None = None,
) -> str:
    """Read a scores CSV, build the allosteric network, and return a text report."""
    rows = read_scores_csv(scores_csv)
    net = build_graph(rows, top_k=top_k)
    if net.num_nodes == 0:
        raise ValueError(
            "No edges in the network after top-k filtering; increase --top-k "
            "or check the scores CSV."
        )
    report = format_report(
        net,
        source_label=source,
        sink_label=sink,
        top_hubs=top_hubs,
        top_paths=top_paths,
    )
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
    return report
```

(The empty-graph guard from Task 5 is folded in here as the function is rewritten — keep it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyze_write.py tests/test_empty_graph_guard.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/allostery/pipeline/analyze.py tests/test_analyze_write.py
git commit -m "feat: let analyze write its report to a file"
```

---

### Task 8: Workflow orchestrator

**Files:**
- Create: `src/allostery/pipeline/workflow.py`
- Test: `tests/test_pipeline_workflow.py`

**Interfaces:**
- Consumes: `run_training`/`run_scoring` (Task 3), `run_network_analysis` (Task 7), `run_interpretation`, `AppConfig`/`AnalyzeConfig`/`InterpretConfig` (Task 6), `Result` (Task 2).
- Produces: `WorkflowError(RuntimeError)` (carries `.stage` and `.artifacts`, chains the cause via `from`); `run_workflow(config: AppConfig, *, backend=None, progress=None) -> Result` (command `"workflow"`, `data["stages"]` lists the stages that ran, `artifacts` lists every file written).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_workflow.py
from __future__ import annotations

from pathlib import Path

import pytest

from allostery.config import load_config
from allostery.pipeline.workflow import WorkflowError, run_workflow


def _cfg(tmp_path: Path, fixture_path: Path, extra: list[str]) -> Path:
    lines = [
        "mode: run",
        "data:",
        f"  pdb_path: {fixture_path / 'tiny_trajectory.pdb'}",
        "  window_size: 3",
        "  horizon_size: 1",
        "  stride: 1",
        "model:",
        "  family: influence",
        "  hidden_dim: 8",
        "  residue_layers: 2",
        "  pair_layers: 1",
        "  dropout: 0.0",
        "training:",
        "  epochs: 1",
        "  learning_rate: 0.01",
        "  consistency_weight: 0.0",
        "  verbose: false",
        "scoring:",
        "  top_k: 3",
        "output:",
        f"  model_path: {tmp_path / 'model.pt'}",
        f"  score_csv_path: {tmp_path / 'scores.csv'}",
    ] + extra
    path = tmp_path / "cfg.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class _FakeBackend:
    def generate_json(self, system, user, schema):
        return {"summary": "s", "mechanism_hypothesis": "m", "key_residues": [],
                "confidence": "low", "parametric": False, "caveats": "c"}


def test_full_workflow_runs_all_stages(tmp_path: Path, fixture_path: Path) -> None:
    config = load_config(_cfg(tmp_path, fixture_path, [
        "analyze:", "  top_k: 3",
        "interpret:", "  llm: ollama", "  top_hubs: 3",
    ]))
    stages: list[str] = []
    result = run_workflow(config, backend=_FakeBackend(), progress=stages.append)
    assert result.command == "workflow"
    assert result.data["stages"] == ["train", "score", "analyze", "interpret"]
    assert stages == ["train", "score", "analyze", "interpret"]
    assert (tmp_path / "scores.csv").exists()
    assert (tmp_path / "scores.network.txt").exists()
    assert (tmp_path / "scores.interpret.json").exists()
    assert (tmp_path / "scores.interpret.md").exists()


def test_workflow_without_post_sections_just_runs_pipeline(tmp_path: Path, fixture_path: Path) -> None:
    config = load_config(_cfg(tmp_path, fixture_path, []))
    result = run_workflow(config)
    assert result.data["stages"] == ["train", "score"]
    assert (tmp_path / "scores.csv").exists()


def test_workflow_backend_failure_preserves_prior_artifacts(tmp_path: Path, fixture_path: Path) -> None:
    config = load_config(_cfg(tmp_path, fixture_path, [
        "analyze:", "  top_k: 3",
        "interpret:", "  llm: ollama",
    ]))

    class _Boom:
        def generate_json(self, system, user, schema):
            raise ImportError("backend unavailable")

    with pytest.raises(WorkflowError) as exc:
        run_workflow(config, backend=_Boom())
    assert exc.value.stage == "interpret"
    assert (tmp_path / "scores.csv").exists()         # preserved
    assert (tmp_path / "scores.network.txt").exists()  # preserved
    assert isinstance(exc.value.__cause__, ImportError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_workflow.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'allostery.pipeline.workflow'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/allostery/pipeline/workflow.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_workflow.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/allostery/pipeline/workflow.py tests/test_pipeline_workflow.py
git commit -m "feat: add config-driven workflow orchestrator"
```

---

### Task 9: `workflow` CLI subcommand

**Files:**
- Modify: `src/allostery/cli.py`
- Test: `tests/test_cli_workflow.py`

**Interfaces:**
- Consumes: `run_workflow` (Task 8).
- Produces: a `workflow` subcommand (`allostery workflow <config.yaml>`) dispatched in `_dispatch`, with stage banners on stderr (suppressed under `--quiet`/`--json`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_workflow.py
from __future__ import annotations

import json
from pathlib import Path

from allostery.cli import main


def _cfg(tmp_path: Path, fixture_path: Path, extra: list[str]) -> Path:
    lines = [
        "mode: run",
        "data:",
        f"  pdb_path: {fixture_path / 'tiny_trajectory.pdb'}",
        "  window_size: 3",
        "  horizon_size: 1",
        "  stride: 1",
        "model:",
        "  family: influence",
        "  hidden_dim: 8",
        "  residue_layers: 2",
        "  pair_layers: 1",
        "  dropout: 0.0",
        "training:",
        "  epochs: 1",
        "  learning_rate: 0.01",
        "  consistency_weight: 0.0",
        "  verbose: false",
        "scoring:",
        "  top_k: 3",
        "output:",
        f"  model_path: {tmp_path / 'model.pt'}",
        f"  score_csv_path: {tmp_path / 'scores.csv'}",
    ] + extra
    path = tmp_path / "cfg.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_workflow_command_runs_end_to_end(tmp_path: Path, fixture_path: Path, capsys) -> None:
    config = _cfg(tmp_path, fixture_path, ["analyze:", "  top_k: 3", "interpret:", "  llm: none"])
    code = main(["workflow", str(config)])
    captured = capsys.readouterr()
    assert code == 0
    assert (tmp_path / "scores.csv").exists()
    assert (tmp_path / "scores.network.txt").exists()
    assert (tmp_path / "scores.interpret.json").exists()
    assert "workflow complete" in captured.out


def test_workflow_command_json_mode(tmp_path: Path, fixture_path: Path, capsys) -> None:
    config = _cfg(tmp_path, fixture_path, ["interpret:", "  llm: none"])
    code = main(["--json", "workflow", str(config)])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["command"] == "workflow"
    assert payload["data"]["stages"] == ["train", "score", "interpret"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_workflow.py -v`
Expected: FAIL — argparse error: `invalid choice: 'workflow'` / `workflow` not a known subcommand.

- [ ] **Step 3: Write minimal implementation**

In `cli.py`, add `'workflow'` to the subcommands set:

```python
_SUBCOMMANDS = frozenset({'run', 'analyze', 'check', 'interpret', 'workflow'})
```

Add the subparser in `build_parser()` (before `return parser`):

```python
    workflow_parser = subparsers.add_parser(
        'workflow', help='Run train/score then analyze+interpret end to end from one config')
    workflow_parser.add_argument('config_path', help='Path to YAML config file')
```

Add the import near the other pipeline imports:

```python
from allostery.pipeline.workflow import run_workflow
```

Add a branch at the start of `_dispatch` (before the `check` branch):

```python
    if args.command == 'workflow':
        import sys as _sys
        config = load_config(args.config_path)
        emit_progress = not args.json and not args.quiet
        progress = (lambda stage: print(f'[{stage}] ...', file=_sys.stderr)) if emit_progress else None
        return run_workflow(config, progress=progress)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_workflow.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

Run: `pytest -q`
Expected: PASS (all existing tests plus every new test in this plan).

```bash
git add src/allostery/cli.py tests/test_cli_workflow.py
git commit -m "feat: add 'workflow' CLI subcommand"
```

---

### Task 10: Documentation — README and help epilogs

**Files:**
- Modify: `README.md`
- Modify: `src/allostery/cli.py` (help text for the new flags/command already added; verify wording)
- Test: `tests/test_cli_help.py`

**Interfaces:**
- Produces: README "Commands" section lists `interpret` and `workflow`; a smoke test confirms `--help` mentions both and the global flags.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_help.py
from __future__ import annotations

import pytest

from allostery.cli import build_parser


def test_help_lists_new_commands_and_flags(capsys) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    text = capsys.readouterr().out
    assert "interpret" in text
    assert "workflow" in text
    assert "--json" in text
    assert "--quiet" in text


def test_readme_documents_interpret_and_workflow() -> None:
    from pathlib import Path
    readme = Path(__file__).resolve().parent.parent / "README.md"
    body = readme.read_text(encoding="utf-8")
    assert "allostery interpret" in body
    assert "allostery workflow" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_help.py -v`
Expected: FAIL — `test_readme_documents_interpret_and_workflow` fails (README has no `allostery interpret`/`allostery workflow`); the help test passes already since the flags/commands exist.

- [ ] **Step 3: Write minimal implementation**

In `README.md`, replace the "## Commands" intro list (the block that currently says "The CLI has three commands:") with an updated listing and two new sections:

```markdown
## Commands

The CLI has five commands and three global output flags:

```
allostery run <config.yaml>              # train / score / run pipeline from YAML config
allostery check <config.yaml>            # validate config without running anything
allostery analyze <scores.csv> [options] # post-process: network + channels
allostery interpret <scores.csv> [opts]  # candidate allosteric networks + optional LLM interpretation
allostery workflow <config.yaml>         # run -> analyze -> interpret end to end from one config
allostery --version                      # print version and exit

# Global flags (before the subcommand):
#   --json    emit a single JSON object on stdout (for scripts)
#   --quiet   print only artifact paths
#   --debug   show full tracebacks instead of a clean error message

# Legacy short form (no subcommand) still works:
allostery my_config.yaml
```

Exit codes: `0` success, `1` user/input error, `2` usage error, `3` external/backend error.

### Interpret

Turn a scores CSV into candidate allosteric networks (communities, pathways, hubs,
coupled-pair clusters) plus an optional biological interpretation:

```bash
# Deterministic report only (no LLM)
allostery interpret outputs/scores.csv --pdb path/to/structure.pdb

# With a local Ollama model
allostery interpret outputs/scores.csv --llm ollama --llm-model qwen3

# With a cloud API (key from the environment)
ANTHROPIC_API_KEY=... allostery interpret outputs/scores.csv --llm anthropic
```

### Workflow

Run the whole pipeline from one config file. Add optional `analyze:` and `interpret:`
sections and `allostery workflow` runs train -> score -> analyze -> interpret:

```yaml
mode: run
# ... data / model / training / scoring / output as usual ...
analyze:
  top_k: 20
interpret:
  llm: none          # none | ollama | anthropic | openai
  top_hubs: 10
```

```bash
allostery workflow config.yaml
```
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_help.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

Run: `pytest -q`
Expected: PASS (entire suite).

```bash
git add README.md tests/test_cli_help.py
git commit -m "docs: document interpret and workflow commands and global flags"
```

---

## Self-Review

**Spec coverage:**
- §3 file layout (`cli_output.py`, `cli_errors.py`, `pipeline/workflow.py`, config sections, analyze write-to-file) → Tasks 1, 2, 5–9. The spec did not call out `pipeline/execute.py`; it was added (Task 3) to let `workflow` reuse training/scoring without a circular import — a refinement, not a scope change. ✓
- §4.1 error taxonomy + exit codes + `--debug` + empty-graph guard → Tasks 1, 4, 5. ✓
- §4.2 `Result` + three render modes, progress to stderr, `--json`/`--quiet` mutually exclusive → Tasks 2, 4, 9. ✓
- §4.3 workflow command + `analyze:`/`interpret:` config + `run_workflow` + stage banners + mode gating → Tasks 6, 8, 9. ✓
- §5 data flow (global flags never reach pipeline funcs; workflow stage list) → Tasks 4, 8. ✓
- §6 edge cases: backend fails mid-workflow (artifacts preserved, exit 3) → Task 8 `WorkflowError` + Task 1 `__cause__` chase; missing dep → Task 1; empty graph → Task 5; `--json --quiet` → Task 4; no-section workflow → Task 8; non-PDB without topology → reuses `load_trajectory` error (exit 1). ✓
- §7 testing strategy → each task ships its tests; backward-compat asserted in Tasks 3 and 4. ✓
- §8 no new dependencies → honored (stdlib only). ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code; every test has real assertions. The stage-banner string `[{stage}] ...` is intentional and concrete. ✓

**Type consistency:**
- `Result(command, status, summary, data, artifacts, error)` defined in Task 2; constructed identically in Tasks 4, 8. ✓
- `exit_code_for` (Task 1) consumed in Task 4; `__cause__` chasing matches `WorkflowError(... ) from exc` in Task 8. ✓
- `run_training(config) -> result.num_samples` / `run_scoring(config) -> int` defined in Task 3; used in Tasks 4 and 8 with the same shape. ✓
- `run_network_analysis(..., out_path=None) -> str` extended in Task 7; called with `out_path=` in Task 8. ✓
- `AnalyzeConfig`/`InterpretConfig` field names (Task 6) match attribute access in Task 8 (`a.top_k`, `it.llm`, `it.out_json`, …). ✓
- `run_workflow(config, *, backend=None, progress=None) -> Result` (Task 8) matches the call in Task 9. ✓
- `run_interpretation(..., backend=...)` already exists (current `pipeline/interpret.py`) and is called with the same keyword in Task 8. ✓
