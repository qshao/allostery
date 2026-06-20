# UX Improvements: Training Progress & Richer Analysis Output — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add live training progress to every `allostery run` / `workflow` invocation and enrich `allostery analyze` with auto-threshold detection, a score histogram, and optional PyMOL script export.

**Architecture:** Training progress lives in a new `TrainingProgress` class (stderr only, TTY-aware) wired via an optional `progress_fn` callback added to all three training functions and threaded through `execute.py`, `workflow.py`, and `cli.py`. Analysis enrichment adds two new functions to `network.py` (threshold + histogram) and a new `pymol_export.py` module, all wired into `run_network_analysis` and the `analyze` CLI subparser.

**Tech Stack:** Python 3.11+, stdlib (`time`, `sys`, `io`), numpy (already required), no new dependencies.

## Global Constraints

- `from __future__ import annotations` at the top of every new module and every modified module that doesn't already have it.
- No new runtime dependencies — stdlib + numpy only.
- All training progress output goes to **stderr**; stdout is never modified (preserves `--json` and piped workflows).
- TDD: write the failing test, verify it fails, implement, verify it passes, commit.
- Frequent small commits — one per task.
- `progress_fn` takes priority over `verbose` when both are set; existing `verbose=True/False` behavior is unchanged when `progress_fn=None`.

---

## File Map

| Action | Path |
|---|---|
| Create | `src/allostery/pipeline/progress.py` |
| Create | `src/allostery/pipeline/pymol_export.py` |
| Modify | `src/allostery/pipeline/influence_train.py` |
| Modify | `src/allostery/pipeline/cri_train.py` |
| Modify | `src/allostery/pipeline/train.py` |
| Modify | `src/allostery/pipeline/execute.py` |
| Modify | `src/allostery/pipeline/workflow.py` |
| Modify | `src/allostery/cli.py` |
| Modify | `src/allostery/network.py` |
| Modify | `src/allostery/pipeline/analyze.py` |
| Create | `tests/test_progress.py` |
| Append | `tests/test_training_progress.py` |
| Create | `tests/test_threshold.py` |
| Create | `tests/test_pymol_export.py` |
| Create | `tests/test_cli_analyze_pml.py` |
| Create | `tests/test_cli_training_progress.py` |

---

### Task 1: TrainingProgress class

**Files:**
- Create: `src/allostery/pipeline/progress.py`
- Create: `tests/test_progress.py`

**Interfaces:**
- Produces:
  - `TrainingProgress(total_epochs, *, quiet=False, tty=None, _stderr=None)` — context manager
  - `TrainingProgress.update(epoch: int, train_loss: float, val_loss: float | None = None) -> None` — called once per epoch
  - `TrainingProgress.finish(best_epoch: int | None = None, best_val_loss: float | None = None) -> None` — called after training completes
  - `_fmt_s(seconds: int) -> str` — formats seconds as `"45s"` or `"1m30s"` (exported for tests)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_progress.py
from __future__ import annotations

import io
import time

import pytest

from allostery.pipeline.progress import TrainingProgress, _fmt_s


def _make(total: int, *, quiet: bool = False, tty: bool = False) -> tuple[TrainingProgress, io.StringIO]:
    buf = io.StringIO()
    tp = TrainingProgress(total, quiet=quiet, tty=tty, _stderr=buf)
    tp._start = time.monotonic()
    tp._last_start = tp._start
    return tp, buf


def test_non_tty_writes_epoch_line_per_update() -> None:
    tp, buf = _make(5, tty=False)
    tp.update(1, 1.2345)
    line = buf.getvalue()
    assert "epoch 1/5" in line
    assert "train=1.2345" in line
    assert "elapsed=" in line
    assert "ETA=" in line
    assert line.endswith("\n")


def test_tty_uses_carriage_return() -> None:
    tp, buf = _make(5, tty=True)
    tp.update(1, 1.2345)
    assert buf.getvalue().startswith("\r")


def test_tty_does_not_end_with_newline_mid_training() -> None:
    tp, buf = _make(5, tty=True)
    tp.update(1, 1.0)
    assert "\n" not in buf.getvalue()


def test_quiet_produces_no_output() -> None:
    tp, buf = _make(5, quiet=True, tty=False)
    with tp:
        tp.update(1, 1.0)
    tp.finish()
    assert buf.getvalue() == ""


def test_update_includes_val_loss_when_given() -> None:
    tp, buf = _make(5, tty=False)
    tp.update(1, 1.0, val_loss=2.0)
    assert "val=2.0000" in buf.getvalue()


def test_finish_prints_summary_with_best_epoch() -> None:
    tp, buf = _make(10, tty=False)
    tp.update(1, 1.0, 1.5)
    buf.truncate(0)
    buf.seek(0)
    tp.finish(best_epoch=3, best_val_loss=1.1234)
    summary = buf.getvalue()
    assert "Training complete" in summary
    assert "best epoch 4" in summary  # 0-indexed 3 → displayed as 4
    assert "1.1234" in summary


def test_finish_without_best_epoch_omits_clause() -> None:
    tp, buf = _make(5, tty=False)
    tp.update(1, 1.0)
    buf.truncate(0)
    buf.seek(0)
    tp.finish()
    assert "best epoch" not in buf.getvalue()
    assert "Training complete" in buf.getvalue()


def test_context_manager_writes_newline_in_tty_mode_on_exit() -> None:
    tp, buf = _make(5, tty=True)
    with tp:
        tp.update(1, 1.0)
    assert "\n" in buf.getvalue()


def test_fmt_s_under_60() -> None:
    assert _fmt_s(45) == "45s"


def test_fmt_s_exactly_60() -> None:
    assert _fmt_s(60) == "1m00s"


def test_fmt_s_over_60() -> None:
    assert _fmt_s(90) == "1m30s"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_progress.py -v
```

Expected: `ImportError: cannot import name 'TrainingProgress'`

- [ ] **Step 3: Implement `src/allostery/pipeline/progress.py`**

```python
from __future__ import annotations

import sys
import time
from typing import IO


class TrainingProgress:
    """Prints training epoch progress to stderr. TTY-aware: overwrites in place or one line per epoch."""

    def __init__(
        self,
        total_epochs: int,
        *,
        quiet: bool = False,
        tty: bool | None = None,
        _stderr: IO[str] | None = None,
    ) -> None:
        self._total = total_epochs
        self._quiet = quiet
        self._tty = sys.stderr.isatty() if tty is None else tty
        self._stderr = _stderr if _stderr is not None else sys.stderr
        self._start = 0.0
        self._last_start = 0.0
        self._epoch_times: list[float] = []
        self._output_started = False

    def __enter__(self) -> TrainingProgress:
        self._start = time.monotonic()
        self._last_start = self._start
        return self

    def __exit__(self, *_: object) -> None:
        if self._tty and not self._quiet and self._output_started:
            self._stderr.write("\n")
            self._stderr.flush()

    def update(self, epoch: int, train_loss: float, val_loss: float | None = None) -> None:
        if self._quiet:
            return
        now = time.monotonic()
        self._epoch_times.append(now - self._last_start)
        self._last_start = now
        total_elapsed = int(now - self._start)
        mean_time = sum(self._epoch_times) / len(self._epoch_times)
        eta = int(mean_time * max(self._total - epoch, 0))

        val_part = f"  val={val_loss:.4f}" if val_loss is not None else ""
        self._output_started = True

        if self._tty:
            width = 20
            filled = int(width * epoch / self._total) if self._total > 0 else width
            arrow = ">" if filled < width else ""
            bar = "=" * filled + arrow + " " * (width - filled - len(arrow))
            self._stderr.write(
                f"\r[{bar}] epoch {epoch}/{self._total}"
                f"  train={train_loss:.4f}{val_part}  ETA {_fmt_s(eta)}"
            )
        else:
            self._stderr.write(
                f"epoch {epoch}/{self._total}  train={train_loss:.4f}{val_part}"
                f"  elapsed={_fmt_s(total_elapsed)}  ETA={_fmt_s(eta)}\n"
            )
        self._stderr.flush()

    def finish(
        self,
        best_epoch: int | None = None,
        best_val_loss: float | None = None,
    ) -> None:
        if self._quiet:
            return
        elapsed_str = _fmt_s(int(time.monotonic() - self._start))
        line = f"Training complete: {self._total} epochs in {elapsed_str}"
        if best_epoch is not None and best_val_loss is not None:
            line += f" — best epoch {best_epoch + 1} (val={best_val_loss:.4f})"
        self._stderr.write(line + "\n")
        self._stderr.flush()


def _fmt_s(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m{seconds % 60:02d}s"


__all__ = ["TrainingProgress"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_progress.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/allostery/pipeline/progress.py tests/test_progress.py
git commit -m "feat: add TrainingProgress class for epoch progress reporting"
```

---

### Task 2: Add `progress_fn` callback to the three training functions

**Files:**
- Modify: `src/allostery/pipeline/influence_train.py`
- Modify: `src/allostery/pipeline/cri_train.py`
- Modify: `src/allostery/pipeline/train.py`
- Append: `tests/test_training_progress.py`

**Interfaces:**
- Consumes: nothing from Task 1 (progress_fn is a plain callable)
- Produces:
  - `train_influence_model(..., progress_fn: Callable[[int, float, float | None], None] | None = None)` — new kwarg
  - `train_cri_model(..., progress_fn: Callable[[int, float, float | None], None] | None = None)` — new kwarg
  - `train_relational_model(..., progress_fn: Callable[[int, float, float | None], None] | None = None)` — new kwarg
  - `train_model(..., progress_fn: Callable[[int, float, float | None], None] | None = None)` — new kwarg (passed through to `train_relational_model`)

  Callback contract: called once per epoch with `(epoch_1based: int, train_loss: float, val_loss: float | None)`. The `val_loss` is `None` when no validation split is used.

- [ ] **Step 1: Write the failing tests**

Append these tests to `tests/test_training_progress.py`:

```python
def test_influence_progress_fn_called_per_epoch(fixture_path: Path, capsys) -> None:
    from allostery.pipeline.influence_train import train_influence_model

    calls: list[tuple[int, float, float | None]] = []

    train_influence_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        hidden_dim=8,
        num_encoder_layers=1,
        dropout=0.0,
        epochs=3,
        learning_rate=1e-3,
        sparsity_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        verbose=False,
        progress_fn=lambda e, t, v: calls.append((e, t, v)),
    )

    assert len(calls) == 3
    assert calls[0][0] == 1
    assert calls[2][0] == 3
    assert all(isinstance(c[1], float) for c in calls)
    assert all(c[2] is None for c in calls)  # no validation split


def test_cri_progress_fn_called_per_epoch(fixture_path: Path) -> None:
    from allostery.pipeline.cri_train import train_cri_model

    calls: list[int] = []

    train_cri_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        distance_cutoff=20.0,
        max_neighbors=2,
        edge_types=2,
        hidden_dim=8,
        dropout=0.0,
        epochs=2,
        learning_rate=1e-3,
        entropy_weight=0.0,
        no_edge_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        verbose=False,
        progress_fn=lambda e, t, v: calls.append(e),
    )

    assert calls == [1, 2]


def test_relational_progress_fn_called_per_epoch(fixture_path: Path) -> None:
    from allostery.pipeline.train import train_model

    calls: list[int] = []

    train_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=1,
        horizon_size=1,
        stride=1,
        hidden_dim=8,
        residue_layers=1,
        pair_layers=1,
        dropout=0.0,
        epochs=2,
        learning_rate=1e-3,
        consistency_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        verbose=False,
        progress_fn=lambda e, t, v: calls.append(e),
    )

    assert calls == [1, 2]


def test_progress_fn_takes_priority_over_verbose(fixture_path: Path, capsys) -> None:
    """When progress_fn is set, verbose=True should not print to stdout."""
    from allostery.pipeline.influence_train import train_influence_model

    calls: list[int] = []

    train_influence_model(
        pdb_path=fixture_path / "tiny_trajectory.pdb",
        window_size=3,
        stride=1,
        time_step=1.0,
        hidden_dim=8,
        num_encoder_layers=1,
        dropout=0.0,
        epochs=2,
        learning_rate=1e-3,
        sparsity_weight=0.0,
        validation_fraction=0.0,
        patience=0,
        seed=0,
        device="cpu",
        batch_size=1,
        verbose=True,
        progress_fn=lambda e, t, v: calls.append(e),
    )

    captured = capsys.readouterr()
    assert captured.out == ""  # verbose print suppressed by progress_fn
    assert len(calls) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_training_progress.py::test_influence_progress_fn_called_per_epoch tests/test_training_progress.py::test_cri_progress_fn_called_per_epoch tests/test_training_progress.py::test_relational_progress_fn_called_per_epoch tests/test_training_progress.py::test_progress_fn_takes_priority_over_verbose -v
```

Expected: `TypeError: train_influence_model() got an unexpected keyword argument 'progress_fn'`

- [ ] **Step 3: Add `progress_fn` to `influence_train.py`**

Add to imports (after the existing imports):
```python
from collections.abc import Callable
```

Add `progress_fn` parameter to `train_influence_model` signature (after `deterministic: bool = False,`):
```python
    progress_fn: Callable[[int, float, float | None], None] | None = None,
```

Replace lines 183–194 (the epoch-print block) with:

```python
            if progress_fn is not None:
                progress_fn(epoch + 1, train_loss, validation_loss)
            elif verbose:
                marker = "  [best]" if is_best else ""
                print(
                    f"epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}  val={validation_loss:.4f}{marker}",
                    flush=True,
                )
            if patience > 0 and epochs_without_improvement >= patience:
                if progress_fn is None and verbose:
                    print(f"early stop at epoch {epoch + 1}", flush=True)
                break
        else:
            if progress_fn is not None:
                progress_fn(epoch + 1, train_loss, None)
            elif verbose:
                print(f"epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}", flush=True)
```

The full replacement context (find this block in `influence_train.py` starting at the `else:` of the `epochs_without_improvement` check):

```python
        # OLD block to replace (lines ~183–194):
        #    else:
        #        epochs_without_improvement += 1
        #        if patience > 0 and epochs_without_improvement >= patience:
        #            if verbose:
        #                print(f"early stop at epoch {epoch + 1}", flush=True)
        #            break
        #    if verbose:
        #        marker = "  [best]" if is_best else ""
        #        print(...)
        #elif verbose:
        #    print(f"epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}", flush=True)

        # NEW block:
        else:
            epochs_without_improvement += 1
        if progress_fn is not None:
            progress_fn(epoch + 1, train_loss, validation_loss)
        elif verbose:
            marker = "  [best]" if is_best else ""
            print(
                f"epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}  val={validation_loss:.4f}{marker}",
                flush=True,
            )
        if patience > 0 and epochs_without_improvement >= patience:
            if progress_fn is None and verbose:
                print(f"early stop at epoch {epoch + 1}", flush=True)
            break
    else:
        if progress_fn is not None:
            progress_fn(epoch + 1, train_loss, None)
        elif verbose:
            print(f"epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}", flush=True)
```

Here is the complete replacement for lines 175–194 of `influence_train.py` to make the structure clear:

```python
            is_best = best_validation_loss is None or validation_loss < best_validation_loss
            if is_best:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if progress_fn is not None:
                progress_fn(epoch + 1, train_loss, validation_loss)
            elif verbose:
                marker = "  [best]" if is_best else ""
                print(
                    f"epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}  val={validation_loss:.4f}{marker}",
                    flush=True,
                )
            if patience > 0 and epochs_without_improvement >= patience:
                if progress_fn is None and verbose:
                    print(f"early stop at epoch {epoch + 1}", flush=True)
                break
        else:
            if progress_fn is not None:
                progress_fn(epoch + 1, train_loss, None)
            elif verbose:
                print(f"epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}", flush=True)
```

- [ ] **Step 4: Add `progress_fn` to `cri_train.py`**

Add to imports (after existing imports):
```python
from collections.abc import Callable
```

Add `progress_fn` parameter to `train_cri_model` signature (after `topology_path: str | Path | None = None,`):
```python
    progress_fn: Callable[[int, float, float | None], None] | None = None,
```

Replace the epoch-print block (same pattern as influence_train.py). The original block is at lines 164–183 of `cri_train.py`:

```python
            is_best = best_validation_loss is None or validation_loss < best_validation_loss
            if is_best:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if progress_fn is not None:
                progress_fn(epoch + 1, train_loss, validation_loss)
            elif verbose:
                marker = "  [best]" if is_best else ""
                print(
                    f"epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}  val={validation_loss:.4f}{marker}",
                    flush=True,
                )
            if patience > 0 and epochs_without_improvement >= patience:
                if progress_fn is None and verbose:
                    print(f"early stop at epoch {epoch + 1}", flush=True)
                break
        else:
            if progress_fn is not None:
                progress_fn(epoch + 1, train_loss, None)
            elif verbose:
                print(f"epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}", flush=True)
```

- [ ] **Step 5: Add `progress_fn` to `train.py`**

Add to imports (after existing imports):
```python
from collections.abc import Callable
```

Add `progress_fn` parameter to `train_relational_model` signature (after `topology_path: str | Path | None = None,`):
```python
    progress_fn: Callable[[int, float, float | None], None] | None = None,
```

Apply the same epoch-print replacement pattern (lines 241–260 of `train.py`):

```python
            is_best = best_validation_loss is None or validation_loss < best_validation_loss
            if is_best:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if progress_fn is not None:
                progress_fn(epoch + 1, train_loss, validation_loss)
            elif verbose:
                marker = "  [best]" if is_best else ""
                print(
                    f"epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}  val={validation_loss:.4f}{marker}",
                    flush=True,
                )
            if patience > 0 and epochs_without_improvement >= patience:
                if progress_fn is None and verbose:
                    print(f"early stop at epoch {epoch + 1}", flush=True)
                break
        else:
            if progress_fn is not None:
                progress_fn(epoch + 1, train_loss, None)
            elif verbose:
                print(f"epoch {epoch + 1:>{width}}/{epochs}  train={train_loss:.4f}", flush=True)
```

Also add `progress_fn` to the `train_model` wrapper (lines 278–318) so it passes through to `train_relational_model`:

Add `progress_fn: Callable[[int, float, float | None], None] | None = None,` to `train_model`'s parameter list (after `topology_path: str | Path | None = None,`), and add `progress_fn=progress_fn,` to the `train_relational_model(...)` call.

- [ ] **Step 6: Run all tests to verify they pass (including existing ones)**

```bash
pytest tests/test_training_progress.py -v
```

Expected: all tests PASS (including the 4 new ones and the 7 existing ones).

- [ ] **Step 7: Commit**

```bash
git add src/allostery/pipeline/influence_train.py src/allostery/pipeline/cri_train.py src/allostery/pipeline/train.py tests/test_training_progress.py
git commit -m "feat: add progress_fn callback to all three training functions"
```

---

### Task 3: Wire TrainingProgress into execute.py, workflow.py, and cli.py

**Files:**
- Modify: `src/allostery/pipeline/execute.py`
- Modify: `src/allostery/pipeline/workflow.py`
- Modify: `src/allostery/cli.py`

**Interfaces:**
- Consumes:
  - `TrainingProgress(total_epochs, *, quiet, tty)` from Task 1
  - `train_*(..., progress_fn=...)` from Task 2
- Produces: `run_training(config, *, progress_fn=None)` — new kwarg; `run_workflow(..., training_progress_fn=None)` — new kwarg

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_training_progress.py`:

```python
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest


def _run_config_text(fixture_path: Path) -> str:
    return "\n".join([
        "mode: run",
        "data:",
        f"  pdb_path: {fixture_path / 'tiny_trajectory.pdb'}",
        "  window_size: 3",
        "  horizon_size: 1",
        "  stride: 1",
        "model:",
        "  family: influence",
        "  hidden_dim: 8",
        "  residue_layers: 1",
        "  pair_layers: 1",
        "  dropout: 0.0",
        "training:",
        "  epochs: 2",
        "  learning_rate: 0.01",
        "  consistency_weight: 0.0",
        "  verbose: false",
        "scoring:",
        "  top_k: 3",
        "output:",
        "  model_path: null",
        "  score_csv_path: SCORES",
    ])


def test_run_command_writes_progress_to_stderr(tmp_path: Path, fixture_path: Path, capsys) -> None:
    from allostery.cli import main

    cfg = tmp_path / "config.yaml"
    scores = tmp_path / "scores.csv"
    cfg.write_text(_run_config_text(fixture_path).replace("SCORES", str(scores)))

    ret = main([str(cfg)])
    assert ret == 0
    captured = capsys.readouterr()
    # Progress goes to stderr, not stdout
    assert "epoch" in captured.err or "Training complete" in captured.err


def test_quiet_flag_suppresses_training_progress(tmp_path: Path, fixture_path: Path, capsys) -> None:
    from allostery.cli import main

    cfg = tmp_path / "config.yaml"
    scores = tmp_path / "scores.csv"
    cfg.write_text(_run_config_text(fixture_path).replace("SCORES", str(scores)))

    ret = main(["--quiet", str(cfg)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "epoch" not in captured.err
    assert "Training complete" not in captured.err


def test_json_flag_suppresses_training_progress(tmp_path: Path, fixture_path: Path, capsys) -> None:
    from allostery.cli import main

    cfg = tmp_path / "config.yaml"
    scores = tmp_path / "scores.csv"
    cfg.write_text(_run_config_text(fixture_path).replace("SCORES", str(scores)))

    ret = main(["--json", str(cfg)])
    assert ret == 0
    captured = capsys.readouterr()
    assert "epoch" not in captured.err
    assert "Training complete" not in captured.err
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_cli_training_progress.py::test_run_command_writes_progress_to_stderr -v
```

Expected: FAIL — progress currently goes to stdout, not stderr.

- [ ] **Step 3: Add `progress_fn` to `execute.py`**

Change the signature of `run_training`:
```python
def run_training(config: AppConfig, *, progress_fn=None) -> Any:
```

Add `progress_fn=progress_fn,` to each of the three training function calls:

```python
    if config.model.family == 'influence':
        return train_influence_model(
            ...
            verbose=training.verbose,
            progress_fn=progress_fn,
            ...
        )

    if config.model.family == 'cri':
        return train_cri_model(
            ...
            verbose=training.verbose,
            progress_fn=progress_fn,
            ...
        )

    return train_model(
        ...
        verbose=training.verbose,
        progress_fn=progress_fn,
        ...
    )
```

- [ ] **Step 4: Add `training_progress_fn` to `workflow.py`**

Change the signature of `run_workflow`:
```python
def run_workflow(
    config: AppConfig,
    *,
    backend=None,
    progress: Callable[[str], None] | None = None,
    training_progress_fn=None,
) -> Result:
```

Change the training call (line 47):
```python
        result = run_training(config, progress_fn=training_progress_fn)
```

- [ ] **Step 5: Wire `TrainingProgress` into `cli.py`**

Add this import near the top of `cli.py` (after the existing imports):
```python
from allostery.pipeline.progress import TrainingProgress
```

In `_dispatch`, replace the `'run'` path's training section. Find:
```python
    if config.mode in {'train', 'run'}:
        result = run_training(config)
        lines.append(f'trained samples={result.num_samples} checkpoint={config.output.model_path}')
        if config.output.model_path is not None:
            artifacts.append(config.output.model_path)
```

Replace with:
```python
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
```

For the `'workflow'` dispatch, replace:
```python
    if args.command == 'workflow':
        import sys as _sys
        config = load_config(args.config_path)
        emit_progress = not args.json and not args.quiet
        progress = (lambda stage: print(f'[{stage}] ...', file=_sys.stderr)) if emit_progress else None
        return run_workflow(config, progress=progress)
```

With:
```python
    if args.command == 'workflow':
        import sys as _sys
        config = load_config(args.config_path)
        emit_progress = not args.json and not args.quiet
        stage_progress = (lambda stage: print(f'[{stage}] ...', file=_sys.stderr)) if emit_progress else None
        total_epochs = config.training.epochs if config.training and config.mode in {'train', 'run'} else 0
        with TrainingProgress(total_epochs, quiet=not emit_progress) as tp:
            result = run_workflow(config, progress=stage_progress, training_progress_fn=tp.update)
        return result
```

- [ ] **Step 6: Run all tests**

```bash
pytest tests/test_cli_training_progress.py tests/test_training_progress.py tests/test_pipeline_workflow.py tests/test_cli_workflow.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/allostery/pipeline/execute.py src/allostery/pipeline/workflow.py src/allostery/cli.py tests/test_cli_training_progress.py
git commit -m "feat: wire TrainingProgress into run and workflow commands"
```

---

### Task 4: Auto-threshold detection and score histogram in `network.py`

**Files:**
- Modify: `src/allostery/network.py`
- Create: `tests/test_threshold.py`

**Interfaces:**
- Produces:
  - `detect_threshold(scores: list[float]) -> tuple[float, int]` — returns `(threshold_score, 1-based rank)` at the knee of the sorted-score curve
  - `format_score_histogram(scores: list[float], *, bins: int = 10, threshold_rank: int | None = None) -> str` — returns a multi-line ASCII bar chart

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_threshold.py
from __future__ import annotations

import pytest

from allostery.network import detect_threshold, format_score_histogram


def test_detect_threshold_step_function() -> None:
    # Clear drop between rank 3 and rank 4 — knee should be at rank ≤ 3
    scores = [0.9, 0.85, 0.8, 0.1, 0.05, 0.02]
    _t, rank = detect_threshold(scores)
    assert 1 <= rank <= 3


def test_detect_threshold_returns_score_at_knee() -> None:
    scores = [0.9, 0.85, 0.8, 0.1, 0.05, 0.02]
    t, rank = detect_threshold(scores)
    sorted_desc = sorted(scores, reverse=True)
    assert t == pytest.approx(sorted_desc[rank - 1])


def test_detect_threshold_uniform_returns_first() -> None:
    scores = [0.5, 0.5, 0.5, 0.5]
    t, rank = detect_threshold(scores)
    assert rank == 1
    assert t == pytest.approx(0.5)


def test_detect_threshold_fewer_than_3_returns_first() -> None:
    scores = [0.9, 0.8]
    t, rank = detect_threshold(scores)
    assert rank == 1
    assert t == pytest.approx(0.9)


def test_detect_threshold_single_score() -> None:
    scores = [0.7]
    t, rank = detect_threshold(scores)
    assert rank == 1
    assert t == pytest.approx(0.7)


def test_format_score_histogram_bin_counts_sum_to_total() -> None:
    import re
    scores = [i / 10.0 for i in range(11)]  # 11 scores 0.0–1.0
    hist = format_score_histogram(scores, bins=5)
    counts = [int(m.group(1)) for m in re.finditer(r"(\d+) pairs", hist)]
    assert sum(counts) == len(scores)


def test_format_score_histogram_contains_header() -> None:
    scores = [0.1, 0.5, 0.9]
    hist = format_score_histogram(scores, bins=3)
    assert "Score Distribution" in hist
    assert "3 pairs" in hist


def test_format_score_histogram_threshold_marker_present() -> None:
    scores = [0.9, 0.8, 0.7, 0.1, 0.05]
    hist = format_score_histogram(scores, bins=5, threshold_rank=3)
    assert "▶ threshold" in hist


def test_format_score_histogram_no_marker_when_rank_none() -> None:
    scores = [0.9, 0.8, 0.7]
    hist = format_score_histogram(scores, bins=3)
    assert "▶ threshold" not in hist


def test_format_score_histogram_uniform_scores() -> None:
    scores = [0.5, 0.5, 0.5]
    hist = format_score_histogram(scores, bins=5)
    assert "Score Distribution" in hist
    assert "all scores equal" in hist
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_threshold.py -v
```

Expected: `ImportError: cannot import name 'detect_threshold'`

- [ ] **Step 3: Add `import numpy as np` to `network.py`**

At the top of `src/allostery/network.py`, after the existing stdlib imports, add:
```python
import numpy as np
```

- [ ] **Step 4: Add `detect_threshold` and `format_score_histogram` to `network.py`**

Add these two functions after `read_scores_csv` and before `build_graph`:

```python
def detect_threshold(scores: list[float]) -> tuple[float, int]:
    """Find the knee of the sorted-score curve using the Kneedle method.

    Returns (threshold_score, 1-based_rank). For < 3 scores or a flat curve,
    returns the highest score at rank 1.
    """
    arr = np.array(sorted(scores, reverse=True), dtype=float)
    n = len(arr)
    if n < 3 or arr[0] == arr[-1]:
        return float(arr[0]), 1
    x = np.arange(n, dtype=float) / (n - 1)
    y = (arr - arr[-1]) / (arr[0] - arr[-1])
    k = int(np.argmax(y - x))
    return float(arr[k]), k + 1


def format_score_histogram(
    scores: list[float],
    *,
    bins: int = 10,
    threshold_rank: int | None = None,
) -> str:
    """Return an ASCII bar chart of the score distribution.

    If threshold_rank is given, marks the bin containing that rank with '▶ threshold'.
    """
    arr = np.array(scores, dtype=float)
    mn, mx = float(arr.min()), float(arr.max())
    header = f"=== Score Distribution ({len(scores)} pairs) ==="
    if mn == mx:
        return f"{header}\n(all scores equal: {mn:.4f})"

    counts, edges = np.histogram(arr, bins=bins)
    max_count = int(counts.max()) or 1
    bar_width = 20

    threshold_bin_idx: int | None = None
    if threshold_rank is not None:
        sorted_desc = np.sort(arr)[::-1]
        t_score = float(sorted_desc[min(threshold_rank - 1, len(sorted_desc) - 1)])
        raw = int((t_score - mn) / (mx - mn) * bins)
        threshold_bin_idx = max(0, min(bins - 1, raw))

    lines = [header]
    for i in range(bins - 1, -1, -1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        count = int(counts[i])
        filled = int(bar_width * count / max_count)
        bar = "█" * filled + " " * (bar_width - filled)
        marker = "  ▶ threshold" if threshold_bin_idx is not None and i == threshold_bin_idx else ""
        lines.append(f"{lo:.2f}–{hi:.2f} |{bar}|  {count} pairs{marker}")
    return "\n".join(lines)
```

Also add both names to `__all__` in `network.py`:
```python
__all__ = [
    "AllostericNetwork",
    "build_graph",
    "betweenness_centrality",
    "channel_summary",
    "connected_components",
    "detect_threshold",
    "dijkstra",
    "format_report",
    "format_score_histogram",
    "hub_summary",
    "network_summary",
    "read_scores_csv",
    "shortest_paths",
]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_threshold.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 6: Run the full test suite to catch regressions**

```bash
pytest -q
```

Expected: all tests pass (the numpy import in network.py is the only change affecting existing code; network.py tests don't check imports so they remain green).

- [ ] **Step 7: Commit**

```bash
git add src/allostery/network.py tests/test_threshold.py
git commit -m "feat: add auto-threshold detection and score histogram to network.py"
```

---

### Task 5: PyMOL script export module

**Files:**
- Create: `src/allostery/pipeline/pymol_export.py`
- Create: `tests/test_pymol_export.py`

**Interfaces:**
- Produces:
  - `write_pymol_script(pml_path: Path, pdb_path: Path, node_labels: list[str], centrality: dict[int, float], top_pairs: list[tuple[str, str, float]], path_edges: list[tuple[str, str]] | None = None) -> None`
  - `_parse_label(label: str) -> tuple[str, str]` — parses `"A:12 GLY"` → `("A", "12")`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pymol_export.py
from __future__ import annotations

from pathlib import Path

import pytest

from allostery.pipeline.pymol_export import write_pymol_script


def test_write_pymol_script_creates_file(tmp_path: Path) -> None:
    pml = tmp_path / "out.pml"
    write_pymol_script(
        pml_path=pml,
        pdb_path=Path("/path/to/protein.pdb"),
        node_labels=["A:1 GLY", "A:2 ALA"],
        centrality={0: 0.8, 1: 0.2},
        top_pairs=[("A:1 GLY", "A:2 ALA", 0.9)],
    )
    assert pml.exists()


def test_write_pymol_script_loads_pdb(tmp_path: Path) -> None:
    pml = tmp_path / "out.pml"
    pdb = Path("/my/structure.pdb")
    write_pymol_script(
        pml_path=pml,
        pdb_path=pdb,
        node_labels=["A:1 GLY"],
        centrality={0: 1.0},
        top_pairs=[],
    )
    content = pml.read_text()
    assert f"load {pdb.resolve()}" in content


def test_write_pymol_script_contains_alter_and_spectrum(tmp_path: Path) -> None:
    pml = tmp_path / "out.pml"
    write_pymol_script(
        pml_path=pml,
        pdb_path=Path("/p.pdb"),
        node_labels=["A:1 GLY", "A:2 ALA"],
        centrality={0: 1.0, 1: 0.5},
        top_pairs=[],
    )
    content = pml.read_text()
    assert "alter chain A and resi 1 and name CA" in content
    assert "spectrum b, white_red" in content


def test_write_pymol_script_contains_pair_distance(tmp_path: Path) -> None:
    pml = tmp_path / "out.pml"
    write_pymol_script(
        pml_path=pml,
        pdb_path=Path("/p.pdb"),
        node_labels=["A:1 GLY", "A:3 SER"],
        centrality={0: 0.9, 1: 0.1},
        top_pairs=[("A:1 GLY", "A:3 SER", 0.9)],
    )
    content = pml.read_text()
    assert "distance pair_1" in content
    assert "color yellow, pair_*" in content


def test_write_pymol_script_includes_path_edges(tmp_path: Path) -> None:
    pml = tmp_path / "out.pml"
    write_pymol_script(
        pml_path=pml,
        pdb_path=Path("/p.pdb"),
        node_labels=["A:1 GLY", "A:2 ALA", "A:3 SER"],
        centrality={0: 1.0, 1: 0.5, 2: 0.0},
        top_pairs=[("A:1 GLY", "A:3 SER", 0.7)],
        path_edges=[("A:1 GLY", "A:2 ALA"), ("A:2 ALA", "A:3 SER")],
    )
    content = pml.read_text()
    assert "distance path_1" in content
    assert "distance path_2" in content
    assert "color cyan, path_*" in content


def test_write_pymol_script_no_path_edges_omits_path_lines(tmp_path: Path) -> None:
    pml = tmp_path / "out.pml"
    write_pymol_script(
        pml_path=pml,
        pdb_path=Path("/p.pdb"),
        node_labels=["A:1 GLY"],
        centrality={0: 1.0},
        top_pairs=[],
        path_edges=None,
    )
    content = pml.read_text()
    assert "path_" not in content
    assert "cyan" not in content
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pymol_export.py -v
```

Expected: `ImportError: cannot import name 'write_pymol_script'`

- [ ] **Step 3: Implement `src/allostery/pipeline/pymol_export.py`**

```python
from __future__ import annotations

from pathlib import Path


def write_pymol_script(
    pml_path: Path,
    pdb_path: Path,
    node_labels: list[str],
    centrality: dict[int, float],
    top_pairs: list[tuple[str, str, float]],
    path_edges: list[tuple[str, str]] | None = None,
) -> None:
    """Write a PyMOL .pml script that colors residues by centrality and shows allosteric pairs."""
    max_c = max(centrality.values(), default=1.0) or 1.0
    lines: list[str] = [
        "# Generated by allostery analyze",
        f"load {pdb_path.resolve()}",
        "hide everything",
        "show cartoon",
        "color white",
        "",
        "# Color residues by betweenness centrality (white=low, red=high)",
        "alter all, b=0.0",
    ]
    for idx, label in enumerate(node_labels):
        chain, resi = _parse_label(label)
        norm_c = centrality.get(idx, 0.0) / max_c
        lines.append(f"alter chain {chain} and resi {resi} and name CA, b={norm_c:.4f}")
    lines += [
        "spectrum b, white_red, minimum=0.0, maximum=1.0",
        "",
        "# Top allosteric pairs",
    ]
    for k, (label_i, label_j, _score) in enumerate(top_pairs, 1):
        chain_i, resi_i = _parse_label(label_i)
        chain_j, resi_j = _parse_label(label_j)
        si = f"(chain {chain_i} and resi {resi_i} and name CA)"
        sj = f"(chain {chain_j} and resi {resi_j} and name CA)"
        lines.append(f"distance pair_{k}, {si}, {sj}")
    lines += ["color yellow, pair_*", ""]
    if path_edges:
        lines.append("# Allosteric path")
        for k, (src, dst) in enumerate(path_edges, 1):
            cs, rs = _parse_label(src)
            cd, rd = _parse_label(dst)
            ss = f"(chain {cs} and resi {rs} and name CA)"
            sd = f"(chain {cd} and resi {rd} and name CA)"
            lines.append(f"distance path_{k}, {ss}, {sd}")
        lines += ["color cyan, path_*", ""]
    lines.append("zoom")
    pml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_label(label: str) -> tuple[str, str]:
    """Parse 'A:12 GLY' → ('A', '12')."""
    chain, rest = label.split(":", 1)
    resi = rest.split()[0]
    return chain, resi


__all__ = ["write_pymol_script"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pymol_export.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/allostery/pipeline/pymol_export.py tests/test_pymol_export.py
git commit -m "feat: add PyMOL script export for allosteric network visualization"
```

---

### Task 6: Wire threshold, histogram, and PyMOL into `analyze.py` and `cli.py`

**Files:**
- Modify: `src/allostery/pipeline/analyze.py`
- Modify: `src/allostery/cli.py`
- Create: `tests/test_cli_analyze_pml.py`

**Interfaces:**
- Consumes:
  - `detect_threshold` from Task 4
  - `format_score_histogram` from Task 4
  - `write_pymol_script` from Task 5
  - `betweenness_centrality`, `shortest_paths` from `network.py`
- Produces:
  - `run_network_analysis(..., out_pml: Path | None = None, pdb_path: Path | None = None) -> str` — extended signature; `--pml-path` CLI flag; threshold + histogram always in report output

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli_analyze_pml.py
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from allostery.cli import main
from allostery.pipeline.analyze import run_network_analysis


def _write_scores(path: Path) -> None:
    fieldnames = [
        "rank", "score",
        "residue_i_index", "residue_i_chain", "residue_i_number", "residue_i_name",
        "residue_j_index", "residue_j_chain", "residue_j_number", "residue_j_name",
    ]
    rows = [
        {"rank": 1, "score": "0.9", "residue_i_index": 0, "residue_i_chain": "A",
         "residue_i_number": "1", "residue_i_name": "GLY",
         "residue_j_index": 1, "residue_j_chain": "A", "residue_j_number": "2", "residue_j_name": "ALA"},
        {"rank": 2, "score": "0.8", "residue_i_index": 1, "residue_i_chain": "A",
         "residue_i_number": "2", "residue_i_name": "ALA",
         "residue_j_index": 2, "residue_j_chain": "A", "residue_j_number": "3", "residue_j_name": "SER"},
        {"rank": 3, "score": "0.1", "residue_i_index": 0, "residue_i_chain": "A",
         "residue_i_number": "1", "residue_i_name": "GLY",
         "residue_j_index": 2, "residue_j_chain": "A", "residue_j_number": "3", "residue_j_name": "SER"},
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_analyze_report_includes_threshold_line(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)
    report = run_network_analysis(scores, top_k=3)
    assert "Suggested threshold" in report


def test_analyze_report_includes_histogram(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)
    report = run_network_analysis(scores, top_k=3)
    assert "Score Distribution" in report


def test_cli_analyze_out_pml_without_pdb_exits_1(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)
    pml = tmp_path / "out.pml"
    ret = main(["analyze", str(scores), "--out-pml", str(pml)])
    assert ret == 1


def test_cli_analyze_out_pml_with_pdb_creates_file(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)
    pml = tmp_path / "out.pml"
    fake_pdb = tmp_path / "protein.pdb"
    fake_pdb.write_text("ATOM record placeholder\n")
    ret = main(["analyze", str(scores), "--top-k", "3", "--out-pml", str(pml), "--pdb", str(fake_pdb)])
    assert ret == 0
    assert pml.exists()
    content = pml.read_text()
    assert "load" in content
    assert "spectrum b, white_red" in content


def test_cli_analyze_out_pml_in_artifacts(tmp_path: Path, capsys) -> None:
    scores = tmp_path / "scores.csv"
    _write_scores(scores)
    pml = tmp_path / "out.pml"
    fake_pdb = tmp_path / "protein.pdb"
    fake_pdb.write_text("placeholder\n")
    ret = main(["--quiet", "analyze", str(scores), "--out-pml", str(pml), "--pdb", str(fake_pdb)])
    assert ret == 0
    captured = capsys.readouterr()
    # --quiet mode prints artifact paths to stdout
    assert str(pml) in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli_analyze_pml.py -v
```

Expected: `FAIL` — `run_network_analysis` currently does not include threshold/histogram in the report, and `--out-pml` / `--pdb` flags don't exist.

- [ ] **Step 3: Update `src/allostery/pipeline/analyze.py`**

Replace the entire file with:

```python
from __future__ import annotations

from pathlib import Path

from allostery.network import (
    betweenness_centrality,
    build_graph,
    detect_threshold,
    format_report,
    format_score_histogram,
    read_scores_csv,
    shortest_paths,
)
from allostery.pipeline.pymol_export import write_pymol_script


def run_network_analysis(
    scores_csv: str | Path,
    top_k: int = 20,
    source: str | None = None,
    sink: str | None = None,
    top_paths: int = 5,
    top_hubs: int = 10,
    out_path: str | Path | None = None,
    out_pml: Path | None = None,
    pdb_path: Path | None = None,
) -> str:
    """Read a scores CSV, build the allosteric network, and return a text report."""
    rows = read_scores_csv(scores_csv)
    all_scores = [float(r["score"]) for r in rows]
    threshold_score, threshold_rank = detect_threshold(all_scores)

    net = build_graph(rows, top_k=top_k)
    if net.num_nodes == 0:
        raise ValueError(
            "No edges in the network after top-k filtering; increase --top-k "
            "or check the scores CSV."
        )

    threshold_line = (
        f"Suggested threshold: {threshold_score:.4f}"
        f" (top {threshold_rank} of {len(all_scores)} pairs"
        f" — largest gap at rank {threshold_rank})"
    )
    body = format_report(
        net,
        source_label=source,
        sink_label=sink,
        top_hubs=top_hubs,
        top_paths=top_paths,
    )
    histogram = format_score_histogram(all_scores, bins=10, threshold_rank=threshold_rank)
    report = f"{threshold_line}\n\n{body}\n\n{histogram}"

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")

    if out_pml is not None and pdb_path is not None:
        centrality = betweenness_centrality(net)
        sorted_rows = sorted(rows, key=lambda r: float(r["score"]), reverse=True)[:top_k]
        top_pairs = [
            (
                f"{r['residue_i_chain']}:{r['residue_i_number']} {r['residue_i_name']}",
                f"{r['residue_j_chain']}:{r['residue_j_number']} {r['residue_j_name']}",
                float(r["score"]),
            )
            for r in sorted_rows
        ]
        path_edges = None
        if source is not None and sink is not None:
            paths = shortest_paths(net, source, sink, top_n=1)
            if paths:
                path_nodes, _ = paths[0]
                path_edges = list(zip(path_nodes, path_nodes[1:]))
        out_pml.parent.mkdir(parents=True, exist_ok=True)
        write_pymol_script(
            pml_path=out_pml,
            pdb_path=pdb_path,
            node_labels=net.node_labels,
            centrality=centrality,
            top_pairs=top_pairs,
            path_edges=path_edges,
        )

    return report


__all__ = ["run_network_analysis"]
```

- [ ] **Step 4: Add `--pdb` and `--out-pml` flags to the `analyze` subparser in `cli.py`**

In `build_parser()`, find the `analyze_parser` block and append:
```python
    analyze_parser.add_argument(
        '--pdb', default=None,
        help='Structure file for PyMOL export (required when --out-pml is given)'
    )
    analyze_parser.add_argument(
        '--out-pml', default=None,
        help='Write a PyMOL .pml script to this path'
    )
```

- [ ] **Step 5: Update the `'analyze'` dispatch branch in `_dispatch` in `cli.py`**

Replace:
```python
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
```

With:
```python
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
```

- [ ] **Step 6: Run the new tests**

```bash
pytest tests/test_cli_analyze_pml.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 7: Run the full test suite**

```bash
pytest -q
```

Expected: all tests pass. The `test_analyze_writes_report_file` test checks `"Allosteric Network" in report` — this still passes because the body still contains `format_report` output which includes that header.

- [ ] **Step 8: Commit**

```bash
git add src/allostery/pipeline/analyze.py src/allostery/cli.py tests/test_cli_analyze_pml.py
git commit -m "feat: add threshold, histogram, and PyMOL export to analyze command"
```
