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
