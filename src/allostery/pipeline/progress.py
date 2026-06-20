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


__all__ = ["TrainingProgress", "_fmt_s"]
