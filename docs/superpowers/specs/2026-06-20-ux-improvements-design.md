# UX Improvements: Training Progress & Richer Analysis Output

## Goal

Reduce the two sharpest friction points for both occasional and power users:
1. Training runs feel like a black box — no ETA, no live feedback.
2. Scores CSVs are hard to interpret — no threshold guidance, no visual summary, no way to open results in a 3-D viewer.

## Architecture

Two independent features, each adding to existing commands with no new runtime dependencies. All progress output goes to stderr so JSON output and piped workflows are unaffected.

**New file:** `src/allostery/pipeline/progress.py`  
**New file:** `src/allostery/pipeline/pymol_export.py`  
**Modified:** `src/allostery/network.py`, `src/allostery/pipeline/analyze.py`, `src/allostery/cli.py`  
**Modified:** `src/allostery/pipeline/influence_train.py`, `src/allostery/pipeline/train.py`, `src/allostery/pipeline/cri_train.py` (inline epoch `print()` → `progress_fn` callback), `src/allostery/pipeline/execute.py` (construct `TrainingProgress`, pass callback)

---

## Feature A: Training Progress Feedback

### Behavior

All progress output is written to **stderr**. Stdout is unmodified — `--json`, piped output, and CI logs are unaffected.

**TTY mode** (stderr is an interactive terminal, `sys.stderr.isatty()` is `True`): each epoch line overwrites the previous one via `\r`. The line format is:

```
[=========>          ] epoch 9/20  train=1.2341  val=1.4123  ETA 18s
```

- The progress bar is 20 characters wide, filled with `=` up to the current fraction and `>` at the frontier.
- ETA is the mean per-epoch wall time × remaining epochs, formatted as `Xs` (<60 s) or `XmYs` (≥60 s).
- Elapsed is not shown per-epoch in TTY mode (it clutters the line); it appears only in the final summary.
- When the final epoch completes, the line is closed with `\n` so it persists on screen.

**Non-TTY mode** (piped, CI, or stderr redirected): one line per epoch, no `\r`:

```
epoch 9/20  train=1.2341  val=1.4123  elapsed=12s  ETA=18s
```

**Final summary** (printed after training, unless `--quiet`):

```
Training complete: 20 epochs in 38s — best epoch 14 (val=1.1872)
```

If no validation set is used (`validation_fraction=0.0`), the best-epoch clause is omitted.

**Suppression rules:**
- `--quiet`: suppresses all epoch output and the summary.
- `--json`: suppresses all epoch output and the summary (JSON consumers parse stdout only).
- `--debug`: no change — progress still goes to stderr, tracebacks go to stderr on error.

### Implementation

**`src/allostery/pipeline/progress.py`** — new module, two public names:

```python
class TrainingProgress:
    """Context manager that prints epoch progress to stderr."""
    def __init__(self, total_epochs: int, *, quiet: bool = False, tty: bool | None = None) -> None: ...
    def update(self, epoch: int, train_loss: float, val_loss: float | None = None) -> None: ...
    def finish(self, best_epoch: int | None, best_val_loss: float | None) -> None: ...
    def __enter__(self) -> TrainingProgress: ...
    def __exit__(self, *_: object) -> None: ...
```

- `tty` defaults to `sys.stderr.isatty()` if `None` — overridable in tests.
- `update()` is called once per epoch by the training loop.
- `finish()` prints the summary line.

**Integration points:** three training modules each contain inline `print()` epoch output to **stdout**: `src/allostery/pipeline/influence_train.py`, `src/allostery/pipeline/train.py`, and `src/allostery/pipeline/cri_train.py`. All three follow the same pattern — direct `print(f"epoch {epoch+1}/{epochs}  train=...")` with no existing callback parameter.

Each training function gains an optional parameter `progress_fn: Callable[[int, float, float | None], None] | None = None`. The inline `print()` calls are replaced with `if progress_fn is not None: progress_fn(epoch + 1, train_loss, val_loss)`. The early-stop `print` (`"early stop at epoch N"`) is also routed through `progress_fn` (val_loss passed as `None` to signal early stop).

`execute.py` constructs a `TrainingProgress` context manager and passes `progress_fn=progress.update` to each training function call. The `run` and `workflow` dispatches in `cli.py` read `--quiet` / `--json` to set `quiet=True`. This moves all epoch output from stdout to stderr.

No changes to model code, loss computation, or checkpointing.

### Tests

`tests/test_training_progress.py`:

- TTY mode: captured stderr contains `\r` and the bar characters.
- Non-TTY mode: captured stderr has one line per epoch with `elapsed=` and `ETA=`.
- `--quiet` / `quiet=True`: stderr is empty.
- `finish()` prints the summary with correct best-epoch and time format.
- ETA calculation: with known per-epoch times, ETA matches `mean_time × remaining`.

---

## Feature B: Richer Analysis Output

### B1 — Auto-threshold Detection

**Function:** `detect_threshold(scores: list[float]) -> tuple[float, int]` in `src/allostery/network.py`.

Algorithm (Kneedle, ~15 lines of numpy):
1. Sort scores descending. Let `n = len(scores)`.
2. Normalize: `x = arange(n) / (n-1)`, `y = (scores - scores[-1]) / (scores[0] - scores[-1])`.
3. Find `k = argmax(y - x)` — the rank at maximum perpendicular distance from the diagonal.
4. Return `(scores[k], k + 1)` — the score at the knee and its 1-based rank.

Edge cases:
- Fewer than 3 scores: return `(scores[0], 1)`.
- All scores equal: return `(scores[0], 1)`.

**Output in the analyze report** (always shown, no flag needed):

```
Suggested threshold: 0.431 (top 14 of 276 pairs — largest gap at rank 14)
```

This line appears at the top of the report, before the network summary.

### B2 — Score Histogram

**Function:** `format_score_histogram(scores: list[float], *, bins: int = 10, threshold_rank: int | None = None) -> str` in `src/allostery/network.py`.

- Bins are equal-width across `[min_score, max_score]`.
- Bar width scales to fill 20 characters for the largest bin.
- The bin containing `scores[threshold_rank - 1]` is marked with `▶ threshold`.

Example output:

```
=== Score Distribution (276 pairs) ===
0.80–1.00 |████████          |  42 pairs  ▶ threshold
0.60–0.80 |██████████████    |  78 pairs
0.40–0.60 |████████████████  |  91 pairs
0.20–0.40 |█████████         |  48 pairs
0.00–0.20 |██████            |  17 pairs
```

Appended to the analyze report after the hub residues section.

### B3 — PyMOL Script Export

**New flags on `allostery analyze`:**

| Flag | Description |
|---|---|
| `--pdb PATH` | Structure/trajectory file to `load` in PyMOL. Required when `--out-pml` is given. |
| `--out-pml PATH` | Write a `.pml` script here. |

Error: `--out-pml` without `--pdb` → exits 1 with message `--out-pml requires --pdb`.

**New module:** `src/allostery/pipeline/pymol_export.py`

```python
def write_pymol_script(
    pml_path: Path,
    pdb_path: Path,
    node_labels: list[str],
    centrality: dict[int, float],
    top_pairs: list[tuple[str, str, float]],
    path_edges: list[tuple[str, str]] | None = None,
) -> None: ...
```

The generated `.pml` script (pure string, no PyMOL import):

```python
# Generated by allostery analyze
load /path/to/structure.pdb
hide everything
show cartoon
color white

# Color residues by betweenness centrality (white=low, red=high)
alter all, b=0.0
alter chain A and resi 12 and name CA, b=0.831
alter chain A and resi 47 and name CA, b=0.612
...
spectrum b, white_red, minimum=0.0, maximum=1.0

# Top allosteric pairs
distance pair_1, (chain A and resi 12 and name CA), (chain A and resi 47 and name CA)
...
color yellow, pair_*

# Allosteric path (source -> sink)  [only if path_edges given]
distance path_1, (chain A and resi 12 and name CA), (chain A and resi 23 and name CA)
...
color cyan, path_*

zoom
```

**Residue label parsing:** `node_labels` entries are in the existing format `"A:12 GLY"`. The export function parses chain (`A`), number (`12`), and name (`GLY`) for the `alter` and `distance` selectors. Name is not used in PyMOL selectors (only chain + resi + atom name).

**Output:** `write_pymol_script` writes the file and returns `None`. The artifact path is added to `Result.artifacts` so `--quiet` mode prints it.

### Integration in `analyze`

`run_network_analysis` in `src/allostery/pipeline/analyze.py` currently returns a string report. It gains two new optional parameters:

```python
def run_network_analysis(
    scores_csv: str,
    top_k: int = 20,
    source: str | None = None,
    sink: str | None = None,
    top_paths: int = 5,
    top_hubs: int = 10,
    out_pml: Path | None = None,
    pdb_path: Path | None = None,
) -> str:
```

- Calls `detect_threshold` on the full score list and prepends the threshold line.
- Appends `format_score_histogram` output.
- If `out_pml` is given, calls `write_pymol_script` and appends `PyMOL script written: <path>` to the report.

`cli.py` passes `out_pml=Path(args.out_pml)` and `pdb_path=Path(args.pdb)` when both flags are present.

### Tests

`tests/test_threshold.py`:
- Perfect step function → knee at the step.
- Uniform distribution → knee at rank 1 (first score is the threshold).
- Fewer than 3 scores → returns `(scores[0], 1)`.

`tests/test_score_histogram.py`:
- Histogram bins sum to total pair count.
- Threshold marker appears in the correct bin.
- Bins=1 edge case: single bar, no crash.

`tests/test_pymol_export.py`:
- Written `.pml` contains `load`, `alter`, `spectrum`, `distance` lines.
- Centrality values appear in `alter` lines.
- `path_edges=None` → no `path_*` distance lines.

`tests/test_cli_analyze.py` (extend existing):
- `--out-pml` without `--pdb` exits 1.
- `--out-pml` with `--pdb` creates the file and adds it to artifacts.

---

## Exit Codes and Error Handling

No new exit codes. All new errors (missing `--pdb`, bad path) use the existing `USER_ERROR = 1` path.

## Global Constraints

- No new runtime dependencies — stdlib + numpy only.
- All progress output to stderr; stdout unchanged for JSON consumers.
- `from __future__ import annotations` at top of every new module.
- All randomness (none here) would use `numpy.random.default_rng(seed)`.
- Tests use `pytest` only; no new test dependencies.
