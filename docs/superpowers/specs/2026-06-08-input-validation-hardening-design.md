# Input Validation Hardening Design

## Goal

Catch the three remaining early-failure modes — missing topology file, wrong CUDA device, malformed
CSV rows — before they surface as cryptic internal errors.

## Background

`validate_config` already checks `pdb_path` existence and collects all field-value errors into a
single `ConfigError` report. `read_scores_csv` validates the CSV header. This spec fills the three
remaining gaps using the same existing patterns.

---

## Section 1 — `topology_path` file existence check

**File:** `src/allostery/config.py` → `validate_config`

**Change:** Immediately after the existing `pdb_path` existence check, add:

```python
if (config.data.topology_path is not None
        and not config.data.topology_path.exists()):
    errors.append(
        f"data.topology_path: file not found: {config.data.topology_path}"
    )
```

**Behaviour:** If `topology_path` is set but the file does not exist, the path is included in the
collected errors and reported alongside any other errors at the end of `validate_config`. No new
imports required.

**Tests** (`tests/test_config.py`):
- `test_missing_topology_path_raises_config_error` — write a valid config YAML with
  `topology_path` pointing to a non-existent file; assert `ConfigError` is raised and the message
  contains `"topology_path"` and the missing path.

---

## Section 2 — CUDA device availability check

**File:** `src/allostery/config.py` → `validate_config`

**Change:** Inside the `config.training is not None` block, adjacent to the existing
`training.device` empty-string check, add:

```python
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
```

**Behaviour:** Any device string starting with `"cuda"` (`"cuda"`, `"cuda:0"`, `"cuda:1"`, …)
triggers the check. If `torch.cuda.is_available()` returns `False`, an error is accumulated.
`import_module` is already imported at the top of `config.py`. The `ImportError` guard allows
`validate_config` to be called in environments where torch is not yet installed (e.g., config-only
test fixtures).

**Tests** (`tests/test_config.py`):
- `test_cuda_device_unavailable_raises_config_error` — patch `torch.cuda.is_available` to return
  `False`; assert `ConfigError` raised with `"CUDA is not available"`.
- `test_cuda_device_available_no_error` — patch `torch.cuda.is_available` to return `True`; assert
  no error raised.

---

## Section 3 — `read_scores_csv` data-row validation

**File:** `src/allostery/network.py` → `read_scores_csv`

**Change:** After the existing header column check, add a loop over data rows:

```python
for row_num, row in enumerate(rows, start=2):  # row 1 is the header
    for col in required:
        if not row.get(col, "").strip():
            raise ValueError(
                f"Row {row_num}: missing or empty value for column {col!r}"
            )
    try:
        float(row["score"])
    except (ValueError, TypeError):
        raise ValueError(
            f"Row {row_num}: 'score' must be a number, got {row['score']!r}"
        )
```

**Behaviour:** `start=2` mirrors spreadsheet row counting (header = row 1), making error messages
immediately actionable. All required string fields are checked for non-empty values. `score` is
additionally checked as a parseable float, since a non-numeric score would otherwise reach
`build_graph` and produce a bare `ValueError` with no row context. Raises immediately on the first
bad row (fail-fast is appropriate here — a malformed CSV is a data problem, not an accumulation of
independent issues).

**Tests** (`tests/test_network.py`):
- `test_read_scores_csv_empty_score_field` — CSV with valid header, row 2 has `score=""` → `ValueError` matching `"Row 2"` and `"score"`.
- `test_read_scores_csv_non_numeric_score` — row 2 has `score="abc"` → `ValueError` matching `"Row 2"` and `"must be a number"`.
- `test_read_scores_csv_empty_chain_field` — row 2 has `residue_i_chain=""` → `ValueError` matching `"Row 2"` and `"residue_i_chain"`.

---

## Architecture summary

| Item | File | Function | Pattern |
|---|---|---|---|
| topology_path existence | `config.py` | `validate_config` | Accumulate into `errors` list |
| CUDA availability | `config.py` | `validate_config` | Accumulate into `errors` list |
| CSV data-row validation | `network.py` | `read_scores_csv` | Fail-fast `raise ValueError` with row number |

No new files, no new imports, no new patterns. Each item is additive to an existing validation
block.
