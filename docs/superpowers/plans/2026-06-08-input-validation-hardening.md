# Input Validation Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three missing pre-flight validations so users get clear, actionable errors instead of cryptic internal crashes.

**Architecture:** All three changes are purely additive. Tasks 1 and 2 insert new guard clauses into the existing `validate_config` error-accumulation block in `config.py`. Task 3 inserts a row-validation loop into `read_scores_csv` in `network.py`. No new files, no new imports beyond what is already present.

**Tech Stack:** Python 3.11, pytest, `unittest.mock` (stdlib)

---

## File Map

| File | Change |
|---|---|
| `src/allostery/config.py` | Add `topology_path` existence check and CUDA availability check inside `validate_config` |
| `src/allostery/network.py` | Add data-row validation loop inside `read_scores_csv` |
| `tests/test_config.py` | Add 3 new tests (topology_path missing, CUDA unavailable, CUDA available) |
| `tests/test_network.py` | Add 3 new tests (empty score, non-numeric score, empty chain field) |

---

## Task 1: `topology_path` file existence check

**Files:**
- Modify: `src/allostery/config.py` — `validate_config` function (immediately after the existing `pdb_path` check)
- Test: `tests/test_config.py`

### Context

`validate_config` (at the bottom of `config.py`) accumulates errors in a `list[str]` called
`errors` and raises a single `ConfigError` at the end. The existing `pdb_path` check looks like:

```python
if not config.data.pdb_path.exists():
    errors.append(f"data.pdb_path does not exist (got {config.data.pdb_path!r})")
```

The `topology_path` check goes on the very next line after that block.

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_config.py` (after the last existing test):

```python
def test_missing_topology_path_raises_config_error(tmp_path: Path) -> None:
    from allostery.config import ConfigError, load_config
    config_path = tmp_path / "topo.yaml"
    missing_topo = tmp_path / "does_not_exist.prmtop"
    _write_config(
        config_path,
        [
            "mode: run",
            "data:",
            f"  pdb_path: {FIXTURE_PDB}",
            f"  topology_path: {missing_topo}",
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
            "  learning_rate: 0.001",
            "  consistency_weight: 0.0",
            "scoring:",
            "  top_k: 5",
            "output:",
            "  model_path: outputs/model.pt",
            "  score_csv_path: outputs/scores.csv",
        ],
    )
    with pytest.raises(ConfigError) as exc_info:
        load_config(config_path)
    assert "topology_path" in str(exc_info.value)
    assert str(missing_topo) in str(exc_info.value)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/qshao/allostery
python -m pytest tests/test_config.py::test_missing_topology_path_raises_config_error -v
```

Expected: `FAILED` — the test fails because no `topology_path` check exists yet.

- [ ] **Step 3: Add the guard to `validate_config`**

In `src/allostery/config.py`, find the `pdb_path` existence block:

```python
    if not config.data.pdb_path.exists():
        errors.append(f"data.pdb_path does not exist (got {config.data.pdb_path!r})")
```

Add the following two lines immediately after it:

```python
    if (config.data.topology_path is not None
            and not config.data.topology_path.exists()):
        errors.append(
            f"data.topology_path: file not found: {config.data.topology_path}"
        )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
python -m pytest tests/test_config.py::test_missing_topology_path_raises_config_error -v
```

Expected: `PASSED`

- [ ] **Step 5: Run the full test suite to check for regressions**

```bash
python -m pytest -q
```

Expected: all tests pass (currently 119).

- [ ] **Step 6: Commit**

```bash
git add src/allostery/config.py tests/test_config.py
git commit -m "feat: validate topology_path existence in validate_config"
```

---

## Task 2: CUDA device availability check

**Files:**
- Modify: `src/allostery/config.py` — `validate_config` function (inside the training block, after the `device` empty-string check)
- Test: `tests/test_config.py`

### Context

Inside `validate_config`, the training-specific checks live in:

```python
    if config.mode in {'train', 'run'}:
        if config.training is None:
            errors.append(...)
        else:
            ...
            if not config.training.device:
                errors.append("training.device must not be empty")
            if config.training.batch_size <= 0:
                ...
```

The CUDA check is inserted immediately after the `device` empty-string check and before the `batch_size` check.

`import_module` is already imported at the top of `config.py`:
```python
from importlib import import_module
```

- [ ] **Step 1: Write the two failing tests**

Add both tests to `tests/test_config.py`:

```python
def test_cuda_device_unavailable_raises_config_error(tmp_path: Path) -> None:
    import unittest.mock
    from allostery.config import ConfigError, load_config
    config_path = tmp_path / "cuda.yaml"
    _write_config(
        config_path,
        [
            "mode: train",
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
            "  learning_rate: 0.001",
            "  consistency_weight: 0.0",
            "  device: cuda",
            "output:",
            "  model_path: outputs/model.pt",
        ],
    )
    with unittest.mock.patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(ConfigError) as exc_info:
            load_config(config_path)
    assert "CUDA is not available" in str(exc_info.value)


def test_cuda_device_available_no_error(tmp_path: Path) -> None:
    import unittest.mock
    from allostery.config import load_config
    config_path = tmp_path / "cuda_ok.yaml"
    _write_config(
        config_path,
        [
            "mode: train",
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
            "  learning_rate: 0.001",
            "  consistency_weight: 0.0",
            "  device: cuda",
            "output:",
            "  model_path: outputs/model.pt",
        ],
    )
    with unittest.mock.patch("torch.cuda.is_available", return_value=True):
        load_config(config_path)  # must not raise
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_config.py::test_cuda_device_unavailable_raises_config_error tests/test_config.py::test_cuda_device_available_no_error -v
```

Expected: both `FAILED` — no CUDA check exists yet.

- [ ] **Step 3: Add the CUDA check to `validate_config`**

In `src/allostery/config.py`, inside `validate_config`, find:

```python
            if not config.training.device:
                errors.append("training.device must not be empty")
            if config.training.batch_size <= 0:
```

Insert between those two `if` blocks:

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

- [ ] **Step 4: Run the two new tests to verify they pass**

```bash
python -m pytest tests/test_config.py::test_cuda_device_unavailable_raises_config_error tests/test_config.py::test_cuda_device_available_no_error -v
```

Expected: both `PASSED`

- [ ] **Step 5: Run the full test suite**

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/allostery/config.py tests/test_config.py
git commit -m "feat: check CUDA availability when training.device starts with cuda"
```

---

## Task 3: `read_scores_csv` data-row validation

**Files:**
- Modify: `src/allostery/network.py` — `read_scores_csv` function (add loop before `return rows`)
- Test: `tests/test_network.py`

### Context

`read_scores_csv` in `src/allostery/network.py` currently validates only the header:

```python
    required = {
        "score", "residue_i_chain", "residue_i_number", "residue_i_name",
        "residue_j_chain", "residue_j_number", "residue_j_name",
    }
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"Scores CSV is missing required columns: {sorted(missing)}")
    return rows
```

The new validation loop goes between the `if missing:` block and `return rows`.

The test helper `_write_scores_csv` in `tests/test_network.py` writes a CSV with these fieldnames:
`["rank", "score", "residue_i_index", "residue_i_chain", "residue_i_number", "residue_i_name", "residue_j_index", "residue_j_chain", "residue_j_number", "residue_j_name"]`

- [ ] **Step 1: Write the three failing tests**

Add these tests to `tests/test_network.py` (after the existing `read_scores_csv` tests):

```python
def test_read_scores_csv_empty_score_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    rows = [
        {
            "score": "",
            "residue_i_chain": "A", "residue_i_number": "1", "residue_i_name": "GLY",
            "residue_j_chain": "A", "residue_j_number": "2", "residue_j_name": "ALA",
        }
    ]
    _write_scores_csv(csv_path, rows)
    with pytest.raises(ValueError) as exc_info:
        read_scores_csv(csv_path)
    assert "Row 2" in str(exc_info.value)
    assert "score" in str(exc_info.value)


def test_read_scores_csv_non_numeric_score_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    rows = [
        {
            "score": "abc",
            "residue_i_chain": "A", "residue_i_number": "1", "residue_i_name": "GLY",
            "residue_j_chain": "A", "residue_j_number": "2", "residue_j_name": "ALA",
        }
    ]
    _write_scores_csv(csv_path, rows)
    with pytest.raises(ValueError) as exc_info:
        read_scores_csv(csv_path)
    assert "Row 2" in str(exc_info.value)
    assert "must be a number" in str(exc_info.value)


def test_read_scores_csv_empty_chain_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    rows = [
        {
            "score": "0.9",
            "residue_i_chain": "",
            "residue_i_number": "1", "residue_i_name": "GLY",
            "residue_j_chain": "A", "residue_j_number": "2", "residue_j_name": "ALA",
        }
    ]
    _write_scores_csv(csv_path, rows)
    with pytest.raises(ValueError) as exc_info:
        read_scores_csv(csv_path)
    assert "Row 2" in str(exc_info.value)
    assert "residue_i_chain" in str(exc_info.value)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
python -m pytest tests/test_network.py::test_read_scores_csv_empty_score_raises tests/test_network.py::test_read_scores_csv_non_numeric_score_raises tests/test_network.py::test_read_scores_csv_empty_chain_raises -v
```

Expected: all three `FAILED` — no data-row validation exists yet.

- [ ] **Step 3: Add the validation loop to `read_scores_csv`**

In `src/allostery/network.py`, find the end of `read_scores_csv`. The function currently ends with:

```python
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"Scores CSV is missing required columns: {sorted(missing)}")
    return rows
```

Replace `return rows` with:

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
    return rows
```

- [ ] **Step 4: Run the three new tests to verify they pass**

```bash
python -m pytest tests/test_network.py::test_read_scores_csv_empty_score_raises tests/test_network.py::test_read_scores_csv_non_numeric_score_raises tests/test_network.py::test_read_scores_csv_empty_chain_raises -v
```

Expected: all three `PASSED`

- [ ] **Step 5: Run the full test suite**

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/allostery/network.py tests/test_network.py
git commit -m "feat: validate data rows in read_scores_csv with row number in errors"
```
