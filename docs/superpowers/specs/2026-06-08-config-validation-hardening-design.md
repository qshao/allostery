# Config Validation Hardening Design

**Date:** 2026-06-08  
**Status:** Approved — proceeding to implementation

## Goal

Make configuration errors immediately actionable. When a user supplies a bad config, they should see:
- Which file had the problem (`my_config.yaml: ...`)
- Which key was wrong and what was received (`data.window_size must be > 0 (got -1)`)
- All problems at once, not one per run

## Scope

`src/allostery/config.py` and `tests/test_config.py` only.

## Changes

### 1. Path context in every error message

Thread the resolved config file path into `load_config` and pass it to `validate_config`. Every error message is prefixed with the filename and includes `(got <value>)` where the received value is known.

Before:
```
ValueError: window_size must be greater than zero
```

After:
```
ConfigError: my_config.yaml:
  data.window_size must be > 0 (got 0)
```

### 2. Collect all errors, raise once (ConfigError)

Introduce `ConfigError(ValueError)`. Refactor `validate_config` to accumulate error strings into a list rather than raising inline. Raise a single `ConfigError` at the end that shows all problems. This eliminates the fix-run-crash loop.

### 3. Unknown-key detection

After parsing each raw section dict (data, model, training, scoring, output), compare keys against the known set. Print a `warning:` line to stderr for any unrecognized key. This catches typos like `learnig_rate: 0.001` before training starts.

Known key sets per section:
- `data`: pdb_path, window_size, horizon_size, stride, time_step, distance_cutoff, max_neighbors, min_sequence_separation, preprocess, topology_path
- `model`: family, hidden_dim, residue_layers, pair_layers, dropout, edge_types
- `training`: epochs, learning_rate, consistency_weight, entropy_weight, no_edge_weight, sparsity_weight, validation_fraction, patience, seed, device, batch_size, verbose
- `scoring`: top_k
- `output`: model_path, score_csv_path

## Architecture

```
load_config(path)
  → resolves path, reads YAML
  → _warn_unknown_keys(section_raw, known_keys, section_name, config_filename)  # prints to stderr
  → builds DataConfig, ModelConfig, etc.
  → validate_config(config, config_filename)
      → collects errors into list[str]
      → raises ConfigError(filename + "\n" + "\n".join(errors)) if non-empty

ConfigError(ValueError)  # subclass so existing except ValueError: still works
```

## Testing

Extend `tests/test_config.py`:
- Error messages include the config filename
- Error messages include the received value
- Multiple validation errors reported in a single raise
- Unknown keys trigger a warning to stderr
- `ConfigError` is a subclass of `ValueError`
