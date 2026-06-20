# CLI Experience Overhaul — Design

**Date:** 2026-06-19
**Status:** Approved (design); pending implementation plan
**Scope:** Front-door usability and robustness of the `allostery` CLI. Three cohesive pillars — consistent error handling, dual human/machine output modes, and a config-driven end-to-end `workflow` command — plus documentation. Accuracy work (score calibration, uncertainty, candidate-extraction tuning) is explicitly deferred to a later cycle.

## 1. Motivation

The CLI has grown to four subcommands (`run`, `check`, `analyze`, `interpret`) added across several cycles, and the experience is now inconsistent:

- **Errors leak tracebacks.** Only `check` wraps execution in try/except. `analyze`, `interpret`, and `run` let exceptions propagate, so a missing file, malformed CSV, or empty-after-`--top-k` graph dumps a raw Python stack trace — burying the careful `ValueError` messages already written into `read_scores_csv` and `load_trajectory`.
- **The new `interpret` command is undocumented.** The README "Commands" section still lists only three commands; the feature is invisible to new users.
- **No end-to-end path.** Users hand-chain `run` → `analyze` → `interpret`. There is no single reproducible artifact describing a whole job.
- **No machine-readable output.** The tool is run both interactively and from scripts, but every command prints free-form text with no `--json`/`--quiet` mode and no documented exit codes.

This subsystem makes the CLI behave consistently for both humans and scripts, without touching model training, scoring, or the scientific outputs themselves.

## 2. Goals / Non-Goals

**Goals**
- Consistent, traceback-free error handling across every subcommand, with documented exit codes and a `--debug` escape hatch.
- A thin presentation layer so every command renders identically in three modes: human-readable default, `--quiet`, and `--json`.
- A config-driven `allostery workflow config.yaml` that runs train → score → analyze → interpret end to end, driven by optional `analyze:` / `interpret:` config sections.
- Document `interpret` and `workflow` in the README and in per-command `--help` epilogs.

**Non-Goals**
- No changes to model training, scoring, or the numerical results.
- No accuracy work (score normalization, uncertainty, ensembling, candidate-cutoff tuning) — deferred to its own cycle.
- No new model families or config modes beyond the additive `workflow` command and its optional config sections.
- The standalone `analyze`/`interpret` commands keep their current CSV-driven behavior; only their *rendering* and *error handling* change.

## 3. Architecture

Three pillars sit at the CLI front door and share one presentation layer:

```
                         ┌─────────────────────────────┐
   argv ──► build_parser │  global flags:               │
                         │  --debug --quiet --json      │
                         └──────────────┬──────────────┘
                                        ▼
                            main() dispatch wrapper
                    (catches user errors → clean message + exit code)
                                        │
        ┌───────────────┬───────────────┼───────────────┬──────────────┐
        ▼               ▼               ▼               ▼              ▼
      check           run           analyze         interpret      workflow  ◄── new
        │               │               │               │              │
        └───────────────┴───────────────┴───────────────┴──────────────┘
                                        │ each returns a structured Result
                                        ▼
                        presentation layer  (render Result as:
                          default human text │ --quiet │ --json)
```

Every subcommand builds a `Result` object and returns it. A single layer decides how to *render* it; a single wrapper decides how to handle *failure*. The new `workflow` command is an orchestrator that reuses the existing pipeline functions.

**New / changed files**
- `src/allostery/cli.py` — global flags, dispatch wrapper, thinner per-command branches.
- `src/allostery/cli_output.py` *(new)* — `Result` dataclass + three renderers.
- `src/allostery/cli_errors.py` *(new)* — error taxonomy and error→exit-code mapping.
- `src/allostery/pipeline/workflow.py` *(new)* — `run_workflow(config)` orchestrator.
- `src/allostery/config.py` — optional `analyze:` / `interpret:` sections + dataclasses + validation.
- `src/allostery/pipeline/analyze.py` — optional write-to-file so `workflow` can persist the network report.
- `README.md` + `--help` epilogs — document `interpret` and `workflow`.

## 4. Components

### 4.1 Error handling (`cli_errors.py`)

A documented exit-code taxonomy:

| Code | Meaning | Triggers |
|---|---|---|
| `0` | success | — |
| `1` | user / input error | `ConfigError`, missing file, malformed CSV, empty graph after `--top-k` |
| `2` | usage error | argparse already exits 2 on bad arguments — left untouched |
| `3` | external / backend error | LLM network or timeout failure, missing optional backend dependency (`ImportError`) |

- The dispatch wrapper in `main()` wraps command execution in one try/except. Known exception types map to codes `1` or `3` and render a **clean one-line message to stderr** through the presentation layer (so `--json` wraps it too). No traceback.
- The existing helpful messages in `read_scores_csv` / `load_trajectory` *become* the user-facing text unchanged.
- `--debug` re-raises everything for the full traceback.
- Unexpected exceptions: without `--debug`, print "internal error; rerun with --debug for details" and exit `1`; with `--debug`, re-raise.
- Edge case to close during implementation: ensure `analyze` / `interpret` raise a clear `ValueError` ("increase --top-k") on an empty graph rather than a bare `IndexError`.

### 4.2 Presentation layer (`cli_output.py`)

Every command stops calling `print()` directly and builds a `Result`:

```python
@dataclass
class Result:
    command: str                  # e.g. "interpret"
    status: str                   # "ok" | "error"
    summary: str                  # human-readable text
    data: dict[str, Any]          # machine fields: counts, params, stages
    artifacts: list[Path]         # files written
    error: str | None = None
```

Three renderers, selected by mutually-exclusive global flags:

- **default** → human-readable text (kept close to today's output so existing behavior holds) followed by artifact paths.
- **`--quiet`** → suppress banners/progress; on success emit only artifact paths (or nothing); errors still go to stderr.
- **`--json`** → a single JSON object on stdout (`status`, `command`, `data`, `artifacts`, `error`) and nothing else.

All progress / stage messages go to **stderr** so `--json` stdout stays clean for piping. `--json` and `--quiet` form an argparse mutually-exclusive group.

### 4.3 Workflow command + config extension

`allostery workflow config.yaml` runs the existing pipeline to produce the scores CSV, then post-processes per two optional config sections:

```yaml
analyze:                 # optional → AnalyzeConfig
  top_k: 20
  source: "A:1 GLY"      # source/sink must be both-or-neither
  sink: "A:3 SER"
  top_paths: 5
  top_hubs: 10
  out_path: outputs/network.txt
interpret:               # optional → InterpretConfig
  llm: none              # none | ollama | anthropic | openai
  llm_model: null
  llm_base_url: null
  pdb_path: null         # defaults to data.pdb_path → structural context for free
  top_k: 20
  top_paths: 5
  top_hubs: 10
  out_json: outputs/report.json
  out_md: outputs/report.md
```

- New `AnalyzeConfig` and `InterpretConfig` frozen dataclasses, key frozensets, and `validate_config` rules: `llm` in the allowed enum; positive ints; `source`/`sink` both-or-neither; `out_*` paths resolved relative to the config file like existing paths.
- `pipeline/workflow.py` exposes `run_workflow(config, *, backend=None) -> Result`. It first runs the pipeline **according to `config.mode`** (`train` trains only, `score` scores from an existing checkpoint, `run` does both) to produce `output.score_csv_path`, reusing the existing `_run_train`/`_run_score` logic via shared helpers. It then — if the sections are present — runs `run_network_analysis` (now able to write to `out_path`) and `run_interpretation` (pdb defaults to `data.pdb_path`). Post-processing requires a scores CSV, so a `train`-only mode with `analyze:`/`interpret:` present is a config error. The optional `backend` parameter allows test injection, mirroring `run_interpretation`.
- Stage banners (e.g. `[1/N] …`) print to stderr, where `N` is the number of stages actually scheduled for this config (skipped train/analyze/interpret stages are not counted). They are suppressed by `--quiet` and captured under `Result.data["stages"]` for `--json`.
- The standalone `run` / `analyze` / `interpret` commands keep their current behavior (only rendering + error handling change). `interpret` stays CSV-driven per its original spec; the new config sections are consumed **only** by `workflow`.

## 5. Data Flow

**Global flags** parse before dispatch and thread into the wrapper and presentation layer only — never into the pipeline functions, which stay pure and return data.

**Standalone command** (e.g. `interpret`): parse → wrapper calls the branch → branch calls `run_interpretation(...)` → builds `Result(command="interpret", status="ok", data={counts}, artifacts=[json, md])` → presentation renders per flags → exit `0`.

**Workflow:**
```
[1/4] train     → train stage of the pipeline
[2/4] score     → writes output.score_csv_path
[3/4] analyze   → run_network_analysis(csv) → writes analyze.out_path   (if analyze: present)
[4/4] interpret → run_interpretation(csv, pdb=interpret.pdb_path or data.pdb_path, …)  (if interpret: present)
                → composite Result(artifacts=[csv, network.txt, report.json, report.md])
```
Each stage appends to `Result.data["stages"]` so `--json` reports exactly what ran.

## 6. Error Handling (edge cases)

- **Backend fails mid-workflow** (LLM network/timeout at stage 4): scores + analyze are already on disk. Wrapper catches → exit `3`; the message names the failed stage and lists artifacts already written. Partial success is never silently lost.
- **Missing optional dependency** (`anthropic`/`openai` not installed): `ImportError` with install hint → exit `3`, hint shown verbatim.
- **Empty graph after `--top-k`**: `ValueError` with "increase --top-k" guidance → exit `1`.
- **`--json --quiet` together**: argparse mutually-exclusive group → usage error, exit `2`.
- **`workflow` with neither section**: valid — behaves like `run` plus a stderr note that no post-processing was configured.
- **`interpret.pdb_path` is a non-PDB trajectory without topology**: reuse `load_trajectory`'s existing error → exit `1`.

## 7. Testing Strategy

- **`cli_errors`**: each exception type maps to its exit code; `--debug` re-raises.
- **`cli_output`**: render a `Result` in all three modes — `--json` stdout parses and contains `status`/`artifacts`; `--quiet` suppresses banners; default matches the established shape.
- **`workflow`**: end-to-end on the bundled fixture with `analyze:` + `interpret:` sections and an **injected FakeBackend** (no network); assert all artifacts written and the stage list is complete. Plus a partial-failure test (backend raises) asserting exit `3` and preserved artifacts.
- **`config`**: new sections parse; validation errors fire for bad `llm` enum, a lone `source` without `sink`, and negative ints.
- **Backward-compat**: existing CLI tests continue to pass; update only where output format intentionally changed.
- No real network or model calls anywhere in the suite.

## 8. Dependencies

No new dependencies. `--json` uses stdlib `json`; optional LLM backends remain lazily imported exactly as today.

## 9. Open Questions (resolved)

- Scope: full UX overhaul (errors + docs + end-to-end glue + output polish). ✓
- Audience: both humans and scripts → human-readable default plus `--json` / `--quiet` and documented exit codes. ✓
- End-to-end shape: config-driven `workflow` command with optional `analyze:` / `interpret:` sections. ✓
- Accuracy work deferred to a later cycle. ✓
