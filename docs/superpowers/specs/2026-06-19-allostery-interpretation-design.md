# Allostery Interpretation Subsystem — Design

**Date:** 2026-06-19
**Status:** Approved (design); pending implementation plan
**Scope:** Subsystem C of a three-part enhancement (C = post-process interpretation, B = preprocessing QC/cleaning, A = shared LLM backend folded into C). This spec covers **C only.** B gets its own spec → plan → implementation cycle later.

## 1. Motivation

The pipeline currently produces a residue-pair scores CSV from one of three model families (`relational`, `cri`, `influence`), and `allostery analyze` builds a single undirected graph reporting connected components, betweenness-centrality hubs, and one source→sink channel set. That output is purely topological — it lists numbers, not biology.

This subsystem turns the deep-learning outcome into **candidate allosteric networks** and an optional **biological interpretation** of them, so a user can see not just "residues A:45 and B:88 are coupled" but "this module spans the dimer interface and likely couples the catalytic loop to the regulatory site."

## 2. Goals / Non-Goals

**Goals**
- Extract candidate allosteric structures from a scores CSV: communities/modules, candidate pathways, hub/bottleneck residues, coupled-pair clusters.
- Compute honest, CA-derivable structural context (per-residue RMSF, contact/burial proxy, per-candidate geometry) to ground interpretation.
- Always emit a deterministic structured report (JSON + markdown), usable with no LLM available.
- Offer an **opt-in** LLM interpretation layer with a **pluggable backend**: local (Ollama) or cloud API (Anthropic / OpenAI).
- Keep the deterministic report authoritative; the LLM annotates, never overwrites.

**Non-Goals**
- No external biological databases (UniProt/Pfam/SIFTS). Grounding is computed structure + topology; biological knowledge comes from the LLM's parametric knowledge.
- No DSSP/true-SASA computation — the project is CA-only and we will not overclaim secondary structure.
- No changes to model training or scoring. This consumes existing outputs.
- Not a config-driven pipeline `mode`; it is a standalone CLI subcommand like `analyze`.

## 3. Architecture

```
scores.csv ─────────────► CandidateExtractor ──► candidates
reference structure ────► StructuralContext ───► per-residue + per-candidate features
(PDB / trajectory)             │
                               ▼
                        ReportBuilder ──► report.json + report.md   (ALWAYS)
                               │
                               ▼  (only if --llm != none)
                     InterpretationEngine ◄── LLMBackend (ollama | anthropic | openai)
                               │
                               ▼
                report.json + report.md  (enriched with interpretation)
```

New package `src/allostery/interpret/`:

```
interpret/
  __init__.py
  candidates.py     # CandidateExtractor + candidate dataclasses
  structure.py      # StructuralContext: CA-derivable features
  report.py         # ReportBuilder: candidates+context -> JSON + markdown
  engine.py         # InterpretationEngine: prompt build, backend call, parse/merge
  prompts.py        # prompt template + JSON response schema
  llm/
    __init__.py     # LLMBackend protocol + make_backend(...) factory
    ollama.py       # OllamaBackend (HTTP, lazy import)
    anthropic.py    # AnthropicBackend (official SDK, lazy import)
    openai.py       # OpenAIBackend (official SDK, lazy import)
```

## 4. Components

### 4.1 CandidateExtractor (`candidates.py`)

Pure/deterministic. Input: the weighted graph built by the existing `network.build_graph(read_scores_csv(...), top_k)`. Output: a `CandidateSet` dataclass with four lists.

Implemented in-house (no `networkx`, matching the existing convention of hand-rolled Dijkstra/Brandes/Yen in `network.py`):

1. **Communities/modules** — greedy modularity maximization (Clauset–Newman–Moore agglomerative merging on the weighted graph). Each `Community` = `{members: [node_idx], modularity_contribution, internal_weight}`.
2. **Candidate pathways** — extends the existing Yen/Dijkstra code: enumerate the top-N highest-weight shortest paths between the strongest-coupled residue pairs (seed pairs = the top-scoring inter-community edges). Each `Pathway` = `{nodes, total_weight, hop_count}`.
3. **Hub/bottleneck residues** — reuse `betweenness_centrality`; emit a ranked `Hub` list `{node_idx, centrality, degree}`.
4. **Coupled-pair clusters** — single-linkage agglomerative clustering over the top-K raw pairs by shared residues (no full-graph dependency). Each `Cluster` = `{members, pair_count, mean_score}`.

All candidate members carry the residue label `"CHAIN:NUM NAME"` so downstream stages don't re-derive it.

### 4.2 StructuralContext (`structure.py`)

Loads the reference structure/trajectory via the existing `io.trajectory.load_trajectory(pdb_path, topology_path)` (CA coordinates only). Computes only honest CA-derivable features:

- **Per-residue RMSF** — root-mean-square fluctuation across frames after centering.
- **Contact number / coordination** — count of CA within a cutoff (default 8.0 Å) in the mean structure; a burial proxy.
- **CA–CA contact map** — boolean contacts at cutoff (used for per-candidate geometry).
- **Per-candidate geometry** — for each candidate: spatial compactness (radius of gyration of member CAs in the mean structure), sequence span (max−min residue index within a chain), and pairwise CA distances for pathway steps.

If no structure is supplied, `StructuralContext` is `None` and the report simply omits structural fields. Single-frame PDBs yield zero RMSF (documented, not an error).

### 4.3 ReportBuilder (`report.py`)

Renders `CandidateSet` (+ optional `StructuralContext`) to:

- **JSON** — a versioned schema: `{schema_version, source, parameters, candidates: {communities, pathways, hubs, clusters}, structural_context?, interpretation?}`. Each candidate item is machine-readable with stable keys + an `evidence` block (its structural/topological features). `interpretation` is absent until the engine fills it.
- **Markdown** — human-readable sections mirroring the JSON, one per candidate type, with a table per candidate.

Always runs; this is the no-LLM deliverable. JSON is the contract the engine and tests consume.

### 4.4 LLMBackend (`llm/`)

```python
class LLMBackend(Protocol):
    def generate_json(self, system: str, user: str, schema: dict) -> dict: ...
```

A single narrow method: given a system prompt, a user prompt, and a JSON schema, return a parsed dict that conforms (best-effort; engine validates). One method keeps all three adapters trivial and uniform.

- **`make_backend(name, *, model, base_url) -> LLMBackend`** — factory. `name in {ollama, anthropic, openai}`.
- **Adapters use lazy imports** (same pattern as `io.trajectory`'s MDAnalysis/MDTraj handling) so none of `requests`/`anthropic`/`openai` is a hard dependency. A missing dependency raises a clear `ImportError` with the install command.
- **API keys come from environment only**: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. Never parameters, never config, never logged.

**AnthropicBackend** (per the `claude-api` reference): official `anthropic` SDK, default model `claude-opus-4-8`, `thinking={"type": "adaptive"}`, structured output via `output_config={"format": {"type": "json_schema", "schema": schema}}`, `max_tokens=16000`. Parse the single text block as JSON.

**OpenAIBackend**: official `openai` SDK, JSON-schema response format, default model `gpt-4.1` (overridable via `--llm-model`).

**OllamaBackend**: POST to `{base_url}/api/chat` (default `base_url=http://localhost:11434`) with `format` set to the JSON schema and `stream=false`; default model from `--llm-model` (e.g. `qwen3`, `gemma3`). Uses `requests` (lazy import) or stdlib `urllib` fallback.

### 4.5 InterpretationEngine (`engine.py`)

1. Builds a grounded prompt **per candidate** from the JSON report's `evidence` block (residue labels, RMSF, burial, geometry, topological role) using `prompts.py`.
2. Calls `backend.generate_json(system, user, RESPONSE_SCHEMA)`.
3. Validates the result against `RESPONSE_SCHEMA`; on failure, retry once; on second failure, attach a `raw_text` fallback with a warning flag.
4. Merges each interpretation into the report's candidate item under `interpretation`, then re-emits JSON + markdown.

**Response schema** (per candidate): `{summary, mechanism_hypothesis, key_residues: [{label, role, evidence_refs}], confidence: "low"|"medium"|"high", parametric: bool, caveats}`.

**Guardrails** (in the system prompt): "Ground every statement in the supplied evidence. If you assert a functional role from prior knowledge not present in the evidence, set `parametric: true` and lower `confidence`. Do not invent residues or numbers." The deterministic candidate data is never mutated by the engine — interpretation is purely additive.

## 5. CLI & Configuration

New subcommand (mirrors the non-config `analyze`):

```
allostery interpret <scores_csv>
    [--pdb <structure>] [--topology <top>]
    [--top-k N] [--top-paths N] [--top-hubs N]
    [--out-json PATH] [--out-md PATH]
    [--llm none|ollama|anthropic|openai]   (default: none)
    [--llm-model NAME] [--llm-base-url URL]
```

- `--llm none` (default) → deterministic report only; no backend imported.
- `--pdb` omitted → structural context skipped; candidates + topology still produced.
- Output defaults: `<scores_csv stem>.interpret.json` and `.md` next to the CSV.
- Wires into `cli.py` as a new `interpret` branch alongside `analyze`, with a `pipeline/interpret.py` orchestrator (`run_interpretation(...)`) analogous to `pipeline/analyze.py`'s `run_network_analysis`.

## 6. Data Flow

1. `read_scores_csv` + `build_graph(top_k)` (reused from `network.py`).
2. `CandidateExtractor.extract(graph, top_paths, top_hubs)` → `CandidateSet`.
3. If `--pdb`: `StructuralContext.compute(trajectory, candidates, cutoff)`; else `None`.
4. `ReportBuilder.build(candidates, context, params)` → JSON + markdown written. **Stop here if `--llm none`.**
5. `make_backend(...)` → `InterpretationEngine.interpret(report, backend)` → enriched JSON + markdown rewritten.

## 7. Error Handling

- Missing/empty CSV, missing columns → reuse `read_scores_csv`'s existing `ValueError`s.
- `--pdb` not found / unreadable → `ValueError` from `load_trajectory`.
- LLM backend dependency missing → `ImportError` with install hint; deterministic report is already written, so partial success is preserved.
- Backend network/timeout error → surfaced with the backend name; deterministic report already on disk.
- Malformed LLM JSON → one retry, then `raw_text` fallback flagged in the report (not a crash).
- Empty graph after `top_k` filtering → `ValueError` with guidance to raise `--top-k`.

## 8. Testing Strategy

- **Candidates**: deterministic unit tests on small synthetic graphs with known communities / shortest paths / hubs / clusters.
- **StructuralContext**: unit tests against the existing `tests/fixtures` `tiny_trajectory.pdb` (known RMSF, contact counts, geometry); single-frame → zero RMSF case.
- **ReportBuilder**: JSON schema round-trip; markdown contains expected section headers; `interpretation` absent when no engine run.
- **LLM layer**: a `FakeBackend` returning canned schema-valid JSON drives `InterpretationEngine` tests deterministically (validation pass, retry path, raw-text fallback). Provider adapters tested with **mocked** HTTP/SDK clients — **no real network or model calls** in the suite.
- **CLI**: `--llm none` end-to-end writes JSON + markdown; `--pdb` omitted path; output-path defaulting.

## 9. Dependencies

- No new hard dependencies. `requests`/`anthropic`/`openai` are optional, lazily imported, only when the matching backend is selected.
- Suggested optional extras in `pyproject.toml`: `interpret-anthropic = ["anthropic"]`, `interpret-openai = ["openai"]`, `interpret-ollama = ["requests"]`.

## 10. Open Questions (resolved)

- Backend set: Ollama (local) + Anthropic + OpenAI (cloud). ✓
- Bio knowledge: LLM parametric + computed structural features only. ✓
- LLM optional, JSON + markdown output. ✓
- Standalone subcommand, not a config mode. ✓
