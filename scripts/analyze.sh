#!/usr/bin/env bash
# Analyze an allosteric network from a scored pairs CSV.
# Prints hub residues, betweenness centrality, score histogram,
# and optionally exports a PyMOL visualisation script.
#
# Usage:
#   ./scripts/analyze.sh SCORES_CSV [OPTIONS]
#
# Options:
#   --top-k N           Number of top pairs to include as edges (default: 30)
#   --source RESIDUE    Source residue for pathway analysis, e.g. "A:12 GLY"
#   --sink   RESIDUE    Sink residue for pathway analysis,   e.g. "A:87 SER"
#   --top-paths N       Number of shortest paths to report (default: 5)
#   --top-hubs N        Number of hub residues to report   (default: 10)
#   --pdb PATH          Structure PDB for PyMOL export
#   --out-pml PATH      Write PyMOL .pml script to this path
#   --out PATH          Write text report to this path
#
# Examples:
#   ./scripts/analyze.sh outputs/kras_wt/influence_scores.csv
#
#   ./scripts/analyze.sh outputs/kras_wt/influence_scores.csv \
#       --top-k 30 \
#       --source "A:48 GLY" --sink "A:121 PRO" \
#       --pdb /home/qshao/WT/WT_fixed.pdb \
#       --out-pml outputs/kras_wt/network.pml \
#       --out outputs/kras_wt/network_report.txt

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 SCORES_CSV [--top-k N] [--source RESIDUE] [--sink RESIDUE]" >&2
    echo "       [--top-paths N] [--top-hubs N] [--pdb PATH] [--out-pml PATH] [--out PATH]" >&2
    exit 1
fi

SCORES="$1"
shift

if [[ ! -f "$SCORES" ]]; then
    echo "ERROR: scores CSV not found: $SCORES" >&2
    exit 1
fi

# ── Defaults ──────────────────────────────────────────────────────────────────
TOP_K=30
SOURCE=""
SINK=""
TOP_PATHS=5
TOP_HUBS=10
PDB=""
OUT_PML=""
OUT=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --top-k)     TOP_K="$2";     shift 2 ;;
        --source)    SOURCE="$2";    shift 2 ;;
        --sink)      SINK="$2";      shift 2 ;;
        --top-paths) TOP_PATHS="$2"; shift 2 ;;
        --top-hubs)  TOP_HUBS="$2";  shift 2 ;;
        --pdb)       PDB="$2";       shift 2 ;;
        --out-pml)   OUT_PML="$2";   shift 2 ;;
        --out)       OUT="$2";       shift 2 ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "=== Network analysis: $SCORES ==="
echo ""

# ── Build allostery analyze command ──────────────────────────────────────────
CMD=(allostery analyze "$SCORES" --top-k "$TOP_K" --top-paths "$TOP_PATHS" --top-hubs "$TOP_HUBS")

[[ -n "$SOURCE"  ]] && CMD+=(--source "$SOURCE")
[[ -n "$SINK"    ]] && CMD+=(--sink   "$SINK")
[[ -n "$PDB"     ]] && CMD+=(--pdb    "$PDB")
[[ -n "$OUT_PML" ]] && CMD+=(--out-pml "$OUT_PML")

# ── Run ───────────────────────────────────────────────────────────────────────
if [[ -n "$OUT" ]]; then
    mkdir -p "$(dirname "$OUT")"
    "${CMD[@]}" | tee "$OUT"
    echo ""
    echo "Report saved to: $OUT"
else
    "${CMD[@]}"
fi

[[ -n "$OUT_PML" ]] && echo "PyMOL script : $OUT_PML"
echo ""
echo "=== Analysis complete ==="
