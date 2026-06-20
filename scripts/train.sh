#!/usr/bin/env bash
# Train the influence model and score all residue pairs.
# Output: model checkpoint and scores CSV as specified in the config.
#
# Usage:
#   ./scripts/train.sh                          # uses configs/kras_wt_influence.yaml
#   ./scripts/train.sh configs/kras_wt_w25.yaml

set -euo pipefail

CONFIG="${1:-configs/kras_wt_influence.yaml}"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    echo "Usage: $0 [CONFIG_YAML]" >&2
    exit 1
fi

echo "=== Training: $CONFIG ==="
echo ""

START=$(date +%s)

allostery run "$CONFIG"

END=$(date +%s)
ELAPSED=$(( END - START ))
echo ""
echo "=== Training complete in ${ELAPSED}s ==="

# Print output locations from config
python3 - <<PYEOF
import yaml
c = yaml.safe_load(open("$CONFIG"))
out = c.get("output", {})
model  = out.get("model_path", "")
scores = out.get("score_csv_path", "")
if model:
    print(f"  model   : {model}")
if scores:
    print(f"  scores  : {scores}")
    print(f"")
    print(f"Next step:")
    print(f"  ./scripts/analyze.sh {scores}")
PYEOF
