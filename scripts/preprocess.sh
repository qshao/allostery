#!/usr/bin/env bash
# Validate a trajectory config before committing to a full training run.
# Checks file existence, validates the YAML, inspects the trajectory,
# and estimates how many training samples will be generated.
#
# Usage:
#   ./scripts/preprocess.sh                          # uses configs/kras_wt_influence.yaml
#   ./scripts/preprocess.sh configs/kras_wt_w25.yaml

set -euo pipefail

CONFIG="${1:-configs/kras_wt_influence.yaml}"

echo "=== Preprocessing check: $CONFIG ==="
echo ""

# ── 1. Config file existence ─────────────────────────────────────────────────
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

# ── 2. YAML schema validation ────────────────────────────────────────────────
echo "[1/4] Validating config YAML..."
allostery check "$CONFIG"
echo "      OK"
echo ""

# ── 3. Extract key fields from config ────────────────────────────────────────
TRAJ=$(python3 -c "
import yaml, sys
c = yaml.safe_load(open('$CONFIG'))
d = c.get('data', {})
print(d.get('pdb_path', d.get('trajectory_path', '')))
")

TOPO=$(python3 -c "
import yaml, sys
c = yaml.safe_load(open('$CONFIG'))
d = c.get('data', {})
print(d.get('topology_path', ''))
")

WINDOW=$(python3 -c "
import yaml
c = yaml.safe_load(open('$CONFIG'))
print(c['data']['window_size'])
")

STRIDE=$(python3 -c "
import yaml
c = yaml.safe_load(open('$CONFIG'))
print(c['data']['stride'])
")

TIMESTEP=$(python3 -c "
import yaml
c = yaml.safe_load(open('$CONFIG'))
print(c['data']['time_step'])
")

# ── 4. Trajectory file existence ─────────────────────────────────────────────
echo "[2/4] Checking trajectory files..."
if [[ ! -f "$TRAJ" ]]; then
    echo "ERROR: trajectory not found: $TRAJ" >&2
    exit 1
fi
TRAJ_SIZE=$(du -h "$TRAJ" | cut -f1)
echo "      trajectory : $TRAJ ($TRAJ_SIZE)"

if [[ -n "$TOPO" ]]; then
    if [[ ! -f "$TOPO" ]]; then
        echo "ERROR: topology not found: $TOPO" >&2
        exit 1
    fi
    TOPO_SIZE=$(du -h "$TOPO" | cut -f1)
    echo "      topology   : $TOPO ($TOPO_SIZE)"
fi
echo ""

# ── 5. Trajectory inspection ─────────────────────────────────────────────────
echo "[3/4] Inspecting trajectory..."
python3 - <<PYEOF
import sys

traj_path = "$TRAJ"
topo_path = "$TOPO"

# Try MDAnalysis then MDTraj
loaded = False
try:
    import MDAnalysis as mda
    kwargs = {"topology": topo_path} if topo_path else {}
    u = mda.Universe(topo_path if topo_path else traj_path, traj_path if topo_path else None) \
        if topo_path else mda.Universe(traj_path)
    ca = u.select_atoms("name CA")
    n_frames = len(u.trajectory)
    n_residues = ca.n_atoms
    # Get chain IDs
    chain_ids = sorted(set(str(a.segid).strip() or "A" for a in ca))
    print(f"      backend    : MDAnalysis")
    print(f"      frames     : {n_frames}")
    print(f"      CA atoms   : {n_residues}")
    print(f"      chains     : {', '.join(chain_ids) if chain_ids else 'A'}")
    loaded = True
except Exception:
    pass

if not loaded:
    try:
        import mdtraj as md
        t = md.load(traj_path, top=topo_path if topo_path else None)
        ca_idx = t.topology.select("name CA")
        n_frames = t.n_frames
        n_residues = len(ca_idx)
        chain_ids = sorted(set(
            chr(ord('A') + min(t.topology.atom(i).residue.chain.index, 25))
            if t.topology.atom(i).residue.chain.chain_id is None
            else str(t.topology.atom(i).residue.chain.chain_id)
            for i in ca_idx
        ))
        print(f"      backend    : MDTraj")
        print(f"      frames     : {n_frames}")
        print(f"      CA atoms   : {n_residues}")
        print(f"      chains     : {', '.join(chain_ids) if chain_ids else 'A'}")
        loaded = True
    except Exception as e:
        print(f"      WARNING: could not inspect trajectory ({e})", file=sys.stderr)

PYEOF
echo ""

# ── 6. Sample count estimate ─────────────────────────────────────────────────
echo "[4/4] Estimating training samples..."
python3 - <<PYEOF
import yaml, math

c = yaml.safe_load(open("$CONFIG"))
d = c["data"]
window   = int(d["window_size"])
stride   = int(d["stride"])
timestep = float(d["time_step"])
val_frac = float(c.get("training", {}).get("validation_fraction", 0.1))

# Read frame count from trajectory inspection above (re-detect)
traj_path = d.get("pdb_path", d.get("trajectory_path", ""))
topo_path = d.get("topology_path", "")

n_frames = None
try:
    import MDAnalysis as mda
    u = mda.Universe(topo_path, traj_path) if topo_path else mda.Universe(traj_path)
    n_frames = len(u.trajectory)
except Exception:
    pass

if n_frames is None:
    try:
        import mdtraj as md
        t = md.load(traj_path, top=topo_path if topo_path else None)
        n_frames = t.n_frames
    except Exception:
        pass

if n_frames is not None:
    n_samples = max(0, (n_frames - window) // stride + 1)
    n_train   = int(n_samples * (1 - val_frac))
    n_val     = n_samples - n_train
    duration_ns = n_frames * timestep / 1000.0
    window_ns   = window * timestep / 1000.0
    print(f"      trajectory duration : {duration_ns:.1f} ns  ({n_frames} frames @ {timestep:.0f} ps/frame)")
    print(f"      window size         : {window} frames = {window_ns:.2f} ns")
    print(f"      stride              : {stride} frame(s)")
    print(f"      total samples       : {n_samples}  (train {n_train} / val {n_val})")
else:
    print("      (could not estimate — trajectory unreadable)")

PYEOF

echo ""
echo "=== Preprocess check complete — ready to train ==="
