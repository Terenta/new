#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$ROOT/configs/motion/ivan_scepter_two_hands_actorless_v2.yaml}"
OUTPUT="${2:-$ROOT/data/jobs/ivan_throne_actorless_pilot_001/prepared/gesture_pose_dwpose_two_hands_v2_16fps.mp4}"
EXTRA=()

if [[ "${CVF_FORCE_POSE:-}" == "YES" ]]; then
  EXTRA+=(--force)
fi

source "$ROOT/.venv/bin/activate"
cvf rights check --manifest "$ROOT/configs/rights_manifest.local.yaml"
python "$ROOT/tools/render_actorless_pose.py" \
  --config "$CONFIG" \
  --output "$OUTPUT" \
  "${EXTRA[@]}"

echo "Actorless DWPose-style control generated: $OUTPUT"
