#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB="${1:-$ROOT/configs/jobs/ivan_throne_actorless.yaml}"
ASSETS="${2:-$ROOT/configs/assets.actorless.local.yaml}"
PROFILE="${CVF_GPU_PROFILE:-rtx_pro_6000_96gb}"
POSE="$ROOT/data/jobs/ivan_throne_actorless_pilot_001/prepared/gesture_pose_dwpose_two_hands_v2_16fps.mp4"

source "$ROOT/.venv/bin/activate"
bash "$ROOT/scripts/remote_preflight.sh" "$PROFILE" "$JOB" "$ASSETS"
if [[ -f "$POSE" ]]; then
  echo "Using bundled actorless pose control: $POSE"
else
  bash "$ROOT/scripts/prepare_actorless_pose.sh"
fi
cvf render run --job "$JOB" --smoke

echo "Actorless smoke completed. Review it before running: cvf render run --job $JOB"
