#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/character_voice_source.mp4" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RVC_ROOT="${CVF_VENDOR_ROOT:-$ROOT/vendor}/RVC"
RVC_PY="$ROOT/.envs/rvc/bin/python"
SOURCE="$1"
EXP=ivan_grozny_pilot
DATASET="$RVC_ROOT/datasets/$EXP"
EXP_DIR="$RVC_ROOT/logs/$EXP"

source "$ROOT/.venv/bin/activate"
cvf rights check --manifest "$ROOT/configs/rights_manifest.local.yaml"
[[ -f "$SOURCE" ]] || { echo "Voice source does not exist: $SOURCE" >&2; exit 5; }
[[ -x "$RVC_PY" ]] || { echo "Run scripts/bootstrap_voice.sh first" >&2; exit 4; }

mkdir -p "$DATASET" "$EXP_DIR"
ffmpeg -hide_banner -loglevel error -y -t 85.8 -i "$SOURCE" -vn -map_metadata -1 \
  -ac 1 -ar 48000 -c:a pcm_s24le "$DATASET/source_001.wav"

cd "$RVC_ROOT"
PYTHONPATH="$RVC_ROOT" "$RVC_PY" train/preprocess.py \
  "$DATASET" 48000 8 "$EXP_DIR" False 3.7
"$RVC_PY" - <<'PY'
from pathlib import Path
import soundfile as sf
root = Path("logs/ivan_grozny_pilot/0_gt_wavs")
files = sorted(root.glob("*.wav"))
duration = sum(sf.info(str(path)).duration for path in files)
print(f"Prepared {len(files)} slices, {duration:.2f} seconds total")
print("Review every slice and remove music, effects, other speakers and clipped/noisy fragments before training.")
if duration < 600:
    print("WARNING: below the official RVC recommendation of at least 10 minutes; treat as pilot-only.")
PY
