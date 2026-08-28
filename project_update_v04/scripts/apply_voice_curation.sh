#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RVC_ROOT="${CVF_VENDOR_ROOT:-$ROOT/vendor}/RVC"
RVC_PY="$ROOT/.envs/rvc/bin/python"
SOURCE_EXP="${CVF_RVC_SOURCE_EXPERIMENT:-ivan_grozny_pilot}"
TARGET_EXP="${CVF_RVC_EXPERIMENT:-ivan_grozny_pilot_curated}"
INCLUDE_FILE="$ROOT/configs/voice/ivan_rvc_curation.include.txt"
SOURCE_DIR="$RVC_ROOT/logs/$SOURCE_EXP"
TARGET_DIR="$RVC_ROOT/logs/$TARGET_EXP"
PROVENANCE="$ROOT/artifacts/provenance/voice_dataset_curation.json"

[[ -x "$RVC_PY" ]] || { echo "Run scripts/bootstrap_voice.sh first" >&2; exit 4; }
[[ -f "$INCLUDE_FILE" ]] || { echo "Missing curation list: $INCLUDE_FILE" >&2; exit 5; }
[[ -d "$SOURCE_DIR/0_gt_wavs" && -d "$SOURCE_DIR/1_16k_wavs" ]] || {
  echo "Run scripts/prepare_voice_dataset.sh first" >&2
  exit 6
}

mapfile -t INCLUDED < <(sed -e 's/[[:space:]]*$//' -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' "$INCLUDE_FILE")
[[ ${#INCLUDED[@]} -gt 0 ]] || { echo "Curation list is empty" >&2; exit 7; }

mkdir -p "$TARGET_DIR/0_gt_wavs" "$TARGET_DIR/1_16k_wavs" "$(dirname "$PROVENANCE")"
for name in "${INCLUDED[@]}"; do
  [[ "$name" =~ ^[A-Za-z0-9_-]+\.wav$ ]] || { echo "Unsafe slice name: $name" >&2; exit 8; }
  for subdir in 0_gt_wavs 1_16k_wavs; do
    src="$SOURCE_DIR/$subdir/$name"
    dst="$TARGET_DIR/$subdir/$name"
    [[ -f "$src" ]] || { echo "Missing selected slice: $src" >&2; exit 9; }
    if [[ -e "$dst" ]]; then
      cmp -s "$src" "$dst" || { echo "Refusing to overwrite different file: $dst" >&2; exit 10; }
    else
      cp -p "$src" "$dst"
    fi
  done
done

export CVF_CURATED_ROOT="$TARGET_DIR/0_gt_wavs"
export CVF_CURATED_PROVENANCE="$PROVENANCE"
export CVF_CURATED_SOURCE_EXP="$SOURCE_EXP"
export CVF_CURATED_TARGET_EXP="$TARGET_EXP"
"$RVC_PY" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import soundfile as sf

root = Path(os.environ["CVF_CURATED_ROOT"])
output = Path(os.environ["CVF_CURATED_PROVENANCE"])
rows = []
for path in sorted(root.glob("*.wav")):
    info = sf.info(str(path))
    rows.append(
        {
            "file": path.name,
            "duration_seconds": round(info.duration, 6),
            "sample_rate": info.samplerate,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
payload = {
    "schema_version": 1,
    "source_experiment": os.environ["CVF_CURATED_SOURCE_EXP"],
    "target_experiment": os.environ["CVF_CURATED_TARGET_EXP"],
    "slice_count": len(rows),
    "duration_seconds": round(sum(row["duration_seconds"] for row in rows), 6),
    "slices": rows,
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
if payload["slice_count"] != 15 or abs(payload["duration_seconds"] - 55.5) > 0.01:
    raise SystemExit("Curated dataset does not match the reviewed 15-slice/55.5-second set")
PY

echo "Curated voice dataset is ready at $TARGET_DIR"
