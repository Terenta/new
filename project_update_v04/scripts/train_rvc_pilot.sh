#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RVC_ROOT="${CVF_VENDOR_ROOT:-$ROOT/vendor}/RVC"
RVC_PY="$ROOT/.envs/rvc/bin/python"
EXP="${CVF_RVC_EXPERIMENT:-ivan_grozny_pilot_curated}"
EXP_DIR="$RVC_ROOT/logs/$EXP"

if [[ "${CVF_VOICE_DATASET_APPROVED:-}" != "YES" ]]; then
  echo "Review every extracted voice slice, then set CVF_VOICE_DATASET_APPROVED=YES." >&2
  exit 3
fi
source "$ROOT/.venv/bin/activate"
cvf rights check --manifest "$ROOT/configs/rights_manifest.local.yaml"
cvf doctor --profile "${CVF_GPU_PROFILE:-rtx_pro_6000_96gb}"
[[ -x "$RVC_PY" ]] || { echo "Run scripts/bootstrap_voice.sh first" >&2; exit 4; }
[[ -d "$EXP_DIR/0_gt_wavs" ]] || { echo "Run scripts/apply_voice_curation.sh first" >&2; exit 5; }
mkdir -p "$RVC_ROOT/assets/weights" "$RVC_ROOT/assets/indices"

cd "$RVC_ROOT"
"$RVC_PY" -m train.dataset.extract_f0 cuda 1 0 0 "$EXP_DIR" True
"$RVC_PY" -m train.dataset.extract_hubert_feature \
  cuda:0 1 0 0 "$EXP_DIR" v2 True
"$RVC_PY" "$ROOT/tools/rvc_build_filelist.py" \
  --rvc-root "$RVC_ROOT" --experiment "$EXP" --sample-rate 48k --version v2 --seed 1001
RVC_CUDA_GRAPH=0 "$RVC_PY" -m train.train \
  -e "$EXP" -sr 48k -f0 1 -bs 8 -g 0 -te 200 -se 25 \
  -pg assets/pretrained_v2/f0G48k.pth -pd assets/pretrained_v2/f0D48k.pth \
  -l 1 -c 1 -sw 1 -v v2
"$RVC_PY" -m train.train_index \
  "$EXP" v2 "$RVC_ROOT/assets/indices" 8 single

CANDIDATES="$ROOT/data/voice/ivan_grozny_1973/candidates"
mkdir -p "$CANDIDATES"
find "$RVC_ROOT/assets/weights" -maxdepth 1 -type f -name "*${EXP}*.pth" -exec cp -p {} "$CANDIDATES/" \;
find "$RVC_ROOT/assets/indices" -maxdepth 1 -type f -name "*${EXP}*.index" -exec cp -p {} "$CANDIDATES/" \;
echo "Training complete. ABX checkpoints at epochs 100/150/200; copy only the approved pair into data/voice/ivan_grozny_1973/approved, update configs/voice/ivan_rvc.yaml, then lock backend rvc."
