#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-scail2}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_ROOT="${CVF_MODELS_ROOT:-/opt/cvf-models}"
RIGHTS_MANIFEST="${CVF_RIGHTS_MANIFEST:-$ROOT/configs/rights_manifest.local.yaml}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Weight download is cloud-host only" >&2
  exit 4
fi

source "$ROOT/.venv/bin/activate"
cvf rights check --manifest "$RIGHTS_MANIFEST"
cvf models audit --backend "$BACKEND"
mkdir -p "$MODELS_ROOT"

case "$BACKEND" in
  scail2)
    if [[ "${CVF_ACCEPT_SAM3_TERMS:-}" != "YES" ]]; then
      echo "Set CVF_ACCEPT_SAM3_TERMS=YES only after the operator has accepted facebook/sam3 gated terms." >&2
      exit 3
    fi
    if [[ -z "${HF_TOKEN:-}" ]]; then
      echo "HF_TOKEN is required for gated SAM3 download" >&2
      exit 3
    fi
    "$ROOT/.envs/scail2/bin/python" -m pip install "huggingface_hub[cli]"
    "$ROOT/.envs/scail2/bin/hf" download zai-org/SCAIL-2 --local-dir "$MODELS_ROOT/SCAIL-2"
    "$ROOT/.envs/scail2/bin/hf" download facebook/sam3 sam3.pt \
      --token "$HF_TOKEN" --local-dir "$MODELS_ROOT/sam3"
    "$ROOT/.envs/scail2/bin/python" "$ROOT/vendor/SCAIL-2/convert.py" \
      --scail-dir "$MODELS_ROOT/SCAIL-2" \
      --save-path "$MODELS_ROOT/SCAIL-2/SCAIL-2.safetensors"
    ;;
  wan22_animate)
    "$ROOT/.envs/wan22/bin/python" -m pip install "huggingface_hub[cli]"
    "$ROOT/.envs/wan22/bin/hf" download Wan-AI/Wan2.2-Animate-14B \
      --local-dir "$MODELS_ROOT/Wan2.2-Animate-14B"
    ;;
  wan22_s2v)
    "$ROOT/.envs/wan22/bin/python" -m pip install "huggingface_hub[cli]" hf_transfer
    "$ROOT/.envs/wan22/bin/hf" download Wan-AI/Wan2.2-S2V-14B \
      --local-dir "$MODELS_ROOT/Wan2.2-S2V-14B"
    ;;
  wan_animate2)
    "$ROOT/.envs/wan-animate2/bin/python" -m pip install "huggingface_hub[cli]"
    "$ROOT/.envs/wan-animate2/bin/hf" download Wan-AI/Wan2.2-Animate-2-14B \
      --local-dir "$MODELS_ROOT/Wan2.2-Animate-2-14B"
    ;;
  *)
    echo "Unsupported backend: $BACKEND" >&2
    exit 2
    ;;
esac

echo "Weights downloaded. Hash them with 'cvf models lock' before inference."
