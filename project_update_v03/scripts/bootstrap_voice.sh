#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RVC_ROOT="${CVF_VENDOR_ROOT:-$ROOT/vendor}/RVC"
RVC_REVISION=81eed5e8f68b6bed1789f682fe78cdd324495afc

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Voice bootstrap is cloud-host only" >&2
  exit 4
fi
if ! command -v python3.12 >/dev/null 2>&1; then
  echo "Current official RVC branch requires Python 3.12; use an Ubuntu 24.04 image or install python3.12." >&2
  exit 4
fi
if [[ ! -d "$RVC_ROOT/.git" ]]; then
  git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git "$RVC_ROOT"
fi
git -C "$RVC_ROOT" fetch --tags --force origin
git -C "$RVC_ROOT" checkout --detach "$RVC_REVISION"
if [[ "$(git -C "$RVC_ROOT" rev-parse HEAD)" != "$RVC_REVISION" ]]; then
  echo "RVC revision mismatch" >&2
  exit 3
fi

python3.12 -m venv "$ROOT/.envs/rvc"
RVC_PY="$ROOT/.envs/rvc/bin/python"
"$RVC_PY" -m pip install --upgrade pip setuptools wheel
"$RVC_PY" -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
"$RVC_PY" -m pip install -r "$RVC_ROOT/requirments_cu128_py312.txt"
"$RVC_PY" -m pip install "huggingface_hub==0.36.2"

cd "$RVC_ROOT"
"$ROOT/.envs/rvc/bin/hf" download lj1995/VoiceConversionWebUI --revision main \
  --include "hubert_base/*" --local-dir assets
"$ROOT/.envs/rvc/bin/hf" download lj1995/VoiceConversionWebUI rmvpe.pt --revision main \
  --local-dir assets/rmvpe
"$ROOT/.envs/rvc/bin/hf" download lj1995/VoiceConversionWebUI --revision main \
  --include "pretrained/*" --include "pretrained_v2/*" --local-dir assets
"$ROOT/.envs/rvc/bin/hf" download lj1995/VoiceConversionWebUI mute.zip --revision main \
  --local-dir .model-downloads
"$RVC_PY" -m zipfile -e .model-downloads/mute.zip logs

"$RVC_PY" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("RVC PyTorch cannot see CUDA")
PY
mkdir -p "$ROOT/artifacts/provenance/env"
"$RVC_PY" -m pip freeze > "$ROOT/artifacts/provenance/env/rvc.freeze.txt"
echo "RVC bootstrap complete; no Gradio service was exposed."
