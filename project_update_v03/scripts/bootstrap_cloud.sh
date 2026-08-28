#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-scail2}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_ROOT="${CVF_VENDOR_ROOT:-$ROOT/vendor}"
PROVENANCE_ROOT="$ROOT/artifacts/provenance/env"
TORCH_INDEX_URL="${CVF_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYTHON_BIN="${CVF_PYTHON_BIN:-python3}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "bootstrap_cloud.sh must run on the rented Linux GPU host" >&2
  exit 4
fi

install_system_packages() {
  local runner=()
  if [[ "$(id -u)" -ne 0 ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
      echo "Need root or sudo to install system packages" >&2
      exit 4
    fi
    runner=(sudo)
  fi
  "${runner[@]}" apt-get update
  "${runner[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-venv python3-dev git git-lfs ffmpeg build-essential ninja-build \
    libgl1 libglib2.0-0 libjpeg-dev curl ca-certificates pkg-config
}

clone_at() {
  local url="$1"
  local revision="$2"
  local target="$3"
  local recursive="${4:-false}"
  if [[ ! -d "$target/.git" ]]; then
    if [[ "$recursive" == "true" ]]; then
      git clone --recursive "$url" "$target"
    else
      git clone "$url" "$target"
    fi
  fi
  git -C "$target" fetch --tags --force origin
  git -C "$target" checkout --detach "$revision"
  if [[ "$recursive" == "true" ]]; then
    git -C "$target" submodule sync --recursive
    git -C "$target" submodule update --init --recursive
  fi
  local actual
  actual="$(git -C "$target" rev-parse HEAD)"
  if [[ "$actual" != "$revision" ]]; then
    echo "Revision mismatch for $target: $actual != $revision" >&2
    exit 3
  fi
}

create_env() {
  local name="$1"
  local env_path="$ROOT/.envs/$name"
  "$PYTHON_BIN" -m venv "$env_path"
  "$env_path/bin/python" -m pip install --upgrade pip setuptools wheel
}

verify_torch() {
  local python_bin="$1"
  "$python_bin" - <<'PY'
import json
import torch
payload = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}
print(json.dumps(payload, ensure_ascii=False))
if not payload["cuda_available"]:
    raise SystemExit("PyTorch cannot see CUDA")
PY
}

install_system_packages
"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
    raise SystemExit(f"Python 3.10-3.12 required, got {sys.version.split()[0]}")
PY
mkdir -p "$VENDOR_ROOT" "$PROVENANCE_ROOT" "$ROOT/.envs"
git lfs install --skip-repo

"$PYTHON_BIN" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$ROOT/.venv/bin/python" -m pip install -e "$ROOT"

case "$BACKEND" in
  scail2)
    clone_at https://github.com/zai-org/SCAIL-2.git \
      78fe19576bb06be96c2375e088574a262a300edb "$VENDOR_ROOT/SCAIL-2" true
    create_env scail2
    SCAIL_ENV="$ROOT/.envs/scail2"
    "$SCAIL_ENV/bin/python" -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
    MAX_JOBS="${MAX_JOBS:-8}" "$SCAIL_ENV/bin/python" -m pip install -r "$VENDOR_ROOT/SCAIL-2/requirements.txt"
    verify_torch "$SCAIL_ENV/bin/python"

    create_env scail2-pose
    POSE_ENV="$ROOT/.envs/scail2-pose"
    "$POSE_ENV/bin/python" -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
    "$POSE_ENV/bin/python" -m pip install -r "$VENDOR_ROOT/SCAIL-2/SCAIL-Pose/requirements.txt"
    verify_torch "$POSE_ENV/bin/python"
    "$SCAIL_ENV/bin/python" -m pip freeze > "$PROVENANCE_ROOT/scail2.freeze.txt"
    "$POSE_ENV/bin/python" -m pip freeze > "$PROVENANCE_ROOT/scail2-pose.freeze.txt"
    ;;
  wan22_animate|wan22_s2v)
    clone_at https://github.com/Wan-Video/Wan2.2.git \
      42bf4cfaa384bc21833865abc2f9e6c0e67233dc "$VENDOR_ROOT/Wan2.2" false
    create_env wan22
    WAN_ENV="$ROOT/.envs/wan22"
    "$WAN_ENV/bin/python" -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
    # flash-attn imports torch while evaluating its build metadata, so PEP 517
    # build isolation fails even when torch is already installed in this env.
    # Install the remaining Wan requirements first, then compile flash-attn
    # against the locked environment without build isolation.
    grep -vE '^[[:space:]]*flash_attn([[:space:]]|$)' \
      "$VENDOR_ROOT/Wan2.2/requirements.txt" \
      > "$PROVENANCE_ROOT/wan22.requirements.no-flash-attn.txt"
    "$WAN_ENV/bin/python" -m pip install \
      -r "$PROVENANCE_ROOT/wan22.requirements.no-flash-attn.txt"
    # Wan imports its S2V and Animate modules eagerly, although their runtime
    # dependencies are split into optional requirement files. Pin a NumPy 1.x
    # compatible audio stack and the PEFT version required by current diffusers.
    "$WAN_ENV/bin/python" -m pip install \
      "numpy==1.26.4" "scipy==1.15.3" "decord==0.6.0" \
      "librosa==0.11.0" "peft==0.17.1"
    FLASH_ARCHS="${CVF_FLASH_ATTN_CUDA_ARCHS:-$("$WAN_ENV/bin/python" -c \
      'import torch; major, minor = torch.cuda.get_device_capability(); print(f"{major}{minor}")')}"
    FLASH_ATTN_CUDA_ARCHS="$FLASH_ARCHS" MAX_JOBS="${MAX_JOBS:-8}" \
      "$WAN_ENV/bin/python" -m pip install flash-attn --no-build-isolation
    verify_torch "$WAN_ENV/bin/python"
    "$WAN_ENV/bin/python" -m pip freeze > "$PROVENANCE_ROOT/wan22.freeze.txt"
    ;;
  wan_animate2)
    clone_at https://github.com/Wan-Video/Wan-Animate-2.git \
      3ad2fef7d61d6200c9c653e0fe47be7616b323f3 "$VENDOR_ROOT/Wan-Animate-2" true
    create_env wan-animate2
    WAN2_ENV="$ROOT/.envs/wan-animate2"
    "$WAN2_ENV/bin/python" -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
    "$WAN2_ENV/bin/python" -m pip install -r "$VENDOR_ROOT/Wan-Animate-2/requirements.txt"
    FLASH_ARCHS="${CVF_FLASH_ATTN_CUDA_ARCHS:-$("$WAN2_ENV/bin/python" -c \
      'import torch; major, minor = torch.cuda.get_device_capability(); print(f"{major}{minor}")')}"
    FLASH_ATTN_CUDA_ARCHS="$FLASH_ARCHS" MAX_JOBS="${MAX_JOBS:-8}" \
      "$WAN2_ENV/bin/python" -m pip install flash-attn --no-build-isolation
    "$WAN2_ENV/bin/python" -m pip install -e "$VENDOR_ROOT/Wan-Animate-2"
    verify_torch "$WAN2_ENV/bin/python"
    "$WAN2_ENV/bin/python" -m pip freeze > "$PROVENANCE_ROOT/wan-animate2.freeze.txt"
    ;;
  *)
    echo "Unsupported backend: $BACKEND" >&2
    exit 2
    ;;
esac

"$ROOT/.venv/bin/cvf" doctor --profile "${CVF_GPU_PROFILE:-rtx_pro_6000_96gb}"
echo "Cloud bootstrap complete for $BACKEND"
