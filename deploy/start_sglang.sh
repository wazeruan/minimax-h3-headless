#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
variant=${1:-fl2va}
profile=${H3_PROFILE:-auto}
model_path=${H3_MODEL_PATH:-MiniMaxAI/MiniMax-H3}
host=${H3_INFERENCE_HOST:-127.0.0.1}

resident_layer_budget() {
  local default=$1 value
  value=${H3_DIT_RESIDENT_LAYERS:-${default}}
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    echo "H3_DIT_RESIDENT_LAYERS must be a non-negative integer." >&2
    exit 2
  }
  printf '%s\n' "${value}"
}

single_h100_default_layers() {
  local memory=
  if command -v nvidia-smi >/dev/null 2>&1; then
    memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk 'NR == 1 {print $1}')
  fi
  if [[ "${memory}" =~ ^[0-9]+$ ]]; then
    if ((memory < 24000 || memory >= 48000)); then
      echo "This checkout supports only an H100 40 GB GPU (24–47 GB accepted); detected ${memory} MiB." >&2
      exit 1
    fi
    printf '%s\n' 8
    return
  fi
  # Keep the low-memory budget when nvidia-smi is unavailable; the scheduler
  # preflight still rejects unsupported hardware before a real launch.
  printf '%s\n' 8
}

case "${variant}" in
  fl2va) port=${H3_INFERENCE_PORT:-30010} ;;
  ref2va) port=${H3_INFERENCE_PORT:-30011} ;;
  *) echo "Usage: $0 [fl2va|ref2va]" >&2; exit 2 ;;
esac

if [[ "${profile}" == "auto" ]]; then
  profile=$("${repo_dir}/deploy/detect_profile.sh" sglang)
  echo "Auto-selected SGLang profile: ${profile}" >&2
fi

case "${profile}" in
  h100x1)
    h100_mode=${H3_H100_MODE:-speed}
    auto_resident_layers=$(single_h100_default_layers)
    case "${h100_mode}" in
      speed)
        resident_layers=$(resident_layer_budget "${auto_resident_layers}")
        ;;
      memory)
        resident_layers=$(resident_layer_budget "$([[ "${auto_resident_layers}" -lt 20 ]] && printf '%s' "${auto_resident_layers}" || printf '%s' 20)")
        ;;
      *)
        echo "H3_H100_MODE must be speed or memory." >&2
        exit 2
        ;;
    esac
    echo "Single-H100 mode: ${h100_mode} (${resident_layers} resident DiT layers)." >&2
    topology=(
      --num-gpus 1 --tp-size 1 --ulysses-degree 1 --performance-mode memory
      --layerwise-offload-components "dit,text_encoder,vae"
      --dit-offload-prefetch-size 1 --dit-layerwise-resident-layers "${resident_layers}"
    )
    topology+=(--quantization kitchen_int8 --attention-backend fa)
    topology+=(--enable-torch-compile false)
    ;;
  *)
    echo "Unknown H3_PROFILE=${profile}" >&2
    echo "This checkout supports only the h100x1 40 GB profile." >&2
    exit 2
    ;;
esac

sglang_bin=${H3_SGLANG_BIN:-"${repo_dir}/.venv/bin/sglang"}
[[ -x "${sglang_bin}" ]] || {
  echo "SGLang not found at ${sglang_bin}; run scripts/bootstrap_sglang.sh." >&2
  exit 1
}

# apache-tvm-ffi 0.1.11 can select HIP merely because /opt/rocm exists on a
# shared node. MiniMax H3 is running on an NVIDIA GPU here, so its JIT kernels
# must be compiled with CUDA rather than ROCm's hipcc.
export TVM_FFI_GPU_BACKEND=cuda
if [[ "${TVM_FFI_GPU_BACKEND}" == "cuda" ]]; then
  cuda_home=${CUDA_HOME:-${CUDA_PATH:-}}
  if ! command -v nvcc >/dev/null 2>&1 && [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
    echo "CUDA compiler (nvcc) is required for SGLang's H3 JIT kernels. Load your CUDA module, then retry." >&2
    exit 1
  fi
fi

exec "${sglang_bin}" serve \
  --model-path "${model_path}" \
  --model-variant "${variant}" \
  "${topology[@]}" \
  --host "${host}" \
  --port "${port}"
