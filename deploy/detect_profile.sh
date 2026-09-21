#!/usr/bin/env bash
set -euo pipefail

backend=${1:-sglang}
fallback=${H3_AUTO_FALLBACK_PROFILE:-}

case "${backend}" in
  sglang) fallback=${fallback:-h100x1} ;;
  vllm_omni) fallback=${fallback:-single_offload} ;;
  *) echo "Usage: $0 [sglang|vllm_omni]" >&2; exit 2 ;;
esac

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "${fallback}"
  exit 0
fi

if ! gpu_output=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null); then
  echo "${fallback}"
  exit 0
fi

all_gpu_names=()
while IFS= read -r gpu_name; do
  [[ -n "${gpu_name}" ]] && all_gpu_names+=("${gpu_name}")
done < <(printf '%s\n' "${gpu_output}")
if ((${#all_gpu_names[@]} == 0)); then
  echo "${fallback}"
  exit 0
fi

gpu_names=("${all_gpu_names[@]}")
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
  IFS=',' read -r -a visible_ids <<<"${CUDA_VISIBLE_DEVICES}"
  selected=()
  for raw_id in "${visible_ids[@]}"; do
    gpu_id=${raw_id//[[:space:]]/}
    if [[ "${gpu_id}" =~ ^[0-9]+$ ]] && ((gpu_id < ${#all_gpu_names[@]})); then
      selected+=("${all_gpu_names[gpu_id]}")
    else
      # UUID/MIG visibility is already enforced by CUDA; preserve its count and
      # use the first reported model name for topology selection.
      selected+=("${all_gpu_names[0]}")
    fi
  done
  ((${#selected[@]} > 0)) && gpu_names=("${selected[@]}")
fi

count=${#gpu_names[@]}
joined=$(printf '%s\n' "${gpu_names[@]}")

if [[ "${backend}" == "vllm_omni" ]]; then
  if ((count >= 4)) && grep -Eqi 'B300' <<<"${joined}"; then
    echo b300x4
  elif ((count >= 2)) && grep -Eqi 'RTX[[:space:]]*5090' <<<"${joined}"; then
    echo rtx5090x2
  elif ((count >= 2)) && grep -Eqi 'RTX[[:space:]]*4090' <<<"${joined}"; then
    echo rtx4090x2
  else
    echo single_offload
  fi
  exit 0
fi

if grep -Eqi 'H100' <<<"${joined}"; then
  echo h100x1
else
  echo "${fallback}"
fi
