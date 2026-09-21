#!/usr/bin/env bash
#SBATCH --job-name=h3-generate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sbatch [SLURM OPTIONS] deploy/slurm/h3-generate.sbatch [GENERATION OPTIONS] [PROMPT] [OUTPUT.mp4]

Starts FL2VA on the allocated H100 node, generates one video, and stops the server.
Set H3_REPO_DIR when submitting from outside the repository. Use CLI options
such as `--duration 10 --aspect-ratio 9:16 --seed 123`.
EOF
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac
(( $# >= 1 )) || { usage >&2; exit 2; }
generation_args=("$@")

repo_hint=${H3_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
[[ -n "${repo_hint}" && -d "${repo_hint}" ]] || {
  echo "Cannot locate the repository. Export H3_REPO_DIR before calling sbatch." >&2
  exit 1
}
repo_dir=$(cd -- "${repo_hint}" && pwd)
cd "${repo_dir}"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env
fi
if [[ -n "${H3_MODULES_FILE:-}" && -f "${H3_MODULES_FILE}" ]]; then
  if ! type module >/dev/null 2>&1 && [[ -r /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]]; then
    # shellcheck disable=SC1091
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
  fi
  # shellcheck disable=SC1090
  source "${H3_MODULES_FILE}"
fi

for command_name in curl nvidia-smi python3; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "${command_name} is required inside the Slurm job." >&2
    exit 1
  }
done

sglang_bin=${H3_SGLANG_BIN:-"${repo_dir}/.venv/bin/sglang"}
model_path=${H3_MODEL_PATH:-${H3_MODEL_DIR:-"${repo_dir}/models/MiniMax-H3"}}
[[ -x "${sglang_bin}" ]] || {
  echo "SGLang not found at ${sglang_bin}. Run setup before submitting." >&2
  exit 1
}
[[ -f "${model_path}/model_index.json" && -d "${model_path}/FL2VA" ]] || {
  echo "FL2VA weights not found at ${model_path}. Download them before submitting." >&2
  exit 1
}

h100_found=false
while IFS=, read -r gpu_name gpu_memory; do
  gpu_name=${gpu_name# }
  gpu_memory=${gpu_memory//[[:space:]]/}
  if [[ "${gpu_name}" =~ H100 && "${gpu_memory}" =~ ^[0-9]+$ && ${gpu_memory} -ge 24000 && ${gpu_memory} -lt 48000 ]]; then
    h100_found=true
    break
  fi
done < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null)
${h100_found} || {
  echo "The Slurm job does not have an H100 40 GB GPU (24–47 GB accepted)." >&2
  exit 1
}

job_tag=${SLURM_JOB_ID:-$$}
output=${repo_dir}/outputs/h3-${job_tag}.mp4
value_option=
prompt_from_option=false
output_from_option=false
literal_arguments=false
positional_arguments=()
for argument in "${generation_args[@]}"; do
  if [[ -n "${value_option}" ]]; then
    case "${value_option}" in
      output) output=${argument}; output_from_option=true ;;
      prompt) prompt_from_option=true ;;
    esac
    value_option=
    continue
  fi
  if ${literal_arguments}; then
    positional_arguments+=("${argument}")
    continue
  fi
  case "${argument}" in
    --output) value_option=output ;;
    --prompt) value_option=prompt ;;
    --duration|--aspect-ratio|--seed|--steps|--flow-shift|--audio-flow-shift|--model|--url|--poll-interval|--timeout) value_option=skip ;;
    --) literal_arguments=true ;;
    -*) ;;
    *) positional_arguments+=("${argument}") ;;
  esac
done
if ! ${output_from_option}; then
  if ${prompt_from_option} && (( ${#positional_arguments[@]} >= 1 )); then
    output=${positional_arguments[0]}
  elif ! ${prompt_from_option} && (( ${#positional_arguments[@]} >= 2 )); then
    output=${positional_arguments[1]}
  fi
fi
server_log_dir=${H3_SLURM_LOG_DIR:-"${repo_dir}/logs"}
server_log="${server_log_dir}/sglang-${job_tag}.log"
mkdir -p -- "${server_log_dir}" "$(dirname -- "${output}")"

port=${H3_INFERENCE_PORT:-}
if [[ -z "${port}" ]]; then
  port=$(python3 - <<'PY'
import socket

with socket.socket() as server:
    server.bind(("127.0.0.1", 0))
    print(server.getsockname()[1])
PY
)
fi
[[ "${port}" =~ ^[0-9]+$ && ${port} -ge 1024 && ${port} -le 65535 ]] || {
  echo "H3_INFERENCE_PORT must be an integer from 1024 through 65535." >&2
  exit 2
}

server_pid=
stop_server() {
  local pid=${server_pid:-}
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  fi
  wait "${pid}" 2>/dev/null || true
  server_pid=
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export H3_SGLANG_BIN="${sglang_bin}"
export H3_MODEL_PATH="${model_path}"
export H3_PROFILE=${H3_PROFILE:-h100x1}
export H3_INFERENCE_HOST=127.0.0.1
export H3_INFERENCE_PORT="${port}"
export H3_SGLANG_URL="http://127.0.0.1:${port}"

echo "job=${job_tag} node=$(hostname) profile=${H3_PROFILE} port=${port}"
echo "Starting FL2VA; server log: ${server_log}"
"${repo_dir}/deploy/start_sglang.sh" fl2va >"${server_log}" 2>&1 &
server_pid=$!

startup_timeout=${H3_STARTUP_TIMEOUT_SECONDS:-1800}
[[ "${startup_timeout}" =~ ^[0-9]+$ ]] || {
  echo "H3_STARTUP_TIMEOUT_SECONDS must be a non-negative integer." >&2
  exit 2
}
deadline=$((SECONDS + startup_timeout))
while ! curl --fail --silent "${H3_SGLANG_URL}/health" >/dev/null 2>&1; do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    if wait "${server_pid}"; then
      server_status=0
    else
      server_status=$?
    fi
    server_pid=
    echo "SGLang exited during startup with status ${server_status}." >&2
    tail -n 40 "${server_log}" >&2 || true
    exit 1
  fi
  if ((SECONDS >= deadline)); then
    echo "SGLang startup timed out after ${startup_timeout} seconds." >&2
    tail -n 40 "${server_log}" >&2 || true
    exit 1
  fi
  sleep 5
done

echo "SGLang is ready; starting generation."
"${repo_dir}/scripts/generate_sglang.sh" "${generation_args[@]}" \
  --url "${H3_SGLANG_URL}" --output "${output}"
echo "Slurm generation complete: ${output}"
