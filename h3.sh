#!/usr/bin/env bash
set -euo pipefail

# SSH-first MiniMax H3 launcher.  The fastest verified H100 topology is four
# 80 GB cards. A single H100 uses lossless CPU/layerwise offload as a capacity
# fallback and must be started inside an existing Slurm allocation on clusters.

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
venv_bin=${H3_VENV_BIN:-"${repo_dir}/.venv/bin"}
model_dir=${H3_MODEL_DIR:-"${repo_dir}/models/MiniMax-H3"}
port=${H3_PORT:-${H3_INFERENCE_PORT:-30010}}
runtime_dir="${repo_dir}/.run"
log_dir="${repo_dir}/logs"
pid_file="${runtime_dir}/sglang.pid"
variant_file="${runtime_dir}/variant"
log_file="${log_dir}/sglang.log"

usage() {
  cat <<'EOF'
Usage: ./h3.sh COMMAND [ARGUMENTS]

Commands:
  setup                    Create the project-local SGLang environment.
  download [fl2va|ref2va]  Download one MiniMax H3 checkpoint partition.
  start [fl2va|ref2va]     Start one local SGLang server in the background.
  generate [OPTIONS] [PROMPT] [FILE]
                           Generate a 768p text-to-video-and-audio MP4.
  status                   Show the local server and health state.
  logs                     Follow the SGLang log.
  stop                     Stop the server started by this launcher.
  restart [fl2va|ref2va]   Stop, then start the selected partition.

The default FL2VA partition serves text-only (t2va) and first/last-frame
generation. Ref2VA is a separate checkpoint: on one H100, stop FL2VA before
starting Ref2VA.
EOF
}

fail() {
  echo "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "$1 is required but was not found in PATH."
}

valid_variant() {
  [[ "$1" == "fl2va" || "$1" == "ref2va" ]]
}

component_dir() {
  case "$1" in
    fl2va) printf '%s\n' "FL2VA" ;;
    ref2va) printf '%s\n' "Ref2VA" ;;
  esac
}

process_start_token() {
  local pid=$1
  if [[ -r "/proc/${pid}/stat" ]]; then
    python3 - "${pid}" <<'PY'
from pathlib import Path
import sys

value = Path(f"/proc/{sys.argv[1]}/stat").read_text()
print(value.rsplit(") ", 1)[1].split()[19])
PY
  else
    ps -o lstart= -p "${pid}" 2>/dev/null | awk '{$1=$1; gsub(/ /, "_"); print}'
  fi
}

record_pid() {
  local pid=$1 token
  token=$(process_start_token "${pid}" 2>/dev/null || true)
  [[ -n "${token}" ]] || return 1
  printf '%s %s\n' "${pid}" "${token}" >"${pid_file}"
}

recorded_pid() {
  [[ -f "${pid_file}" ]] && awk 'NR == 1 {print $1}' "${pid_file}" || true
}

recorded_token() {
  [[ -f "${pid_file}" ]] && awk 'NR == 1 {print $2}' "${pid_file}" || true
}

server_is_owned() {
  local pid token live_token
  pid=$(recorded_pid)
  token=$(recorded_token)
  [[ "${pid}" =~ ^[0-9]+$ && -n "${token}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  live_token=$(process_start_token "${pid}" 2>/dev/null || true)
  [[ -n "${live_token}" && "${live_token}" == "${token}" ]]
}

port_in_use() {
  python3 - "${port}" <<'PY'
import socket
import sys

with socket.socket() as client:
    client.settimeout(0.5)
    raise SystemExit(0 if client.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

select_h100() {
  require_command nvidia-smi
  local rows index name memory selected=
  rows=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits 2>/dev/null) \
    || fail "nvidia-smi could not query a GPU. Run this on the H100 server."

  while IFS=, read -r index name memory; do
    name=${name# }
    memory=${memory//[[:space:]]/}
    if [[ "${name}" =~ H100 && "${memory}" =~ ^[0-9]+$ && ${memory} -ge 70000 ]]; then
      selected=${index//[[:space:]]/}
      break
    fi
  done <<<"${rows}"

  [[ -n "${selected}" ]] || fail "A full H100 80 GB GPU is required; no suitable GPU was found."
  # Slurm owns CUDA_VISIBLE_DEVICES. This check intentionally does not alter it.
}

require_allocation_if_slurm() {
  if command -v sbatch >/dev/null 2>&1 && [[ -z "${SLURM_JOB_ID:-}" ]]; then
    fail "A Slurm cluster was detected. Start H3 through its Slurm launcher; do not run it on the login node."
  fi
}

warn_if_low_host_memory() {
  [[ -r /proc/meminfo ]] || return 0
  local kib gib
  kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
  [[ "${kib}" =~ ^[0-9]+$ ]] || return 0
  gib=$((kib / 1024 / 1024))
  if ((gib < 128)); then
    echo "Warning: ${gib} GiB host RAM detected. 128 GiB is the practical minimum; 256 GiB is recommended for offload." >&2
  fi
}

ensure_model() {
  local variant=$1 component
  component=$(component_dir "${variant}")
  [[ -f "${model_dir}/model_index.json" && -d "${model_dir}/${component}" ]] || {
    fail "${variant} weights are missing from ${model_dir}. Run: ./h3.sh download ${variant}"
  }
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  require_command curl
  echo "Installing uv in your user account ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || fail "uv installation failed. Install uv, then run ./h3.sh setup again."
}

setup() {
  [[ "$(uname -s)" == "Linux" ]] || fail "This deployment script is for a Linux NVIDIA server."
  for command_name in curl git python3 ffmpeg; do
    require_command "${command_name}"
  done
  ensure_uv
  mkdir -p "${model_dir}" "${repo_dir}/outputs" "${runtime_dir}" "${log_dir}"
  (
    cd "${repo_dir}"
    UV_LINK_MODE=copy uv sync --frozen --extra inference --python 3.11
  )
  echo
  echo "Environment ready: ${repo_dir}/.venv"
  echo "Accept the MiniMax H3 license on Hugging Face, then run: ./h3.sh download"
}

download() {
  local variant=${1:-fl2va}
  valid_variant "${variant}" || { usage >&2; exit 2; }
  local hf_bin="${H3_HF_BIN:-${venv_bin}/hf}"
  [[ -x "${hf_bin}" ]] || fail "Environment not found. Run ./h3.sh setup first."

  mkdir -p "${model_dir}"
  local available_kib
  available_kib=$(df -Pk "${model_dir}" | awk 'NR == 2 {print $4}')
  if [[ "${available_kib}" =~ ^[0-9]+$ && ${available_kib} -lt 188743680 ]]; then
    echo "Warning: less than 180 GiB free at ${model_dir}; the large H3 checkpoint may not fit." >&2
  fi

  if ! "${hf_bin}" auth whoami >/dev/null 2>&1; then
    echo "Log in to Hugging Face. You must first accept the MiniMax H3 license on its model page."
    "${hf_bin}" auth login
  fi

  local component
  component=$(component_dir "${variant}")
  local model_revision=${H3_MODEL_REVISION:-42ed227ee7df40d41602854ae760620d6eb651fe}
  "${hf_bin}" download MiniMaxAI/MiniMax-H3 \
    --revision "${model_revision}" \
    --include "model_index.json" "${component}/*" \
    --local-dir "${model_dir}"
  echo "Downloaded ${variant} to ${model_dir}"
}

stop() {
  local pid
  pid=$(recorded_pid)
  if server_is_owned; then
    echo "Stopping SGLang (PID ${pid}) ..."
    kill "${pid}"
    for _ in {1..30}; do
      server_is_owned || break
      sleep 1
    done
    server_is_owned && kill -KILL "${pid}"
  elif [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "Ignoring a stale PID record for unrelated process ${pid}." >&2
  else
    echo "SGLang is not running."
  fi
  rm -f -- "${pid_file}" "${variant_file}"
}

start() {
  local variant=${1:-fl2va}
  valid_variant "${variant}" || { usage >&2; exit 2; }
  [[ -x "${venv_bin}/sglang" ]] || fail "Environment not found. Run ./h3.sh setup first."
  require_command curl
  require_command python3
  require_allocation_if_slurm
  select_h100
  warn_if_low_host_memory
  ensure_model "${variant}"

  mkdir -p "${runtime_dir}" "${log_dir}"
  if server_is_owned; then
    fail "SGLang is already running. Use ./h3.sh status, ./h3.sh stop, or ./h3.sh restart."
  fi
  rm -f -- "${pid_file}" "${variant_file}"
  if port_in_use; then
    fail "Port ${port} is already in use. Stop that service or set H3_PORT before starting."
  fi

  export H3_SGLANG_BIN="${venv_bin}/sglang"
  export H3_MODEL_PATH="${model_dir}"
  export H3_PROFILE=h100x1
  export H3_INFERENCE_HOST=127.0.0.1
  export H3_INFERENCE_PORT="${port}"

  echo "Starting ${variant} on CUDA device ${CUDA_VISIBLE_DEVICES} ..."
  echo "Using the single-H100 ${H3_H100_MODE:-speed} profile with BF16/FP32 offload; ComfyUI is not used."
  nohup "${repo_dir}/deploy/start_sglang.sh" "${variant}" >"${log_file}" 2>&1 &
  local pid=$!
  if ! record_pid "${pid}"; then
    wait "${pid}" 2>/dev/null || true
    fail "SGLang exited before it could be recorded. See ${log_file}"
  fi

  local timeout=${H3_STARTUP_TIMEOUT_SECONDS:-1800}
  [[ "${timeout}" =~ ^[0-9]+$ ]] || fail "H3_STARTUP_TIMEOUT_SECONDS must be a non-negative integer."
  local deadline=$((SECONDS + timeout))
  echo "Waiting for the model to load. This can take several minutes ..."
  while ! curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null 2>&1; do
    if ! server_is_owned; then
      rm -f -- "${pid_file}"
      echo "SGLang stopped during startup. Recent log output:" >&2
      tail -n 30 "${log_file}" >&2 || true
      exit 1
    fi
    if ((SECONDS >= deadline)); then
      stop
      fail "SGLang startup timed out. See ${log_file}"
    fi
    sleep 5
  done

  printf '%s\n' "${variant}" >"${variant_file}"
  echo "Ready: http://127.0.0.1:${port}/v1/videos"
  echo "Generate a video with: ./h3.sh generate \"A red panda making tea in the rain\""
}

status() {
  local variant=unknown
  [[ -f "${variant_file}" ]] && variant=$(<"${variant_file}")
  if server_is_owned; then
    echo "SGLang: running (PID $(recorded_pid), ${variant}, port ${port})"
  else
    echo "SGLang: stopped"
  fi
  if command -v curl >/dev/null 2>&1 && curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "Health: ready"
  else
    echo "Health: unavailable"
  fi
}

logs() {
  [[ -f "${log_file}" ]] || fail "No SGLang log exists yet. Start the server first."
  exec tail -n "${H3_LOG_LINES:-80}" -f "${log_file}"
}

command=${1:-help}
shift || true

case "${command}" in
  setup) (($# == 0)) || { usage >&2; exit 2; }; setup ;;
  download) (($# <= 1)) || { usage >&2; exit 2; }; download "${1:-fl2va}" ;;
  start) (($# <= 1)) || { usage >&2; exit 2; }; start "${1:-fl2va}" ;;
  generate) exec "${repo_dir}/scripts/generate_sglang.sh" "$@" ;;
  status) (($# == 0)) || { usage >&2; exit 2; }; status ;;
  logs) (($# == 0)) || { usage >&2; exit 2; }; logs ;;
  stop) (($# == 0)) || { usage >&2; exit 2; }; stop ;;
  restart)
    (($# <= 1)) || { usage >&2; exit 2; }
    stop
    start "${1:-fl2va}"
    ;;
  help|-h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
