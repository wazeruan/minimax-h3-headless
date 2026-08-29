#!/usr/bin/env bash
set -euo pipefail

# Submit the official SGLang H3 T2VA payload, wait for completion, and download
# the MP4.  This deliberately talks to SGLang directly: there is no ComfyUI or
# project gateway in the default headless workflow.

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
sglang_url=http://127.0.0.1:30010
poll_interval=2
generation_timeout=7200
duration=5
aspect_ratio=16:9
seed=42
steps=50
flow_shift=12.0
audio_flow_shift=3.0
model_id=MiniMaxAI/MiniMax-H3
prompt=
output=

usage() {
  cat <<'EOF'
Usage: scripts/generate_sglang.sh [OPTIONS] [PROMPT] [OUTPUT.mp4]

Submits a text-to-video-and-audio request to the local SGLang server.
If PROMPT is omitted, the script asks for it interactively.

Options:
  --prompt TEXT                   Video prompt (or use the first positional argument)
  --output FILE                   Output MP4 (or use the second positional argument)
  --duration SECONDS              4 through 15 (default 5)
  --aspect-ratio RATIO            21:9, 16:9, 4:3, 1:1, 3:4, or 9:16 (default 16:9)
  --seed INTEGER                  Sampling seed (default 42)
  --steps INTEGER                 Inference steps, 1 through 100 (default 50)
  --flow-shift NUMBER             Video flow shift (default 12.0)
  --audio-flow-shift NUMBER       Audio flow shift (default 3.0)
  --model ID                      SGLang model identifier (default MiniMaxAI/MiniMax-H3)
  --url URL                       Direct SGLang endpoint (default http://127.0.0.1:30010)
  --poll-interval SECONDS         Status-polling interval (default 2)
  --timeout SECONDS               Generation timeout (default 7200)
  -h, --help                      Show this help
EOF
}

require_value() {
  local option=$1
  (( $# >= 2 )) || { echo "${option} requires a value." >&2; exit 2; }
}

while (( $# )); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --prompt) require_value "$@"; prompt=$2; shift 2 ;;
    --output) require_value "$@"; output=$2; shift 2 ;;
    --duration) require_value "$@"; duration=$2; shift 2 ;;
    --aspect-ratio) require_value "$@"; aspect_ratio=$2; shift 2 ;;
    --seed) require_value "$@"; seed=$2; shift 2 ;;
    --steps) require_value "$@"; steps=$2; shift 2 ;;
    --flow-shift) require_value "$@"; flow_shift=$2; shift 2 ;;
    --audio-flow-shift) require_value "$@"; audio_flow_shift=$2; shift 2 ;;
    --model) require_value "$@"; model_id=$2; shift 2 ;;
    --url) require_value "$@"; sglang_url=$2; shift 2 ;;
    --poll-interval) require_value "$@"; poll_interval=$2; shift 2 ;;
    --timeout) require_value "$@"; generation_timeout=$2; shift 2 ;;
    --) shift; while (( $# )); do
      if [[ -z "${prompt}" ]]; then prompt=$1
      elif [[ -z "${output}" ]]; then output=$1
      else usage >&2; exit 2
      fi
      shift
    done ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) if [[ -z "${prompt}" ]]; then prompt=$1
      elif [[ -z "${output}" ]]; then output=$1
      else usage >&2; exit 2
      fi
      shift ;;
  esac
done

if [[ -z "${prompt}" ]]; then
  read -r -p "Video prompt: " prompt
fi
[[ -n "${prompt//[[:space:]]/}" ]] || { echo "Prompt cannot be empty." >&2; exit 2; }

timestamp=$(date +%Y%m%d-%H%M%S)
output=${output:-"${repo_dir}/outputs/h3-${timestamp}.mp4"}

command -v curl >/dev/null 2>&1 || { echo "curl is required." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required." >&2; exit 1; }
[[ "${generation_timeout}" =~ ^[0-9]+$ ]] || {
  echo "--timeout must be a non-negative integer." >&2
  exit 2
}
[[ "${poll_interval}" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "--poll-interval must be a non-negative number." >&2
  exit 2
}

work_dir=$(mktemp -d)
download_file=
cleanup() {
  rm -rf -- "${work_dir}"
  [[ -z "${download_file}" ]] || rm -f -- "${download_file}"
}
trap cleanup EXIT

request_file="${work_dir}/request.json"
response_file="${work_dir}/response.json"

python3 - \
  "${prompt}" "${duration}" "${aspect_ratio}" "${seed}" "${steps}" \
  "${flow_shift}" "${audio_flow_shift}" "${model_id}" >"${request_file}" <<'PY'
import json
import math
import sys

prompt, duration, aspect_ratio, seed, steps, flow_shift, audio_flow_shift, model_id = sys.argv[1:]

try:
    duration_value = float(duration)
    seed_value = int(seed)
    steps_value = int(steps)
    flow_shift_value = float(flow_shift)
    audio_flow_shift_value = float(audio_flow_shift)
except ValueError as exc:
    raise SystemExit(f"Invalid numeric generation setting: {exc}") from exc

if not math.isfinite(duration_value) or not 4 <= duration_value <= 15:
    raise SystemExit("--duration must be between 4 and 15.")
if aspect_ratio not in {"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}:
    raise SystemExit("--aspect-ratio must be one of 21:9, 16:9, 4:3, 1:1, 3:4, or 9:16.")
if seed_value < 0:
    raise SystemExit("--seed must be non-negative.")
if not 1 <= steps_value <= 100:
    raise SystemExit("--steps must be between 1 and 100.")
if not math.isfinite(flow_shift_value) or not math.isfinite(audio_flow_shift_value):
    raise SystemExit("Flow-shift values must be finite numbers.")

# `seconds` and `target.duration_seconds` are both included to match the
# published SGLang request examples for MiniMax H3.
json.dump(
    {
        "model": model_id,
        "prompt": prompt,
        "seconds": duration_value,
        "task": "t2va",
        "conditions": [],
        "target": {
            "short_edge": 768,
            "aspect_ratio": aspect_ratio,
            "duration_seconds": duration_value,
        },
        "num_outputs_per_prompt": 1,
        "num_inference_steps": steps_value,
        "flow_shift": flow_shift_value,
        "audio_flow_shift": audio_flow_shift_value,
        "seed": seed_value,
    },
    sys.stdout,
)
PY

echo "Submitting generation to ${sglang_url} ..."
curl --fail-with-body --silent --show-error \
  --request POST "${sglang_url%/}/v1/videos" \
  --header 'Content-Type: application/json' \
  --data-binary "@${request_file}" \
  --output "${response_file}"

job_id=$(python3 - "${response_file}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
job_id = payload.get("id")
if not isinstance(job_id, str) or not job_id:
    raise SystemExit("SGLang response did not include a job id")
print(job_id)
PY
)
job_path=$(python3 - "${job_id}" <<'PY'
from urllib.parse import quote
import sys

print(quote(sys.argv[1], safe=""))
PY
)

echo "Job: ${job_id}"
started_at=${SECONDS}
last_status=

while true; do
  curl --fail-with-body --silent --show-error \
    "${sglang_url%/}/v1/videos/${job_path}" \
    --output "${response_file}"

  status=$(python3 - "${response_file}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload.get("status", "unknown"))
PY
)

  if [[ "${status}" != "${last_status}" ]]; then
    echo "Status: ${status}"
    last_status=${status}
  fi

  case "${status}" in
    completed|succeeded) break ;;
    failed|cancelled)
      python3 - "${response_file}" <<'PY' >&2
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(f"Generation {payload.get('status')}: {payload.get('error') or 'no error details'}")
PY
      exit 1
      ;;
  esac

  if ((SECONDS - started_at >= generation_timeout)); then
    echo "Generation timed out after ${generation_timeout} seconds." >&2
    exit 1
  fi
  sleep "${poll_interval}"
done

mkdir -p -- "$(dirname -- "${output}")"
download_file=$(mktemp "${output}.partial.XXXXXX")
curl --fail-with-body --silent --show-error \
  --location \
  "${sglang_url%/}/v1/videos/${job_path}/content" \
  --output "${download_file}"

mv -f -- "${download_file}" "${output}"
download_file=

echo "Saved video: ${output}"
