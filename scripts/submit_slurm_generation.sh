#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
  cat <<'EOF'
Usage: scripts/submit_slurm_generation.sh [GENERATION OPTIONS] [PROMPT] [OUTPUT.mp4]

Submits a four-H100 Slurm job that starts MiniMax H3, generates one MP4, and
stops the server. Use the same generation options as `./h3.sh generate`, such
as `--duration 10 --aspect-ratio 9:16 --seed 123`.

Optional submission settings:
  H3_SLURM_ACCOUNT       Slurm account (loaded from .env when present)
  H3_SLURM_PARTITION     Slurm partition
  H3_SLURM_GPU_OPTION    Exact GPU option (default --gpus-per-node=h100:4)
  H3_SLURM_TIME          Override the job time limit
EOF
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac
(( $# >= 1 )) || { usage >&2; exit 2; }

if [[ -f "${repo_dir}/.env" ]]; then
  # Load site defaults such as H3_SLURM_ACCOUNT. The compute job loads the same
  # file itself for model, environment, and module paths.
  # shellcheck disable=SC1091
  source "${repo_dir}/.env"
fi

command -v sbatch >/dev/null 2>&1 || {
  echo "sbatch is required. Run this on the Slurm login node." >&2
  exit 1
}

batch_script="${repo_dir}/deploy/slurm/h3-generate.sbatch"
[[ -f "${batch_script}" ]] || { echo "Missing ${batch_script}" >&2; exit 1; }

gpu_option=${H3_SLURM_GPU_OPTION:---gpus-per-node=h100:4}
sbatch_args=(--export=ALL)
[[ -z "${gpu_option}" ]] || sbatch_args+=("${gpu_option}")
[[ -z "${H3_SLURM_ACCOUNT:-}" ]] || sbatch_args+=(--account="${H3_SLURM_ACCOUNT}")
[[ -z "${H3_SLURM_PARTITION:-}" ]] || sbatch_args+=(--partition="${H3_SLURM_PARTITION}")
[[ -z "${H3_SLURM_TIME:-}" ]] || sbatch_args+=(--time="${H3_SLURM_TIME}")

export H3_REPO_DIR="${repo_dir}"

exec sbatch "${sbatch_args[@]}" "${batch_script}" "$@"
