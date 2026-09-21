#!/usr/bin/env bash
#SBATCH --job-name=h3-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00

# Download a gated H3 partition on a scheduled CPU node, never on a login node.
set -Eeuo pipefail

receipt=${H3_DOWNLOAD_RECEIPT:?H3_DOWNLOAD_RECEIPT is required}
repo_dir=${H3_REPO_DIR:?H3_REPO_DIR is required}
variant=${1:-fl2va}

write_receipt() {
  local state=$1
  local temporary="${receipt}.tmp.$$"
  {
    printf 'state=%s\n' "${state}"
    printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
    printf 'host=%s\n' "$(hostname)"
    printf 'updated_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'variant=%s\n' "${variant}"
    printf 'model_revision=%s\n' "${H3_MODEL_REVISION:-42ed227ee7df40d41602854ae760620d6eb651fe}"
    printf 'model_dir=%s\n' "${H3_MODEL_DIR:-${repo_dir}/models/MiniMax-H3}"
  } >"${temporary}"
  mv -f -- "${temporary}" "${receipt}"
}

failed() { write_receipt FAILED; exit 1; }
trap failed ERR
trap 'write_receipt FAILED; exit 143' TERM INT

cd "${repo_dir}"
[[ -f .env ]] && { set -a; source .env; set +a; }
if [[ -n "${H3_MODULES_FILE:-}" && -f "${H3_MODULES_FILE}" ]]; then
  if ! type module >/dev/null 2>&1 && [[ -r /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]]; then
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
  fi
  source "${H3_MODULES_FILE}"
fi

case "${variant}" in fl2va|ref2va|both) ;; *) echo "Usage: $0 [fl2va|ref2va|both]" >&2; exit 2 ;; esac
hf_bin=${H3_HF_BIN:-"${repo_dir}/.venv/bin/hf"}
[[ -x "${hf_bin}" ]] || { echo "Hugging Face CLI not found: ${hf_bin}" >&2; exit 1; }
"${hf_bin}" auth whoami >/dev/null || {
  echo "Hugging Face authentication is required. Log in over SSH before submitting." >&2
  exit 1
}

write_receipt STARTED
export PATH="$(dirname -- "${hf_bin}"):${PATH}"
export H3_MODEL_DIR=${H3_MODEL_DIR:-"${repo_dir}/models/MiniMax-H3"}
"${repo_dir}/scripts/download_model.sh" "${variant}"
write_receipt SUCCEEDED
