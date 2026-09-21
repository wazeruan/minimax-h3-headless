#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${repo_dir}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
# shellcheck disable=SC1091
variant=${1:-fl2va}
case "${variant}" in
  fl2va|ref2va|both) ;;
  *) echo "Usage: $0 [fl2va|ref2va|both]" >&2; exit 2 ;;
esac

if [[ -x "${repo_dir}/scripts/submit_slurm_download.sh" ]]; then
  exec "${repo_dir}/scripts/submit_slurm_download.sh" "${variant}"
fi

# Compatibility fallback for a deliberately partial checkout. A complete
# checkout always contains the Slurm submitter above, so this path is not used
# for the supported Nibi workflow.
source "${repo_dir}/scripts/require_active_env.sh"
h3_require_active_env
if [[ -n "${H3_MODULES_FILE:-}" && -f "${H3_MODULES_FILE}" ]]; then
  if ! type module >/dev/null 2>&1 && [[ -r /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]]; then
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
  fi
  source "${H3_MODULES_FILE}"
fi
hf_bin=${H3_HF_BIN:-"${repo_dir}/.venv/bin/hf"}
[[ -x "${hf_bin}" ]] || { echo "Environment not found. Run ./setup.sh first." >&2; exit 1; }
if ! "${hf_bin}" auth whoami >/dev/null 2>&1; then
  echo "Log in to Hugging Face to download MiniMax H3."
  "${hf_bin}" auth login
fi
export PATH="$(dirname -- "${hf_bin}"):${PATH}"
export H3_MODEL_DIR=${H3_MODEL_DIR:-"${repo_dir}/models/MiniMax-H3"}
exec "${repo_dir}/scripts/download_model.sh" "${variant}"
