#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
usage() {
  cat <<'EOF'
Usage: scripts/submit_slurm_download.sh [fl2va|ref2va|both]

Submits a CPU Slurm job that downloads the pinned MiniMax-H3 checkpoint onto
remote shared storage. Run this from the remote login node; it never downloads
weights on the login node or the local workstation.
EOF
}

case "${1:-}" in -h|--help) usage; exit 0 ;; esac
variant=${1:-fl2va}
case "${variant}" in fl2va|ref2va|both) ;; *) echo "Usage: $0 [fl2va|ref2va|both]" >&2; exit 2 ;; esac

if [[ -f "${repo_dir}/.env" ]]; then
  set -a
  source "${repo_dir}/.env"
  set +a
fi
command -v sbatch >/dev/null 2>&1 || {
  echo "sbatch is required. Connect to the remote Slurm login node through SSH first." >&2
  exit 1
}
hf_bin=${H3_HF_BIN:-"${repo_dir}/.venv/bin/hf"}
[[ -x "${hf_bin}" ]] || { echo "Environment not found. Run ./setup.sh on the remote host first." >&2; exit 1; }
model_dir=${H3_MODEL_DIR:-"${repo_dir}/models/MiniMax-H3"}
available_kib=$(df -Pk "${model_dir}" | awk 'NR == 2 {print $4}')
[[ "${available_kib}" =~ ^[0-9]+$ && ${available_kib} -ge 188743680 ]] || {
  echo "At least 180 GiB free disk is required at ${model_dir}." >&2
  exit 1
}

stamp=$(date -u +%Y%m%dT%H%M%SZ)
runs_root=${H3_DOWNLOAD_RUNS_DIR:-"${repo_dir}/.run/downloads"}
mkdir -p -- "${runs_root}" "${repo_dir}/logs"
for suffix in '' $(seq -w 1 999); do
  run_dir="${runs_root}/${stamp}${suffix:+-${suffix}}"
  if mkdir "${run_dir}" 2>/dev/null; then break; fi
  run_dir=
done
[[ -n "${run_dir:-}" ]] || { echo "Could not reserve a download receipt directory." >&2; exit 1; }

receipt="${run_dir}/receipt.env"
printf 'state=SUBMITTING\nvariant=%s\nsubmitted_at=%s\nmodel_dir=%s\n' "${variant}" "${stamp}" "${model_dir}" >"${receipt}"
log_base="${repo_dir}/logs/h3-download-${stamp}-%j"
sbatch_args=(--parsable --export="ALL,H3_REPO_DIR=${repo_dir},H3_DOWNLOAD_RECEIPT=${receipt}")
[[ -z "${H3_SLURM_ACCOUNT:-}" ]] || sbatch_args+=(--account="${H3_SLURM_ACCOUNT}")
[[ -z "${H3_SLURM_PARTITION:-}" ]] || sbatch_args+=(--partition="${H3_SLURM_PARTITION}")
[[ -z "${H3_SLURM_DOWNLOAD_TIME:-}" ]] || sbatch_args+=(--time="${H3_SLURM_DOWNLOAD_TIME}")
if ! job_id=$(sbatch "${sbatch_args[@]}" --output="${log_base}.out" --error="${log_base}.err" \
  "${repo_dir}/deploy/slurm/h3-download.sbatch" "${variant}"); then
  printf 'state=FAILED\nreason=sbatch_submission_failed\n' >>"${receipt}"
  exit 1
fi
[[ "${job_id}" =~ ^[0-9]+([_;][0-9]+)?$ ]] || {
  printf 'state=UNKNOWN_UNVERIFIED\nsubmission_response=%s\n' "${job_id}" >>"${receipt}"
  echo "Ambiguous submission response. Do not resubmit blindly; receipt: ${receipt}" >&2
  exit 1
}
printf 'state=SUBMITTED\njob_id=%s\nstdout=%s.out\nstderr=%s.err\n' "${job_id}" "${log_base}" "${log_base}" >>"${receipt}"
if command -v squeue >/dev/null 2>&1 && ! squeue --noheader --jobs="${job_id}" >/dev/null 2>&1; then
  printf 'state=UNKNOWN_UNVERIFIED\nreason=scheduler_record_not_observed\n' >>"${receipt}"
  echo "The scheduler did not confirm ${job_id}. Do not resubmit blindly; receipt: ${receipt}" >&2
  exit 1
fi
echo "Download submitted: ${job_id}"
echo "Receipt: ${receipt}"
echo "Logs: ${log_base}.out / ${log_base}.err"
