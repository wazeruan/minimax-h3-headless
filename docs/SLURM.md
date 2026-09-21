# Slurm generation and serving

## One-shot generation job

The recommended batch path starts FL2VA, waits for it to become healthy,
generates one video, and stops the server automatically. Run setup on the login
node, then submit the model download to a remote CPU allocation. By default, `setup.sh`
stores the environment, model, and caches in `./minimax-h3/` beside the
repository; set `H3_PROJECT_ROOT` only when you want a different storage root.

```bash
cd /scratch/USER/minimax-h3-headless
./setup.sh
source .venv/bin/activate
./download_models.sh fl2va
```

Then submit:

```bash
./scripts/submit_slurm_generation.sh \
  --prompt "A cinematic tracking shot through a lantern-lit forest, with synchronized rain and footsteps." \
  --output outputs/forest.mp4
```

The helper loads `H3_SLURM_ACCOUNT` from `.env`, requests one H100 by default,
and prints the Slurm job ID. If `OUTPUT.mp4` is omitted, the result is written to
`outputs/h3-JOB_ID.mp4`. SGLang's per-job log is `logs/sglang-JOB_ID.log`, while
Slurm stdout and stderr are `slurm-h3-generate-JOB_ID.out` and
`slurm-h3-generate-JOB_ID.err`.

Set generation settings directly on the command line:

```bash
./scripts/submit_slurm_generation.sh \
  --prompt "A vertical shot through a night market." \
  --duration 10 --aspect-ratio 9:16 --seed 123 --steps 50
```

`./scripts/submit_slurm_generation.sh --help` shows every generation flag. The
helper passes them unchanged to the allocated job.

For a different site GPU syntax, pass the exact option through the helper:

```bash
H3_SLURM_GPU_OPTION=--gres=gpu:h100:4 \
  ./scripts/submit_slurm_generation.sh --prompt "A quiet mountain lake at dawn."
```

You can also call the batch file directly. It intentionally omits account,
partition, and GPU directives because those names differ between clusters:

```bash
cd /path/to/minimax-h3-headless
export H3_REPO_DIR="$PWD"
export H3_MODEL_PATH=/path/to/models/MiniMax-H3

sbatch \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_GPU_PARTITION \
  --gpus-per-node=h100:1 \
  --export=ALL \
  deploy/slurm/h3-generate.sbatch \
  "A red panda makes tea in a quiet cabin." \
  outputs/red-panda.mp4
```

The one-shot file binds SGLang only to loopback, chooses a per-job port, uses
the single-H100 CPU-offload profile, and always stops its child server on normal
exit, failure, cancellation, or time-limit warning. The job requests 32 CPU
cores, 256 GB host RAM, and four hours by default; command-line `sbatch` options
override these headers.

## Long-running server job

For several interactive requests in the same allocation, submit the existing
server-only job instead:

```bash
export H3_REPO_DIR="$PWD"
export H3_MODEL_PATH=/path/to/models/MiniMax-H3
export H3_PROFILE=h100x1

sbatch \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_GPU_PARTITION \
  --gpus-per-node=h100:1 \
  --export=ALL \
  deploy/slurm/h3-sglang.sbatch fl2va
```

Use your site's equivalent of `--gres=gpu:h100:1` if it does not support
`--gpus-per-node`. Keep both model partitions on one node only if the node has
enough suitable GPUs. With one H100, run FL2VA and Ref2VA as separate jobs.
The single-H100 profile uses `kitchen_int8` plus layerwise CPU offload; a 40 GB
card defaults to eight resident DiT layers. Keep the 256 GB
host-memory request and expect CPU-offload latency.

Find the compute node and inference port:

```bash
squeue -j JOB_ID -o '%.18i %.20N %.10T'
```

If the gateway runs on the login node and the site allows login-to-compute
traffic, set `H3_FL2VA_URL=http://COMPUTE_NODE:30010`. Otherwise run the gateway
inside the same allocation and create a two-hop SSH tunnel according to your
site policy. Never bind an unauthenticated inference port to a public interface.

The job log contains the selected node, variant, profile, and SGLang startup
output. A successful server reports its health endpoint before accepting video
jobs.
