# MiniMax H3 — H100, headless deployment

This repository now has one recommended path for an SSH-only Linux server:
**SGLang directly on `127.0.0.1`, with no ComfyUI and no extra gateway.**

`MiniMax H3` produces a 768p video and synchronized stereo audio in one request.
The public release splits its weights into two partitions:

- `FL2VA` — text-to-video-and-audio (`t2va`) and first/last-frame generation.
- `Ref2VA` — image, audio, and video reference generation.

For fastest prompt-only work, use `FL2VA`. For the strongest subject, scene,
or style consistency, use `Ref2VA` with reference media. They are separate
partitions, so a four-H100 deployment serves one of them at a time.

## Important limitation

This repository is configured for **one full H100 80 GB** using lossless
BF16/FP32 CPU/layerwise offload plus online INT8 DiT quantization. Quantization
reduces memory use but changes linear-layer numerics; set `H3_QUANTIZATION=off`
when exact BF16/FP32 behavior is more important than capacity.

The locally available model is H3-Base at a 768-pixel short edge. The hosted
H3-Context-IR prompt-preprocessing stage and 2K regeneration are not included
in the open release.

## Server requirements

- Linux with one NVIDIA H100 80 GB GPU visible to `nvidia-smi`
- At least 256 GiB host RAM for CPU offload
- At least 180 GiB free disk for one checkpoint partition (more for both)
- A CUDA driver compatible with the SGLang version locked in this repository
- CUDA Toolkit compiler (`nvcc`), used once to build SGLang's H3 JIT kernels
- A current SGLang Diffusion release, resolved in `uv.lock`
- `curl`, `git`, `python3`, and `ffmpeg` already available on the server

No system packages are installed by these scripts. `uv` is installed into your
user account only when it is missing.

## Remote download and generation

Connect to the Slurm login node over SSH. After accepting the Hugging Face H3
license and logging in there once, submit the download; the checkpoint is
downloaded only on a scheduled remote CPU node, never on this machine or the
login node:

```bash
ssh nibi.alliancecan.ca
cd /scratch/USER/minimax-h3-headless
./setup.sh
source .venv/bin/activate
hf auth login
./download_models.sh fl2va
```

The download job pins MiniMax-H3 revision
`42ed227ee7df40d41602854ae760620d6eb651fe`. Use `ref2va` instead when reference
consistency is the priority, or `both` when storage permits both partitions.

The generated MP4 is written to `outputs/` and the server log to
`logs/sglang.log`.

## Slurm: submit one complete generation job

On a Slurm login node, the submission helper requests one H100, starts SGLang
inside the allocation, creates the video, and stops SGLang before the job exits:

```bash
./scripts/submit_slurm_generation.sh \
  --prompt "A red panda makes tea while rain taps on the cabin window." \
  --output outputs/red-panda.mp4
```

It reads `H3_SLURM_ACCOUNT` from `.env` when present and otherwise uses your
site's default account. The default GPU request is
`--gpus-per-node=h100:1`. For clusters that use GRES instead:

```bash
H3_SLURM_GPU_OPTION=--gres=gpu:h100:1 \
  ./scripts/submit_slurm_generation.sh --prompt "A moonlit mountain lake."
```

See [the Slurm guide](docs/SLURM.md) for direct `sbatch`, partition, duration,
seed, logs, and output details.

## Day-to-day commands

```bash
./h3.sh status
./h3.sh logs
./h3.sh stop
./h3.sh restart
```

The server binds only to `127.0.0.1:30010`, so it is not exposed to the public
network. To call SGLang's native API from your laptop, use an SSH tunnel:

```bash
ssh -N -L 30010:127.0.0.1:30010 USER@GPU_SERVER
```

## Optional settings

The defaults create a 5-second, 16:9, 768p clip with 50 inference steps.

```bash
./h3.sh generate \
  --prompt "A slow vertical tracking shot through a neon night market." \
  --duration 10 --aspect-ratio 9:16 --seed 123 --steps 50 \
  --output outputs/night-market.mp4
```

Run `./h3.sh generate --help` to see every generation flag, including video and
audio flow shift, model identifier, direct server URL, polling interval, and
timeout. The generation command does not read those settings from environment
variables.

The default `speed` mode is tuned to use the 80-GB H100 more aggressively. If
startup or generation runs out of memory, switch to the one-command `memory`
fallback; it uses 20 resident DiT blocks:

```bash
H3_H100_MODE=memory ./h3.sh restart
```

If the request still runs out of memory, lower the resident block count:

```bash
H3_H100_MODE=memory H3_DIT_RESIDENT_LAYERS=4 ./h3.sh restart
```

`H3_DIT_RESIDENT_LAYERS` must be a non-negative integer. The speed-mode default
is `32`; the memory-mode default is `20`. More resident blocks reduce PCIe
weight transfers, but leave less VRAM for activations.

To put model weights on a larger mounted volume, set the same environment
variable for setup, download, and server start:

```bash
export H3_MODEL_DIR=/data/models/MiniMax-H3
./h3.sh setup
./h3.sh download
./h3.sh start
```

To install and switch to the reference partition, stop `FL2VA` first:

```bash
./h3.sh stop
./h3.sh download ref2va
./h3.sh start ref2va
```

`./h3.sh generate` intentionally handles the simple text-only `FL2VA` request.
For first/last-frame or Ref2VA inputs, call the native SGLang endpoint with the
official request schema. The local server accepts `file://` input URIs only for
files visible on that same server.

## What the launcher does

`./h3.sh start` launches this direct SGLang configuration:

```bash
sglang serve \
  --model-path /path/to/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus 1 \
  --tp-size 1 \
  --ulysses-degree 1 \
  --performance-mode memory \
  --layerwise-offload-components dit,text_encoder,vae \
  --dit-offload-prefetch-size 1 \
  --dit-layerwise-resident-layers 32 \
  --enable-torch-compile false \
  --host 127.0.0.1 \
  --port 30010
```

SGLang still needs `performance-mode memory` because the complete pipeline is
larger than 80 GB. Within that policy, retaining more DiT blocks avoids repeated
CPU-to-GPU transfers. The default Hopper attention backend is
left unchanged, and `torch.compile` stays disabled because its current H3 path
changes numerical output. The direct client sends the documented `/v1/videos`
payload, polls its status, then downloads the completed MP4 atomically.

## Troubleshooting

- **Download denied:** accept the MiniMax H3 license in Hugging Face first, then
  rerun `./h3.sh download`.
- **Server exits during startup:** run `./h3.sh logs`. Most often this means
  insufficient host RAM, GPU memory, free disk, or a CUDA/driver mismatch.
- **`Could not detect ROCm GPU architecture`:** update this repository and
  restart. The launcher explicitly selects CUDA for SGLang's JIT kernels on an
  NVIDIA node; make sure the CUDA module supplies `nvcc`.
- **GPU OOM:** keep `H3_QUANTIZATION=kitchen_int8`, restart with
  `H3_H100_MODE=memory`, and if needed add `H3_DIT_RESIDENT_LAYERS=4`. The
  quantized path changes linear-layer numerics, so validate output quality.
- **Slow first request:** expected. The model is loading and CPU-offloaded
  blocks traverse PCIe during denoising.
- **Need 2K output or official Context-IR quality:** use MiniMax's hosted API;
  those stages are not open-sourced.

The older FastAPI gateway, Slurm, and vLLM-Omni files remain in the repository
for existing users, but they are not part of the recommended single-H100 path.

This repository's MIT license covers only its deployment code. MiniMax H3
weights remain subject to the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3).
