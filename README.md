# MiniMax H3 — single-H100, headless deployment

This repository now has one recommended path for an SSH-only Linux server:
**SGLang directly on `127.0.0.1`, with no ComfyUI and no extra gateway.**

`MiniMax H3` produces a 768p video and synchronized stereo audio in one request.
The public release splits its weights into two partitions:

- `FL2VA` — text-to-video-and-audio (`t2va`) and first/last-frame generation.
- `Ref2VA` — image, audio, and video reference generation.

On a single H100 80 GB, run one partition at a time. The default is `FL2VA`,
which is enough for prompt-only generation.

## Important limitation

The full H3 weights are larger than one H100. This project therefore uses
SGLang's BF16/FP32 **layerwise CPU offload** mode. The default speed profile
keeps 32 DiT blocks resident on the H100 while the remaining model weights,
including the VAE, stream from host RAM. It does not quantize the model and does not
depend on ComfyUI, but it is still slower than a resident multi-GPU deployment.

MiniMax and SGLang publish measured H100 recipes for four H100 80 GB GPUs; the
single-H100 offload topology is a practical capacity configuration, not a
published H100 benchmark. Budget several minutes per first 5-second clip and
validate the speed on your own server.

The locally available model is H3-Base at a 768-pixel short edge. The hosted
H3-Context-IR prompt-preprocessing stage and 2K regeneration are not included
in the open release.

## Server requirements

- Linux with one NVIDIA H100 80 GB visible to `nvidia-smi`
- At least 128 GiB host RAM; 256 GiB is recommended for CPU offload
- At least 180 GiB free disk for one checkpoint partition (more for both)
- A CUDA driver compatible with the SGLang version locked in this repository
- CUDA Toolkit compiler (`nvcc`), used once to build SGLang's H3 JIT kernels
- SGLang 0.5.17, which `./h3.sh setup` installs; this is the first locked
  release that accepts the MiniMax-H3 `--model-variant` and DiT-residency flags
- `curl`, `git`, `python3`, and `ffmpeg` already available on the server

No system packages are installed by these scripts. `uv` is installed into your
user account only when it is missing.

## Four commands

Run these on the H100 server after cloning the repository:

```bash
./h3.sh setup
./h3.sh download
./h3.sh start
./h3.sh generate --prompt "A red panda makes tea in a quiet cabin during gentle rain, cinematic close shot, synchronized kettle and rain sounds."
```

Before `download`, open the [MiniMax H3 model page](https://huggingface.co/MiniMaxAI/MiniMax-H3), accept its license, and use a Hugging Face account with access. The command opens the Hugging Face login flow if needed.

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
- **GPU OOM:** restart with `H3_H100_MODE=memory`; if needed, add
  `H3_DIT_RESIDENT_LAYERS=4`. Do not enable arbitrary quantization as a first
  response.
- **Slow first request:** expected. The model is loading and CPU-offloaded
  blocks traverse PCIe during denoising.
- **Need 2K output or official Context-IR quality:** use MiniMax's hosted API;
  those stages are not open-sourced.

The older FastAPI gateway, Slurm, and vLLM-Omni files remain in the repository
for existing users, but they are not part of the recommended single-H100 path.

This repository's MIT license covers only its deployment code. MiniMax H3
weights remain subject to the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3).
