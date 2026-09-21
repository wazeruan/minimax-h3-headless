# Operations and troubleshooting

## Recommended four-H100 process layout

For the SSH-first deployment described in the current README, use `./h3.sh`.
It starts one SGLang process directly on `127.0.0.1:30010`; there is no
ComfyUI and no FastAPI gateway in that path. The four-H100 profile uses the
verified TP2 + Ulysses2 topology, and `./h3.sh generate` talks to SGLang's
native `/v1/videos` API.

```bash
./h3.sh status
./h3.sh logs
./h3.sh stop
```

Use `H3_H100_MODE=memory ./h3.sh restart` as the first capacity fallback when
the one-H100 process runs out of memory. This returns to 20 resident DiT blocks.
Add `H3_DIT_RESIDENT_LAYERS=4` only if more headroom is needed.

## Legacy gateway process layout

FL2VA serves `t2va` and `fl2va` on port 30010. Ref2VA serves `ref2va` on port
30011. The gateway listens on 127.0.0.1:8080 and is the only endpoint clients
should call when using the legacy gateway workflow.

## Hardware profiles

| Profile | Officially documented role | Notes |
| --- | --- | --- |
| `auto` | Detect visible NVIDIA GPU model/count | Falls back to `h100x1` when detection is unavailable |
| `h100x1` | Capacity fallback | Lossless layerwise CPU offload; not a measured speed topology; use at least 256 GB host RAM |
| `h100x4` | SGLang speed default on 4×H100 80 GB | TP2 + Ulysses2 |
| `h100x4_memory` | Lower resident memory on 4×H100 | TP4; modestly slower |
| `h100x4_fsdp` | Capacity fallback on 4×H100 | About 57 GB peak/GPU in the published benchmark |
| `h200x4` | SGLang resident path | Pure Ulysses4 |
| `rtx5090x2` | Lossless SGLang offload path | Requires roughly 384 GB host RAM |

The vLLM-Omni Docker launcher includes its documented B300, two-card DLO, and
single-GPU offload profiles. Its `auto` mode selects `single_offload` for one
H100. SGLang remains the recommended four-H100 performance path.

## Useful checks

```bash
scripts/healthcheck.sh
nvidia-smi
curl --fail http://127.0.0.1:30010/health
curl --fail http://127.0.0.1:30011/health
```

The gateway reports `degraded` if neither backend is reachable. It reports each
partition separately so running only FL2VA is a valid partial deployment.

## Common failures

- Out of memory: confirm the selected profile and exact GPU memory. Use the TP4
  or FSDP H100 profile before reducing request quality.
- First launch appears idle: downloading gated weights can take a long time.
  Authenticate with `hf auth login` and pre-download on shared storage.
- Ref2VA request reaches FL2VA: ensure port 30011 runs `--model-variant ref2va`.
- Client gets 401: use the same `H3_GATEWAY_API_KEY` as the server `.env`.
- SSH disconnect kills processes: use systemd, Slurm, or tmux; do not rely on a
  foreground shell for a long-running server.

## Security

Keep all services on loopback whenever possible. The gateway fails closed when
no API key is configured. Use SSH tunneling, a private overlay network, or a
proper TLS reverse proxy; never place SGLang/vLLM directly on the public web.
