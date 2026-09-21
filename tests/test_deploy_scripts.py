from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
DETECT = REPO / "deploy" / "detect_profile.sh"
START_SGLANG = REPO / "deploy" / "start_sglang.sh"
START_VLLM = REPO / "deploy" / "docker" / "start_vllm_omni.sh"


def _executable(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(0o755)
    return path


def _fake_smi(tmp_path: Path, output: str = "", exit_code: int = 0) -> Path:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    _executable(
        fake_bin / "nvidia-smi",
        "#!/usr/bin/env bash\n"
        "if [[ \"${1:-}\" != \"--query-gpu=name\" || \"${2:-}\" != \"--format=csv,noheader\" ]]; then\n"
        "  exit 64\n"
        "fi\n"
        f"printf '%s' {shlex.quote(output)}\n"
        f"exit {exit_code}\n",
    )
    return fake_bin


def _fake_nvcc(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "cuda-bin"
    fake_bin.mkdir(exist_ok=True)
    _executable(fake_bin / "nvcc", "#!/usr/bin/env bash\nexit 0\n")
    return fake_bin


def _run_detect(
    tmp_path: Path,
    names: list[str] | None,
    *,
    backend: str = "sglang",
    cuda_visible_devices: str | None = None,
    fallback: str | None = None,
    smi_exit_code: int = 0,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("H3_AUTO_FALLBACK_PROFILE", None)
    if names is not None:
        fake_bin = _fake_smi(tmp_path, "".join(f"{name}\n" for name in names), smi_exit_code)
        env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    else:
        # Isolate PATH so this remains a true no-nvidia-smi test on GPU CI.
        empty_bin = tmp_path / "empty-bin"
        empty_bin.mkdir()
        (empty_bin / "bash").symlink_to(shutil.which("bash") or "/bin/bash")
        env["PATH"] = str(empty_bin)
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if fallback is not None:
        env["H3_AUTO_FALLBACK_PROFILE"] = fallback
    return subprocess.run(
        [str(DETECT), backend], env=env, text=True, capture_output=True, check=False
    )


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["NVIDIA H100 80GB HBM3"], "h100x1"),
        (["NVIDIA H100 80GB HBM3"] * 2, "h100x1"),
        (["NVIDIA H100 40GB HBM3"] * 4, "h100x1"),
        (["NVIDIA H200"] * 4, "h100x1"),
        (["NVIDIA B300"] * 4, "h100x1"),
        (["NVIDIA GeForce RTX 5090"] * 2, "h100x1"),
        (["NVIDIA GeForce RTX 4090"] * 2, "h100x1"),
        (["NVIDIA L40S"], "h100x1"),
    ],
)
def test_detect_sglang_profiles(tmp_path: Path, names: list[str], expected: str) -> None:
    result = _run_detect(tmp_path, names)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["NVIDIA H100 80GB HBM3"], "single_offload"),
        (["NVIDIA H100 80GB HBM3"] * 4, "single_offload"),
        (["NVIDIA H200"] * 4, "single_offload"),
        (["NVIDIA B300"] * 4, "b300x4"),
        (["NVIDIA GeForce RTX 5090"] * 2, "rtx5090x2"),
        (["NVIDIA GeForce RTX 4090"] * 2, "rtx4090x2"),
        (["NVIDIA L40S"], "single_offload"),
    ],
)
def test_detect_vllm_profiles(tmp_path: Path, names: list[str], expected: str) -> None:
    result = _run_detect(tmp_path, names, backend="vllm_omni")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.parametrize("backend,expected", [("sglang", "h100x1"), ("vllm_omni", "single_offload")])
def test_detect_falls_back_without_nvidia_smi(
    tmp_path: Path, backend: str, expected: str
) -> None:
    result = _run_detect(tmp_path, None, backend=backend)
    assert result.returncode == 0
    assert result.stdout.strip() == expected


@pytest.mark.parametrize("names,exit_code", [([], 0), (["ignored-on-error"], 1)])
def test_detect_falls_back_on_empty_or_failed_query(
    tmp_path: Path, names: list[str], exit_code: int
) -> None:
    result = _run_detect(tmp_path, names, smi_exit_code=exit_code)
    assert result.returncode == 0
    assert result.stdout.strip() == "h100x1"


@pytest.mark.parametrize("backend", ["sglang", "vllm_omni"])
def test_detect_honors_fallback_override(tmp_path: Path, backend: str) -> None:
    result = _run_detect(tmp_path, None, backend=backend, fallback="site_default")
    assert result.returncode == 0
    assert result.stdout.strip() == "site_default"


def test_detect_rejects_invalid_backend_before_hardware_probe(tmp_path: Path) -> None:
    result = _run_detect(tmp_path, ["NVIDIA H100"], backend="vllm")
    assert result.returncode == 2
    assert result.stdout == ""
    assert "Usage:" in result.stderr


@pytest.mark.parametrize(
    ("visible", "expected"),
    [
        ("0", "h100x1"),
        ("1, 3", "h100x1"),
        ("0,2,3,4", "h100x1"),
        ("GPU-a,GPU-b", "h100x1"),
        ("MIG-GPU-a/1/0,MIG-GPU-b/2/0,MIG-GPU-c/3/0,MIG-GPU-d/4/0", "h100x1"),
    ],
)
def test_detect_respects_numeric_and_uuid_visibility(
    tmp_path: Path, visible: str, expected: str
) -> None:
    names = [
        "NVIDIA H100",
        "NVIDIA RTX 5090",
        "NVIDIA H100",
        "NVIDIA RTX 5090",
        "NVIDIA H100",
    ]
    result = _run_detect(tmp_path, names, cuda_visible_devices=visible)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.parametrize(
    ("visible", "expected"),
    [
        ("0", "single_offload"),
        ("0, 1", "rtx5090x2"),
        ("GPU-a,GPU-b", "rtx5090x2"),
        ("MIG-GPU-a/1/0,MIG-GPU-b/2/0", "rtx5090x2"),
    ],
)
def test_detect_vllm_respects_numeric_and_uuid_visibility(
    tmp_path: Path, visible: str, expected: str
) -> None:
    result = _run_detect(
        tmp_path,
        ["NVIDIA GeForce RTX 5090"] * 4,
        backend="vllm_omni",
        cuda_visible_devices=visible,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


def _arg_lines(stdout: str) -> list[str]:
    return [line[1:-1] for line in stdout.splitlines() if line.startswith("<") and line.endswith(">")]


def _contains_sequence(items: list[str], sequence: list[str]) -> bool:
    width = len(sequence)
    return any(items[index : index + width] == sequence for index in range(len(items) - width + 1))


def test_start_sglang_defaults_to_auto_h100x1_without_gpu(tmp_path: Path) -> None:
    fake_bin = _fake_smi(tmp_path, "partial output\n", exit_code=1)
    _executable(fake_bin / "nvcc", "#!/usr/bin/env bash\nexit 0\n")
    fake_sglang = _executable(
        tmp_path / "sglang",
        "#!/usr/bin/env bash\nprintf '<%s>\\n' \"$@\"\n",
    )
    env = os.environ.copy()
    env.update(
        H3_SGLANG_BIN=str(fake_sglang),
        H3_AUTO_FALLBACK_PROFILE="h100x1",
        PATH=f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    )
    env.pop("H3_PROFILE", None)
    env.pop("H3_H100_MODE", None)
    env.pop("H3_DIT_RESIDENT_LAYERS", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    result = subprocess.run(
        [str(START_SGLANG)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    args = _arg_lines(result.stdout)
    assert args[:5] == ["serve", "--model-path", "MiniMaxAI/MiniMax-H3", "--model-variant", "fl2va"]
    assert _contains_sequence(args, ["--num-gpus", "1", "--tp-size", "1", "--ulysses-degree", "1"])
    assert _contains_sequence(args, ["--performance-mode", "memory"])
    assert _contains_sequence(args, ["--dit-layerwise-resident-layers", "8"])
    assert "--component-residency" not in args
    assert _contains_sequence(args, ["--enable-torch-compile", "false"])
    assert _contains_sequence(args, ["--host", "127.0.0.1", "--port", "30010"])
    assert "Auto-selected SGLang profile: h100x1" in result.stderr
    assert "Single-H100 mode: speed (8 resident DiT layers)." in result.stderr


def test_start_sglang_single_gpu_allows_an_explicit_resident_layer_budget(tmp_path: Path) -> None:
    fake_bin = _fake_nvcc(tmp_path)
    fake_sglang = _executable(
        tmp_path / "sglang",
        "#!/usr/bin/env bash\nprintf '<%s>\\n' \"$@\"\n",
    )
    env = os.environ.copy()
    env.update(
        H3_SGLANG_BIN=str(fake_sglang),
        H3_PROFILE="h100x1",
        H3_DIT_RESIDENT_LAYERS="4",
        PATH=f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    )
    result = subprocess.run(
        [str(START_SGLANG)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert _contains_sequence(_arg_lines(result.stdout), ["--dit-layerwise-resident-layers", "4"])


def test_start_sglang_forces_cuda_jit_backend_on_nvidia_profiles(tmp_path: Path) -> None:
    fake_bin = _fake_nvcc(tmp_path)
    fake_sglang = _executable(
        tmp_path / "sglang",
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$TVM_FFI_GPU_BACKEND\"\n",
    )
    env = os.environ.copy()
    env.update(
        H3_SGLANG_BIN=str(fake_sglang),
        H3_PROFILE="h100x1",
        TVM_FFI_GPU_BACKEND="hip",
        PATH=f"{fake_bin}{os.pathsep}{env['PATH']}",
    )
    result = subprocess.run(
        [str(START_SGLANG)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "cuda"


def test_start_sglang_reports_when_cuda_compiler_is_missing(tmp_path: Path) -> None:
    fake_sglang = _executable(tmp_path / "sglang", "#!/usr/bin/env bash\nexit 99\n")
    env = os.environ.copy()
    env.update(H3_SGLANG_BIN=str(fake_sglang), H3_PROFILE="h100x1", PATH="/usr/bin:/bin")
    result = subprocess.run(
        [str(START_SGLANG)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 1
    assert "CUDA compiler (nvcc) is required" in result.stderr


def test_start_sglang_single_h100_memory_mode_offloads_vae(tmp_path: Path) -> None:
    fake_bin = _fake_nvcc(tmp_path)
    fake_sglang = _executable(
        tmp_path / "sglang",
        "#!/usr/bin/env bash\nprintf '<%s>\\n' \"$@\"\n",
    )
    env = os.environ.copy()
    env.update(
        H3_SGLANG_BIN=str(fake_sglang),
        H3_PROFILE="h100x1",
        H3_H100_MODE="memory",
        PATH=f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    )
    env.pop("H3_DIT_RESIDENT_LAYERS", None)
    result = subprocess.run(
        [str(START_SGLANG)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    args = _arg_lines(result.stdout)
    assert _contains_sequence(args, ["--dit-layerwise-resident-layers", "8"])
    assert "--component-residency" not in args
    assert "Single-H100 mode: memory (8 resident DiT layers)." in result.stderr


def test_start_sglang_rejects_invalid_single_h100_mode(tmp_path: Path) -> None:
    fake_sglang = _executable(tmp_path / "sglang", "#!/usr/bin/env bash\nexit 99\n")
    env = os.environ.copy()
    env.update(
        H3_SGLANG_BIN=str(fake_sglang),
        H3_PROFILE="h100x1",
        H3_H100_MODE="turbo-ish",
    )
    result = subprocess.run(
        [str(START_SGLANG)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 2
    assert "H3_H100_MODE must be speed or memory" in result.stderr


def test_start_sglang_rejects_invalid_single_gpu_resident_layer_budget(tmp_path: Path) -> None:
    fake_sglang = _executable(tmp_path / "sglang", "#!/usr/bin/env bash\nexit 99\n")
    env = os.environ.copy()
    env.update(
        H3_SGLANG_BIN=str(fake_sglang),
        H3_PROFILE="h100x1",
        H3_H100_MODE="speed",
        H3_DIT_RESIDENT_LAYERS="many",
    )
    result = subprocess.run(
        [str(START_SGLANG)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 2
    assert "H3_DIT_RESIDENT_LAYERS" in result.stderr


def test_start_vllm_defaults_to_auto_single_offload_without_gpu(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    (model_root / "FL2VA").mkdir(parents=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _executable(fake_bin / "nvidia-smi", "#!/usr/bin/env bash\nexit 1\n")
    _executable(fake_bin / "docker", "#!/usr/bin/env bash\nprintf '<%s>\\n' \"$@\"\n")
    env = os.environ.copy()
    env.update(
        H3_MODEL_DIR=str(model_root),
        H3_AUTO_FALLBACK_PROFILE="single_offload",
        PATH=f"{fake_bin}{os.pathsep}{env['PATH']}",
    )
    env.pop("H3_PROFILE", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    result = subprocess.run(
        [str(START_VLLM)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    args = _arg_lines(result.stdout)
    assert args[:4] == ["run", "--rm", "--gpus", "1"]
    assert _contains_sequence(args, ["-p", "127.0.0.1:30010:30010"])
    assert _contains_sequence(args, ["--num-gpus", "1", "--enable-cpu-offload"])
    assert "Auto-selected vLLM-Omni profile: single_offload" in result.stderr


def test_start_vllm_auto_expands_two_5090_profile(tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    (model_root / "Ref2VA").mkdir(parents=True)
    fake_bin = _fake_smi(tmp_path, "NVIDIA GeForce RTX 5090\n" * 2)
    _executable(fake_bin / "docker", "#!/usr/bin/env bash\nprintf '<%s>\\n' \"$@\"\n")
    env = os.environ.copy()
    env.update(H3_MODEL_DIR=str(model_root), PATH=f"{fake_bin}{os.pathsep}{env['PATH']}")
    env.pop("H3_PROFILE", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    result = subprocess.run(
        [str(START_VLLM), "Ref2VA"], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    args = _arg_lines(result.stdout)
    assert args[:4] == ["run", "--rm", "--gpus", "2"]
    assert _contains_sequence(args, ["-p", "127.0.0.1:30011:30011"])
    assert _contains_sequence(args, ["--num-gpus", "2", "--tensor-parallel-size", "2"])
    assert _contains_sequence(args, ["--dlo-resident-layers", "20"])
    assert "Auto-selected vLLM-Omni profile: rtx5090x2" in result.stderr


@pytest.mark.parametrize(
    "visible",
    [
        "1,3",
        "GPU-aaaaaaaa,GPU-bbbbbbbb",
        "MIG-GPU-aaaaaaaa/1/0,MIG-GPU-bbbbbbbb/2/0",
    ],
)
def test_start_vllm_pins_exact_visible_device_selector(
    tmp_path: Path, visible: str
) -> None:
    model_root = tmp_path / "models"
    (model_root / "FL2VA").mkdir(parents=True)
    fake_bin = _fake_smi(tmp_path, "NVIDIA GeForce RTX 5090\n" * 4)
    _executable(fake_bin / "docker", "#!/usr/bin/env bash\nprintf '<%s>\\n' \"$@\"\n")
    env = os.environ.copy()
    env.update(
        H3_MODEL_DIR=str(model_root),
        CUDA_VISIBLE_DEVICES=visible,
        PATH=f"{fake_bin}{os.pathsep}{env['PATH']}",
    )
    env.pop("H3_PROFILE", None)
    result = subprocess.run(
        [str(START_VLLM)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
    args = _arg_lines(result.stdout)
    assert args[:4] == ["run", "--rm", "--gpus", f'"device={visible}"']
    assert _contains_sequence(args, ["--num-gpus", "2", "--tensor-parallel-size", "2"])
    assert "Auto-selected vLLM-Omni profile: rtx5090x2" in result.stderr
