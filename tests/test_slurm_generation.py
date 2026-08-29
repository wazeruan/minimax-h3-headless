from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SBATCH_GENERATE = REPO / "deploy" / "slurm" / "h3-generate.sbatch"
SUBMIT = REPO / "scripts" / "submit_slurm_generation.sh"


def _executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    return path


def _copy_runtime(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "repo with spaces"
    batch = root / "deploy" / "slurm" / "h3-generate.sbatch"
    batch.parent.mkdir(parents=True)
    shutil.copy2(SBATCH_GENERATE, batch)

    state = tmp_path / "state"
    state.mkdir()
    fake_bin = tmp_path / "fake bin"
    fake_bin.mkdir()

    _executable(
        fake_bin / "nvidia-smi",
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' 'NVIDIA H100 80GB HBM3, 81559'\n",
    )
    _executable(
        fake_bin / "curl",
        "#!/usr/bin/env bash\n"
        "[[ -f \"$FAKE_SLURM_STATE/ready\" ]]\n",
    )
    _executable(root / ".venv" / "bin" / "sglang", "#!/usr/bin/env bash\nexit 0\n")
    _executable(
        root / "deploy" / "start_sglang.sh",
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$1\" >\"$FAKE_SLURM_STATE/variant\"\n"
        "printf '%s\\n' \"$H3_INFERENCE_HOST\" >\"$FAKE_SLURM_STATE/host\"\n"
        "printf '%s\\n' \"$H3_INFERENCE_PORT\" >\"$FAKE_SLURM_STATE/port\"\n"
        "printf '%s\\n' \"$H3_PROFILE\" >\"$FAKE_SLURM_STATE/profile\"\n"
        "touch \"$FAKE_SLURM_STATE/ready\"\n"
        "trap 'touch \"$FAKE_SLURM_STATE/stopped\"; exit 0' TERM INT\n"
        "while true; do sleep 1; done\n",
    )
    _executable(
        root / "scripts" / "generate_sglang.sh",
        "#!/usr/bin/env bash\n"
        "prompt= output= url=\n"
        "generation_args=(\"$@\")\n"
        "while (( $# )); do\n"
        "  case \"$1\" in\n"
        "    --prompt) prompt=$2; shift 2 ;;\n"
        "    --output) output=$2; shift 2 ;;\n"
        "    --url) url=$2; shift 2 ;;\n"
        "    -*) shift 2 ;;\n"
        "    *) if [[ -z \"$prompt\" ]]; then prompt=$1; elif [[ -z \"$output\" ]]; then output=$1; fi; shift ;;\n"
        "  esac\n"
        "done\n"
        "printf '<%s>\\n' \"${generation_args[@]}\" >\"$FAKE_SLURM_STATE/generation-args\"\n"
        "printf '%s' \"$prompt\" >\"$FAKE_SLURM_STATE/prompt\"\n"
        "printf '%s' \"$output\" >\"$FAKE_SLURM_STATE/output\"\n"
        "printf '%s' \"$url\" >\"$FAKE_SLURM_STATE/url\"\n"
        "[[ \"${FAKE_GENERATION_FAIL:-0}\" != 1 ]] || exit 23\n"
        "mkdir -p -- \"$(dirname -- \"$output\")\"\n"
        "printf 'fake mp4' >\"$output\"\n",
    )
    model = root / "models" / "MiniMax-H3"
    (model / "FL2VA").mkdir(parents=True)
    (model / "model_index.json").write_text("{}")
    return root, batch, state


def test_one_shot_slurm_job_starts_generates_and_stops(tmp_path: Path) -> None:
    root, batch, state = _copy_runtime(tmp_path)
    env = os.environ.copy()
    for key in (
        "H3_PROFILE",
        "H3_INFERENCE_HOST",
        "H3_INFERENCE_PORT",
        "H3_SGLANG_URL",
        "H3_MODULES_FILE",
        "H3_SLURM_LOG_DIR",
    ):
        env.pop(key, None)
    env.update(
        FAKE_SLURM_STATE=str(state),
        H3_REPO_DIR=str(root),
        H3_MODEL_PATH=str(root / "models" / "MiniMax-H3"),
        H3_SGLANG_BIN=str(root / ".venv" / "bin" / "sglang"),
        H3_STARTUP_TIMEOUT_SECONDS="5",
        SLURM_JOB_ID="9876",
        PATH=f"{tmp_path / 'fake bin'}{os.pathsep}{env['PATH']}",
    )
    result = subprocess.run(
        [str(batch), "A quiet moonlit lake"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    output = root / "outputs" / "h3-9876.mp4"
    assert output.read_bytes() == b"fake mp4"
    assert (state / "prompt").read_text() == "A quiet moonlit lake"
    assert (state / "output").read_text() == str(output)
    assert (state / "variant").read_text().strip() == "fl2va"
    assert (state / "host").read_text().strip() == "127.0.0.1"
    port = int((state / "port").read_text())
    assert 1024 <= port <= 65535
    assert (state / "profile").read_text().strip() == "h100x1"
    assert (state / "url").read_text() == f"http://127.0.0.1:{port}"
    assert (state / "stopped").is_file()
    assert "Slurm generation complete" in result.stdout


def test_one_shot_slurm_job_stops_server_when_generation_fails(tmp_path: Path) -> None:
    root, batch, state = _copy_runtime(tmp_path)
    env = os.environ.copy()
    for key in ("H3_PROFILE", "H3_MODULES_FILE", "H3_SLURM_LOG_DIR"):
        env.pop(key, None)
    env.update(
        FAKE_SLURM_STATE=str(state),
        FAKE_GENERATION_FAIL="1",
        H3_REPO_DIR=str(root),
        H3_MODEL_PATH=str(root / "models" / "MiniMax-H3"),
        H3_SGLANG_BIN=str(root / ".venv" / "bin" / "sglang"),
        H3_INFERENCE_PORT="32124",
        H3_STARTUP_TIMEOUT_SECONDS="5",
        SLURM_JOB_ID="9877",
        PATH=f"{tmp_path / 'fake bin'}{os.pathsep}{env['PATH']}",
    )
    result = subprocess.run(
        [str(batch), "A failed generation"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 23
    assert (state / "stopped").is_file()
    assert not (root / "outputs" / "h3-9877.mp4").exists()


def test_one_shot_slurm_job_forwards_cli_generation_options(tmp_path: Path) -> None:
    root, batch, state = _copy_runtime(tmp_path)
    output = root / "outputs" / "vertical.mp4"
    env = os.environ.copy()
    env.update(
        FAKE_SLURM_STATE=str(state),
        H3_REPO_DIR=str(root),
        H3_MODEL_PATH=str(root / "models" / "MiniMax-H3"),
        H3_SGLANG_BIN=str(root / ".venv" / "bin" / "sglang"),
        H3_STARTUP_TIMEOUT_SECONDS="5",
        SLURM_JOB_ID="9878",
        PATH=f"{tmp_path / 'fake bin'}{os.pathsep}{env['PATH']}",
    )
    result = subprocess.run(
        [
            str(batch),
            "--prompt",
            "A vertical city walk",
            "--duration",
            "10",
            "--aspect-ratio",
            "9:16",
            "--seed",
            "123",
            "--output",
            str(output),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == b"fake mp4"
    assert (state / "prompt").read_text() == "A vertical city walk"
    assert (state / "output").read_text() == str(output)
    args = (state / "generation-args").read_text().splitlines()
    assert "<--duration>" in args
    assert "<10>" in args
    assert "<--aspect-ratio>" in args
    assert "<9:16>" in args
    assert "<--seed>" in args
    assert "<123>" in args


def test_one_shot_slurm_job_accepts_output_after_named_prompt(tmp_path: Path) -> None:
    root, batch, state = _copy_runtime(tmp_path)
    output = root / "outputs" / "named-prompt.mp4"
    env = os.environ.copy()
    env.update(
        FAKE_SLURM_STATE=str(state),
        H3_REPO_DIR=str(root),
        H3_MODEL_PATH=str(root / "models" / "MiniMax-H3"),
        H3_SGLANG_BIN=str(root / ".venv" / "bin" / "sglang"),
        H3_STARTUP_TIMEOUT_SECONDS="5",
        SLURM_JOB_ID="9879",
        PATH=f"{tmp_path / 'fake bin'}{os.pathsep}{env['PATH']}",
    )
    result = subprocess.run(
        [str(batch), "--prompt", "A distant lighthouse", str(output)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == b"fake mp4"
    assert (state / "output").read_text() == str(output)


def test_submission_helper_builds_portable_sbatch_command(tmp_path: Path) -> None:
    root = tmp_path / "repo with spaces"
    helper = root / "scripts" / "submit_slurm_generation.sh"
    helper.parent.mkdir(parents=True)
    shutil.copy2(SUBMIT, helper)
    batch = root / "deploy" / "slurm" / "h3-generate.sbatch"
    batch.parent.mkdir(parents=True)
    batch.write_text("#!/usr/bin/env bash\n")
    (root / ".env").write_text("H3_SLURM_ACCOUNT='project account'\nH3_SLURM_PARTITION=gpu\n")

    state = tmp_path / "sbatch.json"
    fake_bin = tmp_path / "fake bin"
    fake_bin.mkdir()
    _executable(
        fake_bin / "sbatch",
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"open({str(state)!r}, 'w').write(json.dumps({{'args': sys.argv[1:], 'repo': os.environ.get('H3_REPO_DIR')}}))\n"
        "print('Submitted batch job 12345')\n",
    )
    env = os.environ.copy()
    for key in (
        "H3_SLURM_ACCOUNT",
        "H3_SLURM_PARTITION",
        "H3_SLURM_GPU_OPTION",
        "H3_SLURM_TIME",
    ):
        env.pop(key, None)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    output = root / "outputs" / "lake.mp4"
    result = subprocess.run(
        [str(helper), "A lake, with rain", str(output)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    call = json.loads(state.read_text())
    assert call["repo"] == str(root)
    assert call["args"] == [
        "--export=ALL",
        "--gpus-per-node=h100:1",
        "--account=project account",
        "--partition=gpu",
        str(batch),
        "A lake, with rain",
        str(output),
    ]
    assert result.stdout == "Submitted batch job 12345\n"


def test_slurm_generation_scripts_are_executable_and_valid_bash() -> None:
    for script in (SBATCH_GENERATE, SUBMIT):
        assert script.stat().st_mode & stat.S_IXUSR
        result = subprocess.run(["bash", "-n", str(script)], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
