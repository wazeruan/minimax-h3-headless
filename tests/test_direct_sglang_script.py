from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
GENERATE = REPO / "scripts" / "generate_sglang.sh"
H3 = REPO / "h3.sh"


def _executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    return path


def _sandbox(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "repo with spaces"
    script = root / "scripts" / "generate_sglang.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(GENERATE, script)

    state = tmp_path / "curl state"
    state.mkdir()
    fake_bin = tmp_path / "fake bin"
    fake_bin.mkdir()
    _executable(
        fake_bin / "curl",
        f"#!{sys.executable}\n"
        + r'''
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
state = Path(os.environ["FAKE_CURL_STATE_DIR"])
with (state / "calls.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(args) + "\n")

def after(flag):
    return args[args.index(flag) + 1]

url = next((item for item in args if item.startswith(("http://", "https://"))), "")
output = Path(after("--output"))
phase = "download" if url.endswith("/content") else ("submit" if "--request" in args else "status")
if os.environ.get("FAKE_CURL_FAIL") == phase:
    if phase == "download" and os.environ.get("FAKE_CURL_PARTIAL") == "1":
        output.write_bytes(b"partial error body")
    print(f"fake {phase} error", file=sys.stderr)
    raise SystemExit(22)

if phase == "submit":
    request_path = Path(after("--data-binary").removeprefix("@"))
    (state / "request.json").write_text(request_path.read_text(encoding="utf-8"), encoding="utf-8")
    output.write_text(os.environ.get("FAKE_SUBMIT_BODY", '{"id":"job/quoted id"}'), encoding="utf-8")
elif phase == "status":
    count_file = state / "status-count"
    count = int(count_file.read_text() if count_file.exists() else "0")
    statuses = os.environ.get("FAKE_STATUSES", "queued,running,completed").split(",")
    status = statuses[min(count, len(statuses) - 1)]
    count_file.write_text(str(count + 1), encoding="utf-8")
    response = {"status": status}
    if status in {"failed", "cancelled"}:
        response["error"] = os.environ.get("FAKE_ERROR", "backend exploded")
    output.write_text(json.dumps(response), encoding="utf-8")
else:
    output.write_bytes(b"fake mp4 payload")
''',
    )
    return root, script, state


def _run(
    tmp_path: Path,
    *args: str,
    stdin: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    root, script, state = _sandbox(tmp_path)
    env = os.environ.copy()
    for key in (
        "H3_SGLANG_URL",
        "H3_PORT",
        "H3_INFERENCE_PORT",
        "H3_POLL_INTERVAL_SECONDS",
        "H3_GENERATION_TIMEOUT_SECONDS",
        "H3_DURATION_SECONDS",
        "H3_ASPECT_RATIO",
        "H3_SEED",
        "H3_NUM_INFERENCE_STEPS",
    ):
        env.pop(key, None)
    env.update(
        FAKE_CURL_STATE_DIR=str(state),
        PATH=f"{tmp_path / 'fake bin'}{os.pathsep}{env['PATH']}",
    )
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(
        [str(script), "--poll-interval", "0", *args],
        input=stdin,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    return result, root, state


def _calls(state: Path) -> list[list[str]]:
    return [json.loads(line) for line in (state / "calls.jsonl").read_text().splitlines()]


def _request(state: Path) -> dict[str, object]:
    return json.loads((state / "request.json").read_text())


def test_direct_sglang_generation_uses_the_official_t2va_shape(tmp_path: Path) -> None:
    output = tmp_path / "nested output" / "movie.mp4"
    result, _, state = _run(
        tmp_path,
        "--url",
        "https://sglang.example/base",
        "--duration",
        "10.5",
        "--aspect-ratio",
        "9:16",
        "--seed",
        "123",
        "--steps",
        "9",
        "--flow-shift",
        "11.5",
        "--audio-flow-shift",
        "2.5",
        "--model",
        "local-minimax-h3",
        "--timeout",
        "5",
        "--prompt",
        "A moonlit fox",
        "--output",
        str(output),
        env_overrides={
            "H3_SGLANG_URL": "https://ignored.example",
            "H3_DURATION_SECONDS": "4",
            "H3_ASPECT_RATIO": "1:1",
            "H3_SEED": "999",
            "H3_NUM_INFERENCE_STEPS": "1",
        },
    )
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == b"fake mp4 payload"
    request = _request(state)
    assert request == {
        "model": "local-minimax-h3",
        "prompt": "A moonlit fox",
        "seconds": 10.5,
        "task": "t2va",
        "conditions": [],
        "target": {"short_edge": 768, "aspect_ratio": "9:16", "duration_seconds": 10.5},
        "num_outputs_per_prompt": 1,
        "num_inference_steps": 9,
        "flow_shift": 11.5,
        "audio_flow_shift": 2.5,
        "seed": 123,
    }
    calls = _calls(state)
    assert "https://sglang.example/base/v1/videos" in calls[0]
    assert any("job%2Fquoted%20id" in item for call in calls[1:] for item in call)
    assert not any("Authorization:" in item for call in calls for item in call)


def test_direct_sglang_generation_preserves_unicode_prompt_without_shell_interpolation(
    tmp_path: Path,
) -> None:
    prompt = '你好 "quoted"\nsecond line $HOME `touch NEVER` \\ end'
    output = tmp_path / "unicode.mp4"
    result, _, state = _run(tmp_path, prompt, str(output))
    assert result.returncode == 0, result.stderr
    assert _request(state)["prompt"] == prompt
    assert not (tmp_path / "NEVER").exists()


def test_direct_sglang_generation_reports_terminal_failure_and_preserves_output(tmp_path: Path) -> None:
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"keep this")
    result, _, _ = _run(
        tmp_path,
        "prompt",
        str(output),
        env_overrides={"FAKE_STATUSES": "failed", "FAKE_ERROR": "GPU unavailable"},
    )
    assert result.returncode == 1
    assert "Generation failed: GPU unavailable" in result.stderr
    assert output.read_bytes() == b"keep this"


def test_direct_sglang_generation_rejects_invalid_settings_before_network_request(tmp_path: Path) -> None:
    result, _, state = _run(
        tmp_path,
        "--duration",
        "3",
        "prompt",
    )
    assert result.returncode == 1
    assert "--duration must be between 4 and 15" in result.stderr
    assert not (state / "calls.jsonl").exists()


def test_h3_wrapper_exposes_a_non_gateway_headless_interface() -> None:
    result = subprocess.run([str(H3), "--help"], text=True, capture_output=True, check=False)
    assert result.returncode == 0
    assert "generate [OPTIONS] [PROMPT] [FILE]" in result.stdout
    assert "ComfyUI" not in result.stdout


def test_single_h100_scripts_are_executable_and_valid_bash() -> None:
    for script in (H3, GENERATE):
        assert script.stat().st_mode & stat.S_IXUSR
        result = subprocess.run(["bash", "-n", str(script)], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
