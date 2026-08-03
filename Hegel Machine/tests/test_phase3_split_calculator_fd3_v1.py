from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from typing import Sequence

import pytest

from hegel_machine.strict_cbor_v1 import canonical_cbor_decode, canonical_cbor_encode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_CALCULATOR = PROJECT_ROOT / "tools" / "phase3_split_calculator_fd3_v1.py"
RUST_SOURCE = PROJECT_ROOT / "tools" / "phase3_split_calculator_fd3_v1.rs"
PINNED_RUST_IMAGE = (
    "rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89"
)
EXPECTED_RUST_IMAGE_ID = (
    "sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89"
)
SCHEMA_ID = b"hegel-phase3-split-calculator-fd3-response/1"
KNOWN_SEED = bytes(range(32))
KNOWN_COMMITMENT = bytes.fromhex(
    "3126668b3227a5e6ab711bcaa66f9d573a7e8bf8b1d1c6cabbb07a96ccf566ba"
)
EXPECTED_PAYLOAD = canonical_cbor_encode((1, SCHEMA_ID, KNOWN_COMMITMENT))
EXPECTED_FRAME = len(EXPECTED_PAYLOAD).to_bytes(8, "big") + EXPECTED_PAYLOAD


@pytest.fixture(scope="session")
def rust_calculator(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp("phase3-fd3-rust")
    binary = output_dir / "phase3_split_calculator_fd3_v1"
    explicit_rustc = os.environ.get("HEGEL_FD3_RUSTC")
    if explicit_rustc:
        rustc = Path(explicit_rustc)
        if not rustc.is_absolute() or not rustc.is_file():
            pytest.fail("HEGEL_FD3_RUSTC must name an explicit absolute rustc path")
        command = [
            str(rustc),
            "--edition=2021",
            "-C",
            "opt-level=2",
            "-C",
            "debuginfo=0",
            "-C",
            "strip=symbols",
            "-C",
            "codegen-units=1",
            "-o",
            str(binary),
            str(RUST_SOURCE),
        ]
    else:
        docker = shutil.which("docker")
        if docker is None:
            pytest.fail("offline pinned Rust compiler image requires docker")
        inspected = subprocess.run(
            [docker, "image", "inspect", "--format", "{{.Id}}", PINNED_RUST_IMAGE],
            check=False,
            capture_output=True,
            text=True,
        )
        assert inspected.returncode == 0, inspected.stderr
        assert inspected.stdout.strip() == EXPECTED_RUST_IMAGE_ID
        command = [
            docker,
            "run",
            "--rm",
            "--pull=never",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--user={os.getuid()}:{os.getgid()}",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m",
            f"--mount=type=bind,src={RUST_SOURCE.parent},dst=/src,readonly",
            f"--mount=type=bind,src={output_dir},dst=/out",
            PINNED_RUST_IMAGE,
            "rustc",
            "--edition=2021",
            "-C",
            "opt-level=2",
            "-C",
            "debuginfo=0",
            "-C",
            "strip=symbols",
            "-C",
            "codegen-units=1",
            "-o",
            "/out/phase3_split_calculator_fd3_v1",
            "/src/phase3_split_calculator_fd3_v1.rs",
        ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert binary.is_file()
    assert stat.S_IMODE(binary.stat().st_mode) & 0o111
    return binary


def _commands(rust_calculator: Path) -> tuple[tuple[str, ...], ...]:
    return (
        (shutil.which("python3") or "python3", str(PYTHON_CALCULATOR)),
        (str(rust_calculator),),
    )


_FD_LAUNCHER = """
import os
import sys
import fcntl

seed_source = int(sys.argv[1])
response_source = int(sys.argv[2])
seed_copy = fcntl.fcntl(seed_source, fcntl.F_DUPFD, 10) if seed_source >= 0 else -1
response_copy = fcntl.fcntl(response_source, fcntl.F_DUPFD, 10) if response_source >= 0 else -1
for target in (3, 5):
    try:
        os.close(target)
    except OSError:
        pass
if seed_copy >= 0:
    os.dup2(seed_copy, 3, inheritable=True)
if response_copy >= 0:
    os.dup2(response_copy, 5, inheritable=True)
for descriptor in (seed_copy, response_copy, seed_source, response_source):
    if descriptor not in (-1, 3, 5):
        try:
            os.close(descriptor)
        except OSError:
            pass
os.execv(sys.argv[3], sys.argv[3:])
"""


def _fd_launcher_command(
    command: Sequence[str],
    seed_read_fd: int | None,
    response_write_fd: int | None,
) -> list[str]:
    return [
        shutil.which("python3") or "python3",
        "-c",
        _FD_LAUNCHER,
        str(seed_read_fd if seed_read_fd is not None else -1),
        str(response_write_fd if response_write_fd is not None else -1),
        *command,
    ]


def _run_with_pipes(
    command: Sequence[str],
    seed: bytes,
    *,
    extra_args: Sequence[str] = (),
    provide_seed_fd: bool = True,
    provide_response_fd: bool = True,
    stdin_payload: bytes | None = None,
    extra_environment: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[bytes], bytes]:
    seed_read_fd, seed_write_fd = os.pipe()
    response_read_fd, response_write_fd = os.pipe()
    pass_fds: list[int] = []
    if provide_seed_fd:
        pass_fds.append(seed_read_fd)
    if provide_response_fd:
        pass_fds.append(response_write_fd)
    environment = os.environ.copy()
    if extra_environment:
        environment.update(extra_environment)
    process = subprocess.Popen(
        _fd_launcher_command(
            [*command, *extra_args],
            seed_read_fd if provide_seed_fd else None,
            response_write_fd if provide_response_fd else None,
        ),
        stdin=subprocess.PIPE if stdin_payload is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=tuple(pass_fds),
        env=environment,
    )
    os.close(seed_read_fd)
    os.close(response_write_fd)
    try:
        if provide_seed_fd:
            try:
                os.write(seed_write_fd, seed)
            except BrokenPipeError:
                pass
    finally:
        os.close(seed_write_fd)
    if not provide_response_fd:
        os.close(response_read_fd)
        response = b""
    else:
        chunks = []
        while True:
            chunk = os.read(response_read_fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
        os.close(response_read_fd)
        response = b"".join(chunks)
    stdout, stderr = process.communicate(input=stdin_payload, timeout=10)
    completed = subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)
    return completed, response


def _decode_single_frame(frame: bytes) -> tuple[object, ...]:
    assert len(frame) >= 8
    payload_length = int.from_bytes(frame[:8], "big")
    payload = frame[8:]
    assert payload_length == len(payload)
    decoded = canonical_cbor_decode(payload)
    assert canonical_cbor_encode(decoded) == payload
    assert isinstance(decoded, tuple)
    return decoded


def test_known_vector_and_python_rust_byte_agreement(rust_calculator: Path) -> None:
    frames = []
    for command in _commands(rust_calculator):
        completed, frame = _run_with_pipes(command, KNOWN_SEED)
        assert completed.returncode == 0
        assert completed.stdout == b""
        assert completed.stderr == b""
        assert frame == EXPECTED_FRAME
        assert _decode_single_frame(frame) == (1, SCHEMA_ID, KNOWN_COMMITMENT)
        frames.append(frame)
    assert frames[0] == frames[1]


@pytest.mark.parametrize("seed", [b"", b"x" * 31, b"x" * 33])
def test_wrong_length_or_extra_byte_fails_closed(
    rust_calculator: Path, seed: bytes
) -> None:
    for command in _commands(rust_calculator):
        completed, frame = _run_with_pipes(command, seed)
        assert completed.returncode != 0
        assert completed.stdout == b""
        assert completed.stderr == b""
        assert frame == b""


def test_missing_contract_fd_fails_closed(rust_calculator: Path) -> None:
    for command in _commands(rust_calculator):
        missing_seed, frame = _run_with_pipes(
            command, KNOWN_SEED, provide_seed_fd=False
        )
        assert missing_seed.returncode != 0
        assert missing_seed.stdout == missing_seed.stderr == frame == b""
        missing_output, frame = _run_with_pipes(
            command, KNOWN_SEED, provide_response_fd=False
        )
        assert missing_output.returncode != 0
        assert missing_output.stdout == missing_output.stderr == frame == b""


def test_argv_stdin_and_environment_are_not_secret_fallbacks(
    rust_calculator: Path,
) -> None:
    seed_hex = KNOWN_SEED.hex()
    for command in _commands(rust_calculator):
        argv_result, argv_frame = _run_with_pipes(
            command, KNOWN_SEED, extra_args=(seed_hex,)
        )
        assert argv_result.returncode != 0
        assert argv_result.stdout == argv_result.stderr == argv_frame == b""

        stdin_result, stdin_frame = _run_with_pipes(
            command,
            b"",
            provide_seed_fd=False,
            stdin_payload=KNOWN_SEED,
        )
        assert stdin_result.returncode != 0
        assert stdin_result.stdout == stdin_result.stderr == stdin_frame == b""

        env_result, env_frame = _run_with_pipes(
            command,
            b"",
            provide_seed_fd=False,
            extra_environment={"HEGEL_SPLIT_SEED": seed_hex},
        )
        assert env_result.returncode != 0
        assert env_result.stdout == env_result.stderr == env_frame == b""


def test_seekable_fd3_seed_replay_is_rejected(rust_calculator: Path) -> None:
    for command in _commands(rust_calculator):
        with tempfile.TemporaryFile() as seed_file:
            seed_file.write(KNOWN_SEED)
            seed_file.seek(0)
            response_read_fd, response_write_fd = os.pipe()
            process = subprocess.Popen(
                _fd_launcher_command(
                    command, seed_file.fileno(), response_write_fd
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(seed_file.fileno(), response_write_fd),
            )
            os.close(response_write_fd)
            frame = os.read(response_read_fd, 4096)
            assert os.read(response_read_fd, 1) == b""
            os.close(response_read_fd)
            stdout, stderr = process.communicate(timeout=10)
        assert process.returncode != 0
        assert stdout == stderr == frame == b""


def test_output_is_exactly_one_length_delimited_canonical_frame(
    rust_calculator: Path,
) -> None:
    expected_digest = hashlib.sha256(
        b"HEGEL/SPLIT_MASTER_SEED_COMMITMENT/V1\x00" + KNOWN_SEED
    ).digest()
    for command in _commands(rust_calculator):
        completed, frame = _run_with_pipes(command, KNOWN_SEED)
        assert completed.returncode == 0
        assert frame[:8] == (len(frame) - 8).to_bytes(8, "big")
        assert _decode_single_frame(frame) == (1, SCHEMA_ID, expected_digest)
        # The pipe reached EOF; no second frame or trailing byte exists.
        assert frame == EXPECTED_FRAME


def test_both_calculators_attempt_lock_then_zeroize_then_unlock() -> None:
    python_source = PYTHON_CALCULATOR.read_text(encoding="utf-8")
    rust_source = RUST_SOURCE.read_text(encoding="utf-8")

    assert "libc.mlock" in python_source
    assert "libc.munlock" in python_source
    assert "seed_locked = _try_mlock(seed)" in python_source
    assert "_zeroize(seed)\n        _try_munlock(seed, seed_locked)" in python_source

    assert "fn mlock(address: *const c_void, length: usize)" in rust_source
    assert "fn munlock(address: *const c_void, length: usize)" in rust_source
    assert "let seed_locked = try_mlock(seed.as_mut_slice());" in rust_source
    assert (
        "zeroize(seed.as_mut_slice());\n"
        "    try_munlock(seed.as_mut_slice(), seed_locked);"
    ) in rust_source
