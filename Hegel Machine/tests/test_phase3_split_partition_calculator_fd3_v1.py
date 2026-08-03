from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import runpy
import shutil
import stat
import subprocess
import tempfile
from typing import Sequence

import pytest

from hegel_machine.phase3_m25_container_ceremony_v1 import (
    M25ContainerCeremonyError,
    SPLIT_RESPONSE_ROWS,
    SPLIT_RESPONSE_SCHEMA_ID,
    SplitCalculatorPublicResponseV2,
    SplitRootCommitment,
    decode_split_calculator_public_frame_v2,
    encode_split_calculator_public_frame_v2,
)
from hegel_machine.phase3_m25_rows_v1 import (
    generate_odd_role_rows_v1,
    generate_sink_role_rows_v1,
)
from hegel_machine.phase3_m25_split_v1 import (
    allocate_typed_role_rows,
    derive_role_key,
    split_partition_commitments,
)
from hegel_machine.strict_cbor_v1 import canonical_cbor_decode, canonical_cbor_encode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_CALCULATOR = (
    PROJECT_ROOT / "tools" / "phase3_split_partition_calculator_fd3_v1.py"
)
RUST_SOURCE = PROJECT_ROOT / "tools" / "phase3_split_partition_calculator_fd3_v1.rs"
PINNED_RUST_IMAGE = (
    "rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89"
)
EXPECTED_RUST_IMAGE_ID = (
    "sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89"
)

KNOWN_SEED = bytes(range(32))
KNOWN_COMMITMENT = bytes.fromhex(
    "3126668b3227a5e6ab711bcaa66f9d573a7e8bf8b1d1c6cabbb07a96ccf566ba"
)
ODD_QUOTAS = (
    (1, 16, 6, 3, 7),
    (2, 16, 6, 3, 7),
    (3, 32, 13, 6, 13),
    (4, 32, 13, 6, 13),
    (5, 64, 26, 13, 25),
    (6, 64, 26, 13, 25),
    (7, 128, 51, 26, 51),
    (8, 128, 51, 26, 51),
)
SINK_QUOTAS = (
    (9, 15, 7, 4, 4),
    (10, 18, 8, 4, 6),
    (11, 19, 9, 4, 6),
    (12, 18, 8, 4, 6),
    (13, 15, 7, 4, 4),
)
KNOWN_ODD_ROOTS = tuple(
    bytes.fromhex(value)
    for value in (
        "f589b02ad89b112dc6041523bd25679d4277f9f83ae7f42c2329daf4abdf68ce",
        "6ef3d0d98e9931368f4e1e0287f15f37258808aee2eee514b9a10fe73299cd9b",
        "7e2169d39bf0fce7b761495c3c0de257e7dcf1f239f19f366a9023a0e6e3d693",
    )
)
KNOWN_SINK_ROOTS = tuple(
    bytes.fromhex(value)
    for value in (
        "d3f0da9f60218d559b6f5250bbbe629a7e1d9407c491c5c43b104d9eae8fe788",
        "b9d9ecea84600a93ecffebeac39293fe7e448ae4fc484c25e50a1ba7cf78aa66",
        "2f446c9b54b6a9087e932a1fdb9a2111531f240b238b13d2449acc936cb674f9",
    )
)


@pytest.fixture(scope="session")
def split_partition_rust_calculator(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp("phase3-split-partition-fd3-rust")
    binary = output_dir / "phase3_split_partition_calculator_fd3_v1"
    explicit_rustc = os.environ.get("HEGEL_SPLIT_PARTITION_FD3_RUSTC")
    if explicit_rustc:
        rustc = Path(explicit_rustc)
        if not rustc.is_absolute() or not rustc.is_file():
            pytest.fail(
                "HEGEL_SPLIT_PARTITION_FD3_RUSTC must name an absolute rustc path"
            )
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
            "/out/phase3_split_partition_calculator_fd3_v1",
            "/src/phase3_split_partition_calculator_fd3_v1.rs",
        ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert binary.is_file()
    assert stat.S_IMODE(binary.stat().st_mode) & 0o111
    return binary


def _commands(rust_calculator: Path) -> tuple[tuple[str, ...], ...]:
    return (
        (shutil.which("python3") or "python3", "-I", str(PYTHON_CALCULATOR)),
        (str(rust_calculator),),
    )


_FD_LAUNCHER = """
import fcntl
import os
import sys

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
        "-I",
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
    if provide_response_fd:
        chunks: list[bytes] = []
        while True:
            chunk = os.read(response_read_fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
        response = b"".join(chunks)
        os.close(response_read_fd)
    else:
        os.close(response_read_fd)
        response = b""
    stdout, stderr = process.communicate(input=stdin_payload, timeout=20)
    return (
        subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr),
        response,
    )


def _decode_and_validate_frame(frame: bytes) -> SplitCalculatorPublicResponseV2:
    response = decode_split_calculator_public_frame_v2(frame)
    assert encode_split_calculator_public_frame_v2(response) == frame
    assert tuple(
        (row.role_id, row.partition_id, row.row_count)
        for row in response.partitions
    ) == SPLIT_RESPONSE_ROWS
    return response


def test_known_vector_roots_counts_quotas_and_dual_byte_agreement(
    split_partition_rust_calculator: Path,
) -> None:
    frames = []
    for command in _commands(split_partition_rust_calculator):
        completed, frame = _run_with_pipes(command, KNOWN_SEED)
        assert completed.returncode == 0
        assert completed.stdout == completed.stderr == b""
        response = _decode_and_validate_frame(frame)
        assert response.seed_commitment == KNOWN_COMMITMENT
        assert tuple(row.root for row in response.partitions[:3]) == KNOWN_ODD_ROOTS
        assert tuple(row.root for row in response.partitions[3:]) == KNOWN_SINK_ROOTS
        expected_response = SplitCalculatorPublicResponseV2(
            seed_commitment=KNOWN_COMMITMENT,
            partitions=tuple(
                SplitRootCommitment(role_id, partition_id, row_count, root)
                for (role_id, partition_id, row_count), root in zip(
                    SPLIT_RESPONSE_ROWS,
                    (*KNOWN_ODD_ROOTS, *KNOWN_SINK_ROOTS),
                    strict=True,
                )
            ),
        )
        assert frame == encode_split_calculator_public_frame_v2(expected_response)
        # No raw seed, role key, per-row rank, or membership appears publicly.
        public_bytes = (
            response.seed_commitment,
            *(row.root for row in response.partitions),
        )
        assert KNOWN_SEED not in public_bytes
        frames.append(frame)
    assert frames[0] == frames[1]


def test_known_vector_is_anchored_to_the_frozen_project_reference() -> None:
    for role_id, rows, expected_counts, expected_roots in (
        (
            1,
            generate_odd_role_rows_v1(),
            (192, 96, 192),
            KNOWN_ODD_ROOTS,
        ),
        (
            2,
            generate_sink_role_rows_v1(),
            (39, 20, 26),
            KNOWN_SINK_ROOTS,
        ),
    ):
        role_key = derive_role_key(KNOWN_SEED, role_id)
        assignments = allocate_typed_role_rows(role_key, rows)
        commitments = split_partition_commitments(role_id, assignments)
        assert (
            commitments.discovery_count,
            commitments.validation_count,
            commitments.sealed_count,
        ) == expected_counts
        assert (
            commitments.discovery_root,
            commitments.validation_root,
            commitments.sealed_root,
        ) == expected_roots


@pytest.mark.parametrize("seed", [bytes(32), bytes([0xFF]) * 32])
def test_additional_seeds_are_deterministic_and_dual_identical(
    split_partition_rust_calculator: Path,
    seed: bytes,
) -> None:
    frames = []
    for command in _commands(split_partition_rust_calculator):
        completed, frame = _run_with_pipes(command, seed)
        assert completed.returncode == 0
        assert completed.stdout == completed.stderr == b""
        _decode_and_validate_frame(frame)
        frames.append(frame)
    assert frames[0] == frames[1]


@pytest.mark.parametrize("seed", [b"", b"x" * 31, b"x" * 33])
def test_wrong_seed_length_or_extra_byte_fails_silently(
    split_partition_rust_calculator: Path,
    seed: bytes,
) -> None:
    for command in _commands(split_partition_rust_calculator):
        completed, frame = _run_with_pipes(command, seed)
        assert completed.returncode != 0
        assert completed.stdout == completed.stderr == frame == b""


def test_missing_or_wrong_contract_fd_fails_silently(
    split_partition_rust_calculator: Path,
) -> None:
    for command in _commands(split_partition_rust_calculator):
        missing_seed, frame = _run_with_pipes(
            command, KNOWN_SEED, provide_seed_fd=False
        )
        assert missing_seed.returncode != 0
        assert missing_seed.stdout == missing_seed.stderr == frame == b""
        missing_response, frame = _run_with_pipes(
            command, KNOWN_SEED, provide_response_fd=False
        )
        assert missing_response.returncode != 0
        assert missing_response.stdout == missing_response.stderr == frame == b""


def test_seekable_seed_and_seekable_output_are_rejected(
    split_partition_rust_calculator: Path,
) -> None:
    for command in _commands(split_partition_rust_calculator):
        with tempfile.TemporaryFile() as seed_file:
            seed_file.write(KNOWN_SEED)
            seed_file.seek(0)
            response_read_fd, response_write_fd = os.pipe()
            process = subprocess.Popen(
                _fd_launcher_command(command, seed_file.fileno(), response_write_fd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(seed_file.fileno(), response_write_fd),
            )
            os.close(response_write_fd)
            frame = b"".join(iter(lambda: os.read(response_read_fd, 4096), b""))
            os.close(response_read_fd)
            stdout, stderr = process.communicate(timeout=20)
        assert process.returncode != 0
        assert stdout == stderr == frame == b""

        seed_read_fd, seed_write_fd = os.pipe()
        with tempfile.TemporaryFile() as response_file:
            process = subprocess.Popen(
                _fd_launcher_command(command, seed_read_fd, response_file.fileno()),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(seed_read_fd, response_file.fileno()),
            )
            os.close(seed_read_fd)
            os.write(seed_write_fd, KNOWN_SEED)
            os.close(seed_write_fd)
            stdout, stderr = process.communicate(timeout=20)
            response_file.seek(0)
            frame = response_file.read()
        assert process.returncode != 0
        assert stdout == stderr == frame == b""


def test_argv_stdin_and_environment_are_not_secret_fallbacks(
    split_partition_rust_calculator: Path,
) -> None:
    seed_hex = KNOWN_SEED.hex()
    for command in _commands(split_partition_rust_calculator):
        argv_result, frame = _run_with_pipes(
            command, KNOWN_SEED, extra_args=(seed_hex,)
        )
        assert argv_result.returncode != 0
        assert argv_result.stdout == argv_result.stderr == frame == b""
        stdin_result, frame = _run_with_pipes(
            command, b"", provide_seed_fd=False, stdin_payload=KNOWN_SEED
        )
        assert stdin_result.returncode != 0
        assert stdin_result.stdout == stdin_result.stderr == frame == b""
        env_result, frame = _run_with_pipes(
            command,
            b"",
            provide_seed_fd=False,
            extra_environment={"HEGEL_SPLIT_SEED": seed_hex},
        )
        assert env_result.returncode != 0
        assert env_result.stdout == env_result.stderr == frame == b""


def test_frame_and_public_schema_validation_fail_closed_on_faults(
    split_partition_rust_calculator: Path,
) -> None:
    command = _commands(split_partition_rust_calculator)[0]
    completed, frame = _run_with_pipes(command, KNOWN_SEED)
    assert completed.returncode == 0
    with pytest.raises(M25ContainerCeremonyError):
        _decode_and_validate_frame(frame[:7])
    with pytest.raises(M25ContainerCeremonyError):
        _decode_and_validate_frame(frame + b"\x00")
    with pytest.raises(M25ContainerCeremonyError):
        _decode_and_validate_frame((len(frame) + 1).to_bytes(8, "big") + frame[8:])

    decoded = list(canonical_cbor_decode(frame[8:]))
    partitions = list(decoded[3])
    first = list(partitions[0])
    first[2] = 191
    partitions[0] = tuple(first)
    decoded[3] = tuple(partitions)
    payload = canonical_cbor_encode(tuple(decoded))
    with pytest.raises(M25ContainerCeremonyError):
        _decode_and_validate_frame(len(payload).to_bytes(8, "big") + payload)


def test_sources_are_independent_and_attempt_secret_memory_hygiene() -> None:
    python_source = PYTHON_CALCULATOR.read_text(encoding="utf-8")
    rust_source = RUST_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(python_source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert "hegel_machine" not in imported_roots
    assert SPLIT_RESPONSE_SCHEMA_ID.decode("ascii") in python_source
    assert SPLIT_RESPONSE_SCHEMA_ID.decode("ascii") in rust_source
    assert "mlock" in python_source and "munlock" in python_source
    assert "_zeroize(seed)" in python_source
    assert "mlock" in rust_source and "munlock" in rust_source
    assert "zeroize(seed.as_mut_slice())" in rust_source
    assert "extern crate" not in rust_source


def test_both_independent_sources_freeze_exact_stratum_quotas() -> None:
    namespace = runpy.run_path(
        str(PYTHON_CALCULATOR),
        run_name="phase3_split_partition_calculator_fd3_v1_static_test",
    )
    assert namespace["ODD_QUOTAS"] == ODD_QUOTAS
    assert namespace["SINK_QUOTAS"] == SINK_QUOTAS

    rust_source = RUST_SOURCE.read_text(encoding="utf-8")
    rust_quotas = tuple(
        tuple(int(value) for value in match)
        for match in re.findall(
            r"Quota \{ stratum: (\d+), universe: (\d+), discovery: (\d+), "
            r"validation: (\d+), sealed: (\d+) \}",
            rust_source,
        )
    )
    assert rust_quotas == (*ODD_QUOTAS, *SINK_QUOTAS)
    assert tuple(sum(row[index] for row in ODD_QUOTAS) for index in range(1, 5)) == (
        480,
        192,
        96,
        192,
    )
    assert tuple(sum(row[index] for row in SINK_QUOTAS) for index in range(1, 5)) == (
        85,
        39,
        20,
        26,
    )


def test_docker_compilation_is_explicitly_offline_and_never_pulls() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    assert '"--pull=never"' in source
    assert '"--network=none"' in source
    assert PINNED_RUST_IMAGE in source
