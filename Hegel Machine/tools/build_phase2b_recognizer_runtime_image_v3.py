#!/usr/bin/env python3
"""Build and negatively qualify the frozen Phase-2B recognizer image.

This tool intentionally establishes only a local Docker image/build/config and
negative-smoke observation.  It does not consume an answer, a reveal token, or
an actual-960 recognizer bundle, and none of its outputs are scoring, gate, effect,
or C1 evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Callable, Final, Mapping, NoReturn, Sequence


ROOT: Final = Path(__file__).resolve().parents[1]
DOCKER_EXECUTABLE_V3: Final = "/usr/bin/docker"
DOCKER_HOST_V3: Final = "unix:///var/run/docker.sock"
DOCKERFILE_CONTEXT_PATH_V3: Final = (
    "docker/phase2b_recognizer_runtime_v3/Dockerfile"
)
DOCKERIGNORE_CONTEXT_PATH_V3: Final = (
    "docker/phase2b_recognizer_runtime_v3/Dockerfile.dockerignore"
)
PROFILE_PATH_V3: Final = "config/phase2b_c1_container_actor_profile_v2.json"
SECCOMP_PATH_V3: Final = "config/phase3_internal_actor_seccomp_v1.json"
BUILDX_SOURCE_PATH_V3: Final = Path(
    "/home/erzhu419/.local/state/hegel-machine/docker-cli-plugins/docker-buildx"
)

RECOGNIZER_RUNTIME_IMAGE_BUILD_RECEIPT_SCHEMA_V3: Final = (
    "hegel-machine-phase2b-recognizer-runtime-image-build-receipt/3"
)
RECOGNIZER_RUNTIME_IMAGE_BUILD_RECEIPT_ID_PREFIX_V3: Final = (
    "phase2b_recognizer_runtime_image_build_receipt_v3_"
)
RECOGNIZER_RUNTIME_IMAGE_BUILD_RECEIPT_DOMAIN_V3: Final = (
    b"HEGEL/PHASE2B/RECOGNIZER/RUNTIME_IMAGE_BUILD_RECEIPT/V3\x00"
)
RECOGNIZER_RUNTIME_IMAGE_BUILD_CLAIM_LEVEL_V3: Final = (
    "NON_AUTHORITATIVE_LOCAL_DOCKER_IMAGE_BUILD_AND_NEGATIVE_SMOKE_ONLY"
)
RECOGNIZER_RUNTIME_IMAGE_BUILD_DISPOSITION_V3: Final = (
    "LOCAL_RECOGNIZER_IMAGE_BUILD_AND_NEGATIVE_SMOKE_COMPLETE_NOT_ACTUAL_960"
)

BASE_IMAGE_DIGEST_V3: Final = (
    "sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"
)
PROFILE_CONTENT_ID_V3: Final = (
    "phase2b_c1_container_actor_profile_v2_"
    "b2f081771f3de8d2a531b08b58109d95c51815904e9612e66f1bbfe69babf906"
)
PROFILE_CANONICAL_SHA256_V3: Final = (
    "8ca1850f026dabb7a4b69add0f89b33a3fcfb789223b2fd95ef0379608adbeed"
)
OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V3: Final = (
    "phase2b_c1_owner_accepted_container_authority_requirement_v1_"
    "53631d563f8da8f26c7202b7dd0681be5423fe2ca5c15e9a9f5deaa2920d9940"
)
RECOGNIZER_SOURCE_ROOT_SHA256_V3: Final = (
    "d71f96926156b78bbd42d016b04c40aaa3e0abc5dad2510adc05c23e8dd54f54"
)
# The wire formula is shared with the scorer, but the role domain is distinct.
RECOGNIZER_SOURCE_ROOT_DOMAIN_V3: Final = (
    b"HEGEL/PHASE2B/RECOGNIZER_RUNTIME_SOURCE_ROOT/V1\x00"
)
DOCKERFILE_SHA256_V3: Final = "519877228fe845d32afc0b30f1af22962b1c54b965b0c522b2930fc69de26208"
DOCKERIGNORE_SHA256_V3: Final = "5157729f6437bcf53e62efeaffc3372fdccde5debab9c1edff54584a05a5b3ee"
PROFILE_FILE_SHA256_V3: Final = (
    "06c565dc526f5987fdc6abac8fb76dcb9dae167a935c40997faafe8fbb2ff473"
)
SECCOMP_SHA256_V3: Final = (
    "d1284e4731683b73352ecdd1577704ea87aa0b5c582b7b00757c3db4d2c950ca"
)
INITIALIZER_SHA256_V3: Final = (
    "54232bdb2417e659fba2198fe83e3c78b206baa4038a41ab12c1f37fb395910d"
)
ENTRYPOINT_SHA256_V3: Final = (
    "5ce3f260627ff99005fe4693c6328aa3a1eb8992133833f08b21d75ffb59036d"
)
BUILDX_SHA256_V3: Final = (
    "2d836b76601cb911a1751b12a7a94a4f64e10fb0bb8f3839d377311a30f0d259"
)
BUILDX_SIZE_BYTES_V3: Final = 70_944_600
BUILDX_VERSION_STDOUT_V3: Final = (
    b"github.com/docker/buildx 0.30.1 0.30.1-0ubuntu1~22.04.1\n"
)
BUILDX_PRIVATE_RELATIVE_PATH_V3: Final = "cli-plugins/docker-buildx"

IMAGE_TAG_V3: Final = (
    "hegel-machine-phase2b-recognizer-runtime-v3:source-"
    + RECOGNIZER_SOURCE_ROOT_SHA256_V3[:16]
)
RECOGNIZER_RUNTIME_UID_GID_V3: Final = "65534:65534"
RECOGNIZER_RUNTIME_IMAGE_TITLE_V3: Final = (
    "Hegel Machine Phase2B V3 recognizer runtime"
)
RECOGNIZER_RUNTIME_IMAGE_DESCRIPTION_V3: Final = (
    "Non-authoritative local Docker recognizer mechanics; no actual-960 "
    "execution, scoring, gate, or C1 evidence"
)
FROZEN_ENTRYPOINT_V3: Final = ("/opt/hegel/bin/phase2b-recognizer",)
FROZEN_COMMAND_V3: Final = (
    "--input",
    "/phase2b/input",
    "--output",
    "/phase2b/output",
)
FROZEN_IMAGE_ARGS_ESCAPED_V3: Final = True
FROZEN_IMAGE_CONFIG_FIELDS_V3: Final = frozenset(
    {"ArgsEscaped", "Cmd", "Entrypoint", "Env", "Labels", "User", "WorkingDir"}
)
FROZEN_IMAGE_ENVIRONMENT_V3: Final = (
    "PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LANG=C.UTF-8",
    "GPG_KEY=A035C8C19219BA821ECEA86B64E628F8D684696D",
    "PYTHON_VERSION=3.11.15",
    "PYTHON_SHA256=272179ddd9a2e41a0fc8e42e33dfbdca0b3711aa5abf372d3f2d51543d09b625",
    "HOME=/nonexistent",
    "PYTHONHASHSEED=0",
    "PYTHONDONTWRITEBYTECODE=1",
    "PYTHONPATH=/opt/hegel/src",
)
FROZEN_IMAGE_LABELS_V3: Final = {
    "org.opencontainers.image.base.digest": BASE_IMAGE_DIGEST_V3,
    "org.opencontainers.image.description": RECOGNIZER_RUNTIME_IMAGE_DESCRIPTION_V3,
    "org.opencontainers.image.title": RECOGNIZER_RUNTIME_IMAGE_TITLE_V3,
    "org.hegel-machine.phase2b.actor-profile.canonical-sha256": PROFILE_CANONICAL_SHA256_V3,
    "org.hegel-machine.phase2b.actor-profile.content-id": PROFILE_CONTENT_ID_V3,
    "org.hegel-machine.phase2b.owner-accepted-authority-requirement.id": (
        OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V3
    ),
    "org.hegel-machine.phase2b.recognizer-source-root.sha256": (
        RECOGNIZER_SOURCE_ROOT_SHA256_V3
    ),
}
# Exact generic fail-closed line emitted by runtime_cli_v2 for missing input.
NEGATIVE_SMOKE_STDERR_V3: Final = (
    b'{"disposition":"ABSTAIN","reason":'
    b'"phase2b_recognizer_runtime_v2_failed"}\n'
)

MAXIMUM_SOURCE_FILE_BYTES_V3: Final = 16 * 1024 * 1024
MAXIMUM_BUILDX_BYTES_V3: Final = 128 * 1024 * 1024
MAXIMUM_CONTEXT_BYTES_V3: Final = 8 * 1024 * 1024
MAXIMUM_BUILD_STDOUT_BYTES_V3: Final = 4 * 1024 * 1024
MAXIMUM_BUILD_STDERR_BYTES_V3: Final = 4 * 1024 * 1024
MAXIMUM_INSPECT_STDOUT_BYTES_V3: Final = 1024 * 1024
MAXIMUM_DOCKER_STDERR_BYTES_V3: Final = 64 * 1024
MAXIMUM_SMOKE_STDOUT_BYTES_V3: Final = 64 * 1024
BUILD_TIMEOUT_SECONDS_V3: Final = 900
INSPECT_TIMEOUT_SECONDS_V3: Final = 60
SMOKE_TIMEOUT_SECONDS_V3: Final = 60
CLEANUP_TIMEOUT_SECONDS_V3: Final = 30
_DIGEST_V3: Final = re.compile(r"sha256:[0-9a-f]{64}")
_CONTAINER_ID_V3: Final = re.compile(r"[0-9a-f]{64}")


# Ordered by exact installed absolute POSIX path.  These are the only 28
# payloads admitted to the recognizer runtime source-root wire.
FROZEN_SOURCE_MANIFEST_V3: Final = (
    ('/opt/hegel/bin/phase2b-recognizer', 'docker/phase2b_recognizer_runtime_v3/phase2b-recognizer', '5ce3f260627ff99005fe4693c6328aa3a1eb8992133833f08b21d75ffb59036d'),
    ('/opt/hegel/src/hegel_machine/__init__.py', 'docker/phase2b_recognizer_runtime_v3/hegel_machine_init.py', '54232bdb2417e659fba2198fe83e3c78b206baa4038a41ab12c1f37fb395910d'),
    ('/opt/hegel/src/hegel_machine/bootstrap.py', 'src/hegel_machine/bootstrap.py', 'aecb9711f0c2238e22140b6ed65ba2533747b590bf8d16f70f65f964fd3f3b4b'),
    ('/opt/hegel/src/hegel_machine/hashing.py', 'src/hegel_machine/hashing.py', '6fcd988ca4e8b75b9aa9e5a8b38c9ddd73e2685140e2cd0095d68f8bda054159'),
    ('/opt/hegel/src/hegel_machine/laws.py', 'src/hegel_machine/laws.py', '2bb3e96544cfc9c52718efdc2880fb33607bdc597bf47c30b640ee07cb654fc2'),
    ('/opt/hegel/src/hegel_machine/milestones.py', 'src/hegel_machine/milestones.py', '049e4ec78798d8798dac8eb19c97921e03b942c7adb46287c62a16e4a98ab276'),
    ('/opt/hegel/src/hegel_machine/ontology.py', 'src/hegel_machine/ontology.py', '19828518e4c3cc6a4c2171766fc576a624e836613f968cfd09589a61b78e03ab'),
    ('/opt/hegel/src/hegel_machine/phase2b_adapter.py', 'src/hegel_machine/phase2b_adapter.py', '3f6d0978d3ca09013e9257da3ca5312eb05adb89dd3525329a5e850e14b3c8bb'),
    ('/opt/hegel/src/hegel_machine/phase2b_exact_bridge_v1.py', 'src/hegel_machine/phase2b_exact_bridge_v1.py', 'bf8637ab3f43ba670d64659568022031df4623182c4d7e6fc546827000b14d6e'),
    ('/opt/hegel/src/hegel_machine/phase2b_exact_derived_witness_bridge_v1.py', 'src/hegel_machine/phase2b_exact_derived_witness_bridge_v1.py', '56fd5b1f1730bc7b870f28fe6908f88a3501bd4ee98b0c4c5622c8f78ed18d93'),
    ('/opt/hegel/src/hegel_machine/phase2b_exact_transform_semantics_v1.py', 'src/hegel_machine/phase2b_exact_transform_semantics_v1.py', '353b10a7464da54553cf84e738bd53c010274ad1b15e9c36729420ea11f8e55c'),
    ('/opt/hegel/src/hegel_machine/phase2b_freeze_v1.py', 'src/hegel_machine/phase2b_freeze_v1.py', '39a289ef0c000c1c62eb3a7a5f97ddfae3e51eb7904757642b0c2d167067f543'),
    ('/opt/hegel/src/hegel_machine/phase2b_protocol.py', 'src/hegel_machine/phase2b_protocol.py', 'aeccff3558e82ebf261cbeb1c389e968ff22c8ae229b9ce201808accf76074ef'),
    ('/opt/hegel/src/hegel_machine/phase2b_recognizer_input_archive_v2.py', 'src/hegel_machine/phase2b_recognizer_input_archive_v2.py', 'a50473bb942de6dea823cd6523c096a9d33c07a8395efa03cc6d7c8b4b035108'),
    ('/opt/hegel/src/hegel_machine/phase2b_recognizer_prediction_archive_v2.py', 'src/hegel_machine/phase2b_recognizer_prediction_archive_v2.py', 'ecfe3ff599f10deae845a32cb979209d0591727bd36ef98583194412d74cb04a'),
    ('/opt/hegel/src/hegel_machine/phase2b_recognizer_prediction_v2.py', 'src/hegel_machine/phase2b_recognizer_prediction_v2.py', 'e32133bfe7d8848b56fcbcf6f68849908b7d9a203a6e6196aae6d954e011c023'),
    ('/opt/hegel/src/hegel_machine/phase2b_recognizer_runtime_cli_v2.py', 'src/hegel_machine/phase2b_recognizer_runtime_cli_v2.py', 'b219489a782bcdcdd7fdca2d8a900afac8c3bb38a3bf8b21276ce29f5d88ec19'),
    ('/opt/hegel/src/hegel_machine/phase2b_runner.py', 'src/hegel_machine/phase2b_runner.py', '24ae136477745356d17aac2906ed46ef41f16f095875643c0c0c7d8d1e11b5ae'),
    ('/opt/hegel/src/hegel_machine/phase2b_trusted_wire_batch_v1.py', 'src/hegel_machine/phase2b_trusted_wire_batch_v1.py', '0b2606bd7ba4e41253beaadb9fee5e5d3a301382dac5a6025f51756e5cd1eeaa'),
    ('/opt/hegel/src/hegel_machine/phase2b_trusted_wire_batch_v2.py', 'src/hegel_machine/phase2b_trusted_wire_batch_v2.py', '42e4fba7ff3829f57eb60b4f8d1ffe87099deed8b751cc2b3101bcfa02f703b5'),
    ('/opt/hegel/src/hegel_machine/phase2b_trusted_wire_typed_authority_v1.py', 'src/hegel_machine/phase2b_trusted_wire_typed_authority_v1.py', 'd6e39f81257026281fd81d191fcc6e37be8ff130a6bde6ac0d8e417cb6c11f63'),
    ('/opt/hegel/src/hegel_machine/phase2b_trusted_wire_typed_authority_v2.py', 'src/hegel_machine/phase2b_trusted_wire_typed_authority_v2.py', '6d6cc72547a8f00402e126b4a339272e6de0353525e7627c082ef62faa407521'),
    ('/opt/hegel/src/hegel_machine/phase2b_trusted_wire_typed_replay_v2.py', 'src/hegel_machine/phase2b_trusted_wire_typed_replay_v2.py', '0608113116bf0f548acd17d509a7ca0e13889d518260e1b9f4a3aaf8ab942309'),
    ('/opt/hegel/src/hegel_machine/phase2b_trusted_wire_v1.py', 'src/hegel_machine/phase2b_trusted_wire_v1.py', '98defeb8455944144e318879688174488e1c13aab2153c019a6db4b67946b5c2'),
    ('/opt/hegel/src/hegel_machine/phase2b_uncertainty_compiler.py', 'src/hegel_machine/phase2b_uncertainty_compiler.py', '529c3d8e7dc2baec20c76ec5f9f21c30c31c1842305872499ba95e6dbb584d35'),
    ('/opt/hegel/src/hegel_machine/phase2b_wire.py', 'src/hegel_machine/phase2b_wire.py', '0330d205b6bdfd9452063e636873e99cb8b8d902decd4124e52207061294d9db'),
    ('/opt/hegel/src/hegel_machine/phase3_dsl_v1.py', 'src/hegel_machine/phase3_dsl_v1.py', 'c0e3ef208d08c7415dd934f8066f6c845a943272e1b9033db3b2a4083694ae27'),
    ('/opt/hegel/src/hegel_machine/schema.py', 'src/hegel_machine/schema.py', '6497a93f8a68f3313d294bc99dc3b8f43508e93772d748521d09ea716b8ae10a'),
)



class Phase2BRecognizerImageBuilderErrorV3(RuntimeError):
    """One stable fail-closed rejection from the local image builder."""


def _fail(detail: str) -> NoReturn:
    raise Phase2BRecognizerImageBuilderErrorV3(detail)


@dataclass(frozen=True)
class BoundedDockerResultV3:
    command: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True)
class FrozenBuildInputsV3:
    context_bytes: bytes
    context_sha256: str
    context_entries: tuple[tuple[str, int, int, str], ...]
    source_entries: tuple[tuple[str, str, int, str], ...]
    source_root_sha256: str
    dockerfile_sha256: str
    dockerignore_sha256: str
    profile_sha256: str
    seccomp_sha256: str
    seccomp_bytes: bytes
    buildx_sha256: str
    buildx_size_bytes: int
    buildx_bytes: bytes


DockerRunnerV3 = Callable[..., BoundedDockerResultV3]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_frozen_file_v3(
    relative_path: str,
    expected_sha256: str,
    *,
    expected_mode: int,
) -> bytes:
    if (
        type(relative_path) is not str
        or not relative_path
        or relative_path.startswith("/")
        or any(part in {"", ".", ".."} for part in relative_path.split("/"))
    ):
        _fail("frozen source path is malformed")
    path = ROOT / relative_path
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if type(nofollow) is not int or nofollow == 0:
        _fail("O_NOFOLLOW is unavailable")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags | nofollow)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != expected_mode
            or not 0 <= before.st_size <= MAXIMUM_SOURCE_FILE_BYTES_V3
        ):
            _fail(f"frozen source metadata differs: {relative_path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                _fail(f"frozen source ended early: {relative_path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1) != b"":
            _fail(f"frozen source grew during read: {relative_path}")
        after = os.fstat(descriptor)
        payload = b"".join(chunks)
        fingerprint = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if fingerprint(before) != fingerprint(after):
            _fail(f"frozen source changed during read: {relative_path}")
    except Phase2BRecognizerImageBuilderErrorV3:
        raise
    except OSError as exc:
        _fail(f"frozen source cannot be opened: {relative_path}: {type(exc).__name__}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if _sha256(payload) != expected_sha256:
        _fail(f"frozen source digest differs: {relative_path}")
    return payload


def _read_stable_absolute_regular_v3(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    expected_mode: int,
    expected_uid: int,
    expected_gid: int,
    maximum_bytes: int,
) -> bytes:
    """Read one absolute regular file through a no-follow parent dirfd."""

    if (
        type(path) is not type(Path("/"))
        or not path.is_absolute()
        or Path(os.path.abspath(os.fspath(path))) != path
        or type(expected_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or type(expected_size) is not int
        or not 1 <= expected_size <= maximum_bytes
        or expected_mode not in {0o600, 0o755}
        or type(expected_uid) is not int
        or expected_uid < 0
        or type(expected_gid) is not int
        or expected_gid < 0
    ):
        _fail("stable absolute file contract differs")
    parent_fd = _open_directory_componentwise_v3(path.parent)
    descriptor: int | None = None
    try:
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if type(nofollow) is not int or nofollow == 0:
            _fail("O_NOFOLLOW is unavailable")
        descriptor = os.open(
            path.name,
            os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != expected_mode
            or before.st_uid != expected_uid
            or before.st_gid != expected_gid
            or before.st_size != expected_size
        ):
            _fail("stable absolute file metadata differs")
        chunks: list[bytes] = []
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                _fail("stable absolute file ended early")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1) != b"":
            _fail("stable absolute file grew during read")
        after = os.fstat(descriptor)
        fingerprint = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_uid,
            value.st_gid,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if fingerprint(before) != fingerprint(after):
            _fail("stable absolute file changed during read")
        payload = b"".join(chunks)
    except Phase2BRecognizerImageBuilderErrorV3:
        raise
    except OSError as exc:
        _fail(f"stable absolute file read failed: {type(exc).__name__}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
    if _sha256(payload) != expected_sha256:
        _fail("stable absolute file digest differs")
    return payload


def _compute_source_root_v3(
    ordered: Sequence[tuple[str, bytes]],
) -> str:
    if (
        type(ordered) not in {list, tuple}
        or len(ordered) != 28
        or tuple(path for path, _payload in ordered)
        != tuple(sorted(path for path, _payload in ordered))
    ):
        _fail("source-root ordered surface differs")
    preimage = bytearray(RECOGNIZER_SOURCE_ROOT_DOMAIN_V3)
    preimage.extend(struct.pack(">I", len(ordered)))
    for installed_path, payload in ordered:
        try:
            encoded = installed_path.encode("ascii")
        except UnicodeEncodeError:
            _fail("installed source-root path is not ASCII")
        preimage.extend(struct.pack(">I", len(encoded)))
        preimage.extend(encoded)
        preimage.extend(struct.pack(">Q", len(payload)))
        preimage.extend(hashlib.sha256(payload).digest())
    return _sha256(bytes(preimage))


def _deterministic_ustar_v3(
    entries: Mapping[str, tuple[bytes, int]],
) -> bytes:
    if type(entries) is not dict or len(entries) != 30:
        _fail("deterministic context entry count differs")
    names = tuple(sorted(entries))
    if any(
        type(name) is not str
        or not name
        or name.startswith("/")
        or any(part in {"", ".", ".."} for part in name.split("/"))
        for name in names
    ):
        _fail("deterministic context path surface differs")
    buffer = io.BytesIO()
    try:
        with tarfile.open(
            fileobj=buffer,
            mode="w",
            format=tarfile.USTAR_FORMAT,
            dereference=False,
        ) as archive:
            for name in names:
                payload, mode = entries[name]
                if type(payload) is not bytes or mode not in {0o644, 0o755}:
                    _fail("deterministic context payload metadata differs")
                info = tarfile.TarInfo(name=name)
                info.size = len(payload)
                info.mode = mode
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                info.type = tarfile.REGTYPE
                archive.addfile(info, io.BytesIO(payload))
    except (OSError, tarfile.TarError, ValueError) as exc:
        _fail(f"deterministic ustar construction failed: {type(exc).__name__}")
    payload = buffer.getvalue()
    if not payload or len(payload) > MAXIMUM_CONTEXT_BYTES_V3:
        _fail("deterministic ustar context exceeds its frozen bound")
    return payload


def load_frozen_build_inputs_v3() -> FrozenBuildInputsV3:
    """Snapshot and verify every byte before any Docker process is started."""

    if (
        len(FROZEN_SOURCE_MANIFEST_V3) != 28
        or tuple(row[0] for row in FROZEN_SOURCE_MANIFEST_V3)
        != tuple(sorted(row[0] for row in FROZEN_SOURCE_MANIFEST_V3))
        or len({row[0] for row in FROZEN_SOURCE_MANIFEST_V3}) != 28
        or len({row[1] for row in FROZEN_SOURCE_MANIFEST_V3}) != 28
    ):
        _fail("frozen 28-entry source manifest differs")

    source_payloads: list[tuple[str, bytes]] = []
    source_entries: list[tuple[str, str, int, str]] = []
    context: dict[str, tuple[bytes, int]] = {}
    for installed_path, context_path, expected_sha256 in FROZEN_SOURCE_MANIFEST_V3:
        mode = 0o755 if installed_path == FROZEN_ENTRYPOINT_V3[0] else 0o644
        payload = _read_frozen_file_v3(
            context_path,
            expected_sha256,
            expected_mode=mode,
        )
        source_payloads.append((installed_path, payload))
        source_entries.append(
            (installed_path, context_path, len(payload), expected_sha256)
        )
        context[context_path] = (payload, mode)

    source_root = _compute_source_root_v3(source_payloads)
    if source_root != RECOGNIZER_SOURCE_ROOT_SHA256_V3:
        _fail("frozen 28-entry source root differs")

    dockerfile = _read_frozen_file_v3(
        DOCKERFILE_CONTEXT_PATH_V3,
        DOCKERFILE_SHA256_V3,
        expected_mode=0o644,
    )
    dockerignore = _read_frozen_file_v3(
        DOCKERIGNORE_CONTEXT_PATH_V3,
        DOCKERIGNORE_SHA256_V3,
        expected_mode=0o644,
    )
    expected_from = ("FROM python@" + BASE_IMAGE_DIGEST_V3 + "\n").encode("ascii")
    folded_dockerfile = dockerfile.lower()
    if (
        not dockerfile.startswith(expected_from)
        or b"\nrun " in folded_dockerfile
        or any(
            token in folded_dockerfile
            for token in (
                b"apt-get",
                b"pip install",
                b"curl ",
                b"wget ",
                b"git clone",
            )
        )
    ):
        _fail("frozen Dockerfile is not the pinned no-RUN offline surface")
    profile = _read_frozen_file_v3(
        PROFILE_PATH_V3,
        PROFILE_FILE_SHA256_V3,
        expected_mode=0o644,
    )
    seccomp = _read_frozen_file_v3(
        SECCOMP_PATH_V3,
        SECCOMP_SHA256_V3,
        expected_mode=0o644,
    )
    buildx = _read_stable_absolute_regular_v3(
        BUILDX_SOURCE_PATH_V3,
        expected_sha256=BUILDX_SHA256_V3,
        expected_size=BUILDX_SIZE_BYTES_V3,
        expected_mode=0o755,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
        maximum_bytes=MAXIMUM_BUILDX_BYTES_V3,
    )
    try:
        profile_value = json.loads(profile)
        seccomp_value = json.loads(seccomp)
        supersession = (
            profile_value.get("supersession_policy")
            if type(profile_value) is dict
            else None
        )
        if (
            type(profile_value) is not dict
            or type(seccomp_value) is not dict
            or profile_value.get("schema_version")
            != "hegel-phase2b-c1-container-actor-profile/2"
            or profile_value.get("profile_id")
            != "hegel-owner-accepted-phase2b-c1-container-technical-actors-v2"
            or type(supersession) is not dict
            or supersession.get("owner_accepted_authority_requirement_reused_exactly")
            is not True
            or supersession.get("new_attempts_must_bind_v2_profile_content_id")
            is not True
        ):
            _fail("profile or seccomp identity differs")
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("profile or seccomp JSON is malformed")

    context[DOCKERFILE_CONTEXT_PATH_V3] = (dockerfile, 0o644)
    context[DOCKERIGNORE_CONTEXT_PATH_V3] = (dockerignore, 0o644)
    archive = _deterministic_ustar_v3(context)
    context_entries = tuple(
        (path, len(payload), mode, _sha256(payload))
        for path, (payload, mode) in sorted(context.items())
    )
    return FrozenBuildInputsV3(
        context_bytes=archive,
        context_sha256=_sha256(archive),
        context_entries=context_entries,
        source_entries=tuple(source_entries),
        source_root_sha256=source_root,
        dockerfile_sha256=_sha256(dockerfile),
        dockerignore_sha256=_sha256(dockerignore),
        profile_sha256=_sha256(profile),
        seccomp_sha256=_sha256(seccomp),
        seccomp_bytes=seccomp,
        buildx_sha256=_sha256(buildx),
        buildx_size_bytes=len(buildx),
        buildx_bytes=buildx,
    )


def _kill_and_reap_v3(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                process.kill()
            except OSError:
                pass
    try:
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _bounded_docker_run_v3(
    command: Sequence[str],
    *,
    stdin_payload: bytes | None,
    environment: Mapping[str, str],
    working_directory: Path,
    timeout_seconds: int,
    maximum_stdout_bytes: int,
    maximum_stderr_bytes: int,
) -> BoundedDockerResultV3:
    """Run one exact Docker command with bounded input, output, and deadline."""

    command_tuple = tuple(command)
    if (
        not command_tuple
        or command_tuple[:2] != (DOCKER_EXECUTABLE_V3, f"--host={DOCKER_HOST_V3}")
        or any(type(item) is not str or not item or "\x00" in item for item in command_tuple)
        or (stdin_payload is not None and type(stdin_payload) is not bytes)
        or type(environment) is not dict
        or not working_directory.is_absolute()
        or type(timeout_seconds) is not int
        or timeout_seconds < 1
        or type(maximum_stdout_bytes) is not int
        or maximum_stdout_bytes < 1
        or type(maximum_stderr_bytes) is not int
        or maximum_stderr_bytes < 1
    ):
        _fail("bounded Docker invocation surface differs")

    process: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    stdout = bytearray()
    stderr = bytearray()
    deadline = time.monotonic() + timeout_seconds
    input_view = memoryview(stdin_payload) if stdin_payload is not None else None
    input_offset = 0
    try:
        process = subprocess.Popen(
            list(command_tuple),
            cwd=working_directory.as_posix(),
            stdin=subprocess.PIPE if stdin_payload is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(environment),
            close_fds=True,
            start_new_session=True,
            bufsize=0,
        )
        if process.stdout is None or process.stderr is None:
            _fail("Docker process capture pipes are unavailable")
        streams: dict[int, tuple[str, object, bytearray, int]] = {}
        for name, stream, target, cap in (
            ("stdout", process.stdout, stdout, maximum_stdout_bytes),
            ("stderr", process.stderr, stderr, maximum_stderr_bytes),
        ):
            descriptor = stream.fileno()
            os.set_blocking(descriptor, False)
            selector.register(stream, selectors.EVENT_READ, data=(name, descriptor))
            streams[descriptor] = (name, stream, target, cap)
        if process.stdin is not None:
            os.set_blocking(process.stdin.fileno(), False)
            selector.register(
                process.stdin,
                selectors.EVENT_WRITE,
                data=("stdin", process.stdin.fileno()),
            )

        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_and_reap_v3(process)
                _fail("Docker command exceeded its fixed deadline")
            events = selector.select(timeout=min(remaining, 0.25))
            if not events and process.poll() is not None:
                events = tuple((key, key.events) for key in selector.get_map().values())
            for key, _mask in events:
                name, descriptor = key.data
                if name == "stdin":
                    if process.stdin is None:
                        _fail("Docker stdin pipe disappeared")
                    try:
                        assert input_view is not None
                        written = os.write(
                            descriptor,
                            input_view[input_offset : input_offset + 65_536],
                        )
                        input_offset += written
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        input_offset = len(input_view) if input_view is not None else 0
                    except OSError as exc:
                        _kill_and_reap_v3(process)
                        _fail(f"Docker stdin write failed: {type(exc).__name__}")
                    if input_view is None or input_offset >= len(input_view):
                        selector.unregister(process.stdin)
                        process.stdin.close()
                    continue
                row = streams.get(descriptor)
                if row is None:
                    continue
                _stream_name, stream, target, cap = row
                try:
                    chunk = os.read(descriptor, 65_536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    streams.pop(descriptor, None)
                    continue
                if len(target) + len(chunk) > cap:
                    _kill_and_reap_v3(process)
                    _fail(f"Docker {name} exceeded its fixed byte cap")
                target.extend(chunk)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_and_reap_v3(process)
            _fail("Docker command exceeded its fixed deadline")
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _kill_and_reap_v3(process)
            _fail("Docker command exceeded its fixed deadline")
        return BoundedDockerResultV3(
            command=command_tuple,
            returncode=returncode,
            stdout=bytes(stdout),
            stderr=bytes(stderr),
        )
    except Phase2BRecognizerImageBuilderErrorV3:
        if process is not None:
            _kill_and_reap_v3(process)
        raise
    except (OSError, ValueError) as exc:
        if process is not None:
            _kill_and_reap_v3(process)
        _fail(f"Docker process could not be executed: {type(exc).__name__}")
    finally:
        selector.close()


def _invoke_v3(
    runner: DockerRunnerV3,
    command: tuple[str, ...],
    *,
    stdin_payload: bytes | None,
    environment: dict[str, str],
    timeout_seconds: int,
    maximum_stdout_bytes: int,
    maximum_stderr_bytes: int,
) -> BoundedDockerResultV3:
    result = runner(
        command,
        stdin_payload=stdin_payload,
        environment=environment,
        working_directory=ROOT,
        timeout_seconds=timeout_seconds,
        maximum_stdout_bytes=maximum_stdout_bytes,
        maximum_stderr_bytes=maximum_stderr_bytes,
    )
    if type(result) is not BoundedDockerResultV3 or result.command != command:
        _fail("Docker runner returned a malformed observation")
    return result


def _canonical_json_v3(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        _fail(f"canonical JSON encoding failed: {type(exc).__name__}")


def _decode_json_v3(payload: bytes, *, label: str) -> object:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail(f"{label} is not one JSON value")


def _validate_image_inspect_v3(payload: bytes) -> tuple[str, str, tuple[str, ...]]:
    value = _decode_json_v3(payload, label="recognizer image inspection")
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        _fail("recognizer image inspection shape differs")
    row = value[0]
    config = row.get("Config")
    rootfs = row.get("RootFS")
    image_id = row.get("Id")
    if (
        type(config) is not dict
        or type(rootfs) is not dict
        or type(image_id) is not str
        or _DIGEST_V3.fullmatch(image_id) is None
        or row.get("Os") != "linux"
        or row.get("Architecture") != "amd64"
        or frozenset(config) != FROZEN_IMAGE_CONFIG_FIELDS_V3
        or config.get("ArgsEscaped") is not FROZEN_IMAGE_ARGS_ESCAPED_V3
        or config.get("User") != RECOGNIZER_RUNTIME_UID_GID_V3
        or config.get("Env") != list(FROZEN_IMAGE_ENVIRONMENT_V3)
        or config.get("Entrypoint") != list(FROZEN_ENTRYPOINT_V3)
        or config.get("Cmd") != list(FROZEN_COMMAND_V3)
        or config.get("WorkingDir") != "/opt/hegel"
        or config.get("Labels") != FROZEN_IMAGE_LABELS_V3
        or config.get("Healthcheck") is not None
        or config.get("Volumes") is not None
        or rootfs.get("Type") != "layers"
        or type(rootfs.get("Layers")) is not list
        or not rootfs["Layers"]
        or any(
            type(layer) is not str or _DIGEST_V3.fullmatch(layer) is None
            for layer in rootfs["Layers"]
        )
    ):
        _fail("recognizer image platform/config/source binding differs")
    projection = {
        "Architecture": row["Architecture"],
        "Config": {
            "ArgsEscaped": config["ArgsEscaped"],
            "Cmd": config["Cmd"],
            "Entrypoint": config["Entrypoint"],
            "Env": config["Env"],
            "Healthcheck": None,
            "Labels": config["Labels"],
            "User": config["User"],
            "Volumes": None,
            "WorkingDir": config["WorkingDir"],
        },
        "Id": image_id,
        "Os": row["Os"],
        "RootFS": rootfs,
    }
    return image_id, _sha256(_canonical_json_v3(projection)), tuple(rootfs["Layers"])


def _validate_smoke_container_v3(
    payload: bytes,
    *,
    image_digest: str,
    container_name: str,
    expected_seccomp: Mapping[str, object],
) -> None:
    value = _decode_json_v3(payload, label="negative smoke container inspection")
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        _fail("negative smoke container inspection shape differs")
    row = value[0]
    state = row.get("State")
    config = row.get("Config")
    host = row.get("HostConfig")
    network = row.get("NetworkSettings")
    expected_tmpfs = {
        "/phase2b/input": (
            "ro,noexec,nosuid,nodev,mode=0500,uid=65534,gid=65534,size=1048576"
        ),
        "/phase2b/output": (
            "rw,noexec,nosuid,nodev,mode=0700,uid=65534,gid=65534,size=1048576"
        ),
        "/tmp": (
            "rw,noexec,nosuid,nodev,mode=0700,uid=65534,gid=65534,size=1048576"
        ),
    }
    security_options = host.get("SecurityOpt") if type(host) is dict else None
    if type(security_options) is not list or len(security_options) != 2:
        _fail("negative smoke container security options differ")
    seccomp_options = tuple(
        item
        for item in security_options
        if type(item) is str and item.startswith("seccomp=")
    )
    if len(seccomp_options) != 1:
        _fail("negative smoke embedded seccomp option differs")
    observed_seccomp = _decode_json_v3(
        seccomp_options[0][len("seccomp=") :].encode("utf-8"),
        label="negative smoke embedded seccomp",
    )
    if (
        not all(type(item) is dict for item in (state, config, host, network))
        or type(row.get("Id")) is not str
        or _CONTAINER_ID_V3.fullmatch(row["Id"]) is None
        or row.get("Name") != "/" + container_name
        or row.get("Image") != image_digest
        or row.get("Path") != FROZEN_ENTRYPOINT_V3[0]
        or row.get("Args") != list(FROZEN_COMMAND_V3)
        or state.get("Status") != "exited"
        or state.get("Running") is not False
        or state.get("Dead") is not False
        or state.get("OOMKilled") is not False
        or state.get("ExitCode") != 2
        or state.get("Error") != ""
        or config.get("Image") != image_digest
        or config.get("User") != RECOGNIZER_RUNTIME_UID_GID_V3
        or config.get("Env") != list(FROZEN_IMAGE_ENVIRONMENT_V3)
        or config.get("Entrypoint") != list(FROZEN_ENTRYPOINT_V3)
        or config.get("Cmd") != list(FROZEN_COMMAND_V3)
        or config.get("WorkingDir") != "/opt/hegel"
        or config.get("Volumes") is not None
        or host.get("NetworkMode") != "none"
        or host.get("ReadonlyRootfs") is not True
        or host.get("Privileged") is not False
        or host.get("AutoRemove") is not False
        or host.get("CapDrop") != ["ALL"]
        or "no-new-privileges:true" not in security_options
        or observed_seccomp != dict(expected_seccomp)
        or host.get("IpcMode") != "none"
        or host.get("PidsLimit") != 64
        or host.get("NanoCpus") != 1_000_000_000
        or host.get("Memory") != 536_870_912
        or host.get("MemorySwap") != 536_870_912
        or host.get("CgroupnsMode") != "private"
        or host.get("PidMode") != ""
        or host.get("UsernsMode") != ""
        or host.get("Runtime") != "runc"
        or host.get("Tmpfs") != expected_tmpfs
        or host.get("Binds") is not None
        or row.get("Mounts") not in (None, [])
        or network.get("Ports") not in (None, {})
    ):
        _fail("negative smoke container security/state differs")


def _validate_cleanup_proof_v3(
    removed: BoundedDockerResultV3,
    inspected: BoundedDockerResultV3,
    *,
    container_name: str,
    allow_already_absent: bool,
) -> None:
    if type(allow_already_absent) is not bool:
        _fail("negative smoke cleanup absence policy differs")
    absent_stderr = (
        f"Error response from daemon: No such container: {container_name}\n"
    ).encode("ascii")
    removed_existing = (
        removed.returncode == 0
        and removed.stdout == (container_name + "\n").encode("ascii")
        and removed.stderr == b""
    )
    removed_absent = (
        removed.returncode == 0
        and removed.stdout == b""
        and removed.stderr == absent_stderr
    )
    inspect_proves_absent = (
        inspected.returncode == 1
        and inspected.stdout == b"[]\n"
        and inspected.stderr == absent_stderr
    )
    if (
        not inspect_proves_absent
        or not (removed_existing or (allow_already_absent and removed_absent))
    ):
        _fail("negative smoke container removal/absence proof differs")


def _write_private_file_v3(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o600,
) -> None:
    if type(payload) is not bytes or mode not in {0o600, 0o755}:
        _fail("private runtime file contract differs")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if type(nofollow) is not int or nofollow == 0:
        _fail("O_NOFOLLOW is unavailable")
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags | nofollow, mode)
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                _fail("private runtime file write made no progress")
            offset += written
        os.fsync(descriptor)
    except Phase2BRecognizerImageBuilderErrorV3:
        raise
    except OSError as exc:
        _fail(f"private runtime file write failed: {type(exc).__name__}")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _fsync_directory_v3(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        os.fsync(descriptor)
    except OSError as exc:
        _fail(f"directory fsync failed: {type(exc).__name__}")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _verify_private_buildx_copy_v3(path: Path, expected_payload: bytes) -> None:
    if (
        type(expected_payload) is not bytes
        or len(expected_payload) != BUILDX_SIZE_BYTES_V3
        or _sha256(expected_payload) != BUILDX_SHA256_V3
    ):
        _fail("frozen buildx snapshot differs")
    observed = _read_stable_absolute_regular_v3(
        path,
        expected_sha256=BUILDX_SHA256_V3,
        expected_size=BUILDX_SIZE_BYTES_V3,
        expected_mode=0o755,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
        maximum_bytes=MAXIMUM_BUILDX_BYTES_V3,
    )
    if observed != expected_payload:
        _fail("private buildx copy bytes differ")


def _build_command_v3() -> tuple[str, ...]:
    return (
        DOCKER_EXECUTABLE_V3,
        f"--host={DOCKER_HOST_V3}",
        "build",
        "--pull=false",
        "--network=none",
        "--no-cache",
        "--platform=linux/amd64",
        "--build-arg=SOURCE_DATE_EPOCH=0",
        f"--file={DOCKERFILE_CONTEXT_PATH_V3}",
        f"--tag={IMAGE_TAG_V3}",
        "-",
    )


def _smoke_command_v3(
    *,
    image_digest: str,
    container_name: str,
    seccomp_path: Path,
) -> tuple[str, ...]:
    return (
        DOCKER_EXECUTABLE_V3,
        f"--host={DOCKER_HOST_V3}",
        "run",
        f"--name={container_name}",
        "--platform=linux/amd64",
        "--pull=never",
        "--no-healthcheck",
        "--read-only",
        "--network=none",
        "--ipc=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        f"--security-opt=seccomp={seccomp_path.as_posix()}",
        f"--user={RECOGNIZER_RUNTIME_UID_GID_V3}",
        "--workdir=/opt/hegel",
        "--restart=no",
        "--stop-timeout=1",
        "--cpus=1",
        "--memory=536870912",
        "--memory-swap=536870912",
        "--pids-limit=64",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,mode=0700,uid=65534,gid=65534,size=1048576",
        "--tmpfs=/phase2b/input:ro,noexec,nosuid,nodev,mode=0500,uid=65534,gid=65534,size=1048576",
        "--tmpfs=/phase2b/output:rw,noexec,nosuid,nodev,mode=0700,uid=65534,gid=65534,size=1048576",
        image_digest,
    )


def _make_receipt_v3(
    *,
    inputs: FrozenBuildInputsV3,
    image_digest: str,
    image_config_sha256: str,
    rootfs_layers: tuple[str, ...],
) -> tuple[dict[str, object], bytes]:
    claims = {
        "actual_960_case_run_verified": False,
        "actual_960_recognizer_executed": False,
        "actual_960_scorer_executed": False,
        "actual_input_archive_supplied": False,
        "actor_profile_authority_verified": False,
        "authority_requirement_satisfied": False,
        "c1_exit_evidence": False,
        "custodian_ceremony_executed": False,
        "effect_evidence": False,
        "exact_source_closure_verified": True,
        "execution_freeze_manifest_supplied": False,
        "formal_c1_report_verified": False,
        "formal_gate_evaluation_performed": False,
        "fresh_role_keys_generated": False,
        "key_generation_performed": False,
        "local_buildx_backend_verified": True,
        "local_build_command_verified": True,
        "local_image_built": True,
        "negative_smoke_executed": True,
        "prediction_archive_materialized": False,
        "prediction_commit_before_reveal_verified": False,
        "prediction_scored": False,
        "recognizer_operation_signature_verified": False,
        "reveal_authority_verified": False,
        "runtime_attestation_authority_verified": False,
        "runtime_image_config_verified": True,
        "scoring_performed": False,
        "signed_execution_registry_materialized": False,
        "signed_monotonic_time_materialized": False,
        "store_receipt_materialized": False,
    }
    body: dict[str, object] = {
        "schema_version": RECOGNIZER_RUNTIME_IMAGE_BUILD_RECEIPT_SCHEMA_V3,
        "claim_level": RECOGNIZER_RUNTIME_IMAGE_BUILD_CLAIM_LEVEL_V3,
        "disposition": RECOGNIZER_RUNTIME_IMAGE_BUILD_DISPOSITION_V3,
        "build": {
            "context_entry_count": len(inputs.context_entries),
            "context_mode": "EXACT_30_FILE_DETERMINISTIC_USTAR_STREAM",
            "context_sha256": inputs.context_sha256,
            "context_size_bytes": len(inputs.context_bytes),
            "docker_executable": DOCKER_EXECUTABLE_V3,
            "docker_host": DOCKER_HOST_V3,
            "dockerfile": DOCKERFILE_CONTEXT_PATH_V3,
            "dockerfile_run_instruction_count": 0,
            "image_tag": IMAGE_TAG_V3,
            "network_mode": "none",
            "no_cache": True,
            "platform": "linux/amd64",
            "pull": False,
            "source_date_epoch": 0,
        },
        "build_inputs": {
            "buildx_plugin_sha256": inputs.buildx_sha256,
            "buildx_plugin_size_bytes": inputs.buildx_size_bytes,
            "buildx_plugin_source_path": BUILDX_SOURCE_PATH_V3.as_posix(),
            "buildx_private_relative_path": BUILDX_PRIVATE_RELATIVE_PATH_V3,
            "buildx_version_stdout": BUILDX_VERSION_STDOUT_V3.decode("ascii").rstrip("\n"),
            "buildx_version_stdout_sha256": _sha256(BUILDX_VERSION_STDOUT_V3),
            "buildx_version_stderr_sha256": _sha256(b""),
            "buildx_private_copy_prebuild_verified": True,
            "buildx_private_copy_postbuild_verified": True,
            "context_entries": [
                {
                    "path": path,
                    "size_bytes": size,
                    "mode": format(mode, "04o"),
                    "sha256": digest,
                }
                for path, size, mode, digest in inputs.context_entries
            ],
            "dockerfile_sha256": inputs.dockerfile_sha256,
            "dockerignore_sha256": inputs.dockerignore_sha256,
            "entrypoint_sha256": ENTRYPOINT_SHA256_V3,
            "minimal_initializer_sha256": INITIALIZER_SHA256_V3,
            "owner_accepted_authority_requirement_id": (
                OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V3
            ),
            "profile_canonical_sha256": PROFILE_CANONICAL_SHA256_V3,
            "profile_content_id": PROFILE_CONTENT_ID_V3,
            "profile_file_sha256": inputs.profile_sha256,
            "runtime_seccomp_sha256": inputs.seccomp_sha256,
            "source_manifest": [
                {
                    "installed_path": installed,
                    "context_path": context,
                    "size_bytes": size,
                    "sha256": digest,
                }
                for installed, context, size, digest in inputs.source_entries
            ],
            "source_root_domain_hex": RECOGNIZER_SOURCE_ROOT_DOMAIN_V3.hex(),
            "source_root_entry_count": len(inputs.source_entries),
            "source_root_sha256": inputs.source_root_sha256,
        },
        "image": {
            "architecture": "amd64",
            "args_escaped": FROZEN_IMAGE_ARGS_ESCAPED_V3,
            "command": list(FROZEN_COMMAND_V3),
            "declared_volumes": None,
            "entrypoint": list(FROZEN_ENTRYPOINT_V3),
            "environment": list(FROZEN_IMAGE_ENVIRONMENT_V3),
            "healthcheck": None,
            "image_config_sha256": image_config_sha256,
            "image_digest": image_digest,
            "labels": dict(FROZEN_IMAGE_LABELS_V3),
            "os": "linux",
            "rootfs_layers": list(rootfs_layers),
            "user": RECOGNIZER_RUNTIME_UID_GID_V3,
            "working_directory": "/opt/hegel",
        },
        "qualification": {
            "container_absence_after_cleanup_verified": True,
            "container_capabilities_dropped": "ALL",
            "container_no_new_privileges": True,
            "container_output_host_mounts": 0,
            "negative_smoke_entrypoint": FROZEN_ENTRYPOINT_V3[0],
            "negative_smoke_command": list(FROZEN_COMMAND_V3),
            "negative_smoke_exit_code": 2,
            "negative_smoke_stderr_sha256": _sha256(NEGATIVE_SMOKE_STDERR_V3),
            "negative_smoke_stdout_sha256": _sha256(b""),
            "runtime_custom_seccomp_applied": True,
            "runtime_network_none": True,
            "runtime_root_read_only": True,
            "runtime_source_closure_exact": True,
            "runtime_uid_gid": RECOGNIZER_RUNTIME_UID_GID_V3,
        },
        "claims": claims,
    }
    body["receipt_id"] = (
        RECOGNIZER_RUNTIME_IMAGE_BUILD_RECEIPT_ID_PREFIX_V3
        + _sha256(
            RECOGNIZER_RUNTIME_IMAGE_BUILD_RECEIPT_DOMAIN_V3
            + _canonical_json_v3(body)
        )
    )
    receipt_bytes = _canonical_json_v3(body) + b"\n"
    return body, receipt_bytes


def build_and_qualify_recognizer_image_v3(
    *,
    runner: DockerRunnerV3 = _bounded_docker_run_v3,
) -> tuple[dict[str, object], bytes]:
    """Build and negatively qualify the image; do not publish a receipt."""

    inputs = load_frozen_build_inputs_v3()
    expected_seccomp = _decode_json_v3(
        inputs.seccomp_bytes,
        label="frozen runtime seccomp",
    )
    if type(expected_seccomp) is not dict:
        _fail("frozen runtime seccomp is not one JSON object")
    with tempfile.TemporaryDirectory(
        prefix="hegel-phase2b-recognizer-image-build-v3-",
        dir="/tmp",
    ) as temporary:
        runtime_root = Path(temporary)
        os.chmod(runtime_root, 0o700)
        docker_config = runtime_root / "docker-config"
        cli_plugins = docker_config / "cli-plugins"
        runtime_tmp = runtime_root / "tmp"
        docker_config.mkdir(mode=0o700)
        os.chmod(docker_config, 0o700)
        cli_plugins.mkdir(mode=0o700)
        os.chmod(cli_plugins, 0o700)
        runtime_tmp.mkdir(mode=0o700)
        os.chmod(runtime_tmp, 0o700)
        _write_private_file_v3(docker_config / "config.json", b"{}\n")
        private_buildx = docker_config / BUILDX_PRIVATE_RELATIVE_PATH_V3
        _write_private_file_v3(
            private_buildx,
            inputs.buildx_bytes,
            mode=0o755,
        )
        seccomp_path = runtime_root / "runtime-seccomp.json"
        _write_private_file_v3(seccomp_path, inputs.seccomp_bytes)
        _fsync_directory_v3(cli_plugins)
        _fsync_directory_v3(docker_config)
        _fsync_directory_v3(runtime_root)
        _verify_private_buildx_copy_v3(private_buildx, inputs.buildx_bytes)
        environment = {
            "BUILDKIT_PROGRESS": "plain",
            "DOCKER_BUILDKIT": "1",
            "DOCKER_CONFIG": docker_config.as_posix(),
            "DOCKER_HOST": DOCKER_HOST_V3,
            "HOME": runtime_root.as_posix(),
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "SOURCE_DATE_EPOCH": "0",
            "TMPDIR": runtime_tmp.as_posix(),
        }

        buildx_version_command = (
            DOCKER_EXECUTABLE_V3,
            f"--host={DOCKER_HOST_V3}",
            "buildx",
            "version",
        )
        buildx_version = _invoke_v3(
            runner,
            buildx_version_command,
            stdin_payload=None,
            environment=environment,
            timeout_seconds=INSPECT_TIMEOUT_SECONDS_V3,
            maximum_stdout_bytes=4_096,
            maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V3,
        )
        if (
            buildx_version.returncode != 0
            or buildx_version.stdout != BUILDX_VERSION_STDOUT_V3
            or buildx_version.stderr != b""
        ):
            _fail("frozen local buildx backend qualification failed")
        _verify_private_buildx_copy_v3(private_buildx, inputs.buildx_bytes)

        build_command = _build_command_v3()
        build = _invoke_v3(
            runner,
            build_command,
            stdin_payload=inputs.context_bytes,
            environment=environment,
            timeout_seconds=BUILD_TIMEOUT_SECONDS_V3,
            maximum_stdout_bytes=MAXIMUM_BUILD_STDOUT_BYTES_V3,
            maximum_stderr_bytes=MAXIMUM_BUILD_STDERR_BYTES_V3,
        )
        if build.returncode != 0:
            _fail("offline recognizer image build failed")
        _verify_private_buildx_copy_v3(private_buildx, inputs.buildx_bytes)

        inspect_tag_command = (
            DOCKER_EXECUTABLE_V3,
            f"--host={DOCKER_HOST_V3}",
            "image",
            "inspect",
            IMAGE_TAG_V3,
        )
        inspected = _invoke_v3(
            runner,
            inspect_tag_command,
            stdin_payload=None,
            environment=environment,
            timeout_seconds=INSPECT_TIMEOUT_SECONDS_V3,
            maximum_stdout_bytes=MAXIMUM_INSPECT_STDOUT_BYTES_V3,
            maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V3,
        )
        if inspected.returncode != 0:
            _fail("built recognizer image cannot be inspected")
        image_digest, image_config_sha256, rootfs_layers = (
            _validate_image_inspect_v3(inspected.stdout)
        )

        nonce = hashlib.sha256(
            (runtime_root.name + "\x00" + str(os.getpid())).encode("utf-8")
        ).hexdigest()[:16]
        container_name = f"hegel-phase2b-recognizer-negative-smoke-v3-{nonce}"
        smoke_command = _smoke_command_v3(
            image_digest=image_digest,
            container_name=container_name,
            seccomp_path=seccomp_path,
        )
        smoke_error: BaseException | None = None
        try:
            smoke = _invoke_v3(
                runner,
                smoke_command,
                stdin_payload=None,
                environment=environment,
                timeout_seconds=SMOKE_TIMEOUT_SECONDS_V3,
                maximum_stdout_bytes=MAXIMUM_SMOKE_STDOUT_BYTES_V3,
                maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V3,
            )
            if (
                smoke.returncode != 2
                or smoke.stdout != b""
                or smoke.stderr != NEGATIVE_SMOKE_STDERR_V3
            ):
                _fail("negative recognizer smoke did not produce exact fail-closed exit 2")
            inspect_container_command = (
                DOCKER_EXECUTABLE_V3,
                f"--host={DOCKER_HOST_V3}",
                "container",
                "inspect",
                container_name,
            )
            smoke_inspect = _invoke_v3(
                runner,
                inspect_container_command,
                stdin_payload=None,
                environment=environment,
                timeout_seconds=INSPECT_TIMEOUT_SECONDS_V3,
                maximum_stdout_bytes=MAXIMUM_INSPECT_STDOUT_BYTES_V3,
                maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V3,
            )
            if smoke_inspect.returncode != 0:
                _fail("negative recognizer smoke container cannot be inspected")
            _validate_smoke_container_v3(
                smoke_inspect.stdout,
                image_digest=image_digest,
                container_name=container_name,
                expected_seccomp=expected_seccomp,
            )
        except BaseException as exc:
            smoke_error = exc
        finally:
            remove_command = (
                DOCKER_EXECUTABLE_V3,
                f"--host={DOCKER_HOST_V3}",
                "container",
                "rm",
                "--force",
                "--volumes",
                container_name,
            )
            cleanup = _invoke_v3(
                runner,
                remove_command,
                stdin_payload=None,
                environment=environment,
                timeout_seconds=CLEANUP_TIMEOUT_SECONDS_V3,
                maximum_stdout_bytes=MAXIMUM_SMOKE_STDOUT_BYTES_V3,
                maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V3,
            )
            absence_command = (
                DOCKER_EXECUTABLE_V3,
                f"--host={DOCKER_HOST_V3}",
                "container",
                "inspect",
                container_name,
            )
            absence = _invoke_v3(
                runner,
                absence_command,
                stdin_payload=None,
                environment=environment,
                timeout_seconds=INSPECT_TIMEOUT_SECONDS_V3,
                maximum_stdout_bytes=MAXIMUM_INSPECT_STDOUT_BYTES_V3,
                maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V3,
            )
            _validate_cleanup_proof_v3(
                cleanup,
                absence,
                container_name=container_name,
                allow_already_absent=smoke_error is not None,
            )
        if smoke_error is not None:
            if isinstance(smoke_error, Phase2BRecognizerImageBuilderErrorV3):
                raise smoke_error
            _fail(f"negative recognizer smoke failed: {type(smoke_error).__name__}")

        inspect_digest_command = (
            DOCKER_EXECUTABLE_V3,
            f"--host={DOCKER_HOST_V3}",
            "image",
            "inspect",
            image_digest,
        )
        reinspected = _invoke_v3(
            runner,
            inspect_digest_command,
            stdin_payload=None,
            environment=environment,
            timeout_seconds=INSPECT_TIMEOUT_SECONDS_V3,
            maximum_stdout_bytes=MAXIMUM_INSPECT_STDOUT_BYTES_V3,
            maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V3,
        )
        if reinspected.returncode != 0:
            _fail("qualified recognizer image cannot be re-inspected by digest")
        second_digest, second_config_sha256, second_layers = (
            _validate_image_inspect_v3(reinspected.stdout)
        )
        if (
            second_digest != image_digest
            or second_config_sha256 != image_config_sha256
            or second_layers != rootfs_layers
        ):
            _fail("recognizer image identity changed across negative qualification")

    return _make_receipt_v3(
        inputs=inputs,
        image_digest=image_digest,
        image_config_sha256=image_config_sha256,
        rootfs_layers=rootfs_layers,
    )


def _canonical_absolute_receipt_path_v3(value: object) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        _fail("receipt path is not exact text")
    path = Path(value)
    if (
        not path.is_absolute()
        or Path(os.path.abspath(os.fspath(path))) != path
        or len(value.encode("utf-8")) > 4096
        or path.name in {"", ".", ".."}
    ):
        _fail("receipt path is not one normalized absolute path")
    return path


def _open_directory_componentwise_v3(path: Path) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open("/", flags)
        for component in path.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            _fail("receipt parent is not a directory")
        return descriptor
    except Phase2BRecognizerImageBuilderErrorV3:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        _fail(f"receipt parent cannot be opened: {type(exc).__name__}")


def assert_receipt_target_absent_v3(path: Path) -> None:
    parent_fd = _open_directory_componentwise_v3(path.parent)
    try:
        try:
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        _fail("receipt target already exists")
    finally:
        os.close(parent_fd)


def publish_receipt_no_replace_v3(path: Path, payload: bytes) -> None:
    """Durably publish one new receipt through hard-link no-replace."""

    if type(payload) is not bytes or not payload or not payload.endswith(b"\n"):
        _fail("receipt payload differs")
    parent_fd = _open_directory_componentwise_v3(path.parent)
    pending = ".phase2b-recognizer-image-" + _sha256(payload)[:24] + ".pending"
    pending_fd: int | None = None
    try:
        try:
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            _fail("receipt target already exists")
        pending_fd = os.open(
            pending,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(pending_fd, payload[offset:])
            if written <= 0:
                _fail("receipt write made no progress")
            offset += written
        os.fsync(pending_fd)
        os.link(
            pending,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if os.fstat(pending_fd).st_nlink != 2:
            _fail("receipt no-replace link count differs")
        os.fsync(parent_fd)
        os.unlink(pending, dir_fd=parent_fd)
        os.fsync(parent_fd)
        os.close(pending_fd)
        pending_fd = None
        verified = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        try:
            metadata = os.fstat(verified)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(verified, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or b"".join(chunks) != payload
            ):
                _fail("published receipt verification differs")
            os.fsync(verified)
        finally:
            os.close(verified)
    except FileExistsError:
        _fail("receipt no-replace target or pending name already exists")
    except Phase2BRecognizerImageBuilderErrorV3:
        raise
    except OSError as exc:
        _fail(f"receipt no-replace publication failed: {type(exc).__name__}")
    finally:
        if pending_fd is not None:
            os.close(pending_fd)
        try:
            os.unlink(pending, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        os.close(parent_fd)


def _parse_cli_v3(argv: Sequence[str] | None) -> Path:
    raw = tuple(sys.argv[1:] if argv is None else argv)
    if len(raw) != 2 or raw[0] != "--receipt" or type(raw[1]) is not str:
        _fail("usage: build_phase2b_recognizer_runtime_image_v3.py --receipt /absolute/new.json")
    return _canonical_absolute_receipt_path_v3(raw[1])


def main(argv: Sequence[str] | None = None) -> int:
    try:
        receipt_path = _parse_cli_v3(argv)
        assert_receipt_target_absent_v3(receipt_path)
        _receipt, payload = build_and_qualify_recognizer_image_v3()
        publish_receipt_no_replace_v3(receipt_path, payload)
    except Exception:
        sys.stderr.write("phase2b recognizer V3 image build/qualification failed\n")
        sys.stderr.flush()
        return 2
    return 0


__all__ = (
    "BASE_IMAGE_DIGEST_V3",
    "BoundedDockerResultV3",
    "BUILDX_PRIVATE_RELATIVE_PATH_V3",
    "BUILDX_SHA256_V3",
    "BUILDX_SIZE_BYTES_V3",
    "BUILDX_SOURCE_PATH_V3",
    "BUILDX_VERSION_STDOUT_V3",
    "DOCKERFILE_CONTEXT_PATH_V3",
    "DOCKERFILE_SHA256_V3",
    "DOCKERIGNORE_CONTEXT_PATH_V3",
    "DOCKERIGNORE_SHA256_V3",
    "DOCKER_EXECUTABLE_V3",
    "DOCKER_HOST_V3",
    "FROZEN_IMAGE_ARGS_ESCAPED_V3",
    "FROZEN_IMAGE_CONFIG_FIELDS_V3",
    "FROZEN_IMAGE_ENVIRONMENT_V3",
    "FROZEN_IMAGE_LABELS_V3",
    "FROZEN_SOURCE_MANIFEST_V3",
    "IMAGE_TAG_V3",
    "NEGATIVE_SMOKE_STDERR_V3",
    "OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V3",
    "Phase2BRecognizerImageBuilderErrorV3",
    "PROFILE_FILE_SHA256_V3",
    "PROFILE_CONTENT_ID_V3",
    "PROFILE_CANONICAL_SHA256_V3",
    "RECOGNIZER_RUNTIME_IMAGE_BUILD_RECEIPT_DOMAIN_V3",
    "RECOGNIZER_RUNTIME_IMAGE_BUILD_RECEIPT_ID_PREFIX_V3",
    "RECOGNIZER_RUNTIME_IMAGE_BUILD_RECEIPT_SCHEMA_V3",
    "RECOGNIZER_SOURCE_ROOT_DOMAIN_V3",
    "RECOGNIZER_SOURCE_ROOT_SHA256_V3",
    "SECCOMP_SHA256_V3",
    "assert_receipt_target_absent_v3",
    "build_and_qualify_recognizer_image_v3",
    "load_frozen_build_inputs_v3",
    "main",
    "publish_receipt_no_replace_v3",
)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
