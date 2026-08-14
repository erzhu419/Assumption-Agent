#!/usr/bin/env python3
"""Build and negatively qualify the frozen Phase-2B scorer image.

This tool intentionally establishes only a local Docker image/build/config and
negative-smoke observation.  It does not consume an answer, a reveal token, or
an actual-960 scorer bundle, and none of its outputs are scoring, gate, effect,
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
DOCKER_EXECUTABLE_V1: Final = "/usr/bin/docker"
DOCKER_HOST_V1: Final = "unix:///var/run/docker.sock"
DOCKERFILE_CONTEXT_PATH_V1: Final = (
    "docker/phase2b_scorer_runtime_v1/Dockerfile"
)
DOCKERIGNORE_CONTEXT_PATH_V1: Final = (
    "docker/phase2b_scorer_runtime_v1/Dockerfile.dockerignore"
)
PROFILE_PATH_V1: Final = "config/phase2b_c1_container_actor_profile_v2.json"
SECCOMP_PATH_V1: Final = "config/phase3_internal_actor_seccomp_v1.json"
BUILDX_SOURCE_PATH_V1: Final = Path(
    "/home/erzhu419/.local/state/hegel-machine/docker-cli-plugins/docker-buildx"
)
SCORING_SOURCE_PATH_V1: Final = (
    "src/hegel_machine/phase2b_unsealed_960_prediction_scoring_mechanics_v2.py"
)

SCORER_RUNTIME_IMAGE_BUILD_RECEIPT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-scorer-runtime-image-build-receipt/1"
)
SCORER_RUNTIME_IMAGE_BUILD_RECEIPT_ID_PREFIX_V1: Final = (
    "phase2b_scorer_runtime_image_build_receipt_v1_"
)
SCORER_RUNTIME_IMAGE_BUILD_RECEIPT_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/SCORER/RUNTIME_IMAGE_BUILD_RECEIPT/V1\x00"
)
SCORER_RUNTIME_IMAGE_BUILD_CLAIM_LEVEL_V1: Final = (
    "NON_AUTHORITATIVE_LOCAL_DOCKER_IMAGE_BUILD_AND_NEGATIVE_SMOKE_ONLY"
)
SCORER_RUNTIME_IMAGE_BUILD_DISPOSITION_V1: Final = (
    "LOCAL_SCORER_IMAGE_BUILD_AND_NEGATIVE_SMOKE_COMPLETE_NOT_ACTUAL_SCORING"
)

BASE_IMAGE_DIGEST_V1: Final = (
    "sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"
)
PROFILE_CONTENT_ID_V1: Final = (
    "phase2b_c1_container_actor_profile_v2_"
    "b2f081771f3de8d2a531b08b58109d95c51815904e9612e66f1bbfe69babf906"
)
PROFILE_CANONICAL_SHA256_V1: Final = (
    "8ca1850f026dabb7a4b69add0f89b33a3fcfb789223b2fd95ef0379608adbeed"
)
SCORER_SOURCE_ROOT_SHA256_V1: Final = (
    "34f4a2404d00d46cede832d34ca9713253da93f1becef4a1d302c1411d6bd5e7"
)
SCORER_SOURCE_ROOT_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/SCORER_RUNTIME_SOURCE_ROOT/V1\x00"
)
SCORING_SOURCE_SHA256_V1: Final = (
    "c3c12e9cd72a930b9ca8667aab22c4eb3675ceab6249b0b26ba940bf794e7f11"
)
DOCKERFILE_SHA256_V1: Final = (
    "33a72fb900e7b90ec5d0b03999703c06afd5d2345d0f7d7cf51c9976f213b923"
)
DOCKERIGNORE_SHA256_V1: Final = (
    "79d1a10ec8bcf1d28f0b586196484c6d5220e85ce84ae1442d75905d9fc2e257"
)
PROFILE_FILE_SHA256_V1: Final = (
    "06c565dc526f5987fdc6abac8fb76dcb9dae167a935c40997faafe8fbb2ff473"
)
SECCOMP_SHA256_V1: Final = (
    "d1284e4731683b73352ecdd1577704ea87aa0b5c582b7b00757c3db4d2c950ca"
)
INITIALIZER_SHA256_V1: Final = (
    "c853b7d94fb820d0efa9dc4961d95e0fa47d6b3fa0ceabb67285b1fb7691b246"
)
ENTRYPOINT_SHA256_V1: Final = (
    "b72cd375c4b17b3ef32ae0efc6b64e2e027f1faca510c09b7ad82675cefadd1c"
)
BUILDX_SHA256_V1: Final = (
    "2d836b76601cb911a1751b12a7a94a4f64e10fb0bb8f3839d377311a30f0d259"
)
BUILDX_SIZE_BYTES_V1: Final = 70_944_600
BUILDX_VERSION_STDOUT_V1: Final = (
    b"github.com/docker/buildx 0.30.1 0.30.1-0ubuntu1~22.04.1\n"
)
BUILDX_PRIVATE_RELATIVE_PATH_V1: Final = "cli-plugins/docker-buildx"

IMAGE_TAG_V1: Final = (
    "hegel-machine-phase2b-scorer-runtime-v1:source-"
    + SCORER_SOURCE_ROOT_SHA256_V1[:16]
)
SCORER_RUNTIME_UID_GID_V1: Final = "65534:65534"
SCORER_RUNTIME_IMAGE_TITLE_V1: Final = (
    "Hegel Machine Phase2B V1 scorer runtime"
)
SCORER_RUNTIME_IMAGE_DESCRIPTION_V1: Final = (
    "Non-authoritative local Docker scorer mechanics; no actual scoring, "
    "formal gate, or C1 evidence"
)
FROZEN_ENTRYPOINT_V1: Final = ("/opt/hegel/bin/phase2b-scorer",)
FROZEN_COMMAND_V1: Final = (
    "--input",
    "/phase2b/input/phase2b_scorer_bundle_v1.bin",
    "--output",
    "/phase2b/output",
)
FROZEN_IMAGE_ARGS_ESCAPED_V1: Final = True
FROZEN_IMAGE_CONFIG_FIELDS_V1: Final = frozenset(
    {"ArgsEscaped", "Cmd", "Entrypoint", "Env", "Labels", "User", "WorkingDir"}
)
FROZEN_IMAGE_ENVIRONMENT_V1: Final = (
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
FROZEN_IMAGE_LABELS_V1: Final = {
    "org.opencontainers.image.base.digest": BASE_IMAGE_DIGEST_V1,
    "org.opencontainers.image.description": SCORER_RUNTIME_IMAGE_DESCRIPTION_V1,
    "org.opencontainers.image.title": SCORER_RUNTIME_IMAGE_TITLE_V1,
    "org.hegel-machine.phase2b.actor-profile.canonical-sha256": (
        PROFILE_CANONICAL_SHA256_V1
    ),
    "org.hegel-machine.phase2b.actor-profile.content-id": PROFILE_CONTENT_ID_V1,
    "org.hegel-machine.phase2b.scorer-source-root.sha256": (
        SCORER_SOURCE_ROOT_SHA256_V1
    ),
    "org.hegel-machine.phase2b.scoring-source.sha256": SCORING_SOURCE_SHA256_V1,
}

MAXIMUM_SOURCE_FILE_BYTES_V1: Final = 16 * 1024 * 1024
MAXIMUM_BUILDX_BYTES_V1: Final = 128 * 1024 * 1024
MAXIMUM_CONTEXT_BYTES_V1: Final = 8 * 1024 * 1024
MAXIMUM_BUILD_STDOUT_BYTES_V1: Final = 4 * 1024 * 1024
MAXIMUM_BUILD_STDERR_BYTES_V1: Final = 4 * 1024 * 1024
MAXIMUM_INSPECT_STDOUT_BYTES_V1: Final = 1024 * 1024
MAXIMUM_DOCKER_STDERR_BYTES_V1: Final = 64 * 1024
MAXIMUM_SMOKE_STDOUT_BYTES_V1: Final = 64 * 1024
BUILD_TIMEOUT_SECONDS_V1: Final = 900
INSPECT_TIMEOUT_SECONDS_V1: Final = 60
SMOKE_TIMEOUT_SECONDS_V1: Final = 60
CLEANUP_TIMEOUT_SECONDS_V1: Final = 30
_DIGEST_V1: Final = re.compile(r"sha256:[0-9a-f]{64}")
_CONTAINER_ID_V1: Final = re.compile(r"[0-9a-f]{64}")


# Ordered by the exact installed absolute POSIX path.  The tuple is the frozen
# 49-entry source manifest; context_path is the only byte source admitted to
# the deterministic build stream.
FROZEN_SOURCE_MANIFEST_V1: Final = (
    ("/opt/hegel/bin/phase2b-scorer", "docker/phase2b_scorer_runtime_v1/phase2b-scorer", "b72cd375c4b17b3ef32ae0efc6b64e2e027f1faca510c09b7ad82675cefadd1c"),
    ("/opt/hegel/src/hegel_machine/__init__.py", "docker/phase2b_scorer_runtime_v1/hegel_machine_init.py", "c853b7d94fb820d0efa9dc4961d95e0fa47d6b3fa0ceabb67285b1fb7691b246"),
    ("/opt/hegel/src/hegel_machine/bootstrap.py", "src/hegel_machine/bootstrap.py", "aecb9711f0c2238e22140b6ed65ba2533747b590bf8d16f70f65f964fd3f3b4b"),
    ("/opt/hegel/src/hegel_machine/hashing.py", "src/hegel_machine/hashing.py", "6fcd988ca4e8b75b9aa9e5a8b38c9ddd73e2685140e2cd0095d68f8bda054159"),
    ("/opt/hegel/src/hegel_machine/laws.py", "src/hegel_machine/laws.py", "2bb3e96544cfc9c52718efdc2880fb33607bdc597bf47c30b640ee07cb654fc2"),
    ("/opt/hegel/src/hegel_machine/milestones.py", "src/hegel_machine/milestones.py", "049e4ec78798d8798dac8eb19c97921e03b942c7adb46287c62a16e4a98ab276"),
    ("/opt/hegel/src/hegel_machine/ontology.py", "src/hegel_machine/ontology.py", "19828518e4c3cc6a4c2171766fc576a624e836613f968cfd09589a61b78e03ab"),
    ("/opt/hegel/src/hegel_machine/phase2b_actual_unsealed_960_external_custodian_checkpoint_verifier_mechanics_v2.py", "src/hegel_machine/phase2b_actual_unsealed_960_external_custodian_checkpoint_verifier_mechanics_v2.py", "078786e4f6ed66bad05598ed839c8cce6f93f6999b92d4c0309f7d3a29da6a5f"),
    ("/opt/hegel/src/hegel_machine/phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2.py", "src/hegel_machine/phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2.py", "18c19fc9e2ea01a39504f5b38422b9f42029d2ae4faa426fc90a473636cf6a07"),
    ("/opt/hegel/src/hegel_machine/phase2b_actual_unsealed_960_replay_admission_evidence_contract_v2.py", "src/hegel_machine/phase2b_actual_unsealed_960_replay_admission_evidence_contract_v2.py", "b1dc11a0ac6e3dfafc9621c10957308b9d50c30e8806e78ff5b6a056a8ff1450"),
    ("/opt/hegel/src/hegel_machine/phase2b_actual_unsealed_960_replay_input_contract_v2.py", "src/hegel_machine/phase2b_actual_unsealed_960_replay_input_contract_v2.py", "35da6f01163835ca90c24cdbd4ad85a1f7f0b2ef78ceb6e70fe80de2534814f2"),
    ("/opt/hegel/src/hegel_machine/phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2.py", "src/hegel_machine/phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2.py", "7ba5e891b08a2a0b2d59161d784f64a9ecd2a11053f5436886f73307c3261e9c"),
    ("/opt/hegel/src/hegel_machine/phase2b_adapter.py", "src/hegel_machine/phase2b_adapter.py", "3f6d0978d3ca09013e9257da3ca5312eb05adb89dd3525329a5e850e14b3c8bb"),
    ("/opt/hegel/src/hegel_machine/phase2b_c1_container_actor_profile_v1.py", "src/hegel_machine/phase2b_c1_container_actor_profile_v1.py", "f254ff8616d73fd8e5fc7ef70bf87cbfabc2d3a053007d01415d54c6fbf7a59c"),
    ("/opt/hegel/src/hegel_machine/phase2b_c1_container_actor_profile_v2.py", "src/hegel_machine/phase2b_c1_container_actor_profile_v2.py", "4e41821df7d32f62f00bc7ec554bc37058f2ca52055b5f4fe3e4df91ca582c5c"),
    ("/opt/hegel/src/hegel_machine/phase2b_covert_audit_v1.py", "src/hegel_machine/phase2b_covert_audit_v1.py", "96585dbf4c9e2eaf019b228b6c3ec13e5bf951b0d761a6277f72fe628045a3a8"),
    ("/opt/hegel/src/hegel_machine/phase2b_exact_bridge_v1.py", "src/hegel_machine/phase2b_exact_bridge_v1.py", "bf8637ab3f43ba670d64659568022031df4623182c4d7e6fc546827000b14d6e"),
    ("/opt/hegel/src/hegel_machine/phase2b_exact_derived_witness_bridge_v1.py", "src/hegel_machine/phase2b_exact_derived_witness_bridge_v1.py", "56fd5b1f1730bc7b870f28fe6908f88a3501bd4ee98b0c4c5622c8f78ed18d93"),
    ("/opt/hegel/src/hegel_machine/phase2b_exact_transform_semantics_v1.py", "src/hegel_machine/phase2b_exact_transform_semantics_v1.py", "353b10a7464da54553cf84e738bd53c010274ad1b15e9c36729420ea11f8e55c"),
    ("/opt/hegel/src/hegel_machine/phase2b_formal_unsealed_prediction_scoring_contract_v2.py", "src/hegel_machine/phase2b_formal_unsealed_prediction_scoring_contract_v2.py", "7070a115669b0a4a9ea7def82b43f7d544c9c7bf3cb1167a3922c5b11e1a233b"),
    ("/opt/hegel/src/hegel_machine/phase2b_freeze_v1.py", "src/hegel_machine/phase2b_freeze_v1.py", "39a289ef0c000c1c62eb3a7a5f97ddfae3e51eb7904757642b0c2d167067f543"),
    ("/opt/hegel/src/hegel_machine/phase2b_projection_compiler.py", "src/hegel_machine/phase2b_projection_compiler.py", "a1ced91263916092a1fa05ba50ec15b50d6ed4b2790142948792865a9e785500"),
    ("/opt/hegel/src/hegel_machine/phase2b_protocol.py", "src/hegel_machine/phase2b_protocol.py", "aeccff3558e82ebf261cbeb1c389e968ff22c8ae229b9ce201808accf76074ef"),
    ("/opt/hegel/src/hegel_machine/phase2b_recognizer_input_archive_v1.py", "src/hegel_machine/phase2b_recognizer_input_archive_v1.py", "a73c21996122f3cb3914efd67f4f70c4c563abfa040add34016553e67e6f5314"),
    ("/opt/hegel/src/hegel_machine/phase2b_recognizer_input_archive_v2.py", "src/hegel_machine/phase2b_recognizer_input_archive_v2.py", "a50473bb942de6dea823cd6523c096a9d33c07a8395efa03cc6d7c8b4b035108"),
    ("/opt/hegel/src/hegel_machine/phase2b_recognizer_prediction_archive_v1.py", "src/hegel_machine/phase2b_recognizer_prediction_archive_v1.py", "f894fbb07ea2eddd8ee510c008f27ed21a6caf6e35be8cec3652e3fd4056ed7d"),
    ("/opt/hegel/src/hegel_machine/phase2b_recognizer_prediction_archive_v2.py", "src/hegel_machine/phase2b_recognizer_prediction_archive_v2.py", "ecfe3ff599f10deae845a32cb979209d0591727bd36ef98583194412d74cb04a"),
    ("/opt/hegel/src/hegel_machine/phase2b_recognizer_prediction_v2.py", "src/hegel_machine/phase2b_recognizer_prediction_v2.py", "e32133bfe7d8848b56fcbcf6f68849908b7d9a203a6e6196aae6d954e011c023"),
    ("/opt/hegel/src/hegel_machine/phase2b_recognizer_runtime_cli_v2.py", "src/hegel_machine/phase2b_recognizer_runtime_cli_v2.py", "b219489a782bcdcdd7fdca2d8a900afac8c3bb38a3bf8b21276ce29f5d88ec19"),
    ("/opt/hegel/src/hegel_machine/phase2b_runner.py", "src/hegel_machine/phase2b_runner.py", "24ae136477745356d17aac2906ed46ef41f16f095875643c0c0c7d8d1e11b5ae"),
    ("/opt/hegel/src/hegel_machine/phase2b_scorer_bundle_v1.py", "src/hegel_machine/phase2b_scorer_bundle_v1.py", "8c2c6ae00d55274f38e8f346b2ababd2beaaee25ade9e9fccd23f4db7d7d63e1"),
    ("/opt/hegel/src/hegel_machine/phase2b_scorer_runtime_cli_v1.py", "src/hegel_machine/phase2b_scorer_runtime_cli_v1.py", "8af11fec73314842f7ab16cf0a0931f313c92c56386b0ce5e01c2d4908c6efba"),
    ("/opt/hegel/src/hegel_machine/phase2b_selector.py", "src/hegel_machine/phase2b_selector.py", "d5f97f038e07800d273cc610c3d6b50d3defb0bc30b1943a3a150f046614eed6"),
    ("/opt/hegel/src/hegel_machine/phase2b_strict_recognizer_cli_v2.py", "src/hegel_machine/phase2b_strict_recognizer_cli_v2.py", "167e3492683e56599d541ce91c9534fcefb01ba5ab702cef0803947e1cd6a1bb"),
    ("/opt/hegel/src/hegel_machine/phase2b_trusted_wire_batch_v1.py", "src/hegel_machine/phase2b_trusted_wire_batch_v1.py", "0b2606bd7ba4e41253beaadb9fee5e5d3a301382dac5a6025f51756e5cd1eeaa"),
    ("/opt/hegel/src/hegel_machine/phase2b_trusted_wire_batch_v2.py", "src/hegel_machine/phase2b_trusted_wire_batch_v2.py", "42e4fba7ff3829f57eb60b4f8d1ffe87099deed8b751cc2b3101bcfa02f703b5"),
    ("/opt/hegel/src/hegel_machine/phase2b_trusted_wire_typed_authority_v1.py", "src/hegel_machine/phase2b_trusted_wire_typed_authority_v1.py", "d6e39f81257026281fd81d191fcc6e37be8ff130a6bde6ac0d8e417cb6c11f63"),
    ("/opt/hegel/src/hegel_machine/phase2b_trusted_wire_typed_authority_v2.py", "src/hegel_machine/phase2b_trusted_wire_typed_authority_v2.py", "6d6cc72547a8f00402e126b4a339272e6de0353525e7627c082ef62faa407521"),
    ("/opt/hegel/src/hegel_machine/phase2b_trusted_wire_typed_replay_v1.py", "src/hegel_machine/phase2b_trusted_wire_typed_replay_v1.py", "91a5c98be76c4b661cb8f72cbe75eb26b26168414ce3040085b0c7a7fbdaba25"),
    ("/opt/hegel/src/hegel_machine/phase2b_trusted_wire_typed_replay_v2.py", "src/hegel_machine/phase2b_trusted_wire_typed_replay_v2.py", "0608113116bf0f548acd17d509a7ca0e13889d518260e1b9f4a3aaf8ab942309"),
    ("/opt/hegel/src/hegel_machine/phase2b_trusted_wire_v1.py", "src/hegel_machine/phase2b_trusted_wire_v1.py", "98defeb8455944144e318879688174488e1c13aab2153c019a6db4b67946b5c2"),
    ("/opt/hegel/src/hegel_machine/phase2b_uncertainty_compiler.py", "src/hegel_machine/phase2b_uncertainty_compiler.py", "529c3d8e7dc2baec20c76ec5f9f21c30c31c1842305872499ba95e6dbb584d35"),
    ("/opt/hegel/src/hegel_machine/phase2b_unsealed_960_available_gate_mechanics_v2.py", "src/hegel_machine/phase2b_unsealed_960_available_gate_mechanics_v2.py", "7581f84a0b117de2c299ea338329c6420fa6e063192de3a6330eff78df983a72"),
    ("/opt/hegel/src/hegel_machine/phase2b_unsealed_960_prediction_scoring_mechanics_v2.py", "src/hegel_machine/phase2b_unsealed_960_prediction_scoring_mechanics_v2.py", "c3c12e9cd72a930b9ca8667aab22c4eb3675ceab6249b0b26ba940bf794e7f11"),
    ("/opt/hegel/src/hegel_machine/phase2b_unsealed_prediction_evaluator_v1.py", "src/hegel_machine/phase2b_unsealed_prediction_evaluator_v1.py", "37fd6add57b525c7f3d84fe9231cd515723b1f0566bd140212bf2f5c99aba725"),
    ("/opt/hegel/src/hegel_machine/phase2b_unsealed_prediction_evaluator_v2.py", "src/hegel_machine/phase2b_unsealed_prediction_evaluator_v2.py", "f7c1e98fe3e354047e0e76d75f699ee6f3f98493a52fde3f9724f4acce5858e5"),
    ("/opt/hegel/src/hegel_machine/phase2b_wire.py", "src/hegel_machine/phase2b_wire.py", "0330d205b6bdfd9452063e636873e99cb8b8d902decd4124e52207061294d9db"),
    ("/opt/hegel/src/hegel_machine/phase3_dsl_v1.py", "src/hegel_machine/phase3_dsl_v1.py", "c0e3ef208d08c7415dd934f8066f6c845a943272e1b9033db3b2a4083694ae27"),
    ("/opt/hegel/src/hegel_machine/schema.py", "src/hegel_machine/schema.py", "6497a93f8a68f3313d294bc99dc3b8f43508e93772d748521d09ea716b8ae10a"),
)


class Phase2BScorerImageBuilderErrorV1(RuntimeError):
    """One stable fail-closed rejection from the local image builder."""


def _fail(detail: str) -> NoReturn:
    raise Phase2BScorerImageBuilderErrorV1(detail)


@dataclass(frozen=True)
class BoundedDockerResultV1:
    command: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True)
class FrozenBuildInputsV1:
    context_bytes: bytes
    context_sha256: str
    context_entries: tuple[tuple[str, int, int, str], ...]
    source_entries: tuple[tuple[str, str, int, str], ...]
    source_root_sha256: str
    dockerfile_sha256: str
    dockerignore_sha256: str
    profile_sha256: str
    scoring_source_sha256: str
    seccomp_sha256: str
    seccomp_bytes: bytes
    buildx_sha256: str
    buildx_size_bytes: int
    buildx_bytes: bytes


DockerRunnerV1 = Callable[..., BoundedDockerResultV1]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_frozen_file_v1(
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
            or not 0 <= before.st_size <= MAXIMUM_SOURCE_FILE_BYTES_V1
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
    except Phase2BScorerImageBuilderErrorV1:
        raise
    except OSError as exc:
        _fail(f"frozen source cannot be opened: {relative_path}: {type(exc).__name__}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if _sha256(payload) != expected_sha256:
        _fail(f"frozen source digest differs: {relative_path}")
    return payload


def _read_stable_absolute_regular_v1(
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
    parent_fd = _open_directory_componentwise_v1(path.parent)
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
    except Phase2BScorerImageBuilderErrorV1:
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


def _compute_source_root_v1(
    ordered: Sequence[tuple[str, bytes]],
) -> str:
    if (
        type(ordered) not in {list, tuple}
        or len(ordered) != 49
        or tuple(path for path, _payload in ordered)
        != tuple(sorted(path for path, _payload in ordered))
    ):
        _fail("source-root ordered surface differs")
    preimage = bytearray(SCORER_SOURCE_ROOT_DOMAIN_V1)
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


def _deterministic_ustar_v1(
    entries: Mapping[str, tuple[bytes, int]],
) -> bytes:
    if type(entries) is not dict or len(entries) != 51:
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
    if not payload or len(payload) > MAXIMUM_CONTEXT_BYTES_V1:
        _fail("deterministic ustar context exceeds its frozen bound")
    return payload


def load_frozen_build_inputs_v1() -> FrozenBuildInputsV1:
    """Snapshot and verify every byte before any Docker process is started."""

    if (
        len(FROZEN_SOURCE_MANIFEST_V1) != 49
        or tuple(row[0] for row in FROZEN_SOURCE_MANIFEST_V1)
        != tuple(sorted(row[0] for row in FROZEN_SOURCE_MANIFEST_V1))
        or len({row[0] for row in FROZEN_SOURCE_MANIFEST_V1}) != 49
        or len({row[1] for row in FROZEN_SOURCE_MANIFEST_V1}) != 49
    ):
        _fail("frozen 49-entry source manifest differs")

    source_payloads: list[tuple[str, bytes]] = []
    source_entries: list[tuple[str, str, int, str]] = []
    context: dict[str, tuple[bytes, int]] = {}
    for installed_path, context_path, expected_sha256 in FROZEN_SOURCE_MANIFEST_V1:
        mode = 0o755 if installed_path == FROZEN_ENTRYPOINT_V1[0] else 0o644
        payload = _read_frozen_file_v1(
            context_path,
            expected_sha256,
            expected_mode=mode,
        )
        source_payloads.append((installed_path, payload))
        source_entries.append(
            (installed_path, context_path, len(payload), expected_sha256)
        )
        context[context_path] = (payload, mode)

    source_root = _compute_source_root_v1(source_payloads)
    if source_root != SCORER_SOURCE_ROOT_SHA256_V1:
        _fail("frozen 49-entry source root differs")

    dockerfile = _read_frozen_file_v1(
        DOCKERFILE_CONTEXT_PATH_V1,
        DOCKERFILE_SHA256_V1,
        expected_mode=0o644,
    )
    dockerignore = _read_frozen_file_v1(
        DOCKERIGNORE_CONTEXT_PATH_V1,
        DOCKERIGNORE_SHA256_V1,
        expected_mode=0o644,
    )
    expected_from = ("FROM python@" + BASE_IMAGE_DIGEST_V1 + "\n").encode("ascii")
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
    profile = _read_frozen_file_v1(
        PROFILE_PATH_V1,
        PROFILE_FILE_SHA256_V1,
        expected_mode=0o644,
    )
    scoring = next(
        payload
        for installed, payload in source_payloads
        if installed.endswith("/phase2b_unsealed_960_prediction_scoring_mechanics_v2.py")
    )
    if _sha256(scoring) != SCORING_SOURCE_SHA256_V1:
        _fail("frozen scoring source digest differs")
    seccomp = _read_frozen_file_v1(
        SECCOMP_PATH_V1,
        SECCOMP_SHA256_V1,
        expected_mode=0o644,
    )
    buildx = _read_stable_absolute_regular_v1(
        BUILDX_SOURCE_PATH_V1,
        expected_sha256=BUILDX_SHA256_V1,
        expected_size=BUILDX_SIZE_BYTES_V1,
        expected_mode=0o755,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
        maximum_bytes=MAXIMUM_BUILDX_BYTES_V1,
    )
    try:
        if type(json.loads(profile)) is not dict or type(json.loads(seccomp)) is not dict:
            _fail("profile or seccomp is not one JSON object")
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail("profile or seccomp JSON is malformed")

    context[DOCKERFILE_CONTEXT_PATH_V1] = (dockerfile, 0o644)
    context[DOCKERIGNORE_CONTEXT_PATH_V1] = (dockerignore, 0o644)
    archive = _deterministic_ustar_v1(context)
    context_entries = tuple(
        (path, len(payload), mode, _sha256(payload))
        for path, (payload, mode) in sorted(context.items())
    )
    return FrozenBuildInputsV1(
        context_bytes=archive,
        context_sha256=_sha256(archive),
        context_entries=context_entries,
        source_entries=tuple(source_entries),
        source_root_sha256=source_root,
        dockerfile_sha256=_sha256(dockerfile),
        dockerignore_sha256=_sha256(dockerignore),
        profile_sha256=_sha256(profile),
        scoring_source_sha256=_sha256(scoring),
        seccomp_sha256=_sha256(seccomp),
        seccomp_bytes=seccomp,
        buildx_sha256=_sha256(buildx),
        buildx_size_bytes=len(buildx),
        buildx_bytes=buildx,
    )


def _kill_and_reap_v1(process: subprocess.Popen[bytes]) -> None:
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


def _bounded_docker_run_v1(
    command: Sequence[str],
    *,
    stdin_payload: bytes | None,
    environment: Mapping[str, str],
    working_directory: Path,
    timeout_seconds: int,
    maximum_stdout_bytes: int,
    maximum_stderr_bytes: int,
) -> BoundedDockerResultV1:
    """Run one exact Docker command with bounded input, output, and deadline."""

    command_tuple = tuple(command)
    if (
        not command_tuple
        or command_tuple[:2] != (DOCKER_EXECUTABLE_V1, f"--host={DOCKER_HOST_V1}")
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
                _kill_and_reap_v1(process)
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
                        _kill_and_reap_v1(process)
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
                    _kill_and_reap_v1(process)
                    _fail(f"Docker {name} exceeded its fixed byte cap")
                target.extend(chunk)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_and_reap_v1(process)
            _fail("Docker command exceeded its fixed deadline")
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _kill_and_reap_v1(process)
            _fail("Docker command exceeded its fixed deadline")
        return BoundedDockerResultV1(
            command=command_tuple,
            returncode=returncode,
            stdout=bytes(stdout),
            stderr=bytes(stderr),
        )
    except Phase2BScorerImageBuilderErrorV1:
        if process is not None:
            _kill_and_reap_v1(process)
        raise
    except (OSError, ValueError) as exc:
        if process is not None:
            _kill_and_reap_v1(process)
        _fail(f"Docker process could not be executed: {type(exc).__name__}")
    finally:
        selector.close()


def _invoke_v1(
    runner: DockerRunnerV1,
    command: tuple[str, ...],
    *,
    stdin_payload: bytes | None,
    environment: dict[str, str],
    timeout_seconds: int,
    maximum_stdout_bytes: int,
    maximum_stderr_bytes: int,
) -> BoundedDockerResultV1:
    result = runner(
        command,
        stdin_payload=stdin_payload,
        environment=environment,
        working_directory=ROOT,
        timeout_seconds=timeout_seconds,
        maximum_stdout_bytes=maximum_stdout_bytes,
        maximum_stderr_bytes=maximum_stderr_bytes,
    )
    if type(result) is not BoundedDockerResultV1 or result.command != command:
        _fail("Docker runner returned a malformed observation")
    return result


def _canonical_json_v1(value: object) -> bytes:
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


def _decode_json_v1(payload: bytes, *, label: str) -> object:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        _fail(f"{label} is not one JSON value")


def _validate_image_inspect_v1(payload: bytes) -> tuple[str, str, tuple[str, ...]]:
    value = _decode_json_v1(payload, label="scorer image inspection")
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        _fail("scorer image inspection shape differs")
    row = value[0]
    config = row.get("Config")
    rootfs = row.get("RootFS")
    image_id = row.get("Id")
    if (
        type(config) is not dict
        or type(rootfs) is not dict
        or type(image_id) is not str
        or _DIGEST_V1.fullmatch(image_id) is None
        or row.get("Os") != "linux"
        or row.get("Architecture") != "amd64"
        or frozenset(config) != FROZEN_IMAGE_CONFIG_FIELDS_V1
        or config.get("ArgsEscaped") is not FROZEN_IMAGE_ARGS_ESCAPED_V1
        or config.get("User") != SCORER_RUNTIME_UID_GID_V1
        or config.get("Env") != list(FROZEN_IMAGE_ENVIRONMENT_V1)
        or config.get("Entrypoint") != list(FROZEN_ENTRYPOINT_V1)
        or config.get("Cmd") != list(FROZEN_COMMAND_V1)
        or config.get("WorkingDir") != "/opt/hegel"
        or config.get("Labels") != FROZEN_IMAGE_LABELS_V1
        or config.get("Healthcheck") is not None
        or config.get("Volumes") is not None
        or rootfs.get("Type") != "layers"
        or type(rootfs.get("Layers")) is not list
        or not rootfs["Layers"]
        or any(
            type(layer) is not str or _DIGEST_V1.fullmatch(layer) is None
            for layer in rootfs["Layers"]
        )
    ):
        _fail("scorer image platform/config/source binding differs")
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
    return image_id, _sha256(_canonical_json_v1(projection)), tuple(rootfs["Layers"])


def _validate_smoke_container_v1(
    payload: bytes,
    *,
    image_digest: str,
    container_name: str,
    expected_seccomp: Mapping[str, object],
) -> None:
    value = _decode_json_v1(payload, label="negative smoke container inspection")
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
    observed_seccomp = _decode_json_v1(
        seccomp_options[0][len("seccomp=") :].encode("utf-8"),
        label="negative smoke embedded seccomp",
    )
    if (
        not all(type(item) is dict for item in (state, config, host, network))
        or type(row.get("Id")) is not str
        or _CONTAINER_ID_V1.fullmatch(row["Id"]) is None
        or row.get("Name") != "/" + container_name
        or row.get("Image") != image_digest
        or row.get("Path") != FROZEN_ENTRYPOINT_V1[0]
        or row.get("Args") != list(FROZEN_COMMAND_V1)
        or state.get("Status") != "exited"
        or state.get("Running") is not False
        or state.get("Dead") is not False
        or state.get("OOMKilled") is not False
        or state.get("ExitCode") != 2
        or state.get("Error") != ""
        or config.get("Image") != image_digest
        or config.get("User") != SCORER_RUNTIME_UID_GID_V1
        or config.get("Env") != list(FROZEN_IMAGE_ENVIRONMENT_V1)
        or config.get("Entrypoint") != list(FROZEN_ENTRYPOINT_V1)
        or config.get("Cmd") != list(FROZEN_COMMAND_V1)
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


def _validate_cleanup_proof_v1(
    removed: BoundedDockerResultV1,
    inspected: BoundedDockerResultV1,
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


def _write_private_file_v1(
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
    except Phase2BScorerImageBuilderErrorV1:
        raise
    except OSError as exc:
        _fail(f"private runtime file write failed: {type(exc).__name__}")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _fsync_directory_v1(path: Path) -> None:
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


def _verify_private_buildx_copy_v1(path: Path, expected_payload: bytes) -> None:
    if (
        type(expected_payload) is not bytes
        or len(expected_payload) != BUILDX_SIZE_BYTES_V1
        or _sha256(expected_payload) != BUILDX_SHA256_V1
    ):
        _fail("frozen buildx snapshot differs")
    observed = _read_stable_absolute_regular_v1(
        path,
        expected_sha256=BUILDX_SHA256_V1,
        expected_size=BUILDX_SIZE_BYTES_V1,
        expected_mode=0o755,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
        maximum_bytes=MAXIMUM_BUILDX_BYTES_V1,
    )
    if observed != expected_payload:
        _fail("private buildx copy bytes differ")


def _build_command_v1() -> tuple[str, ...]:
    return (
        DOCKER_EXECUTABLE_V1,
        f"--host={DOCKER_HOST_V1}",
        "build",
        "--pull=false",
        "--network=none",
        "--no-cache",
        "--platform=linux/amd64",
        "--build-arg=SOURCE_DATE_EPOCH=0",
        f"--file={DOCKERFILE_CONTEXT_PATH_V1}",
        f"--tag={IMAGE_TAG_V1}",
        "-",
    )


def _smoke_command_v1(
    *,
    image_digest: str,
    container_name: str,
    seccomp_path: Path,
) -> tuple[str, ...]:
    return (
        DOCKER_EXECUTABLE_V1,
        f"--host={DOCKER_HOST_V1}",
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
        f"--user={SCORER_RUNTIME_UID_GID_V1}",
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


def _make_receipt_v1(
    *,
    inputs: FrozenBuildInputsV1,
    image_digest: str,
    image_config_sha256: str,
    rootfs_layers: tuple[str, ...],
) -> tuple[dict[str, object], bytes]:
    claims = {
        "actual_960_case_run_verified": False,
        "actual_960_recognizer_executed": False,
        "actual_960_scorer_executed": False,
        "actual_input_archive_supplied": False,
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
        "reveal_authority_verified": False,
        "runtime_image_config_verified": True,
        "scoring_performed": False,
        "signed_execution_registry_materialized": False,
        "signed_monotonic_time_materialized": False,
        "store_receipt_materialized": False,
    }
    body: dict[str, object] = {
        "schema_version": SCORER_RUNTIME_IMAGE_BUILD_RECEIPT_SCHEMA_V1,
        "claim_level": SCORER_RUNTIME_IMAGE_BUILD_CLAIM_LEVEL_V1,
        "disposition": SCORER_RUNTIME_IMAGE_BUILD_DISPOSITION_V1,
        "build": {
            "context_entry_count": len(inputs.context_entries),
            "context_mode": "EXACT_51_FILE_DETERMINISTIC_USTAR_STREAM",
            "context_sha256": inputs.context_sha256,
            "context_size_bytes": len(inputs.context_bytes),
            "docker_executable": DOCKER_EXECUTABLE_V1,
            "docker_host": DOCKER_HOST_V1,
            "dockerfile": DOCKERFILE_CONTEXT_PATH_V1,
            "dockerfile_run_instruction_count": 0,
            "image_tag": IMAGE_TAG_V1,
            "network_mode": "none",
            "no_cache": True,
            "platform": "linux/amd64",
            "pull": False,
            "source_date_epoch": 0,
        },
        "build_inputs": {
            "buildx_plugin_sha256": inputs.buildx_sha256,
            "buildx_plugin_size_bytes": inputs.buildx_size_bytes,
            "buildx_plugin_source_path": BUILDX_SOURCE_PATH_V1.as_posix(),
            "buildx_private_relative_path": BUILDX_PRIVATE_RELATIVE_PATH_V1,
            "buildx_version_stdout": BUILDX_VERSION_STDOUT_V1.decode("ascii").rstrip(
                "\n"
            ),
            "buildx_version_stdout_sha256": _sha256(BUILDX_VERSION_STDOUT_V1),
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
            "entrypoint_sha256": ENTRYPOINT_SHA256_V1,
            "minimal_initializer_sha256": INITIALIZER_SHA256_V1,
            "profile_canonical_sha256": PROFILE_CANONICAL_SHA256_V1,
            "profile_content_id": PROFILE_CONTENT_ID_V1,
            "profile_file_sha256": inputs.profile_sha256,
            "runtime_seccomp_sha256": inputs.seccomp_sha256,
            "scoring_source_sha256": inputs.scoring_source_sha256,
            "source_manifest": [
                {
                    "installed_path": installed,
                    "context_path": context,
                    "size_bytes": size,
                    "sha256": digest,
                }
                for installed, context, size, digest in inputs.source_entries
            ],
            "source_root_entry_count": len(inputs.source_entries),
            "source_root_sha256": inputs.source_root_sha256,
        },
        "image": {
            "architecture": "amd64",
            "args_escaped": FROZEN_IMAGE_ARGS_ESCAPED_V1,
            "command": list(FROZEN_COMMAND_V1),
            "declared_volumes": None,
            "entrypoint": list(FROZEN_ENTRYPOINT_V1),
            "environment": list(FROZEN_IMAGE_ENVIRONMENT_V1),
            "healthcheck": None,
            "image_config_sha256": image_config_sha256,
            "image_digest": image_digest,
            "labels": dict(FROZEN_IMAGE_LABELS_V1),
            "os": "linux",
            "rootfs_layers": list(rootfs_layers),
            "user": SCORER_RUNTIME_UID_GID_V1,
            "working_directory": "/opt/hegel",
        },
        "qualification": {
            "container_absence_after_cleanup_verified": True,
            "container_capabilities_dropped": "ALL",
            "container_no_new_privileges": True,
            "container_output_host_mounts": 0,
            "negative_smoke_exit_code": 2,
            "negative_smoke_stderr_sha256": _sha256(b""),
            "negative_smoke_stdout_sha256": _sha256(b""),
            "runtime_custom_seccomp_applied": True,
            "runtime_network_none": True,
            "runtime_root_read_only": True,
            "runtime_source_closure_exact": True,
            "runtime_uid_gid": SCORER_RUNTIME_UID_GID_V1,
        },
        "claims": claims,
    }
    body["receipt_id"] = (
        SCORER_RUNTIME_IMAGE_BUILD_RECEIPT_ID_PREFIX_V1
        + _sha256(
            SCORER_RUNTIME_IMAGE_BUILD_RECEIPT_DOMAIN_V1
            + _canonical_json_v1(body)
        )
    )
    receipt_bytes = _canonical_json_v1(body) + b"\n"
    return body, receipt_bytes


def build_and_qualify_scorer_image_v1(
    *,
    runner: DockerRunnerV1 = _bounded_docker_run_v1,
) -> tuple[dict[str, object], bytes]:
    """Build and negatively qualify the image; do not publish a receipt."""

    inputs = load_frozen_build_inputs_v1()
    expected_seccomp = _decode_json_v1(
        inputs.seccomp_bytes,
        label="frozen runtime seccomp",
    )
    if type(expected_seccomp) is not dict:
        _fail("frozen runtime seccomp is not one JSON object")
    with tempfile.TemporaryDirectory(
        prefix="hegel-phase2b-scorer-image-build-v1-",
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
        _write_private_file_v1(docker_config / "config.json", b"{}\n")
        private_buildx = docker_config / BUILDX_PRIVATE_RELATIVE_PATH_V1
        _write_private_file_v1(
            private_buildx,
            inputs.buildx_bytes,
            mode=0o755,
        )
        seccomp_path = runtime_root / "runtime-seccomp.json"
        _write_private_file_v1(seccomp_path, inputs.seccomp_bytes)
        _fsync_directory_v1(cli_plugins)
        _fsync_directory_v1(docker_config)
        _fsync_directory_v1(runtime_root)
        _verify_private_buildx_copy_v1(private_buildx, inputs.buildx_bytes)
        environment = {
            "BUILDKIT_PROGRESS": "plain",
            "DOCKER_BUILDKIT": "1",
            "DOCKER_CONFIG": docker_config.as_posix(),
            "DOCKER_HOST": DOCKER_HOST_V1,
            "HOME": runtime_root.as_posix(),
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "SOURCE_DATE_EPOCH": "0",
            "TMPDIR": runtime_tmp.as_posix(),
        }

        buildx_version_command = (
            DOCKER_EXECUTABLE_V1,
            f"--host={DOCKER_HOST_V1}",
            "buildx",
            "version",
        )
        buildx_version = _invoke_v1(
            runner,
            buildx_version_command,
            stdin_payload=None,
            environment=environment,
            timeout_seconds=INSPECT_TIMEOUT_SECONDS_V1,
            maximum_stdout_bytes=4_096,
            maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V1,
        )
        if (
            buildx_version.returncode != 0
            or buildx_version.stdout != BUILDX_VERSION_STDOUT_V1
            or buildx_version.stderr != b""
        ):
            _fail("frozen local buildx backend qualification failed")
        _verify_private_buildx_copy_v1(private_buildx, inputs.buildx_bytes)

        build_command = _build_command_v1()
        build = _invoke_v1(
            runner,
            build_command,
            stdin_payload=inputs.context_bytes,
            environment=environment,
            timeout_seconds=BUILD_TIMEOUT_SECONDS_V1,
            maximum_stdout_bytes=MAXIMUM_BUILD_STDOUT_BYTES_V1,
            maximum_stderr_bytes=MAXIMUM_BUILD_STDERR_BYTES_V1,
        )
        if build.returncode != 0:
            _fail("offline scorer image build failed")
        _verify_private_buildx_copy_v1(private_buildx, inputs.buildx_bytes)

        inspect_tag_command = (
            DOCKER_EXECUTABLE_V1,
            f"--host={DOCKER_HOST_V1}",
            "image",
            "inspect",
            IMAGE_TAG_V1,
        )
        inspected = _invoke_v1(
            runner,
            inspect_tag_command,
            stdin_payload=None,
            environment=environment,
            timeout_seconds=INSPECT_TIMEOUT_SECONDS_V1,
            maximum_stdout_bytes=MAXIMUM_INSPECT_STDOUT_BYTES_V1,
            maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V1,
        )
        if inspected.returncode != 0:
            _fail("built scorer image cannot be inspected")
        image_digest, image_config_sha256, rootfs_layers = (
            _validate_image_inspect_v1(inspected.stdout)
        )

        nonce = hashlib.sha256(
            (runtime_root.name + "\x00" + str(os.getpid())).encode("utf-8")
        ).hexdigest()[:16]
        container_name = f"hegel-phase2b-scorer-negative-smoke-v1-{nonce}"
        smoke_command = _smoke_command_v1(
            image_digest=image_digest,
            container_name=container_name,
            seccomp_path=seccomp_path,
        )
        smoke_error: BaseException | None = None
        try:
            smoke = _invoke_v1(
                runner,
                smoke_command,
                stdin_payload=None,
                environment=environment,
                timeout_seconds=SMOKE_TIMEOUT_SECONDS_V1,
                maximum_stdout_bytes=MAXIMUM_SMOKE_STDOUT_BYTES_V1,
                maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V1,
            )
            if smoke.returncode != 2 or smoke.stdout != b"" or smoke.stderr != b"":
                _fail("negative scorer smoke did not produce exact silent exit 2")
            inspect_container_command = (
                DOCKER_EXECUTABLE_V1,
                f"--host={DOCKER_HOST_V1}",
                "container",
                "inspect",
                container_name,
            )
            smoke_inspect = _invoke_v1(
                runner,
                inspect_container_command,
                stdin_payload=None,
                environment=environment,
                timeout_seconds=INSPECT_TIMEOUT_SECONDS_V1,
                maximum_stdout_bytes=MAXIMUM_INSPECT_STDOUT_BYTES_V1,
                maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V1,
            )
            if smoke_inspect.returncode != 0:
                _fail("negative scorer smoke container cannot be inspected")
            _validate_smoke_container_v1(
                smoke_inspect.stdout,
                image_digest=image_digest,
                container_name=container_name,
                expected_seccomp=expected_seccomp,
            )
        except BaseException as exc:
            smoke_error = exc
        finally:
            remove_command = (
                DOCKER_EXECUTABLE_V1,
                f"--host={DOCKER_HOST_V1}",
                "container",
                "rm",
                "--force",
                "--volumes",
                container_name,
            )
            cleanup = _invoke_v1(
                runner,
                remove_command,
                stdin_payload=None,
                environment=environment,
                timeout_seconds=CLEANUP_TIMEOUT_SECONDS_V1,
                maximum_stdout_bytes=MAXIMUM_SMOKE_STDOUT_BYTES_V1,
                maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V1,
            )
            absence_command = (
                DOCKER_EXECUTABLE_V1,
                f"--host={DOCKER_HOST_V1}",
                "container",
                "inspect",
                container_name,
            )
            absence = _invoke_v1(
                runner,
                absence_command,
                stdin_payload=None,
                environment=environment,
                timeout_seconds=INSPECT_TIMEOUT_SECONDS_V1,
                maximum_stdout_bytes=MAXIMUM_INSPECT_STDOUT_BYTES_V1,
                maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V1,
            )
            _validate_cleanup_proof_v1(
                cleanup,
                absence,
                container_name=container_name,
                allow_already_absent=smoke_error is not None,
            )
        if smoke_error is not None:
            if isinstance(smoke_error, Phase2BScorerImageBuilderErrorV1):
                raise smoke_error
            _fail(f"negative scorer smoke failed: {type(smoke_error).__name__}")

        inspect_digest_command = (
            DOCKER_EXECUTABLE_V1,
            f"--host={DOCKER_HOST_V1}",
            "image",
            "inspect",
            image_digest,
        )
        reinspected = _invoke_v1(
            runner,
            inspect_digest_command,
            stdin_payload=None,
            environment=environment,
            timeout_seconds=INSPECT_TIMEOUT_SECONDS_V1,
            maximum_stdout_bytes=MAXIMUM_INSPECT_STDOUT_BYTES_V1,
            maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V1,
        )
        if reinspected.returncode != 0:
            _fail("qualified scorer image cannot be re-inspected by digest")
        second_digest, second_config_sha256, second_layers = (
            _validate_image_inspect_v1(reinspected.stdout)
        )
        if (
            second_digest != image_digest
            or second_config_sha256 != image_config_sha256
            or second_layers != rootfs_layers
        ):
            _fail("scorer image identity changed across negative qualification")

    return _make_receipt_v1(
        inputs=inputs,
        image_digest=image_digest,
        image_config_sha256=image_config_sha256,
        rootfs_layers=rootfs_layers,
    )


def _canonical_absolute_receipt_path_v1(value: object) -> Path:
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


def _open_directory_componentwise_v1(path: Path) -> int:
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
    except Phase2BScorerImageBuilderErrorV1:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        _fail(f"receipt parent cannot be opened: {type(exc).__name__}")


def assert_receipt_target_absent_v1(path: Path) -> None:
    parent_fd = _open_directory_componentwise_v1(path.parent)
    try:
        try:
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        _fail("receipt target already exists")
    finally:
        os.close(parent_fd)


def publish_receipt_no_replace_v1(path: Path, payload: bytes) -> None:
    """Durably publish one new receipt through hard-link no-replace."""

    if type(payload) is not bytes or not payload or not payload.endswith(b"\n"):
        _fail("receipt payload differs")
    parent_fd = _open_directory_componentwise_v1(path.parent)
    pending = ".phase2b-scorer-image-" + _sha256(payload)[:24] + ".pending"
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
    except Phase2BScorerImageBuilderErrorV1:
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


def _parse_cli_v1(argv: Sequence[str] | None) -> Path:
    raw = tuple(sys.argv[1:] if argv is None else argv)
    if len(raw) != 2 or raw[0] != "--receipt" or type(raw[1]) is not str:
        _fail("usage: build_phase2b_scorer_runtime_image_v1.py --receipt /absolute/new.json")
    return _canonical_absolute_receipt_path_v1(raw[1])


def main(argv: Sequence[str] | None = None) -> int:
    try:
        receipt_path = _parse_cli_v1(argv)
        assert_receipt_target_absent_v1(receipt_path)
        _receipt, payload = build_and_qualify_scorer_image_v1()
        publish_receipt_no_replace_v1(receipt_path, payload)
    except Exception:
        sys.stderr.write("phase2b scorer image build/qualification failed\n")
        sys.stderr.flush()
        return 2
    return 0


__all__ = (
    "BASE_IMAGE_DIGEST_V1",
    "BoundedDockerResultV1",
    "BUILDX_PRIVATE_RELATIVE_PATH_V1",
    "BUILDX_SHA256_V1",
    "BUILDX_SIZE_BYTES_V1",
    "BUILDX_SOURCE_PATH_V1",
    "BUILDX_VERSION_STDOUT_V1",
    "DOCKERFILE_CONTEXT_PATH_V1",
    "DOCKERFILE_SHA256_V1",
    "DOCKERIGNORE_CONTEXT_PATH_V1",
    "DOCKERIGNORE_SHA256_V1",
    "DOCKER_EXECUTABLE_V1",
    "DOCKER_HOST_V1",
    "FROZEN_IMAGE_ARGS_ESCAPED_V1",
    "FROZEN_IMAGE_CONFIG_FIELDS_V1",
    "FROZEN_SOURCE_MANIFEST_V1",
    "IMAGE_TAG_V1",
    "Phase2BScorerImageBuilderErrorV1",
    "PROFILE_FILE_SHA256_V1",
    "SCORER_RUNTIME_IMAGE_BUILD_RECEIPT_DOMAIN_V1",
    "SCORER_RUNTIME_IMAGE_BUILD_RECEIPT_ID_PREFIX_V1",
    "SCORER_RUNTIME_IMAGE_BUILD_RECEIPT_SCHEMA_V1",
    "SCORER_SOURCE_ROOT_SHA256_V1",
    "SCORING_SOURCE_SHA256_V1",
    "SECCOMP_SHA256_V1",
    "assert_receipt_target_absent_v1",
    "build_and_qualify_scorer_image_v1",
    "load_frozen_build_inputs_v1",
    "main",
    "publish_receipt_no_replace_v1",
)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
