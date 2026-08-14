"""Fail-closed local Docker executor for Phase-2B scorer mechanics.

The executor pins one canonical scorer bundle and one empty publication
directory, validates the V2 scorer actor profile and a content-addressed local
image, runs the fixed scorer CLI in a private no-network container, extracts
the two reports from container tmpfs, removes the container, independently
replays the bundle on the host, and only then publishes byte-identical reports.

Its receipt is a local mechanics observation.  It is not a custodian or reveal
attestation, an actual 960-case score, a formal gate decision, effect evidence,
or C1 exit evidence.  A caller with control of the host or Docker daemon remains
outside this local threat model.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field, fields
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import signal
import stat
import struct
import subprocess
import time
from types import MappingProxyType
from typing import Callable, Final, Mapping, NoReturn, Sequence

from .hashing import stable_hash
from .phase2b_c1_container_actor_profile_v2 import (
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256,
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID,
    Phase2BC1ContainerActorProfileSnapshotV2,
    load_phase2b_c1_container_actor_profile_v2,
)
from .phase2b_scorer_bundle_v1 import (
    MAXIMUM_PHASE2B_SCORER_BUNDLE_BYTES_V1,
    PHASE2B_SCORER_AVAILABLE_GATE_MECHANICS_REPORT_V1_FILENAME,
    PHASE2B_SCORER_BUNDLE_V1_FILENAME,
    PHASE2B_SCORER_SCORE_MECHANICS_REPORT_V1_FILENAME,
    GATE_REPORT_DOMAIN_V1,
    GATE_REPORT_ID_PREFIX_V1,
    SCORE_REPORT_DOMAIN_V1,
    SCORE_REPORT_ID_PREFIX_V1,
    DecodedPhase2BScorerBundleV1,
    EvaluatedPhase2BScorerBundleV1,
    decode_phase2b_scorer_bundle_v1,
    evaluate_phase2b_scorer_bundle_v1,
)
from .phase2b_scorer_runtime_cli_v1 import (
    MAXIMUM_SCORER_REPORT_BYTES_V1,
    MAXIMUM_SCORER_RUNTIME_STDOUT_BYTES_V1,
    SCORER_RUNTIME_CLAIM_LEVEL_V1,
    SCORER_RUNTIME_COMPLETE_REASON_V1,
    SCORER_RUNTIME_RECEIPT_ID_PREFIX_V1,
    SCORER_RUNTIME_RECEIPT_SCHEMA_VERSION_V1,
)
from .phase3_local_runtime_v1 import (
    DEFAULT_DOCKER_EXECUTABLE,
    DEFAULT_DOCKER_SOCKET,
    LOCAL_DOCKER_HOST,
    LinuxLocalTemporaryDirectoryV1,
    LocalDockerControlPlaneV1,
    Phase3LocalRuntimeError,
    prepare_local_docker_control_plane_v1,
    validate_linux_local_host_path_v1,
)


LOCAL_SCORER_EXECUTION_RECEIPT_SCHEMA_V1: Final = (
    "hegel-machine-phase2b-scorer-local-execution-receipt/1"
)
LOCAL_SCORER_EXECUTION_COMPLETE_REASON_V1: Final = (
    "phase2b_scorer_local_docker_mechanics_complete_not_actual_scoring"
)
LOCAL_SCORER_EXECUTION_DISPOSITION_V1: Final = (
    "LOCAL_SCORER_MECHANICS_COMPLETE_NOT_AUTHORITY"
)

FAIL_PREFLIGHT: Final = "FAIL_PHASE2B_SCORER_HOST_PREFLIGHT"
FAIL_HOST_PATH: Final = "FAIL_PHASE2B_SCORER_HOST_PATH"
FAIL_BUNDLE_BINDING: Final = "FAIL_PHASE2B_SCORER_BUNDLE_BINDING"
FAIL_PROFILE_BINDING: Final = "FAIL_PHASE2B_SCORER_PROFILE_BINDING"
FAIL_SECCOMP_BINDING: Final = "FAIL_PHASE2B_SCORER_SECCOMP_BINDING"
FAIL_DOCKER_CONTROL_PLANE: Final = "FAIL_PHASE2B_SCORER_DOCKER_CONTROL_PLANE"
FAIL_IMAGE_BINDING: Final = "FAIL_PHASE2B_SCORER_IMAGE_BINDING"
FAIL_DOCKER_EXECUTION: Final = "FAIL_PHASE2B_SCORER_DOCKER_EXECUTION"
FAIL_DOCKER_TIMEOUT: Final = "FAIL_PHASE2B_SCORER_DOCKER_TIMEOUT"
FAIL_DOCKER_CAPTURE: Final = "FAIL_PHASE2B_SCORER_DOCKER_CAPTURE"
FAIL_OUTPUT_BINDING: Final = "FAIL_PHASE2B_SCORER_OUTPUT_BINDING"

SCORER_TIMEOUT_SECONDS_V1: Final = 3_600
IMAGE_INSPECT_TIMEOUT_SECONDS_V1: Final = 60
MAXIMUM_IMAGE_INSPECT_STDOUT_BYTES_V1: Final = 1_048_576
MAXIMUM_DOCKER_STDERR_BYTES_V1: Final = 8_192
MAXIMUM_DOCKER_STDOUT_BYTES_V1: Final = 65_536
MAXIMUM_DOCKER_RESULT_WIRE_BYTES_V1: Final = (
    2 * MAXIMUM_SCORER_REPORT_BYTES_V1 + 128
)
MAXIMUM_SECCOMP_BYTES_V1: Final = 65_536
MAXIMUM_DOCKER_EXECUTABLE_BYTES_V1: Final = 134_217_728
MAXIMUM_PATH_BYTES_V1: Final = 4_096
MAXIMUM_PATH_COMPONENTS_V1: Final = 64
MAXIMUM_NAMESPACE_NAMES_V1: Final = 8
_READ_CHUNK_BYTES_V1: Final = 1 << 20
_EXACT_PATH_TYPE_V1: Final = type(Path("/"))

SCORER_RUNTIME_UID_GID_V1: Final = "65534:65534"
SCORER_RUNTIME_IMAGE_TITLE_V1: Final = (
    "Hegel Machine Phase2B V1 scorer runtime"
)
SCORER_RUNTIME_IMAGE_DESCRIPTION_V1: Final = (
    "Non-authoritative local Docker scorer mechanics; no actual scoring, "
    "formal gate, or C1 evidence"
)
SCORER_RUNTIME_IMAGE_ARGS_ESCAPED_V1: Final = True
SCORER_RUNTIME_IMAGE_CONFIG_FIELDS_V1: Final = frozenset(
    {"ArgsEscaped", "Cmd", "Entrypoint", "Env", "Labels", "User", "WorkingDir"}
)
SCORER_RUNTIME_BASE_DIGEST_V1: Final = (
    "sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"
)
SCORING_MECHANICS_SOURCE_SHA256_V1: Final = (
    "c3c12e9cd72a930b9ca8667aab22c4eb3675ceab6249b0b26ba940bf794e7f11"
)
OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V1: Final = (
    "phase2b_c1_owner_accepted_container_authority_requirement_v1_"
    "53631d563f8da8f26c7202b7dd0681be5423fe2ca5c15e9a9f5deaa2920d9940"
)
ACTUAL_960_CAMPAIGN_ID_PREFIX_V2: Final = (
    "phase2b_actual_unsealed_960_campaign_v2_"
)
ACTUAL_960_ATTEMPT_ID_PREFIX_V2: Final = (
    "phase2b_actual_unsealed_960_attempt_v2_"
)
OWNER_ACCEPTED_EXECUTION_REGISTRY_ID_PREFIX_V1: Final = (
    "phase2b_c1_owner_accepted_execution_registry_v1_"
)
SCORER_SOURCE_ROOT_DOMAIN_V1: Final = (
    b"HEGEL/PHASE2B/SCORER_RUNTIME_SOURCE_ROOT/V1\x00"
)
SCORER_SOURCE_ROOT_ORDERED_ENTRY_PATHS_V1: Final = (
    "/opt/hegel/bin/phase2b-scorer",
    "/opt/hegel/src/hegel_machine/__init__.py",
    "/opt/hegel/src/hegel_machine/bootstrap.py",
    "/opt/hegel/src/hegel_machine/hashing.py",
    "/opt/hegel/src/hegel_machine/laws.py",
    "/opt/hegel/src/hegel_machine/milestones.py",
    "/opt/hegel/src/hegel_machine/ontology.py",
    "/opt/hegel/src/hegel_machine/phase2b_actual_unsealed_960_external_custodian_checkpoint_verifier_mechanics_v2.py",
    "/opt/hegel/src/hegel_machine/phase2b_actual_unsealed_960_local_attempt_ledger_mechanics_v2.py",
    "/opt/hegel/src/hegel_machine/phase2b_actual_unsealed_960_replay_admission_evidence_contract_v2.py",
    "/opt/hegel/src/hegel_machine/phase2b_actual_unsealed_960_replay_input_contract_v2.py",
    "/opt/hegel/src/hegel_machine/phase2b_actual_unsealed_960_signed_attempt_chain_mechanics_v2.py",
    "/opt/hegel/src/hegel_machine/phase2b_adapter.py",
    "/opt/hegel/src/hegel_machine/phase2b_c1_container_actor_profile_v1.py",
    "/opt/hegel/src/hegel_machine/phase2b_c1_container_actor_profile_v2.py",
    "/opt/hegel/src/hegel_machine/phase2b_covert_audit_v1.py",
    "/opt/hegel/src/hegel_machine/phase2b_exact_bridge_v1.py",
    "/opt/hegel/src/hegel_machine/phase2b_exact_derived_witness_bridge_v1.py",
    "/opt/hegel/src/hegel_machine/phase2b_exact_transform_semantics_v1.py",
    "/opt/hegel/src/hegel_machine/phase2b_formal_unsealed_prediction_scoring_contract_v2.py",
    "/opt/hegel/src/hegel_machine/phase2b_freeze_v1.py",
    "/opt/hegel/src/hegel_machine/phase2b_projection_compiler.py",
    "/opt/hegel/src/hegel_machine/phase2b_protocol.py",
    "/opt/hegel/src/hegel_machine/phase2b_recognizer_input_archive_v1.py",
    "/opt/hegel/src/hegel_machine/phase2b_recognizer_input_archive_v2.py",
    "/opt/hegel/src/hegel_machine/phase2b_recognizer_prediction_archive_v1.py",
    "/opt/hegel/src/hegel_machine/phase2b_recognizer_prediction_archive_v2.py",
    "/opt/hegel/src/hegel_machine/phase2b_recognizer_prediction_v2.py",
    "/opt/hegel/src/hegel_machine/phase2b_recognizer_runtime_cli_v2.py",
    "/opt/hegel/src/hegel_machine/phase2b_runner.py",
    "/opt/hegel/src/hegel_machine/phase2b_scorer_bundle_v1.py",
    "/opt/hegel/src/hegel_machine/phase2b_scorer_runtime_cli_v1.py",
    "/opt/hegel/src/hegel_machine/phase2b_selector.py",
    "/opt/hegel/src/hegel_machine/phase2b_strict_recognizer_cli_v2.py",
    "/opt/hegel/src/hegel_machine/phase2b_trusted_wire_batch_v1.py",
    "/opt/hegel/src/hegel_machine/phase2b_trusted_wire_batch_v2.py",
    "/opt/hegel/src/hegel_machine/phase2b_trusted_wire_typed_authority_v1.py",
    "/opt/hegel/src/hegel_machine/phase2b_trusted_wire_typed_authority_v2.py",
    "/opt/hegel/src/hegel_machine/phase2b_trusted_wire_typed_replay_v1.py",
    "/opt/hegel/src/hegel_machine/phase2b_trusted_wire_typed_replay_v2.py",
    "/opt/hegel/src/hegel_machine/phase2b_trusted_wire_v1.py",
    "/opt/hegel/src/hegel_machine/phase2b_uncertainty_compiler.py",
    "/opt/hegel/src/hegel_machine/phase2b_unsealed_960_available_gate_mechanics_v2.py",
    "/opt/hegel/src/hegel_machine/phase2b_unsealed_960_prediction_scoring_mechanics_v2.py",
    "/opt/hegel/src/hegel_machine/phase2b_unsealed_prediction_evaluator_v1.py",
    "/opt/hegel/src/hegel_machine/phase2b_unsealed_prediction_evaluator_v2.py",
    "/opt/hegel/src/hegel_machine/phase2b_wire.py",
    "/opt/hegel/src/hegel_machine/phase3_dsl_v1.py",
    "/opt/hegel/src/hegel_machine/schema.py",
)
SCORER_SOURCE_ROOT_SHA256_V1: Final = (
    "34f4a2404d00d46cede832d34ca9713253da93f1becef4a1d302c1411d6bd5e7"
)
_EXPECTED_METRIC_SURFACE_V1: Final = (
    ("answerable_count", 240),
    ("family_exact_accuracy", 240),
    ("binding_exact_accuracy", 240),
    ("scale_set_accuracy", 240),
    ("unique_scale_accuracy", 228),
    ("joint_exact_accuracy", 240),
    ("abstention_specificity", 228),
    ("nonidentifiability_abstention_accuracy", 96),
    ("set_valued_answer_accuracy", 12),
)
_EXPECTED_OVERALL_GATE_NAMES_V1: Final = (
    "family_exact",
    "binding_exact",
    "scale_set_accuracy",
    "joint_exact",
    "hard_negative_rejection",
    "binding_counterfactual_rejection",
    "scale_counterfactual_rejection",
    "sign_or_invariant_break_rejection",
    "abstention_specificity",
    "nonidentifiable_scale_abstention",
)
_EXPECTED_UNAVAILABLE_GATE_NAMES_V1: Final = (
    "fail_closed_rate",
    "preservation_consistency",
)
_EXPECTED_SLICE_GATE_SURFACE_V1: Final = tuple(
    (metric_name, "family", family_id)
    for family_id in (
        "F01_symmetry_equivariance",
        "F02_monotonicity_order",
        "F03_conservation_balance",
        "F04_additivity_complementarity",
        "F05_locality_composition",
        "F06_negative_feedback_stability",
    )
    for metric_name in (
        "answerable_joint_exact",
        "all_control_rejection",
        "abstention_specificity",
    )
) + tuple(
    (metric_name, "scale", scale_id)
    for scale_id in ("S01", "S02")
    for metric_name in (
        "answerable_joint_exact",
        "all_control_rejection",
        "abstention_specificity",
    )
)
MAXIMUM_SCORER_SOURCE_FILE_BYTES_V1: Final = 16 * 1024 * 1024
SCORER_SECCOMP_SHA256_V1: Final = (
    "d1284e4731683b73352ecdd1577704ea87aa0b5c582b7b00757c3db4d2c950ca"
)
FROZEN_SCORER_ENTRYPOINT_V1: Final = (
    "/opt/hegel/bin/phase2b-scorer",
    "--input",
    "/phase2b/input/phase2b_scorer_bundle_v1.bin",
    "--output",
    "/phase2b/output",
)
_IMAGE_ENVIRONMENT_V1: Final = (
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
_CONTAINER_ENVIRONMENT_V1: Final = (
    "PYTHONHASHSEED=0",
    "PYTHONDONTWRITEBYTECODE=1",
    "PYTHONPATH=/opt/hegel/src",
    "HOME=/nonexistent",
    "NO_PROXY=*",
    *_IMAGE_ENVIRONMENT_V1[:5],
)
_STAGED_INPUT_DIRECTORY_V1: Final = "scorer-input"
_STAGED_SECCOMP_FILENAME_V1: Final = "scorer-seccomp.json"
_CONTAINER_SUPERVISOR_ENTRYPOINT_V1: Final = "/usr/local/bin/python3"
_CONTAINER_RESULT_WIRE_MAGIC_V1: Final = b"HEGEL-PHASE2B-SCORER-RESULT-V1\x00"
_CONTAINER_SUPERVISOR_CODE_V1: Final = "\n".join(
    (
        "import sys, time",
        "from hegel_machine.phase2b_scorer_runtime_cli_v1 import main",
        "exit_code = main()",
        "sys.stdout.flush()",
        "sys.stderr.flush()",
        "if exit_code != 0:",
        "    raise SystemExit(exit_code)",
        "while True:",
        "    time.sleep(3600)",
    )
)
_CONTAINER_RESULT_EXTRACTOR_CODE_V1: Final = "\n".join(
    (
        "import os, stat, struct, sys, time",
        f"score_name = {PHASE2B_SCORER_SCORE_MECHANICS_REPORT_V1_FILENAME!r}",
        f"gate_name = {PHASE2B_SCORER_AVAILABLE_GATE_MECHANICS_REPORT_V1_FILENAME!r}",
        "allowed = {score_name, gate_name, score_name + '.next', gate_name + '.next'}",
        "deadline = time.monotonic() + 3600",
        "while True:",
        "    directory = os.open('/phase2b/output', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)",
        "    names = sorted(os.listdir(directory))",
        "    if not set(names) <= allowed:",
        "        raise SystemExit(90)",
        "    if names == sorted([score_name, gate_name]):",
        "        break",
        "    os.close(directory)",
        "    if time.monotonic() >= deadline:",
        "        raise SystemExit(91)",
        "    time.sleep(0.05)",
        "def read_exact(name):",
        "    descriptor = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)",
        "    try:",
        "        before = os.fstat(descriptor)",
        f"        if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o400 or before.st_uid != 65534 or before.st_gid != 65534 or before.st_nlink != 1 or not 2 <= before.st_size <= {MAXIMUM_SCORER_REPORT_BYTES_V1}:",
        "            raise SystemExit(92)",
        "        chunks = []",
        "        total = 0",
        f"        while total <= {MAXIMUM_SCORER_REPORT_BYTES_V1}:",
        f"            chunk = os.read(descriptor, min(1048576, {MAXIMUM_SCORER_REPORT_BYTES_V1} + 1 - total))",
        "            if not chunk:",
        "                break",
        "            chunks.append(chunk)",
        "            total += len(chunk)",
        "        after = os.fstat(descriptor)",
        "        fingerprint = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_uid, row.st_gid, row.st_nlink, row.st_size, row.st_mtime_ns, row.st_ctime_ns)",
        "        payload = b''.join(chunks)",
        "        if fingerprint(before) != fingerprint(after) or len(payload) != before.st_size:",
        "            raise SystemExit(93)",
        "        return payload",
        "    finally:",
        "        os.close(descriptor)",
        "score = read_exact(score_name)",
        "gate = read_exact(gate_name)",
        "if sorted(os.listdir(directory)) != sorted([score_name, gate_name]):",
        "    raise SystemExit(94)",
        "os.close(directory)",
        f"sys.stdout.buffer.write({_CONTAINER_RESULT_WIRE_MAGIC_V1!r} + struct.pack('>QQ', len(score), len(gate)) + score + gate)",
        "sys.stdout.buffer.flush()",
    )
)

_LOWER_SHA256_V1 = re.compile(r"[0-9a-f]{64}")
_DIGEST_REF_V1 = re.compile(r"sha256:[0-9a-f]{64}")
_CONTAINER_ID_V1 = re.compile(r"[0-9a-f]{64}")
_CONTAINER_NAME_V1 = re.compile(r"hegel-phase2b-scorer-[0-9a-f]{24}")
_RECEIPT_ISSUE_TOKEN_V1 = object()


class Phase2BScorerHostExecutorErrorV1(RuntimeError):
    """Stable fail-closed scorer host error."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BScorerHostExecutorErrorV1(code, detail)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def compute_scorer_source_root_sha256_v1(
    payloads: Mapping[str, bytes],
) -> str:
    """Compute the frozen installed-path/byte root for the scorer image."""

    if type(payloads) not in {dict, MappingProxyType}:
        _fail(FAIL_PREFLIGHT, "scorer source-root payload mapping type differs")
    if tuple(payloads) != SCORER_SOURCE_ROOT_ORDERED_ENTRY_PATHS_V1:
        _fail(FAIL_PREFLIGHT, "scorer source-root path order or field set differs")
    preimage = bytearray(SCORER_SOURCE_ROOT_DOMAIN_V1)
    preimage.extend(struct.pack(">I", len(payloads)))
    for path, payload in payloads.items():
        if type(path) is not str or type(payload) is not bytes:
            _fail(FAIL_PREFLIGHT, "scorer source-root entry type differs")
        try:
            encoded_path = path.encode("ascii")
        except UnicodeEncodeError as exc:
            _fail(FAIL_PREFLIGHT, f"scorer source path is not ASCII: {type(exc).__name__}")
        if not 1 <= len(payload) <= MAXIMUM_SCORER_SOURCE_FILE_BYTES_V1:
            _fail(FAIL_PREFLIGHT, "scorer source-root entry byte bound differs")
        preimage.extend(struct.pack(">I", len(encoded_path)))
        preimage.extend(encoded_path)
        preimage.extend(struct.pack(">Q", len(payload)))
        preimage.extend(hashlib.sha256(payload).digest())
    return hashlib.sha256(preimage).hexdigest()


def _require_sha256(value: object, label: str) -> None:
    if type(value) is not str or _LOWER_SHA256_V1.fullmatch(value) is None:
        _fail(FAIL_PREFLIGHT, f"{label} must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class Phase2BScorerRuntimeLimitsV1:
    cpus: str = "2.0"
    memory: str = "2g"
    memory_swap: str = "2g"
    pids_limit: int = 128
    nofile_limit: str = "64:64"
    nproc_limit: str = "128:128"
    output_tmpfs_bytes: int = 536_870_912
    timeout_seconds: int = SCORER_TIMEOUT_SECONDS_V1

    def __post_init__(self) -> None:
        expected = ("2.0", "2g", "2g", 128, "64:64", "128:128", 536_870_912, 3_600)
        observed = tuple(getattr(self, item.name) for item in fields(type(self)))
        if observed != expected or any(
            type(value) is not type(reference)
            for value, reference in zip(observed, expected, strict=True)
        ):
            raise ValueError("scorer runtime limits differ from frozen values")


@dataclass(frozen=True, slots=True)
class Phase2BScorerRunSpecV1:
    authority_requirement_id: str
    actor_profile_content_id: str
    actor_profile_canonical_sha256: str
    campaign_id: str
    attempt_id: str
    execution_registry_id: str
    scorer_image_digest: str
    scorer_source_root_sha256: str
    protocol_id: str
    exact_freeze_id: str
    execution_freeze_manifest_id: str
    scorer_run_manifest_id: str
    bundle_sha256: str
    bundle_host_path: str
    output_host_directory: str
    repository_host_directory: str
    seccomp_profile_host_path: str
    seccomp_profile_sha256: str
    scoring_mechanics_source_sha256: str = SCORING_MECHANICS_SOURCE_SHA256_V1
    entrypoint: tuple[str, ...] = FROZEN_SCORER_ENTRYPOINT_V1
    limits: Phase2BScorerRuntimeLimitsV1 = field(
        default_factory=Phase2BScorerRuntimeLimitsV1
    )

    def __post_init__(self) -> None:
        text_fields = tuple(
            item.name
            for item in fields(type(self))
            if item.name not in {"entrypoint", "limits"}
        )
        if any(type(getattr(self, name)) is not str for name in text_fields):
            raise TypeError("scorer run-spec text fields must be exact strings")
        if (
            self.authority_requirement_id
            != OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V1
            or self.actor_profile_content_id
            != PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
            or self.actor_profile_canonical_sha256
            != PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
            or not self.campaign_id.startswith(ACTUAL_960_CAMPAIGN_ID_PREFIX_V2)
            or _LOWER_SHA256_V1.fullmatch(
                self.campaign_id.removeprefix(ACTUAL_960_CAMPAIGN_ID_PREFIX_V2)
            )
            is None
            or not self.attempt_id.startswith(ACTUAL_960_ATTEMPT_ID_PREFIX_V2)
            or _LOWER_SHA256_V1.fullmatch(
                self.attempt_id.removeprefix(ACTUAL_960_ATTEMPT_ID_PREFIX_V2)
            )
            is None
            or not self.execution_registry_id.startswith(
                OWNER_ACCEPTED_EXECUTION_REGISTRY_ID_PREFIX_V1
            )
            or _LOWER_SHA256_V1.fullmatch(
                self.execution_registry_id.removeprefix(
                    OWNER_ACCEPTED_EXECUTION_REGISTRY_ID_PREFIX_V1
                )
            )
            is None
            or not self.protocol_id
            or not self.exact_freeze_id
            or not self.execution_freeze_manifest_id
            or not self.scorer_run_manifest_id
            or _DIGEST_REF_V1.fullmatch(self.scorer_image_digest) is None
            or self.scorer_source_root_sha256 != SCORER_SOURCE_ROOT_SHA256_V1
            or self.scoring_mechanics_source_sha256
            != SCORING_MECHANICS_SOURCE_SHA256_V1
            or _LOWER_SHA256_V1.fullmatch(self.bundle_sha256) is None
            or self.seccomp_profile_sha256 != SCORER_SECCOMP_SHA256_V1
            or type(self.entrypoint) is not tuple
            or self.entrypoint != FROZEN_SCORER_ENTRYPOINT_V1
            or type(self.limits) is not Phase2BScorerRuntimeLimitsV1
        ):
            raise ValueError("scorer run-spec frozen binding differs")

    @property
    def run_spec_id(self) -> str:
        return stable_hash(self, prefix="phase2b_scorer_run_spec_v1_")


def _required_open_flag_v1(name: str, code: str) -> int:
    value = getattr(os, name, None)
    if type(value) is not int or value == 0:
        _fail(code, f"host kernel lacks required {name}")
    return value


def _directory_flags_v1(code: str) -> int:
    return (
        os.O_RDONLY
        | _required_open_flag_v1("O_DIRECTORY", code)
        | _required_open_flag_v1("O_NOFOLLOW", code)
        | _required_open_flag_v1("O_NONBLOCK", code)
        | _required_open_flag_v1("O_CLOEXEC", code)
    )


def _regular_flags_v1(code: str) -> int:
    return (
        os.O_RDONLY
        | _required_open_flag_v1("O_NOFOLLOW", code)
        | _required_open_flag_v1("O_NONBLOCK", code)
        | _required_open_flag_v1("O_CLOEXEC", code)
        | getattr(os, "O_NOCTTY", 0)
    )


def _exact_absolute_path_v1(value: object, *, label: str, code: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        _fail(code, f"{label} is not exact text")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        _fail(code, f"{label} is not UTF-8: {type(exc).__name__}")
    path = Path(value)
    if (
        len(encoded) > MAXIMUM_PATH_BYTES_V1
        or len(path.parts) > MAXIMUM_PATH_COMPONENTS_V1
        or not path.is_absolute()
        or Path(os.path.abspath(os.fspath(path))) != path
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        _fail(code, f"{label} is not one normalized absolute path")
    return path


def _metadata_identity_v1(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _directory_identity_v1(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
    )


@dataclass(slots=True)
class _PinnedDirectoryV1:
    path: Path
    descriptor: int
    component_identities: tuple[tuple[str, tuple[int, ...]], ...]

    def close(self) -> None:
        descriptor, self.descriptor = self.descriptor, -1
        if descriptor >= 0:
            os.close(descriptor)


def _open_componentwise_directory_v1(
    path: Path,
    *,
    label: str,
    code: str = FAIL_HOST_PATH,
) -> _PinnedDirectoryV1:
    descriptor: int | None = None
    identities: list[tuple[str, tuple[int, ...]]] = []
    try:
        descriptor = os.open("/", _directory_flags_v1(code))
        root_metadata = os.fstat(descriptor)
        identities.append(("/", _directory_identity_v1(root_metadata)))
        for component in path.parts[1:]:
            next_descriptor = os.open(
                component,
                _directory_flags_v1(code),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                _fail(code, f"{label} component is not a directory")
            identities.append((component, _directory_identity_v1(metadata)))
        return _PinnedDirectoryV1(path, descriptor, tuple(identities))
    except Phase2BScorerHostExecutorErrorV1:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        _fail(code, f"{label} cannot be opened componentwise: {type(exc).__name__}")


def _recheck_pinned_directory_v1(
    pinned: _PinnedDirectoryV1,
    *,
    label: str,
    code: str = FAIL_HOST_PATH,
) -> None:
    reopened = _open_componentwise_directory_v1(pinned.path, label=label, code=code)
    try:
        if (
            reopened.component_identities != pinned.component_identities
            or _directory_identity_v1(os.fstat(pinned.descriptor))
            != pinned.component_identities[-1][1]
        ):
            _fail(code, f"{label} namespace or metadata identity changed")
    finally:
        reopened.close()


def _directory_names_v1(
    pinned: _PinnedDirectoryV1,
    *,
    label: str,
    code: str,
) -> frozenset[str]:
    try:
        names = os.listdir(pinned.descriptor)
    except OSError as exc:
        _fail(code, f"{label} cannot be enumerated: {type(exc).__name__}")
    if (
        len(names) > MAXIMUM_NAMESPACE_NAMES_V1
        or any(type(name) is not str or not name or "/" in name for name in names)
    ):
        _fail(code, f"{label} namespace is malformed or oversized")
    return frozenset(names)


@dataclass(slots=True)
class _PinnedRegularFileV1:
    parent: _PinnedDirectoryV1
    name: str
    descriptor: int
    identity: tuple[int, ...]
    payload: bytes
    sha256: str

    def close(self) -> None:
        descriptor, self.descriptor = self.descriptor, -1
        if descriptor >= 0:
            os.close(descriptor)


def _read_regular_descriptor_v1(
    descriptor: int,
    *,
    minimum_bytes: int,
    maximum_bytes: int,
    single_link_required: bool,
    label: str,
    code: str,
) -> tuple[bytes, tuple[int, ...]]:
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (single_link_required and before.st_nlink != 1)
            or not minimum_bytes <= before.st_size <= maximum_bytes
            or stat.S_IMODE(before.st_mode) & 0o022
        ):
            _fail(code, f"{label} mode/link/size boundary differs")
        chunks: list[bytes] = []
        total = 0
        while total <= maximum_bytes:
            chunk = os.read(
                descriptor,
                min(_READ_CHUNK_BYTES_V1, maximum_bytes + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        payload = b"".join(chunks)
        identity = _metadata_identity_v1(before)
        if (
            identity != _metadata_identity_v1(after)
            or len(payload) != before.st_size
            or len(payload) > maximum_bytes
        ):
            _fail(code, f"{label} changed or exceeded its bound during read")
        return payload, identity
    except Phase2BScorerHostExecutorErrorV1:
        raise
    except OSError as exc:
        _fail(code, f"{label} cannot be read: {type(exc).__name__}")


def _open_relative_regular_file_v1(
    parent: _PinnedDirectoryV1,
    name: str,
    *,
    minimum_bytes: int,
    maximum_bytes: int,
    single_link_required: bool,
    label: str,
    code: str,
) -> _PinnedRegularFileV1:
    if (
        type(name) is not str
        or not name
        or PurePosixPath(name).is_absolute()
        or tuple(PurePosixPath(name).parts) != (name,)
    ):
        _fail(code, f"{label} basename differs")
    descriptor: int | None = None
    recheck: int | None = None
    try:
        descriptor = os.open(name, _regular_flags_v1(code), dir_fd=parent.descriptor)
        payload, identity = _read_regular_descriptor_v1(
            descriptor,
            minimum_bytes=minimum_bytes,
            maximum_bytes=maximum_bytes,
            single_link_required=single_link_required,
            label=label,
            code=code,
        )
        recheck = os.open(name, _regular_flags_v1(code), dir_fd=parent.descriptor)
        if _metadata_identity_v1(os.fstat(recheck)) != identity:
            _fail(code, f"{label} namespace identity changed during read")
        return _PinnedRegularFileV1(
            parent=parent,
            name=name,
            descriptor=descriptor,
            identity=identity,
            payload=payload,
            sha256=_sha256(payload),
        )
    except Phase2BScorerHostExecutorErrorV1:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        _fail(code, f"{label} cannot be opened: {type(exc).__name__}")
    finally:
        if recheck is not None:
            os.close(recheck)


def _recheck_regular_file_v1(
    pinned: _PinnedRegularFileV1,
    *,
    minimum_bytes: int,
    maximum_bytes: int,
    label: str,
    code: str,
) -> None:
    current: int | None = None
    try:
        if _metadata_identity_v1(os.fstat(pinned.descriptor)) != pinned.identity:
            _fail(code, f"{label} open inode changed")
        current = os.open(
            pinned.name,
            _regular_flags_v1(code),
            dir_fd=pinned.parent.descriptor,
        )
        payload, identity = _read_regular_descriptor_v1(
            current,
            minimum_bytes=minimum_bytes,
            maximum_bytes=maximum_bytes,
            single_link_required=True,
            label=label,
            code=code,
        )
        if identity != pinned.identity or payload != pinned.payload:
            _fail(code, f"{label} namespace or bytes changed")
    except Phase2BScorerHostExecutorErrorV1:
        raise
    except OSError as exc:
        _fail(code, f"{label} cannot be rechecked: {type(exc).__name__}")
    finally:
        if current is not None:
            os.close(current)


def _require_nonoverlap_v1(paths: Mapping[str, Path]) -> None:
    rows = tuple(paths.items())
    for index, (left_name, left) in enumerate(rows):
        for right_name, right in rows[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                _fail(
                    FAIL_HOST_PATH,
                    f"{left_name} and {right_name} host namespaces overlap",
                )


def _write_all_v1(descriptor: int, payload: bytes, *, code: str) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        try:
            written = os.write(descriptor, view[offset:])
        except OSError as exc:
            _fail(code, f"bounded file write failed: {type(exc).__name__}")
        if written <= 0:
            _fail(code, "bounded file write was short")
        offset += written


def _write_new_file_v1(
    parent: _PinnedDirectoryV1,
    name: str,
    payload: bytes,
    *,
    mode: int,
    code: str,
) -> _PinnedRegularFileV1:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | _required_open_flag_v1("O_NOFOLLOW", code)
            | _required_open_flag_v1("O_CLOEXEC", code),
            0o600,
            dir_fd=parent.descriptor,
        )
        _write_all_v1(descriptor, payload, code=code)
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.fsync(parent.descriptor)
        return _open_relative_regular_file_v1(
            parent,
            name,
            minimum_bytes=len(payload),
            maximum_bytes=len(payload),
            single_link_required=True,
            label=f"staged {name}",
            code=code,
        )
    except Phase2BScorerHostExecutorErrorV1:
        raise
    except OSError as exc:
        _fail(code, f"staged {name} cannot be created: {type(exc).__name__}")
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _pin_scorer_source_root_v1(
    repository_path: Path,
    *,
    stack: ExitStack,
) -> tuple[
    _PinnedDirectoryV1,
    _PinnedDirectoryV1,
    tuple[_PinnedRegularFileV1, ...],
    str,
]:
    source_directory = _open_componentwise_directory_v1(
        repository_path / "src" / "hegel_machine",
        label="scorer source module directory",
        code=FAIL_BUNDLE_BINDING,
    )
    stack.callback(source_directory.close)
    image_directory = _open_componentwise_directory_v1(
        repository_path / "docker" / "phase2b_scorer_runtime_v1",
        label="scorer image payload directory",
        code=FAIL_BUNDLE_BINDING,
    )
    stack.callback(image_directory.close)
    payloads: dict[str, bytes] = {}
    pinned_files: list[_PinnedRegularFileV1] = []
    for installed_path in SCORER_SOURCE_ROOT_ORDERED_ENTRY_PATHS_V1:
        if installed_path == "/opt/hegel/bin/phase2b-scorer":
            parent = image_directory
            filename = "phase2b-scorer"
        elif installed_path == "/opt/hegel/src/hegel_machine/__init__.py":
            parent = image_directory
            filename = "hegel_machine_init.py"
        else:
            prefix = "/opt/hegel/src/hegel_machine/"
            if not installed_path.startswith(prefix):
                _fail(FAIL_BUNDLE_BINDING, "scorer installed source path differs")
            parent = source_directory
            filename = installed_path.removeprefix(prefix)
        pinned = _open_relative_regular_file_v1(
            parent,
            filename,
            minimum_bytes=1,
            maximum_bytes=MAXIMUM_SCORER_SOURCE_FILE_BYTES_V1,
            single_link_required=True,
            label=f"scorer source payload {installed_path}",
            code=FAIL_BUNDLE_BINDING,
        )
        stack.callback(pinned.close)
        pinned_files.append(pinned)
        payloads[installed_path] = pinned.payload
    root = compute_scorer_source_root_sha256_v1(payloads)
    return source_directory, image_directory, tuple(pinned_files), root


def _unique_json_object_v1(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ValueError("JSON object repeats a key")
        result[key] = value
    return result


def _reject_json_constant_v1(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _decode_json_v1(payload: bytes, *, label: str, code: str) -> object:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object_v1,
            parse_constant=_reject_json_constant_v1,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        _fail(code, f"{label} is not strict duplicate-free JSON: {type(exc).__name__}")


def _canonical_ascii_json_line_v1(value: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                dict(value),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        _fail(FAIL_OUTPUT_BINDING, f"canonical JSON failed: {type(exc).__name__}")


def _extract_container_result_wire_v1(payload: bytes) -> tuple[bytes, bytes]:
    header_bytes = len(_CONTAINER_RESULT_WIRE_MAGIC_V1) + 16
    if (
        type(payload) is not bytes
        or not header_bytes + 4 <= len(payload) <= MAXIMUM_DOCKER_RESULT_WIRE_BYTES_V1
        or not payload.startswith(_CONTAINER_RESULT_WIRE_MAGIC_V1)
    ):
        _fail(FAIL_OUTPUT_BINDING, "container result wire bound or magic differs")
    offset = len(_CONTAINER_RESULT_WIRE_MAGIC_V1)
    try:
        score_bytes, gate_bytes = struct.unpack(">QQ", payload[offset : offset + 16])
    except struct.error as exc:
        _fail(FAIL_OUTPUT_BINDING, f"container result wire header failed: {type(exc).__name__}")
    offset += 16
    if (
        not 2 <= score_bytes <= MAXIMUM_SCORER_REPORT_BYTES_V1
        or not 2 <= gate_bytes <= MAXIMUM_SCORER_REPORT_BYTES_V1
        or len(payload) != offset + score_bytes + gate_bytes
    ):
        _fail(FAIL_OUTPUT_BINDING, "container result wire length projection differs")
    score = payload[offset : offset + score_bytes]
    gate = payload[offset + score_bytes :]
    return score, gate


@dataclass(frozen=True, slots=True)
class _BoundedCompletedProcessV1:
    command: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    started_monotonic_ns: int
    finished_monotonic_ns: int


def _kill_process_group_v1(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except OSError:
            pass


def _kill_and_reap_process_v1(
    process: subprocess.Popen[bytes],
    *,
    code: str,
) -> None:
    if process.poll() is None:
        _kill_process_group_v1(process)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired) as exc:
            _fail(code, f"Docker client process cannot be reaped: {type(exc).__name__}")


def _run_bounded_command_v1(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    working_directory: Path,
    timeout_seconds: int,
    maximum_stdout_bytes: int,
    maximum_stderr_bytes: int,
    timeout_code: str,
    execution_code: str,
    forced_cleanup: Callable[[], None] | None = None,
) -> _BoundedCompletedProcessV1:
    if (
        type(command) not in {tuple, list}
        or not command
        or any(type(item) is not str or not item or "\x00" in item for item in command)
        or command[0] != DEFAULT_DOCKER_EXECUTABLE.as_posix()
        or tuple(command)[1:2] != (f"--host={LOCAL_DOCKER_HOST}",)
        or type(environment) not in {dict, MappingProxyType}
        or type(working_directory) is not _EXACT_PATH_TYPE_V1
    ):
        _fail(execution_code, "Docker command/control surface differs")
    if (
        type(timeout_seconds) is not int
        or timeout_seconds < 1
        or type(maximum_stdout_bytes) is not int
        or maximum_stdout_bytes < 1
        or type(maximum_stderr_bytes) is not int
        or maximum_stderr_bytes < 1
    ):
        _fail(execution_code, "bounded process limits are malformed")
    process: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    overflow_label: str | None = None
    started = time.monotonic_ns()
    deadline = time.monotonic() + timeout_seconds
    cleanup_completed = False

    def cleanup_once() -> None:
        nonlocal cleanup_completed
        if forced_cleanup is not None and not cleanup_completed:
            forced_cleanup()
            cleanup_completed = True

    try:
        process = subprocess.Popen(
            list(command),
            cwd=working_directory.as_posix(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(environment),
            close_fds=True,
            start_new_session=True,
            bufsize=0,
        )
        if process.stdout is None or process.stderr is None:
            _fail(execution_code, "Docker process pipes are unavailable")
        streams = {
            process.stdout.fileno(): (
                "stdout",
                process.stdout,
                stdout_buffer,
                maximum_stdout_bytes,
            ),
            process.stderr.fileno(): (
                "stderr",
                process.stderr,
                stderr_buffer,
                maximum_stderr_bytes,
            ),
        }
        for descriptor, row in streams.items():
            os.set_blocking(descriptor, False)
            selector.register(row[1], selectors.EVENT_READ, data=descriptor)
        open_descriptors = set(streams)
        while open_descriptors:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_and_reap_process_v1(process, code=execution_code)
                cleanup_once()
                _fail(timeout_code, f"Docker command exceeded {timeout_seconds}s")
            events = selector.select(timeout=min(remaining, 0.25))
            if not events and process.poll() is not None:
                events = tuple(
                    (
                        selectors.SelectorKey(
                            fileobj=streams[item][1],
                            fd=item,
                            events=selectors.EVENT_READ,
                            data=item,
                        ),
                        selectors.EVENT_READ,
                    )
                    for item in tuple(open_descriptors)
                )
            for key, _mask in events:
                descriptor = key.data
                if descriptor not in open_descriptors:
                    continue
                name, stream, buffer, cap = streams[descriptor]
                try:
                    chunk = os.read(descriptor, 65_536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    open_descriptors.remove(descriptor)
                    continue
                remaining_capacity = cap - len(buffer)
                if len(chunk) > remaining_capacity:
                    if remaining_capacity > 0:
                        buffer.extend(chunk[:remaining_capacity])
                    overflow_label = name
                    _kill_and_reap_process_v1(process, code=execution_code)
                    break
                buffer.extend(chunk)
            if overflow_label is not None:
                cleanup_once()
                _fail(
                    FAIL_DOCKER_CAPTURE,
                    f"Docker {overflow_label} exceeded its exact byte cap",
                )
        try:
            returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            _kill_and_reap_process_v1(process, code=execution_code)
            cleanup_once()
            _fail(timeout_code, f"Docker command exceeded {timeout_seconds}s")
        return _BoundedCompletedProcessV1(
            command=tuple(command),
            returncode=returncode,
            stdout=bytes(stdout_buffer),
            stderr=bytes(stderr_buffer),
            started_monotonic_ns=started,
            finished_monotonic_ns=time.monotonic_ns(),
        )
    except Phase2BScorerHostExecutorErrorV1:
        if process is not None:
            _kill_and_reap_process_v1(process, code=execution_code)
        cleanup_once()
        raise
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        if process is not None:
            _kill_and_reap_process_v1(process, code=execution_code)
        cleanup_once()
        _fail(execution_code, f"Docker command failed locally: {type(exc).__name__}")
    except BaseException:
        try:
            if process is not None:
                _kill_and_reap_process_v1(process, code=execution_code)
        finally:
            try:
                cleanup_once()
            except Exception:
                pass
        raise
    finally:
        selector.close()
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass


def _validate_control_plane_v1(
    control_plane: LocalDockerControlPlaneV1,
    *,
    runtime_root: Path,
) -> None:
    expected_keys = {
        "DOCKER_CONFIG",
        "DOCKER_HOST",
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
    }
    expected_environment = {
        "DOCKER_CONFIG": (runtime_root / "docker-config").as_posix(),
        "DOCKER_HOST": LOCAL_DOCKER_HOST,
        "HOME": (runtime_root / "docker-home").as_posix(),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    if (
        type(control_plane) is not LocalDockerControlPlaneV1
        or control_plane.executable != DEFAULT_DOCKER_EXECUTABLE
        or type(control_plane.environment) is not MappingProxyType
        or dict(control_plane.environment) != expected_environment
        or set(control_plane.environment) != expected_keys
        or type(control_plane.binding) is not MappingProxyType
        or control_plane.binding.get("docker_executable")
        != DEFAULT_DOCKER_EXECUTABLE.as_posix()
        or control_plane.binding.get("docker_socket_requested")
        != DEFAULT_DOCKER_SOCKET.as_posix()
        or control_plane.binding.get("docker_host") != LOCAL_DOCKER_HOST
        or control_plane.binding.get("environment_keys") != sorted(expected_keys)
        or control_plane.binding.get("proxy_environment_keys") != []
        or control_plane.binding.get("network_endpoint_kind")
        != "LOCAL_UNIX_SOCKET"
    ):
        _fail(FAIL_DOCKER_CONTROL_PLANE, "Docker control-plane binding differs")
    try:
        if control_plane.socket_path != DEFAULT_DOCKER_SOCKET.resolve(strict=True):
            _fail(FAIL_DOCKER_CONTROL_PLANE, "Docker socket resolution differs")
        metadata = DEFAULT_DOCKER_SOCKET.lstat()
    except OSError as exc:
        _fail(FAIL_DOCKER_CONTROL_PLANE, f"Docker socket unavailable: {type(exc).__name__}")
    if not stat.S_ISSOCK(metadata.st_mode):
        _fail(FAIL_DOCKER_CONTROL_PLANE, "Docker endpoint is not a Unix socket")


def _validate_image_inspect_v1(
    payload: bytes,
    *,
    spec: Phase2BScorerRunSpecV1,
) -> tuple[str, str]:
    value = _decode_json_v1(
        payload,
        label="Docker scorer image inspection",
        code=FAIL_IMAGE_BINDING,
    )
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        _fail(FAIL_IMAGE_BINDING, "Docker scorer image inspection shape differs")
    row = value[0]
    config = row.get("Config")
    rootfs = row.get("RootFS")
    if type(config) is not dict or type(rootfs) is not dict:
        _fail(FAIL_IMAGE_BINDING, "Docker scorer image config/rootfs is absent")
    image_id = row.get("Id")
    if (
        image_id != spec.scorer_image_digest
        or type(image_id) is not str
        or _DIGEST_REF_V1.fullmatch(image_id) is None
        or row.get("Os") != "linux"
        or row.get("Architecture") != "amd64"
        or frozenset(config) != SCORER_RUNTIME_IMAGE_CONFIG_FIELDS_V1
        or config.get("ArgsEscaped") is not SCORER_RUNTIME_IMAGE_ARGS_ESCAPED_V1
        or config.get("User") != SCORER_RUNTIME_UID_GID_V1
        or config.get("WorkingDir") != "/opt/hegel"
        or config.get("Entrypoint") != [FROZEN_SCORER_ENTRYPOINT_V1[0]]
        or config.get("Cmd") != list(FROZEN_SCORER_ENTRYPOINT_V1[1:])
        or rootfs.get("Type") != "layers"
        or type(rootfs.get("Layers")) is not list
        or not rootfs["Layers"]
        or any(
            type(item) is not str or _DIGEST_REF_V1.fullmatch(item) is None
            for item in rootfs["Layers"]
        )
    ):
        _fail(FAIL_IMAGE_BINDING, "Docker scorer image identity/config differs")
    if (
        type(config.get("Env")) is not list
        or tuple(config["Env"]) != _IMAGE_ENVIRONMENT_V1
    ):
        _fail(FAIL_IMAGE_BINDING, "Docker scorer image environment differs")
    labels = config.get("Labels")
    expected_labels = {
        "org.opencontainers.image.title": SCORER_RUNTIME_IMAGE_TITLE_V1,
        "org.opencontainers.image.description": SCORER_RUNTIME_IMAGE_DESCRIPTION_V1,
        "org.opencontainers.image.base.digest": SCORER_RUNTIME_BASE_DIGEST_V1,
        "org.hegel-machine.phase2b.actor-profile.content-id": (
            PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
        ),
        "org.hegel-machine.phase2b.actor-profile.canonical-sha256": (
            PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
        ),
        "org.hegel-machine.phase2b.scoring-source.sha256": (
            SCORING_MECHANICS_SOURCE_SHA256_V1
        ),
        "org.hegel-machine.phase2b.scorer-source-root.sha256": (
            spec.scorer_source_root_sha256
        ),
    }
    if type(labels) is not dict or labels != expected_labels:
        _fail(FAIL_IMAGE_BINDING, "Docker scorer image labels differ")
    return image_id, stable_hash(config)


_RUNTIME_RECEIPT_FIELDS_V1: Final = frozenset(
    {
        "schema_version",
        "claim_level",
        "reason",
        "bundle_filename",
        "bundle_id",
        "bundle_sha256",
        "authority_requirement_id",
        "actor_profile_content_id",
        "actor_profile_canonical_sha256",
        "campaign_id",
        "attempt_id",
        "execution_registry_id",
        "scorer_image_digest",
        "scorer_source_root_sha256",
        "protocol_id",
        "exact_freeze_id",
        "execution_freeze_manifest_id",
        "scorer_run_manifest_id",
        "scoring_source_sha256",
        "score_mechanics_result_id",
        "replay_input_contract_result_id",
        "available_gate_mechanics_result_id",
        "score_report_filename",
        "score_report_id",
        "score_report_sha256",
        "score_report_bytes",
        "available_gate_report_filename",
        "available_gate_report_id",
        "available_gate_report_sha256",
        "available_gate_report_bytes",
        "local_container_mechanics_complete",
        "strict_input_prediction_replayed",
        "public_partition_rebuilt",
        "nine_metric_mechanics_materialized",
        "ten_overall_gate_mechanics_materialized",
        "twenty_four_slice_gate_mechanics_materialized",
        "two_unavailable_gate_inputs_retained",
        "reveal_authority_verified",
        "prediction_commit_before_reveal_verified",
        "scoring_performed",
        "prediction_scored",
        "formal_gate_evaluation_performed",
        "actual_960_case_run_verified",
        "effect_evidence",
        "formal_c1_report_verified",
        "c1_exit_evidence",
        "receipt_id",
    }
)
_RUNTIME_TRUE_CLAIMS_V1: Final = (
    "local_container_mechanics_complete",
    "strict_input_prediction_replayed",
    "public_partition_rebuilt",
    "nine_metric_mechanics_materialized",
    "ten_overall_gate_mechanics_materialized",
    "twenty_four_slice_gate_mechanics_materialized",
    "two_unavailable_gate_inputs_retained",
)
_RUNTIME_FALSE_CLAIMS_V1: Final = (
    "reveal_authority_verified",
    "prediction_commit_before_reveal_verified",
    "scoring_performed",
    "prediction_scored",
    "formal_gate_evaluation_performed",
    "actual_960_case_run_verified",
    "effect_evidence",
    "formal_c1_report_verified",
    "c1_exit_evidence",
)


def _validate_runtime_receipt_v1(
    payload: bytes,
    *,
    bundle_payload: bytes,
    decoded: DecodedPhase2BScorerBundleV1,
    evaluation: EvaluatedPhase2BScorerBundleV1,
    score_report: bytes,
    gate_report: bytes,
) -> str:
    if (
        type(payload) is not bytes
        or not 2 <= len(payload) <= MAXIMUM_SCORER_RUNTIME_STDOUT_BYTES_V1
        or not payload.endswith(b"\n")
        or payload.count(b"\n") != 1
    ):
        _fail(FAIL_OUTPUT_BINDING, "scorer runtime receipt framing differs")
    value = _decode_json_v1(
        payload,
        label="scorer runtime receipt",
        code=FAIL_OUTPUT_BINDING,
    )
    if (
        type(value) is not dict
        or frozenset(value) != _RUNTIME_RECEIPT_FIELDS_V1
        or _canonical_ascii_json_line_v1(value) != payload
    ):
        _fail(FAIL_OUTPUT_BINDING, "scorer runtime receipt fields/canonical form differ")
    expected = {
        "schema_version": SCORER_RUNTIME_RECEIPT_SCHEMA_VERSION_V1,
        "claim_level": SCORER_RUNTIME_CLAIM_LEVEL_V1,
        "reason": SCORER_RUNTIME_COMPLETE_REASON_V1,
        "bundle_filename": PHASE2B_SCORER_BUNDLE_V1_FILENAME,
        "bundle_id": decoded.bundle_id,
        "bundle_sha256": _sha256(bundle_payload),
        "authority_requirement_id": decoded.authority_requirement_id,
        "actor_profile_content_id": decoded.actor_profile_content_id,
        "actor_profile_canonical_sha256": (
            PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
        ),
        "campaign_id": decoded.campaign_id,
        "attempt_id": decoded.attempt_id,
        "execution_registry_id": decoded.execution_registry_id,
        "scorer_image_digest": decoded.scorer_image_digest,
        "scorer_source_root_sha256": decoded.scorer_source_root_sha256,
        "protocol_id": decoded.protocol_id,
        "exact_freeze_id": decoded.exact_freeze_id,
        "execution_freeze_manifest_id": decoded.execution_freeze_manifest_id,
        "scorer_run_manifest_id": decoded.run_manifest_id,
        "scoring_source_sha256": SCORING_MECHANICS_SOURCE_SHA256_V1,
        "score_mechanics_result_id": evaluation.scoring_mechanics.result_id,
        "replay_input_contract_result_id": evaluation.replay_input_contract.result_id,
        "available_gate_mechanics_result_id": evaluation.available_gate_mechanics.result_id,
        "score_report_filename": PHASE2B_SCORER_SCORE_MECHANICS_REPORT_V1_FILENAME,
        "score_report_id": (
            SCORE_REPORT_ID_PREFIX_V1
            + _sha256(SCORE_REPORT_DOMAIN_V1 + score_report)
        ),
        "score_report_sha256": _sha256(score_report),
        "score_report_bytes": len(score_report),
        "available_gate_report_filename": (
            PHASE2B_SCORER_AVAILABLE_GATE_MECHANICS_REPORT_V1_FILENAME
        ),
        "available_gate_report_id": (
            GATE_REPORT_ID_PREFIX_V1
            + _sha256(GATE_REPORT_DOMAIN_V1 + gate_report)
        ),
        "available_gate_report_sha256": _sha256(gate_report),
        "available_gate_report_bytes": len(gate_report),
    }
    if any(value.get(name) != item for name, item in expected.items()):
        _fail(FAIL_OUTPUT_BINDING, "scorer runtime receipt identity differs")
    if any(value.get(name) is not True for name in _RUNTIME_TRUE_CLAIMS_V1):
        _fail(FAIL_OUTPUT_BINDING, "scorer runtime mechanics claims differ")
    if any(value.get(name) is not False for name in _RUNTIME_FALSE_CLAIMS_V1):
        _fail(FAIL_OUTPUT_BINDING, "scorer runtime authority claims differ")
    receipt_id = value.get("receipt_id")
    without_id = dict(value)
    without_id.pop("receipt_id")
    expected_id = SCORER_RUNTIME_RECEIPT_ID_PREFIX_V1 + _sha256(
        b"HEGEL/PHASE2B/SCORER_RUNTIME_RECEIPT/V1\x00"
        + json.dumps(
            without_id,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    if receipt_id != expected_id:
        _fail(FAIL_OUTPUT_BINDING, "scorer runtime receipt ID differs")
    assert type(receipt_id) is str
    return receipt_id


def _container_name_v1(runtime_root: Path) -> str:
    name = "hegel-phase2b-scorer-" + hashlib.sha256(
        (runtime_root.as_posix() + "\x00" + str(os.getpid())).encode("utf-8")
    ).hexdigest()[:24]
    if _CONTAINER_NAME_V1.fullmatch(name) is None:
        _fail(FAIL_DOCKER_EXECUTION, "scorer container name projection differs")
    return name


def _docker_create_command_v1(
    control_plane: LocalDockerControlPlaneV1,
    *,
    spec: Phase2BScorerRunSpecV1,
    container_name: str,
    staged_input_directory: Path,
    staged_seccomp_path: Path,
) -> tuple[str, ...]:
    limits = spec.limits
    command = control_plane.command(
        "container",
        "create",
        f"--name={container_name}",
        "--platform=linux/amd64",
        "--no-healthcheck",
        "--pull=never",
        "--read-only",
        "--network=none",
        "--ipc=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        f"--security-opt=seccomp={staged_seccomp_path.as_posix()}",
        f"--user={SCORER_RUNTIME_UID_GID_V1}",
        "--workdir=/opt/hegel",
        "--restart=no",
        "--stop-timeout=1",
        f"--cpus={limits.cpus}",
        f"--memory={limits.memory}",
        f"--memory-swap={limits.memory_swap}",
        f"--pids-limit={limits.pids_limit}",
        f"--ulimit=nofile={limits.nofile_limit}",
        f"--ulimit=nproc={limits.nproc_limit}",
        f"--ulimit=fsize={MAXIMUM_SCORER_REPORT_BYTES_V1}:{MAXIMUM_SCORER_REPORT_BYTES_V1}",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,mode=0700,uid=65534,gid=65534,size=268435456",
        f"--mount=type=bind,src={staged_input_directory.as_posix()},dst=/phase2b/input,readonly",
        (
            "--tmpfs=/phase2b/output:rw,noexec,nosuid,nodev,mode=0700,"
            f"uid=65534,gid=65534,size={limits.output_tmpfs_bytes}"
        ),
        *tuple(f"--env={item}" for item in _CONTAINER_ENVIRONMENT_V1),
        f"--entrypoint={_CONTAINER_SUPERVISOR_ENTRYPOINT_V1}",
        spec.scorer_image_digest,
        "-c",
        _CONTAINER_SUPERVISOR_CODE_V1,
        *FROZEN_SCORER_ENTRYPOINT_V1[1:],
    )
    return tuple(command)


def _parse_created_container_id_v1(payload: bytes) -> str:
    try:
        value = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        _fail(FAIL_DOCKER_EXECUTION, f"container ID is not ASCII: {type(exc).__name__}")
    if not value.endswith("\n") or value.count("\n") != 1:
        _fail(FAIL_DOCKER_EXECUTION, "container create stdout framing differs")
    container_id = value[:-1]
    if _CONTAINER_ID_V1.fullmatch(container_id) is None:
        _fail(FAIL_DOCKER_EXECUTION, "container create ID differs")
    return container_id


def _validate_created_container_inspect_v1(
    payload: bytes,
    *,
    container_id: str,
    container_name: str,
    spec: Phase2BScorerRunSpecV1,
    staged_input_directory: Path,
    expected_seccomp: Mapping[str, object],
) -> None:
    value = _decode_json_v1(
        payload,
        label="created scorer container inspection",
        code=FAIL_DOCKER_EXECUTION,
    )
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        _fail(FAIL_DOCKER_EXECUTION, "created scorer container shape differs")
    row = value[0]
    config = row.get("Config")
    host = row.get("HostConfig")
    state = row.get("State")
    mounts = row.get("Mounts")
    network = row.get("NetworkSettings")
    if not all(type(item) is dict for item in (config, host, state, network)):
        _fail(FAIL_DOCKER_EXECUTION, "created scorer container sections are absent")
    if type(mounts) is not list or len(mounts) != 1 or type(mounts[0]) is not dict:
        _fail(FAIL_DOCKER_EXECUTION, "created scorer container mount count differs")
    mount = mounts[0]
    security_options = host.get("SecurityOpt")
    tmpfs = host.get("Tmpfs")
    ulimits = host.get("Ulimits")
    if type(security_options) is not list or len(security_options) != 2:
        _fail(FAIL_DOCKER_EXECUTION, "created scorer security options differ")
    seccomp_options = tuple(
        item
        for item in security_options
        if type(item) is str and item.startswith("seccomp=")
    )
    if len(seccomp_options) != 1:
        _fail(FAIL_DOCKER_EXECUTION, "created scorer seccomp option is absent")
    observed_seccomp = _decode_json_v1(
        seccomp_options[0][len("seccomp=") :].encode("utf-8"),
        label="created scorer embedded seccomp",
        code=FAIL_DOCKER_EXECUTION,
    )
    observed_ulimits = (
        {
            (item.get("Name"), item.get("Hard"), item.get("Soft"))
            for item in ulimits
        }
        if type(ulimits) is list and all(type(item) is dict for item in ulimits)
        else set()
    )
    expected_ulimits = {
        ("fsize", MAXIMUM_SCORER_REPORT_BYTES_V1, MAXIMUM_SCORER_REPORT_BYTES_V1),
        ("nofile", 64, 64),
        ("nproc", 128, 128),
    }
    expected_tmpfs = {
        "/tmp": (
            "rw,noexec,nosuid,nodev,mode=0700,uid=65534,gid=65534,"
            "size=268435456"
        ),
        "/phase2b/output": (
            "rw,noexec,nosuid,nodev,mode=0700,uid=65534,gid=65534,"
            "size=536870912"
        ),
    }
    expected_mount = {
        "Type": "bind",
        "Source": staged_input_directory.as_posix(),
        "Destination": "/phase2b/input",
        "Mode": "",
        "RW": False,
        "Propagation": "rprivate",
    }
    if (
        row.get("Id") != container_id
        or row.get("Name") != "/" + container_name
        or row.get("Image") != spec.scorer_image_digest
        or row.get("Path") != _CONTAINER_SUPERVISOR_ENTRYPOINT_V1
        or row.get("Args")
        != [
            "-c",
            _CONTAINER_SUPERVISOR_CODE_V1,
            *FROZEN_SCORER_ENTRYPOINT_V1[1:],
        ]
        or config.get("User") != SCORER_RUNTIME_UID_GID_V1
        or config.get("Image") != spec.scorer_image_digest
        or config.get("WorkingDir") != "/opt/hegel"
        or tuple(config.get("Env", ())) != _CONTAINER_ENVIRONMENT_V1
        or config.get("Entrypoint") != [_CONTAINER_SUPERVISOR_ENTRYPOINT_V1]
        or config.get("Cmd")
        != [
            "-c",
            _CONTAINER_SUPERVISOR_CODE_V1,
            *FROZEN_SCORER_ENTRYPOINT_V1[1:],
        ]
        or config.get("Healthcheck") != {"Test": ["NONE"]}
        or config.get("Volumes") is not None
        or state.get("Status") != "created"
        or state.get("Running") is not False
        or host.get("ReadonlyRootfs") is not True
        or host.get("NetworkMode") != "none"
        or host.get("IpcMode") != "none"
        or host.get("Privileged") is not False
        or host.get("AutoRemove") is not False
        or host.get("CapDrop") != ["ALL"]
        or host.get("NanoCpus") != 2_000_000_000
        or host.get("Memory") != 2_147_483_648
        or host.get("MemorySwap") != 2_147_483_648
        or host.get("PidsLimit") != 128
        or host.get("CgroupnsMode") != "private"
        or host.get("PidMode") != ""
        or host.get("UsernsMode") != ""
        or host.get("Runtime") != "runc"
        or "no-new-privileges:true" not in security_options
        or observed_seccomp != dict(expected_seccomp)
        or tmpfs != expected_tmpfs
        or observed_ulimits != expected_ulimits
        or mount != expected_mount
        or network.get("Ports") not in ({}, None)
    ):
        _fail(FAIL_DOCKER_EXECUTION, "created scorer container binding differs")


def _force_remove_container_v1(
    *,
    control_plane: LocalDockerControlPlaneV1,
    environment: Mapping[str, str],
    working_directory: Path,
    container_id: str,
    allow_already_absent: bool = False,
) -> None:
    if (
        _CONTAINER_ID_V1.fullmatch(container_id) is None
        and _CONTAINER_NAME_V1.fullmatch(container_id) is None
    ):
        _fail(FAIL_DOCKER_EXECUTION, "scorer container cleanup ID differs")
    if type(allow_already_absent) is not bool:
        _fail(FAIL_DOCKER_EXECUTION, "scorer cleanup absence policy differs")
    removed = _run_bounded_command_v1(
        tuple(control_plane.command("container", "rm", "--force", container_id)),
        environment=environment,
        working_directory=working_directory,
        timeout_seconds=30,
        maximum_stdout_bytes=256,
        maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V1,
        timeout_code=FAIL_DOCKER_TIMEOUT,
        execution_code=FAIL_DOCKER_EXECUTION,
    )
    inspected = _run_bounded_command_v1(
        tuple(control_plane.command("container", "inspect", container_id)),
        environment=environment,
        working_directory=working_directory,
        timeout_seconds=30,
        maximum_stdout_bytes=256,
        maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V1,
        timeout_code=FAIL_DOCKER_TIMEOUT,
        execution_code=FAIL_DOCKER_EXECUTION,
    )
    absent_stderr = (
        f"Error response from daemon: No such container: {container_id}\n"
    ).encode("ascii")
    removed_existing = (
        removed.returncode == 0
        and removed.stdout == (container_id + "\n").encode("ascii")
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
        _fail(
            FAIL_DOCKER_EXECUTION,
            "scorer container removal/absence proof differs",
        )


def _capture_runtime_receipt_v1(
    *,
    control_plane: LocalDockerControlPlaneV1,
    environment: Mapping[str, str],
    working_directory: Path,
    container_id: str,
    deadline_monotonic_ns: int,
    forced_cleanup: Callable[[], None],
) -> tuple[bytes, _BoundedCompletedProcessV1]:
    last: _BoundedCompletedProcessV1 | None = None
    while time.monotonic_ns() < deadline_monotonic_ns:
        remaining_ns = deadline_monotonic_ns - time.monotonic_ns()
        timeout = max(1, min(30, (remaining_ns + 999_999_999) // 1_000_000_000))
        last = _run_bounded_command_v1(
            tuple(control_plane.command("container", "logs", container_id)),
            environment=environment,
            working_directory=working_directory,
            timeout_seconds=timeout,
            maximum_stdout_bytes=MAXIMUM_SCORER_RUNTIME_STDOUT_BYTES_V1,
            maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V1,
            timeout_code=FAIL_DOCKER_TIMEOUT,
            execution_code=FAIL_OUTPUT_BINDING,
            forced_cleanup=forced_cleanup,
        )
        if last.returncode != 0 or last.stderr != b"":
            _fail(FAIL_OUTPUT_BINDING, "scorer container logs rejected")
        if last.stdout:
            if last.stdout.endswith(b"\n") and last.stdout.count(b"\n") == 1:
                return last.stdout, last
            _fail(FAIL_OUTPUT_BINDING, "scorer runtime emitted non-one-line stdout")
        time.sleep(0.05)
    _fail(FAIL_DOCKER_TIMEOUT, "scorer runtime receipt did not arrive before deadline")


def _publish_reports_v1(
    output: _PinnedDirectoryV1,
    *,
    score_report: bytes,
    gate_report: bytes,
) -> tuple[_PinnedRegularFileV1, _PinnedRegularFileV1]:
    reports = (
        (PHASE2B_SCORER_SCORE_MECHANICS_REPORT_V1_FILENAME, score_report),
        (PHASE2B_SCORER_AVAILABLE_GATE_MECHANICS_REPORT_V1_FILENAME, gate_report),
    )
    if _directory_names_v1(output, label="scorer host output", code=FAIL_OUTPUT_BINDING):
        _fail(FAIL_OUTPUT_BINDING, "scorer host output must be initially empty")
    created: list[tuple[str, tuple[int, int]]] = []
    committed: list[tuple[str, tuple[int, int]]] = []
    opened: list[_PinnedRegularFileV1] = []
    try:
        for filename, payload in reports:
            if (
                type(payload) is not bytes
                or not 2 <= len(payload) <= MAXIMUM_SCORER_REPORT_BYTES_V1
                or not payload.startswith(b"{")
                or not payload.endswith(b"}\n")
                or payload.count(b"\n") != 1
            ):
                _fail(FAIL_OUTPUT_BINDING, f"{filename} report framing differs")
            pending = "." + filename + ".pending"
            descriptor = os.open(
                pending,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | _required_open_flag_v1("O_NOFOLLOW", FAIL_OUTPUT_BINDING)
                | _required_open_flag_v1("O_CLOEXEC", FAIL_OUTPUT_BINDING),
                0o600,
                dir_fd=output.descriptor,
            )
            try:
                metadata = os.fstat(descriptor)
                created.append((pending, (metadata.st_dev, metadata.st_ino)))
                _write_all_v1(descriptor, payload, code=FAIL_OUTPUT_BINDING)
                os.fchmod(descriptor, 0o400)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fsync(output.descriptor)
        for filename, _payload in reports:
            pending = "." + filename + ".pending"
            os.link(
                pending,
                filename,
                src_dir_fd=output.descriptor,
                dst_dir_fd=output.descriptor,
                follow_symlinks=False,
            )
            metadata = os.stat(
                filename,
                dir_fd=output.descriptor,
                follow_symlinks=False,
            )
            committed.append((filename, (metadata.st_dev, metadata.st_ino)))
        os.fsync(output.descriptor)
        for filename, _payload in reports:
            os.unlink("." + filename + ".pending", dir_fd=output.descriptor)
        os.fsync(output.descriptor)
        if _directory_names_v1(
            output,
            label="published scorer output",
            code=FAIL_OUTPUT_BINDING,
        ) != frozenset(filename for filename, _payload in reports):
            _fail(FAIL_OUTPUT_BINDING, "published scorer output namespace differs")
        for filename, payload in reports:
            pinned = _open_relative_regular_file_v1(
                output,
                filename,
                minimum_bytes=len(payload),
                maximum_bytes=len(payload),
                single_link_required=True,
                label=f"published {filename}",
                code=FAIL_OUTPUT_BINDING,
            )
            if pinned.payload != payload:
                pinned.close()
                _fail(FAIL_OUTPUT_BINDING, f"published {filename} bytes differ")
            opened.append(pinned)
        return opened[0], opened[1]
    except BaseException:
        for pinned in opened:
            pinned.close()
        for name, identity in reversed((*committed, *created)):
            try:
                metadata = os.stat(
                    name,
                    dir_fd=output.descriptor,
                    follow_symlinks=False,
                )
                if (metadata.st_dev, metadata.st_ino) == identity:
                    os.unlink(name, dir_fd=output.descriptor)
            except (FileNotFoundError, OSError):
                pass
        try:
            os.fsync(output.descriptor)
        except OSError:
            pass
        raise


@dataclass(frozen=True, slots=True, init=False)
class Phase2BScorerLocalExecutionReceiptV1:
    schema_version: str
    disposition: str
    reason: str
    receipt_id: str
    run_spec_id: str
    authority_requirement_id: str
    actor_profile_content_id: str
    actor_profile_canonical_sha256: str
    campaign_id: str
    attempt_id: str
    execution_registry_id: str
    scorer_image_digest: str
    scorer_source_root_sha256: str
    scoring_mechanics_source_sha256: str
    protocol_id: str
    exact_freeze_id: str
    execution_freeze_manifest_id: str
    scorer_run_manifest_id: str
    observed_image_id: str
    image_inspect_sha256: str
    image_config_projection_sha256: str
    docker_control_plane_binding_sha256: str
    docker_executable_sha256: str
    seccomp_profile_sha256: str
    host_path_binding_sha256: str
    command_sha256: str
    container_supervisor_sha256: str
    container_result_extractor_sha256: str
    bundle_id: str
    bundle_sha256: str
    bundle_bytes: int
    runtime_receipt_id: str
    runtime_receipt_sha256: str
    score_mechanics_result_id: str
    replay_input_contract_result_id: str
    available_gate_mechanics_result_id: str
    score_report_id: str
    score_report_sha256: str
    score_report_bytes: int
    score_output_file_identity_sha256: str
    available_gate_report_id: str
    available_gate_report_sha256: str
    available_gate_report_bytes: int
    available_gate_output_file_identity_sha256: str
    started_monotonic_ns: int
    finished_monotonic_ns: int
    process_exit_code: int
    timed_out: bool
    docker_invoked: bool
    local_mechanics_complete: bool
    host_bundle_replay_verified: bool
    container_bundle_replay_verified: bool
    strict_input_prediction_replayed: bool
    public_partition_rebuilt: bool
    nine_metric_mechanics_materialized: bool
    ten_overall_gate_mechanics_materialized: bool
    twenty_four_slice_gate_mechanics_materialized: bool
    two_unavailable_gate_inputs_retained: bool
    external_runtime_attestation_issued: bool
    reveal_authority_verified: bool
    prediction_commit_before_reveal_verified: bool
    actual_960_case_run_verified: bool
    scoring_performed: bool
    prediction_scored: bool
    formal_gate_evaluation_performed: bool
    effect_evidence: bool
    formal_c1_report_verified: bool
    c1_exit_evidence: bool
    malicious_host_root_resistance: bool
    malicious_docker_daemon_resistance: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("scorer local execution receipts are privately issued")

    def __post_init__(self) -> None:
        true_claims = (
            "docker_invoked",
            "local_mechanics_complete",
            "host_bundle_replay_verified",
            "container_bundle_replay_verified",
            "strict_input_prediction_replayed",
            "public_partition_rebuilt",
            "nine_metric_mechanics_materialized",
            "ten_overall_gate_mechanics_materialized",
            "twenty_four_slice_gate_mechanics_materialized",
            "two_unavailable_gate_inputs_retained",
        )
        false_claims = (
            "timed_out",
            "external_runtime_attestation_issued",
            "reveal_authority_verified",
            "prediction_commit_before_reveal_verified",
            "actual_960_case_run_verified",
            "scoring_performed",
            "prediction_scored",
            "formal_gate_evaluation_performed",
            "effect_evidence",
            "formal_c1_report_verified",
            "c1_exit_evidence",
            "malicious_host_root_resistance",
            "malicious_docker_daemon_resistance",
        )
        if (
            self.schema_version != LOCAL_SCORER_EXECUTION_RECEIPT_SCHEMA_V1
            or self.disposition != LOCAL_SCORER_EXECUTION_DISPOSITION_V1
            or self.reason != LOCAL_SCORER_EXECUTION_COMPLETE_REASON_V1
            or self.authority_requirement_id
            != OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V1
            or self.actor_profile_content_id
            != PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
            or self.actor_profile_canonical_sha256
            != PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
            or _LOWER_SHA256_V1.fullmatch(
                self.campaign_id.removeprefix(ACTUAL_960_CAMPAIGN_ID_PREFIX_V2)
            )
            is None
            or not self.campaign_id.startswith(ACTUAL_960_CAMPAIGN_ID_PREFIX_V2)
            or _LOWER_SHA256_V1.fullmatch(
                self.attempt_id.removeprefix(ACTUAL_960_ATTEMPT_ID_PREFIX_V2)
            )
            is None
            or not self.attempt_id.startswith(ACTUAL_960_ATTEMPT_ID_PREFIX_V2)
            or _LOWER_SHA256_V1.fullmatch(
                self.execution_registry_id.removeprefix(
                    OWNER_ACCEPTED_EXECUTION_REGISTRY_ID_PREFIX_V1
                )
            )
            is None
            or not self.execution_registry_id.startswith(
                OWNER_ACCEPTED_EXECUTION_REGISTRY_ID_PREFIX_V1
            )
            or self.scorer_image_digest != self.observed_image_id
            or self.scoring_mechanics_source_sha256
            != SCORING_MECHANICS_SOURCE_SHA256_V1
            or self.seccomp_profile_sha256 != SCORER_SECCOMP_SHA256_V1
            or self.process_exit_code != 0
            or any(getattr(self, name) is not True for name in true_claims)
            or any(getattr(self, name) is not False for name in false_claims)
        ):
            _fail(FAIL_OUTPUT_BINDING, "scorer local receipt claim boundary differs")
        for name in (
            "image_inspect_sha256",
            "image_config_projection_sha256",
            "docker_control_plane_binding_sha256",
            "docker_executable_sha256",
            "seccomp_profile_sha256",
            "host_path_binding_sha256",
            "command_sha256",
            "container_supervisor_sha256",
            "container_result_extractor_sha256",
            "bundle_sha256",
            "runtime_receipt_sha256",
            "score_report_sha256",
            "score_output_file_identity_sha256",
            "available_gate_report_sha256",
            "available_gate_output_file_identity_sha256",
            "scoring_mechanics_source_sha256",
        ):
            if _LOWER_SHA256_V1.fullmatch(getattr(self, name)) is None:
                _fail(FAIL_OUTPUT_BINDING, f"scorer local receipt {name} differs")
        if (
            not self.receipt_id.startswith(
                "phase2b_scorer_local_execution_receipt_v1_"
            )
            or _DIGEST_REF_V1.fullmatch(self.scorer_image_digest) is None
            or _LOWER_SHA256_V1.fullmatch(self.scorer_source_root_sha256) is None
            or type(self.bundle_bytes) is not int
            or self.bundle_bytes < 1
            or type(self.score_report_bytes) is not int
            or self.score_report_bytes < 2
            or type(self.available_gate_report_bytes) is not int
            or self.available_gate_report_bytes < 2
            or type(self.started_monotonic_ns) is not int
            or type(self.finished_monotonic_ns) is not int
            or self.finished_monotonic_ns < self.started_monotonic_ns
        ):
            _fail(FAIL_OUTPUT_BINDING, "scorer local receipt numeric boundary differs")

    @classmethod
    def _issue(cls, **values: object) -> "Phase2BScorerLocalExecutionReceiptV1":
        value = object.__new__(cls)
        without_id = dict(values)
        receipt_id = stable_hash(
            without_id,
            prefix="phase2b_scorer_local_execution_receipt_v1_",
        )
        without_id["receipt_id"] = receipt_id
        if set(without_id) != {item.name for item in fields(cls)}:
            _fail(FAIL_OUTPUT_BINDING, "scorer local receipt issue field set differs")
        for name, item in without_id.items():
            object.__setattr__(value, name, item)
        value.__post_init__()
        return value


def _issue_receipt_after_publication_v1(
    output: _PinnedDirectoryV1,
    score_file: _PinnedRegularFileV1,
    gate_file: _PinnedRegularFileV1,
    **values: object,
) -> Phase2BScorerLocalExecutionReceiptV1:
    """Issue the receipt or remove only the two exact files just published."""

    try:
        return Phase2BScorerLocalExecutionReceiptV1._issue(**values)
    except BaseException:
        cleanup_failed = False
        published = (
            (PHASE2B_SCORER_SCORE_MECHANICS_REPORT_V1_FILENAME, score_file),
            (
                PHASE2B_SCORER_AVAILABLE_GATE_MECHANICS_REPORT_V1_FILENAME,
                gate_file,
            ),
        )
        for filename, pinned in reversed(published):
            pinned.close()
            try:
                metadata = os.stat(
                    filename,
                    dir_fd=output.descriptor,
                    follow_symlinks=False,
                )
                if _metadata_identity_v1(metadata) != pinned.identity:
                    cleanup_failed = True
                    continue
                os.unlink(filename, dir_fd=output.descriptor)
            except OSError:
                cleanup_failed = True
        try:
            os.fsync(output.descriptor)
        except OSError:
            cleanup_failed = True
        if cleanup_failed:
            _fail(
                FAIL_OUTPUT_BINDING,
                "scorer receipt issuance failed and exact output cleanup was incomplete",
            )
        raise


def _snapshot_run_spec_v1(spec: object) -> Phase2BScorerRunSpecV1:
    if type(spec) is not Phase2BScorerRunSpecV1:
        _fail(FAIL_PREFLIGHT, "executor requires exact Phase2BScorerRunSpecV1")
    try:
        values = {item.name: getattr(spec, item.name) for item in fields(type(spec))}
        limits = values["limits"]
        if type(limits) is not Phase2BScorerRuntimeLimitsV1:
            _fail(FAIL_PREFLIGHT, "executor requires exact scorer runtime limits")
        limit_values = {
            item.name: getattr(limits, item.name) for item in fields(type(limits))
        }
        if any(getattr(spec, name) != item for name, item in values.items()):
            _fail(FAIL_PREFLIGHT, "scorer run spec changed during snapshot")
        return Phase2BScorerRunSpecV1(
            **{name: item for name, item in values.items() if name != "limits"},
            limits=Phase2BScorerRuntimeLimitsV1(**limit_values),
        )
    except Phase2BScorerHostExecutorErrorV1:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        _fail(FAIL_PREFLIGHT, f"scorer run spec snapshot rejected: {type(exc).__name__}")


def _bundle_binding_v1(
    *,
    spec: Phase2BScorerRunSpecV1,
    bundle_payload: bytes,
) -> tuple[
    DecodedPhase2BScorerBundleV1,
    EvaluatedPhase2BScorerBundleV1,
    bytes,
    bytes,
]:
    try:
        decoded = decode_phase2b_scorer_bundle_v1(bundle_payload)
        evaluation = evaluate_phase2b_scorer_bundle_v1(decoded)
    except Exception as exc:
        _fail(FAIL_BUNDLE_BINDING, f"scorer bundle replay rejected: {type(exc).__name__}")
    if (
        type(decoded) is not DecodedPhase2BScorerBundleV1
        or type(evaluation) is not EvaluatedPhase2BScorerBundleV1
        or decoded.bundle_bytes != bundle_payload
        or decoded.bundle_sha256 != spec.bundle_sha256
        or decoded.authority_requirement_id != spec.authority_requirement_id
        or decoded.actor_profile_content_id != spec.actor_profile_content_id
        or decoded.actor_profile_canonical_sha256
        != spec.actor_profile_canonical_sha256
        or decoded.campaign_id != spec.campaign_id
        or decoded.attempt_id != spec.attempt_id
        or decoded.execution_registry_id != spec.execution_registry_id
        or decoded.scorer_image_digest != spec.scorer_image_digest
        or decoded.scorer_source_root_sha256 != spec.scorer_source_root_sha256
        or decoded.protocol_id != spec.protocol_id
        or decoded.exact_freeze_id != spec.exact_freeze_id
        or decoded.execution_freeze_manifest_id
        != spec.execution_freeze_manifest_id
        or decoded.scoring_source_sha256
        != spec.scoring_mechanics_source_sha256
        or decoded.run_manifest_id != spec.scorer_run_manifest_id
        or decoded.execution_freeze_manifest.manifest_id
        != spec.execution_freeze_manifest_id
        or decoded.execution_freeze_manifest.protocol_id != spec.protocol_id
        or decoded.partition_manifest.exact_freeze_id != spec.exact_freeze_id
        or evaluation.scoring_mechanics.protocol_id != spec.protocol_id
        or evaluation.scoring_mechanics.execution_freeze_manifest_id
        != spec.execution_freeze_manifest_id
        or evaluation.replay_input_contract.exact_freeze_id != spec.exact_freeze_id
        or evaluation.available_gate_mechanics.protocol_id != spec.protocol_id
    ):
        _fail(FAIL_BUNDLE_BINDING, "scorer bundle/run-spec graph differs")
    score_report = evaluation.score_report_bytes
    gate_report = evaluation.available_gate_report_bytes
    scoring = evaluation.scoring_mechanics
    available_gate = evaluation.available_gate_mechanics
    metric_names = tuple(name for name, _denominator in _EXPECTED_METRIC_SURFACE_V1)
    if (
        type(score_report) is not bytes
        or type(gate_report) is not bytes
        or evaluation.score_report_id
        != SCORE_REPORT_ID_PREFIX_V1
        + _sha256(SCORE_REPORT_DOMAIN_V1 + score_report)
        or evaluation.available_gate_report_id
        != GATE_REPORT_ID_PREFIX_V1
        + _sha256(GATE_REPORT_DOMAIN_V1 + gate_report)
        or (
            scoring.total_prediction_count,
            scoring.main_row_result_count,
            scoring.metric_eligible_main_row_count,
            scoring.control_row_without_frozen_metric_count,
            scoring.semantic_conflict_excluded_count,
        )
        != (960, 720, 336, 384, 240)
        or type(scoring.metric_results) is not tuple
        or tuple(
            (
                item.metric_name,
                item.expected_denominator,
                item.observed_denominator,
            )
            for item in scoring.metric_results
        )
        != tuple((name, denominator, denominator) for name, denominator in _EXPECTED_METRIC_SURFACE_V1)
        or type(scoring.main_row_results) is not tuple
        or len(scoring.main_row_results) != 720
        or any(
            type(row.metric_outcomes) is not tuple
            or len(row.metric_outcomes) != 9
            or tuple(item.metric_name for item in row.metric_outcomes) != metric_names
            for row in scoring.main_row_results
        )
        or scoring.scoring_performed is not False
        or scoring.prediction_scored is not False
        or (
            available_gate.overall_result_count,
            available_gate.slice_result_count,
            available_gate.unavailable_result_count,
        )
        != (10, 24, 2)
        or type(available_gate.available_overall_gate_mechanics_results) is not tuple
        or tuple(
            item.metric_name
            for item in available_gate.available_overall_gate_mechanics_results
        )
        != _EXPECTED_OVERALL_GATE_NAMES_V1
        or type(available_gate.available_slice_gate_mechanics_results) is not tuple
        or tuple(
            (item.metric_name, item.scope, item.slice_id)
            for item in available_gate.available_slice_gate_mechanics_results
        )
        != _EXPECTED_SLICE_GATE_SURFACE_V1
        or type(available_gate.unavailable_gate_mechanics) is not tuple
        or tuple(item.metric_name for item in available_gate.unavailable_gate_mechanics)
        != _EXPECTED_UNAVAILABLE_GATE_NAMES_V1
        or available_gate.formal_gate_evaluation_performed is not False
        or available_gate.scoring_performed is not False
        or available_gate.actual_960_case_run_verified is not False
    ):
        _fail(FAIL_BUNDLE_BINDING, "scorer mechanics result boundary differs")
    return decoded, evaluation, score_report, gate_report


def execute_phase2b_scorer_docker_v1(
    spec: Phase2BScorerRunSpecV1,
    *,
    runtime_parent: Path = Path("/tmp"),
) -> Phase2BScorerLocalExecutionReceiptV1:
    """Run one fixed local scorer container and publish two replayed reports."""

    spec = _snapshot_run_spec_v1(spec)
    if type(runtime_parent) is not _EXACT_PATH_TYPE_V1:
        _fail(FAIL_PREFLIGHT, "scorer runtime parent must be exact Path")
    runtime_parent = _exact_absolute_path_v1(
        runtime_parent.as_posix(),
        label="scorer private runtime parent",
        code=FAIL_HOST_PATH,
    )
    bundle_path = _exact_absolute_path_v1(
        spec.bundle_host_path,
        label="scorer bundle",
        code=FAIL_HOST_PATH,
    )
    output_path = _exact_absolute_path_v1(
        spec.output_host_directory,
        label="scorer host output",
        code=FAIL_HOST_PATH,
    )
    repository_path = _exact_absolute_path_v1(
        spec.repository_host_directory,
        label="scorer source repository",
        code=FAIL_HOST_PATH,
    )
    seccomp_path = _exact_absolute_path_v1(
        spec.seccomp_profile_host_path,
        label="scorer seccomp profile",
        code=FAIL_HOST_PATH,
    )
    if (
        bundle_path.name != PHASE2B_SCORER_BUNDLE_V1_FILENAME
        or seccomp_path
        != repository_path / "config" / "phase3_internal_actor_seccomp_v1.json"
    ):
        _fail(FAIL_HOST_PATH, "scorer bundle or seccomp exact path differs")
    _require_nonoverlap_v1(
        {
            "bundle parent": bundle_path.parent,
            "host output": output_path,
            "source repository": repository_path,
        }
    )
    try:
        bundle_mount = validate_linux_local_host_path_v1(
            bundle_path.parent,
            repository_root=repository_path,
        )
        output_mount = validate_linux_local_host_path_v1(
            output_path,
            repository_root=repository_path,
        )
    except Phase3LocalRuntimeError as exc:
        _fail(FAIL_HOST_PATH, f"Linux-local scorer paths rejected: {exc.code}")
    try:
        profile = load_phase2b_c1_container_actor_profile_v2(
            repository_path / "config" / "phase2b_c1_container_actor_profile_v2.json",
            repository_root=repository_path,
        )
    except Exception as exc:
        _fail(FAIL_PROFILE_BINDING, f"V2 scorer profile rejected: {type(exc).__name__}")
    if (
        type(profile) is not Phase2BC1ContainerActorProfileSnapshotV2
        or profile.profile_content_id != spec.actor_profile_content_id
        or profile.canonical_profile_sha256 != spec.actor_profile_canonical_sha256
        or profile.scoring_performed is not False
        or profile.c1_exit_evidence is not False
    ):
        _fail(FAIL_PROFILE_BINDING, "V2 scorer profile content binding differs")

    with ExitStack() as stack:
        bundle_parent = _open_componentwise_directory_v1(
            bundle_path.parent,
            label="scorer bundle parent",
            code=FAIL_BUNDLE_BINDING,
        )
        stack.callback(bundle_parent.close)
        output_directory = _open_componentwise_directory_v1(
            output_path,
            label="scorer host output",
            code=FAIL_OUTPUT_BINDING,
        )
        stack.callback(output_directory.close)
        repository_directory = _open_componentwise_directory_v1(
            repository_path,
            label="scorer source repository",
            code=FAIL_HOST_PATH,
        )
        stack.callback(repository_directory.close)
        seccomp_parent = _open_componentwise_directory_v1(
            seccomp_path.parent,
            label="scorer seccomp parent",
            code=FAIL_SECCOMP_BINDING,
        )
        stack.callback(seccomp_parent.close)
        if _directory_names_v1(
            output_directory,
            label="scorer host output",
            code=FAIL_OUTPUT_BINDING,
        ):
            _fail(FAIL_OUTPUT_BINDING, "scorer host output is not initially empty")
        bundle_file = _open_relative_regular_file_v1(
            bundle_parent,
            bundle_path.name,
            minimum_bytes=1,
            maximum_bytes=MAXIMUM_PHASE2B_SCORER_BUNDLE_BYTES_V1,
            single_link_required=True,
            label="scorer bundle",
            code=FAIL_BUNDLE_BINDING,
        )
        stack.callback(bundle_file.close)
        if bundle_file.sha256 != spec.bundle_sha256:
            _fail(FAIL_BUNDLE_BINDING, "scorer bundle digest differs")
        decoded, evaluation, score_report, gate_report = _bundle_binding_v1(
            spec=spec,
            bundle_payload=bundle_file.payload,
        )
        seccomp_file = _open_relative_regular_file_v1(
            seccomp_parent,
            seccomp_path.name,
            minimum_bytes=2,
            maximum_bytes=MAXIMUM_SECCOMP_BYTES_V1,
            single_link_required=True,
            label="scorer seccomp profile",
            code=FAIL_SECCOMP_BINDING,
        )
        stack.callback(seccomp_file.close)
        if (
            seccomp_file.sha256 != spec.seccomp_profile_sha256
            or seccomp_file.sha256 != SCORER_SECCOMP_SHA256_V1
        ):
            _fail(FAIL_SECCOMP_BINDING, "scorer seccomp digest differs")
        seccomp_value = _decode_json_v1(
            seccomp_file.payload,
            label="scorer seccomp profile",
            code=FAIL_SECCOMP_BINDING,
        )
        if (
            type(seccomp_value) is not dict
            or seccomp_value.get("defaultAction") != "SCMP_ACT_ALLOW"
            or type(seccomp_value.get("syscalls")) is not list
            or not seccomp_value["syscalls"]
        ):
            _fail(FAIL_SECCOMP_BINDING, "scorer seccomp semantics differ")
        (
            source_directory,
            image_payload_directory,
            source_payload_files,
            observed_source_root_sha256,
        ) = _pin_scorer_source_root_v1(repository_path, stack=stack)
        if observed_source_root_sha256 != spec.scorer_source_root_sha256:
            _fail(FAIL_BUNDLE_BINDING, "scorer source-root digest differs")

        docker_owner = LinuxLocalTemporaryDirectoryV1(
            prefix="hegel-phase2b-scorer-",
            repository_root=repository_path,
            parent=runtime_parent,
        )
        runtime_root = Path(stack.enter_context(docker_owner))
        staged_input_path = runtime_root / _STAGED_INPUT_DIRECTORY_V1
        try:
            os.mkdir(staged_input_path, 0o755)
        except OSError as exc:
            _fail(FAIL_DOCKER_CONTROL_PLANE, f"scorer staging failed: {type(exc).__name__}")
        staged_input_directory = _open_componentwise_directory_v1(
            staged_input_path,
            label="staged scorer input",
            code=FAIL_DOCKER_CONTROL_PLANE,
        )
        stack.callback(staged_input_directory.close)
        staged_bundle = _write_new_file_v1(
            staged_input_directory,
            PHASE2B_SCORER_BUNDLE_V1_FILENAME,
            bundle_file.payload,
            mode=0o444,
            code=FAIL_BUNDLE_BINDING,
        )
        stack.callback(staged_bundle.close)
        staging_root = _open_componentwise_directory_v1(
            runtime_root,
            label="scorer staging root",
            code=FAIL_DOCKER_CONTROL_PLANE,
        )
        stack.callback(staging_root.close)
        staged_seccomp = _write_new_file_v1(
            staging_root,
            _STAGED_SECCOMP_FILENAME_V1,
            seccomp_file.payload,
            mode=0o400,
            code=FAIL_SECCOMP_BINDING,
        )
        stack.callback(staged_seccomp.close)
        try:
            control_plane = prepare_local_docker_control_plane_v1(
                runtime_root,
                repository_root=repository_path,
            )
        except Phase3LocalRuntimeError as exc:
            _fail(FAIL_DOCKER_CONTROL_PLANE, f"local Docker boundary rejected: {exc.code}")
        except OSError as exc:
            _fail(FAIL_DOCKER_CONTROL_PLANE, f"Docker staging failed: {type(exc).__name__}")
        _validate_control_plane_v1(control_plane, runtime_root=runtime_root)
        runtime_directory = _open_componentwise_directory_v1(
            runtime_root,
            label="private scorer runtime",
            code=FAIL_DOCKER_CONTROL_PLANE,
        )
        stack.callback(runtime_directory.close)
        config_directory = _open_componentwise_directory_v1(
            control_plane.config_directory,
            label="scorer Docker config",
            code=FAIL_DOCKER_CONTROL_PLANE,
        )
        stack.callback(config_directory.close)
        config_file = _open_relative_regular_file_v1(
            config_directory,
            "config.json",
            minimum_bytes=3,
            maximum_bytes=16,
            single_link_required=True,
            label="empty Docker client config",
            code=FAIL_DOCKER_CONTROL_PLANE,
        )
        stack.callback(config_file.close)
        if config_file.payload != b"{}\n":
            _fail(FAIL_DOCKER_CONTROL_PLANE, "Docker client config is not empty")

        path_binding_sha256 = stable_hash(
            {
                "bundle_mount": bundle_mount,
                "output_mount": output_mount,
                "bundle_parent": bundle_parent.component_identities,
                "output": output_directory.component_identities,
                "repository": repository_directory.component_identities,
                "seccomp_parent": seccomp_parent.component_identities,
                "source_modules": source_directory.component_identities,
                "image_payloads": image_payload_directory.component_identities,
                "scorer_source_root_sha256": observed_source_root_sha256,
                "runtime": runtime_directory.component_identities,
                "staged_input": staged_input_directory.component_identities,
            }
        )
        control_plane_binding_sha256 = stable_hash(dict(control_plane.binding))
        image_command = tuple(
            control_plane.command("image", "inspect", spec.scorer_image_digest)
        )
        image_inspect = _run_bounded_command_v1(
            image_command,
            environment=control_plane.environment,
            working_directory=runtime_root,
            timeout_seconds=IMAGE_INSPECT_TIMEOUT_SECONDS_V1,
            maximum_stdout_bytes=MAXIMUM_IMAGE_INSPECT_STDOUT_BYTES_V1,
            maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V1,
            timeout_code=FAIL_DOCKER_TIMEOUT,
            execution_code=FAIL_IMAGE_BINDING,
        )
        if image_inspect.returncode != 0 or image_inspect.stderr != b"":
            _fail(FAIL_IMAGE_BINDING, "local pinned scorer image inspection failed")
        observed_image_id, image_config_sha256 = _validate_image_inspect_v1(
            image_inspect.stdout,
            spec=spec,
        )

        pinned_directories = (
            (bundle_parent, "bundle parent", FAIL_BUNDLE_BINDING),
            (output_directory, "host output", FAIL_OUTPUT_BINDING),
            (repository_directory, "source repository", FAIL_HOST_PATH),
            (seccomp_parent, "seccomp parent", FAIL_SECCOMP_BINDING),
            (runtime_directory, "private runtime", FAIL_DOCKER_CONTROL_PLANE),
            (staged_input_directory, "staged input", FAIL_BUNDLE_BINDING),
            (config_directory, "Docker config", FAIL_DOCKER_CONTROL_PLANE),
            (source_directory, "scorer source modules", FAIL_BUNDLE_BINDING),
            (
                image_payload_directory,
                "scorer image source payloads",
                FAIL_BUNDLE_BINDING,
            ),
        )
        pinned_files = (
            (bundle_file, 1, MAXIMUM_PHASE2B_SCORER_BUNDLE_BYTES_V1, "bundle", FAIL_BUNDLE_BINDING),
            (seccomp_file, 2, MAXIMUM_SECCOMP_BYTES_V1, "seccomp", FAIL_SECCOMP_BINDING),
            (staged_bundle, len(bundle_file.payload), len(bundle_file.payload), "staged bundle", FAIL_BUNDLE_BINDING),
            (staged_seccomp, len(seccomp_file.payload), len(seccomp_file.payload), "staged seccomp", FAIL_SECCOMP_BINDING),
            (config_file, 3, 16, "Docker config", FAIL_DOCKER_CONTROL_PLANE),
        ) + tuple(
            (
                item,
                len(item.payload),
                len(item.payload),
                f"scorer source payload {item.name}",
                FAIL_BUNDLE_BINDING,
            )
            for item in source_payload_files
        )
        for pinned, label, code in pinned_directories:
            _recheck_pinned_directory_v1(pinned, label=label, code=code)
        for pinned, minimum, maximum, label, code in pinned_files:
            _recheck_regular_file_v1(
                pinned,
                minimum_bytes=minimum,
                maximum_bytes=maximum,
                label=label,
                code=code,
            )
        if _directory_names_v1(
            output_directory,
            label="host output before Docker create",
            code=FAIL_OUTPUT_BINDING,
        ):
            _fail(FAIL_OUTPUT_BINDING, "host output changed before Docker create")

        container_name = _container_name_v1(runtime_root)
        create_command = _docker_create_command_v1(
            control_plane,
            spec=spec,
            container_name=container_name,
            staged_input_directory=staged_input_directory.path,
            staged_seccomp_path=runtime_root / _STAGED_SECCOMP_FILENAME_V1,
        )
        named_removed = False

        def remove_named() -> None:
            nonlocal named_removed
            if not named_removed:
                _force_remove_container_v1(
                    control_plane=control_plane,
                    environment=control_plane.environment,
                    working_directory=runtime_root,
                    container_id=container_name,
                    allow_already_absent=True,
                )
                named_removed = True

        created = _run_bounded_command_v1(
            create_command,
            environment=control_plane.environment,
            working_directory=runtime_root,
            timeout_seconds=60,
            maximum_stdout_bytes=256,
            maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V1,
            timeout_code=FAIL_DOCKER_TIMEOUT,
            execution_code=FAIL_DOCKER_EXECUTION,
            forced_cleanup=remove_named,
        )
        if created.returncode != 0 or created.stderr != b"":
            remove_named()
            _fail(FAIL_DOCKER_EXECUTION, "scorer container creation failed")
        try:
            container_id = _parse_created_container_id_v1(created.stdout)
        except BaseException:
            remove_named()
            raise
        removed = False

        def remove_container() -> None:
            nonlocal removed
            if not removed:
                _force_remove_container_v1(
                    control_plane=control_plane,
                    environment=control_plane.environment,
                    working_directory=runtime_root,
                    container_id=container_id,
                )
                removed = True

        try:
            inspect_command = tuple(
                control_plane.command("container", "inspect", container_id)
            )
            inspected = _run_bounded_command_v1(
                inspect_command,
                environment=control_plane.environment,
                working_directory=runtime_root,
                timeout_seconds=30,
                maximum_stdout_bytes=MAXIMUM_IMAGE_INSPECT_STDOUT_BYTES_V1,
                maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V1,
                timeout_code=FAIL_DOCKER_TIMEOUT,
                execution_code=FAIL_DOCKER_EXECUTION,
                forced_cleanup=remove_container,
            )
            if inspected.returncode != 0 or inspected.stderr != b"":
                _fail(FAIL_DOCKER_EXECUTION, "created scorer container inspect failed")
            _validate_created_container_inspect_v1(
                inspected.stdout,
                container_id=container_id,
                container_name=container_name,
                spec=spec,
                staged_input_directory=staged_input_directory.path,
                expected_seccomp=seccomp_value,
            )
            for pinned, minimum, maximum, label, code in pinned_files:
                _recheck_regular_file_v1(
                    pinned,
                    minimum_bytes=minimum,
                    maximum_bytes=maximum,
                    label=f"{label} after create",
                    code=code,
                )
            start_command = tuple(control_plane.command("start", container_id))
            extractor_command = tuple(
                control_plane.command(
                    "container",
                    "exec",
                    f"--user={SCORER_RUNTIME_UID_GID_V1}",
                    container_id,
                    _CONTAINER_SUPERVISOR_ENTRYPOINT_V1,
                    "-c",
                    _CONTAINER_RESULT_EXTRACTOR_CODE_V1,
                )
            )
            started_monotonic_ns = time.monotonic_ns()
            deadline_monotonic_ns = (
                started_monotonic_ns
                + spec.limits.timeout_seconds * 1_000_000_000
            )
            started = _run_bounded_command_v1(
                start_command,
                environment=control_plane.environment,
                working_directory=runtime_root,
                timeout_seconds=60,
                maximum_stdout_bytes=256,
                maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V1,
                timeout_code=FAIL_DOCKER_TIMEOUT,
                execution_code=FAIL_DOCKER_EXECUTION,
                forced_cleanup=remove_container,
            )
            if (
                started.returncode != 0
                or started.stderr != b""
                or started.stdout != (container_id + "\n").encode("ascii")
            ):
                _fail(FAIL_DOCKER_EXECUTION, "scorer container start failed")
            remaining_ns = deadline_monotonic_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                _fail(FAIL_DOCKER_TIMEOUT, "scorer deadline elapsed before extraction")
            result_observation = _run_bounded_command_v1(
                extractor_command,
                environment=control_plane.environment,
                working_directory=runtime_root,
                timeout_seconds=max(1, (remaining_ns + 999_999_999) // 1_000_000_000),
                maximum_stdout_bytes=MAXIMUM_DOCKER_RESULT_WIRE_BYTES_V1,
                maximum_stderr_bytes=MAXIMUM_DOCKER_STDERR_BYTES_V1,
                timeout_code=FAIL_DOCKER_TIMEOUT,
                execution_code=FAIL_OUTPUT_BINDING,
                forced_cleanup=remove_container,
            )
            if result_observation.returncode != 0 or result_observation.stderr != b"":
                _fail(FAIL_OUTPUT_BINDING, "scorer report extraction failed")
            container_score_report, container_gate_report = (
                _extract_container_result_wire_v1(result_observation.stdout)
            )
            runtime_receipt_payload, log_observation = _capture_runtime_receipt_v1(
                control_plane=control_plane,
                environment=control_plane.environment,
                working_directory=runtime_root,
                container_id=container_id,
                deadline_monotonic_ns=deadline_monotonic_ns,
                forced_cleanup=remove_container,
            )
            finished_monotonic_ns = max(
                result_observation.finished_monotonic_ns,
                log_observation.finished_monotonic_ns,
            )
            if finished_monotonic_ns > deadline_monotonic_ns:
                _fail(FAIL_DOCKER_TIMEOUT, "scorer output arrived after deadline")
        finally:
            remove_container()

        for pinned, label, code in pinned_directories:
            _recheck_pinned_directory_v1(
                pinned,
                label=f"{label} after container removal",
                code=code,
            )
        for pinned, minimum, maximum, label, code in pinned_files:
            _recheck_regular_file_v1(
                pinned,
                minimum_bytes=minimum,
                maximum_bytes=maximum,
                label=f"{label} after container removal",
                code=code,
            )
        if (
            container_score_report != score_report
            or container_gate_report != gate_report
        ):
            _fail(FAIL_OUTPUT_BINDING, "container and host scorer reports differ")
        runtime_receipt_id = _validate_runtime_receipt_v1(
            runtime_receipt_payload,
            bundle_payload=bundle_file.payload,
            decoded=decoded,
            evaluation=evaluation,
            score_report=score_report,
            gate_report=gate_report,
        )
        if _directory_names_v1(
            output_directory,
            label="host output before publication",
            code=FAIL_OUTPUT_BINDING,
        ):
            _fail(FAIL_OUTPUT_BINDING, "host output changed during Docker execution")
        score_file, gate_file = _publish_reports_v1(
            output_directory,
            score_report=score_report,
            gate_report=gate_report,
        )
        stack.callback(score_file.close)
        stack.callback(gate_file.close)

        return _issue_receipt_after_publication_v1(
            output_directory,
            score_file,
            gate_file,
            schema_version=LOCAL_SCORER_EXECUTION_RECEIPT_SCHEMA_V1,
            disposition=LOCAL_SCORER_EXECUTION_DISPOSITION_V1,
            reason=LOCAL_SCORER_EXECUTION_COMPLETE_REASON_V1,
            run_spec_id=spec.run_spec_id,
            authority_requirement_id=spec.authority_requirement_id,
            actor_profile_content_id=spec.actor_profile_content_id,
            actor_profile_canonical_sha256=spec.actor_profile_canonical_sha256,
            campaign_id=spec.campaign_id,
            attempt_id=spec.attempt_id,
            execution_registry_id=spec.execution_registry_id,
            scorer_image_digest=spec.scorer_image_digest,
            scorer_source_root_sha256=spec.scorer_source_root_sha256,
            scoring_mechanics_source_sha256=(
                spec.scoring_mechanics_source_sha256
            ),
            protocol_id=spec.protocol_id,
            exact_freeze_id=spec.exact_freeze_id,
            execution_freeze_manifest_id=spec.execution_freeze_manifest_id,
            scorer_run_manifest_id=spec.scorer_run_manifest_id,
            observed_image_id=observed_image_id,
            image_inspect_sha256=_sha256(image_inspect.stdout),
            image_config_projection_sha256=image_config_sha256,
            docker_control_plane_binding_sha256=control_plane_binding_sha256,
            docker_executable_sha256=str(
                control_plane.binding["docker_executable_sha256"]
            ),
            seccomp_profile_sha256=seccomp_file.sha256,
            host_path_binding_sha256=path_binding_sha256,
            command_sha256=stable_hash(
                {
                    "image_inspect": image_command,
                    "create": create_command,
                    "inspect": inspect_command,
                    "start": start_command,
                    "extract": extractor_command,
                    "logs": log_observation.command,
                    "remove": tuple(
                        control_plane.command("container", "rm", "--force", container_id)
                    ),
                }
            ),
            container_supervisor_sha256=_sha256(
                (
                    _CONTAINER_SUPERVISOR_ENTRYPOINT_V1
                    + "\x00"
                    + _CONTAINER_SUPERVISOR_CODE_V1
                ).encode("utf-8")
            ),
            container_result_extractor_sha256=_sha256(
                (
                    _CONTAINER_SUPERVISOR_ENTRYPOINT_V1
                    + "\x00"
                    + _CONTAINER_RESULT_EXTRACTOR_CODE_V1
                ).encode("utf-8")
            ),
            bundle_id=decoded.bundle_id,
            bundle_sha256=bundle_file.sha256,
            bundle_bytes=len(bundle_file.payload),
            runtime_receipt_id=runtime_receipt_id,
            runtime_receipt_sha256=_sha256(runtime_receipt_payload),
            score_mechanics_result_id=evaluation.scoring_mechanics.result_id,
            replay_input_contract_result_id=evaluation.replay_input_contract.result_id,
            available_gate_mechanics_result_id=evaluation.available_gate_mechanics.result_id,
            score_report_id=evaluation.score_report_id,
            score_report_sha256=score_file.sha256,
            score_report_bytes=len(score_report),
            score_output_file_identity_sha256=stable_hash(score_file.identity),
            available_gate_report_id=evaluation.available_gate_report_id,
            available_gate_report_sha256=gate_file.sha256,
            available_gate_report_bytes=len(gate_report),
            available_gate_output_file_identity_sha256=stable_hash(gate_file.identity),
            started_monotonic_ns=started_monotonic_ns,
            finished_monotonic_ns=finished_monotonic_ns,
            process_exit_code=0,
            timed_out=False,
            docker_invoked=True,
            local_mechanics_complete=True,
            host_bundle_replay_verified=True,
            container_bundle_replay_verified=True,
            strict_input_prediction_replayed=True,
            public_partition_rebuilt=True,
            nine_metric_mechanics_materialized=True,
            ten_overall_gate_mechanics_materialized=True,
            twenty_four_slice_gate_mechanics_materialized=True,
            two_unavailable_gate_inputs_retained=True,
            external_runtime_attestation_issued=False,
            reveal_authority_verified=False,
            prediction_commit_before_reveal_verified=False,
            actual_960_case_run_verified=False,
            scoring_performed=False,
            prediction_scored=False,
            formal_gate_evaluation_performed=False,
            effect_evidence=False,
            formal_c1_report_verified=False,
            c1_exit_evidence=False,
            malicious_host_root_resistance=False,
            malicious_docker_daemon_resistance=False,
        )


__all__ = (
    "FAIL_BUNDLE_BINDING",
    "FAIL_DOCKER_CAPTURE",
    "FAIL_DOCKER_CONTROL_PLANE",
    "FAIL_DOCKER_EXECUTION",
    "FAIL_DOCKER_TIMEOUT",
    "FAIL_HOST_PATH",
    "FAIL_IMAGE_BINDING",
    "FAIL_OUTPUT_BINDING",
    "FAIL_PREFLIGHT",
    "FAIL_PROFILE_BINDING",
    "FAIL_SECCOMP_BINDING",
    "FROZEN_SCORER_ENTRYPOINT_V1",
    "IMAGE_INSPECT_TIMEOUT_SECONDS_V1",
    "LOCAL_SCORER_EXECUTION_COMPLETE_REASON_V1",
    "LOCAL_SCORER_EXECUTION_DISPOSITION_V1",
    "LOCAL_SCORER_EXECUTION_RECEIPT_SCHEMA_V1",
    "MAXIMUM_DOCKER_STDERR_BYTES_V1",
    "MAXIMUM_DOCKER_STDOUT_BYTES_V1",
    "Phase2BScorerHostExecutorErrorV1",
    "Phase2BScorerLocalExecutionReceiptV1",
    "Phase2BScorerRunSpecV1",
    "Phase2BScorerRuntimeLimitsV1",
    "SCORER_RUNTIME_BASE_DIGEST_V1",
    "SCORER_RUNTIME_IMAGE_ARGS_ESCAPED_V1",
    "SCORER_RUNTIME_IMAGE_CONFIG_FIELDS_V1",
    "SCORER_RUNTIME_IMAGE_DESCRIPTION_V1",
    "SCORER_RUNTIME_IMAGE_TITLE_V1",
    "SCORER_RUNTIME_UID_GID_V1",
    "SCORER_SECCOMP_SHA256_V1",
    "SCORER_TIMEOUT_SECONDS_V1",
    "SCORING_MECHANICS_SOURCE_SHA256_V1",
    "execute_phase2b_scorer_docker_v1",
)
