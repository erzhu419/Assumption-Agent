"""Pure target-blind Q0.5b actual-admission wire contracts.

This module performs no filesystem, Git, Docker, process, target, truth,
split, or role-membership operation.  Runtime collectors live in the outer
supervisor.  They provide exact preimages to these builders; both the
supervisor and the final artifact replayer use this module to replay the same
bundle, one-attempt decision, and stage-3-to-4 boundary bytes.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Final, Mapping, NoReturn, Sequence


ACTUAL_ADMISSION_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-actual-admission-decision/1"
)
ACTUAL_PRECONDITION_BUNDLE_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-actual-admission-precondition-bundle/1"
)
ACTUAL_ADMISSION_BOUNDARY_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-stage3-to4-admission-boundary/1"
)
ACTUAL_ADMISSION_DECISION_ID: Final = "ADMITTED_FOR_ONE_ATTEMPT"
ACTUAL_ADMISSION_DECISION_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/ADMISSION_DECISION/V1\x00"
)
ACTUAL_ADMISSION_ATTEMPT_ID_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/ATTEMPT_ID/V1\x00"
)
DOCKER_EXECUTION_AUTHORITY_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-docker-execution-authority/1"
)
DOCKER_AUTHORITATIVE_ABSENCE_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-docker-authoritative-absence/1"
)
DOCKER_OWNERSHIP_NAMESPACE_DOMAIN: Final = (
    b"HEGEL/Q05B/DOCKER/OWNERSHIP_NAMESPACE/V1\x00"
)
DOCKER_INITIAL_NAME_ABSENCE_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/DOCKER/INITIAL_NAME_ABSENCE/V1\x00"
)
DOCKER_PRECREATE_ABSENCE_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-docker-precreate-absence/1"
)
DOCKER_PRECREATE_ABSENCE_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/DOCKER/PRECREATE_ABSENCE/V1\x00"
)
DOCKER_EXECUTION_AUTHORITY_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/DOCKER/EXECUTION_AUTHORITY/V1\x00"
)
DOCKER_RESERVED_LABEL_KEYS: Final = (
    "org.hegel.q05b.execution_namespace",
    "org.hegel.q05b.slot",
    "org.hegel.q05b.source_commit",
)
DOCKER_RUST_BASE_LABEL_ROWS: Final = (
    (
        "org.opencontainers.image.source",
        "https://github.com/rust-lang/docker-rust",
    ),
)
DOCKER_EXECUTION_SLOT_REGISTRY: Final = (
    (1, "RUST_TEST", "rust-test"),
    (2, "RUST_RELEASE", "rust-release"),
    (3, "PYTHON_ENDPOINT", "python"),
    (4, "RUST_ENDPOINT", "rust"),
    (5, "TRUSTED_HOST_REPLAY", "host"),
)
DOCKER_CONTAINER_NAME_USAGE: Final = "READ_ONLY_DISCOVERY_ONLY"
DOCKER_DESTRUCTIVE_TARGET: Final = (
    "OWNERSHIP_VALIDATED_64_LOWERHEX_CONTAINER_ID_ONLY"
)
ACTUAL_PRECONDITION_EVIDENCE_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/PRECONDITION_EVIDENCE/V1\x00"
)
ACTUAL_PRECONDITION_BUNDLE_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/PRECONDITION_BUNDLE/V1\x00"
)
ACTUAL_ADMISSION_BOUNDARY_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/STAGE3_TO4_ADMISSION_BOUNDARY/V1\x00"
)
ACTUAL_ADMISSION_ISSUE_RECORD_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-stage3-to4-admission-issue-record/1"
)
ACTUAL_ADMISSION_ISSUED_MARKER_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-admission-issued-marker-evidence/1"
)
ACTUAL_ADMISSION_SPENDING_INTENT_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-admission-spending-intent/1"
)
ACTUAL_ADMISSION_CONSUMED_MARKER_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-admission-consumed-marker-evidence/1"
)
ACTUAL_ADMISSION_LIVE_MARKER_REPLAY_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-live-admission-marker-replay/1"
)
ACTUAL_ADMISSION_ISSUED_MARKER_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/ISSUED_MARKER/V1\x00"
)
ACTUAL_ADMISSION_SPENDING_INTENT_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/SPENDING_INTENT/V1\x00"
)
ACTUAL_ADMISSION_CONSUMED_MARKER_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/CONSUMED_MARKER/V1\x00"
)
ACTUAL_ADMISSION_ISSUE_RECORD_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/ISSUE_RECORD/V1\x00"
)
ACTUAL_ADMISSION_LIVE_MARKER_REPLAY_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/LIVE_MARKER_REPLAY/V1\x00"
)
ACTUAL_ADMISSION_RUN_LOCAL_ANTI_REPLAY_SCOPE: Final = {
    "scope": (
        "ONE_SUPERVISOR_INVOCATION_EXACT_ANCHORED_WORK_ROOT_"
        "INODE_GENERATED_NONCE_ATTEMPT"
    ),
    "supervisor_generated_nonce_required": True,
    "cli_accepts_boundary_or_nonce": False,
    "crash_restart_resume_claimed": False,
    "cross_process_global_replay_protection_claimed": False,
    "work_root_cleanup_preserves_history_claimed": False,
}
ACTUAL_PRECONDITION_REGISTRY_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/PRECONDITION_REGISTRY/V1\x00"
)
ACTUAL_GIT_SOURCE_TRANSCRIPT_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/GIT_SOURCE_TRANSCRIPT/V1\x00"
)
ACTUAL_GIT_SOURCE_TRANSCRIPT_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-admission-git-source-transcript/1"
)
ACTUAL_FRESH_RUNTIME_EVIDENCE_SET_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-fresh-runtime-evidence-set/1"
)
ACTUAL_FRESH_RUNTIME_EVIDENCE_OBJECT_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/FRESH_RUNTIME_EVIDENCE_OBJECT/V1\x00"
)
ACTUAL_FRESH_RUNTIME_EVIDENCE_SET_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/FRESH_RUNTIME_EVIDENCE_SET/V1\x00"
)
ACTUAL_PRECONDITION_BUNDLE_MAX_BYTES: Final = 4 * 1024 * 1024
ACTUAL_ADMISSION_BOUNDARY_MAX_BYTES: Final = 12 * 1024 * 1024
ACTUAL_FRESH_RUNTIME_CHECKPOINT_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-fresh-runtime-checkpoint/1"
)
ACTUAL_FRESH_RUNTIME_CHECKPOINT_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/FRESH_RUNTIME_CHECKPOINT/V1\x00"
)
ACTUAL_FRESH_RUNTIME_CHECKPOINT_MAX_BYTES: Final = 4 * 1024 * 1024
ACTUAL_FRESH_RUNTIME_CHECKPOINT_REGISTRY: Final = (
    (1, "CONSUME_AFTER_SPEND_BEFORE_ENDPOINTS"),
    (2, "STAGE6_BEFORE_HOST_LAUNCH"),
    (3, "STAGE7_BEFORE_PREDICATE19"),
)
ACTUAL_ACTOR_MOUNT_BINDING_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-actor-prelaunch-mount-binding/1"
)
ACTUAL_ACTOR_MOUNT_SOURCE_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-actor-prelaunch-mount-source/1"
)
ACTUAL_ACTOR_MOUNT_BINDING_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/ACTOR_MOUNT_BINDING/V1\x00"
)
ACTUAL_ACTOR_MOUNT_SOURCE_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/ACTOR_MOUNT_SOURCE/V1\x00"
)
ACTUAL_ACTOR_MOUNT_AUTHORITY_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/ACTOR_MOUNT_AUTHORITY/V1\x00"
)
ACTUAL_CHECKPOINT_MOUNT_REGISTRY_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/CHECKPOINT_MOUNT_REGISTRY/V1\x00"
)
ACTUAL_ACTOR_MOUNT_LAUNCH_REPLAY_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-actor-mount-launch-replay/1"
)
ACTUAL_ACTOR_MOUNT_LAUNCH_REPLAY_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/ACTOR_MOUNT_LAUNCH_REPLAY/V1\x00"
)
ACTUAL_DYNAMIC_MOUNT_AUTHORITY_SET_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-dynamic-mount-authority-set/1"
)
ACTUAL_DYNAMIC_MOUNT_AUTHORITY_SET_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/DYNAMIC_MOUNT_AUTHORITY_SET/V1\x00"
)
ACTUAL_STAGE_EVIDENCE_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/STAGE_EVIDENCE/V1\x00"
)
ACTUAL_STAGE_SCHEMA_VERSION: Final = (
    "hegel-phase3a-q05b-actual-orchestration-stage-evidence/1"
)
ACTUAL_STAGE_5_NAME: Final = (
    "SEAL_ENDPOINT_STDOUT_SIDECARS_CONTROLS_AND_RESOURCES"
)
ACTUAL_STAGE_5_BASE_EVIDENCE_KEYS: Final = (
    "actor_completion_rows",
    "five_sidecars",
    "endpoint_stdout_set",
    "strict_endpoint_replay_roots",
    "qualification_receipt",
)
ACTUAL_STAGE_5_INJECTED_EVIDENCE_KEYS: Final = (
    "actual_admission_attempt_id",
    "actual_admission_boundary_root",
    "actual_admission_issue_record_root",
    "actual_admission_consumed_marker_evidence",
    "actual_admission_work_root_replay",
    "actual_admission_consume_git_source_transcript",
    "actual_admission_consume_artifact_absence",
    "actual_admission_fresh_checkpoint_root_rows",
    "actual_actor_mount_binding_root_rows",
    "actual_actor_mount_launch_root_rows",
    "actual_admission_live_marker_replay",
)
ACTUAL_ACTOR_MOUNT_ROLE_REGISTRY: Final = (
    (
        1,
        "PYTHON_ENDPOINT",
        (
            ("/control", True, "DIRECTORY", 0o700),
            ("/output", True, "DIRECTORY", 0o700),
            ("/snapshot", False, "DIRECTORY", 0o555),
        ),
    ),
    (
        2,
        "RUST_ENDPOINT",
        (
            ("/control", True, "DIRECTORY", 0o700),
            ("/output", True, "DIRECTORY", 0o700),
            (
                "/runtime/hegel-q1-archive-projection-oracle",
                False,
                "REGULAR_FILE",
                0o555,
            ),
        ),
    ),
    (
        3,
        "TRUSTED_HOST_REPLAY",
        (
            ("/control", True, "DIRECTORY", 0o700),
            ("/inputs/python", False, "DIRECTORY", 0o555),
            ("/inputs/rust", False, "DIRECTORY", 0o555),
            (
                "/inputs/stdout/manifest.json",
                False,
                "REGULAR_FILE",
                0o444,
            ),
            (
                "/inputs/stdout/python.stdout",
                False,
                "REGULAR_FILE",
                0o444,
            ),
            (
                "/inputs/stdout/rust.stdout",
                False,
                "REGULAR_FILE",
                0o444,
            ),
            ("/snapshot", False, "DIRECTORY", 0o555),
            ("/staging", True, "DIRECTORY", 0o700),
        ),
    ),
)
ACTUAL_PRELAUNCH_WRITABLE_DIRECTORY_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/PRELAUNCH_WRITABLE_DIRECTORY/V1\x00"
)
ACTUAL_RUNTIME_SECCOMP_RELATIVE_PATH: Final = (
    "config/phase3_internal_actor_seccomp_v1.json"
)
ACTUAL_ACTOR_MOUNT_AUTHORITY_REGISTRY: Final = (
    (1, "/control", "PRELAUNCH_WRITABLE_DIRECTORY", "PYTHON_ENDPOINT/control"),
    (1, "/output", "PRELAUNCH_WRITABLE_DIRECTORY", "PYTHON_ENDPOINT/output"),
    (1, "/snapshot", "FRESH_ACTOR_SNAPSHOT", "PYTHON_ENDPOINT/snapshot"),
    (1, "@seccomp", "RUNTIME_SECCOMP_POLICY", "PYTHON_ENDPOINT/@seccomp"),
    (2, "/control", "PRELAUNCH_WRITABLE_DIRECTORY", "RUST_ENDPOINT/control"),
    (2, "/output", "PRELAUNCH_WRITABLE_DIRECTORY", "RUST_ENDPOINT/output"),
    (
        2,
        "/runtime/hegel-q1-archive-projection-oracle",
        "FRESH_PREBUILT_RUST_BINARY",
        "RUST_ENDPOINT/runtime",
    ),
    (2, "@seccomp", "RUNTIME_SECCOMP_POLICY", "RUST_ENDPOINT/@seccomp"),
    (3, "/control", "PRELAUNCH_WRITABLE_DIRECTORY", "TRUSTED_HOST_REPLAY/control"),
    (3, "/inputs/python", "SEALED_ENDPOINT_TREE", "PYTHON_ENDPOINT/output"),
    (3, "/inputs/rust", "SEALED_ENDPOINT_TREE", "RUST_ENDPOINT/output"),
    (
        3,
        "/inputs/stdout/manifest.json",
        "SEALED_STDOUT_FILE",
        "TRUSTED_HOST_REPLAY/inputs/stdout/manifest.json",
    ),
    (
        3,
        "/inputs/stdout/python.stdout",
        "SEALED_STDOUT_FILE",
        "TRUSTED_HOST_REPLAY/inputs/stdout/python.stdout",
    ),
    (
        3,
        "/inputs/stdout/rust.stdout",
        "SEALED_STDOUT_FILE",
        "TRUSTED_HOST_REPLAY/inputs/stdout/rust.stdout",
    ),
    (3, "/snapshot", "FRESH_ACTOR_SNAPSHOT", "TRUSTED_HOST_REPLAY/snapshot"),
    (3, "/staging", "PRELAUNCH_WRITABLE_DIRECTORY", "TRUSTED_HOST_REPLAY/staging"),
    (3, "@seccomp", "RUNTIME_SECCOMP_POLICY", "TRUSTED_HOST_REPLAY/@seccomp"),
)
ACTUAL_COMMIT_A_STATIC_POLICY_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/COMMIT_A_STATIC_POLICY/V1\x00"
)
ACTUAL_COMMAND_MOUNT_RESOURCE_POLICY_ROOT_DOMAIN: Final = (
    b"HEGEL/Q05B/ACTUAL/COMMAND_MOUNT_RESOURCE_POLICY/V1\x00"
)
EXPECTED_COMMIT_A_STATIC_POLICY_ROOT: Final = (
    "56625afd7f459ca877f620b39dff8391c158602d454b1f9066c219b10308d2b2"
)
EXPECTED_COMMAND_MOUNT_RESOURCE_POLICY_ROOT: Final = (
    "90abdfffedd28998994250739528502cfd7e1d44366de8ff41ac19653bf07502"
)
COMMAND_MOUNT_RESOURCE_POLICY_FIELDS: Final = (
    "docker",
    "mount_policy",
    "live_resource_evidence_policy",
    "runtime_command_inspect_policy",
    "actor_commands",
)
EXPECTED_PYTHON_IMAGE_REFERENCE: Final = (
    "python@sha256:"
    "e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"
)
EXPECTED_RUST_IMAGE_REFERENCE: Final = (
    "rust@sha256:"
    "38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89"
)
ACTUAL_RUNTIME_PRECONDITION_REGISTRY: Final = (
    (1, b"CLEAN_HEAD_EQUALS_REQUESTED_FULL40_COMMIT"),
    (2, b"COMMIT_A_CONFIG_BYTES_EQUAL_GIT_BLOB_AND_RUNTIME"),
    (3, b"ACTUAL_ENTRYPOINT_IMPLEMENTED_AND_CONDITIONAL_POLICY"),
    (4, b"ARTIFACT_TARGET_ABSENT_AT_ADMISSION"),
    (5, b"PINNED_LOCAL_IMAGE_IDENTITIES_VERIFIED"),
    (6, b"ACTOR_SOURCE_AND_SNAPSHOT_IDENTITIES_VERIFIED"),
    (7, b"SEALED_CARGO_AND_OFFLINE_BUILD_POLICY_VERIFIED"),
    (8, b"PREBUILT_RUNTIME_AND_SECCOMP_IDENTITIES_VERIFIED"),
    (9, b"PRELAUNCH_COMMAND_MOUNT_RESOURCE_POLICY_BOUND"),
    (10, b"Q1_AUTHORITY_CLOSED_AT_ZERO_OF_TWENTY"),
    (11, b"TOCTOU_REVALIDATION_POLICY_BOUND_AT_ADMISSION"),
    (12, b"ATOMIC_NOREPLACE_PUBLICATION_POLICY_BOUND_AT_ADMISSION"),
)
ACTUAL_PRECONDITION_PREIMAGE_SCHEMAS: Final = (
    "hegel-phase3a-q05b-admission-clean-head-preimage/1",
    "hegel-phase3a-q05b-admission-commit-a-config-preimage/1",
    "hegel-phase3a-q05b-admission-implementation-policy-preimage/1",
    "hegel-phase3a-q05b-admission-artifact-absence-preimage/1",
    "hegel-phase3a-q05b-admission-pinned-images-preimage/1",
    "hegel-phase3a-q05b-admission-source-snapshots-preimage/1",
    "hegel-phase3a-q05b-admission-cargo-build-preimage/1",
    "hegel-phase3a-q05b-admission-runtime-seccomp-preimage/1",
    "hegel-phase3a-q05b-admission-prelaunch-policy-preimage/1",
    "hegel-phase3a-q05b-admission-closed-q1-preimage/1",
    "hegel-phase3a-q05b-admission-toctou-policy-preimage/1",
    "hegel-phase3a-q05b-admission-publication-policy-preimage/1",
)
COMMIT_A_ACTUAL_ENGINEERING_STATUS: Final = (
    "ACTUAL_IMPLEMENTED_CONDITIONALLY_ADMITTED_NOT_EXECUTED"
)
COMMIT_A_ACTUAL_PRECONDITIONS_V1: Final = {
    "actual_entrypoint_implemented": True,
    "source_freeze_execution_status": "NOT_EXECUTED_AT_COMMIT_A",
    "execution_admission_policy": "CONDITIONAL_SINGLE_ATTEMPT_RUNTIME_ADMISSION",
    "implementation_blocked_predicate_ids": [],
    "pending_actual_evidence_predicate_ids": list(range(1, 21)),
    "clean_full40_commit_required": True,
    "head_must_equal_requested_commit": True,
    "commit_a_config_blob_must_equal_runtime_config": True,
    "pinned_local_images_required": True,
    "sealed_source_cargo_runtime_required": True,
    "attempt_unique_docker_execution_authority_required": True,
    "initial_and_precreate_name_absence_required": True,
    "docker_cleanup_owned_cid_only_required": True,
    "foreign_or_unknown_docker_state_zero_mutation_required": True,
    "artifact_target_must_be_absent": True,
    "all_runtime_preconditions_required": True,
    "all_20_predicates_required_before_artifact": True,
    "toctou_revalidation_required": True,
    "atomic_noreplace_artifact_required": True,
}
ACTUAL_ADMISSION_QUALIFICATION_AUTHORITY: Final = {
    "candidate_receipt": None,
    "final_receipt": None,
    "predicate_count": 0,
    "predicate_mask": 0,
    "predicate_total": 20,
}
ACTUAL_ADMISSION_CLOSED_Q1_AUTHORITY: Final = {
    "active_transition_allowed": False,
    "certificate_active": False,
    "formal_fixed_point_claimed": False,
    "formal_output_roots": [None] * 8,
    "gate_count": 0,
    "gate_mask": 0,
    "gate_total": 20,
    "m3_formal_roots": None,
    "outside_certificate_issued": False,
    "q1_receipt": None,
    "q2_state": "NOT_RUN",
    "state": "NOT_RUN",
}


class Q05BActualAdmissionError(ValueError):
    """Stable pure admission-wire rejection."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _fail(detail: str) -> NoReturn:
    raise Q05BActualAdmissionError(detail)


def canonical_json_bytes_v1(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        _fail(f"actual admission JSON differs: {error}")


def _strict_json_object_v1(payload: bytes, name: str) -> dict[str, object]:
    if type(payload) is not bytes or not payload:
        _fail(f"{name} bytes differ")

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if type(key) is not str or key in result:
                _fail(f"{name} duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("ascii", "strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: _fail(f"{name} non-finite {token}"),
            parse_float=lambda token: _fail(f"{name} float {token}"),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail(f"{name} is not strict JSON: {error}")
    if type(value) is not dict or canonical_json_bytes_v1(value) != payload:
        _fail(f"{name} is not one canonical object")
    return value


def _require_type_exact_v1(value: object, expected: object, name: str) -> None:
    if type(value) is not type(expected):
        _fail(f"{name} type differs")
    if type(expected) is dict:
        assert type(value) is dict
        if set(value) != set(expected):
            _fail(f"{name} fields differ")
        for key in expected:
            _require_type_exact_v1(value[key], expected[key], f"{name}.{key}")
    elif type(expected) is list:
        assert type(value) is list
        if len(value) != len(expected):
            _fail(f"{name} length differs")
        for index, (item, expected_item) in enumerate(
            zip(value, expected, strict=True)
        ):
            _require_type_exact_v1(item, expected_item, f"{name}[{index}]")
    elif value != expected:
        _fail(f"{name} value differs")


def _root_hex_v1(value: object, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _fail(f"{name} root differs")
    return value


def _commit_v1(value: object, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        _fail(f"{name} commit differs")
    return value


def _absolute_path_text_v1(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value.startswith("/")
        or ".." in value.split("/")
    ):
        _fail(f"{name} path differs")
    return value


def _hex_bytes_v1(value: object, name: str, *, allow_empty: bool = False) -> bytes:
    if (
        type(value) is not str
        or len(value) % 2
        or (not value and not allow_empty)
        or (value and re.fullmatch(r"[0-9a-f]+", value) is None)
    ):
        _fail(f"{name} hex differs")
    try:
        return bytes.fromhex(value)
    except ValueError as error:
        _fail(f"{name} hex differs: {error}")


def _docker_ownership_namespace_v1(
    source_commit: str,
    attempt_nonce: bytes,
) -> str:
    commit = _commit_v1(source_commit, "Docker ownership source")
    if type(attempt_nonce) is not bytes or len(attempt_nonce) != 32:
        _fail("Docker ownership attempt nonce differs")
    return sha256(
        DOCKER_OWNERSHIP_NAMESPACE_DOMAIN
        + attempt_nonce
        + commit.encode("ascii")
    ).hexdigest()


def _docker_slot_rows_from_namespace_v1(
    source_commit: str,
    execution_namespace: str,
) -> list[dict[str, object]]:
    commit = _commit_v1(source_commit, "Docker slot source")
    namespace = _root_hex_v1(
        execution_namespace, "Docker ownership namespace"
    )
    rows: list[dict[str, object]] = []
    for slot_id, slot, suffix in DOCKER_EXECUTION_SLOT_REGISTRY:
        labels = [
            [DOCKER_RESERVED_LABEL_KEYS[0], namespace],
            [DOCKER_RESERVED_LABEL_KEYS[1], slot],
            [DOCKER_RESERVED_LABEL_KEYS[2], commit],
        ]
        expected_container_labels = list(labels)
        if slot in {"RUST_TEST", "RUST_RELEASE", "RUST_ENDPOINT"}:
            expected_container_labels.extend(
                [key, value] for key, value in DOCKER_RUST_BASE_LABEL_ROWS
            )
        expected_container_labels.sort(key=lambda row: row[0])
        rows.append(
            {
                "slot_id": slot_id,
                "slot": slot,
                "container_name": f"hegel-q05b-{namespace}-{suffix}",
                "labels": labels,
                "expected_container_labels": expected_container_labels,
            }
        )
    return rows


def docker_execution_slot_rows_v1(
    source_commit: str,
    attempt_nonce: bytes,
) -> list[dict[str, object]]:
    """Derive the five ordered, attempt-unique Docker launch identities."""

    namespace = _docker_ownership_namespace_v1(source_commit, attempt_nonce)
    return _docker_slot_rows_from_namespace_v1(source_commit, namespace)


def validate_docker_authoritative_absence_v1(
    value: object,
    expected_container_identity: str,
) -> dict[str, object]:
    """Validate one target-bound, read-only Docker inspect absence sample."""

    if (
        type(expected_container_identity) is not str
        or re.fullmatch(
            r"[a-z0-9][a-z0-9_.-]{0,127}", expected_container_identity
        )
        is None
    ):
        _fail("Docker authoritative absence target differs")
    keys = {
        "schema_version",
        "container_identity",
        "inspect_exit_code",
        "inspect_stdout_hex",
        "inspect_stdout_sha256",
        "inspect_stderr_hex",
        "inspect_stderr_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("Docker authoritative absence fields differ")
    if (
        value["schema_version"]
        != DOCKER_AUTHORITATIVE_ABSENCE_SCHEMA_VERSION
        or type(value["container_identity"]) is not str
        or value["container_identity"] != expected_container_identity
        or type(value["inspect_exit_code"]) is not int
        or value["inspect_exit_code"] != 1
    ):
        _fail("Docker authoritative absence identity differs")
    stdout = _hex_bytes_v1(
        value["inspect_stdout_hex"],
        "Docker authoritative absence stdout",
        allow_empty=True,
    )
    stderr = _hex_bytes_v1(
        value["inspect_stderr_hex"],
        "Docker authoritative absence stderr",
        allow_empty=True,
    )
    stdout_sha256 = _root_hex_v1(
        value["inspect_stdout_sha256"],
        "Docker authoritative absence stdout",
    )
    stderr_sha256 = _root_hex_v1(
        value["inspect_stderr_sha256"],
        "Docker authoritative absence stderr",
    )
    if (
        stdout_sha256 != sha256(stdout).hexdigest()
        or stderr_sha256 != sha256(stderr).hexdigest()
    ):
        _fail("Docker authoritative absence payload digest differs")
    target = expected_container_identity
    authoritative_not_found = {
        (b"", f"Error: No such object: {target}\n".encode("ascii")),
        (b"", f"Error: No such container: {target}\n".encode("ascii")),
        (
            b"",
            (
                "Error response from daemon: No such container: "
                f"{target}\n"
            ).encode("ascii"),
        ),
        (
            b"[]\n",
            f"error: no such object: {target}\n".encode("ascii"),
        ),
    }
    if (stdout, stderr) not in authoritative_not_found:
        _fail("Docker inspect was not authoritative target absence")
    return value


def _docker_slot_spec_v1(
    ordered_slot_rows: object,
    slot_id: int,
) -> dict[str, object]:
    if type(slot_id) is not int or not 1 <= slot_id <= 5:
        _fail("Docker execution slot id differs")
    if type(ordered_slot_rows) is not list or len(ordered_slot_rows) != 5:
        _fail("Docker execution slot registry differs")
    row = ordered_slot_rows[slot_id - 1]
    if type(row) is not dict or row.get("slot_id") != slot_id:
        _fail("Docker execution slot selection differs")
    return row


def _docker_absence_sample_rows_v1(
    first_absence: object,
    second_absence: object,
    container_name: str,
) -> list[list[object]]:
    first = validate_docker_authoritative_absence_v1(
        first_absence, container_name
    )
    second = validate_docker_authoritative_absence_v1(
        second_absence, container_name
    )
    return [[1, dict(first)], [2, dict(second)]]


def _build_docker_initial_name_absence_row_from_spec_v1(
    spec: Mapping[str, object],
    first_absence: object,
    second_absence: object,
) -> dict[str, object]:
    if type(spec) is not dict:
        _fail("Docker initial absence slot differs")
    slot_id = spec.get("slot_id")
    slot = spec.get("slot")
    container_name = spec.get("container_name")
    if (
        type(slot_id) is not int
        or type(slot) is not str
        or type(container_name) is not str
    ):
        _fail("Docker initial absence slot identity differs")
    samples = _docker_absence_sample_rows_v1(
        first_absence, second_absence, container_name
    )
    body: dict[str, object] = {
        "slot_id": slot_id,
        "slot": slot,
        "container_name": container_name,
        "inspect_target": container_name,
        "inspect_command": [
            "/usr/bin/docker",
            "--host=unix:///var/run/docker.sock",
            "inspect",
            container_name,
        ],
        "authoritative_absence_samples": samples,
    }
    value = dict(body)
    value["absence_manifest_sha256"] = sha256(
        DOCKER_INITIAL_NAME_ABSENCE_ROOT_DOMAIN
        + canonical_json_bytes_v1(body)
    ).hexdigest()
    return value


def build_docker_initial_name_absence_row_v1(
    source_commit: str,
    attempt_nonce: bytes,
    slot_id: int,
    first_absence: Mapping[str, object],
    second_absence: Mapping[str, object],
) -> dict[str, object]:
    slots = docker_execution_slot_rows_v1(source_commit, attempt_nonce)
    spec = _docker_slot_spec_v1(slots, slot_id)
    return _build_docker_initial_name_absence_row_from_spec_v1(
        spec, first_absence, second_absence
    )


def _validate_docker_initial_name_absence_row_v1(
    value: object,
    spec: Mapping[str, object],
) -> dict[str, object]:
    keys = {
        "slot_id",
        "slot",
        "container_name",
        "inspect_target",
        "inspect_command",
        "authoritative_absence_samples",
        "absence_manifest_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("Docker initial name absence fields differ")
    samples = value["authoritative_absence_samples"]
    if (
        type(samples) is not list
        or len(samples) != 2
        or any(
            type(row) is not list
            or len(row) != 2
            or type(row[0]) is not int
            or row[0] != ordinal
            for ordinal, row in enumerate(samples, 1)
        )
    ):
        _fail("Docker initial name absence samples differ")
    expected = _build_docker_initial_name_absence_row_from_spec_v1(
        spec, samples[0][1], samples[1][1]
    )
    if canonical_json_bytes_v1(value) != canonical_json_bytes_v1(expected):
        _fail("Docker initial name absence replay differs")
    return value


def _validate_docker_execution_authority_surface_v1(
    value: object,
) -> dict[str, object]:
    keys = {
        "schema_version",
        "source_commit",
        "namespace_domain_hex",
        "attempt_nonce_sha256",
        "execution_namespace",
        "container_name_usage",
        "destructive_target",
        "reserved_label_keys",
        "ordered_slot_rows",
        "initial_name_absence_rows",
        "manifest_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("Docker execution authority fields differ")
    commit = _commit_v1(value["source_commit"], "Docker authority source")
    namespace = _root_hex_v1(
        value["execution_namespace"], "Docker authority namespace"
    )
    _root_hex_v1(
        value["attempt_nonce_sha256"], "Docker authority nonce commitment"
    )
    if (
        value["schema_version"]
        != DOCKER_EXECUTION_AUTHORITY_SCHEMA_VERSION
        or type(value["namespace_domain_hex"]) is not str
        or value["namespace_domain_hex"]
        != DOCKER_OWNERSHIP_NAMESPACE_DOMAIN.hex()
        or value["container_name_usage"] != DOCKER_CONTAINER_NAME_USAGE
        or type(value["container_name_usage"]) is not str
        or value["destructive_target"] != DOCKER_DESTRUCTIVE_TARGET
        or type(value["destructive_target"]) is not str
        or type(value["reserved_label_keys"]) is not list
        or value["reserved_label_keys"] != list(DOCKER_RESERVED_LABEL_KEYS)
        or any(type(item) is not str for item in value["reserved_label_keys"])
    ):
        _fail("Docker execution authority policy differs")
    expected_slots = _docker_slot_rows_from_namespace_v1(commit, namespace)
    _require_type_exact_v1(
        value["ordered_slot_rows"],
        expected_slots,
        "Docker execution authority slots",
    )
    absence_rows = value["initial_name_absence_rows"]
    if type(absence_rows) is not list or len(absence_rows) != 5:
        _fail("Docker initial name absence registry differs")
    for observed, spec in zip(absence_rows, expected_slots, strict=True):
        _validate_docker_initial_name_absence_row_v1(observed, spec)
    body = dict(value)
    manifest = body.pop("manifest_sha256")
    if _root_hex_v1(manifest, "Docker authority manifest") != sha256(
        DOCKER_EXECUTION_AUTHORITY_ROOT_DOMAIN
        + canonical_json_bytes_v1(body)
    ).hexdigest():
        _fail("Docker execution authority manifest differs")
    return value


def build_docker_execution_authority_v1(
    source_commit: str,
    attempt_nonce: bytes,
    initial_name_absence_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    commit = _commit_v1(source_commit, "Docker authority source")
    namespace = _docker_ownership_namespace_v1(commit, attempt_nonce)
    slots = _docker_slot_rows_from_namespace_v1(commit, namespace)
    if (
        type(initial_name_absence_rows) not in (list, tuple)
        or len(initial_name_absence_rows) != 5
    ):
        _fail("Docker initial name absence registry differs")
    absences = [
        dict(_validate_docker_initial_name_absence_row_v1(row, spec))
        for row, spec in zip(initial_name_absence_rows, slots, strict=True)
    ]
    body: dict[str, object] = {
        "schema_version": DOCKER_EXECUTION_AUTHORITY_SCHEMA_VERSION,
        "source_commit": commit,
        "namespace_domain_hex": DOCKER_OWNERSHIP_NAMESPACE_DOMAIN.hex(),
        "attempt_nonce_sha256": sha256(attempt_nonce).hexdigest(),
        "execution_namespace": namespace,
        "container_name_usage": DOCKER_CONTAINER_NAME_USAGE,
        "destructive_target": DOCKER_DESTRUCTIVE_TARGET,
        "reserved_label_keys": list(DOCKER_RESERVED_LABEL_KEYS),
        "ordered_slot_rows": slots,
        "initial_name_absence_rows": absences,
    }
    value = dict(body)
    value["manifest_sha256"] = sha256(
        DOCKER_EXECUTION_AUTHORITY_ROOT_DOMAIN
        + canonical_json_bytes_v1(body)
    ).hexdigest()
    return _validate_docker_execution_authority_surface_v1(value)


def validate_docker_execution_authority_v1(
    value: object,
    source_commit: str,
    attempt_nonce: bytes,
) -> dict[str, object]:
    authority = _validate_docker_execution_authority_surface_v1(value)
    commit = _commit_v1(source_commit, "Docker authority source")
    expected_namespace = _docker_ownership_namespace_v1(commit, attempt_nonce)
    if (
        authority["source_commit"] != commit
        or authority["execution_namespace"] != expected_namespace
        or authority["attempt_nonce_sha256"]
        != sha256(attempt_nonce).hexdigest()
    ):
        _fail("Docker execution authority causal identity differs")
    return authority


def cross_docker_execution_authority_to_admission_decision_v1(
    value: object,
    decision: object,
) -> dict[str, object]:
    """Bind Stage-1 Docker authority to the nonce later spent by admission."""

    if type(decision) is not dict:
        _fail("Docker authority admission decision differs")
    source_commit = _commit_v1(
        decision.get("source_commit"), "Docker authority admission source"
    )
    nonce = _hex_bytes_v1(
        decision.get("attempt_nonce_hex"), "Docker authority admission nonce"
    )
    if (
        len(nonce) != 32
        or decision.get("schema_version") != ACTUAL_ADMISSION_SCHEMA_VERSION
        or decision.get("decision") != ACTUAL_ADMISSION_DECISION_ID
    ):
        _fail("Docker authority admission identity differs")
    _root_hex_v1(
        decision.get("attempt_id"), "Docker authority admission attempt"
    )
    _root_hex_v1(
        decision.get("decision_root"), "Docker authority admission decision"
    )
    return validate_docker_execution_authority_v1(
        value, source_commit, nonce
    )


def build_docker_precreate_absence_v1(
    authority: Mapping[str, object],
    slot_id: int,
    first_absence: Mapping[str, object],
    second_absence: Mapping[str, object],
) -> dict[str, object]:
    validated = _validate_docker_execution_authority_surface_v1(authority)
    spec = _docker_slot_spec_v1(validated["ordered_slot_rows"], slot_id)
    container_name = spec["container_name"]
    assert type(container_name) is str
    samples = _docker_absence_sample_rows_v1(
        first_absence, second_absence, container_name
    )
    body: dict[str, object] = {
        "schema_version": DOCKER_PRECREATE_ABSENCE_SCHEMA_VERSION,
        "docker_execution_authority_manifest_sha256": validated[
            "manifest_sha256"
        ],
        "slot_id": spec["slot_id"],
        "slot": spec["slot"],
        "container_name": container_name,
        "inspect_target": container_name,
        "inspect_command": [
            "/usr/bin/docker",
            "--host=unix:///var/run/docker.sock",
            "inspect",
            container_name,
        ],
        "authoritative_absence_samples": samples,
    }
    value = dict(body)
    value["precreate_absence_root"] = sha256(
        DOCKER_PRECREATE_ABSENCE_ROOT_DOMAIN + canonical_json_bytes_v1(body)
    ).hexdigest()
    return value


def validate_docker_precreate_absence_v1(
    value: object,
    authority: Mapping[str, object],
) -> dict[str, object]:
    keys = {
        "schema_version",
        "docker_execution_authority_manifest_sha256",
        "slot_id",
        "slot",
        "container_name",
        "inspect_target",
        "inspect_command",
        "authoritative_absence_samples",
        "precreate_absence_root",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("Docker precreate absence fields differ")
    samples = value["authoritative_absence_samples"]
    if (
        type(value["slot_id"]) is not int
        or type(samples) is not list
        or len(samples) != 2
        or any(
            type(row) is not list
            or len(row) != 2
            or type(row[0]) is not int
            or row[0] != ordinal
            for ordinal, row in enumerate(samples, 1)
        )
    ):
        _fail("Docker precreate absence samples differ")
    expected = build_docker_precreate_absence_v1(
        authority,
        value["slot_id"],
        samples[0][1],
        samples[1][1],
    )
    if canonical_json_bytes_v1(value) != canonical_json_bytes_v1(expected):
        _fail("Docker precreate absence replay differs")
    return value


def actual_admission_marker_names_v1(
    attempt_id: str,
) -> tuple[str, str, str, str]:
    """Return the only four run-local marker names for one attempt."""

    attempt = _root_hex_v1(attempt_id, "admission marker attempt id")
    return (
        f".q05b-admission-{attempt}.issued",
        f".q05b-admission-{attempt}.spending",
        f".q05b-admission-{attempt}.consumed",
        f".q05b-admission-{attempt}.failed",
    )


def _marker_nonnegative_int_v1(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"{name} integer differs")
    return value


def build_actual_admission_issued_marker_evidence_v1(
    attempt_id: str,
    boundary_root: str,
    boundary_payload: bytes,
    *,
    file_device: int,
    file_inode: int,
    file_nlink: int,
    file_mode: int,
    work_root_device: int,
    work_root_inode: int,
    work_root_mode: int,
) -> dict[str, object]:
    """Build the canonical immutable issued-marker identity object."""

    attempt = _root_hex_v1(attempt_id, "issued marker attempt")
    boundary = _root_hex_v1(boundary_root, "issued marker boundary")
    if type(boundary_payload) is not bytes or not boundary_payload:
        _fail("issued marker boundary payload differs")
    issued, spending, consumed, failed = actual_admission_marker_names_v1(attempt)
    body: dict[str, object] = {
        "schema_version": ACTUAL_ADMISSION_ISSUED_MARKER_SCHEMA_VERSION,
        "attempt_id": attempt,
        "boundary_root": boundary,
        "issued_relative_path": issued,
        "spending_relative_path": spending,
        "consumed_relative_path": consumed,
        "failed_relative_path": failed,
        "payload_length": len(boundary_payload),
        "payload_sha256": sha256(boundary_payload).hexdigest(),
        "file_device": _marker_nonnegative_int_v1(
            file_device, "issued marker file device"
        ),
        "file_inode": _marker_nonnegative_int_v1(
            file_inode, "issued marker file inode"
        ),
        "file_nlink": _marker_nonnegative_int_v1(
            file_nlink, "issued marker file nlink"
        ),
        "file_mode": _marker_nonnegative_int_v1(
            file_mode, "issued marker file mode"
        ),
        "work_root_device": _marker_nonnegative_int_v1(
            work_root_device, "issued marker work-root device"
        ),
        "work_root_inode": _marker_nonnegative_int_v1(
            work_root_inode, "issued marker work-root inode"
        ),
        "work_root_mode": _marker_nonnegative_int_v1(
            work_root_mode, "issued marker work-root mode"
        ),
        "issue_method": (
            "DIRFD_O_NOFOLLOW_O_CREAT_O_EXCL_FSYNC_CHMOD0444_FSYNC"
        ),
    }
    value = dict(body)
    value["issued_marker_root"] = sha256(
        ACTUAL_ADMISSION_ISSUED_MARKER_ROOT_DOMAIN
        + canonical_json_bytes_v1(body)
    ).hexdigest()
    return validate_actual_admission_issued_marker_evidence_v1(
        value, boundary_payload
    )


def validate_actual_admission_issued_marker_evidence_v1(
    value: object,
    boundary_payload: bytes,
) -> dict[str, object]:
    keys = {
        "schema_version", "attempt_id", "boundary_root",
        "issued_relative_path", "spending_relative_path",
        "consumed_relative_path", "failed_relative_path", "payload_length",
        "payload_sha256", "file_device", "file_inode", "file_nlink",
        "file_mode", "work_root_device", "work_root_inode",
        "work_root_mode", "issue_method", "issued_marker_root",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("issued marker evidence fields differ")
    if type(boundary_payload) is not bytes or not boundary_payload:
        _fail("issued marker boundary payload differs")
    attempt = _root_hex_v1(value["attempt_id"], "issued marker attempt")
    _root_hex_v1(value["boundary_root"], "issued marker boundary")
    issued, spending, consumed, failed = actual_admission_marker_names_v1(attempt)
    for name in (
        "payload_length", "file_device", "file_inode", "file_nlink",
        "file_mode", "work_root_device", "work_root_inode", "work_root_mode",
    ):
        _marker_nonnegative_int_v1(value[name], f"issued marker {name}")
    if (
        value["schema_version"] != ACTUAL_ADMISSION_ISSUED_MARKER_SCHEMA_VERSION
        or value["issued_relative_path"] != issued
        or value["spending_relative_path"] != spending
        or value["consumed_relative_path"] != consumed
        or value["failed_relative_path"] != failed
        or value["payload_length"] != len(boundary_payload)
        or value["payload_sha256"] != sha256(boundary_payload).hexdigest()
        or value["file_inode"] <= 0
        or value["file_nlink"] != 1
        or value["file_mode"] != 0o444
        or value["work_root_inode"] <= 0
        or value["work_root_mode"] != 0o700
        or value["issue_method"]
        != "DIRFD_O_NOFOLLOW_O_CREAT_O_EXCL_FSYNC_CHMOD0444_FSYNC"
    ):
        _fail("issued marker evidence differs")
    body = dict(value)
    root = body.pop("issued_marker_root")
    if _root_hex_v1(root, "issued marker") != sha256(
        ACTUAL_ADMISSION_ISSUED_MARKER_ROOT_DOMAIN
        + canonical_json_bytes_v1(body)
    ).hexdigest():
        _fail("issued marker evidence root differs")
    return value


def build_actual_admission_issue_record_v1(
    boundary: Mapping[str, object],
    issued_marker_evidence: Mapping[str, object],
) -> dict[str, object]:
    if type(boundary) is not dict:
        _fail("issued pure admission boundary differs")
    payload = canonical_json_bytes_v1(boundary)
    attempt = _root_hex_v1(
        boundary.get("attempt_id"), "issued boundary attempt"
    )
    boundary_root = _root_hex_v1(
        boundary.get("boundary_root"), "issued boundary"
    )
    marker = validate_actual_admission_issued_marker_evidence_v1(
        issued_marker_evidence, payload
    )
    if (
        marker["attempt_id"] != attempt
        or marker["boundary_root"] != boundary_root
    ):
        _fail("issued marker boundary identity differs")
    body: dict[str, object] = {
        "schema_version": ACTUAL_ADMISSION_ISSUE_RECORD_SCHEMA_VERSION,
        "attempt_id": attempt,
        "boundary_root": boundary_root,
        "pure_boundary_hex": payload.hex(),
        "anti_replay_scope": dict(ACTUAL_ADMISSION_RUN_LOCAL_ANTI_REPLAY_SCOPE),
        "issued_marker_evidence": marker,
    }
    value = dict(body)
    value["issue_record_root"] = sha256(
        ACTUAL_ADMISSION_ISSUE_RECORD_ROOT_DOMAIN
        + canonical_json_bytes_v1(body)
    ).hexdigest()
    validated, _ = validate_actual_admission_issue_record_v1(value)
    return validated


def validate_actual_admission_issue_record_v1(
    value: object,
) -> tuple[dict[str, object], dict[str, object]]:
    keys = {
        "schema_version", "attempt_id", "boundary_root", "pure_boundary_hex",
        "anti_replay_scope", "issued_marker_evidence", "issue_record_root",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("admission issue record fields differ")
    if value["schema_version"] != ACTUAL_ADMISSION_ISSUE_RECORD_SCHEMA_VERSION:
        _fail("admission issue record identity differs")
    attempt = _root_hex_v1(value["attempt_id"], "admission issue attempt")
    boundary_root = _root_hex_v1(
        value["boundary_root"], "admission issue boundary"
    )
    payload = _hex_bytes_v1(value["pure_boundary_hex"], "issued boundary")
    _require_type_exact_v1(
        value["anti_replay_scope"],
        ACTUAL_ADMISSION_RUN_LOCAL_ANTI_REPLAY_SCOPE,
        "admission anti-replay scope",
    )
    boundary = _strict_json_object_v1(payload, "issued pure admission boundary")
    if (
        boundary.get("attempt_id") != attempt
        or boundary.get("boundary_root") != boundary_root
    ):
        _fail("issued boundary identity differs")
    marker = validate_actual_admission_issued_marker_evidence_v1(
        value["issued_marker_evidence"], payload
    )
    if (
        marker["attempt_id"] != attempt
        or marker["boundary_root"] != boundary_root
    ):
        _fail("issued marker evidence identity differs")
    body = dict(value)
    root = body.pop("issue_record_root")
    if _root_hex_v1(root, "admission issue record") != sha256(
        ACTUAL_ADMISSION_ISSUE_RECORD_ROOT_DOMAIN
        + canonical_json_bytes_v1(body)
    ).hexdigest():
        _fail("admission issue record root differs")
    return value, boundary


def build_actual_admission_spending_intent_v1(
    issue_record: Mapping[str, object],
) -> dict[str, object]:
    record, boundary = validate_actual_admission_issue_record_v1(issue_record)
    marker = record["issued_marker_evidence"]
    payload = _hex_bytes_v1(record["pure_boundary_hex"], "issued boundary")
    body: dict[str, object] = {
        "schema_version": ACTUAL_ADMISSION_SPENDING_INTENT_SCHEMA_VERSION,
        "attempt_id": boundary["attempt_id"],
        "boundary_root": boundary["boundary_root"],
        "issue_record_root": record["issue_record_root"],
        "issued_relative_path": marker["issued_relative_path"],
        "spending_relative_path": marker["spending_relative_path"],
        "consumed_relative_path": marker["consumed_relative_path"],
        "failed_relative_path": marker["failed_relative_path"],
        "boundary_payload_length": len(payload),
        "boundary_payload_sha256": sha256(payload).hexdigest(),
        "transition": "ISSUED_TO_SPENDING_BEFORE_ANY_ISSUED_REPLAY",
    }
    value = dict(body)
    value["spending_intent_root"] = sha256(
        ACTUAL_ADMISSION_SPENDING_INTENT_ROOT_DOMAIN
        + canonical_json_bytes_v1(body)
    ).hexdigest()
    return value


def validate_actual_admission_spending_intent_v1(
    value: object,
    issue_record: Mapping[str, object],
) -> dict[str, object]:
    if type(value) is not dict:
        _fail("admission spending intent differs")
    expected = build_actual_admission_spending_intent_v1(issue_record)
    if canonical_json_bytes_v1(value) != canonical_json_bytes_v1(expected):
        _fail("admission spending intent replay differs")
    return value


def build_actual_admission_consumed_marker_evidence_v1(
    issue_record: Mapping[str, object],
    spending_intent: Mapping[str, object],
    *,
    spending_file_device: int,
    spending_file_inode: int,
    spending_file_nlink: int,
    spending_file_mode: int,
    file_device: int,
    file_inode: int,
    file_nlink: int,
    file_mode: int,
    work_root_device: int,
    work_root_inode: int,
    work_root_mode: int,
) -> dict[str, object]:
    record, boundary = validate_actual_admission_issue_record_v1(issue_record)
    spending = validate_actual_admission_spending_intent_v1(
        spending_intent, record
    )
    marker = record["issued_marker_evidence"]
    payload = _hex_bytes_v1(record["pure_boundary_hex"], "issued boundary")
    spending_payload = canonical_json_bytes_v1(spending)
    body: dict[str, object] = {
        "schema_version": ACTUAL_ADMISSION_CONSUMED_MARKER_SCHEMA_VERSION,
        "attempt_id": boundary["attempt_id"],
        "boundary_root": boundary["boundary_root"],
        "issue_record_root": record["issue_record_root"],
        "issued_relative_path": marker["issued_relative_path"],
        "spending_relative_path": marker["spending_relative_path"],
        "consumed_relative_path": marker["consumed_relative_path"],
        "failed_relative_path": marker["failed_relative_path"],
        "spending_intent_hex": spending_payload.hex(),
        "spending_intent_root": spending["spending_intent_root"],
        "spending_file_device": _marker_nonnegative_int_v1(
            spending_file_device, "spending file device"
        ),
        "spending_file_inode": _marker_nonnegative_int_v1(
            spending_file_inode, "spending file inode"
        ),
        "spending_file_nlink": _marker_nonnegative_int_v1(
            spending_file_nlink, "spending file nlink"
        ),
        "spending_file_mode": _marker_nonnegative_int_v1(
            spending_file_mode, "spending file mode"
        ),
        "payload_length": len(payload),
        "payload_sha256": sha256(payload).hexdigest(),
        "file_device": _marker_nonnegative_int_v1(
            file_device, "consumed file device"
        ),
        "file_inode": _marker_nonnegative_int_v1(
            file_inode, "consumed file inode"
        ),
        "file_nlink": _marker_nonnegative_int_v1(
            file_nlink, "consumed file nlink"
        ),
        "file_mode": _marker_nonnegative_int_v1(
            file_mode, "consumed file mode"
        ),
        "work_root_device": _marker_nonnegative_int_v1(
            work_root_device, "consumed work-root device"
        ),
        "work_root_inode": _marker_nonnegative_int_v1(
            work_root_inode, "consumed work-root inode"
        ),
        "work_root_mode": _marker_nonnegative_int_v1(
            work_root_mode, "consumed work-root mode"
        ),
        "consume_method": (
            "O_EXCL_SPENDING_FSYNC_THEN_SAME_DIRFD_HARDLINK_"
            "NOREPLACE_FSYNC_KEEP_ISSUED"
        ),
        "spent_before_preflight": True,
    }
    value = dict(body)
    value["consumed_marker_root"] = sha256(
        ACTUAL_ADMISSION_CONSUMED_MARKER_ROOT_DOMAIN
        + canonical_json_bytes_v1(body)
    ).hexdigest()
    return validate_actual_admission_consumed_marker_evidence_v1(value, record)


def validate_actual_admission_consumed_marker_evidence_v1(
    value: object,
    issue_record: Mapping[str, object],
) -> dict[str, object]:
    record, boundary = validate_actual_admission_issue_record_v1(issue_record)
    marker = record["issued_marker_evidence"]
    keys = {
        "schema_version", "attempt_id", "boundary_root", "issue_record_root",
        "issued_relative_path", "spending_relative_path",
        "consumed_relative_path", "failed_relative_path",
        "spending_intent_hex", "spending_intent_root",
        "spending_file_device", "spending_file_inode", "spending_file_nlink",
        "spending_file_mode", "payload_length", "payload_sha256",
        "file_device", "file_inode", "file_nlink", "file_mode",
        "work_root_device", "work_root_inode", "work_root_mode",
        "consume_method", "spent_before_preflight", "consumed_marker_root",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("consumed marker evidence fields differ")
    for name in (
        "payload_length", "file_device", "file_inode", "file_nlink",
        "file_mode", "work_root_device", "work_root_inode", "work_root_mode",
        "spending_file_device", "spending_file_inode", "spending_file_nlink",
        "spending_file_mode",
    ):
        _marker_nonnegative_int_v1(value[name], f"consumed marker {name}")
    spending_payload = _strict_json_object_v1(
        _hex_bytes_v1(value["spending_intent_hex"], "admission spending intent"),
        "admission spending intent",
    )
    spending = validate_actual_admission_spending_intent_v1(
        spending_payload, record
    )
    if (
        value["schema_version"]
        != ACTUAL_ADMISSION_CONSUMED_MARKER_SCHEMA_VERSION
        or value["attempt_id"] != boundary["attempt_id"]
        or value["boundary_root"] != boundary["boundary_root"]
        or value["issue_record_root"] != record["issue_record_root"]
        or value["issued_relative_path"] != marker["issued_relative_path"]
        or value["spending_relative_path"] != marker["spending_relative_path"]
        or value["consumed_relative_path"] != marker["consumed_relative_path"]
        or value["failed_relative_path"] != marker["failed_relative_path"]
        or value["spending_intent_root"] != spending["spending_intent_root"]
        or value["spending_file_inode"] <= 0
        or value["spending_file_nlink"] != 1
        or value["spending_file_mode"] != 0o444
        or value["payload_length"] != marker["payload_length"]
        or value["payload_sha256"] != marker["payload_sha256"]
        or value["file_device"] != marker["file_device"]
        or value["file_inode"] != marker["file_inode"]
        or value["file_nlink"] != 2
        or value["file_mode"] != 0o444
        or value["work_root_device"] != marker["work_root_device"]
        or value["work_root_inode"] != marker["work_root_inode"]
        or value["work_root_mode"] != 0o700
        or value["consume_method"]
        != (
            "O_EXCL_SPENDING_FSYNC_THEN_SAME_DIRFD_HARDLINK_"
            "NOREPLACE_FSYNC_KEEP_ISSUED"
        )
        or value["spent_before_preflight"] is not True
    ):
        _fail("consumed marker evidence differs")
    body = dict(value)
    root = body.pop("consumed_marker_root")
    if _root_hex_v1(root, "consumed marker") != sha256(
        ACTUAL_ADMISSION_CONSUMED_MARKER_ROOT_DOMAIN
        + canonical_json_bytes_v1(body)
    ).hexdigest():
        _fail("consumed marker evidence root differs")
    return value


def build_actual_admission_live_marker_replay_v1(
    checkpoint: str,
    issue_record: Mapping[str, object],
    consumed_marker_evidence: Mapping[str, object],
    *,
    work_root_device: int,
    work_root_inode: int,
    work_root_nlink: int,
    work_root_mode: int,
    issued_file_device: int,
    issued_file_inode: int,
    issued_file_nlink: int,
    consumed_file_device: int,
    consumed_file_inode: int,
    consumed_file_nlink: int,
    spending_file_device: int,
    spending_file_inode: int,
    spending_file_nlink: int,
) -> dict[str, object]:
    if type(checkpoint) is not str or re.fullmatch(r"[A-Z0-9_]+", checkpoint) is None:
        _fail("admission live checkpoint differs")
    record, boundary = validate_actual_admission_issue_record_v1(issue_record)
    consumed = validate_actual_admission_consumed_marker_evidence_v1(
        consumed_marker_evidence, record
    )
    payload = _hex_bytes_v1(record["pure_boundary_hex"], "issued boundary")
    body: dict[str, object] = {
        "schema_version": ACTUAL_ADMISSION_LIVE_MARKER_REPLAY_SCHEMA_VERSION,
        "checkpoint": checkpoint,
        "attempt_id": boundary["attempt_id"],
        "boundary_root": boundary["boundary_root"],
        "issue_record_root": record["issue_record_root"],
        "consumed_marker_root": consumed["consumed_marker_root"],
        "work_root_device": work_root_device,
        "work_root_inode": work_root_inode,
        "work_root_nlink": work_root_nlink,
        "work_root_mode": work_root_mode,
        "issued_file_device": issued_file_device,
        "issued_file_inode": issued_file_inode,
        "issued_file_nlink": issued_file_nlink,
        "consumed_file_device": consumed_file_device,
        "consumed_file_inode": consumed_file_inode,
        "consumed_file_nlink": consumed_file_nlink,
        "spending_file_device": spending_file_device,
        "spending_file_inode": spending_file_inode,
        "spending_file_nlink": spending_file_nlink,
        "boundary_payload_sha256": sha256(payload).hexdigest(),
        "issued_consumed_same_inode": True,
        "work_root_path_matches_held_descriptor": True,
        "issued_path_matches_held_descriptor": True,
        "spending_path_matches_held_descriptor": True,
        "consumed_path_matches_held_descriptor": True,
    }
    value = dict(body)
    value["live_marker_replay_root"] = sha256(
        ACTUAL_ADMISSION_LIVE_MARKER_REPLAY_ROOT_DOMAIN
        + canonical_json_bytes_v1(body)
    ).hexdigest()
    return validate_actual_admission_live_marker_replay_surface_v1(
        value, checkpoint, record, consumed
    )


def validate_actual_admission_live_marker_replay_surface_v1(
    value: object,
    expected_checkpoint: str,
    issue_record: Mapping[str, object] | None = None,
    consumed_marker_evidence: Mapping[str, object] | None = None,
) -> dict[str, object]:
    keys = {
        "schema_version", "checkpoint", "attempt_id", "boundary_root",
        "issue_record_root", "consumed_marker_root", "work_root_device",
        "work_root_inode", "work_root_nlink", "work_root_mode",
        "issued_file_device", "issued_file_inode", "issued_file_nlink",
        "consumed_file_device", "consumed_file_inode", "consumed_file_nlink",
        "spending_file_device", "spending_file_inode", "spending_file_nlink",
        "boundary_payload_sha256", "issued_consumed_same_inode",
        "work_root_path_matches_held_descriptor",
        "issued_path_matches_held_descriptor",
        "spending_path_matches_held_descriptor",
        "consumed_path_matches_held_descriptor", "live_marker_replay_root",
    }
    if (
        type(value) is not dict
        or set(value) != keys
        or type(expected_checkpoint) is not str
        or re.fullmatch(r"[A-Z0-9_]+", expected_checkpoint) is None
        or value["schema_version"]
        != ACTUAL_ADMISSION_LIVE_MARKER_REPLAY_SCHEMA_VERSION
        or value["checkpoint"] != expected_checkpoint
        or type(value["checkpoint"]) is not str
    ):
        _fail("live admission replay surface differs")
    for name in (
        "attempt_id", "boundary_root", "issue_record_root",
        "consumed_marker_root", "boundary_payload_sha256",
    ):
        _root_hex_v1(value[name], f"live admission replay {name}")
    for name in (
        "work_root_device", "work_root_inode", "work_root_nlink",
        "work_root_mode", "issued_file_device", "issued_file_inode",
        "issued_file_nlink", "consumed_file_device", "consumed_file_inode",
        "consumed_file_nlink", "spending_file_device", "spending_file_inode",
        "spending_file_nlink",
    ):
        _marker_nonnegative_int_v1(value[name], f"live admission replay {name}")
    if (
        value["work_root_inode"] <= 0
        or value["work_root_nlink"] < 2
        or value["work_root_mode"] != 0o700
        or value["issued_file_inode"] <= 0
        or value["issued_file_nlink"] != 2
        or value["consumed_file_inode"] <= 0
        or value["consumed_file_nlink"] != 2
        or value["spending_file_inode"] <= 0
        or value["spending_file_nlink"] != 1
        or value["issued_consumed_same_inode"] is not True
        or value["work_root_path_matches_held_descriptor"] is not True
        or value["issued_path_matches_held_descriptor"] is not True
        or value["spending_path_matches_held_descriptor"] is not True
        or value["consumed_path_matches_held_descriptor"] is not True
    ):
        _fail("live admission replay identity differs")
    if (issue_record is None) != (consumed_marker_evidence is None):
        _fail("live admission replay cross evidence differs")
    if issue_record is not None:
        record, boundary = validate_actual_admission_issue_record_v1(issue_record)
        consumed = validate_actual_admission_consumed_marker_evidence_v1(
            consumed_marker_evidence, record
        )
        payload = _hex_bytes_v1(record["pure_boundary_hex"], "issued boundary")
        if (
            value["attempt_id"] != boundary["attempt_id"]
            or value["boundary_root"] != boundary["boundary_root"]
            or value["issue_record_root"] != record["issue_record_root"]
            or value["consumed_marker_root"] != consumed["consumed_marker_root"]
            or value["boundary_payload_sha256"] != sha256(payload).hexdigest()
            or value["work_root_device"] != consumed["work_root_device"]
            or value["work_root_inode"] != consumed["work_root_inode"]
            or value["issued_file_device"] != consumed["file_device"]
            or value["issued_file_inode"] != consumed["file_inode"]
            or value["consumed_file_device"] != consumed["file_device"]
            or value["consumed_file_inode"] != consumed["file_inode"]
            or value["spending_file_device"]
            != consumed["spending_file_device"]
            or value["spending_file_inode"] != consumed["spending_file_inode"]
        ):
            _fail("live admission replay cross evidence differs")
    body = dict(value)
    root = body.pop("live_marker_replay_root")
    if _root_hex_v1(root, "live admission replay") != sha256(
        ACTUAL_ADMISSION_LIVE_MARKER_REPLAY_ROOT_DOMAIN
        + canonical_json_bytes_v1(body)
    ).hexdigest():
        _fail("live admission replay evidence root differs")
    return value


def _canonical_object_sha256_v1(value: object) -> str:
    return sha256(canonical_json_bytes_v1(value)).hexdigest()


def _validate_config_bytes_v1(value: object) -> dict[str, object]:
    if type(value) is not bytes or not value or len(value) > 4 * 1024 * 1024:
        _fail("Commit-A config bytes differ")
    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in rows:
            if type(key) is not str or key in result:
                _fail("Commit-A config duplicate key")
            result[key] = item
        return result

    try:
        decoded = json.loads(
            value.decode("ascii", "strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: _fail(
                f"Commit-A config non-finite {token}"
            ),
            parse_float=lambda token: _fail(f"Commit-A config float {token}"),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail(f"Commit-A config is not strict JSON: {error}")
    if type(decoded) is not dict:
        _fail("Commit-A config is not one object")
    expected_top = {
        "schema_version", "profile_id", "claim_scope", "engineering_status",
        "images", "docker", "seccomp", "resource_roles", "mount_policy",
        "stdout_capture_policy", "held_actor_protocol",
        "live_resource_evidence_policy", "runtime_command_inspect_policy",
        "source_snapshot_policy", "source_allowlist_policy",
        "rust_build_policy", "actor_commands", "execution_protocol",
        "qualification_receipt_protocol", "artifact_layout",
        "dry_run_authority", "actual_preconditions",
    }
    if (
        set(decoded) != expected_top
        or decoded["schema_version"]
        != "hegel-phase3a-q05b-dual-isolation/1"
        or decoded["profile_id"]
        != "hegel-phase3a-q05b-three-actor-offline-qualification-v1"
        or decoded["claim_scope"] != "Q05B_TARGET_BLIND_QUALIFICATION_ONLY"
        or decoded["engineering_status"] != COMMIT_A_ACTUAL_ENGINEERING_STATUS
    ):
        _fail("Commit-A config admission identity differs")
    _require_type_exact_v1(
        decoded["actual_preconditions"],
        COMMIT_A_ACTUAL_PRECONDITIONS_V1,
        "Commit-A actual preconditions",
    )
    expected_dry = {
        "qualification_predicate_count": 0,
        "qualification_predicate_mask": 0,
        "qualification_predicate_total": 20,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_gate_total": 20,
        "q1_formal_output_roots": [None] * 8,
        "q1_receipt": None,
        "q2_state": "NOT_RUN",
        "m3_formal_roots": None,
        "formal_fixed_point_claimed": False,
        "outside_certificate_issued": False,
        "active_transition_allowed": False,
        "artifact_written": False,
    }
    _require_type_exact_v1(
        decoded["dry_run_authority"], expected_dry, "Commit-A dry authority"
    )
    _require_type_exact_v1(
        decoded["images"],
        {
            "python_endpoint": EXPECTED_PYTHON_IMAGE_REFERENCE,
            "rust_build": EXPECTED_RUST_IMAGE_REFERENCE,
            "rust_runtime": EXPECTED_RUST_IMAGE_REFERENCE,
            "trusted_host": EXPECTED_PYTHON_IMAGE_REFERENCE,
        },
        "Commit-A pinned images",
    )
    static_policy = {
        key: decoded[key]
        for key in decoded
        if key not in {"engineering_status", "actual_preconditions"}
    }
    static_root = sha256(
        ACTUAL_COMMIT_A_STATIC_POLICY_ROOT_DOMAIN
        + canonical_json_bytes_v1(static_policy)
    ).hexdigest()
    if static_root != EXPECTED_COMMIT_A_STATIC_POLICY_ROOT:
        _fail("Commit-A static policy root differs")
    if (
        command_mount_resource_policy_root_v1(decoded)
        != EXPECTED_COMMAND_MOUNT_RESOURCE_POLICY_ROOT
    ):
        _fail("Commit-A command/mount/resource policy root differs")
    return decoded


def validate_commit_a_actual_config_bytes_v1(
    value: object,
) -> dict[str, object]:
    """Public strict replay of the future Commit-A actual config bytes."""

    return _validate_config_bytes_v1(value)


def command_mount_resource_policy_root_v1(
    config_or_bytes: object,
) -> str:
    if type(config_or_bytes) is bytes:
        config = _validate_config_bytes_v1(config_or_bytes)
    elif type(config_or_bytes) is dict:
        config = config_or_bytes
    else:
        _fail("command/mount/resource policy input differs")
    if any(field not in config for field in COMMAND_MOUNT_RESOURCE_POLICY_FIELDS):
        _fail("command/mount/resource policy fields differ")
    subset = {field: config[field] for field in COMMAND_MOUNT_RESOURCE_POLICY_FIELDS}
    return sha256(
        ACTUAL_COMMAND_MOUNT_RESOURCE_POLICY_ROOT_DOMAIN
        + canonical_json_bytes_v1(subset)
    ).hexdigest()


def actual_precondition_registry_root_v1() -> str:
    rows = [
        [predicate_id, predicate_name.decode("ascii")]
        for predicate_id, predicate_name in ACTUAL_RUNTIME_PRECONDITION_REGISTRY
    ]
    return sha256(
        ACTUAL_PRECONDITION_REGISTRY_ROOT_DOMAIN + canonical_json_bytes_v1(rows)
    ).hexdigest()


def validate_work_root_identity_v1(value: object) -> dict[str, object]:
    keys = {
        "schema_version", "absolute_path", "device", "inode", "nlink",
        "mode", "layout_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("admission work-root identity differs")
    if (
        value["schema_version"]
        != "hegel-phase3a-q05b-admission-work-root-identity/1"
        or _absolute_path_text_v1(value["absolute_path"], "work root")
        != value["absolute_path"]
        or type(value["device"]) is not int
        or value["device"] < 0
        or type(value["inode"]) is not int
        or value["inode"] <= 0
        or type(value["nlink"]) is not int
        or value["nlink"] < 2
        or type(value["mode"]) is not int
        or value["mode"] != 0o700
    ):
        _fail("admission work-root identity fields differ")
    _root_hex_v1(value["layout_sha256"], "work-root layout")
    return value


def validate_artifact_absence_evidence_v1(
    value: object, artifact_path: str
) -> dict[str, object]:
    keys = {
        "schema_version", "artifact_path", "parent_path", "parent_device",
        "parent_inode", "parent_nlink", "parent_mode", "target_absent",
        "nofollow_dirfd_checked",
    }
    path = _absolute_path_text_v1(artifact_path, "artifact")
    parent = Path(path).parent.as_posix()
    if type(value) is not dict or set(value) != keys:
        _fail("artifact absence evidence fields differ")
    if (
        value["schema_version"]
        != "hegel-phase3a-q05b-admission-artifact-absence/1"
        or value["artifact_path"] != path
        or value["parent_path"] != parent
        or any(
            type(value[name]) is not int or value[name] < 0
            for name in (
                "parent_device", "parent_inode", "parent_nlink", "parent_mode"
            )
        )
        or value["target_absent"] is not True
        or value["nofollow_dirfd_checked"] is not True
    ):
        _fail("artifact absence evidence differs")
    return value


def validate_prior_stage_root_rows_v1(value: object) -> list[list[object]]:
    if type(value) is not list or len(value) != 3:
        _fail("admission prior-stage root rows differ")
    result: list[list[object]] = []
    for expected_id, row in enumerate(value, start=1):
        if (
            type(row) is not list
            or len(row) != 2
            or type(row[0]) is not int
            or row[0] != expected_id
        ):
            _fail("admission prior-stage root row differs")
        result.append([expected_id, _root_hex_v1(row[1], "prior stage")])
    return result


def validate_git_source_transcript_v1(
    value: object,
    source_commit: str,
) -> dict[str, object]:
    keys = {
        "schema_version",
        "project_root",
        "requested_source_commit",
        "command_rows",
        "transcript_root",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("Git source transcript fields differ")
    project_root = _absolute_path_text_v1(value["project_root"], "Git project root")
    commit = _commit_v1(source_commit, "Git transcript source")
    if (
        value["schema_version"] != ACTUAL_GIT_SOURCE_TRANSCRIPT_SCHEMA_VERSION
        or value["requested_source_commit"] != commit
        or type(value["command_rows"]) is not list
        or len(value["command_rows"]) != 2
    ):
        _fail("Git source transcript identity differs")
    expected = (
        (
            1,
            "VERIFY_HEAD",
            ["git", "-C", project_root, "rev-parse", "--verify", "HEAD"],
            (commit + "\n").encode("ascii"),
        ),
        (
            2,
            "VERIFY_CLEAN_STATUS_Z",
            [
                "git", "-C", project_root, "status", "--porcelain=v1",
                "--untracked-files=all", "-z",
            ],
            b"",
        ),
    )
    row_keys = {
        "ordinal", "purpose", "argv", "returncode", "stdout_hex",
        "stderr_hex", "stdout_sha256", "stderr_sha256",
    }
    for row, (ordinal, purpose, argv, stdout) in zip(
        value["command_rows"], expected, strict=True
    ):
        if (
            type(row) is not dict
            or set(row) != row_keys
            or type(row["ordinal"]) is not int
            or row["ordinal"] != ordinal
            or type(row["purpose"]) is not str
            or row["purpose"] != purpose
            or type(row["argv"]) is not list
            or row["argv"] != argv
            or any(type(item) is not str for item in row["argv"])
            or type(row["returncode"]) is not int
            or row["returncode"] != 0
        ):
            _fail("Git source transcript command row differs")
        observed_stdout = _hex_bytes_v1(
            row["stdout_hex"], "Git stdout", allow_empty=True
        )
        observed_stderr = _hex_bytes_v1(
            row["stderr_hex"], "Git stderr", allow_empty=True
        )
        if (
            observed_stdout != stdout
            or observed_stderr != b""
            or row["stdout_sha256"] != sha256(observed_stdout).hexdigest()
            or row["stderr_sha256"] != sha256(observed_stderr).hexdigest()
        ):
            _fail("Git source transcript bytes differ")
    body = dict(value)
    root = body.pop("transcript_root")
    if (
        _root_hex_v1(root, "Git source transcript")
        != sha256(
            ACTUAL_GIT_SOURCE_TRANSCRIPT_ROOT_DOMAIN
            + canonical_json_bytes_v1(body)
        ).hexdigest()
    ):
        _fail("Git source transcript root differs")
    return value


def fresh_runtime_evidence_object_root_v1(
    evidence_kind: str,
    evidence_label: str,
    evidence: Mapping[str, object],
) -> str:
    """Bind one complete, ordered fresh-runtime evidence object."""

    if (
        type(evidence_kind) is not str
        or re.fullmatch(r"[A-Z0-9_]+", evidence_kind) is None
        or type(evidence_label) is not str
        or re.fullmatch(r"[A-Za-z0-9_./-]+", evidence_label) is None
        or type(evidence) is not dict
    ):
        _fail("fresh runtime evidence object identity differs")
    return sha256(
        ACTUAL_FRESH_RUNTIME_EVIDENCE_OBJECT_ROOT_DOMAIN
        + canonical_json_bytes_v1([evidence_kind, evidence_label, evidence])
    ).hexdigest()


def _strict_json_value_bytes_v1(payload: bytes, name: str) -> object:
    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in rows:
            if type(key) is not str or key in result:
                _fail(f"{name} duplicate key")
            result[key] = item
        return result

    try:
        return json.loads(
            payload.decode("ascii", "strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: _fail(f"{name} non-finite {token}"),
            parse_float=lambda token: _fail(f"{name} float {token}"),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail(f"{name} strict JSON differs: {error}")


def _int_field_v1(
    value: Mapping[str, object],
    field: str,
    name: str,
    *,
    minimum: int = 0,
) -> int:
    item = value[field]
    if type(item) is not int or item < minimum:
        _fail(f"{name} {field} differs")
    return item


def _validate_pinned_image_evidence_v1(
    value: object,
    expected_reference: str,
) -> dict[str, object]:
    keys = {
        "schema_version", "requested_reference", "image_id", "repo_digests",
        "os", "architecture", "raw_inspect_hex", "raw_inspect_sha256",
        "evidence_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("pinned image full evidence fields differ")
    if (
        value["schema_version"]
        != "hegel-phase3a-q05b-pinned-local-image-evidence/1"
        or value["requested_reference"] != expected_reference
        or type(value["image_id"]) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", value["image_id"]) is None
        or type(value["repo_digests"]) is not list
        or expected_reference not in value["repo_digests"]
        or any(type(item) is not str for item in value["repo_digests"])
        or value["os"] != "linux"
        or type(value["architecture"]) is not str
        or not value["architecture"]
    ):
        _fail("pinned image full evidence identity differs")
    raw = _hex_bytes_v1(value["raw_inspect_hex"], "pinned image inspect")
    if value["raw_inspect_sha256"] != sha256(raw).hexdigest():
        _fail("pinned image raw inspect digest differs")
    document = _strict_json_value_bytes_v1(raw, "pinned image inspect")
    if type(document) is not list or len(document) != 1 or type(document[0]) is not dict:
        _fail("pinned image raw inspect shape differs")
    raw_image = document[0]
    if (
        raw_image.get("Id") != value["image_id"]
        or raw_image.get("RepoDigests") != value["repo_digests"]
        or raw_image.get("Os") != value["os"]
        or raw_image.get("Architecture") != value["architecture"]
        or type(raw_image.get("Config")) is not dict
        or type(raw_image["Config"].get("Env")) is not list
        or any(type(item) is not str for item in raw_image["Config"]["Env"])
    ):
        _fail("pinned image raw inspect identity differs")
    body = dict(value)
    root = body.pop("evidence_sha256")
    if _root_hex_v1(root, "pinned image evidence") != _canonical_object_sha256_v1(body):
        _fail("pinned image evidence digest differs")
    return value


def _validate_actor_source_identity_v1(
    value: object,
    expected_actor_id: str,
    source_commit: str,
) -> dict[str, object]:
    keys = {
        "schema_version", "actor_id", "source_commit", "project_git_prefix",
        "path_registry_sha256", "source_identity_sha256", "blob_count",
        "snapshot_file_registry_sha256", "stage_1_source_evidence_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("actor source identity fields differ")
    if (
        value["schema_version"]
        != "hegel-phase3a-q05b-fresh-actor-source-identity/1"
        or value["actor_id"] != expected_actor_id
        or value["source_commit"] != source_commit
        or type(value["project_git_prefix"]) is not str
        or ".." in value["project_git_prefix"].split("/")
        or type(value["blob_count"]) is not int
        or value["blob_count"] < 1
    ):
        _fail("actor source identity differs")
    for field in (
        "path_registry_sha256", "source_identity_sha256",
        "snapshot_file_registry_sha256", "stage_1_source_evidence_sha256",
    ):
        _root_hex_v1(value[field], f"actor source identity {field}")
    return value


def _validate_sealed_tree_evidence_v1(value: object, name: str) -> dict[str, object]:
    keys = {
        "schema_version", "root_path", "root_device", "root_inode",
        "root_nlink", "root_mode", "directory_rows", "file_rows",
        "manifest_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        _fail(f"{name} sealed tree fields differ")
    if (
        value["schema_version"] != "hegel-phase3a-q05b-sealed-tree-identity/1"
        or _absolute_path_text_v1(value["root_path"], name) != value["root_path"]
        or _int_field_v1(value, "root_device", name) < 0
        or _int_field_v1(value, "root_inode", name, minimum=1) < 1
        or _int_field_v1(value, "root_nlink", name, minimum=2) < 2
        or value["root_mode"] != 0o555
        or type(value["directory_rows"]) is not list
        or type(value["file_rows"]) is not list
        or not value["file_rows"]
    ):
        _fail(f"{name} sealed tree identity differs")
    prior = ""
    for row in value["directory_rows"]:
        if (
            type(row) is not list
            or len(row) != 9
            or type(row[0]) is not str
            or row[0] <= prior
            or row[0].startswith("/")
            or ".." in row[0].split("/")
            or any(type(item) is not int for item in row[1:])
            or row[3] < 2
            or row[6] != 0o555
        ):
            _fail(f"{name} sealed tree directory row differs")
        prior = row[0]
    prior = ""
    for row in value["file_rows"]:
        if (
            type(row) is not list
            or len(row) != 11
            or type(row[0]) is not str
            or row[0] <= prior
            or row[0].startswith("/")
            or ".." in row[0].split("/")
            or any(type(item) is not int for item in row[1:10])
            or row[3] != 1
            or row[6] not in (0o444, 0o555)
        ):
            _fail(f"{name} sealed tree file row differs")
        _root_hex_v1(row[10], f"{name} sealed tree file")
        prior = row[0]
    body = dict(value)
    root = body.pop("manifest_sha256")
    if _root_hex_v1(root, name) != _canonical_object_sha256_v1(body):
        _fail(f"{name} sealed tree manifest differs")
    return value


def _validate_sealed_snapshot_evidence_v1(
    value: object, name: str
) -> dict[str, object]:
    keys = {
        "schema_version", "root_device", "root_inode", "root_mode",
        "file_rows", "manifest_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        _fail(f"{name} sealed snapshot fields differ")
    if (
        value["schema_version"]
        != "hegel-phase3a-q05b-sealed-snapshot-identity/1"
        or _int_field_v1(value, "root_device", name) < 0
        or _int_field_v1(value, "root_inode", name, minimum=1) < 1
        or value["root_mode"] != 0o555
        or type(value["file_rows"]) is not list
        or not value["file_rows"]
    ):
        _fail(f"{name} sealed snapshot identity differs")
    prior = ""
    for row in value["file_rows"]:
        if (
            type(row) is not list
            or len(row) != 11
            or type(row[0]) is not str
            or row[0] <= prior
            or any(type(item) is not int for item in row[1:10])
            or row[3] != 1
            or row[6] not in (0o444, 0o555)
        ):
            _fail(f"{name} sealed snapshot row differs")
        _root_hex_v1(row[10], f"{name} sealed snapshot file")
        prior = row[0]
    body = dict(value)
    root = body.pop("manifest_sha256")
    if _root_hex_v1(root, name) != _canonical_object_sha256_v1(body):
        _fail(f"{name} sealed snapshot manifest differs")
    return value


def _validate_seccomp_evidence_v1(
    value: object,
    expected_relative: str,
) -> dict[str, object]:
    keys = {
        "schema_version", "absolute_path", "snapshot_relative_path",
        "file_device", "file_inode", "file_nlink", "file_uid", "file_gid",
        "file_mode", "file_size", "file_mtime_ns", "file_ctime_ns",
        "payload_sha256", "manifest_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("seccomp full evidence fields differ")
    if (
        value["schema_version"] != "hegel-phase3a-q05b-sealed-policy-file/1"
        or _absolute_path_text_v1(value["absolute_path"], "seccomp")
        != value["absolute_path"]
        or value["snapshot_relative_path"] != expected_relative
        or any(
            type(value[field]) is not int or value[field] < 0
            for field in (
                "file_device", "file_inode", "file_nlink", "file_uid",
                "file_gid", "file_size", "file_mtime_ns", "file_ctime_ns",
            )
        )
        or value["file_inode"] <= 0
        or value["file_nlink"] != 1
        or value["file_mode"] != 0o444
        or value["file_size"] <= 0
    ):
        _fail("seccomp full evidence identity differs")
    _root_hex_v1(value["payload_sha256"], "seccomp payload")
    body = dict(value)
    root = body.pop("manifest_sha256")
    if _root_hex_v1(root, "seccomp manifest") != _canonical_object_sha256_v1(body):
        _fail("seccomp full evidence manifest differs")
    return value


def _validate_binary_identity_v1(value: object) -> dict[str, object]:
    keys = {
        "schema_version", "binary_path", "device", "inode", "nlink",
        "uid", "gid", "mode", "size", "mtime_ns", "ctime_ns", "sha256",
        "sealed_binary_manifest_sha256", "stage_3_binary_evidence_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("binary fresh identity fields differ")
    if (
        value["schema_version"]
        != "hegel-phase3a-q05b-fresh-prebuilt-rust-binary-identity/1"
        or _absolute_path_text_v1(value["binary_path"], "binary")
        != value["binary_path"]
        or any(
            type(value[field]) is not int or value[field] < 0
            for field in (
                "device", "inode", "nlink", "uid", "gid", "size",
                "mtime_ns", "ctime_ns",
            )
        )
        or value["inode"] <= 0
        or value["nlink"] != 1
        or value["mode"] != 0o555
        or value["size"] <= 0
    ):
        _fail("binary fresh identity differs")
    for field in (
        "sha256", "sealed_binary_manifest_sha256",
        "stage_3_binary_evidence_sha256",
    ):
        _root_hex_v1(value[field], f"binary identity {field}")
    return value


def _validate_cargo_material_identity_v1(value: object) -> dict[str, object]:
    keys = {
        "schema_version", "root_path", "root_nlink", "file_count",
        "locked_registry_package_count", "locked_packages_sha256",
        "file_registry_sha256", "material_manifest_sha256",
        "sealed_snapshot_manifest_sha256", "sealed_tree_manifest_sha256",
        "stage_2_cargo_evidence_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("Cargo material identity fields differ")
    if (
        value["schema_version"]
        != "hegel-phase3a-q05b-fresh-cargo-material-identity/1"
        or _absolute_path_text_v1(value["root_path"], "Cargo home")
        != value["root_path"]
        or type(value["root_nlink"]) is not int
        or value["root_nlink"] < 2
        or type(value["file_count"]) is not int
        or value["file_count"] < 1
        or type(value["locked_registry_package_count"]) is not int
        or value["locked_registry_package_count"] < 1
    ):
        _fail("Cargo material identity differs")
    for field in (
        "locked_packages_sha256", "file_registry_sha256",
        "material_manifest_sha256", "sealed_snapshot_manifest_sha256",
        "sealed_tree_manifest_sha256", "stage_2_cargo_evidence_sha256",
    ):
        _root_hex_v1(value[field], f"Cargo material identity {field}")
    return value


def build_fresh_runtime_evidence_set_v1(
    source_commit: str,
    image_rows: Sequence[Mapping[str, object]],
    actor_rows: Sequence[Mapping[str, object]],
    cargo_material_identity: Mapping[str, object],
    cargo_snapshot_evidence: Mapping[str, object],
    cargo_tree_evidence: Mapping[str, object],
    seccomp_rows: Sequence[Mapping[str, object]],
    binary_identity: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact pre-launch fresh replay set shared by rows 5--8."""

    commit = _commit_v1(source_commit, "fresh runtime source")
    if type(image_rows) not in (list, tuple) or len(image_rows) != 2:
        _fail("fresh runtime image rows differ")
    images: list[dict[str, object]] = []
    for raw, label, reference in zip(
        image_rows,
        ("python", "rust"),
        (EXPECTED_PYTHON_IMAGE_REFERENCE, EXPECTED_RUST_IMAGE_REFERENCE),
        strict=True,
    ):
        keys = {"label", "reference", "evidence", "evidence_root"}
        if type(raw) is not dict or set(raw) != keys:
            _fail("fresh runtime image row fields differ")
        evidence = _validate_pinned_image_evidence_v1(raw["evidence"], reference)
        expected_root = fresh_runtime_evidence_object_root_v1(
            "PINNED_IMAGE", label, evidence
        )
        if (
            raw["label"] != label
            or raw["reference"] != reference
            or raw["evidence_root"] != expected_root
        ):
            _fail("fresh runtime image row differs")
        images.append(dict(raw))
    if type(actor_rows) not in (list, tuple) or len(actor_rows) != 3:
        _fail("fresh runtime actor rows differ")
    actors: list[dict[str, object]] = []
    for raw, actor_id in zip(
        actor_rows,
        ("PYTHON_ENDPOINT", "RUST_ENDPOINT", "TRUSTED_HOST_REPLAY"),
        strict=True,
    ):
        keys = {
            "actor_id", "source_identity", "source_identity_root",
            "snapshot_evidence", "snapshot_evidence_root",
        }
        if type(raw) is not dict or set(raw) != keys:
            _fail("fresh runtime actor row fields differ")
        source = _validate_actor_source_identity_v1(
            raw["source_identity"], actor_id, commit
        )
        snapshot = _validate_sealed_tree_evidence_v1(
            raw["snapshot_evidence"], f"{actor_id} snapshot"
        )
        if (
            raw["actor_id"] != actor_id
            or raw["source_identity_root"]
            != fresh_runtime_evidence_object_root_v1(
                "ACTOR_SOURCE", actor_id, source
            )
            or raw["snapshot_evidence_root"]
            != fresh_runtime_evidence_object_root_v1(
                "ACTOR_SNAPSHOT", actor_id, snapshot
            )
            or source["blob_count"] != len(snapshot["file_rows"])
            or source["snapshot_file_registry_sha256"]
            != sha256(
                canonical_json_bytes_v1(
                    [[row[0], row[6], row[7], row[10]] for row in snapshot["file_rows"]]
                )
            ).hexdigest()
        ):
            _fail("fresh runtime actor row differs")
        actors.append(dict(raw))
    material = _validate_cargo_material_identity_v1(cargo_material_identity)
    cargo_snapshot = _validate_sealed_snapshot_evidence_v1(
        cargo_snapshot_evidence, "fresh Cargo snapshot"
    )
    cargo_tree = _validate_sealed_tree_evidence_v1(
        cargo_tree_evidence, "fresh Cargo tree"
    )
    if (
        material["sealed_snapshot_manifest_sha256"]
        != cargo_snapshot["manifest_sha256"]
        or material["sealed_tree_manifest_sha256"]
        != cargo_tree["manifest_sha256"]
        or material["root_path"] != cargo_tree["root_path"]
        or material["root_nlink"] != cargo_tree["root_nlink"]
        or material["file_count"] != len(cargo_tree["file_rows"])
    ):
        _fail("fresh Cargo replay differs from material evidence")
    cargo: dict[str, object] = {
        "material_identity": material,
        "material_identity_root": fresh_runtime_evidence_object_root_v1(
            "CARGO_MATERIAL", "cargo-home", material
        ),
        "snapshot_evidence": cargo_snapshot,
        "snapshot_evidence_root": fresh_runtime_evidence_object_root_v1(
            "CARGO_SNAPSHOT", "cargo-home", cargo_snapshot
        ),
        "tree_evidence": cargo_tree,
        "tree_evidence_root": fresh_runtime_evidence_object_root_v1(
            "CARGO_TREE", "cargo-home", cargo_tree
        ),
    }
    if type(seccomp_rows) not in (list, tuple) or len(seccomp_rows) != 2:
        _fail("fresh runtime seccomp rows differ")
    seccomp: list[dict[str, object]] = []
    host_source = actors[2]["source_identity"]
    host_snapshot = actors[2]["snapshot_evidence"]
    host_snapshot_sha = {row[0]: row[10] for row in host_snapshot["file_rows"]}
    for raw, label, relative in zip(
        seccomp_rows,
        ("runtime", "build"),
        (
            "config/phase3_internal_actor_seccomp_v1.json",
            "config/phase3_m3_offline_build_seccomp_v1.json",
        ),
        strict=True,
    ):
        keys = {"label", "relative_path", "evidence", "evidence_root"}
        if type(raw) is not dict or set(raw) != keys:
            _fail("fresh runtime seccomp row fields differ")
        evidence = _validate_seccomp_evidence_v1(raw["evidence"], relative)
        if (
            raw["label"] != label
            or raw["relative_path"] != relative
            or raw["evidence_root"]
            != fresh_runtime_evidence_object_root_v1(
                "SECCOMP_POLICY", label, evidence
            )
            or host_snapshot_sha.get(relative) != evidence["payload_sha256"]
        ):
            _fail("fresh runtime seccomp row differs")
        seccomp.append(dict(raw))
    binary = _validate_binary_identity_v1(binary_identity)
    binary_row = {
        "identity": binary,
        "identity_root": fresh_runtime_evidence_object_root_v1(
            "RUST_BINARY", "runtime", binary
        ),
    }
    body: dict[str, object] = {
        "schema_version": ACTUAL_FRESH_RUNTIME_EVIDENCE_SET_SCHEMA_VERSION,
        "source_commit": commit,
        "image_rows": images,
        "actor_rows": actors,
        "cargo": cargo,
        "seccomp_rows": seccomp,
        "binary": binary_row,
    }
    value = dict(body)
    value["fresh_runtime_evidence_root"] = sha256(
        ACTUAL_FRESH_RUNTIME_EVIDENCE_SET_ROOT_DOMAIN
        + canonical_json_bytes_v1(body)
    ).hexdigest()
    return value


def validate_fresh_runtime_evidence_set_v1(
    value: object,
    source_commit: str,
) -> dict[str, object]:
    keys = {
        "schema_version", "source_commit", "image_rows", "actor_rows",
        "cargo", "seccomp_rows", "binary", "fresh_runtime_evidence_root",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("fresh runtime evidence set fields differ")
    cargo = value["cargo"]
    binary = value["binary"]
    if (
        type(cargo) is not dict
        or set(cargo)
        != {
            "material_identity", "material_identity_root",
            "snapshot_evidence", "snapshot_evidence_root",
            "tree_evidence", "tree_evidence_root",
        }
        or type(binary) is not dict
        or set(binary) != {"identity", "identity_root"}
    ):
        _fail("fresh runtime Cargo/binary rows differ")
    expected = build_fresh_runtime_evidence_set_v1(
        source_commit,
        value["image_rows"],
        value["actor_rows"],
        cargo["material_identity"],
        cargo["snapshot_evidence"],
        cargo["tree_evidence"],
        value["seccomp_rows"],
        binary["identity"],
    )
    if canonical_json_bytes_v1(value) != canonical_json_bytes_v1(expected):
        _fail("fresh runtime evidence set replay differs")
    return value


def _validate_actor_mount_registry_object_v1(
    value: object,
    exact_command: Sequence[str],
) -> dict[str, object]:
    keys = {
        "schema_version", "role_id", "command_sha256", "mount_rows",
        "container_argv", "security_options", "environment_rows",
        "working_directory", "registry_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("actor mount command registry fields differ")
    role_id = value["role_id"]
    if type(role_id) is not int or role_id not in (1, 2, 3):
        _fail("actor mount command registry role differs")
    if (
        type(exact_command) not in (tuple, list)
        or not exact_command
        or any(type(item) is not str or not item for item in exact_command)
    ):
        _fail("actor mount exact command differs")
    command = list(exact_command)
    if value["command_sha256"] != sha256(
        canonical_json_bytes_v1(command)
    ).hexdigest():
        _fail("actor mount command digest differs")
    observed: dict[str, tuple[str, bool]] = {}
    for index, item in enumerate(command):
        if item != "--mount":
            continue
        if index + 1 >= len(command):
            _fail("actor mount command option is truncated")
        match = re.fullmatch(
            r"type=bind,src=([^,]+),dst=([^,]+)(,readonly)?",
            command[index + 1],
        )
        if match is None:
            _fail("actor mount command bind option differs")
        source, destination, read_only = match.groups()
        if destination in observed:
            _fail("actor mount command destination repeats")
        observed[destination] = (source, read_only is None)
    expected_mount_rows = [
        [destination, observed[destination][0], observed[destination][1]]
        for destination in sorted(observed)
    ]
    if value["mount_rows"] != expected_mount_rows:
        _fail("actor mount registry differs from exact command")
    security_options = [
        item.removeprefix("--security-opt=")
        for item in command
        if item.startswith("--security-opt=")
    ]
    if (
        value["security_options"] != security_options
        or security_options[:1] != ["no-new-privileges"]
        or len(security_options) != 2
        or re.fullmatch(r"seccomp=/[^,]+", security_options[1]) is None
    ):
        _fail("actor mount security option registry differs")
    expected_image = (
        EXPECTED_PYTHON_IMAGE_REFERENCE
        if role_id in (1, 3)
        else EXPECTED_RUST_IMAGE_REFERENCE
    )
    indexes = [index for index, item in enumerate(command) if item == expected_image]
    if (
        len(indexes) != 1
        or indexes[0] == len(command) - 1
        or value["container_argv"] != command[indexes[0] + 1 :]
    ):
        _fail("actor mount payload argv differs")
    work_indexes = [index for index, item in enumerate(command) if item == "-w"]
    if len(work_indexes) > 1 or (
        work_indexes and work_indexes[0] + 1 >= len(command)
    ):
        _fail("actor mount working-directory option differs")
    expected_working_directory = (
        command[work_indexes[0] + 1] if work_indexes else ""
    )
    environment_rows = value["environment_rows"]
    if (
        type(environment_rows) is not list
        or any(
            type(row) is not list
            or len(row) != 2
            or any(type(item) is not str for item in row)
            for row in environment_rows
        )
        or environment_rows != sorted(environment_rows)
        or len({row[0] for row in environment_rows}) != len(environment_rows)
        or value["working_directory"] != expected_working_directory
    ):
        _fail("actor mount command environment/working directory differs")
    body = dict(value)
    registry_root = body.pop("registry_sha256")
    if (
        value["schema_version"]
        != "hegel-phase3a-q05b-sealed-command-mount-registry/1"
        or _root_hex_v1(registry_root, "actor command mount registry")
        != sha256(canonical_json_bytes_v1(body)).hexdigest()
    ):
        _fail("actor command mount registry root differs")
    return value


def _actor_mount_authority_spec_v1(
    role_id: int,
    destination: str,
) -> tuple[str, str]:
    matches = tuple(
        (authority_kind, authority_label)
        for row_role, row_destination, authority_kind, authority_label in (
            ACTUAL_ACTOR_MOUNT_AUTHORITY_REGISTRY
        )
        if row_role == role_id and row_destination == destination
    )
    if len(matches) != 1:
        _fail("actor mount authority registry differs")
    return matches[0]


def build_prelaunch_writable_directory_evidence_v1(
    role_id: int,
    destination: str,
    source_path: str,
    device: int,
    inode: int,
    nlink: int,
    uid: int,
    gid: int,
    mode: int,
) -> dict[str, object]:
    if (
        type(role_id) is not int
        or role_id not in (1, 2, 3)
        or type(destination) is not str
        or _absolute_path_text_v1(source_path, "writable mount") != source_path
        or any(type(item) is not int or item < 0 for item in (device, uid, gid))
        or type(inode) is not int
        or inode < 1
        or type(nlink) is not int
        or nlink < 2
        or type(mode) is not int
        or mode != 0o700
    ):
        _fail("prelaunch writable directory identity differs")
    expected_kind, _ = _actor_mount_authority_spec_v1(role_id, destination)
    if expected_kind != "PRELAUNCH_WRITABLE_DIRECTORY":
        _fail("prelaunch writable directory destination differs")
    body: dict[str, object] = {
        "schema_version": (
            "hegel-phase3a-q05b-prelaunch-writable-directory/1"
        ),
        "role_id": role_id,
        "destination": destination,
        "source_path": source_path,
        "device": device,
        "inode": inode,
        "nlink": nlink,
        "uid": uid,
        "gid": gid,
        "mode": mode,
        "empty_at_prelaunch": True,
    }
    value = dict(body)
    value["directory_identity_root"] = sha256(
        ACTUAL_PRELAUNCH_WRITABLE_DIRECTORY_ROOT_DOMAIN
        + role_id.to_bytes(1, "big")
        + canonical_json_bytes_v1(body)
    ).hexdigest()
    return value


def _validate_prelaunch_writable_directory_evidence_v1(
    value: object,
    role_id: int,
    destination: str,
    source: str,
    source_device: int,
    source_inode: int,
    source_nlink: int,
    source_uid: int,
    source_gid: int,
    source_mode: int,
) -> dict[str, object]:
    if type(value) is not dict:
        _fail("prelaunch writable directory evidence differs")
    keys = {
        "schema_version", "role_id", "destination", "source_path",
        "device", "inode", "nlink", "uid", "gid", "mode",
        "empty_at_prelaunch", "directory_identity_root",
    }
    if set(value) != keys:
        _fail("prelaunch writable directory evidence fields differ")
    expected = build_prelaunch_writable_directory_evidence_v1(
        role_id,
        destination,
        source,
        source_device,
        source_inode,
        source_nlink,
        source_uid,
        source_gid,
        source_mode,
    )
    if canonical_json_bytes_v1(value) != canonical_json_bytes_v1(expected):
        _fail("prelaunch writable directory evidence replay differs")
    return value


def _validate_stdout_mount_file_evidence_v1(
    value: object,
    destination: str,
    source: str,
    source_device: int,
    source_inode: int,
    source_nlink: int,
    source_uid: int,
    source_gid: int,
    source_mode: int,
    source_size: int | None,
) -> dict[str, object]:
    keys = {
        "schema_version", "tree_manifest_sha256", "relative_path", "file_row",
    }
    expected_relative = {
        "/inputs/stdout/manifest.json": "manifest.json",
        "/inputs/stdout/python.stdout": "python.stdout",
        "/inputs/stdout/rust.stdout": "rust.stdout",
    }.get(destination)
    if (
        type(value) is not dict
        or set(value) != keys
        or expected_relative is None
        or value["schema_version"]
        != "hegel-phase3a-q05b-sealed-stdout-mount-file/1"
        or value["relative_path"] != expected_relative
        or type(value["file_row"]) is not list
        or len(value["file_row"]) != 11
    ):
        _fail("sealed stdout mount authority differs")
    _root_hex_v1(value["tree_manifest_sha256"], "stdout tree manifest")
    row = value["file_row"]
    if (
        row[0] != expected_relative
        or any(type(item) is not int for item in row[1:10])
        or row[1] != source_device
        or row[2] != source_inode
        or row[3] != source_nlink
        or row[3] != 1
        or row[4] != source_uid
        or row[5] != source_gid
        or row[6] != source_mode
        or row[7] != source_size
        or row[6] != 0o444
        or _root_hex_v1(row[10], "stdout mount file") != row[10]
        or not source.endswith("/" + expected_relative)
    ):
        _fail("sealed stdout mount file identity differs")
    return value


def _validate_actor_mount_authority_evidence_v1(
    role_id: int,
    destination: str,
    source: str,
    source_device: int,
    source_inode: int,
    source_nlink: int,
    source_uid: int,
    source_gid: int,
    source_mode: int,
    source_size: int | None,
    authority_kind: str,
    authority_label: str,
    authority_evidence: object,
) -> dict[str, object]:
    expected_kind, expected_label = _actor_mount_authority_spec_v1(
        role_id, destination
    )
    if authority_kind != expected_kind or authority_label != expected_label:
        _fail("actor mount authority kind or label differs")
    if expected_kind == "PRELAUNCH_WRITABLE_DIRECTORY":
        return _validate_prelaunch_writable_directory_evidence_v1(
            authority_evidence,
            role_id,
            destination,
            source,
            source_device,
            source_inode,
            source_nlink,
            source_uid,
            source_gid,
            source_mode,
        )
    if expected_kind in ("FRESH_ACTOR_SNAPSHOT", "SEALED_ENDPOINT_TREE"):
        evidence = _validate_sealed_tree_evidence_v1(
            authority_evidence, f"role {role_id} {destination} authority"
        )
        if (
            evidence["root_path"] != source
            or evidence["root_device"] != source_device
            or evidence["root_inode"] != source_inode
            or evidence["root_nlink"] != source_nlink
            or evidence["root_mode"] != source_mode
            or source_size is not None
        ):
            _fail("sealed actor mount tree authority differs")
        return evidence
    if expected_kind == "FRESH_PREBUILT_RUST_BINARY":
        evidence = _validate_binary_identity_v1(authority_evidence)
        if (
            evidence["binary_path"] != source
            or evidence["device"] != source_device
            or evidence["inode"] != source_inode
            or evidence["nlink"] != source_nlink
            or evidence["uid"] != source_uid
            or evidence["gid"] != source_gid
            or evidence["mode"] != source_mode
            or evidence["size"] != source_size
        ):
            _fail("sealed Rust binary mount authority differs")
        return evidence
    if expected_kind == "RUNTIME_SECCOMP_POLICY":
        evidence = _validate_seccomp_evidence_v1(
            authority_evidence, ACTUAL_RUNTIME_SECCOMP_RELATIVE_PATH
        )
        if (
            evidence["absolute_path"] != source
            or evidence["file_device"] != source_device
            or evidence["file_inode"] != source_inode
            or evidence["file_nlink"] != source_nlink
            or evidence["file_uid"] != source_uid
            or evidence["file_gid"] != source_gid
            or evidence["file_mode"] != source_mode
            or evidence["file_size"] != source_size
        ):
            _fail("runtime seccomp mount authority differs")
        return evidence
    if expected_kind == "SEALED_STDOUT_FILE":
        return _validate_stdout_mount_file_evidence_v1(
            authority_evidence,
            destination,
            source,
            source_device,
            source_inode,
            source_nlink,
            source_uid,
            source_gid,
            source_mode,
            source_size,
        )
    _fail("actor mount authority registry kind differs")


def _validate_actor_mount_source_row_v1(
    value: object,
    role_id: int,
    destination: str,
    source: str,
    writable: bool,
    source_type: str,
    source_mode: int,
) -> dict[str, object]:
    keys = {
        "schema_version", "role_id", "destination", "source", "writable",
        "source_type", "source_device", "source_inode", "source_nlink",
        "source_uid", "source_gid",
        "source_mode", "source_size", "authority_kind", "authority_label",
        "authority_evidence", "authority_root",
        "path_matched_held_descriptor", "held_descriptor_read_only",
        "source_root",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("actor mount source fields differ")
    if (
        value["schema_version"] != ACTUAL_ACTOR_MOUNT_SOURCE_SCHEMA_VERSION
        or type(value["role_id"]) is not int
        or value["role_id"] != role_id
        or value["destination"] != destination
        or value["source"] != source
        or type(value["writable"]) is not bool
        or value["writable"] is not writable
        or value["source_type"] != source_type
        or type(value["source_device"]) is not int
        or value["source_device"] < 0
        or type(value["source_inode"]) is not int
        or value["source_inode"] < 1
        or type(value["source_nlink"]) is not int
        or value["source_nlink"] < (2 if source_type == "DIRECTORY" else 1)
        or type(value["source_uid"]) is not int
        or value["source_uid"] < 0
        or type(value["source_gid"]) is not int
        or value["source_gid"] < 0
        or type(value["source_mode"]) is not int
        or value["source_mode"] != source_mode
        or (
            source_type == "DIRECTORY" and value["source_size"] is not None
        )
        or (
            source_type == "REGULAR_FILE"
            and (
                type(value["source_size"]) is not int
                or value["source_size"] < 1
                or value["source_nlink"] != 1
            )
        )
        or type(value["authority_kind"]) is not str
        or type(value["authority_label"]) is not str
        or type(value["authority_evidence"]) is not dict
        or type(value["path_matched_held_descriptor"]) is not bool
        or value["path_matched_held_descriptor"] is not True
        or type(value["held_descriptor_read_only"]) is not bool
        or value["held_descriptor_read_only"] is not True
    ):
        _fail("actor mount source identity differs")
    evidence = _validate_actor_mount_authority_evidence_v1(
        role_id,
        destination,
        source,
        value["source_device"],
        value["source_inode"],
        value["source_nlink"],
        value["source_uid"],
        value["source_gid"],
        value["source_mode"],
        value["source_size"],
        value["authority_kind"],
        value["authority_label"],
        value["authority_evidence"],
    )
    if value["authority_root"] != actor_mount_authority_root_v1(
        value["authority_kind"], value["authority_label"], evidence
    ):
        _fail("actor mount source authority root differs")
    body = dict(value)
    source_root = body.pop("source_root")
    if _root_hex_v1(source_root, "actor mount source") != sha256(
        ACTUAL_ACTOR_MOUNT_SOURCE_ROOT_DOMAIN
        + role_id.to_bytes(1, "big")
        + canonical_json_bytes_v1(body)
    ).hexdigest():
        _fail("actor mount source root differs")
    return value


def actor_mount_authority_root_v1(
    authority_kind: str,
    authority_label: str,
    authority_evidence: Mapping[str, object],
) -> str:
    if (
        type(authority_kind) is not str
        or re.fullmatch(r"[A-Z0-9_]+", authority_kind) is None
        or type(authority_label) is not str
        or re.fullmatch(r"[A-Za-z0-9_./@-]+", authority_label) is None
        or type(authority_evidence) is not dict
    ):
        _fail("actor mount authority identity differs")
    return sha256(
        ACTUAL_ACTOR_MOUNT_AUTHORITY_ROOT_DOMAIN
        + canonical_json_bytes_v1(
            [authority_kind, authority_label, authority_evidence]
        )
    ).hexdigest()


def build_actor_mount_source_row_v1(
    role_id: int,
    destination: str,
    source: str,
    writable: bool,
    source_type: str,
    source_device: int,
    source_inode: int,
    source_nlink: int,
    source_uid: int,
    source_gid: int,
    source_mode: int,
    source_size: int | None,
    authority_kind: str,
    authority_label: str,
    authority_evidence: Mapping[str, object],
) -> dict[str, object]:
    authority_root = actor_mount_authority_root_v1(
        authority_kind, authority_label, authority_evidence
    )
    body: dict[str, object] = {
        "schema_version": ACTUAL_ACTOR_MOUNT_SOURCE_SCHEMA_VERSION,
        "role_id": role_id,
        "destination": destination,
        "source": source,
        "writable": writable,
        "source_type": source_type,
        "source_device": source_device,
        "source_inode": source_inode,
        "source_nlink": source_nlink,
        "source_uid": source_uid,
        "source_gid": source_gid,
        "source_mode": source_mode,
        "source_size": source_size,
        "authority_kind": authority_kind,
        "authority_label": authority_label,
        "authority_evidence": dict(authority_evidence),
        "authority_root": authority_root,
        "path_matched_held_descriptor": True,
        "held_descriptor_read_only": True,
    }
    value = dict(body)
    if type(role_id) is not int or role_id not in (1, 2, 3):
        _fail("actor mount source role differs")
    value["source_root"] = sha256(
        ACTUAL_ACTOR_MOUNT_SOURCE_ROOT_DOMAIN
        + role_id.to_bytes(1, "big")
        + canonical_json_bytes_v1(body)
    ).hexdigest()
    return value


def build_actor_mount_binding_v1(
    exact_command: Sequence[str],
    command_mount_registry: Mapping[str, object],
    source_rows: Sequence[Mapping[str, object]],
    seccomp_row: Mapping[str, object],
) -> dict[str, object]:
    """Bind exact argv-derived sources to held prelaunch inode identities."""

    registry = _validate_actor_mount_registry_object_v1(
        command_mount_registry, exact_command
    )
    role_id = registry["role_id"]
    role_row = ACTUAL_ACTOR_MOUNT_ROLE_REGISTRY[role_id - 1]
    actor_id = role_row[1]
    mount_rows = registry["mount_rows"]
    if (
        type(source_rows) not in (tuple, list)
        or len(source_rows) != len(role_row[2])
        or len(mount_rows) != len(role_row[2])
    ):
        _fail("actor mount source registry length differs")
    validated_sources: list[dict[str, object]] = []
    for value, registry_row, expected in zip(
        source_rows, mount_rows, role_row[2], strict=True
    ):
        destination, writable, source_type, source_mode = expected
        if registry_row[0] != destination or registry_row[2] is not writable:
            _fail("actor mount role registry differs")
        validated_sources.append(
            _validate_actor_mount_source_row_v1(
                value,
                role_id,
                destination,
                registry_row[1],
                writable,
                source_type,
                source_mode,
            )
        )
    seccomp_source = registry["security_options"][1].removeprefix("seccomp=")
    validated_seccomp = _validate_actor_mount_source_row_v1(
        seccomp_row,
        role_id,
        "@seccomp",
        seccomp_source,
        False,
        "REGULAR_FILE",
        0o444,
    )
    if validated_seccomp["authority_kind"] != "RUNTIME_SECCOMP_POLICY":
        _fail("actor runtime seccomp authority kind differs")
    body: dict[str, object] = {
        "schema_version": ACTUAL_ACTOR_MOUNT_BINDING_SCHEMA_VERSION,
        "role_id": role_id,
        "actor_id": actor_id,
        "exact_command": list(exact_command),
        "command_mount_registry": dict(registry),
        "source_rows": validated_sources,
        "seccomp_row": validated_seccomp,
        "all_sources_derived_from_exact_command": True,
        "all_source_paths_match_held_descriptors": True,
    }
    value = dict(body)
    value["mount_binding_root"] = sha256(
        ACTUAL_ACTOR_MOUNT_BINDING_ROOT_DOMAIN
        + role_id.to_bytes(1, "big")
        + canonical_json_bytes_v1(body)
    ).hexdigest()
    return value


def validate_actor_mount_binding_v1(value: object) -> dict[str, object]:
    if type(value) is not dict:
        _fail("actor mount binding object differs")
    required = {
        "schema_version", "role_id", "actor_id", "exact_command",
        "command_mount_registry", "source_rows", "seccomp_row",
        "all_sources_derived_from_exact_command",
        "all_source_paths_match_held_descriptors", "mount_binding_root",
    }
    if set(value) != required:
        _fail("actor mount binding fields differ")
    if (
        type(value["all_sources_derived_from_exact_command"]) is not bool
        or value["all_sources_derived_from_exact_command"] is not True
        or type(value["all_source_paths_match_held_descriptors"]) is not bool
        or value["all_source_paths_match_held_descriptors"] is not True
    ):
        _fail("actor mount binding authority flags differ")
    expected = build_actor_mount_binding_v1(
        value["exact_command"],
        value["command_mount_registry"],
        value["source_rows"],
        value["seccomp_row"],
    )
    if canonical_json_bytes_v1(value) != canonical_json_bytes_v1(expected):
        _fail("actor mount binding replay differs")
    return value


def _actor_mount_launch_payload_sha256_v1(
    source_row: Mapping[str, object],
) -> str | None:
    if source_row["source_type"] == "DIRECTORY":
        return None
    evidence = source_row["authority_evidence"]
    kind = source_row["authority_kind"]
    if kind == "FRESH_PREBUILT_RUST_BINARY":
        digest = evidence.get("sha256")
    elif kind == "RUNTIME_SECCOMP_POLICY":
        digest = evidence.get("payload_sha256")
    elif kind == "SEALED_STDOUT_FILE":
        row = evidence.get("file_row")
        digest = row[10] if type(row) is list and len(row) == 11 else None
    else:
        _fail("actor mount launch regular-file authority differs")
    return _root_hex_v1(digest, "actor mount launch payload")


def _validate_actor_mount_launch_source_replay_v1(
    value: object,
    source_row: Mapping[str, object],
) -> dict[str, object]:
    keys = {
        "destination", "source", "source_device", "source_inode",
        "source_nlink", "source_uid", "source_gid", "source_mode",
        "source_type", "payload_sha256", "path_matches_held_descriptor",
        "held_descriptor_read_only",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("actor mount launch source replay fields differ")
    writable_directory = (
        source_row["writable"] is True
        and source_row["source_type"] == "DIRECTORY"
    )
    expected_payload_sha256 = _actor_mount_launch_payload_sha256_v1(source_row)
    if (
        value["destination"] != source_row["destination"]
        or value["source"] != source_row["source"]
        or type(value["source_device"]) is not int
        or value["source_device"] != source_row["source_device"]
        or type(value["source_inode"]) is not int
        or value["source_inode"] != source_row["source_inode"]
        or type(value["source_nlink"]) is not int
        or (
            value["source_nlink"] < 2
            if writable_directory
            else value["source_nlink"] != source_row["source_nlink"]
        )
        or type(value["source_uid"]) is not int
        or value["source_uid"] != source_row["source_uid"]
        or type(value["source_gid"]) is not int
        or value["source_gid"] != source_row["source_gid"]
        or type(value["source_mode"]) is not int
        or value["source_mode"] != source_row["source_mode"]
        or value["source_type"] != source_row["source_type"]
        or value["payload_sha256"] != expected_payload_sha256
        or type(value["path_matches_held_descriptor"]) is not bool
        or value["path_matches_held_descriptor"] is not True
        or type(value["held_descriptor_read_only"]) is not bool
        or value["held_descriptor_read_only"] is not True
    ):
        _fail("actor mount launch source replay differs")
    return value


def build_actor_mount_launch_replay_v1(
    mount_binding: Mapping[str, object],
    source_replay_rows: Sequence[Mapping[str, object]],
    seccomp_replay: Mapping[str, object],
) -> dict[str, object]:
    """Build the pure post-start replay of every held launch descriptor."""

    binding = validate_actor_mount_binding_v1(mount_binding)
    if (
        type(source_replay_rows) not in (tuple, list)
        or len(source_replay_rows) != len(binding["source_rows"])
    ):
        _fail("actor mount launch source replay registry differs")
    sources = [
        _validate_actor_mount_launch_source_replay_v1(observed, expected)
        for observed, expected in zip(
            source_replay_rows, binding["source_rows"], strict=True
        )
    ]
    seccomp = _validate_actor_mount_launch_source_replay_v1(
        seccomp_replay, binding["seccomp_row"]
    )
    body: dict[str, object] = {
        "schema_version": ACTUAL_ACTOR_MOUNT_LAUNCH_REPLAY_SCHEMA_VERSION,
        "role_id": binding["role_id"],
        "actor_id": binding["actor_id"],
        "mount_binding_root": binding["mount_binding_root"],
        "command_sha256": binding["command_mount_registry"]["command_sha256"],
        "mount_registry_sha256": binding["command_mount_registry"][
            "registry_sha256"
        ],
        "source_replay_rows": sources,
        "seccomp_replay": seccomp,
        "all_paths_match_prelaunch_held_descriptors": True,
        "actor_returned_exact_command_and_registry": True,
    }
    result = dict(body)
    result["launch_replay_root"] = sha256(
        ACTUAL_ACTOR_MOUNT_LAUNCH_REPLAY_ROOT_DOMAIN
        + binding["role_id"].to_bytes(1, "big")
        + canonical_json_bytes_v1(body)
    ).hexdigest()
    return result


def validate_actor_mount_launch_replay_v1(
    value: object,
    mount_binding: Mapping[str, object],
) -> dict[str, object]:
    keys = {
        "schema_version", "role_id", "actor_id", "mount_binding_root",
        "command_sha256", "mount_registry_sha256", "source_replay_rows",
        "seccomp_replay", "all_paths_match_prelaunch_held_descriptors",
        "actor_returned_exact_command_and_registry", "launch_replay_root",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("actor mount launch replay fields differ")
    if (
        type(value["all_paths_match_prelaunch_held_descriptors"]) is not bool
        or value["all_paths_match_prelaunch_held_descriptors"] is not True
        or type(value["actor_returned_exact_command_and_registry"]) is not bool
        or value["actor_returned_exact_command_and_registry"] is not True
    ):
        _fail("actor mount launch replay flags differ")
    expected = build_actor_mount_launch_replay_v1(
        mount_binding,
        value["source_replay_rows"],
        value["seccomp_replay"],
    )
    if canonical_json_bytes_v1(value) != canonical_json_bytes_v1(expected):
        _fail("actor mount launch replay differs")
    return value


def checkpoint_mount_registry_root_v1(
    checkpoint_id: int,
    mount_binding_rows: Sequence[Mapping[str, object]],
) -> str:
    if type(checkpoint_id) is not int or checkpoint_id not in (1, 2, 3):
        _fail("checkpoint mount registry id differs")
    expected_roles = {1: (1, 2), 2: (3,), 3: (1, 2, 3)}[checkpoint_id]
    if (
        type(mount_binding_rows) not in (tuple, list)
        or len(mount_binding_rows) != len(expected_roles)
    ):
        _fail("checkpoint mount binding registry length differs")
    values = [validate_actor_mount_binding_v1(row) for row in mount_binding_rows]
    if tuple(row["role_id"] for row in values) != expected_roles:
        _fail("checkpoint mount binding role order differs")
    return sha256(
        ACTUAL_CHECKPOINT_MOUNT_REGISTRY_ROOT_DOMAIN
        + checkpoint_id.to_bytes(1, "big")
        + canonical_json_bytes_v1(
            [[row["role_id"], row["mount_binding_root"]] for row in values]
        )
    ).hexdigest()


def build_actual_stage_5_evidence_v1(
    source_commit: str,
    actor_completion_rows: Sequence[Mapping[str, object]],
    five_sidecars: Mapping[str, object],
    endpoint_stdout_set: Mapping[str, object],
    strict_endpoint_replay_roots: Sequence[str],
    injected_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Build the exact Stage-5 wire surface; strong validation is separate."""

    commit = _commit_v1(source_commit, "Stage-5 source")
    if (
        type(injected_evidence) is not dict
        or set(injected_evidence) != set(ACTUAL_STAGE_5_INJECTED_EVIDENCE_KEYS)
    ):
        _fail("dynamic authority Stage-5 injected evidence fields differ")
    evidence: dict[str, object] = {
        "actor_completion_rows": list(actor_completion_rows),
        "five_sidecars": dict(five_sidecars),
        "endpoint_stdout_set": dict(endpoint_stdout_set),
        "strict_endpoint_replay_roots": list(strict_endpoint_replay_roots),
        "qualification_receipt": None,
        **dict(injected_evidence),
    }
    body: dict[str, object] = {
        "candidate_receipt_hex": None,
        "evidence": evidence,
        "final_receipt_hex": None,
        "q1_authority": {
            "certificate_active": False,
            "formal_output_roots": [None] * 8,
            "gate_count": 0,
            "gate_mask": 0,
            "state": "NOT_RUN",
        },
        "qualification_count": 0,
        "qualification_mask": 0,
        "schema_version": ACTUAL_STAGE_SCHEMA_VERSION,
        "source_commit": commit,
        "stage_id": 5,
        "stage_name": ACTUAL_STAGE_5_NAME,
        "status": "STAGE_COMPLETE_IN_MEMORY_NOT_PUBLISHED",
    }
    value = dict(body)
    value["stage_evidence_root"] = sha256(
        ACTUAL_STAGE_EVIDENCE_ROOT_DOMAIN
        + (5).to_bytes(2, "big")
        + canonical_json_bytes_v1(body)
    ).hexdigest()
    return validate_actual_stage_5_evidence_surface_v1(value, commit)


def _validate_stage_5_work_root_replay_v1(value: object) -> dict[str, object]:
    keys = {
        "schema_version", "absolute_path", "device", "inode", "nlink",
        "mode", "path_matches_anchored_descriptor",
    }
    if (
        type(value) is not dict
        or set(value) != keys
        or value["schema_version"]
        != "hegel-phase3a-q05b-admission-work-root-replay/1"
        or _absolute_path_text_v1(value["absolute_path"], "Stage-5 work root")
        != value["absolute_path"]
        or type(value["device"]) is not int
        or value["device"] < 0
        or type(value["inode"]) is not int
        or value["inode"] < 1
        or type(value["nlink"]) is not int
        or value["nlink"] < 2
        or type(value["mode"]) is not int
        or value["mode"] != 0o700
        or type(value["path_matches_anchored_descriptor"]) is not bool
        or value["path_matches_anchored_descriptor"] is not True
    ):
        _fail("dynamic authority Stage-5 work-root replay differs")
    return value


def _validate_stage_5_root_rows_v1(
    value: object,
    expected: tuple[tuple[int, str], ...],
    name: str,
) -> list[list[object]]:
    if type(value) is not list or len(value) != len(expected):
        _fail(f"dynamic authority Stage-5 {name} registry differs")
    result: list[list[object]] = []
    for row, (expected_role, expected_label) in zip(value, expected, strict=True):
        if (
            type(row) is not list
            or len(row) != 3
            or type(row[0]) is not int
            or row[0] != expected_role
            or type(row[1]) is not str
            or row[1] != expected_label
        ):
            _fail(f"dynamic authority Stage-5 {name} row differs")
        result.append(
            [expected_role, expected_label, _root_hex_v1(row[2], name)]
        )
    return result


def _validate_stage_5_surface_preimage_v1(
    value: object,
    source_commit: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    keys = {
        "candidate_receipt_hex", "evidence", "final_receipt_hex",
        "q1_authority", "qualification_count", "qualification_mask",
        "schema_version", "source_commit", "stage_evidence_root",
        "stage_id", "stage_name", "status",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("dynamic authority Stage-5 preimage fields differ")
    commit = _commit_v1(source_commit, "dynamic authority Stage-5 source")
    q1 = value["q1_authority"]
    if (
        value["schema_version"] != ACTUAL_STAGE_SCHEMA_VERSION
        or value["source_commit"] != commit
        or type(value["stage_id"]) is not int
        or value["stage_id"] != 5
        or value["stage_name"] != ACTUAL_STAGE_5_NAME
        or value["status"] != "STAGE_COMPLETE_IN_MEMORY_NOT_PUBLISHED"
        or type(value["qualification_count"]) is not int
        or value["qualification_count"] != 0
        or type(value["qualification_mask"]) is not int
        or value["qualification_mask"] != 0
        or value["candidate_receipt_hex"] is not None
        or value["final_receipt_hex"] is not None
        or type(q1) is not dict
        or set(q1) != {
            "certificate_active", "formal_output_roots", "gate_count",
            "gate_mask", "state",
        }
        or type(q1["certificate_active"]) is not bool
        or q1["certificate_active"] is not False
        or type(q1["formal_output_roots"]) is not list
        or len(q1["formal_output_roots"]) != 8
        or any(root is not None for root in q1["formal_output_roots"])
        or type(q1["gate_count"]) is not int
        or q1["gate_count"] != 0
        or type(q1["gate_mask"]) is not int
        or q1["gate_mask"] != 0
        or q1["state"] != "NOT_RUN"
        or type(value["evidence"]) is not dict
    ):
        _fail("dynamic authority Stage-5 preimage identity differs")
    evidence = value["evidence"]
    if (
        set(evidence)
        != set(ACTUAL_STAGE_5_BASE_EVIDENCE_KEYS)
        | set(ACTUAL_STAGE_5_INJECTED_EVIDENCE_KEYS)
        or type(evidence["actor_completion_rows"]) is not list
        or len(evidence["actor_completion_rows"]) != 2
        or any(type(row) is not dict for row in evidence["actor_completion_rows"])
        or type(evidence["strict_endpoint_replay_roots"]) is not list
        or len(evidence["strict_endpoint_replay_roots"]) != 2
        or any(
            type(root) is not str or re.fullmatch(r"[0-9a-f]{64}", root) is None
            for root in evidence["strict_endpoint_replay_roots"]
        )
        or len(set(evidence["strict_endpoint_replay_roots"])) != 2
        or evidence["qualification_receipt"] is not None
    ):
        _fail("dynamic authority Stage-5 full evidence differs")
    attempt_id = _root_hex_v1(
        evidence["actual_admission_attempt_id"], "Stage-5 attempt"
    )
    boundary_root = _root_hex_v1(
        evidence["actual_admission_boundary_root"], "Stage-5 boundary"
    )
    issue_record_root = _root_hex_v1(
        evidence["actual_admission_issue_record_root"], "Stage-5 issue record"
    )
    consumed = evidence["actual_admission_consumed_marker_evidence"]
    work = _validate_stage_5_work_root_replay_v1(
        evidence["actual_admission_work_root_replay"]
    )
    validate_git_source_transcript_v1(
        evidence["actual_admission_consume_git_source_transcript"], commit
    )
    absence = evidence["actual_admission_consume_artifact_absence"]
    if type(absence) is not dict or type(absence.get("artifact_path")) is not str:
        _fail("dynamic authority Stage-5 artifact absence differs")
    validate_artifact_absence_evidence_v1(absence, absence["artifact_path"])
    checkpoint_rows = _validate_stage_5_root_rows_v1(
        evidence["actual_admission_fresh_checkpoint_root_rows"],
        (ACTUAL_FRESH_RUNTIME_CHECKPOINT_REGISTRY[0],),
        "fresh checkpoint",
    )
    binding_rows = _validate_stage_5_root_rows_v1(
        evidence["actual_actor_mount_binding_root_rows"],
        tuple((row[0], row[1]) for row in ACTUAL_ACTOR_MOUNT_ROLE_REGISTRY[:2]),
        "mount binding",
    )
    launch_rows = _validate_stage_5_root_rows_v1(
        evidence["actual_actor_mount_launch_root_rows"],
        tuple((row[0], row[1]) for row in ACTUAL_ACTOR_MOUNT_ROLE_REGISTRY[:2]),
        "mount launch",
    )
    live = validate_actual_admission_live_marker_replay_surface_v1(
        evidence["actual_admission_live_marker_replay"],
        "STAGE_05_BEFORE_EVIDENCE",
    )
    if (
        type(consumed) is not dict
        or consumed.get("attempt_id") != attempt_id
        or consumed.get("boundary_root") != boundary_root
        or consumed.get("issue_record_root") != issue_record_root
        or type(consumed.get("consumed_marker_root")) is not str
        or live["attempt_id"] != attempt_id
        or live["boundary_root"] != boundary_root
        or live["issue_record_root"] != issue_record_root
        or live["consumed_marker_root"] != consumed["consumed_marker_root"]
        or live["work_root_device"] != work["device"]
        or live["work_root_inode"] != work["inode"]
        or live["work_root_nlink"] != work["nlink"]
        or live["work_root_mode"] != work["mode"]
        or checkpoint_rows[0][0] != 1
        or [row[0] for row in binding_rows] != [1, 2]
        or [row[0] for row in launch_rows] != [1, 2]
    ):
        _fail("dynamic authority Stage-5 admission binding differs")
    body = dict(value)
    stage_root = body.pop("stage_evidence_root")
    if _root_hex_v1(stage_root, "dynamic authority Stage-5") != sha256(
        ACTUAL_STAGE_EVIDENCE_ROOT_DOMAIN
        + (5).to_bytes(2, "big")
        + canonical_json_bytes_v1(body)
    ).hexdigest():
        _fail("dynamic authority Stage-5 root differs")
    five = evidence["five_sidecars"]
    stdout = evidence["endpoint_stdout_set"]
    if (
        type(five) is not dict
        or set(five) != {
            "canonical_rows", "python_output_tree", "rust_output_tree"
        }
        or type(five["canonical_rows"]) is not list
        or type(stdout) is not dict
        or set(stdout) != {
            "python_stdout_hex", "rust_stdout_hex", "manifest_hex",
            "sealed_stdout_tree",
        }
        or any(
            type(stdout[field]) is not str
            or re.fullmatch(r"(?:[0-9a-f]{2})*", stdout[field]) is None
            for field in ("python_stdout_hex", "rust_stdout_hex", "manifest_hex")
        )
    ):
        _fail("dynamic authority Stage-5 projection differs")
    python_tree = _validate_sealed_tree_evidence_v1(
        five["python_output_tree"], "Stage-5 Python output"
    )
    rust_tree = _validate_sealed_tree_evidence_v1(
        five["rust_output_tree"], "Stage-5 Rust output"
    )
    stdout_tree = _validate_sealed_tree_evidence_v1(
        stdout["sealed_stdout_tree"], "Stage-5 stdout"
    )
    return value, python_tree, rust_tree, stdout_tree


def validate_actual_stage_5_evidence_surface_v1(
    value: object,
    source_commit: str,
) -> dict[str, object]:
    """Check the Stage-5 wire surface only; this is not admission authority."""

    stage, _python_tree, _rust_tree, _stdout_tree = (
        _validate_stage_5_surface_preimage_v1(value, source_commit)
    )
    return stage


def _validate_stage_5_checkpoint_1_v1(
    value: object,
    source_commit: str,
    issue_record: Mapping[str, object],
    consumed_marker_evidence: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    keys = {
        "schema_version", "source_commit", "artifact_path", "checkpoint_id",
        "checkpoint_name", "attempt_id", "boundary_root",
        "issue_record_root", "consumed_marker_root",
        "issue_fresh_runtime_evidence_root",
        "issue_fresh_runtime_evidence_sha256",
        "observed_fresh_runtime_evidence",
        "observed_fresh_runtime_evidence_root",
        "observed_fresh_runtime_evidence_sha256", "canonical_sets_byte_equal",
        "artifact_absence_evidence", "mount_binding_rows",
        "mount_registry_root", "dynamic_authority_set",
        "dynamic_authority_root", "checkpoint_root",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("strong Stage-5 checkpoint-1 fields differ")
    commit = _commit_v1(source_commit, "strong Stage-5 checkpoint-1 source")
    record, _boundary = validate_actual_admission_issue_record_v1(issue_record)
    consumed = validate_actual_admission_consumed_marker_evidence_v1(
        consumed_marker_evidence, record
    )
    path = _absolute_path_text_v1(
        value["artifact_path"], "strong Stage-5 checkpoint-1 artifact"
    )
    observed = validate_fresh_runtime_evidence_set_v1(
        value["observed_fresh_runtime_evidence"], commit
    )
    observed_bytes = canonical_json_bytes_v1(observed)
    observed_sha256 = sha256(observed_bytes).hexdigest()
    validate_artifact_absence_evidence_v1(
        value["artifact_absence_evidence"], path
    )
    if type(value["mount_binding_rows"]) is not list:
        _fail("strong Stage-5 checkpoint-1 mount bindings differ")
    bindings = [
        validate_actor_mount_binding_v1(row)
        for row in value["mount_binding_rows"]
    ]
    if tuple(row["role_id"] for row in bindings) != (1, 2):
        _fail("strong Stage-5 checkpoint-1 mount role order differs")
    for binding in bindings:
        _cross_actor_mount_binding_to_fresh_runtime_v1(binding, observed)
    mount_root = checkpoint_mount_registry_root_v1(1, bindings)
    if (
        value["schema_version"]
        != ACTUAL_FRESH_RUNTIME_CHECKPOINT_SCHEMA_VERSION
        or value["source_commit"] != commit
        or type(value["checkpoint_id"]) is not int
        or value["checkpoint_id"] != 1
        or value["checkpoint_name"]
        != ACTUAL_FRESH_RUNTIME_CHECKPOINT_REGISTRY[0][1]
        or value["attempt_id"] != record["attempt_id"]
        or value["boundary_root"] != record["boundary_root"]
        or value["issue_record_root"] != record["issue_record_root"]
        or value["consumed_marker_root"] != consumed["consumed_marker_root"]
        or value["issue_fresh_runtime_evidence_root"]
        != observed["fresh_runtime_evidence_root"]
        or value["observed_fresh_runtime_evidence_root"]
        != observed["fresh_runtime_evidence_root"]
        or value["issue_fresh_runtime_evidence_sha256"] != observed_sha256
        or value["observed_fresh_runtime_evidence_sha256"] != observed_sha256
        or type(value["canonical_sets_byte_equal"]) is not bool
        or value["canonical_sets_byte_equal"] is not True
        or value["mount_registry_root"] != mount_root
        or value["dynamic_authority_set"] is not None
        or value["dynamic_authority_root"] is not None
    ):
        _fail("strong Stage-5 checkpoint-1 identity differs")
    body = dict(value)
    checkpoint_root = body.pop("checkpoint_root")
    if _root_hex_v1(checkpoint_root, "strong Stage-5 checkpoint-1") != sha256(
        ACTUAL_FRESH_RUNTIME_CHECKPOINT_ROOT_DOMAIN
        + (1).to_bytes(1, "big")
        + canonical_json_bytes_v1(body)
    ).hexdigest():
        _fail("strong Stage-5 checkpoint-1 root differs")
    return value, bindings


def _validate_actual_stage_5_evidence_strong_preimage_v1(
    value: object,
    source_commit: str,
    *,
    issue_record: Mapping[str, object],
    consumed_marker_evidence: Mapping[str, object],
    checkpoint_1: Mapping[str, object],
    mount_launch_replay_rows: Sequence[Mapping[str, object]],
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """Validate full Stage-5 authority against its external causal preimages."""

    stage, _python_tree, _rust_tree, _stdout_tree = (
        _validate_stage_5_surface_preimage_v1(value, source_commit)
    )
    record, _boundary = validate_actual_admission_issue_record_v1(issue_record)
    consumed = validate_actual_admission_consumed_marker_evidence_v1(
        consumed_marker_evidence, record
    )
    checkpoint, bindings = _validate_stage_5_checkpoint_1_v1(
        checkpoint_1, source_commit, record, consumed
    )
    if (
        type(mount_launch_replay_rows) is not list
        or len(mount_launch_replay_rows) != 2
    ):
        _fail("strong Stage-5 launch replay registry differs")
    launches = [
        validate_actor_mount_launch_replay_v1(launch, binding)
        for launch, binding in zip(
            mount_launch_replay_rows, bindings, strict=True
        )
    ]
    evidence = stage["evidence"]
    live = validate_actual_admission_live_marker_replay_surface_v1(
        evidence["actual_admission_live_marker_replay"],
        "STAGE_05_BEFORE_EVIDENCE",
        record,
        consumed,
    )
    work = _validate_stage_5_work_root_replay_v1(
        evidence["actual_admission_work_root_replay"]
    )
    checkpoint_rows = _validate_stage_5_root_rows_v1(
        evidence["actual_admission_fresh_checkpoint_root_rows"],
        (ACTUAL_FRESH_RUNTIME_CHECKPOINT_REGISTRY[0],),
        "fresh checkpoint",
    )
    binding_rows = _validate_stage_5_root_rows_v1(
        evidence["actual_actor_mount_binding_root_rows"],
        tuple((row[0], row[1]) for row in ACTUAL_ACTOR_MOUNT_ROLE_REGISTRY[:2]),
        "mount binding",
    )
    launch_rows = _validate_stage_5_root_rows_v1(
        evidence["actual_actor_mount_launch_root_rows"],
        tuple((row[0], row[1]) for row in ACTUAL_ACTOR_MOUNT_ROLE_REGISTRY[:2]),
        "mount launch",
    )
    if (
        evidence["actual_admission_attempt_id"] != record["attempt_id"]
        or evidence["actual_admission_boundary_root"] != record["boundary_root"]
        or evidence["actual_admission_issue_record_root"]
        != record["issue_record_root"]
        or canonical_json_bytes_v1(
            evidence["actual_admission_consumed_marker_evidence"]
        ) != canonical_json_bytes_v1(consumed)
        or canonical_json_bytes_v1(
            evidence["actual_admission_consume_artifact_absence"]
        ) != canonical_json_bytes_v1(checkpoint["artifact_absence_evidence"])
        or checkpoint_rows
        != [[
            1,
            ACTUAL_FRESH_RUNTIME_CHECKPOINT_REGISTRY[0][1],
            checkpoint["checkpoint_root"],
        ]]
        or binding_rows
        != [[
            row["role_id"], row["actor_id"], row["mount_binding_root"]
        ] for row in bindings]
        or launch_rows
        != [[
            row["role_id"], row["actor_id"], row["launch_replay_root"]
        ] for row in launches]
        or live["work_root_device"] != work["device"]
        or live["work_root_inode"] != work["inode"]
        or live["work_root_nlink"] != work["nlink"]
        or live["work_root_mode"] != work["mode"]
    ):
        _fail("strong Stage-5 causal authority differs")
    return stage, _python_tree, _rust_tree, _stdout_tree


def validate_actual_stage_5_evidence_v1(
    value: object,
    source_commit: str,
    *,
    issue_record: Mapping[str, object],
    consumed_marker_evidence: Mapping[str, object],
    checkpoint_1: Mapping[str, object],
    mount_launch_replay_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Public strong Stage-5 validator; all causal preimages are mandatory."""

    stage, _python_tree, _rust_tree, _stdout_tree = (
        _validate_actual_stage_5_evidence_strong_preimage_v1(
            value,
            source_commit,
            issue_record=issue_record,
            consumed_marker_evidence=consumed_marker_evidence,
            checkpoint_1=checkpoint_1,
            mount_launch_replay_rows=mount_launch_replay_rows,
        )
    )
    return stage


def build_dynamic_mount_authority_set_v1(
    source_commit: str,
    stage_5_evidence: Mapping[str, object],
    python_output_tree: Mapping[str, object],
    rust_output_tree: Mapping[str, object],
    stdout_tree: Mapping[str, object],
    *,
    issue_record: Mapping[str, object],
    consumed_marker_evidence: Mapping[str, object],
    checkpoint_1: Mapping[str, object],
    mount_launch_replay_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    stage, stage_python, stage_rust, stage_stdout = (
        _validate_actual_stage_5_evidence_strong_preimage_v1(
            stage_5_evidence,
            source_commit,
            issue_record=issue_record,
            consumed_marker_evidence=consumed_marker_evidence,
            checkpoint_1=checkpoint_1,
            mount_launch_replay_rows=mount_launch_replay_rows,
        )
    )
    for observed, expected, name in (
        (python_output_tree, stage_python, "Python output"),
        (rust_output_tree, stage_rust, "Rust output"),
        (stdout_tree, stage_stdout, "stdout"),
    ):
        if canonical_json_bytes_v1(observed) != canonical_json_bytes_v1(expected):
            _fail(f"dynamic mount {name} differs from Stage-5 preimage")
    body: dict[str, object] = {
        "schema_version": ACTUAL_DYNAMIC_MOUNT_AUTHORITY_SET_SCHEMA_VERSION,
        "stage_5_evidence_root": stage["stage_evidence_root"],
        "python_output_tree": _validate_sealed_tree_evidence_v1(
            python_output_tree, "dynamic Python output"
        ),
        "rust_output_tree": _validate_sealed_tree_evidence_v1(
            rust_output_tree, "dynamic Rust output"
        ),
        "stdout_tree": _validate_sealed_tree_evidence_v1(
            stdout_tree, "dynamic stdout tree"
        ),
    }
    value = dict(body)
    value["dynamic_authority_root"] = sha256(
        ACTUAL_DYNAMIC_MOUNT_AUTHORITY_SET_ROOT_DOMAIN
        + canonical_json_bytes_v1(body)
    ).hexdigest()
    return value


def validate_dynamic_mount_authority_set_v1(
    value: object,
    source_commit: str,
    stage_5_evidence: Mapping[str, object],
    *,
    issue_record: Mapping[str, object],
    consumed_marker_evidence: Mapping[str, object],
    checkpoint_1: Mapping[str, object],
    mount_launch_replay_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if type(value) is not dict:
        _fail("dynamic mount authority set differs")
    keys = {
        "schema_version", "stage_5_evidence_root", "python_output_tree",
        "rust_output_tree", "stdout_tree", "dynamic_authority_root",
    }
    if set(value) != keys:
        _fail("dynamic mount authority set fields differ")
    expected = build_dynamic_mount_authority_set_v1(
        source_commit,
        stage_5_evidence,
        value["python_output_tree"],
        value["rust_output_tree"],
        value["stdout_tree"],
        issue_record=issue_record,
        consumed_marker_evidence=consumed_marker_evidence,
        checkpoint_1=checkpoint_1,
        mount_launch_replay_rows=mount_launch_replay_rows,
    )
    if canonical_json_bytes_v1(value) != canonical_json_bytes_v1(expected):
        _fail("dynamic mount authority set replay differs")
    return value


def decode_dynamic_mount_authority_set_v1(
    payload: bytes,
    source_commit: str,
    stage_5_evidence: Mapping[str, object],
    *,
    issue_record: Mapping[str, object],
    consumed_marker_evidence: Mapping[str, object],
    checkpoint_1: Mapping[str, object],
    mount_launch_replay_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    value = _strict_json_object_v1(payload, "dynamic mount authority set")
    expected = validate_dynamic_mount_authority_set_v1(
        value,
        source_commit,
        stage_5_evidence,
        issue_record=issue_record,
        consumed_marker_evidence=consumed_marker_evidence,
        checkpoint_1=checkpoint_1,
        mount_launch_replay_rows=mount_launch_replay_rows,
    )
    if canonical_json_bytes_v1(expected) != payload:
        _fail("dynamic mount authority set canonical bytes differ")
    return expected


def _stdout_authority_from_tree_v1(
    stdout_tree: Mapping[str, object],
    relative_path: str,
) -> dict[str, object]:
    matches = [row for row in stdout_tree["file_rows"] if row[0] == relative_path]
    if len(matches) != 1:
        _fail("dynamic stdout authority row differs")
    return {
        "schema_version": "hegel-phase3a-q05b-sealed-stdout-mount-file/1",
        "tree_manifest_sha256": stdout_tree["manifest_sha256"],
        "relative_path": relative_path,
        "file_row": matches[0],
    }


def _cross_role3_mount_binding_to_dynamic_authority_v1(
    binding: Mapping[str, object],
    dynamic: Mapping[str, object],
) -> None:
    if binding["role_id"] != 3:
        _fail("dynamic mount authority role differs")
    sources = {row["destination"]: row for row in binding["source_rows"]}
    expected = {
        "/inputs/python": dynamic["python_output_tree"],
        "/inputs/rust": dynamic["rust_output_tree"],
        "/inputs/stdout/manifest.json": _stdout_authority_from_tree_v1(
            dynamic["stdout_tree"], "manifest.json"
        ),
        "/inputs/stdout/python.stdout": _stdout_authority_from_tree_v1(
            dynamic["stdout_tree"], "python.stdout"
        ),
        "/inputs/stdout/rust.stdout": _stdout_authority_from_tree_v1(
            dynamic["stdout_tree"], "rust.stdout"
        ),
    }
    for destination, evidence in expected.items():
        if canonical_json_bytes_v1(
            sources[destination]["authority_evidence"]
        ) != canonical_json_bytes_v1(evidence):
            _fail("role3 mount authority differs from stage5 dynamic set")


def _cross_actor_mount_binding_to_fresh_runtime_v1(
    binding: Mapping[str, object],
    fresh: Mapping[str, object],
) -> None:
    role_id = binding["role_id"]
    actors = {row["actor_id"]: row for row in fresh["actor_rows"]}
    runtime_seccomp = next(
        (row for row in fresh["seccomp_rows"] if row["label"] == "runtime"),
        None,
    )
    if type(runtime_seccomp) is not dict:
        _fail("fresh runtime seccomp authority is absent")
    if canonical_json_bytes_v1(binding["seccomp_row"]["authority_evidence"]) != (
        canonical_json_bytes_v1(runtime_seccomp["evidence"])
    ):
        _fail("actor mount seccomp authority differs from fresh runtime")
    source_by_destination = {
        row["destination"]: row for row in binding["source_rows"]
    }
    if role_id == 1:
        expected = actors["PYTHON_ENDPOINT"]["snapshot_evidence"]
        observed = source_by_destination["/snapshot"]["authority_evidence"]
    elif role_id == 2:
        expected = fresh["binary"]["identity"]
        observed = source_by_destination[
            "/runtime/hegel-q1-archive-projection-oracle"
        ]["authority_evidence"]
    elif role_id == 3:
        expected = actors["TRUSTED_HOST_REPLAY"]["snapshot_evidence"]
        observed = source_by_destination["/snapshot"]["authority_evidence"]
    else:
        _fail("actor mount fresh authority role differs")
    if canonical_json_bytes_v1(observed) != canonical_json_bytes_v1(expected):
        _fail("actor mount immutable authority differs from fresh runtime")


def build_fresh_runtime_checkpoint_v1(
    source_commit: str,
    artifact_path: str,
    checkpoint_id: int,
    attempt_id: str,
    boundary_root: str,
    issue_record_root: str,
    consumed_marker_root: str,
    issue_fresh_runtime_evidence: Mapping[str, object],
    observed_fresh_runtime_evidence: Mapping[str, object],
    artifact_absence_evidence: Mapping[str, object],
    mount_binding_rows: Sequence[Mapping[str, object]],
    dynamic_authority_set: Mapping[str, object] | None = None,
    stage_5_evidence: Mapping[str, object] | None = None,
    *,
    stage_5_issue_record: Mapping[str, object] | None = None,
    stage_5_consumed_marker_evidence: Mapping[str, object] | None = None,
    stage_5_checkpoint_1: Mapping[str, object] | None = None,
    stage_5_mount_launch_replay_rows: (
        Sequence[Mapping[str, object]] | None
    ) = None,
) -> dict[str, object]:
    """Bind one ordered post-spend fresh replay to the admitted set bytes."""

    commit = _commit_v1(source_commit, "fresh checkpoint source")
    path = _absolute_path_text_v1(artifact_path, "fresh checkpoint artifact")
    if type(checkpoint_id) is not int or not 1 <= checkpoint_id <= 3:
        _fail("fresh runtime checkpoint id differs")
    expected_id, checkpoint_name = ACTUAL_FRESH_RUNTIME_CHECKPOINT_REGISTRY[
        checkpoint_id - 1
    ]
    if checkpoint_id != expected_id:
        _fail("fresh runtime checkpoint registry differs")
    roots = {
        "attempt_id": _root_hex_v1(attempt_id, "fresh checkpoint attempt"),
        "boundary_root": _root_hex_v1(
            boundary_root, "fresh checkpoint boundary"
        ),
        "issue_record_root": _root_hex_v1(
            issue_record_root, "fresh checkpoint issue record"
        ),
        "consumed_marker_root": _root_hex_v1(
            consumed_marker_root, "fresh checkpoint consumed marker"
        ),
    }
    issue = validate_fresh_runtime_evidence_set_v1(
        issue_fresh_runtime_evidence, commit
    )
    observed = validate_fresh_runtime_evidence_set_v1(
        observed_fresh_runtime_evidence, commit
    )
    issue_bytes = canonical_json_bytes_v1(issue)
    observed_bytes = canonical_json_bytes_v1(observed)
    if issue_bytes != observed_bytes:
        _fail("fresh runtime checkpoint set bytes differ")
    absence = validate_artifact_absence_evidence_v1(
        artifact_absence_evidence, path
    )
    bindings = [validate_actor_mount_binding_v1(row) for row in mount_binding_rows]
    for binding in bindings:
        _cross_actor_mount_binding_to_fresh_runtime_v1(binding, issue)
    dynamic: dict[str, object] | None
    if checkpoint_id == 1:
        if dynamic_authority_set is not None or stage_5_evidence is not None:
            _fail("endpoint checkpoint carried dynamic host authority")
        dynamic = None
    else:
        dynamic = validate_dynamic_mount_authority_set_v1(
            dynamic_authority_set,
            commit,
            stage_5_evidence,
            issue_record=stage_5_issue_record,
            consumed_marker_evidence=stage_5_consumed_marker_evidence,
            checkpoint_1=stage_5_checkpoint_1,
            mount_launch_replay_rows=stage_5_mount_launch_replay_rows,
        )
        role3 = next((row for row in bindings if row["role_id"] == 3), None)
        if type(role3) is not dict:
            _fail("host checkpoint lacks role3 mount binding")
        _cross_role3_mount_binding_to_dynamic_authority_v1(role3, dynamic)
    mount_registry_root = checkpoint_mount_registry_root_v1(
        checkpoint_id, bindings
    )
    body: dict[str, object] = {
        "schema_version": ACTUAL_FRESH_RUNTIME_CHECKPOINT_SCHEMA_VERSION,
        "source_commit": commit,
        "artifact_path": path,
        "checkpoint_id": checkpoint_id,
        "checkpoint_name": checkpoint_name,
        **roots,
        "issue_fresh_runtime_evidence_root": issue[
            "fresh_runtime_evidence_root"
        ],
        "issue_fresh_runtime_evidence_sha256": sha256(issue_bytes).hexdigest(),
        "observed_fresh_runtime_evidence": observed,
        "observed_fresh_runtime_evidence_root": observed[
            "fresh_runtime_evidence_root"
        ],
        "observed_fresh_runtime_evidence_sha256": sha256(
            observed_bytes
        ).hexdigest(),
        "canonical_sets_byte_equal": True,
        "artifact_absence_evidence": absence,
        "mount_binding_rows": bindings,
        "mount_registry_root": mount_registry_root,
        "dynamic_authority_set": dynamic,
        "dynamic_authority_root": (
            None if dynamic is None else dynamic["dynamic_authority_root"]
        ),
    }
    value = dict(body)
    value["checkpoint_root"] = sha256(
        ACTUAL_FRESH_RUNTIME_CHECKPOINT_ROOT_DOMAIN
        + checkpoint_id.to_bytes(1, "big")
        + canonical_json_bytes_v1(body)
    ).hexdigest()
    if len(canonical_json_bytes_v1(value)) > ACTUAL_FRESH_RUNTIME_CHECKPOINT_MAX_BYTES:
        _fail("fresh runtime checkpoint exceeds frozen byte limit")
    return value


def decode_fresh_runtime_checkpoint_v1(
    payload: bytes,
    source_commit: str,
    artifact_path: str,
    checkpoint_id: int,
    attempt_id: str,
    boundary_root: str,
    issue_record_root: str,
    consumed_marker_root: str,
    issue_fresh_runtime_evidence: Mapping[str, object],
    artifact_absence_evidence: Mapping[str, object],
    mount_binding_rows: Sequence[Mapping[str, object]],
    dynamic_authority_set: Mapping[str, object] | None = None,
    stage_5_evidence: Mapping[str, object] | None = None,
    *,
    stage_5_issue_record: Mapping[str, object] | None = None,
    stage_5_consumed_marker_evidence: Mapping[str, object] | None = None,
    stage_5_checkpoint_1: Mapping[str, object] | None = None,
    stage_5_mount_launch_replay_rows: (
        Sequence[Mapping[str, object]] | None
    ) = None,
) -> dict[str, object]:
    if (
        type(payload) is not bytes
        or len(payload) > ACTUAL_FRESH_RUNTIME_CHECKPOINT_MAX_BYTES
    ):
        _fail("fresh runtime checkpoint exceeds frozen byte limit")
    value = _strict_json_object_v1(payload, "fresh runtime checkpoint")
    observed = value.get("observed_fresh_runtime_evidence")
    if type(observed) is not dict:
        _fail("fresh runtime checkpoint observed set differs")
    expected = build_fresh_runtime_checkpoint_v1(
        source_commit,
        artifact_path,
        checkpoint_id,
        attempt_id,
        boundary_root,
        issue_record_root,
        consumed_marker_root,
        issue_fresh_runtime_evidence,
        observed,
        artifact_absence_evidence,
        mount_binding_rows,
        dynamic_authority_set,
        stage_5_evidence,
        stage_5_issue_record=stage_5_issue_record,
        stage_5_consumed_marker_evidence=stage_5_consumed_marker_evidence,
        stage_5_checkpoint_1=stage_5_checkpoint_1,
        stage_5_mount_launch_replay_rows=stage_5_mount_launch_replay_rows,
    )
    if value != expected or canonical_json_bytes_v1(expected) != payload:
        _fail("fresh runtime checkpoint replay differs")
    return value


_PREIMAGE_KEY_SETS: Final = (
    {"stage_1_root", "requested_source_commit", "fresh_head_commit", "clean", "porcelain_line_count", "git_source_transcript"},
    {"stage_1_root", "config_relative_path", "commit_a_config_hex", "runtime_loaded_config_hex", "config_length", "config_sha256"},
    {"stage_1_root", "engineering_status", "actual_preconditions", "entrypoint", "entrypoint_implemented", "conditional_single_attempt_policy"},
    {"stage_1_root", "stage_3_root", "artifact_absence_evidence"},
    {"stage_1_root", "image_rows", "fresh_runtime_evidence_root"},
    {"stage_1_root", "stage_2_root", "actor_rows", "fresh_runtime_evidence_root"},
    {
        "stage_2_root", "stage_3_root", "cargo_lock_sha256",
        "cargo_material_identity", "cargo_material_identity_root",
        "cargo_snapshot_evidence", "cargo_snapshot_evidence_root",
        "cargo_tree_evidence", "cargo_tree_evidence_root",
        "offline_build_identity", "offline_build_identity_root",
        "fresh_runtime_evidence_root",
    },
    {
        "stage_2_root", "stage_3_root", "seccomp_rows", "binary_identity",
        "binary_identity_root", "fresh_runtime_evidence_root",
    },
    {"stage_1_root", "planned_command_registry_sha256", "command_mount_resource_policy_sha256", "prelaunch_policy_bound"},
    {"stage_1_root", "qualification_authority", "closed_q1_authority"},
    {"prior_stage_root_rows", "policy_name", "policy_bound_at_admission", "fulfilled_at_admission"},
    {"stage_1_root", "artifact_path", "policy_name", "policy_bound_at_admission", "fulfilled_at_admission"},
)


def _validate_preimage_v1(
    predicate_id: int,
    value: object,
    source_commit: str,
    commit_a_config_bytes: bytes,
    commit_a_config: Mapping[str, object],
    artifact_path: str,
    prior_stage_root_rows: list[list[object]],
) -> dict[str, object]:
    if type(value) is not dict or set(value) != _PREIMAGE_KEY_SETS[predicate_id - 1]:
        _fail(f"admission precondition {predicate_id} fields differ")
    roots = dict(prior_stage_root_rows)
    root_fields = {
        1: ("stage_1_root",),
        2: ("stage_1_root",),
        3: ("stage_1_root",),
        4: ("stage_1_root", "stage_3_root"),
        5: ("stage_1_root",),
        6: ("stage_1_root", "stage_2_root"),
        7: ("stage_2_root", "stage_3_root"),
        8: ("stage_2_root", "stage_3_root"),
        9: ("stage_1_root",),
        10: ("stage_1_root",),
        11: (),
        12: ("stage_1_root",),
    }[predicate_id]
    for field in root_fields:
        expected_stage = int(field.split("_")[1])
        if value[field] != roots[expected_stage]:
            _fail(f"admission precondition {predicate_id} stage root differs")
    if predicate_id == 1:
        if (
            value["requested_source_commit"] != source_commit
            or value["fresh_head_commit"] != source_commit
            or value["clean"] is not True
            or type(value["porcelain_line_count"]) is not int
            or value["porcelain_line_count"] != 0
        ):
            _fail("clean-head admission preimage differs")
        validate_git_source_transcript_v1(
            value["git_source_transcript"], source_commit
        )
    elif predicate_id == 2:
        commit_hex = commit_a_config_bytes.hex()
        if (
            value["config_relative_path"]
            != "config/phase3_q05b_dual_isolation_v1.json"
            or value["commit_a_config_hex"] != commit_hex
            or value["runtime_loaded_config_hex"] != commit_hex
            or type(value["config_length"]) is not int
            or value["config_length"] != len(commit_a_config_bytes)
            or value["config_sha256"]
            != sha256(commit_a_config_bytes).hexdigest()
        ):
            _fail("Commit-A config admission preimage differs")
    elif predicate_id == 3:
        if (
            value["engineering_status"] != COMMIT_A_ACTUAL_ENGINEERING_STATUS
            or value["entrypoint"] != "run_actual_v1"
            or value["entrypoint_implemented"] is not True
            or value["conditional_single_attempt_policy"]
            != "CONDITIONAL_SINGLE_ATTEMPT_RUNTIME_ADMISSION"
        ):
            _fail("implementation-policy admission preimage differs")
        _require_type_exact_v1(
            value["actual_preconditions"],
            COMMIT_A_ACTUAL_PRECONDITIONS_V1,
            "Commit-A actual preconditions",
        )
    elif predicate_id == 4:
        validate_artifact_absence_evidence_v1(
            value["artifact_absence_evidence"], artifact_path
        )
    elif predicate_id == 5:
        rows = value["image_rows"]
        if type(rows) is not list or len(rows) != 2:
            _fail("pinned image admission rows differ")
        for row, label, reference in zip(
            rows,
            ("python", "rust"),
            (EXPECTED_PYTHON_IMAGE_REFERENCE, EXPECTED_RUST_IMAGE_REFERENCE),
            strict=True,
        ):
            if (
                type(row) is not dict
                or set(row) != {"label", "reference", "evidence", "evidence_root"}
                or row["label"] != label
                or row["reference"] != reference
            ):
                _fail("pinned image admission row differs")
            evidence = _validate_pinned_image_evidence_v1(
                row["evidence"], reference
            )
            if row["evidence_root"] != fresh_runtime_evidence_object_root_v1(
                "PINNED_IMAGE", label, evidence
            ):
                _fail("pinned image admission evidence root differs")
        _root_hex_v1(
            value["fresh_runtime_evidence_root"], "fresh runtime evidence"
        )
    elif predicate_id == 6:
        rows = value["actor_rows"]
        labels = ("PYTHON_ENDPOINT", "RUST_ENDPOINT", "TRUSTED_HOST_REPLAY")
        if type(rows) is not list or len(rows) != 3:
            _fail("actor source/snapshot admission rows differ")
        for row, label in zip(rows, labels, strict=True):
            if (
                type(row) is not dict
                or set(row)
                != {
                    "actor_id", "source_identity", "source_identity_root",
                    "snapshot_evidence", "snapshot_evidence_root",
                }
                or row["actor_id"] != label
            ):
                _fail("actor source/snapshot admission row differs")
            source = _validate_actor_source_identity_v1(
                row["source_identity"], label, source_commit
            )
            snapshot = _validate_sealed_tree_evidence_v1(
                row["snapshot_evidence"], f"{label} snapshot"
            )
            if (
                row["source_identity_root"]
                != fresh_runtime_evidence_object_root_v1(
                    "ACTOR_SOURCE", label, source
                )
                or row["snapshot_evidence_root"]
                != fresh_runtime_evidence_object_root_v1(
                    "ACTOR_SNAPSHOT", label, snapshot
                )
            ):
                _fail("actor source/snapshot admission root differs")
        _root_hex_v1(
            value["fresh_runtime_evidence_root"], "fresh runtime evidence"
        )
    elif predicate_id == 7:
        _root_hex_v1(value["cargo_lock_sha256"], "Cargo.lock")
        material = _validate_cargo_material_identity_v1(
            value["cargo_material_identity"]
        )
        snapshot = _validate_sealed_snapshot_evidence_v1(
            value["cargo_snapshot_evidence"], "admission Cargo snapshot"
        )
        tree = _validate_sealed_tree_evidence_v1(
            value["cargo_tree_evidence"], "admission Cargo tree"
        )
        offline = value["offline_build_identity"]
        if (
            type(offline) is not dict
            or set(offline)
            != {
                "schema_version", "stage_3_root", "rust_test_transcript_sha256",
                "rust_release_build_transcript_sha256",
                "rust_snapshot_manifest_sha256",
                "cargo_snapshot_manifest_sha256", "cargo_tree_manifest_sha256",
                "binary_manifest_sha256", "stage_3_evidence_sha256",
            }
            or offline["schema_version"]
            != "hegel-phase3a-q05b-fresh-offline-build-identity/1"
            or offline["stage_3_root"] != value["stage_3_root"]
            or any(
                re.fullmatch(r"[0-9a-f]{64}", offline[field]) is None
                if type(offline[field]) is str else True
                for field in set(offline) - {"schema_version"}
            )
            or value["cargo_material_identity_root"]
            != fresh_runtime_evidence_object_root_v1(
                "CARGO_MATERIAL", "cargo-home", material
            )
            or value["cargo_snapshot_evidence_root"]
            != fresh_runtime_evidence_object_root_v1(
                "CARGO_SNAPSHOT", "cargo-home", snapshot
            )
            or value["cargo_tree_evidence_root"]
            != fresh_runtime_evidence_object_root_v1(
                "CARGO_TREE", "cargo-home", tree
            )
            or value["offline_build_identity_root"]
            != fresh_runtime_evidence_object_root_v1(
                "OFFLINE_BUILD_TRANSCRIPT", "rust", offline
            )
            or material["sealed_snapshot_manifest_sha256"]
            != snapshot["manifest_sha256"]
            or material["sealed_tree_manifest_sha256"]
            != tree["manifest_sha256"]
            or offline["cargo_snapshot_manifest_sha256"]
            != snapshot["manifest_sha256"]
            or offline["cargo_tree_manifest_sha256"]
            != tree["manifest_sha256"]
        ):
            _fail("Cargo/offline-build admission evidence differs")
        _root_hex_v1(
            value["fresh_runtime_evidence_root"], "fresh runtime evidence"
        )
    elif predicate_id == 8:
        rows = value["seccomp_rows"]
        if type(rows) is not list or len(rows) != 2:
            _fail("runtime seccomp admission rows differ")
        seccomp_config = commit_a_config["seccomp"]
        if type(seccomp_config) is not dict:
            _fail("Commit-A seccomp policy differs")
        for row, label, relative, expected_payload_sha256 in zip(
            rows,
            ("runtime", "build"),
            (
                "config/phase3_internal_actor_seccomp_v1.json",
                "config/phase3_m3_offline_build_seccomp_v1.json",
            ),
            (
                seccomp_config.get("runtime_profile_sha256"),
                seccomp_config.get("build_profile_sha256"),
            ),
            strict=True,
        ):
            if (
                type(row) is not dict
                or set(row) != {"label", "relative_path", "evidence", "evidence_root"}
                or row["label"] != label
                or row["relative_path"] != relative
            ):
                _fail("runtime seccomp admission row differs")
            evidence = _validate_seccomp_evidence_v1(row["evidence"], relative)
            if (
                type(expected_payload_sha256) is not str
                or re.fullmatch(r"[0-9a-f]{64}", expected_payload_sha256)
                is None
                or evidence["payload_sha256"] != expected_payload_sha256
            ):
                _fail(
                    "runtime seccomp admission payload differs from Commit-A config"
                )
            if row["evidence_root"] != fresh_runtime_evidence_object_root_v1(
                "SECCOMP_POLICY", label, evidence
            ):
                _fail("runtime seccomp admission root differs")
        binary = _validate_binary_identity_v1(value["binary_identity"])
        if value["binary_identity_root"] != fresh_runtime_evidence_object_root_v1(
            "RUST_BINARY", "runtime", binary
        ):
            _fail("runtime binary admission root differs")
        _root_hex_v1(
            value["fresh_runtime_evidence_root"], "fresh runtime evidence"
        )
    elif predicate_id == 9:
        _root_hex_v1(value["planned_command_registry_sha256"], "planned command")
        if (
            _root_hex_v1(
                value["command_mount_resource_policy_sha256"],
                "prelaunch policy",
            )
            != command_mount_resource_policy_root_v1(commit_a_config_bytes)
        ):
            _fail("prelaunch command/mount/resource policy root differs")
        if value["prelaunch_policy_bound"] is not True:
            _fail("prelaunch policy admission preimage differs")
    elif predicate_id == 10:
        _require_type_exact_v1(
            value["qualification_authority"],
            ACTUAL_ADMISSION_QUALIFICATION_AUTHORITY,
            "admission qualification authority",
        )
        _require_type_exact_v1(
            value["closed_q1_authority"],
            ACTUAL_ADMISSION_CLOSED_Q1_AUTHORITY,
            "admission closed Q1 authority",
        )
    elif predicate_id == 11:
        if (
            validate_prior_stage_root_rows_v1(value["prior_stage_root_rows"])
            != prior_stage_root_rows
            or value["policy_name"]
            != "FRESH_SOURCE_IMAGE_RUNTIME_SNAPSHOT_REPLAY_BEFORE_PREDICATE19"
            or value["policy_bound_at_admission"] is not True
            or value["fulfilled_at_admission"] is not False
        ):
            _fail("TOCTOU policy admission preimage differs")
    else:
        if (
            value["artifact_path"] != artifact_path
            or value["policy_name"]
            != "DIRFD_NOFOLLOW_FSYNC_LINK_NOREPLACE_UNLINK_FSYNC"
            or value["policy_bound_at_admission"] is not True
            or value["fulfilled_at_admission"] is not False
        ):
            _fail("publication policy admission preimage differs")
    return value


def build_actual_precondition_bundle_v1(
    source_commit: str,
    commit_a_config_bytes: bytes,
    artifact_path: str,
    work_root_identity: Mapping[str, object],
    prior_stage_root_rows: Sequence[Sequence[object]],
    ordered_precondition_preimages: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    commit = _commit_v1(source_commit, "admission source")
    commit_a_config = _validate_config_bytes_v1(commit_a_config_bytes)
    path = _absolute_path_text_v1(artifact_path, "artifact")
    work = validate_work_root_identity_v1(work_root_identity)
    stage_rows = validate_prior_stage_root_rows_v1(
        [list(row) for row in prior_stage_root_rows]
        if type(prior_stage_root_rows) in (tuple, list)
        else prior_stage_root_rows
    )
    if (
        type(ordered_precondition_preimages) not in (tuple, list)
        or len(ordered_precondition_preimages) != 12
    ):
        _fail("ordered admission preimages differ")
    rows: list[dict[str, object]] = []
    for (predicate_id, predicate_name), schema, raw_preimage in zip(
        ACTUAL_RUNTIME_PRECONDITION_REGISTRY,
        ACTUAL_PRECONDITION_PREIMAGE_SCHEMAS,
        ordered_precondition_preimages,
        strict=True,
    ):
        preimage = _validate_preimage_v1(
            predicate_id,
            raw_preimage,
            commit,
            commit_a_config_bytes,
            commit_a_config,
            path,
            stage_rows,
        )
        evidence_root = sha256(
            ACTUAL_PRECONDITION_EVIDENCE_ROOT_DOMAIN
            + predicate_id.to_bytes(2, "big")
            + canonical_json_bytes_v1([schema, preimage])
        ).hexdigest()
        rows.append(
            {
                "predicate_id": predicate_id,
                "predicate_name": predicate_name.decode("ascii"),
                "passed": True,
                "preimage_schema": schema,
                "preimage": preimage,
                "evidence_root": evidence_root,
            }
        )
    row5 = rows[4]["preimage"]
    row6 = rows[5]["preimage"]
    row7 = rows[6]["preimage"]
    row8 = rows[7]["preimage"]
    assert all(type(item) is dict for item in (row5, row6, row7, row8))
    fresh_set = build_fresh_runtime_evidence_set_v1(
        commit,
        row5["image_rows"],
        row6["actor_rows"],
        row7["cargo_material_identity"],
        row7["cargo_snapshot_evidence"],
        row7["cargo_tree_evidence"],
        row8["seccomp_rows"],
        row8["binary_identity"],
    )
    fresh_root = fresh_set["fresh_runtime_evidence_root"]
    if any(
        row["fresh_runtime_evidence_root"] != fresh_root
        for row in (row5, row6, row7, row8)
    ):
        _fail("rows 5 through 8 fresh runtime root differs")
    rust_actor_snapshot = row6["actor_rows"][1]["snapshot_evidence"]
    offline = row7["offline_build_identity"]
    if (
        offline["rust_snapshot_manifest_sha256"]
        != rust_actor_snapshot["manifest_sha256"]
        or offline["binary_manifest_sha256"]
        != row8["binary_identity"]["sealed_binary_manifest_sha256"]
    ):
        _fail("offline build differs from fresh actor/runtime evidence")
    body: dict[str, object] = {
        "schema_version": ACTUAL_PRECONDITION_BUNDLE_SCHEMA_VERSION,
        "source_commit": commit,
        "commit_a_config_length": len(commit_a_config_bytes),
        "commit_a_config_sha256": sha256(commit_a_config_bytes).hexdigest(),
        "artifact_path": path,
        "work_root_identity": dict(work),
        "prior_stage_root_rows": stage_rows,
        "ordered_precondition_rows": rows,
        "precondition_count": 12,
        "precondition_mask": 0xFFF,
        "precondition_registry_root": actual_precondition_registry_root_v1(),
    }
    value = dict(body)
    value["bundle_root"] = sha256(
        ACTUAL_PRECONDITION_BUNDLE_ROOT_DOMAIN + canonical_json_bytes_v1(body)
    ).hexdigest()
    if len(canonical_json_bytes_v1(value)) > ACTUAL_PRECONDITION_BUNDLE_MAX_BYTES:
        _fail("actual precondition bundle exceeds frozen byte limit")
    return value


def decode_actual_precondition_bundle_v1(
    payload: bytes,
    source_commit: str,
    commit_a_config_bytes: bytes,
    artifact_path: str,
    work_root_identity: Mapping[str, object],
    prior_stage_root_rows: Sequence[Sequence[object]],
    ordered_precondition_preimages: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if type(payload) is not bytes or len(payload) > ACTUAL_PRECONDITION_BUNDLE_MAX_BYTES:
        _fail("actual precondition bundle exceeds frozen byte limit")
    value = _strict_json_object_v1(payload, "actual precondition bundle")
    expected = build_actual_precondition_bundle_v1(
        source_commit,
        commit_a_config_bytes,
        artifact_path,
        work_root_identity,
        prior_stage_root_rows,
        ordered_precondition_preimages,
    )
    if value != expected or canonical_json_bytes_v1(expected) != payload:
        _fail("actual precondition bundle replay differs")
    return value


def validate_actual_precondition_bundle_object_v1(
    value: object,
    source_commit: str,
    commit_a_config_bytes: bytes,
    artifact_path: str,
) -> dict[str, object]:
    """Strictly replay a self-contained bundle before any downstream use."""

    expected_keys = {
        "schema_version",
        "source_commit",
        "commit_a_config_length",
        "commit_a_config_sha256",
        "artifact_path",
        "work_root_identity",
        "prior_stage_root_rows",
        "ordered_precondition_rows",
        "precondition_count",
        "precondition_mask",
        "precondition_registry_root",
        "bundle_root",
    }
    if type(value) is not dict or set(value) != expected_keys:
        _fail("self-contained admission bundle fields differ")
    rows = value["ordered_precondition_rows"]
    if type(rows) is not list or len(rows) != 12:
        _fail("self-contained admission predicate rows differ")
    preimages: list[Mapping[str, object]] = []
    row_keys = {
        "predicate_id",
        "predicate_name",
        "passed",
        "preimage_schema",
        "preimage",
        "evidence_root",
    }
    for row, (expected_id, expected_name), expected_schema in zip(
        rows,
        ACTUAL_RUNTIME_PRECONDITION_REGISTRY,
        ACTUAL_PRECONDITION_PREIMAGE_SCHEMAS,
        strict=True,
    ):
        if (
            type(row) is not dict
            or set(row) != row_keys
            or type(row["predicate_id"]) is not int
            or row["predicate_id"] != expected_id
            or type(row["predicate_name"]) is not str
            or row["predicate_name"] != expected_name.decode("ascii")
            or row["passed"] is not True
            or type(row["preimage_schema"]) is not str
            or row["preimage_schema"] != expected_schema
            or type(row["preimage"]) is not dict
        ):
            _fail("self-contained admission predicate row differs")
        _root_hex_v1(row["evidence_root"], "admission predicate evidence")
        preimages.append(row["preimage"])
    expected = build_actual_precondition_bundle_v1(
        source_commit,
        commit_a_config_bytes,
        artifact_path,
        value["work_root_identity"],
        value["prior_stage_root_rows"],
        preimages,
    )
    if canonical_json_bytes_v1(value) != canonical_json_bytes_v1(expected):
        _fail("self-contained admission bundle replay differs")
    return value


def build_actual_admission_decision_v1(
    source_commit: str,
    commit_a_config_bytes: bytes,
    artifact_path: str,
    attempt_nonce: bytes,
    precondition_bundle: Mapping[str, object],
) -> dict[str, object]:
    commit = _commit_v1(source_commit, "admission source")
    path = _absolute_path_text_v1(artifact_path, "artifact")
    _validate_config_bytes_v1(commit_a_config_bytes)
    if type(attempt_nonce) is not bytes or len(attempt_nonce) != 32:
        _fail("admission attempt nonce differs")
    bundle = validate_actual_precondition_bundle_object_v1(
        precondition_bundle,
        commit,
        commit_a_config_bytes,
        path,
    )
    config_root = sha256(commit_a_config_bytes).hexdigest()
    if (
        bundle.get("source_commit") != commit
        or bundle.get("artifact_path") != path
        or bundle.get("commit_a_config_length")
        != len(commit_a_config_bytes)
        or bundle.get("commit_a_config_sha256") != config_root
    ):
        _fail("admission bundle identity differs")
    bundle_root = _root_hex_v1(
        bundle.get("bundle_root"), "precondition bundle"
    )
    prior_roots = validate_prior_stage_root_rows_v1(
        bundle.get("prior_stage_root_rows")
    )
    work = validate_work_root_identity_v1(
        bundle.get("work_root_identity")
    )
    attempt_preimage = [
        commit, config_root, path, attempt_nonce.hex(), bundle_root,
        prior_roots, work,
    ]
    attempt_id = sha256(
        ACTUAL_ADMISSION_ATTEMPT_ID_DOMAIN
        + canonical_json_bytes_v1(attempt_preimage)
    ).hexdigest()
    body: dict[str, object] = {
        "schema_version": ACTUAL_ADMISSION_SCHEMA_VERSION,
        "source_commit": commit,
        "decision": ACTUAL_ADMISSION_DECISION_ID,
        "artifact_path": path,
        "attempt_nonce_hex": attempt_nonce.hex(),
        "attempt_id": attempt_id,
        "commit_a_config_length": len(commit_a_config_bytes),
        "commit_a_config_sha256": config_root,
        "precondition_bundle_root": bundle_root,
        "prior_stage_root_rows": prior_roots,
        "work_root_identity": dict(work),
        "precondition_count": 12,
        "precondition_mask": 0xFFF,
        "precondition_registry_root": actual_precondition_registry_root_v1(),
        "qualification_authority_at_admission": dict(
            ACTUAL_ADMISSION_QUALIFICATION_AUTHORITY
        ),
        "closed_q1_authority": {
            **ACTUAL_ADMISSION_CLOSED_Q1_AUTHORITY,
            "formal_output_roots": [None] * 8,
        },
    }
    value = dict(body)
    value["decision_root"] = sha256(
        ACTUAL_ADMISSION_DECISION_ROOT_DOMAIN + canonical_json_bytes_v1(body)
    ).hexdigest()
    return value


def decode_actual_admission_decision_v1(
    payload: bytes,
    commit_a_config_bytes: bytes,
    expected_source_commit: str,
    expected_artifact_path: str,
    expected_precondition_bundle: Mapping[str, object],
) -> dict[str, object]:
    value = _strict_json_object_v1(payload, "actual admission decision")
    nonce = _hex_bytes_v1(value.get("attempt_nonce_hex"), "attempt nonce")
    if len(nonce) != 32:
        _fail("admission attempt nonce differs")
    expected = build_actual_admission_decision_v1(
        expected_source_commit,
        commit_a_config_bytes,
        expected_artifact_path,
        nonce,
        expected_precondition_bundle,
    )
    if value != expected or canonical_json_bytes_v1(expected) != payload:
        _fail("admission decision replay differs")
    return value


def canonical_actual_admission_decision_bytes_v1(
    value: object,
    commit_a_config_bytes: bytes,
    expected_source_commit: str,
    expected_artifact_path: str,
    expected_precondition_bundle: Mapping[str, object],
) -> bytes:
    if type(value) is not dict:
        _fail("admission decision object differs")
    payload = canonical_json_bytes_v1(value)
    decode_actual_admission_decision_v1(
        payload,
        commit_a_config_bytes,
        expected_source_commit,
        expected_artifact_path,
        expected_precondition_bundle,
    )
    return payload


def build_stage3_to4_admission_boundary_v1(
    source_commit: str,
    commit_a_config_bytes: bytes,
    artifact_path: str,
    precondition_bundle: Mapping[str, object],
    decision: Mapping[str, object],
) -> dict[str, object]:
    if type(precondition_bundle) is not dict or type(decision) is not dict:
        _fail("admission boundary inputs differ")
    decision_payload = canonical_actual_admission_decision_bytes_v1(
        decision,
        commit_a_config_bytes,
        source_commit,
        artifact_path,
        precondition_bundle,
    )
    bundle_payload = canonical_json_bytes_v1(precondition_bundle)
    body: dict[str, object] = {
        "schema_version": ACTUAL_ADMISSION_BOUNDARY_SCHEMA_VERSION,
        "source_commit": source_commit,
        "artifact_path": artifact_path,
        "attempt_id": decision["attempt_id"],
        "prior_stage_root_rows": decision["prior_stage_root_rows"],
        "work_root_identity": decision["work_root_identity"],
        "precondition_bundle_hex": bundle_payload.hex(),
        "precondition_bundle_root": precondition_bundle["bundle_root"],
        "decision_hex": decision_payload.hex(),
        "decision_root": decision["decision_root"],
        "qualification_authority_at_boundary": dict(
            ACTUAL_ADMISSION_QUALIFICATION_AUTHORITY
        ),
        "closed_q1_authority": {
            **ACTUAL_ADMISSION_CLOSED_Q1_AUTHORITY,
            "formal_output_roots": [None] * 8,
        },
    }
    value = dict(body)
    value["boundary_root"] = sha256(
        ACTUAL_ADMISSION_BOUNDARY_ROOT_DOMAIN + canonical_json_bytes_v1(body)
    ).hexdigest()
    if len(canonical_json_bytes_v1(value)) > ACTUAL_ADMISSION_BOUNDARY_MAX_BYTES:
        _fail("actual admission boundary exceeds frozen byte limit")
    return value


def decode_stage3_to4_admission_boundary_v1(
    payload: bytes,
    source_commit: str,
    commit_a_config_bytes: bytes,
    artifact_path: str,
    expected_precondition_bundle: Mapping[str, object],
    expected_decision: Mapping[str, object],
) -> dict[str, object]:
    if type(payload) is not bytes or len(payload) > ACTUAL_ADMISSION_BOUNDARY_MAX_BYTES:
        _fail("actual admission boundary exceeds frozen byte limit")
    value = _strict_json_object_v1(payload, "stage3-to4 admission boundary")
    bundle_payload = _hex_bytes_v1(
        value.get("precondition_bundle_hex"), "precondition bundle"
    )
    decision_payload = _hex_bytes_v1(value.get("decision_hex"), "decision")
    if bundle_payload != canonical_json_bytes_v1(expected_precondition_bundle):
        _fail("admission boundary bundle bytes differ")
    decoded_decision = decode_actual_admission_decision_v1(
        decision_payload,
        commit_a_config_bytes,
        source_commit,
        artifact_path,
        expected_precondition_bundle,
    )
    if decoded_decision != expected_decision:
        _fail("admission boundary decision differs")
    expected = build_stage3_to4_admission_boundary_v1(
        source_commit,
        commit_a_config_bytes,
        artifact_path,
        expected_precondition_bundle,
        expected_decision,
    )
    if value != expected or canonical_json_bytes_v1(expected) != payload:
        _fail("admission boundary replay differs")
    return value


def validate_stage3_to4_admission_boundary_surface_v1(
    value: object,
    source_commit: str,
    artifact_path: str,
    prior_stage_root_rows: Sequence[Sequence[object]],
) -> dict[str, object]:
    keys = {
        "schema_version", "source_commit", "artifact_path", "attempt_id",
        "prior_stage_root_rows", "work_root_identity",
        "precondition_bundle_hex", "precondition_bundle_root", "decision_hex",
        "decision_root", "qualification_authority_at_boundary",
        "closed_q1_authority", "boundary_root",
    }
    if type(value) is not dict or set(value) != keys:
        _fail("admission boundary surface fields differ")
    expected_roots = validate_prior_stage_root_rows_v1(
        [list(row) for row in prior_stage_root_rows]
        if type(prior_stage_root_rows) in (tuple, list)
        else prior_stage_root_rows
    )
    if (
        value["schema_version"] != ACTUAL_ADMISSION_BOUNDARY_SCHEMA_VERSION
        or value["source_commit"] != _commit_v1(source_commit, "boundary source")
        or value["artifact_path"]
        != _absolute_path_text_v1(artifact_path, "boundary artifact")
        or value["prior_stage_root_rows"] != expected_roots
    ):
        _fail("admission boundary surface identity differs")
    for field in (
        "attempt_id", "precondition_bundle_root", "decision_root", "boundary_root"
    ):
        _root_hex_v1(value[field], f"admission boundary {field}")
    validate_work_root_identity_v1(value["work_root_identity"])
    _hex_bytes_v1(value["precondition_bundle_hex"], "precondition bundle")
    _hex_bytes_v1(value["decision_hex"], "decision")
    _require_type_exact_v1(
        value["qualification_authority_at_boundary"],
        ACTUAL_ADMISSION_QUALIFICATION_AUTHORITY,
        "admission boundary qualification authority",
    )
    _require_type_exact_v1(
        value["closed_q1_authority"],
        ACTUAL_ADMISSION_CLOSED_Q1_AUTHORITY,
        "admission boundary closed Q1 authority",
    )
    body = dict(value)
    root = body.pop("boundary_root")
    if root != sha256(
        ACTUAL_ADMISSION_BOUNDARY_ROOT_DOMAIN + canonical_json_bytes_v1(body)
    ).hexdigest():
        _fail("admission boundary surface root differs")
    return value


__all__ = [
    "ACTUAL_ACTOR_MOUNT_AUTHORITY_ROOT_DOMAIN",
    "ACTUAL_ACTOR_MOUNT_AUTHORITY_REGISTRY",
    "ACTUAL_ACTOR_MOUNT_BINDING_ROOT_DOMAIN",
    "ACTUAL_ACTOR_MOUNT_BINDING_SCHEMA_VERSION",
    "ACTUAL_ACTOR_MOUNT_LAUNCH_REPLAY_ROOT_DOMAIN",
    "ACTUAL_ACTOR_MOUNT_LAUNCH_REPLAY_SCHEMA_VERSION",
    "ACTUAL_ACTOR_MOUNT_ROLE_REGISTRY",
    "ACTUAL_ACTOR_MOUNT_SOURCE_ROOT_DOMAIN",
    "ACTUAL_ACTOR_MOUNT_SOURCE_SCHEMA_VERSION",
    "ACTUAL_PRELAUNCH_WRITABLE_DIRECTORY_ROOT_DOMAIN",
    "ACTUAL_RUNTIME_SECCOMP_RELATIVE_PATH",
    "ACTUAL_ADMISSION_ATTEMPT_ID_DOMAIN",
    "ACTUAL_ADMISSION_BOUNDARY_ROOT_DOMAIN",
    "ACTUAL_ADMISSION_BOUNDARY_SCHEMA_VERSION",
    "ACTUAL_ADMISSION_BOUNDARY_MAX_BYTES",
    "ACTUAL_ADMISSION_CONSUMED_MARKER_ROOT_DOMAIN",
    "ACTUAL_ADMISSION_CONSUMED_MARKER_SCHEMA_VERSION",
    "ACTUAL_ADMISSION_CLOSED_Q1_AUTHORITY",
    "ACTUAL_ADMISSION_DECISION_ID",
    "ACTUAL_ADMISSION_DECISION_ROOT_DOMAIN",
    "ACTUAL_ADMISSION_ISSUED_MARKER_ROOT_DOMAIN",
    "ACTUAL_ADMISSION_ISSUED_MARKER_SCHEMA_VERSION",
    "ACTUAL_ADMISSION_ISSUE_RECORD_ROOT_DOMAIN",
    "ACTUAL_ADMISSION_ISSUE_RECORD_SCHEMA_VERSION",
    "ACTUAL_ADMISSION_LIVE_MARKER_REPLAY_ROOT_DOMAIN",
    "ACTUAL_ADMISSION_LIVE_MARKER_REPLAY_SCHEMA_VERSION",
    "ACTUAL_ADMISSION_QUALIFICATION_AUTHORITY",
    "ACTUAL_ADMISSION_RUN_LOCAL_ANTI_REPLAY_SCOPE",
    "ACTUAL_ADMISSION_SCHEMA_VERSION",
    "ACTUAL_ADMISSION_SPENDING_INTENT_ROOT_DOMAIN",
    "ACTUAL_ADMISSION_SPENDING_INTENT_SCHEMA_VERSION",
    "DOCKER_AUTHORITATIVE_ABSENCE_SCHEMA_VERSION",
    "DOCKER_CONTAINER_NAME_USAGE",
    "DOCKER_DESTRUCTIVE_TARGET",
    "DOCKER_EXECUTION_AUTHORITY_ROOT_DOMAIN",
    "DOCKER_EXECUTION_AUTHORITY_SCHEMA_VERSION",
    "DOCKER_EXECUTION_SLOT_REGISTRY",
    "DOCKER_INITIAL_NAME_ABSENCE_ROOT_DOMAIN",
    "DOCKER_OWNERSHIP_NAMESPACE_DOMAIN",
    "DOCKER_PRECREATE_ABSENCE_ROOT_DOMAIN",
    "DOCKER_PRECREATE_ABSENCE_SCHEMA_VERSION",
    "DOCKER_RESERVED_LABEL_KEYS",
    "DOCKER_RUST_BASE_LABEL_ROWS",
    "ACTUAL_GIT_SOURCE_TRANSCRIPT_ROOT_DOMAIN",
    "ACTUAL_GIT_SOURCE_TRANSCRIPT_SCHEMA_VERSION",
    "ACTUAL_FRESH_RUNTIME_EVIDENCE_OBJECT_ROOT_DOMAIN",
    "ACTUAL_FRESH_RUNTIME_EVIDENCE_SET_ROOT_DOMAIN",
    "ACTUAL_FRESH_RUNTIME_EVIDENCE_SET_SCHEMA_VERSION",
    "ACTUAL_FRESH_RUNTIME_CHECKPOINT_MAX_BYTES",
    "ACTUAL_FRESH_RUNTIME_CHECKPOINT_REGISTRY",
    "ACTUAL_FRESH_RUNTIME_CHECKPOINT_ROOT_DOMAIN",
    "ACTUAL_FRESH_RUNTIME_CHECKPOINT_SCHEMA_VERSION",
    "ACTUAL_CHECKPOINT_MOUNT_REGISTRY_ROOT_DOMAIN",
    "ACTUAL_DYNAMIC_MOUNT_AUTHORITY_SET_ROOT_DOMAIN",
    "ACTUAL_DYNAMIC_MOUNT_AUTHORITY_SET_SCHEMA_VERSION",
    "ACTUAL_STAGE_EVIDENCE_ROOT_DOMAIN",
    "ACTUAL_STAGE_SCHEMA_VERSION",
    "ACTUAL_STAGE_5_NAME",
    "ACTUAL_STAGE_5_BASE_EVIDENCE_KEYS",
    "ACTUAL_STAGE_5_INJECTED_EVIDENCE_KEYS",
    "ACTUAL_PRECONDITION_BUNDLE_MAX_BYTES",
    "ACTUAL_PRECONDITION_BUNDLE_ROOT_DOMAIN",
    "ACTUAL_PRECONDITION_BUNDLE_SCHEMA_VERSION",
    "ACTUAL_PRECONDITION_EVIDENCE_ROOT_DOMAIN",
    "ACTUAL_PRECONDITION_PREIMAGE_SCHEMAS",
    "ACTUAL_PRECONDITION_REGISTRY_ROOT_DOMAIN",
    "ACTUAL_RUNTIME_PRECONDITION_REGISTRY",
    "COMMIT_A_ACTUAL_ENGINEERING_STATUS",
    "COMMIT_A_ACTUAL_PRECONDITIONS_V1",
    "ACTUAL_COMMAND_MOUNT_RESOURCE_POLICY_ROOT_DOMAIN",
    "ACTUAL_COMMIT_A_STATIC_POLICY_ROOT_DOMAIN",
    "COMMAND_MOUNT_RESOURCE_POLICY_FIELDS",
    "EXPECTED_COMMAND_MOUNT_RESOURCE_POLICY_ROOT",
    "EXPECTED_COMMIT_A_STATIC_POLICY_ROOT",
    "Q05BActualAdmissionError",
    "actual_precondition_registry_root_v1",
    "actor_mount_authority_root_v1",
    "build_actual_admission_decision_v1",
    "build_actual_admission_consumed_marker_evidence_v1",
    "build_actual_admission_issue_record_v1",
    "build_actual_admission_issued_marker_evidence_v1",
    "build_actual_admission_live_marker_replay_v1",
    "build_actual_admission_spending_intent_v1",
    "build_actual_stage_5_evidence_v1",
    "build_actual_precondition_bundle_v1",
    "build_actor_mount_binding_v1",
    "build_actor_mount_launch_replay_v1",
    "build_actor_mount_source_row_v1",
    "build_dynamic_mount_authority_set_v1",
    "build_docker_execution_authority_v1",
    "build_docker_initial_name_absence_row_v1",
    "build_docker_precreate_absence_v1",
    "build_prelaunch_writable_directory_evidence_v1",
    "build_fresh_runtime_evidence_set_v1",
    "build_fresh_runtime_checkpoint_v1",
    "build_stage3_to4_admission_boundary_v1",
    "canonical_actual_admission_decision_bytes_v1",
    "canonical_json_bytes_v1",
    "command_mount_resource_policy_root_v1",
    "cross_docker_execution_authority_to_admission_decision_v1",
    "checkpoint_mount_registry_root_v1",
    "decode_actual_admission_decision_v1",
    "decode_actual_precondition_bundle_v1",
    "decode_stage3_to4_admission_boundary_v1",
    "decode_fresh_runtime_checkpoint_v1",
    "decode_dynamic_mount_authority_set_v1",
    "docker_execution_slot_rows_v1",
    "fresh_runtime_evidence_object_root_v1",
    "actual_admission_marker_names_v1",
    "validate_actual_admission_consumed_marker_evidence_v1",
    "validate_actual_admission_issue_record_v1",
    "validate_actual_admission_issued_marker_evidence_v1",
    "validate_actual_admission_live_marker_replay_surface_v1",
    "validate_actual_admission_spending_intent_v1",
    "validate_actual_stage_5_evidence_v1",
    "validate_actual_stage_5_evidence_surface_v1",
    "validate_artifact_absence_evidence_v1",
    "validate_commit_a_actual_config_bytes_v1",
    "validate_actor_mount_binding_v1",
    "validate_actor_mount_launch_replay_v1",
    "validate_dynamic_mount_authority_set_v1",
    "validate_docker_authoritative_absence_v1",
    "validate_docker_execution_authority_v1",
    "validate_docker_precreate_absence_v1",
    "validate_actual_precondition_bundle_object_v1",
    "validate_git_source_transcript_v1",
    "validate_fresh_runtime_evidence_set_v1",
    "validate_prior_stage_root_rows_v1",
    "validate_stage3_to4_admission_boundary_surface_v1",
    "validate_work_root_identity_v1",
]
