from __future__ import annotations

import ast
from collections.abc import Callable
from copy import deepcopy
from dataclasses import FrozenInstanceError
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import textwrap
from types import MappingProxyType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
FINAL_IMAGE_BUILD_RECEIPT = (
    ROOT / "artifacts" / "phase2b_recognizer_runtime_image_build_v3.json"
)
if SOURCE.as_posix() not in sys.path:
    sys.path.insert(0, SOURCE.as_posix())

import hegel_machine.phase2b_recognizer_host_executor_v3 as executor
from hegel_machine.hashing import stable_hash
from hegel_machine.phase2b_protocol import frozen_phase2b_protocol
from hegel_machine.phase2b_recognizer_input_archive_v2 import (
    DecodedRecognizerInputArchiveV2,
)
from hegel_machine.phase2b_recognizer_prediction_archive_v2 import (
    DecodedRecognizerPredictionArchiveV2,
    PREDICTION_ARCHIVE_HEADER_BYTES_V2,
)
from hegel_machine.phase2b_recognizer_runtime_cli_v2 import (
    EXECUTION_FREEZE_MANIFEST_FILENAME_V2,
    RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2,
    RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2,
    RECOGNIZER_RUNTIME_RECEIPT_SCHEMA_VERSION_V2,
)
from hegel_machine.phase2b_runner import (
    FROZEN_RECOGNIZER_ENTRYPOINT,
    OciRecognizerRunSpec,
    RECOGNIZER_RUNTIME_UID_GID,
    RECOGNIZER_SECCOMP_PROFILE_SHA256,
    RuntimeLimits,
    TOTAL_RECOGNIZER_CASE_COUNT,
)
from hegel_machine.phase3_local_runtime_v1 import LocalDockerControlPlaneV1


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
IMAGE_A = executor.RECOGNIZER_RUNTIME_IMAGE_DIGEST_V3


def _run_spec() -> OciRecognizerRunSpec:
    return OciRecognizerRunSpec(
        freeze_manifest_id="freeze_manifest_v1",
        protocol_id=frozen_phase2b_protocol().protocol_id,
        image_digest=IMAGE_A,
        input_host_directory="/sealed/phase2b/input",
        output_host_directory="/sealed/phase2b/output",
        repository_host_directory="/workspace/assumption-agent",
        seccomp_profile_host_path=(
            "/workspace/assumption-agent/"
            "config/phase3_internal_actor_seccomp_v1.json"
        ),
        seccomp_profile_sha256=RECOGNIZER_SECCOMP_PROFILE_SHA256,
        input_manifest_sha256=SHA_B,
        expected_case_count=TOTAL_RECOGNIZER_CASE_COUNT,
        entrypoint=FROZEN_RECOGNIZER_ENTRYPOINT,
        limits=RuntimeLimits(),
    )


def _image_inspect_value() -> list[dict[str, object]]:
    return [
        {
            "Id": IMAGE_A,
            "Os": "linux",
            "Architecture": "amd64",
            "Config": {
                "ArgsEscaped": True,
                "User": RECOGNIZER_RUNTIME_UID_GID,
                "WorkingDir": "/opt/hegel",
                "Entrypoint": [FROZEN_RECOGNIZER_ENTRYPOINT[0]],
                "Cmd": list(FROZEN_RECOGNIZER_ENTRYPOINT[1:]),
                "Env": list(executor._IMAGE_ENVIRONMENT_V3),
                "Labels": {
                    "org.opencontainers.image.title": executor._IMAGE_TITLE_V3,
                    "org.opencontainers.image.description": (
                        executor._IMAGE_DESCRIPTION_V3
                    ),
                    "org.opencontainers.image.base.digest": (
                        executor._IMAGE_BASE_DIGEST_V3
                    ),
                    "org.hegel-machine.phase2b.actor-profile.content-id": (
                        executor.PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
                    ),
                    "org.hegel-machine.phase2b.actor-profile.canonical-sha256": (
                        executor.PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
                    ),
                    "org.hegel-machine.phase2b.owner-accepted-authority-requirement.id": (
                        executor.OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V3
                    ),
                    "org.hegel-machine.phase2b.recognizer-source-root.sha256": (
                        executor.RECOGNIZER_SOURCE_ROOT_SHA256_V3
                    ),
                },
            },
            "RootFS": {"Type": "layers", "Layers": ["sha256:" + SHA_C]},
        }
    ]


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _local_receipt(**changes: object) -> executor.Phase2BRecognizerLocalExecutionReceiptV3:
    empty_sha = executor._sha256(b"")
    values: dict[str, object] = {
        "schema_version": executor.LOCAL_EXECUTION_RECEIPT_SCHEMA_V3,
        "disposition": executor.LOCAL_EXECUTION_DISPOSITION_V3,
        "reason": executor.LOCAL_EXECUTION_COMPLETE_REASON_V3,
        "run_spec_id": "phase2b_oci_run_spec_" + SHA_A,
        "freeze_manifest_id": "phase2b_execution_freeze_" + SHA_A,
        "protocol_id": "phase2b_protocol_" + SHA_A,
        "image_digest": IMAGE_A,
        "observed_image_id": IMAGE_A,
        "recognizer_source_root_sha256": executor.RECOGNIZER_SOURCE_ROOT_SHA256_V3,
        "actor_profile_content_id": (
            executor.PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
        ),
        "actor_profile_canonical_sha256": (
            executor.PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
        ),
        "owner_accepted_authority_requirement_id": (
            executor.OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V3
        ),
        "image_args_escaped": True,
        "image_inspect_sha256": SHA_A,
        "image_config_projection_sha256": SHA_A,
        "docker_control_plane_binding_sha256": SHA_A,
        "docker_executable_sha256": SHA_A,
        "seccomp_profile_sha256": SHA_A,
        "host_path_binding_sha256": SHA_A,
        "input_manifest_sha256": SHA_A,
        "input_archive_id": "phase2b_recognizer_input_archive_v2_" + SHA_A,
        "input_archive_sha256": SHA_A,
        "execution_freeze_manifest_sha256": SHA_A,
        "command_sha256": SHA_A,
        "container_supervisor_sha256": executor._sha256(
            (
                executor._CONTAINER_SUPERVISOR_ENTRYPOINT_V3
                + "\x00"
                + executor._CONTAINER_SUPERVISOR_CODE_V3
            ).encode("utf-8")
        ),
        "container_result_extractor_sha256": executor._sha256(
            (
                executor._CONTAINER_SUPERVISOR_ENTRYPOINT_V3
                + "\x00"
                + executor._CONTAINER_RESULT_EXTRACTOR_CODE_V3
            ).encode("utf-8")
        ),
        "process_exit_code": 0,
        "timed_out": False,
        "started_monotonic_ns": 1,
        "finished_monotonic_ns": 2,
        "stdout_bytes": 2,
        "stdout_sha256": SHA_A,
        "stderr_bytes": 0,
        "stderr_sha256": empty_sha,
        "runtime_receipt_id": "phase2b_recognizer_runtime_receipt_v2_" + SHA_A,
        "prediction_archive_id": (
            "phase2b_recognizer_prediction_archive_v2_" + SHA_A
        ),
        "prediction_archive_sha256": SHA_A,
        "prediction_archive_bytes": PREDICTION_ARCHIVE_HEADER_BYTES_V2,
        "prediction_case_count": TOTAL_RECOGNIZER_CASE_COUNT,
        "output_file_identity_sha256": SHA_A,
    }
    values.update(changes)
    return executor.Phase2BRecognizerLocalExecutionReceiptV3._issue(
        executor._RECEIPT_ISSUE_TOKEN_V3,
        **values,
    )


def _private_decoded(cls: type[object], **values: object) -> object:
    value = object.__new__(cls)
    for name, item in values.items():
        object.__setattr__(value, name, item)
    return value


def _runtime_receipt_fixture() -> tuple[
    DecodedRecognizerInputArchiveV2,
    DecodedRecognizerPredictionArchiveV2,
    dict[str, object],
]:
    input_archive = _private_decoded(
        DecodedRecognizerInputArchiveV2,
        archive_id="phase2b_recognizer_input_archive_v2_" + SHA_A,
    )
    prediction_archive = _private_decoded(
        DecodedRecognizerPredictionArchiveV2,
        archive_id="phase2b_recognizer_prediction_archive_v2_" + SHA_B,
    )
    receipt_without_id: dict[str, object] = {
        "schema_version": RECOGNIZER_RUNTIME_RECEIPT_SCHEMA_VERSION_V2,
        "disposition": "COMPLETE",
        "reason": "phase2b_recognizer_runtime_v2_complete",
        "input_archive_id": input_archive.archive_id,
        "input_archive_sha256": SHA_A,
        "execution_freeze_manifest_id": "freeze_manifest_v1",
        "execution_freeze_manifest_sha256": SHA_B,
        "prediction_archive_id": prediction_archive.archive_id,
        "prediction_archive_sha256": SHA_C,
        "prediction_case_count": TOTAL_RECOGNIZER_CASE_COUNT,
        "prediction_archive_bytes": PREDICTION_ARCHIVE_HEADER_BYTES_V2,
        "output_filename": RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2,
    }
    receipt = dict(receipt_without_id)
    receipt["receipt_id"] = stable_hash(
        receipt_without_id,
        prefix="phase2b_recognizer_runtime_receipt_v2_",
    )
    return input_archive, prediction_archive, receipt


def _validate_runtime_receipt(
    receipt: dict[str, object],
    input_archive: DecodedRecognizerInputArchiveV2,
    prediction_archive: DecodedRecognizerPredictionArchiveV2,
) -> str:
    return executor._validate_runtime_receipt_v3(
        executor._canonical_ascii_json_line_v3(receipt),
        input_archive=input_archive,
        input_archive_sha256=SHA_A,
        execution_manifest_id="freeze_manifest_v1",
        execution_manifest_sha256=SHA_B,
        prediction_archive=prediction_archive,
        prediction_archive_sha256=SHA_C,
        prediction_archive_bytes=PREDICTION_ARCHIVE_HEADER_BYTES_V2,
    )


def _control_plane() -> LocalDockerControlPlaneV1:
    return LocalDockerControlPlaneV1(
        executable=Path("/usr/bin/docker"),
        socket_path=Path("/var/run/docker.sock"),
        config_directory=Path("/tmp/hegel-docker-config"),
        environment=MappingProxyType({"PATH": "/usr/bin:/bin"}),
        binding=MappingProxyType({}),
    )


def _seccomp_value() -> dict[str, object]:
    return {
        "defaultAction": "SCMP_ACT_ALLOW",
        "syscalls": [
            {
                "names": ["mount"],
                "action": "SCMP_ACT_ERRNO",
                "errnoRet": 1,
            }
        ],
    }


def _created_container_inspect_value(
    *,
    container_id: str = SHA_C,
    container_name: str = "hegel-phase2b-recognizer-v3-" + "d" * 24,
    staged_input: Path = Path("/tmp/hegel-runtime/recognizer-input"),
) -> list[dict[str, object]]:
    spec = _run_spec()
    seccomp = json.dumps(
        _seccomp_value(),
        sort_keys=True,
        separators=(",", ":"),
    )
    return [
        {
            "Id": container_id,
            "Name": "/" + container_name,
            "Image": spec.image_digest,
            "Path": executor._CONTAINER_SUPERVISOR_ENTRYPOINT_V3,
            "Args": [
                "-c",
                executor._CONTAINER_SUPERVISOR_CODE_V3,
                *spec.entrypoint[1:],
            ],
            "State": {"Status": "created", "Running": False},
            "Config": {
                "Image": spec.image_digest,
                "User": RECOGNIZER_RUNTIME_UID_GID,
                "Env": list(executor._CONTAINER_ENVIRONMENT_V3),
                "Entrypoint": [executor._CONTAINER_SUPERVISOR_ENTRYPOINT_V3],
                "Cmd": [
                    "-c",
                    executor._CONTAINER_SUPERVISOR_CODE_V3,
                    *spec.entrypoint[1:],
                ],
                "WorkingDir": "/opt/hegel",
                "Healthcheck": {"Test": ["NONE"]},
                "Volumes": None,
            },
            "HostConfig": {
                "AutoRemove": False,
                "ReadonlyRootfs": True,
                "NetworkMode": "none",
                "IpcMode": "none",
                "CapDrop": ["ALL"],
                "SecurityOpt": [
                    "no-new-privileges:true",
                    "seccomp=" + seccomp,
                ],
                "Tmpfs": {
                    "/tmp": (
                        "rw,noexec,nosuid,nodev,size=268435456,"
                        "mode=0700,uid=65534,gid=65534"
                    ),
                    "/phase2b/output": (
                        "rw,noexec,nosuid,nodev,size=536870912,"
                        "mode=0700,uid=65534,gid=65534"
                    ),
                },
                "Ulimits": [
                    {"Name": "nofile", "Hard": 256, "Soft": 256},
                    {
                        "Name": "fsize",
                        "Hard": 536_870_912,
                        "Soft": 536_870_912,
                    },
                ],
                "Memory": 2_147_483_648,
                "MemorySwap": 2_147_483_648,
                "NanoCpus": 2_000_000_000,
                "PidsLimit": 128,
                "Privileged": False,
                "CgroupnsMode": "private",
                "PidMode": "",
                "UsernsMode": "",
                "Runtime": "runc",
            },
            "Mounts": [
                {
                    "Type": "bind",
                    "Source": staged_input.as_posix(),
                    "Destination": "/phase2b/input",
                    "Mode": "",
                    "RW": False,
                    "Propagation": "rprivate",
                }
            ],
            "NetworkSettings": {"Ports": {}},
        }
    ]


def _validate_created_container(
    value: object,
    *,
    staged_input: Path = Path("/tmp/hegel-runtime/recognizer-input"),
) -> None:
    executor._validate_created_container_inspect_v3(
        _json_bytes(value),
        container_id=SHA_C,
        container_name="hegel-phase2b-recognizer-v3-" + "d" * 24,
        spec=_run_spec(),
        staged_input_directory=staged_input,
        expected_seccomp=_seccomp_value(),
    )


def _result_wire(receipt: bytes, prediction: bytes) -> bytes:
    return (
        executor._CONTAINER_RESULT_WIRE_MAGIC_V3
        + len(receipt).to_bytes(4, "big")
        + len(prediction).to_bytes(4, "big")
        + receipt
        + prediction
    )


def _crossbinding_fixture() -> tuple[object, object]:
    row_ids = ("row-a", "row-b")
    input_archive = _private_decoded(
        DecodedRecognizerInputArchiveV2,
        archive_id="phase2b_recognizer_input_archive_v2_" + SHA_A,
        batch_id="phase2b_trusted_wire_batch_v2_" + SHA_B,
        batch_policy_id="batch-policy-v2",
        row_ids=row_ids,
    )
    context = SimpleNamespace(
        input_archive_id=input_archive.archive_id,
        input_archive_sha256=SHA_A,
        execution_freeze_manifest_id="phase2b_execution_freeze_" + SHA_B,
        protocol_id="phase2b_protocol_" + SHA_C,
        batch_id=input_archive.batch_id,
        batch_policy_id=input_archive.batch_policy_id,
    )
    prediction = _private_decoded(
        DecodedRecognizerPredictionArchiveV2,
        context=context,
        input_row_ids=row_ids,
        records=tuple(SimpleNamespace(input_row_id=item) for item in row_ids),
    )
    return input_archive, prediction


def test_public_surface_is_narrow_and_executor_parameters_are_keyword_safe() -> None:
    assert set(executor.__all__) == {
        "FAIL_DOCKER_CAPTURE",
        "FAIL_DOCKER_CONTROL_PLANE",
        "FAIL_DOCKER_EXECUTION",
        "FAIL_DOCKER_TIMEOUT",
        "FAIL_HOST_PATH",
        "FAIL_IMAGE_BINDING",
        "FAIL_INPUT_BINDING",
        "FAIL_OUTPUT_BINDING",
        "FAIL_PREFLIGHT",
        "FAIL_SECCOMP_BINDING",
        "IMAGE_INSPECT_TIMEOUT_SECONDS_V3",
        "LOCAL_EXECUTION_COMPLETE_REASON_V3",
        "LOCAL_EXECUTION_DISPOSITION_V3",
        "LOCAL_EXECUTION_RECEIPT_SCHEMA_V3",
        "MAXIMUM_DOCKER_STDERR_BYTES_V3",
        "MAXIMUM_DOCKER_STDOUT_BYTES_V3",
        "OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V3",
        "Phase2BRecognizerHostExecutorErrorV3",
        "Phase2BRecognizerLocalExecutionReceiptV3",
        "RECOGNIZER_RUNTIME_IMAGE_DIGEST_V3",
        "RECOGNIZER_SOURCE_ROOT_SHA256_V3",
        "RECOGNIZER_TIMEOUT_SECONDS_V3",
        "execute_phase2b_recognizer_docker_v3",
    }
    parameters = inspect.signature(
        executor.execute_phase2b_recognizer_docker_v3
    ).parameters
    assert tuple(parameters) == ("spec", "runtime_parent")
    assert parameters["spec"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["runtime_parent"].kind is inspect.Parameter.KEYWORD_ONLY
    source = Path(executor.__file__).read_text(encoding="utf-8")
    assert "phase2b_recognizer_host_executor_v1" not in source
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.Global) for node in ast.walk(tree))
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setattr"
        for node in ast.walk(tree)
    )
    assert "object.__setattr__" in source


def test_hard_gate_literals_match_final_non_authority_v3_image_receipt() -> None:
    receipt = json.loads(FINAL_IMAGE_BUILD_RECEIPT.read_bytes())
    assert receipt["image"]["image_digest"] == (
        "sha256:d27247e9b4b6fcbeca38aea81359b29a855c05f7a7a6e9f2c2276d360160054f"
    )
    assert receipt["image"]["image_digest"] == (
        executor.RECOGNIZER_RUNTIME_IMAGE_DIGEST_V3
    )
    assert receipt["build_inputs"]["source_root_sha256"] == (
        executor.RECOGNIZER_SOURCE_ROOT_SHA256_V3
    )
    assert receipt["build_inputs"]["profile_content_id"] == (
        executor.PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
    )
    assert receipt["build_inputs"]["profile_canonical_sha256"] == (
        executor.PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
    )
    assert receipt["build_inputs"]["owner_accepted_authority_requirement_id"] == (
        executor.OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V3
    )
    assert receipt["image"]["args_escaped"] is True
    assert receipt["image"]["entrypoint"] == [FROZEN_RECOGNIZER_ENTRYPOINT[0]]
    assert receipt["image"]["command"] == list(FROZEN_RECOGNIZER_ENTRYPOINT[1:])
    assert receipt["image"]["labels"] == _image_inspect_value()[0]["Config"][
        "Labels"
    ]
    assert receipt["claims"]["actor_profile_authority_verified"] is False
    assert receipt["claims"]["authority_requirement_satisfied"] is False
    assert receipt["claims"]["actual_960_case_run_verified"] is False
    assert receipt["claims"]["formal_gate_evaluation_performed"] is False
    assert receipt["claims"]["c1_exit_evidence"] is False


def test_profile_binding_import_cannot_regress_from_v2_to_v1() -> None:
    tree = ast.parse(Path(executor.__file__).read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "phase2b_c1_container_actor_profile_v2" in imported_modules
    assert "phase2b_c1_container_actor_profile_v1" not in imported_modules
    assert executor.PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID.endswith(
        "b2f081771f3de8d2a531b08b58109d95c51815904e9612e66f1bbfe69babf906"
    )


def test_local_receipt_exposes_only_non_authority_claims_and_is_frozen() -> None:
    with pytest.raises(TypeError, match="privately issued"):
        executor.Phase2BRecognizerLocalExecutionReceiptV3()
    with pytest.raises(TypeError, match="issuer token"):
        executor.Phase2BRecognizerLocalExecutionReceiptV3._issue(object())
    receipt = _local_receipt()
    expected_false = {
        "external_runtime_attestation_issued",
        "runtime_attestation_authority_verified",
        "actor_profile_authority_verified",
        "authority_requirement_satisfied",
        "recognizer_operation_signature_verified",
        "actual_input_archive_supplied",
        "actual_960_recognizer_executed",
        "actual_960_case_run_verified",
        "fresh_role_keys_generated",
        "key_generation_performed",
        "custodian_ceremony_executed",
        "signed_execution_registry_materialized",
        "signed_monotonic_time_materialized",
        "store_receipt_materialized",
        "prediction_commit_before_reveal_verified",
        "reveal_authority_verified",
        "scoring_performed",
        "prediction_scored",
        "formal_gate_evaluation_performed",
        "effect_evidence",
        "formal_c1_report_verified",
        "c1_exit_evidence",
        "malicious_host_root_resistance",
        "malicious_docker_daemon_resistance",
    }
    assert receipt.docker_invoked is True
    assert receipt.local_mechanics_complete is True
    assert isinstance(receipt.claim_boundary, MappingProxyType)
    assert set(receipt.claim_boundary) == expected_false
    assert not any(receipt.claim_boundary.values())
    assert all(getattr(receipt, name) is False for name in expected_false)
    assert receipt.image_digest == executor.RECOGNIZER_RUNTIME_IMAGE_DIGEST_V3
    assert (
        receipt.recognizer_source_root_sha256
        == executor.RECOGNIZER_SOURCE_ROOT_SHA256_V3
    )
    assert (
        receipt.actor_profile_content_id
        == executor.PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
    )
    assert (
        receipt.actor_profile_canonical_sha256
        == executor.PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
    )
    assert (
        receipt.owner_accepted_authority_requirement_id
        == executor.OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V3
    )
    assert receipt.image_args_escaped is True
    assert receipt.receipt_id.startswith(
        "phase2b_recognizer_local_execution_receipt_v3_"
    )
    with pytest.raises(TypeError):
        receipt.claim_boundary["actual_960_case_run_verified"] = True
    with pytest.raises(FrozenInstanceError):
        receipt.actual_960_case_run_verified = True


@pytest.mark.parametrize(
    "changes",
    (
        {"image_digest": "sha256:" + SHA_B},
        {"recognizer_source_root_sha256": SHA_B},
        {"actor_profile_content_id": "phase2b_c1_container_actor_profile_v2_" + SHA_B},
        {"actor_profile_canonical_sha256": SHA_B},
        {"owner_accepted_authority_requirement_id": "wrong"},
        {"image_args_escaped": False},
        {"process_exit_code": 1},
        {"timed_out": True},
        {"stderr_bytes": 1},
        {"prediction_case_count": TOTAL_RECOGNIZER_CASE_COUNT - 1},
        {"image_inspect_sha256": "A" * 64},
        {"container_supervisor_sha256": "A" * 64},
        {"container_result_extractor_sha256": "A" * 64},
    ),
)
def test_local_receipt_rejects_mechanics_and_digest_drift(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="receipt invariant|not SHA-256"):
        _local_receipt(**changes)


def test_image_inspection_binds_digest_platform_config_environment_and_labels() -> None:
    value = _image_inspect_value()
    payload = _json_bytes(value)
    image_id, projection = executor._validate_image_inspect_v3(
        payload,
        spec=_run_spec(),
    )
    assert image_id == IMAGE_A
    assert projection == stable_hash(value[0]["Config"])


def _mutate_image(value: list[dict[str, object]], mutation: str) -> None:
    row = value[0]
    config = row["Config"]
    assert isinstance(config, dict)
    if mutation == "digest":
        row["Id"] = "sha256:" + SHA_B
    elif mutation == "platform":
        row["Architecture"] = "arm64"
    elif mutation == "user":
        config["User"] = "0:0"
    elif mutation == "args_escaped":
        config["ArgsEscaped"] = False
    elif mutation == "missing_args_escaped":
        config.pop("ArgsEscaped")
    elif mutation == "extra_config":
        config["Healthcheck"] = None
    elif mutation == "working_directory":
        config["WorkingDir"] = "/"
    elif mutation == "entrypoint":
        config["Entrypoint"] = ["/bin/sh"]
    elif mutation == "command":
        config["Cmd"] = ["--help"]
    elif mutation == "rootfs":
        row["RootFS"] = {"Type": "layers", "Layers": []}
    elif mutation == "missing_environment":
        config["Env"] = [
            item for item in config["Env"] if not item.startswith("HOME=")
        ]
    elif mutation == "duplicate_environment":
        config["Env"] = [*config["Env"], "HOME=/tmp"]
    elif mutation == "reordered_environment":
        config["Env"] = list(reversed(config["Env"]))
    elif mutation == "label":
        config["Labels"]["org.opencontainers.image.title"] = "other"
    elif mutation == "source_root_label":
        config["Labels"][
            "org.hegel-machine.phase2b.recognizer-source-root.sha256"
        ] = SHA_B
    elif mutation == "profile_label":
        config["Labels"][
            "org.hegel-machine.phase2b.actor-profile.content-id"
        ] = "phase2b_c1_container_actor_profile_v2_" + SHA_B
    elif mutation == "authority_label":
        config["Labels"][
            "org.hegel-machine.phase2b.owner-accepted-authority-requirement.id"
        ] = "wrong"
    elif mutation == "extra_label":
        config["Labels"]["unexpected"] = "value"
    elif mutation == "healthcheck":
        config["Healthcheck"] = {"Test": ["CMD", "true"]}
    elif mutation == "volumes":
        config["Volumes"] = {"/phase2b/output": {}}
    else:  # pragma: no cover - parametrization is a closed local manifest.
        raise AssertionError(mutation)


@pytest.mark.parametrize(
    "mutation",
    (
        "digest",
        "platform",
        "user",
        "args_escaped",
        "missing_args_escaped",
        "extra_config",
        "working_directory",
        "entrypoint",
        "command",
        "rootfs",
        "missing_environment",
        "duplicate_environment",
        "reordered_environment",
        "label",
        "source_root_label",
        "profile_label",
        "authority_label",
        "extra_label",
        "healthcheck",
        "volumes",
    ),
)
def test_image_inspection_rejects_every_frozen_config_drift(mutation: str) -> None:
    value = deepcopy(_image_inspect_value())
    _mutate_image(value, mutation)
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3
    ) as captured:
        executor._validate_image_inspect_v3(_json_bytes(value), spec=_run_spec())
    assert captured.value.code == executor.FAIL_IMAGE_BINDING


def test_supervisor_and_extractor_are_pinned_compilable_inline_programs() -> None:
    compile(
        executor._CONTAINER_SUPERVISOR_CODE_V3,
        "<phase2b-container-supervisor-v3>",
        "exec",
    )
    compile(
        executor._CONTAINER_RESULT_EXTRACTOR_CODE_V3,
        "<phase2b-container-result-extractor-v3>",
        "exec",
    )
    supervisor = executor._CONTAINER_SUPERVISOR_CODE_V3
    extractor = executor._CONTAINER_RESULT_EXTRACTOR_CODE_V3
    assert "contextlib.redirect_stdout" in supervisor
    assert "os.O_EXCL | os.O_NOFOLLOW" in supervisor
    assert executor._RUNTIME_RECEIPT_FILENAME_V3 in supervisor
    assert "while True:" in supervisor and "time.sleep(3600)" in supervisor
    assert "stat.S_ISREG" in extractor
    assert "before.st_nlink != 1" in extractor
    assert "struct.pack('>II'" in extractor
    assert repr(executor._CONTAINER_RESULT_WIRE_MAGIC_V3) in extractor
    assert "docker cp" not in supervisor.casefold()
    assert "docker cp" not in extractor.casefold()


def test_create_command_uses_named_detached_supervisor_and_tmpfs_output_only() -> None:
    spec = _run_spec()
    container_name = "hegel-phase2b-recognizer-v3-" + "d" * 24
    staged_input = Path("/tmp/hegel-runtime/recognizer-input")
    staged_seccomp = Path("/tmp/hegel-runtime/recognizer-seccomp.json")
    command = executor._docker_create_command_v3(
        _control_plane(),
        spec=spec,
        container_name=container_name,
        staged_input_directory=staged_input,
        staged_seccomp_path=staged_seccomp,
    )
    assert command[:5] == (
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "container",
        "create",
        f"--name={container_name}",
    )
    assert "--rm" not in command
    assert "--platform=linux/amd64" in command
    assert "--no-healthcheck" in command
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--workdir=/opt/hegel" in command
    assert "--restart=no" in command
    assert "--stop-timeout=1" in command
    assert f"--entrypoint={executor._CONTAINER_SUPERVISOR_ENTRYPOINT_V3}" in command
    assert f"--security-opt=seccomp={staged_seccomp}" in command
    mounts = tuple(item for item in command if item.startswith("--mount="))
    assert mounts == (
        (
            "--mount=type=bind,src=/tmp/hegel-runtime/recognizer-input,"
            "dst=/phase2b/input,readonly"
        ),
    )
    assert all("dst=/phase2b/output" not in item for item in mounts)
    assert (
        "--tmpfs=/phase2b/output:rw,noexec,nosuid,nodev,"
        "size=536870912,mode=0700,uid=65534,gid=65534"
    ) in command
    image_offset = command.index(spec.image_digest)
    assert command[image_offset + 1 :] == (
        "-c",
        executor._CONTAINER_SUPERVISOR_CODE_V3,
        *spec.entrypoint[1:],
    )


def test_create_command_and_host_path_reject_mount_delimiter_injection() -> None:
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3
    ) as captured:
        executor._docker_create_command_v3(
            _control_plane(),
            spec=_run_spec(),
            container_name="hegel-phase2b-recognizer-v3-" + "d" * 24,
            staged_input_directory=Path("/tmp/hegel,alias/recognizer-input"),
            staged_seccomp_path=Path("/tmp/hegel-runtime/seccomp.json"),
        )
    assert captured.value.code == executor.FAIL_PREFLIGHT
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3
    ) as captured:
        executor._exact_absolute_path(
            "/tmp/hegel,alias",
            "host mount",
            executor.FAIL_HOST_PATH,
        )
    assert captured.value.code == executor.FAIL_HOST_PATH


def test_created_container_inspect_accepts_only_frozen_isolation_projection() -> None:
    _validate_created_container(_created_container_inspect_value())


def _mutate_created_container(value: list[dict[str, object]], mutation: str) -> None:
    row = value[0]
    config = row["Config"]
    host = row["HostConfig"]
    mounts = row["Mounts"]
    assert isinstance(config, dict)
    assert isinstance(host, dict)
    assert isinstance(mounts, list)
    if mutation == "identity_name":
        row["Name"] = "/wrong"
    elif mutation == "identity_image":
        row["Image"] = "sha256:" + SHA_B
    elif mutation == "identity_path":
        row["Path"] = "/bin/false"
    elif mutation == "identity_args":
        row["Args"] = ["--wrong"]
    elif mutation == "state_running":
        row["State"]["Running"] = True
    elif mutation == "image":
        config["Image"] = "sha256:" + SHA_B
    elif mutation == "environment":
        config["Env"] = list(reversed(config["Env"]))
    elif mutation == "entrypoint":
        config["Entrypoint"] = ["/bin/sh"]
    elif mutation == "supervisor":
        config["Cmd"][1] += "\npass"
    elif mutation == "healthcheck":
        config["Healthcheck"] = {"Test": ["CMD", "true"]}
    elif mutation == "volumes":
        config["Volumes"] = {"/phase2b/output": {}}
    elif mutation == "network":
        host["NetworkMode"] = "bridge"
    elif mutation == "read_only":
        host["ReadonlyRootfs"] = False
    elif mutation == "security":
        host["SecurityOpt"] = [host["SecurityOpt"][1]]
    elif mutation == "seccomp":
        host["SecurityOpt"][1] = 'seccomp={"defaultAction":"SCMP_ACT_ALLOW"}'
    elif mutation == "tmpfs":
        host["Tmpfs"]["/phase2b/output"] = "rw,size=536870912"
    elif mutation == "ulimit":
        host["Ulimits"][1]["Hard"] = 536_870_913
    elif mutation == "memory":
        host["Memory"] = 1_073_741_824
    elif mutation == "cpu":
        host["NanoCpus"] = 1_000_000_000
    elif mutation == "pids":
        host["PidsLimit"] = 129
    elif mutation == "runtime":
        host["Runtime"] = "kata"
    elif mutation == "privileged":
        host["Privileged"] = True
    elif mutation == "network_ports":
        row["NetworkSettings"]["Ports"] = {"80/tcp": [{}]}
    elif mutation == "mount_rw":
        mounts[0]["RW"] = True
    elif mutation == "output_bind":
        mounts.append(
            {
                "Type": "bind",
                "Source": "/tmp/output",
                "Destination": "/phase2b/output",
                "Mode": "",
                "RW": True,
                "Propagation": "rprivate",
            }
        )
    else:  # pragma: no cover - parametrization is a closed local manifest.
        raise AssertionError(mutation)


@pytest.mark.parametrize(
    "mutation",
    (
        "identity_name",
        "identity_image",
        "identity_path",
        "identity_args",
        "state_running",
        "image",
        "environment",
        "entrypoint",
        "supervisor",
        "healthcheck",
        "volumes",
        "network",
        "read_only",
        "security",
        "seccomp",
        "tmpfs",
        "ulimit",
        "memory",
        "cpu",
        "pids",
        "runtime",
        "privileged",
        "network_ports",
        "mount_rw",
        "output_bind",
    ),
)
def test_created_container_inspect_rejects_config_hostconfig_or_mount_drift(
    mutation: str,
) -> None:
    value = deepcopy(_created_container_inspect_value())
    _mutate_created_container(value, mutation)
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3
    ) as captured:
        _validate_created_container(value)
    assert captured.value.code == executor.FAIL_DOCKER_EXECUTION


def test_result_wire_round_trips_receipt_and_prediction_without_host_copy() -> None:
    receipt = b"{}\n"
    prediction = b"P" * PREDICTION_ARCHIVE_HEADER_BYTES_V2
    wire = _result_wire(receipt, prediction)
    assert executor._extract_container_result_wire_v3(wire) == (
        receipt,
        prediction,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "bad_magic",
        "truncated",
        "extra_byte",
        "short_receipt_projection",
        "short_prediction_projection",
        "receipt_without_newline",
        "receipt_two_lines",
    ),
)
def test_result_wire_rejects_magic_length_truncation_extra_and_receipt_framing(
    mutation: str,
) -> None:
    receipt = b"{}\n"
    prediction = b"P" * PREDICTION_ARCHIVE_HEADER_BYTES_V2
    wire = _result_wire(receipt, prediction)
    if mutation == "bad_magic":
        wire = b"X" + wire[1:]
    elif mutation == "truncated":
        wire = wire[:-1]
    elif mutation == "extra_byte":
        wire += b"X"
    elif mutation == "short_receipt_projection":
        offset = len(executor._CONTAINER_RESULT_WIRE_MAGIC_V3)
        wire = wire[:offset] + (2).to_bytes(4, "big") + wire[offset + 4 :]
    elif mutation == "short_prediction_projection":
        offset = len(executor._CONTAINER_RESULT_WIRE_MAGIC_V3) + 4
        wire = (
            wire[:offset]
            + (PREDICTION_ARCHIVE_HEADER_BYTES_V2 - 1).to_bytes(4, "big")
            + wire[offset + 4 :]
        )
    elif mutation == "receipt_without_newline":
        wire = _result_wire(b"{}", prediction)
    elif mutation == "receipt_two_lines":
        wire = _result_wire(b"{}\n{}\n", prediction)
    else:  # pragma: no cover - parametrization is a closed local manifest.
        raise AssertionError(mutation)
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3
    ) as captured:
        executor._extract_container_result_wire_v3(wire)
    assert captured.value.code == executor.FAIL_OUTPUT_BINDING


@pytest.mark.parametrize(
    "value",
    (
        "relative/path",
        "/",
        "/tmp/../tmp/target",
        "/tmp//target",
        "/tmp/target/",
        "/tmp/./target",
        "/tmp/target,alias",
        "/tmp/target\nnext",
    ),
)
def test_exact_host_path_rejects_ambiguous_spelling(value: str) -> None:
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3
    ) as captured:
        executor._exact_absolute_path(value, "host path", executor.FAIL_HOST_PATH)
    assert captured.value.code == executor.FAIL_HOST_PATH


def test_componentwise_open_rejects_symlink_and_directory_overlap(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(child, target_is_directory=True)
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3
    ) as captured:
        executor._open_componentwise_directory_v3(alias, label="symlink alias")
    assert captured.value.code == executor.FAIL_HOST_PATH

    pinned_parent = executor._open_componentwise_directory_v3(
        parent,
        label="parent",
    )
    pinned_child = executor._open_componentwise_directory_v3(
        child,
        label="child",
    )
    try:
        with pytest.raises(
            executor.Phase2BRecognizerHostExecutorErrorV3
        ) as captured:
            executor._require_pairwise_nonoverlap_v3(
                {"input": pinned_parent, "output": pinned_child}
            )
        assert captured.value.code == executor.FAIL_HOST_PATH
    finally:
        pinned_child.close()
        pinned_parent.close()


def test_namespace_scan_and_single_link_policy_bind_exact_input_output(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()
    output_path.mkdir()
    (input_path / RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2).write_bytes(b"archive")
    (input_path / EXECUTION_FREEZE_MANIFEST_FILENAME_V2).write_bytes(b"{}")
    pinned_input = executor._open_componentwise_directory_v3(
        input_path,
        label="input",
    )
    pinned_output = executor._open_componentwise_directory_v3(
        output_path,
        label="output",
    )
    try:
        assert executor._directory_names_v3(
            pinned_input,
            label="input",
            code=executor.FAIL_INPUT_BINDING,
        ) == frozenset(
            {
                RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2,
                EXECUTION_FREEZE_MANIFEST_FILENAME_V2,
            }
        )
        assert not executor._directory_names_v3(
            pinned_output,
            label="output",
            code=executor.FAIL_OUTPUT_BINDING,
        )
        (output_path / "unexpected").write_bytes(b"x")
        assert executor._directory_names_v3(
            pinned_output,
            label="output after drift",
            code=executor.FAIL_OUTPUT_BINDING,
        ) == frozenset({"unexpected"})

        hardlink = input_path / "input-hardlink"
        os.link(input_path / RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2, hardlink)
        with pytest.raises(
            executor.Phase2BRecognizerHostExecutorErrorV3
        ) as captured:
            executor._open_relative_regular_file_v3(
                pinned_input,
                RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2,
                maximum_bytes=64,
                minimum_bytes=1,
                single_link_required=True,
                label="input archive",
                code=executor.FAIL_INPUT_BINDING,
            )
        assert captured.value.code == executor.FAIL_INPUT_BINDING
    finally:
        pinned_output.close()
        pinned_input.close()


def test_staged_input_is_repinned_at_final_0555_mode_on_local_ext4() -> None:
    with tempfile.TemporaryDirectory(
        prefix="hegel-v3-staged-repin-test-",
        dir="/tmp",
    ) as temporary:
        runtime = executor._open_componentwise_directory_v3(
            Path(temporary),
            label="test private runtime",
            code=executor.FAIL_INPUT_BINDING,
        )
        staged: executor._PinnedDirectoryV3 | None = None
        input_file: executor._PinnedRegularFileV3 | None = None
        manifest_file: executor._PinnedRegularFileV3 | None = None
        try:
            staged, input_file, manifest_file = (
                executor._stage_and_repin_readonly_input_v3(
                    runtime,
                    input_payload=b"I" * executor.ARCHIVE_HEADER_BYTES_V2,
                    manifest_payload=b"{}\n",
                )
            )
            assert stat.S_IMODE(staged.final_identity.mode) == 0o555
            assert stat.S_IMODE(input_file.metadata_fingerprint[2]) == 0o444
            assert stat.S_IMODE(manifest_file.metadata_fingerprint[2]) == 0o444
            assert input_file.base_directory is staged
            assert manifest_file.base_directory is staged
            executor._recheck_pinned_directory_v3(
                staged,
                label="sealed staged input final mode",
                code=executor.FAIL_INPUT_BINDING,
            )
            executor._recheck_regular_file_v3(
                input_file,
                maximum_bytes=executor.MAXIMUM_RECOGNIZER_INPUT_ARCHIVE_BYTES_V2,
                minimum_bytes=executor.ARCHIVE_HEADER_BYTES_V2,
                label="sealed staged input final mode",
                code=executor.FAIL_INPUT_BINDING,
            )
            executor._recheck_regular_file_v3(
                manifest_file,
                maximum_bytes=executor.MAXIMUM_EXECUTION_FREEZE_MANIFEST_BYTES_V2,
                minimum_bytes=2,
                label="sealed staged manifest final mode",
                code=executor.FAIL_INPUT_BINDING,
            )
        finally:
            if manifest_file is not None:
                manifest_file.close()
            if input_file is not None:
                input_file.close()
            if staged is not None:
                staged.close()
            runtime.close()


def test_partial_publication_failure_removes_pending_and_final_exact_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "output"
    output_path.mkdir()
    output = executor._open_componentwise_directory_v3(
        output_path,
        label="publication output",
        code=executor.FAIL_OUTPUT_BINDING,
    )
    real_fsync = executor.os.fsync
    directory_fsyncs = 0

    def fail_second_directory_fsync(descriptor: int) -> None:
        nonlocal directory_fsyncs
        if descriptor == output.descriptor:
            directory_fsyncs += 1
            if directory_fsyncs == 2:
                raise OSError("injected after hardlink")
        real_fsync(descriptor)

    monkeypatch.setattr(executor.os, "fsync", fail_second_directory_fsync)
    try:
        with pytest.raises(
            executor.Phase2BRecognizerHostExecutorErrorV3
        ) as captured:
            executor._publish_prediction_output_v3(
                output,
                b"P" * PREDICTION_ARCHIVE_HEADER_BYTES_V2,
            )
        assert captured.value.code == executor.FAIL_OUTPUT_BINDING
        assert list(output_path.iterdir()) == []
        assert directory_fsyncs == 3
    finally:
        output.close()


def test_partial_publication_cleanup_failure_is_stable_and_does_not_unlink_alien(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "output"
    output_path.mkdir()
    output = executor._open_componentwise_directory_v3(
        output_path,
        label="publication output",
        code=executor.FAIL_OUTPUT_BINDING,
    )
    real_fsync = executor.os.fsync
    real_unlink = executor.os.unlink
    directory_fsyncs = 0

    def replace_final_after_hardlink(descriptor: int) -> None:
        nonlocal directory_fsyncs
        if descriptor == output.descriptor:
            directory_fsyncs += 1
            if directory_fsyncs == 2:
                real_unlink(
                    RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2,
                    dir_fd=output.descriptor,
                )
                (
                    output_path / RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2
                ).write_bytes(b"alien inode")
                raise OSError("injected inode replacement after hardlink")
        real_fsync(descriptor)

    monkeypatch.setattr(executor.os, "fsync", replace_final_after_hardlink)
    try:
        with pytest.raises(
            executor.Phase2BRecognizerHostExecutorErrorV3,
            match="exact output cleanup was incomplete",
        ) as captured:
            executor._publish_prediction_output_v3(
                output,
                b"P" * PREDICTION_ARCHIVE_HEADER_BYTES_V2,
            )
        assert captured.value.code == executor.FAIL_OUTPUT_BINDING
        assert [item.name for item in output_path.iterdir()] == [
            RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2
        ]
        assert (
            output_path / RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2
        ).read_bytes() == b"alien inode"
    finally:
        try:
            real_unlink(
                RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2,
                dir_fd=output.descriptor,
            )
        except FileNotFoundError:
            pass
        output.close()


def test_receipt_issuance_failure_durably_removes_exact_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "output"
    output_path.mkdir()
    output = executor._open_componentwise_directory_v3(
        output_path,
        label="receipt output",
        code=executor.FAIL_OUTPUT_BINDING,
    )
    prediction = executor._publish_prediction_output_v3(
        output,
        b"P" * PREDICTION_ARCHIVE_HEADER_BYTES_V2,
    )

    def reject_issue(
        cls: type[object], token: object, **values: object
    ) -> object:
        del cls, token, values
        raise ValueError("injected receipt issuance failure")

    monkeypatch.setattr(
        executor.Phase2BRecognizerLocalExecutionReceiptV3,
        "_issue",
        classmethod(reject_issue),
    )
    try:
        with pytest.raises(ValueError, match="injected receipt issuance failure"):
            executor._issue_receipt_after_publication_v3(
                output,
                prediction,
            )
        assert list(output_path.iterdir()) == []
    finally:
        prediction.close()
        output.close()


def test_receipt_failure_refuses_to_delete_replaced_prediction_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "output"
    output_path.mkdir()
    output = executor._open_componentwise_directory_v3(
        output_path,
        label="receipt output",
        code=executor.FAIL_OUTPUT_BINDING,
    )
    prediction = executor._publish_prediction_output_v3(
        output,
        b"P" * PREDICTION_ARCHIVE_HEADER_BYTES_V2,
    )
    final = output_path / RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2
    final.unlink()
    final.write_bytes(b"replacement inode")

    def reject_issue(
        cls: type[object], token: object, **values: object
    ) -> object:
        del cls, token, values
        raise ValueError("injected receipt issuance failure")

    monkeypatch.setattr(
        executor.Phase2BRecognizerLocalExecutionReceiptV3,
        "_issue",
        classmethod(reject_issue),
    )
    try:
        with pytest.raises(
            executor.Phase2BRecognizerHostExecutorErrorV3,
            match="exact prediction cleanup was incomplete",
        ) as captured:
            executor._issue_receipt_after_publication_v3(
                output,
                prediction,
            )
        assert captured.value.code == executor.FAIL_OUTPUT_BINDING
        assert final.read_bytes() == b"replacement inode"
    finally:
        prediction.close()
        output.close()


@pytest.mark.parametrize(
    ("code", "label"),
    (
        (executor.FAIL_SECCOMP_BINDING, "seccomp profile"),
        (executor.FAIL_DOCKER_CONTROL_PLANE, "Docker executable"),
    ),
)
def test_held_regular_file_recheck_detects_seccomp_or_executable_drift(
    tmp_path: Path,
    code: str,
    label: str,
) -> None:
    root = tmp_path / label.replace(" ", "-")
    root.mkdir()
    target = root / "material"
    target.write_bytes(b"original")
    pinned_root = executor._open_componentwise_directory_v3(root, label=label)
    pinned_file = executor._open_relative_regular_file_v3(
        pinned_root,
        target.name,
        maximum_bytes=64,
        minimum_bytes=1,
        single_link_required=True,
        label=label,
        code=code,
    )
    try:
        target.write_bytes(b"mutated!")
        with pytest.raises(
            executor.Phase2BRecognizerHostExecutorErrorV3
        ) as captured:
            executor._recheck_regular_file_v3(
                pinned_file,
                maximum_bytes=64,
                minimum_bytes=1,
                label=f"{label} after run",
                code=code,
            )
        assert captured.value.code == code
    finally:
        pinned_file.close()
        pinned_root.close()


def test_post_run_directory_rebind_is_detected_by_component_identity(
    tmp_path: Path,
) -> None:
    target = tmp_path / "output"
    target.mkdir()
    pinned = executor._open_componentwise_directory_v3(target, label="output")
    old_target = tmp_path / "output-old"
    try:
        target.rename(old_target)
        target.mkdir()
        with pytest.raises(
            executor.Phase2BRecognizerHostExecutorErrorV3
        ) as captured:
            executor._recheck_pinned_directory_v3(
                pinned,
                label="output after run",
                code=executor.FAIL_OUTPUT_BINDING,
            )
        assert captured.value.code == executor.FAIL_OUTPUT_BINDING
    finally:
        pinned.close()


class _CompletedFakePopen:
    def __init__(
        self,
        _command: list[str],
        *,
        stdout_payload: bytes,
        stderr_payload: bytes,
        returncode: int = 0,
        **_kwargs: object,
    ) -> None:
        self.pid = 2_147_000_001
        self.returncode = returncode
        stdout_read, stdout_write = os.pipe()
        stderr_read, stderr_write = os.pipe()
        os.write(stdout_write, stdout_payload)
        os.write(stderr_write, stderr_payload)
        os.close(stdout_write)
        os.close(stderr_write)
        self.stdout = os.fdopen(stdout_read, "rb", buffering=0)
        self.stderr = os.fdopen(stderr_read, "rb", buffering=0)

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


def _bounded_command(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdout_payload: bytes,
    stderr_payload: bytes,
    stdout_cap: int = 64,
    stderr_cap: int = 64,
    command: tuple[str, ...] = (
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "version",
    ),
    forced_cleanup: Callable[[], None] | None = None,
) -> executor._BoundedCompletedProcessV3:
    def factory(command: list[str], **kwargs: object) -> _CompletedFakePopen:
        return _CompletedFakePopen(
            command,
            stdout_payload=stdout_payload,
            stderr_payload=stderr_payload,
            **kwargs,
        )

    monkeypatch.setattr(executor.subprocess, "Popen", factory)
    monkeypatch.setattr(executor, "_kill_process_group_v3", lambda _process: None)
    return executor._run_bounded_command_v3(
        command,
        environment={"PATH": "/usr/bin:/bin"},
        working_directory=Path("/tmp"),
        timeout_seconds=1,
        maximum_stdout_bytes=stdout_cap,
        maximum_stderr_bytes=stderr_cap,
        timeout_code=executor.FAIL_DOCKER_TIMEOUT,
        execution_code=executor.FAIL_DOCKER_EXECUTION,
        forced_cleanup=forced_cleanup,
    )


def test_bounded_subprocess_captures_both_pipes_without_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = _bounded_command(
        monkeypatch,
        stdout_payload=b"stdout",
        stderr_payload=b"stderr",
    )
    assert completed.command == (
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "version",
    )
    assert completed.returncode == 0
    assert completed.stdout == b"stdout"
    assert completed.stderr == b"stderr"
    assert completed.finished_monotonic_ns >= completed.started_monotonic_ns


@pytest.mark.parametrize(
    ("stdout_payload", "stderr_payload", "stdout_cap", "stderr_cap", "stream"),
    (
        (b"1234", b"", 3, 3, "stdout"),
        (b"", b"1234", 3, 3, "stderr"),
    ),
)
def test_bounded_subprocess_kills_on_stdout_or_stderr_overflow(
    monkeypatch: pytest.MonkeyPatch,
    stdout_payload: bytes,
    stderr_payload: bytes,
    stdout_cap: int,
    stderr_cap: int,
    stream: str,
) -> None:
    cleaned: list[str] = []
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3,
        match=f"Docker {stream} exceeded",
    ) as captured:
        _bounded_command(
            monkeypatch,
            stdout_payload=stdout_payload,
            stderr_payload=stderr_payload,
            stdout_cap=stdout_cap,
            stderr_cap=stderr_cap,
            forced_cleanup=lambda: cleaned.append("container"),
        )
    assert captured.value.code == executor.FAIL_DOCKER_CAPTURE
    assert cleaned == ["container"]


class _TimeoutFakePopen:
    def __init__(self, _command: list[str], **_kwargs: object) -> None:
        self.pid = 2_147_000_002
        self.returncode: int | None = None
        self.wait_calls = 0
        self._stdout_read, self._stdout_write = os.pipe()
        self._stderr_read, self._stderr_write = os.pipe()
        self.stdout = os.fdopen(self._stdout_read, "rb", buffering=0)
        self.stderr = os.fdopen(self._stderr_read, "rb", buffering=0)

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.wait_calls += 1
        for descriptor in (self._stdout_write, self._stderr_write):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.returncode = -9
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


class _NoWaitSelector:
    def register(self, *_args: object, **_kwargs: object) -> None:
        return None

    def unregister(self, *_args: object, **_kwargs: object) -> None:
        return None

    def select(self, timeout: float | None = None) -> list[object]:
        del timeout
        return []

    def close(self) -> None:
        return None


class _FailingRegisterSelector(_NoWaitSelector):
    def __init__(self, exception: BaseException) -> None:
        self.exception = exception

    def register(self, *_args: object, **_kwargs: object) -> None:
        raise self.exception


def test_bounded_subprocess_timeout_is_fail_closed_without_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((0.0, 2.0))
    killed: list[bool] = []
    cleaned: list[str] = []
    container_name = "hegel-phase2b-recognizer-v3-" + "d" * 24
    monkeypatch.setattr(
        executor.subprocess,
        "Popen",
        lambda command, **kwargs: _TimeoutFakePopen(command, **kwargs),
    )
    monkeypatch.setattr(executor.selectors, "DefaultSelector", _NoWaitSelector)
    monkeypatch.setattr(executor.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        executor,
        "_kill_process_group_v3",
        lambda _process: killed.append(True),
    )
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3,
        match="exceeded exact 1s timeout",
    ) as captured:
        executor._run_bounded_command_v3(
            (
                "/usr/bin/docker",
                "--host=unix:///var/run/docker.sock",
                "create",
                f"--name={container_name}",
            ),
            environment={"PATH": "/usr/bin:/bin"},
            working_directory=Path("/tmp"),
            timeout_seconds=1,
            maximum_stdout_bytes=1,
            maximum_stderr_bytes=1,
            timeout_code=executor.FAIL_DOCKER_TIMEOUT,
            execution_code=executor.FAIL_DOCKER_EXECUTION,
            forced_cleanup=lambda: cleaned.append(container_name),
        )
    assert captured.value.code == executor.FAIL_DOCKER_TIMEOUT
    assert killed == [True]
    assert cleaned == [container_name]


@pytest.mark.parametrize(
    "container_target",
    ("hegel-phase2b-recognizer-v3-" + "d" * 24, SHA_C),
)
def test_force_remove_uses_rm_then_negative_inspect_for_name_or_full_id(
    monkeypatch: pytest.MonkeyPatch,
    container_target: str,
) -> None:
    calls: list[tuple[str, ...]] = []
    absent_stderr = (
        f"Error response from daemon: No such container: {container_target}\n"
    ).encode("ascii")

    def fake_run(command: tuple[str, ...], **_kwargs: object) -> object:
        calls.append(command)
        if command[2:5] == ("container", "rm", "--force"):
            return SimpleNamespace(
                returncode=0,
                stdout=(container_target + "\n").encode("ascii"),
                stderr=b"",
            )
        return SimpleNamespace(
            returncode=1,
            stdout=b"[]\n",
            stderr=absent_stderr,
        )

    monkeypatch.setattr(executor, "_run_bounded_command_v3", fake_run)
    executor._force_remove_container_v3(
        control_plane=_control_plane(),
        environment={"PATH": "/usr/bin:/bin"},
        working_directory=Path("/tmp"),
        container_id=container_target,
    )
    assert calls == [
        (
            "/usr/bin/docker",
            "--host=unix:///var/run/docker.sock",
            "container",
            "rm",
            "--force",
            container_target,
        ),
        (
            "/usr/bin/docker",
            "--host=unix:///var/run/docker.sock",
            "container",
            "inspect",
            container_target,
        ),
    ]


def test_precreate_name_cleanup_alone_may_accept_exact_already_absent_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_name = "hegel-phase2b-recognizer-v3-" + "d" * 24
    absent_stderr = (
        f"Error response from daemon: No such container: {container_name}\n"
    ).encode("ascii")
    observations = iter(
        (
            SimpleNamespace(returncode=0, stdout=b"", stderr=absent_stderr),
            SimpleNamespace(returncode=1, stdout=b"[]\n", stderr=absent_stderr),
        )
    )
    monkeypatch.setattr(
        executor,
        "_run_bounded_command_v3",
        lambda *_args, **_kwargs: next(observations),
    )
    executor._force_remove_container_v3(
        control_plane=_control_plane(),
        environment={"PATH": "/usr/bin:/bin"},
        working_directory=Path("/tmp"),
        container_id=container_name,
        allow_already_absent=True,
    )


@pytest.mark.parametrize(
    ("allow_already_absent", "remove_stdout", "remove_stderr", "inspect_stdout"),
    (
        (False, b"", b"ABSENT", b"[]\n"),
        (True, b"", b"wrong absence text", b"[]\n"),
        (True, b"", b"ABSENT", b""),
    ),
)
def test_cleanup_rejects_absence_without_exact_policy_and_exact_docker_tuples(
    monkeypatch: pytest.MonkeyPatch,
    allow_already_absent: bool,
    remove_stdout: bytes,
    remove_stderr: bytes,
    inspect_stdout: bytes,
) -> None:
    container_name = "hegel-phase2b-recognizer-v3-" + "d" * 24
    exact_absent = (
        f"Error response from daemon: No such container: {container_name}\n"
    ).encode("ascii")
    normalized_remove_stderr = (
        exact_absent if remove_stderr == b"ABSENT" else remove_stderr
    )
    observations = iter(
        (
            SimpleNamespace(
                returncode=0,
                stdout=remove_stdout,
                stderr=normalized_remove_stderr,
            ),
            SimpleNamespace(
                returncode=1,
                stdout=inspect_stdout,
                stderr=exact_absent,
            ),
        )
    )
    monkeypatch.setattr(
        executor,
        "_run_bounded_command_v3",
        lambda *_args, **_kwargs: next(observations),
    )
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3
    ) as captured:
        executor._force_remove_container_v3(
            control_plane=_control_plane(),
            environment={"PATH": "/usr/bin:/bin"},
            working_directory=Path("/tmp"),
            container_id=container_name,
            allow_already_absent=allow_already_absent,
        )
    assert captured.value.code == executor.FAIL_DOCKER_EXECUTION


def test_start_and_extractor_calls_are_wired_to_the_same_id_cleanup() -> None:
    source = textwrap.dedent(
        inspect.getsource(executor.execute_phase2b_recognizer_docker_v3)
    )
    tree = ast.parse(source)
    cleanup_by_command: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "_run_bounded_command_v3" or not node.args:
            continue
        forced = next(
            (item.value for item in node.keywords if item.arg == "forced_cleanup"),
            None,
        )
        if forced is None:
            continue
        cleanup_by_command.append((ast.unparse(node.args[0]), ast.unparse(forced)))
    assert ("create_command", "remove_named_container") in cleanup_by_command
    assert ("start_command", "remove_container") in cleanup_by_command
    assert ("extractor_command", "remove_container") in cleanup_by_command
    removal_bindings: list[dict[str, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_force_remove_container_v3"
        ):
            removal_bindings.append(
                {
                    item.arg: ast.unparse(item.value)
                    for item in node.keywords
                    if item.arg is not None
                }
            )
    allowed_absent = tuple(
        row for row in removal_bindings if row.get("allow_already_absent") == "True"
    )
    assert len(allowed_absent) == 1
    assert allowed_absent[0]["container_id"] == "container_name"
    assert any(
        row.get("container_id") == "container_id"
        and "allow_already_absent" not in row
        for row in removal_bindings
    )


def test_start_to_extractor_uses_one_absolute_deadline_and_rejects_late_result() -> None:
    source = textwrap.dedent(
        inspect.getsource(executor.execute_phase2b_recognizer_docker_v3)
    )
    compact = "".join(source.split())
    start_mark = compact.index("recognizer_started_monotonic_ns=time.monotonic_ns()")
    start_call = compact.index("container_started=_run_bounded_command_v3")
    remaining_mark = compact.index(
        "recognizer_deadline_monotonic_ns-time.monotonic_ns()"
    )
    extractor_call = compact.index("runtime_observation=_run_bounded_command_v3")
    late_check = compact.index(
        "runtime_observation.finished_monotonic_ns>"
        "recognizer_deadline_monotonic_ns"
    )
    assert start_mark < start_call < remaining_mark < extractor_call < late_check
    assert "remaining_seconds = max(" in source
    assert "timeout_seconds=remaining_seconds" in source
    assert "recognizer deadline elapsed before result extraction" in source
    assert "recognizer result arrived after the one absolute deadline" in source


def test_lifecycle_command_hash_and_inline_program_hashes_cover_full_mechanics() -> None:
    source = textwrap.dedent(
        inspect.getsource(executor.execute_phase2b_recognizer_docker_v3)
    )
    lifecycle_keys = (
        '"image_inspect": image_command',
        '"create": create_command',
        '"created_container_inspect": container_inspect_command',
        '"start": start_command',
        '"extract_result": extractor_command',
        '"remove": tuple(',
        '"verify_removed": tuple(',
    )
    assert all(item in source for item in lifecycle_keys)
    receipt = _local_receipt()
    assert receipt.container_supervisor_sha256 == executor._sha256(
        (
            executor._CONTAINER_SUPERVISOR_ENTRYPOINT_V3
            + "\x00"
            + executor._CONTAINER_SUPERVISOR_CODE_V3
        ).encode("utf-8")
    )
    assert receipt.container_result_extractor_sha256 == executor._sha256(
        (
            executor._CONTAINER_SUPERVISOR_ENTRYPOINT_V3
            + "\x00"
            + executor._CONTAINER_RESULT_EXTRACTOR_CODE_V3
        ).encode("utf-8")
    )


@pytest.mark.parametrize(
    "phase_command",
    (
        ("start", SHA_C),
        (
            "container",
            "exec",
            "--user=65534:65534",
            SHA_C,
            "/usr/local/bin/python3",
            "-c",
            "pass",
        ),
    ),
)
def test_start_or_extractor_overflow_forces_container_id_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    phase_command: tuple[str, ...],
) -> None:
    cleaned: list[str] = []
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3
    ) as captured:
        _bounded_command(
            monkeypatch,
            stdout_payload=b"AB",
            stderr_payload=b"",
            stdout_cap=1,
            stderr_cap=1,
            command=(
                "/usr/bin/docker",
                "--host=unix:///var/run/docker.sock",
                *phase_command,
            ),
            forced_cleanup=lambda: cleaned.append(SHA_C),
        )
    assert captured.value.code == executor.FAIL_DOCKER_CAPTURE
    assert cleaned == [SHA_C]


def test_bounded_command_baseexception_still_forces_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleaned: list[str] = []
    monkeypatch.setattr(
        executor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        executor._run_bounded_command_v3(
            ("/usr/bin/docker", "--host=unix:///var/run/docker.sock", "create"),
            environment={"PATH": "/usr/bin:/bin"},
            working_directory=Path("/tmp"),
            timeout_seconds=1,
            maximum_stdout_bytes=1,
            maximum_stderr_bytes=1,
            timeout_code=executor.FAIL_DOCKER_TIMEOUT,
            execution_code=executor.FAIL_DOCKER_EXECUTION,
            forced_cleanup=lambda: cleaned.append("named-container"),
        )
    assert cleaned == ["named-container"]


@pytest.mark.parametrize(
    "injected",
    (
        executor.Phase2BRecognizerHostExecutorErrorV3(
            executor.FAIL_DOCKER_EXECUTION,
            "injected phase failure",
        ),
        OSError("injected ordinary failure"),
    ),
)
def test_phase_or_ordinary_process_branch_kills_reaps_then_cleans_container(
    monkeypatch: pytest.MonkeyPatch,
    injected: BaseException,
) -> None:
    processes: list[_TimeoutFakePopen] = []
    killed: list[int] = []
    cleaned: list[str] = []

    def popen(command: list[str], **kwargs: object) -> _TimeoutFakePopen:
        process = _TimeoutFakePopen(command, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(executor.subprocess, "Popen", popen)
    monkeypatch.setattr(
        executor.selectors,
        "DefaultSelector",
        lambda: _FailingRegisterSelector(injected),
    )
    monkeypatch.setattr(
        executor,
        "_kill_process_group_v3",
        lambda process: killed.append(process.pid),
    )
    with pytest.raises(executor.Phase2BRecognizerHostExecutorErrorV3):
        executor._run_bounded_command_v3(
            ("/usr/bin/docker", "--host=unix:///var/run/docker.sock", "start"),
            environment={"PATH": "/usr/bin:/bin"},
            working_directory=Path("/tmp"),
            timeout_seconds=1,
            maximum_stdout_bytes=1,
            maximum_stderr_bytes=1,
            timeout_code=executor.FAIL_DOCKER_TIMEOUT,
            execution_code=executor.FAIL_DOCKER_EXECUTION,
            forced_cleanup=lambda: cleaned.append(SHA_C),
        )
    assert len(processes) == 1
    assert killed == [processes[0].pid]
    assert processes[0].wait_calls >= 1
    assert processes[0].returncode == -9
    assert cleaned == [SHA_C]


def test_runtime_receipt_is_canonical_and_crossbound_to_both_archives() -> None:
    input_archive, prediction_archive, receipt = _runtime_receipt_fixture()
    expected_id = receipt["receipt_id"]
    assert _validate_runtime_receipt(
        receipt,
        input_archive,
        prediction_archive,
    ) == expected_id


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("input_archive_sha256", SHA_C),
        ("execution_freeze_manifest_id", "other_freeze"),
        ("prediction_archive_sha256", SHA_A),
        ("prediction_case_count", TOTAL_RECOGNIZER_CASE_COUNT - 1),
        ("output_filename", "other.bin"),
        ("receipt_id", "phase2b_recognizer_runtime_receipt_v2_" + SHA_A),
    ),
)
def test_runtime_receipt_rejects_output_or_receipt_crossbinding_drift(
    field: str,
    replacement: object,
) -> None:
    input_archive, prediction_archive, receipt = _runtime_receipt_fixture()
    receipt[field] = replacement
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3
    ) as captured:
        _validate_runtime_receipt(receipt, input_archive, prediction_archive)
    assert captured.value.code == executor.FAIL_OUTPUT_BINDING


def test_runtime_receipt_rejects_extra_fields_and_noncanonical_json() -> None:
    input_archive, prediction_archive, receipt = _runtime_receipt_fixture()
    extra = dict(receipt)
    extra["actual_960_case_run_verified"] = True
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3
    ) as captured:
        _validate_runtime_receipt(extra, input_archive, prediction_archive)
    assert captured.value.code == executor.FAIL_OUTPUT_BINDING

    noncanonical = json.dumps(receipt, indent=2, sort_keys=False).encode("ascii")
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3
    ) as captured:
        executor._validate_runtime_receipt_v3(
            noncanonical,
            input_archive=input_archive,
            input_archive_sha256=SHA_A,
            execution_manifest_id="freeze_manifest_v1",
            execution_manifest_sha256=SHA_B,
            prediction_archive=prediction_archive,
            prediction_archive_sha256=SHA_C,
            prediction_archive_bytes=PREDICTION_ARCHIVE_HEADER_BYTES_V2,
        )
    assert captured.value.code == executor.FAIL_OUTPUT_BINDING


def test_prediction_archive_context_and_rows_crossbind_to_exact_input() -> None:
    input_archive, prediction = _crossbinding_fixture()
    executor._crossbind_prediction_to_input_v3(
        prediction,
        input_archive=input_archive,
        input_archive_sha256=SHA_A,
        execution_manifest_id="phase2b_execution_freeze_" + SHA_B,
        protocol_id="phase2b_protocol_" + SHA_C,
    )


def _mutate_crossbinding(prediction: object, mutation: str) -> None:
    if mutation == "input_archive_id":
        prediction.context.input_archive_id = "other"
    elif mutation == "input_archive_sha256":
        prediction.context.input_archive_sha256 = SHA_B
    elif mutation == "freeze":
        prediction.context.execution_freeze_manifest_id = "other"
    elif mutation == "protocol":
        prediction.context.protocol_id = "other"
    elif mutation == "batch":
        prediction.context.batch_id = "other"
    elif mutation == "batch_policy":
        prediction.context.batch_policy_id = "other"
    elif mutation == "input_row_order":
        object.__setattr__(prediction, "input_row_ids", ("row-b", "row-a"))
    elif mutation == "record_row_order":
        object.__setattr__(
            prediction,
            "records",
            (
                SimpleNamespace(input_row_id="row-b"),
                SimpleNamespace(input_row_id="row-a"),
            ),
        )
    else:  # pragma: no cover - parametrization is a closed local manifest.
        raise AssertionError(mutation)


@pytest.mark.parametrize(
    "mutation",
    (
        "input_archive_id",
        "input_archive_sha256",
        "freeze",
        "protocol",
        "batch",
        "batch_policy",
        "input_row_order",
        "record_row_order",
    ),
)
def test_prediction_archive_rejects_input_freeze_batch_or_row_crossbinding_drift(
    mutation: str,
) -> None:
    input_archive, prediction = _crossbinding_fixture()
    _mutate_crossbinding(prediction, mutation)
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3
    ) as captured:
        executor._crossbind_prediction_to_input_v3(
            prediction,
            input_archive=input_archive,
            input_archive_sha256=SHA_A,
            execution_manifest_id="phase2b_execution_freeze_" + SHA_B,
            protocol_id="phase2b_protocol_" + SHA_C,
        )
    assert captured.value.code == executor.FAIL_OUTPUT_BINDING


def test_complete_fake_docker_success_executes_and_replays_full_host_graph(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    local_owner = tempfile.TemporaryDirectory(
        prefix="hegel-v3-full-fake-success-",
        dir="/tmp",
    )
    request.addfinalizer(local_owner.cleanup)
    tmp_path = Path(local_owner.name)
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    runtime_parent = tmp_path
    for path in (input_path, output_path):
        path.mkdir()
    input_payload = b"I" * executor.ARCHIVE_HEADER_BYTES_V2
    manifest_payload = b"{}\n"
    prediction_payload = b"P" * PREDICTION_ARCHIVE_HEADER_BYTES_V2
    (input_path / RECOGNIZER_INPUT_ARCHIVE_FILENAME_V2).write_bytes(
        input_payload
    )
    (input_path / EXECUTION_FREEZE_MANIFEST_FILENAME_V2).write_bytes(
        manifest_payload
    )
    freeze_id = "phase2b_execution_freeze_" + SHA_A
    repository_path = ROOT
    spec = OciRecognizerRunSpec(
        freeze_manifest_id=freeze_id,
        protocol_id=frozen_phase2b_protocol().protocol_id,
        image_digest=IMAGE_A,
        input_host_directory=input_path.as_posix(),
        output_host_directory=output_path.as_posix(),
        repository_host_directory=repository_path.as_posix(),
        seccomp_profile_host_path=(
            repository_path
            / "config"
            / "phase3_internal_actor_seccomp_v1.json"
        ).as_posix(),
        seccomp_profile_sha256=RECOGNIZER_SECCOMP_PROFILE_SHA256,
        input_manifest_sha256=executor._sha256(input_payload),
        expected_case_count=TOTAL_RECOGNIZER_CASE_COUNT,
        entrypoint=FROZEN_RECOGNIZER_ENTRYPOINT,
        limits=RuntimeLimits(),
    )
    row_ids = tuple(
        f"phase2b-row-{index:04d}"
        for index in range(TOTAL_RECOGNIZER_CASE_COUNT)
    )
    input_archive = _private_decoded(
        DecodedRecognizerInputArchiveV2,
        disposition=executor.RecognizerInputArchiveDispositionV2.COMPLETE,
        archive=input_payload,
        archive_id="phase2b_recognizer_input_archive_v2_" + SHA_A,
        rows=tuple(object() for _ in row_ids),
        row_ids=row_ids,
        batch_id="phase2b_trusted_wire_batch_v2_" + SHA_B,
        batch_policy_id="phase2b-batch-policy-v2",
    )
    execution_manifest = SimpleNamespace(
        manifest_id=freeze_id,
        protocol_id=spec.protocol_id,
        recognizer_image_digest=spec.image_digest,
    )
    prediction_context = SimpleNamespace(
        input_archive_id=input_archive.archive_id,
        input_archive_sha256=executor._sha256(input_payload),
        execution_freeze_manifest_id=freeze_id,
        protocol_id=spec.protocol_id,
        batch_id=input_archive.batch_id,
        batch_policy_id=input_archive.batch_policy_id,
    )
    prediction_archive = _private_decoded(
        DecodedRecognizerPredictionArchiveV2,
        disposition=executor.PredictionArchiveDispositionV2.COMPLETE,
        archive=prediction_payload,
        archive_id="phase2b_recognizer_prediction_archive_v2_" + SHA_B,
        context=prediction_context,
        input_row_ids=row_ids,
        records=tuple(
            SimpleNamespace(input_row_id=row_id) for row_id in row_ids
        ),
    )
    runtime_receipt_without_id: dict[str, object] = {
        "schema_version": RECOGNIZER_RUNTIME_RECEIPT_SCHEMA_VERSION_V2,
        "disposition": "COMPLETE",
        "reason": "phase2b_recognizer_runtime_v2_complete",
        "input_archive_id": input_archive.archive_id,
        "input_archive_sha256": executor._sha256(input_payload),
        "execution_freeze_manifest_id": freeze_id,
        "execution_freeze_manifest_sha256": executor._sha256(manifest_payload),
        "prediction_archive_id": prediction_archive.archive_id,
        "prediction_archive_sha256": executor._sha256(prediction_payload),
        "prediction_case_count": TOTAL_RECOGNIZER_CASE_COUNT,
        "prediction_archive_bytes": len(prediction_payload),
        "output_filename": RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2,
    }
    runtime_receipt = dict(runtime_receipt_without_id)
    runtime_receipt["receipt_id"] = stable_hash(
        runtime_receipt_without_id,
        prefix="phase2b_recognizer_runtime_receipt_v2_",
    )
    runtime_receipt_payload = executor._canonical_ascii_json_line_v3(
        runtime_receipt
    )
    result_wire = _result_wire(runtime_receipt_payload, prediction_payload)
    lifecycle: list[tuple[str, ...]] = []
    prepared_roots: list[Path] = []
    validated_roots: list[Path] = []
    container_id = SHA_C
    removed = False

    class FakeRuntimeOwner:
        def __init__(
            self,
            *,
            prefix: str,
            repository_root: Path,
            parent: Path,
        ) -> None:
            del repository_root
            self.owner = tempfile.TemporaryDirectory(
                prefix=prefix,
                dir=parent,
            )

        def __enter__(self) -> str:
            return self.owner.name

        def __exit__(self, *_args: object) -> bool:
            self.owner.cleanup()
            return False

    def fake_prepare(
        runtime_root: Path,
        *,
        repository_root: Path,
    ) -> LocalDockerControlPlaneV1:
        del repository_root
        prepared_roots.append(runtime_root)
        config = runtime_root / "docker-config"
        home = runtime_root / "docker-home"
        config.mkdir()
        home.mkdir()
        (config / "config.json").write_bytes(b"{}\n")
        return LocalDockerControlPlaneV1(
            executable=executor.DEFAULT_DOCKER_EXECUTABLE,
            socket_path=executor.DEFAULT_DOCKER_SOCKET,
            config_directory=config,
            environment=MappingProxyType({"PATH": "/usr/bin:/bin"}),
            binding=MappingProxyType(
                {
                    "docker_executable_sha256": executor._sha256(
                        executor.DEFAULT_DOCKER_EXECUTABLE.read_bytes()
                    )
                }
            ),
        )

    def fake_validate_control_plane(
        _control_plane: LocalDockerControlPlaneV1,
        *,
        runtime_root: Path,
    ) -> None:
        validated_roots.append(runtime_root)

    def completed(
        command: tuple[str, ...],
        *,
        returncode: int,
        stdout: bytes,
        stderr: bytes = b"",
    ) -> executor._BoundedCompletedProcessV3:
        now = executor.time.monotonic_ns()
        return executor._BoundedCompletedProcessV3(
            command=command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            started_monotonic_ns=now,
            finished_monotonic_ns=now,
        )

    def fake_run(
        command: object,
        **_kwargs: object,
    ) -> executor._BoundedCompletedProcessV3:
        nonlocal removed
        vector = tuple(command)
        lifecycle.append(vector)
        arguments = vector[2:]
        if arguments[:2] == ("image", "inspect"):
            return completed(
                vector,
                returncode=0,
                stdout=_json_bytes(_image_inspect_value()),
            )
        if arguments[:2] == ("container", "create"):
            return completed(
                vector,
                returncode=0,
                stdout=(container_id + "\n").encode("ascii"),
            )
        if arguments[:2] == ("container", "inspect") and not removed:
            name = next(
                item.removeprefix("--name=")
                for item in lifecycle[-2]
                if item.startswith("--name=")
            )
            mount = next(
                item for item in lifecycle[-2] if item.startswith("--mount=")
            )
            staged = Path(
                mount.removeprefix("--mount=type=bind,src=").split(
                    ",dst=", 1
                )[0]
            )
            seccomp_argument = next(
                item
                for item in lifecycle[-2]
                if item.startswith("--security-opt=seccomp=")
            )
            seccomp_path = Path(
                seccomp_argument.removeprefix("--security-opt=seccomp=")
            )
            value = _created_container_inspect_value(
                container_id=container_id,
                container_name=name,
                staged_input=staged,
            )
            value[0]["HostConfig"]["SecurityOpt"][1] = "seccomp=" + json.dumps(
                json.loads(seccomp_path.read_bytes()),
                sort_keys=True,
                separators=(",", ":"),
            )
            return completed(
                vector,
                returncode=0,
                stdout=_json_bytes(value),
            )
        if arguments[:1] == ("start",):
            return completed(
                vector,
                returncode=0,
                stdout=(container_id + "\n").encode("ascii"),
            )
        if arguments[:2] == ("container", "exec"):
            return completed(vector, returncode=0, stdout=result_wire)
        if arguments[:3] == ("container", "rm", "--force"):
            removed = True
            return completed(
                vector,
                returncode=0,
                stdout=(container_id + "\n").encode("ascii"),
            )
        if arguments[:2] == ("container", "inspect") and removed:
            absent = (
                f"Error response from daemon: No such container: {container_id}\n"
            ).encode("ascii")
            return completed(
                vector,
                returncode=1,
                stdout=b"[]\n",
                stderr=absent,
            )
        raise AssertionError(vector)

    def decode_input(payload: bytes) -> DecodedRecognizerInputArchiveV2:
        assert payload == input_payload
        return input_archive

    def decode_prediction(payload: bytes) -> DecodedRecognizerPredictionArchiveV2:
        assert payload == prediction_payload
        return prediction_archive

    monkeypatch.setattr(
        executor,
        "validate_linux_local_host_path_v1",
        lambda path, **_kwargs: {"path": path.as_posix()},
    )
    monkeypatch.setattr(
        executor,
        "LinuxLocalTemporaryDirectoryV1",
        FakeRuntimeOwner,
    )
    monkeypatch.setattr(
        executor,
        "prepare_local_docker_control_plane_v1",
        fake_prepare,
    )
    monkeypatch.setattr(
        executor,
        "_validate_control_plane_v3",
        fake_validate_control_plane,
    )
    monkeypatch.setattr(executor, "_run_bounded_command_v3", fake_run)
    monkeypatch.setattr(
        executor,
        "decode_execution_freeze_manifest_json_v2",
        lambda payload: execution_manifest if payload == manifest_payload else None,
    )
    monkeypatch.setattr(
        executor,
        "decode_public_recognizer_input_archive_v2",
        decode_input,
    )
    monkeypatch.setattr(
        executor,
        "decode_public_recognizer_prediction_archive_v2",
        decode_prediction,
    )

    receipt = executor.execute_phase2b_recognizer_docker_v3(
        spec,
        runtime_parent=runtime_parent,
    )

    assert receipt.image_digest == IMAGE_A
    assert receipt.local_mechanics_complete is True
    assert receipt.actual_960_case_run_verified is False
    assert receipt.actor_profile_authority_verified is False
    assert receipt.scoring_performed is False
    assert receipt.prediction_archive_sha256 == executor._sha256(
        prediction_payload
    )
    assert (
        output_path / RECOGNIZER_PREDICTION_ARCHIVE_FILENAME_V2
    ).read_bytes() == prediction_payload
    assert not any(
        item.name.startswith("hegel-phase2b-recognizer-v3-")
        for item in runtime_parent.iterdir()
    )
    assert prepared_roots == validated_roots
    assert len(prepared_roots) == 1
    assert [command[2:4] for command in lifecycle] == [
        ("image", "inspect"),
        ("container", "create"),
        ("container", "inspect"),
        ("start", container_id),
        ("container", "exec"),
        ("container", "rm"),
        ("container", "inspect"),
    ]


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("cpus", "3.0"),
        ("memory", "3g"),
        ("pids", 129),
        ("timeout_seconds", 3_599),
        ("tmpfs_bytes", 268_435_455),
        ("maximum_output_bytes", 536_870_911),
    ),
)
def test_executor_revalidates_runtime_limits_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    spec = _run_spec()
    object.__setattr__(spec.limits, field, replacement)
    invoked: list[bool] = []
    monkeypatch.setattr(
        executor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: invoked.append(True),
    )
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3
    ) as captured:
        executor.execute_phase2b_recognizer_docker_v3(spec)
    assert captured.value.code == executor.FAIL_PREFLIGHT
    assert invoked == []


def test_executor_rejects_nonfinal_image_digest_before_host_or_docker_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _run_spec()
    object.__setattr__(spec, "image_digest", "sha256:" + SHA_B)
    invoked: list[bool] = []
    monkeypatch.setattr(
        executor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: invoked.append(True),
    )
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3
    ) as captured:
        executor.execute_phase2b_recognizer_docker_v3(spec)
    assert captured.value.code == executor.FAIL_IMAGE_BINDING
    assert "final V3 recognizer image digest" in captured.value.detail
    assert invoked == []
    body = "".join(
        inspect.getsource(executor.execute_phase2b_recognizer_docker_v3).split()
    )
    assert body.index(
        "spec.image_digest!=RECOGNIZER_RUNTIME_IMAGE_DIGEST_V3"
    ) < body.index("runtime_parent=_exact_absolute_path")


def test_executor_rejects_comma_path_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _run_spec()
    object.__setattr__(spec, "input_host_directory", "/sealed/phase2b/in,put")
    invoked: list[bool] = []
    monkeypatch.setattr(
        executor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: invoked.append(True),
    )
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3
    ) as captured:
        executor.execute_phase2b_recognizer_docker_v3(spec)
    assert captured.value.code == executor.FAIL_HOST_PATH
    assert invoked == []


@pytest.mark.parametrize("nested", (False, True))
def test_runtime_parent_inside_input_is_rejected_before_any_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nested: bool,
) -> None:
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    repository_path = tmp_path / "repository"
    input_path.mkdir()
    output_path.mkdir()
    repository_path.mkdir()
    runtime_parent = input_path / "runtime" if nested else input_path
    if nested:
        runtime_parent.mkdir()
    spec = _run_spec()
    object.__setattr__(spec, "input_host_directory", input_path.as_posix())
    object.__setattr__(spec, "output_host_directory", output_path.as_posix())
    object.__setattr__(spec, "repository_host_directory", repository_path.as_posix())
    object.__setattr__(
        spec,
        "seccomp_profile_host_path",
        (
            repository_path
            / "config"
            / "phase3_internal_actor_seccomp_v1.json"
        ).as_posix(),
    )
    monkeypatch.setattr(
        executor,
        "validate_linux_local_host_path_v1",
        lambda path, **_kwargs: {"path": path.as_posix()},
    )
    invoked: list[bool] = []
    monkeypatch.setattr(
        executor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: invoked.append(True),
    )
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3,
        match="runtime parent equals or is nested inside",
    ) as captured:
        executor.execute_phase2b_recognizer_docker_v3(
            spec,
            runtime_parent=runtime_parent,
        )
    assert captured.value.code == executor.FAIL_HOST_PATH
    assert invoked == []


@pytest.mark.parametrize(
    "runtime_parent",
    (
        Path("tmp/hegel-runtime"),
        Path("/tmp/../tmp/hegel-runtime"),
        Path("/tmp/hegel,runtime"),
    ),
)
def test_executor_rejects_noncanonical_runtime_parent_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    runtime_parent: Path,
) -> None:
    invoked: list[bool] = []
    monkeypatch.setattr(
        executor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: invoked.append(True),
    )
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3
    ) as captured:
        executor.execute_phase2b_recognizer_docker_v3(
            _run_spec(),
            runtime_parent=runtime_parent,
        )
    assert captured.value.code == executor.FAIL_HOST_PATH
    assert invoked == []


def test_executor_rejects_wrong_spec_before_any_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoked: list[bool] = []
    monkeypatch.setattr(
        executor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: invoked.append(True),
    )
    with pytest.raises(
        executor.Phase2BRecognizerHostExecutorErrorV3
    ) as captured:
        executor.execute_phase2b_recognizer_docker_v3(object())
    assert captured.value.code == executor.FAIL_PREFLIGHT
    assert invoked == []
