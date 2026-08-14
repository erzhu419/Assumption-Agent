from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

import hegel_machine.phase2b_scorer_host_executor_v1 as executor
from hegel_machine.phase2b_c1_container_actor_profile_v2 import (
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256,
    PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID,
)
from hegel_machine.phase2b_scorer_bundle_v1 import (
    DecodedPhase2BScorerBundleV1,
    EvaluatedPhase2BScorerBundleV1,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
IMAGE_A = "sha256:" + SHA_A
SOURCE = Path(executor.__file__).read_text(encoding="utf-8")


def _spec(**changes: object) -> executor.Phase2BScorerRunSpecV1:
    values: dict[str, object] = {
        "authority_requirement_id": executor.OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V1,
        "actor_profile_content_id": PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID,
        "actor_profile_canonical_sha256": (
            PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
        ),
        "campaign_id": executor.ACTUAL_960_CAMPAIGN_ID_PREFIX_V2 + SHA_A,
        "attempt_id": executor.ACTUAL_960_ATTEMPT_ID_PREFIX_V2 + SHA_A,
        "execution_registry_id": (
            executor.OWNER_ACCEPTED_EXECUTION_REGISTRY_ID_PREFIX_V1 + SHA_A
        ),
        "scorer_image_digest": IMAGE_A,
        "scorer_source_root_sha256": executor.SCORER_SOURCE_ROOT_SHA256_V1,
        "protocol_id": "phase2b_protocol_" + SHA_A,
        "exact_freeze_id": "phase2b_exact_freeze_" + SHA_A,
        "execution_freeze_manifest_id": "phase2b_execution_freeze_" + SHA_A,
        "scorer_run_manifest_id": "phase2b_scorer_run_manifest_v1_" + SHA_A,
        "bundle_sha256": SHA_B,
        "bundle_host_path": "/sealed/phase2b/phase2b_scorer_bundle_v1.bin",
        "output_host_directory": "/sealed/phase2b/output",
        "repository_host_directory": "/workspace/hegel-machine",
        "seccomp_profile_host_path": (
            "/workspace/hegel-machine/config/phase3_internal_actor_seccomp_v1.json"
        ),
        "seccomp_profile_sha256": executor.SCORER_SECCOMP_SHA256_V1,
    }
    values.update(changes)
    return executor.Phase2BScorerRunSpecV1(**values)


def _private(cls: type[object], **values: object) -> object:
    value = object.__new__(cls)
    for name, item in values.items():
        object.__setattr__(value, name, item)
    return value


def _evaluation_fixture(bundle_payload: bytes) -> tuple[object, object, bytes, bytes]:
    metric_names = tuple(
        name for name, _denominator in executor._EXPECTED_METRIC_SURFACE_V1
    )
    metric_results = tuple(
        SimpleNamespace(
            metric_name=name,
            expected_denominator=denominator,
            observed_denominator=denominator,
        )
        for name, denominator in executor._EXPECTED_METRIC_SURFACE_V1
    )
    metric_outcomes = tuple(SimpleNamespace(metric_name=name) for name in metric_names)
    decoded = _private(
        DecodedPhase2BScorerBundleV1,
        bundle_id="phase2b_scorer_bundle_v1_" + hashlib.sha256(bundle_payload).hexdigest(),
        bundle_sha256=hashlib.sha256(bundle_payload).hexdigest(),
        bundle_bytes=bundle_payload,
        authority_requirement_id=executor.OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V1,
        actor_profile_content_id=PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID,
        actor_profile_canonical_sha256=(
            PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
        ),
        campaign_id=executor.ACTUAL_960_CAMPAIGN_ID_PREFIX_V2 + SHA_A,
        attempt_id=executor.ACTUAL_960_ATTEMPT_ID_PREFIX_V2 + SHA_A,
        execution_registry_id=(
            executor.OWNER_ACCEPTED_EXECUTION_REGISTRY_ID_PREFIX_V1 + SHA_A
        ),
        scorer_image_digest=IMAGE_A,
        scorer_source_root_sha256=executor.SCORER_SOURCE_ROOT_SHA256_V1,
        protocol_id="phase2b_protocol_" + SHA_A,
        exact_freeze_id="phase2b_exact_freeze_" + SHA_A,
        execution_freeze_manifest_id="phase2b_execution_freeze_" + SHA_A,
        scoring_source_sha256=executor.SCORING_MECHANICS_SOURCE_SHA256_V1,
        run_manifest_id="phase2b_scorer_run_manifest_v1_" + SHA_A,
        execution_freeze_manifest=SimpleNamespace(
            manifest_id="phase2b_execution_freeze_" + SHA_A,
            protocol_id="phase2b_protocol_" + SHA_A,
        ),
        partition_manifest=SimpleNamespace(
            exact_freeze_id="phase2b_exact_freeze_" + SHA_A,
        ),
    )
    score = SimpleNamespace(
        result_id="score_result_" + SHA_A,
        protocol_id="phase2b_protocol_" + SHA_A,
        execution_freeze_manifest_id="phase2b_execution_freeze_" + SHA_A,
        total_prediction_count=960,
        main_row_result_count=720,
        metric_eligible_main_row_count=336,
        control_row_without_frozen_metric_count=384,
        semantic_conflict_excluded_count=240,
        metric_results=metric_results,
        main_row_results=tuple(
            SimpleNamespace(metric_outcomes=metric_outcomes) for _ in range(720)
        ),
        scoring_performed=False,
        prediction_scored=False,
    )
    replay = SimpleNamespace(
        result_id="replay_result_" + SHA_A,
        exact_freeze_id="phase2b_exact_freeze_" + SHA_A,
    )
    gate = SimpleNamespace(
        result_id="gate_result_" + SHA_A,
        protocol_id="phase2b_protocol_" + SHA_A,
        overall_result_count=10,
        slice_result_count=24,
        unavailable_result_count=2,
        available_overall_gate_mechanics_results=tuple(
            SimpleNamespace(metric_name=name)
            for name in executor._EXPECTED_OVERALL_GATE_NAMES_V1
        ),
        available_slice_gate_mechanics_results=tuple(
            SimpleNamespace(metric_name=name, scope=scope, slice_id=slice_id)
            for name, scope, slice_id in executor._EXPECTED_SLICE_GATE_SURFACE_V1
        ),
        unavailable_gate_mechanics=tuple(
            SimpleNamespace(metric_name=name)
            for name in executor._EXPECTED_UNAVAILABLE_GATE_NAMES_V1
        ),
        formal_gate_evaluation_performed=False,
        scoring_performed=False,
        actual_960_case_run_verified=False,
    )
    score_report = b'{"report_kind":"score"}\n'
    gate_report = b'{"report_kind":"gate"}\n'
    score_report_id = executor.SCORE_REPORT_ID_PREFIX_V1 + hashlib.sha256(
        executor.SCORE_REPORT_DOMAIN_V1 + score_report
    ).hexdigest()
    gate_report_id = (
        executor.GATE_REPORT_ID_PREFIX_V1
        + hashlib.sha256(
            executor.GATE_REPORT_DOMAIN_V1 + gate_report
        ).hexdigest()
    )
    evaluation = _private(
        EvaluatedPhase2BScorerBundleV1,
        scoring_mechanics=score,
        replay_input_contract=replay,
        available_gate_mechanics=gate,
        score_report_bytes=score_report,
        available_gate_report_bytes=gate_report,
        score_report_id=score_report_id,
        available_gate_report_id=gate_report_id,
    )
    return decoded, evaluation, score_report, gate_report


def _image_inspect() -> bytes:
    value = [
        {
            "Id": IMAGE_A,
            "Os": "linux",
            "Architecture": "amd64",
            "Config": {
                "ArgsEscaped": True,
                "User": executor.SCORER_RUNTIME_UID_GID_V1,
                "Env": list(executor._IMAGE_ENVIRONMENT_V1),
                "Entrypoint": [executor.FROZEN_SCORER_ENTRYPOINT_V1[0]],
                "Cmd": list(executor.FROZEN_SCORER_ENTRYPOINT_V1[1:]),
                "WorkingDir": "/opt/hegel",
                "Labels": {
                    "org.opencontainers.image.title": executor.SCORER_RUNTIME_IMAGE_TITLE_V1,
                    "org.opencontainers.image.description": executor.SCORER_RUNTIME_IMAGE_DESCRIPTION_V1,
                    "org.opencontainers.image.base.digest": executor.SCORER_RUNTIME_BASE_DIGEST_V1,
                    "org.hegel-machine.phase2b.actor-profile.content-id": (
                        PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
                    ),
                    "org.hegel-machine.phase2b.actor-profile.canonical-sha256": (
                        PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
                    ),
                    "org.hegel-machine.phase2b.scoring-source.sha256": (
                        executor.SCORING_MECHANICS_SOURCE_SHA256_V1
                    ),
                    "org.hegel-machine.phase2b.scorer-source-root.sha256": (
                        executor.SCORER_SOURCE_ROOT_SHA256_V1
                    ),
                },
            },
            "RootFS": {"Type": "layers", "Layers": ["sha256:" + SHA_C]},
        }
    ]
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _runtime_receipt(
    bundle_payload: bytes,
    decoded: object,
    evaluation: object,
    score_report: bytes,
    gate_report: bytes,
) -> bytes:
    body: dict[str, object] = {
        "schema_version": executor.SCORER_RUNTIME_RECEIPT_SCHEMA_VERSION_V1,
        "claim_level": executor.SCORER_RUNTIME_CLAIM_LEVEL_V1,
        "reason": executor.SCORER_RUNTIME_COMPLETE_REASON_V1,
        "bundle_filename": executor.PHASE2B_SCORER_BUNDLE_V1_FILENAME,
        "bundle_id": decoded.bundle_id,
        "bundle_sha256": hashlib.sha256(bundle_payload).hexdigest(),
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
        "scoring_source_sha256": executor.SCORING_MECHANICS_SOURCE_SHA256_V1,
        "score_mechanics_result_id": evaluation.scoring_mechanics.result_id,
        "replay_input_contract_result_id": evaluation.replay_input_contract.result_id,
        "available_gate_mechanics_result_id": evaluation.available_gate_mechanics.result_id,
        "score_report_filename": executor.PHASE2B_SCORER_SCORE_MECHANICS_REPORT_V1_FILENAME,
        "score_report_id": evaluation.score_report_id,
        "score_report_sha256": hashlib.sha256(score_report).hexdigest(),
        "score_report_bytes": len(score_report),
        "available_gate_report_filename": (
            executor.PHASE2B_SCORER_AVAILABLE_GATE_MECHANICS_REPORT_V1_FILENAME
        ),
        "available_gate_report_id": evaluation.available_gate_report_id,
        "available_gate_report_sha256": hashlib.sha256(gate_report).hexdigest(),
        "available_gate_report_bytes": len(gate_report),
        "local_container_mechanics_complete": True,
        "strict_input_prediction_replayed": True,
        "public_partition_rebuilt": True,
        "nine_metric_mechanics_materialized": True,
        "ten_overall_gate_mechanics_materialized": True,
        "twenty_four_slice_gate_mechanics_materialized": True,
        "two_unavailable_gate_inputs_retained": True,
        "reveal_authority_verified": False,
        "prediction_commit_before_reveal_verified": False,
        "scoring_performed": False,
        "prediction_scored": False,
        "formal_gate_evaluation_performed": False,
        "actual_960_case_run_verified": False,
        "effect_evidence": False,
        "formal_c1_report_verified": False,
        "c1_exit_evidence": False,
    }
    canonical = json.dumps(
        body,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    body["receipt_id"] = executor.SCORER_RUNTIME_RECEIPT_ID_PREFIX_V1 + hashlib.sha256(
        b"HEGEL/PHASE2B/SCORER_RUNTIME_RECEIPT/V1\x00" + canonical
    ).hexdigest()
    return executor._canonical_ascii_json_line_v1(body)


def _created_container_inspect(
    *,
    container_id: str,
    container_name: str,
    staged_input: Path,
    seccomp: dict[str, object],
) -> bytes:
    seccomp_wire = json.dumps(seccomp, sort_keys=True, separators=(",", ":"))
    value = [
        {
            "Id": container_id,
            "Name": "/" + container_name,
            "Image": IMAGE_A,
            "Path": executor._CONTAINER_SUPERVISOR_ENTRYPOINT_V1,
            "Args": [
                "-c",
                executor._CONTAINER_SUPERVISOR_CODE_V1,
                *executor.FROZEN_SCORER_ENTRYPOINT_V1[1:],
            ],
            "Config": {
                "User": executor.SCORER_RUNTIME_UID_GID_V1,
                "Image": IMAGE_A,
                "WorkingDir": "/opt/hegel",
                "Env": list(executor._CONTAINER_ENVIRONMENT_V1),
                "Entrypoint": [executor._CONTAINER_SUPERVISOR_ENTRYPOINT_V1],
                "Cmd": [
                    "-c",
                    executor._CONTAINER_SUPERVISOR_CODE_V1,
                    *executor.FROZEN_SCORER_ENTRYPOINT_V1[1:],
                ],
                "Healthcheck": {"Test": ["NONE"]},
                "Volumes": None,
            },
            "State": {"Status": "created", "Running": False},
            "HostConfig": {
                "ReadonlyRootfs": True,
                "NetworkMode": "none",
                "IpcMode": "none",
                "Privileged": False,
                "AutoRemove": False,
                "CapDrop": ["ALL"],
                "NanoCpus": 2_000_000_000,
                "Memory": 2_147_483_648,
                "MemorySwap": 2_147_483_648,
                "PidsLimit": 128,
                "CgroupnsMode": "private",
                "PidMode": "",
                "UsernsMode": "",
                "Runtime": "runc",
                "SecurityOpt": [
                    "no-new-privileges:true",
                    "seccomp=" + seccomp_wire,
                ],
                "Tmpfs": {
                    "/tmp": (
                        "rw,noexec,nosuid,nodev,mode=0700,uid=65534,gid=65534,"
                        "size=268435456"
                    ),
                    "/phase2b/output": (
                        "rw,noexec,nosuid,nodev,mode=0700,uid=65534,gid=65534,"
                        "size=536870912"
                    ),
                },
                "Ulimits": [
                    {
                        "Name": "fsize",
                        "Hard": executor.MAXIMUM_SCORER_REPORT_BYTES_V1,
                        "Soft": executor.MAXIMUM_SCORER_REPORT_BYTES_V1,
                    },
                    {"Name": "nofile", "Hard": 64, "Soft": 64},
                    {"Name": "nproc", "Hard": 128, "Soft": 128},
                ],
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
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _local_receipt(**changes: object) -> executor.Phase2BScorerLocalExecutionReceiptV1:
    true_names = {
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
    }
    false_names = {
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
    }
    values: dict[str, object] = {}
    for item in fields(executor.Phase2BScorerLocalExecutionReceiptV1):
        name = item.name
        if name == "receipt_id":
            continue
        if name in true_names:
            values[name] = True
        elif name in false_names:
            values[name] = False
        elif name == "schema_version":
            values[name] = executor.LOCAL_SCORER_EXECUTION_RECEIPT_SCHEMA_V1
        elif name == "disposition":
            values[name] = executor.LOCAL_SCORER_EXECUTION_DISPOSITION_V1
        elif name == "reason":
            values[name] = executor.LOCAL_SCORER_EXECUTION_COMPLETE_REASON_V1
        elif name == "actor_profile_content_id":
            values[name] = PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
        elif name == "actor_profile_canonical_sha256":
            values[name] = PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
        elif name == "authority_requirement_id":
            values[name] = executor.OWNER_ACCEPTED_AUTHORITY_REQUIREMENT_ID_V1
        elif name == "campaign_id":
            values[name] = executor.ACTUAL_960_CAMPAIGN_ID_PREFIX_V2 + SHA_A
        elif name == "attempt_id":
            values[name] = executor.ACTUAL_960_ATTEMPT_ID_PREFIX_V2 + SHA_A
        elif name == "execution_registry_id":
            values[name] = (
                executor.OWNER_ACCEPTED_EXECUTION_REGISTRY_ID_PREFIX_V1 + SHA_A
            )
        elif name in {"scorer_image_digest", "observed_image_id"}:
            values[name] = IMAGE_A
        elif name == "scorer_source_root_sha256":
            values[name] = executor.SCORER_SOURCE_ROOT_SHA256_V1
        elif name == "scoring_mechanics_source_sha256":
            values[name] = executor.SCORING_MECHANICS_SOURCE_SHA256_V1
        elif name == "seccomp_profile_sha256":
            values[name] = executor.SCORER_SECCOMP_SHA256_V1
        elif name in {
            "bundle_bytes",
            "score_report_bytes",
            "available_gate_report_bytes",
            "started_monotonic_ns",
            "finished_monotonic_ns",
        }:
            values[name] = 2
        elif name == "process_exit_code":
            values[name] = 0
        elif name.endswith("_sha256"):
            values[name] = SHA_A
        else:
            values[name] = name + "_" + SHA_A
    values.update(changes)
    return executor.Phase2BScorerLocalExecutionReceiptV1._issue(**values)


def test_run_spec_is_exact_frozen_and_profile_bound() -> None:
    spec = _spec()
    assert spec.actor_profile_content_id == PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CONTENT_ID
    assert spec.scoring_mechanics_source_sha256 == (
        executor.SCORING_MECHANICS_SOURCE_SHA256_V1
    )
    assert spec.scorer_source_root_sha256 == executor.SCORER_SOURCE_ROOT_SHA256_V1
    assert spec.limits == executor.Phase2BScorerRuntimeLimitsV1()
    with pytest.raises(FrozenInstanceError):
        spec.scorer_image_digest = "sha256:" + SHA_B  # type: ignore[misc]
    with pytest.raises(ValueError, match="frozen binding"):
        _spec(actor_profile_content_id="phase2b_c1_container_actor_profile_v2_" + SHA_B)
    with pytest.raises(ValueError, match="frozen binding"):
        _spec(scorer_source_root_sha256=SHA_C)
    with pytest.raises(ValueError, match="frozen values"):
        executor.Phase2BScorerRuntimeLimitsV1(timeout_seconds=1)


def test_image_inspect_binds_exact_profile_and_scoring_labels() -> None:
    assert executor.SCORER_RUNTIME_IMAGE_ARGS_ESCAPED_V1 is True
    assert executor.SCORER_RUNTIME_IMAGE_CONFIG_FIELDS_V1 == {
        "ArgsEscaped",
        "Cmd",
        "Entrypoint",
        "Env",
        "Labels",
        "User",
        "WorkingDir",
    }
    image_id, projection = executor._validate_image_inspect_v1(
        _image_inspect(),
        spec=_spec(),
    )
    assert image_id == IMAGE_A
    assert len(projection) == 64
    tampered = json.loads(_image_inspect())
    tampered[0]["Config"]["Labels"][
        "org.hegel-machine.phase2b.actor-profile.content-id"
    ] = "tampered"
    with pytest.raises(executor.Phase2BScorerHostExecutorErrorV1) as caught:
        executor._validate_image_inspect_v1(
            json.dumps(tampered, separators=(",", ":")).encode(),
            spec=_spec(),
        )
    assert caught.value.code == executor.FAIL_IMAGE_BINDING

    for name, value in (("ArgsEscaped", False), ("Healthcheck", None)):
        tampered = json.loads(_image_inspect())
        tampered[0]["Config"][name] = value
        with pytest.raises(executor.Phase2BScorerHostExecutorErrorV1) as caught:
            executor._validate_image_inspect_v1(
                json.dumps(tampered, separators=(",", ":")).encode(),
                spec=_spec(),
            )
        assert caught.value.code == executor.FAIL_IMAGE_BINDING


def test_bundle_binding_requires_exact_metric_and_gate_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = b"synthetic-host-binding-bundle"
    decoded, evaluation, score_report, gate_report = _evaluation_fixture(bundle)
    monkeypatch.setattr(executor, "decode_phase2b_scorer_bundle_v1", lambda value: decoded)
    monkeypatch.setattr(
        executor,
        "evaluate_phase2b_scorer_bundle_v1",
        lambda value: evaluation,
    )
    result = executor._bundle_binding_v1(
        spec=_spec(bundle_sha256=hashlib.sha256(bundle).hexdigest()),
        bundle_payload=bundle,
    )
    assert result == (decoded, evaluation, score_report, gate_report)

    evaluation.scoring_mechanics.metric_results[0].observed_denominator = 239
    with pytest.raises(
        executor.Phase2BScorerHostExecutorErrorV1,
        match="mechanics result boundary",
    ):
        executor._bundle_binding_v1(
            spec=_spec(bundle_sha256=hashlib.sha256(bundle).hexdigest()),
            bundle_payload=bundle,
        )


def test_bundle_binding_rejects_declared_gate_counts_without_result_tuples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = b"synthetic-host-gate-binding-bundle"
    decoded, evaluation, _score_report, _gate_report = _evaluation_fixture(bundle)
    evaluation.available_gate_mechanics.available_overall_gate_mechanics_results = ()
    monkeypatch.setattr(executor, "decode_phase2b_scorer_bundle_v1", lambda value: decoded)
    monkeypatch.setattr(
        executor,
        "evaluate_phase2b_scorer_bundle_v1",
        lambda value: evaluation,
    )
    with pytest.raises(
        executor.Phase2BScorerHostExecutorErrorV1,
        match="mechanics result boundary",
    ):
        executor._bundle_binding_v1(
            spec=_spec(bundle_sha256=hashlib.sha256(bundle).hexdigest()),
            bundle_payload=bundle,
        )


def test_bundle_binding_rejects_tampered_slice_name_scope_or_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = b"synthetic-host-slice-binding-bundle"
    decoded, evaluation, _score_report, _gate_report = _evaluation_fixture(bundle)
    first = evaluation.available_gate_mechanics.available_slice_gate_mechanics_results[0]
    first.scope = "scale"
    monkeypatch.setattr(executor, "decode_phase2b_scorer_bundle_v1", lambda value: decoded)
    monkeypatch.setattr(
        executor,
        "evaluate_phase2b_scorer_bundle_v1",
        lambda value: evaluation,
    )
    with pytest.raises(
        executor.Phase2BScorerHostExecutorErrorV1,
        match="mechanics result boundary",
    ):
        executor._bundle_binding_v1(
            spec=_spec(bundle_sha256=hashlib.sha256(bundle).hexdigest()),
            bundle_payload=bundle,
        )


def test_docker_create_is_fixed_no_network_readonly_nonroot_tmpfs_only() -> None:
    control = SimpleNamespace(
        command=lambda *args: [
            "/usr/bin/docker",
            "--host=unix:///var/run/docker.sock",
            *args,
        ]
    )
    command = executor._docker_create_command_v1(
        control,
        spec=_spec(),
        container_name="hegel-phase2b-scorer-" + "d" * 24,
        staged_input_directory=Path("/tmp/scorer/input"),
        staged_seccomp_path=Path("/tmp/scorer/seccomp.json"),
    )
    assert "--network=none" in command
    assert "--ipc=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--user=65534:65534" in command
    assert "--pull=never" in command
    assert any(item.startswith("--tmpfs=/phase2b/output:") for item in command)
    assert not any("output_host" in item for item in command)
    assert command[-7:] == (
        IMAGE_A,
        "-c",
        executor._CONTAINER_SUPERVISOR_CODE_V1,
        *executor.FROZEN_SCORER_ENTRYPOINT_V1[1:],
    )


def test_created_container_inspect_replays_exact_command_and_isolation() -> None:
    seccomp = {
        "defaultAction": "SCMP_ACT_ALLOW",
        "syscalls": [{"names": ["mount"], "action": "SCMP_ACT_ERRNO"}],
    }
    executor._validate_created_container_inspect_v1(
        _created_container_inspect(
            container_id=SHA_C,
            container_name="hegel-phase2b-scorer-" + "d" * 24,
            staged_input=Path("/tmp/scorer/input"),
            seccomp=seccomp,
        ),
        container_id=SHA_C,
        container_name="hegel-phase2b-scorer-" + "d" * 24,
        spec=_spec(),
        staged_input_directory=Path("/tmp/scorer/input"),
        expected_seccomp=seccomp,
    )


def test_runtime_receipt_requires_exact_identity_and_nonclaims() -> None:
    bundle = b"bundle"
    decoded, evaluation, score, gate = _evaluation_fixture(bundle)
    payload = _runtime_receipt(bundle, decoded, evaluation, score, gate)
    receipt_id = executor._validate_runtime_receipt_v1(
        payload,
        bundle_payload=bundle,
        decoded=decoded,
        evaluation=evaluation,
        score_report=score,
        gate_report=gate,
    )
    assert receipt_id.startswith(executor.SCORER_RUNTIME_RECEIPT_ID_PREFIX_V1)
    tampered = json.loads(payload)
    tampered["scoring_performed"] = True
    with pytest.raises(executor.Phase2BScorerHostExecutorErrorV1) as caught:
        executor._validate_runtime_receipt_v1(
            executor._canonical_ascii_json_line_v1(tampered),
            bundle_payload=bundle,
            decoded=decoded,
            evaluation=evaluation,
            score_report=score,
            gate_report=gate,
        )
    assert caught.value.code == executor.FAIL_OUTPUT_BINDING


def test_two_report_wire_is_closed_and_bounded() -> None:
    score = b'{"score":1}\n'
    gate = b'{"gate":1}\n'
    payload = (
        executor._CONTAINER_RESULT_WIRE_MAGIC_V1
        + struct.pack(">QQ", len(score), len(gate))
        + score
        + gate
    )
    assert executor._extract_container_result_wire_v1(payload) == (score, gate)
    with pytest.raises(executor.Phase2BScorerHostExecutorErrorV1):
        executor._extract_container_result_wire_v1(payload + b"x")


def test_local_receipt_is_private_and_keeps_actual_claims_false() -> None:
    receipt = _local_receipt()
    assert receipt.docker_invoked is True
    assert receipt.local_mechanics_complete is True
    assert receipt.actor_profile_canonical_sha256 == (
        PHASE2B_C1_CONTAINER_ACTOR_PROFILE_V2_CANONICAL_SHA256
    )
    assert (
        receipt.scoring_mechanics_source_sha256
        == executor.SCORING_MECHANICS_SOURCE_SHA256_V1
    )
    assert receipt.actual_960_case_run_verified is False
    assert receipt.scoring_performed is False
    assert receipt.formal_gate_evaluation_performed is False
    assert receipt.effect_evidence is False
    assert receipt.c1_exit_evidence is False
    with pytest.raises(TypeError, match="privately issued"):
        executor.Phase2BScorerLocalExecutionReceiptV1()
    with pytest.raises(executor.Phase2BScorerHostExecutorErrorV1):
        _local_receipt(scoring_performed=True)


def test_receipt_issuance_failure_removes_exact_published_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "output"
    output_path.mkdir(mode=0o700)
    output = executor._open_componentwise_directory_v1(
        output_path,
        label="test scorer output",
        code=executor.FAIL_OUTPUT_BINDING,
    )
    try:
        score_file, gate_file = executor._publish_reports_v1(
            output,
            score_report=b'{"report_kind":"score"}\n',
            gate_report=b'{"report_kind":"gate"}\n',
        )

        def reject_issue(cls: type[object], **values: object) -> object:
            del cls, values
            raise ValueError("injected receipt issuance failure")

        monkeypatch.setattr(
            executor.Phase2BScorerLocalExecutionReceiptV1,
            "_issue",
            classmethod(reject_issue),
        )
        with pytest.raises(ValueError, match="injected receipt issuance failure"):
            executor._issue_receipt_after_publication_v1(
                output,
                score_file,
                gate_file,
            )
        assert list(output_path.iterdir()) == []
    finally:
        output.close()


@pytest.mark.parametrize("already_absent", [False, True])
def test_container_removal_requires_exact_absence_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    already_absent: bool,
) -> None:
    container_id = (
        "hegel-phase2b-scorer-" + "a" * 24 if already_absent else "a" * 64
    )
    absent = (
        f"Error response from daemon: No such container: {container_id}\n"
    ).encode("ascii")
    observations = iter(
        (
            SimpleNamespace(
                returncode=0,
                stdout=b"" if already_absent else (container_id + "\n").encode(),
                stderr=absent if already_absent else b"",
            ),
            SimpleNamespace(returncode=1, stdout=b"[]\n", stderr=absent),
        )
    )
    monkeypatch.setattr(
        executor,
        "_run_bounded_command_v1",
        lambda *args, **kwargs: next(observations),
    )
    executor._force_remove_container_v1(
        control_plane=SimpleNamespace(command=lambda *parts: list(parts)),
        environment={},
        working_directory=tmp_path,
        container_id=container_id,
        allow_already_absent=already_absent,
    )


def test_container_removal_rejects_daemon_error_as_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container_id = "a" * 64
    observations = iter(
        (
            SimpleNamespace(returncode=1, stdout=b"", stderr=b"daemon unavailable\n"),
            SimpleNamespace(returncode=1, stdout=b"", stderr=b"daemon unavailable\n"),
        )
    )
    monkeypatch.setattr(
        executor,
        "_run_bounded_command_v1",
        lambda *args, **kwargs: next(observations),
    )
    with pytest.raises(
        executor.Phase2BScorerHostExecutorErrorV1,
        match="removal/absence proof",
    ):
        executor._force_remove_container_v1(
            control_plane=SimpleNamespace(command=lambda *parts: list(parts)),
            environment={},
            working_directory=tmp_path,
            container_id=container_id,
        )


def test_executor_preflight_and_bundle_replay_precede_first_docker_command() -> None:
    execute_offset = SOURCE.index("def execute_phase2b_scorer_docker_v1")
    body = SOURCE[execute_offset:]
    assert body.index("load_phase2b_c1_container_actor_profile_v2") < body.index(
        "_run_bounded_command_v1("
    )
    assert body.index("_bundle_binding_v1(") < body.index("_run_bounded_command_v1(")
    assert "receipt_name" not in executor._CONTAINER_SUPERVISOR_CODE_V1
    assert "runtime-receipt" not in executor._CONTAINER_SUPERVISOR_CODE_V1
    assert "sys.stdout.flush()" in executor._CONTAINER_SUPERVISOR_CODE_V1
    tree = ast.parse(SOURCE)
    module_calls = tuple(
        node for node in tree.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    )
    assert module_calls == ()
