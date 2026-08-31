from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from hegel_machine import phase3_q1_prestart_successor_candidate_v3 as successor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_ARTIFACT = (
    PROJECT_ROOT / "artifacts/phase3_q1_prestart_authoritative_amendment_v1.json"
)
MODULE_PATH = (
    PROJECT_ROOT
    / "src/hegel_machine/phase3_q1_prestart_successor_candidate_v3.py"
)


def _tag(byte: int) -> str:
    return "sha256:" + (bytes((byte,)) * 32).hex()


def _terminal_payload() -> bytes:
    value = {
        "artifact_raw_sha256": None,
        "authorization_consumption_count": 1,
        "authorization_digest": successor.Q1_START_V2_AUTHORIZATION_DIGEST,
        "capsule_digest": successor.Q1_START_V2_CAPSULE_DIGEST,
        "consumption_record_digest": successor.Q1_START_V2_CONSUMPTION_RECORD_DIGEST,
        "contract_raw_sha256": "sha256:1a3d507c7cf0354ac4238f5ad4deaf2b640b9bd5e1f25fec544868fd20e318f8",
        "logical_campaign_count": 1,
        "outcome": {
            "actor_start_count_by_actor": {"host": 0, "python": 0, "rust": 0},
            "capsule_digest": successor.Q1_START_V2_CAPSULE_DIGEST,
            "failure_code": "Q1_START_SUPERVISOR_OUTCOME_UNKNOWN",
            "failure_detail": (
                "exit=1; stderr=Traceback (most recent call last):\n"
                "  File \"/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/tools/phase3_q1_start_supervisor_v2.py\", line 3188, in <module>\n"
                "    raise SystemExit(main())\n"
                "  File \"/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/tools/phase3_q1_start_supervisor_v2.py\", line 3115, in main\n"
                "    outcome = run_supervisor_v2(spec)\n"
                "  File \"/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/tools/phase3_q1_start_supervisor_v2.py\", line 2952, in run_supervisor_v2\n"
                "    authority_replay = _replay_successor_authority(spec)\n"
                "  File \"/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/tools/phase3_q1_start_supervisor_v2.py\", line 1295, in _replay_successor_authority\n"
                "    bundle = _validate_successor_bundle(spec, bundle_payload)\n"
                "  File \"/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/tools/phase3_q1_start_supervisor_v2.py\", line 1070, in _validate_successo; supervisor outcome is not strict ASCII JSON: Expecting value: line 1 column 1 (char 0)"
            ),
            "network_accessed": False,
            "q1_gate_count": 14,
            "q1_gate_mask": 16383,
            "q1_output_slots": [None] * 8,
            "q1_state": "NOT_RUN",
            "role_evaluation_performed": False,
            "split_accessed": False,
            "status": "Q1_START_V2_OUTCOME_UNKNOWN_TERMINAL_NO_RESUME",
            "target_truth_accessed": False,
        },
        "record_digest": successor.Q1_START_V2_TERMINAL_RECORD_DIGEST,
        "schema_version": "hegel-phase3a-q1-start-terminal/2",
        "status": "Q1_START_V2_OUTCOME_UNKNOWN_TERMINAL_NO_RESUME",
    }
    payload = successor.canonical_json_bytes_v3(value)
    assert "sha256:" + __import__("hashlib").sha256(payload).hexdigest() == (
        successor.Q1_START_V2_TERMINAL_RAW_SHA256
    )
    return payload


def _inputs() -> dict[str, object]:
    manifests: dict[str, object] = {}
    for index, actor in enumerate(successor.ACTORS, 1):
        rows = [
            {
                "relative_path": successor.ACTOR_ENTRYPOINT_RELATIVE_PATHS[actor],
                "byte_count": index * 10,
                "raw_sha256": _tag(10 + index),
            },
            {
                "relative_path": f"zz-fresh/{actor}.py",
                "byte_count": index * 20,
                "raw_sha256": _tag(20 + index),
            },
        ]
        manifests[actor] = {
            "rows": rows,
            "source_manifest_root": successor.actor_source_manifest_root_v3(
                actor, rows
            ),
        }
    mount_rows = [
        {
            "actor_id": actor,
            "read_only_source_relative_paths": [
                row["relative_path"] for row in manifests[actor]["rows"]
            ],
            "target_truth_mounted": False,
            "split_mounted": False,
            "role_mounted": False,
        }
        for actor in successor.ACTORS
    ]
    return {
        "v6_scratch_observations": deepcopy(
            successor.EXPECTED_V6_SCRATCH_OBSERVATIONS
        ),
        "corrected_resource_envelope": deepcopy(
            successor.EXPECTED_CORRECTED_RESOURCE_ENVELOPE
        ),
        "source_identity": {
            "source_commit": "1" * 40,
            "source_tree": "2" * 40,
            "source_closure_root": _tag(3),
        },
        "actor_source_manifests": manifests,
        "actor_implementation_descriptors": {
            actor: {
                "implementation_id": successor.ACTOR_IMPLEMENTATION_REGISTRY[actor][0],
                "action_id": successor.ACTOR_IMPLEMENTATION_REGISTRY[actor][1],
                "entrypoint_relative_path": successor.ACTOR_ENTRYPOINT_RELATIVE_PATHS[
                    actor
                ],
                "entrypoint_raw_sha256": manifests[actor]["rows"][0]["raw_sha256"],
                "dependency_closure_root": successor.ACTOR_BOUND_ENVIRONMENT_ROOTS[
                    actor
                ]["dependency_closure_root"],
                "runtime_identity_root": successor.ACTOR_BOUND_ENVIRONMENT_ROOTS[
                    actor
                ]["runtime_identity_root"],
                "offline_isolation_root": successor.ACTOR_BOUND_ENVIRONMENT_ROOTS[
                    actor
                ]["offline_isolation_root"],
            }
            for index, actor in enumerate(successor.ACTORS, 1)
        },
        "mount_manifest": {
            "schema_version": successor.MOUNT_MANIFEST_SCHEMA_VERSION,
            "rows": mount_rows,
        },
        "input_absence_manifest": {
            "schema_version": successor.INPUT_ABSENCE_SCHEMA_VERSION,
            "formal_input_keys": [],
            "target_truth_present": False,
            "split_present": False,
            "role_present": False,
        },
        "formal_output_layout_root": successor.FORMAL_OUTPUT_LAYOUT_ROOT,
        "run_id": b"successor-run-v3",
        "created_at_unix_seconds": 1_788_113_000,
    }


def _derive(inputs: dict[str, object] | None = None) -> dict[str, object]:
    chosen = _inputs() if inputs is None else inputs
    return successor.derive_q1_prestart_successor_candidate_v3(
        BASE_ARTIFACT.read_bytes(),
        _terminal_payload(),
        **chosen,
    )


def test_derives_canonical_candidate_and_replays_every_root() -> None:
    inputs = _inputs()
    first = _derive()
    second = _derive()
    assert first == second
    assert set(first) == successor.TOP_LEVEL_KEYS
    assert first["schema_version"] == successor.SCHEMA_VERSION
    assert first["status"] == successor.STATUS
    assert first["candidate_only"] is True
    assert first["authorized"] is False
    assert first["authorization_consumption_count"] == 0
    assert first["campaign_execution_count"] == 0
    assert first["popen_count"] == 0
    assert first["published"] is False
    assert first["q1_execution_started"] is False
    assert first["intended_q1_state"] == "NOT_RUN"
    assert first["intended_q1_gate_count"] == 14
    assert first["intended_q1_gate_mask"] == 0x3FFF
    assert first["q1_output_slots"] == [None] * 8
    assert first["corrected_resource_envelope"]["endpoint"]["scratch_bytes"] == (
        134_217_728
    )
    assert first["actor_source_manifests"] == inputs["actor_source_manifests"]
    assert first["mount_manifest"] == inputs["mount_manifest"]
    assert first["input_absence_manifest"] == inputs["input_absence_manifest"]
    assert first["bundle_root"] == successor.successor_bundle_root_v3(first)
    assert successor.validate_q1_prestart_successor_candidate_v3(first) == first
    payload = successor.canonical_json_bytes_v3(first)
    assert successor.replay_q1_prestart_successor_candidate_v3(payload) == first


def test_carries_only_authoritative_gates_1_to_10_and_rebinds_11_to_14() -> None:
    base = json.loads(BASE_ARTIFACT.read_text(encoding="ascii"))
    bundle = _derive()
    rows = bundle["gate_evidence_rows"]
    assert rows[:10] == base["gate_evidence_rows"][:10]
    assert [row["gate_id"] for row in rows] == list(range(1, 15))
    assert [row["gate_name"] for row in rows[10:]] == list(
        successor.GATE_NAMES[10:]
    )
    assert rows[10:] != base["gate_evidence_rows"][10:]
    assert bundle["base_chain"] == successor._base_chain()
    assert bundle["base_chain"][
        "q1_start_v2_actor_actions_conservatively_counted"
    ] is False
    assert bundle["base_chain"]["q1_start_v2_resume_allowed"] is False


def test_rejects_wrong_resource_observation_envelope_and_stale_source() -> None:
    observations = _inputs()
    observations["v6_scratch_observations"][
        "python_actual_preflight_peak_scratch_bytes"
    ] -= 1
    with pytest.raises(
        successor.Q1PrestartSuccessorCandidateError,
        match="selected scratch observation is not the mechanical maximum",
    ):
        _derive(observations)

    envelope = _inputs()
    envelope["corrected_resource_envelope"]["endpoint"][
        "scratch_bytes"
    ] = 4_194_304
    with pytest.raises(
        successor.Q1PrestartSuccessorCandidateError,
        match="endpoint scratch guard is not pow2ceil twice the selected maximum",
    ):
        _derive(envelope)

    stale = _inputs()
    stale["source_identity"]["source_commit"] = successor.BASE_SOURCE_COMMIT
    with pytest.raises(
        successor.Q1PrestartSuccessorCandidateError,
        match="source identity is not fresh",
    ):
        _derive(stale)

    wrong_actor = _inputs()
    wrong_actor["actor_implementation_descriptors"]["python"][
        "implementation_id"
    ] = "unbound-implementation"
    with pytest.raises(
        successor.Q1PrestartSuccessorCandidateError,
        match="python implementation identity differs",
    ):
        _derive(wrong_actor)

    wrong_entrypoint = _inputs()
    other_row = wrong_entrypoint["actor_source_manifests"]["python"]["rows"][1]
    descriptor = wrong_entrypoint["actor_implementation_descriptors"]["python"]
    descriptor["entrypoint_relative_path"] = other_row["relative_path"]
    descriptor["entrypoint_raw_sha256"] = other_row["raw_sha256"]
    with pytest.raises(
        successor.Q1PrestartSuccessorCandidateError,
        match="python implementation identity differs",
    ):
        _derive(wrong_entrypoint)

    wrong_environment = _inputs()
    wrong_environment["actor_implementation_descriptors"]["rust"][
        "runtime_identity_root"
    ] = _tag(99)
    with pytest.raises(
        successor.Q1PrestartSuccessorCandidateError,
        match="rust bound dependency/runtime/isolation roots differ",
    ):
        _derive(wrong_environment)


def test_rejects_v2_output_or_scratch_path_and_manifest_splice() -> None:
    forbidden = _inputs()
    row = forbidden["actor_source_manifests"]["python"]["rows"][0]
    row["relative_path"] = "q1-start-v2-run/python-output/endpoint-result.json"
    forbidden["actor_source_manifests"]["python"]["source_manifest_root"] = (
        successor.actor_source_manifest_root_v3(
            "python", forbidden["actor_source_manifests"]["python"]["rows"]
        )
    )
    forbidden["mount_manifest"]["rows"][0]["read_only_source_relative_paths"][
        0
    ] = row["relative_path"]
    with pytest.raises(
        successor.Q1PrestartSuccessorCandidateError,
        match="V2_WORK_PRODUCT",
    ):
        _derive(forbidden)

    splice = _inputs()
    splice["actor_source_manifests"]["rust"]["source_manifest_root"] = _tag(99)
    with pytest.raises(
        successor.Q1PrestartSuccessorCandidateError,
        match="rust source manifest root differs",
    ):
        _derive(splice)


def test_rejects_wrong_predecessor_terminal_and_bundle_tampering() -> None:
    artifact = bytearray(BASE_ARTIFACT.read_bytes())
    artifact[-2] ^= 1
    with pytest.raises(
        successor.Q1PrestartSuccessorCandidateError,
        match="authoritative prestart artifact differs",
    ):
        successor.derive_q1_prestart_successor_candidate_v3(
            bytes(artifact), _terminal_payload(), **_inputs()
        )

    terminal = bytearray(_terminal_payload())
    terminal[-2] ^= 1
    with pytest.raises(
        successor.Q1PrestartSuccessorCandidateError,
        match="Q1-start-v2 terminal bytes differ",
    ):
        successor.derive_q1_prestart_successor_candidate_v3(
            BASE_ARTIFACT.read_bytes(), bytes(terminal), **_inputs()
        )

    bundle = _derive()
    bundle["gate_evidence_rows"][10]["evidence_root"] = _tag(100)
    bundle["bundle_root"] = successor.successor_bundle_root_v3(bundle)
    with pytest.raises(
        successor.Q1PrestartSuccessorCandidateError,
        match="Gate 11 evidence differs",
    ):
        successor.validate_q1_prestart_successor_candidate_v3(bundle)

    implementation_splice = _derive()
    implementation_splice["actor_implementation_descriptors"]["python"][
        "dependency_closure_root"
    ] = _tag(101)
    implementation_splice["bundle_root"] = successor.successor_bundle_root_v3(
        implementation_splice
    )
    with pytest.raises(
        successor.Q1PrestartSuccessorCandidateError,
        match="python bound dependency/runtime/isolation roots differ",
    ):
        successor.validate_q1_prestart_successor_candidate_v3(
            implementation_splice
        )


def test_standalone_replay_recomputes_manifest_mount_and_absence_preimages() -> None:
    manifest_splice = _derive()
    python_manifest = manifest_splice["actor_source_manifests"]["python"]
    python_manifest["rows"][0]["raw_sha256"] = _tag(102)
    python_manifest["source_manifest_root"] = successor.actor_source_manifest_root_v3(
        "python", python_manifest["rows"]
    )
    manifest_splice["actor_source_manifest_roots"]["python"] = python_manifest[
        "source_manifest_root"
    ]
    manifest_splice["bundle_root"] = successor.successor_bundle_root_v3(
        manifest_splice
    )
    with pytest.raises(
        successor.Q1PrestartSuccessorCandidateError,
        match="python fixed entrypoint row/hash differs",
    ):
        successor.validate_q1_prestart_successor_candidate_v3(manifest_splice)

    mount_splice = _derive()
    mount_splice["mount_manifest"]["rows"][0]["target_truth_mounted"] = True
    mount_splice["bundle_root"] = successor.successor_bundle_root_v3(
        mount_splice
    )
    with pytest.raises(
        successor.Q1PrestartSuccessorCandidateError,
        match="python mount row differs",
    ):
        successor.validate_q1_prestart_successor_candidate_v3(mount_splice)

    absence_splice = _derive()
    absence_splice["input_absence_manifest"]["formal_input_keys"] = ["target"]
    absence_splice["bundle_root"] = successor.successor_bundle_root_v3(
        absence_splice
    )
    with pytest.raises(
        successor.Q1PrestartSuccessorCandidateError,
        match="input absence manifest differs",
    ):
        successor.validate_q1_prestart_successor_candidate_v3(absence_splice)


def test_module_has_no_process_filesystem_network_or_authority_facility() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported.isdisjoint({"os", "subprocess", "socket", "urllib", "requests"})
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert calls.isdisjoint({"open", "exec", "eval"})
