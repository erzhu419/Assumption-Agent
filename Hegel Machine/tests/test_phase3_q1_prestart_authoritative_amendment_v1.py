from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from hegel_machine import phase3_q1_prestart_authoritative_amendment_v1 as core
from hegel_machine import phase3_q1_prestart_candidate_v1 as candidate
from hegel_machine import phase3_q1_prestart_wire_v1 as wire
from hegel_machine.strict_cbor_v1 import canonical_cbor_encode


def _tagged(byte: int) -> str:
    return "sha256:" + (bytes((byte,)) * 32).hex()


def _candidate_stub(
    *_payloads: bytes,
    source_identity: wire.Q1SourceIdentityV1,
    actor_identities: tuple[wire.Q1FormalActorIdentityV1, ...],
    run_id: bytes,
    created_at_unix_seconds: int,
    prestart_wire_contract_raw_sha256: bytes,
    mount_manifest_root: bytes,
) -> dict[str, object]:
    del prestart_wire_contract_raw_sha256, mount_manifest_root
    implementation_rows = tuple(
        (identity.actor_id, identity.implementation_root)
        for identity in actor_identities
    )
    genesis = wire.Q1RunGenesisV1(
        run_id=run_id,
        source_identity=source_identity,
        q1_semantic_binding_root=b"s" * 32,
        q1_projection_profile_root=b"p" * 32,
        q0_14_of_14_receipt_root=b"0" * 32,
        q05b_qualification_receipt_root=b"5" * 32,
        full_v16_leaf_manifest_root=b"l" * 32,
        final_implementation_root_rows=implementation_rows,
        created_at_unix_seconds=created_at_unix_seconds,
    )
    subjects: list[tuple[object, ...]] = []
    for gate_id, (_registry_id, specs) in enumerate(
        wire.Q1_PRESTART_GATE_SUBJECT_REGISTRY, 1
    ):
        row: list[object] = []
        for _name, kind in specs:
            if kind == b"root32":
                row.append(bytes((gate_id,)) * 32)
            elif kind == b"bytes20":
                row.append(bytes((gate_id,)) * 20)
            elif kind == b"false":
                row.append(False)
            elif kind == b"formal_resource_envelope":
                row.append(wire.FORMAL_RESOURCE_ENVELOPE)
            elif kind == b"implementation_rows":
                row.append(implementation_rows)
            elif kind == b"run_genesis_object":
                row.append(genesis.canonical_object())
            elif kind == b"null_output_slots":
                row.append(wire.q1_null_output_slots_v1())
            else:  # pragma: no cover - frozen test construction guard
                raise AssertionError(kind)
        if gate_id == 14:
            row[0] = genesis.genesis_root
        subjects.append(tuple(row))
    gate_rows = tuple(
        wire.gate_row_v1(source_identity, genesis, gate_id, row)
        for gate_id, row in enumerate(subjects, 1)
    )
    receipt = wire.Q1PrestartAdmissionReceiptV1(genesis, gate_rows)
    return {
        "action_id": candidate.ACTION_ID,
        "actor_implementation_roots": {
            identity.actor_id.decode("ascii"): "sha256:"
            + identity.implementation_root.hex()
            for identity in actor_identities
        },
        "candidate_only": True,
        "formal_output_layout_root": "sha256:"
        + wire.FORMAL_OUTPUT_LAYOUT_ROOT.hex(),
        "gate_evidence_rows": [
            {
                "evidence_root": "sha256:" + gate_rows[index - 1][3].hex(),
                "gate_id": index,
                "gate_name": gate_rows[index - 1][1].decode("ascii"),
                "subjects_cbor_hex": canonical_cbor_encode(
                    subjects[index - 1]
                ).hex(),
            }
            for index in range(1, 15)
        ],
        "intended_q1_gate_count": 14,
        "intended_q1_gate_mask": 0x3FFF,
        "intended_q1_state": "NOT_RUN",
        "network_accessed": False,
        "popen_count": 0,
        "prestart_admission_receipt_cbor_hex": receipt.canonical_bytes.hex(),
        "prestart_admission_receipt_root": "sha256:" + receipt.receipt_root.hex(),
        "published": False,
        "q1_execution_started": False,
        "q1_output_slots": [None] * 8,
        "q1_state_before_authoritative_apply": "NOT_RUN",
        "q1_gate_count_before_authoritative_apply": 0,
        "q1_gate_mask_before_authoritative_apply": 0,
        "run_genesis_cbor_hex": genesis.canonical_bytes.hex(),
        "run_genesis_root": "sha256:" + genesis.genesis_root.hex(),
        "schema_version": candidate.SCHEMA_VERSION,
        "status": candidate.STATUS,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
    }


@dataclass
class SyntheticState:
    root: Path
    contract_path: Path
    inputs_path: Path
    contract: dict[str, object]
    inputs: dict[str, object]
    capsule_digest: str

    @property
    def fixed(self) -> dict[str, str]:
        return self.contract["fixed_paths"]  # type: ignore[return-value]


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(core.canonical_json_bytes_v1(value))


def _make_state(tmp_path: Path) -> SyntheticState:
    project = tmp_path / "project"
    run_root = tmp_path / "run"
    ledger = tmp_path / "ledger"
    artifacts = project / "artifacts"
    for path in (project, run_root, ledger, artifacts):
        path.mkdir(parents=True, exist_ok=True)
    source_paths = ["source-a.py", "source-b.py"]
    for index, relative in enumerate(source_paths, 1):
        (project / relative).write_bytes(f"source-{index}\n".encode("ascii"))
    wire_contract = project / "wire.md"
    wire_contract.write_bytes(b"synthetic wire contract\n")
    evidence_rows = []
    evidence_contract = []
    for index in range(7):
        path = tmp_path / f"evidence-{index}.json"
        payload = f'{{"row":{index}}}\n'.encode("ascii")
        path.write_bytes(payload)
        label = f"input-{index}"
        evidence_contract.append(
            {"label": label, "path": str(path), "maximum_bytes": 4096}
        )
        evidence_rows.append(
            {
                "label": label,
                "path": str(path),
                "byte_count": len(payload),
                "raw_sha256": "sha256:" + __import__("hashlib").sha256(payload).hexdigest(),
            }
        )
    fixed = {
        "project_root": str(project),
        "capsule_path": str(run_root / "capsule.json"),
        "ledger_root": str(ledger),
        "authorization_path": str(ledger / "authorization.json"),
        "consumption_path": str(ledger / "consumed.json"),
        "artifact_path": str(artifacts / "prestart.json"),
    }
    contract: dict[str, object] = {
        "schema_version": core.CONTRACT_SCHEMA_VERSION,
        "contract_id": core.CONTRACT_ID,
        "status": "IMPLEMENTED_NOT_EXECUTED",
        "action_id": core.ACTION_ID,
        "capsule_inputs_schema_version": core.CAPSULE_INPUTS_SCHEMA_VERSION,
        "fixed_paths": fixed,
        "source_closure_relative_paths": source_paths,
        "formal_actor_source_manifest_relative_paths": [
            {
                "actor_id": frozen[0].decode("ascii"),
                "relative_paths": source_paths,
            }
            for frozen in wire.FORMAL_ACTOR_REGISTRY
        ],
        "prestart_wire_contract_relative_path": "wire.md",
        "evidence_inputs": evidence_contract,
        "space_policy": {"minimum_free_bytes": 1},
        "state_machine": {
            "same_digest_preconsumption_pending": True,
            "preflight_capsule_publication_atomic_no_replace": True,
            "authorization_consumption_count": 1,
            "final_artifact_publication_atomic_no_replace": True,
            "consumed_artifact_exact_read_only_replay": True,
            "consumed_artifact_absent_or_mismatch_terminal_no_resume": True,
        },
        "authority_boundary": {
            "q1_state": "NOT_RUN",
            "q1_gate_count": 14,
            "q1_gate_total": 20,
            "q1_gate_mask": 0x3FFF,
            "q1_output_slots": [None] * 8,
            "future_q1_start_requires_distinct_authorization": True,
            "q1_execution_started": False,
            "maximum_popen_count": 0,
            "network_allowed": False,
            "target_split_role_access_allowed": False,
            "adoption_promotion_current_pointer_active_allowed": False,
        },
    }
    contract_path = tmp_path / "contract.json"
    _write_json(contract_path, contract)
    source_rows = []
    for relative in source_paths:
        payload = (project / relative).read_bytes()
        source_rows.append(
            {
                "relative_path": relative,
                "byte_count": len(payload),
                "raw_sha256": "sha256:" + __import__("hashlib").sha256(payload).hexdigest(),
            }
        )
    actors = []
    for index, frozen in enumerate(wire.FORMAL_ACTOR_REGISTRY, 1):
        actor_id = frozen[0].decode("ascii")
        entrypoint = source_paths[(index - 1) % len(source_paths)]
        manifest_paths = contract[
            "formal_actor_source_manifest_relative_paths"
        ][index - 1]["relative_paths"]
        manifest_rows = [
            row for row in source_rows if row["relative_path"] in manifest_paths
        ]
        dependency_identity = {
            "schema_version": core.DEPENDENCY_IDENTITY_SCHEMA_VERSION,
            "actor_id": actor_id,
            "ordered_rows": [
                {"name": "synthetic-dependency", "identity": f"v{index}"}
            ],
        }
        runtime_identity = {
            "schema_version": core.RUNTIME_IDENTITY_SCHEMA_VERSION,
            "actor_id": actor_id,
            "runtime_kind": "synthetic",
            "runtime_identity": f"synthetic-runtime-{index}",
        }
        offline_isolation = {
            "schema_version": core.OFFLINE_ISOLATION_SCHEMA_VERSION,
            "actor_id": actor_id,
            "network_mode": "none",
            "formal_input_keys": [],
            "target_truth_mounted": False,
            "split_mounted": False,
            "role_mounted": False,
        }
        actors.append(
            {
                "actor_id": actor_id,
                "implementation_id": frozen[1].decode("ascii"),
                "action_id": frozen[2].decode("ascii"),
                "source_manifest_root": core.actor_source_manifest_root_v1(
                    actor_id, manifest_rows
                ),
                "entrypoint_relative_path": entrypoint,
                "entrypoint_raw_sha256": next(
                    row["raw_sha256"]
                    for row in manifest_rows
                    if row["relative_path"] == entrypoint
                ),
                "source_manifest_rows": manifest_rows,
                "dependency_identity": dependency_identity,
                "dependency_closure_root": core.actor_dependency_identity_root_v1(
                    dependency_identity
                ),
                "runtime_identity": runtime_identity,
                "runtime_identity_root": core.actor_runtime_identity_root_v1(
                    runtime_identity
                ),
                "offline_isolation": offline_isolation,
                "offline_isolation_root": core.actor_offline_isolation_root_v1(
                    offline_isolation
                ),
            }
        )
    mount_manifest = {
        "schema_version": core.MOUNT_MANIFEST_SCHEMA_VERSION,
        "rows": [
            {
                "actor_id": actor["actor_id"],
                "read_only_source_relative_paths": [
                    row["relative_path"]
                    for row in actor["source_manifest_rows"]
                ],
                "output_layout_root": "sha256:"
                + wire.FORMAL_OUTPUT_LAYOUT_ROOT.hex(),
            }
            for actor in actors
        ],
        "formal_input_keys": [],
        "target_truth_mounted": False,
        "split_mounted": False,
        "role_mounted": False,
        "network_enabled": False,
    }
    formal_input_key_absence = {
        "schema_version": core.INPUT_KEY_ABSENCE_SCHEMA_VERSION,
        "formal_input_keys": [],
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
    }
    inputs: dict[str, object] = {
        "schema_version": core.CAPSULE_INPUTS_SCHEMA_VERSION,
        "source_identity": {
            "source_commit": "c" * 40,
            "source_tree": "d" * 40,
            "source_closure_sha256": core.source_closure_sha256_v1(source_rows),
        },
        "source_closure_rows": source_rows,
        "formal_actor_identities": actors,
        "run_id": "ab" * 16,
        "created_at_unix_seconds": 1_788_000_000,
        "prestart_wire_contract_raw_sha256": "sha256:"
        + __import__("hashlib").sha256(wire_contract.read_bytes()).hexdigest(),
        "mount_manifest": mount_manifest,
        "mount_manifest_root": core.mount_manifest_root_v1(mount_manifest),
        "formal_input_key_absence": formal_input_key_absence,
        "formal_input_key_absence_root": core.formal_input_key_absence_root_v1(
            formal_input_key_absence
        ),
        "evidence_inputs": evidence_rows,
    }
    inputs_path = tmp_path / "capsule-inputs.json"
    _write_json(inputs_path, inputs)
    loaded = core.load_contract_v1(contract_path)
    capsule = core.build_capsule_v1(loaded, inputs)
    return SyntheticState(
        tmp_path,
        contract_path,
        inputs_path,
        contract,
        inputs,
        capsule["capsule_digest"],
    )


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SyntheticState:
    monkeypatch.setattr(candidate, "derive_q1_prestart_candidate_v1", _candidate_stub)
    return _make_state(tmp_path)


def test_preflight_publishes_post_commit_capsule_and_exact_replay(
    state: SyntheticState,
) -> None:
    first = core.preflight_v1(state.contract_path, state.inputs_path)
    assert first["status"] == core.CAPSULE_READY
    assert first["capsule_digest"] == state.capsule_digest
    assert first["read_only_replay"] is False
    capsule_path = Path(state.fixed["capsule_path"])
    original = capsule_path.read_bytes()
    second = core.preflight_v1(state.contract_path, state.inputs_path)
    assert second["status"] == core.CAPSULE_READY
    assert second["read_only_replay"] is True
    assert capsule_path.read_bytes() == original
    contract_raw = state.contract_path.read_text("ascii")
    assert '"source_commit"' not in contract_raw
    assert '"run_id"' not in contract_raw
    assert "ab" * 16 not in contract_raw


def test_missing_predecessor_is_same_digest_pending_without_state(
    state: SyntheticState,
) -> None:
    Path(state.inputs["evidence_inputs"][2]["path"]).unlink()  # type: ignore[index]
    result = core.preflight_v1(state.contract_path, state.inputs_path)
    assert result["status"] == core.PENDING_NOT_CONSUMED
    assert result["capsule_digest"] == state.capsule_digest
    assert result["same_capsule_digest_reentry_allowed"] is True
    assert result["authorization_consumption_count"] == 0
    assert result["popen_count"] == 0
    assert not Path(state.fixed["capsule_path"]).exists()
    assert not Path(state.fixed["authorization_path"]).exists()
    assert not Path(state.fixed["consumption_path"]).exists()
    assert not Path(state.fixed["artifact_path"]).exists()


def test_actor_and_mount_roots_are_recomputed_from_closed_inputs(
    state: SyntheticState,
) -> None:
    loaded = core.load_contract_v1(state.contract_path)
    assert state.inputs["formal_input_key_absence_root"] == (
        "sha256:" + candidate.FORMAL_INPUT_KEY_REGISTRY_ABSENCE_ROOT.hex()
    )
    changed_dependency = deepcopy(state.inputs)
    changed_dependency["formal_actor_identities"][0]["dependency_identity"][  # type: ignore[index]
        "ordered_rows"
    ][0]["identity"] = "different"
    with pytest.raises(core.AmendmentValidationError, match="dependency identity root"):
        core.build_capsule_v1(loaded, changed_dependency)
    changed_mount = deepcopy(state.inputs)
    changed_mount["mount_manifest"]["rows"][0][  # type: ignore[index]
        "read_only_source_relative_paths"
    ] = ["source-b.py"]
    with pytest.raises(core.AmendmentValidationError, match="mount manifest"):
        core.build_capsule_v1(loaded, changed_mount)
    changed_entrypoint = deepcopy(state.inputs)
    changed_entrypoint["formal_actor_identities"][0][  # type: ignore[index]
        "entrypoint_raw_sha256"
    ] = _tagged(99)
    with pytest.raises(core.AmendmentValidationError, match="entrypoint raw"):
        core.build_capsule_v1(loaded, changed_entrypoint)


def test_actor_manifest_cannot_omit_a_registered_nonentrypoint_source(
    state: SyntheticState,
) -> None:
    loaded = core.load_contract_v1(state.contract_path)
    changed = deepcopy(state.inputs)
    actor = changed["formal_actor_identities"][0]  # type: ignore[index]
    actor["source_manifest_rows"].pop()  # type: ignore[index]
    actor["source_manifest_root"] = core.actor_source_manifest_root_v1(
        actor["actor_id"], actor["source_manifest_rows"]  # type: ignore[index]
    )
    changed["mount_manifest"]["rows"][0][  # type: ignore[index]
        "read_only_source_relative_paths"
    ].pop()
    changed["mount_manifest_root"] = core.mount_manifest_root_v1(
        changed["mount_manifest"]  # type: ignore[index]
    )
    with pytest.raises(core.AmendmentValidationError, match="exact registry"):
        core.build_capsule_v1(loaded, changed)
    changed_absence = deepcopy(state.inputs)
    changed_absence["formal_input_key_absence"][  # type: ignore[index]
        "target_truth_accessed"
    ] = True
    with pytest.raises(core.AmendmentValidationError, match="input-key absence"):
        core.build_capsule_v1(loaded, changed_absence)


def test_source_bytes_are_replayed_before_actor_identity_is_admitted(
    state: SyntheticState,
) -> None:
    (Path(state.fixed["project_root"]) / "source-a.py").write_bytes(
        b"changed after capsule inputs\n"
    )
    result = core.preflight_v1(state.contract_path, state.inputs_path)
    assert result["status"] == core.PENDING_NOT_CONSUMED
    assert result["capsule_digest"] == state.capsule_digest
    assert result["condition"] == "SOURCE_CLOSURE_BINDING_DIFFERS"


def test_space_and_probe_failures_are_preconsumption_pending(
    state: SyntheticState, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        core.os,
        "statvfs",
        lambda _path: SimpleNamespace(f_bavail=0, f_frsize=1),
    )
    no_space = core.preflight_v1(state.contract_path, state.inputs_path)
    assert no_space["condition"] == "INSUFFICIENT_FREE_SPACE"
    assert no_space["authorization_consumption_count"] == 0
    monkeypatch.undo()
    monkeypatch.setattr(candidate, "derive_q1_prestart_candidate_v1", _candidate_stub)
    monkeypatch.setattr(
        core,
        "_probe_directory_v1",
        lambda *_args: (_ for _ in ()).throw(
            core.AmendmentPending(
                "PATH_SPACE_AND_PROBE", "DIRECTORY_PROBE_FAILED", "synthetic"
            )
        ),
    )
    probe = core.preflight_v1(state.contract_path, state.inputs_path)
    assert probe["condition"] == "DIRECTORY_PROBE_FAILED"
    assert not Path(state.fixed["capsule_path"]).exists()


def test_record_run_and_consumed_exact_read_only_replay(
    state: SyntheticState, monkeypatch: pytest.MonkeyPatch
) -> None:
    core.preflight_v1(state.contract_path, state.inputs_path)
    not_authorized = core.run_v1(state.contract_path, state.capsule_digest)
    assert not_authorized["status"] == core.NOT_AUTHORIZED
    authorization = core.record_authorization_v1(
        state.contract_path, state.capsule_digest
    )
    assert authorization["status"] == core.AUTHORIZATION_RECORDED
    assert authorization["campaign_execution_count"] == 0
    assert authorization["authorization_consumption_count"] == 0
    applied = core.run_v1(state.contract_path, state.capsule_digest)
    assert applied["status"] == core.APPLIED
    assert applied["campaign_execution_count"] == 1
    assert applied["q1_state"] == "NOT_RUN"
    assert applied["q1_gate_count"] == 14
    assert applied["q1_gate_total"] == 20
    assert applied["q1_gate_mask"] == 0x3FFF
    assert applied["q1_output_slots"] == [None] * 8
    assert applied["future_q1_start_requires_distinct_authorization"] is True
    assert applied["q1_execution_started"] is False
    assert applied["popen_count"] == 0
    consumed_path = Path(state.fixed["consumption_path"])
    artifact_path = Path(state.fixed["artifact_path"])
    consumed_bytes = consumed_path.read_bytes()
    artifact_bytes = artifact_path.read_bytes()
    monkeypatch.setattr(
        core,
        "_derive_and_replay_candidate_v1",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not derive")),
    )
    monkeypatch.setattr(
        core,
        "_probe_directory_v1",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not probe")),
    )
    replay = core.run_v1(state.contract_path, state.capsule_digest)
    assert replay["status"] == core.APPLIED
    assert replay["read_only_replay"] is True
    assert consumed_path.read_bytes() == consumed_bytes
    assert artifact_path.read_bytes() == artifact_bytes


def test_consumed_preflight_is_exact_applied_read_only_replay(
    state: SyntheticState, monkeypatch: pytest.MonkeyPatch
) -> None:
    core.preflight_v1(state.contract_path, state.inputs_path)
    core.record_authorization_v1(state.contract_path, state.capsule_digest)
    core.run_v1(state.contract_path, state.capsule_digest)
    monkeypatch.setattr(
        core,
        "_derive_and_replay_candidate_v1",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not derive")),
    )
    replay = core.preflight_v1(state.contract_path, state.inputs_path)
    assert replay["status"] == core.APPLIED
    assert replay["read_only_replay"] is True
    assert replay["campaign_execution_count"] == 1
    assert replay["authorization_consumption_count"] == 1


def test_consumed_capsule_mismatch_is_terminal_from_preflight(
    state: SyntheticState,
) -> None:
    core.preflight_v1(state.contract_path, state.inputs_path)
    core.record_authorization_v1(state.contract_path, state.capsule_digest)
    core.run_v1(state.contract_path, state.capsule_digest)
    capsule_path = Path(state.fixed["capsule_path"])
    capsule = core.strict_json_bytes_v1(capsule_path.read_bytes(), "capsule")
    capsule["q1_state"] = "BROKEN"
    _write_json(capsule_path, capsule)
    result = core.preflight_v1(state.contract_path, state.inputs_path)
    assert result["status"] == core.TERMINAL_NO_RESUME
    assert result["code"] == "CONSUMED_CAPSULE_MISMATCH"
    assert result["authorization_consumption_count"] == 1


def test_artifact_link_then_publication_error_resolves_to_success_replay(
    state: SyntheticState, monkeypatch: pytest.MonkeyPatch
) -> None:
    core.preflight_v1(state.contract_path, state.inputs_path)
    core.record_authorization_v1(state.contract_path, state.capsule_digest)
    artifact_path = Path(state.fixed["artifact_path"])
    original = core._atomic_no_replace_v1

    def publish_then_fail(path: Path, payload: bytes, digest: str) -> None:
        original(path, payload, digest)
        if path == artifact_path:
            raise OSError("synthetic post-link publication failure")

    monkeypatch.setattr(core, "_atomic_no_replace_v1", publish_then_fail)
    first = core.run_v1(state.contract_path, state.capsule_digest)
    assert first["status"] == core.APPLIED
    assert first["read_only_replay"] is True
    second = core.run_v1(state.contract_path, state.capsule_digest)
    assert second["status"] == core.APPLIED
    assert second["read_only_replay"] is True


@pytest.mark.parametrize("mutation", ["absent", "mismatch"])
def test_consumed_missing_or_mismatched_artifact_is_terminal_no_resume(
    state: SyntheticState,
    mutation: str,
) -> None:
    core.preflight_v1(state.contract_path, state.inputs_path)
    core.record_authorization_v1(state.contract_path, state.capsule_digest)
    core.run_v1(state.contract_path, state.capsule_digest)
    artifact_path = Path(state.fixed["artifact_path"])
    if mutation == "absent":
        artifact_path.unlink()
        expected_code = "CONSUMED_ARTIFACT_ABSENT"
    else:
        artifact = core.strict_json_bytes_v1(
            artifact_path.read_bytes(), "synthetic artifact"
        )
        artifact["q1_gate_mask"] = 0
        _write_json(artifact_path, artifact)
        expected_code = "CONSUMED_ARTIFACT_MISMATCH"
    result = core.run_v1(state.contract_path, state.capsule_digest)
    assert result["status"] == core.TERMINAL_NO_RESUME
    assert result["code"] == expected_code
    assert result["authorization_consumption_count"] == 1
    assert result["retry_or_resume_allowed"] is False


def test_consumed_capsule_mismatch_is_terminal_no_resume(
    state: SyntheticState,
) -> None:
    core.preflight_v1(state.contract_path, state.inputs_path)
    core.record_authorization_v1(state.contract_path, state.capsule_digest)
    core.run_v1(state.contract_path, state.capsule_digest)
    capsule_path = Path(state.fixed["capsule_path"])
    capsule = core.strict_json_bytes_v1(capsule_path.read_bytes(), "capsule")
    capsule["q1_state"] = "BROKEN"
    _write_json(capsule_path, capsule)
    result = core.run_v1(state.contract_path, state.capsule_digest)
    assert result["status"] == core.TERMINAL_NO_RESUME
    assert result["code"] == "CONSUMED_CAPSULE_MISMATCH"
    assert result["authorization_consumption_count"] == 1


def test_requested_capsule_digest_mismatch_is_terminal(
    state: SyntheticState,
) -> None:
    core.preflight_v1(state.contract_path, state.inputs_path)
    result = core.run_v1(state.contract_path, "sha256:" + "00" * 32)
    assert result["status"] == core.TERMINAL_NO_RESUME
    assert result["code"] == "CAPSULE_DIGEST_MISMATCH"
    assert result["authorization_consumption_count"] == 0


def test_preconsumption_authorization_format_error_is_pending(
    state: SyntheticState,
) -> None:
    core.preflight_v1(state.contract_path, state.inputs_path)
    Path(state.fixed["authorization_path"]).write_bytes(b"not-json\n")
    result = core.run_v1(state.contract_path, state.capsule_digest)
    assert result["status"] == core.PENDING_NOT_CONSUMED
    assert result["capsule_digest"] == state.capsule_digest
    assert result["authorization_consumption_count"] == 0


def test_exact_integer_fields_reject_bool_even_with_recomputed_digest(
    state: SyntheticState,
) -> None:
    loaded = core.load_contract_v1(state.contract_path)
    capsule = core.build_capsule_v1(loaded, state.inputs)
    capsule["intended_q1_gate_count"] = True
    unsigned = deepcopy(capsule)
    unsigned.pop("capsule_digest")
    capsule["capsule_digest"] = core._digest(core.DOMAINS["capsule"], unsigned)
    with pytest.raises(core.AmendmentValidationError, match="intended_q1_gate_count"):
        core.validate_capsule_v1(capsule, loaded)


@pytest.mark.parametrize(
    ("section", "field", "integer_value"),
    [
        ("state_machine", "same_digest_preconsumption_pending", 1),
        ("state_machine", "preflight_capsule_publication_atomic_no_replace", 1),
        ("state_machine", "final_artifact_publication_atomic_no_replace", 1),
        ("state_machine", "consumed_artifact_exact_read_only_replay", 1),
        (
            "state_machine",
            "consumed_artifact_absent_or_mismatch_terminal_no_resume",
            1,
        ),
        (
            "authority_boundary",
            "future_q1_start_requires_distinct_authorization",
            1,
        ),
        ("authority_boundary", "q1_execution_started", 0),
        ("authority_boundary", "network_allowed", 0),
        ("authority_boundary", "target_split_role_access_allowed", 0),
        (
            "authority_boundary",
            "adoption_promotion_current_pointer_active_allowed",
            0,
        ),
    ],
)
def test_contract_boolean_fields_reject_integer_aliases(
    state: SyntheticState,
    section: str,
    field: str,
    integer_value: int,
) -> None:
    changed = deepcopy(state.contract)
    changed[section][field] = integer_value  # type: ignore[index]
    with pytest.raises(core.AmendmentValidationError):
        core.validate_contract_v1(changed)


@pytest.mark.parametrize("malformed", [False, True])
def test_all_entries_return_structured_pending_for_unavailable_contract(
    state: SyntheticState,
    malformed: bool,
) -> None:
    contract_path = state.root / ("malformed.json" if malformed else "missing.json")
    if malformed:
        contract_path.write_bytes(b"not-json\n")
    results = (
        core.preflight_v1(contract_path, state.inputs_path),
        core.record_authorization_v1(contract_path, state.capsule_digest),
        core.run_v1(contract_path, state.capsule_digest),
    )
    for result in results:
        assert result["status"] == core.PENDING_NOT_CONSUMED
        assert result["condition"] == "CONTRACT_UNAVAILABLE_OR_INVALID"
        assert result["contract_raw_sha256"] is None
        assert result["authorization_consumption_count"] == 0


def test_committed_q0_small_file_bound_covers_observed_bytes() -> None:
    contract_path = (
        Path(__file__).resolve().parents[1]
        / "config/phase3_q1_prestart_authoritative_amendment_v1.json"
    )
    contract = core.load_contract_v1(contract_path).value
    q0 = contract["evidence_inputs"][0]
    assert q0["label"] == "q0_artifact"
    assert q0["maximum_bytes"] == 2_097_152
    assert q0["maximum_bytes"] >= 1_443_257


def _literal_assignment(path: Path, name: str) -> object:
    tree = ast.parse(path.read_text("utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing {name}")


def test_committed_source_closure_contains_exact_formal_actor_sources() -> None:
    project = Path(__file__).resolve().parents[1]
    contract = core.load_contract_v1(
        project / "config/phase3_q1_prestart_authoritative_amendment_v1.json"
    ).value
    closure_rows = contract["source_closure_relative_paths"]
    closure = set(closure_rows)
    python_names = _literal_assignment(
        project / "tools/phase3_q1_full_node6_formal_materializer_entrypoint_v1.py",
        "MODULE_NAMES",
    )
    python_paths = {
        f"src/hegel_machine/{name}.py" for name in python_names
    } | {"tools/phase3_q1_full_node6_formal_materializer_entrypoint_v1.py"}
    host_paths = set(
        _literal_assignment(
            project / "tools/phase3_q1_formal_host_replay_v1.py",
            "EXPECTED_MODULE_SOURCE_PATHS",
        )
    ) | {"tools/phase3_q1_formal_host_replay_v1.py"}
    rust_paths = {
        "rust/q1_full_capacity_preflight_engine/Cargo.lock",
        "rust/q1_full_capacity_preflight_engine/Cargo.toml",
        "rust/q1_full_capacity_preflight_engine/src/formal_materializer_main.rs",
        "rust/q1_full_capacity_preflight_engine/src/frozen_v1.rs",
        "rust/q1_full_capacity_preflight_engine/src/lib.rs",
        "rust/q1_archive_projection_oracle/Cargo.toml",
        "rust/q1_archive_projection_oracle/src/lib.rs",
        "rust/strict_canonicalizer/Cargo.toml",
        "rust/strict_canonicalizer/src/lib.rs",
        "rust/strict_canonicalizer_shrink1/Cargo.toml",
        "rust/strict_canonicalizer_shrink1/src/lib.rs",
        "rust/strict_canonicalizer_shrink2/Cargo.toml",
        "rust/strict_canonicalizer_shrink2/src/lib.rs",
        "rust/strict_canonicalizer_shrink3/Cargo.toml",
        "rust/strict_canonicalizer_shrink3/src/lib.rs",
        "rust/strict_canonicalizer_shrink4/Cargo.toml",
        "rust/strict_canonicalizer_shrink4/src/lib.rs",
        "rust/strict_canonicalizer_shrink5/Cargo.toml",
        "rust/strict_canonicalizer_shrink5/src/lib.rs",
        "rust/strict_canonicalizer_shrink6/Cargo.toml",
        "rust/strict_canonicalizer_shrink6/src/lib.rs",
    }
    assert python_paths <= closure
    assert host_paths <= closure
    assert rust_paths <= closure
    assert {
        "src/hegel_machine/phase3_q0_quotient_contract_v1.py",
        "src/hegel_machine/phase3_q1_semantic_coverage_v1.py",
        "src/hegel_machine/phase3_q1_prestart_authoritative_amendment_v1.py",
        "src/hegel_machine/phase3_q1_prestart_candidate_v1.py",
        "tools/phase3_q1_prestart_authoritative_amendment_v1.py",
    } <= closure
    manifest_registry = {
        row["actor_id"]: row["relative_paths"]
        for row in contract["formal_actor_source_manifest_relative_paths"]
    }
    assert manifest_registry == {
        "python": [path for path in closure_rows if path in python_paths],
        "rust": [path for path in closure_rows if path in rust_paths],
        "host": [path for path in closure_rows if path in host_paths],
    }
    assert all((project / relative).is_file() for relative in closure)


def _load_cli() -> object:
    path = (
        Path(__file__).resolve().parents[1]
        / "tools/phase3_q1_prestart_authoritative_amendment_v1.py"
    )
    spec = importlib.util.spec_from_file_location("_q1_prestart_cli_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_exact_mode_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_cli()
    with pytest.raises(SystemExit, match="2"):
        tool.main(["--preflight"])
    with pytest.raises(SystemExit, match="2"):
        tool.main(
            [
                "--run",
                "--capsule-digest",
                "sha256:" + "00" * 32,
                "--capsule-inputs",
                "unexpected.json",
            ]
        )
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        tool._core,
        "run_v1",
        lambda contract, digest: calls.append((contract, digest))
        or {"status": core.NOT_AUTHORIZED},
    )
    monkeypatch.setattr(
        tool._core,
        "canonical_json_bytes_v1",
        lambda _value: b'{}\n',
    )
    assert (
        tool.main(["--run", "--capsule-digest", "sha256:" + "11" * 32])
        == 4
    )
    assert calls == [(tool.DEFAULT_CONTRACT, "sha256:" + "11" * 32)]
