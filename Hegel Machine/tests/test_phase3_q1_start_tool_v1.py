from __future__ import annotations

from dataclasses import dataclass, replace
import fcntl
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Callable, Mapping

import pytest

from hegel_machine import phase3_q1_start_campaign_v1 as campaign
from hegel_machine import phase3_q1_start_completion_wire_v1 as wire


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "phase3_q1_start_tool_v1",
    ROOT / "tools" / "phase3_q1_start_v1.py",
)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)
supervisor = tool._load_supervisor_module()


@pytest.fixture
def tmp_path() -> Path:
    with TemporaryDirectory(prefix="q1-start-tool-", dir="/tmp") as root:
        yield Path(root)


def _sha(value: int) -> str:
    return "sha256:" + f"{value:02x}" * 32


def _root(value: int) -> bytes:
    return bytes([value]) * 32


def _contract_value(tmp_path: Path) -> dict[str, object]:
    project = tmp_path / "project"
    run = tmp_path / "run"
    ledger = tmp_path / "ledger"
    prestart = tmp_path / "prestart"
    return {
        "schema_version": campaign.CONTRACT_SCHEMA_VERSION,
        "contract_id": campaign.CONTRACT_ID,
        "status": "IMPLEMENTED_NOT_EXECUTED",
        "action_id": campaign.ACTION_ID,
        "capsule_inputs_schema_version": campaign.CAPSULE_INPUTS_SCHEMA_VERSION,
        "fixed_paths": {
            "project_root": str(project),
            "campaign_lock_path": str(tmp_path / "q1-start.lock"),
            "run_root": str(run),
            "ledger_root": str(ledger),
            "capsule_path": str(run / "capsule.json"),
            "authorization_path": str(ledger / "authorization.json"),
            "consumption_path": str(ledger / "consumption.json"),
            "checkpoint_root": str(run / "checkpoints"),
            "outcome_path": str(run / "outcome.json"),
            "terminal_path": str(ledger / "terminal.json"),
            "artifact_path": str(project / "artifacts" / "q1-complete.cbor"),
            "rust_binary_path": str(tmp_path / "build" / "formal-materializer"),
            "admission_path": str(run / "admission.json"),
            "run_spec_path": str(run / "run-spec.json"),
            "python_source_root": str(run / "python-source"),
            "rust_source_root": str(run / "rust-source"),
            "host_source_root": str(run / "host-source"),
            "python_output_path": str(run / "python-output"),
            "python_scratch_path": str(run / "python-scratch"),
            "rust_output_path": str(run / "rust-output"),
            "rust_scratch_path": str(run / "rust-scratch"),
            "host_output_path": str(run / "host-output"),
            "host_scratch_path": str(run / "host-scratch"),
        },
        "controller_source_closure_relative_paths": [
            "src/hegel_machine/phase3_q1_start_campaign_v1.py",
            "src/hegel_machine/phase3_q1_start_completion_wire_v1.py",
            "tools/phase3_q1_start_supervisor_v1.py",
            "tools/phase3_q1_start_v1.py",
        ],
        "predecessor": {
            "prestart_artifact_path": str(prestart / "artifact.json"),
            "prestart_artifact_raw_sha256": _sha(1),
            "prestart_capsule_path": str(prestart / "capsule.json"),
            "prestart_capsule_digest": _sha(2),
            "prestart_authorization_path": str(prestart / "authorization.json"),
            "prestart_authorization_digest": _sha(3),
            "prestart_consumption_path": str(prestart / "consumption.json"),
            "prestart_consumption_record_digest": _sha(4),
            "formal_source_commit": "11" * 20,
            "formal_source_tree": "22" * 20,
            "run_genesis_root": _sha(20),
            "prestart_admission_receipt_root": _sha(21),
            "actor_implementation_roots": {
                "host": _sha(35),
                "python": _sha(30),
                "rust": _sha(32),
            },
            "q1_state": "NOT_RUN",
            "q1_gate_count": 14,
            "q1_gate_mask": 0x003FFF,
            "q1_output_slots": [None] * 8,
        },
        "runtime": {
            "docker_path": "/usr/bin/docker",
            "supervisor_python_path": "/usr/bin/python3.10",
            "python_image": campaign.PYTHON_IMAGE,
            "rust_image": campaign.RUST_IMAGE,
            "seccomp_profile_path": str(project / "seccomp.json"),
            "seccomp_profile_raw_sha256": _sha(10),
            "python_cpuset": "0-11",
            "rust_cpuset": "12-23",
            "host_cpuset": "0-11",
            "endpoint_memory_bytes": 2_147_483_648,
            "endpoint_output_bytes": 16_777_216,
            "endpoint_scratch_bytes": 4_194_304,
            "endpoint_wall_seconds": 3600,
            "host_memory_bytes": 67_108_864,
            "host_output_bytes": 2_097_152,
            "host_scratch_bytes": 1,
            "host_wall_seconds": 3600,
            "pids_limit": 128,
            "minimum_available_memory_bytes": 4_294_967_296,
            "minimum_available_storage_bytes": 134_217_728,
        },
        "execution_policy": {
            "actor_order": ["python", "rust", "host"],
            "maximum_start_count_by_actor": {"host": 1, "python": 1, "rust": 1},
            "maximum_concurrently_live_actor_containers": 1,
            "clean_checkpoint_resume_allowed": True,
            "checkpoint_resume_repeats_completed_actor": False,
            "intent_or_partial_without_exact_completion_terminal_no_resume": True,
            "preconsumption_pending_consumes_authorization": False,
            "preconsumption_pending_starts_actor": False,
            "network_allowed": False,
        },
        "authority_boundary": {
            "q1_start_authorized_by_future_exact_capsule": True,
            "q1_state_before": "NOT_RUN",
            "q1_gate_count_before": 14,
            "q1_gate_mask_before": 0x003FFF,
            "q1_success_state": "COMPLETE",
            "q1_success_gate_count": 20,
            "q1_success_gate_mask": 0x0FFFFF,
            "q2_state": "NOT_RUN",
            "m3_formal_roots": None,
            "role_evaluation_performed": False,
            "outside_certificate_issued": False,
            "adoption_allowed": False,
            "promotion_allowed": False,
            "current_pointer_update_allowed": False,
            "active_transition_allowed": False,
        },
    }


def _receipt(*, host_implementation: int = 35) -> bytes:
    signatures = (
        wire.Q1SignatureCompletionRowV1(
            1, _root(1), _root(2), _root(3), wire.signature_guard_ceiling_rows_v1(1)
        ),
        wire.Q1SignatureCompletionRowV1(
            2, _root(4), _root(5), _root(6), wire.signature_guard_ceiling_rows_v1(2)
        ),
    )
    python = wire.Q1EndpointCompletionEvidenceV1(
        b"python", _root(30), _root(20), _root(21), _root(40), _root(22),
        signatures, 100, 10, 1000, 1000,
    )
    rust = wire.Q1EndpointCompletionEvidenceV1(
        b"rust", _root(32), _root(20), _root(21), _root(41), _root(22),
        signatures, 100, 10, 1000, 1000,
    )
    agreement = wire.Q1DualReplayAgreementV1(
        _root(20), _root(21), python.evidence_root, rust.evidence_root,
        _root(42), signatures, _root(22), _root(host_implementation), _root(43),
        100, 0, 1000, 1000,
    )
    actor_rows = tuple(
        wire.Q1IsolationEvidenceAdapterV1(
            actor_id=actor,
            actor_implementation_root=implementation,
            source_manifest_rows=(
                wire.Q1SourceManifestEvidenceRowV1(
                    f"actor-{index}.py", 1, _root(50 + index)
                ),
            ),
            mount_manifest_rows=(
                wire.Q1MountManifestEvidenceRowV1(
                    "/output", f"/run/{actor.decode()}/output", True
                ),
                wire.Q1MountManifestEvidenceRowV1(
                    "/scratch", f"/run/{actor.decode()}/scratch", True
                ),
                wire.Q1MountManifestEvidenceRowV1(
                    "/source", f"/run/{actor.decode()}/source", False
                ),
            ),
            output_files=(wire.Q1FileEvidenceV1("result.json", b"{}\n"),),
            scratch_path=f"/run/{actor.decode()}/scratch",
        ).isolation_row()
        for index, (actor, implementation) in enumerate(
            (
                (b"python", _root(30)),
                (b"rust", _root(32)),
                (b"host", _root(host_implementation)),
            )
        )
    )
    ledger = wire.Q1TargetBlindAccessLedgerV1(
        _root(20),
        _root(21),
        agreement.agreement_root,
        wire.runtime_mount_registry_root_v1(actor_rows),
        wire.formal_input_key_registry_absence_root_v1(),
        actor_rows,
    )
    subjects = (
        (_root(20), _root(21), (python.canonical_object(), rust.canonical_object())),
        (
            python.evidence_root, rust.evidence_root, agreement.formal_file_closure_root,
            tuple(row.canonical_object() for row in signatures), _root(22), True, True,
        ),
        (agreement.canonical_object(), agreement.agreement_root),
        (ledger.canonical_object(), ledger.ledger_root),
    )
    core = wire.Q1CompletionCoreV1(
        _root(20), _root(21), subjects,
        wire.output_slots_1_through_7_v1(agreement, ledger),
    )
    return wire.Q1CompletionReceiptV1(core).canonical_bytes


@dataclass
class Harness:
    contract_path: Path
    inputs_path: Path
    contract: campaign.LoadedContractV1
    capsule: dict[str, object]
    source_state: tool.SourceStateV1
    predecessor: tool.PredecessorReplayV1
    blobs: dict[str, bytes]
    supervisor_result: dict[str, object]
    supervisor_calls: int = 0
    live_failure: tool.ControllerPending | None = None
    completion_failure: tool.ControllerPending | None = None
    completion_payload: bytes | None = None

    def dependencies(self) -> tool.ControllerDependenciesV1:
        def live_gate(_contract: object, _capsule: Mapping[str, object]) -> None:
            if self.live_failure is not None:
                raise self.live_failure

        def invoke(_python: Path, _supervisor: Path, _spec: Path) -> Mapping[str, object]:
            self.supervisor_calls += 1
            return self.supervisor_result

        def completion_builder(*_args: object) -> bytes:
            if self.completion_failure is not None:
                raise self.completion_failure
            if self.completion_payload is not None:
                return self.completion_payload
            return _receipt()

        return tool.ControllerDependenciesV1(
            source_state=lambda _contract: self.source_state,
            replay_predecessor=lambda _contract: self.predecessor,
            formal_blob=lambda _project, _commit, relative: self.blobs[relative],
            live_gate=live_gate,
            invoke_supervisor=invoke,
            build_completion_artifact=completion_builder,
        )


def _success(capsule: Mapping[str, object], contract: campaign.LoadedContractV1) -> dict[str, object]:
    predecessor = contract.value["predecessor"]
    return {
        "schema_version": supervisor.OUTCOME_SCHEMA,
        "status": supervisor.SUCCESS_STATUS,
        "capsule_digest": capsule["capsule_digest"],
        "source_commit": predecessor["formal_source_commit"],
        "run_genesis_root": predecessor["run_genesis_root"],
        "prestart_admission_receipt_root": predecessor["prestart_admission_receipt_root"],
        "completed_stages": ["python", "rust", "host"],
        "q1_completion_receipt_issued": False,
        "state_published": False,
    }


def _persisted_success(
    capsule: Mapping[str, object], contract: campaign.LoadedContractV1
) -> dict[str, object]:
    value = _success(capsule, contract)
    checkpoint = Path(contract.value["fixed_paths"]["checkpoint_root"])
    value.update(
        {
            "checkpoint_paths": [
                str(checkpoint / "01-python-complete.json"),
                str(checkpoint / "02-rust-complete.json"),
                str(checkpoint / "03-host-complete.json"),
            ],
            "host_result": {},
            "gate15_candidate": {},
            "gate16_candidate": {},
            "gate17_candidate": {},
            "gate18_runtime_receipt_inputs": {},
        }
    )
    return value


def _pending(
    capsule: Mapping[str, object],
    contract: campaign.LoadedContractV1,
    completed: list[str],
) -> dict[str, object]:
    predecessor = contract.value["predecessor"]
    next_stage = (set(("python", "rust", "host")) - set(completed)).pop()
    return {
        "schema_version": supervisor.OUTCOME_SCHEMA,
        "status": supervisor.PENDING_STATUS,
        "capsule_digest": capsule["capsule_digest"],
        "source_commit": predecessor["formal_source_commit"],
        "run_genesis_root": predecessor["run_genesis_root"],
        "prestart_admission_receipt_root": predecessor["prestart_admission_receipt_root"],
        "next_stage": next_stage,
        "completed_stages": completed,
        "pending_code": "Q1_START_LOCAL_IMAGE_PENDING",
        "pending_detail": "pinned image temporarily unavailable",
        "next_actor_intent_absent": True,
        "actor_popen_count_this_entry": len(completed),
        "resume_allowed": True,
    }


def _harness(tmp_path: Path) -> Harness:
    value = _contract_value(tmp_path)
    project = Path(value["fixed_paths"]["project_root"])
    (project / "artifacts").mkdir(parents=True)
    Path(value["fixed_paths"]["rust_binary_path"]).parent.mkdir()
    Path(value["fixed_paths"]["rust_binary_path"]).write_bytes(b"rust-binary\n")
    Path(value["fixed_paths"]["rust_binary_path"]).chmod(0o555)
    Path(value["runtime"]["seccomp_profile_path"]).write_bytes(b"{}\n")
    contract_path = tmp_path / "contract.json"
    contract_path.write_bytes(campaign.canonical_json_bytes_v1(value))
    contract = campaign.load_contract_v1(contract_path)
    source_rows = []
    for index, relative in enumerate(contract.value["controller_source_closure_relative_paths"]):
        payload = f"controller-{index}\n".encode("ascii")
        source_rows.append(
            {
                "relative_path": relative,
                "byte_count": len(payload),
                "raw_sha256": campaign.raw_sha256_v1(payload),
            }
        )
    source_identity = {
        "source_commit": "33" * 20,
        "source_tree": "44" * 20,
        "source_closure_root": campaign.source_closure_root_v1(source_rows),
    }
    binary = Path(contract.value["fixed_paths"]["rust_binary_path"]).read_bytes()
    inputs = {
        "schema_version": campaign.CAPSULE_INPUTS_SCHEMA_VERSION,
        "controller_source_identity": source_identity,
        "controller_source_closure_rows": source_rows,
        "rust_binary": {
            "path": contract.value["fixed_paths"]["rust_binary_path"],
            "byte_count": len(binary),
            "raw_sha256": campaign.raw_sha256_v1(binary),
        },
        "run_id": "55" * 16,
        "created_at_unix_seconds": 1_788_100_000,
    }
    inputs_path = tmp_path / "capsule-inputs.json"
    inputs_path.write_bytes(campaign.canonical_json_bytes_v1(inputs))
    capsule = campaign.build_capsule_v1(contract, inputs)
    blobs = {stage: f"{stage}-formal-source\n".encode("ascii") for stage in ("python", "rust", "host")}
    manifests = {
        stage: (
            {
                "relative_path": f"actors/{stage}.txt",
                "byte_count": len(blobs[stage]),
                "raw_sha256": campaign.raw_sha256_v1(blobs[stage]),
            },
        )
        for stage in ("python", "rust", "host")
    }
    by_relative = {
        manifests[stage][0]["relative_path"]: blobs[stage]
        for stage in manifests
    }
    predecessor = tool.PredecessorReplayV1(
        capsule={
            "mount_manifest_root": _sha(80),
            "formal_input_key_absence_root": _sha(81),
        },
        authorization={},
        consumption={},
        artifact={},
        actor_manifests=manifests,
    )
    source_state = tool.SourceStateV1(source_identity, tuple(source_rows))
    harness = Harness(
        contract_path,
        inputs_path,
        contract,
        capsule,
        source_state,
        predecessor,
        by_relative,
        {},
    )
    harness.supervisor_result = _success(capsule, contract)
    return harness


def _preflight_and_authorize(state: Harness) -> None:
    preflight = tool.preflight_v1(
        state.contract_path, state.inputs_path, dependencies=state.dependencies()
    )
    assert preflight["status"] == campaign.CAPSULE_READY
    authorization = tool.record_authorization_v1(
        state.contract_path,
        state.capsule["capsule_digest"],
        dependencies=state.dependencies(),
    )
    assert authorization["status"] == campaign.AUTHORIZATION_RECORDED


def _persist_consumption(state: Harness) -> dict[str, object]:
    authorization = campaign.build_authorization_v1(state.contract, state.capsule)
    consumption = campaign.build_consumption_v1(
        state.contract, state.capsule, authorization
    )
    Path(state.contract.value["fixed_paths"]["consumption_path"]).write_bytes(
        campaign.canonical_json_bytes_v1(consumption)
    )
    return consumption


def _persist_supervisor_success(state: Harness) -> tuple[bytes, dict[str, object]]:
    value = _persisted_success(state.capsule, state.contract)
    raw = campaign.canonical_json_bytes_v1(value)
    checkpoint = Path(state.contract.value["fixed_paths"]["checkpoint_root"])
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "supervisor-outcome.json").write_bytes(raw)
    return raw, value


def test_preflight_publishes_only_exact_capsule_and_replays(tmp_path: Path) -> None:
    state = _harness(tmp_path)
    first = tool.preflight_v1(
        state.contract_path, state.inputs_path, dependencies=state.dependencies()
    )
    second = tool.preflight_v1(
        state.contract_path, state.inputs_path, dependencies=state.dependencies()
    )
    fixed = state.contract.value["fixed_paths"]

    assert first["status"] == campaign.CAPSULE_READY
    assert first["read_only_replay"] is False
    assert second["read_only_replay"] is True
    assert campaign.validate_capsule_v1(
        campaign.strict_json_bytes_v1(Path(fixed["capsule_path"]).read_bytes(), "capsule"),
        state.contract,
    ) == state.capsule
    for name in ("authorization_path", "consumption_path", "terminal_path", "admission_path", "run_spec_path"):
        assert not Path(fixed[name]).exists()
    for stage in ("python", "rust", "host"):
        assert not Path(fixed[f"{stage}_source_root"]).exists()


@pytest.mark.parametrize(
    "root_name",
    [
        "artifact_path",
        "checkpoint_root",
        "python_output_path",
        "python_scratch_path",
        "rust_output_path",
        "rust_scratch_path",
        "host_output_path",
        "host_scratch_path",
    ],
)
def test_preflight_rejects_every_fresh_future_execution_root(
    tmp_path: Path, root_name: str
) -> None:
    state = _harness(tmp_path)
    fixed = state.contract.value["fixed_paths"]
    occupied = Path(fixed[root_name])
    if root_name == "artifact_path":
        occupied.write_bytes(b"stale\n")
    else:
        occupied.mkdir(parents=True)

    result = tool.preflight_v1(
        state.contract_path,
        state.inputs_path,
        dependencies=state.dependencies(),
    )

    assert result["status"] == campaign.PENDING_NOT_CONSUMED
    assert result["condition"] == "FUTURE_RUN_ROOTS_NOT_ABSENT"
    assert not Path(fixed["capsule_path"]).exists()


def test_record_authorization_materializes_and_replays_bound_inputs(tmp_path: Path) -> None:
    state = _harness(tmp_path)
    tool.preflight_v1(state.contract_path, state.inputs_path, dependencies=state.dependencies())
    first = tool.record_authorization_v1(
        state.contract_path, state.capsule["capsule_digest"], dependencies=state.dependencies()
    )
    second = tool.record_authorization_v1(
        state.contract_path, state.capsule["capsule_digest"], dependencies=state.dependencies()
    )
    fixed = state.contract.value["fixed_paths"]

    assert first == second == campaign.build_authorization_v1(state.contract, state.capsule)
    assert not Path(fixed["consumption_path"]).exists()
    assert state.supervisor_calls == 0
    assert Path(fixed["admission_path"]).stat().st_mode & 0o777 == 0o444
    assert Path(fixed["run_spec_path"]).stat().st_mode & 0o777 == 0o400
    assert Path(fixed["authorization_path"]).stat().st_mode & 0o777 == 0o400
    supervisor.load_run_spec_v1(Path(fixed["run_spec_path"]).read_bytes())
    for stage in ("python", "rust", "host"):
        row = state.predecessor.actor_manifests[stage][0]
        assert (Path(fixed[f"{stage}_source_root"]) / row["relative_path"]).read_bytes() == state.blobs[row["relative_path"]]


def test_record_replay_rejects_owner_only_python_admission(tmp_path: Path) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    fixed = state.contract.value["fixed_paths"]
    admission = Path(fixed["admission_path"])
    admission.chmod(0o400)

    result = tool.record_authorization_v1(
        state.contract_path,
        state.capsule["capsule_digest"],
        dependencies=state.dependencies(),
    )

    assert result["status"] == campaign.PENDING_NOT_CONSUMED
    assert result["condition"] == "PYTHON_ADMISSION_MODE_DIFFERS"
    assert not Path(fixed["consumption_path"]).exists()


def test_run_consumes_once_and_terminalizes_complete_receipt(tmp_path: Path) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    result = tool.run_v1(
        state.contract_path, state.capsule["capsule_digest"], dependencies=state.dependencies()
    )
    fixed = state.contract.value["fixed_paths"]

    assert result["status"] == campaign.COMPLETE
    assert state.supervisor_calls == 1
    consumption = campaign.validate_consumption_v1(
        campaign.strict_json_bytes_v1(Path(fixed["consumption_path"]).read_bytes(), "consumption"),
        state.contract,
        state.capsule,
        campaign.build_authorization_v1(state.contract, state.capsule),
    )
    assert consumption["authorization_consumption_count"] == 1
    receipt = wire.decode_completion_receipt_v1(Path(fixed["artifact_path"]).read_bytes())
    assert result["outcome"]["q1_output_slots"] == [
        "sha256:" + row[2].hex() for row in receipt.output_slots
    ]
    replay = tool.run_v1(
        state.contract_path, state.capsule["capsule_digest"], dependencies=state.dependencies()
    )
    assert replay == result
    assert state.supervisor_calls == 1


def test_partial_finalization_seals_existing_complete_outcome_without_supervisor(
    tmp_path: Path,
) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    _persist_consumption(state)
    supervisor_raw, _value = _persist_supervisor_success(state)
    artifact = _receipt()
    fixed = state.contract.value["fixed_paths"]
    Path(fixed["artifact_path"]).write_bytes(artifact)
    receipt = wire.decode_completion_receipt_v1(artifact)
    expected_outcome = tool._complete_outcome_v1(
        state.capsule, receipt, supervisor_raw
    )
    Path(fixed["outcome_path"]).write_bytes(
        campaign.canonical_json_bytes_v1(expected_outcome)
    )

    result = tool.run_v1(
        state.contract_path,
        state.capsule["capsule_digest"],
        dependencies=state.dependencies(),
    )
    replay = tool.run_v1(
        state.contract_path,
        state.capsule["capsule_digest"],
        dependencies=state.dependencies(),
    )

    assert result["status"] == campaign.COMPLETE
    assert replay == result
    assert state.supervisor_calls == 0
    assert Path(fixed["terminal_path"]).exists()


def test_partial_finalization_rebuilds_outcome_from_artifact_without_supervisor(
    tmp_path: Path,
) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    _persist_consumption(state)
    supervisor_raw, _value = _persist_supervisor_success(state)
    artifact = _receipt()
    fixed = state.contract.value["fixed_paths"]
    Path(fixed["artifact_path"]).write_bytes(artifact)

    result = tool.run_v1(
        state.contract_path,
        state.capsule["capsule_digest"],
        dependencies=state.dependencies(),
    )

    receipt = wire.decode_completion_receipt_v1(artifact)
    expected_outcome = tool._complete_outcome_v1(
        state.capsule, receipt, supervisor_raw
    )
    assert result["status"] == campaign.COMPLETE
    assert campaign.strict_json_bytes_v1(
        Path(fixed["outcome_path"]).read_bytes(), "recovered outcome"
    ) == expected_outcome
    assert state.supervisor_calls == 0
    assert Path(fixed["terminal_path"]).exists()


def test_partial_finalization_replays_capsule_bound_controller_source(
    tmp_path: Path,
) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    _persist_consumption(state)
    _persist_supervisor_success(state)
    fixed = state.contract.value["fixed_paths"]
    Path(fixed["artifact_path"]).write_bytes(_receipt())
    state.source_state = tool.SourceStateV1(
        {
            "source_commit": "ff" * 20,
            "source_tree": state.source_state.identity["source_tree"],
            "source_closure_root": state.source_state.identity["source_closure_root"],
        },
        state.source_state.rows,
    )

    result = tool.run_v1(
        state.contract_path,
        state.capsule["capsule_digest"],
        dependencies=state.dependencies(),
    )

    assert result["status"] == campaign.OUTCOME_UNKNOWN
    assert result["outcome"]["failure_code"] == "Q1_START_UNSEALED_ORPHAN_ARTIFACT"
    assert "CAPSULE_CONTROLLER_SOURCE_DIFFERS" in result["outcome"]["failure_detail"]
    assert result["outcome"]["q1_state"] == "NOT_RUN"
    assert state.supervisor_calls == 0


def test_artifact_link_tail_failure_recovers_without_second_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    _persist_supervisor_success(state)
    fixed = state.contract.value["fixed_paths"]
    artifact_path = Path(fixed["artifact_path"])
    original_publish = tool._publish_bytes
    injected = False

    def link_then_fail(
        path: Path,
        payload: bytes,
        digest: str,
        *,
        mode: int = 0o400,
    ) -> None:
        nonlocal injected
        if path == artifact_path and not injected:
            injected = True
            path.write_bytes(payload)
            raise tool.ControllerPending(
                "PATH_AND_FORMAT",
                "DURABLE_NO_REPLACE_PUBLICATION_FAILED",
                "synthetic post-link artifact tail failure",
            )
        original_publish(path, payload, digest, mode=mode)

    monkeypatch.setattr(tool, "_publish_bytes", link_then_fail)
    first = tool.run_v1(
        state.contract_path,
        state.capsule["capsule_digest"],
        dependencies=state.dependencies(),
    )
    second = tool.run_v1(
        state.contract_path,
        state.capsule["capsule_digest"],
        dependencies=state.dependencies(),
    )

    assert injected is True
    assert first["status"] == campaign.COMPLETE
    assert second == first
    assert state.supervisor_calls == 1
    assert Path(fixed["terminal_path"]).exists()


def test_unrecoverable_orphan_artifact_terminalizes_and_never_invokes(
    tmp_path: Path,
) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    _persist_consumption(state)
    fixed = state.contract.value["fixed_paths"]
    Path(fixed["artifact_path"]).write_bytes(b"not-a-completion-receipt\n")

    result = tool.run_v1(
        state.contract_path,
        state.capsule["capsule_digest"],
        dependencies=state.dependencies(),
    )
    replay = tool.run_v1(
        state.contract_path,
        state.capsule["capsule_digest"],
        dependencies=state.dependencies(),
    )

    assert result["status"] == campaign.OUTCOME_UNKNOWN
    assert result["outcome"]["failure_code"] == "Q1_START_UNSEALED_ORPHAN_ARTIFACT"
    assert replay == result
    assert state.supervisor_calls == 0
    assert Path(fixed["terminal_path"]).exists()


def test_non_orphan_failure_terminal_still_rejects_physical_artifact(
    tmp_path: Path,
) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    _persist_consumption(state)
    fixed = state.contract.value["fixed_paths"]
    authorization = campaign.build_authorization_v1(state.contract, state.capsule)
    outcome = {
        "status": campaign.OUTCOME_UNKNOWN,
        "capsule_digest": state.capsule["capsule_digest"],
        "actor_start_count_by_actor": {"host": 0, "python": 0, "rust": 0},
        "network_accessed": False,
        "target_truth_accessed": False,
        "split_accessed": False,
        "role_evaluation_performed": False,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 14,
        "q1_gate_mask": 16_383,
        "q1_output_slots": [None] * 8,
        "failure_code": "Q1_START_SYNTHETIC_FAILURE",
        "failure_detail": "synthetic",
    }
    terminal = campaign.build_terminal_record_v1(
        state.contract,
        state.capsule,
        authorization,
        outcome,
        artifact_raw_sha256=None,
    )
    Path(fixed["terminal_path"]).write_bytes(
        campaign.canonical_json_bytes_v1(terminal)
    )
    Path(fixed["artifact_path"]).write_bytes(b"orphan\n")

    with pytest.raises(tool.ControllerTerminal, match="NON_SUCCESS_ARTIFACT_PRESENT"):
        tool._read_terminal(state.contract, state.capsule, authorization)


def test_terminal_replay_rejects_valid_artifact_with_different_projection(tmp_path: Path) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    result = tool.run_v1(
        state.contract_path, state.capsule["capsule_digest"], dependencies=state.dependencies()
    )
    fixed = state.contract.value["fixed_paths"]
    authorization = campaign.build_authorization_v1(state.contract, state.capsule)
    changed_outcome = dict(result["outcome"])
    changed_outcome["q1_output_slots"] = list(changed_outcome["q1_output_slots"])
    changed_outcome["q1_output_slots"][0] = _sha(99)
    artifact = Path(fixed["artifact_path"]).read_bytes()
    changed_terminal = campaign.build_terminal_record_v1(
        state.contract,
        state.capsule,
        authorization,
        changed_outcome,
        artifact_raw_sha256=campaign.raw_sha256_v1(artifact),
    )
    terminal_path = Path(fixed["terminal_path"])
    terminal_path.chmod(0o600)
    terminal_path.write_bytes(campaign.canonical_json_bytes_v1(changed_terminal))

    with pytest.raises(
        tool.ControllerTerminal,
        match="TERMINAL_ARTIFACT_AUTHORITY_DIFFERS",
    ):
        tool._read_terminal(state.contract, state.capsule, authorization)


@pytest.mark.parametrize("completed", [[], ["python"], ["python", "rust"]])
def test_run_preserves_formal_clean_checkpoint_pending(
    tmp_path: Path, completed: list[str]
) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    state.supervisor_result = _pending(state.capsule, state.contract, completed)

    result = tool.run_v1(
        state.contract_path, state.capsule["capsule_digest"], dependencies=state.dependencies()
    )

    assert result["status"] == campaign.PENDING_CLEAN_CHECKPOINT
    assert result["completed_actors"] == completed
    assert result["authorization_consumption_count"] == 1
    assert not Path(state.contract.value["fixed_paths"]["terminal_path"]).exists()
    campaign.validate_pending_record_v1(result, state.contract, state.capsule)


def test_post_intent_unknown_terminalizes_and_never_invokes_again(tmp_path: Path) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    checkpoint = Path(state.contract.value["fixed_paths"]["checkpoint_root"])
    checkpoint.mkdir()
    (checkpoint / "01-python-intent.json").write_bytes(b"{}\n")
    predecessor = state.contract.value["predecessor"]
    state.supervisor_result = {
        "schema_version": supervisor.OUTCOME_SCHEMA,
        "status": supervisor.TERMINAL_STATUS,
        "capsule_digest": state.capsule["capsule_digest"],
        "source_commit": predecessor["formal_source_commit"],
        "run_genesis_root": predecessor["run_genesis_root"],
        "prestart_admission_receipt_root": predecessor["prestart_admission_receipt_root"],
        "active_stage": "python",
        "completed_stages": [],
        "error_code": "Q1_START_ACTOR_OUTCOME_UNKNOWN",
        "error_detail": "container result unavailable",
        "resume_allowed": False,
        "q1_completion_receipt_issued": False,
        "state_published": False,
    }

    result = tool.run_v1(
        state.contract_path, state.capsule["capsule_digest"], dependencies=state.dependencies()
    )
    replay = tool.run_v1(
        state.contract_path, state.capsule["capsule_digest"], dependencies=state.dependencies()
    )

    assert result["status"] == campaign.OUTCOME_UNKNOWN
    assert result["outcome"]["actor_start_count_by_actor"] == {"host": 0, "python": 1, "rust": 0}
    assert replay == result
    assert state.supervisor_calls == 1


def test_preconsumption_live_failure_is_pending_and_consumes_nothing(tmp_path: Path) -> None:
    state = _harness(tmp_path)
    tool.preflight_v1(state.contract_path, state.inputs_path, dependencies=state.dependencies())
    state.live_failure = tool.ControllerPending(
        "LOCAL_BINDINGS", "RUST_BINARY_BINDING_DIFFERS", "binary changed"
    )

    result = tool.record_authorization_v1(
        state.contract_path, state.capsule["capsule_digest"], dependencies=state.dependencies()
    )

    assert result["status"] == campaign.PENDING_NOT_CONSUMED
    assert result["condition"] == "RUST_BINARY_BINDING_DIFFERS"
    fixed = state.contract.value["fixed_paths"]
    assert not Path(fixed["authorization_path"]).exists()
    assert not Path(fixed["consumption_path"]).exists()
    assert state.supervisor_calls == 0


def test_consumption_prelink_failure_remains_not_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    consumption_path = Path(
        state.contract.value["fixed_paths"]["consumption_path"]
    )
    original_publish = tool._publish_bytes

    def fail_before_link(
        path: Path,
        payload: bytes,
        digest: str,
        *,
        mode: int = 0o400,
    ) -> None:
        if path == consumption_path:
            raise tool.ControllerPending(
                "PATH_AND_FORMAT",
                "DURABLE_NO_REPLACE_PUBLICATION_FAILED",
                "synthetic consumption pre-link failure",
            )
        original_publish(path, payload, digest, mode=mode)

    monkeypatch.setattr(tool, "_publish_bytes", fail_before_link)
    result = tool.run_v1(
        state.contract_path,
        state.capsule["capsule_digest"],
        dependencies=state.dependencies(),
    )

    assert result["status"] == campaign.PENDING_NOT_CONSUMED
    assert result["authorization_consumption_count"] == 0
    assert result["logical_campaign_count"] == 0
    assert not consumption_path.exists()
    assert state.supervisor_calls == 0


def test_consumption_postlink_tail_failure_replays_as_consumed_clean_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    consumption_path = Path(
        state.contract.value["fixed_paths"]["consumption_path"]
    )
    original_publish = tool._publish_bytes

    def link_then_fail(
        path: Path,
        payload: bytes,
        digest: str,
        *,
        mode: int = 0o400,
    ) -> None:
        if path == consumption_path:
            path.write_bytes(payload)
            raise tool.ControllerPending(
                "PATH_AND_FORMAT",
                "DURABLE_NO_REPLACE_PUBLICATION_FAILED",
                "synthetic consumption post-link tail failure",
            )
        original_publish(path, payload, digest, mode=mode)

    monkeypatch.setattr(tool, "_publish_bytes", link_then_fail)
    result = tool.run_v1(
        state.contract_path,
        state.capsule["capsule_digest"],
        dependencies=state.dependencies(),
    )

    assert result["status"] == campaign.PENDING_CLEAN_CHECKPOINT
    assert result["authorization_consumption_count"] == 1
    assert result["logical_campaign_count"] == 1
    assert result["completed_actors"] == []
    assert consumption_path.exists()
    assert state.supervisor_calls == 0


def test_present_invalid_consumption_is_terminal_not_unconsumed_pending(
    tmp_path: Path,
) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    consumption_path = Path(
        state.contract.value["fixed_paths"]["consumption_path"]
    )
    consumption_path.write_bytes(b'{"partial":true}\n')

    first = tool.run_v1(
        state.contract_path,
        state.capsule["capsule_digest"],
        dependencies=state.dependencies(),
    )
    second = tool.run_v1(
        state.contract_path,
        state.capsule["capsule_digest"],
        dependencies=state.dependencies(),
    )

    assert first["status"] == campaign.OUTCOME_UNKNOWN
    assert first["authorization_consumption_count"] == 1
    assert first["logical_campaign_count"] == 1
    assert first["outcome"]["failure_code"] == "Q1_START_CONSUMPTION_REPLAY_DIFFERS"
    assert second == first
    assert state.supervisor_calls == 0


def test_existing_consumption_live_failure_stays_same_campaign_pending(tmp_path: Path) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    authorization = campaign.build_authorization_v1(state.contract, state.capsule)
    consumption = campaign.build_consumption_v1(
        state.contract, state.capsule, authorization
    )
    consumption_path = Path(state.contract.value["fixed_paths"]["consumption_path"])
    consumption_path.write_bytes(campaign.canonical_json_bytes_v1(consumption))
    state.live_failure = tool.ControllerPending(
        "LOCAL_CAPACITY", "INSUFFICIENT_AVAILABLE_MEMORY", "capacity moved"
    )

    result = tool.run_v1(
        state.contract_path, state.capsule["capsule_digest"], dependencies=state.dependencies()
    )

    assert result["status"] == campaign.PENDING_CLEAN_CHECKPOINT
    assert result["completed_actors"] == []
    assert result["authorization_consumption_count"] == 1
    assert result["consumption_record_digest"] == consumption["record_digest"]
    assert state.supervisor_calls == 0
    campaign.validate_pending_record_v1(result, state.contract, state.capsule)


def test_actual_invoker_pre_popen_failure_is_clean_consumed_pending(tmp_path: Path) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)

    def invoke_without_executable(
        _python: Path, supervisor_path: Path, run_spec_path: Path
    ) -> Mapping[str, object]:
        return tool._actual_invoke_supervisor(
            tmp_path / "missing-python",
            supervisor_path,
            run_spec_path,
        )

    dependencies = replace(
        state.dependencies(), invoke_supervisor=invoke_without_executable
    )
    result = tool.run_v1(
        state.contract_path,
        state.capsule["capsule_digest"],
        dependencies=dependencies,
    )

    fixed = state.contract.value["fixed_paths"]
    assert result["status"] == campaign.PENDING_CLEAN_CHECKPOINT
    assert result["phase"] == "SUPERVISOR_START"
    assert result["condition"] == "SUPERVISOR_POPEN_UNAVAILABLE"
    assert result["completed_actors"] == []
    assert result["authorization_consumption_count"] == 1
    assert result["actor_start_count_by_actor"] == {
        "host": 0,
        "python": 0,
        "rust": 0,
    }
    assert not (Path(fixed["checkpoint_root"]) / "01-python-intent.json").exists()
    assert not Path(fixed["terminal_path"]).exists()
    campaign.validate_pending_record_v1(result, state.contract, state.capsule)


def test_existing_consumption_outer_pending_rejects_corrupt_checkpoint(tmp_path: Path) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    authorization = campaign.build_authorization_v1(state.contract, state.capsule)
    consumption = campaign.build_consumption_v1(
        state.contract, state.capsule, authorization
    )
    fixed = state.contract.value["fixed_paths"]
    Path(fixed["consumption_path"]).write_bytes(
        campaign.canonical_json_bytes_v1(consumption)
    )
    checkpoint = Path(fixed["checkpoint_root"])
    checkpoint.mkdir()
    (checkpoint / "01-python-intent.json").write_bytes(b"{}\n")
    (checkpoint / "01-python-complete.json").write_bytes(b"{}\n")
    state.live_failure = tool.ControllerPending(
        "LOCAL_CAPACITY", "INSUFFICIENT_AVAILABLE_MEMORY", "capacity moved"
    )

    result = tool.run_v1(
        state.contract_path, state.capsule["capsule_digest"], dependencies=state.dependencies()
    )

    assert result["status"] == campaign.OUTCOME_UNKNOWN
    assert result["outcome"]["failure_code"] == "Q1_START_DURABLE_CHECKPOINT_REPLAY_DIFFERS"
    assert result["outcome"]["actor_start_count_by_actor"] == {
        "host": 0,
        "python": 1,
        "rust": 0,
    }
    assert state.supervisor_calls == 0
    assert Path(fixed["terminal_path"]).exists()


def test_completion_pending_after_supervisor_success_is_terminal(tmp_path: Path) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    state.completion_failure = tool.ControllerPending(
        "COMPLETION_ADAPTER", "COMPLETION_OUTPUT_TEMPORARILY_UNAVAILABLE", "read failed"
    )

    result = tool.run_v1(
        state.contract_path, state.capsule["capsule_digest"], dependencies=state.dependencies()
    )

    assert result["status"] == campaign.OUTCOME_UNKNOWN
    assert result["outcome"]["actor_start_count_by_actor"] == {
        "host": 1,
        "python": 1,
        "rust": 1,
    }
    assert result["outcome"]["failure_code"] == "Q1_START_POST_SUPERVISOR_PENDING_NOT_RESUMABLE"
    assert Path(state.contract.value["fixed_paths"]["terminal_path"]).exists()
    replay = tool.run_v1(
        state.contract_path, state.capsule["capsule_digest"], dependencies=state.dependencies()
    )
    assert replay == result
    assert state.supervisor_calls == 1


def test_first_completion_publication_rejects_wrong_actor_implementation_root(
    tmp_path: Path,
) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    state.completion_payload = _receipt(host_implementation=36)

    result = tool.run_v1(
        state.contract_path, state.capsule["capsule_digest"], dependencies=state.dependencies()
    )

    fixed = state.contract.value["fixed_paths"]
    assert result["status"] == campaign.OUTCOME_UNKNOWN
    assert result["outcome"]["failure_code"] == "Q1_START_TERMINAL_ARTIFACT_AUTHORITY_DIFFERS"
    assert result["outcome"]["actor_start_count_by_actor"] == {
        "host": 1,
        "python": 1,
        "rust": 1,
    }
    assert not Path(fixed["artifact_path"]).exists()
    assert Path(fixed["terminal_path"]).exists()
    assert state.supervisor_calls == 1


def test_zero_checkpoint_pending_rejects_existing_exact_intent(tmp_path: Path) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    state.supervisor_result = _pending(state.capsule, state.contract, [])
    checkpoint = Path(state.contract.value["fixed_paths"]["checkpoint_root"])
    checkpoint.mkdir()
    (checkpoint / "01-python-intent.json").write_bytes(b"{}\n")

    result = tool.run_v1(
        state.contract_path, state.capsule["capsule_digest"], dependencies=state.dependencies()
    )

    assert result["status"] == campaign.OUTCOME_UNKNOWN
    assert result["outcome"]["failure_code"] == "Q1_START_ZERO_CHECKPOINT_INTENT_PRESENT"
    assert result["outcome"]["actor_start_count_by_actor"] == {
        "host": 0,
        "python": 1,
        "rust": 0,
    }


def test_mutated_snapshot_is_rejected_before_consumption(tmp_path: Path) -> None:
    state = _harness(tmp_path)
    _preflight_and_authorize(state)
    fixed = state.contract.value["fixed_paths"]
    source = Path(fixed["python_source_root"])
    source.chmod(0o755)
    target = source / state.predecessor.actor_manifests["python"][0]["relative_path"]
    target.chmod(0o644)
    target.write_bytes(b"mutated\n")

    result = tool.run_v1(
        state.contract_path, state.capsule["capsule_digest"], dependencies=state.dependencies()
    )

    assert result["status"] == campaign.PENDING_NOT_CONSUMED
    assert result["condition"] == "SOURCE_SNAPSHOT_BYTES_DIFFER"
    assert not Path(fixed["consumption_path"]).exists()
    assert state.supervisor_calls == 0


def test_source_identity_mutation_and_busy_lock_are_pending(tmp_path: Path) -> None:
    state = _harness(tmp_path)
    changed_identity = dict(state.source_state.identity)
    changed_identity["source_tree"] = "99" * 20
    state.source_state = tool.SourceStateV1(changed_identity, state.source_state.rows)
    source_result = tool.preflight_v1(
        state.contract_path, state.inputs_path, dependencies=state.dependencies()
    )
    assert source_result["condition"] == "CAPSULE_CONTROLLER_SOURCE_DIFFERS"

    state = _harness(tmp_path / "locked")
    lock_path = Path(state.contract.value["fixed_paths"]["campaign_lock_path"])
    descriptor = lock_path.open("w+")
    try:
        fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_result = tool.preflight_v1(
            state.contract_path, state.inputs_path, dependencies=state.dependencies()
        )
    finally:
        descriptor.close()
    assert lock_result["condition"] == "CAMPAIGN_LOCK_BUSY"


def test_noncanonical_capsule_inputs_do_not_publish(tmp_path: Path) -> None:
    state = _harness(tmp_path)
    state.inputs_path.write_text("{}", encoding="ascii")

    result = tool.preflight_v1(
        state.contract_path, state.inputs_path, dependencies=state.dependencies()
    )

    assert result["status"] == campaign.PENDING_NOT_CONSUMED
    assert result["capsule_digest"] is None
    assert not Path(state.contract.value["fixed_paths"]["capsule_path"]).exists()
