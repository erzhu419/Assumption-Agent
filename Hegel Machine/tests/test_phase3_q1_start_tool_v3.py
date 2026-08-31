from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Mapping

import pytest

from hegel_machine import phase3_q1_start_campaign_v3 as campaign
from hegel_machine import phase3_q1_start_completion_wire_v1 as wire


ROOT = Path(__file__).resolve().parents[1]
TOOL_SPEC = importlib.util.spec_from_file_location(
    "phase3_q1_start_tool_v3",
    ROOT / "tools" / "phase3_q1_start_v3.py",
)
assert TOOL_SPEC is not None and TOOL_SPEC.loader is not None
tool = importlib.util.module_from_spec(TOOL_SPEC)
sys.modules[TOOL_SPEC.name] = tool
TOOL_SPEC.loader.exec_module(tool)
supervisor = tool._load_supervisor_module()

FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "phase3_q1_start_campaign_v3_fixture",
    ROOT / "tests" / "test_phase3_q1_start_campaign_v3.py",
)
assert FIXTURE_SPEC is not None and FIXTURE_SPEC.loader is not None
campaign_fixture = importlib.util.module_from_spec(FIXTURE_SPEC)
sys.modules[FIXTURE_SPEC.name] = campaign_fixture
FIXTURE_SPEC.loader.exec_module(campaign_fixture)


def _root(value: int) -> bytes:
    return bytes([value]) * 32


def _tagged_bytes(value: object) -> bytes:
    assert isinstance(value, str) and value.startswith("sha256:")
    return bytes.fromhex(value.removeprefix("sha256:"))


def _receipt(bundle: Mapping[str, object]) -> bytes:
    genesis = _tagged_bytes(bundle["run_genesis_root"])
    prestart = _tagged_bytes(bundle["receipt_root"])
    implementations = bundle["actor_implementation_roots"]
    assert isinstance(implementations, dict)
    signatures = (
        wire.Q1SignatureCompletionRowV1(
            1, _root(1), _root(2), _root(3), wire.signature_guard_ceiling_rows_v1(1)
        ),
        wire.Q1SignatureCompletionRowV1(
            2, _root(4), _root(5), _root(6), wire.signature_guard_ceiling_rows_v1(2)
        ),
    )
    python = wire.Q1EndpointCompletionEvidenceV1(
        b"python", _tagged_bytes(implementations["python"]), genesis, prestart,
        _root(40), _root(22), signatures, 100, 10, 1000, 1000,
    )
    rust = wire.Q1EndpointCompletionEvidenceV1(
        b"rust", _tagged_bytes(implementations["rust"]), genesis, prestart,
        _root(41), _root(22), signatures, 100, 10, 1000, 1000,
    )
    agreement = wire.Q1DualReplayAgreementV1(
        genesis, prestart, python.evidence_root, rust.evidence_root,
        _root(42), signatures, _root(22), _tagged_bytes(implementations["host"]),
        _root(43), 100, 0, 1000, 1000,
    )
    actor_rows = tuple(
        wire.Q1IsolationEvidenceAdapterV1(
            actor_id=actor.encode("ascii"),
            actor_implementation_root=_tagged_bytes(implementations[actor]),
            source_manifest_rows=(
                wire.Q1SourceManifestEvidenceRowV1(
                    f"actor-{index}.py", 1, _root(50 + index)
                ),
            ),
            mount_manifest_rows=(
                wire.Q1MountManifestEvidenceRowV1(
                    "/output", f"/run/{actor}/output", True
                ),
                wire.Q1MountManifestEvidenceRowV1(
                    "/scratch", f"/run/{actor}/scratch", True
                ),
                wire.Q1MountManifestEvidenceRowV1(
                    "/source", f"/run/{actor}/source", False
                ),
            ),
            output_files=(wire.Q1FileEvidenceV1("result.json", b"{}\n"),),
            scratch_path=f"/run/{actor}/scratch",
        ).isolation_row()
        for index, actor in enumerate(("python", "rust", "host"))
    )
    ledger = wire.Q1TargetBlindAccessLedgerV1(
        genesis,
        prestart,
        agreement.agreement_root,
        wire.runtime_mount_registry_root_v1(actor_rows),
        wire.formal_input_key_registry_absence_root_v1(),
        actor_rows,
    )
    subjects = (
        (genesis, prestart, (python.canonical_object(), rust.canonical_object())),
        (
            python.evidence_root,
            rust.evidence_root,
            agreement.formal_file_closure_root,
            tuple(row.canonical_object() for row in signatures),
            _root(22),
            True,
            True,
        ),
        (agreement.canonical_object(), agreement.agreement_root),
        (ledger.canonical_object(), ledger.ledger_root),
    )
    core = wire.Q1CompletionCoreV1(
        genesis,
        prestart,
        subjects,
        wire.output_slots_1_through_7_v1(agreement, ledger),
    )
    return wire.Q1CompletionReceiptV1(core).canonical_bytes


@dataclass
class State:
    contract_path: Path
    inputs_path: Path
    contract: campaign.LoadedContractV3
    capsule: dict[str, object]
    bundle: dict[str, object]
    source_state: tool.SourceStateV3
    predecessor: tool.PredecessorReplayV3
    supervisor_calls: int = 0
    smoke_calls: int = 0
    events: list[str] | None = None
    smoke_error: Exception | None = None

    def supervisor_success(self) -> dict[str, object]:
        binding, authority = tool._expected_supervisor_bindings(
            self.contract, self.capsule
        )
        checkpoint = Path(self.contract.value["fixed_paths"]["checkpoint_root"])
        return {
            "schema_version": supervisor.OUTCOME_SCHEMA,
            "status": supervisor.SUCCESS_STATUS,
            **binding,
            **authority,
            "completed_stages": ["python", "rust", "host"],
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
            "q1_completion_receipt_issued": False,
            "state_published": False,
        }

    def dependencies(self) -> tool.ControllerDependenciesV3:
        if self.events is None:
            self.events = []

        def smoke(_contract: object, _spec: Mapping[str, object]) -> Mapping[str, object]:
            self.smoke_calls += 1
            assert self.events is not None
            self.events.append("smoke")
            fixed = self.contract.value["fixed_paths"]
            assert Path(fixed["run_spec_path"]).exists()
            assert not Path(fixed["consumption_path"]).exists()
            if self.smoke_error is not None:
                raise self.smoke_error
            mutable = [
                str(Path(_spec[name]))
                for name in (
                    "checkpoint_root", "python_output", "python_scratch",
                    "rust_output", "rust_scratch", "host_output", "host_scratch",
                )
            ]
            return {
                "schema_version": supervisor.AUTHORITY_SMOKE_SCHEMA,
                "status": supervisor.AUTHORITY_SMOKE_STATUS,
                "capsule_digest": _spec["start_capsule_digest"],
                "start_authorization_digest": _spec["start_authorization_digest"],
                "failed_start_v2_terminal_record_digest": _spec[
                    "failed_start_v2_terminal_record_digest"
                ],
                "successor_prestart_bundle_root": _spec[
                    "successor_prestart_bundle_root"
                ],
                "authority_replay": {},
                "stage_binding_replays": {stage: {} for stage in ("python", "rust", "host")},
                "mutable_roots_proven_absent": mutable,
                "actor_popen_count": 0,
                "checkpoint_or_mutable_roots_created": False,
                "subprocess_or_docker_invoked": False,
            }

        def invoke(_python: Path, _supervisor: Path, _spec: Path) -> Mapping[str, object]:
            self.supervisor_calls += 1
            assert self.events is not None
            self.events.append("supervisor")
            assert Path(self.contract.value["fixed_paths"]["consumption_path"]).exists()
            return self.supervisor_success()

        def live_gate(_contract: object, _capsule: Mapping[str, object]) -> None:
            assert self.events is not None
            self.events.append("gates")

        return tool.ControllerDependenciesV3(
            source_state=lambda _contract: self.source_state,
            replay_predecessor=lambda _contract: self.predecessor,
            formal_blob=lambda _project, _commit, relative: relative.encode("ascii"),
            live_gate=live_gate,
            startup_smoke=smoke,
            invoke_supervisor=invoke,
            build_completion_artifact=lambda *_args: _receipt(self.bundle),
        )


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> State:
    value = campaign_fixture._contract_value(tmp_path)
    fixed = value["fixed_paths"]
    assert isinstance(fixed, dict)
    Path(fixed["project_root"]).mkdir()
    run = Path(fixed["run_root"])
    run.mkdir(parents=True)
    rust_path = Path(fixed["rust_binary_path"])
    rust_payload = b"fresh q1 start v3 Rust materializer\n"
    rust_path.write_bytes(rust_payload)
    rust_path.chmod(0o555)
    contract_path = tmp_path / "contract.json"
    contract_path.write_bytes(campaign.canonical_json_bytes_v3(value))
    contract = campaign.load_contract_v3(contract_path)
    rows: list[dict[str, object]] = []
    for relative in contract.value["controller_source_closure_relative_paths"]:
        payload = f"source:{relative}\n".encode("ascii")
        rows.append(
            {
                "relative_path": relative,
                "byte_count": len(payload),
                "raw_sha256": campaign.raw_sha256_v3(payload),
            }
        )
    source_identity = {
        "source_commit": "99" * 20,
        "source_tree": "aa" * 20,
        "source_closure_root": campaign.source_closure_root_v3(rows),
    }
    bundle, manifests = campaign_fixture._bundle(monkeypatch, source_identity)
    inputs = {
        "schema_version": campaign.CAPSULE_INPUTS_SCHEMA_VERSION,
        "controller_source_identity": source_identity,
        "controller_source_closure_rows": rows,
        "rust_binary": {
            "path": str(rust_path),
            "byte_count": len(rust_payload),
            "raw_sha256": campaign.raw_sha256_v3(rust_payload),
        },
        "successor_prestart_bundle": bundle,
        "actor_source_manifests": manifests,
        "run_id": "bb" * 16,
        "created_at_unix_seconds": 1_788_100_200,
    }
    inputs_path = tmp_path / "capsule-inputs.json"
    inputs_path.write_bytes(campaign.canonical_json_bytes_v3(inputs))
    capsule = campaign.build_capsule_v3(contract, inputs)
    predecessor = tool.PredecessorReplayV3(
        {}, {}, {}, {}, {}, {}, {}, {}, bundle
    )
    return State(
        contract_path,
        inputs_path,
        contract,
        capsule,
        bundle,
        tool.SourceStateV3(source_identity, tuple(rows)),
        predecessor,
    )


def _preflight_and_authorize(
    state: State,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tool, "_materialize_snapshot", lambda *_args: None)
    preflight = tool.preflight_v3(
        state.contract_path, state.inputs_path, dependencies=state.dependencies()
    )
    assert preflight["status"] == campaign.CAPSULE_READY, (
        preflight.get("condition"), preflight.get("detail")
    )
    authorization = tool.record_authorization_v3(
        state.contract_path,
        str(state.capsule["capsule_digest"]),
        dependencies=state.dependencies(),
    )
    assert authorization == campaign.build_authorization_v3(
        state.contract, state.capsule
    )


def test_record_authorization_materializes_exact_v3_run_spec(
    state: State,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _preflight_and_authorize(state, monkeypatch)
    fixed = state.contract.value["fixed_paths"]
    raw = Path(fixed["run_spec_path"]).read_bytes()
    value = campaign.strict_json_bytes_v3(raw, "v3 run spec")
    spec = supervisor.load_run_spec_v3(raw)

    assert set(value) == supervisor._RUN_SPEC_KEYS
    assert spec.start_capsule.parent == spec.run_root
    assert spec.start_authorization_raw_sha256 == campaign.raw_sha256_v3(
        Path(fixed["authorization_path"]).read_bytes()
    )
    assert spec.actor_source_manifest_roots == state.bundle[
        "actor_source_manifest_roots"
    ]
    assert state.supervisor_calls == 0


def test_prepare_capsule_inputs_rebuilds_current_actor_manifests(
    state: State,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = Path(state.contract.value["fixed_paths"]["project_root"])
    rust_path = Path(state.contract.value["fixed_paths"]["rust_binary_path"])
    base_path = Path(
        state.contract.value["base_authoritative_predecessor"][
            "prestart_artifact_path"
        ]
    )
    failed_path = Path(
        state.contract.value["failed_start_v2_predecessor"]["terminal_path"]
    )

    def payload_for(path: Path) -> bytes:
        if path == rust_path:
            return rust_path.read_bytes()
        if path == base_path:
            return b"synthetic admitted base artifact"
        if path == failed_path:
            return b"synthetic admitted failed-v2 terminal"
        return path.relative_to(project).as_posix().encode("ascii")

    monkeypatch.setattr(tool, "_actual_source_state", lambda _contract: state.source_state)
    monkeypatch.setattr(tool, "_actual_replay_predecessor", lambda _contract: state.predecessor)
    monkeypatch.setattr(
        tool,
        "_actual_formal_blob",
        lambda _project, _commit, relative: relative.encode("ascii"),
    )
    monkeypatch.setattr(
        tool,
        "_regular_file",
        lambda path, _label, executable=False: payload_for(path),
    )

    inputs = tool.prepare_capsule_inputs_v3(
        state.contract,
        "cc" * 16,
        1_788_100_300,
        rust_path,
    )
    capsule = campaign.build_capsule_v3(state.contract, inputs)

    for actor in ("python", "rust", "host"):
        rows = inputs["actor_source_manifests"][actor]["rows"]
        assert [row["relative_path"] for row in rows] == list(
            tool._ACTOR_SOURCE_RELATIVE_PATHS[actor]
        )
        assert inputs["successor_prestart_bundle"]["actor_source_manifests"][
            actor
        ]["rows"] == rows
    assert capsule["status"] == campaign.CAPSULE_READY


def test_run_consumes_once_and_completes_with_successor_authority(
    state: State,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _preflight_and_authorize(state, monkeypatch)
    state.events = []
    result = tool.run_v3(
        state.contract_path,
        str(state.capsule["capsule_digest"]),
        dependencies=state.dependencies(),
    )
    replay = tool.run_v3(
        state.contract_path,
        str(state.capsule["capsule_digest"]),
        dependencies=state.dependencies(),
    )

    assert result["status"] == campaign.COMPLETE
    assert replay == result
    assert state.supervisor_calls == 1
    assert state.smoke_calls == 1
    assert state.events == ["gates", "smoke", "supervisor"]
    assert result["outcome"]["q1_gate_count"] == 20


def test_startup_smoke_failure_is_exact_preconsumption_pending(
    state: State,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _preflight_and_authorize(state, monkeypatch)
    state.events = []
    state.smoke_error = RuntimeError("synthetic startup import failure\nlast-line")

    result = tool.run_v3(
        state.contract_path,
        str(state.capsule["capsule_digest"]),
        dependencies=state.dependencies(),
    )

    fixed = state.contract.value["fixed_paths"]
    assert result["status"] == campaign.PENDING_NOT_CONSUMED
    assert result["phase"] == "SUPERVISOR_STARTUP_SMOKE"
    assert result["condition"] == "AUTHORITY_STARTUP_SMOKE_FAILED"
    assert "last-line" in result["detail"]
    assert not Path(fixed["consumption_path"]).exists()
    assert state.smoke_calls == 1
    assert state.supervisor_calls == 0
    assert state.events == ["gates", "smoke"]


def test_startup_smoke_cannot_create_checkpoint_or_mutable_roots(
    state: State,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _preflight_and_authorize(state, monkeypatch)
    state.events = []
    base = state.dependencies()

    def mutating_smoke(
        contract: object,
        spec: Mapping[str, object],
    ) -> Mapping[str, object]:
        receipt = base.startup_smoke(contract, spec)
        Path(spec["checkpoint_root"]).mkdir()
        return receipt

    dependencies = tool.ControllerDependenciesV3(
        source_state=base.source_state,
        replay_predecessor=base.replay_predecessor,
        formal_blob=base.formal_blob,
        live_gate=base.live_gate,
        startup_smoke=mutating_smoke,
        invoke_supervisor=base.invoke_supervisor,
        build_completion_artifact=base.build_completion_artifact,
    )
    result = tool.run_v3(
        state.contract_path,
        str(state.capsule["capsule_digest"]),
        dependencies=dependencies,
    )

    fixed = state.contract.value["fixed_paths"]
    assert result["status"] == campaign.PENDING_NOT_CONSUMED
    assert result["condition"] == "SMOKE_OR_FUTURE_EXECUTION_ROOTS_NOT_ABSENT"
    assert not Path(fixed["consumption_path"]).exists()
    assert state.supervisor_calls == 0


def test_successful_startup_smoke_itself_leaves_all_execution_roots_absent(
    state: State,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _preflight_and_authorize(state, monkeypatch)
    fixed = state.contract.value["fixed_paths"]
    run_spec = campaign.strict_json_bytes_v3(
        Path(fixed["run_spec_path"]).read_bytes(), "v3 run spec"
    )
    receipt = state.dependencies().startup_smoke(state.contract, run_spec)

    assert receipt["checkpoint_or_mutable_roots_created"] is False
    for name in (
        "consumption_path", "checkpoint_root", "python_output_path",
        "python_scratch_path", "rust_output_path", "rust_scratch_path",
        "host_output_path", "host_scratch_path", "terminal_path", "artifact_path",
    ):
        assert not Path(fixed[name]).exists()


def test_bounded_process_diagnostic_preserves_head_and_tail() -> None:
    payload = b"HEAD" + b"x" * tool._DIAGNOSTIC_BOUND + b"TAIL"
    observed = tool._bounded_diagnostic(payload)

    assert observed.startswith("HEAD")
    assert observed.endswith("TAIL")
    assert "diagnostic bytes omitted" in observed


def test_partial_finalization_uses_v3_run_spec_without_second_supervisor(
    state: State,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _preflight_and_authorize(state, monkeypatch)
    fixed = state.contract.value["fixed_paths"]
    authorization = campaign.build_authorization_v3(state.contract, state.capsule)
    consumption = campaign.build_consumption_v3(
        state.contract, state.capsule, authorization
    )
    Path(fixed["consumption_path"]).write_bytes(
        campaign.canonical_json_bytes_v3(consumption)
    )
    checkpoint = Path(fixed["checkpoint_root"])
    checkpoint.mkdir()
    supervisor_raw = campaign.canonical_json_bytes_v3(state.supervisor_success())
    (checkpoint / "supervisor-outcome.json").write_bytes(supervisor_raw)
    artifact = _receipt(state.bundle)
    Path(fixed["artifact_path"]).write_bytes(artifact)

    result = tool.run_v3(
        state.contract_path,
        str(state.capsule["capsule_digest"]),
        dependencies=state.dependencies(),
    )

    assert result["status"] == campaign.COMPLETE
    assert state.supervisor_calls == 0
    assert Path(fixed["terminal_path"]).exists()


def test_actual_completion_adapter_validates_v3_roots_then_builds_v1_receipt(
    state: State,
    tmp_path: Path,
) -> None:
    closure_bundle_root = "sha256:" + "16" * 32
    signature_preimages = [
        {
            "input_signature_id": 1,
            "fixed_point_record_root": "sha256:" + "01" * 32,
            "signature_manifest_root": "sha256:" + "02" * 32,
            "saturation_state_root": "sha256:" + "03" * 32,
        },
        {
            "input_signature_id": 2,
            "fixed_point_record_root": "sha256:" + "04" * 32,
            "signature_manifest_root": "sha256:" + "05" * 32,
            "saturation_state_root": "sha256:" + "06" * 32,
        },
    ]
    run_spec: dict[str, object] = {
        "start_capsule_digest": state.capsule["capsule_digest"],
        "source_commit": state.bundle["source_identity"]["source_commit"],
        "run_genesis_root": state.bundle["run_genesis_root"],
        "prestart_admission_receipt_root": state.bundle["receipt_root"],
        "actor_implementation_roots": state.bundle["actor_implementation_roots"],
        "actor_source_manifests": {},
    }
    stage_rows: list[dict[str, object]] = []
    for index, stage in enumerate(("python", "rust", "host"), 1):
        output = tmp_path / f"{stage}-adapter-output"
        output.mkdir()
        scratch = tmp_path / f"{stage}-adapter-scratch"
        source = tmp_path / f"{stage}-adapter-source"
        source.mkdir()
        if stage == "host":
            host_signatures = [
                {**row, "universe_root": "sha256:" + f"{90 + ordinal:02x}" * 32}
                for ordinal, row in enumerate(signature_preimages, 1)
            ]
            host_result = {
                "schema_version": "hegel-q1-formal-host-replay-result/1",
                "actor_id": "host",
                "implementation_id": "hegel-host-q1-formal-archive-replay-v1",
                "action_id": "replay-full-node6-formal-archive-v1",
                "status": "DUAL_FORMAL_ARCHIVE_EXACT_AGREEMENT",
                "capsule_digest": run_spec["start_capsule_digest"],
                "source_commit": run_spec["source_commit"],
                "run_genesis_root": run_spec["run_genesis_root"],
                "prestart_admission_receipt_root": run_spec[
                    "prestart_admission_receipt_root"
                ],
                "q1_state": "NOT_RUN",
                "q1_gate_count": 14,
                "q1_gate_total": 20,
                "q1_gate_mask": 16_383,
                "q1_output_slots": [None] * 8,
                "formal_capacity_limits_by_signature": wire._formal_capacity_limit_rows_v1(),
                "formal_output_bytes_limit": wire._ACTOR_LIMITS[b"python"][9],
                "formal_scratch_bytes_limit": wire.FORMAL_ENDPOINT_SCRATCH_BYTES_LIMIT,
                "semantic_binding_root": "sha256:" + "17" * 32,
                "projection_profile_root": "sha256:" + "18" * 32,
                "closure_bundle_root": closure_bundle_root,
                "archive_payload_bytes_per_endpoint": 20,
                "formal_file_count_per_endpoint": 1,
                "signatures": host_signatures,
                "python_rust_formal_bytes_exact": True,
                "python_rust_replayed_roots_exact": True,
                "target_truth_accessed": False,
                "split_accessed": False,
                "role_evaluation_performed": False,
                "target_split_role_absence_attested": True,
                "q1_completion_receipt_issued": False,
                "state_published": False,
            }
            (output / "host-result.json").write_bytes(
                campaign.canonical_json_bytes_v3(host_result)
            )
        else:
            endpoint_result = {
                "schema_version": "hegel-phase3a-q1-full-node6-formal-archive-materializer/1",
                "actor_id": stage,
                "implementation_id": f"hegel-{stage}-q1-formal-archive-materializer-v1",
                "action_id": "materialize-full-node6-formal-archive-v1",
                "status": "FULL_NODE6_FORMAL_ARCHIVE_MATERIALIZED_Q1_NOT_RUN",
                "capsule_digest": run_spec["start_capsule_digest"],
                "source_commit": run_spec["source_commit"],
                "run_genesis_root": run_spec["run_genesis_root"],
                "prestart_admission_receipt_root": run_spec[
                    "prestart_admission_receipt_root"
                ],
                "closure_bundle_root": closure_bundle_root,
                "closure_semantics_version": "v1",
                "dsl_version": "v1",
                "diagnostic_only": False,
                "formal_archive_materialized": True,
                "full_v16_structural_limits_applied": True,
                "maximum_ast_depth": 3,
                "maximum_ast_node_count": 6,
                "formal_archive_payload_bytes": 20,
                "formal_capacity_limits_by_signature": wire._formal_capacity_limit_rows_v1(),
                "formal_output_bytes_limit": wire._ACTOR_LIMITS[stage.encode("ascii")][9],
                "formal_scratch_bytes_limit": wire.FORMAL_ENDPOINT_SCRATCH_BYTES_LIMIT,
                "actual_formal_peak_scratch_bytes": 1,
                "actual_formal_terminal_live_scratch_bytes": 0,
                "partitions": signature_preimages,
                "q1_state": "NOT_RUN",
                "q1_gate_count": 14,
                "q1_gate_total": 20,
                "q1_gate_mask": 16_383,
                "q1_output_slots": [None] * 8,
                "target_truth_accessed": False,
                "split_accessed": False,
                "role_evaluation_performed": False,
            }
            (output / "endpoint-result.json").write_bytes(
                campaign.canonical_json_bytes_v3(endpoint_result)
            )
            (output / "formal.bin").write_bytes(b"same formal archive\n")
        manifest = [
            {
                "relative_path": f"actor-{stage}.py",
                "byte_count": index,
                "raw_sha256": "sha256:" + f"{70 + index:02x}" * 32,
            }
        ]
        mounts = sorted(
            [
                {"destination": "/output", "read_write": True, "source": str(output)},
                {"destination": "/scratch", "read_write": True, "source": str(scratch)},
                {"destination": "/source", "read_write": False, "source": str(source)},
            ],
            key=lambda row: row["destination"],
        )
        runtime = {
            "mounts": mounts,
            "measurement_interval_start_monotonic_ns": 1,
            "actor_return_monotonic_ns": 2,
            "container_exit_monotonic_ns": 3,
            "measurement_wall_time_ns": 2,
            "memory_peak_bytes": 4,
            "actor_terminal_scratch_payload_bytes": 0,
            "supervisor_control_peak_payload_bytes": 0,
            "command": [],
        }
        run_spec[f"{stage}_output"] = str(output)
        run_spec[f"{stage}_scratch"] = str(scratch)
        run_spec["actor_source_manifests"][stage] = manifest
        stage_rows.append(
            {
                "stage": stage,
                "source_manifest_replay": {
                    "source_closure_root": supervisor._object_root(
                        supervisor._SOURCE_CLOSURE_DOMAIN, manifest
                    )
                },
                "mount_registry_root": supervisor._object_root(
                    supervisor._MOUNT_REGISTRY_DOMAIN, mounts
                ),
                "output_tree_closure_root": supervisor._output_tree_closure_root(output),
                "scratch_absence_root": supervisor._object_root(
                    supervisor._SCRATCH_ABSENCE_DOMAIN,
                    {"path": str(scratch), "absent": True},
                ),
                "runtime_evidence": runtime,
            }
        )
    formal_paths = ["formal.bin"]
    outcome = {
        "host_result": {"closure_bundle_root": closure_bundle_root},
        "gate16_candidate": {
            "signature_state_preimages": signature_preimages,
            "formal_relative_paths": formal_paths,
            "formal_file_closure_root": supervisor._file_closure_root(
                Path(run_spec["python_output"]),
                formal_paths,
                supervisor._FORMAL_FILE_CLOSURE_DOMAIN,
            ),
        },
        "gate18_runtime_receipt_inputs": {"stage_rows": stage_rows},
    }

    payload = tool._actual_build_completion_artifact(
        outcome, state.contract, state.capsule, run_spec
    )
    receipt = wire.decode_completion_receipt_v1(payload)
    assert receipt.core.run_genesis_root == _tagged_bytes(
        state.bundle["run_genesis_root"]
    )
