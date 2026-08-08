from __future__ import annotations

import json
from pathlib import Path
import tempfile
from threading import Barrier, Lock
from types import MappingProxyType, SimpleNamespace

import pytest

import hegel_machine.phase3_m3_dual_enumeration_supervisor_v1 as supervisor
from hegel_machine.phase3_m25_wire_v1 import (
    build_formal_object,
    candidate_content_root,
    git_sha1_commit_id,
)
from hegel_machine.phase3_m25_formal_container_executor_v1 import (
    load_gate_evidence_inputs_v1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _committed_gate_inputs():
    payload = json.loads(
        (
            PROJECT_ROOT
            / "artifacts/phase3_m25_external/formal_genesis_v2/"
            "phase3_m25_formal_gate_evidence_v1.json"
        ).read_text(encoding="utf-8")
    )
    return load_gate_evidence_inputs_v1(payload)


GATE_INPUTS = _committed_gate_inputs()
RUN_ID = GATE_INPUTS.execution_candidate_fields["run_id"]
MANIFEST_ROOT = candidate_content_root(
    "M3ExecutionManifestV2", GATE_INPUTS.execution_manifest_fields
)


@pytest.fixture
def linux_tmp_path() -> Path:
    with tempfile.TemporaryDirectory(
        prefix="hegel-m3-supervisor-test-", dir="/tmp"
    ) as raw:
        path = Path(raw)
        path.chmod(0o700)
        yield path


def _golden() -> dict[str, object]:
    return json.loads(
        (
            PROJECT_ROOT
            / "golden_vectors/phase3_m3_bounded_dual_agreement_v1.json"
        ).read_text(encoding="utf-8")
    )


def _formal_roots() -> MappingProxyType[str, bytes]:
    candidate = GATE_INPUTS.execution_candidate_fields
    return MappingProxyType(
        {
            "child_dsl_spec_root": candidate["child_dsl_spec_root"],
            "operator_semantics_root": candidate["operator_semantics_root"],
            "identifier_registry_root": candidate["identifier_registry_root"],
            "canonical_ast_schema_root": candidate["canonical_ast_schema_root"],
            "canonical_cbor_profile_root": candidate["canonical_cbor_profile_root"],
            "m3_execution_candidate_root": candidate_content_root(
                "M3ExecutionCandidateV1", candidate
            ),
            "m3_execution_manifest_root": MANIFEST_ROOT,
            "m3_run_genesis_root": candidate_content_root(
                "M3RunGenesisV1", _run_genesis()
            ),
        }
    )


def _run_genesis() -> dict[str, object]:
    fields: dict[str, object] = {
        "run_id": RUN_ID,
        "execution_manifest_root": MANIFEST_ROOT,
        "initial_state_id": 0,
        "canonical_program_archive_root_or_null": None,
        "program_chunk_manifest_root_or_null": None,
        "bucket_accounting_root_or_null": None,
        "outside_program_output_archive_root_or_null": None,
        "outside_output_chunk_manifest_root_or_null": None,
        "outside_match_set_root_or_null": None,
        "outside_role_evaluation_receipt_root_or_null": None,
        "null_program_output_archive_root_or_null": None,
        "null_output_chunk_manifest_root_or_null": None,
        "null_match_set_root_or_null": None,
        "null_role_evaluation_receipt_root_or_null": None,
        "python_enumeration_receipt_root_or_null": None,
        "rust_enumeration_receipt_root_or_null": None,
        "dual_replay_agreement_root_or_null": None,
        "final_state_record_root_or_null": None,
        "created_at_unix_seconds": 90,
        "repository_commit_id": git_sha1_commit_id(bytes.fromhex(supervisor.COMMIT_A)),
    }
    build_formal_object("M3RunGenesisV1", fields)
    return fields


def _start_record() -> tuple[dict[str, object], bytes]:
    fields: dict[str, object] = {
        "run_id": RUN_ID,
        "transition_index": 0,
        "previous_state_record_root_or_null": None,
        "from_state_id": 0,
        "from_phase_id": 0,
        "to_state_id": 1,
        "to_phase_id": 1,
        "transition_reason_id": 1,
        "execution_manifest_root": MANIFEST_ROOT,
        "triggering_receipt_root_or_null": None,
        "recorded_at_unix_seconds": 100,
    }
    return fields, candidate_content_root("M3RunStateRecordV1", fields)


def _qualification_receipt() -> dict[str, object]:
    value: dict[str, object] = {
        "basis_commit": supervisor.COMMIT_A,
        "pull_policy_never": True,
        "network_mode_none": True,
        "m3_state": "NOT_RUN",
    }
    for name, binding in supervisor.FROZEN_IMPLEMENTATIONS.items():
        value[name] = {
            "implementation_id": binding.implementation_id,
            "source_root": binding.source_root.hex(),
            "binary_digest": binding.binary_digest.hex(),
            "image_ref": binding.image_ref,
            "execution_environment_spec_root": (
                binding.execution_environment_spec_root.hex()
            ),
            "implementation_binding_root": binding.implementation_binding_root.hex(),
            "bound_executable_locator": binding.bound_executable_locator,
        }
    return value


def _report(implementation: str, *, program_root: str = "44" * 32) -> dict[str, object]:
    golden = _golden()
    expected = dict(golden["expected"])  # type: ignore[arg-type]
    identity = {
        "python": (
            "hegel-m3-python-closure-enumerator-report/1",
            1,
            "hegel-python-m3-bounded-closure-enumerator-v1",
        ),
        "rust": (
            "hegel-m3-rust-closure-enumerator-report/1",
            2,
            "hegel-rust-m3-bounded-closure-enumerator-v1",
        ),
    }[implementation]
    roots = _formal_roots()
    report: dict[str, object] = {
        "schema_version": identity[0],
        "claim_level": "FORMAL_PROFILE_CANDIDATE_NOT_AUTHORITY",
        "authoritative_claim_allowed": False,
        "implementation": implementation,
        "implementation_id": identity[1],
        "implementation_machine_id": identity[2],
        "raw_expansion_limit_hit": False,
        "wall_clock_abort_hit": False,
        "target_roles_evaluated": False,
        "split_material_accessed": False,
        "secrets_accessed": False,
        "child_dsl_spec_root": roots["child_dsl_spec_root"].hex(),
        "operator_semantics_root": roots["operator_semantics_root"].hex(),
        "identifier_registry_root": roots["identifier_registry_root"].hex(),
    }
    renamed = {
        "bucket_accounting_root": "bucket_accounting_root_or_null",
        "canonical_program_archive_root": "canonical_program_archive_root_or_null",
        "first_out_of_budget_program_cbor_hex": "first_out_of_budget_program_cbor_hex_or_null",
        "first_out_of_budget_program_hash": "first_out_of_budget_program_hash_or_null",
        "program_chunk_manifest_root": "program_chunk_manifest_root_or_null",
    }
    for key, value in expected.items():
        report[renamed.get(key, key)] = value
    report["canonical_program_archive_root_or_null"] = program_root
    report["program_chunk_manifest_root_or_null"] = "55" * 32
    return report


def _budget_report(implementation: str) -> dict[str, object]:
    report = _report(implementation)
    report.update(
        {
            "closure_status": "INCONCLUSIVE_BUDGET",
            "closure_status_id": 3,
            "raw_operator_application_count": 5_000_000,
            "canonical_program_count": 0,
            "closure_cardinality_or_null": None,
            "frontier_exhausted": False,
            "all_type_buckets_closed": False,
            "raw_expansion_limit_hit": True,
            "wall_clock_abort_hit": False,
            "canonical_program_archive_root_or_null": None,
            "program_chunk_manifest_root_or_null": None,
            "bucket_accounting_root_or_null": None,
            "first_out_of_budget_program_hash_or_null": None,
            "first_out_of_budget_program_cbor_hex_or_null": None,
            "program_record_count": 0,
            "chunk_manifest_count": 0,
            "bucket_record_count": 0,
            "traversal_prefix_complete": False,
        }
    )
    return report


def _write_budget_archive(
    output_parent: Path, report: dict[str, object], *, nonempty_stream: str | None = None
) -> None:
    archive = output_parent / "archive"
    archive.mkdir(mode=0o700)
    (archive / "report.json").write_text(
        json.dumps(report, sort_keys=True), encoding="utf-8"
    )
    for stream_name in supervisor._ARCHIVE_STREAM_NAMES:
        (archive / f"{stream_name}.cborframed").write_bytes(
            b"not-empty" if stream_name == nonempty_stream else b""
        )


def _fake_replayed_archive(payload: bytes = b"same") -> dict[str, object]:
    return {
        "streams": {
            "canonical_program_records": payload,
            "program_chunk_manifests": payload,
            "bucket_accounting_records": payload,
        },
        "witness_adjacency_verified": True,
    }


def _install_public_validation_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        supervisor,
        "promote_gate_evidence_v1",
        lambda _evidence: {
            "basis_commit": supervisor.COMMIT_A,
            "m3_entry_qualified": True,
            "child_state": "NOT_RUN",
            "m3_run_started": False,
        },
    )
    monkeypatch.setattr(
        supervisor._qualification,
        "validate_qualification_receipt_v1",
        lambda *_args, **_kwargs: b"q" * 32,
    )
    monkeypatch.setattr(
        supervisor._qualification,
        "_host_validate_enumerator_archive_v1",
        lambda *_args, **_kwargs: _fake_replayed_archive(),
    )


def test_parallel_exact_dual_dsl_too_large_builds_receipts_agreement_and_terminal(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_public_validation_fakes(monkeypatch)
    gate = SimpleNamespace(formal_roots=_formal_roots())
    start, start_root = _start_record()
    barrier = Barrier(2)
    lock = Lock()
    active = 0
    peak = 0

    def runner(call: supervisor.EnumerationInvocationV1):
        nonlocal active, peak
        assert call.pull_policy == "never"
        assert call.network_mode == "none"
        assert call.output_parent.name == call.implementation
        call.output_parent.mkdir(mode=0o700, exist_ok=False)
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=2)
        with lock:
            active -= 1
        offset = 0 if call.implementation == "python" else 1
        return supervisor.EnumerationRunResultV1(
            invocation=call,
            report=_report(call.implementation),
            started_at_unix_seconds=110 + offset,
            finished_at_unix_seconds=120 + offset,
            process_exit_code=0,
        )

    outcome = supervisor._run_m3_dual_enumeration_core_v1(
        qualified_gate_evidence=gate,  # type: ignore[arg-type]
        execution_candidate_fields=GATE_INPUTS.execution_candidate_fields,
        run_genesis_fields=_run_genesis(),
        start_record_fields=start,
        start_record_root=start_root,
        implementation_qualification_receipt=_qualification_receipt(),
        committed_golden=_golden(),
        output_root=linux_tmp_path / "formal-enumeration",
        runner=runner,
        clock=iter((130, 131)).__next__,
    )

    assert peak == 2
    assert outcome.python_receipt_fields["closure_status_id"] == 2
    assert outcome.rust_receipt_fields["canonical_program_count"] == 50_000
    assert outcome.agreement_fields["enumeration_agreement"] is True
    assert outcome.agreement_fields["role_agreement_entries"] == ()
    assert outcome.agreement_fields["role_agreement_status_id"] == 0
    assert outcome.terminal_state_fields == {
        "run_id": RUN_ID,
        "transition_index": 1,
        "previous_state_record_root_or_null": start_root,
        "from_state_id": 1,
        "from_phase_id": 1,
        "to_state_id": 3,
        "to_phase_id": 0,
        "transition_reason_id": 3,
        "execution_manifest_root": MANIFEST_ROOT,
        "triggering_receipt_root_or_null": outcome.agreement_root,
        "recorded_at_unix_seconds": 131,
    }


def test_dual_mismatch_fails_before_any_terminal_is_returned(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_public_validation_fakes(monkeypatch)
    gate = SimpleNamespace(formal_roots=_formal_roots())
    start, start_root = _start_record()

    def runner(call: supervisor.EnumerationInvocationV1):
        call.output_parent.mkdir(mode=0o700, exist_ok=False)
        return supervisor.EnumerationRunResultV1(
            invocation=call,
            report=_report(
                call.implementation,
                program_root=("66" * 32 if call.implementation == "rust" else "44" * 32),
            ),
            started_at_unix_seconds=110,
            finished_at_unix_seconds=120,
            process_exit_code=0,
        )

    with pytest.raises(
        supervisor.M3DualEnumerationSupervisorError,
        match=supervisor.FAIL_DUAL,
    ) as caught:
        supervisor._run_m3_dual_enumeration_core_v1(
            qualified_gate_evidence=gate,  # type: ignore[arg-type]
            execution_candidate_fields=GATE_INPUTS.execution_candidate_fields,
            run_genesis_fields=_run_genesis(),
            start_record_fields=start,
            start_record_root=start_root,
            implementation_qualification_receipt=_qualification_receipt(),
            committed_golden=_golden(),
            output_root=linux_tmp_path / "mismatch",
            runner=runner,
            clock=iter((130, 131)).__next__,
        )
    assert caught.value.code == supervisor.FAIL_DUAL


def test_runner_report_mutation_after_validation_cannot_change_formal_receipts(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_public_validation_fakes(monkeypatch)
    gate = SimpleNamespace(formal_roots=_formal_roots())
    start, start_root = _start_record()
    runner_reports: dict[str, dict[str, object]] = {}

    def runner(call: supervisor.EnumerationInvocationV1):
        call.output_parent.mkdir(mode=0o700, exist_ok=False)
        report = _report(call.implementation)
        runner_reports[call.implementation] = report
        return supervisor.EnumerationRunResultV1(
            invocation=call,
            report=report,
            started_at_unix_seconds=110,
            finished_at_unix_seconds=120,
            process_exit_code=0,
        )

    def mutate_runner_owned_reports(*_args, **_kwargs):
        for report in runner_reports.values():
            report["raw_operator_application_count"] = 0
            report["canonical_program_count"] = 0
            report["canonical_program_archive_root_or_null"] = "ff" * 32
        return _fake_replayed_archive()

    monkeypatch.setattr(
        supervisor._qualification,
        "_host_validate_enumerator_archive_v1",
        mutate_runner_owned_reports,
    )
    outcome = supervisor._run_m3_dual_enumeration_core_v1(
        qualified_gate_evidence=gate,  # type: ignore[arg-type]
        execution_candidate_fields=GATE_INPUTS.execution_candidate_fields,
        run_genesis_fields=_run_genesis(),
        start_record_fields=start,
        start_record_root=start_root,
        implementation_qualification_receipt=_qualification_receipt(),
        committed_golden=_golden(),
        output_root=linux_tmp_path / "mutation-detached",
        runner=runner,
        clock=iter((130, 131)).__next__,
    )
    assert runner_reports["python"]["canonical_program_count"] == 0
    assert outcome.python_receipt_fields["raw_operator_application_count"] == 3_292_439
    assert outcome.python_receipt_fields["canonical_program_count"] == 50_000
    assert outcome.python_receipt_fields[
        "canonical_program_archive_root_or_null"
    ] == bytes.fromhex("44" * 32)


def test_reserved_raw_cap_budget_requires_runtime_requalification(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_public_validation_fakes(monkeypatch)
    gate = SimpleNamespace(formal_roots=_formal_roots())
    start, start_root = _start_record()

    def runner(call: supervisor.EnumerationInvocationV1):
        call.output_parent.mkdir(mode=0o700, exist_ok=False)
        report = _budget_report(call.implementation)
        _write_budget_archive(call.output_parent, report)
        return supervisor.EnumerationRunResultV1(
            invocation=call,
            report=report,
            started_at_unix_seconds=110,
            finished_at_unix_seconds=120,
            process_exit_code=0,
        )

    with pytest.raises(supervisor.M3DualEnumerationSupervisorError) as caught:
        supervisor._run_m3_dual_enumeration_core_v1(
            qualified_gate_evidence=gate,  # type: ignore[arg-type]
            execution_candidate_fields=GATE_INPUTS.execution_candidate_fields,
            run_genesis_fields=_run_genesis(),
            start_record_fields=start,
            start_record_root=start_root,
            implementation_qualification_receipt=_qualification_receipt(),
            committed_golden=_golden(),
            output_root=linux_tmp_path / "budget",
            runner=runner,
            clock=iter((130, 131)).__next__,
        )
    assert caught.value.code == supervisor.FAIL_RUNNER
    assert "runtime requalification is required" in caught.value.detail


@pytest.mark.parametrize("failure_kind", ("report", "archive", "nonzero"))
def test_single_implementation_strict_failure_is_runner_not_dual(
    linux_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    _install_public_validation_fakes(monkeypatch)
    gate = SimpleNamespace(formal_roots=_formal_roots())
    start, start_root = _start_record()

    if failure_kind == "archive":
        def replay_archive(*_args, implementation: str, **_kwargs):
            if implementation == "python":
                raise OSError("deliberately missing single archive")
            return _fake_replayed_archive()

        monkeypatch.setattr(
            supervisor._qualification,
            "_host_validate_enumerator_archive_v1",
            replay_archive,
        )

    def runner(call: supervisor.EnumerationInvocationV1):
        call.output_parent.mkdir(mode=0o700, exist_ok=False)
        report = _report(call.implementation)
        if failure_kind == "report" and call.implementation == "python":
            report["schema_version"] = "malformed-single-report"
        return supervisor.EnumerationRunResultV1(
            invocation=call,
            report=report,
            started_at_unix_seconds=110,
            finished_at_unix_seconds=120,
            process_exit_code=(
                2
                if failure_kind == "nonzero" and call.implementation == "python"
                else 0
            ),
        )

    with pytest.raises(supervisor.M3DualEnumerationSupervisorError) as caught:
        supervisor._run_m3_dual_enumeration_core_v1(
            qualified_gate_evidence=gate,  # type: ignore[arg-type]
            execution_candidate_fields=GATE_INPUTS.execution_candidate_fields,
            run_genesis_fields=_run_genesis(),
            start_record_fields=start,
            start_record_root=start_root,
            implementation_qualification_receipt=_qualification_receipt(),
            committed_golden=_golden(),
            output_root=linux_tmp_path / f"single-{failure_kind}",
            runner=runner,
            clock=iter((130, 131)).__next__,
        )
    assert caught.value.code == supervisor.FAIL_RUNNER


def test_cross_archive_byte_mismatch_after_independent_replay_is_dual(
    linux_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_public_validation_fakes(monkeypatch)
    gate = SimpleNamespace(formal_roots=_formal_roots())
    start, start_root = _start_record()
    monkeypatch.setattr(
        supervisor._qualification,
        "_host_validate_enumerator_archive_v1",
        lambda *_args, implementation, **_kwargs: _fake_replayed_archive(
            implementation.encode("ascii")
        ),
    )

    def runner(call: supervisor.EnumerationInvocationV1):
        call.output_parent.mkdir(mode=0o700, exist_ok=False)
        return supervisor.EnumerationRunResultV1(
            invocation=call,
            report=_report(call.implementation),
            started_at_unix_seconds=110,
            finished_at_unix_seconds=120,
            process_exit_code=0,
        )

    with pytest.raises(supervisor.M3DualEnumerationSupervisorError) as caught:
        supervisor._run_m3_dual_enumeration_core_v1(
            qualified_gate_evidence=gate,  # type: ignore[arg-type]
            execution_candidate_fields=GATE_INPUTS.execution_candidate_fields,
            run_genesis_fields=_run_genesis(),
            start_record_fields=start,
            start_record_root=start_root,
            implementation_qualification_receipt=_qualification_receipt(),
            committed_golden=_golden(),
            output_root=linux_tmp_path / "dual-archive-mismatch",
            runner=runner,
            clock=iter((130, 131)).__next__,
        )
    assert caught.value.code == supervisor.FAIL_DUAL


@pytest.mark.parametrize("tamper", ("raw_count", "nonempty_stream"))
def test_malformed_finalized_budget_is_runner_failure(
    linux_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    _install_public_validation_fakes(monkeypatch)
    gate = SimpleNamespace(formal_roots=_formal_roots())
    start, start_root = _start_record()

    def runner(call: supervisor.EnumerationInvocationV1):
        call.output_parent.mkdir(mode=0o700, exist_ok=False)
        report = _budget_report(call.implementation)
        if tamper == "raw_count" and call.implementation == "python":
            report["raw_operator_application_count"] = 4_999_999
        _write_budget_archive(
            call.output_parent,
            report,
            nonempty_stream=(
                "canonical_program_records"
                if tamper == "nonempty_stream" and call.implementation == "python"
                else None
            ),
        )
        return supervisor.EnumerationRunResultV1(
            invocation=call,
            report=report,
            started_at_unix_seconds=110,
            finished_at_unix_seconds=120,
            process_exit_code=0,
        )

    with pytest.raises(supervisor.M3DualEnumerationSupervisorError) as caught:
        supervisor._run_m3_dual_enumeration_core_v1(
            qualified_gate_evidence=gate,  # type: ignore[arg-type]
            execution_candidate_fields=GATE_INPUTS.execution_candidate_fields,
            run_genesis_fields=_run_genesis(),
            start_record_fields=start,
            start_record_root=start_root,
            implementation_qualification_receipt=_qualification_receipt(),
            committed_golden=_golden(),
            output_root=linux_tmp_path / f"bad-budget-{tamper}",
            runner=runner,
            clock=iter((130, 131)).__next__,
        )
    assert caught.value.code == supervisor.FAIL_RUNNER


@pytest.mark.parametrize(
    ("tamper", "expected_code"),
    (("illegal_start", supervisor.FAIL_START), ("candidate_identity", supervisor.FAIL_BINDING)),
)
def test_illegal_start_or_execution_identity_is_rejected_before_runner(
    linux_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
    expected_code: str,
) -> None:
    _install_public_validation_fakes(monkeypatch)
    gate = SimpleNamespace(formal_roots=_formal_roots())
    start, start_root = _start_record()
    candidate = dict(GATE_INPUTS.execution_candidate_fields)
    if tamper == "illegal_start":
        start["transition_reason_id"] = 2
    else:
        candidate["canonical_program_budget"] = 49_999
    runner_called = False

    def forbidden_runner(_call: supervisor.EnumerationInvocationV1):
        nonlocal runner_called
        runner_called = True
        raise AssertionError("runner must not start for invalid formal identity")

    with pytest.raises(supervisor.M3DualEnumerationSupervisorError) as caught:
        supervisor._run_m3_dual_enumeration_core_v1(
            qualified_gate_evidence=gate,  # type: ignore[arg-type]
            execution_candidate_fields=candidate,
            run_genesis_fields=_run_genesis(),
            start_record_fields=start,
            start_record_root=start_root,
            implementation_qualification_receipt=_qualification_receipt(),
            committed_golden=_golden(),
            output_root=linux_tmp_path / f"rejected-{tamper}",
            runner=forbidden_runner,
            clock=iter((130, 131)).__next__,
        )
    assert caught.value.code == expected_code
    assert runner_called is False


def test_unsafe_terminalization_has_precedence_over_ordinary_peer_failure(
    linux_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_public_validation_fakes(monkeypatch)
    gate = SimpleNamespace(formal_roots=_formal_roots())
    start, start_root = _start_record()

    class UnsafeTerminalization(RuntimeError):
        code = "FAIL_M3_FORMAL_RUNNER_UNSAFE_TERMINALIZATION"

    def runner(call: supervisor.EnumerationInvocationV1):
        if call.implementation == "python":
            raise RuntimeError("ordinary peer failure")
        raise UnsafeTerminalization("named container may still be running")

    with pytest.raises(UnsafeTerminalization):
        supervisor._run_m3_dual_enumeration_core_v1(
            qualified_gate_evidence=gate,  # type: ignore[arg-type]
            execution_candidate_fields=GATE_INPUTS.execution_candidate_fields,
            run_genesis_fields=_run_genesis(),
            start_record_fields=start,
            start_record_root=start_root,
            implementation_qualification_receipt=_qualification_receipt(),
            committed_golden=_golden(),
            output_root=linux_tmp_path / "unsafe-precedence",
            runner=runner,
            clock=iter((130, 131)).__next__,
        )
