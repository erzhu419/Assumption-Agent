from __future__ import annotations

import hashlib
import multiprocessing
import os
from pathlib import Path
import shutil
import stat
import tempfile

import pytest

from hegel_machine.phase3_m25_commit_b_publication_audit_v1 import canonical_json_v1
import hegel_machine.phase3_m3_dual_enumeration_supervisor_v1 as supervisor
import hegel_machine.phase3_m3_formal_execution_v1 as formal
import hegel_machine.phase3_m3_local_admission_v1 as local_admission
import hegel_machine.phase3_m3_offline_docker_runner_v1 as offline_runner
import hegel_machine.phase3_m3_start_v1 as start


RECORDED_AT = 1_785_779_400


def _fake_local_admission() -> local_admission.LocalTwoCommitAdmissionResultV1:
    runtime_commit_c = "c" * 40
    approval_commit_d = "d" * 40
    paths = local_admission.M3_RUNTIME_SOURCE_PATHS
    artifact = {
        "publication_commit_b": start.PUBLICATION_COMMIT_B,
        "basis_commit_a": local_admission.BASIS_COMMIT_A,
    }
    manifest = {
        "runtime_source_files": [
            {"repository_path": path} for path in paths
        ],
    }
    receipt = {
        "runtime_commit_c": runtime_commit_c,
        "approval_commit_d": approval_commit_d,
        "formal_run_id_hex": start.FORMAL_RUN_ID_HEX,
        "execution_manifest_root_hex": local_admission.EXECUTION_MANIFEST_ROOT_HEX,
    }
    return local_admission.LocalTwoCommitAdmissionResultV1(
        runtime_commit_c=runtime_commit_c,
        approval_commit_d=approval_commit_d,
        artifact_fields=artifact,
        manifest_fields=manifest,
        receipt_fields=receipt,
    )


def _execution_lease_process_worker(run_root_text: str, channel: object) -> None:
    directory_descriptor: int | None = None
    lock_descriptor: int | None = None
    try:
        channel.send("ATTEMPTING")  # type: ignore[attr-defined]
        directory_descriptor, lock_descriptor = formal._acquire_execution_lease_v1(
            Path(run_root_text)
        )
        channel.send("ACQUIRED")  # type: ignore[attr-defined]
        channel.recv()  # type: ignore[attr-defined]
    finally:
        if directory_descriptor is not None and lock_descriptor is not None:
            formal._release_execution_lease_v1(
                directory_descriptor,
                lock_descriptor,
            )
        channel.close()  # type: ignore[attr-defined]


@pytest.fixture
def linux_tmp_path() -> Path:
    with tempfile.TemporaryDirectory(prefix="hegel-m3-formal-test-", dir="/tmp") as raw:
        path = Path(raw).resolve()
        path.chmod(0o700)
        yield path


@pytest.fixture(scope="module")
def prepared_execution() -> formal.PreparedM3FormalExecutionV1:
    patcher = pytest.MonkeyPatch()
    admitted = _fake_local_admission()
    patcher.setattr(
        local_admission,
        "validate_local_admission_receipt_v1",
        lambda *_args, **_kwargs: None,
    )
    patcher.setattr(
        local_admission,
        "validate_live_local_admission_v1",
        lambda *_args, **_kwargs: admitted,
    )
    old_parent = start.FORMAL_RUN_PARENT
    with tempfile.TemporaryDirectory(
        prefix="hegel-m3-formal-prepared-", dir="/tmp"
    ) as raw:
        state_parent = Path(raw).resolve()
        state_parent.chmod(0o700)
        start.FORMAL_RUN_PARENT = state_parent
        try:
            commit, evidence, promotion = start.load_publication_blobs_v1()
            prepared_start = start.prepare_authoritative_m3_start_v1(
                evidence,
                promotion,
                publication_commit=commit,
                recorded_at_unix_seconds=RECORDED_AT,
            )
            state_path = start.canonical_start_state_path_v1(start.FORMAL_RUN_ID_HEX)
            assert (
                start.write_state_exact_once_v1(
                    state_path,
                    prepared_start,
                    local_admission=admitted,
                )
                == "STARTED_NEW"
            )
            prepared = formal.prepare_formal_execution_v1(
                state_path,
                evidence,
                promotion,
                publication_commit=commit,
                expected_admission_revision=admitted.approval_commit_d,
            )
            yield prepared
        finally:
            start.FORMAL_RUN_PARENT = old_parent
            patcher.undo()


def _self_hash(document: dict[str, object]) -> None:
    document.pop("outcome_artifact_sha256", None)
    document["outcome_artifact_sha256"] = hashlib.sha256(
        canonical_json_v1(document)
    ).hexdigest()


def _forge_unsafe_terminal_document(
    document: dict[str, object],
) -> dict[str, object]:
    unsafe = dict(document)
    objects = document["formal_objects"]
    assert type(objects) is dict
    partial_entry = objects["partial_diagnostic_bundle"]
    terminal_entry = objects["terminal_state_record"]
    assert type(partial_entry) is dict and type(terminal_entry) is dict
    partial = dict(
        formal.decode_formal_object(
            bytes.fromhex(partial_entry["cbor_hex"]),
            expected_name="PartialDiagnosticBundleV1",
        ).fields
    )
    partial["terminal_failure_code_id_digest"] = formal.id_digest_v1(
        formal.FAIL_TERMINALIZE
    )
    changed_partial = formal._formal_entry("PartialDiagnosticBundleV1", partial)
    terminal = dict(
        formal.decode_formal_object(
            bytes.fromhex(terminal_entry["cbor_hex"]),
            expected_name="M3RunStateRecordV1",
        ).fields
    )
    terminal["triggering_receipt_root_or_null"] = bytes.fromhex(
        changed_partial["content_root_hex"]
    )
    changed_terminal = formal._formal_entry("M3RunStateRecordV1", terminal)
    unsafe["failure_code"] = formal.FAIL_TERMINALIZE
    unsafe["failure_code_id_digest_hex"] = formal.id_digest_v1(
        formal.FAIL_TERMINALIZE
    ).hex()
    unsafe["formal_objects"] = {
        "partial_diagnostic_bundle": changed_partial,
        "terminal_state_record": changed_terminal,
    }
    _self_hash(unsafe)
    return unsafe


def _preflight(
    prepared: formal.PreparedM3FormalExecutionV1,
    *,
    attempt_intent_sha256: str,
    python_probe_stdout_sha256: str,
    python_probe_start_sha256: str,
    python_probe_completion_sha256: str,
) -> dict[str, object]:
    attempt_root = (
        start.canonical_run_root_v1(prepared.start_document["run_id_hex"])
        / "attempts"
        / formal.CANONICAL_ATTEMPT_ID
    )
    token = hashlib.sha256(attempt_root.as_posix().encode("utf-8")).hexdigest()[:16]
    return {
        "basis_commit": supervisor.COMMIT_A,
        "python_source_root": supervisor.FROZEN_IMPLEMENTATIONS[
            "python"
        ].source_root.hex(),
        "rust_source_root": supervisor.FROZEN_IMPLEMENTATIONS[
            "rust"
        ].source_root.hex(),
        "python_input_tree_sha256": "11" * 32,
        "rust_input_tree_sha256": "22" * 32,
        "all_immutable_inputs_sha256": "33" * 32,
        "rust_binary_sha256": supervisor.FROZEN_IMPLEMENTATIONS[
            "rust"
        ].binary_digest.hex(),
        "runtime_seccomp_sha256": "44" * 32,
        "python_probe_stdout_sha256": python_probe_stdout_sha256,
        "python_probe_start_sha256": python_probe_start_sha256,
        "python_probe_completion_sha256": python_probe_completion_sha256,
        "python_probe_container_name": f"hegel-m3-{token}-python-probe",
        "docker_daemon_receipt_binding": prepared.implementation_qualification_receipt[
            "local_docker_daemon_receipt_binding"
        ],
        "pull_policy": "never",
        "network_mode": "none",
        "maximum_enumeration_seconds": offline_runner.MAX_ENUMERATION_SECONDS,
        "container_names": {
            name: f"hegel-m3-{token}-{name}" for name in ("python", "rust")
        },
        "attempt_intent_sha256": attempt_intent_sha256,
    }


def _write_runner_evidence_fixture(
    attempt_root: Path,
    prepared: formal.PreparedM3FormalExecutionV1,
) -> dict[str, object]:
    (attempt_root / "immutable-inputs").mkdir(mode=0o700)
    qualification_root = prepared.implementation_qualification_receipt[
        "receipt_root"
    ]
    token = hashlib.sha256(attempt_root.as_posix().encode("utf-8")).hexdigest()[:16]
    container_names = {
        name: f"hegel-m3-{token}-{name}" for name in ("python", "rust")
    }
    probe_container_name = f"hegel-m3-{token}-python-probe"
    intent_document = {
        "schema": formal.ATTEMPT_INTENT_SCHEMA,
        "basis_commit": supervisor.COMMIT_A,
        "attempt_root_path_sha256": hashlib.sha256(
            attempt_root.as_posix().encode("utf-8")
        ).hexdigest(),
        "implementation_qualification_receipt_root": qualification_root,
        "python_implementation_binding_root": supervisor.FROZEN_IMPLEMENTATIONS[
            "python"
        ].implementation_binding_root.hex(),
        "rust_implementation_binding_root": supervisor.FROZEN_IMPLEMENTATIONS[
            "rust"
        ].implementation_binding_root.hex(),
        "all_immutable_inputs_sha256": "33" * 32,
        "enumeration_output_relative_path": "formal-enumeration",
        "pull_policy": "never",
        "network_mode": "none",
        "restart_policy": offline_runner.RESTART_POLICY,
        "failure_cleanup_policy": offline_runner.FAILURE_CLEANUP_POLICY,
        "maximum_enumeration_seconds": offline_runner.MAX_ENUMERATION_SECONDS,
        "container_names": container_names,
        "python_probe_container_name": probe_container_name,
        "python_probe_maximum_seconds": offline_runner.PYTHON_PROBE_MAXIMUM_SECONDS,
        "python_probe_auto_remove": False,
    }
    intent = canonical_json_v1(intent_document)
    (attempt_root / "runner-attempt-intent.json").write_bytes(intent)
    journal = attempt_root / "runner-journal"
    journal.mkdir(mode=0o700)
    probe_stdout = canonical_json_v1(
        {
            "binary_path": "/usr/local/bin/python3.12",
            "binary_sha256": supervisor.FROZEN_IMPLEMENTATIONS[
                "python"
            ].binary_digest.hex(),
            "version": "3.12.11 (synthetic formal fixture)",
        }
    )
    (attempt_root / "python-runtime-probe-stdout.json").write_bytes(probe_stdout)
    probe_started_at = RECORDED_AT + 2
    probe_finished_at = RECORDED_AT + 3
    probe_start = canonical_json_v1(
        {
            "schema": offline_runner.PROBE_START_SCHEMA,
            "container_name": probe_container_name,
            "attempt_intent_sha256": hashlib.sha256(intent).hexdigest(),
            "image_ref": supervisor.FROZEN_IMPLEMENTATIONS["python"].image_ref,
            "started_at_unix_seconds": probe_started_at,
        }
    )
    (journal / "python-probe-started.json").write_bytes(probe_start)
    probe_completion = canonical_json_v1(
        {
            "schema": offline_runner.PROBE_COMPLETION_SCHEMA,
            "container_name": probe_container_name,
            "attempt_intent_sha256": hashlib.sha256(intent).hexdigest(),
            "image_ref": supervisor.FROZEN_IMPLEMENTATIONS["python"].image_ref,
            "binary_path": "/usr/local/bin/python3.12",
            "binary_sha256": supervisor.FROZEN_IMPLEMENTATIONS[
                "python"
            ].binary_digest.hex(),
            "version_sha256": hashlib.sha256(
                b"3.12.11 (synthetic formal fixture)"
            ).hexdigest(),
            "stdout_sha256": hashlib.sha256(probe_stdout).hexdigest(),
            "started_at_unix_seconds": probe_started_at,
            "finished_at_unix_seconds": probe_finished_at,
            "docker_started_at": "2026-08-08T00:00:00Z",
            "docker_finished_at": "2026-08-08T00:00:01Z",
        }
    )
    (journal / "python-probe-completed.json").write_bytes(probe_completion)
    preflight = _preflight(
        prepared,
        attempt_intent_sha256=hashlib.sha256(intent).hexdigest(),
        python_probe_stdout_sha256=hashlib.sha256(probe_stdout).hexdigest(),
        python_probe_start_sha256=hashlib.sha256(probe_start).hexdigest(),
        python_probe_completion_sha256=hashlib.sha256(
            probe_completion
        ).hexdigest(),
    )
    (attempt_root / "runner-preflight.json").write_bytes(
        canonical_json_v1(preflight)
    )
    return preflight


def _synthetic_report(implementation: str) -> dict[str, object]:
    return {
        "implementation": implementation,
        "closure_status": "DSL_TOO_LARGE",
        "closure_status_id": 2,
        "raw_operator_application_count": 3_292_439,
        "canonical_program_count": 50_000,
        "closure_cardinality_or_null": None,
        "frontier_exhausted": False,
        "all_type_buckets_closed": False,
        "raw_expansion_limit_hit": False,
        "wall_clock_abort_hit": False,
        "canonical_program_archive_root_or_null": "51" * 32,
        "program_chunk_manifest_root_or_null": "52" * 32,
        "bucket_accounting_root_or_null": "53" * 32,
        "first_out_of_budget_program_hash_or_null": "54" * 32,
    }


def _synthetic_budget_report(
    prepared: formal.PreparedM3FormalExecutionV1,
    implementation: str,
) -> dict[str, object]:
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
    roots = prepared.replay.qualified_gate_evidence.formal_roots
    report: dict[str, object] = {
        "schema_version": identity[0],
        "claim_level": "FORMAL_PROFILE_CANDIDATE_NOT_AUTHORITY",
        "authoritative_claim_allowed": False,
        "implementation": implementation,
        "implementation_id": identity[1],
        "implementation_machine_id": identity[2],
        "raw_expansion_limit_hit": True,
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
        "canonical_program_archive_root": (
            "canonical_program_archive_root_or_null"
        ),
        "first_out_of_budget_program_cbor_hex": (
            "first_out_of_budget_program_cbor_hex_or_null"
        ),
        "first_out_of_budget_program_hash": (
            "first_out_of_budget_program_hash_or_null"
        ),
        "program_chunk_manifest_root": "program_chunk_manifest_root_or_null",
    }
    expected = dict(prepared.committed_golden["expected"])
    for key, value in expected.items():
        report[renamed.get(key, key)] = value
    report.update(
        {
            "closure_status": "INCONCLUSIVE_BUDGET",
            "closure_status_id": 3,
            "raw_operator_application_count": (
                supervisor.RAW_OPERATOR_APPLICATION_CAP
            ),
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


def _write_archive_fixture(
    output: Path,
    implementation: str,
    report: dict[str, object],
    *,
    budget: bool = False,
) -> None:
    archive = output / implementation / "archive"
    archive.mkdir(mode=0o700, parents=True)
    files = {
        "report.json": canonical_json_v1(report),
        "canonical_program_records.cborframed": b"" if budget else b"programs",
        "program_chunk_manifests.cborframed": b"" if budget else b"chunks",
        "bucket_accounting_records.cborframed": b"" if budget else b"buckets",
    }
    for name, payload in files.items():
        (archive / name).write_bytes(payload)
    (output / implementation / "execution-stdout.json").write_bytes(
        canonical_json_v1(report)
    )
    (output / implementation / "execution-stderr.bin").write_bytes(b"")
    (output / implementation / "process-completion.json").write_bytes(b"{}\n")


def _synthetic_dual_outcome(
    prepared: formal.PreparedM3FormalExecutionV1,
    output_root: Path,
    preflight: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    *,
    budget: bool = False,
) -> supervisor.M3DualEnumerationOutcomeV1:
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
        lambda *_args, **_kwargs: b"qualified" * 4,
    )
    monkeypatch.setattr(
        supervisor._qualification,
        "validate_enumerator_report_v1",
        lambda report, **_kwargs: dict(report),
    )
    def replay_archive(
        output_parent: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        archive = output_parent / "archive"
        return {
            "streams": {
                "canonical_program_records": (
                    archive / "canonical_program_records.cborframed"
                ).read_bytes(),
                "program_chunk_manifests": (
                    archive / "program_chunk_manifests.cborframed"
                ).read_bytes(),
                "bucket_accounting_records": (
                    archive / "bucket_accounting_records.cborframed"
                ).read_bytes(),
            },
            "report_payload": (archive / "report.json").read_bytes(),
            "witness_adjacency_verified": not budget,
        }

    monkeypatch.setattr(
        supervisor._qualification,
        "_host_validate_enumerator_archive_v1",
        replay_archive,
    )
    monkeypatch.setattr(
        supervisor._qualification,
        "_validate_dual_archive_bytes_equal_v1",
        lambda *_args, **_kwargs: None,
    )

    def runner(
        invocation: supervisor.EnumerationInvocationV1,
    ) -> supervisor.EnumerationRunResultV1:
        report = (
            _synthetic_budget_report(prepared, invocation.implementation)
            if budget
            else _synthetic_report(invocation.implementation)
        )
        _write_archive_fixture(
            output_root,
            invocation.implementation,
            report,
            budget=budget,
        )
        offset = 0 if invocation.implementation == "python" else 1
        started_at = RECORDED_AT + 10 + offset
        finished_at = RECORDED_AT + 20 + offset
        attempt_root = output_root.parent
        invocation_sha256 = formal._invocation_digest_v1(
            invocation,
            attempt_root=attempt_root,
        )
        common = {
            "implementation": invocation.implementation,
            "implementation_id": invocation.implementation_id,
            "container_name": preflight["container_names"][
                invocation.implementation
            ],
            "invocation_sha256": invocation_sha256,
            "attempt_intent_sha256": preflight["attempt_intent_sha256"],
        }
        started_document = {
            "schema": formal.START_MARKER_SCHEMA,
            **common,
            "started_at_unix_seconds": started_at,
        }
        (attempt_root / f"runner-journal/{invocation.implementation}-started.json").write_bytes(
            canonical_json_v1(started_document)
        )
        stdout_path = output_root / invocation.implementation / "execution-stdout.json"
        stderr_path = output_root / invocation.implementation / "execution-stderr.bin"
        process_document = {
            "schema": formal.COMPLETION_MARKER_SCHEMA,
            **common,
            "started_at_unix_seconds": started_at,
            "finished_at_unix_seconds": finished_at,
            "process_exit_code": 0,
            "stdout_sha256": hashlib.sha256(stdout_path.read_bytes()).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr_path.read_bytes()).hexdigest(),
            "pull_policy": "never",
            "network_mode": "none",
            "docker_started_at": "2026-08-08T00:00:00Z",
            "docker_finished_at": "2026-08-08T00:00:01Z",
            "docker_oom_killed": False,
            "docker_error": "",
        }
        process_payload = canonical_json_v1(process_document)
        (output_root / invocation.implementation / "process-completion.json").write_bytes(
            process_payload
        )
        completed_document = {
            "schema": formal.JOURNAL_COMPLETION_SCHEMA,
            **common,
            "started_at_unix_seconds": started_at,
            "finished_at_unix_seconds": finished_at,
            "process_completion_sha256": hashlib.sha256(process_payload).hexdigest(),
        }
        (attempt_root / f"runner-journal/{invocation.implementation}-completed.json").write_bytes(
            canonical_json_v1(completed_document)
        )
        return supervisor.EnumerationRunResultV1(
            invocation=invocation,
            report=report,
            started_at_unix_seconds=started_at,
            finished_at_unix_seconds=finished_at,
            process_exit_code=0,
        )

    return supervisor._run_m3_dual_enumeration_core_v1(
        qualified_gate_evidence=prepared.replay.qualified_gate_evidence,
        execution_candidate_fields=prepared.replay.gate_inputs.execution_candidate_fields,
        run_genesis_fields=prepared.replay.gate_inputs.run_genesis_fields,
        start_record_fields=prepared.start_record_fields,
        start_record_root=prepared.start_record_root,
        implementation_qualification_receipt=(
            prepared.implementation_qualification_receipt
        ),
        committed_golden=prepared.committed_golden,
        output_root=output_root,
        runner=runner,
        clock=iter((RECORDED_AT + 30, RECORDED_AT + 31)).__next__,
    )


def test_prepare_uses_commit_b_publication_with_commit_a_qualification_basis(
    prepared_execution: formal.PreparedM3FormalExecutionV1,
) -> None:
    prepared = prepared_execution
    assert prepared.replay.publication_commit == start.PUBLICATION_COMMIT_B
    assert (
        prepared.implementation_qualification_receipt["basis_commit"]
        == supervisor.COMMIT_A
    )
    assert prepared.persisted_start_path == start.canonical_start_state_path_v1(
        start.FORMAL_RUN_ID_HEX
    )


def test_prepare_rejects_state_without_explicit_start_publication_receipt(
    linux_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_parent = (linux_tmp_path / "missing-start-receipt").resolve()
    state_parent.mkdir(mode=0o700)
    monkeypatch.setattr(start, "FORMAL_RUN_PARENT", state_parent)
    commit, evidence, promotion = start.load_publication_blobs_v1()
    prepared_start = start.prepare_authoritative_m3_start_v1(
        evidence,
        promotion,
        publication_commit=commit,
        recorded_at_unix_seconds=RECORDED_AT,
    )
    admitted = _fake_local_admission()
    monkeypatch.setattr(
        local_admission,
        "validate_local_admission_receipt_v1",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        local_admission,
        "validate_live_local_admission_v1",
        lambda *_args, **_kwargs: admitted,
    )
    state_path = start.canonical_start_state_path_v1(start.FORMAL_RUN_ID_HEX)
    assert (
        start.write_state_exact_once_v1(
            state_path,
            prepared_start,
            local_admission=admitted,
        )
        == "STARTED_NEW"
    )
    start.canonical_start_publication_receipt_path_v1(
        start.FORMAL_RUN_ID_HEX
    ).unlink()

    with pytest.raises(formal.M3FormalExecutionError) as caught:
        formal.prepare_formal_execution_v1(
            state_path,
            evidence,
            promotion,
            publication_commit=commit,
            expected_admission_revision=admitted.approval_commit_d,
        )
    assert caught.value.code == formal.FAIL_INPUT


def test_failure_outcome_has_exact_semantic_or_execution_terminal_mapping(
    prepared_execution: formal.PreparedM3FormalExecutionV1,
) -> None:
    prepared = prepared_execution
    attempt_root = (
        start.canonical_run_root_v1(start.FORMAL_RUN_ID_HEX)
        / "attempts"
        / formal.CANONICAL_ATTEMPT_ID
    )
    semantic = formal.build_failure_outcome_document_v1(
        prepared,
        attempt_id=formal.CANONICAL_ATTEMPT_ID,
        attempt_root=attempt_root,
        error=supervisor.M3DualEnumerationSupervisorError(
            supervisor.FAIL_DUAL, "deliberate mismatch"
        ),
        preflight_receipt_or_null=None,
    )
    assert semantic["terminal_status"] == "INCONCLUSIVE_SEMANTICS"
    execution = formal.build_failure_outcome_document_v1(
        prepared,
        attempt_id=formal.CANONICAL_ATTEMPT_ID,
        attempt_root=attempt_root,
        error=RuntimeError("stable runner failure"),
        preflight_receipt_or_null=None,
    )
    assert execution["terminal_status"] == "INCONCLUSIVE_EXECUTION"

    unsafe_error = offline_runner.M3OfflineDockerRunnerError(
        offline_runner.FAIL_TERMINALIZE,
        "container state unavailable",
    )
    with pytest.raises(formal.M3FormalExecutionError) as captured:
        formal.build_failure_outcome_document_v1(
            prepared,
            attempt_id=formal.CANONICAL_ATTEMPT_ID,
            attempt_root=attempt_root,
            error=unsafe_error,
            preflight_receipt_or_null=None,
        )
    assert captured.value.code == formal.FAIL_TERMINALIZE

    unsafe_document = _forge_unsafe_terminal_document(execution)
    with pytest.raises(formal.M3FormalExecutionError) as captured:
        formal.validate_failure_outcome_document_v1(
            unsafe_document,
            prepared=prepared,
        )
    assert captured.value.code == formal.FAIL_TERMINALIZE

    tampered = dict(semantic)
    tampered["terminal_status"] = "INCONCLUSIVE_EXECUTION"
    _self_hash(tampered)
    with pytest.raises(formal.M3FormalExecutionError):
        formal.validate_failure_outcome_document_v1(tampered, prepared=prepared)


def test_existing_unsafe_containment_outcome_cannot_replay_as_terminal(
    prepared_execution: formal.PreparedM3FormalExecutionV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepared_execution
    run_root = start.canonical_run_root_v1(start.FORMAL_RUN_ID_HEX)
    attempt_root = run_root / "attempts" / formal.CANONICAL_ATTEMPT_ID
    execution = formal.build_failure_outcome_document_v1(
        prepared,
        attempt_id=formal.CANONICAL_ATTEMPT_ID,
        attempt_root=attempt_root,
        error=RuntimeError("stable runner failure"),
        preflight_receipt_or_null=None,
    )
    unsafe_document = _forge_unsafe_terminal_document(execution)
    outcome_path = start.canonical_terminal_outcome_path_v1(
        start.FORMAL_RUN_ID_HEX
    )
    assert not outcome_path.exists() and not outcome_path.is_symlink()
    outcome_path.write_bytes(canonical_json_v1(unsafe_document))
    outcome_path.chmod(0o600)
    monkeypatch.setattr(
        formal,
        "_replay_live_local_admission_identity_v1",
        lambda _prepared: None,
    )
    try:
        with pytest.raises(formal.M3FormalExecutionError) as captured:
            formal.execute_formal_m3_v1(
                prepared,
                run_root=run_root,
                attempt_id=formal.CANONICAL_ATTEMPT_ID,
                outcome_path=outcome_path,
            )
        assert captured.value.code == formal.FAIL_TERMINALIZE
    finally:
        outcome_path.unlink()


def test_post_lease_terminal_recheck_never_enters_docker_runner(
    prepared_execution: formal.PreparedM3FormalExecutionV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepared_execution
    run_root = start.canonical_run_root_v1(start.FORMAL_RUN_ID_HEX)
    outcome_path = start.canonical_terminal_outcome_path_v1(
        start.FORMAL_RUN_ID_HEX
    )
    attempt_root = run_root / "attempts" / formal.CANONICAL_ATTEMPT_ID
    terminal = formal.build_failure_outcome_document_v1(
        prepared,
        attempt_id=formal.CANONICAL_ATTEMPT_ID,
        attempt_root=attempt_root,
        error=RuntimeError("winner terminalized before waiter acquired lease"),
        preflight_receipt_or_null=None,
    )
    original_acquire = formal._acquire_execution_lease_v1

    def acquire_after_winner(root: Path) -> tuple[int, int]:
        directory_descriptor, lock_descriptor = original_acquire(root)
        outcome_path.write_bytes(canonical_json_v1(terminal))
        outcome_path.chmod(0o600)
        return directory_descriptor, lock_descriptor

    monkeypatch.setattr(
        formal,
        "_replay_live_local_admission_identity_v1",
        lambda _prepared: None,
    )
    monkeypatch.setattr(
        formal,
        "_acquire_execution_lease_v1",
        acquire_after_winner,
    )

    def forbidden_runner(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("post-lease terminal waiter entered Docker runner")

    monkeypatch.setattr(formal, "OfflineDockerEnumerationRunnerV1", forbidden_runner)
    try:
        publication = formal.execute_formal_m3_v1(
            prepared,
            run_root=run_root,
            attempt_id=formal.CANONICAL_ATTEMPT_ID,
            outcome_path=outcome_path,
        )
        assert publication.status == "ALREADY_TERMINAL_VERIFIED"
        assert publication.attempt_root is None
        assert publication.document == terminal
    finally:
        outcome_path.unlink(missing_ok=True)


def test_post_lease_admission_replay_precedes_terminal_or_docker(
    prepared_execution: formal.PreparedM3FormalExecutionV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepared_execution
    run_root = start.canonical_run_root_v1(start.FORMAL_RUN_ID_HEX)
    outcome_path = start.canonical_terminal_outcome_path_v1(
        start.FORMAL_RUN_ID_HEX
    )
    outcome_path.unlink(missing_ok=True)
    replay_count = 0

    def admission_changes_while_waiting(
        _prepared: formal.PreparedM3FormalExecutionV1,
    ) -> None:
        nonlocal replay_count
        replay_count += 1
        if replay_count == 2:
            raise formal.M3FormalExecutionError(
                formal.FAIL_BINDING,
                "local admission changed while waiting for the execution lease",
            )

    monkeypatch.setattr(
        formal,
        "_replay_live_start_publication_v1",
        lambda _prepared: None,
    )
    monkeypatch.setattr(
        formal,
        "_replay_live_local_admission_identity_v1",
        admission_changes_while_waiting,
    )

    def forbidden_runner(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("changed post-lease admission entered Docker")

    monkeypatch.setattr(formal, "OfflineDockerEnumerationRunnerV1", forbidden_runner)
    with pytest.raises(formal.M3FormalExecutionError) as captured:
        formal.execute_formal_m3_v1(
            prepared,
            run_root=run_root,
            attempt_id=formal.CANONICAL_ATTEMPT_ID,
            outcome_path=outcome_path,
        )
    assert captured.value.code == formal.FAIL_BINDING
    assert replay_count == 2


def test_success_document_replays_cross_object_identity_and_live_file_hashes(
    prepared_execution: formal.PreparedM3FormalExecutionV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepared_execution
    attempt_root = (
        start.canonical_run_root_v1(start.FORMAL_RUN_ID_HEX)
        / "attempts"
        / formal.CANONICAL_ATTEMPT_ID
    )
    attempt_root.mkdir(mode=0o700, parents=True)
    preflight = _write_runner_evidence_fixture(attempt_root, prepared)
    output_root = attempt_root / "formal-enumeration"
    outcome = _synthetic_dual_outcome(
        prepared,
        output_root,
        preflight,
        monkeypatch,
    )
    document = formal.build_outcome_document_v1(
        prepared,
        outcome,
        attempt_id=formal.CANONICAL_ATTEMPT_ID,
        attempt_root=attempt_root,
        preflight_receipt=preflight,
        enumeration_output_root=output_root,
    )
    formal.validate_outcome_document_v1(document, prepared=prepared)

    def rebound_runner_document(
        *,
        replacement_payloads: dict[str, bytes],
        replacement_preflight: dict[str, object],
    ) -> dict[str, object]:
        preflight_payload = canonical_json_v1(replacement_preflight)
        replacement_payloads = {
            **replacement_payloads,
            "runner-preflight.json": preflight_payload,
        }
        (attempt_root / "runner-preflight.json").write_bytes(preflight_payload)
        candidate = dict(document)
        candidate["offline_runtime_preflight"] = replacement_preflight
        original_rows = document["runner_evidence_files"]
        assert type(original_rows) is list
        rebound_rows: list[dict[str, object]] = []
        for original_row in original_rows:
            assert type(original_row) is dict
            row = dict(original_row)
            relative = row["relative_path"]
            if relative in replacement_payloads:
                payload = replacement_payloads[relative]
                row["byte_length"] = len(payload)
                row["sha256"] = hashlib.sha256(payload).hexdigest()
            rebound_rows.append(row)
        candidate["runner_evidence_files"] = rebound_rows
        _self_hash(candidate)
        return candidate

    intent_path = attempt_root / "runner-attempt-intent.json"
    preflight_path = attempt_root / "runner-preflight.json"
    original_intent_payload = intent_path.read_bytes()
    original_preflight_payload = preflight_path.read_bytes()
    intent_document = dict(
        start.strict_json_loads_v1(
            original_intent_payload,
            label="synthetic runner intent",
        )
    )
    intent_document["python_probe_maximum_seconds"] = (
        offline_runner.PYTHON_PROBE_MAXIMUM_SECONDS + 1
    )
    changed_intent_payload = canonical_json_v1(intent_document)
    changed_preflight = dict(preflight)
    changed_preflight["attempt_intent_sha256"] = hashlib.sha256(
        changed_intent_payload
    ).hexdigest()
    try:
        intent_path.write_bytes(changed_intent_payload)
        changed = rebound_runner_document(
            replacement_payloads={
                "runner-attempt-intent.json": changed_intent_payload,
            },
            replacement_preflight=changed_preflight,
        )
        with pytest.raises(formal.M3FormalExecutionError) as captured:
            formal.validate_outcome_document_v1(changed, prepared=prepared)
        assert "attempt intent differs" in captured.value.detail
    finally:
        intent_path.write_bytes(original_intent_payload)
        preflight_path.write_bytes(original_preflight_payload)

    probe_completion_path = (
        attempt_root / "runner-journal/python-probe-completed.json"
    )
    original_probe_completion = probe_completion_path.read_bytes()
    probe_completion = dict(
        start.strict_json_loads_v1(
            original_probe_completion,
            label="synthetic probe completion",
        )
    )
    probe_completion["stdout_sha256"] = "00" * 32
    changed_probe_completion = canonical_json_v1(probe_completion)
    changed_preflight = dict(preflight)
    changed_preflight["python_probe_completion_sha256"] = hashlib.sha256(
        changed_probe_completion
    ).hexdigest()
    try:
        probe_completion_path.write_bytes(changed_probe_completion)
        changed = rebound_runner_document(
            replacement_payloads={
                "runner-journal/python-probe-completed.json": (
                    changed_probe_completion
                ),
            },
            replacement_preflight=changed_preflight,
        )
        with pytest.raises(formal.M3FormalExecutionError) as captured:
            formal.validate_outcome_document_v1(changed, prepared=prepared)
        assert "runtime probe identity differs" in captured.value.detail
    finally:
        probe_completion_path.write_bytes(original_probe_completion)
        preflight_path.write_bytes(original_preflight_payload)

    probe_stdout_path = attempt_root / "python-runtime-probe-stdout.json"
    original_probe_stdout = probe_stdout_path.read_bytes()
    probe_stdout = dict(
        start.strict_json_loads_v1(
            original_probe_stdout,
            label="synthetic probe stdout",
        )
    )
    probe_stdout["unexpected"] = True
    changed_probe_stdout = canonical_json_v1(probe_stdout)
    changed_preflight = dict(preflight)
    changed_preflight["python_probe_stdout_sha256"] = hashlib.sha256(
        changed_probe_stdout
    ).hexdigest()
    try:
        probe_stdout_path.write_bytes(changed_probe_stdout)
        changed = rebound_runner_document(
            replacement_payloads={
                "python-runtime-probe-stdout.json": changed_probe_stdout,
            },
            replacement_preflight=changed_preflight,
        )
        with pytest.raises(formal.M3FormalExecutionError) as captured:
            formal.validate_outcome_document_v1(changed, prepared=prepared)
        assert "probe stdout schema differs" in captured.value.detail
    finally:
        probe_stdout_path.write_bytes(original_probe_stdout)
        preflight_path.write_bytes(original_preflight_payload)

    tampered = dict(document)
    tampered["run_id_hex"] = "00" * 16
    _self_hash(tampered)
    with pytest.raises(formal.M3FormalExecutionError):
        formal.validate_outcome_document_v1(tampered, prepared=prepared)

    archive_file = output_root / "python/archive/report.json"
    archive_file.write_bytes(b"[]\n")
    with pytest.raises(formal.M3FormalExecutionError):
        formal.validate_outcome_document_v1(document, prepared=prepared)


def test_outcome_publication_is_inode_verified_exact_once(
    linux_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = (linux_tmp_path / "outcome-parent").resolve()
    parent.mkdir(mode=0o700)
    path = parent / "terminal.json"
    first: dict[str, object] = {"value": 1}
    _self_hash(first)
    observed_fchmod_modes: list[int] = []
    real_fchmod = formal.os.fchmod

    def record_fchmod(descriptor: int, mode: int) -> None:
        observed_fchmod_modes.append(mode)
        real_fchmod(descriptor, mode)

    monkeypatch.setattr(formal.os, "fchmod", record_fchmod)
    assert formal._write_outcome_exact_once(path, first) == "TERMINAL_PUBLISHED_NEW"
    assert observed_fchmod_modes == [0o600]
    assert path.stat().st_mode & 0o777 == 0o600
    assert formal._write_outcome_exact_once(path, first) == "ALREADY_TERMINAL_IDENTICAL"
    second: dict[str, object] = {"value": 2}
    _self_hash(second)
    with pytest.raises(formal.M3FormalExecutionError) as captured:
        formal._write_outcome_exact_once(path, second)
    assert captured.value.code == formal.FAIL_ALREADY_TERMINAL
    assert not [item for item in parent.iterdir() if item.name.endswith(".pending")]


def test_outcome_read_rejects_post_read_namespace_rebinding(
    linux_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = (linux_tmp_path / "outcome-read-parent").resolve()
    parent.mkdir(mode=0o700)
    target = parent / "terminal.json"
    decoy = parent / "decoy.json"
    target.write_bytes(b"original")
    target.chmod(0o600)
    decoy.write_bytes(b"original")
    decoy.chmod(0o600)
    directory_descriptor = formal._open_outcome_directory_v1(parent)
    real_stat = formal.os.stat

    def rebound_stat(
        path: object,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        if (
            path == target.name
            and dir_fd == directory_descriptor
            and follow_symlinks is False
        ):
            return real_stat(
                decoy.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(formal.os, "stat", rebound_stat)
    try:
        with pytest.raises(formal.M3FormalExecutionError) as captured:
            formal._read_outcome_at_v1(directory_descriptor, target.name)
        assert captured.value.code == formal.FAIL_OUTPUT
        assert "changed while being read" in captured.value.detail
    finally:
        os.close(directory_descriptor)


def test_pending_outcome_fchmod_failure_leaves_no_publishable_file(
    linux_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = (linux_tmp_path / "outcome-fchmod-parent").resolve()
    parent.mkdir(mode=0o700)
    path = parent / "terminal.json"
    document: dict[str, object] = {"value": 1}
    _self_hash(document)

    def reject_fchmod(_descriptor: int, _mode: int) -> None:
        raise PermissionError("synthetic fchmod rejection")

    monkeypatch.setattr(formal.os, "fchmod", reject_fchmod)
    with pytest.raises(formal.M3FormalExecutionError) as captured:
        formal._write_outcome_exact_once(path, document)
    assert captured.value.code == formal.FAIL_OUTPUT
    assert not path.exists() and not path.is_symlink()
    assert list(parent.iterdir()) == []


def test_attempt_tree_creation_is_dirfd_bounded_and_pinned(
    linux_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = (linux_tmp_path / "attempt-tree-run").resolve()
    run_root.mkdir(mode=0o700)
    lease_directory, lease_descriptor = formal._acquire_execution_lease_v1(run_root)
    attempts_descriptor: int | None = None
    attempt_descriptor: int | None = None

    def forbidden_path_mkdir(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("formal attempt creation escaped to Path.mkdir")

    monkeypatch.setattr(Path, "mkdir", forbidden_path_mkdir)
    try:
        (
            attempt_root,
            attempts_descriptor,
            attempt_descriptor,
        ) = formal._open_or_create_attempt_tree_v1(
            lease_directory,
            run_root,
            formal.CANONICAL_ATTEMPT_ID,
        )
        assert attempt_root == (
            run_root / "attempts" / formal.CANONICAL_ATTEMPT_ID
        )
        formal._assert_pinned_attempt_tree_v1(
            lease_directory,
            attempts_descriptor,
            attempt_descriptor,
            run_root=run_root,
            attempt_id=formal.CANONICAL_ATTEMPT_ID,
        )
        for descriptor in (attempts_descriptor, attempt_descriptor):
            metadata = os.fstat(descriptor)
            assert stat.S_ISDIR(metadata.st_mode)
            assert metadata.st_uid == os.geteuid()
            assert stat.S_IMODE(metadata.st_mode) == 0o700
    finally:
        if attempt_descriptor is not None:
            os.close(attempt_descriptor)
        if attempts_descriptor is not None:
            os.close(attempts_descriptor)
        formal._release_execution_lease_v1(lease_directory, lease_descriptor)


def test_pinned_attempt_tree_rejects_same_mode_namespace_replacement(
    linux_tmp_path: Path,
) -> None:
    run_root = (linux_tmp_path / "rebound-attempt-tree-run").resolve()
    run_root.mkdir(mode=0o700)
    lease_directory, lease_descriptor = formal._acquire_execution_lease_v1(run_root)
    attempts_descriptor: int | None = None
    attempt_descriptor: int | None = None
    try:
        (
            _attempt_root,
            attempts_descriptor,
            attempt_descriptor,
        ) = formal._open_or_create_attempt_tree_v1(
            lease_directory,
            run_root,
            formal.CANONICAL_ATTEMPT_ID,
        )
        os.rename(
            formal.CANONICAL_ATTEMPT_ID,
            "displaced-attempt",
            src_dir_fd=attempts_descriptor,
            dst_dir_fd=attempts_descriptor,
        )
        os.mkdir(
            formal.CANONICAL_ATTEMPT_ID,
            mode=0o700,
            dir_fd=attempts_descriptor,
        )
        with pytest.raises(formal.M3FormalExecutionError) as captured:
            formal._assert_pinned_attempt_tree_v1(
                lease_directory,
                attempts_descriptor,
                attempt_descriptor,
                run_root=run_root,
                attempt_id=formal.CANONICAL_ATTEMPT_ID,
            )
        assert captured.value.code == formal.FAIL_OUTPUT
        assert "namespace identity differs" in captured.value.detail
    finally:
        if attempt_descriptor is not None:
            os.close(attempt_descriptor)
        if attempts_descriptor is not None:
            os.close(attempts_descriptor)
        formal._release_execution_lease_v1(lease_directory, lease_descriptor)


@pytest.mark.parametrize("unsafe_kind", ["symlink", "wrong-mode"])
def test_attempt_tree_rejects_unsafe_existing_attempts_component(
    linux_tmp_path: Path,
    unsafe_kind: str,
) -> None:
    run_root = (linux_tmp_path / f"unsafe-attempt-tree-{unsafe_kind}").resolve()
    run_root.mkdir(mode=0o700)
    outside = (linux_tmp_path / f"outside-{unsafe_kind}").resolve()
    outside.mkdir(mode=0o700)
    attempts = run_root / "attempts"
    if unsafe_kind == "symlink":
        attempts.symlink_to(outside, target_is_directory=True)
    else:
        attempts.mkdir(mode=0o755)
    lease_directory, lease_descriptor = formal._acquire_execution_lease_v1(run_root)
    try:
        with pytest.raises(formal.M3FormalExecutionError) as captured:
            formal._open_or_create_attempt_tree_v1(
                lease_directory,
                run_root,
                formal.CANONICAL_ATTEMPT_ID,
            )
        assert captured.value.code == formal.FAIL_OUTPUT
        assert not (outside / formal.CANONICAL_ATTEMPT_ID).exists()
    finally:
        formal._release_execution_lease_v1(lease_directory, lease_descriptor)


def test_execution_lease_serializes_processes_and_releases_after_crash(
    linux_tmp_path: Path,
) -> None:
    run_root = (linux_tmp_path / "leased-run").resolve()
    run_root.mkdir(mode=0o700)
    parent_directory, parent_lock = formal._acquire_execution_lease_v1(run_root)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=True)
    process = context.Process(
        target=_execution_lease_process_worker,
        args=(run_root.as_posix(), sender),
    )
    process.start()
    sender.close()
    parent_lease_held = True
    try:
        assert receiver.poll(10)
        assert receiver.recv() == "ATTEMPTING"
        assert not receiver.poll(0.5)
        formal._release_execution_lease_v1(parent_directory, parent_lock)
        parent_lease_held = False
        assert receiver.poll(10)
        assert receiver.recv() == "ACQUIRED"

        # Abrupt termination skips the worker's finally block.  flock must be
        # released by the kernel so a recovery execution can acquire the lease.
        process.terminate()
        process.join(10)
        assert not process.is_alive()
        recovered_directory, recovered_lock = formal._acquire_execution_lease_v1(
            run_root
        )
        formal._release_execution_lease_v1(recovered_directory, recovered_lock)
        lock_path = run_root / formal._EXECUTION_LOCK_NAME
        assert lock_path.is_file() and not lock_path.is_symlink()
        assert lock_path.stat().st_mode & 0o777 == 0o600
        assert lock_path.stat().st_size == 0
    finally:
        if parent_lease_held:
            formal._release_execution_lease_v1(parent_directory, parent_lock)
        if process.is_alive():
            process.terminate()
            process.join(10)
        receiver.close()


def test_unique_pending_ignores_a_crash_orphan(
    linux_tmp_path: Path,
) -> None:
    parent = (linux_tmp_path / "orphan-parent").resolve()
    parent.mkdir(mode=0o700)
    path = parent / "terminal.json"
    document: dict[str, object] = {"value": 1}
    _self_hash(document)
    orphan = parent / (
        f".{path.name}.{document['outcome_artifact_sha256']}.pending"
    )
    orphan.write_bytes(b"crashed-before-link")
    orphan.chmod(0o600)

    assert (
        formal._write_outcome_exact_once(path, document)
        == "TERMINAL_PUBLISHED_NEW"
    )
    assert orphan.read_bytes() == b"crashed-before-link"
    assert path.read_bytes() == canonical_json_v1(document)
    assert [item.name for item in parent.iterdir() if item.name.endswith(".pending")] == [
        orphan.name
    ]


def test_reserved_raw_cap_budget_cannot_enter_formal_state_four(
    prepared_execution: formal.PreparedM3FormalExecutionV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepared_execution
    attempt_root = (
        start.canonical_run_root_v1(start.FORMAL_RUN_ID_HEX)
        / "attempts"
        / formal.CANONICAL_ATTEMPT_ID
    )
    if attempt_root.exists():
        shutil.rmtree(attempt_root)
    attempt_root.mkdir(mode=0o700, parents=True)
    preflight = _write_runner_evidence_fixture(attempt_root, prepared)
    output_root = attempt_root / "formal-enumeration"
    with pytest.raises(supervisor.M3DualEnumerationSupervisorError) as caught:
        _synthetic_dual_outcome(
            prepared,
            output_root,
            preflight,
            monkeypatch,
            budget=True,
        )
    assert caught.value.code == supervisor.FAIL_RUNNER
    assert "runtime requalification is required" in caught.value.detail
