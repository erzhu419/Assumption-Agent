"""Fail-closed supervisor for the formal M3 bounded dual enumeration.

The supervisor consumes a *previously qualified* public gate result and the
already-created index-zero start record.  It never loads a split seed or a
private key.  Two target-free enumerators are dispatched concurrently into
disjoint, exclusively-created output directories.  Their reports and complete
archives are replayed by the strict M2.5 qualification validators before any
formal receipt, agreement, or terminal state is constructed.

This module deliberately does not implement ``phase3-m3-start`` and does not
publish artifacts.  The caller owns both operations.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat
import time
from types import MappingProxyType
from typing import Callable, Final, Mapping, NoReturn

from . import phase3_m3_implementation_qualification_v1 as _qualification
from .phase3_m25_container_ceremony_v1 import (
    QualifiedGateEvidenceV1,
    promote_gate_evidence_v1,
)
from .phase3_m25_wire_v1 import (
    build_formal_object,
    candidate_content_root,
    validate_m3_state_chain_link,
    validate_timestamp_ordering_v1,
)


COMMIT_A: Final = "0af65964235390ce2bebefea7379eaa9c50eda24"
CANONICAL_PROGRAM_BUDGET: Final = 50_000
RAW_OPERATOR_APPLICATION_CAP: Final = 5_000_000
_ARCHIVE_STREAM_NAMES: Final = (
    "canonical_program_records",
    "program_chunk_manifests",
    "bucket_accounting_records",
)
_ARCHIVE_FILENAMES: Final = frozenset(
    {
        "report.json",
        "canonical_program_records.cborframed",
        "program_chunk_manifests.cborframed",
        "bucket_accounting_records.cborframed",
    }
)

FAIL_GATE = "FAIL_M3_SUPERVISOR_GATE_EVIDENCE"
FAIL_START = "FAIL_M3_SUPERVISOR_START_RECORD"
FAIL_BINDING = "FAIL_M3_SUPERVISOR_IMPLEMENTATION_BINDING"
FAIL_OUTPUT = "FAIL_M3_SUPERVISOR_OUTPUT_DIRECTORY"
FAIL_RUNNER = "FAIL_M3_SUPERVISOR_RUNNER"
FAIL_DUAL = "FAIL_DUAL_REPLAY_MISMATCH"
FAIL_TIMESTAMP = "FAIL_M3_SUPERVISOR_TIMESTAMP"
_UNSAFE_RUNNER_TERMINALIZATION_CODE: Final = (
    "FAIL_M3_FORMAL_RUNNER_UNSAFE_TERMINALIZATION"
)


class M3DualEnumerationSupervisorError(RuntimeError):
    """Stable fail-closed supervisor error."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise M3DualEnumerationSupervisorError(code, detail)


def _hex32(value: object, label: str) -> bytes:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _fail(FAIL_DUAL, f"{label} must be lowercase 32-byte hex")
    return bytes.fromhex(value)


@dataclass(frozen=True, slots=True)
class FrozenImplementationBindingV1:
    implementation: str
    implementation_id: int
    source_root: bytes
    binary_digest: bytes
    image_ref: str
    execution_environment_spec_root: bytes
    implementation_binding_root: bytes
    bound_executable_locator: str


FROZEN_IMPLEMENTATIONS: Final = MappingProxyType(
    {
        "python": FrozenImplementationBindingV1(
            implementation="python",
            implementation_id=1,
            source_root=bytes.fromhex(
                "e2bcc11cb663650205d66878da755452ac5732fb83254420672e26ab839af971"
            ),
            binary_digest=bytes.fromhex(
                "92d7e40ec50be176cb1b790c7568b7e08cd862137b5aa69f1413ba1967886b79"
            ),
            image_ref=(
                "python@sha256:"
                "e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3"
            ),
            execution_environment_spec_root=bytes.fromhex(
                "01eaf1df7fc96e570f00d80203254793942f5086710ecd9e857f054551a8b9c6"
            ),
            implementation_binding_root=bytes.fromhex(
                "0d71bd6e7830df4aab31aae9759bcb93a7b084750acd892eb3073c2a63a3322c"
            ),
            bound_executable_locator=(
                "oci://python@sha256:"
                "e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3/"
                "usr/local/bin/python3.11"
            ),
        ),
        "rust": FrozenImplementationBindingV1(
            implementation="rust",
            implementation_id=2,
            source_root=bytes.fromhex(
                "5bea2009e3ba45531af3b915f0bea2d460d650bb02ec7634d40bc6cbe3040e44"
            ),
            binary_digest=bytes.fromhex(
                "1339ebf38d1a2bf10d7604c10db9ec0c7b3ba98b016656e5b65f0ef35fdf8f8e"
            ),
            image_ref=(
                "rust@sha256:"
                "38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89"
            ),
            execution_environment_spec_root=bytes.fromhex(
                "9f8d1c2a900a0a331dee935d3c675a4a5190a8a450fdf36b4dc607a79a9dec9b"
            ),
            implementation_binding_root=bytes.fromhex(
                "02c2aae4099d8bd762caf00f0ab7794850c4987545810b8e583be50657fb0962"
            ),
            bound_executable_locator=(
                "generated-target://rust/m3_closure_enumerator/target/"
                f"m3_qualification/{COMMIT_A}/hegel-m3-closure-enumerator"
            ),
        ),
    }
)


@dataclass(frozen=True, slots=True)
class EnumerationInvocationV1:
    implementation: str
    implementation_id: int
    basis_commit: str
    source_root: bytes
    binary_digest: bytes
    image_ref: str
    implementation_binding_root: bytes
    bound_executable_locator: str
    child_dsl_spec_root: bytes
    operator_semantics_root: bytes
    identifier_registry_root: bytes
    canonical_program_budget: int
    raw_operator_application_cap: int
    pull_policy: str
    network_mode: str
    output_parent: Path


@dataclass(frozen=True, slots=True)
class EnumerationRunResultV1:
    invocation: EnumerationInvocationV1
    report: Mapping[str, object]
    started_at_unix_seconds: int
    finished_at_unix_seconds: int
    process_exit_code: int


EnumerationRunnerV1 = Callable[[EnumerationInvocationV1], EnumerationRunResultV1]
ClockV1 = Callable[[], int]


def _system_clock_v1() -> int:
    return int(time.time())


@dataclass(frozen=True, slots=True)
class M3DualEnumerationOutcomeV1:
    python_receipt_fields: Mapping[str, object]
    python_receipt_root: bytes
    rust_receipt_fields: Mapping[str, object]
    rust_receipt_root: bytes
    agreement_fields: Mapping[str, object]
    agreement_root: bytes
    terminal_state_fields: Mapping[str, object]
    terminal_state_root: bytes


def _validate_frozen_qualification_receipt(
    receipt: Mapping[str, object], golden: Mapping[str, object]
) -> None:
    try:
        _qualification.validate_qualification_receipt_v1(
            receipt, golden=golden, basis_commit=COMMIT_A
        )
    except Exception as exc:
        _fail(FAIL_BINDING, f"implementation qualification receipt failed replay: {exc}")
    if (
        receipt.get("basis_commit") != COMMIT_A
        or receipt.get("pull_policy_never") is not True
        or receipt.get("network_mode_none") is not True
        or receipt.get("m3_state") != "NOT_RUN"
    ):
        _fail(FAIL_BINDING, "qualification policy or Commit-A identity differs")
    for name, frozen in FROZEN_IMPLEMENTATIONS.items():
        row = receipt.get(name)
        if not isinstance(row, Mapping):
            _fail(FAIL_BINDING, f"{name} qualification binding is absent")
        observed = (
            row.get("implementation_id"),
            row.get("source_root"),
            row.get("binary_digest"),
            row.get("image_ref"),
            row.get("execution_environment_spec_root"),
            row.get("implementation_binding_root"),
            row.get("bound_executable_locator"),
        )
        expected = (
            frozen.implementation_id,
            frozen.source_root.hex(),
            frozen.binary_digest.hex(),
            frozen.image_ref,
            frozen.execution_environment_spec_root.hex(),
            frozen.implementation_binding_root.hex(),
            frozen.bound_executable_locator,
        )
        if observed != expected:
            _fail(FAIL_BINDING, f"{name} Commit-A source/image/binary binding differs")


def _runtime_golden(
    golden: Mapping[str, object], report: Mapping[str, object], roots: Mapping[str, bytes]
) -> Mapping[str, object]:
    expected = dict(golden["expected"])  # type: ignore[arg-type]
    # These two roots legitimately change because CanonicalProgramRecordV2
    # binds the formal (rather than 11/22/33 qualification) roots.  The strict
    # archive replay below, not the self-report, authenticates their values.
    expected["canonical_program_archive_root"] = report.get(
        "canonical_program_archive_root_or_null"
    )
    expected["program_chunk_manifest_root"] = report.get(
        "program_chunk_manifest_root_or_null"
    )
    value = dict(golden)
    value["binding_roots"] = {
        name: roots[name].hex()
        for name in (
            "child_dsl_spec_root",
            "operator_semantics_root",
            "identifier_registry_root",
        )
    }
    value["expected"] = expected
    return MappingProxyType(value)


def _validate_budget_report_v1(
    report: Mapping[str, object],
    *,
    implementation: str,
    roots: Mapping[str, bytes],
) -> Mapping[str, object]:
    """Validate the reserved raw-cap wire shape for diagnostic tests only.

    The qualified Python executable cannot persist this shape.  The live dual
    supervisor therefore rejects it before this validator and keeps state 4
    reserved until a future implementation is independently requalified.
    """

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
    }.get(implementation)
    if identity is None:
        raise ValueError("unknown budget-report implementation")
    expected: dict[str, object] = {
        "schema_version": identity[0],
        "claim_level": "FORMAL_PROFILE_CANDIDATE_NOT_AUTHORITY",
        "authoritative_claim_allowed": False,
        "implementation": implementation,
        "implementation_id": identity[1],
        "implementation_machine_id": identity[2],
        "dsl_version": "hegel-old-dsl-v1.1.0",
        "freeze_version": "hegel-freeze-p2b-p3-v1.1.2",
        "canonicalizer_profile": "hegel-canonical-ast-v1",
        "mdl_code_table_id": "hegel-mdl-prefix-v1.0.0",
        "closure_status": "INCONCLUSIVE_BUDGET",
        "closure_status_id": 3,
        "raw_operator_application_count": RAW_OPERATOR_APPLICATION_CAP,
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
        "records_per_chunk": 4096,
        "maximum_canonical_programs": CANONICAL_PROGRAM_BUDGET,
        "maximum_raw_operator_applications": RAW_OPERATOR_APPLICATION_CAP,
        "traversal_prefix_complete": False,
        "target_roles_evaluated": False,
        "split_material_accessed": False,
        "secrets_accessed": False,
        "aliases_excluded_before_count": [
            "greater_equal",
            "approx_equal:tolerance=0",
        ],
        "active_aggregate_map_ids": [0, 1, 5],
        "tombstoned_aggregate_map_ids": [2, 3, 4],
        "child_dsl_spec_root": roots["child_dsl_spec_root"].hex(),
        "operator_semantics_root": roots["operator_semantics_root"].hex(),
        "identifier_registry_root": roots["identifier_registry_root"].hex(),
    }
    if set(expected) != set(_qualification.REPORT_FIELDS):
        raise RuntimeError("internal budget-report field registry differs")
    if not isinstance(report, Mapping) or set(report) != set(expected):
        raise ValueError("budget report field set differs")
    for field, expected_value in expected.items():
        observed = report[field]
        if type(observed) is not type(expected_value) or observed != expected_value:
            raise ValueError(f"budget report differs at {field}")
    return MappingProxyType(dict(report))


def _validate_budget_archive_v1(
    output_parent: Path,
    *,
    implementation: str,
    stdout_report: Mapping[str, object],
) -> Mapping[str, object]:
    """Replay a finalized raw-cap archive whose three framed streams are empty."""

    directory = output_parent / "archive"
    try:
        metadata = directory.lstat()
        entries = tuple(directory.iterdir())
    except OSError as exc:
        raise ValueError(f"cannot inspect {implementation} budget archive: {exc}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or {entry.name for entry in entries} != _ARCHIVE_FILENAMES
    ):
        raise ValueError(f"{implementation} budget archive file set differs")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError(f"{implementation} budget archive contains a non-regular file")
    report_payload = _qualification._read_regular_file(
        directory / "report.json",
        maximum_bytes=1_048_576,
        label=f"{implementation} budget archive report",
    )
    disk_report = _qualification._parse_single_json(
        report_payload, label=f"{implementation} budget archive report"
    )
    if dict(disk_report) != dict(stdout_report):
        raise ValueError(f"{implementation} budget disk/stdout reports disagree")
    streams: dict[str, bytes] = {}
    for stream_name in _ARCHIVE_STREAM_NAMES:
        payload = _qualification._read_regular_file(
            directory / f"{stream_name}.cborframed",
            maximum_bytes=64 * 1024 * 1024,
            label=f"{implementation} budget {stream_name}",
        )
        if payload != b"":
            raise ValueError(f"{implementation} budget {stream_name} is not empty")
        streams[stream_name] = payload
    return MappingProxyType(
        {
            "streams": MappingProxyType(streams),
            "report_payload": report_payload,
            "witness_adjacency_verified": False,
        }
    )


def _validate_replayed_archive_shape_v1(
    archive: Mapping[str, object], *, implementation: str, budget: bool
) -> None:
    # The established DSL_TOO_LARGE host validator already checks the complete
    # archive and returns only after witness adjacency succeeds.  The local
    # budget validator is new, so assert its smaller result contract here.
    if not isinstance(archive, Mapping):
        raise ValueError(f"{implementation} replayed archive shape differs")
    if budget:
        streams = archive.get("streams")
        if (
            not isinstance(streams, Mapping)
            or set(streams) != set(_ARCHIVE_STREAM_NAMES)
            or any(type(streams[name]) is not bytes for name in _ARCHIVE_STREAM_NAMES)
            or archive.get("witness_adjacency_verified") is not False
        ):
            raise ValueError(f"{implementation} replayed budget archive shape differs")


def _receipt_fields(
    result: EnumerationRunResultV1,
    *,
    validated_report: Mapping[str, object],
    run_id: bytes,
    execution_manifest_root: bytes,
    roots: Mapping[str, bytes],
) -> Mapping[str, object]:
    report = validated_report
    binding = FROZEN_IMPLEMENTATIONS[result.invocation.implementation]
    fields: dict[str, object] = {
        "implementation_id": binding.implementation_id,
        "run_id": run_id,
        "execution_manifest_root": execution_manifest_root,
        "implementation_source_root": binding.source_root,
        "implementation_binary_digest": binding.binary_digest,
        # Frozen wire §12.5: this is the ExecutionEnvironmentSpecV1 object
        # root.  The OCI manifest digest is only one field of that object.
        "environment_image_digest": binding.execution_environment_spec_root,
        "child_dsl_spec_root": roots["child_dsl_spec_root"],
        "operator_semantics_root": roots["operator_semantics_root"],
        "identifier_registry_root": roots["identifier_registry_root"],
        "canonical_ast_schema_root": roots["canonical_ast_schema_root"],
        "canonical_cbor_profile_root": roots["canonical_cbor_profile_root"],
        "closure_status_id": report["closure_status_id"],
        "raw_operator_application_count": report["raw_operator_application_count"],
        "canonical_program_count": report["canonical_program_count"],
        "closure_cardinality_or_null": report["closure_cardinality_or_null"],
        "frontier_exhausted": report["frontier_exhausted"],
        "all_type_buckets_closed": report["all_type_buckets_closed"],
        "raw_expansion_limit_hit": report["raw_expansion_limit_hit"],
        "wall_clock_abort_hit": report["wall_clock_abort_hit"],
        "canonical_program_archive_root_or_null": (
            None
            if report["canonical_program_archive_root_or_null"] is None
            else _hex32(
                report["canonical_program_archive_root_or_null"],
                "program archive root",
            )
        ),
        "program_chunk_manifest_root_or_null": (
            None
            if report["program_chunk_manifest_root_or_null"] is None
            else _hex32(
                report["program_chunk_manifest_root_or_null"],
                "chunk manifest root",
            )
        ),
        "bucket_accounting_root_or_null": (
            None
            if report["bucket_accounting_root_or_null"] is None
            else _hex32(
                report["bucket_accounting_root_or_null"],
                "bucket accounting root",
            )
        ),
        "first_out_of_budget_program_hash_or_null": (
            None
            if report["first_out_of_budget_program_hash_or_null"] is None
            else _hex32(
                report["first_out_of_budget_program_hash_or_null"],
                "50,001 witness hash",
            )
        ),
        "partial_diagnostic_bundle_root_or_null": None,
        "started_at_unix_seconds": result.started_at_unix_seconds,
        "finished_at_unix_seconds": result.finished_at_unix_seconds,
        "process_exit_code": result.process_exit_code,
    }
    build_formal_object("M3ImplementationEnumerationReceiptV1", fields)
    return MappingProxyType(fields)


def _validate_result(
    result: EnumerationRunResultV1,
    invocation: EnumerationInvocationV1,
    *,
    golden: Mapping[str, object],
    roots: Mapping[str, bytes],
) -> Mapping[str, object]:
    if not isinstance(result, EnumerationRunResultV1) or result.invocation != invocation:
        _fail(FAIL_RUNNER, f"{invocation.implementation} runner invocation echo differs")
    if result.process_exit_code != 0:
        _fail(FAIL_RUNNER, f"{invocation.implementation} exited {result.process_exit_code}")
    try:
        validate_timestamp_ordering_v1(
            result.started_at_unix_seconds, result.finished_at_unix_seconds
        )
    except Exception as exc:
        _fail(FAIL_TIMESTAMP, f"{invocation.implementation} timestamps differ: {exc}")
    path = invocation.output_parent
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        _fail(FAIL_OUTPUT, f"{invocation.implementation} output is absent: {exc}")
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        _fail(FAIL_OUTPUT, f"{invocation.implementation} output is not a real directory")
    if result.report.get("closure_status") == "INCONCLUSIVE_BUDGET":
        _fail(
            FAIL_RUNNER,
            "reserved state-4 wire is not emitted by the qualified Python "
            "enumerator; runtime requalification is required",
        )
    try:
        runtime_golden = _runtime_golden(golden, result.report, roots)
        validated = _qualification.validate_enumerator_report_v1(
            dict(result.report),
            implementation=invocation.implementation,
            golden=runtime_golden,
        )
        # Detach all downstream formal objects from the mutable runner-owned
        # mapping, even if a replacement validator returns its input object.
        return MappingProxyType(dict(validated))
    except Exception as exc:
        _fail(FAIL_RUNNER, f"{invocation.implementation} report failed strict replay: {exc}")


def _run_m3_dual_enumeration_core_v1(
    *,
    qualified_gate_evidence: QualifiedGateEvidenceV1,
    execution_candidate_fields: Mapping[str, object],
    run_genesis_fields: Mapping[str, object],
    start_record_fields: Mapping[str, object],
    start_record_root: bytes,
    implementation_qualification_receipt: Mapping[str, object],
    committed_golden: Mapping[str, object],
    output_root: Path,
    runner: EnumerationRunnerV1,
    clock: ClockV1 = _system_clock_v1,
    resume_existing_output: bool = False,
) -> M3DualEnumerationOutcomeV1:
    """Non-authoritative core; only the sealed formal orchestrator may publish."""

    try:
        promoted = promote_gate_evidence_v1(qualified_gate_evidence)
    except Exception as exc:
        _fail(FAIL_GATE, f"qualified gate evidence is not sealed: {exc}")
    if (
        promoted.get("basis_commit") != COMMIT_A
        or promoted.get("m3_entry_qualified") is not True
        or promoted.get("child_state") != "NOT_RUN"
        or promoted.get("m3_run_started") is not False
    ):
        _fail(FAIL_GATE, "gate evidence is not Commit-A 24/24 NOT_RUN")
    roots = qualified_gate_evidence.formal_roots
    required_roots = {
        "child_dsl_spec_root",
        "operator_semantics_root",
        "identifier_registry_root",
        "canonical_ast_schema_root",
        "canonical_cbor_profile_root",
        "m3_execution_candidate_root",
        "m3_execution_manifest_root",
        "m3_run_genesis_root",
    }
    if not required_roots.issubset(roots) or any(
        type(roots[name]) is not bytes or len(roots[name]) != 32 for name in required_roots
    ):
        _fail(FAIL_GATE, "qualified formal root set is incomplete")
    _validate_frozen_qualification_receipt(
        implementation_qualification_receipt, committed_golden
    )

    try:
        build_formal_object("M3ExecutionCandidateV1", execution_candidate_fields)
        build_formal_object("M3RunGenesisV1", run_genesis_fields)
        build_formal_object("M3RunStateRecordV1", start_record_fields)
        computed_candidate_root = candidate_content_root(
            "M3ExecutionCandidateV1", execution_candidate_fields
        )
        computed_genesis_root = candidate_content_root(
            "M3RunGenesisV1", run_genesis_fields
        )
        computed_start_root = candidate_content_root(
            "M3RunStateRecordV1", start_record_fields
        )
    except Exception as exc:
        _fail(FAIL_START, f"start/genesis wire is invalid: {exc}")
    if computed_candidate_root != roots["m3_execution_candidate_root"]:
        _fail(FAIL_BINDING, "execution candidate root differs from qualified evidence")
    if computed_genesis_root != roots["m3_run_genesis_root"]:
        _fail(FAIL_START, "run genesis root differs from qualified evidence")
    if computed_start_root != start_record_root:
        _fail(FAIL_START, "start record root differs")
    run_id = run_genesis_fields.get("run_id")
    execution_manifest_root = roots["m3_execution_manifest_root"]
    candidate_root_bindings = {
        "python_implementation_binding_root": FROZEN_IMPLEMENTATIONS[
            "python"
        ].implementation_binding_root,
        "rust_implementation_binding_root": FROZEN_IMPLEMENTATIONS[
            "rust"
        ].implementation_binding_root,
        "canonical_program_budget": CANONICAL_PROGRAM_BUDGET,
        "raw_operator_application_cap": RAW_OPERATOR_APPLICATION_CAP,
        "child_dsl_spec_root": roots["child_dsl_spec_root"],
        "operator_semantics_root": roots["operator_semantics_root"],
        "identifier_registry_root": roots["identifier_registry_root"],
        "canonical_ast_schema_root": roots["canonical_ast_schema_root"],
        "canonical_cbor_profile_root": roots["canonical_cbor_profile_root"],
    }
    if (
        type(run_id) is not bytes
        or len(run_id) != 16
        or execution_candidate_fields.get("run_id") != run_id
        or any(
            execution_candidate_fields.get(name) != expected
            for name, expected in candidate_root_bindings.items()
        )
        or run_genesis_fields.get("execution_manifest_root") != execution_manifest_root
        or start_record_fields.get("run_id") != run_id
        or start_record_fields.get("execution_manifest_root") != execution_manifest_root
    ):
        _fail(FAIL_START, "run/start/execution identity differs")
    try:
        validate_timestamp_ordering_v1(
            run_genesis_fields["created_at_unix_seconds"],
            start_record_fields["recorded_at_unix_seconds"],
        )
    except Exception as exc:
        _fail(FAIL_TIMESTAMP, f"genesis/start ordering differs: {exc}")

    if not output_root.is_absolute():
        _fail(FAIL_OUTPUT, "output root must be absolute")
    try:
        parent_mode = output_root.parent.lstat().st_mode
        if stat.S_ISLNK(parent_mode) or not stat.S_ISDIR(parent_mode):
            _fail(FAIL_OUTPUT, "output parent must be a real directory")
        output_root.mkdir(
            mode=0o700,
            parents=False,
            exist_ok=resume_existing_output,
        )
        output_metadata = output_root.lstat()
        if (
            stat.S_ISLNK(output_metadata.st_mode)
            or not stat.S_ISDIR(output_metadata.st_mode)
            or output_root.resolve(strict=True) != output_root
            or output_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(output_metadata.st_mode) != 0o700
        ):
            _fail(FAIL_OUTPUT, "output root identity differs")
    except OSError as exc:
        _fail(FAIL_OUTPUT, f"cannot create or resume output root: {exc}")
    invocations = {
        name: EnumerationInvocationV1(
            implementation=name,
            implementation_id=binding.implementation_id,
            basis_commit=COMMIT_A,
            source_root=binding.source_root,
            binary_digest=binding.binary_digest,
            image_ref=binding.image_ref,
            implementation_binding_root=binding.implementation_binding_root,
            bound_executable_locator=binding.bound_executable_locator,
            child_dsl_spec_root=roots["child_dsl_spec_root"],
            operator_semantics_root=roots["operator_semantics_root"],
            identifier_registry_root=roots["identifier_registry_root"],
            canonical_program_budget=CANONICAL_PROGRAM_BUDGET,
            raw_operator_application_cap=RAW_OPERATOR_APPLICATION_CAP,
            pull_policy="never",
            network_mode="none",
            output_parent=output_root / name,
        )
        for name, binding in FROZEN_IMPLEMENTATIONS.items()
    }
    results: dict[str, EnumerationRunResultV1] = {}
    failures: dict[str, Exception] = {}
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hegel-m3") as pool:
        futures = {
            name: pool.submit(runner, call) for name, call in invocations.items()
        }
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except Exception as exc:
                failures[name] = exc
    if failures:
        unsafe = [
            error
            for error in failures.values()
            if getattr(error, "code", None)
            == _UNSAFE_RUNNER_TERMINALIZATION_CODE
        ]
        if unsafe:
            # Never hide an uncontained named container behind an ordinary
            # peer failure.  The formal layer must preserve RUNNING and must
            # not publish a terminal record until containment is proven.
            raise unsafe[0]
        detail = "; ".join(
            f"{name}={type(error).__name__}"
            for name, error in sorted(failures.items())
        )
        _fail(FAIL_RUNNER, f"dual enumeration runner failed: {detail}")

    reports = {
        name: _validate_result(
            results[name], invocations[name], golden=committed_golden, roots=roots
        )
        for name in ("python", "rust")
    }
    binding_roots = (
        roots["child_dsl_spec_root"],
        roots["operator_semantics_root"],
        roots["identifier_registry_root"],
    )
    archives: dict[str, Mapping[str, object]] = {}
    for name in ("python", "rust"):
        try:
            archive = _qualification._host_validate_enumerator_archive_v1(
                invocations[name].output_parent,
                implementation=name,
                stdout_report=reports[name],
                roots=binding_roots,
            )
            _validate_replayed_archive_shape_v1(
                archive, implementation=name, budget=False
            )
            archives[name] = archive
        except Exception as exc:
            _fail(FAIL_RUNNER, f"{name} archive failed strict replay: {exc}")

    compared_fields = (
        "closure_status",
        "closure_status_id",
        "raw_operator_application_count",
        "canonical_program_count",
        "closure_cardinality_or_null",
        "frontier_exhausted",
        "all_type_buckets_closed",
        "raw_expansion_limit_hit",
        "wall_clock_abort_hit",
        "canonical_program_archive_root_or_null",
        "program_chunk_manifest_root_or_null",
        "bucket_accounting_root_or_null",
        "first_out_of_budget_program_hash_or_null",
    )
    python_signature = tuple(reports["python"][field] for field in compared_fields)
    rust_signature = tuple(reports["rust"][field] for field in compared_fields)
    dsl_too_large_signature = (
        "DSL_TOO_LARGE",
        2,
        3_292_439,
        50_000,
        None,
        False,
        False,
        False,
        False,
        reports["python"]["canonical_program_archive_root_or_null"],
        reports["python"]["program_chunk_manifest_root_or_null"],
        reports["python"]["bucket_accounting_root_or_null"],
        reports["python"]["first_out_of_budget_program_hash_or_null"],
    )
    if python_signature != rust_signature:
        _fail(FAIL_DUAL, "enumerator report fields differ after independent replay")
    if python_signature == dsl_too_large_signature:
        try:
            _qualification._validate_dual_archive_bytes_equal_v1(
                archives["python"], archives["rust"]
            )
        except Exception as exc:
            _fail(FAIL_DUAL, f"strict dual archive bytes differ: {exc}")
        closure_status_id = 2
        canonical_program_count = CANONICAL_PROGRAM_BUDGET
        terminal_state_id = 3
        terminal_reason_id = 3
    else:
        _fail(FAIL_DUAL, "enumerators lack an exact supported dual agreement")

    agreement_created_at_unix_seconds = clock()
    start_time = start_record_fields["recorded_at_unix_seconds"]
    for result in results.values():
        try:
            validate_timestamp_ordering_v1(start_time, result.started_at_unix_seconds)
            validate_timestamp_ordering_v1(
                result.finished_at_unix_seconds, agreement_created_at_unix_seconds
            )
        except Exception as exc:
            _fail(FAIL_TIMESTAMP, f"enumeration/start/agreement ordering differs: {exc}")
    terminal_recorded_at_unix_seconds = clock()
    try:
        validate_timestamp_ordering_v1(
            agreement_created_at_unix_seconds, terminal_recorded_at_unix_seconds
        )
    except Exception as exc:
        _fail(FAIL_TIMESTAMP, f"agreement/terminal ordering differs: {exc}")

    receipt_fields = {
        name: _receipt_fields(
            results[name],
            validated_report=reports[name],
            run_id=run_id,
            execution_manifest_root=execution_manifest_root,
            roots=roots,
        )
        for name in ("python", "rust")
    }
    receipt_roots = {
        name: candidate_content_root(
            "M3ImplementationEnumerationReceiptV1", receipt_fields[name]
        )
        for name in ("python", "rust")
    }
    reference = reports["python"]
    agreement_fields: dict[str, object] = {
        "run_id": run_id,
        "execution_manifest_root": execution_manifest_root,
        "python_enumeration_receipt_root": receipt_roots["python"],
        "rust_enumeration_receipt_root": receipt_roots["rust"],
        "agreed_closure_status_id": closure_status_id,
        "canonical_program_count_or_null": canonical_program_count,
        "closure_cardinality_or_null": None,
        "canonical_program_archive_root_or_null": (
            None
            if reference["canonical_program_archive_root_or_null"] is None
            else _hex32(
                reference["canonical_program_archive_root_or_null"],
                "program archive root",
            )
        ),
        "program_chunk_manifest_root_or_null": (
            None
            if reference["program_chunk_manifest_root_or_null"] is None
            else _hex32(
                reference["program_chunk_manifest_root_or_null"],
                "chunk manifest root",
            )
        ),
        "bucket_accounting_root_or_null": (
            None
            if reference["bucket_accounting_root_or_null"] is None
            else _hex32(
                reference["bucket_accounting_root_or_null"],
                "bucket accounting root",
            )
        ),
        "first_out_of_budget_program_hash_or_null": (
            None
            if reference["first_out_of_budget_program_hash_or_null"] is None
            else _hex32(
                reference["first_out_of_budget_program_hash_or_null"],
                "50,001 witness hash",
            )
        ),
        "role_agreement_entries": (),
        "enumeration_agreement": True,
        "role_agreement_status_id": 0,
        "mismatch_record_root_or_null": None,
        "created_at_unix_seconds": agreement_created_at_unix_seconds,
    }
    build_formal_object("M3DualReplayAgreementV1", agreement_fields)
    agreement_root = candidate_content_root(
        "M3DualReplayAgreementV1", agreement_fields
    )
    terminal_fields: dict[str, object] = {
        "run_id": run_id,
        "transition_index": 1,
        "previous_state_record_root_or_null": start_record_root,
        "from_state_id": 1,
        "from_phase_id": 1,
        "to_state_id": terminal_state_id,
        "to_phase_id": 0,
        "transition_reason_id": terminal_reason_id,
        "execution_manifest_root": execution_manifest_root,
        # The dual agreement is the strongest causal receipt for this terminal
        # transition; neither single-implementation receipt can trigger it.
        "triggering_receipt_root_or_null": agreement_root,
        "recorded_at_unix_seconds": terminal_recorded_at_unix_seconds,
    }
    validate_m3_state_chain_link(
        terminal_fields["previous_state_record_root_or_null"], start_record_root
    )
    build_formal_object("M3RunStateRecordV1", terminal_fields)
    terminal_root = candidate_content_root("M3RunStateRecordV1", terminal_fields)
    return M3DualEnumerationOutcomeV1(
        python_receipt_fields=receipt_fields["python"],
        python_receipt_root=receipt_roots["python"],
        rust_receipt_fields=receipt_fields["rust"],
        rust_receipt_root=receipt_roots["rust"],
        agreement_fields=MappingProxyType(agreement_fields),
        agreement_root=agreement_root,
        terminal_state_fields=MappingProxyType(terminal_fields),
        terminal_state_root=terminal_root,
    )


__all__ = [
    "EnumerationInvocationV1",
    "EnumerationRunResultV1",
    "FROZEN_IMPLEMENTATIONS",
    "M3DualEnumerationOutcomeV1",
    "M3DualEnumerationSupervisorError",
]
