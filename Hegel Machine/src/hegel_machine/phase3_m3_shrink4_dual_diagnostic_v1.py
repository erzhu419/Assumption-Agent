"""Host replay for the shrink-step-4 non-formal dual diagnostic.

This module trusts neither enumerator's JSON roots nor either implementation's
record decoder.  It reads the two public output directories, requires the
three framed archives to be byte-identical, and replays every strict CBOR
record with the Python host decoder. The returned receipt is a
``NON_FORMAL_DUAL_CHILD_DIAGNOSTIC`` with qualification level
``DIAGNOSTIC_ONLY``: it cannot create formal roots, start M3, evaluate a role,
or authorize a state transition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Final, NoReturn

from . import phase3_m3_bounded_enumerator_shrink4_v1 as _child_enumerator
from .phase3_m3_bounded_enumerator_v1 import (
    CHUNK_BLOB_DOMAIN,
    OUTPUT_SORT_IDS,
    program_mdl_length_q32,
)
from .phase3_m3_shrink4_diagnostic_profile_v1 import (
    BINDING_PROFILE_ID,
    CANONICAL_AST_SCHEMA_ROOT,
    CANONICAL_CBOR_PROFILE_ROOT,
    CLAIM_LEVEL,
    DUAL_COMPLETE_ENUMERATION_STATUS,
    NON_FORMAL_SYNTHETIC_CHILD_BINDINGS,
    PROFILE_ID,
    STRICT_QUALIFICATION_ARTIFACT_PATH,
    STRICT_QUALIFICATION_ARTIFACT_SHA256,
    STRICT_QUALIFICATION_DIAGNOSTIC_REPORT_HASH,
    STRICT_QUALIFICATION_EVIDENCE_COMMIT,
    STRICT_QUALIFICATION_SOURCE_COMMIT,
    STRICT_QUALIFICATION_STATUS,
)
from .strict_ast_shrink4_v1 import decode_shrink4_canonical_ast
from .strict_cbor_v1 import canonical_cbor_decode


RECEIPT_SCHEMA: Final = "hegel-m3-shrink4-dual-diagnostic-validation-receipt/1"
CANONICAL_PROGRAM_BUDGET: Final = 50_000
RAW_OPERATOR_APPLICATION_CAP: Final = 5_000_000
RECORDS_PER_CHUNK: Final = 4096
MAXIMUM_AST_DEPTH: Final = 4
MAXIMUM_AST_NODE_COUNT: Final = 7
MAXIMUM_TOP_LEVEL_CLAUSES: Final = 2
FORMAL_BUCKET_COUNT: Final = 175

PROGRAM_RECORD_TAG: Final = 0x3207
PROGRAM_RECORD_SCHEMA: Final = b"hegel-canonical-program-record/2"
CHUNK_MANIFEST_TAG: Final = 0x3209
CHUNK_MANIFEST_SCHEMA: Final = b"hegel-program-chunk-manifest/2"
BUCKET_RECORD_TAG: Final = 0x320C
BUCKET_RECORD_SCHEMA: Final = b"hegel-bucket-accounting-record/1"

_OUTPUT_FILES: Final = frozenset(
    {
        "report.json",
        "canonical_program_records.cborframed",
        "program_chunk_manifests.cborframed",
        "bucket_accounting_records.cborframed",
    }
)
_STREAM_NAMES: Final = (
    "canonical_program_records",
    "program_chunk_manifests",
    "bucket_accounting_records",
)
_EXPECTED_SCHEMAS: Final = {
    "python": "hegel-m3-shrink4-python-closure-enumerator-report/1",
    "rust": "hegel-m3-shrink4-rust-closure-enumerator-report/1",
}
_EXPECTED_IMPLEMENTATIONS: Final = {
    "python": (1, "hegel-python-m3-shrink4-complete-closure-diagnostic-v1"),
    "rust": (2, "hegel-rust-m3-shrink4-complete-closure-diagnostic-v1"),
}
_EXPECTED_ROOTS: Final = (
    *NON_FORMAL_SYNTHETIC_CHILD_BINDINGS,
    CANONICAL_AST_SCHEMA_ROOT,
    CANONICAL_CBOR_PROFILE_ROOT,
)
_ROOT_NAMES: Final = (
    "child_dsl_spec_root",
    "operator_semantics_root",
    "identifier_registry_root",
    "canonical_ast_schema_root",
    "canonical_cbor_profile_root",
)
_COMMON_REPORT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "profile_id",
        "claim_level",
        "binding_profile_id",
        "diagnostic_only",
        "authoritative_claim_allowed",
        "execution_state",
        "formal_roots_generated",
        "formal_roots",
        "implementation",
        "implementation_id",
        "implementation_machine_id",
        "strict_qualification_source_commit",
        "strict_qualification_evidence_commit",
        "strict_qualification_artifact_path",
        "strict_qualification_artifact_sha256",
        "strict_qualification_diagnostic_report_hash",
        "strict_qualification_status",
        "dsl_version",
        "freeze_version",
        "parent_dsl_version",
        "parent_freeze_version",
        "human_amendment_id",
        "shrink_step_id",
        "canonicalizer_profile",
        "mdl_code_table_id",
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
        "first_out_of_budget_program_cbor_hex_or_null",
        "first_out_of_budget_program_ordinal_or_null",
        "program_record_count",
        "chunk_manifest_count",
        "bucket_record_count",
        "records_per_chunk",
        "maximum_canonical_programs",
        "maximum_raw_operator_applications",
        "maximum_ast_depth",
        "maximum_ast_node_count",
        "maximum_top_level_clauses",
        "and3_generator_attempts_allowed",
        "and3_raw_operator_application_count",
        "formal_bucket_count",
        "traversal_prefix_complete",
        "target_roles_evaluated",
        "split_material_accessed",
        "secrets_accessed",
        "aliases_excluded_before_count",
        "active_aggregate_map_ids",
        "tombstoned_aggregate_map_ids",
        "active_rational_parameter_ids",
        "tombstoned_rational_parameter_ids",
        "reserved_rational_parameter_ids",
        "active_source_binary_operator_ids",
        "active_formal_canonical_binary_operator_ids",
        "source_alias_binary_operator_ids",
        "tombstoned_binary_operator_ids",
        "reserved_binary_operator_ids",
        "operator_id_compaction_performed",
        "automatic_operator_migration_performed",
        "child_dsl_spec_root",
        "operator_semantics_root",
        "identifier_registry_root",
        "canonical_ast_schema_root",
        "canonical_cbor_profile_root",
    }
)
_PYTHON_ISOLATION_REPORT_FIELDS: Final = frozenset(
    {
        "loaded_hegel_modules",
        "target_free_isolation_verified",
        "target_or_split_modules_loaded",
    }
)
_EXPECTED_REPORT_FIELDS: Final = {
    "python": _COMMON_REPORT_FIELDS | _PYTHON_ISOLATION_REPORT_FIELDS,
    "rust": _COMMON_REPORT_FIELDS,
}
_EXPECTED_PYTHON_MODULES: Final = [
    "hegel_machine.phase3_m3_bounded_enumerator_shrink2_v1",
    "hegel_machine.phase3_m3_bounded_enumerator_shrink3_v1",
    "hegel_machine.phase3_m3_bounded_enumerator_shrink4_v1",
    "hegel_machine.phase3_m3_bounded_enumerator_v1",
    "hegel_machine.phase3_m3_dsl_core_v1",
    "hegel_machine.phase3_m3_record_wire_v1",
    "hegel_machine.phase3_m3_shrink1_core_v1",
    "hegel_machine.phase3_m3_shrink2_core_v1",
    "hegel_machine.phase3_m3_shrink3_core_v1",
    "hegel_machine.phase3_m3_shrink4_core_v1",
    "hegel_machine.phase3_m3_shrink4_diagnostic_profile_v1",
    "hegel_machine.strict_ast_shrink1_v1",
    "hegel_machine.strict_ast_shrink2_v1",
    "hegel_machine.strict_ast_shrink3_v1",
    "hegel_machine.strict_ast_shrink4_v1",
    "hegel_machine.strict_ast_v1",
    "hegel_machine.strict_cbor_v1",
]
_EXPECTED_HOST_PROJECT_MODULES: Final = frozenset(
    {
        *_EXPECTED_PYTHON_MODULES,
        "hegel_machine.phase3_m3_shrink4_dual_diagnostic_v1",
    }
)
_FORBIDDEN_HOST_MODULE_FRAGMENTS: Final = (
    "_evaluator",
    "_odd",
    "_role",
    "_seed",
    "_sink",
    "_split_",
    "_target",
)

FAIL_OUTPUT = "FAIL_SHRINK4_DUAL_OUTPUT"
FAIL_REPORT = "FAIL_SHRINK4_DUAL_REPORT"
FAIL_DUAL = "FAIL_SHRINK4_DUAL_MISMATCH"
FAIL_PROGRAM = "FAIL_SHRINK4_DUAL_PROGRAM_REPLAY"
FAIL_CHUNK = "FAIL_SHRINK4_DUAL_CHUNK_REPLAY"
FAIL_BUCKET = "FAIL_SHRINK4_DUAL_BUCKET_REPLAY"
FAIL_WITNESS = "FAIL_SHRINK4_DUAL_WITNESS_ADJACENCY"
FAIL_AUTHORITY = "FAIL_SHRINK4_DUAL_AUTHORITY_ESCALATION"
FAIL_HOST_ISOLATION = "FAIL_SHRINK4_DUAL_HOST_ISOLATION"


class Shrink4DualDiagnosticError(RuntimeError):
    """Stable fail-closed host validation error."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _fail(code: str, detail: str) -> NoReturn:
    raise Shrink4DualDiagnosticError(code, detail)


def _assert_host_module_closure() -> tuple[str, ...]:
    loaded = frozenset(
        name for name in sys.modules if name.startswith("hegel_machine.")
    )
    forbidden = sorted(
        name
        for name in loaded
        if any(
            fragment in name for fragment in _FORBIDDEN_HOST_MODULE_FRAGMENTS
        )
    )
    if forbidden:
        _fail(
            FAIL_HOST_ISOLATION,
            f"target/split/seed/role dependency loaded: {forbidden!r}",
        )
    if loaded != _EXPECTED_HOST_PROJECT_MODULES:
        _fail(
            FAIL_HOST_ISOLATION,
            "host dependency closure drift; "
            f"missing={sorted(_EXPECTED_HOST_PROJECT_MODULES - loaded)!r}; "
            f"unexpected={sorted(loaded - _EXPECTED_HOST_PROJECT_MODULES)!r}",
        )
    return tuple(sorted(loaded))


def _strict_json(payload: bytes, *, label: str) -> Mapping[str, object]:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail(FAIL_REPORT, f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> NoReturn:
        _fail(FAIL_REPORT, f"{label} contains forbidden JSON constant {value!r}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(FAIL_REPORT, f"{label} is not strict UTF-8 JSON: {error}")
    if not isinstance(value, dict):
        _fail(FAIL_REPORT, f"{label} must be a JSON object")
    return value


def _read_regular_file(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    """Read one immutable regular file without following a final symlink."""

    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(FAIL_OUTPUT, f"{label} is not a regular file")
        if before.st_size < 0 or before.st_size > maximum_bytes:
            _fail(FAIL_OUTPUT, f"{label} exceeds its byte limit")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1_048_576))
            if not chunk:
                _fail(FAIL_OUTPUT, f"{label} ended before its recorded size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(FAIL_OUTPUT, f"{label} grew while being read")
        after = os.fstat(descriptor)
        namespace = path.lstat()
    except Shrink4DualDiagnosticError:
        raise
    except OSError as error:
        _fail(FAIL_OUTPUT, f"cannot read {label}: {error}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if identity(before) != identity(after) or (
        namespace.st_dev,
        namespace.st_ino,
    ) != (after.st_dev, after.st_ino):
        _fail(FAIL_OUTPUT, f"{label} changed while being read")
    return b"".join(chunks)


def _load_output_directory(
    directory: Path | str, *, implementation: str
) -> tuple[Mapping[str, object], bytes, dict[str, bytes]]:
    path = Path(directory)
    try:
        metadata = path.lstat()
        entries = tuple(path.iterdir())
    except OSError as error:
        _fail(FAIL_OUTPUT, f"cannot inspect {implementation} output directory: {error}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        _fail(FAIL_OUTPUT, f"{implementation} output path is not a real directory")
    if {entry.name for entry in entries} != _OUTPUT_FILES:
        _fail(FAIL_OUTPUT, f"{implementation} output file set differs")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        _fail(FAIL_OUTPUT, f"{implementation} output contains a non-regular file")
    report_payload = _read_regular_file(
        path / "report.json",
        maximum_bytes=1_048_576,
        label=f"{implementation} report.json",
    )
    streams = {
        name: _read_regular_file(
            path / f"{name}.cborframed",
            maximum_bytes=256 * 1024 * 1024,
            label=f"{implementation} {name}.cborframed",
        )
        for name in _STREAM_NAMES
    }
    return _strict_json(report_payload, label=f"{implementation} report"), report_payload, streams


def _field(
    report: Mapping[str, object],
    name: str,
) -> object:
    if name not in report:
        _fail(FAIL_REPORT, f"report lacks required field {name}")
    return report[name]


def _expect(value: object, expected: object, *, label: str) -> None:
    def same_typed(left: object, right: object) -> bool:
        if type(left) is not type(right):
            return False
        if isinstance(left, (list, tuple)):
            assert isinstance(right, (list, tuple))
            return len(left) == len(right) and all(
                same_typed(a, b) for a, b in zip(left, right, strict=True)
            )
        return left == right

    if not same_typed(value, expected):
        _fail(FAIL_REPORT, f"{label} differs: expected {expected!r}, got {value!r}")


def _uint(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail(FAIL_REPORT, f"{label} must be an unsigned integer")
    return value


def _hex_bytes(value: object, *, label: str, length: int | None = None) -> bytes:
    if (
        type(value) is not str
        or len(value) % 2
        or re.fullmatch(r"[0-9a-f]*", value) is None
        or (length is not None and len(value) != length * 2)
    ):
        suffix = "" if length is None else f" ({length} bytes)"
        _fail(FAIL_REPORT, f"{label} must be lowercase even-length hex{suffix}")
    return bytes.fromhex(value)


def _normalise_report(
    report: Mapping[str, object], *, implementation: str
) -> dict[str, object]:
    if STRICT_QUALIFICATION_STATUS != "SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS":
        _fail(FAIL_AUTHORITY, "Evidence P strict qualification is not PASS")
    expected_fields = _EXPECTED_REPORT_FIELDS[implementation]
    observed_fields = frozenset(report)
    if observed_fields != expected_fields:
        _fail(
            FAIL_REPORT,
            f"{implementation} report field set differs; "
            f"missing={sorted(expected_fields - observed_fields)!r}; "
            f"unknown={sorted(observed_fields - expected_fields)!r}",
        )
    expected_id, expected_machine = _EXPECTED_IMPLEMENTATIONS[implementation]
    _expect(
        _field(report, "schema_version"),
        _EXPECTED_SCHEMAS[implementation],
        label=f"{implementation}.schema_version",
    )
    _expect(_field(report, "implementation"), implementation, label="implementation")
    _expect(_field(report, "implementation_id"), expected_id, label="implementation_id")
    _expect(
        _field(report, "implementation_machine_id"),
        expected_machine,
        label="implementation_machine_id",
    )
    _expect(_field(report, "profile_id"), PROFILE_ID, label="profile_id")
    _expect(
        _field(report, "binding_profile_id"),
        BINDING_PROFILE_ID,
        label="binding_profile_id",
    )
    _expect(_field(report, "claim_level"), CLAIM_LEVEL, label="claim_level")
    _expect(_field(report, "diagnostic_only"), True, label="diagnostic_only")
    _expect(
        _field(report, "authoritative_claim_allowed"),
        False,
        label="authoritative_claim_allowed",
    )
    _expect(
        _field(report, "execution_state"),
        DUAL_COMPLETE_ENUMERATION_STATUS,
        label="execution_state",
    )
    _expect(
        _field(report, "formal_roots_generated"),
        False,
        label="formal_roots_generated",
    )
    if _field(report, "formal_roots") is not None:
        _fail(FAIL_AUTHORITY, f"{implementation} report generated formal roots")

    if implementation == "python":
        _expect(
            _field(report, "loaded_hegel_modules"),
            _EXPECTED_PYTHON_MODULES,
            label="python.loaded_hegel_modules",
        )
        _expect(
            _field(report, "target_free_isolation_verified"),
            True,
            label="python.target_free_isolation_verified",
        )
        _expect(
            _field(report, "target_or_split_modules_loaded"),
            False,
            label="python.target_or_split_modules_loaded",
        )

    static_values: tuple[tuple[str, object], ...] = (
        ("strict_qualification_source_commit", STRICT_QUALIFICATION_SOURCE_COMMIT),
        (
            "strict_qualification_evidence_commit",
            STRICT_QUALIFICATION_EVIDENCE_COMMIT,
        ),
        ("strict_qualification_artifact_path", STRICT_QUALIFICATION_ARTIFACT_PATH),
        (
            "strict_qualification_artifact_sha256",
            STRICT_QUALIFICATION_ARTIFACT_SHA256,
        ),
        (
            "strict_qualification_diagnostic_report_hash",
            STRICT_QUALIFICATION_DIAGNOSTIC_REPORT_HASH,
        ),
        ("strict_qualification_status", STRICT_QUALIFICATION_STATUS),
        ("parent_dsl_version", "hegel-old-dsl-v1.3.0"),
        ("parent_freeze_version", "hegel-freeze-p2b-p3-v1.3.0"),
        ("dsl_version", "hegel-old-dsl-v1.4.0"),
        ("freeze_version", "hegel-freeze-p2b-p3-v1.4.0"),
        ("human_amendment_id", "hegel-freeze-p2b-p3-v1.4.0-shrink-step4"),
        (
            "shrink_step_id",
            "SHRINK_STEP_4_REDUCE_MAX_TOP_LEVEL_CLAUSES_3_TO_2",
        ),
        ("canonicalizer_profile", "hegel-canonical-ast-v1"),
        ("mdl_code_table_id", "hegel-mdl-prefix-v1.0.0"),
        ("raw_expansion_limit_hit", False),
        ("wall_clock_abort_hit", False),
        ("bucket_record_count", FORMAL_BUCKET_COUNT),
        ("records_per_chunk", RECORDS_PER_CHUNK),
        ("maximum_canonical_programs", CANONICAL_PROGRAM_BUDGET),
        ("maximum_raw_operator_applications", RAW_OPERATOR_APPLICATION_CAP),
        ("maximum_ast_depth", MAXIMUM_AST_DEPTH),
        ("maximum_ast_node_count", MAXIMUM_AST_NODE_COUNT),
        ("maximum_top_level_clauses", MAXIMUM_TOP_LEVEL_CLAUSES),
        ("and3_generator_attempts_allowed", False),
        ("and3_raw_operator_application_count", 0),
        ("formal_bucket_count", FORMAL_BUCKET_COUNT),
        ("traversal_prefix_complete", True),
        ("target_roles_evaluated", False),
        ("split_material_accessed", False),
        ("secrets_accessed", False),
        (
            "aliases_excluded_before_count",
            ["greater_equal", "approx_equal:tolerance=0"],
        ),
        ("active_aggregate_map_ids", [0, 1, 5]),
        ("tombstoned_aggregate_map_ids", [2, 3, 4]),
        ("active_rational_parameter_ids", [1, 3, 5]),
        ("tombstoned_rational_parameter_ids", [0, 2, 4, 6]),
        ("reserved_rational_parameter_ids", [7]),
        ("active_source_binary_operator_ids", [1, 2, 3, 4, 5, 6]),
        ("active_formal_canonical_binary_operator_ids", [1, 2, 3, 5, 6]),
        ("source_alias_binary_operator_ids", [4]),
        ("tombstoned_binary_operator_ids", [0]),
        ("reserved_binary_operator_ids", [7]),
        ("operator_id_compaction_performed", False),
        ("automatic_operator_migration_performed", False),
    )
    normalised: dict[str, object] = {}
    for name, expected in static_values:
        observed = _field(report, name)
        _expect(observed, expected, label=f"{implementation}.{name}")
        normalised[name] = observed

    status = _field(report, "closure_status")
    if type(status) is not str or status not in {"DSL_TOO_LARGE", "COMPLETE"}:
        _fail(FAIL_REPORT, f"{implementation}.closure_status is not terminal")
    count = _uint(
        _field(report, "canonical_program_count"),
        label=f"{implementation}.canonical_program_count",
    )
    if count < 1 or count > CANONICAL_PROGRAM_BUDGET:
        _fail(FAIL_REPORT, f"{implementation} canonical count is outside the budget")
    if status == "DSL_TOO_LARGE":
        dynamic_values: tuple[tuple[str, object], ...] = (
            ("closure_status_id", 2),
            ("canonical_program_count", CANONICAL_PROGRAM_BUDGET),
            ("closure_cardinality_or_null", None),
            ("frontier_exhausted", False),
            ("all_type_buckets_closed", False),
            ("program_record_count", CANONICAL_PROGRAM_BUDGET),
            (
                "chunk_manifest_count",
                (CANONICAL_PROGRAM_BUDGET + RECORDS_PER_CHUNK - 1)
                // RECORDS_PER_CHUNK,
            ),
        )
    else:
        dynamic_values = (
            ("closure_status_id", 1),
            ("canonical_program_count", count),
            ("closure_cardinality_or_null", count),
            ("frontier_exhausted", True),
            ("all_type_buckets_closed", True),
            ("program_record_count", count),
            (
                "chunk_manifest_count",
                (count + RECORDS_PER_CHUNK - 1) // RECORDS_PER_CHUNK,
            ),
        )
    normalised["closure_status"] = status
    for name, expected in dynamic_values:
        observed = _field(report, name)
        _expect(observed, expected, label=f"{implementation}.{name}")
        normalised[name] = observed

    raw_count = _uint(
        _field(report, "raw_operator_application_count"),
        label=f"{implementation}.raw_operator_application_count",
    )
    minimum_raw = count + (1 if status == "DSL_TOO_LARGE" else 0)
    if raw_count < minimum_raw or raw_count > RAW_OPERATOR_APPLICATION_CAP:
        _fail(FAIL_REPORT, f"{implementation} raw count is outside the frozen budget")
    normalised["raw_operator_application_count"] = raw_count

    for name, expected in zip(_ROOT_NAMES, _EXPECTED_ROOTS, strict=True):
        observed = _hex_bytes(
            _field(report, name),
            label=f"{implementation}.{name}",
            length=32,
        )
        if observed != expected:
            _fail(FAIL_REPORT, f"{implementation}.{name} differs from diagnostic profile")
        normalised[name] = observed.hex()

    for name in (
        "canonical_program_archive_root_or_null",
        "program_chunk_manifest_root_or_null",
        "bucket_accounting_root_or_null",
    ):
        normalised[name] = _hex_bytes(
            _field(report, name),
            label=f"{implementation}.{name}",
            length=32,
        ).hex()
    witness_fields = (
        "first_out_of_budget_program_hash_or_null",
        "first_out_of_budget_program_cbor_hex_or_null",
        "first_out_of_budget_program_ordinal_or_null",
    )
    if status == "DSL_TOO_LARGE":
        normalised[witness_fields[0]] = _hex_bytes(
            _field(report, witness_fields[0]),
            label=f"{implementation}.{witness_fields[0]}",
            length=32,
        ).hex()
        witness_payload = _hex_bytes(
            _field(report, witness_fields[1]),
            label=f"{implementation}.{witness_fields[1]}",
        )
        if not witness_payload:
            _fail(FAIL_REPORT, f"{implementation} witness payload is empty")
        normalised[witness_fields[1]] = witness_payload.hex()
        ordinal = _field(
            report,
            witness_fields[2],
        )
        _expect(
            ordinal,
            CANONICAL_PROGRAM_BUDGET + 1,
            label=f"{implementation}.{witness_fields[2]}",
        )
        normalised[witness_fields[2]] = ordinal
    else:
        for name in witness_fields:
            observed = _field(report, name)
            if observed is not None:
                _fail(FAIL_REPORT, f"{implementation}.{name} must be null for COMPLETE")
            normalised[name] = None
    return normalised


def _decode_framed_stream(
    payload: bytes, *, expected_count: int, label: str
) -> tuple[bytes, ...]:
    records: list[bytes] = []
    offset = 0
    while offset < len(payload):
        if len(payload) - offset < 4:
            _fail(FAIL_OUTPUT, f"{label} has a truncated uint32-be frame header")
        length = int.from_bytes(payload[offset : offset + 4], "big")
        offset += 4
        if length < 1 or length > 1_048_576 or offset + length > len(payload):
            _fail(FAIL_OUTPUT, f"{label} has an invalid frame length")
        records.append(payload[offset : offset + length])
        offset += length
        if len(records) > expected_count:
            _fail(FAIL_OUTPUT, f"{label} contains too many records")
    if offset != len(payload) or len(records) != expected_count:
        _fail(
            FAIL_OUTPUT,
            f"{label} count differs: expected {expected_count}, got {len(records)}",
        )
    return tuple(records)


def _largest_power_of_two_less_than(value: int) -> int:
    return 1 << ((value - 1).bit_length() - 1)


def _rfc6962_root_from_payloads(records: Sequence[bytes]) -> bytes:
    """RFC6962 MTH over already-validated canonical record bytes."""

    if any(type(record) is not bytes for record in records):
        raise TypeError("RFC6962 raw records must be bytes")

    def subtree(first: int, last: int) -> bytes:
        count = last - first
        if count == 0:
            return sha256(b"").digest()
        if count == 1:
            return sha256(b"\x00" + records[first]).digest()
        split = _largest_power_of_two_less_than(count)
        return sha256(
            b"\x01"
            + subtree(first, first + split)
            + subtree(first + split, last)
        ).digest()

    return subtree(0, len(records))


def _array(
    payload: bytes,
    *,
    length: int,
    tag: int,
    schema: bytes,
    label: str,
) -> tuple[object, ...]:
    try:
        value = canonical_cbor_decode(payload)
    except (TypeError, ValueError) as error:
        _fail(FAIL_PROGRAM, f"{label} is not strict canonical CBOR: {error}")
    if (
        not isinstance(value, tuple)
        or len(value) != length
        or type(value[0]) is not int
        or value[0] != 1
        or type(value[1]) is not int
        or value[1] != tag
        or type(value[2]) is not bytes
        or value[2] != schema
    ):
        _fail(FAIL_PROGRAM, f"{label} schema/tag/array width differs")
    return value


def _record_uint(value: object, *, label: str, code: str) -> int:
    if type(value) is not int or value < 0:
        _fail(code, f"{label} must be uint")
    return value


def _record_bytes(
    value: object, *, label: str, code: str, length: int | None = None
) -> bytes:
    if type(value) is not bytes or (length is not None and len(value) != length):
        _fail(code, f"{label} must be bytes" + ("" if length is None else f"[{length}]"))
    return value


def _program_key(ast: object) -> tuple[int, int, int, int, bytes]:
    metrics = ast.metrics  # type: ignore[attr-defined]
    try:
        sort_id = OUTPUT_SORT_IDS[metrics.output_sort]
    except KeyError:
        _fail(FAIL_PROGRAM, "AST output sort is not registered")
    return (
        metrics.depth,
        metrics.node_count,
        sort_id,
        ast.root_operator_id,  # type: ignore[attr-defined]
        ast.cbor_bytes,  # type: ignore[attr-defined]
    )


def _derive_frozen_language_boundary() -> tuple[
    str,
    tuple[tuple[int, int, int, int, bytes], ...],
    tuple[int, int, int, int, bytes] | None,
    tuple[tuple[object, ...], ...],
    int,
    int,
]:
    """Derive the budget boundary before consulting either observed witness.

    Every traversal bucket is closed before the cumulative count is inspected.
    The returned prefix, optional rank-50,001 witness, and bucket counters are
    therefore consequences of the frozen typed source surface rather than of a
    byte pattern discovered in an endpoint report.
    """

    state = _child_enumerator._Shrink4Enumerator(
        raw_cap=RAW_OPERATOR_APPLICATION_CAP
    )
    ordered: list[object] = []
    traversal_complete = False
    stop = False
    sort_names = {value: key for key, value in OUTPUT_SORT_IDS.items()}
    try:
        for depth in range(MAXIMUM_AST_DEPTH + 1):
            for nodes in range(1, MAXIMUM_AST_NODE_COUNT + 1):
                for sort_id in range(1, len(OUTPUT_SORT_IDS) + 1):
                    if depth == 0 and nodes == 1:
                        state.leaves(sort_id)
                    elif depth >= 1 and nodes >= 2:
                        state.unary(depth, nodes, sort_id)
                        if nodes >= 3:
                            state.binary_and_ternary(depth, nodes, sort_id)
                            if sort_id == 1:
                                state.conjunctions(depth, nodes)
                    ordered.extend(
                        sorted(
                            state.groups.get(
                                (sort_names[sort_id], depth, nodes), ()
                            ),
                            key=lambda program: (
                                program.ast.root_operator_id,
                                program.ast.cbor_bytes,
                            ),
                        )
                    )
                    if len(ordered) > CANONICAL_PROGRAM_BUDGET:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break
        else:
            traversal_complete = True
    except _child_enumerator.BoundedEnumerationError as error:
        _fail(
            FAIL_WITNESS,
            f"frozen typed traversal could not derive a closed boundary: {error}",
        )

    prefix = tuple(ordered[:CANONICAL_PROGRAM_BUDGET])
    witness = (
        ordered[CANONICAL_PROGRAM_BUDGET]
        if len(ordered) > CANONICAL_PROGRAM_BUDGET
        else None
    )
    if witness is not None:
        status = "DSL_TOO_LARGE"
    elif traversal_complete:
        status = "COMPLETE"
    else:
        _fail(FAIL_WITNESS, "typed traversal ended without a witness or closed frontier")
    program_keys = tuple(_program_key(program.ast) for program in prefix)
    witness_key = None if witness is None else _program_key(witness.ast)
    bucket_records = tuple(
        _child_enumerator._base._bucket_records(state, prefix)
    )
    residual_count = len(ordered) - len(prefix)
    return (
        status,
        program_keys,
        witness_key,
        bucket_records,
        state.raw_count,
        residual_count,
    )


def _replay_programs(
    frames: Sequence[bytes], *, roots: tuple[bytes, bytes, bytes]
) -> tuple[
    tuple[tuple[object, ...], ...],
    list[tuple[int, int, int]],
    list[tuple[int, int, int, int, bytes]],
    set[bytes],
    set[bytes],
]:
    values: list[tuple[object, ...]] = []
    structural_keys: list[tuple[int, int, int]] = []
    program_keys: list[tuple[int, int, int, int, bytes]] = []
    ast_payloads: set[bytes] = set()
    ast_hashes: set[bytes] = set()
    previous_key: tuple[int, int, int, int, bytes] | None = None
    for index, payload in enumerate(frames):
        row = _array(
            payload,
            length=14,
            tag=PROGRAM_RECORD_TAG,
            schema=PROGRAM_RECORD_SCHEMA,
            label=f"CanonicalProgramRecordV2[{index}]",
        )
        if _record_uint(row[3], label="program_index", code=FAIL_PROGRAM) != index:
            _fail(FAIL_PROGRAM, f"program indices are not contiguous at {index}")
        ast_payload = _record_bytes(
            row[4], label="canonical_ast_cbor_bytes", code=FAIL_PROGRAM
        )
        ast_hash = _record_bytes(
            row[5], label="canonical_ast_hash", code=FAIL_PROGRAM, length=32
        )
        try:
            ast = decode_shrink4_canonical_ast(ast_payload)
        except (TypeError, ValueError) as error:
            _fail(FAIL_PROGRAM, f"program AST {index} is not admitted by shrink-4: {error}")
        key = _program_key(ast)
        sort_id = key[2]
        metadata = (
            sort_id,
            ast.metrics.depth,
            ast.metrics.node_count,
            len(ast.metrics.distinct_bit_slots),
            program_mdl_length_q32(ast),
        )
        observed_metadata = tuple(
            _record_uint(row[position], label=f"program[{index}][{position}]", code=FAIL_PROGRAM)
            for position in range(6, 11)
        )
        if ast.digest != ast_hash or observed_metadata != metadata:
            _fail(FAIL_PROGRAM, f"program AST/hash/metadata differs at {index}")
        observed_roots = tuple(
            _record_bytes(
                row[position],
                label=f"program[{index}].binding[{position - 11}]",
                code=FAIL_PROGRAM,
                length=32,
            )
            for position in range(11, 14)
        )
        if observed_roots != roots:
            _fail(FAIL_PROGRAM, f"program binding roots differ at {index}")
        if previous_key is not None and key <= previous_key:
            _fail(FAIL_PROGRAM, f"program traversal order is not strict at {index}")
        if ast_payload in ast_payloads or ast_hash in ast_hashes:
            _fail(FAIL_PROGRAM, f"program identity is duplicated at {index}")
        ast_payloads.add(ast_payload)
        ast_hashes.add(ast_hash)
        previous_key = key
        structural_keys.append((sort_id, ast.metrics.depth, ast.metrics.node_count))
        program_keys.append(key)
        values.append(row)
    if previous_key is None:
        _fail(FAIL_PROGRAM, "canonical program archive is empty")
    return tuple(values), structural_keys, program_keys, ast_payloads, ast_hashes


def _replay_chunks(
    frames: Sequence[bytes], program_frames: Sequence[bytes]
) -> tuple[tuple[object, ...], ...]:
    values: list[tuple[object, ...]] = []
    for chunk_index, payload in enumerate(frames):
        row = _array(
            payload,
            length=10,
            tag=CHUNK_MANIFEST_TAG,
            schema=CHUNK_MANIFEST_SCHEMA,
            label=f"ProgramChunkManifestV2[{chunk_index}]",
        )
        first = chunk_index * RECORDS_PER_CHUNK
        subset = tuple(program_frames[first : first + RECORDS_PER_CHUNK])
        if not subset:
            _fail(FAIL_CHUNK, f"chunk {chunk_index} is empty")
        blob = b"".join(len(item).to_bytes(4, "big") + item for item in subset)
        expected = (
            chunk_index,
            first,
            first + len(subset) - 1,
            len(subset),
            _rfc6962_root_from_payloads(subset),
            sha256(CHUNK_BLOB_DOMAIN + b"\x00" + blob).digest(),
            len(blob),
        )
        observed = (
            *(
                _record_uint(row[position], label=f"chunk[{chunk_index}][{position}]", code=FAIL_CHUNK)
                for position in range(3, 7)
            ),
            _record_bytes(row[7], label="chunk subtree root", code=FAIL_CHUNK, length=32),
            _record_bytes(row[8], label="chunk blob hash", code=FAIL_CHUNK, length=32),
            _record_uint(row[9], label="chunk blob length", code=FAIL_CHUNK),
        )
        if observed != expected:
            _fail(FAIL_CHUNK, f"chunk manifest {chunk_index} does not replay")
        values.append(row)
    return tuple(values)


def _replay_buckets(
    frames: Sequence[bytes],
    structural_keys: Sequence[tuple[int, int, int]],
    *,
    raw_count: int,
    witness_structural_key: tuple[int, int, int] | None,
) -> tuple[tuple[tuple[object, ...], ...], int]:
    expected_keys = tuple(
        (sort_id, depth, node_count)
        for sort_id in range(1, 6)
        for depth in range(5)
        for node_count in range(1, 8)
    )
    indices_by_key = {key: [] for key in expected_keys}
    for index, key in enumerate(structural_keys):
        try:
            indices_by_key[key].append(index)
        except KeyError:
            _fail(FAIL_BUCKET, f"program {index} lies outside the 175-bucket registry")

    values: list[tuple[object, ...]] = []
    residuals: list[tuple[tuple[int, int, int], int]] = []
    raw_total = 0
    accepted_total = 0
    raw_by_key: dict[tuple[int, int, int], int] = {}
    for bucket_index, (payload, expected_key) in enumerate(
        zip(frames, expected_keys, strict=True)
    ):
        row = _array(
            payload,
            length=15,
            tag=BUCKET_RECORD_TAG,
            schema=BUCKET_RECORD_SCHEMA,
            label=f"BucketAccountingRecordV1[{bucket_index}]",
        )
        identity = tuple(
            _record_uint(
                row[position],
                label=f"bucket[{bucket_index}][{position}]",
                code=FAIL_BUCKET,
            )
            for position in range(3, 7)
        )
        if identity != (bucket_index, *expected_key):
            _fail(FAIL_BUCKET, f"bucket identity/order differs at {bucket_index}")
        counters = tuple(
            _record_uint(
                row[position],
                label=f"bucket[{bucket_index}][{position}]",
                code=FAIL_BUCKET,
            )
            for position in range(7, 13)
        )
        raw, accepted, duplicate, type_reject, structural, rewrite = counters
        indices = indices_by_key[expected_key]
        expected_first = indices[0] if indices else None
        expected_last = indices[-1] if indices else None
        first_value, last_value = row[13], row[14]
        if first_value is not None:
            first_value = _record_uint(
                first_value, label=f"bucket[{bucket_index}].first", code=FAIL_BUCKET
            )
        if last_value is not None:
            last_value = _record_uint(
                last_value, label=f"bucket[{bucket_index}].last", code=FAIL_BUCKET
            )
        if (
            accepted != len(indices)
            or first_value != expected_first
            or last_value != expected_last
        ):
            _fail(FAIL_BUCKET, f"bucket archive range differs at {bucket_index}")
        residual = raw - accepted - duplicate - type_reject - structural - rewrite
        if residual < 0:
            _fail(FAIL_BUCKET, f"bucket counters over-partition raw at {bucket_index}")
        if residual:
            residuals.append((expected_key, residual))
        raw_by_key[expected_key] = raw
        raw_total += raw
        accepted_total += accepted
        values.append(row)
    if raw_total != raw_count or accepted_total != len(structural_keys):
        _fail(FAIL_BUCKET, "bucket raw/accepted totals do not replay")
    if witness_structural_key is None:
        if residuals:
            _fail(FAIL_BUCKET, "COMPLETE archive leaves residual canonical programs")
        return tuple(values), 0
    if len(residuals) != 1 or residuals[0][0] != witness_structural_key:
        _fail(
            FAIL_WITNESS,
            "exactly one residual canonical bucket must equal the witness bucket",
        )
    witness_traversal_key = (
        witness_structural_key[1],
        witness_structural_key[2],
        witness_structural_key[0],
    )
    if any(
        raw != 0
        for key, raw in raw_by_key.items()
        if (key[1], key[2], key[0]) > witness_traversal_key
    ):
        _fail(FAIL_WITNESS, "a traversal bucket after the witness bucket was touched")
    return tuple(values), residuals[0][1]


def validate_shrink4_dual_diagnostic_v1(
    python_output_directory: Path | str,
    rust_output_directory: Path | str,
) -> dict[str, object]:
    """Replay and compare both non-formal shrink-4 enumeration outputs."""

    loaded_before = _assert_host_module_closure()
    python_report, python_report_payload, python_streams = _load_output_directory(
        python_output_directory, implementation="python"
    )
    rust_report, rust_report_payload, rust_streams = _load_output_directory(
        rust_output_directory, implementation="rust"
    )
    python_normalised = _normalise_report(python_report, implementation="python")
    rust_normalised = _normalise_report(rust_report, implementation="rust")
    if python_normalised != rust_normalised:
        differing = sorted(
            name
            for name in set(python_normalised) | set(rust_normalised)
            if python_normalised.get(name) != rust_normalised.get(name)
        )
        _fail(FAIL_DUAL, f"Python/Rust report fields differ: {differing}")
    if python_streams != rust_streams:
        differing = sorted(
            name for name in _STREAM_NAMES if python_streams[name] != rust_streams[name]
        )
        _fail(FAIL_DUAL, f"Python/Rust framed archive bytes differ: {differing}")

    closure_status = str(python_normalised["closure_status"])
    program_count = int(python_normalised["canonical_program_count"])
    chunk_count = int(python_normalised["chunk_manifest_count"])
    program_frames = _decode_framed_stream(
        python_streams["canonical_program_records"],
        expected_count=program_count,
        label="canonical_program_records.cborframed",
    )
    chunk_frames = _decode_framed_stream(
        python_streams["program_chunk_manifests"],
        expected_count=chunk_count,
        label="program_chunk_manifests.cborframed",
    )
    bucket_frames = _decode_framed_stream(
        python_streams["bucket_accounting_records"],
        expected_count=FORMAL_BUCKET_COUNT,
        label="bucket_accounting_records.cborframed",
    )
    roots = _EXPECTED_ROOTS[:3]
    program_values, structural_keys, program_keys, ast_payloads, ast_hashes = (
        _replay_programs(program_frames, roots=roots)
    )
    program_root = _rfc6962_root_from_payloads(program_frames)
    if program_root.hex() != python_normalised["canonical_program_archive_root_or_null"]:
        _fail(FAIL_PROGRAM, "canonical program archive root does not replay")

    (
        derived_status,
        derived_program_keys,
        derived_witness_key,
        derived_bucket_values,
        derived_raw_count,
        derived_residual_count,
    ) = _derive_frozen_language_boundary()
    if closure_status != derived_status:
        _fail(
            FAIL_WITNESS,
            f"reported terminal status {closure_status} differs from the "
            f"independently derived {derived_status}",
        )
    if tuple(program_keys) != derived_program_keys:
        _fail(
            FAIL_PROGRAM,
            "archive is not the exact prefix independently derived from the "
            "frozen typed source surface",
        )
    if int(python_normalised["raw_operator_application_count"]) != derived_raw_count:
        _fail(FAIL_BUCKET, "reported raw count differs from typed traversal replay")

    witness_payload: bytes | None = None
    witness_hash: bytes | None = None
    witness_structural_key: tuple[int, int, int] | None = None
    if closure_status == "DSL_TOO_LARGE":
        witness_payload = bytes.fromhex(
            str(python_normalised["first_out_of_budget_program_cbor_hex_or_null"])
        )
        witness_hash = bytes.fromhex(
            str(python_normalised["first_out_of_budget_program_hash_or_null"])
        )
        try:
            witness = decode_shrink4_canonical_ast(witness_payload)
        except (TypeError, ValueError) as error:
            _fail(FAIL_WITNESS, f"witness is not a shrink-4 canonical AST: {error}")
        if witness.digest != witness_hash:
            _fail(FAIL_WITNESS, "witness hash does not bind witness CBOR")
        if witness_payload in ast_payloads or witness_hash in ast_hashes:
            _fail(FAIL_WITNESS, "50,001 witness is already in the archive")
        witness_key = _program_key(witness)
        witness_structural_key = (
            witness_key[2],
            witness_key[0],
            witness_key[1],
        )
        if derived_witness_key is None or witness_key != derived_witness_key:
            _fail(
                FAIL_WITNESS,
                "reported witness is not rank 50,001 in the independently "
                "closed typed traversal bucket",
            )
    elif derived_witness_key is not None:
        _fail(FAIL_WITNESS, "COMPLETE report conflicts with a derived witness")

    chunk_values = _replay_chunks(chunk_frames, program_frames)
    chunk_root = _rfc6962_root_from_payloads(chunk_frames)
    if chunk_root.hex() != python_normalised["program_chunk_manifest_root_or_null"]:
        _fail(FAIL_CHUNK, "program chunk manifest root does not replay")
    bucket_values, residual_count = _replay_buckets(
        bucket_frames,
        structural_keys,
        raw_count=int(python_normalised["raw_operator_application_count"]),
        witness_structural_key=witness_structural_key,
    )
    bucket_root = _rfc6962_root_from_payloads(bucket_frames)
    if bucket_root.hex() != python_normalised["bucket_accounting_root_or_null"]:
        _fail(FAIL_BUCKET, "bucket accounting root does not replay")
    if bucket_values != derived_bucket_values:
        _fail(
            FAIL_BUCKET,
            "bucket accounting differs from the independently replayed typed traversal",
        )
    if residual_count != derived_residual_count:
        _fail(
            FAIL_WITNESS,
            "residual canonical count differs from the closed boundary bucket",
        )

    # Silence accidental future substitutions of decoded-object roots for raw
    # roots: the values are retained only to show that every frame was decoded.
    if (
        len(program_values) != program_count
        or len(chunk_values) != len(chunk_frames)
        or len(bucket_values) != FORMAL_BUCKET_COUNT
    ):
        _fail(FAIL_OUTPUT, "internal decoded archive cardinality differs")

    stream_digests = {
        name: sha256(python_streams[name]).hexdigest() for name in _STREAM_NAMES
    }
    loaded_after = _assert_host_module_closure()
    if loaded_after != loaded_before:
        _fail(FAIL_HOST_ISOLATION, "host module closure changed during replay")
    return {
        "schema_version": RECEIPT_SCHEMA,
        "claim_level": CLAIM_LEVEL,
        "qualification_level": "DIAGNOSTIC_ONLY",
        "profile_id": PROFILE_ID,
        "binding_profile_id": BINDING_PROFILE_ID,
        "diagnostic_only": True,
        "authoritative_claim_allowed": False,
        "execution_state": "NOT_RUN",
        "formal_roots_generated": False,
        "formal_roots": None,
        "formal_state_transition_allowed": False,
        "strict_qualification_source_commit": STRICT_QUALIFICATION_SOURCE_COMMIT,
        "strict_qualification_evidence_commit": (
            STRICT_QUALIFICATION_EVIDENCE_COMMIT
        ),
        "strict_qualification_artifact_path": STRICT_QUALIFICATION_ARTIFACT_PATH,
        "strict_qualification_artifact_sha256": (
            STRICT_QUALIFICATION_ARTIFACT_SHA256
        ),
        "strict_qualification_diagnostic_report_hash": (
            STRICT_QUALIFICATION_DIAGNOSTIC_REPORT_HASH
        ),
        "strict_qualification_status": STRICT_QUALIFICATION_STATUS,
        "maximum_top_level_clauses": MAXIMUM_TOP_LEVEL_CLAUSES,
        "and3_generator_attempts_allowed": False,
        "and3_raw_operator_application_count": 0,
        "dual_reports_equal": True,
        "dual_archive_bytes_equal": True,
        "host_strict_archive_replay_verified": True,
        "host_loaded_hegel_modules": list(loaded_after),
        "host_target_free_isolation_verified": True,
        "host_target_or_split_modules_loaded": False,
        "independence_scope": (
            "INDEPENDENT_OF_ENDPOINT_REPORTED_WITNESS_NOT_A_THIRD_IMPLEMENTATION"
        ),
        "typed_language_boundary_independently_derived": True,
        "archive_prefix_exact": True,
        "program_indices_verified": True,
        "program_binding_roots_verified": True,
        "binary_operator_registry_verified": True,
        "removed_binary_operator_absent_from_archive": True,
        "operator_id_compaction_performed": False,
        "automatic_operator_migration_performed": False,
        "chunk_framing_and_blob_hashes_verified": True,
        "bucket_accounting_verified": True,
        "witness_adjacency_verified": (
            True if closure_status == "DSL_TOO_LARGE" else None
        ),
        "witness_closed_bucket_rank_verified": (
            True if closure_status == "DSL_TOO_LARGE" else None
        ),
        "post_witness_traversal_buckets_untouched": (
            True if closure_status == "DSL_TOO_LARGE" else None
        ),
        "residual_out_of_budget_canonical_programs": residual_count,
        "closure_status": closure_status,
        "canonical_program_count": program_count,
        "raw_operator_application_count": python_normalised[
            "raw_operator_application_count"
        ],
        "raw_operator_application_count_scope": (
            "THROUGH_FULLY_CLOSED_BOUNDARY_BUCKET"
            if closure_status == "DSL_TOO_LARGE"
            else "THROUGH_FULLY_CLOSED_FRONTIER"
        ),
        "canonical_program_archive_root": program_root.hex(),
        "program_chunk_manifest_root": chunk_root.hex(),
        "bucket_accounting_root": bucket_root.hex(),
        "first_out_of_budget_program_hash": (
            None if witness_hash is None else witness_hash.hex()
        ),
        "first_out_of_budget_program_cbor_hex": (
            None if witness_payload is None else witness_payload.hex()
        ),
        "python_report_sha256": sha256(python_report_payload).hexdigest(),
        "rust_report_sha256": sha256(rust_report_payload).hexdigest(),
        "stream_sha256": stream_digests,
        "target_roles_evaluated": False,
        "split_material_accessed": False,
        "secrets_accessed": False,
    }


__all__ = [
    "FAIL_AUTHORITY",
    "FAIL_BUCKET",
    "FAIL_CHUNK",
    "FAIL_DUAL",
    "FAIL_OUTPUT",
    "FAIL_PROGRAM",
    "FAIL_REPORT",
    "FAIL_WITNESS",
    "FAIL_HOST_ISOLATION",
    "RECEIPT_SCHEMA",
    "Shrink4DualDiagnosticError",
    "validate_shrink4_dual_diagnostic_v1",
]
