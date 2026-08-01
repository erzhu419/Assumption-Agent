"""Independent typed-row qualification for the M2.5 v1.1.2 amendment.

The report produced here compares a Python implementation, an independently
implemented Rust executable, and the checked-in public golden fixture.  The
four 480/85-row RFC6962 values are *candidate deterministic roots*.  They are
not the externally attested formal roots required by M3 gates 15--24.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Final, Mapping

from .hashing import stable_hash
from .phase3_m25_rows_v1 import complete_typed_rows_report_v1


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
MACHINE_FREEZE_ID: Final = "hegel-freeze-p2b-p3-v1.1.2"
CHILD_DSL_ID: Final = "hegel-old-dsl-v1.1.0"
REPORT_SCHEMA: Final = "hegel-phase3-m25-wire-completion-qualification/1"
ARTIFACT_KIND: Final = "DETERMINISTIC_CANDIDATE_NON_AUTHORITATIVE"
STATUS: Final = "DUAL_TYPED_ROWS_AND_ROOTS_CANDIDATE_PASS"
BINARY_PROVENANCE: Final = "CALLER_SUPPLIED_UNATTESTED_V112_REPLAY"
CLAIM_BOUNDARY: Final = (
    "Python and a caller-supplied, unattested Rust executable reproduce "
    "the same public typed inputs, 565 formal rows, and four "
    "amendment-listed RFC6962 values. The report binds the executable "
    "digest but does not claim it was built from the listed Rust source. "
    "The values are deterministic candidate roots only: no external "
    "custodian, auditor, attester, seed, signature, formal-root "
    "publication, M3 execution identity, closure verdict, or certificate "
    "is claimed."
)

NORMATIVE_DOCUMENT: Final = (
    PROJECT_ROOT
    / "docs"
    / "Hegel_Machine_Phase3A_M25_Bit_Exact_Wire_Completion_Amendment.md"
)
GOLDEN_VECTOR_PATH: Final = (
    PROJECT_ROOT / "golden_vectors" / "phase3_m25_typed_rows_v1.json"
)
RUST_CRATE_ROOT: Final = PROJECT_ROOT / "rust" / "formal_bridge_m25"
DEFAULT_RUST_BINARY: Final = (
    RUST_CRATE_ROOT / "target" / "debug" / "hegel-formal-bridge-m25"
)

EXPECTED_ROLE_ROOTS: Final = {
    "odd": {
        "row_count": 480,
        "universe_root_hex": (
            "b7e6eed1174ceee1944bd540cbb0ace4c5c7fc0dffea79dd956a51f7a2410a05"
        ),
        "truth_root_hex": (
            "f5bbdc26bec62f9966e5ef31eaa800190ed52dedc73ee61545e0f9c122a1a506"
        ),
    },
    "sink": {
        "row_count": 85,
        "universe_root_hex": (
            "1a46f9967ad48df6ec1d9be609de701413ce86171fb0f92495a982bef0f40ff5"
        ),
        "truth_root_hex": (
            "9c0f5d75ea3c31f6cb1ea9917346a7a3f480ae9ce0ac0cb3bb21aac9d3bd7808"
        ),
    },
}

SOURCE_PATHS: Final = (
    "src/hegel_machine/strict_cbor_v1.py",
    "src/hegel_machine/hashing.py",
    "src/hegel_machine/phase3_m25_rows_v1.py",
    "src/hegel_machine/phase3_m25_qualification_v112.py",
    "rust/formal_bridge_m25/Cargo.toml",
    "rust/formal_bridge_m25/Cargo.lock",
    "rust/formal_bridge_m25/src/lib.rs",
    "rust/formal_bridge_m25/src/main.rs",
    "golden_vectors/phase3_m25_typed_rows_v1.json",
)


class M25QualificationError(RuntimeError):
    """Fail-closed error raised by the cross-language qualification runner."""


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _source_bindings() -> dict[str, str]:
    bindings: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise M25QualificationError(f"missing qualification source: {relative}")
        bindings[relative] = _sha256_file(path)
    return bindings


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise M25QualificationError(f"{name} must be a string-keyed object")
    return value


def _json_type_strict_equal(left: object, right: object) -> bool:
    """Compare diagnostic JSON without bool/int or int/float coercion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_type_strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_type_strict_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _rust_request(rust_binary: Path, request: Mapping[str, object]) -> dict[str, object]:
    if not rust_binary.is_file():
        raise M25QualificationError(f"Rust executable is missing: {rust_binary}")
    completed = subprocess.run(
        [str(rust_binary)],
        input=json.dumps(dict(request), sort_keys=True),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise M25QualificationError(
            "Rust typed-row replay returned invalid JSON: " + completed.stderr
        ) from exc
    response = dict(_mapping(payload, "Rust response"))
    if completed.returncode != 0 or response.get("ok") is not True:
        raise M25QualificationError(f"Rust typed-row replay failed: {response}")
    return response


def _strip_rust_response_envelope(
    response: Mapping[str, object],
    *,
    expected_operation: str,
    exact_payload_fields: frozenset[str],
) -> dict[str, object]:
    expected_fields = {"ok", "op"} | set(exact_payload_fields)
    if set(response) != expected_fields:
        raise M25QualificationError(
            f"Rust {expected_operation} response field-set mismatch"
        )
    if response.get("ok") is not True:
        raise M25QualificationError(f"Rust {expected_operation} response is not ok")
    if response.get("op") != expected_operation:
        raise M25QualificationError(
            f"Rust response op does not echo {expected_operation!r}"
        )
    return {
        field: response[field]
        for field in sorted(exact_payload_fields)
    }


def _rust_report(rust_binary: Path) -> dict[str, object]:
    python_report = complete_typed_rows_report_v1()
    id_vector = _mapping(python_report["id_digest"], "Python IdDigest vector")
    machine_id = id_vector.get("machine_id")
    if not isinstance(machine_id, str):
        raise M25QualificationError("Python IdDigest machine_id is not text")
    if machine_id != CHILD_DSL_ID:
        raise M25QualificationError("typed-row IdDigest does not bind the child DSL ID")

    raw_id = _strip_rust_response_envelope(
        _rust_request(
            rust_binary,
            {"op": "id_digest", "machine_id": machine_id},
        ),
        expected_operation="id_digest",
        exact_payload_fields=frozenset(
            {"machine_id", "preimage_hex", "digest_hex"}
        ),
    )

    roles: list[dict[str, object]] = []
    for role_id in (1, 2):
        raw_role = _strip_rust_response_envelope(
            _rust_request(
                rust_binary,
                {"op": "typed_rows", "role_id": role_id},
            ),
            expected_operation="typed_rows",
            exact_payload_fields=frozenset(
                {
                    "role_name",
                    "input_signature_id",
                    "row_count",
                    "samples",
                    "universe_two_row_root_hex",
                    "truth_two_row_root_hex",
                    "universe_root_hex",
                    "truth_root_hex",
                }
            ),
        )
        roles.append(raw_role)
    return {
        "machine_freeze_id": MACHINE_FREEZE_ID,
        "id_digest": raw_id,
        "roles": roles,
    }


def _load_golden() -> dict[str, object]:
    payload = json.loads(GOLDEN_VECTOR_PATH.read_text(encoding="utf-8"))
    fixture = dict(_mapping(payload, "typed-row golden fixture"))
    expected_keys = {
        "schema_version",
        "artifact_kind",
        "machine_freeze_id",
        "cbor_profile_id",
        "hash_algorithm",
        "authority_boundary",
        "id_digest",
        "roles",
    }
    if set(fixture) != expected_keys:
        raise M25QualificationError("typed-row fixture field-set mismatch")
    expected_metadata = {
        "schema_version": "hegel-phase3-m25-typed-rows-golden/1",
        "artifact_kind": "SYNTHETIC_NON_AUTHORITATIVE",
        "machine_freeze_id": MACHINE_FREEZE_ID,
        "cbor_profile_id": "hegel-cbor-det-v1",
        "hash_algorithm": "SHA-256",
        "authority_boundary": {
            "gate_effect": "NONE",
            "m3_gates_before": 14,
            "m3_gates_after": 14,
            "child_state": "NOT_RUN",
            "contains_real_secret_material": False,
            "authoritative_root_generation": False,
            "formal_roots_generated": False,
            "seed_genesis_performed": False,
            "signature_claim": False,
        },
    }
    for field, expected in expected_metadata.items():
        if not _json_type_strict_equal(fixture.get(field), expected):
            raise M25QualificationError(f"typed-row fixture {field} drift")
    return fixture


def _assert_expected_roots(report: Mapping[str, object]) -> None:
    roles = report.get("roles")
    if not isinstance(roles, list) or len(roles) != 2:
        raise M25QualificationError("typed-row report must contain odd and sink roles")
    seen: set[str] = set()
    for raw_role in roles:
        role = _mapping(raw_role, "typed role")
        role_name = role.get("role_name")
        if not isinstance(role_name, str) or role_name not in EXPECTED_ROLE_ROOTS:
            raise M25QualificationError(f"unexpected typed role: {role_name!r}")
        expected = EXPECTED_ROLE_ROOTS[role_name]
        for field, expected_value in expected.items():
            if role.get(field) != expected_value:
                raise M25QualificationError(
                    f"{role_name} {field} differs from the amendment"
                )
        seen.add(role_name)
    if seen != set(EXPECTED_ROLE_ROOTS):
        raise M25QualificationError("typed-row role registry is incomplete")


def dual_typed_rows_qualification_report(
    rust_binary: Path = DEFAULT_RUST_BINARY,
) -> dict[str, object]:
    """Run the public Python/Rust v1.1.2 comparison without minting authority."""

    rust_binary = rust_binary.resolve()
    source_bindings = _source_bindings()
    normative_document_sha256 = _sha256_file(NORMATIVE_DOCUMENT)
    golden_fixture_sha256 = _sha256_file(GOLDEN_VECTOR_PATH)
    rust_binary_sha256 = _sha256_file(rust_binary)

    golden = _load_golden()
    python_report = complete_typed_rows_report_v1()
    rust_report = _rust_report(rust_binary)
    golden_core = {
        "machine_freeze_id": golden.get("machine_freeze_id"),
        "id_digest": golden.get("id_digest"),
        "roles": golden.get("roles"),
    }
    if not _json_type_strict_equal(python_report, golden_core):
        raise M25QualificationError("Python typed rows differ from the golden fixture")
    if not _json_type_strict_equal(rust_report, golden_core):
        raise M25QualificationError("Rust typed rows differ from the golden fixture")
    _assert_expected_roots(python_report)
    if _source_bindings() != source_bindings:
        raise M25QualificationError("qualification sources changed during replay")
    if _sha256_file(NORMATIVE_DOCUMENT) != normative_document_sha256:
        raise M25QualificationError("normative document changed during replay")
    if _sha256_file(GOLDEN_VECTOR_PATH) != golden_fixture_sha256:
        raise M25QualificationError("golden fixture changed during replay")
    if _sha256_file(rust_binary) != rust_binary_sha256:
        raise M25QualificationError("Rust executable changed during replay")

    payload: dict[str, object] = {
        "artifact": "phase3_m25_wire_completion_qualification_v112",
        "schema_version": REPORT_SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "machine_freeze_id": MACHINE_FREEZE_ID,
        "child_dsl_id": CHILD_DSL_ID,
        "status": STATUS,
        "normative_document_sha256": normative_document_sha256,
        "golden_fixture_sha256": golden_fixture_sha256,
        "source_bindings": source_bindings,
        "source_snapshot_stable_during_replay": True,
        "rust_execution": {
            "binary_sha256": rust_binary_sha256,
            "binary_provenance": BINARY_PROVENANCE,
            "binary_source_binding_claim": False,
            "listed_rust_sources_are_build_attestation": False,
        },
        "python_report": python_report,
        "rust_report": rust_report,
        "cross_language_exact_match": True,
        "amendment_expected_roots_match": True,
        "qualified_row_counts": {"odd": 480, "sink": 85},
        "candidate_role_roots": {
            role_name: dict(values)
            for role_name, values in EXPECTED_ROLE_ROOTS.items()
        },
        "formal_input_roots": None,
        "formal_roots_generated": False,
        "m3_execution_manifest_root": None,
        "authority_boundary": {
            "candidate_roots_are_formal_roots": False,
            "authoritative_root_generation": False,
            "seed_genesis_performed": False,
            "signature_claim": False,
            "m3_gate_delta": 0,
            "m3_gates_before": 14,
            "m3_gates_after": 14,
            "child_state": "NOT_RUN",
            "m3_start_authorized": False,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["diagnostic_report_id"] = stable_hash(
        payload,
        prefix="phase3_m25_wire_completion_qualification_",
    )
    return payload


def validate_dual_typed_rows_qualification_report(
    report: Mapping[str, object],
    rust_binary: Path = DEFAULT_RUST_BINARY,
) -> None:
    """Replay Rust and reject any difference from the supplied fresh report."""

    validate_checked_typed_rows_qualification_report(report)
    expected = dual_typed_rows_qualification_report(rust_binary)
    if not _json_type_strict_equal(dict(report), expected):
        raise M25QualificationError("qualification report differs from current replay")


def validate_checked_typed_rows_qualification_report(
    report: Mapping[str, object],
) -> None:
    """Validate checked evidence without requiring a reproducible binary hash.

    Debug-binary bytes may differ across toolchains and build paths. The stored
    binary digest and report self-ID are checked for internal integrity, while
    current source/golden hashes and both endpoint payloads are checked exactly.
    A fresh replay is a separate operation.
    """

    expected_fields = {
        "artifact",
        "schema_version",
        "artifact_kind",
        "machine_freeze_id",
        "child_dsl_id",
        "status",
        "normative_document_sha256",
        "golden_fixture_sha256",
        "source_bindings",
        "source_snapshot_stable_during_replay",
        "rust_execution",
        "python_report",
        "rust_report",
        "cross_language_exact_match",
        "amendment_expected_roots_match",
        "qualified_row_counts",
        "candidate_role_roots",
        "formal_input_roots",
        "formal_roots_generated",
        "m3_execution_manifest_root",
        "authority_boundary",
        "claim_boundary",
        "diagnostic_report_id",
    }
    if set(report) != expected_fields:
        raise M25QualificationError("qualification report field-set drift")
    if report.get("artifact") != "phase3_m25_wire_completion_qualification_v112":
        raise M25QualificationError("qualification artifact identity drift")
    if report.get("schema_version") != REPORT_SCHEMA:
        raise M25QualificationError("qualification report schema drift")
    if report.get("artifact_kind") != ARTIFACT_KIND:
        raise M25QualificationError("qualification artifact must remain non-authoritative")
    if report.get("status") != STATUS:
        raise M25QualificationError("qualification status drift")
    if report.get("machine_freeze_id") != MACHINE_FREEZE_ID:
        raise M25QualificationError("qualification freeze drift")
    if report.get("child_dsl_id") != CHILD_DSL_ID:
        raise M25QualificationError("qualification child DSL drift")
    if report.get("claim_boundary") != CLAIM_BOUNDARY:
        raise M25QualificationError("qualification claim boundary drift")
    if report.get("source_bindings") != _source_bindings():
        raise M25QualificationError("qualification source bindings are stale")
    if report.get("normative_document_sha256") != _sha256_file(NORMATIVE_DOCUMENT):
        raise M25QualificationError("qualification normative document is stale")
    if report.get("golden_fixture_sha256") != _sha256_file(GOLDEN_VECTOR_PATH):
        raise M25QualificationError("qualification golden fixture is stale")
    if report.get("source_snapshot_stable_during_replay") is not True:
        raise M25QualificationError("qualification source-snapshot claim drift")
    if report.get("cross_language_exact_match") is not True:
        raise M25QualificationError("qualification cross-language match drift")
    if report.get("amendment_expected_roots_match") is not True:
        raise M25QualificationError("qualification amendment-root match drift")
    if not _json_type_strict_equal(
        report.get("qualified_row_counts"),
        {"odd": 480, "sink": 85},
    ):
        raise M25QualificationError("qualification row counts drift")
    if not _json_type_strict_equal(
        report.get("candidate_role_roots"),
        EXPECTED_ROLE_ROOTS,
    ):
        raise M25QualificationError("qualification candidate roots drift")
    if report.get("formal_input_roots") is not None:
        raise M25QualificationError("qualification cannot contain formal input roots")
    if report.get("formal_roots_generated") is not False:
        raise M25QualificationError("qualification cannot claim formal roots")
    if report.get("m3_execution_manifest_root") is not None:
        raise M25QualificationError("qualification cannot contain an M3 execution root")

    golden = _load_golden()
    golden_core = {
        "machine_freeze_id": golden.get("machine_freeze_id"),
        "id_digest": golden.get("id_digest"),
        "roles": golden.get("roles"),
    }
    python_report = complete_typed_rows_report_v1()
    if not _json_type_strict_equal(report.get("python_report"), python_report):
        raise M25QualificationError("qualification Python report is stale")
    if not _json_type_strict_equal(report.get("rust_report"), golden_core):
        raise M25QualificationError("qualification stored Rust report is stale")
    _assert_expected_roots(python_report)

    rust_execution = _mapping(report.get("rust_execution"), "Rust execution")
    if set(rust_execution) != {
        "binary_sha256",
        "binary_provenance",
        "binary_source_binding_claim",
        "listed_rust_sources_are_build_attestation",
    }:
        raise M25QualificationError("Rust execution field-set drift")
    if rust_execution.get("binary_provenance") != BINARY_PROVENANCE:
        raise M25QualificationError("Rust executable provenance boundary drift")
    if rust_execution.get("binary_source_binding_claim") is not False:
        raise M25QualificationError("unattested Rust binary cannot claim source binding")
    if rust_execution.get("listed_rust_sources_are_build_attestation") is not False:
        raise M25QualificationError("source hashes are not a Rust build attestation")
    binary_sha256 = rust_execution.get("binary_sha256")
    if (
        not isinstance(binary_sha256, str)
        or len(binary_sha256) != 71
        or not binary_sha256.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in binary_sha256[7:])
    ):
        raise M25QualificationError("Rust binary digest syntax drift")
    authority = _mapping(report.get("authority_boundary"), "authority boundary")
    expected_authority = {
        "candidate_roots_are_formal_roots": False,
        "authoritative_root_generation": False,
        "seed_genesis_performed": False,
        "signature_claim": False,
        "m3_gate_delta": 0,
        "m3_gates_before": 14,
        "m3_gates_after": 14,
        "child_state": "NOT_RUN",
        "m3_start_authorized": False,
    }
    if not _json_type_strict_equal(dict(authority), expected_authority):
        raise M25QualificationError("qualification authority boundary drift")

    provided_id = report.get("diagnostic_report_id")
    body = dict(report)
    body.pop("diagnostic_report_id", None)
    if provided_id != stable_hash(
        body,
        prefix="phase3_m25_wire_completion_qualification_",
    ):
        raise M25QualificationError("qualification report self-ID mismatch")


__all__ = [
    "ARTIFACT_KIND",
    "CLAIM_BOUNDARY",
    "DEFAULT_RUST_BINARY",
    "EXPECTED_ROLE_ROOTS",
    "M25QualificationError",
    "MACHINE_FREEZE_ID",
    "STATUS",
    "dual_typed_rows_qualification_report",
    "validate_checked_typed_rows_qualification_report",
    "validate_dual_typed_rows_qualification_report",
]
