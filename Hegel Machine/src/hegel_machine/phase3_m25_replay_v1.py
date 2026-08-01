"""Synthetic Python/Rust replay for the Phase-3A M2.5 primitives.

The shared fixture contains only public deterministic test inputs.  Replaying
it qualifies byte-level foundation implementations; it does not instantiate a
split seed, sign a custodian manifest, mint a formal root, advance an M3 gate,
or move the child state out of ``NOT_RUN``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Final

from . import phase3_m25_split_v1 as split
from . import phase3_m25_wire_v1 as wire
from .hashing import stable_hash
from .strict_cbor_v1 import StrictCborError


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
GOLDEN_VECTOR_PATH: Final = (
    PROJECT_ROOT / "golden_vectors" / "phase3_m25_formal_wire_v1.json"
)
RUST_CRATE_ROOT: Final = PROJECT_ROOT / "rust" / "formal_bridge_m25"
DEFAULT_RUST_BINARY: Final = (
    RUST_CRATE_ROOT / "target" / "debug" / "hegel-formal-bridge-m25"
)
CHECKED_IN_REPORT_PATH: Final = (
    PROJECT_ROOT / "artifacts" / "phase3_m25_synthetic_dual_replay_v1.json"
)
HISTORICAL_CHECKED_IN_REPORT_ID: Final = (
    "phase3_m25_synthetic_dual_replay_"
    "2ecdda87155ac648fa4dee91bd56fcbae488f423b4ca9699e0d145f5d675706a"
)
HISTORICAL_CHECKED_IN_COMMIT: Final = "d772b844"

MACHINE_FREEZE_ID: Final = "hegel-freeze-p2b-p3-v1.1.1"
GOLDEN_SCHEMA: Final = "hegel-phase3-m25-formal-wire-golden/1"
REPORT_SCHEMA: Final = "hegel-phase3-m25-synthetic-dual-replay/1"
ARTIFACT_KIND: Final = "SYNTHETIC_NON_AUTHORITATIVE"
BINARY_PROVENANCE: Final = "CALLER_SUPPLIED_UNATTESTED_SYNTHETIC_REPLAY"
CLAIM_BOUNDARY: Final = (
    "This is a synthetic cross-language primitive replay only. It does "
    "not instantiate the split seed, exercise custody, create a signed "
    "manifest, generate a formal root, advance any of the 24 M3 gates, "
    "or authorize NOT_RUN to RUNNING."
)

PYTHON_SOURCE_FILES: Final = (
    PROJECT_ROOT / "src" / "hegel_machine" / "strict_cbor_v1.py",
    PROJECT_ROOT / "src" / "hegel_machine" / "phase3_m25_wire_v1.py",
    PROJECT_ROOT / "src" / "hegel_machine" / "phase3_m25_split_v1.py",
    Path(__file__),
)
RUST_SOURCE_FILES: Final = (
    RUST_CRATE_ROOT / "Cargo.toml",
    RUST_CRATE_ROOT / "Cargo.lock",
    RUST_CRATE_ROOT / "src" / "lib.rs",
    RUST_CRATE_ROOT / "src" / "main.rs",
)

SUPPORTED_OPERATIONS: Final = frozenset(
    {
        "encode",
        "decode",
        "reject_decode",
        "content_hash",
        "rfc6962_root",
        "derive_role_key",
        "row_rank",
        "seed_commitment",
    }
)
RESULT_FIELDS: Final = {
    "encode": ("cbor_hex",),
    "decode": ("canonical_cbor_hex", "value"),
    "reject_decode": ("accepted", "error_code"),
    "content_hash": ("cbor_hex", "digest_hex"),
    "rfc6962_root": ("leaf_count", "root_hex"),
    "derive_role_key": ("role_key_hex",),
    "row_rank": ("rank_hex",),
    "seed_commitment": ("commitment_hex",),
}


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _source_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {_relative(path): _sha256_file(path) for path in sorted(paths)}


def _source_set_hash(paths: Sequence[Path]) -> str:
    """Bind ordered path names and bytes without claiming a formal root."""

    digest = hashlib.sha256()
    digest.update(b"HEGEL/M25/SYNTHETIC_SOURCE_SET/V1\x00")
    for path in sorted(paths, key=_relative):
        relative = _relative(path).encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return "sha256:" + digest.hexdigest()


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _require_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _json_type_strict_equal(left: object, right: object) -> bool:
    """Compare diagnostic JSON without Python's ``False == 0`` coercion."""

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


def _decode_hex(value: object, name: str) -> bytes:
    encoded = _require_string(value, name)
    if len(encoded) % 2 or encoded != encoded.lower():
        raise ValueError(f"{name} must be lowercase even-length hexadecimal")
    try:
        decoded = bytes.fromhex(encoded)
    except ValueError as exc:
        raise ValueError(f"{name} must be lowercase hexadecimal") from exc
    if decoded.hex() != encoded:
        raise ValueError(f"{name} must use canonical lowercase hexadecimal")
    return decoded


def _formal_from_transport(value: object) -> wire.StrictCborValue:
    if value is None or value is False or value is True or type(value) is int:
        return value
    if isinstance(value, list):
        return tuple(_formal_from_transport(item) for item in value)
    if isinstance(value, Mapping):
        if set(value) != {"bytes_hex"}:
            raise ValueError(
                "formal byte strings require an exact bytes_hex wrapper; "
                "formal maps are forbidden"
            )
        return _decode_hex(value["bytes_hex"], "bytes_hex")
    raise ValueError(f"unsupported formal transport value {type(value).__name__}")


def _transport_from_formal(value: wire.StrictCborValue) -> object:
    if type(value) is bytes:
        return {"bytes_hex": value.hex()}
    if isinstance(value, tuple):
        return [_transport_from_formal(item) for item in value]
    if value is None or value is False or value is True or type(value) is int:
        return value
    raise TypeError(f"unexpected strict CBOR value {type(value).__name__}")


def load_golden_vectors(path: Path = GOLDEN_VECTOR_PATH) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fixture = dict(_require_mapping(payload, "golden fixture"))
    if fixture.get("schema_version") != GOLDEN_SCHEMA:
        raise ValueError("M2.5 golden fixture schema mismatch")
    if fixture.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("M2.5 golden fixture must be SYNTHETIC_NON_AUTHORITATIVE")
    if fixture.get("machine_freeze_id") != MACHINE_FREEZE_ID:
        raise ValueError("M2.5 golden fixture freeze mismatch")
    authority = _require_mapping(fixture.get("authority_boundary"), "authority_boundary")
    exact_authority = {
        "gate_effect": "NONE",
        "m3_gates_before": 14,
        "m3_gates_after": 14,
        "child_state": "NOT_RUN",
        "contains_real_secret_material": False,
        "authoritative_root_generation": False,
        "seed_genesis_performed": False,
        "custodian_signature_claim": False,
    }
    if dict(authority) != exact_authority:
        raise ValueError("synthetic authority boundary was weakened or changed")
    vectors = fixture.get("vectors")
    if not isinstance(vectors, list) or not vectors:
        raise ValueError("M2.5 golden fixture must contain vectors")
    names: set[str] = set()
    operations: set[str] = set()
    for index, raw_vector in enumerate(vectors):
        vector = _require_mapping(raw_vector, f"vectors[{index}]")
        name = _require_string(vector.get("name"), f"vectors[{index}].name")
        operation = _require_string(vector.get("op"), f"vectors[{index}].op")
        if name in names:
            raise ValueError(f"duplicate M2.5 golden vector {name!r}")
        if operation not in SUPPORTED_OPERATIONS:
            raise ValueError(f"unsupported M2.5 golden operation {operation!r}")
        _require_mapping(vector.get("input"), f"vectors[{index}].input")
        expected = _require_mapping(
            vector.get("expected"), f"vectors[{index}].expected"
        )
        if tuple(expected) != RESULT_FIELDS[operation]:
            raise ValueError(f"unexpected expected-result fields for {name!r}")
        names.add(name)
        operations.add(operation)
    if operations != SUPPORTED_OPERATIONS:
        raise ValueError("M2.5 golden fixture does not cover every primitive operation")
    return fixture


def _vector_parts(
    raw_vector: object,
) -> tuple[str, str, Mapping[str, object], Mapping[str, object]]:
    vector = _require_mapping(raw_vector, "vector")
    return (
        _require_string(vector.get("name"), "vector.name"),
        _require_string(vector.get("op"), "vector.op"),
        _require_mapping(vector.get("input"), "vector.input"),
        _require_mapping(vector.get("expected"), "vector.expected"),
    )


def _python_actual(raw_vector: object) -> dict[str, object]:
    _, operation, inputs, _ = _vector_parts(raw_vector)
    if operation == "encode":
        value = _formal_from_transport(inputs.get("value"))
        return {"cbor_hex": wire.canonical_cbor_encode(value).hex()}
    if operation == "decode":
        payload = _decode_hex(inputs.get("cbor_hex"), "cbor_hex")
        value = wire.canonical_cbor_decode(payload)
        return {
            "canonical_cbor_hex": wire.canonical_cbor_encode(value).hex(),
            "value": _transport_from_formal(value),
        }
    if operation == "reject_decode":
        payload = _decode_hex(inputs.get("cbor_hex"), "cbor_hex")
        try:
            wire.canonical_cbor_decode(payload)
        except StrictCborError as exc:
            return {"accepted": False, "error_code": exc.code}
        return {"accepted": True, "error_code": None}
    if operation == "content_hash":
        domain = _require_string(inputs.get("domain"), "domain")
        value = _formal_from_transport(inputs.get("value"))
        return {
            "cbor_hex": wire.canonical_cbor_encode(value).hex(),
            "digest_hex": wire.content_hash(domain, value).hex(),
        }
    if operation == "rfc6962_root":
        records = inputs.get("records")
        if not isinstance(records, list):
            raise ValueError("records must be a JSON array")
        formal_records = [_formal_from_transport(record) for record in records]
        return {
            "leaf_count": len(formal_records),
            "root_hex": wire.rfc6962_root(formal_records).hex(),
        }
    if operation == "derive_role_key":
        synthetic_input = _decode_hex(
            inputs.get("synthetic_public_test_input_hex"),
            "synthetic_public_test_input_hex",
        )
        role_id = _require_int(inputs.get("role_id"), "role_id")
        return {"role_key_hex": split.derive_role_key(synthetic_input, role_id).hex()}
    if operation == "row_rank":
        return {
            "rank_hex": split.split_rank(
                _decode_hex(inputs.get("role_key_hex"), "role_key_hex"),
                _require_int(inputs.get("role_id"), "role_id"),
                _require_int(inputs.get("stratum_id"), "stratum_id"),
                _decode_hex(
                    inputs.get("canonical_input_hash_hex"),
                    "canonical_input_hash_hex",
                ),
            ).hex()
        }
    if operation == "seed_commitment":
        synthetic_input = _decode_hex(
            inputs.get("synthetic_public_test_input_hex"),
            "synthetic_public_test_input_hex",
        )
        return {"commitment_hex": split.split_seed_commitment(synthetic_input).hex()}
    raise AssertionError(f"unreachable operation {operation!r}")


def _rust_request(raw_vector: object) -> dict[str, object]:
    _, operation, inputs, _ = _vector_parts(raw_vector)
    request: dict[str, object] = {
        "op": "decode" if operation == "reject_decode" else operation
    }
    if operation == "rfc6962_root":
        records = inputs.get("records")
        if not isinstance(records, list):
            raise ValueError("records must be a JSON array")
        request["leaves_hex"] = [
            wire.canonical_cbor_encode(_formal_from_transport(record)).hex()
            for record in records
        ]
    elif operation in {"derive_role_key", "seed_commitment"}:
        request["master_seed_hex"] = inputs.get("synthetic_public_test_input_hex")
        if operation == "derive_role_key":
            request["role_id"] = inputs.get("role_id")
    else:
        request.update(inputs)
    return request


def _run_rust_request(
    raw_vector: object,
    rust_binary: Path,
) -> dict[str, object]:
    _, operation, _, _ = _vector_parts(raw_vector)
    if not rust_binary.is_file():
        raise FileNotFoundError(f"compiled Rust M2.5 replay missing: {rust_binary}")
    completed = subprocess.run(
        [str(rust_binary)],
        cwd=PROJECT_ROOT,
        input=json.dumps(_rust_request(raw_vector), sort_keys=True) + "\n",
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if operation == "reject_decode":
        response = _require_mapping(json.loads(completed.stdout), "Rust response")
        if completed.returncode != 0:
            if (
                response.get("ok") is not False
                or set(response) != {"ok", "error_code", "error"}
            ):
                raise RuntimeError(
                    "Rust M2.5 reject replay returned an invalid error response: "
                    f"{response!r}"
                )
            return {
                "accepted": False,
                "error_code": _require_string(
                    response.get("error_code"), "Rust error_code"
                ),
            }
        if response.get("ok") is not True or response.get("op") != "decode":
            raise RuntimeError(
                "Rust M2.5 reject replay returned an invalid success response: "
                f"{response!r}"
            )
        return {"accepted": True, "error_code": None}
    if completed.returncode != 0:
        raise RuntimeError(
            "Rust M2.5 replay failed "
            f"({completed.returncode}): stdout={completed.stdout!r}; "
            f"stderr={completed.stderr!r}"
        )
    response = _require_mapping(json.loads(completed.stdout), "Rust response")
    if response.get("ok") is not True or response.get("op") != operation:
        raise RuntimeError(f"Rust M2.5 replay returned an invalid response: {response!r}")
    fields = RESULT_FIELDS[operation]
    if set(response) != {"ok", "op", *fields}:
        raise RuntimeError(f"Rust M2.5 replay returned unexpected fields: {response!r}")
    return {field: response[field] for field in fields}


def _endpoint_report(
    implementation: str,
    vectors: Sequence[object],
    actual_function: object,
) -> dict[str, object]:
    if not callable(actual_function):
        raise TypeError("actual_function must be callable")
    results: list[dict[str, object]] = []
    for raw_vector in vectors:
        name, operation, _, expected = _vector_parts(raw_vector)
        actual = actual_function(raw_vector)
        if not isinstance(actual, dict):
            raise TypeError("endpoint replay must return a result object")
        results.append(
            {
                "name": name,
                "op": operation,
                "expected": dict(expected),
                "actual": actual,
                "expected_match": _json_type_strict_equal(actual, dict(expected)),
            }
        )
    passed = sum(result["expected_match"] is True for result in results)
    return {
        "implementation": implementation,
        "vector_count": len(results),
        "expected_match_count": passed,
        "all_expected_outputs_match": passed == len(results),
        "results": results,
    }


def python_synthetic_replay(
    path: Path = GOLDEN_VECTOR_PATH,
) -> dict[str, object]:
    fixture = load_golden_vectors(path)
    vectors = fixture["vectors"]
    assert isinstance(vectors, list)
    report = _endpoint_report("python", vectors, _python_actual)
    report["source_hashes"] = _source_hashes(PYTHON_SOURCE_FILES)
    report["source_set_sha256"] = _source_set_hash(PYTHON_SOURCE_FILES)
    return report


def rust_synthetic_replay(
    rust_binary: Path = DEFAULT_RUST_BINARY,
    path: Path = GOLDEN_VECTOR_PATH,
) -> dict[str, object]:
    fixture = load_golden_vectors(path)
    vectors = fixture["vectors"]
    assert isinstance(vectors, list)
    report = _endpoint_report(
        "rust",
        vectors,
        lambda vector: _run_rust_request(vector, rust_binary),
    )
    report["source_hashes"] = _source_hashes(RUST_SOURCE_FILES)
    report["source_set_sha256"] = _source_set_hash(RUST_SOURCE_FILES)
    report["binary_sha256"] = _sha256_file(rust_binary)
    report["binary_provenance"] = BINARY_PROVENANCE
    report["binary_source_binding_claim"] = False
    return report


def _actual_index(report: Mapping[str, object]) -> dict[str, object]:
    raw_results = report.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("endpoint report results must be a list")
    indexed: dict[str, object] = {}
    for raw_result in raw_results:
        result = _require_mapping(raw_result, "endpoint result")
        name = _require_string(result.get("name"), "endpoint result name")
        if name in indexed:
            raise ValueError(f"duplicate endpoint result {name!r}")
        indexed[name] = result.get("actual")
    return indexed


def dual_synthetic_replay_report(
    rust_binary: Path = DEFAULT_RUST_BINARY,
    path: Path = GOLDEN_VECTOR_PATH,
) -> dict[str, object]:
    """Replay both implementations while preserving every authority boundary."""

    fixture = load_golden_vectors(path)
    python_report = python_synthetic_replay(path)
    rust_report = rust_synthetic_replay(rust_binary, path)
    python_actual = _actual_index(python_report)
    rust_actual = _actual_index(rust_report)
    mismatches = [
        name
        for name in sorted(set(python_actual) | set(rust_actual))
        if not _json_type_strict_equal(
            python_actual.get(name),
            rust_actual.get(name),
        )
    ]
    dual_equal = not mismatches and set(python_actual) == set(rust_actual)
    expected_match = (
        python_report["all_expected_outputs_match"] is True
        and rust_report["all_expected_outputs_match"] is True
    )
    passed = dual_equal and expected_match
    report: dict[str, object] = {
        "artifact": "phase3_m25_synthetic_dual_replay_v1",
        "schema_version": REPORT_SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "machine_freeze_id": MACHINE_FREEZE_ID,
        "status": (
            "SYNTHETIC_FOUNDATION_DUAL_REPLAY_PASS"
            if passed
            else "SYNTHETIC_FOUNDATION_DUAL_REPLAY_FAIL"
        ),
        "golden_vector_path": _relative(path),
        "golden_vector_sha256": _sha256_file(path),
        "synthetic_material_notice": fixture["synthetic_material_notice"],
        "vector_count": len(python_actual),
        "python": python_report,
        "rust": rust_report,
        "both_endpoints_match_expected": expected_match,
        "cross_language_actual_equal": dual_equal,
        "cross_language_mismatches": mismatches,
        "m3_gates_satisfied": 14,
        "m3_gates_total": 24,
        "m3_gate_delta": 0,
        "child_state": "NOT_RUN",
        "m3_entry_allowed": False,
        "split_seed_first_instantiated": False,
        "custodian_signature_claim": False,
        "formal_input_roots": None,
        "formal_output_roots": None,
        "m3_execution_manifest_root": None,
        "formal_roots_generated": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    report["diagnostic_report_id"] = stable_hash(
        report,
        prefix="phase3_m25_synthetic_dual_replay_",
    )
    return report


def _validate_endpoint_report(
    report: Mapping[str, object],
    *,
    implementation: str,
    vectors: Sequence[object],
    require_current_sources: bool = True,
) -> None:
    """Validate all portable endpoint claims against current sources/fixture."""

    source_files = (
        PYTHON_SOURCE_FILES if implementation == "python" else RUST_SOURCE_FILES
    )
    required_fields = {
        "implementation",
        "vector_count",
        "expected_match_count",
        "all_expected_outputs_match",
        "results",
        "source_hashes",
        "source_set_sha256",
    }
    if implementation == "rust":
        required_fields.update(
            {
                "binary_sha256",
                "binary_provenance",
                "binary_source_binding_claim",
            }
        )
    if set(report) != required_fields:
        raise AssertionError(f"{implementation} endpoint report field-set mismatch")
    if report.get("implementation") != implementation:
        raise AssertionError(f"{implementation} endpoint identity mismatch")
    if report.get("vector_count") != len(vectors):
        raise AssertionError(f"{implementation} endpoint vector count mismatch")
    if report.get("expected_match_count") != len(vectors):
        raise AssertionError(f"{implementation} endpoint expected-match count mismatch")
    if report.get("all_expected_outputs_match") is not True:
        raise AssertionError(f"{implementation} endpoint must match every expected output")
    if require_current_sources:
        if report.get("source_hashes") != _source_hashes(source_files):
            raise AssertionError(
                f"{implementation} endpoint source hashes are stale or forged"
            )
        if report.get("source_set_sha256") != _source_set_hash(source_files):
            raise AssertionError(f"{implementation} endpoint source-set hash mismatch")
    else:
        source_hashes = report.get("source_hashes")
        expected_paths = {_relative(path) for path in source_files}
        if not isinstance(source_hashes, Mapping) or set(source_hashes) != expected_paths:
            raise AssertionError(f"{implementation} historical source paths mismatch")
        digests = tuple(source_hashes.values()) + (report.get("source_set_sha256"),)
        if any(
            not isinstance(digest, str)
            or len(digest) != 71
            or not digest.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in digest[7:])
            for digest in digests
        ):
            raise AssertionError(
                f"{implementation} historical source digest syntax mismatch"
            )

    results = report.get("results")
    if not isinstance(results, list) or len(results) != len(vectors):
        raise AssertionError(f"{implementation} endpoint result list mismatch")
    for index, (raw_vector, raw_result) in enumerate(
        zip(vectors, results, strict=True)
    ):
        name, operation, _, expected = _vector_parts(raw_vector)
        result = _require_mapping(raw_result, f"{implementation}.results[{index}]")
        exact_result = {
            "name": name,
            "op": operation,
            "expected": dict(expected),
            "actual": dict(expected),
            "expected_match": True,
        }
        if not _json_type_strict_equal(dict(result), exact_result):
            raise AssertionError(
                f"{implementation} endpoint result {name!r} is stale or forged"
            )

    if implementation == "rust":
        if report.get("binary_provenance") != BINARY_PROVENANCE:
            raise AssertionError("Rust binary provenance boundary mismatch")
        if report.get("binary_source_binding_claim") is not False:
            raise AssertionError("synthetic replay cannot claim an attested binary build")
        binary_sha256 = report.get("binary_sha256")
        if (
            not isinstance(binary_sha256, str)
            or len(binary_sha256) != len("sha256:") + 64
            or not binary_sha256.startswith("sha256:")
            or any(
                character not in "0123456789abcdef"
                for character in binary_sha256[7:]
            )
        ):
            raise AssertionError("Rust binary digest must be canonical sha256 hexadecimal")


def _validate_dual_synthetic_replay_report(
    report: Mapping[str, object],
    *,
    require_current_sources: bool,
) -> None:
    """Validate exact replay results and a current or pinned source boundary."""

    if not isinstance(report, Mapping):
        raise TypeError("M2.5 dual replay report must be a mapping")
    expected_top_level_fields = {
        "artifact",
        "schema_version",
        "artifact_kind",
        "machine_freeze_id",
        "status",
        "golden_vector_path",
        "golden_vector_sha256",
        "synthetic_material_notice",
        "vector_count",
        "python",
        "rust",
        "both_endpoints_match_expected",
        "cross_language_actual_equal",
        "cross_language_mismatches",
        "m3_gates_satisfied",
        "m3_gates_total",
        "m3_gate_delta",
        "child_state",
        "m3_entry_allowed",
        "split_seed_first_instantiated",
        "custodian_signature_claim",
        "formal_input_roots",
        "formal_output_roots",
        "m3_execution_manifest_root",
        "formal_roots_generated",
        "claim_boundary",
        "diagnostic_report_id",
    }
    if set(report) != expected_top_level_fields:
        raise AssertionError("synthetic dual replay top-level field-set mismatch")
    fixture = load_golden_vectors()
    vectors = fixture["vectors"]
    assert isinstance(vectors, list)
    required_values = {
        "artifact": "phase3_m25_synthetic_dual_replay_v1",
        "schema_version": REPORT_SCHEMA,
        "artifact_kind": ARTIFACT_KIND,
        "machine_freeze_id": MACHINE_FREEZE_ID,
        "status": "SYNTHETIC_FOUNDATION_DUAL_REPLAY_PASS",
        "golden_vector_path": _relative(GOLDEN_VECTOR_PATH),
        "golden_vector_sha256": _sha256_file(GOLDEN_VECTOR_PATH),
        "synthetic_material_notice": fixture["synthetic_material_notice"],
        "vector_count": len(vectors),
        "m3_gates_satisfied": 14,
        "m3_gates_total": 24,
        "m3_gate_delta": 0,
        "child_state": "NOT_RUN",
        "m3_entry_allowed": False,
        "split_seed_first_instantiated": False,
        "custodian_signature_claim": False,
        "formal_input_roots": None,
        "formal_output_roots": None,
        "m3_execution_manifest_root": None,
        "formal_roots_generated": False,
        "both_endpoints_match_expected": True,
        "cross_language_actual_equal": True,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    for field, expected in required_values.items():
        if not _json_type_strict_equal(report.get(field), expected):
            raise AssertionError(
                f"synthetic dual replay boundary requires {field}={expected!r}"
            )
    if report.get("cross_language_mismatches") != []:
        raise AssertionError("synthetic dual replay mismatch list must be empty")
    python = _require_mapping(report.get("python"), "python endpoint report")
    rust = _require_mapping(report.get("rust"), "rust endpoint report")
    _validate_endpoint_report(
        python,
        implementation="python",
        vectors=vectors,
        require_current_sources=require_current_sources,
    )
    _validate_endpoint_report(
        rust,
        implementation="rust",
        vectors=vectors,
        require_current_sources=require_current_sources,
    )
    if not _json_type_strict_equal(_actual_index(python), _actual_index(rust)):
        raise AssertionError("synthetic dual replay endpoint outputs differ")

    provided_id = report.get("diagnostic_report_id")
    body = dict(report)
    body.pop("diagnostic_report_id", None)
    expected_id = stable_hash(
        body,
        prefix="phase3_m25_synthetic_dual_replay_",
    )
    if provided_id != expected_id:
        raise AssertionError("synthetic dual replay diagnostic report ID mismatch")


def validate_dual_synthetic_replay_report(report: Mapping[str, object]) -> None:
    """Validate a replay against the current Python and Rust source bytes."""

    _validate_dual_synthetic_replay_report(
        report,
        require_current_sources=True,
    )


def validate_historical_dual_synthetic_replay_report(
    report: Mapping[str, object],
) -> None:
    """Validate the immutable d772 evidence without calling it current.

    The historical source bytes are no longer present at HEAD. The portable
    replay and non-authority fields are checked, then the exact d772 diagnostic
    report ID is required; this API never claims current source binding.
    """

    _validate_dual_synthetic_replay_report(
        report,
        require_current_sources=False,
    )
    if report.get("diagnostic_report_id") != HISTORICAL_CHECKED_IN_REPORT_ID:
        raise AssertionError(
            "historical synthetic dual replay diagnostic report ID mismatch"
        )


__all__ = [
    "ARTIFACT_KIND",
    "BINARY_PROVENANCE",
    "CHECKED_IN_REPORT_PATH",
    "CLAIM_BOUNDARY",
    "DEFAULT_RUST_BINARY",
    "GOLDEN_VECTOR_PATH",
    "HISTORICAL_CHECKED_IN_COMMIT",
    "HISTORICAL_CHECKED_IN_REPORT_ID",
    "PYTHON_SOURCE_FILES",
    "RUST_CRATE_ROOT",
    "RUST_SOURCE_FILES",
    "SUPPORTED_OPERATIONS",
    "dual_synthetic_replay_report",
    "load_golden_vectors",
    "python_synthetic_replay",
    "rust_synthetic_replay",
    "validate_dual_synthetic_replay_report",
    "validate_historical_dual_synthetic_replay_report",
]
