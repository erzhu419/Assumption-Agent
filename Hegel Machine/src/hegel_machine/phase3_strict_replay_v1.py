"""Dual strict-acceptance and conditional-capacity replay gates.

The v1.0.2 freeze separates four states that older readiness prose could blur:
specification freeze, implementation verification, closure execution, and
certificate issuance.  This module advances only the first two executable
gates.  Its commitments are diagnostic cross-language comparison digests, not
formal archive roots or signed outside certificates.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Final, Mapping

from .hashing import stable_hash
from .phase3_capacity_witness_v1 import (
    CAPACITY_GENERATOR_RULE,
    EXPECTED_CAPACITY_SOURCE_COUNT,
    iter_capacity_witness_candidate_asts,
)
from .strict_ast_v1 import (
    StrictAstError,
    canonicalize_source_ast,
    decode_canonical_ast,
)
from .strict_cbor_v1 import (
    StrictCborError,
    canonical_cbor_decode,
    canonical_cbor_encode,
    rfc6962_root,
)


FREEZE_VERSION: Final = "hegel-freeze-p2b-p3-v1.0.2"
GOLDEN_GATE_NAME: Final = (
    "DUAL_STRICT_CANONICAL_IMPLEMENTATION_AND_GOLDEN_VECTOR_VERIFICATION"
)
GOLDEN_GATE_SCHEMA: Final = "hegel-dual-strict-gate/1"
CAPACITY_REPLAY_SCHEMA: Final = "hegel-strict-capacity-replay/1"
CAPACITY_SET_DOMAIN: Final = b"HEGEL/STRICT_CAPACITY_SET/V1"
CANONICAL_PROGRAM_BUDGET: Final = 50_000

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
GOLDEN_VECTOR_PATH: Final = (
    PROJECT_ROOT / "golden_vectors" / "strict_ast_cbor_v1.json"
)
RUST_CRATE_ROOT: Final = PROJECT_ROOT / "rust" / "strict_canonicalizer"
DEFAULT_RUST_BINARY: Final = (
    RUST_CRATE_ROOT / "target" / "release" / "hegel-strict-canonicalizer"
)
PYTHON_STRICT_GATE_SOURCES: Final = (
    PROJECT_ROOT / "src" / "hegel_machine" / "strict_cbor_v1.py",
    PROJECT_ROOT / "src" / "hegel_machine" / "strict_ast_v1.py",
    Path(__file__),
)
PYTHON_CAPACITY_SOURCES: Final = (
    PROJECT_ROOT / "src" / "hegel_machine" / "strict_cbor_v1.py",
    PROJECT_ROOT / "src" / "hegel_machine" / "strict_ast_v1.py",
    PROJECT_ROOT / "src" / "hegel_machine" / "phase3_capacity_witness_v1.py",
    PROJECT_ROOT / "src" / "hegel_machine" / "phase3_dsl_v1.py",
    PROJECT_ROOT / "src" / "hegel_machine" / "hashing.py",
    Path(__file__),
)
RUST_STRICT_SOURCES: Final = (
    RUST_CRATE_ROOT / "Cargo.toml",
    RUST_CRATE_ROOT / "Cargo.lock",
    RUST_CRATE_ROOT / "src" / "lib.rs",
    RUST_CRATE_ROOT / "src" / "main.rs",
)


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _source_root(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"HEGEL/SOURCE_SET/V1\x00")
    for path in sorted(paths, key=lambda item: item.relative_to(PROJECT_ROOT).as_posix()):
        relative = path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return "sha256:" + digest.hexdigest()


def _source_paths(paths: tuple[Path, ...]) -> list[str]:
    return sorted(path.relative_to(PROJECT_ROOT).as_posix() for path in paths)


def _load_vectors(path: Path = GOLDEN_VECTOR_PATH) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("strict golden-vector fixture must be a JSON object")
    if value.get("freeze_version") != FREEZE_VERSION:
        raise ValueError("strict golden vectors are not bound to v1.0.2")
    return value


def _result(
    group: str,
    name: str,
    status: str,
    expectation_match: bool,
    **fields: object,
) -> dict[str, object]:
    return {
        "group": group,
        "name": name,
        "status": status,
        "expectation_match": expectation_match,
        **fields,
    }


def python_golden_vector_report(
    path: Path = GOLDEN_VECTOR_PATH,
) -> dict[str, object]:
    """Replay every shared vector using only the Python implementation."""

    vectors = _load_vectors(path)
    results: list[dict[str, object]] = []

    for vector in vectors["cbor_encode_vectors"]:  # type: ignore[index]
        assert isinstance(vector, dict)
        value = (
            bytes.fromhex(vector["byte_string_hex"])
            if "byte_string_hex" in vector
            else vector["value"]
        )
        encoded = canonical_cbor_encode(value)
        round_trip = canonical_cbor_decode(encoded)
        expected_value = tuple(value) if isinstance(value, list) else value
        matched = (
            encoded.hex() == vector["expected_cbor_hex"]
            and round_trip == expected_value
        )
        results.append(
            _result(
                "cbor_encode_vectors",
                str(vector["name"]),
                "ACCEPTED",
                matched,
                canonical_cbor_hex=encoded.hex(),
            )
        )

    for vector in vectors["cbor_reject_vectors"]:  # type: ignore[index]
        assert isinstance(vector, dict)
        try:
            canonical_cbor_decode(bytes.fromhex(vector["encoded_hex"]))
        except StrictCborError as error:
            results.append(
                _result(
                    "cbor_reject_vectors",
                    str(vector["name"]),
                    "REJECTED",
                    error.code == vector["error_code"],
                    error_code=error.code,
                )
            )
        else:
            results.append(
                _result(
                    "cbor_reject_vectors",
                    str(vector["name"]),
                    "ACCEPTED",
                    False,
                )
            )

    for vector in vectors["ast_accept_vectors"]:  # type: ignore[index]
        assert isinstance(vector, dict)
        accepted = canonicalize_source_ast(vector["source_ast"])
        fields = {
            "canonical_cbor_hex": accepted.cbor_bytes.hex(),
            "canonical_ast_hash": accepted.hash_id,
            "root_operator_id": accepted.root_operator_id,
            "output_sort": accepted.metrics.output_sort,
            "depth": accepted.metrics.depth,
            "node_count": accepted.metrics.node_count,
        }
        matched = all(fields[key] == vector[key] for key in fields)
        results.append(
            _result(
                "ast_accept_vectors",
                str(vector["name"]),
                "ACCEPTED",
                matched,
                **fields,
            )
        )

    for vector in vectors["ast_reject_vectors"]:  # type: ignore[index]
        assert isinstance(vector, dict)
        try:
            canonicalize_source_ast(vector["source_ast"])
        except StrictAstError as error:
            results.append(
                _result(
                    "ast_reject_vectors",
                    str(vector["name"]),
                    "REJECTED",
                    error.code == vector["error_code"],
                    error_code=error.code,
                )
            )
        else:
            results.append(
                _result(
                    "ast_reject_vectors",
                    str(vector["name"]),
                    "ACCEPTED",
                    False,
                )
            )

    for vector in vectors["rfc6962_vectors"]:  # type: ignore[index]
        assert isinstance(vector, dict)
        records = [(1, index) for index in range(int(vector["leaf_count"]))]
        root_hex = rfc6962_root(records).hex()
        expected = str(vector["expected_root"]).removeprefix("sha256:")
        results.append(
            _result(
                "rfc6962_vectors",
                str(vector["name"]),
                "ACCEPTED",
                root_hex == expected,
                leaf_count=vector["leaf_count"],
                root="sha256:" + root_hex,
                root_hex=root_hex,
            )
        )

    passed = sum(bool(item["expectation_match"]) for item in results)
    accepted = sum(item["status"] == "ACCEPTED" for item in results)
    return {
        "schema_version": "hegel-strict-canonicalizer-replay/1",
        "implementation": "python",
        "freeze_version": FREEZE_VERSION,
        "cbor_profile_id": vectors["cbor_profile_id"],
        "ast_schema_id": vectors["ast_schema_id"],
        "source_path": str(path.relative_to(PROJECT_ROOT)),
        "vector_count": len(results),
        "passed_count": passed,
        "failed_count": len(results) - passed,
        "accepted_result_count": accepted,
        "rejected_result_count": len(results) - accepted,
        "all_expectations_match": passed == len(results),
        "metadata_errors": [],
        "results": results,
    }


def _run_json_command(command: list[str]) -> tuple[dict[str, object], str]:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Rust strict replay failed ({completed.returncode}): {completed.stderr}"
        )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("Rust strict replay did not emit a JSON object")
    return value, completed.stderr


def rust_golden_vector_report(
    rust_binary: Path = DEFAULT_RUST_BINARY,
    path: Path = GOLDEN_VECTOR_PATH,
) -> dict[str, object]:
    if not rust_binary.is_file():
        raise FileNotFoundError(f"compiled Rust strict replay missing: {rust_binary}")
    report, _ = _run_json_command(
        [str(rust_binary), "--vectors", str(path)]
    )
    return report


_COMPARABLE_FIELDS: Final = (
    "status",
    "canonical_cbor_hex",
    "canonical_ast_hash",
    "root_operator_id",
    "output_sort",
    "depth",
    "node_count",
    "error_code",
    "leaf_count",
    "root_hex",
)


def _golden_identity_agreement(
    python_report: Mapping[str, object], rust_report: Mapping[str, object]
) -> tuple[bool, list[str]]:
    def index(report: Mapping[str, object]) -> dict[tuple[str, str], Mapping[str, object]]:
        raw = report.get("results")
        if not isinstance(raw, list):
            raise ValueError("golden replay results must be a list")
        indexed: dict[tuple[str, str], Mapping[str, object]] = {}
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError("golden replay result must be an object")
            key = (str(item.get("group")), str(item.get("name")))
            if key in indexed:
                raise ValueError(f"duplicate golden result {key}")
            indexed[key] = item
        return indexed

    python_index = index(python_report)
    rust_index = index(rust_report)
    mismatches: list[str] = []
    if set(python_index) != set(rust_index):
        mismatches.append("result_key_set")
    for key in sorted(set(python_index) & set(rust_index)):
        left, right = python_index[key], rust_index[key]
        for field in _COMPARABLE_FIELDS:
            if left.get(field) != right.get(field):
                mismatches.append(f"{key[0]}/{key[1]}:{field}")
    return not mismatches, mismatches


def dual_strict_gate_report(
    rust_binary: Path = DEFAULT_RUST_BINARY,
) -> dict[str, object]:
    python_report = python_golden_vector_report()
    rust_report = rust_golden_vector_report(rust_binary)
    identity_agreement, mismatches = _golden_identity_agreement(
        python_report,
        rust_report,
    )
    verified = (
        python_report["all_expectations_match"] is True
        and rust_report.get("all_expectations_match") is True
        and identity_agreement
    )
    payload: dict[str, object] = {
        "artifact": "phase3_dual_strict_gate_v1",
        "schema_version": GOLDEN_GATE_SCHEMA,
        "freeze_version": FREEZE_VERSION,
        "gate_name": GOLDEN_GATE_NAME,
        "status": "VERIFIED" if verified else "FAILED",
        "strict_acceptance_specification_complete": True,
        "strict_acceptance_implementation_verified": verified,
        "golden_vector_path": str(GOLDEN_VECTOR_PATH.relative_to(PROJECT_ROOT)),
        "golden_vector_sha256": _sha256_file(GOLDEN_VECTOR_PATH),
        "python": {
            "source_root": _source_root(PYTHON_STRICT_GATE_SOURCES),
            "source_paths": _source_paths(PYTHON_STRICT_GATE_SOURCES),
            "vector_count": python_report["vector_count"],
            "passed_count": python_report["passed_count"],
            "all_expectations_match": python_report["all_expectations_match"],
        },
        "rust": {
            "source_root": _source_root(RUST_STRICT_SOURCES),
            "source_paths": _source_paths(RUST_STRICT_SOURCES),
            "binary_sha256": _sha256_file(rust_binary),
            "vector_count": rust_report.get("vector_count"),
            "passed_count": rust_report.get("passed_count"),
            "all_expectations_match": rust_report.get("all_expectations_match"),
        },
        "cross_language_vector_identity_equal": identity_agreement,
        "cross_language_mismatches": mismatches,
        "formal_root_generation_allowed": False,
        "executed_closure_status": "NOT_RUN",
        "outside_certificate_issued": False,
        "active_promotion_allowed": False,
        "repository_commit_binding": None,
        "claim_boundary": (
            "This verifies two strict acceptance implementations against the "
            "shared v1.0.2 vectors. It does not generate formal archive roots, "
            "execute complete closure, issue a certificate, or authorize ACTIVE."
        ),
    }
    payload["gate_report_id"] = stable_hash(
        payload,
        prefix="phase3_dual_strict_gate_",
    )
    return payload


def _capacity_set_commitment(canonical_bytes: list[bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(CAPACITY_SET_DOMAIN + b"\x00")
    for item in canonical_bytes:
        digest.update(len(item).to_bytes(8, "big"))
        digest.update(item)
    return "sha256:" + digest.hexdigest()


def python_capacity_replay() -> dict[str, object]:
    accepted: list[bytes] = []
    rejection_counts: dict[str, int] = {}
    source_count = 0
    for source_ast in iter_capacity_witness_candidate_asts():
        source_count += 1
        try:
            accepted.append(canonicalize_source_ast(source_ast).cbor_bytes)
        except StrictAstError as error:
            rejection_counts[error.code] = rejection_counts.get(error.code, 0) + 1
    canonical_set = sorted(set(accepted))
    out_of_budget = (
        canonical_set[CANONICAL_PROGRAM_BUDGET]
        if len(canonical_set) > CANONICAL_PROGRAM_BUDGET
        else None
    )
    return {
        "implementation": "python",
        "source_candidate_count": source_count,
        "accepted_source_count": len(accepted),
        "accepted_unique_count": len(canonical_set),
        "rejected_count": sum(rejection_counts.values()),
        "rejection_counts": rejection_counts,
        "rewrite_collapsed_count": len(accepted) - len(canonical_set),
        "accepted_set_commitment": _capacity_set_commitment(canonical_set),
        "first_out_of_budget_ordinal": (
            CANONICAL_PROGRAM_BUDGET + 1 if out_of_budget is not None else None
        ),
        "first_out_of_budget_cbor_hex": (
            out_of_budget.hex() if out_of_budget is not None else None
        ),
        "first_out_of_budget_ast_hash": (
            decode_canonical_ast(out_of_budget).hash_id
            if out_of_budget is not None
            else None
        ),
    }


def rust_capacity_replay(
    rust_binary: Path = DEFAULT_RUST_BINARY,
) -> dict[str, object]:
    if not rust_binary.is_file():
        raise FileNotFoundError(f"compiled Rust strict replay missing: {rust_binary}")
    report, _ = _run_json_command([str(rust_binary), "--capacity-replay"])
    aliases = {
        "source_candidate_count": "source_count",
        "accepted_source_count": "accepted_total_count",
        "first_out_of_budget_ordinal": "first_accepted_out_of_budget_ordinal",
        "first_out_of_budget_cbor_hex": "first_accepted_out_of_budget_cbor_hex",
        "first_out_of_budget_ast_hash": "first_accepted_out_of_budget_hash",
    }
    for canonical_name, legacy_name in aliases.items():
        if canonical_name not in report and legacy_name in report:
            report[canonical_name] = report[legacy_name]
    return report


_CAPACITY_AGREEMENT_FIELDS: Final = (
    "source_candidate_count",
    "accepted_source_count",
    "accepted_unique_count",
    "rejected_count",
    "rewrite_collapsed_count",
    "accepted_set_commitment",
    "first_out_of_budget_ordinal",
    "first_out_of_budget_cbor_hex",
    "first_out_of_budget_ast_hash",
)


def dual_capacity_replay_report(
    rust_binary: Path = DEFAULT_RUST_BINARY,
) -> dict[str, object]:
    gate = dual_strict_gate_report(rust_binary)
    if gate["status"] != "VERIFIED":
        raise RuntimeError("capacity replay is forbidden until the dual strict gate passes")
    python = python_capacity_replay()
    rust = rust_capacity_replay(rust_binary)
    mismatches = [
        field
        for field in _CAPACITY_AGREEMENT_FIELDS
        if python.get(field) != rust.get(field)
    ]
    equal = not mismatches
    accepted_count = int(python["accepted_unique_count"])
    dsl_too_large = (
        equal
        and accepted_count >= CANONICAL_PROGRAM_BUDGET + 1
        and int(python["source_candidate_count"]) == EXPECTED_CAPACITY_SOURCE_COUNT
    )
    payload: dict[str, object] = {
        "artifact": "phase3_dual_strict_capacity_replay_v1",
        "schema_version": CAPACITY_REPLAY_SCHEMA,
        "freeze_version": FREEZE_VERSION,
        "prerequisite_gate_report_id": gate["gate_report_id"],
        "strict_acceptance_implementation_verified": True,
        "canonical_program_budget": CANONICAL_PROGRAM_BUDGET,
        "python": python,
        "rust": rust,
        "dual_replay_equal": equal,
        "cross_language_mismatches": mismatches,
        "execution_bindings": {
            "golden_vector_sha256": gate["golden_vector_sha256"],
            "python_capacity_source_root": _source_root(PYTHON_CAPACITY_SOURCES),
            "python_capacity_source_paths": _source_paths(
                PYTHON_CAPACITY_SOURCES
            ),
            "capacity_generator_rule": CAPACITY_GENERATOR_RULE,
            "rust_source_root": gate["rust"]["source_root"],
            "rust_binary_sha256": gate["rust"]["binary_sha256"],
            "repository_commit_binding": None,
        },
        "executed_closure_status": (
            "DSL_TOO_LARGE" if dsl_too_large else "INCONCLUSIVE_EXECUTION"
        ),
        "dsl_too_large_claim_allowed": dsl_too_large,
        "complete_closure_enumerated": False,
        "extensional_quotient_computed": False,
        "formal_archive_roots_generated": False,
        "outside_certificate_issued": False,
        "target_synthesis_allowed": False,
        "hidden_sink_formal_verdict_allowed": False,
        "mdl_certificate_allowed": False,
        "active_promotion_allowed": False,
        "required_next_action": (
            "PUBLISH_SHRUNK_OLD_DSL_VERSION_USING_FROZEN_STEP_1"
            if dsl_too_large
            else "INVESTIGATE_DUAL_REPLAY_MISMATCH"
        ),
        "claim_boundary": (
            "DSL_TOO_LARGE is bounded to the v1.0.2 strict old DSL and the "
            "50,000 syntactic canonical-program budget. This is not COMPLETE, "
            "an outside-language result, an extensional target verdict, or a "
            "signed certificate. The set commitment is diagnostic, not a "
            "formal RFC6962 archive root."
        ),
    }
    payload["capacity_replay_report_id"] = stable_hash(
        payload,
        prefix="phase3_dual_strict_capacity_replay_",
    )
    return payload


__all__ = [
    "CANONICAL_PROGRAM_BUDGET",
    "CAPACITY_REPLAY_SCHEMA",
    "DEFAULT_RUST_BINARY",
    "EXPECTED_CAPACITY_SOURCE_COUNT",
    "FREEZE_VERSION",
    "GOLDEN_GATE_NAME",
    "GOLDEN_GATE_SCHEMA",
    "dual_capacity_replay_report",
    "dual_strict_gate_report",
    "python_capacity_replay",
    "python_golden_vector_report",
    "rust_capacity_replay",
    "rust_golden_vector_report",
]
