"""Dual Python/Rust gates for Phase-3 shrink step 1.

The gate verifies sparse registry rejection and surviving syntax stability.
The capacity replay checks only the preregistered 25,872-source constructive
subset. A within-budget result leaves the child closure at ``NOT_RUN`` and can
never be interpreted as ``COMPLETE``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Final, Mapping

from .hashing import stable_hash
from .phase3_shrink1_capacity_v1 import (
    EXPECTED_SHRINK1_SOURCE_COUNT,
    SHRINK1_CAPACITY_GENERATOR_RULE,
    iter_shrink1_capacity_candidate_asts,
)
from .phase3_shrink1_registry_v1 import (
    AST_HASH_DOMAIN,
    AST_SCHEMA_ID,
    CBOR_PROFILE_ID,
    DSL_VERSION,
    FREEZE_VERSION,
    PARENT_DSL_VERSION,
    REMOVED_AGGREGATE_ERROR,
)
from .strict_ast_shrink1_v1 import (
    canonicalize_shrink1_source_ast,
    decode_shrink1_canonical_ast,
)
from .strict_ast_v1 import (
    StrictAstError,
    canonicalize_source_ast as canonicalize_parent_source_ast,
    decode_canonical_ast,
)
from .strict_cbor_v1 import canonical_cbor_decode


GATE_SCHEMA: Final = "hegel-dual-strict-gate-shrink1/1"
CAPACITY_REPLAY_SCHEMA: Final = "hegel-dual-capacity-replay-shrink1/1"
GOLDEN_VECTOR_SCHEMA: Final = "hegel-strict-golden-shrink1-v1"
CAPACITY_SET_DOMAIN: Final = b"HEGEL/STRICT_CAPACITY_SET/V1"
CANONICAL_PROGRAM_BUDGET: Final = 50_000
EXPECTED_SHRINK1_CAPACITY_SET_COMMITMENT: Final = (
    "sha256:653fcb9428684cfed11c3f2345ac95ed98ded6e31564c9eeabf97c57ee71a7e9"
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
GOLDEN_VECTOR_PATH: Final = (
    PROJECT_ROOT / "golden_vectors" / "strict_ast_cbor_shrink1_v1.json"
)
RUST_CRATE_ROOT: Final = PROJECT_ROOT / "rust" / "strict_canonicalizer_shrink1"
PARENT_RUST_CRATE_ROOT: Final = PROJECT_ROOT / "rust" / "strict_canonicalizer"
DEFAULT_RUST_BINARY: Final = (
    RUST_CRATE_ROOT / "target" / "release" / "hegel-strict-canonicalizer-shrink1"
)

PYTHON_STRICT_SOURCES: Final = (
    PROJECT_ROOT / "src" / "hegel_machine" / "strict_cbor_v1.py",
    PROJECT_ROOT / "src" / "hegel_machine" / "strict_ast_v1.py",
    PROJECT_ROOT / "src" / "hegel_machine" / "strict_ast_shrink1_v1.py",
    PROJECT_ROOT / "src" / "hegel_machine" / "phase3_dsl_v1.py",
    PROJECT_ROOT / "src" / "hegel_machine" / "phase3_shrink1_registry_v1.py",
    PROJECT_ROOT / "src" / "hegel_machine" / "phase3_shrink1_capacity_v1.py",
    PROJECT_ROOT / "src" / "hegel_machine" / "hashing.py",
    Path(__file__),
)
PYTHON_CAPACITY_SOURCES: Final = PYTHON_STRICT_SOURCES
RUST_STRICT_SOURCES: Final = (
    PARENT_RUST_CRATE_ROOT / "Cargo.toml",
    PARENT_RUST_CRATE_ROOT / "Cargo.lock",
    PARENT_RUST_CRATE_ROOT / "src" / "lib.rs",
    PARENT_RUST_CRATE_ROOT / "src" / "main.rs",
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
        raise ValueError("shrink-1 golden vectors must be a JSON object")
    required = {
        "schema_version": GOLDEN_VECTOR_SCHEMA,
        "dsl_version": DSL_VERSION,
        "parent_dsl_version": PARENT_DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "ast_schema_id": AST_SCHEMA_ID,
        "cbor_profile_id": CBOR_PROFILE_ID,
        "ast_hash_domain": AST_HASH_DOMAIN,
    }
    for field, expected in required.items():
        if value.get(field) != expected:
            raise ValueError(f"shrink-1 golden-vector {field} drift")
    for group in (
        "source_accept_vectors",
        "source_reject_vectors",
        "formal_accept_vectors",
        "formal_reject_vectors",
    ):
        vectors = value.get(group)
        if not isinstance(vectors, list):
            raise ValueError(f"shrink-1 golden-vector {group} must be a list")
        if any(not isinstance(vector, dict) for vector in vectors):
            raise ValueError(f"shrink-1 golden-vector {group} entries must be objects")
    for vector in value["formal_reject_vectors"]:
        assert isinstance(vector, dict)
        for field in ("generic_cbor_parse_required", "parent_ast_accept_required"):
            if type(vector.get(field)) is not bool:
                raise ValueError(
                    f"shrink-1 formal-reject vector {field} must be boolean"
                )
    return value


def _program_fields(program: object) -> dict[str, object]:
    return {
        "canonical_cbor_hex": program.cbor_bytes.hex(),  # type: ignore[attr-defined]
        "canonical_ast_hash": program.hash_id,  # type: ignore[attr-defined]
        "root_operator_id": program.root_operator_id,  # type: ignore[attr-defined]
        "output_sort": program.metrics.output_sort,  # type: ignore[attr-defined]
        "depth": program.metrics.depth,  # type: ignore[attr-defined]
        "node_count": program.metrics.node_count,  # type: ignore[attr-defined]
    }


def python_shrink1_vector_report(
    path: Path = GOLDEN_VECTOR_PATH,
) -> dict[str, object]:
    vectors = _load_vectors(path)
    results: list[dict[str, object]] = []
    for vector in vectors["source_accept_vectors"]:  # type: ignore[index]
        assert isinstance(vector, dict)
        child = canonicalize_shrink1_source_ast(vector["source_ast"])
        parent = canonicalize_parent_source_ast(vector["source_ast"])
        fields = _program_fields(child)
        expected_fields = {
            key: vector[key]
            for key in (
                "canonical_cbor_hex",
                "canonical_ast_hash",
                "root_operator_id",
                "output_sort",
                "depth",
                "node_count",
            )
        }
        results.append(
            {
                "group": "source_accept_vectors",
                "name": vector["name"],
                "status": "ACCEPTED",
                "expectation_match": fields == expected_fields
                and child.cbor_bytes == parent.cbor_bytes
                and child.hash_id == parent.hash_id,
                "parent_child_syntax_equal": child.cbor_bytes == parent.cbor_bytes,
                **fields,
            }
        )
    for vector in vectors["source_reject_vectors"]:  # type: ignore[index]
        assert isinstance(vector, dict)
        try:
            canonicalize_shrink1_source_ast(vector["source_ast"])
        except StrictAstError as error:
            results.append(
                {
                    "group": "source_reject_vectors",
                    "name": vector["name"],
                    "status": "REJECTED",
                    "error_code": error.code,
                    "expectation_match": error.code == vector["error_code"],
                }
            )
        else:
            results.append(
                {
                    "group": "source_reject_vectors",
                    "name": vector["name"],
                    "status": "ACCEPTED",
                    "expectation_match": False,
                }
            )
    for vector in vectors["formal_accept_vectors"]:  # type: ignore[index]
        assert isinstance(vector, dict)
        payload = bytes.fromhex(vector["canonical_cbor_hex"])
        generic_parse = canonical_cbor_decode(payload) is not None
        parent = decode_canonical_ast(payload)
        child = decode_shrink1_canonical_ast(payload)
        results.append(
            {
                "group": "formal_accept_vectors",
                "name": vector["name"],
                "status": "ACCEPTED",
                "canonical_cbor_hex": child.cbor_bytes.hex(),
                "canonical_ast_hash": child.hash_id,
                "root_operator_id": child.root_operator_id,
                "output_sort": child.metrics.output_sort,
                "depth": child.metrics.depth,
                "node_count": child.metrics.node_count,
                "generic_cbor_parse": generic_parse,
                "parent_ast_accept": True,
                "expectation_match": child.cbor_bytes.hex()
                == vector["canonical_cbor_hex"]
                and child.hash_id == vector["canonical_ast_hash"]
                and child.cbor_bytes == parent.cbor_bytes
                and child.hash_id == parent.hash_id,
            }
        )
    for vector in vectors["formal_reject_vectors"]:  # type: ignore[index]
        assert isinstance(vector, dict)
        payload = bytes.fromhex(vector["canonical_cbor_hex"])
        generic_parse = canonical_cbor_decode(payload) is not None
        try:
            decode_canonical_ast(payload)
        except StrictAstError:
            parent_accept = False
        else:
            parent_accept = True
        try:
            decode_shrink1_canonical_ast(payload)
        except StrictAstError as error:
            matched = (
                error.code == vector["error_code"]
                and generic_parse is bool(vector["generic_cbor_parse_required"])
                and parent_accept is bool(vector["parent_ast_accept_required"])
            )
            results.append(
                {
                    "group": "formal_reject_vectors",
                    "name": vector["name"],
                    "status": "REJECTED",
                    "error_code": error.code,
                    "generic_cbor_parse": generic_parse,
                    "parent_ast_accept": parent_accept,
                    "expectation_match": matched,
                }
            )
        else:
            results.append(
                {
                    "group": "formal_reject_vectors",
                    "name": vector["name"],
                    "status": "ACCEPTED",
                    "expectation_match": False,
                }
            )
    passed = sum(item["expectation_match"] is True for item in results)
    return {
        "implementation": "python",
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "vector_count": len(results),
        "passed_count": passed,
        "failed_count": len(results) - passed,
        "all_expectations_match": passed == len(results),
        "results": results,
    }


def _run_rust_command(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(
            f"Rust shrink-1 replay failed ({completed.returncode}): {completed.stderr}"
        )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("Rust shrink-1 replay did not emit a JSON object")
    return value


def _rust_vector_metadata_matches(report: Mapping[str, object]) -> bool:
    """Require every Rust vector response to identify the frozen child profile."""

    return (
        report.get("dsl_version") == DSL_VERSION
        and report.get("parent_dsl_version") == PARENT_DSL_VERSION
        and report.get("freeze_version") == FREEZE_VERSION
        and report.get("ast_schema_id") == AST_SCHEMA_ID
        and report.get("cbor_profile_id") == CBOR_PROFILE_ID
        and report.get("ast_hash_domain") == AST_HASH_DOMAIN
    )


def rust_shrink1_vector_report(
    rust_binary: Path = DEFAULT_RUST_BINARY,
    path: Path = GOLDEN_VECTOR_PATH,
) -> dict[str, object]:
    if not rust_binary.is_file():
        raise FileNotFoundError(f"compiled Rust shrink-1 replay missing: {rust_binary}")
    vectors = _load_vectors(path)
    results: list[dict[str, object]] = []
    for group in ("source_accept_vectors", "source_reject_vectors"):
        for vector in vectors[group]:  # type: ignore[index]
            assert isinstance(vector, dict)
            report = _run_rust_command(
                [
                    str(rust_binary),
                    "--ast-json",
                    json.dumps(vector["source_ast"], separators=(",", ":")),
                ]
            )
            expected_status = "ACCEPTED" if group == "source_accept_vectors" else "REJECTED"
            matched = (
                report.get("status") == expected_status
                and _rust_vector_metadata_matches(report)
            )
            if expected_status == "ACCEPTED":
                for field in (
                    "canonical_cbor_hex",
                    "canonical_ast_hash",
                    "root_operator_id",
                    "output_sort",
                    "depth",
                    "node_count",
                ):
                    matched = matched and report.get(field) == vector.get(field)
            else:
                matched = matched and report.get("error_code") == vector.get("error_code")
            results.append(
                {
                    "group": group,
                    "name": vector["name"],
                    "expectation_match": matched,
                    **{
                        key: value
                        for key, value in report.items()
                        if key
                        in {
                            "status",
                            "canonical_cbor_hex",
                            "canonical_ast_hash",
                            "root_operator_id",
                            "output_sort",
                            "depth",
                            "node_count",
                            "error_code",
                        }
                    },
                }
            )
    for group in ("formal_accept_vectors", "formal_reject_vectors"):
        for vector in vectors[group]:  # type: ignore[index]
            assert isinstance(vector, dict)
            report = _run_rust_command(
                [str(rust_binary), "--decode-cbor-hex", vector["canonical_cbor_hex"]]
            )
            expected_status = "ACCEPTED" if group == "formal_accept_vectors" else "REJECTED"
            matched = (
                report.get("status") == expected_status
                and _rust_vector_metadata_matches(report)
            )
            if expected_status == "ACCEPTED":
                matched = (
                    matched
                    and report.get("canonical_cbor_hex") == vector["canonical_cbor_hex"]
                    and report.get("canonical_ast_hash") == vector["canonical_ast_hash"]
                    and report.get("generic_cbor_parse") is True
                    and report.get("parent_ast_accept") is True
                )
            else:
                matched = (
                    matched
                    and report.get("error_code") == vector["error_code"]
                    and report.get("generic_cbor_parse")
                    is bool(vector["generic_cbor_parse_required"])
                    and report.get("parent_ast_accept")
                    is bool(vector["parent_ast_accept_required"])
                )
            results.append(
                {
                    "group": group,
                    "name": vector["name"],
                    "expectation_match": matched,
                    **{
                        key: value
                        for key, value in report.items()
                        if key
                        in {
                            "status",
                            "canonical_cbor_hex",
                            "canonical_ast_hash",
                            "root_operator_id",
                            "output_sort",
                            "depth",
                            "node_count",
                            "error_code",
                            "generic_cbor_parse",
                            "parent_ast_accept",
                        }
                    },
                }
            )
    passed = sum(item["expectation_match"] is True for item in results)
    return {
        "implementation": "rust",
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "vector_count": len(results),
        "passed_count": passed,
        "failed_count": len(results) - passed,
        "all_expectations_match": passed == len(results),
        "results": results,
    }


_COMPARABLE_VECTOR_FIELDS: Final = (
    "status",
    "canonical_cbor_hex",
    "canonical_ast_hash",
    "root_operator_id",
    "output_sort",
    "depth",
    "node_count",
    "error_code",
    "generic_cbor_parse",
    "parent_ast_accept",
)


def _vector_agreement(
    python: Mapping[str, object], rust: Mapping[str, object]
) -> tuple[bool, list[str]]:
    def indexed(report: Mapping[str, object]) -> dict[tuple[str, str], Mapping[str, object]]:
        results = report.get("results")
        if not isinstance(results, list):
            raise ValueError("vector results must be a list")
        return {
            (str(item["group"]), str(item["name"])): item
            for item in results
            if isinstance(item, Mapping)
        }

    left, right = indexed(python), indexed(rust)
    mismatches: list[str] = []
    if set(left) != set(right):
        mismatches.append("vector_key_set")
    for key in sorted(set(left) & set(right)):
        for field in _COMPARABLE_VECTOR_FIELDS:
            if left[key].get(field) != right[key].get(field):
                mismatches.append(f"{key[0]}/{key[1]}:{field}")
    return not mismatches, mismatches


def dual_shrink1_strict_gate_report(
    rust_binary: Path = DEFAULT_RUST_BINARY,
) -> dict[str, object]:
    python = python_shrink1_vector_report()
    rust = rust_shrink1_vector_report(rust_binary)
    equal, mismatches = _vector_agreement(python, rust)
    verified = (
        python["all_expectations_match"] is True
        and rust["all_expectations_match"] is True
        and equal
    )
    payload: dict[str, object] = {
        "artifact": "phase3_shrink1_dual_strict_gate_v1",
        "schema_version": GATE_SCHEMA,
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "parent_dsl_version": PARENT_DSL_VERSION,
        "status": "VERIFIED" if verified else "FAILED",
        "golden_vector_path": str(GOLDEN_VECTOR_PATH.relative_to(PROJECT_ROOT)),
        "golden_vector_sha256": _sha256_file(GOLDEN_VECTOR_PATH),
        "python": {
            "source_root": _source_root(PYTHON_STRICT_SOURCES),
            "source_paths": _source_paths(PYTHON_STRICT_SOURCES),
            "vector_count": python["vector_count"],
            "passed_count": python["passed_count"],
            "all_expectations_match": python["all_expectations_match"],
        },
        "rust": {
            "source_root": _source_root(RUST_STRICT_SOURCES),
            "source_paths": _source_paths(RUST_STRICT_SOURCES),
            "binary_sha256": _sha256_file(rust_binary),
            "vector_count": rust["vector_count"],
            "passed_count": rust["passed_count"],
            "all_expectations_match": rust["all_expectations_match"],
        },
        "cross_language_vector_identity_equal": equal,
        "cross_language_mismatches": mismatches,
        "surviving_ast_bytes_stable": verified,
        "surviving_ast_hash_stable": verified,
        "tombstone_rejection_verified": verified,
        "removed_map_error_code": REMOVED_AGGREGATE_ERROR,
        "formal_roots": None,
        "executed_closure_status": "NOT_RUN",
        "complete_closure_enumerated": False,
        "target_synthesis_allowed": False,
        "outside_certificate_issued": False,
        "active_promotion_allowed": False,
        "repository_commit_binding": None,
        "claim_boundary": (
            "This verifies child DSL admission, tombstone rejection, and surviving "
            "syntax identity in two implementations. It does not generate formal "
            "roots, execute closure, issue a certificate, or authorize synthesis."
        ),
    }
    payload["gate_report_id"] = stable_hash(
        payload, prefix="phase3_shrink1_dual_strict_gate_"
    )
    return payload


def _capacity_set_commitment(canonical_bytes: list[bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(CAPACITY_SET_DOMAIN + b"\x00")
    for item in canonical_bytes:
        digest.update(len(item).to_bytes(8, "big"))
        digest.update(item)
    return "sha256:" + digest.hexdigest()


def _is_sha256_id(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    if len(digest) != 64:
        return False
    try:
        bytes.fromhex(digest)
    except ValueError:
        return False
    return True


def _capacity_report_satisfies_frozen_subset(report: Mapping[str, object]) -> bool:
    """Validate one replay without trusting its self-declared status.

    The frozen subset is stronger than merely having a count below the budget:
    it must materialize all 25,872 sources, accept them all without collapse,
    and carry no form of out-of-budget witness.  Missing or ill-typed fields
    fail closed.
    """

    rejection_counts = report.get("rejection_counts")
    return (
        report.get("dsl_version") == DSL_VERSION
        and report.get("freeze_version") == FREEZE_VERSION
        and type(report.get("source_candidate_count")) is int
        and report.get("source_candidate_count") == EXPECTED_SHRINK1_SOURCE_COUNT
        and type(report.get("accepted_source_count")) is int
        and report.get("accepted_source_count") == EXPECTED_SHRINK1_SOURCE_COUNT
        and type(report.get("accepted_unique_count")) is int
        and report.get("accepted_unique_count") == EXPECTED_SHRINK1_SOURCE_COUNT
        and type(report.get("rejected_count")) is int
        and report.get("rejected_count") == 0
        and isinstance(rejection_counts, Mapping)
        and not rejection_counts
        and type(report.get("rewrite_collapsed_count")) is int
        and report.get("rewrite_collapsed_count") == 0
        and _is_sha256_id(report.get("accepted_set_commitment"))
        and report.get("accepted_set_commitment")
        == EXPECTED_SHRINK1_CAPACITY_SET_COMMITMENT
        and type(report.get("canonical_program_budget")) is int
        and report.get("canonical_program_budget") == CANONICAL_PROGRAM_BUDGET
        and report.get("first_out_of_budget_ordinal") is None
        and report.get("first_out_of_budget_cbor_hex") is None
        and report.get("first_out_of_budget_ast_hash") is None
        and report.get("subset_status") == "VERIFIED_WITHIN_BUDGET"
        and report.get("executed_closure_status") == "NOT_RUN"
        and report.get("complete_closure_enumerated") is False
        and report.get("interpreted_as_complete_closure") is False
    )


def _out_of_budget_witnesses(
    python: Mapping[str, object], rust: Mapping[str, object]
) -> dict[str, object] | None:
    fields = (
        "first_out_of_budget_ordinal",
        "first_out_of_budget_cbor_hex",
        "first_out_of_budget_ast_hash",
    )
    if all(python.get(field) is None and rust.get(field) is None for field in fields):
        return None
    return {
        "python": {field: python.get(field) for field in fields},
        "rust": {field: rust.get(field) for field in fields},
    }


def python_shrink1_capacity_replay() -> dict[str, object]:
    accepted: list[bytes] = []
    rejection_counts: dict[str, int] = {}
    source_count = 0
    for source_ast in iter_shrink1_capacity_candidate_asts():
        source_count += 1
        try:
            accepted.append(canonicalize_shrink1_source_ast(source_ast).cbor_bytes)
        except StrictAstError as error:
            rejection_counts[error.code] = rejection_counts.get(error.code, 0) + 1
    canonical_set = sorted(set(accepted))
    out_of_budget = (
        canonical_set[CANONICAL_PROGRAM_BUDGET]
        if len(canonical_set) > CANONICAL_PROGRAM_BUDGET
        else None
    )
    payload: dict[str, object] = {
        "implementation": "python",
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "source_candidate_count": source_count,
        "accepted_source_count": len(accepted),
        "accepted_unique_count": len(canonical_set),
        "rejected_count": sum(rejection_counts.values()),
        "rejection_counts": rejection_counts,
        "rewrite_collapsed_count": len(accepted) - len(canonical_set),
        "accepted_set_commitment": _capacity_set_commitment(canonical_set),
        "canonical_program_budget": CANONICAL_PROGRAM_BUDGET,
        "first_out_of_budget_ordinal": (
            CANONICAL_PROGRAM_BUDGET + 1 if out_of_budget is not None else None
        ),
        "first_out_of_budget_cbor_hex": out_of_budget.hex() if out_of_budget else None,
        "first_out_of_budget_ast_hash": (
            decode_shrink1_canonical_ast(out_of_budget).hash_id
            if out_of_budget is not None
            else None
        ),
        "subset_status": "PENDING_INVARIANT_CHECK",
        "executed_closure_status": "NOT_RUN",
        "complete_closure_enumerated": False,
        "interpreted_as_complete_closure": False,
    }
    payload["subset_status"] = (
        "VERIFIED_WITHIN_BUDGET"
        if _capacity_report_satisfies_frozen_subset(
            {**payload, "subset_status": "VERIFIED_WITHIN_BUDGET"}
        )
        else "INCONCLUSIVE_EXECUTION"
    )
    return payload


def rust_shrink1_capacity_replay(
    rust_binary: Path = DEFAULT_RUST_BINARY,
) -> dict[str, object]:
    if not rust_binary.is_file():
        raise FileNotFoundError(f"compiled Rust shrink-1 replay missing: {rust_binary}")
    return _run_rust_command([str(rust_binary), "--capacity-replay"])


_CAPACITY_AGREEMENT_FIELDS: Final = (
    "dsl_version",
    "freeze_version",
    "source_candidate_count",
    "accepted_source_count",
    "accepted_unique_count",
    "rejected_count",
    "rejection_counts",
    "rewrite_collapsed_count",
    "accepted_set_commitment",
    "canonical_program_budget",
    "first_out_of_budget_ordinal",
    "first_out_of_budget_cbor_hex",
    "first_out_of_budget_ast_hash",
    "subset_status",
    "executed_closure_status",
    "complete_closure_enumerated",
    "interpreted_as_complete_closure",
)


def dual_shrink1_capacity_replay_report(
    rust_binary: Path = DEFAULT_RUST_BINARY,
) -> dict[str, object]:
    gate = dual_shrink1_strict_gate_report(rust_binary)
    if gate["status"] != "VERIFIED":
        raise RuntimeError("shrink-1 subset replay is forbidden until the dual gate passes")
    python = python_shrink1_capacity_replay()
    rust = rust_shrink1_capacity_replay(rust_binary)
    mismatches = [
        field
        for field in _CAPACITY_AGREEMENT_FIELDS
        if python.get(field) != rust.get(field)
    ]
    equal = not mismatches
    source_count_valid = (
        equal
        and python.get("source_candidate_count") == EXPECTED_SHRINK1_SOURCE_COUNT
        and rust.get("source_candidate_count") == EXPECTED_SHRINK1_SOURCE_COUNT
    )
    python_qualified = _capacity_report_satisfies_frozen_subset(python)
    rust_qualified = _capacity_report_satisfies_frozen_subset(rust)
    out_of_budget_witness = _out_of_budget_witnesses(python, rust)
    within_budget = (
        equal
        and source_count_valid
        and python_qualified
        and rust_qualified
        and out_of_budget_witness is None
    )
    payload: dict[str, object] = {
        "artifact": "phase3_shrink1_dual_capacity_replay_v1",
        "schema_version": CAPACITY_REPLAY_SCHEMA,
        "dsl_version": DSL_VERSION,
        "freeze_version": FREEZE_VERSION,
        "prerequisite_gate_report_id": gate["gate_report_id"],
        "status": "VERIFIED_WITHIN_BUDGET" if within_budget else "INCONCLUSIVE_EXECUTION",
        "python": python,
        "rust": rust,
        "dual_replay_equal": equal,
        "cross_language_mismatches": mismatches,
        "source_count_precommitment_satisfied": source_count_valid,
        "accepted_unique_count_le_50000": within_budget,
        "first_out_of_budget_witness": out_of_budget_witness,
        "subset_is_complete_closure": False,
        "complete_closure_enumerated": False,
        "closure_cardinality": None,
        "match_set_count": None,
        "executed_closure_status": "NOT_RUN",
        "child_initial_state": "NOT_RUN",
        "formal_roots": None,
        "formal_archive_roots_generated": False,
        "dsl_too_large_claim_allowed": False,
        "complete_claim_allowed": False,
        "target_synthesis_allowed": False,
        "hidden_sink_formal_verdict_allowed": False,
        "outside_certificate_issued": False,
        "mdl_certificate_issued": False,
        "active_promotion_allowed": False,
        "execution_bindings": {
            "golden_vector_sha256": gate["golden_vector_sha256"],
            "python_capacity_source_root": _source_root(PYTHON_CAPACITY_SOURCES),
            "python_capacity_source_paths": _source_paths(PYTHON_CAPACITY_SOURCES),
            "rust_source_root": gate["rust"]["source_root"],
            "rust_binary_sha256": gate["rust"]["binary_sha256"],
            "capacity_generator_rule": SHRINK1_CAPACITY_GENERATOR_RULE,
            "repository_commit_binding": None,
        },
        "required_next_action": (
            "COMPLETE_TARGET_SPLIT_CUSTODIAN_BINDINGS_THEN_GENERATE_"
            "DUAL_FORMAL_ROOTS_BEFORE_M3"
            if within_budget
            else "INVESTIGATE_SHRINK1_DUAL_REPLAY_INVARIANT_FAILURE"
        ),
        "claim_boundary": (
            "The 25,872-source constructive subset is dual-replay equal and within "
            "the 50,000 budget. It is not the complete grammar closure, does not "
            "establish a closure cardinality, and leaves the child run NOT_RUN."
            if within_budget
            else "The shrink-1 replay did not satisfy every frozen subset invariant. "
            "No within-budget qualification, closure status, or downstream gate is "
            "authorized."
        ),
    }
    payload["capacity_replay_report_id"] = stable_hash(
        payload, prefix="phase3_shrink1_dual_capacity_replay_"
    )
    return payload


__all__ = [
    "DEFAULT_RUST_BINARY",
    "EXPECTED_SHRINK1_CAPACITY_SET_COMMITMENT",
    "GOLDEN_VECTOR_PATH",
    "dual_shrink1_capacity_replay_report",
    "dual_shrink1_strict_gate_report",
    "python_shrink1_capacity_replay",
    "python_shrink1_vector_report",
    "rust_shrink1_capacity_replay",
    "rust_shrink1_vector_report",
]
