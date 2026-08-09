from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools/phase3_shrink4_dual_strict_qualification_v1.py"
SPEC = importlib.util.spec_from_file_location("shrink4_dual_strict_tool", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def _normalized_acceptance(implementation: str) -> dict[str, object]:
    cbor = bytes.fromhex("820183000001")
    digest = sha256(b"HEGEL/AST/V1\x00" + cbor).hexdigest()
    return {
        "schema_version": "hegel-strict-canonicalizer-shrink4-replay/1",
        "implementation": implementation,
        "dsl_version": "hegel-old-dsl-v1.4.0",
        "freeze_version": "hegel-freeze-p2b-p3-v1.4.0",
        "status": "ACCEPTED",
        "canonical_cbor_hex": cbor.hex(),
        "canonical_ast_hash": "sha256:" + digest,
        "root_operator_id": 0,
        "output_sort": "RationalValue",
        "depth": 0,
        "node_count": 1,
        "maximum_top_level_clauses": 2,
    }


def _normalized_rejection(implementation: str) -> dict[str, object]:
    return {
        "schema_version": "hegel-strict-canonicalizer-shrink4-replay/1",
        "implementation": implementation,
        "dsl_version": "hegel-old-dsl-v1.4.0",
        "freeze_version": "hegel-freeze-p2b-p3-v1.4.0",
        "status": "REJECTED",
        "error_code": "REJECT_REMOVED_BINARY_OPERATOR",
        "maximum_top_level_clauses": 2,
    }


def _python_strict(status: str, *, boundary: str = "SOURCE_JSON") -> dict[str, object]:
    report = _normalized_acceptance("python") if status == "ACCEPTED" else _normalized_rejection("python")
    report.update(
        {
            "boundary": boundary,
            "loaded_hegel_modules": tool.EXPECTED_PYTHON_STRICT_MODULES,
            "target_or_split_modules_loaded": False,
        }
    )
    if status == "REJECTED":
        report["error_detail"] = "non-normative"
    return report


def _rust_strict(status: str, *, boundary: str = "SOURCE_JSON") -> dict[str, object]:
    report = _normalized_acceptance("rust") if status == "ACCEPTED" else _normalized_rejection("rust")
    report.update(
        {
            "parent_dsl_version": "hegel-old-dsl-v1.3.0",
            "parent_freeze_version": "hegel-freeze-p2b-p3-v1.3.0",
            "boundary": boundary,
            "cbor_profile_id": "hegel-cbor-det-v1",
            "ast_schema_id": "hegel-canonical-ast-v1",
            "ast_hash_domain": "HEGEL/AST/V1",
            "target_or_split_modules_loaded": False,
        }
    )
    if status == "ACCEPTED":
        report["scalar_parameter_occurrence_count"] = 1
    else:
        report["error_message"] = "non-normative"
    if boundary == "FORMAL_CBOR":
        report["generic_cbor_parse"] = True
    return report


def _capacity_common() -> dict[str, object]:
    return {
        "schema_version": "hegel-strict-capacity-replay-shrink4/1",
        "parent_dsl_version": "hegel-old-dsl-v1.3.0",
        "parent_freeze_version": "hegel-freeze-p2b-p3-v1.3.0",
        "dsl_version": "hegel-old-dsl-v1.4.0",
        "freeze_version": "hegel-freeze-p2b-p3-v1.4.0",
        "human_amendment_id": "hegel-freeze-p2b-p3-v1.4.0-shrink-step4",
        "shrink_step_id": "SHRINK_STEP_4_REDUCE_MAX_TOP_LEVEL_CLAUSES_3_TO_2",
        "generator_rule": "exact inherited AND2 survivor set",
        "removed_binary_operator_ids": [0],
        "retained_difference_id": 1,
        "maximum_top_level_clauses": 2,
        "constant_atom_count": 15,
        "rational_aggregate_count": 16,
        "mixed_atom_count": 144,
        "source_candidate_count": 2160,
        "normalized_and2_count": 2160,
        "accepted_source_count": 2160,
        "accepted_unique_count": 2160,
        "parent_identity_match_count": 2160,
        "rejected_count": 0,
        "rejection_counts": {},
        "rewrite_collapsed_count": 0,
        "accepted_set_commitment": tool.EXPECTED_CAPACITY_COMMITMENT,
        "first_canonical_cbor_hex": "00",
        "first_canonical_ast_hash": "sha256:" + "00" * 32,
        "last_canonical_cbor_hex": "01",
        "last_canonical_ast_hash": "sha256:" + "11" * 32,
        "canonical_program_budget": 50_000,
        "first_out_of_budget_ordinal": None,
        "subset_status": "FULL_AND2_SURVIVOR_SET_ONLY_NOT_COMPLETE",
        "executed_closure_status": "NOT_RUN",
        "complete_closure_enumerated": False,
        "interpreted_as_complete_closure": False,
        "formal_roots": None,
        "target_or_split_modules_loaded": False,
    }


def test_normalized_acceptance_is_cross_implementation_exact() -> None:
    python_outcome, python_fields = tool.normalize_endpoint_report(
        _normalized_acceptance("python"), implementation="python"
    )
    rust_outcome, rust_fields = tool.normalize_endpoint_report(
        _normalized_acceptance("rust"), implementation="rust"
    )
    assert python_outcome == rust_outcome
    assert python_fields == rust_fields
    assert python_fields["maximum_top_level_clauses"] == 2


def test_normalized_rejection_ignores_non_normative_details() -> None:
    python = _normalized_rejection("python")
    rust = _normalized_rejection("rust")
    python["error_detail"] = "Python wording"
    rust["error_message"] = "Rust wording"
    assert tool.normalize_endpoint_report(
        python, implementation="python"
    ) == tool.normalize_endpoint_report(rust, implementation="rust")


def test_normalized_acceptance_recomputes_ast_hash() -> None:
    report = _normalized_acceptance("python")
    report["canonical_ast_hash"] = "sha256:" + "00" * 32
    with pytest.raises(tool.QualificationError) as raised:
        tool.normalize_endpoint_report(report, implementation="python")
    assert raised.value.code == tool.FAIL_VECTOR


@pytest.mark.parametrize("status", ["ACCEPTED", "REJECTED"])
@pytest.mark.parametrize("boundary", ["SOURCE_JSON", "FORMAL_CBOR"])
def test_strict_report_schemas_are_exact(status: str, boundary: str) -> None:
    tool._strict_report_guard(
        _python_strict(status, boundary=boundary),
        implementation="python",
        boundary=boundary,
    )
    tool._strict_report_guard(
        _rust_strict(status, boundary=boundary),
        implementation="rust",
        boundary=boundary,
    )


def test_strict_report_rejects_missing_maximum() -> None:
    report = _python_strict("REJECTED")
    report.pop("maximum_top_level_clauses")
    with pytest.raises(tool.QualificationError) as raised:
        tool._strict_report_guard(
            report, implementation="python", boundary="SOURCE_JSON"
        )
    assert raised.value.code == tool.FAIL_VECTOR


def test_capacity_comparison_is_exact_and_nonterminal() -> None:
    common = _capacity_common()
    python = {
        "implementation": "python",
        "loaded_hegel_modules": tool.EXPECTED_PYTHON_CAPACITY_MODULES,
        **common,
    }
    rust = {"implementation": "rust", **common}
    report = tool.compare_capacity_reports(python, rust)
    assert report["all_comparable_fields_equal"] is True
    assert report["normalized_and2_count"] == 2160
    assert report["executed_closure_status"] == "NOT_RUN"
    assert report["formal_roots"] is None


def test_capacity_comparison_fails_on_any_shared_field_drift() -> None:
    common = _capacity_common()
    python = {
        "implementation": "python",
        "loaded_hegel_modules": tool.EXPECTED_PYTHON_CAPACITY_MODULES,
        **common,
    }
    rust = {"implementation": "rust", **common, "accepted_unique_count": 2159}
    with pytest.raises(tool.QualificationError) as raised:
        tool.compare_capacity_reports(python, rust)
    assert raised.value.code == tool.FAIL_CAPACITY


@pytest.mark.parametrize(
    "command",
    [
        tool.python_runtime_command(Path("/snapshot"), ("--source-json", "[]")),
        tool.rust_runtime_command(
            Path("/snapshot"), "fresh-volume", ("--ast-json", "[]")
        ),
    ],
)
def test_recognizer_commands_are_pinned_offline_and_read_only(
    command: list[str],
) -> None:
    assert command[:3] == [
        "/usr/bin/docker", "--host=unix:///var/run/docker.sock", "run"
    ]
    assert "--pull=never" in command
    assert "--network=none" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--read-only" in command
    assert "--pids-limit=64" in command
    assert "--memory=512m" in command
    assert "--memory-swap=512m" in command
    assert "--ulimit=nofile=128:128" in command
    assert tool.RUNTIME_TMPFS in command
    assert any(argument.startswith("seccomp=/snapshot/config/") for argument in command)
    assert "65534:65534" in command


def test_sealed_roots_manifest_and_order_are_literal_pinned() -> None:
    assert tool.diagnostic_root_hex_v1() == tool.EXPECTED_DIAGNOSTIC_ROOTS
    assert tool.strict_golden_manifest_root_v1() == tool.EXPECTED_MANIFEST_ROOT
    assert tuple(vector.vector_id for vector in tool.STRICT_GOLDEN_VECTORS_V1) == (
        "S01", "S02", "S03", "N01", "N02", "L01", "L02",
        "P01", "P02", "P03", "P04", "P05",
        "F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10",
    )
    assert len(tool.STRICT_GOLDEN_VECTORS_V1) == 22


def test_report_field_cardinalities_match_built_in_contract() -> None:
    assert len(tool.RUST_CAPACITY_FIELDS) == 36
    assert len(tool.PYTHON_CAPACITY_FIELDS) == 37
    assert len(tool.RUST_GOLDEN_FIELDS) == 32
    assert len(tool.PYTHON_GOLDEN_FIELDS) == 33


def test_rust_build_profile_is_cache_index_only() -> None:
    profile = json.loads(
        (PROJECT_ROOT / "config/phase3_shrink4_offline_build_profile_v1.json").read_text()
    )
    assert profile["cargo_registry_mount"] == (
        "read-only cache and index subtrees only; no pre-unpacked src"
    )
    assert profile["cargo_home"] == "fresh build-container tmpfs"
    assert profile["network"] == "none"
    assert profile["pull_policy"] == "never"
    assert profile["cargo_seed_manifest"] == (
        "exact regular-file rows with mode, size and sha256 under a "
        "domain-separated root; checked before and after build"
    )


@pytest.mark.parametrize(
    "fields",
    [
        tool.REPOSITORY_BINDING_FIELDS,
        tool.DUAL_VECTOR_REPLAY_FIELDS,
        tool.VECTOR_RECEIPT_FIELDS,
        tool.DUAL_CAPACITY_REPLAY_FIELDS,
        tool.RUNTIME_ISOLATION_FIELDS,
        tool.DOCKER_DAEMON_RECEIPT_FIELDS,
        tool.DOCKER_SERVER_RECEIPT_FIELDS,
        tool.DOCKER_INFO_RECEIPT_FIELDS,
        tool.TARGET_VOLUME_RECEIPT_FIELDS,
        tool.PYTHON_RUNTIME_RECEIPT_FIELDS,
        tool.RUST_RUNTIME_RECEIPT_FIELDS,
    ],
)
def test_final_nested_object_schemas_reject_extra_and_missing_keys(
    fields: frozenset[str],
) -> None:
    candidate = {field: None for field in fields}
    assert tool._exact_mapping(candidate, "test receipt", fields) == candidate
    with_extra = {**candidate, "observed_closure_cardinality": 1}
    with pytest.raises(tool.QualificationError) as extra:
        tool._exact_mapping(with_extra, "test receipt", fields)
    assert extra.value.code == tool.FAIL_GUARD
    with_missing = dict(candidate)
    with_missing.pop(next(iter(fields)))
    with pytest.raises(tool.QualificationError) as missing:
        tool._exact_mapping(with_missing, "test receipt", fields)
    assert missing.value.code == tool.FAIL_GUARD


def test_capacity_receipt_rejects_future_enumeration_value() -> None:
    common = _capacity_common()
    python = {
        "implementation": "python",
        "loaded_hegel_modules": tool.EXPECTED_PYTHON_CAPACITY_MODULES,
        **common,
    }
    rust = {"implementation": "rust", **common}
    capacity = tool.compare_capacity_reports(python, rust)
    tool._validate_dual_capacity_receipt(capacity)
    capacity["observed_closure_cardinality"] = 1
    with pytest.raises(tool.QualificationError) as raised:
        tool._validate_dual_capacity_receipt(capacity)
    assert raised.value.code == tool.FAIL_GUARD


def test_source_file_rows_reject_extra_keys() -> None:
    row = {
        "path": "Hegel Machine/example.py",
        "mode": "100644",
        "git_blob_oid": "00" * 20,
        "sha256": "11" * 32,
        "size": 0,
    }
    assert tool.source_file_set_root_v1([row]).startswith("sha256:")
    row["observed_closure_cardinality"] = 1
    with pytest.raises(tool.QualificationError):
        tool.source_file_set_root_v1([row])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", 7),
        ("path", "."),
        ("path", "../Hegel Machine/example.py"),
        ("path", "outside/example.py"),
        ("mode", 100644),
        ("mode", "120000"),
        ("git_blob_oid", 0),
        ("sha256", b"1" * 64),
        ("size", "0"),
        ("size", True),
    ],
)
def test_source_file_rows_reject_noncanonical_types_and_values(
    field: str, value: object
) -> None:
    row: dict[str, object] = {
        "path": "Hegel Machine/example.py",
        "mode": "100644",
        "git_blob_oid": "00" * 20,
        "sha256": "11" * 32,
        "size": 0,
    }
    row[field] = value
    with pytest.raises(tool.QualificationError):
        tool.source_file_set_root_v1([row])


def test_cargo_seed_manifest_is_exact_and_detects_drift(tmp_path: Path) -> None:
    cache = tmp_path / "cache" / "registry"
    index = tmp_path / "index" / "registry" / ".cache"
    cache.mkdir(parents=True)
    index.mkdir(parents=True)
    crate = cache / "demo-1.0.0.crate"
    metadata = index / "demo"
    crate.write_bytes(b"crate-bytes")
    metadata.write_bytes(b"index-bytes")

    receipt = tool._cargo_seed_receipt(tmp_path)
    assert receipt["file_count"] == 2
    assert receipt["subtrees"] == ["cache", "index"]
    assert receipt["manifest_root"].startswith("sha256:")
    assert tool._validate_cargo_seed_receipt(receipt) == receipt
    tool._assert_cargo_seed_unchanged(tmp_path, receipt)

    extra_receipt = {**receipt, "unexpected": False}
    with pytest.raises(tool.QualificationError) as extra:
        tool._validate_cargo_seed_receipt(extra_receipt)
    assert extra.value.code == tool.FAIL_GUARD

    files = [dict(row) for row in receipt["files"]]
    files[0]["observed_closure_cardinality"] = 1
    extra_row = {**receipt, "files": files}
    with pytest.raises(tool.QualificationError) as row_extra:
        tool._validate_cargo_seed_receipt(extra_row)
    assert row_extra.value.code == tool.FAIL_GUARD

    crate.write_bytes(b"crate-bytes-drift")
    with pytest.raises(tool.QualificationError) as drift:
        tool._assert_cargo_seed_unchanged(tmp_path, receipt)
    assert drift.value.code == tool.FAIL_BUILD
