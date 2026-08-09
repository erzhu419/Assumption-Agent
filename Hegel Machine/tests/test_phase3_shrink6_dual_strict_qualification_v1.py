from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools/phase3_shrink6_dual_strict_qualification_v1.py"
SPEC = importlib.util.spec_from_file_location("shrink6_dual_strict_tool", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def test_tool_imports_with_all_commitments_sealed() -> None:
    tool._require_sealed_commitments()


SOURCE_W_TOPOLOGY = (
    "Hegel Machine/config/phase3_shrink6_offline_build_profile_v1.json",
    "Hegel Machine/docs/Hegel_Machine_Phase3_Shrink6_Sealed_Dual_Strict_Qualification_Protocol_v1.md",
    "Hegel Machine/docs/Hegel_Machine_Phase3_Shrink_Step6_Engineering_Freeze_v1.md",
    "Hegel Machine/rust/strict_canonicalizer_shrink6/.gitignore",
    "Hegel Machine/rust/strict_canonicalizer_shrink6/Cargo.lock",
    "Hegel Machine/rust/strict_canonicalizer_shrink6/Cargo.toml",
    "Hegel Machine/rust/strict_canonicalizer_shrink6/README.md",
    "Hegel Machine/rust/strict_canonicalizer_shrink6/src/lib.rs",
    "Hegel Machine/rust/strict_canonicalizer_shrink6/src/main.rs",
    "Hegel Machine/rust/strict_canonicalizer_shrink6/tests/cli.rs",
    "Hegel Machine/src/hegel_machine/phase3_m3_bounded_enumerator_shrink6_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink6_core_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink6_diagnostic_profile_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_m3_shrink6_isolated_entrypoint_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink6_capacity_entrypoint_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink6_capacity_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink6_golden_vectors_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink6_registry_v1.py",
    "Hegel Machine/src/hegel_machine/phase3_shrink6_strict_entrypoint_v1.py",
    "Hegel Machine/src/hegel_machine/strict_ast_shrink6_v1.py",
    "Hegel Machine/tests/test_phase3_shrink6_capacity_and_enumerator_v1.py",
    "Hegel Machine/tests/test_phase3_shrink6_dual_strict_qualification_v1.py",
    "Hegel Machine/tests/test_phase3_shrink6_python_surface_v1.py",
    "Hegel Machine/tools/phase3_shrink6_dual_strict_qualification_v1.py",
)


EXPECTED_VECTOR_IDS = (
    "S01", "S02", "S03", "N01", "N02", "L01", "L02", "L03",
    "P01", "P02", "P03", "P04", "P05", "F01", "F02", "F03",
    "F04", "F05", "F06", "F07", "F08", "F09", "F10", "F11",
    "F12",
)


EXPECTED_CATEGORY_COUNTS = {
    "surviving_identity_checks": 3,
    "source_normalization_before_limit_checks": 2,
    "source_depth_limit_checks": 3,
    "source_priority_checks": 5,
    "formal_surviving_identity_checks": 1,
    "formal_depth_limit_checks": 3,
    "formal_priority_checks": 8,
}


EXPECTED_AUTHORITY_GUARDS = {
    "execution_state": "NOT_RUN",
    "closure_executed": False,
    "formal_roots_generated": False,
    "formal_roots": None,
    "certificate_issued": False,
    "signature_generated": False,
    "seed_generated": False,
    "target_roles_evaluated": False,
    "active_governance_changed": False,
    "formal_state_transition_allowed": False,
}


def _normalized_acceptance(implementation: str) -> dict[str, object]:
    cbor = bytes.fromhex("820183000001")
    digest = sha256(b"HEGEL/AST/V1\x00" + cbor).hexdigest()
    return {
        "schema_version": "hegel-strict-canonicalizer-shrink6-replay/1",
        "implementation": implementation,
        "dsl_version": "hegel-old-dsl-v1.6.0",
        "freeze_version": "hegel-freeze-p2b-p3-v1.6.0",
        "status": "ACCEPTED",
        "canonical_cbor_hex": cbor.hex(),
        "canonical_ast_hash": "sha256:" + digest,
        "root_operator_id": 0,
        "output_sort": "RationalValue",
        "depth": 0,
        "node_count": 1,
        "maximum_ast_depth": 3,
        "maximum_ast_node_count": 6,
        "maximum_top_level_clauses": 2,
    }


def _normalized_rejection(implementation: str) -> dict[str, object]:
    return {
        "schema_version": "hegel-strict-canonicalizer-shrink6-replay/1",
        "implementation": implementation,
        "dsl_version": "hegel-old-dsl-v1.6.0",
        "freeze_version": "hegel-freeze-p2b-p3-v1.6.0",
        "status": "REJECTED",
        "error_code": "REJECT_STRUCTURAL_LIMIT",
        "maximum_ast_depth": 3,
        "maximum_ast_node_count": 6,
        "maximum_top_level_clauses": 2,
    }


def _python_strict(status: str, *, boundary: str = "SOURCE_JSON") -> dict[str, object]:
    report = (
        _normalized_acceptance("python")
        if status == "ACCEPTED"
        else _normalized_rejection("python")
    )
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
    report = (
        _normalized_acceptance("rust")
        if status == "ACCEPTED"
        else _normalized_rejection("rust")
    )
    report.update(
        {
            "parent_dsl_version": "hegel-old-dsl-v1.5.0",
            "parent_freeze_version": "hegel-freeze-p2b-p3-v1.5.0",
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


def _capacity_pair() -> tuple[dict[str, object], dict[str, object]]:
    common = tool._capacity_guard_values()
    return (
        {
            "implementation": "python",
            "loaded_hegel_modules": tool.EXPECTED_PYTHON_CAPACITY_MODULES,
            **common,
        },
        {"implementation": "rust", **common},
    )


def test_live_roots_match_literal_sealed_constants() -> None:
    assert tool.diagnostic_root_hex_v1() == tool.EXPECTED_DIAGNOSTIC_ROOTS
    assert tool.strict_golden_manifest_root_v1() == tool.EXPECTED_MANIFEST_ROOT
    assert tool.EXPECTED_OUTCOME_ROOT == (
        "sha256:e5fd0885f95669dc6d369d0d3274778425fabb7e8c6286a27237a1b2bc8d3960"
    )
    assert tool.EXPECTED_DIAGNOSTIC_ROOTS == {
        "child_dsl_spec_root": (
            "da5ed2db33a88a0912d5003999f787cc26ba18564876615773a82bb742d9f8ae"
        ),
        "operator_semantics_root": (
            "922e48ada22dfa8621a4d516e07ec9aa7dc8fc10c165d1cafc963575aed5ec03"
        ),
        "identifier_registry_root": (
            "64c9415f7759eec140e439030c5a5374851b9024d7d4849b52b995704ba76ff1"
        ),
        "canonical_ast_schema_root": (
            "5de72fc51e27e5501561ffda6b05522f4941d1a13c4b324f5edcc15fa584a0bd"
        ),
        "canonical_cbor_profile_root": (
            "ef0008912962de9da322eaeea6e421e1e58d16be152f968298774af0fd3249ab"
        ),
    }


def test_parent_evidence_v_bytes_and_authority_are_live_bound() -> None:
    evidence_path = PROJECT_ROOT.parent / tool.PARENT_EVIDENCE_PATH
    payload = evidence_path.read_bytes()
    evidence = json.loads(payload)
    assert sha256(payload).hexdigest() == tool.PARENT_EVIDENCE_SHA256
    assert evidence["evidence_record_id"] == tool.PARENT_EVIDENCE_RECORD_ID
    assert evidence["execution_state"] == "NOT_RUN"
    assert evidence["formal_roots"] is None
    assert evidence["formal_state_transition_allowed"] is False
    assert tool._parent_evidence_binding(tool.PARENT_EVIDENCE_COMMIT) == {
        "evidence_commit": tool.PARENT_EVIDENCE_COMMIT,
        "evidence_path": tool.PARENT_EVIDENCE_PATH,
        "evidence_record_id": tool.PARENT_EVIDENCE_RECORD_ID,
        "evidence_sha256": "sha256:" + tool.PARENT_EVIDENCE_SHA256,
        "admitted_operation": "reduce max_total_ast_depth from 4 to 3",
        "formal_status_promotion_allowed": False,
    }


def test_source_w_topology_is_the_frozen_24_file_set() -> None:
    assert len(SOURCE_W_TOPOLOGY) == 24
    assert len(set(SOURCE_W_TOPOLOGY)) == 24
    assert all((PROJECT_ROOT.parent / path).is_file() for path in SOURCE_W_TOPOLOGY)
    assert tool.SUPERVISOR_TEST_PATH in SOURCE_W_TOPOLOGY
    assert tool.SUPERVISOR_PATH in SOURCE_W_TOPOLOGY
    assert tool.BUILD_PROFILE_PATH in SOURCE_W_TOPOLOGY
    assert sum("rust/strict_canonicalizer_shrink6/" in path for path in SOURCE_W_TOPOLOGY) == 7


def test_vector_ids_categories_and_boundaries_are_exact() -> None:
    vectors = tool.STRICT_GOLDEN_VECTORS_V1
    assert tuple(vector.vector_id for vector in vectors) == EXPECTED_VECTOR_IDS
    assert tuple(tool.EXPECTED_VECTOR_IDS) == EXPECTED_VECTOR_IDS
    assert len(vectors) == tool.EXPECTED_VECTOR_COUNT == 25
    observed_categories: dict[str, int] = {}
    for vector in vectors:
        observed_categories[vector.category] = observed_categories.get(vector.category, 0) + 1
        assert vector.category in tool.RUST_GOLDEN_FIELDS
        assert vector.boundary in {"SOURCE_JSON", "FORMAL_CBOR"}
    assert observed_categories == EXPECTED_CATEGORY_COUNTS
    assert sum(vector.boundary == "SOURCE_JSON" for vector in vectors) == 13
    assert sum(vector.boundary == "FORMAL_CBOR" for vector in vectors) == 12


def test_normalized_acceptance_is_cross_implementation_exact() -> None:
    python_outcome, python_fields = tool.normalize_endpoint_report(
        _normalized_acceptance("python"), implementation="python"
    )
    rust_outcome, rust_fields = tool.normalize_endpoint_report(
        _normalized_acceptance("rust"), implementation="rust"
    )
    assert python_outcome == rust_outcome
    assert python_fields == rust_fields
    assert python_fields["maximum_ast_depth"] == 3
    assert python_fields["maximum_ast_node_count"] == 6
    assert python_fields["maximum_top_level_clauses"] == 2


def test_normalized_acceptance_recomputes_ast_hash() -> None:
    report = _normalized_acceptance("python")
    report["canonical_ast_hash"] = "sha256:" + "00" * 32
    with pytest.raises(tool.QualificationError) as raised:
        tool.normalize_endpoint_report(report, implementation="python")
    assert raised.value.code == tool.FAIL_VECTOR


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("depth", 4),
        ("depth", -1),
        ("node_count", 7),
        ("canonical_cbor_hex", ""),
    ],
)
def test_accepted_metadata_rejects_out_of_language_values(
    field: str, value: object
) -> None:
    report = _normalized_acceptance("python")
    report[field] = value
    with pytest.raises(tool.QualificationError) as normalized:
        tool.normalize_endpoint_report(report, implementation="python")
    assert normalized.value.code == tool.FAIL_VECTOR
    with pytest.raises(tool.QualificationError) as archived:
        tool._accepted_metadata_guard(
            report,
            code=tool.FAIL_GUARD,
            label="archived vector",
        )
    assert archived.value.code == tool.FAIL_GUARD


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("maximum_ast_depth", 3.0),
        ("maximum_ast_depth", True),
        ("maximum_ast_node_count", 6.0),
        ("maximum_top_level_clauses", True),
        ("node_count", 1.0),
        ("depth", False),
    ],
)
def test_strict_report_rejects_numeric_type_aliases(
    field: str, value: object
) -> None:
    report = _python_strict("ACCEPTED")
    report[field] = value
    with pytest.raises(tool.QualificationError) as raised:
        tool._strict_report_guard(
            report, implementation="python", boundary="SOURCE_JSON"
        )
    assert raised.value.code == tool.FAIL_VECTOR


def test_capacity_schema_cardinalities_and_guard_values_are_exact() -> None:
    assert len(tool.RUST_CAPACITY_FIELDS) == 62
    assert len(tool.PYTHON_CAPACITY_FIELDS) == 63
    assert len(tool.RUST_GOLDEN_FIELDS) == 34
    assert len(tool.PYTHON_GOLDEN_FIELDS) == 35
    assert tool.PYTHON_CAPACITY_FIELDS == (
        tool.RUST_CAPACITY_FIELDS | {"loaded_hegel_modules"}
    )
    guards = tool._capacity_guard_values()
    assert len(guards) == 61
    assert guards["maximum_ast_depth"] == 3
    assert guards["challenge_source_candidate_count"] == 1_266
    assert guards["parent_only_source_candidate_count"] == 1_199
    assert guards["survivor_source_candidate_count"] == 242
    assert guards["survivor_unique_count"] == 225
    assert guards["subset_status"] == (
        "FROZEN_DEPTH4_CHALLENGE_LATTICE_ONLY_NOT_COMPLETE"
    )
    assert guards["executed_closure_status"] == "NOT_RUN"
    assert guards["complete_closure_enumerated"] is False
    assert guards["interpreted_as_complete_closure"] is False
    assert guards["formal_roots"] is None


def test_capacity_comparison_is_exact_and_nonterminal() -> None:
    python, rust = _capacity_pair()
    report = tool.compare_capacity_reports(python, rust)
    assert report["all_comparable_fields_equal"] is True
    assert report["executed_closure_status"] == "NOT_RUN"
    assert report["formal_roots"] is None
    tool._validate_dual_capacity_receipt(report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("challenge_source_candidate_count", 1_266.0),
        ("parent_only_source_candidate_count", True),
        ("target_or_split_modules_loaded", 0),
        ("formal_roots", {}),
        ("executed_closure_status", "COMPLETE"),
    ],
)
def test_capacity_comparison_rejects_drift_and_python_coercions(
    field: str, value: object
) -> None:
    python, rust = _capacity_pair()
    rust[field] = value
    with pytest.raises(tool.QualificationError) as raised:
        tool.compare_capacity_reports(python, rust)
    assert raised.value.code == tool.FAIL_CAPACITY


def test_canonical_json_is_ascii_compact_sorted_and_type_exact() -> None:
    assert tool._canonical_json_bytes({"z": "é", "a": 1}) == (
        b'{"a":1,"z":"\\u00e9"}'
    )
    assert tool._json_exact_equal(1, 1)
    assert not tool._json_exact_equal(True, 1)
    assert not tool._json_exact_equal(1, 1.0)
    assert not tool._json_exact_equal(float("nan"), float("nan"))


@pytest.mark.parametrize("payload", [b'{"x":1,"x":2}\n', b'{"x":NaN}\n'])
def test_endpoint_json_rejects_duplicates_and_nonfinite_numbers(
    payload: bytes,
) -> None:
    result = subprocess.CompletedProcess([], 0, payload, b"")
    with pytest.raises(tool.QualificationError) as raised:
        tool._one_json(result, "malformed endpoint")
    assert raised.value.code == tool.FAIL_ENDPOINT


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
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "run",
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


def test_committed_profiles_forbid_network_and_pull() -> None:
    receipt = tool._profile_images(PROJECT_ROOT)
    assert receipt["actor_profile_id"] == (
        "hegel-owner-accepted-container-technical-actors-v1"
    )
    assert receipt["build_profile_id"] == "hegel-shrink6-rust-offline-build-v1"
    build = json.loads(
        (PROJECT_ROOT / "config/phase3_shrink6_offline_build_profile_v1.json").read_text()
    )
    assert build["network"] == "none"
    assert build["pull_policy"] == "never"
    assert build["cargo_flags"] == ["--release", "--locked", "--offline"]


def test_authority_literal_is_fail_closed_not_run_and_null() -> None:
    source = TOOL_PATH.read_text()
    marker = '"authority_guards": {'
    start = source.index(marker)
    literal = source[start : source.index("\n        },", start) + len("\n        }")]
    for key, value in EXPECTED_AUTHORITY_GUARDS.items():
        encoded = json.dumps(value, sort_keys=True)
        if value is None:
            assert f'"{key}": None' in literal
        elif value is True:
            assert f'"{key}": True' in literal
        elif value is False:
            assert f'"{key}": False' in literal
        else:
            assert f'"{key}": {encoded}' in literal
    assert "RUNNING" not in literal
    assert "COMPLETE" not in literal


def test_nested_receipt_schemas_reject_extra_and_missing_keys() -> None:
    for fields in (
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
    ):
        candidate = {field: None for field in fields}
        assert tool._exact_mapping(candidate, "test receipt", fields) == candidate
        with pytest.raises(tool.QualificationError) as extra:
            tool._exact_mapping(
                {**candidate, "observed_closure_cardinality": 1},
                "test receipt",
                fields,
            )
        assert extra.value.code == tool.FAIL_GUARD
        missing = dict(candidate)
        missing.pop(next(iter(fields)))
        with pytest.raises(tool.QualificationError) as absent:
            tool._exact_mapping(missing, "test receipt", fields)
        assert absent.value.code == tool.FAIL_GUARD


def test_tool_source_compiles_and_malformed_report_fails_closed() -> None:
    compile(TOOL_PATH.read_text(), str(TOOL_PATH), "exec")
    with pytest.raises(tool.QualificationError) as raised:
        tool.validate_qualification_report({})
    assert raised.value.code == tool.FAIL_GUARD


def test_cli_invalid_basis_is_machine_readable_fail_closed() -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL_PATH), "--basis-commit", "not-a-commit"],
        cwd=PROJECT_ROOT.parent,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert result.returncode == 2
    assert result.stdout == b""
    failure = json.loads(result.stderr)
    assert failure["status"] == "FAIL_CLOSED"
    assert failure["failure_code"] == tool.FAIL_ARGUMENT


def test_source_file_rows_are_canonical_and_type_strict() -> None:
    row: dict[str, object] = {
        "path": "Hegel Machine/example.py",
        "mode": "100644",
        "git_blob_oid": "00" * 20,
        "sha256": "11" * 32,
        "size": 0,
    }
    assert tool.source_file_set_root_v1([row]).startswith("sha256:")
    for field, value in (
        ("path", "../Hegel Machine/example.py"),
        ("mode", 100644),
        ("git_blob_oid", 0),
        ("sha256", b"1" * 64),
        ("size", True),
    ):
        mutated = dict(row)
        mutated[field] = value
        with pytest.raises(tool.QualificationError):
            tool.source_file_set_root_v1([mutated])
