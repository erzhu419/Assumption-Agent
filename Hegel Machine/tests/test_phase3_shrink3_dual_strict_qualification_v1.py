from __future__ import annotations

from hashlib import sha256
import importlib.util
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools/phase3_shrink3_dual_strict_qualification_v1.py"
SPEC = importlib.util.spec_from_file_location("shrink3_dual_strict_tool", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def _accepted(implementation: str) -> dict[str, object]:
    cbor = bytes.fromhex("820183000001")
    digest = sha256(b"HEGEL/AST/V1\x00" + cbor).hexdigest()
    return {
        "schema_version": "hegel-strict-canonicalizer-shrink3-replay/1",
        "implementation": implementation,
        "dsl_version": "hegel-old-dsl-v1.3.0",
        "freeze_version": "hegel-freeze-p2b-p3-v1.3.0",
        "status": "ACCEPTED",
        "canonical_cbor_hex": cbor.hex(),
        "canonical_ast_hash": "sha256:" + digest,
        "root_operator_id": 0,
        "output_sort": "RationalValue",
        "depth": 0,
        "node_count": 1,
    }


def _rejected(implementation: str) -> dict[str, object]:
    return {
        "schema_version": "hegel-strict-canonicalizer-shrink3-replay/1",
        "implementation": implementation,
        "dsl_version": "hegel-old-dsl-v1.3.0",
        "freeze_version": "hegel-freeze-p2b-p3-v1.3.0",
        "status": "REJECTED",
        "error_code": "REJECT_REMOVED_BINARY_OPERATOR",
    }


def test_normalized_acceptance_is_cross_implementation_exact() -> None:
    python_outcome, python_fields = tool.normalize_endpoint_report(
        _accepted("python"), implementation="python"
    )
    rust_outcome, rust_fields = tool.normalize_endpoint_report(
        _accepted("rust"), implementation="rust"
    )
    assert python_outcome == rust_outcome
    assert python_fields == rust_fields
    assert python_outcome.startswith(b"ACCEPT\x00")


def test_normalized_rejection_ignores_non_normative_details() -> None:
    python = _rejected("python")
    rust = _rejected("rust")
    python["error_detail"] = "Python parser wording"
    rust["error_message"] = "Rust parser wording"
    assert tool.normalize_endpoint_report(
        python, implementation="python"
    ) == tool.normalize_endpoint_report(rust, implementation="rust")


def test_normalized_acceptance_recomputes_ast_hash() -> None:
    report = _accepted("python")
    report["canonical_ast_hash"] = "sha256:" + "00" * 32
    with pytest.raises(tool.QualificationError) as raised:
        tool.normalize_endpoint_report(report, implementation="python")
    assert raised.value.code == tool.FAIL_VECTOR


def test_capacity_comparison_is_exact_and_nonterminal() -> None:
    common = {
        "schema_version": "hegel-strict-capacity-replay-shrink3/1",
        "source_candidate_count": 2160,
        "accepted_source_count": 2160,
        "accepted_unique_count": 2160,
        "parent_identity_match_count": 2160,
        "rejected_count": 0,
        "rewrite_collapsed_count": 0,
        "accepted_set_commitment": tool.EXPECTED_CAPACITY_COMMITMENT,
        "subset_status": "SURVIVOR_SUBSET_ONLY_NOT_COMPLETE",
        "executed_closure_status": "NOT_RUN",
        "complete_closure_enumerated": False,
        "interpreted_as_complete_closure": False,
        "formal_roots": None,
        "target_or_split_modules_loaded": False,
    }
    python = {"implementation": "python", "loaded_hegel_modules": [], **common}
    rust = {"implementation": "rust", **common}
    report = tool.compare_capacity_reports(python, rust)
    assert report["all_comparable_fields_equal"] is True
    assert report["executed_closure_status"] == "NOT_RUN"
    assert report["formal_roots"] is None


def test_capacity_comparison_fails_on_any_shared_field_drift() -> None:
    common = {
        "source_candidate_count": 2160,
        "accepted_source_count": 2160,
        "accepted_unique_count": 2160,
        "parent_identity_match_count": 2160,
        "rejected_count": 0,
        "rewrite_collapsed_count": 0,
        "accepted_set_commitment": tool.EXPECTED_CAPACITY_COMMITMENT,
        "subset_status": "SURVIVOR_SUBSET_ONLY_NOT_COMPLETE",
        "executed_closure_status": "NOT_RUN",
        "complete_closure_enumerated": False,
        "interpreted_as_complete_closure": False,
        "formal_roots": None,
        "target_or_split_modules_loaded": False,
    }
    python = {"implementation": "python", **common}
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
    assert not any(
        "/var/run/docker.sock" in argument and ":" in argument.split("=")[-1]
        for argument in command[3:]
    )


def test_sealed_roots_and_manifest_order_are_literal_pinned() -> None:
    assert tool.strict_golden_manifest_root_v1() == tool.EXPECTED_MANIFEST_ROOT
    assert len(tool.STRICT_GOLDEN_VECTORS_V1) == 36
    assert tuple(vector.vector_id for vector in tool.STRICT_GOLDEN_VECTORS_V1) == (
        "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08",
        "A01", "A02", "A03", "A04",
        "P01", "P02", "P03", "P04", "P05", "P06",
        "F01", "F02", "F03",
        "Q01", "Q02", "Q03", "Q04", "Q05", "Q06",
        "H01", "H02", "H03", "H04", "H05", "H06",
        "R01", "R02", "R03",
    )
