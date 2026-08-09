from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
ARTIFACT = (
    PROJECT_ROOT
    / "artifacts/phase3_m3_runtime/"
    "phase3_shrink5_sealed_dual_strict_qualification_v1.json"
)
TOOL_PATH = PROJECT_ROOT / "tools/phase3_shrink5_dual_strict_qualification_v1.py"
SOURCE_S = "320b0a3458901090cb738023a4398220fb1d9277"
EVIDENCE_R = "1bbdae8f3131625621c0bc1cfdfe5d7da6035e13"
REPORT_FILE_SHA256 = (
    "75761fc536d96d5d0bc91c5c0ba30dbc7c9ee21aac8d3f1dc5c96f6aca919b76"
)
REPORT_HASH = (
    "sha256:5ee04b21477fd9f09271272fd6ecbf876b885b7831b37a868343a93996a187db"
)


SPEC = importlib.util.spec_from_file_location("shrink5_evidence_tool", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_artifact() -> tuple[bytes, dict[str, object]]:
    payload = ARTIFACT.read_bytes()
    value = json.loads(
        payload.decode("ascii"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    assert isinstance(value, dict)
    return payload, value


def _git(*arguments: str, binary: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return result.stdout if binary else result.stdout.decode("utf-8").strip()


def _rehash(report: dict[str, object]) -> None:
    body = dict(report)
    body.pop("diagnostic_report_hash", None)
    report["diagnostic_report_hash"] = "sha256:" + sha256(
        tool.REPORT_DOMAIN + b"\x00" + tool._canonical_json_bytes(body)
    ).hexdigest()


def test_report_is_canonical_self_validating_source_s_evidence() -> None:
    payload, value = _strict_artifact()
    frozen_tool = _git(
        "show",
        f"{SOURCE_S}:Hegel Machine/tools/"
        "phase3_shrink5_dual_strict_qualification_v1.py",
        binary=True,
    )
    assert isinstance(frozen_tool, bytes)
    assert TOOL_PATH.read_bytes() == frozen_tool
    assert payload == tool._canonical_json_bytes(value) + b"\n"
    assert sha256(payload).hexdigest() == REPORT_FILE_SHA256
    assert value["diagnostic_report_hash"] == REPORT_HASH
    tool.validate_qualification_report(value)

    repository = value["repository_binding"]
    assert isinstance(repository, dict)
    assert repository["qualification_basis_commit"] == SOURCE_S
    assert repository["qualification_basis_parent_commits"] == [EVIDENCE_R]
    assert repository["qualification_basis_subject"] == (
        "hegel: freeze shrink5 six-node admission"
    )
    assert type(repository["source_file_count"]) is int
    assert repository["source_file_count"] == 71
    assert repository["source_file_set_root"] == (
        "sha256:4a7ae37381f7ec77a362d0cb945f2ddaf0649353777b911e32f696e747ebfeaf"
    )
    assert repository["supervisor_source_sha256"] == (
        "e481d72cd5b9b86d34d0daa175f390e310cbbe7bd3d4e7321af71edd68316f0d"
    )
    assert repository["parent_evidence_binding"] == {
        "evidence_commit": EVIDENCE_R,
        "evidence_path": (
            "Hegel Machine/artifacts/"
            "phase3_shrink4_dual_complete_enumeration_diagnostic_v1.json"
        ),
        "evidence_record_id": (
            "phase3_shrink4_dual_complete_enumeration_diagnostic_"
            "5693b38315689969a1a525b75bec2917f95af1aa54951267797a0319afc60521"
        ),
        "evidence_sha256": (
            "sha256:2d653f667d8d43e0e8e68c54d6f0a939aab57bf6ba3add9b334809ca17745058"
        ),
        "admitted_operation": "reduce max_total_node_count from 7 to 6",
        "formal_status_promotion_allowed": False,
    }


def test_source_rows_and_git_archive_replay_exactly() -> None:
    _, value = _strict_artifact()
    repository = value["repository_binding"]
    assert isinstance(repository, dict)
    rows = repository["source_files"]
    assert isinstance(rows, list)
    assert repository["source_file_set_root"] == tool.source_file_set_root_v1(rows)
    assert _git("rev-parse", f"{SOURCE_S}:Hegel Machine") == (
        repository["project_tree_oid"]
    )
    assert _git("show", "-s", "--format=%s", SOURCE_S) == (
        repository["qualification_basis_subject"]
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_S, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
    )

    previous = ""
    for row in rows:
        assert isinstance(row, dict)
        assert set(row) == {"path", "mode", "git_blob_oid", "size", "sha256"}
        path = row["path"]
        assert isinstance(path, str)
        assert path > previous
        previous = path
        assert PurePosixPath(path).parts[0] == "Hegel Machine"
        assert type(row["size"]) is int and row["size"] >= 0
        assert isinstance(row["mode"], str)
        assert isinstance(row["git_blob_oid"], str)
        assert isinstance(row["sha256"], str)

        listing = str(_git("ls-tree", SOURCE_S, "--", path))
        metadata, listed_path = listing.split("\t", 1)
        mode, kind, oid = metadata.split(" ")
        assert listed_path == path
        assert kind == "blob"
        assert mode == row["mode"]
        assert oid == row["git_blob_oid"]
        content = _git("show", f"{SOURCE_S}:{path}", binary=True)
        assert isinstance(content, bytes)
        assert len(content) == row["size"]
        assert sha256(content).hexdigest() == row["sha256"]

    archive = _git(
        "archive", "--format=tar", SOURCE_S, "--", *tool.ARCHIVE_PATHS,
        binary=True,
    )
    assert isinstance(archive, bytes)
    assert sha256(archive).hexdigest() == (
        "3362e19a39940276c3628ddea5de5c8df93679750e55383a53395c444e14720e"
    )


def test_portable_validator_is_type_strict_and_rejects_extra_fields() -> None:
    _, value = _strict_artifact()
    mutations: list[dict[str, object]] = []

    float_alias = deepcopy(value)
    float_alias["dual_capacity_replay"]["survivor_accepted_count"] = 175.0
    _rehash(float_alias)
    mutations.append(float_alias)

    boolean_alias = deepcopy(value)
    boolean_alias["authority_guards"]["closure_executed"] = 0
    _rehash(boolean_alias)
    mutations.append(boolean_alias)

    extra_field = deepcopy(value)
    extra_field["unregistered_future_field"] = None
    _rehash(extra_field)
    mutations.append(extra_field)

    for mutated in mutations:
        with pytest.raises(tool.QualificationError) as raised:
            tool.validate_qualification_report(mutated)
        assert raised.value.code == tool.FAIL_GUARD


def test_vectors_and_both_capacity_boundaries_are_exact_nonterminal_controls() -> None:
    _, value = _strict_artifact()
    vectors = value["dual_vector_replay"]
    capacity = value["dual_capacity_replay"]
    controls = value["built_in_replay_controls"]
    assert isinstance(vectors, dict)
    assert isinstance(capacity, dict)
    assert isinstance(controls, dict)

    assert vectors["status"] == "SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS"
    assert type(vectors["vector_count"]) is int and vectors["vector_count"] == 22
    assert vectors["all_normalized_outcomes_equal"] is True
    assert vectors["python_outcome_root"] == vectors["rust_outcome_root"] == (
        "sha256:8f82178c0f33d5295601d2e112b0b6e25ef18d73e5fc35d8d601024c1f0ddf94"
    )
    assert value["sealed_basis"]["golden_vector_manifest_root"] == (
        "sha256:156f7e20407437bb753b097a87932f469701d1de6d1d577b0fa1b7a98f47e52e"
    )
    assert [row["vector_id"] for row in vectors["vectors"]] == [
        "S01", "S02", "S03", "N01", "N02", "L01", "L02",
        "P01", "P02", "P03", "P04", "P05", "F01", "F02",
        "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10",
    ]
    for row in vectors["vectors"]:
        assert row["dual_equal"] is True
        assert type(row["input_wire_size"]) is int
        normalized = row["normalized"]
        if normalized["status"] == "ACCEPTED":
            assert type(normalized["maximum_ast_node_count"]) is int
            assert normalized["maximum_ast_node_count"] == 6
            assert type(normalized["maximum_top_level_clauses"]) is int
            assert normalized["maximum_top_level_clauses"] == 2

    assert controls == {
        "python_report_sha256": (
            "dec68fbfa4a255c1145eaee07d8ca3de9cbd267dbd26bc9e942fa312f0a89c33"
        ),
        "rust_report_sha256": (
            "8a76ce2d49955b969587b4df39bb1c4654d16093392e2e170c79dac8b67bf5e2"
        ),
        "python_passed_count": 22,
        "rust_passed_count": 22,
        "python_golden_field_count": 34,
        "python_capacity_field_count": 46,
        "python_combined_field_count": 80,
        "rust_golden_field_count": 33,
        "rust_capacity_field_count": 45,
        "rust_combined_field_count": 78,
    }

    assert capacity["status"] == (
        "DUAL_SHRINK5_SURVIVOR_AND_PARENT_NODE7_BOUNDARY_REPLAY_PASS_NOT_COMPLETE"
    )
    assert capacity["subset_status"] == (
        "FULL_175_SURVIVOR_AND_2160_PARENT_NODE7_BOUNDARY_SETS_ONLY_NOT_COMPLETE"
    )
    for field in (
        "survivor_source_candidate_count",
        "survivor_accepted_count",
        "survivor_unique_count",
        "survivor_parent_identity_match_count",
    ):
        assert type(capacity[field]) is int and capacity[field] == 175
    assert capacity["survivor_rejected_count"] == 0
    assert capacity["survivor_rejection_counts"] == {}
    assert capacity["survivor_accepted_set_commitment"] == (
        "sha256:f5ab7f079ad943d65a74881eb59c7bb46385e1c437ca8ab036bb071dfa3874ac"
    )
    assert capacity["parent_only_source_candidate_count"] == 2160
    assert capacity["parent_only_parent_accepted_count"] == 2160
    assert capacity["parent_only_node_count"] == 7
    assert capacity["parent_only_set_commitment"] == (
        "sha256:7e0e8780149f03ce85723408f7e3eff2cd684e8938896125cf8e34be9ac70b5e"
    )
    assert capacity["parent_only_source_child_rejected_count"] == 2160
    assert capacity["parent_only_source_child_rejection_counts"] == {
        "REJECT_STRUCTURAL_LIMIT": 2160
    }
    assert capacity["parent_only_source_rejection_outcome_commitment"] == (
        "sha256:8617b56bdfa347f11f2c68b6a41f0992652f1e23e6d651017b17eb50169a9f39"
    )
    assert capacity["parent_only_formal_child_rejected_count"] == 2160
    assert capacity["parent_only_formal_child_rejection_counts"] == {
        "REJECT_STRUCTURAL_LIMIT": 2160
    }
    assert capacity["parent_only_formal_rejection_outcome_commitment"] == (
        "sha256:9a6b489ed90960008aebbecdbcf0bc5cf1595b7a8206d179bbe898540dabf617"
    )
    assert capacity["canonical_program_budget"] == 50_000
    assert capacity["first_out_of_budget_ordinal"] is None
    assert capacity["complete_closure_enumerated"] is False
    assert capacity["interpreted_as_complete_closure"] is False
    assert capacity["executed_closure_status"] == "NOT_RUN"
    assert capacity["formal_roots"] is None
    assert capacity["target_or_split_modules_loaded"] is False


def test_isolation_authority_and_future_closure_absence_are_fail_closed() -> None:
    _, value = _strict_artifact()
    runtime = value["runtime_isolation"]
    assert isinstance(runtime, dict)
    assert runtime["role_topology"] == (
        "HOST_SUPERVISOR_PLUS_TWO_DISJOINT_PINNED_CONTAINERS"
    )
    assert runtime["pull_policy"] == "never"
    assert runtime["network_mode"] == "none"
    assert runtime["capabilities_dropped"] == "ALL"
    assert runtime["no_new_privileges"] is True
    assert runtime["container_root_filesystem_read_only"] is True
    assert runtime["source_snapshot_mount_read_only"] is True
    assert runtime["cargo_locked"] is True
    assert runtime["cargo_offline"] is True
    assert runtime["fresh_ephemeral_rust_target_volume"] is True
    assert runtime["rust_target_volume_removed_after_run"] is True
    assert runtime["technical_role_independence"] is True
    assert runtime["same_admin_controller"] is True
    assert runtime["organizational_independence"] is False
    assert runtime["independent_human_actors"] is False
    assert type(runtime["worker_count"]) is int and runtime["worker_count"] == 8
    assert re.fullmatch(r"[0-9a-f]{64}", runtime["rust_binary_sha256"])
    assert runtime["rust_binary_sha256"] == (
        "f1c7b5295e7e42a2d2ca92054c7ae37f41cacb02a4ad20a265ba8fd8ad6413a5"
    )

    cargo = runtime["cargo_seed_manifest_receipt"]
    assert isinstance(cargo, dict)
    assert tool._validate_cargo_seed_receipt(cargo) == cargo
    assert type(cargo["file_count"]) is int and cargo["file_count"] == 43
    assert type(cargo["total_byte_count"]) is int
    assert cargo["total_byte_count"] == 3_907_160
    assert cargo["manifest_root"] == (
        "sha256:60e5cad5134fc5aeac81185e73597469356d51da59e8fe72379ecdd402b38b59"
    )

    assert value["claim_level"] == "NON_FORMAL_DUAL_STRICT_QUALIFICATION"
    assert value["authority_guards"] == {
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

    forbidden = {
        "observed_closure_cardinality",
        "first_out_of_budget_program_hash",
        "first_out_of_budget_cbor",
        "canonical_program_archive_root",
        "program_output_archive_root",
        "program_chunk_manifest_root",
        "bucket_accounting_root",
        "target_root",
        "match_program_hash",
        "match_record",
        "closure_verdict",
        "outside_certificate",
        "mdl_certificate",
    }

    def keys(node: object) -> set[str]:
        if isinstance(node, dict):
            return set(node).union(*(keys(item) for item in node.values()))
        if isinstance(node, list):
            return set().union(*(keys(item) for item in node))
        return set()

    assert keys(value).isdisjoint(forbidden)
