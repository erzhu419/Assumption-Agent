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
    "phase3_shrink6_sealed_dual_strict_qualification_v1.json"
)
TOOL_PATH = PROJECT_ROOT / "tools/phase3_shrink6_dual_strict_qualification_v1.py"
SOURCE_W = "a69bf6d9746e302a07019f122047ac0bc74aa1c1"
EVIDENCE_V = "5bfe8474ca63abbadb1d3484a51ce3012081dfb3"
REPORT_FILE_SHA256 = (
    "d5417639c651ea5d8dfbc224c79b0af56f1eb9d8705ee244f19dc9d95e6f2d08"
)
REPORT_HASH = (
    "sha256:3d2a6f06daa47b34aa56ae0d318cc818ba211859063d7a6b81271bc6bf1f8287"
)


SPEC = importlib.util.spec_from_file_location("shrink6_evidence_tool", TOOL_PATH)
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


def test_report_is_exact_canonical_self_validating_source_w_evidence() -> None:
    payload, value = _strict_artifact()
    frozen_tool = _git(
        "show",
        f"{SOURCE_W}:Hegel Machine/tools/"
        "phase3_shrink6_dual_strict_qualification_v1.py",
        binary=True,
    )
    assert isinstance(frozen_tool, bytes)
    assert TOOL_PATH.read_bytes() == frozen_tool
    assert payload == tool._canonical_json_bytes(value) + b"\n"
    assert sha256(payload).hexdigest() == REPORT_FILE_SHA256
    assert value["diagnostic_report_hash"] == REPORT_HASH
    assert "evidence_record_id" not in value
    tool.validate_qualification_report(value)

    repository = value["repository_binding"]
    assert isinstance(repository, dict)
    assert repository["qualification_basis_commit"] == SOURCE_W
    assert repository["qualification_basis_parent_commits"] == [EVIDENCE_V]
    assert repository["qualification_basis_subject"] == (
        "hegel: freeze shrink6 depth-three admission"
    )
    assert type(repository["source_file_count"]) is int
    assert repository["source_file_count"] == 81
    assert repository["source_file_set_root"] == (
        "sha256:0858b3e379106f24bf8c2038062d79d5c3574a5738e34325a02240c941acbea0"
    )
    assert repository["supervisor_source_sha256"] == (
        "76beb647dc572104622754b3a9a19e7f33aa1587720a7b21e9808508c9aa4f87"
    )
    assert repository["parent_evidence_binding"] == {
        "evidence_commit": EVIDENCE_V,
        "evidence_path": (
            "Hegel Machine/artifacts/"
            "phase3_shrink5_dual_complete_enumeration_diagnostic_v1.json"
        ),
        "evidence_record_id": (
            "phase3_shrink5_dual_complete_enumeration_diagnostic_"
            "f33b86f3fbab70acb7d8e61fa47f59568a0d56c884c4cf75dfef961cc73dd34b"
        ),
        "evidence_sha256": (
            "sha256:99a799e34876754a8f938f8e25f756992d0784b03bae398b1434e57320b80c82"
        ),
        "admitted_operation": "reduce max_total_ast_depth from 4 to 3",
        "formal_status_promotion_allowed": False,
    }


def test_source_rows_and_git_archive_replay_exactly() -> None:
    _, value = _strict_artifact()
    repository = value["repository_binding"]
    assert isinstance(repository, dict)
    rows = repository["source_files"]
    assert isinstance(rows, list)
    assert repository["source_file_set_root"] == tool.source_file_set_root_v1(rows)
    assert _git("rev-parse", f"{SOURCE_W}:Hegel Machine") == (
        repository["project_tree_oid"]
    )
    assert _git("show", "-s", "--format=%s", SOURCE_W) == (
        repository["qualification_basis_subject"]
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_W, "HEAD"],
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

        listing = str(_git("ls-tree", SOURCE_W, "--", path))
        metadata, listed_path = listing.split("\t", 1)
        mode, kind, oid = metadata.split(" ")
        assert listed_path == path
        assert kind == "blob"
        assert mode == row["mode"]
        assert oid == row["git_blob_oid"]
        content = _git("show", f"{SOURCE_W}:{path}", binary=True)
        assert isinstance(content, bytes)
        assert len(content) == row["size"]
        assert sha256(content).hexdigest() == row["sha256"]

    archive = _git(
        "archive", "--format=tar", SOURCE_W, "--", *tool.ARCHIVE_PATHS,
        binary=True,
    )
    assert isinstance(archive, bytes)
    assert sha256(archive).hexdigest() == (
        "234cfb6f8fe79e15e48b3c6eda8d9b3c28d014f02cdc51a9aa82a71e6b4e9a06"
    )


def test_portable_validator_is_type_strict_and_rejects_extra_fields() -> None:
    _, value = _strict_artifact()
    mutations: list[dict[str, object]] = []

    float_alias = deepcopy(value)
    float_alias["dual_capacity_replay"]["survivor_accepted_count"] = 242.0
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


def test_vectors_and_capacity_lattice_are_exact_nonterminal_controls() -> None:
    _, value = _strict_artifact()
    vectors = value["dual_vector_replay"]
    capacity = value["dual_capacity_replay"]
    controls = value["built_in_replay_controls"]
    assert isinstance(vectors, dict)
    assert isinstance(capacity, dict)
    assert isinstance(controls, dict)

    assert vectors["status"] == "SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS"
    assert type(vectors["vector_count"]) is int and vectors["vector_count"] == 25
    assert vectors["all_normalized_outcomes_equal"] is True
    assert vectors["python_outcome_root"] == vectors["rust_outcome_root"] == (
        "sha256:e5fd0885f95669dc6d369d0d3274778425fabb7e8c6286a27237a1b2bc8d3960"
    )
    assert value["sealed_basis"]["golden_vector_manifest_root"] == (
        "sha256:2690413926d15db52dbd5a502ebe3fdfb1dc74d5ee3c82b2ed868cd16ab34a42"
    )
    assert [row["vector_id"] for row in vectors["vectors"]] == [
        "S01", "S02", "S03", "N01", "N02", "L01", "L02", "L03",
        "P01", "P02", "P03", "P04", "P05", "F01", "F02", "F03",
        "F04", "F05", "F06", "F07", "F08", "F09", "F10", "F11",
        "F12",
    ]
    for row in vectors["vectors"]:
        assert row["dual_equal"] is True
        assert type(row["input_wire_size"]) is int
        normalized = row["normalized"]
        if normalized["status"] == "ACCEPTED":
            assert type(normalized["maximum_ast_depth"]) is int
            assert normalized["maximum_ast_depth"] == 3
            assert type(normalized["maximum_ast_node_count"]) is int
            assert normalized["maximum_ast_node_count"] == 6
            assert type(normalized["maximum_top_level_clauses"]) is int
            assert normalized["maximum_top_level_clauses"] == 2

    assert controls == {
        "python_report_sha256": (
            "8dbd9d1d1860c9afd966652f49351761159eee5f9eeafe3f802a535a9554a43f"
        ),
        "rust_report_sha256": (
            "1a5804cbbea092fafe62b8930a3d333a32b4df09cc8acab8aa4916b09f2a6868"
        ),
        "python_passed_count": 25,
        "rust_passed_count": 25,
        "python_golden_field_count": 35,
        "python_capacity_field_count": 63,
        "python_combined_field_count": 98,
        "rust_golden_field_count": 34,
        "rust_capacity_field_count": 62,
        "rust_combined_field_count": 96,
    }

    assert capacity["status"] == (
        "DUAL_SHRINK6_FROZEN_DEPTH4_CHALLENGE_LATTICE_REPLAY_PASS_NOT_COMPLETE"
    )
    assert capacity["subset_status"] == (
        "FROZEN_DEPTH4_CHALLENGE_LATTICE_ONLY_NOT_COMPLETE"
    )
    assert capacity["challenge_source_candidate_count"] == 1_266
    assert capacity["challenge_parent_accepted_count"] == 1_266
    assert capacity["challenge_parent_canonical_unique_count"] == 1_249
    assert capacity["challenge_source_family_counts"] == {
        "A": 486,
        "B_abs": 390,
        "B_sign": 390,
    }
    assert capacity["challenge_source_lattice_commitment"] == (
        "sha256:a8cfb37278000933c2c51a2797e5bc0f4e7aad6970b37e178fc681f9358574d0"
    )
    assert capacity["challenge_parent_canonical_set_commitment"] == (
        "sha256:8f125763d3098d087dd7e9eb484b93097295ebd765b6f079795e8009623fb13e"
    )
    assert capacity["inherited_survivor_source_count"] == 175
    assert capacity["inherited_survivor_unique_count"] == 175
    assert capacity["normalized_survivor_source_count"] == 67
    assert capacity["normalized_survivor_unique_count"] == 50
    assert capacity["survivor_source_candidate_count"] == 242
    assert capacity["survivor_accepted_count"] == 242
    assert capacity["survivor_unique_count"] == 225
    assert capacity["survivor_parent_identity_match_count"] == 242
    assert capacity["survivor_rejected_count"] == 0
    assert capacity["survivor_rejection_counts"] == {}
    assert capacity["survivor_accepted_set_commitment"] == (
        "sha256:6787cd6c0782fda149e1ee93b37ca8d425f5ac78850c610e21cebf9da13a16d1"
    )
    assert capacity["parent_only_source_candidate_count"] == 1_199
    assert capacity["parent_only_unique_count"] == 1_199
    assert capacity["parent_only_parent_accepted_count"] == 1_199
    assert capacity["parent_only_depth"] == 4
    assert capacity["parent_only_node_count"] == 6
    assert capacity["parent_only_set_commitment"] == (
        "sha256:d3eb2b2d9caf1eece5a709d8113540e4709d579cdfbe3194f1cf176c9100b20d"
    )
    assert capacity["parent_only_source_child_rejected_count"] == 1_199
    assert capacity["parent_only_source_child_rejection_counts"] == {
        "REJECT_STRUCTURAL_LIMIT": 1_199
    }
    assert capacity["parent_only_source_rejection_outcome_commitment"] == (
        "sha256:9b0b766a4139db6297aea8b6032ad49147c1a26bf9b56291444a83681428cb0e"
    )
    assert capacity["parent_only_formal_child_rejected_count"] == 1_199
    assert capacity["parent_only_formal_child_rejection_counts"] == {
        "REJECT_STRUCTURAL_LIMIT": 1_199
    }
    assert capacity["parent_only_formal_rejection_outcome_commitment"] == (
        "sha256:97d50c34f51683a2502157961acc79d3b4e108b28bdaa266cf3721ffda8b3a96"
    )
    assert capacity["maximum_ast_depth"] == 3
    assert capacity["maximum_ast_node_count"] == 6
    assert capacity["maximum_top_level_clauses"] == 2
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
        "615567e6ec4965e7fdad2b2d83a7553b02574119f27f64d9a38df0d7160c6df4"
    )

    cargo = runtime["cargo_seed_manifest_receipt"]
    assert isinstance(cargo, dict)
    assert tool._validate_cargo_seed_receipt(cargo) == cargo
    assert type(cargo["file_count"]) is int and cargo["file_count"] == 43
    assert type(cargo["total_byte_count"]) is int
    assert cargo["total_byte_count"] == 3_907_160
    assert cargo["manifest_root"] == (
        "sha256:1341f70413f05de8a8d9293ae3147e2b0d367f99b45d5355df9027246bfeb397"
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
