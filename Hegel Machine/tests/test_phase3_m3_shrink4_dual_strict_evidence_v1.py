from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
ARTIFACT = (
    PROJECT_ROOT
    / "artifacts/phase3_m3_runtime/"
    "phase3_shrink4_sealed_dual_strict_qualification_v1.json"
)
TOOL_PATH = (
    PROJECT_ROOT / "tools/phase3_shrink4_dual_strict_qualification_v1.py"
)
SOURCE_O = "cd2c32bd3a27004b40f4550229f33afd73647433"
REPORT_FILE_SHA256 = (
    "41fdea5fd9b16ab436386ef7794412ffa46e17e68efc6b8448deed17c7f99aae"
)
REPORT_HASH = (
    "sha256:44b4e0c0a2b79f6afb67ace348c1b3726e0ba64058c97c4c61be0c111ef6acec"
)


SPEC = importlib.util.spec_from_file_location("shrink4_evidence_tool", TOOL_PATH)
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


def test_report_is_canonical_self_validating_source_o_evidence() -> None:
    payload, value = _strict_artifact()
    frozen_tool = _git(
        "show",
        f"{SOURCE_O}:Hegel Machine/tools/"
        "phase3_shrink4_dual_strict_qualification_v1.py",
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
    assert repository["qualification_basis_commit"] == SOURCE_O
    assert repository["qualification_basis_parent_commits"] == [
        "c286732c140bd9adcfd3eef2b1788b3eac0eb3e9"
    ]
    assert repository["qualification_basis_subject"] == (
        "hegel: freeze shrink4 two-clause admission"
    )
    assert repository["source_file_count"] == 61
    assert repository["source_file_set_root"] == (
        "sha256:03d5ab95e02f5fa6bb48db11ccb3682e0250985cb2ea17ad4372f4b2969c1a8e"
    )


def test_source_rows_and_git_archive_replay_exactly() -> None:
    _, value = _strict_artifact()
    repository = value["repository_binding"]
    assert isinstance(repository, dict)
    rows = repository["source_files"]
    assert isinstance(rows, list)
    assert repository["source_file_set_root"] == tool.source_file_set_root_v1(rows)
    assert _git("rev-parse", f"{SOURCE_O}:Hegel Machine") == (
        repository["project_tree_oid"]
    )
    assert _git("show", "-s", "--format=%s", SOURCE_O) == (
        repository["qualification_basis_subject"]
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_O, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
    )

    for row in rows:
        assert isinstance(row, dict)
        path = str(row["path"])
        assert PurePosixPath(path).parts[0] == "Hegel Machine"
        listing = str(_git("ls-tree", SOURCE_O, "--", path))
        metadata, listed_path = listing.split("\t", 1)
        mode, kind, oid = metadata.split(" ")
        assert listed_path == path
        assert kind == "blob"
        assert mode == row["mode"]
        assert oid == row["git_blob_oid"]
        content = _git("show", f"{SOURCE_O}:{path}", binary=True)
        assert isinstance(content, bytes)
        assert len(content) == row["size"]
        assert sha256(content).hexdigest() == row["sha256"]

    archive = _git(
        "archive", "--format=tar", SOURCE_O, "--", *tool.ARCHIVE_PATHS,
        binary=True,
    )
    assert isinstance(archive, bytes)
    assert sha256(archive).hexdigest() == repository["archive_sha256"]


def test_vectors_capacity_and_cargo_seed_are_exact_nonterminal_controls() -> None:
    _, value = _strict_artifact()
    vectors = value["dual_vector_replay"]
    capacity = value["dual_capacity_replay"]
    runtime = value["runtime_isolation"]
    assert isinstance(vectors, dict)
    assert isinstance(capacity, dict)
    assert isinstance(runtime, dict)

    assert vectors["status"] == "SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS"
    assert vectors["vector_count"] == 22
    assert vectors["all_normalized_outcomes_equal"] is True
    assert vectors["python_outcome_root"] == vectors["rust_outcome_root"] == (
        "sha256:c19341f08ac5f5759c2cdcb3681a37d958de362b81d02c184f7e2413dca18d7c"
    )
    assert value["sealed_basis"]["golden_vector_manifest_root"] == (
        "sha256:f84035e632bf5a655a9ebd636a0cafe7ab1097c45be87d4db944a0012f52aa90"
    )

    assert capacity["status"] == "DUAL_SURVIVOR_SUBSET_REPLAY_PASS_NOT_COMPLETE"
    assert capacity["source_candidate_count"] == 2160
    assert capacity["normalized_and2_count"] == 2160
    assert capacity["accepted_source_count"] == 2160
    assert capacity["accepted_unique_count"] == 2160
    assert capacity["parent_identity_match_count"] == 2160
    assert capacity["accepted_set_commitment"] == (
        "sha256:9045e4ebe6416dcbf699e7972f25468aef45c0f0aec0e58806061b7ce64d790e"
    )
    assert capacity["subset_status"] == "FULL_AND2_SURVIVOR_SET_ONLY_NOT_COMPLETE"
    assert capacity["complete_closure_enumerated"] is False
    assert capacity["interpreted_as_complete_closure"] is False

    cargo = runtime["cargo_seed_manifest_receipt"]
    assert isinstance(cargo, dict)
    assert tool._validate_cargo_seed_receipt(cargo) == cargo
    assert cargo["file_count"] == 43
    assert cargo["total_byte_count"] == 3_907_160
    assert cargo["manifest_root"] == (
        "sha256:a280e5a05d54c2904c19b5ad296650acd90de853ce5260deb93cdade595cef80"
    )
    assert runtime["pull_policy"] == "never"
    assert runtime["network_mode"] == "none"
    assert runtime["rust_target_volume_removed_after_run"] is True
    assert re.fullmatch(r"[0-9a-f]{64}", str(runtime["rust_binary_sha256"]))


def test_authority_guards_and_future_enumeration_absence_are_fail_closed() -> None:
    _, value = _strict_artifact()
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
        "program_chunk_manifest_root",
        "bucket_accounting_root",
        "target_root",
        "match_program_hash",
        "closure_verdict",
    }

    def keys(node: object) -> set[str]:
        if isinstance(node, dict):
            return set(node).union(*(keys(item) for item in node.values()))
        if isinstance(node, list):
            return set().union(*(keys(item) for item in node))
        return set()

    assert keys(value).isdisjoint(forbidden)
