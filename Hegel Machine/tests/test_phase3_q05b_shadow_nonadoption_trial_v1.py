from __future__ import annotations

import ast
from copy import deepcopy
from hashlib import sha256
import io
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import types
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKTREE = ROOT.parent
CONTRACT_PATH = ROOT / "config/phase3_q05b_shadow_nonadoption_trial_v1.json"
CORE_PATH = ROOT / "src/hegel_machine/phase3_q05b_shadow_nonadoption_trial_v1.py"
TOOL_PATH = ROOT / "tools/phase3_q05b_shadow_nonadoption_trial_v1.py"
DOC_PATH = ROOT / "docs/Hegel_Machine_Q05b_Shadow_Nonadoption_Trial_v1.md"
OLD_ATTEMPT2_WORKTREE = Path(
    "/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree"
)
FROZEN_COMMIT = "f2b0a1fd37a2f488cea64fc2c8d1dd877639bf9f"
FROZEN_TREE = "12d354842f0359aa5b4ecbeebd83cbe123567788"
FROZEN_PARENT = "65dfc5f19450cee9510c63a85de32279160f291f"
OLD_SUPERVISOR_SHA256 = (
    "af406bba2e4d11f8f59c1d7f91d3f84c887d1e90d10cc1749baf2a8610a7c5d2"
)
TAGGED_CAPSULE = "sha256:" + "1" * 64
TAGGED_AUTHORIZATION = "sha256:" + "2" * 64
TAGGED_INTENT = "sha256:" + "3" * 64
TAGGED_CONSUMED = "sha256:" + "4" * 64
RAW_STDOUT = "5" * 64
RAW_STDERR = "6" * 64
FINAL_SOURCE_STATUS = (
    "SOURCE_CORE_TOOL_TESTS_PRESENT_AND_BOUND_EXECUTION_STILL_REQUIRES_OWNER_AUTHORIZATION"
)
FAILURE_STATES = (
    "LEDGER_ROOT_OR_LOCK_PUBLICATION_UNCERTAIN_TERMINAL",
    "INTENT_PUBLICATION_UNCERTAIN_TERMINAL",
    "INTENT_STAGED_PREPARATION_FAILED_CLOSED_UNCONSUMED",
    "CONSUMPTION_PUBLICATION_UNCERTAIN_TERMINAL",
    "CONSUMED_BEFORE_SUPERVISOR_LAUNCH_FAILED",
    "SUPERVISOR_RAISED_TIMED_OUT_OR_CLEANUP_UNCERTAIN",
    "SUPERVISOR_RETURNED_NONZERO_TERMINAL",
    "SUPERVISOR_RETURNED_ARTIFACT_MISSING_OR_INVALID",
    "POSTRETURN_NONADOPTION_OR_ISOLATION_REPLAY_FAILED",
    "SHADOW_OUTCOME_LINK_PUBLICATION_UNCERTAIN",
)
FAILURE_FACT_FIELDS = (
    "ledger_created",
    "ledger_consumed",
    "consumption_publication_uncertain",
    "supervisor_invocation_count",
    "supervisor_spawn_attempted",
    "supervisor_called",
    "supervisor_returned",
    "docker_or_cargo_may_have_run",
    "artifact_may_exist",
    "cleanup_attempted",
    "cleanup_complete",
    "survivor_absence_verified",
    "attempt2_state_touched",
    "meta_or_production_state_touched",
    "adoption_or_current_pointer_written",
    "failure_record_publication_uncertain",
    "no_retry",
)


def _canonical_without_lf(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _canonical(value: object) -> bytes:
    return _canonical_without_lf(value) + b"\n"


def _domain_digest(domain: str, value: object) -> str:
    return "sha256:" + sha256(domain.encode("ascii") + _canonical_without_lf(value)).hexdigest()


def _load(name: str, path: Path):
    """Load exact source bytes without importlib bytecode-cache behavior."""

    assert path.is_file(), path
    raw = path.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = os.fspath(path)
    module.__package__ = None
    code = compile(raw.decode("utf-8"), os.fspath(path), "exec", dont_inherit=True)
    exec(code, module.__dict__)
    return module


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        assert key not in result, f"duplicate JSON key: {key}"
        result[key] = value
    return result


def _resign_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    unsigned = deepcopy(result)
    unsigned.pop("contract_digest")
    result["contract_digest"] = _domain_digest(
        result["hash_domains"]["contract"], unsigned
    )
    return result


def _resign_record(
    contract: Mapping[str, Any], kind: str, value: Mapping[str, Any]
) -> dict[str, Any]:
    result = deepcopy(dict(value))
    policy = contract["canonicalization"]["record_digest_policy"][kind]
    digest_field = policy["digest_field"]
    unsigned = deepcopy(result)
    unsigned.pop(digest_field)
    result[digest_field] = _domain_digest(
        contract["hash_domains"][policy["domain_key"]], unsigned
    )
    return result


def _executable_identity(path: str, digit: str) -> dict[str, Any]:
    return {
        "path": path,
        "raw_sha256": digit * 64,
        "mode": 0o100755,
        "uid": 1000,
        "nlink": 1,
        "size": 1,
        "dev": 1,
        "ino": int(digit, 16) + 1,
    }


def _cargo_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    policy = contract["actual_execution_policy"]["isolated_cargo_cache_policy"]
    return {
        "source_root": policy["copy_source"],
        "target_root": policy["copy_target"],
        "file_count": 1,
        "directory_count": 1,
        "total_bytes": 1,
        "manifest_digest": "sha256:" + "a" * 64,
        "target_initially_absent": True,
        "copy_required": True,
    }


def _finalized_contract(contract: Mapping[str, Any], tool: Any) -> dict[str, Any]:
    result = deepcopy(dict(contract))
    binding = result["source_binding"]
    core_raw = CORE_PATH.read_bytes()
    tests_raw = Path(__file__).read_bytes()
    tool_raw = TOOL_PATH.read_bytes()
    assert binding["status"] == FINAL_SOURCE_STATUS
    assert binding["normalized_source_digest"] == (
        "sha256:" + sha256(tool._normalize_tool_bytes(tool_raw)).hexdigest()
    )
    assert binding["source_raw_sha256"] == sha256(core_raw).hexdigest()
    assert binding["source_static_surface_sha256"] == sha256(
        tool._canonical_json_bytes(tool._source_static_surface(core_raw, tool_raw))
    ).hexdigest()
    assert binding["tests_raw_sha256"] == sha256(tests_raw).hexdigest()
    unsigned_binding = deepcopy(binding)
    unsigned_binding.pop("source_binding_digest")
    assert binding["source_binding_digest"] == _domain_digest(
        result["hash_domains"]["source_binding"], unsigned_binding
    )
    assert _resign_contract(result) == result
    return result


def _placeholder_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(contract))
    binding = result["source_binding"]
    binding["status"] = "PLACEHOLDER_SOURCE_NOT_IMPLEMENTED_NOT_BOUND_EXECUTION_FORBIDDEN"
    for field in (
        "normalized_source_digest",
        "source_binding_digest",
        "source_raw_sha256",
        "source_static_surface_sha256",
        "tests_raw_sha256",
    ):
        binding[field] = None
    return _resign_contract(result)


def _failure_row(contract: Mapping[str, Any], state: str) -> dict[str, Any]:
    matches = [
        row
        for row in contract["failure_state_policy"]["state_fact_matrix"]
        if row["error_state"] == state
    ]
    assert len(matches) == 1
    return matches[0]


def _failure_receipt(
    contract: Mapping[str, Any], state: str, phase: str | None = None
) -> dict[str, Any]:
    policy = contract["failure_state_policy"]
    matrix = _failure_row(contract, state)
    facts = {
        field: deepcopy(matrix["allowed_facts"][field][0])
        for field in FAILURE_FACT_FIELDS
    }
    if facts["supervisor_called"] is True:
        returned_values = matrix["allowed_facts"]["supervisor_returned"]
        facts["supervisor_returned"] = (
            None if any(item is None for item in returned_values) else True
        )
    cleanup_values = matrix["allowed_facts"]["cleanup_complete"]
    survivor_values = matrix["allowed_facts"]["survivor_absence_verified"]
    common_cleanup = next(
        item
        for item in cleanup_values
        if any(type(item) is type(other) and item == other for other in survivor_values)
    )
    facts["cleanup_complete"] = deepcopy(common_cleanup)
    facts["survivor_absence_verified"] = deepcopy(common_cleanup)
    consumed_policy = matrix["consumed_record_digest_policy"]
    consumed_digest = (
        TAGGED_CONSUMED
        if consumed_policy == "TAGGED" or facts["ledger_consumed"] is True
        else None
    )
    exit_policy = matrix["supervisor_exit_code_policy"]
    exit_code: int | None
    if exit_policy == "ZERO":
        exit_code = 0
    elif exit_policy == "NONZERO_INT":
        exit_code = 7
    else:
        exit_code = None
    output_policy = matrix["supervisor_output_digest_policy"]
    returned = facts["supervisor_returned"]
    if returned is not True and exit_policy == "NULL_OR_INT":
        exit_code = None
    stdout_digest = RAW_STDOUT if output_policy == "RAW" or returned is True else None
    stderr_digest = RAW_STDERR if output_policy == "RAW" or returned is True else None
    value = {
        "schema_version": policy["schema_version"],
        "status": policy["fixed_fields"]["status"],
        "trial_id": contract["trial_identity"]["trial_id"],
        "contract_digest": contract["contract_digest"],
        "capsule_digest": TAGGED_CAPSULE,
        "authorization_digest": TAGGED_AUTHORIZATION,
        "intent_record_digest": TAGGED_INTENT,
        "consumed_record_digest": consumed_digest,
        "error_state": state,
        "phase_id": phase or matrix["phase_ids"][0],
        "detail": "synthetic producer-free KAT",
        **facts,
        "supervisor_exit_code": exit_code,
        "supervisor_stdout_raw_sha256": stdout_digest,
        "supervisor_stderr_raw_sha256": stderr_digest,
        "receipt_digest": None,
    }
    assert set(value) == set(policy["receipt_exact_fields"])
    return _resign_record(contract, "failure", value)


def _failure_context(receipt: Mapping[str, Any], channel: str) -> dict[str, Any]:
    return {
        "capsule_digest": receipt["capsule_digest"],
        "authorization_digest": receipt["authorization_digest"],
        "intent_record_digest": receipt["intent_record_digest"],
        "consumed_record_digest": receipt["consumed_record_digest"],
        "receipt_channel": channel,
    }


def _outside_allowed(values: list[Any]) -> Any:
    candidates: tuple[Any, ...] = (None, False, True, 0, 1, 2, "FORBIDDEN")
    for candidate in candidates:
        if not any(type(candidate) is type(item) and candidate == item for item in values):
            return candidate
    raise AssertionError("test candidate registry is exhausted")


@pytest.fixture(scope="module")
def contract() -> dict[str, object]:
    return json.loads(
        CONTRACT_PATH.read_text(encoding="ascii"), object_pairs_hook=_unique_pairs
    )


@pytest.fixture(scope="module")
def core():
    return _load("_q05b_shadow_nonadoption_core_test", CORE_PATH)


@pytest.fixture(scope="module")
def tool():
    return _load("_q05b_shadow_nonadoption_tool_test", TOOL_PATH)


def test_contract_surface_canonical_digest_and_exact_predecessor(contract) -> None:
    raw = CONTRACT_PATH.read_bytes()
    assert raw.decode("ascii")
    assert raw == _canonical(contract)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert set(contract) == set(
        contract["canonicalization"]["contract_exact_top_level_fields"]
    )
    body = deepcopy(contract)
    observed = body.pop("contract_digest")
    assert observed == _domain_digest(contract["hash_domains"]["contract"], body)
    source = contract["isolated_source_policy"]
    assert source["checkout_commit"] == FROZEN_COMMIT
    assert source["checkout_tree"] == FROZEN_TREE
    assert source["checkout_commit_parent"] == FROZEN_PARENT
    assert source["old_supervisor_unchanged_from_parent"] is True
    assert (
        contract["actual_execution_policy"]["old_supervisor_raw_sha256"]
        == OLD_SUPERVISOR_SHA256
    )
    assert contract["actual_execution_policy"]["maximum_supervisor_invocations"] == 1


def test_attempt2_protected_identity_is_frozen_without_live_git(contract) -> None:
    protected = contract["protected_state_policy"]
    assert protected["attempt2_worktree_root"] == os.fspath(OLD_ATTEMPT2_WORKTREE)
    assert protected["attempt2_worktree_git_toplevel"] == os.fspath(
        OLD_ATTEMPT2_WORKTREE
    )
    assert protected["attempt2_worktree_git_dir"] == (
        "/home/erzhu419/mine_code/Asumption Agent/.git/worktrees/"
        "m3-start-runtime-worktree"
    )
    assert protected["attempt2_worktree_commit"] == FROZEN_COMMIT
    assert protected["attempt2_worktree_tree"] == FROZEN_TREE
    assert protected["protected_state_mutation_allowed"] is False


def test_attempt2_git_discovery_precedes_pinned_admin_dir_replay(
    monkeypatch, contract, tool
) -> None:
    frozen = _resign_contract(contract)
    protected = frozen["protected_state_policy"]
    worktree = protected["attempt2_worktree_root"]
    git_dir = protected["attempt2_worktree_git_dir"]
    expected_identity = _executable_identity(
        frozen["isolated_source_policy"]["git_executable_path"], "b"
    )
    admin_identity = {
        "path": git_dir,
        "exists": True,
        "existing_prefix": git_dir,
        "mode": 0o040755,
        "uid": 1000,
        "nlink": 2,
        "dev": 1,
        "ino": 2,
        "mnt_id": 3,
    }
    git_calls: list[tuple[list[str], str, Mapping[str, Any]]] = []
    snapshot_calls: list[tuple[str, str]] = []
    discovery_drift = {"kind": None}
    admin_drift = {"value": False}

    def snapshot(path: str, label: str) -> dict[str, Any]:
        assert path == git_dir
        snapshot_calls.append((path, label))
        return {
            **admin_identity,
            "ino": admin_identity["ino"]
            + int(admin_drift["value"] and label.endswith("replay")),
        }

    pinned = [f"--git-dir={git_dir}", f"--work-tree={worktree}"]
    expected_arguments = [
        ["-C", worktree, "rev-parse", "--show-toplevel"],
        ["-C", worktree, "rev-parse", "--absolute-git-dir"],
        [*pinned, "rev-parse", "HEAD"],
        [*pinned, "rev-parse", "HEAD^{tree}"],
        [*pinned, "status", "--porcelain=v1", "--untracked-files=all"],
        [*pinned, "rev-parse", "--show-toplevel"],
        [*pinned, "rev-parse", "--absolute-git-dir"],
    ]
    expected_labels = [
        "attempt-2 discovered worktree root replay",
        "attempt-2 discovered Git directory replay",
        "attempt-2 HEAD replay",
        "attempt-2 tree replay",
        "attempt-2 cleanliness replay",
        "attempt-2 pinned worktree root replay",
        "attempt-2 pinned Git directory replay",
    ]

    def git_readonly(_contract, arguments, label, identity):
        args = list(arguments)
        git_calls.append((args, label, identity))
        assert identity == expected_identity
        if args == expected_arguments[0]:
            value = "/synthetic/drift" if discovery_drift["kind"] == "toplevel" else worktree
            return (value + "\n").encode("utf-8")
        if args == expected_arguments[1]:
            value = "/synthetic/drift" if discovery_drift["kind"] == "git-dir" else git_dir
            return (value + "\n").encode("utf-8")
        if args == expected_arguments[2]:
            return (protected["attempt2_worktree_commit"] + "\n").encode("ascii")
        if args == expected_arguments[3]:
            return (protected["attempt2_worktree_tree"] + "\n").encode("ascii")
        if args == expected_arguments[4]:
            return b""
        if args == expected_arguments[5]:
            value = (
                "/synthetic/drift"
                if discovery_drift["kind"] == "pinned-toplevel"
                else worktree
            )
            return (value + "\n").encode("utf-8")
        if args == expected_arguments[6]:
            value = (
                "/synthetic/drift"
                if discovery_drift["kind"] == "pinned-git-dir"
                else git_dir
            )
            return (value + "\n").encode("utf-8")
        raise AssertionError(f"unexpected attempt-2 Git argv: {args!r}")

    authorization_raw = b"synthetic-attempt2-owner-authorization"
    protected["attempt2_owner_authorization_raw_sha256"] = sha256(
        authorization_raw
    ).hexdigest()
    monkeypatch.setattr(tool, "_path_snapshot", snapshot)
    monkeypatch.setattr(tool, "_git_readonly", git_readonly)
    monkeypatch.setattr(tool.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        tool,
        "_read_regular_nofollow",
        lambda path, label, maximum: authorization_raw
        if (
            path == Path(protected["attempt2_owner_authorization_path"])
            and label == "attempt-2 owner authorization"
            and maximum == 1 << 20
        )
        else (_ for _ in ()).throw(AssertionError("authorization replay differs")),
    )
    monkeypatch.setattr(tool.os.path, "lexists", lambda _path: False)
    tool._verify_attempt2_protected_state(frozen, expected_identity)
    assert [row[0] for row in git_calls] == expected_arguments
    assert [row[1] for row in git_calls] == expected_labels
    assert snapshot_calls == [
        (git_dir, "attempt-2 Git admin directory"),
        (git_dir, "attempt-2 Git admin directory replay"),
    ]

    for drift, expected_count in (("toplevel", 2), ("git-dir", 2)):
        discovery_drift["kind"] = drift
        git_calls.clear()
        snapshot_calls.clear()
        with pytest.raises(
            tool.ShadowNonadoptionTrialError,
            match="attempt-2 discovered Git identity differs",
        ):
            tool._verify_attempt2_protected_state(frozen, expected_identity)
        assert len(git_calls) == expected_count
        assert snapshot_calls == [(git_dir, "attempt-2 Git admin directory")]
    discovery_drift["kind"] = None

    for drift in ("pinned-toplevel", "pinned-git-dir"):
        discovery_drift["kind"] = drift
        git_calls.clear()
        snapshot_calls.clear()
        with pytest.raises(
            tool.ShadowNonadoptionTrialError, match="attempt-2 worktree changed"
        ):
            tool._verify_attempt2_protected_state(frozen, expected_identity)
        assert [row[0] for row in git_calls] == expected_arguments
        assert [row[1] for row in git_calls] == expected_labels
        assert snapshot_calls == [
            (git_dir, "attempt-2 Git admin directory"),
            (git_dir, "attempt-2 Git admin directory replay"),
        ]
    discovery_drift["kind"] = None

    git_calls.clear()
    admin_drift["value"] = True
    with pytest.raises(tool.ShadowNonadoptionTrialError, match="attempt-2 worktree changed"):
        tool._verify_attempt2_protected_state(frozen, expected_identity)
    assert [row[0] for row in git_calls] == expected_arguments


def test_construction_authority_does_not_promote_execution(contract) -> None:
    boundary = contract["authority_boundary"]
    assert boundary["contract_document_source_tool_and_test_implementation_authorized"] is True
    assert boundary["source_runner_implementation_authorized_by_this_release"] is True
    for key in (
        "shadow_q05b_mechanics_trial_execution_authorized",
        "attempt2_or_attempt3_authorized",
        "retry_authorized",
        "production_access_authorized",
        "meta_bridge_access_authorized",
        "meta_adoption_authorized",
        "meta_promotion_authorized",
        "meta_current_pointer_authorized",
        "hegel_q1_authorized",
        "active_transition_authorized",
        "efficacy_claim_authorized",
        "c1_claim_authorized",
        "c2_claim_authorized",
        "network_access_authorized",
    ):
        assert boundary[key] is False
    outcome = contract["outcome_policy"]
    assert outcome["single_trial_establishes_v3_efficacy"] is False
    assert outcome["production_consumption_allowed"] is False
    assert outcome["artifact_export_allowed"] is False
    assert "MECHANICS_ONLY" in outcome["claim_ceiling"]


def test_capsule_and_ledger_schemas_are_exact_and_one_shot(contract) -> None:
    capsule = contract["capsule_policy"]
    assert capsule["capsule_record_name"] is None
    assert "IN_MEMORY_REDERIVATION" in capsule["capsule_storage_policy"]
    assert len(capsule["exact_fields"]) == len(set(capsule["exact_fields"]))
    assert capsule["exact_fields"][-1] == "capsule_digest"
    assert capsule["fixed_metric_registry"] == {
        "mechanics_only": True,
        "q05b_expected_predicate_count": 20,
        "q05b_expected_predicate_mask": 1048575,
        "q1_expected_state": "NOT_RUN",
        "v3_efficacy_metric_present": False,
    }
    ledger = contract["trial_ledger_policy"]
    assert ledger["consume_before_supervisor"] is True
    assert ledger["consume_record_is_sole_authorization_spend"] is True
    assert ledger["retry_after_consume_allowed"] is False
    assert ledger["state_transition_policy"][-1].startswith(
        "ANY_CONSUMED_OR_CONSUMPTION_UNCERTAIN_STATE_FORBIDS_RETRY"
    )
    for key in ("intent_record_schema", "consumed_record_schema", "success_record_schema"):
        fields = ledger[key]["exact_fields"]
        assert len(fields) == len(set(fields))
        assert fields[-1] == "record_digest"
    failure_fields = contract["failure_state_policy"]["receipt_exact_fields"]
    assert len(failure_fields) == len(set(failure_fields))
    assert failure_fields[-1] == "receipt_digest"


def test_fixed_roots_are_pairwise_disjoint_and_attempt2_paths_are_protected(contract) -> None:
    source = contract["isolated_source_policy"]
    auth = Path(contract["owner_authorization_policy"]["authorization_root"])
    ledger = Path(contract["trial_ledger_policy"]["ledger_root"])
    clone = Path(source["clone_root"])
    cache = Path(contract["actual_execution_policy"]["cargo_cache_source"])
    values = (auth, ledger, clone, cache)
    assert len({os.fspath(value) for value in values}) == 4
    for left in values:
        for right in values:
            if left == right:
                continue
            assert left not in right.parents
    protected = contract["protected_state_policy"]
    forbidden = set(protected["forbidden_exact_roots"])
    assert os.fspath(OLD_ATTEMPT2_WORKTREE) in forbidden
    assert protected["attempt2_ledger_root"] in forbidden
    assert "/home/erzhu419/.local/state/hegel-machine/rust-cargo-cache" in forbidden
    assert protected["attempt2_owner_authorization_digest"].startswith("sha256:")
    assert protected["protected_state_mutation_allowed"] is False


def test_path_boundaries_never_snapshot_meta_or_production_roots(
    monkeypatch, contract, tool
) -> None:
    frozen = _resign_contract(contract)
    protected = frozen["protected_state_policy"]
    meta_or_production_roots = {
        path
        for path in protected["forbidden_exact_roots"]
        if path.startswith("/home/erzhu419/.local/state/kg-op/")
    }
    assert len(meta_or_production_roots) == 4
    assert protected["meta_or_production_path_open_allowed"] is False
    snapshots: list[str] = []
    lexical_pairs: list[tuple[str, str]] = []
    real_ancestor_check = tool._path_is_ancestor_or_equal

    def snapshot(path: str, _label: str) -> dict[str, Any]:
        assert path not in meta_or_production_roots
        snapshots.append(path)
        return {
            "path": path,
            "exists": False,
            "existing_prefix": "/",
            "mode": None,
            "uid": None,
            "nlink": None,
            "dev": None,
            "ino": None,
            "mnt_id": None,
        }

    def lexical(left: str, right: str) -> bool:
        lexical_pairs.append((left, right))
        return real_ancestor_check(left, right)

    monkeypatch.setattr(tool, "_path_snapshot", snapshot)
    monkeypatch.setattr(tool, "_path_is_ancestor_or_equal", lexical)
    tool._verify_path_boundaries(frozen)

    source = frozen["isolated_source_policy"]
    actual = frozen["actual_execution_policy"]
    expected_snapshots = [
        frozen["owner_authorization_policy"]["authorization_root"],
        frozen["trial_ledger_policy"]["ledger_root"],
        source["clone_root"],
        actual["cargo_cache_source"],
        actual["artifact_path"],
        protected["attempt2_worktree_root"],
        protected["attempt2_ledger_root"],
        actual["isolated_cargo_cache_policy"]["copy_source"],
        protected["attempt2_artifact_path"],
        protected["attempt2_owner_authorization_path"],
        source["clone_source_repository"],
    ]
    assert snapshots == expected_snapshots
    assert not meta_or_production_roots.intersection(snapshots)
    for root in meta_or_production_roots:
        assert any(root in pair for pair in lexical_pairs)


def test_path_snapshot_returns_same_fd_mode_uid_nlink_without_lstat(
    monkeypatch, tool
) -> None:
    directory_a = types.SimpleNamespace(
        st_dev=1, st_ino=11, st_mode=0o040755, st_uid=1000, st_nlink=3
    )
    directory_b = types.SimpleNamespace(
        st_dev=2, st_ino=22, st_mode=0o040700, st_uid=1001, st_nlink=5
    )
    opened: list[tuple[object, int | None]] = []
    closed: list[int] = []
    stat_calls: list[tuple[str, int, bool]] = []
    fstat_calls: list[int] = []
    next_fds = iter((10, 11, 12))

    def open_path(path, _flags, *, dir_fd=None):
        opened.append((path, dir_fd))
        return next(next_fds)

    def component_stat(part, *, dir_fd, follow_symlinks):
        stat_calls.append((part, dir_fd, follow_symlinks))
        return {"a": directory_a, "b": directory_b}[part]

    def held_fstat(fd: int):
        fstat_calls.append(fd)
        return {11: directory_a, 12: directory_b}[fd]

    monkeypatch.setattr(tool.os, "open", open_path)
    monkeypatch.setattr(tool.os, "stat", component_stat)
    monkeypatch.setattr(tool.os, "fstat", held_fstat)
    monkeypatch.setattr(tool.os, "close", closed.append)
    monkeypatch.setattr(
        tool.os,
        "lstat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("_path_snapshot used pathname lstat")
        ),
    )
    monkeypatch.setattr(
        tool,
        "_fd_mount_id",
        lambda fd, label: 77
        if (fd, label) == (12, "same-fd KAT")
        else (_ for _ in ()).throw(AssertionError("mount id used a different fd")),
    )
    assert tool._path_snapshot("/a/b", "same-fd KAT") == {
        "path": "/a/b",
        "exists": True,
        "existing_prefix": "/a/b",
        "mode": directory_b.st_mode,
        "uid": directory_b.st_uid,
        "nlink": directory_b.st_nlink,
        "dev": directory_b.st_dev,
        "ino": directory_b.st_ino,
        "mnt_id": 77,
    }
    assert opened == [("/", None), ("a", 10), ("b", 11)]
    assert stat_calls == [("a", 10, False), ("b", 11, False)]
    assert fstat_calls == [11, 12]
    assert closed == [10, 11, 12]


def test_frozen_git_replay_uses_only_contract_source_repository(
    monkeypatch, contract, tool
) -> None:
    frozen = _resign_contract(contract)
    source = frozen["isolated_source_policy"]
    actual = frozen["actual_execution_policy"]
    contract_only_repository = "/contract-only/source-repository"
    source["clone_source_repository"] = contract_only_repository
    assert source["git_source_forbidden_metadata_relative_paths"] == [
        ".git/shallow",
        ".git/objects/info/alternates",
        ".git/objects/info/http-alternates",
        ".git/info/grafts",
        ".git/refs/replace",
    ]
    expected_git_identity = _executable_identity(source["git_executable_path"], "b")
    object_digests = {
        actual["old_supervisor_path"]: actual["old_supervisor_raw_sha256"],
        **source["actual_supervisor_config_raw_sha256"],
    }
    calls: list[tuple[list[str], str, Mapping[str, Any]]] = []
    synthetic_digests: dict[bytes, str] = {}
    metadata_snapshots: list[str] = []
    present_metadata = {"path": None}

    def path_snapshot(path: str, label: str) -> dict[str, Any]:
        metadata_snapshots.append(path)
        relative = Path(path).relative_to(contract_only_repository).as_posix()
        assert relative in source["git_source_forbidden_metadata_relative_paths"]
        assert label == f"source Git forbidden metadata {relative}"
        return {
            "path": path,
            "exists": path == present_metadata["path"],
            "existing_prefix": contract_only_repository,
            "mode": None,
            "uid": None,
            "nlink": None,
            "dev": None,
            "ino": None,
            "mnt_id": None,
        }

    def git_readonly(_contract, arguments, label, identity):
        args = list(arguments)
        calls.append((args, label, identity))
        assert args[:2] == ["-C", contract_only_repository]
        if args[2:4] == ["rev-parse", f"{source['checkout_commit']}^{{tree}}"]:
            return (source["checkout_tree"] + "\n").encode("ascii")
        if args[2:4] == ["rev-parse", f"{source['checkout_commit']}^"]:
            return (source["checkout_commit_parent"] + "\n").encode("ascii")
        if args[2] == "show":
            revision_and_path = args[3]
            _revision, relative = revision_and_path.split(":", 1)
            raw = f"synthetic:{revision_and_path}".encode("ascii")
            synthetic_digests[raw] = object_digests[relative]
            return raw
        if args[2:7] == [
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            source["checkout_commit"],
        ]:
            return b""
        raise AssertionError(f"unexpected frozen Git replay argv: {args!r}")

    class SyntheticHash:
        def __init__(self, raw: bytes) -> None:
            self.raw = raw

        def hexdigest(self) -> str:
            return synthetic_digests[self.raw]

    monkeypatch.setattr(tool, "_path_snapshot", path_snapshot)
    monkeypatch.setattr(tool, "_git_readonly", git_readonly)
    monkeypatch.setattr(tool, "sha256", lambda raw=b"": SyntheticHash(bytes(raw)))
    tool._verify_frozen_git_objects(frozen, expected_git_identity)
    assert "SOURCE_REPOSITORY" not in tool.__dict__
    assert "SOURCE_REPOSITORY" not in TOOL_PATH.read_text(encoding="utf-8")
    assert all(identity == expected_git_identity for _args, _label, identity in calls)
    assert len(calls) == 2 + len(object_digests) + 1 + 1
    expected_metadata_paths = [
        os.fspath(Path(contract_only_repository) / relative)
        for relative in source["git_source_forbidden_metadata_relative_paths"]
    ]
    assert metadata_snapshots == expected_metadata_paths

    git_call_count = len(calls)
    for forbidden_path in expected_metadata_paths:
        present_metadata["path"] = forbidden_path
        metadata_snapshots.clear()
        with pytest.raises(
            tool.ShadowNonadoptionTrialError,
            match="source Git forbidden metadata exists",
        ):
            tool._verify_frozen_git_objects(frozen, expected_git_identity)
        assert calls[git_call_count:] == []
        assert metadata_snapshots[-1] == forbidden_path
    present_metadata["path"] = None


def test_clone_replay_uses_path_snapshots_and_same_fd_alternates_stat(
    monkeypatch, contract, tool
) -> None:
    frozen = _resign_contract(contract)
    source = frozen["isolated_source_policy"]
    actual = frozen["actual_execution_policy"]
    clone_root = Path(source["clone_root"])
    git_identity = _executable_identity(source["git_executable_path"], "b")
    capsule = {
        "artifact_path": actual["artifact_path"],
        "clone_git_commit": source["checkout_commit"],
        "clone_git_tree": source["checkout_tree"],
        "clone_git_parent": source["checkout_commit_parent"],
        "old_supervisor_raw_sha256": actual["old_supervisor_raw_sha256"],
        "git_executable_identity": git_identity,
    }
    snapshot_calls: list[tuple[str, str]] = []
    reader_calls: list[tuple[Path, str, int | None, bool]] = []
    digest_by_raw: dict[bytes, str] = {}
    directory_mode = 0o040700
    snapshots = {
        os.fspath(clone_root): (10, 10),
        os.fspath(Path(source["clone_source_repository"]) / ".git/objects"): (20, 20),
        os.fspath(clone_root / ".git/objects"): (30, 30),
        source["clone_git_dir"]: (40, 40),
    }
    forbidden_metadata_paths = {
        os.fspath(clone_root / relative)
        for relative in source["git_clone_forbidden_metadata_relative_paths"]
    }
    assert source["git_clone_forbidden_metadata_relative_paths"] == [
        ".git/shallow",
        ".git/objects/info/http-alternates",
        ".git/info/grafts",
        ".git/info/attributes",
        ".git/refs/replace",
    ]
    present_metadata = {"path": None}

    def path_snapshot(path: str, label: str) -> dict[str, Any]:
        snapshot_calls.append((path, label))
        if path in forbidden_metadata_paths:
            relative = Path(path).relative_to(clone_root).as_posix()
            assert label in (
                f"pre-mutator forbidden Git metadata {relative}",
                f"clone Git forbidden metadata {relative}",
            )
            return {
                "path": path,
                "exists": path == present_metadata["path"],
                "existing_prefix": os.fspath(clone_root),
                "mode": None,
                "uid": None,
                "nlink": None,
                "dev": None,
                "ino": None,
                "mnt_id": None,
            }
        dev, ino = snapshots[path]
        observed_mode = (
            0o040000 | int(source["clone_git_dir_mode"], 8)
            if path == source["clone_git_dir"]
            else directory_mode
        )
        return {
            "path": path,
            "exists": True,
            "existing_prefix": path,
            "mode": observed_mode,
            "uid": 1000,
            "nlink": 2,
            "dev": dev,
            "ino": ino,
            "mnt_id": dev,
        }

    regular_stat = types.SimpleNamespace(st_mode=0o100644, st_uid=1000, st_nlink=1)

    def read_bound(path, label, maximum=None, *, with_stat=False):
        path = Path(path)
        reader_calls.append((path, label, maximum, with_stat))
        assert with_stat is True
        if label in ("Git alternates", "pre-mutator Git alternates"):
            assert maximum == 4096
            return source["git_postclone_alternates_exact_bytes"].encode("ascii"), regular_stat
        if label == "clone-local supervisor":
            raw = b"synthetic clone-local supervisor"
            digest_by_raw[raw] = actual["old_supervisor_raw_sha256"]
            return raw, regular_stat
        prefix = "clone-local config "
        assert label.startswith(prefix)
        relative = label[len(prefix) :]
        raw = f"synthetic config:{relative}".encode("utf-8")
        digest_by_raw[raw] = source["actual_supervisor_config_raw_sha256"][relative]
        return raw, regular_stat

    def git_readonly(_contract, _arguments, label, identity):
        assert identity == git_identity
        return {
            "clone HEAD replay": (source["checkout_commit"] + "\n").encode("ascii"),
            "clone tree replay": (source["checkout_tree"] + "\n").encode("ascii"),
            "clone parent replay": (source["checkout_commit_parent"] + "\n").encode("ascii"),
            "clone tracked cleanliness replay": b"",
            "clone ignored-and-unignored other-file replay": b"",
            "clone remote replay": b"",
            "clone detached-HEAD replay": b"",
            "clone tag replay": b"",
            "clone worktree root replay": (source["clone_git_toplevel"] + "\n").encode(
                "utf-8"
            ),
            "clone Git directory replay": (source["clone_git_dir"] + "\n").encode(
                "utf-8"
            ),
        }[label]

    class SyntheticHash:
        def __init__(self, raw: bytes) -> None:
            self.raw = raw

        def hexdigest(self) -> str:
            return digest_by_raw[self.raw]

    monkeypatch.setattr(tool, "_path_snapshot", path_snapshot)
    monkeypatch.setattr(tool, "_read_regular_nofollow", read_bound)
    monkeypatch.setattr(tool, "_git_readonly", git_readonly)
    monkeypatch.setattr(tool, "sha256", lambda raw=b"": SyntheticHash(bytes(raw)))
    monkeypatch.setattr(tool.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(tool.os.path, "lexists", lambda _path: False)
    monkeypatch.setattr(
        tool.os,
        "lstat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("clone replay used pathname lstat")
        ),
    )
    tool._verify_isolated_clone(frozen, capsule, require_artifact_absent=True)
    assert snapshot_calls == [
        (os.fspath(clone_root), "isolated clone root"),
        (source["clone_git_dir"], "clone Git directory"),
        *[
            (
                os.fspath(clone_root / relative),
                f"pre-mutator forbidden Git metadata {relative}",
            )
            for relative in source["git_clone_forbidden_metadata_relative_paths"]
        ],
        (
            os.fspath(Path(source["clone_source_repository"]) / ".git/objects"),
            "shared source Git object directory",
        ),
        (os.fspath(clone_root / ".git/objects"), "clone Git object directory"),
        *[
            (
                os.fspath(clone_root / relative),
                f"clone Git forbidden metadata {relative}",
            )
            for relative in source["git_clone_forbidden_metadata_relative_paths"]
        ],
    ]
    alternates_calls = [row for row in reader_calls if row[1] == "Git alternates"]
    assert alternates_calls == [
        (clone_root / ".git/objects/info/alternates", "Git alternates", 4096, True)
    ]
    assert [row for row in reader_calls if row[1] == "pre-mutator Git alternates"] == [
        (
            clone_root / ".git/objects/info/alternates",
            "pre-mutator Git alternates",
            4096,
            True,
        )
    ]

    for forbidden_path in sorted(forbidden_metadata_paths):
        present_metadata["path"] = forbidden_path
        with pytest.raises(
            tool.ShadowNonadoptionTrialError,
            match="forbidden Git metadata exists",
        ):
            tool._verify_isolated_clone(
                frozen, capsule, require_artifact_absent=True
            )
    present_metadata["path"] = None

    original_git_readonly = git_readonly
    artifact_relative = Path(capsule["artifact_path"]).relative_to(clone_root).as_posix()

    def postrun_git_readonly(observed_contract, arguments, label, identity):
        if label == "clone ignored-and-unignored other-file replay":
            return artifact_relative.encode("utf-8") + b"\x00"
        return original_git_readonly(observed_contract, arguments, label, identity)

    monkeypatch.setattr(tool, "_git_readonly", postrun_git_readonly)
    tool._verify_isolated_clone(frozen, capsule, require_artifact_absent=False)

    def extra_ignored_file(observed_contract, arguments, label, identity):
        if label == "clone ignored-and-unignored other-file replay":
            return (
                artifact_relative.encode("utf-8")
                + b"\x00synthetic-ignored-extra.bin\x00"
            )
        return original_git_readonly(observed_contract, arguments, label, identity)

    monkeypatch.setattr(tool, "_git_readonly", extra_ignored_file)
    with pytest.raises(
        tool.ShadowNonadoptionTrialError, match="isolated clone identity differs"
    ):
        tool._verify_isolated_clone(frozen, capsule, require_artifact_absent=False)

    for drift_label in ("clone worktree root replay", "clone Git directory replay"):
        def drifted_git_readonly(
            observed_contract,
            arguments,
            label,
            identity,
            *,
            selected=drift_label,
        ):
            if label == selected:
                return b"/synthetic/drift\n"
            return original_git_readonly(observed_contract, arguments, label, identity)

        monkeypatch.setattr(tool, "_git_readonly", drifted_git_readonly)
        with pytest.raises(
            tool.ShadowNonadoptionTrialError, match="isolated clone identity differs"
        ):
            tool._verify_isolated_clone(
                frozen, capsule, require_artifact_absent=True
            )


def test_clone_git_mutator_boundary_replays_same_nofollow_admin_dir_and_metadata(
    monkeypatch, contract, tool
) -> None:
    frozen = _resign_contract(contract)
    source = frozen["isolated_source_policy"]
    clone_root = Path(source["clone_root"])
    git_dir = Path(source["clone_git_dir"])
    assert git_dir == clone_root / ".git"
    assert source["git_clone_forbidden_metadata_relative_paths"] == [
        ".git/shallow",
        ".git/objects/info/http-alternates",
        ".git/info/grafts",
        ".git/info/attributes",
        ".git/refs/replace",
    ]
    git_dir_identity = {
        "path": os.fspath(git_dir),
        "exists": True,
        "existing_prefix": os.fspath(git_dir),
        "mode": int(source["clone_git_dir_mode"], 8) | 0o040000,
        "uid": 1000,
        "nlink": 2,
        "dev": 1,
        "ino": 2,
        "mnt_id": 3,
    }
    present_forbidden = {"path": None}
    drift_git_dir = {"value": False}
    snapshot_calls: list[tuple[str, str]] = []

    def snapshot(path: str, label: str) -> dict[str, Any]:
        snapshot_calls.append((path, label))
        if path == os.fspath(git_dir):
            return {
                **git_dir_identity,
                "ino": git_dir_identity["ino"] + int(drift_git_dir["value"]),
            }
        relative = Path(path).relative_to(clone_root).as_posix()
        assert relative in source["git_clone_forbidden_metadata_relative_paths"]
        assert label == f"pre-mutator forbidden Git metadata {relative}"
        return {
            "path": path,
            "exists": path == present_forbidden["path"],
            "existing_prefix": os.fspath(git_dir),
            "mode": None,
            "uid": None,
            "nlink": None,
            "dev": None,
            "ino": None,
            "mnt_id": None,
        }

    regular_stat = types.SimpleNamespace(
        st_mode=0o100644, st_uid=1000, st_nlink=1
    )
    alternates_reads: list[tuple[Path, str, int | None, bool]] = []

    def read_alternates(path, label, maximum=None, *, with_stat=False):
        alternates_reads.append((Path(path), label, maximum, with_stat))
        return (
            source["git_postclone_alternates_exact_bytes"].encode("ascii"),
            regular_stat,
        )

    monkeypatch.setattr(tool, "_path_snapshot", snapshot)
    monkeypatch.setattr(tool, "_read_regular_nofollow", read_alternates)
    monkeypatch.setattr(tool.os, "geteuid", lambda: 1000)
    first = tool._clone_git_mutator_boundary(frozen, None)
    assert first == git_dir_identity
    assert tool._clone_git_mutator_boundary(frozen, first) == first
    assert alternates_reads == [
        (
            git_dir / "objects/info/alternates",
            "pre-mutator Git alternates",
            4096,
            True,
        ),
        (
            git_dir / "objects/info/alternates",
            "pre-mutator Git alternates",
            4096,
            True,
        ),
    ]

    drift_git_dir["value"] = True
    with pytest.raises(
        tool.ShadowNonadoptionTrialError,
        match="Git directory changed between mutators",
    ):
        tool._clone_git_mutator_boundary(frozen, first)
    drift_git_dir["value"] = False

    attributes_path = os.fspath(clone_root / ".git/info/attributes")
    present_forbidden["path"] = attributes_path
    with pytest.raises(
        tool.ShadowNonadoptionTrialError,
        match="pre-mutator forbidden Git metadata exists: .git/info/attributes",
    ):
        tool._clone_git_mutator_boundary(frozen, first)
    assert (os.fspath(git_dir), "clone Git directory") in snapshot_calls


def test_cargo_manifest_and_copy_replay_regular_bytes_with_same_fd_stat(
    monkeypatch, contract, tool
) -> None:
    frozen = _resign_contract(contract)
    policy = frozen["actual_execution_policy"]["isolated_cargo_cache_policy"]
    source_root = Path(policy["copy_source"])
    target_root = Path(policy["copy_target"])
    relative = "registry/cache/crate.bin"
    raw = b"synthetic-cargo-cache-entry"
    directory_stat = types.SimpleNamespace(st_mode=0o040700)
    regular_stat = types.SimpleNamespace(
        st_mode=0o100600,
        st_uid=1000,
        st_nlink=1,
        st_size=len(raw),
    )
    manifest_reads: list[tuple[Path, str, int | None, bool, bool]] = []
    lstat_paths: list[Path] = []

    def lstat(path: object) -> Any:
        observed = Path(path)
        lstat_paths.append(observed)
        assert observed == source_root
        return directory_stat

    def manifest_reader(
        path: Path,
        label: str,
        maximum: int | None = None,
        *,
        with_stat: bool = False,
        reject_xattrs: bool = False,
    ):
        manifest_reads.append(
            (Path(path), label, maximum, with_stat, reject_xattrs)
        )
        assert Path(path) == source_root / relative
        return raw, regular_stat

    monkeypatch.setattr(tool.os, "lstat", lstat)
    monkeypatch.setattr(
        tool.os,
        "walk",
        lambda root, **_kwargs: [
            (
                os.fspath(Path(root) / Path(relative).parent),
                [],
                [Path(relative).name],
            )
        ],
    )
    monkeypatch.setattr(tool.os, "listxattr", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(tool.os.path, "lexists", lambda _path: False)
    monkeypatch.setattr(tool, "_read_regular_nofollow", manifest_reader)
    cargo_identity, rows = tool._tree_manifest(frozen, source_root)
    assert lstat_paths == [source_root]
    assert manifest_reads == [
        (
            source_root / relative,
            f"Cargo cache file {Path(relative).name}",
            None,
            True,
            True,
        )
    ]
    assert rows == [
        {
            "relative_path": relative,
            "file_kind": "REGULAR",
            "mode": 0o600,
            "size": len(raw),
            "raw_sha256": sha256(raw).hexdigest(),
        }
    ]

    copy_reads: list[tuple[Path, str, int | None, bool, bool]] = []
    mkdir_calls: list[Path] = []
    open_calls: list[str] = []
    writes: list[tuple[int, bytes]] = []
    closed: list[int] = []
    next_fds = iter((81, 82, 83))

    def copy_reader(
        path: Path,
        label: str,
        maximum: int | None = None,
        *,
        with_stat: bool = False,
        reject_xattrs: bool = False,
    ):
        copy_reads.append((Path(path), label, maximum, with_stat, reject_xattrs))
        return raw, regular_stat

    def mkdir(path: Path, *_args: Any, **_kwargs: Any) -> None:
        mkdir_calls.append(Path(path))

    def open_path(path: object, _flags: int, _mode: int | None = None) -> int:
        open_calls.append(os.fspath(path))
        return next(next_fds)

    with monkeypatch.context() as scoped:
        scoped.setattr(tool, "_verify_path_boundaries", lambda _contract: None)
        scoped.setattr(tool.os.path, "lexists", lambda _path: False)
        scoped.setattr(Path, "mkdir", mkdir)
        scoped.setattr(Path, "rglob", lambda _path, _pattern: [])
        scoped.setattr(tool.os, "open", open_path)
        scoped.setattr(
            tool.os,
            "write",
            lambda fd, data: writes.append((fd, bytes(data))) or len(data),
        )
        scoped.setattr(tool.os, "fchmod", lambda *_args: None)
        scoped.setattr(tool.os, "fsync", lambda *_args: None)
        scoped.setattr(tool.os, "close", closed.append)
        scoped.setattr(tool, "_read_regular_nofollow", copy_reader)
        scoped.setattr(
            tool,
            "_tree_manifest",
            lambda _contract, root, *, require_target_absent: (
                cargo_identity,
                deepcopy(rows),
            )
            if (root, require_target_absent) == (source_root, False)
            else (_ for _ in ()).throw(AssertionError("source replay differs")),
        )
        scoped.setattr(
            tool,
            "_tree_manifest_for_target",
            lambda _contract, root: (cargo_identity, deepcopy(rows))
            if root == target_root
            else (_ for _ in ()).throw(AssertionError("target replay differs")),
        )
        scoped.setattr(
            tool.os,
            "lstat",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("Cargo regular copy used pathname lstat")
            ),
        )
        tool._prepare_isolated_cargo_cache(frozen, rows)

    assert copy_reads == [
        (
            source_root / relative,
            f"Cargo copy source {relative}",
            None,
            True,
            True,
        )
    ]
    assert writes == [(82, raw)]
    assert closed == [81, 82, 83]
    assert mkdir_calls == [target_root, (target_root / relative).parent]
    assert open_calls == [
        os.fspath(target_root.parent),
        os.fspath(target_root / relative),
        os.fspath(target_root),
    ]


def test_public_surfaces_and_signatures_are_frozen(contract, core, tool) -> None:
    surface = contract["source_surface_policy"]
    assert tuple(core.__all__) == (
        "ShadowNonadoptionTrialCoreError",
        *surface["core_public_python_apis"],
    )
    assert tuple(tool.__all__) == (
        "ShadowNonadoptionTrialError",
        *surface["public_python_apis"],
    )
    for row in surface["core_public_python_api_signatures"]:
        assert list(inspect.signature(getattr(core, row["name"])).parameters) == row[
            "parameters"
        ]
    for row in surface["public_python_api_signatures"]:
        signature = inspect.signature(getattr(tool, row["name"]))
        expected_names = [parameter.replace("=None", "") for parameter in row["parameters"]]
        assert list(signature.parameters) == expected_names
        if row["name"] == "main":
            assert signature.parameters["argv"].default is None
        else:
            assert all(
                parameter.default is inspect.Parameter.empty
                for parameter in signature.parameters.values()
            )


def test_core_rejects_nonexact_json_runtime_types(core) -> None:
    class DictSubclass(dict):
        pass

    for value in (
        ("tuple",),
        DictSubclass({"x": 1}),
        {"x": ("tuple",)},
        {"x": float("nan")},
        {1: "non-string key"},
    ):
        with pytest.raises(core.ShadowNonadoptionTrialCoreError):
            core.canonical_json_bytes_v1(value)


def test_contract_parser_rejects_duplicate_keys_and_noncanonical_raw(tool) -> None:
    with pytest.raises(tool.ShadowNonadoptionTrialError, match="duplicate JSON key"):
        tool._parse_canonical_json(b'{"a":1,"a":2}\n', "duplicate KAT")
    with pytest.raises(tool.ShadowNonadoptionTrialError, match="not canonical"):
        tool._parse_canonical_json(b'{ "a": 1 }\n', "whitespace KAT")
    assert tool._parse_canonical_json(b'{"a":1}\n', "canonical KAT") == {"a": 1}


def test_producer_free_failure_policy_rejects_rehashed_matrix_widening(
    contract, core
) -> None:
    frozen = _resign_contract(contract)
    assert core.validate_shadow_nonadoption_trial_contract_v1(frozen) == frozen
    widened = deepcopy(frozen)
    matrix = widened["failure_state_policy"]["state_fact_matrix"]
    matrix[2]["allowed_facts"]["attempt2_state_touched"].append(True)
    widened = _resign_contract(widened)
    with pytest.raises(
        core.ShadowNonadoptionTrialCoreError,
        match="producer-free frozen policy",
    ):
        core.validate_shadow_nonadoption_trial_contract_v1(widened)


def test_failure_registry_is_exactly_ten_states_and_all_phases_are_mapped(
    contract, tool
) -> None:
    frozen = _resign_contract(contract)
    policy = frozen["failure_state_policy"]
    assert tuple(policy["states"]) == FAILURE_STATES
    assert tuple(policy["matrix_fact_fields"]) == FAILURE_FACT_FIELDS
    assert tuple(row["error_state"] for row in policy["state_fact_matrix"]) == FAILURE_STATES
    observed_phases: set[str] = set()
    for row in policy["state_fact_matrix"]:
        assert set(row) == {
            "error_state",
            "consumed_record_digest_policy",
            "supervisor_exit_code_policy",
            "supervisor_output_digest_policy",
            "phase_ids",
            "allowed_facts",
        }
        assert set(row["allowed_facts"]) == set(FAILURE_FACT_FIELDS)
        for phase in row["phase_ids"]:
            assert phase not in observed_phases
            observed_phases.add(phase)
            assert tool._failure_state_for_phase(phase) == row["error_state"]
    assert observed_phases == {
        "CREATE_LEDGER_ROOT",
        "PUBLISH_INTENT",
        "PREPARE_CLONE",
        "PREPARE_CARGO_CACHE",
        "PRECONSUMPTION_FULL_REPLAY",
        "PUBLISH_CONSUMED",
        "SPAWN_PREPARE",
        "SUPERVISOR_SPAWN",
        "SUPERVISOR_WAIT",
        "TIMEOUT_CLEANUP",
        "DOCKER_CLEANUP",
        "VALIDATE_EXIT",
        "VALIDATE_STDOUT",
        "VALIDATE_ARTIFACT",
        "POSTRUN_PROTECTED_REPLAY",
        "POSTRUN_DOCKER_ABSENCE",
        "PUBLISH_SUCCESS",
    }


def test_all_failure_states_phases_and_channels_have_positive_kats(
    contract, core
) -> None:
    frozen = _resign_contract(contract)
    transport_only_states = {
        "LEDGER_ROOT_OR_LOCK_PUBLICATION_UNCERTAIN_TERMINAL",
        "INTENT_PUBLICATION_UNCERTAIN_TERMINAL",
        "SHADOW_OUTCOME_LINK_PUBLICATION_UNCERTAIN",
    }
    for state in FAILURE_STATES:
        matrix = _failure_row(frozen, state)
        for phase in matrix["phase_ids"]:
            receipt = _failure_receipt(frozen, state, phase)
            transport_context = _failure_context(receipt, "TRANSPORT_ONLY")
            assert (
                core.validate_shadow_nonadoption_ledger_record_v1(
                    receipt, "failure", transport_context, frozen
                )
                == receipt
            )
            if state not in transport_only_states:
                ledger_context = _failure_context(receipt, "LEDGER_RECORD")
                assert (
                    core.validate_shadow_nonadoption_ledger_record_v1(
                        receipt, "failure", ledger_context, frozen
                    )
                    == receipt
                )


def test_every_failure_state_rejects_rehashed_phase_and_fact_mutations(
    contract, core
) -> None:
    frozen = _resign_contract(contract)
    for state in FAILURE_STATES:
        matrix = _failure_row(frozen, state)
        baseline = _failure_receipt(frozen, state)
        bad_phase = deepcopy(baseline)
        bad_phase["phase_id"] = "NOT_A_FROZEN_PHASE"
        bad_phase = _resign_record(frozen, "failure", bad_phase)
        with pytest.raises(core.ShadowNonadoptionTrialCoreError):
            core.validate_shadow_nonadoption_ledger_record_v1(
                bad_phase,
                "failure",
                _failure_context(bad_phase, "TRANSPORT_ONLY"),
                frozen,
            )
        for field in FAILURE_FACT_FIELDS:
            mutated = deepcopy(baseline)
            mutated[field] = _outside_allowed(matrix["allowed_facts"][field])
            mutated = _resign_record(frozen, "failure", mutated)
            with pytest.raises(core.ShadowNonadoptionTrialCoreError):
                core.validate_shadow_nonadoption_ledger_record_v1(
                    mutated,
                    "failure",
                    _failure_context(mutated, "TRANSPORT_ONLY"),
                    frozen,
                )


def test_failure_state_digest_exit_and_output_policy_mutations_are_rejected(
    contract, core
) -> None:
    frozen = _resign_contract(contract)
    for state in FAILURE_STATES:
        matrix = _failure_row(frozen, state)
        baseline = _failure_receipt(frozen, state)
        mutations: list[dict[str, Any]] = []

        consumed = deepcopy(baseline)
        if matrix["consumed_record_digest_policy"] == "NULL":
            consumed["consumed_record_digest"] = TAGGED_CONSUMED
        elif matrix["consumed_record_digest_policy"] == "TAGGED":
            consumed["consumed_record_digest"] = None
        else:
            consumed["consumed_record_digest"] = "not-a-tagged-digest"
        mutations.append(consumed)

        exit_mutation = deepcopy(baseline)
        exit_policy = matrix["supervisor_exit_code_policy"]
        exit_mutation["supervisor_exit_code"] = {
            "NULL": 0,
            "ZERO": 1,
            "NONZERO_INT": 0,
            "NULL_OR_INT": "0",
        }[exit_policy]
        mutations.append(exit_mutation)

        output_mutation = deepcopy(baseline)
        output_policy = matrix["supervisor_output_digest_policy"]
        output_mutation["supervisor_stdout_raw_sha256"] = (
            RAW_STDOUT if output_policy == "NULL" else None
        )
        if output_policy == "NULL_OR_RAW":
            output_mutation["supervisor_stdout_raw_sha256"] = "not-a-raw-digest"
        mutations.append(output_mutation)

        for mutation in mutations:
            mutation = _resign_record(frozen, "failure", mutation)
            with pytest.raises(core.ShadowNonadoptionTrialCoreError):
                core.validate_shadow_nonadoption_ledger_record_v1(
                    mutation,
                    "failure",
                    _failure_context(mutation, "TRANSPORT_ONLY"),
                    frozen,
                )


def test_consumed_digest_condition_and_both_supervisor_return_branches(
    contract, core
) -> None:
    frozen = _resign_contract(contract)
    publication = _failure_receipt(
        frozen, "CONSUMPTION_PUBLICATION_UNCERTAIN_TERMINAL"
    )
    publication["ledger_consumed"] = True
    publication["consumed_record_digest"] = TAGGED_CONSUMED
    publication = _resign_record(frozen, "failure", publication)
    assert core.validate_shadow_nonadoption_ledger_record_v1(
        publication,
        "failure",
        _failure_context(publication, "TRANSPORT_ONLY"),
        frozen,
    ) == publication

    missing_consumed_digest = deepcopy(publication)
    missing_consumed_digest["consumed_record_digest"] = None
    missing_consumed_digest = _resign_record(
        frozen, "failure", missing_consumed_digest
    )
    with pytest.raises(core.ShadowNonadoptionTrialCoreError):
        core.validate_shadow_nonadoption_ledger_record_v1(
            missing_consumed_digest,
            "failure",
            _failure_context(missing_consumed_digest, "TRANSPORT_ONLY"),
            frozen,
        )

    uncertain_return = _failure_receipt(
        frozen, "SUPERVISOR_RAISED_TIMED_OUT_OR_CLEANUP_UNCERTAIN"
    )
    assert uncertain_return["supervisor_returned"] is None
    assert uncertain_return["supervisor_exit_code"] is None
    assert uncertain_return["supervisor_stdout_raw_sha256"] is None
    assert uncertain_return["supervisor_stderr_raw_sha256"] is None
    assert core.validate_shadow_nonadoption_ledger_record_v1(
        uncertain_return,
        "failure",
        _failure_context(uncertain_return, "TRANSPORT_ONLY"),
        frozen,
    ) == uncertain_return

    exact_return = deepcopy(uncertain_return)
    exact_return["supervisor_returned"] = True
    exact_return["supervisor_exit_code"] = 0
    exact_return["supervisor_stdout_raw_sha256"] = RAW_STDOUT
    exact_return["supervisor_stderr_raw_sha256"] = RAW_STDERR
    exact_return = _resign_record(frozen, "failure", exact_return)
    assert core.validate_shadow_nonadoption_ledger_record_v1(
        exact_return,
        "failure",
        _failure_context(exact_return, "TRANSPORT_ONLY"),
        frozen,
    ) == exact_return


def test_failure_relational_invariants_reject_individually_allowed_bad_pairs(
    contract, core
) -> None:
    frozen = _resign_contract(contract)
    state = "SUPERVISOR_RAISED_TIMED_OUT_OR_CLEANUP_UNCERTAIN"
    baseline = _failure_receipt(frozen, state)

    returned_without_result = deepcopy(baseline)
    returned_without_result["supervisor_returned"] = True
    returned_without_result["supervisor_exit_code"] = None
    returned_without_result["supervisor_stdout_raw_sha256"] = None
    returned_without_result["supervisor_stderr_raw_sha256"] = None
    returned_without_result = _resign_record(
        frozen, "failure", returned_without_result
    )
    with pytest.raises(core.ShadowNonadoptionTrialCoreError):
        core.validate_shadow_nonadoption_ledger_record_v1(
            returned_without_result,
            "failure",
            _failure_context(returned_without_result, "TRANSPORT_ONLY"),
            frozen,
        )

    cleanup_claim_without_absence = deepcopy(baseline)
    cleanup_claim_without_absence["cleanup_complete"] = True
    cleanup_claim_without_absence["survivor_absence_verified"] = False
    cleanup_claim_without_absence = _resign_record(
        frozen, "failure", cleanup_claim_without_absence
    )
    with pytest.raises(core.ShadowNonadoptionTrialCoreError):
        core.validate_shadow_nonadoption_ledger_record_v1(
            cleanup_claim_without_absence,
            "failure",
            _failure_context(cleanup_claim_without_absence, "TRANSPORT_ONLY"),
            frozen,
        )


def test_failure_channel_rejects_invalid_or_uncertain_ledger_receipts(
    contract, core
) -> None:
    frozen = _resign_contract(contract)
    with pytest.raises(core.ShadowNonadoptionTrialCoreError, match="channel"):
        receipt = _failure_receipt(
            frozen, "INTENT_STAGED_PREPARATION_FAILED_CLOSED_UNCONSUMED"
        )
        core.validate_shadow_nonadoption_ledger_record_v1(
            receipt,
            "failure",
            _failure_context(receipt, "SIDE_CHANNEL"),
            frozen,
        )

    for state in (
        "LEDGER_ROOT_OR_LOCK_PUBLICATION_UNCERTAIN_TERMINAL",
        "INTENT_PUBLICATION_UNCERTAIN_TERMINAL",
        "SHADOW_OUTCOME_LINK_PUBLICATION_UNCERTAIN",
    ):
        receipt = _failure_receipt(frozen, state)
        with pytest.raises(core.ShadowNonadoptionTrialCoreError):
            core.validate_shadow_nonadoption_ledger_record_v1(
                receipt,
                "failure",
                _failure_context(receipt, "LEDGER_RECORD"),
                frozen,
            )

    uncertain = _failure_receipt(
        frozen, "INTENT_STAGED_PREPARATION_FAILED_CLOSED_UNCONSUMED"
    )
    uncertain["failure_record_publication_uncertain"] = True
    uncertain = _resign_record(frozen, "failure", uncertain)
    with pytest.raises(core.ShadowNonadoptionTrialCoreError):
        core.validate_shadow_nonadoption_ledger_record_v1(
            uncertain,
            "failure",
            _failure_context(uncertain, "LEDGER_RECORD"),
            frozen,
        )


def test_ledger_root_failure_is_a_transport_receipt_with_dynamic_error_wire(
    monkeypatch, contract, core, tool
) -> None:
    frozen = _resign_contract(contract)
    monkeypatch.setattr(tool, "_CORE_MODULE", core)
    monkeypatch.setattr(tool.os.path, "lexists", lambda _path: False)
    capsule = {"capsule_digest": TAGGED_CAPSULE}
    authorization = {"authorization_digest": TAGGED_AUTHORIZATION}
    intent = {"record_digest": TAGGED_INTENT}
    receipt = tool._failure_record(
        frozen,
        capsule,
        authorization,
        intent,
        None,
        phase_id="CREATE_LEDGER_ROOT",
        detail="synthetic ledger-root publication ambiguity",
        ledger_consumed=False,
        spawn_attempted=False,
        supervisor_called=False,
        process_result=None,
        survivor_absence_verified=True,
        child_reaped_verified=True,
        failure_publication_uncertain=True,
        ledger_created=None,
    )
    assert receipt["error_state"] == FAILURE_STATES[0]
    assert receipt["failure_record_publication_uncertain"] is True
    assert tool._validate_failure_channel(
        frozen,
        receipt,
        capsule,
        authorization,
        intent,
        None,
        "TRANSPORT_ONLY",
    ) == receipt
    with pytest.raises(core.ShadowNonadoptionTrialCoreError):
        tool._validate_failure_channel(
            frozen,
            receipt,
            capsule,
            authorization,
            intent,
            None,
            "LEDGER_RECORD",
        )
    error = tool.ShadowNonadoptionTrialError(
        "synthetic ledger-root publication ambiguity",
        receipt=receipt,
        terminal_after_intent=True,
    )
    wire = tool._error_wire(error)
    assert set(wire) == set(
        frozen["source_surface_policy"]["cli_wire_policy"]["error_exact_fields"]
    )
    assert wire["intent_staged"] is False
    assert wire["authorization_consumed"] is False
    assert wire["supervisor_called"] is False
    assert wire["supervisor_invocation_count"] == 0
    assert wire["receipt"] == receipt


def test_error_wire_derives_state_instead_of_claiming_static_success(
    contract, tool
) -> None:
    frozen = _resign_contract(contract)
    preintent = tool._error_wire(tool.ShadowNonadoptionTrialError("preintent"))
    assert preintent == {
        "schema_version": "hegel-phase3a-q05b-shadow-nonadoption-trial-error/1",
        "status": "SHADOW_NONADOPTION_TRIAL_PREINTENT_REJECTED",
        "detail": "preintent",
        "receipt": None,
        "intent_staged": False,
        "authorization_consumed": False,
        "supervisor_called": False,
        "trial_execution_performed": False,
        "supervisor_invocation_count": 0,
    }
    for state in FAILURE_STATES:
        receipt = _failure_receipt(frozen, state)
        wire = tool._error_wire(
            tool.ShadowNonadoptionTrialError(
                state, receipt=receipt, terminal_after_intent=True
            )
        )
        expected_intent: bool | None
        if state == FAILURE_STATES[0]:
            expected_intent = False
        elif state == FAILURE_STATES[1]:
            expected_intent = None
        else:
            expected_intent = True
        assert wire["status"] == "SHADOW_NONADOPTION_TRIAL_TERMINAL_FAILURE"
        assert wire["intent_staged"] is expected_intent
        assert wire["authorization_consumed"] is receipt["ledger_consumed"]
        assert wire["supervisor_called"] is receipt["supervisor_called"]
        assert wire["trial_execution_performed"] is receipt["supervisor_called"]
        assert wire["supervisor_invocation_count"] == receipt[
            "supervisor_invocation_count"
        ]


def test_wrapper_entrypoint_guard_is_exact_and_subprocess_free(
    monkeypatch, tool
) -> None:
    assert [
        tool._WRAPPER_PYTHON_PATH,
        "-I",
        "-S",
        "-B",
        "-X",
        "utf8",
        tool._WRAPPER_TOOL_PATH,
    ] == [
        "/usr/bin/python3.10",
        "-I",
        "-S",
        "-B",
        "-X",
        "utf8",
        os.fspath(TOOL_PATH),
    ]
    subprocess_calls: list[str] = []

    def forbidden_subprocess(*_args, **_kwargs):
        subprocess_calls.append("subprocess")
        raise AssertionError("wrapper guard invoked subprocess")

    monkeypatch.setattr(tool.subprocess, "run", forbidden_subprocess)
    monkeypatch.setattr(tool.subprocess, "Popen", forbidden_subprocess)
    base_flags = {
        "isolated": 1,
        "no_site": 1,
        "no_user_site": 1,
        "ignore_environment": 1,
        "dont_write_bytecode": 1,
        "utf8_mode": 1,
        "optimize": 0,
        "inspect": 0,
        "interactive": 0,
        "dev_mode": False,
    }

    def install(
        scoped,
        *,
        executable: str = tool._WRAPPER_PYTHON_PATH,
        tool_path: Path = Path(tool._WRAPPER_TOOL_PATH),
        argv: list[str] | None = None,
        sys_path: list[str] | None = None,
        flags: Mapping[str, Any] = base_flags,
        xoptions: Mapping[str, Any] | None = None,
        warnoptions: list[str] | None = None,
    ) -> None:
        scoped.setattr(tool.sys, "executable", executable)
        scoped.setattr(tool, "TOOL_PATH", tool_path)
        scoped.setattr(
            tool.sys,
            "argv",
            [tool._WRAPPER_TOOL_PATH] if argv is None else argv,
        )
        scoped.setattr(
            tool.sys,
            "path",
            list(tool._WRAPPER_SYS_PATH) if sys_path is None else sys_path,
        )
        scoped.setattr(tool.sys, "flags", types.SimpleNamespace(**dict(flags)))
        scoped.setattr(
            tool.sys,
            "_xoptions",
            {"utf8": True} if xoptions is None else dict(xoptions),
        )
        scoped.setattr(tool.sys, "warnoptions", [] if warnoptions is None else warnoptions)

    with monkeypatch.context() as scoped:
        install(scoped)
        tool._verify_wrapper_entrypoint()

    attacks: list[dict[str, Any]] = [
        {"executable": "/usr/bin/python3"},
        {"tool_path": Path("/wrong/tool.py")},
        {"argv": []},
        {"argv": ["/wrong/tool.py"]},
        {"sys_path": [*tool._WRAPPER_SYS_PATH, "/ambient/site-packages"]},
        {"xoptions": {}},
        {"xoptions": {"utf8": True, "dev": True}},
        {"warnoptions": ["default"]},
    ]
    for field, expected in base_flags.items():
        changed = dict(base_flags)
        changed[field] = (
            not expected if type(expected) is bool else (0 if expected == 1 else 1)
        )
        attacks.append({"flags": changed})
    for attack in attacks:
        with monkeypatch.context() as scoped:
            install(scoped, **attack)
            with pytest.raises(
                tool.ShadowNonadoptionTrialError, match="wrapper interpreter entrypoint"
            ):
                tool._verify_wrapper_entrypoint()
    assert subprocess_calls == []


def test_main_emits_canonical_dynamic_error_wire_without_calling_real_entrypoints(
    monkeypatch, tool
) -> None:
    stdout = io.BytesIO()
    stderr = io.BytesIO()
    monkeypatch.setattr(tool.sys, "stdout", types.SimpleNamespace(buffer=stdout))
    monkeypatch.setattr(tool.sys, "stderr", types.SimpleNamespace(buffer=stderr))
    monkeypatch.setattr(tool, "_verify_wrapper_entrypoint", lambda: None)
    monkeypatch.setattr(
        tool,
        "preflight_phase3_q05b_shadow_nonadoption_trial_v1",
        lambda: (_ for _ in ()).throw(tool.ShadowNonadoptionTrialError("blocked")),
    )
    assert tool.main(["--preflight"]) == 2
    assert stdout.getvalue() == b""
    parsed = tool._parse_canonical_json(stderr.getvalue(), "error wire KAT")
    assert parsed == tool._error_wire(tool.ShadowNonadoptionTrialError("blocked"))


def test_source_binding_placeholder_validates_but_cannot_build_capsule(
    contract, core
) -> None:
    frozen = _placeholder_contract(contract)
    assert frozen["source_binding"]["status"].startswith("PLACEHOLDER_")
    assert core.validate_shadow_nonadoption_trial_contract_v1(frozen) == frozen
    policy = frozen["capsule_policy"]
    with pytest.raises(core.ShadowNonadoptionTrialCoreError, match="not final"):
        core.build_shadow_nonadoption_execution_capsule_v1(
            frozen,
            _executable_identity(policy["python_executable_path"], "a"),
            _executable_identity(policy["git_executable_path"], "b"),
            _executable_identity(policy["docker_executable_path"], "c"),
            _cargo_identity(frozen),
        )


def test_final_source_binding_replays_exact_live_bytes_and_rejects_forgery(
    monkeypatch, contract, core, tool
) -> None:
    final = _finalized_contract(contract, tool)
    assert core.validate_shadow_nonadoption_trial_contract_v1(final) == final
    final_raw = _canonical(final)
    expected_literal = {
        "contract_digest": final["contract_digest"],
        "contract_raw_sha256": sha256(final_raw).hexdigest(),
        "normalized_source_digest": final["source_binding"][
            "normalized_source_digest"
        ],
    }
    monkeypatch.setattr(tool, "SOURCE_BINDING_LITERAL", expected_literal)
    tool._verify_live_source_binding(final, final_raw)

    forged = deepcopy(final)
    forged["source_binding"]["source_raw_sha256"] = "f" * 64
    unsigned_binding = deepcopy(forged["source_binding"])
    unsigned_binding.pop("source_binding_digest")
    forged["source_binding"]["source_binding_digest"] = _domain_digest(
        forged["hash_domains"]["source_binding"], unsigned_binding
    )
    forged = _resign_contract(forged)
    forged_raw = _canonical(forged)
    forged_literal = {
        "contract_digest": forged["contract_digest"],
        "contract_raw_sha256": sha256(forged_raw).hexdigest(),
        "normalized_source_digest": forged["source_binding"][
            "normalized_source_digest"
        ],
    }
    monkeypatch.setattr(tool, "SOURCE_BINDING_LITERAL", forged_literal)
    live_tool_raw = TOOL_PATH.read_bytes()
    literal_start = live_tool_raw.index(tool._LITERAL_START) + len(tool._LITERAL_START)
    literal_end = live_tool_raw.index(tool._LITERAL_END)
    forged_tool_raw = (
        live_tool_raw[:literal_start]
        + b"SOURCE_BINDING_LITERAL = "
        + _canonical_without_lf(forged_literal)
        + b"\n"
        + live_tool_raw[literal_end:]
    )
    original_reader = tool._read_regular_nofollow

    def forged_source_reader(path: Path, label: str, *args, **kwargs):
        if path == tool.TOOL_PATH:
            return forged_tool_raw
        return original_reader(path, label, *args, **kwargs)

    monkeypatch.setattr(tool, "_read_regular_nofollow", forged_source_reader)
    assert core.validate_shadow_nonadoption_trial_contract_v1(forged) == forged
    with pytest.raises(tool.ShadowNonadoptionTrialError, match="core raw bytes"):
        tool._verify_live_source_binding(forged, forged_raw)


def test_final_disk_contract_literal_and_all_bound_sources_load_live(
    contract, tool
) -> None:
    final = _finalized_contract(contract, tool)
    contract_raw = CONTRACT_PATH.read_bytes()
    core_raw = CORE_PATH.read_bytes()
    tool_raw = TOOL_PATH.read_bytes()
    tests_raw = Path(__file__).read_bytes()
    binding = final["source_binding"]
    assert contract_raw == _canonical(final)
    assert binding["status"] == FINAL_SOURCE_STATUS
    assert binding["source_raw_sha256"] == sha256(core_raw).hexdigest()
    assert binding["tests_raw_sha256"] == sha256(tests_raw).hexdigest()
    assert binding["normalized_source_digest"] == (
        "sha256:" + sha256(tool._normalize_tool_bytes(tool_raw)).hexdigest()
    )
    assert tool.SOURCE_BINDING_LITERAL == {
        "contract_digest": final["contract_digest"],
        "contract_raw_sha256": sha256(contract_raw).hexdigest(),
        "normalized_source_digest": binding["normalized_source_digest"],
    }
    assert tool.load_phase3_q05b_shadow_nonadoption_trial_contract_v1() == final
    assert tool._CORE_MODULE is not None
    assert tool._CORE_RAW_DIGEST == sha256(core_raw).hexdigest()


def test_source_binding_rejects_each_live_source_mutation_and_marker_attack(
    monkeypatch, contract, tool
) -> None:
    final = _finalized_contract(contract, tool)
    final_raw = _canonical(final)
    monkeypatch.setattr(
        tool,
        "SOURCE_BINDING_LITERAL",
        {
            "contract_digest": final["contract_digest"],
            "contract_raw_sha256": sha256(final_raw).hexdigest(),
            "normalized_source_digest": final["source_binding"][
                "normalized_source_digest"
            ],
        },
    )
    original_reader = tool._read_regular_nofollow
    for target in (tool.CORE_PATH, tool.TESTS_PATH, tool.TOOL_PATH):
        def mutated_reader(
            path: Path,
            label: str,
            maximum: int | None = None,
            *,
            _target: Path = target,
        ) -> bytes:
            raw = original_reader(path, label, maximum)
            return raw + b"# mutation\n" if path == _target else raw

        with monkeypatch.context() as scoped:
            scoped.setattr(tool, "_read_regular_nofollow", mutated_reader)
            with pytest.raises(tool.ShadowNonadoptionTrialError):
                tool._verify_live_source_binding(final, final_raw)

    tool_raw = TOOL_PATH.read_bytes()
    argv_needle = b'["-C", repository, "rev-parse", f"{commit}^"]'
    argv_replacement = b'["-C", repository, "show", f"{commit}^"]'
    assert tool_raw.count(argv_needle) == 1

    def argv_mutated_reader(
        path: Path, label: str, maximum: int | None = None
    ) -> bytes:
        raw = original_reader(path, label, maximum)
        return raw.replace(argv_needle, argv_replacement, 1) if path == tool.TOOL_PATH else raw

    with monkeypatch.context() as scoped:
        scoped.setattr(tool, "_read_regular_nofollow", argv_mutated_reader)
        with pytest.raises(tool.ShadowNonadoptionTrialError):
            tool._verify_live_source_binding(final, final_raw)

    attacks = (
        tool._LITERAL_START + tool_raw,
        tool_raw + tool._LITERAL_END,
        tool_raw.replace(
            b"# SOURCE_BINDING_LITERAL_END\n",
            b"SOURCE_BINDING_LITERAL = {}\n# SOURCE_BINDING_LITERAL_END\n",
            1,
        ),
        tool_raw.replace(
            b"# SOURCE_BINDING_LITERAL_END\n",
            b"extra_statement = True\n# SOURCE_BINDING_LITERAL_END\n",
            1,
        ),
        tool_raw.replace(
            tool_raw[
                tool_raw.index(tool._LITERAL_START) + len(tool._LITERAL_START) :
                tool_raw.index(tool._LITERAL_END)
            ],
            b'SOURCE_BINDING_LITERAL = {"contract_digest": dangerous()}\n',
            1,
        ),
        tool_raw.replace(
            tool_raw[
                tool_raw.index(tool._LITERAL_START) + len(tool._LITERAL_START) :
                tool_raw.index(tool._LITERAL_END)
            ],
            b"SOURCE_BINDING_LITERAL = {**payload}\n",
            1,
        ),
    )
    for attack in attacks:
        with pytest.raises(tool.ShadowNonadoptionTrialError):
            tool._normalize_tool_bytes(attack)


def test_tool_import_is_lazy_and_bound_core_bootstrap_is_exact_byte_no_pyc(
    monkeypatch, contract, tool
) -> None:
    before = {
        os.fspath(path): (
            path.stat().st_dev,
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in ROOT.rglob("*.pyc")
    }
    fresh_tool = _load("_q05b_shadow_nonadoption_lazy_import_kat", TOOL_PATH)
    assert fresh_tool._CORE_MODULE is None
    assert fresh_tool._CORE_RAW_DIGEST is None
    assert "_hegel_q05b_shadow_nonadoption_trial_core_v1" not in sys.modules

    final = _finalized_contract(contract, tool)
    final_raw = _canonical(final)
    core_raw = CORE_PATH.read_bytes()
    monkeypatch.setattr(fresh_tool, "_CORE_MODULE", None)
    monkeypatch.setattr(fresh_tool, "_CORE_RAW_DIGEST", None)
    monkeypatch.setattr(
        fresh_tool,
        "SOURCE_BINDING_LITERAL",
        {
            "contract_digest": final["contract_digest"],
            "contract_raw_sha256": sha256(final_raw).hexdigest(),
            "normalized_source_digest": final["source_binding"][
                "normalized_source_digest"
            ],
        },
    )
    reads: list[Path] = []

    def exact_core_reader(path: Path, _label: str, _maximum=None) -> bytes:
        reads.append(path)
        assert path == fresh_tool.CORE_PATH
        return core_raw

    monkeypatch.setattr(fresh_tool, "_read_regular_nofollow", exact_core_reader)
    loaded = fresh_tool._bootstrap_and_load_bound_core(final, final_raw)
    assert loaded.__file__ == os.fspath(CORE_PATH)
    assert fresh_tool._CORE_RAW_DIGEST == sha256(core_raw).hexdigest()
    assert reads == [fresh_tool.CORE_PATH]
    assert "_hegel_q05b_shadow_nonadoption_trial_core_v1" not in sys.modules

    monkeypatch.setattr(
        fresh_tool,
        "_read_regular_nofollow",
        lambda *_args, **_kwargs: core_raw + b"# changed\n",
    )
    with pytest.raises(fresh_tool.ShadowNonadoptionTrialError, match="core bytes"):
        fresh_tool._bootstrap_and_load_bound_core(final, final_raw)
    after = {
        os.fspath(path): (
            path.stat().st_dev,
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in ROOT.rglob("*.pyc")
    }
    assert after == before


def test_preflight_context_is_read_only_and_calls_only_frozen_derivers_in_order(
    monkeypatch, contract, tool
) -> None:
    frozen = _resign_contract(contract)
    events: list[str] = []
    identities = {
        "Python": _executable_identity(
            frozen["capsule_policy"]["python_executable_path"], "a"
        ),
        "Git": _executable_identity(
            frozen["capsule_policy"]["git_executable_path"], "b"
        ),
        "Docker": _executable_identity(
            frozen["capsule_policy"]["docker_executable_path"], "c"
        ),
    }
    cargo_identity = _cargo_identity(frozen)
    cargo_rows = [{"relative_path": "x", "file_kind": "regular"}]
    capsule = {"capsule_digest": TAGGED_CAPSULE}

    monkeypatch.setattr(
        tool, "_verify_path_boundaries", lambda _contract: events.append("paths")
    )
    monkeypatch.setattr(
        tool,
        "_verify_frozen_git_objects",
        lambda _contract, identity: events.append("git")
        if identity == identities["Git"]
        else (_ for _ in ()).throw(AssertionError("Git identity changed")),
    )
    monkeypatch.setattr(
        tool,
        "_verify_attempt2_protected_state",
        lambda _contract, identity: events.append("attempt2")
        if identity == identities["Git"]
        else (_ for _ in ()).throw(AssertionError("Git identity changed")),
    )

    def identity(_path: str, label: str) -> dict[str, Any]:
        events.append(f"identity:{label}")
        return identities[label]

    monkeypatch.setattr(tool, "_executable_identity", identity)

    def manifest(_contract, _root):
        events.append("cargo_manifest")
        return cargo_identity, cargo_rows

    monkeypatch.setattr(tool, "_tree_manifest", manifest)

    class CoreProxy:
        @staticmethod
        def build_shadow_nonadoption_execution_capsule_v1(*args):
            events.append("capsule")
            assert args == (
                frozen,
                identities["Python"],
                identities["Git"],
                identities["Docker"],
                cargo_identity,
            )
            return capsule

    monkeypatch.setattr(tool, "_core", lambda: CoreProxy)
    for name in (
        "_prepare_isolated_clone",
        "_prepare_isolated_cargo_cache",
        "_publish_ledger_record",
        "_spawn_supervisor_once",
        "_docker_inventory",
    ):
        monkeypatch.setattr(
            tool,
            name,
            lambda *_args, _name=name, **_kwargs: (_ for _ in ()).throw(
                AssertionError(f"preflight called mutator {_name}")
            ),
        )
    context = tool._preflight_context(frozen)
    assert events == [
        "paths",
        "identity:Python",
        "identity:Git",
        "identity:Docker",
        "git",
        "attempt2",
        "cargo_manifest",
        "capsule",
    ]
    assert context == {
        "capsule": capsule,
        "cargo_rows": cargo_rows,
        "python_identity": identities["Python"],
        "git_identity": identities["Git"],
        "docker_identity": identities["Docker"],
    }


def test_preconsume_full_replay_has_exact_fail_closed_helper_order(
    monkeypatch, contract, tool
) -> None:
    frozen = _resign_contract(contract)
    git_identity = _executable_identity(
        frozen["capsule_policy"]["git_executable_path"], "b"
    )
    capsule = {"git_executable_identity": git_identity}
    events: list[str] = []
    monkeypatch.setattr(
        tool,
        "load_phase3_q05b_shadow_nonadoption_trial_contract_v1",
        lambda: events.append("live_contract") or frozen,
    )
    monkeypatch.setattr(
        tool, "_verify_path_boundaries", lambda _contract: events.append("paths")
    )
    monkeypatch.setattr(
        tool,
        "_verify_frozen_git_objects",
        lambda _contract, identity: events.append("git")
        if identity == git_identity
        else (_ for _ in ()).throw(AssertionError("Git identity changed")),
    )
    monkeypatch.setattr(
        tool,
        "_verify_attempt2_protected_state",
        lambda _contract, identity: events.append("attempt2")
        if identity == git_identity
        else (_ for _ in ()).throw(AssertionError("Git identity changed")),
    )
    monkeypatch.setattr(
        tool,
        "_replay_capsule_executable_identities",
        lambda _capsule: events.append("executables"),
    )

    def clone(_contract, _capsule, *, require_artifact_absent):
        assert require_artifact_absent is True
        events.append("clone")

    monkeypatch.setattr(tool, "_verify_isolated_clone", clone)
    monkeypatch.setattr(
        tool,
        "_replay_cargo_source_and_target",
        lambda *_args: events.append("cargo"),
    )
    monkeypatch.setattr(
        tool,
        "_docker_inventory",
        lambda *_args: events.append("docker") or [],
    )
    tool._preconsume_full_replay(frozen, capsule, [])
    assert events == [
        "live_contract",
        "paths",
        "git",
        "attempt2",
        "executables",
        "clone",
        "cargo",
        "docker",
    ]

    events.clear()
    monkeypatch.setattr(
        tool,
        "_verify_attempt2_protected_state",
        lambda _contract, _identity: (_ for _ in ()).throw(
            tool.ShadowNonadoptionTrialError("stop")
        ),
    )
    with pytest.raises(tool.ShadowNonadoptionTrialError, match="stop"):
        tool._preconsume_full_replay(frozen, capsule, [])
    assert events == ["live_contract", "paths", "git"]


def test_run_orders_full_replay_before_consumption_and_spawn_using_only_fakes(
    monkeypatch, contract, tool
) -> None:
    frozen = _resign_contract(contract)
    capsule = {
        "capsule_digest": TAGGED_CAPSULE,
        "git_executable_identity": _executable_identity(
            frozen["capsule_policy"]["git_executable_path"], "b"
        ),
    }
    authorization = {"authorization_digest": TAGGED_AUTHORIZATION}
    intent = {"record_digest": TAGGED_INTENT}
    consumed = {"record_digest": TAGGED_CONSUMED}
    success = {"record_digest": "sha256:" + "7" * 64, "status": "synthetic"}
    events: list[str] = []

    monkeypatch.setattr(tool, "_verify_wrapper_entrypoint", lambda: None)
    monkeypatch.setattr(
        tool, "_reject_preexisting_ledger_before_preflight", lambda _contract: None
    )
    monkeypatch.setattr(
        tool, "load_phase3_q05b_shadow_nonadoption_trial_contract_v1", lambda: frozen
    )
    monkeypatch.setattr(
        tool,
        "_preflight_context",
        lambda _contract: {"capsule": capsule, "cargo_rows": []},
    )
    monkeypatch.setattr(tool, "_load_authorization", lambda *_args: authorization)
    monkeypatch.setattr(tool, "_intent_record", lambda *_args: intent)
    monkeypatch.setattr(tool, "_consumed_record", lambda *_args: consumed)
    monkeypatch.setattr(
        tool,
        "_success_record",
        lambda *_args: events.append("success_record") or success,
    )

    class CoreProxy:
        @staticmethod
        def validate_shadow_nonadoption_ledger_record_v1(*_args):
            events.append("validate")
            return {}

    monkeypatch.setattr(tool, "_core", lambda: CoreProxy)
    monkeypatch.setattr(
        tool,
        "_create_and_hold_ledger_root",
        lambda _path: events.append("create_ledger") or (101, 102),
    )
    ledger_identities = {
        101: {"dev": 1, "ino": 2, "mode": 0o040700, "uid": 1000, "nlink": 3, "mnt_id": 4},
        102: {"dev": 1, "ino": 5, "mode": 0o040700, "uid": 1000, "nlink": 2, "mnt_id": 4},
    }
    monkeypatch.setattr(
        tool,
        "_held_directory_identity",
        lambda fd, _label: deepcopy(ledger_identities[fd]),
    )
    monkeypatch.setattr(
        tool,
        "_verify_held_ledger_path",
        lambda *_args: events.append("ledger_path"),
    )
    monkeypatch.setattr(tool.fcntl, "flock", lambda *_args: events.append("flock"))
    monkeypatch.setattr(tool.os, "listdir", lambda _fd: [])
    monkeypatch.setattr(tool.os, "close", lambda fd: events.append(f"close:{fd}"))
    monkeypatch.setattr(tool, "_audit_event", lambda name: events.append(name))
    monkeypatch.setattr(tool, "_verify_path_boundaries", lambda *_args: None)
    monkeypatch.setattr(
        tool, "_verify_exact_ledger_state", lambda *_args: events.append("ledger_replay")
    )

    def publish(_fd: int, name: str, _record: Mapping[str, Any]) -> None:
        events.append(f"publish:{name}")

    monkeypatch.setattr(tool, "_publish_ledger_record", publish)
    monkeypatch.setattr(
        tool, "_prepare_isolated_clone", lambda *_args: events.append("clone")
    )
    monkeypatch.setattr(
        tool, "_prepare_isolated_cargo_cache", lambda *_args: events.append("cargo")
    )
    monkeypatch.setattr(
        tool, "_preconsume_full_replay", lambda *_args: events.append("full_replay")
    )
    monkeypatch.setattr(
        tool,
        "_prepare_supervisor_spawn",
        lambda *_args: events.append("spawn_prepare"),
    )

    class Stream:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            events.append(f"close_stream:{self.name}")

    process = types.SimpleNamespace(stdout=Stream("stdout"), stderr=Stream("stderr"))

    def spawn(_capsule, enter_popen_boundary):
        enter_popen_boundary()
        events.append("spawn")
        return process, None

    monkeypatch.setattr(
        tool,
        "_spawn_supervisor_once",
        spawn,
    )
    monkeypatch.setattr(
        tool,
        "_bounded_communicate",
        lambda observed, _capsule: (
            events.append("communicate") or (0, b"{}\n", b"", False)
        )
        if observed is process
        else (_ for _ in ()).throw(AssertionError("process identity changed")),
    )
    monkeypatch.setattr(
        tool,
        "_decode_supervisor_success",
        lambda *_args: events.append("decode") or {"synthetic": True},
    )
    monkeypatch.setattr(
        tool, "_docker_inventory", lambda *_args: events.append("docker") or []
    )
    monkeypatch.setattr(
        tool,
        "_replay_artifact_evidence",
        lambda *_args: events.append("artifact_replay"),
    )
    monkeypatch.setattr(
        tool,
        "_verify_attempt2_protected_state",
        lambda *_args: events.append("protected"),
    )
    monkeypatch.setattr(
        tool,
        "_replay_capsule_executable_identities",
        lambda *_args: events.append("executables"),
    )
    monkeypatch.setattr(
        tool,
        "_verify_isolated_clone",
        lambda *_args, **kwargs: events.append(
            f"postrun_clone:{kwargs['require_artifact_absent']}"
        ),
    )
    monkeypatch.setattr(
        tool,
        "_replay_cargo_source_and_target",
        lambda *_args: events.append("postrun_cargo"),
    )
    assert tool.run_phase3_q05b_shadow_nonadoption_trial_v1() == success
    assert events.count("ledger_path") == 9
    for name in (intent_name := frozen["trial_ledger_policy"]["intent_record_name"],
                 consumed_name := frozen["trial_ledger_policy"]["consume_record_name"],
                 success_name := frozen["trial_ledger_policy"]["success_record_name"]):
        publish_index = events.index(f"publish:{name}")
        assert events[publish_index - 1 : publish_index + 2] == [
            "ledger_path",
            f"publish:{name}",
            "ledger_path",
        ]

    critical = [
        event
        for event in events
        if event
        in {
            f"publish:{intent_name}",
            "clone",
            "cargo",
            "full_replay",
            f"publish:{consumed_name}",
            "spawn_prepare",
            "spawn",
            "communicate",
            "decode",
            "artifact_replay",
            "docker",
            "protected",
            "executables",
            "postrun_clone:False",
            "postrun_cargo",
            "success_record",
            f"publish:{success_name}",
        }
    ]
    assert critical == [
        f"publish:{intent_name}",
        "clone",
        "cargo",
        "full_replay",
        f"publish:{consumed_name}",
        "spawn_prepare",
        "spawn",
        "communicate",
        "decode",
        "artifact_replay",
        "docker",
        "artifact_replay",
        "protected",
        "executables",
        "postrun_clone:False",
        "postrun_cargo",
        "success_record",
        "artifact_replay",
        f"publish:{success_name}",
        "artifact_replay",
    ]


def test_preconsume_failure_uses_null_consumed_lineage_even_if_record_was_built(
    monkeypatch, contract, tool
) -> None:
    frozen = _resign_contract(contract)
    capsule = {"capsule_digest": TAGGED_CAPSULE}
    authorization = {"authorization_digest": TAGGED_AUTHORIZATION}
    intent = {"record_digest": TAGGED_INTENT}
    constructed_consumed = {"record_digest": TAGGED_CONSUMED}
    channel_consumed_arguments: list[Mapping[str, Any] | None] = []
    ledger_events: list[str] = []

    monkeypatch.setattr(tool, "_verify_wrapper_entrypoint", lambda: None)
    monkeypatch.setattr(
        tool, "_reject_preexisting_ledger_before_preflight", lambda _contract: None
    )
    monkeypatch.setattr(
        tool, "load_phase3_q05b_shadow_nonadoption_trial_contract_v1", lambda: frozen
    )
    monkeypatch.setattr(
        tool,
        "_preflight_context",
        lambda _contract: {"capsule": capsule, "cargo_rows": []},
    )
    monkeypatch.setattr(tool, "_load_authorization", lambda *_args: authorization)
    monkeypatch.setattr(tool, "_intent_record", lambda *_args: intent)
    monkeypatch.setattr(
        tool, "_consumed_record", lambda *_args: constructed_consumed
    )

    class CoreProxy:
        @staticmethod
        def validate_shadow_nonadoption_ledger_record_v1(*_args):
            return {}

    monkeypatch.setattr(tool, "_core", lambda: CoreProxy)
    monkeypatch.setattr(tool, "_create_and_hold_ledger_root", lambda _path: (101, 102))
    monkeypatch.setattr(
        tool,
        "_held_directory_identity",
        lambda fd, _label: {
            "dev": 1,
            "ino": fd,
            "mode": 0o040700,
            "uid": 1000,
            "nlink": 2,
            "mnt_id": 3,
        },
    )
    monkeypatch.setattr(
        tool,
        "_verify_held_ledger_path",
        lambda *_args: ledger_events.append("ledger_path"),
    )
    monkeypatch.setattr(tool.fcntl, "flock", lambda *_args: None)
    monkeypatch.setattr(tool.os, "listdir", lambda _fd: [])
    monkeypatch.setattr(tool.os, "close", lambda _fd: None)
    monkeypatch.setattr(tool, "_verify_path_boundaries", lambda *_args: None)
    monkeypatch.setattr(
        tool,
        "_verify_exact_ledger_state",
        lambda *_args: ledger_events.append("ledger_state"),
    )
    monkeypatch.setattr(
        tool,
        "_publish_ledger_record",
        lambda _fd, name, _record: ledger_events.append(f"publish:{name}"),
    )
    monkeypatch.setattr(tool, "_audit_event", lambda _name: None)
    monkeypatch.setattr(tool, "_prepare_isolated_clone", lambda *_args: None)
    monkeypatch.setattr(tool, "_prepare_isolated_cargo_cache", lambda *_args: None)
    monkeypatch.setattr(
        tool,
        "_preconsume_full_replay",
        lambda *_args: (_ for _ in ()).throw(
            tool.ShadowNonadoptionTrialError(
                "synthetic preconsume replay failure", terminal_after_intent=True
            )
        ),
    )
    receipt = {
        "error_state": "INTENT_STAGED_PREPARATION_FAILED_CLOSED_UNCONSUMED",
        "failure_record_publication_uncertain": False,
    }
    monkeypatch.setattr(tool, "_failure_record", lambda *_args, **_kwargs: receipt)

    def validate_channel(
        _contract,
        _receipt,
        _capsule,
        _authorization,
        _intent,
        consumed,
        _channel,
    ):
        channel_consumed_arguments.append(consumed)
        return receipt

    monkeypatch.setattr(tool, "_validate_failure_channel", validate_channel)
    with pytest.raises(
        tool.ShadowNonadoptionTrialError, match="synthetic preconsume replay failure"
    ):
        tool.run_phase3_q05b_shadow_nonadoption_trial_v1()
    assert channel_consumed_arguments == [None, None]
    failure_name = frozen["trial_ledger_policy"]["failure_record_name"]
    failure_index = ledger_events.index(f"publish:{failure_name}")
    assert ledger_events[failure_index - 1 : failure_index + 4] == [
        "ledger_path",
        f"publish:{failure_name}",
        "ledger_path",
        "ledger_state",
        "ledger_path",
    ]


def test_postdecode_artifact_deletion_blocks_success_and_emits_valid_receipt(
    monkeypatch, contract, core, tool
) -> None:
    frozen = _resign_contract(contract)
    capsule = {
        "capsule_digest": TAGGED_CAPSULE,
        "artifact_path": frozen["actual_execution_policy"]["artifact_path"],
    }
    authorization = {"authorization_digest": TAGGED_AUTHORIZATION}
    intent = {"record_digest": TAGGED_INTENT}
    consumed = {"record_digest": TAGGED_CONSUMED}
    artifact_live = {"value": True}
    published: list[str] = []
    failure_channels: list[str] = []

    monkeypatch.setattr(tool, "_verify_wrapper_entrypoint", lambda: None)
    monkeypatch.setattr(
        tool, "_reject_preexisting_ledger_before_preflight", lambda _contract: None
    )
    monkeypatch.setattr(
        tool, "load_phase3_q05b_shadow_nonadoption_trial_contract_v1", lambda: frozen
    )
    monkeypatch.setattr(
        tool,
        "_preflight_context",
        lambda _contract: {"capsule": capsule, "cargo_rows": []},
    )
    monkeypatch.setattr(tool, "_load_authorization", lambda *_args: authorization)
    monkeypatch.setattr(tool, "_intent_record", lambda *_args: intent)
    monkeypatch.setattr(tool, "_consumed_record", lambda *_args: consumed)

    class CoreProxy:
        @staticmethod
        def validate_shadow_nonadoption_ledger_record_v1(
            record, kind, context, observed_contract
        ):
            if kind == "failure":
                failure_channels.append(context["receipt_channel"])
                return core.validate_shadow_nonadoption_ledger_record_v1(
                    record, kind, context, observed_contract
                )
            return record

    monkeypatch.setattr(tool, "_core", lambda: CoreProxy)
    monkeypatch.setattr(tool, "_verify_path_boundaries", lambda *_args: None)
    monkeypatch.setattr(tool, "_create_and_hold_ledger_root", lambda _path: (101, 102))
    monkeypatch.setattr(
        tool,
        "_held_directory_identity",
        lambda fd, _label: {
            "dev": 1,
            "ino": fd,
            "mode": 0o040700,
            "uid": 1000,
            "nlink": 2,
            "mnt_id": 3,
        },
    )
    monkeypatch.setattr(tool, "_verify_held_ledger_path", lambda *_args: None)
    monkeypatch.setattr(tool.fcntl, "flock", lambda *_args: None)
    monkeypatch.setattr(tool.os, "listdir", lambda _fd: [])
    monkeypatch.setattr(tool.os, "close", lambda _fd: None)
    monkeypatch.setattr(tool, "_verify_exact_ledger_state", lambda *_args: None)
    monkeypatch.setattr(tool, "_audit_event", lambda _name: None)
    monkeypatch.setattr(
        tool,
        "_publish_ledger_record",
        lambda _fd, name, _record: published.append(name),
    )
    monkeypatch.setattr(tool, "_prepare_isolated_clone", lambda *_args: None)
    monkeypatch.setattr(tool, "_prepare_isolated_cargo_cache", lambda *_args: None)
    monkeypatch.setattr(tool, "_preconsume_full_replay", lambda *_args: None)
    monkeypatch.setattr(tool, "_prepare_supervisor_spawn", lambda *_args: None)

    class Stream:
        @staticmethod
        def close() -> None:
            return None

    process = types.SimpleNamespace(stdout=Stream(), stderr=Stream())
    monkeypatch.setattr(
        tool,
        "_spawn_supervisor_once",
        lambda _capsule, enter_popen_boundary: (
            enter_popen_boundary() or process,
            None,
        ),
    )
    monkeypatch.setattr(
        tool,
        "_bounded_communicate",
        lambda observed, _capsule: (0, b"{}\n", b"", False)
        if observed is process
        else (_ for _ in ()).throw(AssertionError("process identity changed")),
    )

    def decode(*_args):
        artifact_live["value"] = False
        return {"synthetic": True}

    monkeypatch.setattr(tool, "_decode_supervisor_success", decode)
    monkeypatch.setattr(tool, "_docker_inventory", lambda *_args: [])

    def replay_artifact(*_args):
        if not artifact_live["value"]:
            raise tool.ShadowNonadoptionTrialError(
                "artifact deleted after stdout decode", terminal_after_intent=True
            )

    monkeypatch.setattr(tool, "_replay_artifact_evidence", replay_artifact)
    monkeypatch.setattr(
        tool,
        "_success_record",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("success was constructed after artifact deletion")
        ),
    )
    monkeypatch.setattr(
        tool.os.path,
        "lexists",
        lambda path: artifact_live["value"]
        if os.fspath(path) == capsule["artifact_path"]
        else False,
    )

    with pytest.raises(tool.ShadowNonadoptionTrialError) as captured:
        tool.run_phase3_q05b_shadow_nonadoption_trial_v1()
    receipt = captured.value.receipt
    assert receipt is not None
    assert receipt["error_state"] == "SUPERVISOR_RETURNED_ARTIFACT_MISSING_OR_INVALID"
    assert receipt["phase_id"] == "VALIDATE_ARTIFACT"
    assert receipt["artifact_may_exist"] is False
    assert failure_channels == ["LEDGER_RECORD", "TRANSPORT_ONLY"]
    ledger = frozen["trial_ledger_policy"]
    assert published == [
        ledger["intent_record_name"],
        ledger["consume_record_name"],
        ledger["failure_record_name"],
    ]
    assert ledger["success_record_name"] not in published
    assert core.validate_shadow_nonadoption_ledger_record_v1(
        receipt,
        "failure",
        _failure_context(receipt, "TRANSPORT_ONLY"),
        frozen,
    ) == receipt


def test_keyboardinterrupt_after_popen_forces_process_group_cleanup_and_reap(
    monkeypatch, tool
) -> None:
    class Stream:
        def __init__(self, fd: int) -> None:
            self._fd = fd

        def fileno(self) -> int:
            return self._fd

    class Process:
        pid = 4242
        stdout = Stream(11)
        stderr = Stream(12)

        def __init__(self) -> None:
            self.wait_calls = 0

        @staticmethod
        def poll():
            return None

        def wait(self, *, timeout):
            assert timeout == 30
            self.wait_calls += 1
            return -signal.SIGKILL

    class InterruptingSelector:
        def __init__(self) -> None:
            self.registered: list[Any] = []
            self.closed = False

        def register(self, stream, _events, _name):
            self.registered.append(stream)

        def get_map(self):
            return {index: stream for index, stream in enumerate(self.registered)}

        @staticmethod
        def select(_timeout):
            raise KeyboardInterrupt()

        def close(self):
            self.closed = True

    process = Process()
    selector = InterruptingSelector()
    clock = {"value": 0.0}

    def monotonic() -> float:
        clock["value"] += 1000.0
        return clock["value"]

    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(tool.selectors, "DefaultSelector", lambda: selector)
    monkeypatch.setattr(tool.os, "set_blocking", lambda *_args: None)
    monkeypatch.setattr(tool.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(tool.time, "monotonic", monotonic)
    monkeypatch.setattr(tool.time, "sleep", lambda _seconds: None)
    capsule = {
        "resource_limits": {"stdout_max_bytes": 16, "stderr_max_bytes": 16},
        "monotonic_timeout_policy": {"maximum_seconds": 7200},
        "kill_reap_survivor_policy": {
            "first_signal_grace_seconds": 120,
            "postkill_wait_seconds": 30,
        },
    }
    with pytest.raises((KeyboardInterrupt, tool.ShadowNonadoptionTrialError)):
        tool._bounded_communicate(process, capsule)
    assert (process.pid, signal.SIGINT) in signals
    assert (process.pid, signal.SIGKILL) in signals
    assert process.wait_calls == 1
    assert selector.closed is True


def test_exited_leader_with_descendant_held_pipes_still_kills_group_and_reaps(
    monkeypatch, tool
) -> None:
    class Stream:
        def __init__(self, fd: int) -> None:
            self._fd = fd
            self.closed = False

        def fileno(self) -> int:
            return self._fd

        def close(self) -> None:
            self.closed = True

    class Process:
        pid = 5252
        returncode = 0

        def __init__(self) -> None:
            self.stdout = Stream(21)
            self.stderr = Stream(22)
            self.poll_calls = 0
            self.wait_calls = 0

        def poll(self):
            self.poll_calls += 1
            return 0

        def wait(self, *, timeout):
            assert timeout == 30
            self.wait_calls += 1
            return 0

    class HeldPipeSelector:
        def __init__(self) -> None:
            self.registered: list[Any] = []
            self.closed = False

        def register(self, stream, _events, _name):
            self.registered.append(stream)

        def get_map(self):
            return {index: stream for index, stream in enumerate(self.registered)}

        def select(self, _timeout):
            raise AssertionError("zero timeout should clean up before selector wait")

        def close(self):
            self.closed = True

    process = Process()
    selector = HeldPipeSelector()
    clock = {"value": 0.0}
    signals: list[tuple[int, int]] = []
    group = {"alive": True}

    def killpg(pid: int, sig: int) -> None:
        assert pid == process.pid
        if sig == 0:
            if group["alive"]:
                return
            raise ProcessLookupError
        signals.append((pid, sig))
        if sig == signal.SIGKILL:
            group["alive"] = False

    monkeypatch.setattr(tool.selectors, "DefaultSelector", lambda: selector)
    monkeypatch.setattr(tool.os, "set_blocking", lambda *_args: None)
    monkeypatch.setattr(tool.os, "killpg", killpg)
    monkeypatch.setattr(
        tool.time,
        "monotonic",
        lambda: clock.__setitem__("value", clock["value"] + 1.0)
        or clock["value"],
    )
    monkeypatch.setattr(tool.time, "sleep", lambda _seconds: None)
    capsule = {
        "resource_limits": {"stdout_max_bytes": 16, "stderr_max_bytes": 16},
        "monotonic_timeout_policy": {"maximum_seconds": 0},
        "kill_reap_survivor_policy": {
            "first_signal_grace_seconds": 0,
            "postkill_wait_seconds": 30,
        },
    }
    assert tool._bounded_communicate(process, capsule) == (0, b"", b"", True)
    assert (process.pid, signal.SIGINT) in signals
    assert (process.pid, signal.SIGKILL) in signals
    assert process.wait_calls == 1
    assert process.stdout.closed is True
    assert process.stderr.closed is True
    assert selector.closed is True


def test_rc0_eof_with_live_process_group_fails_wait_after_cleanup_and_reap(
    monkeypatch, tool
) -> None:
    class Stream:
        def __init__(self, fd: int) -> None:
            self._fd = fd
            self.closed = False

        def fileno(self) -> int:
            return self._fd

        def close(self) -> None:
            self.closed = True

    class Process:
        pid = 5353
        returncode = 0

        def __init__(self) -> None:
            self.stdout = Stream(31)
            self.stderr = Stream(32)
            self.wait_calls = 0

        @staticmethod
        def poll() -> int:
            return 0

        def wait(self, *, timeout: int) -> int:
            assert timeout == 30
            self.wait_calls += 1
            return 0

    class EofSelector:
        def __init__(self) -> None:
            self.registered: dict[Any, str] = {}
            self.closed = False

        def register(self, stream: Any, _events: int, name: str) -> None:
            self.registered[stream] = name

        def unregister(self, stream: Any) -> None:
            del self.registered[stream]

        def get_map(self) -> dict[Any, str]:
            return dict(self.registered)

        def select(self, _timeout: float) -> list[tuple[Any, int]]:
            return [
                (types.SimpleNamespace(fileobj=stream, data=name), 1)
                for stream, name in list(self.registered.items())
            ]

        def close(self) -> None:
            self.closed = True

    process = Process()
    selector = EofSelector()
    group = {"alive": True}
    signals: list[tuple[int, int]] = []

    def killpg(pid: int, sig: int) -> None:
        assert pid == process.pid
        if sig == 0:
            if group["alive"]:
                return
            raise ProcessLookupError
        signals.append((pid, sig))
        if sig == signal.SIGKILL:
            group["alive"] = False

    clock = {"value": 0.0}
    monkeypatch.setattr(tool.selectors, "DefaultSelector", lambda: selector)
    monkeypatch.setattr(tool.os, "set_blocking", lambda *_args: None)
    monkeypatch.setattr(tool.os, "read", lambda *_args: b"")
    monkeypatch.setattr(tool.os, "killpg", killpg)
    monkeypatch.setattr(
        tool.time,
        "monotonic",
        lambda: clock.__setitem__("value", clock["value"] + 0.1)
        or clock["value"],
    )
    monkeypatch.setattr(tool.time, "sleep", lambda _seconds: None)
    capsule = {
        "resource_limits": {"stdout_max_bytes": 16, "stderr_max_bytes": 16},
        "monotonic_timeout_policy": {"maximum_seconds": 60},
        "kill_reap_survivor_policy": {
            "first_signal_grace_seconds": 0,
            "postkill_wait_seconds": 30,
        },
    }
    with pytest.raises(
        tool.ShadowNonadoptionTrialError,
        match="process group survived direct-child return",
    ) as captured:
        tool._bounded_communicate(process, capsule)
    assert captured.value.child_reaped_verified is True
    assert signals == [
        (process.pid, signal.SIGINT),
        (process.pid, signal.SIGKILL),
    ]
    assert process.wait_calls == 2
    assert process.stdout.closed is True
    assert process.stderr.closed is True
    assert selector.closed is True
    run_source = inspect.getsource(tool.run_phase3_q05b_shadow_nonadoption_trial_v1)
    assert run_source.index('phase_id = "SUPERVISOR_WAIT"') < run_source.index(
        "_bounded_communicate("
    ) < run_source.index('phase_id = "VALIDATE_STDOUT"')


def test_spawn_umask_restore_failure_cannot_lose_live_process_handle(
    monkeypatch, tool
) -> None:
    class Process:
        pid = 4343

        def __init__(self) -> None:
            self.wait_calls = 0

    process = Process()
    monkeypatch.setattr(
        tool, "_replay_capsule_executable_identities", lambda _capsule: None
    )
    monkeypatch.setattr(tool.subprocess, "Popen", lambda *_args, **_kwargs: process)
    mask_calls: list[tuple[Any, Any]] = []

    def pthread_sigmask(how, mask):
        mask_calls.append((how, mask))
        return {signal.SIGTERM}

    monkeypatch.setattr(tool.signal, "pthread_sigmask", pthread_sigmask)
    umask_calls = {"count": 0}

    def umask(_value: int) -> int:
        umask_calls["count"] += 1
        if umask_calls["count"] == 1:
            return 0o022
        raise OSError("synthetic umask restore failure")

    monkeypatch.setattr(tool.os, "umask", umask)
    capsule = {
        "ordered_argv": ["/fixed/python", "supervisor.py"],
        "clone_project_root": "/fixed/clone",
        "runtime_environment": {},
    }
    boundary_calls: list[str] = []
    observed, restoration_error = tool._spawn_supervisor_once(
        capsule, lambda: boundary_calls.append("entered")
    )
    assert observed is process
    assert restoration_error is not None
    assert "umask restoration failed" in restoration_error
    assert process.wait_calls == 0
    assert boundary_calls == ["entered"]
    assert mask_calls == [
        (signal.SIG_SETMASK, set()),
        (signal.SIG_SETMASK, {signal.SIGTERM}),
    ]


def test_spawn_boundary_callback_separates_pre_popen_and_popen_failures(
    monkeypatch, tool
) -> None:
    capsule = {
        "ordered_argv": ["/fixed/python", "supervisor.py"],
        "clone_project_root": "/fixed/clone",
        "runtime_environment": {},
    }

    def state_and_callback():
        state = {
            "phase": "SPAWN_PREPARE",
            "supervisor_spawn_attempted": False,
            "supervisor_called": False,
            "supervisor_invocation_count": 0,
        }

        def enter_popen_boundary() -> None:
            state["phase"] = "SUPERVISOR_SPAWN"
            state["supervisor_spawn_attempted"] = True
            state["supervisor_called"] = True
            state["supervisor_invocation_count"] = 1

        return state, enter_popen_boundary

    def forbidden(*_args, **_kwargs):
        raise AssertionError("pre-Popen failure crossed the Popen boundary")

    state, callback = state_and_callback()
    with monkeypatch.context() as scoped:
        scoped.setattr(
            tool,
            "_replay_capsule_executable_identities",
            lambda _capsule: (_ for _ in ()).throw(
                tool.ShadowNonadoptionTrialError("identity replay failed")
            ),
        )
        scoped.setattr(tool.os, "umask", forbidden)
        scoped.setattr(tool.signal, "pthread_sigmask", forbidden)
        scoped.setattr(tool.subprocess, "Popen", forbidden)
        with pytest.raises(tool.ShadowNonadoptionTrialError, match="identity replay"):
            tool._spawn_supervisor_once(capsule, callback)
    assert state == {
        "phase": "SPAWN_PREPARE",
        "supervisor_spawn_attempted": False,
        "supervisor_called": False,
        "supervisor_invocation_count": 0,
    }

    state, callback = state_and_callback()
    with monkeypatch.context() as scoped:
        scoped.setattr(tool, "_replay_capsule_executable_identities", lambda _capsule: None)
        scoped.setattr(
            tool.os,
            "umask",
            lambda _value: (_ for _ in ()).throw(OSError("umask failed")),
        )
        scoped.setattr(tool.signal, "pthread_sigmask", forbidden)
        scoped.setattr(tool.subprocess, "Popen", forbidden)
        with pytest.raises(OSError, match="umask failed"):
            tool._spawn_supervisor_once(capsule, callback)
    assert state == {
        "phase": "SPAWN_PREPARE",
        "supervisor_spawn_attempted": False,
        "supervisor_called": False,
        "supervisor_invocation_count": 0,
    }

    state, callback = state_and_callback()
    umask_values: list[int] = []

    def umask(value: int) -> int:
        umask_values.append(value)
        return 0o022

    with monkeypatch.context() as scoped:
        scoped.setattr(tool, "_replay_capsule_executable_identities", lambda _capsule: None)
        scoped.setattr(tool.os, "umask", umask)
        scoped.setattr(
            tool.signal,
            "pthread_sigmask",
            lambda *_args: (_ for _ in ()).throw(OSError("pthread failed")),
        )
        scoped.setattr(tool.subprocess, "Popen", forbidden)
        with pytest.raises(OSError, match="pthread failed"):
            tool._spawn_supervisor_once(capsule, callback)
    assert state == {
        "phase": "SPAWN_PREPARE",
        "supervisor_spawn_attempted": False,
        "supervisor_called": False,
        "supervisor_invocation_count": 0,
    }
    assert umask_values == [0o077, 0o022]

    state, callback = state_and_callback()
    popen_calls: list[str] = []
    with monkeypatch.context() as scoped:
        scoped.setattr(tool, "_replay_capsule_executable_identities", lambda _capsule: None)
        scoped.setattr(tool.os, "umask", lambda _value: 0o022)
        scoped.setattr(tool.signal, "pthread_sigmask", lambda _how, _mask: set())

        def popen(*_args, **_kwargs):
            popen_calls.append("Popen")
            raise OSError("Popen failed")

        scoped.setattr(tool.subprocess, "Popen", popen)
        with pytest.raises(OSError, match="Popen failed"):
            tool._spawn_supervisor_once(capsule, callback)
    assert popen_calls == ["Popen"]
    assert state == {
        "phase": "SUPERVISOR_SPAWN",
        "supervisor_spawn_attempted": True,
        "supervisor_called": True,
        "supervisor_invocation_count": 1,
    }


def test_outcome_publication_rejects_existing_success_failure_sibling(
    monkeypatch, tool
) -> None:
    opened: list[str] = []

    def forbidden_open(name, *_args, **_kwargs):
        opened.append(os.fspath(name))
        raise AssertionError("publication opened a target despite an outcome sibling")

    monkeypatch.setattr(tool.os, "open", forbidden_open)
    for target, sibling in (
        ("shadow-trial.success.json", "shadow-trial.failure.json"),
        ("shadow-trial.failure.json", "shadow-trial.success.json"),
    ):
        monkeypatch.setattr(tool.os, "listdir", lambda _fd, row=sibling: [row])
        with pytest.raises(tool.ShadowNonadoptionTrialError):
            tool._publish_ledger_record(99, target, {"synthetic": True})
    assert opened == []


def test_ledger_records_enforce_one_mib_bound_before_write_and_during_read(
    monkeypatch, tool
) -> None:
    assert tool._LEDGER_RECORD_MAX_BYTES == 1 << 20
    opened_for_write: list[str] = []

    def forbidden_write_open(name, *_args, **_kwargs):
        opened_for_write.append(os.fspath(name))
        raise AssertionError("oversized ledger publication opened a file")

    monkeypatch.setattr(tool.os, "open", forbidden_write_open)
    with pytest.raises(tool.ShadowNonadoptionTrialError, match="byte bound"):
        tool._publish_ledger_record(
            99,
            "synthetic-ledger-record.json",
            {"payload": "x" * tool._LEDGER_RECORD_MAX_BYTES},
        )
    assert opened_for_write == []

    def ledger_stat(size: int) -> Any:
        return types.SimpleNamespace(
            st_dev=1,
            st_ino=2,
            st_mode=0o100444,
            st_uid=1000,
            st_nlink=1,
            st_size=size,
            st_mtime_ns=3,
            st_ctime_ns=4,
        )

    monkeypatch.setattr(tool.os, "geteuid", lambda: 1000)
    closed: list[int] = []
    with monkeypatch.context() as scoped:
        scoped.setattr(tool.os, "open", lambda *_args, **_kwargs: 51)
        scoped.setattr(
            tool.os,
            "fstat",
            lambda fd: ledger_stat(tool._LEDGER_RECORD_MAX_BYTES + 1)
            if fd == 51
            else (_ for _ in ()).throw(AssertionError("unexpected ledger fd")),
        )
        scoped.setattr(
            tool.os,
            "read",
            lambda *_args: (_ for _ in ()).throw(
                AssertionError("oversized ledger identity reached read")
            ),
        )
        scoped.setattr(tool.os, "close", closed.append)
        with pytest.raises(tool.ShadowNonadoptionTrialError, match="identity differs"):
            tool._read_ledger_record_at(99, "oversized.json", "oversized")
    assert closed == [51]

    chunks = [b"x" * tool._LEDGER_RECORD_MAX_BYTES, b"y"]
    closed.clear()
    with monkeypatch.context() as scoped:
        scoped.setattr(tool.os, "open", lambda *_args, **_kwargs: 52)
        scoped.setattr(tool.os, "fstat", lambda fd: ledger_stat(1))
        scoped.setattr(tool.os, "read", lambda _fd, _size: chunks.pop(0))
        scoped.setattr(tool.os, "close", closed.append)
        with pytest.raises(tool.ShadowNonadoptionTrialError, match="exceeds its byte bound"):
            tool._read_ledger_record_at(99, "growing.json", "growing")
    assert chunks == []
    assert closed == [52]


def test_intent_consumed_and_outcome_child_replacement_fails_after_same_fd_replay(
    monkeypatch, tool
) -> None:
    directory_fd = 59
    record_fd = 60
    raw = b'{"synthetic":true}\n'
    held = types.SimpleNamespace(
        st_dev=1,
        st_ino=2,
        st_mode=0o100444,
        st_uid=1000,
        st_nlink=1,
        st_size=len(raw),
        st_mtime_ns=3,
        st_ctime_ns=4,
    )
    replacement = types.SimpleNamespace(
        st_dev=held.st_dev,
        st_ino=held.st_ino + 1,
        st_mode=held.st_mode,
        st_uid=held.st_uid,
        st_nlink=held.st_nlink,
        st_size=held.st_size,
        st_mtime_ns=held.st_mtime_ns,
        st_ctime_ns=held.st_ctime_ns,
    )
    record_names = (
        "shadow-trial.intent.json",
        "shadow-trial.consumed.json",
        "shadow-trial.success.json",
        "shadow-trial.failure.json",
    )
    monkeypatch.setattr(tool.os, "geteuid", lambda: 1000)
    for name in record_names:
        reads = [raw, b""]
        opened: list[tuple[str, int]] = []
        pathname_stats: list[tuple[str, int, bool]] = []
        closed: list[int] = []

        def open_record(
            observed_name: str, _flags: int, *, dir_fd: int
        ) -> int:
            opened.append((observed_name, dir_fd))
            return record_fd

        def pathname_stat(
            observed_name: str, *, dir_fd: int, follow_symlinks: bool
        ) -> Any:
            pathname_stats.append((observed_name, dir_fd, follow_symlinks))
            return replacement

        with monkeypatch.context() as scoped:
            scoped.setattr(tool.os, "open", open_record)
            scoped.setattr(tool.os, "fstat", lambda fd: held if fd == record_fd else None)
            scoped.setattr(tool.os, "read", lambda fd, _size: reads.pop(0) if fd == record_fd else b"")
            scoped.setattr(tool.os, "stat", pathname_stat)
            scoped.setattr(tool.os, "close", closed.append)
            with pytest.raises(
                tool.ShadowNonadoptionTrialError,
                match="ledger pathname identity changed during replay",
            ):
                tool._read_ledger_record_at(directory_fd, name, name)
        assert reads == []
        assert opened == [(name, directory_fd)]
        assert pathname_stats == [(name, directory_fd, False)]
        assert closed == [record_fd]


def test_held_ledger_path_rejects_rename_or_replacement_and_allows_parent_nlink_growth(
    monkeypatch, tool
) -> None:
    parent_path = Path("/fixed/shadow-ledgers")
    ledger_name = "trial-v1"
    parent_fd = 61
    ledger_fd = 62
    expected_parent = {
        "dev": 1,
        "ino": 2,
        "mode": 0o040700,
        "uid": 1000,
        "nlink": 2,
        "mnt_id": 3,
    }
    expected_ledger = {
        "dev": 1,
        "ino": 4,
        "mode": 0o040700,
        "uid": 1000,
        "nlink": 2,
        "mnt_id": 3,
    }
    parent_now = {**expected_parent, "nlink": 9}
    parent_replay = {
        "path": os.fspath(parent_path),
        "exists": True,
        "existing_prefix": os.fspath(parent_path),
        **parent_now,
    }

    def directory_identity(fd: int, label: str) -> dict[str, int]:
        assert (fd, label) in {
            (parent_fd, "held ledger parent"),
            (ledger_fd, "held ledger root"),
        }
        return deepcopy(parent_now if fd == parent_fd else expected_ledger)

    def ledger_stat(*, ino: int = expected_ledger["ino"]) -> Any:
        return types.SimpleNamespace(
            st_dev=expected_ledger["dev"],
            st_ino=ino,
            st_mode=expected_ledger["mode"],
            st_uid=expected_ledger["uid"],
            st_nlink=expected_ledger["nlink"],
        )

    stat_calls: list[tuple[str, int, bool]] = []

    def pathname_stat(name: str, *, dir_fd: int, follow_symlinks: bool):
        stat_calls.append((name, dir_fd, follow_symlinks))
        return ledger_stat()

    monkeypatch.setattr(tool.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(tool, "_held_directory_identity", directory_identity)
    monkeypatch.setattr(tool, "_path_snapshot", lambda *_args: parent_replay)
    monkeypatch.setattr(tool.os, "stat", pathname_stat)
    tool._verify_held_ledger_path(
        parent_path,
        ledger_name,
        parent_fd,
        ledger_fd,
        expected_parent,
        expected_ledger,
    )
    assert stat_calls == [(ledger_name, parent_fd, False)]

    with monkeypatch.context() as scoped:
        scoped.setattr(
            tool.os,
            "stat",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                FileNotFoundError("synthetic ledger rename")
            ),
        )
        with pytest.raises(
            tool.ShadowNonadoptionTrialError, match="pathname cannot be replayed"
        ):
            tool._verify_held_ledger_path(
                parent_path,
                ledger_name,
                parent_fd,
                ledger_fd,
                expected_parent,
                expected_ledger,
            )

    with monkeypatch.context() as scoped:
        scoped.setattr(tool.os, "stat", lambda *_args, **_kwargs: ledger_stat(ino=99))
        with pytest.raises(
            tool.ShadowNonadoptionTrialError, match="root pathname identity changed"
        ):
            tool._verify_held_ledger_path(
                parent_path,
                ledger_name,
                parent_fd,
                ledger_fd,
                expected_parent,
                expected_ledger,
            )

    with monkeypatch.context() as scoped:
        scoped.setattr(
            tool,
            "_path_snapshot",
            lambda *_args: {**parent_replay, "ino": 98},
        )
        with pytest.raises(
            tool.ShadowNonadoptionTrialError, match="parent pathname identity changed"
        ):
            tool._verify_held_ledger_path(
                parent_path,
                ledger_name,
                parent_fd,
                ledger_fd,
                expected_parent,
                expected_ledger,
            )

    with monkeypatch.context() as scoped:
        scoped.setattr(
            tool,
            "_held_directory_identity",
            lambda fd, _label: (
                deepcopy(parent_now)
                if fd == parent_fd
                else {**expected_ledger, "ino": 97}
            ),
        )
        with pytest.raises(
            tool.ShadowNonadoptionTrialError, match="directory identity changed"
        ):
            tool._verify_held_ledger_path(
                parent_path,
                ledger_name,
                parent_fd,
                ledger_fd,
                expected_parent,
                expected_ledger,
            )


def test_spawn_replays_python_git_and_docker_identity_immediately_before_popen(
    monkeypatch, tool
) -> None:
    identities = {
        "Python": _executable_identity("/fixed/python", "a"),
        "Git": _executable_identity("/fixed/git", "b"),
        "Docker": _executable_identity("/fixed/docker", "c"),
    }
    capsule = {
        "python_executable_identity": identities["Python"],
        "git_executable_identity": identities["Git"],
        "docker_executable_identity": identities["Docker"],
        "ordered_argv": ["/fixed/python", "supervisor.py", "--run"],
        "clone_project_root": "/fixed/clone/Hegel Machine",
        "runtime_environment": {"PATH": "/usr/bin:/bin"},
    }
    events: list[str] = []

    def executable(path: str, label: str) -> dict[str, Any]:
        events.append(f"identity:{label}")
        assert path == identities[label]["path"]
        return identities[label]

    monkeypatch.setattr(tool, "_executable_identity", executable)
    monkeypatch.setattr(
        tool.signal,
        "pthread_sigmask",
        lambda _how, mask: set() if mask == set() else set(),
    )
    umasks: list[int] = []

    def umask(value: int) -> int:
        umasks.append(value)
        return 0o022

    monkeypatch.setattr(tool.os, "umask", umask)
    sentinel = object()
    popen_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def popen(*args, **kwargs):
        events.append("popen")
        popen_calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(tool.subprocess, "Popen", popen)
    assert tool._spawn_supervisor_once(
        capsule, lambda: events.append("boundary")
    ) == (sentinel, None)
    assert events == [
        "identity:Python",
        "identity:Git",
        "identity:Docker",
        "boundary",
        "popen",
    ]
    assert umasks == [0o077, 0o022]
    assert len(popen_calls) == 1
    args, kwargs = popen_calls[0]
    assert args == (capsule["ordered_argv"],)
    assert kwargs == {
        "cwd": capsule["clone_project_root"],
        "env": capsule["runtime_environment"],
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "start_new_session": True,
        "close_fds": True,
        "pass_fds": (),
        "restore_signals": True,
    }

    events.clear()

    def docker_drift(path: str, label: str) -> dict[str, Any]:
        events.append(f"identity:{label}")
        if label == "Docker":
            changed = deepcopy(identities[label])
            changed["ino"] += 1
            return changed
        return identities[label]

    monkeypatch.setattr(tool, "_executable_identity", docker_drift)
    with pytest.raises(tool.ShadowNonadoptionTrialError, match="Docker executable"):
        tool._spawn_supervisor_once(capsule, lambda: events.append("boundary"))
    assert events == ["identity:Python", "identity:Git", "identity:Docker"]


def test_git_readonly_wrapper_uses_fixed_hardened_prefix_and_environment(
    monkeypatch, contract, tool
) -> None:
    frozen = _resign_contract(contract)
    source = frozen["isolated_source_policy"]
    git_identity = _executable_identity(source["git_executable_path"], "b")
    identity_calls: list[tuple[str, str]] = []
    run_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def identity(path: str, label: str) -> dict[str, Any]:
        identity_calls.append((path, label))
        return git_identity

    def run(*args, **kwargs):
        run_calls.append((args, kwargs))
        return types.SimpleNamespace(returncode=0, stdout=b"readonly-output", stderr=b"")

    monkeypatch.setattr(tool, "_executable_identity", identity)
    monkeypatch.setattr(tool.subprocess, "run", run)
    arguments = ["-C", source["clone_source_repository"], "rev-parse", "HEAD"]
    assert tool._git_readonly(
        frozen, arguments, "readonly hardening KAT", git_identity
    ) == b"readonly-output"
    assert identity_calls == [
        (source["git_executable_path"], "Git"),
        (source["git_executable_path"], "Git"),
    ]
    assert len(run_calls) == 1
    args, kwargs = run_calls[0]
    assert args == (
        [
            source["git_executable_path"],
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            *arguments,
        ],
    )
    assert kwargs == {
        "cwd": "/",
        "env": source["git_runtime_environment"],
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "close_fds": True,
        "check": False,
    }


def test_run_exact_fixes_git_umask_and_restores_ambient_on_all_failure_paths(
    monkeypatch, contract, tool
) -> None:
    frozen = _resign_contract(contract)
    source = frozen["isolated_source_policy"]
    argv = source["git_remove_origin_argv"]
    environment = source["git_runtime_environment"]
    git_identity = _executable_identity(source["git_executable_path"], "b")
    events: list[str] = []
    run_kwargs: list[dict[str, Any]] = []

    def identity(path: str, label: str) -> dict[str, Any]:
        assert (path, label) == (source["git_executable_path"], "Git")
        events.append("identity")
        return git_identity

    def umask(value: int) -> int:
        events.append(f"umask:{value:o}")
        return 0o077 if value == 0o022 else 0o022

    def success_run(observed_argv, **kwargs):
        events.append("subprocess")
        run_kwargs.append(kwargs)
        assert observed_argv == argv
        return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(tool, "_executable_identity", identity)
    monkeypatch.setattr(tool.os, "umask", umask)
    monkeypatch.setattr(tool.subprocess, "run", success_run)
    result = tool._run_exact(
        argv,
        environment,
        "Git mutation",
        git_identity,
        source["git_mutator_umask"],
    )
    assert result.returncode == 0
    assert events == [
        "identity",
        "umask:22",
        "subprocess",
        "umask:77",
        "identity",
    ]
    assert run_kwargs == [
        {
            "cwd": "/",
            "env": environment,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "close_fds": True,
            "check": False,
        }
    ]

    for subprocess_failure, restore_failure, expected in (
        (True, False, "Git mutation raised: OSError"),
        (False, True, "Git mutation umask restoration raised: OSError"),
        (True, True, "Git mutation raised: OSError; umask restoration raised: OSError"),
    ):
        case_events: list[str] = []
        umask_count = {"value": 0}

        def case_identity(_path: str, _label: str) -> dict[str, Any]:
            case_events.append("identity")
            return git_identity

        def case_umask(value: int) -> int:
            umask_count["value"] += 1
            case_events.append(f"umask:{value:o}")
            if umask_count["value"] == 2 and restore_failure:
                raise OSError("synthetic restore failure")
            return 0o077 if value == 0o022 else 0o022

        def case_run(*_args, **_kwargs):
            case_events.append("subprocess")
            if subprocess_failure:
                raise OSError("synthetic subprocess failure")
            return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        with monkeypatch.context() as scoped:
            scoped.setattr(tool, "_executable_identity", case_identity)
            scoped.setattr(tool.os, "umask", case_umask)
            scoped.setattr(tool.subprocess, "run", case_run)
            with pytest.raises(tool.ShadowNonadoptionTrialError, match=expected):
                tool._run_exact(
                    argv,
                    environment,
                    "Git mutation",
                    git_identity,
                    0o022,
                )
        assert case_events == [
            "identity",
            "umask:22",
            "subprocess",
            "umask:77",
        ]

    with pytest.raises(
        tool.ShadowNonadoptionTrialError, match="Git mutator umask differs"
    ):
        tool._run_exact(argv, environment, "Git mutation", git_identity, 0o077)


def test_clone_preparation_uses_only_the_three_exact_frozen_git_argv_rows(
    monkeypatch, contract, tool
) -> None:
    frozen = _resign_contract(contract)
    source = frozen["isolated_source_policy"]
    assert source["git_clone_argv"] == [
        "/usr/bin/git",
        "-c",
        "protocol.allow=never",
        "-c",
        "protocol.file.allow=always",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "clone",
        "--quiet",
        "--shared",
        "--no-checkout",
        "--no-tags",
        "--no-recurse-submodules",
        "--",
        "/home/erzhu419/mine_code/Asumption Agent",
        source["clone_root"],
    ]
    assert source["git_detached_checkout_argv"] == [
        "/usr/bin/git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        f"--git-dir={source['clone_git_dir']}",
        f"--work-tree={source['clone_root']}",
        "checkout",
        "--quiet",
        "--detach",
        "--force",
        FROZEN_COMMIT,
    ]
    assert source["git_remove_origin_argv"] == [
        "/usr/bin/git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        f"--git-dir={source['clone_git_dir']}",
        f"--work-tree={source['clone_root']}",
        "remote",
        "remove",
        "origin",
    ]
    assert source["git_mutator_umask"] == 0o022
    joined = "\0".join(
        source["git_clone_argv"]
        + source["git_detached_checkout_argv"]
        + source["git_remove_origin_argv"]
    )
    assert "fetch" not in joined and "pull" not in joined and "submodule update" not in joined
    common_git_hardening = {
        "GIT_ALLOW_PROTOCOL": "file",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_KEY_0": "core.fsmonitor",
        "GIT_CONFIG_KEY_1": "core.hooksPath",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_VALUE_0": "false",
        "GIT_CONFIG_VALUE_1": "/dev/null",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    wrapper_environment = source["git_runtime_environment"]
    child_environment = frozen["capsule_policy"]["fixed_runtime_environment"]
    for environment in (wrapper_environment, child_environment):
        assert {
            key: environment[key] for key in common_git_hardening
        } == common_git_hardening
        assert not {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"}.intersection(
            environment
        )
    assert wrapper_environment["GIT_CONFIG_COUNT"] == "2"
    assert "GIT_CONFIG_KEY_2" not in wrapper_environment
    assert "GIT_CONFIG_VALUE_2" not in wrapper_environment
    assert child_environment["GIT_CONFIG_COUNT"] == "3"
    assert child_environment["GIT_CONFIG_KEY_2"] == "core.worktree"
    assert child_environment["GIT_CONFIG_VALUE_2"] == source["clone_root"]

    git_identity = _executable_identity(source["git_executable_path"], "b")
    capsule = {"git_executable_identity": git_identity}
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    events: list[str] = []
    audits: list[str] = []
    umask_calls: list[int] = []
    monkeypatch.setattr(tool.os.path, "lexists", lambda _path: False)
    monkeypatch.setattr(
        tool, "_verify_path_boundaries", lambda _contract: events.append("paths")
    )

    labels = {
        tuple(source["git_clone_argv"]): "Git clone",
        tuple(source["git_detached_checkout_argv"]): "Git detached checkout",
        tuple(source["git_remove_origin_argv"]): "Git remote removal",
    }

    def identity(path: str, label: str) -> dict[str, Any]:
        assert (path, label) == (source["git_executable_path"], "Git")
        events.append("identity:Git")
        return git_identity

    def run_exact(*args, **kwargs):
        calls.append((args, kwargs))
        events.append(f"run:{labels[tuple(args[0])]}")
        return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    def umask(value: int) -> int:
        umask_calls.append(value)
        return 0o077 if value == 0o022 else 0o022

    git_dir_identity = {"synthetic": "fixed-clone-git-dir"}

    def mutator_boundary(_contract, expected):
        if expected is None:
            events.append("gitdir:first")
            return git_dir_identity
        assert expected is git_dir_identity
        events.append("gitdir:same")
        return git_dir_identity

    monkeypatch.setattr(tool, "_executable_identity", identity)
    monkeypatch.setattr(tool.subprocess, "run", run_exact)
    monkeypatch.setattr(tool.os, "umask", umask)
    monkeypatch.setattr(tool, "_clone_git_mutator_boundary", mutator_boundary)
    monkeypatch.setattr(
        tool,
        "_verify_isolated_clone",
        lambda _contract, observed_capsule, *, require_artifact_absent: audits.append(
            f"verify:{require_artifact_absent}"
        )
        if observed_capsule is capsule
        else (_ for _ in ()).throw(AssertionError("capsule identity changed")),
    )
    monkeypatch.setattr(tool, "_audit_event", audits.append)
    tool._prepare_isolated_clone(frozen, capsule)
    assert events == [
        "paths",
        "identity:Git",
        "run:Git clone",
        "identity:Git",
        "paths",
        "gitdir:first",
        "identity:Git",
        "run:Git detached checkout",
        "identity:Git",
        "paths",
        "gitdir:same",
        "identity:Git",
        "run:Git remote removal",
        "identity:Git",
        "paths",
        "gitdir:same",
    ]
    assert umask_calls == [0o022, 0o077] * 3
    assert [list(args[0]) for args, _kwargs in calls] == [
        source["git_clone_argv"],
        source["git_detached_checkout_argv"],
        source["git_remove_origin_argv"],
    ]
    for args, kwargs in calls:
        assert len(args) == 1
        assert kwargs == {
            "cwd": "/",
            "env": source["git_runtime_environment"],
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "close_fds": True,
            "check": False,
        }
    assert audits == ["before_git_clone", "verify:True", "after_git_clone"]

    events.clear()
    calls.clear()
    replay_count = {"value": 0}

    def post_mutation_drift(path: str, label: str) -> dict[str, Any]:
        assert (path, label) == (source["git_executable_path"], "Git")
        replay_count["value"] += 1
        events.append("identity:Git")
        if replay_count["value"] == 2:
            changed = deepcopy(git_identity)
            changed["ino"] += 1
            return changed
        return git_identity

    monkeypatch.setattr(tool, "_executable_identity", post_mutation_drift)
    with pytest.raises(tool.ShadowNonadoptionTrialError, match="Git clone Git executable"):
        tool._prepare_isolated_clone(frozen, capsule)
    assert events == ["paths", "identity:Git", "run:Git clone", "identity:Git"]
    assert len(calls) == 1


def test_docker_surface_is_read_only_exact_identity_inventory_only(
    monkeypatch, contract, tool
) -> None:
    frozen = _resign_contract(contract)
    policy = frozen["capsule_policy"]["fixed_kill_reap_survivor_policy"]
    assert policy["docker_mutation_argv"] is None
    assert policy["docker_wrapper_container_mutation_allowed"] is False
    assert policy["docker_list_argv_template"][:3] == [
        "/usr/bin/docker",
        "--host=unix:///var/run/docker.sock",
        "ps",
    ]
    assert policy["docker_slot_values"] == [
        "RUST_TEST",
        "RUST_RELEASE",
        "PYTHON_ENDPOINT",
        "RUST_ENDPOINT",
        "TRUSTED_HOST_REPLAY",
    ]
    docker_identity = _executable_identity(policy["docker_executable_path"], "c")
    capsule = {
        "kill_reap_survivor_policy": deepcopy(policy),
        "docker_executable_identity": docker_identity,
        "runtime_environment": {"PATH": "/usr/bin:/bin"},
    }
    monkeypatch.setattr(
        tool,
        "_executable_identity",
        lambda path, label: docker_identity
        if (path, label) == (policy["docker_executable_path"], "Docker")
        else (_ for _ in ()).throw(AssertionError("unexpected identity replay")),
    )
    stdout_rows = [b""]
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return types.SimpleNamespace(returncode=0, stdout=stdout_rows[-1], stderr=b"")

    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    assert tool._docker_inventory(frozen, capsule) == []
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (policy["docker_list_argv_template"],)
    assert kwargs == {
        "cwd": "/",
        "env": capsule["runtime_environment"],
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "close_fds": True,
        "check": False,
    }

    container_id = "a" * 64
    namespace = "b" * 64
    commit = policy["docker_source_commit_label_value"]
    stdout_rows.append(
        f"{container_id}|{namespace}|RUST_TEST|{commit}\n".encode("ascii")
    )
    assert tool._docker_inventory(frozen, capsule) == [
        {
            "container_id": container_id,
            "execution_namespace": namespace,
            "slot": "RUST_TEST",
            "source_commit": commit,
        }
    ]

    attacks = (
        f"short|{namespace}|RUST_TEST|{commit}\n",
        f"{container_id}|{namespace}|rust-test|{commit}\n",
        f"{container_id}|{namespace}|RUST_TEST|{'0' * 40}\n",
        (
            f"{container_id}|{namespace}|RUST_TEST|{commit}\n"
            f"{'c' * 64}|{namespace}|RUST_TEST|{commit}\n"
        ),
        (
            f"{container_id}|{namespace}|RUST_TEST|{commit}\n"
            f"{'c' * 64}|{'d' * 64}|RUST_RELEASE|{commit}\n"
        ),
    )
    for attack in attacks:
        stdout_rows.append(attack.encode("ascii"))
        with pytest.raises(tool.ShadowNonadoptionTrialError):
            tool._docker_inventory(frozen, capsule)


def test_owner_authorization_root_and_file_identity_reject_before_mutation(
    monkeypatch, contract, tool
) -> None:
    frozen = _resign_contract(contract)
    capsule = {"capsule_digest": TAGGED_CAPSULE}
    mutation_calls: list[str] = []

    def forbidden_mutation(*_args, **_kwargs):
        mutation_calls.append("mutation")
        raise AssertionError("authorization identity failure reached a mutation")

    monkeypatch.setattr(tool, "_verify_wrapper_entrypoint", lambda: None)
    monkeypatch.setattr(
        tool, "_reject_preexisting_ledger_before_preflight", lambda _contract: None
    )
    monkeypatch.setattr(
        tool, "load_phase3_q05b_shadow_nonadoption_trial_contract_v1", lambda: frozen
    )
    monkeypatch.setattr(
        tool,
        "_preflight_context",
        lambda _contract: {"capsule": capsule, "cargo_rows": []},
    )
    monkeypatch.setattr(tool.os, "geteuid", lambda: 1000)
    for name in (
        "_verify_path_boundaries",
        "_create_and_hold_ledger_root",
        "_publish_ledger_record",
        "_prepare_isolated_clone",
        "_prepare_isolated_cargo_cache",
        "_prepare_supervisor_spawn",
        "_spawn_supervisor_once",
    ):
        monkeypatch.setattr(tool, name, forbidden_mutation)
    monkeypatch.setattr(tool.subprocess, "Popen", forbidden_mutation)

    def identity_stat(
        *,
        mode: int,
        uid: int = 1000,
        nlink: int,
        dev: int,
        ino: int,
    ) -> Any:
        return types.SimpleNamespace(
            st_dev=dev,
            st_ino=ino,
            st_mode=mode,
            st_uid=uid,
            st_nlink=nlink,
            st_size=17,
            st_mtime_ns=3,
            st_ctime_ns=4,
        )

    valid_root = identity_stat(
        mode=0o040700, nlink=2, dev=10, ino=11
    )
    valid_file = identity_stat(
        mode=0o100444, nlink=1, dev=10, ino=12
    )

    def root_snapshot(root_stat: Any, mount_id: int = 13) -> dict[str, Any]:
        return {
            "path": os.fspath(tool._authorization_path(frozen).parent),
            "exists": True,
            "existing_prefix": os.fspath(tool._authorization_path(frozen).parent),
            "mode": root_stat.st_mode,
            "uid": root_stat.st_uid,
            "nlink": root_stat.st_nlink,
            "dev": root_stat.st_dev,
            "ino": root_stat.st_ino,
            "mnt_id": mount_id,
        }

    attacks = (
        (
            "root mode",
            identity_stat(mode=0o040755, nlink=2, dev=10, ino=11),
            valid_file,
            root_snapshot(valid_root),
            valid_root,
            False,
            "root identity",
        ),
        (
            "root euid",
            identity_stat(mode=0o040700, uid=1001, nlink=2, dev=10, ino=11),
            valid_file,
            root_snapshot(valid_root),
            valid_root,
            False,
            "root identity",
        ),
        (
            "file mode",
            valid_root,
            identity_stat(mode=0o100644, nlink=1, dev=10, ino=12),
            root_snapshot(valid_root),
            valid_root,
            True,
            "filesystem identity",
        ),
        (
            "file euid",
            valid_root,
            identity_stat(mode=0o100444, uid=1001, nlink=1, dev=10, ino=12),
            root_snapshot(valid_root),
            valid_root,
            True,
            "filesystem identity",
        ),
        (
            "file nlink",
            valid_root,
            identity_stat(mode=0o100444, nlink=2, dev=10, ino=12),
            root_snapshot(valid_root),
            valid_root,
            True,
            "filesystem identity",
        ),
        (
            "root pathname swap",
            valid_root,
            valid_file,
            {
                **root_snapshot(valid_root),
                "ino": valid_root.st_ino + 1,
            },
            valid_root,
            True,
            "root changed during replay",
        ),
        (
            "root held-fd nlink change",
            valid_root,
            valid_file,
            root_snapshot(valid_root),
            identity_stat(mode=0o040700, nlink=3, dev=10, ino=11),
            True,
            "root changed during replay",
        ),
    )
    authorization_path = tool._authorization_path(frozen)
    for (
        label,
        root_stat,
        file_stat,
        replay_snapshot,
        held_after,
        reader_expected,
        expected_message,
    ) in attacks:
        reader_calls: list[tuple[int, str, str, int]] = []
        closed: list[int] = []

        def open_root(path: Path, open_label: str):
            assert path == authorization_path.parent
            assert open_label == "owner authorization root"
            return 71, root_stat, 13

        def reader(directory_fd: int, name: str, reader_label: str, maximum: int):
            reader_calls.append((directory_fd, name, reader_label, maximum))
            return b"{}\n", file_stat

        with monkeypatch.context() as scoped:
            scoped.setattr(tool, "_open_held_directory_nofollow", open_root)
            scoped.setattr(tool, "_read_regular_at_nofollow", reader)
            scoped.setattr(tool, "_path_snapshot", lambda *_args: replay_snapshot)
            scoped.setattr(tool.os, "fstat", lambda fd: held_after if fd == 71 else None)
            scoped.setattr(tool.os, "close", closed.append)
            scoped.setattr(
                tool,
                "_core",
                lambda: (_ for _ in ()).throw(
                    AssertionError("authorization validator reached after identity drift")
                ),
            )
            with pytest.raises(tool.ShadowNonadoptionTrialError, match=expected_message):
                tool.run_phase3_q05b_shadow_nonadoption_trial_v1()
        assert reader_calls == (
            [(71, authorization_path.name, "owner authorization", 1 << 20)]
            if reader_expected
            else []
        )
        assert closed == [71], label
    assert mutation_calls == []


def test_placeholder_or_missing_authorization_rejects_before_mutation(
    monkeypatch, contract, tool
) -> None:
    calls: list[str] = []
    frozen = _placeholder_contract(contract)
    capsule = {"capsule_digest": TAGGED_CAPSULE}

    def forbidden(*_args, **_kwargs):
        calls.append("mutation")
        raise AssertionError("mutation occurred before authorization rejection")

    monkeypatch.setattr(tool, "_verify_wrapper_entrypoint", lambda: None)
    monkeypatch.setattr(
        tool, "_reject_preexisting_ledger_before_preflight", lambda _contract: None
    )
    for name in (
        "_prepare_isolated_clone",
        "_prepare_isolated_cargo_cache",
        "_publish_ledger_record",
        "_spawn_supervisor_once",
    ):
        monkeypatch.setattr(tool, name, forbidden)
    monkeypatch.setattr(
        tool, "load_phase3_q05b_shadow_nonadoption_trial_contract_v1", lambda: frozen
    )
    monkeypatch.setattr(
        tool,
        "_preflight_context",
        lambda _contract: {"capsule": capsule, "cargo_rows": []},
    )
    monkeypatch.setattr(
        tool,
        "_load_authorization",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            tool.ShadowNonadoptionTrialError("authorization absent")
        ),
    )
    monkeypatch.setattr(
        tool,
        "_create_and_hold_ledger_root",
        forbidden,
    )
    with pytest.raises(tool.ShadowNonadoptionTrialError):
        tool.run_phase3_q05b_shadow_nonadoption_trial_v1()
    assert calls == []
    assert frozen["source_binding"]["status"].startswith("PLACEHOLDER_")


def test_preexisting_ledger_is_terminal_before_preflight_with_nullable_wire_facts(
    monkeypatch, contract, tool
) -> None:
    frozen = _resign_contract(contract)
    ledger = frozen["trial_ledger_policy"]
    ledger_root = Path(ledger["ledger_root"])
    valid_root = types.SimpleNamespace(
        st_dev=1,
        st_ino=2,
        st_mode=0o040700,
        st_uid=1000,
        st_nlink=2,
        st_size=0,
        st_mtime_ns=3,
        st_ctime_ns=4,
    )
    valid_record = types.SimpleNamespace(
        st_mode=0o100444,
        st_uid=1000,
        st_nlink=1,
    )
    intent_name = ledger["intent_record_name"]
    consumed_name = ledger["consume_record_name"]
    scenarios = (
        ("root only", set(), {}, None, None),
        ("durable intent", {intent_name}, {intent_name: b'{"intent":true}\n'}, True, None),
        (
            "partial consumed record",
            {intent_name, consumed_name},
            {intent_name: b'{"intent":true}\n', consumed_name: b'{"partial"'},
            True,
            None,
        ),
        (
            "durable consumption",
            {intent_name, consumed_name},
            {
                intent_name: b'{"intent":true}\n',
                consumed_name: b'{"consumed":true}\n',
            },
            True,
            True,
        ),
    )

    for mode in ("preflight", "run"):
        for label, names, payloads, expected_intent, expected_consumed in scenarios:
            preflight_calls: list[str] = []
            supervisor_calls: list[str] = []
            closed: list[int] = []

            def forbidden_preflight(*_args, **_kwargs):
                preflight_calls.append("preflight")
                raise AssertionError("existing ledger reached preflight context")

            def forbidden_supervisor(*_args, **_kwargs):
                supervisor_calls.append("supervisor")
                raise AssertionError("existing ledger reached supervisor")

            def read_record(
                directory_fd: int, name: str, _read_label: str, maximum: int
            ):
                assert directory_fd == 91
                assert maximum == 1 << 20
                return payloads[name], valid_record

            with monkeypatch.context() as scoped:
                scoped.setattr(tool, "_verify_wrapper_entrypoint", lambda: None)
                scoped.setattr(
                    tool,
                    "load_phase3_q05b_shadow_nonadoption_trial_contract_v1",
                    lambda: frozen,
                )
                scoped.setattr(tool.os.path, "lexists", lambda path: Path(path) == ledger_root)
                scoped.setattr(
                    tool,
                    "_open_held_directory_nofollow",
                    lambda path, open_label: (91, valid_root, 5)
                    if (path, open_label) == (ledger_root, "preexisting ledger root")
                    else (_ for _ in ()).throw(AssertionError("ledger root open differs")),
                )
                scoped.setattr(
                    tool,
                    "_path_snapshot",
                    lambda path, replay_label: {
                        "path": path,
                        "exists": True,
                        "existing_prefix": path,
                        "mode": valid_root.st_mode,
                        "uid": valid_root.st_uid,
                        "nlink": valid_root.st_nlink,
                        "dev": valid_root.st_dev,
                        "ino": valid_root.st_ino,
                        "mnt_id": 5,
                    }
                    if replay_label == "preexisting ledger root replay"
                    else (_ for _ in ()).throw(AssertionError("ledger replay differs")),
                )
                scoped.setattr(tool.os, "geteuid", lambda: 1000)
                scoped.setattr(tool.os, "listdir", lambda fd: list(names) if fd == 91 else [])
                scoped.setattr(tool.os, "close", closed.append)
                scoped.setattr(tool, "_read_regular_at_nofollow", read_record)
                scoped.setattr(tool, "_preflight_context", forbidden_preflight)
                scoped.setattr(tool, "_spawn_supervisor_once", forbidden_supervisor)
                entrypoint = (
                    tool.preflight_phase3_q05b_shadow_nonadoption_trial_v1
                    if mode == "preflight"
                    else tool.run_phase3_q05b_shadow_nonadoption_trial_v1
                )
                with pytest.raises(
                    tool.ShadowNonadoptionTrialError,
                    match="resume and retry are forbidden",
                ) as captured:
                    entrypoint()

            error = captured.value
            assert error.terminal_after_intent is True, (mode, label)
            assert error.receipt is None
            assert error.wire_state == {
                "intent_staged": expected_intent,
                "authorization_consumed": expected_consumed,
                "supervisor_called": None,
                "trial_execution_performed": None,
                "supervisor_invocation_count": None,
            }
            wire = tool._error_wire(error)
            assert wire["status"] == "SHADOW_NONADOPTION_TRIAL_TERMINAL_FAILURE"
            assert wire["receipt"] is None
            assert wire["intent_staged"] is expected_intent
            assert wire["authorization_consumed"] is expected_consumed
            assert wire["supervisor_called"] is None
            assert wire["trial_execution_performed"] is None
            assert wire["supervisor_invocation_count"] is None
            assert preflight_calls == []
            assert supervisor_calls == []
            assert closed == [91]


def test_preflight_is_read_only_and_never_calls_supervisor(
    monkeypatch, contract, tool
) -> None:
    calls: list[str] = []
    frozen = _resign_contract(contract)
    capsule = {"capsule_digest": TAGGED_CAPSULE}

    def forbidden(*_args, **_kwargs):
        calls.append("forbidden")
        raise AssertionError("preflight attempted a mutating or execution helper")

    monkeypatch.setattr(tool, "_verify_wrapper_entrypoint", lambda: None)
    for name in (
        "_prepare_isolated_clone",
        "_prepare_isolated_cargo_cache",
        "_publish_ledger_record",
        "_spawn_supervisor_once",
        "_create_and_hold_ledger_root",
        "_run_exact",
        "_docker_inventory",
    ):
        monkeypatch.setattr(tool, name, forbidden)
    monkeypatch.setattr(
        tool, "load_phase3_q05b_shadow_nonadoption_trial_contract_v1", lambda: frozen
    )
    monkeypatch.setattr(
        tool,
        "_preflight_context",
        lambda _contract: {"capsule": capsule, "cargo_rows": []},
    )
    observed_paths: list[str] = []

    def absent(path: object) -> bool:
        observed_paths.append(os.fspath(path))
        return False

    monkeypatch.setattr(tool.os.path, "lexists", absent)
    monkeypatch.setattr(tool.subprocess, "Popen", forbidden)
    monkeypatch.setattr(tool.subprocess, "run", forbidden)
    result = tool.preflight_phase3_q05b_shadow_nonadoption_trial_v1()
    assert type(result) is dict
    assert result["execution_authorized"] is False
    assert result["supervisor_invocation_count"] == 0
    assert result["attempt2_state_touched"] is False
    assert result["meta_or_production_state_touched"] is False
    assert calls == []
    assert set(observed_paths) == {
        frozen["isolated_source_policy"]["clone_root"],
        frozen["actual_execution_policy"]["cargo_cache_source"],
        frozen["trial_ledger_policy"]["ledger_root"],
        os.fspath(
            Path(frozen["owner_authorization_policy"]["authorization_root"])
            / frozen["owner_authorization_policy"]["authorization_record_name"]
        ),
    }


def test_source_ast_has_no_attempt2_import_network_or_unbounded_public_path() -> None:
    for path in (CORE_PATH, TOOL_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=os.fspath(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert "phase3_q05b_attempt2" not in name
                assert not name.startswith(("socket", "urllib", "http", "requests"))
        text = path.read_text(encoding="utf-8")
        assert "run_actual_v1(" not in text
        assert "phase3_q05b_attempt2_meta_v11_bridge_v1" not in text


def test_document_states_mechanics_only_and_separate_worktree() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    assert "codex/hegel-q05b-shadow-nonadoption-v1" in text
    assert "attempt-2 worktree remains clean" in text
    assert "It is not evidence of v3 efficacy" in text
    assert "Q1" in text and "C1" in text and "C2" in text
