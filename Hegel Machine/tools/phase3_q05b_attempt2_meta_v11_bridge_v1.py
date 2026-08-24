#!/usr/bin/env python3
"""Exact Meta-V11 bridge and exactly-once Q0.5b attempt-2 supervisor.

``--preflight`` is read-only and never imports or calls the old actual tool.
``--run`` may call that unchanged source entrypoint once, but only after one
qualified bridge report and one immutable consumed record are durably
published in the frozen ledger.  A consumed record is terminal even when no
outcome link exists.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import fcntl
from hashlib import sha256
import importlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types
from typing import Any, Final, Iterator, Mapping, NoReturn, Sequence


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
WORKTREE_ROOT: Final = PROJECT_ROOT.parent
BRIDGE_CORE_PATH: Final = (
    PROJECT_ROOT
    / "src/hegel_machine/phase3_q05b_attempt2_meta_v11_bridge_v1.py"
)
BRIDGE_CORE_RAW_SHA256: Final = (
    "29d0b0250db17d1e3c02eb8612fdd8fe860a290f319eb89855b908cdc25f8c15"
)
DEFAULT_BRIDGE_CONTRACT: Final = (
    PROJECT_ROOT / "config/phase3_q05b_attempt2_meta_v11_bridge_v1.json"
)
OLD_ACTUAL_TOOL: Final = PROJECT_ROOT / "tools/phase3_q05b_dual_qualification_v1.py"
META_ROOT: Final = Path(
    "/home/erzhu419/.local/state/kg-op/meta-prior-theory-operations-worktree/"
    "v1/SC-OLH-KG"
)
FROZEN_PROJECT_ROOT: Final = Path(
    "/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/"
    "Hegel Machine"
)
FROZEN_LEDGER_ROOT: Final = Path(
    "/home/erzhu419/.local/state/hegel-machine/q05b-attempt2-ledger-v1"
)
FROZEN_ARTIFACT: Final = (
    FROZEN_PROJECT_ROOT / "artifacts/phase3_q05b_dual_qualification_v1.json"
)
FROZEN_CARGO_CACHE: Final = Path(
    "/home/erzhu419/.local/state/hegel-machine/rust-cargo-cache"
)
BASELINE_COMMIT: Final = "65dfc5f19450cee9510c63a85de32279160f291f"
BASELINE_TREE: Final = "f6909ff647609ddca40302d890eaede01c8fc308"
REPAIR_COMMIT: Final = "d55c017fb5ae16a83ff57758cdb3e9b57c29afb9"
ATTEMPT1_COMMIT: Final = "07d91847ceff1dc1692b05084b17718409155392"
META_COMMIT: Final = "fc9e781ce29ca0fef0f37ed97d07d5037adfdfff"
META_TREE: Final = "932ea7acbea5e8fbf36ddbd11c648611e1fcdae0"
GIT: Final = "/usr/bin/git"
HEGEL_PYCACHE_PREFIX: Final = (
    "/dev/null/hegel-q05b-attempt2-hegel-actual-source-only"
)
META_V11_CONTRACT_DIGEST: Final = (
    "sha256:9a52bb7ea3f4ce0f0cb16c5a5a296d284a85f4bfbdb2ee671e8c865bb2d3d493"
)

# The runner must establish this before importing either project core.  The
# scoped source-only contexts below additionally replace and restore every
# import hook/cache that could otherwise select ambient bytecode.
sys.dont_write_bytecode = True
sys.pycache_prefix = HEGEL_PYCACHE_PREFIX

V11_INPUT_ROWS: Final = (
    ("competition_input", "--competition-input"),
    ("competition_contract", "--competition-contract"),
    ("competition_report", "--competition-report"),
    ("transition_contract", "--transition-contract"),
    ("transition_report", "--transition-report"),
    ("qualification_input", "--qualification-input"),
    ("qualification_contract", "--qualification-contract"),
    ("qualification_report", "--qualification-report"),
    ("review_contract", "--review-contract"),
    ("review_report", "--review-report"),
    ("probe_input", "--probe-input"),
    ("probe_contract", "--probe-contract"),
    ("probe_report", "--probe-report"),
    ("restriction_input", "--restriction-input"),
    ("restriction_contract", "--restriction-contract"),
    ("restriction_report", "--restriction-report"),
    ("adjudication_input", "--adjudication-input"),
    ("adjudication_contract", "--adjudication-contract"),
    ("adjudication_report", "--adjudication-report"),
    ("adapter_input", "--adapter-input"),
    ("adapter_contract", "--adapter-contract"),
    ("adapter_report", "--adapter-report"),
    ("interval_competition_input", "--interval-competition-input"),
    ("interval_competition_contract", "--interval-competition-contract"),
    ("interval_competition_report", "--interval-competition-report"),
    ("interval_transition_contract", "--interval-transition-contract"),
    ("interval_transition_report", "--interval-transition-report"),
    (
        "post_transition_qualification_input",
        "--post-transition-qualification-input",
    ),
    (
        "post_transition_qualification_contract",
        "--post-transition-qualification-contract",
    ),
    (
        "post_transition_qualification_report",
        "--post-transition-qualification-report",
    ),
)
BRIDGE_INPUT_ROWS: Final = V11_INPUT_ROWS + (
    ("attempt2_bridge_input", "--bridge-input"),
    ("attempt2_bridge_contract", "--bridge-contract"),
)
EXPECTED_DIGEST_NAMES: Final = (
    "competition_contract", "competition_report", "transition_contract",
    "transition_report", "qualification_input", "qualification_contract",
    "qualification_report", "review_contract", "review_report",
    "probe_input", "probe_contract", "probe_report", "restriction_input",
    "restriction_contract", "restriction_report", "adjudication_input",
    "adjudication_contract", "adjudication_report", "adapter_input",
    "adapter_contract", "adapter_report", "interval_competition_input",
    "interval_competition_contract", "interval_competition_report",
    "interval_transition_contract", "interval_transition_report",
    "post_transition_qualification_input",
    "post_transition_qualification_contract",
    "post_transition_qualification_report",
)
EXACT_FIVE_PATHS: Final = (
    "Hegel Machine/src/hegel_machine/phase3_q05b_attempt2_meta_v11_bridge_v1.py",
    "Hegel Machine/config/phase3_q05b_attempt2_meta_v11_bridge_v1.json",
    "Hegel Machine/tools/phase3_q05b_attempt2_meta_v11_bridge_v1.py",
    "Hegel Machine/tests/test_phase3_q05b_attempt2_meta_v11_bridge_v1.py",
    "Hegel Machine/docs/Hegel_Machine_Phase3A_Q05b_Attempt2_Meta_V11_Bridge_v1.md",
)
META_RAW_SHA256: Final = {
    "performance/shadow_interval_multi_q_post_transition_qualification.py":
        "d9ef2a166abb78f04cf41a5a4f4ae48a7fa3227817cd42dbc11272dc68bdeb36",
    "performance/manifests/shadow_interval_multi_q_post_transition_qualification_v1.json":
        "34c63a1ee3825543e3db612c9621af9af6441c6f58685423307164f05d4ce554",
    "runners/run_shadow_interval_multi_q_post_transition_qualification.py":
        "29b14485d2fa7d268af0723ff7b36bd6bd90f1ecab53e0c1bc6f3624181416eb",
    "tests/test_shadow_interval_multi_q_post_transition_qualification.py":
        "2046e500a37d06bab00bbaa351a6ce942dc109faf9ab9d7165f83097a8a9f45d",
    "docs/shadow_interval_multi_q_post_transition_qualification_v1.md":
        "660488c35fc443759d2c3387030d132ee43ec71adfb3e2906fc6f1becc968d6c",
}
UNCHANGED_HEGEL_SHA256: Final = {
    "config/phase3_q05b_dual_isolation_v1.json":
        "d3a1189cce5a40c5be76f0c5ff1c4e375781a803c11258361fc131a70a7c7242",
    "config/phase3_q05b_node3_dual_projection_qualification_v1.json":
        "29fb2c0d8a6bd657212d1799c4e7a751bf42d3937ff865223e593e46f96d14ab",
    "docs/Hegel_Machine_Phase3A_Q05b_Actual_Attempt_1_Failure_Errata_v1.md":
        "03e299ae6d0e6354c9afc259137a290009ff1b52765674ba310387204bcd0894",
    "src/hegel_machine/phase3_q05b_actual_admission_v1.py":
        "d1de0335dee62dbc737e5bd64410de392632648aa92aeabdb4fa078551e9b216",
    "src/hegel_machine/phase3_q05b_actual_artifact_v1.py":
        "75dc2a784280cd59e5bde201b9d9aef78130f6713be59c833864297ab9b36837",
    "tools/phase3_q05b_dual_qualification_v1.py":
        "af406bba2e4d11f8f59c1d7f91d3f84c887d1e90d10cc1749baf2a8610a7c5d2",
}
LEDGER_NAMES: Final = {
    "bridge": "attempt-2.bridge-report.json",
    "consumed": "attempt-2.consumed.json",
    "execution": "attempt-2.execution-link.json",
    "failure": "attempt-2.failure-link.json",
}


class RunnerError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class MissingV11(RunnerError):
    def __init__(self) -> None:
        super().__init__(
            "MISSING_REAL_V11_REPORT",
            "required real post-transition qualification report is absent",
        )


class _StrictParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise RunnerError("INVALID_ATTEMPT2_META_V11_BRIDGE", message)


def _pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in rows:
        if key in result:
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE", f"duplicate JSON key: {key}"
            )
        result[key] = value
    return result


def _reject_constant(token: str) -> NoReturn:
    raise RunnerError(
        "INVALID_ATTEMPT2_META_V11_BRIDGE",
        f"non-finite JSON number: {token}",
    )


def _canonical(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", f"canonical JSON differs: {error}"
        ) from error


def _strict_object(raw: bytes, label: str, *, missing: bool = False) -> dict[str, Any]:
    if missing and not raw:
        raise MissingV11()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        if missing and not raw.strip():
            raise MissingV11() from error
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE",
            f"{label} is not strict JSON: {error}",
        ) from error
    if type(value) is not dict:
        if missing:
            raise MissingV11()
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", f"{label} JSON must be an object"
        )
    return value


def _normalized_absolute(path: Path, label: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", f"{label} must be absolute"
        )
    normalized = Path(os.path.abspath(os.fspath(path)))
    if normalized != path:
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", f"{label} must be normalized"
        )
    return normalized


def _read_file(path: Path, label: str, *, missing: bool = False) -> tuple[bytes, os.stat_result]:
    try:
        path = _normalized_absolute(path, label)
        if os.path.realpath(path) != os.fspath(path):
            if missing:
                raise MissingV11()
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE", f"{label} traverses a symlink"
            )
        status = path.lstat()
        if not stat.S_ISREG(status.st_mode):
            if missing:
                raise MissingV11()
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE",
                f"{label} must be an ordinary file",
            )
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
    except MissingV11:
        raise
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        if missing:
            raise MissingV11() from error
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", f"{label} is unavailable: {error}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (status.st_dev, status.st_ino)
        ):
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE", f"{label} identity drifted"
            )
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > 128 * 1024 * 1024:
                raise RunnerError(
                    "INVALID_ATTEMPT2_META_V11_BRIDGE", f"{label} is too large"
                )
            chunks.append(chunk)
        return b"".join(chunks), opened
    finally:
        os.close(descriptor)


def _load_json(
    path: Path,
    label: str,
    *,
    missing: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], tuple[int, int], bytes]:
    raw, status = _read_file(path, label, missing=missing)
    value = _strict_object(raw, label, missing=missing)
    artifact = {
        "path": os.fspath(path),
        "bytes": len(raw),
        "sha256": sha256(raw).hexdigest(),
    }
    return value, artifact, (status.st_dev, status.st_ino), raw


def _sha_file(path: Path, label: str) -> str:
    raw, _status = _read_file(path, label)
    return sha256(raw).hexdigest()


def _source_file_evidence(
    root: Path,
    expected: Mapping[str, str],
    label: str,
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    identities: set[tuple[int, int]] = set()
    for relative, expected_sha in expected.items():
        relative_path = Path(relative)
        if (
            type(relative) is not str
            or not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
        ):
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE",
                f"{label} relative path/hash declaration differs",
            )
        path = root / relative
        raw, status = _read_file(path, f"{label} {relative}")
        identity = (status.st_dev, status.st_ino)
        if (
            status.st_nlink != 1
            or not stat.S_ISREG(status.st_mode)
            or identity in identities
            or sha256(raw).hexdigest() != expected_sha
        ):
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE",
                f"{label} ordinary source identity/hash differs: {relative}",
            )
        identities.add(identity)
        evidence[relative] = {
            "path": os.fspath(path),
            "device": status.st_dev,
            "inode": status.st_ino,
            "nlink": status.st_nlink,
            "mode": status.st_mode,
            "size": status.st_size,
            "sha256": expected_sha,
        }
    return evidence


def _ordinary_directory(path: Path, label: str, *, mode: int | None = None) -> None:
    path = _normalized_absolute(path, label)
    if os.path.realpath(path) != os.fspath(path):
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", f"{label} traverses a symlink"
        )
    try:
        status = path.lstat()
    except OSError as error:
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", f"{label} is unavailable: {error}"
        ) from error
    if not stat.S_ISDIR(status.st_mode):
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", f"{label} is not a directory"
        )
    if mode is not None and stat.S_IMODE(status.st_mode) != mode:
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE",
            f"{label} mode must be {mode:04o}",
        )


def _verify_fixed_self_location() -> None:
    runner = Path(__file__).absolute()
    expected_runner = (
        FROZEN_PROJECT_ROOT
        / "tools/phase3_q05b_attempt2_meta_v11_bridge_v1.py"
    )
    expected = {
        "project": FROZEN_PROJECT_ROOT,
        "runner": expected_runner,
        "core": FROZEN_PROJECT_ROOT
        / "src/hegel_machine/phase3_q05b_attempt2_meta_v11_bridge_v1.py",
        "contract": FROZEN_PROJECT_ROOT
        / "config/phase3_q05b_attempt2_meta_v11_bridge_v1.json",
        "old_actual": FROZEN_PROJECT_ROOT
        / "tools/phase3_q05b_dual_qualification_v1.py",
    }
    if (
        PROJECT_ROOT != expected["project"]
        or runner != expected["runner"]
        or runner.resolve(strict=True) != expected["runner"]
        or runner.is_symlink()
        or BRIDGE_CORE_PATH != expected["core"]
        or DEFAULT_BRIDGE_CONTRACT != expected["contract"]
        or OLD_ACTUAL_TOOL != expected["old_actual"]
    ):
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE",
            "runner/project/core/contract/actual fixed self-location differs",
        )
    status = runner.lstat()
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE",
            "runner must be an ordinary single-link source file",
        )
    _ordinary_directory(PROJECT_ROOT, "frozen Hegel project root")


def _git(root: Path, *arguments: str, check: bool = True) -> str:
    git_arguments = list(arguments)
    if git_arguments and git_arguments[0] == "diff":
        git_arguments[1:1] = ["--no-ext-diff", "--no-textconv"]
    completed = subprocess.run(
        [
            GIT,
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "core.quotePath=false",
            "-C",
            os.fspath(root),
            *git_arguments,
        ],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_PROTOCOL_FROM_USER": "0",
            "GIT_NO_REPLACE_OBJECTS": "1",
        },
    )
    if check and completed.returncode != 0:
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE",
            "Git replay failed: " + completed.stderr.decode("utf-8", "replace").strip(),
        )
    if not check:
        return str(completed.returncode)
    try:
        return completed.stdout.decode("ascii").rstrip("\n")
    except UnicodeDecodeError as error:
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", "Git output is not ASCII"
        ) from error


def _verify_meta_source(binding: Mapping[str, Any]) -> dict[str, Any]:
    _ordinary_directory(META_ROOT, "--meta-project-root")
    if binding.get("repo_root") != os.fspath(META_ROOT):
        raise RunnerError("INVALID_ATTEMPT2_META_V11_BRIDGE", "Meta root binding differs")
    if (
        _git(META_ROOT, "rev-parse", "HEAD") != META_COMMIT
        or _git(META_ROOT, "rev-parse", "HEAD^{tree}") != META_TREE
        or _git(META_ROOT, "status", "--porcelain=v1", "--untracked-files=all")
    ):
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", "Meta commit/tree/clean binding differs"
        )
    evidence = _source_file_evidence(META_ROOT, META_RAW_SHA256, "Meta source")
    if binding.get("exact_five_raw_sha256") != META_RAW_SHA256:
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", "Meta raw hash declaration differs"
        )
    return {
        "repo_root": os.fspath(META_ROOT),
        "git_commit": META_COMMIT,
        "git_tree": META_TREE,
        "git_clean": True,
        "v11_contract_digest": META_V11_CONTRACT_DIGEST,
        "exact_five_file_evidence": evidence,
        "commit_tree_clean_checks_performed": True,
        "exact_five_file_checks_performed": True,
        "v11_contract_digest_check_performed": True,
    }


def _verify_hegel_source(
    binding: Mapping[str, Any], source_commit: str
) -> dict[str, Any]:
    _ordinary_directory(FROZEN_PROJECT_ROOT, "--hegel-project-root")
    if type(source_commit) is not str or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", "--source-commit must be full lowercase hex"
        )
    head = _git(FROZEN_PROJECT_ROOT, "rev-parse", "HEAD")
    tree = _git(FROZEN_PROJECT_ROOT, "rev-parse", "HEAD^{tree}")
    if head != source_commit or _git(
        FROZEN_PROJECT_ROOT, "status", "--porcelain=v1", "--untracked-files=all"
    ):
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", "Hegel requested HEAD must be clean"
        )
    parents = _git(FROZEN_PROJECT_ROOT, "rev-list", "--parents", "-n", "1", head).split()
    if parents != [head, BASELINE_COMMIT]:
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", "Hegel source is not the direct child baseline"
        )
    if _git(FROZEN_PROJECT_ROOT, "rev-list", "--count", f"{BASELINE_COMMIT}..{head}") != "1":
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", "Hegel source commit count differs"
        )
    if _git(FROZEN_PROJECT_ROOT, "merge-base", "--is-ancestor", REPAIR_COMMIT, head, check=False) != "0":
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", "repair commit is not an ancestor"
        )
    if _git(FROZEN_PROJECT_ROOT, "merge-base", "--is-ancestor", ATTEMPT1_COMMIT, head, check=False) != "0":
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", "attempt-1 source is not an ancestor"
        )
    diff_rows = tuple(
        row for row in _git(
            FROZEN_PROJECT_ROOT,
            "diff",
            "--name-status",
            "--no-renames",
            BASELINE_COMMIT,
            head,
        ).splitlines() if row
    )
    parsed_diff = [row.split("\t", 1) for row in diff_rows]
    if (
        len(parsed_diff) != len(EXACT_FIVE_PATHS)
        or any(len(row) != 2 or row[0] != "A" for row in parsed_diff)
        or {row[1] for row in parsed_diff} != set(EXACT_FIVE_PATHS)
    ):
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", "Hegel change is not exact-five additive"
        )
    declared_exact_hashes = binding.get("exact_five_raw_sha256")
    if (
        type(declared_exact_hashes) is not dict
        or set(declared_exact_hashes) != set(EXACT_FIVE_PATHS)
        or any(
            type(value) is not str
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in declared_exact_hashes.values()
        )
    ):
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE",
            "declared exact-five hash map is unavailable",
        )
    exact_file_evidence = _source_file_evidence(
        WORKTREE_ROOT,
        declared_exact_hashes,
        "exact-five Hegel source",
    )
    exact_hashes = {
        relative: row["sha256"]
        for relative, row in exact_file_evidence.items()
    }
    unchanged = {
        relative: _sha_file(PROJECT_ROOT / relative, f"unchanged Hegel {relative}")
        for relative in UNCHANGED_HEGEL_SHA256
    }
    if unchanged != UNCHANGED_HEGEL_SHA256:
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", "unchanged Hegel source hash differs"
        )
    expected = {
        "project_root": os.fspath(FROZEN_PROJECT_ROOT),
        "source_commit": head,
        "source_tree": tree,
        "clean": True,
        "baseline_commit": BASELINE_COMMIT,
        "baseline_tree": BASELINE_TREE,
        "repair_is_ancestor": True,
        "exact_five_diff_paths": list(EXACT_FIVE_PATHS),
        "exact_five_raw_sha256": exact_hashes,
        "existing_actual_tool_raw_sha256": UNCHANGED_HEGEL_SHA256[
            "tools/phase3_q05b_dual_qualification_v1.py"
        ],
        "existing_admission_core_raw_sha256": UNCHANGED_HEGEL_SHA256[
            "src/hegel_machine/phase3_q05b_actual_admission_v1.py"
        ],
        "existing_artifact_core_raw_sha256": UNCHANGED_HEGEL_SHA256[
            "src/hegel_machine/phase3_q05b_actual_artifact_v1.py"
        ],
        "isolation_config_raw_sha256": UNCHANGED_HEGEL_SHA256[
            "config/phase3_q05b_dual_isolation_v1.json"
        ],
        "qualification_config_raw_sha256": UNCHANGED_HEGEL_SHA256[
            "config/phase3_q05b_node3_dual_projection_qualification_v1.json"
        ],
        "existing_unchanged_raw_sha256": unchanged,
    }
    if dict(binding) != expected:
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", "declared Hegel source binding differs live"
        )
    errata_blob = _git(
        FROZEN_PROJECT_ROOT,
        "rev-parse",
        REPAIR_COMMIT
        + ":Hegel Machine/docs/"
        "Hegel_Machine_Phase3A_Q05b_Actual_Attempt_1_Failure_Errata_v1.md",
    )
    if errata_blob != "6ba3adb22c77af28e64bc56bf7a63180d75d1609":
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", "attempt-1 errata blob differs"
        )
    return {
        "project_root": os.fspath(FROZEN_PROJECT_ROOT),
        "source_commit": head,
        "source_tree": tree,
        "clean": True,
        "baseline_commit": BASELINE_COMMIT,
        "baseline_tree": BASELINE_TREE,
        "source_parent_commit": BASELINE_COMMIT,
        "source_commit_count_after_baseline": 1,
        "source_direct_child_of_baseline": True,
        "attempt1_source_commit": ATTEMPT1_COMMIT,
        "repair_commit": REPAIR_COMMIT,
        "repair_is_ancestor": True,
        "exact_five_diff_paths": list(EXACT_FIVE_PATHS),
        "exact_five_file_evidence": exact_file_evidence,
        "commit_tree_clean_checks_performed": True,
        "direct_child_and_commit_count_checks_performed": True,
        "attempt1_repair_lineage_checks_performed": True,
        "exact_five_diff_check_performed": True,
        "exact_five_file_checks_performed": True,
        "no_deletions_renames_or_other_edits": True,
    }


def _verify_ledger_argument(args: argparse.Namespace) -> None:
    supplied = _normalized_absolute(args.ledger_root, "--ledger-root")
    if supplied != FROZEN_LEDGER_ROOT:
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE",
            "--ledger-root differs from frozen path",
        )


def _verify_runtime_arguments(args: argparse.Namespace) -> None:
    exact = {
        "meta_project_root": META_ROOT,
        "hegel_project_root": FROZEN_PROJECT_ROOT,
        "artifact": FROZEN_ARTIFACT,
        "cargo_cache_source": FROZEN_CARGO_CACHE,
    }
    for field, expected in exact.items():
        supplied = _normalized_absolute(getattr(args, field), f"--{field.replace('_', '-')}")
        if supplied != expected:
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE",
                f"--{field.replace('_', '-')} differs from frozen path",
            )
    _ordinary_directory(FROZEN_CARGO_CACHE, "--cargo-cache-source")


def _load_all_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], tuple[Path, ...]]:
    loaded: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    paths: list[Path] = []
    normalized: set[Path] = set()
    inodes: set[tuple[int, int]] = set()
    for name, flag in BRIDGE_INPUT_ROWS:
        path = _normalized_absolute(getattr(args, name), flag)
        value, artifact, inode, _raw = _load_json(
            path,
            flag,
            missing=name == "post_transition_qualification_report",
        )
        if path in normalized:
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE",
                "all thirty-two JSON input paths must be distinct",
            )
        if inode in inodes:
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE",
                "all thirty-two JSON input inodes must be distinct",
            )
        normalized.add(path)
        inodes.add(inode)
        paths.append(path)
        loaded[name] = value
        artifacts[name] = artifact
    output_paths = (FROZEN_ARTIFACT,) + tuple(
        FROZEN_LEDGER_ROOT / name for name in LEDGER_NAMES.values()
    )
    for source in paths:
        for output in output_paths:
            if source == output:
                raise RunnerError(
                    "INVALID_ATTEMPT2_META_V11_BRIDGE", "JSON input aliases a fixed output"
                )
            if output.exists() and source.exists() and os.path.samefile(source, output):
                raise RunnerError(
                    "INVALID_ATTEMPT2_META_V11_BRIDGE",
                    "JSON input hard-links a fixed output",
                )
    return loaded, artifacts, tuple(paths)


@contextmanager
def _standard_source_imports(prefix: str, source_root: Path) -> Iterator[None]:
    previous_dont = sys.dont_write_bytecode
    previous_prefix = sys.pycache_prefix
    previous_path = list(sys.path)
    previous_meta = list(sys.meta_path)
    previous_hooks = list(sys.path_hooks)
    previous_cache = sys.path_importer_cache
    try:
        sys.dont_write_bytecode = True
        sys.pycache_prefix = prefix
        sys.path[:] = [os.fspath(source_root)] + [
            item for item in previous_path if item != os.fspath(source_root)
        ]
        sys.meta_path[:] = [
            importlib.machinery.BuiltinImporter,
            importlib.machinery.FrozenImporter,
            importlib.machinery.PathFinder,
        ]
        sys.path_hooks[:] = [
            importlib.machinery.FileFinder.path_hook(
                (
                    importlib.machinery.SourceFileLoader,
                    importlib.machinery.SOURCE_SUFFIXES,
                ),
                (
                    importlib.machinery.ExtensionFileLoader,
                    importlib.machinery.EXTENSION_SUFFIXES,
                ),
            )
        ]
        sys.path_importer_cache = {}
        importlib.invalidate_caches()
        yield
    finally:
        sys.dont_write_bytecode = previous_dont
        sys.pycache_prefix = previous_prefix
        sys.path[:] = previous_path
        sys.meta_path[:] = previous_meta
        sys.path_hooks[:] = previous_hooks
        sys.path_importer_cache = previous_cache
        importlib.invalidate_caches()


def _load_bridge_core():
    existing = {
        name for name in sys.modules
        if name == "hegel_machine" or name.startswith("hegel_machine.")
    }
    if existing:
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE",
            "ambient Hegel modules forbid bridge source-only import",
        )
    raw, status = _read_file(BRIDGE_CORE_PATH, "bridge core source")
    if (
        status.st_nlink != 1
        or sha256(raw).hexdigest() != BRIDGE_CORE_RAW_SHA256
    ):
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE",
            "bridge core source raw identity/hash differs",
        )
    module_name = "hegel_machine.phase3_q05b_attempt2_meta_v11_bridge_v1"
    package_name = "hegel_machine"
    package_root = PROJECT_ROOT / "src" / package_name
    with _standard_source_imports(
        HEGEL_PYCACHE_PREFIX, PROJECT_ROOT / "src"
    ):
        package = types.ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [os.fspath(package_root)]
        package_spec = importlib.machinery.ModuleSpec(
            package_name, loader=None, is_package=True
        )
        package_spec.submodule_search_locations = [os.fspath(package_root)]
        package.__spec__ = package_spec
        sys.modules[package_name] = package
        loader = importlib.machinery.SourceFileLoader(
            module_name, os.fspath(BRIDGE_CORE_PATH)
        )
        spec = importlib.util.spec_from_loader(
            module_name, loader, origin=os.fspath(BRIDGE_CORE_PATH)
        )
        if spec is None:
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE",
                "bridge core source spec is unavailable",
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            code = compile(
                raw,
                os.fspath(BRIDGE_CORE_PATH),
                "exec",
                dont_inherit=True,
            )
            exec(code, module.__dict__)
            _verify_loaded_bridge_core(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            sys.modules.pop(package_name, None)
            raise
    return module


def _verify_loaded_bridge_core(core: object) -> None:
    module_name = "hegel_machine.phase3_q05b_attempt2_meta_v11_bridge_v1"
    package = sys.modules.get("hegel_machine")
    module = sys.modules.get(module_name)
    if (
        module is not core
        or package is None
        or list(getattr(package, "__path__", ()))
        != [os.fspath(PROJECT_ROOT / "src" / "hegel_machine")]
        or Path(getattr(module, "__file__", "")).resolve() != BRIDGE_CORE_PATH
        or type(getattr(module, "__loader__", None))
        is not importlib.machinery.SourceFileLoader
        or _sha_file(BRIDGE_CORE_PATH, "loaded bridge core source")
        != BRIDGE_CORE_RAW_SHA256
    ):
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE",
            "loaded bridge core source identity differs",
        )


def _error_wire(
    code: str,
    detail: str,
    *,
    bridge_report_created: bool = False,
    ledger_created: bool = False,
    ledger_consumed: bool = False,
    actual_called: bool = False,
    docker_or_cargo_called: bool = False,
    artifact_written: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": "hegel-phase3a-q05b-attempt2-meta-v11-bridge-error/1",
        "status": code,
        "detail": detail[:4096],
        "bridge_report_created": bridge_report_created,
        "ledger_created": ledger_created,
        "ledger_consumed": ledger_consumed,
        "actual_called": actual_called,
        "docker_or_cargo_called": docker_or_cargo_called,
        "q05b_artifact_written": artifact_written,
    }


def _missing_wire(core: object | None = None) -> dict[str, Any]:
    if core is not None and hasattr(core, "build_missing_real_v11_report_error_v1"):
        return core.build_missing_real_v11_report_error_v1()
    return _error_wire(
        "MISSING_REAL_V11_REPORT",
        "required real post-transition qualification report is absent",
    )


def _write_stream(stream: object, payload: bytes) -> None:
    binary = getattr(stream, "buffer", stream)
    binary.write(payload)
    binary.flush()


def _ledger_read(fd: int, name: str) -> tuple[dict[str, Any], bytes, os.stat_result]:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=fd,
    )
    try:
        status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(status.st_mode)
            or stat.S_IMODE(status.st_mode) != 0o444
            or status.st_nlink != 1
            or status.st_uid != os.geteuid()
        ):
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE", f"ledger record {name} is not immutable"
            )
        raw = b""
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            raw += chunk
            if len(raw) > 128 * 1024 * 1024:
                raise RunnerError(
                    "INVALID_ATTEMPT2_META_V11_BRIDGE", f"ledger record {name} is too large"
                )
    finally:
        os.close(descriptor)
    value = _strict_object(raw, f"ledger record {name}")
    if _canonical(value) != raw:
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", f"ledger record {name} is not canonical"
        )
    return value, raw, status


def _open_ledger(*, create: bool) -> int | None:
    root = FROZEN_LEDGER_ROOT
    _ordinary_directory(root.parent, "ledger parent")
    parent = os.open(
        root.parent,
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    descriptor: int | None = None
    created = False
    try:
        if not os.path.lexists(root):
            if not create:
                return None
            try:
                os.mkdir(root.name, 0o700, dir_fd=parent)
                created = True
            except FileExistsError:
                pass
        descriptor = os.open(
            root.name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent,
        )
        status = os.fstat(descriptor)
        anchored = os.stat(root.name, dir_fd=parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(status.st_mode)
            or (status.st_dev, status.st_ino) != (anchored.st_dev, anchored.st_ino)
            or status.st_uid != os.geteuid()
        ):
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE",
                "ledger root identity/owner differs",
            )
        if created:
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
            os.fsync(parent)
            status = os.fstat(descriptor)
        if stat.S_IMODE(status.st_mode) != 0o700:
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE",
                "ledger root mode must be 0700",
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        result = descriptor
        descriptor = None
        return result
    except FileNotFoundError:
        if not create:
            return None
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _ledger_snapshot(fd: int | None) -> tuple[str, dict[str, Any]]:
    if fd is None:
        return "OPEN_ALL_FOUR_RECORDS_ABSENT", {}
    entries = set(os.listdir(fd))
    allowed = set(LEDGER_NAMES.values())
    if not entries.issubset(allowed):
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", "ledger contains foreign entries"
        )
    values: dict[str, Any] = {}
    for kind, name in LEDGER_NAMES.items():
        if name in entries:
            values[kind] = _ledger_read(fd, name)
    present = set(values)
    if not present:
        return "OPEN_ALL_FOUR_RECORDS_ABSENT", values
    if present == {"bridge"}:
        return "BRIDGE_STAGED_NOT_CONSUMED", values
    if "bridge" not in present or "consumed" not in present:
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", "ledger record combination is invalid"
        )
    outcomes = present & {"execution", "failure"}
    if len(outcomes) > 1:
        raise RunnerError(
            "INVALID_ATTEMPT2_META_V11_BRIDGE", "success and failure links coexist"
        )
    if not outcomes:
        return "CONSUMED_OUTCOME_UNKNOWN", values
    return (
        "CONSUMED_QUALIFIED_LINKED" if "execution" in outcomes
        else "CONSUMED_FAILED_CLOSED",
        values,
    )


def _publish(fd: int, name: str, payload: bytes) -> os.stat_result:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(name, flags, 0o400, dir_fd=fd)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError(errno.EIO, "short ledger write")
            offset += written
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        status = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(fd)
    return status


def _consumed_evidence(status: os.stat_result, payload: bytes) -> dict[str, Any]:
    return {
        "path": os.fspath(FROZEN_LEDGER_ROOT / LEDGER_NAMES["consumed"]),
        "device": status.st_dev,
        "inode": status.st_ino,
        "nlink": status.st_nlink,
        "mode": stat.S_IMODE(status.st_mode),
        "size": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def _validate_consumed_state(core: object, values: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    bridge = values["bridge"][0]
    consumed = values["consumed"][0]
    core.validate_phase3_q05b_attempt2_meta_v11_bridge_report_v1(bridge, contract)
    core.validate_attempt2_consumed_record_v1(consumed, bridge, contract)
    if "execution" in values:
        core.validate_attempt2_execution_link_record_v1(
            values["execution"][0], bridge, consumed, contract
        )
    if "failure" in values:
        core.validate_attempt2_failure_link_record_v1(
            values["failure"][0], bridge, consumed, contract
        )


class _ActualSourceSession:
    def __init__(self, core: object) -> None:
        self.core = core
        self.preloaded: list[str] = []
        self.module_name = "_hegel_q05b_attempt2_old_actual_source_v1"
        self.run_actual = None
        self.origins: dict[str, str] = {}
        self.unchanged_source_file_evidence: dict[str, dict[str, Any]] = {}
        self._context = None

    def __enter__(self):
        _verify_loaded_bridge_core(self.core)
        allowed = {
            "hegel_machine",
            "hegel_machine.phase3_q05b_attempt2_meta_v11_bridge_v1",
        }
        self.preloaded = sorted(
            name for name in sys.modules
            if name == "hegel_machine" or name.startswith("hegel_machine.")
        )
        if not set(self.preloaded).issubset(allowed):
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE",
                "preloaded Hegel modules forbid source-only actual",
            )
        self._context = _standard_source_imports(
            HEGEL_PYCACHE_PREFIX, PROJECT_ROOT / "src"
        )
        self._context.__enter__()
        try:
            raw, _status = _read_file(OLD_ACTUAL_TOOL, "old actual tool")
            if sha256(raw).hexdigest() != UNCHANGED_HEGEL_SHA256[
                "tools/phase3_q05b_dual_qualification_v1.py"
            ]:
                raise RunnerError(
                    "INVALID_ATTEMPT2_META_V11_BRIDGE", "old actual tool raw hash differs"
                )
            loader = importlib.machinery.SourceFileLoader(
                self.module_name, os.fspath(OLD_ACTUAL_TOOL)
            )
            spec = importlib.util.spec_from_loader(
                self.module_name, loader, origin=os.fspath(OLD_ACTUAL_TOOL)
            )
            if spec is None:
                raise RunnerError(
                    "INVALID_ATTEMPT2_META_V11_BRIDGE", "old actual source spec is unavailable"
                )
            module = importlib.util.module_from_spec(spec)
            sys.modules[self.module_name] = module
            code = compile(raw, os.fspath(OLD_ACTUAL_TOOL), "exec", dont_inherit=True)
            exec(code, module.__dict__)
            callable_value = getattr(module, "run_actual_v1", None)
            if (
                not callable(callable_value)
                or getattr(callable_value, "__module__", None) != self.module_name
            ):
                raise RunnerError(
                    "INVALID_ATTEMPT2_META_V11_BRIDGE", "old run_actual_v1 identity differs"
                )
            self.run_actual = callable_value
            self.unchanged_source_file_evidence = _source_file_evidence(
                PROJECT_ROOT, UNCHANGED_HEGEL_SHA256, "unchanged Hegel source"
            )
            self._collect_origins()
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def _collect_origins(self) -> None:
        root = PROJECT_ROOT / "src" / "hegel_machine"
        result: dict[str, str] = {}
        for name, module in sorted(sys.modules.items()):
            if not name.startswith("hegel_machine."):
                continue
            origin_text = getattr(module, "__file__", None)
            loader = getattr(module, "__loader__", None)
            if type(origin_text) is not str or type(loader) is not importlib.machinery.SourceFileLoader:
                raise RunnerError(
                    "INVALID_ATTEMPT2_META_V11_BRIDGE", f"Hegel module {name} is not source-loaded"
                )
            origin = Path(origin_text).resolve(strict=True)
            expected = PROJECT_ROOT / "src" / (name.replace(".", "/") + ".py")
            if origin != expected or origin.is_symlink() or not origin.is_file():
                raise RunnerError(
                    "INVALID_ATTEMPT2_META_V11_BRIDGE", f"Hegel module origin differs: {name}"
                )
            try:
                origin.relative_to(root)
            except ValueError as error:
                raise RunnerError(
                    "INVALID_ATTEMPT2_META_V11_BRIDGE", f"Hegel module escapes source root: {name}"
                ) from error
            result[name] = os.fspath(origin)
        if not result:
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE", "no Hegel source module origins were recorded"
            )
        self.origins = result

    def call(self, source_commit: str) -> Mapping[str, Any]:
        if self.run_actual is None:
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE", "actual session is not prepared"
            )
        value = self.run_actual(
            FROZEN_PROJECT_ROOT,
            source_commit,
            FROZEN_ARTIFACT,
            FROZEN_CARGO_CACHE,
        )
        self._collect_origins()
        if not isinstance(value, Mapping):
            raise RunnerError(
                "ATTEMPT2_FAILED_CLOSED", "run_actual_v1 returned a non-object"
            )
        return value

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if self._context is not None:
                self._context.__exit__(exc_type, exc, traceback)
        finally:
            sys.modules.pop(self.module_name, None)
            for name in tuple(sys.modules):
                if (
                    (name == "hegel_machine" or name.startswith("hegel_machine."))
                    and name not in self.preloaded
                ):
                    sys.modules.pop(name, None)

    def receipt(self) -> dict[str, Any]:
        return {
            "policy": "SOURCE_ONLY_HEGEL_ACTUAL_IMPORT_AND_CALL",
            "preloaded_hegel_machine_modules": self.preloaded,
            "preloaded_modules_within_allowlist": True,
            "dont_write_bytecode_active": True,
            "pycache_prefix": HEGEL_PYCACHE_PREFIX,
            "standard_import_machinery_only": True,
            "source_file_loader_only_for_project": True,
            "actual_tool_source_origin": os.fspath(OLD_ACTUAL_TOOL),
            "actual_tool_raw_sha256": UNCHANGED_HEGEL_SHA256[
                "tools/phase3_q05b_dual_qualification_v1.py"
            ],
            "unchanged_source_file_evidence": self.unchanged_source_file_evidence,
            "run_actual_callable_identity_verified": True,
            "loaded_hegel_source_module_origins": self.origins,
            "in_tree_bytecode_loaded": False,
            "path_hooks_and_importer_cache_restored": True,
            "interpreter_import_state_restored": True,
            "live_source_commit_tree_clean_raw_hash_checks_performed": True,
            "unchanged_source_path_hash_checks_performed": True,
        }


def _parser() -> argparse.ArgumentParser:
    parser = _StrictParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    for name, flag in V11_INPUT_ROWS:
        parser.add_argument(flag, dest=name, type=Path, required=True)
    parser.add_argument("--bridge-input", dest="attempt2_bridge_input", type=Path, required=True)
    parser.add_argument(
        "--bridge-contract",
        dest="attempt2_bridge_contract",
        type=Path,
        default=DEFAULT_BRIDGE_CONTRACT,
    )
    parser.add_argument("--meta-project-root", type=Path, required=True)
    parser.add_argument("--hegel-project-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--cargo-cache-source", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    for name in EXPECTED_DIGEST_NAMES:
        parser.add_argument(
            f"--expected-{name.replace('_', '-')}-digest",
            dest=f"expected_{name}_digest",
            required=True,
        )
    return parser


def _close_ledger(fd: int | None) -> None:
    if fd is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: object | None = None,
    stderr: object | None = None,
) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    core = None
    ledger_fd: int | None = None
    bridge_report_created = False
    ledger_created = False
    ledger_consumed = False
    actual_called = False
    docker_or_cargo_called = False
    artifact_written = False
    try:
        args = _parser().parse_args(argv)
        # This gate deliberately precedes ledger inspection and every import.
        _load_json(
            args.post_transition_qualification_report,
            "--post-transition-qualification-report",
            missing=True,
        )
        _verify_fixed_self_location()
        _verify_ledger_argument(args)
        core = _load_bridge_core()

        # Spent authorization is terminal before replacement input replay.
        ledger_fd = _open_ledger(create=False)
        ledger_created = ledger_fd is not None
        state, ledger_values = _ledger_snapshot(ledger_fd)
        if state == "BRIDGE_STAGED_NOT_CONSUMED":
            bridge_report_created = True
        if state.startswith("CONSUMED_"):
            ledger_consumed = True
            bridge_report_created = "bridge" in ledger_values
            frozen_contract = _load_json(
                DEFAULT_BRIDGE_CONTRACT, "frozen bridge contract"
            )[0]
            _validate_consumed_state(core, ledger_values, frozen_contract)
            bridge_report_created = True
            raise RunnerError(
                core.ALREADY_CONSUMED,
                "attempt-2 authorization was already consumed; retry is forbidden",
            )
        _close_ledger(ledger_fd)
        ledger_fd = None

        _verify_runtime_arguments(args)
        loaded, artifacts, _paths = _load_all_inputs(args)
        supplied_digests = {
            f"expected_{name}_digest": getattr(args, f"expected_{name}_digest")
            for name in EXPECTED_DIGEST_NAMES
        }
        bridge_input = loaded["attempt2_bridge_input"]
        if bridge_input.get("meta_v11_expected_digests") != supplied_digests:
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE",
                "29 independent CLI digest anchors differ from bridge input",
            )
        meta_live_source_receipt = _verify_meta_source(
            bridge_input["meta_source_binding"]
        )
        core.validate_meta_live_source_validation_receipt_v1(
            meta_live_source_receipt
        )
        hegel_live_source_receipt = _verify_hegel_source(
            bridge_input["hegel_source_binding"], args.source_commit
        )
        core.validate_hegel_live_source_validation_receipt_v1(
            hegel_live_source_receipt,
            bridge_input["hegel_source_binding"],
        )
        if os.path.lexists(FROZEN_ARTIFACT):
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE",
                "frozen Q0.5b artifact target must be absent before attempt-2",
            )
        result = core.build_phase3_q05b_attempt2_meta_v11_bridge_v1(
            *(loaded[name] for name, _flag in BRIDGE_INPUT_ROWS),
            input_artifacts=artifacts,
        )
        report = result.to_dict()
        verified = core.verify_phase3_q05b_attempt2_meta_v11_bridge_v1(
            *(loaded[name] for name, _flag in BRIDGE_INPUT_ROWS),
            report,
            expected_attempt2_bridge_report_digest=report["report_digest"],
            expected_input_artifacts=artifacts,
        )
        expected_verifier_receipt = {
            "status": "VERIFIED_" + report["disposition"],
            "disposition": report["disposition"],
            "report_digest": report["report_digest"],
            "contract_digest": report["contract_digest"],
            "bridge_input_digest": report["bridge_input_digest"],
            "meta_v11_report_digest": report["meta_v11_replay_receipt"][
                "report_digest"
            ],
            "destination_adoption_receipt_digest": (
                None
                if report["destination_adoption_receipt"] is None
                else report["destination_adoption_receipt"]["receipt_digest"]
            ),
            "admitted": report["disposition"] == core.BRIDGE_ADMITTED,
        }
        if verified != expected_verifier_receipt:
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE",
                "public 33-object bridge replay differs from fresh builder",
            )
        bridge_report_created = True
        report_payload = core.canonical_json_bytes_v1(report)
        if args.preflight:
            if (
                state == "BRIDGE_STAGED_NOT_CONSUMED"
                and ledger_values["bridge"][1] != report_payload
            ):
                raise RunnerError(
                    "INVALID_ATTEMPT2_META_V11_BRIDGE",
                    "staged bridge report differs from exact preflight replay",
                )
            _write_stream(out, report_payload)
            return 0
        if report["disposition"] != core.BRIDGE_ADMITTED:
            raise RunnerError(
                core.BRIDGE_REFUSED,
                "nonqualified Meta V11 report cannot admit Q0.5b attempt-2",
            )

        ledger_fd = _open_ledger(create=True)
        assert ledger_fd is not None
        ledger_created = True
        state, ledger_values = _ledger_snapshot(ledger_fd)
        if state.startswith("CONSUMED_"):
            ledger_consumed = True
            bridge_report_created = "bridge" in ledger_values
            _validate_consumed_state(
                core, ledger_values, loaded["attempt2_bridge_contract"]
            )
            raise RunnerError(
                core.ALREADY_CONSUMED,
                "attempt-2 authorization was already consumed; retry is forbidden",
            )
        if state == "BRIDGE_STAGED_NOT_CONSUMED":
            if ledger_values["bridge"][1] != report_payload:
                raise RunnerError(
                    "INVALID_ATTEMPT2_META_V11_BRIDGE",
                    "staged bridge report differs from exact replay",
                )
        else:
            _publish(ledger_fd, LEDGER_NAMES["bridge"], report_payload)

        with _ActualSourceSession(core) as actual:
            consumed = core.build_attempt2_consumed_record_v1(
                report, loaded["attempt2_bridge_contract"]
            )
            consumed_payload = core.canonical_json_bytes_v1(consumed)
            try:
                consumed_status = _publish(
                    ledger_fd, LEDGER_NAMES["consumed"], consumed_payload
                )
            except FileExistsError as error:
                ledger_consumed = True
                raise RunnerError(
                    core.ALREADY_CONSUMED,
                    "attempt-2 authorization was concurrently consumed",
                ) from error
            ledger_consumed = True
            evidence = _consumed_evidence(consumed_status, consumed_payload)
            core.validate_consumed_file_evidence_v1(evidence, consumed)
            _close_ledger(ledger_fd)
            ledger_fd = None
            actual_error: BaseException | None = None
            actual_result: Mapping[str, Any] | None = None
            actual_returned = False
            try:
                actual_called = True
                # The unchanged entrypoint owns Docker/Cargo.  Once control
                # crosses this boundary the wrapper conservatively records
                # that either may have been invoked, even if the call raises.
                docker_or_cargo_called = True
                actual_result = actual.call(args.source_commit)
                actual_returned = True
            except BaseException as error:  # consumption makes every failure terminal
                actual_error = error
        source_receipt = actual.receipt()

        if actual_error is None:
            try:
                artifact_raw, _artifact_status = _read_file(
                    FROZEN_ARTIFACT, "Q0.5b actual artifact"
                )
                artifact_written = True
                execution = core.build_attempt2_execution_link_v1(
                    report,
                    consumed,
                    actual_result,
                    artifact_raw,
                    loaded["attempt2_bridge_contract"],
                    consumed_file_evidence=evidence,
                    hegel_actual_source_only_import_receipt=source_receipt,
                    meta_live_source_validation_receipt=(
                        meta_live_source_receipt
                    ),
                    hegel_live_source_validation_receipt=(
                        hegel_live_source_receipt
                    ),
                )
            except BaseException as error:
                actual_error = error
            else:
                payload = core.canonical_json_bytes_v1(execution)
                ledger_fd = _open_ledger(create=False)
                assert ledger_fd is not None
                state, _values = _ledger_snapshot(ledger_fd)
                if state != "CONSUMED_OUTCOME_UNKNOWN":
                    raise RunnerError(
                        "INVALID_ATTEMPT2_META_V11_BRIDGE",
                        "ledger state changed before success linkage",
                    )
                _publish(ledger_fd, LEDGER_NAMES["execution"], payload)
                _close_ledger(ledger_fd)
                ledger_fd = None
                _write_stream(out, payload)
                return 0

        assert actual_error is not None
        failure_code = getattr(actual_error, "code", type(actual_error).__name__)
        failure_detail = getattr(actual_error, "detail", str(actual_error))
        if not failure_detail:
            failure_detail = type(actual_error).__name__
        artifact_written = os.path.lexists(FROZEN_ARTIFACT)
        failure = core.build_attempt2_failure_link_v1(
            report,
            consumed,
            str(failure_code),
            str(failure_detail)[:4096],
            artifact_written,
            loaded["attempt2_bridge_contract"],
            consumed_file_evidence=evidence,
            hegel_actual_source_only_import_receipt=source_receipt,
            meta_live_source_validation_receipt=meta_live_source_receipt,
            hegel_live_source_validation_receipt=hegel_live_source_receipt,
            actual_returned=actual_returned,
        )
        payload = core.canonical_json_bytes_v1(failure)
        ledger_fd = _open_ledger(create=False)
        assert ledger_fd is not None
        state, _values = _ledger_snapshot(ledger_fd)
        if state != "CONSUMED_OUTCOME_UNKNOWN":
            raise RunnerError(
                "INVALID_ATTEMPT2_META_V11_BRIDGE",
                "ledger state changed before failure linkage",
            )
        _publish(ledger_fd, LEDGER_NAMES["failure"], payload)
        _close_ledger(ledger_fd)
        ledger_fd = None
        _write_stream(out, payload)
        return 1
    except SystemExit as error:
        # ``argparse`` uses ``SystemExit(0)`` for ``--help``. Preserve that
        # successful public CLI contract instead of translating it into a
        # supervisor error wire. Parser errors use ``_StrictParser.error``
        # above and therefore still take the fail-closed RunnerError route.
        if error.code == 0:
            return 0
        raise
    except MissingV11:
        _write_stream(err, _canonical(_missing_wire(core)))
        return 2
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            raise
        code = getattr(error, "code", "INVALID_ATTEMPT2_META_V11_BRIDGE")
        detail = getattr(error, "detail", str(error)) or type(error).__name__
        if actual_called:
            artifact_written = artifact_written or os.path.lexists(FROZEN_ARTIFACT)
        facts = {
            "bridge_report_created": bridge_report_created,
            "ledger_created": ledger_created,
            "ledger_consumed": ledger_consumed,
            "actual_called": actual_called,
            "docker_or_cargo_called": docker_or_cargo_called,
            "q05b_artifact_written": artifact_written,
        }
        if core is not None and hasattr(core, "build_attempt2_supervisor_error_v1"):
            wire = core.build_attempt2_supervisor_error_v1(
                str(code), str(detail)[:4096], **facts
            )
            core.validate_attempt2_supervisor_error_v1(wire)
        else:
            wire = _error_wire(
                str(code),
                str(detail),
                bridge_report_created=facts["bridge_report_created"],
                ledger_created=facts["ledger_created"],
                ledger_consumed=facts["ledger_consumed"],
                actual_called=facts["actual_called"],
                docker_or_cargo_called=facts["docker_or_cargo_called"],
                artifact_written=facts["q05b_artifact_written"],
            )
        _write_stream(err, _canonical(wire))
        return 2
    finally:
        _close_ledger(ledger_fd)


if __name__ == "__main__":
    raise SystemExit(main())
