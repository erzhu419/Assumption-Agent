#!/usr/bin/env python3
"""Preflight or execute the one formal Q0.5b attempt-2 Target-D campaign.

Every recoverable environment/path/format/dependency/CPU/scheduling failure is
reported as stateless ``PENDING_ATTEMPT2_TARGET_D_V2_NOT_CONSUMED`` before a
ledger or supervisor process exists.  Once the authorization is consumed, the
attempt is one-shot and every outcome is terminal.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
from hashlib import sha256
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Final, Mapping, NoReturn, Sequence


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
SRC_ROOT: Final = PROJECT_ROOT / "src"
if os.fspath(SRC_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(SRC_ROOT))

from hegel_machine import phase3_q05b_attempt2_target_d_campaign_v2 as _core


CONFIG_PATH: Final = (
    PROJECT_ROOT / "config/phase3_q05b_attempt2_target_d_campaign_v2.json"
)
SHADOW_LOCATOR_CONFIG_PATH: Final = (
    PROJECT_ROOT / "config/phase3_q05b_shadow_retry_campaign_v2.json"
)
SHADOW_TOOL_RELATIVE_PATH: Final = Path(
    "tools/phase3_q05b_shadow_retry_campaign_v2.py"
)
META_BRIDGE_CORE_PATH: Final = (
    PROJECT_ROOT / "src/hegel_machine/phase3_q05b_attempt2_meta_v11_bridge_v1.py"
)
MAX_SMALL_JSON: Final = 64 * 1024 * 1024
PENDING_EXIT_CODE: Final = 75


class GatePending(RuntimeError):
    def __init__(self, phase: str, condition: str, detail: str) -> None:
        self.phase = phase
        self.condition = condition
        self.detail = detail
        super().__init__(f"{phase}/{condition}: {detail}")


class CampaignTerminal(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _terminal(code: str, detail: str) -> NoReturn:
    raise CampaignTerminal(code, detail)


class _CampaignLock:
    def __init__(self, path: Path, descriptor: int, identity: tuple[int, int]) -> None:
        self.path = path
        self.descriptor = descriptor
        self.identity = identity


def _campaign_lock_identity(lock: _CampaignLock) -> tuple[int, int]:
    try:
        held = os.fstat(lock.descriptor)
        current = lock.path.lstat()
    except OSError as error:
        _terminal("CAMPAIGN_LOCK_IDENTITY_UNCERTAIN", str(error))
    if (
        lock.path.is_symlink()
        or not stat.S_ISREG(held.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or held.st_nlink != 1
        or current.st_nlink != 1
        or held.st_uid != os.geteuid()
        or current.st_uid != os.geteuid()
        or stat.S_IMODE(held.st_mode) != 0o600
        or stat.S_IMODE(current.st_mode) != 0o600
        or held.st_size != 0
        or current.st_size != 0
        or (held.st_dev, held.st_ino) != lock.identity
        or (current.st_dev, current.st_ino) != lock.identity
    ):
        _terminal("CAMPAIGN_LOCK_IDENTITY_UNCERTAIN", "stable lock identity differs")
    return lock.identity


def _contract_campaign_lock_identity(
    contract: Mapping[str, Any], lock: _CampaignLock
) -> tuple[int, int]:
    expected = Path(str(contract["fixed_paths"]["campaign_lock_path"]))
    if lock.path != expected:
        _terminal("CAMPAIGN_LOCK_IDENTITY_UNCERTAIN", "lock path is not contract-bound")
    return _campaign_lock_identity(lock)


def _acquire_campaign_lock(contract: Mapping[str, Any]) -> _CampaignLock:
    path = Path(str(contract["fixed_paths"]["campaign_lock_path"]))
    parent_fd = -1
    descriptor = -1
    created = False
    try:
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        flags = os.O_RDWR | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(
                path.name,
                flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_fd,
            )
            created = True
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            os.fsync(parent_fd)
        except FileExistsError:
            descriptor = os.open(path.name, flags, dir_fd=parent_fd)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        status = os.fstat(descriptor)
        lock = _CampaignLock(path, descriptor, (status.st_dev, status.st_ino))
        _campaign_lock_identity(lock)
        os.close(parent_fd)
        return lock
    except BaseException as original:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if created and parent_fd >= 0:
            try:
                os.unlink(path.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        if parent_fd >= 0:
            try:
                os.close(parent_fd)
            except OSError:
                pass
        if isinstance(original, CampaignTerminal):
            raise
        _terminal(
            "CAMPAIGN_LOCK_UNAVAILABLE_STATE_UNKNOWN_NO_RESUME",
            f"{type(original).__name__}:{original}",
        )


def _release_campaign_lock(lock: _CampaignLock) -> None:
    _campaign_lock_identity(lock)
    try:
        fcntl.flock(lock.descriptor, fcntl.LOCK_UN)
        os.close(lock.descriptor)
    except OSError as error:
        _terminal("CAMPAIGN_LOCK_RELEASE_UNCERTAIN", str(error))


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _terminal("INVALID_CANONICAL_JSON", f"duplicate key: {key}")
        result[key] = value
    return result


def _strict_json(raw: bytes, label: str, maximum: int = MAX_SMALL_JSON) -> object:
    if type(raw) is not bytes or len(raw) > maximum:
        _terminal("INVALID_CANONICAL_JSON", f"{label} exceeds byte bound")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=lambda token: _terminal(
                "INVALID_CANONICAL_JSON", f"{label} constant differs: {token}"
            ),
        )
    except CampaignTerminal:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        _terminal("INVALID_CANONICAL_JSON", f"{label} decode failed: {error}")


def _read_regular(path: Path, label: str, maximum: int = MAX_SMALL_JSON) -> bytes:
    try:
        status = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise OSError("not one ordinary single-link file")
        if status.st_size > maximum:
            raise OSError(f"exceeds {maximum}-byte bound")
        raw = path.read_bytes()
    except OSError as error:
        raise GatePending("PATH_AND_FORMAT", "REPAIRABLE_FILE_UNAVAILABLE", f"{label}: {error}")
    if len(raw) != status.st_size:
        raise GatePending("PATH_AND_FORMAT", "REPAIRABLE_FILE_CHANGED", f"{label} size changed")
    return raw


def _legacy_domain_digest(domain: str, value: object) -> str:
    raw = _core.canonical_json_bytes_v2(value)
    return "sha256:" + sha256(domain.encode("ascii") + b"\x00" + raw[:-1]).hexdigest()


def _receipt_digest(label: str, value: object) -> str:
    domain = f"HEGEL/Q05B/ATTEMPT2/TARGET_D/GATE/{label}/V2".encode("ascii") + b"\x00"
    return "sha256:" + sha256(domain + _core.canonical_json_bytes_v2(value)).hexdigest()


def load_contract_v2() -> dict[str, Any]:
    try:
        raw = CONFIG_PATH.read_bytes()
    except OSError as error:
        _terminal("INVALID_CAMPAIGN_CONTRACT", f"contract unavailable: {error}")
    decoded = _strict_json(raw, "campaign contract")
    try:
        return _core.validate_contract_v2(decoded)
    except _core.Attempt2TargetDCampaignValidationError as error:
        _terminal("INVALID_CAMPAIGN_CONTRACT", str(error))


def _git(project: Path, *arguments: str, timeout: float = 30.0) -> str:
    try:
        result = subprocess.run(
            ["/usr/bin/git", "-C", os.fspath(project), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GatePending("SOURCE_LINEAGE", "READONLY_GIT_UNAVAILABLE", str(error))
    if result.returncode != 0:
        raise GatePending(
            "SOURCE_LINEAGE",
            "READONLY_GIT_REJECTED",
            result.stderr.decode("utf-8", "replace")[:2048],
        )
    try:
        return result.stdout.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise GatePending("SOURCE_LINEAGE", "READONLY_GIT_FORMAT", str(error))


def _authorization_gate(contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("authorization") != _core.OWNER_AUTHORIZATION:
        _terminal("AUTHORIZATION_BINDING_DIFFERS", "exact owner authorization text differs")
    return {
        "status": "PASS",
        "authorization_text": _core.OWNER_AUTHORIZATION_TEXT,
        "authorization_digest": _core.OWNER_AUTHORIZATION_DIGEST,
    }


def _path_format_gate(
    contract: Mapping[str, Any], source_commit: str | None, manifest: Path | None
) -> dict[str, Any]:
    paths = contract["fixed_paths"]
    assert type(paths) is dict
    if Path(str(paths["project_root"])) != PROJECT_ROOT:
        _terminal("PATH_BINDING_DIFFERS", "tool project root differs from contract")
    if source_commit is None or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise GatePending(
            "PATH_AND_FORMAT", "SOURCE_COMMIT_REQUIRED", "provide one full lowercase --source-commit"
        )
    if manifest is not None and not manifest.is_absolute():
        raise GatePending(
            "PATH_AND_FORMAT", "META_CHAIN_MANIFEST_NOT_ABSOLUTE", "manifest path must be absolute"
        )
    return {
        "status": "PASS",
        "source_commit": source_commit,
        "meta_chain_manifest_or_null": None if manifest is None else os.fspath(manifest),
    }


def _source_lineage_gate(contract: Mapping[str, Any], source_commit: str) -> dict[str, Any]:
    lineage = contract["source_lineage"]
    assert type(lineage) is dict
    head = _git(PROJECT_ROOT, "rev-parse", "--verify", "HEAD")
    tree = _git(PROJECT_ROOT, "rev-parse", "--verify", "HEAD^{tree}")
    status = _git(PROJECT_ROOT, "status", "--porcelain=v1", "--untracked-files=all")
    if head != source_commit:
        raise GatePending("SOURCE_LINEAGE", "REQUESTED_HEAD_DIFFERS", f"requested={source_commit}; head={head}")
    if status:
        raise GatePending("SOURCE_LINEAGE", "SOURCE_TREE_NOT_CLEAN", "commit the authorized additive slices first")
    if _git(PROJECT_ROOT, "rev-parse", "--verify", str(lineage["required_ancestor_commit"]) + "^{tree}") != lineage["required_ancestor_tree"]:
        _terminal("SOURCE_LINEAGE_CORRUPT", "required post-Meta ancestor tree differs")
    if _git(PROJECT_ROOT, "rev-parse", "--verify", str(lineage["required_ancestor_commit"]) + "^") != lineage["required_ancestor_parent"]:
        _terminal("SOURCE_LINEAGE_CORRUPT", "required post-Meta ancestor parent differs")
    try:
        _git(PROJECT_ROOT, "merge-base", "--is-ancestor", str(lineage["required_ancestor_commit"]), head)
        _git(PROJECT_ROOT, "merge-base", "--is-ancestor", str(lineage["target_d_commit"]), head)
    except GatePending as error:
        raise GatePending("SOURCE_LINEAGE", "REQUIRED_ANCESTRY_MISSING", error.detail)
    parent_row = _git(PROJECT_ROOT, "rev-list", "--parents", "-n", "1", head).split()
    if parent_row != [head, str(lineage["required_ancestor_commit"])]:
        raise GatePending(
            "SOURCE_LINEAGE",
            "ATTEMPT2_SUCCESSOR_PARENT_DIFFERS",
            "formal attempt-2 source must be the direct additive child of the exact post-Meta base",
        )
    project_prefix = _git(PROJECT_ROOT, "rev-parse", "--show-prefix")
    if project_prefix and not project_prefix.endswith("/"):
        _terminal("SOURCE_LINEAGE_CORRUPT", "Git project prefix format differs")
    expected_diff = [
        f"A\t{project_prefix}{relative}"
        for relative in lineage["additive_paths_in_exact_order"]
    ]
    observed_diff = _git(
        PROJECT_ROOT,
        "diff",
        "--name-status",
        "--no-renames",
        str(lineage["required_ancestor_commit"]),
        head,
    ).splitlines()
    if observed_diff != expected_diff:
        raise GatePending(
            "SOURCE_LINEAGE",
            "ATTEMPT2_SUCCESSOR_DIFF_NOT_EXACT_ADDITIVE_FIVE",
            repr(observed_diff)[:4096],
        )
    for relative in lineage["additive_paths_in_exact_order"]:
        path = PROJECT_ROOT / str(relative)
        try:
            tracked = _git(PROJECT_ROOT, "ls-files", "--error-unmatch", str(relative))
        except GatePending:
            raise GatePending("SOURCE_LINEAGE", "ATTEMPT2_ADDITIVE_PATH_UNCOMMITTED", str(relative))
        if tracked != relative or not path.is_file():
            raise GatePending("SOURCE_LINEAGE", "ATTEMPT2_ADDITIVE_PATH_MISSING", str(relative))
    return {"status": "PASS", "head": head, "tree": tree, "required_ancestor": lineage["required_ancestor_commit"]}


def _load_meta_chain_manifest(
    contract: Mapping[str, Any], manifest_path: Path
) -> tuple[dict[str, Any], tuple[object, ...], dict[str, str], dict[str, dict[str, Any]]]:
    policy = contract["meta_v11_bridge"]
    assert type(policy) is dict
    raw = _read_regular(manifest_path, "Meta V11 chain manifest")
    try:
        value = _strict_json(raw, "Meta V11 chain manifest")
    except CampaignTerminal as error:
        raise GatePending("REAL_META_V11_CHAIN", "META_CHAIN_MANIFEST_FORMAT", error.detail)
    if type(value) is not dict or set(value) != {"schema_version", "artifacts", "expected_digests"}:
        raise GatePending("REAL_META_V11_CHAIN", "META_CHAIN_MANIFEST_FORMAT", "field registry differs")
    if value["schema_version"] != policy["chain_manifest_schema_version"]:
        raise GatePending("REAL_META_V11_CHAIN", "META_CHAIN_MANIFEST_FORMAT", "schema differs")
    artifacts = value["artifacts"]
    expected = value["expected_digests"]
    names = list(policy["positional_names_in_exact_order"])
    digest_names = list(policy["expected_digest_names_in_exact_order"])
    if type(artifacts) is not dict or list(artifacts) != names:
        raise GatePending("REAL_META_V11_CHAIN", "INCOMPLETE_REAL_META_V11_CHAIN", "artifact order or registry differs")
    if type(expected) is not dict or list(expected) != digest_names:
        raise GatePending("REAL_META_V11_CHAIN", "INCOMPLETE_REAL_META_V11_CHAIN", "expected digest registry differs")
    positional: list[object] = []
    rows: dict[str, dict[str, Any]] = {}
    identities: set[tuple[str, int, str]] = set()
    for name in names:
        row = artifacts[name]
        if type(row) is not dict or set(row) != {"path", "bytes", "sha256"}:
            raise GatePending("REAL_META_V11_CHAIN", "META_CHAIN_ARTIFACT_FORMAT", name)
        path = Path(str(row.get("path")))
        if not path.is_absolute() or type(row.get("bytes")) is not int or type(row.get("sha256")) is not str:
            raise GatePending("REAL_META_V11_CHAIN", "META_CHAIN_ARTIFACT_FORMAT", name)
        payload = _read_regular(path, f"Meta artifact {name}")
        digest = sha256(payload).hexdigest()
        if len(payload) != row["bytes"] or digest != row["sha256"]:
            raise GatePending("REAL_META_V11_CHAIN", "META_CHAIN_ARTIFACT_CHANGED", name)
        identity = (os.fspath(path), len(payload), digest)
        if identity in identities:
            raise GatePending("REAL_META_V11_CHAIN", "META_CHAIN_ARTIFACT_NOT_DISTINCT", name)
        identities.add(identity)
        try:
            decoded = _strict_json(payload, f"Meta artifact {name}")
        except CampaignTerminal as error:
            raise GatePending("REAL_META_V11_CHAIN", "META_CHAIN_ARTIFACT_FORMAT", f"{name}: {error.detail}")
        if type(decoded) is not dict:
            raise GatePending("REAL_META_V11_CHAIN", "META_CHAIN_ARTIFACT_FORMAT", name)
        positional.append(decoded)
        rows[name] = {"path": os.fspath(path), "bytes": len(payload), "sha256": digest}
    for name in digest_names:
        if type(expected[name]) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", expected[name]) is None:
            raise GatePending("REAL_META_V11_CHAIN", "META_CHAIN_EXPECTED_DIGEST_FORMAT", name)
    return value, tuple(positional), dict(expected), rows


def _meta_v11_gate(
    contract: Mapping[str, Any], manifest_path: Path
) -> dict[str, Any]:
    policy = contract["meta_v11_bridge"]
    assert type(policy) is dict
    meta_root = Path(str(policy["source_repo_root"]))
    if sha256(_read_regular(META_BRIDGE_CORE_PATH, "frozen Meta bridge core")).hexdigest() != policy["bridge_core_raw_sha256"]:
        _terminal("META_BRIDGE_CORE_DIFFERS", "old bridge core raw hash differs")
    meta_head = _git(meta_root, "rev-parse", "--verify", "HEAD")
    meta_tree = _git(meta_root, "rev-parse", "--verify", "HEAD^{tree}")
    meta_status = _git(meta_root, "status", "--porcelain=v1", "--untracked-files=all")
    if meta_head != policy["source_commit"] or meta_tree != policy["source_tree"] or meta_status:
        raise GatePending("REAL_META_V11_CHAIN", "META_SOURCE_BINDING_UNAVAILABLE", "Meta V11 source root is not exact clean frozen source")
    _manifest, positional, expected, artifacts = _load_meta_chain_manifest(contract, manifest_path)
    bridge = importlib.import_module("hegel_machine.phase3_q05b_attempt2_meta_v11_bridge_v1")
    try:
        receipt, source_only = bridge._public_v11_replay(positional, expected, artifacts)
    except bridge.MissingRealV11ReportError:
        raise GatePending("REAL_META_V11_CHAIN", "MISSING_REAL_META_V11_CHAIN", "real post-transition report is absent")
    except bridge.Phase3Q05BAttempt2MetaV11BridgeValidationError as error:
        raise GatePending("REAL_META_V11_CHAIN", "META_V11_REPLAY_NOT_YET_QUALIFIED", str(error))
    if receipt.get("status") != policy["required_replay_status"] or receipt.get("disposition") != policy["required_disposition"]:
        raise GatePending("REAL_META_V11_CHAIN", "META_V11_REPLAY_NOT_YET_QUALIFIED", repr(receipt.get("status")))
    return {
        "status": "PASS",
        "receipt_digest": receipt["report_digest"],
        "replay_receipt": receipt,
        "source_only_receipt_digest": _receipt_digest("META_SOURCE_ONLY", source_only),
        "manifest_raw_sha256": sha256(_read_regular(manifest_path, "Meta V11 chain manifest replay")).hexdigest(),
    }


def _target_d_gate(contract: Mapping[str, Any]) -> dict[str, Any]:
    policy = contract["target_d_v2"]
    assert type(policy) is dict
    rows: list[dict[str, object]] = []
    for relative, expected in policy["required_raw_sha256_by_path"].items():
        path = PROJECT_ROOT / str(relative)
        raw = _read_regular(path, f"Target-D source {relative}", maximum=512 * 1024 * 1024)
        observed = sha256(raw).hexdigest()
        if observed != expected:
            raise GatePending("TARGET_D_OBJECT_CLOSURE", "TARGET_D_SOURCE_DRIFT", str(relative))
        rows.append({"path": relative, "bytes": len(raw), "sha256": observed})
    started = time.monotonic_ns()
    try:
        completed = subprocess.run(
            [
                "/usr/bin/git", "-C", os.fspath(PROJECT_ROOT), "fsck",
                "--connectivity-only", "--no-dangling", str(policy["target_commit"]),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    except subprocess.TimeoutExpired:
        raise GatePending("TARGET_D_OBJECT_CLOSURE", "READONLY_COMMAND_TIMEOUT", "git fsck reached 300 seconds")
    except OSError as error:
        raise GatePending("TARGET_D_OBJECT_CLOSURE", "READONLY_COMMAND_UNAVAILABLE", str(error))
    if completed.returncode != 0 or completed.stdout or completed.stderr:
        _terminal(
            "TARGET_D_OBJECT_CLOSURE_CORRUPT",
            (completed.stdout + completed.stderr).decode("utf-8", "replace")[:4096],
        )
    return {
        "status": "PASS",
        "target_commit": policy["target_commit"],
        "target_tree": policy["target_tree"],
        "source_rows_digest": _receipt_digest("TARGET_D_SOURCE_ROWS", rows),
        "fsck_elapsed_ns": time.monotonic_ns() - started,
    }


_DEPENDENCY_PROGRAM: Final = r'''
import hashlib, importlib.util, json, pathlib, sys
source_root = pathlib.Path(sys.argv[1])
lock_path = pathlib.Path(sys.argv[2])
sys.path.insert(0, str(source_root))
import hegel_machine._vendor.tomli as tomli
raw = lock_path.read_bytes()
value = tomli.loads(raw.decode("utf-8"))
packages = value.get("package")
if type(packages) is not list:
    raise SystemExit(91)
result = {
    "python": [sys.version_info.major, sys.version_info.minor, sys.version_info.micro],
    "tomli_origin": str(pathlib.Path(tomli.__file__).resolve()),
    "top_level_tomli_spec_is_null": importlib.util.find_spec("tomli") is None,
    "cargo_lock_sha256": hashlib.sha256(raw).hexdigest(),
    "cargo_lock_bytes": len(raw),
    "cargo_lock_version": value.get("version"),
    "cargo_package_count": len(packages),
}
print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
'''


def _dependency_gate(contract: Mapping[str, Any]) -> dict[str, Any]:
    interpreter = Path(str(contract["execution_policy"]["supervisor_interpreter"]))
    lock_path = PROJECT_ROOT / "rust/q1_archive_projection_oracle/Cargo.lock"
    if not interpreter.is_file():
        raise GatePending("PYTHON310_VENDORED_TOMLI_CARGO", "ISOLATED_PYTHON310_UNAVAILABLE", os.fspath(interpreter))
    try:
        completed = subprocess.run(
            [
                os.fspath(interpreter), "-I", "-S", "-B", "-X", "utf8",
                "-X", "pycache_prefix=/nonexistent/hegel-q05b-attempt2-v2-pyc",
                "-c", _DEPENDENCY_PROGRAM, os.fspath(SRC_ROOT), os.fspath(lock_path),
            ],
            cwd="/",
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GatePending("PYTHON310_VENDORED_TOMLI_CARGO", "DEPENDENCY_GATE_UNAVAILABLE", str(error))
    if completed.returncode != 0 or completed.stderr:
        raise GatePending(
            "PYTHON310_VENDORED_TOMLI_CARGO",
            "DEPENDENCY_GATE_REJECTED",
            completed.stderr.decode("utf-8", "replace")[:4096],
        )
    try:
        receipt = _strict_json(completed.stdout, "dependency gate receipt")
    except CampaignTerminal as error:
        raise GatePending("PYTHON310_VENDORED_TOMLI_CARGO", "DEPENDENCY_GATE_FORMAT", error.detail)
    if (
        type(receipt) is not dict
        or receipt.get("python", [None, None])[:2] != [3, 10]
        or receipt.get("top_level_tomli_spec_is_null") is not True
        or receipt.get("cargo_lock_sha256") != "c4b9d1eff522017c85b151696d082a0ff9f88bd2890c93d932f7467323a13b49"
        or receipt.get("cargo_lock_bytes") != 6493
        or receipt.get("cargo_lock_version") != 4
        or receipt.get("cargo_package_count") != 29
    ):
        raise GatePending("PYTHON310_VENDORED_TOMLI_CARGO", "DEPENDENCY_GATE_REPLAY_DIFFERS", repr(receipt)[:4096])
    return {"status": "PASS", "receipt": receipt}


def _history_gate(contract: Mapping[str, Any]) -> dict[str, Any]:
    policy = contract["historical_v2r20_evidence"]
    assert type(policy) is dict
    receipt_path = Path(str(policy["preauthorization_live_gauntlet_receipt_path"]))
    raw = _read_regular(receipt_path, "v2r20 live gauntlet receipt")
    if sha256(raw).hexdigest() != policy["preauthorization_live_gauntlet_receipt_raw_sha256"]:
        raise GatePending("HISTORICAL_V2R20_EVIDENCE", "LIVE_GAUNTLET_RECEIPT_CHANGED", os.fspath(receipt_path))
    value = _strict_json(raw, "v2r20 live gauntlet receipt")
    if type(value) is not dict:
        raise GatePending("HISTORICAL_V2R20_EVIDENCE", "LIVE_GAUNTLET_RECEIPT_FORMAT", "not an object")
    unsigned = deepcopy(value)
    observed = unsigned.pop("receipt_digest", None)
    if (
        observed != policy["preauthorization_live_gauntlet_receipt_digest"]
        or _legacy_domain_digest(policy["preauthorization_live_gauntlet_receipt_domain"], unsigned) != observed
        or value.get("status") != "PASS"
        or any(
            value.get(field) != 0
            for field in (
                "authorization_consumption_count",
                "campaign_execution_count",
                "supervisor_invocation_count",
                "docker_daemon_contact_count",
                "network_contact_count",
            )
        )
    ):
        raise GatePending("HISTORICAL_V2R20_EVIDENCE", "LIVE_GAUNTLET_RECEIPT_REPLAY_DIFFERS", "immutable receipt differs")
    shadow_config = _strict_json(
        _read_regular(SHADOW_LOCATOR_CONFIG_PATH, "v2r20 contract"),
        "v2r20 contract",
    )
    if (
        type(shadow_config) is not dict
        or shadow_config.get("decision_audit_policy", {}).get("policy_digest")
        != policy["decision_audit_policy_digest"]
    ):
        raise GatePending("HISTORICAL_V2R20_EVIDENCE", "DECISION_AUDIT_POLICY_DIFFERS", "v2r20 policy digest differs")
    slot = _strict_json(
        _read_regular(Path(str(policy["successful_shadow_slot_result_path"])), "v2r20 successful slot result"),
        "v2r20 successful slot result",
    )
    completion = _strict_json(
        _read_regular(Path(str(policy["successful_shadow_completion_path"])), "v2r20 completion"),
        "v2r20 completion",
    )
    if (
        type(slot) is not dict
        or slot.get("decision") != "SUCCESS"
        or slot.get("decision_audit", {}).get("decision") != slot.get("decision")
        or type(completion) is not dict
        or completion.get("status") != "CAMPAIGN_COMPLETION_COMMITTED"
        or completion.get("body", {}).get("terminal_status")
        != "CAMPAIGN_SHADOW_Q05B_SUCCESS_NONADOPTING"
    ):
        raise GatePending("HISTORICAL_V2R20_EVIDENCE", "SHADOW_SUCCESS_REPLAY_DIFFERS", "historical evidence differs")
    return {
        "status": "PASS_HISTORICAL_EVIDENCE_ONLY_NOT_FORMAL_ACTUAL",
        "live_gauntlet_receipt_digest": observed,
        "decision_audit_policy_digest": policy["decision_audit_policy_digest"],
        "shadow_slot_result_digest": slot.get("record_digest"),
        "shadow_completion_record_digest": completion.get("record_digest"),
    }


def _load_shadow_tool(contract: Mapping[str, Any]) -> Any:
    history = contract["historical_v2r20_evidence"]
    assert type(history) is dict
    locator = _strict_json(
        _read_regular(SHADOW_LOCATOR_CONFIG_PATH, "v2r20 source locator"),
        "v2r20 source locator",
    )
    execution = locator.get("execution_policy") if type(locator) is dict else None
    if (
        type(locator) is not dict
        or locator.get("contract_digest") != history["campaign_contract_digest"]
        or type(execution) is not dict
        or type(execution.get("clone_source_repository")) is not str
        or type(execution.get("clone_source_project_relative_path")) is not str
    ):
        raise GatePending(
            "HISTORICAL_V2R20_EVIDENCE",
            "BOUND_SOURCE_LOCATOR_DIFFERS",
            "frozen v2r20 source locator or contract digest differs",
        )
    worktree_root = Path(execution["clone_source_repository"])
    project_relative = Path(execution["clone_source_project_relative_path"])
    if (
        not worktree_root.is_absolute()
        or project_relative.is_absolute()
        or project_relative.parts != (PROJECT_ROOT.name,)
    ):
        raise GatePending(
            "HISTORICAL_V2R20_EVIDENCE",
            "BOUND_SOURCE_PATH_DIFFERS",
            "frozen v2r20 worktree/project path is not the exact supported form",
        )
    historical_project_root = worktree_root / project_relative
    try:
        historical_head = _git(
            historical_project_root,
            "rev-parse",
            "--verify",
            "HEAD",
        )
    except GatePending as error:
        raise GatePending(
            "HISTORICAL_V2R20_EVIDENCE",
            "BOUND_SOURCE_COMMIT_UNAVAILABLE",
            error.detail,
        )
    if historical_head != history["campaign_commit"]:
        raise GatePending(
            "HISTORICAL_V2R20_EVIDENCE",
            "BOUND_SOURCE_COMMIT_DIFFERS",
            f"expected={history['campaign_commit']}; observed={historical_head}",
        )
    shadow_tool_path = historical_project_root / SHADOW_TOOL_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location(
        "_hegel_q05b_v2r20_gate_source",
        shadow_tool_path,
    )
    if spec is None or spec.loader is None:
        raise GatePending(
            "HISTORICAL_V2R20_EVIDENCE",
            "BOUND_SOURCE_TOOL_UNAVAILABLE",
            os.fspath(shadow_tool_path),
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(spec.name, None)
        raise GatePending(
            "HISTORICAL_V2R20_EVIDENCE",
            "BOUND_SOURCE_TOOL_UNAVAILABLE",
            f"{type(error).__name__}:{error}",
        )
    return module


def _cpu_gate(shadow: Any, shadow_contract: Mapping[str, object]) -> dict[str, Any]:
    try:
        receipt = shadow._host_cpu_topology_affinity_admission_v2r17(
            shadow_contract,
            dependency_gate_invocation_count=1,
            target_fsck_invocation_count=1,
        )
    except shadow.ShadowRetryCampaignPendingV1 as error:
        raise GatePending(error.phase, error.condition, error.detail)
    except shadow.ShadowRetryCampaignErrorV2 as error:
        _terminal(error.code, error.detail)
    return {"status": "PASS", "receipt": receipt}


def _open_scheduling_probe_descriptor(path: Path) -> int:
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)


def _scheduling_gate(
    contract: Mapping[str, Any], shadow: Any, shadow_contract: Mapping[str, object]
) -> dict[str, Any]:
    probe = Path(str(contract["fixed_paths"]["scheduling_probe_root"]))
    if os.path.lexists(probe):
        raise GatePending("SCHEDULING_ADMISSION", "TRANSIENT_PROBE_ROOT_PREEXISTS", os.fspath(probe))
    parent = probe.parent
    parent_fd = -1
    probe_fd = -1
    created = False
    try:
        parent_status = parent.lstat()
        if parent.is_symlink() or not stat.S_ISDIR(parent_status.st_mode):
            raise OSError("probe parent is not an ordinary directory")
        os.mkdir(probe, 0o700)
        created = True
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        os.fsync(parent_fd)
        probe_fd = _open_scheduling_probe_descriptor(probe)
    except OSError as error:
        cleanup_errors: list[str] = []
        if probe_fd >= 0:
            try:
                os.close(probe_fd)
            except OSError as cleanup_error:
                cleanup_errors.append(str(cleanup_error))
            probe_fd = -1
        cleanup_parent_fd = parent_fd
        if created and cleanup_parent_fd < 0:
            try:
                cleanup_parent_fd = os.open(
                    parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
                )
            except OSError as cleanup_error:
                cleanup_errors.append(str(cleanup_error))
        if created and cleanup_parent_fd >= 0:
            try:
                os.rmdir(probe.name, dir_fd=cleanup_parent_fd)
                os.fsync(cleanup_parent_fd)
                if os.path.lexists(probe):
                    cleanup_errors.append("probe root remains after creation failure cleanup")
            except OSError as cleanup_error:
                cleanup_errors.append(str(cleanup_error))
        if cleanup_parent_fd >= 0:
            try:
                os.close(cleanup_parent_fd)
            except OSError as cleanup_error:
                cleanup_errors.append(str(cleanup_error))
        parent_fd = -1
        if cleanup_errors:
            _terminal(
                "SCHEDULING_PROBE_CLEANUP_UNCERTAIN",
                "; ".join(cleanup_errors),
            )
        raise GatePending("SCHEDULING_ADMISSION", "PROBE_ROOT_UNAVAILABLE", str(error))
    admission: dict[str, Any] | None = None
    pending: GatePending | None = None
    failure: BaseException | None = None
    try:
        try:
            admission = shadow._run_scheduling_admission_gate_v2(shadow_contract, probe_fd, "SLOT_01")
        except shadow.ShadowRetryCampaignErrorV2 as error:
            if shadow._scheduling_gate_exhaustion_is_exact_v2r17(shadow_contract, "SLOT_01", error):
                pending = GatePending("SCHEDULING_ADMISSION", "COMPLETE_GATE_EXHAUSTION", error.detail)
            elif shadow._scheduling_gate_joined_task_timeout_is_exact_v2r17(shadow_contract, "SLOT_01", error):
                pending = GatePending("SCHEDULING_ADMISSION", "KNOWN_DEAD_ADMISSION_TASK_QUIESCENCE_TIMEOUT", error.detail)
            else:
                failure = error
    finally:
        cleanup_errors: list[str] = []
        try:
            if os.listdir(probe_fd):
                cleanup_errors.append("probe root is not empty")
        except OSError as error:
            cleanup_errors.append(str(error))
        try:
            os.close(probe_fd)
        except OSError as error:
            cleanup_errors.append(str(error))
        if not cleanup_errors:
            try:
                os.rmdir(probe)
                os.fsync(parent_fd)
                if os.path.lexists(probe):
                    cleanup_errors.append("probe root remains after fsync")
            except OSError as error:
                cleanup_errors.append(str(error))
        try:
            os.close(parent_fd)
        except OSError as error:
            cleanup_errors.append(str(error))
        if cleanup_errors:
            _terminal("SCHEDULING_PROBE_CLEANUP_UNCERTAIN", "; ".join(cleanup_errors))
    if pending is not None:
        raise pending
    if failure is not None:
        _terminal(getattr(failure, "code", "SCHEDULING_ADMISSION_FAILED"), str(failure))
    if type(admission) is not dict or admission.get("passed") is not True:
        _terminal("SCHEDULING_ADMISSION_FAILED", "gate did not return exact PASS")
    return {"status": "PASS", "admission": admission, "probe_cleanup": "FSYNCED_ABSENT"}


def _formal_roots_gate(contract: Mapping[str, Any], *, allow_existing_ledger: bool) -> dict[str, Any]:
    paths = contract["fixed_paths"]
    assert type(paths) is dict
    checked: list[str] = []
    for key in ("artifact_path", "evidence_root", "trajectory_root", "scheduling_probe_root"):
        path = Path(str(paths[key]))
        if os.path.lexists(path):
            raise GatePending("FORMAL_ROOTS_ABSENT", "FORMAL_OUTPUT_ROOT_PREEXISTS", os.fspath(path))
        checked.append(os.fspath(path))
    ledger = Path(str(paths["ledger_root"]))
    if os.path.lexists(ledger) and not allow_existing_ledger:
        raise GatePending("FORMAL_ROOTS_ABSENT", "FORMAL_LEDGER_ROOT_PREEXISTS", os.fspath(ledger))
    checked.append(os.fspath(ledger))
    return {"status": "PASS", "paths": checked, "ledger_existing_allowed_for_terminal_replay": allow_existing_ledger}


def _stable_gate_bindings(
    contract: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    meta: Mapping[str, Any],
    shadow_contract: Mapping[str, object],
) -> dict[str, str]:
    order = list(contract["preconsumption_gate_order"])
    target = contract["target_d_v2"]
    history = contract["historical_v2r20_evidence"]
    fixed = contract["fixed_paths"]
    execution = shadow_contract["execution_policy"]
    assert all(type(item) is dict for item in (target, history, fixed, execution))
    values: dict[str, object] = {
        "AUTHORIZATION_TEXT": contract["authorization"],
        "PATH_AND_FORMAT": fixed,
        "SOURCE_LINEAGE": {"head": source["head"], "tree": source["tree"]},
        "REAL_META_V11_CHAIN": {
            "report_digest": meta["receipt_digest"],
            "required_status": contract["meta_v11_bridge"]["required_replay_status"],
        },
        "TARGET_D_OBJECT_CLOSURE": {
            "target_commit": target["target_commit"],
            "target_tree": target["target_tree"],
            "required_raw_sha256_by_path": target["required_raw_sha256_by_path"],
        },
        "PYTHON310_VENDORED_TOMLI_CARGO": execution["python310_dependency_preflight_gate"],
        "HISTORICAL_V2R20_EVIDENCE": {
            "live_gauntlet_receipt_digest": history["preauthorization_live_gauntlet_receipt_digest"],
            "decision_audit_policy_digest": history["decision_audit_policy_digest"],
        },
        "CPU_AFFINITY": execution["host_cpu_topology_affinity_admission"],
        "SCHEDULING_ADMISSION": execution["scheduling_admission_gate"],
        "FORMAL_ROOTS_ABSENT": {"fixed_paths": fixed, "required_before_first_ledger_write": True},
    }
    if list(values) != order:
        _terminal("GATE_ORDER_DIFFERS", "stable gate binding registry differs")
    return {name: _receipt_digest(name, values[name]) for name in order}


def _evidence_summary(name: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gate": name,
        "status": receipt.get("status"),
        "live_receipt_digest": _receipt_digest(name + "/LIVE", receipt),
    }


def derive_preflight_v2(
    contract: Mapping[str, Any],
    *,
    source_commit: str | None,
    meta_chain_manifest: Path | None,
    require_initial_roots_absent: bool,
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    authorization = _authorization_gate(contract)
    evidence.append(_evidence_summary("AUTHORIZATION_TEXT", authorization))
    paths = _path_format_gate(contract, source_commit, meta_chain_manifest)
    evidence.append(_evidence_summary("PATH_AND_FORMAT", paths))
    assert source_commit is not None
    source = _source_lineage_gate(contract, source_commit)
    evidence.append(_evidence_summary("SOURCE_LINEAGE", source))
    if meta_chain_manifest is None:
        raise GatePending(
            "REAL_META_V11_CHAIN",
            "MISSING_REAL_META_V11_CHAIN",
            "real thirty-object Meta V11 chain manifest is not available",
        )
    meta = _meta_v11_gate(contract, meta_chain_manifest)
    evidence.append(_evidence_summary("REAL_META_V11_CHAIN", meta))
    target = _target_d_gate(contract)
    evidence.append(_evidence_summary("TARGET_D_OBJECT_CLOSURE", target))
    dependency = _dependency_gate(contract)
    evidence.append(_evidence_summary("PYTHON310_VENDORED_TOMLI_CARGO", dependency))
    history = _history_gate(contract)
    evidence.append(_evidence_summary("HISTORICAL_V2R20_EVIDENCE", history))
    shadow = _load_shadow_tool(contract)
    try:
        shadow_contract = shadow.load_phase3_q05b_shadow_retry_campaign_contract_v2()
    except Exception as error:
        raise GatePending(
            "HISTORICAL_V2R20_EVIDENCE",
            "BOUND_CONTRACT_REPLAY_FAILED",
            f"{type(error).__name__}:{error}",
        )
    if shadow_contract.get("contract_digest") != contract[
        "historical_v2r20_evidence"
    ]["campaign_contract_digest"]:
        raise GatePending(
            "HISTORICAL_V2R20_EVIDENCE",
            "BOUND_CONTRACT_DIGEST_DIFFERS",
            "loaded v2r20 contract is not the bound historical contract",
        )
    cpu = _cpu_gate(shadow, shadow_contract)
    evidence.append(_evidence_summary("CPU_AFFINITY", cpu))
    scheduling = _scheduling_gate(contract, shadow, shadow_contract)
    evidence.append(_evidence_summary("SCHEDULING_ADMISSION", scheduling))
    if require_initial_roots_absent:
        roots = _formal_roots_gate(contract, allow_existing_ledger=False)
    else:
        roots = {
            "status": "DEFERRED_UNTIL_AFTER_POST_GATE_LEDGER_EXISTENCE_CHECK",
            "paths": list(contract["fixed_paths"].values()),
        }
    evidence.append(_evidence_summary("FORMAL_ROOTS_ABSENT", roots))
    bindings = _stable_gate_bindings(
        contract,
        source=source,
        meta=meta,
        shadow_contract=shadow_contract,
    )
    capsule = _core.build_capsule_v2(
        contract,
        source_commit=source["head"],
        source_tree=source["tree"],
        meta_v11_receipt_digest=meta["receipt_digest"],
        gate_receipt_digests=bindings,
    )
    return {
        "schema_version": "hegel-phase3a-q05b-attempt2-target-d-preflight-result/2",
        "status": _core.PREFLIGHT_ADMITTED,
        "capsule": capsule,
        "gate_evidence_in_exact_order": evidence,
        "campaign_execution_count": 0,
        "authorization_consumption_count": 0,
        "ledger_creation_or_write_count": 0,
        "supervisor_popen_count": 0,
    }


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write")
        offset += written


def _publish_json_no_replace(root_fd: int, name: str, value: object) -> bytes:
    raw = _core.canonical_json_bytes_v2(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o400, dir_fd=root_fd)
    try:
        _write_all(descriptor, raw)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(root_fd)
    return raw


def _create_private_root(path: Path, label: str) -> tuple[int, int]:
    parent = path.parent
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    root_fd = -1
    created = False
    try:
        os.mkdir(path.name, 0o700, dir_fd=parent_fd)
        created = True
        os.fsync(parent_fd)
        root_fd = os.open(
            path.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        status = os.fstat(root_fd)
        if stat.S_IMODE(status.st_mode) != 0o700 or status.st_uid != os.geteuid():
            raise OSError("created root identity differs")
    except BaseException as original:
        cleanup_errors: list[str] = []
        if root_fd >= 0:
            try:
                os.close(root_fd)
            except OSError as error:
                cleanup_errors.append(str(error))
            root_fd = -1
        if created:
            try:
                os.rmdir(path.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
                if os.path.lexists(path):
                    cleanup_errors.append("created root remains")
            except OSError as error:
                cleanup_errors.append(str(error))
        try:
            os.close(parent_fd)
        except OSError as error:
            cleanup_errors.append(str(error))
        if cleanup_errors:
            _terminal(
                "FORMAL_ROOT_CREATION_CLEANUP_UNCERTAIN",
                f"{label}: {'; '.join(cleanup_errors)}; original={original}",
            )
        raise
    return parent_fd, root_fd


def _read_ledger_record(root: Path, name: str) -> dict[str, Any] | None:
    path = root / name
    if not os.path.lexists(path):
        return None
    try:
        raw = _read_regular(path, f"ledger {name}")
    except GatePending as error:
        _terminal("LEDGER_RECORD_UNREADABLE", f"{name}: {error.detail}")
    value = _strict_json(raw, f"ledger {name}")
    if type(value) is not dict:
        _terminal("LEDGER_RECORD_INVALID", name)
    unsigned = deepcopy(value)
    observed = unsigned.pop("record_digest", None)
    if observed != _core.domain_digest_v2(_core.DOMAINS["ledger"], unsigned):
        _terminal("LEDGER_RECORD_INVALID", f"{name} digest differs")
    return value


def _replay_existing_ledger(
    contract: Mapping[str, Any], expected_capsule_digest_or_null: str | None
) -> tuple[int, dict[str, Any]]:
    root = Path(str(contract["fixed_paths"]["ledger_root"]))
    consumed = _read_ledger_record(root, "attempt-2.consumed.json")
    terminal = _read_ledger_record(root, "attempt-2.terminal.json")
    if consumed is None:
        _terminal("NONRESUMABLE_LEDGER_PREFIX", "nonempty ledger lacks first durable consumption")
    fields = {
        "schema_version", "contract_digest", "capsule_digest",
        "authorization_digest", "sequence", "record_type", "status",
        "previous_record_digest_or_null", "body", "record_digest",
    }
    if (
        set(consumed) != fields
        or consumed.get("schema_version") != _core.LEDGER_SCHEMA_VERSION
        or consumed.get("contract_digest") != contract["contract_digest"]
        or consumed.get("authorization_digest") != _core.OWNER_AUTHORIZATION_DIGEST
        or consumed.get("sequence") != 1
        or consumed.get("record_type") != "CONSUMED"
        or consumed.get("status") != _core.CONSUMED
        or consumed.get("previous_record_digest_or_null") is not None
        or type(consumed.get("body")) is not dict
        or type(consumed["body"].get("capsule")) is not dict
        or set(consumed["body"])
        != {
            "capsule",
            "attempt_ordinal",
            "authorization_digest",
            "actual_supervisor_popen_limit",
            "retry_or_resume_after_this_record",
        }
        or consumed["body"].get("attempt_ordinal") != 2
        or consumed["body"].get("authorization_digest")
        != _core.OWNER_AUTHORIZATION_DIGEST
        or consumed["body"].get("actual_supervisor_popen_limit") != 1
        or consumed["body"].get("retry_or_resume_after_this_record") is not False
    ):
        _terminal("LEDGER_RECORD_INVALID", "consumed record semantics differ")
    try:
        capsule = _core.validate_capsule_v2(consumed["body"]["capsule"], contract)
    except _core.Attempt2TargetDCampaignValidationError as error:
        _terminal("LEDGER_CAPSULE_DIFFERS", str(error))
    if consumed.get("capsule_digest") != capsule["capsule_digest"]:
        _terminal("LEDGER_CAPSULE_DIFFERS", "consumed top-level capsule differs")
    if (
        expected_capsule_digest_or_null is not None
        and expected_capsule_digest_or_null != capsule["capsule_digest"]
    ):
        _terminal("LEDGER_CAPSULE_DIFFERS", "requested capsule differs from consumed capsule")
    if terminal is not None:
        if (
            set(terminal) != fields
            or terminal.get("schema_version") != _core.LEDGER_SCHEMA_VERSION
            or terminal.get("contract_digest") != contract["contract_digest"]
            or terminal.get("capsule_digest") != capsule["capsule_digest"]
            or terminal.get("authorization_digest") != _core.OWNER_AUTHORIZATION_DIGEST
            or terminal.get("sequence") != 2
            or terminal.get("record_type") != "TERMINAL"
            or terminal.get("previous_record_digest_or_null") != consumed["record_digest"]
            or type(terminal.get("body")) is not dict
        ):
            _terminal("LEDGER_RECORD_INVALID", "terminal record semantics differ")
        body = terminal["body"]
        outcome = body.get("outcome")
        audit = body.get("decision_audit")
        if (
            set(body)
            != {
                "outcome",
                "decision",
                "decision_audit",
                "external_outcome_path",
                "post_popen_retry_or_resume_allowed",
            }
            or body.get("external_outcome_path")
            != os.fspath(
                Path(str(contract["fixed_paths"]["evidence_root"]))
                / "attempt-2.outcome.json"
            )
            or body.get("post_popen_retry_or_resume_allowed") is not False
            or type(outcome) is not dict
            or type(audit) is not dict
        ):
            _terminal("LEDGER_RECORD_INVALID", "terminal outcome or audit differs")
        external_outcome_path = Path(str(body["external_outcome_path"]))
        try:
            external_outcome_raw = _read_regular(
                external_outcome_path, "external attempt-2 outcome"
            )
            external_outcome = _strict_json(
                external_outcome_raw, "external attempt-2 outcome"
            )
        except GatePending as error:
            _terminal("LEDGER_RECORD_INVALID", f"external outcome unavailable: {error}")
        except CampaignTerminal as error:
            _terminal("LEDGER_RECORD_INVALID", f"external outcome invalid: {error}")
        if external_outcome != outcome:
            _terminal("LEDGER_RECORD_INVALID", "external and ledger outcomes differ")
        outcome_fields = {
            "schema_version",
            "status",
            "contract_digest",
            "capsule_digest",
            "attempt_ordinal",
            "actual_supervisor_popen_count",
            "supervisor_return_code",
            "stdout_sha256",
            "stderr_sha256",
            "artifact_sha256_or_null",
            "supervisor_error_or_null",
            "decision",
            "decision_audit",
            "q1_state",
            "retry_or_resume_allowed",
            "outcome_digest",
        }
        audit_fields = {
            "schema_version",
            "decision",
            "predicate_results",
            "return_code",
            "artifact_sha256_or_null",
            "decision_audit_digest",
        }
        unsigned_outcome = deepcopy(outcome)
        observed_outcome_digest = unsigned_outcome.pop("outcome_digest", None)
        unsigned_audit = deepcopy(audit)
        observed_audit_digest = unsigned_audit.pop("decision_audit_digest", None)
        predicates = audit.get("predicate_results")
        if (
            set(outcome) != outcome_fields
            or outcome.get("schema_version") != _core.OUTCOME_SCHEMA_VERSION
            or outcome.get("contract_digest") != contract["contract_digest"]
            or outcome.get("capsule_digest") != capsule["capsule_digest"]
            or outcome.get("attempt_ordinal") != 2
            or outcome.get("actual_supervisor_popen_count") != 1
            or outcome.get("q1_state") != "NOT_RUN"
            or outcome.get("retry_or_resume_allowed") is not False
            or type(outcome.get("supervisor_return_code")) is not int
            or re.fullmatch(r"[0-9a-f]{64}", str(outcome.get("stdout_sha256")))
            is None
            or re.fullmatch(r"[0-9a-f]{64}", str(outcome.get("stderr_sha256")))
            is None
            or (
                outcome.get("artifact_sha256_or_null") is not None
                and re.fullmatch(
                    r"[0-9a-f]{64}", str(outcome.get("artifact_sha256_or_null"))
                )
                is None
            )
            or set(audit) != audit_fields
            or audit.get("schema_version") != _core.DECISION_AUDIT_SCHEMA_VERSION
            or type(predicates) is not list
            or outcome.get("decision_audit") != audit
            or body.get("decision") != audit.get("decision")
            or body.get("decision") != outcome.get("decision")
            or outcome.get("status") != outcome.get("decision")
            or terminal.get("status") != outcome.get("status")
            or audit.get("return_code") != outcome.get("supervisor_return_code")
            or audit.get("artifact_sha256_or_null")
            != outcome.get("artifact_sha256_or_null")
            or observed_audit_digest
            != _core.domain_digest_v2(_core.DOMAINS["decision_audit"], unsigned_audit)
            or observed_outcome_digest
            != _core.domain_digest_v2(_core.DOMAINS["outcome"], unsigned_outcome)
        ):
            _terminal(
                "LEDGER_RECORD_INVALID",
                "terminal decision-audit semantics differ",
            )
        expected_predicate_names = [
            "SUPERVISOR_RETURN_CODE_ZERO",
            "TARGET_D_ARTIFACT_REPLAY_PASS",
            "TARGET_D_ARTIFACT_STATUS_EXACT",
            "FORMAL_ACTUAL_NOT_SHADOW",
        ]
        if (
            len(predicates) != len(expected_predicate_names)
            or any(
                type(row) is not dict
                or set(row) != {"predicate", "passed"}
                or row.get("predicate") != name
                or type(row.get("passed")) is not bool
                for row, name in zip(predicates, expected_predicate_names)
            )
            or predicates[3]["passed"] is not True
        ):
            _terminal("LEDGER_RECORD_INVALID", "decision predicates differ")
        rebuilt_decision = _core.decide_actual_return_v2(
            return_code=outcome["supervisor_return_code"],
            artifact_replay_passed=predicates[1]["passed"],
            artifact_status_or_null=(
                "Q05B_QUALIFICATION_20_OF_20_Q1_NOT_RUN"
                if predicates[2]["passed"]
                else None
            ),
            artifact_sha256_or_null=outcome["artifact_sha256_or_null"],
        )
        if (
            rebuilt_decision.decision != body.get("decision")
            or rebuilt_decision.audit != audit
        ):
            _terminal("LEDGER_RECORD_INVALID", "decision-audit replay differs")
        status = outcome.get("status")
        return (0 if status == _core.QUALIFIED else 1), {
            "schema_version": "hegel-phase3a-q05b-attempt2-target-d-replay/2",
            "status": status,
            "replayed_terminal": True,
            "supervisor_popen_count_this_entry": 0,
            "terminal_record": terminal,
        }
    return 1, {
        "schema_version": "hegel-phase3a-q05b-attempt2-target-d-replay/2",
        "status": _core.OUTCOME_UNKNOWN_TERMINAL,
        "replayed_terminal": True,
        "retry_or_resume_allowed": False,
        "supervisor_popen_count_this_entry": 0,
    }


def _replay_artifact(contract: Mapping[str, Any]) -> tuple[bool, str | None, str | None]:
    path = Path(str(contract["fixed_paths"]["artifact_path"]))
    if not os.path.lexists(path):
        return False, None, None
    try:
        status = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(status.st_mode) or status.st_size > 2_147_483_648:
            return False, None, None
        raw = path.read_bytes()
        module = importlib.import_module("hegel_machine.phase3_q05b_actual_artifact_v2")
        replayed = module.decode_and_replay_actual_artifact_v1(raw)
        module.actual_artifact_summary_v1(replayed)
    except BaseException:
        return False, None, None
    return True, str(replayed.get("status")), sha256(raw).hexdigest()


def _actual_command(contract: Mapping[str, Any], source_commit: str) -> list[str]:
    paths = contract["fixed_paths"]
    target = contract["target_d_v2"]
    return [
        str(contract["execution_policy"]["supervisor_interpreter"]),
        "-I", "-S", "-B",
        os.fspath(PROJECT_ROOT / str(target["supervisor_relative_path"])),
        "--run",
        "--project-root", str(paths["project_root"]),
        "--source-commit", source_commit,
        "--artifact", str(paths["artifact_path"]),
        "--cargo-cache-source", str(paths["cargo_cache_source"]),
        "--trajectory-root", str(paths["trajectory_root"]),
        "--trajectory-trace-id", "q05b-attempt2-target-d-v2-formal-actual",
    ]


def _spawn_actual(command: Sequence[str], **kwargs: Any) -> subprocess.Popen[bytes]:
    return subprocess.Popen(command, **kwargs)


def _cleanup_exact_empty_preconsumption_roots(
    contract: Mapping[str, Any],
    campaign_lock: _CampaignLock,
) -> bool:
    """Recover only the exact zero-consumption residue left before record 1."""

    _contract_campaign_lock_identity(contract, campaign_lock)
    paths = contract["fixed_paths"]
    ledger = Path(str(paths["ledger_root"]))
    evidence = Path(str(paths["evidence_root"]))
    ledger_present = os.path.lexists(ledger)
    evidence_present = os.path.lexists(evidence)
    if not ledger_present and not evidence_present:
        return False
    for key in ("artifact_path", "trajectory_root", "scheduling_probe_root"):
        if os.path.lexists(Path(str(paths[key]))):
            return False
    try:
        if ledger_present:
            ledger_status = ledger.lstat()
            if (
                ledger.is_symlink()
                or not stat.S_ISDIR(ledger_status.st_mode)
                or list(ledger.iterdir())
            ):
                return False
        if evidence_present:
            evidence_status = evidence.lstat()
            if evidence.is_symlink() or not stat.S_ISDIR(evidence_status.st_mode):
                return False
            names = sorted(path.name for path in evidence.iterdir())
            if not set(names).issubset({"supervisor.stderr", "supervisor.stdout"}):
                return False
            for path in evidence.iterdir():
                status = path.lstat()
                if path.is_symlink() or not stat.S_ISREG(status.st_mode) or status.st_size != 0:
                    return False
        if evidence_present:
            evidence_parent_fd = os.open(
                evidence.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            )
            evidence_fd = os.open(
                evidence, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            )
            try:
                for name in os.listdir(evidence_fd):
                    os.unlink(name, dir_fd=evidence_fd)
                os.fsync(evidence_fd)
                os.rmdir(evidence.name, dir_fd=evidence_parent_fd)
                os.fsync(evidence_parent_fd)
            finally:
                os.close(evidence_fd)
                os.close(evidence_parent_fd)
        if ledger_present:
            ledger_parent_fd = os.open(
                ledger.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            )
            ledger_fd = os.open(
                ledger, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            )
            try:
                if os.listdir(ledger_fd):
                    return False
                os.rmdir(ledger.name, dir_fd=ledger_parent_fd)
                os.fsync(ledger_parent_fd)
            finally:
                os.close(ledger_fd)
                os.close(ledger_parent_fd)
    except OSError as error:
        _terminal("PRECONSUMPTION_ROOT_CLEANUP_UNCERTAIN", str(error))
    if os.path.lexists(ledger) or os.path.lexists(evidence):
        _terminal("PRECONSUMPTION_ROOT_CLEANUP_UNCERTAIN", "root remains after fsync")
    _contract_campaign_lock_identity(contract, campaign_lock)
    return True


def _rollback_preconsumption_setup(
    *,
    contract: Mapping[str, Any],
    campaign_lock: _CampaignLock,
    evidence_root: Path,
    evidence_parent_fd: int,
    evidence_fd: int,
    stdout_descriptor: int,
    stderr_descriptor: int,
    ledger_root: Path,
    ledger_parent_fd: int,
    ledger_fd: int,
) -> None:
    _contract_campaign_lock_identity(contract, campaign_lock)
    errors: list[str] = []
    for descriptor in (stdout_descriptor, stderr_descriptor):
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as error:
                errors.append(str(error))
    if ledger_fd >= 0:
        try:
            if os.listdir(ledger_fd):
                errors.append("ledger root is not empty before consumption")
            os.close(ledger_fd)
        except OSError as error:
            errors.append(str(error))
    if ledger_parent_fd >= 0:
        try:
            if not errors and os.path.lexists(ledger_root):
                os.rmdir(ledger_root.name, dir_fd=ledger_parent_fd)
                os.fsync(ledger_parent_fd)
            os.close(ledger_parent_fd)
        except OSError as error:
            errors.append(str(error))
    if evidence_fd >= 0:
        try:
            for name in os.listdir(evidence_fd):
                os.unlink(name, dir_fd=evidence_fd)
            os.fsync(evidence_fd)
            os.close(evidence_fd)
        except OSError as error:
            errors.append(str(error))
    if evidence_parent_fd >= 0:
        try:
            if os.path.lexists(evidence_root):
                os.rmdir(evidence_root.name, dir_fd=evidence_parent_fd)
                os.fsync(evidence_parent_fd)
            os.close(evidence_parent_fd)
        except OSError as error:
            errors.append(str(error))
    if errors or os.path.lexists(ledger_root) or os.path.lexists(evidence_root):
        _terminal(
            "PRECONSUMPTION_ROOT_CLEANUP_UNCERTAIN",
            "; ".join(errors) or "root remains after cleanup",
        )
    _contract_campaign_lock_identity(contract, campaign_lock)


def _execute_once_locked_v2(
    contract: Mapping[str, Any],
    *,
    campaign_lock: _CampaignLock,
    source_commit: str | None,
    meta_chain_manifest: Path | None,
    expected_capsule_digest: str | None,
) -> tuple[int, dict[str, Any]]:
    _contract_campaign_lock_identity(contract, campaign_lock)
    ledger_root = Path(str(contract["fixed_paths"]["ledger_root"]))
    evidence_root = Path(str(contract["fixed_paths"]["evidence_root"]))
    if os.path.lexists(ledger_root) or os.path.lexists(evidence_root):
        if _cleanup_exact_empty_preconsumption_roots(contract, campaign_lock):
            raise GatePending(
                "PATH_AND_FORMAT",
                "EMPTY_PRECONSUMPTION_ROOTS_CLEANED",
                "empty zero-consumption roots were removed and fsynced; replay the same RUN",
            )
        if os.path.lexists(ledger_root):
            return _replay_existing_ledger(contract, expected_capsule_digest)
    preflight = derive_preflight_v2(
        contract,
        source_commit=source_commit,
        meta_chain_manifest=meta_chain_manifest,
        require_initial_roots_absent=False,
    )
    capsule = preflight["capsule"]
    if (
        type(expected_capsule_digest) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", expected_capsule_digest) is None
    ):
        raise GatePending(
            "PATH_AND_FORMAT",
            "CAPSULE_DIGEST_REQUIRED",
            "--run requires the exact digest returned by --preflight",
        )
    if expected_capsule_digest != capsule["capsule_digest"]:
        raise GatePending(
            "PATH_AND_FORMAT",
            "CAPSULE_DIGEST_REPLAY_DIFFERS",
            f"expected={expected_capsule_digest}; replayed={capsule['capsule_digest']}",
        )
    _formal_roots_gate(contract, allow_existing_ledger=False)
    _contract_campaign_lock_identity(contract, campaign_lock)

    evidence_parent_fd = evidence_fd = ledger_parent_fd = ledger_fd = -1
    stdout_descriptor = stderr_descriptor = -1
    try:
        try:
            evidence_parent_fd, evidence_fd = _create_private_root(evidence_root, "evidence root")
            stdout_descriptor = os.open(
                "supervisor.stdout",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=evidence_fd,
            )
            stderr_descriptor = os.open(
                "supervisor.stderr",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=evidence_fd,
            )
            os.fsync(evidence_fd)
            ledger_parent_fd, ledger_fd = _create_private_root(ledger_root, "ledger root")
            fcntl.flock(ledger_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            _rollback_preconsumption_setup(
                contract=contract,
                campaign_lock=campaign_lock,
                evidence_root=evidence_root,
                evidence_parent_fd=evidence_parent_fd,
                evidence_fd=evidence_fd,
                stdout_descriptor=stdout_descriptor,
                stderr_descriptor=stderr_descriptor,
                ledger_root=ledger_root,
                ledger_parent_fd=ledger_parent_fd,
                ledger_fd=ledger_fd,
            )
            evidence_parent_fd = evidence_fd = ledger_parent_fd = ledger_fd = -1
            stdout_descriptor = stderr_descriptor = -1
            raise GatePending("PATH_AND_FORMAT", "FORMAL_ROOT_OR_LOCK_UNAVAILABLE", str(error))
        consumed = _core.build_ledger_record_v2(
            contract,
            capsule,
            sequence=1,
            record_type="CONSUMED",
            status=_core.CONSUMED,
            previous_record_digest_or_null=None,
            body={
                "capsule": capsule,
                "attempt_ordinal": 2,
                "authorization_digest": _core.OWNER_AUTHORIZATION_DIGEST,
                "actual_supervisor_popen_limit": 1,
                "retry_or_resume_after_this_record": False,
            },
        )
        try:
            _publish_json_no_replace(ledger_fd, "attempt-2.consumed.json", consumed)
        except OSError as error:
            consumed_path = ledger_root / "attempt-2.consumed.json"
            if not os.path.lexists(consumed_path):
                _rollback_preconsumption_setup(
                    contract=contract,
                    campaign_lock=campaign_lock,
                    evidence_root=evidence_root,
                    evidence_parent_fd=evidence_parent_fd,
                    evidence_fd=evidence_fd,
                    stdout_descriptor=stdout_descriptor,
                    stderr_descriptor=stderr_descriptor,
                    ledger_root=ledger_root,
                    ledger_parent_fd=ledger_parent_fd,
                    ledger_fd=ledger_fd,
                )
                evidence_parent_fd = evidence_fd = ledger_parent_fd = ledger_fd = -1
                stdout_descriptor = stderr_descriptor = -1
                raise GatePending(
                    "PATH_AND_FORMAT", "CONSUMPTION_PUBLICATION_NOT_STARTED", str(error)
                )
            try:
                replayed_consumed = _read_ledger_record(
                    ledger_root, "attempt-2.consumed.json"
                )
            except CampaignTerminal as replay_error:
                _terminal(
                    "CONSUMPTION_PUBLICATION_UNCERTAIN",
                    f"consumption residue cannot be replayed exactly: {replay_error}",
                )
            if replayed_consumed != consumed:
                _terminal(
                    "CONSUMPTION_PUBLICATION_UNCERTAIN",
                    f"consumption publication failed with non-exact residue: {error}",
                )

        _contract_campaign_lock_identity(contract, campaign_lock)
        command = _actual_command(contract, str(source_commit))
        popen_error: BaseException | None = None
        return_code = 127
        child: subprocess.Popen[bytes] | None = None
        try:
            child = _spawn_actual(
                command,
                cwd="/",
                stdin=subprocess.DEVNULL,
                stdout=stdout_descriptor,
                stderr=stderr_descriptor,
                env={
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PATH": "/usr/bin:/bin",
                    "PYTHONHASHSEED": "0",
                },
                close_fds=True,
                start_new_session=True,
            )
            try:
                return_code = child.wait(timeout=float(contract["execution_policy"]["supervisor_timeout_seconds"]))
            except subprocess.TimeoutExpired as error:
                popen_error = error
                try:
                    os.killpg(child.pid, signal.SIGINT)
                    return_code = child.wait(timeout=120)
                except BaseException:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return_code = child.wait(timeout=30)
        except BaseException as error:
            popen_error = error
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        finally:
            os.fsync(stdout_descriptor)
            os.fchmod(stdout_descriptor, 0o444)
            os.fsync(stdout_descriptor)
            os.close(stdout_descriptor)
            stdout_descriptor = -1
            os.fsync(stderr_descriptor)
            os.fchmod(stderr_descriptor, 0o444)
            os.fsync(stderr_descriptor)
            os.close(stderr_descriptor)
            stderr_descriptor = -1
            os.fsync(evidence_fd)

        replay_passed, artifact_status, artifact_sha = _replay_artifact(contract)
        decision = _core.decide_actual_return_v2(
            return_code=return_code,
            artifact_replay_passed=replay_passed,
            artifact_status_or_null=artifact_status,
            artifact_sha256_or_null=artifact_sha,
        )
        try:
            stdout_raw = _read_regular(
                evidence_root / "supervisor.stdout", "formal supervisor stdout"
            )
            stderr_raw = _read_regular(
                evidence_root / "supervisor.stderr", "formal supervisor stderr"
            )
        except GatePending as error:
            _terminal(
                "POSTCONSUMPTION_EVIDENCE_UNREADABLE_TERMINAL_NO_RESUME",
                error.detail,
            )
        outcome = _core.build_outcome_v2(
            contract,
            capsule,
            decision=decision,
            supervisor_return_code=return_code,
            stdout_sha256=sha256(stdout_raw).hexdigest(),
            stderr_sha256=sha256(stderr_raw).hexdigest(),
            artifact_sha256_or_null=artifact_sha,
            supervisor_error_or_null=(
                None
                if popen_error is None
                else f"{type(popen_error).__name__}:{popen_error}"[:4096]
            ),
        )
        _publish_json_no_replace(evidence_fd, "attempt-2.outcome.json", outcome)
        terminal = _core.build_ledger_record_v2(
            contract,
            capsule,
            sequence=2,
            record_type="TERMINAL",
            status=decision.decision,
            previous_record_digest_or_null=consumed["record_digest"],
            body={
                "outcome": outcome,
                "decision": decision.decision,
                "decision_audit": decision.audit,
                "external_outcome_path": os.fspath(evidence_root / "attempt-2.outcome.json"),
                "post_popen_retry_or_resume_allowed": False,
            },
        )
        _publish_json_no_replace(ledger_fd, "attempt-2.terminal.json", terminal)
        _contract_campaign_lock_identity(contract, campaign_lock)
        return (0 if decision.decision == _core.QUALIFIED else 1), {
            "schema_version": "hegel-phase3a-q05b-attempt2-target-d-command-result/2",
            "status": decision.decision,
            "capsule_digest": capsule["capsule_digest"],
            "actual_supervisor_popen_count": 1,
            "authorization_consumption_count": 1,
            "retry_or_resume_allowed": False,
            "outcome": outcome,
            "terminal_record_digest": terminal["record_digest"],
        }
    finally:
        for descriptor in (stdout_descriptor, stderr_descriptor):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        if ledger_fd >= 0:
            try:
                fcntl.flock(ledger_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(ledger_fd)
        if ledger_parent_fd >= 0:
            os.close(ledger_parent_fd)
        if evidence_fd >= 0:
            os.close(evidence_fd)
        if evidence_parent_fd >= 0:
            os.close(evidence_parent_fd)


def execute_once_v2(
    contract: Mapping[str, Any],
    *,
    source_commit: str | None,
    meta_chain_manifest: Path | None,
    expected_capsule_digest: str | None,
) -> tuple[int, dict[str, Any]]:
    campaign_lock = _acquire_campaign_lock(contract)
    try:
        return _execute_once_locked_v2(
            contract,
            campaign_lock=campaign_lock,
            source_commit=source_commit,
            meta_chain_manifest=meta_chain_manifest,
            expected_capsule_digest=expected_capsule_digest,
        )
    finally:
        _release_campaign_lock(campaign_lock)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        _terminal("INVALID_COMMAND_LINE", message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--source-commit")
    parser.add_argument("--meta-chain-manifest", type=Path)
    parser.add_argument("--capsule-digest")
    return parser


def _emit(value: object, *, stream: object = sys.stdout) -> None:
    stream.write(_core.canonical_json_bytes_v2(value).decode("ascii"))
    stream.flush()


def main(arguments: Sequence[str] | None = None) -> int:
    contract: dict[str, Any] | None = None
    options: argparse.Namespace | None = None
    try:
        options = _parser().parse_args(arguments)
        contract = load_contract_v2()
        if options.preflight:
            if options.capsule_digest is not None:
                raise GatePending("PATH_AND_FORMAT", "PREFLIGHT_CAPSULE_ARGUMENT_FORBIDDEN", "omit --capsule-digest")
            result = derive_preflight_v2(
                contract,
                source_commit=options.source_commit,
                meta_chain_manifest=options.meta_chain_manifest,
                require_initial_roots_absent=True,
            )
            _emit(result)
            return 0
        code, result = execute_once_v2(
            contract,
            source_commit=options.source_commit,
            meta_chain_manifest=options.meta_chain_manifest,
            expected_capsule_digest=options.capsule_digest,
        )
        _emit(result)
        return code
    except GatePending as error:
        if contract is None:
            _emit(
                {
                    "schema_version": "hegel-phase3a-q05b-attempt2-target-d-error/2",
                    "status": _core.PENDING_NOT_CONSUMED,
                    "phase": error.phase,
                    "condition": error.condition,
                    "detail": error.detail,
                }
            )
            return PENDING_EXIT_CODE
        pending = _core.build_pending_v2(
            contract,
            phase=error.phase,
            condition=error.condition,
            detail=error.detail,
            source_commit_or_null=(None if options is None else options.source_commit),
            meta_chain_manifest_or_null=(
                None
                if options is None or options.meta_chain_manifest is None
                else os.fspath(options.meta_chain_manifest)
            ),
        )
        _emit(pending)
        return PENDING_EXIT_CODE
    except (CampaignTerminal, _core.Attempt2TargetDCampaignValidationError) as error:
        code = getattr(error, "code", "ATTEMPT2_TARGET_D_V2_TERMINAL")
        detail = getattr(error, "detail", str(error))
        _emit(
            {
                "schema_version": "hegel-phase3a-q05b-attempt2-target-d-error/2",
                "status": str(code),
                "detail": str(detail)[:4096],
                "retry_or_resume_allowed": False,
            },
            stream=sys.stderr,
        )
        return 1
    except OSError as error:
        _emit(
            {
                "schema_version": "hegel-phase3a-q05b-attempt2-target-d-error/2",
                "status": "ATTEMPT2_TARGET_D_V2_IO_UNCERTAIN_TERMINAL_NO_RESUME",
                "detail": f"{type(error).__name__}:{error}"[:4096],
                "retry_or_resume_allowed": False,
            },
            stream=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
