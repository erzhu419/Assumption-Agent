"""One-shot, shadow-only Q0.5b mechanics wrapper.

The public preflight is read-only.  The run entry remains unusable until this
file, its pure core, tests, and contract are source-bound and the project owner
publishes a separate authorization for the exact derived capsule digest.

This module never imports or calls the attempt-2 bridge.  Its only eventual
Hegel execution surface is the unchanged, clone-local direct Q0.5b supervisor
CLI frozen by the contract.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import errno
import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
import types
from typing import Any, Callable, Final, Mapping, NoReturn, Sequence


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
WORKTREE_ROOT: Final = PROJECT_ROOT.parent
CONTRACT_PATH: Final = (
    PROJECT_ROOT / "config/phase3_q05b_shadow_nonadoption_trial_v1.json"
)
CORE_PATH: Final = (
    PROJECT_ROOT / "src/hegel_machine/phase3_q05b_shadow_nonadoption_trial_v1.py"
)
TESTS_PATH: Final = (
    PROJECT_ROOT / "tests/test_phase3_q05b_shadow_nonadoption_trial_v1.py"
)
TOOL_PATH: Final = Path(__file__).resolve()
_WRAPPER_PYTHON_PATH: Final = "/usr/bin/python3.10"
_WRAPPER_TOOL_PATH: Final = (
    "/home/erzhu419/.local/state/hegel-machine/q05b-shadow-nonadoption-dev-worktree/"
    "Hegel Machine/tools/phase3_q05b_shadow_nonadoption_trial_v1.py"
)
_WRAPPER_SYS_PATH: Final = (
    "/usr/lib/python310.zip",
    "/usr/lib/python3.10",
    "/usr/lib/python3.10/lib-dynload",
)

_CORE_MODULE: Any | None = None
_CORE_RAW_DIGEST: str | None = None

__all__ = (
    "ShadowNonadoptionTrialError",
    "load_phase3_q05b_shadow_nonadoption_trial_contract_v1",
    "preflight_phase3_q05b_shadow_nonadoption_trial_v1",
    "run_phase3_q05b_shadow_nonadoption_trial_v1",
    "main",
)


# SOURCE_BINDING_LITERAL_START
SOURCE_BINDING_LITERAL = {"contract_digest":"sha256:c5691cd995939d31c1274620349294587c3a562598bba363b947d29cbdf29754","contract_raw_sha256":"e37902c8dc1d90aae34a3c741b7d0d314a18cea40f074f440ebbc8d31b31cf78","normalized_source_digest":"sha256:d4d523ba30ce940e3fb7374041e2a412543b0b274560f411725891d317629bef"}
# SOURCE_BINDING_LITERAL_END


_LITERAL_START: Final = b"# SOURCE_BINDING_LITERAL_START\n"
_LITERAL_END: Final = b"# SOURCE_BINDING_LITERAL_END\n"
_LITERAL_SENTINEL: Final = (
    b'SOURCE_BINDING_LITERAL = {"contract_digest":"sha256:MASKED",'
    b'"contract_raw_sha256":"MASKED","normalized_source_digest":'
    b'"sha256:MASKED"}\n'
)
_RAW_SHA256: Final = re.compile(r"[0-9a-f]{64}\Z")
_TAGGED_SHA256: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT: Final = re.compile(r"[0-9a-f]{40}\Z")
_CONTAINER_ID: Final = re.compile(r"[0-9a-f]{64}\Z")
_NAMESPACE: Final = re.compile(r"[0-9a-f]{64}\Z")
_LEDGER_RECORD_MAX_BYTES: Final = 1 << 20
_FINAL_SOURCE_STATUS: Final = (
    "SOURCE_CORE_TOOL_TESTS_PRESENT_AND_BOUND_EXECUTION_STILL_REQUIRES_OWNER_AUTHORIZATION"
)


class ShadowNonadoptionTrialError(RuntimeError):
    """Fail-closed wrapper error, optionally carrying a typed receipt."""

    def __init__(
        self,
        detail: str,
        *,
        receipt: Mapping[str, Any] | None = None,
        terminal_after_intent: bool = False,
        child_reaped_verified: bool | None = None,
        wire_state: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.receipt = deepcopy(dict(receipt)) if receipt is not None else None
        self.terminal_after_intent = terminal_after_intent
        self.child_reaped_verified = child_reaped_verified
        self.wire_state = deepcopy(dict(wire_state)) if wire_state is not None else None


def _fail(
    detail: str,
    *,
    receipt: Mapping[str, Any] | None = None,
    terminal_after_intent: bool = False,
    child_reaped_verified: bool | None = None,
    wire_state: Mapping[str, Any] | None = None,
) -> NoReturn:
    raise ShadowNonadoptionTrialError(
        detail,
        receipt=receipt,
        terminal_after_intent=terminal_after_intent,
        child_reaped_verified=child_reaped_verified,
        wire_state=wire_state,
    )


def _audit_event(_name: str) -> None:
    """Test-only observation seam; it has no production semantics."""


def _verify_wrapper_entrypoint() -> None:
    """Accept only the source-frozen isolated interpreter entrypoint."""

    flags = sys.flags
    if (
        sys.executable != _WRAPPER_PYTHON_PATH
        or os.fspath(TOOL_PATH) != _WRAPPER_TOOL_PATH
        or not sys.argv
        or sys.argv[0] != _WRAPPER_TOOL_PATH
        or tuple(sys.path) != _WRAPPER_SYS_PATH
        or flags.isolated != 1
        or flags.no_site != 1
        or flags.no_user_site != 1
        or flags.ignore_environment != 1
        or flags.dont_write_bytecode != 1
        or flags.utf8_mode != 1
        or flags.optimize != 0
        or flags.inspect != 0
        or flags.interactive != 0
        or flags.dev_mode
        or sys._xoptions != {"utf8": True}
        or sys.warnoptions
    ):
        _fail("wrapper interpreter entrypoint differs")


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_json_tree_local(value: object, label: str = "value") -> None:
    value_type = type(value)
    if value is None or value_type in (bool, int, str):
        return
    if value_type is float:
        if not math.isfinite(value):
            _fail(f"{label} contains a non-finite float")
        return
    if value_type is list:
        for index, item in enumerate(value):
            _validate_json_tree_local(item, f"{label}[{index}]")
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                _fail(f"{label} contains a non-string key")
            _validate_json_tree_local(item, f"{label}.{key}")
        return
    _fail(f"{label} contains a non-exact JSON runtime type")


def _canonical_json_bytes(value: object) -> bytes:
    _validate_json_tree_local(value)
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


def _parse_canonical_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("ascii")
        value = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=lambda token: (_fail(f"{label} contains {token}")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        _fail(f"{label} JSON differs: {error}")
    if type(value) is not dict:
        _fail(f"{label} is not an exact mapping")
    if _canonical_json_bytes(value) != raw:
        _fail(f"{label} raw bytes are not canonical single-LF JSON")
    return value


def _read_regular_nofollow(
    path: Path,
    label: str,
    maximum: int | None = None,
    *,
    with_stat: bool = False,
    reject_xattrs: bool = False,
) -> bytes | tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(os.fspath(path), flags)
    except OSError as error:
        _fail(f"{label} cannot be opened: {error}")
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink < 1:
            _fail(f"{label} is not one linked regular file")
        if maximum is not None and before.st_size > maximum:
            _fail(f"{label} exceeds its byte bound")
        if reject_xattrs:
            try:
                before_xattrs = os.listxattr(fd)
            except OSError as error:
                _fail(f"{label} extended attributes cannot be replayed: {error}")
            if before_xattrs:
                _fail(f"{label} contains an extended attribute")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if maximum is not None and total > maximum:
                _fail(f"{label} exceeds its byte bound")
            chunks.append(chunk)
        after = os.fstat(fd)
        if reject_xattrs:
            try:
                after_xattrs = os.listxattr(fd)
            except OSError as error:
                _fail(f"{label} extended attributes cannot be replayed: {error}")
            if after_xattrs:
                _fail(f"{label} contains an extended attribute")
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _fail(f"{label} changed during replay")
        try:
            path_after = os.stat(path, follow_symlinks=False)
        except OSError as error:
            _fail(f"{label} pathname cannot be replayed: {error}")
        if identity != (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_mode,
            path_after.st_uid,
            path_after.st_nlink,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        ):
            _fail(f"{label} pathname identity changed during replay")
        raw = b"".join(chunks)
        return (raw, before) if with_stat else raw
    finally:
        os.close(fd)


def _normalize_tool_bytes(raw: bytes) -> bytes:
    if raw.count(_LITERAL_START) != 1 or raw.count(_LITERAL_END) != 1:
        _fail("source binding markers differ")
    start = raw.index(_LITERAL_START) + len(_LITERAL_START)
    end = raw.index(_LITERAL_END)
    if start >= end:
        _fail("source binding markers are out of order")
    body = raw[start:end]
    if body.count(b"\n") != 1 or not body.endswith(b"\n"):
        _fail("source binding literal is not exactly one line")
    try:
        tree = ast.parse(body.decode("ascii"), mode="exec")
    except (UnicodeDecodeError, SyntaxError) as error:
        _fail(f"source binding literal syntax differs: {error}")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        _fail("source binding literal contains an extra statement")
    assignment = tree.body[0]
    if (
        len(assignment.targets) != 1
        or not isinstance(assignment.targets[0], ast.Name)
        or assignment.targets[0].id != "SOURCE_BINDING_LITERAL"
        or not isinstance(assignment.value, ast.Dict)
    ):
        _fail("source binding literal assignment differs")
    expected_keys = (
        "contract_digest",
        "contract_raw_sha256",
        "normalized_source_digest",
    )
    if (
        len(assignment.value.keys) != len(expected_keys)
        or len(assignment.value.values) != len(expected_keys)
        or any(
            not isinstance(key, ast.Constant)
            or type(key.value) is not str
            or key.value != expected
            for key, expected in zip(
                assignment.value.keys, expected_keys, strict=True
            )
        )
        or any(
            not isinstance(value, ast.Constant) or type(value.value) is not str
            for value in assignment.value.values
        )
    ):
        _fail("source binding literal is not three ordered string constants")
    parsed_literal = {
        key.value: value.value
        for key, value in zip(
            assignment.value.keys, assignment.value.values, strict=True
        )
    }
    if parsed_literal != SOURCE_BINDING_LITERAL:
        _fail("source binding literal AST and runtime values differ")
    return raw[:start] + _LITERAL_SENTINEL + raw[end:]


def _public_ast_surface(path: Path, raw: bytes) -> dict[str, Any]:
    text = raw.decode("utf-8")
    attempt2_module_token = "phase3_q05b_" + "attempt2"
    direct_python_call_token = "run_actual_" + "v1("
    tree = ast.parse(text, filename=os.fspath(path))
    imports: list[str] = []
    public_functions: list[dict[str, Any]] = []
    public_classes: list[str] = []
    subprocess_calls: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            public_functions.append(
                {
                    "name": node.name,
                    "parameters": [item.arg for item in node.args.args],
                    "has_vararg": node.args.vararg is not None,
                    "has_kwarg": node.args.kwarg is not None,
                }
            )
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            public_classes.append(node.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess":
                subprocess_calls.append(node.func.attr)
    return {
        "path": path.relative_to(WORKTREE_ROOT).as_posix(),
        "imports": sorted(imports),
        "public_functions": sorted(public_functions, key=lambda row: row["name"]),
        "public_classes": sorted(public_classes),
        "subprocess_calls": sorted(subprocess_calls),
        "forbidden_attempt2_token_present": attempt2_module_token in text,
        "forbidden_direct_python_call_present": direct_python_call_token in text,
        "network_module_present": any(
            item.split(".", 1)[0] in {"socket", "urllib", "http", "requests"}
            for item in imports
        ),
    }


def _source_static_surface(core_raw: bytes, tool_raw: bytes) -> dict[str, Any]:
    return {
        "schema_version": "hegel-phase3a-q05b-shadow-nonadoption-source-surface/1",
        "ordered_sources": [
            _public_ast_surface(CORE_PATH, core_raw),
            _public_ast_surface(TOOL_PATH, tool_raw),
        ],
    }


def _domain_digest(domain: str, value: object) -> str:
    if type(domain) is not str or not domain.endswith("\x00"):
        _fail("hash domain differs")
    return "sha256:" + sha256(
        domain.encode("ascii") + _canonical_json_bytes(value)[:-1]
    ).hexdigest()


def _verify_live_source_binding(contract: Mapping[str, Any], contract_raw: bytes) -> None:
    binding = contract["source_binding"]
    if binding["status"] != _FINAL_SOURCE_STATUS:
        _fail("source binding is not final")
    core_raw = _read_regular_nofollow(CORE_PATH, "pure core source")
    tests_raw = _read_regular_nofollow(TESTS_PATH, "focused tests")
    tool_raw = _read_regular_nofollow(TOOL_PATH, "I/O tool source")
    normalized = "sha256:" + sha256(_normalize_tool_bytes(tool_raw)).hexdigest()
    static_digest = sha256(
        _canonical_json_bytes(_source_static_surface(core_raw, tool_raw))
    ).hexdigest()
    if binding["source_raw_sha256"] != sha256(core_raw).hexdigest():
        _fail("pure core raw bytes differ from source binding")
    if binding["tests_raw_sha256"] != sha256(tests_raw).hexdigest():
        _fail("focused tests raw bytes differ from source binding")
    if binding["normalized_source_digest"] != normalized:
        _fail("normalized I/O tool bytes differ from source binding")
    if binding["source_static_surface_sha256"] != static_digest:
        _fail("source static surface differs from source binding")
    expected_literal = {
        "contract_digest": contract["contract_digest"],
        "contract_raw_sha256": sha256(contract_raw).hexdigest(),
        "normalized_source_digest": normalized,
    }
    if SOURCE_BINDING_LITERAL != expected_literal:
        _fail("source binding literal differs")


def _bootstrap_and_load_bound_core(
    parsed: Mapping[str, Any], contract_raw: bytes
) -> Any:
    global _CORE_MODULE, _CORE_RAW_DIGEST
    if parsed.get("schema_version") != "hegel-phase3a-q05b-shadow-nonadoption-trial-contract/1":
        _fail("bootstrap contract schema differs")
    if parsed.get("contract_id") != "hegel-phase3a-q05b-shadow-nonadoption-trial-v1":
        _fail("bootstrap contract id differs")
    canonical = parsed.get("canonicalization")
    domains = parsed.get("hash_domains")
    binding = parsed.get("source_binding")
    if type(canonical) is not dict or type(domains) is not dict or type(binding) is not dict:
        _fail("bootstrap contract surface differs")
    top_fields = canonical.get("contract_exact_top_level_fields")
    if (
        type(top_fields) is not list
        or any(type(item) is not str for item in top_fields)
        or set(parsed) != set(top_fields)
        or len(parsed) != len(top_fields)
    ):
        _fail("bootstrap contract fields differ")
    body = deepcopy(dict(parsed))
    observed_contract_digest = body.pop("contract_digest", None)
    expected_contract_digest = _domain_digest(domains.get("contract"), body)
    if observed_contract_digest != expected_contract_digest:
        _fail("bootstrap contract digest differs")
    source_fields = canonical.get("source_binding_exact_fields")
    if (
        type(source_fields) is not list
        or any(type(item) is not str for item in source_fields)
        or set(binding) != set(source_fields)
        or len(binding) != len(source_fields)
        or binding.get("status") != _FINAL_SOURCE_STATUS
    ):
        _fail("bootstrap source binding differs")
    core_digest = binding.get("source_raw_sha256")
    if type(core_digest) is not str or _RAW_SHA256.fullmatch(core_digest) is None:
        _fail("bootstrap core digest differs")
    literal_prefix = {
        "contract_digest": observed_contract_digest,
        "contract_raw_sha256": sha256(contract_raw).hexdigest(),
        "normalized_source_digest": binding.get("normalized_source_digest"),
    }
    if any(SOURCE_BINDING_LITERAL.get(key) != value for key, value in literal_prefix.items()):
        _fail("bootstrap source binding literal differs")
    core_raw = _read_regular_nofollow(CORE_PATH, "bootstrap pure core source")
    if sha256(core_raw).hexdigest() != core_digest:
        _fail("bootstrap pure core bytes differ")
    if _CORE_MODULE is not None:
        if _CORE_RAW_DIGEST != core_digest:
            _fail("loaded pure core digest differs from current contract")
        return _CORE_MODULE
    try:
        code = compile(
            core_raw.decode("utf-8"),
            os.fspath(CORE_PATH),
            "exec",
            dont_inherit=True,
        )
    except (UnicodeDecodeError, SyntaxError) as error:
        _fail(f"bound pure core cannot be compiled: {error}")
    module = types.ModuleType("_hegel_q05b_shadow_nonadoption_trial_core_v1")
    module.__file__ = os.fspath(CORE_PATH)
    module.__package__ = None
    exec(code, module.__dict__)
    _CORE_MODULE = module
    _CORE_RAW_DIGEST = core_digest
    return module


def _core() -> Any:
    if _CORE_MODULE is None:
        _fail("bound pure core is not loaded")
    return _CORE_MODULE


def load_phase3_q05b_shadow_nonadoption_trial_contract_v1() -> dict[str, Any]:
    """Load the exact canonical, source-bound immutable contract."""

    raw = _read_regular_nofollow(CONTRACT_PATH, "shadow trial contract", 1 << 20)
    parsed = _parse_canonical_json(raw, "shadow trial contract")
    core = _bootstrap_and_load_bound_core(parsed, raw)
    contract = core.validate_shadow_nonadoption_trial_contract_v1(parsed)
    _verify_live_source_binding(contract, raw)
    return contract


def _executable_identity(path_text: str, label: str) -> dict[str, Any]:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path_text, flags)
    except OSError as error:
        _fail(f"{label} executable cannot be opened: {error}")
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink < 1 or before.st_size < 1:
            _fail(f"{label} executable identity differs")
        digest = sha256()
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _fail(f"{label} executable changed during replay")
        try:
            path_after = os.stat(path_text, follow_symlinks=False)
        except OSError as error:
            _fail(f"{label} executable pathname cannot be replayed: {error}")
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_mode,
            path_after.st_uid,
            path_after.st_nlink,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        ):
            _fail(f"{label} executable pathname identity changed")
        return {
            "path": path_text,
            "raw_sha256": digest.hexdigest(),
            "mode": before.st_mode,
            "uid": before.st_uid,
            "nlink": before.st_nlink,
            "size": before.st_size,
            "dev": before.st_dev,
            "ino": before.st_ino,
        }
    finally:
        os.close(fd)


def _git_readonly(
    contract: Mapping[str, Any],
    arguments: Sequence[str],
    label: str,
    expected_git_identity: Mapping[str, Any],
) -> bytes:
    source = contract["isolated_source_policy"]
    if _executable_identity(source["git_executable_path"], "Git") != expected_git_identity:
        _fail(f"{label} Git executable identity differs")
    argv = [
        source["git_executable_path"],
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.hooksPath=/dev/null",
        *arguments,
    ]
    result = subprocess.run(
        argv,
        cwd="/",
        env=deepcopy(source["git_runtime_environment"]),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        check=False,
    )
    if result.returncode != 0 or result.stderr:
        _fail(f"{label} failed")
    if _executable_identity(source["git_executable_path"], "Git") != expected_git_identity:
        _fail(f"{label} Git executable changed during replay")
    return result.stdout


def _fd_mount_id(fd: int, label: str) -> int:
    raw = _read_regular_nofollow(
        Path(f"/proc/self/fdinfo/{fd}"), f"{label} fdinfo", 1 << 16
    )
    rows = [line for line in raw.splitlines() if line.startswith(b"mnt_id:\t")]
    if len(rows) != 1:
        _fail(f"{label} mount identity differs")
    try:
        value = int(rows[0].split(b"\t", 1)[1])
    except ValueError:
        _fail(f"{label} mount identity is not an integer")
    if value < 1:
        _fail(f"{label} mount identity differs")
    return value


def _path_snapshot(path_text: str, label: str) -> dict[str, Any]:
    path = Path(path_text)
    if (
        not path.is_absolute()
        or os.path.normpath(path_text) != path_text
        or any(part in ("", ".", "..") for part in path.parts[1:])
    ):
        _fail(f"{label} is not an exact normalized absolute path")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    current_fd = os.open("/", flags)
    current_path = Path("/")
    try:
        for index, part in enumerate(path.parts[1:]):
            is_final = index == len(path.parts[1:]) - 1
            try:
                before = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
            except FileNotFoundError:
                return {
                    "path": path_text,
                    "exists": False,
                    "existing_prefix": os.fspath(current_path),
                    "mode": None,
                    "uid": None,
                    "nlink": None,
                    "dev": None,
                    "ino": None,
                    "mnt_id": None,
                }
            if stat.S_ISLNK(before.st_mode):
                _fail(f"{label} contains a symlink component")
            if not is_final and not stat.S_ISDIR(before.st_mode):
                _fail(f"{label} contains a non-directory ancestor")
            open_flags = os.O_CLOEXEC
            if hasattr(os, "O_PATH"):
                open_flags |= os.O_PATH
            elif stat.S_ISDIR(before.st_mode):
                open_flags |= os.O_RDONLY | os.O_DIRECTORY
            else:
                open_flags |= os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                open_flags |= os.O_NOFOLLOW
            next_fd = os.open(part, open_flags, dir_fd=current_fd)
            after = os.fstat(next_fd)
            if (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_uid,
                before.st_nlink,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_uid,
                after.st_nlink,
            ):
                os.close(next_fd)
                _fail(f"{label} changed during component replay")
            os.close(current_fd)
            current_fd = next_fd
            current_path /= part
        return {
            "path": path_text,
            "exists": True,
            "existing_prefix": path_text,
            "mode": after.st_mode,
            "uid": after.st_uid,
            "nlink": after.st_nlink,
            "dev": after.st_dev,
            "ino": after.st_ino,
            "mnt_id": _fd_mount_id(current_fd, label),
        }
    finally:
        os.close(current_fd)


def _path_is_ancestor_or_equal(left: str, right: str) -> bool:
    left_path = Path(left)
    right_path = Path(right)
    return left_path == right_path or left_path in right_path.parents


def _verify_path_boundaries(contract: Mapping[str, Any]) -> None:
    """Reject lexical, symlink, inode, and existing bind-alias collisions."""

    protected = contract["protected_state_policy"]
    source = contract["isolated_source_policy"]
    actual = contract["actual_execution_policy"]
    owner = contract["owner_authorization_policy"]
    ledger = contract["trial_ledger_policy"]
    trial_roots = [
        owner["authorization_root"],
        ledger["ledger_root"],
        source["clone_root"],
        actual["cargo_cache_source"],
    ]
    if len(trial_roots) != len(set(trial_roots)):
        _fail("trial roots are not pairwise distinct")
    for index, left in enumerate(trial_roots):
        for right in trial_roots[index + 1 :]:
            if _path_is_ancestor_or_equal(left, right) or _path_is_ancestor_or_equal(
                right, left
            ):
                _fail("trial roots overlap by ancestry")
    protected_paths = [
        *protected["forbidden_exact_roots"],
        protected["attempt2_artifact_path"],
        protected["attempt2_owner_authorization_path"],
        source["clone_source_repository"],
    ]
    for trial in [*trial_roots, actual["artifact_path"]]:
        for forbidden in protected_paths:
            if _path_is_ancestor_or_equal(trial, forbidden) or _path_is_ancestor_or_equal(
                forbidden, trial
            ):
                _fail("trial and protected paths overlap by ancestry")
    trial_paths = [*trial_roots, actual["artifact_path"]]
    trial_snapshots = [
        _path_snapshot(path, f"trial path {index}")
        for index, path in enumerate(trial_paths)
    ]
    protected_snapshot_paths = [
        protected["attempt2_worktree_root"],
        protected["attempt2_ledger_root"],
        actual["isolated_cargo_cache_policy"]["copy_source"],
        protected["attempt2_artifact_path"],
        protected["attempt2_owner_authorization_path"],
        source["clone_source_repository"],
    ]
    protected_snapshots = [
        _path_snapshot(path, f"protected path {index}")
        for index, path in enumerate(protected_snapshot_paths)
    ]
    for index, left in enumerate(trial_snapshots):
        if not left["exists"]:
            continue
        for right in trial_snapshots[index + 1 :] + protected_snapshots:
            if right["exists"] and (left["dev"], left["ino"]) == (
                right["dev"],
                right["ino"],
            ):
                _fail("trial path aliases another trial or protected inode")


def _verify_frozen_git_objects(
    contract: Mapping[str, Any], expected_git_identity: Mapping[str, Any]
) -> None:
    source = contract["isolated_source_policy"]
    repository = source["clone_source_repository"]
    for relative in source["git_source_forbidden_metadata_relative_paths"]:
        candidate = Path(repository) / relative
        if _path_snapshot(
            os.fspath(candidate), f"source Git forbidden metadata {relative}"
        )["exists"]:
            _fail(f"source Git forbidden metadata exists: {relative}")
    commit = source["checkout_commit"]
    if _COMMIT.fullmatch(commit) is None:
        _fail("frozen Git commit differs")
    tree = _git_readonly(
        contract, ["-C", repository, "rev-parse", f"{commit}^{{tree}}"], "Git tree replay", expected_git_identity
    ).decode("ascii").strip()
    parent = _git_readonly(
        contract, ["-C", repository, "rev-parse", f"{commit}^"], "Git parent replay", expected_git_identity
    ).decode("ascii").strip()
    if tree != source["checkout_tree"] or parent != source["checkout_commit_parent"]:
        _fail("frozen Git lineage differs")
    actual = contract["actual_execution_policy"]
    for relative, expected in {
        actual["old_supervisor_path"]: actual["old_supervisor_raw_sha256"],
        **source["actual_supervisor_config_raw_sha256"],
    }.items():
        raw = _git_readonly(
            contract,
            ["-C", repository, "show", f"{commit}:{relative}"],
            f"Git object {relative}",
            expected_git_identity,
        )
        if sha256(raw).hexdigest() != expected:
            _fail(f"frozen Git object differs: {relative}")
    parent_supervisor_raw = _git_readonly(
        contract,
        [
            "-C",
            repository,
            "show",
            f"{source['checkout_commit_parent']}:{actual['old_supervisor_path']}",
        ],
        "parent Git supervisor object replay",
        expected_git_identity,
    )
    if sha256(parent_supervisor_raw).hexdigest() != actual["old_supervisor_raw_sha256"]:
        _fail("old supervisor differs from the frozen parent")
    listing = _git_readonly(
        contract,
        ["-C", repository, "ls-tree", "-r", "-z", "--full-tree", commit],
        "Git tree special-entry replay",
        expected_git_identity,
    )
    symlink_count = 0
    gitlink_count = 0
    gitattributes_count = 0
    gitmodules_count = 0
    for entry in listing.split(b"\x00"):
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        mode, object_kind, _object_id = metadata.split(b" ", 2)
        if mode == b"120000":
            symlink_count += 1
        if mode == b"160000" or object_kind == b"commit":
            gitlink_count += 1
        basename = path.rsplit(b"/", 1)[-1]
        if basename == b".gitattributes":
            gitattributes_count += 1
        if basename == b".gitmodules":
            gitmodules_count += 1
    if {
        "symlink_count": symlink_count,
        "gitlink_count": gitlink_count,
        "gitattributes_count": gitattributes_count,
        "gitmodules_count": gitmodules_count,
    } != source["git_source_tree_special_entries"]:
        _fail("frozen Git special-entry registry differs")


def _tree_manifest(
    contract: Mapping[str, Any], root: Path, *, require_target_absent: bool = True
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        root_stat = os.lstat(root)
    except OSError as error:
        _fail(f"Cargo cache source is unavailable: {error}")
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        _fail("Cargo cache source is not a directory")
    if hasattr(os, "listxattr") and os.listxattr(root, follow_symlinks=False):
        _fail("Cargo cache root contains an extended attribute")
    rows: list[dict[str, Any]] = []
    directory_count = 1
    file_count = 0
    total_bytes = 0
    for current, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        directory_names.sort()
        file_names.sort()
        current_path = Path(current)
        for name in list(directory_names):
            child = current_path / name
            child_stat = os.lstat(child)
            if not stat.S_ISDIR(child_stat.st_mode) or stat.S_ISLNK(child_stat.st_mode):
                _fail("Cargo cache contains a non-directory tree entry")
            if hasattr(os, "listxattr") and os.listxattr(
                child, follow_symlinks=False
            ):
                _fail("Cargo cache directory contains an extended attribute")
            relative = child.relative_to(root).as_posix()
            if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
                _fail("Cargo cache relative path differs")
            rows.append(
                {
                    "relative_path": relative,
                    "file_kind": "DIRECTORY",
                    "mode": stat.S_IMODE(child_stat.st_mode),
                    "size": 0,
                    "raw_sha256": None,
                }
            )
            directory_count += 1
        for name in file_names:
            child = current_path / name
            raw, child_stat = _read_regular_nofollow(
                child,
                f"Cargo cache file {name}",
                with_stat=True,
                reject_xattrs=True,
            )
            if (
                not stat.S_ISREG(child_stat.st_mode)
                or child_stat.st_nlink != 1
            ):
                _fail("Cargo cache contains a non-regular or aliased file")
            relative = child.relative_to(root).as_posix()
            rows.append(
                {
                    "relative_path": relative,
                    "file_kind": "REGULAR",
                    "mode": stat.S_IMODE(child_stat.st_mode),
                    "size": len(raw),
                    "raw_sha256": sha256(raw).hexdigest(),
                }
            )
            file_count += 1
            total_bytes += len(raw)
    rows.sort(key=lambda row: row["relative_path"])
    manifest = {
        "schema_version": "hegel-phase3a-q05b-shadow-nonadoption-cargo-manifest/1",
        "source_root": os.fspath(root),
        "ordered_rows": rows,
    }
    identity = {
        "source_root": os.fspath(root),
        "target_root": contract["actual_execution_policy"]["cargo_cache_source"],
        "file_count": file_count,
        "directory_count": directory_count,
        "total_bytes": total_bytes,
        "manifest_digest": _domain_digest(
            contract["hash_domains"]["cargo_manifest"], manifest
        ),
        "target_initially_absent": not os.path.lexists(
            contract["actual_execution_policy"]["cargo_cache_source"]
        ),
        "copy_required": True,
    }
    if require_target_absent and identity["target_initially_absent"] is not True:
        _fail("isolated Cargo cache target already exists")
    return identity, rows


def _verify_attempt2_protected_state(
    contract: Mapping[str, Any], expected_git_identity: Mapping[str, Any]
) -> None:
    protected = contract["protected_state_policy"]
    worktree = protected["attempt2_worktree_root"]
    expected_toplevel = protected["attempt2_worktree_git_toplevel"]
    expected_git_dir = protected["attempt2_worktree_git_dir"]
    if expected_toplevel != worktree:
        _fail("attempt-2 Git toplevel contract differs")
    git_dir_before = _path_snapshot(expected_git_dir, "attempt-2 Git admin directory")
    if (
        git_dir_before["exists"] is not True
        or not stat.S_ISDIR(git_dir_before["mode"])
        or git_dir_before["uid"] != os.geteuid()
        or git_dir_before["nlink"] < 2
    ):
        _fail("attempt-2 Git admin directory identity differs")
    discovered_toplevel = _git_readonly(
        contract,
        ["-C", worktree, "rev-parse", "--show-toplevel"],
        "attempt-2 discovered worktree root replay",
        expected_git_identity,
    ).decode("utf-8").strip()
    discovered_git_dir = _git_readonly(
        contract,
        ["-C", worktree, "rev-parse", "--absolute-git-dir"],
        "attempt-2 discovered Git directory replay",
        expected_git_identity,
    ).decode("utf-8").strip()
    if (
        discovered_toplevel != expected_toplevel
        or discovered_git_dir != expected_git_dir
    ):
        _fail("attempt-2 discovered Git identity differs")
    pinned_git = [
        f"--git-dir={expected_git_dir}",
        f"--work-tree={expected_toplevel}",
    ]
    head = _git_readonly(
        contract,
        [*pinned_git, "rev-parse", "HEAD"],
        "attempt-2 HEAD replay",
        expected_git_identity,
    ).decode("ascii").strip()
    tree = _git_readonly(
        contract,
        [*pinned_git, "rev-parse", "HEAD^{tree}"],
        "attempt-2 tree replay",
        expected_git_identity,
    ).decode("ascii").strip()
    porcelain = _git_readonly(
        contract,
        [
            *pinned_git,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        "attempt-2 cleanliness replay",
        expected_git_identity,
    )
    pinned_toplevel = _git_readonly(
        contract,
        [*pinned_git, "rev-parse", "--show-toplevel"],
        "attempt-2 pinned worktree root replay",
        expected_git_identity,
    ).decode("utf-8").strip()
    pinned_git_dir = _git_readonly(
        contract,
        [*pinned_git, "rev-parse", "--absolute-git-dir"],
        "attempt-2 pinned Git directory replay",
        expected_git_identity,
    ).decode("utf-8").strip()
    git_dir_after = _path_snapshot(
        expected_git_dir, "attempt-2 Git admin directory replay"
    )
    git_identity_fields = ("dev", "ino", "mnt_id", "mode", "uid", "nlink")
    if (
        head != protected["attempt2_worktree_commit"]
        or tree != protected["attempt2_worktree_tree"]
        or porcelain != b""
        or pinned_toplevel != expected_toplevel
        or pinned_git_dir != expected_git_dir
        or any(
            git_dir_after[field] != git_dir_before[field]
            for field in git_identity_fields
        )
    ):
        _fail("attempt-2 worktree changed")
    authorization_raw = _read_regular_nofollow(
        Path(protected["attempt2_owner_authorization_path"]),
        "attempt-2 owner authorization",
        1 << 20,
    )
    if sha256(authorization_raw).hexdigest() != protected["attempt2_owner_authorization_raw_sha256"]:
        _fail("attempt-2 owner authorization bytes changed")
    if os.path.lexists(protected["attempt2_ledger_root"]):
        _fail("attempt-2 ledger is no longer absent")
    if os.path.lexists(protected["attempt2_artifact_path"]):
        _fail("attempt-2 artifact is no longer absent")


def _preflight_context(contract: Mapping[str, Any]) -> dict[str, Any]:
    _verify_path_boundaries(contract)
    capsule_policy = contract["capsule_policy"]
    python_identity = _executable_identity(
        capsule_policy["python_executable_path"], "Python"
    )
    git_identity = _executable_identity(capsule_policy["git_executable_path"], "Git")
    docker_identity = _executable_identity(
        capsule_policy["docker_executable_path"], "Docker"
    )
    _verify_frozen_git_objects(contract, git_identity)
    _verify_attempt2_protected_state(contract, git_identity)
    cargo_identity, cargo_rows = _tree_manifest(
        contract,
        Path(contract["actual_execution_policy"]["isolated_cargo_cache_policy"]["copy_source"]),
    )
    capsule = _core().build_shadow_nonadoption_execution_capsule_v1(
        contract,
        python_identity,
        git_identity,
        docker_identity,
        cargo_identity,
    )
    return {
        "capsule": capsule,
        "cargo_rows": cargo_rows,
        "python_identity": python_identity,
        "git_identity": git_identity,
        "docker_identity": docker_identity,
    }


def preflight_phase3_q05b_shadow_nonadoption_trial_v1() -> dict[str, Any]:
    """Derive the exact capsule without mutating any target or invoking Docker."""

    _verify_wrapper_entrypoint()
    contract = load_phase3_q05b_shadow_nonadoption_trial_contract_v1()
    _reject_preexisting_ledger_before_preflight(contract)
    context = _preflight_context(contract)
    source = contract["isolated_source_policy"]
    actual = contract["actual_execution_policy"]
    ledger = contract["trial_ledger_policy"]
    owner = contract["owner_authorization_policy"]
    if any(
        os.path.lexists(path)
        for path in (source["clone_root"], actual["cargo_cache_source"], ledger["ledger_root"])
    ):
        _fail("one or more fresh shadow trial targets already exist")
    authorization_path = Path(owner["authorization_root"]) / owner["authorization_record_name"]
    result: dict[str, Any] = {
        "schema_version": contract["source_surface_policy"]["preflight_schema_version"],
        "status": "SHADOW_NONADOPTION_TRIAL_PREFLIGHT_CAPSULE_DERIVED_EXECUTION_NOT_AUTHORIZED",
        "contract_digest": contract["contract_digest"],
        "source_binding_digest": contract["source_binding"]["source_binding_digest"],
        "capsule": context["capsule"],
        "capsule_digest": context["capsule"]["capsule_digest"],
        "owner_authorization_present": os.path.lexists(authorization_path),
        "execution_authorized": False,
        "clone_root_absent": True,
        "cargo_cache_target_absent": True,
        "ledger_root_absent": True,
        "supervisor_invocation_count": 0,
        "attempt2_state_touched": False,
        "meta_or_production_state_touched": False,
        "preflight_digest": None,
    }
    fields = contract["source_surface_policy"]["preflight_result_exact_fields"]
    if set(result) != set(fields) or len(result) != len(fields):
        _fail("preflight result fields differ")
    unsigned = deepcopy(result)
    unsigned.pop("preflight_digest")
    result["preflight_digest"] = _domain_digest(
        contract["hash_domains"]["preflight"], unsigned
    )
    return result


def _authorization_path(contract: Mapping[str, Any]) -> Path:
    policy = contract["owner_authorization_policy"]
    return Path(policy["authorization_root"]) / policy["authorization_record_name"]


def _open_held_directory_nofollow(path: Path, label: str) -> tuple[int, os.stat_result, int]:
    path_text = os.fspath(path)
    if (
        not path.is_absolute()
        or os.path.normpath(path_text) != path_text
        or any(part in ("", ".", "..") for part in path.parts[1:])
    ):
        _fail(f"{label} path differs")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    current_fd = os.open("/", flags)
    try:
        for part in path.parts[1:]:
            before = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
                _fail(f"{label} contains a non-directory or symlink component")
            next_fd = os.open(part, flags, dir_fd=current_fd)
            after = os.fstat(next_fd)
            if (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_uid,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_uid,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                os.close(next_fd)
                _fail(f"{label} changed during component replay")
            os.close(current_fd)
            current_fd = next_fd
        held_stat = os.fstat(current_fd)
        mount_id = _fd_mount_id(current_fd, label)
        result_fd = current_fd
        current_fd = -1
        return result_fd, held_stat, mount_id
    finally:
        if current_fd >= 0:
            os.close(current_fd)


def _read_regular_at_nofollow(
    directory_fd: int,
    name: str,
    label: str,
    maximum: int,
) -> tuple[bytes, os.stat_result]:
    if Path(name).name != name or name in ("", ".", ".."):
        _fail(f"{label} basename differs")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink < 1:
            _fail(f"{label} is not one linked regular file")
        if before.st_size > maximum:
            _fail(f"{label} exceeds its byte bound")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                _fail(f"{label} exceeds its byte bound")
            chunks.append(chunk)
        after = os.fstat(fd)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _fail(f"{label} changed during replay")
        pathname_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if identity != (
            pathname_stat.st_dev,
            pathname_stat.st_ino,
            pathname_stat.st_mode,
            pathname_stat.st_uid,
            pathname_stat.st_nlink,
            pathname_stat.st_size,
            pathname_stat.st_mtime_ns,
            pathname_stat.st_ctime_ns,
        ):
            _fail(f"{label} pathname identity changed during replay")
        return b"".join(chunks), before
    finally:
        os.close(fd)


def _load_authorization(
    contract: Mapping[str, Any], capsule: Mapping[str, Any]
) -> dict[str, Any]:
    path = _authorization_path(contract)
    root_fd, root_stat, root_mount_id = _open_held_directory_nofollow(
        path.parent, "owner authorization root"
    )
    try:
        if (
            stat.S_IMODE(root_stat.st_mode) != 0o700
            or root_stat.st_uid != os.geteuid()
        ):
            _fail("owner authorization root identity differs")
        raw, authorization_stat = _read_regular_at_nofollow(
            root_fd, path.name, "owner authorization", 1 << 20
        )
        if (
            stat.S_IMODE(authorization_stat.st_mode) != 0o444
            or authorization_stat.st_uid != os.geteuid()
            or authorization_stat.st_nlink != 1
        ):
            _fail("owner authorization filesystem identity differs")
        replay = _path_snapshot(os.fspath(path.parent), "owner authorization root replay")
        root_after = os.fstat(root_fd)
        if (
            replay["exists"] is not True
            or (replay["dev"], replay["ino"], replay["mnt_id"])
            != (root_stat.st_dev, root_stat.st_ino, root_mount_id)
            or (replay["mode"], replay["uid"], replay["nlink"])
            != (root_stat.st_mode, root_stat.st_uid, root_stat.st_nlink)
            or (
                root_after.st_dev,
                root_after.st_ino,
                root_after.st_mode,
                root_after.st_uid,
                root_after.st_nlink,
                root_after.st_size,
                root_after.st_mtime_ns,
                root_after.st_ctime_ns,
            )
            != (
                root_stat.st_dev,
                root_stat.st_ino,
                root_stat.st_mode,
                root_stat.st_uid,
                root_stat.st_nlink,
                root_stat.st_size,
                root_stat.st_mtime_ns,
                root_stat.st_ctime_ns,
            )
        ):
            _fail("owner authorization root changed during replay")
        value = _parse_canonical_json(raw, "owner authorization")
        return _core().validate_shadow_nonadoption_owner_authorization_v1(
            value, capsule, contract
        )
    finally:
        os.close(root_fd)


def _publish_ledger_record(
    directory_fd: int,
    name: str,
    record: Mapping[str, Any],
) -> None:
    outcome_sibling = {
        "shadow-trial.success.json": "shadow-trial.failure.json",
        "shadow-trial.failure.json": "shadow-trial.success.json",
    }.get(name)
    if outcome_sibling is not None and outcome_sibling in set(os.listdir(directory_fd)):
        _fail(
            "ledger outcome sibling already exists",
            terminal_after_intent=True,
        )
    raw = _canonical_json_bytes(record)
    if len(raw) > _LEDGER_RECORD_MAX_BYTES:
        _fail("ledger record exceeds its byte bound", terminal_after_intent=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(name, flags, 0o444, dir_fd=directory_fd)
    try:
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            written = os.write(fd, view[offset:])
            if written <= 0:
                _fail("ledger publication made no progress", terminal_after_intent=True)
            offset += written
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.fsync(directory_fd)


def _read_ledger_record_at(
    directory_fd: int, name: str, label: str
) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_size > _LEDGER_RECORD_MAX_BYTES
        ):
            _fail(f"{label} ledger file identity differs", terminal_after_intent=True)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > _LEDGER_RECORD_MAX_BYTES:
                _fail(
                    f"{label} ledger file exceeds its byte bound",
                    terminal_after_intent=True,
                )
            chunks.append(chunk)
        after = os.fstat(fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_uid,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _fail(f"{label} ledger file changed during replay", terminal_after_intent=True)
        try:
            pathname_stat = os.stat(
                name, dir_fd=directory_fd, follow_symlinks=False
            )
        except OSError as exc:
            _fail(
                f"{label} ledger pathname unavailable during replay: "
                f"{type(exc).__name__}",
                terminal_after_intent=True,
            )
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_uid,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            pathname_stat.st_dev,
            pathname_stat.st_ino,
            pathname_stat.st_mode,
            pathname_stat.st_uid,
            pathname_stat.st_nlink,
            pathname_stat.st_size,
            pathname_stat.st_mtime_ns,
            pathname_stat.st_ctime_ns,
        ):
            _fail(
                f"{label} ledger pathname identity changed during replay",
                terminal_after_intent=True,
            )
        return b"".join(chunks)
    finally:
        os.close(fd)


def _verify_exact_ledger_state(
    directory_fd: int,
    contract: Mapping[str, Any],
    expected_records: Mapping[str, Mapping[str, Any]],
) -> None:
    ledger = contract["trial_ledger_policy"]
    allowed_names = {
        ledger["intent_record_name"],
        ledger["consume_record_name"],
        ledger["success_record_name"],
        ledger["failure_record_name"],
    }
    expected_names = set(expected_records)
    if not expected_names <= allowed_names:
        _fail("ledger expected-record registry differs", terminal_after_intent=True)
    observed_names = set(os.listdir(directory_fd))
    if observed_names != expected_names:
        _fail("ledger state or outcome sibling exclusivity differs", terminal_after_intent=True)
    for name, record in expected_records.items():
        if _read_ledger_record_at(directory_fd, name, name) != _canonical_json_bytes(record):
            _fail(f"ledger record bytes differ: {name}", terminal_after_intent=True)


def _record_with_digest(
    contract: Mapping[str, Any], kind: str, value: dict[str, Any]
) -> dict[str, Any]:
    policy = contract["canonicalization"]["record_digest_policy"][kind]
    field = policy["digest_field"]
    unsigned = deepcopy(value)
    unsigned.pop(field)
    value[field] = _domain_digest(
        contract["hash_domains"][policy["domain_key"]], unsigned
    )
    return value


def _intent_record(
    contract: Mapping[str, Any], capsule: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, Any]:
    schema = contract["trial_ledger_policy"]["intent_record_schema"]
    value = {
        "schema_version": schema["schema_version"],
        "status": "INTENT_STAGED_UNCONSUMED",
        "trial_id": contract["trial_identity"]["trial_id"],
        "contract_digest": contract["contract_digest"],
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "clone_git_commit": contract["isolated_source_policy"]["checkout_commit"],
        "clone_git_tree": contract["isolated_source_policy"]["checkout_tree"],
        "old_supervisor_raw_sha256": contract["actual_execution_policy"]["old_supervisor_raw_sha256"],
        "artifact_path": contract["actual_execution_policy"]["artifact_path"],
        "previous_ledger_state": "OPEN_ALL_FOUR_RECORDS_ABSENT",
        "no_retry": True,
        "record_digest": None,
    }
    return _record_with_digest(contract, "intent", value)


def _consumed_record(
    contract: Mapping[str, Any], capsule: Mapping[str, Any], authorization: Mapping[str, Any], intent: Mapping[str, Any]
) -> dict[str, Any]:
    schema = contract["trial_ledger_policy"]["consumed_record_schema"]
    value = {
        "schema_version": schema["schema_version"],
        "status": "CONSUMED_OUTCOME_UNKNOWN",
        "trial_id": contract["trial_identity"]["trial_id"],
        "contract_digest": contract["contract_digest"],
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "intent_record_digest": intent["record_digest"],
        "previous_ledger_state": "INTENT_STAGED_UNCONSUMED",
        "consumed_before_supervisor": True,
        "retry_allowed": False,
        "record_digest": None,
    }
    return _record_with_digest(contract, "consumed", value)


def _run_exact(
    argv: Sequence[str],
    env: Mapping[str, str],
    label: str,
    expected_git_identity: Mapping[str, Any],
    git_mutator_umask: int,
) -> subprocess.CompletedProcess[bytes]:
    if _executable_identity(argv[0], "Git") != expected_git_identity:
        _fail(f"{label} Git executable identity differs", terminal_after_intent=True)
    if type(git_mutator_umask) is not int or git_mutator_umask != 0o022:
        _fail(f"{label} Git mutator umask differs", terminal_after_intent=True)
    old_umask = os.umask(git_mutator_umask)
    result: subprocess.CompletedProcess[bytes] | None = None
    primary_error: BaseException | None = None
    restore_error: BaseException | None = None
    try:
        result = subprocess.run(
            list(argv),
            cwd="/",
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            check=False,
        )
    except BaseException as error:
        primary_error = error
    finally:
        try:
            os.umask(old_umask)
        except BaseException as error:
            restore_error = error
    if primary_error is not None:
        detail = f"{label} raised: {type(primary_error).__name__}"
        if restore_error is not None:
            detail += f"; umask restoration raised: {type(restore_error).__name__}"
        _fail(detail, terminal_after_intent=True)
    if restore_error is not None:
        _fail(
            f"{label} umask restoration raised: {type(restore_error).__name__}",
            terminal_after_intent=True,
        )
    if result is None:
        _fail(f"{label} returned no result", terminal_after_intent=True)
    if result.returncode != 0 or result.stdout or result.stderr:
        _fail(f"{label} failed", terminal_after_intent=True)
    if _executable_identity(argv[0], "Git") != expected_git_identity:
        _fail(f"{label} Git executable changed", terminal_after_intent=True)
    return result


def _clone_git_mutator_boundary(
    contract: Mapping[str, Any],
    expected_git_dir: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source = contract["isolated_source_policy"]
    clone_root = Path(source["clone_root"])
    git_dir = Path(source["clone_git_dir"])
    if git_dir != clone_root / ".git":
        _fail("clone Git directory path differs", terminal_after_intent=True)
    snapshot = _path_snapshot(os.fspath(git_dir), "clone Git directory")
    if (
        snapshot["exists"] is not True
        or not stat.S_ISDIR(snapshot["mode"])
        or source["clone_git_dir_mode"] != "0755"
        or stat.S_IMODE(snapshot["mode"]) != 0o755
        or snapshot["uid"] != os.geteuid()
        or snapshot["nlink"] < 2
    ):
        _fail("clone Git directory identity differs", terminal_after_intent=True)
    identity_fields = ("dev", "ino", "mnt_id", "mode", "uid", "nlink")
    if expected_git_dir is not None and any(
        snapshot[field] != expected_git_dir[field] for field in identity_fields
    ):
        _fail("clone Git directory changed between mutators", terminal_after_intent=True)
    alternates_raw, alternates_stat = _read_regular_nofollow(
        git_dir / "objects/info/alternates",
        "pre-mutator Git alternates",
        4096,
        with_stat=True,
    )
    if (
        not stat.S_ISREG(alternates_stat.st_mode)
        or stat.S_IMODE(alternates_stat.st_mode) != 0o644
        or alternates_stat.st_nlink != 1
        or alternates_stat.st_uid != os.geteuid()
        or alternates_raw
        != source["git_postclone_alternates_exact_bytes"].encode("ascii")
    ):
        _fail("pre-mutator Git alternates identity differs", terminal_after_intent=True)
    for relative in source["git_clone_forbidden_metadata_relative_paths"]:
        candidate = clone_root / relative
        if _path_snapshot(
            os.fspath(candidate), f"pre-mutator forbidden Git metadata {relative}"
        )["exists"]:
            _fail(
                f"pre-mutator forbidden Git metadata exists: {relative}",
                terminal_after_intent=True,
            )
    return snapshot


def _prepare_isolated_clone(
    contract: Mapping[str, Any], capsule: Mapping[str, Any]
) -> None:
    source = contract["isolated_source_policy"]
    clone_root = Path(source["clone_root"])
    if os.path.lexists(clone_root):
        _fail("isolated clone target is not absent", terminal_after_intent=True)
    _verify_path_boundaries(contract)
    expected_git_identity = capsule["git_executable_identity"]
    _audit_event("before_git_clone")
    _run_exact(
        source["git_clone_argv"],
        source["git_runtime_environment"],
        "Git clone",
        expected_git_identity,
        source["git_mutator_umask"],
    )
    _verify_path_boundaries(contract)
    git_dir_identity = _clone_git_mutator_boundary(contract, None)
    _run_exact(
        source["git_detached_checkout_argv"],
        source["git_runtime_environment"],
        "Git detached checkout",
        expected_git_identity,
        source["git_mutator_umask"],
    )
    _verify_path_boundaries(contract)
    _clone_git_mutator_boundary(contract, git_dir_identity)
    _run_exact(
        source["git_remove_origin_argv"],
        source["git_runtime_environment"],
        "Git remote removal",
        expected_git_identity,
        source["git_mutator_umask"],
    )
    _verify_path_boundaries(contract)
    _clone_git_mutator_boundary(contract, git_dir_identity)
    _verify_isolated_clone(contract, capsule, require_artifact_absent=True)
    _audit_event("after_git_clone")


def _verify_isolated_clone(
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
    *,
    require_artifact_absent: bool,
) -> None:
    source = contract["isolated_source_policy"]
    clone_root = Path(source["clone_root"])
    clone_snapshot = _path_snapshot(os.fspath(clone_root), "isolated clone root")
    if (
        clone_snapshot["exists"] is not True
        or not stat.S_ISDIR(clone_snapshot["mode"])
        or clone_snapshot["uid"] != os.geteuid()
    ):
        _fail("isolated clone root identity differs", terminal_after_intent=True)
    _clone_git_mutator_boundary(contract, None)
    alternates = clone_root / ".git/objects/info/alternates"
    source_objects = Path(source["clone_source_repository"]) / ".git/objects"
    clone_objects = clone_root / ".git/objects"
    source_objects_snapshot = _path_snapshot(
        os.fspath(source_objects), "shared source Git object directory"
    )
    clone_objects_snapshot = _path_snapshot(
        os.fspath(clone_objects), "clone Git object directory"
    )
    if (
        source_objects_snapshot["exists"] is not True
        or clone_objects_snapshot["exists"] is not True
        or not stat.S_ISDIR(source_objects_snapshot["mode"])
        or not stat.S_ISDIR(clone_objects_snapshot["mode"])
        or (source_objects_snapshot["dev"], source_objects_snapshot["ino"])
        == (clone_objects_snapshot["dev"], clone_objects_snapshot["ino"])
    ):
        _fail("shared Git object directory identity differs", terminal_after_intent=True)
    alternates_raw, alternates_stat = _read_regular_nofollow(
        alternates, "Git alternates", 4096, with_stat=True
    )
    if (
        not stat.S_ISREG(alternates_stat.st_mode)
        or stat.S_IMODE(alternates_stat.st_mode) != 0o644
        or alternates_stat.st_nlink != 1
        or alternates_stat.st_uid != os.geteuid()
        or alternates_raw != source["git_postclone_alternates_exact_bytes"].encode("ascii")
    ):
        _fail("Git alternates identity differs", terminal_after_intent=True)
    for relative in source["git_clone_forbidden_metadata_relative_paths"]:
        candidate = clone_root / relative
        if _path_snapshot(
            os.fspath(candidate), f"clone Git forbidden metadata {relative}"
        )["exists"]:
            _fail(
                f"clone Git forbidden metadata exists: {relative}",
                terminal_after_intent=True,
            )
    pinned_git = [
        f"--git-dir={source['clone_git_dir']}",
        f"--work-tree={source['clone_git_toplevel']}",
    ]
    head = _git_readonly(
        contract,
        [*pinned_git, "rev-parse", "HEAD"],
        "clone HEAD replay",
        capsule["git_executable_identity"],
    ).decode("ascii").strip()
    tree = _git_readonly(
        contract,
        [*pinned_git, "rev-parse", "HEAD^{tree}"],
        "clone tree replay",
        capsule["git_executable_identity"],
    ).decode("ascii").strip()
    parent = _git_readonly(
        contract,
        [*pinned_git, "rev-parse", "HEAD^"],
        "clone parent replay",
        capsule["git_executable_identity"],
    ).decode("ascii").strip()
    tracked_status = _git_readonly(
        contract,
        [
            *pinned_git,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=no",
        ],
        "clone tracked cleanliness replay",
        capsule["git_executable_identity"],
    )
    untracked_paths = _git_readonly(
        contract,
        [*pinned_git, "ls-files", "--others", "-z"],
        "clone ignored-and-unignored other-file replay",
        capsule["git_executable_identity"],
    )
    remotes = _git_readonly(
        contract,
        [*pinned_git, "remote"],
        "clone remote replay",
        capsule["git_executable_identity"],
    )
    current_branch = _git_readonly(
        contract,
        [*pinned_git, "branch", "--show-current"],
        "clone detached-HEAD replay",
        capsule["git_executable_identity"],
    )
    tags = _git_readonly(
        contract,
        [*pinned_git, "tag", "--list"],
        "clone tag replay",
        capsule["git_executable_identity"],
    )
    git_toplevel = _git_readonly(
        contract,
        ["-C", os.fspath(clone_root), "rev-parse", "--show-toplevel"],
        "clone worktree root replay",
        capsule["git_executable_identity"],
    ).decode("utf-8").strip()
    git_dir = _git_readonly(
        contract,
        ["-C", os.fspath(clone_root), "rev-parse", "--absolute-git-dir"],
        "clone Git directory replay",
        capsule["git_executable_identity"],
    ).decode("utf-8").strip()
    allowed_untracked = b""
    if not require_artifact_absent:
        artifact_relative = Path(capsule["artifact_path"]).relative_to(clone_root).as_posix()
        allowed_untracked = artifact_relative.encode("utf-8") + b"\x00"
    if (
        head != capsule["clone_git_commit"]
        or tree != capsule["clone_git_tree"]
        or parent != capsule["clone_git_parent"]
        or tracked_status != b""
        or untracked_paths != allowed_untracked
        or remotes != b""
        or current_branch != b""
        or tags != b""
        or git_toplevel != source["clone_git_toplevel"]
        or git_dir != source["clone_git_dir"]
    ):
        _fail("isolated clone identity differs", terminal_after_intent=True)
    supervisor = clone_root / contract["actual_execution_policy"]["old_supervisor_path"]
    supervisor_raw, supervisor_stat = _read_regular_nofollow(
        supervisor, "clone-local supervisor", with_stat=True
    )
    if (
        sha256(supervisor_raw).hexdigest() != capsule["old_supervisor_raw_sha256"]
        or supervisor_stat.st_nlink != 1
        or supervisor_stat.st_uid != os.geteuid()
    ):
        _fail("clone-local supervisor differs", terminal_after_intent=True)
    for relative_path, expected_digest in source[
        "actual_supervisor_config_raw_sha256"
    ].items():
        config_path = clone_root / relative_path
        config_raw, config_stat = _read_regular_nofollow(
            config_path, f"clone-local config {relative_path}", with_stat=True
        )
        observed_digest = sha256(config_raw).hexdigest()
        if (
            observed_digest != expected_digest
            or config_stat.st_nlink != 1
            or config_stat.st_uid != os.geteuid()
        ):
            _fail("clone-local supervisor config differs", terminal_after_intent=True)
    if require_artifact_absent and os.path.lexists(capsule["artifact_path"]):
        _fail("clone-local artifact is not initially absent", terminal_after_intent=True)


def _prepare_isolated_cargo_cache(
    contract: Mapping[str, Any], expected_rows: Sequence[Mapping[str, Any]]
) -> None:
    _verify_path_boundaries(contract)
    policy = contract["actual_execution_policy"]["isolated_cargo_cache_policy"]
    source_root = Path(policy["copy_source"])
    target_root = Path(policy["copy_target"])
    if os.path.lexists(target_root):
        _fail("isolated Cargo target is not absent", terminal_after_intent=True)
    target_root.mkdir(mode=0o700, parents=False)
    parent_fd = os.open(
        os.fspath(target_root.parent), os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    )
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    rows = [deepcopy(dict(row)) for row in expected_rows]
    for row in rows:
        target = target_root / row["relative_path"]
        if row["file_kind"] == "DIRECTORY":
            target.mkdir(mode=row["mode"], parents=True, exist_ok=False)
            os.chmod(target, row["mode"], follow_symlinks=False)
    for row in rows:
        if row["file_kind"] != "REGULAR":
            continue
        source = source_root / row["relative_path"]
        target = target_root / row["relative_path"]
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        raw, source_stat = _read_regular_nofollow(
            source,
            f"Cargo copy source {row['relative_path']}",
            with_stat=True,
            reject_xattrs=True,
        )
        if (
            not stat.S_ISREG(source_stat.st_mode)
            or source_stat.st_nlink != 1
            or stat.S_IMODE(source_stat.st_mode) != row["mode"]
            or len(raw) != row["size"]
            or sha256(raw).hexdigest() != row["raw_sha256"]
        ):
            _fail("Cargo source changed before copy", terminal_after_intent=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(os.fspath(target), flags, row["mode"])
        try:
            offset = 0
            while offset < len(raw):
                written = os.write(fd, raw[offset:])
                if written <= 0:
                    _fail("Cargo copy made no progress", terminal_after_intent=True)
                offset += written
            os.fchmod(fd, row["mode"])
            os.fsync(fd)
        finally:
            os.close(fd)
    for directory in sorted(
        [target_root, *(path for path in target_root.rglob("*") if path.is_dir())],
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        fd = os.open(os.fspath(directory), os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    pre_identity, pre_rows = _tree_manifest(
        contract, source_root, require_target_absent=False
    )
    target_identity, target_rows = _tree_manifest_for_target(contract, target_root)
    if pre_rows != rows or target_rows != rows:
        _fail("Cargo source or target manifest differs after copy", terminal_after_intent=True)
    if pre_identity["manifest_digest"] != target_identity["manifest_digest"]:
        _fail("Cargo source and target manifest digests differ", terminal_after_intent=True)


def _tree_manifest_for_target(
    contract: Mapping[str, Any], target_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_root = Path(
        contract["actual_execution_policy"]["isolated_cargo_cache_policy"]["copy_source"]
    )
    shadow = deepcopy(contract)
    shadow["actual_execution_policy"]["isolated_cargo_cache_policy"]["copy_source"] = os.fspath(target_root)
    shadow["actual_execution_policy"]["cargo_cache_source"] = os.fspath(source_root)
    identity, rows = _tree_manifest(
        shadow, target_root, require_target_absent=False
    )
    manifest = {
        "schema_version": "hegel-phase3a-q05b-shadow-nonadoption-cargo-manifest/1",
        "source_root": os.fspath(source_root),
        "ordered_rows": rows,
    }
    identity["source_root"] = os.fspath(source_root)
    identity["target_root"] = os.fspath(target_root)
    identity["manifest_digest"] = _domain_digest(
        contract["hash_domains"]["cargo_manifest"], manifest
    )
    return identity, rows


def _replay_capsule_executable_identities(
    capsule: Mapping[str, Any],
) -> None:
    for field, label in (
        ("python_executable_identity", "Python"),
        ("git_executable_identity", "Git"),
        ("docker_executable_identity", "Docker"),
    ):
        expected = capsule[field]
        if _executable_identity(expected["path"], label) != expected:
            _fail(f"{label} executable identity changed", terminal_after_intent=True)


def _replay_cargo_source_and_target(
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
    expected_rows: Sequence[Mapping[str, Any]],
) -> None:
    policy = contract["actual_execution_policy"]["isolated_cargo_cache_policy"]
    source_root = Path(policy["copy_source"])
    target_root = Path(policy["copy_target"])
    target_stat = os.lstat(target_root)
    if (
        not stat.S_ISDIR(target_stat.st_mode)
        or stat.S_ISLNK(target_stat.st_mode)
        or stat.S_IMODE(target_stat.st_mode) != 0o700
        or target_stat.st_uid != os.geteuid()
    ):
        _fail("isolated Cargo target root identity differs", terminal_after_intent=True)
    source_identity, source_rows = _tree_manifest(
        contract, source_root, require_target_absent=False
    )
    target_identity, target_rows = _tree_manifest_for_target(contract, target_root)
    frozen_rows = [deepcopy(dict(row)) for row in expected_rows]
    capsule_identity = capsule["cargo_cache_identity"]
    if (
        source_rows != frozen_rows
        or target_rows != frozen_rows
        or source_identity["manifest_digest"] != capsule_identity["manifest_digest"]
        or target_identity["manifest_digest"] != capsule_identity["manifest_digest"]
        or source_identity["file_count"] != capsule_identity["file_count"]
        or source_identity["directory_count"] != capsule_identity["directory_count"]
        or source_identity["total_bytes"] != capsule_identity["total_bytes"]
        or target_identity["file_count"] != capsule_identity["file_count"]
        or target_identity["directory_count"] != capsule_identity["directory_count"]
        or target_identity["total_bytes"] != capsule_identity["total_bytes"]
    ):
        _fail("Cargo source or isolated target replay differs", terminal_after_intent=True)


def _preconsume_full_replay(
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
    expected_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Replay every mutable capsule dependency immediately before spending."""

    live_contract = load_phase3_q05b_shadow_nonadoption_trial_contract_v1()
    if _canonical_json_bytes(live_contract) != _canonical_json_bytes(contract):
        _fail("live contract or source binding changed", terminal_after_intent=True)
    _verify_path_boundaries(contract)
    _verify_frozen_git_objects(contract, capsule["git_executable_identity"])
    _verify_attempt2_protected_state(
        contract, capsule["git_executable_identity"]
    )
    _replay_capsule_executable_identities(capsule)
    _verify_isolated_clone(contract, capsule, require_artifact_absent=True)
    _replay_cargo_source_and_target(contract, capsule, expected_rows)
    if _docker_inventory(contract, capsule):
        _fail(
            "preconsumption Docker commit-label inventory is not empty",
            terminal_after_intent=True,
        )


def _docker_inventory(contract: Mapping[str, Any], capsule: Mapping[str, Any]) -> list[dict[str, str]]:
    policy = capsule["kill_reap_survivor_policy"]
    if _executable_identity(policy["docker_executable_path"], "Docker") != capsule["docker_executable_identity"]:
        _fail("Docker executable identity changed")
    result = subprocess.run(
        policy["docker_list_argv_template"],
        cwd="/",
        env=deepcopy(capsule["runtime_environment"]),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        check=False,
    )
    if result.returncode != 0 or result.stderr:
        _fail("Docker read-only inventory failed")
    rows: list[dict[str, str]] = []
    seen_slots: set[str] = set()
    seen_container_ids: set[str] = set()
    namespaces: set[str] = set()
    for raw_line in result.stdout.splitlines():
        parts = raw_line.decode("ascii").split("|")
        if len(parts) != 4:
            _fail("Docker inventory row differs")
        container_id, namespace, slot, commit = parts
        if (
            _CONTAINER_ID.fullmatch(container_id) is None
            or _NAMESPACE.fullmatch(namespace) is None
            or slot not in policy["docker_slot_values"]
            or commit != policy["docker_source_commit_label_value"]
            or slot in seen_slots
            or container_id in seen_container_ids
        ):
            _fail("Docker inventory identity differs")
        seen_slots.add(slot)
        seen_container_ids.add(container_id)
        namespaces.add(namespace)
        rows.append(
            {"container_id": container_id, "execution_namespace": namespace, "slot": slot, "source_commit": commit}
        )
    if len(namespaces) > 1:
        _fail("Docker inventory spans multiple namespaces")
    return rows


def _bounded_communicate(
    process: subprocess.Popen[bytes], capsule: Mapping[str, Any]
) -> tuple[int, bytes, bytes, bool]:
    limits = capsule["resource_limits"]
    deadline = time.monotonic() + capsule["monotonic_timeout_policy"]["maximum_seconds"]
    selector = selectors.DefaultSelector()
    streams = {process.stdout: "stdout", process.stderr: "stderr"}
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    overflow = False
    timed_out = False
    try:
        for stream, name in streams.items():
            if stream is None:
                _fail(f"supervisor {name} pipe is absent", terminal_after_intent=True)
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while selector.get_map() or process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            for key, _mask in selector.select(min(remaining, 0.25)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                name = key.data
                buffers[name].extend(chunk)
                if len(buffers[name]) > limits[f"{name}_max_bytes"]:
                    overflow = True
                    timed_out = True
                    break
            if timed_out:
                break
        if timed_out:
            child_reaped = _terminate_and_reap_supervisor_group(process, capsule)
            if not child_reaped:
                _fail(
                    "supervisor direct child could not be reaped",
                    terminal_after_intent=True,
                    child_reaped_verified=False,
                )
            return_code = process.returncode
            if type(return_code) is not int:
                _fail(
                    "supervisor return code is unavailable after reap",
                    terminal_after_intent=True,
                    child_reaped_verified=True,
                )
            return (
                return_code,
                bytes(buffers["stdout"]),
                bytes(buffers["stderr"]),
                True,
            )
        try:
            return_code = process.wait(
                timeout=capsule["kill_reap_survivor_policy"]["postkill_wait_seconds"]
            )
        except subprocess.TimeoutExpired:
            _fail("supervisor direct child wait exceeded its bound", terminal_after_intent=True)
        for stream, name in streams.items():
            if stream is None:
                continue
            while True:
                try:
                    chunk = os.read(stream.fileno(), 65536)
                except BlockingIOError:
                    break
                if not chunk:
                    break
                buffers[name].extend(chunk)
                if len(buffers[name]) > limits[f"{name}_max_bytes"]:
                    overflow = True
        group_present = _supervisor_group_present(process)
        if group_present is not False:
            child_reaped = _terminate_and_reap_supervisor_group(process, capsule)
            _fail(
                "supervisor process group survived direct-child return",
                terminal_after_intent=True,
                child_reaped_verified=child_reaped,
            )
        return (
            return_code,
            bytes(buffers["stdout"]),
            bytes(buffers["stderr"]),
            overflow,
        )
    except BaseException as error:
        if (
            isinstance(error, ShadowNonadoptionTrialError)
            and error.child_reaped_verified is not None
        ):
            raise
        child_reaped = _terminate_and_reap_supervisor_group(process, capsule)
        detail = (
            error.detail
            if isinstance(error, ShadowNonadoptionTrialError)
            else f"{type(error).__name__}: {error}"
        )
        _fail(
            detail,
            terminal_after_intent=True,
            child_reaped_verified=child_reaped,
        )
    finally:
        try:
            selector.close()
        except BaseException:
            pass


def _supervisor_group_present(process: subprocess.Popen[bytes]) -> bool | None:
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except BaseException:
        return None
    return True


def _terminate_and_reap_supervisor_group(
    process: subprocess.Popen[bytes], capsule: Mapping[str, Any]
) -> bool:
    """Best-effort idempotent cleanup for every exception after Popen."""

    policy = capsule["kill_reap_survivor_policy"]
    group_control_ok = True

    try:
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        except BaseException:
            group_control_ok = False
        grace_deadline = time.monotonic() + policy["first_signal_grace_seconds"]
        while time.monotonic() < grace_deadline:
            present = _supervisor_group_present(process)
            if present is False:
                break
            if present is None:
                group_control_ok = False
                break
            time.sleep(0.05)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except BaseException:
            group_control_ok = False
        direct_child_reaped = False
        try:
            process.wait(timeout=policy["postkill_wait_seconds"])
        except (subprocess.TimeoutExpired, ChildProcessError):
            direct_child_reaped = False
        else:
            direct_child_reaped = process.poll() is not None
        group_deadline = time.monotonic() + policy["postkill_wait_seconds"]
        group_absent: bool | None = _supervisor_group_present(process)
        while group_absent is True and time.monotonic() < group_deadline:
            time.sleep(0.05)
            group_absent = _supervisor_group_present(process)
        return direct_child_reaped and group_absent is False and group_control_ok
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except BaseException:
            pass
        try:
            process.wait(timeout=policy["postkill_wait_seconds"])
        except BaseException:
            return False
        return (
            process.poll() is not None
            and _supervisor_group_present(process) is False
        )
    finally:
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except BaseException:
                    pass


def _prepare_supervisor_spawn(
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
    expected_rows: Sequence[Mapping[str, Any]],
) -> None:
    _verify_path_boundaries(contract)
    _replay_capsule_executable_identities(capsule)
    _verify_isolated_clone(contract, capsule, require_artifact_absent=True)
    _replay_cargo_source_and_target(contract, capsule, expected_rows)
    if not callable(getattr(signal, "pthread_sigmask", None)):
        _fail("pthread signal-mask control is unavailable", terminal_after_intent=True)


def _spawn_supervisor_once(
    capsule: Mapping[str, Any],
    enter_popen_boundary: Callable[[], None],
) -> tuple[subprocess.Popen[bytes], str | None]:
    _replay_capsule_executable_identities(capsule)
    ordered_argv = deepcopy(capsule["ordered_argv"])
    clone_project_root = capsule["clone_project_root"]
    runtime_environment = deepcopy(capsule["runtime_environment"])
    popen = subprocess.Popen
    old_umask = os.umask(0o077)
    previous_mask: set[signal.Signals] | None = None
    process: subprocess.Popen[bytes] | None = None
    primary_error: BaseException | None = None
    restoration_errors: list[str] = []
    try:
        previous_mask = signal.pthread_sigmask(signal.SIG_SETMASK, set())
        enter_popen_boundary()
        process = popen(
            ordered_argv,
            cwd=clone_project_root,
            env=runtime_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            close_fds=True,
            pass_fds=(),
            restore_signals=True,
        )
    except BaseException as error:
        primary_error = error
    finally:
        if previous_mask is not None:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            except BaseException as error:
                restoration_errors.append(f"signal-mask restoration failed: {error}")
        try:
            os.umask(old_umask)
        except BaseException as error:
            restoration_errors.append(f"umask restoration failed: {error}")
    if primary_error is not None:
        if process is not None:
            details = [f"supervisor spawn raised after Popen: {primary_error}"]
            details.extend(restoration_errors)
            return process, "; ".join(details)
        if restoration_errors:
            _fail(
                f"supervisor Popen failed: {primary_error}; {'; '.join(restoration_errors)}",
                terminal_after_intent=True,
            )
        raise primary_error
    if process is None:
        _fail("supervisor Popen returned no process", terminal_after_intent=True)
    return process, "; ".join(restoration_errors) or None


def _decode_supervisor_success(
    contract: Mapping[str, Any], capsule: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, Any]:
    stdout = result["stdout"]
    stderr = result["stderr"]
    if result["interrupted"] is True:
        _fail("supervisor timed out or exceeded capture bounds", terminal_after_intent=True)
    if type(result["return_code"]) is not int or result["return_code"] != 0:
        _fail("supervisor returned nonzero", terminal_after_intent=True)
    if stderr != b"":
        _fail("supervisor stderr is not empty", terminal_after_intent=True)
    value = _parse_canonical_json(stdout, "supervisor stdout")
    stdout_fields = {
        "artifact_path",
        "artifact_sha256",
        "artifact_set_root",
        "candidate_receipt_root",
        "final_receipt_root",
        "qualification_count",
        "qualification_mask",
        "q1_state",
        "q1_gate_count",
        "q1_gate_mask",
        "q1_formal_output_roots",
        "actual_admission_prepublication_live_replay_root",
        "actual_admission_pre_artifact_live_replay_root",
        "actual_admission_stage10_live_replay_root",
        "actual_admission_postpublication_live_replay_root",
        "actual_final_delivery_identity",
        "actual_final_delivery_root",
        "publisher_handle_close_status",
        "stage_roots",
    }
    if set(value) != stdout_fields or len(value) != len(stdout_fields):
        _fail("supervisor stdout fields differ", terminal_after_intent=True)
    required = {
        "artifact_path": capsule["artifact_path"],
        "qualification_count": 20,
        "qualification_mask": 1048575,
        "q1_state": "NOT_RUN",
        "q1_gate_count": 0,
        "q1_gate_mask": 0,
        "q1_formal_output_roots": [None] * 8,
        "publisher_handle_close_status": "CLOSED",
    }
    for field, expected in required.items():
        if type(value.get(field)) is not type(expected) or value.get(field) != expected:
            _fail(f"supervisor stdout field differs: {field}", terminal_after_intent=True)
    raw_root_fields = (
        "artifact_sha256",
        "artifact_set_root",
        "candidate_receipt_root",
        "final_receipt_root",
        "actual_admission_prepublication_live_replay_root",
        "actual_admission_pre_artifact_live_replay_root",
        "actual_admission_stage10_live_replay_root",
        "actual_admission_postpublication_live_replay_root",
        "actual_final_delivery_root",
    )
    if any(
        type(value[field]) is not str or _RAW_SHA256.fullmatch(value[field]) is None
        for field in raw_root_fields
    ):
        _fail("supervisor stdout root differs", terminal_after_intent=True)
    if (
        type(value["stage_roots"]) is not list
        or len(value["stage_roots"]) != 10
        or any(type(item) is not str or _RAW_SHA256.fullmatch(item) is None for item in value["stage_roots"])
    ):
        _fail("supervisor stage roots differ", terminal_after_intent=True)
    artifact_path = Path(capsule["artifact_path"])
    artifact_raw, artifact_stat = _read_regular_nofollow(
        artifact_path,
        "clone-local Q0.5b artifact",
        1 << 30,
        with_stat=True,
    )
    if value.get("artifact_sha256") != sha256(artifact_raw).hexdigest():
        _fail("supervisor artifact digest differs", terminal_after_intent=True)
    if (
        not stat.S_ISREG(artifact_stat.st_mode)
        or stat.S_IMODE(artifact_stat.st_mode) != 0o444
        or artifact_stat.st_nlink != 1
        or artifact_stat.st_uid != os.geteuid()
        or artifact_stat.st_size != len(artifact_raw)
    ):
        _fail("supervisor artifact filesystem identity differs", terminal_after_intent=True)
    artifact = _parse_canonical_json(artifact_raw, "clone-local Q0.5b artifact")
    if set(artifact) != {"derived", "schema_version", "sections", "status"}:
        _fail("supervisor artifact fields differ", terminal_after_intent=True)
    if (
        artifact["schema_version"] != "hegel-phase3a-q05b-actual-artifact/1"
        or artifact["status"] != "Q05B_QUALIFICATION_20_OF_20_Q1_NOT_RUN"
        or type(artifact["derived"]) is not dict
        or type(artifact["sections"]) is not dict
    ):
        _fail("supervisor artifact header differs", terminal_after_intent=True)
    derived = artifact["derived"]
    derived_fields = {
        "actual_admission_evidence_root",
        "artifact_set_root",
        "bundle_evidence_root",
        "candidate_receipt_cbor_hex",
        "candidate_receipt_root",
        "closed_q1_authority",
        "final_receipt_cbor_hex",
        "final_receipt_root",
        "host_execution_binding_root",
        "isolation_evidence_root",
        "ordered_predicate_rows",
        "qualification_count",
        "qualification_mask",
        "resource_evidence_root",
    }
    closed_q1 = {
        "active_transition_allowed": False,
        "certificate_active": False,
        "formal_output_roots": [None] * 8,
        "gate_count": 0,
        "gate_mask": 0,
        "gate_total": 20,
        "formal_fixed_point_claimed": False,
        "m3_formal_roots": None,
        "outside_certificate_issued": False,
        "q1_receipt": None,
        "q2_state": "NOT_RUN",
        "state": "NOT_RUN",
    }
    if (
        set(derived) != derived_fields
        or type(derived["qualification_count"]) is not int
        or derived["qualification_count"] != 20
        or type(derived["qualification_mask"]) is not int
        or derived["qualification_mask"] != 1048575
        or derived["closed_q1_authority"] != closed_q1
        or derived["artifact_set_root"] != value["artifact_set_root"]
        or derived["candidate_receipt_root"] != value["candidate_receipt_root"]
        or derived["final_receipt_root"] != value["final_receipt_root"]
    ):
        _fail("supervisor artifact derived registry differs", terminal_after_intent=True)
    predicate_rows = derived["ordered_predicate_rows"]
    if (
        type(predicate_rows) is not list
        or len(predicate_rows) != 19
        or any(
            type(row) is not list
            or len(row) != 4
            or type(row[0]) is not int
            or row[0] != index + 1
            or type(row[1]) is not str
            or not row[1]
            or row[2] is not True
            or type(row[3]) is not str
            or _RAW_SHA256.fullmatch(row[3]) is None
            for index, row in enumerate(predicate_rows)
        )
    ):
        _fail("supervisor artifact predicate rows differ", terminal_after_intent=True)
    for field in (
        "actual_admission_evidence_root",
        "artifact_set_root",
        "bundle_evidence_root",
        "candidate_receipt_root",
        "final_receipt_root",
        "host_execution_binding_root",
        "isolation_evidence_root",
        "resource_evidence_root",
    ):
        if type(derived[field]) is not str or _RAW_SHA256.fullmatch(derived[field]) is None:
            _fail(f"supervisor artifact derived root differs: {field}", terminal_after_intent=True)
    for field in ("candidate_receipt_cbor_hex", "final_receipt_cbor_hex"):
        value_hex = derived[field]
        if (
            type(value_hex) is not str
            or not value_hex
            or len(value_hex) % 2 != 0
            or re.fullmatch(r"[0-9a-f]+", value_hex) is None
        ):
            _fail(f"supervisor artifact receipt bytes differ: {field}", terminal_after_intent=True)
    artifact_set_preimage = deepcopy(artifact)
    artifact_set_preimage["derived"].pop("artifact_set_root")
    artifact_set_root = sha256(
        b"HEGEL/Q05B/QUALIFICATION/ARTIFACT_SET/V1\x00"
        + _canonical_json_bytes(artifact_set_preimage)
    ).hexdigest()
    if artifact_set_root != derived["artifact_set_root"]:
        _fail("supervisor artifact-set root differs", terminal_after_intent=True)
    source_commit = contract["isolated_source_policy"]["checkout_commit"]
    sections = artifact["sections"]
    if (
        set(sections) != {
            "source_wire_profile",
            "five_sidecars",
            "endpoint_stdout_set",
            "host_stage",
            "actor_rows",
            "cargo_build_binary",
            "final_resource_rows",
            "negative_corpus",
            "scratch_rows",
            "actual_admission",
            "semantic_execution",
        }
        or type(sections.get("source_wire_profile")) is not dict
        or sections["source_wire_profile"].get("source_commit") != source_commit
        or type(sections.get("actual_admission")) is not dict
        or sections["actual_admission"].get("source_commit") != source_commit
        or sections["actual_admission"].get("actual_admission_evidence_root")
        != derived["actual_admission_evidence_root"]
    ):
        _fail("supervisor artifact source binding differs", terminal_after_intent=True)
    delivery = value["actual_final_delivery_identity"]
    delivery_fields = {
        "schema_version",
        "handoff_point",
        "source_commit",
        "artifact_path",
        "ordered_stage_root_rows",
        "actual_admission_evidence_root",
        "pre_artifact_live_marker_replay_root",
        "stage_10_live_marker_replay_root",
        "prepublication_live_marker_replay_root",
        "postpublication_live_marker_replay_root",
        "artifact_set_root",
        "candidate_receipt_root",
        "final_receipt_root",
        "publisher_parent_device",
        "publisher_parent_inode",
        "publisher_file_device",
        "publisher_file_inode",
        "publisher_file_mode",
        "publisher_file_nlink",
        "payload_length",
        "payload_sha256",
        "first_anchored_replay_sha256",
        "final_path_matches_publisher_owned_inode",
        "same_uid_mutation_after_handoff_outside_claim",
    }
    stage_names = [
        "FREEZE_SOURCE_IMAGES_AND_COMMAND_REGISTRY",
        "MATERIALIZE_AND_SEAL_THREE_SOURCE_SNAPSHOTS_AND_CARGO",
        "OFFLINE_RUST_TEST_AND_RELEASE_BUILD_AND_SEAL_BINARY",
        "RUN_TWO_HELD_ENDPOINTS_IN_PARALLEL",
        "SEAL_ENDPOINT_STDOUT_SIDECARS_CONTROLS_AND_RESOURCES",
        "RUN_HELD_TRUSTED_HOST_SEMANTIC_REPLAY",
        "CLOSE_HOST_RESOURCE_ISOLATION_NEGATIVE_VECTORS_AND_TOCTOU",
        "BUILD_AND_STRICTLY_DECODE_NINETEEN_ROW_CANDIDATE",
        "ADD_PREDICATE20_AND_BUILD_CANONICAL_ARTIFACT_IN_MEMORY",
        "STRICTLY_REPLAY_COMPLETE_ARTIFACT_BEFORE_PUBLICATION",
    ]
    expected_stage_rows = [
        [index + 1, name, value["stage_roots"][index]]
        for index, name in enumerate(stage_names)
    ]
    artifact_parent_stat = os.lstat(artifact_path.parent)
    if (
        type(delivery) is not dict
        or set(delivery) != delivery_fields
        or delivery.get("schema_version") != "hegel-phase3a-q05b-final-delivery-identity/2"
        or delivery.get("handoff_point")
        != "FINAL_ANCHORED_REPLAY_AFTER_POSTPUBLICATION_LIVE_BEFORE_HANDLE_CLOSE_AND_RETURN"
        or delivery.get("source_commit") != source_commit
        or delivery.get("artifact_path") != capsule["artifact_path"]
        or delivery.get("ordered_stage_root_rows") != expected_stage_rows
        or delivery.get("actual_admission_evidence_root")
        != derived["actual_admission_evidence_root"]
        or delivery.get("pre_artifact_live_marker_replay_root")
        != value["actual_admission_pre_artifact_live_replay_root"]
        or delivery.get("stage_10_live_marker_replay_root")
        != value["actual_admission_stage10_live_replay_root"]
        or delivery.get("prepublication_live_marker_replay_root")
        != value["actual_admission_prepublication_live_replay_root"]
        or delivery.get("postpublication_live_marker_replay_root")
        != value["actual_admission_postpublication_live_replay_root"]
        or delivery.get("artifact_set_root") != value["artifact_set_root"]
        or delivery.get("candidate_receipt_root") != value["candidate_receipt_root"]
        or delivery.get("final_receipt_root") != value["final_receipt_root"]
        or delivery.get("publisher_parent_device") != artifact_parent_stat.st_dev
        or delivery.get("publisher_parent_inode") != artifact_parent_stat.st_ino
        or delivery.get("payload_length") != len(artifact_raw)
        or delivery.get("payload_sha256") != sha256(artifact_raw).hexdigest()
        or delivery.get("first_anchored_replay_sha256") != sha256(artifact_raw).hexdigest()
        or delivery.get("publisher_file_device") != artifact_stat.st_dev
        or delivery.get("publisher_file_inode") != artifact_stat.st_ino
        or delivery.get("publisher_file_mode") != artifact_stat.st_mode
        or delivery.get("publisher_file_nlink") != artifact_stat.st_nlink
        or delivery.get("final_path_matches_publisher_owned_inode") is not True
        or delivery.get("same_uid_mutation_after_handoff_outside_claim") is not True
    ):
        _fail("supervisor final delivery identity differs", terminal_after_intent=True)
    delivery_root = sha256(
        b"HEGEL/Q05B/ACTUAL/FINAL_DELIVERY/V1\x00"
        + _canonical_json_bytes(delivery)
    ).hexdigest()
    if delivery_root != value["actual_final_delivery_root"]:
        _fail("supervisor final delivery root differs", terminal_after_intent=True)
    return {
        "stdout_raw_sha256": sha256(stdout).hexdigest(),
        "stderr_raw_sha256": sha256(stderr).hexdigest(),
        "artifact_raw_sha256": sha256(artifact_raw).hexdigest(),
        "artifact_dev": artifact_stat.st_dev,
        "artifact_ino": artifact_stat.st_ino,
        "artifact_mode": artifact_stat.st_mode,
        "artifact_nlink": artifact_stat.st_nlink,
        "artifact_uid": artifact_stat.st_uid,
        "artifact_size": artifact_stat.st_size,
    }


def _replay_artifact_evidence(
    capsule: Mapping[str, Any], evidence: Mapping[str, Any]
) -> None:
    artifact_path = Path(capsule["artifact_path"])
    artifact_raw, artifact_stat = _read_regular_nofollow(
        artifact_path,
        "replayed clone-local Q0.5b artifact",
        1 << 30,
        with_stat=True,
    )
    expected_identity = (
        evidence["artifact_dev"],
        evidence["artifact_ino"],
        evidence["artifact_mode"],
        evidence["artifact_nlink"],
        evidence["artifact_uid"],
        evidence["artifact_size"],
    )
    observed_identity = (
        artifact_stat.st_dev,
        artifact_stat.st_ino,
        artifact_stat.st_mode,
        artifact_stat.st_nlink,
        artifact_stat.st_uid,
        artifact_stat.st_size,
    )
    if (
        observed_identity != expected_identity
        or stat.S_IMODE(artifact_stat.st_mode) != 0o444
        or artifact_stat.st_nlink != 1
        or sha256(artifact_raw).hexdigest() != evidence["artifact_raw_sha256"]
    ):
        _fail("clone-local artifact changed after validation", terminal_after_intent=True)


def _success_record(
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    intent: Mapping[str, Any],
    consumed: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    schema = contract["trial_ledger_policy"]["success_record_schema"]
    value = {
        "schema_version": schema["schema_version"],
        "status": contract["outcome_policy"]["shadow_success_status"],
        "trial_id": contract["trial_identity"]["trial_id"],
        "contract_digest": contract["contract_digest"],
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "intent_record_digest": intent["record_digest"],
        "consumed_record_digest": consumed["record_digest"],
        "supervisor_invocation_count": 1,
        "supervisor_returned": True,
        "supervisor_exit_code": 0,
        "supervisor_stdout_raw_sha256": evidence["stdout_raw_sha256"],
        "supervisor_stderr_raw_sha256": evidence["stderr_raw_sha256"],
        "artifact_path": capsule["artifact_path"],
        "artifact_raw_sha256": evidence["artifact_raw_sha256"],
        "q05b_mechanics_predicate_count": 20,
        "q05b_mechanics_predicate_mask": 1048575,
        "q1_state": "NOT_RUN",
        "mechanics_only": True,
        "efficacy_claimed": False,
        "c1_claimed": False,
        "c2_claimed": False,
        "production_consumption_allowed": False,
        "meta_adoption_promotion_current_written": False,
        "attempt2_state_touched": False,
        "protected_state_replay_passed": True,
        "no_retry": True,
        "record_digest": None,
    }
    return _record_with_digest(contract, "success", value)


def _failure_state_for_phase(phase_id: str) -> str:
    mapping = {
        "CREATE_LEDGER_ROOT": "LEDGER_ROOT_OR_LOCK_PUBLICATION_UNCERTAIN_TERMINAL",
        "PUBLISH_INTENT": "INTENT_PUBLICATION_UNCERTAIN_TERMINAL",
        "PREPARE_CLONE": "INTENT_STAGED_PREPARATION_FAILED_CLOSED_UNCONSUMED",
        "PREPARE_CARGO_CACHE": "INTENT_STAGED_PREPARATION_FAILED_CLOSED_UNCONSUMED",
        "PRECONSUMPTION_FULL_REPLAY": "INTENT_STAGED_PREPARATION_FAILED_CLOSED_UNCONSUMED",
        "PUBLISH_CONSUMED": "CONSUMPTION_PUBLICATION_UNCERTAIN_TERMINAL",
        "SPAWN_PREPARE": "CONSUMED_BEFORE_SUPERVISOR_LAUNCH_FAILED",
        "SUPERVISOR_SPAWN": "SUPERVISOR_RAISED_TIMED_OUT_OR_CLEANUP_UNCERTAIN",
        "SUPERVISOR_WAIT": "SUPERVISOR_RAISED_TIMED_OUT_OR_CLEANUP_UNCERTAIN",
        "TIMEOUT_CLEANUP": "SUPERVISOR_RAISED_TIMED_OUT_OR_CLEANUP_UNCERTAIN",
        "DOCKER_CLEANUP": "SUPERVISOR_RAISED_TIMED_OUT_OR_CLEANUP_UNCERTAIN",
        "VALIDATE_EXIT": "SUPERVISOR_RETURNED_NONZERO_TERMINAL",
        "VALIDATE_STDOUT": "SUPERVISOR_RETURNED_ARTIFACT_MISSING_OR_INVALID",
        "VALIDATE_ARTIFACT": "SUPERVISOR_RETURNED_ARTIFACT_MISSING_OR_INVALID",
        "POSTRUN_PROTECTED_REPLAY": "POSTRETURN_NONADOPTION_OR_ISOLATION_REPLAY_FAILED",
        "POSTRUN_DOCKER_ABSENCE": "POSTRETURN_NONADOPTION_OR_ISOLATION_REPLAY_FAILED",
        "PUBLISH_SUCCESS": "SHADOW_OUTCOME_LINK_PUBLICATION_UNCERTAIN",
    }
    try:
        return mapping[phase_id]
    except KeyError:
        _fail(f"failure phase has no frozen state: {phase_id}", terminal_after_intent=True)


def _failure_record(
    contract: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    intent: Mapping[str, Any],
    consumed: Mapping[str, Any] | None,
    *,
    phase_id: str,
    detail: str,
    ledger_consumed: bool | None,
    spawn_attempted: bool,
    supervisor_called: bool,
    process_result: Mapping[str, Any] | None,
    survivor_absence_verified: bool | None,
    child_reaped_verified: bool | None,
    failure_publication_uncertain: bool,
    ledger_created: bool | None,
) -> dict[str, Any]:
    state = _failure_state_for_phase(phase_id)
    returned: bool | None
    if supervisor_called:
        returned = True if process_result is not None else None
    else:
        returned = False
    exit_code = process_result["return_code"] if process_result is not None else None
    stdout_digest = (
        sha256(process_result["stdout"]).hexdigest() if process_result is not None else None
    )
    stderr_digest = (
        sha256(process_result["stderr"]).hexdigest() if process_result is not None else None
    )
    artifact_path = contract["actual_execution_policy"]["artifact_path"]
    try:
        artifact_exists: bool | None = os.path.lexists(artifact_path)
    except OSError:
        artifact_exists = None
    after_spawn = supervisor_called
    if after_spawn:
        cleanup_attempted = True
        if child_reaped_verified is None or survivor_absence_verified is None:
            cleanup_complete: bool | None = None
        elif child_reaped_verified is False or survivor_absence_verified is False:
            cleanup_complete = False
        else:
            cleanup_complete = True
        if child_reaped_verified is False or survivor_absence_verified is False:
            survivor_absence: bool | None = False
        elif child_reaped_verified is True and survivor_absence_verified is True:
            survivor_absence = True
        else:
            survivor_absence = None
    else:
        cleanup_attempted = False
        cleanup_complete = True
        survivor_absence = True
    value: dict[str, Any] = {
        "schema_version": contract["failure_state_policy"]["schema_version"],
        "status": contract["failure_state_policy"]["fixed_fields"]["status"],
        "trial_id": contract["trial_identity"]["trial_id"],
        "contract_digest": contract["contract_digest"],
        "capsule_digest": capsule["capsule_digest"],
        "authorization_digest": authorization["authorization_digest"],
        "intent_record_digest": intent["record_digest"],
        "consumed_record_digest": (
            consumed["record_digest"] if consumed is not None and ledger_consumed is not False else None
        ),
        "error_state": state,
        "phase_id": phase_id,
        "detail": detail,
        "ledger_created": ledger_created,
        "ledger_consumed": ledger_consumed,
        "consumption_publication_uncertain": phase_id == "PUBLISH_CONSUMED",
        "supervisor_invocation_count": 1 if supervisor_called else 0,
        "supervisor_spawn_attempted": spawn_attempted,
        "supervisor_called": supervisor_called,
        "supervisor_returned": returned,
        "supervisor_exit_code": exit_code,
        "supervisor_stdout_raw_sha256": stdout_digest,
        "supervisor_stderr_raw_sha256": stderr_digest,
        "docker_or_cargo_may_have_run": after_spawn,
        "artifact_may_exist": artifact_exists,
        "cleanup_attempted": cleanup_attempted,
        "cleanup_complete": cleanup_complete,
        "survivor_absence_verified": survivor_absence,
        "attempt2_state_touched": False,
        "meta_or_production_state_touched": False,
        "adoption_or_current_pointer_written": False,
        "failure_record_publication_uncertain": failure_publication_uncertain,
        "no_retry": True,
        "receipt_digest": None,
    }
    return _record_with_digest(contract, "failure", value)


def _validate_failure_channel(
    contract: Mapping[str, Any],
    receipt: Mapping[str, Any],
    capsule: Mapping[str, Any],
    authorization: Mapping[str, Any],
    intent: Mapping[str, Any],
    consumed: Mapping[str, Any] | None,
    channel: str,
) -> dict[str, Any]:
    receipt_consumed_digest = receipt["consumed_record_digest"]
    if receipt_consumed_digest is not None:
        if consumed is None or consumed["record_digest"] != receipt_consumed_digest:
            _fail("failure receipt consumed lineage differs", terminal_after_intent=True)
    return _core().validate_shadow_nonadoption_ledger_record_v1(
        receipt,
        "failure",
        {
            "capsule_digest": capsule["capsule_digest"],
            "authorization_digest": authorization["authorization_digest"],
            "intent_record_digest": intent["record_digest"],
            "consumed_record_digest": receipt_consumed_digest,
            "receipt_channel": channel,
        },
        contract,
    )


def _create_and_hold_ledger_root(path: Path) -> tuple[int, int]:
    parent = path.parent
    if path.name in ("", ".", "..") or path.parent == path:
        _fail("ledger root path differs")
    parent_fd, _parent_stat, _parent_mount_id = _open_held_directory_nofollow(
        parent, "ledger parent"
    )
    try:
        try:
            os.mkdir(path.name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            _fail("shadow trial ledger already exists; resume and retry are forbidden")
        os.fsync(parent_fd)
        child_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            child_flags |= os.O_NOFOLLOW
        ledger_fd = os.open(path.name, child_flags, dir_fd=parent_fd)
        child_stat = os.fstat(ledger_fd)
        if (
            not stat.S_ISDIR(child_stat.st_mode)
            or stat.S_IMODE(child_stat.st_mode) != 0o700
            or child_stat.st_uid != os.geteuid()
            or child_stat.st_nlink < 2
        ):
            os.close(ledger_fd)
            _fail("held ledger root identity differs", terminal_after_intent=True)
        return parent_fd, ledger_fd
    except BaseException:
        os.close(parent_fd)
        raise


def _held_directory_identity(directory_fd: int, label: str) -> dict[str, int]:
    observed = os.fstat(directory_fd)
    if not stat.S_ISDIR(observed.st_mode):
        _fail(f"{label} is not a held directory", terminal_after_intent=True)
    return {
        "dev": observed.st_dev,
        "ino": observed.st_ino,
        "mode": observed.st_mode,
        "uid": observed.st_uid,
        "nlink": observed.st_nlink,
        "mnt_id": _fd_mount_id(directory_fd, label),
    }


def _verify_held_ledger_path(
    parent_path: Path,
    ledger_name: str,
    parent_fd: int,
    ledger_fd: int,
    expected_parent: Mapping[str, Any],
    expected_ledger: Mapping[str, Any],
) -> None:
    parent_now = _held_directory_identity(parent_fd, "held ledger parent")
    ledger_now = _held_directory_identity(ledger_fd, "held ledger root")
    parent_identity_fields = ("dev", "ino", "mode", "uid", "mnt_id")
    if (
        any(parent_now[field] != expected_parent[field] for field in parent_identity_fields)
        or ledger_now != expected_ledger
    ):
        _fail("held ledger directory identity changed", terminal_after_intent=True)
    parent_replay = _path_snapshot(os.fspath(parent_path), "ledger parent replay")
    if (
        parent_replay["exists"] is not True
        or any(
            parent_replay[field] != expected_parent[field]
            for field in parent_identity_fields
        )
    ):
        _fail("ledger parent pathname identity changed", terminal_after_intent=True)
    try:
        pathname_stat = os.stat(
            ledger_name, dir_fd=parent_fd, follow_symlinks=False
        )
    except OSError as error:
        _fail(
            f"ledger root pathname cannot be replayed: {error}",
            terminal_after_intent=True,
        )
    if (
        not stat.S_ISDIR(pathname_stat.st_mode)
        or stat.S_ISLNK(pathname_stat.st_mode)
        or stat.S_IMODE(pathname_stat.st_mode) != 0o700
        or pathname_stat.st_uid != os.geteuid()
        or (
            pathname_stat.st_dev,
            pathname_stat.st_ino,
            pathname_stat.st_mode,
            pathname_stat.st_uid,
            pathname_stat.st_nlink,
        )
        != (
            expected_ledger["dev"],
            expected_ledger["ino"],
            expected_ledger["mode"],
            expected_ledger["uid"],
            expected_ledger["nlink"],
        )
    ):
        _fail("ledger root pathname identity changed", terminal_after_intent=True)


def _reject_preexisting_ledger_before_preflight(
    contract: Mapping[str, Any],
) -> None:
    policy = contract["trial_ledger_policy"]
    ledger_root = Path(policy["ledger_root"])
    if not os.path.lexists(ledger_root):
        return
    intent_staged: bool | None = None
    authorization_consumed: bool | None = None
    root_fd: int | None = None
    try:
        root_fd, root_stat, root_mount_id = _open_held_directory_nofollow(
            ledger_root, "preexisting ledger root"
        )
        if (
            stat.S_IMODE(root_stat.st_mode) != 0o700
            or root_stat.st_uid != os.geteuid()
        ):
            raise OSError("preexisting ledger root identity differs")
        replay = _path_snapshot(
            os.fspath(ledger_root), "preexisting ledger root replay"
        )
        if (
            replay["exists"] is not True
            or (replay["dev"], replay["ino"], replay["mnt_id"])
            != (root_stat.st_dev, root_stat.st_ino, root_mount_id)
        ):
            raise OSError("preexisting ledger root pathname differs")
        names = set(os.listdir(root_fd))
        allowed = {
            policy["intent_record_name"],
            policy["consume_record_name"],
            policy["success_record_name"],
            policy["failure_record_name"],
        }
        if not names <= allowed:
            raise OSError("preexisting ledger contains an unknown record")

        def durable_record_present(name: str) -> bool | None:
            if name not in names:
                return None
            try:
                raw, observed = _read_regular_at_nofollow(
                    root_fd, name, f"preexisting ledger {name}", _LEDGER_RECORD_MAX_BYTES
                )
                if (
                    stat.S_IMODE(observed.st_mode) != 0o444
                    or observed.st_uid != os.geteuid()
                    or observed.st_nlink != 1
                ):
                    return None
                _parse_canonical_json(raw, f"preexisting ledger {name}")
            except BaseException:
                return None
            return True

        intent_staged = durable_record_present(policy["intent_record_name"])
        authorization_consumed = durable_record_present(
            policy["consume_record_name"]
        )
    except BaseException:
        intent_staged = None
        authorization_consumed = None
    finally:
        if root_fd is not None:
            os.close(root_fd)
    _fail(
        "shadow trial ledger already exists; resume and retry are forbidden",
        terminal_after_intent=True,
        wire_state={
            "intent_staged": intent_staged,
            "authorization_consumed": authorization_consumed,
            "supervisor_called": None,
            "trial_execution_performed": None,
            "supervisor_invocation_count": None,
        },
    )


def run_phase3_q05b_shadow_nonadoption_trial_v1() -> dict[str, Any]:
    """Execute at most one authorized clone-local shadow mechanics trial."""

    _verify_wrapper_entrypoint()
    contract = load_phase3_q05b_shadow_nonadoption_trial_contract_v1()
    _reject_preexisting_ledger_before_preflight(contract)
    context = _preflight_context(contract)
    capsule = context["capsule"]
    authorization = _load_authorization(contract, capsule)
    ledger_policy = contract["trial_ledger_policy"]
    ledger_root = Path(ledger_policy["ledger_root"])
    intent = _intent_record(contract, capsule, authorization)
    _core().validate_shadow_nonadoption_ledger_record_v1(
        intent,
        "intent",
        {
            "capsule_digest": capsule["capsule_digest"],
            "authorization_digest": authorization["authorization_digest"],
        },
        contract,
    )
    parent_fd: int | None = None
    ledger_fd: int | None = None
    ledger_parent_identity: dict[str, int] | None = None
    ledger_identity: dict[str, int] | None = None
    ledger_locked = False
    ledger_created: bool | None = None
    consumed: dict[str, Any] | None = None
    consumed_durable: bool | None = False
    spawn_attempted = False
    supervisor_called = False
    process_result: dict[str, Any] | None = None
    process: subprocess.Popen[bytes] | None = None
    phase_id = "CREATE_LEDGER_ROOT"
    success_publication_attempted = False

    def verify_ledger_path() -> None:
        if (
            parent_fd is None
            or ledger_fd is None
            or ledger_parent_identity is None
            or ledger_identity is None
        ):
            _fail("held ledger path state is incomplete", terminal_after_intent=True)
        _verify_held_ledger_path(
            ledger_root.parent,
            ledger_root.name,
            parent_fd,
            ledger_fd,
            ledger_parent_identity,
            ledger_identity,
        )

    try:
        _verify_path_boundaries(contract)
        parent_fd, ledger_fd = _create_and_hold_ledger_root(ledger_root)
        ledger_parent_identity = _held_directory_identity(
            parent_fd, "initial ledger parent"
        )
        ledger_identity = _held_directory_identity(ledger_fd, "initial ledger root")
        verify_ledger_path()
        ledger_created = True
        fcntl.flock(ledger_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_locked = True
        verify_ledger_path()
        if os.listdir(ledger_fd):
            _fail("shadow trial ledger is not empty", terminal_after_intent=True)
        phase_id = "PUBLISH_INTENT"
        verify_ledger_path()
        _publish_ledger_record(ledger_fd, ledger_policy["intent_record_name"], intent)
        verify_ledger_path()
        _verify_exact_ledger_state(
            ledger_fd,
            contract,
            {ledger_policy["intent_record_name"]: intent},
        )
        _audit_event("after_intent_durable")
        phase_id = "PREPARE_CLONE"
        _prepare_isolated_clone(contract, capsule)
        phase_id = "PREPARE_CARGO_CACHE"
        _prepare_isolated_cargo_cache(contract, context["cargo_rows"])
        consumed = _consumed_record(contract, capsule, authorization, intent)
        _core().validate_shadow_nonadoption_ledger_record_v1(
            consumed,
            "consumed",
            {
                "capsule_digest": capsule["capsule_digest"],
                "authorization_digest": authorization["authorization_digest"],
                "intent_record_digest": intent["record_digest"],
            },
            contract,
        )
        phase_id = "PRECONSUMPTION_FULL_REPLAY"
        _preconsume_full_replay(
            contract, capsule, context["cargo_rows"]
        )
        _verify_exact_ledger_state(
            ledger_fd,
            contract,
            {ledger_policy["intent_record_name"]: intent},
        )
        phase_id = "PUBLISH_CONSUMED"
        consumed_durable = None
        verify_ledger_path()
        _publish_ledger_record(
            ledger_fd, ledger_policy["consume_record_name"], consumed
        )
        verify_ledger_path()
        consumed_durable = True
        _verify_exact_ledger_state(
            ledger_fd,
            contract,
            {
                ledger_policy["intent_record_name"]: intent,
                ledger_policy["consume_record_name"]: consumed,
            },
        )
        _audit_event("after_consumed_durable")
        phase_id = "SPAWN_PREPARE"
        _prepare_supervisor_spawn(
            contract, capsule, context["cargo_rows"]
        )

        def enter_popen_boundary() -> None:
            nonlocal phase_id, spawn_attempted, supervisor_called
            phase_id = "SUPERVISOR_SPAWN"
            spawn_attempted = True
            supervisor_called = True

        process, spawn_restoration_error = _spawn_supervisor_once(
            capsule, enter_popen_boundary
        )
        if spawn_restoration_error is not None:
            _fail(spawn_restoration_error, terminal_after_intent=True)
        phase_id = "SUPERVISOR_WAIT"
        return_code, stdout, stderr, interrupted = _bounded_communicate(
            process, capsule
        )
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        process_result = {
            "return_code": return_code,
            "stdout": stdout,
            "stderr": stderr,
            "interrupted": interrupted,
        }
        if interrupted:
            _fail(
                "supervisor timed out or exceeded capture bounds",
                terminal_after_intent=True,
            )
        if return_code != 0:
            phase_id = "VALIDATE_EXIT"
            _fail("supervisor returned nonzero", terminal_after_intent=True)
        phase_id = "VALIDATE_STDOUT"
        evidence = _decode_supervisor_success(contract, capsule, process_result)
        phase_id = "VALIDATE_ARTIFACT"
        _replay_artifact_evidence(capsule, evidence)
        phase_id = "POSTRUN_DOCKER_ABSENCE"
        if _docker_inventory(contract, capsule):
            _fail("postreturn Docker commit-label inventory is not empty", terminal_after_intent=True)
        phase_id = "POSTRUN_PROTECTED_REPLAY"
        _replay_artifact_evidence(capsule, evidence)
        live_contract = load_phase3_q05b_shadow_nonadoption_trial_contract_v1()
        if _canonical_json_bytes(live_contract) != _canonical_json_bytes(contract):
            _fail("postreturn contract or source binding changed", terminal_after_intent=True)
        _verify_path_boundaries(contract)
        _verify_attempt2_protected_state(
            contract, capsule["git_executable_identity"]
        )
        _replay_capsule_executable_identities(capsule)
        _verify_isolated_clone(
            contract, capsule, require_artifact_absent=False
        )
        _replay_cargo_source_and_target(
            contract, capsule, context["cargo_rows"]
        )
        success = _success_record(
            contract, capsule, authorization, intent, consumed, evidence
        )
        _core().validate_shadow_nonadoption_ledger_record_v1(
            success,
            "success",
            {
                "capsule_digest": capsule["capsule_digest"],
                "authorization_digest": authorization["authorization_digest"],
                "intent_record_digest": intent["record_digest"],
                "consumed_record_digest": consumed["record_digest"],
            },
            contract,
        )
        phase_id = "PUBLISH_SUCCESS"
        _replay_artifact_evidence(capsule, evidence)
        _verify_exact_ledger_state(
            ledger_fd,
            contract,
            {
                ledger_policy["intent_record_name"]: intent,
                ledger_policy["consume_record_name"]: consumed,
            },
        )
        success_publication_attempted = True
        verify_ledger_path()
        _publish_ledger_record(
            ledger_fd, ledger_policy["success_record_name"], success
        )
        verify_ledger_path()
        _verify_exact_ledger_state(
            ledger_fd,
            contract,
            {
                ledger_policy["intent_record_name"]: intent,
                ledger_policy["consume_record_name"]: consumed,
                ledger_policy["success_record_name"]: success,
            },
        )
        _replay_artifact_evidence(capsule, evidence)
        verify_ledger_path()
        return success
    except BaseException as error:
        if phase_id == "CREATE_LEDGER_ROOT" and ledger_created is None:
            try:
                ledger_created = os.path.lexists(ledger_root)
            except OSError:
                ledger_created = None
        detail = (
            error.detail
            if isinstance(error, ShadowNonadoptionTrialError)
            else f"{type(error).__name__}: {error}"
        )
        child_reaped: bool | None
        if not supervisor_called:
            child_reaped = True
        elif process_result is not None:
            child_reaped = True
        elif (
            isinstance(error, ShadowNonadoptionTrialError)
            and error.child_reaped_verified is not None
        ):
            child_reaped = error.child_reaped_verified
        elif process is not None:
            child_reaped = _terminate_and_reap_supervisor_group(process, capsule)
        else:
            child_reaped = None
        docker_absence: bool | None
        if not supervisor_called:
            docker_absence = True
        else:
            try:
                docker_absence = not bool(_docker_inventory(contract, capsule))
            except BaseException:
                docker_absence = None
        transport_only = phase_id in (
            "CREATE_LEDGER_ROOT",
            "PUBLISH_INTENT",
            "PUBLISH_CONSUMED",
            "PUBLISH_SUCCESS",
        )
        failure_publication_uncertain = phase_id in (
            "CREATE_LEDGER_ROOT",
            "PUBLISH_INTENT",
        )
        receipt = _failure_record(
            contract,
            capsule,
            authorization,
            intent,
            consumed,
            phase_id=phase_id,
            detail=detail,
            ledger_consumed=consumed_durable,
            spawn_attempted=spawn_attempted,
            supervisor_called=supervisor_called,
            process_result=process_result,
            survivor_absence_verified=docker_absence,
            child_reaped_verified=child_reaped,
            failure_publication_uncertain=failure_publication_uncertain,
            ledger_created=ledger_created,
        )
        failure_lineage_consumed = consumed if consumed_durable is not False else None
        if (
            not transport_only
            and not success_publication_attempted
            and ledger_fd is not None
        ):
            _validate_failure_channel(
                contract,
                receipt,
                capsule,
                authorization,
                intent,
                failure_lineage_consumed,
                "LEDGER_RECORD",
            )
            try:
                expected_ledger: dict[str, Mapping[str, Any]] = {
                    ledger_policy["intent_record_name"]: intent,
                }
                if consumed_durable is True and consumed is not None:
                    expected_ledger[ledger_policy["consume_record_name"]] = consumed
                _verify_exact_ledger_state(
                    ledger_fd, contract, expected_ledger
                )
                verify_ledger_path()
                _publish_ledger_record(
                    ledger_fd, ledger_policy["failure_record_name"], receipt
                )
                verify_ledger_path()
                expected_ledger[ledger_policy["failure_record_name"]] = receipt
                _verify_exact_ledger_state(
                    ledger_fd, contract, expected_ledger
                )
                verify_ledger_path()
            except BaseException:
                receipt["failure_record_publication_uncertain"] = True
                receipt = _record_with_digest(contract, "failure", receipt)
        _validate_failure_channel(
            contract,
            receipt,
            capsule,
            authorization,
            intent,
            failure_lineage_consumed,
            "TRANSPORT_ONLY",
        )
        _fail(detail, receipt=receipt, terminal_after_intent=True)
    finally:
        if ledger_fd is not None:
            try:
                if ledger_locked:
                    fcntl.flock(ledger_fd, fcntl.LOCK_UN)
            except BaseException:
                pass
            try:
                os.close(ledger_fd)
            except BaseException:
                pass
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except BaseException:
                pass


class _CanonicalArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        _fail(f"command-line arguments differ: {message}")


def _parser() -> argparse.ArgumentParser:
    parser = _CanonicalArgumentParser(
        description=__doc__, add_help=False, allow_abbrev=False
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser


def _error_wire(error: ShadowNonadoptionTrialError) -> dict[str, Any]:
    receipt = error.receipt
    wire_state = error.wire_state
    if receipt is None and wire_state is not None:
        expected_fields = {
            "intent_staged",
            "authorization_consumed",
            "supervisor_called",
            "trial_execution_performed",
            "supervisor_invocation_count",
        }
        if set(wire_state) != expected_fields or len(wire_state) != len(expected_fields):
            _fail("error wire override fields differ")
        invocation_count = wire_state["supervisor_invocation_count"]
        supervisor_called = wire_state["supervisor_called"]
        authorization_consumed = wire_state["authorization_consumed"]
        trial_execution_performed = wire_state["trial_execution_performed"]
        intent_staged = wire_state["intent_staged"]
    else:
        invocation_count = (
            receipt["supervisor_invocation_count"] if receipt is not None else 0
        )
        supervisor_called = receipt["supervisor_called"] if receipt is not None else False
        authorization_consumed = receipt["ledger_consumed"] if receipt is not None else False
        trial_execution_performed = supervisor_called
        if receipt is None or receipt["error_state"] == (
            "LEDGER_ROOT_OR_LOCK_PUBLICATION_UNCERTAIN_TERMINAL"
        ):
            intent_staged = False
        elif receipt["error_state"] == "INTENT_PUBLICATION_UNCERTAIN_TERMINAL":
            intent_staged = None
        else:
            intent_staged = True
    return {
        "schema_version": "hegel-phase3a-q05b-shadow-nonadoption-trial-error/1",
        "status": (
            "SHADOW_NONADOPTION_TRIAL_TERMINAL_FAILURE"
            if error.terminal_after_intent
            else "SHADOW_NONADOPTION_TRIAL_PREINTENT_REJECTED"
        ),
        "detail": error.detail,
        "receipt": receipt,
        "intent_staged": intent_staged,
        "authorization_consumed": authorization_consumed,
        "supervisor_called": supervisor_called,
        "trial_execution_performed": trial_execution_performed,
        "supervisor_invocation_count": invocation_count,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        _verify_wrapper_entrypoint()
        options = _parser().parse_args(argv)
        value = (
            preflight_phase3_q05b_shadow_nonadoption_trial_v1()
            if options.preflight
            else run_phase3_q05b_shadow_nonadoption_trial_v1()
        )
    except Exception as error:
        if isinstance(error, ShadowNonadoptionTrialError):
            wrapper_error = error
        else:
            wrapper_error = ShadowNonadoptionTrialError(str(error))
        sys.stderr.buffer.write(_canonical_json_bytes(_error_wire(wrapper_error)))
        return 1 if wrapper_error.terminal_after_intent else 2
    sys.stdout.buffer.write(_canonical_json_bytes(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
