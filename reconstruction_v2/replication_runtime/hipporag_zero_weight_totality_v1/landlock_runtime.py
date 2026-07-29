"""Dependency-free extraction of the Landlock policy used by WikiSQL UAO."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import stat
from typing import Sequence


_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38
_LL_EXECUTE = 1 << 0
_LL_WRITE_FILE = 1 << 1
_LL_READ_FILE = 1 << 2
_LL_READ_DIR = 1 << 3
_LL_REMOVE_DIR = 1 << 4
_LL_REMOVE_FILE = 1 << 5
_LL_MAKE_CHAR = 1 << 6
_LL_MAKE_DIR = 1 << 7
_LL_MAKE_REG = 1 << 8
_LL_MAKE_SOCK = 1 << 9
_LL_MAKE_FIFO = 1 << 10
_LL_MAKE_BLOCK = 1 << 11
_LL_MAKE_SYM = 1 << 12
_LL_REFER = 1 << 13
_LL_TRUNCATE = 1 << 14
_LL_BASE = (
    _LL_EXECUTE
    | _LL_WRITE_FILE
    | _LL_READ_FILE
    | _LL_READ_DIR
    | _LL_REMOVE_DIR
    | _LL_REMOVE_FILE
    | _LL_MAKE_CHAR
    | _LL_MAKE_DIR
    | _LL_MAKE_REG
    | _LL_MAKE_SOCK
    | _LL_MAKE_FIFO
    | _LL_MAKE_BLOCK
    | _LL_MAKE_SYM
)


class LandlockRuntimeError(RuntimeError):
    """Landlock is unavailable or an allowlist policy cannot be installed."""


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


def landlock_abi_version() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    result = int(
        libc.syscall(
            _SYS_LANDLOCK_CREATE_RULESET,
            ctypes.c_void_p(),
            ctypes.c_size_t(0),
            ctypes.c_uint(_LANDLOCK_CREATE_RULESET_VERSION),
        )
    )
    if result < 1:
        raise LandlockRuntimeError("Landlock ABI is unavailable")
    return result


def _landlock_rights(abi: int) -> int:
    rights = _LL_BASE
    if abi >= 2:
        rights |= _LL_REFER
    if abi >= 3:
        rights |= _LL_TRUNCATE
    return rights


def apply_landlock(
    *,
    read_paths: Sequence[Path],
    write_paths: Sequence[Path],
    device_paths: Sequence[Path] = (),
) -> None:
    """Restrict the calling process; failure is terminal and has no fallback."""

    abi = landlock_abi_version()
    handled = _landlock_rights(abi)
    libc = ctypes.CDLL(None, use_errno=True)
    ruleset_attr = _RulesetAttr(handled_access_fs=handled)
    ruleset_fd = int(
        libc.syscall(
            _SYS_LANDLOCK_CREATE_RULESET,
            ctypes.byref(ruleset_attr),
            ctypes.sizeof(ruleset_attr),
            ctypes.c_uint(0),
        )
    )
    if ruleset_fd < 0:
        raise LandlockRuntimeError("Landlock ruleset creation failed")

    def add(path: Path, rights: int) -> None:
        flags = os.O_PATH | os.O_CLOEXEC
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise LandlockRuntimeError(
                "Landlock allowlist path is unavailable"
            ) from exc
        try:
            metadata = os.fstat(descriptor)
            allowed = rights
            if not stat.S_ISDIR(metadata.st_mode):
                allowed &= ~(
                    _LL_READ_DIR
                    | _LL_REMOVE_DIR
                    | _LL_REMOVE_FILE
                    | _LL_MAKE_CHAR
                    | _LL_MAKE_DIR
                    | _LL_MAKE_REG
                    | _LL_MAKE_SOCK
                    | _LL_MAKE_FIFO
                    | _LL_MAKE_BLOCK
                    | _LL_MAKE_SYM
                    | _LL_REFER
                )
            attribute = _PathBeneathAttr(
                allowed_access=allowed,
                parent_fd=descriptor,
                reserved=0,
            )
            result = int(
                libc.syscall(
                    _SYS_LANDLOCK_ADD_RULE,
                    ruleset_fd,
                    _LANDLOCK_RULE_PATH_BENEATH,
                    ctypes.byref(attribute),
                    ctypes.c_uint(0),
                )
            )
            if result != 0:
                raise LandlockRuntimeError(
                    "Landlock path rule creation failed"
                )
        finally:
            os.close(descriptor)

    try:
        read_rights = _LL_EXECUTE | _LL_READ_FILE | _LL_READ_DIR
        for path in dict.fromkeys(Path(row) for row in read_paths):
            add(path, read_rights)
        for path in dict.fromkeys(Path(row) for row in write_paths):
            add(path, handled)
        for path in dict.fromkeys(Path(row) for row in device_paths):
            add(path, _LL_READ_FILE | _LL_WRITE_FILE)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise LandlockRuntimeError("no_new_privs could not be set")
        if (
            libc.syscall(
                _SYS_LANDLOCK_RESTRICT_SELF,
                ruleset_fd,
                ctypes.c_uint(0),
            )
            != 0
        ):
            raise LandlockRuntimeError("Landlock restriction failed")
    finally:
        os.close(ruleset_fd)
