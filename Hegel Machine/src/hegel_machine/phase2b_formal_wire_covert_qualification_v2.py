"""Integrated synthetic/dev qualification candidate for Phase-2B Wire V3.

The public entry point owns its 1,024-row synthetic fixtures, entropy, trusted
wire builds, recognizer adapter, and two full covert-auditor executions.  It
does not accept a corpus, labels, keys, receipts, campaign identifiers, or an
execution configuration from its caller.

An ordinary host invocation can issue only
``CANDIDATE_IMPLEMENTATION_EXERCISED_NOT_R03_R04_R05``.  The six live runtime
bindings required by readiness nodes R03--R05 are deliberately outside this
entry point.  ``_VerifiedOfflineRuntimeEvidenceTokenV2`` reserves that later
offline-CLI boundary, but this module contains no issuer for that token and no
R03/R04/R05 receipt issuer.  Its candidate and calibration replay tokens are
private mechanics capabilities and grant no readiness authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass, replace
from enum import Enum
from fractions import Fraction
import argparse
import fcntl
import gc
import hashlib
from importlib.machinery import FileFinder, ModuleSpec, SOURCE_SUFFIXES, SourceFileLoader
import os
from pathlib import Path
import platform
import json
import math
import random
import re
import selectors
import shutil
import signal
import stat
import struct
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import time
from types import MappingProxyType, ModuleType
from typing import Final, Iterable, Mapping, NoReturn


_BOOTSTRAP_MODULE_RELATIVE_PATHS_V2 = {
    "hegel_machine.hashing": "src/hegel_machine/hashing.py",
    "hegel_machine.phase2b_exact_transform_core_v1": (
        "src/hegel_machine/phase2b_exact_transform_core_v1.py"
    ),
    "hegel_machine.phase2b_formal_covert_auditor_v1": (
        "src/hegel_machine/phase2b_formal_covert_auditor_v1.py"
    ),
    "hegel_machine.phase2b_formal_trusted_wire_v3": (
        "src/hegel_machine/phase2b_formal_trusted_wire_v3.py"
    ),
    "hegel_machine.phase2b_freeze_v1": "src/hegel_machine/phase2b_freeze_v1.py",
    "hegel_machine.rfc8785_v1": "src/hegel_machine/rfc8785_v1.py",
    "hegel_machine.schema": "src/hegel_machine/schema.py",
}
_BOOTSTRAP_FIXED_SOURCE_SHA256_V2 = {
    "src/hegel_machine/hashing.py": (
        "6fcd988ca4e8b75b9aa9e5a8b38c9ddd73e2685140e2cd0095d68f8bda054159"
    ),
    "src/hegel_machine/phase2b_exact_transform_core_v1.py": (
        "e0fa517b4275a1d2daee1fd1f877b61a8f79ca745a702979615b2130e53bdc5f"
    ),
    "src/hegel_machine/phase2b_formal_covert_auditor_v1.py": (
        "f4ea0bb54fd7b4f5707a6eea22ac8b728279e4f2e5559899626e4bf477d87312"
    ),
    "src/hegel_machine/phase2b_formal_trusted_wire_v3.py": (
        "c8120920f0e31b959043f63a0f672471fcbea50d05ca2cd8eaf7fd0c29089765"
    ),
    "src/hegel_machine/phase2b_freeze_v1.py": (
        "39a289ef0c000c1c62eb3a7a5f97ddfae3e51eb7904757642b0c2d167067f543"
    ),
    "src/hegel_machine/rfc8785_v1.py": (
        "16dcc6524a1c4a507edb9f85455c72b00fbc6c0c1b67471f03b67b2f25d32654"
    ),
    "src/hegel_machine/schema.py": (
        "6497a93f8a68f3313d294bc99dc3b8f43508e93772d748521d09ea716b8ae10a"
    ),
}
_BOOTSTRAP_MAIN_RELATIVE_PATH_V2 = (
    "src/hegel_machine/phase2b_formal_wire_covert_qualification_v2.py"
)
_QUALIFICATION_TRANSACTION_RELATIVE_PATHS_V2 = {
    "config": "config/phase2b_formal_wire_covert_qualification_v2.json",
    "documentation": (
        "docs/Hegel_Machine_Phase2B_Formal_Wire_Covert_Qualification_v2.md"
    ),
    "lock": "config/phase2b_formal_wire_covert_qualification_v2.lock.json",
    "source": _BOOTSTRAP_MAIN_RELATIVE_PATH_V2,
    "test": "tests/test_phase2b_formal_wire_covert_qualification_v2.py",
}
SOURCE_EXECUTION_BOOTSTRAP_CODE_V2 = r'''import hashlib,json,os,re,stat,sys
m=json.loads(sys.argv[1]);root=sys.argv[2]
mods={"hegel_machine.hashing":"src/hegel_machine/hashing.py","hegel_machine.phase2b_exact_transform_core_v1":"src/hegel_machine/phase2b_exact_transform_core_v1.py","hegel_machine.phase2b_formal_covert_auditor_v1":"src/hegel_machine/phase2b_formal_covert_auditor_v1.py","hegel_machine.phase2b_formal_trusted_wire_v3":"src/hegel_machine/phase2b_formal_trusted_wire_v3.py","hegel_machine.phase2b_freeze_v1":"src/hegel_machine/phase2b_freeze_v1.py","hegel_machine.rfc8785_v1":"src/hegel_machine/rfc8785_v1.py","hegel_machine.schema":"src/hegel_machine/schema.py"}
main="src/hegel_machine/phase2b_formal_wire_covert_qualification_v2.py"
if set(m)!={"main","modules","schema_version","sha256"} or m["schema_version"]!="hegel-machine-python-source-open-once-bootstrap/2" or m["main"]!=main or m["modules"]!=mods or set(m["sha256"])!=set(mods.values())|{main}:raise RuntimeError("source bootstrap manifest drift")
if not os.path.isabs(root) or os.path.islink(root) or not os.path.isdir(root) or os.path.realpath(root)!=root:raise RuntimeError("source bootstrap root drift")
blobs={};snaps={}
for rel in sorted(m["sha256"]):
 d=m["sha256"][rel]
 if re.fullmatch(r"[0-9a-f]{64}",d) is None:raise RuntimeError("source bootstrap digest drift")
 p=os.path.join(root,*rel.split("/"));fd=os.open(p,os.O_RDONLY|os.O_CLOEXEC|os.O_NONBLOCK|os.O_NOFOLLOW)
 try:
  a=os.fstat(fd)
  if not stat.S_ISREG(a.st_mode) or not 1<=a.st_size<=16*1024*1024 or os.path.realpath("/proc/self/fd/"+str(fd))!=p:raise RuntimeError("source bootstrap descriptor drift")
  b=b"";left=a.st_size
  while left:
   x=os.read(fd,min(left,65536))
   if not x:raise RuntimeError("source bootstrap short read")
   b+=x;left-=len(x)
  if os.read(fd,1):raise RuntimeError("source bootstrap growth")
  z=os.fstat(fd)
  if (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)!=(z.st_dev,z.st_ino,z.st_size,z.st_mtime_ns) or hashlib.sha256(b).hexdigest()!=d:raise RuntimeError("source bootstrap byte mismatch")
 finally:os.close(fd)
 blobs[rel]=b;snaps[rel]={"descriptor_realpath":p,"device":a.st_dev,"inode":a.st_ino,"mtime_nanoseconds":a.st_mtime_ns,"sha256":d,"size_bytes":a.st_size}
g={"__builtins__":__builtins__,"__cached__":None,"__doc__":None,"__file__":snaps[main]["descriptor_realpath"],"__loader__":None,"__name__":"__main__","__package__":None,"__spec__":None,"_EXECUTED_MAIN_SOURCE_BOOTSTRAP_V2":dict(snaps[main],bootstrap_kind="EXTERNAL_PYTHON_C_OPEN_ONCE_V2",reexecuted_from_stable_descriptor=True),"_HEGEL_PREOPENED_SOURCE_BYTES_V2":{n:blobs[r] for n,r in mods.items()},"_HEGEL_PREOPENED_SOURCE_SNAPSHOTS_V2":snaps,"_HEGEL_SOURCE_BOOTSTRAP_ACTIVE_V2":True,"_HEGEL_SOURCE_BOOTSTRAP_MANIFEST_V2":m}
sys.argv=[g["__file__"],*sys.argv[3:]]
exec(compile(blobs[main],g["__file__"],"exec",dont_inherit=True),g)
'''


_EXECUTED_MAIN_SOURCE_BOOTSTRAP_V2 = globals().get(
    "_EXECUTED_MAIN_SOURCE_BOOTSTRAP_V2"
)
if (
    __name__ == "__main__"
    and not __package__
    and globals().get("_HEGEL_SOURCE_BOOTSTRAP_ACTIVE_V2") is not True
):
    # A direct file invocation is only a launcher.  Before any project import or
    # CLI business entry, replace it with the exact isolated external open-once
    # bootstrap used by retained inner/outer/RFC evidence.
    if (
        sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.dont_write_bytecode != 1
        or sys.flags.ignore_environment != 1
    ):
        raise RuntimeError(
            "direct source bootstrap requires initial Python -I -S -B flags"
        )
    _bootstrap_path_v2 = Path(__file__).resolve(strict=True)
    _bootstrap_project_root_v2 = _bootstrap_path_v2.parents[2]
    if _bootstrap_path_v2 != (
        _bootstrap_project_root_v2 / _BOOTSTRAP_MAIN_RELATIVE_PATH_V2
    ):
        raise RuntimeError("direct source bootstrap project-root layout drifted")
    _bootstrap_flags_v2 = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("direct source bootstrap requires O_NOFOLLOW")
    _bootstrap_flags_v2 |= os.O_NOFOLLOW
    _bootstrap_fd_v2 = os.open(_bootstrap_path_v2, _bootstrap_flags_v2)
    try:
        _bootstrap_before_v2 = os.fstat(_bootstrap_fd_v2)
        if (
            not stat.S_ISREG(_bootstrap_before_v2.st_mode)
            or not 1 <= _bootstrap_before_v2.st_size <= 16 * 1_024 * 1_024
            or os.path.realpath(f"/proc/self/fd/{_bootstrap_fd_v2}")
            != str(_bootstrap_path_v2)
        ):
            raise RuntimeError("direct source bootstrap rejected its descriptor")
        _bootstrap_blocks_v2: list[bytes] = []
        _bootstrap_remaining_v2 = _bootstrap_before_v2.st_size
        while _bootstrap_remaining_v2:
            _bootstrap_block_v2 = os.read(
                _bootstrap_fd_v2, min(_bootstrap_remaining_v2, 65_536)
            )
            if not _bootstrap_block_v2:
                raise RuntimeError("direct source bootstrap observed a short read")
            _bootstrap_blocks_v2.append(_bootstrap_block_v2)
            _bootstrap_remaining_v2 -= len(_bootstrap_block_v2)
        if os.read(_bootstrap_fd_v2, 1):
            raise RuntimeError("direct source bootstrap observed source growth")
        _bootstrap_after_v2 = os.fstat(_bootstrap_fd_v2)
        if (
            _bootstrap_before_v2.st_dev,
            _bootstrap_before_v2.st_ino,
            _bootstrap_before_v2.st_size,
            _bootstrap_before_v2.st_mtime_ns,
        ) != (
            _bootstrap_after_v2.st_dev,
            _bootstrap_after_v2.st_ino,
            _bootstrap_after_v2.st_size,
            _bootstrap_after_v2.st_mtime_ns,
        ):
            raise RuntimeError("direct source bootstrap observed source mutation")
        _bootstrap_source_v2 = b"".join(_bootstrap_blocks_v2)
    finally:
        os.close(_bootstrap_fd_v2)
    _bootstrap_main_sha256_v2 = hashlib.sha256(_bootstrap_source_v2).hexdigest()
    _bootstrap_manifest_v2 = {
        "schema_version": "hegel-machine-python-source-open-once-bootstrap/2",
        "main": _BOOTSTRAP_MAIN_RELATIVE_PATH_V2,
        "modules": dict(_BOOTSTRAP_MODULE_RELATIVE_PATHS_V2),
        "sha256": {
            **_BOOTSTRAP_FIXED_SOURCE_SHA256_V2,
            _BOOTSTRAP_MAIN_RELATIVE_PATH_V2: _bootstrap_main_sha256_v2,
        },
    }
    _bootstrap_python_v2 = os.path.realpath(sys.executable)
    if not os.path.isabs(_bootstrap_python_v2):
        raise RuntimeError("direct source bootstrap Python path is not absolute")
    os.execv(
        _bootstrap_python_v2,
        [
            _bootstrap_python_v2,
            "-I",
            "-S",
            "-B",
            "-c",
            SOURCE_EXECUTION_BOOTSTRAP_CODE_V2,
            json.dumps(
                _bootstrap_manifest_v2,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            str(_bootstrap_project_root_v2),
            *sys.argv[1:],
        ],
    )
    raise RuntimeError("direct source bootstrap exec returned unexpectedly")


_SOURCE_ONLY_IMPORTER_ACTIVE_V2 = False
_PREOPENED_SOURCE_BYTES_V2 = globals().get("_HEGEL_PREOPENED_SOURCE_BYTES_V2")
_PREOPENED_SOURCE_SNAPSHOTS_V2 = globals().get(
    "_HEGEL_PREOPENED_SOURCE_SNAPSHOTS_V2"
)
_SOURCE_EXECUTION_BOOTSTRAP_ACTIVE_V2 = (
    globals().get("_HEGEL_SOURCE_BOOTSTRAP_ACTIVE_V2") is True
)
if __name__ == "__main__" and not __package__:
    class _SourceOnlyLoaderV2(SourceFileLoader):
        """Compile the pinned source file directly; never consult a ``.pyc``."""

        def get_code(self, fullname: str):  # type: ignore[override]
            source_path = self.get_filename(fullname)
            if _SOURCE_EXECUTION_BOOTSTRAP_ACTIVE_V2:
                if (
                    type(_PREOPENED_SOURCE_BYTES_V2) is not dict
                    or type(_PREOPENED_SOURCE_SNAPSHOTS_V2) is not dict
                    or fullname not in _BOOTSTRAP_MODULE_RELATIVE_PATHS_V2
                    or set(_PREOPENED_SOURCE_BYTES_V2)
                    != set(_BOOTSTRAP_MODULE_RELATIVE_PATHS_V2)
                ):
                    raise ImportError(
                        "source-only loader rejected an unregistered module"
                    )
                relative = _BOOTSTRAP_MODULE_RELATIVE_PATHS_V2[fullname]
                snapshot = _PREOPENED_SOURCE_SNAPSHOTS_V2.get(relative)
                source = _PREOPENED_SOURCE_BYTES_V2[fullname]
                if (
                    type(source) is not bytes
                    or type(snapshot) is not dict
                    or snapshot.get("descriptor_realpath") != source_path
                    or snapshot.get("sha256")
                    != _BOOTSTRAP_FIXED_SOURCE_SHA256_V2.get(relative)
                    or hashlib.sha256(source).hexdigest() != snapshot.get("sha256")
                    or len(source) != snapshot.get("size_bytes")
                ):
                    raise ImportError(
                        "source-only loader preopened byte/digest binding drifted"
                    )
                return compile(source, source_path, "exec", dont_inherit=True)
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(source_path, flags)
            try:
                before = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or not 1 <= before.st_size <= 16 * 1_024 * 1_024
                    or os.path.realpath(f"/proc/self/fd/{descriptor}")
                    != source_path
                ):
                    raise ImportError("source-only loader rejected its source file")
                blocks: list[bytes] = []
                remaining = before.st_size
                while remaining:
                    block = os.read(descriptor, min(remaining, 65_536))
                    if not block:
                        raise ImportError("source-only loader observed a short read")
                    blocks.append(block)
                    remaining -= len(block)
                if os.read(descriptor, 1):
                    raise ImportError("source-only loader observed source growth")
                after = os.fstat(descriptor)
                if (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                ) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    raise ImportError("source-only loader observed source mutation")
                source = b"".join(blocks)
            finally:
                os.close(descriptor)
            return compile(source, source_path, "exec", dont_inherit=True)

    _PACKAGE_DIRECTORY = Path(__file__).resolve(strict=True).parent
    _PACKAGE_STUB = ModuleType("hegel_machine")
    _PACKAGE_STUB.__file__ = str(_PACKAGE_DIRECTORY / "__init__.py")
    _PACKAGE_STUB.__package__ = "hegel_machine"
    _PACKAGE_STUB.__path__ = [str(_PACKAGE_DIRECTORY)]
    _PACKAGE_STUB.__spec__ = ModuleSpec(
        "hegel_machine", loader=None, is_package=True
    )
    sys.modules["hegel_machine"] = _PACKAGE_STUB
    sys.path_importer_cache[str(_PACKAGE_DIRECTORY)] = FileFinder(
        str(_PACKAGE_DIRECTORY),
        (_SourceOnlyLoaderV2, SOURCE_SUFFIXES),
    )
    _SOURCE_ONLY_IMPORTER_ACTIVE_V2 = True
    __package__ = "hegel_machine"

from .hashing import stable_hash
from .rfc8785_v1 import (
    RFC8785Error,
    canonicalize_rfc8785_v1,
    decode_and_require_canonical_rfc8785_v1,
)
from . import phase2b_formal_covert_auditor_v1 as auditor
from . import phase2b_exact_transform_core_v1 as exact_core
from . import phase2b_formal_trusted_wire_v3 as wire


PROFILE_SCHEMA_VERSION: Final = (
    "hegel-machine-phase2b-formal-wire-covert-qualification/6"
)
PROFILE_STATUS: Final = (
    "CANDIDATE_SOURCE_FROZEN_FULL_1024_SHAPE_CALIBRATION_PENDING"
)
PROFILE_CLAIM_LEVEL: Final = (
    "CANDIDATE_IMPLEMENTATION_EXERCISED_NOT_R03_R04_R05"
)
PROFILE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/FORMAL_WIRE_COVERT_QUALIFICATION/PROFILE/V2\x00"
)
PROFILE_ID_PREFIX: Final = (
    "phase2b_formal_wire_covert_qualification_profile_v2_"
)
QUALIFICATION_RECEIPT_DOMAIN: Final = (
    b"HEGEL/PHASE2B/FORMAL_WIRE_COVERT_QUALIFICATION/RECEIPT/V6\x00"
)
QUALIFICATION_RECEIPT_PREFIX: Final = (
    "phase2b_formal_wire_covert_qualification_receipt_v6_"
)
SYNTHETIC_FIXTURE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/FORMAL_WIRE_COVERT_QUALIFICATION/FIXTURE/V2\x00"
)
RUN_FIXTURE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/FORMAL_WIRE_COVERT_QUALIFICATION/RUN_FIXTURE/V2\x00"
)
SYNTHETIC_CASE_DOMAIN: Final = (
    b"HEGEL/PHASE2B/FORMAL_WIRE_COVERT_QUALIFICATION/CASE_ID/V2\x00"
)
SYNTHETIC_KAT_VECTOR_DOMAIN: Final = (
    b"HEGEL/PHASE2B/FORMAL_WIRE_COVERT_QUALIFICATION/KAT_VECTOR/V2\x00"
)

DEPENDENCY_LOCK_SHA256: Final = (
    "8c375133fb17d484d129464975ed4deb737d8f1e200d850f9ac75f4046bbfa07"
)
EXACT_AUTHORITY_ROWS: Final = 1_024
EXACT_RECOGNIZER_INVOCATIONS: Final = 81
EXACT_GLOBAL_RENAMINGS: Final = 32
EXACT_CASE_ORDER_PERMUTATIONS: Final = 32
EXACT_OBSERVATION_ORDER_PERMUTATIONS: Final = 16
EXACT_LABEL_PERMUTATIONS: Final = 10_000
LIVE_HARD_TOTAL_TIMEOUT_SECONDS: Final = 28_800
# Kernel containment headroom is operational only.  Auditor qualification and
# every explicit physical-peak gate remain frozen at the independent 12 GiB cap.
OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2: Final = 24 * 1_024 * 1_024 * 1_024
POST_OUTPUT_DIRECT_EXIT_TAIL_RESERVE_BYTES_V2: Final = 64 * 1_024 * 1_024
_MAXIMUM_SOURCE_ARCHIVE_PATH_DEPTH_V2: Final = 32
_MAXIMUM_SOURCE_ARCHIVE_TREE_ROWS_V2: Final = 80_000
_MAXIMUM_RETAINED_ATTEMPT_TREE_ROWS_V2: Final = 100_000
_MAXIMUM_RETAINED_ATTEMPT_TREE_BYTES_V2: Final = 2 * 1_024 * 1_024 * 1_024
_FINALIZATION_TREE_SCAN_PASSES_V2: Final = 2
_FINALIZATION_MINIMUM_STREAM_BYTES_PER_SECOND_V2: Final = 2 * 1_024 * 1_024
_FINALIZATION_MINIMUM_METADATA_ROWS_PER_SECOND_V2: Final = 1_000
_FINALIZATION_FSYNC_AND_PUBLICATION_MARGIN_SECONDS_V2: Final = 152
_OUTER_TERMINAL_PUBLICATION_RESERVE_SECONDS_V2: Final = (
    (
        _FINALIZATION_TREE_SCAN_PASSES_V2
        * _MAXIMUM_RETAINED_ATTEMPT_TREE_BYTES_V2
        + _FINALIZATION_MINIMUM_STREAM_BYTES_PER_SECOND_V2
        - 1
    )
    // _FINALIZATION_MINIMUM_STREAM_BYTES_PER_SECOND_V2
    + (
        _FINALIZATION_TREE_SCAN_PASSES_V2
        * _MAXIMUM_RETAINED_ATTEMPT_TREE_ROWS_V2
        + _FINALIZATION_MINIMUM_METADATA_ROWS_PER_SECOND_V2
        - 1
    )
    // _FINALIZATION_MINIMUM_METADATA_ROWS_PER_SECOND_V2
    + _FINALIZATION_FSYNC_AND_PUBLICATION_MARGIN_SECONDS_V2
)
_OUTER_CLEANUP_CONTROL_RESERVE_SECONDS_V2: Final = 600
_OUTER_LAUNCHER_FINALIZATION_RESERVE_SECONDS_V2: Final = 10
_OUTER_FAILURE_CLEANUP_RESERVE_SECONDS_V2: Final = (
    _OUTER_TERMINAL_PUBLICATION_RESERVE_SECONDS_V2
    + _OUTER_CLEANUP_CONTROL_RESERVE_SECONDS_V2
)
_MAX_MAIN_SYSTEMCTL_COMMAND_ROWS_V2: Final = 51
_MAX_FAILURE_CLEANUP_SUFFIX_ROWS_V2: Final = 40
_MAX_FAILURE_TRANSCRIPT_COMMAND_ROWS_V2: Final = (
    _MAX_MAIN_SYSTEMCTL_COMMAND_ROWS_V2
    + _MAX_FAILURE_CLEANUP_SUFFIX_ROWS_V2
)
_BOUNDED_COMMAND_REAP_RESERVE_NANOSECONDS_V2: Final = 3_000_000_000
_BOUNDED_COMMAND_TERM_GRACE_NANOSECONDS_V2: Final = 1_000_000_000
# Numeric process-group signalling is valid only inside this explicit outer
# TCB.  It is not an adversarial orphan-descendant ownership primitive.
OUTER_PROCESS_GROUP_OWNERSHIP_TCB_V2: Final = (
    "cooperative-pinned-direct-child-no-orphan-descendants"
)
_RFC8785_RANDOM_DIFFERENTIAL_SEED_V2: Final = 0x8785_2026_0814
LIVE_MODE_ORDER: Final = ("KNOWN_NULL", "KNOWN_BINDING_LEAK")
LIVE_SELECTED_DEFAULT_SIGNALS: Final = (
    signal.SIGHUP,
    signal.SIGINT,
    signal.SIGQUIT,
    signal.SIGALRM,
    signal.SIGTERM,
    signal.SIGCHLD,
)
EXACT_TRANSFORM_OPERATIONS: Final = tuple(
    item.value for item in exact_core.CoreTransformOperationV1
)
EXACT_ROWS_PER_TRANSFORM_OPERATION: Final = 128
SYNTHETIC_RECOGNIZER_ID: Final = (
    "hegel-machine-phase2b-synthetic-rich-v3-recognizer-adapter/2"
)

WIRE_SOURCE_SHA256: Final = (
    "c8120920f0e31b959043f63a0f672471fcbea50d05ca2cd8eaf7fd0c29089765"
)
WIRE_CONFIG_SHA256: Final = (
    "4f82ab060e2d53912cd17c8966a681a53d11e96ce3c6bb02ea3a42c5d2e85784"
)
WIRE_TEST_SHA256: Final = (
    "253001d11d7603456b80cf43837ed71de567bfd2cd6613cc78b21ca23f7db0f0"
)
WIRE_DOCUMENTATION_SHA256: Final = (
    "704a937967441ebbb9b2a77a865f83890e241d37031e2b0a87f1da2877b3456a"
)
AUDITOR_SOURCE_SHA256: Final = (
    "f4ea0bb54fd7b4f5707a6eea22ac8b728279e4f2e5559899626e4bf477d87312"
)
AUDITOR_CONFIG_SHA256: Final = (
    "e98052fd9e545ef17c577ab294388793a44502c91b22a1d784b41b86ecf54e83"
)
AUDITOR_TEST_SHA256: Final = (
    "0a383c06b4d3a7d048134e74f1465e85f2bc35add262bc065463e6b984d01b4f"
)
AUDITOR_DOCUMENTATION_SHA256: Final = (
    "07000f2f505731704392883062ac3d7ae1cd19411026e327f688ce441a7c6936"
)
AUDITOR_SCHEMA_VERSION_V2: Final = (
    "hegel-machine-phase2b-formal-covert-auditor/5"
)
AUDITOR_PROFILE_SCHEMA_VERSION_V2: Final = (
    "hegel-machine-phase2b-formal-covert-auditor-profile/5"
)
AUDITOR_SEMANTICS_ID_V2: Final = (
    "phase2b_formal_covert_semantics_"
    "d50564375ce978d2ee585662adc6d9ac5c2335aa662214f508897f1076918321"
)
AUDITOR_POLICY_BINDING_ID_V2: Final = (
    "phase2b_formal_covert_policy_binding_"
    "122bb24b8d8983a294a66e64c59fd74ac4017eaead1b7888141ff1679ac4272b"
)
if (
    auditor.SCHEMA_VERSION != AUDITOR_SCHEMA_VERSION_V2
    or auditor.SEMANTICS_ID != AUDITOR_SEMANTICS_ID_V2
    or auditor.POLICY_BINDING_ID != AUDITOR_POLICY_BINDING_ID_V2
):
    raise RuntimeError("qualification auditor schema/semantics/policy drifted")
RFC8785_SOURCE_SHA256: Final = (
    "16dcc6524a1c4a507edb9f85455c72b00fbc6c0c1b67471f03b67b2f25d32654"
)
RFC8785_TEST_SHA256: Final = (
    "ef0a0b06d78ec42702dd5b90179d9cd7767fc9c939ddad308a3485ec441f7df3"
)
RFC8785_DOCUMENTATION_SHA256: Final = (
    "4e1822ddcb861b269f8326f9584fad2f16e549b4a11d59d70836ef697d85b4e4"
)
RFC8785_GOLDEN_VECTORS_SHA256: Final = (
    "35135221efaf64756a471d19c0e53fdb13e703bad36b183708dec17b8a5d1f55"
)
RFC8785_NODE_ORACLE_SHA256: Final = (
    "297b3393eae6acbe9f6f4c857232aaa1acb1f7994cd096c1e38ab55769db1f31"
)
EXACT_CORE_SOURCE_SHA256: Final = (
    "e0fa517b4275a1d2daee1fd1f877b61a8f79ca745a702979615b2130e53bdc5f"
)
EXACT_CORE_TEST_SHA256: Final = (
    "15d0d6084743d21c2b7a79ec0d4f2f07c84ecfef641fb3d47f5253c95e5a3920"
)
HASHING_SOURCE_SHA256: Final = (
    "6fcd988ca4e8b75b9aa9e5a8b38c9ddd73e2685140e2cd0095d68f8bda054159"
)
PHASE2B_FREEZE_SOURCE_SHA256: Final = (
    "39a289ef0c000c1c62eb3a7a5f97ddfae3e51eb7904757642b0c2d167067f543"
)
SCHEMA_SOURCE_SHA256: Final = (
    "6497a93f8a68f3313d294bc99dc3b8f43508e93772d748521d09ea716b8ae10a"
)
BYPASSED_PACKAGE_INIT_SHA256: Final = (
    "38a9024bead0a7ebfd54362d57790c7e83ed14c189d084ad1fd97d9bca4140a3"
)
RFC8785_PREREQUISITE_SCHEMA_V2: Final = (
    "hegel-machine-rfc8785-runtime-prerequisite-receipt/2"
)
OUTER_SUPERVISOR_RESULT_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-formal-wire-covert-outer-supervisor-result/3"
)
MECHANICS_VERIFIER_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-formal-wire-covert-mechanics-verifier/2"
)
OUTER_ATTEMPT_SUCCESS_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-formal-wire-covert-outer-attempt-success/2"
)
OUTER_ATTEMPT_FAILURE_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-formal-wire-covert-outer-attempt-failure/4"
)
_OUTER_ATTEMPT_FAILURE_SCHEMA_V3: Final = (
    "hegel-machine-phase2b-formal-wire-covert-outer-attempt-failure/3"
)
_FAILURE_CHILD_PIPE_CAPTURE_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-formal-wire-covert-failure-child-pipe-capture/1"
)
_FAILURE_CAS_PROJECTION_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-formal-wire-covert-failure-cas-projection/1"
)
_FAILURE_CHILD_PIPE_CAPTURE_BUNDLE_DOMAIN_V2: Final = (
    b"HEGEL/PHASE2B/QV2/FAILURE_CHILD_PIPE_CAPTURE/V1\x00"
)
RUNTIME_ENVIRONMENT_MANIFEST_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-formal-wire-covert-runtime-environment/2"
)
OUTER_LAUNCH_INTENT_SCHEMA_V2: Final = (
    "hegel-machine-phase2b-formal-wire-covert-launch-intent/1"
)

FAIL_INPUT: Final = "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V2_INPUT"
FAIL_PROFILE: Final = "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V2_PROFILE"
FAIL_CANONICAL: Final = (
    "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V2_CANONICAL"
)
FAIL_CONTENT_ID: Final = (
    "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V2_CONTENT_ID"
)
FAIL_ISOLATED_EXECUTION: Final = (
    "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V2_ISOLATED_EXECUTION"
)
FAIL_WIRE: Final = "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V2_WIRE"
FAIL_AUDITOR: Final = "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V2_AUDITOR"
FAIL_KNOWN_NULL: Final = (
    "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V2_KNOWN_NULL"
)
FAIL_KNOWN_LEAK: Final = (
    "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V2_KNOWN_LEAK"
)
FAIL_RESOURCE: Final = (
    "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V2_RESOURCE"
)
FAIL_CLAIM_BOUNDARY: Final = (
    "FAIL_PHASE2B_FORMAL_WIRE_COVERT_QUALIFICATION_V2_CLAIM_BOUNDARY"
)

_MAXIMUM_PROFILE_BYTES: Final = 256 * 1_024
_ROOT_FIELDS: Final = frozenset(
    {"qualification_profile", "qualification_profile_id"}
)
_HEX64 = frozenset("0123456789abcdef")
_ZERO_DIGEST: Final = "sha256:" + "0" * 64
_RUN_MODES: Final = ("KNOWN_NULL", "KNOWN_BINDING_LEAK")
FROZEN_STATIC_RESOURCE_PLAN_FIELDS_V2: Final = (
    "rows",
    "concrete_features",
    "unique_statistical_partitions",
    "hypotheses",
    "maximum_feature_values",
    "maximum_label_classes",
    "bitset_words_per_mask",
    "permutation_passes",
    "label_permutations",
    "exchangeable_sample_size",
    "intersections_per_exchangeable_draw",
    "estimated_intersection_work_units",
    "bias_center_canonicalization_work_units",
    "scalar_omnibus_scan_work_units",
    "estimated_total_statistical_work_units",
    "maximum_label_shuffle_randbelow_calls",
    "maximum_shuffle_rejection_draws",
    "maximum_intersection_work_units",
    "maximum_total_statistical_work_units",
    "scalar_row_feature_permutation_work_avoided",
    "permutation_label_mask_bytes_bound",
    "metric_cache_bytes_bound",
    "worker_result_artifact_bytes_bound",
    "preextraction_feature_matrix_bytes_bound",
    "feature_matrix_bytes_bound",
    "envelope_and_decode_bytes_bound",
    "partition_allocation_bytes_bound",
    "partition_matrix_bytes_bound",
    "parent_result_working_bytes_bound",
    "adapter_retained_graph_bytes_bound",
    "invariance_variant_suite_output_bytes_bound",
    "variant_suite_resident_bytes_per_row_bound",
    "variant_suite_resident_fixed_bytes_bound",
    "invariance_variant_suite_resident_bytes_bound",
    "invariance_recognizer_live_output_bytes_bound",
    "invariance_worker_artifact_bytes_bound",
    "maximum_invariance_estimated_rss_bytes",
    "invariance_rss_formula",
    "maximum_estimated_rss_bytes",
    "aggregate_rss_formula",
    "host_available_memory_safety_margin_bytes",
    "partition_preparation_timeout_seconds",
    "invariance_worker_timeout_seconds",
    "statistics_worker_timeout_seconds",
    "statistics_execution_mode",
)
DYNAMIC_RESOURCE_PLAN_FIELDS_NOT_EXACT_COMPARED_V2: Final = (
    "parent_current_rss_bytes_at_plan",
    "invariance_parent_current_rss_bytes_at_plan",
    "invariance_resource_plan_root",
    "permutation_metric_tuple_bytes_bound",
    "target_max_vector_bytes_bound",
    "estimated_invariance_peak_rss_bytes",
    "estimated_peak_rss_bytes",
    "effective_cpu_capacity_at_preflight",
    "maximum_parallel_workers",
    "admissible",
    "resource_plan_root",
)
_RESOURCE_PLAN_FIELDS_V2: Final = frozenset(
    FROZEN_STATIC_RESOURCE_PLAN_FIELDS_V2
) | frozenset(DYNAMIC_RESOURCE_PLAN_FIELDS_NOT_EXACT_COMPARED_V2)
_RESOURCE_PLAN_INIT_FIELDS_V2: Final = tuple(
    item.name for item in dataclass_fields(auditor.ResourcePlanV1) if item.init
)
_PHASE_RESOURCE_PREFLIGHT_FIELDS_V2: Final = frozenset(
    {
        "phase",
        "resource_plan_root",
        "parent_current_rss_bytes_at_preflight",
        "static_parent_bytes_bound",
        "fork_parent_rss_multiplier",
        "additional_bytes_bound",
        "projected_aggregate_rss_bytes_at_preflight",
        "maximum_estimated_rss_bytes",
        "host_available_memory_safety_margin_bytes",
    }
)
_RECEIPT_ISSUANCE_RUNTIME_FIELDS_V2: Final = (
    "absolute_deadline_kind",
    "absolute_deadline_monotonic_nanoseconds",
    "absolute_deadline_timeout_seconds",
    "diagnostic_worker_historical_peak_rss_bytes_sum",
    "diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization",
    "diagnostic_parent_plus_worker_historical_peak_sum_bytes",
    "unified_cgroup_membership",
    "unified_cgroup_memory_scope_root",
    "unified_cgroup_leaf_device",
    "unified_cgroup_leaf_inode",
    "physical_memory_current_bytes_before_receipt_finalization",
    "physical_memory_peak_bytes_before_receipt_finalization",
    "physical_memory_max_bytes",
    "physical_memory_cap_bytes",
    "receipt_finalization_complete_pending_validation_count",
    "receipt_finalization_locked_deadline_check_count",
    "receipt_finalization_locked_deadline_within_absolute_limit",
    "receipt_finalization_locked_physical_peak_within_cap",
)
if (
    frozenset(FROZEN_STATIC_RESOURCE_PLAN_FIELDS_V2)
    & frozenset(DYNAMIC_RESOURCE_PLAN_FIELDS_NOT_EXACT_COMPARED_V2)
    or _RESOURCE_PLAN_FIELDS_V2
    != {item.name for item in dataclass_fields(auditor.ResourcePlanV1)}
):
    raise RuntimeError("qualification resource-plan field partition drifted")
if _PHASE_RESOURCE_PREFLIGHT_FIELDS_V2 != {
    item.name
    for item in dataclass_fields(auditor._PhaseResourcePreflightEvidenceV1)
}:
    raise RuntimeError("qualification phase resource-preflight schema drifted")
if tuple(
    item.name
    for item in dataclass_fields(auditor.ReceiptIssuanceRuntimeEvidenceV1)
) != _RECEIPT_ISSUANCE_RUNTIME_FIELDS_V2:
    raise RuntimeError("qualification receipt-issuance runtime schema drifted")

_TASK4_PROJECTION_ONLY_BOUNDARY_V2: Final = {
    "child_result_projection_validated_by_pinned_cooperative_code": True,
    "pure_verifier_recomputes_full_statistics": False,
    "full_result_or_worker_artifacts_retained": False,
    "promotable_to_R04_R05": False,
    "host_runtime_manifest_is_cooperative_fingerprint_not_full_runtime_closure": True,
    "systemd_user_manager_same_uid_is_cooperative_outer_tcb": True,
    "fresh_dedicated_user_service_manager_session_required": True,
    "existing_task4_artifact_cannot_be_promoted_by_signature_or_cas": True,
    "step5_must_rerun_and_retain_full_result_and_worker_artifacts": True,
    "step5_must_retain_full_runtime_image_and_import_native_closure": True,
}

_POST_OUTPUT_PHYSICAL_GATE_CONTRACT_V2: Final = {
    "canonical_stdout_write_then_flush_required": True,
    "cooperative_tail_headroom_not_formal_whole_lifetime_bound": True,
    "direct_os_exit_zero_required_after_final_gate": True,
    "final_memory_peak_fresh_reread_required": True,
    "fixed_direct_exit_tail_reserve_bytes": (
        POST_OUTPUT_DIRECT_EXIT_TAIL_RESERVE_BYTES_V2
    ),
    "full_post_output_cgroup_snapshot_required": True,
    "gc_disabled_no_trace_profile_or_python_signal_handlers_before_full_sample": True,
    "memory_events_required_zero": ["max", "oom", "oom_kill"],
    "memory_events_zero_claim_ends_at_full_post_output_sample": True,
    "numeric_post_output_peak_serialized": False,
    "physical_peak_evidence_ends_at_final_fresh_sample": True,
    "same_runtime_exit_cgroup_scope_and_events_required": True,
    "single_process_zero_descendants_at_full_sample": True,
    "success_peak_threshold_bytes": (
        auditor.MAXIMUM_ESTIMATED_RSS_BYTES
        - POST_OUTPUT_DIRECT_EXIT_TAIL_RESERVE_BYTES_V2
    ),
    "successful_return_or_interpreter_teardown_forbidden": True,
    "systemd_memory_max_bytes": OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2,
}
_DIRECT_PROCESS_EXIT_V2: Final = os._exit


_FIXTURE_OPERATION_SHAPE_CONTRACT_V2: Final = {
    "coordinate_affine": {
        "dimension": 1,
        "forward_equation": "target_0=2*source_0+1",
        "inverse_equation": "source_0=(target_0-1)/2",
        "source_frame": "frame:qualification-a",
        "source_support_lower": ["0"],
        "source_support_upper": ["1"],
        "target_frame": "frame:qualification-b",
        "target_support_lower": ["1"],
        "target_support_upper": ["3"],
    },
    "sampling_resolution": {
        "axis": "temporal",
        "boundary_policy": "valid_only",
        "discarded_input_coordinates": [["1"]],
        "grid_dimension": 1,
        "missing_policy": "reject",
        "output_is_exact_selected_metadata_copy": True,
        "selected_input_coordinates": [["0"]],
        "source_input_coordinates": [["0"], ["1"]],
    },
}


_FIXTURE_FEATURE_CAPACITY_CONTRACT_V2: Final = {
    "auditor_maximum_features": 32_768,
    "path_saturation_operation_rows": 8,
    "rows_with_identical_union_shape": [8, 32, 1_024],
    "modes": {
        "KNOWN_NULL": {
            "channel_only_owned_union_path_count": 229,
            "channel_only_owned_union_root": (
                "phase2b_formal_wire_covert_channel_only_owner_union_v2_"
                "063e3f71c9772e45e873452c9073ae8c2afb68c46f4f50ae51b25429c302bf4b"
            ),
            "union_shape": {
                "channel_union_path_count": 81,
                "descriptor_segment_position_count": 1,
                "descriptor_union_path_count": 1,
                "eligibility_absent_feature_count": 0,
                "generic_channel_union_path_count": 76,
                "provenance_union_path_count": 4,
                "uuid_union_path_count": 160,
            },
            "projections": [
                {
                    "projected_concrete_feature_count": 32_337,
                    "projected_feature_matrix_bytes_bound": 132_426_684,
                    "projected_raw_semantic_feature_count": 50_257,
                    "projection_root": (
                        "phase2b_formal_covert_feature_projection_"
                        "4b755abe8e386e07b497b51f1210bda8955c6bc8613dcbed5b31e77e40d61fbc"
                    ),
                    "rows": 8,
                },
                {
                    "projected_concrete_feature_count": 32_337,
                    "projected_feature_matrix_bytes_bound": 331_987_740,
                    "projected_raw_semantic_feature_count": 50_257,
                    "projection_root": (
                        "phase2b_formal_covert_feature_projection_"
                        "46122d52cf006777e6750f5478779997e791119503957980f9e305d38ad60d10"
                    ),
                    "rows": 32,
                },
                {
                    "projected_concrete_feature_count": 32_337,
                    "projected_feature_matrix_bytes_bound": 8_580_511_388,
                    "projected_raw_semantic_feature_count": 50_257,
                    "projection_root": (
                        "phase2b_formal_covert_feature_projection_"
                        "090d420d392e7df0a7dd7d4bbe896b01d589284bd2dbf1124e9b05ae45ef3110"
                    ),
                    "rows": 1_024,
                },
            ],
        },
        "KNOWN_BINDING_LEAK": {
            "channel_only_owned_union_path_count": 228,
            "channel_only_owned_union_root": (
                "phase2b_formal_wire_covert_channel_only_owner_union_v2_"
                "b58d3584425ca689772af6c701fb7b7c5769e9aa4ab267852bd0af3ccecef18e"
            ),
            "union_shape": {
                "channel_union_path_count": 81,
                "descriptor_segment_position_count": 1,
                "descriptor_union_path_count": 1,
                "eligibility_absent_feature_count": 0,
                "generic_channel_union_path_count": 76,
                "provenance_union_path_count": 4,
                "uuid_union_path_count": 159,
            },
            "projections": [
                {
                    "projected_concrete_feature_count": 32_186,
                    "projected_feature_matrix_bytes_bound": 131_777_640,
                    "projected_raw_semantic_feature_count": 49_994,
                    "projection_root": (
                        "phase2b_formal_covert_feature_projection_"
                        "667ff7f915d808e589d34c316042c112f9e2923de49d93f6c59a26f8cd979009"
                    ),
                    "rows": 8,
                },
                {
                    "projected_concrete_feature_count": 32_186,
                    "projected_feature_matrix_bytes_bound": 330_410_952,
                    "projected_raw_semantic_feature_count": 49_994,
                    "projection_root": (
                        "phase2b_formal_covert_feature_projection_"
                        "65034b47914ecf4a87a4561e7474f130bfa1d6633b6d93e46b35bb5370247d8b"
                    ),
                    "rows": 32,
                },
                {
                    "projected_concrete_feature_count": 32_186,
                    "projected_feature_matrix_bytes_bound": 8_540_587_848,
                    "projected_raw_semantic_feature_count": 49_994,
                    "projection_root": (
                        "phase2b_formal_covert_feature_projection_"
                        "59d3b166ca6eb860c0a9025c9fdf1d1dbbf60a46180843606f143fe4bb8b61f7"
                    ),
                    "rows": 1_024,
                },
            ],
        },
    },
}


if (
    _FIXTURE_FEATURE_CAPACITY_CONTRACT_V2["auditor_maximum_features"]
    != auditor.MAXIMUM_FEATURES
    or max(
        projection["projected_concrete_feature_count"]
        for mode in _FIXTURE_FEATURE_CAPACITY_CONTRACT_V2["modes"].values()
        for projection in mode["projections"]
    )
    > auditor.MAXIMUM_FEATURES
):
    raise RuntimeError("fixture feature-capacity contract exceeds the pinned auditor")


_EXPECTED_PROFILE: Final = {
    "auditor_binding": {
        "balanced_accuracy_advantage_limit": "0.05",
        "channel_only_owner_groups": list(auditor.CHANNEL_ONLY_OWNER_GROUPS),
        "exact_case_order_permutations": auditor.GLOBAL_CASE_ORDER_PERMUTATIONS,
        "exact_global_renamings": auditor.GLOBAL_RENAMINGS,
        "exact_label_permutations": auditor.LABEL_PERMUTATIONS,
        "exact_observation_order_permutations": (
            auditor.OBSERVATION_ORDER_PERMUTATIONS
        ),
        "family_wise_alpha": "0.01",
        "full_raw_byte_categorical_hypotheses_required": True,
        "maximum_static_phase_plan_rss_bytes": (
            auditor.MAXIMUM_ESTIMATED_RSS_BYTES
        ),
        "execution_scope_physical_memory_peak_cap_bytes": (
            auditor.MAXIMUM_ESTIMATED_RSS_BYTES
        ),
        "maximum_features": auditor.MAXIMUM_FEATURES,
        "maximum_rows": auditor.MAXIMUM_ROWS,
        "maximum_statistical_intersection_work_units": (
            auditor.MAXIMUM_STATISTICAL_INTERSECTION_WORK_UNITS
        ),
        "maximum_total_statistical_work_units": (
            auditor.MAXIMUM_TOTAL_STATISTICAL_WORK_UNITS
        ),
        "nmi_limit": "0.02",
        "pairwise_holm_decision_eligible": False,
        "positive_decision_claim": auditor.POSITIVE_DECISION_CLAIM,
        "policy_binding_id": AUDITOR_POLICY_BINDING_ID_V2,
        "profile_schema_version": AUDITOR_PROFILE_SCHEMA_VERSION_V2,
        "schema_version": AUDITOR_SCHEMA_VERSION_V2,
        "semantics_id": AUDITOR_SEMANTICS_ID_V2,
        "receipt_issuance_runtime_evidence_fields": list(
            _RECEIPT_ISSUANCE_RUNTIME_FIELDS_V2
        ),
        "receipt_issuance_runtime_evidence_required": True,
        "receipt_issuance_runtime_evidence_branch_contract": (
            "ONE_UNIFIED_CGROUP_V2_ATTEMPT_LEAF_MEMORY_PEAK_AT_OR_BELOW_12GIB_"
            "WITH_SCOPE_STABILITY_AND_ONE_LOCKED_PENDING_VALIDATION"
        ),
        "statistics_runtime_evidence_required": True,
        "target_family_inference": (
            "B_PLUS_1_10001_TARGET_MAXT_THEN_EXACT_RATIONAL_FIVE_TARGET_HOLM"
        ),
        "targets": list(auditor.TARGETS),
    },
    "canonicalization": {
        "config_transport": "RFC8785_CANONICAL_BYTES_PLUS_ONE_LF",
        "implementation": "hegel_machine.rfc8785_v1",
        "runtime_qualification_requires": (
            "BOUND_PREREQUISITE_RECEIPT_FOR_OFFICIAL_RFC8785_VECTORS_AND_"
            "INDEPENDENT_NODE_V8_BINARY64_DIFFERENTIAL"
        ),
    },
    "claim_level": PROFILE_CLAIM_LEVEL,
    "dependency_lock": {
        "path": "config/phase2b_formal_wire_covert_qualification_v2.lock.json",
        "schema_version": (
            "hegel-machine-phase2b-formal-wire-covert-qualification-"
            "dependency-lock/3"
        ),
        "sha256": DEPENDENCY_LOCK_SHA256,
        "standard_library_only": True,
        "standard_library_only_scope": "PYTHON_THIRD_PARTY_PACKAGES_ONLY",
        "host_runtime_manifest_scope": (
            "COOPERATIVE_FINGERPRINT_NOT_FULL_LIBPYTHON_STDLIB_NATIVE_CLOSURE"
        ),
        "step5_full_runtime_image_and_import_native_closure_required": True,
        "broad_package_init_bypassed": True,
        "bypassed_package_init_sha256_non_tcb": BYPASSED_PACKAGE_INIT_SHA256,
        "direct_file_open_once_bootstrap_required": True,
        "python_argv_flags": ["-I", "-S", "-B"],
        "source_only_loader_required": True,
        "runtime_import_closure_sha256": {
            "src/hegel_machine/hashing.py": HASHING_SOURCE_SHA256,
            "src/hegel_machine/phase2b_exact_transform_core_v1.py": (
                EXACT_CORE_SOURCE_SHA256
            ),
            "src/hegel_machine/phase2b_formal_covert_auditor_v1.py": (
                AUDITOR_SOURCE_SHA256
            ),
            "src/hegel_machine/phase2b_formal_trusted_wire_v3.py": (
                WIRE_SOURCE_SHA256
            ),
            "src/hegel_machine/phase2b_freeze_v1.py": (
                PHASE2B_FREEZE_SOURCE_SHA256
            ),
            "src/hegel_machine/rfc8785_v1.py": RFC8785_SOURCE_SHA256,
            "src/hegel_machine/schema.py": SCHEMA_SOURCE_SHA256,
        },
    },
    "fixture_contract": {
        "authority_rows": EXACT_AUTHORITY_ROWS,
        "case_graph": (
            "CAPACITY_BOUNDED_EXACT_10_EIGHT_OPERATION_FORMAL_CASE_GRAPH_V4"
        ),
        "feature_capacity_contract": _FIXTURE_FEATURE_CAPACITY_CONTRACT_V2,
        "input_class": "SYNTHETIC_DEV_ONLY",
        "known_leak": (
            "LEGAL_ELIGIBILITY_OBSERVATION_COUNT_DEPENDS_ONLY_ON_PRIVATE_BINDING"
        ),
        "known_leak_expected_disposition": "FAIL",
        "known_null_expected_disposition": "PASS",
        "fixture_specific_only": True,
        "joint_decision_classes": 2,
        "private_label_factorial": (
            "2_CASE_TYPE_X_2_FAMILY_X_2_SCALE_X_2_BINDING_X_"
            "2_ANSWERABILITY_X_2_JOINT_X_16_REPEATS"
        ),
        "recognizer_adapter_invocations_per_audit": (
            EXACT_RECOGNIZER_INVOCATIONS
        ),
        "operation_shape_contract": _FIXTURE_OPERATION_SHAPE_CONTRACT_V2,
        "rich_positive_shape": {
            "calibration_artifact_cannot_be_promoted": True,
            "calibration_artifact_cas_root": None,
            "calibration_artifact_sha256": None,
            "calibration_artifact_size_bytes": None,
            "calibration_attempt_id": None,
            "calibration_attempt_manifest_cas_root": None,
            "calibration_attempt_manifest_root": None,
            "calibration_attempt_manifest_sha256": None,
            "calibration_attempt_manifest_size_bytes": None,
            "calibration_basis_profile_cas_root": None,
            "calibration_basis_profile_id": None,
            "calibration_basis_profile_sha256": None,
            "calibration_basis_profile_size_bytes": None,
            "calibration_campaign_id": None,
            "calibration_mechanics_artifact_cas_root": None,
            "calibration_mechanics_artifact_sha256": None,
            "calibration_mechanics_artifact_size_bytes": None,
            "calibration_outer_supervisor_result_cas_root": None,
            "calibration_outer_supervisor_result_sha256": None,
            "calibration_outer_supervisor_result_size_bytes": None,
            "calibration_rfc8785_receipt_cas_root": None,
            "calibration_rfc8785_receipt_sha256": None,
            "calibration_rfc8785_receipt_size_bytes": None,
            "calibration_campaign_status": (
                "PENDING_FULL_1024_DEVELOPMENT_MEASUREMENT"
            ),
            "combined_fixture_root": None,
            "dynamic_resource_plan_fields_not_exact_compared": list(
                DYNAMIC_RESOURCE_PLAN_FIELDS_NOT_EXACT_COMPARED_V2
            ),
            "fresh_qualification_must_equal_frozen_calibration": True,
            "known_leak_frozen_static_projection": None,
            "known_null_frozen_static_projection": None,
            "live_qualification_enabled": False,
            "old_identity_only_counts_valid": False,
            "static_resource_plan_fields": list(
                FROZEN_STATIC_RESOURCE_PLAN_FIELDS_V2
            ),
        },
        "transform_operation_rows": {
            operation: EXACT_ROWS_PER_TRANSFORM_OPERATION
            for operation in EXACT_TRANSFORM_OPERATIONS
        },
        "wire_wider_graphs_and_100_class_profile_not_qualified": True,
    },
    "readiness_boundary": {
        "candidate_disposition": PROFILE_CLAIM_LEVEL,
        "dependencies_in_order": [
            "R02_EXECUTION_BRANCH_BINDINGS_REPLAYED",
            "R03_FORMAL_TRUSTED_WIRE_IMPLEMENTATION_QUALIFIED",
            "R04_FORMAL_COVERT_AUDITOR_IMPLEMENTATION_QUALIFIED",
            "R05_SYNTHETIC_DEV_WIRE_COVERT_QUALIFICATION_VERIFIED",
        ],
        "host_tests_or_uncommitted_source_satisfy_r03_r04_r05": False,
        "later_offline_cli_private_runtime_token_required": True,
        "live_execution_contract": {
            "attempt_manifest_frozen_before_fixture_read": True,
            "attempt_manifest_fields": [
                "campaign_id",
                "attempt_id",
                "profile_id",
                "source_commit_sha",
                "source_archive_root",
                "source_archive_size_bytes",
                "dependency_lock_sha256",
                "qualification_sha256",
                "runtime_environment_manifest_root",
                "runtime_environment_manifest_size_bytes",
                "runtime_image_digest",
                "rfc8785_prerequisite_receipt_root",
                "rfc8785_prerequisite_receipt_sha256",
                "rfc8785_prerequisite_receipt_size_bytes",
                "mode_order",
                "hard_total_timeout_seconds",
                "requested_cgroup_memory_max_bytes",
                "requested_cgroup_pids_max",
            ],
            "attempt_manifest_fields_are_exact_top_level_projection": True,
            "cleanup_control_reserve_seconds": (
                _OUTER_CLEANUP_CONTROL_RESERVE_SECONDS_V2
            ),
            "default_signal_dispositions_required": True,
            "finite_cgroup_v2_memory_max_required": True,
            "finite_cgroup_v2_pids_max_required": True,
            "fresh_dedicated_user_service_manager_session_required": True,
            "fresh_isolated_outer_process_required": True,
            "hard_end_to_end_deadline_required": True,
            "hard_total_timeout_seconds": LIVE_HARD_TOTAL_TIMEOUT_SECONDS,
            "entry_exit_memory_peak_at_or_below_auditor_cap": True,
            "entry_exit_memory_peak_monotonic": True,
            "receipt_scope_and_prefinal_peak_crossbound_to_exit": True,
            "requested_memory_max_equals_auditor_physical_cap": False,
            "post_output_physical_gate_contract": dict(
                _POST_OUTPUT_PHYSICAL_GATE_CONTRACT_V2
            ),
            "post_exit_numeric_memory_peak_sample_claimed": False,
            "mode_order": list(LIVE_MODE_ORDER),
            "outer_process_group_ownership_tcb": (
                OUTER_PROCESS_GROUP_OWNERSHIP_TCB_V2
            ),
            "adversarial_orphan_descendant_cleanup_claimed": False,
            "outer_mechanics_disposition": "MECHANICS_VERIFIED_NOT_AUTHORITY",
            "outer_mechanics_schema": MECHANICS_VERIFIER_SCHEMA_V2,
            "outer_mechanics_verifier_issues_r03_r04_r05": False,
            "outer_supervisor_result_schema": OUTER_SUPERVISOR_RESULT_SCHEMA_V2,
            "systemd_user_manager_same_uid_is_cooperative_outer_tcb": True,
            "terminal_publication_reserve_seconds": (
                _OUTER_TERMINAL_PUBLICATION_RESERVE_SECONDS_V2
            ),
            "work_deadline_reserve_seconds": (
                _OUTER_FAILURE_CLEANUP_RESERVE_SECONDS_V2
            ),
            "rfc8785_canonical_receipt_bytes_required": True,
            "cpu_max_required": False,
            "cpu_affinity_and_capacity_recorded": True,
            "task4_issues_r03_r04_r05": False,
            "retry_after_statistical_false_reject": (
                "NEW_ATTEMPT_ID_RETAIN_PRIOR_FAILURE_NEVER_RETRY_IN_PLACE"
            ),
        },
        "task4_projection_only_boundary": dict(
            _TASK4_PROJECTION_ONLY_BOUNDARY_V2
        ),
    },
    "schema_version": PROFILE_SCHEMA_VERSION,
    "status": PROFILE_STATUS,
    "wire_binding": {
        "case_graph_completeness_policy_id": (
            wire.FORMAL_CASE_GRAPH_COMPLETENESS_POLICY_ID_V3
        ),
        "case_graph_schema_version": wire.FORMAL_CASE_GRAPH_V3_SCHEMA_VERSION,
        "envelope_bytes": wire.ENVELOPE_BYTES_V3,
        "field_classification_manifest_id": (
            wire.FIELD_CLASSIFICATION_MANIFEST_ID_V3
        ),
        "global_transform_catalog_binding_schema": (
            wire.FORMAL_GLOBAL_TRANSFORM_CATALOG_BINDING_V3_SCHEMA_VERSION
        ),
        "global_transform_catalog_schema": (
            wire.FORMAL_GLOBAL_TRANSFORM_CATALOG_V3_SCHEMA_VERSION
        ),
        "maximum_batch_cases": wire.MAXIMUM_BATCH_CASES_V3,
        "namespaces_in_order": list(wire.NAMESPACE_REGISTRY_V3),
        "payload_schema_version": (
            wire.FORMAL_TRUSTED_WIRE_V3_PAYLOAD_SCHEMA_VERSION
        ),
        "schema_version": wire.FORMAL_TRUSTED_WIRE_V3_SCHEMA_VERSION,
        "variant_suite_schema_version": (
            wire.FORMAL_VARIANT_SUITE_V3_SCHEMA_VERSION
        ),
        "pinned_artifact_sha256": {
            "auditor_config": AUDITOR_CONFIG_SHA256,
            "auditor_documentation": AUDITOR_DOCUMENTATION_SHA256,
            "auditor_source": AUDITOR_SOURCE_SHA256,
            "auditor_test": AUDITOR_TEST_SHA256,
            "wire_config": WIRE_CONFIG_SHA256,
            "wire_documentation": WIRE_DOCUMENTATION_SHA256,
            "wire_source": WIRE_SOURCE_SHA256,
            "wire_test": WIRE_TEST_SHA256,
        },
    },
}


class Phase2BFormalWireCovertQualificationErrorV2(ValueError):
    """Closed qualification input/profile error."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _fail(code: str, detail: str) -> NoReturn:
    raise Phase2BFormalWireCovertQualificationErrorV2(code, detail)


def _deep_freeze(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if type(value) is list:
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_deep_thaw(item) for item in value]
    return value


def _restore_json_integers(value: object) -> object:
    if type(value) is float:
        if not value.is_integer() or abs(value) > 2**53:
            _fail(FAIL_PROFILE, "qualification profile contains a non-safe integer")
        return int(value)
    if type(value) is list:
        return [_restore_json_integers(item) for item in value]
    if type(value) is dict:
        return {key: _restore_json_integers(item) for key, item in value.items()}
    return value


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _exact_json_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_json_equal(item, expected)
            for item, expected in zip(left, right, strict=True)
        )
    return left == right


def _decode_canonical_transport_object_v2(
    payload: bytes,
    name: str,
    *,
    maximum_bytes: int,
) -> dict[str, object]:
    if (
        type(payload) is not bytes
        or not 2 <= len(payload) <= maximum_bytes
        or not payload.endswith(b"\n")
        or payload.endswith(b"\n\n")
    ):
        raise ValueError(f"{name} must be bounded canonical bytes plus one LF")
    try:
        decoded = decode_and_require_canonical_rfc8785_v1(payload[:-1])
    except (RFC8785Error, TypeError) as exc:
        raise ValueError(f"{name} is not exact RFC 8785: {exc}") from exc
    decoded = _restore_json_integers(decoded)
    if type(decoded) is not dict:
        raise ValueError(f"{name} root must be an exact object")
    if canonicalize_rfc8785_v1(decoded) + b"\n" != payload:
        raise ValueError(f"{name} canonical replay drift")
    return decoded


def qualification_profile_id_v2(profile: object) -> str:
    if not _exact_json_equal(profile, _EXPECTED_PROFILE):
        _fail(FAIL_PROFILE, "qualification profile differs from the closed V2 profile")
    try:
        encoded = canonicalize_rfc8785_v1(profile)  # type: ignore[arg-type]
    except (RFC8785Error, TypeError) as exc:
        _fail(FAIL_CANONICAL, f"profile is outside RFC 8785: {exc}")
    return PROFILE_ID_PREFIX + hashlib.sha256(PROFILE_DOMAIN + encoded).hexdigest()


PROFILE_ID: Final = qualification_profile_id_v2(_EXPECTED_PROFILE)


@dataclass(frozen=True, slots=True, init=False)
class Phase2BFormalWireCovertQualificationProfileV2:
    profile_id: str
    profile: Mapping[str, object] = field(repr=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("qualification profiles are privately issued by the loader")

    @classmethod
    def _issue(
        cls, profile: dict[str, object]
    ) -> "Phase2BFormalWireCovertQualificationProfileV2":
        value = object.__new__(cls)
        object.__setattr__(value, "profile_id", qualification_profile_id_v2(profile))
        object.__setattr__(value, "profile", _deep_freeze(profile))
        value._validate()
        return value

    def _validate(self) -> None:
        if type(self) is not Phase2BFormalWireCovertQualificationProfileV2:
            raise TypeError("qualification profile exact type drift")
        if self.profile_id != PROFILE_ID:
            raise ValueError("qualification profile ID drift")
        if qualification_profile_id_v2(_deep_thaw(self.profile)) != self.profile_id:
            raise ValueError("qualification profile content drift")


def load_phase2b_formal_wire_covert_qualification_profile_v2(
    payload: bytes,
) -> Phase2BFormalWireCovertQualificationProfileV2:
    """Load the single closed RFC-8785 profile without filesystem access."""

    if type(payload) is not bytes or not 1 <= len(payload) <= _MAXIMUM_PROFILE_BYTES:
        _fail(FAIL_INPUT, "profile input must be bounded exact bytes")
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        _fail(FAIL_CANONICAL, "profile transport needs exactly one terminal LF")
    canonical = payload[:-1]
    try:
        decoded = decode_and_require_canonical_rfc8785_v1(canonical)
    except (RFC8785Error, TypeError) as exc:
        _fail(FAIL_CANONICAL, f"profile is not exact RFC 8785: {exc}")
    decoded = _restore_json_integers(decoded)
    if type(decoded) is not dict or set(decoded) != _ROOT_FIELDS:
        _fail(FAIL_PROFILE, "profile root is not the exact closed two-field object")
    profile = decoded["qualification_profile"]
    profile_id = decoded["qualification_profile_id"]
    if not _exact_json_equal(profile, _EXPECTED_PROFILE):
        _fail(FAIL_PROFILE, "qualification profile content drift")
    expected_id = qualification_profile_id_v2(profile)
    if type(profile_id) is not str or profile_id != expected_id or profile_id != PROFILE_ID:
        _fail(FAIL_CONTENT_ID, "qualification profile content ID mismatch")
    if canonicalize_rfc8785_v1(decoded) + b"\n" != payload:
        _fail(FAIL_CANONICAL, "qualification profile transport bytes drift")
    return Phase2BFormalWireCovertQualificationProfileV2._issue(profile)


def phase2b_formal_wire_covert_qualification_profile_bytes_v2() -> bytes:
    """Return the exact checked-in profile bytes for independent comparison."""

    root = {
        "qualification_profile": _EXPECTED_PROFILE,
        "qualification_profile_id": PROFILE_ID,
    }
    return canonicalize_rfc8785_v1(root) + b"\n"


def _uuid_v4(namespace: str, row_index: int, slot: int = 0) -> str:
    if (
        type(namespace) is not str
        or not namespace
        or type(row_index) is not int
        or not 0 <= row_index < 2**32
        or type(slot) is not int
        or not 0 <= slot < 2**32
    ):
        raise ValueError("synthetic UUID coordinates are outside bounds")
    digest = hashlib.sha256(
        SYNTHETIC_CASE_DOMAIN
        + namespace.encode("ascii")
        + row_index.to_bytes(4, "big")
        + slot.to_bytes(4, "big")
    ).digest()
    value = bytearray(digest[:16])
    value[6] = (value[6] & 0x0F) | 0x40
    value[8] = (value[8] & 0x3F) | 0x80
    text = value.hex()
    return (
        f"{text[:8]}-{text[8:12]}-{text[12:16]}-"
        f"{text[16:20]}-{text[20:]}"
    )


def _ordered_case_uuid_v2(row_index: int, slot: int) -> str:
    """Label-blind domain-separated UUIDv4 for the global object schedule."""

    if (
        type(row_index) is not int
        or type(slot) is not int
        or not 0 <= row_index < EXACT_AUTHORITY_ROWS
        or not 1 <= slot < 1_000_000
    ):
        raise ValueError("ordered synthetic UUID coordinates are outside bounds")
    return _uuid_v4("global-object-schedule-v2", row_index, slot)


_F = Fraction
_SOURCE_SCALE = exact_core.CoreScaleRefV1("source")
_TARGET_SCALE = exact_core.CoreScaleRefV1("target")
_CLOCK = exact_core.CoreClockBindingV1("clock:qualification-v2")
_FRAME_A = exact_core.CoreFrameBindingV1("frame:qualification-a", "right-handed")
_FRAME_B = exact_core.CoreFrameBindingV1("frame:qualification-b", "right-handed")
_UNIT_ONE = exact_core.CoreUnitBindingV1("unit:one", _F(1), _F(0))
_UNIT_CENTI = exact_core.CoreUnitBindingV1("unit:centi", _F(1, 100), _F(0))
_DIMENSIONLESS = exact_core.CoreQuantityMetadataV1(
    "quantity:dimensionless", (0, 0, 0, 0, 0, 0, 0), _UNIT_ONE
)
_LENGTH_ONE = exact_core.CoreQuantityMetadataV1(
    "quantity:length", (1, 0, 0, 0, 0, 0, 0), _UNIT_ONE
)
_LENGTH_CENTI = exact_core.CoreQuantityMetadataV1(
    "quantity:length", (1, 0, 0, 0, 0, 0, 0), _UNIT_CENTI
)


def _core_ref_v2(
    scale: exact_core.CoreScaleRefV1,
    observation: str,
    index: int = 0,
) -> exact_core.CoreComponentRefV1:
    return exact_core.CoreComponentRefV1(
        scale, exact_core.CoreObservationRefV1(observation), index
    )


def _core_metadata_v2(
    component_ref: exact_core.CoreComponentRefV1,
    *,
    label: str = "value",
    quantity: exact_core.CoreQuantityMetadataV1 = _DIMENSIONLESS,
    axis: exact_core.CoreComponentAxisV1 = exact_core.CoreComponentAxisV1.SCALAR,
    role: exact_core.CoreComponentValueRoleV1 = (
        exact_core.CoreComponentValueRoleV1.INTENSIVE
    ),
    temporal: exact_core.CoreTemporalSupportV1 | None = None,
    spatial: exact_core.CoreSpatialSupportV1 | None = None,
) -> exact_core.CoreComponentMetadataV1:
    return exact_core.CoreComponentMetadataV1(
        component_ref,
        label,
        ("entity:subject",),
        ("role:measured",),
        quantity,
        axis,
        role,
        temporal,
        spatial,
    )


def _core_numeric_v2(
    metadata: exact_core.CoreComponentMetadataV1,
    value: Fraction,
    radius: Fraction,
) -> exact_core.CoreComponentValueV1:
    return exact_core.CoreComponentValueV1(
        metadata,
        exact_core.CoreComponentValueKindV1.NUMERIC,
        value,
        exact_core.CoreUncertaintyModelV1.ABSOLUTE_BOUND,
        radius,
    )


def _core_row_v2(
    output_ref: exact_core.CoreComponentRefV1,
    terms: tuple[tuple[exact_core.CoreComponentRefV1, Fraction], ...],
    offset: Fraction = _F(0),
) -> exact_core.CoreExactRowV1:
    return exact_core.CoreExactRowV1(
        output_ref,
        tuple(
            exact_core.CoreExactTermV1(input_ref, coefficient)
            for input_ref, coefficient in sorted(terms)
        ),
        offset,
    )


def _core_group_v2(
    inputs: tuple[exact_core.CoreComponentRefV1, ...],
    outputs: tuple[exact_core.CoreComponentRefV1, ...],
    label: str,
) -> exact_core.CoreAggregateGroupV1:
    return exact_core.CoreAggregateGroupV1(
        tuple(sorted(inputs)), tuple(sorted(outputs)), label
    )


def _core_profile_v2(
    *,
    operation: exact_core.CoreTransformOperationV1,
    sources: tuple[exact_core.CoreComponentValueV1, ...],
    outputs: tuple[exact_core.CoreComponentMetadataV1, ...],
    rows: tuple[exact_core.CoreExactRowV1, ...],
    certificate: exact_core.CoreOperationCertificateV1,
    mappings: tuple[exact_core.CoreDiscreteMappingV1, ...] = (),
) -> exact_core.ExactTransformCoreProfileV1:
    return exact_core.ExactTransformCoreProfileV1(
        tuple(sorted(sources, key=lambda item: item.ref)),
        exact_core.CoreExactTransformContractV1(
            operation,
            _SOURCE_SCALE,
            _TARGET_SCALE,
            tuple(sorted(item.ref for item in sources)),
            tuple(sorted(outputs, key=lambda item: item.ref)),
            tuple(sorted(rows, key=lambda item: item.output_ref)),
            tuple(sorted(mappings)),
            certificate,
        ),
    )


def _core_fixture_profile_v2(
    operation: exact_core.CoreTransformOperationV1,
) -> exact_core.ExactTransformCoreProfileV1:
    if operation is exact_core.CoreTransformOperationV1.IDENTITY:
        source = _core_ref_v2(_SOURCE_SCALE, "identity")
        target = _core_ref_v2(_TARGET_SCALE, "identity")
        source_metadata = _core_metadata_v2(source)
        target_metadata = replace(source_metadata, ref=target)
        return _core_profile_v2(
            operation=operation,
            sources=(_core_numeric_v2(source_metadata, _F(3, 2), _F(1, 10)),),
            outputs=(target_metadata,),
            rows=(_core_row_v2(target, ((source, _F(1)),)),),
            certificate=exact_core.CoreIdentityCertificateV1(),
        )
    if operation is exact_core.CoreTransformOperationV1.UNIT_CONVERSION:
        source = _core_ref_v2(_SOURCE_SCALE, "unit")
        target = _core_ref_v2(_TARGET_SCALE, "unit")
        source_metadata = _core_metadata_v2(source, quantity=_LENGTH_ONE)
        target_metadata = _core_metadata_v2(target, quantity=_LENGTH_CENTI)
        return _core_profile_v2(
            operation=operation,
            sources=(_core_numeric_v2(source_metadata, _F(2), _F(1, 10)),),
            outputs=(target_metadata,),
            rows=(_core_row_v2(target, ((source, _F(100)),)),),
            certificate=exact_core.CoreUnitConversionCertificateV1(
                _F(100), _F(1, 100)
            ),
        )
    if operation is exact_core.CoreTransformOperationV1.COORDINATE_AFFINE:
        source_refs = (_core_ref_v2(_SOURCE_SCALE, "coordinate-source", 0),)
        target_refs = (_core_ref_v2(_TARGET_SCALE, "coordinate-target", 0),)
        source_support = exact_core.CoreSpatialSupportV1(
            _FRAME_A, (_F(0),), (_F(1),)
        )
        target_support = exact_core.CoreSpatialSupportV1(
            _FRAME_B, (_F(1),), (_F(3),)
        )
        source_metadata = tuple(
            _core_metadata_v2(
                ref,
                label=f"coordinate-{index}",
                quantity=_LENGTH_ONE,
                axis=exact_core.CoreComponentAxisV1.COORDINATE,
                role=exact_core.CoreComponentValueRoleV1.COORDINATE,
                spatial=source_support,
            )
            for index, ref in enumerate(source_refs)
        )
        target_metadata = tuple(
            _core_metadata_v2(
                ref,
                label=f"coordinate-{index}",
                quantity=_LENGTH_ONE,
                axis=exact_core.CoreComponentAxisV1.COORDINATE,
                role=exact_core.CoreComponentValueRoleV1.COORDINATE,
                spatial=target_support,
            )
            for index, ref in enumerate(target_refs)
        )
        forward = (
            _core_row_v2(
                target_refs[0],
                ((source_refs[0], _F(2)),),
                _F(1),
            ),
        )
        inverse = (
            _core_row_v2(
                source_refs[0],
                ((target_refs[0], _F(1, 2)),),
                _F(-1, 2),
            ),
        )
        return _core_profile_v2(
            operation=operation,
            sources=(_core_numeric_v2(source_metadata[0], _F(1, 2), _F(1, 10)),),
            outputs=target_metadata,
            rows=forward,
            certificate=exact_core.CoreCoordinateAffineCertificateV1(
                _FRAME_A, _FRAME_B, 1, inverse
            ),
        )
    if operation is exact_core.CoreTransformOperationV1.TEMPORAL_AGGREGATION:
        source_refs = (
            _core_ref_v2(_SOURCE_SCALE, "temporal-a"),
            _core_ref_v2(_SOURCE_SCALE, "temporal-b"),
        )
        target = _core_ref_v2(_TARGET_SCALE, "temporal")
        source_metadata = (
            _core_metadata_v2(
                source_refs[0],
                label="flow",
                axis=exact_core.CoreComponentAxisV1.TEMPORAL,
                role=exact_core.CoreComponentValueRoleV1.EXTENSIVE,
                temporal=exact_core.CoreTemporalSupportV1(_CLOCK, _F(0), _F(1)),
            ),
            _core_metadata_v2(
                source_refs[1],
                label="flow",
                axis=exact_core.CoreComponentAxisV1.TEMPORAL,
                role=exact_core.CoreComponentValueRoleV1.EXTENSIVE,
                temporal=exact_core.CoreTemporalSupportV1(_CLOCK, _F(1), _F(2)),
            ),
        )
        target_metadata = _core_metadata_v2(
            target,
            label="flow",
            axis=exact_core.CoreComponentAxisV1.TEMPORAL,
            role=exact_core.CoreComponentValueRoleV1.EXTENSIVE,
            temporal=exact_core.CoreTemporalSupportV1(_CLOCK, _F(0), _F(2)),
        )
        group = _core_group_v2(source_refs, (target,), "temporal-bin")
        return _core_profile_v2(
            operation=operation,
            sources=(
                _core_numeric_v2(source_metadata[0], _F(1), _F(1, 10)),
                _core_numeric_v2(source_metadata[1], _F(2), _F(1, 5)),
            ),
            outputs=(target_metadata,),
            rows=(
                _core_row_v2(
                    target,
                    ((source_refs[0], _F(1)), (source_refs[1], _F(1))),
                ),
            ),
            certificate=exact_core.CoreTemporalAggregationCertificateV1(
                exact_core.CoreReducerV1.SUM, (group,)
            ),
        )
    if operation is exact_core.CoreTransformOperationV1.SPATIAL_AGGREGATION:
        source_refs = (
            _core_ref_v2(_SOURCE_SCALE, "spatial-a"),
            _core_ref_v2(_SOURCE_SCALE, "spatial-b"),
        )
        target = _core_ref_v2(_TARGET_SCALE, "spatial")
        source_metadata = (
            _core_metadata_v2(
                source_refs[0],
                label="density",
                axis=exact_core.CoreComponentAxisV1.SPATIAL,
                spatial=exact_core.CoreSpatialSupportV1(
                    _FRAME_A, (_F(0), _F(0)), (_F(1), _F(1))
                ),
            ),
            _core_metadata_v2(
                source_refs[1],
                label="density",
                axis=exact_core.CoreComponentAxisV1.SPATIAL,
                spatial=exact_core.CoreSpatialSupportV1(
                    _FRAME_A, (_F(1), _F(0)), (_F(2), _F(1))
                ),
            ),
        )
        target_metadata = _core_metadata_v2(
            target,
            label="density",
            axis=exact_core.CoreComponentAxisV1.SPATIAL,
            spatial=exact_core.CoreSpatialSupportV1(
                _FRAME_A, (_F(0), _F(0)), (_F(2), _F(1))
            ),
        )
        group = _core_group_v2(source_refs, (target,), "spatial-bin")
        return _core_profile_v2(
            operation=operation,
            sources=(
                _core_numeric_v2(source_metadata[0], _F(2), _F(1, 10)),
                _core_numeric_v2(source_metadata[1], _F(4), _F(1, 5)),
            ),
            outputs=(target_metadata,),
            rows=(
                _core_row_v2(
                    target,
                    ((source_refs[0], _F(1, 2)), (source_refs[1], _F(1, 2))),
                ),
            ),
            certificate=exact_core.CoreSpatialAggregationCertificateV1(
                exact_core.CoreReducerV1.WEIGHTED_MEAN, (group,)
            ),
        )
    if operation is exact_core.CoreTransformOperationV1.SAMPLING_RESOLUTION:
        source_refs = tuple(
            _core_ref_v2(_SOURCE_SCALE, f"sample-{name}")
            for name in ("a", "b")
        )
        source_metadata = tuple(
            _core_metadata_v2(
                ref,
                label="series",
                axis=exact_core.CoreComponentAxisV1.TEMPORAL,
                temporal=exact_core.CoreTemporalSupportV1(
                    _CLOCK, _F(index), _F(index)
                ),
            )
            for index, ref in enumerate(source_refs)
        )
        outputs = (
            replace(source_metadata[0], ref=_core_ref_v2(_TARGET_SCALE, "sample-a")),
        )
        selected = (source_refs[0],)
        return _core_profile_v2(
            operation=operation,
            sources=tuple(
                _core_numeric_v2(item, _F(index + 1), _F(index + 1, 10))
                for index, item in enumerate(source_metadata)
            ),
            outputs=outputs,
            rows=(
                _core_row_v2(outputs[0].ref, ((source_refs[0], _F(1)),)),
            ),
            certificate=exact_core.CoreSamplingResolutionCertificateV1(
                exact_core.CoreComponentAxisV1.TEMPORAL,
                selected,
                (source_refs[1],),
                (
                    exact_core.CoreSamplingGridPointV1(source_refs[0], (_F(0),)),
                ),
                1,
                None,
            ),
        )
    if operation is exact_core.CoreTransformOperationV1.EQUIVALENT_SPLIT_MERGE:
        source = _core_ref_v2(_SOURCE_SCALE, "split")
        targets = (
            _core_ref_v2(_TARGET_SCALE, "split-a"),
            _core_ref_v2(_TARGET_SCALE, "split-b"),
        )
        source_metadata = _core_metadata_v2(
            source, label="mass", role=exact_core.CoreComponentValueRoleV1.EXTENSIVE
        )
        outputs = tuple(
            _core_metadata_v2(
                ref,
                label=f"mass-{index}",
                role=exact_core.CoreComponentValueRoleV1.EXTENSIVE,
            )
            for index, ref in enumerate(targets)
        )
        group = _core_group_v2((source,), targets, "split-equivalence")
        inverse = (
            _core_row_v2(
                source, ((targets[0], _F(1)), (targets[1], _F(1)))
            ),
        )
        bindings = tuple(
            exact_core.CoreOutputLabelBindingV1(
                (source,), item.ref, item.component_label, "split-equivalence"
            )
            for item in outputs
        )
        return _core_profile_v2(
            operation=operation,
            sources=(_core_numeric_v2(source_metadata, _F(8), _F(1)),),
            outputs=outputs,
            rows=(
                _core_row_v2(targets[0], ((source, _F(1, 4)),)),
                _core_row_v2(targets[1], ((source, _F(3, 4)),)),
            ),
            certificate=exact_core.CoreEquivalentSplitMergeCertificateV1(
                exact_core.CoreSplitMergeDirectionV1.SPLIT,
                (group,),
                inverse,
                bindings,
            ),
        )
    if operation is exact_core.CoreTransformOperationV1.COARSE_GRAINING:
        source_refs = (
            _core_ref_v2(_SOURCE_SCALE, "coarse-a"),
            _core_ref_v2(_SOURCE_SCALE, "coarse-b"),
        )
        target = _core_ref_v2(_TARGET_SCALE, "coarse")
        source_metadata = tuple(
            _core_metadata_v2(
                ref,
                label="amount",
                role=exact_core.CoreComponentValueRoleV1.EXTENSIVE,
            )
            for ref in source_refs
        )
        output = _core_metadata_v2(
            target,
            label="amount",
            axis=exact_core.CoreComponentAxisV1.COARSE,
            role=exact_core.CoreComponentValueRoleV1.EXTENSIVE,
        )
        group = _core_group_v2(source_refs, (target,), "quotient-a")
        label_binding = exact_core.CoreOutputLabelBindingV1(
            source_refs, target, output.component_label, "quotient-a"
        )
        return _core_profile_v2(
            operation=operation,
            sources=(
                _core_numeric_v2(source_metadata[0], _F(2), _F(1, 10)),
                _core_numeric_v2(source_metadata[1], _F(4), _F(1, 5)),
            ),
            outputs=(output,),
            rows=(
                _core_row_v2(
                    target,
                    ((source_refs[0], _F(1)), (source_refs[1], _F(1))),
                ),
            ),
            certificate=exact_core.CoreCoarseGrainingCertificateV1(
                exact_core.CoreReducerV1.SUM,
                (group,),
                ("quotient-a",),
                tuple(_core_row_v2(ref, ((ref, _F(1)),)) for ref in source_refs),
                (_core_row_v2(target, ((target, _F(1)),)),),
                (label_binding,),
            ),
        )
    raise ValueError("unrecognized exact transform operation")


def _authority_label_input_v2(
    row_index: int,
) -> auditor.FormalAuthorityLabelInputV1:
    if type(row_index) is not int or not 0 <= row_index < EXACT_AUTHORITY_ROWS:
        raise ValueError("synthetic authority row index is outside 0..1023")
    cell = row_index % 64
    joint = ("joint-a", "joint-b")[cell & 1]
    cell >>= 1
    answerability = (
        auditor.AnswerabilityLabelV1.ANSWERABLE,
        auditor.AnswerabilityLabelV1.ABSTAIN,
    )[cell & 1]
    cell >>= 1
    binding = ("binding-a", "binding-b")[cell & 1]
    cell >>= 1
    scale = ("root", "child")[cell & 1]
    cell >>= 1
    family = ("F01", "F02")[cell & 1]
    cell >>= 1
    case_type = ("synthetic-type-a", "synthetic-type-b")[cell & 1]
    return auditor.FormalAuthorityLabelInputV1(
        row_index,
        family,
        binding,
        scale,
        answerability,
        joint,
        case_type,
    )


def _synthetic_label_inputs_v2(
    row_count: int = EXACT_AUTHORITY_ROWS,
) -> tuple[auditor.FormalAuthorityLabelInputV1, ...]:
    if type(row_count) is not int or not 1 <= row_count <= EXACT_AUTHORITY_ROWS:
        raise ValueError("synthetic label row count is outside 1..1024")
    return tuple(_authority_label_input_v2(index) for index in range(row_count))


def _synthetic_source_case_v2(
    row_index: int,
    *,
    binding_leak: bool,
    standard_error: bool = False,
    operation_index: int | None = None,
) -> wire.FormalSourceCaseV3:
    if type(row_index) is not int or not 0 <= row_index < EXACT_AUTHORITY_ROWS:
        raise ValueError("synthetic source row index is outside 0..1023")
    if type(binding_leak) is not bool or type(standard_error) is not bool:
        raise TypeError("synthetic case switches must be exact booleans")
    if operation_index is None:
        operation_index = (row_index // 64) % len(EXACT_TRANSFORM_OPERATIONS)
    if type(operation_index) is not int or not 0 <= operation_index < len(
        EXACT_TRANSFORM_OPERATIONS
    ):
        raise ValueError("synthetic transform operation index is outside 0..7")
    operation = tuple(exact_core.CoreTransformOperationV1)[operation_index]
    profile = _core_fixture_profile_v2(operation)
    computed = exact_core.run_exact_transform_core_v1(profile)
    if (
        computed.disposition is not exact_core.CoreTransformDispositionV1.COMPLETE
        or not computed.output_components
    ):
        raise ValueError("exact transform fixture did not complete")
    all_values = tuple(
        sorted(
            (*profile.source_components, *computed.output_components),
            key=lambda item: item.ref,
        )
    )
    next_slot = 0

    def new_uuid() -> str:
        nonlocal next_slot
        next_slot += 1
        return _ordered_case_uuid_v2(row_index, next_slot)

    def rational(value: Fraction) -> wire.RationalValueV3:
        return wire.RationalValueV3(str(value.numerator), str(value.denominator))

    bundle_id, aggregate_map_id, transform_id, task_id = (
        new_uuid() for _ in range(4)
    )
    entity_keys = tuple(
        sorted({key for item in all_values for key in item.metadata.entity_keys})
    )
    role_keys = tuple(
        sorted({key for item in all_values for key in item.metadata.role_keys})
    )
    entity_ids = {key: new_uuid() for key in entity_keys}
    role_ids = {key: new_uuid() for key in role_keys}
    scale_keys = tuple(sorted({item.ref.scale_ref.key for item in all_values}))
    scale_ids = {key: new_uuid() for key in scale_keys}

    quantities: list[exact_core.CoreQuantityMetadataV1] = []
    for item in all_values:
        if item.metadata.quantity not in quantities:
            quantities.append(item.metadata.quantity)
    quantity_ids = {value: new_uuid() for value in quantities}
    semantic_keys: dict[str, str] = {}
    unit_keys: dict[exact_core.CoreUnitBindingV1, str] = {}
    quantity_records: list[wire.FormalQuantityRecordV3] = []
    for quantity in quantities:
        semantic_key = semantic_keys.setdefault(
            quantity.semantic_key, f"quantity-{len(semantic_keys)}"
        )
        if quantity.unit_binding is None:
            unit_binding = wire.FormalUnitBindingV3(
                "none", rational(_F(1)), rational(_F(0))
            )
            value_kind = "boolean"
        else:
            unit_binding = wire.FormalUnitBindingV3(
                unit_keys.setdefault(
                    quantity.unit_binding, f"unit-{len(unit_keys)}"
                ),
                rational(quantity.unit_binding.scale_to_si),
                rational(quantity.unit_binding.offset_to_si),
            )
            value_kind = "numeric"
        quantity_records.append(
            wire.FormalQuantityRecordV3(
                quantity_ids[quantity],
                semantic_key,
                quantity.si_exponents,
                unit_binding,
                value_kind,
            )
        )

    context_keys: list[
        tuple[
            exact_core.CoreTemporalSupportV1 | None,
            exact_core.CoreSpatialSupportV1 | None,
        ]
    ] = []
    for item in all_values:
        key = (item.metadata.temporal_support, item.metadata.spatial_support)
        if key not in context_keys:
            context_keys.append(key)
    context_ids = {key: new_uuid() for key in context_keys}
    clock_keys: dict[exact_core.CoreClockBindingV1, str] = {}
    frame_keys: dict[exact_core.CoreFrameBindingV1, str] = {}
    context_records: list[wire.FormalContextRecordV3] = []
    for temporal, spatial in context_keys:
        if temporal is None:
            temporal_kind = "none"
            clock_binding = "none"
            temporal_support = wire.NullableTemporalSupportV3(False, None)
        else:
            temporal_kind = "domain_epoch"
            clock_binding = clock_keys.setdefault(
                temporal.clock_binding, f"clock-{len(clock_keys)}"
            )
            temporal_support = wire.NullableTemporalSupportV3(
                True,
                wire.ExactTemporalSupportV3(
                    rational(temporal.start), rational(temporal.end)
                ),
            )
        if spatial is None:
            spatial_kind = "none"
            frame_binding = "none"
            coordinate_labels: tuple[str, ...] = ()
            spatial_support = wire.NullableSpatialSupportV3(False, None)
            orientation = wire.NullableTextV3(False, None)
        else:
            spatial_kind = "cartesian"
            frame_binding = frame_keys.setdefault(
                spatial.frame_binding, f"frame-{len(frame_keys)}"
            )
            coordinate_labels = tuple(
                f"axis-{dimension}" for dimension in range(len(spatial.lower))
            )
            spatial_support = wire.NullableSpatialSupportV3(
                True,
                wire.ExactSpatialSupportV3(
                    tuple(rational(value) for value in spatial.lower),
                    tuple(rational(value) for value in spatial.upper),
                ),
            )
            orientation = wire.NullableTextV3(
                spatial.frame_binding.orientation is not None,
                spatial.frame_binding.orientation,
            )
        context_records.append(
            wire.FormalContextRecordV3(
                context_ids[(temporal, spatial)],
                temporal_kind,
                clock_binding,
                spatial_kind,
                frame_binding,
                coordinate_labels,
                temporal_support,
                spatial_support,
                orientation,
            )
        )

    owner_values: dict[
        tuple[exact_core.CoreScaleRefV1, exact_core.CoreObservationRefV1],
        list[exact_core.CoreComponentValueV1],
    ] = {}
    owner_for_ref: dict[
        exact_core.CoreComponentRefV1,
        tuple[exact_core.CoreScaleRefV1, exact_core.CoreObservationRefV1],
    ] = {}
    coordinate_source_owner = exact_core.CoreObservationRefV1(
        "qualification-v2-coordinate-source-vector"
    )
    for item in all_values:
        owner = (
            item.ref.scale_ref,
            coordinate_source_owner
            if operation is exact_core.CoreTransformOperationV1.COORDINATE_AFFINE
            and item.ref.scale_ref == profile.contract.source_scale_ref
            else item.ref.observation_ref,
        )
        owner_for_ref[item.ref] = owner
        owner_values.setdefault(owner, []).append(item)
    observation_ids = {owner: new_uuid() for owner in sorted(owner_values)}
    local_component_indices: dict[exact_core.CoreComponentRefV1, int] = {}
    observation_records: list[wire.FormalObservationRecordV3] = []
    for owner in sorted(owner_values):
        values = tuple(
            sorted(
                owner_values[owner],
                key=lambda item: (item.ref.component_index, item.ref),
            )
        )
        for local_index, item in enumerate(values):
            local_component_indices[item.ref] = local_index
        metadata = values[0].metadata
        if any(
            item.metadata.quantity != metadata.quantity
            or item.metadata.entity_keys != metadata.entity_keys
            or item.metadata.role_keys != metadata.role_keys
            or item.metadata.temporal_support != metadata.temporal_support
            or item.metadata.spatial_support != metadata.spatial_support
            for item in values
        ):
            raise ValueError("synthetic observation components disagree on ownership")
        kinds = {item.value_kind for item in values}
        if len(kinds) != 1:
            raise ValueError("synthetic observation mixes value kinds")
        kind = values[0].value_kind
        if kind is exact_core.CoreComponentValueKindV1.NUMERIC:
            numeric_value = wire.NullableRationalVectorV3(
                True,
                tuple(rational(item.numeric_value) for item in values),
            )
            boolean_value = wire.NullableBooleanV3(False, None)
            uncertainty_model = (
                "standard_error"
                if standard_error
                else values[0].uncertainty_model.value
            )
            uncertainty_radius = wire.NullableRationalVectorV3(
                True,
                tuple(rational(item.absolute_radius) for item in values),
            )
        else:
            if len(values) != 1:
                raise ValueError("non-numeric synthetic observation is not scalar")
            numeric_value = wire.NullableRationalVectorV3(False, None)
            boolean_value = wire.NullableBooleanV3(
                kind is exact_core.CoreComponentValueKindV1.BOOLEAN,
                values[0].boolean_value,
            )
            uncertainty_model = "not_applicable"
            uncertainty_radius = wire.NullableRationalVectorV3(False, None)
        observation_records.append(
            wire.FormalObservationRecordV3(
                observation_ids[owner],
                tuple(sorted(entity_ids[key] for key in metadata.entity_keys)),
                tuple(sorted(role_ids[key] for key in metadata.role_keys)),
                quantity_ids[metadata.quantity],
                context_ids[(metadata.temporal_support, metadata.spatial_support)],
                scale_ids[owner[0].key],
                tuple(
                    wire.FormalComponentDescriptorV3(
                        item.metadata.component_label,
                        item.metadata.axis.value,
                        item.metadata.value_role.value,
                    )
                    for item in values
                ),
                numeric_value,
                boolean_value,
                uncertainty_model,
                uncertainty_radius,
                _ZERO_DIGEST,
            )
        )
    observation_by_owner = {
        owner: record
        for owner, record in zip(
            sorted(owner_values), observation_records, strict=True
        )
    }

    def formal_ref(
        ref: exact_core.CoreComponentRefV1,
    ) -> wire.FormalComponentRefV3:
        record = observation_by_owner[owner_for_ref[ref]]
        local_index = local_component_indices[ref]
        descriptor = record.components[local_index]
        return wire.FormalComponentRefV3(
            record.observation_id,
            record.quantity_id,
            record.context_id,
            record.scale_id,
            descriptor.label,
            local_index,
        )

    def formal_row(row: exact_core.CoreExactRowV1) -> wire.FormalExactRowV3:
        return wire.FormalExactRowV3(
            formal_ref(row.output_ref),
            tuple(
                sorted(
                    (
                        wire.FormalExactTermV3(
                            formal_ref(term.input_ref),
                            rational(term.coefficient),
                        )
                        for term in row.terms
                    ),
                    key=lambda term: term.input_ref,
                )
            ),
            rational(row.offset),
        )

    def formal_rows(
        rows: tuple[exact_core.CoreExactRowV1, ...],
    ) -> tuple[wire.FormalExactRowV3, ...]:
        return tuple(
            sorted(
                (formal_row(row) for row in rows),
                key=lambda row: row.output_ref,
            )
        )

    certificate = profile.contract.certificate
    if hasattr(certificate, "groups"):
        groups = tuple(
            wire.FormalAggregateGroupV3(
                tuple(sorted(formal_ref(ref) for ref in group.input_refs)),
                tuple(sorted(formal_ref(ref) for ref in group.output_refs)),
                group.quotient_label,
            )
            for group in certificate.groups
        )
    else:
        groups = (
            wire.FormalAggregateGroupV3(
                tuple(sorted(formal_ref(ref) for ref in profile.contract.input_refs)),
                tuple(
                    sorted(
                        formal_ref(item.ref)
                        for item in profile.contract.output_metadata
                    )
                ),
                "all-components",
            ),
        )
    operation_text = operation.value
    reducer = (
        certificate.reducer.value
        if hasattr(certificate, "reducer")
        else "not_applicable"
    )
    missing_policy = certificate.missing_policy.value
    if type(certificate) is exact_core.CoreIdentityCertificateV1:
        formal_certificate: wire.FormalOperationCertificateV3 = (
            wire.FormalIdentityCertificateV3(
                wire.FORMAL_OPERATION_SEMANTICS_VERSION_V3,
                missing_policy,
                "exact_one_to_one_identity",
            )
        )
    elif type(certificate) is exact_core.CoreUnitConversionCertificateV1:
        formal_certificate = wire.FormalUnitConversionCertificateV3(
            wire.FORMAL_OPERATION_SEMANTICS_VERSION_V3,
            rational(certificate.factor),
            rational(certificate.inverse_factor),
            missing_policy,
            "target_value_equals_source_value_times_factor",
            "si_dimension_and_support_preserved",
        )
    elif type(certificate) is exact_core.CoreCoordinateAffineCertificateV1:
        formal_certificate = wire.FormalCoordinateAffineCertificateV3(
            wire.FORMAL_OPERATION_SEMANTICS_VERSION_V3,
            frame_keys[certificate.source_frame_binding],
            frame_keys[certificate.target_frame_binding],
            certificate.dimension,
            formal_rows(certificate.inverse_rows),
            missing_policy,
            "exact_affine_box_enclosure",
            "two_sided_exact_sparse_affine_inverse",
        )
    elif type(certificate) is exact_core.CoreTemporalAggregationCertificateV1:
        formal_certificate = wire.FormalTemporalAggregationCertificateV3(
            wire.FORMAL_OPERATION_SEMANTICS_VERSION_V3,
            aggregate_map_id,
            reducer,
            missing_policy,
            certificate.boundary_policy.value,
            "same_clock_half_open_partition_modulo_final_endpoint",
            certificate.weighted_mean_basis.value,
        )
    elif type(certificate) is exact_core.CoreSpatialAggregationCertificateV1:
        formal_certificate = wire.FormalSpatialAggregationCertificateV3(
            wire.FORMAL_OPERATION_SEMANTICS_VERSION_V3,
            aggregate_map_id,
            reducer,
            missing_policy,
            certificate.boundary_policy.value,
            "same_frame_interior_disjoint_positive_volume_partition",
            certificate.weighted_mean_basis.value,
        )
    elif type(certificate) is exact_core.CoreSamplingResolutionCertificateV1:
        formal_certificate = wire.FormalSamplingResolutionCertificateV3(
            wire.FORMAL_OPERATION_SEMANTICS_VERSION_V3,
            certificate.axis.value,
            tuple(formal_ref(ref) for ref in certificate.selected_inputs),
            tuple(sorted(formal_ref(ref) for ref in certificate.discarded_inputs)),
            tuple(
                wire.FormalSamplingGridPointV3(
                    formal_ref(point.input_ref),
                    tuple(rational(value) for value in point.coordinates),
                )
                for point in certificate.grid_points
            ),
            certificate.grid_dimension,
            (
                "none"
                if certificate.grid_frame_binding is None
                else frame_keys[certificate.grid_frame_binding]
            ),
            missing_policy,
            certificate.boundary_policy.value,
            "single_series_ordered_exact_subselection",
        )
    elif type(certificate) is exact_core.CoreEquivalentSplitMergeCertificateV1:
        formal_certificate = wire.FormalEquivalentSplitMergeCertificateV3(
            wire.FORMAL_OPERATION_SEMANTICS_VERSION_V3,
            aggregate_map_id,
            certificate.direction.value,
            formal_rows(certificate.inverse_rows),
            missing_policy,
            "extensive_sum_one_sided_exact_inverse",
        )
    elif type(certificate) is exact_core.CoreCoarseGrainingCertificateV1:
        formal_certificate = wire.FormalCoarseGrainingCertificateV3(
            wire.FORMAL_OPERATION_SEMANTICS_VERSION_V3,
            aggregate_map_id,
            reducer,
            formal_rows(certificate.source_commutation_rows),
            formal_rows(certificate.target_commutation_rows),
            aggregate_map_id,
            certificate.quotient_labels,
            missing_policy,
            certificate.boundary_policy.value,
            "transform_after_source_equals_target_after_transform",
            certificate.weighted_mean_basis.value,
        )
    else:
        raise TypeError("synthetic exact transform certificate type drift")

    input_refs = tuple(sorted(formal_ref(ref) for ref in profile.contract.input_refs))
    output_refs = tuple(
        sorted(formal_ref(item.ref) for item in profile.contract.output_metadata)
    )
    private_binding = _authority_label_input_v2(row_index).binding
    semantic_statements = [f"Exact typed {operation_text} witness."]
    eligible_observation_ids = tuple(
        sorted(record.observation_id for record in observation_records)
    )
    if binding_leak:
        eligible_observation_ids = eligible_observation_ids[
            : 1 if private_binding == "binding-a" else 2
        ]
    graph = wire.FormalCaseGraphV3(
        wire.FORMAL_CASE_GRAPH_V3_SCHEMA_VERSION,
        bundle_id,
        tuple(
            wire.FormalEntityRecordV3(
                entity_ids[key], tuple(sorted(role_ids.values()))
            )
            for key in entity_keys
        ),
        tuple(sorted(role_ids.values())),
        tuple(quantity_records),
        tuple(context_records),
        tuple(observation_records),
        wire.FormalTaskTargetV3(
            task_id,
            tuple(sorted(entity_ids.values())),
            tuple(sorted(quantity_ids.values())),
        ),
        wire.FormalSemanticDescriptorV3(
            "symbolic", tuple(semantic_statements)
        ),
        tuple(
            wire.FormalScaleRecordV3(
                scale_ids[key], key, key == profile.contract.source_scale_ref.key
            )
            for key in scale_keys
        ),
        (
            wire.FormalAggregateMapRecordV3(
                aggregate_map_id,
                scale_ids[profile.contract.source_scale_ref.key],
                scale_ids[profile.contract.target_scale_ref.key],
                transform_id,
                operation_text,
                reducer,
                groups,
            ),
        ),
        (
            wire.FormalTransformContractV3(
                transform_id,
                operation_text,
                scale_ids[profile.contract.source_scale_ref.key],
                scale_ids[profile.contract.target_scale_ref.key],
                input_refs,
                output_refs,
                formal_rows(profile.contract.kernel_rows),
                tuple(
                    sorted(
                        (
                            wire.FormalDiscreteMappingV3(
                                formal_ref(mapping.input_ref),
                                formal_ref(mapping.output_ref),
                            )
                            for mapping in profile.contract.discrete_mappings
                        ),
                        key=lambda mapping: (
                            mapping.input_ref,
                            mapping.output_ref,
                        ),
                    )
                ),
                formal_certificate,
            ),
        ),
        wire.FormalRecognizerRegistryV3(
            "exact-transform-core-v1",
            (wire.FormalFamilyAliasV3(operation_text, operation_text),),
            (
                wire.FormalLawBindingV3(
                    operation_text + "-law",
                    operation_text,
                    operation_text,
                    tuple(
                        sorted(
                            record.observation_id for record in observation_records
                        )
                    ),
                    tuple(sorted(role_ids.values())),
                    tuple(sorted(quantity_ids.values())),
                ),
            ),
            tuple(
                sorted(
                    (
                        wire.FormalObservableChannelV3(
                            record.observation_id, record.quantity_id
                        )
                        for record in observation_records
                    ),
                    key=lambda item: item.observation_id,
                )
            ),
            100,
        ),
        wire.FormalEligibilityMaskV3(
            eligible_observation_ids,
            tuple(sorted(role_ids.values())),
            tuple(sorted(scale_ids.values())),
        ),
    )
    return wire.FormalSourceCaseV3(graph)


_SYNTHETIC_ENVELOPE_HEADER_V2: Final = struct.Struct(">8sHHI32s32s")
_SYNTHETIC_PAYLOAD_KEYS_V2: Final = {
    "batch_context",
    "field_classification_manifest_id",
    "graph_completeness_policy_id",
    "public_case",
    "schema_version",
}
_SYNTHETIC_PUBLIC_CASE_KEYS_V2: Final = {
    "graph",
    "public_provenance_commitment",
    "schema_version",
}
_SYNTHETIC_GRAPH_KEYS_V2: Final = {
    item.name for item in dataclass_fields(wire.FormalCaseGraphV3)
}
_SYNTHETIC_ENTITY_KEYS_V2: Final = {
    item.name for item in dataclass_fields(wire.FormalEntityRecordV3)
}
_SYNTHETIC_SCALE_KEYS_V2: Final = {
    item.name for item in dataclass_fields(wire.FormalScaleRecordV3)
}
_SYNTHETIC_REGISTRY_KEYS_V2: Final = {
    item.name for item in dataclass_fields(wire.FormalRecognizerRegistryV3)
}
_SYNTHETIC_FAMILY_ALIAS_KEYS_V2: Final = {
    item.name for item in dataclass_fields(wire.FormalFamilyAliasV3)
}


def _synthetic_json_pairs_v2(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("synthetic recognizer rejects duplicate JSON names")
        result[key] = value
    return result


def _lightweight_synthetic_public_projection_v2(
    envelope: bytes,
) -> tuple[str, tuple[auditor.FormalBindingV1, ...], tuple[str, ...]]:
    """Consume an already Wire-suite-validated public frame without redoing Wire V3.

    The auditor and its Wire adapter perform the full semantic, provenance, channel,
    and canonical decode before every recognizer invocation.  This cooperative
    synthetic recognizer independently replays the bounded frame/digests and reads
    only the three public fields that determine its decision.  It never accesses
    adapter-private alignment or authority state.
    """

    if type(envelope) is not bytes or len(envelope) != wire.ENVELOPE_BYTES_V3:
        raise ValueError("synthetic recognizer envelope framing drifted")
    (
        magic,
        version,
        header_bytes,
        payload_length,
        payload_digest,
        padding_digest,
    ) = _SYNTHETIC_ENVELOPE_HEADER_V2.unpack(
        envelope[: wire.ENVELOPE_HEADER_BYTES_V3]
    )
    if (
        (magic, version, header_bytes)
        != (
            wire.ENVELOPE_MAGIC_V3,
            wire.ENVELOPE_VERSION_V3,
            wire.ENVELOPE_HEADER_BYTES_V3,
        )
        or type(payload_length) is not int
        or not 0 < payload_length <= wire.MAXIMUM_PAYLOAD_BYTES_V3
    ):
        raise ValueError("synthetic recognizer header drifted")
    payload_bytes = envelope[header_bytes : header_bytes + payload_length]
    padding_bytes = envelope[header_bytes + payload_length :]
    if (
        len(padding_bytes) < 32
        or hashlib.sha256(payload_bytes).digest() != payload_digest
        or hashlib.sha256(padding_bytes).digest() != padding_digest
    ):
        raise ValueError("synthetic recognizer frame digest drifted")
    try:
        decoded = json.loads(
            payload_bytes.decode("utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-JSON constant {token}")
            ),
            object_pairs_hook=_synthetic_json_pairs_v2,
        )
    except (RecursionError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("synthetic recognizer public JSON decode failed") from exc
    nodes = 0
    stack: list[tuple[object, int]] = [(decoded, 0)]
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > 65_536 or depth > 64:
            raise ValueError("synthetic recognizer public JSON budget exceeded")
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("synthetic recognizer JSON key type drifted")
            stack.extend((item, depth + 1) for item in value.values())
        elif type(value) is list:
            stack.extend((item, depth + 1) for item in value)
        elif type(value) not in (type(None), bool, int, float, str):
            raise ValueError("synthetic recognizer JSON value type drifted")
    if type(decoded) is not dict or set(decoded) != _SYNTHETIC_PAYLOAD_KEYS_V2:
        raise ValueError("synthetic recognizer payload wrapper drifted")
    if (
        decoded["schema_version"]
        != wire.FORMAL_TRUSTED_WIRE_V3_PAYLOAD_SCHEMA_VERSION
        or decoded["field_classification_manifest_id"]
        != wire.FIELD_CLASSIFICATION_MANIFEST_ID_V3
        or decoded["graph_completeness_policy_id"]
        != wire.FORMAL_CASE_GRAPH_COMPLETENESS_POLICY_ID_V3
    ):
        raise ValueError("synthetic recognizer payload identity drifted")
    public_case = decoded["public_case"]
    if (
        type(public_case) is not dict
        or set(public_case) != _SYNTHETIC_PUBLIC_CASE_KEYS_V2
        or public_case["schema_version"] != wire.FORMAL_PUBLIC_CASE_V3_SCHEMA_VERSION
    ):
        raise ValueError("synthetic recognizer public case drifted")
    graph = public_case["graph"]
    if type(graph) is not dict or set(graph) != _SYNTHETIC_GRAPH_KEYS_V2:
        raise ValueError("synthetic recognizer graph fields drifted")
    entities = graph["entities"]
    scales = graph["scales"]
    registry = graph["recognizer_registry"]
    if (
        type(entities) is not list
        or not 1 <= len(entities) <= 4_096
        or type(scales) is not list
        or not 1 <= len(scales) <= 4_096
        or type(registry) is not dict
        or set(registry) != _SYNTHETIC_REGISTRY_KEYS_V2
    ):
        raise ValueError("synthetic recognizer selected graph shape drifted")
    binding_rows: list[auditor.FormalBindingV1] = []
    for entity in entities:
        if type(entity) is not dict or set(entity) != _SYNTHETIC_ENTITY_KEYS_V2:
            raise ValueError("synthetic recognizer entity fields drifted")
        entity_id = entity["entity_id"]
        roles = entity["role_candidate_ids"]
        if (
            type(entity_id) is not str
            or type(roles) is not list
            or not roles
            or roles != sorted(set(roles))
            or any(type(role) is not str for role in roles)
        ):
            raise ValueError("synthetic recognizer entity binding drifted")
        binding_rows.extend(
            auditor.FormalBindingV1(role, entity_id) for role in roles
        )
    roots: list[str] = []
    for scale in scales:
        if type(scale) is not dict or set(scale) != _SYNTHETIC_SCALE_KEYS_V2:
            raise ValueError("synthetic recognizer scale fields drifted")
        if type(scale["is_root"]) is not bool or type(scale["scale_id"]) is not str:
            raise ValueError("synthetic recognizer scale projection drifted")
        if scale["is_root"]:
            roots.append(scale["scale_id"])
    aliases = registry["family_aliases"]
    if type(aliases) is not list or not 1 <= len(aliases) <= 4_096:
        raise ValueError("synthetic recognizer family-alias shape drifted")
    families: set[str] = set()
    for alias in aliases:
        if (
            type(alias) is not dict
            or set(alias) != _SYNTHETIC_FAMILY_ALIAS_KEYS_V2
            or type(alias["canonical_family"]) is not str
            or not alias["canonical_family"]
        ):
            raise ValueError("synthetic recognizer family alias drifted")
        families.add(alias["canonical_family"])
    if len(families) != 1 or not roots or not binding_rows:
        raise ValueError("synthetic recognizer decision projection is empty")
    return (
        next(iter(families)),
        tuple(sorted(binding_rows)),
        tuple(sorted(roots)),
    )


def _trusted_synthetic_public_projection_v2(
    decoded: auditor.DecodedFormalEnvelopeV1,
) -> tuple[str, tuple[auditor.FormalBindingV1, ...], tuple[str, ...]]:
    """Project the same decision fields from the pinned full adapter DTO."""

    if type(decoded) is not auditor.DecodedFormalEnvelopeV1:
        raise TypeError("trusted synthetic projection needs the exact decoded DTO")
    decoded.__post_init__()
    payload = decoded.payload
    if type(payload) is not dict:
        raise ValueError("trusted synthetic payload is not an object")
    public_case = payload.get("public_case")
    if type(public_case) is not dict:
        raise ValueError("trusted synthetic public case is absent")
    graph = public_case.get("graph")
    if type(graph) is not dict:
        raise ValueError("trusted synthetic graph is absent")
    entities = graph.get("entities")
    scales = graph.get("scales")
    registry = graph.get("recognizer_registry")
    if type(entities) is not list or type(scales) is not list or type(registry) is not dict:
        raise ValueError("trusted synthetic decision graph is malformed")
    bindings = tuple(
        sorted(
            auditor.FormalBindingV1(role, entity["entity_id"])
            for entity in entities
            for role in entity["role_candidate_ids"]
        )
    )
    roots = tuple(
        sorted(scale["scale_id"] for scale in scales if scale["is_root"] is True)
    )
    aliases = registry.get("family_aliases")
    if type(aliases) is not list:
        raise ValueError("trusted synthetic family aliases are malformed")
    families = {alias["canonical_family"] for alias in aliases}
    if len(families) != 1 or not bindings or not roots:
        raise ValueError("trusted synthetic decision projection is empty")
    return next(iter(families)), bindings, roots


def _verify_lightweight_trusted_projection_equivalence_v2(
    envelope: bytes,
    decoded: auditor.DecodedFormalEnvelopeV1,
) -> tuple[str, tuple[auditor.FormalBindingV1, ...], tuple[str, ...]]:
    lightweight = _lightweight_synthetic_public_projection_v2(envelope)
    trusted = _trusted_synthetic_public_projection_v2(decoded)
    if lightweight != trusted:
        raise ValueError("lightweight/trusted synthetic decision projection drifted")
    return lightweight


class _SyntheticRichRecognizerV2(auditor.FormalRecognizerAdapterV1):
    __slots__ = ("_adapter", "calls", "row_counts")

    def __init__(self, adapter: auditor.FormalWireV3AdapterV1) -> None:
        if type(adapter) is not auditor.FormalWireV3AdapterV1:
            raise TypeError("synthetic recognizer needs the exact Wire V3 adapter")
        self._adapter = adapter
        self.calls = 0
        self.row_counts: list[int] = []

    @property
    def implementation_id(self) -> str:
        return SYNTHETIC_RECOGNIZER_ID

    def run_batch(
        self, envelopes: tuple[bytes, ...]
    ) -> tuple[auditor.FormalRecognizerDecisionV1, ...]:
        if type(envelopes) is not tuple:
            raise TypeError("recognizer input must use an exact tuple")
        self.calls += 1
        self.row_counts.append(len(envelopes))
        rows: list[auditor.FormalRecognizerDecisionV1] = []
        for envelope in envelopes:
            family, bindings, roots = _lightweight_synthetic_public_projection_v2(
                envelope
            )
            rows.append(
                auditor.FormalRecognizerDecisionV1(
                    "SELECT",
                    family,
                    bindings,
                    roots,
                    b"synthetic-identity-residual-v2",
                )
            )
        return tuple(rows)


_REGISTERED_BINDING_LEAK_OWNER_PATH_V2: Final = (
    "public_case",
    "graph",
    "eligibility_mask",
    "observation_ids",
    1,
)


def _json_path_text_v2(path: tuple[str | int, ...]) -> str:
    """Render the auditor's public JSON-path spelling without using a private API."""

    if type(path) is not tuple or not path or any(
        type(token) not in (str, int) for token in path
    ):
        raise TypeError("qualification JSON path is not an exact nonempty path")
    return "/" + "/".join(
        str(token).replace("~", "~0").replace("/", "~1") for token in path
    )


def _registered_binding_leak_presence_alias_v2(
    execution: "_FixtureExecutionV2",
) -> str:
    """Prove the registered mask-owned UUID-presence oracle from public decodes."""

    if execution.mode != "KNOWN_BINDING_LEAK":
        raise ValueError("binding leak alias proof is only defined for the leak mode")
    aliases: set[str] = set()
    present_rows = 0
    for envelope in execution.adapter.envelopes:
        decoded = execution.adapter.decode_envelope(envelope)
        matching_owners = tuple(
            owner
            for owner in decoded.channel_only_leaf_owners
            if owner.path == _REGISTERED_BINDING_LEAK_OWNER_PATH_V2
        )
        matching_namespaces = tuple(
            leaf.namespace
            for leaf in decoded.namespace_leaves
            if leaf.path == _REGISTERED_BINDING_LEAK_OWNER_PATH_V2
        )
        if not matching_owners and not matching_namespaces:
            continue
        if (
            len(matching_owners) != 1
            or matching_owners[0].owner_group != "eligibility_mask"
            or matching_namespaces != ("observation_id",)
        ):
            raise ValueError(
                "registered binding leak path lacks exact eligibility ownership"
            )
        present_rows += 1
        aliases.add(
            "uuid_leaf/observation_id/"
            + _json_path_text_v2(_REGISTERED_BINDING_LEAK_OWNER_PATH_V2)
            + "/presence"
        )
    if present_rows != EXACT_AUTHORITY_ROWS // 2 or len(aliases) != 1:
        raise ValueError("registered binding leak presence schedule drifted")
    return next(iter(aliases))


def _is_registered_binding_leak_alias_v2(alias: object) -> bool:
    return (
        type(alias) is str
        and alias
        == "uuid_leaf/observation_id/"
        + _json_path_text_v2(_REGISTERED_BINDING_LEAK_OWNER_PATH_V2)
        + "/presence"
    )


def _validate_fixture_channel_only_owner_union_count_v2(
    mode: str,
    row_count: int,
    exact_owned_channel_only_leaf_paths: int,
) -> int:
    """Bind the decoded owner-path union without inventing a receipt field."""

    if (
        mode not in _RUN_MODES
        or type(row_count) is not int
        or not 1 <= row_count <= EXACT_AUTHORITY_ROWS
        or type(exact_owned_channel_only_leaf_paths) is not int
    ):
        raise TypeError("fixture channel-only owner union input is not exact")
    expected = _FIXTURE_FEATURE_CAPACITY_CONTRACT_V2["modes"][mode][
        "channel_only_owned_union_path_count"
    ]
    saturation_rows = _FIXTURE_FEATURE_CAPACITY_CONTRACT_V2[
        "path_saturation_operation_rows"
    ]
    if (
        exact_owned_channel_only_leaf_paths <= 0
        or exact_owned_channel_only_leaf_paths > expected
        or (
            row_count >= saturation_rows
            and exact_owned_channel_only_leaf_paths != expected
        )
    ):
        raise ValueError("fixture channel-only owner union count drift")
    return exact_owned_channel_only_leaf_paths


def _channel_only_owner_union_projection_v2(
    owners: Iterable[auditor.ChannelOnlyLeafOwnerV1],
) -> tuple[int, str]:
    """Canonicalize the unique path-to-owner assignment across decoded rows."""

    owner_by_path: dict[tuple[str | int, ...], str] = {}
    for owner in owners:
        if type(owner) is not auditor.ChannelOnlyLeafOwnerV1:
            raise TypeError("channel-only owner union needs exact owner DTOs")
        owner.__post_init__()
        prior = owner_by_path.setdefault(owner.path, owner.owner_group)
        if prior != owner.owner_group:
            raise ValueError("channel-only owner union has a conflicting owner")
    if not owner_by_path:
        raise ValueError("channel-only owner union cannot be empty")
    canonical_rows = tuple(
        sorted(
            (
                (path, owner_group)
                for path, owner_group in owner_by_path.items()
            ),
            key=lambda row: (auditor._path_text(row[0]), row[1]),
        )
    )
    return (
        len(canonical_rows),
        stable_hash(
            canonical_rows,
            prefix=(
                "phase2b_formal_wire_covert_channel_only_owner_union_v2_"
            ),
        ),
    )


def _validate_fixture_channel_only_owner_union_v2(
    mode: str,
    row_count: int,
    exact_owned_channel_only_leaf_paths: int,
    channel_only_owner_union_root: str,
) -> tuple[int, str]:
    count = _validate_fixture_channel_only_owner_union_count_v2(
        mode,
        row_count,
        exact_owned_channel_only_leaf_paths,
    )
    prefix = "phase2b_formal_wire_covert_channel_only_owner_union_v2_"
    if (
        type(channel_only_owner_union_root) is not str
        or not channel_only_owner_union_root.startswith(prefix)
        or len(channel_only_owner_union_root) != len(prefix) + 64
    ):
        raise TypeError("fixture channel-only owner union root is not exact")
    saturation_rows = _FIXTURE_FEATURE_CAPACITY_CONTRACT_V2[
        "path_saturation_operation_rows"
    ]
    expected_root = _FIXTURE_FEATURE_CAPACITY_CONTRACT_V2["modes"][mode][
        "channel_only_owned_union_root"
    ]
    if row_count >= saturation_rows and channel_only_owner_union_root != expected_root:
        raise ValueError("fixture channel-only owner union root drift")
    return count, channel_only_owner_union_root


def _validate_fixture_execution_feature_capacity_projection_v2(
    mode: str,
    row_count: int,
    projection: auditor._FeatureCardinalityProjectionV1,
) -> auditor._FeatureCardinalityProjectionV1:
    if (
        mode not in _RUN_MODES
        or type(row_count) is not int
        or not 1 <= row_count <= EXACT_AUTHORITY_ROWS
        or type(projection) is not auditor._FeatureCardinalityProjectionV1
    ):
        raise TypeError("fixture execution feature projection is not exact")
    projection.__post_init__()
    if (
        projection.rows != row_count
        or projection.projected_concrete_feature_count > auditor.MAXIMUM_FEATURES
    ):
        raise ValueError("fixture execution feature projection row/cap drift")
    if row_count in _FIXTURE_FEATURE_CAPACITY_CONTRACT_V2[
        "rows_with_identical_union_shape"
    ]:
        _validate_fixture_feature_capacity_projection_v2(mode, projection)
    return projection


@dataclass(frozen=True, slots=True)
class AuditRunDiagnosticsV2:
    mode: str
    wall_time_ns: int
    parent_historical_peak_rss_before_receipt_finalization_kib: int
    statistics_worker_historical_peak_rss_kib: tuple[int, ...]
    parent_plus_worker_historical_peak_rss_kib: int
    unified_cgroup_membership: str
    unified_cgroup_memory_scope_root: str
    unified_cgroup_leaf_device: int
    unified_cgroup_leaf_inode: int
    physical_memory_current_before_receipt_finalization_bytes: int
    physical_memory_peak_before_receipt_finalization_bytes: int
    physical_memory_max_bytes: int
    physical_memory_cap_bytes: int
    statistics_wall_time_ns: int
    host_available_memory_bytes_at_preflight: int
    recognizer_adapter_invocations: int
    minimum_rows_per_invocation: int
    maximum_rows_per_invocation: int

    def __post_init__(self) -> None:
        if self.mode not in _RUN_MODES:
            raise ValueError("audit diagnostic mode drift")
        for name in (
            "wall_time_ns",
            "parent_historical_peak_rss_before_receipt_finalization_kib",
            "parent_plus_worker_historical_peak_rss_kib",
            "physical_memory_current_before_receipt_finalization_bytes",
            "physical_memory_peak_before_receipt_finalization_bytes",
            "physical_memory_max_bytes",
            "physical_memory_cap_bytes",
            "unified_cgroup_leaf_device",
            "unified_cgroup_leaf_inode",
            "statistics_wall_time_ns",
            "host_available_memory_bytes_at_preflight",
            "recognizer_adapter_invocations",
            "minimum_rows_per_invocation",
            "maximum_rows_per_invocation",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise TypeError("audit diagnostic values must be nonnegative integers")
        if type(self.statistics_worker_historical_peak_rss_kib) is not tuple or any(
            type(item) is not int or item < 0
            for item in self.statistics_worker_historical_peak_rss_kib
        ):
            raise TypeError("worker peak RSS values must be an exact integer tuple")
        if (
            self.parent_plus_worker_historical_peak_rss_kib
            != self.parent_historical_peak_rss_before_receipt_finalization_kib
            + sum(self.statistics_worker_historical_peak_rss_kib)
        ):
            raise ValueError("audit historical RSS diagnostic drift")
        try:
            snapshot = auditor.UnifiedCgroupMemorySnapshotV1(
                self.unified_cgroup_membership,
                self.unified_cgroup_memory_scope_root,
                self.unified_cgroup_leaf_device,
                self.unified_cgroup_leaf_inode,
                self.physical_memory_current_before_receipt_finalization_bytes,
                self.physical_memory_peak_before_receipt_finalization_bytes,
                self.physical_memory_max_bytes,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("audit physical-memory diagnostic drift") from exc
        if (
            self.physical_memory_cap_bytes != auditor.MAXIMUM_ESTIMATED_RSS_BYTES
            or snapshot.memory_peak_bytes > self.physical_memory_cap_bytes
        ):
            raise ValueError("audit physical-memory cap diagnostic drift")
        if (
            self.recognizer_adapter_invocations != EXACT_RECOGNIZER_INVOCATIONS
            or self.minimum_rows_per_invocation != EXACT_AUTHORITY_ROWS
            or self.maximum_rows_per_invocation != EXACT_AUTHORITY_ROWS
        ):
            raise ValueError("audit diagnostic recognizer coverage drift")


@dataclass(frozen=True, slots=True)
class _FixtureExecutionV2:
    mode: str
    row_count: int
    source_cases: tuple[wire.FormalSourceCaseV3, ...] = field(repr=False)
    labels: tuple[auditor.FormalAuthorityLabelInputV1, ...] = field(repr=False)
    global_transform_catalog: wire.FormalGlobalTransformCatalogV3 = field(
        repr=False
    )
    transform_catalog_binding_receipt: wire.FormalGlobalTransformCatalogBindingReceiptV3 = field(
        repr=False
    )
    source_case_commitments: tuple[str, ...]
    run_id: bytes = field(repr=False)
    key_sources: wire.FormalWireKeySourcesV3 = field(repr=False)
    build_success: wire.FormalTrustedWireBuildSuccessV3 = field(repr=False)
    adapter: auditor.FormalWireV3AdapterV1 = field(repr=False)
    authority: auditor.FormalAuditAuthorityV1 = field(repr=False)
    recognizer: _SyntheticRichRecognizerV2 = field(repr=False)
    run_fixture_root: str
    source_pair_delta_root: str | None
    operation_counts: tuple[tuple[str, int], ...]
    label_factorial_proof_root: str
    namespace_row_proof_root: str
    feature_capacity_projection: auditor._FeatureCardinalityProjectionV1 = field(
        repr=False
    )
    exact_owned_channel_only_leaf_paths: int
    channel_only_owner_union_root: str

    def __post_init__(self) -> None:
        if self.mode not in _RUN_MODES:
            raise ValueError("fixture execution mode drift")
        if type(self.row_count) is not int or not 1 <= self.row_count <= 1024:
            raise ValueError("fixture execution row count drift")
        if len(self.source_cases) != self.row_count or len(self.labels) != self.row_count:
            raise ValueError("fixture execution row coverage drift")
        if type(self.build_success) is not wire.FormalTrustedWireBuildSuccessV3:
            raise TypeError("fixture execution lacks exact Wire build success")
        if type(self.adapter) is not auditor.FormalWireV3AdapterV1:
            raise TypeError("fixture execution lacks exact Wire auditor adapter")
        if type(self.authority) is not auditor.FormalAuditAuthorityV1:
            raise TypeError("fixture execution lacks exact bound authority")
        _validate_fixture_execution_feature_capacity_projection_v2(
            self.mode,
            self.row_count,
            self.feature_capacity_projection,
        )
        _validate_fixture_channel_only_owner_union_v2(
            self.mode,
            self.row_count,
            self.exact_owned_channel_only_leaf_paths,
            self.channel_only_owner_union_root,
        )
        if (
            type(self.label_factorial_proof_root) is not str
            or not self.label_factorial_proof_root.startswith(
                "phase2b_formal_wire_covert_label_factorial_v2_"
            )
        ):
            raise ValueError("fixture execution lacks its label-factorial proof root")
        if self.mode == "KNOWN_NULL" and self.source_pair_delta_root is not None:
            raise ValueError("known-null fixture cannot claim a pair delta")
        if self.mode == "KNOWN_BINDING_LEAK" and type(
            self.source_pair_delta_root
        ) is not str:
            raise ValueError("known-leak fixture needs its isolated pair delta root")


@dataclass(frozen=True, slots=True)
class _IsolatedAuditEvidenceV2:
    mode: str
    disposition: str
    run_fixture_root: str
    source_pair_delta_root: str | None
    transform_catalog_root: str
    operation_counts: tuple[tuple[str, int], ...]
    label_factorial_proof_root: str
    namespace_row_proof_root: str
    namespace_uuid_leaf_paths: int
    channel_leaf_paths: int
    exact_owned_channel_only_leaf_paths: int
    channel_only_owner_union_root: str
    feature_capacity_projection: auditor._FeatureCardinalityProjectionV1 = field(
        repr=False
    )
    feature_capacity_projection_root: str
    full_raw_byte_categorical_hypotheses: int
    diagnostics: AuditRunDiagnosticsV2
    source_row_index_by_public_row: tuple[int, ...] = field(repr=False)
    audit_receipt: auditor.FormalCovertAuditReceiptV1 = field(repr=False)
    validated_receipt_id: str
    build_success_id: str
    batch_id: str
    key_source_receipt_id: str
    private_alignment_id: str
    variant_context_id: str
    fresh_build_verification_root: str
    leak_feature_id: str | None = None
    leak_feature_aliases: tuple[str, ...] | None = None
    leak_raw_nmi_hex: str | None = None
    leak_bias_corrected_nmi_hex: str | None = None
    leak_balanced_accuracy_advantage_hex: str | None = None
    leak_observed_omnibus_hex: str | None = None
    leak_feature_max_t_p: str | None = None
    leak_target_holm_critical_value: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in _RUN_MODES or self.disposition not in ("PASS", "FAIL"):
            raise ValueError("isolated audit evidence mode/disposition drift")
        for value in (
            self.run_fixture_root,
            self.transform_catalog_root,
            self.label_factorial_proof_root,
            self.namespace_row_proof_root,
            self.channel_only_owner_union_root,
            self.feature_capacity_projection_root,
            self.build_success_id,
            self.batch_id,
            self.key_source_receipt_id,
            self.private_alignment_id,
            self.variant_context_id,
            self.fresh_build_verification_root,
            self.validated_receipt_id,
        ):
            if type(value) is not str or not value:
                raise TypeError("isolated audit evidence root must be nonempty")
        for name in (
            "namespace_uuid_leaf_paths",
            "channel_leaf_paths",
            "exact_owned_channel_only_leaf_paths",
            "full_raw_byte_categorical_hypotheses",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise TypeError("isolated audit evidence count must be positive")
        if tuple(sorted(self.source_row_index_by_public_row)) != tuple(
            range(EXACT_AUTHORITY_ROWS)
        ):
            raise ValueError("isolated audit evidence lost the secret-shuffle bijection")
        self.diagnostics.__post_init__()
        if type(self.audit_receipt) is not auditor.FormalCovertAuditReceiptV1:
            raise TypeError("audit evidence lacks exact same-process receipt")
        _require_auditor_receipt_id_v2(
            self.validated_receipt_id,
            "isolated audit evidence receipt ID",
        )
        if self.validated_receipt_id != self.audit_receipt.receipt_id:
            raise ValueError("audit evidence validated receipt ID drift")
        _validate_fixture_channel_only_owner_union_v2(
            self.mode,
            EXACT_AUTHORITY_ROWS,
            self.exact_owned_channel_only_leaf_paths,
            self.channel_only_owner_union_root,
        )
        _validate_fixture_execution_feature_capacity_projection_v2(
            self.mode,
            EXACT_AUTHORITY_ROWS,
            self.feature_capacity_projection,
        )
        if (
            self.feature_capacity_projection_root
            != self.feature_capacity_projection.projection_root
        ):
            raise ValueError("audit evidence feature-capacity root drift")
        if self.mode == "KNOWN_NULL":
            if (
                self.disposition != "PASS"
                or self.source_pair_delta_root is not None
                or any(
                    value is not None
                    for value in (
                        self.leak_feature_id,
                        self.leak_feature_aliases,
                        self.leak_raw_nmi_hex,
                        self.leak_bias_corrected_nmi_hex,
                        self.leak_balanced_accuracy_advantage_hex,
                        self.leak_observed_omnibus_hex,
                        self.leak_feature_max_t_p,
                        self.leak_target_holm_critical_value,
                    )
                )
            ):
                raise ValueError("known-null evidence is incomplete")
        elif (
            self.disposition != "FAIL"
            or type(self.source_pair_delta_root) is not str
            or type(self.leak_feature_aliases) is not tuple
            or not self.leak_feature_aliases
            or not any(
                _is_registered_binding_leak_alias_v2(alias)
                for alias in self.leak_feature_aliases
            )
            or any(
                type(value) is not str
                for value in (
                    self.leak_feature_id,
                    self.leak_raw_nmi_hex,
                    self.leak_bias_corrected_nmi_hex,
                    self.leak_balanced_accuracy_advantage_hex,
                    self.leak_observed_omnibus_hex,
                    self.leak_feature_max_t_p,
                    self.leak_target_holm_critical_value,
                )
            )
        ):
            raise ValueError("known-leak evidence omits E1 witness diagnostics")


def _registered_synthetic_kat_vectors_v2(mode: str) -> tuple[bytes, ...]:
    """Reproducible fixture vectors; never evidence of independence or origin."""

    if mode not in _RUN_MODES:
        raise ValueError("unknown synthetic KAT vector mode")
    labels = ("shuffle", "identifier", "padding", "latent", "answer", "run")
    values = tuple(
        hashlib.sha256(
            SYNTHETIC_KAT_VECTOR_DOMAIN
            + mode.encode("ascii")
            + b"\x00"
            + label.encode("ascii")
        ).digest()
        for label in labels
    )
    if len(set(values)) != len(values):
        raise RuntimeError("registered synthetic KAT vectors are not distinct")
    return values


def _run_fixture_root_v2(
    mode: str,
    authority: auditor.FormalAuditAuthorityV1,
    catalog_root: str,
    source_commitments: tuple[str, ...],
    label_factorial_proof_root: str,
) -> str:
    body = {
        "authority_transcript_root": authority.transcript_root,
        "mode": mode,
        "label_factorial_proof_root": label_factorial_proof_root,
        "source_case_commitments_in_order": list(source_commitments),
        "transform_catalog_root": catalog_root,
    }
    return "phase2b_formal_wire_covert_run_fixture_v2_" + hashlib.sha256(
        RUN_FIXTURE_DOMAIN + canonicalize_rfc8785_v1(body)
    ).hexdigest()


def _validate_fixture_transform_catalog_contract_v2(
    *,
    catalog: wire.FormalGlobalTransformCatalogV3,
    operation_counts: tuple[tuple[str, int], ...],
    row_count: int,
) -> None:
    if (
        type(catalog) is not wire.FormalGlobalTransformCatalogV3
        or type(row_count) is not int
        or not 1 <= row_count <= EXACT_AUTHORITY_ROWS
        or type(operation_counts) is not tuple
        or tuple(operation for operation, _ in operation_counts)
        != EXACT_TRANSFORM_OPERATIONS
        or any(type(count) is not int or count < 0 for _, count in operation_counts)
        or sum(count for _, count in operation_counts) != row_count
    ):
        raise ValueError("fixture transform catalog expectation is malformed")
    catalog_operation_counts = tuple(
        (
            operation,
            sum(entry.operation == operation for entry in catalog.entries),
        )
        for operation in EXACT_TRANSFORM_OPERATIONS
    )
    if (
        len(catalog.entries) != row_count
        or catalog_operation_counts != operation_counts
    ):
        raise ValueError(
            "fixture transform catalog is not the exact one UUID-bearing "
            "source use per row and operation mapping"
        )


def _build_fixture_execution_v2(
    mode: str,
    *,
    row_count: int = EXACT_AUTHORITY_ROWS,
) -> _FixtureExecutionV2:
    if mode not in _RUN_MODES:
        raise ValueError("unknown integrated qualification run mode")
    if type(row_count) is not int or not 1 <= row_count <= EXACT_AUTHORITY_ROWS:
        raise ValueError("fixture row count is outside 1..1024")
    labels = _synthetic_label_inputs_v2(row_count)
    cases = tuple(
        _synthetic_source_case_v2(
            index,
            binding_leak=mode == "KNOWN_BINDING_LEAK",
            operation_index=(None if row_count == 1024 else index % 8),
        )
        for index in range(row_count)
    )
    source_pair_delta_root: str | None = None
    paired_null_cases: tuple[wire.FormalSourceCaseV3, ...] = ()
    pair_rows: list[dict[str, object]] = []
    if mode == "KNOWN_BINDING_LEAK":
        null_cases: list[wire.FormalSourceCaseV3] = []
        for index, leak_case in enumerate(cases):
            null_case = _synthetic_source_case_v2(
                index,
                binding_leak=False,
                operation_index=(None if row_count == 1024 else index % 8),
            )
            restored_mask = replace(
                leak_case.graph.eligibility_mask,
                observation_ids=(
                    null_case.graph.eligibility_mask.observation_ids
                ),
            )
            restored = wire.FormalSourceCaseV3(
                replace(
                    leak_case.graph,
                    eligibility_mask=restored_mask,
                )
            )
            if restored != null_case:
                raise ValueError(
                    "known leak changes a field outside eligibility observation IDs"
                )
            null_count = len(null_case.graph.eligibility_mask.observation_ids)
            leak_count = len(leak_case.graph.eligibility_mask.observation_ids)
            binding = _authority_label_input_v2(index).binding
            if leak_count != (1 if binding == "binding-a" else 2):
                raise ValueError("known leak eligibility observation oracle drift")
            null_cases.append(null_case)
            pair_rows.append(
                {
                    "binding": binding,
                    "leak_graph_root": stable_hash(leak_case.graph),
                    "leak_observation_ids": list(
                        leak_case.graph.eligibility_mask.observation_ids
                    ),
                    "leak_observation_count": leak_count,
                    "null_graph_root": stable_hash(null_case.graph),
                    "null_observation_ids": list(
                        null_case.graph.eligibility_mask.observation_ids
                    ),
                    "null_observation_count": null_count,
                    "operation": leak_case.graph.transforms[0].operation,
                    "row_index": index,
                }
            )
        paired_null_cases = tuple(null_cases)
    catalog, catalog_receipt = wire.issue_formal_global_transform_catalog_v3(
        source_cases=cases
    )
    if wire.verify_formal_global_transform_catalog_binding_v3(
        catalog=catalog,
        binding_receipt=catalog_receipt,
    ) is not True:
        raise ValueError("global transform catalog binding did not replay")
    operation_counts = tuple(
        (operation, sum(case.graph.transforms[0].operation == operation for case in cases))
        for operation in EXACT_TRANSFORM_OPERATIONS
    )
    if row_count == EXACT_AUTHORITY_ROWS and any(
        count != EXACT_ROWS_PER_TRANSFORM_OPERATION
        for _, count in operation_counts
    ):
        raise ValueError("full fixture does not contain exact 8 x 128 operations")
    cells: dict[tuple[str, ...], list[int]] = {}
    for label in labels:
        cell = (
            label.case_type,
            label.family,
            label.scale,
            label.binding,
            label.answerability.value,
            label.joint_decision_class,
        )
        cells.setdefault(cell, []).append(label.row_index)
    factor_levels = {
        "answerability": sorted({label.answerability.value for label in labels}),
        "binding": sorted({label.binding for label in labels}),
        "case_type": sorted({label.case_type for label in labels}),
        "family": sorted({label.family for label in labels}),
        "joint_decision_class": sorted(
            {label.joint_decision_class for label in labels}
        ),
        "scale": sorted({label.scale for label in labels}),
    }
    cell_rows: list[dict[str, object]] = []
    for cell, indices in sorted(cells.items()):
        crossed = {
            operation: sum(
                cases[index].graph.transforms[0].operation == operation
                for index in indices
            )
            for operation in EXACT_TRANSFORM_OPERATIONS
        }
        cell_rows.append(
            {
                "label_cell": list(cell),
                "operation_counts": crossed,
                "row_count": len(indices),
            }
        )
    if row_count == EXACT_AUTHORITY_ROWS:
        expected_levels = {
            "answerability": ["abstain", "answerable"],
            "binding": ["binding-a", "binding-b"],
            "case_type": ["synthetic-type-a", "synthetic-type-b"],
            "family": ["F01", "F02"],
            "joint_decision_class": ["joint-a", "joint-b"],
            "scale": ["child", "root"],
        }
        if factor_levels != expected_levels:
            raise ValueError("full fixture does not contain the exact six binary factors")
        if len(cells) != 64 or any(len(indices) != 16 for indices in cells.values()):
            raise ValueError("full fixture is not the exact 64-cell x16 factorial")
        if any(
            count != 2
            for row in cell_rows
            for count in row["operation_counts"].values()
        ):
            raise ValueError(
                "full fixture does not cross every label cell with every operation twice"
            )
    label_factorial_proof_root = stable_hash(
        {
            "factor_levels": factor_levels,
            "label_cells": cell_rows,
            "row_count": row_count,
        },
        prefix="phase2b_formal_wire_covert_label_factorial_v2_",
    )
    _validate_fixture_transform_catalog_contract_v2(
        catalog=catalog,
        operation_counts=operation_counts,
        row_count=row_count,
    )
    source_commitments = tuple(
        wire.source_case_commitment_v3(
            case,
            transform_catalog_root=catalog_receipt.transform_catalog_root,
        )
        for case in cases
    )
    if mode == "KNOWN_BINDING_LEAK":
        null_commitments = tuple(
            wire.source_case_commitment_v3(
                case,
                transform_catalog_root=catalog_receipt.transform_catalog_root,
            )
            for case in paired_null_cases
        )
        for index, row in enumerate(pair_rows):
            row["leak_source_case_commitment"] = source_commitments[index]
            row["null_source_case_commitment"] = null_commitments[index]
        source_pair_delta_root = stable_hash(
            {
                "only_allowed_delta": "eligibility_mask.observation_ids",
                "transform_catalog_root": catalog_receipt.transform_catalog_root,
                "rows": pair_rows,
            },
            prefix="phase2b_formal_wire_covert_source_pair_delta_v2_",
        )
    shuffle, identifier, padding, latent, answer, run_id = (
        _registered_synthetic_kat_vectors_v2(mode)
    )
    key_sources = wire.FormalWireKeySourcesV3(shuffle, identifier, padding)
    key_receipt = wire.issue_key_source_transcript_receipt_v3(
        key_sources=key_sources,
        latent_seed_commitments=(wire.seed_commitment_v3(latent),),
        answer_seed_commitments=(wire.seed_commitment_v3(answer),),
    )
    build_success = wire.build_formal_trusted_wire_batch_v3(
        source_cases=cases,
        global_transform_catalog=catalog,
        transform_catalog_binding_receipt=catalog_receipt,
        run_id=run_id,
        key_sources=key_sources,
        key_source_receipt=key_receipt,
    )
    if type(build_success) is not wire.FormalTrustedWireBuildSuccessV3:
        raise ValueError(
            "formal Wire V3 rejected the exact synthetic fixture: "
            + getattr(build_success, "reason", "UNKNOWN")
        )
    if wire.verify_formal_trusted_wire_build_success_v3(
        build_success=build_success,
        run_id=run_id,
        key_sources=key_sources,
        source_cases=cases,
        global_transform_catalog=catalog,
        transform_catalog_binding_receipt=catalog_receipt,
    ) is not True:
        raise ValueError("external fresh Wire build verifier did not return exact true")
    adapter = auditor.FormalWireV3AdapterV1(
        build_success,
        run_id=run_id,
        key_sources=key_sources,
        source_cases=cases,
        global_transform_catalog=catalog,
        transform_catalog_binding_receipt=catalog_receipt,
    )
    authority = auditor.issue_formal_wire_v3_audit_authority_v1(
        build_success, labels
    )
    authority_binding = adapter.bind_private_authority(authority, adapter.envelopes)
    if (
        authority_binding.alignment_issuance_verified is not True
        or authority_binding.label_row_binding_verified is not True
    ):
        raise ValueError("wire-issued private authority did not cross-verify")
    namespace_rows: list[dict[str, object]] = []
    decoded_rows: list[auditor.DecodedFormalEnvelopeV1] = []
    required_namespaces = set(wire.NAMESPACE_REGISTRY_V3)
    auditor._preflight_decode_memory(row_count)
    for row_index, envelope in enumerate(adapter.envelopes):
        decoded = adapter.decode_envelope(envelope)
        if type(decoded) is not auditor.DecodedFormalEnvelopeV1:
            raise TypeError("fixture Wire adapter returned a non-exact decoded DTO")
        auditor._validate_decoded(decoded, adapter)
        if (
            auditor.DEEP_SIZE_SAFETY_MULTIPLIER
            * auditor._deep_object_graph_bytes(decoded)
            > auditor.MAXIMUM_DECODE_GRAPH_BYTES_PER_ROW
        ):
            raise ValueError(
                "fixture decoded row exceeds the frozen conservative graph bound"
            )
        decoded_rows.append(decoded)
        decision_projection = (
            _verify_lightweight_trusted_projection_equivalence_v2(
                envelope, decoded
            )
        )
        observed_namespaces = {item.namespace for item in decoded.namespace_leaves}
        if observed_namespaces != required_namespaces:
            raise ValueError(
                f"rich fixture row {row_index} does not populate exact-ten namespaces"
            )
        namespace_rows.append(
            {
                "envelope_sha256": "sha256:"
                + hashlib.sha256(envelope).hexdigest(),
                "namespace_leaf_evidence_root": stable_hash(
                    tuple(
                        (leaf.namespace, leaf.path, leaf.public_uuid)
                        for leaf in decoded.namespace_leaves
                    ),
                    prefix="phase2b_formal_wire_covert_namespace_leaf_row_v2_",
                ),
                "namespaces": sorted(observed_namespaces),
                "lightweight_trusted_decision_projection": {
                    "family": decision_projection[0],
                    "bindings": [
                        [binding.role_candidate_id, binding.entity_id]
                        for binding in decision_projection[1]
                    ],
                    "roots": list(decision_projection[2]),
                },
                "row_index": row_index,
            }
        )
    namespace_row_proof_root = stable_hash(
        {
            "batch_id": build_success.public_batch.batch_id,
            "build_success_id": build_success.success_id,
            "rows": namespace_rows,
        },
        prefix="phase2b_formal_wire_covert_namespace_row_proof_v2_",
    )
    decoded_tuple = tuple(decoded_rows)
    feature_capacity_projection = (
        _validate_fixture_execution_feature_capacity_projection_v2(
            mode,
            row_count,
            auditor._feature_cardinality_projection(decoded_tuple),
        )
    )
    (
        exact_owned_channel_only_leaf_paths,
        channel_only_owner_union_root,
    ) = _channel_only_owner_union_projection_v2(
        owner
        for decoded_row in decoded_tuple
        for owner in decoded_row.channel_only_leaf_owners
    )
    _validate_fixture_channel_only_owner_union_v2(
        mode,
        row_count,
        exact_owned_channel_only_leaf_paths,
        channel_only_owner_union_root,
    )
    del decoded, decoded_rows, decoded_tuple
    recognizer = _SyntheticRichRecognizerV2(adapter)
    run_fixture_root = _run_fixture_root_v2(
        mode,
        authority,
        catalog_receipt.transform_catalog_root,
        source_commitments,
        label_factorial_proof_root,
    )
    execution = _FixtureExecutionV2(
        mode,
        row_count,
        cases,
        labels,
        catalog,
        catalog_receipt,
        source_commitments,
        run_id,
        key_sources,
        build_success,
        adapter,
        authority,
        recognizer,
        run_fixture_root,
        source_pair_delta_root,
        operation_counts,
        label_factorial_proof_root,
        namespace_row_proof_root,
        feature_capacity_projection,
        exact_owned_channel_only_leaf_paths,
        channel_only_owner_union_root,
    )
    execution.__post_init__()
    return execution


def _validate_fixture_feature_capacity_projection_v2(
    mode: str,
    projection: auditor._FeatureCardinalityProjectionV1,
) -> auditor._FeatureCardinalityProjectionV1:
    if mode not in _RUN_MODES or type(
        projection
    ) is not auditor._FeatureCardinalityProjectionV1:
        raise TypeError("fixture feature capacity needs an exact mode and projection")
    projection.__post_init__()
    mode_contract = _FIXTURE_FEATURE_CAPACITY_CONTRACT_V2["modes"][mode]
    expected_projection = _fixture_feature_capacity_projection_row_v2(
        mode,
        projection.rows,
    )
    observed_union_shape = {
        "channel_union_path_count": projection.channel_union_path_count,
        "descriptor_segment_position_count": (
            projection.descriptor_segment_position_count
        ),
        "descriptor_union_path_count": projection.descriptor_union_path_count,
        "eligibility_absent_feature_count": (
            projection.eligibility_absent_feature_count
        ),
        "generic_channel_union_path_count": (
            projection.generic_channel_union_path_count
        ),
        "provenance_union_path_count": projection.provenance_union_path_count,
        "uuid_union_path_count": projection.uuid_union_path_count,
    }
    observed_projection = {
        "projected_concrete_feature_count": (
            projection.projected_concrete_feature_count
        ),
        "projected_feature_matrix_bytes_bound": (
            projection.projected_feature_matrix_bytes_bound
        ),
        "projected_raw_semantic_feature_count": (
            projection.projected_raw_semantic_feature_count
        ),
        "projection_root": projection.projection_root,
        "rows": projection.rows,
    }
    if (
        observed_union_shape != mode_contract["union_shape"]
        or observed_projection != expected_projection
        or projection.projected_concrete_feature_count > auditor.MAXIMUM_FEATURES
    ):
        raise ValueError("fixture feature-capacity projection drift")
    return projection


def _fixture_feature_capacity_projection_row_v2(
    mode: str,
    rows: int,
) -> dict[str, object]:
    if mode not in _RUN_MODES or type(rows) is not int:
        raise TypeError("fixture feature-capacity lookup needs an exact mode and rows")
    matches = tuple(
        row
        for row in _FIXTURE_FEATURE_CAPACITY_CONTRACT_V2["modes"][mode][
            "projections"
        ]
        if row["rows"] == rows
    )
    if len(matches) != 1:
        raise ValueError("fixture feature-capacity row is not frozen")
    return matches[0]


def _fixture_capacity_receipt_projection_from_execution_v2(
    execution: _FixtureExecutionV2,
) -> dict[str, object]:
    """Project exactly the six real receipt fields from decoded construction."""

    if type(execution) is not _FixtureExecutionV2:
        raise TypeError("receipt capacity execution binding needs exact execution")
    projection = _validate_fixture_execution_feature_capacity_projection_v2(
        execution.mode,
        execution.row_count,
        execution.feature_capacity_projection,
    )
    return {
        "audited_rows": execution.row_count,
        "channel_leaf_count": projection.channel_union_path_count,
        "concrete_feature_count": projection.projected_concrete_feature_count,
        "preextraction_feature_matrix_bytes_bound": (
            projection.projected_feature_matrix_bytes_bound
        ),
        "raw_semantic_feature_count": (
            projection.projected_raw_semantic_feature_count
        ),
        "uuid_leaf_count": projection.uuid_union_path_count,
    }


def _validate_closed_feature_capacity_projection_v2(
    mode: str,
    rows: int,
    value: object,
) -> dict[str, object]:
    fields = dataclass_fields(auditor._FeatureCardinalityProjectionV1)
    expected_keys = {item.name for item in fields}
    if type(value) is not dict or set(value) != expected_keys:
        raise TypeError("closed feature-capacity projection fields are not exact")
    try:
        replay = auditor._FeatureCardinalityProjectionV1(
            **{item.name: value[item.name] for item in fields if item.init}
        )
        replay.__post_init__()
    except (TypeError, ValueError) as exc:
        raise ValueError("closed feature-capacity projection semantics drift") from exc
    _validate_fixture_execution_feature_capacity_projection_v2(
        mode,
        rows,
        replay,
    )
    if (
        value["projection_root"] != replay.projection_root
        or _closed_primitive_v2(replay) != value
    ):
        raise ValueError("closed feature-capacity projection root drift")
    return value


def _closed_feature_capacity_projection_v2(
    mode: str,
    rows: int,
    projection: auditor._FeatureCardinalityProjectionV1,
) -> dict[str, object]:
    _validate_fixture_execution_feature_capacity_projection_v2(
        mode,
        rows,
        projection,
    )
    value = _closed_primitive_v2(projection)
    return _validate_closed_feature_capacity_projection_v2(mode, rows, value)


def _validate_fixture_capacity_receipt_projection_v2(
    mode: str,
    projection: dict[str, object],
) -> dict[str, object]:
    expected_keys = {
        "audited_rows",
        "channel_leaf_count",
        "concrete_feature_count",
        "preextraction_feature_matrix_bytes_bound",
        "raw_semantic_feature_count",
        "uuid_leaf_count",
    }
    if (
        mode not in _RUN_MODES
        or type(projection) is not dict
        or set(projection) != expected_keys
        or any(type(projection[name]) is not int for name in expected_keys)
    ):
        raise TypeError("fixture receipt capacity projection is not exact")
    expected_projection = _fixture_feature_capacity_projection_row_v2(
        mode,
        EXACT_AUTHORITY_ROWS,
    )
    union_shape = _FIXTURE_FEATURE_CAPACITY_CONTRACT_V2["modes"][mode][
        "union_shape"
    ]
    expected = {
        "audited_rows": EXACT_AUTHORITY_ROWS,
        "channel_leaf_count": union_shape["channel_union_path_count"],
        "concrete_feature_count": expected_projection[
            "projected_concrete_feature_count"
        ],
        "preextraction_feature_matrix_bytes_bound": expected_projection[
            "projected_feature_matrix_bytes_bound"
        ],
        "raw_semantic_feature_count": expected_projection[
            "projected_raw_semantic_feature_count"
        ],
        "uuid_leaf_count": union_shape["uuid_union_path_count"],
    }
    if projection != expected:
        raise ValueError("full fixture receipt capacity projection drift")
    return projection


def _validate_isolated_audit_receipt_v2(
    mode: str,
    started: int,
    execution: _FixtureExecutionV2,
    receipt: auditor.FormalCovertAuditReceiptV1,
) -> _IsolatedAuditEvidenceV2:
    adapter = execution.adapter
    validated_receipt_id = auditor.validate_formal_covert_audit_receipt_v1(receipt)
    if validated_receipt_id != receipt.receipt_id:
        raise ValueError("same-process audit receipt validation ID drift")
    _validate_fixture_channel_only_owner_union_v2(
        mode,
        execution.row_count,
        execution.exact_owned_channel_only_leaf_paths,
        execution.channel_only_owner_union_root,
    )
    observed_receipt_capacity = {
        "audited_rows": receipt.audited_row_count,
        "channel_leaf_count": receipt.feature_manifest.channel_leaf_count,
        "concrete_feature_count": (
            receipt.feature_manifest.deduplicated_feature_count
        ),
        "preextraction_feature_matrix_bytes_bound": (
            receipt.resource_plan.preextraction_feature_matrix_bytes_bound
        ),
        "raw_semantic_feature_count": (
            receipt.feature_manifest.raw_semantic_feature_count
        ),
        "uuid_leaf_count": receipt.feature_manifest.uuid_leaf_count,
    }
    _validate_fixture_capacity_receipt_projection_v2(
        mode,
        observed_receipt_capacity,
    )
    if observed_receipt_capacity != (
        _fixture_capacity_receipt_projection_from_execution_v2(execution)
    ):
        raise ValueError("receipt capacity differs from decoded fixture execution")
    invariant_runtime = receipt.invariant_runtime_evidence
    if type(invariant_runtime) is not auditor.InvariantRuntimeEvidenceV1:
        raise ValueError("full audit omitted invariant runtime evidence")
    invariant_runtime.__post_init__()
    auditor._validate_invariant_phase_preflight_against_resource_plan(
        receipt.resource_plan,
        invariant_runtime.resource_preflight,
    )
    if (
        invariant_runtime.invariance_resource_plan_root
        != receipt.resource_plan.invariance_resource_plan_root
    ):
        raise ValueError("invariant runtime/resource-plan root drift")
    raw_byte_count = sum(
        any(
            alias.endswith("/raw_byte_categorical")
            for alias in item.semantic_aliases
        )
        for item in receipt.feature_manifest.alias_coverage
    )
    if raw_byte_count <= 0:
        raise ValueError("full raw-byte categorical hypotheses were reduced")
    expected = (
        auditor.FormalCovertDisposition.PASS
        if mode == "KNOWN_NULL"
        else auditor.FormalCovertDisposition.FAIL
    )
    if receipt.disposition is not expected:
        raise ValueError(
            f"{mode} observed {receipt.disposition.value}, expected {expected.value}"
        )
    if (
        receipt.audited_row_count != EXACT_AUTHORITY_ROWS
        or receipt.invariant_run_count != EXACT_RECOGNIZER_INVOCATIONS
        or receipt.exact_invariant_comparisons
        != EXACT_AUTHORITY_ROWS * 80 * 5
        or invariant_runtime.consumer_decode_count
        != EXACT_AUTHORITY_ROWS * 80
        or receipt.resource_plan.label_permutations != EXACT_LABEL_PERMUTATIONS
        or not receipt.resource_plan.admissible
        or not receipt.wire_alignment_issuance_verified
        or not receipt.eligibility_mask_quota.exact_quota_matched
        or any(
            target.maximum_absolute_cross_product_difference != 0
            or target.exact_quota_matched is not True
            for target in receipt.eligibility_mask_quota.targets
        )
        or receipt.holm_family_size != len(auditor.TARGETS)
        or len(receipt.target_family_results) != len(auditor.TARGETS)
        or receipt.pairwise_holm_decision_eligible is not False
        or receipt.statistics_runtime_evidence is None
    ):
        raise ValueError("integrated auditor coverage or hard gate drift")
    leak_feature_id: str | None = None
    leak_feature_aliases: tuple[str, ...] | None = None
    leak_nmi_hex: str | None = None
    leak_bias_hex: str | None = None
    leak_ba_hex: str | None = None
    leak_omnibus_hex: str | None = None
    leak_max_t_p: str | None = None
    leak_critical: str | None = None
    if mode == "KNOWN_BINDING_LEAK":
        binding_family = next(
            row for row in receipt.target_family_results if row.target == "binding"
        )
        registered_presence_alias = _registered_binding_leak_presence_alias_v2(
            execution
        )
        alias_by_feature = {
            row.concrete_feature_id: row.semantic_aliases
            for row in receipt.feature_manifest.alias_coverage
        }
        witnesses = tuple(
            row
            for row in receipt.results
            if row.target == "binding"
            and row.failure_witness
            and registered_presence_alias
            in alias_by_feature.get(row.feature_id, ())
        )
        if binding_family.holm_rejected is not True or len(witnesses) != 1:
            raise ValueError("known binding leak lacks target-family E1 witness")
        witness = witnesses[0]
        if (
            witness.observed_omnibus <= 1.0
            or witness.material_effect is not True
            or witness.target_family_holm_rejected is not True
            or witness.feature_localization_rejected is not True
            or witness.passed is not False
        ):
            raise ValueError("known binding leak witness does not satisfy E1")
        leak_feature_id = witness.feature_id
        leak_feature_aliases = alias_by_feature[witness.feature_id]
        leak_nmi_hex = witness.raw_nmi.hex()
        leak_bias_hex = witness.bias_corrected_nmi.hex()
        leak_ba_hex = witness.balanced_accuracy_advantage.hex()
        leak_omnibus_hex = witness.observed_omnibus.hex()
        leak_max_t_p = (
            f"{witness.feature_max_t_p.numerator}/"
            f"{witness.feature_max_t_p.denominator}"
        )
        leak_critical = (
            f"{witness.target_holm_critical_value.numerator}/"
            f"{witness.target_holm_critical_value.denominator}"
        )
    elif (
        receipt.positive_decision_claim != auditor.POSITIVE_DECISION_CLAIM
        or any(row.failure_witness for row in receipt.results)
    ):
        raise ValueError("known-null positive claim or witness boundary drift")
    runtime = receipt.statistics_runtime_evidence
    if type(runtime) is not auditor.StatisticalRuntimeEvidenceV1:
        raise ValueError("full audit omitted statistical runtime evidence")
    runtime.__post_init__()
    auditor._validate_statistics_phase_preflight_against_resource_plan(
        receipt.resource_plan,
        runtime.resource_preflight,
    )
    if runtime.resource_plan_root != receipt.resource_plan.resource_plan_root:
        raise ValueError("statistics runtime/resource-plan root drift")
    issuance = receipt.receipt_issuance_runtime_evidence
    if type(issuance) is not auditor.ReceiptIssuanceRuntimeEvidenceV1:
        raise ValueError("full audit omitted branch-neutral issuance evidence")
    issuance.__post_init__()
    if (
        issuance.absolute_deadline_kind != "STATISTICS"
        or issuance.diagnostic_worker_historical_peak_rss_bytes_sum
        != sum(runtime.worker_peak_rss_bytes)
        or issuance.diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization
        != runtime.parent_historical_peak_rss_bytes_before_receipt_finalization
        or issuance.diagnostic_parent_plus_worker_historical_peak_sum_bytes
        != runtime.diagnostic_parent_plus_worker_historical_peak_sum_bytes
        or issuance.unified_cgroup_membership
        != runtime.unified_cgroup_membership
        or issuance.unified_cgroup_memory_scope_root
        != runtime.unified_cgroup_memory_scope_root
        or issuance.unified_cgroup_leaf_device
        != runtime.unified_cgroup_leaf_device
        or issuance.unified_cgroup_leaf_inode
        != runtime.unified_cgroup_leaf_inode
        or issuance.physical_memory_current_bytes_before_receipt_finalization
        != runtime.physical_memory_current_bytes_before_receipt_finalization
        or issuance.physical_memory_peak_bytes_before_receipt_finalization
        != runtime.physical_memory_peak_bytes_before_receipt_finalization
        or issuance.physical_memory_max_bytes
        != runtime.physical_memory_max_bytes
        or issuance.physical_memory_cap_bytes
        != runtime.physical_memory_cap_bytes
        or issuance.receipt_finalization_complete_pending_validation_count
        != runtime.receipt_finalization_complete_pending_validation_count
        or issuance.receipt_finalization_locked_physical_peak_within_cap
        is not runtime.receipt_finalization_physical_peak_within_cap
    ):
        raise ValueError("isolated statistical/issuance runtime binding drift")
    if any(
        value % 1_024
        for value in (
            *runtime.worker_peak_rss_bytes,
            runtime.parent_historical_peak_rss_bytes_before_receipt_finalization,
            runtime.diagnostic_parent_plus_worker_historical_peak_sum_bytes,
        )
    ):
        raise ValueError("runtime RSS evidence is not representable in Linux KiB units")
    if runtime.within_frozen_physical_peak_cap is not True:
        raise ValueError("statistics runtime exceeded the frozen physical peak cap")
    parent_historical_peak = (
        runtime.parent_historical_peak_rss_bytes_before_receipt_finalization // 1_024
    )
    worker_peaks = tuple(value // 1_024 for value in runtime.worker_peak_rss_bytes)
    historical_sum = (
        runtime.diagnostic_parent_plus_worker_historical_peak_sum_bytes // 1_024
    )
    diagnostics = AuditRunDiagnosticsV2(
        mode=mode,
        wall_time_ns=time.perf_counter_ns() - started,
        parent_historical_peak_rss_before_receipt_finalization_kib=(
            parent_historical_peak
        ),
        statistics_worker_historical_peak_rss_kib=worker_peaks,
        parent_plus_worker_historical_peak_rss_kib=historical_sum,
        unified_cgroup_membership=runtime.unified_cgroup_membership,
        unified_cgroup_memory_scope_root=(
            runtime.unified_cgroup_memory_scope_root
        ),
        unified_cgroup_leaf_device=runtime.unified_cgroup_leaf_device,
        unified_cgroup_leaf_inode=runtime.unified_cgroup_leaf_inode,
        physical_memory_current_before_receipt_finalization_bytes=(
            runtime.physical_memory_current_bytes_before_receipt_finalization
        ),
        physical_memory_peak_before_receipt_finalization_bytes=(
            runtime.physical_memory_peak_bytes_before_receipt_finalization
        ),
        physical_memory_max_bytes=runtime.physical_memory_max_bytes,
        physical_memory_cap_bytes=runtime.physical_memory_cap_bytes,
        statistics_wall_time_ns=(
            runtime.statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds
        ),
        host_available_memory_bytes_at_preflight=(
            runtime.host_available_memory_bytes_at_preflight
        ),
        recognizer_adapter_invocations=receipt.invariant_run_count,
        minimum_rows_per_invocation=receipt.audited_row_count,
        maximum_rows_per_invocation=receipt.audited_row_count,
    )
    build_success = execution.build_success
    evidence = _IsolatedAuditEvidenceV2(
        mode=mode,
        disposition=receipt.disposition.value,
        run_fixture_root=execution.run_fixture_root,
        source_pair_delta_root=execution.source_pair_delta_root,
        transform_catalog_root=(
            execution.transform_catalog_binding_receipt.transform_catalog_root
        ),
        operation_counts=execution.operation_counts,
        label_factorial_proof_root=execution.label_factorial_proof_root,
        namespace_row_proof_root=execution.namespace_row_proof_root,
        namespace_uuid_leaf_paths=receipt.feature_manifest.uuid_leaf_count,
        channel_leaf_paths=receipt.feature_manifest.channel_leaf_count,
        exact_owned_channel_only_leaf_paths=(
            execution.exact_owned_channel_only_leaf_paths
        ),
        channel_only_owner_union_root=(
            execution.channel_only_owner_union_root
        ),
        feature_capacity_projection=execution.feature_capacity_projection,
        feature_capacity_projection_root=(
            execution.feature_capacity_projection.projection_root
        ),
        full_raw_byte_categorical_hypotheses=raw_byte_count,
        diagnostics=diagnostics,
        source_row_index_by_public_row=(
            build_success.private_audit_alignment.source_row_index_by_public_row
        ),
        audit_receipt=receipt,
        validated_receipt_id=validated_receipt_id,
        build_success_id=build_success.success_id,
        batch_id=build_success.public_batch.batch_id,
        key_source_receipt_id=build_success.key_source_receipt.receipt_id,
        private_alignment_id=build_success.private_audit_alignment.alignment_id,
        variant_context_id=build_success.variant_context.context_id,
        fresh_build_verification_root=adapter.fresh_build_verification_root,
        leak_feature_id=leak_feature_id,
        leak_feature_aliases=leak_feature_aliases,
        leak_raw_nmi_hex=leak_nmi_hex,
        leak_bias_corrected_nmi_hex=leak_bias_hex,
        leak_balanced_accuracy_advantage_hex=leak_ba_hex,
        leak_observed_omnibus_hex=leak_omnibus_hex,
        leak_feature_max_t_p=leak_max_t_p,
        leak_target_holm_critical_value=leak_critical,
    )
    evidence.__post_init__()
    return evidence


def _execute_isolated_audit_v2(mode: str) -> _IsolatedAuditEvidenceV2:
    """Own and release any receipt issued before a post-audit rejection."""

    started = time.perf_counter_ns()
    execution = _build_fixture_execution_v2(mode)
    receipt: auditor.FormalCovertAuditReceiptV1 | None = None
    try:
        receipt = auditor.run_formal_covert_audit_v1(
            execution.adapter.envelopes,
            execution.authority,
            execution.adapter,
            execution.recognizer,
        )
        return _validate_isolated_audit_receipt_v2(
            mode, started, execution, receipt
        )
    except BaseException:
        if type(receipt) is auditor.FormalCovertAuditReceiptV1:
            try:
                auditor.release_formal_covert_audit_receipt_v1(receipt)
            except (KeyError, TypeError, ValueError):
                pass
        raise


class QualificationDispositionV2(str, Enum):
    CANDIDATE_IMPLEMENTATION_EXERCISED_NOT_R03_R04_R05 = (
        PROFILE_CLAIM_LEVEL
    )
    ATOMIC_EMPTY_REJECTION = "ATOMIC_EMPTY_REJECTION"


@dataclass(frozen=True, slots=True)
class Phase2BFormalWireCovertQualificationRejectionV2:
    disposition: QualificationDispositionV2
    reason_code: str
    profile_id: str | None
    qualification_receipt_id: None = None
    synthetic_fixture_root: None = None
    positive_audit_receipt_id: None = None
    positive_batch_id: None = None
    positive_invariant_execution_root: None = None
    positive_permutation_schedule_root: None = None
    negative_audit_receipt_id: None = None
    negative_input_root: None = None
    completed_audit_runs: int = 0

    def __post_init__(self) -> None:
        if (
            type(self) is not Phase2BFormalWireCovertQualificationRejectionV2
            or self.disposition is not QualificationDispositionV2.ATOMIC_EMPTY_REJECTION
        ):
            raise TypeError("qualification rejection exact disposition/type drift")
        if type(self.reason_code) is not str or not self.reason_code:
            raise ValueError("qualification rejection needs a nonempty reason code")
        if self.profile_id is not None and self.profile_id != PROFILE_ID:
            raise ValueError("qualification rejection profile ID drift")
        if any(
            value is not None
            for value in (
                self.qualification_receipt_id,
                self.synthetic_fixture_root,
                self.positive_audit_receipt_id,
                self.positive_batch_id,
                self.positive_invariant_execution_root,
                self.positive_permutation_schedule_root,
                self.negative_audit_receipt_id,
                self.negative_input_root,
            )
        ) or type(self.completed_audit_runs) is not int or self.completed_audit_runs != 0:
            raise ValueError("atomic rejection exposes partial qualification roots")


_RECEIPT_ISSUE_TOKEN: Final = object()


@dataclass(frozen=True, slots=True, init=False)
class Phase2BFormalWireCovertCandidateReceiptV2:
    disposition: QualificationDispositionV2
    profile_id: str
    claim_level: str
    synthetic_fixture_root: str
    dependency_lock_sha256: str
    positive_batch_id: str
    positive_transform_catalog_root: str
    positive_audit_receipt_id: str
    positive_diagnostics: AuditRunDiagnosticsV2
    positive_resource_plan_projection: dict[str, object]
    positive_resource_plan_projection_root: str
    positive_statistics_runtime_projection: dict[str, object]
    positive_statistics_runtime_projection_root: str
    positive_receipt_issuance_runtime_projection: dict[str, object]
    positive_receipt_issuance_runtime_projection_root: str
    negative_expected_fail_observed: bool
    negative_disposition: str
    negative_audit_receipt_id: str
    negative_diagnostics: AuditRunDiagnosticsV2
    negative_resource_plan_projection: dict[str, object]
    negative_resource_plan_projection_root: str
    negative_statistics_runtime_projection: dict[str, object]
    negative_statistics_runtime_projection_root: str
    negative_receipt_issuance_runtime_projection: dict[str, object]
    negative_receipt_issuance_runtime_projection_root: str
    negative_leak_feature_id: str
    negative_leak_feature_aliases: tuple[str, ...]
    negative_leak_observed_omnibus_hex: str
    negative_leak_feature_max_t_p: str
    negative_leak_target_holm_critical_value: str
    source_pair_delta_root: str
    operation_counts: tuple[tuple[str, int], ...]
    label_factorial_proof_root: str
    positive_feature_manifest_root: str
    positive_concrete_features: int
    positive_raw_semantic_features: int
    positive_unique_statistical_partitions: int
    positive_hypotheses: int
    namespace_uuid_leaf_paths: int
    channel_leaf_paths: int
    exact_owned_channel_only_leaf_paths: int
    channel_only_owner_union_root: str
    full_raw_byte_categorical_hypotheses: int
    source_commit_sha: None
    source_archive_root: None
    runtime_image_digest: None
    runtime_environment_manifest_root: None
    dependency_lock_live_verified: bool
    synthetic_fixture_archive_verified: bool
    r03_formal_trusted_wire_implementation_qualified: bool
    r04_formal_covert_auditor_implementation_qualified: bool
    r05_synthetic_dev_wire_covert_qualification_verified: bool
    actual_corpus_audit: bool
    actual_execution_claimed: bool
    qualification_receipt_id: str = field(init=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("candidate receipts are privately issued by qualification")

    @classmethod
    def _issue(
        cls,
        token: object,
        *,
        synthetic_fixture_root: str,
        positive: _IsolatedAuditEvidenceV2,
        negative: _IsolatedAuditEvidenceV2,
    ) -> "Phase2BFormalWireCovertCandidateReceiptV2":
        if token is not _RECEIPT_ISSUE_TOKEN:
            raise TypeError("candidate receipt issuer token mismatch")
        if positive.mode != "KNOWN_NULL" or negative.mode != "KNOWN_BINDING_LEAK":
            raise ValueError("candidate audit evidence order drift")
        if positive.label_factorial_proof_root != negative.label_factorial_proof_root:
            raise ValueError("paired candidate label-factorial proof roots differ")
        audit_receipt = positive.audit_receipt
        negative_receipt = negative.audit_receipt
        if (
            type(audit_receipt) is not auditor.FormalCovertAuditReceiptV1
            or type(negative_receipt) is not auditor.FormalCovertAuditReceiptV1
        ):
            raise ValueError("candidate audit receipt is absent")
        positive_receipt_id = _require_auditor_receipt_id_v2(
            positive.validated_receipt_id,
            "candidate positive audit receipt ID",
        )
        negative_receipt_id = _require_auditor_receipt_id_v2(
            negative.validated_receipt_id,
            "candidate negative audit receipt ID",
        )
        if (
            audit_receipt.receipt_id != positive_receipt_id
            or negative_receipt.receipt_id != negative_receipt_id
        ):
            raise ValueError("candidate validated receipt ID drift")
        runtime_projections: dict[
            str,
            tuple[dict[str, object], dict[str, object], dict[str, object]],
        ] = {}
        for name, evidence, receipt in (
            ("positive", positive, audit_receipt),
            ("negative", negative, negative_receipt),
        ):
            resource_plan = _closed_primitive_v2(receipt.resource_plan)
            statistics = receipt.statistics_runtime_evidence
            issuance = receipt.receipt_issuance_runtime_evidence
            if (
                type(statistics) is not auditor.StatisticalRuntimeEvidenceV1
                or type(issuance) is not auditor.ReceiptIssuanceRuntimeEvidenceV1
            ):
                raise ValueError(f"candidate {name} runtime evidence is absent")
            statistics_projection = _closed_primitive_v2(statistics)
            issuance_projection = _closed_primitive_v2(issuance)
            _validate_complete_resource_plan_projection_v2(
                resource_plan,
                context=f"candidate {name}",
            )
            _validate_complete_statistics_runtime_projection_v2(
                statistics_projection,
                resource_plan,
                context=f"candidate {name}",
            )
            _validate_complete_receipt_issuance_runtime_projection_v2(
                issuance_projection,
                statistics_projection,
                context=f"candidate {name}",
            )
            _validate_executed_runtime_bundle_v2(
                resource_plan=resource_plan,
                statistics_runtime=statistics_projection,
                receipt_issuance_runtime=issuance_projection,
                diagnostics=_diagnostic_projection(evidence.diagnostics),
                run=None,
                expected_mode=evidence.mode,
                visible_audited_rows=receipt.audited_row_count,
                visible_concrete_features=(
                    receipt.feature_manifest.deduplicated_feature_count
                ),
                context=f"candidate {name}",
            )
            runtime_projections[name] = (
                resource_plan,
                statistics_projection,
                issuance_projection,
            )
        (
            positive_resource_plan,
            positive_statistics,
            positive_issuance,
        ) = runtime_projections["positive"]
        (
            negative_resource_plan,
            negative_statistics,
            negative_issuance,
        ) = runtime_projections["negative"]
        _validate_candidate_branch_physical_scope_v2(
            positive_statistics,
            negative_statistics,
        )
        value = object.__new__(cls)
        assignments = {
            "disposition": QualificationDispositionV2.CANDIDATE_IMPLEMENTATION_EXERCISED_NOT_R03_R04_R05,
            "profile_id": PROFILE_ID,
            "claim_level": PROFILE_CLAIM_LEVEL,
            "synthetic_fixture_root": synthetic_fixture_root,
            "dependency_lock_sha256": DEPENDENCY_LOCK_SHA256,
            "positive_batch_id": positive.batch_id,
            "positive_transform_catalog_root": positive.transform_catalog_root,
            "positive_audit_receipt_id": positive_receipt_id,
            "positive_diagnostics": positive.diagnostics,
            "positive_resource_plan_projection": positive_resource_plan,
            "positive_resource_plan_projection_root": stable_hash(
                positive_resource_plan,
                prefix="phase2b_formal_wire_covert_resource_plan_v3_",
            ),
            "positive_statistics_runtime_projection": positive_statistics,
            "positive_statistics_runtime_projection_root": stable_hash(
                positive_statistics,
                prefix="phase2b_formal_wire_covert_statistics_runtime_v4_",
            ),
            "positive_receipt_issuance_runtime_projection": positive_issuance,
            "positive_receipt_issuance_runtime_projection_root": stable_hash(
                positive_issuance,
                prefix=(
                    "phase2b_formal_wire_covert_receipt_issuance_runtime_v4_"
                ),
            ),
            "negative_expected_fail_observed": True,
            "negative_disposition": negative.disposition,
            "negative_audit_receipt_id": negative_receipt_id,
            "negative_diagnostics": negative.diagnostics,
            "negative_resource_plan_projection": negative_resource_plan,
            "negative_resource_plan_projection_root": stable_hash(
                negative_resource_plan,
                prefix="phase2b_formal_wire_covert_resource_plan_v3_",
            ),
            "negative_statistics_runtime_projection": negative_statistics,
            "negative_statistics_runtime_projection_root": stable_hash(
                negative_statistics,
                prefix="phase2b_formal_wire_covert_statistics_runtime_v4_",
            ),
            "negative_receipt_issuance_runtime_projection": negative_issuance,
            "negative_receipt_issuance_runtime_projection_root": stable_hash(
                negative_issuance,
                prefix=(
                    "phase2b_formal_wire_covert_receipt_issuance_runtime_v4_"
                ),
            ),
            "negative_leak_feature_id": negative.leak_feature_id,
            "negative_leak_feature_aliases": negative.leak_feature_aliases,
            "negative_leak_observed_omnibus_hex": (
                negative.leak_observed_omnibus_hex
            ),
            "negative_leak_feature_max_t_p": negative.leak_feature_max_t_p,
            "negative_leak_target_holm_critical_value": (
                negative.leak_target_holm_critical_value
            ),
            "source_pair_delta_root": negative.source_pair_delta_root,
            "operation_counts": positive.operation_counts,
            "label_factorial_proof_root": positive.label_factorial_proof_root,
            "positive_feature_manifest_root": audit_receipt.feature_manifest.manifest_root,
            "positive_concrete_features": (
                audit_receipt.feature_manifest.deduplicated_feature_count
            ),
            "positive_raw_semantic_features": (
                audit_receipt.feature_manifest.raw_semantic_feature_count
            ),
            "positive_unique_statistical_partitions": (
                audit_receipt.resource_plan.unique_statistical_partitions
            ),
            "positive_hypotheses": audit_receipt.resource_plan.hypotheses,
            "namespace_uuid_leaf_paths": positive.namespace_uuid_leaf_paths,
            "channel_leaf_paths": positive.channel_leaf_paths,
            "exact_owned_channel_only_leaf_paths": positive.exact_owned_channel_only_leaf_paths,
            "channel_only_owner_union_root": positive.channel_only_owner_union_root,
            "full_raw_byte_categorical_hypotheses": positive.full_raw_byte_categorical_hypotheses,
            "source_commit_sha": None,
            "source_archive_root": None,
            "runtime_image_digest": None,
            "runtime_environment_manifest_root": None,
            "dependency_lock_live_verified": False,
            "synthetic_fixture_archive_verified": False,
            "r03_formal_trusted_wire_implementation_qualified": False,
            "r04_formal_covert_auditor_implementation_qualified": False,
            "r05_synthetic_dev_wire_covert_qualification_verified": False,
            "actual_corpus_audit": False,
            "actual_execution_claimed": False,
        }
        released_negative = auditor.release_formal_covert_audit_receipt_v1(
            negative_receipt
        )
        released_positive = auditor.release_formal_covert_audit_receipt_v1(
            audit_receipt
        )
        if (
            released_negative != negative_receipt_id
            or released_positive != positive_receipt_id
        ):
            raise ValueError("candidate receipt release returned another content ID")
        for name, item in assignments.items():
            object.__setattr__(value, name, item)
        object.__setattr__(
            value,
            "qualification_receipt_id",
            QUALIFICATION_RECEIPT_PREFIX
            + hashlib.sha256(
                QUALIFICATION_RECEIPT_DOMAIN
                + canonicalize_rfc8785_v1(value._receipt_projection())
            ).hexdigest(),
        )
        value._validate()
        return value

    def _receipt_projection(self) -> dict[str, object]:
        return {
            "claim_level": self.claim_level,
            "claims": {
                "actual_corpus_audit": self.actual_corpus_audit,
                "actual_execution_claimed": self.actual_execution_claimed,
                "dependency_lock_live_verified": self.dependency_lock_live_verified,
                "r03_formal_trusted_wire_implementation_qualified": (
                    self.r03_formal_trusted_wire_implementation_qualified
                ),
                "r04_formal_covert_auditor_implementation_qualified": (
                    self.r04_formal_covert_auditor_implementation_qualified
                ),
                "r05_synthetic_dev_wire_covert_qualification_verified": (
                    self.r05_synthetic_dev_wire_covert_qualification_verified
                ),
                "synthetic_fixture_archive_verified": (
                    self.synthetic_fixture_archive_verified
                ),
            },
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "disposition": self.disposition.value,
            "negative": {
                "audit_receipt_id": self.negative_audit_receipt_id,
                "diagnostics": _diagnostic_projection(self.negative_diagnostics),
                "resource_plan_projection": (
                    self.negative_resource_plan_projection
                ),
                "resource_plan_projection_root": (
                    self.negative_resource_plan_projection_root
                ),
                "statistics_runtime_projection": (
                    self.negative_statistics_runtime_projection
                ),
                "statistics_runtime_projection_root": (
                    self.negative_statistics_runtime_projection_root
                ),
                "receipt_issuance_runtime_projection": (
                    self.negative_receipt_issuance_runtime_projection
                ),
                "receipt_issuance_runtime_projection_root": (
                    self.negative_receipt_issuance_runtime_projection_root
                ),
                "disposition": self.negative_disposition,
                "expected_fail_observed": self.negative_expected_fail_observed,
                "leak_feature_id": self.negative_leak_feature_id,
                "leak_feature_aliases": list(self.negative_leak_feature_aliases),
                "leak_feature_max_t_p": self.negative_leak_feature_max_t_p,
                "leak_observed_omnibus_hex": (
                    self.negative_leak_observed_omnibus_hex
                ),
                "leak_target_holm_critical_value": (
                    self.negative_leak_target_holm_critical_value
                ),
                "source_pair_delta_root": self.source_pair_delta_root,
            },
            "operation_counts": dict(self.operation_counts),
            "label_factorial_proof_root": self.label_factorial_proof_root,
            "positive": {
                "audit_receipt_id": self.positive_audit_receipt_id,
                "batch_id": self.positive_batch_id,
                "diagnostics": _diagnostic_projection(self.positive_diagnostics),
                "resource_plan_projection": (
                    self.positive_resource_plan_projection
                ),
                "resource_plan_projection_root": (
                    self.positive_resource_plan_projection_root
                ),
                "statistics_runtime_projection": (
                    self.positive_statistics_runtime_projection
                ),
                "statistics_runtime_projection_root": (
                    self.positive_statistics_runtime_projection_root
                ),
                "receipt_issuance_runtime_projection": (
                    self.positive_receipt_issuance_runtime_projection
                ),
                "receipt_issuance_runtime_projection_root": (
                    self.positive_receipt_issuance_runtime_projection_root
                ),
                "transform_catalog_root": self.positive_transform_catalog_root,
                "feature_manifest_root": self.positive_feature_manifest_root,
                "concrete_features": self.positive_concrete_features,
                "raw_semantic_features": self.positive_raw_semantic_features,
                "unique_statistical_partitions": (
                    self.positive_unique_statistical_partitions
                ),
                "hypotheses": self.positive_hypotheses,
            },
            "profile_id": self.profile_id,
            "projection_only_boundary": dict(_TASK4_PROJECTION_ONLY_BOUNDARY_V2),
            "rich_counts": {
                "channel_leaf_paths": self.channel_leaf_paths,
                "exact_owned_channel_only_leaf_paths": self.exact_owned_channel_only_leaf_paths,
                "channel_only_owner_union_root": self.channel_only_owner_union_root,
                "full_raw_byte_categorical_hypotheses": self.full_raw_byte_categorical_hypotheses,
                "namespace_uuid_leaf_paths": self.namespace_uuid_leaf_paths,
            },
            "runtime_bindings": {
                "dependency_lock_live_verified": self.dependency_lock_live_verified,
                "runtime_image_digest": self.runtime_image_digest,
                "runtime_environment_manifest_root": (
                    self.runtime_environment_manifest_root
                ),
                "source_archive_root": self.source_archive_root,
                "source_commit_sha": self.source_commit_sha,
                "synthetic_fixture_archive_verified": self.synthetic_fixture_archive_verified,
            },
            "synthetic_fixture_root": self.synthetic_fixture_root,
        }

    def _validate(self) -> None:
        if type(self) is not Phase2BFormalWireCovertCandidateReceiptV2:
            raise TypeError("candidate receipt exact type drift")
        if (
            self.disposition
            is not QualificationDispositionV2.CANDIDATE_IMPLEMENTATION_EXERCISED_NOT_R03_R04_R05
            or self.profile_id != PROFILE_ID
            or self.claim_level != PROFILE_CLAIM_LEVEL
            or self.dependency_lock_sha256 != DEPENDENCY_LOCK_SHA256
        ):
            raise ValueError("candidate positive integrated evidence drift")
        _require_auditor_receipt_id_v2(
            self.positive_audit_receipt_id,
            "candidate positive audit receipt ID",
        )
        _require_auditor_receipt_id_v2(
            self.negative_audit_receipt_id,
            "candidate negative audit receipt ID",
        )
        if self.positive_audit_receipt_id == self.negative_audit_receipt_id:
            raise ValueError("candidate audit receipt IDs are not branch-distinct")
        self.positive_diagnostics.__post_init__()
        self.negative_diagnostics.__post_init__()
        for name, diagnostics, expected_mode in (
            ("positive", self.positive_diagnostics, "KNOWN_NULL"),
            ("negative", self.negative_diagnostics, "KNOWN_BINDING_LEAK"),
        ):
            resource_plan = getattr(self, f"{name}_resource_plan_projection")
            resource_plan_root = getattr(
                self, f"{name}_resource_plan_projection_root"
            )
            statistics = getattr(self, f"{name}_statistics_runtime_projection")
            statistics_root = getattr(
                self, f"{name}_statistics_runtime_projection_root"
            )
            issuance = getattr(
                self, f"{name}_receipt_issuance_runtime_projection"
            )
            issuance_root = getattr(
                self, f"{name}_receipt_issuance_runtime_projection_root"
            )
            _validate_complete_resource_plan_projection_v2(
                resource_plan,
                context=f"candidate {name}",
            )
            _validate_complete_statistics_runtime_projection_v2(
                statistics,
                resource_plan,
                context=f"candidate {name}",
            )
            _validate_complete_receipt_issuance_runtime_projection_v2(
                issuance,
                statistics,
                context=f"candidate {name}",
            )
            _validate_executed_runtime_bundle_v2(
                resource_plan=resource_plan,
                statistics_runtime=statistics,
                receipt_issuance_runtime=issuance,
                diagnostics=_diagnostic_projection(diagnostics),
                run=None,
                expected_mode=expected_mode,
                visible_audited_rows=EXACT_AUTHORITY_ROWS,
                visible_concrete_features=(
                    _fixture_feature_capacity_projection_row_v2(
                        expected_mode,
                        EXACT_AUTHORITY_ROWS,
                    )["projected_concrete_feature_count"]
                ),
                context=f"candidate {name}",
            )
            if (
                resource_plan_root
                != stable_hash(
                    resource_plan,
                    prefix="phase2b_formal_wire_covert_resource_plan_v3_",
                )
                or statistics_root
                != stable_hash(
                    statistics,
                    prefix="phase2b_formal_wire_covert_statistics_runtime_v4_",
                )
                or issuance_root
                != stable_hash(
                    issuance,
                    prefix=(
                        "phase2b_formal_wire_covert_receipt_issuance_runtime_v4_"
                    ),
                )
                or statistics.get("worker_peak_rss_bytes")
                != [
                    value * 1_024
                    for value in diagnostics.statistics_worker_historical_peak_rss_kib
                ]
                or statistics.get(
                    "parent_historical_peak_rss_bytes_before_receipt_finalization"
                )
                != diagnostics.parent_historical_peak_rss_before_receipt_finalization_kib
                * 1_024
                or statistics.get(
                    "diagnostic_parent_plus_worker_historical_peak_sum_bytes"
                )
                != diagnostics.parent_plus_worker_historical_peak_rss_kib
                * 1_024
                or statistics.get("unified_cgroup_membership")
                != diagnostics.unified_cgroup_membership
                or statistics.get("unified_cgroup_memory_scope_root")
                != diagnostics.unified_cgroup_memory_scope_root
                or statistics.get(
                    "physical_memory_peak_bytes_before_receipt_finalization"
                )
                != diagnostics.physical_memory_peak_before_receipt_finalization_bytes
                or statistics.get("physical_memory_cap_bytes")
                != diagnostics.physical_memory_cap_bytes
                or statistics.get(
                    "statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds"
                )
                != diagnostics.statistics_wall_time_ns
                or statistics.get("host_available_memory_bytes_at_preflight")
                != diagnostics.host_available_memory_bytes_at_preflight
            ):
                raise ValueError(
                    f"candidate {name} statistics/issuance/diagnostics binding drift"
                )
        _validate_candidate_branch_physical_scope_v2(
            self.positive_statistics_runtime_projection,
            self.negative_statistics_runtime_projection,
        )
        if (
            not self.negative_expected_fail_observed
            or self.negative_disposition != "FAIL"
            or not self.negative_audit_receipt_id
            or not self.negative_leak_feature_id
            or not self.negative_leak_feature_aliases
            or not any(
                _is_registered_binding_leak_alias_v2(alias)
                for alias in self.negative_leak_feature_aliases
            )
            or float.fromhex(self.negative_leak_observed_omnibus_hex) <= 1.0
            or not self.negative_leak_feature_max_t_p
            or not self.negative_leak_target_holm_critical_value
            or not self.source_pair_delta_root
            or self.operation_counts
            != tuple(
                (operation, EXACT_ROWS_PER_TRANSFORM_OPERATION)
                for operation in EXACT_TRANSFORM_OPERATIONS
            )
            or not self.label_factorial_proof_root.startswith(
                "phase2b_formal_wire_covert_label_factorial_v2_"
            )
            or type(self.positive_feature_manifest_root) is not str
            or not self.positive_feature_manifest_root
            or type(self.positive_concrete_features) is not int
            or type(self.positive_raw_semantic_features) is not int
            or type(self.positive_unique_statistical_partitions) is not int
            or type(self.positive_hypotheses) is not int
            or self.positive_resource_plan_projection.get("rows")
            != EXACT_AUTHORITY_ROWS
            or self.positive_resource_plan_projection.get("concrete_features")
            != self.positive_concrete_features
            or self.positive_resource_plan_projection.get(
                "unique_statistical_partitions"
            )
            != self.positive_unique_statistical_partitions
            or self.positive_resource_plan_projection.get("hypotheses")
            != self.positive_hypotheses
            or self.exact_owned_channel_only_leaf_paths
            != _FIXTURE_FEATURE_CAPACITY_CONTRACT_V2["modes"]["KNOWN_NULL"][
                "channel_only_owned_union_path_count"
            ]
            or self.channel_only_owner_union_root
            != _FIXTURE_FEATURE_CAPACITY_CONTRACT_V2["modes"]["KNOWN_NULL"][
                "channel_only_owned_union_root"
            ]
            or min(
                self.positive_concrete_features,
                self.positive_raw_semantic_features,
                self.positive_unique_statistical_partitions,
                self.positive_hypotheses,
                self.full_raw_byte_categorical_hypotheses,
                self.namespace_uuid_leaf_paths,
                self.channel_leaf_paths,
                self.exact_owned_channel_only_leaf_paths,
            )
            <= 0
        ):
            raise ValueError("candidate E1 KAT or fresh rich-shape evidence drift")
        if any(
            (
                self.dependency_lock_live_verified,
                self.synthetic_fixture_archive_verified,
                self.r03_formal_trusted_wire_implementation_qualified,
                self.r04_formal_covert_auditor_implementation_qualified,
                self.r05_synthetic_dev_wire_covert_qualification_verified,
                self.actual_corpus_audit,
                self.actual_execution_claimed,
            )
        ) or any(
            item is not None
            for item in (
                self.source_commit_sha,
                self.source_archive_root,
                self.runtime_image_digest,
                self.runtime_environment_manifest_root,
            )
        ):
            raise ValueError("candidate receipt crosses its runtime/readiness boundary")
        expected_id = QUALIFICATION_RECEIPT_PREFIX + hashlib.sha256(
            QUALIFICATION_RECEIPT_DOMAIN
            + canonicalize_rfc8785_v1(self._receipt_projection())
        ).hexdigest()
        if self.qualification_receipt_id != expected_id:
            raise ValueError("candidate qualification receipt ID drift")


def _diagnostic_projection(value: AuditRunDiagnosticsV2) -> dict[str, object]:
    value.__post_init__()
    return {
        "parent_plus_worker_historical_peak_rss_kib": (
            value.parent_plus_worker_historical_peak_rss_kib
        ),
        "host_available_memory_bytes_at_preflight": (
            value.host_available_memory_bytes_at_preflight
        ),
        "maximum_rows_per_invocation": value.maximum_rows_per_invocation,
        "minimum_rows_per_invocation": value.minimum_rows_per_invocation,
        "mode": value.mode,
        "recognizer_adapter_invocations": value.recognizer_adapter_invocations,
        "parent_historical_peak_rss_before_receipt_finalization_kib": (
            value.parent_historical_peak_rss_before_receipt_finalization_kib
        ),
        "physical_memory_cap_bytes": value.physical_memory_cap_bytes,
        "physical_memory_current_before_receipt_finalization_bytes": (
            value.physical_memory_current_before_receipt_finalization_bytes
        ),
        "physical_memory_max_bytes": value.physical_memory_max_bytes,
        "physical_memory_peak_before_receipt_finalization_bytes": (
            value.physical_memory_peak_before_receipt_finalization_bytes
        ),
        "statistics_wall_time_ns": value.statistics_wall_time_ns,
        "statistics_worker_historical_peak_rss_kib": list(
            value.statistics_worker_historical_peak_rss_kib
        ),
        "unified_cgroup_membership": value.unified_cgroup_membership,
        "unified_cgroup_leaf_device": value.unified_cgroup_leaf_device,
        "unified_cgroup_leaf_inode": value.unified_cgroup_leaf_inode,
        "unified_cgroup_memory_scope_root": value.unified_cgroup_memory_scope_root,
        "wall_time_ns": value.wall_time_ns,
    }


@dataclass(frozen=True, slots=True, init=False)
class _VerifiedOfflineRuntimeEvidenceTokenV2:
    """Reserved capability type; no issuer exists in this candidate slice."""

    source_commit_sha: str
    source_archive_root: str
    dependency_lock_sha256: str
    runtime_image_digest: None
    runtime_environment_manifest_root: str
    synthetic_fixture_root: str
    qualification_receipt_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "runtime evidence tokens are issued only by the later live offline CLI"
        )


def _combined_fixture_root_v2(
    positive: _IsolatedAuditEvidenceV2,
    negative: _IsolatedAuditEvidenceV2,
) -> str:
    return _combined_fixture_root_from_run_roots_v2(
        positive.run_fixture_root, negative.run_fixture_root
    )


def _combined_fixture_root_from_run_roots_v2(
    positive_run_fixture_root: str,
    negative_run_fixture_root: str,
) -> str:
    return "phase2b_formal_wire_covert_synthetic_fixture_v2_" + hashlib.sha256(
        SYNTHETIC_FIXTURE_DOMAIN
        + canonicalize_rfc8785_v1(
            {
                "negative_run_fixture_root": negative_run_fixture_root,
                "positive_run_fixture_root": positive_run_fixture_root,
                "schema_version": PROFILE_SCHEMA_VERSION,
            }
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class Phase2BFormalWireCovertAttemptManifestV2:
    """Pre-registered outer-attempt data; it grants no execution claim."""

    campaign_id: str
    attempt_id: str
    profile_id: str
    source_commit_sha: str
    source_archive_root: str
    source_archive_size_bytes: int
    dependency_lock_sha256: str
    qualification_source_sha256: str
    qualification_config_sha256: str
    qualification_lock_sha256: str
    qualification_test_sha256: str
    qualification_documentation_sha256: str
    runtime_environment_manifest_root: str
    runtime_environment_manifest_size_bytes: int
    runtime_image_digest: None
    rfc8785_prerequisite_receipt_root: str
    rfc8785_prerequisite_receipt_sha256: str
    rfc8785_prerequisite_receipt_size_bytes: int
    mode_order: tuple[str, str]
    hard_total_timeout_seconds: int
    requested_cgroup_memory_max_bytes: int
    requested_cgroup_pids_max: int
    attempt_manifest_root: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not Phase2BFormalWireCovertAttemptManifestV2:
            raise TypeError("attempt manifest exact type drift")
        for name in ("campaign_id", "attempt_id"):
            value = getattr(self, name)
            if type(value) is not str or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value
            ) is None:
                raise ValueError(f"attempt {name} is outside the closed identifier grammar")
        if re.fullmatch(
            r"phase2b_formal_wire_covert_qualification_profile_v2_[0-9a-f]{64}",
            self.profile_id,
        ) is None:
            raise ValueError("attempt profile ID is outside the closed grammar")
        if re.fullmatch(r"[0-9a-f]{40}", self.source_commit_sha) is None:
            raise ValueError("attempt source commit must be an exact Git SHA-1")
        for name in (
            "source_archive_root",
            "runtime_environment_manifest_root",
            "rfc8785_prerequisite_receipt_root",
        ):
            if re.fullmatch(r"sha256:[0-9a-f]{64}", getattr(self, name)) is None:
                raise ValueError(f"attempt {name} must be an exact SHA-256 root")
        for name in (
            "dependency_lock_sha256",
            "qualification_source_sha256",
            "qualification_config_sha256",
            "qualification_lock_sha256",
            "qualification_test_sha256",
            "qualification_documentation_sha256",
            "rfc8785_prerequisite_receipt_sha256",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)) is None:
                raise ValueError(f"attempt {name} must be an exact raw SHA-256")
        if self.runtime_image_digest is not None:
            raise ValueError("host transient-service qualification has no image claim")
        size_caps = {
            "source_archive_size_bytes": 512 * 1_024 * 1_024,
            "runtime_environment_manifest_size_bytes": 1_048_576,
        }
        for name, maximum in size_caps.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"attempt {name} is outside bounded CAS size")
        if (
            self.rfc8785_prerequisite_receipt_root
            != "sha256:" + self.rfc8785_prerequisite_receipt_sha256
            or type(self.rfc8785_prerequisite_receipt_size_bytes) is not int
            or not 1 <= self.rfc8785_prerequisite_receipt_size_bytes <= 1_048_576
        ):
            raise ValueError("attempt RFC-8785 receipt byte binding is incomplete")
        if self.mode_order != LIVE_MODE_ORDER:
            raise ValueError("attempt mode order drift")
        if self.hard_total_timeout_seconds != LIVE_HARD_TOTAL_TIMEOUT_SECONDS:
            raise ValueError("attempt hard total timeout drift")
        if (
            type(self.requested_cgroup_memory_max_bytes) is not int
            or self.requested_cgroup_memory_max_bytes
            != OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2
            or type(self.requested_cgroup_pids_max) is not int
            or not 2 <= self.requested_cgroup_pids_max <= 4_096
        ):
            raise ValueError("attempt cgroup limits are outside the frozen live bounds")
        projection = _attempt_manifest_projection_v2(self)
        object.__setattr__(
            self,
            "attempt_manifest_root",
            stable_hash(
                projection,
                prefix="phase2b_formal_wire_covert_attempt_manifest_v3_",
            ),
        )


def _attempt_manifest_projection_v2(
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
) -> dict[str, object]:
    return {
        "campaign_id": attempt.campaign_id,
        "attempt_id": attempt.attempt_id,
        "profile_id": attempt.profile_id,
        "source_commit_sha": attempt.source_commit_sha,
        "source_archive_root": attempt.source_archive_root,
        "source_archive_size_bytes": attempt.source_archive_size_bytes,
        "dependency_lock_sha256": attempt.dependency_lock_sha256,
        "qualification_sha256": {
            "config": attempt.qualification_config_sha256,
            "documentation": attempt.qualification_documentation_sha256,
            "lock": attempt.qualification_lock_sha256,
            "source": attempt.qualification_source_sha256,
            "test": attempt.qualification_test_sha256,
        },
        "runtime_environment_manifest_root": (
            attempt.runtime_environment_manifest_root
        ),
        "runtime_environment_manifest_size_bytes": (
            attempt.runtime_environment_manifest_size_bytes
        ),
        "runtime_image_digest": None,
        "rfc8785_prerequisite_receipt_root": (
            attempt.rfc8785_prerequisite_receipt_root
        ),
        "rfc8785_prerequisite_receipt_sha256": (
            attempt.rfc8785_prerequisite_receipt_sha256
        ),
        "rfc8785_prerequisite_receipt_size_bytes": (
            attempt.rfc8785_prerequisite_receipt_size_bytes
        ),
        "mode_order": list(attempt.mode_order),
        "hard_total_timeout_seconds": attempt.hard_total_timeout_seconds,
        "requested_cgroup_memory_max_bytes": (
            attempt.requested_cgroup_memory_max_bytes
        ),
        "requested_cgroup_pids_max": attempt.requested_cgroup_pids_max,
    }


def phase2b_formal_wire_covert_attempt_manifest_bytes_v2(
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
) -> bytes:
    if type(attempt) is not Phase2BFormalWireCovertAttemptManifestV2:
        raise TypeError("attempt serializer needs the exact manifest type")
    attempt.__post_init__()
    return canonicalize_rfc8785_v1(
        {
            "attempt_manifest": _attempt_manifest_projection_v2(attempt),
            "attempt_manifest_root": attempt.attempt_manifest_root,
        }
    ) + b"\n"


def load_phase2b_formal_wire_covert_attempt_manifest_v2(
    payload: bytes,
    *,
    expected_profile_id: str = PROFILE_ID,
) -> Phase2BFormalWireCovertAttemptManifestV2:
    decoded = _decode_canonical_transport_object_v2(
        payload, "attempt manifest", maximum_bytes=256 * 1_024
    )
    if set(decoded) != {"attempt_manifest", "attempt_manifest_root"}:
        raise ValueError("attempt manifest transport root is not closed")
    body = decoded["attempt_manifest"]
    if type(body) is not dict:
        raise ValueError("attempt manifest body is not an exact object")
    qualification_sha256 = body.get("qualification_sha256")
    if type(qualification_sha256) is not dict:
        raise ValueError("attempt qualification hash manifest is absent")
    expected_body_keys = {
        "campaign_id",
        "attempt_id",
        "profile_id",
        "source_commit_sha",
        "source_archive_root",
        "source_archive_size_bytes",
        "dependency_lock_sha256",
        "qualification_sha256",
        "runtime_environment_manifest_root",
        "runtime_environment_manifest_size_bytes",
        "runtime_image_digest",
        "rfc8785_prerequisite_receipt_root",
        "rfc8785_prerequisite_receipt_sha256",
        "rfc8785_prerequisite_receipt_size_bytes",
        "mode_order",
        "hard_total_timeout_seconds",
        "requested_cgroup_memory_max_bytes",
        "requested_cgroup_pids_max",
    }
    if set(body) != expected_body_keys or set(qualification_sha256) != {
        "config", "documentation", "lock", "source", "test"
    }:
        raise ValueError("attempt manifest fields are not exact")
    mode_order = body["mode_order"]
    if type(mode_order) is not list:
        raise ValueError("attempt mode order must be an exact JSON array")
    attempt = Phase2BFormalWireCovertAttemptManifestV2(
        body["campaign_id"],
        body["attempt_id"],
        body["profile_id"],
        body["source_commit_sha"],
        body["source_archive_root"],
        body["source_archive_size_bytes"],
        body["dependency_lock_sha256"],
        qualification_sha256["source"],
        qualification_sha256["config"],
        qualification_sha256["lock"],
        qualification_sha256["test"],
        qualification_sha256["documentation"],
        body["runtime_environment_manifest_root"],
        body["runtime_environment_manifest_size_bytes"],
        body["runtime_image_digest"],
        body["rfc8785_prerequisite_receipt_root"],
        body["rfc8785_prerequisite_receipt_sha256"],
        body["rfc8785_prerequisite_receipt_size_bytes"],
        tuple(mode_order),
        body["hard_total_timeout_seconds"],
        body["requested_cgroup_memory_max_bytes"],
        body["requested_cgroup_pids_max"],
    )
    if attempt.profile_id != expected_profile_id:
        raise ValueError("attempt manifest does not bind the expected profile revision")
    if decoded["attempt_manifest_root"] != attempt.attempt_manifest_root:
        raise ValueError("attempt manifest content root mismatch")
    if phase2b_formal_wire_covert_attempt_manifest_bytes_v2(attempt) != payload:
        raise ValueError("attempt manifest canonical replay drift")
    return attempt


def _read_bounded_cgroup_ascii_v2(
    path: Path,
    name: str,
    *,
    maximum_bytes: int = 64 * 1_024,
) -> str:
    try:
        with path.open("rb") as handle:
            payload = handle.read(maximum_bytes + 1)
    except OSError as exc:
        raise ValueError(f"{name} cgroup file is unreadable") from exc
    if len(payload) > maximum_bytes:
        raise ValueError(f"{name} cgroup file is outside bounds")
    try:
        return payload.decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise ValueError(f"{name} cgroup file is not exact ASCII") from exc


def _parse_canonical_cgroup_decimal_v2(
    payload: str,
    name: str,
    *,
    positive: bool = False,
) -> int:
    if (
        type(payload) is not str
        or re.fullmatch(r"(?:0|[1-9][0-9]*)\n?", payload) is None
    ):
        raise ValueError(f"{name} is not one canonical cgroup-v2 decimal")
    parsed = int(payload.removesuffix("\n"))
    if (
        parsed > auditor.MAXIMUM_LINUX_RESOURCE_COUNTER_BYTES
        or positive
        and parsed <= 0
    ):
        raise ValueError(f"{name} is outside its finite cgroup-v2 bounds")
    return parsed


def _parse_cgroup_counter_file_v2(payload: str, name: str) -> dict[str, int]:
    if (
        type(payload) is not str
        or not payload
        or len(payload.encode("utf-8")) > 64 * 1_024
        or payload.endswith("\n\n")
    ):
        raise ValueError(f"{name} cgroup counter file is outside bounds")
    result: dict[str, int] = {}
    for line in payload.splitlines():
        match = re.fullmatch(
            r"([a-z0-9_.-]+) ((?:0|[1-9][0-9]*))",
            line,
        )
        if match is None:
            raise ValueError(f"{name} cgroup counter row is malformed")
        key, decimal = match.groups()
        if key in result:
            raise ValueError(f"{name} cgroup counter key is duplicated")
        result[key] = _parse_canonical_cgroup_decimal_v2(
            decimal,
            f"{name}.{key}",
        )
    return result


def _parse_finite_cgroup_limit_v2(payload: str, name: str) -> int:
    if payload in {"max", "max\n"}:
        raise ValueError(f"{name} must be a finite cgroup-v2 decimal limit")
    return _parse_canonical_cgroup_decimal_v2(payload, name, positive=True)


def _parse_cgroup_procs_v2(payload: str) -> list[int]:
    if (
        type(payload) is not str
        or not payload
        or payload.endswith("\n\n")
        or len(payload.encode("ascii", errors="ignore")) > 64 * 1_024
    ):
        raise ValueError("cgroup.procs is outside bounds")
    rows = payload.splitlines()
    pids = [
        _parse_canonical_cgroup_decimal_v2(row, "cgroup.procs PID", positive=True)
        for row in rows
    ]
    if len(pids) != len(set(pids)):
        raise ValueError("cgroup.procs contains a duplicate PID")
    return pids


def inspect_current_live_cgroup_v2(
    *,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict[str, object]:
    """Read the current unified cgroup; this never creates or changes a cgroup."""

    membership_path = proc_root / "self/cgroup"
    membership_text = _read_bounded_cgroup_ascii_v2(
        membership_path,
        "live qualification unified membership",
    )
    membership_rows = membership_text.splitlines()
    if len(membership_rows) != 1 or not membership_rows[0].startswith("0::"):
        raise ValueError("live qualification needs one exact unified cgroup membership")
    membership = _require_safe_cgroup_path_v2(
        membership_rows[0][3:], "live qualification unified cgroup membership"
    )
    relative = membership.lstrip("/")
    directory = cgroup_root / relative
    try:
        before_stat = directory.stat(follow_symlinks=False)
        resolved_directory = directory.resolve(strict=True)
    except OSError as exc:
        raise ValueError("live qualification cgroup leaf is unreadable") from exc
    if not stat.S_ISDIR(before_stat.st_mode) or resolved_directory != directory:
        raise ValueError("live qualification cgroup leaf is not one real directory")
    memory_max = _parse_finite_cgroup_limit_v2(
        _read_bounded_cgroup_ascii_v2(
            directory / "memory.max", "memory.max", maximum_bytes=4_096
        ),
        "memory.max",
    )
    pids_max = _parse_finite_cgroup_limit_v2(
        _read_bounded_cgroup_ascii_v2(
            directory / "pids.max", "pids.max", maximum_bytes=4_096
        ),
        "pids.max",
    )
    decimal_files: dict[str, int] = {}
    for name in ("memory.current", "memory.peak", "pids.current"):
        decimal_files[name] = _parse_canonical_cgroup_decimal_v2(
            _read_bounded_cgroup_ascii_v2(
                directory / name, name, maximum_bytes=4_096
            ),
            name,
        )
    memory_events = _parse_cgroup_counter_file_v2(
        _read_bounded_cgroup_ascii_v2(
            directory / "memory.events", "memory.events"
        ),
        "memory.events",
    )
    pids_events = _parse_cgroup_counter_file_v2(
        _read_bounded_cgroup_ascii_v2(
            directory / "pids.events", "pids.events"
        ),
        "pids.events",
    )
    cgroup_procs = _parse_cgroup_procs_v2(
        _read_bounded_cgroup_ascii_v2(
            directory / "cgroup.procs", "cgroup.procs"
        )
    )
    cgroup_stat = _parse_cgroup_counter_file_v2(
        _read_bounded_cgroup_ascii_v2(
            directory / "cgroup.stat", "cgroup.stat"
        ),
        "cgroup.stat",
    )
    if (
        not {"max", "oom", "oom_kill"}.issubset(memory_events)
        or "max" not in pids_events
        or cgroup_procs != [os.getpid()]
        or not {"nr_descendants", "nr_dying_descendants"}.issubset(cgroup_stat)
        or cgroup_stat["nr_descendants"] != 0
        or cgroup_stat["nr_dying_descendants"] != 0
        or decimal_files["memory.peak"] < decimal_files["memory.current"]
    ):
        raise ValueError("live cgroup structural snapshot drifted")
    cpu_max_path = directory / "cpu.max"
    cpu_max = (
        _read_bounded_cgroup_ascii_v2(
            cpu_max_path, "cpu.max", maximum_bytes=4_096
        ).removesuffix("\n")
        if cpu_max_path.is_file()
        else None
    )
    affinity = tuple(sorted(os.sched_getaffinity(0)))
    if not affinity:
        raise ValueError("live qualification CPU affinity is empty")
    try:
        after_stat = directory.stat(follow_symlinks=False)
        membership_replay = _read_bounded_cgroup_ascii_v2(
            membership_path,
            "live qualification unified membership replay",
        )
    except OSError as exc:
        raise ValueError("live qualification cgroup identity replay failed") from exc
    if (
        (before_stat.st_dev, before_stat.st_ino)
        != (after_stat.st_dev, after_stat.st_ino)
        or membership_replay != membership_text
    ):
        raise ValueError("live qualification cgroup identity drifted during snapshot")
    memory_scope_root = stable_hash(
        {
            "leaf_device": before_stat.st_dev,
            "leaf_inode": before_stat.st_ino,
            "unified_membership": membership,
        },
        prefix="phase2b_formal_covert_unified_cgroup_memory_scope_v1_",
    )
    return {
        "cgroup_path": membership,
        "cgroup_procs": cgroup_procs,
        "cgroup_stat": cgroup_stat,
        "unified_cgroup_memory_scope_root": memory_scope_root,
        "unified_cgroup_leaf_device": before_stat.st_dev,
        "unified_cgroup_leaf_inode": before_stat.st_ino,
        "memory_current": decimal_files["memory.current"],
        "memory_events": memory_events,
        "memory_max": memory_max,
        "memory_peak": decimal_files["memory.peak"],
        "pids_current": decimal_files["pids.current"],
        "pids_events": pids_events,
        "pids_max": pids_max,
        "cpu_max": cpu_max,
        "cpu_affinity": list(affinity),
        "cpu_capacity": len(affinity),
    }


def _verify_source_only_runtime_bootstrap_v2() -> dict[str, object]:
    flags = sys.flags
    expected = {
        "dont_write_bytecode": 1,
        "ignore_environment": 1,
        "isolated": 1,
        "no_site": 1,
    }
    actual = {
        "dont_write_bytecode": flags.dont_write_bytecode,
        "ignore_environment": flags.ignore_environment,
        "isolated": flags.isolated,
        "no_site": flags.no_site,
    }
    if not _exact_json_equal(actual, expected):
        raise ValueError("live inner interpreter is not exact -I -S -B isolated mode")
    bootstrap = _EXECUTED_MAIN_SOURCE_BOOTSTRAP_V2
    manifest = globals().get("_HEGEL_SOURCE_BOOTSTRAP_MANIFEST_V2")
    if (
        _SOURCE_ONLY_IMPORTER_ACTIVE_V2 is not True
        or _SOURCE_EXECUTION_BOOTSTRAP_ACTIVE_V2 is not True
        or type(bootstrap) is not dict
        or bootstrap.get("bootstrap_kind") != "EXTERNAL_PYTHON_C_OPEN_ONCE_V2"
        or type(manifest) is not dict
        or not _exact_json_equal(
            manifest,
            _source_execution_bootstrap_manifest_v2(bootstrap.get("sha256")),
        )
        or type(_PREOPENED_SOURCE_SNAPSHOTS_V2) is not dict
    ):
        raise ValueError("live inner lacks the external open-once source bootstrap")
    package_directory = Path(__file__).resolve(strict=True).parent
    for root, directory_names, file_names in os.walk(
        package_directory, topdown=True, followlinks=False
    ):
        root_path = Path(root)
        for name in tuple(directory_names):
            candidate = root_path / name
            if candidate.is_symlink():
                raise ValueError("live source package contains a directory symlink")
        for name in file_names:
            candidate = root_path / name
            if candidate.is_symlink():
                raise ValueError("live source package contains a file symlink")
            if candidate.suffix in {".pyc", ".pyo"}:
                raise ValueError("live source package contains forbidden bytecode")
    return {
        "bytecode_candidates_found": [],
        "dont_write_bytecode": 1,
        "ignore_environment": 1,
        "isolated": 1,
        "no_site": 1,
        "package_directory": str(package_directory),
        "executed_main_source_sha256": bootstrap["sha256"],
        "preopened_source_snapshot_root": stable_hash(
            _PREOPENED_SOURCE_SNAPSHOTS_V2,
            prefix="phase2b_formal_wire_covert_preopened_sources_v2_",
        ),
        "source_execution_bootstrap_active": True,
        "source_execution_bootstrap_manifest_root": stable_hash(
            manifest,
            prefix="phase2b_formal_wire_covert_source_bootstrap_manifest_v2_",
        ),
        "source_only_importer_active": True,
    }


def prepare_live_inner_process_v2() -> dict[str, object]:
    """Reset selected dispositions, then prove a one-thread unblocked child."""

    interpreter_runtime = _verify_source_only_runtime_bootstrap_v2()
    for number in LIVE_SELECTED_DEFAULT_SIGNALS:
        signal.signal(number, signal.SIG_DFL)
    status = (Path("/proc") / "self/status").read_text(
        encoding="ascii", errors="strict"
    )
    fields: dict[str, str] = {}
    for line in status.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    if fields.get("Threads") != "1" or int(fields.get("SigBlk", "1"), 16) != 0:
        raise ValueError("live inner process is not one-threaded with an empty signal mask")
    if any(
        signal.getsignal(number) is not signal.SIG_DFL
        for number in LIVE_SELECTED_DEFAULT_SIGNALS
    ):
        raise ValueError("live inner selected signal disposition is not default")
    cgroup = inspect_current_live_cgroup_v2()
    if cgroup["pids_current"] != 1:
        raise ValueError("live transient service must start with exactly one task")
    return {
        "cgroup": cgroup,
        "interpreter_runtime": interpreter_runtime,
        "selected_default_signals": [number.name for number in LIVE_SELECTED_DEFAULT_SIGNALS],
        "sigblk_hex": fields["SigBlk"],
        "threads": 1,
    }


def _bounded_cgroup_physical_resource_detail_v2(
    *,
    stage: str,
    cgroup: dict[str, object],
    cap: int,
    reserve: int,
) -> str:
    """Format one exact bounded diagnostic for an explicit physical gate."""

    if (
        type(stage) is not str
        or re.fullmatch(r"[A-Z_]{1,64}", stage) is None
        or type(cap) is not int
        or cap <= 0
        or type(reserve) is not int
        or not 0 <= reserve < cap
    ):
        raise TypeError("cgroup physical resource detail inputs are malformed")
    scope = cgroup.get("unified_cgroup_memory_scope_root")
    if type(scope) is not str or len(scope) > 128:
        raise ValueError("cgroup physical resource scope root is malformed")
    current = _require_exact_int_v2(
        cgroup.get("memory_current"), f"{stage} memory.current"
    )
    peak = _require_exact_int_v2(cgroup.get("memory_peak"), f"{stage} memory.peak")
    memory_max = _require_exact_int_v2(
        cgroup.get("memory_max"), f"{stage} memory.max", minimum=1
    )
    pids_max = _require_exact_int_v2(
        cgroup.get("pids_max"), f"{stage} pids.max", minimum=1
    )
    memory_events = cgroup.get("memory_events")
    pids_events = cgroup.get("pids_events")
    if type(memory_events) is not dict or type(pids_events) is not dict:
        raise ValueError("cgroup physical resource events are absent")
    event_max = _require_exact_int_v2(
        memory_events.get("max"), f"{stage} memory.events.max"
    )
    event_oom = _require_exact_int_v2(
        memory_events.get("oom"), f"{stage} memory.events.oom"
    )
    event_oom_kill = _require_exact_int_v2(
        memory_events.get("oom_kill"), f"{stage} memory.events.oom_kill"
    )
    pids_event_max = _require_exact_int_v2(
        pids_events.get("max"), f"{stage} pids.events.max"
    )
    detail = (
        f"stage={stage};scope={scope};current={current};peak={peak};"
        f"memory.max={memory_max};cap={cap};reserve={reserve};"
        f"overage={max(0, peak - (cap - reserve))};"
        f"memory.events.max={event_max};memory.events.oom={event_oom};"
        f"memory.events.oom_kill={event_oom_kill};pids.max={pids_max};"
        f"pids.events.max={pids_event_max}"
    )
    if "\n" in detail or "\r" in detail or len(detail.encode("ascii")) > 1_024:
        raise RuntimeError("cgroup physical resource diagnostic is not bounded")
    return detail


def _validate_live_runtime_entry_v2(
    runtime: dict[str, object],
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
) -> None:
    cgroup = runtime.get("cgroup")
    expected_interpreter_runtime = _verify_source_only_runtime_bootstrap_v2()
    if (
        type(cgroup) is not dict
        or cgroup.get("memory_max")
        != attempt.requested_cgroup_memory_max_bytes
        or cgroup.get("pids_max") != attempt.requested_cgroup_pids_max
        or cgroup.get("pids_current") != 1
        or attempt.dependency_lock_sha256 != DEPENDENCY_LOCK_SHA256
        or not _exact_json_equal(
            runtime.get("interpreter_runtime"), expected_interpreter_runtime
        )
    ):
        _fail(FAIL_RESOURCE, "actual live entry bindings differ from the attempt manifest")
    current = _require_exact_int_v2(
        cgroup.get("memory_current"), "live entry memory.current"
    )
    peak = _require_exact_int_v2(
        cgroup.get("memory_peak"), "live entry memory.peak"
    )
    detail = _bounded_cgroup_physical_resource_detail_v2(
        stage="RUNTIME_ENTRY_SAMPLE",
        cgroup=cgroup,
        cap=auditor.MAXIMUM_ESTIMATED_RSS_BYTES,
        reserve=0,
    )
    if current > peak or peak > auditor.MAXIMUM_ESTIMATED_RSS_BYTES:
        _fail(
            FAIL_RESOURCE,
            "live inner entry physical resource gate rejected; " + detail,
        )
    try:
        _validate_zero_cgroup_events_v2(
            cgroup.get("memory_events"),
            "live entry memory",
            required_keys=frozenset({"max", "oom", "oom_kill"}),
        )
        _validate_zero_cgroup_events_v2(
            cgroup.get("pids_events"),
            "live entry pids",
            required_keys=frozenset({"max"}),
        )
    except ValueError as exc:
        raise Phase2BFormalWireCovertQualificationErrorV2(
            FAIL_RESOURCE,
            "live inner entry physical resource gate rejected; " + detail,
        ) from exc


def _validate_receipt_physical_scope_against_exit_cgroup_v2(
    runs: object,
    entry_cgroup: object,
    exit_cgroup: object,
    *,
    context: str,
) -> None:
    """Bind sequential Auditor samples to one non-reset inner cgroup leaf."""

    if (
        type(runs) is not list
        or len(runs) != len(LIVE_MODE_ORDER)
        or type(entry_cgroup) is not dict
        or type(exit_cgroup) is not dict
    ):
        raise ValueError(f"{context} receipt/cgroup binding inputs are absent")
    exit_membership = _require_safe_cgroup_path_v2(
        exit_cgroup.get("cgroup_path"), f"{context} exit cgroup"
    )
    exit_peak = _require_exact_int_v2(
        exit_cgroup.get("memory_peak"), f"{context} exit memory.peak"
    )
    exit_current = _require_exact_int_v2(
        exit_cgroup.get("memory_current"), f"{context} exit memory.current"
    )
    exit_max = _require_exact_int_v2(
        exit_cgroup.get("memory_max"),
        f"{context} exit memory.max",
        minimum=1,
    )
    exit_leaf_device = _require_exact_int_v2(
        exit_cgroup.get("unified_cgroup_leaf_device"),
        f"{context} exit cgroup leaf device",
        minimum=1,
    )
    exit_leaf_inode = _require_exact_int_v2(
        exit_cgroup.get("unified_cgroup_leaf_inode"),
        f"{context} exit cgroup leaf inode",
        minimum=1,
    )
    exit_scope_root = exit_cgroup.get("unified_cgroup_memory_scope_root")
    expected_exit_scope_root = stable_hash(
        {
            "leaf_device": exit_leaf_device,
            "leaf_inode": exit_leaf_inode,
            "unified_membership": exit_membership,
        },
        prefix="phase2b_formal_covert_unified_cgroup_memory_scope_v1_",
    )
    if (
        exit_current > exit_peak
        or exit_peak > auditor.MAXIMUM_ESTIMATED_RSS_BYTES
        or exit_scope_root != expected_exit_scope_root
    ):
        raise ValueError(f"{context} exit physical peak exceeds the Auditor cap")
    exit_scope = (
        exit_membership,
        exit_scope_root,
        exit_leaf_device,
        exit_leaf_inode,
        exit_max,
    )
    entry_scope = (
        _require_safe_cgroup_path_v2(
            entry_cgroup.get("cgroup_path"), f"{context} entry cgroup"
        ),
        entry_cgroup.get("unified_cgroup_memory_scope_root"),
        _require_exact_int_v2(
            entry_cgroup.get("unified_cgroup_leaf_device"),
            f"{context} entry cgroup leaf device",
            minimum=1,
        ),
        _require_exact_int_v2(
            entry_cgroup.get("unified_cgroup_leaf_inode"),
            f"{context} entry cgroup leaf inode",
            minimum=1,
        ),
        _require_exact_int_v2(
            entry_cgroup.get("memory_max"),
            f"{context} entry memory.max",
            minimum=1,
        ),
    )
    entry_current = _require_exact_int_v2(
        entry_cgroup.get("memory_current"), f"{context} entry memory.current"
    )
    entry_peak = _require_exact_int_v2(
        entry_cgroup.get("memory_peak"), f"{context} entry memory.peak"
    )
    if (
        entry_scope != exit_scope
        or entry_current > entry_peak
        or entry_peak > exit_peak
    ):
        raise ValueError(f"{context} entry/exit physical scope binding drifted")
    expected_scope: tuple[object, ...] | None = None
    previous_physical_peak = entry_peak
    invariant_fields = {
        item.name for item in dataclass_fields(auditor.InvariantRuntimeEvidenceV1)
    }
    for index, expected_mode in enumerate(LIVE_MODE_ORDER):
        run = runs[index]
        audit_row = run.get("audit") if type(run) is dict else None
        invariant_row = (
            audit_row.get("invariance") if type(audit_row) is dict else None
        )
        issuance_row = (
            audit_row.get("receipt_issuance")
            if type(audit_row) is dict
            else None
        )
        complete = (
            issuance_row.get("complete_runtime_projection")
            if type(issuance_row) is dict
            else None
        )
        invariant_complete = (
            invariant_row.get("complete_runtime_projection")
            if type(invariant_row) is dict
            else None
        )
        if (
            type(run) is not dict
            or run.get("mode") != expected_mode
            or type(invariant_complete) is not dict
            or set(invariant_complete) != invariant_fields
            or type(complete) is not dict
            or set(complete) != set(_RECEIPT_ISSUANCE_RUNTIME_FIELDS_V2)
        ):
            raise ValueError(f"{context} receipt physical scope is not closed")
        scope = (
            complete.get("unified_cgroup_membership"),
            complete.get("unified_cgroup_memory_scope_root"),
            complete.get("unified_cgroup_leaf_device"),
            complete.get("unified_cgroup_leaf_inode"),
            complete.get("physical_memory_max_bytes"),
        )
        if expected_scope is None:
            expected_scope = scope
        invariant_scope = (
            invariant_complete.get("unified_cgroup_membership"),
            invariant_complete.get("unified_cgroup_memory_scope_root"),
            invariant_complete.get("unified_cgroup_leaf_device"),
            invariant_complete.get("unified_cgroup_leaf_inode"),
            invariant_complete.get("physical_memory_max_bytes"),
        )
        invariant_current = _require_exact_int_v2(
            invariant_complete.get("physical_memory_current_bytes_at_completion"),
            f"{context} {expected_mode} invariant memory.current",
        )
        invariant_peak = _require_exact_int_v2(
            invariant_complete.get("physical_memory_peak_bytes_at_completion"),
            f"{context} {expected_mode} invariant memory.peak",
        )
        receipt_current = _require_exact_int_v2(
            complete.get(
                "physical_memory_current_bytes_before_receipt_finalization"
            ),
            f"{context} {expected_mode} receipt memory.current",
        )
        receipt_peak = _require_exact_int_v2(
            complete.get(
                "physical_memory_peak_bytes_before_receipt_finalization"
            ),
            f"{context} {expected_mode} receipt memory.peak",
        )
        if (
            scope != expected_scope
            or scope != exit_scope
            or invariant_scope != exit_scope
            or invariant_complete.get("physical_memory_cap_bytes")
            != auditor.MAXIMUM_ESTIMATED_RSS_BYTES
            or complete.get("physical_memory_cap_bytes")
            != auditor.MAXIMUM_ESTIMATED_RSS_BYTES
            or invariant_current > invariant_peak
            or invariant_peak < previous_physical_peak
            or receipt_current > receipt_peak
            or receipt_peak < invariant_peak
            or receipt_peak > exit_peak
        ):
            raise ValueError(f"{context} receipt/exit physical scope binding drifted")
        previous_physical_peak = receipt_peak


def _capture_live_runtime_exit_v2(
    runtime_entry: dict[str, object],
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
    runs: list[dict[str, object]],
) -> dict[str, object]:
    entry_cgroup = runtime_entry.get("cgroup")
    if type(entry_cgroup) is not dict:
        raise ValueError("live runtime entry cgroup evidence is absent")
    status = Path("/proc/self/status").read_text(
        encoding="ascii", errors="strict"
    )
    fields: dict[str, str] = {}
    for line in status.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    if fields.get("Threads") != "1" or int(fields.get("SigBlk", "1"), 16) != 0:
        raise ValueError("live inner exit is not one-threaded with an empty signal mask")
    if any(
        signal.getsignal(number) is not signal.SIG_DFL
        for number in LIVE_SELECTED_DEFAULT_SIGNALS
    ):
        raise ValueError("live inner selected signal disposition changed before exit")
    cgroup = inspect_current_live_cgroup_v2()
    if (
        cgroup["cgroup_path"] != entry_cgroup.get("cgroup_path")
        or cgroup["unified_cgroup_memory_scope_root"]
        != entry_cgroup.get("unified_cgroup_memory_scope_root")
        or cgroup["unified_cgroup_leaf_device"]
        != entry_cgroup.get("unified_cgroup_leaf_device")
        or cgroup["unified_cgroup_leaf_inode"]
        != entry_cgroup.get("unified_cgroup_leaf_inode")
    ):
        raise ValueError("live inner exit cgroup scope drifted")
    cap = auditor.MAXIMUM_ESTIMATED_RSS_BYTES
    current = _require_exact_int_v2(
        cgroup.get("memory_current"), "live exit memory.current"
    )
    peak = _require_exact_int_v2(
        cgroup.get("memory_peak"), "live exit memory.peak"
    )
    memory_max = _require_exact_int_v2(
        cgroup.get("memory_max"), "live exit memory.max", minimum=1
    )
    memory_events = cgroup.get("memory_events")
    pids_events = cgroup.get("pids_events")
    if type(memory_events) is not dict or type(pids_events) is not dict:
        _fail(FAIL_RESOURCE, "live inner exit cgroup resource events are absent")
    event_max = _require_exact_int_v2(
        memory_events.get("max"), "live exit memory.events.max"
    )
    event_oom = _require_exact_int_v2(
        memory_events.get("oom"), "live exit memory.events.oom"
    )
    event_oom_kill = _require_exact_int_v2(
        memory_events.get("oom_kill"), "live exit memory.events.oom_kill"
    )
    pids_event_max = _require_exact_int_v2(
        pids_events.get("max"), "live exit pids.events.max"
    )
    if current > peak or peak > cap:
        _fail(
            FAIL_RESOURCE,
            "live inner exit physical resource gate rejected; "
            + _bounded_cgroup_physical_resource_detail_v2(
                stage="RUNTIME_EXIT_SAMPLE",
                cgroup=cgroup,
                cap=cap,
                reserve=0,
            ),
        )
    if (
        cgroup["memory_max"] != attempt.requested_cgroup_memory_max_bytes
        or cgroup["pids_max"] != attempt.requested_cgroup_pids_max
        or cgroup["pids_current"] != 1
        or cgroup["cgroup_procs"] != entry_cgroup.get("cgroup_procs")
        or cgroup["cgroup_stat"] != entry_cgroup.get("cgroup_stat")
        or cgroup["memory_peak"] < entry_cgroup.get("memory_peak", 0)
    ):
        raise ValueError("live inner exit cgroup events/topology drifted")
    try:
        _validate_zero_cgroup_events_v2(
            cgroup.get("memory_events"),
            "live exit memory",
            required_keys=frozenset({"max", "oom", "oom_kill"}),
        )
        _validate_zero_cgroup_events_v2(
            cgroup.get("pids_events"),
            "live exit pids",
            required_keys=frozenset({"max"}),
        )
    except ValueError as exc:
        raise Phase2BFormalWireCovertQualificationErrorV2(
            FAIL_RESOURCE,
            "live inner exit physical resource gate rejected; "
            + _bounded_cgroup_physical_resource_detail_v2(
                stage="RUNTIME_EXIT_SAMPLE",
                cgroup=cgroup,
                cap=cap,
                reserve=0,
            ),
        ) from exc
    if (
        not _exact_json_equal(
            cgroup.get("memory_events"), entry_cgroup.get("memory_events")
        )
        or not _exact_json_equal(
            cgroup.get("pids_events"), entry_cgroup.get("pids_events")
        )
    ):
        raise ValueError("live inner exit cgroup events/topology drifted")
    _validate_receipt_physical_scope_against_exit_cgroup_v2(
        runs,
        entry_cgroup,
        cgroup,
        context="live runtime capture",
    )
    return {
        "cgroup": cgroup,
        "audit_receipts_validated_and_released_in_process": True,
        "selected_default_signals": [
            number.name for number in LIVE_SELECTED_DEFAULT_SIGNALS
        ],
        "sigblk_hex": fields["SigBlk"],
        "threads": 1,
        "cgroup_events_unchanged_from_entry": True,
    }


def _validate_post_output_full_cgroup_snapshot_v2(
    runtime_exit: object,
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
    snapshot: object,
) -> int:
    """Gate the post-flush full snapshot against the serialized exit scope."""

    exit_cgroup = (
        runtime_exit.get("cgroup") if type(runtime_exit) is dict else None
    )
    if type(exit_cgroup) is not dict or type(snapshot) is not dict:
        _fail(FAIL_RESOURCE, "post-output cgroup evidence is absent")
    scope_fields = (
        "cgroup_path",
        "unified_cgroup_memory_scope_root",
        "unified_cgroup_leaf_device",
        "unified_cgroup_leaf_inode",
    )
    if any(snapshot.get(name) != exit_cgroup.get(name) for name in scope_fields):
        _fail(FAIL_RESOURCE, "post-output cgroup scope drifted after stdout flush")
    cap = auditor.MAXIMUM_ESTIMATED_RSS_BYTES
    ceiling = cap - POST_OUTPUT_DIRECT_EXIT_TAIL_RESERVE_BYTES_V2
    peak = _require_exact_int_v2(
        snapshot.get("memory_peak"), "post-output full memory.peak"
    )
    current = _require_exact_int_v2(
        snapshot.get("memory_current"), "post-output full memory.current"
    )
    exit_peak = _require_exact_int_v2(
        exit_cgroup.get("memory_peak"), "serialized runtime-exit memory.peak"
    )
    _require_safe_cgroup_path_v2(
        snapshot.get("cgroup_path"), "post-output full cgroup"
    )
    scope = snapshot.get("unified_cgroup_memory_scope_root")
    if type(scope) is not str or len(scope) > 128:
        _fail(FAIL_RESOURCE, "post-output cgroup scope root is malformed")
    memory_max = _require_exact_int_v2(
        snapshot.get("memory_max"), "post-output full memory.max", minimum=1
    )
    pids_max = _require_exact_int_v2(
        snapshot.get("pids_max"), "post-output full pids.max", minimum=1
    )
    pids_current = _require_exact_int_v2(
        snapshot.get("pids_current"), "post-output full pids.current"
    )
    memory_events = snapshot.get("memory_events")
    pids_events = snapshot.get("pids_events")
    if type(memory_events) is not dict or type(pids_events) is not dict:
        _fail(FAIL_RESOURCE, "post-output cgroup resource events are absent")
    memory_events_max = _require_exact_int_v2(
        memory_events.get("max"), "post-output memory.events.max"
    )
    memory_events_oom = _require_exact_int_v2(
        memory_events.get("oom"), "post-output memory.events.oom"
    )
    memory_events_oom_kill = _require_exact_int_v2(
        memory_events.get("oom_kill"), "post-output memory.events.oom_kill"
    )
    pids_events_max = _require_exact_int_v2(
        pids_events.get("max"), "post-output pids.events.max"
    )
    if (
        attempt.requested_cgroup_memory_max_bytes
        != OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2
        or memory_max != OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2
        or pids_max != attempt.requested_cgroup_pids_max
        or pids_current != 1
        or type(snapshot.get("cgroup_procs")) is not list
        or len(snapshot["cgroup_procs"]) != 1
        or type(snapshot.get("cgroup_stat")) is not dict
        or snapshot["cgroup_stat"].get("nr_descendants") != 0
        or snapshot["cgroup_stat"].get("nr_dying_descendants") != 0
        or current > peak
        or peak < exit_peak
        or peak > ceiling
        or memory_events_max != 0
        or memory_events_oom != 0
        or memory_events_oom_kill != 0
        or pids_events_max != 0
    ):
        _fail(
            FAIL_RESOURCE,
            "post-output full physical resource gate rejected; "
            + _bounded_cgroup_physical_resource_detail_v2(
                stage="POST_OUTPUT_FULL_SAMPLE",
                cgroup=snapshot,
                cap=cap,
                reserve=POST_OUTPUT_DIRECT_EXIT_TAIL_RESERVE_BYTES_V2,
            ),
        )
    _validate_zero_cgroup_events_v2(
        memory_events,
        "post-output memory",
        required_keys=frozenset({"max", "oom", "oom_kill"}),
    )
    _validate_zero_cgroup_events_v2(
        pids_events,
        "post-output pids",
        required_keys=frozenset({"max"}),
    )
    if (
        not _exact_json_equal(memory_events, exit_cgroup.get("memory_events"))
        or not _exact_json_equal(pids_events, exit_cgroup.get("pids_events"))
        or not _exact_json_equal(
            snapshot.get("cgroup_procs"), exit_cgroup.get("cgroup_procs")
        )
        or not _exact_json_equal(
            snapshot.get("cgroup_stat"), exit_cgroup.get("cgroup_stat")
        )
    ):
        _fail(
            FAIL_RESOURCE,
            "post-output cgroup events/process topology drifted after stdout flush",
        )
    return peak


def _fresh_current_cgroup_memory_peak_v2(
    expected_snapshot: dict[str, object],
    *,
    proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> int:
    """Fresh-open the same leaf's memory.peak as the final cgroup read."""

    expected_membership = _require_safe_cgroup_path_v2(
        expected_snapshot.get("cgroup_path"), "final peak cgroup"
    )
    membership_text = _read_bounded_cgroup_ascii_v2(
        proc_root / "self/cgroup", "final peak unified membership"
    )
    if membership_text not in {
        f"0::{expected_membership}",
        f"0::{expected_membership}\n",
    }:
        _fail(FAIL_RESOURCE, "final peak unified membership drifted")
    directory = cgroup_root / expected_membership.lstrip("/")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        _fail(FAIL_RESOURCE, "final peak read requires Linux no-follow dirfd flags")
    directory_fd: int | None = None
    peak_fd: int | None = None
    try:
        directory_fd = os.open(
            directory, flags | os.O_NOFOLLOW | os.O_DIRECTORY
        )
        opened = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened.st_dev != expected_snapshot.get("unified_cgroup_leaf_device")
            or opened.st_ino != expected_snapshot.get("unified_cgroup_leaf_inode")
        ):
            _fail(FAIL_RESOURCE, "final peak cgroup leaf identity drifted")
        peak_fd = os.open(
            "memory.peak", flags | os.O_NOFOLLOW, dir_fd=directory_fd
        )
        payload = os.read(peak_fd, 4_097)
        if len(payload) > 4_096 or os.read(peak_fd, 1):
            _fail(FAIL_RESOURCE, "final peak memory.peak is outside bounds")
    except OSError as exc:
        _fail(FAIL_RESOURCE, f"final peak memory.peak is unreadable: {exc.errno}")
    finally:
        _close_owned_descriptors_v2(
            (peak_fd, directory_fd), name="final peak cgroup read"
        )
    membership_replay = _read_bounded_cgroup_ascii_v2(
        proc_root / "self/cgroup", "final peak unified membership replay"
    )
    if membership_replay != membership_text:
        _fail(FAIL_RESOURCE, "final peak unified membership drifted during read")
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise Phase2BFormalWireCovertQualificationErrorV2(
            FAIL_RESOURCE, "final peak memory.peak is not exact ASCII"
        ) from exc
    return _parse_canonical_cgroup_decimal_v2(text, "final peak memory.peak")


def _validate_final_post_output_memory_peak_v2(
    full_snapshot: dict[str, object],
    full_peak: int,
    final_peak: object,
) -> None:
    """Require monotonic headroom immediately before the direct process exit."""

    final = _require_exact_int_v2(final_peak, "final fresh memory.peak")
    cap = auditor.MAXIMUM_ESTIMATED_RSS_BYTES
    ceiling = cap - POST_OUTPUT_DIRECT_EXIT_TAIL_RESERVE_BYTES_V2
    membership = _require_safe_cgroup_path_v2(
        full_snapshot.get("cgroup_path"), "final peak cgroup"
    )
    if final < full_peak or final > ceiling:
        overage = max(0, final - ceiling)
        _fail(
            FAIL_RESOURCE,
            "final fresh physical peak gate rejected; "
            f"scope={membership}; peak={final}; prior_peak={full_peak}; "
            f"cap={cap}; reserve={POST_OUTPUT_DIRECT_EXIT_TAIL_RESERVE_BYTES_V2}; "
            f"overage={overage}",
        )


def _bounded_post_output_failure_line_v2(exc: BaseException) -> bytes:
    detail = str(exc).replace("\r", " ").replace("\n", " ")
    detail = " ".join(detail.split())[:768]
    line = (
        f"{FAIL_RESOURCE}: post-output physical gate failed; "
        f"exception={type(exc).__name__}; detail={detail}\n"
    )
    return line.encode("ascii", errors="backslashreplace")[:1_024].rstrip(b"\n") + b"\n"


def _write_post_output_failure_line_v2(payload: bytes) -> None:
    written = os.write(2, payload)
    if written != len(payload):
        raise OSError("post-output failure stderr write was short")


def _inner_stdout_buffer_v2() -> object:
    return sys.stdout.buffer


def _prepare_post_output_quiescent_runtime_v2() -> None:
    """Remove Python-managed tail work and block selected terminal signals."""

    if sys.gettrace() is not None or sys.getprofile() is not None:
        _fail(FAIL_RESOURCE, "post-output gate rejects active trace/profile hooks")
    catchable = signal.valid_signals() - {signal.SIGKILL, signal.SIGSTOP}
    try:
        dispositions = {number: signal.getsignal(number) for number in catchable}
    except (OSError, RuntimeError, ValueError) as exc:
        raise Phase2BFormalWireCovertQualificationErrorV2(
            FAIL_RESOURCE, "post-output signal disposition inspection failed"
        ) from exc
    if any(callable(value) for value in dispositions.values()) or any(
        dispositions.get(number) is not signal.SIG_DFL
        for number in LIVE_SELECTED_DEFAULT_SIGNALS
    ):
        _fail(FAIL_RESOURCE, "post-output gate rejects Python signal handlers")
    gc.disable()
    if gc.isenabled():
        _fail(FAIL_RESOURCE, "post-output gate could not disable cyclic GC")
    selected = set(LIVE_SELECTED_DEFAULT_SIGNALS)
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, selected)
    if previous:
        _fail(FAIL_RESOURCE, "post-output gate observed a pre-existing signal mask")
    status = (Path("/proc") / "self/status").read_text(
        encoding="ascii", errors="strict"
    )
    fields: dict[str, str] = {}
    for line in status.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    expected_mask = sum(1 << (int(number) - 1) for number in selected)
    if (
        fields.get("Threads") != "1"
        or int(fields.get("SigBlk", "0"), 16) != expected_mask
    ):
        _fail(FAIL_RESOURCE, "post-output signal/thread quiescence drifted")


def _write_flush_gate_and_direct_exit_v2(
    artifact: bytes,
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
) -> NoReturn:
    """Publish one canonical inner artifact, gate its tail, then bypass teardown."""

    try:
        body = _decode_canonical_transport_object_v2(
            artifact, "inner CLI artifact", maximum_bytes=8 * 1_024 * 1_024
        )
        runtime_exit = body.get("runtime_exit_evidence")
        if not _exact_json_equal(
            body.get("post_output_physical_gate_contract"),
            _POST_OUTPUT_PHYSICAL_GATE_CONTRACT_V2,
        ):
            _fail(FAIL_RESOURCE, "inner post-output physical gate contract drifted")
        stdout_buffer = _inner_stdout_buffer_v2()
        written = stdout_buffer.write(artifact)
        if written != len(artifact):
            raise OSError("inner canonical stdout write was short")
        stdout_buffer.flush()
        _prepare_post_output_quiescent_runtime_v2()
        full_snapshot = inspect_current_live_cgroup_v2()
        full_peak = _validate_post_output_full_cgroup_snapshot_v2(
            runtime_exit, attempt, full_snapshot
        )
        direct_exit = _DIRECT_PROCESS_EXIT_V2
        final_ceiling = (
            auditor.MAXIMUM_ESTIMATED_RSS_BYTES
            - POST_OUTPUT_DIRECT_EXIT_TAIL_RESERVE_BYTES_V2
        )
        final_peak = _fresh_current_cgroup_memory_peak_v2(full_snapshot)
        if type(final_peak) is not int or final_peak < full_peak or final_peak > final_ceiling:
            _validate_final_post_output_memory_peak_v2(
                full_snapshot, full_peak, final_peak
            )
    except BaseException as exc:
        try:
            _write_post_output_failure_line_v2(
                _bounded_post_output_failure_line_v2(exc)
            )
        finally:
            _DIRECT_PROCESS_EXIT_V2(86)
        raise RuntimeError("direct nonzero os._exit returned unexpectedly")
    direct_exit(0)
    raise RuntimeError("direct zero os._exit returned unexpectedly")


def _exact_probability_text_v2(value: auditor.ExactProbabilityV1) -> str:
    return f"{value.numerator}/{value.denominator}"


def _closed_primitive_v2(value: object) -> object:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return {"binary64_hex": value.hex()}
    if type(value) is bytes:
        return {"bytes_hex": value.hex()}
    if isinstance(value, Enum):
        return value.value
    if type(value) is tuple:
        return [_closed_primitive_v2(item) for item in value]
    if type(value) is list:
        return [_closed_primitive_v2(item) for item in value]
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("closed primitive mapping keys must be exact strings")
        return {
            key: _closed_primitive_v2(value[key]) for key in sorted(value)
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _closed_primitive_v2(getattr(value, item.name))
            for item in dataclass_fields(value)
        }
    raise TypeError(
        f"closed primitive projection rejects {type(value).__name__}"
    )


def _static_resource_plan_projection_v2(
    plan: auditor.ResourcePlanV1,
) -> dict[str, object]:
    if type(plan) is not auditor.ResourcePlanV1:
        raise TypeError("static resource projection needs the exact ResourcePlanV1")
    plan.__post_init__()
    known_fields = set(FROZEN_STATIC_RESOURCE_PLAN_FIELDS_V2) | set(
        DYNAMIC_RESOURCE_PLAN_FIELDS_NOT_EXACT_COMPARED_V2
    )
    actual_fields = {item.name for item in dataclass_fields(plan)}
    if actual_fields != known_fields:
        raise ValueError("resource-plan static/dynamic whitelist is no longer exhaustive")
    return {
        name: _closed_primitive_v2(getattr(plan, name))
        for name in FROZEN_STATIC_RESOURCE_PLAN_FIELDS_V2
    }


def _replay_complete_resource_plan_projection_v2(
    value: object, *, context: str
) -> auditor.ResourcePlanV1:
    if type(value) is not dict or set(value) != _RESOURCE_PLAN_FIELDS_V2:
        raise ValueError(f"{context} resource-plan fields are not exact")
    try:
        replay = auditor.ResourcePlanV1(
            **{name: value[name] for name in _RESOURCE_PLAN_INIT_FIELDS_V2}
        )
        replay.__post_init__()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} resource-plan semantics drifted") from exc
    if _closed_primitive_v2(replay) != value:
        raise ValueError(f"{context} resource-plan derived root drifted")
    return replay


def _validate_complete_resource_plan_projection_v2(
    value: object, *, context: str
) -> dict[str, object]:
    _replay_complete_resource_plan_projection_v2(value, context=context)
    return value


def _replay_phase_resource_preflight_projection_v2(
    value: object,
    *,
    phase: str,
    resource_plan_root: str,
    context: str,
) -> auditor._PhaseResourcePreflightEvidenceV1:
    if type(value) is not dict or set(value) != _PHASE_RESOURCE_PREFLIGHT_FIELDS_V2:
        raise ValueError(f"{context} phase resource-preflight fields are not exact")
    fields = dataclass_fields(auditor._PhaseResourcePreflightEvidenceV1)
    try:
        replay = auditor._PhaseResourcePreflightEvidenceV1(
            **{item.name: value[item.name] for item in fields if item.init}
        )
        replay.__post_init__()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} phase resource-preflight semantics drifted"
        ) from exc
    if (
        replay.phase != phase
        or replay.resource_plan_root != resource_plan_root
        or _closed_primitive_v2(replay) != value
    ):
        raise ValueError(f"{context} phase resource-preflight root/kind drifted")
    return replay


def _validate_complete_invariant_runtime_projection_v2(
    value: object,
    resource_plan: object,
    *,
    context: str,
) -> dict[str, object]:
    plan = _replay_complete_resource_plan_projection_v2(
        resource_plan,
        context=f"{context} invariant runtime",
    )
    fields = dataclass_fields(auditor.InvariantRuntimeEvidenceV1)
    expected_fields = {item.name for item in fields}
    if type(value) is not dict or set(value) != expected_fields:
        raise ValueError(f"{context} invariant runtime fields are not exact")
    preflight = _replay_phase_resource_preflight_projection_v2(
        value.get("resource_preflight"),
        phase="INVARIANCE",
        resource_plan_root=plan.invariance_resource_plan_root,
        context=f"{context} invariant runtime",
    )
    try:
        arguments = {item.name: value[item.name] for item in fields if item.init}
        arguments["resource_preflight"] = preflight
        replay = auditor.InvariantRuntimeEvidenceV1(**arguments)
        replay.__post_init__()
        auditor._validate_invariant_phase_preflight_against_resource_plan(
            plan,
            preflight,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} invariant runtime semantics drifted") from exc
    if (
        replay.invariance_resource_plan_root
        != plan.invariance_resource_plan_root
        or _closed_primitive_v2(replay) != value
    ):
        raise ValueError(f"{context} invariant runtime/plan binding drifted")
    return value


def _validate_complete_statistics_runtime_projection_v2(
    value: object,
    resource_plan: object,
    *,
    context: str,
) -> dict[str, object]:
    plan = _replay_complete_resource_plan_projection_v2(
        resource_plan,
        context=f"{context} statistics runtime",
    )
    fields = dataclass_fields(auditor.StatisticalRuntimeEvidenceV1)
    expected_fields = {item.name for item in fields}
    if type(value) is not dict or set(value) != expected_fields:
        raise ValueError(f"{context} statistics runtime fields are not exact")
    preflight = _replay_phase_resource_preflight_projection_v2(
        value.get("resource_preflight"),
        phase="STATISTICS",
        resource_plan_root=plan.resource_plan_root,
        context=f"{context} statistics runtime",
    )
    try:
        arguments = {item.name: value[item.name] for item in fields if item.init}
        arguments["resource_preflight"] = preflight
        for tuple_field in (
            "worker_peak_rss_bytes",
            "worker_result_artifact_bytes",
            "worker_result_artifact_sha256",
        ):
            if type(arguments[tuple_field]) is not list:
                raise TypeError(
                    f"{context} statistics runtime {tuple_field} is not a closed list"
                )
            arguments[tuple_field] = tuple(arguments[tuple_field])
        replay = auditor.StatisticalRuntimeEvidenceV1(**arguments)
        replay.__post_init__()
        auditor._validate_statistics_phase_preflight_against_resource_plan(
            plan,
            preflight,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} statistics runtime semantics drifted") from exc
    if (
        replay.resource_plan_root != plan.resource_plan_root
        or _closed_primitive_v2(replay) != value
    ):
        raise ValueError(f"{context} statistics runtime/plan binding drifted")
    return value


def _validate_complete_receipt_issuance_runtime_projection_v2(
    value: object,
    statistics_runtime: object,
    *,
    context: str,
) -> dict[str, object]:
    expected_fields = set(_RECEIPT_ISSUANCE_RUNTIME_FIELDS_V2)
    if type(value) is not dict or set(value) != expected_fields:
        raise ValueError(f"{context} receipt-issuance runtime fields are not exact")
    try:
        replay = auditor.ReceiptIssuanceRuntimeEvidenceV1(
            **{name: value[name] for name in _RECEIPT_ISSUANCE_RUNTIME_FIELDS_V2}
        )
        replay.__post_init__()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} receipt-issuance runtime semantics drifted"
        ) from exc
    if _closed_primitive_v2(replay) != value:
        raise ValueError(f"{context} receipt-issuance runtime replay drifted")
    if statistics_runtime is None:
        if (
            replay.absolute_deadline_kind != "PARTITION_PREPARATION"
            or replay.diagnostic_worker_historical_peak_rss_bytes_sum != 0
            or replay.physical_memory_cap_bytes != auditor.MAXIMUM_ESTIMATED_RSS_BYTES
        ):
            raise ValueError(
                f"{context} insufficient receipt-issuance branch drifted"
            )
        return value
    statistics_fields = {
        item.name for item in dataclass_fields(auditor.StatisticalRuntimeEvidenceV1)
    }
    if type(statistics_runtime) is not dict or set(statistics_runtime) != statistics_fields:
        raise ValueError(
            f"{context} statistical receipt-issuance cross-binding is not closed"
        )
    worker_peaks = statistics_runtime.get("worker_peak_rss_bytes")
    if type(worker_peaks) is not list or any(type(item) is not int for item in worker_peaks):
        raise ValueError(
            f"{context} statistical receipt-issuance worker peaks are not exact"
        )
    if (
        replay.absolute_deadline_kind != "STATISTICS"
        or replay.diagnostic_worker_historical_peak_rss_bytes_sum
        != sum(worker_peaks)
        or replay.diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization
        != statistics_runtime.get(
            "parent_historical_peak_rss_bytes_before_receipt_finalization"
        )
        or replay.diagnostic_parent_plus_worker_historical_peak_sum_bytes
        != statistics_runtime.get(
            "diagnostic_parent_plus_worker_historical_peak_sum_bytes"
        )
        or replay.unified_cgroup_membership
        != statistics_runtime.get("unified_cgroup_membership")
        or replay.unified_cgroup_memory_scope_root
        != statistics_runtime.get("unified_cgroup_memory_scope_root")
        or replay.unified_cgroup_leaf_device
        != statistics_runtime.get("unified_cgroup_leaf_device")
        or replay.unified_cgroup_leaf_inode
        != statistics_runtime.get("unified_cgroup_leaf_inode")
        or replay.physical_memory_current_bytes_before_receipt_finalization
        != statistics_runtime.get(
            "physical_memory_current_bytes_before_receipt_finalization"
        )
        or replay.physical_memory_peak_bytes_before_receipt_finalization
        != statistics_runtime.get(
            "physical_memory_peak_bytes_before_receipt_finalization"
        )
        or replay.physical_memory_max_bytes
        != statistics_runtime.get("physical_memory_max_bytes")
        or replay.physical_memory_cap_bytes
        != statistics_runtime.get("physical_memory_cap_bytes")
        or replay.receipt_finalization_complete_pending_validation_count
        != statistics_runtime.get(
            "receipt_finalization_complete_pending_validation_count"
        )
        or replay.receipt_finalization_locked_physical_peak_within_cap
        is not statistics_runtime.get(
            "receipt_finalization_physical_peak_within_cap"
        )
    ):
        raise ValueError(
            f"{context} statistical and receipt-issuance runtime cross-binding drifted"
        )
    return value


def _validate_candidate_branch_physical_scope_v2(
    positive_statistics: dict[str, object],
    negative_statistics: dict[str, object],
) -> None:
    """Bind sequential candidate audits to one non-reset cgroup leaf."""

    scope_fields = (
        "unified_cgroup_membership",
        "unified_cgroup_memory_scope_root",
        "unified_cgroup_leaf_device",
        "unified_cgroup_leaf_inode",
        "physical_memory_max_bytes",
        "physical_memory_cap_bytes",
    )
    positive_scope = tuple(positive_statistics.get(name) for name in scope_fields)
    negative_scope = tuple(negative_statistics.get(name) for name in scope_fields)
    positive_peak = _require_exact_int_v2(
        positive_statistics.get(
            "physical_memory_peak_bytes_before_receipt_finalization"
        ),
        "candidate positive physical memory peak",
    )
    negative_peak = _require_exact_int_v2(
        negative_statistics.get(
            "physical_memory_peak_bytes_before_receipt_finalization"
        ),
        "candidate negative physical memory peak",
    )
    if positive_scope != negative_scope or negative_peak < positive_peak:
        raise ValueError("candidate audit branches changed physical memory scope")


def _validate_executed_runtime_bundle_v2(
    *,
    resource_plan: object,
    statistics_runtime: object,
    receipt_issuance_runtime: object,
    diagnostics: object,
    run: object,
    expected_mode: str,
    visible_audited_rows: int,
    visible_concrete_features: int,
    context: str,
    measurement: object = None,
) -> None:
    """Replay the detached executed-runtime evidence as one closed bundle."""

    _validate_complete_resource_plan_projection_v2(
        resource_plan,
        context=f"{context} executed bundle",
    )
    _validate_complete_statistics_runtime_projection_v2(
        statistics_runtime,
        resource_plan,
        context=f"{context} executed bundle",
    )
    _validate_complete_receipt_issuance_runtime_projection_v2(
        receipt_issuance_runtime,
        statistics_runtime,
        context=f"{context} executed bundle",
    )
    expected_diagnostic_keys = {
        "host_available_memory_bytes_at_preflight",
        "maximum_rows_per_invocation",
        "minimum_rows_per_invocation",
        "mode",
        "recognizer_adapter_invocations",
        "parent_historical_peak_rss_before_receipt_finalization_kib",
        "parent_plus_worker_historical_peak_rss_kib",
        "physical_memory_cap_bytes",
        "physical_memory_current_before_receipt_finalization_bytes",
        "physical_memory_max_bytes",
        "physical_memory_peak_before_receipt_finalization_bytes",
        "statistics_wall_time_ns",
        "statistics_worker_historical_peak_rss_kib",
        "unified_cgroup_membership",
        "unified_cgroup_leaf_device",
        "unified_cgroup_leaf_inode",
        "unified_cgroup_memory_scope_root",
        "wall_time_ns",
    }
    if (
        type(resource_plan) is not dict
        or type(statistics_runtime) is not dict
        or type(receipt_issuance_runtime) is not dict
        or type(diagnostics) is not dict
        or set(diagnostics) != expected_diagnostic_keys
        or expected_mode not in _RUN_MODES
        or type(visible_audited_rows) is not int
        or type(visible_concrete_features) is not int
        or visible_audited_rows != EXACT_AUTHORITY_ROWS
        or visible_concrete_features <= 0
        or resource_plan.get("rows") != visible_audited_rows
        or resource_plan.get("concrete_features") != visible_concrete_features
        or resource_plan.get("admissible") is not True
        or diagnostics.get("mode") != expected_mode
    ):
        raise ValueError(f"{context} executed bundle projections are not exact")
    try:
        diagnostic_arguments = {
            item.name: diagnostics[item.name]
            for item in dataclass_fields(AuditRunDiagnosticsV2)
        }
        if type(diagnostic_arguments["statistics_worker_historical_peak_rss_kib"]) is not list:
            raise TypeError("diagnostic worker peaks are not one closed list")
        diagnostic_arguments["statistics_worker_historical_peak_rss_kib"] = tuple(
            diagnostic_arguments["statistics_worker_historical_peak_rss_kib"]
        )
        diagnostic_replay = AuditRunDiagnosticsV2(**diagnostic_arguments)
        diagnostic_replay.__post_init__()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} diagnostic projection semantics drifted") from exc
    if not _exact_json_equal(_diagnostic_projection(diagnostic_replay), diagnostics):
        raise ValueError(f"{context} diagnostic projection replay drifted")
    worker_peaks = statistics_runtime.get("worker_peak_rss_bytes")
    artifact_bytes = statistics_runtime.get("worker_result_artifact_bytes")
    artifact_sha256 = statistics_runtime.get("worker_result_artifact_sha256")
    diagnostic_worker_peaks = diagnostics.get(
        "statistics_worker_historical_peak_rss_kib"
    )
    if (
        len(auditor.TARGETS) != 5
        or type(worker_peaks) is not list
        or type(artifact_bytes) is not list
        or type(artifact_sha256) is not list
        or type(diagnostic_worker_peaks) is not list
        or len(worker_peaks) != resource_plan.get("maximum_parallel_workers")
        or len(artifact_bytes) != len(auditor.TARGETS)
        or sum(artifact_bytes) > resource_plan.get("worker_result_artifact_bytes_bound")
        or statistics_runtime.get("parent_result_object_graph_bytes_observed")
        > resource_plan.get("parent_result_working_bytes_bound")
        or worker_peaks != [value * 1_024 for value in diagnostic_worker_peaks]
        or statistics_runtime.get(
            "parent_historical_peak_rss_bytes_before_receipt_finalization"
        )
        != diagnostics.get(
            "parent_historical_peak_rss_before_receipt_finalization_kib"
        )
        * 1_024
        or statistics_runtime.get(
            "diagnostic_parent_plus_worker_historical_peak_sum_bytes"
        )
        != diagnostics.get(
            "parent_plus_worker_historical_peak_rss_kib"
        )
        * 1_024
        or statistics_runtime.get("unified_cgroup_membership")
        != diagnostics.get("unified_cgroup_membership")
        or statistics_runtime.get("unified_cgroup_leaf_device")
        != diagnostics.get("unified_cgroup_leaf_device")
        or statistics_runtime.get("unified_cgroup_leaf_inode")
        != diagnostics.get("unified_cgroup_leaf_inode")
        or statistics_runtime.get("unified_cgroup_memory_scope_root")
        != diagnostics.get("unified_cgroup_memory_scope_root")
        or statistics_runtime.get(
            "physical_memory_current_bytes_before_receipt_finalization"
        )
        != diagnostics.get(
            "physical_memory_current_before_receipt_finalization_bytes"
        )
        or statistics_runtime.get(
            "physical_memory_peak_bytes_before_receipt_finalization"
        )
        != diagnostics.get(
            "physical_memory_peak_before_receipt_finalization_bytes"
        )
        or statistics_runtime.get("physical_memory_max_bytes")
        != diagnostics.get("physical_memory_max_bytes")
        or statistics_runtime.get("physical_memory_cap_bytes")
        != diagnostics.get("physical_memory_cap_bytes")
        or diagnostics.get("physical_memory_cap_bytes")
        != auditor.MAXIMUM_ESTIMATED_RSS_BYTES
        or statistics_runtime.get(
            "statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds"
        )
        != diagnostics.get("statistics_wall_time_ns")
        or statistics_runtime.get("host_available_memory_bytes_at_preflight")
        != diagnostics.get("host_available_memory_bytes_at_preflight")
        or diagnostics.get("minimum_rows_per_invocation")
        != resource_plan.get("rows")
        or diagnostics.get("maximum_rows_per_invocation")
        != resource_plan.get("rows")
    ):
        raise ValueError(f"{context} statistics/plan/diagnostics bundle drifted")
    if run is None:
        if measurement is not None:
            raise ValueError(f"{context} measurement has no paired run")
        return
    if type(run) is not dict or type(run.get("audit")) is not dict:
        raise ValueError(f"{context} executed run bundle is absent")
    audit_row = run["audit"]
    wire_row = run.get("wire")
    invariant = audit_row.get("invariance")
    statistics = audit_row.get("statistics")
    issuance = audit_row.get("receipt_issuance")
    quota = audit_row.get("eligibility_quota")
    target_rows = audit_row.get("target_family_max_t_holm")
    if any(
        type(item) is not dict
        for item in (wire_row, invariant, statistics, issuance, quota)
    ) or type(target_rows) is not list:
        raise ValueError(f"{context} executed run groups are absent")
    _require_eligibility_quota_receipt_id_v2(
        quota.get("receipt_id"),
        f"{context} eligibility-mask quota receipt ID",
    )
    if audit_row.get("eligibility_quota_root") != stable_hash(
        quota,
        prefix="phase2b_formal_wire_covert_eligibility_quota_v2_",
    ):
        raise ValueError(f"{context} eligibility-mask quota root drifted")
    invariant_complete = invariant.get("complete_runtime_projection")
    statistics_complete = statistics.get("complete_runtime_projection")
    issuance_complete = issuance.get("complete_runtime_projection")
    _validate_complete_invariant_runtime_projection_v2(
        invariant_complete,
        resource_plan,
        context=f"{context} executed bundle",
    )
    invariant_scope = (
        invariant_complete.get("unified_cgroup_membership"),
        invariant_complete.get("unified_cgroup_memory_scope_root"),
        invariant_complete.get("unified_cgroup_leaf_device"),
        invariant_complete.get("unified_cgroup_leaf_inode"),
        invariant_complete.get("physical_memory_max_bytes"),
        invariant_complete.get("physical_memory_cap_bytes"),
    )
    statistics_scope = (
        statistics_runtime.get("unified_cgroup_membership"),
        statistics_runtime.get("unified_cgroup_memory_scope_root"),
        statistics_runtime.get("unified_cgroup_leaf_device"),
        statistics_runtime.get("unified_cgroup_leaf_inode"),
        statistics_runtime.get("physical_memory_max_bytes"),
        statistics_runtime.get("physical_memory_cap_bytes"),
    )
    issuance_scope = (
        receipt_issuance_runtime.get("unified_cgroup_membership"),
        receipt_issuance_runtime.get("unified_cgroup_memory_scope_root"),
        receipt_issuance_runtime.get("unified_cgroup_leaf_device"),
        receipt_issuance_runtime.get("unified_cgroup_leaf_inode"),
        receipt_issuance_runtime.get("physical_memory_max_bytes"),
        receipt_issuance_runtime.get("physical_memory_cap_bytes"),
    )
    if (
        not _exact_json_equal(statistics_complete, statistics_runtime)
        or not _exact_json_equal(issuance_complete, receipt_issuance_runtime)
        or not _exact_json_equal(run.get("diagnostics"), diagnostics)
        or invariant_scope != statistics_scope
        or statistics_scope != issuance_scope
        or invariant_complete.get("physical_memory_peak_bytes_at_completion")
        > statistics_runtime.get(
            "physical_memory_peak_bytes_before_receipt_finalization"
        )
        or invariant.get("complete_runtime_projection_root")
        != stable_hash(
            invariant_complete,
            prefix="phase2b_formal_wire_covert_invariant_runtime_v4_",
        )
        or statistics.get("complete_runtime_projection_root")
        != stable_hash(
            statistics_runtime,
            prefix="phase2b_formal_wire_covert_statistics_runtime_v4_",
        )
        or issuance.get("complete_runtime_projection_root")
        != stable_hash(
            receipt_issuance_runtime,
            prefix=(
                "phase2b_formal_wire_covert_receipt_issuance_runtime_v4_"
            ),
        )
    ):
        raise ValueError(f"{context} compact/full runtime selection drifted")
    receipt_id = _require_auditor_receipt_id_v2(
        audit_row.get("receipt_id"),
        f"{context} run audit receipt ID",
    )
    _require_auditor_receipt_id_v2(
        issuance.get("receipt_id"),
        f"{context} issuance receipt ID",
    )
    if (
        issuance.get("receipt_id") != receipt_id
        or issuance.get("absolute_deadline_kind")
        != receipt_issuance_runtime.get("absolute_deadline_kind")
        or not _exact_json_equal(
            issuance.get("diagnostic_worker_historical_peak_rss_bytes_sum"),
            receipt_issuance_runtime.get(
                "diagnostic_worker_historical_peak_rss_bytes_sum"
            ),
        )
        or not _exact_json_equal(
            issuance.get(
                "diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization"
            ),
            receipt_issuance_runtime.get(
                "diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization"
            ),
        )
        or not _exact_json_equal(
            issuance.get(
                "diagnostic_parent_plus_worker_historical_peak_sum_bytes"
            ),
            receipt_issuance_runtime.get(
                "diagnostic_parent_plus_worker_historical_peak_sum_bytes"
            ),
        )
        or issuance.get("unified_cgroup_membership")
        != receipt_issuance_runtime.get("unified_cgroup_membership")
        or issuance.get("unified_cgroup_memory_scope_root")
        != receipt_issuance_runtime.get("unified_cgroup_memory_scope_root")
        or not _exact_json_equal(
            issuance.get("unified_cgroup_leaf_device"),
            receipt_issuance_runtime.get("unified_cgroup_leaf_device"),
        )
        or not _exact_json_equal(
            issuance.get("unified_cgroup_leaf_inode"),
            receipt_issuance_runtime.get("unified_cgroup_leaf_inode"),
        )
        or not _exact_json_equal(
            issuance.get(
                "physical_memory_current_bytes_before_receipt_finalization"
            ),
            receipt_issuance_runtime.get(
                "physical_memory_current_bytes_before_receipt_finalization"
            ),
        )
        or not _exact_json_equal(
            issuance.get(
                "physical_memory_peak_bytes_before_receipt_finalization"
            ),
            receipt_issuance_runtime.get(
                "physical_memory_peak_bytes_before_receipt_finalization"
            ),
        )
        or not _exact_json_equal(
            issuance.get("physical_memory_cap_bytes"),
            receipt_issuance_runtime.get("physical_memory_cap_bytes"),
        )
        or not _exact_json_equal(
            issuance.get("physical_memory_max_bytes"),
            receipt_issuance_runtime.get("physical_memory_max_bytes"),
        )
        or not _exact_json_equal(
            issuance.get(
                "receipt_finalization_complete_pending_validation_count"
            ),
            receipt_issuance_runtime.get(
                "receipt_finalization_complete_pending_validation_count"
            ),
        )
        or not _exact_json_equal(
            issuance.get("receipt_finalization_locked_deadline_check_count"),
            receipt_issuance_runtime.get(
                "receipt_finalization_locked_deadline_check_count"
            ),
        )
        or issuance.get(
            "receipt_finalization_locked_deadline_within_absolute_limit"
        )
        is not receipt_issuance_runtime.get(
            "receipt_finalization_locked_deadline_within_absolute_limit"
        )
        or issuance.get(
            "receipt_finalization_locked_physical_peak_within_cap"
        )
        is not receipt_issuance_runtime.get(
            "receipt_finalization_locked_physical_peak_within_cap"
        )
    ):
        raise ValueError(f"{context} transcript/issuance ID binding drifted")
    mode = run.get("mode")
    if (
        mode != expected_mode
        or run.get("disposition")
        != ("PASS" if mode == "KNOWN_NULL" else "FAIL")
        or audit_row.get("audited_rows") != visible_audited_rows
        or audit_row.get("concrete_features") != visible_concrete_features
        or audit_row.get("resource_plan_root")
        != stable_hash(
            resource_plan,
            prefix="phase2b_formal_wire_covert_resource_plan_v3_",
        )
        or re.fullmatch(
            r"phase2b_formal_covert_feature_manifest_[0-9a-f]{64}",
            audit_row.get("feature_manifest_root", ""),
        )
        is None
    ):
        raise ValueError(f"{context} visible manifest/resource binding drifted")
    if (
        re.fullmatch(
            r"phase2b_formal_covert_adapter_execution_pins_[0-9a-f]{64}",
            wire_row.get("adapter_execution_pin_root", ""),
        )
        is None
        or invariant_complete.get("adapter_execution_pin_root")
        != wire_row.get("adapter_execution_pin_root")
        or invariant.get("invariance_resource_plan_root")
        != invariant_complete.get("invariance_resource_plan_root")
        or invariant_complete.get("invariance_resource_plan_root")
        != resource_plan.get("invariance_resource_plan_root")
        or not _exact_json_equal(
            invariant.get("consumer_decode_count"),
            invariant_complete.get("consumer_decode_count"),
        )
        or invariant_complete.get("consumer_decode_count")
        != resource_plan.get("rows") * 80
        or not _exact_json_equal(
            invariant.get("run_count"),
            invariant_complete.get("invariant_run_count"),
        )
        or diagnostics.get("recognizer_adapter_invocations")
        != invariant_complete.get("invariant_run_count")
        or not _exact_json_equal(
            invariant.get("exact_comparisons"),
            invariant_complete.get("exact_invariant_comparisons"),
        )
        or invariant_complete.get("exact_invariant_comparisons")
        != resource_plan.get("rows") * 80 * 5
        or invariant.get("variant_suite_root")
        != invariant_complete.get("variant_suite_root")
        or invariant.get("worker_artifact_sha256")
        != invariant_complete.get("worker_result_artifact_sha256")
        or not _exact_json_equal(
            invariant.get("worker_peak_rss_bytes"),
            invariant_complete.get("worker_peak_rss_bytes"),
        )
        or not _exact_json_equal(
            invariant.get("wall_time_nanoseconds"),
            invariant_complete.get("invariance_wall_time_nanoseconds"),
        )
        or invariant.get("unified_cgroup_membership")
        != invariant_complete.get("unified_cgroup_membership")
        or invariant.get("unified_cgroup_memory_scope_root")
        != invariant_complete.get("unified_cgroup_memory_scope_root")
        or not _exact_json_equal(
            invariant.get("unified_cgroup_leaf_device"),
            invariant_complete.get("unified_cgroup_leaf_device"),
        )
        or not _exact_json_equal(
            invariant.get("unified_cgroup_leaf_inode"),
            invariant_complete.get("unified_cgroup_leaf_inode"),
        )
        or not _exact_json_equal(
            invariant.get("physical_memory_current_bytes_at_completion"),
            invariant_complete.get("physical_memory_current_bytes_at_completion"),
        )
        or not _exact_json_equal(
            invariant.get("physical_memory_peak_bytes_at_completion"),
            invariant_complete.get("physical_memory_peak_bytes_at_completion"),
        )
        or not _exact_json_equal(
            invariant.get("physical_memory_max_bytes"),
            invariant_complete.get("physical_memory_max_bytes"),
        )
        or not _exact_json_equal(
            invariant.get("physical_memory_cap_bytes"),
            invariant_complete.get("physical_memory_cap_bytes"),
        )
        or invariant.get("within_frozen_physical_peak_cap")
        is not invariant_complete.get("within_frozen_physical_peak_cap")
        or invariant_complete.get(
            "variant_suite_resident_object_graph_bytes_observed"
        )
        > resource_plan.get("invariance_variant_suite_resident_bytes_bound")
        or re.fullmatch(
            r"phase2b_formal_covert_invariant_execution_[0-9a-f]{64}",
            invariant.get("execution_root", ""),
        )
        is None
    ):
        raise ValueError(f"{context} compact/full invariance binding drifted")
    if (
        statistics.get("resource_plan_root")
        != statistics_runtime.get("resource_plan_root")
        or statistics_runtime.get("resource_plan_root")
        != resource_plan.get("resource_plan_root")
        or statistics.get("worker_artifact_sha256")
        != statistics_runtime.get("worker_result_artifact_sha256")
        or not _exact_json_equal(
            statistics.get("worker_peak_rss_bytes"),
            statistics_runtime.get("worker_peak_rss_bytes"),
        )
        or not _exact_json_equal(
            statistics.get(
                "statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds"
            ),
            statistics_runtime.get(
                "statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds"
            ),
        )
        or not _exact_json_equal(
            statistics.get(
                "diagnostic_parent_plus_worker_historical_peak_sum_bytes"
            ),
            statistics_runtime.get(
                "diagnostic_parent_plus_worker_historical_peak_sum_bytes"
            ),
        )
        or statistics.get("unified_cgroup_membership")
        != statistics_runtime.get("unified_cgroup_membership")
        or statistics.get("unified_cgroup_memory_scope_root")
        != statistics_runtime.get("unified_cgroup_memory_scope_root")
        or not _exact_json_equal(
            statistics.get("unified_cgroup_leaf_device"),
            statistics_runtime.get("unified_cgroup_leaf_device"),
        )
        or not _exact_json_equal(
            statistics.get("unified_cgroup_leaf_inode"),
            statistics_runtime.get("unified_cgroup_leaf_inode"),
        )
        or not _exact_json_equal(
            statistics.get(
                "physical_memory_current_bytes_before_receipt_finalization"
            ),
            statistics_runtime.get(
                "physical_memory_current_bytes_before_receipt_finalization"
            ),
        )
        or not _exact_json_equal(
            statistics.get(
                "physical_memory_peak_bytes_before_receipt_finalization"
            ),
            statistics_runtime.get(
                "physical_memory_peak_bytes_before_receipt_finalization"
            ),
        )
        or not _exact_json_equal(
            statistics.get("physical_memory_max_bytes"),
            statistics_runtime.get("physical_memory_max_bytes"),
        )
        or not _exact_json_equal(
            statistics.get("physical_memory_cap_bytes"),
            statistics_runtime.get("physical_memory_cap_bytes"),
        )
        or statistics.get("external_supervisor_required")
        is not statistics_runtime.get("external_supervisor_required")
        or statistics.get("within_frozen_physical_peak_cap")
        is not statistics_runtime.get("within_frozen_physical_peak_cap")
        or statistics.get("pairwise_holm_decision_eligible") is not False
        or statistics.get("pairwise_holm_family_size")
        != 5 * resource_plan.get("concrete_features")
        or re.fullmatch(
            r"phase2b_formal_covert_permutation_schedule_[0-9a-f]{64}",
            statistics.get("permutation_schedule_root", ""),
        )
        is None
        or type(statistics.get("permutation_seed_commitment")) is not str
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            statistics.get("permutation_seed_commitment", ""),
        )
        is None
        or type(statistics.get("permutation_seed_reveal_hex")) is not str
        or re.fullmatch(
            r"[0-9a-f]{64}",
            statistics.get("permutation_seed_reveal_hex", ""),
        )
        is None
        or auditor._permutation_seed_commitment(
            bytes.fromhex(statistics["permutation_seed_reveal_hex"])
        )
        != statistics.get("permutation_seed_commitment")
    ):
        raise ValueError(f"{context} compact/full statistics binding drifted")
    if len(target_rows) != 5:
        raise ValueError(f"{context} target-family size drifted")
    _validate_target_family_holm_projection_v2(target_rows, context=context)
    if measurement is not None:
        if type(measurement) is not dict:
            raise ValueError(f"{context} measurement is absent")
        observed_id = _require_auditor_receipt_id_v2(
            measurement.get("observed_receipt_id"),
            f"{context} observed receipt ID",
        )
        if (
            observed_id != receipt_id
            or not _exact_json_equal(
                measurement.get("observed_complete_resource_plan"),
                resource_plan,
            )
            or not _exact_json_equal(
                measurement.get("observed_complete_receipt_issuance_runtime"),
                receipt_issuance_runtime,
            )
            or not _exact_json_equal(measurement.get("diagnostics"), diagnostics)
        ):
            raise ValueError(f"{context} measurement/runtime bundle drifted")


def _eligibility_quota_projection_v2(
    receipt: auditor.FormalCovertAuditReceiptV1,
) -> dict[str, object]:
    quota = receipt.eligibility_mask_quota
    quota_receipt_id = _require_eligibility_quota_receipt_id_v2(
        quota.receipt_id,
        "eligibility-mask quota projection receipt ID",
    )
    return {
        "exact_quota_matched": quota.exact_quota_matched,
        "receipt_id": quota_receipt_id,
        "targets": [
            {
                "target": row.target,
                "checked_integer_cells": row.checked_integer_cells,
                "maximum_absolute_cross_product_difference": (
                    row.maximum_absolute_cross_product_difference
                ),
                "exact_quota_matched": row.exact_quota_matched,
            }
            for row in quota.targets
        ],
    }


def _frozen_static_shape_projection_v2(
    evidence: _IsolatedAuditEvidenceV2,
) -> dict[str, object]:
    receipt = evidence.audit_receipt
    resource = _static_resource_plan_projection_v2(receipt.resource_plan)
    quota = _eligibility_quota_projection_v2(receipt)
    capacity = _closed_feature_capacity_projection_v2(
        evidence.mode,
        receipt.audited_row_count,
        evidence.feature_capacity_projection,
    )
    if capacity["projection_root"] != evidence.feature_capacity_projection_root:
        raise ValueError("static feature-capacity projection root drift")
    return {
        "schema_version": (
            "hegel-machine-phase2b-formal-wire-covert-static-shape/3"
        ),
        "mode": evidence.mode,
        "rows": receipt.audited_row_count,
        "joint_decision_classes": 2,
        "fixture_specific_only": True,
        "run_fixture_root": evidence.run_fixture_root,
        "source_pair_delta_root": evidence.source_pair_delta_root,
        "transform_catalog_root": evidence.transform_catalog_root,
        "namespace_row_proof_root": evidence.namespace_row_proof_root,
        "build_success_id": evidence.build_success_id,
        "batch_id": evidence.batch_id,
        "operation_counts": dict(evidence.operation_counts),
        "label_factorial_proof_root": evidence.label_factorial_proof_root,
        "label_factorial_summary": {
            "binary_factor_count": 6,
            "joint_decision_class_levels": ["joint-a", "joint-b"],
            "label_cells": 64,
            "rows_per_label_cell": 16,
            "operations_per_label_cell": 8,
            "rows_per_operation_per_label_cell": 2,
        },
        "feature_manifest_root": receipt.feature_manifest.manifest_root,
        "feature_capacity_projection": capacity,
        "feature_capacity_projection_root": (
            evidence.feature_capacity_projection_root
        ),
        "concrete_features": receipt.feature_manifest.deduplicated_feature_count,
        "raw_semantic_features": receipt.feature_manifest.raw_semantic_feature_count,
        "uuid_leaf_count": receipt.feature_manifest.uuid_leaf_count,
        "channel_leaf_count": receipt.feature_manifest.channel_leaf_count,
        "channel_only_owned_leaf_count": (
            evidence.exact_owned_channel_only_leaf_paths
        ),
        "channel_only_owned_leaf_union_root": (
            evidence.channel_only_owner_union_root
        ),
        "full_raw_byte_categorical_hypotheses": (
            evidence.full_raw_byte_categorical_hypotheses
        ),
        "eligibility_quota": quota,
        "eligibility_quota_root": stable_hash(
            quota,
            prefix="phase2b_formal_wire_covert_eligibility_quota_v2_",
        ),
        "resource_plan_static": resource,
        "resource_plan_static_root": stable_hash(
            resource,
            prefix="phase2b_formal_wire_covert_static_resource_plan_v2_",
        ),
    }


def _measured_shape_projection_v2(
    evidence: _IsolatedAuditEvidenceV2,
) -> dict[str, object]:
    receipt = evidence.audit_receipt
    observed_receipt_id = _require_auditor_receipt_id_v2(
        evidence.validated_receipt_id,
        "measured shape observed receipt ID",
    )
    if receipt.receipt_id != observed_receipt_id:
        raise ValueError("measured shape receipt ID drifted")
    resource = _closed_primitive_v2(receipt.resource_plan)
    issuance = _closed_primitive_v2(receipt.receipt_issuance_runtime_evidence)
    statistics = _closed_primitive_v2(receipt.statistics_runtime_evidence)
    _validate_complete_receipt_issuance_runtime_projection_v2(
        issuance,
        statistics,
        context="measured shape",
    )
    frozen = _frozen_static_shape_projection_v2(evidence)
    return {
        "mode": evidence.mode,
        "frozen_static_projection": frozen,
        "frozen_static_projection_root": stable_hash(
            frozen,
            prefix="phase2b_formal_wire_covert_static_shape_v3_",
        ),
        "observed_complete_resource_plan": resource,
        "observed_complete_resource_plan_root": stable_hash(
            resource,
            prefix="phase2b_formal_wire_covert_observed_resource_plan_v3_",
        ),
        "observed_receipt_id": observed_receipt_id,
        "observed_complete_receipt_issuance_runtime": issuance,
        "observed_complete_receipt_issuance_runtime_root": stable_hash(
            issuance,
            prefix="phase2b_formal_wire_covert_receipt_issuance_runtime_v4_",
        ),
        "dynamic_fields_are_recorded_but_not_exact_compared": list(
            DYNAMIC_RESOURCE_PLAN_FIELDS_NOT_EXACT_COMPARED_V2
        ),
        "diagnostics": _diagnostic_projection(evidence.diagnostics),
    }


def _require_pending_calibration_profile_v2(shape: object) -> dict[str, object]:
    if type(shape) is not dict:
        _fail(FAIL_PROFILE, "rich shape calibration profile is not an exact object")
    if (
        shape.get("calibration_campaign_status")
        != "PENDING_FULL_1024_DEVELOPMENT_MEASUREMENT"
        or shape.get("live_qualification_enabled") is not False
        or shape.get("calibration_artifact_cannot_be_promoted") is not True
        or shape.get("fresh_qualification_must_equal_frozen_calibration")
        is not True
    ):
        _fail(FAIL_PROFILE, "development calibration entry needs the pending profile")
    pending_fields = (
        "calibration_artifact_cas_root",
        "calibration_artifact_sha256",
        "calibration_artifact_size_bytes",
        "calibration_attempt_id",
        "calibration_attempt_manifest_cas_root",
        "calibration_attempt_manifest_root",
        "calibration_attempt_manifest_sha256",
        "calibration_attempt_manifest_size_bytes",
        "calibration_basis_profile_cas_root",
        "calibration_basis_profile_id",
        "calibration_basis_profile_sha256",
        "calibration_basis_profile_size_bytes",
        "calibration_campaign_id",
        "calibration_mechanics_artifact_cas_root",
        "calibration_mechanics_artifact_sha256",
        "calibration_mechanics_artifact_size_bytes",
        "calibration_outer_supervisor_result_cas_root",
        "calibration_outer_supervisor_result_sha256",
        "calibration_outer_supervisor_result_size_bytes",
        "calibration_rfc8785_receipt_cas_root",
        "calibration_rfc8785_receipt_sha256",
        "calibration_rfc8785_receipt_size_bytes",
        "combined_fixture_root",
        "known_leak_frozen_static_projection",
        "known_null_frozen_static_projection",
    )
    if any(shape.get(name) is not None for name in pending_fields):
        _fail(FAIL_PROFILE, "pending calibration profile contains observed results")
    return shape


def _verify_frozen_shape_v2(
    shape: object,
    positive: _IsolatedAuditEvidenceV2,
    negative: _IsolatedAuditEvidenceV2,
    *,
    attempt: Phase2BFormalWireCovertAttemptManifestV2 | None,
) -> None:
    if type(shape) is not dict:
        _fail(FAIL_PROFILE, "frozen rich-shape profile is not an exact object")
    if (
        shape.get("calibration_campaign_status")
        != "FROZEN_FULL_1024_DEVELOPMENT_MEASUREMENT"
        or shape.get("live_qualification_enabled") is not True
        or shape.get("calibration_artifact_cannot_be_promoted") is not True
    ):
        _fail(FAIL_PROFILE, "full qualification lacks an independently frozen calibration")
    calibration_campaign = shape.get("calibration_campaign_id")
    calibration_attempt = shape.get("calibration_attempt_id")
    calibration_sha256 = shape.get("calibration_artifact_sha256")
    calibration_cas = shape.get("calibration_artifact_cas_root")
    calibration_size = shape.get("calibration_artifact_size_bytes")
    calibration_attempt_manifest_root = shape.get(
        "calibration_attempt_manifest_root"
    )
    calibration_attempt_manifest_sha256 = shape.get(
        "calibration_attempt_manifest_sha256"
    )
    calibration_attempt_manifest_cas = shape.get(
        "calibration_attempt_manifest_cas_root"
    )
    calibration_attempt_manifest_size = shape.get(
        "calibration_attempt_manifest_size_bytes"
    )
    calibration_basis_profile_id = shape.get("calibration_basis_profile_id")
    calibration_basis_profile_sha256 = shape.get(
        "calibration_basis_profile_sha256"
    )
    calibration_basis_profile_cas = shape.get(
        "calibration_basis_profile_cas_root"
    )
    calibration_basis_profile_size = shape.get(
        "calibration_basis_profile_size_bytes"
    )
    calibration_mechanics_sha256 = shape.get(
        "calibration_mechanics_artifact_sha256"
    )
    calibration_mechanics_cas = shape.get(
        "calibration_mechanics_artifact_cas_root"
    )
    calibration_mechanics_size = shape.get(
        "calibration_mechanics_artifact_size_bytes"
    )
    calibration_outer_sha256 = shape.get(
        "calibration_outer_supervisor_result_sha256"
    )
    calibration_outer_cas = shape.get(
        "calibration_outer_supervisor_result_cas_root"
    )
    calibration_outer_size = shape.get(
        "calibration_outer_supervisor_result_size_bytes"
    )
    calibration_rfc_sha256 = shape.get("calibration_rfc8785_receipt_sha256")
    calibration_rfc_cas = shape.get("calibration_rfc8785_receipt_cas_root")
    calibration_rfc_size = shape.get("calibration_rfc8785_receipt_size_bytes")
    if (
        type(calibration_campaign) is not str
        or not calibration_campaign
        or type(calibration_attempt) is not str
        or not calibration_attempt
        or type(calibration_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", calibration_sha256) is None
        or type(calibration_cas) is not str
        or calibration_cas != "sha256:" + calibration_sha256
        or type(calibration_size) is not int
        or not 1 <= calibration_size <= 8 * 1_024 * 1_024
        or type(calibration_attempt_manifest_root) is not str
        or re.fullmatch(
            r"phase2b_formal_wire_covert_attempt_manifest_v3_[0-9a-f]{64}",
            calibration_attempt_manifest_root,
        )
        is None
        or type(calibration_attempt_manifest_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", calibration_attempt_manifest_sha256)
        is None
        or calibration_attempt_manifest_cas
        != "sha256:" + calibration_attempt_manifest_sha256
        or type(calibration_attempt_manifest_size) is not int
        or not 1 <= calibration_attempt_manifest_size <= 256 * 1_024
        or type(calibration_basis_profile_id) is not str
        or re.fullmatch(
            r"phase2b_formal_wire_covert_qualification_profile_v2_[0-9a-f]{64}",
            calibration_basis_profile_id,
        )
        is None
        or calibration_basis_profile_id == PROFILE_ID
        or type(calibration_basis_profile_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", calibration_basis_profile_sha256)
        is None
        or calibration_basis_profile_cas
        != "sha256:" + calibration_basis_profile_sha256
        or type(calibration_basis_profile_size) is not int
        or not 1 <= calibration_basis_profile_size <= _MAXIMUM_PROFILE_BYTES
        or type(calibration_mechanics_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", calibration_mechanics_sha256) is None
        or type(calibration_mechanics_cas) is not str
        or calibration_mechanics_cas != "sha256:" + calibration_mechanics_sha256
        or type(calibration_mechanics_size) is not int
        or not 1 <= calibration_mechanics_size <= 8 * 1_024 * 1_024
        or type(calibration_outer_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", calibration_outer_sha256) is None
        or calibration_outer_cas != "sha256:" + calibration_outer_sha256
        or type(calibration_outer_size) is not int
        or not 1 <= calibration_outer_size <= 1_048_576
        or type(calibration_rfc_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", calibration_rfc_sha256) is None
        or calibration_rfc_cas != "sha256:" + calibration_rfc_sha256
        or type(calibration_rfc_size) is not int
        or not 1 <= calibration_rfc_size <= 1_048_576
    ):
        _fail(FAIL_PROFILE, "frozen calibration identity/CAS binding is incomplete")
    if attempt is not None and (
        attempt.campaign_id == calibration_campaign
        or attempt.attempt_id == calibration_attempt
    ):
        _fail(FAIL_PROFILE, "qualification must use a fresh campaign and attempt ID")
    actual = {
        "known_null_frozen_static_projection": (
            _frozen_static_shape_projection_v2(positive)
        ),
        "known_leak_frozen_static_projection": (
            _frozen_static_shape_projection_v2(negative)
        ),
        "combined_fixture_root": _combined_fixture_root_v2(positive, negative),
    }
    if any(
        not _exact_json_equal(actual[name], shape.get(name))
        for name in actual
    ):
        _fail(FAIL_PROFILE, "fresh fixture-derived shape differs from calibration")


def _validate_and_release_completed_audits_v2(
    evidences: tuple[_IsolatedAuditEvidenceV2, _IsolatedAuditEvidenceV2],
) -> None:
    for evidence in evidences:
        if evidence.audit_receipt.receipt_id != evidence.validated_receipt_id:
            raise ValueError("inner validated receipt ID changed before release")
        released_id = auditor.release_formal_covert_audit_receipt_v1(
            evidence.audit_receipt
        )
        if released_id != evidence.validated_receipt_id:
            raise ValueError("inner receipt release returned another content ID")


def _canonical_inner_artifact_v2(body: dict[str, object]) -> bytes:
    payload = canonicalize_rfc8785_v1(body)
    if len(payload) > 8 * 1_024 * 1_024:
        raise ValueError("inner primitive artifact exceeds eight MiB")
    replay = _restore_json_integers(
        decode_and_require_canonical_rfc8785_v1(payload)
    )
    if not _exact_json_equal(replay, body):
        raise ValueError("inner primitive artifact canonical replay drift")
    return payload + b"\n"


def _sha256_file_v2(path: Path, *, deadline_ns: int | None = None) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("live hash reached its absolute deadline")
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= 2**31:
            raise ValueError("live hash target is not a bounded regular file")
        descriptor_realpath = os.path.realpath(f"/proc/self/fd/{descriptor}")
        if descriptor_realpath != str(path):
            raise ValueError("live hash descriptor realpath differs from its path")
        total = 0
        while True:
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("live hash reached its absolute deadline")
            block = os.read(descriptor, 1_048_576)
            if not block:
                break
            total += len(block)
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            total != before.st_size
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise ValueError("live hash target changed while it was read")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _read_bounded_regular_file_v2(
    path: Path,
    *,
    maximum_bytes: int,
    name: str,
    deadline_ns: int | None = None,
) -> bytes:
    if not path.is_absolute():
        raise ValueError(f"{name} path must be absolute")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            raise TimeoutError(f"{name} read reached its absolute deadline")
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not 1 <= before.st_size <= maximum_bytes
        ):
            raise ValueError(f"{name} is not a bounded exact regular file")
        descriptor_realpath = os.path.realpath(f"/proc/self/fd/{descriptor}")
        if descriptor_realpath != str(path):
            raise ValueError(f"{name} descriptor realpath differs from its path")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                raise TimeoutError(f"{name} read reached its absolute deadline")
            block = os.read(descriptor, min(remaining, 65_536))
            if not block:
                raise ValueError(f"{name} ended before its fstat size")
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise ValueError(f"{name} grew beyond its fstat size")
        after = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise ValueError(f"{name} changed while it was read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _require_raw_sha256_v2(value: object, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be an exact raw SHA-256")
    return value


def _require_auditor_receipt_id_v2(value: object, name: str) -> str:
    if (
        type(value) is not str
        or re.fullmatch(
            r"phase2b_formal_covert_audit_transcript_[0-9a-f]{64}",
            value,
        )
        is None
    ):
        raise ValueError(f"{name} must be an exact Auditor receipt ID")
    return value


def _require_eligibility_quota_receipt_id_v2(
    value: object,
    name: str,
) -> str:
    if (
        type(value) is not str
        or re.fullmatch(
            r"phase2b_formal_eligibility_mask_quota_[0-9a-f]{64}",
            value,
        )
        is None
    ):
        raise ValueError(
            f"{name} must be an exact eligibility-mask quota receipt ID"
        )
    return value


def _require_exact_int_v2(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} is outside its exact integer bounds")
    return value


def _require_safe_cgroup_path_v2(value: object, name: str) -> str:
    if (
        type(value) is not str
        or value == "/"
        or not value.startswith("/")
        or value.endswith("/")
        or "//" in value
        or "\x00" in value
        or re.fullmatch(r"/[A-Za-z0-9_.@:-]+(?:/[A-Za-z0-9_.@:-]+)*", value)
        is None
        or any(part in {".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"{name} is not a safe absolute cgroup path")
    return value


def _validate_zero_cgroup_events_v2(
    row: object,
    name: str,
    *,
    required_keys: frozenset[str],
) -> dict[str, int]:
    if type(row) is not dict or not row:
        raise ValueError(f"{name} cgroup events are absent")
    if not required_keys.issubset(row):
        raise ValueError(f"{name} cgroup events are missing required counters")
    for key, value in row.items():
        if (
            type(key) is not str
            or re.fullmatch(r"[a-z0-9_.-]+", key) is None
            or _require_exact_int_v2(value, f"{name}.{key}") != 0
        ):
            raise ValueError(f"{name} cgroup event counters are not exact zeroes")
    return row


def _executable_path_security_projection_v2(path: Path) -> dict[str, object]:
    if os.geteuid() == 0:
        raise ValueError("qualification executable checks cannot run as root")
    components = [path, *path.parents]
    checked: list[dict[str, object]] = []
    for index, component in enumerate(components):
        opened = component.stat(follow_symlinks=False)
        expected_type = stat.S_ISREG if index == 0 else stat.S_ISDIR
        if (
            not expected_type(opened.st_mode)
            or opened.st_uid != 0
            or opened.st_mode & 0o022
            or (index == 0 and not opened.st_mode & 0o111)
        ):
            raise ValueError(
                "executable or ancestor is not root-owned and replacement-resistant"
            )
        checked.append(
            {
                "mode_octal": format(stat.S_IMODE(opened.st_mode), "04o"),
                "owner_uid": opened.st_uid,
                "realpath": str(component),
                "type": "regular" if index == 0 else "directory",
            }
        )
    return {
        "caller_effective_uid": os.geteuid(),
        "checked_components_in_order": checked,
        "root_owned_no_group_or_other_write": True,
        "same_uid_path_replacement_excluded": True,
    }


def _verify_executable_projection_shape_v2(
    row: object, name: str
) -> dict[str, object]:
    if type(row) is not dict or set(row) != {
        "execution_path_security", "realpath", "sha256", "version"
    }:
        raise ValueError(f"{name} executable projection is not closed")
    realpath = row["realpath"]
    digest = _require_raw_sha256_v2(row["sha256"], f"{name} executable")
    version = row["version"]
    if (
        type(realpath) is not str
        or not Path(realpath).is_absolute()
        or type(version) is not str
        or not version
    ):
        raise ValueError(f"{name} executable identity is malformed")
    security = row["execution_path_security"]
    if (
        type(security) is not dict
        or set(security)
        != {
            "caller_effective_uid",
            "checked_components_in_order",
            "root_owned_no_group_or_other_write",
            "same_uid_path_replacement_excluded",
        }
        or type(security["caller_effective_uid"]) is not int
        or type(security["checked_components_in_order"]) is not list
        or not security["checked_components_in_order"]
        or security["root_owned_no_group_or_other_write"] is not True
        or security["same_uid_path_replacement_excluded"] is not True
    ):
        raise ValueError(f"{name} executable path-security receipt is malformed")
    return row


def _verify_live_executable_v2(
    row: object,
    name: str,
    *,
    deadline_ns: int | None = None,
) -> dict[str, object]:
    row = _verify_executable_projection_shape_v2(row, name)
    if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
        raise TimeoutError(f"{name} executable replay reached its deadline")
    realpath = row["realpath"]
    digest = row["sha256"]
    path = Path(realpath)
    if str(path.resolve(strict=True)) != realpath:
        raise ValueError(f"{name} executable realpath is not live")
    if _sha256_file_v2(path, deadline_ns=deadline_ns) != digest:
        raise ValueError(f"{name} executable live hash differs")
    if not _exact_json_equal(
        row["execution_path_security"],
        _executable_path_security_projection_v2(path),
    ):
        raise ValueError(f"{name} executable path-security projection drifted")
    return row


def _verify_rfc8785_prerequisite_receipt_v2(
    payload: bytes,
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
    *,
    verify_live_executables: bool = True,
    deadline_ns: int | None = None,
) -> dict[str, object]:
    if type(verify_live_executables) is not bool:
        raise TypeError("RFC executable verification switch must be exact bool")
    if (
        len(payload) != attempt.rfc8785_prerequisite_receipt_size_bytes
        or hashlib.sha256(payload).hexdigest()
        != attempt.rfc8785_prerequisite_receipt_sha256
    ):
        raise ValueError("RFC-8785 receipt bytes differ from the attempt binding")
    decoded = _decode_canonical_transport_object_v2(
        payload, "RFC-8785 prerequisite receipt", maximum_bytes=1_048_576
    )
    if set(decoded) != {
        "cas",
        "dependency_sha256",
        "differential",
        "disposition",
        "executables",
        "node_oracle",
        "source_only_kat_18",
        "raw_execution",
        "schema_version",
    }:
        raise ValueError("RFC-8785 prerequisite receipt fields are not exact")
    dependencies = decoded["dependency_sha256"]
    if dependencies != {
        "documentation": RFC8785_DOCUMENTATION_SHA256,
        "golden_vectors": RFC8785_GOLDEN_VECTORS_SHA256,
        "node_oracle": RFC8785_NODE_ORACLE_SHA256,
        "source": RFC8785_SOURCE_SHA256,
        "test": RFC8785_TEST_SHA256,
    }:
        raise ValueError("RFC-8785 prerequisite dependency hashes drifted")
    if (
        decoded["schema_version"] != RFC8785_PREREQUISITE_SCHEMA_V2
        or decoded["disposition"] != "PASS"
    ):
        raise ValueError("RFC-8785 prerequisite did not issue exact PASS")
    executables = decoded["executables"]
    if type(executables) is not dict or set(executables) != {
        "node", "python", "v8_version"
    }:
        raise ValueError("RFC-8785 executable receipt is not closed")
    if verify_live_executables:
        _verify_live_executable_v2(
            executables["python"], "RFC Python", deadline_ns=deadline_ns
        )
        _verify_live_executable_v2(
            executables["node"], "RFC Node", deadline_ns=deadline_ns
        )
    else:
        _verify_executable_projection_shape_v2(executables["python"], "RFC Python")
        _verify_executable_projection_shape_v2(executables["node"], "RFC Node")
    if type(executables["v8_version"]) is not str or not executables["v8_version"]:
        raise ValueError("RFC-8785 V8 version is absent")
    selfcheck_row = decoded["source_only_kat_18"]
    if (
        type(selfcheck_row) is not dict
        or set(selfcheck_row)
        != {
            "collected",
            "exit_code",
            "passed",
            "runner",
            "stderr_sha256",
            "stdout_sha256",
        }
    ):
        raise ValueError("RFC-8785 stdlib selfcheck fields are not exact")
    if (
        selfcheck_row["runner"] != "STDLIB_SOURCE_ONLY_PINNED_TEST_HARNESS_V1"
        or _require_exact_int_v2(
            selfcheck_row["collected"], "RFC selfcheck collected"
        )
        != 18
        or _require_exact_int_v2(
            selfcheck_row["passed"], "RFC selfcheck passed"
        )
        != 18
        or _require_exact_int_v2(
            selfcheck_row["exit_code"], "RFC selfcheck exit"
        )
        != 0
    ):
        raise ValueError("RFC-8785 stdlib selfcheck is not exact 18/18 exit-zero")
    oracle_row = decoded["node_oracle"]
    if (
        type(oracle_row) is not dict
        or set(oracle_row)
        != {
            "disposition",
            "exit_code",
            "number_cases",
            "rejection_cases",
            "serialization_cases",
            "stderr_sha256",
            "stdout_sha256",
        }
        or oracle_row["disposition"] != "PASS"
    ):
        raise ValueError("RFC-8785 Node oracle fields are not exact")
    if (
        _require_exact_int_v2(oracle_row["exit_code"], "RFC oracle exit") != 0
        or _require_exact_int_v2(
            oracle_row["serialization_cases"], "RFC serialization cases"
        )
        != 7
        or _require_exact_int_v2(
            oracle_row["number_cases"], "RFC oracle number cases"
        )
        != 24
        or _require_exact_int_v2(
            oracle_row["rejection_cases"], "RFC oracle rejection cases"
        )
        != 2
    ):
        raise ValueError("RFC-8785 Node oracle evidence drifted")
    differential = decoded["differential"]
    if (
        type(differential) is not dict
        or set(differential)
        != {
            "batch_size",
            "checked",
            "exit_code",
            "finite_binary64_cases",
            "mismatch_count",
            "seed_hex",
            "stderr_sha256",
            "stdout_sha256",
        }
        or differential["seed_hex"] != "0x878520260814"
    ):
        raise ValueError("RFC-8785 differential fields are not exact")
    if (
        _require_exact_int_v2(
            differential["finite_binary64_cases"], "RFC finite cases"
        )
        != 100_000
        or _require_exact_int_v2(
            differential["batch_size"], "RFC differential batch size"
        )
        != 2_000
        or _require_exact_int_v2(
            differential["checked"], "RFC differential checked"
        )
        != 100_000
        or _require_exact_int_v2(
            differential["mismatch_count"], "RFC differential mismatches"
        )
        != 0
        or _require_exact_int_v2(
            differential["exit_code"], "RFC differential exit"
        )
        != 0
    ):
        raise ValueError("RFC-8785 100000-case differential evidence drifted")
    for group, names in (
        (selfcheck_row, ("stdout_sha256", "stderr_sha256")),
        (oracle_row, ("stdout_sha256", "stderr_sha256")),
        (differential, ("stdout_sha256", "stderr_sha256")),
    ):
        for name in names:
            _require_raw_sha256_v2(group[name], f"RFC-8785 {name}")
    raw_execution = decoded["raw_execution"]
    if (
        type(raw_execution) is not dict
        or set(raw_execution)
        != {
            "bundle_manifest",
            "command_count",
            "commands",
            "node_oracle_execution",
            "source_only_kat_version",
            "qualification_source_sha256",
            "raw_artifact_total_bytes",
            "schema_version",
            "transcript_root",
            "working_tree_root",
        }
        or raw_execution["schema_version"]
        != "hegel-machine-rfc8785-raw-execution-bundle/2"
        or type(raw_execution["commands"]) is not list
        or _require_exact_int_v2(
            raw_execution["command_count"], "RFC raw command count", minimum=56,
            maximum=56,
        )
        != len(raw_execution["commands"])
        or type(raw_execution["bundle_manifest"]) is not list
        or len(raw_execution["bundle_manifest"]) != 56 * 3
        or raw_execution["source_only_kat_version"]
        != "hegel-rfc8785-stdlib-selfcheck-v1"
        or type(raw_execution["working_tree_root"]) is not str
        or not Path(raw_execution["working_tree_root"]).is_absolute()
        or type(raw_execution["transcript_root"]) is not str
        or re.fullmatch(
            r"phase2b_rfc8785_raw_command_transcript_v1_[0-9a-f]{64}",
            raw_execution["transcript_root"],
        )
        is None
        or _require_raw_sha256_v2(
            raw_execution["qualification_source_sha256"],
            "RFC qualification bootstrap source",
        )
        != attempt.qualification_source_sha256
    ):
        raise ValueError("RFC-8785 raw execution projection is not closed")
    node_execution = raw_execution["node_oracle_execution"]
    if (
        type(node_execution) is not dict
        or node_execution
        != {
            "golden_vectors_sha256": RFC8785_GOLDEN_VECTORS_SHA256,
            "golden_vectors_size_bytes": 4_982,
            "oracle_source_sha256": RFC8785_NODE_ORACLE_SHA256,
            "oracle_source_size_bytes": 5_236,
            "source_transport": "NODE_ESM_EVAL_ARGV",
            "vectors_transport": "SEALED_MEMFD_PROC_FD",
        }
    ):
        raise ValueError("RFC-8785 Node open-once execution projection drifted")
    if raw_execution["transcript_root"] != stable_hash(
        raw_execution["commands"],
        prefix="phase2b_rfc8785_raw_command_transcript_v1_",
    ):
        raise ValueError("RFC-8785 raw command transcript root drifted")
    cas = decoded["cas"]
    if (
        type(cas) is not dict
        or set(cas)
        != {
            "content_sha256",
            "object_key",
            "object_size_bytes",
            "prior_object_count",
            "put_exit_code",
            "replay_verified",
            "resulting_object_count",
        }
        or cas["replay_verified"] is not True
    ):
        raise ValueError("RFC-8785 execution CAS receipt is incomplete")
    if (
        _require_exact_int_v2(cas["prior_object_count"], "RFC CAS prior count")
        != 0
        or _require_exact_int_v2(
            cas["resulting_object_count"], "RFC CAS resulting count"
        )
        != 1
        or _require_exact_int_v2(cas["put_exit_code"], "RFC CAS put exit")
        != 0
    ):
        raise ValueError("RFC-8785 execution CAS counter evidence drifted")
    cas_sha = _require_raw_sha256_v2(cas["content_sha256"], "RFC CAS content")
    evidence_body = {
        key: decoded[key] for key in sorted(decoded) if key != "cas"
    }
    evidence_bytes = canonicalize_rfc8785_v1(evidence_body)
    evidence_sha = hashlib.sha256(evidence_bytes).hexdigest()
    if (
        cas_sha != evidence_sha
        or cas["object_key"] != "sha256:" + evidence_sha
        or _require_exact_int_v2(
            cas["object_size_bytes"], "RFC CAS object size", minimum=1
        )
        != len(evidence_bytes)
    ):
        raise ValueError("RFC-8785 execution CAS does not bind its evidence body")
    return decoded


def _verify_rfc8785_raw_execution_bundle_v2(
    receipt: dict[str, object],
    bundle_directory: Path,
    *,
    expected_qualification_source_sha256: str,
    replay_project_root: Path | None = None,
    fresh_execute: bool = False,
    deadline_ns: int | None = None,
    retained_bundle_fd: int | None = None,
    replay_project_root_fd: int | None = None,
) -> dict[str, object]:
    if type(fresh_execute) is not bool:
        raise TypeError("RFC fresh execution switch must be exact bool")

    def check_deadline(stage: str) -> None:
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            raise TimeoutError(f"RFC raw {stage} reached the absolute deadline")

    check_deadline("entry")
    if not isinstance(bundle_directory, Path) or not bundle_directory.is_absolute():
        raise ValueError("RFC raw execution bundle display path is not absolute")
    if retained_bundle_fd is None:
        if (
            bundle_directory.is_symlink()
            or not bundle_directory.is_dir()
            or str(bundle_directory.resolve(strict=True)) != str(bundle_directory)
        ):
            raise ValueError(
                "RFC raw execution bundle is not an absolute real directory"
            )
    elif not stat.S_ISDIR(os.fstat(retained_bundle_fd).st_mode):
        raise ValueError("RFC raw retained bundle descriptor is not a directory")
    raw = receipt["raw_execution"]
    if type(raw) is not dict:
        raise ValueError("RFC raw execution bundle projection is absent")
    if (
        _require_raw_sha256_v2(
            expected_qualification_source_sha256,
            "expected RFC qualification bootstrap",
        )
        != raw.get("qualification_source_sha256")
    ):
        raise ValueError("RFC bootstrap does not bind the current attempt source")
    commands = raw["commands"]
    manifest = raw["bundle_manifest"]
    if type(commands) is not list or type(manifest) is not list:
        raise ValueError("RFC raw execution arrays are absent")
    bundle_fd = (
        _open_real_directory_fd_v2(bundle_directory)
        if retained_bundle_fd is None
        else os.dup(retained_bundle_fd)
    )
    material: dict[str, bytes] = {}
    total = 0
    try:
        for index, reference in enumerate(manifest):
            check_deadline("artifact replay")
            if type(reference) is not dict or set(reference) != {
                "path", "sha256", "size_bytes"
            }:
                raise ValueError("RFC raw artifact reference is not closed")
            name = reference["path"]
            size = _require_exact_int_v2(
                reference["size_bytes"],
                f"RFC raw artifact {index} size",
                maximum=2 * 1_024 * 1_024,
            )
            digest = _require_raw_sha256_v2(
                reference["sha256"], f"RFC raw artifact {index}"
            )
            if (
                type(name) is not str
                or re.fullmatch(r"command-[0-9]{3}-[a-z0-9-]+\.(?:argv\.json|stdout\.bin|stderr\.bin)", name)
                is None
                or name in material
            ):
                raise ValueError("RFC raw artifact name is unsafe or duplicated")
            payload, opened = _read_regular_at_fd_v2(
                bundle_fd,
                name,
                maximum_bytes=max(1, size),
                deadline_ns=deadline_ns,
            )
            if opened.st_size != size or hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError("RFC raw artifact bytes drifted")
            total += size
            if total > 64 * 1_024 * 1_024:
                raise ValueError("RFC raw artifact aggregate exceeds 64 MiB")
            material[name] = payload
        evidence_body = {
            key: receipt[key] for key in sorted(receipt) if key != "cas"
        }
        evidence_bytes = canonicalize_rfc8785_v1(evidence_body)
        evidence_sha = hashlib.sha256(evidence_bytes).hexdigest()
        retained_body, _ = _read_regular_at_fd_v2(
            bundle_fd,
            evidence_sha,
            maximum_bytes=max(1, len(evidence_bytes)),
            deadline_ns=deadline_ns,
        )
        if retained_body != evidence_bytes:
            raise ValueError("RFC raw bundle evidence-body CAS object drifted")
    finally:
        _close_owned_descriptors_v2((bundle_fd,), name="RFC raw artifact bundle")
    if (
        _require_exact_int_v2(
            raw["raw_artifact_total_bytes"], "RFC raw aggregate bytes"
        )
        != total
        or manifest != sorted(manifest, key=lambda row: row["path"])
    ):
        raise ValueError("RFC raw artifact manifest summary drifted")
    referenced: list[dict[str, object]] = []
    parsed_commands: list[tuple[dict[str, object], list[str], bytes, bytes]] = []
    expected_environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": str(Path(receipt["executables"]["node"]["realpath"]).parent),
        "TZ": "UTC",
    }
    previous_end: int | None = None
    for index, command in enumerate(commands):
        check_deadline("command parsing")
        if type(command) is not dict or set(command) != {
            "argv_artifact",
            "cwd",
            "elapsed_nanoseconds",
            "ended_monotonic_nanoseconds",
            "environment_overrides",
            "name",
            "returncode",
            "started_monotonic_nanoseconds",
            "stderr_artifact",
            "stdout_artifact",
        }:
            raise ValueError("RFC raw command row is not closed")
        for key in ("argv_artifact", "stdout_artifact", "stderr_artifact"):
            reference = command[key]
            if type(reference) is not dict:
                raise ValueError("RFC raw command artifact reference is absent")
            referenced.append(reference)
        start = _require_exact_int_v2(
            command["started_monotonic_nanoseconds"], "RFC command start"
        )
        end = _require_exact_int_v2(
            command["ended_monotonic_nanoseconds"], "RFC command end"
        )
        elapsed = _require_exact_int_v2(
            command["elapsed_nanoseconds"], "RFC command elapsed"
        )
        if (
            end < start
            or elapsed != end - start
            or (previous_end is not None and start < previous_end)
            or command["cwd"] != raw["working_tree_root"]
            or not _exact_json_equal(command["environment_overrides"], expected_environment)
            or _require_exact_int_v2(command["returncode"], "RFC command exit") != 0
        ):
            raise ValueError("RFC raw command timing/environment drifted")
        previous_end = end
        argv_ref = command["argv_artifact"]
        stdout_ref = command["stdout_artifact"]
        stderr_ref = command["stderr_artifact"]
        argv_payload = material[argv_ref["path"]]
        if not argv_payload.endswith(b"\n"):
            raise ValueError("RFC raw argv artifact lacks its LF frame")
        argv_decoded = _restore_json_integers(
            decode_and_require_canonical_rfc8785_v1(argv_payload[:-1])
        )
        if (
            type(argv_decoded) is not list
            or any(type(item) is not str or "\x00" in item for item in argv_decoded)
            or canonicalize_rfc8785_v1(argv_decoded) + b"\n" != argv_payload
        ):
            raise ValueError("RFC raw argv artifact is not canonical string argv")
        parsed_commands.append(
            (
                command,
                argv_decoded,
                material[stdout_ref["path"]],
                material[stderr_ref["path"]],
            )
        )
    if sorted(referenced, key=lambda row: row["path"]) != manifest:
        raise ValueError("RFC raw command references do not equal the manifest")
    python_row = receipt["executables"]["python"]
    node_row = receipt["executables"]["node"]
    recorded_root = Path(raw["working_tree_root"])
    if not recorded_root.is_absolute():
        raise ValueError("RFC raw recorded working tree is not absolute")
    root = recorded_root if replay_project_root is None else replay_project_root
    if not isinstance(root, Path) or not root.is_absolute():
        raise ValueError("RFC replay project root is not an absolute Path")
    dependency_paths = {
        "docs/Hegel_Machine_Phase2B_RFC8785_Implementation_v1.md": (
            RFC8785_DOCUMENTATION_SHA256
        ),
        "golden_vectors/phase2b_rfc8785_v1.json": RFC8785_GOLDEN_VECTORS_SHA256,
        "src/hegel_machine/rfc8785_v1.py": RFC8785_SOURCE_SHA256,
        "tests/test_rfc8785_v1.py": RFC8785_TEST_SHA256,
        "tools/phase2b_rfc8785_oracle_v1.mjs": RFC8785_NODE_ORACLE_SHA256,
        **_BOOTSTRAP_FIXED_SOURCE_SHA256_V2,
        _BOOTSTRAP_MAIN_RELATIVE_PATH_V2: expected_qualification_source_sha256,
    }
    root_fd = (
        _open_real_directory_fd_v2(root)
        if replay_project_root_fd is None
        else os.dup(replay_project_root_fd)
    )
    try:
        if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
            raise ValueError("RFC raw retained working tree is not a directory")
        for relative, digest in dependency_paths.items():
            actual, _ = _hash_relative_regular_at_fd_v2(
                root_fd,
                relative,
                maximum_bytes=16 * 1_024 * 1_024,
                deadline_ns=deadline_ns,
            )
            if actual != digest:
                raise ValueError(f"RFC raw dependency drifted: {relative}")
        qualification_source_digest, _ = _hash_relative_regular_at_fd_v2(
            root_fd,
            _BOOTSTRAP_MAIN_RELATIVE_PATH_V2,
            maximum_bytes=16 * 1_024 * 1_024,
            deadline_ns=deadline_ns,
        )
        oracle_source_bytes, _ = _read_relative_regular_at_fd_v2(
            root_fd,
            "tools/phase2b_rfc8785_oracle_v1.mjs",
            maximum_bytes=256 * 1_024,
            deadline_ns=deadline_ns,
        )
        vectors_bytes, _ = _read_relative_regular_at_fd_v2(
            root_fd,
            "golden_vectors/phase2b_rfc8785_v1.json",
            maximum_bytes=256 * 1_024,
            deadline_ns=deadline_ns,
        )
    finally:
        _close_owned_descriptors_v2((root_fd,), name="RFC retained project root")
    expected_names = [
        "python-version",
        "node-version",
        "v8-version",
        "source-only-kat-version",
        "source-only-kat-18",
        "node-oracle",
    ] + [
        f"differential-{index:03d}" for index in range(50)
    ]
    if [item[0]["name"] for item in parsed_commands] != expected_names:
        raise ValueError("RFC raw command order drifted")
    python_version = parsed_commands[0]
    node_version = parsed_commands[1]
    retained_v8 = parsed_commands[2]
    if (
        python_version[1] != [python_row["realpath"], "--version"]
        or (python_version[2] + python_version[3]).decode("utf-8").strip()
        != python_row["version"]
        or node_version[1] != [node_row["realpath"], "--version"]
        or (node_version[2] + node_version[3]).decode("utf-8").strip()
        != node_row["version"]
        or retained_v8[1]
        != [node_row["realpath"], "-p", "process.versions.v8"]
        or retained_v8[3]
        or retained_v8[2].decode("utf-8").strip()
        != receipt["executables"]["v8_version"]
    ):
        raise ValueError("RFC raw executable version probes drifted")
    kat_version_command, kat_version_argv, kat_version_stdout, kat_version_stderr = parsed_commands[3]
    qualification_source_sha256 = raw["qualification_source_sha256"]
    expected_kat_version_argv = _source_execution_bootstrap_argv_v2(
        python_realpath=python_row["realpath"],
        project_root=recorded_root,
        qualification_source_sha256=qualification_source_sha256,
        source_arguments=[
            "--entry",
            "rfc-selfcheck-version",
            "--project-root",
            str(recorded_root),
        ],
    )
    if (
        kat_version_argv
        != expected_kat_version_argv
        or kat_version_stderr
        or kat_version_stdout.decode("ascii").strip()
        != raw["source_only_kat_version"]
    ):
        raise ValueError("RFC raw source-only KAT version command drifted")
    kat_command, kat_argv, kat_stdout, kat_stderr = parsed_commands[4]
    expected_kat_argv = _source_execution_bootstrap_argv_v2(
        python_realpath=python_row["realpath"],
        project_root=recorded_root,
        qualification_source_sha256=qualification_source_sha256,
        source_arguments=[
            "--entry", "rfc-selfcheck", "--project-root", str(recorded_root)
        ],
    )
    if (
        kat_argv != expected_kat_argv
        or qualification_source_digest != raw["qualification_source_sha256"]
        or kat_stderr
        or re.fullmatch(rb"\.{18}[ ]+\[100%\]\n", kat_stdout) is None
        or hashlib.sha256(kat_stdout).hexdigest()
        != receipt["source_only_kat_18"]["stdout_sha256"]
        or hashlib.sha256(kat_stderr).hexdigest()
        != receipt["source_only_kat_18"]["stderr_sha256"]
    ):
        raise ValueError("RFC raw source-only KAT evidence drifted")
    oracle_command, oracle_argv, oracle_stdout, oracle_stderr = parsed_commands[5]
    node_execution = raw["node_oracle_execution"]
    if (
        type(node_execution) is not dict
        or hashlib.sha256(oracle_source_bytes).hexdigest()
        != node_execution.get("oracle_source_sha256")
        or len(oracle_source_bytes) != node_execution.get("oracle_source_size_bytes")
        or hashlib.sha256(vectors_bytes).hexdigest()
        != node_execution.get("golden_vectors_sha256")
        or len(vectors_bytes) != node_execution.get("golden_vectors_size_bytes")
    ):
        raise ValueError("RFC replay Node open-once input bytes drifted")
    oracle_source_text = oracle_source_bytes.decode("utf-8", errors="strict")
    expected_oracle_prefix = [
        node_row["realpath"],
        "--input-type=module",
        "--eval",
        oracle_source_text,
        "--",
        str(recorded_root / "tools/phase2b_rfc8785_oracle_v1.mjs"),
        "--check",
    ]
    if (
        len(oracle_argv) != len(expected_oracle_prefix) + 1
        or oracle_argv[:-1] != expected_oracle_prefix
        or re.fullmatch(r"/proc/self/fd/[0-9]+", oracle_argv[-1]) is None
        or oracle_stderr
    ):
        raise ValueError("RFC raw Node oracle argv/result drifted")
    oracle_decoded = json.loads(oracle_stdout)
    if (
        oracle_decoded
        != {
            "disposition": "PASS",
            "node": node_row["version"],
            "number_cases": 24,
            "oracle": "node-v8-json-stringify",
            "rejection_cases": 2,
            "serialization_cases": 7,
            "v8": receipt["executables"]["v8_version"],
        }
        or hashlib.sha256(oracle_stdout).hexdigest()
        != receipt["node_oracle"]["stdout_sha256"]
        or hashlib.sha256(oracle_stderr).hexdigest()
        != receipt["node_oracle"]["stderr_sha256"]
        or receipt["node_oracle"]["stderr_sha256"]
        != hashlib.sha256(b"").hexdigest()
        or receipt["differential"]["stderr_sha256"]
        != hashlib.sha256(b"").hexdigest()
    ):
        raise ValueError("RFC raw Node oracle payload drifted")
    rng = random.Random(0x8785_2026_0814)
    finite: list[tuple[str, float]] = []
    while len(finite) < 100_000:
        if len(finite) % 1_024 == 0:
            check_deadline("finite vector generation")
        bits = rng.getrandbits(64)
        value = struct.unpack(">d", bits.to_bytes(8, "big"))[0]
        if math.isfinite(value):
            finite.append((f"{bits:016x}", value))
    differential_rows: list[dict[str, object]] = []
    checked = 0
    for batch_index, parsed in enumerate(parsed_commands[6:]):
        check_deadline("differential batch replay")
        command, argv, stdout, stderr = parsed
        batch = finite[batch_index * 2_000 : (batch_index + 1) * 2_000]
        expected_argv = [
            node_row["realpath"],
            "--input-type=module",
            "--eval",
            oracle_source_text,
            "--",
            str(recorded_root / "tools/phase2b_rfc8785_oracle_v1.mjs"),
            "--number-bits",
            *[bits for bits, _ in batch],
        ]
        if argv != expected_argv or stderr:
            raise ValueError("RFC raw differential argv/result drifted")
        oracle_rows = json.loads(stdout)
        if type(oracle_rows) is not list or len(oracle_rows) != len(batch):
            raise ValueError("RFC raw differential output shape drifted")
        for (hex_bits, value), oracle_row in zip(batch, oracle_rows, strict=True):
            if checked % 1_024 == 0:
                check_deadline("differential row replay")
            if (
                type(oracle_row) is not dict
                or set(oracle_row) != {"canonical", "ieee754_hex"}
                or oracle_row["ieee754_hex"] != hex_bits
                or oracle_row["canonical"]
                != canonicalize_rfc8785_v1(value).decode("utf-8")
            ):
                raise ValueError("RFC raw differential mismatch")
            checked += 1
        differential_rows.append(
            {
                "batch_index": batch_index,
                "case_count": len(batch),
                "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                "stdout_size_bytes": len(stdout),
            }
        )
    if (
        checked != 100_000
        or hashlib.sha256(canonicalize_rfc8785_v1(differential_rows)).hexdigest()
        != receipt["differential"]["stdout_sha256"]
    ):
        raise ValueError("RFC raw differential transcript summary drifted")
    if fresh_execute:
        if deadline_ns is None or type(deadline_ns) is not int:
            raise ValueError("RFC fresh replay requires one absolute deadline")
        projection_deadline = deadline_ns
        fresh_python = _executable_projection_for_runner_v2(
            Path(sys.executable),
            version_argv=("--version",),
            name="fresh RFC Python anchor",
            deadline_ns=projection_deadline,
        )
        node_lookup = shutil.which("node")
        if node_lookup is None:
            raise ValueError("fresh RFC verifier cannot locate Node")
        fresh_node = _executable_projection_for_runner_v2(
            Path(node_lookup),
            version_argv=("--version",),
            name="fresh RFC Node anchor",
            deadline_ns=projection_deadline,
        )
        v8_evidence, v8_stdout, v8_stderr = _run_bounded_control_command_v2(
            [fresh_node["realpath"], "-p", "process.versions.v8"],
            purpose="fresh-rfc-v8-anchor",
            deadline_ns=projection_deadline,
            working_directory=root,
            environment_overrides=expected_environment,
            clean_environment=True,
        )
        if (
            not _exact_json_equal(fresh_python, python_row)
            or not _exact_json_equal(fresh_node, node_row)
            or v8_evidence["returncode"] != 0
            or v8_stderr
            or v8_stdout.decode("utf-8").strip()
            != receipt["executables"]["v8_version"]
        ):
            raise ValueError("RFC executable anchors differ from fresh host binaries")
        recorded_prefix = str(recorded_root) + os.sep
        replay_prefix = str(root) + os.sep
        for index, (command, allowlisted_argv, expected_stdout, expected_stderr) in enumerate(
            parsed_commands
        ):
            check_deadline("fresh command replay")
            rebased_argv = [
                (
                    str(root)
                    if item == str(recorded_root)
                    else replay_prefix + item[len(recorded_prefix):]
                    if item.startswith(recorded_prefix)
                    else item
                )
                for item in allowlisted_argv
            ]
            inherited: tuple[int, ...] = ()
            vectors_fd: int | None = None
            if index == 5:
                vectors_fd = _open_sealed_memfd_v2(
                    vectors_bytes, "hegel-rfc8785-fresh-vectors"
                )
                rebased_argv[-1] = f"/proc/self/fd/{vectors_fd}"
                inherited = (vectors_fd,)
            try:
                evidence, stdout, stderr = _run_bounded_control_command_v2(
                    rebased_argv,
                    purpose=f"rfc-fresh-replay-{index:03d}",
                    deadline_ns=deadline_ns,
                    stdout_cap=max(65_536, len(expected_stdout)),
                    stderr_cap=max(65_536, len(expected_stderr)),
                    working_directory=root,
                    environment_overrides=expected_environment,
                    clean_environment=True,
                    pass_fds=inherited,
                )
            finally:
                if vectors_fd is not None:
                    _close_owned_descriptors_v2(
                        (vectors_fd,), name="RFC fresh replay sealed vectors"
                    )
            if (
                evidence["returncode"] != command["returncode"]
                or stdout != expected_stdout
                or stderr != expected_stderr
            ):
                raise ValueError("RFC fresh command replay differs from retained bytes")
    projection = {
        "bundle_manifest_root": stable_hash(
            manifest, prefix="phase2b_rfc8785_raw_bundle_manifest_v1_"
        ),
        "command_count": len(commands),
        "raw_artifact_total_bytes": total,
        "transcript_root": raw["transcript_root"],
    }
    check_deadline("completion")
    return projection


def _copy_rfc8785_raw_execution_bundle_v2(
    *,
    receipt: dict[str, object],
    receipt_bytes: bytes,
    source_directory: Path,
    destination_directory: Path,
    destination_directory_fd: int,
    deadline_ns: int,
    expected_qualification_source_sha256: str,
    replay_project_root: Path,
) -> tuple[Path, dict[str, object]]:
    projection = _verify_rfc8785_raw_execution_bundle_v2(
        receipt,
        source_directory,
        expected_qualification_source_sha256=expected_qualification_source_sha256,
        replay_project_root=replay_project_root,
        fresh_execute=True,
        deadline_ns=deadline_ns,
    )
    destination = destination_directory
    if not stat.S_ISDIR(os.fstat(destination_directory_fd).st_mode):
        raise ValueError("RFC retained raw bundle descriptor is not a directory")
    raw = receipt["raw_execution"]
    evidence_body = {
        key: receipt[key] for key in sorted(receipt) if key != "cas"
    }
    evidence_bytes = canonicalize_rfc8785_v1(evidence_body)
    evidence_sha = hashlib.sha256(evidence_bytes).hexdigest()
    names_and_caps: list[tuple[str, int]] = [
        (row["path"], max(1, row["size_bytes"]))
        for row in raw["bundle_manifest"]
    ]
    names_and_caps.extend(
        [
            (evidence_sha, max(1, len(evidence_bytes))),
            ("receipt.json", max(1, len(receipt_bytes))),
        ]
    )
    source_fd = _open_real_directory_fd_v2(source_directory)
    try:
        for name, maximum in names_and_caps:
            if time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("RFC raw bundle copy reached the absolute deadline")
            payload, _ = _read_regular_at_fd_v2(
                source_fd,
                name,
                maximum_bytes=maximum,
                deadline_ns=deadline_ns,
            )
            if name == evidence_sha and payload != evidence_bytes:
                raise ValueError("RFC evidence-body CAS changed before copy")
            if name == "receipt.json" and payload != receipt_bytes:
                raise ValueError("RFC receipt changed before copy")
            _atomic_write_new_file_at_fd_v2(
                destination_directory_fd,
                name,
                payload,
                deadline_ns=deadline_ns,
            )
    finally:
        _close_owned_descriptors_v2((source_fd,), name="RFC raw bundle copy")
    copied_projection = _verify_rfc8785_raw_execution_bundle_v2(
        receipt,
        destination,
        expected_qualification_source_sha256=expected_qualification_source_sha256,
        replay_project_root=replay_project_root,
        fresh_execute=False,
        deadline_ns=deadline_ns,
        retained_bundle_fd=destination_directory_fd,
    )
    if not _exact_json_equal(projection, copied_projection):
        raise ValueError("copied RFC raw bundle projection drifted")
    return destination, projection


def _verify_cas_store_projection_v2(
    row: object,
    *,
    expected_sha256: str,
    expected_size: int,
    expected_store_namespace: str,
    name: str,
) -> None:
    if type(row) is not dict or set(row) != {
        "content_sha256",
        "object_key",
        "object_size_bytes",
        "prior_object_count",
        "put_exit_code",
        "replay_verified",
        "resulting_object_count",
        "store_namespace",
    }:
        raise ValueError(f"{name} CAS projection is not closed")
    if (
        row["content_sha256"] != expected_sha256
        or row["object_key"] != "sha256:" + expected_sha256
        or _require_exact_int_v2(
            row["object_size_bytes"], f"{name} CAS size", minimum=1
        )
        != expected_size
        or _require_exact_int_v2(
            row["prior_object_count"], f"{name} CAS prior count"
        )
        != 0
        or _require_exact_int_v2(
            row["resulting_object_count"], f"{name} CAS resulting count"
        )
        != 1
        or _require_exact_int_v2(row["put_exit_code"], f"{name} CAS exit")
        != 0
        or row["replay_verified"] is not True
        or row["store_namespace"] != expected_store_namespace
    ):
        raise ValueError(f"{name} CAS 0-to-1 store/replay receipt drifted")


def _audit_primitive_projection_v2(
    evidence: _IsolatedAuditEvidenceV2,
) -> dict[str, object]:
    receipt = evidence.audit_receipt
    invariant = receipt.invariant_runtime_evidence
    statistics = receipt.statistics_runtime_evidence
    issuance = receipt.receipt_issuance_runtime_evidence
    if (
        type(invariant) is not auditor.InvariantRuntimeEvidenceV1
        or type(statistics) is not auditor.StatisticalRuntimeEvidenceV1
        or type(issuance) is not auditor.ReceiptIssuanceRuntimeEvidenceV1
    ):
        raise ValueError("audit primitive projection lacks runtime evidence")
    resource_projection = _closed_primitive_v2(receipt.resource_plan)
    invariant_projection = _closed_primitive_v2(invariant)
    statistics_projection = _closed_primitive_v2(statistics)
    issuance_projection = _closed_primitive_v2(issuance)
    quota_projection = _eligibility_quota_projection_v2(receipt)
    _validate_complete_resource_plan_projection_v2(
        resource_projection,
        context="audit primitive",
    )
    _validate_complete_invariant_runtime_projection_v2(
        invariant_projection,
        resource_projection,
        context="audit primitive",
    )
    _validate_complete_statistics_runtime_projection_v2(
        statistics_projection,
        resource_projection,
        context="audit primitive",
    )
    _validate_complete_receipt_issuance_runtime_projection_v2(
        issuance_projection,
        statistics_projection,
        context="audit primitive",
    )
    receipt_id = _require_auditor_receipt_id_v2(
        evidence.validated_receipt_id,
        "audit primitive validated receipt ID",
    )
    if receipt.receipt_id != receipt_id:
        raise ValueError("audit primitive validated receipt ID drift")
    capacity_projection = _closed_feature_capacity_projection_v2(
        evidence.mode,
        receipt.audited_row_count,
        evidence.feature_capacity_projection,
    )
    if (
        capacity_projection["projection_root"]
        != evidence.feature_capacity_projection_root
    ):
        raise ValueError("audit primitive feature-capacity projection root drift")
    leak_witness = None
    if evidence.mode == "KNOWN_BINDING_LEAK":
        matches = tuple(
            row
            for row in receipt.results
            if row.target == "binding"
            and row.feature_id == evidence.leak_feature_id
            and row.failure_witness
        )
        if len(matches) != 1:
            raise ValueError("audit primitive projection lost its exact leak witness")
        leak_witness = matches[0]
    projection = {
        "mode": evidence.mode,
        "disposition": evidence.disposition,
        "run_fixture_root": evidence.run_fixture_root,
        "source_pair_delta_root": evidence.source_pair_delta_root,
        "operation_counts": dict(evidence.operation_counts),
        "label_factorial_proof_root": evidence.label_factorial_proof_root,
        "namespace_row_proof_root": evidence.namespace_row_proof_root,
        "diagnostics": _diagnostic_projection(evidence.diagnostics),
        "wire": {
            "adapter_execution_pin_root": receipt.adapter_execution_pin_root,
            "batch_id": evidence.batch_id,
            "build_success_id": evidence.build_success_id,
            "fresh_build_verification_root": evidence.fresh_build_verification_root,
            "key_source_receipt_id": evidence.key_source_receipt_id,
            "private_alignment_id": evidence.private_alignment_id,
            "transform_catalog_root": evidence.transform_catalog_root,
            "variant_context_id": evidence.variant_context_id,
        },
        "audit": {
            "receipt_id": receipt.receipt_id,
            "audited_rows": receipt.audited_row_count,
            "feature_manifest_root": receipt.feature_manifest.manifest_root,
            "feature_capacity_projection": capacity_projection,
            "feature_capacity_projection_root": (
                evidence.feature_capacity_projection_root
            ),
            "concrete_features": receipt.feature_manifest.deduplicated_feature_count,
            "raw_semantic_features": receipt.feature_manifest.raw_semantic_feature_count,
            "namespace_uuid_leaf_paths": receipt.feature_manifest.uuid_leaf_count,
            "channel_leaf_paths": receipt.feature_manifest.channel_leaf_count,
            "channel_only_owned_leaf_paths": (
                evidence.exact_owned_channel_only_leaf_paths
            ),
            "channel_only_owned_leaf_union_root": (
                evidence.channel_only_owner_union_root
            ),
            "full_raw_byte_categorical_hypotheses": (
                evidence.full_raw_byte_categorical_hypotheses
            ),
            "resource_plan": resource_projection,
            "resource_plan_root": stable_hash(
                resource_projection,
                prefix="phase2b_formal_wire_covert_resource_plan_v3_",
            ),
            "invariance": {
                "invariance_resource_plan_root": (
                    invariant.invariance_resource_plan_root
                ),
                "consumer_decode_count": invariant.consumer_decode_count,
                "exact_comparisons": receipt.exact_invariant_comparisons,
                "execution_root": receipt.invariant_execution_root,
                "run_count": receipt.invariant_run_count,
                "variant_suite_root": invariant.variant_suite_root,
                "worker_artifact_sha256": invariant.worker_result_artifact_sha256,
                "worker_peak_rss_bytes": invariant.worker_peak_rss_bytes,
                "wall_time_nanoseconds": invariant.invariance_wall_time_nanoseconds,
                "unified_cgroup_membership": invariant.unified_cgroup_membership,
                "unified_cgroup_memory_scope_root": (
                    invariant.unified_cgroup_memory_scope_root
                ),
                "unified_cgroup_leaf_device": invariant.unified_cgroup_leaf_device,
                "unified_cgroup_leaf_inode": invariant.unified_cgroup_leaf_inode,
                "physical_memory_current_bytes_at_completion": (
                    invariant.physical_memory_current_bytes_at_completion
                ),
                "physical_memory_peak_bytes_at_completion": (
                    invariant.physical_memory_peak_bytes_at_completion
                ),
                "physical_memory_max_bytes": invariant.physical_memory_max_bytes,
                "physical_memory_cap_bytes": invariant.physical_memory_cap_bytes,
                "within_frozen_physical_peak_cap": (
                    invariant.within_frozen_physical_peak_cap
                ),
                "complete_runtime_projection": invariant_projection,
                "complete_runtime_projection_root": stable_hash(
                    invariant_projection,
                    prefix="phase2b_formal_wire_covert_invariant_runtime_v4_",
                ),
            },
            "statistics": {
                "resource_plan_root": statistics.resource_plan_root,
                "diagnostic_parent_plus_worker_historical_peak_sum_bytes": (
                    statistics.diagnostic_parent_plus_worker_historical_peak_sum_bytes
                ),
                "unified_cgroup_membership": statistics.unified_cgroup_membership,
                "unified_cgroup_memory_scope_root": (
                    statistics.unified_cgroup_memory_scope_root
                ),
                "unified_cgroup_leaf_device": statistics.unified_cgroup_leaf_device,
                "unified_cgroup_leaf_inode": statistics.unified_cgroup_leaf_inode,
                "physical_memory_current_bytes_before_receipt_finalization": (
                    statistics.physical_memory_current_bytes_before_receipt_finalization
                ),
                "physical_memory_peak_bytes_before_receipt_finalization": (
                    statistics.physical_memory_peak_bytes_before_receipt_finalization
                ),
                "physical_memory_max_bytes": statistics.physical_memory_max_bytes,
                "physical_memory_cap_bytes": statistics.physical_memory_cap_bytes,
                "external_supervisor_required": statistics.external_supervisor_required,
                "pairwise_holm_decision_eligible": (
                    receipt.pairwise_holm_decision_eligible
                ),
                "pairwise_holm_family_size": receipt.pairwise_holm_family_size,
                "permutation_schedule_root": receipt.permutation_schedule_root,
                "permutation_seed_commitment": receipt.permutation_seed_commitment,
                "permutation_seed_reveal_hex": receipt.permutation_seed_reveal_hex,
                "worker_artifact_sha256": list(
                    statistics.worker_result_artifact_sha256
                ),
                "worker_peak_rss_bytes": list(statistics.worker_peak_rss_bytes),
                "statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds": (
                    statistics.statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds
                ),
                "within_frozen_physical_peak_cap": (
                    statistics.within_frozen_physical_peak_cap
                ),
                "complete_runtime_projection": statistics_projection,
                "complete_runtime_projection_root": stable_hash(
                    statistics_projection,
                    prefix="phase2b_formal_wire_covert_statistics_runtime_v4_",
                ),
            },
            "receipt_issuance": {
                "receipt_id": receipt_id,
                "absolute_deadline_kind": issuance.absolute_deadline_kind,
                "diagnostic_worker_historical_peak_rss_bytes_sum": (
                    issuance.diagnostic_worker_historical_peak_rss_bytes_sum
                ),
                "diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization": (
                    issuance.diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization
                ),
                "diagnostic_parent_plus_worker_historical_peak_sum_bytes": (
                    issuance.diagnostic_parent_plus_worker_historical_peak_sum_bytes
                ),
                "unified_cgroup_membership": issuance.unified_cgroup_membership,
                "unified_cgroup_memory_scope_root": (
                    issuance.unified_cgroup_memory_scope_root
                ),
                "unified_cgroup_leaf_device": issuance.unified_cgroup_leaf_device,
                "unified_cgroup_leaf_inode": issuance.unified_cgroup_leaf_inode,
                "physical_memory_current_bytes_before_receipt_finalization": (
                    issuance.physical_memory_current_bytes_before_receipt_finalization
                ),
                "physical_memory_peak_bytes_before_receipt_finalization": (
                    issuance.physical_memory_peak_bytes_before_receipt_finalization
                ),
                "physical_memory_max_bytes": issuance.physical_memory_max_bytes,
                "physical_memory_cap_bytes": issuance.physical_memory_cap_bytes,
                "receipt_finalization_complete_pending_validation_count": (
                    issuance.receipt_finalization_complete_pending_validation_count
                ),
                "receipt_finalization_locked_deadline_check_count": (
                    issuance.receipt_finalization_locked_deadline_check_count
                ),
                "receipt_finalization_locked_deadline_within_absolute_limit": (
                    issuance.receipt_finalization_locked_deadline_within_absolute_limit
                ),
                "receipt_finalization_locked_physical_peak_within_cap": (
                    issuance.receipt_finalization_locked_physical_peak_within_cap
                ),
                "complete_runtime_projection": issuance_projection,
                "complete_runtime_projection_root": stable_hash(
                    issuance_projection,
                    prefix=(
                        "phase2b_formal_wire_covert_receipt_issuance_runtime_v4_"
                    ),
                ),
            },
            "target_family_max_t_holm": [
                {
                    "target": row.target,
                    "exceedance_count": row.exceedance_count,
                    "observed_target_max_omnibus_hex": (
                        row.observed_target_max_omnibus.hex()
                    ),
                    "raw_p": _exact_probability_text_v2(row.raw_p),
                    "holm_adjusted_p": _exact_probability_text_v2(
                        row.holm_adjusted_p
                    ),
                    "holm_critical_value": _exact_probability_text_v2(
                        row.holm_critical_value
                    ),
                    "holm_rank": row.holm_rank,
                    "holm_rejected": row.holm_rejected,
                }
                for row in receipt.target_family_results
            ],
            "eligibility_quota": quota_projection,
            "eligibility_quota_root": stable_hash(
                quota_projection,
                prefix="phase2b_formal_wire_covert_eligibility_quota_v2_",
            ),
            "positive_decision_claim": receipt.positive_decision_claim,
            "failure_witness_count": sum(
                row.failure_witness for row in receipt.results
            ),
            "actual_corpus_audit": receipt.actual_corpus_audit,
            "authority_origin_authenticated": receipt.authority_origin_authenticated,
            "authority_independence_verified": receipt.authority_independence_verified,
        },
        "known_leak_witness": (
            None
            if evidence.mode == "KNOWN_NULL"
            else {
                "feature_id": evidence.leak_feature_id,
                "feature_aliases": list(evidence.leak_feature_aliases),
                "feature_owner": "eligibility_mask",
                "feature_owner_path": _json_path_text_v2(
                    _REGISTERED_BINDING_LEAK_OWNER_PATH_V2
                ),
                "registered_presence_alias": next(
                    alias
                    for alias in evidence.leak_feature_aliases
                    if _is_registered_binding_leak_alias_v2(alias)
                ),
                "raw_nmi_hex": evidence.leak_raw_nmi_hex,
                "bias_corrected_nmi_hex": evidence.leak_bias_corrected_nmi_hex,
                "balanced_accuracy_advantage_hex": (
                    evidence.leak_balanced_accuracy_advantage_hex
                ),
                "observed_omnibus_hex": evidence.leak_observed_omnibus_hex,
                "feature_max_t_exceedance_count": (
                    leak_witness.feature_max_t_exceedance_count
                ),
                "feature_max_t_p": evidence.leak_feature_max_t_p,
                "target_holm_critical_value": (
                    evidence.leak_target_holm_critical_value
                ),
                "material_effect": leak_witness.material_effect,
                "target_family_holm_rejected": (
                    leak_witness.target_family_holm_rejected
                ),
                "feature_localization_rejected": (
                    leak_witness.feature_localization_rejected
                ),
                "failure_witness": leak_witness.failure_witness,
                "passed": leak_witness.passed,
            }
        ),
    }
    _validate_executed_runtime_bundle_v2(
        resource_plan=resource_projection,
        statistics_runtime=statistics_projection,
        receipt_issuance_runtime=issuance_projection,
        diagnostics=projection["diagnostics"],
        run=projection,
        expected_mode=evidence.mode,
        visible_audited_rows=receipt.audited_row_count,
        visible_concrete_features=(
            receipt.feature_manifest.deduplicated_feature_count
        ),
        context="audit primitive",
    )
    return projection


def build_development_calibration_artifact_v2(
    profile: Phase2BFormalWireCovertQualificationProfileV2,
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
) -> bytes:
    """Measure exact full shapes once; the artifact can never be promoted."""

    if type(profile) is not Phase2BFormalWireCovertQualificationProfileV2:
        _fail(FAIL_INPUT, "development calibration needs the exact loaded profile")
    profile._validate()
    attempt.__post_init__()
    if attempt.profile_id != PROFILE_ID:
        _fail(FAIL_PROFILE, "development calibration attempt must bind current profile")
    shape = _EXPECTED_PROFILE["fixture_contract"]["rich_positive_shape"]
    _require_pending_calibration_profile_v2(shape)
    runtime = prepare_live_inner_process_v2()
    _validate_live_runtime_entry_v2(runtime, attempt)
    positive: _IsolatedAuditEvidenceV2 | None = None
    negative: _IsolatedAuditEvidenceV2 | None = None
    try:
        positive = _execute_isolated_audit_v2(LIVE_MODE_ORDER[0])
        negative = _execute_isolated_audit_v2(LIVE_MODE_ORDER[1])
        fixture_root = _combined_fixture_root_v2(positive, negative)
        measurements = [
            _measured_shape_projection_v2(positive),
            _measured_shape_projection_v2(negative),
        ]
        runs = [
            _audit_primitive_projection_v2(positive),
            _audit_primitive_projection_v2(negative),
        ]
        _validate_and_release_completed_audits_v2((positive, negative))
        runtime_exit = _capture_live_runtime_exit_v2(runtime, attempt, runs)
        body = {
            "schema_version": (
                "hegel-machine-phase2b-formal-wire-covert-development-calibration/6"
            ),
            "profile_id": PROFILE_ID,
            "attempt_manifest_root": attempt.attempt_manifest_root,
            "calibration_campaign_id": attempt.campaign_id,
            "calibration_attempt_id": attempt.attempt_id,
            "mode_order": list(LIVE_MODE_ORDER),
            "claim_level": "DEVELOPMENT_MEASUREMENT_ONLY_NOT_R03_R04_R05",
            "artifact_cannot_be_finalized_or_promoted": True,
            "synthetic_fixture_root": fixture_root,
            "runtime_entry_evidence": runtime,
            "runtime_exit_evidence": runtime_exit,
            "post_output_physical_gate_contract": dict(
                _POST_OUTPUT_PHYSICAL_GATE_CONTRACT_V2
            ),
            "frozen_static_measurement_candidates_in_order": measurements,
            "runs_in_order": runs,
            "claims": {
                "r03_formal_trusted_wire_implementation_qualified": False,
                "r04_formal_covert_auditor_implementation_qualified": False,
                "r05_synthetic_dev_wire_covert_qualification_verified": False,
                "actual_corpus_audit": False,
                "actual_unsealed_960_generation": False,
                "origin_authenticated": False,
                "key_independence_verified": False,
                "campaign_execution_authorized": False,
                "fixture_specific_only": True,
                "joint_decision_classes": 2,
                "wire_wider_graphs_and_100_class_profile_qualified": False,
            },
            "projection_only_boundary": dict(_TASK4_PROJECTION_ONLY_BOUNDARY_V2),
            "outer_hard_deadline_self_verified": False,
            "external_supervisor_evidence_required": True,
            "independent_review_and_profile_update_required": True,
            "same_run_shape_acceptance_forbidden": True,
            "content_sha256_is_not_a_signature": True,
        }
        return _canonical_inner_artifact_v2(body)
    except BaseException:
        for evidence in (positive, negative):
            if evidence is not None:
                try:
                    auditor.release_formal_covert_audit_receipt_v1(
                        evidence.audit_receipt
                    )
                except (KeyError, TypeError, ValueError):
                    pass
        raise


def build_live_inner_candidate_artifact_v2(
    profile: Phase2BFormalWireCovertQualificationProfileV2,
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
) -> bytes:
    """Build a fresh-comparison inner artifact; it never issues R03/R04/R05."""

    if type(profile) is not Phase2BFormalWireCovertQualificationProfileV2:
        _fail(FAIL_INPUT, "live inner candidate needs the exact loaded profile")
    profile._validate()
    attempt.__post_init__()
    if attempt.profile_id != PROFILE_ID:
        _fail(FAIL_PROFILE, "live inner attempt must bind current profile")
    shape = _EXPECTED_PROFILE["fixture_contract"]["rich_positive_shape"]
    if type(shape) is not dict or shape.get("live_qualification_enabled") is not True:
        _fail(
            FAIL_PROFILE,
            "live inner execution is disabled until full-1024 shape calibration freezes",
        )
    runtime = prepare_live_inner_process_v2()
    _validate_live_runtime_entry_v2(runtime, attempt)
    positive: _IsolatedAuditEvidenceV2 | None = None
    negative: _IsolatedAuditEvidenceV2 | None = None
    try:
        positive = _execute_isolated_audit_v2(LIVE_MODE_ORDER[0])
        negative = _execute_isolated_audit_v2(LIVE_MODE_ORDER[1])
        _verify_frozen_shape_v2(shape, positive, negative, attempt=attempt)
        fixture_root = _combined_fixture_root_v2(positive, negative)
        measurements = [
            _measured_shape_projection_v2(positive),
            _measured_shape_projection_v2(negative),
        ]
        runs = [
            _audit_primitive_projection_v2(positive),
            _audit_primitive_projection_v2(negative),
        ]
        _validate_and_release_completed_audits_v2((positive, negative))
        runtime_exit = _capture_live_runtime_exit_v2(runtime, attempt, runs)
        body = {
            "schema_version": (
                "hegel-machine-phase2b-formal-wire-covert-live-inner-candidate/6"
            ),
            "profile_id": PROFILE_ID,
            "attempt_manifest_root": attempt.attempt_manifest_root,
            "campaign_id": attempt.campaign_id,
            "attempt_id": attempt.attempt_id,
            "mode_order": list(LIVE_MODE_ORDER),
            "claim_level": "INNER_CANDIDATE_NOT_R03_R04_R05",
            "synthetic_fixture_root": fixture_root,
            "runtime_entry_evidence": runtime,
            "runtime_exit_evidence": runtime_exit,
            "post_output_physical_gate_contract": dict(
                _POST_OUTPUT_PHYSICAL_GATE_CONTRACT_V2
            ),
            "fresh_static_measurements_in_order": measurements,
            "frozen_static_shape_exact_match": True,
            "runs_in_order": runs,
            "claims": {
                "r03_formal_trusted_wire_implementation_qualified": False,
                "r04_formal_covert_auditor_implementation_qualified": False,
                "r05_synthetic_dev_wire_covert_qualification_verified": False,
                "actual_corpus_audit": False,
                "actual_unsealed_960_generation": False,
                "origin_authenticated": False,
                "key_independence_verified": False,
                "campaign_execution_authorized": False,
                "fixture_specific_only": True,
                "joint_decision_classes": 2,
                "wire_wider_graphs_and_100_class_profile_qualified": False,
            },
            "projection_only_boundary": dict(_TASK4_PROJECTION_ONLY_BOUNDARY_V2),
            "outer_hard_deadline_self_verified": False,
            "external_finalizer_required": True,
            "content_sha256_is_not_a_signature": True,
        }
        return _canonical_inner_artifact_v2(body)
    except BaseException:
        for evidence in (positive, negative):
            if evidence is not None:
                try:
                    auditor.release_formal_covert_audit_receipt_v1(
                        evidence.audit_receipt
                    )
                except (KeyError, TypeError, ValueError):
                    pass
        raise


def _static_shape_from_run_projection_v2(
    run: dict[str, object],
) -> dict[str, object]:
    audit_row = run.get("audit")
    wire_row = run.get("wire")
    if type(audit_row) is not dict or type(wire_row) is not dict:
        raise ValueError("run lacks audit/wire data for static-shape replay")
    resource = audit_row.get("resource_plan")
    quota = audit_row.get("eligibility_quota")
    if type(resource) is not dict or type(quota) is not dict:
        raise ValueError("run lacks resource/quota data for static-shape replay")
    _require_eligibility_quota_receipt_id_v2(
        quota.get("receipt_id"),
        "static-shape eligibility-mask quota receipt ID",
    )
    _validate_complete_resource_plan_projection_v2(
        resource, context="static-shape replay"
    )
    capacity = _validate_closed_feature_capacity_projection_v2(
        run.get("mode"),
        audit_row.get("audited_rows"),
        audit_row.get("feature_capacity_projection"),
    )
    if audit_row.get("feature_capacity_projection_root") != capacity[
        "projection_root"
    ]:
        raise ValueError("replayed feature-capacity projection root drift")
    capacity_receipt_projection = {
        "audited_rows": capacity["rows"],
        "channel_leaf_count": capacity["channel_union_path_count"],
        "concrete_feature_count": capacity["projected_concrete_feature_count"],
        "preextraction_feature_matrix_bytes_bound": capacity[
            "projected_feature_matrix_bytes_bound"
        ],
        "raw_semantic_feature_count": capacity[
            "projected_raw_semantic_feature_count"
        ],
        "uuid_leaf_count": capacity["uuid_union_path_count"],
    }
    observed_receipt_projection = {
        "audited_rows": audit_row.get("audited_rows"),
        "channel_leaf_count": audit_row.get("channel_leaf_paths"),
        "concrete_feature_count": audit_row.get("concrete_features"),
        "preextraction_feature_matrix_bytes_bound": resource.get(
            "preextraction_feature_matrix_bytes_bound"
        ),
        "raw_semantic_feature_count": audit_row.get("raw_semantic_features"),
        "uuid_leaf_count": audit_row.get("namespace_uuid_leaf_paths"),
    }
    _validate_fixture_capacity_receipt_projection_v2(
        run.get("mode"),
        observed_receipt_projection,
    )
    if observed_receipt_projection != capacity_receipt_projection:
        raise ValueError("replayed receipt capacity differs from closed projection")
    (
        exact_owned_channel_only_leaf_paths,
        channel_only_owner_union_root,
    ) = _validate_fixture_channel_only_owner_union_v2(
        run.get("mode"),
        observed_receipt_projection["audited_rows"],
        audit_row.get("channel_only_owned_leaf_paths"),
        audit_row.get("channel_only_owned_leaf_union_root"),
    )
    static_resource = {
        name: resource[name] for name in FROZEN_STATIC_RESOURCE_PLAN_FIELDS_V2
    }
    return {
        "schema_version": (
            "hegel-machine-phase2b-formal-wire-covert-static-shape/3"
        ),
        "mode": run.get("mode"),
        "rows": observed_receipt_projection["audited_rows"],
        "joint_decision_classes": 2,
        "fixture_specific_only": True,
        "run_fixture_root": run.get("run_fixture_root"),
        "source_pair_delta_root": run.get("source_pair_delta_root"),
        "transform_catalog_root": wire_row.get("transform_catalog_root"),
        "namespace_row_proof_root": run.get("namespace_row_proof_root"),
        "build_success_id": wire_row.get("build_success_id"),
        "batch_id": wire_row.get("batch_id"),
        "operation_counts": run.get("operation_counts"),
        "label_factorial_proof_root": run.get("label_factorial_proof_root"),
        "label_factorial_summary": {
            "binary_factor_count": 6,
            "joint_decision_class_levels": ["joint-a", "joint-b"],
            "label_cells": 64,
            "rows_per_label_cell": 16,
            "operations_per_label_cell": 8,
            "rows_per_operation_per_label_cell": 2,
        },
        "feature_manifest_root": audit_row.get("feature_manifest_root"),
        "feature_capacity_projection": capacity,
        "feature_capacity_projection_root": audit_row.get(
            "feature_capacity_projection_root"
        ),
        "concrete_features": observed_receipt_projection[
            "concrete_feature_count"
        ],
        "raw_semantic_features": observed_receipt_projection[
            "raw_semantic_feature_count"
        ],
        "uuid_leaf_count": observed_receipt_projection["uuid_leaf_count"],
        "channel_leaf_count": observed_receipt_projection["channel_leaf_count"],
        "channel_only_owned_leaf_count": exact_owned_channel_only_leaf_paths,
        "channel_only_owned_leaf_union_root": channel_only_owner_union_root,
        "full_raw_byte_categorical_hypotheses": audit_row.get(
            "full_raw_byte_categorical_hypotheses"
        ),
        "eligibility_quota": quota,
        "eligibility_quota_root": stable_hash(
            quota,
            prefix="phase2b_formal_wire_covert_eligibility_quota_v2_",
        ),
        "resource_plan_static": static_resource,
        "resource_plan_static_root": stable_hash(
            static_resource,
            prefix="phase2b_formal_wire_covert_static_resource_plan_v2_",
        ),
    }


def _validate_run_and_measurement_binding_v2(
    run: dict[str, object],
    measurement: dict[str, object],
) -> dict[str, object]:
    expected_run_keys = {
        "audit",
        "diagnostics",
        "disposition",
        "known_leak_witness",
        "label_factorial_proof_root",
        "mode",
        "namespace_row_proof_root",
        "operation_counts",
        "run_fixture_root",
        "source_pair_delta_root",
        "wire",
    }
    expected_measurement_keys = {
        "diagnostics",
        "dynamic_fields_are_recorded_but_not_exact_compared",
        "frozen_static_projection",
        "frozen_static_projection_root",
        "mode",
        "observed_complete_receipt_issuance_runtime",
        "observed_complete_receipt_issuance_runtime_root",
        "observed_complete_resource_plan",
        "observed_complete_resource_plan_root",
        "observed_receipt_id",
    }
    expected_wire_keys = {
        "adapter_execution_pin_root",
        "batch_id",
        "build_success_id",
        "fresh_build_verification_root",
        "key_source_receipt_id",
        "private_alignment_id",
        "transform_catalog_root",
        "variant_context_id",
    }
    expected_audit_keys = {
        "actual_corpus_audit",
        "audited_rows",
        "authority_independence_verified",
        "authority_origin_authenticated",
        "channel_leaf_paths",
        "channel_only_owned_leaf_paths",
        "channel_only_owned_leaf_union_root",
        "concrete_features",
        "eligibility_quota",
        "eligibility_quota_root",
        "failure_witness_count",
        "feature_capacity_projection",
        "feature_capacity_projection_root",
        "feature_manifest_root",
        "full_raw_byte_categorical_hypotheses",
        "invariance",
        "namespace_uuid_leaf_paths",
        "positive_decision_claim",
        "raw_semantic_features",
        "receipt_id",
        "resource_plan",
        "resource_plan_root",
        "receipt_issuance",
        "statistics",
        "target_family_max_t_holm",
    }
    expected_invariance_keys = {
        "complete_runtime_projection",
        "complete_runtime_projection_root",
        "consumer_decode_count",
        "exact_comparisons",
        "execution_root",
        "invariance_resource_plan_root",
        "run_count",
        "variant_suite_root",
        "wall_time_nanoseconds",
        "worker_artifact_sha256",
        "worker_peak_rss_bytes",
        "unified_cgroup_membership",
        "unified_cgroup_memory_scope_root",
        "unified_cgroup_leaf_device",
        "unified_cgroup_leaf_inode",
        "physical_memory_current_bytes_at_completion",
        "physical_memory_peak_bytes_at_completion",
        "physical_memory_max_bytes",
        "physical_memory_cap_bytes",
        "within_frozen_physical_peak_cap",
    }
    expected_statistics_keys = {
        "complete_runtime_projection",
        "complete_runtime_projection_root",
        "diagnostic_parent_plus_worker_historical_peak_sum_bytes",
        "external_supervisor_required",
        "pairwise_holm_decision_eligible",
        "pairwise_holm_family_size",
        "permutation_schedule_root",
        "permutation_seed_commitment",
        "permutation_seed_reveal_hex",
        "resource_plan_root",
        "unified_cgroup_membership",
        "unified_cgroup_memory_scope_root",
        "unified_cgroup_leaf_device",
        "unified_cgroup_leaf_inode",
        "physical_memory_current_bytes_before_receipt_finalization",
        "physical_memory_peak_bytes_before_receipt_finalization",
        "physical_memory_max_bytes",
        "physical_memory_cap_bytes",
        "statistics_and_parent_postprocess_elapsed_at_receipt_build_nanoseconds",
        "worker_artifact_sha256",
        "worker_peak_rss_bytes",
        "within_frozen_physical_peak_cap",
    }
    expected_receipt_issuance_keys = {
        "absolute_deadline_kind",
        "complete_runtime_projection",
        "complete_runtime_projection_root",
        "diagnostic_worker_historical_peak_rss_bytes_sum",
        "diagnostic_parent_historical_peak_rss_bytes_before_receipt_finalization",
        "diagnostic_parent_plus_worker_historical_peak_sum_bytes",
        "unified_cgroup_membership",
        "unified_cgroup_memory_scope_root",
        "unified_cgroup_leaf_device",
        "unified_cgroup_leaf_inode",
        "physical_memory_current_bytes_before_receipt_finalization",
        "physical_memory_peak_bytes_before_receipt_finalization",
        "physical_memory_max_bytes",
        "physical_memory_cap_bytes",
        "receipt_finalization_complete_pending_validation_count",
        "receipt_finalization_locked_deadline_check_count",
        "receipt_finalization_locked_deadline_within_absolute_limit",
        "receipt_finalization_locked_physical_peak_within_cap",
        "receipt_id",
    }
    expected_diagnostic_keys = {
        "host_available_memory_bytes_at_preflight",
        "maximum_rows_per_invocation",
        "minimum_rows_per_invocation",
        "mode",
        "recognizer_adapter_invocations",
        "parent_historical_peak_rss_before_receipt_finalization_kib",
        "parent_plus_worker_historical_peak_rss_kib",
        "physical_memory_cap_bytes",
        "physical_memory_current_before_receipt_finalization_bytes",
        "physical_memory_max_bytes",
        "physical_memory_peak_before_receipt_finalization_bytes",
        "statistics_wall_time_ns",
        "statistics_worker_historical_peak_rss_kib",
        "unified_cgroup_membership",
        "unified_cgroup_leaf_device",
        "unified_cgroup_leaf_inode",
        "unified_cgroup_memory_scope_root",
        "wall_time_ns",
    }
    if (
        type(run) is not dict
        or set(run) != expected_run_keys
        or type(measurement) is not dict
        or set(measurement) != expected_measurement_keys
        or type(run.get("wire")) is not dict
        or set(run["wire"]) != expected_wire_keys
        or type(run.get("audit")) is not dict
        or set(run["audit"]) != expected_audit_keys
        or type(run.get("diagnostics")) is not dict
        or set(run["diagnostics"]) != expected_diagnostic_keys
        or type(measurement.get("diagnostics")) is not dict
        or set(measurement["diagnostics"]) != expected_diagnostic_keys
    ):
        raise ValueError("run/static-measurement nested fields are not exact")
    audit_row = run["audit"]
    invariant = audit_row["invariance"]
    statistics = audit_row["statistics"]
    receipt_issuance = audit_row["receipt_issuance"]
    quota = audit_row["eligibility_quota"]
    if (
        type(invariant) is not dict
        or set(invariant) != expected_invariance_keys
        or type(statistics) is not dict
        or set(statistics) != expected_statistics_keys
        or type(receipt_issuance) is not dict
        or set(receipt_issuance) != expected_receipt_issuance_keys
        or type(quota) is not dict
        or set(quota) != {"exact_quota_matched", "receipt_id", "targets"}
        or type(invariant.get("complete_runtime_projection")) is not dict
        or set(invariant["complete_runtime_projection"])
        != {item.name for item in dataclass_fields(auditor.InvariantRuntimeEvidenceV1)}
        or type(statistics.get("complete_runtime_projection")) is not dict
        or set(statistics["complete_runtime_projection"])
        != {item.name for item in dataclass_fields(auditor.StatisticalRuntimeEvidenceV1)}
        or type(receipt_issuance.get("complete_runtime_projection")) is not dict
        or set(receipt_issuance["complete_runtime_projection"])
        != set(_RECEIPT_ISSUANCE_RUNTIME_FIELDS_V2)
    ):
        raise ValueError("run audit nested runtime fields are not exact")
    frozen = _static_shape_from_run_projection_v2(run)
    full_resource = audit_row["resource_plan"]
    _validate_complete_resource_plan_projection_v2(
        full_resource, context="run/measurement binding"
    )
    invariant_complete = invariant["complete_runtime_projection"]
    statistics_complete = statistics["complete_runtime_projection"]
    receipt_issuance_complete = receipt_issuance["complete_runtime_projection"]
    _validate_complete_invariant_runtime_projection_v2(
        invariant_complete,
        full_resource,
        context="run/measurement binding",
    )
    _validate_complete_statistics_runtime_projection_v2(
        statistics_complete,
        full_resource,
        context="run/measurement binding",
    )
    _validate_complete_receipt_issuance_runtime_projection_v2(
        receipt_issuance_complete,
        statistics_complete,
        context="run/measurement binding",
    )
    _validate_executed_runtime_bundle_v2(
        resource_plan=full_resource,
        statistics_runtime=statistics_complete,
        receipt_issuance_runtime=receipt_issuance_complete,
        diagnostics=run["diagnostics"],
        run=run,
        measurement=measurement,
        expected_mode=run["mode"],
        visible_audited_rows=EXACT_AUTHORITY_ROWS,
        visible_concrete_features=(
            _fixture_feature_capacity_projection_row_v2(
                run["mode"],
                EXACT_AUTHORITY_ROWS,
            )["projected_concrete_feature_count"]
        ),
        context="run/measurement binding",
    )
    if (
        invariant.get("invariance_resource_plan_root")
        != full_resource["invariance_resource_plan_root"]
        or invariant_complete["invariance_resource_plan_root"]
        != full_resource["invariance_resource_plan_root"]
        or statistics.get("resource_plan_root")
        != full_resource["resource_plan_root"]
        or statistics_complete["resource_plan_root"]
        != full_resource["resource_plan_root"]
        or receipt_issuance.get("receipt_id") != audit_row.get("receipt_id")
        or any(
            not _exact_json_equal(
                receipt_issuance.get(name), receipt_issuance_complete[name]
            )
            for name in _RECEIPT_ISSUANCE_RUNTIME_FIELDS_V2
            if name in expected_receipt_issuance_keys
        )
    ):
        raise ValueError("run two-phase resource/issuance bindings drifted")
    if (
        measurement.get("mode") != run.get("mode")
        or not _exact_json_equal(
            measurement.get("frozen_static_projection"), frozen
        )
        or measurement.get("frozen_static_projection_root")
        != stable_hash(
            frozen,
            prefix="phase2b_formal_wire_covert_static_shape_v3_",
        )
        or not _exact_json_equal(
            measurement.get("observed_complete_resource_plan"), full_resource
        )
        or measurement.get("observed_complete_resource_plan_root")
        != stable_hash(
            full_resource,
            prefix="phase2b_formal_wire_covert_observed_resource_plan_v3_",
        )
        or measurement.get("observed_receipt_id") != audit_row.get("receipt_id")
        or not _exact_json_equal(
            measurement.get("observed_complete_receipt_issuance_runtime"),
            receipt_issuance_complete,
        )
        or measurement.get("observed_complete_receipt_issuance_runtime_root")
        != stable_hash(
            receipt_issuance_complete,
            prefix="phase2b_formal_wire_covert_receipt_issuance_runtime_v4_",
        )
        or measurement.get("dynamic_fields_are_recorded_but_not_exact_compared")
        != list(DYNAMIC_RESOURCE_PLAN_FIELDS_NOT_EXACT_COMPARED_V2)
        or not _exact_json_equal(measurement.get("diagnostics"), run.get("diagnostics"))
    ):
        raise ValueError("run/static-measurement cross-binding drifted")
    resource_projection = audit_row["resource_plan"]
    if audit_row.get("resource_plan_root") != stable_hash(
        resource_projection,
        prefix="phase2b_formal_wire_covert_resource_plan_v3_",
    ):
        raise ValueError("run complete resource-plan root drifted")
    if audit_row.get("eligibility_quota_root") != stable_hash(
        quota,
        prefix="phase2b_formal_wire_covert_eligibility_quota_v2_",
    ):
        raise ValueError("run eligibility-quota root drifted")
    for group_name, prefix in (
        ("invariance", "phase2b_formal_wire_covert_invariant_runtime_v4_"),
        ("statistics", "phase2b_formal_wire_covert_statistics_runtime_v4_"),
        (
            "receipt_issuance",
            "phase2b_formal_wire_covert_receipt_issuance_runtime_v4_",
        ),
    ):
        group = audit_row[group_name]
        if type(group) is not dict or group.get(
            "complete_runtime_projection_root"
        ) != stable_hash(group.get("complete_runtime_projection"), prefix=prefix):
            raise ValueError(f"run {group_name} complete runtime root drifted")
    return frozen


def _validate_interpreter_runtime_projection_v2(
    value: object, name: str
) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "bytecode_candidates_found",
        "dont_write_bytecode",
        "executed_main_source_sha256",
        "ignore_environment",
        "isolated",
        "no_site",
        "package_directory",
        "preopened_source_snapshot_root",
        "source_execution_bootstrap_active",
        "source_execution_bootstrap_manifest_root",
        "source_only_importer_active",
    }:
        raise ValueError(f"{name} interpreter runtime projection is not closed")
    for flag_name in (
        "dont_write_bytecode",
        "ignore_environment",
        "isolated",
        "no_site",
    ):
        if _require_exact_int_v2(
            value[flag_name], f"{name} interpreter {flag_name}"
        ) != 1:
            raise ValueError(f"{name} interpreter flags drifted")
    if (
        value["source_only_importer_active"] is not True
        or value["source_execution_bootstrap_active"] is not True
        or not _exact_json_equal(value["bytecode_candidates_found"], [])
        or re.fullmatch(r"[0-9a-f]{64}", value["executed_main_source_sha256"])
        is None
        or type(value["preopened_source_snapshot_root"]) is not str
        or re.fullmatch(
            r"phase2b_formal_wire_covert_preopened_sources_v2_[0-9a-f]{64}",
            value["preopened_source_snapshot_root"],
        )
        is None
        or type(value["source_execution_bootstrap_manifest_root"]) is not str
        or re.fullmatch(
            r"phase2b_formal_wire_covert_source_bootstrap_manifest_v2_[0-9a-f]{64}",
            value["source_execution_bootstrap_manifest_root"],
        )
        is None
        or type(value["package_directory"]) is not str
        or not Path(value["package_directory"]).is_absolute()
    ):
        raise ValueError(f"{name} source-only interpreter evidence drifted")
    return value


def _validate_runtime_entry_exit_contract_v2(
    entry: object,
    exit_row: object,
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
    *,
    context: str,
) -> None:
    if type(entry) is not dict or set(entry) != {
        "cgroup",
        "interpreter_runtime",
        "selected_default_signals",
        "sigblk_hex",
        "threads",
    }:
        raise ValueError(f"{context} entry runtime projection is not closed")
    if type(exit_row) is not dict or set(exit_row) != {
        "audit_receipts_validated_and_released_in_process",
        "cgroup",
        "cgroup_events_unchanged_from_entry",
        "selected_default_signals",
        "sigblk_hex",
        "threads",
    }:
        raise ValueError(f"{context} exit runtime projection is not closed")
    _validate_interpreter_runtime_projection_v2(
        entry["interpreter_runtime"], context
    )
    entry_cgroup = entry["cgroup"]
    exit_cgroup = exit_row["cgroup"]
    expected_cgroup_keys = {
        "cgroup_path",
        "cgroup_procs",
        "cgroup_stat",
        "cpu_affinity",
        "cpu_capacity",
        "cpu_max",
        "memory_current",
        "memory_events",
        "memory_max",
        "memory_peak",
        "pids_current",
        "pids_events",
        "pids_max",
        "unified_cgroup_leaf_device",
        "unified_cgroup_leaf_inode",
        "unified_cgroup_memory_scope_root",
    }
    if (
        type(entry_cgroup) is not dict
        or type(exit_cgroup) is not dict
        or set(entry_cgroup) != expected_cgroup_keys
        or set(exit_cgroup) != expected_cgroup_keys
    ):
        raise ValueError(f"{context} cgroup projection is not closed")
    expected_signals = [number.name for number in LIVE_SELECTED_DEFAULT_SIGNALS]
    if (
        _require_exact_int_v2(entry["threads"], f"{context} entry threads") != 1
        or _require_exact_int_v2(exit_row["threads"], f"{context} exit threads")
        != 1
        or entry["sigblk_hex"] != "0000000000000000"
        or exit_row["sigblk_hex"] != "0000000000000000"
        or not _exact_json_equal(entry["selected_default_signals"], expected_signals)
        or not _exact_json_equal(
            exit_row["selected_default_signals"], expected_signals
        )
        or exit_row["audit_receipts_validated_and_released_in_process"]
        is not True
        or exit_row["cgroup_events_unchanged_from_entry"] is not True
        or entry_cgroup["cgroup_path"] != exit_cgroup["cgroup_path"]
        or entry_cgroup["unified_cgroup_memory_scope_root"]
        != exit_cgroup["unified_cgroup_memory_scope_root"]
        or entry_cgroup["unified_cgroup_leaf_device"]
        != exit_cgroup["unified_cgroup_leaf_device"]
        or entry_cgroup["unified_cgroup_leaf_inode"]
        != exit_cgroup["unified_cgroup_leaf_inode"]
    ):
        raise ValueError(f"{context} signal/thread/runtime identity drifted")
    for phase, cgroup in (("entry", entry_cgroup), ("exit", exit_cgroup)):
        affinity = cgroup["cpu_affinity"]
        membership = _require_safe_cgroup_path_v2(
            cgroup["cgroup_path"], f"{context} {phase} cgroup"
        )
        leaf_device = _require_exact_int_v2(
            cgroup["unified_cgroup_leaf_device"],
            f"{context} {phase} cgroup leaf device",
            minimum=1,
        )
        leaf_inode = _require_exact_int_v2(
            cgroup["unified_cgroup_leaf_inode"],
            f"{context} {phase} cgroup leaf inode",
            minimum=1,
        )
        expected_scope_root = stable_hash(
            {
                "leaf_device": leaf_device,
                "leaf_inode": leaf_inode,
                "unified_membership": membership,
            },
            prefix="phase2b_formal_covert_unified_cgroup_memory_scope_v1_",
        )
        if (
            membership != cgroup["cgroup_path"]
            or cgroup["unified_cgroup_memory_scope_root"]
            != expected_scope_root
            or type(affinity) is not list
            or not affinity
            or any(type(cpu) is not int or cpu < 0 for cpu in affinity)
            or len(set(affinity)) != len(affinity)
            or _require_exact_int_v2(
                cgroup["cpu_capacity"], f"{context} {phase} CPU capacity", minimum=1
            )
            != len(affinity)
            or (
                cgroup["cpu_max"] is not None
                and (
                    type(cgroup["cpu_max"]) is not str
                    or not cgroup["cpu_max"].strip()
                )
            )
            or _require_exact_int_v2(
                cgroup["memory_max"], f"{context} {phase} memory.max", minimum=1
            )
            != attempt.requested_cgroup_memory_max_bytes
            or _require_exact_int_v2(
                cgroup["pids_max"], f"{context} {phase} pids.max", minimum=1
            )
            != attempt.requested_cgroup_pids_max
            or _require_exact_int_v2(
                cgroup["pids_current"], f"{context} {phase} pids.current"
            )
            != 1
            or type(cgroup["cgroup_procs"]) is not list
            or len(cgroup["cgroup_procs"]) != 1
            or _require_exact_int_v2(
                cgroup["cgroup_procs"][0], f"{context} {phase} cgroup.procs PID",
                minimum=1,
            )
            < 1
            or type(cgroup["cgroup_stat"]) is not dict
            or cgroup["cgroup_stat"].get("nr_descendants") != 0
            or cgroup["cgroup_stat"].get("nr_dying_descendants") != 0
            or _require_exact_int_v2(
                cgroup["memory_current"], f"{context} {phase} memory.current"
            )
            > _require_exact_int_v2(
                cgroup["memory_peak"], f"{context} {phase} memory.peak"
            )
            or _require_exact_int_v2(
                cgroup["memory_peak"], f"{context} {phase} memory.peak"
            )
            > auditor.MAXIMUM_ESTIMATED_RSS_BYTES
        ):
            raise ValueError(f"{context} cgroup integer evidence drifted")
        _validate_zero_cgroup_events_v2(
            cgroup["memory_events"],
            f"{context} {phase} memory",
            required_keys=frozenset({"max", "oom", "oom_kill"}),
        )
        _validate_zero_cgroup_events_v2(
            cgroup["pids_events"],
            f"{context} {phase} pids",
            required_keys=frozenset({"max"}),
        )
    if (
        not _exact_json_equal(
            entry_cgroup["cgroup_procs"], exit_cgroup["cgroup_procs"]
        )
        or not _exact_json_equal(
            entry_cgroup["cgroup_stat"], exit_cgroup["cgroup_stat"]
        )
        or
        not _exact_json_equal(
            entry_cgroup["memory_events"], exit_cgroup["memory_events"]
        )
        or not _exact_json_equal(
            entry_cgroup["pids_events"], exit_cgroup["pids_events"]
        )
        or _require_exact_int_v2(
            exit_cgroup["memory_peak"], f"{context} exit memory.peak"
        )
        < _require_exact_int_v2(
            entry_cgroup["memory_peak"], f"{context} entry memory.peak"
        )
    ):
        raise ValueError(f"{context} cgroup entry/exit binding drifted")


def _parse_canonical_finite_hex_float_v2(value: object, name: str) -> float:
    if type(value) is not str:
        raise ValueError(f"{name} is not a hexadecimal float string")
    try:
        parsed = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not a hexadecimal float") from exc
    if not math.isfinite(parsed) or parsed.hex() != value:
        raise ValueError(f"{name} is not a canonical finite hexadecimal float")
    return parsed


def _parse_exact_probability_text_v2(value: object, name: str) -> Fraction:
    if (
        type(value) is not str
        or re.fullmatch(r"(?:0|[1-9][0-9]*)/[1-9][0-9]*", value) is None
    ):
        raise ValueError(f"{name} is not an exact probability text")
    numerator_text, denominator_text = value.split("/", 1)
    probability = Fraction(int(numerator_text), int(denominator_text))
    if (
        not 0 <= probability <= 1
        or value != f"{probability.numerator}/{probability.denominator}"
    ):
        raise ValueError(f"{name} is outside [0,1]")
    return probability


def _validate_known_leak_witness_semantics_v2(
    witness: object,
    *,
    audit_row: dict[str, object],
    target_rows: list[dict[str, object]],
    context: str,
) -> None:
    expected_keys = {
        "balanced_accuracy_advantage_hex",
        "bias_corrected_nmi_hex",
        "failure_witness",
        "feature_aliases",
        "feature_id",
        "feature_localization_rejected",
        "feature_max_t_exceedance_count",
        "feature_max_t_p",
        "feature_owner",
        "feature_owner_path",
        "material_effect",
        "observed_omnibus_hex",
        "passed",
        "raw_nmi_hex",
        "registered_presence_alias",
        "target_family_holm_rejected",
        "target_holm_critical_value",
    }
    if type(witness) is not dict or set(witness) != expected_keys:
        raise ValueError(f"{context} known-leak witness fields are not exact")
    raw_nmi = _parse_canonical_finite_hex_float_v2(
        witness["raw_nmi_hex"], f"{context} leak raw NMI"
    )
    corrected_nmi = _parse_canonical_finite_hex_float_v2(
        witness["bias_corrected_nmi_hex"], f"{context} leak corrected NMI"
    )
    advantage = _parse_canonical_finite_hex_float_v2(
        witness["balanced_accuracy_advantage_hex"],
        f"{context} leak balanced-accuracy advantage",
    )
    observed = _parse_canonical_finite_hex_float_v2(
        witness["observed_omnibus_hex"], f"{context} leak omnibus"
    )
    expected_observed = max(
        corrected_nmi / auditor.NMI_LIMIT,
        max(0.0, advantage) / auditor.BALANCED_ACCURACY_ADVANTAGE_LIMIT,
    )
    feature_p = _parse_exact_probability_text_v2(
        witness["feature_max_t_p"], f"{context} leak feature maxT p"
    )
    feature_exceedance_count = _require_exact_int_v2(
        witness["feature_max_t_exceedance_count"],
        f"{context} leak feature maxT exceedance count",
        maximum=EXACT_LABEL_PERMUTATIONS,
    )
    critical = _parse_exact_probability_text_v2(
        witness["target_holm_critical_value"],
        f"{context} leak target Holm critical",
    )
    binding_rows = [row for row in target_rows if row.get("target") == "binding"]
    if len(binding_rows) != 1:
        raise ValueError(f"{context} binding target row is not unique")
    binding = binding_rows[0]
    binding_observed = _parse_canonical_finite_hex_float_v2(
        binding["observed_target_max_omnibus_hex"],
        f"{context} binding target maximum",
    )
    binding_critical = _parse_exact_probability_text_v2(
        binding["holm_critical_value"], f"{context} binding Holm critical"
    )
    binding_raw = _parse_exact_probability_text_v2(
        binding["raw_p"], f"{context} binding target raw p"
    )
    aliases = witness["feature_aliases"]
    registered = witness["registered_presence_alias"]
    if (
        not 0.0 <= raw_nmi <= 1.0
        or not 0.0 <= corrected_nmi <= raw_nmi
        or not -1.0 <= advantage <= 1.0
        or observed != expected_observed
        or not observed > 1.0
        or binding_observed < observed
        or binding_critical != critical
        or feature_p
        != Fraction(feature_exceedance_count + 1, EXACT_LABEL_PERMUTATIONS + 1)
        or feature_p < binding_raw
        or binding["holm_rejected"] is not True
        or feature_p >= critical
        or witness["material_effect"] is not (observed > 1.0)
        or witness["target_family_holm_rejected"]
        is not binding["holm_rejected"]
        or witness["feature_localization_rejected"]
        is not (binding["holm_rejected"] and feature_p < critical)
        or witness["failure_witness"]
        is not (
            observed > 1.0
            and binding["holm_rejected"]
            and feature_p < critical
        )
        or witness["passed"] is not False
        or witness["feature_owner"] != "eligibility_mask"
        or witness["feature_owner_path"]
        != _json_path_text_v2(_REGISTERED_BINDING_LEAK_OWNER_PATH_V2)
        or type(aliases) is not list
        or not aliases
        or any(type(alias) is not str or not alias for alias in aliases)
        or len(set(aliases)) != len(aliases)
        or type(registered) is not str
        or registered not in aliases
        or not _is_registered_binding_leak_alias_v2(registered)
        or type(witness["feature_id"]) is not str
        or not witness["feature_id"]
        or _require_exact_int_v2(
            audit_row.get("failure_witness_count"),
            f"{context} exact known-leak failure witness count",
            minimum=1,
            maximum=1,
        )
        != 1
    ):
        raise ValueError(f"{context} known-leak E1 numeric witness drifted")


def _validate_target_family_holm_projection_v2(
    target_rows: object, *, context: str
) -> dict[str, dict[str, object]]:
    keys = {
        "exceedance_count",
        "holm_adjusted_p",
        "holm_critical_value",
        "holm_rank",
        "holm_rejected",
        "observed_target_max_omnibus_hex",
        "raw_p",
        "target",
    }
    if (
        type(target_rows) is not list
        or len(target_rows) != len(auditor.TARGETS)
        or any(type(row) is not dict or set(row) != keys for row in target_rows)
        or [row["target"] for row in target_rows] != list(auditor.TARGETS)
    ):
        raise ValueError(f"{context} target-family rows/order are not exact")
    raw_values: list[Fraction] = []
    adjusted_values: list[Fraction] = []
    critical_values: list[Fraction] = []
    exceedances: list[int] = []
    for row in target_rows:
        exceedance = _require_exact_int_v2(
            row["exceedance_count"],
            f"{context} target exceedance count",
            maximum=EXACT_LABEL_PERMUTATIONS,
        )
        observed = _parse_canonical_finite_hex_float_v2(
            row["observed_target_max_omnibus_hex"],
            f"{context} observed target maximum",
        )
        if observed < 0.0 or type(row["holm_rejected"]) is not bool:
            raise ValueError(f"{context} target statistic/decision drifted")
        exceedances.append(exceedance)
        raw_values.append(
            _parse_exact_probability_text_v2(
                row["raw_p"], f"{context} target raw p"
            )
        )
        adjusted_values.append(
            _parse_exact_probability_text_v2(
                row["holm_adjusted_p"], f"{context} target adjusted p"
            )
        )
        critical_values.append(
            _parse_exact_probability_text_v2(
                row["holm_critical_value"], f"{context} target critical"
            )
        )
    order = sorted(
        range(len(raw_values)), key=lambda index: (raw_values[index], index)
    )
    ranks = [0] * len(raw_values)
    adjusted_expected = [Fraction(0, 1)] * len(raw_values)
    running = Fraction(0, 1)
    for zero_rank, index in enumerate(order):
        ranks[index] = zero_rank + 1
        running = max(
            running,
            min(Fraction(1, 1), (len(raw_values) - zero_rank) * raw_values[index]),
        )
        adjusted_expected[index] = running
    for index, row in enumerate(target_rows):
        critical_expected = Fraction(
            1, 100 * (len(raw_values) - ranks[index] + 1)
        )
        if (
            raw_values[index]
            != Fraction(exceedances[index] + 1, EXACT_LABEL_PERMUTATIONS + 1)
            or _require_exact_int_v2(
                row["holm_rank"], f"{context} target Holm rank", minimum=1
            )
            != ranks[index]
            or adjusted_values[index] != adjusted_expected[index]
            or critical_values[index] != critical_expected
            or row["holm_rejected"]
            is not (adjusted_expected[index] < Fraction(1, 100))
        ):
            raise ValueError(f"{context} target-family B+1/Holm replay drifted")
    return {row["target"]: row for row in target_rows}


def _validate_two_run_hard_contract_v2(
    measurements: object,
    runs: object,
    *,
    context: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if (
        type(measurements) is not list
        or type(runs) is not list
        or len(measurements) != 2
        or len(runs) != 2
    ):
        raise ValueError(f"{context} does not contain exactly two modes")
    typed_measurements: list[dict[str, object]] = []
    typed_runs: list[dict[str, object]] = []
    quota_targets = {"family", "scale", "answerable_vs_abstain"}
    target_result_keys = {
        "exceedance_count",
        "holm_adjusted_p",
        "holm_critical_value",
        "holm_rank",
        "holm_rejected",
        "observed_target_max_omnibus_hex",
        "raw_p",
        "target",
    }
    for index, mode in enumerate(LIVE_MODE_ORDER):
        run = runs[index]
        measurement = measurements[index]
        if (
            type(run) is not dict
            or type(measurement) is not dict
            or run.get("mode") != mode
            or measurement.get("mode") != mode
        ):
            raise ValueError(f"{context} mode order drifted")
        typed_runs.append(run)
        typed_measurements.append(measurement)
        _validate_run_and_measurement_binding_v2(run, measurement)
        audit_row = run.get("audit")
        if type(audit_row) is not dict:
            raise ValueError(f"{context} audit projection is absent")
        invariant = audit_row.get("invariance")
        statistics = audit_row.get("statistics")
        receipt_issuance = audit_row.get("receipt_issuance")
        resource = audit_row.get("resource_plan")
        quota = audit_row.get("eligibility_quota")
        target_rows = audit_row.get("target_family_max_t_holm")
        if (
            type(invariant) is not dict
            or type(resource) is not dict
            or type(statistics) is not dict
            or type(receipt_issuance) is not dict
        ):
            raise ValueError(f"{context} executed runtime bundle is absent")
        _validate_executed_runtime_bundle_v2(
            resource_plan=resource,
            statistics_runtime=statistics.get("complete_runtime_projection"),
            receipt_issuance_runtime=receipt_issuance.get(
                "complete_runtime_projection"
            ),
            diagnostics=run.get("diagnostics"),
            run=run,
            measurement=measurement,
            expected_mode=mode,
            visible_audited_rows=EXACT_AUTHORITY_ROWS,
            visible_concrete_features=(
                _fixture_feature_capacity_projection_row_v2(
                    mode,
                    EXACT_AUTHORITY_ROWS,
                )["projected_concrete_feature_count"]
            ),
            context=f"{context} two-run {mode}",
        )
        if (
            resource.get("admissible") is not True
            or statistics.get("external_supervisor_required") is not True
            or statistics.get("pairwise_holm_decision_eligible") is not False
            or statistics.get("within_frozen_physical_peak_cap") is not True
            or type(receipt_issuance) is not dict
            or receipt_issuance.get("absolute_deadline_kind") != "STATISTICS"
            or receipt_issuance.get(
                "receipt_finalization_complete_pending_validation_count"
            )
            != 1
            or receipt_issuance.get(
                "receipt_finalization_locked_deadline_check_count"
            )
            != 1
            or receipt_issuance.get(
                "receipt_finalization_locked_deadline_within_absolute_limit"
            )
            is not True
            or receipt_issuance.get(
                "receipt_finalization_locked_physical_peak_within_cap"
            )
            is not True
            or type(quota) is not dict
            or quota.get("exact_quota_matched") is not True
            or type(target_rows) is not list
            or len(target_rows) != len(auditor.TARGETS)
            or audit_row.get("actual_corpus_audit") is not False
            or audit_row.get("authority_origin_authenticated") is not False
            or audit_row.get("authority_independence_verified") is not False
        ):
            raise ValueError(f"{context} audit hard coverage gate drifted")
        _validate_target_family_holm_projection_v2(
            target_rows, context=context
        )
        if (
            _require_exact_int_v2(
                audit_row.get("audited_rows"), f"{context} audited rows", minimum=1
            )
            != EXACT_AUTHORITY_ROWS
            or _require_exact_int_v2(
                invariant.get("run_count"),
                f"{context} invariant run count",
                minimum=1,
            )
            != EXACT_RECOGNIZER_INVOCATIONS
            or _require_exact_int_v2(
                invariant.get("exact_comparisons"),
                f"{context} invariant comparisons",
                minimum=1,
            )
            != EXACT_AUTHORITY_ROWS * 80 * 5
            or _require_exact_int_v2(
                invariant.get("consumer_decode_count"),
                f"{context} consumer decode count",
                minimum=1,
            )
            != EXACT_AUTHORITY_ROWS * 80
            or _require_exact_int_v2(
                resource.get("label_permutations"),
                f"{context} label permutations",
                minimum=1,
            )
            != EXACT_LABEL_PERMUTATIONS
        ):
            raise ValueError(f"{context} exact audit counts drifted")
        invariant_complete = invariant.get("complete_runtime_projection")
        statistics_complete = statistics.get("complete_runtime_projection")
        receipt_issuance_complete = receipt_issuance.get(
            "complete_runtime_projection"
        )
        if (
            type(invariant_complete) is not dict
            or invariant_complete.get("process_group_empty_at_exit") is not True
            or invariant_complete.get("within_frozen_physical_peak_cap") is not True
            or type(statistics_complete) is not dict
            or statistics_complete.get("external_supervisor_required") is not True
            or statistics_complete.get("within_frozen_physical_peak_cap") is not True
            or type(receipt_issuance_complete) is not dict
            or receipt_issuance_complete.get("absolute_deadline_kind")
            != "STATISTICS"
            or receipt_issuance_complete.get(
                "receipt_finalization_locked_deadline_within_absolute_limit"
            )
            is not True
            or receipt_issuance_complete.get(
                "receipt_finalization_locked_physical_peak_within_cap"
            )
            is not True
        ):
            raise ValueError(f"{context} runtime containment evidence drifted")
        quota_rows = quota.get("targets")
        if (
            type(quota_rows) is not list
            or len(quota_rows) != len(quota_targets)
            or {
                row.get("target")
                for row in quota_rows
                if type(row) is dict
            }
            != quota_targets
        ):
            raise ValueError(f"{context} eligibility quota targets drifted")
        for row in quota_rows:
            if (
                type(row) is not dict
                or set(row)
                != {
                    "checked_integer_cells",
                    "exact_quota_matched",
                    "maximum_absolute_cross_product_difference",
                    "target",
                }
                or row["exact_quota_matched"] is not True
                or _require_exact_int_v2(
                    row["checked_integer_cells"],
                    f"{context} quota checked cells",
                    minimum=1,
                )
                < 1
                or _require_exact_int_v2(
                    row["maximum_absolute_cross_product_difference"],
                    f"{context} quota maximum difference",
                )
                != 0
            ):
                raise ValueError(f"{context} eligibility quota row drifted")
        seen_target_rows: set[str] = set()
        seen_holm_ranks: set[int] = set()
        raw_probabilities: list[Fraction] = []
        adjusted_probabilities: list[Fraction] = []
        critical_probabilities: list[Fraction] = []
        observed_target_maxima: list[float] = []
        if [row.get("target") for row in target_rows if type(row) is dict] != list(
            auditor.TARGETS
        ):
            raise ValueError(f"{context} target maxT order drifted")
        for target_row in target_rows:
            if type(target_row) is not dict or set(target_row) != target_result_keys:
                raise ValueError(f"{context} target maxT row is not closed")
            target_name = target_row["target"]
            holm_rank = _require_exact_int_v2(
                target_row["holm_rank"],
                f"{context} target Holm rank",
                minimum=1,
                maximum=len(auditor.TARGETS),
            )
            if (
                type(target_name) is not str
                or target_name not in auditor.TARGETS
                or target_name in seen_target_rows
                or holm_rank in seen_holm_ranks
                or type(target_row["holm_rejected"]) is not bool
                or _require_exact_int_v2(
                    target_row["exceedance_count"],
                    f"{context} target exceedance count",
                    maximum=EXACT_LABEL_PERMUTATIONS,
                )
                > EXACT_LABEL_PERMUTATIONS
            ):
                raise ValueError(f"{context} target maxT/Holm evidence drifted")
            raw = _parse_exact_probability_text_v2(
                target_row["raw_p"], f"{context} target raw p"
            )
            adjusted = _parse_exact_probability_text_v2(
                target_row["holm_adjusted_p"], f"{context} target Holm adjusted p"
            )
            critical = _parse_exact_probability_text_v2(
                target_row["holm_critical_value"], f"{context} target Holm critical"
            )
            observed_maximum = _parse_canonical_finite_hex_float_v2(
                target_row["observed_target_max_omnibus_hex"],
                f"{context} observed target maximum",
            )
            if observed_maximum < 0.0:
                raise ValueError(f"{context} observed target maximum is negative")
            raw_probabilities.append(raw)
            adjusted_probabilities.append(adjusted)
            critical_probabilities.append(critical)
            observed_target_maxima.append(observed_maximum)
            seen_target_rows.add(target_name)
            seen_holm_ranks.add(holm_rank)
        order = sorted(
            range(len(raw_probabilities)),
            key=lambda row_index: (raw_probabilities[row_index], row_index),
        )
        expected_ranks = [0] * len(raw_probabilities)
        expected_adjusted: list[Fraction] = [Fraction(0, 1)] * len(
            raw_probabilities
        )
        running_adjusted = Fraction(0, 1)
        for zero_rank, row_index in enumerate(order):
            expected_ranks[row_index] = zero_rank + 1
            running_adjusted = max(
                running_adjusted,
                min(
                    Fraction(1, 1),
                    (len(raw_probabilities) - zero_rank)
                    * raw_probabilities[row_index],
                ),
            )
            expected_adjusted[row_index] = running_adjusted
        for row_index, target_row in enumerate(target_rows):
            exceedance = _require_exact_int_v2(
                target_row["exceedance_count"],
                f"{context} target exceedance replay",
                maximum=EXACT_LABEL_PERMUTATIONS,
            )
            rank = expected_ranks[row_index]
            expected_critical = Fraction(
                1, 100 * (len(raw_probabilities) - rank + 1)
            )
            if (
                raw_probabilities[row_index]
                != Fraction(exceedance + 1, EXACT_LABEL_PERMUTATIONS + 1)
                or target_row["holm_rank"] != rank
                or adjusted_probabilities[row_index]
                != expected_adjusted[row_index]
                or critical_probabilities[row_index] != expected_critical
                or target_row["holm_rejected"]
                is not (expected_adjusted[row_index] < Fraction(1, 100))
            ):
                raise ValueError(f"{context} five-target maxT/Holm replay drifted")
        if mode == "KNOWN_NULL":
            if (
                run.get("disposition") != "PASS"
                or _require_exact_int_v2(
                    audit_row.get("failure_witness_count"),
                    f"{context} known-null failure witness count",
                )
                != 0
                or audit_row.get("positive_decision_claim")
                != auditor.POSITIVE_DECISION_CLAIM
                or run.get("known_leak_witness") is not None
            ):
                raise ValueError(f"{context} known-null claim polarity drifted")
        else:
            witness = run.get("known_leak_witness")
            _validate_known_leak_witness_semantics_v2(
                witness,
                audit_row=audit_row,
                target_rows=target_rows,
                context=context,
            )
            binding_rows = [
                row for row in target_rows if row.get("target") == "binding"
            ]
            if (
                run.get("disposition") != "FAIL"
                or type(witness) is not dict
                or witness.get("feature_owner") != "eligibility_mask"
                or witness.get("feature_owner_path")
                != _json_path_text_v2(_REGISTERED_BINDING_LEAK_OWNER_PATH_V2)
                or not _is_registered_binding_leak_alias_v2(
                    witness.get("registered_presence_alias")
                )
                or witness.get("material_effect") is not True
                or witness.get("target_family_holm_rejected") is not True
                or witness.get("feature_localization_rejected") is not True
                or witness.get("failure_witness") is not True
                or witness.get("passed") is not False
                or type(witness.get("feature_aliases")) is not list
                or witness.get("registered_presence_alias")
                not in witness["feature_aliases"]
                or len(binding_rows) != 1
                or binding_rows[0]["holm_rejected"] is not True
                or _require_exact_int_v2(
                    audit_row.get("failure_witness_count"),
                    f"{context} known-leak failure witness count",
                    minimum=1,
                    maximum=1,
                )
                != 1
            ):
                raise ValueError(f"{context} known-leak E1 witness drifted")
    return typed_measurements, typed_runs


def _validate_live_inner_artifact_v2(
    body: dict[str, object],
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
) -> None:
    expected_keys = {
        "attempt_id",
        "attempt_manifest_root",
        "campaign_id",
        "claim_level",
        "claims",
        "content_sha256_is_not_a_signature",
        "external_finalizer_required",
        "fresh_static_measurements_in_order",
        "frozen_static_shape_exact_match",
        "mode_order",
        "outer_hard_deadline_self_verified",
        "profile_id",
        "post_output_physical_gate_contract",
        "projection_only_boundary",
        "runs_in_order",
        "runtime_entry_evidence",
        "runtime_exit_evidence",
        "schema_version",
        "synthetic_fixture_root",
    }
    if set(body) != expected_keys:
        raise ValueError("live inner artifact root fields are not exact")
    if (
        body["schema_version"]
        != "hegel-machine-phase2b-formal-wire-covert-live-inner-candidate/6"
        or body["profile_id"] != attempt.profile_id
        or body["attempt_manifest_root"] != attempt.attempt_manifest_root
        or body["campaign_id"] != attempt.campaign_id
        or body["attempt_id"] != attempt.attempt_id
        or body["mode_order"] != list(LIVE_MODE_ORDER)
        or body["claim_level"] != "INNER_CANDIDATE_NOT_R03_R04_R05"
        or body["frozen_static_shape_exact_match"] is not True
        or body["outer_hard_deadline_self_verified"] is not False
        or body["external_finalizer_required"] is not True
        or body["content_sha256_is_not_a_signature"] is not True
        or not _exact_json_equal(
            body["post_output_physical_gate_contract"],
            _POST_OUTPUT_PHYSICAL_GATE_CONTRACT_V2,
        )
        or not _exact_json_equal(
            body["projection_only_boundary"], _TASK4_PROJECTION_ONLY_BOUNDARY_V2
        )
    ):
        raise ValueError("live inner artifact identity/claim boundary drifted")
    expected_claims = {
        "r03_formal_trusted_wire_implementation_qualified": False,
        "r04_formal_covert_auditor_implementation_qualified": False,
        "r05_synthetic_dev_wire_covert_qualification_verified": False,
        "actual_corpus_audit": False,
        "actual_unsealed_960_generation": False,
        "origin_authenticated": False,
        "key_independence_verified": False,
        "campaign_execution_authorized": False,
        "fixture_specific_only": True,
        "joint_decision_classes": 2,
        "wire_wider_graphs_and_100_class_profile_qualified": False,
    }
    if not _exact_json_equal(body["claims"], expected_claims):
        raise ValueError("live inner artifact claim boundary drifted")
    entry = body["runtime_entry_evidence"]
    exit_row = body["runtime_exit_evidence"]
    if type(entry) is not dict or type(exit_row) is not dict:
        raise ValueError("live inner runtime entry/exit evidence is absent")
    _validate_runtime_entry_exit_contract_v2(
        entry, exit_row, attempt, context="live inner"
    )
    _validate_interpreter_runtime_projection_v2(
        entry.get("interpreter_runtime"), "live inner"
    )
    entry_cgroup = entry.get("cgroup")
    exit_cgroup = exit_row.get("cgroup")
    if type(entry_cgroup) is not dict or type(exit_cgroup) is not dict:
        raise ValueError("live inner cgroup entry/exit evidence is absent")
    if (
        _require_exact_int_v2(entry.get("threads"), "inner entry threads") != 1
        or _require_exact_int_v2(exit_row.get("threads"), "inner exit threads")
        != 1
        or entry.get("sigblk_hex") != "0000000000000000"
        or exit_row.get("sigblk_hex") != "0000000000000000"
        or exit_row.get("cgroup_events_unchanged_from_entry") is not True
        or exit_row.get("audit_receipts_validated_and_released_in_process")
        is not True
        or entry_cgroup.get("cgroup_path") != exit_cgroup.get("cgroup_path")
        or entry_cgroup.get("memory_max")
        != attempt.requested_cgroup_memory_max_bytes
        or exit_cgroup.get("memory_max")
        != attempt.requested_cgroup_memory_max_bytes
        or entry_cgroup.get("pids_max") != attempt.requested_cgroup_pids_max
        or exit_cgroup.get("pids_max") != attempt.requested_cgroup_pids_max
    ):
        raise ValueError("live inner runtime/cgroup evidence drifted")
    for name, cgroup in (("entry", entry_cgroup), ("exit", exit_cgroup)):
        if (
            _require_exact_int_v2(
                cgroup.get("memory_max"), f"inner {name} memory.max", minimum=1
            )
            != attempt.requested_cgroup_memory_max_bytes
            or _require_exact_int_v2(
                cgroup.get("pids_max"), f"inner {name} pids.max", minimum=1
            )
            != attempt.requested_cgroup_pids_max
            or _require_exact_int_v2(
                cgroup.get("pids_current"), f"inner {name} pids.current"
            )
            != 1
            or _require_exact_int_v2(
                cgroup.get("memory_current"), f"inner {name} memory.current"
            )
            > _require_exact_int_v2(
                cgroup.get("memory_peak"), f"inner {name} memory.peak"
            )
            or _require_exact_int_v2(
                cgroup.get("memory_peak"), f"inner {name} memory.peak"
            )
            > auditor.MAXIMUM_ESTIMATED_RSS_BYTES
        ):
            raise ValueError("live inner cgroup integer evidence drifted")
        _validate_zero_cgroup_events_v2(
            cgroup.get("memory_events"),
            f"inner {name} memory",
            required_keys=frozenset({"max", "oom", "oom_kill"}),
        )
        _validate_zero_cgroup_events_v2(
            cgroup.get("pids_events"),
            f"inner {name} pids",
            required_keys=frozenset({"max"}),
        )
    if (
        not _exact_json_equal(
            entry_cgroup.get("memory_events"), exit_cgroup.get("memory_events")
        )
        or not _exact_json_equal(
            entry_cgroup.get("pids_events"), exit_cgroup.get("pids_events")
        )
        or entry.get("selected_default_signals")
        != [number.name for number in LIVE_SELECTED_DEFAULT_SIGNALS]
        or exit_row.get("selected_default_signals")
        != [number.name for number in LIVE_SELECTED_DEFAULT_SIGNALS]
    ):
        raise ValueError("live inner signal/event evidence drifted")
    shape = _EXPECTED_PROFILE["fixture_contract"]["rich_positive_shape"]
    if type(shape) is not dict:
        raise ValueError("live inner verifier cannot find the frozen shape")
    measurements = body["fresh_static_measurements_in_order"]
    if type(measurements) is not list or len(measurements) != 2:
        raise ValueError("live inner static measurements do not cover two modes")
    for index, mode in enumerate(LIVE_MODE_ORDER):
        measurement = measurements[index]
        if (
            type(measurement) is not dict
            or measurement.get("mode") != mode
            or not _exact_json_equal(
                measurement.get("frozen_static_projection"),
                shape.get(
                    "known_null_frozen_static_projection"
                    if mode == "KNOWN_NULL"
                    else "known_leak_frozen_static_projection"
                ),
            )
        ):
            raise ValueError("live inner static measurement differs from profile")
    runs = body["runs_in_order"]
    _validate_two_run_hard_contract_v2(
        measurements, runs, context="live inner"
    )
    if type(runs) is not list or len(runs) != 2:
        raise ValueError("live inner artifact does not contain exactly two runs")
    _validate_receipt_physical_scope_against_exit_cgroup_v2(
        runs,
        entry_cgroup,
        exit_cgroup,
        context="live inner",
    )
    for index, mode in enumerate(LIVE_MODE_ORDER):
        run = runs[index]
        if type(run) is not dict or run.get("mode") != mode:
            raise ValueError("live inner audit order drifted")
        measurement = measurements[index]
        if type(measurement) is not dict:
            raise ValueError("live inner measurement is not an exact object")
        _validate_run_and_measurement_binding_v2(run, measurement)
        audit_row = run.get("audit")
        if type(audit_row) is not dict:
            raise ValueError("live inner audit projection is absent")
        invariant = audit_row.get("invariance")
        statistics = audit_row.get("statistics")
        resource = audit_row.get("resource_plan")
        quota = audit_row.get("eligibility_quota")
        target_rows = audit_row.get("target_family_max_t_holm")
        if (
            type(invariant) is not dict
            or type(resource) is not dict
            or resource.get("admissible") is not True
            or type(statistics) is not dict
            or statistics.get("external_supervisor_required") is not True
            or statistics.get("pairwise_holm_decision_eligible") is not False
            or statistics.get("within_frozen_physical_peak_cap") is not True
            or type(quota) is not dict
            or quota.get("exact_quota_matched") is not True
            or type(target_rows) is not list
            or len(target_rows) != len(auditor.TARGETS)
            or {row.get("target") for row in target_rows if type(row) is dict}
            != set(auditor.TARGETS)
            or audit_row.get("actual_corpus_audit") is not False
            or audit_row.get("authority_origin_authenticated") is not False
            or audit_row.get("authority_independence_verified") is not False
        ):
            raise ValueError("live inner audit hard coverage gate drifted")
        if (
            _require_exact_int_v2(
                audit_row.get("audited_rows"), "inner audited rows", minimum=1
            )
            != EXACT_AUTHORITY_ROWS
            or _require_exact_int_v2(
                invariant.get("run_count"), "inner invariant run count", minimum=1
            )
            != EXACT_RECOGNIZER_INVOCATIONS
            or _require_exact_int_v2(
                invariant.get("exact_comparisons"),
                "inner exact invariant comparisons",
                minimum=1,
            )
            != EXACT_AUTHORITY_ROWS * 80 * 5
            or _require_exact_int_v2(
                invariant.get("consumer_decode_count"),
                "inner consumer decode count",
                minimum=1,
            )
            != EXACT_AUTHORITY_ROWS * 80
            or _require_exact_int_v2(
                resource.get("label_permutations"),
                "inner label permutations",
                minimum=1,
            )
            != EXACT_LABEL_PERMUTATIONS
        ):
            raise ValueError("live inner exact audit counts drifted")
        for target_row in target_rows:
            if (
                type(target_row) is not dict
                or _require_exact_int_v2(
                    target_row.get("exceedance_count"),
                    "inner target exceedance count",
                    maximum=EXACT_LABEL_PERMUTATIONS,
                )
                > EXACT_LABEL_PERMUTATIONS
            ):
                raise ValueError("live inner target maxT evidence is malformed")
        invariant_complete = invariant.get("complete_runtime_projection")
        if (
            type(invariant_complete) is not dict
            or invariant_complete.get("process_group_empty_at_exit") is not True
            or invariant_complete.get("within_frozen_physical_peak_cap") is not True
        ):
            raise ValueError("live inner invariance containment evidence drifted")
        if mode == "KNOWN_NULL":
            if (
                run.get("disposition") != "PASS"
                or _require_exact_int_v2(
                    audit_row.get("failure_witness_count"),
                    "known-null failure witness count",
                )
                != 0
                or audit_row.get("positive_decision_claim")
                != auditor.POSITIVE_DECISION_CLAIM
                or run.get("known_leak_witness") is not None
            ):
                raise ValueError("live known-null claim polarity drifted")
        else:
            witness = run.get("known_leak_witness")
            if (
                run.get("disposition") != "FAIL"
                or type(witness) is not dict
                or witness.get("feature_owner") != "eligibility_mask"
                or witness.get("feature_owner_path")
                != _json_path_text_v2(_REGISTERED_BINDING_LEAK_OWNER_PATH_V2)
                or not _is_registered_binding_leak_alias_v2(
                    witness.get("registered_presence_alias")
                )
                or witness.get("material_effect") is not True
                or witness.get("target_family_holm_rejected") is not True
                or witness.get("feature_localization_rejected") is not True
                or witness.get("failure_witness") is not True
                or witness.get("passed") is not False
                or type(witness.get("feature_aliases")) is not list
                or witness.get("registered_presence_alias")
                not in witness["feature_aliases"]
            ):
                raise ValueError("live known-leak E1 witness drifted")
    expected_fixture_root = _combined_fixture_root_from_run_roots_v2(
        runs[0]["run_fixture_root"], runs[1]["run_fixture_root"]
    )
    if (
        body["synthetic_fixture_root"] != expected_fixture_root
        or body["synthetic_fixture_root"] != shape.get("combined_fixture_root")
    ):
        raise ValueError("live inner combined synthetic fixture root drifted")


def _validate_development_calibration_artifact_v2(
    body: dict[str, object],
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
) -> None:
    if set(body) != {
        "artifact_cannot_be_finalized_or_promoted",
        "attempt_manifest_root",
        "calibration_attempt_id",
        "calibration_campaign_id",
        "claim_level",
        "claims",
        "content_sha256_is_not_a_signature",
        "external_supervisor_evidence_required",
        "frozen_static_measurement_candidates_in_order",
        "independent_review_and_profile_update_required",
        "mode_order",
        "outer_hard_deadline_self_verified",
        "profile_id",
        "post_output_physical_gate_contract",
        "projection_only_boundary",
        "runs_in_order",
        "runtime_entry_evidence",
        "runtime_exit_evidence",
        "same_run_shape_acceptance_forbidden",
        "schema_version",
        "synthetic_fixture_root",
    }:
        raise ValueError("calibration inner artifact root fields are not exact")
    if (
        body["schema_version"]
        != "hegel-machine-phase2b-formal-wire-covert-development-calibration/6"
        or body["profile_id"] != attempt.profile_id
        or body["attempt_manifest_root"] != attempt.attempt_manifest_root
        or body["calibration_campaign_id"] != attempt.campaign_id
        or body["calibration_attempt_id"] != attempt.attempt_id
        or body["mode_order"] != list(LIVE_MODE_ORDER)
        or body["claim_level"]
        != "DEVELOPMENT_MEASUREMENT_ONLY_NOT_R03_R04_R05"
        or body["artifact_cannot_be_finalized_or_promoted"] is not True
        or body["outer_hard_deadline_self_verified"] is not False
        or body["external_supervisor_evidence_required"] is not True
        or body["independent_review_and_profile_update_required"] is not True
        or body["same_run_shape_acceptance_forbidden"] is not True
        or body["content_sha256_is_not_a_signature"] is not True
        or not _exact_json_equal(
            body["post_output_physical_gate_contract"],
            _POST_OUTPUT_PHYSICAL_GATE_CONTRACT_V2,
        )
        or not _exact_json_equal(
            body["projection_only_boundary"], _TASK4_PROJECTION_ONLY_BOUNDARY_V2
        )
    ):
        raise ValueError("calibration inner identity/claim boundary drifted")
    claims = body["claims"]
    if (
        type(claims) is not dict
        or claims.get("r03_formal_trusted_wire_implementation_qualified")
        is not False
        or claims.get("r04_formal_covert_auditor_implementation_qualified")
        is not False
        or claims.get("r05_synthetic_dev_wire_covert_qualification_verified")
        is not False
        or claims.get("actual_corpus_audit") is not False
        or claims.get("actual_unsealed_960_generation") is not False
        or claims.get("origin_authenticated") is not False
        or claims.get("key_independence_verified") is not False
        or claims.get("campaign_execution_authorized") is not False
        or claims.get("fixture_specific_only") is not True
        or _require_exact_int_v2(
            claims.get("joint_decision_classes"),
            "calibration joint decision classes",
            minimum=1,
        )
        != 2
        or claims.get("wire_wider_graphs_and_100_class_profile_qualified")
        is not False
        or len(claims) != 11
    ):
        raise ValueError("calibration inner claim projection drifted")
    entry = body["runtime_entry_evidence"]
    exit_row = body["runtime_exit_evidence"]
    if type(entry) is not dict or type(exit_row) is not dict:
        raise ValueError("calibration runtime evidence is absent")
    _validate_runtime_entry_exit_contract_v2(
        entry, exit_row, attempt, context="development calibration"
    )
    _validate_interpreter_runtime_projection_v2(
        entry.get("interpreter_runtime"), "calibration"
    )
    entry_cgroup = entry.get("cgroup")
    exit_cgroup = exit_row.get("cgroup")
    if type(entry_cgroup) is not dict or type(exit_cgroup) is not dict:
        raise ValueError("calibration cgroup evidence is absent")
    for phase, row in (("entry", entry_cgroup), ("exit", exit_cgroup)):
        if (
            _require_exact_int_v2(
                row.get("memory_max"), f"calibration {phase} memory.max", minimum=1
            )
            != attempt.requested_cgroup_memory_max_bytes
            or _require_exact_int_v2(
                row.get("pids_max"), f"calibration {phase} pids.max", minimum=1
            )
            != attempt.requested_cgroup_pids_max
            or _require_exact_int_v2(
                row.get("pids_current"), f"calibration {phase} pids.current"
            )
            != 1
            or _require_exact_int_v2(
                row.get("memory_peak"), f"calibration {phase} memory.peak"
            )
            > auditor.MAXIMUM_ESTIMATED_RSS_BYTES
        ):
            raise ValueError("calibration cgroup integer evidence drifted")
        _validate_zero_cgroup_events_v2(
            row.get("memory_events"),
            f"calibration {phase} memory",
            required_keys=frozenset({"max", "oom", "oom_kill"}),
        )
        _validate_zero_cgroup_events_v2(
            row.get("pids_events"),
            f"calibration {phase} pids",
            required_keys=frozenset({"max"}),
        )
    if (
        entry_cgroup.get("cgroup_path") != exit_cgroup.get("cgroup_path")
        or not _exact_json_equal(
            entry_cgroup.get("memory_events"), exit_cgroup.get("memory_events")
        )
        or not _exact_json_equal(
            entry_cgroup.get("pids_events"), exit_cgroup.get("pids_events")
        )
        or exit_row.get("audit_receipts_validated_and_released_in_process")
        is not True
        or exit_row.get("cgroup_events_unchanged_from_entry") is not True
    ):
        raise ValueError("calibration cgroup entry/exit binding drifted")
    measurements = body["frozen_static_measurement_candidates_in_order"]
    runs = body["runs_in_order"]
    _validate_two_run_hard_contract_v2(
        measurements, runs, context="development calibration"
    )
    if (
        type(measurements) is not list
        or type(runs) is not list
        or len(measurements) != 2
        or len(runs) != 2
    ):
        raise ValueError("calibration does not contain exactly two modes")
    _validate_receipt_physical_scope_against_exit_cgroup_v2(
        runs,
        entry_cgroup,
        exit_cgroup,
        context="development calibration",
    )
    for index, mode in enumerate(LIVE_MODE_ORDER):
        run = runs[index]
        measurement = measurements[index]
        if (
            type(run) is not dict
            or type(measurement) is not dict
            or run.get("mode") != mode
        ):
            raise ValueError("calibration mode order drifted")
        _validate_run_and_measurement_binding_v2(run, measurement)
        audit_row = run.get("audit")
        if type(audit_row) is not dict:
            raise ValueError("calibration audit projection is absent")
        invariant = audit_row.get("invariance")
        resource = audit_row.get("resource_plan")
        if (
            type(invariant) is not dict
            or type(resource) is not dict
            or _require_exact_int_v2(
                audit_row.get("audited_rows"), "calibration audited rows", minimum=1
            )
            != EXACT_AUTHORITY_ROWS
            or _require_exact_int_v2(
                invariant.get("run_count"), "calibration invariant runs", minimum=1
            )
            != EXACT_RECOGNIZER_INVOCATIONS
            or _require_exact_int_v2(
                invariant.get("exact_comparisons"),
                "calibration invariant comparisons",
                minimum=1,
            )
            != EXACT_AUTHORITY_ROWS * 80 * 5
            or _require_exact_int_v2(
                invariant.get("consumer_decode_count"),
                "calibration consumer decodes",
                minimum=1,
            )
            != EXACT_AUTHORITY_ROWS * 80
            or resource.get("admissible") is not True
        ):
            raise ValueError("calibration audit coverage drifted")
        if mode == "KNOWN_NULL":
            if (
                run.get("disposition") != "PASS"
                or _require_exact_int_v2(
                    audit_row.get("failure_witness_count"),
                    "calibration null failure witnesses",
                )
                != 0
                or audit_row.get("positive_decision_claim")
                != auditor.POSITIVE_DECISION_CLAIM
            ):
                raise ValueError("calibration known-null claim polarity drifted")
        else:
            witness = run.get("known_leak_witness")
            if (
                run.get("disposition") != "FAIL"
                or type(witness) is not dict
                or witness.get("feature_owner") != "eligibility_mask"
                or witness.get("feature_owner_path")
                != _json_path_text_v2(_REGISTERED_BINDING_LEAK_OWNER_PATH_V2)
                or not _is_registered_binding_leak_alias_v2(
                    witness.get("registered_presence_alias")
                )
                or witness.get("failure_witness") is not True
                or witness.get("target_family_holm_rejected") is not True
                or witness.get("feature_localization_rejected") is not True
            ):
                raise ValueError("calibration known-leak E1 evidence drifted")
    expected_fixture_root = _combined_fixture_root_from_run_roots_v2(
        runs[0]["run_fixture_root"], runs[1]["run_fixture_root"]
    )
    if body["synthetic_fixture_root"] != expected_fixture_root:
        raise ValueError("calibration combined fixture root drifted")


_CALIBRATION_PAIR_REPLAY_TOKEN_ISSUER_V2 = object()


def _calibration_pair_payload_root_v2(
    payloads: tuple[bytes, bytes, bytes, bytes, bytes, bytes],
) -> str:
    if (
        type(payloads) is not tuple
        or len(payloads) != 6
        or any(type(payload) is not bytes for payload in payloads)
    ):
        raise TypeError("calibration pair payloads must be six exact byte strings")
    return stable_hash(
        [
            {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for payload in payloads
        ],
        prefix="phase2b_formal_wire_covert_verified_calibration_pair_v2_",
    )


class _VerifiedCalibrationPairReplayTokenV2:
    """Opaque proof that the retained development attempt replay succeeded."""

    __slots__ = ()

    def __init__(self, issuer: object) -> None:
        if issuer is not _CALIBRATION_PAIR_REPLAY_TOKEN_ISSUER_V2:
            raise TypeError("calibration pair tokens are issued only by retained replay")


_CALIBRATION_PAIR_REPLAY_TOKEN_REGISTRY_V2: dict[
    _VerifiedCalibrationPairReplayTokenV2, tuple[str, str, str]
] = {}


def _calibration_pair_projection_root_v2(projection: object) -> str:
    closed = _validate_calibration_pair_projection_shape_v2(
        projection, required=True
    )
    if type(closed) is not dict:
        raise TypeError("verified calibration pair projection is absent")
    return stable_hash(
        closed,
        prefix="phase2b_formal_wire_covert_verified_pair_projection_v2_",
    )


def _issue_verified_calibration_pair_replay_token_v2(
    issuer: object,
    payloads: tuple[bytes, bytes, bytes, bytes, bytes, bytes],
    pair_projection: object,
    purpose: str,
) -> _VerifiedCalibrationPairReplayTokenV2:
    if issuer is not _CALIBRATION_PAIR_REPLAY_TOKEN_ISSUER_V2:
        raise TypeError("calibration pair replay token issuer is private")
    if purpose not in {"outer-mechanics", "retained-attempt-replay"}:
        raise ValueError("calibration pair replay token purpose is not closed")
    token = _VerifiedCalibrationPairReplayTokenV2(issuer)
    _CALIBRATION_PAIR_REPLAY_TOKEN_REGISTRY_V2[token] = (
        _calibration_pair_payload_root_v2(payloads),
        _calibration_pair_projection_root_v2(pair_projection),
        purpose,
    )
    return token


def _consume_verified_calibration_pair_replay_token_v2(
    token: object,
    payloads: tuple[bytes, bytes, bytes, bytes, bytes, bytes],
    pair_projection: object,
    purpose: str,
) -> None:
    retained = (
        _CALIBRATION_PAIR_REPLAY_TOKEN_REGISTRY_V2.pop(token, None)
        if type(token) is _VerifiedCalibrationPairReplayTokenV2
        else None
    )
    expected = (
        _calibration_pair_payload_root_v2(payloads),
        _calibration_pair_projection_root_v2(pair_projection),
        purpose,
    )
    if retained is None or retained != expected:
        raise TypeError(
            "pure calibration replay needs a fresh retained-directory verified token"
        )


def _discard_verified_calibration_pair_replay_token_v2(token: object) -> None:
    if type(token) is _VerifiedCalibrationPairReplayTokenV2:
        _CALIBRATION_PAIR_REPLAY_TOKEN_REGISTRY_V2.pop(token, None)


def _validate_frozen_calibration_bytes_v2(
    shape: dict[str, object],
    *,
    calibration_basis_profile_bytes: bytes,
    calibration_attempt_manifest_bytes: bytes,
    calibration_inner_artifact_bytes: bytes,
    calibration_outer_supervisor_result_bytes: bytes,
    calibration_rfc8785_prerequisite_receipt_bytes: bytes,
    calibration_mechanics_artifact_bytes: bytes,
    _pair_replay_token: object | None = None,
    _pair_projection: object | None = None,
    _pair_replay_purpose: str | None = None,
    deadline_ns: int | None = None,
) -> None:
    if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
        raise TimeoutError("frozen calibration replay reached its deadline")
    if (
        type(calibration_basis_profile_bytes) is not bytes
        or type(calibration_attempt_manifest_bytes) is not bytes
        or type(calibration_inner_artifact_bytes) is not bytes
        or type(calibration_outer_supervisor_result_bytes) is not bytes
        or type(calibration_rfc8785_prerequisite_receipt_bytes) is not bytes
        or type(calibration_mechanics_artifact_bytes) is not bytes
    ):
        raise TypeError("frozen calibration replay needs exact canonical bytes")
    pair_payloads = (
        calibration_basis_profile_bytes,
        calibration_attempt_manifest_bytes,
        calibration_inner_artifact_bytes,
        calibration_outer_supervisor_result_bytes,
        calibration_rfc8785_prerequisite_receipt_bytes,
        calibration_mechanics_artifact_bytes,
    )
    if _pair_replay_token is not None:
        _consume_verified_calibration_pair_replay_token_v2(
            _pair_replay_token,
            pair_payloads,
            _pair_projection,
            _pair_replay_purpose,
        )
    elif _pair_projection is not None or _pair_replay_purpose is not None:
        raise TypeError("calibration pair projection lacks its private replay token")
    basis_sha256 = hashlib.sha256(calibration_basis_profile_bytes).hexdigest()
    attempt_sha256 = hashlib.sha256(
        calibration_attempt_manifest_bytes
    ).hexdigest()
    inner_sha256 = hashlib.sha256(calibration_inner_artifact_bytes).hexdigest()
    outer_sha256 = hashlib.sha256(
        calibration_outer_supervisor_result_bytes
    ).hexdigest()
    rfc_sha256 = hashlib.sha256(
        calibration_rfc8785_prerequisite_receipt_bytes
    ).hexdigest()
    mechanics_sha256 = hashlib.sha256(
        calibration_mechanics_artifact_bytes
    ).hexdigest()
    if (
        basis_sha256 != shape.get("calibration_basis_profile_sha256")
        or len(calibration_basis_profile_bytes)
        != shape.get("calibration_basis_profile_size_bytes")
        or shape.get("calibration_basis_profile_cas_root")
        != "sha256:" + basis_sha256
        or attempt_sha256 != shape.get("calibration_attempt_manifest_sha256")
        or len(calibration_attempt_manifest_bytes)
        != shape.get("calibration_attempt_manifest_size_bytes")
        or shape.get("calibration_attempt_manifest_cas_root")
        != "sha256:" + attempt_sha256
        or inner_sha256 != shape.get("calibration_artifact_sha256")
        or len(calibration_inner_artifact_bytes)
        != shape.get("calibration_artifact_size_bytes")
        or shape.get("calibration_artifact_cas_root")
        != "sha256:" + inner_sha256
        or mechanics_sha256
        != shape.get("calibration_mechanics_artifact_sha256")
        or len(calibration_mechanics_artifact_bytes)
        != shape.get("calibration_mechanics_artifact_size_bytes")
        or shape.get("calibration_mechanics_artifact_cas_root")
        != "sha256:" + mechanics_sha256
        or outer_sha256
        != shape.get("calibration_outer_supervisor_result_sha256")
        or len(calibration_outer_supervisor_result_bytes)
        != shape.get("calibration_outer_supervisor_result_size_bytes")
        or shape.get("calibration_outer_supervisor_result_cas_root")
        != "sha256:" + outer_sha256
        or rfc_sha256 != shape.get("calibration_rfc8785_receipt_sha256")
        or len(calibration_rfc8785_prerequisite_receipt_bytes)
        != shape.get("calibration_rfc8785_receipt_size_bytes")
        or shape.get("calibration_rfc8785_receipt_cas_root")
        != "sha256:" + rfc_sha256
    ):
        raise ValueError("frozen calibration artifact byte/CAS binding drifted")
    basis_transport = _decode_canonical_transport_object_v2(
        calibration_basis_profile_bytes,
        "frozen calibration basis profile",
        maximum_bytes=_MAXIMUM_PROFILE_BYTES,
    )
    if set(basis_transport) != _ROOT_FIELDS:
        raise ValueError("frozen calibration basis profile root is not closed")
    basis_profile = basis_transport["qualification_profile"]
    if type(basis_profile) is not dict:
        raise ValueError("frozen calibration basis profile body is absent")
    computed_basis_id = PROFILE_ID_PREFIX + hashlib.sha256(
        PROFILE_DOMAIN + canonicalize_rfc8785_v1(basis_profile)
    ).hexdigest()
    basis_shape = (
        basis_profile.get("fixture_contract", {}).get("rich_positive_shape")
        if type(basis_profile.get("fixture_contract")) is dict
        else None
    )
    if (
        basis_transport["qualification_profile_id"] != computed_basis_id
        or computed_basis_id != shape.get("calibration_basis_profile_id")
        or type(basis_shape) is not dict
        or basis_shape.get("calibration_campaign_status")
        != "PENDING_FULL_1024_DEVELOPMENT_MEASUREMENT"
        or basis_shape.get("live_qualification_enabled") is not False
        or not _exact_json_equal(
            basis_profile.get("auditor_binding"),
            _EXPECTED_PROFILE.get("auditor_binding"),
        )
        or not _exact_json_equal(
            basis_profile.get("wire_binding"), _EXPECTED_PROFILE.get("wire_binding")
        )
        or not _exact_json_equal(
            basis_profile.get("dependency_lock"),
            _EXPECTED_PROFILE.get("dependency_lock"),
        )
        or not _exact_json_equal(
            basis_profile.get("canonicalization"),
            _EXPECTED_PROFILE.get("canonicalization"),
        )
    ):
        raise ValueError("frozen calibration basis profile revision drifted")
    calibration_attempt = load_phase2b_formal_wire_covert_attempt_manifest_v2(
        calibration_attempt_manifest_bytes,
        expected_profile_id=computed_basis_id,
    )
    if (
        calibration_attempt.attempt_manifest_root
        != shape.get("calibration_attempt_manifest_root")
        or calibration_attempt.campaign_id != shape.get("calibration_campaign_id")
        or calibration_attempt.attempt_id != shape.get("calibration_attempt_id")
        or calibration_attempt.qualification_config_sha256 != basis_sha256
    ):
        raise ValueError("frozen calibration attempt manifest identity drifted")
    rfc_receipt = _verify_rfc8785_prerequisite_receipt_v2(
        calibration_rfc8785_prerequisite_receipt_bytes,
        calibration_attempt,
        verify_live_executables=_pair_replay_token is None,
        deadline_ns=deadline_ns,
    )
    inner = _decode_canonical_transport_object_v2(
        calibration_inner_artifact_bytes,
        "frozen calibration inner artifact",
        maximum_bytes=8 * 1_024 * 1_024,
    )
    if (
        inner.get("schema_version")
        != "hegel-machine-phase2b-formal-wire-covert-development-calibration/6"
        or inner.get("profile_id") != calibration_attempt.profile_id
        or inner.get("calibration_campaign_id")
        != shape.get("calibration_campaign_id")
        or inner.get("calibration_attempt_id")
        != shape.get("calibration_attempt_id")
        or inner.get("attempt_manifest_root")
        != shape.get("calibration_attempt_manifest_root")
        or inner.get("artifact_cannot_be_finalized_or_promoted") is not True
        or inner.get("same_run_shape_acceptance_forbidden") is not True
    ):
        raise ValueError("frozen calibration inner identity drifted")
    _validate_development_calibration_artifact_v2(inner, calibration_attempt)
    measurements = inner.get("frozen_static_measurement_candidates_in_order")
    runs = inner.get("runs_in_order")
    if (
        type(measurements) is not list
        or type(runs) is not list
        or len(measurements) != 2
        or len(runs) != 2
    ):
        raise ValueError("frozen calibration replay lacks exactly two modes")
    for index, mode in enumerate(LIVE_MODE_ORDER):
        measurement = measurements[index]
        run = runs[index]
        if (
            type(measurement) is not dict
            or type(run) is not dict
            or measurement.get("mode") != mode
            or run.get("mode") != mode
        ):
            raise ValueError("frozen calibration replay mode order drifted")
        _validate_run_and_measurement_binding_v2(run, measurement)
        expected_projection_name = (
            "known_null_frozen_static_projection"
            if mode == "KNOWN_NULL"
            else "known_leak_frozen_static_projection"
        )
        if not _exact_json_equal(
            measurement.get("frozen_static_projection"),
            shape.get(expected_projection_name),
        ):
            raise ValueError("frozen calibration static projection splice")
    expected_fixture_root = _combined_fixture_root_from_run_roots_v2(
        runs[0]["run_fixture_root"], runs[1]["run_fixture_root"]
    )
    if (
        inner.get("synthetic_fixture_root") != expected_fixture_root
        or expected_fixture_root != shape.get("combined_fixture_root")
    ):
        raise ValueError("frozen calibration combined fixture root drifted")
    mechanics = _decode_canonical_transport_object_v2(
        calibration_mechanics_artifact_bytes,
        "frozen calibration mechanics artifact",
        maximum_bytes=8 * 1_024 * 1_024,
    )
    if set(mechanics) != {
        "attempt_manifest_root",
        "authority_issuer_present",
        "calibration_artifact_cannot_be_promoted",
        "claims",
        "content_sha256_is_not_a_signature",
        "disposition",
        "entry_kind",
        "inner_artifact_sha256",
        "outer_supervisor_result_sha256",
        "profile_id",
        "projection_only_boundary",
        "rfc8785_prerequisite_receipt_sha256",
        "schema_version",
        "step5_signed_cas_authority_still_required",
    }:
        raise ValueError("frozen calibration mechanics fields are not exact")
    expected_claims = {
        "actual_corpus_audit": False,
        "actual_unsealed_960_generation": False,
        "campaign_execution_authorized": False,
        "key_independence_verified": False,
        "origin_authenticated": False,
        "r03_formal_trusted_wire_implementation_qualified": False,
        "r04_formal_covert_auditor_implementation_qualified": False,
        "r05_synthetic_dev_wire_covert_qualification_verified": False,
    }
    if (
        mechanics["schema_version"] != MECHANICS_VERIFIER_SCHEMA_V2
        or mechanics["disposition"] != "MECHANICS_VERIFIED_NOT_AUTHORITY"
        or mechanics["entry_kind"] != "development-calibration"
        or mechanics["profile_id"] != calibration_attempt.profile_id
        or mechanics["attempt_manifest_root"]
        != shape.get("calibration_attempt_manifest_root")
        or mechanics["inner_artifact_sha256"] != inner_sha256
        or mechanics["outer_supervisor_result_sha256"] != outer_sha256
        or mechanics["rfc8785_prerequisite_receipt_sha256"] != rfc_sha256
        or mechanics["authority_issuer_present"] is not False
        or mechanics["calibration_artifact_cannot_be_promoted"] is not True
        or not _exact_json_equal(
            mechanics["projection_only_boundary"], _TASK4_PROJECTION_ONLY_BOUNDARY_V2
        )
        or mechanics["content_sha256_is_not_a_signature"] is not True
        or mechanics["step5_signed_cas_authority_still_required"] is not True
        or not _exact_json_equal(mechanics["claims"], expected_claims)
    ):
        raise ValueError("frozen calibration mechanics claim/binding drifted")
    outer = _decode_canonical_transport_object_v2(
        calibration_outer_supervisor_result_bytes,
        "frozen calibration outer supervisor result",
        maximum_bytes=1_048_576,
    )
    if _pair_replay_token is not None:
        if (
            outer.get("schema_version") != OUTER_SUPERVISOR_RESULT_SCHEMA_V2
            or outer.get("mode") != "development-calibration"
            or outer.get("attempt_manifest_root")
            != calibration_attempt.attempt_manifest_root
            or outer.get("profile_id") != calibration_attempt.profile_id
        ):
            raise ValueError("pure paired calibration outer identity drifted")
        return
    _validate_outer_supervisor_result_v2(
        outer,
        attempt=calibration_attempt,
        inner=inner,
        inner_bytes=calibration_inner_artifact_bytes,
        rfc_receipt=rfc_receipt,
        rfc_receipt_bytes=calibration_rfc8785_prerequisite_receipt_bytes,
        expected_entry="development-calibration",
        deadline_ns=deadline_ns,
    )


def _load_retained_development_basis_profile_id_v2(
    payload: bytes,
    *,
    frozen_shape: object,
    expected_entry_kind: str,
) -> str:
    """Load the one historical pending profile pinned by the current frozen shape."""

    if expected_entry_kind != "development-calibration":
        raise ValueError("depth-one pair replay accepts only a development attempt")
    if (
        type(frozen_shape) is not dict
        or frozen_shape.get("calibration_campaign_status")
        != "FROZEN_FULL_1024_DEVELOPMENT_MEASUREMENT"
        or frozen_shape.get("live_qualification_enabled") is not True
    ):
        raise ValueError("current profile does not expose one frozen calibration basis")
    expected_id = frozen_shape.get("calibration_basis_profile_id")
    expected_sha256 = frozen_shape.get("calibration_basis_profile_sha256")
    expected_size = frozen_shape.get("calibration_basis_profile_size_bytes")
    if (
        type(expected_id) is not str
        or re.fullmatch(
            r"phase2b_formal_wire_covert_qualification_profile_v2_[0-9a-f]{64}",
            expected_id,
        )
        is None
        or expected_id == PROFILE_ID
        or type(expected_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or type(expected_size) is not int
        or type(expected_size) is bool
        or not 1 <= expected_size <= _MAXIMUM_PROFILE_BYTES
        or type(payload) is not bytes
        or len(payload) != expected_size
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise ValueError("retained calibration basis identity/bytes are not frozen")
    transport = _decode_canonical_transport_object_v2(
        payload,
        "retained historical calibration basis profile",
        maximum_bytes=_MAXIMUM_PROFILE_BYTES,
    )
    if set(transport) != _ROOT_FIELDS:
        raise ValueError("retained historical basis profile root is not closed")
    profile = transport["qualification_profile"]
    transported_id = transport["qualification_profile_id"]
    if type(profile) is not dict:
        raise ValueError("retained historical basis profile body is absent")
    computed_id = PROFILE_ID_PREFIX + hashlib.sha256(
        PROFILE_DOMAIN + canonicalize_rfc8785_v1(profile)
    ).hexdigest()
    fixture_contract = profile.get("fixture_contract")
    pending_shape = (
        fixture_contract.get("rich_positive_shape")
        if type(fixture_contract) is dict
        else None
    )
    if (
        transported_id != computed_id
        or computed_id != expected_id
        or profile.get("schema_version") != PROFILE_SCHEMA_VERSION
        or not _exact_json_equal(
            profile.get("auditor_binding"), _EXPECTED_PROFILE.get("auditor_binding")
        )
        or not _exact_json_equal(
            profile.get("wire_binding"), _EXPECTED_PROFILE.get("wire_binding")
        )
        or not _exact_json_equal(
            profile.get("dependency_lock"), _EXPECTED_PROFILE.get("dependency_lock")
        )
        or not _exact_json_equal(
            profile.get("canonicalization"),
            _EXPECTED_PROFILE.get("canonicalization"),
        )
    ):
        raise ValueError("retained historical basis profile revision drifted")
    _require_pending_calibration_profile_v2(pending_shape)
    return computed_id


def _expected_live_file_sha256_v2(
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
) -> dict[str, str]:
    return {
        "config/phase2b_formal_covert_auditor_v1.json": AUDITOR_CONFIG_SHA256,
        "config/phase2b_formal_trusted_wire_v3.json": WIRE_CONFIG_SHA256,
        "config/phase2b_formal_wire_covert_qualification_v2.json": (
            attempt.qualification_config_sha256
        ),
        "config/phase2b_formal_wire_covert_qualification_v2.lock.json": (
            attempt.qualification_lock_sha256
        ),
        "docs/Hegel_Machine_Phase2B_Formal_Covert_Auditor_v1.md": (
            AUDITOR_DOCUMENTATION_SHA256
        ),
        "docs/Hegel_Machine_Phase2B_Formal_Trusted_Wire_V3.md": (
            WIRE_DOCUMENTATION_SHA256
        ),
        "docs/Hegel_Machine_Phase2B_Formal_Wire_Covert_Qualification_v2.md": (
            attempt.qualification_documentation_sha256
        ),
        "docs/Hegel_Machine_Phase2B_RFC8785_Implementation_v1.md": (
            RFC8785_DOCUMENTATION_SHA256
        ),
        "golden_vectors/phase2b_rfc8785_v1.json": (
            RFC8785_GOLDEN_VECTORS_SHA256
        ),
        "src/hegel_machine/hashing.py": HASHING_SOURCE_SHA256,
        "src/hegel_machine/phase2b_exact_transform_core_v1.py": (
            EXACT_CORE_SOURCE_SHA256
        ),
        "src/hegel_machine/phase2b_formal_covert_auditor_v1.py": (
            AUDITOR_SOURCE_SHA256
        ),
        "src/hegel_machine/phase2b_formal_trusted_wire_v3.py": WIRE_SOURCE_SHA256,
        "src/hegel_machine/phase2b_formal_wire_covert_qualification_v2.py": (
            attempt.qualification_source_sha256
        ),
        "src/hegel_machine/phase2b_freeze_v1.py": PHASE2B_FREEZE_SOURCE_SHA256,
        "src/hegel_machine/rfc8785_v1.py": RFC8785_SOURCE_SHA256,
        "src/hegel_machine/schema.py": SCHEMA_SOURCE_SHA256,
        "tests/test_phase2b_exact_transform_core_v1.py": EXACT_CORE_TEST_SHA256,
        "tests/test_phase2b_formal_covert_auditor_v1.py": AUDITOR_TEST_SHA256,
        "tests/test_phase2b_formal_trusted_wire_v3.py": WIRE_TEST_SHA256,
        "tests/test_phase2b_formal_wire_covert_qualification_v2.py": (
            attempt.qualification_test_sha256
        ),
        "tests/test_rfc8785_v1.py": RFC8785_TEST_SHA256,
        "tools/phase2b_rfc8785_oracle_v1.mjs": RFC8785_NODE_ORACLE_SHA256,
    }


def _source_execution_bootstrap_manifest_v2(
    qualification_source_sha256: str,
) -> dict[str, object]:
    _require_raw_sha256_v2(
        qualification_source_sha256, "source bootstrap qualification source"
    )
    return {
        "schema_version": "hegel-machine-python-source-open-once-bootstrap/2",
        "main": _BOOTSTRAP_MAIN_RELATIVE_PATH_V2,
        "modules": dict(_BOOTSTRAP_MODULE_RELATIVE_PATHS_V2),
        "sha256": {
            **_BOOTSTRAP_FIXED_SOURCE_SHA256_V2,
            _BOOTSTRAP_MAIN_RELATIVE_PATH_V2: qualification_source_sha256,
        },
    }


def _replay_extracted_child_preopened_sources_v2(
    source_tree: Path,
    qualification_source_sha256: str,
    *,
    retained_source_tree_fd: int | None = None,
    deadline_ns: int | None = None,
) -> tuple[dict[str, dict[str, object]], str]:
    """Rebuild the child's open-once snapshot root from retained source bytes."""

    if not source_tree.is_absolute():
        raise ValueError("child preopened-source replay root is not absolute")
    owned_fd: int | None = None
    if retained_source_tree_fd is None:
        owned_fd = _open_real_directory_fd_v2(source_tree)
        source_tree_fd = owned_fd
    elif type(retained_source_tree_fd) is int and retained_source_tree_fd >= 0:
        source_tree_fd = retained_source_tree_fd
    else:
        raise TypeError("child preopened-source retained descriptor is malformed")
    try:
        _verify_retained_directory_path_identity_v2(
            source_tree_fd, source_tree, name="child preopened-source tree"
        )
        manifest = _source_execution_bootstrap_manifest_v2(
            qualification_source_sha256
        )
        snapshots: dict[str, dict[str, object]] = {}
        for relative, expected_digest in sorted(manifest["sha256"].items()):
            digest, opened = _hash_relative_regular_at_fd_v2(
                source_tree_fd,
                relative,
                maximum_bytes=16 * 1_024 * 1_024,
                deadline_ns=deadline_ns,
            )
            if opened.st_size < 1 or digest != expected_digest:
                raise ValueError("child preopened-source retained bytes drifted")
            snapshots[relative] = {
                "descriptor_realpath": str(source_tree / relative),
                "device": opened.st_dev,
                "inode": opened.st_ino,
                "mtime_nanoseconds": opened.st_mtime_ns,
                "sha256": digest,
                "size_bytes": opened.st_size,
            }
        return snapshots, stable_hash(
            snapshots,
            prefix="phase2b_formal_wire_covert_preopened_sources_v2_",
        )
    finally:
        _close_owned_descriptors_v2(
            (owned_fd,), name="child preopened-source replay"
        )


def _source_execution_bootstrap_argv_v2(
    *,
    python_realpath: str,
    project_root: Path,
    qualification_source_sha256: str,
    source_arguments: list[str],
) -> list[str]:
    if (
        type(python_realpath) is not str
        or not Path(python_realpath).is_absolute()
        or not project_root.is_absolute()
        or type(source_arguments) is not list
        or any(type(item) is not str or "\x00" in item for item in source_arguments)
    ):
        raise ValueError("source bootstrap argv inputs are malformed")
    manifest = _source_execution_bootstrap_manifest_v2(
        qualification_source_sha256
    )
    return [
        python_realpath,
        "-I",
        "-S",
        "-B",
        "-c",
        SOURCE_EXECUTION_BOOTSTRAP_CODE_V2,
        canonicalize_rfc8785_v1(manifest).decode("utf-8"),
        str(project_root),
        *source_arguments,
    ]


def _verify_qualified_cli_source_bootstrap_v2(project_root: Path) -> str:
    """Verify the external open-once bootstrap before any CLI business entry."""

    if (
        not isinstance(project_root, Path)
        or not project_root.is_absolute()
        or project_root.is_symlink()
        or not project_root.is_dir()
        or project_root.resolve(strict=True) != project_root
    ):
        raise ValueError("qualified CLI project root is not one canonical directory")
    expected_source = project_root / _BOOTSTRAP_MAIN_RELATIVE_PATH_V2
    if (
        expected_source.is_symlink()
        or not expected_source.is_file()
        or Path(__file__).resolve(strict=True) != expected_source
        or _SOURCE_EXECUTION_BOOTSTRAP_ACTIVE_V2 is not True
        or _SOURCE_ONLY_IMPORTER_ACTIVE_V2 is not True
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.dont_write_bytecode != 1
        or sys.flags.ignore_environment != 1
    ):
        raise ValueError("qualified CLI did not enter through the isolated bootstrap")
    executed = _EXECUTED_MAIN_SOURCE_BOOTSTRAP_V2
    manifest = globals().get("_HEGEL_SOURCE_BOOTSTRAP_MANIFEST_V2")
    snapshots = _PREOPENED_SOURCE_SNAPSHOTS_V2
    source_bytes = globals().get("_HEGEL_PREOPENED_SOURCE_BYTES_V2")
    expected_relatives = set(_BOOTSTRAP_FIXED_SOURCE_SHA256_V2) | {
        _BOOTSTRAP_MAIN_RELATIVE_PATH_V2
    }
    if (
        type(executed) is not dict
        or type(manifest) is not dict
        or type(snapshots) is not dict
        or type(source_bytes) is not dict
        or set(snapshots) != expected_relatives
        or set(source_bytes) != set(_BOOTSTRAP_MODULE_RELATIVE_PATHS_V2)
    ):
        raise ValueError("qualified CLI bootstrap evidence is incomplete")
    main_snapshot = snapshots.get(_BOOTSTRAP_MAIN_RELATIVE_PATH_V2)
    if type(main_snapshot) is not dict:
        raise ValueError("qualified CLI main-source snapshot is absent")
    main_sha256 = _require_raw_sha256_v2(
        main_snapshot.get("sha256"), "qualified CLI main source"
    )
    expected_manifest = _source_execution_bootstrap_manifest_v2(main_sha256)
    expected_executed = {
        **main_snapshot,
        "bootstrap_kind": "EXTERNAL_PYTHON_C_OPEN_ONCE_V2",
        "reexecuted_from_stable_descriptor": True,
    }
    if (
        not _exact_json_equal(manifest, expected_manifest)
        or not _exact_json_equal(executed, expected_executed)
        or main_snapshot.get("descriptor_realpath") != str(expected_source)
        or main_snapshot.get("size_bytes") != expected_source.stat().st_size
    ):
        raise ValueError("qualified CLI bootstrap manifest/source binding drifted")
    for module_name, relative in _BOOTSTRAP_MODULE_RELATIVE_PATHS_V2.items():
        snapshot = snapshots.get(relative)
        payload = source_bytes.get(module_name)
        if (
            type(snapshot) is not dict
            or type(payload) is not bytes
            or snapshot.get("descriptor_realpath") != str(project_root / relative)
            or snapshot.get("sha256")
            != _BOOTSTRAP_FIXED_SOURCE_SHA256_V2[relative]
            or snapshot.get("size_bytes") != len(payload)
            or hashlib.sha256(payload).hexdigest() != snapshot.get("sha256")
        ):
            raise ValueError("qualified CLI imported-source snapshot drifted")
    if _sha256_file_v2(expected_source) != main_sha256:
        raise ValueError("qualified CLI current source differs from executed bytes")
    return main_sha256


def _verify_live_package_tree_is_source_only_v2(
    source_tree: Path, *, deadline_ns: int | None = None
) -> None:
    package_directory = source_tree / "src/hegel_machine"
    if package_directory.is_symlink() or not package_directory.is_dir():
        raise ValueError("outer live package directory is not a real directory")
    visited = 0
    for root, directory_names, file_names in os.walk(
        package_directory, topdown=True, followlinks=False
    ):
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("outer package walk reached its absolute deadline")
        visited += len(directory_names) + len(file_names)
        if visited > _MAXIMUM_RETAINED_ATTEMPT_TREE_ROWS_V2:
            raise ValueError("outer package walk exceeds its entry cap")
        root_path = Path(root)
        for name in tuple(directory_names):
            if (root_path / name).is_symlink():
                raise ValueError("outer live package tree contains a directory symlink")
        for name in file_names:
            candidate = root_path / name
            if candidate.is_symlink():
                raise ValueError("outer live package tree contains a file symlink")
            if candidate.suffix in {".pyc", ".pyo"}:
                raise ValueError("outer live package tree contains forbidden bytecode")


def _outer_runner_tcb_snapshot_v2(
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
    *,
    deadline_ns: int | None = None,
) -> dict[str, object]:
    if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
        raise TimeoutError("outer runner TCB snapshot reached its deadline")
    flags = sys.flags
    if (
        _SOURCE_ONLY_IMPORTER_ACTIVE_V2 is not True
        or flags.isolated != 1
        or flags.no_site != 1
        or flags.dont_write_bytecode != 1
        or flags.ignore_environment != 1
    ):
        raise ValueError("outer runner itself is not direct -I -S -B source-only code")
    source = Path(__file__).resolve(strict=True)
    bootstrap = _EXECUTED_MAIN_SOURCE_BOOTSTRAP_V2
    bootstrap_keys = {
        "bootstrap_kind",
        "descriptor_realpath",
        "device",
        "inode",
        "mtime_nanoseconds",
        "reexecuted_from_stable_descriptor",
        "sha256",
        "size_bytes",
    }
    source_stat = source.stat(follow_symlinks=False)
    expected_bootstrap_manifest = _source_execution_bootstrap_manifest_v2(
        attempt.qualification_source_sha256
    )
    injected_bootstrap_manifest = globals().get(
        "_HEGEL_SOURCE_BOOTSTRAP_MANIFEST_V2"
    )
    if (
        _SOURCE_EXECUTION_BOOTSTRAP_ACTIVE_V2 is not True
        or
        type(bootstrap) is not dict
        or set(bootstrap) != bootstrap_keys
        or bootstrap["bootstrap_kind"] != "EXTERNAL_PYTHON_C_OPEN_ONCE_V2"
        or bootstrap["reexecuted_from_stable_descriptor"] is not True
        or bootstrap["descriptor_realpath"] != str(source)
        or bootstrap["sha256"] != attempt.qualification_source_sha256
        or bootstrap["device"] != source_stat.st_dev
        or bootstrap["inode"] != source_stat.st_ino
        or bootstrap["mtime_nanoseconds"] != source_stat.st_mtime_ns
        or bootstrap["size_bytes"] != source_stat.st_size
        or _sha256_file_v2(source, deadline_ns=deadline_ns)
        != bootstrap["sha256"]
        or not _exact_json_equal(
            injected_bootstrap_manifest, expected_bootstrap_manifest
        )
        or type(_PREOPENED_SOURCE_BYTES_V2) is not dict
        or set(_PREOPENED_SOURCE_BYTES_V2)
        != set(_BOOTSTRAP_MODULE_RELATIVE_PATHS_V2)
        or type(_PREOPENED_SOURCE_SNAPSHOTS_V2) is not dict
        or set(_PREOPENED_SOURCE_SNAPSHOTS_V2)
        != set(expected_bootstrap_manifest["sha256"])
    ):
        raise ValueError("outer runner executed-byte bootstrap/source binding drifted")
    source_tree = source.parents[2]
    expected_files = _expected_live_file_sha256_v2(attempt)
    for relative, digest in expected_files.items():
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("outer runner TCB file replay reached its deadline")
        candidate = source_tree / relative
        if (
            not candidate.is_file()
            or _sha256_file_v2(candidate, deadline_ns=deadline_ns) != digest
        ):
            raise ValueError(f"outer runner host TCB file drifted: {relative}")
    if (
        str(source) != str(
            source_tree
            / "src/hegel_machine/phase2b_formal_wire_covert_qualification_v2.py"
        )
        or expected_files[
            "src/hegel_machine/phase2b_formal_wire_covert_qualification_v2.py"
        ]
        != attempt.qualification_source_sha256
    ):
        raise ValueError("outer runner host source does not bind the attempt")
    return {
        "broad_package_init_executed": False,
        "bypassed_package_init_sha256": BYPASSED_PACKAGE_INIT_SHA256,
        "executed_main_source_bootstrap": bootstrap,
        "interpreter_flags": {
            "dont_write_bytecode": 1,
            "ignore_environment": 1,
            "isolated": 1,
            "no_site": 1,
        },
        "live_files_sha256": expected_files,
        "preopened_source_snapshots": _PREOPENED_SOURCE_SNAPSHOTS_V2,
        "source_bootstrap_manifest": expected_bootstrap_manifest,
        "source_execution_bootstrap_active": True,
        "source_only_importer_active": True,
        "source_realpath": str(source),
        "source_sha256": attempt.qualification_source_sha256,
        "source_tree_root": str(source_tree),
    }


def _expected_post_output_gate_attestation_v2(
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
    inner: dict[str, object],
    inner_bytes: bytes,
    *,
    expected_entry: str,
    child_argv_sha256: object,
    owned_invocation_id: object,
) -> dict[str, object]:
    """Reconstruct the source-and-exit attestation without inventing a peak."""

    if not _exact_json_equal(
        inner.get("post_output_physical_gate_contract"),
        _POST_OUTPUT_PHYSICAL_GATE_CONTRACT_V2,
    ):
        raise ValueError("outer attestation cannot bind an absent inner tail contract")
    if (
        expected_entry not in {"development-calibration", "live-inner"}
        or type(child_argv_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", child_argv_sha256) is None
        or type(owned_invocation_id) is not str
        or re.fullmatch(r"[0-9a-f]{32}", owned_invocation_id) is None
    ):
        raise ValueError("outer attestation launch identity is malformed")
    return {
        "attestation_basis": (
            "PINNED_INNER_SOURCE_PLUS_SYSTEMD_EXEC_EXIT_ZERO;"
            "NOT_A_POST_EXIT_NUMERIC_MEMORY_PEAK_SAMPLE"
        ),
        "canonical_stdout_matches_inner_artifact": True,
        "child_exec_main_code": 1,
        "child_exec_main_status": 0,
        "child_result": "success",
        "child_systemd_exit_kind": "CLD_EXITED",
        "direct_os_exit_success_path_required_by_pinned_source": True,
        "empty_stderr": True,
        "entry_kind": expected_entry,
        "exact_exec_start_argv_sha256": child_argv_sha256,
        "inner_artifact_sha256": hashlib.sha256(inner_bytes).hexdigest(),
        "inner_artifact_size_bytes": len(inner_bytes),
        "inner_post_output_physical_gate_contract": dict(
            _POST_OUTPUT_PHYSICAL_GATE_CONTRACT_V2
        ),
        "pinned_qualification_source_sha256": (
            attempt.qualification_source_sha256
        ),
        "post_exit_numeric_memory_peak_sample_claimed": False,
        "success_exit_status_override_can_satisfy_attestation": False,
        "systemd_memory_max_bytes": attempt.requested_cgroup_memory_max_bytes,
        "systemd_owned_invocation_id": owned_invocation_id,
    }


def _validate_post_output_gate_attestation_v2(
    value: object,
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
    inner: dict[str, object],
    inner_bytes: bytes,
    *,
    context: str,
    expected_entry: str,
    child_argv_sha256: object,
    owned_invocation_id: object,
) -> None:
    if not _exact_json_equal(
        value,
        _expected_post_output_gate_attestation_v2(
            attempt,
            inner,
            inner_bytes,
            expected_entry=expected_entry,
            child_argv_sha256=child_argv_sha256,
            owned_invocation_id=owned_invocation_id,
        ),
    ):
        raise ValueError(f"{context} post-output physical gate attestation drifted")


def _validate_outer_supervisor_result_v2(
    body: dict[str, object],
    *,
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
    inner: dict[str, object],
    inner_bytes: bytes,
    rfc_receipt: dict[str, object],
    rfc_receipt_bytes: bytes,
    expected_entry: str,
    deadline_ns: int | None = None,
) -> None:
    if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
        raise TimeoutError("outer supervisor replay reached its absolute deadline")
    expected_keys = {
        "attempt_manifest_root",
        "child_source_bootstrap_preopened_snapshots",
        "child_source_bootstrap_preopened_snapshots_root",
        "content_sha256_is_not_a_signature",
        "inner_artifact_cas",
        "launcher",
        "live_files_sha256",
        "mode",
        "outer_runner_tcb_entry",
        "outer_runner_tcb_exit",
        "post_output_physical_gate_attestation",
        "profile_id",
        "result",
        "rfc8785_receipt_cas",
        "rfc8785_raw_bundle_directory",
        "rfc8785_raw_bundle_projection",
        "runtime_environment_manifest_root",
        "runtime_environment_manifest_cas",
        "runtime_environment_manifest_object_path",
        "runtime_image_digest",
        "schema_version",
        "source_archive_cas",
        "source_archive_extraction",
        "source_archive_object_path",
        "source_archive_root",
        "source_commit_sha",
        "source_tree_root",
        "stderr",
        "stdout",
        "timing",
        "unit",
    }
    if set(body) != expected_keys:
        raise ValueError("outer supervisor result fields are not exact")
    if (
        body["schema_version"] != OUTER_SUPERVISOR_RESULT_SCHEMA_V2
        or body["mode"] != expected_entry
        or body["profile_id"] != attempt.profile_id
        or body["attempt_manifest_root"] != attempt.attempt_manifest_root
        or body["source_commit_sha"] != attempt.source_commit_sha
        or body["source_archive_root"] != attempt.source_archive_root
        or body["runtime_environment_manifest_root"]
        != attempt.runtime_environment_manifest_root
        or body["runtime_image_digest"] is not None
        or body["content_sha256_is_not_a_signature"] is not True
    ):
        raise ValueError("outer supervisor identity/CAS boundary drifted")
    outer_tcb_entry = body["outer_runner_tcb_entry"]
    outer_tcb_exit = body["outer_runner_tcb_exit"]
    expected_tcb_keys = {
        "broad_package_init_executed",
        "bypassed_package_init_sha256",
        "executed_main_source_bootstrap",
        "interpreter_flags",
        "live_files_sha256",
        "preopened_source_snapshots",
        "source_bootstrap_manifest",
        "source_execution_bootstrap_active",
        "source_only_importer_active",
        "source_realpath",
        "source_sha256",
        "source_tree_root",
    }
    if (
        type(outer_tcb_entry) is not dict
        or set(outer_tcb_entry) != expected_tcb_keys
        or not _exact_json_equal(outer_tcb_entry, outer_tcb_exit)
        or outer_tcb_entry["broad_package_init_executed"] is not False
        or outer_tcb_entry["source_only_importer_active"] is not True
        or outer_tcb_entry["source_execution_bootstrap_active"] is not True
        or outer_tcb_entry["bypassed_package_init_sha256"]
        != BYPASSED_PACKAGE_INIT_SHA256
        or outer_tcb_entry["source_sha256"] != attempt.qualification_source_sha256
        or type(outer_tcb_entry["executed_main_source_bootstrap"]) is not dict
        or outer_tcb_entry["executed_main_source_bootstrap"].get("sha256")
        != attempt.qualification_source_sha256
        or outer_tcb_entry["executed_main_source_bootstrap"].get(
            "reexecuted_from_stable_descriptor"
        )
        is not True
        or outer_tcb_entry["executed_main_source_bootstrap"].get(
            "bootstrap_kind"
        )
        != "EXTERNAL_PYTHON_C_OPEN_ONCE_V2"
        or not _exact_json_equal(
            outer_tcb_entry["source_bootstrap_manifest"],
            _source_execution_bootstrap_manifest_v2(
                attempt.qualification_source_sha256
            ),
        )
        or type(outer_tcb_entry["source_tree_root"]) is not str
        or type(outer_tcb_entry["source_realpath"]) is not str
        or outer_tcb_entry["source_realpath"]
        != str(
            Path(outer_tcb_entry["source_tree_root"])
            / "src/hegel_machine/phase2b_formal_wire_covert_qualification_v2.py"
        )
        or not _exact_json_equal(
            outer_tcb_entry["live_files_sha256"],
            _expected_live_file_sha256_v2(attempt),
        )
        or not _exact_json_equal(
            outer_tcb_entry["interpreter_flags"],
            {
                "dont_write_bytecode": 1,
                "ignore_environment": 1,
                "isolated": 1,
                "no_site": 1,
            },
        )
    ):
        raise ValueError("outer runner entry/exit TCB evidence drifted")
    outer_source_tree_path = Path(outer_tcb_entry["source_tree_root"])
    outer_snapshots, outer_preopened_root = _replay_extracted_child_preopened_sources_v2(
        outer_source_tree_path,
        attempt.qualification_source_sha256,
        deadline_ns=deadline_ns,
    )
    outer_main_snapshot = outer_snapshots[_BOOTSTRAP_MAIN_RELATIVE_PATH_V2]
    if (
        not _exact_json_equal(
            outer_tcb_entry["preopened_source_snapshots"], outer_snapshots
        )
        or not _exact_json_equal(
            outer_tcb_entry["executed_main_source_bootstrap"],
            {
                **outer_main_snapshot,
                "bootstrap_kind": "EXTERNAL_PYTHON_C_OPEN_ONCE_V2",
                "reexecuted_from_stable_descriptor": True,
            },
        )
    ):
        raise ValueError("outer runner preopened source snapshots drifted")
    source_tree = body["source_tree_root"]
    if type(source_tree) is not str or not Path(source_tree).is_absolute():
        raise ValueError("outer source tree root is not absolute")
    source_tree_path = Path(source_tree)
    if (
        source_tree_path.is_symlink()
        or not source_tree_path.is_dir()
        or str(source_tree_path.resolve(strict=True)) != source_tree
        or source_tree_path == outer_source_tree_path
    ):
        raise ValueError("outer source tree root is not a live realpath")
    _verify_live_package_tree_is_source_only_v2(
        source_tree_path, deadline_ns=deadline_ns
    )
    expected_files = _expected_live_file_sha256_v2(attempt)
    if not _exact_json_equal(body["live_files_sha256"], expected_files):
        raise ValueError("outer live-file manifest differs from the attempt/dependency pins")
    for relative, expected_digest in expected_files.items():
        path = source_tree_path / relative
        if (
            not path.is_file()
            or _sha256_file_v2(path, deadline_ns=deadline_ns)
            != expected_digest
        ):
            raise ValueError(f"outer live file differs: {relative}")
    if (
        attempt.dependency_lock_sha256 != DEPENDENCY_LOCK_SHA256
        or attempt.qualification_lock_sha256 != DEPENDENCY_LOCK_SHA256
    ):
        raise ValueError("attempt dependency/qualification lock hashes drifted")
    _verify_cas_store_projection_v2(
        body["source_archive_cas"],
        expected_sha256=attempt.source_archive_root.removeprefix("sha256:"),
        expected_size=attempt.source_archive_size_bytes,
        expected_store_namespace=attempt.attempt_manifest_root,
        name="source archive",
    )
    _verify_cas_store_projection_v2(
        body["runtime_environment_manifest_cas"],
        expected_sha256=attempt.runtime_environment_manifest_root.removeprefix(
            "sha256:"
        ),
        expected_size=attempt.runtime_environment_manifest_size_bytes,
        expected_store_namespace=attempt.attempt_manifest_root,
        name="runtime environment manifest",
    )
    source_archive_object = body["source_archive_object_path"]
    runtime_manifest_object = body["runtime_environment_manifest_object_path"]
    if (
        type(source_archive_object) is not str
        or not Path(source_archive_object).is_absolute()
        or type(runtime_manifest_object) is not str
        or not Path(runtime_manifest_object).is_absolute()
    ):
        raise ValueError("outer CAS object paths are not absolute")
    source_archive_object_path = Path(source_archive_object)
    runtime_manifest_object_path = Path(runtime_manifest_object)
    source_archive_digest = attempt.source_archive_root.removeprefix("sha256:")
    runtime_manifest_digest = attempt.runtime_environment_manifest_root.removeprefix(
        "sha256:"
    )
    for path, digest, size, name in (
        (
            source_archive_object_path,
            source_archive_digest,
            attempt.source_archive_size_bytes,
            "source archive CAS object",
        ),
        (
            runtime_manifest_object_path,
            runtime_manifest_digest,
            attempt.runtime_environment_manifest_size_bytes,
            "runtime manifest CAS object",
        ),
    ):
        if (
            path.name != digest
            or path.is_symlink()
            or not path.is_file()
            or str(path.resolve(strict=True)) != str(path)
            or path.stat(follow_symlinks=False).st_size != size
            or _sha256_file_v2(path, deadline_ns=deadline_ns) != digest
        ):
            raise ValueError(f"{name} does not replay its retained bytes")
    (
        archive_regular_root,
        archive_regular_count,
        archive_regular_bytes,
        archive_member_count,
        archive_directory_count,
        archive_tree_rows,
        archive_tree_root,
    ) = (
        _inspect_source_archive_regular_manifest_v2(
            source_archive_object_path,
            expected_sha256=source_archive_digest,
            expected_size=attempt.source_archive_size_bytes,
            expected_source_commit_sha=attempt.source_commit_sha,
            deadline_ns=deadline_ns,
        )
    )
    extraction = body["source_archive_extraction"]
    extraction_keys = {
        "archive_format",
        "archive_sha256",
        "archive_size_bytes",
        "directory_count",
        "forbidden_bytecode_count",
        "member_count",
        "regular_file_count",
        "regular_file_manifest_root",
        "source_tree_manifest_row_count",
        "source_tree_manifest_root",
        "symlink_hardlink_special_count",
        "total_regular_bytes",
    }
    tree_rows, tree_manifest_root = _source_tree_manifest_v2(
        source_tree_path, deadline_ns=deadline_ns
    )
    _verify_source_archive_full_tree_equivalence_v2(
        archive_tree_rows,
        archive_tree_root,
        tree_rows,
        tree_manifest_root,
    )
    tree_regular_root = _regular_file_manifest_root_v2(tree_rows)
    if (
        type(extraction) is not dict
        or set(extraction) != extraction_keys
        or extraction["archive_format"]
        != "TAR_REGULAR_AND_DIRECTORY_ENTRIES_ONLY"
        or extraction["archive_sha256"] != source_archive_digest
        or _require_exact_int_v2(
            extraction["archive_size_bytes"], "archive extraction size", minimum=1
        )
        != attempt.source_archive_size_bytes
        or _require_exact_int_v2(
            extraction["regular_file_count"], "archive regular files", minimum=1
        )
        != archive_regular_count
        or _require_exact_int_v2(
            extraction["total_regular_bytes"], "archive expanded bytes", minimum=1
        )
        != archive_regular_bytes
        or extraction["regular_file_manifest_root"] != archive_regular_root
        or extraction["regular_file_manifest_root"] != tree_regular_root
        or extraction["source_tree_manifest_root"] != tree_manifest_root
        or _require_exact_int_v2(
            extraction["source_tree_manifest_row_count"], "source tree rows", minimum=1
        )
        != len(tree_rows)
        or _require_exact_int_v2(
            extraction["directory_count"], "archive directory count"
        )
        != archive_directory_count
        or _require_exact_int_v2(
            extraction["member_count"], "archive member count", minimum=1
        )
        != archive_member_count
        or _require_exact_int_v2(
            extraction["forbidden_bytecode_count"], "archive bytecode count"
        )
        != 0
        or _require_exact_int_v2(
            extraction["symlink_hardlink_special_count"], "archive special count"
        )
        != 0
    ):
        raise ValueError("source archive/extracted tree evidence drifted")
    launcher = body["launcher"]
    if type(launcher) is not dict or set(launcher) != {
        "argv",
        "environment",
        "python",
        "systemd_argv",
        "systemd_run",
        "systemctl",
        "systemctl_commands",
    }:
        raise ValueError("outer launcher projection is not closed")
    python_row = _verify_live_executable_v2(
        launcher["python"], "outer Python", deadline_ns=deadline_ns
    )
    systemd_run_row = _verify_live_executable_v2(
        launcher["systemd_run"], "systemd-run", deadline_ns=deadline_ns
    )
    systemctl_row = _verify_live_executable_v2(
        launcher["systemctl"], "systemctl", deadline_ns=deadline_ns
    )
    rfc_executables = rfc_receipt["executables"]
    if type(rfc_executables) is not dict or python_row != rfc_executables["python"]:
        raise ValueError("outer and RFC prerequisite Python executable differ")
    argv = launcher["argv"]
    expected_source = str(
        source_tree_path
        / "src/hegel_machine/phase2b_formal_wire_covert_qualification_v2.py"
    )
    expected_argv = _source_execution_bootstrap_argv_v2(
        python_realpath=python_row["realpath"],
        project_root=source_tree_path,
        qualification_source_sha256=attempt.qualification_source_sha256,
        source_arguments=[
            "--entry",
            expected_entry,
            "--profile",
            str(source_tree_path.parent / "inputs/profile.json"),
            "--attempt",
            str(source_tree_path.parent / "inputs/attempt.json"),
        ],
    )
    if type(argv) is not list or argv != expected_argv or expected_source != str(
        source_tree_path / _BOOTSTRAP_MAIN_RELATIVE_PATH_V2
    ):
        raise ValueError("outer launcher is not the open-once frozen source entry")
    expected_environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    if not _exact_json_equal(launcher["environment"], expected_environment):
        raise ValueError("outer selected environment drifted")
    unit = body["unit"]
    if type(unit) is not dict or set(unit) != {
        "cgroup_path",
        "child_argv_sha256",
        "effective_properties",
        "invocation_id",
        "owned_invocation_id",
        "ownership_marker",
        "ownership_nonce",
        "prelaunch_absent_verified",
        "selected_environment_root",
        "unit_name",
        "working_directory",
    }:
        raise ValueError("outer unit projection is not closed")
    properties = unit["effective_properties"]
    expected_properties = {
        "FinalKillSignal": int(signal.SIGKILL),
        "KillMode": "control-group",
        "KillSignal": int(signal.SIGTERM),
        "MemoryMax": attempt.requested_cgroup_memory_max_bytes,
        "RuntimeMaxUSec": LIVE_HARD_TOTAL_TIMEOUT_SECONDS * 1_000_000,
        "RemainAfterExit": True,
        "SendSIGKILL": True,
        "TasksMax": attempt.requested_cgroup_pids_max,
        "TimeoutStopUSec": 30_000_000,
        "Type": "exec",
    }
    if type(properties) is not dict or not _exact_json_equal(
        properties, expected_properties
    ):
        raise ValueError("outer unit effective properties drifted")
    unit_name = unit["unit_name"]
    ownership_marker = unit["ownership_marker"]
    ownership_nonce = unit["ownership_nonce"]
    child_argv_sha256 = unit["child_argv_sha256"]
    cgroup_path = _require_safe_cgroup_path_v2(
        unit["cgroup_path"], "outer unit cgroup"
    )
    inner_exit = inner["runtime_exit_evidence"]
    if type(inner_exit) is not dict or type(inner_exit.get("cgroup")) is not dict:
        raise ValueError("inner exit cgroup is absent from outer comparison")
    inner_exit_cgroup = inner_exit["cgroup"]
    inner_entry = inner.get("runtime_entry_evidence")
    if type(inner_entry) is not dict:
        raise ValueError("inner entry runtime evidence is absent")
    interpreter_runtime = _validate_interpreter_runtime_projection_v2(
        inner_entry.get("interpreter_runtime"), "outer-bound inner"
    )
    child_snapshots, expected_inner_preopened_root = (
        _replay_extracted_child_preopened_sources_v2(
            source_tree_path,
            attempt.qualification_source_sha256,
            deadline_ns=deadline_ns,
        )
    )
    if (
        outer_preopened_root == expected_inner_preopened_root
        or not _exact_json_equal(
            body["child_source_bootstrap_preopened_snapshots"],
            child_snapshots,
        )
        or body["child_source_bootstrap_preopened_snapshots_root"]
        != expected_inner_preopened_root
    ):
        raise ValueError("outer child preopened source snapshots drifted")
    expected_inner_bootstrap_manifest_root = stable_hash(
        _source_execution_bootstrap_manifest_v2(
            attempt.qualification_source_sha256
        ),
        prefix="phase2b_formal_wire_covert_source_bootstrap_manifest_v2_",
    )
    if (
        type(unit_name) is not str
        or re.fullmatch(r"hegel-qv2-[A-Za-z0-9_-]{8,96}\.service", unit_name)
        is None
        or type(ownership_nonce) is not str
        or re.fullmatch(r"[0-9a-f]{32}", ownership_nonce) is None
        or unit_name
        != "hegel-qv2-"
        + hashlib.sha256(
            (
                attempt.attempt_manifest_root
                + "\x00"
                + expected_entry
                + "\x00"
                + ownership_nonce
            ).encode("ascii")
        ).hexdigest()[:32]
        + ".service"
        or type(unit["invocation_id"]) is not str
        or re.fullmatch(r"[0-9a-f]{32}", unit["invocation_id"]) is None
        or cgroup_path != inner_exit_cgroup["cgroup_path"]
        or unit["owned_invocation_id"] != unit["invocation_id"]
        or unit["prelaunch_absent_verified"] is not True
        or child_argv_sha256
        != hashlib.sha256(canonicalize_rfc8785_v1(argv)).hexdigest()
        or ownership_marker
        != "hegel-qv2-owned-"
        + hashlib.sha256(
            b"HEGEL/PHASE2B/QV2/UNIT_OWNERSHIP/V2\x00"
            + attempt.attempt_manifest_root.encode("ascii")
            + bytes.fromhex(ownership_nonce)
            + bytes.fromhex(child_argv_sha256)
        ).hexdigest()
        or unit["working_directory"] != source_tree
        or unit["selected_environment_root"]
        != attempt.runtime_environment_manifest_root
        or interpreter_runtime["package_directory"]
        != str(source_tree_path / "src/hegel_machine")
        or interpreter_runtime["executed_main_source_sha256"]
        != attempt.qualification_source_sha256
        or interpreter_runtime["preopened_source_snapshot_root"]
        != expected_inner_preopened_root
        or interpreter_runtime["source_execution_bootstrap_manifest_root"]
        != expected_inner_bootstrap_manifest_root
    ):
        raise ValueError("outer unit identity/membership drifted")
    systemd_row = launcher["systemd_run"]
    systemd_argv = launcher["systemd_argv"]
    expected_systemd_argv = [
        systemd_row["realpath"],
        "--user",
        "--pipe",
        "--wait",
        "--quiet",
        f"--unit={unit_name.removesuffix('.service')}",
        f"--property=Description={ownership_marker}",
        "--property=Type=exec",
        "--property=RemainAfterExit=yes",
        f"--property=MemoryMax={attempt.requested_cgroup_memory_max_bytes}",
        f"--property=TasksMax={attempt.requested_cgroup_pids_max}",
        f"--property=RuntimeMaxSec={LIVE_HARD_TOTAL_TIMEOUT_SECONDS}s",
        "--property=KillMode=control-group",
        "--property=KillSignal=SIGTERM",
        "--property=SendSIGKILL=yes",
        "--property=FinalKillSignal=SIGKILL",
        "--property=TimeoutStopSec=30s",
        f"--working-directory={source_tree}",
        "--setenv=LANG=C.UTF-8",
        "--setenv=LC_ALL=C.UTF-8",
        "--",
        *argv,
    ]
    if not _exact_json_equal(systemd_argv, expected_systemd_argv):
        raise ValueError("outer full systemd-run argv drifted")
    entry_show, terminal_show, absent_show = _verify_systemctl_transcript_v2(
        launcher["systemctl_commands"],
        systemctl_realpath=systemctl_row["realpath"],
        unit_name=unit_name,
        attempt=attempt,
        source_tree=source_tree_path,
        ownership_marker=ownership_marker,
    )
    entry_main_pid = _parse_systemd_unsigned_v2(
        entry_show["MainPID"], "outer entry MainPID"
    )
    if (
        entry_main_pid <= 0
        or type(inner_entry.get("cgroup")) is not dict
        or inner_entry["cgroup"].get("cgroup_procs") != [entry_main_pid]
        or inner_exit_cgroup.get("cgroup_procs") != [entry_main_pid]
        or entry_show["InvocationID"] != unit["invocation_id"]
        or entry_show["ControlGroup"] != cgroup_path
        or terminal_show["ControlGroup"] != ""
        or absent_show["LoadState"] != "not-found"
        or not _exact_json_equal(
            _normalize_effective_systemd_properties_v2(
                entry_show, attempt=attempt, source_tree=source_tree_path
            ),
            properties,
        )
    ):
        raise ValueError("systemctl transcript does not bind unit/result evidence")
    runtime_manifest_bytes = _read_bounded_regular_file_v2(
        runtime_manifest_object_path,
        maximum_bytes=1_048_576,
        name="runtime manifest CAS object",
        deadline_ns=deadline_ns,
    )
    _verify_runtime_environment_manifest_v2(
        runtime_manifest_bytes,
        attempt=attempt,
        python_row=python_row,
        systemd_run_row=systemd_run_row,
        systemctl_row=systemctl_row,
    )
    timing = body["timing"]
    if type(timing) is not dict or set(timing) != {
        "elapsed_to_supervisor_ready_nanoseconds",
        "service_ended_monotonic_nanoseconds",
        "started_monotonic_nanoseconds",
        "supervisor_ready_monotonic_nanoseconds",
        "watchdog_deadline_monotonic_nanoseconds",
    }:
        raise ValueError("outer timing projection is not closed")
    start = timing["started_monotonic_nanoseconds"]
    service_end = timing["service_ended_monotonic_nanoseconds"]
    end = timing["supervisor_ready_monotonic_nanoseconds"]
    elapsed = timing["elapsed_to_supervisor_ready_nanoseconds"]
    if (
        _require_exact_int_v2(start, "outer monotonic start") < 0
        or _require_exact_int_v2(
            service_end, "outer service monotonic end"
        )
        < start
        or _require_exact_int_v2(end, "outer monotonic end") < start
        or _require_exact_int_v2(elapsed, "outer monotonic elapsed")
        != end - start
        or end < start
        or end < service_end
        or elapsed > LIVE_HARD_TOTAL_TIMEOUT_SECONDS * 1_000_000_000
        or _require_exact_int_v2(
            timing["watchdog_deadline_monotonic_nanoseconds"],
            "outer watchdog deadline",
        )
        != start + LIVE_HARD_TOTAL_TIMEOUT_SECONDS * 1_000_000_000
    ):
        raise ValueError("outer monotonic hard-deadline evidence drifted")
    result = body["result"]
    expected_result = {
        "control_group_empty": True,
        "control_group_property_at_terminal_snapshot": "",
        "cgroup_path_absent_at_terminal_snapshot": True,
        "exec_main_code": 1,
        "exec_main_status": 0,
        "kill_sent": False,
        "launcher_exit_code": 0,
        "launcher_reaped": True,
        "main_pid_at_terminal_snapshot": 0,
        "result": "success",
        "term_sent": False,
        "timed_out": False,
        "unit_absent_after_collection": True,
        "unit_active_state_at_terminal_snapshot": "active",
        "unit_collected_after_terminal_snapshot": True,
        "unit_sub_state_at_terminal_snapshot": "exited",
    }
    if type(result) is not dict or not _exact_json_equal(result, expected_result):
        raise ValueError("outer systemd/launcher terminal result drifted")
    if Path("/sys/fs/cgroup" + cgroup_path).exists():
        raise ValueError("outer terminal snapshot still has a live control group")
    inner_sha = hashlib.sha256(inner_bytes).hexdigest()
    rfc_sha = hashlib.sha256(rfc_receipt_bytes).hexdigest()
    if not _exact_json_equal(
        body["stdout"], {"sha256": inner_sha, "size_bytes": len(inner_bytes)}
    ):
        raise ValueError("outer stdout does not bind the exact inner artifact")
    stderr_row = body["stderr"]
    if (
        type(stderr_row) is not dict
        or set(stderr_row) != {"sha256", "size_bytes"}
    ):
        raise ValueError("outer stderr projection is not closed")
    if (
        _require_exact_int_v2(stderr_row["size_bytes"], "outer stderr size")
        != 0
        or _require_raw_sha256_v2(stderr_row["sha256"], "outer stderr")
        != hashlib.sha256(b"").hexdigest()
    ):
        raise ValueError("successful outer service must have exact empty stderr")
    _validate_post_output_gate_attestation_v2(
        body["post_output_physical_gate_attestation"],
        attempt,
        inner,
        inner_bytes,
        context="outer",
        expected_entry=expected_entry,
        child_argv_sha256=unit["child_argv_sha256"],
        owned_invocation_id=unit["owned_invocation_id"],
    )
    _verify_cas_store_projection_v2(
        body["inner_artifact_cas"],
        expected_sha256=inner_sha,
        expected_size=len(inner_bytes),
        expected_store_namespace=attempt.attempt_manifest_root,
        name="inner artifact",
    )
    _verify_cas_store_projection_v2(
        body["rfc8785_receipt_cas"],
        expected_sha256=rfc_sha,
        expected_size=len(rfc_receipt_bytes),
        expected_store_namespace=attempt.attempt_manifest_root,
        name="RFC-8785 prerequisite receipt",
    )
    raw_bundle_directory = body["rfc8785_raw_bundle_directory"]
    if type(raw_bundle_directory) is not str or not Path(raw_bundle_directory).is_absolute():
        raise ValueError("RFC raw bundle directory is not absolute")
    fresh_raw_projection = _verify_rfc8785_raw_execution_bundle_v2(
        rfc_receipt,
        Path(raw_bundle_directory),
        expected_qualification_source_sha256=attempt.qualification_source_sha256,
        replay_project_root=source_tree_path,
        fresh_execute=False,
        deadline_ns=deadline_ns,
    )
    if not _exact_json_equal(
        fresh_raw_projection, body["rfc8785_raw_bundle_projection"]
    ):
        raise ValueError("RFC raw bundle projection differs on mechanics replay")


def _validate_outer_supervisor_semantics_pure_v2(
    body: dict[str, object],
    *,
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
    inner: dict[str, object],
    inner_bytes: bytes,
    rfc_receipt: dict[str, object],
    rfc_receipt_bytes: bytes,
    runtime_manifest_bytes: bytes,
    source_tree_rows: list[dict[str, object]],
    source_archive_tree_rows: list[dict[str, object]],
    raw_bundle_projection: dict[str, object],
    expected_entry: str,
    retained_source_tree_fd: int | None = None,
) -> None:
    """Replay outer semantics from retained bytes without producer execution."""

    expected_keys = {
        "attempt_manifest_root",
        "child_source_bootstrap_preopened_snapshots",
        "child_source_bootstrap_preopened_snapshots_root",
        "content_sha256_is_not_a_signature",
        "inner_artifact_cas",
        "launcher",
        "live_files_sha256",
        "mode",
        "outer_runner_tcb_entry",
        "outer_runner_tcb_exit",
        "post_output_physical_gate_attestation",
        "profile_id",
        "result",
        "rfc8785_receipt_cas",
        "rfc8785_raw_bundle_directory",
        "rfc8785_raw_bundle_projection",
        "runtime_environment_manifest_root",
        "runtime_environment_manifest_cas",
        "runtime_environment_manifest_object_path",
        "runtime_image_digest",
        "schema_version",
        "source_archive_cas",
        "source_archive_extraction",
        "source_archive_object_path",
        "source_archive_root",
        "source_commit_sha",
        "source_tree_root",
        "stderr",
        "stdout",
        "timing",
        "unit",
    }
    if (
        type(body) is not dict
        or set(body) != expected_keys
        or body["schema_version"] != OUTER_SUPERVISOR_RESULT_SCHEMA_V2
        or body["mode"] != expected_entry
        or body["profile_id"] != attempt.profile_id
        or body["attempt_manifest_root"] != attempt.attempt_manifest_root
        or body["source_commit_sha"] != attempt.source_commit_sha
        or body["source_archive_root"] != attempt.source_archive_root
        or body["runtime_environment_manifest_root"]
        != attempt.runtime_environment_manifest_root
        or body["runtime_image_digest"] is not None
        or body["content_sha256_is_not_a_signature"] is not True
    ):
        raise ValueError("pure outer supervisor identity fields drifted")
    source_tree_text = body["source_tree_root"]
    if type(source_tree_text) is not str or not Path(source_tree_text).is_absolute():
        raise ValueError("pure outer source-tree display path is malformed")
    source_tree = Path(source_tree_text)
    child_snapshots, expected_inner_preopened_root = (
        _replay_extracted_child_preopened_sources_v2(
            source_tree,
            attempt.qualification_source_sha256,
            retained_source_tree_fd=retained_source_tree_fd,
        )
    )
    expected_files = _expected_live_file_sha256_v2(attempt)
    if not _exact_json_equal(body["live_files_sha256"], expected_files):
        raise ValueError("pure outer live-file manifest drifted")
    tree_root = stable_hash(
        source_tree_rows, prefix="phase2b_formal_wire_covert_source_tree_v2_"
    )
    tree_regular_root = _regular_file_manifest_root_v2(source_tree_rows)
    _verify_source_archive_full_tree_equivalence_v2(
        source_archive_tree_rows,
        stable_hash(
            source_archive_tree_rows,
            prefix="phase2b_formal_wire_covert_source_tree_v2_",
        ),
        source_tree_rows,
        tree_root,
    )
    extraction = body["source_archive_extraction"]
    if (
        type(extraction) is not dict
        or set(extraction)
        != {
            "archive_format",
            "archive_sha256",
            "archive_size_bytes",
            "directory_count",
            "forbidden_bytecode_count",
            "member_count",
            "regular_file_count",
            "regular_file_manifest_root",
            "source_tree_manifest_row_count",
            "source_tree_manifest_root",
            "symlink_hardlink_special_count",
            "total_regular_bytes",
        }
        or extraction["archive_format"]
        != "TAR_REGULAR_AND_DIRECTORY_ENTRIES_ONLY"
        or extraction["archive_sha256"]
        != attempt.source_archive_root.removeprefix("sha256:")
        or _require_exact_int_v2(
            extraction["archive_size_bytes"], "pure archive size", minimum=1
        )
        != attempt.source_archive_size_bytes
        or extraction["source_tree_manifest_root"] != tree_root
        or _require_exact_int_v2(
            extraction["source_tree_manifest_row_count"], "pure source rows"
        )
        != len(source_tree_rows)
        or extraction["regular_file_manifest_root"] != tree_regular_root
        or _require_exact_int_v2(
            extraction["regular_file_count"], "pure source regular count"
        )
        != sum(row.get("type") == "regular" for row in source_tree_rows)
        or _require_exact_int_v2(
            extraction["forbidden_bytecode_count"], "pure archive bytecode count"
        )
        != 0
        or _require_exact_int_v2(
            extraction["symlink_hardlink_special_count"],
            "pure archive special count",
        )
        != 0
    ):
        raise ValueError("pure outer archive/extraction projection drifted")

    source_digest = attempt.source_archive_root.removeprefix("sha256:")
    runtime_digest = attempt.runtime_environment_manifest_root.removeprefix("sha256:")
    inner_sha = hashlib.sha256(inner_bytes).hexdigest()
    rfc_sha = hashlib.sha256(rfc_receipt_bytes).hexdigest()
    for row, digest, size, name in (
        (
            body["source_archive_cas"],
            source_digest,
            attempt.source_archive_size_bytes,
            "pure source archive",
        ),
        (
            body["runtime_environment_manifest_cas"],
            runtime_digest,
            attempt.runtime_environment_manifest_size_bytes,
            "pure runtime manifest",
        ),
        (body["inner_artifact_cas"], inner_sha, len(inner_bytes), "pure inner"),
        (
            body["rfc8785_receipt_cas"],
            rfc_sha,
            len(rfc_receipt_bytes),
            "pure RFC receipt",
        ),
    ):
        _verify_cas_store_projection_v2(
            row,
            expected_sha256=digest,
            expected_size=size,
            expected_store_namespace=attempt.attempt_manifest_root,
            name=name,
        )
    if (
        body["source_archive_object_path"]
        != str(source_tree.parent / "cas" / source_digest)
        or body["runtime_environment_manifest_object_path"]
        != str(source_tree.parent / "cas" / runtime_digest)
        or body["rfc8785_raw_bundle_directory"]
        != str(source_tree.parent / "rfc8785-raw-bundle")
        or not _exact_json_equal(
            body["rfc8785_raw_bundle_projection"], raw_bundle_projection
        )
    ):
        raise ValueError("pure outer retained path/projection binding drifted")

    tcb_entry = body["outer_runner_tcb_entry"]
    tcb_exit = body["outer_runner_tcb_exit"]
    tcb_keys = {
        "broad_package_init_executed",
        "bypassed_package_init_sha256",
        "executed_main_source_bootstrap",
        "interpreter_flags",
        "live_files_sha256",
        "preopened_source_snapshots",
        "source_bootstrap_manifest",
        "source_execution_bootstrap_active",
        "source_only_importer_active",
        "source_realpath",
        "source_sha256",
        "source_tree_root",
    }
    outer_tcb_root_text = (
        tcb_entry.get("source_tree_root") if type(tcb_entry) is dict else None
    )
    outer_tcb_root = (
        Path(outer_tcb_root_text)
        if type(outer_tcb_root_text) is str
        else Path(".")
    )
    if (
        type(tcb_entry) is not dict
        or set(tcb_entry) != tcb_keys
        or not _exact_json_equal(tcb_entry, tcb_exit)
        or tcb_entry["broad_package_init_executed"] is not False
        or tcb_entry["bypassed_package_init_sha256"]
        != BYPASSED_PACKAGE_INIT_SHA256
        or tcb_entry["source_execution_bootstrap_active"] is not True
        or tcb_entry["source_only_importer_active"] is not True
        or tcb_entry["source_sha256"] != attempt.qualification_source_sha256
        or type(outer_tcb_root_text) is not str
        or not outer_tcb_root.is_absolute()
        or outer_tcb_root == source_tree
        or tcb_entry["source_realpath"]
        != str(outer_tcb_root / _BOOTSTRAP_MAIN_RELATIVE_PATH_V2)
        or not _exact_json_equal(tcb_entry["live_files_sha256"], expected_files)
        or not _exact_json_equal(
            tcb_entry["source_bootstrap_manifest"],
            _source_execution_bootstrap_manifest_v2(
                attempt.qualification_source_sha256
            ),
        )
        or not _exact_json_equal(
            tcb_entry["interpreter_flags"],
            {
                "dont_write_bytecode": 1,
                "ignore_environment": 1,
                "isolated": 1,
                "no_site": 1,
            },
        )
    ):
        raise ValueError("pure outer executed-byte TCB projection drifted")
    bootstrap = tcb_entry["executed_main_source_bootstrap"]
    snapshots = tcb_entry["preopened_source_snapshots"]
    manifest = tcb_entry["source_bootstrap_manifest"]
    snapshot_keys = {
        "descriptor_realpath",
        "device",
        "inode",
        "mtime_nanoseconds",
        "sha256",
        "size_bytes",
    }
    bootstrap_keys = snapshot_keys | {
        "bootstrap_kind",
        "reexecuted_from_stable_descriptor",
    }
    if (
        type(bootstrap) is not dict
        or set(bootstrap) != bootstrap_keys
        or bootstrap.get("bootstrap_kind") != "EXTERNAL_PYTHON_C_OPEN_ONCE_V2"
        or bootstrap.get("reexecuted_from_stable_descriptor") is not True
        or bootstrap.get("sha256") != attempt.qualification_source_sha256
        or type(snapshots) is not dict
        or set(snapshots) != set(manifest["sha256"])
    ):
        raise ValueError("pure outer preopened source snapshot set drifted")
    regular_tree_rows = {
        row["path"]: row
        for row in source_tree_rows
        if row.get("type") == "regular"
    }
    for relative, digest in manifest["sha256"].items():
        snapshot = snapshots[relative]
        tree_row = regular_tree_rows.get(relative)
        if (
            type(snapshot) is not dict
            or set(snapshot) != snapshot_keys
            or snapshot.get("sha256") != digest
            or snapshot.get("descriptor_realpath") != str(outer_tcb_root / relative)
            or _require_exact_int_v2(
                snapshot.get("size_bytes"), "pure source snapshot size", minimum=1
            )
            != (tree_row.get("size_bytes") if type(tree_row) is dict else -1)
            or digest != (tree_row.get("sha256") if type(tree_row) is dict else None)
            or _require_exact_int_v2(
                snapshot.get("device"), "pure source snapshot device"
            )
            < 0
            or _require_exact_int_v2(
                snapshot.get("inode"), "pure source snapshot inode", minimum=1
            )
            < 1
            or _require_exact_int_v2(
                snapshot.get("mtime_nanoseconds"),
                "pure source snapshot mtime",
            )
            < 0
        ):
            raise ValueError("pure outer preopened source snapshot drifted")
    if any(
        bootstrap[key] != snapshots[_BOOTSTRAP_MAIN_RELATIVE_PATH_V2][key]
        for key in snapshot_keys
    ):
        raise ValueError("pure main bootstrap differs from its preopened snapshot")
    outer_preopened_root = stable_hash(
        snapshots,
        prefix="phase2b_formal_wire_covert_preopened_sources_v2_",
    )
    if (
        outer_preopened_root == expected_inner_preopened_root
        or not _exact_json_equal(
            body["child_source_bootstrap_preopened_snapshots"],
            child_snapshots,
        )
        or body["child_source_bootstrap_preopened_snapshots_root"]
        != expected_inner_preopened_root
    ):
        raise ValueError("pure outer child preopened source snapshots drifted")
    expected_bootstrap_manifest_root = stable_hash(
        manifest,
        prefix="phase2b_formal_wire_covert_source_bootstrap_manifest_v2_",
    )

    launcher = body["launcher"]
    if type(launcher) is not dict or set(launcher) != {
        "argv",
        "environment",
        "python",
        "systemd_argv",
        "systemd_run",
        "systemctl",
        "systemctl_commands",
    }:
        raise ValueError("pure outer launcher projection is not closed")
    python_row = _verify_executable_projection_shape_v2(
        launcher["python"], "pure outer Python"
    )
    systemd_run_row = _verify_executable_projection_shape_v2(
        launcher["systemd_run"], "pure outer systemd-run"
    )
    systemctl_row = _verify_executable_projection_shape_v2(
        launcher["systemctl"], "pure outer systemctl"
    )
    if not _exact_json_equal(python_row, rfc_receipt["executables"]["python"]):
        raise ValueError("pure outer/RFC Python projections differ")
    expected_argv = _source_execution_bootstrap_argv_v2(
        python_realpath=python_row["realpath"],
        project_root=source_tree,
        qualification_source_sha256=attempt.qualification_source_sha256,
        source_arguments=[
            "--entry",
            expected_entry,
            "--profile",
            str(source_tree.parent / "inputs/profile.json"),
            "--attempt",
            str(source_tree.parent / "inputs/attempt.json"),
        ],
    )
    if (
        launcher["argv"] != expected_argv
        or not _exact_json_equal(
            launcher["environment"], {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
        )
    ):
        raise ValueError("pure outer launcher argv/environment drifted")
    unit = body["unit"]
    if type(unit) is not dict or set(unit) != {
        "cgroup_path",
        "child_argv_sha256",
        "effective_properties",
        "invocation_id",
        "owned_invocation_id",
        "ownership_marker",
        "ownership_nonce",
        "prelaunch_absent_verified",
        "selected_environment_root",
        "unit_name",
        "working_directory",
    }:
        raise ValueError("pure outer unit projection is not closed")
    nonce = unit["ownership_nonce"]
    child_sha = unit["child_argv_sha256"]
    invocation = unit["invocation_id"]
    if (
        type(nonce) is not str
        or re.fullmatch(r"[0-9a-f]{32}", nonce) is None
        or type(child_sha) is not str
        or child_sha != hashlib.sha256(canonicalize_rfc8785_v1(expected_argv)).hexdigest()
        or type(invocation) is not str
        or re.fullmatch(r"[0-9a-f]{32}", invocation) is None
        or unit["owned_invocation_id"] != invocation
        or unit["prelaunch_absent_verified"] is not True
        or unit["unit_name"]
        != "hegel-qv2-"
        + hashlib.sha256(
            (
                attempt.attempt_manifest_root
                + "\x00"
                + expected_entry
                + "\x00"
                + nonce
            ).encode("ascii")
        ).hexdigest()[:32]
        + ".service"
        or unit["ownership_marker"]
        != "hegel-qv2-owned-"
        + hashlib.sha256(
            b"HEGEL/PHASE2B/QV2/UNIT_OWNERSHIP/V2\x00"
            + attempt.attempt_manifest_root.encode("ascii")
            + bytes.fromhex(nonce)
            + bytes.fromhex(child_sha)
        ).hexdigest()
        or unit["working_directory"] != source_tree_text
        or unit["selected_environment_root"]
        != attempt.runtime_environment_manifest_root
    ):
        raise ValueError("pure outer unit ownership identity drifted")
    expected_properties = {
        "FinalKillSignal": int(signal.SIGKILL),
        "KillMode": "control-group",
        "KillSignal": int(signal.SIGTERM),
        "MemoryMax": attempt.requested_cgroup_memory_max_bytes,
        "RemainAfterExit": True,
        "RuntimeMaxUSec": LIVE_HARD_TOTAL_TIMEOUT_SECONDS * 1_000_000,
        "SendSIGKILL": True,
        "TasksMax": attempt.requested_cgroup_pids_max,
        "TimeoutStopUSec": 30_000_000,
        "Type": "exec",
    }
    if not _exact_json_equal(unit["effective_properties"], expected_properties):
        raise ValueError("pure outer effective service properties drifted")
    expected_systemd_argv = [
        systemd_run_row["realpath"],
        "--user",
        "--pipe",
        "--wait",
        "--quiet",
        f"--unit={unit['unit_name'].removesuffix('.service')}",
        f"--property=Description={unit['ownership_marker']}",
        "--property=Type=exec",
        "--property=RemainAfterExit=yes",
        f"--property=MemoryMax={attempt.requested_cgroup_memory_max_bytes}",
        f"--property=TasksMax={attempt.requested_cgroup_pids_max}",
        f"--property=RuntimeMaxSec={LIVE_HARD_TOTAL_TIMEOUT_SECONDS}s",
        "--property=KillMode=control-group",
        "--property=KillSignal=SIGTERM",
        "--property=SendSIGKILL=yes",
        "--property=FinalKillSignal=SIGKILL",
        "--property=TimeoutStopSec=30s",
        f"--working-directory={source_tree_text}",
        "--setenv=LANG=C.UTF-8",
        "--setenv=LC_ALL=C.UTF-8",
        "--",
        *expected_argv,
    ]
    if launcher["systemd_argv"] != expected_systemd_argv:
        raise ValueError("pure outer systemd-run argv drifted")
    entry_show, terminal_show, absent_show = _verify_systemctl_transcript_v2(
        launcher["systemctl_commands"],
        systemctl_realpath=systemctl_row["realpath"],
        unit_name=unit["unit_name"],
        attempt=attempt,
        source_tree=source_tree,
        ownership_marker=unit["ownership_marker"],
    )
    entry_main_pid = _parse_systemd_unsigned_v2(
        entry_show["MainPID"], "pure outer entry MainPID"
    )
    if (
        entry_main_pid <= 0
        or entry_show["InvocationID"] != invocation
        or entry_show["ControlGroup"] != unit["cgroup_path"]
        or terminal_show["InvocationID"] != invocation
        or not _systemd_show_is_exact_absent_v2(absent_show)
    ):
        raise ValueError("pure outer systemctl ownership transcript drifted")
    inner_entry = inner.get("runtime_entry_evidence")
    inner_exit = inner.get("runtime_exit_evidence")
    interpreter_runtime = _validate_interpreter_runtime_projection_v2(
        inner_entry.get("interpreter_runtime"), "pure outer-bound inner"
    ) if type(inner_entry) is dict else None
    if (
        type(inner_entry) is not dict
        or type(inner_exit) is not dict
        or type(inner_entry.get("cgroup")) is not dict
        or type(inner_exit.get("cgroup")) is not dict
        or inner_entry["cgroup"].get("cgroup_path") != unit["cgroup_path"]
        or inner_exit["cgroup"].get("cgroup_path") != unit["cgroup_path"]
        or inner_entry["cgroup"].get("cgroup_procs") != [entry_main_pid]
        or inner_exit["cgroup"].get("cgroup_procs") != [entry_main_pid]
        or type(interpreter_runtime) is not dict
        or interpreter_runtime["package_directory"]
        != str(source_tree / "src/hegel_machine")
        or interpreter_runtime["executed_main_source_sha256"]
        != attempt.qualification_source_sha256
        or interpreter_runtime["preopened_source_snapshot_root"]
        != expected_inner_preopened_root
        or interpreter_runtime["source_execution_bootstrap_manifest_root"]
        != expected_bootstrap_manifest_root
    ):
        raise ValueError("pure outer inner/unit membership drifted")

    runtime = _decode_canonical_transport_object_v2(
        runtime_manifest_bytes, "pure runtime manifest", maximum_bytes=1_048_576
    )
    if (
        set(runtime)
        != {
            "cgroup2",
            "cpu_affinity",
            "dependency_lock_runtime",
            "dependency_lock_sha256",
            "environment_scope",
            "executables",
            "fresh_dedicated_user_service_manager_session_required",
            "host",
            "host_runtime_fingerprint_only_not_full_runtime_image_closure",
            "interpreter_contract",
            "runtime_image_digest",
            "schema_version",
            "selected_locale_overrides",
            "systemd_user_manager_same_uid_is_cooperative_outer_tcb",
        }
        or runtime["schema_version"] != RUNTIME_ENVIRONMENT_MANIFEST_SCHEMA_V2
        or runtime["dependency_lock_sha256"] != DEPENDENCY_LOCK_SHA256
        or runtime["runtime_image_digest"] is not None
        or runtime["host_runtime_fingerprint_only_not_full_runtime_image_closure"]
        is not True
        or runtime["systemd_user_manager_same_uid_is_cooperative_outer_tcb"]
        is not True
        or runtime["fresh_dedicated_user_service_manager_session_required"]
        is not True
        or not _exact_json_equal(
            runtime["executables"],
            {
                "python": python_row,
                "systemd_run": systemd_run_row,
                "systemctl": systemctl_row,
            },
        )
        or not _exact_json_equal(
            runtime["selected_locale_overrides"],
            {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        )
        or not _exact_json_equal(
            runtime["interpreter_contract"],
            {
                "argv_flags": ["-I", "-S", "-B"],
                "ignore_python_environment": True,
                "no_site": True,
                "source_only_loader": True,
                "forbid_pyc_pyo": True,
            },
        )
    ):
        raise ValueError("pure outer runtime manifest semantics drifted")
    dependency_runtime = runtime["dependency_lock_runtime"]
    host = runtime["host"]
    cgroup2 = runtime["cgroup2"]
    affinity = runtime["cpu_affinity"]
    if (
        type(dependency_runtime) is not dict
        or set(dependency_runtime)
        != {
            "lock_platform",
            "platform_machine",
            "platform_system",
            "python_abi",
            "python_cache_tag",
            "python_implementation",
            "python_soabi",
            "python_version",
        }
        or dependency_runtime["lock_platform"] != "linux/amd64"
        or dependency_runtime["platform_system"] != "Linux"
        or dependency_runtime["platform_machine"] not in {"x86_64", "amd64"}
        or dependency_runtime["python_implementation"] != "CPython"
        or dependency_runtime["python_version"] != "3.10.12"
        or dependency_runtime["python_abi"] != "cp310"
        or dependency_runtime["python_cache_tag"] != "cpython-310"
        or type(dependency_runtime["python_soabi"]) is not str
        or not dependency_runtime["python_soabi"].startswith("cpython-310-")
        or runtime["environment_scope"]
        != (
            "ONLY_LANG_AND_LC_ALL_OVERRIDES_ARE_QUALIFIED;"
            "OTHER_USER_MANAGER_ENVIRONMENT_IS_NOT_QUALIFIED"
        )
        or type(host) is not dict
        or set(host)
        != {
            "architecture",
            "kernel_release",
            "kernel_system",
            "kernel_version",
            "libc_name",
            "libc_version",
        }
        or host["architecture"] != dependency_runtime["platform_machine"]
        or host["kernel_system"] != "Linux"
        or any(type(host[key]) is not str for key in host)
        or type(cgroup2) is not dict
        or set(cgroup2) != {"controllers", "membership", "mount"}
        or type(cgroup2["controllers"]) is not list
        or cgroup2["controllers"] != sorted(set(cgroup2["controllers"]))
        or not {"memory", "pids"}.issubset(cgroup2["controllers"])
        or any(
            type(controller) is not str
            or re.fullmatch(r"[a-z_]+", controller) is None
            for controller in cgroup2["controllers"]
        )
        or type(cgroup2["membership"]) is not str
        or re.fullmatch(r"0::/[A-Za-z0-9_./@:-]*", cgroup2["membership"])
        is None
        or not _exact_json_equal(
            cgroup2["mount"],
            {
                "filesystem_type": "cgroup2",
                "mount_options": cgroup2["mount"].get("mount_options")
                if type(cgroup2["mount"]) is dict
                else None,
                "mount_point": "/sys/fs/cgroup",
                "mount_root": "/",
            },
        )
        or type(cgroup2["mount"]) is not dict
        or type(cgroup2["mount"].get("mount_options")) is not str
        or not cgroup2["mount"]["mount_options"]
        or type(affinity) is not list
        or not affinity
        or any(type(cpu) is not int or cpu < 0 for cpu in affinity)
        or affinity != sorted(set(affinity))
        or not _exact_json_equal(inner_entry["cgroup"].get("cpu_affinity"), affinity)
    ):
        raise ValueError("pure outer runtime host/dependency projection drifted")

    timing = body["timing"]
    if type(timing) is not dict or set(timing) != {
        "elapsed_to_supervisor_ready_nanoseconds",
        "service_ended_monotonic_nanoseconds",
        "started_monotonic_nanoseconds",
        "supervisor_ready_monotonic_nanoseconds",
        "watchdog_deadline_monotonic_nanoseconds",
    }:
        raise ValueError("pure outer timing projection is not closed")
    start = _require_exact_int_v2(timing["started_monotonic_nanoseconds"], "pure start")
    service_end = _require_exact_int_v2(
        timing["service_ended_monotonic_nanoseconds"], "pure service end"
    )
    ready = _require_exact_int_v2(
        timing["supervisor_ready_monotonic_nanoseconds"], "pure supervisor ready"
    )
    elapsed = _require_exact_int_v2(
        timing["elapsed_to_supervisor_ready_nanoseconds"],
        "pure elapsed to supervisor ready",
    )
    watchdog = _require_exact_int_v2(
        timing["watchdog_deadline_monotonic_nanoseconds"],
        "pure watchdog deadline",
    )
    if (
        not start <= service_end <= ready
        or elapsed != ready - start
        or watchdog != start + LIVE_HARD_TOTAL_TIMEOUT_SECONDS * 1_000_000_000
        or ready >= watchdog
    ):
        raise ValueError("pure outer deadline evidence drifted")
    expected_result = {
        "control_group_empty": True,
        "control_group_property_at_terminal_snapshot": "",
        "cgroup_path_absent_at_terminal_snapshot": True,
        "exec_main_code": 1,
        "exec_main_status": 0,
        "kill_sent": False,
        "launcher_exit_code": 0,
        "launcher_reaped": True,
        "main_pid_at_terminal_snapshot": 0,
        "result": "success",
        "term_sent": False,
        "timed_out": False,
        "unit_absent_after_collection": True,
        "unit_active_state_at_terminal_snapshot": "active",
        "unit_collected_after_terminal_snapshot": True,
        "unit_sub_state_at_terminal_snapshot": "exited",
    }
    if not _exact_json_equal(body["result"], expected_result):
        raise ValueError("pure outer terminal result drifted")
    if not _exact_json_equal(
        body["stdout"], {"sha256": inner_sha, "size_bytes": len(inner_bytes)}
    ) or not _exact_json_equal(
        body["stderr"],
        {"sha256": hashlib.sha256(b"").hexdigest(), "size_bytes": 0},
    ):
        raise ValueError("pure outer stdout/stderr binding drifted")
    _validate_post_output_gate_attestation_v2(
        body["post_output_physical_gate_attestation"],
        attempt,
        inner,
        inner_bytes,
        context="pure outer",
        expected_entry=expected_entry,
        child_argv_sha256=unit["child_argv_sha256"],
        owned_invocation_id=unit["owned_invocation_id"],
    )


def _verify_phase2b_formal_wire_covert_outer_mechanics_core_v2(
    profile: Phase2BFormalWireCovertQualificationProfileV2,
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
    *,
    inner_artifact_bytes: bytes,
    outer_supervisor_result_bytes: bytes,
    rfc8785_prerequisite_receipt_bytes: bytes,
    calibration_basis_profile_bytes: bytes | None = None,
    calibration_attempt_manifest_bytes: bytes | None = None,
    calibration_inner_artifact_bytes: bytes | None = None,
    calibration_outer_supervisor_result_bytes: bytes | None = None,
    calibration_rfc8785_prerequisite_receipt_bytes: bytes | None = None,
    calibration_mechanics_artifact_bytes: bytes | None = None,
    _calibration_pair_replay_token: object | None = None,
    _calibration_pair_projection: object | None = None,
    _deadline_ns: int | None = None,
) -> bytes:
    """Private retained-pair-aware mechanics verifier core."""

    if type(profile) is not Phase2BFormalWireCovertQualificationProfileV2:
        raise TypeError("outer mechanics verifier needs the exact profile")
    if (
        _calibration_pair_replay_token is not None
        and type(_calibration_pair_replay_token)
        is not _VerifiedCalibrationPairReplayTokenV2
    ):
        raise TypeError(
            "outer mechanics pure replay needs a retained-directory verified "
            "calibration pair token"
        )
    profile._validate()
    attempt.__post_init__()
    if attempt.profile_id != PROFILE_ID:
        raise ValueError("outer mechanics fresh attempt does not bind current profile")
    shape = _EXPECTED_PROFILE["fixture_contract"]["rich_positive_shape"]
    if type(shape) is not dict:
        _fail(FAIL_PROFILE, "outer mechanics cannot find the rich-shape profile")
    rfc_receipt = _verify_rfc8785_prerequisite_receipt_v2(
        rfc8785_prerequisite_receipt_bytes,
        attempt,
        deadline_ns=_deadline_ns,
    )
    inner = _decode_canonical_transport_object_v2(
        inner_artifact_bytes, "inner artifact", maximum_bytes=8 * 1_024 * 1_024
    )
    inner_schema = inner.get("schema_version")
    if inner_schema == (
        "hegel-machine-phase2b-formal-wire-covert-development-calibration/6"
    ):
        if (
            calibration_basis_profile_bytes is not None
            or calibration_attempt_manifest_bytes is not None
            or calibration_inner_artifact_bytes is not None
            or calibration_outer_supervisor_result_bytes is not None
            or calibration_rfc8785_prerequisite_receipt_bytes is not None
            or calibration_mechanics_artifact_bytes is not None
            or _calibration_pair_replay_token is not None
            or _calibration_pair_projection is not None
        ):
            raise ValueError("pending calibration cannot consume prior calibration bytes")
        _require_pending_calibration_profile_v2(shape)
        expected_entry = "development-calibration"
        _validate_development_calibration_artifact_v2(inner, attempt)
    elif inner_schema == (
        "hegel-machine-phase2b-formal-wire-covert-live-inner-candidate/6"
    ):
        if shape.get("live_qualification_enabled") is not True:
            _fail(FAIL_PROFILE, "live outer mechanics needs the frozen profile")
        expected_entry = "live-inner"
        if (
            type(calibration_basis_profile_bytes) is not bytes
            or type(calibration_attempt_manifest_bytes) is not bytes
            or type(calibration_inner_artifact_bytes) is not bytes
            or type(calibration_outer_supervisor_result_bytes) is not bytes
            or type(calibration_rfc8785_prerequisite_receipt_bytes) is not bytes
            or type(calibration_mechanics_artifact_bytes) is not bytes
        ):
            _fail(
                FAIL_PROFILE,
                "live mechanics needs both frozen calibration canonical artifacts",
            )
        _validate_frozen_calibration_bytes_v2(
            shape,
            calibration_basis_profile_bytes=calibration_basis_profile_bytes,
            calibration_attempt_manifest_bytes=calibration_attempt_manifest_bytes,
            calibration_inner_artifact_bytes=calibration_inner_artifact_bytes,
            calibration_outer_supervisor_result_bytes=(
                calibration_outer_supervisor_result_bytes
            ),
            calibration_rfc8785_prerequisite_receipt_bytes=(
                calibration_rfc8785_prerequisite_receipt_bytes
            ),
            calibration_mechanics_artifact_bytes=(
                calibration_mechanics_artifact_bytes
            ),
            _pair_replay_token=_calibration_pair_replay_token,
            _pair_projection=_calibration_pair_projection,
            _pair_replay_purpose="outer-mechanics",
            deadline_ns=_deadline_ns,
        )
        _validate_live_inner_artifact_v2(inner, attempt)
    else:
        raise ValueError("outer mechanics received an unknown inner artifact schema")
    supervisor = _decode_canonical_transport_object_v2(
        outer_supervisor_result_bytes,
        "outer supervisor result",
        maximum_bytes=1_048_576,
    )
    _validate_outer_supervisor_result_v2(
        supervisor,
        attempt=attempt,
        inner=inner,
        inner_bytes=inner_artifact_bytes,
        rfc_receipt=rfc_receipt,
        rfc_receipt_bytes=rfc8785_prerequisite_receipt_bytes,
        expected_entry=expected_entry,
        deadline_ns=_deadline_ns,
    )
    body = {
        "schema_version": MECHANICS_VERIFIER_SCHEMA_V2,
        "disposition": "MECHANICS_VERIFIED_NOT_AUTHORITY",
        "entry_kind": expected_entry,
        "profile_id": PROFILE_ID,
        "attempt_manifest_root": attempt.attempt_manifest_root,
        "inner_artifact_sha256": hashlib.sha256(inner_artifact_bytes).hexdigest(),
        "outer_supervisor_result_sha256": hashlib.sha256(
            outer_supervisor_result_bytes
        ).hexdigest(),
        "rfc8785_prerequisite_receipt_sha256": hashlib.sha256(
            rfc8785_prerequisite_receipt_bytes
        ).hexdigest(),
        "claims": {
            "r03_formal_trusted_wire_implementation_qualified": False,
            "r04_formal_covert_auditor_implementation_qualified": False,
            "r05_synthetic_dev_wire_covert_qualification_verified": False,
            "actual_corpus_audit": False,
            "actual_unsealed_960_generation": False,
            "origin_authenticated": False,
            "key_independence_verified": False,
            "campaign_execution_authorized": False,
        },
        "authority_issuer_present": False,
        "calibration_artifact_cannot_be_promoted": (
            expected_entry == "development-calibration"
        ),
        "projection_only_boundary": dict(_TASK4_PROJECTION_ONLY_BOUNDARY_V2),
        "content_sha256_is_not_a_signature": True,
        "step5_signed_cas_authority_still_required": True,
    }
    return _canonical_inner_artifact_v2(body)


def verify_phase2b_formal_wire_covert_outer_mechanics_v2(
    profile: Phase2BFormalWireCovertQualificationProfileV2,
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
    *,
    inner_artifact_bytes: bytes,
    outer_supervisor_result_bytes: bytes,
    rfc8785_prerequisite_receipt_bytes: bytes,
    calibration_basis_profile_bytes: bytes | None = None,
    calibration_attempt_manifest_bytes: bytes | None = None,
    calibration_inner_artifact_bytes: bytes | None = None,
    calibration_outer_supervisor_result_bytes: bytes | None = None,
    calibration_rfc8785_prerequisite_receipt_bytes: bytes | None = None,
    calibration_mechanics_artifact_bytes: bytes | None = None,
) -> bytes:
    """Verify public mechanics; retained-pair bypass authority is not accepted."""

    return _verify_phase2b_formal_wire_covert_outer_mechanics_core_v2(
        profile,
        attempt,
        inner_artifact_bytes=inner_artifact_bytes,
        outer_supervisor_result_bytes=outer_supervisor_result_bytes,
        rfc8785_prerequisite_receipt_bytes=(
            rfc8785_prerequisite_receipt_bytes
        ),
        calibration_basis_profile_bytes=calibration_basis_profile_bytes,
        calibration_attempt_manifest_bytes=calibration_attempt_manifest_bytes,
        calibration_inner_artifact_bytes=calibration_inner_artifact_bytes,
        calibration_outer_supervisor_result_bytes=(
            calibration_outer_supervisor_result_bytes
        ),
        calibration_rfc8785_prerequisite_receipt_bytes=(
            calibration_rfc8785_prerequisite_receipt_bytes
        ),
        calibration_mechanics_artifact_bytes=calibration_mechanics_artifact_bytes,
    )


class _AtomicPublicationUncertainV2(RuntimeError):
    """A no-replace link committed; callers must not publish another terminal."""

    def __init__(
        self,
        target_name: str,
        message: str,
        *,
        parent_identity: tuple[int, int] | None = None,
        target_identity: tuple[int, int] | None = None,
        target_sha256: str | None = None,
        temporary_name: str | None = None,
        publication_state: str | None = None,
    ) -> None:
        super().__init__(message)
        self.target_name = target_name
        self.parent_identity = parent_identity
        self.target_identity = target_identity
        self.target_sha256 = target_sha256
        self.temporary_name = temporary_name
        self.publication_state = publication_state


def _atomic_publication_uncertainty_after_failure_v2(
    directory_fd: int,
    *,
    target_name: str,
    temporary_name: str,
    staged_stat: os.stat_result | None,
    expected_sha256: str,
    linked_hint: bool,
    message: str,
) -> _AtomicPublicationUncertainV2 | None:
    """Reconcile the kernel namespace even across link-to-marker interruption."""

    _require_raw_sha256_v2(expected_sha256, "atomic intended target")
    if staged_stat is None:
        return None
    parent_stat = os.fstat(directory_fd)
    expected_identity = (staged_stat.st_dev, staged_stat.st_ino)
    target_matches = False
    target_exists = False
    try:
        target_stat = os.stat(
            target_name, dir_fd=directory_fd, follow_symlinks=False
        )
        target_exists = True
        target_matches = (
            stat.S_ISREG(target_stat.st_mode)
            and (target_stat.st_dev, target_stat.st_ino) == expected_identity
        )
    except FileNotFoundError:
        pass
    temporary_matches = False
    try:
        temporary_stat = os.stat(
            temporary_name, dir_fd=directory_fd, follow_symlinks=False
        )
        temporary_matches = (
            stat.S_ISREG(temporary_stat.st_mode)
            and (temporary_stat.st_dev, temporary_stat.st_ino)
            == expected_identity
        )
    except FileNotFoundError:
        pass
    # ``linked_hint`` covers later replay/fsync failures.  ``target_matches``
    # closes the asynchronous exception window after the kernel link succeeds
    # but before Python stores ``linked = True``.
    if not linked_hint and not target_matches and not temporary_matches:
        return None
    if target_matches:
        publication_state = "TARGET_LINK_COMMITTED"
    elif temporary_matches and not target_exists and not linked_hint:
        publication_state = "STAGED_TEMP_ONLY"
    else:
        publication_state = "COLLISION_OR_COMMIT_STATE_UNRESOLVED"
    return _AtomicPublicationUncertainV2(
        target_name,
        message,
        parent_identity=(parent_stat.st_dev, parent_stat.st_ino),
        target_identity=expected_identity,
        target_sha256=expected_sha256,
        temporary_name=temporary_name,
        publication_state=publication_state,
    )


def _atomic_uncertainty_blocks_another_terminal_v2(exc: object) -> bool:
    return (
        type(exc) is _AtomicPublicationUncertainV2
        and exc.target_name == "terminal.json"
    )


def _verify_parent_identity_after_terminal_commit_v2(
    output_root_fd: int,
    attempt_name: str,
    attempt_fd: int,
    *,
    name: str,
    deadline_ns: int,
) -> None:
    try:
        if time.monotonic_ns() >= deadline_ns:
            raise TimeoutError(
                f"{name} parent identity started after the hard deadline"
            )
        _verify_retained_child_directory_identity_at_fd_v2(
            output_root_fd, attempt_name, attempt_fd, name=name
        )
        if time.monotonic_ns() >= deadline_ns:
            raise TimeoutError(
                f"{name} parent identity completed after the hard deadline"
            )
    except BaseException as exc:
        raise _AtomicPublicationUncertainV2(
            "terminal.json",
            f"{name} terminal committed before parent-entry identity failure",
        ) from exc


class _OuterCleanupUnprovenV2(RuntimeError):
    """A claims-false terminal exists, but external cleanup remains unproven."""


class _DirectoryCreationCommittedV2(RuntimeError):
    """mkdir committed; the retained child fd, when present, must be finalized."""

    def __init__(
        self, name: str, descriptor: int | None, cause: BaseException
    ) -> None:
        super().__init__(f"directory creation committed before closure: {name}")
        self.name = name
        self.descriptor = descriptor
        self.__cause__ = cause


def _append_exception_note_v2(exc: BaseException, message: str) -> None:
    """Attach cleanup context on both pinned Python 3.10 and newer runtimes."""

    add_note = getattr(exc, "add_note", None)
    if callable(add_note):
        add_note(message)
        return
    if exc.args:
        exc.args = (f"{exc.args[0]} [note: {message}]", *exc.args[1:])
    else:
        exc.args = (f"note: {message}",)


def _close_owned_descriptors_v2(
    descriptors: Iterable[int | None], *, name: str
) -> None:
    """Close every owned fd; never let one close hide the remaining fds."""

    primary = sys.exc_info()[1]
    errors: list[BaseException] = []
    seen: set[int] = set()
    for descriptor in descriptors:
        if descriptor is None or descriptor in seen:
            continue
        seen.add(descriptor)
        try:
            os.close(descriptor)
        except BaseException as exc:
            errors.append(exc)
    if not errors:
        return
    message = f"{name} could not close {len(errors)} owned descriptors"
    if primary is not None:
        _append_exception_note_v2(primary, message)
        return
    raise RuntimeError(message) from errors[0]


def _open_real_directory_fd_v2(path: Path) -> int:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError("publisher directory must be an absolute real directory")
    if str(path.resolve(strict=True)) != str(path):
        raise ValueError("publisher directory must already be a realpath")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("qualification publisher requires Linux no-follow dirfd flags")
    flags |= os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or os.path.realpath(f"/proc/self/fd/{descriptor}") != str(path)
        ):
            raise ValueError("publisher descriptor is not a directory")
        return descriptor
    except BaseException:
        _close_owned_descriptors_v2(
            (descriptor,), name="real directory acquisition"
        )
        raise


def _write_all_fd_v2(
    descriptor: int, payload: bytes, *, deadline_ns: int | None = None
) -> None:
    offset = 0
    while offset < len(payload):
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("publisher write reached the absolute deadline")
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("publisher write made no progress")
        offset += written


def _open_directory_at_fd_v2(
    parent_fd: int, name: str, *, expected_path: Path | None = None
) -> int:
    if (
        type(name) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}", name) is None
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
    ):
        raise ValueError("retained directory child is outside the closed grammar")
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode):
            raise ValueError("retained directory child is not a directory")
        if (
            expected_path is not None
            and os.path.realpath(f"/proc/self/fd/{descriptor}")
            != str(expected_path)
        ):
            raise ValueError("retained directory child realpath drifted")
        return descriptor
    except BaseException:
        _close_owned_descriptors_v2(
            (descriptor,), name="retained child directory acquisition"
        )
        raise


def _open_and_retain_directory_at_fd_v2(
    owned: list[int],
    parent_fd: int,
    name: str,
    *,
    expected_path: Path | None = None,
) -> int:
    """Acquire exactly one directory and transfer ownership immediately."""

    descriptor = _open_directory_at_fd_v2(
        parent_fd, name, expected_path=expected_path
    )
    owned.append(descriptor)
    return descriptor


def _directory_names_at_fd_v2(directory_fd: int) -> tuple[str, ...]:
    if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
        raise ValueError("retained directory descriptor is not a directory")
    names = os.listdir(directory_fd)
    if (
        len(names) > _MAXIMUM_RETAINED_ATTEMPT_TREE_ROWS_V2
        or any(type(name) is not str or "\x00" in name for name in names)
    ):
        raise ValueError("retained directory contains an invalid name")
    return tuple(sorted(names))


def _verify_retained_directory_path_identity_v2(
    directory_fd: int, path: Path, *, name: str
) -> None:
    if not path.is_absolute():
        raise ValueError(f"{name} retained path is not absolute")
    retained = os.fstat(directory_fd)
    live = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISDIR(retained.st_mode)
        or not stat.S_ISDIR(live.st_mode)
        or (retained.st_dev, retained.st_ino) != (live.st_dev, live.st_ino)
        or os.path.realpath(f"/proc/self/fd/{directory_fd}") != str(path)
    ):
        raise ValueError(f"{name} retained-dirfd/path identity drifted")


def _verify_retained_child_directory_identity_at_fd_v2(
    parent_fd: int, child_name: str, retained_child_fd: int, *, name: str
) -> None:
    opened = _open_directory_at_fd_v2(parent_fd, child_name)
    try:
        expected = os.fstat(retained_child_fd)
        actual = os.fstat(opened)
        if (
            not stat.S_ISDIR(expected.st_mode)
            or (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino)
        ):
            raise ValueError(f"{name} parent entry no longer names retained inode")
    finally:
        os.close(opened)


def _read_regular_at_fd_v2(
    directory_fd: int,
    name: str,
    *,
    maximum_bytes: int,
    deadline_ns: int | None = None,
) -> tuple[bytes, os.stat_result]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("qualification publisher requires O_NOFOLLOW")
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    try:
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("publisher replay reached its absolute deadline")
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink < 1
            or not 0 <= before.st_size <= maximum_bytes
        ):
            raise ValueError("publisher replay target is not a bounded regular file")
        blocks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("publisher replay reached its absolute deadline")
            block = os.read(descriptor, min(remaining, 65_536))
            if not block:
                raise ValueError("publisher replay target ended early")
            blocks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise ValueError("publisher replay target grew")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("publisher replay target mutated")
        return b"".join(blocks), after
    finally:
        os.close(descriptor)


def _open_relative_regular_at_fd_v2(root_fd: int, relative: str) -> int:
    """Open one regular file beneath a retained root without path re-resolution."""

    if (
        type(relative) is not str
        or relative.startswith("/")
        or "\\" in relative
        or "\x00" in relative
    ):
        raise ValueError("retained relative regular path is malformed")
    parts = relative.split("/")
    if not parts or any(
        part in {"", ".", ".."}
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}", part) is None
        for part in parts
    ):
        raise ValueError("retained relative regular path is outside the grammar")
    parent_fd = os.dup(root_fd)
    descriptor: int | None = None
    try:
        try:
            for component in parts[:-1]:
                next_fd = _open_directory_at_fd_v2(parent_fd, component)
                try:
                    os.close(parent_fd)
                except BaseException:
                    _close_owned_descriptors_v2(
                        (next_fd,), name="retained relative directory transfer"
                    )
                    raise
                parent_fd = next_fd
            descriptor = os.open(
                parts[-1],
                os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        finally:
            _close_owned_descriptors_v2(
                (parent_fd,), name="retained relative parent traversal"
            )
    except BaseException:
        _close_owned_descriptors_v2(
            (descriptor,), name="retained relative target after parent failure"
        )
        raise
    if descriptor is None:
        raise AssertionError("retained relative descriptor acquisition vanished")
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink < 1:
            raise ValueError("retained relative target is not a regular file")
        return descriptor
    except BaseException:
        _close_owned_descriptors_v2(
            (descriptor,), name="retained relative regular acquisition"
        )
        raise


def _read_relative_regular_at_fd_v2(
    root_fd: int,
    relative: str,
    *,
    maximum_bytes: int,
    deadline_ns: int | None = None,
) -> tuple[bytes, os.stat_result]:
    descriptor = _open_relative_regular_at_fd_v2(root_fd, relative)
    try:
        before = os.fstat(descriptor)
        if not 0 <= before.st_size <= maximum_bytes:
            raise ValueError("retained relative regular file exceeds its bound")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("retained relative read reached its deadline")
            block = os.read(descriptor, min(remaining, 65_536))
            if not block:
                raise ValueError("retained relative regular file ended early")
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise ValueError("retained relative regular file grew")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("retained relative regular file mutated")
        return b"".join(chunks), after
    finally:
        os.close(descriptor)


def _hash_relative_regular_at_fd_v2(
    root_fd: int,
    relative: str,
    *,
    maximum_bytes: int,
    deadline_ns: int | None = None,
) -> tuple[str, os.stat_result]:
    descriptor = _open_relative_regular_at_fd_v2(root_fd, relative)
    try:
        before = os.fstat(descriptor)
        if not 0 <= before.st_size <= maximum_bytes:
            raise ValueError("retained relative hash target exceeds its bound")
        digest = hashlib.sha256()
        total = 0
        while True:
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("retained relative hash reached its deadline")
            block = os.read(descriptor, 1_048_576)
            if not block:
                break
            total += len(block)
            if total > before.st_size:
                raise ValueError("retained relative hash target grew")
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            total != before.st_size
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise ValueError("retained relative hash target mutated")
        return digest.hexdigest(), after
    finally:
        os.close(descriptor)


def _hash_regular_at_fd_v2(
    directory_fd: int,
    name: str,
    *,
    maximum_bytes: int,
    deadline_ns: int | None = None,
) -> tuple[str, os.stat_result]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("qualification replay requires O_NOFOLLOW")
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink < 1
            or not 0 <= before.st_size <= maximum_bytes
        ):
            raise ValueError("streaming replay target is not a bounded regular file")
        digest = hashlib.sha256()
        total = 0
        while True:
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("streaming replay reached the absolute deadline")
            block = os.read(descriptor, 1_048_576)
            if not block:
                break
            total += len(block)
            if total > before.st_size:
                raise ValueError("streaming replay target grew")
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            total != before.st_size
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise ValueError("streaming replay target mutated")
        return digest.hexdigest(), after
    finally:
        os.close(descriptor)


def _atomic_write_new_file_at_fd_v2(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    deadline_ns: int | None = None,
) -> os.stat_result:
    if (
        type(name) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}", name) is None
        or type(payload) is not bytes
    ):
        raise ValueError("atomic retained-dirfd target is outside the closed grammar")
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("qualification publisher requires O_NOFOLLOW")
    if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
        raise TimeoutError("atomic retained-dirfd publication reached its deadline")
    if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
        raise ValueError("atomic retained-dirfd parent is not a directory")
    temporary_name = f".{name}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    descriptor: int | None = None
    linked = False
    written: os.stat_result | None = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        _write_all_fd_v2(descriptor, payload, deadline_ns=deadline_ns)
        os.fsync(descriptor)
        written = os.fstat(descriptor)
        if (
            not stat.S_ISREG(written.st_mode)
            or written.st_nlink != 1
            or written.st_size != len(payload)
        ):
            raise ValueError("atomic retained-dirfd temporary file drifted")
        os.close(descriptor)
        descriptor = None
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("atomic retained-dirfd link reached its deadline")
        os.link(
            temporary_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        linked = True
        os.unlink(temporary_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
        replay, replay_stat = _read_regular_at_fd_v2(
            directory_fd,
            name,
            maximum_bytes=max(1, len(payload)),
            deadline_ns=deadline_ns,
        )
        if replay != payload or replay_stat.st_ino != written.st_ino:
            raise ValueError("atomic retained-dirfd published replay drifted")
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("atomic retained-dirfd fsync/replay reached its deadline")
        return replay_stat
    except BaseException as primary:
        cleanup_error: BaseException | None = None
        if written is None and descriptor is not None:
            try:
                candidate_stat = os.fstat(descriptor)
                if stat.S_ISREG(candidate_stat.st_mode):
                    written = candidate_stat
            except BaseException:
                pass
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as exc:
                cleanup_error = exc
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
        if written is None:
            try:
                candidate_stat = os.stat(
                    temporary_name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if stat.S_ISREG(candidate_stat.st_mode):
                    written = candidate_stat
            except BaseException:
                pass
        uncertainty = _atomic_publication_uncertainty_after_failure_v2(
            directory_fd,
            target_name=name,
            temporary_name=temporary_name,
            staged_stat=written,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            linked_hint=linked,
            message=(
                f"atomic publication of {name} committed before a later failure"
            ),
        )
        if uncertainty is not None:
            if cleanup_error is not None:
                _append_exception_note_v2(
                    uncertainty,
                    "temporary-file cleanup also failed: "
                    + type(cleanup_error).__name__
                )
            raise uncertainty from primary
        if cleanup_error is not None:
            cleanup_failure = RuntimeError(
                f"atomic publication of {name} failed and cleanup was incomplete"
            )
            _append_exception_note_v2(
                cleanup_failure, type(cleanup_error).__name__
            )
            raise cleanup_failure from primary
        raise


def _atomic_write_new_file_v2(
    directory: Path,
    name: str,
    payload: bytes,
    *,
    deadline_ns: int | None = None,
) -> Path:
    directory_fd = _open_real_directory_fd_v2(directory)
    try:
        _atomic_write_new_file_at_fd_v2(
            directory_fd,
            name,
            payload,
            deadline_ns=deadline_ns,
        )
    finally:
        _close_owned_descriptors_v2(
            (directory_fd,), name="atomic path publisher"
        )
    return directory / name


def _publish_terminal_once_at_fd_v2(
    attempt_fd: int, payload: bytes, *, deadline_ns: int
) -> os.stat_result:
    names_before = _directory_names_at_fd_v2(attempt_fd)
    if (
        "terminal.json" in names_before
        or any(name.startswith(".terminal.json.tmp-") for name in names_before)
    ):
        raise ValueError("attempt already has a terminal publication")
    published = _atomic_write_new_file_at_fd_v2(
        attempt_fd,
        "terminal.json",
        payload,
        deadline_ns=deadline_ns,
    )
    try:
        names_after = _directory_names_at_fd_v2(attempt_fd)
        if names_after.count("terminal.json") != 1 or any(
            name.startswith(".terminal.json.tmp-") for name in names_after
        ):
            raise ValueError(
                "terminal retained directory enumeration drifted after link"
            )
        replay, replay_stat = _read_regular_at_fd_v2(
            attempt_fd,
            "terminal.json",
            maximum_bytes=max(1, len(payload)),
            deadline_ns=deadline_ns,
        )
        if replay != payload or replay_stat.st_ino != published.st_ino:
            raise ValueError("terminal retained-dirfd replay drifted after link")
        return replay_stat
    except BaseException as exc:
        raise _AtomicPublicationUncertainV2(
            "terminal.json",
            "terminal link committed before a later enumeration/replay failure",
        ) from exc


def _retime_verified_outer_terminal_v2(
    verified_payload: bytes,
    *,
    prospective_replay_completed_ns: int,
    prospective_deadline_ns: int,
) -> bytes:
    """Change only the ready/elapsed pair after full prospective replay."""

    completed = _require_exact_int_v2(
        prospective_replay_completed_ns,
        "prospective replay completion",
    )
    cutoff = _require_exact_int_v2(
        prospective_deadline_ns,
        "prospective replay deadline",
        minimum=1,
    )
    terminal = _decode_canonical_transport_object_v2(
        verified_payload,
        "prospectively verified outer terminal",
        maximum_bytes=8 * 1_024 * 1_024,
    )
    schema = terminal.get("schema_version")
    timing = terminal.get("timing")
    if type(timing) is not dict:
        raise ValueError("prospectively verified terminal timing is absent")
    if schema == OUTER_ATTEMPT_SUCCESS_SCHEMA_V2:
        expected_keys = {
            "elapsed_to_finalization_ready_nanoseconds",
            "finalization_ready_monotonic_nanoseconds",
            "service_ended_monotonic_nanoseconds",
            "started_monotonic_nanoseconds",
            "watchdog_deadline_monotonic_nanoseconds",
        }
        ready_key = "finalization_ready_monotonic_nanoseconds"
        elapsed_key = "elapsed_to_finalization_ready_nanoseconds"
    elif schema == OUTER_ATTEMPT_FAILURE_SCHEMA_V2:
        expected_keys = {
            "elapsed_nanoseconds_at_finalization_ready",
            "finalization_ready_monotonic_nanoseconds",
            "started_monotonic_nanoseconds",
            "watchdog_deadline_monotonic_nanoseconds",
        }
        ready_key = "finalization_ready_monotonic_nanoseconds"
        elapsed_key = "elapsed_nanoseconds_at_finalization_ready"
    else:
        raise ValueError("prospective terminal schema cannot be timing-finalized")
    if set(timing) != expected_keys:
        raise ValueError("prospective terminal timing fields are not exact")
    start = _require_exact_int_v2(
        timing["started_monotonic_nanoseconds"], "prospective terminal start"
    )
    previous_ready = _require_exact_int_v2(
        timing[ready_key], "prospective terminal prior ready"
    )
    prior_elapsed = _require_exact_int_v2(
        timing[elapsed_key], "prospective terminal prior elapsed"
    )
    watchdog = _require_exact_int_v2(
        timing["watchdog_deadline_monotonic_nanoseconds"],
        "prospective terminal watchdog",
    )
    if (
        prior_elapsed != previous_ready - start
        or watchdog != start
        + LIVE_HARD_TOTAL_TIMEOUT_SECONDS * 1_000_000_000
        or not start <= previous_ready <= completed < cutoff <= watchdog
    ):
        raise ValueError("prospective terminal timing-only transition is invalid")
    updated_timing = dict(timing)
    updated_timing[ready_key] = completed
    updated_timing[elapsed_key] = completed - start
    updated = dict(terminal)
    updated["timing"] = updated_timing
    # This helper has exactly two writable leaves; every other terminal value
    # remains the object that the full retained-directory verifier accepted.
    return _canonical_inner_artifact_v2(updated)


def _mkdir_new_directory_at_fd_v2(
    parent_fd: int,
    name: str,
    *,
    expected_path: Path | None = None,
    deadline_ns: int | None = None,
) -> int:
    if (
        type(name) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}", name) is None
    ):
        raise ValueError("new directory name is outside the closed grammar")
    if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
        raise TimeoutError("new directory creation reached its absolute deadline")
    created = False
    child_fd: int | None = None
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        created = True
        child_fd = _open_directory_at_fd_v2(
            parent_fd, name, expected_path=expected_path
        )
        os.fsync(child_fd)
        os.fsync(parent_fd)
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("new directory fsync reached its absolute deadline")
        return child_fd
    except BaseException as exc:
        if created:
            raise _DirectoryCreationCommittedV2(name, child_fd, exc) from exc
        if child_fd is not None:
            os.close(child_fd)
        raise


def _mkdir_new_directory_v2(
    parent: Path,
    name: str,
    *,
    deadline_ns: int | None = None,
) -> Path:
    parent_fd = _open_real_directory_fd_v2(parent)
    child_fd: int | None = None
    try:
        child_fd = _mkdir_new_directory_at_fd_v2(
            parent_fd,
            name,
            expected_path=parent / name,
            deadline_ns=deadline_ns,
        )
    except _DirectoryCreationCommittedV2 as exc:
        if exc.descriptor is not None:
            _close_owned_descriptors_v2(
                (exc.descriptor,), name="committed directory wrapper"
            )
            exc.descriptor = None
        raise
    finally:
        _close_owned_descriptors_v2(
            (child_fd, parent_fd), name="directory wrapper"
        )
    return parent / name


def _cas_projection_v2(
    digest: str, size: int, store_namespace: str
) -> dict[str, object]:
    return {
        "content_sha256": digest,
        "object_key": "sha256:" + digest,
        "object_size_bytes": size,
        "prior_object_count": 0,
        "put_exit_code": 0,
        "replay_verified": True,
        "resulting_object_count": 1,
        "store_namespace": store_namespace,
    }


def _cas_put_bytes_v2(
    cas_directory: Path,
    payload: bytes,
    *,
    store_namespace: str,
    deadline_ns: int | None = None,
) -> dict[str, object]:
    cas_fd = _open_real_directory_fd_v2(cas_directory)
    try:
        return _cas_put_bytes_at_fd_v2(
            cas_fd,
            payload,
            store_namespace=store_namespace,
            deadline_ns=deadline_ns,
        )
    finally:
        _close_owned_descriptors_v2((cas_fd,), name="CAS byte wrapper")


def _cas_put_bytes_at_fd_v2(
    cas_fd: int,
    payload: bytes,
    *,
    store_namespace: str,
    deadline_ns: int | None = None,
) -> dict[str, object]:
    if type(payload) is not bytes or not payload:
        raise ValueError("CAS byte object must be nonempty exact bytes")
    digest = hashlib.sha256(payload).hexdigest()
    _atomic_write_new_file_at_fd_v2(
        cas_fd, digest, payload, deadline_ns=deadline_ns
    )
    return _cas_projection_v2(digest, len(payload), store_namespace)


def _failure_child_pipe_bundle_bytes_v2(stdout: bytes, stderr: bytes) -> bytes:
    if (
        type(stdout) is not bytes
        or type(stderr) is not bytes
        or len(stdout) > 8 * 1_024 * 1_024
        or len(stderr) > 256 * 1_024
    ):
        raise ValueError("failure child pipe bytes exceed their exact caps")
    return (
        _FAILURE_CHILD_PIPE_CAPTURE_BUNDLE_DOMAIN_V2
        + len(stdout).to_bytes(8, "big")
        + len(stderr).to_bytes(8, "big")
        + stdout
        + stderr
    )


def _decode_failure_child_pipe_bundle_v2(payload: bytes) -> tuple[bytes, bytes]:
    prefix_size = len(_FAILURE_CHILD_PIPE_CAPTURE_BUNDLE_DOMAIN_V2)
    if (
        type(payload) is not bytes
        or len(payload) < prefix_size + 16
        or payload[:prefix_size] != _FAILURE_CHILD_PIPE_CAPTURE_BUNDLE_DOMAIN_V2
    ):
        raise ValueError("failure child pipe CAS bundle framing drifted")
    stdout_size = int.from_bytes(payload[prefix_size : prefix_size + 8], "big")
    stderr_size = int.from_bytes(payload[prefix_size + 8 : prefix_size + 16], "big")
    body = payload[prefix_size + 16 :]
    if (
        stdout_size > 8 * 1_024 * 1_024
        or stderr_size > 256 * 1_024
        or len(body) != stdout_size + stderr_size
    ):
        raise ValueError("failure child pipe CAS bundle lengths drifted")
    return body[:stdout_size], body[stdout_size:]


def _ensure_attempt_private_cas_bytes_at_fd_v2(
    cas_fd: int,
    payload: bytes,
    *,
    store_namespace: str,
    deadline_ns: int,
) -> dict[str, object]:
    """Publish or exact-replay one nonempty attempt-private CAS object."""

    if type(payload) is not bytes or not payload:
        raise ValueError("attempt-private CAS bundle must be nonempty bytes")
    digest = hashlib.sha256(payload).hexdigest()
    names = set(_directory_names_at_fd_v2(cas_fd))
    if digest in names:
        replay, replay_stat = _read_regular_at_fd_v2(
            cas_fd,
            digest,
            maximum_bytes=len(payload),
            deadline_ns=deadline_ns,
        )
        if replay != payload or replay_stat.st_size != len(payload):
            raise ValueError("attempt-private CAS dedup replay drifted")
    else:
        _atomic_write_new_file_at_fd_v2(
            cas_fd, digest, payload, deadline_ns=deadline_ns
        )
    return _cas_projection_v2(digest, len(payload), store_namespace)


def _failure_pipe_file_reference_v2(path: str, payload: bytes) -> dict[str, object]:
    if (
        type(path) is not str
        or re.fullmatch(
            r"evidence/failure-child-(?:stdout|stderr)(?:-prefix)?\.bin", path
        )
        is None
        or type(payload) is not bytes
    ):
        raise ValueError("failure child pipe file reference is malformed")
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _persist_failure_child_pipe_prefix_v2(
    evidence_fd: int,
    *,
    stdout: bytes,
    stderr: bytes,
    deadline_ns: int,
) -> dict[str, object]:
    """Durably retain the selector-consumed prefix before any cleanup control."""

    if len(stdout) > 8 * 1_024 * 1_024 or len(stderr) > 256 * 1_024:
        raise ValueError("failure child pipe prefix exceeds its exact cap")
    rows: dict[str, object] = {}
    for stream, payload in (("stdout", stdout), ("stderr", stderr)):
        name = f"failure-child-{stream}-prefix.bin"
        _atomic_write_new_file_at_fd_v2(
            evidence_fd, name, payload, deadline_ns=deadline_ns
        )
        rows[stream] = _failure_pipe_file_reference_v2(
            "evidence/" + name, payload
        )
    os.fsync(evidence_fd)
    if time.monotonic_ns() >= deadline_ns:
        raise TimeoutError("failure child pipe prefix fsync reached its deadline")
    return rows


def _persist_failure_child_pipe_capture_v2(
    evidence_fd: int,
    cas_fd: int,
    *,
    stdout: bytes,
    stderr: bytes,
    prefix_projection: dict[str, object],
    capture_complete: bool,
    capture_truncated: bool,
    store_namespace: str,
    deadline_ns: int,
) -> dict[str, object]:
    if (
        type(capture_complete) is not bool
        or type(capture_truncated) is not bool
        or (capture_complete and capture_truncated)
        or (
            capture_truncated
            and len(stdout) != 8 * 1_024 * 1_024
            and len(stderr) != 256 * 1_024
        )
        or type(prefix_projection) is not dict
    ):
        raise ValueError("failure child pipe capture arguments are malformed")
    stream_rows: dict[str, object] = {}
    for stream, payload, cap in (
        ("stdout", stdout, 8 * 1_024 * 1_024),
        ("stderr", stderr, 256 * 1_024),
    ):
        prefix_row = prefix_projection.get(stream)
        if type(prefix_row) is not dict or set(prefix_row) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("failure child pipe prefix projection is not closed")
        prefix_name = f"failure-child-{stream}-prefix.bin"
        prefix_bytes, _ = _read_regular_at_fd_v2(
            evidence_fd,
            prefix_name,
            maximum_bytes=cap,
            deadline_ns=deadline_ns,
        )
        if (
            prefix_row["path"] != "evidence/" + prefix_name
            or prefix_row["sha256"] != hashlib.sha256(prefix_bytes).hexdigest()
            or _require_exact_int_v2(
                prefix_row["size_bytes"],
                f"failure {stream} prefix size",
                maximum=cap,
            )
            != len(prefix_bytes)
            or type(payload) is not bytes
            or len(payload) > cap
            or not payload.startswith(prefix_bytes)
        ):
            raise ValueError("failure child pipe prefix/final splice drifted")
        final_name = f"failure-child-{stream}.bin"
        _atomic_write_new_file_at_fd_v2(
            evidence_fd, final_name, payload, deadline_ns=deadline_ns
        )
        stream_rows[stream] = {
            "final": _failure_pipe_file_reference_v2(
                "evidence/" + final_name, payload
            ),
            "prefix": prefix_row,
            "tail_size_bytes": len(payload) - len(prefix_bytes),
        }
    bundle = _failure_child_pipe_bundle_bytes_v2(stdout, stderr)
    bundle_cas = _ensure_attempt_private_cas_bytes_at_fd_v2(
        cas_fd,
        bundle,
        store_namespace=store_namespace,
        deadline_ns=deadline_ns,
    )
    core = {
        "schema_version": _FAILURE_CHILD_PIPE_CAPTURE_SCHEMA_V2,
        "bundle_cas": bundle_cas,
        "capture_boundary": (
            "COMPLETE_EOF_AND_LAUNCHER_REAPED"
            if capture_complete
            else (
                "INCOMPLETE_TRUNCATED_AT_STREAM_CAP"
                if capture_truncated
                else "INCOMPLETE_PREFIX_OR_PARTIAL_DRAIN_ONLY"
            )
        ),
        "capture_complete": capture_complete,
        "capture_persistence_verified": True,
        "capture_truncated_at_cap": capture_truncated,
        "prefix_persisted_before_cleanup": True,
        "stderr": stream_rows["stderr"],
        "stdout": stream_rows["stdout"],
        "systemd_run_pipe_relay_bytes_not_independent_child_authentication": True,
    }
    return {
        **core,
        "capture_root": stable_hash(
            core, prefix="phase2b_formal_wire_covert_failure_pipe_capture_v1_"
        ),
    }


def _failure_child_pipe_persistence_unverified_projection_v2(
    evidence_fd: int,
    *,
    persistence_error_type: str,
    capture_truncated: bool,
    deadline_ns: int,
) -> dict[str, object]:
    if (
        type(persistence_error_type) is not str
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", persistence_error_type)
        is None
        or type(capture_truncated) is not bool
    ):
        raise ValueError("failure pipe persistence error type is malformed")
    names = set(_directory_names_at_fd_v2(evidence_fd))
    partial_names = {
        name for name in names if name.startswith("failure-child-")
    }
    stdout_prefix_name = "failure-child-stdout-prefix.bin"
    stderr_prefix_name = "failure-child-stderr-prefix.bin"
    stdout_final_name = "failure-child-stdout.bin"
    stderr_final_name = "failure-child-stderr.bin"
    reachable_sets = (
        set(),
        {stdout_prefix_name},
        {stdout_prefix_name, stderr_prefix_name},
        {stdout_prefix_name, stderr_prefix_name, stdout_final_name},
        {
            stdout_prefix_name,
            stderr_prefix_name,
            stdout_final_name,
            stderr_final_name,
        },
    )
    if partial_names not in reachable_sets:
        raise ValueError("failure pipe partial publication set is unreachable")
    partial: list[dict[str, object]] = []
    payloads: dict[str, bytes] = {}
    ordered_names = (
        stdout_prefix_name,
        stdout_final_name,
        stderr_prefix_name,
        stderr_final_name,
    )
    for name in ordered_names:
        if name not in partial_names:
            continue
        cap = 8 * 1_024 * 1_024 if "stdout" in name else 256 * 1_024
        payload, _ = _read_regular_at_fd_v2(
            evidence_fd, name, maximum_bytes=cap, deadline_ns=deadline_ns
        )
        payloads[name] = payload
        partial.append(
            _failure_pipe_file_reference_v2("evidence/" + name, payload)
        )
    if (
        stdout_final_name in payloads
        and not payloads[stdout_final_name].startswith(payloads[stdout_prefix_name])
    ) or (
        stderr_final_name in payloads
        and not payloads[stderr_final_name].startswith(payloads[stderr_prefix_name])
    ):
        raise ValueError("failure pipe partial final does not extend its prefix")
    core = {
        "schema_version": _FAILURE_CHILD_PIPE_CAPTURE_SCHEMA_V2,
        "capture_boundary": "PERSISTENCE_UNVERIFIED_NO_COMPLETE_CAPTURE_CLAIM",
        "capture_complete": False,
        "capture_persistence_verified": False,
        "capture_truncated_at_cap": capture_truncated,
        "persistence_error_type": persistence_error_type,
        "retained_partial_artifacts": partial,
        "systemd_run_pipe_relay_bytes_not_independent_child_authentication": True,
    }
    return {
        **core,
        "capture_root": stable_hash(
            core, prefix="phase2b_formal_wire_covert_failure_pipe_capture_v1_"
        ),
    }


def _verify_failure_child_pipe_capture_projection_v2(
    evidence_fd: int,
    capture: object,
    *,
    attempt_manifest_root: str,
    deadline_ns: int | None,
) -> tuple[bytes | None, bytes | None, bool]:
    if type(capture) is not dict:
        raise ValueError("launched /failure/4 lacks child pipe capture evidence")
    common = {
        "capture_boundary",
        "capture_complete",
        "capture_persistence_verified",
        "capture_root",
        "capture_truncated_at_cap",
        "schema_version",
        "systemd_run_pipe_relay_bytes_not_independent_child_authentication",
    }
    if (
        capture.get("schema_version") != _FAILURE_CHILD_PIPE_CAPTURE_SCHEMA_V2
        or type(capture.get("capture_complete")) is not bool
        or type(capture.get("capture_persistence_verified")) is not bool
        or type(capture.get("capture_truncated_at_cap")) is not bool
        or (
            capture.get("capture_complete") is True
            and capture.get("capture_truncated_at_cap") is True
        )
        or capture.get(
            "systemd_run_pipe_relay_bytes_not_independent_child_authentication"
        )
        is not True
    ):
        raise ValueError("failure child pipe capture header drifted")
    observed_failure_names = {
        name
        for name in _directory_names_at_fd_v2(evidence_fd)
        if name.startswith("failure-child-")
    }
    if capture["capture_persistence_verified"] is False:
        expected = common | {"persistence_error_type", "retained_partial_artifacts"}
        partial = capture.get("retained_partial_artifacts")
        error_type = capture.get("persistence_error_type")
        if (
            set(capture) != expected
            or capture["capture_complete"] is not False
            or capture["capture_boundary"]
            != "PERSISTENCE_UNVERIFIED_NO_COMPLETE_CAPTURE_CLAIM"
            or type(error_type) is not str
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", error_type) is None
            or type(partial) is not list
            or len(partial) > 4
        ):
            raise ValueError("unverified failure pipe persistence shape drifted")
        stdout_prefix_name = "failure-child-stdout-prefix.bin"
        stderr_prefix_name = "failure-child-stderr-prefix.bin"
        stdout_final_name = "failure-child-stdout.bin"
        stderr_final_name = "failure-child-stderr.bin"
        reachable_sets = (
            set(),
            {stdout_prefix_name},
            {stdout_prefix_name, stderr_prefix_name},
            {stdout_prefix_name, stderr_prefix_name, stdout_final_name},
            {
                stdout_prefix_name,
                stderr_prefix_name,
                stdout_final_name,
                stderr_final_name,
            },
        )
        if observed_failure_names not in reachable_sets:
            raise ValueError("unverified failure pipe object set is unreachable")
        expected_rows: list[dict[str, object]] = []
        payloads: dict[str, bytes] = {}
        for index, name in enumerate(
            (
                stdout_prefix_name,
                stdout_final_name,
                stderr_prefix_name,
                stderr_final_name,
            )
        ):
            if name not in observed_failure_names:
                continue
            cap = 8 * 1_024 * 1_024 if "stdout" in name else 256 * 1_024
            payload, _ = _read_regular_at_fd_v2(
                evidence_fd, name, maximum_bytes=cap, deadline_ns=deadline_ns
            )
            payloads[name] = payload
            expected_rows.append(
                _failure_pipe_file_reference_v2("evidence/" + name, payload)
            )
        if (
            stdout_final_name in payloads
            and not payloads[stdout_final_name].startswith(
                payloads[stdout_prefix_name]
            )
        ) or (
            stderr_final_name in payloads
            and not payloads[stderr_final_name].startswith(
                payloads[stderr_prefix_name]
            )
        ):
            raise ValueError("unverified failure pipe final/prefix splice drifted")
        if not _exact_json_equal(partial, expected_rows):
            raise ValueError("partial failure pipe artifact order/bytes drifted")
        if observed_failure_names != {
            row["path"].removeprefix("evidence/") for row in expected_rows
        }:
            raise ValueError("unverified failure pipe retained object set drifted")
        core = {key: capture[key] for key in capture if key != "capture_root"}
        if capture["capture_root"] != stable_hash(
            core, prefix="phase2b_formal_wire_covert_failure_pipe_capture_v1_"
        ):
            raise ValueError("unverified failure pipe capture root drifted")
        return None, None, False

    expected = common | {
        "bundle_cas",
        "prefix_persisted_before_cleanup",
        "stderr",
        "stdout",
    }
    if (
        set(capture) != expected
        or capture["prefix_persisted_before_cleanup"] is not True
        or capture["capture_boundary"]
        != (
            "COMPLETE_EOF_AND_LAUNCHER_REAPED"
            if capture["capture_complete"]
            else (
                "INCOMPLETE_TRUNCATED_AT_STREAM_CAP"
                if capture["capture_truncated_at_cap"]
                else "INCOMPLETE_PREFIX_OR_PARTIAL_DRAIN_ONLY"
            )
        )
        or observed_failure_names
        != {
            "failure-child-stderr-prefix.bin",
            "failure-child-stderr.bin",
            "failure-child-stdout-prefix.bin",
            "failure-child-stdout.bin",
        }
    ):
        raise ValueError("verified failure pipe capture shape/object set drifted")
    payloads: dict[str, bytes] = {}
    for stream, cap in (("stdout", 8 * 1_024 * 1_024), ("stderr", 256 * 1_024)):
        row = capture.get(stream)
        if type(row) is not dict or set(row) != {"final", "prefix", "tail_size_bytes"}:
            raise ValueError("failure pipe stream projection is not closed")
        prefix_row = row["prefix"]
        final_row = row["final"]
        if (
            type(prefix_row) is not dict
            or type(final_row) is not dict
            or set(prefix_row) != {"path", "sha256", "size_bytes"}
            or set(final_row) != {"path", "sha256", "size_bytes"}
            or prefix_row["path"] != f"evidence/failure-child-{stream}-prefix.bin"
            or final_row["path"] != f"evidence/failure-child-{stream}.bin"
        ):
            raise ValueError("failure pipe stream paths drifted")
        prefix_bytes, _ = _read_regular_at_fd_v2(
            evidence_fd,
            f"failure-child-{stream}-prefix.bin",
            maximum_bytes=cap,
            deadline_ns=deadline_ns,
        )
        final_bytes, _ = _read_regular_at_fd_v2(
            evidence_fd,
            f"failure-child-{stream}.bin",
            maximum_bytes=cap,
            deadline_ns=deadline_ns,
        )
        if (
            prefix_row["sha256"] != hashlib.sha256(prefix_bytes).hexdigest()
            or final_row["sha256"] != hashlib.sha256(final_bytes).hexdigest()
            or _require_exact_int_v2(
                prefix_row["size_bytes"], f"failure {stream} prefix size", maximum=cap
            )
            != len(prefix_bytes)
            or _require_exact_int_v2(
                final_row["size_bytes"], f"failure {stream} final size", maximum=cap
            )
            != len(final_bytes)
            or not final_bytes.startswith(prefix_bytes)
            or _require_exact_int_v2(
                row["tail_size_bytes"], f"failure {stream} tail size", maximum=cap
            )
            != len(final_bytes) - len(prefix_bytes)
        ):
            raise ValueError("failure pipe stream bytes/prefix splice drifted")
        payloads[stream] = final_bytes
    if (
        capture["capture_truncated_at_cap"]
        and len(payloads["stdout"]) != 8 * 1_024 * 1_024
        and len(payloads["stderr"]) != 256 * 1_024
    ):
        raise ValueError("truncated failure capture did not reach a stream cap")
    bundle_cas = capture["bundle_cas"]
    if type(bundle_cas) is not dict or set(bundle_cas) != {
        "content_sha256",
        "object_key",
        "object_size_bytes",
        "prior_object_count",
        "put_exit_code",
        "replay_verified",
        "resulting_object_count",
        "store_namespace",
    }:
        raise ValueError("failure pipe bundle CAS projection is not closed")
    expected_bundle = _failure_child_pipe_bundle_bytes_v2(
        payloads["stdout"], payloads["stderr"]
    )
    digest = hashlib.sha256(expected_bundle).hexdigest()
    if (
        bundle_cas["content_sha256"] != digest
        or bundle_cas["object_key"] != "sha256:" + digest
        or _require_exact_int_v2(
            bundle_cas["object_size_bytes"], "failure pipe bundle CAS size", minimum=1
        )
        != len(expected_bundle)
        or _require_exact_int_v2(
            bundle_cas["prior_object_count"],
            "failure pipe bundle prior object count",
            maximum=0,
        )
        != 0
        or _require_exact_int_v2(
            bundle_cas["put_exit_code"],
            "failure pipe bundle put exit code",
            maximum=0,
        )
        != 0
        or bundle_cas["replay_verified"] is not True
        or _require_exact_int_v2(
            bundle_cas["resulting_object_count"],
            "failure pipe bundle resulting object count",
            minimum=1,
            maximum=1,
        )
        != 1
        or bundle_cas["store_namespace"] != attempt_manifest_root
    ):
        raise ValueError("failure pipe bundle CAS identity drifted")
    cas_fd = _open_directory_at_fd_v2(evidence_fd, "cas")
    try:
        retained_bundle, _ = _read_regular_at_fd_v2(
            cas_fd,
            digest,
            maximum_bytes=len(expected_bundle),
            deadline_ns=deadline_ns,
        )
    finally:
        _close_owned_descriptors_v2((cas_fd,), name="failure pipe CAS replay")
    if retained_bundle != expected_bundle:
        raise ValueError("failure pipe bundle CAS bytes drifted")
    decoded_stdout, decoded_stderr = _decode_failure_child_pipe_bundle_v2(
        retained_bundle
    )
    if decoded_stdout != payloads["stdout"] or decoded_stderr != payloads["stderr"]:
        raise ValueError("failure pipe bundle/raw stream splice drifted")
    core = {key: capture[key] for key in capture if key != "capture_root"}
    if capture["capture_root"] != stable_hash(
        core, prefix="phase2b_formal_wire_covert_failure_pipe_capture_v1_"
    ):
        raise ValueError("failure child pipe capture root drifted")
    return payloads["stdout"], payloads["stderr"], capture["capture_complete"]


def _cas_put_file_v2(
    cas_directory: Path,
    source: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    store_namespace: str,
    name: str,
    deadline_ns: int | None = None,
    retained_cas_fd: int | None = None,
) -> dict[str, object]:
    if (
        not source.is_absolute()
        or type(expected_size) is not int
        or not 1 <= expected_size <= 2**31
    ):
        raise ValueError(f"{name} CAS source is outside runner bounds")
    _require_raw_sha256_v2(expected_sha256, f"{name} CAS source")
    source_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("qualification CAS requires O_NOFOLLOW")
    source_flags |= os.O_NOFOLLOW
    destination_flags |= os.O_NOFOLLOW
    source_descriptor = os.open(source, source_flags)
    try:
        cas_fd = (
            os.dup(retained_cas_fd)
            if retained_cas_fd is not None
            else _open_real_directory_fd_v2(cas_directory)
        )
    except BaseException:
        _close_owned_descriptors_v2(
            (source_descriptor,), name=f"{name} CAS acquisition"
        )
        raise
    temporary_name = (
        f".{expected_sha256}.tmp-{os.getpid()}-{time.monotonic_ns()}"
    )
    destination_descriptor: int | None = None
    linked = False
    destination_stat: os.stat_result | None = None
    try:
        if not stat.S_ISDIR(os.fstat(cas_fd).st_mode):
            raise ValueError(f"{name} retained CAS descriptor is not a directory")
        before = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size != expected_size
            or os.path.realpath(f"/proc/self/fd/{source_descriptor}")
            != str(source)
        ):
            raise ValueError(f"{name} CAS source identity drifted")
        destination_descriptor = os.open(
            temporary_name,
            destination_flags,
            0o600,
            dir_fd=cas_fd,
        )
        digest = hashlib.sha256()
        copied = 0
        while True:
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                raise TimeoutError(f"{name} CAS copy reached the absolute deadline")
            block = os.read(source_descriptor, 1_048_576)
            if not block:
                break
            copied += len(block)
            if copied > expected_size:
                raise ValueError(f"{name} CAS source grew")
            digest.update(block)
            _write_all_fd_v2(
                destination_descriptor, block, deadline_ns=deadline_ns
            )
        after = os.fstat(source_descriptor)
        if (
            copied != expected_size
            or digest.hexdigest() != expected_sha256
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise ValueError(f"{name} CAS source hash/identity drifted")
        os.fsync(destination_descriptor)
        destination_stat = os.fstat(destination_descriptor)
        if (
            not stat.S_ISREG(destination_stat.st_mode)
            or destination_stat.st_size != expected_size
            or destination_stat.st_nlink != 1
        ):
            raise ValueError(f"{name} CAS staged object drifted")
        os.close(destination_descriptor)
        destination_descriptor = None
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            raise TimeoutError(f"{name} CAS link reached the absolute deadline")
        os.link(
            temporary_name,
            expected_sha256,
            src_dir_fd=cas_fd,
            dst_dir_fd=cas_fd,
            follow_symlinks=False,
        )
        linked = True
        os.unlink(temporary_name, dir_fd=cas_fd)
        os.fsync(cas_fd)
        replay_digest, replay_stat = _hash_regular_at_fd_v2(
            cas_fd,
            expected_sha256,
            maximum_bytes=expected_size,
            deadline_ns=deadline_ns,
        )
        if (
            replay_stat.st_size != expected_size
            or replay_digest != expected_sha256
            or replay_stat.st_ino != destination_stat.st_ino
        ):
            raise ValueError(f"{name} CAS retained-dirfd replay drifted")
    except BaseException as primary:
        cleanup_error: BaseException | None = None
        if destination_stat is None and destination_descriptor is not None:
            try:
                candidate_stat = os.fstat(destination_descriptor)
                if stat.S_ISREG(candidate_stat.st_mode):
                    destination_stat = candidate_stat
            except BaseException:
                pass
        if destination_descriptor is not None:
            try:
                os.close(destination_descriptor)
            except BaseException as exc:
                cleanup_error = exc
        try:
            os.unlink(temporary_name, dir_fd=cas_fd)
        except FileNotFoundError:
            pass
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
        if destination_stat is None:
            try:
                candidate_stat = os.stat(
                    temporary_name,
                    dir_fd=cas_fd,
                    follow_symlinks=False,
                )
                if stat.S_ISREG(candidate_stat.st_mode):
                    destination_stat = candidate_stat
            except BaseException:
                pass
        uncertainty = _atomic_publication_uncertainty_after_failure_v2(
            cas_fd,
            target_name=expected_sha256,
            temporary_name=temporary_name,
            staged_stat=destination_stat,
            expected_sha256=expected_sha256,
            linked_hint=linked,
            message=f"CAS publication of {name} committed before a later failure",
        )
        if uncertainty is not None:
            if cleanup_error is not None:
                _append_exception_note_v2(
                    uncertainty,
                    "CAS temporary cleanup also failed: "
                    + type(cleanup_error).__name__
                )
            raise uncertainty from primary
        if cleanup_error is not None:
            cleanup_failure = RuntimeError(
                f"CAS publication of {name} failed and cleanup was incomplete"
            )
            _append_exception_note_v2(
                cleanup_failure, type(cleanup_error).__name__
            )
            raise cleanup_failure from primary
        raise
    finally:
        _close_owned_descriptors_v2(
            (source_descriptor, cas_fd), name=f"{name} CAS copy"
        )
    return _cas_projection_v2(expected_sha256, expected_size, store_namespace)


def _executable_projection_for_runner_v2(
    executable: Path,
    *,
    version_argv: tuple[str, ...],
    name: str,
    deadline_ns: int,
) -> dict[str, object]:
    realpath = executable.resolve(strict=True)
    if realpath.is_symlink() or not realpath.is_file():
        raise ValueError(f"{name} executable is not a regular realpath")
    version_evidence, stdout, stderr = _run_bounded_control_command_v2(
        [str(realpath), *version_argv],
        purpose=f"{name}-version",
        deadline_ns=min(deadline_ns, time.monotonic_ns() + 10_000_000_000),
        stdout_cap=65_536,
        stderr_cap=65_536,
    )
    if version_evidence["returncode"] != 0:
        raise ValueError(f"{name} version probe failed")
    version = (stdout + stderr).decode(
        "utf-8", errors="strict"
    ).strip()
    row = {
        "execution_path_security": _executable_path_security_projection_v2(realpath),
        "realpath": str(realpath),
        "sha256": _sha256_file_v2(realpath, deadline_ns=deadline_ns),
        "version": version,
    }
    _verify_live_executable_v2(row, name, deadline_ns=deadline_ns)
    return row


def _read_bounded_proc_text_v2(path: Path, *, maximum_bytes: int, name: str) -> str:
    if not path.is_absolute() or not 1 <= maximum_bytes <= 1_048_576:
        raise ValueError(f"{name} proc read contract is malformed")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("runtime manifest requires O_NOFOLLOW")
    descriptor = os.open(path, flags | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{name} is not a proc regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, 16_384)
            if not block:
                break
            total += len(block)
            if total > maximum_bytes:
                raise ValueError(f"{name} exceeded its read cap")
            chunks.append(block)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError(f"{name} descriptor identity drifted")
    finally:
        os.close(descriptor)
    return b"".join(chunks).decode("utf-8", errors="strict")


def _cgroup2_host_projection_v2() -> dict[str, object]:
    mountinfo = _read_bounded_proc_text_v2(
        Path("/proc/self/mountinfo"), maximum_bytes=1_048_576, name="mountinfo"
    )
    matches: list[dict[str, str]] = []
    for line in mountinfo.splitlines():
        fields = line.split(" ")
        try:
            separator = fields.index("-")
        except ValueError:
            raise ValueError("mountinfo row lacks its separator") from None
        if separator + 2 >= len(fields):
            raise ValueError("mountinfo row is truncated")
        if fields[separator + 1] == "cgroup2":
            matches.append(
                {
                    "mount_point": fields[4],
                    "mount_root": fields[3],
                    "mount_options": fields[5],
                    "filesystem_type": fields[separator + 1],
                }
            )
    if len(matches) != 1 or matches[0]["mount_point"] != "/sys/fs/cgroup":
        raise ValueError("host does not expose one exact /sys/fs/cgroup cgroup2 mount")
    if matches[0]["mount_root"] != "/":
        raise ValueError("host cgroup2 mount root is not exact slash")
    membership_text = _read_bounded_proc_text_v2(
        Path("/proc/self/cgroup"), maximum_bytes=65_536, name="self cgroup"
    )
    membership_lines = membership_text.splitlines()
    if (
        len(membership_lines) != 1
        or re.fullmatch(r"0::/[A-Za-z0-9_./@:-]*", membership_lines[0]) is None
        or any(part in {".", ".."} for part in Path(membership_lines[0][3:]).parts)
    ):
        raise ValueError("host unified cgroup membership is not closed")
    controllers = _read_bounded_proc_text_v2(
        Path("/sys/fs/cgroup/cgroup.controllers"),
        maximum_bytes=65_536,
        name="cgroup controllers",
    ).strip().split()
    if (
        len(controllers) != len(set(controllers))
        or not {"memory", "pids"}.issubset(controllers)
        or any(re.fullmatch(r"[a-z_]+", item) is None for item in controllers)
    ):
        raise ValueError("host cgroup controller list is malformed")
    return {
        "controllers": sorted(controllers),
        "membership": membership_lines[0],
        "mount": matches[0],
    }


def _live_dependency_lock_runtime_projection_v2() -> dict[str, object]:
    implementation = platform.python_implementation()
    version = platform.python_version()
    cache_tag = sys.implementation.cache_tag
    soabi = sysconfig.get_config_var("SOABI")
    machine = platform.machine()
    system_name = platform.system()
    if (
        implementation != "CPython"
        or version != "3.10.12"
        or cache_tag != "cpython-310"
        or type(soabi) is not str
        or not soabi.startswith("cpython-310-")
        or system_name != "Linux"
        or machine not in {"x86_64", "amd64"}
    ):
        raise ValueError("live interpreter/platform differs from dependency lock")
    return {
        "lock_platform": "linux/amd64",
        "platform_machine": machine,
        "platform_system": system_name,
        "python_abi": "cp310",
        "python_cache_tag": cache_tag,
        "python_implementation": implementation,
        "python_soabi": soabi,
        "python_version": version,
    }


def _verify_dependency_lock_bytes_v2(payload: bytes) -> dict[str, object]:
    if hashlib.sha256(payload).hexdigest() != DEPENDENCY_LOCK_SHA256:
        raise ValueError("dependency lock bytes differ from the pinned SHA-256")
    decoded = _decode_canonical_transport_object_v2(
        payload, "qualification dependency lock", maximum_bytes=256 * 1_024
    )
    if set(decoded) != {"dependency_lock", "schema_version"}:
        raise ValueError("dependency lock root fields are not exact")
    lock = decoded["dependency_lock"]
    expected_lock = {
        "execution_bootstrap": {
            "broad_package_init_bypassed": True,
            "bypassed_package_init_sha256_non_tcb": (
                BYPASSED_PACKAGE_INIT_SHA256
            ),
            "direct_file_open_once_bootstrap_required": True,
            "python_argv_flags": ["-I", "-S", "-B"],
            "source_only_loader_required": True,
        },
        "host_runtime_manifest_scope": (
            "COOPERATIVE_FINGERPRINT_NOT_FULL_LIBPYTHON_STDLIB_NATIVE_CLOSURE"
        ),
        "platform": "linux/amd64",
        "project": "hegel-machine-phase2b-formal-wire-covert-qualification-v2",
        "project_local_dependencies": {
            "auditor": {
                "config_sha256": AUDITOR_CONFIG_SHA256,
                "documentation_sha256": AUDITOR_DOCUMENTATION_SHA256,
                "policy_binding_id": AUDITOR_POLICY_BINDING_ID_V2,
                "profile_schema_version": AUDITOR_PROFILE_SCHEMA_VERSION_V2,
                "schema_version": AUDITOR_SCHEMA_VERSION_V2,
                "semantics_id": AUDITOR_SEMANTICS_ID_V2,
                "source_sha256": AUDITOR_SOURCE_SHA256,
                "test_sha256": AUDITOR_TEST_SHA256,
            },
            "exact_transform_core": {
                "source_sha256": EXACT_CORE_SOURCE_SHA256,
                "test_sha256": EXACT_CORE_TEST_SHA256,
            },
            "rfc8785": {
                "documentation_sha256": RFC8785_DOCUMENTATION_SHA256,
                "golden_vectors_sha256": RFC8785_GOLDEN_VECTORS_SHA256,
                "node_oracle_sha256": RFC8785_NODE_ORACLE_SHA256,
                "source_sha256": RFC8785_SOURCE_SHA256,
                "test_sha256": RFC8785_TEST_SHA256,
            },
            "runtime_import_closure": dict(_BOOTSTRAP_FIXED_SOURCE_SHA256_V2),
            "wire": {
                "config_sha256": WIRE_CONFIG_SHA256,
                "documentation_sha256": WIRE_DOCUMENTATION_SHA256,
                "source_sha256": WIRE_SOURCE_SHA256,
                "test_sha256": WIRE_TEST_SHA256,
            },
        },
        "python_abi": "cp310",
        "python_implementation": "CPython",
        "python_version": "3.10.12",
        "standard_library_only": True,
        "standard_library_only_scope": "PYTHON_THIRD_PARTY_PACKAGES_ONLY",
        "step5_full_runtime_image_and_import_native_closure_required": True,
        "third_party_dependencies": [],
    }
    if (
        decoded["schema_version"]
        != "hegel-machine-phase2b-formal-wire-covert-qualification-dependency-lock/3"
        or type(lock) is not dict
        or not _exact_json_equal(lock, expected_lock)
    ):
        raise ValueError("dependency lock runtime contract drifted")
    return _live_dependency_lock_runtime_projection_v2()


def _runtime_environment_manifest_body_v2(
    *,
    python_row: dict[str, object],
    systemd_run_row: dict[str, object],
    systemctl_row: dict[str, object],
) -> dict[str, object]:
    affinity = sorted(os.sched_getaffinity(0))
    if not affinity or any(type(cpu) is not int or cpu < 0 for cpu in affinity):
        raise ValueError("host CPU affinity is empty or malformed")
    libc_name, libc_version = platform.libc_ver()
    return {
        "schema_version": RUNTIME_ENVIRONMENT_MANIFEST_SCHEMA_V2,
        "dependency_lock_sha256": DEPENDENCY_LOCK_SHA256,
        "dependency_lock_runtime": _live_dependency_lock_runtime_projection_v2(),
        "executables": {
            "python": python_row,
            "systemd_run": systemd_run_row,
            "systemctl": systemctl_row,
        },
        "host": {
            "architecture": platform.machine(),
            "kernel_release": platform.release(),
            "kernel_system": platform.system(),
            "kernel_version": platform.version(),
            "libc_name": libc_name,
            "libc_version": libc_version,
        },
        "cgroup2": _cgroup2_host_projection_v2(),
        "cpu_affinity": affinity,
        "selected_locale_overrides": {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        "interpreter_contract": {
            "argv_flags": ["-I", "-S", "-B"],
            "ignore_python_environment": True,
            "no_site": True,
            "source_only_loader": True,
            "forbid_pyc_pyo": True,
        },
        "environment_scope": (
            "ONLY_LANG_AND_LC_ALL_OVERRIDES_ARE_QUALIFIED;"
            "OTHER_USER_MANAGER_ENVIRONMENT_IS_NOT_QUALIFIED"
        ),
        "host_runtime_fingerprint_only_not_full_runtime_image_closure": True,
        "systemd_user_manager_same_uid_is_cooperative_outer_tcb": True,
        "fresh_dedicated_user_service_manager_session_required": True,
        "runtime_image_digest": None,
    }


def phase2b_formal_wire_covert_runtime_environment_manifest_bytes_v2() -> bytes:
    """Snapshot the exact host/runtime projection later replayed by the runner."""

    deadline_ns = time.monotonic_ns() + 30_000_000_000
    python_row = _executable_projection_for_runner_v2(
        Path(sys.executable),
        version_argv=("--version",),
        name="Python",
        deadline_ns=deadline_ns,
    )
    systemd_run = shutil.which("systemd-run")
    systemctl = shutil.which("systemctl")
    if systemd_run is None or systemctl is None:
        raise ValueError("runtime manifest cannot locate systemd executables")
    systemd_run_row = _executable_projection_for_runner_v2(
        Path(systemd_run),
        version_argv=("--version",),
        name="systemd-run",
        deadline_ns=deadline_ns,
    )
    systemctl_row = _executable_projection_for_runner_v2(
        Path(systemctl),
        version_argv=("--version",),
        name="systemctl",
        deadline_ns=deadline_ns,
    )
    return _canonical_inner_artifact_v2(
        _runtime_environment_manifest_body_v2(
            python_row=python_row,
            systemd_run_row=systemd_run_row,
            systemctl_row=systemctl_row,
        )
    )


def _verify_runtime_environment_manifest_v2(
    payload: bytes,
    *,
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
    python_row: dict[str, object],
    systemd_run_row: dict[str, object],
    systemctl_row: dict[str, object],
) -> dict[str, object]:
    if (
        len(payload) != attempt.runtime_environment_manifest_size_bytes
        or hashlib.sha256(payload).hexdigest()
        != attempt.runtime_environment_manifest_root.removeprefix("sha256:")
    ):
        raise ValueError("runtime environment manifest bytes differ from the attempt")
    decoded = _decode_canonical_transport_object_v2(
        payload, "runtime environment manifest", maximum_bytes=1_048_576
    )
    expected = _runtime_environment_manifest_body_v2(
        python_row=python_row,
        systemd_run_row=systemd_run_row,
        systemctl_row=systemctl_row,
    )
    if not _exact_json_equal(decoded, expected):
        raise ValueError("runtime environment manifest differs from fresh host evidence")
    return decoded


def _append_bounded_pipe_block_v2(
    buffer: bytearray,
    block: bytes,
    *,
    cap: int,
    stream: str,
    capture_state: dict[str, bool] | None,
) -> None:
    if (
        type(buffer) is not bytearray
        or type(block) is not bytes
        or type(cap) is not int
        or type(cap) is bool
        or cap < 0
        or stream not in {"stdout", "stderr"}
        or (
            capture_state is not None
            and (
                type(capture_state) is not dict
                or set(capture_state)
                != {"capture_complete", "capture_truncated"}
                or any(type(value) is not bool for value in capture_state.values())
            )
        )
    ):
        raise ValueError("bounded child pipe append arguments are malformed")
    remaining = max(0, cap - len(buffer))
    buffer.extend(block[:remaining])
    if len(block) > remaining:
        if capture_state is not None:
            capture_state["capture_complete"] = False
            capture_state["capture_truncated"] = True
        raise ValueError(f"bounded child {stream} exceeded its cap")


def _capture_process_pipes_v2(
    process: subprocess.Popen[bytes],
    *,
    deadline_ns: int,
    stdout_cap: int,
    stderr_cap: int,
    wait_for_process_exit: bool,
    stdout_buffer: bytearray | None = None,
    stderr_buffer: bytearray | None = None,
    capture_state: dict[str, bool] | None = None,
) -> tuple[bytes, bytes, int | None]:
    if process.stdout is None or process.stderr is None:
        raise ValueError("bounded capture needs both child pipes")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    if (
        (stdout_buffer is not None and type(stdout_buffer) is not bytearray)
        or (stderr_buffer is not None and type(stderr_buffer) is not bytearray)
        or (
            capture_state is not None
            and (
                type(capture_state) is not dict
                or set(capture_state)
                != {"capture_complete", "capture_truncated"}
                or any(type(value) is not bool for value in capture_state.values())
            )
        )
    ):
        raise TypeError("bounded capture buffers/state are malformed")
    output = {
        "stdout": bytearray() if stdout_buffer is None else stdout_buffer,
        "stderr": bytearray() if stderr_buffer is None else stderr_buffer,
    }
    caps = {"stdout": stdout_cap, "stderr": stderr_cap}
    starts = {name: len(buffer) for name, buffer in output.items()}
    if any(
        type(caps[name]) is not int
        or type(caps[name]) is bool
        or caps[name] < starts[name]
        for name in output
    ):
        raise ValueError("bounded capture total cap is smaller than its prefix")
    try:
        while selector.get_map():
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                raise TimeoutError("bounded child capture reached its absolute deadline")
            events = selector.select(min(1.0, remaining_ns / 1_000_000_000))
            for key, _ in events:
                block = os.read(key.fileobj.fileno(), 65_536)
                stream = key.data
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                _append_bounded_pipe_block_v2(
                    output[stream],
                    block,
                    cap=caps[stream],
                    stream=stream,
                    capture_state=capture_state,
                )
        returncode: int | None = None
        if wait_for_process_exit:
            remaining_seconds = max(
                0.001, (deadline_ns - time.monotonic_ns()) / 1_000_000_000
            )
            returncode = process.wait(timeout=remaining_seconds)
            if capture_state is not None:
                if capture_state["capture_truncated"]:
                    raise ValueError(
                        "bounded child capture cannot complete after truncation"
                    )
                capture_state["capture_complete"] = True
        return (
            bytes(output["stdout"][starts["stdout"] :]),
            bytes(output["stderr"][starts["stderr"] :]),
            returncode,
        )
    finally:
        selector.close()


def _terminate_process_group_v2(
    process: subprocess.Popen[bytes], *, deadline_ns: int
) -> None:
    if type(deadline_ns) is not int or time.monotonic_ns() >= deadline_ns:
        raise TimeoutError("process-group cleanup lacks a future deadline")
    pgid = process.pid

    def group_exists() -> bool:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    # Recheck ownership at the innermost signal boundary.  The outer caller's
    # poll may have observed a live child that exited and was reaped before
    # this helper ran; its numeric PID/PGID is then no longer ours to signal.
    if process.poll() is not None:
        return
    if group_exists():
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    term_deadline = min(
        deadline_ns,
        time.monotonic_ns() + _BOUNDED_COMMAND_TERM_GRACE_NANOSECONDS_V2,
    )
    while time.monotonic_ns() < term_deadline:
        if process.poll() is not None:
            return
        if not group_exists():
            break
        time.sleep(0.01)
    if process.poll() is not None:
        return
    if group_exists():
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    while time.monotonic_ns() < deadline_ns:
        if process.poll() is not None:
            return
        if not group_exists():
            break
        time.sleep(0.01)
    if process.poll() is not None:
        return
    if group_exists():
        raise RuntimeError("outer process group survived SIGKILL")
    try:
        process.wait(
            timeout=max(0.001, (deadline_ns - time.monotonic_ns()) / 1_000_000_000)
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("outer launcher could not be reaped after SIGKILL") from exc


def _close_process_pipes_v2(
    process: subprocess.Popen[bytes], *, name: str
) -> None:
    primary = sys.exc_info()[1]
    errors: list[BaseException] = []
    seen: set[int] = set()
    for stream in (process.stdout, process.stderr):
        if stream is None:
            continue
        try:
            descriptor = stream.fileno()
        except (OSError, ValueError):
            continue
        if descriptor in seen:
            continue
        seen.add(descriptor)
        try:
            stream.close()
        except BaseException as exc:
            errors.append(exc)
    if not errors:
        return
    message = f"{name} could not close {len(errors)} child pipes"
    if primary is not None:
        _append_exception_note_v2(primary, message)
        return
    raise RuntimeError(message) from errors[0]


def _finalize_outer_launcher_v2(
    launcher: subprocess.Popen[bytes], *, deadline_ns: int
) -> None:
    """Reap a live owned launcher once; never signal a reaped numeric PGID."""

    try:
        # ``poll`` performs waitpid(WNOHANG).  A non-None result means the
        # numeric PID/PGID is no longer an ownership handle and must never be
        # signalled again because it may already have been reused.
        if launcher.poll() is None:
            _terminate_process_group_v2(launcher, deadline_ns=deadline_ns)
    finally:
        _close_process_pipes_v2(
            launcher, name="outer-owned launcher finalization"
        )


def _command_evidence_v2(
    *,
    argv: list[str],
    purpose: str,
    returncode: int,
    stdout: bytes,
    stderr: bytes,
) -> dict[str, object]:
    return {
        "argv": argv,
        "purpose": purpose,
        "returncode": returncode,
        "stderr_hex": stderr.hex(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stderr_size_bytes": len(stderr),
        "stdout_hex": stdout.hex(),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stdout_size_bytes": len(stdout),
    }


def _verify_command_evidence_v2(
    row: object, *, name: str
) -> tuple[list[str], str, int, bytes, bytes]:
    expected = {
        "argv",
        "purpose",
        "returncode",
        "stderr_hex",
        "stderr_sha256",
        "stderr_size_bytes",
        "stdout_hex",
        "stdout_sha256",
        "stdout_size_bytes",
    }
    if type(row) is not dict or set(row) != expected:
        raise ValueError(f"{name} command evidence is not closed")
    argv = row["argv"]
    purpose = row["purpose"]
    returncode = row["returncode"]
    if (
        type(argv) is not list
        or not 1 <= len(argv) <= 64
        or any(type(item) is not str or not item or "\x00" in item for item in argv)
        or type(purpose) is not str
        or re.fullmatch(r"[a-z0-9-]{1,96}", purpose) is None
        or type(returncode) is not int
        or not -255 <= returncode <= 255
    ):
        raise ValueError(f"{name} command identity is malformed")
    streams: list[bytes] = []
    for stream in ("stdout", "stderr"):
        size = _require_exact_int_v2(
            row[f"{stream}_size_bytes"], f"{name} {stream} size", maximum=256 * 1_024
        )
        hex_value = row[f"{stream}_hex"]
        if (
            type(hex_value) is not str
            or len(hex_value) != size * 2
            or re.fullmatch(r"[0-9a-f]*", hex_value) is None
        ):
            raise ValueError(f"{name} {stream} hex framing drifted")
        payload = bytes.fromhex(hex_value)
        if (
            hashlib.sha256(payload).hexdigest()
            != _require_raw_sha256_v2(row[f"{stream}_sha256"], f"{name} {stream}")
        ):
            raise ValueError(f"{name} {stream} hash drifted")
        streams.append(payload)
    return argv, purpose, returncode, streams[0], streams[1]


def _verify_systemctl_transcript_v2(
    commands: object,
    *,
    systemctl_realpath: str,
    unit_name: str,
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
    source_tree: Path,
    ownership_marker: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    if (
        type(commands) is not list
        or not 6 <= len(commands) <= _MAX_MAIN_SYSTEMCTL_COMMAND_ROWS_V2
    ):
        raise ValueError("systemctl transcript count is outside its frozen bounds")
    decoded: list[tuple[list[str], str, int, bytes, bytes]] = []
    aggregate = 0
    for index, row in enumerate(commands):
        item = _verify_command_evidence_v2(row, name=f"systemctl[{index}]")
        aggregate += len(item[3]) + len(item[4])
        if aggregate > 512 * 1_024:
            raise ValueError("systemctl transcript aggregate bytes exceed the cap")
        decoded.append(item)
    show_argv = _systemd_show_argv_v2(systemctl_realpath, unit_name)
    pre_argv, pre_purpose, pre_rc, pre_stdout, pre_stderr = decoded[0]
    if (
        pre_argv != show_argv
        or pre_purpose != "prelaunch-absent-check"
        or pre_rc != 0
        or pre_stderr
    ):
        raise ValueError("systemctl prelaunch absence evidence drifted")
    prelaunch = _parse_systemd_show_v2(
        pre_stdout, set(_SYSTEMD_SHOW_PROPERTIES_V2)
    )
    if (
        prelaunch["LoadState"] != "not-found"
        or prelaunch["ActiveState"] != "inactive"
        or prelaunch["SubState"] != "dead"
        or prelaunch["MainPID"] != "0"
        or prelaunch["ControlGroup"] != ""
        or prelaunch["InvocationID"] != ""
    ):
        raise ValueError("systemctl prelaunch unit was not exact absent")
    position = 1
    entry_rows: list[dict[str, str]] = []
    locked_invocation_id: str | None = None
    saw_loaded_entry = False
    ready_entry: dict[str, str] | None = None
    while position < len(decoded) and decoded[position][1].startswith("entry-show-"):
        argv, purpose, returncode, stdout, stderr = decoded[position]
        if purpose != f"entry-show-{len(entry_rows)}" or argv != show_argv:
            raise ValueError("systemctl entry probe sequence drifted")
        if len(entry_rows) >= 8:
            raise ValueError("systemctl entry probe count exceeds eight")
        if returncode != 0:
            entry_rows.append({})
            position += 1
            continue
        if stderr:
            raise ValueError("successful systemctl entry probe wrote stderr")
        candidate = _parse_systemd_show_v2(
            stdout, set(_SYSTEMD_SHOW_PROPERTIES_V2)
        )
        entry_kind, invocation_id = _evaluate_owned_or_absent_systemd_show_v2(
            candidate,
            attempt=attempt,
            source_tree=source_tree,
            ownership_marker=ownership_marker,
            expected_invocation_id=locked_invocation_id,
            name="systemctl entry",
        )
        if entry_kind == "absent":
            if saw_loaded_entry:
                raise ValueError(
                    "systemctl entry became absent after an owned invocation appeared"
                )
            entry_rows.append({})
        else:
            if invocation_id is None:
                raise RuntimeError("owned systemctl entry lost its InvocationID")
            if locked_invocation_id is None:
                locked_invocation_id = invocation_id
            saw_loaded_entry = True
            entry_rows.append(candidate)
        position += 1
        if entry_kind == "owned" and _systemd_entry_ready_v2(
            candidate, name="systemctl entry"
        ):
            ready_entry = candidate
            break
    if not 1 <= len(entry_rows) <= 8:
        raise ValueError("systemctl entry probe count drifted")
    if ready_entry is None:
        raise ValueError("systemctl entry terminal probe is not active/running")
    if (
        position < len(decoded)
        and decoded[position][1].startswith("entry-show-")
    ):
        raise ValueError("systemctl entry probes continued after ready")
    entry = ready_entry
    terminal_rows: list[dict[str, str]] = []
    while position < len(decoded) and decoded[position][1].startswith("terminal-show-"):
        argv, purpose, returncode, stdout, stderr = decoded[position]
        if purpose != f"terminal-show-{len(terminal_rows)}" or argv != show_argv:
            raise ValueError("systemctl terminal probe sequence drifted")
        if returncode != 0 or stderr:
            raise ValueError("systemctl terminal probe did not close successfully")
        candidate = _parse_systemd_show_v2(
            stdout, set(_SYSTEMD_SHOW_PROPERTIES_V2)
        )
        terminal_kind, _ = _evaluate_owned_or_absent_systemd_show_v2(
            candidate,
            attempt=attempt,
            source_tree=source_tree,
            ownership_marker=ownership_marker,
            expected_invocation_id=entry["InvocationID"],
            name="systemctl terminal probe",
        )
        if terminal_kind != "owned":
            raise ValueError("systemctl terminal probe lost its owned invocation")
        _systemd_terminal_success_v2(
            candidate, name="systemctl terminal probe"
        )
        terminal_rows.append(candidate)
        position += 1
    if not 2 <= len(terminal_rows) <= 33:
        raise ValueError("systemctl terminal probe count drifted")
    for earlier in terminal_rows[:-2]:
        if _systemd_terminal_success_v2(
            earlier, name="systemctl terminal loop"
        ):
            raise ValueError(
                "systemctl terminal probes continued after the loop exit snapshot"
            )
    terminal = terminal_rows[-1]
    for name, candidate in (
        ("loop-exit", terminal_rows[-2]),
        ("pre-stop", terminal),
    ):
        if not _systemd_terminal_success_v2(
            candidate, name=f"systemctl {name} terminal"
        ):
            raise ValueError(f"systemctl {name} terminal snapshot drifted")
    if position >= len(decoded):
        raise ValueError("systemctl transcript lacks stop")
    stop_argv, stop_purpose, stop_rc, stop_stdout, stop_stderr = decoded[position]
    if (
        stop_argv != [systemctl_realpath, "--user", "stop", unit_name]
        or stop_purpose != "stop-after-terminal-snapshot"
        or stop_rc != 0
        or stop_stdout
        or stop_stderr
    ):
        raise ValueError("systemctl stop evidence drifted")
    position += 1
    absent_rows: list[dict[str, str]] = []
    while position < len(decoded):
        absent_argv, absent_purpose, absent_rc, absent_stdout, absent_stderr = (
            decoded[position]
        )
        if (
            absent_argv != show_argv
            or absent_purpose != f"post-stop-absent-check-{len(absent_rows)}"
            or absent_rc != 0
            or absent_stderr
        ):
            raise ValueError("systemctl absent command evidence drifted")
        candidate = _parse_systemd_show_v2(
            absent_stdout, set(_SYSTEMD_SHOW_PROPERTIES_V2)
        )
        if not _systemd_show_is_exact_absent_v2(candidate):
            post_kind, _ = _evaluate_owned_or_absent_systemd_show_v2(
                candidate,
                attempt=attempt,
                source_tree=source_tree,
                ownership_marker=ownership_marker,
                expected_invocation_id=entry["InvocationID"],
                name="systemctl post-stop probe",
            )
            if post_kind != "owned":
                raise ValueError(
                    "systemctl post-stop probe observed a replacement unit"
                )
        elif position != len(decoded) - 1:
            raise ValueError("systemctl transcript continued after exact absence")
        absent_rows.append(candidate)
        position += 1
    if not 1 <= len(absent_rows) <= 8:
        raise ValueError("systemctl absent probe count drifted")
    absent = absent_rows[-1]
    if (
        absent["LoadState"] != "not-found"
        or absent["ActiveState"] != "inactive"
        or absent["SubState"] != "dead"
    ):
        raise ValueError("systemctl post-stop snapshot is not exact not-found")
    _normalize_effective_systemd_properties_v2(
        entry, attempt=attempt, source_tree=source_tree
    )
    return entry, terminal, absent


def _systemd_show_is_exact_absent_v2(show: dict[str, str]) -> bool:
    return (
        show.get("LoadState") == "not-found"
        and show.get("ActiveState") == "inactive"
        and show.get("SubState") == "dead"
        and show.get("MainPID") == "0"
        and show.get("ControlGroup") == ""
        and show.get("InvocationID") == ""
    )


def _validate_calibration_pair_projection_shape_v2(
    value: object, *, required: bool
) -> dict[str, object] | None:
    if type(required) is not bool:
        raise TypeError("calibration pair requirement must be exact bool")
    if not required:
        if value is not None:
            raise ValueError("development calibration unexpectedly binds a prior pair")
        return None
    keys = {
        "attempt_directory",
        "attempt_manifest_root",
        "pair_must_be_jointly_signed_by_step5",
        "pair_retention_required",
        "retained_tree_before_terminal",
        "schema_version",
        "terminal_sha256",
        "terminal_size_bytes",
    }
    if (
        type(value) is not dict
        or set(value) != keys
        or value["schema_version"]
        != "hegel-machine-phase2b-formal-wire-covert-calibration-attempt-pair/1"
        or type(value["attempt_directory"]) is not str
        or not Path(value["attempt_directory"]).is_absolute()
        or type(value["attempt_manifest_root"]) is not str
        or re.fullmatch(
            r"phase2b_formal_wire_covert_attempt_manifest_v3_[0-9a-f]{64}",
            value["attempt_manifest_root"],
        )
        is None
        or _require_raw_sha256_v2(
            value["terminal_sha256"], "calibration pair terminal"
        )
        != value["terminal_sha256"]
        or _require_exact_int_v2(
            value["terminal_size_bytes"],
            "calibration pair terminal size",
            minimum=1,
            maximum=8 * 1_024 * 1_024,
        )
        < 1
        or value["pair_retention_required"] is not True
        or value["pair_must_be_jointly_signed_by_step5"] is not True
        or type(value["retained_tree_before_terminal"]) is not dict
    ):
        raise ValueError("calibration pair projection is not closed")
    attempt_directory = Path(value["attempt_directory"])
    expected_attempt_name = "attempt-" + hashlib.sha256(
        value["attempt_manifest_root"].encode("ascii")
    ).hexdigest()[:48]
    tree = value["retained_tree_before_terminal"]
    if (
        attempt_directory.name != expected_attempt_name
        or set(tree)
        != {
            "regular_file_count",
            "root",
            "row_count",
            "top_level_names",
            "total_regular_bytes",
        }
        or _require_exact_int_v2(
            tree["row_count"], "calibration pair retained row count",
            maximum=100_000,
        )
        < 0
        or _require_exact_int_v2(
            tree["regular_file_count"],
            "calibration pair retained regular count",
            maximum=100_000,
        )
        < 0
        or _require_exact_int_v2(
            tree["total_regular_bytes"],
            "calibration pair retained byte count",
            maximum=_MAXIMUM_RETAINED_ATTEMPT_TREE_BYTES_V2,
        )
        < 0
        or type(tree["top_level_names"]) is not list
        or len(tree["top_level_names"]) > 32
        or any(
            type(name) is not str
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) is None
            for name in tree["top_level_names"]
        )
        or tree["top_level_names"] != sorted(set(tree["top_level_names"]))
        or type(tree["root"]) is not str
        or re.fullmatch(
            r"phase2b_formal_wire_covert_failure_tree_v3_[0-9a-f]{64}",
            tree["root"],
        )
        is None
    ):
        raise ValueError("calibration pair retained-tree projection is not closed")
    return value


def _validate_launch_intent_projection_v2(
    intent: object,
    *,
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
    entry_kind: str,
    source_tree: Path,
) -> dict[str, object]:
    """Validate the immutable pre-Popen ownership/control preimage."""

    keys = {
        "attempt_manifest_root",
        "child_argv",
        "child_argv_sha256",
        "entry_kind",
        "executables",
        "ownership_marker",
        "ownership_nonce",
        "prelaunch_command",
        "prior_calibration_attempt_pair",
        "profile_id",
        "runtime_environment_manifest_root",
        "runtime_environment_manifest_size_bytes",
        "schema_version",
        "source_tree_root",
        "systemd_argv",
        "unit_name",
    }
    if (
        type(intent) is not dict
        or set(intent) != keys
        or intent["schema_version"] != OUTER_LAUNCH_INTENT_SCHEMA_V2
        or intent["attempt_manifest_root"] != attempt.attempt_manifest_root
        or intent["profile_id"] != attempt.profile_id
        or intent["entry_kind"] != entry_kind
        or entry_kind not in {"development-calibration", "live-inner"}
        or intent["source_tree_root"] != str(source_tree)
        or intent["runtime_environment_manifest_root"]
        != attempt.runtime_environment_manifest_root
        or _require_exact_int_v2(
            intent["runtime_environment_manifest_size_bytes"],
            "launch-intent runtime manifest size",
            minimum=1,
            maximum=1_048_576,
        )
        != attempt.runtime_environment_manifest_size_bytes
    ):
        raise ValueError("launch-intent identity fields drifted")
    _validate_calibration_pair_projection_shape_v2(
        intent["prior_calibration_attempt_pair"],
        required=entry_kind == "live-inner",
    )
    executables = intent["executables"]
    if type(executables) is not dict or set(executables) != {
        "python",
        "systemd_run",
        "systemctl",
    }:
        raise ValueError("launch-intent executable set is not closed")
    python_row = _verify_executable_projection_shape_v2(
        executables["python"], "launch-intent Python"
    )
    systemd_run_row = _verify_executable_projection_shape_v2(
        executables["systemd_run"], "launch-intent systemd-run"
    )
    systemctl_row = _verify_executable_projection_shape_v2(
        executables["systemctl"], "launch-intent systemctl"
    )
    nonce = intent["ownership_nonce"]
    if type(nonce) is not str or re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
        raise ValueError("launch-intent ownership nonce is malformed")
    expected_unit = "hegel-qv2-" + hashlib.sha256(
        (
            attempt.attempt_manifest_root
            + "\x00"
            + entry_kind
            + "\x00"
            + nonce
        ).encode("ascii")
    ).hexdigest()[:32] + ".service"
    expected_child_argv = _source_execution_bootstrap_argv_v2(
        python_realpath=python_row["realpath"],
        project_root=source_tree,
        qualification_source_sha256=attempt.qualification_source_sha256,
        source_arguments=[
            "--entry",
            entry_kind,
            "--profile",
            str(source_tree.parent / "inputs/profile.json"),
            "--attempt",
            str(source_tree.parent / "inputs/attempt.json"),
        ],
    )
    child_sha = hashlib.sha256(
        canonicalize_rfc8785_v1(expected_child_argv)
    ).hexdigest()
    expected_marker = "hegel-qv2-owned-" + hashlib.sha256(
        b"HEGEL/PHASE2B/QV2/UNIT_OWNERSHIP/V2\x00"
        + attempt.attempt_manifest_root.encode("ascii")
        + bytes.fromhex(nonce)
        + bytes.fromhex(child_sha)
    ).hexdigest()
    expected_systemd_argv = [
        systemd_run_row["realpath"],
        "--user",
        "--pipe",
        "--wait",
        "--quiet",
        f"--unit={expected_unit.removesuffix('.service')}",
        f"--property=Description={expected_marker}",
        "--property=Type=exec",
        "--property=RemainAfterExit=yes",
        f"--property=MemoryMax={attempt.requested_cgroup_memory_max_bytes}",
        f"--property=TasksMax={attempt.requested_cgroup_pids_max}",
        f"--property=RuntimeMaxSec={LIVE_HARD_TOTAL_TIMEOUT_SECONDS}s",
        "--property=KillMode=control-group",
        "--property=KillSignal=SIGTERM",
        "--property=SendSIGKILL=yes",
        "--property=FinalKillSignal=SIGKILL",
        "--property=TimeoutStopSec=30s",
        f"--working-directory={source_tree}",
        "--setenv=LANG=C.UTF-8",
        "--setenv=LC_ALL=C.UTF-8",
        "--",
        *expected_child_argv,
    ]
    if (
        intent["unit_name"] != expected_unit
        or intent["child_argv"] != expected_child_argv
        or intent["child_argv_sha256"] != child_sha
        or intent["ownership_marker"] != expected_marker
        or intent["systemd_argv"] != expected_systemd_argv
    ):
        raise ValueError("launch-intent unit/argv ownership preimage drifted")
    pre_argv, pre_purpose, pre_rc, pre_stdout, pre_stderr = (
        _verify_command_evidence_v2(
            intent["prelaunch_command"], name="launch-intent prelaunch"
        )
    )
    if (
        pre_argv
        != _systemd_show_argv_v2(systemctl_row["realpath"], expected_unit)
        or pre_purpose != "prelaunch-absent-check"
        or pre_rc != 0
        or pre_stderr
        or not _systemd_show_is_exact_absent_v2(
            _parse_systemd_show_v2(
                pre_stdout, set(_SYSTEMD_SHOW_PROPERTIES_V2)
            )
        )
    ):
        raise ValueError("launch-intent prelaunch absence preimage drifted")
    return intent


def _cleanup_transient_attempt_v2(
    *,
    launcher: subprocess.Popen[bytes],
    systemctl_realpath: str,
    unit_name: str,
    ownership_marker: str,
    owned_invocation_id: str | None,
    prelaunch_absent_verified: bool,
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
    source_tree: Path,
    deadline_ns: int,
    commands: list[dict[str, object]],
    cleanup_start_command_index: int,
    stdout_capture: bytearray | None = None,
    stderr_capture: bytearray | None = None,
    capture_state: dict[str, bool] | None = None,
) -> dict[str, object]:
    if prelaunch_absent_verified is not True:
        raise RuntimeError("failure cleanup lacks prelaunch absence ownership evidence")
    if (
        type(cleanup_start_command_index) is not int
        or type(cleanup_start_command_index) is bool
        or cleanup_start_command_index != len(commands)
        or not 1
        <= cleanup_start_command_index
        <= _MAX_MAIN_SYSTEMCTL_COMMAND_ROWS_V2
    ):
        raise RuntimeError("failure cleanup boundary is not exact")
    if (
        (stdout_capture is not None and type(stdout_capture) is not bytearray)
        or (stderr_capture is not None and type(stderr_capture) is not bytearray)
        or (
            capture_state is not None
            and (
                type(capture_state) is not dict
                or set(capture_state)
                != {"capture_complete", "capture_truncated"}
                or any(type(value) is not bool for value in capture_state.values())
            )
        )
    ):
        raise RuntimeError("failure cleanup pipe-capture state is malformed")
    show_argv = _systemd_show_argv_v2(systemctl_realpath, unit_name)

    def show_once(purpose: str) -> dict[str, str]:
        evidence, stdout, stderr = _run_bounded_control_command_v2(
            show_argv,
            purpose=purpose,
            deadline_ns=min(deadline_ns, time.monotonic_ns() + 5_000_000_000),
        )
        commands.append(evidence)
        if evidence["returncode"] != 0 or stderr:
            raise RuntimeError("failure cleanup systemctl show did not close")
        return _parse_systemd_show_v2(
            stdout, set(_SYSTEMD_SHOW_PROPERTIES_V2)
        )

    current = show_once("failure-owned-show")
    recovered_invocation = owned_invocation_id

    def require_owned_or_absent(show: dict[str, str], name: str) -> None:
        nonlocal recovered_invocation
        state, invocation = _evaluate_owned_or_absent_systemd_show_v2(
            show,
            attempt=attempt,
            source_tree=source_tree,
            ownership_marker=ownership_marker,
            expected_invocation_id=recovered_invocation,
            name=f"failure cleanup {name}",
        )
        if state == "owned":
            if invocation is None:
                raise RuntimeError("owned cleanup show lost its InvocationID")
            if recovered_invocation is None:
                recovered_invocation = invocation

    require_owned_or_absent(current, "initial")
    term_sent = False
    kill_sent = False

    def unit_has_live_members(show: dict[str, str]) -> bool:
        return _systemd_show_has_live_members_v2(
            show, name="failure cleanup"
        )

    if unit_has_live_members(current):
        term_argv = [
            systemctl_realpath,
            "--user",
            "kill",
            "--kill-who=all",
            "--signal=SIGTERM",
            unit_name,
        ]
        term_evidence, term_stdout, term_stderr = _run_bounded_control_command_v2(
            term_argv,
            purpose="failure-owned-sigterm",
            deadline_ns=min(deadline_ns, time.monotonic_ns() + 5_000_000_000),
        )
        commands.append(term_evidence)
        if term_evidence["returncode"] != 0 or term_stdout or term_stderr:
            raise RuntimeError("failure cleanup SIGTERM command failed")
        term_sent = True
        grace_deadline = min(deadline_ns - 15_000_000_000, time.monotonic_ns() + 5_000_000_000)
        for index in range(20):
            current = show_once(f"failure-term-grace-{index}")
            require_owned_or_absent(current, "TERM grace")
            if not unit_has_live_members(current):
                break
            if time.monotonic_ns() >= grace_deadline:
                break
            time.sleep(0.25)
        if unit_has_live_members(current):
            require_owned_or_absent(current, "pre-SIGKILL")
            kill_argv = [
                systemctl_realpath,
                "--user",
                "kill",
                "--kill-who=all",
                "--signal=SIGKILL",
                unit_name,
            ]
            kill_evidence, kill_stdout, kill_stderr = (
                _run_bounded_control_command_v2(
                    kill_argv,
                    purpose="failure-owned-sigkill",
                    deadline_ns=min(
                        deadline_ns, time.monotonic_ns() + 5_000_000_000
                    ),
                )
            )
            commands.append(kill_evidence)
            if kill_evidence["returncode"] != 0 or kill_stdout or kill_stderr:
                raise RuntimeError("failure cleanup SIGKILL command failed")
            kill_sent = True
            for index in range(8):
                current = show_once(f"failure-kill-grace-{index}")
                require_owned_or_absent(current, "KILL grace")
                if not unit_has_live_members(current):
                    break
                time.sleep(0.1)
            if unit_has_live_members(current):
                raise RuntimeError("failure cleanup SIGKILL left live unit members")
    if not _systemd_show_is_exact_absent_v2(current):
        require_owned_or_absent(current, "pre-stop")
        if (
            current["Description"] != ownership_marker
            or current["InvocationID"] != recovered_invocation
        ):
            raise RuntimeError("failure cleanup lost owned unit identity before stop")
        stop_argv = [systemctl_realpath, "--user", "stop", unit_name]
        stop_evidence, stop_stdout, stop_stderr = _run_bounded_control_command_v2(
            stop_argv,
            purpose="failure-owned-stop",
            deadline_ns=min(deadline_ns, time.monotonic_ns() + 5_000_000_000),
        )
        commands.append(stop_evidence)
        if stop_evidence["returncode"] != 0 or stop_stdout or stop_stderr:
            raise RuntimeError("failure cleanup stop command failed")
    if launcher.stdout is None or launcher.stderr is None:
        raise RuntimeError("failure cleanup launcher pipes are absent")
    launcher_capture_deadline_ns = (
        deadline_ns - _BOUNDED_COMMAND_REAP_RESERVE_NANOSECONDS_V2
    )
    try:
        try:
            if (
                launcher.poll() is not None
                and launcher.stdout.closed
                and launcher.stderr.closed
            ):
                if (
                    capture_state is None
                    or capture_state["capture_complete"] is not True
                    or capture_state["capture_truncated"] is not False
                ):
                    raise RuntimeError(
                        "closed failure pipes lack prior EOF/reap capture proof"
                    )
                drained_stdout = b""
                drained_stderr = b""
                launcher_returncode = launcher.returncode
            else:
                drained_stdout, drained_stderr, launcher_returncode = (
                    _capture_process_pipes_v2(
                        launcher,
                        deadline_ns=launcher_capture_deadline_ns,
                        stdout_cap=8 * 1_024 * 1_024,
                        stderr_cap=256 * 1_024,
                        wait_for_process_exit=True,
                        stdout_buffer=stdout_capture,
                        stderr_buffer=stderr_capture,
                        capture_state=capture_state,
                    )
                )
        except BaseException:
            _terminate_process_group_v2(launcher, deadline_ns=deadline_ns)
            raise
    finally:
        _close_process_pipes_v2(
            launcher, name="failure cleanup launcher"
        )
    absent: dict[str, str] | None = None
    absence_observed = _systemd_show_is_exact_absent_v2(current)
    for index in range(8):
        candidate = show_once(f"failure-absent-{index}")
        if absence_observed and not _systemd_show_is_exact_absent_v2(candidate):
            raise RuntimeError(
                "failure cleanup unit reappeared after exact absence"
            )
        require_owned_or_absent(candidate, "final absence")
        if _systemd_show_is_exact_absent_v2(candidate):
            absent = candidate
            absence_observed = True
            break
        if (
            candidate["Description"] != ownership_marker
            or candidate["InvocationID"] != recovered_invocation
        ):
            raise RuntimeError("failure cleanup observed a replacement unit")
        time.sleep(0.1)
    if absent is None or launcher.poll() is None:
        raise RuntimeError("failure cleanup did not prove reaped launcher and absent unit")
    return {
        "cleanup_verified": True,
        "cleanup_start_command_index": cleanup_start_command_index,
        "commands": list(commands),
        "drained_stderr_sha256": hashlib.sha256(drained_stderr).hexdigest(),
        "drained_stderr_size_bytes": len(drained_stderr),
        "drained_stdout_sha256": hashlib.sha256(drained_stdout).hexdigest(),
        "drained_stdout_size_bytes": len(drained_stdout),
        "kill_sent": kill_sent,
        "launcher_reaped": True,
        "launcher_returncode": launcher_returncode,
        "owned_invocation_id": recovered_invocation,
        "ownership_marker": ownership_marker,
        "post_cleanup_unit_absent": True,
        "prelaunch_unit_absent": True,
        "term_sent": term_sent,
        "transcript_root": stable_hash(
            commands, prefix="phase2b_formal_wire_covert_failure_cleanup_v2_"
        ),
        "unit_name": unit_name,
    }


def _verify_failure_cleanup_projection_v2(
    cleanup: object,
    *,
    cleanup_verified: bool,
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
    entry_kind: str,
    source_tree: Path,
    launch_intent: dict[str, object],
    failure_schema_version: str = OUTER_ATTEMPT_FAILURE_SCHEMA_V2,
) -> None:
    """Replay a stage-bounded cleanup FSM from the immutable launch intent."""

    if (
        type(cleanup_verified) is not bool
        or type(cleanup) is not dict
        or failure_schema_version
        not in {OUTER_ATTEMPT_FAILURE_SCHEMA_V2, _OUTER_ATTEMPT_FAILURE_SCHEMA_V3}
    ):
        raise ValueError("failure cleanup projection arguments are malformed")
    intent = _validate_launch_intent_projection_v2(
        launch_intent,
        attempt=attempt,
        entry_kind=entry_kind,
        source_tree=source_tree,
    )
    common = {
        "cleanup_start_command_index",
        "cleanup_verified",
        "commands",
        "owned_invocation_id",
        "ownership_marker",
        "prelaunch_unit_absent",
        "transcript_root",
        "unit_name",
    }
    verified_extra = {
        "drained_stderr_sha256",
        "drained_stderr_size_bytes",
        "drained_stdout_sha256",
        "drained_stdout_size_bytes",
        "kill_sent",
        "launcher_reaped",
        "launcher_returncode",
        "post_cleanup_unit_absent",
        "term_sent",
    }
    unproven_extra = {"cleanup_error_type", "external_intervention_required"}
    expected_keys = common | (verified_extra if cleanup_verified else unproven_extra)
    commands = cleanup.get("commands")
    if (
        set(cleanup) != expected_keys
        or cleanup["cleanup_verified"] is not cleanup_verified
        or cleanup["prelaunch_unit_absent"] is not True
        or type(commands) is not list
        or not 1 <= len(commands) <= _MAX_FAILURE_TRANSCRIPT_COMMAND_ROWS_V2
        or cleanup["transcript_root"]
        != stable_hash(
            commands,
            prefix="phase2b_formal_wire_covert_failure_cleanup_v2_",
        )
        or cleanup["unit_name"] != intent["unit_name"]
        or cleanup["ownership_marker"] != intent["ownership_marker"]
        or (
            cleanup["owned_invocation_id"] is not None
            and (
                type(cleanup["owned_invocation_id"]) is not str
                or re.fullmatch(r"[0-9a-f]{32}", cleanup["owned_invocation_id"])
                is None
            )
        )
    ):
        raise ValueError("failure cleanup projection is not launch-intent bound")
    boundary = _require_exact_int_v2(
        cleanup["cleanup_start_command_index"],
        "failure cleanup command boundary",
        minimum=1,
        maximum=min(len(commands), _MAX_MAIN_SYSTEMCTL_COMMAND_ROWS_V2),
    )
    decoded = [
        _verify_command_evidence_v2(row, name=f"failure cleanup[{index}]")
        for index, row in enumerate(commands)
    ]
    if sum(len(row[3]) + len(row[4]) for row in decoded) > 2 * 1_024 * 1_024:
        raise ValueError("failure cleanup command transcript exceeds its cap")
    if commands[0] != intent["prelaunch_command"]:
        raise ValueError("failure cleanup prelaunch row differs from launch intent")
    systemctl_realpath = intent["executables"]["systemctl"]["realpath"]
    unit_name = intent["unit_name"]
    ownership_marker = intent["ownership_marker"]
    kill_who_option = (
        "--kill-who=all"
        if failure_schema_version == OUTER_ATTEMPT_FAILURE_SCHEMA_V2
        else "--kill-" + "whom=all"
    )
    show_argv = _systemd_show_argv_v2(systemctl_realpath, unit_name)
    if len(decoded) - boundary > _MAX_FAILURE_CLEANUP_SUFFIX_ROWS_V2:
        raise ValueError("failure cleanup suffix exceeds its exact row bound")
    term_argv = [
        systemctl_realpath,
        "--user",
        "kill",
        kill_who_option,
        "--signal=SIGTERM",
        unit_name,
    ]
    kill_argv = [
        systemctl_realpath,
        "--user",
        "kill",
        kill_who_option,
        "--signal=SIGKILL",
        unit_name,
    ]
    stop_argv = [systemctl_realpath, "--user", "stop", unit_name]
    main_show_pattern = re.compile(
        r"(?:entry-show|terminal-show|post-stop-absent-check)-[0-9]+"
    )
    cleanup_show_pattern = re.compile(
        r"(?:failure-term-grace|failure-kill-grace|failure-absent)-[0-9]+"
    )
    for index, (argv, purpose, _, _, _) in enumerate(decoded[1:], start=1):
        if index < boundary:
            allowed = (
                argv == show_argv and main_show_pattern.fullmatch(purpose) is not None
            ) or (argv == stop_argv and purpose == "stop-after-terminal-snapshot")
        else:
            allowed = (
                argv == show_argv
                and (
                    purpose == "failure-owned-show"
                    or cleanup_show_pattern.fullmatch(purpose) is not None
                )
            ) or (argv == term_argv and purpose == "failure-owned-sigterm") or (
                argv == kill_argv and purpose == "failure-owned-sigkill"
            ) or (argv == stop_argv and purpose == "failure-owned-stop")
        if not allowed:
            raise ValueError(
                f"failure cleanup command {index} crosses its phase boundary"
            )
    if not cleanup_verified:
        if (
            cleanup["external_intervention_required"] is not True
            or type(cleanup["cleanup_error_type"]) is not str
            or re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]{0,127}", cleanup["cleanup_error_type"]
            )
            is None
        ):
            raise ValueError("unproven cleanup intervention marker drifted")
        # Retained /failure/3 evidence predates the strict ownership/FSM
        # projection below and remains replay-only.  Current /failure/4
        # unproven transcripts are negative claims, but their recorded rows
        # must still satisfy the same launch-intent, ownership, phase, and
        # ordering constraints as verified cleanup up to the exact final
        # failing transition.
        if failure_schema_version == _OUTER_ATTEMPT_FAILURE_SCHEMA_V3:
            return
    owned_invocation: str | None = None

    def parse_owned_show(
        item: tuple[list[str], str, int, bytes, bytes],
        *,
        allow_absent: bool,
        allow_failed_last: bool = False,
        phase_name: str = "failure cleanup",
    ) -> dict[str, str] | None:
        nonlocal owned_invocation
        argv, _, returncode, stdout, stderr = item
        if argv != show_argv:
            raise ValueError("failure cleanup show argv drifted")
        if returncode != 0 or stderr:
            if allow_failed_last:
                return None
            raise ValueError("failure cleanup show command did not close")
        previous_owned_invocation = owned_invocation
        try:
            show = _parse_systemd_show_v2(
                stdout, set(_SYSTEMD_SHOW_PROPERTIES_V2)
            )
            state, invocation = _evaluate_owned_or_absent_systemd_show_v2(
                show,
                attempt=attempt,
                source_tree=source_tree,
                ownership_marker=ownership_marker,
                expected_invocation_id=owned_invocation,
                name=phase_name,
            )
            if state == "absent":
                if not allow_absent:
                    raise ValueError("failure cleanup unexpectedly observed absence")
                return show
            if invocation is None:
                raise ValueError("failure cleanup owned show lost its InvocationID")
            if owned_invocation is None:
                owned_invocation = invocation
            return show
        except ValueError:
            if allow_failed_last:
                owned_invocation = previous_owned_invocation
                return None
            raise

    # Main-run prefix: exact purpose/argv sequence, with only its final row
    # permitted to be the command that triggered the failure boundary.
    entry_index = 0
    terminal_index = 0
    post_stop_index = 0
    prefix_phase = "entry"
    last_terminal_show: dict[str, str] | None = None
    entry_ready = False
    terminal_exit_seen = False
    prestop_snapshot_seen = False
    for position in range(1, boundary):
        item = decoded[position]
        purpose = item[1]
        failed_boundary_row = position == boundary - 1
        if purpose == f"entry-show-{entry_index}" and prefix_phase == "entry":
            if entry_ready:
                raise ValueError("cleanup prefix continued entry probes after ready")
            entry_index += 1
            if entry_index > 8:
                raise ValueError("cleanup prefix entry probe count exceeds eight")
            # The producer intentionally retries a bounded entry `show` whose
            # command returns nonzero.  A successful command, however, must
            # parse and prove exact absence/ownership; only the final primary-
            # failure row may carry a successful-but-malformed projection.
            if item[2] != 0:
                continue
            if item[4]:
                if failed_boundary_row:
                    continue
                raise ValueError(
                    "successful cleanup-prefix entry show did not close: wrote stderr"
                )
            previous_owned_invocation = owned_invocation
            show = parse_owned_show(
                item,
                allow_absent=True,
                allow_failed_last=failed_boundary_row,
                phase_name="cleanup-prefix entry",
            )
            if show is None:
                continue
            if _systemd_show_is_exact_absent_v2(show):
                if previous_owned_invocation is not None:
                    if failed_boundary_row:
                        continue
                    raise ValueError("owned entry invocation disappeared")
                continue
            try:
                entry_ready = _systemd_entry_ready_v2(
                    show, name="cleanup-prefix entry"
                )
            except ValueError:
                # Ownership was already proved and locked by the producer;
                # malformed readiness is nevertheless an exact-final failure.
                if failed_boundary_row:
                    continue
                raise
            continue
        if purpose == f"terminal-show-{terminal_index}" and prefix_phase in {
            "entry",
            "terminal",
        }:
            if prefix_phase == "entry" and not entry_ready:
                raise ValueError("cleanup prefix entered terminal phase without active entry")
            prefix_phase = "terminal"
            terminal_index += 1
            if terminal_index > 33:
                raise ValueError("cleanup prefix terminal probe count exceeds 33")
            terminal_candidate = parse_owned_show(
                item,
                allow_absent=False,
                allow_failed_last=failed_boundary_row,
                phase_name="cleanup-prefix terminal",
            )
            if terminal_candidate is not None:
                try:
                    terminal_success = _systemd_terminal_success_v2(
                        terminal_candidate,
                        name="cleanup-prefix terminal",
                    )
                except ValueError:
                    if failed_boundary_row:
                        continue
                    raise
                if terminal_exit_seen:
                    if prestop_snapshot_seen:
                        raise ValueError(
                            "cleanup prefix continued terminal probes after pre-stop"
                        )
                    if not terminal_success:
                        if failed_boundary_row:
                            continue
                        raise ValueError(
                            "cleanup prefix pre-stop terminal snapshot drifted"
                        )
                    prestop_snapshot_seen = True
                elif terminal_success:
                    terminal_exit_seen = True
                last_terminal_show = terminal_candidate
            continue
        if purpose == "stop-after-terminal-snapshot" and prefix_phase == "terminal":
            if (
                not prestop_snapshot_seen
                or not terminal_exit_seen
                or last_terminal_show is None
                or not _systemd_terminal_success_v2(
                    last_terminal_show,
                    name="cleanup-prefix pre-stop terminal",
                )
            ):
                raise ValueError("cleanup prefix stop lacked terminal success state")
            prefix_phase = "post-stop"
            argv, _, rc, stdout, stderr = item
            if (
                argv != [systemctl_realpath, "--user", "stop", unit_name]
                or ((rc != 0 or stdout or stderr) and not failed_boundary_row)
            ):
                raise ValueError("main-prefix stop command drifted")
            continue
        if (
            purpose == f"post-stop-absent-check-{post_stop_index}"
            and prefix_phase == "post-stop"
        ):
            post_stop_index += 1
            if post_stop_index > 8:
                raise ValueError("cleanup prefix post-stop probe count exceeds eight")
            post_stop_candidate = parse_owned_show(
                item,
                allow_absent=True,
                allow_failed_last=failed_boundary_row,
                phase_name="cleanup-prefix post-stop",
            )
            if (
                post_stop_candidate is not None
                and _systemd_show_is_exact_absent_v2(post_stop_candidate)
                and position != boundary - 1
            ):
                raise ValueError("cleanup prefix continued after exact absence")
            continue
        raise ValueError("failure cleanup main-prefix FSM drifted")

    projected_owned_invocation = cleanup["owned_invocation_id"]
    if (
        owned_invocation is not None
        and projected_owned_invocation != owned_invocation
    ):
        raise ValueError(
            "failure cleanup main-prefix InvocationID is not projected exactly"
        )

    suffix = decoded[boundary:]
    if not suffix:
        if cleanup_verified:
            raise ValueError("verified cleanup lacks its cleanup suffix")
    else:
        if suffix[0][1] != "failure-owned-show":
            raise ValueError("failure cleanup suffix lacks its ownership probe")
        current = parse_owned_show(
            suffix[0],
            allow_absent=True,
            allow_failed_last=(not cleanup_verified and len(suffix) == 1),
            phase_name="failure cleanup initial ownership probe",
        )
        cursor = 1
        saw_term = False
        saw_kill = False
        saw_stop = False

        def live_members(show: dict[str, str] | None) -> bool:
            return _systemd_show_has_live_members_v2(
                show, name="failure cleanup replay"
            )

        def live_members_or_exact_final_failure(
            show: dict[str, str] | None, *, consumed_rows: int
        ) -> bool | None:
            try:
                return live_members(show)
            except ValueError:
                if not cleanup_verified and consumed_rows == len(suffix):
                    return None
                raise

        def require_next_purpose_or_unproven_end(expected: str, name: str) -> bool:
            if cursor == len(suffix):
                if cleanup_verified:
                    raise ValueError(f"verified cleanup omitted {name}")
                return False
            if suffix[cursor][1] != expected:
                raise ValueError(f"failure cleanup omitted required {name}")
            return True

        current_live = live_members_or_exact_final_failure(
            current, consumed_rows=cursor
        )
        if current is None or current_live is None:
            # The initial ownership probe itself is the exact final failing
            # transition of an unproven transcript.
            if cleanup_verified or cursor != len(suffix):
                raise ValueError("failure cleanup initial ownership probe drifted")
        elif current_live:
            require_next_purpose_or_unproven_end(
                "failure-owned-sigterm", "SIGTERM after a live ownership probe"
            )
        elif _systemd_show_is_exact_absent_v2(current):
            require_next_purpose_or_unproven_end(
                "failure-absent-0", "final absence confirmation"
            )
        else:
            require_next_purpose_or_unproven_end(
                "failure-owned-stop", "stop after a non-live owned probe"
            )

        if cursor < len(suffix) and suffix[cursor][1] == "failure-owned-sigterm":
            if current_live is not True:
                raise ValueError("cleanup SIGTERM lacked a live owned pre-state")
            argv, _, rc, stdout, stderr = suffix[cursor]
            action_failed = (
                argv
                != [
                    systemctl_realpath,
                    "--user",
                    "kill",
                    kill_who_option,
                    "--signal=SIGTERM",
                    unit_name,
                ]
                or rc != 0
                or stdout
                or stderr
            )
            if action_failed and not (
                not cleanup_verified and cursor == len(suffix) - 1
            ):
                raise ValueError("failure cleanup SIGTERM preimage drifted")
            saw_term = True
            cursor += 1
            grace = 0
            while cursor < len(suffix) and suffix[cursor][1] == f"failure-term-grace-{grace}":
                if grace >= 20:
                    raise ValueError("failure cleanup TERM grace exceeds twenty")
                current = parse_owned_show(
                    suffix[cursor],
                    allow_absent=True,
                    allow_failed_last=(
                        not cleanup_verified and cursor == len(suffix) - 1
                    ),
                    phase_name="failure cleanup TERM grace",
                )
                grace += 1
                cursor += 1
                current_live = live_members_or_exact_final_failure(
                    current, consumed_rows=cursor
                )
                if current is None or current_live is None or not current_live:
                    if (
                        cursor < len(suffix)
                        and suffix[cursor][1] == f"failure-term-grace-{grace}"
                    ):
                        raise ValueError("TERM grace continued after non-live state")
                    break
            signal_only_internal_failure = (
                not cleanup_verified and grace == 0 and cursor == len(suffix)
            )
            if not action_failed and not (
                1 <= grace <= 20 or signal_only_internal_failure
            ):
                raise ValueError("failure cleanup TERM grace sequence drifted")
            if not action_failed and grace > 0 and current is not None:
                if current_live:
                    require_next_purpose_or_unproven_end(
                        "failure-owned-sigkill",
                        "SIGKILL after live TERM grace",
                    )
                elif _systemd_show_is_exact_absent_v2(current):
                    require_next_purpose_or_unproven_end(
                        "failure-absent-0", "absence after TERM grace"
                    )
                else:
                    require_next_purpose_or_unproven_end(
                        "failure-owned-stop", "stop after TERM grace"
                    )
        if cursor < len(suffix) and suffix[cursor][1] == "failure-owned-sigkill":
            if not saw_term or current_live is not True:
                raise ValueError("cleanup SIGKILL lacked TERM/live owned pre-state")
            argv, _, rc, stdout, stderr = suffix[cursor]
            action_failed = (
                argv
                != [
                    systemctl_realpath,
                    "--user",
                    "kill",
                    kill_who_option,
                    "--signal=SIGKILL",
                    unit_name,
                ]
                or rc != 0
                or stdout
                or stderr
            )
            if action_failed and not (
                not cleanup_verified and cursor == len(suffix) - 1
            ):
                raise ValueError("failure cleanup SIGKILL preimage drifted")
            saw_kill = True
            cursor += 1
            grace = 0
            while cursor < len(suffix) and suffix[cursor][1] == f"failure-kill-grace-{grace}":
                if grace >= 8:
                    raise ValueError("failure cleanup KILL grace exceeds eight")
                current = parse_owned_show(
                    suffix[cursor],
                    allow_absent=True,
                    allow_failed_last=(
                        not cleanup_verified and cursor == len(suffix) - 1
                    ),
                    phase_name="failure cleanup KILL grace",
                )
                grace += 1
                cursor += 1
                current_live = live_members_or_exact_final_failure(
                    current, consumed_rows=cursor
                )
                if current is None or current_live is None or not current_live:
                    if (
                        cursor < len(suffix)
                        and suffix[cursor][1] == f"failure-kill-grace-{grace}"
                    ):
                        raise ValueError("KILL grace continued after non-live state")
                    break
            signal_only_internal_failure = (
                not cleanup_verified and grace == 0 and cursor == len(suffix)
            )
            if not action_failed and not (
                1 <= grace <= 8 or signal_only_internal_failure
            ):
                raise ValueError("failure cleanup KILL grace sequence drifted")
            if not action_failed and grace > 0 and current is not None:
                if current_live:
                    if cursor != len(suffix):
                        raise ValueError(
                            "failure cleanup continued after live KILL grace"
                        )
                    if cleanup_verified:
                        raise ValueError(
                            "verified cleanup retained live members after SIGKILL"
                        )
                elif _systemd_show_is_exact_absent_v2(current):
                    require_next_purpose_or_unproven_end(
                        "failure-absent-0", "absence after KILL grace"
                    )
                else:
                    require_next_purpose_or_unproven_end(
                        "failure-owned-stop", "stop after KILL grace"
                    )
        if cursor < len(suffix) and suffix[cursor][1] == "failure-owned-stop":
            if (
                current is None
                or _systemd_show_is_exact_absent_v2(current)
                or current_live is not False
            ):
                raise ValueError("cleanup stop lacked an owned present pre-state")
            argv, _, rc, stdout, stderr = suffix[cursor]
            action_failed = (
                argv != [systemctl_realpath, "--user", "stop", unit_name]
                or rc != 0
                or stdout
                or stderr
            )
            if action_failed and not (
                not cleanup_verified and cursor == len(suffix) - 1
            ):
                raise ValueError("failure cleanup stop preimage drifted")
            saw_stop = True
            cursor += 1
            if not action_failed:
                require_next_purpose_or_unproven_end(
                    "failure-absent-0", "absence probe after stop"
                )
        absent_count = 0
        saw_final_absent = False
        absence_observed = bool(
            current is not None and _systemd_show_is_exact_absent_v2(current)
        )
        while cursor < len(suffix):
            if suffix[cursor][1] != f"failure-absent-{absent_count}":
                raise ValueError("failure cleanup final-absence sequence drifted")
            if absent_count >= 8:
                raise ValueError("failure cleanup final-absence exceeds eight")
            previous_owned_invocation = owned_invocation
            current = parse_owned_show(
                suffix[cursor],
                allow_absent=True,
                allow_failed_last=(not cleanup_verified and cursor == len(suffix) - 1),
                phase_name="failure cleanup final absence",
            )
            absent_count += 1
            cursor += 1
            if (
                current is not None
                and absence_observed
                and not _systemd_show_is_exact_absent_v2(current)
            ):
                # The producer rejects reappearance before recording the new
                # InvocationID as owned cleanup state.
                owned_invocation = previous_owned_invocation
                if not cleanup_verified and cursor == len(suffix):
                    current = None
                    continue
                raise ValueError("cleanup unit reappeared after exact absence")
            if current is not None and _systemd_show_is_exact_absent_v2(current):
                saw_final_absent = True
                absence_observed = True
                if cursor != len(suffix):
                    raise ValueError("cleanup transcript continued after final absence")
        if cleanup_verified:
            if (
                not 1 <= absent_count <= 8
                or not saw_final_absent
                or cleanup["term_sent"] is not saw_term
                or cleanup["kill_sent"] is not saw_kill
                or cleanup["launcher_reaped"] is not True
                or cleanup["post_cleanup_unit_absent"] is not True
                or type(cleanup["launcher_returncode"]) is not int
                or type(cleanup["launcher_returncode"]) is bool
                or not -255 <= cleanup["launcher_returncode"] <= 255
            ):
                raise ValueError("verified failure cleanup completion drifted")
            if saw_stop and owned_invocation is None:
                raise ValueError("cleanup stopped a unit without owned identity")
        elif cursor != len(suffix):
            raise ValueError("unproven cleanup FSM did not consume its transcript")

    if (
        cleanup_verified
        and projected_owned_invocation != owned_invocation
    ) or (
        not cleanup_verified
        and projected_owned_invocation is not None
        and projected_owned_invocation != owned_invocation
    ):
        raise ValueError("failure cleanup owned invocation is not transcript-bound")
    if cleanup_verified:
        for stream, cap in (("stdout", 8 * 1_024 * 1_024), ("stderr", 256 * 1_024)):
            _require_exact_int_v2(
                cleanup[f"drained_{stream}_size_bytes"],
                f"failure cleanup drained {stream}",
                maximum=cap,
            )
            _require_raw_sha256_v2(
                cleanup[f"drained_{stream}_sha256"],
                f"failure cleanup drained {stream}",
            )
    elif (
        cleanup["external_intervention_required"] is not True
        or type(cleanup["cleanup_error_type"]) is not str
        or re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]{0,127}", cleanup["cleanup_error_type"]
        )
        is None
    ):
        raise ValueError("unproven cleanup intervention marker drifted")


def _open_sealed_memfd_v2(payload: bytes, name: str) -> int:
    if (
        type(payload) is not bytes
        or not payload
        or type(name) is not str
        or re.fullmatch(r"[a-z0-9-]{1,64}", name) is None
        or not hasattr(os, "memfd_create")
        or not hasattr(os, "MFD_ALLOW_SEALING")
    ):
        raise ValueError("sealed memfd input is outside the closed contract")
    descriptor = os.memfd_create(
        name, flags=os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
    )
    try:
        _write_all_fd_v2(descriptor, payload)
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        seals = (
            fcntl.F_SEAL_SEAL
            | fcntl.F_SEAL_SHRINK
            | fcntl.F_SEAL_GROW
            | fcntl.F_SEAL_WRITE
        )
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        if fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) != seals:
            raise ValueError("sealed memfd did not retain its exact seals")
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size != len(payload):
            raise ValueError("sealed memfd size/type drifted")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _run_bounded_control_command_v2(
    argv: list[str],
    *,
    purpose: str,
    deadline_ns: int,
    cleanup_deadline_ns: int | None = None,
    stdout_cap: int = 256 * 1_024,
    stderr_cap: int = 256 * 1_024,
    working_directory: Path | None = None,
    environment_overrides: dict[str, str] | None = None,
    clean_environment: bool = False,
    pass_fds: tuple[int, ...] = (),
) -> tuple[dict[str, object], bytes, bytes]:
    if type(clean_environment) is not bool:
        raise TypeError("bounded command clean-environment switch must be exact bool")
    if (
        type(pass_fds) is not tuple
        or len(pass_fds) != len(set(pass_fds))
        or any(
            type(descriptor) is not int
            or descriptor < 0
            or not stat.S_ISREG(os.fstat(descriptor).st_mode)
            for descriptor in pass_fds
        )
    ):
        raise ValueError("bounded command inherited descriptors are malformed")
    if type(deadline_ns) is not int or type(deadline_ns) is bool:
        raise TypeError("bounded command deadline must be one exact integer")
    if cleanup_deadline_ns is None:
        cleanup_deadline_ns = deadline_ns
        command_deadline_ns = (
            deadline_ns - _BOUNDED_COMMAND_REAP_RESERVE_NANOSECONDS_V2
        )
    else:
        if (
            type(cleanup_deadline_ns) is not int
            or type(cleanup_deadline_ns) is bool
            or cleanup_deadline_ns <= deadline_ns
        ):
            raise ValueError(
                "bounded command cleanup deadline must follow command deadline"
            )
        command_deadline_ns = deadline_ns
    if time.monotonic_ns() >= command_deadline_ns:
        raise TimeoutError("bounded command expired before child launch")
    environment = (
        {}
        if clean_environment
        else {**os.environ, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    )
    if environment_overrides is not None:
        if any(
            type(key) is not str
            or type(value) is not str
            or "\x00" in key
            or "\x00" in value
            for key, value in environment_overrides.items()
        ):
            raise ValueError("bounded command environment overrides are malformed")
        environment.update(environment_overrides)
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env=environment,
        cwd=None if working_directory is None else str(working_directory),
        pass_fds=pass_fds,
    )
    try:
        try:
            stdout, stderr, returncode = _capture_process_pipes_v2(
                process,
                deadline_ns=command_deadline_ns,
                stdout_cap=stdout_cap,
                stderr_cap=stderr_cap,
                wait_for_process_exit=True,
            )
        except BaseException as primary:
            try:
                _terminate_process_group_v2(
                    process, deadline_ns=cleanup_deadline_ns
                )
            except BaseException as cleanup_exc:
                raise RuntimeError(
                    "bounded command failed and its process group was not reaped"
                ) from cleanup_exc
            raise primary
        if type(returncode) is not int:
            raise RuntimeError("bounded control command has no exact exit code")
        return (
            _command_evidence_v2(
                argv=argv,
                purpose=purpose,
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            ),
            stdout,
            stderr,
        )
    finally:
        _close_process_pipes_v2(process, name="bounded control command")


def _rfc_raw_artifact_reference_v2(
    bundle_directory: Path,
    name: str,
    payload: bytes,
    *,
    deadline_ns: int,
) -> dict[str, object]:
    _atomic_write_new_file_v2(
        bundle_directory, name, payload, deadline_ns=deadline_ns
    )
    return {
        "path": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _rfc_command_row_v2(
    *,
    bundle_directory: Path,
    index: int,
    name: str,
    argv: list[str],
    cwd: Path,
    environment_overrides: dict[str, str],
    returncode: int,
    stdout: bytes,
    stderr: bytes,
    started_ns: int,
    ended_ns: int,
    deadline_ns: int,
) -> dict[str, object]:
    prefix = f"command-{index:03d}-{name}"
    argv_bytes = canonicalize_rfc8785_v1(argv) + b"\n"
    return {
        "argv_artifact": _rfc_raw_artifact_reference_v2(
            bundle_directory,
            prefix + ".argv.json",
            argv_bytes,
            deadline_ns=deadline_ns,
        ),
        "cwd": str(cwd),
        "elapsed_nanoseconds": ended_ns - started_ns,
        "ended_monotonic_nanoseconds": ended_ns,
        "environment_overrides": environment_overrides,
        "name": name,
        "returncode": returncode,
        "started_monotonic_nanoseconds": started_ns,
        "stderr_artifact": _rfc_raw_artifact_reference_v2(
            bundle_directory,
            prefix + ".stderr.bin",
            stderr,
            deadline_ns=deadline_ns,
        ),
        "stdout_artifact": _rfc_raw_artifact_reference_v2(
            bundle_directory,
            prefix + ".stdout.bin",
            stdout,
            deadline_ns=deadline_ns,
        ),
    }


def build_rfc8785_prerequisite_bundle_v2(
    *, project_root: Path, bundle_directory: Path
) -> bytes:
    """Execute and retain the raw RFC-8785 prerequisite command bundle."""

    if (
        not isinstance(project_root, Path)
        or not isinstance(bundle_directory, Path)
        or not project_root.is_absolute()
        or not bundle_directory.is_absolute()
        or bundle_directory.exists()
    ):
        raise ValueError("RFC prerequisite builder paths are not fresh absolute paths")
    deadline_ns = time.monotonic_ns() + 1_800_000_000_000
    project_root = project_root.resolve(strict=True)
    expected_dependencies = {
        "docs/Hegel_Machine_Phase2B_RFC8785_Implementation_v1.md": (
            RFC8785_DOCUMENTATION_SHA256
        ),
        "golden_vectors/phase2b_rfc8785_v1.json": RFC8785_GOLDEN_VECTORS_SHA256,
        "src/hegel_machine/rfc8785_v1.py": RFC8785_SOURCE_SHA256,
        "tests/test_rfc8785_v1.py": RFC8785_TEST_SHA256,
        "tools/phase2b_rfc8785_oracle_v1.mjs": RFC8785_NODE_ORACLE_SHA256,
    }
    for relative, digest in expected_dependencies.items():
        path = project_root / relative
        if (
            time.monotonic_ns() >= deadline_ns
            or not path.is_file()
            or _sha256_file_v2(path, deadline_ns=deadline_ns) != digest
        ):
            raise ValueError(f"RFC prerequisite dependency drifted: {relative}")
    bundle_directory = _mkdir_new_directory_v2(
        bundle_directory.parent,
        bundle_directory.name,
        deadline_ns=deadline_ns,
    )
    python_row = _executable_projection_for_runner_v2(
        Path(sys.executable),
        version_argv=("--version",),
        name="RFC Python",
        deadline_ns=deadline_ns,
    )
    node_lookup = shutil.which("node")
    if node_lookup is None:
        raise ValueError("RFC prerequisite cannot locate Node")
    node_row = _executable_projection_for_runner_v2(
        Path(node_lookup),
        version_argv=("--version",),
        name="RFC Node",
        deadline_ns=deadline_ns,
    )
    node_realpath = node_row["realpath"]
    v8_evidence, v8_stdout, v8_stderr = _run_bounded_control_command_v2(
        [node_realpath, "-p", "process.versions.v8"],
        purpose="rfc-v8-version",
        deadline_ns=deadline_ns,
        working_directory=project_root,
    )
    if v8_evidence["returncode"] != 0 or v8_stderr:
        raise ValueError("RFC prerequisite V8 version probe failed")
    v8_version = v8_stdout.decode("utf-8", errors="strict").strip()
    if not v8_version:
        raise ValueError("RFC prerequisite V8 version is empty")
    environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": str(Path(node_realpath).parent),
        "TZ": "UTC",
    }
    commands: list[dict[str, object]] = []

    def run_and_retain(
        name: str,
        argv: list[str],
        *,
        stdout_cap: int = 2 * 1_024 * 1_024,
        command_environment: dict[str, str] | None = None,
        pass_fds: tuple[int, ...] = (),
    ) -> tuple[bytes, bytes]:
        started = time.monotonic_ns()
        if started >= deadline_ns:
            raise TimeoutError("RFC retained command reached the builder deadline")
        evidence, stdout, stderr = _run_bounded_control_command_v2(
            argv,
            purpose="rfc-" + name,
            deadline_ns=deadline_ns,
            stdout_cap=stdout_cap,
            stderr_cap=256 * 1_024,
            working_directory=project_root,
            environment_overrides=(
                environment if command_environment is None else command_environment
            ),
            clean_environment=True,
            pass_fds=pass_fds,
        )
        ended = time.monotonic_ns()
        row = _rfc_command_row_v2(
            bundle_directory=bundle_directory,
            index=len(commands),
            name=name,
            argv=argv,
            cwd=project_root,
            environment_overrides=(
                environment if command_environment is None else command_environment
            ),
            returncode=evidence["returncode"],
            stdout=stdout,
            stderr=stderr,
            started_ns=started,
            ended_ns=ended,
            deadline_ns=deadline_ns,
        )
        commands.append(row)
        return stdout, stderr

    python_version_stdout, python_version_stderr = run_and_retain(
        "python-version", [python_row["realpath"], "--version"]
    )
    node_version_stdout, node_version_stderr = run_and_retain(
        "node-version", [node_realpath, "--version"]
    )
    retained_v8_stdout, retained_v8_stderr = run_and_retain(
        "v8-version", [node_realpath, "-p", "process.versions.v8"]
    )
    if (
        (python_version_stdout + python_version_stderr).decode("utf-8").strip()
        != python_row["version"]
        or (node_version_stdout + node_version_stderr).decode("utf-8").strip()
        != node_row["version"]
        or retained_v8_stderr
        or retained_v8_stdout.decode("utf-8").strip() != v8_version
    ):
        raise ValueError("RFC retained executable version probes drifted")
    qualification_source = (
        project_root
        / "src/hegel_machine/phase2b_formal_wire_covert_qualification_v2.py"
    )
    qualification_source_sha256 = _sha256_file_v2(
        qualification_source, deadline_ns=deadline_ns
    )
    kat_version_argv = _source_execution_bootstrap_argv_v2(
        python_realpath=python_row["realpath"],
        project_root=project_root,
        qualification_source_sha256=qualification_source_sha256,
        source_arguments=[
            "--entry",
            "rfc-selfcheck-version",
            "--project-root",
            str(project_root),
        ],
    )
    kat_version_stdout, kat_version_stderr = run_and_retain(
        "source-only-kat-version", kat_version_argv
    )
    if (
        kat_version_stderr
        or kat_version_stdout != b"hegel-rfc8785-stdlib-selfcheck-v1\n"
    ):
        raise ValueError("RFC prerequisite source-only KAT version drifted")
    kat_argv = _source_execution_bootstrap_argv_v2(
        python_realpath=python_row["realpath"],
        project_root=project_root,
        qualification_source_sha256=qualification_source_sha256,
        source_arguments=[
            "--entry",
            "rfc-selfcheck",
            "--project-root",
            str(project_root),
        ],
    )
    kat_stdout, kat_stderr = run_and_retain("source-only-kat-18", kat_argv)
    if (
        kat_stderr
        or commands[-1]["returncode"] != 0
        or re.fullmatch(rb"\.{18}[ ]+\[100%\]\n", kat_stdout) is None
    ):
        raise ValueError("RFC prerequisite source-only KAT did not report 18/18")
    oracle_path = project_root / "tools/phase2b_rfc8785_oracle_v1.mjs"
    vectors_path = project_root / "golden_vectors/phase2b_rfc8785_v1.json"
    oracle_source_bytes = _read_bounded_regular_file_v2(
        oracle_path,
        maximum_bytes=256 * 1_024,
        name="RFC Node oracle source",
        deadline_ns=deadline_ns,
    )
    vectors_bytes = _read_bounded_regular_file_v2(
        vectors_path,
        maximum_bytes=256 * 1_024,
        name="RFC golden vectors",
        deadline_ns=deadline_ns,
    )
    if (
        hashlib.sha256(oracle_source_bytes).hexdigest() != RFC8785_NODE_ORACLE_SHA256
        or hashlib.sha256(vectors_bytes).hexdigest()
        != RFC8785_GOLDEN_VECTORS_SHA256
    ):
        raise ValueError("RFC Node execution inputs drifted before open-once launch")
    oracle_source_text = oracle_source_bytes.decode("utf-8", errors="strict")
    vectors_fd = _open_sealed_memfd_v2(vectors_bytes, "hegel-rfc8785-vectors")
    oracle_argv = [
        node_realpath,
        "--input-type=module",
        "--eval",
        oracle_source_text,
        "--",
        str(oracle_path),
        "--check",
        f"/proc/self/fd/{vectors_fd}",
    ]
    try:
        oracle_stdout, oracle_stderr = run_and_retain(
            "node-oracle", oracle_argv, pass_fds=(vectors_fd,)
        )
    finally:
        _close_owned_descriptors_v2(
            (vectors_fd,), name="RFC builder sealed vectors"
        )
    if oracle_stderr or commands[-1]["returncode"] != 0:
        raise ValueError("RFC prerequisite Node oracle command failed")
    oracle_decoded = json.loads(oracle_stdout)
    if oracle_decoded != {
        "disposition": "PASS",
        "node": node_row["version"],
        "number_cases": 24,
        "oracle": "node-v8-json-stringify",
        "rejection_cases": 2,
        "serialization_cases": 7,
        "v8": v8_version,
    }:
        raise ValueError("RFC prerequisite Node oracle payload drifted")
    rng = random.Random(0x8785_2026_0814)
    finite: list[tuple[str, float]] = []
    while len(finite) < 100_000:
        if (
            len(finite) % 1_024 == 0
            and time.monotonic_ns() >= deadline_ns
        ):
            raise TimeoutError("RFC finite vector generation reached deadline")
        bits = rng.getrandbits(64)
        value = struct.unpack(">d", bits.to_bytes(8, "big"))[0]
        if math.isfinite(value):
            finite.append((f"{bits:016x}", value))
    differential_rows: list[dict[str, object]] = []
    checked = 0
    for batch_index, start in enumerate(range(0, 100_000, 2_000)):
        if time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("RFC differential batch reached deadline")
        batch = finite[start : start + 2_000]
        argv = [
            node_realpath,
            "--input-type=module",
            "--eval",
            oracle_source_text,
            "--",
            str(oracle_path),
            "--number-bits",
            *[bits for bits, _ in batch],
        ]
        stdout, stderr = run_and_retain(
            f"differential-{batch_index:03d}", argv, stdout_cap=2 * 1_024 * 1_024
        )
        if stderr or commands[-1]["returncode"] != 0:
            raise ValueError("RFC prerequisite differential Node batch failed")
        decoded_rows = json.loads(stdout)
        if type(decoded_rows) is not list or len(decoded_rows) != len(batch):
            raise ValueError("RFC prerequisite differential batch shape drifted")
        for (hex_bits, value), oracle_row in zip(batch, decoded_rows, strict=True):
            if checked % 1_024 == 0 and time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("RFC differential row replay reached deadline")
            if (
                type(oracle_row) is not dict
                or set(oracle_row) != {"canonical", "ieee754_hex"}
                or oracle_row["ieee754_hex"] != hex_bits
                or oracle_row["canonical"]
                != canonicalize_rfc8785_v1(value).decode("utf-8")
            ):
                raise ValueError("RFC prerequisite differential mismatch")
            checked += 1
        differential_rows.append(
            {
                "batch_index": batch_index,
                "case_count": len(batch),
                "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                "stdout_size_bytes": len(stdout),
            }
        )
    if checked != 100_000:
        raise ValueError("RFC prerequisite differential count drifted")
    raw_manifest: list[dict[str, object]] = []
    for command in commands:
        for key in ("argv_artifact", "stdout_artifact", "stderr_artifact"):
            artifact = command[key]
            if type(artifact) is not dict:
                raise ValueError("RFC raw artifact projection drifted")
            raw_manifest.append(artifact)
    raw_manifest.sort(key=lambda row: row["path"])
    raw_execution = {
        "schema_version": "hegel-machine-rfc8785-raw-execution-bundle/2",
        "bundle_manifest": raw_manifest,
        "command_count": len(commands),
        "commands": commands,
        "source_only_kat_version": "hegel-rfc8785-stdlib-selfcheck-v1",
        "qualification_source_sha256": _sha256_file_v2(
            qualification_source, deadline_ns=deadline_ns
        ),
        "node_oracle_execution": {
            "golden_vectors_sha256": hashlib.sha256(vectors_bytes).hexdigest(),
            "golden_vectors_size_bytes": len(vectors_bytes),
            "oracle_source_sha256": hashlib.sha256(oracle_source_bytes).hexdigest(),
            "oracle_source_size_bytes": len(oracle_source_bytes),
            "source_transport": "NODE_ESM_EVAL_ARGV",
            "vectors_transport": "SEALED_MEMFD_PROC_FD",
        },
        "raw_artifact_total_bytes": sum(row["size_bytes"] for row in raw_manifest),
        "transcript_root": stable_hash(
            commands, prefix="phase2b_rfc8785_raw_command_transcript_v1_"
        ),
        "working_tree_root": str(project_root),
    }
    evidence_body = {
        "schema_version": RFC8785_PREREQUISITE_SCHEMA_V2,
        "disposition": "PASS",
        "dependency_sha256": {
            "documentation": RFC8785_DOCUMENTATION_SHA256,
            "golden_vectors": RFC8785_GOLDEN_VECTORS_SHA256,
            "node_oracle": RFC8785_NODE_ORACLE_SHA256,
            "source": RFC8785_SOURCE_SHA256,
            "test": RFC8785_TEST_SHA256,
        },
        "executables": {
            "node": node_row,
            "python": python_row,
            "v8_version": v8_version,
        },
        "source_only_kat_18": {
            "collected": 18,
            "exit_code": 0,
            "passed": 18,
            "runner": "STDLIB_SOURCE_ONLY_PINNED_TEST_HARNESS_V1",
            "stderr_sha256": hashlib.sha256(kat_stderr).hexdigest(),
            "stdout_sha256": hashlib.sha256(kat_stdout).hexdigest(),
        },
        "node_oracle": {
            "disposition": "PASS",
            "exit_code": 0,
            "number_cases": 24,
            "rejection_cases": 2,
            "serialization_cases": 7,
            "stderr_sha256": hashlib.sha256(oracle_stderr).hexdigest(),
            "stdout_sha256": hashlib.sha256(oracle_stdout).hexdigest(),
        },
        "differential": {
            "batch_size": 2_000,
            "checked": checked,
            "exit_code": 0,
            "finite_binary64_cases": 100_000,
            "mismatch_count": 0,
            "seed_hex": "0x878520260814",
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "stdout_sha256": hashlib.sha256(
                canonicalize_rfc8785_v1(differential_rows)
            ).hexdigest(),
        },
        "raw_execution": raw_execution,
    }
    evidence_bytes = canonicalize_rfc8785_v1(evidence_body)
    evidence_sha = hashlib.sha256(evidence_bytes).hexdigest()
    _atomic_write_new_file_v2(
        bundle_directory,
        evidence_sha,
        evidence_bytes,
        deadline_ns=deadline_ns,
    )
    receipt = {
        **evidence_body,
        "cas": {
            "content_sha256": evidence_sha,
            "object_key": "sha256:" + evidence_sha,
            "object_size_bytes": len(evidence_bytes),
            "prior_object_count": 0,
            "put_exit_code": 0,
            "replay_verified": True,
            "resulting_object_count": 1,
        },
    }
    receipt_bytes = _canonical_inner_artifact_v2(receipt)
    _verify_rfc8785_raw_execution_bundle_v2(
        receipt,
        bundle_directory,
        expected_qualification_source_sha256=_sha256_file_v2(
            qualification_source, deadline_ns=deadline_ns
        ),
        fresh_execute=False,
        deadline_ns=deadline_ns,
    )
    _atomic_write_new_file_v2(
        bundle_directory,
        "receipt.json",
        receipt_bytes,
        deadline_ns=deadline_ns,
    )
    return receipt_bytes


def _parse_systemd_show_v2(payload: bytes, expected_keys: set[str]) -> dict[str, str]:
    if len(payload) > 256 * 1_024:
        raise ValueError("systemd show output exceeds its cap")
    text = payload.decode("utf-8", errors="strict")
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            raise ValueError("systemd show row is malformed")
        key, value = line.split("=", 1)
        if key in result:
            raise ValueError("systemd show key is duplicated")
        result[key] = value
    if set(result) != expected_keys:
        raise ValueError("systemd show property set drifted")
    return result


def _parse_systemd_boolean_v2(value: str, name: str) -> bool:
    if value == "yes":
        return True
    if value == "no":
        return False
    raise ValueError(f"systemd {name} is not an exact yes/no property")


_SYSTEMD_SHOW_PROPERTIES_V2: Final = (
    "ActiveState",
    "ControlGroup",
    "Description",
    "ExecMainCode",
    "ExecMainStatus",
    "FinalKillSignal",
    "InvocationID",
    "KillMode",
    "KillSignal",
    "LoadState",
    "MainPID",
    "MemoryMax",
    "RemainAfterExit",
    "Result",
    "RuntimeMaxUSec",
    "SendSIGKILL",
    "SubState",
    "TasksMax",
    "TimeoutStopUSec",
    "Type",
    "WorkingDirectory",
)


def _systemd_show_argv_v2(systemctl: str, unit_name: str) -> list[str]:
    return [
        systemctl,
        "--user",
        "show",
        unit_name,
        "--no-pager",
        *[f"--property={name}" for name in _SYSTEMD_SHOW_PROPERTIES_V2],
    ]


def _parse_systemd_unsigned_v2(value: str, name: str) -> int:
    if re.fullmatch(r"(?:0|[1-9][0-9]*)", value) is None:
        raise ValueError(f"systemd {name} is not exact unsigned decimal")
    return int(value)


def _parse_systemd_duration_microseconds_v2(value: str, name: str) -> int:
    factors = {
        "us": 1,
        "ms": 1_000,
        "s": 1_000_000,
        "min": 60_000_000,
        "h": 3_600_000_000,
        "d": 86_400_000_000,
    }
    position = 0
    total = 0
    matched = False
    pattern = re.compile(r"(?:[ ]*)([0-9]+)(us|ms|s|min|h|d)")
    while position < len(value):
        match = pattern.match(value, position)
        if match is None:
            raise ValueError(f"systemd {name} duration is outside the closed grammar")
        matched = True
        total += int(match.group(1)) * factors[match.group(2)]
        position = match.end()
    if not matched:
        raise ValueError(f"systemd {name} duration is empty")
    return total


def _normalize_effective_systemd_properties_v2(
    show: dict[str, str],
    *,
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
    source_tree: Path,
) -> dict[str, object]:
    working_directory = show["WorkingDirectory"]
    if working_directory.startswith("!"):
        working_directory = working_directory[1:]
    result = {
        "FinalKillSignal": _parse_systemd_unsigned_v2(
            show["FinalKillSignal"], "FinalKillSignal"
        ),
        "KillMode": show["KillMode"],
        "KillSignal": _parse_systemd_unsigned_v2(
            show["KillSignal"], "KillSignal"
        ),
        "MemoryMax": _parse_systemd_unsigned_v2(show["MemoryMax"], "MemoryMax"),
        "RemainAfterExit": _parse_systemd_boolean_v2(
            show["RemainAfterExit"], "RemainAfterExit"
        ),
        "RuntimeMaxUSec": _parse_systemd_duration_microseconds_v2(
            show["RuntimeMaxUSec"], "RuntimeMaxUSec"
        ),
        "SendSIGKILL": _parse_systemd_boolean_v2(
            show["SendSIGKILL"], "SendSIGKILL"
        ),
        "TasksMax": _parse_systemd_unsigned_v2(show["TasksMax"], "TasksMax"),
        "TimeoutStopUSec": _parse_systemd_duration_microseconds_v2(
            show["TimeoutStopUSec"], "TimeoutStopUSec"
        ),
        "Type": show["Type"],
    }
    expected = {
        "FinalKillSignal": int(signal.SIGKILL),
        "KillMode": "control-group",
        "KillSignal": int(signal.SIGTERM),
        "MemoryMax": attempt.requested_cgroup_memory_max_bytes,
        "RemainAfterExit": True,
        "RuntimeMaxUSec": LIVE_HARD_TOTAL_TIMEOUT_SECONDS * 1_000_000,
        "SendSIGKILL": True,
        "TasksMax": attempt.requested_cgroup_pids_max,
        "TimeoutStopUSec": 30_000_000,
        "Type": "exec",
    }
    if not _exact_json_equal(result, expected) or working_directory != str(source_tree):
        raise ValueError("effective systemd service properties drifted")
    return result


def _evaluate_owned_or_absent_systemd_show_v2(
    show: dict[str, str],
    *,
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
    source_tree: Path,
    ownership_marker: str,
    expected_invocation_id: str | None,
    name: str,
) -> tuple[str, str | None]:
    """Validate the common show projection used by producer and replayers."""

    if _systemd_show_is_exact_absent_v2(show):
        return "absent", None
    invocation = show["InvocationID"]
    if (
        show["LoadState"] != "loaded"
        or show["Description"] != ownership_marker
        or re.fullmatch(r"[0-9a-f]{32}", invocation) is None
    ):
        raise ValueError(f"{name} observed an unowned unit")
    if expected_invocation_id is not None and invocation != expected_invocation_id:
        raise ValueError(f"{name} InvocationID changed")
    _normalize_effective_systemd_properties_v2(
        show, attempt=attempt, source_tree=source_tree
    )
    return "owned", invocation


def _systemd_entry_ready_v2(show: dict[str, str], *, name: str) -> bool:
    """Return exact readiness; reject malformed PID/cgroup observations."""

    if _systemd_show_is_exact_absent_v2(show):
        return False
    main_pid = _parse_systemd_unsigned_v2(show["MainPID"], f"{name} MainPID")
    control_group = show["ControlGroup"]
    if control_group:
        _require_safe_cgroup_path_v2(control_group, f"{name} cgroup")
    ready = show["ActiveState"] == "active" and show["SubState"] == "running"
    if ready and (main_pid <= 0 or not control_group):
        raise ValueError(f"{name} active/running state lacks PID/cgroup")
    return ready


def _systemd_terminal_success_v2(
    show: dict[str, str], *, name: str
) -> bool:
    """Return exact active/exited success; reject noncanonical terminal scalars."""

    if _systemd_show_is_exact_absent_v2(show):
        return False
    exec_code = _parse_systemd_unsigned_v2(
        show["ExecMainCode"], f"{name} ExecMainCode"
    )
    exec_status = _parse_systemd_unsigned_v2(
        show["ExecMainStatus"], f"{name} ExecMainStatus"
    )
    main_pid = _parse_systemd_unsigned_v2(show["MainPID"], f"{name} MainPID")
    is_exited = show["ActiveState"] == "active" and show["SubState"] == "exited"
    if not is_exited:
        return False
    if (
        show["Result"] != "success"
        or exec_code != 1
        or exec_status != 0
        or main_pid != 0
        or show["ControlGroup"] != ""
    ):
        raise ValueError(f"{name} active/exited success projection drifted")
    return True


def _systemd_show_has_live_members_v2(
    show: dict[str, str] | None, *, name: str
) -> bool:
    if show is None or _systemd_show_is_exact_absent_v2(show):
        return False
    main_pid = _parse_systemd_unsigned_v2(show["MainPID"], f"{name} MainPID")
    control_group = show["ControlGroup"]
    if control_group:
        _require_safe_cgroup_path_v2(control_group, f"{name} cgroup")
    return bool(
        main_pid > 0
        or control_group
        or show["SubState"] in {"running", "start", "start-pre", "stop"}
    )


def _closed_outer_failure_artifact_v2(
    attempt: Phase2BFormalWireCovertAttemptManifestV2,
    *,
    entry_kind: str,
    stage: str,
    primary_failure_exception_type: str,
    reason_code: str,
    cleanup: dict[str, object] | None,
    cleanup_failure_exception_type: str | None,
    child_pipe_capture: dict[str, object] | None,
    failure_cas_projection: dict[str, object],
    cleanup_verified: bool,
    external_intervention_required: bool,
    launcher_started: bool,
    managed_atomic_residues: list[dict[str, object]],
    retained_tree_before_terminal: dict[str, object],
    started_ns: int,
    deadline_ns: int,
) -> bytes:
    if entry_kind not in {"development-calibration", "live-inner"}:
        raise ValueError("outer failure entry kind is not closed")
    if (
        type(cleanup_verified) is not bool
        or type(external_intervention_required) is not bool
        or cleanup_verified is external_intervention_required
        or type(launcher_started) is not bool
        or type(primary_failure_exception_type) is not str
        or re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]{0,127}", primary_failure_exception_type
        )
        is None
        or (
            cleanup_failure_exception_type is not None
            and (
                type(cleanup_failure_exception_type) is not str
                or re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_]{0,127}",
                    cleanup_failure_exception_type,
                )
                is None
            )
        )
        or (cleanup_verified and cleanup_failure_exception_type is not None)
        or (not cleanup_verified and cleanup_failure_exception_type is None)
        or launcher_started is (child_pipe_capture is None)
    ):
        raise ValueError("outer failure cleanup disposition is inconsistent")
    _managed_atomic_residue_paths_v2(managed_atomic_residues)
    _validate_failure_cas_projection_shape_v2(failure_cas_projection)
    if (
        type(retained_tree_before_terminal) is not dict
        or set(retained_tree_before_terminal)
        != {
            "regular_file_count",
            "root",
            "row_count",
            "top_level_names",
            "total_regular_bytes",
        }
        or _require_exact_int_v2(
            retained_tree_before_terminal["row_count"],
            "failure retained-tree row count",
            maximum=100_000,
        )
        < 0
        or _require_exact_int_v2(
            retained_tree_before_terminal["regular_file_count"],
            "failure retained-tree regular count",
            maximum=100_000,
        )
        < 0
        or _require_exact_int_v2(
            retained_tree_before_terminal["total_regular_bytes"],
            "failure retained-tree byte count",
            maximum=_MAXIMUM_RETAINED_ATTEMPT_TREE_BYTES_V2,
        )
        < 0
        or type(retained_tree_before_terminal["top_level_names"]) is not list
        or len(retained_tree_before_terminal["top_level_names"]) > 32
        or any(
            type(name) is not str
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) is None
            for name in retained_tree_before_terminal["top_level_names"]
        )
        or retained_tree_before_terminal["top_level_names"]
        != sorted(set(retained_tree_before_terminal["top_level_names"]))
        or type(retained_tree_before_terminal["root"]) is not str
        or re.fullmatch(
            r"phase2b_formal_wire_covert_failure_tree_v3_[0-9a-f]{64}",
            retained_tree_before_terminal["root"],
        )
        is None
    ):
        raise ValueError("outer failure retained-tree manifest is not closed")
    if re.fullmatch(r"[A-Z0-9_]{1,96}", stage) is None:
        stage = "UNCLASSIFIED"
    capture_persistence_verified = (
        child_pipe_capture is None
        or (
            type(child_pipe_capture) is dict
            and child_pipe_capture.get("capture_persistence_verified") is True
        )
    )
    if (
        launcher_started
        and cleanup_verified
        and capture_persistence_verified
        and (
            type(child_pipe_capture) is not dict
            or child_pipe_capture.get("capture_complete") is not True
        )
    ):
        raise ValueError("verified cleanup lacks complete child pipe evidence")
    if (
        launcher_started
        and not cleanup_verified
        and (
            type(child_pipe_capture) is not dict
            or child_pipe_capture.get("capture_complete") is not False
        )
    ):
        raise ValueError("unproven cleanup cannot claim complete child pipe evidence")
    finalization_ready_ns = time.monotonic_ns()
    if finalization_ready_ns >= deadline_ns:
        raise TimeoutError("outer failure terminal missed the hard total deadline")
    body = {
        "schema_version": OUTER_ATTEMPT_FAILURE_SCHEMA_V2,
        "disposition": (
            "ATOMIC_FAILURE_CLEANUP_UNPROVEN_EXTERNAL_INTERVENTION_REQUIRED"
            if not cleanup_verified
            else (
                "ATOMIC_EMPTY_REJECTION"
                if capture_persistence_verified
                else "ATOMIC_FAILURE_PIPE_CAPTURE_PERSISTENCE_UNVERIFIED"
            )
        ),
        "campaign_id": attempt.campaign_id,
        "attempt_id": attempt.attempt_id,
        "attempt_manifest_root": attempt.attempt_manifest_root,
        "entry_kind": entry_kind,
        "profile_id": attempt.profile_id,
        "reason_code": reason_code,
        "primary_failure_stage": stage,
        "primary_failure_exception_type": primary_failure_exception_type,
        "cleanup_failure_exception_type": cleanup_failure_exception_type,
        "child_pipe_capture": child_pipe_capture,
        "failure_cas_projection": failure_cas_projection,
        "cleanup": cleanup,
        "cleanup_verified": cleanup_verified,
        "external_intervention_required": external_intervention_required,
        "launcher_started": launcher_started,
        "managed_atomic_residues": managed_atomic_residues,
        "published_accepted_audit_runs": 0,
        "execution_completion_not_claimed": True,
        "qualification_receipt_id": None,
        "run_roots": [],
        "retry_requires_new_attempt_id": True,
        "retained_tree_before_terminal": retained_tree_before_terminal,
        "terminal_publication_count": 1,
        "timing": {
            "elapsed_nanoseconds_at_finalization_ready": (
                finalization_ready_ns - started_ns
            ),
            "finalization_ready_monotonic_nanoseconds": finalization_ready_ns,
            "started_monotonic_nanoseconds": started_ns,
            "watchdog_deadline_monotonic_nanoseconds": deadline_ns,
        },
        "claims": {
            "r03_formal_trusted_wire_implementation_qualified": False,
            "r04_formal_covert_auditor_implementation_qualified": False,
            "r05_synthetic_dev_wire_covert_qualification_verified": False,
            "actual_corpus_audit": False,
            "actual_unsealed_960_generation": False,
            "origin_authenticated": False,
            "key_independence_verified": False,
            "campaign_execution_authorized": False,
        },
    }
    return _canonical_inner_artifact_v2(body)


def _source_tree_manifest_at_fd_v2(
    root_fd: int, *, deadline_ns: int | None = None
) -> tuple[list[dict[str, object]], str]:
    if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
        raise ValueError("source tree manifest descriptor is not a directory")
    rows: list[dict[str, object]] = []
    total_regular_bytes = 0
    walk_fd = os.dup(root_fd)
    try:
        for directory, directory_names, file_names, directory_fd in os.fwalk(
            ".", topdown=True, follow_symlinks=False, dir_fd=walk_fd
        ):
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("source tree manifest reached the absolute deadline")
            directory_names.sort()
            file_names.sort()
            relative_directory = "" if directory == "." else directory[2:]
            for name in tuple(directory_names):
                if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                    raise TimeoutError(
                        "source tree manifest reached the absolute deadline"
                    )
                child_stat = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
                if not stat.S_ISDIR(child_stat.st_mode):
                    raise ValueError("source tree manifest rejects directory symlinks")
                relative = (
                    name
                    if not relative_directory
                    else relative_directory + "/" + name
                )
                rows.append({"path": relative, "type": "directory"})
                if len(rows) > _MAXIMUM_RETAINED_ATTEMPT_TREE_ROWS_V2:
                    raise ValueError("source tree manifest row count exceeds the cap")
            for name in file_names:
                relative = (
                    name
                    if not relative_directory
                    else relative_directory + "/" + name
                )
                if name.endswith((".pyc", ".pyo")):
                    raise ValueError("source tree manifest rejects bytecode")
                digest, child_stat = _hash_regular_at_fd_v2(
                    directory_fd,
                    name,
                    maximum_bytes=512 * 1_024 * 1_024,
                    deadline_ns=deadline_ns,
                )
                rows.append(
                    {
                        "path": relative,
                        "type": "regular",
                        "size_bytes": child_stat.st_size,
                        "sha256": digest,
                    }
                )
                total_regular_bytes += child_stat.st_size
                if (
                    len(rows) > _MAXIMUM_RETAINED_ATTEMPT_TREE_ROWS_V2
                    or total_regular_bytes
                    > _MAXIMUM_RETAINED_ATTEMPT_TREE_BYTES_V2
                ):
                    raise ValueError("source tree manifest aggregate exceeds the cap")
    finally:
        _close_owned_descriptors_v2((walk_fd,), name="source tree manifest walk")
    rows.sort(key=lambda row: (row["path"], row["type"]))
    root_hash = stable_hash(
        rows, prefix="phase2b_formal_wire_covert_source_tree_v2_"
    )
    return rows, root_hash


def _source_tree_manifest_v2(
    root: Path, *, deadline_ns: int | None = None
) -> tuple[list[dict[str, object]], str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("source tree manifest root is not a real directory")
    root_fd = _open_real_directory_fd_v2(root)
    try:
        return _source_tree_manifest_at_fd_v2(root_fd, deadline_ns=deadline_ns)
    finally:
        _close_owned_descriptors_v2((root_fd,), name="source tree manifest root")


def _regular_file_manifest_root_v2(rows: list[dict[str, object]]) -> str:
    regular_rows = [row for row in rows if row.get("type") == "regular"]
    return stable_hash(
        regular_rows,
        prefix="phase2b_formal_wire_covert_source_archive_regular_files_v2_",
    )


def _verify_source_archive_full_tree_equivalence_v2(
    archive_rows: list[dict[str, object]],
    archive_root: str,
    extracted_rows: list[dict[str, object]],
    extracted_root: str,
) -> None:
    if (
        type(archive_rows) is not list
        or type(extracted_rows) is not list
        or type(archive_root) is not str
        or type(extracted_root) is not str
        or archive_rows != extracted_rows
        or archive_root != extracted_root
        or archive_root
        != stable_hash(
            archive_rows, prefix="phase2b_formal_wire_covert_source_tree_v2_"
        )
        or extracted_root
        != stable_hash(
            extracted_rows, prefix="phase2b_formal_wire_covert_source_tree_v2_"
        )
    ):
        raise ValueError("source archive/extracted full-tree paths drifted")


def _managed_atomic_target_is_publisher_reachable_v2(
    relative_parent: str,
    target_name: str,
    target_sha256: str,
) -> bool:
    if relative_parent == "":
        return target_name == "reservation.json"
    if relative_parent == "evidence":
        return target_name in {
            "failure-child-stderr-prefix.bin",
            "failure-child-stderr.bin",
            "failure-child-stdout-prefix.bin",
            "failure-child-stdout.bin",
            "inner.json",
            "launch-intent.json",
            "mechanics.json",
            "outer-supervisor.json",
        }
    if relative_parent == "evidence/inputs":
        return target_name in {
            "attempt.json",
            "profile.json",
            "runtime-environment.json",
        }
    if relative_parent == "evidence/cas":
        return (
            re.fullmatch(r"[0-9a-f]{64}", target_name) is not None
            and target_name == target_sha256
        )
    if relative_parent == "evidence/rfc8785-raw-bundle":
        return (
            target_name == "receipt.json"
            or (
                re.fullmatch(r"[0-9a-f]{64}", target_name) is not None
                and target_name == target_sha256
            )
            or re.fullmatch(
                r"command-[0-9]{3}-[a-z0-9-]+\."
                r"(?:argv\.json|stdout\.bin|stderr\.bin)",
                target_name,
            )
            is not None
        )
    return False


def _managed_atomic_residue_paths_v2(
    value: object,
) -> dict[str, dict[str, object]]:
    if type(value) is not list or len(value) > 16:
        raise ValueError("managed atomic residue list is outside its bound")
    paths: dict[str, dict[str, object]] = {}
    for index, row in enumerate(value):
        if type(row) is not dict or set(row) != {
            "device",
            "inode",
            "link_count",
            "relative_parent",
            "same_inode",
            "size_bytes",
            "state",
            "target_name",
            "target_sha256",
            "temporary_name",
        }:
            raise ValueError(f"managed atomic residue[{index}] is not closed")
        relative_parent = row["relative_parent"]
        target_name = row["target_name"]
        temporary_name = row["temporary_name"]
        if (
            type(relative_parent) is not str
            or relative_parent.startswith("/")
            or "\\" in relative_parent
            or "\x00" in relative_parent
            or any(
                component in {"", ".", ".."}
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}", component)
                is None
                for component in relative_parent.split("/")
                if relative_parent
            )
            or type(target_name) is not str
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}", target_name)
            is None
            or type(row["target_sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", row["target_sha256"]) is None
            or type(temporary_name) is not str
            or re.fullmatch(
                r"\." + re.escape(target_name) + r"\.tmp-[0-9]+-[0-9]+",
                temporary_name,
            )
            is None
            or row["state"]
            not in {
                "STAGED_TEMP_ONLY",
                "TARGET_AND_MANAGED_TEMP_HARDLINK",
            }
            or row["same_inode"]
            is not (row["state"] == "TARGET_AND_MANAGED_TEMP_HARDLINK")
            or type(row["device"]) is not str
            or re.fullmatch(r"(?:0|[1-9][0-9]{0,19})", row["device"])
            is None
            or type(row["inode"]) is not str
            or re.fullmatch(r"[1-9][0-9]{0,19}", row["inode"])
            is None
            or _require_exact_int_v2(
                row["link_count"],
                f"managed atomic residue[{index}] link count",
                minimum=(
                    2
                    if row["state"] == "TARGET_AND_MANAGED_TEMP_HARDLINK"
                    else 1
                ),
            )
            < (
                2
                if row["state"] == "TARGET_AND_MANAGED_TEMP_HARDLINK"
                else 1
            )
            or _require_exact_int_v2(
                row["size_bytes"],
                f"managed atomic residue[{index}] size",
                maximum=_MAXIMUM_RETAINED_ATTEMPT_TREE_BYTES_V2,
            )
            < 0
        ):
            raise ValueError(f"managed atomic residue[{index}] identity drifted")
        if not _managed_atomic_target_is_publisher_reachable_v2(
            relative_parent, target_name, row["target_sha256"]
        ):
            raise ValueError(
                f"managed atomic residue[{index}] target is not publisher-reachable"
            )
        _require_raw_sha256_v2(
            row["target_sha256"], f"managed atomic residue[{index}] target"
        )
        target_relative = (
            target_name
            if not relative_parent
            else relative_parent + "/" + target_name
        )
        temporary_relative = (
            temporary_name
            if not relative_parent
            else relative_parent + "/" + temporary_name
        )
        if temporary_relative in paths or target_relative == temporary_relative:
            raise ValueError("managed atomic residue paths are not unique")
        paths[temporary_relative] = row
    return paths


def _validate_failure_cas_projection_shape_v2(
    value: object,
) -> dict[str, object]:
    """Validate the exact self-addressed CAS snapshot of one /failure/4."""

    expected = {
        "cas_directory_present",
        "object_count",
        "objects_in_lexicographic_digest_order",
        "objects_root",
        "partial_cas_roles_attributed",
        "schema_version",
        "total_regular_bytes",
    }
    if (
        type(value) is not dict
        or set(value) != expected
        or value.get("schema_version") != _FAILURE_CAS_PROJECTION_SCHEMA_V2
        or type(value.get("cas_directory_present")) is not bool
        or value.get("partial_cas_roles_attributed") is not False
        or type(value.get("objects_in_lexicographic_digest_order")) is not list
    ):
        raise ValueError("failure CAS projection header drifted")
    rows = value["objects_in_lexicographic_digest_order"]
    if len(rows) > _MAXIMUM_RETAINED_ATTEMPT_TREE_ROWS_V2:
        raise ValueError("failure CAS projection object count exceeds its cap")
    digests: list[str] = []
    total = 0
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != {
            "content_sha256",
            "object_size_bytes",
        }:
            raise ValueError(f"failure CAS object[{index}] is not closed")
        digest = _require_raw_sha256_v2(
            row["content_sha256"], f"failure CAS object[{index}]"
        )
        size = _require_exact_int_v2(
            row["object_size_bytes"],
            f"failure CAS object[{index}] size",
            minimum=1,
            maximum=_MAXIMUM_RETAINED_ATTEMPT_TREE_BYTES_V2,
        )
        digests.append(digest)
        total += size
        if total > _MAXIMUM_RETAINED_ATTEMPT_TREE_BYTES_V2:
            raise ValueError("failure CAS aggregate bytes exceed the retained cap")
    if (
        digests != sorted(set(digests))
        or _require_exact_int_v2(
            value["object_count"],
            "failure CAS object count",
            maximum=_MAXIMUM_RETAINED_ATTEMPT_TREE_ROWS_V2,
        )
        != len(rows)
        or _require_exact_int_v2(
            value["total_regular_bytes"],
            "failure CAS total bytes",
            maximum=_MAXIMUM_RETAINED_ATTEMPT_TREE_BYTES_V2,
        )
        != total
        or (value["cas_directory_present"] is False and rows != [])
        or value["objects_root"]
        != stable_hash(
            rows, prefix="phase2b_formal_wire_covert_failure_cas_v1_"
        )
    ):
        raise ValueError("failure CAS projection aggregate/root drifted")
    return value


def _failure_cas_projection_at_attempt_fd_v2(
    attempt_fd: int,
    *,
    managed_atomic_residues: object,
    deadline_ns: int | None,
) -> dict[str, object]:
    """Snapshot every formal CAS object, excluding only proven managed temps."""

    residue_paths = _managed_atomic_residue_paths_v2(managed_atomic_residues)
    allowed_cas_temporary_names = {
        relative.removeprefix("evidence/cas/")
        for relative in residue_paths
        if relative.startswith("evidence/cas/")
        and "/" not in relative.removeprefix("evidence/cas/")
    }
    attempt_names = set(_directory_names_at_fd_v2(attempt_fd))
    cas_fd: int | None = None
    evidence_fd: int | None = None
    if "evidence" not in attempt_names:
        if allowed_cas_temporary_names:
            raise ValueError("failure CAS residue exists without evidence directory")
        rows: list[dict[str, object]] = []
        core = {
            "schema_version": _FAILURE_CAS_PROJECTION_SCHEMA_V2,
            "cas_directory_present": False,
            "object_count": 0,
            "objects_in_lexicographic_digest_order": rows,
            "partial_cas_roles_attributed": False,
            "total_regular_bytes": 0,
        }
        return _validate_failure_cas_projection_shape_v2(
            {
                **core,
                "objects_root": stable_hash(
                    rows, prefix="phase2b_formal_wire_covert_failure_cas_v1_"
                ),
            }
        )
    try:
        evidence_fd = _open_directory_at_fd_v2(attempt_fd, "evidence")
        evidence_names = set(_directory_names_at_fd_v2(evidence_fd))
        if "cas" not in evidence_names:
            if allowed_cas_temporary_names:
                raise ValueError("failure CAS residue exists without CAS directory")
            rows = []
            core = {
                "schema_version": _FAILURE_CAS_PROJECTION_SCHEMA_V2,
                "cas_directory_present": False,
                "object_count": 0,
                "objects_in_lexicographic_digest_order": rows,
                "partial_cas_roles_attributed": False,
                "total_regular_bytes": 0,
            }
            return _validate_failure_cas_projection_shape_v2(
                {
                    **core,
                    "objects_root": stable_hash(
                        rows,
                        prefix="phase2b_formal_wire_covert_failure_cas_v1_",
                    ),
                }
            )
        cas_fd = _open_directory_at_fd_v2(evidence_fd, "cas")
        cas_names = set(_directory_names_at_fd_v2(cas_fd))
        if not allowed_cas_temporary_names.issubset(cas_names):
            raise ValueError("declared failure CAS managed temp is absent")
        object_names = cas_names - allowed_cas_temporary_names
        if any(re.fullmatch(r"[0-9a-f]{64}", name) is None for name in object_names):
            raise ValueError("failure CAS contains an unknown non-object name")
        rows = []
        total = 0
        for name in sorted(object_names):
            digest, opened = _hash_regular_at_fd_v2(
                cas_fd,
                name,
                maximum_bytes=_MAXIMUM_RETAINED_ATTEMPT_TREE_BYTES_V2,
                deadline_ns=deadline_ns,
            )
            if digest != name or opened.st_size < 1:
                raise ValueError("failure CAS object is not content-addressed")
            total += opened.st_size
            if total > _MAXIMUM_RETAINED_ATTEMPT_TREE_BYTES_V2:
                raise ValueError("failure CAS aggregate bytes exceed the retained cap")
            rows.append(
                {
                    "content_sha256": name,
                    "object_size_bytes": opened.st_size,
                }
            )
        core = {
            "schema_version": _FAILURE_CAS_PROJECTION_SCHEMA_V2,
            "cas_directory_present": True,
            "object_count": len(rows),
            "objects_in_lexicographic_digest_order": rows,
            "partial_cas_roles_attributed": False,
            "total_regular_bytes": total,
        }
        return _validate_failure_cas_projection_shape_v2(
            {
                **core,
                "objects_root": stable_hash(
                    rows, prefix="phase2b_formal_wire_covert_failure_cas_v1_"
                ),
            }
        )
    finally:
        _close_owned_descriptors_v2(
            (cas_fd, evidence_fd), name="failure CAS snapshot"
        )


def _derive_failure_v4_retained_projection_at_fd_v2(
    attempt_fd: int,
    *,
    managed_atomic_residues: object,
    deadline_ns: int | None,
) -> dict[str, dict[str, object]]:
    """Derive the /failure/4 tree, CAS, and residue evidence in one walk.

    This function is deliberately invoked independently by the producer and
    public verifier.  Within either trust boundary it hashes each regular inode
    at most once, including hard-linked managed atomic residues.
    """

    if (
        deadline_ns is not None
        and (
            type(deadline_ns) is not int
            or type(deadline_ns) is bool
            or time.monotonic_ns() >= deadline_ns
        )
    ):
        raise TimeoutError(
            "failure retained projection lacks a future absolute deadline"
        )
    if not stat.S_ISDIR(os.fstat(attempt_fd).st_mode):
        raise ValueError("failure retained projection root is not a directory")
    residue_paths = _managed_atomic_residue_paths_v2(managed_atomic_residues)
    target_paths: dict[str, dict[str, object]] = {}
    for temporary_relative, row in residue_paths.items():
        relative_parent = str(row["relative_parent"])
        target_relative = (
            str(row["target_name"])
            if not relative_parent
            else relative_parent + "/" + str(row["target_name"])
        )
        if target_relative in target_paths or target_relative == temporary_relative:
            raise ValueError("managed atomic residue targets are not unique")
        target_paths[target_relative] = row

    rows: list[dict[str, object]] = []
    observed_regular: dict[str, tuple[os.stat_result, str]] = {}
    digest_by_inode: dict[
        tuple[int, int], tuple[int, int, int, int, str]
    ] = {}
    total_regular_bytes = 0
    cas_directory_present = False
    cas_rows: list[dict[str, object]] = []
    walk_fd = os.dup(attempt_fd)
    try:
        for directory, directory_names, file_names, directory_fd in os.fwalk(
            ".", topdown=True, follow_symlinks=False, dir_fd=walk_fd
        ):
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                raise TimeoutError(
                    "failure retained projection reached its absolute deadline"
                )
            directory_names.sort()
            file_names.sort()
            relative_directory = "" if directory == "." else directory[2:]
            for name in tuple(directory_names):
                if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                    raise TimeoutError(
                        "failure retained projection reached its absolute deadline"
                    )
                child_stat = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
                if not stat.S_ISDIR(child_stat.st_mode):
                    raise ValueError(
                        "failure retained projection rejects directory symlinks"
                    )
                relative = (
                    name
                    if not relative_directory
                    else relative_directory + "/" + name
                )
                if relative.startswith("evidence/cas/"):
                    raise ValueError("failure CAS contains a nested directory")
                if relative == "evidence/cas":
                    cas_directory_present = True
                rows.append({"path": relative, "type": "directory"})
                if len(rows) > _MAXIMUM_RETAINED_ATTEMPT_TREE_ROWS_V2:
                    raise ValueError(
                        "failure retained projection row count exceeds the cap"
                    )
            for name in file_names:
                if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                    raise TimeoutError(
                        "failure retained projection reached its absolute deadline"
                    )
                relative = (
                    name
                    if not relative_directory
                    else relative_directory + "/" + name
                )
                if name.endswith((".pyc", ".pyo")):
                    raise ValueError("failure retained projection rejects bytecode")
                preliminary = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
                if (
                    not stat.S_ISREG(preliminary.st_mode)
                    or not 0
                    <= preliminary.st_size
                    <= _MAXIMUM_RETAINED_ATTEMPT_TREE_BYTES_V2
                ):
                    raise ValueError(
                        "failure retained projection rejects a nonregular file"
                    )
                inode_key = (preliminary.st_dev, preliminary.st_ino)
                cached = digest_by_inode.get(inode_key)
                if cached is None:
                    digest, opened = _hash_regular_at_fd_v2(
                        directory_fd,
                        name,
                        maximum_bytes=_MAXIMUM_RETAINED_ATTEMPT_TREE_BYTES_V2,
                        deadline_ns=deadline_ns,
                    )
                    if (
                        preliminary.st_dev,
                        preliminary.st_ino,
                        preliminary.st_size,
                        preliminary.st_mtime_ns,
                        preliminary.st_ctime_ns,
                        preliminary.st_nlink,
                    ) != (
                        opened.st_dev,
                        opened.st_ino,
                        opened.st_size,
                        opened.st_mtime_ns,
                        opened.st_ctime_ns,
                        opened.st_nlink,
                    ):
                        raise ValueError(
                            "failure retained path changed before content replay"
                        )
                    digest_by_inode[inode_key] = (
                        opened.st_size,
                        opened.st_mtime_ns,
                        opened.st_ctime_ns,
                        opened.st_nlink,
                        digest,
                    )
                    opened_stat = opened
                else:
                    (
                        cached_size,
                        cached_mtime,
                        cached_ctime,
                        cached_nlink,
                        digest,
                    ) = cached
                    alias_fd = os.open(
                        name,
                        os.O_RDONLY
                        | os.O_CLOEXEC
                        | os.O_NONBLOCK
                        | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                    try:
                        opened_stat = os.fstat(alias_fd)
                    finally:
                        os.close(alias_fd)
                    if (
                        opened_stat.st_dev,
                        opened_stat.st_ino,
                        opened_stat.st_size,
                        opened_stat.st_mtime_ns,
                        opened_stat.st_ctime_ns,
                        opened_stat.st_nlink,
                    ) != (
                        inode_key[0],
                        inode_key[1],
                        cached_size,
                        cached_mtime,
                        cached_ctime,
                        cached_nlink,
                    ):
                        raise ValueError(
                            "failure retained hard-linked inode mutated during replay"
                        )
                observed_regular[relative] = (opened_stat, digest)
                rows.append(
                    {
                        "path": relative,
                        "type": "regular",
                        "size_bytes": opened_stat.st_size,
                        "sha256": digest,
                    }
                )
                total_regular_bytes += opened_stat.st_size
                if (
                    len(rows) > _MAXIMUM_RETAINED_ATTEMPT_TREE_ROWS_V2
                    or total_regular_bytes
                    > _MAXIMUM_RETAINED_ATTEMPT_TREE_BYTES_V2
                ):
                    raise ValueError(
                        "failure retained projection aggregate exceeds the cap"
                    )
                if relative.startswith("evidence/cas/"):
                    if relative.count("/") != 2:
                        raise ValueError("failure CAS object is not a direct child")
                    if relative in residue_paths:
                        continue
                    if re.fullmatch(r"[0-9a-f]{64}", name) is None:
                        raise ValueError(
                            "failure CAS contains an unknown non-object name"
                        )
                    if digest != name or opened_stat.st_size < 1:
                        raise ValueError(
                            "failure CAS object is not content-addressed"
                        )
                    cas_rows.append(
                        {
                            "content_sha256": name,
                            "object_size_bytes": opened_stat.st_size,
                        }
                    )
    finally:
        _close_owned_descriptors_v2(
            (walk_fd,), name="failure retained projection walk"
        )

    rows.sort(key=lambda row: (row["path"], row["type"]))
    row_paths = {str(row["path"]): row for row in rows}
    cas_path_row = row_paths.get("evidence/cas")
    if cas_path_row is not None and cas_path_row.get("type") != "directory":
        raise ValueError("failure CAS path is not a directory")
    if cas_directory_present is not (cas_path_row is not None):
        raise ValueError("failure CAS directory presence drifted")

    for temporary_relative, residue in residue_paths.items():
        observed_temp = observed_regular.get(temporary_relative)
        if observed_temp is None:
            raise ValueError("declared managed atomic residue is absent")
        temp_stat, _ = observed_temp
        expected_identity = (int(residue["device"]), int(residue["inode"]))
        if (
            (temp_stat.st_dev, temp_stat.st_ino) != expected_identity
            or temp_stat.st_nlink != residue["link_count"]
            or temp_stat.st_size != residue["size_bytes"]
        ):
            raise ValueError("managed atomic residue inode/linkage drifted")
        relative_parent = str(residue["relative_parent"])
        target_relative = (
            str(residue["target_name"])
            if not relative_parent
            else relative_parent + "/" + str(residue["target_name"])
        )
        target = observed_regular.get(target_relative)
        if residue["state"] == "STAGED_TEMP_ONLY":
            if target is not None or temp_stat.st_nlink != 1:
                raise ValueError("staged-only atomic target/linkage drifted")
            # A failed write may leave only a prefix of the intended target.
            # Its intended digest is bound by the residue row; its actual
            # partial bytes remain independently bound by the tree row.
        else:
            if target is None:
                raise ValueError("managed atomic hard-link target is absent")
            target_stat, target_digest = target
            if (
                (target_stat.st_dev, target_stat.st_ino) != expected_identity
                or target_stat.st_nlink != residue["link_count"]
                or target_stat.st_nlink < 2
                or target_stat.st_size != residue["size_bytes"]
                or target_digest != residue["target_sha256"]
            ):
                raise ValueError("managed atomic hard-link target drifted")

    terminal_rows = [row for row in rows if row.get("path") == "terminal.json"]
    filtered = [row for row in rows if row.get("path") != "terminal.json"]
    allowed_temporary_paths = set(residue_paths)
    if (
        len(terminal_rows) > 1
        or any(row.get("type") != "regular" for row in terminal_rows)
        or any(
            any(
                re.fullmatch(r"\..+\.tmp-[0-9]+-[0-9]+", part) is not None
                and str(row["path"]) not in allowed_temporary_paths
                for part in str(row["path"]).split("/")
            )
            for row in filtered
        )
        or not allowed_temporary_paths.issubset(row_paths)
    ):
        raise ValueError("retained failure tree contains terminal/unknown temp objects")
    top_level_names = sorted(
        {str(row["path"]).split("/", 1)[0] for row in filtered}
    )
    regular_rows = [row for row in filtered if row.get("type") == "regular"]
    retained_tree = {
        "regular_file_count": len(regular_rows),
        "root": stable_hash(
            filtered, prefix="phase2b_formal_wire_covert_failure_tree_v3_"
        ),
        "row_count": len(filtered),
        "top_level_names": top_level_names,
        "total_regular_bytes": sum(
            _require_exact_int_v2(
                row.get("size_bytes"), "failure retained regular file size"
            )
            for row in regular_rows
        ),
    }
    cas_rows.sort(key=lambda row: str(row["content_sha256"]))
    cas_core = {
        "schema_version": _FAILURE_CAS_PROJECTION_SCHEMA_V2,
        "cas_directory_present": cas_directory_present,
        "object_count": len(cas_rows),
        "objects_in_lexicographic_digest_order": cas_rows,
        "partial_cas_roles_attributed": False,
        "total_regular_bytes": sum(
            int(row["object_size_bytes"]) for row in cas_rows
        ),
    }
    failure_cas = _validate_failure_cas_projection_shape_v2(
        {
            **cas_core,
            "objects_root": stable_hash(
                cas_rows, prefix="phase2b_formal_wire_covert_failure_cas_v1_"
            ),
        }
    )
    if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
        raise TimeoutError(
            "failure retained projection reached its absolute deadline"
        )
    return {
        "failure_cas_projection": failure_cas,
        "retained_tree_before_terminal": retained_tree,
    }


def _open_relative_directory_at_fd_v2(root_fd: int, relative: str) -> int:
    if type(relative) is not str or relative.startswith("/") or "\\" in relative:
        raise ValueError("retained relative directory path is malformed")
    descriptor = os.dup(root_fd)
    if not relative:
        return descriptor
    try:
        for component in relative.split("/"):
            if (
                component in {"", ".", ".."}
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}", component)
                is None
            ):
                raise ValueError("retained relative directory left its grammar")
            child = _open_directory_at_fd_v2(descriptor, component)
            try:
                os.close(descriptor)
            except BaseException:
                _close_owned_descriptors_v2(
                    (child,), name="retained relative directory transfer"
                )
                raise
            descriptor = child
        return descriptor
    except BaseException:
        _close_owned_descriptors_v2(
            (descriptor,), name="retained relative directory acquisition"
        )
        raise


def _verify_managed_atomic_residues_at_fd_v2(
    attempt_fd: int,
    value: object,
    *,
    deadline_ns: int | None,
) -> set[str]:
    paths = _managed_atomic_residue_paths_v2(value)
    for temporary_relative, row in paths.items():
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("managed atomic residue replay reached its deadline")
        relative_parent = str(row["relative_parent"])
        parent_fd = _open_relative_directory_at_fd_v2(
            attempt_fd, relative_parent
        )
        try:
            temporary_stat = os.stat(
                str(row["temporary_name"]),
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            expected_identity = (int(row["device"]), int(row["inode"]))
            if not stat.S_ISREG(temporary_stat.st_mode) or (
                temporary_stat.st_dev,
                temporary_stat.st_ino,
                temporary_stat.st_nlink,
                temporary_stat.st_size,
            ) != (
                int(row["device"]),
                int(row["inode"]),
                row["link_count"],
                row["size_bytes"],
            ):
                raise ValueError("managed atomic residue inode/linkage drifted")
            hash_name = str(row["temporary_name"])
            if row["state"] == "TARGET_AND_MANAGED_TEMP_HARDLINK":
                target_stat = os.stat(
                    str(row["target_name"]),
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(target_stat.st_mode)
                    or (target_stat.st_dev, target_stat.st_ino)
                    != expected_identity
                    or target_stat.st_nlink != row["link_count"]
                    or target_stat.st_nlink < 2
                    or target_stat.st_size != row["size_bytes"]
                ):
                    raise ValueError("managed atomic hard-link target drifted")
                hash_name = str(row["target_name"])
            else:
                if row["link_count"] != 1:
                    raise ValueError("staged-only atomic residue has extra links")
                try:
                    os.stat(
                        str(row["target_name"]),
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    pass
                else:
                    raise ValueError("staged-only atomic target unexpectedly exists")
            digest, replay_stat = _hash_regular_at_fd_v2(
                parent_fd,
                hash_name,
                maximum_bytes=_MAXIMUM_RETAINED_ATTEMPT_TREE_BYTES_V2,
                deadline_ns=deadline_ns,
            )
            if (
                (
                    row["state"] == "TARGET_AND_MANAGED_TEMP_HARDLINK"
                    and digest != row["target_sha256"]
                )
                or replay_stat.st_size != row["size_bytes"]
                or (replay_stat.st_dev, replay_stat.st_ino) != expected_identity
            ):
                raise ValueError("managed atomic residue target bytes drifted")
        finally:
            _close_owned_descriptors_v2(
                (parent_fd,), name="managed atomic residue replay"
            )
    return set(paths)


def _retained_attempt_tree_before_terminal_v2(
    attempt_fd: int,
    *,
    deadline_ns: int | None = None,
    managed_atomic_residues: object = (),
) -> dict[str, object]:
    if managed_atomic_residues == ():
        managed_atomic_residues = []
    allowed_managed_temporary_paths = _verify_managed_atomic_residues_at_fd_v2(
        attempt_fd, managed_atomic_residues, deadline_ns=deadline_ns
    )
    rows, _ = _source_tree_manifest_at_fd_v2(
        attempt_fd, deadline_ns=deadline_ns
    )
    terminal_rows = [row for row in rows if row.get("path") == "terminal.json"]
    filtered = [row for row in rows if row.get("path") != "terminal.json"]
    if (
        len(terminal_rows) > 1
        or any(row.get("type") != "regular" for row in terminal_rows)
        or any(
            type(row.get("path")) is not str
            or any(
                re.fullmatch(r"\..+\.tmp-[0-9]+-[0-9]+", part) is not None
                and row["path"] not in allowed_managed_temporary_paths
                for part in row["path"].split("/")
            )
            for row in filtered
        )
        or not allowed_managed_temporary_paths.issubset(
            {str(row.get("path")) for row in filtered}
        )
    ):
        raise ValueError("retained failure tree contains terminal/unknown temp objects")
    top_level_names = sorted(
        {str(row["path"]).split("/", 1)[0] for row in filtered}
    )
    regular_rows = [row for row in filtered if row.get("type") == "regular"]
    return {
        "regular_file_count": len(regular_rows),
        "root": stable_hash(
            filtered,
            prefix="phase2b_formal_wire_covert_failure_tree_v3_",
        ),
        "row_count": len(filtered),
        "top_level_names": top_level_names,
        "total_regular_bytes": sum(
            _require_exact_int_v2(
                row.get("size_bytes"), "retained regular file size"
            )
            for row in regular_rows
        ),
    }


def _reconcile_nonterminal_atomic_uncertainty_v2(
    exc: BaseException,
    *,
    attempt_fd: int,
    deadline_ns: int,
    known_parent_directories: dict[str, int],
) -> list[dict[str, object]]:
    """Clean or explicitly bind one owned hard-link staging residue."""

    if (
        type(deadline_ns) is not int
        or type(deadline_ns) is bool
        or time.monotonic_ns() >= deadline_ns
    ):
        raise TimeoutError("atomic uncertainty recovery lacks a future deadline")
    if type(exc) is not _AtomicPublicationUncertainV2:
        return []
    if exc.target_name == "terminal.json":
        raise ValueError("terminal uncertainty cannot enter nonterminal recovery")
    if (
        exc.parent_identity is None
        or exc.target_identity is None
        or exc.temporary_name is None
    ):
        # Synthetic/internal callers may report a clean committed target without
        # residue metadata.  The retained-tree scanner still rejects any actual
        # unmanaged temporary name, so this is not a bypass.
        return []
    if (
        type(exc.parent_identity) is not tuple
        or len(exc.parent_identity) != 2
        or any(type(value) is not int or value < 0 for value in exc.parent_identity)
        or type(exc.target_identity) is not tuple
        or len(exc.target_identity) != 2
        or any(type(value) is not int or value < 0 for value in exc.target_identity)
        or type(exc.target_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", exc.target_sha256) is None
        or type(exc.target_name) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}", exc.target_name)
        is None
        or type(exc.temporary_name) is not str
        or re.fullmatch(
            r"\."
            + re.escape(exc.target_name)
            + r"\.tmp-[0-9]+-[0-9]+",
            exc.temporary_name,
        )
        is None
        or exc.publication_state
        not in {"STAGED_TEMP_ONLY", "TARGET_LINK_COMMITTED"}
    ):
        raise ValueError("atomic uncertainty metadata is malformed")
    if (
        type(known_parent_directories) is not dict
        or any(
            type(relative) is not str
            or type(descriptor) is not int
            or descriptor < 0
            for relative, descriptor in known_parent_directories.items()
        )
    ):
        raise ValueError("known retained parent directory map is malformed")
    matching = [
        (relative, descriptor)
        for relative, descriptor in known_parent_directories.items()
        if (
            (opened := os.fstat(descriptor)).st_dev,
            opened.st_ino,
        )
        == exc.parent_identity
    ]
    if len(matching) != 1:
        raise ValueError(
            "atomic uncertainty parent is outside the retained-fd registry"
        )
    relative_parent, retained_parent_fd = matching[0]
    parent_fd = _open_relative_directory_at_fd_v2(
        attempt_fd, relative_parent
    )
    try:
        retained_stat = os.fstat(retained_parent_fd)
        reopened_stat = os.fstat(parent_fd)
        if (
            (retained_stat.st_dev, retained_stat.st_ino)
            != (reopened_stat.st_dev, reopened_stat.st_ino)
            or (reopened_stat.st_dev, reopened_stat.st_ino)
            != exc.parent_identity
        ):
            raise ValueError(
                "atomic uncertainty retained parent path identity drifted"
            )
    except BaseException:
        _close_owned_descriptors_v2(
            (parent_fd,), name="atomic residue retained-parent identity"
        )
        raise
    try:
        target_stat: os.stat_result | None
        try:
            target_stat = os.stat(
                exc.target_name, dir_fd=parent_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            target_stat = None
        if exc.publication_state == "TARGET_LINK_COMMITTED":
            if (
                target_stat is None
                or not stat.S_ISREG(target_stat.st_mode)
                or (target_stat.st_dev, target_stat.st_ino)
                != exc.target_identity
            ):
                raise ValueError("atomic uncertainty target inode was replaced")
        elif target_stat is not None:
            raise ValueError("staged-only atomic target unexpectedly exists")
        try:
            temporary_stat = os.stat(
                exc.temporary_name, dir_fd=parent_fd, follow_symlinks=False
            )
        except FileNotFoundError:
            return []
        if (
            not stat.S_ISREG(temporary_stat.st_mode)
            or (temporary_stat.st_dev, temporary_stat.st_ino)
            != exc.target_identity
            or (
                target_stat is not None
                and temporary_stat.st_size != target_stat.st_size
            )
            or (
                exc.publication_state == "TARGET_LINK_COMMITTED"
                and (
                    target_stat is None
                    or target_stat.st_nlink < 2
                    or temporary_stat.st_nlink != target_stat.st_nlink
                )
            )
            or (
                exc.publication_state == "STAGED_TEMP_ONLY"
                and temporary_stat.st_nlink != 1
            )
        ):
            raise ValueError("atomic uncertainty temporary inode is not owned")
        if time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("atomic uncertainty recovery reached its deadline")
        try:
            os.remove(exc.temporary_name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except BaseException:
            pass
        else:
            try:
                os.stat(
                    exc.temporary_name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if exc.publication_state == "STAGED_TEMP_ONLY":
                    return []
                target_after = os.stat(
                    exc.target_name, dir_fd=parent_fd, follow_symlinks=False
                )
                if stat.S_ISREG(target_after.st_mode) and (
                    target_after.st_dev,
                    target_after.st_ino,
                ) == exc.target_identity:
                    return []
            raise ValueError("atomic uncertainty temporary cleanup did not close")
        # Cleanup can itself be unavailable (for example a persistent filesystem
        # error).  Bind the exact surviving hard-link pair into the negative
        # terminal instead of silently accepting a generic managed temp.
        temporary_stat = os.stat(
            exc.temporary_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if not stat.S_ISREG(temporary_stat.st_mode) or (
            temporary_stat.st_dev,
            temporary_stat.st_ino,
        ) != exc.target_identity:
            raise ValueError("atomic uncertainty residue changed during capture")
        state = "STAGED_TEMP_ONLY"
        same_inode = False
        if exc.publication_state == "TARGET_LINK_COMMITTED":
            target_stat = os.stat(
                exc.target_name, dir_fd=parent_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(target_stat.st_mode)
                or (target_stat.st_dev, target_stat.st_ino)
                != exc.target_identity
                or target_stat.st_nlink < 2
                or temporary_stat.st_nlink != target_stat.st_nlink
            ):
                raise ValueError("atomic hard-link residue changed during capture")
            state = "TARGET_AND_MANAGED_TEMP_HARDLINK"
            same_inode = True
        elif temporary_stat.st_nlink != 1:
            raise ValueError("staged-only atomic residue acquired another link")
        return [
            {
                "device": str(temporary_stat.st_dev),
                "inode": str(temporary_stat.st_ino),
                "link_count": temporary_stat.st_nlink,
                "relative_parent": relative_parent,
                "same_inode": same_inode,
                "size_bytes": temporary_stat.st_size,
                "state": state,
                "target_name": exc.target_name,
                "target_sha256": exc.target_sha256,
                "temporary_name": exc.temporary_name,
            }
        ]
    finally:
        _close_owned_descriptors_v2(
            (parent_fd,), name="nonterminal atomic uncertainty recovery"
        )


def _inspect_source_archive_regular_manifest_v2(
    archive: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    expected_source_commit_sha: str | None = None,
    deadline_ns: int | None = None,
    retained_directory_fd: int | None = None,
) -> tuple[
    str,
    int,
    int,
    int,
    int,
    list[dict[str, object]],
    str,
]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("source archive inspection requires O_NOFOLLOW")
    descriptor = os.open(
        archive.name if retained_directory_fd is not None else archive,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW,
        dir_fd=retained_directory_fd,
    )
    rows: list[dict[str, object]] = []
    seen_explicit_paths: set[str] = set()
    tree_types: dict[str, str] = {}
    total = 0
    directory_count = 0
    member_count = 0
    observed_pax_headers: dict[str, str] | None = None
    if (
        expected_source_commit_sha is not None
        and (
            type(expected_source_commit_sha) is not str
            or re.fullmatch(r"[0-9a-f]{40}", expected_source_commit_sha) is None
        )
    ):
        raise ValueError("source archive expected commit is not exact Git SHA-1")
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size != expected_size
            or (
                retained_directory_fd is None
                and os.path.realpath(f"/proc/self/fd/{descriptor}") != str(archive)
            )
        ):
            raise ValueError("source archive inspection descriptor drifted")
        if not 1 <= expected_size <= 512 * 1_024 * 1_024:
            raise ValueError("source archive inspection size is outside bounds")
        digest = hashlib.sha256()
        while True:
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("source archive inspection hash reached deadline")
            block = os.read(descriptor, 1_048_576)
            if not block:
                break
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("source archive inspection hash drifted")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as stream:
            with tarfile.open(fileobj=stream, mode="r|*") as bundle:
                for member in bundle:
                    if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                        raise TimeoutError("source archive inspection reached deadline")
                    member_count += 1
                    if member_count > 100_000:
                        raise ValueError("source archive inspection member cap exceeded")
                    if (
                        expected_source_commit_sha is not None
                        and member_count == 1
                        and dict(bundle.pax_headers)
                        != {"comment": expected_source_commit_sha}
                    ):
                        raise ValueError("source archive Git PAX commit binding drifted")
                    parts = Path(member.name).parts
                    canonical_path = "/".join(parts)
                    if (
                        not member.name
                        or member.name.startswith("/")
                        or "\\" in member.name
                        or any(part in {"", ".", ".."} for part in parts)
                        or len(member.name.encode("utf-8")) > 4_096
                        or canonical_path in seen_explicit_paths
                        or member.name.endswith((".pyc", ".pyo"))
                        or len(parts) > _MAXIMUM_SOURCE_ARCHIVE_PATH_DEPTH_V2
                        or member.name.rstrip("/") != canonical_path
                    ):
                        raise ValueError("source archive inspection rejects a path")
                    seen_explicit_paths.add(canonical_path)
                    for depth in range(1, len(parts)):
                        ancestor = "/".join(parts[:depth])
                        if tree_types.get(ancestor) == "regular":
                            raise ValueError(
                                "source archive path descends through a regular file"
                            )
                        tree_types.setdefault(ancestor, "directory")
                    if member.isdir():
                        if tree_types.get(canonical_path) == "regular":
                            raise ValueError(
                                "source archive directory conflicts with a regular file"
                            )
                        tree_types[canonical_path] = "directory"
                    elif member.isreg():
                        if canonical_path in tree_types:
                            raise ValueError(
                                "source archive regular file conflicts with a directory"
                            )
                        tree_types[canonical_path] = "regular"
                    if len(tree_types) > _MAXIMUM_SOURCE_ARCHIVE_TREE_ROWS_V2:
                        raise ValueError(
                            "source archive inspection implicit tree cap exceeded"
                        )
                    if member.isdir():
                        directory_count += 1
                        continue
                    if not member.isreg():
                        raise ValueError("source archive inspection rejects special files")
                    if not 0 <= member.size <= 64 * 1_024 * 1_024:
                        raise ValueError("source archive inspection member is too large")
                    if total + member.size > 512 * 1_024 * 1_024:
                        raise ValueError("source archive inspection expanded cap exceeded")
                    source_stream = bundle.extractfile(member)
                    if source_stream is None:
                        raise ValueError("source archive inspection lacks member data")
                    member_digest = hashlib.sha256()
                    copied = 0
                    try:
                        while True:
                            if (
                                deadline_ns is not None
                                and time.monotonic_ns() >= deadline_ns
                            ):
                                raise TimeoutError(
                                    "source archive member inspection reached deadline"
                                )
                            block = source_stream.read(65_536)
                            if not block:
                                break
                            copied += len(block)
                            if copied > member.size:
                                raise ValueError("source archive inspection member grew")
                            member_digest.update(block)
                    finally:
                        source_stream.close()
                    if copied != member.size:
                        raise ValueError("source archive inspection member size drifted")
                    total += copied
                    rows.append(
                        {
                            "path": canonical_path,
                            "type": "regular",
                            "size_bytes": copied,
                            "sha256": member_digest.hexdigest(),
                        }
                    )
                if member_count < 1:
                    raise ValueError("source archive inspection found no members")
                observed_pax_headers = dict(bundle.pax_headers)
        if expected_source_commit_sha is not None and observed_pax_headers != {
            "comment": expected_source_commit_sha
        }:
            raise ValueError("source archive Git PAX commit binding drifted")
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("source archive inspection observed mutation")
    finally:
        _close_owned_descriptors_v2(
            (descriptor,), name="source archive inspection"
        )
    rows.sort(key=lambda row: row["path"])
    regular_rows = {row["path"]: row for row in rows}
    tree_rows: list[dict[str, object]] = []
    for path, kind in tree_types.items():
        if kind == "directory":
            tree_rows.append({"path": path, "type": "directory"})
        else:
            tree_rows.append(regular_rows[path])
    tree_rows.sort(key=lambda row: (row["path"], row["type"]))
    tree_root = stable_hash(
        tree_rows, prefix="phase2b_formal_wire_covert_source_tree_v2_"
    )
    return (
        _regular_file_manifest_root_v2(rows),
        len(rows),
        total,
        member_count,
        directory_count,
        tree_rows,
        tree_root,
    )


def _open_or_create_directory_at_v2(parent_fd: int, name: str) -> int:
    if (
        type(name) is not str
        or name in {"", ".", ".."}
        or "/" in name
        or "\x00" in name
    ):
        raise ValueError("source archive directory component is unsafe")
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except FileExistsError:
        pass
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("source extraction requires Linux no-follow dirfd flags")
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode):
            raise ValueError("source archive directory component is not a directory")
        return descriptor
    except BaseException:
        _close_owned_descriptors_v2(
            (descriptor,), name="source archive directory acquisition"
        )
        raise


def _safe_extract_source_archive_v2(
    archive: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    expected_source_commit_sha: str | None = None,
    deadline_ns: int | None = None,
) -> dict[str, object]:
    if (
        destination.exists()
        or not destination.parent.is_dir()
        or destination.parent.is_symlink()
    ):
        raise ValueError("source archive extraction destination is not fresh")
    archive_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("source archive extraction requires O_NOFOLLOW")
    archive_flags |= os.O_NOFOLLOW
    archive_descriptor = os.open(archive, archive_flags)
    try:
        destination_parent_fd = _open_real_directory_fd_v2(destination.parent)
    except BaseException:
        _close_owned_descriptors_v2(
            (archive_descriptor,), name="source archive parent acquisition"
        )
        raise
    destination_fd: int | None = None
    member_paths: set[str] = set()
    tree_paths: set[str] = set()
    regular_files = 0
    directory_count = 0
    total_regular_bytes = 0
    if (
        expected_source_commit_sha is not None
        and (
            type(expected_source_commit_sha) is not str
            or re.fullmatch(r"[0-9a-f]{40}", expected_source_commit_sha) is None
        )
    ):
        raise ValueError("source extraction expected commit is not exact Git SHA-1")
    try:
        archive_before = os.fstat(archive_descriptor)
        if (
            not stat.S_ISREG(archive_before.st_mode)
            or archive_before.st_size != expected_size
            or os.path.realpath(f"/proc/self/fd/{archive_descriptor}")
            != str(archive)
        ):
            raise ValueError("source archive descriptor identity drifted")
        archive_digest = hashlib.sha256()
        while True:
            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                raise TimeoutError("source archive hash reached the absolute deadline")
            block = os.read(archive_descriptor, 1_048_576)
            if not block:
                break
            archive_digest.update(block)
        if archive_digest.hexdigest() != expected_sha256:
            raise ValueError("source archive bytes differ from the attempt manifest")
        os.lseek(archive_descriptor, 0, os.SEEK_SET)
        os.mkdir(destination.name, mode=0o700, dir_fd=destination_parent_fd)
        os.fsync(destination_parent_fd)
        destination_fd = _open_or_create_directory_at_v2(
            destination_parent_fd, destination.name
        )
        with os.fdopen(os.dup(archive_descriptor), "rb", closefd=True) as stream:
            with tarfile.open(fileobj=stream, mode="r|*") as bundle:
                member_count = 0
                for member in bundle:
                    if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                        raise TimeoutError("source extraction reached the absolute deadline")
                    member_count += 1
                    if member_count > 100_000:
                        raise ValueError("source archive member count exceeds the cap")
                    if (
                        expected_source_commit_sha is not None
                        and member_count == 1
                        and dict(bundle.pax_headers)
                        != {"comment": expected_source_commit_sha}
                    ):
                        raise ValueError("source extraction Git PAX commit binding drifted")
                    pure_parts = Path(member.name).parts
                    if (
                        not member.name
                        or member.name.startswith("/")
                        or "\\" in member.name
                        or any(part in {"", ".", ".."} for part in pure_parts)
                        or len(member.name.encode("utf-8")) > 4_096
                        or member.name in member_paths
                        or len(pure_parts) > _MAXIMUM_SOURCE_ARCHIVE_PATH_DEPTH_V2
                    ):
                        raise ValueError("source archive path is unsafe or duplicated")
                    member_paths.add(member.name)
                    for depth in range(1, len(pure_parts) + 1):
                        tree_paths.add("/".join(pure_parts[:depth]))
                    if len(tree_paths) > _MAXIMUM_SOURCE_ARCHIVE_TREE_ROWS_V2:
                        raise ValueError(
                            "source archive implicit tree rows exceed the cap"
                        )
                    if member.name.endswith((".pyc", ".pyo")):
                        raise ValueError("source archive contains forbidden bytecode")
                    if member.isdir():
                        current_fd = os.dup(destination_fd)
                        try:
                            for part in pure_parts:
                                next_fd = _open_or_create_directory_at_v2(
                                    current_fd, part
                                )
                                try:
                                    os.close(current_fd)
                                except BaseException:
                                    _close_owned_descriptors_v2(
                                        (next_fd,),
                                        name="source directory traversal transfer",
                                    )
                                    raise
                                current_fd = next_fd
                            os.fsync(current_fd)
                        finally:
                            _close_owned_descriptors_v2(
                                (current_fd,), name="source directory traversal"
                            )
                        directory_count += 1
                        continue
                    if not member.isreg():
                        raise ValueError("source archive rejects links and special files")
                    if not 0 <= member.size <= 64 * 1_024 * 1_024:
                        raise ValueError("source archive regular member is too large")
                    total_regular_bytes += member.size
                    if total_regular_bytes > 512 * 1_024 * 1_024:
                        raise ValueError("source archive expanded bytes exceed the cap")
                    source_stream = bundle.extractfile(member)
                    if source_stream is None:
                        raise ValueError("source archive regular member has no data")
                    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
                    flags |= os.O_NOFOLLOW
                    parent_fd = os.dup(destination_fd)
                    output_descriptor: int | None = None
                    try:
                        for part in pure_parts[:-1]:
                            next_fd = _open_or_create_directory_at_v2(parent_fd, part)
                            try:
                                os.close(parent_fd)
                            except BaseException:
                                _close_owned_descriptors_v2(
                                    (next_fd,),
                                    name="source member parent transfer",
                                )
                                raise
                            parent_fd = next_fd
                        output_descriptor = os.open(
                            pure_parts[-1], flags, 0o600, dir_fd=parent_fd
                        )
                    except BaseException:
                        try:
                            _close_owned_descriptors_v2(
                                (output_descriptor, parent_fd),
                                name="source member acquisition",
                            )
                        finally:
                            source_stream.close()
                        raise
                    if output_descriptor is None:
                        raise AssertionError("source member descriptor acquisition vanished")
                    copied = 0
                    try:
                        while True:
                            if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
                                raise TimeoutError(
                                    "source member extraction reached the absolute deadline"
                                )
                            block = source_stream.read(65_536)
                            if not block:
                                break
                            copied += len(block)
                            if copied > member.size:
                                raise ValueError("source archive member grew while read")
                            _write_all_fd_v2(
                                output_descriptor, block, deadline_ns=deadline_ns
                            )
                        os.fsync(output_descriptor)
                        written = os.fstat(output_descriptor)
                        if (
                            copied != member.size
                            or written.st_size != member.size
                            or not stat.S_ISREG(written.st_mode)
                            or written.st_nlink != 1
                        ):
                            raise ValueError("source archive member extraction drifted")
                        os.fsync(parent_fd)
                    finally:
                        try:
                            _close_owned_descriptors_v2(
                                (output_descriptor, parent_fd),
                                name="source member extraction",
                            )
                        finally:
                            source_stream.close()
                    regular_files += 1
                if member_count < 1:
                    raise ValueError("source archive contains no members")
                if expected_source_commit_sha is not None and dict(
                    bundle.pax_headers
                ) != {"comment": expected_source_commit_sha}:
                    raise ValueError("source extraction Git PAX commit binding drifted")
        archive_after = os.fstat(archive_descriptor)
        if (
            archive_before.st_dev,
            archive_before.st_ino,
            archive_before.st_size,
            archive_before.st_mtime_ns,
        ) != (
            archive_after.st_dev,
            archive_after.st_ino,
            archive_after.st_size,
            archive_after.st_mtime_ns,
        ):
            raise ValueError("source archive changed during extraction")
        if destination_fd is None:
            raise ValueError("source archive extraction root was not opened")
        os.fsync(destination_fd)
    finally:
        _close_owned_descriptors_v2(
            (destination_fd, destination_parent_fd, archive_descriptor),
            name="source archive extraction",
        )
    destination_fd = _open_real_directory_fd_v2(destination)
    try:
        os.fsync(destination_fd)
    finally:
        _close_owned_descriptors_v2(
            (destination_fd,), name="source extraction final fsync"
        )
    rows, tree_root = _source_tree_manifest_v2(
        destination, deadline_ns=deadline_ns
    )
    regular_file_root = _regular_file_manifest_root_v2(rows)
    return {
        "archive_format": "TAR_REGULAR_AND_DIRECTORY_ENTRIES_ONLY",
        "archive_sha256": expected_sha256,
        "archive_size_bytes": expected_size,
        "directory_count": directory_count,
        "forbidden_bytecode_count": 0,
        "member_count": len(member_paths),
        "regular_file_count": regular_files,
        "regular_file_manifest_root": regular_file_root,
        "source_tree_manifest_row_count": len(rows),
        "source_tree_manifest_root": tree_root,
        "symlink_hardlink_special_count": 0,
        "total_regular_bytes": total_regular_bytes,
    }


def verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
    attempt_directory: Path,
    *,
    expected_terminal_bytes: bytes | None = None,
    deadline_ns: int | None = None,
    retained_attempt_fd: int | None = None,
    retained_output_root_fd: int | None = None,
    retained_evidence_fd: int | None = None,
    retained_cas_fd: int | None = None,
    retained_inputs_fd: int | None = None,
    retained_source_tree_fd: int | None = None,
    retained_rfc_raw_bundle_fd: int | None = None,
    prospective_terminal: bool = False,
    _pair_replay_depth: int = 0,
) -> bytes:
    """Purely replay one retained attempt tree; issue no qualification authority.

    When ``retained_attempt_fd`` is supplied it is the authority root.  The
    display path is checked only inside evidence strings and is never reopened.
    A finalizer may validate prospective success bytes before the sole
    ``terminal.json`` link becomes visible by setting ``prospective_terminal``.
    """

    if (
        not isinstance(attempt_directory, Path)
        or not attempt_directory.is_absolute()
        or type(prospective_terminal) is not bool
        or (prospective_terminal and type(expected_terminal_bytes) is not bytes)
        or (retained_attempt_fd is not None and retained_output_root_fd is None)
        or type(_pair_replay_depth) is not int
        or type(_pair_replay_depth) is bool
        or not 0 <= _pair_replay_depth <= 1
    ):
        raise ValueError("attempt-directory verifier arguments are malformed")

    def check_deadline() -> None:
        if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("attempt-directory replay reached its absolute deadline")

    def same_directory(left: int, right: int, name: str) -> None:
        left_stat = os.fstat(left)
        right_stat = os.fstat(right)
        if (
            not stat.S_ISDIR(left_stat.st_mode)
            or not stat.S_ISDIR(right_stat.st_mode)
            or (left_stat.st_dev, left_stat.st_ino)
            != (right_stat.st_dev, right_stat.st_ino)
        ):
            raise ValueError(f"retained {name} directory identity drifted")

    owned_fds: list[int] = []
    owned_pair_tokens: list[_VerifiedCalibrationPairReplayTokenV2] = []
    attempt_fd = (
        _open_real_directory_fd_v2(attempt_directory)
        if retained_attempt_fd is None
        else os.dup(retained_attempt_fd)
    )
    owned_fds.append(attempt_fd)
    try:
        check_deadline()
        if re.fullmatch(r"attempt-[0-9a-f]{48}", attempt_directory.name) is None:
            raise ValueError("attempt directory does not use its content address")
        if retained_output_root_fd is not None:
            parent_entry_fd = _open_directory_at_fd_v2(
                retained_output_root_fd, attempt_directory.name
            )
            owned_fds.append(parent_entry_fd)
            same_directory(attempt_fd, parent_entry_fd, "attempt parent entry")
        attempt_names = set(_directory_names_at_fd_v2(attempt_fd))
        if prospective_terminal:
            if "terminal.json" in attempt_names:
                raise ValueError("prospective replay found an already-visible terminal")

        if expected_terminal_bytes is None:
            terminal_bytes, _ = _read_regular_at_fd_v2(
                attempt_fd,
                "terminal.json",
                maximum_bytes=8 * 1_024 * 1_024,
                deadline_ns=deadline_ns,
            )
        else:
            terminal_bytes = expected_terminal_bytes
            if not prospective_terminal:
                retained_terminal, _ = _read_regular_at_fd_v2(
                    attempt_fd,
                    "terminal.json",
                    maximum_bytes=8 * 1_024 * 1_024,
                    deadline_ns=deadline_ns,
                )
                if retained_terminal != terminal_bytes:
                    raise ValueError("retained terminal differs from expected bytes")
        terminal = _decode_canonical_transport_object_v2(
            terminal_bytes, "outer attempt terminal", maximum_bytes=8 * 1_024 * 1_024
        )
        if _pair_replay_depth == 1 and (
            terminal.get("schema_version") != OUTER_ATTEMPT_SUCCESS_SCHEMA_V2
            or terminal.get("entry_kind") != "development-calibration"
        ):
            raise ValueError(
                "depth-one calibration pair replay accepts only development success"
            )
        failure_schema_version = terminal.get("schema_version")
        if (
            failure_schema_version == _OUTER_ATTEMPT_FAILURE_SCHEMA_V3
            and prospective_terminal
        ):
            raise ValueError("legacy /failure/3 is retained replay-only")
        if failure_schema_version in {
            OUTER_ATTEMPT_FAILURE_SCHEMA_V2,
            _OUTER_ATTEMPT_FAILURE_SCHEMA_V3,
        }:
            failure_schema_is_v4 = (
                failure_schema_version == OUTER_ATTEMPT_FAILURE_SCHEMA_V2
            )
            required = {
                "attempt_id",
                "attempt_manifest_root",
                "campaign_id",
                "claims",
                "cleanup",
                "cleanup_verified",
                "disposition",
                "entry_kind",
                "execution_completion_not_claimed",
                "external_intervention_required",
                "launcher_started",
                "managed_atomic_residues",
                "profile_id",
                "published_accepted_audit_runs",
                "qualification_receipt_id",
                "reason_code",
                "retry_requires_new_attempt_id",
                "retained_tree_before_terminal",
                "run_roots",
                "schema_version",
                "terminal_publication_count",
                "timing",
            }
            if failure_schema_is_v4:
                required |= {
                    "child_pipe_capture",
                    "cleanup_failure_exception_type",
                    "failure_cas_projection",
                    "primary_failure_exception_type",
                    "primary_failure_stage",
                }
            else:
                required.add("failure_stage")
            expected_claims = {
                "r03_formal_trusted_wire_implementation_qualified": False,
                "r04_formal_covert_auditor_implementation_qualified": False,
                "r05_synthetic_dev_wire_covert_qualification_verified": False,
                "actual_corpus_audit": False,
                "actual_unsealed_960_generation": False,
                "origin_authenticated": False,
                "key_independence_verified": False,
                "campaign_execution_authorized": False,
            }
            tree_projection = terminal.get("retained_tree_before_terminal")
            managed_atomic_residues = terminal.get("managed_atomic_residues")
            _managed_atomic_residue_paths_v2(managed_atomic_residues)
            failure_cas_projection: dict[str, object] | None = None
            if failure_schema_is_v4:
                failure_cas_projection = _validate_failure_cas_projection_shape_v2(
                    terminal.get("failure_cas_projection")
                )
                live_failure_projection = (
                    _derive_failure_v4_retained_projection_at_fd_v2(
                        attempt_fd,
                        managed_atomic_residues=managed_atomic_residues,
                        deadline_ns=deadline_ns,
                    )
                )
                if not _exact_json_equal(
                    failure_cas_projection,
                    live_failure_projection["failure_cas_projection"],
                ):
                    raise ValueError("failure CAS retained snapshot drifted")
                live_tree_projection = live_failure_projection[
                    "retained_tree_before_terminal"
                ]
            else:
                live_tree_projection = _retained_attempt_tree_before_terminal_v2(
                    attempt_fd,
                    deadline_ns=deadline_ns,
                    managed_atomic_residues=managed_atomic_residues,
                )
            failure_root = terminal.get("attempt_manifest_root")
            if (
                type(failure_root) is not str
                or re.fullmatch(
                    r"phase2b_formal_wire_covert_attempt_manifest_v3_[0-9a-f]{64}",
                    failure_root,
                )
                is None
            ):
                raise ValueError("failure attempt manifest root is malformed")
            expected_failure_disposition = (
                "ATOMIC_FAILURE_CLEANUP_UNPROVEN_EXTERNAL_INTERVENTION_REQUIRED"
                if terminal.get("cleanup_verified") is False
                else (
                    "ATOMIC_FAILURE_PIPE_CAPTURE_PERSISTENCE_UNVERIFIED"
                    if failure_schema_is_v4
                    and type(terminal.get("child_pipe_capture")) is dict
                    and terminal["child_pipe_capture"].get(
                        "capture_persistence_verified"
                    )
                    is False
                    else "ATOMIC_EMPTY_REJECTION"
                )
            )
            if (
                set(terminal) != required
                or terminal["execution_completion_not_claimed"] is not True
                or _require_exact_int_v2(
                    terminal["published_accepted_audit_runs"],
                    "failure published accepted runs",
                )
                != 0
                or terminal["qualification_receipt_id"] is not None
                or terminal["run_roots"] != []
                or _require_exact_int_v2(
                    terminal["terminal_publication_count"],
                    "failure terminal count",
                    minimum=1,
                    maximum=1,
                )
                != 1
                or type(terminal["cleanup_verified"]) is not bool
                or type(terminal["external_intervention_required"]) is not bool
                or type(terminal["launcher_started"]) is not bool
                or terminal["cleanup_verified"]
                is terminal["external_intervention_required"]
                or terminal["disposition"] != expected_failure_disposition
                or terminal["entry_kind"]
                not in {"development-calibration", "live-inner"}
                or type(terminal["campaign_id"]) is not str
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", terminal["campaign_id"])
                is None
                or type(terminal["attempt_id"]) is not str
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", terminal["attempt_id"])
                is None
                or terminal["profile_id"] != PROFILE_ID
                or type(terminal["reason_code"]) is not str
                or re.fullmatch(r"[A-Z0-9_]{1,96}", terminal["reason_code"])
                is None
                or type(
                    terminal[
                        "primary_failure_stage"
                        if failure_schema_is_v4
                        else "failure_stage"
                    ]
                )
                is not str
                or re.fullmatch(
                    r"[A-Z0-9_]{1,96}",
                    terminal[
                        "primary_failure_stage"
                        if failure_schema_is_v4
                        else "failure_stage"
                    ],
                )
                is None
                or (
                    failure_schema_is_v4
                    and (
                        type(terminal["primary_failure_exception_type"]) is not str
                        or re.fullmatch(
                            r"[A-Za-z_][A-Za-z0-9_]{0,127}",
                            terminal["primary_failure_exception_type"],
                        )
                        is None
                        or (
                            terminal["cleanup_verified"]
                            and terminal["cleanup_failure_exception_type"] is not None
                        )
                        or (
                            not terminal["cleanup_verified"]
                            and (
                                type(terminal["cleanup_failure_exception_type"])
                                is not str
                                or re.fullmatch(
                                    r"[A-Za-z_][A-Za-z0-9_]{0,127}",
                                    terminal["cleanup_failure_exception_type"],
                                )
                                is None
                            )
                        )
                    )
                )
                or terminal["retry_requires_new_attempt_id"] is not True
                or not _exact_json_equal(terminal["claims"], expected_claims)
                or not _exact_json_equal(tree_projection, live_tree_projection)
                or attempt_directory.name
                != "attempt-"
                + hashlib.sha256(failure_root.encode("ascii")).hexdigest()[:48]
            ):
                raise ValueError("outer failure terminal is not closed")
            timing = terminal["timing"]
            if type(timing) is not dict or set(timing) != {
                "elapsed_nanoseconds_at_finalization_ready",
                "finalization_ready_monotonic_nanoseconds",
                "started_monotonic_nanoseconds",
                "watchdog_deadline_monotonic_nanoseconds",
            }:
                raise ValueError("outer failure terminal timing is not closed")
            failure_start = _require_exact_int_v2(
                timing["started_monotonic_nanoseconds"], "failure start"
            )
            failure_ready = _require_exact_int_v2(
                timing["finalization_ready_monotonic_nanoseconds"],
                "failure finalization ready",
            )
            failure_elapsed = _require_exact_int_v2(
                timing["elapsed_nanoseconds_at_finalization_ready"],
                "failure finalization elapsed",
            )
            failure_watchdog = _require_exact_int_v2(
                timing["watchdog_deadline_monotonic_nanoseconds"],
                "failure watchdog deadline",
            )
            if (
                failure_ready < failure_start
                or failure_elapsed != failure_ready - failure_start
                or failure_watchdog != failure_start
                + LIVE_HARD_TOTAL_TIMEOUT_SECONDS * 1_000_000_000
                or failure_ready >= failure_watchdog
            ):
                raise ValueError("outer failure timing/deadline drifted")
            if "reservation.json" not in attempt_names:
                raise ValueError("outer failure lacks its exclusive reservation")
            reservation_bytes, _ = _read_regular_at_fd_v2(
                attempt_fd,
                "reservation.json",
                maximum_bytes=256 * 1_024,
                deadline_ns=deadline_ns,
            )
            expected_reservation = _canonical_inner_artifact_v2(
                {
                    "schema_version": (
                        "hegel-machine-phase2b-formal-wire-covert-attempt-reservation/2"
                    ),
                    "campaign_id": terminal["campaign_id"],
                    "attempt_id": terminal["attempt_id"],
                    "profile_id": terminal["profile_id"],
                    "attempt_manifest_root": terminal["attempt_manifest_root"],
                    "entry_kind": terminal["entry_kind"],
                    "reserved_exclusively": True,
                    "terminal_publication_count": 0,
                }
            )
            if reservation_bytes != expected_reservation:
                raise ValueError("outer failure reservation identity drifted")
            retained_attempt: Phase2BFormalWireCovertAttemptManifestV2 | None = None
            failure_launch_intent: dict[str, object] | None = None
            if "evidence" in attempt_names:
                failure_evidence_fd = _open_directory_at_fd_v2(
                    attempt_fd, "evidence"
                )
                owned_fds.append(failure_evidence_fd)
                evidence_names = set(
                    _directory_names_at_fd_v2(failure_evidence_fd)
                )
                if "inputs" in evidence_names:
                    failure_inputs_fd = _open_directory_at_fd_v2(
                        failure_evidence_fd, "inputs"
                    )
                    owned_fds.append(failure_inputs_fd)
                    input_names = set(_directory_names_at_fd_v2(failure_inputs_fd))
                    if "attempt.json" in input_names:
                        retained_attempt_bytes, _ = _read_regular_at_fd_v2(
                            failure_inputs_fd,
                            "attempt.json",
                            maximum_bytes=256 * 1_024,
                            deadline_ns=deadline_ns,
                        )
                        retained_attempt = (
                            load_phase2b_formal_wire_covert_attempt_manifest_v2(
                                retained_attempt_bytes
                            )
                        )
                        if (
                            retained_attempt.campaign_id != terminal["campaign_id"]
                            or retained_attempt.attempt_id != terminal["attempt_id"]
                            or retained_attempt.profile_id != terminal["profile_id"]
                            or retained_attempt.attempt_manifest_root
                            != terminal["attempt_manifest_root"]
                        ):
                            raise ValueError("failure retained attempt identity drifted")
                    if terminal["launcher_started"]:
                        launched_baseline_names = {
                            "cas",
                            "inputs",
                            "launch-intent.json",
                            "rfc8785-raw-bundle",
                            "source-tree",
                        }
                        if input_names != {
                            "attempt.json",
                            "profile.json",
                            "runtime-environment.json",
                        } or not launched_baseline_names.issubset(evidence_names):
                            raise ValueError("launched failure input set drifted")
                        if retained_attempt is None:
                            raise ValueError("launched failure lacks retained attempt")
                        for baseline_directory in (
                            "cas",
                            "rfc8785-raw-bundle",
                            "source-tree",
                        ):
                            baseline_fd = _open_directory_at_fd_v2(
                                failure_evidence_fd, baseline_directory
                            )
                            owned_fds.append(baseline_fd)
                        retained_profile_bytes, _ = _read_regular_at_fd_v2(
                            failure_inputs_fd,
                            "profile.json",
                            maximum_bytes=_MAXIMUM_PROFILE_BYTES,
                            deadline_ns=deadline_ns,
                        )
                        retained_runtime_bytes, _ = _read_regular_at_fd_v2(
                            failure_inputs_fd,
                            "runtime-environment.json",
                            maximum_bytes=1_048_576,
                            deadline_ns=deadline_ns,
                        )
                        if (
                            hashlib.sha256(retained_profile_bytes).hexdigest()
                            != retained_attempt.qualification_config_sha256
                            or len(retained_runtime_bytes)
                            != retained_attempt.runtime_environment_manifest_size_bytes
                            or hashlib.sha256(retained_runtime_bytes).hexdigest()
                            != retained_attempt.runtime_environment_manifest_root.removeprefix(
                                "sha256:"
                            )
                            or "launch-intent.json" not in evidence_names
                            or "source-tree" not in evidence_names
                        ):
                            raise ValueError("launched failure retained inputs drifted")
                        failure_launch_intent_bytes, _ = _read_regular_at_fd_v2(
                            failure_evidence_fd,
                            "launch-intent.json",
                            maximum_bytes=1_048_576,
                            deadline_ns=deadline_ns,
                        )
                        failure_launch_intent = _decode_canonical_transport_object_v2(
                            failure_launch_intent_bytes,
                            "failure launch intent",
                            maximum_bytes=1_048_576,
                        )
                        _validate_launch_intent_projection_v2(
                            failure_launch_intent,
                            attempt=retained_attempt,
                            entry_kind=terminal["entry_kind"],
                            source_tree=attempt_directory / "evidence/source-tree",
                        )
            cleanup = terminal["cleanup"]
            if failure_schema_is_v4:
                capture = terminal["child_pipe_capture"]
                if terminal["launcher_started"] is False:
                    if capture is not None:
                        raise ValueError("unlaunched /failure/4 retained pipe capture")
                else:
                    if "evidence" not in attempt_names:
                        raise ValueError("launched /failure/4 lacks retained evidence")
                    captured_stdout, captured_stderr, capture_complete = (
                        _verify_failure_child_pipe_capture_projection_v2(
                            failure_evidence_fd,
                            capture,
                            attempt_manifest_root=terminal["attempt_manifest_root"],
                            deadline_ns=deadline_ns,
                        )
                    )
                    if (
                        terminal["cleanup_verified"]
                        and type(capture) is dict
                        and capture.get("capture_persistence_verified") is True
                        and capture_complete is not True
                    ):
                        raise ValueError(
                            "verified cleanup lacks a complete child pipe capture"
                        )
                    if (
                        not terminal["cleanup_verified"]
                        and capture_complete is not False
                    ):
                        raise ValueError(
                            "unproven cleanup claims a complete child pipe capture"
                        )
                    if (
                        type(capture) is dict
                        and capture.get("capture_persistence_verified") is True
                    ):
                        if failure_cas_projection is None:
                            raise ValueError("verified pipe bundle lacks failure CAS")
                        bundle = capture.get("bundle_cas")
                        if type(bundle) is not dict:
                            raise ValueError("verified pipe bundle CAS row is absent")
                        bundle_digest = bundle.get("content_sha256")
                        bundle_size = bundle.get("object_size_bytes")
                        matching_rows = [
                            row
                            for row in failure_cas_projection[
                                "objects_in_lexicographic_digest_order"
                            ]
                            if row.get("content_sha256") == bundle_digest
                            and row.get("object_size_bytes") == bundle_size
                        ]
                        if len(matching_rows) != 1:
                            raise ValueError(
                                "failure pipe bundle is not cross-linked into CAS snapshot"
                            )
            if terminal["launcher_started"] is False:
                if cleanup is not None or terminal["cleanup_verified"] is not True:
                    raise ValueError("unlaunched failure has inconsistent cleanup")
            elif cleanup is None:
                raise ValueError("launched failure lacks cleanup evidence")
            if cleanup is not None:
                if retained_attempt is None or failure_launch_intent is None:
                    raise ValueError("launched failure cleanup lacks retained launch intent")
                _verify_failure_cleanup_projection_v2(
                    cleanup,
                    cleanup_verified=terminal["cleanup_verified"],
                    attempt=retained_attempt,
                    entry_kind=terminal["entry_kind"],
                    source_tree=attempt_directory / "evidence/source-tree",
                    launch_intent=failure_launch_intent,
                    failure_schema_version=failure_schema_version,
                )
                if failure_schema_is_v4:
                    if terminal["cleanup_failure_exception_type"] != (
                        None
                        if terminal["cleanup_verified"]
                        else cleanup["cleanup_error_type"]
                    ):
                        raise ValueError(
                            "failure pipe capture/cleanup projection linkage drifted"
                        )
                    if (
                        terminal["cleanup_verified"]
                        and capture.get("capture_persistence_verified") is True
                    ):
                        if (
                            type(captured_stdout) is not bytes
                            or type(captured_stderr) is not bytes
                            or type(capture) is not dict
                        ):
                            raise ValueError(
                                "verified cleanup lacks replayed child pipe bytes"
                            )
                        stdout_prefix_size = _require_exact_int_v2(
                            capture["stdout"]["prefix"]["size_bytes"],
                            "failure stdout prefix size linkage",
                            maximum=8 * 1_024 * 1_024,
                        )
                        stderr_prefix_size = _require_exact_int_v2(
                            capture["stderr"]["prefix"]["size_bytes"],
                            "failure stderr prefix size linkage",
                            maximum=256 * 1_024,
                        )
                        stdout_tail = captured_stdout[stdout_prefix_size:]
                        stderr_tail = captured_stderr[stderr_prefix_size:]
                        if (
                            _require_exact_int_v2(
                                cleanup["drained_stdout_size_bytes"],
                                "failure cleanup drained stdout size",
                                maximum=8 * 1_024 * 1_024,
                            )
                            != len(stdout_tail)
                            or cleanup["drained_stdout_sha256"]
                            != hashlib.sha256(stdout_tail).hexdigest()
                            or _require_exact_int_v2(
                                cleanup["drained_stderr_size_bytes"],
                                "failure cleanup drained stderr size",
                                maximum=256 * 1_024,
                            )
                            != len(stderr_tail)
                            or cleanup["drained_stderr_sha256"]
                            != hashlib.sha256(stderr_tail).hexdigest()
                        ):
                            raise ValueError(
                                "failure cleanup drained tail differs from pipe capture"
                            )
            return terminal_bytes
        if terminal.get("schema_version") != OUTER_ATTEMPT_SUCCESS_SCHEMA_V2:
            raise ValueError("outer attempt terminal schema is unknown")
        expected_attempt_names = {"evidence", "reservation.json"}
        if not prospective_terminal:
            expected_attempt_names.add("terminal.json")
        if attempt_names != expected_attempt_names:
            raise ValueError("success attempt directory exact object set drifted")

        evidence_fd = _open_directory_at_fd_v2(attempt_fd, "evidence")
        owned_fds.append(evidence_fd)
        if retained_evidence_fd is not None:
            same_directory(evidence_fd, retained_evidence_fd, "evidence")
        evidence_names = set(_directory_names_at_fd_v2(evidence_fd))
        if evidence_names != {
            "cas",
            "inner.json",
            "inputs",
            "launch-intent.json",
            "mechanics.json",
            "outer-supervisor.json",
            "rfc8785-raw-bundle",
            "source-tree",
        }:
            raise ValueError("success evidence exact object set drifted")
        cas_fd = _open_and_retain_directory_at_fd_v2(
            owned_fds, evidence_fd, "cas"
        )
        inputs_fd = _open_and_retain_directory_at_fd_v2(
            owned_fds, evidence_fd, "inputs"
        )
        source_tree_fd = _open_and_retain_directory_at_fd_v2(
            owned_fds, evidence_fd, "source-tree"
        )
        raw_fd = _open_and_retain_directory_at_fd_v2(
            owned_fds, evidence_fd, "rfc8785-raw-bundle"
        )
        for opened, retained, name in (
            (cas_fd, retained_cas_fd, "CAS"),
            (inputs_fd, retained_inputs_fd, "inputs"),
            (source_tree_fd, retained_source_tree_fd, "source tree"),
            (raw_fd, retained_rfc_raw_bundle_fd, "RFC raw bundle"),
        ):
            if retained is not None:
                same_directory(opened, retained, name)
        if set(_directory_names_at_fd_v2(inputs_fd)) != {
            "attempt.json",
            "profile.json",
            "runtime-environment.json",
        }:
            raise ValueError("success input exact object set drifted")

        profile_bytes, _ = _read_regular_at_fd_v2(
            inputs_fd,
            "profile.json",
            maximum_bytes=_MAXIMUM_PROFILE_BYTES,
            deadline_ns=deadline_ns,
        )
        attempt_bytes, _ = _read_regular_at_fd_v2(
            inputs_fd,
            "attempt.json",
            maximum_bytes=256 * 1_024,
            deadline_ns=deadline_ns,
        )
        retained_runtime_input_bytes, _ = _read_regular_at_fd_v2(
            inputs_fd,
            "runtime-environment.json",
            maximum_bytes=1_048_576,
            deadline_ns=deadline_ns,
        )
        launch_intent_bytes, _ = _read_regular_at_fd_v2(
            evidence_fd,
            "launch-intent.json",
            maximum_bytes=1_048_576,
            deadline_ns=deadline_ns,
        )
        inner_bytes, _ = _read_regular_at_fd_v2(
            evidence_fd,
            "inner.json",
            maximum_bytes=8 * 1_024 * 1_024,
            deadline_ns=deadline_ns,
        )
        outer_bytes, _ = _read_regular_at_fd_v2(
            evidence_fd,
            "outer-supervisor.json",
            maximum_bytes=1_048_576,
            deadline_ns=deadline_ns,
        )
        mechanics_bytes, _ = _read_regular_at_fd_v2(
            evidence_fd,
            "mechanics.json",
            maximum_bytes=8 * 1_024 * 1_024,
            deadline_ns=deadline_ns,
        )
        if _pair_replay_depth == 0:
            replay_profile_id = (
                load_phase2b_formal_wire_covert_qualification_profile_v2(
                    profile_bytes
                ).profile_id
            )
        else:
            frozen_shape = _EXPECTED_PROFILE["fixture_contract"][
                "rich_positive_shape"
            ]
            replay_profile_id = _load_retained_development_basis_profile_id_v2(
                profile_bytes,
                frozen_shape=frozen_shape,
                expected_entry_kind=terminal.get("entry_kind"),
            )
        attempt = load_phase2b_formal_wire_covert_attempt_manifest_v2(
            attempt_bytes, expected_profile_id=replay_profile_id
        )
        expected_attempt_name = "attempt-" + hashlib.sha256(
            attempt.attempt_manifest_root.encode("ascii")
        ).hexdigest()[:48]
        if (
            attempt_directory.name != expected_attempt_name
            or hashlib.sha256(profile_bytes).hexdigest()
            != attempt.qualification_config_sha256
        ):
            raise ValueError("attempt directory/profile content address drifted")
        expected_reservation = _canonical_inner_artifact_v2(
            {
                "schema_version": (
                    "hegel-machine-phase2b-formal-wire-covert-attempt-reservation/2"
                ),
                "campaign_id": attempt.campaign_id,
                "attempt_id": attempt.attempt_id,
                "profile_id": attempt.profile_id,
                "attempt_manifest_root": attempt.attempt_manifest_root,
                "entry_kind": terminal.get("entry_kind"),
                "reserved_exclusively": True,
                "terminal_publication_count": 0,
            }
        )
        reservation_bytes, _ = _read_regular_at_fd_v2(
            attempt_fd,
            "reservation.json",
            maximum_bytes=256 * 1_024,
            deadline_ns=deadline_ns,
        )
        if reservation_bytes != expected_reservation:
            raise ValueError("attempt reservation bytes drifted")

        claims = {
            "r03_formal_trusted_wire_implementation_qualified": False,
            "r04_formal_covert_auditor_implementation_qualified": False,
            "r05_synthetic_dev_wire_covert_qualification_verified": False,
            "actual_corpus_audit": False,
            "actual_unsealed_960_generation": False,
            "origin_authenticated": False,
            "key_independence_verified": False,
            "campaign_execution_authorized": False,
        }
        success_keys = {
            "artifacts",
            "attempt_id",
            "attempt_manifest_root",
            "attempt_private_cas_directory",
            "attempt_private_cas_store_namespace",
            "campaign_id",
            "claims",
            "content_sha256_is_not_a_signature",
            "disposition",
            "entry_kind",
            "profile_id",
            "projection_only_boundary",
            "retained_tree_before_terminal",
            "schema_version",
            "step5_signed_cas_authority_still_required",
            "terminal_publication_count",
            "timing",
        }
        if (
            set(terminal) != success_keys
            or terminal["disposition"] != "MECHANICS_VERIFIED_NOT_AUTHORITY"
            or terminal["campaign_id"] != attempt.campaign_id
            or terminal["attempt_id"] != attempt.attempt_id
            or terminal["profile_id"] != replay_profile_id
            or terminal["attempt_manifest_root"] != attempt.attempt_manifest_root
            or terminal["attempt_private_cas_directory"]
            != str(attempt_directory / "evidence/cas")
            or terminal["attempt_private_cas_store_namespace"]
            != attempt.attempt_manifest_root
            or _require_exact_int_v2(
                terminal["terminal_publication_count"],
                "success terminal publication count",
                minimum=1,
                maximum=1,
            )
            != 1
            or terminal["content_sha256_is_not_a_signature"] is not True
            or terminal["step5_signed_cas_authority_still_required"] is not True
            or not _exact_json_equal(
                terminal["projection_only_boundary"],
                _TASK4_PROJECTION_ONLY_BOUNDARY_V2,
            )
            or not _exact_json_equal(terminal["claims"], claims)
            or not _exact_json_equal(
                terminal["retained_tree_before_terminal"],
                _retained_attempt_tree_before_terminal_v2(
                    attempt_fd, deadline_ns=deadline_ns
                ),
            )
        ):
            raise ValueError("outer success terminal identity/claim drifted")
        entry_kind = terminal["entry_kind"]
        if entry_kind not in {"development-calibration", "live-inner"}:
            raise ValueError("outer success terminal entry kind drifted")
        timing = terminal["timing"]
        if type(timing) is not dict or set(timing) != {
            "elapsed_to_finalization_ready_nanoseconds",
            "finalization_ready_monotonic_nanoseconds",
            "service_ended_monotonic_nanoseconds",
            "started_monotonic_nanoseconds",
            "watchdog_deadline_monotonic_nanoseconds",
        }:
            raise ValueError("outer success terminal timing is not closed")
        start = _require_exact_int_v2(
            timing["started_monotonic_nanoseconds"], "terminal start"
        )
        ready = _require_exact_int_v2(
            timing["finalization_ready_monotonic_nanoseconds"],
            "terminal finalization ready",
        )
        watchdog = _require_exact_int_v2(
            timing["watchdog_deadline_monotonic_nanoseconds"],
            "terminal watchdog deadline",
        )
        terminal_elapsed = _require_exact_int_v2(
            timing["elapsed_to_finalization_ready_nanoseconds"],
            "terminal finalization elapsed",
        )
        if (
            watchdog != start + LIVE_HARD_TOTAL_TIMEOUT_SECONDS * 1_000_000_000
            or not start
            <= _require_exact_int_v2(
                timing["service_ended_monotonic_nanoseconds"],
                "terminal service end",
            )
            <= ready
            < watchdog
            or terminal_elapsed != ready - start
        ):
            raise ValueError("outer success terminal exceeded its hard deadline")

        artifacts = terminal["artifacts"]
        artifact_keys = {
            "attempt_manifest",
            "inner",
            "launch_intent",
            "mechanics",
            "outer_supervisor",
            "prior_calibration_attempt_pair",
            "prior_calibration_in_order",
            "profile",
            "rfc8785_prerequisite_receipt",
            "rfc8785_raw_bundle",
            "runtime_environment_manifest",
            "source_archive",
        }
        if type(artifacts) is not dict or set(artifacts) != artifact_keys:
            raise ValueError("outer success artifact set is not closed")

        cas_names: set[str] = set()

        def verify_cas(
            row: object,
            name: str,
            *,
            expected_payload: bytes | None = None,
            expected_digest: str | None = None,
            expected_size: int | None = None,
            read_bytes: bool = False,
        ) -> bytes | None:
            if type(row) is not dict:
                raise ValueError(f"{name} CAS projection is absent")
            digest = _require_raw_sha256_v2(row.get("content_sha256"), name)
            size = _require_exact_int_v2(
                row.get("object_size_bytes"), f"{name} size", minimum=1,
                maximum=512 * 1_024 * 1_024,
            )
            if expected_payload is not None:
                expected_digest = hashlib.sha256(expected_payload).hexdigest()
                expected_size = len(expected_payload)
            if expected_digest is not None and digest != expected_digest:
                raise ValueError(f"{name} CAS digest binding drifted")
            if expected_size is not None and size != expected_size:
                raise ValueError(f"{name} CAS size binding drifted")
            _verify_cas_store_projection_v2(
                row,
                expected_sha256=digest,
                expected_size=size,
                expected_store_namespace=attempt.attempt_manifest_root,
                name=name,
            )
            cas_names.add(digest)
            actual_digest, opened = _hash_regular_at_fd_v2(
                cas_fd,
                digest,
                maximum_bytes=512 * 1_024 * 1_024,
                deadline_ns=deadline_ns,
            )
            if actual_digest != digest or opened.st_size != size:
                raise ValueError(f"{name} retained CAS bytes drifted")
            if expected_payload is not None or read_bytes:
                payload, _ = _read_regular_at_fd_v2(
                    cas_fd,
                    digest,
                    maximum_bytes=max(1, size),
                    deadline_ns=deadline_ns,
                )
                if expected_payload is not None and payload != expected_payload:
                    raise ValueError(f"{name} retained CAS replay differs")
                return payload
            return None

        verify_cas(artifacts["profile"], "profile", expected_payload=profile_bytes)
        verify_cas(
            artifacts["attempt_manifest"],
            "attempt manifest",
            expected_payload=attempt_bytes,
        )
        verify_cas(
            artifacts["launch_intent"],
            "launch intent",
            expected_payload=launch_intent_bytes,
        )
        verify_cas(artifacts["inner"], "inner", expected_payload=inner_bytes)
        verify_cas(
            artifacts["outer_supervisor"],
            "outer supervisor",
            expected_payload=outer_bytes,
        )
        verify_cas(
            artifacts["mechanics"], "mechanics", expected_payload=mechanics_bytes
        )
        source_archive_digest = attempt.source_archive_root.removeprefix("sha256:")
        verify_cas(
            artifacts["source_archive"],
            "source archive",
            expected_digest=source_archive_digest,
            expected_size=attempt.source_archive_size_bytes,
        )
        (
            retained_archive_regular_root,
            retained_archive_regular_count,
            retained_archive_regular_bytes,
            retained_archive_member_count,
            retained_archive_directory_count,
            retained_archive_tree_rows,
            retained_archive_tree_root,
        ) = _inspect_source_archive_regular_manifest_v2(
            attempt_directory / "evidence/cas" / source_archive_digest,
            expected_sha256=source_archive_digest,
            expected_size=attempt.source_archive_size_bytes,
            expected_source_commit_sha=attempt.source_commit_sha,
            deadline_ns=deadline_ns,
            retained_directory_fd=cas_fd,
        )
        runtime_digest = attempt.runtime_environment_manifest_root.removeprefix(
            "sha256:"
        )
        runtime_bytes = verify_cas(
            artifacts["runtime_environment_manifest"],
            "runtime environment manifest",
            expected_digest=runtime_digest,
            expected_size=attempt.runtime_environment_manifest_size_bytes,
            read_bytes=True,
        )
        rfc_bytes = verify_cas(
            artifacts["rfc8785_prerequisite_receipt"],
            "RFC prerequisite receipt",
            expected_digest=attempt.rfc8785_prerequisite_receipt_sha256,
            expected_size=attempt.rfc8785_prerequisite_receipt_size_bytes,
            read_bytes=True,
        )
        if type(runtime_bytes) is not bytes or type(rfc_bytes) is not bytes:
            raise AssertionError("small retained CAS replay did not return bytes")
        if retained_runtime_input_bytes != runtime_bytes:
            raise ValueError("retained runtime input differs from its CAS bytes")
        runtime = _decode_canonical_transport_object_v2(
            runtime_bytes, "retained runtime manifest", maximum_bytes=1_048_576
        )
        if (
            runtime.get("schema_version") != RUNTIME_ENVIRONMENT_MANIFEST_SCHEMA_V2
            or hashlib.sha256(runtime_bytes).hexdigest() != runtime_digest
        ):
            raise ValueError("retained runtime manifest identity drifted")
        rfc_receipt = _verify_rfc8785_prerequisite_receipt_v2(
            rfc_bytes,
            attempt,
            verify_live_executables=False,
            deadline_ns=deadline_ns,
        )

        prior_rows = artifacts["prior_calibration_in_order"]
        if type(prior_rows) is not list or (
            entry_kind == "development-calibration" and prior_rows != []
        ) or (entry_kind == "live-inner" and len(prior_rows) != 6):
            raise ValueError("prior calibration CAS row count drifted")
        prior_bytes: list[bytes] = []
        for index, row in enumerate(prior_rows):
            payload = verify_cas(
                row, f"prior calibration {index}", read_bytes=True
            )
            if type(payload) is not bytes:
                raise AssertionError("prior calibration CAS replay did not return bytes")
            prior_bytes.append(payload)
        pair_replay_token: _VerifiedCalibrationPairReplayTokenV2 | None = None
        pair_projection = artifacts["prior_calibration_attempt_pair"]
        if entry_kind == "development-calibration":
            if pair_projection is not None:
                raise ValueError("development calibration cannot retain a prior pair")
        else:
            if _pair_replay_depth >= 1:
                raise ValueError("calibration pair replay depth exceeds one")
            if type(pair_projection) is not dict:
                raise ValueError("live attempt lacks its retained calibration pair")
            pair_directory_text = pair_projection.get("attempt_directory")
            if (
                type(pair_directory_text) is not str
                or not Path(pair_directory_text).is_absolute()
            ):
                raise ValueError("live calibration pair directory is malformed")
            (
                derived_prior_bytes,
                derived_pair_projection,
                pair_replay_token,
            ) = (
                _load_verified_development_calibration_pair_v2(
                    Path(pair_directory_text),
                    deadline_ns=deadline_ns,
                    pair_replay_depth=_pair_replay_depth + 1,
                    token_purpose="retained-attempt-replay",
                )
            )
            owned_pair_tokens.append(pair_replay_token)
            if (
                tuple(prior_bytes) != derived_prior_bytes
                or not _exact_json_equal(pair_projection, derived_pair_projection)
            ):
                raise ValueError("live retained calibration pair linkage drifted")
        if set(_directory_names_at_fd_v2(cas_fd)) != cas_names:
            raise ValueError("success CAS directory has missing or extra objects")

        outer = _decode_canonical_transport_object_v2(
            outer_bytes, "retained outer supervisor", maximum_bytes=1_048_576
        )
        inner = _decode_canonical_transport_object_v2(
            inner_bytes, "retained inner artifact", maximum_bytes=8 * 1_024 * 1_024
        )
        launch_intent = _decode_canonical_transport_object_v2(
            launch_intent_bytes,
            "retained launch intent",
            maximum_bytes=1_048_576,
        )
        _validate_launch_intent_projection_v2(
            launch_intent,
            attempt=attempt,
            entry_kind=entry_kind,
            source_tree=attempt_directory / "evidence/source-tree",
        )
        if not _exact_json_equal(
            launch_intent["prior_calibration_attempt_pair"], pair_projection
        ):
            raise ValueError("launch intent/calibration pair linkage drifted")
        if (
            outer.get("schema_version") != OUTER_SUPERVISOR_RESULT_SCHEMA_V2
            or outer.get("profile_id") != attempt.profile_id
            or outer.get("attempt_manifest_root") != attempt.attempt_manifest_root
            or outer.get("mode") != entry_kind
            or outer.get("source_archive_root") != attempt.source_archive_root
            or outer.get("runtime_environment_manifest_root")
            != attempt.runtime_environment_manifest_root
            or outer.get("source_tree_root")
            != str(attempt_directory / "evidence/source-tree")
            or outer.get("source_archive_object_path")
            != str(attempt_directory / "evidence/cas" / source_archive_digest)
            or outer.get("runtime_environment_manifest_object_path")
            != str(attempt_directory / "evidence/cas" / runtime_digest)
            or not _exact_json_equal(
                outer.get("source_archive_cas"), artifacts["source_archive"]
            )
            or not _exact_json_equal(
                outer.get("runtime_environment_manifest_cas"),
                artifacts["runtime_environment_manifest"],
            )
            or not _exact_json_equal(
                outer.get("rfc8785_receipt_cas"),
                artifacts["rfc8785_prerequisite_receipt"],
            )
            or not _exact_json_equal(outer.get("inner_artifact_cas"), artifacts["inner"])
            or outer.get("launcher", {}).get("argv")
            != launch_intent["child_argv"]
            or outer.get("launcher", {}).get("systemd_argv")
            != launch_intent["systemd_argv"]
            or outer.get("launcher", {}).get("python")
            != launch_intent["executables"]["python"]
            or outer.get("launcher", {}).get("systemd_run")
            != launch_intent["executables"]["systemd_run"]
            or outer.get("launcher", {}).get("systemctl")
            != launch_intent["executables"]["systemctl"]
            or outer.get("unit", {}).get("unit_name")
            != launch_intent["unit_name"]
            or outer.get("unit", {}).get("ownership_marker")
            != launch_intent["ownership_marker"]
            or outer.get("unit", {}).get("ownership_nonce")
            != launch_intent["ownership_nonce"]
            or outer.get("unit", {}).get("child_argv_sha256")
            != launch_intent["child_argv_sha256"]
            or outer.get("launcher", {}).get("systemctl_commands", [None])[0]
            != launch_intent["prelaunch_command"]
        ):
            raise ValueError("retained outer supervisor CAS/path binding drifted")
        expected_inner_schema = (
            "hegel-machine-phase2b-formal-wire-covert-development-calibration/6"
            if entry_kind == "development-calibration"
            else "hegel-machine-phase2b-formal-wire-covert-live-inner-candidate/6"
        )
        if (
            inner.get("schema_version") != expected_inner_schema
            or inner.get("attempt_manifest_root") != attempt.attempt_manifest_root
        ):
            raise ValueError("retained inner artifact identity drifted")

        live_files = _expected_live_file_sha256_v2(attempt)
        if not _exact_json_equal(outer.get("live_files_sha256"), live_files):
            raise ValueError("retained outer live-file manifest drifted")
        for relative, digest in live_files.items():
            actual, _ = _hash_relative_regular_at_fd_v2(
                source_tree_fd,
                relative,
                maximum_bytes=64 * 1_024 * 1_024,
                deadline_ns=deadline_ns,
            )
            if actual != digest:
                raise ValueError(f"retained source-tree file drifted: {relative}")
        tree_rows, tree_root = _source_tree_manifest_at_fd_v2(
            source_tree_fd, deadline_ns=deadline_ns
        )
        _verify_source_archive_full_tree_equivalence_v2(
            retained_archive_tree_rows,
            retained_archive_tree_root,
            tree_rows,
            tree_root,
        )
        extraction = outer.get("source_archive_extraction")
        if (
            type(extraction) is not dict
            or extraction.get("source_tree_manifest_root") != tree_root
            or extraction.get("source_tree_manifest_row_count") != len(tree_rows)
            or extraction.get("regular_file_manifest_root")
            != _regular_file_manifest_root_v2(tree_rows)
            or extraction.get("regular_file_manifest_root")
            != retained_archive_regular_root
            or _require_exact_int_v2(
                extraction.get("regular_file_count"),
                "retained archive regular count",
            )
            != retained_archive_regular_count
            or _require_exact_int_v2(
                extraction.get("total_regular_bytes"),
                "retained archive regular bytes",
            )
            != retained_archive_regular_bytes
            or _require_exact_int_v2(
                extraction.get("member_count"), "retained archive member count"
            )
            != retained_archive_member_count
            or _require_exact_int_v2(
                extraction.get("directory_count"),
                "retained archive directory count",
            )
            != retained_archive_directory_count
        ):
            raise ValueError("retained source-tree manifest projection drifted")

        raw_row = artifacts["rfc8785_raw_bundle"]
        if (
            type(raw_row) is not dict
            or set(raw_row) != {"directory", "projection"}
            or raw_row["directory"]
            != str(attempt_directory / "evidence/rfc8785-raw-bundle")
            or outer.get("rfc8785_raw_bundle_directory") != raw_row["directory"]
        ):
            raise ValueError("retained RFC raw bundle path binding drifted")
        raw = rfc_receipt["raw_execution"]
        evidence_body = {
            key: rfc_receipt[key] for key in sorted(rfc_receipt) if key != "cas"
        }
        evidence_sha = hashlib.sha256(
            canonicalize_rfc8785_v1(evidence_body)
        ).hexdigest()
        expected_raw_names = {
            row["path"] for row in raw["bundle_manifest"]
        } | {evidence_sha, "receipt.json"}
        if set(_directory_names_at_fd_v2(raw_fd)) != expected_raw_names:
            raise ValueError("retained RFC raw directory has missing or extra objects")
        retained_receipt, _ = _read_regular_at_fd_v2(
            raw_fd,
            "receipt.json",
            maximum_bytes=1_048_576,
            deadline_ns=deadline_ns,
        )
        if retained_receipt != rfc_bytes:
            raise ValueError("retained RFC raw receipt bytes drifted")
        raw_projection = _verify_rfc8785_raw_execution_bundle_v2(
            rfc_receipt,
            attempt_directory / "evidence/rfc8785-raw-bundle",
            expected_qualification_source_sha256=attempt.qualification_source_sha256,
            replay_project_root=attempt_directory / "evidence/source-tree",
            fresh_execute=False,
            deadline_ns=deadline_ns,
            retained_bundle_fd=raw_fd,
            replay_project_root_fd=source_tree_fd,
        )
        if (
            not _exact_json_equal(raw_projection, raw_row["projection"])
            or not _exact_json_equal(
                outer.get("rfc8785_raw_bundle_projection"), raw_projection
            )
        ):
            raise ValueError("retained RFC raw bundle projection drifted")

        if entry_kind == "development-calibration":
            _validate_development_calibration_artifact_v2(inner, attempt)
        else:
            _validate_live_inner_artifact_v2(inner, attempt)
        _validate_outer_supervisor_semantics_pure_v2(
            outer,
            attempt=attempt,
            inner=inner,
            inner_bytes=inner_bytes,
            rfc_receipt=rfc_receipt,
            rfc_receipt_bytes=rfc_bytes,
            runtime_manifest_bytes=runtime_bytes,
            source_tree_rows=tree_rows,
            source_archive_tree_rows=retained_archive_tree_rows,
            raw_bundle_projection=raw_projection,
            expected_entry=entry_kind,
            retained_source_tree_fd=source_tree_fd,
        )
        outer_timing = outer["timing"]
        if (
            type(outer_timing) is not dict
            or _require_exact_int_v2(
                outer_timing.get("started_monotonic_nanoseconds"),
                "cross-bound outer start",
            )
            != start
            or _require_exact_int_v2(
                outer_timing.get("service_ended_monotonic_nanoseconds"),
                "cross-bound outer service end",
            )
            != _require_exact_int_v2(
                timing["service_ended_monotonic_nanoseconds"],
                "cross-bound terminal service end",
            )
            or _require_exact_int_v2(
                outer_timing.get("watchdog_deadline_monotonic_nanoseconds"),
                "cross-bound outer watchdog",
            )
            != watchdog
            or _require_exact_int_v2(
                outer_timing.get("supervisor_ready_monotonic_nanoseconds"),
                "cross-bound outer supervisor ready",
            )
            > ready
        ):
            raise ValueError("outer/terminal monotonic timing binding drifted")

        expected_mechanics = _canonical_inner_artifact_v2(
            {
                "schema_version": MECHANICS_VERIFIER_SCHEMA_V2,
                "disposition": "MECHANICS_VERIFIED_NOT_AUTHORITY",
                "entry_kind": entry_kind,
                "profile_id": replay_profile_id,
                "attempt_manifest_root": attempt.attempt_manifest_root,
                "inner_artifact_sha256": hashlib.sha256(inner_bytes).hexdigest(),
                "outer_supervisor_result_sha256": hashlib.sha256(
                    outer_bytes
                ).hexdigest(),
                "rfc8785_prerequisite_receipt_sha256": hashlib.sha256(
                    rfc_bytes
                ).hexdigest(),
                "claims": claims,
                "authority_issuer_present": False,
                "calibration_artifact_cannot_be_promoted": (
                    entry_kind == "development-calibration"
                ),
                "projection_only_boundary": dict(
                    _TASK4_PROJECTION_ONLY_BOUNDARY_V2
                ),
                "content_sha256_is_not_a_signature": True,
                "step5_signed_cas_authority_still_required": True,
            }
        )
        if mechanics_bytes != expected_mechanics:
            raise ValueError("retained mechanics artifact bytes drifted")
        if entry_kind == "live-inner":
            shape = _EXPECTED_PROFILE["fixture_contract"]["rich_positive_shape"]
            if type(shape) is not dict:
                raise ValueError("live retained replay lacks its frozen shape")
            _validate_frozen_calibration_bytes_v2(
                shape,
                calibration_basis_profile_bytes=prior_bytes[0],
                calibration_attempt_manifest_bytes=prior_bytes[1],
                calibration_inner_artifact_bytes=prior_bytes[2],
                calibration_outer_supervisor_result_bytes=prior_bytes[3],
                calibration_rfc8785_prerequisite_receipt_bytes=prior_bytes[4],
                calibration_mechanics_artifact_bytes=prior_bytes[5],
                _pair_replay_token=pair_replay_token,
                _pair_projection=pair_projection,
                _pair_replay_purpose="retained-attempt-replay",
                deadline_ns=deadline_ns,
            )
        check_deadline()
        return terminal_bytes
    finally:
        for token in owned_pair_tokens:
            _discard_verified_calibration_pair_replay_token_v2(token)
        _close_owned_descriptors_v2(
            reversed(owned_fds), name="retained attempt directory replay"
        )


def _load_verified_development_calibration_pair_v2(
    calibration_attempt_directory: Path,
    *,
    deadline_ns: int | None,
    pair_replay_depth: int,
    token_purpose: str,
) -> tuple[
    tuple[bytes, bytes, bytes, bytes, bytes, bytes],
    dict[str, object],
    _VerifiedCalibrationPairReplayTokenV2,
]:
    """Derive the six frozen bytes from one retained development attempt."""

    if (
        not isinstance(calibration_attempt_directory, Path)
        or not calibration_attempt_directory.is_absolute()
        or type(pair_replay_depth) is not int
        or type(pair_replay_depth) is bool
        or pair_replay_depth != 1
        or token_purpose not in {"outer-mechanics", "retained-attempt-replay"}
    ):
        raise ValueError("calibration pair replay arguments are not depth-one paths")
    parent_fd = _open_real_directory_fd_v2(calibration_attempt_directory.parent)
    owned: list[int] = [parent_fd]
    try:
        attempt_fd = _open_directory_at_fd_v2(
            parent_fd,
            calibration_attempt_directory.name,
            expected_path=calibration_attempt_directory,
        )
        owned.append(attempt_fd)
        terminal_bytes = verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
            calibration_attempt_directory,
            deadline_ns=deadline_ns,
            retained_attempt_fd=attempt_fd,
            retained_output_root_fd=parent_fd,
            _pair_replay_depth=pair_replay_depth,
        )
        terminal = _decode_canonical_transport_object_v2(
            terminal_bytes,
            "paired calibration terminal",
            maximum_bytes=8 * 1_024 * 1_024,
        )
        if (
            terminal.get("schema_version") != OUTER_ATTEMPT_SUCCESS_SCHEMA_V2
            or terminal.get("entry_kind") != "development-calibration"
            or terminal.get("disposition") != "MECHANICS_VERIFIED_NOT_AUTHORITY"
        ):
            raise ValueError("paired attempt is not a development calibration success")
        evidence_fd = _open_and_retain_directory_at_fd_v2(
            owned, attempt_fd, "evidence"
        )
        inputs_fd = _open_and_retain_directory_at_fd_v2(
            owned, evidence_fd, "inputs"
        )
        cas_fd = _open_and_retain_directory_at_fd_v2(
            owned, evidence_fd, "cas"
        )
        profile_bytes, _ = _read_regular_at_fd_v2(
            inputs_fd,
            "profile.json",
            maximum_bytes=_MAXIMUM_PROFILE_BYTES,
            deadline_ns=deadline_ns,
        )
        attempt_bytes, _ = _read_regular_at_fd_v2(
            inputs_fd,
            "attempt.json",
            maximum_bytes=256 * 1_024,
            deadline_ns=deadline_ns,
        )
        inner_bytes, _ = _read_regular_at_fd_v2(
            evidence_fd,
            "inner.json",
            maximum_bytes=8 * 1_024 * 1_024,
            deadline_ns=deadline_ns,
        )
        outer_bytes, _ = _read_regular_at_fd_v2(
            evidence_fd,
            "outer-supervisor.json",
            maximum_bytes=1_048_576,
            deadline_ns=deadline_ns,
        )
        mechanics_bytes, _ = _read_regular_at_fd_v2(
            evidence_fd,
            "mechanics.json",
            maximum_bytes=8 * 1_024 * 1_024,
            deadline_ns=deadline_ns,
        )
        artifacts = terminal["artifacts"]
        if type(artifacts) is not dict:
            raise ValueError("paired calibration terminal artifacts are absent")
        rfc_row = artifacts.get("rfc8785_prerequisite_receipt")
        if type(rfc_row) is not dict:
            raise ValueError("paired calibration RFC CAS reference is absent")
        rfc_digest = _require_raw_sha256_v2(
            rfc_row.get("content_sha256"), "paired calibration RFC"
        )
        rfc_size = _require_exact_int_v2(
            rfc_row.get("object_size_bytes"),
            "paired calibration RFC size",
            minimum=1,
            maximum=1_048_576,
        )
        rfc_bytes, rfc_stat = _read_regular_at_fd_v2(
            cas_fd,
            rfc_digest,
            maximum_bytes=rfc_size,
            deadline_ns=deadline_ns,
        )
        if (
            rfc_stat.st_size != rfc_size
            or hashlib.sha256(rfc_bytes).hexdigest() != rfc_digest
        ):
            raise ValueError("paired calibration RFC retained bytes drifted")
        tree_projection = _retained_attempt_tree_before_terminal_v2(
            attempt_fd, deadline_ns=deadline_ns
        )
        if not _exact_json_equal(
            tree_projection, terminal.get("retained_tree_before_terminal")
        ):
            raise ValueError("paired calibration terminal/tree projection drifted")
        projection = {
            "schema_version": (
                "hegel-machine-phase2b-formal-wire-covert-calibration-attempt-pair/1"
            ),
            "attempt_directory": str(calibration_attempt_directory),
            "attempt_manifest_root": terminal["attempt_manifest_root"],
            "terminal_sha256": hashlib.sha256(terminal_bytes).hexdigest(),
            "terminal_size_bytes": len(terminal_bytes),
            "retained_tree_before_terminal": tree_projection,
            "pair_retention_required": True,
            "pair_must_be_jointly_signed_by_step5": True,
        }
        _validate_calibration_pair_projection_shape_v2(projection, required=True)
        payloads = (
            profile_bytes,
            attempt_bytes,
            inner_bytes,
            outer_bytes,
            rfc_bytes,
            mechanics_bytes,
        )
        return (
            payloads,
            projection,
            _issue_verified_calibration_pair_replay_token_v2(
                _CALIBRATION_PAIR_REPLAY_TOKEN_ISSUER_V2,
                payloads,
                projection,
                token_purpose,
            ),
        )
    finally:
        _close_owned_descriptors_v2(
            reversed(owned), name="retained calibration pair replay"
        )


def _run_phase2b_formal_wire_covert_outer_attempt_impl_v2(
    *,
    entry_kind: str,
    profile_path: Path,
    attempt_manifest_path: Path,
    rfc8785_prerequisite_receipt_path: Path,
    source_archive_path: Path,
    runtime_environment_manifest_path: Path,
    output_root: Path,
    calibration_attempt_directory_path: Path | None = None,
    calibration_basis_profile_path: Path | None = None,
    calibration_attempt_manifest_path: Path | None = None,
    calibration_inner_artifact_path: Path | None = None,
    calibration_outer_supervisor_result_path: Path | None = None,
    calibration_rfc8785_prerequisite_receipt_path: Path | None = None,
    calibration_mechanics_artifact_path: Path | None = None,
    _retained_descriptors: list[int],
    _retained_pair_tokens: list[_VerifiedCalibrationPairReplayTokenV2],
) -> bytes:
    """Run one reserved transient-service attempt and atomically publish evidence."""

    if not (
        0
        < _OUTER_TERMINAL_PUBLICATION_RESERVE_SECONDS_V2
        < _OUTER_FAILURE_CLEANUP_RESERVE_SECONDS_V2
        < LIVE_HARD_TOTAL_TIMEOUT_SECONDS
    ):
        raise RuntimeError("outer cleanup/finalization reserves are not nested")
    started_ns = time.monotonic_ns()
    hard_deadline_ns = (
        started_ns + LIVE_HARD_TOTAL_TIMEOUT_SECONDS * 1_000_000_000
    )
    deadline_ns = hard_deadline_ns - (
        _OUTER_FAILURE_CLEANUP_RESERVE_SECONDS_V2 * 1_000_000_000
    )
    if entry_kind not in {"development-calibration", "live-inner"}:
        raise ValueError("outer runner entry kind is not closed")
    for name, path in (
        ("profile", profile_path),
        ("attempt manifest", attempt_manifest_path),
        ("RFC-8785 prerequisite", rfc8785_prerequisite_receipt_path),
        ("source archive", source_archive_path),
        ("runtime environment manifest", runtime_environment_manifest_path),
        ("output root", output_root),
    ):
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError(f"outer runner {name} path must be exact absolute Path")
    profile_bytes = _read_bounded_regular_file_v2(
        profile_path,
        maximum_bytes=_MAXIMUM_PROFILE_BYTES,
        name="runner profile",
        deadline_ns=deadline_ns,
    )
    profile = load_phase2b_formal_wire_covert_qualification_profile_v2(profile_bytes)
    attempt_bytes = _read_bounded_regular_file_v2(
        attempt_manifest_path,
        maximum_bytes=256 * 1_024,
        name="runner attempt manifest",
        deadline_ns=deadline_ns,
    )
    attempt = load_phase2b_formal_wire_covert_attempt_manifest_v2(attempt_bytes)
    if attempt.profile_id != PROFILE_ID:
        raise ValueError("outer runner current attempt profile drifted")
    if _SOURCE_EXECUTION_BOOTSTRAP_ACTIVE_V2:
        executed = _EXECUTED_MAIN_SOURCE_BOOTSTRAP_V2
        if (
            type(executed) is not dict
            or executed.get("bootstrap_kind")
            != "EXTERNAL_PYTHON_C_OPEN_ONCE_V2"
            or executed.get("sha256") != attempt.qualification_source_sha256
        ):
            raise ValueError(
                "outer runner executed source differs before attempt reservation"
            )
    if (
        hashlib.sha256(profile_bytes).hexdigest()
        != attempt.qualification_config_sha256
        or attempt.qualification_lock_sha256 != DEPENDENCY_LOCK_SHA256
        or attempt.dependency_lock_sha256 != DEPENDENCY_LOCK_SHA256
    ):
        raise ValueError("outer runner profile/config/dependency binding drifted")
    if output_root.is_symlink() or not output_root.is_dir():
        raise ValueError("outer runner output root must preexist as a real directory")
    output_root = output_root.resolve(strict=True)
    retained_descriptors = _retained_descriptors
    if retained_descriptors:
        raise ValueError("outer descriptor registry must start empty")

    def retain_descriptor(descriptor: int) -> int:
        if descriptor in retained_descriptors:
            raise RuntimeError("outer runner attempted to retain one fd twice")
        retained_descriptors.append(descriptor)
        return descriptor

    def close_retained_descriptors() -> None:
        failed: list[tuple[int, BaseException]] = []
        while retained_descriptors:
            descriptor = retained_descriptors.pop()
            try:
                os.close(descriptor)
            except BaseException as exc:
                failed.append((descriptor, exc))
        retained_descriptors.extend(descriptor for descriptor, _ in failed)
        if failed:
            raise RuntimeError(
                f"outer runner could not close {len(failed)} retained descriptors"
            ) from failed[0][1]

    output_root_fd: int | None = None
    attempt_fd: int | None = None
    evidence_fd: int | None = None
    cas_fd: int | None = None
    inputs_fd: int | None = None
    source_tree_fd: int | None = None
    rfc_retained_bundle_fd: int | None = None

    def known_attempt_parent_directories() -> dict[str, int]:
        return {
            relative: descriptor
            for relative, descriptor in (
                ("", attempt_fd),
                ("evidence", evidence_fd),
                ("evidence/cas", cas_fd),
                ("evidence/inputs", inputs_fd),
                ("evidence/source-tree", source_tree_fd),
                ("evidence/rfc8785-raw-bundle", rfc_retained_bundle_fd),
            )
            if descriptor is not None
        }
    attempt_directory_name = "attempt-" + hashlib.sha256(
        attempt.attempt_manifest_root.encode("ascii")
    ).hexdigest()[:48]
    attempt_directory = output_root / attempt_directory_name
    stage = "RESERVATION"
    launcher: subprocess.Popen[bytes] | None = None
    systemctl_realpath: str | None = None
    unit_name: str | None = None
    unit_ownership_marker: str | None = None
    owned_unit_invocation_id: str | None = None
    prelaunch_absent_verified = False
    systemctl_commands: list[dict[str, object]] = []
    launch_intent_bytes: bytes | None = None
    cleanup_start_command_index: int | None = None
    terminal_committed = False
    managed_atomic_residues: list[dict[str, object]] = []
    service_output = {"stdout": bytearray(), "stderr": bytearray()}
    service_capture_state = {
        "capture_complete": False,
        "capture_truncated": False,
    }
    child_source_bootstrap_preopened_snapshots: dict[str, dict[str, object]] | None = None
    child_source_bootstrap_preopened_snapshots_root: str | None = None
    reservation = _canonical_inner_artifact_v2(
        {
            "schema_version": (
                "hegel-machine-phase2b-formal-wire-covert-attempt-reservation/2"
            ),
            "campaign_id": attempt.campaign_id,
            "attempt_id": attempt.attempt_id,
            "profile_id": attempt.profile_id,
            "attempt_manifest_root": attempt.attempt_manifest_root,
            "entry_kind": entry_kind,
            "reserved_exclusively": True,
            "terminal_publication_count": 0,
        }
    )
    try:
        output_root_fd = retain_descriptor(_open_real_directory_fd_v2(output_root))
        attempt_fd = retain_descriptor(
            _mkdir_new_directory_at_fd_v2(
                output_root_fd,
                attempt_directory_name,
                expected_path=attempt_directory,
                deadline_ns=deadline_ns,
            )
        )
        _atomic_write_new_file_at_fd_v2(
            attempt_fd, "reservation.json", reservation, deadline_ns=deadline_ns
        )
        evidence_directory = attempt_directory / "evidence"
        evidence_fd = retain_descriptor(
            _mkdir_new_directory_at_fd_v2(
                attempt_fd,
                "evidence",
                expected_path=evidence_directory,
                deadline_ns=deadline_ns,
            )
        )
        cas_directory = evidence_directory / "cas"
        cas_fd = retain_descriptor(
            _mkdir_new_directory_at_fd_v2(
                evidence_fd,
                "cas",
                expected_path=cas_directory,
                deadline_ns=deadline_ns,
            )
        )
        inputs_directory = evidence_directory / "inputs"
        inputs_fd = retain_descriptor(
            _mkdir_new_directory_at_fd_v2(
                evidence_fd,
                "inputs",
                expected_path=inputs_directory,
                deadline_ns=deadline_ns,
            )
        )
        _atomic_write_new_file_at_fd_v2(
            inputs_fd, "profile.json", profile_bytes, deadline_ns=deadline_ns
        )
        staged_profile_path = inputs_directory / "profile.json"
        _atomic_write_new_file_at_fd_v2(
            inputs_fd, "attempt.json", attempt_bytes, deadline_ns=deadline_ns
        )
        staged_attempt_path = inputs_directory / "attempt.json"
    except BaseException as exc:
        if type(exc) is _DirectoryCreationCommittedV2:
            if exc.descriptor is not None:
                committed_fd = retain_descriptor(exc.descriptor)
                exc.descriptor = None
                if attempt_fd is None and exc.name == attempt_directory_name:
                    attempt_fd = committed_fd
            elif (
                attempt_fd is None
                and exc.name == attempt_directory_name
                and output_root_fd is not None
            ):
                try:
                    attempt_fd = retain_descriptor(
                        _open_directory_at_fd_v2(
                            output_root_fd,
                            attempt_directory_name,
                            expected_path=attempt_directory,
                        )
                    )
                except BaseException as reopen_exc:
                    try:
                        os.rmdir(attempt_directory_name, dir_fd=output_root_fd)
                        os.fsync(output_root_fd)
                    except BaseException as rollback_exc:
                        raise _AtomicPublicationUncertainV2(
                            "reservation.json",
                            "attempt mkdir committed but retained reopen/rollback failed",
                        ) from rollback_exc
                    close_retained_descriptors()
                    raise RuntimeError(
                        "attempt mkdir committed without reservation and was rolled back"
                    ) from reopen_exc
        if attempt_fd is None:
            close_retained_descriptors()
            raise RuntimeError(
                "attempt reservation committed without a recoverable retained fd"
            ) from exc
        if _atomic_uncertainty_blocks_another_terminal_v2(exc):
            close_retained_descriptors()
            raise
        managed_atomic_residues = _reconcile_nonterminal_atomic_uncertainty_v2(
            exc,
            attempt_fd=attempt_fd,
            deadline_ns=hard_deadline_ns,
            known_parent_directories=known_attempt_parent_directories(),
        )
        if "reservation.json" not in _directory_names_at_fd_v2(attempt_fd):
            if (
                output_root_fd is None
                or _directory_names_at_fd_v2(attempt_fd)
            ):
                raise _AtomicPublicationUncertainV2(
                    "reservation.json",
                    "reservation failed before publication but attempt tree is not empty",
                ) from exc
            retained_attempt_stat = os.fstat(attempt_fd)
            live_attempt_stat = os.stat(
                attempt_directory_name,
                dir_fd=output_root_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISDIR(retained_attempt_stat.st_mode)
                or (retained_attempt_stat.st_dev, retained_attempt_stat.st_ino)
                != (live_attempt_stat.st_dev, live_attempt_stat.st_ino)
            ):
                raise _AtomicPublicationUncertainV2(
                    "reservation.json",
                    "unreserved attempt directory identity drifted",
                ) from exc
            os.rmdir(attempt_directory_name, dir_fd=output_root_fd)
            os.fsync(output_root_fd)
            close_retained_descriptors()
            raise RuntimeError(
                "attempt reservation did not commit; empty directory rolled back"
            ) from exc
        failure_projection = _derive_failure_v4_retained_projection_at_fd_v2(
            attempt_fd,
            managed_atomic_residues=managed_atomic_residues,
            deadline_ns=hard_deadline_ns,
        )
        failure_cas_projection = failure_projection["failure_cas_projection"]
        retained_tree = failure_projection["retained_tree_before_terminal"]
        failure_bytes = _closed_outer_failure_artifact_v2(
            attempt,
            entry_kind=entry_kind,
            stage=stage,
            primary_failure_exception_type=type(exc).__name__,
            reason_code=FAIL_ISOLATED_EXECUTION,
            cleanup=None,
            cleanup_failure_exception_type=None,
            child_pipe_capture=None,
            failure_cas_projection=failure_cas_projection,
            cleanup_verified=True,
            external_intervention_required=False,
            launcher_started=False,
            managed_atomic_residues=managed_atomic_residues,
            retained_tree_before_terminal=retained_tree,
            started_ns=started_ns,
            deadline_ns=hard_deadline_ns,
        )
        verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
            attempt_directory,
            expected_terminal_bytes=failure_bytes,
            deadline_ns=hard_deadline_ns,
            retained_attempt_fd=attempt_fd,
            retained_output_root_fd=output_root_fd,
            prospective_terminal=True,
        )
        failure_bytes = _retime_verified_outer_terminal_v2(
            failure_bytes,
            prospective_replay_completed_ns=time.monotonic_ns(),
            prospective_deadline_ns=hard_deadline_ns,
        )
        _publish_terminal_once_at_fd_v2(
            attempt_fd, failure_bytes, deadline_ns=hard_deadline_ns
        )
        terminal_committed = True
        _verify_parent_identity_after_terminal_commit_v2(
            output_root_fd,
            attempt_directory_name,
            attempt_fd,
            name="reservation-failure attempt",
            deadline_ns=hard_deadline_ns,
        )
        close_retained_descriptors()
        return failure_bytes
    stage = "PREFLIGHT"
    try:
        outer_runner_tcb_entry = _outer_runner_tcb_snapshot_v2(
            attempt, deadline_ns=deadline_ns
        )
        source_archive_digest = attempt.source_archive_root.removeprefix("sha256:")
        runtime_manifest_digest = attempt.runtime_environment_manifest_root.removeprefix(
            "sha256:"
        )
        if (
            source_archive_path.stat(follow_symlinks=False).st_size
            != attempt.source_archive_size_bytes
            or runtime_environment_manifest_path.stat(follow_symlinks=False).st_size
            != attempt.runtime_environment_manifest_size_bytes
            or _sha256_file_v2(source_archive_path, deadline_ns=deadline_ns)
            != source_archive_digest
            or _sha256_file_v2(
                runtime_environment_manifest_path, deadline_ns=deadline_ns
            )
            != runtime_manifest_digest
        ):
            raise ValueError("outer runner source/runtime CAS inputs drifted")
        runtime_manifest_bytes = _read_bounded_regular_file_v2(
            runtime_environment_manifest_path,
            maximum_bytes=1_048_576,
            name="runner runtime environment manifest",
            deadline_ns=deadline_ns,
        )
        source_tree = evidence_directory / "source-tree"
        extraction = _safe_extract_source_archive_v2(
            source_archive_path,
            source_tree,
            expected_sha256=source_archive_digest,
            expected_size=attempt.source_archive_size_bytes,
            expected_source_commit_sha=attempt.source_commit_sha,
            deadline_ns=deadline_ns,
        )
        if evidence_fd is None:
            raise RuntimeError("retained evidence descriptor is absent")
        source_tree_fd = retain_descriptor(
            _open_directory_at_fd_v2(
                evidence_fd, "source-tree", expected_path=source_tree
            )
        )
        expected_source = (
            source_tree
            / "src/hegel_machine/phase2b_formal_wire_covert_qualification_v2.py"
        )
        if not expected_source.is_file():
            raise ValueError("source archive lacks the qualification entry source")
        _verify_live_package_tree_is_source_only_v2(
            source_tree, deadline_ns=deadline_ns
        )
        expected_files = _expected_live_file_sha256_v2(attempt)
        for relative, expected_digest in expected_files.items():
            path = source_tree / relative
            if (
                not path.is_file()
                or _sha256_file_v2(path, deadline_ns=deadline_ns)
                != expected_digest
            ):
                raise ValueError(f"extracted live file differs: {relative}")
        (
            child_source_bootstrap_preopened_snapshots,
            child_source_bootstrap_preopened_snapshots_root,
        ) = _replay_extracted_child_preopened_sources_v2(
            source_tree,
            attempt.qualification_source_sha256,
            retained_source_tree_fd=source_tree_fd,
            deadline_ns=deadline_ns,
        )
        archive_profile_bytes = _read_bounded_regular_file_v2(
            source_tree / "config/phase2b_formal_wire_covert_qualification_v2.json",
            maximum_bytes=_MAXIMUM_PROFILE_BYTES,
            name="extracted qualification profile/config",
            deadline_ns=deadline_ns,
        )
        if archive_profile_bytes != profile_bytes:
            raise ValueError("executed profile bytes differ from archive config bytes")
        archive_lock_bytes = _read_bounded_regular_file_v2(
            source_tree
            / "config/phase2b_formal_wire_covert_qualification_v2.lock.json",
            maximum_bytes=256 * 1_024,
            name="extracted qualification dependency lock",
            deadline_ns=deadline_ns,
        )
        dependency_lock_runtime = _verify_dependency_lock_bytes_v2(
            archive_lock_bytes
        )
        rfc_bytes = _read_bounded_regular_file_v2(
            rfc8785_prerequisite_receipt_path,
            maximum_bytes=1_048_576,
            name="runner RFC-8785 prerequisite receipt",
            deadline_ns=deadline_ns,
        )
        rfc_receipt = _verify_rfc8785_prerequisite_receipt_v2(
            rfc_bytes, attempt, deadline_ns=deadline_ns
        )
        if rfc8785_prerequisite_receipt_path.name != "receipt.json":
            raise ValueError("RFC prerequisite receipt must use its frozen bundle name")
        rfc_source_bundle = rfc8785_prerequisite_receipt_path.parent.resolve(
            strict=True
        )
        rfc_retained_bundle = _mkdir_new_directory_v2(
            evidence_directory, "rfc8785-raw-bundle"
        )
        rfc_retained_bundle_fd = retain_descriptor(
            _open_directory_at_fd_v2(
                evidence_fd,
                "rfc8785-raw-bundle",
                expected_path=rfc_retained_bundle,
            )
        )
        _, rfc_raw_projection = _copy_rfc8785_raw_execution_bundle_v2(
            receipt=rfc_receipt,
            receipt_bytes=rfc_bytes,
            source_directory=rfc_source_bundle,
            destination_directory=rfc_retained_bundle,
            destination_directory_fd=rfc_retained_bundle_fd,
            deadline_ns=deadline_ns - 35_000_000_000,
            expected_qualification_source_sha256=(
                attempt.qualification_source_sha256
            ),
            replay_project_root=source_tree,
        )
        calibration_paths = (
            calibration_basis_profile_path,
            calibration_attempt_manifest_path,
            calibration_inner_artifact_path,
            calibration_outer_supervisor_result_path,
            calibration_rfc8785_prerequisite_receipt_path,
            calibration_mechanics_artifact_path,
        )
        calibration_bytes: tuple[bytes, ...] | None = None
        calibration_pair_projection: dict[str, object] | None = None
        calibration_pair_replay_token: (
            _VerifiedCalibrationPairReplayTokenV2 | None
        ) = None
        if entry_kind == "development-calibration":
            if (
                calibration_attempt_directory_path is not None
                or any(path is not None for path in calibration_paths)
            ):
                raise ValueError("calibration attempt cannot consume prior calibration")
        else:
            if (
                not isinstance(calibration_attempt_directory_path, Path)
                or not calibration_attempt_directory_path.is_absolute()
                or any(path is not None for path in calibration_paths)
            ):
                raise ValueError(
                    "live attempt needs one retained calibration attempt directory; "
                    "loose calibration artifact paths are forbidden"
                )
            (
                calibration_bytes,
                calibration_pair_projection,
                calibration_pair_replay_token,
            ) = (
                _load_verified_development_calibration_pair_v2(
                    calibration_attempt_directory_path,
                    deadline_ns=deadline_ns,
                    pair_replay_depth=1,
                    token_purpose="outer-mechanics",
                )
            )
            _retained_pair_tokens.append(calibration_pair_replay_token)
            shape = _EXPECTED_PROFILE["fixture_contract"]["rich_positive_shape"]
            if type(shape) is not dict:
                raise ValueError("live runner lacks the frozen rich-shape profile")
        python_row = _verify_live_executable_v2(
            rfc_receipt["executables"]["python"],
            "runner Python",
            deadline_ns=deadline_ns,
        )
        systemd_run_lookup = shutil.which("systemd-run")
        systemctl_lookup = shutil.which("systemctl")
        if systemd_run_lookup is None or systemctl_lookup is None:
            raise ValueError("outer runner cannot locate systemd control executables")
        systemd_run_row = _executable_projection_for_runner_v2(
            Path(systemd_run_lookup),
            version_argv=("--version",),
            name="systemd-run",
            deadline_ns=deadline_ns,
        )
        systemctl_row = _executable_projection_for_runner_v2(
            Path(systemctl_lookup),
            version_argv=("--version",),
            name="systemctl",
            deadline_ns=deadline_ns,
        )
        runtime_manifest = _verify_runtime_environment_manifest_v2(
            runtime_manifest_bytes,
            attempt=attempt,
            python_row=python_row,
            systemd_run_row=systemd_run_row,
            systemctl_row=systemctl_row,
        )
        if not _exact_json_equal(
            runtime_manifest.get("dependency_lock_runtime"),
            dependency_lock_runtime,
        ):
            raise ValueError("runtime manifest dependency-lock projection drifted")
        if inputs_fd is None or evidence_fd is None:
            raise RuntimeError("launch preflight lacks retained inputs/evidence fds")
        _atomic_write_new_file_at_fd_v2(
            inputs_fd,
            "runtime-environment.json",
            runtime_manifest_bytes,
            deadline_ns=deadline_ns,
        )
        systemctl_realpath = systemctl_row["realpath"]
        ownership_nonce = os.urandom(16).hex()
        unit_token = hashlib.sha256(
            (
                attempt.attempt_manifest_root
                + "\x00"
                + entry_kind
                + "\x00"
                + ownership_nonce
            ).encode("ascii")
        ).hexdigest()[:32]
        unit_base = "hegel-qv2-" + unit_token
        unit_name = unit_base + ".service"
        child_argv = _source_execution_bootstrap_argv_v2(
            python_realpath=python_row["realpath"],
            project_root=source_tree,
            qualification_source_sha256=attempt.qualification_source_sha256,
            source_arguments=[
                "--entry",
                entry_kind,
                "--profile",
                str(staged_profile_path),
                "--attempt",
                str(staged_attempt_path),
            ],
        )
        child_argv_sha256 = hashlib.sha256(
            canonicalize_rfc8785_v1(child_argv)
        ).hexdigest()
        unit_ownership_marker = "hegel-qv2-owned-" + hashlib.sha256(
            b"HEGEL/PHASE2B/QV2/UNIT_OWNERSHIP/V2\x00"
            + attempt.attempt_manifest_root.encode("ascii")
            + bytes.fromhex(ownership_nonce)
            + bytes.fromhex(child_argv_sha256)
        ).hexdigest()
        systemd_argv = [
            systemd_run_row["realpath"],
            "--user",
            "--pipe",
            "--wait",
            "--quiet",
            f"--unit={unit_base}",
            f"--property=Description={unit_ownership_marker}",
            "--property=Type=exec",
            "--property=RemainAfterExit=yes",
            f"--property=MemoryMax={attempt.requested_cgroup_memory_max_bytes}",
            f"--property=TasksMax={attempt.requested_cgroup_pids_max}",
            f"--property=RuntimeMaxSec={LIVE_HARD_TOTAL_TIMEOUT_SECONDS}s",
            "--property=KillMode=control-group",
            "--property=KillSignal=SIGTERM",
            "--property=SendSIGKILL=yes",
            "--property=FinalKillSignal=SIGKILL",
            "--property=TimeoutStopSec=30s",
            f"--working-directory={source_tree}",
            "--setenv=LANG=C.UTF-8",
            "--setenv=LC_ALL=C.UTF-8",
            "--",
            *child_argv,
        ]
        prelaunch_evidence, prelaunch_stdout, prelaunch_stderr = (
            _run_bounded_control_command_v2(
                _systemd_show_argv_v2(systemctl_realpath, unit_name),
                purpose="prelaunch-absent-check",
                deadline_ns=min(deadline_ns, time.monotonic_ns() + 5_000_000_000),
            )
        )
        systemctl_commands.append(prelaunch_evidence)
        prelaunch_show = _parse_systemd_show_v2(
            prelaunch_stdout, set(_SYSTEMD_SHOW_PROPERTIES_V2)
        )
        if (
            prelaunch_evidence["returncode"] != 0
            or prelaunch_stderr
            or prelaunch_show["LoadState"] != "not-found"
            or prelaunch_show["ActiveState"] != "inactive"
            or prelaunch_show["SubState"] != "dead"
            or prelaunch_show["MainPID"] != "0"
            or prelaunch_show["ControlGroup"] != ""
            or prelaunch_show["InvocationID"] != ""
        ):
            raise ValueError("transient unit name was not exact absent before launch")
        prelaunch_absent_verified = True
        launch_intent = {
            "schema_version": OUTER_LAUNCH_INTENT_SCHEMA_V2,
            "attempt_manifest_root": attempt.attempt_manifest_root,
            "entry_kind": entry_kind,
            "profile_id": attempt.profile_id,
            "source_tree_root": str(source_tree),
            "runtime_environment_manifest_root": (
                attempt.runtime_environment_manifest_root
            ),
            "runtime_environment_manifest_size_bytes": (
                attempt.runtime_environment_manifest_size_bytes
            ),
            "executables": {
                "python": python_row,
                "systemd_run": systemd_run_row,
                "systemctl": systemctl_row,
            },
            "ownership_nonce": ownership_nonce,
            "unit_name": unit_name,
            "ownership_marker": unit_ownership_marker,
            "child_argv": child_argv,
            "child_argv_sha256": child_argv_sha256,
            "systemd_argv": systemd_argv,
            "prelaunch_command": prelaunch_evidence,
            "prior_calibration_attempt_pair": calibration_pair_projection,
        }
        _validate_launch_intent_projection_v2(
            launch_intent,
            attempt=attempt,
            entry_kind=entry_kind,
            source_tree=source_tree,
        )
        launch_intent_bytes = _canonical_inner_artifact_v2(launch_intent)
        _atomic_write_new_file_at_fd_v2(
            evidence_fd,
            "launch-intent.json",
            launch_intent_bytes,
            deadline_ns=deadline_ns,
        )
        stage = "LAUNCH"
        launcher = subprocess.Popen(
            systemd_argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        entry_show: dict[str, str] | None = None
        for probe_index in range(8):
            probe_deadline = min(deadline_ns, time.monotonic_ns() + 5_000_000_000)
            evidence, stdout, entry_stderr = _run_bounded_control_command_v2(
                _systemd_show_argv_v2(systemctl_realpath, unit_name),
                purpose=f"entry-show-{probe_index}",
                deadline_ns=probe_deadline,
            )
            systemctl_commands.append(evidence)
            if evidence["returncode"] == 0:
                if entry_stderr:
                    raise ValueError("successful entry systemd snapshot wrote stderr")
                candidate = _parse_systemd_show_v2(
                    stdout, set(_SYSTEMD_SHOW_PROPERTIES_V2)
                )
                entry_state, invocation = _evaluate_owned_or_absent_systemd_show_v2(
                    candidate,
                    attempt=attempt,
                    source_tree=source_tree,
                    ownership_marker=unit_ownership_marker,
                    expected_invocation_id=owned_unit_invocation_id,
                    name="post-launch entry",
                )
                if entry_state == "owned":
                    if invocation is None:
                        raise RuntimeError("owned entry lost its InvocationID")
                    if owned_unit_invocation_id is None:
                        owned_unit_invocation_id = invocation
                else:
                    if owned_unit_invocation_id is not None:
                        raise ValueError(
                            "post-launch unit disappeared after ownership"
                        )
                if entry_state == "owned" and _systemd_entry_ready_v2(
                    candidate, name="entry service"
                ):
                    entry_show = candidate
                    break
            time.sleep(0.05)
        if entry_show is None:
            raise ValueError("transient service never reached active/running")
        if owned_unit_invocation_id != entry_show["InvocationID"]:
            raise ValueError("entry service does not equal the proven owned invocation")
        effective_properties = _normalize_effective_systemd_properties_v2(
            entry_show, attempt=attempt, source_tree=source_tree
        )
        stage = "INNER_EXECUTION"
        if launcher.stdout is None or launcher.stderr is None:
            raise ValueError("systemd-run launcher lacks exact stdout/stderr pipes")
        service_selector = selectors.DefaultSelector()
        service_selector.register(launcher.stdout, selectors.EVENT_READ, "stdout")
        service_selector.register(launcher.stderr, selectors.EVENT_READ, "stderr")
        terminal_show: dict[str, str] | None = None
        terminal_probe_count = 0
        try:
            while terminal_show is None:
                capture_deadline = deadline_ns - 35_000_000_000
                remaining_ns = capture_deadline - time.monotonic_ns()
                if remaining_ns <= 0:
                    raise TimeoutError("inner service reached its reserved cleanup deadline")
                for key, _ in service_selector.select(
                    min(1.0, remaining_ns / 1_000_000_000)
                ):
                    block = os.read(key.fileobj.fileno(), 65_536)
                    if not block:
                        service_selector.unregister(key.fileobj)
                        continue
                    cap = (
                        8 * 1_024 * 1_024
                        if key.data == "stdout"
                        else 256 * 1_024
                    )
                    _append_bounded_pipe_block_v2(
                        service_output[key.data],
                        block,
                        cap=cap,
                        stream=key.data,
                        capture_state=service_capture_state,
                    )
                if service_output["stderr"]:
                    raise ValueError(
                        "inner service stderr appeared before terminal success was proven"
                    )
                framed = bytes(service_output["stdout"])
                if framed.endswith(b"\n"):
                    _decode_canonical_transport_object_v2(
                        framed,
                        "runner framed inner artifact",
                        maximum_bytes=8 * 1_024 * 1_024,
                    )
                    if terminal_probe_count >= 32:
                        raise ValueError("inner artifact emitted but service did not exit")
                    terminal_evidence, terminal_stdout, terminal_stderr = (
                        _run_bounded_control_command_v2(
                            _systemd_show_argv_v2(systemctl_realpath, unit_name),
                            purpose=f"terminal-show-{terminal_probe_count}",
                            deadline_ns=min(
                                capture_deadline,
                                time.monotonic_ns() + 5_000_000_000,
                            ),
                        )
                    )
                    systemctl_commands.append(terminal_evidence)
                    terminal_probe_count += 1
                    if terminal_evidence["returncode"] != 0 or terminal_stderr:
                        raise ValueError("terminal systemd snapshot failed")
                    candidate = _parse_systemd_show_v2(
                        terminal_stdout, set(_SYSTEMD_SHOW_PROPERTIES_V2)
                    )
                    terminal_state, _ = _evaluate_owned_or_absent_systemd_show_v2(
                        candidate,
                        attempt=attempt,
                        source_tree=source_tree,
                        ownership_marker=unit_ownership_marker,
                        expected_invocation_id=owned_unit_invocation_id,
                        name="terminal probe",
                    )
                    if terminal_state != "owned":
                        raise ValueError("terminal probe lost its owned unit")
                    if _systemd_terminal_success_v2(
                        candidate, name="terminal loop"
                    ):
                        terminal_show = candidate
                    else:
                        time.sleep(0.05)
                elif launcher.poll() is not None:
                    raise ValueError("launcher exited without one complete inner artifact")
        finally:
            service_selector.close()
        inner_bytes = bytes(service_output["stdout"])
        service_stderr = bytes(service_output["stderr"])
        if not inner_bytes or service_stderr:
            raise ValueError("successful inner service needs nonempty stdout and empty stderr")
        if not _systemd_terminal_success_v2(
            terminal_show, name="terminal loop result"
        ):
            raise ValueError("terminal active/exited service result drifted")
        cgroup_path = entry_show["ControlGroup"]
        if Path("/sys/fs/cgroup" + cgroup_path).exists():
            raise ValueError("terminal active/exited service still has a cgroup path")
        pre_stop_evidence, pre_stop_stdout, pre_stop_stderr = (
            _run_bounded_control_command_v2(
                _systemd_show_argv_v2(systemctl_realpath, unit_name),
                purpose=f"terminal-show-{terminal_probe_count}",
                deadline_ns=min(deadline_ns, time.monotonic_ns() + 5_000_000_000),
            )
        )
        systemctl_commands.append(pre_stop_evidence)
        terminal_probe_count += 1
        pre_stop_show = _parse_systemd_show_v2(
            pre_stop_stdout, set(_SYSTEMD_SHOW_PROPERTIES_V2)
        )
        pre_stop_state, _ = _evaluate_owned_or_absent_systemd_show_v2(
            pre_stop_show,
            attempt=attempt,
            source_tree=source_tree,
            ownership_marker=unit_ownership_marker,
            expected_invocation_id=owned_unit_invocation_id,
            name="pre-stop snapshot",
        )
        if (
            pre_stop_evidence["returncode"] != 0
            or pre_stop_stderr
            or pre_stop_state != "owned"
            or not _systemd_terminal_success_v2(
                pre_stop_show, name="pre-stop snapshot"
            )
        ):
            raise ValueError("pre-stop unit ownership/result revalidation drifted")
        terminal_show = pre_stop_show
        stop_argv = [systemctl_realpath, "--user", "stop", unit_name]
        stop_evidence, stop_stdout, stop_stderr = _run_bounded_control_command_v2(
            stop_argv,
            purpose="stop-after-terminal-snapshot",
            deadline_ns=deadline_ns - 5_000_000_000,
        )
        systemctl_commands.append(stop_evidence)
        if stop_evidence["returncode"] != 0 or stop_stdout or stop_stderr:
            raise ValueError("terminal unit stop failed")
        try:
            trailing_stdout, trailing_stderr, launcher_returncode = (
                _capture_process_pipes_v2(
                    launcher,
                    deadline_ns=deadline_ns,
                    stdout_cap=8 * 1_024 * 1_024,
                    stderr_cap=256 * 1_024,
                    wait_for_process_exit=True,
                    stdout_buffer=service_output["stdout"],
                    stderr_buffer=service_output["stderr"],
                    capture_state=service_capture_state,
                )
            )
        except BaseException:
            raise
        else:
            _close_process_pipes_v2(
                launcher, name="successful outer launcher capture"
            )
        inner_bytes = bytes(service_output["stdout"])
        service_stderr = bytes(service_output["stderr"])
        inner_body = _decode_canonical_transport_object_v2(
            inner_bytes,
            "runner final drained inner artifact",
            maximum_bytes=8 * 1_024 * 1_024,
        )
        if service_stderr:
            raise ValueError("successful launcher emitted stderr after service stop")
        if launcher_returncode != 0:
            raise ValueError("systemd-run launcher did not report success")
        absent_show: dict[str, str] | None = None
        for absent_index in range(8):
            absent_evidence, absent_stdout, absent_stderr = (
                _run_bounded_control_command_v2(
                    _systemd_show_argv_v2(systemctl_realpath, unit_name),
                    purpose=f"post-stop-absent-check-{absent_index}",
                    deadline_ns=min(
                        deadline_ns, time.monotonic_ns() + 5_000_000_000
                    ),
                )
            )
            systemctl_commands.append(absent_evidence)
            if absent_evidence["returncode"] != 0 or absent_stderr:
                raise ValueError("post-stop absence probe command failed")
            candidate_absent = _parse_systemd_show_v2(
                absent_stdout, set(_SYSTEMD_SHOW_PROPERTIES_V2)
            )
            post_stop_state, _ = _evaluate_owned_or_absent_systemd_show_v2(
                candidate_absent,
                attempt=attempt,
                source_tree=source_tree,
                ownership_marker=unit_ownership_marker,
                expected_invocation_id=owned_unit_invocation_id,
                name="post-stop absence probe",
            )
            if post_stop_state == "absent":
                absent_show = candidate_absent
                break
            time.sleep(0.05)
        if absent_show is None:
            raise ValueError("transient service was not exact not-found after stop")
        service_ended_ns = time.monotonic_ns()
        if service_ended_ns > deadline_ns:
            raise TimeoutError("outer attempt exceeded its hard total deadline")
        for relative, expected_digest in expected_files.items():
            if (
                _sha256_file_v2(
                    source_tree / relative, deadline_ns=deadline_ns
                )
                != expected_digest
            ):
                raise ValueError(f"post-run live file differs: {relative}")
        (
            post_child_snapshots,
            post_child_snapshots_root,
        ) = _replay_extracted_child_preopened_sources_v2(
            source_tree,
            attempt.qualification_source_sha256,
            retained_source_tree_fd=source_tree_fd,
            deadline_ns=deadline_ns,
        )
        if (
            child_source_bootstrap_preopened_snapshots is None
            or child_source_bootstrap_preopened_snapshots_root is None
            or not _exact_json_equal(
                post_child_snapshots,
                child_source_bootstrap_preopened_snapshots,
            )
            or post_child_snapshots_root
            != child_source_bootstrap_preopened_snapshots_root
        ):
            raise ValueError("child preopened source snapshots drifted during execution")
        outer_runner_tcb_exit = _outer_runner_tcb_snapshot_v2(
            attempt, deadline_ns=deadline_ns
        )
        if not _exact_json_equal(outer_runner_tcb_entry, outer_runner_tcb_exit):
            raise ValueError("outer runner TCB changed during the attempt")
        stage = "CAS"
        cas_by_digest: dict[str, dict[str, object]] = {}

        def put_bytes(payload: bytes) -> dict[str, object]:
            digest = hashlib.sha256(payload).hexdigest()
            existing = cas_by_digest.get(digest)
            if existing is not None:
                if existing["object_size_bytes"] != len(payload):
                    raise ValueError("attempt-private CAS digest/size collision")
                return existing
            if cas_fd is None:
                raise RuntimeError("attempt-private retained CAS descriptor is absent")
            row = _cas_put_bytes_at_fd_v2(
                cas_fd,
                payload,
                store_namespace=attempt.attempt_manifest_root,
                deadline_ns=deadline_ns,
            )
            cas_by_digest[digest] = row
            return row

        if cas_fd is None:
            raise RuntimeError("attempt-private retained CAS descriptor is absent")
        source_archive_cas = _cas_put_file_v2(
            cas_directory,
            source_archive_path,
            expected_sha256=source_archive_digest,
            expected_size=attempt.source_archive_size_bytes,
            store_namespace=attempt.attempt_manifest_root,
            name="source archive",
            deadline_ns=deadline_ns,
            retained_cas_fd=cas_fd,
        )
        cas_by_digest[source_archive_digest] = source_archive_cas
        runtime_manifest_cas = _cas_put_file_v2(
            cas_directory,
            runtime_environment_manifest_path,
            expected_sha256=runtime_manifest_digest,
            expected_size=attempt.runtime_environment_manifest_size_bytes,
            store_namespace=attempt.attempt_manifest_root,
            name="runtime environment manifest",
            deadline_ns=deadline_ns,
            retained_cas_fd=cas_fd,
        )
        cas_by_digest[runtime_manifest_digest] = runtime_manifest_cas
        rfc_cas = put_bytes(rfc_bytes)
        inner_cas = put_bytes(inner_bytes)
        profile_cas = put_bytes(profile_bytes)
        attempt_cas = put_bytes(attempt_bytes)
        if launch_intent_bytes is None:
            raise RuntimeError("successful launch lacks its retained intent")
        launch_intent_cas = put_bytes(launch_intent_bytes)
        calibration_cas: list[dict[str, object]] = []
        if calibration_bytes is not None:
            calibration_cas = [put_bytes(payload) for payload in calibration_bytes]
        source_archive_object_path = cas_directory / source_archive_digest
        runtime_manifest_object_path = cas_directory / runtime_manifest_digest
        supervisor_ready_ns = time.monotonic_ns()
        if supervisor_ready_ns >= deadline_ns:
            raise TimeoutError("outer supervisor construction reached the hard deadline")
        supervisor_body = {
            "schema_version": OUTER_SUPERVISOR_RESULT_SCHEMA_V2,
            "mode": entry_kind,
            "child_source_bootstrap_preopened_snapshots": (
                child_source_bootstrap_preopened_snapshots
            ),
            "child_source_bootstrap_preopened_snapshots_root": (
                child_source_bootstrap_preopened_snapshots_root
            ),
            "outer_runner_tcb_entry": outer_runner_tcb_entry,
            "outer_runner_tcb_exit": outer_runner_tcb_exit,
            "profile_id": PROFILE_ID,
            "attempt_manifest_root": attempt.attempt_manifest_root,
            "source_commit_sha": attempt.source_commit_sha,
            "source_archive_root": attempt.source_archive_root,
            "source_archive_cas": source_archive_cas,
            "source_archive_object_path": str(source_archive_object_path),
            "source_archive_extraction": extraction,
            "runtime_environment_manifest_root": (
                attempt.runtime_environment_manifest_root
            ),
            "runtime_environment_manifest_cas": runtime_manifest_cas,
            "runtime_environment_manifest_object_path": str(
                runtime_manifest_object_path
            ),
            "runtime_image_digest": None,
            "source_tree_root": str(source_tree),
            "live_files_sha256": expected_files,
            "launcher": {
                "argv": child_argv,
                "environment": {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
                "python": python_row,
                "systemd_argv": systemd_argv,
                "systemd_run": systemd_run_row,
                "systemctl": systemctl_row,
                "systemctl_commands": systemctl_commands,
            },
            "unit": {
                "cgroup_path": cgroup_path,
                "child_argv_sha256": child_argv_sha256,
                "effective_properties": effective_properties,
                "invocation_id": entry_show["InvocationID"],
                "owned_invocation_id": owned_unit_invocation_id,
                "ownership_marker": unit_ownership_marker,
                "ownership_nonce": ownership_nonce,
                "prelaunch_absent_verified": prelaunch_absent_verified,
                "selected_environment_root": (
                    attempt.runtime_environment_manifest_root
                ),
                "unit_name": unit_name,
                "working_directory": str(source_tree),
            },
            "timing": {
                "started_monotonic_nanoseconds": started_ns,
                "service_ended_monotonic_nanoseconds": service_ended_ns,
                "supervisor_ready_monotonic_nanoseconds": supervisor_ready_ns,
                "elapsed_to_supervisor_ready_nanoseconds": (
                    supervisor_ready_ns - started_ns
                ),
                "watchdog_deadline_monotonic_nanoseconds": hard_deadline_ns,
            },
            "result": {
                "control_group_empty": True,
                "control_group_property_at_terminal_snapshot": "",
                "cgroup_path_absent_at_terminal_snapshot": True,
                "exec_main_code": 1,
                "exec_main_status": 0,
                "kill_sent": False,
                "launcher_exit_code": 0,
                "launcher_reaped": True,
                "main_pid_at_terminal_snapshot": 0,
                "result": "success",
                "term_sent": False,
                "timed_out": False,
                "unit_absent_after_collection": True,
                "unit_active_state_at_terminal_snapshot": "active",
                "unit_collected_after_terminal_snapshot": True,
                "unit_sub_state_at_terminal_snapshot": "exited",
            },
            "post_output_physical_gate_attestation": (
                _expected_post_output_gate_attestation_v2(
                    attempt,
                    inner_body,
                    inner_bytes,
                    expected_entry=entry_kind,
                    child_argv_sha256=child_argv_sha256,
                    owned_invocation_id=owned_unit_invocation_id,
                )
            ),
            "stdout": {
                "sha256": hashlib.sha256(inner_bytes).hexdigest(),
                "size_bytes": len(inner_bytes),
            },
            "stderr": {
                "sha256": hashlib.sha256(service_stderr).hexdigest(),
                "size_bytes": len(service_stderr),
            },
            "inner_artifact_cas": inner_cas,
            "rfc8785_receipt_cas": rfc_cas,
            "rfc8785_raw_bundle_directory": str(rfc_retained_bundle),
            "rfc8785_raw_bundle_projection": rfc_raw_projection,
            "content_sha256_is_not_a_signature": True,
        }
        outer_bytes = _canonical_inner_artifact_v2(supervisor_body)
        stage = "MECHANICS_VERIFY"
        verify_kwargs: dict[str, object] = {}
        if calibration_bytes is not None:
            verify_kwargs = {
                "calibration_basis_profile_bytes": calibration_bytes[0],
                "calibration_attempt_manifest_bytes": calibration_bytes[1],
                "calibration_inner_artifact_bytes": calibration_bytes[2],
                "calibration_outer_supervisor_result_bytes": calibration_bytes[3],
                "calibration_rfc8785_prerequisite_receipt_bytes": calibration_bytes[4],
                "calibration_mechanics_artifact_bytes": calibration_bytes[5],
                "_calibration_pair_replay_token": calibration_pair_replay_token,
                "_calibration_pair_projection": calibration_pair_projection,
            }
        mechanics_bytes = _verify_phase2b_formal_wire_covert_outer_mechanics_core_v2(
            profile,
            attempt,
            inner_artifact_bytes=inner_bytes,
            outer_supervisor_result_bytes=outer_bytes,
            rfc8785_prerequisite_receipt_bytes=rfc_bytes,
            _deadline_ns=deadline_ns,
            **verify_kwargs,
        )
        outer_cas = put_bytes(outer_bytes)
        mechanics_cas = put_bytes(mechanics_bytes)
        stage = "FINALIZE_DIRECTORY"
        if evidence_fd is None:
            raise RuntimeError("retained evidence descriptor is absent at finalization")
        _atomic_write_new_file_at_fd_v2(
            evidence_fd,
            "inner.json",
            inner_bytes,
            deadline_ns=deadline_ns,
        )
        _atomic_write_new_file_at_fd_v2(
            evidence_fd,
            "outer-supervisor.json",
            outer_bytes,
            deadline_ns=deadline_ns,
        )
        _atomic_write_new_file_at_fd_v2(
            evidence_fd,
            "mechanics.json",
            mechanics_bytes,
            deadline_ns=deadline_ns,
        )
        for retained_fd, retained_path, retained_name in (
            (attempt_fd, attempt_directory, "attempt"),
            (evidence_fd, evidence_directory, "evidence"),
            (cas_fd, cas_directory, "CAS"),
            (inputs_fd, inputs_directory, "inputs"),
            (source_tree_fd, source_tree, "source tree"),
            (rfc_retained_bundle_fd, rfc_retained_bundle, "RFC raw bundle"),
        ):
            if retained_fd is None:
                raise RuntimeError(f"retained {retained_name} descriptor is absent")
            _verify_retained_directory_path_identity_v2(
                retained_fd, retained_path, name=retained_name
            )
        success_tree_before_terminal = _retained_attempt_tree_before_terminal_v2(
            attempt_fd, deadline_ns=deadline_ns
        )
        provisional_ready_ns = time.monotonic_ns()
        if provisional_ready_ns >= deadline_ns:
            raise TimeoutError("outer finalization reached the work deadline")
        success_body = {
            "schema_version": OUTER_ATTEMPT_SUCCESS_SCHEMA_V2,
            "disposition": "MECHANICS_VERIFIED_NOT_AUTHORITY",
            "campaign_id": attempt.campaign_id,
            "attempt_id": attempt.attempt_id,
            "profile_id": PROFILE_ID,
            "attempt_manifest_root": attempt.attempt_manifest_root,
            "entry_kind": entry_kind,
            "retained_tree_before_terminal": success_tree_before_terminal,
            "attempt_private_cas_directory": str(cas_directory),
            "attempt_private_cas_store_namespace": attempt.attempt_manifest_root,
            "artifacts": {
                "profile": profile_cas,
                "attempt_manifest": attempt_cas,
                "launch_intent": launch_intent_cas,
                "source_archive": source_archive_cas,
                "runtime_environment_manifest": runtime_manifest_cas,
                "rfc8785_prerequisite_receipt": rfc_cas,
                "rfc8785_raw_bundle": {
                    "directory": str(rfc_retained_bundle),
                    "projection": rfc_raw_projection,
                },
                "inner": inner_cas,
                "outer_supervisor": outer_cas,
                "mechanics": mechanics_cas,
                "prior_calibration_attempt_pair": calibration_pair_projection,
                "prior_calibration_in_order": calibration_cas,
            },
            "terminal_publication_count": 1,
            "timing": {
                "elapsed_to_finalization_ready_nanoseconds": (
                    provisional_ready_ns - started_ns
                ),
                "finalization_ready_monotonic_nanoseconds": provisional_ready_ns,
                "service_ended_monotonic_nanoseconds": service_ended_ns,
                "started_monotonic_nanoseconds": started_ns,
                "watchdog_deadline_monotonic_nanoseconds": hard_deadline_ns,
            },
            "content_sha256_is_not_a_signature": True,
            "projection_only_boundary": dict(_TASK4_PROJECTION_ONLY_BOUNDARY_V2),
            "step5_signed_cas_authority_still_required": True,
            "claims": {
                "r03_formal_trusted_wire_implementation_qualified": False,
                "r04_formal_covert_auditor_implementation_qualified": False,
                "r05_synthetic_dev_wire_covert_qualification_verified": False,
                "actual_corpus_audit": False,
                "actual_unsealed_960_generation": False,
                "origin_authenticated": False,
                "key_independence_verified": False,
                "campaign_execution_authorized": False,
            },
        }
        success_bytes = _canonical_inner_artifact_v2(success_body)
        verified_terminal = (
            verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
                attempt_directory,
                expected_terminal_bytes=success_bytes,
                deadline_ns=deadline_ns,
                retained_attempt_fd=attempt_fd,
                retained_output_root_fd=output_root_fd,
                retained_evidence_fd=evidence_fd,
                retained_cas_fd=cas_fd,
                retained_inputs_fd=inputs_fd,
                retained_source_tree_fd=source_tree_fd,
                retained_rfc_raw_bundle_fd=rfc_retained_bundle_fd,
                prospective_terminal=True,
            )
        )
        prospective_completed_ns = time.monotonic_ns()
        if (
            verified_terminal != success_bytes
            or prospective_completed_ns >= deadline_ns
        ):
            raise TimeoutError(
                "full-scope prospective replay did not finish before deadline"
            )
        success_bytes = _retime_verified_outer_terminal_v2(
            success_bytes,
            prospective_replay_completed_ns=prospective_completed_ns,
            prospective_deadline_ns=deadline_ns,
        )
        _publish_terminal_once_at_fd_v2(
            attempt_fd, success_bytes, deadline_ns=hard_deadline_ns
        )
        terminal_committed = True
        _verify_parent_identity_after_terminal_commit_v2(
            output_root_fd,
            attempt_directory_name,
            attempt_fd,
            name="successful attempt",
            deadline_ns=hard_deadline_ns,
        )
        close_retained_descriptors()
        return success_bytes
    except BaseException as exc:
        if terminal_committed or _atomic_uncertainty_blocks_another_terminal_v2(exc):
            close_retained_descriptors()
            raise
        managed_atomic_residues = _reconcile_nonterminal_atomic_uncertainty_v2(
            exc,
            attempt_fd=attempt_fd,
            deadline_ns=hard_deadline_ns,
            known_parent_directories=known_attempt_parent_directories(),
        )
        primary_failure_stage = stage
        primary_failure_exception_type = type(exc).__name__
        reason_code = (
            exc.code
            if type(exc) is Phase2BFormalWireCovertQualificationErrorV2
            else FAIL_ISOLATED_EXECUTION
        )
        cleanup: dict[str, object] | None = None
        cleanup_failure_exception_type: str | None = None
        child_pipe_capture: dict[str, object] | None = None
        capture_publication_failures: list[BaseException] = []

        def reconcile_capture_publication_failure(
            capture_exc: BaseException,
        ) -> None:
            additional = _reconcile_nonterminal_atomic_uncertainty_v2(
                capture_exc,
                attempt_fd=attempt_fd,
                deadline_ns=hard_deadline_ns,
                known_parent_directories=known_attempt_parent_directories(),
            )
            for row in additional:
                if not any(
                    _exact_json_equal(row, existing)
                    for existing in managed_atomic_residues
                ):
                    managed_atomic_residues.append(row)

        if launcher is not None:
            if (
                systemctl_realpath is None
                or unit_name is None
                or unit_ownership_marker is None
                or source_tree_fd is None
                or evidence_fd is None
                or cas_fd is None
            ):
                close_retained_descriptors()
                raise RuntimeError(
                    "launched attempt lacks the evidence needed for safe cleanup"
                ) from exc
            cleanup_deadline_ns = hard_deadline_ns - (
                _OUTER_TERMINAL_PUBLICATION_RESERVE_SECONDS_V2
                * 1_000_000_000
            )
            cleanup_control_deadline_ns = cleanup_deadline_ns - (
                _OUTER_LAUNCHER_FINALIZATION_RESERVE_SECONDS_V2
                * 1_000_000_000
            )
            prefix_projection: dict[str, object] | None = None
            capture_persistence_error_type: str | None = None
            try:
                prefix_projection = _persist_failure_child_pipe_prefix_v2(
                    evidence_fd,
                    stdout=bytes(service_output["stdout"]),
                    stderr=bytes(service_output["stderr"]),
                    deadline_ns=cleanup_control_deadline_ns,
                )
            except BaseException as capture_exc:
                capture_persistence_error_type = type(capture_exc).__name__
                capture_publication_failures.append(capture_exc)
            cleanup_start_command_index = len(systemctl_commands)
            cleanup_exc: BaseException | None = None
            try:
                try:
                    cleanup = _cleanup_transient_attempt_v2(
                        launcher=launcher,
                        systemctl_realpath=systemctl_realpath,
                        unit_name=unit_name,
                        ownership_marker=unit_ownership_marker,
                        owned_invocation_id=owned_unit_invocation_id,
                        prelaunch_absent_verified=prelaunch_absent_verified,
                        attempt=attempt,
                        source_tree=source_tree,
                        deadline_ns=cleanup_control_deadline_ns,
                        commands=systemctl_commands,
                        cleanup_start_command_index=cleanup_start_command_index,
                        stdout_capture=service_output["stdout"],
                        stderr_capture=service_output["stderr"],
                        capture_state=service_capture_state,
                    )
                finally:
                    _finalize_outer_launcher_v2(
                        launcher, deadline_ns=cleanup_deadline_ns
                    )
            except BaseException as observed_cleanup_exc:
                cleanup_exc = observed_cleanup_exc
                cleanup_failure_exception_type = type(cleanup_exc).__name__
                cleanup = {
                    "cleanup_error_type": cleanup_failure_exception_type,
                    "cleanup_start_command_index": cleanup_start_command_index,
                    "cleanup_verified": False,
                    "commands": list(systemctl_commands),
                    "external_intervention_required": True,
                    "owned_invocation_id": owned_unit_invocation_id,
                    "ownership_marker": unit_ownership_marker,
                    "prelaunch_unit_absent": prelaunch_absent_verified,
                    "transcript_root": stable_hash(
                        systemctl_commands,
                        prefix=(
                            "phase2b_formal_wire_covert_failure_cleanup_v2_"
                        ),
                    ),
                    "unit_name": unit_name,
                }
            if cleanup_exc is not None:
                # /failure/4 unproven cleanup has no positive launcher-reap
                # fields that a pure replay can cross-bind.  Preserve every
                # byte actually accumulated, but conservatively withhold the
                # complete-capture claim even if EOF/reap preceded a later
                # ownership or absence-probe failure.
                service_capture_state["capture_complete"] = False
            if capture_persistence_error_type is None:
                if prefix_projection is None:
                    raise RuntimeError(
                        "failure child pipe prefix projection disappeared"
                    )
                try:
                    child_pipe_capture = _persist_failure_child_pipe_capture_v2(
                        evidence_fd,
                        cas_fd,
                        stdout=bytes(service_output["stdout"]),
                        stderr=bytes(service_output["stderr"]),
                        prefix_projection=prefix_projection,
                        capture_complete=service_capture_state["capture_complete"],
                        capture_truncated=service_capture_state[
                            "capture_truncated"
                        ],
                        store_namespace=attempt.attempt_manifest_root,
                        deadline_ns=cleanup_deadline_ns,
                    )
                except BaseException as capture_exc:
                    capture_persistence_error_type = type(capture_exc).__name__
                    capture_publication_failures.append(capture_exc)
            for capture_exc in capture_publication_failures:
                reconcile_capture_publication_failure(capture_exc)
            if capture_persistence_error_type is not None:
                child_pipe_capture = (
                    _failure_child_pipe_persistence_unverified_projection_v2(
                        evidence_fd,
                        persistence_error_type=capture_persistence_error_type,
                        capture_truncated=service_capture_state[
                            "capture_truncated"
                        ],
                        deadline_ns=hard_deadline_ns,
                    )
                )
            if cleanup_exc is not None:
                failure_projection = _derive_failure_v4_retained_projection_at_fd_v2(
                    attempt_fd,
                    managed_atomic_residues=managed_atomic_residues,
                    deadline_ns=hard_deadline_ns,
                )
                failure_cas_projection = failure_projection[
                    "failure_cas_projection"
                ]
                retained_tree = failure_projection[
                    "retained_tree_before_terminal"
                ]
                unproven_bytes = _closed_outer_failure_artifact_v2(
                    attempt,
                    entry_kind=entry_kind,
                    stage=primary_failure_stage,
                    primary_failure_exception_type=(
                        primary_failure_exception_type
                    ),
                    reason_code=reason_code,
                    cleanup=cleanup,
                    cleanup_failure_exception_type=(
                        cleanup_failure_exception_type
                    ),
                    child_pipe_capture=child_pipe_capture,
                    failure_cas_projection=failure_cas_projection,
                    cleanup_verified=False,
                    external_intervention_required=True,
                    launcher_started=True,
                    managed_atomic_residues=managed_atomic_residues,
                    retained_tree_before_terminal=retained_tree,
                    started_ns=started_ns,
                    deadline_ns=hard_deadline_ns,
                )
                verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
                    attempt_directory,
                    expected_terminal_bytes=unproven_bytes,
                    deadline_ns=hard_deadline_ns,
                    retained_attempt_fd=attempt_fd,
                    retained_output_root_fd=output_root_fd,
                    prospective_terminal=True,
                )
                unproven_bytes = _retime_verified_outer_terminal_v2(
                    unproven_bytes,
                    prospective_replay_completed_ns=time.monotonic_ns(),
                    prospective_deadline_ns=hard_deadline_ns,
                )
                _publish_terminal_once_at_fd_v2(
                    attempt_fd,
                    unproven_bytes,
                    deadline_ns=hard_deadline_ns,
                )
                terminal_committed = True
                _verify_parent_identity_after_terminal_commit_v2(
                    output_root_fd,
                    attempt_directory_name,
                    attempt_fd,
                    name="cleanup-unproven attempt",
                    deadline_ns=hard_deadline_ns,
                )
                close_retained_descriptors()
                raise _OuterCleanupUnprovenV2(
                    "failure cleanup is unproven; inspect retained terminal"
                ) from cleanup_exc
        if "terminal.json" in _directory_names_at_fd_v2(attempt_fd):
            raise _AtomicPublicationUncertainV2(
                "terminal.json",
                "failure path found an already-published terminal"
            ) from exc
        failure_projection = _derive_failure_v4_retained_projection_at_fd_v2(
            attempt_fd,
            managed_atomic_residues=managed_atomic_residues,
            deadline_ns=hard_deadline_ns,
        )
        failure_cas_projection = failure_projection["failure_cas_projection"]
        retained_tree = failure_projection["retained_tree_before_terminal"]
        failure_bytes = _closed_outer_failure_artifact_v2(
            attempt,
            entry_kind=entry_kind,
            stage=primary_failure_stage,
            primary_failure_exception_type=primary_failure_exception_type,
            reason_code=reason_code,
            cleanup=cleanup,
            cleanup_failure_exception_type=cleanup_failure_exception_type,
            child_pipe_capture=child_pipe_capture,
            failure_cas_projection=failure_cas_projection,
            cleanup_verified=True,
            external_intervention_required=False,
            launcher_started=launcher is not None,
            managed_atomic_residues=managed_atomic_residues,
            retained_tree_before_terminal=retained_tree,
            started_ns=started_ns,
            deadline_ns=hard_deadline_ns,
        )
        verify_phase2b_formal_wire_covert_outer_attempt_directory_v2(
            attempt_directory,
            expected_terminal_bytes=failure_bytes,
            deadline_ns=hard_deadline_ns,
            retained_attempt_fd=attempt_fd,
            retained_output_root_fd=output_root_fd,
            prospective_terminal=True,
        )
        failure_bytes = _retime_verified_outer_terminal_v2(
            failure_bytes,
            prospective_replay_completed_ns=time.monotonic_ns(),
            prospective_deadline_ns=hard_deadline_ns,
        )
        _publish_terminal_once_at_fd_v2(
            attempt_fd, failure_bytes, deadline_ns=hard_deadline_ns
        )
        terminal_committed = True
        _verify_parent_identity_after_terminal_commit_v2(
            output_root_fd,
            attempt_directory_name,
            attempt_fd,
            name="closed failure attempt",
            deadline_ns=hard_deadline_ns,
        )
        close_retained_descriptors()
        return failure_bytes


def run_phase2b_formal_wire_covert_outer_attempt_v2(
    *,
    entry_kind: str,
    profile_path: Path,
    attempt_manifest_path: Path,
    rfc8785_prerequisite_receipt_path: Path,
    source_archive_path: Path,
    runtime_environment_manifest_path: Path,
    output_root: Path,
    calibration_attempt_directory_path: Path | None = None,
    calibration_basis_profile_path: Path | None = None,
    calibration_attempt_manifest_path: Path | None = None,
    calibration_inner_artifact_path: Path | None = None,
    calibration_outer_supervisor_result_path: Path | None = None,
    calibration_rfc8785_prerequisite_receipt_path: Path | None = None,
    calibration_mechanics_artifact_path: Path | None = None,
) -> bytes:
    """Own the fd registry for one outer transaction and close it on every exit."""

    retained_descriptors: list[int] = []
    retained_pair_tokens: list[_VerifiedCalibrationPairReplayTokenV2] = []
    primary: BaseException | None = None
    try:
        return _run_phase2b_formal_wire_covert_outer_attempt_impl_v2(
            entry_kind=entry_kind,
            profile_path=profile_path,
            attempt_manifest_path=attempt_manifest_path,
            rfc8785_prerequisite_receipt_path=rfc8785_prerequisite_receipt_path,
            source_archive_path=source_archive_path,
            runtime_environment_manifest_path=runtime_environment_manifest_path,
            output_root=output_root,
            calibration_attempt_directory_path=calibration_attempt_directory_path,
            calibration_basis_profile_path=calibration_basis_profile_path,
            calibration_attempt_manifest_path=calibration_attempt_manifest_path,
            calibration_inner_artifact_path=calibration_inner_artifact_path,
            calibration_outer_supervisor_result_path=(
                calibration_outer_supervisor_result_path
            ),
            calibration_rfc8785_prerequisite_receipt_path=(
                calibration_rfc8785_prerequisite_receipt_path
            ),
            calibration_mechanics_artifact_path=calibration_mechanics_artifact_path,
            _retained_descriptors=retained_descriptors,
            _retained_pair_tokens=retained_pair_tokens,
        )
    except BaseException as exc:
        primary = exc
        raise
    finally:
        while retained_pair_tokens:
            _discard_verified_calibration_pair_replay_token_v2(
                retained_pair_tokens.pop()
            )
        close_errors: list[BaseException] = []
        while retained_descriptors:
            descriptor = retained_descriptors.pop()
            try:
                os.close(descriptor)
            except BaseException as exc:
                close_errors.append(exc)
        if close_errors:
            message = (
                f"outer transaction could not close {len(close_errors)} "
                "owned descriptors"
            )
            if primary is not None:
                _append_exception_note_v2(primary, message)
            else:
                raise RuntimeError(message) from close_errors[0]


def qualify_phase2b_formal_wire_covert_candidate_v2(
    profile: Phase2BFormalWireCovertQualificationProfileV2,
) -> (
    Phase2BFormalWireCovertCandidateReceiptV2
    | Phase2BFormalWireCovertQualificationRejectionV2
):
    """Run two real 1,024-row audits; issue no R03/R04/R05 claim."""

    profile_id = getattr(profile, "profile_id", None)
    positive: _IsolatedAuditEvidenceV2 | None = None
    negative: _IsolatedAuditEvidenceV2 | None = None
    try:
        if type(profile) is not Phase2BFormalWireCovertQualificationProfileV2:
            _fail(FAIL_INPUT, "candidate qualification needs the exact loaded profile")
        profile._validate()
        if (
            wire.ENVELOPE_BYTES_V3 != auditor.ENVELOPE_BYTES
            or wire.MAXIMUM_BATCH_CASES_V3 != EXACT_AUTHORITY_ROWS
            or auditor.MAXIMUM_ROWS != EXACT_AUTHORITY_ROWS
            or auditor.LABEL_PERMUTATIONS != EXACT_LABEL_PERMUTATIONS
            or 1
            + auditor.GLOBAL_RENAMINGS
            + auditor.GLOBAL_CASE_ORDER_PERMUTATIONS
            + auditor.OBSERVATION_ORDER_PERMUTATIONS
            != EXACT_RECOGNIZER_INVOCATIONS
        ):
            _fail(FAIL_PROFILE, "wire/auditor scalar binding drift")
        shape = _EXPECTED_PROFILE["fixture_contract"]["rich_positive_shape"]
        if type(shape) is not dict or shape["live_qualification_enabled"] is not True:
            _fail(
                FAIL_PROFILE,
                "full-1024 development shape calibration is still pending",
            )
        positive = _execute_isolated_audit_v2("KNOWN_NULL")
        negative = _execute_isolated_audit_v2("KNOWN_BINDING_LEAK")
        _verify_frozen_shape_v2(shape, positive, negative, attempt=None)
        fixture_root = _combined_fixture_root_v2(positive, negative)
        return Phase2BFormalWireCovertCandidateReceiptV2._issue(
            _RECEIPT_ISSUE_TOKEN,
            synthetic_fixture_root=fixture_root,
            positive=positive,
            negative=negative,
        )
    except BaseException as exc:
        for evidence in (positive, negative):
            if evidence is not None:
                try:
                    auditor.release_formal_covert_audit_receipt_v1(
                        evidence.audit_receipt
                    )
                except (KeyError, TypeError, ValueError):
                    pass
        reason = (
            exc.code
            if type(exc) is Phase2BFormalWireCovertQualificationErrorV2
            else FAIL_ISOLATED_EXECUTION
        )
        rejection = Phase2BFormalWireCovertQualificationRejectionV2(
            QualificationDispositionV2.ATOMIC_EMPTY_REJECTION,
            reason,
            profile_id if profile_id == PROFILE_ID else None,
        )
        rejection.__post_init__()
        return rejection


class _RFC8785ExpectedExceptionV2:
    __slots__ = ("_expected", "_match")

    def __init__(self, expected: type[BaseException], match: str | None) -> None:
        if not isinstance(expected, type) or not issubclass(expected, BaseException):
            raise TypeError("RFC selfcheck expected exception is invalid")
        self._expected = expected
        self._match = match

    def __enter__(self) -> "_RFC8785ExpectedExceptionV2":
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> bool:
        del traceback
        if exception_type is None or exception is None:
            raise AssertionError("RFC selfcheck expected an exception")
        if not issubclass(exception_type, self._expected):
            return False
        if self._match is not None and re.search(self._match, str(exception)) is None:
            raise AssertionError("RFC selfcheck exception text did not match")
        return True


class _RFC8785MarkStubV2:
    @staticmethod
    def parametrize(name: str, values: object):
        if name != "value" or type(values) is not list:
            raise ValueError("RFC selfcheck accepts only its pinned parameterization")
        frozen = tuple(values)

        def decorate(function):
            setattr(function, "__hegel_rfc8785_parameters_v2__", frozen)
            return function

        return decorate


class _RFC8785SealedTextPathV2:
    __slots__ = ("_display", "_text")

    def __init__(self, display: Path, payload: bytes) -> None:
        self._display = str(display)
        self._text = payload.decode("utf-8", errors="strict")

    def read_text(self, *, encoding: str) -> str:
        if encoding != "utf-8":
            raise ValueError("RFC sealed path accepts only UTF-8 text")
        return self._text

    def __str__(self) -> str:
        return self._display


class _RFC8785SealedSubprocessV2:
    __slots__ = (
        "_check_calls",
        "_finite_rng",
        "_node_fd",
        "_number_batches",
        "_oracle_display",
        "_oracle_text",
        "_vectors_display",
        "_vectors_fd",
    )

    def __init__(
        self,
        *,
        node_fd: int,
        oracle_display: Path,
        oracle_bytes: bytes,
        vectors_display: Path,
        vectors_fd: int,
    ) -> None:
        self._check_calls = 0
        self._finite_rng = random.Random(
            _RFC8785_RANDOM_DIFFERENTIAL_SEED_V2
        )
        self._node_fd = node_fd
        self._number_batches = 0
        self._oracle_display = str(oracle_display)
        self._oracle_text = oracle_bytes.decode("utf-8", errors="strict")
        self._vectors_display = str(vectors_display)
        self._vectors_fd = vectors_fd

    def run(self, argv: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if kwargs != {"check": True, "capture_output": True, "text": True}:
            raise ValueError("RFC sealed subprocess options drifted")
        if (
            type(argv) is not list
            or len(argv) < 3
            or argv[0] != "node"
            or argv[1] != self._oracle_display
            or any(type(item) is not str or "\x00" in item for item in argv)
        ):
            raise ValueError("RFC sealed subprocess argv left its allowlist")
        suffix = list(argv[2:])
        inherited = [self._node_fd]
        if suffix == ["--check", self._vectors_display]:
            if self._check_calls != 0 or self._number_batches != 0:
                raise ValueError("RFC sealed --check ordering/count drifted")
            self._check_calls = 1
            suffix[-1] = f"/proc/self/fd/{self._vectors_fd}"
            inherited.append(self._vectors_fd)
        elif suffix and suffix[0] == "--number-bits":
            if (
                self._check_calls != 1
                or self._number_batches >= 50
                or len(suffix) != 2_001
                or any(
                    re.fullmatch(r"[0-9a-f]{16}", item) is None
                    for item in suffix[1:]
                )
            ):
                raise ValueError("RFC sealed --number-bits batch drifted")
            expected: list[str] = []
            while len(expected) < 2_000:
                bits = self._finite_rng.getrandbits(64)
                value = struct.unpack(">d", bits.to_bytes(8, "big"))[0]
                if math.isfinite(value):
                    expected.append(f"{bits:016x}")
            if suffix[1:] != expected:
                raise ValueError("RFC sealed --number-bits seed sequence drifted")
            self._number_batches += 1
        else:
            raise ValueError("RFC sealed subprocess operation is not allowlisted")
        rebased = [
            f"/proc/self/fd/{self._node_fd}",
            "--input-type=module",
            "--eval",
            self._oracle_text,
            "--",
            self._oracle_display,
            *suffix,
        ]
        return subprocess.run(
            rebased,
            check=True,
            capture_output=True,
            text=True,
            pass_fds=tuple(inherited),
        )

    def assert_complete(self) -> None:
        if self._check_calls != 1 or self._number_batches != 50:
            raise ValueError("RFC sealed subprocess did not close 1+50 operations")


def _run_rfc8785_stdlib_selfcheck_entry_v2(project_root: Path) -> int:
    if (
        not isinstance(project_root, Path)
        or not project_root.is_absolute()
        or project_root.is_symlink()
        or not project_root.is_dir()
        or str(project_root.resolve(strict=True)) != str(project_root)
        or _SOURCE_ONLY_IMPORTER_ACTIVE_V2 is not True
        or _SOURCE_EXECUTION_BOOTSTRAP_ACTIVE_V2 is not True
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.dont_write_bytecode != 1
    ):
        raise ValueError("RFC selfcheck entry is not direct isolated source-only code")
    expected_source = (
        project_root
        / "src/hegel_machine/phase2b_formal_wire_covert_qualification_v2.py"
    )
    if Path(__file__).resolve(strict=True) != expected_source:
        raise ValueError("RFC selfcheck entry source differs from its project root")
    test_path = project_root / "tests/test_rfc8785_v1.py"
    test_source = _read_bounded_regular_file_v2(
        test_path, maximum_bytes=256 * 1_024, name="RFC pinned test source"
    )
    if hashlib.sha256(test_source).hexdigest() != RFC8785_TEST_SHA256:
        raise ValueError("RFC selfcheck test source hash drifted")
    oracle_path = project_root / "tools/phase2b_rfc8785_oracle_v1.mjs"
    vectors_path = project_root / "golden_vectors/phase2b_rfc8785_v1.json"
    oracle_bytes = _read_bounded_regular_file_v2(
        oracle_path,
        maximum_bytes=256 * 1_024,
        name="RFC selfcheck oracle source",
    )
    vectors_bytes = _read_bounded_regular_file_v2(
        vectors_path,
        maximum_bytes=256 * 1_024,
        name="RFC selfcheck golden vectors",
    )
    if (
        hashlib.sha256(oracle_bytes).hexdigest() != RFC8785_NODE_ORACLE_SHA256
        or hashlib.sha256(vectors_bytes).hexdigest()
        != RFC8785_GOLDEN_VECTORS_SHA256
    ):
        raise ValueError("RFC selfcheck sealed data bytes drifted")
    pytest_stub = ModuleType("pytest")
    pytest_stub.raises = lambda expected, match=None: _RFC8785ExpectedExceptionV2(  # type: ignore[attr-defined]
        expected, match
    )
    pytest_stub.mark = _RFC8785MarkStubV2()  # type: ignore[attr-defined]
    test_module = ModuleType("_hegel_rfc8785_pinned_test_v2")
    test_module.__file__ = str(test_path)
    prior_pytest = sys.modules.get("pytest")
    sys.modules["pytest"] = pytest_stub
    try:
        exec(
            compile(test_source, str(test_path), "exec", dont_inherit=True),
            test_module.__dict__,
        )
    finally:
        if prior_pytest is None:
            del sys.modules["pytest"]
        else:
            sys.modules["pytest"] = prior_pytest
    ordered_names = (
        "test_official_rfc8785_appendix_sample_and_utf8_bytes",
        "test_official_utf16_code_unit_property_sorting",
        "test_official_appendix_b_all_binary64_vectors",
        "test_independent_literal_vectors",
        "test_utf16_order_really_differs_from_python_codepoint_order",
        "test_unicode_is_preserved_without_normalization_or_unneeded_escaping",
        "test_surrogates_and_ijson_noncharacters_are_rejected",
        "test_programmatic_number_domain_and_ecmascript_thresholds",
        "test_decode_and_require_canonical_accepts_only_exact_bytes",
        "test_duplicate_names_are_rejected_after_escape_processing",
        "test_json_data_model_is_closed_and_cycles_fail",
        "test_node12_v8_oracle_passes_frozen_vectors_without_python_import",
        "test_fixed_seed_100000_finite_binary64_values_match_node_v8",
    )
    node_lookup = shutil.which("node")
    if node_lookup is None:
        raise ValueError("RFC selfcheck cannot locate Node")
    node_realpath = Path(node_lookup).resolve(strict=True)
    node_security = _executable_path_security_projection_v2(node_realpath)
    if node_security.get("root_owned_no_group_or_other_write") is not True:
        raise ValueError("RFC selfcheck Node path is not immutable to the caller")
    node_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW
    node_fd = os.open(node_realpath, node_flags)
    vectors_fd: int | None = None
    try:
        opened_node = os.fstat(node_fd)
        if (
            not stat.S_ISREG(opened_node.st_mode)
            or os.path.realpath(f"/proc/self/fd/{node_fd}") != str(node_realpath)
        ):
            raise ValueError("RFC selfcheck Node descriptor drifted")
        vectors_fd = _open_sealed_memfd_v2(
            vectors_bytes, "hegel-rfc8785-selfcheck-vectors"
        )
        test_module.VECTORS_PATH = _RFC8785SealedTextPathV2(
            vectors_path, vectors_bytes
        )
        test_module.ORACLE_PATH = _RFC8785SealedTextPathV2(
            oracle_path, oracle_bytes
        )
        sealed_subprocess = _RFC8785SealedSubprocessV2(
            node_fd=node_fd,
            oracle_display=oracle_path,
            oracle_bytes=oracle_bytes,
            vectors_display=vectors_path,
            vectors_fd=vectors_fd,
        )
        test_module.subprocess = sealed_subprocess
        discovered = {
            name
            for name, value in vars(test_module).items()
            if name.startswith("test_") and callable(value)
        }
        if discovered != set(ordered_names):
            raise ValueError("RFC selfcheck discovered test set drifted")
        passed = 0
        for name in ordered_names:
            function = getattr(test_module, name)
            if name == "test_surrogates_and_ijson_noncharacters_are_rejected":
                values = getattr(function, "__hegel_rfc8785_parameters_v2__", None)
                expected_values = (
                    "\ud800",
                    "\udead",
                    "\ufdd0",
                    "\ufffe",
                    "\U0001ffff",
                    "\U0010ffff",
                )
                if values != expected_values:
                    raise ValueError("RFC selfcheck parameter vector drifted")
                for value in values:
                    function(value)
                    passed += 1
            else:
                function()
                passed += 1
        sealed_subprocess.assert_complete()
    finally:
        _close_owned_descriptors_v2(
            (vectors_fd, node_fd), name="RFC selfcheck sealed execution"
        )
    if passed != 18:
        raise ValueError("RFC selfcheck did not execute exactly 18 checks")
    sys.stdout.buffer.write(b".................. [100%]\n")
    sys.stdout.buffer.flush()
    return 0


def _argparse_exact_positive_decimal_v2(value: str) -> int:
    if re.fullmatch(r"[1-9][0-9]{0,18}", value) is None:
        raise argparse.ArgumentTypeError("value must be one exact positive decimal")
    parsed = int(value, 10)
    if parsed > 2**63 - 1:
        raise argparse.ArgumentTypeError("value exceeds the signed 64-bit bound")
    return parsed


def _require_cli_argument_shape_v2(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    expected_names: set[str],
) -> None:
    actual_names = {
        name
        for name, value in vars(args).items()
        if name != "entry" and value is not None
    }
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        forbidden = sorted(actual_names - expected_names)
        parser.error(
            "entry argument set is not exact; "
            f"missing={missing!r}; forbidden={forbidden!r}"
        )


def _build_development_attempt_manifest_bytes_v2(
    *,
    campaign_id: str,
    attempt_id: str,
    source_commit_sha: str,
    source_archive: Path,
    runtime_environment_manifest: Path,
    rfc8785_prerequisite_receipt: Path,
    requested_cgroup_memory_max_bytes: int,
    requested_cgroup_pids_max: int,
    project_root: Path,
) -> bytes:
    """Build one pending-development attempt after replaying every prerequisite."""

    for name, value in (("campaign_id", campaign_id), ("attempt_id", attempt_id)):
        if type(value) is not str or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value
        ) is None:
            raise ValueError(f"development attempt {name} is outside grammar")
    if type(source_commit_sha) is not str or re.fullmatch(
        r"[0-9a-f]{40}", source_commit_sha
    ) is None:
        raise ValueError("development attempt source commit is not exact Git SHA-1")
    if (
        type(requested_cgroup_memory_max_bytes) is not int
        or requested_cgroup_memory_max_bytes
        != OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2
        or type(requested_cgroup_pids_max) is not int
        or not 2 <= requested_cgroup_pids_max <= 4_096
    ):
        raise ValueError("development attempt cgroup limits are outside bounds")
    deadline_ns = time.monotonic_ns() + 600_000_000_000
    qualification_source_sha256 = _verify_qualified_cli_source_bootstrap_v2(
        project_root
    )
    for name, path in (
        ("source archive", source_archive),
        ("runtime environment manifest", runtime_environment_manifest),
        ("RFC-8785 prerequisite receipt", rfc8785_prerequisite_receipt),
    ):
        if (
            not isinstance(path, Path)
            or not path.is_absolute()
            or path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True) != path
        ):
            raise ValueError(f"development attempt {name} is not one canonical file")
    if (
        rfc8785_prerequisite_receipt.name != "receipt.json"
        or rfc8785_prerequisite_receipt.parent.is_symlink()
        or rfc8785_prerequisite_receipt.parent.resolve(strict=True)
        != rfc8785_prerequisite_receipt.parent
    ):
        raise ValueError(
            "development attempt RFC receipt is not the canonical raw-bundle receipt"
        )

    transaction_sha256: dict[str, str] = {}
    for role, relative in _QUALIFICATION_TRANSACTION_RELATIVE_PATHS_V2.items():
        if time.monotonic_ns() >= deadline_ns:
            raise TimeoutError("development attempt project hash reached deadline")
        candidate = project_root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"development attempt {role} file is absent or a symlink")
        transaction_sha256[role] = _sha256_file_v2(
            candidate, deadline_ns=deadline_ns
        )
    if transaction_sha256["source"] != qualification_source_sha256:
        raise ValueError("development attempt source changed after bootstrap")

    profile_path = project_root / _QUALIFICATION_TRANSACTION_RELATIVE_PATHS_V2["config"]
    profile_bytes = _read_bounded_regular_file_v2(
        profile_path,
        maximum_bytes=_MAXIMUM_PROFILE_BYTES,
        name="development attempt qualification profile",
        deadline_ns=deadline_ns,
    )
    if (
        profile_bytes != phase2b_formal_wire_covert_qualification_profile_bytes_v2()
        or hashlib.sha256(profile_bytes).hexdigest() != transaction_sha256["config"]
    ):
        raise ValueError("development attempt profile bytes differ from this source")
    profile = load_phase2b_formal_wire_covert_qualification_profile_v2(profile_bytes)
    _require_pending_calibration_profile_v2(
        _deep_thaw(profile.profile)["fixture_contract"]["rich_positive_shape"]
    )

    lock_path = project_root / _QUALIFICATION_TRANSACTION_RELATIVE_PATHS_V2["lock"]
    lock_bytes = _read_bounded_regular_file_v2(
        lock_path,
        maximum_bytes=256 * 1_024,
        name="development attempt dependency lock",
        deadline_ns=deadline_ns,
    )
    _verify_dependency_lock_bytes_v2(lock_bytes)
    if hashlib.sha256(lock_bytes).hexdigest() != transaction_sha256["lock"]:
        raise ValueError("development attempt lock hash replay drifted")

    archive_stat = source_archive.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(archive_stat.st_mode)
        or not 1 <= archive_stat.st_size <= 512 * 1_024 * 1_024
    ):
        raise ValueError("development attempt source archive size/type is invalid")
    archive_sha256 = _sha256_file_v2(source_archive, deadline_ns=deadline_ns)
    (
        _archive_regular_root,
        _archive_regular_count,
        _archive_regular_bytes,
        _archive_member_count,
        _archive_directory_count,
        archive_tree_rows,
        _archive_tree_root,
    ) = _inspect_source_archive_regular_manifest_v2(
        source_archive,
        expected_sha256=archive_sha256,
        expected_size=archive_stat.st_size,
        expected_source_commit_sha=source_commit_sha,
        deadline_ns=deadline_ns,
    )

    runtime_bytes = _read_bounded_regular_file_v2(
        runtime_environment_manifest,
        maximum_bytes=1_048_576,
        name="development attempt runtime environment manifest",
        deadline_ns=deadline_ns,
    )
    _decode_canonical_transport_object_v2(
        runtime_bytes,
        "development attempt runtime environment manifest",
        maximum_bytes=1_048_576,
    )
    runtime_sha256 = hashlib.sha256(runtime_bytes).hexdigest()

    rfc_bytes = _read_bounded_regular_file_v2(
        rfc8785_prerequisite_receipt,
        maximum_bytes=1_048_576,
        name="development attempt RFC-8785 prerequisite receipt",
        deadline_ns=deadline_ns,
    )
    rfc_sha256 = hashlib.sha256(rfc_bytes).hexdigest()
    attempt = Phase2BFormalWireCovertAttemptManifestV2(
        campaign_id,
        attempt_id,
        profile.profile_id,
        source_commit_sha,
        "sha256:" + archive_sha256,
        archive_stat.st_size,
        DEPENDENCY_LOCK_SHA256,
        transaction_sha256["source"],
        transaction_sha256["config"],
        transaction_sha256["lock"],
        transaction_sha256["test"],
        transaction_sha256["documentation"],
        "sha256:" + runtime_sha256,
        len(runtime_bytes),
        None,
        "sha256:" + rfc_sha256,
        rfc_sha256,
        len(rfc_bytes),
        LIVE_MODE_ORDER,
        LIVE_HARD_TOTAL_TIMEOUT_SECONDS,
        requested_cgroup_memory_max_bytes,
        requested_cgroup_pids_max,
    )

    archive_regular_rows = {
        row["path"]: row
        for row in archive_tree_rows
        if row.get("type") == "regular"
    }
    expected_files = _expected_live_file_sha256_v2(attempt)
    if not set(expected_files).issubset(archive_regular_rows):
        raise ValueError("development attempt archive omits a pinned live file")
    for relative, expected_sha256 in expected_files.items():
        archive_row = archive_regular_rows[relative]
        if archive_row.get("sha256") != expected_sha256:
            raise ValueError("development attempt archive pinned-file hash drifted")
        live_path = project_root / relative
        if (
            live_path.is_symlink()
            or not live_path.is_file()
            or _sha256_file_v2(live_path, deadline_ns=deadline_ns)
            != expected_sha256
        ):
            raise ValueError("development attempt project-root pinned file drifted")
    fresh_runtime_bytes = phase2b_formal_wire_covert_runtime_environment_manifest_bytes_v2()
    if runtime_bytes != fresh_runtime_bytes:
        raise ValueError(
            "development attempt runtime manifest differs from fresh host evidence"
        )
    rfc_receipt = _verify_rfc8785_prerequisite_receipt_v2(
        rfc_bytes,
        attempt,
        verify_live_executables=True,
        deadline_ns=deadline_ns,
    )
    _verify_rfc8785_raw_execution_bundle_v2(
        rfc_receipt,
        rfc8785_prerequisite_receipt.parent,
        expected_qualification_source_sha256=qualification_source_sha256,
        replay_project_root=project_root,
        fresh_execute=False,
        deadline_ns=deadline_ns,
    )
    if time.monotonic_ns() >= deadline_ns:
        raise TimeoutError("development attempt final serialization reached deadline")
    return phase2b_formal_wire_covert_attempt_manifest_bytes_v2(attempt)


def _main_v2(argv: tuple[str, ...] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -I -S -B /absolute/phase2b_formal_wire_covert_qualification_v2.py",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--entry",
        required=True,
        choices=(
            "development-calibration",
            "live-inner",
            "outer-development-calibration",
            "outer-live",
            "build-runtime-environment-manifest",
            "build-development-attempt-manifest",
            "build-rfc8785-prerequisite",
            "rfc-selfcheck",
            "rfc-selfcheck-version",
        ),
    )
    parser.add_argument("--campaign-id")
    parser.add_argument("--attempt-id")
    parser.add_argument("--source-commit-sha")
    parser.add_argument(
        "--cgroup-memory-max-bytes", type=_argparse_exact_positive_decimal_v2
    )
    parser.add_argument("--cgroup-pids-max", type=_argparse_exact_positive_decimal_v2)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--rfc8785-prerequisite-receipt", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--runtime-environment-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--calibration-attempt-directory", type=Path)
    parser.add_argument("--calibration-basis-profile", type=Path)
    parser.add_argument("--calibration-attempt-manifest", type=Path)
    parser.add_argument("--calibration-inner-artifact", type=Path)
    parser.add_argument("--calibration-outer-supervisor-result", type=Path)
    parser.add_argument("--calibration-rfc8785-prerequisite-receipt", type=Path)
    parser.add_argument("--calibration-mechanics-artifact", type=Path)
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    seen_options: set[str] = set()
    for token in raw_argv:
        if token.startswith("--"):
            if "=" in token:
                parser.error("CLI long options require one separate value token")
            if token in seen_options:
                parser.error(f"CLI option is repeated: {token}")
            seen_options.add(token)
    args = parser.parse_args(raw_argv)
    for name, value in vars(args).items():
        if name == "entry" or value is None:
            continue
        if isinstance(value, Path) and not value.is_absolute():
            parser.error(f"--{name.replace('_', '-')} must name an absolute path")

    expected_argument_names = {
        "rfc-selfcheck-version": {"project_root"},
        "rfc-selfcheck": {"project_root"},
        "build-rfc8785-prerequisite": {"project_root", "output_root"},
        "build-runtime-environment-manifest": {"project_root"},
        "build-development-attempt-manifest": {
            "attempt_id",
            "campaign_id",
            "cgroup_memory_max_bytes",
            "cgroup_pids_max",
            "project_root",
            "rfc8785_prerequisite_receipt",
            "runtime_environment_manifest",
            "source_archive",
            "source_commit_sha",
        },
        "development-calibration": {"attempt", "profile"},
        "live-inner": {"attempt", "profile"},
        "outer-development-calibration": {
            "attempt",
            "output_root",
            "profile",
            "rfc8785_prerequisite_receipt",
            "runtime_environment_manifest",
            "source_archive",
        },
        "outer-live": {
            "attempt",
            "calibration_attempt_directory",
            "output_root",
            "profile",
            "rfc8785_prerequisite_receipt",
            "runtime_environment_manifest",
            "source_archive",
        },
    }
    _require_cli_argument_shape_v2(
        parser, args, expected_argument_names[args.entry]
    )
    bootstrap_project_root = (
        args.project_root
        if args.project_root is not None
        else Path(__file__).resolve(strict=True).parents[2]
    )
    bootstrap_source_sha256 = _verify_qualified_cli_source_bootstrap_v2(
        bootstrap_project_root
    )

    preflight_profile_bytes: bytes | None = None
    preflight_attempt_bytes: bytes | None = None
    preflight_attempt: Phase2BFormalWireCovertAttemptManifestV2 | None = None
    if args.attempt is not None:
        preflight_attempt_bytes = _read_bounded_regular_file_v2(
            args.attempt,
            maximum_bytes=256 * 1_024,
            name="CLI preflight attempt input",
        )
        preflight_attempt = load_phase2b_formal_wire_covert_attempt_manifest_v2(
            preflight_attempt_bytes
        )
        if preflight_attempt.qualification_source_sha256 != bootstrap_source_sha256:
            raise ValueError(
                "CLI executed source bytes differ from the attempt manifest"
            )
        if args.profile is None:
            raise ValueError("CLI attempt entry lacks its profile bytes")
        preflight_profile_bytes = _read_bounded_regular_file_v2(
            args.profile,
            maximum_bytes=_MAXIMUM_PROFILE_BYTES,
            name="CLI preflight profile input",
        )
        if (
            hashlib.sha256(preflight_profile_bytes).hexdigest()
            != preflight_attempt.qualification_config_sha256
        ):
            raise ValueError("CLI profile bytes differ from the attempt manifest")
        load_phase2b_formal_wire_covert_qualification_profile_v2(
            preflight_profile_bytes
        )

    if args.entry == "rfc-selfcheck-version":
        sys.stdout.write("hegel-rfc8785-stdlib-selfcheck-v1\n")
        sys.stdout.flush()
        return 0
    if args.entry == "rfc-selfcheck":
        return _run_rfc8785_stdlib_selfcheck_entry_v2(args.project_root)
    if args.entry == "build-rfc8785-prerequisite":
        artifact = build_rfc8785_prerequisite_bundle_v2(
            project_root=args.project_root,
            bundle_directory=args.output_root,
        )
        sys.stdout.buffer.write(artifact)
        sys.stdout.buffer.flush()
        return 0
    if args.entry == "build-runtime-environment-manifest":
        artifact = phase2b_formal_wire_covert_runtime_environment_manifest_bytes_v2()
        sys.stdout.buffer.write(artifact)
        sys.stdout.buffer.flush()
        return 0
    if args.entry == "build-development-attempt-manifest":
        artifact = _build_development_attempt_manifest_bytes_v2(
            campaign_id=args.campaign_id,
            attempt_id=args.attempt_id,
            source_commit_sha=args.source_commit_sha,
            source_archive=args.source_archive,
            runtime_environment_manifest=args.runtime_environment_manifest,
            rfc8785_prerequisite_receipt=args.rfc8785_prerequisite_receipt,
            requested_cgroup_memory_max_bytes=args.cgroup_memory_max_bytes,
            requested_cgroup_pids_max=args.cgroup_pids_max,
            project_root=args.project_root,
        )
        sys.stdout.buffer.write(artifact)
        sys.stdout.buffer.flush()
        return 0
    if args.entry.startswith("outer-"):
        calibration_names = (
            "calibration_basis_profile",
            "calibration_attempt_manifest",
            "calibration_inner_artifact",
            "calibration_outer_supervisor_result",
            "calibration_rfc8785_prerequisite_receipt",
            "calibration_mechanics_artifact",
        )
        if args.entry == "outer-development-calibration" and (
            args.calibration_attempt_directory is not None
            or any(getattr(args, name) is not None for name in calibration_names)
        ):
            parser.error("outer development calibration cannot consume prior calibration")
        if args.entry == "outer-live" and (
            args.calibration_attempt_directory is None
            or any(getattr(args, name) is not None for name in calibration_names)
        ):
            parser.error(
                "outer live requires one retained calibration attempt directory; "
                "loose calibration artifact paths are forbidden"
            )
        artifact = run_phase2b_formal_wire_covert_outer_attempt_v2(
            entry_kind=(
                "development-calibration"
                if args.entry == "outer-development-calibration"
                else "live-inner"
            ),
            profile_path=args.profile,
            attempt_manifest_path=args.attempt,
            rfc8785_prerequisite_receipt_path=args.rfc8785_prerequisite_receipt,
            source_archive_path=args.source_archive,
            runtime_environment_manifest_path=args.runtime_environment_manifest,
            output_root=args.output_root,
            calibration_attempt_directory_path=args.calibration_attempt_directory,
            calibration_basis_profile_path=args.calibration_basis_profile,
            calibration_attempt_manifest_path=args.calibration_attempt_manifest,
            calibration_inner_artifact_path=args.calibration_inner_artifact,
            calibration_outer_supervisor_result_path=(
                args.calibration_outer_supervisor_result
            ),
            calibration_rfc8785_prerequisite_receipt_path=(
                args.calibration_rfc8785_prerequisite_receipt
            ),
            calibration_mechanics_artifact_path=args.calibration_mechanics_artifact,
        )
        sys.stdout.buffer.write(artifact)
        sys.stdout.buffer.flush()
        return 0
    if preflight_profile_bytes is None or preflight_attempt_bytes is None:
        raise RuntimeError("CLI inner preflight did not retain exact inputs")
    profile_bytes = preflight_profile_bytes
    attempt_bytes = preflight_attempt_bytes
    profile = load_phase2b_formal_wire_covert_qualification_profile_v2(
        profile_bytes
    )
    attempt = load_phase2b_formal_wire_covert_attempt_manifest_v2(attempt_bytes)
    if args.entry == "development-calibration":
        artifact = build_development_calibration_artifact_v2(profile, attempt)
    else:
        artifact = build_live_inner_candidate_artifact_v2(profile, attempt)
    _write_flush_gate_and_direct_exit_v2(artifact, attempt)


__all__ = [
    "AUDITOR_POLICY_BINDING_ID_V2",
    "AUDITOR_PROFILE_SCHEMA_VERSION_V2",
    "AUDITOR_SCHEMA_VERSION_V2",
    "AUDITOR_SEMANTICS_ID_V2",
    "AuditRunDiagnosticsV2",
    "DEPENDENCY_LOCK_SHA256",
    "DYNAMIC_RESOURCE_PLAN_FIELDS_NOT_EXACT_COMPARED_V2",
    "EXACT_AUTHORITY_ROWS",
    "EXACT_LABEL_PERMUTATIONS",
    "EXACT_RECOGNIZER_INVOCATIONS",
    "EXACT_ROWS_PER_TRANSFORM_OPERATION",
    "EXACT_TRANSFORM_OPERATIONS",
    "FROZEN_STATIC_RESOURCE_PLAN_FIELDS_V2",
    "LIVE_HARD_TOTAL_TIMEOUT_SECONDS",
    "LIVE_MODE_ORDER",
    "OPERATIONAL_HARD_MEMORY_MAX_BYTES_V2",
    "PROFILE_CLAIM_LEVEL",
    "PROFILE_ID",
    "PROFILE_SCHEMA_VERSION",
    "Phase2BFormalWireCovertCandidateReceiptV2",
    "Phase2BFormalWireCovertAttemptManifestV2",
    "Phase2BFormalWireCovertQualificationErrorV2",
    "Phase2BFormalWireCovertQualificationProfileV2",
    "Phase2BFormalWireCovertQualificationRejectionV2",
    "QualificationDispositionV2",
    "build_development_calibration_artifact_v2",
    "build_live_inner_candidate_artifact_v2",
    "build_rfc8785_prerequisite_bundle_v2",
    "inspect_current_live_cgroup_v2",
    "load_phase2b_formal_wire_covert_attempt_manifest_v2",
    "load_phase2b_formal_wire_covert_qualification_profile_v2",
    "phase2b_formal_wire_covert_qualification_profile_bytes_v2",
    "phase2b_formal_wire_covert_attempt_manifest_bytes_v2",
    "prepare_live_inner_process_v2",
    "phase2b_formal_wire_covert_runtime_environment_manifest_bytes_v2",
    "qualification_profile_id_v2",
    "qualify_phase2b_formal_wire_covert_candidate_v2",
    "run_phase2b_formal_wire_covert_outer_attempt_v2",
    "verify_phase2b_formal_wire_covert_outer_mechanics_v2",
]


if __name__ == "__main__":
    raise SystemExit(_main_v2())
