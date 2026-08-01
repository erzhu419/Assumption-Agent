"""Executable trust boundary for the GSCL/ARN intrinsic harness.

This module does not claim that an implementation is effective and it does not
add a measurement gate.  It replaces the older, caller-asserted
``freeze_ready``/capability receipts with one executable supervisor boundary:

* formal state lives below one compiled-in root;
* every state operation is relative to a held directory descriptor and rejects
  symlink components;
* source bytes are verified and parsed from the same descriptor read;
* a source/action/freeze-derived marker makes the formal invocation one-shot;
* arm outputs can only enter the formal path after a real subprocess completed
  inside a Landlock filesystem sandbox and passed real label/linkage denial
  probes; and
* the fixed label custodian is only invoked after all four prediction packs
  have been sealed.

The supervisor and its frozen code/commands are trusted.  Landlock constrains
the arm processes; it is not presented as protection from a malicious
supervisor, kernel, root user, ptrace-capable peer, or network side channel.
Item-level packs and predictions remain private.  Tests exercise only
synthetic, source-free fixtures and never open the official ARN release.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import socket
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

from assumption_agent import gscl_arn_raw_adapter_v1 as raw_adapter
from assumption_agent.benchmarks import gscl_arn_intrinsic_protocol_v1 as protocol


VERSION = "gscl_arn_formal_supervisor_v1"
FORMAL_ROOT = Path("/var/tmp/gscl_arn_intrinsic_formal_v1")
FORMAL_SOURCE_DATASET_RELATIVE = "source/arn.csv"
FORMAL_SOURCE_METADATA_RELATIVE = "source/metadata.json"
_RECONSTRUCTION_ROOT = Path(__file__).resolve().parents[2]
_WORKSPACE_ROOT = _RECONSTRUCTION_ROOT.parent
_LOCAL_PYTHONPATH_ROOTS = (
    _WORKSPACE_ROOT,
    _RECONSTRUCTION_ROOT,
)
_LOCAL_PYTHONPATH = os.pathsep.join(
    str(path) for path in _LOCAL_PYTHONPATH_ROOTS
)
_INTERNAL_FORMAL_IMPLEMENTATION_PATHS = {
    "supervisor": Path(__file__).resolve(),
    "qualification_runner": (
        _RECONSTRUCTION_ROOT
        / (
            "assumption_agent/benchmarks/"
            "gscl_arn_internal_factory_qualification_v1.py"
        )
    ),
    "item_factory": (
        _RECONSTRUCTION_ROOT
        / "assumption_agent/benchmarks/gscl_arn_formal_item_factory_v1.py"
    ),
    "raw_adapter": (
        _RECONSTRUCTION_ROOT
        / "assumption_agent/gscl_arn_raw_adapter_v1.py"
    ),
    "narrative_core": (
        _RECONSTRUCTION_ROOT
        / "assumption_agent/gscl_narrative_correspondence_v1.py"
    ),
    "intrinsic_arms": (
        _RECONSTRUCTION_ROOT
        / "assumption_agent/gscl_arn_intrinsic_arms_v1.py"
    ),
    "intrinsic_scorers": (
        _RECONSTRUCTION_ROOT
        / "assumption_agent/gscl_arn_intrinsic_scorers_v1.py"
    ),
    "extractor_contract": (
        _RECONSTRUCTION_ROOT
        / "replication_runtime/gscl_narrative_extractor_v1/contract.py"
    ),
    "extractor_init": (
        _RECONSTRUCTION_ROOT
        / "replication_runtime/gscl_narrative_extractor_v1/__init__.py"
    ),
    "extractor_worker": (
        _RECONSTRUCTION_ROOT
        / "replication_runtime/gscl_narrative_extractor_v1/worker.py"
    ),
    "extractor_multi_pack_support": (
        _RECONSTRUCTION_ROOT
        / (
            "replication_runtime/gscl_narrative_extractor_v1/"
            "multi_pack_worker.py"
        )
    ),
    "extractor_closed_choice_worker": (
        _RECONSTRUCTION_ROOT
        / (
            "replication_runtime/gscl_narrative_extractor_v1/"
            "closed_choice_worker.py"
        )
    ),
    "extractor_closed_choice_qwen_runtime": (
        _RECONSTRUCTION_ROOT
        / (
            "replication_runtime/gscl_narrative_extractor_v1/"
            "closed_choice_qwen_runtime.py"
        )
    ),
    "extractor_multi_pack_worker": (
        _RECONSTRUCTION_ROOT
        / (
            "replication_runtime/gscl_narrative_extractor_v1/"
            "closed_choice_multi_pack_worker.py"
        )
    ),
    "gscl_minilm_init": (
        _RECONSTRUCTION_ROOT
        / "replication_runtime/gscl_minilm_portable_v1/__init__.py"
    ),
    "gscl_minilm_binding": (
        _RECONSTRUCTION_ROOT
        / "replication_runtime/gscl_minilm_portable_v1/binding.py"
    ),
    "gscl_minilm_target_qualification": (
        _RECONSTRUCTION_ROOT
        / (
            "replication_runtime/gscl_minilm_portable_v1/"
            "target_qualification.py"
        )
    ),
    "portable_minilm_binding": (
        _RECONSTRUCTION_ROOT
        / "replication_runtime/qasper_minilm_portable_v2/binding.py"
    ),
    "base_minilm_binding": (
        _RECONSTRUCTION_ROOT
        / "replication_runtime/qasper_minilm_v1/binding.py"
    ),
    "protocol": (
        _RECONSTRUCTION_ROOT
        / "assumption_agent/benchmarks/gscl_arn_intrinsic_protocol_v1.py"
    ),
}
_INTERNAL_QUALIFICATION_TEST_PATHS = {
    "qualification_runner_test": (
        _RECONSTRUCTION_ROOT
        / (
            "tests/"
            "test_gscl_arn_internal_factory_qualification_v1.py"
        )
    ),
    "extractor_runtime_test": (
        _RECONSTRUCTION_ROOT
        / "tests/test_gscl_narrative_extractor_runtime_v1.py"
    ),
    "extractor_multi_pack_runtime_test": (
        _RECONSTRUCTION_ROOT
        / (
            "tests/"
            "test_gscl_narrative_extractor_multi_pack_worker_v1.py"
        )
    ),
    "extractor_closed_choice_runtime_test": (
        _RECONSTRUCTION_ROOT
        / "tests/test_gscl_narrative_closed_choice_worker_v1.py"
    ),
    "extractor_closed_choice_multi_pack_runtime_test": (
        _RECONSTRUCTION_ROOT
        / (
            "tests/"
            "test_gscl_narrative_closed_choice_multi_pack_worker_v1.py"
        )
    ),
    "minilm_portable_runtime_test": (
        _RECONSTRUCTION_ROOT
        / "tests/test_gscl_minilm_portable_v1.py"
    ),
    "minilm_target_qualification_test": (
        _RECONSTRUCTION_ROOT
        / "tests/test_gscl_minilm_target_qualification_v1.py"
    ),
    "formal_item_factory_test": (
        _RECONSTRUCTION_ROOT
        / "tests/test_gscl_arn_formal_item_factory_v1.py"
    ),
    "formal_supervisor_test": (
        _RECONSTRUCTION_ROOT
        / "tests/test_gscl_arn_formal_supervisor_v1.py"
    ),
    "intrinsic_arms_test": (
        _RECONSTRUCTION_ROOT
        / "tests/test_gscl_arn_intrinsic_arms_v1.py"
    ),
    "intrinsic_scorers_test": (
        _RECONSTRUCTION_ROOT
        / "tests/test_gscl_arn_intrinsic_scorers_v1.py"
    ),
    "narrative_correspondence_test": (
        _RECONSTRUCTION_ROOT
        / "tests/test_gscl_narrative_correspondence_v1.py"
    ),
    "intrinsic_protocol_test": (
        _RECONSTRUCTION_ROOT
        / "tests/test_gscl_arn_intrinsic_protocol_v1.py"
    ),
    "raw_adapter_test": (
        _RECONSTRUCTION_ROOT
        / "tests/test_gscl_arn_raw_adapter_v1.py"
    ),
}
_INTERNAL_SUPPORT_MODULE_PATHS = {
    "assumption_os": _WORKSPACE_ROOT / "assumption_os/__init__.py",
    "assumption_os.graph_memory": (
        _WORKSPACE_ROOT / "assumption_os/graph_memory.py"
    ),
    "assumption_os.schema": (
        _WORKSPACE_ROOT / "assumption_os/schema.py"
    ),
    "assumption_os.selector": (
        _WORKSPACE_ROOT / "assumption_os/selector.py"
    ),
    "assumption_os.formal_mapping": (
        _WORKSPACE_ROOT / "assumption_os/formal_mapping.py"
    ),
    "assumption_os.structural_patterns": (
        _WORKSPACE_ROOT / "assumption_os/structural_patterns.py"
    ),
}
_STABLE_SUPPORT_MODULE_SHA256S = {
    "assumption_os": (
        "73c6ed77abdeec0cbd2d3554c4cb7adae2c79f95018a5d17df10d9e5647ec237"
    ),
    "assumption_os.graph_memory": (
        "7e02d853999ac526a34958d4534cd97cf7ec711698f0601c6dd8aed32e40bd51"
    ),
    "assumption_os.schema": (
        "9423aa791f232613d6979a21c82efcbec07caa272a481f3adc8e8672c3a13928"
    ),
    "assumption_os.selector": (
        "91dac54c3d65c086998d9014b4c4550fea9784f37fac92bacb99061c67593f0e"
    ),
    "assumption_os.formal_mapping": (
        "6c042ffc06cd03119401c3f8946120caf675c4596247d67d0db632827193628f"
    ),
    "assumption_os.structural_patterns": (
        "64602b951cb53e3d4baad588668ab4ac7b9d91fcbe6ac8e6bf6d7460830dad93"
    ),
}
_STABLE_EXTRACTOR_SHA256S = {
    "extractor_init": (
        "3c05e6098090bb61cc1839a8810df259f29f3b84a389c63537be60aba035bae4"
    ),
    "extractor_contract": (
        "4e4c8aed46dd00a5a5b0b6cf6a026df5c4cbcd87455105bc49db3c0276c883c4"
    ),
    "extractor_worker": (
        "e0f5606da1c0cbc9ba5d74d758cb24b22102926fe8fc078e8b218445ec19271c"
    ),
    "extractor_multi_pack_support": (
        "40cafb033d3e5c5f22ce6b952c533eaa955ea97e0ba4a7becbb76a7f0a89896c"
    ),
    "extractor_closed_choice_worker": (
        "dcd871d18c804fbcedffe14bfa3f571b622738eb1f535274dfb296b2010f6d31"
    ),
    "extractor_closed_choice_qwen_runtime": (
        "e116a71ac1f5794edf6792129a72f0a70802ad1a5d148d133e0a86ed06d59681"
    ),
    "extractor_multi_pack_worker": (
        "bfc706118f94e1fc2a6c498ab3a9146c2e2629ae01efb69fd205e687423d3135"
    ),
}
_STABLE_QUALIFICATION_TEST_SHA256S = {
    "qualification_runner_test": (
        "88ead737a48a178c4c236d2d0510d268dd0fc8e8d230c6fa1acc47ef4a881393"
    ),
    "extractor_runtime_test": (
        "89b67c96ad8439ecd42d555d7bffad076bd31adbd46b4a56a4f15a6b809adc4a"
    ),
    "extractor_multi_pack_runtime_test": (
        "ca6f627ec9780eefc98194a933c7b7f2ef1af50b01775fbc635b13f8df412a02"
    ),
    "extractor_closed_choice_runtime_test": (
        "a9d6df27b50561c521190546b41d29cf1f38bc23c50e76bfe9f050851cd69e9c"
    ),
    "extractor_closed_choice_multi_pack_runtime_test": (
        "6222f21ea433fb58e4d13c3735e6fba21bc3c465215e9b7c4eb17724efdc4fb3"
    ),
    "minilm_portable_runtime_test": (
        "3f1be688b84e5726cc5b56371e6bd5dee569a27cdce774eb0f701bc5288dd444"
    ),
    "minilm_target_qualification_test": (
        "5203cc34cf5ed7efb9a7b825c4640f957746852574bb88301533653936b21ab4"
    ),
    "formal_item_factory_test": (
        "9c50888098753b0199f8030285c4e3f52b3c224c0c8ebc51447c4001596d779b"
    ),
    "formal_supervisor_test": (
        "3467c5d7f5fc3ddcd66d142bdece4591d6eb48c7d77e6d21cb77e57519cdb24d"
    ),
    "intrinsic_arms_test": (
        "c153960ca752e78e9c567e88986bfcacbda87c1ceb223cb6aa69a9464b1851be"
    ),
    "intrinsic_scorers_test": (
        "5c2557400c92f54c0a10513686a0e12da427543ebc2ce2e60230443c65220aeb"
    ),
    "narrative_correspondence_test": (
        "0791d116a77cd0aada4ba3563e84b091d4abe657e22d5721e7514af5e149052e"
    ),
    "intrinsic_protocol_test": (
        "33f98b6a31dd181a4d5706d3cefd287303dfbda4465081bc6318343ac59e30d2"
    ),
    "raw_adapter_test": (
        "cc89d22a3993823f62afad7ec5c8387116d9eaecc5d3886b681f1b15c795559d"
    ),
}
_STABLE_MINILM_SHA256S = {
    "gscl_minilm_init": (
        "340101410731b8ded48c70bbc72a9eb275250fc42c8a7bb2ba9f40b93cd297d1"
    ),
    "gscl_minilm_binding": (
        "3000d2dca3348ecf8f17ca14fb7b0ba4a22f1dd2ccea4bbd3c32e14a348aa674"
    ),
    "gscl_minilm_target_qualification": (
        "553c563dd8fa3b3b56c424de33f3c70a44e6e3e50555d8d5caf4e4ad9e64726d"
    ),
}

ACTION_SCHEMA = f"{VERSION}.action_freeze.v1"
ATTEMPT_SCHEMA = f"{VERSION}.one_shot_attempt.v1"
SOURCE_SCHEMA = f"{VERSION}.same_fd_source_receipt.v1"
PACK_BARRIER_SCHEMA = f"{VERSION}.private_pack_barrier.v1"
SANDBOX_SPEC_SCHEMA = f"{VERSION}.sandbox_spec.v1"
SANDBOX_RECEIPT_SCHEMA = f"{VERSION}.sandbox_receipt.v1"
PREDICTION_PACK_SCHEMA = f"{VERSION}.sealed_prediction_pack.v1"
FOUR_ARM_BARRIER_SCHEMA = f"{VERSION}.four_arm_barrier.v1"
SCORE_RECEIPT_SCHEMA = f"{VERSION}.aggregate_score_receipt.v1"
CLOSURE_SCHEMA = f"{VERSION}.runtime_closure.v1"
TEST_ATTESTATION_SCHEMA = f"{VERSION}.source_free_test_attestation.v1"
QWEN_MULTI_SAFE_RECEIPT_SCHEMA = (
    "gscl_narrative_closed_choice_multi_pack_worker_v1."
    "private_runtime_receipt.v1"
)
CLOSED_CHOICE_ACTUAL_CANARY_LINEAGE_SCHEMA = (
    "gscl_closed_choice_actual_canary_lineage_terminal_v2"
)
CLOSED_CHOICE_ACTUAL_CANARY_LINEAGE_FILE_SHA256 = (
    "4a15b4b209896a7999f62371df1c69e65b9d7b95b5821de58a9b8497ccac5f6a"
)
CLOSED_CHOICE_ACTUAL_CANARY_LINEAGE_SELF_SHA256 = (
    "712a3311982248668bd6c797fa5cb436e0c0b66d755b2c4db2e9f3761769ce7a"
)
CLOSED_CHOICE_WORKER_SHA256 = (
    "e0f5606da1c0cbc9ba5d74d758cb24b22102926fe8fc078e8b218445ec19271c"
)
CLOSED_CHOICE_PROMPT_SHA256 = (
    "8f7c31f559b060ad56ccbd9319c3df47b616d4a52e0e764b905a8422ce7cc092"
)
CLOSED_CHOICE_PARSER_SHA256 = (
    "100143ff1bb126e48a72450c774b65a52e616b84bb8e6420a1e0f481efcbe3b9"
)
GSCL_MINILM_TARGET_SCHEMA = "gscl_minilm_portable_target_manifest_v1"
GSCL_MINILM_RUNTIME_SCHEMA = (
    "gscl_minilm_portable_runtime_receipt_v1"
)
GSCL_MINILM_CANARY_SCHEMA = (
    "gscl_minilm_portable_canary_receipt_v1"
)
INTERNAL_FACTORY_QUALIFICATION_SCHEMA = (
    f"{VERSION}.internal_factory_qualification.safe.v1"
)
INTERNAL_FACTORY_EXECUTION_SCHEMA = (
    f"{VERSION}.internal_factory_execution.safe.v1"
)
OUTER_SYSTEMD_ATTESTATION_SCHEMA = (
    f"{VERSION}.outer_systemd_attestation.v1"
)
OUTER_SYSTEMD_CONTRACT = {
    "CPUQuotaPerSecUSec": "4s",
    "CPUWeight": "25",
    # systemd 255 canonicalises the unit spelling ``IPAddressDeny=any`` to
    # this exact value in ``systemctl show``.
    "IPAddressDeny": "::/0 0.0.0.0/0",
    "IOSchedulingClass": "3",
    "IOWeight": "25",
    "KillMode": "control-group",
    "MemoryHigh": "25769803776",
    "MemoryMax": "34359738368",
    "MemorySwapMax": "0",
    "Nice": "10",
    "NoNewPrivileges": "yes",
    "PrivateDevices": "no",
    "PrivateTmp": "no",
    "ProtectSystem": "no",
    "ReadOnlyPaths": "",
    "ReadWritePaths": "",
    "Restart": "no",
    "RestrictAddressFamilies": "AF_UNIX",
    "RuntimeMaxUSec": "infinity",
    "TasksMax": "96",
    "TimeoutStartUSec": "infinity",
    "Type": "oneshot",
    "UMask": "0077",
}
_OUTER_PRIVATE_TMP_TRADEOFF = (
    "jtl311linux user services deny open('/') when any ProtectSystem or "
    "ReadOnlyPaths/ReadWritePaths mount-namespace property is enabled. "
    "The outer service therefore attests ProtectSystem=no and empty path "
    "lists; PrivateTmp=no preserves the host /var/tmp evidence root. "
    "Filesystem isolation is provided by the frozen inner Landlock "
    "allowlist plus same-FD/openat custody."
)
_OUTER_SYSTEMD_LIVE_PROPERTIES = (
    "ActiveState",
    "ControlGroup",
    "FragmentPath",
    "Id",
    "InvocationID",
    "MainPID",
    "NRestarts",
    "SubState",
    *tuple(OUTER_SYSTEMD_CONTRACT),
)
_SYSTEMCTL = Path("/usr/bin/systemctl")
_QWEN_CRITICAL_DISTRIBUTIONS = frozenset(
    {
        "huggingface-hub",
        "numpy",
        "safetensors",
        "tokenizers",
        "torch",
        "transformers",
    }
)
_PYTEST_RUNTIME_DISTRIBUTIONS = (
    ("pytest", "pytest"),
    ("pluggy", "pluggy"),
    ("iniconfig", "iniconfig"),
    ("packaging", "packaging"),
    ("tomli", "tomli"),
    ("exceptiongroup", "exceptiongroup"),
    ("typing_extensions", "typing_extensions"),
    ("numpy", "numpy"),
)
_PYTEST_KNOWN_ABSENT_RECORD_ENTRIES = (
    "../../bin/py.test",
    "../../bin/pytest",
)
PYTEST_WHEEL_BUNDLE_SCHEMA = "gscl_pytest_wheel_bundle_v1"
PYTEST_WHEEL_BUNDLE_FILE_SHA256 = (
    "fcf918fba6d579ce738e395282c5428712712cc510afb62375a3a7a1306ee2c8"
)
PYTEST_WHEEL_BUNDLE_SELF_SHA256 = (
    "4186aa76be9091a4b19ce01ff7882ccb02976ef9692412739b88e8d94a38cb6e"
)
_PYTEST_WHEEL_DISTRIBUTION_VERSIONS = {
    "exceptiongroup": "1.3.1",
    "iniconfig": "2.3.0",
    "packaging": "26.2",
    "pluggy": "1.6.0",
    "pytest": "8.3.3",
    "tomli": "2.4.1",
    "typing_extensions": "4.16.0",
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9_.-]{1,160}\Z")
_QWEN_BATCH_ID = re.compile(r"[a-z][a-z0-9.-]{1,63}\Z")
_QWEN_MAXIMUM_BATCH_COUNT = 4_096
_QWEN_MAXIMUM_STORY_COUNT = 64
_QWEN_MAXIMUM_COMPLETION_TOKENS = 512
_TOKEN = object()
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_CLOSURE_FILES = 20_000

LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
LANDLOCK_MINIMUM_ABI = 3
LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
LANDLOCK_ACCESS_FS_REFER = 1 << 13
LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14
LANDLOCK_HANDLED_ACCESS_FS = (1 << 15) - 1
LANDLOCK_READ_EXECUTE_ACCESS = (
    LANDLOCK_ACCESS_FS_EXECUTE
    | LANDLOCK_ACCESS_FS_READ_FILE
    | LANDLOCK_ACCESS_FS_READ_DIR
)
LANDLOCK_WORK_ACCESS = (
    LANDLOCK_ACCESS_FS_WRITE_FILE
    | LANDLOCK_ACCESS_FS_READ_FILE
    | LANDLOCK_ACCESS_FS_READ_DIR
    | LANDLOCK_ACCESS_FS_REMOVE_DIR
    | LANDLOCK_ACCESS_FS_REMOVE_FILE
    | LANDLOCK_ACCESS_FS_MAKE_DIR
    | LANDLOCK_ACCESS_FS_MAKE_REG
    | LANDLOCK_ACCESS_FS_TRUNCATE
)
PR_SET_NO_NEW_PRIVS = 38
SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446

OFFLINE_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_MODULE_LOADING": "LAZY",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_OFFLINE": "1",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_OFFLINE": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
SUPERVISOR_LANDLOCK_DIRECT_PARENT_ENVIRONMENT_KEY = (
    "GSCL_SUPERVISOR_LANDLOCK_DIRECT_PARENT_AUTHORITY_V1"
)
SUPERVISOR_LANDLOCK_DIRECT_PARENT_AUTHORITY = (
    "97ff3a77c33a3113712a4c11a9fd347902a12b45f76935023d2ac66377936c35"
)
QWEN_CUDA_VISIBLE_DEVICES = ("0", "1")
_FROZEN_PHYSICAL_GPU_INDICES = (0, 1)
_GPU_CONTROL_DEVICES = (
    Path("/dev/nvidiactl"),
    Path("/dev/nvidia-uvm"),
)
_GPU_OPTIONAL_CONTROL_DEVICES = (
    Path("/dev/nvidia-uvm-tools"),
    Path("/dev/nvidia-modeset"),
)


class FormalSupervisorError(RuntimeError):
    """A supervisor invariant failed closed."""

    def __init__(self, issue_id: str) -> None:
        self.issue_id = issue_id
        super().__init__(issue_id)


def _canonical_bytes(value: Any) -> bytes:
    def validate(item: Any) -> None:
        if item is None or type(item) in {bool, int, str}:
            return
        if isinstance(item, list):
            for child in item:
                validate(child)
            return
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise FormalSupervisorError("canonical_key_invalid")
            for child in item.values():
                validate(child)
            return
        raise FormalSupervisorError("canonical_type_invalid")

    validate(value)
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise FormalSupervisorError("canonical_json_invalid") from exc


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _canonical_embedded_receipt_bytes(value: object) -> bytes:
    """Canonicalize a frozen embedded runtime receipt without widening outputs.

    Supervisor-owned receipts deliberately forbid floats.  The frozen MiniLM
    canary receipt is instead carried as a hashed JSON string and contains two
    finite numerical error diagnostics.  Re-encode only that embedded object
    with its producer's exact JSON contract while keeping the outer supervisor
    receipt on the stricter integer/string-only contract.
    """

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FormalSupervisorError(
            "factory_encoder_binding_invalid"
        ) from exc


def _require_hash(value: object, issue_id: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FormalSupervisorError(issue_id)
    return value


def _parse_json(raw: bytes, *, issue_id: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise FormalSupervisorError(issue_id) from exc
    if not isinstance(value, dict) or _canonical_bytes(value) != raw:
        raise FormalSupervisorError(issue_id)
    return value


def _json_content_hash(raw: bytes, *, issue_id: str) -> str:
    return _content_hash(dict(_parse_json(raw, issue_id=issue_id)))


def _validate_minilm_target_manifest_bytes(
    raw: bytes,
) -> Mapping[str, Any]:
    manifest = _parse_json(
        raw, issue_id="minilm_target_manifest_invalid"
    )
    body = dict(manifest)
    claimed = body.pop("self_sha256", None)
    if (
        manifest.get("schema") != GSCL_MINILM_TARGET_SCHEMA
        or _require_hash(
            claimed, "minilm_target_manifest_self_hash_invalid"
        )
        != _content_hash(body)
    ):
        raise FormalSupervisorError(
            "minilm_target_manifest_invalid"
        )
    return manifest


def _validate_factory_encoder_binding(
    value: object,
    *,
    expected_target_file_sha256: str | None = None,
    expected_target_self_sha256: str | None = None,
) -> Mapping[str, str]:
    expected_fields = {
        "encoder_canary_receipt_json",
        "encoder_canary_receipt_sha256",
        "encoder_exact_type",
        "encoder_runtime_receipt_json",
        "encoder_runtime_receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise FormalSupervisorError(
            "factory_encoder_binding_invalid"
        )
    if (
        value.get("encoder_exact_type")
        != (
            "replication_runtime.gscl_minilm_portable_v1.binding."
            "GSCLPortableOfflineMiniLMEncoder"
        )
    ):
        raise FormalSupervisorError(
            "factory_encoder_binding_invalid"
        )
    decoded_receipts: dict[str, Mapping[str, Any]] = {}
    for prefix in ("encoder_runtime", "encoder_canary"):
        text = value.get(f"{prefix}_receipt_json")
        claimed = value.get(f"{prefix}_receipt_sha256")
        if (
            not isinstance(text, str)
            or not text
            or not isinstance(claimed, str)
            or _SHA256.fullmatch(claimed) is None
        ):
            raise FormalSupervisorError(
                "factory_encoder_binding_invalid"
            )
        try:
            decoded = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda _: (
                    _ for _ in ()
                ).throw(ValueError()),
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise FormalSupervisorError(
                "factory_encoder_binding_invalid"
            ) from exc
        try:
            encoded = text.encode("ascii")
        except UnicodeError as exc:
            raise FormalSupervisorError(
                "factory_encoder_binding_invalid"
            ) from exc
        if (
            not isinstance(decoded, dict)
            or _canonical_embedded_receipt_bytes(decoded) != encoded
            or hashlib.sha256(encoded).hexdigest() != claimed
        ):
            raise FormalSupervisorError(
                "factory_encoder_binding_invalid"
            )
        decoded_receipts[prefix] = decoded
    runtime = decoded_receipts["encoder_runtime"]
    canary = decoded_receipts["encoder_canary"]
    if (
        runtime.get("schema") != GSCL_MINILM_RUNTIME_SCHEMA
        or runtime.get("status")
        != "verified_exact_gscl_target_local_minilm_runtime"
        or runtime.get("formal_source_or_rows_accessed") is not False
        or runtime.get("labels_accessed") is not False
        or runtime.get("network_calls") != 0
        or canary.get("schema") != GSCL_MINILM_CANARY_SCHEMA
        or canary.get("status")
        != "passed_target_local_repeat_exact_canary"
        or canary.get("repeat_count") != 2
        or canary.get("repeat_byte_exact") is not True
        or canary.get("repeat_elementwise_exact") is not True
        or canary.get("cross_hardware_byte_identity_claimed")
        is not False
        or canary.get("formal_source_or_rows_accessed") is not False
        or canary.get("labels_accessed") is not False
        or canary.get("network_calls") != 0
        or (
            expected_target_file_sha256 is not None
            and runtime.get("target_manifest_file_sha256")
            != expected_target_file_sha256
        )
        or (
            expected_target_self_sha256 is not None
            and (
                runtime.get("target_manifest_self_sha256")
                != expected_target_self_sha256
                or canary.get("target_manifest_self_sha256")
                != expected_target_self_sha256
            )
        )
    ):
        raise FormalSupervisorError(
            "factory_encoder_binding_invalid"
        )
    return value


def _factory_encoder_bindings_content_equivalent(
    qualified: object,
    observed: object,
    *,
    expected_observed_target_manifest_path: Path,
    expected_target_file_sha256: str,
    expected_target_self_sha256: str,
) -> bool:
    """Compare runtime identity while treating custody location separately.

    A qualification target manifest and its byte-identical formal copy must
    live below different roots.  Their absolute paths are therefore location
    evidence, not runtime-content identity.  Every other runtime and canary
    field remains byte-exact.  The observed location is fixed by the caller;
    the qualification location remains bound by the already-frozen
    qualification receipt.
    """

    qualified_binding = _validate_factory_encoder_binding(
        qualified,
        expected_target_file_sha256=expected_target_file_sha256,
        expected_target_self_sha256=expected_target_self_sha256,
    )
    observed_binding = _validate_factory_encoder_binding(
        observed,
        expected_target_file_sha256=expected_target_file_sha256,
        expected_target_self_sha256=expected_target_self_sha256,
    )
    try:
        qualified_runtime = json.loads(
            qualified_binding["encoder_runtime_receipt_json"],
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _: (
                _ for _ in ()
            ).throw(ValueError()),
        )
        observed_runtime = json.loads(
            observed_binding["encoder_runtime_receipt_json"],
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _: (
                _ for _ in ()
            ).throw(ValueError()),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FormalSupervisorError(
            "factory_encoder_binding_invalid"
        ) from exc
    if not isinstance(qualified_runtime, dict) or not isinstance(
        observed_runtime, dict
    ):
        raise FormalSupervisorError(
            "factory_encoder_binding_invalid"
        )
    qualified_path = qualified_runtime.pop(
        "target_manifest_path", None
    )
    observed_path = observed_runtime.pop(
        "target_manifest_path", None
    )
    if (
        not isinstance(qualified_path, str)
        or not Path(qualified_path).is_absolute()
        or observed_path
        != str(expected_observed_target_manifest_path)
    ):
        return False
    return (
        qualified_binding["encoder_exact_type"]
        == observed_binding["encoder_exact_type"]
        and qualified_binding["encoder_canary_receipt_json"]
        == observed_binding["encoder_canary_receipt_json"]
        and qualified_binding["encoder_canary_receipt_sha256"]
        == observed_binding["encoder_canary_receipt_sha256"]
        and _canonical_embedded_receipt_bytes(qualified_runtime)
        == _canonical_embedded_receipt_bytes(observed_runtime)
    )


def _validated_qwen_batch_rows(
    receipt: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    rows = receipt.get("batches")
    batch_count = receipt.get("batch_count")
    expected_fields = {
        "batch_id",
        "decision_elapsed_ns",
        "decision_invalid_count",
        "decision_valid_count",
        "input_file_sha256",
        "input_pack_commitment",
        "output_file_sha256",
        "selection_receipt_commitment",
        "selection_receipt_count",
        "sequence",
        "story_count",
        "valid_wire_completion_token_count_maximum",
        "valid_wire_completion_token_count_sum",
    }
    if (
        isinstance(batch_count, bool)
        or not isinstance(batch_count, int)
        or not 1 <= batch_count <= _QWEN_MAXIMUM_BATCH_COUNT
        or not isinstance(rows, list)
        or len(rows) != batch_count
    ):
        raise FormalSupervisorError(
            "qwen_runtime_batch_receipts_invalid"
        )
    validated: dict[int, Mapping[str, Any]] = {}
    previous_sequence = -1
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise FormalSupervisorError(
                "qwen_runtime_batch_receipts_invalid"
            )
        integer_fields = (
            "decision_elapsed_ns",
            "decision_invalid_count",
            "decision_valid_count",
            "selection_receipt_count",
            "sequence",
            "story_count",
            "valid_wire_completion_token_count_maximum",
            "valid_wire_completion_token_count_sum",
        )
        if any(
            isinstance(row.get(field), bool)
            or not isinstance(row.get(field), int)
            for field in integer_fields
        ):
            raise FormalSupervisorError(
                "qwen_runtime_batch_receipts_invalid"
            )
        sequence = row["sequence"]
        story_count = row["story_count"]
        valid_count = row["decision_valid_count"]
        invalid_count = row["decision_invalid_count"]
        token_maximum = row[
            "valid_wire_completion_token_count_maximum"
        ]
        token_sum = row[
            "valid_wire_completion_token_count_sum"
        ]
        if (
            not isinstance(row.get("batch_id"), str)
            or _QWEN_BATCH_ID.fullmatch(row["batch_id"]) is None
            or sequence <= previous_sequence
            or not 0 <= sequence <= 9_999_999_999
            or not 1 <= story_count <= _QWEN_MAXIMUM_STORY_COUNT
            or not 0 <= valid_count <= story_count
            or not 0 <= invalid_count <= story_count
            or valid_count + invalid_count != story_count
            or row["decision_elapsed_ns"] <= 0
            or row["selection_receipt_count"] != valid_count
            or not 0 <= token_maximum <= (
                _QWEN_MAXIMUM_COMPLETION_TOKENS
            )
            or not 0 <= token_sum <= (
                valid_count * _QWEN_MAXIMUM_COMPLETION_TOKENS
            )
            or (valid_count == 0 and (token_maximum != 0 or token_sum != 0))
            or (valid_count > 0 and token_sum < token_maximum)
        ):
            raise FormalSupervisorError(
                "qwen_runtime_batch_receipts_invalid"
            )
        for field in (
            "input_file_sha256",
            "input_pack_commitment",
            "output_file_sha256",
            "selection_receipt_commitment",
        ):
            _require_hash(
                row.get(field),
                "qwen_runtime_batch_receipts_invalid",
            )
        validated[sequence] = row
        previous_sequence = sequence
    if len(validated) != batch_count:
        raise FormalSupervisorError(
            "qwen_runtime_batch_receipts_invalid"
        )
    return validated


def _balanced_triplet_batch_plan(
    expected_stories: Sequence[tuple[str, str, str]],
    *,
    shard_count: int,
    maximum_story_count: int = 63,
) -> tuple[Mapping[str, Any], ...]:
    """Round-robin complete item triplets, then pack each shard evenly."""

    if (
        not isinstance(expected_stories, Sequence)
        or isinstance(expected_stories, (str, bytes))
        or not expected_stories
        or len(expected_stories) % 3 != 0
        or isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count < 1
        or isinstance(maximum_story_count, bool)
        or not isinstance(maximum_story_count, int)
        or maximum_story_count < 3
        or maximum_story_count % 3 != 0
    ):
        raise FormalSupervisorError(
            "extractor_triplet_plan_invalid"
        )
    item_triplets: list[
        tuple[int, tuple[tuple[str, str, str], ...]]
    ] = []
    expected_roles = (
        "query",
        "first_choice",
        "second_choice",
    )
    for item_index, offset in enumerate(
        range(0, len(expected_stories), 3)
    ):
        triplet = tuple(expected_stories[offset : offset + 3])
        if (
            len(triplet) != 3
            or any(
                not isinstance(row, tuple)
                or len(row) != 3
                or not all(
                    isinstance(value, str) and value
                    for value in row
                )
                for row in triplet
            )
            or len({row[0] for row in triplet}) != 1
            or tuple(row[1] for row in triplet)
            != expected_roles
        ):
            raise FormalSupervisorError(
                "extractor_triplet_plan_invalid"
            )
        item_triplets.append((item_index, triplet))
    if len(item_triplets) < shard_count:
        raise FormalSupervisorError(
            "formal_story_set_too_small_for_frozen_parallelism"
        )
    by_shard: list[
        list[tuple[int, tuple[tuple[str, str, str], ...]]]
    ] = [[] for _ in range(shard_count)]
    for item_index, triplet in item_triplets:
        by_shard[item_index % shard_count].append(
            (item_index, triplet)
        )
    items_per_batch = maximum_story_count // 3
    maximum_rounds = max(
        (
            len(rows) + items_per_batch - 1
        )
        // items_per_batch
        for rows in by_shard
    )
    plan: list[Mapping[str, Any]] = []
    for round_index in range(maximum_rounds):
        start = round_index * items_per_batch
        stop = start + items_per_batch
        for shard_index, rows in enumerate(by_shard):
            selected = rows[start:stop]
            if not selected:
                continue
            item_indices = tuple(
                item_index for item_index, _ in selected
            )
            stories = tuple(
                story
                for _, triplet in selected
                for story in triplet
            )
            if (
                len(stories) != 3 * len(item_indices)
                or len(stories) > maximum_story_count
            ):
                raise FormalSupervisorError(
                    "extractor_triplet_plan_invalid"
                )
            plan.append(
                {
                    "item_indices": item_indices,
                    "shard_index": shard_index,
                    "stories": stories,
                }
            )
    observed = [
        item_index
        for row in plan
        for item_index in row["item_indices"]
    ]
    if sorted(observed) != list(range(len(item_triplets))):
        raise FormalSupervisorError(
            "extractor_triplet_plan_invalid"
        )
    return tuple(plan)


def _validate_closed_choice_actual_canary_lineage(
    raw: bytes,
    *,
    expected_model_manifest_sha256: str,
) -> Mapping[str, Any]:
    file_sha256 = hashlib.sha256(raw).hexdigest()
    if (
        file_sha256
        != CLOSED_CHOICE_ACTUAL_CANARY_LINEAGE_FILE_SHA256
    ):
        raise FormalSupervisorError(
            "closed_choice_actual_canary_lineage_hash_invalid"
        )
    try:
        receipt = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError()
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise FormalSupervisorError(
            "closed_choice_actual_canary_lineage_invalid"
        ) from exc
    if not isinstance(receipt, dict):
        raise FormalSupervisorError(
            "closed_choice_actual_canary_lineage_invalid"
        )
    body = dict(receipt)
    claimed = body.pop("self_sha256", None)
    previous = receipt.get("previous_lineage")
    current = receipt.get("current_ext4_actual")
    if (
        receipt.get("schema")
        != CLOSED_CHOICE_ACTUAL_CANARY_LINEAGE_SCHEMA
        or receipt.get("status")
        != "PASS_CURRENT_EXT4_SOURCE_FREE_ACTUAL_LINEAGE"
        or claimed
        != CLOSED_CHOICE_ACTUAL_CANARY_LINEAGE_SELF_SHA256
        or _content_hash(body) != claimed
        or receipt.get("claim_scope")
        != "source_free_runtime_qualification_only_no_efficacy"
        or receipt.get("formal_measurement") is not False
        or receipt.get("effect_study_count") != 0
        or receipt.get("effect_gate_count") != 0
        or receipt.get("official_source_access_count") != 0
        or receipt.get(
            "official_source_content_supplied_to_model"
        )
        is not False
        or receipt.get(
            "public_synthetic_content_supplied_to_model"
        )
        is not True
        or receipt.get("free_form_generation_count") != 0
        or receipt.get("lineage_model_weight_load_count") != 3
        or receipt.get(
            "successful_teacher_forced_qualification_run_count"
        )
        != 2
        or receipt.get("model_asset_manifest_file_sha256")
        != expected_model_manifest_sha256
        or previous
        != {
            "file_sha256": (
                "47e46091aabc8f35eca7f3ebe2a0730a"
                "39da27df1be124ee6e9d5eab9a86bae9"
            ),
            "lineage_model_weight_load_count": 2,
            "self_sha256": (
                "c66f624607fcc231f4fddbb98966c275a"
                "d65784bef5d315445156a387de70371"
            ),
            "successful_teacher_forced_qualification_run_count": 1,
            "worker_sha256": (
                "622e0f8b9c97d014225113a7da3e0d8d"
                "d9ba637a7f7b597a3f8ee9a7f015bfb1"
            ),
        }
        or not isinstance(current, dict)
        or current.get("status")
        != "PASS_CLOSED_CHOICE_ACTUAL_SOURCE_FREE_CANARY"
        or current.get("model_weight_load_count") != 1
        or current.get(
            "successful_teacher_forced_qualification_run_count"
        )
        != 1
        or current.get("worker_sha256")
        != CLOSED_CHOICE_WORKER_SHA256
        or current.get("contract_sha256")
        != _STABLE_EXTRACTOR_SHA256S["extractor_contract"]
        or current.get("runtime_python_sha256")
        != (
            "7d51cd6b48b521277f5caa4610a82126"
            "e315fa2be4df069823a8b1eeb5bd4a86"
        )
        or current.get("launch_contract_file_sha256")
        != (
            "e1ebc44bf3401083928882893398bf767"
            "4e78d2d772095b8311269f5c06e288e"
        )
        or re.fullmatch(
            r"[0-9a-f]{32}",
            current.get("invocation_id", ""),
        )
        is None
    ):
        raise FormalSupervisorError(
            "closed_choice_actual_canary_lineage_invalid"
        )
    for value in (
        current.get("file_sha256"),
        current.get("self_sha256"),
        current.get("runtime_receipt_sha256"),
        current.get("target_double_run_receipt_sha256"),
    ):
        _require_hash(
            value,
            "closed_choice_actual_canary_lineage_invalid",
        )
    return {
        "file_sha256": file_sha256,
        "lineage_model_weight_load_count": 3,
        "repaired_actual_file_sha256": current[
            "file_sha256"
        ],
        "repaired_actual_self_sha256": current[
            "self_sha256"
        ],
        "self_sha256": claimed,
        "successful_teacher_forced_qualification_run_count": 2,
        "worker_sha256": current["worker_sha256"],
    }


def _validate_qwen_runtime_safe_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_model_manifest_sha256: str,
    expected_visible_device: str,
    expected_lineage: str,
) -> Mapping[str, Any]:
    expected_keys = {
        "batch_count",
        "batches",
        "claim_scope",
        "execution_closure",
        "free_form_generation_count",
        "input_manifest_file_sha256",
        "lineage",
        "loaded_distribution_closure_sha256",
        "loaded_distributions",
        "logical_gpu_binding",
        "model_asset_manifest_file_sha256",
        "runtime_receipt",
        "runtime_receipt_sha256",
        "schema",
        "score_operation",
        "selection_receipt_commitments_sha256",
        "selection_receipt_count",
        "self_sha256",
        "single_model_load_count",
        "source_content_supplied",
        "target_double_run_receipt",
        "target_double_run_receipt_sha256",
        "teacher_forced_backend_commitment",
        "worker_version",
    }
    if set(receipt) != expected_keys:
        raise FormalSupervisorError(
            "qwen_runtime_receipt_fields_invalid"
        )
    body = {
        key: value
        for key, value in receipt.items()
        if key != "self_sha256"
    }
    if (
        receipt.get("schema") != QWEN_MULTI_SAFE_RECEIPT_SCHEMA
        or receipt.get("worker_version")
        != "gscl_narrative_closed_choice_multi_pack_worker_v1"
        or receipt.get("single_model_load_count") != 1
        or receipt.get("lineage") != expected_lineage
        or receipt.get("source_content_supplied")
        is not (expected_lineage == "formal_measurement")
        or receipt.get("claim_scope")
        != "untrusted_grounded_closed_choice_proposal_only"
        or receipt.get("free_form_generation_count") != 0
        or receipt.get("score_operation")
        != "teacher_forced_forward_log_softmax"
        or receipt.get("model_asset_manifest_file_sha256")
        != expected_model_manifest_sha256
        or _require_hash(
            receipt.get("self_sha256"),
            "qwen_runtime_receipt_self_hash_invalid",
        )
        != _content_hash(body)
    ):
        raise FormalSupervisorError(
            "qwen_runtime_receipt_identity_invalid"
        )
    batch_rows = _validated_qwen_batch_rows(receipt)
    selection_count = receipt.get("selection_receipt_count")
    if (
        isinstance(selection_count, bool)
        or not isinstance(selection_count, int)
        or selection_count
        != sum(
            row["selection_receipt_count"]
            for row in batch_rows.values()
        )
    ):
        raise FormalSupervisorError(
            "qwen_runtime_selection_receipts_invalid"
        )
    _require_hash(
        receipt.get("selection_receipt_commitments_sha256"),
        "qwen_runtime_selection_receipts_invalid",
    )
    if receipt["selection_receipt_commitments_sha256"] != (
        _content_hash(
            [
                {
                    "selection_receipt_commitment": row[
                        "selection_receipt_commitment"
                    ],
                    "selection_receipt_count": row[
                        "selection_receipt_count"
                    ],
                    "sequence": row["sequence"],
                }
                for row in batch_rows.values()
            ]
        )
    ):
        raise FormalSupervisorError(
            "qwen_runtime_selection_receipts_invalid"
        )
    _require_hash(
        receipt.get("teacher_forced_backend_commitment"),
        "qwen_runtime_teacher_forced_binding_invalid",
    )
    _require_hash(
        receipt.get("input_manifest_file_sha256"),
        "qwen_runtime_manifest_hash_invalid",
    )
    runtime_receipt = receipt.get("runtime_receipt")
    double_receipt = receipt.get("target_double_run_receipt")
    execution_closure = receipt.get("execution_closure")
    distributions = receipt.get("loaded_distributions")
    logical_gpu = receipt.get("logical_gpu_binding")
    if (
        not isinstance(runtime_receipt, dict)
        or not isinstance(double_receipt, dict)
        or not isinstance(execution_closure, dict)
        or not isinstance(distributions, list)
        or not distributions
        or not isinstance(logical_gpu, dict)
        or set(logical_gpu)
        != {
            "cuda_visible_devices",
            "logical_compute_capability",
            "logical_device_count",
            "logical_device_index",
            "logical_device_name",
            "logical_device_uuid",
            "model_parameter_logical_device_indices",
        }
        or logical_gpu.get("cuda_visible_devices")
        != expected_visible_device
        or logical_gpu.get("logical_device_count") != 1
        or logical_gpu.get("logical_device_index") != 0
        or logical_gpu.get(
            "model_parameter_logical_device_indices"
        )
        != [0]
        or not isinstance(
            logical_gpu.get("logical_device_name"), str
        )
        or not logical_gpu["logical_device_name"]
        or not isinstance(
            logical_gpu.get("logical_device_uuid"), str
        )
        or not logical_gpu["logical_device_uuid"]
        or not isinstance(
            logical_gpu.get("logical_compute_capability"), list
        )
        or len(logical_gpu["logical_compute_capability"]) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in logical_gpu["logical_compute_capability"]
        )
        or hashlib.sha256(
            _canonical_bytes(runtime_receipt)
        ).hexdigest()
        != _require_hash(
            receipt.get("runtime_receipt_sha256"),
            "qwen_runtime_receipt_hash_invalid",
        )
        or hashlib.sha256(
            _canonical_bytes(double_receipt)
        ).hexdigest()
        != _require_hash(
            receipt.get("target_double_run_receipt_sha256"),
            "qwen_double_run_receipt_hash_invalid",
        )
        or double_receipt.get("runtime_receipt_sha256")
        != receipt["runtime_receipt_sha256"]
        or runtime_receipt.get("schema")
        != (
            "gscl_narrative_closed_choice_qwen_runtime_v1."
            "runtime_receipt.v1"
        )
        or double_receipt.get("schema")
        != (
            "gscl_narrative_closed_choice_qwen_runtime_v1."
            "double_run_receipt.v1"
        )
        or double_receipt.get("repeat_exact") is not True
        or double_receipt.get("repeat_count") != 2
        or double_receipt.get("free_form_generation_count") != 0
        or runtime_receipt.get("free_form_generation_count")
        != 0
        or runtime_receipt.get("score_operation")
        != "teacher_forced_forward_log_softmax"
        or execution_closure.get(
            "target_double_run_receipt_sha256"
        )
        != receipt["target_double_run_receipt_sha256"]
        or execution_closure.get("model_asset_manifest_sha256")
        != expected_model_manifest_sha256
        or execution_closure.get("prompt_sha256")
        != CLOSED_CHOICE_PROMPT_SHA256
        or execution_closure.get("parser_closure_sha256")
        != CLOSED_CHOICE_PARSER_SHA256
        or execution_closure.get("model_runtime_closure_sha256")
        != _content_hash(
            {
                "double_run_receipt_sha256": receipt[
                    "target_double_run_receipt_sha256"
                ],
                "model_asset_manifest_sha256": (
                    expected_model_manifest_sha256
                ),
                "runtime_receipt_sha256": receipt[
                    "runtime_receipt_sha256"
                ],
                "teacher_forced_backend_commitment": receipt[
                    "teacher_forced_backend_commitment"
                ],
            }
        )
        or set(execution_closure)
        != {
            "model_asset_manifest_sha256",
            "model_runtime_closure_sha256",
            "parser_closure_sha256",
            "prompt_sha256",
            "target_double_run_receipt_sha256",
        }
    ):
        raise FormalSupervisorError(
            "qwen_runtime_receipt_preimage_invalid"
        )
    distribution_names: set[str] = set()
    for row in distributions:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "closure_sha256",
                "critical",
                "distribution",
                "loaded_top_level_modules",
                "version",
            }
            or not isinstance(row["distribution"], str)
            or not row["distribution"]
            or row["distribution"] in distribution_names
            or not isinstance(row["version"], str)
            or not row["version"]
            or not isinstance(row["critical"], bool)
            or not isinstance(row["loaded_top_level_modules"], list)
            or any(
                not isinstance(value, str)
                for value in row["loaded_top_level_modules"]
            )
        ):
            raise FormalSupervisorError(
                "qwen_distribution_closure_invalid"
            )
        _require_hash(
            row["closure_sha256"],
            "qwen_distribution_hash_invalid",
        )
        distribution_names.add(row["distribution"])
    if (
        not _QWEN_CRITICAL_DISTRIBUTIONS.issubset(
            distribution_names
        )
        or any(
            row["critical"]
            != (
                row["distribution"]
                in _QWEN_CRITICAL_DISTRIBUTIONS
            )
            for row in distributions
        )
        or _content_hash(distributions)
        != _require_hash(
            receipt.get("loaded_distribution_closure_sha256"),
            "qwen_distribution_closure_hash_invalid",
        )
    ):
        raise FormalSupervisorError(
            "qwen_distribution_closure_invalid"
        )
    return {
        "execution_closure": dict(execution_closure),
        "logical_gpu_binding": dict(logical_gpu),
        "loaded_distribution_closure_sha256": receipt[
            "loaded_distribution_closure_sha256"
        ],
        "runtime_receipt_sha256": receipt[
            "runtime_receipt_sha256"
        ],
        "target_double_run_receipt_sha256": receipt[
            "target_double_run_receipt_sha256"
        ],
        "teacher_forced_backend_commitment": receipt[
            "teacher_forced_backend_commitment"
        ],
    }


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _relative_components(relative: str) -> tuple[str, ...]:
    if not isinstance(relative, str) or not relative:
        raise FormalSupervisorError("relative_path_invalid")
    path = PurePosixPath(relative)
    components = path.parts
    if (
        path.is_absolute()
        or not components
        or any(
            component in {"", ".", ".."}
            or _SAFE_COMPONENT.fullmatch(component) is None
            for component in components
        )
    ):
        raise FormalSupervisorError("relative_path_invalid")
    return components


def _open_absolute_directory(path: Path) -> int:
    """Open an absolute directory without following any component symlink."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise FormalSupervisorError("trusted_root_not_absolute")
    descriptor = os.open(
        "/",
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            opened = os.fstat(next_descriptor)
            if not stat.S_ISDIR(opened.st_mode):
                os.close(next_descriptor)
                raise FormalSupervisorError(
                    "trusted_root_component_not_directory"
                )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        raise FormalSupervisorError("trusted_root_open_failed") from exc


def _create_fixed_private_root_once(path: Path) -> None:
    if not path.is_absolute() or path == Path("/"):
        raise FormalSupervisorError("trusted_root_not_absolute")
    parent = path.parent
    parent_fd = _open_absolute_directory(parent)
    try:
        try:
            os.mkdir(path.name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        child_fd = os.open(
            path.name,
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_CLOEXEC
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            opened = os.fstat(child_fd)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or opened.st_uid != os.getuid()
                or stat.S_IMODE(opened.st_mode) != 0o700
            ):
                raise FormalSupervisorError(
                    "trusted_root_topology_invalid"
                )
        finally:
            os.close(child_fd)
    except OSError as exc:
        raise FormalSupervisorError("trusted_root_create_failed") from exc
    finally:
        os.close(parent_fd)


class SecureDirectory:
    """Dirfd-relative storage with component-wise no-symlink traversal."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._fd = _open_absolute_directory(root)
        opened = os.fstat(self._fd)
        self.identity = (
            int(opened.st_dev),
            int(opened.st_ino),
            int(opened.st_uid),
        )

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> "SecureDirectory":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _open_parent(
        self, relative: str, *, create: bool
    ) -> tuple[int, str]:
        components = _relative_components(relative)
        descriptor = os.dup(self._fd)
        try:
            for component in components[:-1]:
                try:
                    next_descriptor = os.open(
                        component,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_CLOEXEC
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=descriptor,
                    )
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                    next_descriptor = os.open(
                        component,
                        os.O_RDONLY
                        | os.O_DIRECTORY
                        | os.O_CLOEXEC
                        | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=descriptor,
                    )
                metadata = os.fstat(next_descriptor)
                if (
                    not stat.S_ISDIR(metadata.st_mode)
                    or metadata.st_uid != os.getuid()
                ):
                    os.close(next_descriptor)
                    raise FormalSupervisorError(
                        "secure_parent_topology_invalid"
                    )
                os.close(descriptor)
                descriptor = next_descriptor
            return descriptor, components[-1]
        except OSError as exc:
            os.close(descriptor)
            raise FormalSupervisorError("secure_parent_open_failed") from exc
        except Exception:
            os.close(descriptor)
            raise

    def ensure_directory(self, relative: str) -> Path:
        components = _relative_components(relative)
        descriptor = os.dup(self._fd)
        try:
            for component in components:
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_CLOEXEC
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
                metadata = os.fstat(next_descriptor)
                if (
                    not stat.S_ISDIR(metadata.st_mode)
                    or metadata.st_uid != os.getuid()
                    or stat.S_IMODE(metadata.st_mode) != 0o700
                ):
                    os.close(next_descriptor)
                    raise FormalSupervisorError(
                        "secure_directory_topology_invalid"
                    )
                os.close(descriptor)
                descriptor = next_descriptor
        except OSError as exc:
            raise FormalSupervisorError(
                "secure_directory_create_failed"
            ) from exc
        finally:
            os.close(descriptor)
        return self.root.joinpath(*components)

    def read_bytes(
        self, relative: str, *, maximum_bytes: int = _MAX_JSON_BYTES
    ) -> bytes:
        parent_fd, name = self._open_parent(relative, create=False)
        descriptor = -1
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_uid != os.getuid()
                or before.st_size < 0
                or before.st_size > maximum_bytes
            ):
                raise FormalSupervisorError("secure_file_topology_invalid")
            chunks: list[bytes] = []
            observed = 0
            while True:
                chunk = os.read(descriptor, min(1 << 20, maximum_bytes + 1))
                if not chunk:
                    break
                chunks.append(chunk)
                observed += len(chunk)
                if observed > maximum_bytes:
                    raise FormalSupervisorError("secure_file_too_large")
            after = os.fstat(descriptor)
            if (
                after.st_dev != before.st_dev
                or after.st_ino != before.st_ino
                or after.st_size != before.st_size
                or after.st_mtime_ns != before.st_mtime_ns
                or after.st_ctime_ns != before.st_ctime_ns
                or after.st_nlink != 1
            ):
                raise FormalSupervisorError("secure_file_changed_during_read")
            return b"".join(chunks)
        except OSError as exc:
            raise FormalSupervisorError("secure_file_read_failed") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent_fd)

    def write_exclusive(self, relative: str, raw: bytes) -> str:
        if not isinstance(raw, bytes):
            raise FormalSupervisorError("secure_write_not_bytes")
        parent_fd, name = self._open_parent(relative, create=True)
        descriptor = -1
        try:
            descriptor = os.open(
                name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_fd,
            )
            os.fchmod(descriptor, 0o600)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
            ):
                raise FormalSupervisorError("secure_output_topology_invalid")
            view = memoryview(raw)
            offset = 0
            while offset < len(view):
                offset += os.write(descriptor, view[offset:])
            os.fsync(descriptor)
            os.fsync(parent_fd)
        except FileExistsError as exc:
            raise FormalSupervisorError("secure_output_already_exists") from exc
        except OSError as exc:
            raise FormalSupervisorError("secure_output_write_failed") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent_fd)
        return hashlib.sha256(raw).hexdigest()

    def write_json_exclusive(
        self, relative: str, value: Mapping[str, Any]
    ) -> str:
        return self.write_exclusive(relative, _canonical_bytes(dict(value)))

    def read_json(self, relative: str) -> Mapping[str, Any]:
        return _parse_json(
            self.read_bytes(relative), issue_id="secure_json_invalid"
        )

    def exists(self, relative: str) -> bool:
        try:
            self.read_bytes(relative, maximum_bytes=1)
        except FormalSupervisorError as exc:
            if exc.issue_id == "secure_file_read_failed":
                return False
            if exc.issue_id == "secure_file_too_large":
                return True
            raise
        return True


@dataclass(frozen=True)
class TestAttestation:
    receipt: Mapping[str, Any]
    _token: object


@dataclass(frozen=True)
class RuntimeClosure:
    manifest: Mapping[str, Any]
    file_hashes: Mapping[str, str]
    _token: object


@dataclass(frozen=True)
class FrozenAction:
    receipt: Mapping[str, Any]
    root_identity: tuple[int, int, int]
    closure: RuntimeClosure
    _token: object


@dataclass(frozen=True)
class FormalInvocation:
    receipt: Mapping[str, Any]
    action: FrozenAction
    root_identity: tuple[int, int, int]
    _token: object


@dataclass(frozen=True)
class ArmCommand:
    arm_id: str
    command_template: tuple[str, ...]
    code_roots: tuple[Path, ...]
    model_roots: tuple[Path, ...]
    implementation_path: Path
    implementation_sha256: str


@dataclass(frozen=True)
class ScorerCommand:
    command_template: tuple[str, ...]
    implementation_path: Path
    implementation_sha256: str


def _safe_absolute_path(path: Path, *, allow_file: bool) -> Path:
    """Return a resolved absolute path after rejecting symlink components."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise FormalSupervisorError("closure_path_not_absolute")
    current = Path("/")
    for index, component in enumerate(path.parts[1:]):
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise FormalSupervisorError("closure_path_unavailable") from exc
        is_last = index == len(path.parts[1:]) - 1
        if stat.S_ISLNK(metadata.st_mode):
            raise FormalSupervisorError("closure_path_symlink")
        if is_last and allow_file:
            if not stat.S_ISREG(metadata.st_mode):
                raise FormalSupervisorError("closure_file_not_regular")
        elif not stat.S_ISDIR(metadata.st_mode):
            raise FormalSupervisorError("closure_parent_not_directory")
    return path


def _validate_absent_absolute_path(path: Path) -> Path:
    """Bind a lexically normalized absent path without trusting symlinks.

    A missing intermediate directory is permitted because an installed
    distribution's RECORD entry may normalize below a directory that was
    never created.  Every existing prefix is nevertheless required to be a
    real directory, and the final path must remain absent.
    """

    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or Path(os.path.abspath(os.fspath(path))) != path
    ):
        raise FormalSupervisorError("closure_absent_path_invalid")
    current = Path("/")
    missing_prefix = False
    for component in path.parts[1:]:
        current = current / component
        if missing_prefix:
            continue
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            missing_prefix = True
            continue
        except OSError as exc:
            raise FormalSupervisorError(
                "closure_absent_path_unavailable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise FormalSupervisorError("closure_absent_path_symlink")
        if current == path:
            raise FormalSupervisorError("closure_absent_path_present")
        if not stat.S_ISDIR(metadata.st_mode):
            raise FormalSupervisorError(
                "closure_absent_parent_not_directory"
            )
    if not missing_prefix:
        raise FormalSupervisorError("closure_absent_path_present")
    return path


def _hash_regular_absolute(path: Path) -> str:
    _safe_absolute_path(path, allow_file=True)
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise FormalSupervisorError("closure_file_topology_invalid")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
        ):
            raise FormalSupervisorError("closure_file_changed_during_hash")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _read_regular_absolute_exact(
    path: Path, *, expected_sha256: str, maximum_bytes: int
) -> bytes:
    _safe_absolute_path(path, allow_file=True)
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 1 <= before.st_size <= maximum_bytes
        ):
            raise FormalSupervisorError("absolute_file_topology_invalid")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            or digest.hexdigest() != expected_sha256
        ):
            raise FormalSupervisorError("absolute_file_identity_drifted")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _outer_systemd_full_contract(
    writable_root: Path,
) -> dict[str, str]:
    """Return the common service contract plus its one writable evidence root."""

    if os.path.normpath(str(writable_root)) != str(writable_root):
        raise FormalSupervisorError(
            "outer_systemd_writable_root_invalid"
        )
    root = _safe_absolute_path(writable_root, allow_file=False)
    try:
        root.relative_to(Path("/var/tmp"))
    except ValueError as exc:
        raise FormalSupervisorError(
            "outer_systemd_writable_root_invalid"
        ) from exc
    metadata = root.lstat()
    if (
        root == Path("/var/tmp")
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise FormalSupervisorError(
            "outer_systemd_writable_root_invalid"
        )
    return dict(OUTER_SYSTEMD_CONTRACT)


def _outer_filesystem_namespace_probe(
    writable_root: Path,
) -> dict[str, Any]:
    """Prove the 311-compatible no-mount-namespace service topology."""

    _outer_systemd_full_contract(writable_root)
    root_descriptor = -1
    evidence_descriptor = -1
    try:
        root_descriptor = _open_absolute_directory(Path("/"))
        evidence_descriptor = _open_absolute_directory(
            writable_root
        )
        root_metadata = os.fstat(root_descriptor)
        evidence_metadata = os.fstat(evidence_descriptor)
    except (OSError, FormalSupervisorError) as exc:
        raise FormalSupervisorError(
            "outer_filesystem_namespace_probe_failed"
        ) from exc
    finally:
        if evidence_descriptor >= 0:
            os.close(evidence_descriptor)
        if root_descriptor >= 0:
            os.close(root_descriptor)
    evidence_identity = {
        "device": int(evidence_metadata.st_dev),
        "inode": int(evidence_metadata.st_ino),
        "mode": int(stat.S_IMODE(evidence_metadata.st_mode)),
        "uid": int(evidence_metadata.st_uid),
    }
    return {
        "evidence_root_directory_open_allowed": True,
        "evidence_root_identity_sha256": _content_hash(
            evidence_identity
        ),
        "filesystem_write_isolation_authority": (
            "inner_landlock_plus_same_fd_openat_custody"
        ),
        "mount_namespace_path_properties_enabled": False,
        "probe_performed_without_write": True,
        "root_directory_open_allowed": (
            stat.S_ISDIR(root_metadata.st_mode)
        ),
    }


def _current_unified_cgroup() -> str:
    try:
        raw = Path("/proc/self/cgroup").read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise FormalSupervisorError(
            "outer_systemd_cgroup_unavailable"
        ) from exc
    matches = [
        line[3:]
        for line in raw.splitlines()
        if line.startswith("0::/")
    ]
    if (
        len(matches) != 1
        or not matches[0].startswith("/")
        or "\x00" in matches[0]
        or ".." in PurePosixPath(matches[0]).parts
    ):
        raise FormalSupervisorError(
            "outer_systemd_cgroup_invalid"
        )
    return matches[0]


def _outer_service_unit_from_cgroup(control_group: str) -> str:
    # A user service is nested below the user-manager unit, so a real cgroup
    # normally contains both ``user@UID.service`` and the leaf workload
    # service.  Authority belongs to the leaf cgroup, not to the ancestor
    # manager.  Requiring exactly one ``.service`` component would therefore
    # reject every ordinary ``systemctl --user`` launch.
    components = PurePosixPath(control_group).parts
    candidate = components[-1] if components else ""
    if (
        not candidate.endswith(".service")
        or candidate.startswith("user@")
        or _SAFE_COMPONENT.fullmatch(candidate) is None
    ):
        raise FormalSupervisorError(
            "outer_systemd_service_unit_invalid"
        )
    return candidate


def _parse_systemd_show(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise FormalSupervisorError(
            "outer_systemd_show_invalid"
        ) from exc
    properties: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if (
            separator != "="
            or key not in _OUTER_SYSTEMD_LIVE_PROPERTIES
            or key in properties
            or "\x00" in value
        ):
            raise FormalSupervisorError(
                "outer_systemd_show_invalid"
            )
        # systemd exposes IPAddressDeny as a space-delimited set, but its
        # IPv4/IPv6 presentation order is not stable across manager restarts.
        # Bind the raw stdout hash separately and canonicalise only this
        # semantically unordered property for exact contract comparison.
        if key == "IPAddressDeny":
            tokens = value.split()
            if (
                len(tokens) == 2
                and set(tokens) == {"0.0.0.0/0", "::/0"}
            ):
                value = OUTER_SYSTEMD_CONTRACT["IPAddressDeny"]
        properties[key] = value
    if set(properties) != set(_OUTER_SYSTEMD_LIVE_PROPERTIES):
        raise FormalSupervisorError(
            "outer_systemd_show_incomplete"
        )
    return properties


def _outer_network_family_probe() -> dict[str, Any]:
    denied: dict[str, int] = {}
    for label, family in (
        ("AF_INET", socket.AF_INET),
        ("AF_INET6", socket.AF_INET6),
    ):
        candidate: socket.socket | None = None
        try:
            candidate = socket.socket(family, socket.SOCK_STREAM)
        except OSError as exc:
            if exc.errno not in {
                errno.EACCES,
                errno.EAFNOSUPPORT,
                errno.EPERM,
            }:
                raise FormalSupervisorError(
                    "outer_systemd_network_probe_invalid"
                ) from exc
            denied[label] = int(exc.errno)
        else:
            raise FormalSupervisorError(
                "outer_systemd_network_family_not_denied"
            )
        finally:
            if candidate is not None:
                candidate.close()
    unix_candidate: socket.socket | None = None
    try:
        unix_candidate = socket.socket(
            socket.AF_UNIX, socket.SOCK_STREAM
        )
    except OSError as exc:
        raise FormalSupervisorError(
            "outer_systemd_unix_family_unavailable"
        ) from exc
    finally:
        if unix_candidate is not None:
            unix_candidate.close()
    return {
        "AF_INET_socket_creation_denied": True,
        "AF_INET_socket_denial_errno": denied["AF_INET"],
        "AF_INET6_socket_creation_denied": True,
        "AF_INET6_socket_denial_errno": denied["AF_INET6"],
        "AF_UNIX_socket_creation_allowed": True,
        "network_endpoint_contacted": False,
    }


def _validate_outer_systemd_attestation(
    receipt: Mapping[str, Any],
    *,
    expected_writable_root: Path,
) -> Mapping[str, Any]:
    expected_contract = _outer_systemd_full_contract(
        expected_writable_root
    )
    expected_fields = {
        "active_state",
        "common_contract",
        "common_contract_self_hash",
        "contract",
        "contract_self_hash",
        "control_group_sha256",
        "filesystem_namespace_probe",
        "fragment_source_file_sha256",
        "invocation_id",
        "main_pid",
        "network_family_probe",
        "nrestarts",
        "properties",
        "private_tmp_tradeoff",
        "schema",
        "self_hash",
        "stable_binding_sha256",
        "sub_state",
        "systemctl_file_sha256",
        "systemd_show_stdout_sha256",
        "unit_id",
        "writable_root",
    }
    if set(receipt) != expected_fields:
        raise FormalSupervisorError(
            "outer_systemd_attestation_fields_invalid"
        )
    body = {
        key: value
        for key, value in receipt.items()
        if key != "self_hash"
    }
    properties = receipt.get("properties")
    probe = receipt.get("network_family_probe")
    filesystem_probe = receipt.get(
        "filesystem_namespace_probe"
    )
    if (
        receipt.get("schema") != OUTER_SYSTEMD_ATTESTATION_SCHEMA
        or receipt.get("common_contract")
        != OUTER_SYSTEMD_CONTRACT
        or receipt.get("common_contract_self_hash")
        != _content_hash(OUTER_SYSTEMD_CONTRACT)
        or receipt.get("contract") != expected_contract
        or receipt.get("contract_self_hash")
        != _content_hash(expected_contract)
        or receipt.get("writable_root")
        != str(expected_writable_root)
        or receipt.get("private_tmp_tradeoff")
        != _OUTER_PRIVATE_TMP_TRADEOFF
        or filesystem_probe
        != _outer_filesystem_namespace_probe(
            expected_writable_root
        )
        or not isinstance(properties, dict)
        or set(properties) != set(_OUTER_SYSTEMD_LIVE_PROPERTIES)
        or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            for key, value in properties.items()
        )
        or any(
            properties.get(key) != value
            for key, value in expected_contract.items()
        )
        or receipt.get("active_state")
        not in {"active", "activating"}
        or receipt.get("sub_state")
        not in {"running", "start"}
        or receipt.get("nrestarts") != 0
        or isinstance(receipt.get("main_pid"), bool)
        or not isinstance(receipt.get("main_pid"), int)
        or receipt["main_pid"] <= 0
        or not isinstance(receipt.get("unit_id"), str)
        or _SAFE_COMPONENT.fullmatch(receipt["unit_id"]) is None
        or not receipt["unit_id"].endswith(".service")
        or not isinstance(receipt.get("invocation_id"), str)
        or re.fullmatch(
            r"[0-9a-f]{32}", receipt["invocation_id"]
        )
        is None
        or not isinstance(probe, dict)
        or set(probe)
        != {
            "AF_INET_socket_creation_denied",
            "AF_INET_socket_denial_errno",
            "AF_INET6_socket_creation_denied",
            "AF_INET6_socket_denial_errno",
            "AF_UNIX_socket_creation_allowed",
            "network_endpoint_contacted",
        }
        or probe.get("AF_INET_socket_creation_denied") is not True
        or probe.get("AF_INET6_socket_creation_denied") is not True
        or probe.get("AF_UNIX_socket_creation_allowed") is not True
        or probe.get("network_endpoint_contacted") is not False
        or any(
            isinstance(probe.get(key), bool)
            or probe.get(key)
            not in {errno.EACCES, errno.EAFNOSUPPORT, errno.EPERM}
            for key in (
                "AF_INET_socket_denial_errno",
                "AF_INET6_socket_denial_errno",
            )
        )
        or _require_hash(
            receipt.get("control_group_sha256"),
            "outer_systemd_attestation_hash_invalid",
        )
        != receipt["control_group_sha256"]
        or _require_hash(
            receipt.get("fragment_source_file_sha256"),
            "outer_systemd_attestation_hash_invalid",
        )
        != receipt["fragment_source_file_sha256"]
        or _require_hash(
            receipt.get("systemctl_file_sha256"),
            "outer_systemd_attestation_hash_invalid",
        )
        != receipt["systemctl_file_sha256"]
        or _require_hash(
            receipt.get("systemd_show_stdout_sha256"),
            "outer_systemd_attestation_hash_invalid",
        )
        != receipt["systemd_show_stdout_sha256"]
        or _require_hash(
            receipt.get("stable_binding_sha256"),
            "outer_systemd_attestation_hash_invalid",
        )
        != _content_hash(
            {
                "contract_self_hash": receipt[
                    "contract_self_hash"
                ],
                "common_contract_self_hash": receipt[
                    "common_contract_self_hash"
                ],
                "control_group_sha256": receipt[
                    "control_group_sha256"
                ],
                "filesystem_namespace_probe": (
                    filesystem_probe
                ),
                "fragment_source_file_sha256": receipt[
                    "fragment_source_file_sha256"
                ],
                "invocation_id": receipt["invocation_id"],
                "main_pid": receipt["main_pid"],
                "network_family_probe": probe,
                "nrestarts": receipt["nrestarts"],
                "systemctl_file_sha256": receipt[
                    "systemctl_file_sha256"
                ],
                "unit_id": receipt["unit_id"],
                "writable_root": receipt["writable_root"],
            }
        )
        or _require_hash(
            receipt.get("self_hash"),
            "outer_systemd_attestation_self_hash_invalid",
        )
        != _content_hash(body)
    ):
        raise FormalSupervisorError(
            "outer_systemd_attestation_invalid"
        )
    return receipt


def _attest_current_outer_systemd_service(
    *, writable_root: Path
) -> Mapping[str, Any]:
    expected_contract = _outer_systemd_full_contract(writable_root)
    control_group = _current_unified_cgroup()
    unit_id = _outer_service_unit_from_cgroup(control_group)
    systemctl_path = _safe_absolute_path(
        _SYSTEMCTL, allow_file=True
    )
    command = [
        str(systemctl_path),
        "--user",
        "show",
        unit_id,
        "--no-pager",
    ]
    for key in _OUTER_SYSTEMD_LIVE_PROPERTIES:
        command.extend(("--property", key))
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FormalSupervisorError(
            "outer_systemd_show_failed"
        ) from exc
    if completed.returncode != 0:
        raise FormalSupervisorError(
            "outer_systemd_show_failed"
        )
    properties = _parse_systemd_show(completed.stdout)
    if (
        properties["Id"] != unit_id
        or properties["ControlGroup"] != control_group
        or properties["NRestarts"] != "0"
        or not properties["MainPID"].isdigit()
        or int(properties["MainPID"]) != os.getpid()
    ):
        raise FormalSupervisorError(
            "outer_systemd_live_identity_invalid"
        )
    try:
        fragment_target = Path(
            properties["FragmentPath"]
        ).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FormalSupervisorError(
            "outer_systemd_fragment_unavailable"
        ) from exc
    probe = _outer_network_family_probe()
    filesystem_probe = _outer_filesystem_namespace_probe(
        writable_root
    )
    stable_binding = {
        "contract_self_hash": _content_hash(
            expected_contract
        ),
        "common_contract_self_hash": _content_hash(
            OUTER_SYSTEMD_CONTRACT
        ),
        "control_group_sha256": hashlib.sha256(
            control_group.encode("utf-8")
        ).hexdigest(),
        "filesystem_namespace_probe": filesystem_probe,
        "fragment_source_file_sha256": (
            _hash_regular_absolute(fragment_target)
        ),
        "invocation_id": properties["InvocationID"],
        "main_pid": int(properties["MainPID"]),
        "network_family_probe": probe,
        "nrestarts": 0,
        "systemctl_file_sha256": _hash_regular_absolute(
            systemctl_path
        ),
        "unit_id": unit_id,
        "writable_root": str(writable_root),
    }
    body: dict[str, Any] = {
        "schema": OUTER_SYSTEMD_ATTESTATION_SCHEMA,
        "active_state": properties["ActiveState"],
        "sub_state": properties["SubState"],
        "common_contract": dict(OUTER_SYSTEMD_CONTRACT),
        "common_contract_self_hash": stable_binding[
            "common_contract_self_hash"
        ],
        "contract": dict(expected_contract),
        "contract_self_hash": stable_binding[
            "contract_self_hash"
        ],
        "writable_root": stable_binding["writable_root"],
        "private_tmp_tradeoff": _OUTER_PRIVATE_TMP_TRADEOFF,
        "filesystem_namespace_probe": filesystem_probe,
        "properties": dict(sorted(properties.items())),
        "unit_id": unit_id,
        "invocation_id": properties["InvocationID"],
        "main_pid": int(properties["MainPID"]),
        "nrestarts": 0,
        "control_group_sha256": stable_binding[
            "control_group_sha256"
        ],
        "fragment_source_file_sha256": stable_binding[
            "fragment_source_file_sha256"
        ],
        "systemctl_file_sha256": stable_binding[
            "systemctl_file_sha256"
        ],
        "systemd_show_stdout_sha256": hashlib.sha256(
            completed.stdout
        ).hexdigest(),
        "network_family_probe": probe,
        "stable_binding_sha256": _content_hash(
            stable_binding
        ),
    }
    receipt = {**body, "self_hash": _content_hash(body)}
    return _validate_outer_systemd_attestation(
        receipt,
        expected_writable_root=writable_root,
    )


def _walk_regular_files(root: Path) -> tuple[Path, ...]:
    """Walk one asset root without following or silently accepting symlinks."""

    _safe_absolute_path(root, allow_file=False)
    files: list[Path] = []
    for directory, names, filenames, directory_fd in os.fwalk(
        root, topdown=True, follow_symlinks=False
    ):
        directory_path = Path(directory)
        for name in tuple(names):
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise FormalSupervisorError("closure_tree_contains_symlink")
            if not stat.S_ISDIR(metadata.st_mode):
                raise FormalSupervisorError("closure_tree_entry_invalid")
        for name in filenames:
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise FormalSupervisorError("closure_tree_contains_symlink")
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise FormalSupervisorError("closure_tree_entry_invalid")
            files.append(directory_path / name)
            if len(files) > _MAX_CLOSURE_FILES:
                raise FormalSupervisorError("closure_tree_too_large")
    return tuple(sorted(files))


def _validate_pytest_wheel_bundle_manifest(
    path: Path,
) -> Mapping[str, Any]:
    raw = _read_regular_absolute_exact(
        path,
        expected_sha256=PYTEST_WHEEL_BUNDLE_FILE_SHA256,
        maximum_bytes=1024 * 1024,
    )
    try:
        receipt = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError()
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise FormalSupervisorError(
            "pytest_wheel_bundle_manifest_invalid"
        ) from exc
    if not isinstance(receipt, dict):
        raise FormalSupervisorError(
            "pytest_wheel_bundle_manifest_invalid"
        )
    body = dict(receipt)
    claimed = body.pop("self_sha256", None)
    wheels = receipt.get("wheels")
    observed_versions: dict[str, str] = {}
    if isinstance(wheels, list):
        for row in wheels:
            if not isinstance(row, dict):
                continue
            filename = row.get("filename")
            if not isinstance(filename, str):
                continue
            normalized = filename.split("-", 2)
            if len(normalized) >= 2:
                observed_versions[
                    normalized[0].replace("-", "_")
                ] = normalized[1]
    if (
        receipt.get("schema") != PYTEST_WHEEL_BUNDLE_SCHEMA
        or receipt.get("status")
        != "QUALIFIED_OFFLINE_WHEEL_BUNDLE"
        or receipt.get("claim_scope")
        != "offline_dependency_transport_only_no_evaluation"
        or receipt.get("pytest_version") != "8.3.3"
        or receipt.get("python_compatibility") != "3.10"
        or receipt.get("formal_measurement") is not False
        or receipt.get("effect_gate_added") is not False
        or receipt.get("official_source_open_count") != 0
        or receipt.get("network_evaluation_count") != 0
        or claimed != PYTEST_WHEEL_BUNDLE_SELF_SHA256
        or _content_hash(body) != claimed
        or observed_versions
        != _PYTEST_WHEEL_DISTRIBUTION_VERSIONS
    ):
        raise FormalSupervisorError(
            "pytest_wheel_bundle_manifest_invalid"
        )
    return {
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "path": str(path),
        "self_sha256": claimed,
        "distribution_versions": dict(
            sorted(observed_versions.items())
        ),
    }


def _pytest_inventory_probe_code(
    *, explicit_frozen_interpreter: bool
) -> str:
    """Compile the exact runtime-inventory probe executed by the test Python."""

    probe_pairs = json.dumps(
        list(_PYTEST_RUNTIME_DISTRIBUTIONS),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    known_absent_record_entries = json.dumps(
        list(_PYTEST_KNOWN_ABSENT_RECORD_ENTRIES),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return f"""
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import pathlib
import site
import sys

strict_inventory = {explicit_frozen_interpreter!r}

def pth_snapshot():
    roots = sorted(set(site.getsitepackages()))
    result = []
    for root in roots:
        for path in sorted(pathlib.Path(root).glob("*.pth")):
            resolved = path.resolve(strict=True)
            raw = resolved.read_bytes()
            text = raw.decode("utf-8", errors="strict")
            executable = False
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("import\\t"):
                    executable = True
            if executable:
                if strict_inventory:
                    raise RuntimeError("executable pth line forbidden")
                continue
            result.append({{
                "path": str(resolved),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }})
    return result

pht_before = pth_snapshot()
pairs = {probe_pairs}
known_absent_record_entries = {known_absent_record_entries}
rows = []
for module_name, distribution_name in pairs:
    importlib.import_module(module_name)
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RuntimeError("module origin unavailable")
    origin = pathlib.Path(spec.origin).resolve(strict=True)
    distribution = importlib.metadata.distribution(distribution_name)
    files = []
    for declared in sorted(
        distribution.files or (), key=lambda value: str(value)
    ):
        declared_path = str(declared)
        located = pathlib.Path(distribution.locate_file(declared))
        try:
            path = located.resolve(strict=True)
        except FileNotFoundError:
            path = located.resolve(strict=False)
            allowed_absent = (
                not strict_inventory
                or (
                    distribution_name == "pytest"
                    and distribution.version == "8.3.3"
                    and declared_path in known_absent_record_entries
                )
            )
            if not allowed_absent:
                raise RuntimeError("distribution entry missing")
            files.append({{
                "declared_path": declared_path,
                "path": str(path),
                "present": False,
                "sha256": None,
                "size": None,
            }})
            continue
        if not path.is_file():
            raise RuntimeError("distribution entry not regular")
        raw = path.read_bytes()
        files.append({{
            "declared_path": declared_path,
            "path": str(path),
            "present": True,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
        }})
    if not files:
        raise RuntimeError("distribution file set empty")
    encoded = json.dumps(
        files,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    rows.append({{
        "distribution": distribution_name,
        "distribution_content_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
        "module": module_name,
        "origin": str(origin),
        "origin_sha256": hashlib.sha256(origin.read_bytes()).hexdigest(),
        "version": distribution.version,
    }})
pht_after = pth_snapshot()
if pht_before != pht_after:
    raise RuntimeError("pth closure changed")
print(json.dumps({{
    "base_prefix": str(pathlib.Path(sys.base_prefix).resolve()),
    "distributions": rows,
    "executable": str(pathlib.Path(sys.executable).resolve()),
    "pth_files": pht_before,
    "prefix": str(pathlib.Path(sys.prefix).resolve()),
}}, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
"""


def run_source_free_tests(
    *,
    code_root: Path,
    test_files: Sequence[Path],
    extra_pytest_arguments: Sequence[str] = (),
    deselected_test_nodes: Sequence[str] = (),
    test_python: Path | None = None,
    pytest_wheel_bundle_manifest: Path | None = None,
) -> TestAttestation:
    """Execute one exact source-free test command in a bound interpreter."""

    _safe_absolute_path(code_root, allow_file=False)
    if not test_files:
        raise FormalSupervisorError("qualification_tests_empty")
    exact_tests: list[str] = []
    test_hashes: dict[str, str] = {}
    for path in test_files:
        _safe_absolute_path(path, allow_file=True)
        try:
            relative = path.relative_to(code_root)
        except ValueError as exc:
            raise FormalSupervisorError(
                "qualification_test_outside_code_root"
            ) from exc
        exact_tests.append(str(relative))
        test_hashes[str(path)] = _hash_regular_absolute(path)
    exact_test_set = set(exact_tests)
    deselected_nodes: list[str] = []
    for raw_node in deselected_test_nodes:
        node = str(raw_node)
        test_name, separator, test_suffix = node.partition("::")
        if (
            not separator
            or not test_suffix
            or test_name not in exact_test_set
            or "\x00" in node
            or any(character.isspace() for character in node)
            or node in deselected_nodes
        ):
            raise FormalSupervisorError(
                "qualification_deselected_node_invalid"
            )
        deselected_nodes.append(node)
    arguments = [str(value) for value in extra_pytest_arguments]
    if any(
        "\x00" in value
        or "cacheprovider" in value
        or value.startswith("--cache")
        or value in {"-c", "--config-file"}
        or value.startswith("--rootdir")
        or value.startswith("--deselect")
        for value in arguments
    ):
        raise FormalSupervisorError("qualification_argument_invalid")
    explicit_frozen_interpreter = test_python is not None
    wheel_bundle_binding: Mapping[str, Any] | None = None
    if explicit_frozen_interpreter:
        if pytest_wheel_bundle_manifest is None:
            raise FormalSupervisorError(
                "pytest_wheel_bundle_manifest_required"
            )
        wheel_bundle_binding = (
            _validate_pytest_wheel_bundle_manifest(
                pytest_wheel_bundle_manifest
            )
        )
    elif pytest_wheel_bundle_manifest is not None:
        raise FormalSupervisorError(
            "pytest_wheel_bundle_manifest_ambiguous"
        )
    candidate_python = (
        Path(sys.executable)
        if test_python is None
        else test_python
    )
    if (
        not isinstance(candidate_python, Path)
        or not candidate_python.is_absolute()
        or Path(os.path.abspath(os.fspath(candidate_python)))
        != candidate_python
    ):
        raise FormalSupervisorError("pytest_runner_unavailable")
    try:
        invocation_metadata = candidate_python.lstat()
        resolved_python = candidate_python.resolve(strict=True)
        resolved_metadata = resolved_python.stat()
    except (OSError, RuntimeError) as exc:
        raise FormalSupervisorError(
            "pytest_runner_unavailable"
        ) from exc
    if (
        not (
            stat.S_ISREG(invocation_metadata.st_mode)
            or stat.S_ISLNK(invocation_metadata.st_mode)
        )
        or not stat.S_ISREG(resolved_metadata.st_mode)
        or resolved_metadata.st_nlink < 1
    ):
        raise FormalSupervisorError("pytest_runner_unavailable")
    test_python_path = candidate_python
    probe_code = _pytest_inventory_probe_code(
        explicit_frozen_interpreter=explicit_frozen_interpreter
    )
    probe_environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "HOME": "/var/empty",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TEMP": os.environ.get("TMPDIR", "/tmp"),
        "TMP": os.environ.get("TMPDIR", "/tmp"),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        **OFFLINE_ENVIRONMENT,
    }
    probe_arguments = [str(test_python_path), "-B"]
    if explicit_frozen_interpreter:
        probe_arguments.append("-I")
        probe_environment["PYTHONNOUSERSITE"] = "1"
    else:
        pytest_spec = importlib.util.find_spec("pytest")
        if pytest_spec is None or pytest_spec.origin is None:
            raise FormalSupervisorError("pytest_runner_unavailable")
        probe_environment["PYTHONPATH"] = str(
            Path(pytest_spec.origin).resolve().parents[1]
        )
    probe_arguments.extend(("-c", probe_code))
    try:
        probe = subprocess.run(
            tuple(probe_arguments),
            cwd=code_root,
            env=probe_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FormalSupervisorError(
            "pytest_runner_unavailable"
        ) from exc
    try:
        probe_receipt = json.loads(
            probe.stdout.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError()
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise FormalSupervisorError(
            "pytest_runner_unavailable"
        ) from exc
    expected_probe_keys = {
        "base_prefix",
        "distributions",
        "executable",
        "pth_files",
        "prefix",
    }
    if (
        probe.returncode != 0
        or not isinstance(probe_receipt, dict)
        or set(probe_receipt) != expected_probe_keys
        or probe_receipt["executable"] != str(resolved_python)
        or not isinstance(probe_receipt["prefix"], str)
        or not isinstance(probe_receipt["base_prefix"], str)
        or not isinstance(probe_receipt["distributions"], list)
        or len(probe_receipt["distributions"])
        != len(_PYTEST_RUNTIME_DISTRIBUTIONS)
        or not isinstance(probe_receipt["pth_files"], list)
        or len(probe_receipt["pth_files"]) > 1_000
        or len(probe.stdout) > 8 * 1024 * 1024
    ):
        raise FormalSupervisorError("pytest_runner_unavailable")
    pyvenv_path: Path | None = None
    pyvenv_sha256: str | None = None
    if explicit_frozen_interpreter:
        candidate_venv_root = test_python_path.parent.parent
        pyvenv_path = candidate_venv_root / "pyvenv.cfg"
        _safe_absolute_path(pyvenv_path, allow_file=True)
        pyvenv_sha256 = _hash_regular_absolute(pyvenv_path)
        pyvenv_raw = _read_regular_absolute_exact(
            pyvenv_path,
            expected_sha256=pyvenv_sha256,
            maximum_bytes=64 * 1024,
        )
        try:
            pyvenv_lines = {
                key.strip().lower(): value.strip().lower()
                for line in pyvenv_raw.decode("ascii").splitlines()
                if line.strip() and "=" in line
                for key, value in (line.split("=", 1),)
            }
        except UnicodeError as exc:
            raise FormalSupervisorError(
                "pytest_runner_not_isolated_venv"
            ) from exc
        if (
            pyvenv_lines.get("include-system-site-packages")
            != "false"
        ):
            raise FormalSupervisorError(
                "pytest_runner_not_isolated_venv"
            )
        try:
            probe_prefix = Path(
                probe_receipt["prefix"]
            ).resolve(strict=True)
            probe_base_prefix = Path(
                probe_receipt["base_prefix"]
            ).resolve(strict=True)
            expected_prefix = candidate_venv_root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise FormalSupervisorError(
                "pytest_runner_unavailable"
            ) from exc
        if (
            probe_prefix != expected_prefix
            or probe_base_prefix == probe_prefix
        ):
            raise FormalSupervisorError(
                "pytest_runner_not_isolated_venv"
            )
    pth_file_hashes: dict[str, str] = {}
    for raw_pth in probe_receipt["pth_files"]:
        if (
            not isinstance(raw_pth, dict)
            or set(raw_pth) != {"path", "sha256", "size"}
            or not isinstance(raw_pth["path"], str)
            or not isinstance(raw_pth["sha256"], str)
            or not isinstance(raw_pth["size"], int)
            or isinstance(raw_pth["size"], bool)
            or not 1 <= raw_pth["size"] <= 1024 * 1024
        ):
            raise FormalSupervisorError(
                "pytest_pth_closure_invalid"
            )
        pth_path = _safe_absolute_path(
            Path(raw_pth["path"]), allow_file=True
        )
        pth_raw = _read_regular_absolute_exact(
            pth_path,
            expected_sha256=raw_pth["sha256"],
            maximum_bytes=1024 * 1024,
        )
        if len(pth_raw) != raw_pth["size"]:
            raise FormalSupervisorError(
                "pytest_pth_closure_invalid"
            )
        try:
            pth_text = pth_raw.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise FormalSupervisorError(
                "pytest_pth_closure_invalid"
            ) from exc
        if any(
            line.strip().startswith(("import ", "import\t"))
            for line in pth_text.splitlines()
        ):
            raise FormalSupervisorError(
                "pytest_pth_executable_line_forbidden"
            )
        if str(pth_path) in pth_file_hashes:
            raise FormalSupervisorError(
                "pytest_pth_closure_invalid"
            )
        pth_file_hashes[str(pth_path)] = raw_pth["sha256"]
    distribution_closures: list[dict[str, Any]] = []
    distribution_file_hashes: dict[str, str] = {}
    pytest_origin: Path | None = None
    pytest_version: str | None = None
    expected_pairs = tuple(_PYTEST_RUNTIME_DISTRIBUTIONS)
    for expected_pair, raw_distribution in zip(
        expected_pairs,
        probe_receipt["distributions"],
        strict=True,
    ):
        module_name, distribution_name = expected_pair
        if (
            not isinstance(raw_distribution, dict)
            or set(raw_distribution)
            != {
                "distribution",
                "distribution_content_sha256",
                "files",
                "module",
                "origin",
                "origin_sha256",
                "version",
            }
            or raw_distribution["module"] != module_name
            or raw_distribution["distribution"] != distribution_name
            or not isinstance(raw_distribution["version"], str)
            or not raw_distribution["version"]
            or not isinstance(raw_distribution["files"], list)
            or not 1 <= len(raw_distribution["files"]) <= 10_000
        ):
            raise FormalSupervisorError(
                "pytest_distribution_closure_invalid"
            )
        checked_entries: list[dict[str, Any]] = []
        checked_present_files: list[dict[str, Any]] = []
        checked_absent_entries: list[dict[str, str]] = []
        previous_declared_path: str | None = None
        for raw_file in raw_distribution["files"]:
            if (
                not isinstance(raw_file, dict)
                or set(raw_file)
                != {
                    "declared_path",
                    "path",
                    "present",
                    "sha256",
                    "size",
                }
                or not isinstance(raw_file["declared_path"], str)
                or not raw_file["declared_path"]
                or "\x00" in raw_file["declared_path"]
                or not isinstance(raw_file["path"], str)
                or type(raw_file["present"]) is not bool
                or (
                    previous_declared_path is not None
                    and raw_file["declared_path"]
                    < previous_declared_path
                )
            ):
                raise FormalSupervisorError(
                    "pytest_distribution_closure_invalid"
                )
            previous_declared_path = raw_file["declared_path"]
            if raw_file["present"]:
                if (
                    not isinstance(raw_file["sha256"], str)
                    or not isinstance(raw_file["size"], int)
                    or isinstance(raw_file["size"], bool)
                    or raw_file["size"] < 0
                ):
                    raise FormalSupervisorError(
                        "pytest_distribution_closure_invalid"
                    )
                path = _safe_absolute_path(
                    Path(raw_file["path"]), allow_file=True
                )
                actual_hash = _hash_regular_absolute(path)
                if (
                    raw_file["sha256"] != actual_hash
                    or path.stat().st_size != raw_file["size"]
                ):
                    raise FormalSupervisorError(
                        "pytest_distribution_closure_changed"
                    )
                previous = distribution_file_hashes.setdefault(
                    str(path), actual_hash
                )
                if previous != actual_hash:
                    raise FormalSupervisorError(
                        "pytest_distribution_closure_alias_invalid"
                    )
                checked_present_files.append(dict(raw_file))
            else:
                frozen_absence_invalid = (
                    explicit_frozen_interpreter
                    and (
                        distribution_name != "pytest"
                        or raw_distribution["version"] != "8.3.3"
                        or raw_file["declared_path"]
                        not in _PYTEST_KNOWN_ABSENT_RECORD_ENTRIES
                    )
                )
                if (
                    frozen_absence_invalid
                    or raw_file["sha256"] is not None
                    or raw_file["size"] is not None
                ):
                    raise FormalSupervisorError(
                        "pytest_distribution_missing_entry_invalid"
                    )
                absent_path = _validate_absent_absolute_path(
                    Path(raw_file["path"])
                )
                if str(absent_path) in distribution_file_hashes:
                    raise FormalSupervisorError(
                        "pytest_distribution_closure_alias_invalid"
                    )
                checked_absent_entries.append(
                    {
                        "declared_path": raw_file["declared_path"],
                        "path": str(absent_path),
                    }
                )
            checked_entries.append(dict(raw_file))
        if (
            raw_distribution["distribution_content_sha256"]
            != _content_hash(checked_entries)
        ):
            raise FormalSupervisorError(
                "pytest_distribution_closure_invalid"
            )
        origin = _safe_absolute_path(
            Path(raw_distribution["origin"]),
            allow_file=True,
        )
        if (
            raw_distribution["origin_sha256"]
            != _hash_regular_absolute(origin)
            or distribution_file_hashes.get(str(origin))
            != raw_distribution["origin_sha256"]
        ):
            raise FormalSupervisorError(
                "pytest_distribution_origin_invalid"
            )
        if module_name == "pytest":
            pytest_origin = origin
            pytest_version = raw_distribution["version"]
        distribution_closures.append(
            {
                key: value
                for key, value in raw_distribution.items()
                if key != "files"
            }
            | {
                "absent_entries": checked_absent_entries,
                "declared_entry_count": len(checked_entries),
                "file_count": len(checked_present_files),
                "present_file_count": len(checked_present_files),
            }
        )
    if pytest_origin is None or pytest_version is None:
        raise FormalSupervisorError("pytest_runner_unavailable")
    if wheel_bundle_binding is not None:
        observed_wheel_versions = {
            row["distribution"]: row["version"]
            for row in distribution_closures
            if row["distribution"]
            in _PYTEST_WHEEL_DISTRIBUTION_VERSIONS
        }
        if observed_wheel_versions != wheel_bundle_binding[
            "distribution_versions"
        ]:
            raise FormalSupervisorError(
                "pytest_wheel_distribution_version_mismatch"
            )
    command = [
        str(test_python_path),
        "-B",
    ]
    if explicit_frozen_interpreter:
        command.append("-I")
    command.extend(
        (
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-c",
            "/dev/null",
            "--rootdir",
            str(code_root),
            *exact_tests,
            *(
                f"--deselect={node}"
                for node in deselected_nodes
            ),
            *arguments,
        )
    )
    environment = {
        **probe_environment,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    code_tree_before = {
        str(path): _hash_regular_absolute(path)
        for path in _walk_regular_files(code_root)
    }
    completed = subprocess.run(
        command,
        cwd=code_root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=900,
    )
    code_tree_after = {
        str(path): _hash_regular_absolute(path)
        for path in _walk_regular_files(code_root)
    }
    if code_tree_after != code_tree_before:
        raise FormalSupervisorError(
            "source_free_tests_modified_code_tree"
        )
    receipt: dict[str, Any] = {
        "schema": TEST_ATTESTATION_SCHEMA,
        "status": (
            "SOURCE_FREE_TESTS_PASSED"
            if completed.returncode == 0
            else "SOURCE_FREE_TESTS_FAILED"
        ),
        "command": command,
        "cwd": str(code_root),
        "offline_environment": dict(sorted(OFFLINE_ENVIRONMENT.items())),
        "test_runner": {
            "interpreter_path": str(test_python_path),
            "interpreter_sha256": _hash_regular_absolute(
                resolved_python
            ),
            "interpreter_resolved_path": str(resolved_python),
            "interpreter_invocation_is_symlink": (
                stat.S_ISLNK(invocation_metadata.st_mode)
            ),
            "interpreter_invocation_binding_sha256": hashlib.sha256(
                (
                    os.readlink(test_python_path)
                    if stat.S_ISLNK(invocation_metadata.st_mode)
                    else str(test_python_path)
                ).encode("utf-8")
            ).hexdigest(),
            "pyvenv_config_path": (
                None if pyvenv_path is None else str(pyvenv_path)
            ),
            "pyvenv_config_sha256": pyvenv_sha256,
            "pytest_origin": str(pytest_origin),
            "pytest_origin_sha256": _hash_regular_absolute(pytest_origin),
            "pytest_version": pytest_version,
            "distribution_closures": distribution_closures,
            "distribution_file_sha256s": dict(
                sorted(distribution_file_hashes.items())
            ),
            "pth_file_sha256s": dict(
                sorted(pth_file_hashes.items())
            ),
            "pytest_wheel_bundle_manifest": (
                None
                if wheel_bundle_binding is None
                else dict(wheel_bundle_binding)
            ),
            "explicit_frozen_interpreter": (
                explicit_frozen_interpreter
            ),
            "isolated_mode": explicit_frozen_interpreter,
            "plugin_autoload_disabled": True,
            "bytecode_writes_disabled_by_cli": True,
            "pytest_config_file": "/dev/null",
            "pytest_rootdir": str(code_root),
            "cuda_visible_devices": "",
            "pythonpath_injected": (
                not explicit_frozen_interpreter
            ),
        },
        "test_file_sha256s": dict(sorted(test_hashes.items())),
        "deselected_test_nodes": list(deselected_nodes),
        "exit_code": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "code_tree_file_count": len(code_tree_before),
        "code_tree_sha256": _content_hash(code_tree_before),
        "code_tree_unchanged": True,
        "source_content_supplied": False,
        "official_source_access_count": 0,
        "formal_measurement_run": False,
    }
    receipt["self_hash"] = _content_hash(receipt)
    if completed.returncode != 0:
        raise FormalSupervisorError("source_free_tests_failed")
    return TestAttestation(receipt=receipt, _token=_TOKEN)


def _validate_test_runner_closure(
    test_runner: object,
    *,
    require_frozen: bool,
) -> None:
    if not isinstance(test_runner, dict):
        raise FormalSupervisorError(
            "qualification_test_runner_invalid"
        )
    explicit = test_runner.get("explicit_frozen_interpreter")
    if require_frozen and (
        explicit is not True
        or test_runner.get("isolated_mode") is not True
        or test_runner.get("pythonpath_injected") is not False
        or test_runner.get("cuda_visible_devices") != ""
        or test_runner.get(
            "bytecode_writes_disabled_by_cli"
        )
        is not True
        or test_runner.get("pytest_config_file") != "/dev/null"
    ):
        raise FormalSupervisorError(
            "frozen_qualification_test_runner_required"
        )
    if explicit is not True:
        return
    required_strings = (
        "interpreter_path",
        "interpreter_resolved_path",
        "interpreter_sha256",
        "interpreter_invocation_binding_sha256",
        "pyvenv_config_path",
        "pyvenv_config_sha256",
        "pytest_origin",
        "pytest_origin_sha256",
        "pytest_rootdir",
        "pytest_version",
    )
    if any(
        not isinstance(test_runner.get(key), str)
        or not test_runner[key]
        for key in required_strings
    ):
        raise FormalSupervisorError(
            "qualification_test_runner_invalid"
        )
    invocation_path = Path(test_runner["interpreter_path"])
    _safe_absolute_path(
        Path(test_runner["pytest_rootdir"]), allow_file=False
    )
    try:
        metadata = invocation_path.lstat()
        resolved = invocation_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FormalSupervisorError(
            "qualification_test_runner_changed"
        ) from exc
    binding_preimage = (
        os.readlink(invocation_path)
        if stat.S_ISLNK(metadata.st_mode)
        else str(invocation_path)
    )
    pyvenv_path = Path(test_runner["pyvenv_config_path"])
    distribution_files = test_runner.get(
        "distribution_file_sha256s"
    )
    pth_files = test_runner.get("pth_file_sha256s")
    closures = test_runner.get("distribution_closures")
    wheel_binding = test_runner.get(
        "pytest_wheel_bundle_manifest"
    )
    if (
        str(resolved) != test_runner["interpreter_resolved_path"]
        or test_runner.get(
            "interpreter_invocation_is_symlink"
        )
        != stat.S_ISLNK(metadata.st_mode)
        or _hash_regular_absolute(resolved)
        != test_runner["interpreter_sha256"]
        or hashlib.sha256(
            binding_preimage.encode("utf-8")
        ).hexdigest()
        != test_runner["interpreter_invocation_binding_sha256"]
        or _hash_regular_absolute(pyvenv_path)
        != test_runner["pyvenv_config_sha256"]
        or not isinstance(distribution_files, dict)
        or not distribution_files
        or not isinstance(pth_files, dict)
        or not isinstance(closures, list)
        or tuple(
            (
                row.get("module"),
                row.get("distribution"),
            )
            for row in closures
            if isinstance(row, dict)
        )
        != tuple(_PYTEST_RUNTIME_DISTRIBUTIONS)
    ):
        raise FormalSupervisorError(
            "qualification_test_runner_changed"
        )
    pyvenv_raw = _read_regular_absolute_exact(
        pyvenv_path,
        expected_sha256=test_runner["pyvenv_config_sha256"],
        maximum_bytes=64 * 1024,
    )
    try:
        pyvenv_settings = {
            key.strip().lower(): value.strip().lower()
            for line in pyvenv_raw.decode("ascii").splitlines()
            if line.strip() and "=" in line
            for key, value in (line.split("=", 1),)
        }
    except UnicodeError as exc:
        raise FormalSupervisorError(
            "qualification_test_runner_changed"
        ) from exc
    if (
        pyvenv_settings.get("include-system-site-packages")
        != "false"
    ):
        raise FormalSupervisorError(
            "qualification_test_runner_changed"
        )
    if (
        not isinstance(wheel_binding, dict)
        or not isinstance(wheel_binding.get("path"), str)
        or dict(
            _validate_pytest_wheel_bundle_manifest(
                Path(wheel_binding["path"])
            )
        )
        != wheel_binding
    ):
        raise FormalSupervisorError(
            "qualification_test_wheel_bundle_changed"
        )
    for raw_path, expected in distribution_files.items():
        if (
            not isinstance(raw_path, str)
            or not isinstance(expected, str)
            or _hash_regular_absolute(Path(raw_path)) != expected
        ):
            raise FormalSupervisorError(
                "qualification_test_distribution_changed"
            )
    for closure in closures:
        expected_closure_keys = {
            "absent_entries",
            "declared_entry_count",
            "distribution",
            "distribution_content_sha256",
            "file_count",
            "module",
            "origin",
            "origin_sha256",
            "present_file_count",
            "version",
        }
        if (
            not isinstance(closure, dict)
            or set(closure) != expected_closure_keys
            or not isinstance(closure["absent_entries"], list)
            or not isinstance(closure["declared_entry_count"], int)
            or isinstance(closure["declared_entry_count"], bool)
            or not isinstance(closure["present_file_count"], int)
            or isinstance(closure["present_file_count"], bool)
            or closure["file_count"]
            != closure["present_file_count"]
            or closure["declared_entry_count"]
            != (
                closure["present_file_count"]
                + len(closure["absent_entries"])
            )
            or closure["present_file_count"] < 1
            or not isinstance(
                closure["distribution_content_sha256"], str
            )
            or _SHA256.fullmatch(
                closure["distribution_content_sha256"]
            )
            is None
        ):
            raise FormalSupervisorError(
                "qualification_test_distribution_changed"
            )
        observed_absent_declarations: list[str] = []
        for absent in closure["absent_entries"]:
            if (
                not isinstance(absent, dict)
                or set(absent) != {"declared_path", "path"}
                or not isinstance(absent["declared_path"], str)
                or not isinstance(absent["path"], str)
                or absent["declared_path"]
                not in _PYTEST_KNOWN_ABSENT_RECORD_ENTRIES
                or closure["distribution"] != "pytest"
                or absent["declared_path"]
                in observed_absent_declarations
            ):
                raise FormalSupervisorError(
                    "qualification_test_distribution_changed"
                )
            absent_path = _validate_absent_absolute_path(
                Path(absent["path"])
            )
            if str(absent_path) in distribution_files:
                raise FormalSupervisorError(
                    "qualification_test_distribution_changed"
                )
            observed_absent_declarations.append(
                absent["declared_path"]
            )
        if observed_absent_declarations != sorted(
            observed_absent_declarations
        ):
            raise FormalSupervisorError(
                "qualification_test_distribution_changed"
            )
    for raw_path, expected in pth_files.items():
        if not isinstance(raw_path, str) or not isinstance(
            expected, str
        ):
            raise FormalSupervisorError(
                "qualification_test_pth_changed"
            )
        raw = _read_regular_absolute_exact(
            Path(raw_path),
            expected_sha256=expected,
            maximum_bytes=1024 * 1024,
        )
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise FormalSupervisorError(
                "qualification_test_pth_changed"
            ) from exc
        if any(
            line.strip().startswith(("import ", "import\t"))
            for line in text.splitlines()
        ):
            raise FormalSupervisorError(
                "qualification_test_pth_changed"
            )
    pytest_origin = Path(test_runner["pytest_origin"])
    if (
        distribution_files.get(str(pytest_origin))
        != test_runner["pytest_origin_sha256"]
        or _hash_regular_absolute(pytest_origin)
        != test_runner["pytest_origin_sha256"]
    ):
        raise FormalSupervisorError(
            "qualification_test_distribution_changed"
        )


def _module_name_for_local_file(path: Path, root: Path) -> str | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if relative.suffix != ".py":
        return None
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else None


def _local_module_index(code_roots: Sequence[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in code_roots:
        for path in _walk_regular_files(root):
            if path.suffix != ".py":
                continue
            name = _module_name_for_local_file(path, root)
            if name is not None:
                previous = index.setdefault(name, path)
                if previous != path:
                    raise FormalSupervisorError("closure_module_ambiguous")
    return index


def _enqueue_local_module_with_package_initializers(
    module_name: str,
    *,
    local_index: Mapping[str, Path],
    queue: list[tuple[str | None, Path]],
    scheduled_modules: set[str],
) -> None:
    """Queue a module and each concrete ancestor package initializer."""

    parts = module_name.split(".")
    for width in range(1, len(parts) + 1):
        ancestor = ".".join(parts[:width])
        path = local_index.get(ancestor)
        if path is None:
            continue
        if width < len(parts) and path.name != "__init__.py":
            raise FormalSupervisorError(
                "closure_package_initializer_invalid"
            )
        if ancestor not in scheduled_modules:
            scheduled_modules.add(ancestor)
            queue.append((ancestor, path))


def _imports_from_source(path: Path, module_name: str | None) -> set[str]:
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise FormalSupervisorError("closure_python_parse_failed") from exc
    result: set[str] = set()
    package_parts = (module_name or "").split(".")
    if path.name != "__init__.py" and package_parts:
        package_parts = package_parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base_parts = list(package_parts) if node.level else []
            if node.level:
                remove = node.level - 1
                if remove > len(base_parts):
                    raise FormalSupervisorError(
                        "closure_relative_import_invalid"
                    )
                if remove:
                    base_parts = base_parts[:-remove]
            if node.module:
                base_parts.extend(node.module.split("."))
            base = ".".join(base_parts)
            if base:
                result.add(base)
            for alias in node.names:
                if alias.name != "*":
                    child = ".".join([part for part in (base, alias.name) if part])
                    if child:
                        result.add(child)
    return result


def _distribution_for_module(module_name: str) -> list[dict[str, str]]:
    top = module_name.split(".", 1)[0]
    names = importlib.metadata.packages_distributions().get(top, [])
    result: list[dict[str, str]] = []
    for name in sorted(names):
        try:
            version = importlib.metadata.version(name)
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            result.append(
                {
                    "name": name,
                    "version": "UNKNOWN",
                    "metadata_sha256": "UNAVAILABLE",
                    "record_sha256": "UNAVAILABLE",
                }
            )
            continue
        metadata_text = distribution.read_text("METADATA")
        record_text = distribution.read_text("RECORD")
        result.append(
            {
                "name": name,
                "version": version,
                "metadata_sha256": (
                    hashlib.sha256(
                        metadata_text.encode("utf-8")
                    ).hexdigest()
                    if metadata_text is not None
                    else "UNAVAILABLE"
                ),
                "record_sha256": (
                    hashlib.sha256(
                        record_text.encode("utf-8")
                    ).hexdigest()
                    if record_text is not None
                    else "UNAVAILABLE"
                ),
            }
        )
    return result


def attest_runtime_closure(
    *,
    code_roots: Sequence[Path],
    entry_files: Sequence[Path],
    config_files: Sequence[Path],
    asset_roots: Sequence[Path],
    test_attestation: TestAttestation,
    support_module_files: Mapping[str, Path] | None = None,
) -> RuntimeClosure:
    """Bind reachable Python origins, runtime, distributions, configs/assets."""

    if (
        not isinstance(test_attestation, TestAttestation)
        or test_attestation._token is not _TOKEN
        or test_attestation.receipt.get("status")
        != "SOURCE_FREE_TESTS_PASSED"
    ):
        raise FormalSupervisorError("test_attestation_not_executed")
    test_body = dict(test_attestation.receipt)
    test_claimed = test_body.pop("self_hash", None)
    if (
        not isinstance(test_claimed, str)
        or _content_hash(test_body) != test_claimed
    ):
        raise FormalSupervisorError("test_attestation_changed")
    roots = tuple(_safe_absolute_path(path, allow_file=False) for path in code_roots)
    if not roots:
        raise FormalSupervisorError("closure_code_roots_empty")
    local_index = _local_module_index(roots)
    support_paths: dict[str, Path] = {}
    supplied_support = (
        {} if support_module_files is None else support_module_files
    )
    if not isinstance(supplied_support, Mapping):
        raise FormalSupervisorError(
            "closure_support_module_invalid"
        )
    for module_name, raw_path in supplied_support.items():
        if (
            not isinstance(module_name, str)
            or not module_name
            or any(
                not part.isidentifier()
                for part in module_name.split(".")
            )
            or not isinstance(raw_path, Path)
        ):
            raise FormalSupervisorError(
                "closure_support_module_invalid"
            )
        path = _safe_absolute_path(raw_path, allow_file=True)
        if path.suffix != ".py":
            raise FormalSupervisorError(
                "closure_support_module_invalid"
            )
        previous = local_index.setdefault(module_name, path)
        if previous != path:
            raise FormalSupervisorError(
                "closure_module_ambiguous"
            )
        support_paths[module_name] = path
    queue: list[tuple[str | None, Path]] = []
    scheduled_modules: set[str] = set()
    for entry in entry_files:
        _safe_absolute_path(entry, allow_file=True)
        module_name = next(
            (
                name
                for root in roots
                if (name := _module_name_for_local_file(entry, root))
                is not None
            ),
            None,
        )
        if module_name is None:
            raise FormalSupervisorError("closure_entry_outside_code_roots")
        _enqueue_local_module_with_package_initializers(
            module_name,
            local_index=local_index,
            queue=queue,
            scheduled_modules=scheduled_modules,
        )
    # Explicit support modules are part of the executable closure even when
    # they are loaded dynamically by a sandbox child and therefore have no
    # statically visible import edge from an entry file.  Queue every declared
    # support module (and its package initializers) so its bytes and imports
    # are attested rather than leaving a binding with no reachable hash.
    for module_name in sorted(support_paths):
        _enqueue_local_module_with_package_initializers(
            module_name,
            local_index=local_index,
            queue=queue,
            scheduled_modules=scheduled_modules,
        )

    origins: dict[str, dict[str, Any]] = {}
    file_hashes: dict[str, str] = {}
    processed_files: set[Path] = set()
    while queue:
        module_name, path = queue.pop(0)
        if path in processed_files:
            continue
        processed_files.add(path)
        path_hash = _hash_regular_absolute(path)
        file_hashes[str(path)] = path_hash
        if module_name:
            origins[module_name] = {
                "origin": str(path),
                "origin_sha256": path_hash,
                "origin_kind": "python_source",
                "distributions": _distribution_for_module(module_name),
            }
        for imported in sorted(_imports_from_source(path, module_name)):
            candidates = [imported]
            parts = imported.split(".")
            candidates.extend(
                ".".join(parts[:index])
                for index in range(len(parts) - 1, 0, -1)
            )
            local_name = next(
                (candidate for candidate in candidates if candidate in local_index),
                None,
            )
            if local_name is not None:
                _enqueue_local_module_with_package_initializers(
                    local_name,
                    local_index=local_index,
                    queue=queue,
                    scheduled_modules=scheduled_modules,
                )
                continue
            top = imported.split(".", 1)[0]
            if top in origins:
                continue
            try:
                spec = importlib.util.find_spec(top)
            except (ImportError, AttributeError, ValueError) as exc:
                raise FormalSupervisorError(
                    "closure_import_resolution_failed"
                ) from exc
            if spec is None:
                raise FormalSupervisorError(
                    "closure_import_resolution_failed"
                )
            origin = spec.origin
            record: dict[str, Any] = {
                "origin": origin,
                "origin_sha256": None,
                "origin_kind": (
                    "builtin_or_frozen"
                    if origin in {None, "built-in", "frozen"}
                    else "external_file"
                ),
                "distributions": _distribution_for_module(top),
            }
            if origin not in {None, "built-in", "frozen"}:
                origin_path = Path(origin)
                record["origin_sha256"] = _hash_regular_absolute(
                    origin_path.resolve()
                )
                file_hashes[str(origin_path.resolve())] = record[
                    "origin_sha256"
                ]
            origins[top] = record
            if len(origins) > _MAX_CLOSURE_FILES:
                raise FormalSupervisorError("closure_module_count_exceeded")

    config_hashes: dict[str, str] = {}
    for path in config_files:
        config_hashes[str(path)] = _hash_regular_absolute(path)
        file_hashes[str(path)] = config_hashes[str(path)]
    asset_hashes: dict[str, str] = {}
    for root in asset_roots:
        for path in _walk_regular_files(root):
            asset_hashes[str(path)] = _hash_regular_absolute(path)
            file_hashes[str(path)] = asset_hashes[str(path)]
    for raw_path, expected_hash in test_attestation.receipt[
        "test_file_sha256s"
    ].items():
        if _hash_regular_absolute(Path(raw_path)) != expected_hash:
            raise FormalSupervisorError(
                "test_file_changed_before_closure"
            )
        file_hashes[raw_path] = expected_hash
    pytest_origin = test_attestation.receipt["test_runner"][
        "pytest_origin"
    ]
    pytest_hash = test_attestation.receipt["test_runner"][
        "pytest_origin_sha256"
    ]
    if _hash_regular_absolute(Path(pytest_origin)) != pytest_hash:
        raise FormalSupervisorError("test_runner_changed_before_closure")
    file_hashes[pytest_origin] = pytest_hash
    test_runner = test_attestation.receipt["test_runner"]
    for field in (
        "distribution_file_sha256s",
        "pth_file_sha256s",
    ):
        for raw_path, expected_hash in test_runner.get(
            field, {}
        ).items():
            if _hash_regular_absolute(Path(raw_path)) != expected_hash:
                raise FormalSupervisorError(
                    "test_runner_changed_before_closure"
                )
            file_hashes[raw_path] = expected_hash
    if test_runner.get("explicit_frozen_interpreter") is True:
        for raw_path, expected_hash in (
            (
                test_runner["interpreter_resolved_path"],
                test_runner["interpreter_sha256"],
            ),
            (
                test_runner["pyvenv_config_path"],
                test_runner["pyvenv_config_sha256"],
            ),
        ):
            if _hash_regular_absolute(Path(raw_path)) != expected_hash:
                raise FormalSupervisorError(
                    "test_runner_changed_before_closure"
                )
            file_hashes[raw_path] = expected_hash
        wheel_binding = test_runner[
            "pytest_wheel_bundle_manifest"
        ]
        file_hashes[wheel_binding["path"]] = wheel_binding[
            "file_sha256"
        ]

    resolved_interpreter = Path(sys.executable).resolve()
    interpreter_hash = _hash_regular_absolute(resolved_interpreter)
    file_hashes[str(resolved_interpreter)] = interpreter_hash
    runtime_roots = sorted(
        {
            str(
                _safe_absolute_path(
                    Path(raw_root).resolve(), allow_file=False
                )
            )
            for raw_root in (
                sys.prefix,
                sys.exec_prefix,
                sys.base_prefix,
                sys.base_exec_prefix,
                resolved_interpreter.parent.parent,
            )
        }
    )
    support_bindings = {
        module_name: {
            "path": str(path),
            "sha256": file_hashes.get(str(path)),
        }
        for module_name, path in sorted(support_paths.items())
    }
    if any(
        not isinstance(binding["sha256"], str)
        for binding in support_bindings.values()
    ):
        raise FormalSupervisorError(
            "closure_support_module_unreachable"
        )
    body: dict[str, Any] = {
        "schema": CLOSURE_SCHEMA,
        "status": "EXECUTED_SOURCE_FREE_RUNTIME_CLOSURE",
        "code_roots": [str(root) for root in roots],
        "support_roots": sorted(
            {str(path.parent) for path in support_paths.values()}
        ),
        "support_module_files": support_bindings,
        "runtime_roots": runtime_roots,
        "entry_files": [str(path) for path in entry_files],
        "python_origins": dict(sorted(origins.items())),
        "config_sha256s": dict(sorted(config_hashes.items())),
        "asset_roots": [str(path) for path in asset_roots],
        "asset_sha256s": dict(sorted(asset_hashes.items())),
        "interpreter": {
            "invoked_path": sys.executable,
            "resolved_path": str(resolved_interpreter),
            "sha256": interpreter_hash,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": sys.implementation.cache_tag,
            "platform": platform.platform(),
        },
        "test_attestation": dict(test_attestation.receipt),
        "source_content_supplied": False,
        "formal_measurement_run": False,
    }
    body["self_hash"] = _content_hash(body)
    return RuntimeClosure(
        manifest=body,
        file_hashes=dict(sorted(file_hashes.items())),
        _token=_TOKEN,
    )


def _revalidate_closure(closure: RuntimeClosure) -> None:
    if not isinstance(closure, RuntimeClosure) or closure._token is not _TOKEN:
        raise FormalSupervisorError("runtime_closure_not_attested")
    body = dict(closure.manifest)
    claimed = body.pop("self_hash", None)
    if (
        closure.manifest.get("schema") != CLOSURE_SCHEMA
        or closure.manifest.get("status")
        != "EXECUTED_SOURCE_FREE_RUNTIME_CLOSURE"
        or not isinstance(claimed, str)
        or _content_hash(body) != claimed
    ):
        raise FormalSupervisorError("runtime_closure_manifest_invalid")
    for raw_path, expected_hash in closure.file_hashes.items():
        if _hash_regular_absolute(Path(raw_path)) != expected_hash:
            raise FormalSupervisorError("runtime_closure_changed")
    for field in ("code_roots", "support_roots", "runtime_roots"):
        roots = closure.manifest.get(field)
        if not isinstance(roots, list):
            raise FormalSupervisorError(
                "runtime_closure_roots_invalid"
            )
        for raw_root in roots:
            if not isinstance(raw_root, str):
                raise FormalSupervisorError(
                    "runtime_closure_roots_invalid"
                )
            _safe_absolute_path(
                Path(raw_root), allow_file=False
            )
    _validate_test_runner_closure(
        closure.manifest.get("test_attestation", {}).get(
            "test_runner"
        ),
        require_frozen=False,
    )


def _validate_stable_support_closure(
    closure: RuntimeClosure,
) -> dict[str, str]:
    expected_bindings = {
        module_name: {
            "path": str(_INTERNAL_SUPPORT_MODULE_PATHS[module_name]),
            "sha256": expected_hash,
        }
        for module_name, expected_hash in sorted(
            _STABLE_SUPPORT_MODULE_SHA256S.items()
        )
    }
    if (
        closure.manifest.get("support_module_files")
        != expected_bindings
        or closure.manifest.get("support_roots")
        != [str(_WORKSPACE_ROOT / "assumption_os")]
    ):
        raise FormalSupervisorError(
            "stable_support_runtime_closure_missing"
        )
    for module_name, binding in expected_bindings.items():
        path = _INTERNAL_SUPPORT_MODULE_PATHS[module_name]
        if (
            closure.file_hashes.get(str(path))
            != binding["sha256"]
            or _hash_regular_absolute(path) != binding["sha256"]
        ):
            raise FormalSupervisorError(
                "stable_support_runtime_closure_hash_drifted"
            )
    return dict(sorted(_STABLE_SUPPORT_MODULE_SHA256S.items()))


class FormalSupervisor:
    """Stateful supervisor whose formal constructor always uses FORMAL_ROOT."""

    def __init__(self, *, _root: Path | None = None, _test_token: object = None):
        if _root is None:
            root = FORMAL_ROOT
            lineage = "formal_fixed_root"
        elif _test_token is _TOKEN:
            root = _root
            lineage = "synthetic_source_free_qualification"
        else:
            raise FormalSupervisorError("caller_selected_root_forbidden")
        _create_fixed_private_root_once(root)
        self.store = SecureDirectory(root)
        self.root = root
        self.lineage = lineage
        for relative in (
            "source",
            "state",
            "state/predictions",
            "private",
            "private/custodian",
            "work",
            "work/arms",
            "work/scorer",
        ):
            self.store.ensure_directory(relative)

    @classmethod
    def _source_free_qualification(cls, root: Path) -> "FormalSupervisor":
        return cls(_root=root, _test_token=_TOKEN)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "FormalSupervisor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def freeze_action_once(
        self,
        *,
        closure: RuntimeClosure,
        arm_commands: Sequence[ArmCommand],
        scorer_command: ScorerCommand,
        freeze_commitments: Mapping[str, str],
        source_sha256: str,
    ) -> FrozenAction:
        if self.lineage == "formal_fixed_root":
            raise FormalSupervisorError(
                "formal_external_arm_commands_forbidden"
            )
        _revalidate_closure(closure)
        _require_hash(source_sha256, "action_source_hash_invalid")
        if set(freeze_commitments) == set() or any(
            _require_hash(value, "freeze_commitment_invalid") != value
            for value in freeze_commitments.values()
        ):
            raise FormalSupervisorError("freeze_commitments_empty")
        by_arm: dict[str, Any] = {}
        for command in arm_commands:
            if (
                not isinstance(command, ArmCommand)
                or command.arm_id not in protocol.ARM_IDS
                or command.arm_id in by_arm
                or not command.command_template
            ):
                raise FormalSupervisorError("arm_command_invalid")
            _require_hash(
                command.implementation_sha256,
                "arm_implementation_hash_invalid",
            )
            implementation_path = _safe_absolute_path(
                command.implementation_path, allow_file=True
            )
            if (
                _hash_regular_absolute(implementation_path)
                != command.implementation_sha256
                or str(implementation_path)
                not in closure.file_hashes
                or closure.file_hashes[str(implementation_path)]
                != command.implementation_sha256
            ):
                raise FormalSupervisorError(
                    "arm_implementation_not_in_closure"
                )
            for path in (*command.code_roots, *command.model_roots):
                _safe_absolute_path(path, allow_file=False)
            if not any(
                implementation_path.is_relative_to(root)
                for root in command.code_roots
            ):
                raise FormalSupervisorError(
                    "arm_implementation_outside_code_roots"
                )
            if not set(map(str, command.code_roots)).issubset(
                set(closure.manifest["code_roots"])
            ) or not set(map(str, command.model_roots)).issubset(
                set(closure.manifest["asset_roots"])
            ):
                raise FormalSupervisorError(
                    "arm_allowlist_not_in_closure"
                )
            if any(
                not isinstance(value, str)
                or not value
                or "\x00" in value
                for value in command.command_template
            ):
                raise FormalSupervisorError("arm_command_invalid")
            executable = _safe_absolute_path(
                Path(command.command_template[0]).resolve(),
                allow_file=True,
            )
            by_arm[command.arm_id] = {
                "command_template": list(command.command_template),
                "code_roots": [str(path) for path in command.code_roots],
                "model_roots": [str(path) for path in command.model_roots],
                "implementation_path": str(implementation_path),
                "implementation_sha256": command.implementation_sha256,
                "command_executable_path": str(executable),
                "command_executable_sha256": _hash_regular_absolute(
                    executable
                ),
            }
        if set(by_arm) != set(protocol.ARM_IDS):
            raise FormalSupervisorError("four_arm_command_set_incomplete")
        if not isinstance(scorer_command, ScorerCommand):
            raise FormalSupervisorError("scorer_command_invalid")
        scorer_template = list(scorer_command.command_template)
        if not scorer_template or any(
            not isinstance(value, str)
            or not value
            or "\x00" in value
            for value in scorer_template
        ):
            raise FormalSupervisorError("scorer_command_invalid")
        scorer_implementation = _safe_absolute_path(
            scorer_command.implementation_path, allow_file=True
        )
        if (
            _require_hash(
                scorer_command.implementation_sha256,
                "scorer_implementation_hash_invalid",
            )
            != _hash_regular_absolute(scorer_implementation)
            or closure.file_hashes.get(str(scorer_implementation))
            != scorer_command.implementation_sha256
        ):
            raise FormalSupervisorError(
                "scorer_implementation_not_in_closure"
            )
        scorer_executable = _safe_absolute_path(
            Path(scorer_template[0]).resolve(), allow_file=True
        )
        receipt: dict[str, Any] = {
            "schema": ACTION_SCHEMA,
            "status": "EXECUTABLE_SUPERVISOR_ACTION_FROZEN",
            "supervisor_version": VERSION,
            "lineage": self.lineage,
            "root_identity": list(self.store.identity),
            "source_sha256": source_sha256,
            "runtime_closure_self_hash": closure.manifest["self_hash"],
            "arm_commands": dict(sorted(by_arm.items())),
            "scorer_command": {
                "command_template": scorer_template,
                "implementation_path": str(scorer_implementation),
                "implementation_sha256": (
                    scorer_command.implementation_sha256
                ),
                "command_executable_path": str(scorer_executable),
                "command_executable_sha256": _hash_regular_absolute(
                    scorer_executable
                ),
            },
            "freeze_commitments": dict(sorted(freeze_commitments.items())),
            "legacy_freeze_ready_receipt_is_authority": False,
            "formal_authority": (
                "qualification_only_external_subprocess_harness"
            ),
            "formal_requires_supervisor_internal_item_factory": True,
            "effect_gate_added": False,
        }
        receipt["self_hash"] = _content_hash(receipt)
        self.store.write_json_exclusive("state/action.freeze.json", receipt)
        return FrozenAction(
            receipt=receipt,
            root_identity=self.store.identity,
            closure=closure,
            _token=_TOKEN,
        )

    def freeze_internal_factory_action_once(
        self,
        *,
        closure: RuntimeClosure,
        freeze_commitments: Mapping[str, str],
        qwen_model_root: Path,
        qwen_model_manifest: Path,
        qwen_actual_canary_lineage_terminal: Path,
        qwen_runtime_qualification_receipts: Sequence[Path],
        internal_factory_qualification_receipt: Path,
        minilm_model_root: Path,
        minilm_asset_manifest: Path,
        minilm_target_manifest: Path,
    ) -> FrozenAction:
        """Freeze the only formal action path after qualification.

        No arm command, scorer callable, prediction, result, commitment
        receipt, or ``freeze_ready`` object is accepted from the caller.  The
        two public model roots/manifests are scientific inputs whose complete
        bytes must already be present in the executed closure.
        """

        if self.lineage != "formal_fixed_root":
            raise FormalSupervisorError(
                "formal_action_requires_fixed_root"
            )
        outer_service = _attest_current_outer_systemd_service(
            writable_root=self.root
        )
        _revalidate_closure(closure)
        if set(freeze_commitments) == set() or any(
            _require_hash(value, "freeze_commitment_invalid") != value
            for value in freeze_commitments.values()
        ):
            raise FormalSupervisorError("freeze_commitments_empty")
        implementation_hashes: dict[str, str] = {}
        for component, path in _INTERNAL_FORMAL_IMPLEMENTATION_PATHS.items():
            expected = closure.file_hashes.get(str(path))
            actual = _hash_regular_absolute(path)
            if expected != actual:
                raise FormalSupervisorError(
                    "internal_factory_implementation_not_in_closure"
                )
            implementation_hashes[component] = actual
        if any(
            implementation_hashes.get(component) != expected
            for component, expected in _STABLE_EXTRACTOR_SHA256S.items()
        ) or any(
            implementation_hashes.get(component) != expected
            for component, expected in _STABLE_MINILM_SHA256S.items()
        ):
            raise FormalSupervisorError(
                "stable_internal_runtime_closure_hash_drifted"
            )
        support_hashes = _validate_stable_support_closure(closure)
        attested_tests = closure.manifest["test_attestation"][
            "test_file_sha256s"
        ]
        if any(
            attested_tests.get(
                str(_INTERNAL_QUALIFICATION_TEST_PATHS[component])
            )
            != expected
            for component, expected in (
                _STABLE_QUALIFICATION_TEST_SHA256S.items()
            )
        ):
            raise FormalSupervisorError(
                "stable_extractor_tests_not_executed"
            )
        _validate_test_runner_closure(
            closure.manifest["test_attestation"].get(
                "test_runner"
            ),
            require_frozen=True,
        )
        model_roots = (
            _safe_absolute_path(qwen_model_root, allow_file=False),
            _safe_absolute_path(minilm_model_root, allow_file=False),
        )
        manifests = (
            _safe_absolute_path(qwen_model_manifest, allow_file=True),
            _safe_absolute_path(minilm_asset_manifest, allow_file=True),
            _safe_absolute_path(minilm_target_manifest, allow_file=True),
        )
        canary_lineage_path = _safe_absolute_path(
            qwen_actual_canary_lineage_terminal,
            allow_file=True,
        )
        canary_lineage_raw = _read_regular_absolute_exact(
            canary_lineage_path,
            expected_sha256=(
                CLOSED_CHOICE_ACTUAL_CANARY_LINEAGE_FILE_SHA256
            ),
            maximum_bytes=2 * 1024 * 1024,
        )
        canary_lineage_binding = (
            _validate_closed_choice_actual_canary_lineage(
                canary_lineage_raw,
                expected_model_manifest_sha256=(
                    _hash_regular_absolute(manifests[0])
                ),
            )
        )
        minilm_target_raw = _read_regular_absolute_exact(
            manifests[2],
            expected_sha256=_hash_regular_absolute(manifests[2]),
            maximum_bytes=2 * 1024 * 1024,
        )
        minilm_target = _validate_minilm_target_manifest_bytes(
            minilm_target_raw
        )
        if (
            not isinstance(qwen_runtime_qualification_receipts, Sequence)
            or isinstance(
                qwen_runtime_qualification_receipts, (str, bytes)
            )
            or len(qwen_runtime_qualification_receipts)
            != len(QWEN_CUDA_VISIBLE_DEVICES)
        ):
            raise FormalSupervisorError(
                "qwen_runtime_qualification_receipts_invalid"
            )
        qualification_receipt_paths = tuple(
            _safe_absolute_path(Path(path), allow_file=True)
            for path in qwen_runtime_qualification_receipts
        )
        internal_qualification_path = _safe_absolute_path(
            internal_factory_qualification_receipt,
            allow_file=True,
        )
        if len(set(qualification_receipt_paths)) != len(
            qualification_receipt_paths
        ):
            raise FormalSupervisorError(
                "qwen_runtime_qualification_receipts_invalid"
            )
        closure_asset_roots = set(closure.manifest["asset_roots"])
        closure_configs = closure.manifest["config_sha256s"]
        if (
            not set(map(str, model_roots)).issubset(closure_asset_roots)
            or any(
                closure_configs.get(str(path))
                != _hash_regular_absolute(path)
                for path in manifests
            )
            or any(
                closure_configs.get(str(path))
                != _hash_regular_absolute(path)
                for path in qualification_receipt_paths
            )
            or closure_configs.get(str(canary_lineage_path))
            != CLOSED_CHOICE_ACTUAL_CANARY_LINEAGE_FILE_SHA256
            or closure_configs.get(str(internal_qualification_path))
            != _hash_regular_absolute(internal_qualification_path)
        ):
            raise FormalSupervisorError(
                "internal_factory_assets_not_in_closure"
            )
        qwen_runtime_bindings: dict[str, Mapping[str, Any]] = {}
        qualification_receipts: dict[
            str, tuple[Path, Mapping[str, Any]]
        ] = {}
        for device, path in zip(
            QWEN_CUDA_VISIBLE_DEVICES,
            qualification_receipt_paths,
            strict=True,
        ):
            qualification_raw = _read_regular_absolute_exact(
                path,
                expected_sha256=closure_configs[str(path)],
                maximum_bytes=16 * 1024 * 1024,
            )
            qualification_receipt = _parse_json(
                qualification_raw,
                issue_id=(
                    "qwen_runtime_qualification_receipt_invalid"
                ),
            )
            qwen_runtime_bindings[device] = (
                _validate_qwen_runtime_safe_receipt(
                    qualification_receipt,
                    expected_model_manifest_sha256=(
                        _hash_regular_absolute(manifests[0])
                    ),
                    expected_visible_device=device,
                    expected_lineage="source_free_qualification",
                )
            )
            qualification_receipts[device] = (
                path,
                qualification_receipt,
            )
        common_qwen_runtime_bindings = {
            _content_hash(
                {
                    key: value
                    for key, value in binding.items()
                    if key != "logical_gpu_binding"
                }
            )
            for binding in qwen_runtime_bindings.values()
        }
        if (
            len(common_qwen_runtime_bindings) != 1
            or len(
                {
                    (
                        binding["logical_gpu_binding"][
                            "logical_device_name"
                        ],
                        tuple(
                            binding["logical_gpu_binding"][
                                "logical_compute_capability"
                            ]
                        ),
                    )
                    for binding in qwen_runtime_bindings.values()
                }
            )
            != 1
            or len(
                {
                    binding["logical_gpu_binding"][
                        "logical_device_uuid"
                    ]
                    for binding in qwen_runtime_bindings.values()
                }
            )
            != len(qwen_runtime_bindings)
        ):
            raise FormalSupervisorError(
                "qwen_cross_gpu_runtime_binding_mismatch"
            )
        internal_qualification_raw = _read_regular_absolute_exact(
            internal_qualification_path,
            expected_sha256=closure_configs[
                str(internal_qualification_path)
            ],
            maximum_bytes=16 * 1024 * 1024,
        )
        internal_qualification = _parse_json(
            internal_qualification_raw,
            issue_id=(
                "internal_factory_qualification_receipt_invalid"
            ),
        )
        internal_qualification_body = dict(
            internal_qualification
        )
        internal_qualification_claimed = (
            internal_qualification_body.pop("self_hash", None)
        )
        expected_internal_qualification_fields = {
            "common_item_count",
            "closed_choice_selection_count",
            "effect_gate_added",
            "efficacy_evidence",
            "factory_encoder_binding",
            "factory_execution_receipt_self_hash",
            "factory_output_file_sha256",
            "factory_output_self_hash",
            "factory_receipt_self_hash",
            "formal_measurement_authorized",
            "formal_result",
            "formal_root_used",
            "free_form_generation_count",
            "four_arm_barrier_self_hash",
            "internal_implementation_sha256s",
            "item_content_emitted",
            "label_open_count",
            "minilm_asset_manifest_sha256",
            "minilm_target_manifest_file_sha256",
            "minilm_target_manifest_self_sha256",
            "official_source_access_count",
            "one_shot_key",
            "online_or_api_evaluation_count",
            "outer_systemd_attestation_self_hash",
            "outer_systemd_common_contract_self_hash",
            "outer_systemd_contract",
            "outer_systemd_contract_self_hash",
            "outer_systemd_invocation_id",
            "outer_systemd_private_tmp_tradeoff",
            "outer_systemd_stable_binding_sha256",
            "outer_systemd_unit_id",
            "outer_systemd_writable_root",
            "qualification_action_self_hash",
            "qualification_runtime_closure_self_hash",
            "qwen_model_manifest_sha256",
            "qwen_actual_canary_lineage_binding",
            "qwen_parallel_runtime_receipt_self_hash",
            "qwen_private_safe_receipts",
            "qwen_runtime_bindings",
            "score_operation",
            "schema",
            "self_hash",
            "official_source_content_supplied_to_model",
            "public_synthetic_content_supplied_to_model",
            "status",
            "synthetic_source_receipt_self_hash",
            "synthetic_source_sha256",
        }
        safe_rows = internal_qualification.get(
            "qwen_private_safe_receipts"
        )
        qualification_writable_root_raw = (
            internal_qualification.get(
                "outer_systemd_writable_root"
            )
        )
        if (
            not isinstance(qualification_writable_root_raw, str)
            or not Path(qualification_writable_root_raw).is_absolute()
        ):
            raise FormalSupervisorError(
                "internal_factory_qualification_receipt_invalid"
            )
        qualification_writable_root = Path(
            qualification_writable_root_raw
        )
        try:
            qualification_outer_contract = (
                _outer_systemd_full_contract(
                    qualification_writable_root
                )
            )
        except FormalSupervisorError as exc:
            raise FormalSupervisorError(
                "internal_factory_qualification_receipt_invalid"
            ) from exc
        if (
            set(internal_qualification)
            != expected_internal_qualification_fields
            or internal_qualification.get("schema")
            != INTERNAL_FACTORY_QUALIFICATION_SCHEMA
            or internal_qualification.get("status")
            != (
                "PASS_SOURCE_FREE_EXACT_INTERNAL_FACTORY_"
                "QUALIFICATION"
            )
            or not isinstance(
                internal_qualification_claimed, str
            )
            or _content_hash(internal_qualification_body)
            != internal_qualification_claimed
            or internal_qualification.get(
                "internal_implementation_sha256s"
            )
            != dict(sorted(implementation_hashes.items()))
            or internal_qualification.get(
                "qwen_model_manifest_sha256"
            )
            != _hash_regular_absolute(manifests[0])
            or internal_qualification.get(
                "qwen_actual_canary_lineage_binding"
            )
            != dict(canary_lineage_binding)
            or internal_qualification.get(
                "minilm_asset_manifest_sha256"
            )
            != _hash_regular_absolute(manifests[1])
            or internal_qualification.get(
                "minilm_target_manifest_file_sha256"
            )
            != hashlib.sha256(minilm_target_raw).hexdigest()
            or internal_qualification.get(
                "minilm_target_manifest_self_sha256"
            )
            != minilm_target["self_sha256"]
            or internal_qualification.get(
                "qwen_runtime_bindings"
            )
            != dict(sorted(qwen_runtime_bindings.items()))
            or internal_qualification.get(
                "closed_choice_selection_count"
            )
            != 3 * internal_qualification.get(
                "common_item_count", -1
            )
            or internal_qualification.get(
                "free_form_generation_count"
            )
            != 0
            or internal_qualification.get("score_operation")
            != "teacher_forced_forward_log_softmax"
            or not isinstance(safe_rows, dict)
            or set(safe_rows)
            != set(QWEN_CUDA_VISIBLE_DEVICES)
            or any(
                not isinstance(safe_rows.get(device), dict)
                or set(safe_rows[device])
                != {
                    "file_sha256",
                    "path",
                    "sandbox_receipt_self_hash",
                    "self_sha256",
                }
                or not isinstance(
                    safe_rows[device].get("path"), str
                )
                or not Path(safe_rows[device]["path"]).is_absolute()
                or safe_rows[device].get("file_sha256")
                != closure_configs[str(path)]
                or safe_rows[device].get("self_sha256")
                != qualification_receipt["self_sha256"]
                or _require_hash(
                    safe_rows[device].get(
                        "sandbox_receipt_self_hash"
                    ),
                    (
                        "internal_factory_qualification_"
                        "sandbox_hash_invalid"
                    ),
                )
                != safe_rows[device][
                    "sandbox_receipt_self_hash"
                ]
                for device, (
                    path,
                    qualification_receipt,
                ) in qualification_receipts.items()
            )
            or internal_qualification.get(
                "outer_systemd_contract"
            )
            != qualification_outer_contract
            or internal_qualification.get(
                "outer_systemd_contract_self_hash"
            )
            != _content_hash(qualification_outer_contract)
            or internal_qualification.get(
                "outer_systemd_common_contract_self_hash"
            )
            != outer_service["common_contract_self_hash"]
            or internal_qualification.get(
                "outer_systemd_private_tmp_tradeoff"
            )
            != _OUTER_PRIVATE_TMP_TRADEOFF
            or qualification_writable_root == self.root
            or _require_hash(
                internal_qualification.get(
                    "outer_systemd_attestation_self_hash"
                ),
                (
                    "internal_factory_qualification_outer_"
                    "hash_invalid"
                ),
            )
            != internal_qualification[
                "outer_systemd_attestation_self_hash"
            ]
            or _require_hash(
                internal_qualification.get(
                    "outer_systemd_stable_binding_sha256"
                ),
                (
                    "internal_factory_qualification_outer_"
                    "hash_invalid"
                ),
            )
            != internal_qualification[
                "outer_systemd_stable_binding_sha256"
            ]
            or not isinstance(
                internal_qualification.get(
                    "outer_systemd_unit_id"
                ),
                str,
            )
            or not internal_qualification[
                "outer_systemd_unit_id"
            ].endswith(".service")
            or not isinstance(
                internal_qualification.get(
                    "outer_systemd_invocation_id"
                ),
                str,
            )
            or re.fullmatch(
                r"[0-9a-f]{32}",
                internal_qualification[
                    "outer_systemd_invocation_id"
                ],
            )
            is None
            or isinstance(
                internal_qualification.get("common_item_count"),
                bool,
            )
            or not isinstance(
                internal_qualification.get("common_item_count"),
                int,
            )
            or internal_qualification["common_item_count"] <= 0
            or internal_qualification.get(
                "official_source_content_supplied_to_model"
            )
            is not False
            or internal_qualification.get(
                "public_synthetic_content_supplied_to_model"
            )
            is not True
            or internal_qualification.get(
                "official_source_access_count"
            )
            != 0
            or internal_qualification.get("label_open_count") != 0
            or internal_qualification.get(
                "online_or_api_evaluation_count"
            )
            != 0
            or internal_qualification.get(
                "formal_measurement_authorized"
            )
            is not False
            or internal_qualification.get("formal_root_used")
            is not False
            or internal_qualification.get("formal_result")
            is not False
            or internal_qualification.get("efficacy_evidence")
            is not False
            or internal_qualification.get("effect_gate_added")
            is not False
            or internal_qualification.get("item_content_emitted")
            is not False
        ):
            raise FormalSupervisorError(
                "internal_factory_qualification_receipt_invalid"
            )
        _validate_factory_encoder_binding(
            internal_qualification.get(
                "factory_encoder_binding"
            ),
            expected_target_file_sha256=hashlib.sha256(
                minilm_target_raw
            ).hexdigest(),
            expected_target_self_sha256=minilm_target[
                "self_sha256"
            ],
        )
        for field in (
            "factory_execution_receipt_self_hash",
            "factory_output_file_sha256",
            "factory_output_self_hash",
            "factory_receipt_self_hash",
            "four_arm_barrier_self_hash",
            "qualification_action_self_hash",
            "qualification_runtime_closure_self_hash",
            "qwen_parallel_runtime_receipt_self_hash",
            "synthetic_source_receipt_self_hash",
            "synthetic_source_sha256",
        ):
            _require_hash(
                internal_qualification.get(field),
                "internal_factory_qualification_hash_invalid",
            )
        receipt: dict[str, Any] = {
            "schema": ACTION_SCHEMA,
            "status": "SUPERVISOR_INTERNAL_FACTORY_ACTION_FROZEN",
            "execution_mode": "supervisor_internal_factory",
            "supervisor_version": VERSION,
            "lineage": self.lineage,
            "root_identity": list(self.store.identity),
            "source_sha256": protocol.OFFICIAL_DATASET_SHA256,
            "runtime_closure_self_hash": closure.manifest["self_hash"],
            "outer_systemd_attestation": dict(outer_service),
            "outer_systemd_contract_self_hash": outer_service[
                "contract_self_hash"
            ],
            "outer_systemd_common_contract_self_hash": outer_service[
                "common_contract_self_hash"
            ],
            "outer_systemd_writable_root": outer_service[
                "writable_root"
            ],
            "outer_systemd_private_tmp_tradeoff": outer_service[
                "private_tmp_tradeoff"
            ],
            "freeze_commitments": dict(sorted(freeze_commitments.items())),
            "internal_implementation_sha256s": dict(
                sorted(implementation_hashes.items())
            ),
            "support_module_sha256s": support_hashes,
            "qwen_model_root": str(model_roots[0]),
            "qwen_model_manifest": str(manifests[0]),
            "qwen_model_manifest_sha256": _hash_regular_absolute(
                manifests[0]
            ),
            "qwen_actual_canary_lineage_terminal": {
                **dict(canary_lineage_binding),
                "path": str(canary_lineage_path),
            },
            "qwen_cuda_visible_devices": list(
                QWEN_CUDA_VISIBLE_DEVICES
            ),
            "qwen_gpu_device_allowlist": [
                str(path)
                for path in sorted(
                    {
                        path
                        for device in QWEN_CUDA_VISIBLE_DEVICES
                        for path in _gpu_device_candidates(int(device))
                    },
                    key=str,
                )
            ],
            "system_service_device_allow_required": True,
            "qwen_runtime_qualification_receipts": {
                device: {
                    "file_sha256": closure_configs[str(path)],
                    "path": str(path),
                    "self_sha256": qualification_receipt[
                        "self_sha256"
                    ],
                }
                for device, (
                    path,
                    qualification_receipt,
                ) in qualification_receipts.items()
            },
            "qwen_runtime_bindings": dict(
                sorted(qwen_runtime_bindings.items())
            ),
            "internal_factory_qualification_receipt": {
                "file_sha256": closure_configs[
                    str(internal_qualification_path)
                ],
                "path": str(internal_qualification_path),
                "self_sha256": internal_qualification_claimed,
            },
            "minilm_qualification_encoder_binding": (
                internal_qualification["factory_encoder_binding"]
            ),
            "minilm_model_root": str(model_roots[1]),
            "minilm_asset_manifest": str(manifests[1]),
            "minilm_asset_manifest_sha256": _hash_regular_absolute(
                manifests[1]
            ),
            "minilm_target_manifest": str(manifests[2]),
            "minilm_target_manifest_file_sha256": hashlib.sha256(
                minilm_target_raw
            ).hexdigest(),
            "minilm_target_manifest_self_sha256": minilm_target[
                "self_sha256"
            ],
            "arm_commands_accepted": False,
            "caller_scorer_accepted": False,
            "caller_predictions_accepted": False,
            "caller_prepared_results_accepted": False,
            "legacy_freeze_ready_receipt_is_authority": False,
            "formal_authority": (
                "closure_bound_supervisor_internal_item_factory_only"
            ),
            "effect_gate_added": False,
        }
        receipt["self_hash"] = _content_hash(receipt)
        self.store.write_json_exclusive("state/action.freeze.json", receipt)
        return FrozenAction(
            receipt=receipt,
            root_identity=self.store.identity,
            closure=closure,
            _token=_TOKEN,
        )

    def freeze_internal_factory_qualification_action_once(
        self,
        *,
        closure: RuntimeClosure,
        freeze_commitments: Mapping[str, str],
        qwen_model_root: Path,
        qwen_model_manifest: Path,
        qwen_actual_canary_lineage_terminal: Path,
        minilm_model_root: Path,
        minilm_asset_manifest: Path,
        minilm_target_manifest: Path,
        source_sha256: str,
    ) -> FrozenAction:
        """Freeze one source-free actual-assets execution of the formal core.

        This action can exist only in an explicitly temporary qualification
        lineage.  It accepts no arm command, runtime receipt, prediction,
        scorer, or prepared result, and its outputs can never cross into the
        fixed formal root.  Its purpose is to execute the exact Qwen→MiniLM
        internal core before the official ARN source is opened.
        """

        if self.lineage != "synthetic_source_free_qualification":
            raise FormalSupervisorError(
                "internal_factory_qualification_root_required"
            )
        outer_service = _attest_current_outer_systemd_service(
            writable_root=self.root
        )
        _require_hash(
            source_sha256, "qualification_source_hash_invalid"
        )
        _revalidate_closure(closure)
        if set(freeze_commitments) == set() or any(
            _require_hash(value, "freeze_commitment_invalid") != value
            for value in freeze_commitments.values()
        ):
            raise FormalSupervisorError("freeze_commitments_empty")
        implementation_hashes: dict[str, str] = {}
        for component, path in _INTERNAL_FORMAL_IMPLEMENTATION_PATHS.items():
            expected = closure.file_hashes.get(str(path))
            actual = _hash_regular_absolute(path)
            if expected != actual:
                raise FormalSupervisorError(
                    "internal_factory_implementation_not_in_closure"
                )
            implementation_hashes[component] = actual
        if any(
            implementation_hashes.get(component) != expected
            for component, expected in _STABLE_EXTRACTOR_SHA256S.items()
        ) or any(
            implementation_hashes.get(component) != expected
            for component, expected in _STABLE_MINILM_SHA256S.items()
        ):
            raise FormalSupervisorError(
                "stable_internal_runtime_closure_hash_drifted"
            )
        support_hashes = _validate_stable_support_closure(closure)
        attested_tests = closure.manifest["test_attestation"][
            "test_file_sha256s"
        ]
        if any(
            attested_tests.get(
                str(_INTERNAL_QUALIFICATION_TEST_PATHS[component])
            )
            != expected
            for component, expected in (
                _STABLE_QUALIFICATION_TEST_SHA256S.items()
            )
        ):
            raise FormalSupervisorError(
                "stable_extractor_tests_not_executed"
            )
        _validate_test_runner_closure(
            closure.manifest["test_attestation"].get(
                "test_runner"
            ),
            require_frozen=True,
        )
        model_roots = (
            _safe_absolute_path(qwen_model_root, allow_file=False),
            _safe_absolute_path(minilm_model_root, allow_file=False),
        )
        manifests = (
            _safe_absolute_path(qwen_model_manifest, allow_file=True),
            _safe_absolute_path(minilm_asset_manifest, allow_file=True),
            _safe_absolute_path(minilm_target_manifest, allow_file=True),
        )
        canary_lineage_path = _safe_absolute_path(
            qwen_actual_canary_lineage_terminal,
            allow_file=True,
        )
        canary_lineage_raw = _read_regular_absolute_exact(
            canary_lineage_path,
            expected_sha256=(
                CLOSED_CHOICE_ACTUAL_CANARY_LINEAGE_FILE_SHA256
            ),
            maximum_bytes=2 * 1024 * 1024,
        )
        canary_lineage_binding = (
            _validate_closed_choice_actual_canary_lineage(
                canary_lineage_raw,
                expected_model_manifest_sha256=(
                    _hash_regular_absolute(manifests[0])
                ),
            )
        )
        minilm_target_raw = _read_regular_absolute_exact(
            manifests[2],
            expected_sha256=_hash_regular_absolute(manifests[2]),
            maximum_bytes=2 * 1024 * 1024,
        )
        minilm_target = _validate_minilm_target_manifest_bytes(
            minilm_target_raw
        )
        closure_asset_roots = set(closure.manifest["asset_roots"])
        closure_configs = closure.manifest["config_sha256s"]
        if (
            not set(map(str, model_roots)).issubset(
                closure_asset_roots
            )
            or any(
                closure_configs.get(str(path))
                != _hash_regular_absolute(path)
                for path in manifests
            )
            or closure_configs.get(str(canary_lineage_path))
            != CLOSED_CHOICE_ACTUAL_CANARY_LINEAGE_FILE_SHA256
        ):
            raise FormalSupervisorError(
                "internal_factory_assets_not_in_closure"
            )
        receipt: dict[str, Any] = {
            "schema": ACTION_SCHEMA,
            "status": (
                "SUPERVISOR_INTERNAL_FACTORY_QUALIFICATION_ACTION_FROZEN"
            ),
            "execution_mode": (
                "supervisor_internal_factory_qualification"
            ),
            "supervisor_version": VERSION,
            "lineage": self.lineage,
            "root_identity": list(self.store.identity),
            "source_sha256": source_sha256,
            "runtime_closure_self_hash": closure.manifest["self_hash"],
            "outer_systemd_attestation": dict(outer_service),
            "outer_systemd_contract_self_hash": outer_service[
                "contract_self_hash"
            ],
            "outer_systemd_common_contract_self_hash": outer_service[
                "common_contract_self_hash"
            ],
            "outer_systemd_writable_root": outer_service[
                "writable_root"
            ],
            "outer_systemd_private_tmp_tradeoff": outer_service[
                "private_tmp_tradeoff"
            ],
            "freeze_commitments": dict(
                sorted(freeze_commitments.items())
            ),
            "internal_implementation_sha256s": dict(
                sorted(implementation_hashes.items())
            ),
            "support_module_sha256s": support_hashes,
            "qwen_model_root": str(model_roots[0]),
            "qwen_model_manifest": str(manifests[0]),
            "qwen_model_manifest_sha256": _hash_regular_absolute(
                manifests[0]
            ),
            "qwen_actual_canary_lineage_terminal": {
                **dict(canary_lineage_binding),
                "path": str(canary_lineage_path),
            },
            "qwen_cuda_visible_devices": list(
                QWEN_CUDA_VISIBLE_DEVICES
            ),
            "qwen_gpu_device_allowlist": [
                str(path)
                for path in sorted(
                    {
                        path
                        for device in QWEN_CUDA_VISIBLE_DEVICES
                        for path in _gpu_device_candidates(int(device))
                    },
                    key=str,
                )
            ],
            "minilm_model_root": str(model_roots[1]),
            "minilm_asset_manifest": str(manifests[1]),
            "minilm_asset_manifest_sha256": _hash_regular_absolute(
                manifests[1]
            ),
            "minilm_target_manifest": str(manifests[2]),
            "minilm_target_manifest_file_sha256": hashlib.sha256(
                minilm_target_raw
            ).hexdigest(),
            "minilm_target_manifest_self_sha256": minilm_target[
                "self_sha256"
            ],
            "arm_commands_accepted": False,
            "caller_runtime_receipts_accepted": False,
            "caller_predictions_accepted": False,
            "caller_scorer_accepted": False,
            "caller_prepared_results_accepted": False,
            "legacy_freeze_ready_receipt_is_authority": False,
            "formal_measurement_authorized": False,
            "official_source_access_authorized": False,
            "qualification_receipts_produced_by_execution": True,
            "effect_gate_added": False,
        }
        receipt["self_hash"] = _content_hash(receipt)
        self.store.write_json_exclusive(
            "state/action.freeze.json", receipt
        )
        return FrozenAction(
            receipt=receipt,
            root_identity=self.store.identity,
            closure=closure,
            _token=_TOKEN,
        )

    def _validate_action(self, action: FrozenAction) -> None:
        if (
            not isinstance(action, FrozenAction)
            or action._token is not _TOKEN
            or action.root_identity != self.store.identity
        ):
            raise FormalSupervisorError("frozen_action_not_supervisor_bound")
        _revalidate_closure(action.closure)
        on_disk = self.store.read_json("state/action.freeze.json")
        if on_disk != action.receipt:
            raise FormalSupervisorError("frozen_action_changed")
        body = dict(on_disk)
        claimed = body.pop("self_hash", None)
        if (
            on_disk.get("schema") != ACTION_SCHEMA
            or on_disk.get("legacy_freeze_ready_receipt_is_authority")
            is not False
            or not isinstance(claimed, str)
            or _content_hash(body) != claimed
        ):
            raise FormalSupervisorError("frozen_action_invalid")
        execution_mode = on_disk.get("execution_mode")
        if execution_mode in {
            "supervisor_internal_factory",
            "supervisor_internal_factory_qualification",
        }:
            if (
                on_disk.get("arm_commands_accepted") is not False
                or on_disk.get("caller_scorer_accepted") is not False
                or on_disk.get("caller_predictions_accepted") is not False
            ):
                raise FormalSupervisorError(
                    "internal_factory_action_invalid"
                )
            if execution_mode == "supervisor_internal_factory":
                if (
                    self.lineage != "formal_fixed_root"
                    or on_disk.get("source_sha256")
                    != protocol.OFFICIAL_DATASET_SHA256
                ):
                    raise FormalSupervisorError(
                        "internal_factory_action_invalid"
                    )
            elif (
                self.lineage
                != "synthetic_source_free_qualification"
                or on_disk.get("formal_measurement_authorized")
                is not False
                or on_disk.get("official_source_access_authorized")
                is not False
                or on_disk.get("caller_runtime_receipts_accepted")
                is not False
            ):
                raise FormalSupervisorError(
                    "internal_factory_qualification_action_invalid"
                )
            stored_outer = on_disk.get(
                "outer_systemd_attestation"
            )
            if not isinstance(stored_outer, dict):
                raise FormalSupervisorError(
                    "internal_factory_outer_systemd_missing"
                )
            _validate_outer_systemd_attestation(
                stored_outer,
                expected_writable_root=self.root,
            )
            current_outer = _attest_current_outer_systemd_service(
                writable_root=self.root
            )
            if (
                on_disk.get("outer_systemd_contract_self_hash")
                != stored_outer["contract_self_hash"]
                or on_disk.get(
                    "outer_systemd_common_contract_self_hash"
                )
                != _content_hash(OUTER_SYSTEMD_CONTRACT)
                or on_disk.get("outer_systemd_writable_root")
                != str(self.root)
                or on_disk.get(
                    "outer_systemd_private_tmp_tradeoff"
                )
                != _OUTER_PRIVATE_TMP_TRADEOFF
                or stored_outer["stable_binding_sha256"]
                != current_outer["stable_binding_sha256"]
            ):
                raise FormalSupervisorError(
                    "internal_factory_outer_systemd_changed"
                )
            for component, path in (
                _INTERNAL_FORMAL_IMPLEMENTATION_PATHS.items()
            ):
                if (
                    _hash_regular_absolute(path)
                    != on_disk["internal_implementation_sha256s"][
                        component
                    ]
                ):
                    raise FormalSupervisorError(
                        "internal_factory_implementation_changed"
                    )
            if (
                on_disk.get("support_module_sha256s")
                != _validate_stable_support_closure(
                    action.closure
                )
            ):
                raise FormalSupervisorError(
                    "internal_factory_support_modules_changed"
                )
            canary_lineage = on_disk.get(
                "qwen_actual_canary_lineage_terminal"
            )
            if (
                not isinstance(canary_lineage, dict)
                or set(canary_lineage)
                != {
                    "file_sha256",
                    "lineage_model_weight_load_count",
                    "path",
                    "repaired_actual_file_sha256",
                    "repaired_actual_self_sha256",
                    "self_sha256",
                    "successful_teacher_forced_qualification_run_count",
                    "worker_sha256",
                }
                or not isinstance(
                    canary_lineage.get("path"), str
                )
                or not Path(canary_lineage["path"]).is_absolute()
            ):
                raise FormalSupervisorError(
                    "closed_choice_actual_canary_lineage_changed"
                )
            canary_raw = _read_regular_absolute_exact(
                Path(canary_lineage["path"]),
                expected_sha256=(
                    CLOSED_CHOICE_ACTUAL_CANARY_LINEAGE_FILE_SHA256
                ),
                maximum_bytes=2 * 1024 * 1024,
            )
            observed_canary = (
                _validate_closed_choice_actual_canary_lineage(
                    canary_raw,
                    expected_model_manifest_sha256=on_disk[
                        "qwen_model_manifest_sha256"
                    ],
                )
            )
            if canary_lineage != {
                **dict(observed_canary),
                "path": canary_lineage["path"],
            }:
                raise FormalSupervisorError(
                    "closed_choice_actual_canary_lineage_changed"
                )
            for path_key, hash_key in (
                ("qwen_model_manifest", "qwen_model_manifest_sha256"),
                (
                    "minilm_asset_manifest",
                    "minilm_asset_manifest_sha256",
                ),
                (
                    "minilm_target_manifest",
                    "minilm_target_manifest_file_sha256",
                ),
            ):
                if _hash_regular_absolute(Path(on_disk[path_key])) != on_disk[
                    hash_key
                ]:
                    raise FormalSupervisorError(
                        "internal_factory_manifest_changed"
                    )
            target_raw = _read_regular_absolute_exact(
                Path(on_disk["minilm_target_manifest"]),
                expected_sha256=on_disk[
                    "minilm_target_manifest_file_sha256"
                ],
                maximum_bytes=2 * 1024 * 1024,
            )
            target_manifest = (
                _validate_minilm_target_manifest_bytes(target_raw)
            )
            if (
                target_manifest["self_sha256"]
                != on_disk["minilm_target_manifest_self_sha256"]
            ):
                raise FormalSupervisorError(
                    "internal_factory_manifest_changed"
                )
            if execution_mode == "supervisor_internal_factory":
                _validate_factory_encoder_binding(
                    on_disk.get(
                        "minilm_qualification_encoder_binding"
                    ),
                    expected_target_file_sha256=on_disk[
                        "minilm_target_manifest_file_sha256"
                    ],
                    expected_target_self_sha256=on_disk[
                        "minilm_target_manifest_self_sha256"
                    ],
                )
        else:
            for arm in on_disk["arm_commands"].values():
                if (
                    _hash_regular_absolute(Path(arm["implementation_path"]))
                    != arm["implementation_sha256"]
                    or _hash_regular_absolute(
                        Path(arm["command_executable_path"])
                    )
                    != arm["command_executable_sha256"]
                ):
                    raise FormalSupervisorError(
                        "frozen_arm_command_changed"
                    )
            scorer = on_disk["scorer_command"]
            if (
                _hash_regular_absolute(
                    Path(scorer["implementation_path"])
                )
                != scorer["implementation_sha256"]
                or _hash_regular_absolute(
                    Path(scorer["command_executable_path"])
                )
                != scorer["command_executable_sha256"]
            ):
                raise FormalSupervisorError(
                    "frozen_scorer_command_changed"
                )

    def begin_once(self, action: FrozenAction) -> FormalInvocation:
        self._validate_action(action)
        key_body = {
            "source_sha256": action.receipt["source_sha256"],
            "action_self_hash": action.receipt["self_hash"],
            "freeze_commitments": action.receipt["freeze_commitments"],
        }
        one_shot_key = _content_hash(key_body)
        receipt: dict[str, Any] = {
            "schema": ATTEMPT_SCHEMA,
            "status": "FORMAL_ATTEMPT_CREATED_ONCE",
            "lineage": self.lineage,
            "root_identity": list(self.store.identity),
            "one_shot_key": one_shot_key,
            "source_sha256": action.receipt["source_sha256"],
            "action_self_hash": action.receipt["self_hash"],
            "freeze_commitments": action.receipt["freeze_commitments"],
            "item_content_emitted": False,
        }
        receipt["self_hash"] = _content_hash(receipt)
        self.store.write_json_exclusive(
            f"state/attempts/{one_shot_key}.json", receipt
        )
        secret = os.urandom(32)
        self.store.write_exclusive(
            f"private/attempts/{one_shot_key}.linkage_secret", secret
        )
        return FormalInvocation(
            receipt=receipt,
            action=action,
            root_identity=self.store.identity,
            _token=_TOKEN,
        )

    def _validate_invocation(self, invocation: FormalInvocation) -> None:
        if (
            not isinstance(invocation, FormalInvocation)
            or invocation._token is not _TOKEN
            or invocation.root_identity != self.store.identity
        ):
            raise FormalSupervisorError("formal_invocation_not_bound")
        self._validate_action(invocation.action)
        key = invocation.receipt.get("one_shot_key")
        _require_hash(key, "one_shot_key_invalid")
        on_disk = self.store.read_json(f"state/attempts/{key}.json")
        if on_disk != invocation.receipt:
            raise FormalSupervisorError("formal_invocation_changed")
        body = dict(on_disk)
        claimed = body.pop("self_hash", None)
        if (
            on_disk.get("schema") != ATTEMPT_SCHEMA
            or on_disk.get("action_self_hash")
            != invocation.action.receipt["self_hash"]
            or not isinstance(claimed, str)
            or _content_hash(body) != claimed
        ):
            raise FormalSupervisorError("formal_invocation_invalid")

    def _read_source_same_descriptor(
        self,
        relative: str,
        *,
        expected_size: int,
        expected_sha256: str,
    ) -> bytes:
        raw = self.store.read_bytes(
            relative, maximum_bytes=expected_size
        )
        if (
            len(raw) != expected_size
            or hashlib.sha256(raw).hexdigest() != expected_sha256
        ):
            raise FormalSupervisorError("source_identity_drifted")
        return raw

    def materialize_official_packs_once(
        self, invocation: FormalInvocation
    ) -> Mapping[str, Any]:
        """Verify and parse the official CSV from the exact same byte read.

        This method is intentionally not used by source-free qualification
        tests.  It contains no pathname-based adapter re-open.
        """

        self._validate_invocation(invocation)
        if self.lineage != "formal_fixed_root":
            raise FormalSupervisorError(
                "official_source_forbidden_in_qualification"
            )
        raw = self._read_source_same_descriptor(
            FORMAL_SOURCE_DATASET_RELATIVE,
            expected_size=protocol.OFFICIAL_DATASET_SIZE,
            expected_sha256=protocol.OFFICIAL_DATASET_SHA256,
        )
        metadata = self._read_source_same_descriptor(
            FORMAL_SOURCE_METADATA_RELATIVE,
            expected_size=protocol.OFFICIAL_METADATA_SIZE,
            expected_sha256=protocol.OFFICIAL_METADATA_SHA256,
        )
        # `raw` is the exact verified descriptor read.  The adapter receives
        # these bytes directly; no second path open is possible here.
        rows = raw_adapter.parse_arn_csv_bytes(
            raw, expected_topology=raw_adapter.OFFICIAL_TOPOLOGY
        )
        key = invocation.receipt["one_shot_key"]
        secret = self.store.read_bytes(
            f"private/attempts/{key}.linkage_secret", maximum_bytes=32
        )
        source_receipt: dict[str, Any] = {
            "schema": SOURCE_SCHEMA,
            "status": "SAME_DESCRIPTOR_BYTES_VERIFIED_AND_ADAPTED",
            "one_shot_key": key,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
            "adapter_file_sha256": _hash_regular_absolute(
                Path(raw_adapter.__file__).resolve()
            ),
            "adapted_output_commitment": _content_hash(
                [row.__dict__ for row in rows]
            ),
            "adapted_row_count": len(rows),
            "path_reopened_by_adapter": False,
            "item_content_emitted": False,
        }
        source_receipt["self_hash"] = _content_hash(source_receipt)
        bundle = protocol._build_private_packs(  # noqa: SLF001
            rows,
            source_sha256=protocol.OFFICIAL_DATASET_SHA256,
            linkage_secret=secret,
            lineage="official_arn_measurement",
            schemas=(
                protocol.OFFICIAL_PREDICTOR_PACK_SCHEMA,
                protocol.OFFICIAL_LINKAGE_PACK_SCHEMA,
                protocol.OFFICIAL_LABEL_PACK_SCHEMA,
            ),
            source_verification_self_hash=source_receipt["self_hash"],
            adapter_qualification_self_hash=invocation.action.closure.manifest[
                "self_hash"
            ],
            quarantine_source_id=protocol.IMPLEMENTATION_EXPOSURE_SOURCE_ID,
        )
        return self._seal_private_bundle(invocation, bundle, source_receipt)

    def materialize_synthetic_packs_once(
        self,
        invocation: FormalInvocation,
        *,
        raw: bytes,
        expected_topology: raw_adapter.ArnTopology,
    ) -> Mapping[str, Any]:
        """Source-free qualification path; it can never claim formal lineage."""

        self._validate_invocation(invocation)
        if self.lineage != "synthetic_source_free_qualification":
            raise FormalSupervisorError(
                "synthetic_materialization_forbidden_in_formal"
            )
        source_hash = hashlib.sha256(raw).hexdigest()
        if source_hash != invocation.action.receipt["source_sha256"]:
            raise FormalSupervisorError("synthetic_source_hash_drifted")
        rows = raw_adapter.parse_arn_csv_bytes(
            raw, expected_topology=expected_topology
        )
        key = invocation.receipt["one_shot_key"]
        secret = self.store.read_bytes(
            f"private/attempts/{key}.linkage_secret", maximum_bytes=32
        )
        source_receipt: dict[str, Any] = {
            "schema": SOURCE_SCHEMA,
            "status": "SYNTHETIC_SAME_BYTES_ADAPTED",
            "one_shot_key": key,
            "source_sha256": source_hash,
            "metadata_sha256": None,
            "adapter_file_sha256": _hash_regular_absolute(
                Path(raw_adapter.__file__).resolve()
            ),
            "adapted_output_commitment": _content_hash(
                [row.__dict__ for row in rows]
            ),
            "adapted_row_count": len(rows),
            "path_reopened_by_adapter": False,
            "item_content_emitted": False,
        }
        source_receipt["self_hash"] = _content_hash(source_receipt)
        bundle = protocol._build_private_packs_from_adapted_fixtures(  # noqa: SLF001
            rows,
            source_sha256=source_hash,
            linkage_secret=secret,
        )
        return self._seal_private_bundle(invocation, bundle, source_receipt)

    def _seal_private_bundle(
        self,
        invocation: FormalInvocation,
        bundle: protocol.PrivatePackBundle,
        source_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        key = invocation.receipt["one_shot_key"]
        predictor_raw = _canonical_bytes(dict(bundle.predictor_pack))
        linkage_raw = _canonical_bytes(dict(bundle.linkage_pack))
        label_raw = _canonical_bytes(dict(bundle.label_pack))
        self.store.write_exclusive(
            f"private/attempts/{key}.predictor.json", predictor_raw
        )
        self.store.write_exclusive(
            f"private/custodian/{key}.linkage.json", linkage_raw
        )
        self.store.write_exclusive(
            f"private/custodian/{key}.labels.json", label_raw
        )
        if (
            _json_content_hash(
                predictor_raw, issue_id="predictor_pack_invalid"
            )
            != bundle.pack_commitments["predictor_pack_sha256"]
            or _json_content_hash(
                linkage_raw, issue_id="linkage_pack_invalid"
            )
            != bundle.pack_commitments["linkage_pack_sha256"]
            or _json_content_hash(
                label_raw, issue_id="label_pack_invalid"
            )
            != bundle.pack_commitments["label_pack_sha256"]
        ):
            raise FormalSupervisorError("private_pack_commitment_drifted")
        receipt: dict[str, Any] = {
            "schema": PACK_BARRIER_SCHEMA,
            "status": "PRIVATE_PACKS_SEALED",
            "one_shot_key": key,
            "action_self_hash": invocation.action.receipt["self_hash"],
            "source_receipt_self_hash": source_receipt["self_hash"],
            "pack_commitments": dict(bundle.pack_commitments),
            "safe_split_aggregates": dict(bundle.safe_split_aggregates),
            "label_opened_by_arm": False,
            "linkage_opened_by_arm": False,
            "item_content_emitted": False,
        }
        receipt["self_hash"] = _content_hash(receipt)
        self.store.write_json_exclusive(
            f"state/attempts/{key}.packs.json", receipt
        )
        self.store.write_json_exclusive(
            f"state/attempts/{key}.source.json", dict(source_receipt)
        )
        return receipt

    def _run_internal_sandbox_command(
        self,
        invocation: FormalInvocation,
        *,
        stage_id: str,
        command: Sequence[str],
        implementation_sha256: str,
        code_roots: Sequence[Path],
        model_roots: Sequence[Path],
        work_relative: str,
        output_relative: str,
        input_binding_sha256: str,
        gpu_device_index: int | None = None,
        environment_overrides: Mapping[str, str] | None = None,
    ) -> tuple[bytes, Mapping[str, Any]]:
        if (
            _SAFE_COMPONENT.fullmatch(stage_id) is None
            or not command
            or any(
                not isinstance(value, str)
                or not value
                or "\x00" in value
                for value in command
            )
        ):
            raise FormalSupervisorError("internal_stage_invalid")
        _require_hash(
            implementation_sha256,
            "internal_stage_implementation_hash_invalid",
        )
        _require_hash(
            input_binding_sha256,
            "internal_stage_input_hash_invalid",
        )
        overrides = _validate_internal_environment_overrides(
            environment_overrides or {}
        )
        if (
            gpu_device_index is not None
            and (
                isinstance(gpu_device_index, bool)
                or not isinstance(gpu_device_index, int)
                or not 0 <= gpu_device_index <= 15
                or overrides.get("CUDA_VISIBLE_DEVICES")
                != str(gpu_device_index)
            )
        ):
            raise FormalSupervisorError(
                "internal_stage_gpu_binding_invalid"
            )
        if (
            gpu_device_index is None
            and overrides.get("CUDA_VISIBLE_DEVICES") not in {None, ""}
        ):
            raise FormalSupervisorError(
                "internal_stage_gpu_binding_invalid"
            )
        work_path = self.root / work_relative
        output_path = self.root / output_relative
        receipt_relative = f"{work_relative}/{stage_id}.sandbox.safe.json"
        receipt_path = self.root / receipt_relative
        key = invocation.receipt["one_shot_key"]
        probes = {
            "label_pack": str(
                self.root / f"private/custodian/{key}.labels.json"
            ),
            "linkage_pack": str(
                self.root / f"private/custodian/{key}.linkage.json"
            ),
        }
        spec: dict[str, Any] = {
            "schema": SANDBOX_SPEC_SCHEMA,
            "arm_id": stage_id,
            "one_shot_key": key,
            "action_self_hash": invocation.action.receipt["self_hash"],
            "command": list(command),
            "implementation_sha256": implementation_sha256,
            "code_roots": [str(path) for path in code_roots],
            "model_roots": [str(path) for path in model_roots],
            "work_root": str(work_path),
            "predictor_pack_sha256": input_binding_sha256,
            "prediction_output_path": str(output_path),
            "sandbox_receipt_path": str(receipt_path),
            "private_denial_probes": probes,
            "gpu_device_index": gpu_device_index,
            "environment_overrides": overrides,
        }
        spec["self_hash"] = _content_hash(spec)
        spec_relative = f"{work_relative}/{stage_id}.sandbox.spec.json"
        self.store.write_json_exclusive(spec_relative, spec)
        module_file = Path(__file__).resolve()
        completed = subprocess.run(
            [
                str(Path(sys.executable).resolve()),
                str(module_file),
                "--sandbox-child",
                str(self.root / spec_relative),
            ],
            cwd=work_path,
            env={
                "HOME": str(work_path),
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "PYTHONPATH": _LOCAL_PYTHONPATH,
                "TEMP": str(work_path),
                "TMP": str(work_path),
                "TMPDIR": str(work_path),
                **OFFLINE_ENVIRONMENT,
            },
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=3_600,
        )
        if completed.returncode != 0:
            raise FormalSupervisorError("internal_stage_subprocess_failed")
        receipt = self.store.read_json(receipt_relative)
        receipt_body = dict(receipt)
        claimed = receipt_body.pop("self_hash", None)
        if (
            receipt.get("schema") != SANDBOX_RECEIPT_SCHEMA
            or receipt.get("sandbox_spec_self_hash") != spec["self_hash"]
            or receipt.get("arm_exit_code") != 0
            or receipt.get("implementation_sha256")
            != implementation_sha256
            or receipt.get("label_denial_errno")
            not in {errno.EACCES, errno.EPERM}
            or receipt.get("linkage_denial_errno")
            not in {errno.EACCES, errno.EPERM}
            or receipt.get("gpu_device_index") != gpu_device_index
            or receipt.get("environment_overrides") != overrides
            or not isinstance(claimed, str)
            or _content_hash(receipt_body) != claimed
        ):
            raise FormalSupervisorError(
                "internal_stage_sandbox_receipt_invalid"
            )
        device_rows = receipt.get("gpu_device_rows")
        if gpu_device_index is None:
            if device_rows != []:
                raise FormalSupervisorError(
                    "internal_stage_gpu_receipt_invalid"
                )
        elif (
            not isinstance(device_rows, list)
            or str(Path(f"/dev/nvidia{gpu_device_index}"))
            not in {
                row.get("path")
                for row in device_rows
                if isinstance(row, dict)
            }
        ):
            raise FormalSupervisorError(
                "internal_stage_gpu_receipt_invalid"
            )
        raw = self.store.read_bytes(output_relative)
        if hashlib.sha256(raw).hexdigest() != receipt.get(
            "prediction_output_sha256"
        ):
            raise FormalSupervisorError("internal_stage_output_changed")
        return raw, receipt

    def run_arm_once(
        self, invocation: FormalInvocation, *, arm_id: str
    ) -> Mapping[str, Any]:
        """Execute one frozen arm; predictions cannot be supplied by caller."""

        self._validate_invocation(invocation)
        if self.lineage != "synthetic_source_free_qualification":
            raise FormalSupervisorError(
                "formal_external_arm_output_forbidden"
            )
        if arm_id not in protocol.ARM_IDS:
            raise FormalSupervisorError("arm_id_invalid")
        key = invocation.receipt["one_shot_key"]
        pack_receipt = self.store.read_json(
            f"state/attempts/{key}.packs.json"
        )
        predictor_raw = self.store.read_bytes(
            f"private/attempts/{key}.predictor.json"
        )
        if _json_content_hash(
            predictor_raw, issue_id="predictor_pack_invalid"
        ) != pack_receipt[
            "pack_commitments"
        ]["predictor_pack_sha256"]:
            raise FormalSupervisorError("predictor_pack_changed")
        arm_binding = invocation.action.receipt["arm_commands"][arm_id]
        work_relative = f"work/arms/{key}.{arm_id}"
        work_path = self.store.ensure_directory(work_relative)
        input_relative = f"{work_relative}/predictor.json"
        output_relative = f"{work_relative}/raw_predictions.json"
        sandbox_receipt_relative = f"{work_relative}/sandbox.safe.json"
        self.store.write_exclusive(input_relative, predictor_raw)
        input_path = self.root / input_relative
        output_path = self.root / output_relative
        sandbox_receipt_path = self.root / sandbox_receipt_relative
        command = [
            value.format(
                input=str(input_path),
                output=str(output_path),
                arm_id=arm_id,
            )
            for value in arm_binding["command_template"]
        ]
        private_label = (
            self.root / f"private/custodian/{key}.labels.json"
        )
        private_linkage = (
            self.root / f"private/custodian/{key}.linkage.json"
        )
        effective_code_roots = list(
            dict.fromkeys(
                (
                    *arm_binding["code_roots"],
                    *invocation.action.closure.manifest.get(
                        "support_roots", ()
                    ),
                    *invocation.action.closure.manifest[
                        "runtime_roots"
                    ],
                    *map(str, _LOCAL_PYTHONPATH_ROOTS),
                )
            )
        )
        spec: dict[str, Any] = {
            "schema": SANDBOX_SPEC_SCHEMA,
            "arm_id": arm_id,
            "one_shot_key": key,
            "action_self_hash": invocation.action.receipt["self_hash"],
            "command": command,
            "implementation_sha256": arm_binding[
                "implementation_sha256"
            ],
            "code_roots": effective_code_roots,
            "model_roots": arm_binding["model_roots"],
            "work_root": str(work_path),
            "predictor_pack_sha256": _json_content_hash(
                predictor_raw, issue_id="predictor_pack_invalid"
            ),
            "prediction_output_path": str(output_path),
            "sandbox_receipt_path": str(sandbox_receipt_path),
            "private_denial_probes": {
                "label_pack": str(private_label),
                "linkage_pack": str(private_linkage),
            },
            "gpu_device_index": None,
            "environment_overrides": {
                "CUDA_VISIBLE_DEVICES": "",
            },
        }
        spec["self_hash"] = _content_hash(spec)
        spec_relative = f"{work_relative}/sandbox.spec.json"
        self.store.write_json_exclusive(spec_relative, spec)
        module_file = Path(__file__).resolve()
        environment = {
            "HOME": str(work_path),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": _LOCAL_PYTHONPATH,
            "TEMP": str(work_path),
            "TMP": str(work_path),
            "TMPDIR": str(work_path),
            **OFFLINE_ENVIRONMENT,
        }
        completed = subprocess.run(
            [
                str(Path(sys.executable).resolve()),
                str(module_file),
                "--sandbox-child",
                str(self.root / spec_relative),
            ],
            cwd=work_path,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=3_600,
        )
        if completed.returncode != 0:
            raise FormalSupervisorError("arm_subprocess_failed")
        sandbox_receipt = self.store.read_json(sandbox_receipt_relative)
        if (
            sandbox_receipt.get("schema") != SANDBOX_RECEIPT_SCHEMA
            or sandbox_receipt.get("sandbox_spec_self_hash")
            != spec["self_hash"]
            or sandbox_receipt.get("arm_exit_code") != 0
            or sandbox_receipt.get("label_denial_errno")
            not in {errno.EACCES, errno.EPERM}
            or sandbox_receipt.get("linkage_denial_errno")
            not in {errno.EACCES, errno.EPERM}
        ):
            raise FormalSupervisorError("sandbox_receipt_invalid")
        output_raw = self.store.read_bytes(output_relative)
        if hashlib.sha256(output_raw).hexdigest() != sandbox_receipt.get(
            "prediction_output_sha256"
        ):
            raise FormalSupervisorError("arm_output_changed")
        predictions = _validate_raw_arm_output(
            output_raw,
            expected_item_ids=_predictor_item_ids(predictor_raw),
        )
        pack: dict[str, Any] = {
            "schema": PREDICTION_PACK_SCHEMA,
            "arm_id": arm_id,
            "one_shot_key": key,
            "action_self_hash": invocation.action.receipt["self_hash"],
            "implementation_sha256": arm_binding[
                "implementation_sha256"
            ],
            "runtime_closure_self_hash": invocation.action.closure.manifest[
                "self_hash"
            ],
            "predictor_pack_sha256": _json_content_hash(
                predictor_raw, issue_id="predictor_pack_invalid"
            ),
            "sandbox_spec_self_hash": spec["self_hash"],
            "sandbox_receipt_self_hash": sandbox_receipt["self_hash"],
            "predictions": predictions,
            "item_content_emitted": False,
        }
        pack["self_hash"] = _content_hash(pack)
        self.store.write_json_exclusive(
            f"state/predictions/{key}.{arm_id}.json", pack
        )
        return {
            "arm_id": arm_id,
            "prediction_pack_self_hash": pack["self_hash"],
            "sandbox_receipt_self_hash": sandbox_receipt["self_hash"],
            "item_count": len(predictions),
            "item_content_emitted": False,
        }

    def run_internal_factory_once(
        self, invocation: FormalInvocation
    ) -> Mapping[str, Any]:
        """Run the exact Qwen→internal-factory bridge and seal all four arms.

        This formal method accepts no item result, prediction, callable,
        commitment, prepared mapping, or score table from its caller.
        """

        self._validate_invocation(invocation)
        action = invocation.action.receipt
        execution_mode = action.get("execution_mode")
        if (
            execution_mode == "supervisor_internal_factory"
            and self.lineage == "formal_fixed_root"
        ):
            qwen_lineage = "formal_measurement"
            expected_qwen_bindings = action["qwen_runtime_bindings"]
        elif (
            execution_mode
            == "supervisor_internal_factory_qualification"
            and self.lineage
            == "synthetic_source_free_qualification"
            and action.get("formal_measurement_authorized") is False
            and action.get("official_source_access_authorized") is False
        ):
            qwen_lineage = "source_free_qualification"
            expected_qwen_bindings = None
        else:
            raise FormalSupervisorError(
                "internal_factory_formal_action_required"
            )
        if action.get("caller_predictions_accepted") is not False:
            raise FormalSupervisorError(
                "internal_factory_formal_action_required"
            )
        from assumption_agent.benchmarks import (  # local trusted import
            gscl_arn_formal_item_factory_v1 as item_factory,
        )
        from replication_runtime.gscl_narrative_extractor_v1 import (
            contract as extractor_contract,
        )
        from replication_runtime.gscl_narrative_extractor_v1 import (
            closed_choice_multi_pack_worker as extractor_multi,
        )

        key = invocation.receipt["one_shot_key"]
        pack_receipt = self.store.read_json(
            f"state/attempts/{key}.packs.json"
        )
        predictor_relative = f"private/attempts/{key}.predictor.json"
        predictor_raw = self.store.read_bytes(predictor_relative)
        if _json_content_hash(
            predictor_raw, issue_id="predictor_pack_invalid"
        ) != pack_receipt["pack_commitments"]["predictor_pack_sha256"]:
            raise FormalSupervisorError("predictor_pack_changed")
        predictor_rows = item_factory._predictor_rows(predictor_raw)  # noqa: SLF001
        expected_stories = item_factory._expected_stories(  # noqa: SLF001
            predictor_rows
        )
        if not expected_stories:
            raise FormalSupervisorError("formal_story_set_empty")

        base_relative = f"work/formal_factory/{key}"
        self.store.ensure_directory(base_relative)
        qwen_manifest_raw = _read_regular_absolute_exact(
            Path(action["qwen_model_manifest"]),
            expected_sha256=action["qwen_model_manifest_sha256"],
            maximum_bytes=4 * 1024 * 1024,
        )
        code_roots = tuple(
            Path(value)
            for value in (
                *invocation.action.closure.manifest["code_roots"],
                *invocation.action.closure.manifest["support_roots"],
                *invocation.action.closure.manifest["runtime_roots"],
                *map(str, _LOCAL_PYTHONPATH_ROOTS),
            )
        )
        multi_worker_hash = action[
            "internal_implementation_sha256s"
        ]["extractor_multi_pack_worker"]
        batch_records: list[dict[str, Any]] = []
        shard_batches: dict[int, list[dict[str, Any]]] = {
            index: []
            for index in range(
                len(action["qwen_cuda_visible_devices"])
            )
        }
        # Assign complete item triplets round-robin before packing.  This
        # keeps both GPUs balanced even for the two-item public qualification
        # fixture while retaining the fixed <=63-story pack bound.
        batch_plan = _balanced_triplet_batch_plan(
            expected_stories,
            shard_count=len(shard_batches),
            maximum_story_count=63,
        )
        for sequence, planned in enumerate(batch_plan):
            selected = planned["stories"]
            shard_index = planned["shard_index"]
            item_indices = planned["item_indices"]
            batch_relative = (
                f"{base_relative}/extractor_shard_{shard_index}/"
                f"batch_{sequence:04d}"
            )
            self.store.ensure_directory(batch_relative)
            requests = tuple(
                extractor_contract.StoryRequest(
                    ordinal=ordinal, story_text=story
                )
                for ordinal, (_, _, story) in enumerate(selected)
            )
            input_raw = extractor_contract.encode_input(
                batch_id=f"arn-{sequence:04d}",
                sequence=sequence,
                requests=requests,
            )
            input_relative = f"{batch_relative}/input.json"
            output_relative = f"{batch_relative}/output.json"
            input_sha = self.store.write_exclusive(
                input_relative, input_raw
            )
            record = {
                "sequence": sequence,
                "input_relative": input_relative,
                "input_file_sha256": input_sha,
                "item_indices": list(item_indices),
                "output_relative": output_relative,
                "shard_index": shard_index,
            }
            batch_records.append(record)
            shard_batches[shard_index].append(record)

        active_shards = [
            index for index, rows in shard_batches.items() if rows
        ]
        if len(active_shards) != len(
            action["qwen_cuda_visible_devices"]
        ):
            raise FormalSupervisorError(
                "formal_story_set_too_small_for_frozen_parallelism"
            )
        shard_specs: list[dict[str, Any]] = []
        for shard_index in active_shards:
            shard_relative = (
                f"{base_relative}/extractor_shard_{shard_index}"
            )
            manifest_relative = (
                f"{shard_relative}/qwen.model.json"
            )
            input_manifest_relative = (
                f"{shard_relative}/multi_pack.input.json"
            )
            safe_receipt_relative = (
                f"{shard_relative}/runtime.safe.json"
            )
            self.store.write_exclusive(
                manifest_relative, qwen_manifest_raw
            )
            manifest_body = {
                "batches": [
                    {
                        "input_file_sha256": row[
                            "input_file_sha256"
                        ],
                        "input_path": str(
                            self.root / row["input_relative"]
                        ),
                        "output_path": str(
                            self.root / row["output_relative"]
                        ),
                        "sequence": row["sequence"],
                    }
                    for row in shard_batches[shard_index]
                ],
                "schema": extractor_multi.INPUT_MANIFEST_SCHEMA,
                "lineage": qwen_lineage,
                "work_root": str(self.root / shard_relative),
            }
            multi_manifest = {
                **manifest_body,
                "self_sha256": _content_hash(manifest_body),
            }
            input_manifest_sha = self.store.write_json_exclusive(
                input_manifest_relative, multi_manifest
            )
            cuda_device = action["qwen_cuda_visible_devices"][
                shard_index
            ]
            command = (
                str(Path(sys.executable).resolve()),
                "-m",
                (
                    "replication_runtime."
                    "gscl_narrative_extractor_v1."
                    "closed_choice_multi_pack_worker"
                ),
                "--input-manifest",
                str(self.root / input_manifest_relative),
                "--model",
                action["qwen_model_root"],
                "--model-manifest",
                str(self.root / manifest_relative),
                "--safe-receipt",
                str(self.root / safe_receipt_relative),
            )
            shard_specs.append(
                {
                    "cuda_device": cuda_device,
                    "input_binding_sha256": input_manifest_sha,
                    "output_relative": safe_receipt_relative,
                    "shard_index": shard_index,
                    "stage_id": f"extractor_shard_{shard_index}",
                    "command": command,
                    "work_relative": shard_relative,
                }
            )

        def run_shard(
            shard: Mapping[str, Any],
        ) -> tuple[int, bytes, Mapping[str, Any]]:
            raw, sandbox = self._run_internal_sandbox_command(
                invocation,
                stage_id=shard["stage_id"],
                command=shard["command"],
                implementation_sha256=multi_worker_hash,
                code_roots=code_roots,
                model_roots=(Path(action["qwen_model_root"]),),
                work_relative=shard["work_relative"],
                output_relative=shard["output_relative"],
                input_binding_sha256=shard[
                    "input_binding_sha256"
                ],
                gpu_device_index=int(shard["cuda_device"]),
                environment_overrides={
                    "CUDA_VISIBLE_DEVICES": shard["cuda_device"],
                },
            )
            return shard["shard_index"], raw, sandbox

        shard_results: dict[
            int, tuple[bytes, Mapping[str, Any]]
        ] = {}
        with ThreadPoolExecutor(
            max_workers=len(shard_specs),
            thread_name_prefix="gscl-qwen",
        ) as executor:
            futures = [
                executor.submit(run_shard, shard)
                for shard in shard_specs
            ]
            for future in futures:
                shard_index, raw, sandbox = future.result()
                shard_results[shard_index] = (raw, sandbox)
        observed_qwen_bindings: dict[
            str, Mapping[str, Any]
        ] = {}
        qwen_safe_receipt_rows: dict[str, Mapping[str, Any]] = {}
        shard_specs_by_index = {
            row["shard_index"]: row for row in shard_specs
        }
        for shard_index, (safe_raw, sandbox_receipt) in sorted(
            shard_results.items()
        ):
            safe_receipt = _parse_json(
                safe_raw,
                issue_id="qwen_formal_runtime_receipt_invalid",
            )
            runtime_binding = _validate_qwen_runtime_safe_receipt(
                safe_receipt,
                expected_model_manifest_sha256=action[
                    "qwen_model_manifest_sha256"
                ],
                expected_visible_device=action[
                    "qwen_cuda_visible_devices"
                ][shard_index],
                expected_lineage=qwen_lineage,
            )
            safe_batch_rows = _validated_qwen_batch_rows(
                safe_receipt
            )
            shard_spec = shard_specs_by_index[shard_index]
            expected_shard_sequences = {
                row["sequence"]
                for row in shard_batches[shard_index]
            }
            if (
                safe_receipt.get("input_manifest_file_sha256")
                != shard_spec["input_binding_sha256"]
                or set(safe_batch_rows)
                != expected_shard_sequences
            ):
                raise FormalSupervisorError(
                    "qwen_runtime_batch_binding_invalid"
                )
            visible_device = action["qwen_cuda_visible_devices"][
                shard_index
            ]
            observed_qwen_bindings[visible_device] = runtime_binding
            qwen_safe_receipt_rows[visible_device] = {
                "path": str(
                    self.root
                    / shard_specs_by_index[shard_index][
                        "output_relative"
                    ]
                ),
                "file_sha256": hashlib.sha256(safe_raw).hexdigest(),
                "self_sha256": safe_receipt["self_sha256"],
                "sandbox_receipt_self_hash": sandbox_receipt[
                    "self_hash"
                ],
            }
            if (
                expected_qwen_bindings is not None
                and runtime_binding
                != expected_qwen_bindings[visible_device]
            ):
                raise FormalSupervisorError(
                    "qwen_runtime_changed_after_qualification"
                )
            for row in shard_batches[shard_index]:
                output_raw = self.store.read_bytes(
                    row["output_relative"]
                )
                pack = (
                    extractor_contract.load_trusted_story_only_input_pack(
                        self.root / row["input_relative"]
                    )
                )
                decoded_output = extractor_contract.decode_private_output(
                    output_raw, expected_pack=pack
                )
                output_file_sha256 = hashlib.sha256(
                    output_raw
                ).hexdigest()
                safe_batch = safe_batch_rows[row["sequence"]]
                results = decoded_output["results"]
                valid_token_counts = [
                    result["completion_token_count"]
                    for result in results
                    if result["generation_valid"] is True
                ]
                actual_batch_binding = {
                    "batch_id": pack.batch_id,
                    "decision_invalid_count": sum(
                        result["generation_valid"] is False
                        for result in results
                    ),
                    "decision_valid_count": sum(
                        result["generation_valid"] is True
                        for result in results
                    ),
                    "input_file_sha256": pack.input_file_sha256,
                    "input_pack_commitment": (
                        pack.input_pack_commitment
                    ),
                    "output_file_sha256": output_file_sha256,
                    "selection_receipt_count": sum(
                        result["generation_valid"] is True
                        for result in results
                    ),
                    "sequence": pack.sequence,
                    "story_count": len(pack.requests),
                    "valid_wire_completion_token_count_maximum": max(
                        valid_token_counts, default=0
                    ),
                    "valid_wire_completion_token_count_sum": sum(
                        valid_token_counts
                    ),
                }
                if (
                    decoded_output.get("execution_closure")
                    != safe_receipt["execution_closure"]
                    or any(
                        safe_batch.get(field) != value
                        for field, value in (
                            actual_batch_binding.items()
                        )
                    )
                    or pack.input_file_sha256
                    != row["input_file_sha256"]
                ):
                    raise FormalSupervisorError(
                        "qwen_runtime_batch_binding_invalid"
                    )
                row["batch_id"] = pack.batch_id
                row["input_pack_commitment"] = (
                    pack.input_pack_commitment
                )
                row["story_count"] = len(pack.requests)
                row["execution_closure_commitment"] = (
                    _content_hash(
                        decoded_output["execution_closure"]
                    )
                )
                row["output_file_sha256"] = output_file_sha256
                row["sandbox_receipt_self_hash"] = (
                    sandbox_receipt["self_hash"]
                )
                row["qwen_runtime_receipt_self_hash"] = (
                    safe_receipt["self_sha256"]
                )
        batch_records.sort(key=lambda row: row["sequence"])
        common_observed_bindings = {
            _content_hash(
                {
                    field: value
                    for field, value in binding.items()
                    if field != "logical_gpu_binding"
                }
            )
            for binding in observed_qwen_bindings.values()
        }
        logical_gpu_bindings = [
            binding["logical_gpu_binding"]
            for binding in observed_qwen_bindings.values()
        ]
        if (
            set(observed_qwen_bindings)
            != set(action["qwen_cuda_visible_devices"])
            or len(common_observed_bindings) != 1
            or len(
                {
                    (
                        binding["logical_device_name"],
                        tuple(
                            binding["logical_compute_capability"]
                        ),
                    )
                    for binding in logical_gpu_bindings
                }
            )
            != 1
            or len(
                {
                    binding["logical_device_uuid"]
                    for binding in logical_gpu_bindings
                }
            )
            != len(logical_gpu_bindings)
        ):
            raise FormalSupervisorError(
                "qwen_cross_gpu_runtime_binding_mismatch"
            )
        qwen_runtime_receipt: dict[str, Any] = {
            "schema": f"{VERSION}.qwen_parallel_runtime_receipt.v1",
            "status": "QWEN_PARALLEL_RUNTIME_BOUND",
            "execution_mode": execution_mode,
            "one_shot_key": key,
            "action_self_hash": action["self_hash"],
            "qwen_runtime_bindings": dict(
                sorted(observed_qwen_bindings.items())
            ),
            "private_safe_receipts": dict(
                sorted(qwen_safe_receipt_rows.items())
            ),
            "single_model_load_per_worker": True,
            "worker_count": len(shard_results),
            "source_lineage": qwen_lineage,
            "closed_choice_selection_count": sum(
                receipt["selection_receipt_count"]
                for receipt in (
                    _parse_json(
                        raw,
                        issue_id=(
                            "qwen_formal_runtime_receipt_invalid"
                        ),
                    )
                    for raw, _ in shard_results.values()
                )
            ),
            "free_form_generation_count": 0,
            "score_operation": (
                "teacher_forced_forward_log_softmax"
            ),
            "official_source_accessed_during_qualification": False,
            "item_content_emitted": False,
        }
        qwen_runtime_receipt["self_hash"] = _content_hash(
            qwen_runtime_receipt
        )
        self.store.write_json_exclusive(
            f"state/attempts/{key}.qwen_runtime.safe.json",
            qwen_runtime_receipt,
        )

        factory_relative = f"{base_relative}/item_factory"
        self.store.ensure_directory(factory_relative)
        predictor_copy_relative = f"{factory_relative}/predictor.json"
        minilm_manifest_relative = (
            f"{factory_relative}/minilm.asset.json"
        )
        minilm_target_manifest_relative = (
            f"{factory_relative}/minilm.target.json"
        )
        factory_output_relative = (
            f"{factory_relative}/private_four_arm.json"
        )
        batch_manifest_relative = (
            f"{factory_relative}/extractor_batches.json"
        )
        predictor_file_sha = self.store.write_exclusive(
            predictor_copy_relative, predictor_raw
        )
        minilm_manifest_raw = _read_regular_absolute_exact(
            Path(action["minilm_asset_manifest"]),
            expected_sha256=action["minilm_asset_manifest_sha256"],
            maximum_bytes=4 * 1024 * 1024,
        )
        self.store.write_exclusive(
            minilm_manifest_relative, minilm_manifest_raw
        )
        minilm_target_manifest_raw = _read_regular_absolute_exact(
            Path(action["minilm_target_manifest"]),
            expected_sha256=action[
                "minilm_target_manifest_file_sha256"
            ],
            maximum_bytes=2 * 1024 * 1024,
        )
        _validate_minilm_target_manifest_bytes(
            minilm_target_manifest_raw
        )
        self.store.write_exclusive(
            minilm_target_manifest_relative,
            minilm_target_manifest_raw,
        )
        factory_batches: list[dict[str, Any]] = []
        for record in batch_records:
            sequence = record["sequence"]
            copied_input_relative = (
                f"{factory_relative}/batch_{sequence:04d}.input.json"
            )
            copied_output_relative = (
                f"{factory_relative}/batch_{sequence:04d}.output.json"
            )
            copied_input = self.store.read_bytes(
                record["input_relative"]
            )
            copied_output = self.store.read_bytes(
                record["output_relative"]
            )
            copied_input_sha = self.store.write_exclusive(
                copied_input_relative, copied_input
            )
            copied_output_sha = self.store.write_exclusive(
                copied_output_relative, copied_output
            )
            if (
                copied_input_sha != record["input_file_sha256"]
                or copied_output_sha != record["output_file_sha256"]
            ):
                raise FormalSupervisorError(
                    "factory_batch_copy_changed"
                )
            factory_batches.append(
                {
                    "sequence": sequence,
                    "item_indices": record["item_indices"],
                    "input_path": str(
                        self.root / copied_input_relative
                    ),
                    "input_file_sha256": copied_input_sha,
                    "output_path": str(
                        self.root / copied_output_relative
                    ),
                    "output_file_sha256": copied_output_sha,
                }
            )
        batch_manifest = {
            "schema": item_factory.BATCH_MANIFEST_SCHEMA,
            "predictor_pack_file_sha256": predictor_file_sha,
            "batches": factory_batches,
        }
        self.store.write_json_exclusive(
            batch_manifest_relative, batch_manifest
        )
        factory_hash = action["internal_implementation_sha256s"][
            "item_factory"
        ]
        factory_command = (
            str(Path(sys.executable).resolve()),
            "-m",
            "assumption_agent.benchmarks.gscl_arn_formal_item_factory_v1",
            "--predictor",
            str(self.root / predictor_copy_relative),
            "--batch-manifest",
            str(self.root / batch_manifest_relative),
            "--minilm-manifest",
            action["minilm_asset_manifest"],
            "--minilm-model",
            action["minilm_model_root"],
            "--minilm-target-manifest",
            str(self.root / minilm_target_manifest_relative),
            "--output",
            str(self.root / factory_output_relative),
        )
        factory_raw, factory_sandbox = (
            self._run_internal_sandbox_command(
                invocation,
                stage_id="item_factory",
                command=factory_command,
                implementation_sha256=factory_hash,
                code_roots=code_roots,
                model_roots=(
                    Path(action["minilm_model_root"]),
                    Path(action["minilm_asset_manifest"]),
                ),
                work_relative=factory_relative,
                output_relative=factory_output_relative,
                input_binding_sha256=hashlib.sha256(
                    _canonical_bytes(
                        {
                            "batch_manifest": batch_manifest,
                            "minilm_asset_manifest_sha256": action[
                                "minilm_asset_manifest_sha256"
                            ],
                            "minilm_target_manifest_file_sha256": (
                                action[
                                    "minilm_target_manifest_file_sha256"
                                ]
                            ),
                            "minilm_target_manifest_self_sha256": (
                                action[
                                    "minilm_target_manifest_self_sha256"
                                ]
                            ),
                        }
                    )
                ).hexdigest(),
                environment_overrides={
                    "CUDA_VISIBLE_DEVICES": "",
                },
            )
        )
        factory_output = item_factory._decode_canonical_object(  # noqa: SLF001
            factory_raw, issue_id="formal_factory_output_invalid"
        )
        expected_ids = {
            row["opaque_item_id"] for row in predictor_rows
        }
        expected_factory_batch_receipts = [
            {
                "batch_id": record["batch_id"],
                "execution_closure_commitment": record[
                    "execution_closure_commitment"
                ],
                "input_file_sha256": record[
                    "input_file_sha256"
                ],
                "input_pack_commitment": record[
                    "input_pack_commitment"
                ],
                "output_file_sha256": record[
                    "output_file_sha256"
                ],
                "sequence": record["sequence"],
                "story_count": record["story_count"],
            }
            for record in batch_records
        ]
        normalized_by_arm = _validate_factory_output_receipt(
            factory_output,
            expected_schema=item_factory.PRIVATE_OUTPUT_SCHEMA,
            expected_lineage="formal_frozen_assets",
            expected_predictor_file_sha256=predictor_file_sha,
            expected_batch_receipts=(
                expected_factory_batch_receipts
            ),
            expected_item_ids=expected_ids,
        )
        encoder_binding = _validate_factory_encoder_binding(
            factory_output.get("encoder_binding"),
            expected_target_file_sha256=action[
                "minilm_target_manifest_file_sha256"
            ],
            expected_target_self_sha256=action[
                "minilm_target_manifest_self_sha256"
            ],
        )
        if (
            execution_mode == "supervisor_internal_factory"
            and not _factory_encoder_bindings_content_equivalent(
                action["minilm_qualification_encoder_binding"],
                encoder_binding,
                expected_observed_target_manifest_path=(
                    self.root / minilm_target_manifest_relative
                ),
                expected_target_file_sha256=action[
                    "minilm_target_manifest_file_sha256"
                ],
                expected_target_self_sha256=action[
                    "minilm_target_manifest_self_sha256"
                ],
            )
        ):
            raise FormalSupervisorError(
                "minilm_runtime_changed_after_qualification"
            )
        factory_claimed = factory_output["self_hash"]
        sealed_hashes: dict[str, str] = {}
        for arm_id in protocol.ARM_IDS:
            normalized = normalized_by_arm[arm_id]
            pack: dict[str, Any] = {
                "schema": PREDICTION_PACK_SCHEMA,
                "arm_id": arm_id,
                "one_shot_key": key,
                "action_self_hash": action["self_hash"],
                "implementation_sha256": factory_hash,
                "runtime_closure_self_hash": (
                    invocation.action.closure.manifest["self_hash"]
                ),
                "predictor_pack_sha256": (
                    pack_receipt["pack_commitments"][
                        "predictor_pack_sha256"
                    ]
                ),
                "sandbox_spec_self_hash": (
                    factory_sandbox["sandbox_spec_self_hash"]
                ),
                "sandbox_receipt_self_hash": factory_sandbox["self_hash"],
                "factory_output_self_hash": factory_claimed,
                "execution_mode": execution_mode,
                "source_lineage": qwen_lineage,
                "predictions": normalized,
                "item_content_emitted": False,
            }
            pack["self_hash"] = _content_hash(pack)
            self.store.write_json_exclusive(
                f"state/predictions/{key}.{arm_id}.json", pack
            )
            sealed_hashes[arm_id] = pack["self_hash"]
        execution_body: dict[str, Any] = {
            "schema": INTERNAL_FACTORY_EXECUTION_SCHEMA,
            "status": "INTERNAL_FACTORY_ALL_FOUR_ARMS_SEALED",
            "one_shot_key": key,
            "action_self_hash": action["self_hash"],
            "prediction_pack_self_hashes": dict(
                sorted(sealed_hashes.items())
            ),
            "item_count": len(expected_ids),
            "error_item_count": factory_output["error_item_count"],
            "extractor_batch_count": len(batch_records),
            "factory_output_self_hash": factory_claimed,
            "factory_output_file_sha256": hashlib.sha256(
                factory_raw
            ).hexdigest(),
            "factory_receipt_self_hash": factory_output[
                "factory_receipt_self_hash"
            ],
            "factory_encoder_binding": dict(encoder_binding),
            "factory_sandbox_receipt_self_hash": factory_sandbox[
                "self_hash"
            ],
            "qwen_parallel_runtime_receipt_self_hash": (
                qwen_runtime_receipt["self_hash"]
            ),
            "qwen_private_safe_receipts": dict(
                sorted(qwen_safe_receipt_rows.items())
            ),
            "execution_mode": execution_mode,
            "source_lineage": qwen_lineage,
            "label_opened": False,
            "online_or_api_evaluator_used": False,
            "item_content_emitted": False,
        }
        execution_receipt = {
            **execution_body,
            "self_hash": _content_hash(execution_body),
        }
        self.store.write_json_exclusive(
            f"state/attempts/{key}.internal_factory_execution.safe.json",
            execution_receipt,
        )
        return execution_receipt

    def seal_four_arm_barrier_once(
        self, invocation: FormalInvocation
    ) -> Mapping[str, Any]:
        self._validate_invocation(invocation)
        key = invocation.receipt["one_shot_key"]
        pack_hashes: dict[str, str] = {}
        common_item_ids: set[str] | None = None
        for arm_id in protocol.ARM_IDS:
            pack = self.store.read_json(
                f"state/predictions/{key}.{arm_id}.json"
            )
            if (
                pack.get("schema") != PREDICTION_PACK_SCHEMA
                or pack.get("arm_id") != arm_id
                or pack.get("one_shot_key") != key
                or pack.get("action_self_hash")
                != invocation.action.receipt["self_hash"]
                or (
                    invocation.action.receipt.get("execution_mode")
                    in {
                        "supervisor_internal_factory",
                        (
                            "supervisor_internal_factory_"
                            "qualification"
                        ),
                    }
                    and (
                        pack.get("execution_mode")
                        != invocation.action.receipt.get(
                            "execution_mode"
                        )
                        or pack.get("source_lineage")
                        != (
                            "formal_measurement"
                            if invocation.action.receipt.get(
                                "execution_mode"
                            )
                            == "supervisor_internal_factory"
                            else "source_free_qualification"
                        )
                    )
                )
            ):
                raise FormalSupervisorError("prediction_pack_invalid")
            body = dict(pack)
            claimed = body.pop("self_hash", None)
            if not isinstance(claimed, str) or _content_hash(body) != claimed:
                raise FormalSupervisorError("prediction_pack_invalid")
            item_ids = {
                row["opaque_item_id"] for row in pack["predictions"]
            }
            if common_item_ids is None:
                common_item_ids = item_ids
            elif item_ids != common_item_ids:
                raise FormalSupervisorError("arm_item_sets_differ")
            pack_hashes[arm_id] = claimed
        barrier: dict[str, Any] = {
            "schema": FOUR_ARM_BARRIER_SCHEMA,
            "status": "ALL_FOUR_ARMS_SEALED_BEFORE_LABEL_OPEN",
            "one_shot_key": key,
            "action_self_hash": invocation.action.receipt["self_hash"],
            "prediction_pack_self_hashes": dict(sorted(pack_hashes.items())),
            "common_item_count": len(common_item_ids or set()),
            "execution_mode": invocation.action.receipt.get(
                "execution_mode"
            ),
            "label_opened": False,
            "item_content_emitted": False,
        }
        barrier["self_hash"] = _content_hash(barrier)
        self.store.write_json_exclusive(
            f"state/attempts/{key}.four_arm_barrier.json", barrier
        )
        if (
            invocation.action.receipt.get("execution_mode")
            == "supervisor_internal_factory_qualification"
        ):
            execution = self.store.read_json(
                f"state/attempts/{key}."
                "internal_factory_execution.safe.json"
            )
            execution_body = dict(execution)
            execution_claimed = execution_body.pop(
                "self_hash", None
            )
            qwen_runtime = self.store.read_json(
                f"state/attempts/{key}.qwen_runtime.safe.json"
            )
            qwen_body = dict(qwen_runtime)
            qwen_claimed = qwen_body.pop("self_hash", None)
            source_receipt = self.store.read_json(
                f"state/attempts/{key}.source.json"
            )
            source_body = dict(source_receipt)
            source_claimed = source_body.pop("self_hash", None)
            action = invocation.action.receipt
            outer = action["outer_systemd_attestation"]
            encoder_binding = _validate_factory_encoder_binding(
                execution.get("factory_encoder_binding"),
                expected_target_file_sha256=action[
                    "minilm_target_manifest_file_sha256"
                ],
                expected_target_self_sha256=action[
                    "minilm_target_manifest_self_sha256"
                ],
            )
            if (
                execution.get("schema")
                != INTERNAL_FACTORY_EXECUTION_SCHEMA
                or execution.get("execution_mode")
                != "supervisor_internal_factory_qualification"
                or execution.get("source_lineage")
                != "source_free_qualification"
                or execution.get("one_shot_key") != key
                or execution.get("action_self_hash")
                != action["self_hash"]
                or execution.get("prediction_pack_self_hashes")
                != barrier["prediction_pack_self_hashes"]
                or execution.get("item_count")
                != barrier["common_item_count"]
                or execution.get("item_count", 0) <= 0
                or execution.get("error_item_count") != 0
                or execution.get("label_opened") is not False
                or execution.get("online_or_api_evaluator_used")
                is not False
                or not isinstance(execution_claimed, str)
                or _content_hash(execution_body)
                != execution_claimed
                or qwen_runtime.get("execution_mode")
                != "supervisor_internal_factory_qualification"
                or qwen_runtime.get("source_lineage")
                != "source_free_qualification"
                or qwen_runtime.get(
                    "official_source_accessed_during_qualification"
                )
                is not False
                or qwen_runtime.get("free_form_generation_count")
                != 0
                or qwen_runtime.get("score_operation")
                != "teacher_forced_forward_log_softmax"
                or qwen_runtime.get(
                    "closed_choice_selection_count"
                )
                != 3 * execution["item_count"]
                or not isinstance(qwen_claimed, str)
                or _content_hash(qwen_body) != qwen_claimed
                or source_receipt.get("status")
                != "SYNTHETIC_SAME_BYTES_ADAPTED"
                or source_receipt.get("metadata_sha256") is not None
                or not isinstance(source_claimed, str)
                or _content_hash(source_body) != source_claimed
                or action.get("formal_measurement_authorized")
                is not False
                or action.get("official_source_access_authorized")
                is not False
            ):
                raise FormalSupervisorError(
                    "internal_factory_qualification_incomplete"
                )
            terminal_body: dict[str, Any] = {
                "schema": INTERNAL_FACTORY_QUALIFICATION_SCHEMA,
                "status": (
                    "PASS_SOURCE_FREE_EXACT_INTERNAL_FACTORY_"
                    "QUALIFICATION"
                ),
                "one_shot_key": key,
                "qualification_action_self_hash": action["self_hash"],
                "qualification_runtime_closure_self_hash": (
                    invocation.action.closure.manifest["self_hash"]
                ),
                "internal_implementation_sha256s": action[
                    "internal_implementation_sha256s"
                ],
                "qwen_model_manifest_sha256": action[
                    "qwen_model_manifest_sha256"
                ],
                "qwen_actual_canary_lineage_binding": {
                    key: value
                    for key, value in action[
                        "qwen_actual_canary_lineage_terminal"
                    ].items()
                    if key != "path"
                },
                "minilm_asset_manifest_sha256": action[
                    "minilm_asset_manifest_sha256"
                ],
                "minilm_target_manifest_file_sha256": action[
                    "minilm_target_manifest_file_sha256"
                ],
                "minilm_target_manifest_self_sha256": action[
                    "minilm_target_manifest_self_sha256"
                ],
                "outer_systemd_contract": dict(outer["contract"]),
                "outer_systemd_contract_self_hash": action[
                    "outer_systemd_contract_self_hash"
                ],
                "outer_systemd_common_contract_self_hash": action[
                    "outer_systemd_common_contract_self_hash"
                ],
                "outer_systemd_writable_root": action[
                    "outer_systemd_writable_root"
                ],
                "outer_systemd_private_tmp_tradeoff": action[
                    "outer_systemd_private_tmp_tradeoff"
                ],
                "outer_systemd_attestation_self_hash": outer[
                    "self_hash"
                ],
                "outer_systemd_stable_binding_sha256": outer[
                    "stable_binding_sha256"
                ],
                "outer_systemd_unit_id": outer["unit_id"],
                "outer_systemd_invocation_id": outer[
                    "invocation_id"
                ],
                "qwen_parallel_runtime_receipt_self_hash": (
                    qwen_claimed
                ),
                "qwen_runtime_bindings": qwen_runtime[
                    "qwen_runtime_bindings"
                ],
                "closed_choice_selection_count": qwen_runtime[
                    "closed_choice_selection_count"
                ],
                "free_form_generation_count": 0,
                "score_operation": (
                    "teacher_forced_forward_log_softmax"
                ),
                "qwen_private_safe_receipts": qwen_runtime[
                    "private_safe_receipts"
                ],
                "factory_execution_receipt_self_hash": (
                    execution_claimed
                ),
                "factory_output_self_hash": execution[
                    "factory_output_self_hash"
                ],
                "factory_output_file_sha256": execution[
                    "factory_output_file_sha256"
                ],
                "factory_receipt_self_hash": execution[
                    "factory_receipt_self_hash"
                ],
                "factory_encoder_binding": dict(encoder_binding),
                "four_arm_barrier_self_hash": barrier["self_hash"],
                "common_item_count": barrier["common_item_count"],
                "synthetic_source_receipt_self_hash": (
                    source_claimed
                ),
                "synthetic_source_sha256": source_receipt[
                    "source_sha256"
                ],
                "official_source_content_supplied_to_model": False,
                "public_synthetic_content_supplied_to_model": True,
                "official_source_access_count": 0,
                "label_open_count": 0,
                "online_or_api_evaluation_count": 0,
                "formal_measurement_authorized": False,
                "formal_root_used": False,
                "formal_result": False,
                "efficacy_evidence": False,
                "effect_gate_added": False,
                "item_content_emitted": False,
            }
            terminal = {
                **terminal_body,
                "self_hash": _content_hash(terminal_body),
            }
            self.store.write_json_exclusive(
                f"state/attempts/{key}."
                "internal_factory_qualification.safe.json",
                terminal,
            )
        return barrier

    def run_fixed_scorer_once(
        self, invocation: FormalInvocation
    ) -> Mapping[str, Any]:
        """Open labels only after the frozen four-arm barrier exists."""

        self._validate_invocation(invocation)
        key = invocation.receipt["one_shot_key"]
        barrier = self.store.read_json(
            f"state/attempts/{key}.four_arm_barrier.json"
        )
        if (
            barrier.get("schema") != FOUR_ARM_BARRIER_SCHEMA
            or barrier.get("status")
            != "ALL_FOUR_ARMS_SEALED_BEFORE_LABEL_OPEN"
            or barrier.get("label_opened") is not False
        ):
            raise FormalSupervisorError("four_arm_barrier_invalid")
        if (
            invocation.action.receipt.get("execution_mode")
            == "supervisor_internal_factory"
        ):
            return self._run_internal_fixed_scorer_once(
                invocation, barrier=barrier
            )
        if (
            invocation.action.receipt.get("execution_mode")
            == "supervisor_internal_factory_qualification"
        ):
            raise FormalSupervisorError(
                "qualification_labels_must_remain_unopened"
            )
        # Re-read and revalidate every sealed prediction before label access.
        prediction_paths: list[str] = []
        for arm_id in protocol.ARM_IDS:
            relative = f"state/predictions/{key}.{arm_id}.json"
            pack = self.store.read_json(relative)
            if (
                pack.get("self_hash")
                != barrier["prediction_pack_self_hashes"][arm_id]
            ):
                raise FormalSupervisorError(
                    "prediction_changed_before_scoring"
                )
            prediction_paths.append(str(self.root / relative))
        label_relative = f"private/custodian/{key}.labels.json"
        label_raw = self.store.read_bytes(label_relative)
        pack_receipt = self.store.read_json(
            f"state/attempts/{key}.packs.json"
        )
        if _json_content_hash(
            label_raw, issue_id="label_pack_invalid"
        ) != pack_receipt[
            "pack_commitments"
        ]["label_pack_sha256"]:
            raise FormalSupervisorError("label_pack_changed")
        result_relative = f"work/scorer/{key}.aggregate.json"
        command = [
            value.format(
                labels=str(self.root / label_relative),
                predictions=",".join(prediction_paths),
                output=str(self.root / result_relative),
            )
            for value in invocation.action.receipt["scorer_command"][
                "command_template"
            ]
        ]
        environment = {
            "HOME": str(self.root / "work/scorer"),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            **OFFLINE_ENVIRONMENT,
        }
        completed = subprocess.run(
            command,
            cwd=self.root / "work/scorer",
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=3_600,
        )
        if completed.returncode != 0:
            raise FormalSupervisorError("fixed_scorer_failed")
        aggregate_raw = self.store.read_bytes(result_relative)
        aggregate = _parse_json(
            aggregate_raw, issue_id="aggregate_result_invalid"
        )
        _validate_safe_aggregate(aggregate)
        receipt: dict[str, Any] = {
            "schema": SCORE_RECEIPT_SCHEMA,
            "status": "FIXED_OFFLINE_SCORER_COMPLETED",
            "one_shot_key": key,
            "action_self_hash": invocation.action.receipt["self_hash"],
            "four_arm_barrier_self_hash": barrier["self_hash"],
            "label_pack_sha256": _json_content_hash(
                label_raw, issue_id="label_pack_invalid"
            ),
            "aggregate_result_sha256": hashlib.sha256(
                aggregate_raw
            ).hexdigest(),
            "aggregate_result": dict(aggregate),
            "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
            "item_content_emitted": False,
        }
        receipt["self_hash"] = _content_hash(receipt)
        self.store.write_json_exclusive(
            f"state/attempts/{key}.score.safe.json", receipt
        )
        return receipt

    def _run_internal_fixed_scorer_once(
        self,
        invocation: FormalInvocation,
        *,
        barrier: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Open labels once and invoke the frozen in-process aggregate metric."""

        if self.lineage != "formal_fixed_root":
            raise FormalSupervisorError(
                "internal_scorer_requires_formal_root"
            )
        key = invocation.receipt["one_shot_key"]
        pack_receipt = self.store.read_json(
            f"state/attempts/{key}.packs.json"
        )
        linkage_raw = self.store.read_bytes(
            f"private/custodian/{key}.linkage.json"
        )
        if _json_content_hash(
            linkage_raw, issue_id="linkage_pack_invalid"
        ) != pack_receipt["pack_commitments"]["linkage_pack_sha256"]:
            raise FormalSupervisorError("linkage_pack_changed")
        linkage_pack = _parse_json(
            linkage_raw, issue_id="linkage_pack_invalid"
        )
        expected_item_ids = {
            row["opaque_item_id"]
            for row in linkage_pack.get("rows", [])
            if isinstance(row, dict)
            and row.get("measurement_eligible") is True
        }
        if (
            not expected_item_ids
            or len(expected_item_ids)
            != barrier["common_item_count"]
        ):
            raise FormalSupervisorError(
                "scorer_item_set_changed"
            )
        action = invocation.action.receipt
        protocol_contract_sha256 = action[
            "internal_implementation_sha256s"
        ]["protocol"]
        protocol_packs: dict[str, Mapping[str, Any]] = {}
        for arm_id in protocol.ARM_IDS:
            sealed = self.store.read_json(
                f"state/predictions/{key}.{arm_id}.json"
            )
            if (
                sealed.get("self_hash")
                != barrier["prediction_pack_self_hashes"][arm_id]
            ):
                raise FormalSupervisorError(
                    "prediction_changed_before_scoring"
                )
            body: dict[str, Any] = {
                "schema": protocol.PREDICTION_PACK_SCHEMA,
                "arm_id": arm_id,
                "arm_implementation_sha256": sealed[
                    "implementation_sha256"
                ],
                "arm_qualification_receipt_sha256": (
                    action["runtime_closure_self_hash"]
                ),
                "protocol_contract_sha256": (
                    protocol_contract_sha256
                ),
                "predictor_pack_sha256": pack_receipt[
                    "pack_commitments"
                ]["predictor_pack_sha256"],
                "linkage_pack_sha256": pack_receipt[
                    "pack_commitments"
                ]["linkage_pack_sha256"],
                "predictions": sealed["predictions"],
            }
            body["self_hash"] = protocol._content_hash(body)  # noqa: SLF001
            protocol._validate_prediction_pack(  # noqa: SLF001
                body,
                expected_arm_id=arm_id,
                expected_protocol_contract_sha256=(
                    protocol_contract_sha256
                ),
                expected_predictor_pack_sha256=pack_receipt[
                    "pack_commitments"
                ]["predictor_pack_sha256"],
                expected_linkage_pack_sha256=pack_receipt[
                    "pack_commitments"
                ]["linkage_pack_sha256"],
                expected_item_ids=expected_item_ids,
            )
            protocol_packs[arm_id] = body

        marker: dict[str, Any] = {
            "schema": f"{VERSION}.formal_label_open_claim.v1",
            "status": "CLAIMED_AFTER_FOUR_ARM_BARRIER",
            "one_shot_key": key,
            "action_self_hash": action["self_hash"],
            "four_arm_barrier_self_hash": barrier["self_hash"],
            "label_pack_sha256": pack_receipt["pack_commitments"][
                "label_pack_sha256"
            ],
            "retry_or_replay_allowed": False,
        }
        marker["self_hash"] = _content_hash(marker)
        self.store.write_json_exclusive(
            f"state/attempts/{key}.labels_open.claim.json",
            marker,
        )
        label_raw = self.store.read_bytes(
            f"private/custodian/{key}.labels.json"
        )
        if _json_content_hash(
            label_raw, issue_id="label_pack_invalid"
        ) != pack_receipt["pack_commitments"]["label_pack_sha256"]:
            raise FormalSupervisorError("label_pack_changed")
        label_pack = _parse_json(
            label_raw, issue_id="label_pack_invalid"
        )
        scoring = protocol._score_aggregates(  # noqa: SLF001
            action_seal={
                "label_pack_sha256": pack_receipt[
                    "pack_commitments"
                ]["label_pack_sha256"]
            },
            prediction_packs=protocol_packs,
            linkage_pack=linkage_pack,
            label_pack=label_pack,
        )
        body = {
            "schema": SCORE_RECEIPT_SCHEMA,
            "status": "FIXED_OFFLINE_SCORER_COMPLETED",
            "one_shot_key": key,
            "action_self_hash": action["self_hash"],
            "four_arm_barrier_self_hash": barrier["self_hash"],
            "label_open_claim_self_hash": marker["self_hash"],
            "arm_aggregates": scoring["arm_aggregates"],
            "paired_aggregate_differences": scoring[
                "paired_differences"
            ],
            "uncertainty_method": (
                "intercept_only_cluster_robust_sandwich_by_opaque_"
                "proverb_cluster"
            ),
            "abstain_and_error_counted_wrong": True,
            "online_or_api_evaluator_used": False,
            "effect_gate_added": False,
            "item_content_emitted": False,
        }
        body["self_hash"] = protocol._content_hash(body)  # noqa: SLF001
        self.store.write_exclusive(
            f"state/attempts/{key}.score.safe.json",
            protocol._canonical_bytes(body),  # noqa: SLF001
        )
        return body


def _predictor_item_ids(predictor_raw: bytes) -> set[str]:
    pack = _parse_json(predictor_raw, issue_id="predictor_pack_invalid")
    rows = pack.get("rows")
    if not isinstance(rows, list):
        raise FormalSupervisorError("predictor_pack_invalid")
    result: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise FormalSupervisorError("predictor_pack_invalid")
        opaque = row.get("opaque_item_id")
        if not isinstance(opaque, str) or _SHA256.fullmatch(opaque) is None:
            raise FormalSupervisorError("predictor_pack_invalid")
        if opaque in result:
            raise FormalSupervisorError("predictor_pack_duplicate_item")
        result.add(opaque)
    if not result:
        raise FormalSupervisorError("predictor_pack_empty")
    return result


def _validate_raw_arm_output(
    raw: bytes, *, expected_item_ids: set[str]
) -> list[dict[str, Any]]:
    value = _parse_json(raw, issue_id="raw_arm_output_invalid")
    if set(value) != {"predictions"} or not isinstance(
        value["predictions"], list
    ):
        raise FormalSupervisorError("raw_arm_output_invalid")
    normalized: list[dict[str, Any]] = []
    observed: set[str] = set()
    for row in value["predictions"]:
        if not isinstance(row, dict) or set(row) != {
            "opaque_item_id",
            "disposition",
            "selected_choice",
            "error_code",
        }:
            raise FormalSupervisorError("raw_arm_prediction_invalid")
        opaque = row["opaque_item_id"]
        disposition = row["disposition"]
        selected = row["selected_choice"]
        error_code = row["error_code"]
        if (
            not isinstance(opaque, str)
            or _SHA256.fullmatch(opaque) is None
            or opaque in observed
            or disposition not in protocol.DISPOSITIONS
        ):
            raise FormalSupervisorError("raw_arm_prediction_invalid")
        if disposition == "ANSWER":
            if selected not in protocol.CHOICE_IDS or error_code is not None:
                raise FormalSupervisorError("raw_arm_prediction_invalid")
        elif disposition == "ABSTAIN":
            if selected is not None or error_code is not None:
                raise FormalSupervisorError("raw_arm_prediction_invalid")
        elif selected is not None or error_code not in protocol.ERROR_CODES:
            raise FormalSupervisorError("raw_arm_prediction_invalid")
        observed.add(opaque)
        normalized.append(dict(row))
    if observed != expected_item_ids:
        raise FormalSupervisorError("raw_arm_item_set_invalid")
    return sorted(normalized, key=lambda row: row["opaque_item_id"])


def _validate_factory_output_receipt(
    output: Mapping[str, Any],
    *,
    expected_schema: str,
    expected_lineage: str,
    expected_predictor_file_sha256: str,
    expected_batch_receipts: Sequence[Mapping[str, Any]],
    expected_item_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    expected_fields = {
        "by_arm",
        "caller_commitments_accepted",
        "caller_predictions_accepted",
        "encoder_binding",
        "error_item_count",
        "extractor_batch_receipts",
        "factory_receipt_self_hash",
        "item_content_emitted",
        "item_count",
        "lineage",
        "predictor_pack_file_sha256",
        "private_item_recomputation_receipts",
        "schema",
        "self_hash",
        "status",
    }
    if not isinstance(output, dict) or set(output) != expected_fields:
        raise FormalSupervisorError(
            "formal_factory_output_fields_invalid"
        )
    body = dict(output)
    claimed = body.pop("self_hash", None)
    item_count = output.get("item_count")
    error_count = output.get("error_item_count")
    if (
        output.get("schema") != expected_schema
        or output.get("status")
        != "PRIVATE_ALL_FOUR_ITEM_RESULTS_RECOMPUTED"
        or output.get("lineage") != expected_lineage
        or output.get("caller_predictions_accepted") is not False
        or output.get("caller_commitments_accepted") is not False
        or output.get("item_content_emitted") is not False
        or output.get("predictor_pack_file_sha256")
        != expected_predictor_file_sha256
        or output.get("extractor_batch_receipts")
        != [dict(row) for row in expected_batch_receipts]
        or isinstance(item_count, bool)
        or not isinstance(item_count, int)
        or item_count != len(expected_item_ids)
        or isinstance(error_count, bool)
        or not isinstance(error_count, int)
        or not 0 <= error_count <= item_count
        or not isinstance(claimed, str)
        or _SHA256.fullmatch(claimed) is None
        or _content_hash(body) != claimed
    ):
        raise FormalSupervisorError(
            "formal_factory_output_invalid"
        )
    by_arm = output.get("by_arm")
    if not isinstance(by_arm, dict) or set(by_arm) != set(
        protocol.ARM_IDS
    ):
        raise FormalSupervisorError(
            "formal_factory_output_invalid"
        )
    normalized_by_arm: dict[str, list[dict[str, Any]]] = {}
    error_sets: list[set[str]] = []
    for arm_id in protocol.ARM_IDS:
        predictions = by_arm[arm_id]
        normalized = _validate_raw_arm_output(
            _canonical_bytes({"predictions": predictions}),
            expected_item_ids=expected_item_ids,
        )
        normalized_by_arm[arm_id] = normalized
        if any(
            row["disposition"] == "ERROR"
            and row["error_code"] != "ARM_RUNTIME_ERROR"
            for row in normalized
        ):
            raise FormalSupervisorError(
                "formal_factory_error_set_invalid"
            )
        error_sets.append(
            {
                row["opaque_item_id"]
                for row in normalized
                if row["disposition"] == "ERROR"
            }
        )
    if (
        not error_sets
        or any(current != error_sets[0] for current in error_sets[1:])
        or len(error_sets[0]) != error_count
    ):
        raise FormalSupervisorError(
            "formal_factory_error_set_invalid"
        )
    private_receipts = output.get(
        "private_item_recomputation_receipts"
    )
    if not isinstance(private_receipts, list):
        raise FormalSupervisorError(
            "formal_factory_recomputation_receipts_invalid"
        )
    recomputed_ids: set[str] = set()
    for row in private_receipts:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "opaque_item_id",
                "recomputation_receipt_self_hash",
            }
            or not isinstance(row.get("opaque_item_id"), str)
            or row["opaque_item_id"] not in expected_item_ids
            or row["opaque_item_id"] in recomputed_ids
        ):
            raise FormalSupervisorError(
                "formal_factory_recomputation_receipts_invalid"
            )
        _require_hash(
            row.get("recomputation_receipt_self_hash"),
            "formal_factory_recomputation_receipts_invalid",
        )
        recomputed_ids.add(row["opaque_item_id"])
    if recomputed_ids != expected_item_ids - error_sets[0]:
        raise FormalSupervisorError(
            "formal_factory_recomputation_receipts_invalid"
        )
    factory_receipt_self_hash = output.get(
        "factory_receipt_self_hash"
    )
    if error_count == item_count:
        if factory_receipt_self_hash is not None:
            raise FormalSupervisorError(
                "formal_factory_receipt_hash_invalid"
            )
    else:
        _require_hash(
            factory_receipt_self_hash,
            "formal_factory_receipt_hash_invalid",
        )
    return normalized_by_arm


def _validate_safe_numeric_tree(value: object, *, depth: int = 0) -> None:
    if depth > 5:
        raise FormalSupervisorError("aggregate_result_not_safe")
    if value is None or type(value) in {bool, int}:
        return
    if isinstance(value, list):
        if len(value) > 64:
            raise FormalSupervisorError("aggregate_result_not_safe")
        for child in value:
            _validate_safe_numeric_tree(child, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 256:
            raise FormalSupervisorError("aggregate_result_not_safe")
        for key, child in value.items():
            if (
                not isinstance(key, str)
                or _SAFE_COMPONENT.fullmatch(key) is None
                or _SHA256.fullmatch(key) is not None
            ):
                raise FormalSupervisorError("aggregate_result_not_safe")
            _validate_safe_numeric_tree(child, depth=depth + 1)
        return
    raise FormalSupervisorError("aggregate_result_not_safe")


def _validate_safe_aggregate(value: Mapping[str, Any]) -> None:
    if (
        set(value)
        != {
            "status",
            "arm_aggregates",
            "paired_aggregate_differences",
        }
        or not isinstance(value["status"], str)
        or _SAFE_COMPONENT.fullmatch(value["status"]) is None
        or not isinstance(value["arm_aggregates"], dict)
        or set(value["arm_aggregates"]) != set(protocol.ARM_IDS)
        or not isinstance(value["paired_aggregate_differences"], dict)
    ):
        raise FormalSupervisorError("aggregate_result_not_safe")
    _validate_safe_numeric_tree(value["arm_aggregates"])
    _validate_safe_numeric_tree(value["paired_aggregate_differences"])


class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _LandlockPathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


def _validate_internal_environment_overrides(
    value: Mapping[str, str],
) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) - {"CUDA_VISIBLE_DEVICES"}
        or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in value.items()
        )
    ):
        raise FormalSupervisorError(
            "internal_environment_override_invalid"
        )
    cuda = value.get("CUDA_VISIBLE_DEVICES")
    if cuda is not None and (
        cuda != ""
        and (
            re.fullmatch(r"[0-9]{1,2}", cuda) is None
            or not 0 <= int(cuda) <= 15
        )
    ):
        raise FormalSupervisorError(
            "internal_environment_override_invalid"
        )
    return dict(sorted(value.items()))


def _existing_system_read_paths() -> tuple[Path, ...]:
    """Frozen local-runtime reads; no system path is writable."""

    candidates = (
        Path("/usr"),
        Path("/etc"),
        Path("/proc"),
        Path("/sys"),
        Path("/dev/null"),
        Path("/dev/urandom"),
    )
    return tuple(path for path in candidates if path.exists())


def _gpu_device_candidates(index: int) -> tuple[Path, ...]:
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index <= 15
    ):
        raise FormalSupervisorError("gpu_device_index_invalid")
    return (
        *(
            Path(f"/dev/nvidia{physical}")
            for physical in _FROZEN_PHYSICAL_GPU_INDICES
        ),
        *_GPU_CONTROL_DEVICES,
        *_GPU_OPTIONAL_CONTROL_DEVICES,
    )


def _validated_gpu_device_rows(index: int) -> list[dict[str, Any]]:
    """Bind both frozen nodes needed for driver enumeration plus controls.

    Logical single-GPU execution remains fixed by ``CUDA_VISIBLE_DEVICES``;
    this file allowlist is not claimed to provide physical GPU exclusivity.
    """

    required = {
        *(
            Path(f"/dev/nvidia{physical}")
            for physical in _FROZEN_PHYSICAL_GPU_INDICES
        ),
        *_GPU_CONTROL_DEVICES,
    }
    rows: list[dict[str, Any]] = []
    for path in _gpu_device_candidates(index):
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            if path not in required:
                continue
            raise FormalSupervisorError(
                "required_gpu_device_missing"
            ) from exc
        if (
            path.is_symlink()
            or not stat.S_ISCHR(metadata.st_mode)
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode)
            not in {0o600, 0o620, 0o660, 0o666}
            or not os.access(path, os.R_OK | os.W_OK)
        ):
            raise FormalSupervisorError("gpu_device_topology_invalid")
        rows.append(
            {
                "gid": int(metadata.st_gid),
                "major": int(os.major(metadata.st_rdev)),
                "minor": int(os.minor(metadata.st_rdev)),
                "mode": int(stat.S_IMODE(metadata.st_mode)),
                "path": str(path),
                "uid": int(metadata.st_uid),
            }
        )
    if not required.issubset(
        {Path(row["path"]) for row in rows}
    ):
        raise FormalSupervisorError("required_gpu_device_missing")
    return sorted(rows, key=lambda row: row["path"])


def _landlock_syscall(libc: Any, number: int, *arguments: object) -> int:
    result = int(libc.syscall(number, *arguments))
    if result < 0:
        raise FormalSupervisorError(
            f"landlock_syscall_{number}_errno_{ctypes.get_errno()}"
        )
    return result


def _add_landlock_rule(
    *,
    libc: Any,
    ruleset_fd: int,
    path: Path,
    allowed_access: int,
) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise FormalSupervisorError(
            "landlock_allowlist_path_unavailable"
        ) from exc
    if path.is_symlink():
        raise FormalSupervisorError("landlock_allowlist_symlink")
    if stat.S_ISDIR(metadata.st_mode):
        _safe_absolute_path(path, allow_file=False)
    elif stat.S_ISREG(metadata.st_mode):
        _safe_absolute_path(path, allow_file=True)
        allowed_access &= ~LANDLOCK_ACCESS_FS_READ_DIR
    elif stat.S_ISCHR(metadata.st_mode):
        allowed_access &= (
            LANDLOCK_ACCESS_FS_READ_FILE
            | LANDLOCK_ACCESS_FS_WRITE_FILE
        )
    else:
        raise FormalSupervisorError("landlock_allowlist_type_invalid")
    descriptor = os.open(
        path,
        os.O_PATH | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        attribute = _LandlockPathBeneathAttr(
            allowed_access=allowed_access,
            parent_fd=descriptor,
        )
        _landlock_syscall(
            libc,
            SYS_LANDLOCK_ADD_RULE,
            ruleset_fd,
            LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(attribute),
            0,
        )
    finally:
        os.close(descriptor)


def _apply_landlock(
    *,
    read_execute_roots: Sequence[Path],
    work_root: Path,
    denial_probes: Mapping[str, Path],
    gpu_device_index: int | None = None,
) -> Mapping[str, Any]:
    """Apply the same real Landlock mechanism used by the v3 qualifier."""

    if platform.machine() not in {"x86_64", "AMD64"}:
        raise FormalSupervisorError("landlock_architecture_unsupported")
    libc = ctypes.CDLL(None, use_errno=True)
    abi = _landlock_syscall(
        libc,
        SYS_LANDLOCK_CREATE_RULESET,
        0,
        0,
        LANDLOCK_CREATE_RULESET_VERSION,
    )
    if abi < LANDLOCK_MINIMUM_ABI:
        raise FormalSupervisorError("landlock_abi_too_old")
    ruleset_attribute = _LandlockRulesetAttr(
        handled_access_fs=LANDLOCK_HANDLED_ACCESS_FS
    )
    ruleset_fd = _landlock_syscall(
        libc,
        SYS_LANDLOCK_CREATE_RULESET,
        ctypes.byref(ruleset_attribute),
        ctypes.sizeof(ruleset_attribute),
        0,
    )
    gpu_device_rows = (
        _validated_gpu_device_rows(gpu_device_index)
        if gpu_device_index is not None
        else []
    )
    try:
        for path in read_execute_roots:
            _add_landlock_rule(
                libc=libc,
                ruleset_fd=ruleset_fd,
                path=path,
                allowed_access=LANDLOCK_READ_EXECUTE_ACCESS,
            )
        _add_landlock_rule(
            libc=libc,
            ruleset_fd=ruleset_fd,
            path=work_root,
            allowed_access=LANDLOCK_WORK_ACCESS,
        )
        _add_landlock_rule(
            libc=libc,
            ruleset_fd=ruleset_fd,
            path=Path("/dev/null"),
            allowed_access=(
                LANDLOCK_ACCESS_FS_READ_FILE
                | LANDLOCK_ACCESS_FS_WRITE_FILE
            ),
        )
        for row in gpu_device_rows:
            _add_landlock_rule(
                libc=libc,
                ruleset_fd=ruleset_fd,
                path=Path(row["path"]),
                allowed_access=(
                    LANDLOCK_ACCESS_FS_READ_FILE
                    | LANDLOCK_ACCESS_FS_WRITE_FILE
                ),
            )
        # CUDA writes the current task name while initializing.  This rule is
        # installed in the actual exec child and binds its numeric proc task
        # directory.  The numeric form avoids the /proc/self symlink while
        # retaining the same-process-only scope.
        if gpu_device_index is not None:
            current_process_task_root = Path(
                f"/proc/{os.getpid()}/task"
            )
            _add_landlock_rule(
                libc=libc,
                ruleset_fd=ruleset_fd,
                path=current_process_task_root,
                allowed_access=(
                    LANDLOCK_ACCESS_FS_READ_FILE
                    | LANDLOCK_ACCESS_FS_READ_DIR
                    | LANDLOCK_ACCESS_FS_WRITE_FILE
                    | LANDLOCK_ACCESS_FS_TRUNCATE
                ),
            )
        if int(libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) < 0:
            raise FormalSupervisorError(
                f"landlock_no_new_privs_errno_{ctypes.get_errno()}"
            )
        _landlock_syscall(
            libc, SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0
        )
    finally:
        os.close(ruleset_fd)

    denial_errnos: dict[str, int] = {}
    for label, path in denial_probes.items():
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EPERM}:
                raise FormalSupervisorError(
                    f"landlock_probe_{label}_wrong_errno"
                ) from exc
            denial_errnos[label] = int(exc.errno)
        else:
            os.close(descriptor)
            raise FormalSupervisorError(
                f"landlock_probe_{label}_was_readable"
            )
    return {
        "abi": abi,
        "direct_parent_authority_sha256": hashlib.sha256(
            SUPERVISOR_LANDLOCK_DIRECT_PARENT_AUTHORITY.encode("ascii")
        ).hexdigest(),
        "denial_errnos": dict(sorted(denial_errnos.items())),
        "gpu_device_index": gpu_device_index,
        "gpu_device_rows": gpu_device_rows,
        "gpu_device_allowlist_self_hash": _content_hash(
            {"gpu_device_rows": gpu_device_rows}
        ),
        "gpu_proc_self_task_write_allowed": (
            gpu_device_index is not None
        ),
        "handled_access_fs": LANDLOCK_HANDLED_ACCESS_FS,
    }


def _write_child_json_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(dict(value))
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            offset += os.write(descriptor, view[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def _hash_child_work_file(path: Path, *, work_root: Path) -> str:
    try:
        path.relative_to(work_root)
    except ValueError as exc:
        raise FormalSupervisorError("child_output_outside_work") from exc
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise FormalSupervisorError("child_output_topology_invalid")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _sandbox_child(spec_path: Path) -> int:
    """Trusted launcher entry used by :meth:`FormalSupervisor.run_arm_once`."""

    raw_spec = spec_path.read_bytes()
    spec = _parse_json(raw_spec, issue_id="sandbox_spec_invalid")
    body = dict(spec)
    claimed = body.pop("self_hash", None)
    if (
        spec.get("schema") != SANDBOX_SPEC_SCHEMA
        or not isinstance(claimed, str)
        or _content_hash(body) != claimed
    ):
        raise FormalSupervisorError("sandbox_spec_invalid")
    overrides = _validate_internal_environment_overrides(
        spec.get("environment_overrides", {})
    )
    gpu_device_index = spec.get("gpu_device_index")
    if gpu_device_index is not None and (
        isinstance(gpu_device_index, bool)
        or not isinstance(gpu_device_index, int)
        or overrides.get("CUDA_VISIBLE_DEVICES")
        != str(gpu_device_index)
    ):
        raise FormalSupervisorError("sandbox_gpu_binding_invalid")
    if (
        gpu_device_index is None
        and overrides.get("CUDA_VISIBLE_DEVICES") not in {None, ""}
    ):
        raise FormalSupervisorError("sandbox_gpu_binding_invalid")
    work_root = Path(spec["work_root"])
    runtime_roots = [
        *_existing_system_read_paths(),
        *[Path(value) for value in spec["code_roots"]],
        *[Path(value) for value in spec["model_roots"]],
    ]
    # Keep only existing roots; on usr-merged systems /lib* can be symlinks and
    # /usr already covers their resolved targets.
    allowed_roots: list[Path] = []
    for path in runtime_roots:
        resolved = path.resolve()
        if resolved.exists() and resolved not in allowed_roots:
            allowed_roots.append(resolved)
    probes = {
        label: Path(path)
        for label, path in spec["private_denial_probes"].items()
    }
    environment = {
        "HOME": str(work_root),
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": _LOCAL_PYTHONPATH,
        "TEMP": str(work_root),
        "TMP": str(work_root),
        "TMPDIR": str(work_root),
        **OFFLINE_ENVIRONMENT,
        SUPERVISOR_LANDLOCK_DIRECT_PARENT_ENVIRONMENT_KEY: (
            SUPERVISOR_LANDLOCK_DIRECT_PARENT_AUTHORITY
        ),
        **overrides,
    }
    read_descriptor, write_descriptor = os.pipe2(
        os.O_CLOEXEC
    )

    def restrict_actual_exec_child() -> None:
        os.close(read_descriptor)
        landlock_child = _apply_landlock(
            read_execute_roots=allowed_roots,
            work_root=work_root,
            denial_probes=probes,
            gpu_device_index=gpu_device_index,
        )
        raw = _canonical_bytes(dict(landlock_child))
        if len(raw) > 64 * 1024:
            os._exit(96)
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            offset += os.write(write_descriptor, view[offset:])
        os.close(write_descriptor)

    try:
        completed = subprocess.run(
            list(spec["command"]),
            cwd=work_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=3_600,
            pass_fds=(write_descriptor,),
            preexec_fn=restrict_actual_exec_child,
        )
    finally:
        os.close(write_descriptor)
    landlock_raw = bytearray()
    try:
        while True:
            chunk = os.read(read_descriptor, 4096)
            if not chunk:
                break
            landlock_raw.extend(chunk)
            if len(landlock_raw) > 64 * 1024:
                raise FormalSupervisorError(
                    "landlock_child_receipt_too_large"
                )
    finally:
        os.close(read_descriptor)
    landlock = _parse_json(
        bytes(landlock_raw), issue_id="landlock_child_receipt_invalid"
    )
    if (
        landlock.get("gpu_device_index") != gpu_device_index
        or not isinstance(landlock.get("gpu_device_rows"), list)
        or landlock.get("direct_parent_authority_sha256")
        != hashlib.sha256(
            SUPERVISOR_LANDLOCK_DIRECT_PARENT_AUTHORITY.encode("ascii")
        ).hexdigest()
        or not isinstance(
            landlock.get("gpu_device_allowlist_self_hash"), str
        )
        or _content_hash(
            {"gpu_device_rows": landlock["gpu_device_rows"]}
        )
        != landlock["gpu_device_allowlist_self_hash"]
    ):
        raise FormalSupervisorError("landlock_child_receipt_invalid")
    output_path = Path(spec["prediction_output_path"])
    output_hash = (
        _hash_child_work_file(output_path, work_root=work_root)
        if completed.returncode == 0
        else None
    )
    receipt: dict[str, Any] = {
        "schema": SANDBOX_RECEIPT_SCHEMA,
        "status": (
            "LANDLOCK_ARM_COMPLETED"
            if completed.returncode == 0
            else "LANDLOCK_ARM_FAILED"
        ),
        "sandbox_spec_self_hash": spec["self_hash"],
        "one_shot_key": spec["one_shot_key"],
        "arm_id": spec["arm_id"],
        "implementation_sha256": spec["implementation_sha256"],
        "landlock_abi": landlock["abi"],
        "landlock_handled_access_fs": landlock["handled_access_fs"],
        "landlock_direct_parent_authority_sha256": landlock[
            "direct_parent_authority_sha256"
        ],
        "gpu_device_index": gpu_device_index,
        "gpu_device_rows": landlock["gpu_device_rows"],
        "gpu_device_allowlist_self_hash": landlock[
            "gpu_device_allowlist_self_hash"
        ],
        "gpu_proc_self_task_write_allowed": landlock[
            "gpu_proc_self_task_write_allowed"
        ],
        "environment_overrides": overrides,
        "label_denial_errno": landlock["denial_errnos"]["label_pack"],
        "linkage_denial_errno": landlock["denial_errnos"][
            "linkage_pack"
        ],
        "arm_exit_code": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "prediction_output_sha256": output_hash,
        "item_content_emitted": False,
    }
    receipt["self_hash"] = _content_hash(receipt)
    _write_child_json_exclusive(
        Path(spec["sandbox_receipt_path"]), receipt
    )
    return 0 if completed.returncode == 0 else 1


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sandbox-child", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.sandbox_child is None:
        raise FormalSupervisorError("formal_cli_not_materialized")
    return _sandbox_child(arguments.sandbox_child)


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except FormalSupervisorError:
        raise SystemExit(97)


__all__ = [
    "ACTION_SCHEMA",
    "ArmCommand",
    "FORMAL_ROOT",
    "FormalInvocation",
    "FormalSupervisor",
    "FormalSupervisorError",
    "FrozenAction",
    "INTERNAL_FACTORY_QUALIFICATION_SCHEMA",
    "OUTER_SYSTEMD_ATTESTATION_SCHEMA",
    "OUTER_SYSTEMD_CONTRACT",
    "RuntimeClosure",
    "ScorerCommand",
    "SecureDirectory",
    "TestAttestation",
    "VERSION",
    "attest_runtime_closure",
    "run_source_free_tests",
]
