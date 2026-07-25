"""A_hold candidate-restricted official HippoRAG block runtime for MMQA P1.

The frozen ERASER Evidence-Inference adapter runs each official HippoRAG item
inside a fresh ephemeral index.  The 311 host's AppArmor/user-namespace policy
makes nested bubblewrap unavailable, so network denial is inherited from the
parent systemd service's effective ``RestrictAddressFamilies=AF_UNIX`` policy.
Before preflight, receipt revalidation, and the formal block, this module
actively proves that creating both AF_INET and AF_INET6 sockets fails with
EAFNOSUPPORT.  Runtime-inspector, canary, and item workers are direct child
processes and inherit that restriction; this module makes no private-network-
namespace claim.

The attested environment explicitly sets ``CUDA_VISIBLE_DEVICES=""``.
Therefore the MMQA design phrase "two per GPU" cannot truthfully be implemented
without changing the already-attested adapter/runtime identity.  This
prospective source-unopened implementation freezes the compatible disposition
instead: exactly one four-worker CPU pool, zero visible CUDA devices, one
retrieve-only official subprocess per A_hold item, and no retry.  The study
design/manifest is deliberately not edited here; the conflict is exposed in
constants and terminal metadata for a later unified design disposition.

Inputs are opaque work IDs paired only with validated
``CandidateRestrictedHippoRAGPayload`` objects.  Per-item launchers receive no
source, gold, answer, family, qid, or evaluator.  The one-shot mode-0600
terminal archive stores only work IDs, top-five source-local ordinals, hashes,
and fixed audit counters/identities.  Query text, closure text, exact-text
quotients, indexes, and worker stdout are never persisted in that archive.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import subprocess
from typing import Callable, Mapping, Sequence

from assumption_agent.benchmarks.eraser_evidence_inference_official_hipporag_v1 import (
    adapter as official_adapter,
)
from assumption_agent.benchmarks.eraser_evidence_inference_official_hipporag_v1 import (
    contract as official_contract,
)

from . import mmqa_p1_local_action_executor_v1 as action_executor
from . import mmqa_p1_typed_proof_e5_core_v1 as core


VERSION = "mmqa_p1_official_hipporag_block_v1"
STUDY_ID = core.STUDY_ID
STUDY_DESIGN_SELF_SHA256 = action_executor.STUDY_DESIGN_SELF_SHA256
BLOCK = "A_hold"
A_HOLD_ITEM_COUNT = 45
MAX_WORKERS = 4
ITEM_TIMEOUT_SECONDS = 900

# Implementation-before-source disposition.  The existing frozen adapter
# hides CUDA and the attested file identity below binds that fact.
ATTESTED_CUDA_VISIBLE_DEVICES = ""
EXECUTION_DEVICE_DISPOSITION = (
    "four_way_cpu_official_adapter_cuda_hidden_v1"
)
DESIGN_TWO_PER_GPU_APPLIED = False
DESIGN_CONCURRENCY_CONFLICT_REQUIRES_UNIFIED_DISPOSITION = True
NETWORK_ISOLATION_POLICY = (
    "parent_systemd_RestrictAddressFamilies_AF_UNIX_inherited_v1"
)
# Compatibility alias for callers that imported the old constant name.  Its
# value explicitly does not claim a private network namespace.
NETWORK_NAMESPACE_POLICY = NETWORK_ISOLATION_POLICY
ADDRESS_FAMILY_ISOLATION_PROBE_SCHEMA = (
    f"{VERSION}_restrict_address_families_probe_v1"
)
ADDRESS_FAMILY_ISOLATION_CONTRACT = {
    "supervisor": "systemd_service",
    "required_RestrictAddressFamilies": ["AF_UNIX"],
    "denied_socket_families": ["AF_INET", "AF_INET6"],
    "required_socket_creation_errno": "EAFNOSUPPORT",
    "worker_launch": "direct_subprocess_inherits_parent_restriction",
    "private_network_namespace_claimed": False,
}
INDEX_LIFECYCLE_POLICY = "fresh_isolated_destroyed_per_item_v1"
RETRIEVAL_POLICY = "official_retrieve_only_no_qa_v1"
CPU_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
FRESH_PREFLIGHT_SCHEMA = f"{VERSION}_fresh_comparator_preflight_v1"

ERASER_ADAPTER_FILE_SHA256 = (
    "d2b564e3cd65bc038b03a60c6950dcace113f5743c6c7f6424d9294d7a187eb3"
)
ERASER_WORKER_FILE_SHA256 = (
    "6e1829f8a4ce0066bc665141441ed4f3a9f011e5ef0ae27088f974ad9ef0e835"
)
ERASER_CONTRACT_FILE_SHA256 = (
    "7c97261dad90803877cd791393e489236e807207a5a5d02e1d7a6345b7425679"
)
TERMINAL_ARCHIVE_SCHEMA = f"{VERSION}_private_terminal_archive_v1"
_WORK_ID = re.compile(r"mmqa-work-v1-[0-9a-f]{64}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_ARCHIVE_FIELDS = frozenset(
    {
        "schema",
        "study_id",
        "study_design_self_sha256",
        "block",
        "item_count",
        "max_workers",
        "item_timeout_seconds",
        "execution_device_disposition",
        "attested_cuda_visible_devices",
        "cpu_thread_env_sha256",
        "design_two_per_gpu_applied",
        "design_concurrency_conflict_requires_unified_disposition",
        "network_isolation_policy",
        "address_family_isolation_contract_sha256",
        "address_family_isolation_probe_sha256",
        "index_lifecycle_policy",
        "retrieval_policy",
        "runtime_binding_sha256",
        "A_hold_input_sha256",
        "item_launcher_call_count",
        "fresh_isolated_index_count",
        "parent_address_family_restriction_inherited_count",
        "bwrap_call_count",
        "retry_replay_resample_count",
        "online_evaluator_call_count",
        "rows",
        "archive_sha256",
    }
)
_ROW_FIELDS = frozenset(
    {
        "work_id",
        "top5_source_ordinals",
        "candidate_payload_sha256",
        "closure_ordinal_bytes_sha256",
        "worker_output_sha256",
    }
)


class MmqaP1OfficialHippoRAGBlockError(RuntimeError):
    """The A_hold official runtime, concurrency, or terminal contract drifted."""


def _canonical_json_bytes(value: object, *, newline: bool = False) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "official block value is not canonical JSON"
        ) from exc
    return (text + ("\n" if newline else "")).encode("ascii")


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise MmqaP1OfficialHippoRAGBlockError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _work_id(value: object) -> str:
    if not isinstance(value, str) or _WORK_ID.fullmatch(value) is None:
        raise MmqaP1OfficialHippoRAGBlockError(
            "work_id must be an opaque MMQA work identity"
        )
    return value


def _absolute_lexical_path(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value == "/"
        or value.endswith("/")
        or "\x00" in value
        or "//" in value
        or "/./" in value
        or "/../" in value
        or value.endswith("/.")
        or value.endswith("/..")
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            f"{field} must be one normalized absolute lexical path"
        )
    return value


def _strict_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise MmqaP1OfficialHippoRAGBlockError(
            f"{field} must be an exact integer at least {minimum}"
        )
    return value


def _exact_fields(
    value: Mapping[str, object], expected: frozenset[str], field: str
) -> None:
    if set(value) != expected:
        raise MmqaP1OfficialHippoRAGBlockError(f"{field} schema drifted")


_ADDRESS_FAMILY_PROBE_FIELDS = frozenset(
    {
        "schema",
        "required_RestrictAddressFamilies",
        "AF_INET_socket_creation_errno",
        "AF_INET6_socket_creation_errno",
        "all_inet_socket_creation_denied",
        "private_network_namespace_claimed",
        "probe_count",
    }
)


def _validated_address_family_isolation_probe(value: object) -> dict[str, object]:
    """Validate the exact effective AF_UNIX-only socket-creation observation."""

    if not isinstance(value, Mapping):
        raise MmqaP1OfficialHippoRAGBlockError(
            "address-family isolation probe returned no mapping"
        )
    _exact_fields(
        value,
        _ADDRESS_FAMILY_PROBE_FIELDS,
        "address-family isolation probe",
    )
    if (
        value.get("schema") != ADDRESS_FAMILY_ISOLATION_PROBE_SCHEMA
        or value.get("required_RestrictAddressFamilies") != ["AF_UNIX"]
        or value.get("AF_INET_socket_creation_errno") != "EAFNOSUPPORT"
        or value.get("AF_INET6_socket_creation_errno") != "EAFNOSUPPORT"
        or value.get("all_inet_socket_creation_denied") is not True
        or value.get("private_network_namespace_claimed") is not False
        or value.get("probe_count") != 2
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "effective RestrictAddressFamilies=AF_UNIX probe failed"
        )
    return dict(value)


def production_address_family_isolation_probe() -> dict[str, object]:
    """Require AF_INET/AF_INET6 socket creation to fail with EAFNOSUPPORT.

    This is an effective-kernel-policy probe.  It deliberately does not infer
    or claim that systemd successfully created a private network namespace.
    """

    observed: dict[str, str] = {}
    for name, family in (
        ("AF_INET", socket.AF_INET),
        ("AF_INET6", socket.AF_INET6),
    ):
        handle: socket.socket | None = None
        try:
            handle = socket.socket(family, socket.SOCK_STREAM)
        except OSError as exc:
            error_name = errno.errorcode.get(exc.errno or -1, "UNKNOWN")
            if exc.errno != errno.EAFNOSUPPORT:
                raise MmqaP1OfficialHippoRAGBlockError(
                    f"{name} socket creation failed with {error_name}, "
                    "not EAFNOSUPPORT"
                ) from exc
            observed[name] = error_name
        else:
            raise MmqaP1OfficialHippoRAGBlockError(
                f"{name} socket creation succeeded; "
                "RestrictAddressFamilies=AF_UNIX is not effective"
            )
        finally:
            if handle is not None:
                handle.close()
    return _validated_address_family_isolation_probe(
        {
            "schema": ADDRESS_FAMILY_ISOLATION_PROBE_SCHEMA,
            "required_RestrictAddressFamilies": ["AF_UNIX"],
            "AF_INET_socket_creation_errno": observed["AF_INET"],
            "AF_INET6_socket_creation_errno": observed["AF_INET6"],
            "all_inet_socket_creation_denied": True,
            "private_network_namespace_claimed": False,
            "probe_count": 2,
        }
    )


@dataclass(frozen=True)
class AHoldHippoItem:
    work_id: str
    payload: action_executor.CandidateRestrictedHippoRAGPayload

    def __post_init__(self) -> None:
        object.__setattr__(self, "work_id", _work_id(self.work_id))
        if not isinstance(
            self.payload,
            action_executor.CandidateRestrictedHippoRAGPayload,
        ):
            raise MmqaP1OfficialHippoRAGBlockError(
                "A_hold item requires a candidate-restricted payload"
            )

    @property
    def candidate_payload_sha256(self) -> str:
        return hashlib.sha256(
            self.payload.canonical_worker_bytes()
        ).hexdigest()

    def binding(self) -> dict[str, object]:
        return {
            "work_id": self.work_id,
            "candidate_payload_sha256": self.candidate_payload_sha256,
            "closure_ordinal_bytes_sha256": (
                self.payload.closure_ordinal_bytes_sha256
            ),
            "logical_source_ordinal_vector_sha256": _semantic_hash(
                list(self.payload.logical_source_ordinals)
            ),
        }


def validate_ahold_items(
    items: Sequence[AHoldHippoItem],
) -> tuple[AHoldHippoItem, ...]:
    if (
        isinstance(items, (str, bytes))
        or not isinstance(items, Sequence)
        or len(items) != A_HOLD_ITEM_COUNT
        or not all(isinstance(item, AHoldHippoItem) for item in items)
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "official runtime requires the complete 45-item A_hold block"
        )
    checked = tuple(items)
    identifiers = tuple(item.work_id for item in checked)
    if len(set(identifiers)) != len(identifiers):
        raise MmqaP1OfficialHippoRAGBlockError(
            "A_hold contains duplicate work_id"
        )
    return checked


def ahold_input_sha256(items: Sequence[AHoldHippoItem]) -> str:
    return _semantic_hash(
        [item.binding() for item in validate_ahold_items(items)]
    )


_FRESH_BINDING_CAPABILITY = object()
_FILESYSTEM_BINDING_FIELDS = frozenset(
    {
        "runtime_python",
        "pyvenv_cfg",
        "overlay_root",
        "hipporag_source_root",
        "p16_site_root",
        "local_llm_model",
        "local_embedding_model",
        "eraser_adapter_file",
        "eraser_worker_file",
        "eraser_contract_file",
    }
)
_RUNTIME_PROBE_FIELDS = frozenset(
    {
        "python_version",
        "package_versions",
        "module_origins",
        "module_import_roots",
        "sys_path_sha256",
        "cuda_visible_devices",
        "cpu_thread_env",
        "address_family_isolation_probe",
    }
)
_PREFLIGHT_FIELDS = frozenset(
    {
        "schema",
        "study_id",
        "study_design_self_sha256",
        "status",
        "path_binding_sha256",
        "filesystem_binding",
        "filesystem_binding_sha256",
        "runtime_probe",
        "runtime_probe_sha256",
        "expected_package_versions",
        "expected_module_import_roots",
        "public_synthetic_fixture_sha256",
        "public_synthetic_output_sha256",
        "public_synthetic_candidate_count",
        "public_synthetic_output_count",
        "official_core_index_called",
        "official_core_retrieve_called",
        "benchmark_rows_read",
        "scores_computed",
        "address_family_isolation_contract",
        "address_family_isolation_contract_sha256",
        "address_family_isolation_probe",
        "address_family_isolation_probe_sha256",
        "worker_subprocess_inherits_parent_restriction",
        "bwrap_call_count",
        "cuda_visible_devices",
        "cpu_thread_env",
        "retry_count",
        "receipt_sha256",
    }
)
_REQUIRED_VERSION_KEYS = frozenset(
    {"torch", "sentence-transformers", "transformers"}
)
_REQUIRED_MODULE_KEYS = frozenset(
    {"torch", "sentence_transformers", "transformers", "hipporag"}
)


@dataclass(frozen=True)
class FreshComparatorRuntimePaths:
    """Lexical paths for the actual 311 comparator runtime and import roots."""

    runtime_python: str
    pyvenv_cfg: str
    overlay_root: str
    hipporag_source_root: str
    p16_site_root: str
    local_llm_model: str
    local_embedding_model: str

    def __post_init__(self) -> None:
        for field in (
            "runtime_python",
            "pyvenv_cfg",
            "overlay_root",
            "hipporag_source_root",
            "p16_site_root",
            "local_llm_model",
            "local_embedding_model",
        ):
            object.__setattr__(
                self,
                field,
                _absolute_lexical_path(getattr(self, field), field),
            )

    def path_binding(self) -> dict[str, str]:
        return {
            f"{field}_path_sha256": hashlib.sha256(
                getattr(self, field).encode("utf-8")
            ).hexdigest()
            for field in (
                "runtime_python",
                "pyvenv_cfg",
                "overlay_root",
                "hipporag_source_root",
                "p16_site_root",
                "local_llm_model",
                "local_embedding_model",
            )
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> str:
    lexical = path.absolute()
    try:
        if lexical.is_symlink():
            target = os.readlink(lexical)
            resolved = lexical.resolve(strict=True)
            value = {
                "kind": "symlink_to_regular_file",
                "link_target": target,
                "resolved_path_sha256": hashlib.sha256(
                    str(resolved).encode("utf-8")
                ).hexdigest(),
                "resolved_file_sha256": _sha256_file(resolved),
            }
        elif lexical.is_file():
            value = {
                "kind": "regular_file",
                "file_sha256": _sha256_file(lexical),
            }
        else:
            raise MmqaP1OfficialHippoRAGBlockError(
                "fresh comparator file is unavailable"
            )
    except (OSError, RuntimeError) as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh comparator file cannot be inspected"
        ) from exc
    return _semantic_hash(value)


def _tree_identity(path: Path) -> str:
    root = path.absolute()
    if root.is_symlink() or not root.is_dir():
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh comparator tree root is unavailable or a symlink"
        )
    rows = []
    try:
        for entry in sorted(
            root.rglob("*"),
            key=lambda value: value.relative_to(root).as_posix(),
        ):
            relative = entry.relative_to(root).as_posix()
            metadata = entry.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                resolved = entry.resolve(strict=True)
                if not resolved.is_file():
                    raise MmqaP1OfficialHippoRAGBlockError(
                        "fresh comparator tree contains a non-file symlink"
                    )
                rows.append(
                    {
                        "path": relative,
                        "kind": "symlink_to_regular_file",
                        "target": os.readlink(entry),
                        "resolved_path_sha256": hashlib.sha256(
                            str(resolved).encode("utf-8")
                        ).hexdigest(),
                        "resolved_size": resolved.stat().st_size,
                        "resolved_file_sha256": _sha256_file(resolved),
                    }
                )
            elif stat.S_ISDIR(metadata.st_mode):
                rows.append({"path": relative, "kind": "directory"})
            elif stat.S_ISREG(metadata.st_mode):
                rows.append(
                    {
                        "path": relative,
                        "kind": "regular_file",
                        "size": metadata.st_size,
                        "sha256": _sha256_file(entry),
                    }
                )
            else:
                raise MmqaP1OfficialHippoRAGBlockError(
                    "fresh comparator tree has an unclassified entry"
                )
    except OSError as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh comparator tree cannot be inspected"
        ) from exc
    return _semantic_hash(rows)


def production_filesystem_inspector(
    paths: FreshComparatorRuntimePaths,
) -> dict[str, str]:
    """Hash the actual runtime, overlay/source/site roots, and model assets."""

    if not isinstance(paths, FreshComparatorRuntimePaths):
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh comparator paths type drifted"
        )
    adapter_path = Path(str(official_adapter.__file__)).absolute()
    worker_path = adapter_path.with_name("worker.py")
    contract_path = adapter_path.with_name("contract.py")
    result = {
        "runtime_python": _file_identity(Path(paths.runtime_python)),
        "pyvenv_cfg": _file_identity(Path(paths.pyvenv_cfg)),
        "overlay_root": _tree_identity(Path(paths.overlay_root)),
        "hipporag_source_root": _tree_identity(
            Path(paths.hipporag_source_root)
        ),
        "p16_site_root": _tree_identity(Path(paths.p16_site_root)),
        "local_llm_model": _tree_identity(Path(paths.local_llm_model)),
        "local_embedding_model": _tree_identity(
            Path(paths.local_embedding_model)
        ),
        "eraser_adapter_file": _sha256_file(adapter_path),
        "eraser_worker_file": _sha256_file(worker_path),
        "eraser_contract_file": _sha256_file(contract_path),
    }
    if (
        result["eraser_adapter_file"] != ERASER_ADAPTER_FILE_SHA256
        or result["eraser_worker_file"] != ERASER_WORKER_FILE_SHA256
        or result["eraser_contract_file"] != ERASER_CONTRACT_FILE_SHA256
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "frozen ERASER adapter/worker/contract identity drifted"
        )
    return result


def _import_root_label(
    origin: str, paths: FreshComparatorRuntimePaths
) -> str:
    if not isinstance(origin, str) or not origin.startswith("/"):
        raise MmqaP1OfficialHippoRAGBlockError(
            "runtime module origin is not an absolute file"
        )
    candidate = Path(origin).absolute()
    roots = {
        "overlay_root": Path(paths.overlay_root).absolute(),
        "hipporag_source_root": Path(paths.hipporag_source_root).absolute(),
        "p16_site_root": Path(paths.p16_site_root).absolute(),
    }
    matches = []
    for label, root in roots.items():
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        matches.append((len(root.parts), label))
    if not matches:
        raise MmqaP1OfficialHippoRAGBlockError(
            "runtime module origin escaped declared import roots"
        )
    return max(matches)[1]


def production_runtime_inspector(
    paths: FreshComparatorRuntimePaths,
    *,
    package_names: Sequence[str],
    module_names: Sequence[str],
) -> dict[str, object]:
    """Inspect imports in a direct child inheriting the AF_UNIX-only sandbox."""

    if not isinstance(paths, FreshComparatorRuntimePaths):
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh runtime inspector paths drifted"
        )
    parent_isolation = production_address_family_isolation_probe()
    packages = tuple(package_names)
    modules = tuple(module_names)
    if (
        not packages
        or len(set(packages)) != len(packages)
        or not modules
        or len(set(modules)) != len(modules)
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "runtime inspection registry drifted"
        )
    script = (
        "import errno,importlib,importlib.metadata,json,os,socket,sys\n"
        "packages=json.loads(sys.argv[1]);modules=json.loads(sys.argv[2])\n"
        "def socket_probe(name):\n"
        " family=getattr(socket,name)\n"
        " try:\n"
        "  handle=socket.socket(family,socket.SOCK_STREAM)\n"
        " except OSError as exc:\n"
        "  return errno.errorcode.get(exc.errno or -1,'UNKNOWN')\n"
        " else:\n"
        "  handle.close();return 'ALLOWED'\n"
        "loaded={name:importlib.import_module(name) for name in modules}\n"
        "value={'python_version':'.'.join(map(str,sys.version_info[:3])),"
        "'package_versions':{name:importlib.metadata.version(name) for name in packages},"
        "'module_origins':{name:str(getattr(module,'__file__',None)) for name,module in loaded.items()},"
        "'sys_path':list(sys.path),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),"
        "'cpu_thread_env':{name:os.environ.get(name) for name in "
        + repr(tuple(CPU_THREAD_ENV))
        + "},'address_family_isolation_probe':{"
        "'schema':"
        + repr(ADDRESS_FAMILY_ISOLATION_PROBE_SCHEMA)
        + ",'required_RestrictAddressFamilies':['AF_UNIX'],"
        "'AF_INET_socket_creation_errno':socket_probe('AF_INET'),"
        "'AF_INET6_socket_creation_errno':socket_probe('AF_INET6'),"
        "'all_inet_socket_creation_denied':True,"
        "'private_network_namespace_claimed':False,'probe_count':2}}\n"
        "print(json.dumps(value,sort_keys=True,separators=(',',':')))\n"
    )
    environment = {
        "PATH": f"{Path(paths.runtime_python).parent}:/usr/bin:/bin",
        "HOME": "/tmp",
        "HF_HOME": "/tmp/hf",
        "TMPDIR": "/tmp",
        "TMP": "/tmp",
        "TEMP": "/tmp",
        "PYTHONPATH": os.pathsep.join(
            (
                paths.overlay_root,
                paths.hipporag_source_root,
                paths.p16_site_root,
            )
        ),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        **CPU_THREAD_ENV,
    }
    command = [
        paths.runtime_python,
        "-B",
        "-c",
        script,
        json.dumps(list(packages), separators=(",", ":")),
        json.dumps(list(modules), separators=(",", ":")),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=environment,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh runtime inspection failed"
        ) from exc
    if completed.returncode != 0:
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh runtime inspection failed; "
            f"stdout_sha256={hashlib.sha256(completed.stdout).hexdigest()}; "
            f"stderr_sha256={hashlib.sha256(completed.stderr).hexdigest()}"
        )
    try:
        value = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh runtime inspection output is invalid"
        ) from exc
    if not isinstance(value, Mapping):
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh runtime inspection output is not an object"
        )
    origins = value.get("module_origins")
    sys_path = value.get("sys_path")
    if not isinstance(origins, Mapping) or not isinstance(sys_path, list):
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh runtime inspection origins drifted"
        )
    child_isolation = _validated_address_family_isolation_probe(
        value.get("address_family_isolation_probe")
    )
    if child_isolation != parent_isolation:
        raise MmqaP1OfficialHippoRAGBlockError(
            "runtime inspector child did not inherit address-family restriction"
        )
    return {
        "python_version": value.get("python_version"),
        "package_versions": dict(value.get("package_versions", {})),  # type: ignore[arg-type]
        "module_origins": dict(origins),
        "module_import_roots": {
            name: _import_root_label(str(origin), paths)
            for name, origin in origins.items()
        },
        "sys_path_sha256": _semantic_hash(sys_path),
        "cuda_visible_devices": value.get("cuda_visible_devices"),
        "cpu_thread_env": value.get("cpu_thread_env"),
        "address_family_isolation_probe": child_isolation,
    }


def public_synthetic_candidate_payload(
) -> action_executor.CandidateRestrictedHippoRAGPayload:
    texts = (
        "Synthetic Alpha links to Synthetic Beta.",
        "A locally generated copper circle is unrelated.",
        "Synthetic Beta completes the entirely local chain.",
        "A locally generated silver square is unrelated.",
        "A locally generated green triangle is unrelated.",
        "A locally generated amber hexagon is unrelated.",
    )
    ordinals = tuple(range(len(texts)))
    ordinal_bytes = _canonical_json_bytes(list(ordinals))
    return action_executor.CandidateRestrictedHippoRAGPayload(
        query="Which synthetic sentence completes the entirely local chain?",
        logical_source_ordinals=ordinals,
        exact_sentence_texts=texts,
        closure_ordinal_bytes_sha256=hashlib.sha256(
            ordinal_bytes
        ).hexdigest(),
        exact_text_quotient_count=len(texts),
    )


def _write_private_official_input(
    path: Path,
    payload: action_executor.CandidateRestrictedHippoRAGPayload,
) -> None:
    raw = payload.canonical_worker_bytes()
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh official private input cannot be created"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _launch_fresh_bound_worker(
    *,
    paths: FreshComparatorRuntimePaths,
    input_path: Path,
    output_path: Path,
    index_root: Path,
    writable_root: Path,
    timeout_seconds: int,
) -> None:
    # Probe immediately before the direct child launch.  The systemd
    # RestrictAddressFamilies filter is inherited across fork/exec.
    production_address_family_isolation_probe()
    environment = {
        "PATH": f"{Path(paths.runtime_python).parent}:/usr/bin:/bin",
        "HOME": str(writable_root / "home"),
        "HF_HOME": str(writable_root / "cache"),
        "TMPDIR": str(writable_root / "tmp"),
        "TMP": str(writable_root / "tmp"),
        "TEMP": str(writable_root / "tmp"),
        "PYTHONPATH": os.pathsep.join(
            (
                paths.overlay_root,
                paths.hipporag_source_root,
                paths.p16_site_root,
            )
        ),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        **CPU_THREAD_ENV,
    }
    command = [
        paths.runtime_python,
        "-B",
        "-m",
        (
            "assumption_agent.benchmarks."
            "eraser_evidence_inference_official_hipporag_v1.worker"
        ),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--index-root",
        str(index_root),
        "--llm-model",
        paths.local_llm_model,
        "--embedding-model",
        paths.local_embedding_model,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=environment,
            cwd=writable_root,
            stdin=subprocess.DEVNULL,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh official worker failed; no retry permitted"
        ) from exc
    if completed.returncode != 0:
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh official worker failed; "
            f"returncode={completed.returncode}; "
            f"stdout_sha256={hashlib.sha256(completed.stdout).hexdigest()}; "
            f"stderr_sha256={hashlib.sha256(completed.stderr).hexdigest()}"
        )


def run_fresh_bound_item(
    *,
    payload: action_executor.CandidateRestrictedHippoRAGPayload,
    runtime_paths: FreshComparatorRuntimePaths,
    work_root: Path,
    timeout_seconds: int = ITEM_TIMEOUT_SECONDS,
) -> tuple[int, ...]:
    """Run official worker semantics without the invalid legacy v3 receipt."""

    if (
        not isinstance(
            payload,
            action_executor.CandidateRestrictedHippoRAGPayload,
        )
        or not isinstance(runtime_paths, FreshComparatorRuntimePaths)
        or timeout_seconds != ITEM_TIMEOUT_SECONDS
        or os.path.lexists(work_root)
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh bound item contract drifted"
        )
    work_root.mkdir(mode=0o700)
    try:
        for name in ("home", "cache", "tmp"):
            (work_root / name).mkdir(mode=0o700)
        input_path = work_root / "single_item.input.json"
        output_path = work_root / "retrieved_ordinals.json"
        index_root = work_root / "official_item_index"
        _write_private_official_input(input_path, payload)
        _launch_fresh_bound_worker(
            paths=runtime_paths,
            input_path=input_path,
            output_path=output_path,
            index_root=index_root,
            writable_root=work_root,
            timeout_seconds=ITEM_TIMEOUT_SECONDS,
        )
        if output_path.is_symlink() or not output_path.is_file():
            raise MmqaP1OfficialHippoRAGBlockError(
                "fresh official worker emitted no ordinal-only output"
            )
        try:
            return official_contract.parse_ordinals_only_output(
                output_path.read_bytes(),
                logical_sentence_count=len(
                    payload.logical_source_ordinals
                ),
            )
        except official_contract.EraserEvidenceInferenceOfficialHippoRAGError as exc:
            raise MmqaP1OfficialHippoRAGBlockError(
                "fresh official worker terminal drifted"
            ) from exc
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


def production_public_canary_launcher(
    *,
    payload: action_executor.CandidateRestrictedHippoRAGPayload,
    runtime_paths: FreshComparatorRuntimePaths,
    work_root: Path,
    timeout_seconds: int,
) -> tuple[int, ...]:
    return run_fresh_bound_item(
        payload=payload,
        runtime_paths=runtime_paths,
        work_root=work_root,
        timeout_seconds=timeout_seconds,
    )


@dataclass(frozen=True)
class FreshComparatorRuntimeBinding:
    paths: FreshComparatorRuntimePaths
    receipt_sha256: str
    filesystem_binding_sha256: str
    runtime_probe_sha256: str
    address_family_isolation_probe_sha256: str
    public_synthetic_output_sha256: str
    _capability: object

    def __post_init__(self) -> None:
        if (
            not isinstance(self.paths, FreshComparatorRuntimePaths)
            or self._capability is not _FRESH_BINDING_CAPABILITY
        ):
            raise MmqaP1OfficialHippoRAGBlockError(
                "fresh comparator binding was not produced by preflight"
            )
        for value, field in (
            (self.receipt_sha256, "fresh preflight receipt"),
            (self.filesystem_binding_sha256, "filesystem binding"),
            (self.runtime_probe_sha256, "runtime probe"),
            (
                self.address_family_isolation_probe_sha256,
                "address-family isolation probe",
            ),
            (
                self.public_synthetic_output_sha256,
                "public synthetic output",
            ),
        ):
            _sha256(value, field)

    @property
    def binding_sha256(self) -> str:
        return _semantic_hash(
            {
                "receipt_sha256": self.receipt_sha256,
                "filesystem_binding_sha256": (
                    self.filesystem_binding_sha256
                ),
                "runtime_probe_sha256": self.runtime_probe_sha256,
                "address_family_isolation_contract_sha256": _semantic_hash(
                    ADDRESS_FAMILY_ISOLATION_CONTRACT
                ),
                "address_family_isolation_probe_sha256": (
                    self.address_family_isolation_probe_sha256
                ),
                "public_synthetic_output_sha256": (
                    self.public_synthetic_output_sha256
                ),
                "path_binding_sha256": _semantic_hash(
                    self.paths.path_binding()
                ),
                "cuda_visible_devices": "",
                "cpu_thread_env": CPU_THREAD_ENV,
                "max_workers": MAX_WORKERS,
                "worker_subprocess_inherits_parent_restriction": True,
                "bwrap_call_count": 0,
                "retry_count": 0,
            }
        )


def _validated_expected_maps(
    expected_package_versions: Mapping[str, str],
    expected_module_import_roots: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    if not isinstance(expected_package_versions, Mapping) or not isinstance(
        expected_module_import_roots, Mapping
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh runtime expected version/import registry drifted"
        )
    versions = dict(expected_package_versions)
    roots = dict(expected_module_import_roots)
    if (
        not _REQUIRED_VERSION_KEYS.issubset(versions)
        or not _REQUIRED_MODULE_KEYS.issubset(roots)
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in (*versions.items(), *roots.items())
        )
        or any(
            value
            not in {"overlay_root", "hipporag_source_root", "p16_site_root"}
            for value in roots.values()
        )
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh runtime expected version/import registry drifted"
        )
    return versions, roots


def _assert_stage_outside_runtime_roots(
    paths: FreshComparatorRuntimePaths,
    stage_parent: str | Path,
) -> None:
    stage = Path(stage_parent).expanduser().absolute()
    for root_text in (
        paths.overlay_root,
        paths.hipporag_source_root,
        paths.p16_site_root,
        paths.local_llm_model,
        paths.local_embedding_model,
    ):
        try:
            stage.relative_to(Path(root_text).absolute())
        except ValueError:
            continue
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh stage parent would mutate a bound runtime tree"
        )


def _validated_filesystem_binding(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh filesystem inspector returned no mapping"
        )
    _exact_fields(value, _FILESYSTEM_BINDING_FIELDS, "filesystem binding")
    result = {
        key: _sha256(raw, f"filesystem binding {key}")
        for key, raw in value.items()
    }
    if (
        result["eraser_adapter_file"] != ERASER_ADAPTER_FILE_SHA256
        or result["eraser_worker_file"] != ERASER_WORKER_FILE_SHA256
        or result["eraser_contract_file"] != ERASER_CONTRACT_FILE_SHA256
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh filesystem ERASER code identity drifted"
        )
    return result


def _validated_runtime_probe(
    value: object,
    *,
    expected_package_versions: Mapping[str, str],
    expected_module_import_roots: Mapping[str, str],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh runtime inspector returned no mapping"
        )
    _exact_fields(value, _RUNTIME_PROBE_FIELDS, "fresh runtime probe")
    isolation = _validated_address_family_isolation_probe(
        value.get("address_family_isolation_probe")
    )
    if (
        not isinstance(value.get("python_version"), str)
        or value.get("package_versions")
        != dict(expected_package_versions)
        or value.get("module_import_roots")
        != dict(expected_module_import_roots)
        or not isinstance(value.get("module_origins"), Mapping)
        or set(value["module_origins"])
        != set(expected_module_import_roots)
        or any(
            not isinstance(origin, str) or not origin.startswith("/")
            for origin in value["module_origins"].values()
        )
        or _HEX64.fullmatch(str(value.get("sys_path_sha256"))) is None
        or value.get("cuda_visible_devices") != ""
        or value.get("cpu_thread_env") != CPU_THREAD_ENV
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh runtime versions, imports, CPU, or no-network policy drifted"
        )
    return {**dict(value), "address_family_isolation_probe": isolation}


def build_fresh_comparator_preflight(
    *,
    paths: FreshComparatorRuntimePaths,
    expected_package_versions: Mapping[str, str],
    expected_module_import_roots: Mapping[str, str],
    canary_stage_parent: str | Path,
    filesystem_inspector: Callable[..., object] = production_filesystem_inspector,
    runtime_inspector: Callable[..., object] = production_runtime_inspector,
    canary_launcher: Callable[..., object] = production_public_canary_launcher,
    isolation_inspector: Callable[
        [], object
    ] = production_address_family_isolation_probe,
) -> tuple[dict[str, object], FreshComparatorRuntimeBinding]:
    """Inspect current runtime and execute one public no-network canary."""

    if (
        not isinstance(paths, FreshComparatorRuntimePaths)
        or not callable(filesystem_inspector)
        or not callable(runtime_inspector)
        or not callable(canary_launcher)
        or not callable(isolation_inspector)
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh comparator preflight interface drifted"
        )
    versions, roots = _validated_expected_maps(
        expected_package_versions, expected_module_import_roots
    )
    _assert_stage_outside_runtime_roots(paths, canary_stage_parent)
    try:
        isolation = _validated_address_family_isolation_probe(
            isolation_inspector()
        )
        filesystem = _validated_filesystem_binding(
            filesystem_inspector(paths)
        )
        runtime = _validated_runtime_probe(
            runtime_inspector(
                paths,
                package_names=tuple(sorted(versions)),
                module_names=tuple(sorted(roots)),
            ),
            expected_package_versions=versions,
            expected_module_import_roots=roots,
        )
    except MmqaP1OfficialHippoRAGBlockError:
        raise
    except Exception as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh comparator static/runtime inspection failed"
        ) from exc
    if runtime["address_family_isolation_probe"] != isolation:
        raise MmqaP1OfficialHippoRAGBlockError(
            "runtime inspector did not inherit the parent address-family restriction"
        )

    fixture = public_synthetic_candidate_payload()
    parent = _create_private_stage_parent(canary_stage_parent)
    canary_work_root = parent / "public-synthetic-candidate.work"
    try:
        launched = canary_launcher(
            payload=fixture,
            runtime_paths=paths,
            work_root=canary_work_root,
            timeout_seconds=ITEM_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh public synthetic canary failed; no retry permitted"
        ) from exc
    if os.path.lexists(canary_work_root):
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh public canary did not destroy its item-local work root"
        )
    top5 = _validated_logical_top5(
        launched,
        logical_count=len(fixture.logical_source_ordinals),
    )
    fixture_sha256 = hashlib.sha256(
        fixture.canonical_worker_bytes()
    ).hexdigest()
    output_sha256 = _semantic_hash(list(top5))
    body: dict[str, object] = {
        "schema": FRESH_PREFLIGHT_SCHEMA,
        "study_id": STUDY_ID,
        "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
        "status": "passed_public_synthetic_candidate_only_fresh_runtime",
        "path_binding_sha256": _semantic_hash(paths.path_binding()),
        "filesystem_binding": filesystem,
        "filesystem_binding_sha256": _semantic_hash(filesystem),
        "runtime_probe": runtime,
        "runtime_probe_sha256": _semantic_hash(runtime),
        "expected_package_versions": versions,
        "expected_module_import_roots": roots,
        "public_synthetic_fixture_sha256": fixture_sha256,
        "public_synthetic_output_sha256": output_sha256,
        "public_synthetic_candidate_count": len(
            fixture.logical_source_ordinals
        ),
        "public_synthetic_output_count": core.TOP_K,
        "official_core_index_called": True,
        "official_core_retrieve_called": True,
        "benchmark_rows_read": 0,
        "scores_computed": 0,
        "address_family_isolation_contract": dict(
            ADDRESS_FAMILY_ISOLATION_CONTRACT
        ),
        "address_family_isolation_contract_sha256": _semantic_hash(
            ADDRESS_FAMILY_ISOLATION_CONTRACT
        ),
        "address_family_isolation_probe": isolation,
        "address_family_isolation_probe_sha256": _semantic_hash(isolation),
        "worker_subprocess_inherits_parent_restriction": True,
        "bwrap_call_count": 0,
        "cuda_visible_devices": "",
        "cpu_thread_env": CPU_THREAD_ENV,
        "retry_count": 0,
    }
    receipt = {**body, "receipt_sha256": _semantic_hash(body)}
    binding = FreshComparatorRuntimeBinding(
        paths=paths,
        receipt_sha256=str(receipt["receipt_sha256"]),
        filesystem_binding_sha256=str(
            receipt["filesystem_binding_sha256"]
        ),
        runtime_probe_sha256=str(receipt["runtime_probe_sha256"]),
        address_family_isolation_probe_sha256=str(
            receipt["address_family_isolation_probe_sha256"]
        ),
        public_synthetic_output_sha256=output_sha256,
        _capability=_FRESH_BINDING_CAPABILITY,
    )
    return receipt, binding


def validate_fresh_preflight_receipt(
    value: Mapping[str, object],
    *,
    paths: FreshComparatorRuntimePaths,
    filesystem_inspector: Callable[..., object] | None = None,
    isolation_inspector: Callable[
        [], object
    ] = production_address_family_isolation_probe,
) -> FreshComparatorRuntimeBinding:
    if not isinstance(value, Mapping):
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh preflight receipt must be a mapping"
        )
    _exact_fields(value, _PREFLIGHT_FIELDS, "fresh preflight receipt")
    body = {key: value[key] for key in value if key != "receipt_sha256"}
    versions, roots = _validated_expected_maps(
        value.get("expected_package_versions", {}),  # type: ignore[arg-type]
        value.get("expected_module_import_roots", {}),  # type: ignore[arg-type]
    )
    filesystem = _validated_filesystem_binding(
        value.get("filesystem_binding")
    )
    runtime = _validated_runtime_probe(
        value.get("runtime_probe"),
        expected_package_versions=versions,
        expected_module_import_roots=roots,
    )
    isolation = _validated_address_family_isolation_probe(
        value.get("address_family_isolation_probe")
    )
    if not callable(isolation_inspector):
        raise MmqaP1OfficialHippoRAGBlockError(
            "address-family isolation revalidator is not callable"
        )
    if (
        value.get("schema") != FRESH_PREFLIGHT_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("study_design_self_sha256")
        != STUDY_DESIGN_SELF_SHA256
        or value.get("status")
        != "passed_public_synthetic_candidate_only_fresh_runtime"
        or value.get("path_binding_sha256")
        != _semantic_hash(paths.path_binding())
        or value.get("filesystem_binding_sha256")
        != _semantic_hash(filesystem)
        or value.get("runtime_probe_sha256") != _semantic_hash(runtime)
        or _HEX64.fullmatch(
            str(value.get("public_synthetic_fixture_sha256"))
        )
        is None
        or _HEX64.fullmatch(
            str(value.get("public_synthetic_output_sha256"))
        )
        is None
        or value.get("public_synthetic_candidate_count") != 6
        or value.get("public_synthetic_output_count") != core.TOP_K
        or value.get("official_core_index_called") is not True
        or value.get("official_core_retrieve_called") is not True
        or value.get("benchmark_rows_read") != 0
        or value.get("scores_computed") != 0
        or value.get("address_family_isolation_contract")
        != ADDRESS_FAMILY_ISOLATION_CONTRACT
        or value.get("address_family_isolation_contract_sha256")
        != _semantic_hash(ADDRESS_FAMILY_ISOLATION_CONTRACT)
        or value.get("address_family_isolation_probe_sha256")
        != _semantic_hash(isolation)
        or runtime.get("address_family_isolation_probe") != isolation
        or value.get("worker_subprocess_inherits_parent_restriction")
        is not True
        or value.get("bwrap_call_count") != 0
        or value.get("cuda_visible_devices") != ""
        or value.get("cpu_thread_env") != CPU_THREAD_ENV
        or value.get("retry_count") != 0
        or value.get("receipt_sha256") != _semantic_hash(body)
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh preflight receipt identity or policy drifted"
        )
    try:
        current_isolation = _validated_address_family_isolation_probe(
            isolation_inspector()
        )
    except Exception as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "effective address-family isolation revalidation failed"
        ) from exc
    if current_isolation != isolation:
        raise MmqaP1OfficialHippoRAGBlockError(
            "effective address-family isolation changed after canary"
        )
    if filesystem_inspector is not None:
        try:
            current = _validated_filesystem_binding(
                filesystem_inspector(paths)
            )
        except Exception as exc:
            raise MmqaP1OfficialHippoRAGBlockError(
                "fresh runtime filesystem revalidation failed"
            ) from exc
        if current != filesystem:
            raise MmqaP1OfficialHippoRAGBlockError(
                "fresh runtime filesystem changed after canary"
            )
    return FreshComparatorRuntimeBinding(
        paths=paths,
        receipt_sha256=str(value["receipt_sha256"]),
        filesystem_binding_sha256=str(
            value["filesystem_binding_sha256"]
        ),
        runtime_probe_sha256=str(value["runtime_probe_sha256"]),
        address_family_isolation_probe_sha256=str(
            value["address_family_isolation_probe_sha256"]
        ),
        public_synthetic_output_sha256=str(
            value["public_synthetic_output_sha256"]
        ),
        _capability=_FRESH_BINDING_CAPABILITY,
    )


def write_fresh_preflight_receipt(
    path: str | Path, receipt: Mapping[str, object]
) -> str:
    # Full validation needs its path binding and is performed before this
    # writer; persistence itself is deliberately generic and one-shot.
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema") != FRESH_PREFLIGHT_SCHEMA
        or receipt.get("receipt_sha256")
        != _semantic_hash(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        )
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh preflight receipt is not self-consistent"
        )
    destination = Path(path).expanduser().absolute()
    raw = _canonical_json_bytes(receipt, newline=True)
    descriptor = -1
    try:
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh preflight receipt exists or cannot be created"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (
        stat.S_IMODE(destination.stat().st_mode) != 0o600
        or destination.read_bytes() != raw
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh preflight receipt reopen or mode drifted"
        )
    return hashlib.sha256(raw).hexdigest()


def load_fresh_preflight_binding(
    path: str | Path,
    *,
    paths: FreshComparatorRuntimePaths,
    expected_receipt_sha256: str,
    filesystem_inspector: Callable[..., object] = production_filesystem_inspector,
    isolation_inspector: Callable[
        [], object
    ] = production_address_family_isolation_probe,
) -> FreshComparatorRuntimeBinding:
    """Load a canonical private receipt and re-hash every bound filesystem root."""

    expected = _sha256(
        expected_receipt_sha256, "expected fresh preflight receipt"
    )
    source = Path(path).expanduser().absolute()
    try:
        metadata = source.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise MmqaP1OfficialHippoRAGBlockError(
                "fresh preflight receipt is not a regular mode-0600 file"
            )
        raw = source.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except MmqaP1OfficialHippoRAGBlockError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh preflight receipt cannot be read"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or raw != _canonical_json_bytes(value, newline=True)
        or value.get("receipt_sha256") != expected
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "fresh preflight receipt is noncanonical or unexpected"
        )
    return validate_fresh_preflight_receipt(
        value,
        paths=paths,
        filesystem_inspector=filesystem_inspector,
        isolation_inspector=isolation_inspector,
    )


@dataclass(frozen=True)
class OfficialTerminalRow:
    work_id: str
    top5_source_ordinals: tuple[int, ...]
    candidate_payload_sha256: str
    closure_ordinal_bytes_sha256: str
    worker_output_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "work_id", _work_id(self.work_id))
        top5 = tuple(self.top5_source_ordinals)
        if (
            len(top5) != core.TOP_K
            or len(set(top5)) != core.TOP_K
            or any(type(value) is not int or value < 0 for value in top5)
        ):
            raise MmqaP1OfficialHippoRAGBlockError(
                "official terminal is not five unique source ordinals"
            )
        for value, field in (
            (self.candidate_payload_sha256, "candidate payload identity"),
            (
                self.closure_ordinal_bytes_sha256,
                "closure ordinal bytes identity",
            ),
            (self.worker_output_sha256, "worker output identity"),
        ):
            _sha256(value, field)
        object.__setattr__(self, "top5_source_ordinals", top5)

    def payload(self) -> dict[str, object]:
        return {
            "work_id": self.work_id,
            "top5_source_ordinals": list(self.top5_source_ordinals),
            "candidate_payload_sha256": self.candidate_payload_sha256,
            "closure_ordinal_bytes_sha256": (
                self.closure_ordinal_bytes_sha256
            ),
            "worker_output_sha256": self.worker_output_sha256,
        }


@dataclass(frozen=True)
class OfficialTerminalArchive:
    runtime_binding_sha256: str
    address_family_isolation_probe_sha256: str
    A_hold_input_sha256: str
    rows: tuple[OfficialTerminalRow, ...]

    def __post_init__(self) -> None:
        _sha256(self.runtime_binding_sha256, "runtime binding identity")
        _sha256(
            self.address_family_isolation_probe_sha256,
            "address-family isolation probe identity",
        )
        _sha256(self.A_hold_input_sha256, "A_hold input identity")
        rows = tuple(self.rows)
        if (
            len(rows) != A_HOLD_ITEM_COUNT
            or not all(isinstance(row, OfficialTerminalRow) for row in rows)
            or len({row.work_id for row in rows}) != len(rows)
        ):
            raise MmqaP1OfficialHippoRAGBlockError(
                "official terminal archive rows drifted"
            )
        object.__setattr__(self, "rows", rows)

    def body(self) -> dict[str, object]:
        return {
            "schema": TERMINAL_ARCHIVE_SCHEMA,
            "study_id": STUDY_ID,
            "study_design_self_sha256": STUDY_DESIGN_SELF_SHA256,
            "block": BLOCK,
            "item_count": A_HOLD_ITEM_COUNT,
            "max_workers": MAX_WORKERS,
            "item_timeout_seconds": ITEM_TIMEOUT_SECONDS,
            "execution_device_disposition": EXECUTION_DEVICE_DISPOSITION,
            "attested_cuda_visible_devices": ATTESTED_CUDA_VISIBLE_DEVICES,
            "cpu_thread_env_sha256": _semantic_hash(CPU_THREAD_ENV),
            "design_two_per_gpu_applied": DESIGN_TWO_PER_GPU_APPLIED,
            "design_concurrency_conflict_requires_unified_disposition": (
                DESIGN_CONCURRENCY_CONFLICT_REQUIRES_UNIFIED_DISPOSITION
            ),
            "network_isolation_policy": NETWORK_ISOLATION_POLICY,
            "address_family_isolation_contract_sha256": _semantic_hash(
                ADDRESS_FAMILY_ISOLATION_CONTRACT
            ),
            "address_family_isolation_probe_sha256": (
                self.address_family_isolation_probe_sha256
            ),
            "index_lifecycle_policy": INDEX_LIFECYCLE_POLICY,
            "retrieval_policy": RETRIEVAL_POLICY,
            "runtime_binding_sha256": self.runtime_binding_sha256,
            "A_hold_input_sha256": self.A_hold_input_sha256,
            "item_launcher_call_count": A_HOLD_ITEM_COUNT,
            "fresh_isolated_index_count": A_HOLD_ITEM_COUNT,
            "parent_address_family_restriction_inherited_count": (
                A_HOLD_ITEM_COUNT
            ),
            "bwrap_call_count": 0,
            "retry_replay_resample_count": 0,
            "online_evaluator_call_count": 0,
            "rows": [row.payload() for row in self.rows],
        }

    def payload(self) -> dict[str, object]:
        body = self.body()
        return {**body, "archive_sha256": _semantic_hash(body)}

    @property
    def archive_sha256(self) -> str:
        return str(self.payload()["archive_sha256"])


def _validated_logical_top5(
    value: object, *, logical_count: int
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise MmqaP1OfficialHippoRAGBlockError(
            "official launcher result must be ordinal-only"
        )
    try:
        top5 = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "official launcher result must be ordinal-only"
        ) from exc
    if (
        len(top5) != core.TOP_K
        or len(set(top5)) != core.TOP_K
        or any(
            type(ordinal) is not int or not 0 <= ordinal < logical_count
            for ordinal in top5
        )
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "official launcher result escaped its item-local corpus"
        )
    return top5


def production_item_launcher(
    *,
    payload: action_executor.CandidateRestrictedHippoRAGPayload,
    runtime_binding: FreshComparatorRuntimeBinding,
    work_root: Path,
    timeout_seconds: int,
) -> tuple[int, ...]:
    """Call fresh-bound official worker semantics with no legacy v3 receipt."""

    if (
        not isinstance(
            payload,
            action_executor.CandidateRestrictedHippoRAGPayload,
        )
        or not isinstance(runtime_binding, FreshComparatorRuntimeBinding)
        or timeout_seconds != ITEM_TIMEOUT_SECONDS
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "production item launcher contract drifted"
        )
    isolation = production_address_family_isolation_probe()
    if (
        _semantic_hash(isolation)
        != runtime_binding.address_family_isolation_probe_sha256
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "official item launcher address-family isolation drifted"
        )
    try:
        result = run_fresh_bound_item(
            payload=payload,
            runtime_paths=runtime_binding.paths,
            work_root=work_root,
            timeout_seconds=ITEM_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "official item-local adapter failed; no retry permitted"
        ) from exc
    return _validated_logical_top5(
        result,
        logical_count=len(payload.logical_source_ordinals),
    )


def _run_one(
    item: AHoldHippoItem,
    *,
    runtime_binding: FreshComparatorRuntimeBinding,
    work_root: Path,
    launcher: Callable[..., object],
) -> OfficialTerminalRow:
    if os.path.lexists(work_root):
        raise MmqaP1OfficialHippoRAGBlockError(
            "per-item official work root is not fresh"
        )
    try:
        launched = launcher(
            payload=item.payload,
            runtime_binding=runtime_binding,
            work_root=work_root,
            timeout_seconds=ITEM_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "official per-item launch failed; no retry permitted"
        ) from exc
    if os.path.lexists(work_root):
        raise MmqaP1OfficialHippoRAGBlockError(
            "official adapter did not destroy its item-local work root"
        )
    logical_top5 = _validated_logical_top5(
        launched,
        logical_count=len(item.payload.logical_source_ordinals),
    )
    raw = official_contract.canonical_json_bytes(list(logical_top5))
    try:
        terminal = (
            action_executor.parse_candidate_restricted_hipporag_terminal(
                item.payload, raw
            )
        )
    except action_executor.MmqaP1LocalActionExecutorError as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "official terminal mapping failed"
        ) from exc
    return OfficialTerminalRow(
        work_id=item.work_id,
        top5_source_ordinals=terminal.top5_source_ordinals,
        candidate_payload_sha256=item.candidate_payload_sha256,
        closure_ordinal_bytes_sha256=(
            terminal.closure_ordinal_bytes_sha256
        ),
        worker_output_sha256=terminal.worker_output_sha256,
    )


def _create_private_stage_parent(path: str | Path) -> Path:
    root = Path(path).expanduser().absolute()
    if os.path.lexists(root):
        raise MmqaP1OfficialHippoRAGBlockError(
            "A_hold official stage parent already exists"
        )
    if root.parent.is_symlink() or not root.parent.is_dir():
        raise MmqaP1OfficialHippoRAGBlockError(
            "A_hold official stage parent has an unsafe parent"
        )
    try:
        os.mkdir(root, 0o700)
    except OSError as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "A_hold official stage parent cannot be created"
        ) from exc
    metadata = root.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "A_hold official stage parent is not mode 0700"
        )
    return root


def run_ahold_official_hipporag_block(
    items: Sequence[AHoldHippoItem],
    *,
    runtime_binding: FreshComparatorRuntimeBinding,
    stage_parent: str | Path,
    item_launcher: Callable[..., object] = production_item_launcher,
    _executor_factory: Callable[..., object] = ThreadPoolExecutor,
    _isolation_inspector: Callable[
        [], object
    ] = production_address_family_isolation_probe,
) -> OfficialTerminalArchive:
    """Run the complete A_hold block once in one fixed four-worker CPU pool."""

    checked = validate_ahold_items(items)
    if not isinstance(runtime_binding, FreshComparatorRuntimeBinding):
        raise MmqaP1OfficialHippoRAGBlockError(
            "official block requires a frozen runtime binding"
        )
    if (
        not callable(item_launcher)
        or not callable(_executor_factory)
        or not callable(_isolation_inspector)
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "official launcher, executor, or isolation probe is not callable"
        )
    isolation = _validated_address_family_isolation_probe(
        _isolation_inspector()
    )
    if (
        _semantic_hash(isolation)
        != runtime_binding.address_family_isolation_probe_sha256
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "A_hold address-family isolation drifted after preflight"
        )
    _assert_stage_outside_runtime_roots(
        runtime_binding.paths, stage_parent
    )
    root = _create_private_stage_parent(stage_parent)
    futures: dict[Future[object], int] = {}
    ordered: list[OfficialTerminalRow | None] = [None] * len(checked)
    try:
        pool_value = _executor_factory(
            max_workers=MAX_WORKERS,
            thread_name_prefix="mmqa-p1-official-cpu",
        )
        with pool_value as pool:
            for index, item in enumerate(checked):
                work_root = root / f"{item.work_id}.work"
                future = pool.submit(
                    _run_one,
                    item,
                    runtime_binding=runtime_binding,
                    work_root=work_root,
                    launcher=item_launcher,
                )
                futures[future] = index
            try:
                for future in as_completed(futures):
                    index = futures[future]
                    result = future.result()
                    if not isinstance(result, OfficialTerminalRow):
                        raise MmqaP1OfficialHippoRAGBlockError(
                            "official future returned a malformed terminal"
                        )
                    ordered[index] = result
            except Exception as exc:
                for future in futures:
                    future.cancel()
                if isinstance(exc, MmqaP1OfficialHippoRAGBlockError):
                    raise
                raise MmqaP1OfficialHippoRAGBlockError(
                    "A_hold official block failed; no retry permitted"
                ) from exc
    except MmqaP1OfficialHippoRAGBlockError:
        raise
    except Exception as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "A_hold official pool failed; no retry permitted"
        ) from exc
    if any(row is None for row in ordered):
        raise MmqaP1OfficialHippoRAGBlockError(
            "A_hold official block terminal is incomplete"
        )
    archive = OfficialTerminalArchive(
        runtime_binding_sha256=runtime_binding.binding_sha256,
        address_family_isolation_probe_sha256=(
            runtime_binding.address_family_isolation_probe_sha256
        ),
        A_hold_input_sha256=ahold_input_sha256(checked),
        rows=tuple(row for row in ordered if row is not None),
    )
    return validate_terminal_archive_for_items(archive, checked)


def validate_terminal_archive_for_items(
    archive: OfficialTerminalArchive,
    items: Sequence[AHoldHippoItem],
) -> OfficialTerminalArchive:
    checked = validate_ahold_items(items)
    if not isinstance(archive, OfficialTerminalArchive):
        raise MmqaP1OfficialHippoRAGBlockError(
            "official terminal archive type drifted"
        )
    expected = tuple(
        (
            item.work_id,
            item.candidate_payload_sha256,
            item.payload.closure_ordinal_bytes_sha256,
        )
        for item in checked
    )
    observed = tuple(
        (
            row.work_id,
            row.candidate_payload_sha256,
            row.closure_ordinal_bytes_sha256,
        )
        for row in archive.rows
    )
    by_work_id = {item.work_id: item for item in checked}
    if (
        observed != expected
        or len(set(observed)) != len(observed)
        or archive.A_hold_input_sha256 != ahold_input_sha256(checked)
        or any(
            not set(row.top5_source_ordinals).issubset(
                by_work_id[row.work_id].payload.logical_source_ordinals
            )
            for row in archive.rows
        )
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "official terminal rows are missing, duplicate, reordered, "
            "cross-item, or outside the common closure"
        )
    return archive


def parse_terminal_archive_payload(
    value: Mapping[str, object],
) -> OfficialTerminalArchive:
    if not isinstance(value, Mapping):
        raise MmqaP1OfficialHippoRAGBlockError(
            "official terminal archive must be a mapping"
        )
    _exact_fields(value, _ARCHIVE_FIELDS, "official terminal archive")
    body = {key: value[key] for key in value if key != "archive_sha256"}
    if (
        value.get("schema") != TERMINAL_ARCHIVE_SCHEMA
        or value.get("study_id") != STUDY_ID
        or value.get("study_design_self_sha256")
        != STUDY_DESIGN_SELF_SHA256
        or value.get("block") != BLOCK
        or value.get("item_count") != A_HOLD_ITEM_COUNT
        or value.get("max_workers") != MAX_WORKERS
        or value.get("item_timeout_seconds") != ITEM_TIMEOUT_SECONDS
        or value.get("execution_device_disposition")
        != EXECUTION_DEVICE_DISPOSITION
        or value.get("attested_cuda_visible_devices") != ""
        or value.get("cpu_thread_env_sha256")
        != _semantic_hash(CPU_THREAD_ENV)
        or value.get("design_two_per_gpu_applied") is not False
        or value.get(
            "design_concurrency_conflict_requires_unified_disposition"
        )
        is not True
        or value.get("network_isolation_policy")
        != NETWORK_ISOLATION_POLICY
        or value.get("address_family_isolation_contract_sha256")
        != _semantic_hash(ADDRESS_FAMILY_ISOLATION_CONTRACT)
        or _HEX64.fullmatch(
            str(value.get("address_family_isolation_probe_sha256"))
        )
        is None
        or value.get("index_lifecycle_policy") != INDEX_LIFECYCLE_POLICY
        or value.get("retrieval_policy") != RETRIEVAL_POLICY
        or value.get("item_launcher_call_count") != A_HOLD_ITEM_COUNT
        or value.get("fresh_isolated_index_count") != A_HOLD_ITEM_COUNT
        or value.get("parent_address_family_restriction_inherited_count")
        != A_HOLD_ITEM_COUNT
        or value.get("bwrap_call_count") != 0
        or value.get("retry_replay_resample_count") != 0
        or value.get("online_evaluator_call_count") != 0
        or value.get("archive_sha256") != _semantic_hash(body)
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "official terminal archive identity or policy drifted"
        )
    raw_rows = value.get("rows")
    if not isinstance(raw_rows, list):
        raise MmqaP1OfficialHippoRAGBlockError(
            "official terminal rows must be an array"
        )
    rows = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise MmqaP1OfficialHippoRAGBlockError(
                "official terminal row must be a mapping"
            )
        _exact_fields(raw, _ROW_FIELDS, "official terminal row")
        top5 = raw.get("top5_source_ordinals")
        if not isinstance(top5, list):
            raise MmqaP1OfficialHippoRAGBlockError(
                "official terminal top five must be an array"
            )
        rows.append(
            OfficialTerminalRow(
                work_id=_work_id(raw.get("work_id")),
                top5_source_ordinals=tuple(
                    _strict_int(value, "top-five source ordinal")
                    for value in top5
                ),
                candidate_payload_sha256=_sha256(
                    raw.get("candidate_payload_sha256"),
                    "candidate payload identity",
                ),
                closure_ordinal_bytes_sha256=_sha256(
                    raw.get("closure_ordinal_bytes_sha256"),
                    "closure ordinal bytes identity",
                ),
                worker_output_sha256=_sha256(
                    raw.get("worker_output_sha256"),
                    "worker output identity",
                ),
            )
        )
    return OfficialTerminalArchive(
        runtime_binding_sha256=_sha256(
            value.get("runtime_binding_sha256"),
            "runtime binding identity",
        ),
        address_family_isolation_probe_sha256=_sha256(
            value.get("address_family_isolation_probe_sha256"),
            "address-family isolation probe identity",
        ),
        A_hold_input_sha256=_sha256(
            value.get("A_hold_input_sha256"), "A_hold input identity"
        ),
        rows=tuple(rows),
    )


def write_private_terminal_archive(
    path: str | Path,
    archive: OfficialTerminalArchive,
) -> str:
    if not isinstance(archive, OfficialTerminalArchive):
        raise MmqaP1OfficialHippoRAGBlockError(
            "terminal writer requires a validated archive"
        )
    destination = Path(path).expanduser().absolute()
    raw = _canonical_json_bytes(archive.payload(), newline=True)
    descriptor = -1
    try:
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "private terminal archive exists or cannot be created"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    observed = destination.lstat()
    if (
        not stat.S_ISREG(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o600
        or destination.read_bytes() != raw
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "private terminal archive reopen or mode drifted"
        )
    return hashlib.sha256(raw).hexdigest()


def load_private_terminal_archive(
    path: str | Path,
) -> OfficialTerminalArchive:
    source = Path(path).expanduser().absolute()
    try:
        metadata = source.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise MmqaP1OfficialHippoRAGBlockError(
                "private terminal archive is not a regular mode-0600 file"
            )
        raw = source.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except MmqaP1OfficialHippoRAGBlockError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            "private terminal archive cannot be read"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or raw != _canonical_json_bytes(value, newline=True)
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "private terminal archive is not canonical JSON"
        )
    return parse_terminal_archive_payload(value)


def _json_mapping_argument(raw: str, *, field: str) -> dict[str, str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MmqaP1OfficialHippoRAGBlockError(
            f"{field} is not valid inline JSON"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or not value
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(item, str)
            or not item
            for key, item in value.items()
        )
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            f"{field} must be a nonempty string mapping"
        )
    return dict(value)


def _preflight_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the one-shot MMQA P1 official-HippoRAG public canary "
            "receipt inside an AF_UNIX-only systemd service."
        )
    )
    parser.add_argument("--runtime-python", required=True)
    parser.add_argument("--pyvenv-cfg", required=True)
    parser.add_argument("--overlay-root", required=True)
    parser.add_argument("--hipporag-source-root", required=True)
    parser.add_argument("--p16-site-root", required=True)
    parser.add_argument("--local-llm-model", required=True)
    parser.add_argument("--local-embedding-model", required=True)
    parser.add_argument("--expected-package-versions-json", required=True)
    parser.add_argument("--expected-module-import-roots-json", required=True)
    parser.add_argument("--canary-stage-parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _preflight_main(
    argv: Sequence[str] | None = None,
    *,
    builder: Callable[..., object] = build_fresh_comparator_preflight,
    writer: Callable[..., object] = write_fresh_preflight_receipt,
) -> int:
    arguments = _preflight_parser().parse_args(argv)
    if not callable(builder) or not callable(writer):
        raise MmqaP1OfficialHippoRAGBlockError(
            "preflight CLI dependency drifted"
        )
    paths = FreshComparatorRuntimePaths(
        runtime_python=arguments.runtime_python,
        pyvenv_cfg=arguments.pyvenv_cfg,
        overlay_root=arguments.overlay_root,
        hipporag_source_root=arguments.hipporag_source_root,
        p16_site_root=arguments.p16_site_root,
        local_llm_model=arguments.local_llm_model,
        local_embedding_model=arguments.local_embedding_model,
    )
    built = builder(
        paths=paths,
        expected_package_versions=_json_mapping_argument(
            arguments.expected_package_versions_json,
            field="expected package versions",
        ),
        expected_module_import_roots=_json_mapping_argument(
            arguments.expected_module_import_roots_json,
            field="expected module import roots",
        ),
        canary_stage_parent=arguments.canary_stage_parent,
    )
    if (
        not isinstance(built, tuple)
        or len(built) != 2
        or not isinstance(built[0], Mapping)
        or not isinstance(built[1], FreshComparatorRuntimeBinding)
    ):
        raise MmqaP1OfficialHippoRAGBlockError(
            "preflight CLI builder terminal drifted"
        )
    receipt = dict(built[0])
    file_sha256 = writer(arguments.output, receipt)
    _sha256(file_sha256, "preflight receipt file identity")
    safe_body = {
        "schema": f"{VERSION}_preflight_cli_safe_receipt_v1",
        "study_id": STUDY_ID,
        "status": "fresh_comparator_preflight_written_once",
        "preflight_schema": FRESH_PREFLIGHT_SCHEMA,
        "preflight_receipt_sha256": _sha256(
            receipt.get("receipt_sha256"), "preflight receipt identity"
        ),
        "preflight_receipt_file_sha256": file_sha256,
        "address_family_isolation_contract_sha256": _semantic_hash(
            ADDRESS_FAMILY_ISOLATION_CONTRACT
        ),
        "address_family_isolation_probe_sha256": _sha256(
            receipt.get("address_family_isolation_probe_sha256"),
            "address-family isolation probe identity",
        ),
        "formal_source_read_count": 0,
        "benchmark_rows_read": 0,
        "scores_computed": 0,
        "bwrap_call_count": 0,
    }
    safe = {**safe_body, "self_sha256": _semantic_hash(safe_body)}
    print(_canonical_json_bytes(safe).decode("ascii"))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return _preflight_main(argv)


__all__ = [
    "VERSION",
    "STUDY_ID",
    "STUDY_DESIGN_SELF_SHA256",
    "BLOCK",
    "A_HOLD_ITEM_COUNT",
    "MAX_WORKERS",
    "ITEM_TIMEOUT_SECONDS",
    "ATTESTED_CUDA_VISIBLE_DEVICES",
    "EXECUTION_DEVICE_DISPOSITION",
    "DESIGN_TWO_PER_GPU_APPLIED",
    "DESIGN_CONCURRENCY_CONFLICT_REQUIRES_UNIFIED_DISPOSITION",
    "NETWORK_ISOLATION_POLICY",
    "NETWORK_NAMESPACE_POLICY",
    "ADDRESS_FAMILY_ISOLATION_PROBE_SCHEMA",
    "ADDRESS_FAMILY_ISOLATION_CONTRACT",
    "INDEX_LIFECYCLE_POLICY",
    "RETRIEVAL_POLICY",
    "CPU_THREAD_ENV",
    "FRESH_PREFLIGHT_SCHEMA",
    "ERASER_ADAPTER_FILE_SHA256",
    "ERASER_WORKER_FILE_SHA256",
    "ERASER_CONTRACT_FILE_SHA256",
    "TERMINAL_ARCHIVE_SCHEMA",
    "MmqaP1OfficialHippoRAGBlockError",
    "AHoldHippoItem",
    "FreshComparatorRuntimePaths",
    "FreshComparatorRuntimeBinding",
    "OfficialTerminalRow",
    "OfficialTerminalArchive",
    "validate_ahold_items",
    "ahold_input_sha256",
    "production_filesystem_inspector",
    "production_address_family_isolation_probe",
    "production_runtime_inspector",
    "public_synthetic_candidate_payload",
    "run_fresh_bound_item",
    "production_public_canary_launcher",
    "build_fresh_comparator_preflight",
    "validate_fresh_preflight_receipt",
    "write_fresh_preflight_receipt",
    "load_fresh_preflight_binding",
    "production_item_launcher",
    "run_ahold_official_hipporag_block",
    "validate_terminal_archive_for_items",
    "parse_terminal_archive_payload",
    "write_private_terminal_archive",
    "load_private_terminal_archive",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
