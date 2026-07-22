#!/usr/bin/env python3
"""Freeze the pre-identity P16 wired-host runtime and implementation bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence

_SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(_SCRIPT_PROJECT_ROOT),
    str(_SCRIPT_PROJECT_ROOT / "reconstruction_v2"),
]

from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p14_acquisition_v1 as p14_acquisition,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p16_all_remote_c_confirm_v1 as p16,
)
from reconstruction_v2.assumption_agent.benchmarks import (
    bright_p16_extension_acquisition_v1 as acquisition,
)


FORMAL_IMPLEMENTATION_COMMIT = "477531767460006135468311f8a8dc0b9dd39ea0"
REMOTE_HOST = "jtl311linux"
REMOTE_ROOT = Path("/home/erzhu419/p16_all_remote_20260722")
REMOTE_BASE = REMOTE_ROOT / "runtime/reconstruction_v2"
REMOTE_PYTHON = (
    REMOTE_BASE
    / "artifacts/bright_reasoning_retrieval_runtime_v1/hipporag_venv/bin/python"
)
FINGERPRINT_PATH = Path("manifests/bright_p16_remote_runtime_fingerprint_v1.json")
ACQUISITION_FREEZE_PATH = Path(
    "manifests/bright_p16_extension_acquisition_freeze_v1.json"
)
IMPLEMENTATION_FREEZE_PATH = Path(
    "manifests/bright_p16_all_remote_implementation_freeze_v1.json"
)

DEPENDENCIES = (
    "assumption_agent/benchmarks/bright_p16_extension_acquisition_v1.py",
    "assumption_agent/benchmarks/bright_p14_direct_c_confirm_v1.py",
    "assumption_agent/benchmarks/bright_p14_acquisition_v1.py",
    "assumption_agent/benchmarks/bright_p14_source_qualification_v1.py",
    "assumption_agent/benchmarks/bright_reasoning_retrieval_core_v1.py",
    "assumption_agent/benchmarks/bright_reasoning_retrieval_study_v1.py",
    "assumption_agent/benchmarks/bright_bridge_expansion_core_v1.py",
    "assumption_agent/benchmarks/fiqa_bridge_expansion_train_runtime_v1.py",
    "assumption_agent/benchmarks/hipporag_upstream_hardening_qualification_v1.py",
    "assumption_agent/benchmarks/nanobeir_p11_acquisition_v1.py",
    "assumption_agent/benchmarks/nanobeir_p11_c_confirm_runtime_v1.py",
    "assumption_agent/benchmarks/nanobeir_p12_acquisition_v1.py",
    "assumption_agent/benchmarks/nanobeir_p12_c_confirm_runtime_v1.py",
    "assumption_agent/benchmarks/nanobeir_p13_bridge_safe_candidate_v1.py",
    "assumption_agent/benchmarks/p11_raw_ce_rrf_v1.py",
    "replication_runtime/bright_query_generator_v1/contract.py",
    "replication_runtime/bright_query_generator_v1/worker.py",
    "replication_runtime/bridge_expanded_cross_encoder_v1/contract.py",
    "replication_runtime/bridge_expanded_cross_encoder_v1/worker.py",
    "replication_runtime/bright_official_hipporag_v1/contract.py",
    "replication_runtime/bright_official_hipporag_v1/worker.py",
    "replication_runtime/qasper_minilm_v1/binding.py",
    "replication_runtime/hipporag_upstream_hardening_v1/backport.py",
)

IMPLEMENTATIONS = (
    p16.IMPLEMENTATION_RELATIVE.as_posix(),
    p16.MINILM_ENCODER_RELATIVE.as_posix(),
    p16.RUNNER_RELATIVE.as_posix(),
    p16.TEST_RELATIVE.as_posix(),
)


class P16FreezeError(RuntimeError):
    """The fresh remote runtime is not safe to freeze."""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(value: Any) -> str:
    return p14_acquisition.utilities.stable_hash(value)


def _manifest_file_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        p14_acquisition.utilities.canonical_json_bytes(value)
    ).hexdigest()


def _self_hashed(value: Mapping[str, Any], field: str = "self_sha256") -> dict[str, Any]:
    return p14_acquisition.utilities.self_hashed(value, field=field)


def _binding(base: Path, relative: str) -> dict[str, str]:
    path = base / relative
    if path.is_symlink() or not path.is_file():
        raise P16FreezeError(f"binding is unavailable: {relative}")
    return {"relative_path": relative, "sha256": _file_sha256(path)}


def _run(command: Sequence[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _assert_commit_bindings(project_root: Path, base: Path, relatives: Sequence[str]) -> None:
    if _run(["git", "merge-base", "--is-ancestor", FORMAL_IMPLEMENTATION_COMMIT, "HEAD"], cwd=project_root) != "":
        raise P16FreezeError("unexpected git merge-base output")
    for relative in relatives:
        committed = subprocess.run(
            [
                "git",
                "show",
                f"{FORMAL_IMPLEMENTATION_COMMIT}:reconstruction_v2/{relative}",
            ],
            cwd=project_root,
            check=True,
            capture_output=True,
        ).stdout
        if hashlib.sha256(committed).hexdigest() != _file_sha256(base / relative):
            raise P16FreezeError(f"formal implementation drifted: {relative}")


def _remote_facts() -> Mapping[str, Any]:
    code = r'''import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

from reconstruction_v2.assumption_agent.benchmarks import bright_p14_source_qualification_v1 as source
from reconstruction_v2.assumption_agent.benchmarks import nanobeir_p12_acquisition_v1 as utilities
from reconstruction_v2.replication_runtime.bright_p16_all_remote_v1.runner import _frozen_asset_receipts, _network_audit, _runtime_inventory_receipt

root = Path("/home/erzhu419/p16_all_remote_20260722")
base = root / "runtime/reconstruction_v2"
preflight = root / "preflight"

for forbidden in (
    base / "manifests/bright_p16_extension_acquisition_result_v1.json",
    base / "artifacts/bright_p16_extension_acquisition_v1",
    root / "work",
):
    if forbidden.exists() or forbidden.is_symlink():
        raise RuntimeError("P16 item identity or formal work already exists")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def elapsed(run):
    return (run / "output.json").stat().st_mtime - (run / "input.json").stat().st_mtime

def model_run(run, trace="network.trace"):
    return {
        "elapsed_seconds_from_input_to_output_mtime": elapsed(run),
        "input_file_sha256": sha(run / "input.json"),
        "network_audit": _network_audit(run, trace),
        "output_file_sha256": sha(run / "output.json"),
    }

qwen1 = preflight / "qwen1"
qwen2 = preflight / "qwen2"
cross1 = preflight / "cross/canonical1"
cross2 = preflight / "cross/canonical2"
minilm = preflight / "minilm4"
hippo = preflight / "hippo/run4"
if not all((path / "output.json").is_file() for path in (qwen1, qwen2, cross1, cross2, hippo)):
    raise RuntimeError("a final worker canary is incomplete")
if not (minilm / "receipt.json").is_file():
    raise RuntimeError("the final MiniLM canary is incomplete")

q1 = model_run(qwen1)
q2 = model_run(qwen2)
c1 = model_run(cross1)
c2 = model_run(cross2)
if (qwen1 / "output.json").read_bytes() != (qwen2 / "output.json").read_bytes():
    raise RuntimeError("Qwen canary is not repeat exact")
if (cross1 / "output.json").read_bytes() != (cross2 / "output.json").read_bytes():
    raise RuntimeError("cross-encoder canary is not repeat exact")

mini = json.loads((minilm / "receipt.json").read_text(encoding="ascii"))
mini_audit = _network_audit(minilm, "network.trace")
if mini["canary_receipt"].get("repeat_exact") is not True:
    raise RuntimeError("MiniLM canary is not repeat exact")

hippo_payload = json.loads((hippo / "output.json").read_text(encoding="ascii"))
hippo_audit = _network_audit(hippo, "network.trace")
service = dict(
    line.split("=", 1)
    for line in subprocess.run(
        [
            "systemctl", "--user", "show", "p16-hippo-preflight-v4.service",
            "-p", "ActiveState", "-p", "Result", "-p", "ExecMainStatus",
            "--no-pager",
        ],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    if "=" in line
)
if (
    hippo_payload.get("schema") != "bright_official_hipporag_candidate_retrieval_v1_output"
    or len(hippo_payload.get("top_ordinals", [])) != 10
    or hippo_payload.get("graph_node_count", 0) <= 32
    or hippo_payload.get("graph_edge_count", 0) <= 0
    or service.get("ActiveState") != "inactive"
    or service.get("Result") != "success"
    or service.get("ExecMainStatus") != "0"
):
    raise RuntimeError("HippoRAG canary is nonterminal")

audits = [q1["network_audit"], q2["network_audit"], c1["network_audit"], c2["network_audit"], mini_audit, hippo_audit]
if any(audit["denied_external_network_syscall_count"] for audit in audits):
    raise RuntimeError("an external network syscall was attempted")

lscpu = json.loads(subprocess.run(["lscpu", "-J"], check=True, capture_output=True, text=True).stdout)
cpu = {row["field"].rstrip(":"): row["data"] for row in lscpu["lscpu"]}
flags = set(cpu["Flags"].split())
gpu_rows = []
driver = None
for line in subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv,noheader,nounits"],
    check=True, capture_output=True, text=True,
).stdout.splitlines():
    index, name, uuid, memory, live_driver = [part.strip() for part in line.split(",")]
    driver = live_driver if driver is None else driver
    if live_driver != driver:
        raise RuntimeError("GPU driver versions disagree")
    gpu_rows.append({"UUID": uuid, "index": int(index), "memory_total_MiB": int(memory), "name": name})

python_path = base / "artifacts/bright_reasoning_retrieval_runtime_v1/hipporag_venv/bin/python"
python_target = python_path.resolve(strict=True)
strace = Path("/usr/bin/strace")
systemd_run = Path("/usr/bin/systemd-run")

facts = {
    "frozen_asset_receipts": _frozen_asset_receipts(base),
    "hardware": {
        "GPUs": gpu_rows,
        "NVIDIA_driver_version": driver,
        "architecture": platform.machine(),
        "cpu": {
            "core_count": int(cpu["Core(s) per socket"]) * int(cpu["Socket(s)"]),
            "logical_CPU_count": int(cpu["CPU(s)"]),
            "model": cpu["Model name"],
            "required_instruction_flags": [flag for flag in ("avx2", "avx512f") if flag in flags],
            "socket_count": int(cpu["Socket(s)"]),
            "threads_per_core": int(cpu["Thread(s) per core"]),
        },
        "glibc_version": " ".join(platform.libc_ver()).strip(),
        "kernel": subprocess.run(["uname", "-a"], check=True, capture_output=True, text=True).stdout.strip(),
        "memory_total_bytes": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
    },
    "minilm_canary_receipt": mini["canary_receipt"],
    "minilm_runtime_receipt": mini["runtime_receipt"],
    "network_isolation": {
        "controller": "transient_systemd_user_service_with_IPAddressDeny_any",
        "external_network_call_count_allowed": 0,
        "model_environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "strace": {
            "command": "/usr/bin/strace -ff -e trace=network -e inject=connect,sendto,sendmsg,sendmmsg:error=EPERM",
            "sha256": sha(strace),
            "size_bytes": strace.stat().st_size,
            "version": subprocess.run([str(strace), "-V"], check=True, capture_output=True, text=True).stdout.splitlines()[0],
        },
        "systemd_run": {
            "sha256": sha(systemd_run),
            "size_bytes": systemd_run.stat().st_size,
            "version": subprocess.run([str(systemd_run), "--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0],
        },
    },
    "python_executable": {
        "path": str(python_path),
        "resolved_target_sha256": sha(python_target),
        "resolved_target_size_bytes": python_target.stat().st_size,
    },
    "runtime_inventory_receipt": _runtime_inventory_receipt(),
    "source_document_bindings": {
        family: {
            "relative_path": f"{source.SOURCE_ROOT_RELATIVE.as_posix()}/documents/{source.SLUGS[family]}-00000-of-00001.parquet",
            **source.SOURCE_FILES[f"documents/{source.SLUGS[family]}-00000-of-00001.parquet"],
        }
        for family in source.FAMILIES
    },
    "worker_canaries": {
        "HippoRAG_CPU": {
            "candidate_document_count": 32,
            "elapsed_seconds_from_input_to_output_mtime": elapsed(hippo),
            "graph_edge_count": hippo_payload["graph_edge_count"],
            "graph_node_count": hippo_payload["graph_node_count"],
            "input_file_sha256": sha(hippo / "input.json"),
            "network_audit": hippo_audit,
            "output_file_sha256": sha(hippo / "output.json"),
            "service_result": "success",
            "terminal": True,
            "top_ordinal_count": len(hippo_payload["top_ordinals"]),
            "top_ordinals_sha256": utilities.stable_hash(hippo_payload["top_ordinals"]),
            "visible_GPU": "",
        },
        "MiniLM_GPU0": {"network_audit": mini_audit},
        "Qwen_GPU0": {"first": q1, "output_repeat_exact": True, "repeat": q2},
        "cross_encoder_GPU1": {"first": c1, "output_repeat_exact": True, "repeat": c2},
    },
}
print(json.dumps(facts, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
'''
    command = (
        f"cd {shlex.quote(str(REMOTE_ROOT / 'runtime'))} && "
        "env -i PATH=/usr/bin:/bin HOME=/home/erzhu419/p16_all_remote_20260722/preflight/home "
        "HF_HOME=/home/erzhu419/p16_all_remote_20260722/preflight/home/.cache "
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 "
        f"{shlex.quote(str(REMOTE_PYTHON))} -I -B -c {shlex.quote(code)}"
    )
    value = json.loads(_run(["ssh", "-o", "BatchMode=yes", REMOTE_HOST, command]))
    if not isinstance(value, Mapping):
        raise P16FreezeError("remote facts are malformed")
    return value


def _write_manifest(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise P16FreezeError(f"manifest is already consumed: {path}")
    p14_acquisition.utilities._write_json(path, value)


def freeze(project_root: Path) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    for relative in (FINGERPRINT_PATH, ACQUISITION_FREEZE_PATH, IMPLEMENTATION_FREEZE_PATH):
        if (base / relative).exists() or (base / relative).is_symlink():
            raise P16FreezeError(f"P16 freeze output already exists: {relative}")
    for forbidden in (acquisition.RESULT_RELATIVE, acquisition.RUN_ROOT_RELATIVE, p16.RUN_ROOT_RELATIVE):
        if (base / forbidden).exists() or (base / forbidden).is_symlink():
            raise P16FreezeError("P16 item identity or action root already exists")

    _assert_commit_bindings(
        project_root,
        base,
        tuple(dict.fromkeys((*DEPENDENCIES, *IMPLEMENTATIONS, acquisition.TEST_RELATIVE.as_posix()))),
    )

    facts = dict(_remote_facts())
    acquisition_freeze = _self_hashed(
        {
            "P14_acquisition_result_binding": {
                "file_sha256": acquisition.P14_RESULT_FILE_SHA256,
                "self_sha256": acquisition.P14_RESULT_SELF_SHA256,
            },
            "P14_selection_secret_sha256": acquisition.SELECTION_SECRET_SHA256,
            "dependency_bindings": [
                _binding(base, "assumption_agent/benchmarks/bright_p14_acquisition_v1.py"),
                _binding(base, "assumption_agent/benchmarks/bright_p14_source_qualification_v1.py"),
            ],
            "formal_implementation_commit": FORMAL_IMPLEMENTATION_COMMIT,
            "implementation_bindings": [
                _binding(base, acquisition.IMPLEMENTATION_RELATIVE.as_posix()),
                _binding(base, acquisition.TEST_RELATIVE.as_posix()),
            ],
            "recorded_date": "2026-07-22",
            "schema": acquisition.FREEZE_SCHEMA,
            "study_design_self_sha256": acquisition.DESIGN_SELF_SHA256,
        }
    )
    fingerprint = _self_hashed(
        {
            "candidate_freeze_self_sha256": p16.p14.CANDIDATE_FREEZE_SELF_SHA256,
            "formal_implementation_commit": FORMAL_IMPLEMENTATION_COMMIT,
            **facts,
            "item_identity_or_label_access_count": 0,
            "recorded_date": "2026-07-22",
            "remote_host_alias": p16.REMOTE_HOST_ALIAS,
            "remote_hostname": p16.REMOTE_HOSTNAME,
            "remote_root": str(p16.REMOTE_ROOT),
            "runtime_claim_boundary": {
                "cross_hardware_byte_reproducibility_claim": False,
                "fresh_runtime_fingerprinted_before_item_identity": True,
                "numerical_comparison_scope": "within_this_single_P16_remote_runtime",
            },
            "schema": p16.FINGERPRINT_SCHEMA,
            "stage_assignment": {
                "HippoRAG": {
                    "OMP_threads_per_process": 2,
                    "candidate_document_count": 32,
                    "process_concurrency": 8,
                    "timeout_seconds_per_process": 21_600,
                    "visible_GPU": "",
                },
                "MiniLM_and_Qwen": {"visible_physical_GPU": "0"},
                "cross_encoder": {"process_concurrency": 1, "visible_physical_GPU": "1"},
            },
            "status": "frozen_before_P16_item_identity_materialization",
            "study_design_self_sha256": acquisition.DESIGN_SELF_SHA256,
        }
    )
    implementation_freeze = _self_hashed(
        {
            "acquisition_freeze_binding": {
                "file_sha256": _manifest_file_sha256(acquisition_freeze),
                "relative_path": ACQUISITION_FREEZE_PATH.as_posix(),
                "self_sha256": acquisition_freeze["self_sha256"],
            },
            "candidate_freeze_self_sha256": p16.p14.CANDIDATE_FREEZE_SELF_SHA256,
            "dependency_bindings": [_binding(base, relative) for relative in DEPENDENCIES],
            "formal_implementation_commit": FORMAL_IMPLEMENTATION_COMMIT,
            "implementation_bindings": [_binding(base, relative) for relative in IMPLEMENTATIONS],
            "recorded_date": "2026-07-22",
            "remote_runtime_fingerprint_binding": {
                "file_sha256": _manifest_file_sha256(fingerprint),
                "relative_path": FINGERPRINT_PATH.as_posix(),
                "self_sha256": fingerprint["self_sha256"],
            },
            "schema": p16.FREEZE_SCHEMA,
            "status": "frozen_before_P16_item_identity_materialization",
            "study_design_self_sha256": acquisition.DESIGN_SELF_SHA256,
        }
    )
    _write_manifest(base / ACQUISITION_FREEZE_PATH, acquisition_freeze)
    _write_manifest(base / FINGERPRINT_PATH, fingerprint)
    _write_manifest(base / IMPLEMENTATION_FREEZE_PATH, implementation_freeze)
    return acquisition_freeze, fingerprint, implementation_freeze


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    values = freeze(arguments.project_root)
    print(
        json.dumps(
            {
                "acquisition_freeze_self_sha256": values[0]["self_sha256"],
                "fingerprint_self_sha256": values[1]["self_sha256"],
                "implementation_freeze_self_sha256": values[2]["self_sha256"],
                "status": "frozen",
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
