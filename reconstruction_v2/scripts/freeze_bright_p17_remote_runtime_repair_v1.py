#!/usr/bin/env python3
"""Freeze the one-shot, outcome-blind P17 prelaunch runtime repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "reconstruction_v2")]

from reconstruction_v2.assumption_agent.benchmarks import (  # noqa: E402
    bright_p14_acquisition_v1 as p14_acquisition,
)
from reconstruction_v2.scripts import (  # noqa: E402
    freeze_bright_p17_remote_runtime_v1 as original,
)


OLD_FINGERPRINT_FILE_SHA256 = (
    "a6afcd21a05c54d13fdbe50d320309fb025b276a77990c381a40425855948feb"
)
OLD_FINGERPRINT_SELF_SHA256 = (
    "6c32d54012d3480ad5b04852c7a2e5805107bc20880b2b2eafee708fddc3ed42"
)
OLD_IMPLEMENTATION_FREEZE_FILE_SHA256 = (
    "514c75746933746ca4588cb6b9ca03e4991fc78dabb576728a316151fc745b11"
)
OLD_IMPLEMENTATION_FREEZE_SELF_SHA256 = (
    "d99f97becce409d54aa37fed45a81be42361e975966f885b9ef96ff60cbcb7ce"
)
DISPOSITION_RELATIVE = Path("manifests/bright_p17_prelaunch_runtime_disposition_v1.json")
DISPOSITION_FILE_SHA256 = (
    "03dcb9ed4ba246a447d8d5d2b74cd442b79825d4300291450b4353c0d5fb0141"
)
DISPOSITION_SELF_SHA256 = (
    "3063be7b205ed5cd29b8d1f17c7711a160925786088b804dd4d34fc4914c108a"
)
PATCHED_SOURCE_SHA256 = (
    "960561b080531fe4d668bde635e81f8e65620ce50bdacdd9a25531e856fa3e05"
)
ACQUISITION_RESULT_FILE_SHA256 = (
    "2b086884fb12ff1b2ab7cc24db6981aa8d2ec0b33330c05f857fb3fd66e413ac"
)
ACQUISITION_RESULT_SELF_SHA256 = (
    "45577b489943bf610d882713e68c9460cc3eb068f814cb6211d7793e57cb6813"
)
STAGED_ITEM_COUNT = 27


class P17RepairFreezeError(RuntimeError):
    """The single permitted P17 infrastructure repair cannot be frozen."""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return p14_acquisition.utilities.canonical_json_bytes(value)


def _self_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    return p14_acquisition.utilities.self_hashed(value, field="self_sha256")


def _read_canonical(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P17RepairFreezeError(f"{name} is unavailable") from exc
    if not isinstance(value, Mapping) or path.read_bytes() != _canonical_json_bytes(value):
        raise P17RepairFreezeError(f"{name} is not canonical")
    return value


def _verify_self(value: Mapping[str, Any], name: str) -> None:
    body = dict(value)
    declared = body.pop("self_sha256", None)
    if not isinstance(declared, str) or p14_acquisition.utilities.stable_hash(body) != declared:
        raise P17RepairFreezeError(f"{name} self hash drifted")


def _run(command: Sequence[str]) -> str:
    return subprocess.run(
        list(command), check=True, capture_output=True, text=True
    ).stdout.strip()


def _remote_repair_facts() -> Mapping[str, Any]:
    code = r'''import hashlib
import json
import os
from pathlib import Path
import subprocess

from reconstruction_v2.assumption_agent.benchmarks import bright_p17_extension_acquisition_v1 as acquisition
from reconstruction_v2.assumption_agent.benchmarks import nanobeir_p12_acquisition_v1 as utilities
from reconstruction_v2.replication_runtime.bright_p17_all_remote_v1 import runner
from reconstruction_v2.replication_runtime.hipporag_upstream_hardening_v1 import backport

root = Path("/home/erzhu419/p17_all_remote_20260722")
base = root / "runtime/reconstruction_v2"
run = root / "preflight/hippo_repaired_p17/run1"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

if (root / "work").exists() or (root / "work").is_symlink():
    raise RuntimeError("P17 formal work already exists")
for forbidden in (
    base / acquisition.SELECTION_SECRET_RELATIVE,
    base / acquisition.source.SOURCE_ROOT_RELATIVE / "examples",
    base / "artifacts/bright_p14_direct_c_confirm_v1/hipporag",
):
    if forbidden.exists() or forbidden.is_symlink():
        raise RuntimeError("a secret, label, or prior action artifact is staged")

acquisition_path = base / acquisition.RESULT_RELATIVE
if sha(acquisition_path) != "2b086884fb12ff1b2ab7cc24db6981aa8d2ec0b33330c05f857fb3fd66e413ac":
    raise RuntimeError("the staged acquisition result drifted")
view_path = base / "artifacts/bright_p17_extension_acquisition_v1/private/C_confirm.view.json"
if not view_path.is_file() or view_path.is_symlink():
    raise RuntimeError("the staged P17 view is absent")

source_path = (
    base
    / "reference/self_evo_continual_20260707/repos/HippoRAG/src/hipporag/HippoRAG.py"
)
if sha(source_path) != backport.PATCHED_SOURCE_SHA256:
    raise RuntimeError("the deterministic source repair target drifted")

output_path = run / "output.json"
if not output_path.is_file() or output_path.is_symlink():
    raise RuntimeError("the repaired HippoRAG canary is incomplete")
payload = json.loads(output_path.read_text(encoding="ascii"))
audit = runner._network_audit(run, "network.trace")
service = dict(
    line.split("=", 1)
    for line in subprocess.run(
        [
            "systemctl", "--user", "show", "p17-hippo-repair-preflight-v1.service",
            "-p", "ActiveState", "-p", "Result", "-p", "ExecMainStatus", "--no-pager",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}"},
    ).stdout.splitlines()
    if "=" in line
)
if (
    payload.get("schema") != "bright_official_hipporag_candidate_retrieval_v1_output"
    or len(payload.get("top_ordinals", [])) != 10
    or payload.get("graph_node_count", 0) <= 32
    or payload.get("graph_edge_count", 0) <= 0
    or audit["denied_external_network_syscall_count"] != 0
    or service.get("ActiveState") != "inactive"
    or service.get("Result") != "success"
    or service.get("ExecMainStatus") != "0"
):
    raise RuntimeError("the repaired HippoRAG canary is nonterminal or network-invalid")

facts = {
    "frozen_asset_receipts": runner._frozen_asset_receipts(base),
    "hardened_source_file_sha256": sha(source_path),
    "hipporag_canary": {
        "candidate_document_count": 32,
        "elapsed_seconds_from_input_to_output_mtime": output_path.stat().st_mtime - (run / "input.json").stat().st_mtime,
        "graph_edge_count": payload["graph_edge_count"],
        "graph_node_count": payload["graph_node_count"],
        "input_file_sha256": sha(run / "input.json"),
        "network_audit": audit,
        "origin": "P17_outcome_blind_prelaunch_deterministic_source_repair_canary",
        "output_file_sha256": sha(output_path),
        "service_result": "success",
        "terminal": True,
        "top_ordinal_count": len(payload["top_ordinals"]),
        "top_ordinals_sha256": utilities.stable_hash(payload["top_ordinals"]),
        "visible_GPU": "",
    },
    "measurement_access": {
        "gold_column_read_count": 0,
        "model_or_comparator_action_count": 0,
        "online_evaluator_call_count": 0,
        "performance_score_count": 0,
        "remote_item_identity_staged_count": 27,
        "remote_work_root_created": False,
    },
    "runtime_inventory_receipt": runner._runtime_inventory_receipt(),
    "view_file_sha256": sha(view_path),
}
print(json.dumps(facts, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
'''
    command = (
        f"cd {shlex.quote(str(original.REMOTE_ROOT / 'runtime'))} && "
        "env -i PATH=/usr/bin:/bin HOME=/home/erzhu419/p17_all_remote_20260722/preflight/home "
        "HF_HOME=/home/erzhu419/p17_all_remote_20260722/preflight/home/.cache "
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 "
        f"{shlex.quote(str(original.REMOTE_PYTHON))} -I -B -c {shlex.quote(code)}"
    )
    try:
        value = json.loads(
            _run(["ssh", "-o", "BatchMode=yes", original.REMOTE_HOST, command])
        )
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise P17RepairFreezeError("remote repair facts are unavailable") from exc
    if not isinstance(value, Mapping):
        raise P17RepairFreezeError("remote repair facts are malformed")
    return value


def _write_replacement(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".repair.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise P17RepairFreezeError(f"repair temporary already exists: {temporary}")
    p14_acquisition.utilities._write_json(temporary, value, mode=0o644)
    temporary.replace(path)


def freeze(project_root: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    project_root = project_root.resolve(strict=True)
    base = project_root / "reconstruction_v2"
    fingerprint_path = base / original.FINGERPRINT_PATH
    implementation_path = base / original.IMPLEMENTATION_FREEZE_PATH
    disposition_path = base / DISPOSITION_RELATIVE

    if _file_sha256(fingerprint_path) != OLD_FINGERPRINT_FILE_SHA256:
        raise P17RepairFreezeError("the invalid P17 fingerprint was already consumed")
    if _file_sha256(implementation_path) != OLD_IMPLEMENTATION_FREEZE_FILE_SHA256:
        raise P17RepairFreezeError("the invalid P17 implementation freeze was already consumed")
    if _file_sha256(disposition_path) != DISPOSITION_FILE_SHA256:
        raise P17RepairFreezeError("the P17 prelaunch disposition drifted")

    fingerprint_old = _read_canonical(fingerprint_path, "invalid P17 fingerprint")
    implementation_old = _read_canonical(
        implementation_path, "invalid P17 implementation freeze"
    )
    disposition = _read_canonical(disposition_path, "P17 prelaunch disposition")
    for value, expected, name in (
        (fingerprint_old, OLD_FINGERPRINT_SELF_SHA256, "invalid P17 fingerprint"),
        (
            implementation_old,
            OLD_IMPLEMENTATION_FREEZE_SELF_SHA256,
            "invalid P17 implementation freeze",
        ),
        (disposition, DISPOSITION_SELF_SHA256, "P17 prelaunch disposition"),
    ):
        _verify_self(value, name)
        if value.get("self_sha256") != expected:
            raise P17RepairFreezeError(f"{name} identity drifted")

    acquisition_path = base / "manifests/bright_p17_extension_acquisition_result_v1.json"
    acquisition_result = _read_canonical(acquisition_path, "P17 acquisition result")
    _verify_self(acquisition_result, "P17 acquisition result")
    if (
        _file_sha256(acquisition_path) != ACQUISITION_RESULT_FILE_SHA256
        or acquisition_result.get("self_sha256") != ACQUISITION_RESULT_SELF_SHA256
    ):
        raise P17RepairFreezeError("P17 acquisition result drifted")
    if (original.REMOTE_ROOT / "work").exists():
        raise P17RepairFreezeError("local code confused a remote path with a local work root")

    facts = dict(_remote_repair_facts())
    if facts.get("hardened_source_file_sha256") != PATCHED_SOURCE_SHA256:
        raise P17RepairFreezeError("the repaired source hash drifted")
    if facts.get("measurement_access") != {
        "gold_column_read_count": 0,
        "model_or_comparator_action_count": 0,
        "online_evaluator_call_count": 0,
        "performance_score_count": 0,
        "remote_item_identity_staged_count": STAGED_ITEM_COUNT,
        "remote_work_root_created": False,
    }:
        raise P17RepairFreezeError("the outcome-blind repair boundary drifted")
    if facts.get("runtime_inventory_receipt") != fingerprint_old.get(
        "runtime_inventory_receipt"
    ):
        raise P17RepairFreezeError("the P17 package inventory changed during repair")
    old_assets = fingerprint_old.get("frozen_asset_receipts")
    new_assets = facts.get("frozen_asset_receipts")
    if not isinstance(old_assets, Mapping) or not isinstance(new_assets, Mapping):
        raise P17RepairFreezeError("P17 asset receipts are malformed")
    for key in ("HippoRAG_LLM", "MiniLM", "Qwen", "cross_encoder"):
        if old_assets.get(key) != new_assets.get(key):
            raise P17RepairFreezeError(f"an unrelated frozen asset changed: {key}")
    if old_assets.get("HippoRAG_source") == new_assets.get("HippoRAG_source"):
        raise P17RepairFreezeError("the required HippoRAG source repair did not occur")

    fingerprint_body = dict(fingerprint_old)
    fingerprint_body.pop("self_sha256")
    fingerprint_body["frozen_asset_receipts"] = dict(new_assets)
    worker_canaries = dict(fingerprint_body["worker_canaries"])
    worker_canaries["HippoRAG_CPU"] = facts["hipporag_canary"]
    fingerprint_body["worker_canaries"] = worker_canaries
    fingerprint_body["legacy_item_identity_or_label_access_count_scope"] = (
        "original_fingerprint_before_P17_acquisition_only"
    )
    fingerprint_body["remote_item_identity_staged_count_before_repair"] = STAGED_ITEM_COUNT
    fingerprint_body["prelaunch_runtime_repair"] = {
        "disposition_binding": {
            "file_sha256": DISPOSITION_FILE_SHA256,
            "relative_path": DISPOSITION_RELATIVE.as_posix(),
            "self_sha256": DISPOSITION_SELF_SHA256,
        },
        "measurement_access": facts["measurement_access"],
        "original_invalid_fingerprint": {
            "file_sha256": OLD_FINGERPRINT_FILE_SHA256,
            "self_sha256": OLD_FINGERPRINT_SELF_SHA256,
        },
        "repair_choice_count": 1,
        "repaired_source_file_sha256": PATCHED_SOURCE_SHA256,
        "source_tree_receipt_scope": (
            "same_host_post_repair_tree_including_nonportable_cached_bytecode"
        ),
        "view_file_sha256": facts["view_file_sha256"],
    }
    fingerprint_body["runtime_claim_boundary"] = {
        "cross_hardware_byte_reproducibility_claim": False,
        "fresh_runtime_fingerprinted_before_item_identity": False,
        "gold_or_score_access_before_repair": False,
        "identity_exposed_before_deterministic_repair": True,
        "numerical_comparison_scope": "within_this_single_P17_remote_runtime",
        "repair_target_precommitted_in_formal_code": True,
        "source_tree_contains_nonportable_cached_bytecode": True,
    }
    fingerprint_body["status"] = (
        "corrected_after_P17_item_identity_staging_before_any_model_or_comparator_action"
    )
    fingerprint = _self_hashed(fingerprint_body)

    implementation_body = dict(implementation_old)
    implementation_body.pop("self_sha256")
    dependencies = list(implementation_body["dependency_bindings"])
    if any(row.get("relative_path") == DISPOSITION_RELATIVE.as_posix() for row in dependencies):
        raise P17RepairFreezeError("the repair disposition was already bound")
    dependencies.append(
        {
            "relative_path": DISPOSITION_RELATIVE.as_posix(),
            "sha256": DISPOSITION_FILE_SHA256,
        }
    )
    implementation_body["dependency_bindings"] = dependencies
    implementation_body["prelaunch_runtime_repair_binding"] = {
        "disposition_self_sha256": DISPOSITION_SELF_SHA256,
        "original_invalid_implementation_freeze": {
            "file_sha256": OLD_IMPLEMENTATION_FREEZE_FILE_SHA256,
            "self_sha256": OLD_IMPLEMENTATION_FREEZE_SELF_SHA256,
        },
        "repaired_fingerprint_self_sha256": fingerprint["self_sha256"],
    }
    implementation_body["remote_runtime_fingerprint_binding"] = {
        "file_sha256": hashlib.sha256(_canonical_json_bytes(fingerprint)).hexdigest(),
        "relative_path": original.FINGERPRINT_PATH.as_posix(),
        "self_sha256": fingerprint["self_sha256"],
    }
    implementation_body["status"] = (
        "outcome_blind_prelaunch_runtime_repair_frozen_after_identity_staging"
    )
    implementation = _self_hashed(implementation_body)

    _write_replacement(fingerprint_path, fingerprint)
    _write_replacement(implementation_path, implementation)
    return fingerprint, implementation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    fingerprint, implementation = freeze(arguments.project_root)
    print(
        json.dumps(
            {
                "fingerprint_self_sha256": fingerprint["self_sha256"],
                "implementation_freeze_self_sha256": implementation["self_sha256"],
                "status": "P17_outcome_blind_prelaunch_runtime_repair_frozen",
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
