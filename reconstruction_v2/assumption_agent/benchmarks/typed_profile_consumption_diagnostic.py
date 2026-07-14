from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..events import Event, JsonlEventSink, MemoryEventSink
from ..models import SplitName, stable_hash
from ..secure_env import (
    configured_api_origin,
    configured_model,
    configured_skilllearn_provider_mode,
    load_dotenv,
    map_legacy_model_env,
)
from ..splits import SplitManifest
from .docker_egress import configured_trial_network_byte_limit
from .paper_protocol import PaperProtocol
from .prewarm import (
    FrozenTaskInputPrebuiltImageCache,
    validate_development_prewarm_receipt,
)
from .runtime_profile_injection import RUNTIME_PROFILE_PROMPT_DELIVERY_VERSION
from .skilllearn_compiler import verify_compiled_skill_source
from .skilllearn_lifecycle import (
    SkillLearnModelInferenceLimiter,
    SkillLearnProviderCircuit,
    SkillLearnSubprocessBackend,
    SkillLearnTrialObservation,
    SkillLearnTrialRequest,
    TrialVariant,
)
from .task_input_freeze import (
    FrozenTaskInputClosure,
    expected_prewarm_closure_rows,
    load_frozen_task_input_closure,
)
from .typed_profile_injection_integration import (
    _read_preregistration as _read_injection_preregistration,
)


PROFILE_CONSUMPTION_DIAGNOSTIC_VERSION = (
    "v320_consumed_development_profile_consumption_diagnostic_v1"
)
PROFILE_CONSUMPTION_REPORT_VERSION = (
    "typed_profile_consumption_diagnostic_report_v1"
)
PROFILE_CONSUMPTION_RESULT_VERSION = (
    "typed_profile_consumption_diagnostic_result_receipt_v1"
)
_EXPECTED_NEW_TRIAL_COUNT = 6
_EXPECTED_ITEM_COUNT = 3
_EXPECTED_SOURCE_KEYS = {
    "benchmark_anchor_file",
    "g1_compile_manifest",
    "g2_compile_manifest",
    "injection_integration_preregistration",
    "injection_integration_result",
    "manifest",
    "paper_protocol",
    "prewarm_receipt",
    "protocol_lock",
    "v320_event_ledger",
}
_REQUIRED_IMPLEMENTATION_FILES = {
    "assumption_agent/benchmarks/docker_egress.py",
    "assumption_agent/benchmarks/offline_verifier.py",
    "assumption_agent/benchmarks/prewarm.py",
    "assumption_agent/benchmarks/runtime_profile_injection.py",
    "assumption_agent/benchmarks/skilllearn_compiler.py",
    "assumption_agent/benchmarks/skilllearn_lifecycle.py",
    "assumption_agent/benchmarks/task_input_closure.py",
    "assumption_agent/benchmarks/task_input_freeze.py",
    "assumption_agent/benchmarks/typed_profile_consumption_diagnostic.py",
    "assumption_agent/events.py",
    "tests/test_typed_profile_consumption_diagnostic.py",
}
_EXPECTED_EXECUTION_CONTRACT: Mapping[str, Any] = {
    "claim_eligible": False,
    "development_validation_consumed_before_preregistration": True,
    "evaluation_mode": "offline_post_agent_verifier",
    "fresh_validation": False,
    "new_policy_off_model_calls": 0,
    "new_policy_on_trial_budget": _EXPECTED_NEW_TRIAL_COUNT,
    "parallel_workers": _EXPECTED_NEW_TRIAL_COUNT,
    "promotion_evaluated": False,
    "proposal_or_training_model_calls": 0,
    "retry_allowed": False,
    "sealed_test_bytes_exposed_to_model": False,
    "sealed_test_scoring_performed": False,
    "stored_raw_baseline_reused": True,
    "test_infrastructure_metadata_inspected": True,
    "test_task_input_bytes_inspected": False,
    "test_trial_executed": False,
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PermissionError(f"expected one JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_source_path(project_root: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise PermissionError("diagnostic source path is malformed")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("diagnostic source path escaped the project")
    path = (project_root / relative).resolve(strict=True)
    path.relative_to(project_root)
    if not path.is_file():
        raise PermissionError("diagnostic source file is missing")
    return path


def _safe_output_path(project_root: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise PermissionError("diagnostic output path is malformed")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("diagnostic output path escaped the project")
    path = (project_root / relative).resolve()
    path.relative_to(project_root)
    return path


def _source_path(
    project_root: Path,
    preregistration: Mapping[str, Any],
    key: str,
) -> Path:
    return _safe_source_path(
        project_root,
        preregistration["sources"][key]["path"],
    )


def _artifact_paths(
    project_root: Path,
    preregistration: Mapping[str, Any],
) -> dict[str, Path]:
    artifacts = preregistration.get("artifacts")
    expected = {"completion_lock", "events", "report", "result_receipt", "run_root"}
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected:
        raise PermissionError("diagnostic artifact map drifted")
    paths = {
        key: _safe_output_path(project_root, artifacts[key])
        for key in expected
    }
    if len(set(paths.values())) != len(paths):
        raise PermissionError("diagnostic artifact paths overlap")
    source_paths = {
        _safe_source_path(project_root, row["path"])
        for row in preregistration.get("sources", {}).values()
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }
    if any(path in source_paths for path in paths.values()):
        raise PermissionError("diagnostic artifact overlaps a frozen source")
    run_root = paths["run_root"]
    for key in expected - {"run_root", "result_receipt"}:
        if run_root not in paths[key].parents:
            raise PermissionError("diagnostic runtime artifact escaped its run root")
    return paths


def _implementation_rows(
    project_root: Path,
    paths: Sequence[str],
) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "path": raw,
            "sha256": _sha256_file(_safe_source_path(project_root, raw)),
        }
        for raw in paths
    )


def _pair_events(
    event_path: Path,
    trial_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    old_hash_index = {
        str(row["old_request_hash"]): (int(row["generation"]), str(row["item_id"]))
        for row in trial_rows
    }
    pairs: dict[tuple[int, str], Mapping[str, Any]] = {}
    for raw in event_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(raw)
        if (
            not isinstance(event, Mapping)
            or event.get("event") != "skilllearn_counterfactual_pair_completed"
        ):
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        key = old_hash_index.get(str(payload.get("candidate_request_hash") or ""))
        if key is None:
            continue
        if key in pairs:
            raise PermissionError("diagnostic historical pair is duplicated")
        pairs[key] = dict(payload)
    if set(pairs) != {
        (int(row["generation"]), str(row["item_id"])) for row in trial_rows
    }:
        raise PermissionError("diagnostic historical pair coverage drifted")
    return pairs


def _historical_trial_receipts(
    event_path: Path,
    trial_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], Mapping[str, Any]]:
    old_hash_index = {
        str(row["old_request_hash"]): (int(row["generation"]), str(row["item_id"]))
        for row in trial_rows
    }
    receipts: dict[tuple[int, str], Mapping[str, Any]] = {}
    for raw in event_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(raw)
        if (
            not isinstance(event, Mapping)
            or event.get("event") != "skilllearn_trial_completed"
        ):
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        key = old_hash_index.get(str(payload.get("request_hash") or ""))
        if key is None:
            continue
        if key in receipts:
            raise PermissionError("diagnostic historical trial receipt is duplicated")
        receipts[key] = dict(payload)
    if set(receipts) != {
        (int(row["generation"]), str(row["item_id"])) for row in trial_rows
    }:
        raise PermissionError("diagnostic historical trial receipt coverage drifted")
    return receipts


def _request_from_row(
    *,
    project_root: Path,
    preregistration: Mapping[str, Any],
    protocol: PaperProtocol,
    manifest: SplitManifest,
    row: Mapping[str, Any],
    delivery_enabled: bool,
) -> tuple[SkillLearnTrialRequest, Path]:
    common = preregistration["request_common"]
    generation = int(row["generation"])
    compile_key = "g1_compile_manifest" if generation == 1 else "g2_compile_manifest"
    compile_root = _source_path(project_root, preregistration, compile_key).parent
    item_id = str(row["item_id"])
    item_hash = stable_hash({"item_id": item_id})
    source = compile_root / "items" / item_hash
    request = SkillLearnTrialRequest(
        item_id=item_id,
        family=str(row["family"]),
        split=SplitName.VALIDATION,
        variant=TrialVariant.POLICY_ON,
        evaluator_epoch=str(common["evaluator_epoch"]),
        pair_id=str(row["pair_id"]),
        repeat=int(common["repeat"]),
        agent_id=str(common["agent_id"]),
        model=str(common["model"]),
        max_steps=int(common["max_steps"]),
        manifest_hash=manifest.manifest_hash,
        codex_agent_execution_policy_hash=(
            protocol.codex_agent_execution_policy.policy_hash
        ),
        program_id=None,
        program_set_hash=str(row["program_set_hash"]),
        treatment_hash=str(row["treatment_hash"]),
        compile_manifest_hash=str(row["compile_manifest_hash"]),
        skill_source_receipt_hash=str(row["skill_source_receipt_hash"]),
        compile_root=compile_root,
        typed_binding_set_hash=str(row["typed_binding_set_hash"]),
        typed_snapshot_hashes=tuple(common["typed_snapshot_hashes"]),
        typed_snapshot_ledger_hash=str(common["typed_snapshot_ledger_hash"]),
        portable_capability_compiler_mode=str(
            common["portable_capability_compiler_mode"]
        ),
        portable_capability_role_spec_set_hash=str(
            row["portable_capability_role_spec_set_hash"]
        ),
        portable_capability_role_spec_hashes=tuple(
            row["portable_capability_role_spec_hashes"]
        ),
        portable_capability_delivery_mode=(
            RUNTIME_PROFILE_PROMPT_DELIVERY_VERSION if delivery_enabled else ""
        ),
        candidate_delta_program_set_hash=str(
            row["candidate_delta_program_set_hash"]
        ),
        candidate_full_program_set_hash=str(
            row["candidate_full_program_set_hash"]
        ),
        matched_candidate_program_set_hash=str(
            row["matched_candidate_program_set_hash"]
        ),
        selected_candidate_hypothesis_ids=tuple(
            row["selected_candidate_hypothesis_ids"]
        ),
        matched_candidate_hypothesis_ids=tuple(
            row["matched_candidate_hypothesis_ids"]
        ),
    )
    return request, source


def _verify_trial_sources(
    *,
    project_root: Path,
    preregistration: Mapping[str, Any],
    protocol: PaperProtocol,
    manifest: SplitManifest,
    pair_events: Mapping[tuple[int, str], Mapping[str, Any]],
) -> None:
    for row in preregistration["trial_requests"]:
        old_request, source = _request_from_row(
            project_root=project_root,
            preregistration=preregistration,
            protocol=protocol,
            manifest=manifest,
            row=row,
            delivery_enabled=False,
        )
        new_request, rebound_source = _request_from_row(
            project_root=project_root,
            preregistration=preregistration,
            protocol=protocol,
            manifest=manifest,
            row=row,
            delivery_enabled=True,
        )
        old_payload = old_request.to_dict()
        expected_new_payload = dict(old_payload)
        expected_new_payload["portable_capability_delivery_mode"] = (
            RUNTIME_PROFILE_PROMPT_DELIVERY_VERSION
        )
        if (
            source != rebound_source
            or old_request.request_hash != row["old_request_hash"]
            or new_request.request_hash != row["new_request_hash"]
            or new_request.to_dict() != expected_new_payload
        ):
            raise PermissionError("diagnostic request is not a delivery-only delta")
        receipt = verify_compiled_skill_source(
            compile_root=old_request.compile_root,
            item_id=old_request.item_id,
            skill_source_dir=source,
            expected_compile_manifest_hash=old_request.compile_manifest_hash,
            expected_program_set_hash=old_request.program_set_hash,
            expected_treatment_hash=old_request.treatment_hash,
            expected_typed_binding_set_hash=old_request.typed_binding_set_hash,
            expected_typed_snapshot_hashes=old_request.typed_snapshot_hashes,
            expected_typed_snapshot_ledger_hash=(
                old_request.typed_snapshot_ledger_hash
            ),
            expected_portable_capability_compiler_mode=(
                old_request.portable_capability_compiler_mode
            ),
            expected_portable_capability_role_spec_set_hash=(
                old_request.portable_capability_role_spec_set_hash
            ),
            expected_portable_capability_role_spec_hashes=(
                old_request.portable_capability_role_spec_hashes
            ),
        )
        if receipt.receipt_hash != old_request.skill_source_receipt_hash:
            raise PermissionError("diagnostic compiled source receipt drifted")
        pair = pair_events[(int(row["generation"]), str(row["item_id"]))]
        comparisons = {
            "pair_id": old_request.pair_id,
            "candidate_request_hash": old_request.request_hash,
            "candidate_compile_manifest_hash": old_request.compile_manifest_hash,
            "candidate_program_set_hash": old_request.program_set_hash,
            "candidate_treatment_hash": old_request.treatment_hash,
            "candidate_skill_source_receipt_hash": (
                old_request.skill_source_receipt_hash
            ),
            "candidate_typed_binding_set_hash": (
                old_request.typed_binding_set_hash
            ),
            "candidate_typed_snapshot_hashes": list(
                old_request.typed_snapshot_hashes
            ),
            "candidate_typed_snapshot_ledger_hash": (
                old_request.typed_snapshot_ledger_hash
            ),
            "candidate_delta_program_set_hash": (
                old_request.candidate_delta_program_set_hash
            ),
            "candidate_full_program_set_hash": (
                old_request.candidate_full_program_set_hash
            ),
            "matched_candidate_program_set_hash": (
                old_request.matched_candidate_program_set_hash
            ),
            "selected_candidate_hypothesis_ids": list(
                old_request.selected_candidate_hypothesis_ids
            ),
            "matched_candidate_hypothesis_ids": list(
                old_request.matched_candidate_hypothesis_ids
            ),
        }
        if any(pair.get(key) != value for key, value in comparisons.items()):
            raise PermissionError("diagnostic request differs from v3.20 pair evidence")


def _read_preregistration(
    path: str | Path,
) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    project_root = resolved.parent.parent.resolve(strict=True)
    payload = _read_json(resolved)
    if payload.get("diagnostic_policy") != PROFILE_CONSUMPTION_DIAGNOSTIC_VERSION:
        raise PermissionError("profile consumption diagnostic policy drifted")
    if payload.get("execution_contract") != dict(_EXPECTED_EXECUTION_CONTRACT):
        raise PermissionError("profile consumption execution contract drifted")
    implementation_paths = payload.get("implementation_files")
    if not isinstance(implementation_paths, list) or not all(
        isinstance(value, str) for value in implementation_paths
    ):
        raise PermissionError("diagnostic implementation set is malformed")
    if not _REQUIRED_IMPLEMENTATION_FILES <= set(implementation_paths):
        raise PermissionError("diagnostic implementation set is incomplete")
    rows = _implementation_rows(project_root, implementation_paths)
    if stable_hash({"files": list(rows)}) != payload.get(
        "implementation_file_set_hash"
    ):
        raise PermissionError("profile consumption implementation drifted")
    sources = payload.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != _EXPECTED_SOURCE_KEYS:
        raise PermissionError("diagnostic source set drifted")
    for key, source_row in sources.items():
        if not isinstance(source_row, Mapping):
            raise PermissionError(f"diagnostic source is malformed: {key}")
        source = _safe_source_path(project_root, source_row.get("path"))
        if _sha256_file(source) != source_row.get("sha256"):
            raise PermissionError(f"diagnostic source drifted: {key}")
    injection_path = _source_path(
        project_root,
        payload,
        "injection_integration_preregistration",
    )
    _read_injection_preregistration(injection_path)
    injection_result = _read_json(
        _source_path(project_root, payload, "injection_integration_result")
    )
    if (
        injection_result.get("integration_passed") is not True
        or injection_result.get("exact_replay_verified") is not True
        or injection_result.get("development_task_execution_authorized") is not False
    ):
        raise PermissionError("runtime injection source boundary drifted")
    protocol = PaperProtocol.read(
        _source_path(project_root, payload, "paper_protocol")
    )
    manifest = SplitManifest.read(_source_path(project_root, payload, "manifest"))
    protocol_lock = _read_json(
        _source_path(project_root, payload, "protocol_lock")
    )
    common = payload.get("request_common")
    if not isinstance(common, Mapping) or common != {
        "agent_id": protocol.payload["agent_id"],
        "codex_agent_execution_policy_hash": (
            protocol.codex_agent_execution_policy.policy_hash
        ),
        "evaluator_epoch": f"skilllearn-eval-{manifest.manifest_hash[:12]}",
        "manifest_hash": manifest.manifest_hash,
        "max_steps": protocol.payload["max_steps"],
        "model": protocol.payload["model"],
        "portable_capability_compiler_mode": (
            protocol.payload["execution"]["portable_capability_compiler_mode"]
        ),
        "repeat": 1,
        "typed_snapshot_hashes": [
            "08a64b925c2f39bec6bd7e8105bb3c13dfc4e3af9e09d49196e14eacab01ea83",
            "4c70b1d811007bc9ac987ea7bac3d11c9e276ac2bd73348dac18d58a4cfdc6ae",
            "8ab2615c39e0f4d10538032fe287ed4532705c634a8466d60c5ef7f43b426682",
        ],
        "typed_snapshot_ledger_hash": (
            "d560903a5df0da0a464b3636ef2f80bd86cba3f5230de53f5da6f3acc4597bbf"
        ),
    }:
        raise PermissionError("diagnostic frozen request common fields drifted")
    if (
        protocol_lock.get("lock_hash") != payload.get("source_protocol_lock_hash")
        or protocol_lock.get("protocol_hash") != protocol.protocol_hash
        or protocol_lock.get("primary_manifest_hash") != manifest.manifest_hash
        or protocol_lock.get("resolved_codex_agent_execution_policy_hash")
        != protocol.codex_agent_execution_policy.policy_hash
    ):
        raise PermissionError("diagnostic historical protocol lock drifted")
    item_rows = payload.get("validation_items")
    if not isinstance(item_rows, list) or len(item_rows) != _EXPECTED_ITEM_COUNT:
        raise PermissionError("diagnostic validation item set drifted")
    if stable_hash({"rows": item_rows}) != payload.get("validation_item_set_hash"):
        raise PermissionError("diagnostic validation item hash drifted")
    expected_item_keys = {
        "fairness_fingerprint",
        "family",
        "family_hash",
        "item_id",
        "item_id_hash",
        "offline_verifier_profile_id",
        "offline_verifier_runtime_key",
        "prebuilt_image_id",
        "prebuilt_image_key",
        "provider_fingerprint",
        "task_input_closure_hash",
        "task_input_integrity_receipt_hash",
    }
    if any(
        not isinstance(row, Mapping) or set(row) != expected_item_keys
        for row in item_rows
    ):
        raise PermissionError("diagnostic validation item receipt fields drifted")
    item_ids = {str(row.get("item_id") or "") for row in item_rows}
    if len(item_ids) != _EXPECTED_ITEM_COUNT:
        raise PermissionError("diagnostic validation items are not unique")
    for row in item_rows:
        item_id = str(row.get("item_id") or "")
        family = str(row.get("family") or "")
        if (
            item_id not in manifest.validation_ids
            or item_id in manifest.train_ids
            or item_id in manifest.test_ids
            or manifest.family_by_id.get(item_id) != family
            or row.get("item_id_hash") != stable_hash({"item_id": item_id})
            or row.get("family_hash") != stable_hash({"family": family})
        ):
            raise PermissionError("diagnostic item is not consumed validation")
    trial_rows = payload.get("trial_requests")
    if not isinstance(trial_rows, list) or len(trial_rows) != _EXPECTED_NEW_TRIAL_COUNT:
        raise PermissionError("diagnostic request set drifted")
    trial_keys = {
        (int(row.get("generation") or 0), str(row.get("item_id") or ""))
        for row in trial_rows
    }
    if trial_keys != {
        (generation, item_id)
        for generation in (1, 2)
        for item_id in item_ids
    }:
        raise PermissionError("diagnostic request coverage drifted")
    pairs = _pair_events(
        _source_path(project_root, payload, "v320_event_ledger"),
        trial_rows,
    )
    required_pair_fields = {
        "baseline_cost",
        "baseline_evidence_hash",
        "baseline_observation_hash",
        "baseline_replayed",
        "baseline_request_hash",
        "baseline_success",
        "baseline_trial_executed",
        "baseline_valid",
        "candidate_cost",
        "candidate_observation_hash",
        "candidate_success",
        "candidate_trial_executed",
        "candidate_valid",
    }
    if any(not required_pair_fields <= set(pair) for pair in pairs.values()):
        raise PermissionError("diagnostic historical pair schema drifted")
    for pair in pairs.values():
        if (
            pair.get("split") != "validation"
            or pair.get("provider_matched") is not True
            or pair.get("budget_matched") is not True
            or pair.get("treatment_applied") is not True
            or pair.get("action_activated") is not True
            or pair.get("candidate_trial_executed") is not True
            or pair.get("candidate_valid") is not True
            or pair.get("candidate_success") is not False
        ):
            raise PermissionError("diagnostic historical treatment evidence drifted")
    for item_id in item_ids:
        first = pairs[(1, item_id)]
        second = pairs[(2, item_id)]
        if (
            first.get("baseline_trial_executed") is not True
            or first.get("baseline_replayed") is not False
            or second.get("baseline_trial_executed") is not False
            or second.get("baseline_replayed") is not True
            or any(
                first.get(name) != second.get(name)
                for name in (
                    "baseline_cost",
                    "baseline_evidence_hash",
                    "baseline_success",
                    "baseline_valid",
                )
            )
        ):
            raise PermissionError("diagnostic stored RAW replay evidence drifted")
    historical_receipts = _historical_trial_receipts(
        _source_path(project_root, payload, "v320_event_ledger"),
        trial_rows,
    )
    item_by_id = {str(row["item_id"]): row for row in item_rows}
    historical_receipt_keys = (
        "fairness_fingerprint",
        "offline_verifier_profile_id",
        "offline_verifier_runtime_key",
        "prebuilt_image_id",
        "prebuilt_image_key",
        "provider_fingerprint",
    )
    for trial_row in trial_rows:
        key = (int(trial_row["generation"]), str(trial_row["item_id"]))
        historical = historical_receipts[key]
        item = item_by_id[str(trial_row["item_id"])]
        if (
            historical.get("valid") is not True
            or any(historical.get(name) != item[name] for name in historical_receipt_keys)
        ):
            raise PermissionError("diagnostic historical runtime receipt drifted")
    _verify_trial_sources(
        project_root=project_root,
        preregistration=payload,
        protocol=protocol,
        manifest=manifest,
        pair_events=pairs,
    )
    prior_rows = payload.get("prior_results")
    if not isinstance(prior_rows, list) or len(prior_rows) != 9:
        raise PermissionError("diagnostic prior-result set drifted")
    prior_keys: set[tuple[str, str]] = set()
    for row in prior_rows:
        arm = str(row.get("arm") or "")
        item_id = str(row.get("item_id") or "")
        key = (arm, item_id)
        if key in prior_keys:
            raise PermissionError("diagnostic prior result is duplicated")
        prior_keys.add(key)
        result_path = _safe_source_path(project_root, row.get("path"))
        result = _read_json(result_path)
        generation = 1 if arm in {"raw", "g1_without_prompt_delivery"} else 2
        pair = pairs[(generation, item_id)]
        if arm == "raw":
            request_hash = str(pair["baseline_request_hash"])
            expected_trial_id = f"v2_policy_off_{request_hash[:18]}"
            expected_skill_config = "no_skill"
        else:
            request_hash = str(pair["candidate_request_hash"])
            expected_trial_id = f"v2_policy_on_{request_hash[:18]}"
            expected_skill_config = "assumption-agent-v2-challenger"
        if (
            _sha256_file(result_path) != row.get("sha256")
            or result.get("task_id") != f"{row.get('family')}/{item_id}"
            or result.get("agent") != common["agent_id"]
            or result.get("model") != common["model"]
            or result.get("trial_id") != expected_trial_id
            or result.get("trial_name") != expected_trial_id
            or result.get("skill_config") != expected_skill_config
            or result.get("passed") is not False
        ):
            raise PermissionError("diagnostic prior result drifted")
        if arm == "raw":
            valid = pair.get("baseline_valid")
            success = pair.get("baseline_success")
        else:
            valid = pair.get("candidate_valid")
            success = pair.get("candidate_success")
        if valid is not True or success is not False:
            raise PermissionError("diagnostic prior outcome evidence drifted")
    expected_prior_keys = {
        (arm, item_id)
        for item_id in item_ids
        for arm in ("raw", "g1_without_prompt_delivery", "g2_without_prompt_delivery")
    }
    if prior_keys != expected_prior_keys:
        raise PermissionError("diagnostic prior-result coverage drifted")
    _artifact_paths(project_root, payload)
    return project_root, payload


def _validate_environment(
    *,
    env_file: Path,
    protocol: PaperProtocol,
) -> None:
    load_dotenv(env_file)
    status = map_legacy_model_env()
    if not status["api_key_present"] or not status["base_url_present"]:
        raise RuntimeError("profile consumption provider is not configured")
    os.environ["ASSUMPTION_V2_SKILLLEARN_PROVIDER_MODE"] = str(
        protocol.payload["trial_provider_mode"]
    )
    os.environ["ASSUMPTION_V2_API_ALLOWED_IPV4S"] = ",".join(
        str(value) for value in protocol.payload["provider_endpoint_ipv4s"]
    )
    os.environ["ASSUMPTION_V2_SKILLLEARN_CACHE_ONLY"] = "1"
    os.environ["ASSUMPTION_V2_TRIAL_NETWORK_BYTE_LIMIT"] = str(
        protocol.payload["execution"]["trial_network_byte_limit"]
    )
    if (
        configured_model() != protocol.payload["model"]
        or configured_skilllearn_provider_mode()
        != protocol.payload["trial_provider_mode"]
        or configured_api_origin() != protocol.payload["provider_endpoint_origin"]
        or configured_trial_network_byte_limit()
        != protocol.payload["execution"]["trial_network_byte_limit"]
    ):
        raise RuntimeError("profile consumption provider contract drifted")


def _historical_comparison(
    *,
    pair: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "stored_raw": {
            "request_hash": pair["baseline_request_hash"],
            "observation_hash": pair["baseline_observation_hash"],
            "evidence_hash": pair["baseline_evidence_hash"],
            "success": pair["baseline_success"],
            "valid": pair["baseline_valid"],
            "cost": pair["baseline_cost"],
        },
        "same_generation_without_prompt_delivery": {
            "request_hash": pair["candidate_request_hash"],
            "observation_hash": pair["candidate_observation_hash"],
            "success": pair["candidate_success"],
            "valid": pair["candidate_valid"],
            "cost": pair["candidate_cost"],
        },
    }


def _safe_observation_row(
    *,
    frozen_row: Mapping[str, Any],
    item: Mapping[str, Any],
    observation: SkillLearnTrialObservation,
    historical_pair: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "generation": frozen_row["generation"],
        "item_id_hash": item["item_id_hash"],
        "family_hash": item["family_hash"],
        "old_request_hash": frozen_row["old_request_hash"],
        "new_request_hash": observation.request.request_hash,
        "observation_hash": observation.observation_hash,
        "valid": observation.valid,
        "success": observation.success,
        "score": observation.score,
        "metrics": dict(sorted(observation.metrics.items())),
        "error_type": observation.error_type,
        "total_tokens": observation.total_tokens,
        "steps": observation.steps,
        "duration_seconds": observation.duration_seconds,
        "provider_fingerprint": observation.provider_fingerprint,
        "fairness_fingerprint": observation.fairness_fingerprint,
        "prebuilt_image_id": observation.prebuilt_image_id,
        "prebuilt_image_key": observation.prebuilt_image_key,
        "prebuilt_cache_reused": observation.prebuilt_cache_reused,
        "offline_verifier_profile_id": observation.offline_verifier_profile_id,
        "offline_verifier_runtime_key": observation.offline_verifier_runtime_key,
        "installed_skill_source_receipt_hash": (
            observation.installed_skill_source_receipt_hash
        ),
        "runtime_profile_prompt_delivery_policy": (
            observation.runtime_profile_prompt_delivery_policy
        ),
        "runtime_profile_prompt_injection_receipt_hash": (
            observation.runtime_profile_prompt_injection_receipt_hash
        ),
        "runtime_profile_effective_prompt_sha256": (
            observation.runtime_profile_effective_prompt_sha256
        ),
        "historical_comparison": _historical_comparison(pair=historical_pair),
        "raw_trial_artifacts_local_only": observation.raw_trial_artifacts_persisted,
        "canonical_raw_content_persisted": False,
    }


def _receipt_binding_passed(
    row: Mapping[str, Any],
    item: Mapping[str, Any],
    frozen_request: Mapping[str, Any],
) -> bool:
    return bool(
        row.get("valid") is True
        and row.get("new_request_hash") == frozen_request["new_request_hash"]
        and row.get("provider_fingerprint") == item["provider_fingerprint"]
        and row.get("fairness_fingerprint") == item["fairness_fingerprint"]
        and row.get("prebuilt_cache_reused") is True
        and row.get("prebuilt_image_id") == item["prebuilt_image_id"]
        and row.get("prebuilt_image_key") == item["prebuilt_image_key"]
        and row.get("offline_verifier_profile_id")
        == item["offline_verifier_profile_id"]
        and row.get("offline_verifier_runtime_key")
        == item["offline_verifier_runtime_key"]
        and row.get("installed_skill_source_receipt_hash")
        == frozen_request["skill_source_receipt_hash"]
        and row.get("runtime_profile_prompt_delivery_policy")
        == RUNTIME_PROFILE_PROMPT_DELIVERY_VERSION
        and row.get("runtime_profile_prompt_injection_receipt_hash")
        and row.get("runtime_profile_effective_prompt_sha256")
    )


def _scoped_frozen_task_inputs(
    frozen: FrozenTaskInputClosure,
    item_rows: Sequence[Mapping[str, Any]],
) -> FrozenTaskInputClosure:
    selected_hashes = {
        str(row["item_id_hash"])
        for row in item_rows
        if row.get("task_input_closure_hash") is not None
    }
    ledger = {
        item_hash: frozen.ledger_by_item_hash[item_hash]
        for item_hash in sorted(selected_hashes)
        if item_hash in frozen.ledger_by_item_hash
    }
    if set(ledger) != selected_hashes:
        raise PermissionError("diagnostic scoped task-input ledger is incomplete")
    return FrozenTaskInputClosure(
        source=frozen.source,
        receipt=frozen.receipt,
        receipt_path=frozen.receipt_path,
        ledger_by_item_hash=ledger,
    )


def _preflight_trial_assets(
    *,
    benchmark_root: Path,
    protocol: PaperProtocol,
    frozen_task_inputs: Any,
    prewarm_payload: Mapping[str, Any],
    item_rows: Sequence[Mapping[str, Any]],
    task_input_cache_root: str | Path | None,
) -> None:
    """Resolve every local image and verifier before any model trial starts."""

    memory_sink = MemoryEventSink()
    cache = FrozenTaskInputPrebuiltImageCache(
        benchmark_root,
        frozen_task_inputs=frozen_task_inputs,
        expected_prewarm_rows=expected_prewarm_closure_rows(prewarm_payload),
        cache_only=True,
        event_sink=memory_sink,
        task_input_cache_root=task_input_cache_root,
    )
    circuit = SkillLearnProviderCircuit()
    limiter = SkillLearnModelInferenceLimiter(
        int(protocol.payload["execution"]["model_inference_slots"])
    )
    backends = tuple(
        SkillLearnSubprocessBackend(
            benchmark_root,
            agent_id=str(protocol.payload["agent_id"]),
            model=str(protocol.payload["model"]),
            max_steps=int(protocol.payload["max_steps"]),
            provider_mode=str(protocol.payload["trial_provider_mode"]),
            record_upstream=False,
            prebuilt_cache=cache,
            provider_circuit=circuit,
            model_inference_limiter=limiter,
            codex_agent_execution_policy=protocol.codex_agent_execution_policy,
            event_sink=memory_sink,
        )
        for _ in item_rows
    )

    def resolve(
        spec: tuple[Mapping[str, Any], SkillLearnSubprocessBackend],
    ) -> tuple[Mapping[str, Any], Any, Any]:
        item, backend = spec
        image, runtime = backend.prewarm_trial_environment(
            family=str(item["family"]),
            item_id=str(item["item_id"]),
            trace_id=f"typed-profile-consumption-preflight:{item['item_id_hash'][:20]}",
        )
        return item, image, runtime

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=_EXPECTED_ITEM_COUNT
    ) as executor:
        resolved = tuple(executor.map(resolve, zip(item_rows, backends, strict=True)))
    for item, image, runtime in resolved:
        if (
            image.image_id != item["prebuilt_image_id"]
            or image.cache_key != item["prebuilt_image_key"]
            or image.reused is not True
            or runtime is None
            or runtime.profile.profile_id != item["offline_verifier_profile_id"]
            or runtime.runtime_key != item["offline_verifier_runtime_key"]
        ):
            raise PermissionError("diagnostic local asset preflight drifted")
    if limiter.maximum_active != 0 or circuit.error_type is not None:
        raise PermissionError("diagnostic preflight crossed the no-model boundary")


def run_profile_consumption_diagnostic(
    *,
    preregistration_path: str | Path,
    env_file: str | Path,
    task_input_cache_root: str | Path | None = None,
) -> dict[str, Any]:
    project_root, preregistration = _read_preregistration(
        preregistration_path
    )
    paths = _artifact_paths(project_root, preregistration)
    completion_paths = (
        paths["report"],
        paths["events"],
        paths["completion_lock"],
        paths["result_receipt"],
    )
    if any(path.exists() for path in completion_paths):
        raise PermissionError("profile consumption diagnostic already started")
    run_root = paths["run_root"]
    if run_root.exists() and any(run_root.iterdir()):
        raise PermissionError("profile consumption run root is not fresh")

    protocol = PaperProtocol.read(
        _source_path(project_root, preregistration, "paper_protocol")
    )
    manifest = SplitManifest.read(
        _source_path(project_root, preregistration, "manifest")
    )
    _validate_environment(
        env_file=Path(env_file).expanduser().resolve(strict=True),
        protocol=protocol,
    )
    prewarm_payload = _read_json(
        _source_path(project_root, preregistration, "prewarm_receipt")
    )
    frozen_task_inputs = load_frozen_task_input_closure(
        protocol.payload,
        project_root=project_root,
    )
    if frozen_task_inputs is None:
        raise PermissionError("diagnostic task-input closure is not frozen")
    validate_development_prewarm_receipt(
        prewarm_payload,
        manifest=manifest,
        expected_version=str(
            protocol.payload["execution"]["development_prewarm"]
        ),
        frozen_task_inputs=frozen_task_inputs,
    )
    prewarm_by_item_hash = {
        str(row["item_id_hash"]): row for row in prewarm_payload["items"]
    }
    item_by_id = {
        str(row["item_id"]): row for row in preregistration["validation_items"]
    }
    for item in item_by_id.values():
        prewarm = prewarm_by_item_hash.get(str(item["item_id_hash"]))
        if not isinstance(prewarm, Mapping) or any(
            prewarm.get(key) != item.get(key)
            for key in (
                "family_hash",
                "prebuilt_image_id",
                "prebuilt_image_key",
                "task_input_closure_hash",
                "task_input_integrity_receipt_hash",
            )
        ):
            raise PermissionError("diagnostic item differs from frozen prewarm")

    benchmark_root = _source_path(
        project_root,
        preregistration,
        "benchmark_anchor_file",
    ).parent
    scoped_task_inputs = _scoped_frozen_task_inputs(
        frozen_task_inputs,
        tuple(item_by_id.values()),
    )
    _preflight_trial_assets(
        benchmark_root=benchmark_root,
        protocol=protocol,
        frozen_task_inputs=scoped_task_inputs,
        prewarm_payload=prewarm_payload,
        item_rows=tuple(item_by_id.values()),
        task_input_cache_root=task_input_cache_root,
    )

    run_root.mkdir(parents=True, exist_ok=True)
    sink = JsonlEventSink(paths["events"])
    cache = FrozenTaskInputPrebuiltImageCache(
        benchmark_root,
        frozen_task_inputs=scoped_task_inputs,
        expected_prewarm_rows=expected_prewarm_closure_rows(prewarm_payload),
        cache_only=True,
        event_sink=sink,
        task_input_cache_root=task_input_cache_root,
    )
    circuit = SkillLearnProviderCircuit()
    limiter = SkillLearnModelInferenceLimiter(
        int(protocol.payload["execution"]["model_inference_slots"])
    )
    backends = tuple(
        SkillLearnSubprocessBackend(
            benchmark_root,
            agent_id=str(protocol.payload["agent_id"]),
            model=str(protocol.payload["model"]),
            max_steps=int(protocol.payload["max_steps"]),
            provider_mode=str(protocol.payload["trial_provider_mode"]),
            trials_dir=run_root / "upstream_trials",
            record_upstream=True,
            prebuilt_cache=cache,
            provider_circuit=circuit,
            model_inference_limiter=limiter,
            codex_agent_execution_policy=protocol.codex_agent_execution_policy,
            event_sink=sink,
        )
        for _ in range(_EXPECTED_NEW_TRIAL_COUNT)
    )
    trial_rows = tuple(
        sorted(
            preregistration["trial_requests"],
            key=lambda row: (int(row["generation"]), str(row["item_id"])),
        )
    )
    pair_events = _pair_events(
        _source_path(project_root, preregistration, "v320_event_ledger"),
        trial_rows,
    )
    specs: list[
        tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
            SkillLearnTrialRequest,
            Path,
            SkillLearnSubprocessBackend,
        ]
    ] = []
    for index, frozen_row in enumerate(trial_rows):
        request, source = _request_from_row(
            project_root=project_root,
            preregistration=preregistration,
            protocol=protocol,
            manifest=manifest,
            row=frozen_row,
            delivery_enabled=True,
        )
        key = (int(frozen_row["generation"]), str(frozen_row["item_id"]))
        specs.append(
            (
                frozen_row,
                item_by_id[str(frozen_row["item_id"])],
                pair_events[key],
                request,
                source,
                backends[index],
            )
        )

    def execute_spec(
        spec: tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
            SkillLearnTrialRequest,
            Path,
            SkillLearnSubprocessBackend,
        ],
    ) -> tuple[
        Mapping[str, Any],
        Mapping[str, Any],
        Mapping[str, Any],
        SkillLearnTrialObservation,
    ]:
        frozen_row, item, pair, request, source, backend = spec
        observation = backend.run(
            request,
            skill_source_dir=source,
            trace_id=(
                f"typed-profile-consumption:g{frozen_row['generation']}:"
                f"{request.request_hash[:20]}:single-attempt"
            ),
        )
        return frozen_row, item, pair, observation

    executed: list[
        tuple[
            Mapping[str, Any],
            Mapping[str, Any],
            Mapping[str, Any],
            SkillLearnTrialObservation,
        ]
    ] = []
    execution_errors: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=_EXPECTED_NEW_TRIAL_COUNT
    ) as executor:
        futures = {
            executor.submit(execute_spec, spec): spec[0] for spec in specs
        }
        for future in concurrent.futures.as_completed(futures):
            frozen_row = futures[future]
            try:
                executed.append(future.result())
            except Exception as exc:  # terminal evidence; no automatic retry
                execution_errors.append(
                    {
                        "error_type": type(exc).__name__,
                        "generation": int(frozen_row["generation"]),
                        "item_id_hash": stable_hash(
                            {"item_id": str(frozen_row["item_id"])}
                        ),
                        "new_request_hash": str(frozen_row["new_request_hash"]),
                    }
                )
    execution_errors.sort(
        key=lambda row: (int(row["generation"]), str(row["item_id_hash"]))
    )
    rows = [
        _safe_observation_row(
            frozen_row=frozen_row,
            item=item,
            observation=observation,
            historical_pair=pair,
        )
        for frozen_row, item, pair, observation in executed
    ]
    rows.sort(key=lambda row: (int(row["generation"]), str(row["item_id_hash"])))
    frozen_by_key = {
        (int(row["generation"]), stable_hash({"item_id": str(row["item_id"])})): row
        for row in trial_rows
    }
    for row in rows:
        item = next(
            item
            for item in item_by_id.values()
            if item["item_id_hash"] == row["item_id_hash"]
        )
        frozen = frozen_by_key[
            (int(row["generation"]), str(row["item_id_hash"]))
        ]
        binding = _receipt_binding_passed(row, item, frozen)
        row["runtime_receipt_binding_passed"] = binding
        historical = row["historical_comparison"]
        raw = historical["stored_raw"]
        same_generation = historical["same_generation_without_prompt_delivery"]
        row["utility_signal_against_stored_raw"] = bool(
            binding
            and row["success"] is True
            and raw["valid"] is True
            and raw["success"] is False
        )
        row["delivery_delta_signal_against_same_generation"] = bool(
            binding
            and row["success"] is True
            and same_generation["valid"] is True
            and same_generation["success"] is False
        )
    receipt_binding_passed = len(rows) == _EXPECTED_NEW_TRIAL_COUNT and all(
        row["runtime_receipt_binding_passed"] is True for row in rows
    )
    all_valid = len(rows) == _EXPECTED_NEW_TRIAL_COUNT and all(
        row["valid"] is True for row in rows
    )
    agent_launch_count = sum(
        json.loads(raw).get("event") == "skilllearn_agent_slot_acquired"
        for raw in paths["events"].read_text(encoding="utf-8").splitlines()
    )
    utility_signals = sum(
        row["utility_signal_against_stored_raw"] is True for row in rows
    )
    delivery_delta_signals = sum(
        row["delivery_delta_signal_against_same_generation"] is True
        for row in rows
    )
    generation_summaries = {
        f"g{generation}": {
            "valid": sum(
                row["valid"] is True
                for row in rows
                if row["generation"] == generation
            ),
            "successes": sum(
                row["success"] is True
                for row in rows
                if row["generation"] == generation
            ),
            "utility_signals_against_stored_raw": sum(
                row["utility_signal_against_stored_raw"] is True
                for row in rows
                if row["generation"] == generation
            ),
            "delivery_delta_signals_against_same_generation": sum(
                row["delivery_delta_signal_against_same_generation"] is True
                for row in rows
                if row["generation"] == generation
            ),
        }
        for generation in (1, 2)
    }
    report = {
        "report_version": PROFILE_CONSUMPTION_REPORT_VERSION,
        "diagnostic_policy": PROFILE_CONSUMPTION_DIAGNOSTIC_VERSION,
        "preregistration_sha256": _sha256_file(
            Path(preregistration_path).expanduser().resolve(strict=True)
        ),
        "manifest_hash": manifest.manifest_hash,
        "delivery_mode": RUNTIME_PROFILE_PROMPT_DELIVERY_VERSION,
        "execution": {
            **dict(preregistration["execution_contract"]),
            "agent_launch_count": agent_launch_count,
            "maximum_parallel_agent_launches": limiter.maximum_active,
            "new_trial_attempts_scheduled": len(specs),
            "new_trial_observations_completed": len(rows),
            "new_trials_valid": sum(row["valid"] is True for row in rows),
            "runtime_receipt_binding_passed": receipt_binding_passed,
            "local_raw_trial_artifacts_retained_for_posthoc_analysis": True,
            "canonical_raw_content_persisted": False,
            "secret_value_persisted": False,
        },
        "generation_summaries": generation_summaries,
        "execution_errors": execution_errors,
        "rows": rows,
        "diagnostic_completed": bool(
            all_valid
            and receipt_binding_passed
            and agent_launch_count == _EXPECTED_NEW_TRIAL_COUNT
            and not execution_errors
        ),
        "utility_signal_count": utility_signals,
        "utility_signal_observed": bool(utility_signals),
        "delivery_delta_signal_count": delivery_delta_signals,
        "delivery_delta_signal_observed": bool(delivery_delta_signals),
        "semantic_profile_consumption_claimed": False,
        "task_utility_causal_claimed": False,
        "incumbent_created": False,
        "promotion_evaluated": False,
        "fresh_validation": False,
        "sealed_test_bytes_exposed_to_model": False,
        "sealed_test_scoring_performed": False,
        "test_infrastructure_metadata_inspected": True,
        "test_task_input_bytes_inspected": False,
        "test_trial_executed": False,
        "hipporag_run": False,
        "hipporag_omission_reason": (
            "this isolates runtime-profile delivery against frozen executable "
            "SkillLearnBench file-task RAW controls; HippoRAG is not an executable "
            "file-task arm"
        ),
        "raw_content_persisted": False,
    }
    report["report_hash"] = stable_hash(report)
    _write_json(paths["report"], report)
    sink.emit(
        Event(
            event="typed_profile_consumption_diagnostic_completed",
            stage="benchmark.skilllearn.profile_consumption",
            trace_id=report["report_hash"][:20],
            payload={
                "diagnostic_completed": report["diagnostic_completed"],
                "agent_launch_count": agent_launch_count,
                "new_trial_attempt_count": len(specs),
                "valid_count": sum(row["valid"] is True for row in rows),
                "utility_signal_count": utility_signals,
                "delivery_delta_signal_count": delivery_delta_signals,
                "runtime_receipt_binding_passed": receipt_binding_passed,
                "report_hash": report["report_hash"],
                "sealed_test_scoring_performed": False,
                "test_trial_executed": False,
                "raw_content_persisted": False,
            },
        )
    )
    lock = {
        "completion_lock_version": "typed_profile_consumption_completion_lock_v1",
        "diagnostic_policy": PROFILE_CONSUMPTION_DIAGNOSTIC_VERSION,
        "diagnostic_completed": report["diagnostic_completed"],
        "report_hash": report["report_hash"],
        "agent_launch_count": agent_launch_count,
        "new_trial_attempt_count": len(specs),
        "promotion_evaluated": False,
        "sealed_test_scoring_performed": False,
        "test_trial_executed": False,
        "raw_content_persisted": False,
    }
    lock["completion_lock_hash"] = stable_hash(lock)
    _write_json(paths["completion_lock"], lock)
    canonical_artifacts = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in (paths["report"], paths["events"], paths["completion_lock"])
    ]
    result = {
        "result_receipt_version": PROFILE_CONSUMPTION_RESULT_VERSION,
        "diagnostic_policy": PROFILE_CONSUMPTION_DIAGNOSTIC_VERSION,
        "diagnostic_completed": report["diagnostic_completed"],
        "report_hash": report["report_hash"],
        "utility_signal_count": utility_signals,
        "utility_signal_observed": bool(utility_signals),
        "delivery_delta_signal_count": delivery_delta_signals,
        "delivery_delta_signal_observed": bool(delivery_delta_signals),
        "agent_launch_count": agent_launch_count,
        "new_trial_attempt_count": len(specs),
        "valid_trial_count": sum(row["valid"] is True for row in rows),
        "runtime_receipt_binding_passed": receipt_binding_passed,
        "fresh_validation": False,
        "incumbent_created": False,
        "promotion_evaluated": False,
        "sealed_test_scoring_performed": False,
        "test_trial_executed": False,
        "canonical_artifacts": canonical_artifacts,
        "raw_content_persisted": False,
    }
    result["result_receipt_hash"] = stable_hash(result)
    _write_json(paths["result_receipt"], result)
    if not report["diagnostic_completed"]:
        raise RuntimeError("typed profile consumption diagnostic is incomplete")
    return report


def verify_existing_profile_consumption_diagnostic(
    *,
    preregistration_path: str | Path,
) -> dict[str, Any]:
    project_root, preregistration = _read_preregistration(
        preregistration_path
    )
    paths = _artifact_paths(project_root, preregistration)
    report = _read_json(paths["report"])
    lock = _read_json(paths["completion_lock"])
    result = _read_json(paths["result_receipt"])
    report_hash = report.get("report_hash")
    if report_hash != stable_hash(
        {key: value for key, value in report.items() if key != "report_hash"}
    ):
        raise PermissionError("profile consumption report hash drifted")
    manifest = SplitManifest.read(
        _source_path(project_root, preregistration, "manifest")
    )
    execution = report.get("execution")
    rows = report.get("rows")
    if (
        report.get("report_version") != PROFILE_CONSUMPTION_REPORT_VERSION
        or report.get("diagnostic_policy") != PROFILE_CONSUMPTION_DIAGNOSTIC_VERSION
        or report.get("preregistration_sha256")
        != _sha256_file(Path(preregistration_path).expanduser().resolve(strict=True))
        or report.get("manifest_hash") != manifest.manifest_hash
        or report.get("delivery_mode") != RUNTIME_PROFILE_PROMPT_DELIVERY_VERSION
        or report.get("diagnostic_completed") is not True
        or not isinstance(execution, Mapping)
        or any(
            execution.get(key) != value
            for key, value in _EXPECTED_EXECUTION_CONTRACT.items()
        )
        or execution.get("new_trial_attempts_scheduled")
        != _EXPECTED_NEW_TRIAL_COUNT
        or execution.get("new_trial_observations_completed")
        != _EXPECTED_NEW_TRIAL_COUNT
        or execution.get("new_trials_valid") != _EXPECTED_NEW_TRIAL_COUNT
        or execution.get("agent_launch_count") != _EXPECTED_NEW_TRIAL_COUNT
        or execution.get("runtime_receipt_binding_passed") is not True
        or report.get("execution_errors") != []
        or not isinstance(rows, list)
        or len(rows) != _EXPECTED_NEW_TRIAL_COUNT
        or report.get("sealed_test_bytes_exposed_to_model") is not False
        or report.get("sealed_test_scoring_performed") is not False
        or report.get("test_task_input_bytes_inspected") is not False
        or report.get("test_trial_executed") is not False
    ):
        raise PermissionError("profile consumption completion receipt drifted")

    trial_rows = preregistration["trial_requests"]
    pairs = _pair_events(
        _source_path(project_root, preregistration, "v320_event_ledger"),
        trial_rows,
    )
    frozen_by_key = {
        (int(row["generation"]), stable_hash({"item_id": str(row["item_id"])})): row
        for row in trial_rows
    }
    item_by_hash = {
        str(row["item_id_hash"]): row
        for row in preregistration["validation_items"]
    }
    seen: set[tuple[int, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise PermissionError("profile consumption report row is malformed")
        key = (int(row.get("generation") or 0), str(row.get("item_id_hash") or ""))
        frozen = frozen_by_key.get(key)
        item = item_by_hash.get(key[1])
        if frozen is None or item is None or key in seen:
            raise PermissionError("profile consumption report row identity drifted")
        seen.add(key)
        expected_historical = _historical_comparison(
            pair=pairs[(key[0], str(frozen["item_id"]))]
        )
        binding = _receipt_binding_passed(row, item, frozen)
        raw = expected_historical["stored_raw"]
        same_generation = expected_historical[
            "same_generation_without_prompt_delivery"
        ]
        expected_utility = bool(
            binding
            and row.get("success") is True
            and raw["valid"] is True
            and raw["success"] is False
        )
        expected_delivery = bool(
            binding
            and row.get("success") is True
            and same_generation["valid"] is True
            and same_generation["success"] is False
        )
        if (
            row.get("old_request_hash") != frozen["old_request_hash"]
            or row.get("new_request_hash") != frozen["new_request_hash"]
            or row.get("valid") is not True
            or row.get("historical_comparison") != expected_historical
            or row.get("runtime_receipt_binding_passed") is not binding
            or binding is not True
            or row.get("utility_signal_against_stored_raw") is not expected_utility
            or row.get("delivery_delta_signal_against_same_generation")
            is not expected_delivery
        ):
            raise PermissionError("profile consumption report row semantics drifted")
    if seen != set(frozen_by_key):
        raise PermissionError("profile consumption report row coverage drifted")

    utility_count = sum(
        row["utility_signal_against_stored_raw"] is True for row in rows
    )
    delivery_count = sum(
        row["delivery_delta_signal_against_same_generation"] is True
        for row in rows
    )
    expected_summaries = {
        f"g{generation}": {
            "valid": sum(
                row["valid"] is True
                for row in rows
                if row["generation"] == generation
            ),
            "successes": sum(
                row["success"] is True
                for row in rows
                if row["generation"] == generation
            ),
            "utility_signals_against_stored_raw": sum(
                row["utility_signal_against_stored_raw"] is True
                for row in rows
                if row["generation"] == generation
            ),
            "delivery_delta_signals_against_same_generation": sum(
                row["delivery_delta_signal_against_same_generation"] is True
                for row in rows
                if row["generation"] == generation
            ),
        }
        for generation in (1, 2)
    }
    if (
        report.get("generation_summaries") != expected_summaries
        or report.get("utility_signal_count") != utility_count
        or report.get("utility_signal_observed") is not bool(utility_count)
        or report.get("delivery_delta_signal_count") != delivery_count
        or report.get("delivery_delta_signal_observed") is not bool(delivery_count)
    ):
        raise PermissionError("profile consumption report summary drifted")

    event_rows = [
        json.loads(raw)
        for raw in paths["events"].read_text(encoding="utf-8").splitlines()
    ]
    for event in event_rows:
        if not isinstance(event, Mapping):
            raise PermissionError("profile consumption event is malformed")
        expected_event = Event(
            event=str(event.get("event") or ""),
            stage=str(event.get("stage") or ""),
            trace_id=str(event.get("trace_id") or ""),
            payload=dict(event.get("payload") or {}),
        ).to_dict()
        if dict(event) != expected_event:
            raise PermissionError("profile consumption event receipt drifted")
    completed_trials = [
        event
        for event in event_rows
        if event.get("event") == "skilllearn_trial_completed"
        and event.get("payload", {}).get("request_hash")
        in {str(row["new_request_hash"]) for row in trial_rows}
    ]
    terminal_events = [
        event
        for event in event_rows
        if event.get("event") == "typed_profile_consumption_diagnostic_completed"
    ]
    slot_count = sum(
        event.get("event") == "skilllearn_agent_slot_acquired"
        for event in event_rows
    )
    if (
        len(completed_trials) != _EXPECTED_NEW_TRIAL_COUNT
        or {
            str(event["payload"]["request_hash"]) for event in completed_trials
        }
        != {str(row["new_request_hash"]) for row in trial_rows}
        or len(terminal_events) != 1
        or terminal_events[0]["payload"].get("report_hash") != report_hash
        or slot_count != _EXPECTED_NEW_TRIAL_COUNT
        or slot_count != execution.get("agent_launch_count")
    ):
        raise PermissionError("profile consumption event coverage drifted")

    expected_lock_keys = {
        "agent_launch_count",
        "completion_lock_hash",
        "completion_lock_version",
        "diagnostic_completed",
        "diagnostic_policy",
        "new_trial_attempt_count",
        "promotion_evaluated",
        "raw_content_persisted",
        "report_hash",
        "sealed_test_scoring_performed",
        "test_trial_executed",
    }
    if (
        set(lock) != expected_lock_keys
        or lock.get("completion_lock_version")
        != "typed_profile_consumption_completion_lock_v1"
        or lock.get("diagnostic_policy") != PROFILE_CONSUMPTION_DIAGNOSTIC_VERSION
        or lock.get("report_hash") != report_hash
        or lock.get("diagnostic_completed") is not True
        or lock.get("agent_launch_count") != _EXPECTED_NEW_TRIAL_COUNT
        or lock.get("new_trial_attempt_count") != _EXPECTED_NEW_TRIAL_COUNT
        or lock.get("completion_lock_hash")
        != stable_hash(
            {key: value for key, value in lock.items() if key != "completion_lock_hash"}
        )
    ):
        raise PermissionError("profile consumption lock receipt drifted")
    expected_artifacts = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in (paths["report"], paths["events"], paths["completion_lock"])
    ]
    expected_result_keys = {
        "agent_launch_count",
        "canonical_artifacts",
        "delivery_delta_signal_count",
        "delivery_delta_signal_observed",
        "diagnostic_completed",
        "diagnostic_policy",
        "fresh_validation",
        "incumbent_created",
        "new_trial_attempt_count",
        "promotion_evaluated",
        "raw_content_persisted",
        "report_hash",
        "result_receipt_hash",
        "result_receipt_version",
        "runtime_receipt_binding_passed",
        "sealed_test_scoring_performed",
        "test_trial_executed",
        "utility_signal_count",
        "utility_signal_observed",
        "valid_trial_count",
    }
    result_hash = result.get("result_receipt_hash")
    if (
        set(result) != expected_result_keys
        or result.get("result_receipt_version") != PROFILE_CONSUMPTION_RESULT_VERSION
        or result.get("diagnostic_policy") != PROFILE_CONSUMPTION_DIAGNOSTIC_VERSION
        or result.get("report_hash") != report_hash
        or result.get("diagnostic_completed") is not True
        or result.get("agent_launch_count") != _EXPECTED_NEW_TRIAL_COUNT
        or result.get("new_trial_attempt_count") != _EXPECTED_NEW_TRIAL_COUNT
        or result.get("valid_trial_count") != _EXPECTED_NEW_TRIAL_COUNT
        or result.get("runtime_receipt_binding_passed") is not True
        or result.get("utility_signal_count") != utility_count
        or result.get("delivery_delta_signal_count") != delivery_count
        or result.get("canonical_artifacts") != expected_artifacts
        or result_hash
        != stable_hash(
            {
                key: value
                for key, value in result.items()
                if key != "result_receipt_hash"
            }
        )
    ):
        raise PermissionError("profile consumption result receipt drifted")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run or verify the consumed-development runtime-profile consumption "
            "diagnostic."
        )
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path(
            "manifests/skilllearn_typed_profile_consumption_diagnostic_v1.json"
        ),
    )
    parser.add_argument("--env-file", type=Path, default=Path("../.env"))
    parser.add_argument("--task-input-cache-root", type=Path)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.verify_existing:
        report = verify_existing_profile_consumption_diagnostic(
            preregistration_path=args.preregistration,
        )
    else:
        report = run_profile_consumption_diagnostic(
            preregistration_path=args.preregistration,
            env_file=args.env_file,
            task_input_cache_root=args.task_input_cache_root,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
