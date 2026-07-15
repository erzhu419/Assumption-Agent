#!/usr/bin/env python3
"""Finalize already-completed offline verifier evidence without rerunning it."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

_PROJECT_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_IMPORT_ROOT))

from assumption_agent.benchmarks.financial_semantic_fresh_runner_v1 import (
    ACTIVE_FRESH_ITEM_ID,
    OFFICIAL_HIPPORAG_STATUS,
    _prepare_runtime_assets_v1,
    _worker_artifact_closure,
)
from assumption_agent.benchmarks.offline_verifier import (
    offline_verifier_profile_for_family,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnSubprocessBackend,
)
from assumption_agent.events import Event, JsonlEventSink
from assumption_agent.models import stable_hash
from scripts.recover_financial_semantic_fresh_v1 import (
    CONTINUATION_PREREG_VERSION,
    EXPECTED_ORPHANS,
    EXPECTED_PHYSICAL,
    ORIGINAL_EVENTS,
    ORIGINAL_PREFLIGHT,
    RECOVERY_FAILURE,
    RECOVERY_REPORT,
    RECOVERY_VERSION,
    SCRIPT_RELATIVE_PATH,
    SEMANTIC_STAGE,
    SESSION_RECEIPT,
    VERIFIER_STAGE,
    _atomic_json,
    _base_result,
    _configure_nonsecret_environment,
    _container_name,
    _git,
    _git_bytes,
    _idle_process_receipt,
    _load_context,
    _observation_for_work,
    _read_json,
    _self_hashed,
    _sha256,
    _trial_path,
    _validated_hashed_json,
)


FINALIZER_VERSION = "financial_semantic_scheduler_loss_recovery_finalizer_v1"
FINALIZER_EVENTS = "recovery.finalization.events.jsonl"


class FinalizationError(RuntimeError):
    pass


def _emit(sink: JsonlEventSink, event: str, payload: Mapping[str, Any]) -> None:
    sink.emit(
        Event(
            event=event,
            stage="benchmark.skilllearn.scheduler_loss_finalization_v1",
            trace_id="financial-semantic-fresh-finalization-v1",
            payload=dict(payload),
        )
    )


def _verify_committed_recovery_source(
    project: Path, manifest: Mapping[str, Any]
) -> None:
    prefix = _git(project, "rev-parse", "--show-prefix")
    content = _git_bytes(
        project,
        "show",
        (
            f"{manifest['recovery_source_commit']}:"
            f"{prefix}{SCRIPT_RELATIVE_PATH}"
        ),
    )
    if hashlib.sha256(content).hexdigest() != manifest["recovery_script_sha256"]:
        raise FinalizationError("measurement recovery source binding drifted")


def _started_stage_hash_from_completed(stage: Mapping[str, Any]) -> str:
    keys = (
        "stage_version",
        "started_at_utc",
        "work_unit_hash",
        "request_hash",
        "container_name_hash",
        "tests_content_hash",
        "verifier_profile_id",
        "verifier_profile_hash",
        "verifier_command_hash",
        "model_calls",
        "online_judge_calls",
    )
    body = {key: stage[key] for key in keys}
    body["status"] = "started"
    return stable_hash(body)


def _validate_completed_measurement(
    *, context: Mapping[str, Any], measurement_manifest_path: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    manifest = _validated_hashed_json(measurement_manifest_path, "manifest_hash")
    if (
        manifest.get("manifest_version") != CONTINUATION_PREREG_VERSION
        or manifest.get("recovery_version") != RECOVERY_VERSION
        or manifest.get("execution_plan_hash") != context["plan"].plan_hash
        or manifest.get("treatment_manifest_hash")
        != context["treatment"].manifest_hash
        or manifest.get("fresh_split_manifest_hash") != context["split"].manifest_hash
    ):
        raise FinalizationError("measurement continuation manifest drifted")
    _verify_committed_recovery_source(context["project"], manifest)
    if (
        _sha256(context["batch_root"] / ORIGINAL_EVENTS)
        != manifest["snapshot"]["original_event_ledger_sha256"]
        or _sha256(context["batch_root"] / ORIGINAL_PREFLIGHT)
        != manifest["snapshot"]["original_preflight_sha256"]
    ):
        raise FinalizationError("original batch evidence changed")
    session = _validated_hashed_json(
        context["batch_root"] / SESSION_RECEIPT, "session_hash"
    )
    if session.get("session_hash") != manifest.get("parent_session_hash"):
        raise FinalizationError("recovery session binding drifted")
    aggregation_failure = _validated_hashed_json(
        context["batch_root"] / RECOVERY_FAILURE, "report_hash"
    )

    semantic_work = next(
        work for work in context["plan"].physical_work_units if work.arm == "semantic"
    )
    semantic_stage = _validated_hashed_json(
        _trial_path(context, semantic_work) / "semantic_runtime" / SEMANTIC_STAGE,
        "stage_hash",
    )
    evidence = semantic_stage.get("runtime_evidence")
    if semantic_stage.get("status") != "completed" or not isinstance(evidence, dict):
        raise FinalizationError("semantic runtime completion is unavailable")
    evidence_body = dict(evidence)
    evidence_hash = evidence_body.pop("evidence_hash", None)
    if (
        evidence_hash != stable_hash(evidence_body)
        or evidence_hash
        != manifest["continuation_state"]["semantic_runtime_evidence_hash"]
        or evidence.get("plan_hash")
        != manifest["snapshot"]["semantic_pre_agent_binding"]["plan_hash"]
    ):
        raise FinalizationError("semantic runtime evidence drifted")

    prereg_rows = {
        row["work_unit_hash"]: row
        for row in manifest["continuation_state"]["verifier_rows"]
    }
    completed: dict[str, dict[str, Any]] = {}
    for work in context["plan"].physical_work_units:
        if work.work_unit_hash not in prereg_rows:
            continue
        stage = _validated_hashed_json(
            _trial_path(context, work) / "verifier" / VERIFIER_STAGE,
            "stage_hash",
        )
        profile = offline_verifier_profile_for_family(work.family)
        verifier_dir = _trial_path(context, work) / "verifier"
        if (
            stage.get("status") != "completed"
            or stage.get("work_unit_hash") != work.work_unit_hash
            or stage.get("request_hash") != work.request.request_hash
            or profile is None
            or stage.get("verifier_profile_hash") != profile.profile_hash
            or stage.get("verifier_command_hash")
            != stable_hash({"command": profile.verifier_command})
            or stage.get("verifier_exit") != 0
            or stage.get("reward") not in {0, 1}
            or _sha256(verifier_dir / "reward.txt") != stage.get("reward_sha256")
            or _sha256(verifier_dir / "ctrf.json") != stage.get("ctrf_sha256")
            or _started_stage_hash_from_completed(stage)
            != prereg_rows[work.work_unit_hash]["verifier_stage_hash"]
        ):
            raise FinalizationError("completed verifier evidence drifted")
        _idle_process_receipt(_container_name(work))
        completed[work.work_unit_hash] = stage
    if len(completed) != EXPECTED_ORPHANS:
        raise FinalizationError("completed verifier coverage is incomplete")
    return manifest, completed, semantic_stage, aggregation_failure


def _transient_terminal_reconciliation(
    *, trial_path: Path, observation: Any
) -> dict[str, Any] | None:
    if observation.error_type != "codex_turn_failed":
        return None
    receipt = _read_json(trial_path / "agent" / "codex_action_budget_receipt.json")
    trace = trial_path / "agent" / "codex.txt"
    error_count = 0
    reconnect_count = 0
    for raw in trace.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or row.get("type") != "error":
            continue
        error_count += 1
        message = str(row.get("message") or "").lower()
        if "reconnecting" in message and "stream disconnected" in message:
            reconnect_count += 1
    terminal_receipt_success = (
        receipt.get("turn_completed_count") == 1
        and receipt.get("turn_failed_count") == 0
        and receipt.get("agent_exit_code") == 0
        and receipt.get("token_usage_complete") is True
        and receipt.get("agent_processes_exit_confirmed") is True
    )
    return {
        "frozen_audit_error_type": observation.error_type,
        "trace_error_record_count": error_count,
        "transient_reconnect_error_count": reconnect_count,
        "turn_completed_count": receipt.get("turn_completed_count"),
        "turn_failed_count": receipt.get("turn_failed_count"),
        "agent_exit_code": receipt.get("agent_exit_code"),
        "terminal_action_budget_receipt_valid": terminal_receipt_success,
        "diagnosis": (
            "nonterminal_transport_reconnect_then_turn_completed"
            if terminal_receipt_success and reconnect_count == error_count == 1
            else "unreconciled_terminal_trace_error"
        ),
        "frozen_observation_overridden": False,
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project_root.resolve(strict=True)
    batch_root = args.batch_root.resolve(strict=True)
    context = _load_context(
        project=project,
        batch_root=batch_root,
        treatment_manifest=args.treatment_manifest,
    )
    manifest, stages, semantic_stage, aggregation_failure = (
        _validate_completed_measurement(
            context=context,
            measurement_manifest_path=args.measurement_manifest.resolve(strict=True),
        )
    )
    _configure_nonsecret_environment(context["protocol"])
    sink = JsonlEventSink(batch_root / FINALIZER_EVENTS)
    _emit(
        sink,
        "financial_semantic_recovery_finalization_started_v1",
        {
            "measurement_manifest_hash": manifest["manifest_hash"],
            "completed_verifier_count": len(stages),
            "verifier_calls_authorized": 0,
            "model_calls_authorized": 0,
        },
    )
    asset_root = batch_root / "recovery_finalization_asset_validation"
    asset_root.mkdir(exist_ok=True)
    assets = _prepare_runtime_assets_v1(
        project=project,
        destination=asset_root,
        benchmark_root=context["benchmark_root"],
        protocol=context["protocol"],
        split=context["split"],
        event_sink=sink,
        task_input_cache_root=None,
    )
    if (
        assets.preflight_report["report_hash"]
        != manifest["snapshot"]["original_preflight_report_hash"]
    ):
        raise FinalizationError("offline runtime asset validation drifted")
    common = {
        "agent_id": str(context["protocol"].payload["agent_id"]),
        "model": str(context["protocol"].payload["model"]),
        "max_steps": int(context["protocol"].payload["max_steps"]),
        "provider_mode": str(context["protocol"].payload["trial_provider_mode"]),
        "record_upstream": True,
        "prebuilt_cache": assets.prebuilt_cache,
        "offline_verifier_cache": assets.offline_cache,
        "provider_circuit": assets.provider_circuit,
        "model_inference_limiter": assets.model_limiter,
        "codex_agent_execution_policy": context["protocol"].codex_agent_execution_policy,
    }
    original_completions = {
        str(row["payload"]["request_hash"]): row["payload"]
        for row in (
            json.loads(raw)
            for raw in (batch_root / ORIGINAL_EVENTS).read_text(encoding="utf-8").splitlines()
        )
        if row.get("event") == "skilllearn_trial_completed"
    }
    installed_receipt = semantic_stage["installed_candidate_receipt"]
    semantic_evidence = semantic_stage["runtime_evidence"]
    physical_rows = []
    observations: dict[str, Any] = {}
    invalid_diagnostics = []
    for work in context["plan"].physical_work_units:
        trial_path = _trial_path(context, work)
        stage = stages.get(work.work_unit_hash)
        if stage is None:
            result = _read_json(trial_path / "result.json")
            duration = float(original_completions[work.request.request_hash]["duration_seconds"])
            recovery_kind = "original_precompleted"
        else:
            result = _base_result(work, stage)
            if work.arm == "semantic":
                result["installed_skill_source_receipt_hash"] = (
                    context["treatment"].external_skill_source_receipt_hash
                )
                result["installed_skill_tree_hash"] = installed_receipt[
                    "installed_tree_hash"
                ]
                result["installed_skill_destination_count"] = installed_receipt[
                    "destination_count"
                ]
            _atomic_json(trial_path / "result.json", result)
            duration = 0.0
            recovery_kind = "post_agent_offline_verifier_recovered"
        backend = SkillLearnSubprocessBackend(
            context["benchmark_root"],
            trials_dir=context["trials_root"] / work.work_unit_hash,
            event_sink=sink,
            **common,
        )
        image, verifier_runtime = backend.prewarm_trial_environment(
            family=work.family,
            item_id=work.item_id,
            trace_id=f"financial-finalization-asset:{work.work_unit_hash[:20]}",
        )
        observation, _ = _observation_for_work(
            context=context,
            work=work,
            backend=backend,
            prebuilt_image=image,
            verifier_runtime=verifier_runtime,
            result=result,
            duration_seconds=duration,
        )
        if stage is None and observation.observation_hash != original_completions[
            work.request.request_hash
        ]["observation_hash"]:
            raise FinalizationError("original precompleted observation was not reproduced")
        observations[work.work_unit_hash] = observation
        reconciliation = _transient_terminal_reconciliation(
            trial_path=trial_path,
            observation=observation,
        )
        if not observation.valid:
            invalid_diagnostics.append(
                {
                    "work_unit_hash": work.work_unit_hash,
                    "request_hash": work.request.request_hash,
                    "arm": work.arm,
                    "error_type": observation.error_type,
                    "reconciliation": reconciliation,
                }
            )
        physical_rows.append(
            {
                **work.safe_payload(),
                "observation": observation.to_dict(),
                "observation_hash": observation.observation_hash,
                "recovery_kind": recovery_kind,
                "duration_reconstructed": stage is None,
                "verifier_stage_hash": stage["stage_hash"] if stage else None,
                "semantic_runtime_evidence_hash": (
                    semantic_evidence["evidence_hash"] if work.arm == "semantic" else None
                ),
            }
        )

    raw_by_item = {
        work.item_id: observations[work.work_unit_hash]
        for work in context["plan"].physical_work_units
        if work.arm == "raw"
    }
    semantic_work = next(
        work for work in context["plan"].physical_work_units if work.arm == "semantic"
    )
    candidate = observations[semantic_work.work_unit_hash]
    active_raw = raw_by_item[ACTIVE_FRESH_ITEM_ID]
    if not active_raw.valid or not candidate.valid:
        raise FinalizationError("active paired measurement is invalid")
    projections = []
    for item_id in context["split"].item_ids:
        if item_id == ACTIVE_FRESH_ITEM_ID:
            continue
        raw = raw_by_item[item_id]
        projected = raw.as_variant(context["plan"].candidate_requests_by_item[item_id])
        projections.append(
            {
                "projection_policy": "exact_raw_inactive_route_projection_v1",
                "item_id_hash": stable_hash({"item_id": item_id}),
                "raw_observation_hash": raw.observation_hash,
                "candidate_request_hash": projected.request.request_hash,
                "projected_observation_hash": projected.observation_hash,
                "raw_success": raw.success,
                "projected_success": projected.success,
                "raw_valid": raw.valid,
                "projected_valid": projected.valid,
                "behavior_identical_by_predeclared_inactive_route": True,
                "model_calls": 0,
            }
        )
    raw_successes = sum(int(row.success) for row in raw_by_item.values())
    projected_candidate_successes = (
        raw_successes - int(active_raw.success) + int(candidate.success)
    )
    artifact_closure = _worker_artifact_closure(context["trials_root"])

    cleanup_rows = []
    orphan_works = [
        work
        for work in context["plan"].physical_work_units
        if work.work_unit_hash in stages
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=EXPECTED_ORPHANS) as executor:
        removals = {
            executor.submit(
                subprocess.run,
                ["docker", "rm", "-f", _container_name(work)],
                capture_output=True,
                text=True,
            ): work
            for work in orphan_works
        }
        for future, work in removals.items():
            completed = future.result()
            if completed.returncode != 0:
                raise FinalizationError("orphan container cleanup failed")
            cleanup_rows.append(
                {
                    "work_unit_hash": work.work_unit_hash,
                    "container_name_hash": stable_hash(
                        {"container_name": _container_name(work)}
                    ),
                    "removed": True,
                }
            )

    gain = int(candidate.success) - int(active_raw.success)
    body = {
        "report_version": FINALIZER_VERSION,
        "execution_completed": True,
        "scheduler_process_loss": True,
        "pristine_runner_completion": False,
        "post_agent_recovery_disclosed": True,
        "measurement_manifest_hash": manifest["manifest_hash"],
        "aggregation_failure_report_hash": aggregation_failure["report_hash"],
        "treatment_manifest_hash": context["treatment"].manifest_hash,
        "fresh_split_manifest_hash": context["split"].manifest_hash,
        "execution_plan_hash": context["plan"].plan_hash,
        "active_pair_evidence_valid": True,
        "full_physical_batch_frozen_audit_valid": not invalid_diagnostics,
        "invalid_physical_result_count": len(invalid_diagnostics),
        "invalid_physical_diagnostics": invalid_diagnostics,
        "physical_results": sorted(physical_rows, key=lambda row: row["work_unit_hash"]),
        "physical_result_set_hash": stable_hash(
            {"rows": sorted(physical_rows, key=lambda row: row["work_unit_hash"])}
        ),
        "inactive_projections": sorted(projections, key=lambda row: row["item_id_hash"]),
        "active_pair": {
            "item_id_hash": stable_hash({"item_id": ACTIVE_FRESH_ITEM_ID}),
            "raw_observation_hash": active_raw.observation_hash,
            "candidate_observation_hash": candidate.observation_hash,
            "raw_success": active_raw.success,
            "candidate_success": candidate.success,
            "candidate_minus_raw": gain,
            "raw_error_type": active_raw.error_type,
            "candidate_error_type": candidate.error_type,
        },
        "paired_task_utility_measurement_valid": True,
        "causal_measurement_status": "valid_preregistered_post_agent_resume",
        "cohort_descriptive_summary": {
            "raw_success_count": raw_successes,
            "raw_item_count": len(raw_by_item),
            "projected_candidate_success_count": projected_candidate_successes,
            "projected_candidate_item_count": len(raw_by_item),
            "projected_candidate_minus_raw": projected_candidate_successes - raw_successes,
            "contains_one_frozen_audit_invalid_inactive_route": bool(invalid_diagnostics),
            "promotion_use_authorized": False,
        },
        "financial_runtime_evidence": [semantic_evidence],
        "financial_runtime_evidence_set_hash": stable_hash(
            {"rows": [semantic_evidence]}
        ),
        "worker_artifact_closure": artifact_closure,
        "original_model_execution_count": EXPECTED_PHYSICAL,
        "replayed_model_execution_count": 0,
        "finalizer_model_call_count": 0,
        "offline_verifier_execution_count": EXPECTED_ORPHANS + 1,
        "replayed_offline_verifier_execution_count": 0,
        "finalizer_verifier_call_count": 0,
        "offline_evaluation_only": True,
        "online_judge_calls": 0,
        "network_fallback_used": False,
        "selected_provider_unchanged": True,
        "official_hipporag": False,
        "hipporag_status": OFFICIAL_HIPPORAG_STATUS,
        "official_hipporag_execution_count": 0,
        "new_performance_gate_added": False,
        "promotion_gate_applied": False,
        "promotion_authorized": False,
        "sealed_test_accessed": False,
        "duration_or_cost_comparison_authorized": False,
        "container_cleanup": sorted(cleanup_rows, key=lambda row: row["work_unit_hash"]),
        "secret_value_persisted": False,
    }
    report = _self_hashed(body, "report_hash")
    _atomic_json(batch_root / RECOVERY_REPORT, report, refuse=True)
    _emit(
        sink,
        "financial_semantic_recovery_finalization_completed_v1",
        {
            "report_hash": report["report_hash"],
            "active_pair_valid": True,
            "raw_success": active_raw.success,
            "candidate_success": candidate.success,
            "candidate_minus_raw": gain,
            "invalid_physical_result_count": len(invalid_diagnostics),
            "model_calls": 0,
            "verifier_calls": 0,
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--treatment-manifest", type=Path, required=True)
    parser.add_argument("--measurement-manifest", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "OPENAI_API_KEY",
        "GPT5_API_KEY",
        "RUOLI_API_KEY",
        "ASSUMPTION_V2_API_KEY",
    ):
        os.environ.pop(name, None)
    report = finalize(args)
    print(
        json.dumps(
            {"completed": True, "report_hash": report["report_hash"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
