from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from assumption_agent.benchmarks.historical_raw_train_projection_v2 import (
    HISTORICAL_SOURCE_COMMIT,
    HistoricalRawTrainProjectionError,
    historical_observation_hash_v2,
    load_historical_raw_train_projection_v2,
)
from assumption_agent.benchmarks.skilllearn_compiler import (
    NO_SKILL_TREATMENT_HASH,
    skilllearn_program_set_treatment_hash,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnTrialObservation,
    SkillLearnTrialRequest,
    TrialVariant,
    _extract_train_action_trace_profile,
)
from assumption_agent.events import Event
from assumption_agent.models import SplitName, stable_hash


SOURCE_TRACE_ID = "historical-fixture:shared-train"
EVALUATOR_EPOCH = "historical-fixture-evaluator"
MANIFEST_HASH = stable_hash({"manifest": "historical-fixture"})
CODEX_POLICY_HASH = stable_hash({"policy": "historical-fixture"})
ITEM_ID = "fixture-family-1"
FAMILY = "fixture-family"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture_source(tmp_path: Path) -> tuple[Path, Path, SkillLearnTrialObservation]:
    source_root = tmp_path / "source"
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "benchmark": "skilllearnbench",
        "family_by_id": {ITEM_ID: FAMILY},
        "manifest_hash": MANIFEST_HASH,
        "raw_content_persisted": False,
        "sealed_test": True,
        "train_ids": [ITEM_ID],
    }
    _write_json(manifest_path, manifest)

    program_set_hash = skilllearn_program_set_treatment_hash(())
    pair_id = stable_hash(
        {
            "trace_id": SOURCE_TRACE_ID,
            "item_id": ITEM_ID,
            "stage": "training_baseline",
            "program_set_hash": program_set_hash,
            "treatment_hash": NO_SKILL_TREATMENT_HASH,
        }
    )[:20]
    request = SkillLearnTrialRequest(
        item_id=ITEM_ID,
        family=FAMILY,
        split=SplitName.TRAIN,
        variant=TrialVariant.POLICY_OFF,
        evaluator_epoch=EVALUATOR_EPOCH,
        pair_id=pair_id,
        repeat=1,
        agent_id="codex",
        model="gpt-5.4-mini",
        max_steps=100,
        manifest_hash=MANIFEST_HASH,
        codex_agent_execution_policy_hash=CODEX_POLICY_HASH,
        program_set_hash=program_set_hash,
        treatment_hash=NO_SKILL_TREATMENT_HASH,
    )
    trial_root = (
        source_root
        / "development_recursive"
        / "upstream_trials"
        / "no_skill"
        / FAMILY
        / ITEM_ID
        / request.trial_id
    )
    trace_path = trial_root / "agent" / "codex.txt"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_row = {
        "type": "item.completed",
        "item": {
            "id": "fixture-command",
            "type": "command_execution",
            "command": "cat /root/input.txt",
            "exit_code": 0,
            "status": "completed",
        },
    }
    trace_path.write_text(json.dumps(trace_row) + "\n", encoding="utf-8")
    action_trace = _extract_train_action_trace_profile(
        trace_path.resolve(),
        containment_root=(
            source_root / "development_recursive" / "upstream_trials"
        ).resolve(),
    )
    observation = SkillLearnTrialObservation(
        request=request,
        success=True,
        score=1.0,
        metrics=MappingProxyType(
            {"evaluation_valid": 1.0, "task_success": 1.0}
        ),
        total_tokens=123,
        steps=2,
        duration_seconds=1.25,
        provider_fingerprint=stable_hash({"provider": "fixture"}),
        fairness_fingerprint=stable_hash({"fairness": "fixture"}),
        upstream_result_hash=stable_hash({"upstream": "fixture"}),
        raw_trial_artifacts_persisted=True,
        prebuilt_image_key=stable_hash({"image-key": "fixture"}),
        prebuilt_image_id="sha256:" + stable_hash({"image": "fixture"}),
        prebuilt_cache_reused=True,
        agent_runtime_key=stable_hash({"runtime": "fixture"}),
        agent_runtime_version="codex-cli fixture",
        offline_verifier_profile_id="fixture-verifier",
        offline_verifier_runtime_key=stable_hash({"verifier": "fixture"}),
        step_budget_policy="codex_jsonl_action_start_budget_v1",
        step_budget_unit="codex_action_start_v1",
        step_budget_limit=100,
        step_budget_truncated=False,
        step_budget_token_usage_complete=True,
        step_budget_receipt_hash=stable_hash({"budget": "fixture"}),
        proposal_action_trace=MappingProxyType(action_trace),
    )
    _write_json(
        trial_root / "result.json",
        {
            "task_id": f"{FAMILY}/{ITEM_ID}",
            "trial_id": request.trial_id,
            "model": request.model,
            "skill_config": "no_skill",
            "passed": True,
            "reward": 1,
        },
    )

    source_observation_hash = historical_observation_hash_v2(observation)
    completed_payload = {
        "agent_runtime_key": observation.agent_runtime_key,
        "agent_runtime_version": observation.agent_runtime_version,
        "duration_seconds": observation.duration_seconds,
        "error_type": None,
        "fairness_fingerprint": observation.fairness_fingerprint,
        "installed_skill_source_receipt_hash": "",
        "metrics": dict(observation.metrics),
        "observation_hash": source_observation_hash,
        "offline_verifier_profile_id": observation.offline_verifier_profile_id,
        "offline_verifier_runtime_key": observation.offline_verifier_runtime_key,
        "prebuilt_cache_reused": True,
        "prebuilt_image_id": observation.prebuilt_image_id,
        "prebuilt_image_key": observation.prebuilt_image_key,
        "provider_fingerprint": observation.provider_fingerprint,
        "raw_trial_artifacts_persisted": True,
        "request_hash": request.request_hash,
        "step_budget_limit": 100,
        "step_budget_policy": observation.step_budget_policy,
        "step_budget_receipt_hash": observation.step_budget_receipt_hash,
        "step_budget_token_usage_complete": True,
        "step_budget_truncated": False,
        "step_budget_unit": observation.step_budget_unit,
        "steps": observation.steps,
        "success": True,
        "total_tokens": observation.total_tokens,
        "upstream_result_hash": observation.upstream_result_hash,
        "valid": True,
        "variant": TrialVariant.POLICY_OFF.value,
    }
    observation_set_hash = stable_hash({"hashes": [source_observation_hash]})
    evidence_payload = {
        "new_training_executions": 1,
        "observation_count": 1,
        "observation_set_hash": observation_set_hash,
        "raw_content_persisted": False,
        "sealed_test_accessed": False,
        "source_trace_id": SOURCE_TRACE_ID,
    }
    events = (
        Event(
            event="skilllearn_trial_completed",
            stage="benchmark.skilllearn.trial",
            trace_id=f"{SOURCE_TRACE_ID}:{pair_id}:train:attempt-1",
            payload=completed_payload,
        ).to_dict(),
        Event(
            event="training_evidence_recorded",
            stage="benchmark.skilllearn.training_replay",
            trace_id=SOURCE_TRACE_ID,
            payload=evidence_payload,
        ).to_dict(),
    )
    events_path = source_root / "development_recursive.events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in events),
        encoding="utf-8",
    )
    protocol = {
        "claim_eligible": True,
        "git": {
            "commit": HISTORICAL_SOURCE_COMMIT,
            "scoped_dirty": False,
        },
        "max_steps": 100,
        "model": "gpt-5.4-mini",
        "primary_manifest_hash": MANIFEST_HASH,
        "resolved_codex_agent_execution_policy_hash": CODEX_POLICY_HASH,
    }
    protocol["lock_hash"] = stable_hash(protocol)
    _write_json(source_root / "protocol_lock.json", protocol)
    return source_root, manifest_path, observation


def test_projects_exact_historical_raw_train_without_new_calls(tmp_path: Path) -> None:
    source_root, manifest_path, source_observation = _fixture_source(tmp_path)
    source_hash = historical_observation_hash_v2(source_observation)
    result = load_historical_raw_train_projection_v2(
        source_root=source_root,
        manifest_path=manifest_path,
        source_trace_id=SOURCE_TRACE_ID,
        evaluator_epoch=EVALUATOR_EPOCH,
        expected_source_observation_set_hash=stable_hash(
            {"hashes": [source_hash]}
        ),
    )

    result.verify()
    assert result.receipt.row_count == 1
    assert result.receipt.success_count == 1
    assert result.receipt.source_observation_set_hash == stable_hash(
        {"hashes": [source_hash]}
    )
    assert result.receipt.projected_observation_set_hash == stable_hash(
        {"hashes": [source_observation.observation_hash]}
    )
    assert result.receipt.safe_payload()["model_calls"] == 0
    assert result.receipt.safe_payload()["evaluator_calls"] == 0
    assert result.baseline_set.source_raw_trial_artifact_row_count == 1
    assert result.baseline_set.source_train_receipt_hash == (
        result.receipt.receipt_hash
    )

    repeated = load_historical_raw_train_projection_v2(
        source_root=source_root,
        manifest_path=manifest_path,
        source_trace_id=SOURCE_TRACE_ID,
        evaluator_epoch=EVALUATOR_EPOCH,
        expected_source_observation_set_hash=stable_hash(
            {"hashes": [source_hash]}
        ),
    )
    assert repeated.receipt.receipt_hash == result.receipt.receipt_hash
    assert repeated.baseline_set.baseline_set_hash == (
        result.baseline_set.baseline_set_hash
    )


def test_projection_fails_closed_on_source_or_schema_drift(tmp_path: Path) -> None:
    source_root, manifest_path, source_observation = _fixture_source(tmp_path)
    with pytest.raises(
        HistoricalRawTrainProjectionError,
        match="evidence receipt",
    ):
        load_historical_raw_train_projection_v2(
            source_root=source_root,
            manifest_path=manifest_path,
            source_trace_id=SOURCE_TRACE_ID,
            evaluator_epoch=EVALUATOR_EPOCH,
            expected_source_observation_set_hash="0" * 64,
        )

    with pytest.raises(
        HistoricalRawTrainProjectionError,
        match="nonempty current field",
    ):
        historical_observation_hash_v2(
            replace(
                source_observation,
                runtime_profile_prompt_delivery_policy="unexpected",
            )
        )

    events_path = source_root / "development_recursive.events.jsonl"
    events_path.write_text(
        events_path.read_text(encoding="utf-8").replace(
            '"valid": true',
            '"valid": false',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        HistoricalRawTrainProjectionError,
        match="event envelope failed",
    ):
        load_historical_raw_train_projection_v2(
            source_root=source_root,
            manifest_path=manifest_path,
            source_trace_id=SOURCE_TRACE_ID,
            evaluator_epoch=EVALUATOR_EPOCH,
            expected_source_observation_set_hash=stable_hash(
                {"hashes": [historical_observation_hash_v2(source_observation)]}
            ),
        )
