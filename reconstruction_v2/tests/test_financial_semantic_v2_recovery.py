from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from assumption_agent.benchmarks.codex_action_budget import (
    CODEX_ACTION_BUDGET_OVERFLOW_POLICY,
    CODEX_ACTION_BUDGET_POLICY_VERSION,
    CODEX_ACTION_BUDGET_UNIT,
    CODEX_ACTION_PROCESS_SCOPE_DEDICATED_CONTAINER,
    inspect_codex_action_trace,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    CODEX_ACTION_SUPERVISOR_PATH,
)
from assumption_agent.models import stable_hash
from replication_runtime.financial_semantic_v2.backends import (
    WORK_STAGE_ORDER_V2,
    initialize_work_state_v2,
)
from replication_runtime.financial_semantic_v2.durable_state import (
    atomic_write_hashed_json_v2,
    load_durable_stage_chain_v2,
)
from replication_runtime.financial_semantic_v2.recovery import (
    OBSERVATION_FILENAME,
    SEMANTIC_EVIDENCE_FILENAME,
    RecoveryEvidenceError,
    authorize_clean_model_execution_once_v2,
    load_completed_observation_without_model_v2,
    recover_existing_artifacts_without_model_v2,
    reconcile_work_without_model_v2,
)


LIMIT = 100
PROGRAM_ID = "1" * 64
TREATMENT_HASH = "2" * 64
SOURCE_RECEIPT_HASH = "3" * 64
PLAN_HASH = "4" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request(arm: str) -> tuple[dict[str, object], str, str]:
    request: dict[str, object] = {
        "variant": "policy_off" if arm == "raw" else "policy_on",
        "pair_id": "pair",
        "repeat": 0,
    }
    if arm == "candidate":
        request.update(
            {
                "program_id": PROGRAM_ID,
                "program_set_hash": stable_hash(
                    {"recipe_ids": [PROGRAM_ID]}
                ),
                "treatment_hash": TREATMENT_HASH,
                "external_skill_source_receipt_hash": (
                    SOURCE_RECEIPT_HASH
                ),
            }
        )
    request_hash = stable_hash(request)
    variant = "policy_off" if arm == "raw" else "policy_on"
    return request, request_hash, f"v2_{variant}_{request_hash[:18]}"


def _initialize(
    root: Path,
    *,
    arm: str,
    request_hash: str,
    trial_id: str,
) -> tuple[Path, str]:
    state = root / "state"
    trial = root / "trials" / trial_id
    trial.mkdir(parents=True)
    work_hash = "a" * 64 if arm == "raw" else "b" * 64
    initialize_work_state_v2(
        state_root=state,
        work_unit_hash=work_hash,
        request_hash=request_hash,
        planned_payload={"arm": arm, "trial_id": trial_id},
        semantic_plan_payload=(
            {"plan_hash": PLAN_HASH}
            if arm == "candidate"
            else {"applicable": False}
        ),
    )
    return trial, work_hash


def _write_terminal_evidence(trial: Path) -> None:
    agent = trial / "agent"
    agent.mkdir()
    nonce = "0" * 32
    rows = [
        {
            "type": "assumption.action_budget.started",
            "run_nonce": nonce,
            "policy": CODEX_ACTION_BUDGET_POLICY_VERSION,
            "unit": CODEX_ACTION_BUDGET_UNIT,
            "limit": LIMIT,
        },
        {
            "type": "error",
            "message": "transient reconnect",
        },
        {
            "type": "item.started",
            "item": {"id": "one", "type": "command_execution"},
        },
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 7, "output_tokens": 3},
        },
    ]
    trace = agent / "codex.txt"
    trace.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    counter = inspect_codex_action_trace(
        trace.read_text(encoding="utf-8").splitlines(),
        limit=LIMIT,
    )
    body = {
        "policy": CODEX_ACTION_BUDGET_POLICY_VERSION,
        "unit": CODEX_ACTION_BUDGET_UNIT,
        "overflow_policy": CODEX_ACTION_BUDGET_OVERFLOW_POLICY,
        "limit": LIMIT,
        "observed_steps": counter.observed_action_starts,
        "budget_reached": False,
        "trigger_event_index": None,
        "action_event_hash": counter.action_event_hash,
        "invalid_action_event_count": 0,
        "run_nonce": nonce,
        "trace_sha256": _sha256(trace),
        "turn_completed_observed": True,
        "turn_completed_count": 1,
        "turn_failed_count": 0,
        "invalid_terminal_usage_count": 0,
        "token_usage_complete": True,
        "token_usage": {
            "input_tokens": 7,
            "output_tokens": 3,
            "total_tokens": 10,
        },
        "spawn_error": False,
        "sigterm_attempted": False,
        "sigterm_delivered": False,
        "sigkill_attempted": False,
        "sigkill_delivered": False,
        "sigkill_grace_upper_bound_seconds": 15,
        "agent_exit_code": 0,
        "agent_exit_signal": None,
        "agent_exit_confirmed": True,
        "process_group_exit_confirmed": True,
        "process_scope": CODEX_ACTION_PROCESS_SCOPE_DEDICATED_CONTAINER,
        "process_baseline_hash": "5" * 64,
        "process_task_scan_complete": True,
        "residual_process_count": 0,
        "residual_tid_count": 0,
        "residual_sigkill_attempted_count": 0,
        "residual_sigkill_delivered_count": 0,
        "agent_processes_exit_confirmed": True,
        "post_trigger_started_count": 0,
        "budget_truncated": False,
        "supervisor_hash": _sha256(CODEX_ACTION_SUPERVISOR_PATH),
        "raw_content_persisted": False,
    }
    receipt = {**body, "receipt_hash": stable_hash(body)}
    (agent / "codex_action_budget_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_semantic_evidence(state: Path, request_hash: str) -> None:
    evidence = {
        "request_hash": request_hash,
        "plan_hash": PLAN_HASH,
        "program_id": PROGRAM_ID,
        "treatment_hash": TREATMENT_HASH,
        "external_skill_source_receipt_hash": SOURCE_RECEIPT_HASH,
        "executed_after_agent_exit": True,
        "executed_before_verifier_materialization": True,
        "online_calls": 0,
    }
    evidence["evidence_hash"] = stable_hash(evidence)
    atomic_write_hashed_json_v2(
        state / SEMANTIC_EVIDENCE_FILENAME,
        {
            "request_hash": request_hash,
            "evidence": evidence,
            "evidence_hash": evidence["evidence_hash"],
            "persisted_before_verifier": True,
        },
        hash_field="receipt_hash",
    )


def _write_verifier(trial: Path, reward: int = 1) -> None:
    verifier = trial / "verifier"
    verifier.mkdir(exist_ok=True)
    (verifier / "reward.txt").write_text(f"{reward}\n", encoding="utf-8")
    (verifier / "ctrf.json").write_text(
        json.dumps(
            {
                "results": {
                    "summary": {
                        "tests": 1,
                        "passed": reward,
                        "failed": 1 - reward,
                        "skipped": 0,
                        "pending": 0,
                        "other": 0,
                    },
                    "tests": [
                        {
                            "name": "test_outputs.py::test_output",
                            "status": "passed" if reward else "failed",
                        }
                    ],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_observation(
    state: Path,
    *,
    request: dict[str, object],
    request_hash: str,
    arm: str,
    reward: int = 1,
) -> None:
    observation = {
        "request": request,
        "success": bool(reward),
        "score": float(reward),
        "metrics": {"evaluation_valid": 1.0},
        "error_type": None,
    }
    atomic_write_hashed_json_v2(
        state / OBSERVATION_FILENAME,
        {
            "request_hash": request_hash,
            "observation": observation,
            "observation_hash": stable_hash(observation),
            "arm": arm,
            "secret_value_persisted": False,
        },
        hash_field="receipt_hash",
    )


def _reconcile(
    root: Path,
    *,
    arm: str,
    work_hash: str,
    request_hash: str,
    trial_id: str,
) -> object:
    return reconcile_work_without_model_v2(
        state_root=root / "state",
        trial_root=root / "trials" / trial_id,
        work_unit_hash=work_hash,
        request_hash=request_hash,
        trial_id=trial_id,
        arm=arm,
        expected_action_limit=LIMIT,
        expected_plan_hash=PLAN_HASH if arm == "candidate" else None,
        expected_program_id=PROGRAM_ID if arm == "candidate" else None,
        expected_treatment_hash=(
            TREATMENT_HASH if arm == "candidate" else None
        ),
        expected_external_source_receipt_hash=(
            SOURCE_RECEIPT_HASH if arm == "candidate" else None
        ),
    )


def test_clean_work_can_be_claimed_once_and_never_replayed(
    tmp_path: Path,
) -> None:
    _, request_hash, trial_id = _request("raw")
    trial, work_hash = _initialize(
        tmp_path,
        arm="raw",
        request_hash=request_hash,
        trial_id=trial_id,
    )
    clean = _reconcile(
        tmp_path,
        arm="raw",
        work_hash=work_hash,
        request_hash=request_hash,
        trial_id=trial_id,
    )
    assert clean.status == "clean_never_started"
    assert clean.may_claim_clean_model_execution
    assert clean.model_calls_accounted == 0

    authorize_clean_model_execution_once_v2(
        state_root=tmp_path / "state",
        trial_root=trial,
        work_unit_hash=work_hash,
        request_hash=request_hash,
        trial_id=trial_id,
        arm="raw",
    )
    with pytest.raises((FileExistsError, RecoveryEvidenceError)):
        authorize_clean_model_execution_once_v2(
            state_root=tmp_path / "state",
            trial_root=trial,
            work_unit_hash=work_hash,
            request_hash=request_hash,
            trial_id=trial_id,
            arm="raw",
        )
    interrupted = _reconcile(
        tmp_path,
        arm="raw",
        work_hash=work_hash,
        request_hash=request_hash,
        trial_id=trial_id,
    )
    assert interrupted.status == "recovery_required"
    assert interrupted.model_calls_accounted == 1
    assert not interrupted.model_replay_authorized


def test_candidate_crash_boundaries_advance_without_model_or_backend(
    tmp_path: Path,
) -> None:
    model = Mock(name="one_frozen_model_execution")
    model()
    request, request_hash, trial_id = _request("candidate")
    trial, work_hash = _initialize(
        tmp_path,
        arm="candidate",
        request_hash=request_hash,
        trial_id=trial_id,
    )
    authorize_clean_model_execution_once_v2(
        state_root=tmp_path / "state",
        trial_root=trial,
        work_unit_hash=work_hash,
        request_hash=request_hash,
        trial_id=trial_id,
        arm="candidate",
        expected_plan_hash=PLAN_HASH,
    )
    _write_terminal_evidence(trial)
    _write_semantic_evidence(tmp_path / "state", request_hash)

    evidence_boundary = _reconcile(
        tmp_path,
        arm="candidate",
        work_hash=work_hash,
        request_hash=request_hash,
        trial_id=trial_id,
    )
    assert evidence_boundary.status == "recovery_required"
    assert evidence_boundary.current_stage == "operator_completed"
    assert evidence_boundary.transitions_applied == (
        "agent_completed",
        "operator_completed",
    )
    assert evidence_boundary.model_calls_accounted == 1
    assert not evidence_boundary.model_replay_authorized
    assert model.call_count == 1

    _write_verifier(trial)
    verifier_boundary = _reconcile(
        tmp_path,
        arm="candidate",
        work_hash=work_hash,
        request_hash=request_hash,
        trial_id=trial_id,
    )
    assert verifier_boundary.status == "recovery_required"
    assert verifier_boundary.current_stage == "verifier_completed"
    assert verifier_boundary.transitions_applied == ("verifier_completed",)
    assert verifier_boundary.model_calls_accounted == 1
    assert model.call_count == 1

    _write_observation(
        tmp_path / "state",
        request=request,
        request_hash=request_hash,
        arm="candidate",
    )
    observation_boundary = _reconcile(
        tmp_path,
        arm="candidate",
        work_hash=work_hash,
        request_hash=request_hash,
        trial_id=trial_id,
    )
    assert observation_boundary.status == "reconciled_completed"
    assert observation_boundary.transitions_applied == (
        "observation_finalized",
    )
    assert observation_boundary.model_calls_accounted == 1
    assert model.call_count == 1
    assert observation_boundary.completed
    assert observation_boundary.safe_payload()["backend_calls"] == 0
    assert observation_boundary.safe_payload()[
        "model_calls_during_recovery"
    ] == 0

    chain = load_durable_stage_chain_v2(
        tmp_path / "state",
        stage_order=WORK_STAGE_ORDER_V2,
        work_unit_hash=work_hash,
        request_hash=request_hash,
    )
    assert [row.stage for row in chain] == list(WORK_STAGE_ORDER_V2)
    assert chain[2].payload["model_calls"] == 1

    loaded, frozen_observation = (
        load_completed_observation_without_model_v2(
            state_root=tmp_path / "state",
            trial_root=trial,
            work_unit_hash=work_hash,
            request_hash=request_hash,
            trial_id=trial_id,
            arm="candidate",
            expected_action_limit=LIMIT,
            expected_plan_hash=PLAN_HASH,
            expected_program_id=PROGRAM_ID,
            expected_treatment_hash=TREATMENT_HASH,
            expected_external_source_receipt_hash=SOURCE_RECEIPT_HASH,
        )
    )
    assert loaded.completed
    assert frozen_observation["request"] == request
    assert frozen_observation["score"] == 1.0
    assert model.call_count == 1


def test_raw_post_agent_and_verifier_artifacts_reconcile_without_replay(
    tmp_path: Path,
) -> None:
    model = Mock(name="one_frozen_model_execution")
    model()
    request, request_hash, trial_id = _request("raw")
    trial, work_hash = _initialize(
        tmp_path,
        arm="raw",
        request_hash=request_hash,
        trial_id=trial_id,
    )
    _write_terminal_evidence(trial)
    _write_verifier(trial, reward=0)
    _write_observation(
        tmp_path / "state",
        request=request,
        request_hash=request_hash,
        arm="raw",
        reward=0,
    )

    recovered = recover_existing_artifacts_without_model_v2(
        state_root=tmp_path / "state",
        trial_root=tmp_path / "trials" / trial_id,
        work_unit_hash=work_hash,
        request_hash=request_hash,
        trial_id=trial_id,
        arm="raw",
        expected_action_limit=LIMIT,
    )
    assert recovered.status == "reconciled_completed"
    assert recovered.transitions_applied == (
        "agent_completed",
        "operator_completed",
        "verifier_completed",
        "observation_finalized",
    )
    assert recovered.model_calls_accounted == 1
    assert not recovered.model_replay_authorized
    assert model.call_count == 1


def test_partial_or_tampered_outcome_fails_closed(tmp_path: Path) -> None:
    _, request_hash, trial_id = _request("raw")
    trial, work_hash = _initialize(
        tmp_path,
        arm="raw",
        request_hash=request_hash,
        trial_id=trial_id,
    )
    _write_terminal_evidence(trial)
    verifier = trial / "verifier"
    verifier.mkdir()
    (verifier / "reward.txt").write_text("1\n", encoding="utf-8")

    invalid = _reconcile(
        tmp_path,
        arm="raw",
        work_hash=work_hash,
        request_hash=request_hash,
        trial_id=trial_id,
    )
    assert invalid.status == "invalid"
    assert invalid.error_type == "offline_verifier_artifacts_partial"
    assert invalid.model_calls_accounted == 1
    assert not invalid.model_replay_authorized
