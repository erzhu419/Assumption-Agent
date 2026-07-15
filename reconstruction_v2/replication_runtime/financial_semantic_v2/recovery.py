from __future__ import annotations

"""Pure, no-model recovery for future financial-semantic work units.

The functions in this module only inspect durable host artifacts and append
missing stage receipts.  They never instantiate a benchmark backend, invoke a
model, run an operator, or execute an offline verifier.  A model call is
authorized at most once by a no-clobber claim written while the state is still
provably clean.  Once any execution evidence exists, model replay is always
forbidden.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from assumption_agent.benchmarks.codex_action_budget import (
    CODEX_ACTION_PROCESS_SCOPE_DEDICATED_CONTAINER,
    audit_codex_action_budget,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    CODEX_ACTION_SUPERVISOR_PATH,
)
from assumption_agent.models import stable_hash

from .backends import WORK_STAGE_ORDER_V2
from .durable_state import (
    DurableStageReceiptV2,
    DurableStateError,
    atomic_write_hashed_json_v2,
    load_durable_stage_chain_v2,
    read_hashed_json_v2,
    transition_durable_stage_v2,
)
from .terminal_audit import audit_codex_terminal_trace_v2


RECOVERY_VERSION = "financial_semantic_no_model_replay_recovery_v2"
MODEL_EXECUTION_CLAIM_VERSION = (
    "financial_semantic_model_execution_claim_v2"
)
MODEL_EXECUTION_CLAIM_FILENAME = "model_execution_claim.json"
CONTAINER_RECEIPT_FILENAME = "container_execution.receipt.json"
SEMANTIC_EVIDENCE_FILENAME = "semantic_runtime_evidence.json"
OBSERVATION_FILENAME = "observation.json"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARMS = frozenset({"raw", "candidate"})


class RecoveryEvidenceError(RuntimeError):
    def __init__(self, error_type: str) -> None:
        super().__init__(error_type)
        self.error_type = error_type


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _require_hash(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise RecoveryEvidenceError(f"{label}_invalid")
    return str(value)


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RecoveryEvidenceError("recovery_artifact_not_regular")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, error_type: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RecoveryEvidenceError(error_type)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryEvidenceError(error_type) from exc
    if not isinstance(value, dict):
        raise RecoveryEvidenceError(error_type)
    return value


def _expected_trial_id(*, arm: str, request_hash: str) -> str:
    variant = "policy_off" if arm == "raw" else "policy_on"
    return f"v2_{variant}_{request_hash[:18]}"


def _validate_initial_identity(
    chain: Sequence[DurableStageReceiptV2],
    *,
    arm: str,
    trial_id: str,
    expected_plan_hash: str | None,
) -> None:
    if len(chain) < 2 or [row.stage for row in chain[:2]] != list(
        WORK_STAGE_ORDER_V2[:2]
    ):
        raise RecoveryEvidenceError("premodel_stage_prefix_incomplete")
    planned = chain[0].payload
    semantic = chain[1].payload
    if planned.get("arm") != arm or planned.get("trial_id") != trial_id:
        raise RecoveryEvidenceError("planned_trial_identity_mismatch")
    if arm == "candidate":
        if not _is_sha256(expected_plan_hash):
            raise RecoveryEvidenceError("candidate_plan_hash_missing")
        if semantic.get("plan_hash") != expected_plan_hash:
            raise RecoveryEvidenceError("semantic_plan_stage_mismatch")
    elif expected_plan_hash is not None:
        raise RecoveryEvidenceError("raw_work_has_candidate_plan_hash")


def _artifact_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _execution_artifact_paths(
    *,
    state_root: Path,
    trial_root: Path,
) -> tuple[Path, ...]:
    ordinary_artifacts = (
        state_root / MODEL_EXECUTION_CLAIM_FILENAME,
        state_root / CONTAINER_RECEIPT_FILENAME,
        state_root / SEMANTIC_EVIDENCE_FILENAME,
        state_root / OBSERVATION_FILENAME,
        trial_root / "agent" / "codex.txt",
        trial_root / "agent" / "codex_action_budget_receipt.json",
        trial_root / "verifier" / "reward.txt",
        trial_root / "verifier" / "ctrf.json",
        trial_root / "result.json",
    )
    postmodel_stages = tuple(
        state_root / f"{index:03d}_{stage}.stage.json"
        for index, stage in enumerate(WORK_STAGE_ORDER_V2)
        if index >= 2
    )
    return (*ordinary_artifacts, *postmodel_stages)


def _claim_path(state_root: Path) -> Path:
    return state_root / MODEL_EXECUTION_CLAIM_FILENAME


def _validate_claim(
    path: Path,
    *,
    state_head_hash: str,
    work_unit_hash: str,
    request_hash: str,
    trial_id: str,
    trial_root: Path,
    arm: str,
) -> Mapping[str, Any] | None:
    if not _artifact_exists(path):
        return None
    try:
        value = read_hashed_json_v2(path, hash_field="receipt_hash")
    except (DurableStateError, OSError, ValueError) as exc:
        raise RecoveryEvidenceError("model_execution_claim_invalid") from exc
    expected = {
        "claim_version": MODEL_EXECUTION_CLAIM_VERSION,
        "work_unit_hash": work_unit_hash,
        "request_hash": request_hash,
        "trial_id": trial_id,
        "trial_path_hash": stable_hash(
            {"path": str(trial_root.resolve())}
        ),
        "arm": arm,
        "premodel_stage_head_hash": state_head_hash,
        "model_call_authorization_count": 1,
        "model_replay_authorized": False,
        "claim_consumed_by_first_execution_only": True,
    }
    body = dict(value)
    body.pop("receipt_hash", None)
    if body != expected:
        raise RecoveryEvidenceError("model_execution_claim_identity_mismatch")
    return value


def _validate_container_receipt(
    path: Path,
    *,
    work_unit_hash: str,
    request_hash: str,
    trial_id: str,
) -> Mapping[str, Any] | None:
    if not _artifact_exists(path):
        return None
    try:
        value = read_hashed_json_v2(path, hash_field="receipt_hash")
    except (DurableStateError, OSError, ValueError) as exc:
        raise RecoveryEvidenceError("container_execution_receipt_invalid") from exc
    if (
        value.get("work_unit_hash") != work_unit_hash
        or value.get("request_hash") != request_hash
        or value.get("trial_id") != trial_id
        or value.get("container_started") is not True
    ):
        raise RecoveryEvidenceError("container_execution_identity_mismatch")
    return value


def authorize_clean_model_execution_once_v2(
    *,
    state_root: str | Path,
    trial_root: str | Path,
    work_unit_hash: str,
    request_hash: str,
    trial_id: str,
    arm: str,
    expected_plan_hash: str | None = None,
    container_present: bool = False,
) -> dict[str, Any]:
    """Consume the one model-call authorization of a pristine work unit.

    A second call always fails because the claim is persisted with atomic
    no-clobber semantics.  If the runner dies after this claim but before model
    output appears, recovery still forbids replay.
    """

    state = Path(state_root).resolve()
    trial = Path(trial_root).resolve()
    work_hash = _require_hash(work_unit_hash, "work_unit_hash")
    request = _require_hash(request_hash, "request_hash")
    if arm not in _ARMS:
        raise RecoveryEvidenceError("recovery_arm_invalid")
    if trial_id != _expected_trial_id(arm=arm, request_hash=request):
        raise RecoveryEvidenceError("trial_id_request_identity_mismatch")
    if trial.name != trial_id:
        raise RecoveryEvidenceError("trial_path_identity_mismatch")
    chain = load_durable_stage_chain_v2(
        state,
        stage_order=WORK_STAGE_ORDER_V2,
        work_unit_hash=work_hash,
        request_hash=request,
    )
    _validate_initial_identity(
        chain,
        arm=arm,
        trial_id=trial_id,
        expected_plan_hash=expected_plan_hash,
    )
    if len(chain) != 2:
        raise RecoveryEvidenceError("model_execution_state_is_not_clean")
    if container_present or any(
        _artifact_exists(path)
        for path in _execution_artifact_paths(
            state_root=state,
            trial_root=trial,
        )
    ):
        raise RecoveryEvidenceError("model_execution_evidence_already_exists")
    body = {
        "claim_version": MODEL_EXECUTION_CLAIM_VERSION,
        "work_unit_hash": work_hash,
        "request_hash": request,
        "trial_id": trial_id,
        "trial_path_hash": stable_hash({"path": str(trial)}),
        "arm": arm,
        "premodel_stage_head_hash": chain[-1].stage_hash,
        "model_call_authorization_count": 1,
        "model_replay_authorized": False,
        "claim_consumed_by_first_execution_only": True,
    }
    return atomic_write_hashed_json_v2(
        _claim_path(state),
        body,
        hash_field="receipt_hash",
        refuse_existing=True,
    )


@dataclass(frozen=True)
class RecoveryDecisionV2:
    status: str
    error_type: str | None
    recovery_action: str | None
    current_stage: str | None
    transitions_applied: tuple[str, ...]
    model_calls_accounted: int
    model_replay_authorized: bool
    may_claim_clean_model_execution: bool
    completed: bool

    def safe_payload(self) -> dict[str, Any]:
        return {
            "recovery_version": RECOVERY_VERSION,
            "status": self.status,
            "error_type": self.error_type,
            "recovery_action": self.recovery_action,
            "current_stage": self.current_stage,
            "transitions_applied": list(self.transitions_applied),
            "model_calls_accounted": self.model_calls_accounted,
            "model_replay_authorized": self.model_replay_authorized,
            "may_claim_clean_model_execution": (
                self.may_claim_clean_model_execution
            ),
            "completed": self.completed,
            "backend_calls": 0,
            "model_calls_during_recovery": 0,
        }

    @property
    def decision_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.safe_payload()
        return {**payload, "decision_hash": self.decision_hash}


def _decision(
    *,
    status: str,
    error_type: str | None = None,
    recovery_action: str | None = None,
    chain: Sequence[DurableStageReceiptV2] = (),
    transitions: Sequence[str] = (),
    model_calls: int = 0,
    may_claim: bool = False,
    completed: bool = False,
) -> RecoveryDecisionV2:
    return RecoveryDecisionV2(
        status=status,
        error_type=error_type,
        recovery_action=recovery_action,
        current_stage=chain[-1].stage if chain else None,
        transitions_applied=tuple(transitions),
        model_calls_accounted=model_calls,
        model_replay_authorized=False,
        may_claim_clean_model_execution=may_claim,
        completed=completed,
    )


def _terminal_and_action_evidence(
    *,
    trial_root: Path,
    expected_action_limit: int,
    expected_process_scope: str,
    supervisor_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trace = trial_root / "agent" / "codex.txt"
    action_path = trial_root / "agent" / "codex_action_budget_receipt.json"
    trace_present = _artifact_exists(trace)
    action_present = _artifact_exists(action_path)
    if trace_present != action_present:
        raise RecoveryEvidenceError("agent_terminal_evidence_partial")
    if not trace_present:
        raise RecoveryEvidenceError("agent_terminal_evidence_missing")
    terminal = audit_codex_terminal_trace_v2(trace)
    if not terminal.valid:
        raise RecoveryEvidenceError(
            terminal.error_type or "agent_terminal_trace_invalid"
        )
    action = audit_codex_action_budget(
        trace_path=trace,
        receipt_path=action_path,
        supervisor_path=supervisor_path,
        expected_limit=expected_action_limit,
        expected_process_scope=expected_process_scope,
    )
    raw_receipt = _read_json_object(
        action_path,
        "agent_action_budget_receipt_malformed",
    )
    if (
        not action.valid
        or not action.token_usage_complete
        or not action.turn_completed_observed
        or not action.process_group_exit_confirmed
        or not action.agent_processes_exit_confirmed
        or raw_receipt.get("agent_exit_code") != 0
        or raw_receipt.get("turn_completed_count") != 1
        or raw_receipt.get("turn_failed_count") != 0
    ):
        raise RecoveryEvidenceError(
            action.error_type or "agent_action_budget_receipt_invalid"
        )
    terminal_payload = terminal.to_dict()
    action_payload = {
        "action_budget_receipt_sha256": _sha256_file(action_path),
        "action_budget_receipt_hash": action.receipt_hash,
        "action_event_hash": action.action_event_hash,
        "observed_steps": action.observed_steps,
        "token_usage_complete": action.token_usage_complete,
        "process_group_exit_confirmed": action.process_group_exit_confirmed,
        "agent_processes_exit_confirmed": (
            action.agent_processes_exit_confirmed
        ),
    }
    return terminal_payload, action_payload


def _validate_existing_agent_stage(
    stage: DurableStageReceiptV2,
    *,
    arm: str,
    terminal: Mapping[str, Any],
    action: Mapping[str, Any],
) -> None:
    payload = stage.payload
    expected = {
        "arm": arm,
        "model_calls": 1,
        "terminal_audit": terminal,
        "terminal_audit_valid": True,
        "action_budget_receipt_sha256": action[
            "action_budget_receipt_sha256"
        ],
        "action_budget_receipt_hash": action[
            "action_budget_receipt_hash"
        ],
        "action_event_hash": action["action_event_hash"],
        "observed_steps": action["observed_steps"],
        "token_usage_complete": True,
        "process_group_exit_confirmed": True,
        "agent_processes_exit_confirmed": True,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RecoveryEvidenceError("agent_stage_artifact_mismatch")


def _load_semantic_evidence(
    path: Path,
    *,
    request_hash: str,
    expected_plan_hash: str,
    expected_program_id: str | None,
    expected_treatment_hash: str | None,
    expected_external_source_receipt_hash: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        receipt = read_hashed_json_v2(path, hash_field="receipt_hash")
    except (DurableStateError, OSError, ValueError) as exc:
        raise RecoveryEvidenceError("semantic_evidence_receipt_invalid") from exc
    evidence = receipt.get("evidence")
    if not isinstance(evidence, Mapping):
        raise RecoveryEvidenceError("semantic_evidence_payload_invalid")
    evidence_body = dict(evidence)
    evidence_hash = evidence_body.pop("evidence_hash", None)
    if (
        receipt.get("request_hash") != request_hash
        or receipt.get("persisted_before_verifier") is not True
        or not _is_sha256(evidence_hash)
        or receipt.get("evidence_hash") != evidence_hash
        or stable_hash(evidence_body) != evidence_hash
        or evidence.get("request_hash") != request_hash
        or evidence.get("plan_hash") != expected_plan_hash
        or evidence.get("executed_after_agent_exit") is not True
        or evidence.get("executed_before_verifier_materialization") is not True
        or evidence.get("online_calls") != 0
    ):
        raise RecoveryEvidenceError("semantic_evidence_identity_mismatch")
    for field, expected in (
        ("program_id", expected_program_id),
        ("treatment_hash", expected_treatment_hash),
        (
            "external_skill_source_receipt_hash",
            expected_external_source_receipt_hash,
        ),
    ):
        if expected is not None and evidence.get(field) != expected:
            raise RecoveryEvidenceError(
                f"semantic_evidence_{field}_mismatch"
            )
    return receipt, dict(evidence)


def _validate_existing_operator_stage(
    stage: DurableStageReceiptV2,
    *,
    arm: str,
    expected_plan_hash: str | None,
    semantic_receipt: Mapping[str, Any] | None,
    semantic_evidence: Mapping[str, Any] | None,
) -> None:
    payload = stage.payload
    if payload.get("arm") != arm:
        raise RecoveryEvidenceError("operator_stage_arm_mismatch")
    if arm == "raw":
        if (
            payload.get("applicable") is not False
            or payload.get("operator_calls") != 0
        ):
            raise RecoveryEvidenceError("raw_operator_stage_invalid")
        return
    assert semantic_receipt is not None and semantic_evidence is not None
    if (
        payload.get("applicable") is not True
        or payload.get("operator_calls") != 1
        or payload.get("plan_hash") != expected_plan_hash
        or payload.get("semantic_evidence_hash")
        != semantic_evidence.get("evidence_hash")
        or payload.get("semantic_evidence_receipt_hash")
        != semantic_receipt.get("receipt_hash")
        or payload.get("persisted_before_verifier") is not True
    ):
        raise RecoveryEvidenceError("candidate_operator_stage_invalid")


def _load_verifier_artifacts(
    trial_root: Path,
) -> tuple[dict[str, Any], int] | None:
    reward_path = trial_root / "verifier" / "reward.txt"
    ctrf_path = trial_root / "verifier" / "ctrf.json"
    reward_present = _artifact_exists(reward_path)
    ctrf_present = _artifact_exists(ctrf_path)
    if reward_present != ctrf_present:
        raise RecoveryEvidenceError("offline_verifier_artifacts_partial")
    if not reward_present:
        return None
    raw_reward = reward_path.read_text(encoding="utf-8").strip()
    if raw_reward not in {"0", "1"}:
        raise RecoveryEvidenceError("offline_verifier_reward_invalid")
    ctrf = _read_json_object(ctrf_path, "offline_verifier_ctrf_invalid")
    results = ctrf.get("results")
    summary = results.get("summary") if isinstance(results, Mapping) else None
    tests = results.get("tests") if isinstance(results, Mapping) else None
    test_count = summary.get("tests") if isinstance(summary, Mapping) else None
    if (
        not isinstance(summary, Mapping)
        or not isinstance(tests, list)
        or isinstance(test_count, bool)
        or not isinstance(test_count, int)
        or test_count <= 0
        or len(tests) != test_count
    ):
        raise RecoveryEvidenceError("offline_verifier_ctrf_incomplete")
    payload = {
        "offline": True,
        "reward_sha256": _sha256_file(reward_path),
        "ctrf_sha256": _sha256_file(ctrf_path),
        "online_judge_calls": 0,
    }
    return payload, int(raw_reward)


def _validate_existing_verifier_stage(
    stage: DurableStageReceiptV2,
    *,
    verifier_payload: Mapping[str, Any],
) -> None:
    if any(
        stage.payload.get(key) != value
        for key, value in verifier_payload.items()
    ):
        raise RecoveryEvidenceError("verifier_stage_artifact_mismatch")


def _load_observation_receipt(
    path: Path,
    *,
    request_hash: str,
    arm: str,
    expected_program_id: str | None,
    expected_treatment_hash: str | None,
    expected_external_source_receipt_hash: str | None,
    expected_reward: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        receipt = read_hashed_json_v2(path, hash_field="receipt_hash")
    except (DurableStateError, OSError, ValueError) as exc:
        raise RecoveryEvidenceError("observation_receipt_invalid") from exc
    observation = receipt.get("observation")
    if not isinstance(observation, Mapping):
        raise RecoveryEvidenceError("observation_payload_invalid")
    observation_hash = stable_hash(dict(observation))
    request = observation.get("request")
    score = observation.get("score")
    expected_variant = "policy_off" if arm == "raw" else "policy_on"
    if (
        receipt.get("request_hash") != request_hash
        or receipt.get("arm") != arm
        or receipt.get("observation_hash") != observation_hash
        or not isinstance(request, Mapping)
        or stable_hash(dict(request)) != request_hash
        or request.get("variant") != expected_variant
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or float(score) != float(expected_reward)
    ):
        raise RecoveryEvidenceError("observation_request_identity_mismatch")
    for field, expected in (
        ("program_id", expected_program_id),
        ("treatment_hash", expected_treatment_hash),
        (
            "external_skill_source_receipt_hash",
            expected_external_source_receipt_hash,
        ),
    ):
        if expected is not None and request.get(field) != expected:
            raise RecoveryEvidenceError(f"observation_{field}_mismatch")
    return receipt, dict(observation)


def _validate_existing_observation_stage(
    stage: DurableStageReceiptV2,
    *,
    receipt: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> None:
    expected = {
        "observation_hash": receipt["observation_hash"],
        "observation_receipt_hash": receipt["receipt_hash"],
        "valid": observation.get("error_type") is None,
        "success": observation.get("success"),
        "score": observation.get("score"),
    }
    if any(stage.payload.get(key) != value for key, value in expected.items()):
        raise RecoveryEvidenceError("observation_stage_artifact_mismatch")


def reconcile_work_without_model_v2(
    *,
    state_root: str | Path,
    trial_root: str | Path,
    work_unit_hash: str,
    request_hash: str,
    trial_id: str,
    arm: str,
    expected_action_limit: int,
    expected_process_scope: str = (
        CODEX_ACTION_PROCESS_SCOPE_DEDICATED_CONTAINER
    ),
    supervisor_path: str | Path = CODEX_ACTION_SUPERVISOR_PATH,
    expected_plan_hash: str | None = None,
    expected_program_id: str | None = None,
    expected_treatment_hash: str | None = None,
    expected_external_source_receipt_hash: str | None = None,
    container_present: bool = False,
) -> RecoveryDecisionV2:
    """Inspect and append only stages already proven by durable artifacts."""

    chain: tuple[DurableStageReceiptV2, ...] = ()
    transitions: list[str] = []
    model_calls = 0
    try:
        state = Path(state_root).resolve()
        trial = Path(trial_root).resolve()
        model_calls = int(
            bool(
                container_present
                or any(
                    _artifact_exists(path)
                    for path in _execution_artifact_paths(
                        state_root=state,
                        trial_root=trial,
                    )
                )
            )
        )
        supervisor = Path(supervisor_path).resolve(strict=True)
        work_hash = _require_hash(work_unit_hash, "work_unit_hash")
        request = _require_hash(request_hash, "request_hash")
        if arm not in _ARMS:
            raise RecoveryEvidenceError("recovery_arm_invalid")
        if (
            isinstance(expected_action_limit, bool)
            or not isinstance(expected_action_limit, int)
            or expected_action_limit <= 0
        ):
            raise RecoveryEvidenceError("expected_action_limit_invalid")
        if trial_id != _expected_trial_id(arm=arm, request_hash=request):
            raise RecoveryEvidenceError("trial_id_request_identity_mismatch")
        if trial.name != trial_id:
            raise RecoveryEvidenceError("trial_path_identity_mismatch")
        chain = load_durable_stage_chain_v2(
            state,
            stage_order=WORK_STAGE_ORDER_V2,
            work_unit_hash=work_hash,
            request_hash=request,
        )
        _validate_initial_identity(
            chain,
            arm=arm,
            trial_id=trial_id,
            expected_plan_hash=expected_plan_hash,
        )
        claim = _validate_claim(
            _claim_path(state),
            state_head_hash=chain[1].stage_hash,
            work_unit_hash=work_hash,
            request_hash=request,
            trial_id=trial_id,
            trial_root=trial,
            arm=arm,
        )
        container_receipt = _validate_container_receipt(
            state / CONTAINER_RECEIPT_FILENAME,
            work_unit_hash=work_hash,
            request_hash=request,
            trial_id=trial_id,
        )
        execution_started = bool(
            claim
            or container_receipt
            or container_present
            or len(chain) > 2
            or any(
                _artifact_exists(path)
                for path in _execution_artifact_paths(
                    state_root=state,
                    trial_root=trial,
                )
                if path != _claim_path(state)
                and path != state / CONTAINER_RECEIPT_FILENAME
            )
        )
        model_calls = 1 if execution_started else model_calls
        if len(chain) == 2 and not execution_started:
            return _decision(
                status="clean_never_started",
                recovery_action="claim_model_execution_once",
                chain=chain,
                model_calls=0,
                may_claim=True,
            )

        trace = trial / "agent" / "codex.txt"
        action_path = trial / "agent" / "codex_action_budget_receipt.json"
        if not _artifact_exists(trace) and not _artifact_exists(action_path):
            return _decision(
                status="recovery_required",
                error_type="model_execution_claimed_without_terminal_evidence",
                recovery_action="do_not_replay_model_inspect_container",
                chain=chain,
                model_calls=1,
            )
        terminal, action = _terminal_and_action_evidence(
            trial_root=trial,
            expected_action_limit=expected_action_limit,
            expected_process_scope=expected_process_scope,
            supervisor_path=supervisor,
        )
        if len(chain) == 2:
            stage = transition_durable_stage_v2(
                state,
                stage_order=WORK_STAGE_ORDER_V2,
                work_unit_hash=work_hash,
                request_hash=request,
                stage="agent_completed",
                predecessor_stage_hash=chain[-1].stage_hash,
                payload={
                    "arm": arm,
                    "terminal_audit": terminal,
                    "terminal_audit_valid": True,
                    **action,
                    "reconciled_after_backend_return": False,
                    "recovered_without_model_call": True,
                    "model_calls": 1,
                    "raw_trace_persisted_in_stage": False,
                },
            )
            transitions.append(stage.stage)
            chain = (*chain, stage)
        else:
            _validate_existing_agent_stage(
                chain[2],
                arm=arm,
                terminal=terminal,
                action=action,
            )

        semantic_receipt: Mapping[str, Any] | None = None
        semantic_evidence: Mapping[str, Any] | None = None
        if arm == "candidate":
            evidence_path = state / SEMANTIC_EVIDENCE_FILENAME
            if not _artifact_exists(evidence_path):
                if len(chain) >= 4:
                    raise RecoveryEvidenceError(
                        "candidate_operator_stage_lacks_evidence"
                    )
                return _decision(
                    status="recovery_required",
                    error_type="candidate_operator_not_completed",
                    recovery_action=(
                        "resume_frozen_operator_only_without_model_call"
                    ),
                    chain=chain,
                    transitions=transitions,
                    model_calls=1,
                )
            assert expected_plan_hash is not None
            semantic_receipt, semantic_evidence = _load_semantic_evidence(
                evidence_path,
                request_hash=request,
                expected_plan_hash=expected_plan_hash,
                expected_program_id=expected_program_id,
                expected_treatment_hash=expected_treatment_hash,
                expected_external_source_receipt_hash=(
                    expected_external_source_receipt_hash
                ),
            )

        if len(chain) == 3:
            if arm == "raw":
                operator_payload = {
                    "arm": "raw",
                    "applicable": False,
                    "operator_calls": 0,
                    "recovered_without_model_call": True,
                }
            else:
                assert semantic_receipt is not None
                assert semantic_evidence is not None
                operator_payload = {
                    "arm": "candidate",
                    "applicable": True,
                    "plan_hash": expected_plan_hash,
                    "semantic_evidence_hash": semantic_evidence[
                        "evidence_hash"
                    ],
                    "semantic_evidence_receipt_hash": semantic_receipt[
                        "receipt_hash"
                    ],
                    "operator_calls": 1,
                    "online_calls": 0,
                    "persisted_before_verifier": True,
                    "recovered_without_model_call": True,
                }
            stage = transition_durable_stage_v2(
                state,
                stage_order=WORK_STAGE_ORDER_V2,
                work_unit_hash=work_hash,
                request_hash=request,
                stage="operator_completed",
                predecessor_stage_hash=chain[-1].stage_hash,
                payload=operator_payload,
            )
            transitions.append(stage.stage)
            chain = (*chain, stage)
        else:
            _validate_existing_operator_stage(
                chain[3],
                arm=arm,
                expected_plan_hash=expected_plan_hash,
                semantic_receipt=semantic_receipt,
                semantic_evidence=semantic_evidence,
            )

        verifier = _load_verifier_artifacts(trial)
        if verifier is None:
            if len(chain) >= 5:
                raise RecoveryEvidenceError(
                    "verifier_stage_lacks_complete_artifacts"
                )
            return _decision(
                status="recovery_required",
                error_type="offline_verifier_not_completed",
                recovery_action=(
                    "resume_frozen_offline_verifier_only_without_model_call"
                ),
                chain=chain,
                transitions=transitions,
                model_calls=1,
            )
        verifier_payload, verifier_reward = verifier
        if len(chain) == 4:
            stage = transition_durable_stage_v2(
                state,
                stage_order=WORK_STAGE_ORDER_V2,
                work_unit_hash=work_hash,
                request_hash=request,
                stage="verifier_completed",
                predecessor_stage_hash=chain[-1].stage_hash,
                payload={
                    **verifier_payload,
                    "recovered_without_backend_call": True,
                },
            )
            transitions.append(stage.stage)
            chain = (*chain, stage)
        else:
            _validate_existing_verifier_stage(
                chain[4],
                verifier_payload=verifier_payload,
            )

        observation_path = state / OBSERVATION_FILENAME
        if not _artifact_exists(observation_path):
            if len(chain) >= 6:
                raise RecoveryEvidenceError(
                    "observation_stage_lacks_receipt"
                )
            return _decision(
                status="recovery_required",
                error_type="observation_not_materialized",
                recovery_action=(
                    "reconstruct_observation_from_frozen_result_without_backend"
                ),
                chain=chain,
                transitions=transitions,
                model_calls=1,
            )
        observation_receipt, observation = _load_observation_receipt(
            observation_path,
            request_hash=request,
            arm=arm,
            expected_program_id=expected_program_id,
            expected_treatment_hash=expected_treatment_hash,
            expected_external_source_receipt_hash=(
                expected_external_source_receipt_hash
            ),
            expected_reward=verifier_reward,
        )
        if len(chain) == 5:
            stage = transition_durable_stage_v2(
                state,
                stage_order=WORK_STAGE_ORDER_V2,
                work_unit_hash=work_hash,
                request_hash=request,
                stage="observation_finalized",
                predecessor_stage_hash=chain[-1].stage_hash,
                payload={
                    "observation_hash": observation_receipt[
                        "observation_hash"
                    ],
                    "observation_receipt_hash": observation_receipt[
                        "receipt_hash"
                    ],
                    "valid": observation.get("error_type") is None,
                    "success": observation.get("success"),
                    "score": observation.get("score"),
                    "recovered_without_backend_call": True,
                },
            )
            transitions.append(stage.stage)
            chain = (*chain, stage)
        else:
            _validate_existing_observation_stage(
                chain[5],
                receipt=observation_receipt,
                observation=observation,
            )
        return _decision(
            status=(
                "reconciled_completed" if transitions else "completed"
            ),
            chain=chain,
            transitions=transitions,
            model_calls=1,
            completed=True,
        )
    except (
        RecoveryEvidenceError,
        DurableStateError,
        FileExistsError,
        OSError,
        ValueError,
    ) as exc:
        error_type = (
            exc.error_type
            if isinstance(exc, RecoveryEvidenceError)
            else f"recovery_{type(exc).__name__.lower()}"
        )
        return _decision(
            status="invalid",
            error_type=error_type,
            recovery_action="do_not_replay_model",
            chain=chain,
            transitions=transitions,
            model_calls=model_calls,
        )


def recover_existing_artifacts_without_model_v2(
    *,
    state_root: str | Path,
    trial_root: str | Path,
    work_unit_hash: str,
    request_hash: str,
    trial_id: str,
    arm: str,
    expected_action_limit: int,
    expected_process_scope: str = (
        CODEX_ACTION_PROCESS_SCOPE_DEDICATED_CONTAINER
    ),
    supervisor_path: str | Path = CODEX_ACTION_SUPERVISOR_PATH,
    expected_plan_hash: str | None = None,
    expected_program_id: str | None = None,
    expected_treatment_hash: str | None = None,
    expected_external_source_receipt_hash: str | None = None,
    container_present: bool = False,
) -> RecoveryDecisionV2:
    """Run the explicit artifact-only post-agent recovery path.

    The upstream lifecycle does not expose a safe public continuation API for
    re-entering its private post-agent container state.  Consequently this
    entry point advances only phases already proved complete by the existing
    terminal trace, semantic evidence, offline-verifier outputs, and frozen
    observation receipt.  It never instantiates or calls a backend, model,
    operator, verifier, or observation materializer.  A missing artifact is
    reported as ``recovery_required`` and must never trigger model replay.

    A clean never-started unit is returned unchanged.  That result merely
    describes the durable state; this recovery entry point does not consume
    its one execution authorization.
    """

    return reconcile_work_without_model_v2(
        state_root=state_root,
        trial_root=trial_root,
        work_unit_hash=work_unit_hash,
        request_hash=request_hash,
        trial_id=trial_id,
        arm=arm,
        expected_action_limit=expected_action_limit,
        expected_process_scope=expected_process_scope,
        supervisor_path=supervisor_path,
        expected_plan_hash=expected_plan_hash,
        expected_program_id=expected_program_id,
        expected_treatment_hash=expected_treatment_hash,
        expected_external_source_receipt_hash=(
            expected_external_source_receipt_hash
        ),
        container_present=container_present,
    )


def load_completed_observation_without_model_v2(
    *,
    state_root: str | Path,
    trial_root: str | Path,
    work_unit_hash: str,
    request_hash: str,
    trial_id: str,
    arm: str,
    expected_action_limit: int,
    expected_process_scope: str = (
        CODEX_ACTION_PROCESS_SCOPE_DEDICATED_CONTAINER
    ),
    supervisor_path: str | Path = CODEX_ACTION_SUPERVISOR_PATH,
    expected_plan_hash: str | None = None,
    expected_program_id: str | None = None,
    expected_treatment_hash: str | None = None,
    expected_external_source_receipt_hash: str | None = None,
    container_present: bool = False,
) -> tuple[RecoveryDecisionV2, dict[str, Any]]:
    """Return one fully re-audited frozen observation without a backend call.

    This is the only recovery read intended for the execution runner.  It
    first reconciles the complete durable chain, then independently re-reads
    the reward, CTRF, and self-hashed observation receipt.  Incomplete or
    invalid work is never converted into an observation.
    """

    decision = reconcile_work_without_model_v2(
        state_root=state_root,
        trial_root=trial_root,
        work_unit_hash=work_unit_hash,
        request_hash=request_hash,
        trial_id=trial_id,
        arm=arm,
        expected_action_limit=expected_action_limit,
        expected_process_scope=expected_process_scope,
        supervisor_path=supervisor_path,
        expected_plan_hash=expected_plan_hash,
        expected_program_id=expected_program_id,
        expected_treatment_hash=expected_treatment_hash,
        expected_external_source_receipt_hash=(
            expected_external_source_receipt_hash
        ),
        container_present=container_present,
    )
    if not decision.completed:
        raise RecoveryEvidenceError(
            decision.error_type or "recovery_observation_not_completed"
        )
    verifier = _load_verifier_artifacts(Path(trial_root).resolve())
    if verifier is None:
        raise RecoveryEvidenceError("offline_verifier_not_completed")
    _, reward = verifier
    _, observation = _load_observation_receipt(
        Path(state_root).resolve() / OBSERVATION_FILENAME,
        request_hash=_require_hash(request_hash, "request_hash"),
        arm=arm,
        expected_program_id=expected_program_id,
        expected_treatment_hash=expected_treatment_hash,
        expected_external_source_receipt_hash=(
            expected_external_source_receipt_hash
        ),
        expected_reward=reward,
    )
    return decision, observation
