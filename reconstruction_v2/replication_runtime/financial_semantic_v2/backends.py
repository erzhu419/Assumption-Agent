from __future__ import annotations

"""Future-only backend wrappers for the SEC-13F period-out replication.

The frozen semantic planner and operator remain unchanged.  This module adds
only evaluator-side terminal-trace semantics and crash-durable receipts.  It
must never be used to reinterpret an already frozen observation.
"""

from contextlib import contextmanager
import hashlib
from pathlib import Path
import threading
from typing import Any, Iterator, Mapping, Sequence

from assumption_agent.benchmarks import skilllearn_lifecycle
from assumption_agent.benchmarks.financial_semantic_integration_v1 import (
    FinancialSemanticSubprocessBackendV1,
)
from assumption_agent.benchmarks.codex_action_budget import (
    audit_codex_action_budget,
)
from assumption_agent.benchmarks.skilllearn_compiler import (
    NO_SKILL_TREATMENT_HASH,
)
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnSubprocessBackend,
    SkillLearnTrialObservation,
    SkillLearnTrialRequest,
    TrialVariant,
)
from assumption_agent.models import stable_hash

from .durable_state import (
    DurableStageReceiptV2,
    atomic_write_hashed_json_v2,
    load_durable_stage_chain_v2,
    transition_durable_stage_v2,
)
from .terminal_audit import audit_codex_terminal_trace_v2


FUTURE_TERMINAL_PATCH_VERSION = (
    "financial_semantic_future_terminal_patch_v2"
)
WORK_STAGE_ORDER_V2 = (
    "planned",
    "semantic_plan_ready",
    "agent_completed",
    "operator_completed",
    "verifier_completed",
    "observation_finalized",
)

_PATCH_LOCK = threading.RLock()
_ORIGINAL_TERMINAL_LABEL = skilllearn_lifecycle._codex_terminal_error_label
_PATCH_DEPTH = 0


class FinancialSemanticReplicationBackendError(RuntimeError):
    """A future-only backend receipt failed closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _future_terminal_error_label_v2(*streams: object) -> str | None:
    """Legacy-compatible label backed by ordered completion semantics.

    A generic transport error is recoverable only when the same ordered trace
    subsequently contains exactly one ``turn.completed``.  Fatal or malformed
    traces retain the original provider-specific label when one is available.
    """

    original = _ORIGINAL_TERMINAL_LABEL(*streams)
    audit = audit_codex_terminal_trace_v2(*streams)
    if audit.turn_completed_count or audit.turn_failed_count:
        if audit.valid:
            return None
        return original or audit.error_type or "codex_terminal_trace_invalid_v2"

    # Lifecycle also calls this function on deliberately truncated stdout and
    # stderr snippets during sanitization.  Absence of ``turn.completed`` in
    # such a snippet is not evidence that the durable trace lacks completion.
    # Preserve immediately actionable provider failures, while deferring a
    # generic incomplete-trace decision to the complete codex.txt audit and
    # the action-budget receipt.
    if original in {
        "provider_usage_limit",
        "provider_rate_limit",
        "provider_authentication_failed",
        "provider_model_unavailable",
    }:
        return original
    return None


@contextmanager
def future_terminal_semantics_v2() -> Iterator[None]:
    """Patch one complete future run, never historical result material.

    The lock is deliberately held for the full context.  All worker threads
    see one immutable function, and a second in-process evaluator cannot race
    the patch or restore the legacy function early.
    """

    global _PATCH_DEPTH
    with _PATCH_LOCK:
        current = skilllearn_lifecycle._codex_terminal_error_label
        if current not in {
            _ORIGINAL_TERMINAL_LABEL,
            _future_terminal_error_label_v2,
        }:
            raise FinancialSemanticReplicationBackendError(
                "terminal label function was modified by another evaluator"
            )
        if _PATCH_DEPTH == 0:
            skilllearn_lifecycle._codex_terminal_error_label = (
                _future_terminal_error_label_v2
            )
        _PATCH_DEPTH += 1
        try:
            yield
        finally:
            _PATCH_DEPTH -= 1
            if _PATCH_DEPTH == 0:
                skilllearn_lifecycle._codex_terminal_error_label = (
                    _ORIGINAL_TERMINAL_LABEL
                )


def initialize_work_state_v2(
    *,
    state_root: str | Path,
    work_unit_hash: str,
    request_hash: str,
    planned_payload: Mapping[str, Any],
    semantic_plan_payload: Mapping[str, Any],
) -> tuple[DurableStageReceiptV2, DurableStageReceiptV2]:
    """Create the two pre-model stages exactly once."""

    root = Path(state_root)
    planned = transition_durable_stage_v2(
        root,
        stage_order=WORK_STAGE_ORDER_V2,
        work_unit_hash=work_unit_hash,
        request_hash=request_hash,
        stage="planned",
        predecessor_stage_hash=None,
        payload=planned_payload,
    )
    semantic = transition_durable_stage_v2(
        root,
        stage_order=WORK_STAGE_ORDER_V2,
        work_unit_hash=work_unit_hash,
        request_hash=request_hash,
        stage="semantic_plan_ready",
        predecessor_stage_hash=planned.stage_hash,
        payload=semantic_plan_payload,
    )
    return planned, semantic


class _DurableBackendMixinV2:
    durable_state_root: Path
    durable_work_unit_hash: str
    durable_request_hash: str
    durable_arm: str

    def _durable_chain(self) -> tuple[DurableStageReceiptV2, ...]:
        return load_durable_stage_chain_v2(
            self.durable_state_root,
            stage_order=WORK_STAGE_ORDER_V2,
            work_unit_hash=self.durable_work_unit_hash,
            request_hash=self.durable_request_hash,
        )

    def _transition_next(
        self,
        stage: str,
        payload: Mapping[str, Any],
    ) -> DurableStageReceiptV2:
        chain = self._durable_chain()
        predecessor = chain[-1].stage_hash if chain else None
        return transition_durable_stage_v2(
            self.durable_state_root,
            stage_order=WORK_STAGE_ORDER_V2,
            work_unit_hash=self.durable_work_unit_hash,
            request_hash=self.durable_request_hash,
            stage=stage,
            predecessor_stage_hash=predecessor,
            payload=payload,
        )

    def _trial_path(self, request: SkillLearnTrialRequest) -> Path:
        trials = getattr(self, "trials_dir", None)
        if not isinstance(trials, Path):
            raise FinancialSemanticReplicationBackendError(
                "durable backend requires an explicit trials directory"
            )
        skill_config = (
            "no_skill"
            if self.durable_arm == "raw"
            else "assumption-agent-v2-challenger"
        )
        return (
            trials
            / skill_config
            / request.family
            / request.item_id
            / request.trial_id
        )

    def _agent_completion_payload(
        self,
        request: SkillLearnTrialRequest,
        *,
        reconciled_after_backend_return: bool,
    ) -> dict[str, Any]:
        trial = self._trial_path(request)
        trace = trial / "agent" / "codex.txt"
        if trace.is_symlink() or not trace.is_file():
            raise FinancialSemanticReplicationBackendError(
                "durable agent trace is missing"
            )
        audit = audit_codex_terminal_trace_v2(trace)
        if not audit.valid:
            raise FinancialSemanticReplicationBackendError(
                "complete Codex trace failed the future terminal audit"
            )
        action_receipt = trial / "agent" / "codex_action_budget_receipt.json"
        if action_receipt.is_symlink() or not action_receipt.is_file():
            raise FinancialSemanticReplicationBackendError(
                "durable action-budget receipt is missing"
            )
        budget = audit_codex_action_budget(
            trace_path=trace,
            receipt_path=action_receipt,
            supervisor_path=skilllearn_lifecycle.CODEX_ACTION_SUPERVISOR_PATH,
            expected_limit=int(getattr(self, "max_steps")),
            expected_process_scope=str(
                getattr(self, "codex_agent_execution_policy")
                .action_budget_process_scope
            ),
        )
        if (
            not budget.valid
            or not budget.turn_completed_observed
            or not budget.token_usage_complete
            or not budget.process_group_exit_confirmed
            or not budget.agent_processes_exit_confirmed
        ):
            raise FinancialSemanticReplicationBackendError(
                "action-budget receipt does not prove agent completion"
            )
        return {
            "arm": self.durable_arm,
            "terminal_audit": audit.to_dict(),
            "terminal_audit_valid": audit.valid,
            "action_budget_receipt_sha256": _sha256_file(action_receipt),
            "action_budget_receipt_hash": budget.receipt_hash,
            "action_event_hash": budget.action_event_hash,
            "observed_steps": budget.observed_steps,
            "token_usage_complete": budget.token_usage_complete,
            "process_group_exit_confirmed": (
                budget.process_group_exit_confirmed
            ),
            "agent_processes_exit_confirmed": (
                budget.agent_processes_exit_confirmed
            ),
            "reconciled_after_backend_return": (
                reconciled_after_backend_return
            ),
            "model_calls": 1,
            "raw_trace_persisted_in_stage": False,
        }

    def _complete_after_observation(
        self,
        request: SkillLearnTrialRequest,
        observation: SkillLearnTrialObservation,
    ) -> None:
        chain = self._durable_chain()
        stages = [row.stage for row in chain]
        if stages == list(WORK_STAGE_ORDER_V2[:2]):
            self._transition_next(
                "agent_completed",
                self._agent_completion_payload(
                    request,
                    reconciled_after_backend_return=True,
                ),
            )
            self._transition_next(
                "operator_completed",
                {
                    "arm": self.durable_arm,
                    "applicable": False,
                    "operator_calls": 0,
                },
            )
        elif stages != list(WORK_STAGE_ORDER_V2[:4]):
            raise FinancialSemanticReplicationBackendError(
                "backend returned at an unexpected durable stage"
            )

        trial = self._trial_path(request)
        verifier = trial / "verifier"
        reward = verifier / "reward.txt"
        ctrf = verifier / "ctrf.json"
        if (
            reward.is_symlink()
            or ctrf.is_symlink()
            or not reward.is_file()
            or not ctrf.is_file()
        ):
            raise FinancialSemanticReplicationBackendError(
                "offline verifier artifacts are incomplete"
            )
        self._transition_next(
            "verifier_completed",
            {
                "offline": True,
                "reward_sha256": _sha256_file(reward),
                "ctrf_sha256": _sha256_file(ctrf),
                "online_judge_calls": 0,
            },
        )
        observation_body = observation.to_dict()
        observation_receipt = atomic_write_hashed_json_v2(
            self.durable_state_root / "observation.json",
            {
                "request_hash": request.request_hash,
                "observation": observation_body,
                "observation_hash": observation.observation_hash,
                "arm": self.durable_arm,
                "secret_value_persisted": False,
            },
            hash_field="receipt_hash",
        )
        self._transition_next(
            "observation_finalized",
            {
                "observation_hash": observation.observation_hash,
                "observation_receipt_hash": observation_receipt[
                    "receipt_hash"
                ],
                "valid": observation.valid,
                "success": observation.success,
                "score": observation.score,
            },
        )


class _RawVerifierCheckpointProxyV2:
    """Persist RAW post-agent state immediately before offline tests."""

    def __init__(self, delegate: Any, *, backend: Any) -> None:
        self.delegate = delegate
        self.backend = backend

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def run(self, args: Any, *positional: Any, **kwargs: Any) -> Any:
        command = list(args) if isinstance(args, (list, tuple)) else args
        if (
            isinstance(command, list)
            and len(command) >= 4
            and command[:2] == ["docker", "exec"]
            and "/tests/test.sh" in {str(value) for value in command[3:]}
        ):
            self.backend._checkpoint_raw_before_verifier_v2()
        return self.delegate.run(command, *positional, **kwargs)


class DurableRawSubprocessBackendV2(
    _DurableBackendMixinV2,
    SkillLearnSubprocessBackend,
):
    def __init__(
        self,
        *args: Any,
        durable_state_root: str | Path,
        durable_work_unit_hash: str,
        durable_request_hash: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.durable_state_root = Path(durable_state_root).resolve()
        self.durable_work_unit_hash = durable_work_unit_hash
        self.durable_request_hash = durable_request_hash
        self.durable_arm = "raw"

    @contextmanager
    def _verifier_isolation(
        self,
        runner: Any,
        **kwargs: Any,
    ) -> Iterator[None]:
        with super()._verifier_isolation(runner, **kwargs):
            base_proxy = runner.subprocess
            runner.subprocess = _RawVerifierCheckpointProxyV2(
                base_proxy,
                backend=self,
            )
            try:
                yield
            finally:
                runner.subprocess = base_proxy

    def _checkpoint_raw_before_verifier_v2(self) -> None:
        request = getattr(self, "_active_request", None)
        if not isinstance(request, SkillLearnTrialRequest):
            raise FinancialSemanticReplicationBackendError(
                "RAW verifier started without an active request"
            )
        chain = self._durable_chain()
        if [row.stage for row in chain] != list(WORK_STAGE_ORDER_V2[:2]):
            raise FinancialSemanticReplicationBackendError(
                "RAW verifier started at an unexpected durable stage"
            )
        self._transition_next(
            "agent_completed",
            self._agent_completion_payload(
                request,
                reconciled_after_backend_return=False,
            ),
        )
        self._transition_next(
            "operator_completed",
            {
                "arm": "raw",
                "applicable": False,
                "operator_calls": 0,
                "persisted_before_verifier": True,
            },
        )

    def run(
        self,
        request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> SkillLearnTrialObservation:
        if request.request_hash != self.durable_request_hash:
            raise FinancialSemanticReplicationBackendError(
                "raw request no longer matches durable state"
            )
        if (
            request.variant is not TrialVariant.POLICY_OFF
            or request.treatment_hash != NO_SKILL_TREATMENT_HASH
            or request.program_id is not None
            or request.external_skill_source_receipt_hash
            or skill_source_dir is not None
        ):
            raise FinancialSemanticReplicationBackendError(
                "RAW arm identity or source drifted"
            )
        self._active_request = request
        try:
            observation = super().run(
                request,
                skill_source_dir=skill_source_dir,
                trace_id=trace_id,
            )
            self._complete_after_observation(request, observation)
            return observation
        finally:
            self._active_request = None


class DurableFinancialSemanticSubprocessBackendV2(
    _DurableBackendMixinV2,
    FinancialSemanticSubprocessBackendV1,
):
    def __init__(
        self,
        *args: Any,
        durable_state_root: str | Path,
        durable_work_unit_hash: str,
        durable_request_hash: str,
        expected_precomputed_plan_hash: str,
        expected_program_set_hash: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.durable_state_root = Path(durable_state_root).resolve()
        self.durable_work_unit_hash = durable_work_unit_hash
        self.durable_request_hash = durable_request_hash
        self.durable_arm = "candidate"
        self.expected_precomputed_plan_hash = expected_precomputed_plan_hash
        self.expected_program_set_hash = expected_program_set_hash

    def _execute_financial_plan_before_verifier_v1(
        self,
        *,
        delegate: Any,
        container_name: str,
    ) -> None:
        state = getattr(self._financial_local, "state", None)
        if state is None or state.plan.get("plan_hash") != (
            self.expected_precomputed_plan_hash
        ):
            raise FinancialSemanticReplicationBackendError(
                "runtime semantic plan differs from the precomputed plan"
            )
        chain = self._durable_chain()
        if [row.stage for row in chain] != list(WORK_STAGE_ORDER_V2[:2]):
            raise FinancialSemanticReplicationBackendError(
                "semantic operator started at an unexpected durable stage"
            )
        self._transition_next(
            "agent_completed",
            self._agent_completion_payload(
                # The thread-local state is bound to this request.
                self._active_request,
                reconciled_after_backend_return=False,
            ),
        )
        super()._execute_financial_plan_before_verifier_v1(
            delegate=delegate,
            container_name=container_name,
        )
        evidence = getattr(state, "runtime_evidence", None)
        if not isinstance(evidence, Mapping):
            raise FinancialSemanticReplicationBackendError(
                "semantic runtime evidence was not produced"
            )
        evidence_receipt = atomic_write_hashed_json_v2(
            self.durable_state_root / "semantic_runtime_evidence.json",
            {
                "request_hash": self.durable_request_hash,
                "evidence": dict(evidence),
                "evidence_hash": evidence.get("evidence_hash"),
                "persisted_before_verifier": True,
            },
            hash_field="receipt_hash",
        )
        self._transition_next(
            "operator_completed",
            {
                "arm": "candidate",
                "applicable": True,
                "plan_hash": self.expected_precomputed_plan_hash,
                "semantic_evidence_hash": evidence.get("evidence_hash"),
                "semantic_evidence_receipt_hash": evidence_receipt[
                    "receipt_hash"
                ],
                "operator_calls": 1,
                "online_calls": 0,
                "persisted_before_verifier": True,
            },
        )

    def run(
        self,
        request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> SkillLearnTrialObservation:
        if request.request_hash != self.durable_request_hash:
            raise FinancialSemanticReplicationBackendError(
                "candidate request no longer matches durable state"
            )
        if (
            request.variant is not TrialVariant.POLICY_ON
            or skill_source_dir is None
            or request.program_id != self.expected_program_id
            or request.program_set_hash != self.expected_program_set_hash
            or request.treatment_hash != self.expected_treatment_hash
            or request.external_skill_source_receipt_hash
            != self.expected_external_skill_source_receipt_hash
        ):
            raise FinancialSemanticReplicationBackendError(
                "candidate arm identity or source drifted"
            )
        self._active_request = request
        try:
            observation = super().run(
                request,
                skill_source_dir=skill_source_dir,
                trace_id=trace_id,
            )
            self._complete_after_observation(request, observation)
            return observation
        finally:
            self._active_request = None


def backend_runtime_identity_v2() -> dict[str, Any]:
    payload = {
        "terminal_patch_version": FUTURE_TERMINAL_PATCH_VERSION,
        "work_stage_order": list(WORK_STAGE_ORDER_V2),
        "work_stage_order_hash": stable_hash(
            {"stages": list(WORK_STAGE_ORDER_V2)}
        ),
        "legacy_results_reinterpreted": False,
    }
    return {**payload, "runtime_identity_hash": stable_hash(payload)}
