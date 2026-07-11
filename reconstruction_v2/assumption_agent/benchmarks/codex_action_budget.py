from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..models import stable_hash


CODEX_ACTION_BUDGET_POLICY_VERSION = "codex_jsonl_action_start_budget_v1"
CODEX_ACTION_BUDGET_UNIT = "codex_action_start_v1"
CODEX_ACTION_BUDGET_OVERFLOW_POLICY = "terminate_on_limit_action_start_v1"
CODEX_ACTION_BUDGET_COST_ACCOUNTING_POLICY = (
    "uniform_codex_action_start_cost_v1"
)
CODEX_ACTION_PROCESS_SCOPE_PROCESS_GROUP = "process_group"
CODEX_ACTION_PROCESS_SCOPE_DEDICATED_CONTAINER = "dedicated_container"
CODEX_ACTION_SUPERVISOR_START_EVENT = "assumption.action_budget.started"


@dataclass
class CodexActionBudgetCounter:
    """Inspect one supervisor-owned ``codex exec --json`` trace."""

    limit: int
    _action_events: list[dict[str, Any]] = field(default_factory=list)
    _run_nonces: list[str] = field(default_factory=list)
    _token_usage: dict[str, int] = field(default_factory=dict)
    invalid_action_event_count: int = 0
    invalid_supervisor_start_count: int = 0
    turn_completed_count: int = 0
    turn_failed_count: int = 0
    invalid_terminal_usage_count: int = 0

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise ValueError("Codex action budget limit must be positive")

    @property
    def observed_action_starts(self) -> int:
        return len(self._action_events)

    @property
    def limit_reached(self) -> bool:
        return self.observed_action_starts >= self.limit

    @property
    def overflow_attempted(self) -> bool:
        return self.observed_action_starts > self.limit

    @property
    def action_event_hash(self) -> str:
        return stable_hash(self._action_events)

    @property
    def run_nonce(self) -> str:
        if (
            len(self._run_nonces) == 1
            and len(self._run_nonces[0]) == 32
            and self.invalid_supervisor_start_count == 0
        ):
            return self._run_nonces[0]
        return ""

    @property
    def turn_completed_observed(self) -> bool:
        return self.turn_completed_count > 0

    @property
    def token_usage_complete(self) -> bool:
        return (
            self.turn_completed_count == 1
            and self.turn_failed_count == 0
            and self.invalid_terminal_usage_count == 0
            and bool(self._token_usage)
        )

    @property
    def token_usage(self) -> dict[str, int]:
        return dict(self._token_usage) if self.token_usage_complete else {}

    def observe_line(self, raw_line: str) -> bool:
        """Return true exactly when the configured action-start limit is reached."""

        try:
            row = json.loads(raw_line)
        except (TypeError, json.JSONDecodeError):
            return False
        if not isinstance(row, Mapping):
            return False
        event_type = str(row.get("type") or "")
        if event_type == CODEX_ACTION_SUPERVISOR_START_EVENT:
            nonce = str(row.get("run_nonce") or "")
            if (
                len(nonce) != 32
                or row.get("policy") != CODEX_ACTION_BUDGET_POLICY_VERSION
                or row.get("unit") != CODEX_ACTION_BUDGET_UNIT
                or _integer(row.get("limit")) != self.limit
            ):
                self.invalid_supervisor_start_count += 1
            self._run_nonces.append(nonce)
            return False
        if event_type == "turn.completed":
            self.turn_completed_count += 1
            usage = _normalized_token_usage(row.get("usage"))
            if not usage:
                self.invalid_terminal_usage_count += 1
            else:
                self._token_usage = usage
            return False
        if event_type == "turn.failed":
            self.turn_failed_count += 1
            return False
        if event_type != "item.started":
            return False
        item = row.get("item")
        item_type = str(item.get("type") or "") if isinstance(item, Mapping) else ""
        item_id = str(item.get("id") or "") if isinstance(item, Mapping) else ""
        malformed = not isinstance(item, Mapping) or not item_type or not item_id
        if malformed:
            self.invalid_action_event_count += 1
        self._action_events.append(
            {
                "event_index": self.observed_action_starts + 1,
                "item_id": item_id,
                "item_type": item_type,
                "malformed": malformed,
            }
        )
        return self.observed_action_starts == self.limit


@dataclass(frozen=True)
class CodexActionBudgetReceipt:
    policy: str
    unit: str
    overflow_policy: str
    limit: int
    observed_steps: int
    budget_reached: bool
    trigger_event_index: int | None
    action_event_hash: str
    invalid_action_event_count: int
    run_nonce: str
    trace_sha256: str
    turn_completed_observed: bool
    turn_completed_count: int
    turn_failed_count: int
    invalid_terminal_usage_count: int
    token_usage_complete: bool
    token_usage: Mapping[str, int]
    spawn_error: bool
    sigterm_attempted: bool
    sigterm_delivered: bool
    sigkill_attempted: bool
    sigkill_delivered: bool
    sigkill_grace_upper_bound_seconds: int
    agent_exit_code: int | None
    agent_exit_signal: str | None
    agent_exit_confirmed: bool
    process_group_exit_confirmed: bool
    process_scope: str
    process_baseline_hash: str
    process_task_scan_complete: bool
    residual_process_count: int
    residual_tid_count: int
    residual_sigkill_attempted_count: int
    residual_sigkill_delivered_count: int
    agent_processes_exit_confirmed: bool
    post_trigger_started_count: int
    budget_truncated: bool
    supervisor_hash: str
    raw_content_persisted: bool
    receipt_hash: str

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "CodexActionBudgetReceipt":
        return cls(
            policy=str(payload.get("policy") or ""),
            unit=str(payload.get("unit") or ""),
            overflow_policy=str(payload.get("overflow_policy") or ""),
            limit=_integer(payload.get("limit")),
            observed_steps=_integer(payload.get("observed_steps")),
            budget_reached=payload.get("budget_reached") is True,
            trigger_event_index=(
                _integer(payload.get("trigger_event_index"))
                if payload.get("trigger_event_index") is not None
                else None
            ),
            action_event_hash=str(payload.get("action_event_hash") or ""),
            invalid_action_event_count=_integer(
                payload.get("invalid_action_event_count")
            ),
            run_nonce=str(payload.get("run_nonce") or ""),
            trace_sha256=str(payload.get("trace_sha256") or ""),
            turn_completed_observed=(
                payload.get("turn_completed_observed") is True
            ),
            turn_completed_count=_integer(payload.get("turn_completed_count")),
            turn_failed_count=_integer(payload.get("turn_failed_count")),
            invalid_terminal_usage_count=_integer(
                payload.get("invalid_terminal_usage_count")
            ),
            token_usage_complete=payload.get("token_usage_complete") is True,
            token_usage=_normalized_token_usage(payload.get("token_usage")),
            spawn_error=payload.get("spawn_error") is True,
            sigterm_attempted=payload.get("sigterm_attempted") is True,
            sigterm_delivered=payload.get("sigterm_delivered") is True,
            sigkill_attempted=payload.get("sigkill_attempted") is True,
            sigkill_delivered=payload.get("sigkill_delivered") is True,
            sigkill_grace_upper_bound_seconds=_integer(
                payload.get("sigkill_grace_upper_bound_seconds")
            ),
            agent_exit_code=(
                _integer(payload.get("agent_exit_code"), allow_negative=True)
                if payload.get("agent_exit_code") is not None
                else None
            ),
            agent_exit_signal=(
                str(payload.get("agent_exit_signal"))
                if payload.get("agent_exit_signal")
                else None
            ),
            agent_exit_confirmed=payload.get("agent_exit_confirmed") is True,
            process_group_exit_confirmed=(
                payload.get("process_group_exit_confirmed") is True
            ),
            process_scope=str(payload.get("process_scope") or ""),
            process_baseline_hash=str(payload.get("process_baseline_hash") or ""),
            process_task_scan_complete=(
                payload.get("process_task_scan_complete") is True
            ),
            residual_process_count=_integer(payload.get("residual_process_count")),
            residual_tid_count=_integer(payload.get("residual_tid_count")),
            residual_sigkill_attempted_count=_integer(
                payload.get("residual_sigkill_attempted_count")
            ),
            residual_sigkill_delivered_count=_integer(
                payload.get("residual_sigkill_delivered_count")
            ),
            agent_processes_exit_confirmed=(
                payload.get("agent_processes_exit_confirmed") is True
            ),
            post_trigger_started_count=_integer(
                payload.get("post_trigger_started_count")
            ),
            budget_truncated=payload.get("budget_truncated") is True,
            supervisor_hash=str(payload.get("supervisor_hash") or ""),
            raw_content_persisted=payload.get("raw_content_persisted") is True,
            receipt_hash=str(payload.get("receipt_hash") or ""),
        )

    @property
    def valid(self) -> bool:
        normal_completion = (
            self.agent_exit_code == 0
            and self.agent_exit_signal is None
            and self.token_usage_complete
        )
        controlled_budget_termination = (
            self.budget_reached
            and (self.sigterm_delivered or self.sigkill_delivered)
        )
        signal_state_valid = (
            (not self.sigterm_delivered or self.sigterm_attempted)
            and (not self.sigkill_delivered or self.sigkill_attempted)
            and (not self.sigkill_attempted or self.sigterm_attempted)
        )
        budget_signal_state_valid = (
            (
                self.budget_reached
                and self.sigterm_attempted
                and (controlled_budget_termination or normal_completion)
            )
            or (
                not self.budget_reached
                and not self.sigterm_attempted
                and not self.sigkill_attempted
                and normal_completion
            )
        )
        token_state_valid = (
            self.turn_completed_observed == (self.turn_completed_count > 0)
            and self.token_usage_complete
            == (
                self.turn_completed_count == 1
                and self.turn_failed_count == 0
                and self.invalid_terminal_usage_count == 0
                and bool(self.token_usage)
            )
        )
        residual_state_valid = (
            self.process_task_scan_complete
            and self.residual_process_count <= self.residual_tid_count
            and (self.residual_process_count == 0)
            == (self.residual_tid_count == 0)
            and self.residual_sigkill_delivered_count
            <= self.residual_sigkill_attempted_count
            and (
                self.process_scope
                == CODEX_ACTION_PROCESS_SCOPE_DEDICATED_CONTAINER
                or (
                    self.residual_process_count == 0
                    and self.residual_sigkill_attempted_count == 0
                    and self.residual_sigkill_delivered_count == 0
                )
            )
            and (self.budget_reached or self.residual_process_count == 0)
        )
        return (
            self.policy == CODEX_ACTION_BUDGET_POLICY_VERSION
            and self.unit == CODEX_ACTION_BUDGET_UNIT
            and self.overflow_policy == CODEX_ACTION_BUDGET_OVERFLOW_POLICY
            and self.limit > 0
            and 0 <= self.observed_steps <= self.limit
            and self.budget_reached == (self.observed_steps == self.limit)
            and self.trigger_event_index
            == (self.limit if self.budget_reached else None)
            and not self.spawn_error
            and signal_state_valid
            and budget_signal_state_valid
            and token_state_valid
            and residual_state_valid
            and self.agent_exit_confirmed
            and self.process_group_exit_confirmed
            and self.agent_processes_exit_confirmed
            and self.post_trigger_started_count == 0
            and self.invalid_action_event_count == 0
            and self.budget_truncated
            == (
                self.budget_reached
                and not self.token_usage_complete
                and (self.sigterm_delivered or self.sigkill_delivered)
            )
            and len(self.run_nonce) == 32
            and len(self.trace_sha256) == 64
            and len(self.action_event_hash) == 64
            and self.sigkill_grace_upper_bound_seconds == 15
            and self.process_scope
            in {
                CODEX_ACTION_PROCESS_SCOPE_PROCESS_GROUP,
                CODEX_ACTION_PROCESS_SCOPE_DEDICATED_CONTAINER,
            }
            and len(self.process_baseline_hash) == 64
            and len(self.supervisor_hash) == 64
            and not self.raw_content_persisted
            and self.receipt_hash == stable_hash(self.to_dict())
        )

    @property
    def error_type(self) -> str | None:
        return None if self.valid else "codex_action_budget_receipt_invalid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "unit": self.unit,
            "overflow_policy": self.overflow_policy,
            "limit": self.limit,
            "observed_steps": self.observed_steps,
            "budget_reached": self.budget_reached,
            "trigger_event_index": self.trigger_event_index,
            "action_event_hash": self.action_event_hash,
            "invalid_action_event_count": self.invalid_action_event_count,
            "run_nonce": self.run_nonce,
            "trace_sha256": self.trace_sha256,
            "turn_completed_observed": self.turn_completed_observed,
            "turn_completed_count": self.turn_completed_count,
            "turn_failed_count": self.turn_failed_count,
            "invalid_terminal_usage_count": self.invalid_terminal_usage_count,
            "token_usage_complete": self.token_usage_complete,
            "token_usage": dict(self.token_usage),
            "spawn_error": self.spawn_error,
            "sigterm_attempted": self.sigterm_attempted,
            "sigterm_delivered": self.sigterm_delivered,
            "sigkill_attempted": self.sigkill_attempted,
            "sigkill_delivered": self.sigkill_delivered,
            "sigkill_grace_upper_bound_seconds": (
                self.sigkill_grace_upper_bound_seconds
            ),
            "agent_exit_code": self.agent_exit_code,
            "agent_exit_signal": self.agent_exit_signal,
            "agent_exit_confirmed": self.agent_exit_confirmed,
            "process_group_exit_confirmed": self.process_group_exit_confirmed,
            "process_scope": self.process_scope,
            "process_baseline_hash": self.process_baseline_hash,
            "process_task_scan_complete": self.process_task_scan_complete,
            "residual_process_count": self.residual_process_count,
            "residual_tid_count": self.residual_tid_count,
            "residual_sigkill_attempted_count": (
                self.residual_sigkill_attempted_count
            ),
            "residual_sigkill_delivered_count": (
                self.residual_sigkill_delivered_count
            ),
            "agent_processes_exit_confirmed": (
                self.agent_processes_exit_confirmed
            ),
            "post_trigger_started_count": self.post_trigger_started_count,
            "budget_truncated": self.budget_truncated,
            "supervisor_hash": self.supervisor_hash,
            "raw_content_persisted": self.raw_content_persisted,
        }


def inspect_codex_action_trace(
    lines: Iterable[str],
    *,
    limit: int,
) -> CodexActionBudgetCounter:
    counter = CodexActionBudgetCounter(limit=limit)
    for line in lines:
        counter.observe_line(line)
    return counter


@dataclass(frozen=True)
class CodexActionBudgetAudit:
    valid: bool
    error_type: str | None
    observed_steps: int
    budget_reached: bool
    budget_truncated: bool
    turn_completed_observed: bool
    token_usage_complete: bool
    token_usage: Mapping[str, int]
    process_group_exit_confirmed: bool
    agent_processes_exit_confirmed: bool
    receipt_hash: str
    action_event_hash: str


def audit_codex_action_budget(
    *,
    trace_path: Path,
    receipt_path: Path,
    supervisor_path: Path,
    expected_limit: int,
    expected_process_scope: str = CODEX_ACTION_PROCESS_SCOPE_PROCESS_GROUP,
) -> CodexActionBudgetAudit:
    if not receipt_path.is_file():
        return _invalid_audit("codex_action_budget_receipt_missing")
    if not trace_path.is_file():
        return _invalid_audit("codex_action_budget_trace_missing")
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("receipt is not an object")
        receipt = CodexActionBudgetReceipt.from_mapping(payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return _invalid_audit("codex_action_budget_receipt_malformed")
    try:
        expected_supervisor_hash = _file_sha256(supervisor_path)
        trace_text = trace_path.read_text(encoding="utf-8", errors="replace")
        trace_counter = inspect_codex_action_trace(
            trace_text.splitlines(),
            limit=expected_limit,
        )
        trace_sha256 = _file_sha256(trace_path)
    except OSError:
        return _invalid_audit("codex_action_budget_evidence_unreadable")
    error_type: str | None = None
    if not receipt.valid:
        error_type = receipt.error_type
    elif receipt.limit != expected_limit:
        error_type = "codex_action_budget_limit_mismatch"
    elif receipt.process_scope != expected_process_scope:
        error_type = "codex_action_budget_process_scope_mismatch"
    elif receipt.supervisor_hash != expected_supervisor_hash:
        error_type = "codex_action_budget_supervisor_hash_mismatch"
    elif (
        receipt.trace_sha256 != trace_sha256
        or receipt.run_nonce != trace_counter.run_nonce
        or receipt.observed_steps != trace_counter.observed_action_starts
        or receipt.action_event_hash != trace_counter.action_event_hash
        or receipt.invalid_action_event_count
        != trace_counter.invalid_action_event_count
        or receipt.turn_completed_observed
        != trace_counter.turn_completed_observed
        or receipt.turn_completed_count != trace_counter.turn_completed_count
        or receipt.turn_failed_count != trace_counter.turn_failed_count
        or receipt.invalid_terminal_usage_count
        != trace_counter.invalid_terminal_usage_count
        or receipt.token_usage_complete != trace_counter.token_usage_complete
        or dict(receipt.token_usage) != trace_counter.token_usage
    ):
        error_type = "codex_action_budget_trace_mismatch"
    return CodexActionBudgetAudit(
        valid=error_type is None,
        error_type=error_type,
        observed_steps=receipt.observed_steps,
        budget_reached=receipt.budget_reached,
        budget_truncated=receipt.budget_truncated,
        turn_completed_observed=receipt.turn_completed_observed,
        token_usage_complete=receipt.token_usage_complete,
        token_usage=dict(receipt.token_usage),
        process_group_exit_confirmed=receipt.process_group_exit_confirmed,
        agent_processes_exit_confirmed=(
            receipt.agent_processes_exit_confirmed
        ),
        receipt_hash=receipt.receipt_hash,
        action_event_hash=receipt.action_event_hash,
    )


def _invalid_audit(error_type: str) -> CodexActionBudgetAudit:
    return CodexActionBudgetAudit(
        valid=False,
        error_type=error_type,
        observed_steps=0,
        budget_reached=False,
        budget_truncated=False,
        turn_completed_observed=False,
        token_usage_complete=False,
        token_usage={},
        process_group_exit_confirmed=False,
        agent_processes_exit_confirmed=False,
        receipt_hash="",
        action_event_hash="",
    )


def _normalized_token_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    input_tokens = _strict_nonnegative_integer(value.get("input_tokens"))
    output_tokens = _strict_nonnegative_integer(value.get("output_tokens"))
    if input_tokens is None or output_tokens is None:
        return {}
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    for key in ("cached_input_tokens", "reasoning_output_tokens"):
        if value.get(key) is None:
            continue
        parsed = _strict_nonnegative_integer(value.get(key))
        if parsed is None:
            return {}
        usage[key] = parsed
    return usage


def _strict_nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(
    value: Any,
    *,
    fallback: int = 0,
    allow_negative: bool = False,
) -> int:
    if isinstance(value, bool):
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if allow_negative else max(0, parsed)
