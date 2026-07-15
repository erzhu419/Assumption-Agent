from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..events import Event, JsonlEventSink
from ..models import stable_hash
from ..splits import SplitManifest
from .codex_action_budget import CodexActionBudgetReceipt
from .execution_contract_integration_v2 import (
    ExecutionContractCompileBundleV2,
    ExecutionContractSubprocessBackendV2,
)
from .execution_contract_prompt_v2 import (
    ExecutionContractPromptInjectionReceiptV2,
)
from .paper_protocol import PaperProtocol
from .skilllearn_lifecycle import SkillLearnTrialObservation
from .train_execution_contract_actual_v2 import (
    ACTUAL_REPORT_FILENAME,
    FAILURE_REPORT_FILENAME,
    MODEL_INFERENCE_SLOTS,
    OUTER_WORKERS,
    V320_PROTOCOL_RELATIVE_PATH,
    TrainExecutionContractActualError,
    _configure_environment,
    _prepare_runtime_assets,
    _read_json,
    _sha256_file,
    _verify_canary,
)
from .train_execution_contract_development_v2 import (
    SKILLLEARN_BENCHMARK_RELATIVE_ROOT,
    V320_MANIFEST_RELATIVE_PATH,
    TrainExecutionContractIntegrationV2,
    compile_v320_train_execution_contract_candidates_v2,
)
from .train_outcome_production_runner_v2 import (
    PROVIDER_MODEL_CAPACITY_ERROR_TYPE,
    ProductionTrainCandidateRunnerV2,
    classify_v2_provider_capacity_terminal,
)
from .train_outcome_ranker_v2 import (
    TrainCandidateRunResultV2,
    TrainCandidateWorkUnitV2,
    TrainOutcomeRankerV2,
    TrainOutcomeRankingResultV2,
)
from .v320_train_candidate_material_v2 import (
    V320_EVALUATOR_EPOCH,
    V320_MANIFEST_HASH,
    V320_MODEL,
)


TRAIN_EXECUTION_CONTRACT_RESUME_VERSION = (
    "v320_train_execution_contract_event_resume_offline_ranking_v2"
)
SOURCE_EVENTS_FILENAME = "execution.events.jsonl"
RETRY_EVENTS_FILENAME = "retry.execution.events.jsonl"
SOURCE_WORKER_STATE_DIRNAME = "worker_state"
RECOVERY_BACKEND_IDENTITY_VERSION = (
    "event_ledger_trace_container_recovery_identity_v1"
)
SEMANTIC_RANKING_VERSION = "train_outcome_semantic_ranking_v1"
EXPECTED_ACTIVE_WORK_COUNT = 56
EXPECTED_REPLAY_WORK_COUNT = 476
EXPECTED_RECOVERED_VALID_COUNT = 55
EXPECTED_RETRY_COUNT = 1
MAX_CAPACITY_TRACE_BYTES = 16 * 1024 * 1024


class TrainExecutionContractResumeError(PermissionError):
    """A resume input or result crossed its evidence boundary."""


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainExecutionContractResumeError(f"{label} is not sha256")
    return value


def _verified_event_ledger(
    path: Path,
) -> tuple[tuple[dict[str, Any], ...], str]:
    source = path.resolve(strict=True)
    if path.is_symlink() or not source.is_file():
        raise TrainExecutionContractResumeError(
            "source event ledger is not a regular file"
        )
    try:
        raw = source.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise TrainExecutionContractResumeError(
            "source event ledger is unreadable"
        ) from exc
    rows: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise TrainExecutionContractResumeError(
                f"source event ledger has a blank line at {line_number}"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainExecutionContractResumeError(
                "source event ledger contains malformed JSON"
            ) from exc
        if not isinstance(row, dict) or set(row) != {
            "event",
            "stage",
            "trace_id",
            "payload",
            "payload_hash",
            "event_id",
            "raw_content_persisted",
        }:
            raise TrainExecutionContractResumeError(
                "source event envelope drifted"
            )
        payload = row.get("payload")
        if not isinstance(payload, dict):
            raise TrainExecutionContractResumeError(
                "source event payload is not an object"
            )
        reconstructed = Event(
            event=str(row.get("event") or ""),
            stage=str(row.get("stage") or ""),
            trace_id=str(row.get("trace_id") or ""),
            payload=payload,
        ).to_dict()
        if row != reconstructed or row["event_id"] in event_ids:
            raise TrainExecutionContractResumeError(
                "source event ledger integrity check failed"
            )
        event_ids.add(row["event_id"])
        rows.append(row)
    if not rows:
        raise TrainExecutionContractResumeError(
            "source event ledger is empty"
        )
    return tuple(rows), hashlib.sha256(raw).hexdigest()


def _event_payloads_by_request(
    events: Sequence[Mapping[str, Any]],
    event_name: str,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in events:
        if row.get("event") != event_name:
            continue
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            raise TrainExecutionContractResumeError(
                f"{event_name} payload drifted"
            )
        request_hash = _require_sha256(
            payload.get("request_hash"),
            f"{event_name} request hash",
        )
        if request_hash in result:
            raise TrainExecutionContractResumeError(
                f"{event_name} duplicated a request"
            )
        result[request_hash] = payload
    return result


def _event_rows_by_trace(
    events: Sequence[Mapping[str, Any]],
    event_name: str,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in events:
        if row.get("event") != event_name:
            continue
        trace_id = str(row.get("trace_id") or "")
        if not trace_id or trace_id in result:
            raise TrainExecutionContractResumeError(
                f"{event_name} trace binding drifted"
            )
        result[trace_id] = row
    return result


def _prompt_receipt_from_event(
    payload: Mapping[str, Any],
) -> ExecutionContractPromptInjectionReceiptV2:
    try:
        receipt = ExecutionContractPromptInjectionReceiptV2(
            capsule_hash=str(payload["capsule_hash"]),
            request_hash=str(payload["request_hash"]),
            base_runtime_context_hash=str(
                payload["base_runtime_context_hash"]
            ),
            source_receipt_hash=str(payload["source_receipt_hash"]),
            typed_binding_set_hash=str(
                payload["typed_binding_set_hash"]
            ),
            public_instruction_hash=str(
                payload["public_instruction_hash"]
            ),
            bundle_manifest_hash=str(payload["bundle_manifest_hash"]),
            profile_set_hash=str(payload["profile_set_hash"]),
            profile_count=int(payload["profile_count"]),
            effect_receipt_hashes=tuple(
                str(value) for value in payload["effect_receipt_hashes"]
            ),
            profile_output_sha256s=tuple(
                str(value) for value in payload["profile_output_sha256s"]
            ),
            contract_set_hash=str(
                payload["execution_contract_set_hash"]
            ),
            contract_hashes=tuple(
                str(value) for value in payload["execution_contract_hashes"]
            ),
            profile_contract_binding_set_hash=str(
                payload["profile_contract_binding_set_hash"]
            ),
            profile_contract_binding_hashes=tuple(
                str(value)
                for value in payload["profile_contract_binding_hashes"]
            ),
            fragment_sha256=str(payload["fragment_sha256"]),
            fragment_size=int(payload["fragment_size"]),
            container_path_hash=str(payload["container_path_hash"]),
            container_readback_sha256=str(
                payload["container_readback_sha256"]
            ),
            run_template_before_hash=str(
                payload["run_template_before_hash"]
            ),
            run_template_after_hash=str(
                payload["run_template_after_hash"]
            ),
            effective_prompt_sha256=str(
                payload["effective_prompt_sha256"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainExecutionContractResumeError(
            "prompt receipt event is malformed"
        ) from exc
    expected = {
        **receipt.safe_payload(),
        "receipt_hash": receipt.receipt_hash,
    }
    if dict(payload) != expected:
        raise TrainExecutionContractResumeError(
            "prompt receipt event failed reconstruction"
        )
    return receipt


def _expected_active_requests(
    integration: TrainExecutionContractIntegrationV2,
    request_builder: ProductionTrainCandidateRunnerV2,
) -> dict[str, tuple[TrainCandidateWorkUnitV2, Any]]:
    result: dict[str, tuple[TrainCandidateWorkUnitV2, Any]] = {}
    for candidate in sorted(
        integration.candidate_specs,
        key=lambda row: row.candidate_hash,
    ):
        bundle = integration.candidate_bundles_by_hash[
            candidate.candidate_hash
        ]
        for baseline in integration.raw_projection.baseline_set.rows:
            work = TrainCandidateWorkUnitV2(candidate, baseline)
            if not work.candidate_active:
                continue
            request, _source = request_builder._request_for(work, bundle)
            if request.request_hash in result:
                raise TrainExecutionContractResumeError(
                    "active TRAIN requests are not unique"
                )
            result[request.request_hash] = (work, request)
    if len(result) != EXPECTED_ACTIVE_WORK_COUNT:
        raise TrainExecutionContractResumeError(
            "active TRAIN request grid drifted"
        )
    return result


@dataclass(frozen=True)
class _RecoveredSourceRunV2:
    results_by_work_hash: Mapping[str, TrainCandidateRunResultV2] = field(
        compare=False,
        repr=False,
    )
    retry_work: TrainCandidateWorkUnitV2 = field(
        compare=False,
        repr=False,
    )
    retry_request_hash: str
    event_ledger_sha256: str
    event_count: int
    valid_result_count: int
    invalid_result_count: int
    maximum_source_model_active: int
    recovery_backend_identity_count: int

    @property
    def receipt_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        return {
            "recovery_policy": TRAIN_EXECUTION_CONTRACT_RESUME_VERSION,
            "event_ledger_sha256": self.event_ledger_sha256,
            "event_count": self.event_count,
            "valid_result_count": self.valid_result_count,
            "invalid_result_count": self.invalid_result_count,
            "retry_work_unit_hash": self.retry_work.work_unit_hash,
            "retry_candidate_hash": self.retry_work.candidate.candidate_hash,
            "retry_item_id_hash": self.retry_work.baseline.item_id_hash,
            "retry_request_hash": self.retry_request_hash,
            "recovered_work_unit_hashes": sorted(
                self.results_by_work_hash
            ),
            "recovered_run_receipt_hashes": sorted(
                result.run_receipt_hash
                for result in self.results_by_work_hash.values()
            ),
            "maximum_source_model_active": (
                self.maximum_source_model_active
            ),
            "recovery_backend_identity_count": (
                self.recovery_backend_identity_count
            ),
            "original_backend_instance_hashes_available": False,
            "bit_exact_original_ranking_hash_recoverable": False,
            "raw_event_or_task_content_embedded": False,
        }


def _reconstruct_source_run(
    *,
    source_run_root: Path,
    integration: TrainExecutionContractIntegrationV2,
    request_builder: ProductionTrainCandidateRunnerV2,
) -> _RecoveredSourceRunV2:
    events, ledger_sha256 = _verified_event_ledger(
        source_run_root / SOURCE_EVENTS_FILENAME
    )
    expected = _expected_active_requests(integration, request_builder)
    started = _event_payloads_by_request(events, "skilllearn_trial_started")
    completed = _event_payloads_by_request(
        events,
        "skilllearn_trial_completed",
    )
    injected = _event_payloads_by_request(
        events,
        "skilllearn_pre_agent_execution_contract_prompt_v2_injected",
    )
    receipted = _event_payloads_by_request(
        events,
        "skilllearn_execution_contract_trial_receipted_v2",
    )
    terminal = _event_payloads_by_request(
        events,
        "skilllearn_execution_contract_trial_completed_v2",
    )
    started_rows = _event_rows_by_trace(events, "skilllearn_trial_started")
    network_rows = _event_rows_by_trace(
        events,
        "skilllearn_trial_container_network_restricted",
    )
    network_final_rows = _event_rows_by_trace(
        events,
        "skilllearn_trial_network_usage_finalized",
    )
    expected_request_hashes = set(expected)
    for label, mapping in (
        ("started", started),
        ("completed", completed),
        ("prompt injected", injected),
        ("prompt receipted", receipted),
        ("v2 terminal", terminal),
    ):
        if set(mapping) != expected_request_hashes:
            raise TrainExecutionContractResumeError(
                f"source {label} request grid is incomplete"
            )

    started_trace_by_request = {
        str(row["payload"]["request_hash"]): trace_id
        for trace_id, row in started_rows.items()
    }
    if set(started_trace_by_request) != expected_request_hashes:
        raise TrainExecutionContractResumeError(
            "source trace/request mapping drifted"
        )

    results: dict[str, TrainCandidateRunResultV2] = {}
    invalid: list[tuple[TrainCandidateWorkUnitV2, str]] = []
    backend_identities: set[str] = set()
    for request_hash, (work, request) in expected.items():
        start = started[request_hash]
        base = completed[request_hash]
        prompt_payload = injected[request_hash]
        prompt_event = receipted[request_hash]
        final = terminal[request_hash]
        receipt = _prompt_receipt_from_event(prompt_payload)
        trace_id = started_trace_by_request[request_hash]
        network_row = network_rows.get(trace_id)
        network_final = network_final_rows.get(trace_id)
        if network_row is None or network_final is None:
            raise TrainExecutionContractResumeError(
                "source network lifecycle is incomplete"
            )
        network_payload = network_row.get("payload")
        network_final_payload = network_final.get("payload")
        if (
            not isinstance(network_payload, Mapping)
            or not isinstance(network_final_payload, Mapping)
            or network_payload.get("trial_network_byte_limit")
            != 67_108_864
            or network_final_payload.get("byte_limit") != 67_108_864
            or network_final_payload.get("limit_exceeded") is not False
            or start.get("request_hash") != request.request_hash
            or start.get("skill_source_receipt_hash")
            != request.skill_source_receipt_hash
            or start.get("split") != "train"
            or start.get("variant") != "policy_on"
            or start.get("model") != request.model
            or start.get("max_steps") != request.max_steps
            or start.get("codex_agent_execution_policy_hash")
            != request.codex_agent_execution_policy_hash
            or receipt.request_hash != request_hash
            or receipt.bundle_manifest_hash
            != work.candidate.compile_bundle_manifest_hash
            or prompt_event.get("prompt_receipt_hash")
            != receipt.receipt_hash
            or prompt_event.get("bundle_manifest_hash")
            != receipt.bundle_manifest_hash
            or prompt_event.get("request_hash") != request_hash
            or final.get("request_hash") != request_hash
            or final.get("contract_route_expected") is not True
            or final.get("prompt_receipt_valid") is not True
            or base.get("request_hash") != request_hash
            or base.get("variant") != "policy_on"
            or base.get("success") != final.get("success")
            or base.get("metrics") != final.get("metrics")
            or base.get("error_type") != final.get("error_type")
            or bool(base.get("valid")) != bool(final.get("valid"))
        ):
            raise TrainExecutionContractResumeError(
                "source request execution chain drifted"
            )
        try:
            observation = SkillLearnTrialObservation(
                request=request,
                success=bool(final["success"]),
                score=float(final["score"]),
                metrics={
                    str(key): float(value)
                    for key, value in dict(final["metrics"]).items()
                },
                total_tokens=int(base["total_tokens"]),
                steps=int(base["steps"]),
                duration_seconds=float(base["duration_seconds"]),
                provider_fingerprint=str(base["provider_fingerprint"]),
                fairness_fingerprint=str(base["fairness_fingerprint"]),
                error_type=(
                    str(final["error_type"])
                    if final.get("error_type") is not None
                    else None
                ),
                upstream_result_hash=str(base["upstream_result_hash"]),
                raw_trial_artifacts_persisted=bool(
                    base["raw_trial_artifacts_persisted"]
                ),
                prebuilt_image_key=str(base["prebuilt_image_key"]),
                prebuilt_image_id=str(base["prebuilt_image_id"]),
                prebuilt_cache_reused=bool(base["prebuilt_cache_reused"]),
                agent_runtime_key=str(base["agent_runtime_key"]),
                agent_runtime_version=str(base["agent_runtime_version"]),
                offline_verifier_profile_id=str(
                    base["offline_verifier_profile_id"]
                ),
                offline_verifier_runtime_key=str(
                    base["offline_verifier_runtime_key"]
                ),
                step_budget_policy=str(base["step_budget_policy"]),
                step_budget_unit=str(base["step_budget_unit"]),
                step_budget_limit=int(base["step_budget_limit"]),
                step_budget_truncated=bool(
                    base["step_budget_truncated"]
                ),
                step_budget_token_usage_complete=bool(
                    base["step_budget_token_usage_complete"]
                ),
                step_budget_receipt_hash=str(
                    base["step_budget_receipt_hash"]
                ),
                installed_skill_source_receipt_hash=str(
                    base["installed_skill_source_receipt_hash"]
                ),
                runtime_profile_prompt_delivery_policy=str(
                    prompt_payload["delivery_policy"]
                ),
                runtime_profile_prompt_injection_receipt_hash=(
                    receipt.receipt_hash
                ),
                runtime_profile_effective_prompt_sha256=(
                    receipt.effective_prompt_sha256
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TrainExecutionContractResumeError(
                "source observation is malformed"
            ) from exc
        if (
            observation.observation_hash != final.get("observation_hash")
            or observation.observation_hash
            != prompt_event.get("observation_hash")
        ):
            raise TrainExecutionContractResumeError(
                "source observation failed exact reconstruction"
            )
        container_name_hash = _require_sha256(
            network_payload.get("container_name_hash"),
            "source container name hash",
        )
        recovery_backend_identity = stable_hash(
            {
                "recovery_identity_policy": (
                    RECOVERY_BACKEND_IDENTITY_VERSION
                ),
                "event_ledger_sha256": ledger_sha256,
                "trace_id": trace_id,
                "request_hash": request_hash,
                "container_name_hash": container_name_hash,
            }
        )
        if recovery_backend_identity in backend_identities:
            raise TrainExecutionContractResumeError(
                "source recovery backend identity duplicated"
            )
        backend_identities.add(recovery_backend_identity)
        if not observation.valid:
            invalid.append((work, request_hash))
            continue
        result = TrainCandidateRunResultV2.from_observation(
            work,
            observation,
            execution_backend_instance_hash=recovery_backend_identity,
            prompt_receipt=receipt,
        )
        result.verify(work, integration.raw_projection.baseline_set)
        results[work.work_unit_hash] = result

    if (
        len(results) != EXPECTED_RECOVERED_VALID_COUNT
        or len(invalid) != EXPECTED_RETRY_COUNT
    ):
        raise TrainExecutionContractResumeError(
            "source valid/invalid result cardinality drifted"
        )
    retry_work, retry_request_hash = invalid[0]
    invalid_final = terminal[retry_request_hash]
    if invalid_final.get("error_type") != "codex_turn_failed":
        raise TrainExecutionContractResumeError(
            "source invalid is not the expected terminal class"
        )
    _verify_capacity_artifacts(
        source_run_root=source_run_root,
        retry_work=retry_work,
    )
    acquired = [
        row
        for row in events
        if row.get("event") == "skilllearn_agent_slot_acquired"
    ]
    if len(acquired) != EXPECTED_ACTIVE_WORK_COUNT:
        raise TrainExecutionContractResumeError(
            "source model-slot acquisition grid drifted"
        )
    maximum_source_model_active = max(
        int(row["payload"]["active_count"])
        for row in acquired
    )
    return _RecoveredSourceRunV2(
        results_by_work_hash=results,
        retry_work=retry_work,
        retry_request_hash=retry_request_hash,
        event_ledger_sha256=ledger_sha256,
        event_count=len(events),
        valid_result_count=len(results),
        invalid_result_count=len(invalid),
        maximum_source_model_active=maximum_source_model_active,
        recovery_backend_identity_count=len(backend_identities),
    )


def _verify_capacity_artifacts(
    *,
    source_run_root: Path,
    retry_work: TrainCandidateWorkUnitV2,
) -> None:
    root = (
        source_run_root
        / SOURCE_WORKER_STATE_DIRNAME
        / retry_work.work_unit_hash
    ).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise TrainExecutionContractResumeError(
            "capacity worker state is unavailable"
        )
    traces = [
        path.resolve(strict=True)
        for path in root.rglob("codex.txt")
        if path.is_file() and not path.is_symlink()
    ]
    if len(traces) != 1:
        raise TrainExecutionContractResumeError(
            "capacity trace is missing or ambiguous"
        )
    trace = traces[0]
    try:
        trace.relative_to(root)
    except ValueError as exc:
        raise TrainExecutionContractResumeError(
            "capacity trace escaped its worker root"
        ) from exc
    if trace.stat().st_size > MAX_CAPACITY_TRACE_BYTES:
        raise TrainExecutionContractResumeError(
            "capacity trace exceeded the inspection bound"
        )
    raw = trace.read_bytes()
    if (
        classify_v2_provider_capacity_terminal(raw)
        != PROVIDER_MODEL_CAPACITY_ERROR_TYPE
    ):
        raise TrainExecutionContractResumeError(
            "source terminal was not an exact provider-capacity event"
        )
    receipt_path = trace.with_name("codex_action_budget_receipt.json")
    payload = _read_json(receipt_path, "capacity action-budget receipt")
    try:
        receipt = CodexActionBudgetReceipt.from_mapping(payload)
    except (TypeError, ValueError) as exc:
        raise TrainExecutionContractResumeError(
            "capacity action-budget receipt is malformed"
        ) from exc
    if (
        receipt.receipt_hash != stable_hash(receipt.to_dict())
        or receipt.policy != "codex_jsonl_action_start_budget_v1"
        or receipt.unit != "codex_action_start_v1"
        or receipt.overflow_policy
        != "terminate_on_limit_action_start_v1"
        or receipt.trace_sha256 != hashlib.sha256(raw).hexdigest()
        or receipt.limit != 100
        or receipt.observed_steps <= 0
        or receipt.observed_steps >= receipt.limit
        or receipt.budget_reached
        or receipt.turn_failed_count != 1
        or receipt.agent_exit_code != 1
        or not receipt.agent_processes_exit_confirmed
        or not receipt.agent_exit_confirmed
        or not receipt.process_group_exit_confirmed
        or not receipt.process_task_scan_complete
        or receipt.residual_process_count != 0
        or receipt.residual_tid_count != 0
        or receipt.raw_content_persisted
    ):
        raise TrainExecutionContractResumeError(
            "capacity action-budget receipt failed verification"
        )


def _semantic_ranking_payload(
    ranking: TrainOutcomeRankingResultV2,
) -> dict[str, Any]:
    aggregates = [
        {
            "candidate_hash": row.candidate_hash,
            "invalid_count": row.invalid_count,
            "regression_count": row.regression_count,
            "recovery_count": row.recovery_count,
            "candidate_success_count": row.candidate_success_count,
            "score_delta_units": row.score_delta_units,
            "total_cost_units": row.total_cost_units,
            "static_complexity": row.static_complexity,
            "ranking_key": list(row.ranking_key),
        }
        for row in ranking.aggregates
    ]
    return {
        "semantic_ranking_version": SEMANTIC_RANKING_VERSION,
        "aggregates": aggregates,
        "ordered_candidate_hashes": list(
            ranking.ordered_candidate_hashes
        ),
        "top_candidate_hash": ranking.top_candidate_hash,
        "backend_instance_identities_excluded": True,
    }


@dataclass(frozen=True)
class TrainExecutionContractResumeV2:
    output_root: Path = field(compare=False)
    integration: TrainExecutionContractIntegrationV2 = field(
        compare=False,
        repr=False,
    )
    ranking: TrainOutcomeRankingResultV2 = field(
        compare=False,
        repr=False,
    )
    report: Mapping[str, Any]

    @property
    def report_path(self) -> Path:
        return self.output_root / ACTUAL_REPORT_FILENAME

    def verify(self) -> None:
        self.integration.verify()
        self.ranking.verify()
        persisted = _read_json(self.report_path, "TRAIN resume report")
        without_hash = dict(persisted)
        report_hash = without_hash.pop("report_hash", None)
        if (
            persisted != dict(self.report)
            or report_hash != stable_hash(without_hash)
            or persisted.get("execution_completed") is not True
            or persisted.get("ranking_hash") != self.ranking.ranking_hash
            or persisted.get("recovered_from_events") is not True
            or persisted.get("validation_accessed") is not False
            or persisted.get("test_accessed") is not False
            or persisted.get("online_judge_calls") != 0
        ):
            raise TrainExecutionContractResumeError(
                "TRAIN resume report drifted"
            )


def run_v320_train_execution_contract_resume_v2(
    *,
    project_root: Path,
    output_root: Path,
    source_run_root: Path,
    canary_report_path: Path,
    provider_label: str,
    task_input_cache_root: Path | None = None,
) -> TrainExecutionContractResumeV2:
    """Recover 55 valid results and execute only the capacity-failed route."""

    project = project_root.resolve(strict=True)
    source_root = source_run_root.resolve(strict=True)
    destination = output_root.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("TRAIN resume output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    try:
        source_failure = _read_json(
            source_root / FAILURE_REPORT_FILENAME,
            "source TRAIN failure report",
        )
        source_failure_without_hash = dict(source_failure)
        source_failure_hash = source_failure_without_hash.pop(
            "report_hash",
            None,
        )
        if (
            source_failure_hash != stable_hash(source_failure_without_hash)
            or source_failure.get("execution_completed") is not False
            or source_failure.get("error_type")
            != "TrainOutcomeRankingError"
        ):
            raise TrainExecutionContractResumeError(
                "source TRAIN failure report drifted"
            )
        protocol = PaperProtocol.read(project / V320_PROTOCOL_RELATIVE_PATH)
        if (
            protocol.payload.get("protocol_version") != "3.20.0"
            or protocol.payload.get("model") != V320_MODEL
            or protocol.payload.get("max_steps") != 100
        ):
            raise TrainExecutionContractResumeError(
                "v3.20 execution protocol drifted"
            )
        _configure_environment(protocol)
        canary = _verify_canary(
            canary_report_path.resolve(strict=True),
            provider_label=provider_label,
        )
        manifest = SplitManifest.read(project / V320_MANIFEST_RELATIVE_PATH)
        if manifest.manifest_hash != V320_MANIFEST_HASH:
            raise TrainExecutionContractResumeError(
                "v3.20 execution manifest drifted"
            )
        integration = compile_v320_train_execution_contract_candidates_v2(
            project_root=project,
            output_root=destination / "compile_integration",
        )
        retry_event_sink = JsonlEventSink(
            destination / RETRY_EVENTS_FILENAME
        )
        assets = _prepare_runtime_assets(
            project_root=project,
            destination=destination,
            protocol=protocol,
            manifest=manifest,
            integration=integration,
            event_sink=retry_event_sink,
            task_input_cache_root=task_input_cache_root,
        )
        benchmark_root = (
            project / SKILLLEARN_BENCHMARK_RELATIVE_ROOT
        ).resolve(strict=True)
        execution = protocol.payload["execution"]
        assert isinstance(execution, Mapping)
        trials_root = destination / "retry_worker_state"

        def backend_factory(
            work: TrainCandidateWorkUnitV2,
            bundle: ExecutionContractCompileBundleV2,
        ) -> ExecutionContractSubprocessBackendV2:
            baseline_request = work.baseline.observation.request
            return ExecutionContractSubprocessBackendV2(
                benchmark_root,
                agent_id=baseline_request.agent_id,
                model=baseline_request.model,
                max_steps=baseline_request.max_steps,
                provider_mode=str(protocol.payload["trial_provider_mode"]),
                trials_dir=trials_root / work.work_unit_hash,
                record_upstream=False,
                prebuilt_cache=assets.prebuilt_cache,
                offline_verifier_cache=assets.offline_cache,
                provider_circuit=assets.provider_circuit,
                model_inference_limiter=assets.model_limiter,
                train_action_design_policy=str(
                    execution["train_action_design_policy"]
                ),
                codex_agent_execution_policy=(
                    protocol.codex_agent_execution_policy
                ),
                event_sink=retry_event_sink,
                execution_contract_bundle=bundle,
            )

        production_runner = ProductionTrainCandidateRunnerV2(
            baseline_set=integration.raw_projection.baseline_set,
            candidate_bundles=integration.candidate_bundles_by_hash,
            backend_factory=backend_factory,
            trace_prefix="v320-train-contract-resume-pro01",
        )
        recovered = _reconstruct_source_run(
            source_run_root=source_root,
            integration=integration,
            request_builder=production_runner,
        )
        retry_result = production_runner(recovered.retry_work)
        retry_result.verify(
            recovered.retry_work,
            integration.raw_projection.baseline_set,
        )
        if retry_result.observation.request.request_hash != (
            recovered.retry_request_hash
        ):
            raise TrainExecutionContractResumeError(
                "retry changed the frozen request"
            )
        resolved_results = dict(recovered.results_by_work_hash)
        if retry_result.work_unit_hash in resolved_results:
            raise TrainExecutionContractResumeError(
                "retry duplicated a recovered work unit"
            )
        resolved_results[retry_result.work_unit_hash] = retry_result
        if len(resolved_results) != EXPECTED_ACTIVE_WORK_COUNT:
            raise TrainExecutionContractResumeError(
                "resumed active result grid is incomplete"
            )

        def resolved_runner(
            work: TrainCandidateWorkUnitV2,
        ) -> TrainCandidateRunResultV2:
            try:
                return resolved_results[work.work_unit_hash]
            except KeyError as exc:
                raise TrainExecutionContractResumeError(
                    "ranking requested an unresolved work unit"
                ) from exc

        ranking = TrainOutcomeRankerV2(max_workers=OUTER_WORKERS).rank(
            baseline_set=integration.raw_projection.baseline_set,
            candidates=integration.candidate_specs,
            runner=resolved_runner,
        )
        ranking.verify()
        if (
            len(ranking.run_results) != EXPECTED_ACTIVE_WORK_COUNT
            or len(ranking.replay_receipts) != EXPECTED_REPLAY_WORK_COUNT
            or production_runner.retained_backend_count != EXPECTED_RETRY_COUNT
            or len(production_runner.backend_instance_hashes)
            != EXPECTED_RETRY_COUNT
            or assets.model_limiter.maximum_active != EXPECTED_RETRY_COUNT
            or assets.provider_circuit.error_type is not None
        ):
            raise TrainExecutionContractResumeError(
                "resumed TRAIN execution or concurrency drifted"
            )
        candidate_rows = [
            {
                **candidate.safe_payload(),
                "historical_candidate_subset_hash": (
                    compiled.subset.subset_hash
                ),
                "historical_canonical_set_hash": (
                    compiled.subset.canonical_set_hash
                ),
                "generation": compiled.subset.generation,
            }
            for candidate, compiled in zip(
                ranking.candidates,
                sorted(
                    integration.candidates,
                    key=lambda row: row.spec.candidate_hash,
                ),
                strict=True,
            )
        ]
        semantic_ranking = _semantic_ranking_payload(ranking)
        report_without_hash: dict[str, Any] = {
            "execution_policy": TRAIN_EXECUTION_CONTRACT_RESUME_VERSION,
            "execution_completed": True,
            "provider_canary": canary,
            "source_failure_report_sha256": _sha256_file(
                source_root / FAILURE_REPORT_FILENAME
            ),
            "source_event_ledger_sha256": (
                recovered.event_ledger_sha256
            ),
            "source_event_recovery_receipt": recovered.safe_payload(),
            "source_event_recovery_receipt_hash": recovered.receipt_hash,
            "recovered_from_events": True,
            "recovered_valid_result_count": (
                recovered.valid_result_count
            ),
            "source_invalid_result_count": recovered.invalid_result_count,
            "actual_retry_result_count": EXPECTED_RETRY_COUNT,
            "actual_retry_provider_label_hash": stable_hash(
                {"provider_label": provider_label}
            ),
            "source_capacity_terminal_verified": True,
            "source_original_backend_instance_hashes_available": False,
            "source_bit_exact_original_ranking_hash_recoverable": False,
            "integration_report_hash": integration.report["report_hash"],
            "asset_preflight_report_hash": assets.preflight_report[
                "report_hash"
            ],
            "manifest_hash": manifest.manifest_hash,
            "evaluator_epoch": V320_EVALUATOR_EPOCH,
            "model_hash": stable_hash({"model": V320_MODEL}),
            "candidate_rows": candidate_rows,
            "candidate_row_set_hash": stable_hash(
                {"candidate_rows": candidate_rows}
            ),
            "ranking": ranking.to_dict(),
            "ranking_hash": ranking.ranking_hash,
            "semantic_ranking": semantic_ranking,
            "semantic_ranking_hash": stable_hash(semantic_ranking),
            "outcomes": [row.safe_payload() for row in ranking.outcomes],
            "outcome_set_hash": ranking.outcome_set_hash,
            "run_receipts": [
                row.safe_payload() for row in ranking.run_results
            ],
            "replay_receipts": [
                row.safe_payload() for row in ranking.replay_receipts
            ],
            "source_outer_worker_limit": OUTER_WORKERS,
            "source_model_inference_slot_limit": MODEL_INFERENCE_SLOTS,
            "source_maximum_concurrent_model_calls": (
                recovered.maximum_source_model_active
            ),
            "resume_actual_model_call_count": EXPECTED_RETRY_COUNT,
            "resume_maximum_concurrent_model_calls": (
                assets.model_limiter.maximum_active
            ),
            "ranking_replay_worker_limit": OUTER_WORKERS,
            "ranking_replay_maximum_concurrent_runner_calls": (
                ranking.maximum_concurrent_runner_calls
            ),
            "offline_evaluation_only": True,
            "online_judge_calls": 0,
            "network_fallback_used": False,
            "validation_accessed": False,
            "test_accessed": False,
            "promotion_gate_applied": False,
            "promotion_authorized": False,
            "fresh_development_claim_authorized": False,
            "raw_event_or_task_content_embedded_in_report": False,
            "secret_value_persisted": False,
        }
        report = {
            **report_without_hash,
            "report_hash": stable_hash(report_without_hash),
        }
        (destination / ACTUAL_REPORT_FILENAME).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = TrainExecutionContractResumeV2(
            output_root=destination,
            integration=integration,
            ranking=ranking,
            report=report,
        )
        result.verify()
        return result
    except Exception as exc:
        failure_without_hash = {
            "execution_policy": TRAIN_EXECUTION_CONTRACT_RESUME_VERSION,
            "execution_completed": False,
            "error_type": type(exc).__name__,
            "error_message_hash": stable_hash({"message": str(exc)}),
            "raw_error_persisted": False,
            "secret_value_persisted": False,
        }
        failure = {
            **failure_without_hash,
            "report_hash": stable_hash(failure_without_hash),
        }
        (destination / FAILURE_REPORT_FILENAME).write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recover a v3.20 TRAIN event ledger and retry only its single "
            "provider-capacity work unit."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--canary-report", type=Path, required=True)
    parser.add_argument(
        "--provider-label",
        choices=("plus", "pro"),
        required=True,
    )
    parser.add_argument("--task-input-cache-root", type=Path)
    args = parser.parse_args()
    result = run_v320_train_execution_contract_resume_v2(
        project_root=args.project_root,
        output_root=args.output_root,
        source_run_root=args.source_run_root,
        canary_report_path=args.canary_report,
        provider_label=args.provider_label,
        task_input_cache_root=args.task_input_cache_root,
    )
    top = next(
        row
        for row in result.ranking.aggregates
        if row.candidate_hash == result.ranking.top_candidate_hash
    )
    print(
        json.dumps(
            {
                "execution_completed": True,
                "ranking_hash": result.ranking.ranking_hash,
                "semantic_ranking_hash": result.report[
                    "semantic_ranking_hash"
                ],
                "top_candidate_hash": result.ranking.top_candidate_hash,
                "top_invalid_count": top.invalid_count,
                "top_regression_count": top.regression_count,
                "top_recovery_count": top.recovery_count,
                "top_score_delta_units": top.score_delta_units,
                "recovered_valid_result_count": (
                    EXPECTED_RECOVERED_VALID_COUNT
                ),
                "actual_retry_result_count": EXPECTED_RETRY_COUNT,
                "active_execution_count": len(result.ranking.run_results),
                "inactive_replay_count": len(
                    result.ranking.replay_receipts
                ),
                "online_judge_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
