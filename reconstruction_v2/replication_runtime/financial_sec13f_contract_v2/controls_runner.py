from __future__ import annotations

"""One-shot formal runner for the post-promotion SEC-13F controls.

The completed Replication-C RAW/full observations are reused by hash.  The
only physical work is the preregistered eight skill-only model calls and
eight zero-model operator-only calls.  This module has deliberately no
resume, retry, recovery, replacement, promotion, or sealed-evaluation path.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from assumption_agent.benchmarks.financial_semantic_fresh_runner_v1 import (
    _configure_environment,
)
from assumption_agent.benchmarks.paper_protocol import PaperProtocol
from assumption_agent.benchmarks.skilllearn_lifecycle import (
    SkillLearnModelInferenceLimiter,
    SkillLearnOfflineVerifierRuntimeCache,
    SkillLearnProviderCircuit,
)
from assumption_agent.events import JsonlEventSink
from assumption_agent.models import stable_hash

from replication_runtime.financial_semantic_v2.backends import (
    future_terminal_semantics_v2,
)
from replication_runtime.financial_semantic_v2.durable_state import (
    atomic_write_hashed_json_v2,
    load_durable_stage_chain_v2,
    read_hashed_json_v2,
)
from replication_runtime.financial_semantic_v2.pack import (
    payload_hash,
    sha256_file,
    verify_measurement_view,
)
from replication_runtime.financial_semantic_v2 import runner as _legacy

from .controls import (
    CONTROL_STAGE_ORDER_V1,
    SKILL_ONLY_VERIFIER_NETWORK_AFTER_FILENAME,
    SKILL_ONLY_VERIFIER_NETWORK_BEFORE_FILENAME,
    ControlExecutionV1,
    ControlPlanV1,
    ControlTargetBindingV1,
    ControlWorkUnitV1,
    DurableSkillOnlyBackendV1,
    build_control_plan_v1,
    execute_control_plan_once_v1,
    validate_operator_only_query_receipt_v1,
    validate_prior_measurement_reuse_v1,
)
from .controls_formal import (
    FrozenOperatorOnlyBackendV1,
    prepare_operator_only_shared_runtime_v1,
    validate_controls_execution_freeze_v1,
)
from .provider import (
    load_provider_environment_v1,
)
from .runner import (
    _prewarm_by_item_v2,
    _require_benchmark_tree_hash_v2,
    _validate_execution_freeze_v2,
)
from .treatment import (
    FixedContractCandidateV2,
    load_fixed_contract_candidate_v2,
)


CONTROLS_RUNNER_VERSION = "financial_sec13f_contract_controls_runner_v1"
CONTROLS_REPORT_VERSION = "financial_sec13f_contract_controls_report_v1"
CONTROL_TREATMENT_VERSION = "financial_sec13f_contract_control_treatment_v1"
REPORT_FILENAME = "controls.report.json"
FAILURE_FILENAME = "controls.failure.json"
EVENTS_FILENAME = "controls.events.jsonl"
PLAN_FILENAME = "controls.execution.plan.json"
BATCH_FILENAME = "controls.batch.started.json"
REUSE_FILENAME = "controls.prior-reuse.json"

_SHA256_HEX = frozenset("0123456789abcdef")
_CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("full_minus_raw", "full", "raw"),
    ("skill_only_minus_raw", "skill_only", "raw"),
    ("full_minus_skill_only", "full", "skill_only"),
    ("operator_only_minus_raw", "operator_only", "raw"),
    ("full_minus_operator_only", "full", "operator_only"),
)
_FORBIDDEN_REPORT_KEYS = frozenset(
    {
        "answer",
        "answers",
        "answers_payload",
        "entity",
        "expected_answer",
        "expected_output",
        "gold",
        "gold_payload",
        "instruction",
        "output_payload",
        "plan",
        "query",
        "question",
        "raw_answer",
        "raw_entity",
        "raw_instruction",
        "raw_plan",
        "sealed_payload",
        "trace",
        "trajectory",
    }
)


class ControlsRunnerError(RuntimeError):
    """A formal controls identity or one-shot execution failed closed."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ControlsRunnerError(f"{label} is not a lowercase SHA-256")
    return str(value)


def _reject_raw_report_content(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str) or key.casefold() in _FORBIDDEN_REPORT_KEYS:
                raise ControlsRunnerError(
                    "controls report contains forbidden raw content"
                )
            _reject_raw_report_content(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_raw_report_content(nested)


def _verify_self_hash(
    value: Mapping[str, Any], *, field: str, label: str
) -> str:
    body = dict(value)
    declared = body.pop(field, None)
    if not _is_sha256(declared) or payload_hash(body) != declared:
        raise ControlsRunnerError(f"{label} self hash mismatch")
    return str(declared)


def _read_object(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path).expanduser()
    if source.is_symlink() or not source.is_file():
        raise ControlsRunnerError(f"{label} is not a regular file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControlsRunnerError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ControlsRunnerError(f"{label} is not one JSON object")
    return value


def _regular_file_input(path: str | Path, label: str) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink() or not unresolved.is_file():
        raise ControlsRunnerError(f"{label} is not a regular file")
    return unresolved.resolve(strict=True)


def _bound_project_path(
    project: Path,
    binding: Mapping[str, Any],
    *,
    label: str,
    supplied: str | Path | None = None,
) -> Path:
    relative = binding.get("relative_path")
    expected_sha = binding.get("file_sha256")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not _is_sha256(expected_sha)
    ):
        raise ControlsRunnerError(f"{label} binding is malformed")
    path = project / relative
    if path.is_symlink() or not path.is_file():
        raise ControlsRunnerError(f"{label} bound file is unavailable")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(project)
    except ValueError as exc:
        raise ControlsRunnerError(f"{label} escaped the project") from exc
    if sha256_file(resolved) != expected_sha:
        raise ControlsRunnerError(f"{label} file hash drifted")
    if supplied is not None:
        supplied_path = Path(supplied).expanduser()
        if supplied_path.is_symlink() or not supplied_path.is_file():
            raise ControlsRunnerError(f"supplied {label} is not regular")
        if supplied_path.resolve(strict=True) != resolved:
            raise ControlsRunnerError(f"supplied {label} differs from freeze")
    return resolved


def _require_head_committed_file(project: Path, path: Path, label: str) -> str:
    """Require one activation input to equal its tracked HEAD blob exactly."""

    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=project,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if top.returncode != 0:
        raise ControlsRunnerError(f"{label} has no authoritative Git root")
    git_root = Path(top.stdout.strip()).resolve(strict=True)
    try:
        relative = path.resolve(strict=True).relative_to(git_root).as_posix()
    except ValueError as exc:
        raise ControlsRunnerError(f"{label} escaped the Git repository") from exc
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=git_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    blob = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=git_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            relative,
        ],
        cwd=git_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    commit = head.stdout.strip()
    if (
        head.returncode != 0
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or blob.returncode != 0
        or hashlib.sha256(blob.stdout).hexdigest() != sha256_file(path)
        or status.returncode != 0
        or status.stdout.strip()
    ):
        raise ControlsRunnerError(f"{label} is not cleanly committed at HEAD")
    return commit


def _verify_preregistration(value: Mapping[str, Any]) -> str:
    declared = _verify_self_hash(
        value,
        field="manifest_hash",
        label="controls preregistration",
    )
    design = value.get("control_design")
    arms = design.get("arms") if isinstance(design, Mapping) else None
    execution = design.get("execution") if isinstance(design, Mapping) else None
    evaluation = design.get("evaluation") if isinstance(design, Mapping) else None
    analysis = value.get("analysis_policy")
    sealed = value.get("sealed_boundary")
    if (
        value.get("manifest_version")
        != "financial_sec13f_contract_v2_controls_preregistration_v1"
        or not isinstance(arms, Mapping)
        or set(arms) != {"raw", "full", "skill_only", "operator_only"}
        or not isinstance(execution, Mapping)
        or execution.get("new_physical_work_units") != 16
        or execution.get("new_model_calls") != 8
        or execution.get("incremental_operator_calls") != 8
        or execution.get("incremental_offline_verifier_calls") != 16
        or execution.get("maximum_concurrent_incremental_work_units") != 16
        or execution.get("maximum_concurrent_model_calls") != 8
        or execution.get("all_incremental_futures_submitted_before_results_read")
        is not True
        or execution.get("completed_arm_reexecution_authorized") is not False
        or execution.get("model_replay_authorized") is not False
        or execution.get("provider_retry_authorized") is not False
        or execution.get("operator_replay_authorized") is not False
        or execution.get("verifier_replay_authorized") is not False
        or execution.get("resampling_authorized") is not False
        or execution.get("retry_count") != 0
        or not isinstance(evaluation, Mapping)
        or evaluation.get("offline_evaluation_only") is not True
        or evaluation.get("online_judge_calls") != 0
        or evaluation.get("thresholds") != []
        or evaluation.get("report_preregistered_descriptive_contrasts")
        != [row[0] for row in _CONTRASTS]
        or not isinstance(analysis, Mapping)
        or analysis.get("performance_gate_bound") is not False
        or analysis.get("numeric_performance_threshold_bound") is not False
        or analysis.get("promotion_gate_reopened") is not False
        or analysis.get("candidate_mutation_authorized") is not False
        or not isinstance(sealed, Mapping)
        or sealed.get("sealed_evaluation_authorized") is not False
        or sealed.get("sealed_content_read") is not False
    ):
        raise ControlsRunnerError("controls preregistration policy drifted")
    return declared


def _control_treatment_hash(
    *,
    preregistration_hash: str,
    candidate: FixedContractCandidateV2,
    arm: str,
) -> str:
    if arm == "skill_only":
        semantics = {
            "agent_model_enabled": True,
            "candidate_skill_enabled": True,
            "operator_enabled": False,
            "operator_knockout_enforced": True,
        }
    elif arm == "operator_only":
        semantics = {
            "agent_model_enabled": False,
            "candidate_skill_enabled": False,
            "operator_enabled": True,
            "operator_is_exact_frozen_candidate_content": True,
        }
    else:
        raise ControlsRunnerError("unknown incremental control arm")
    return stable_hash(
        {
            "control_treatment_version": CONTROL_TREATMENT_VERSION,
            "controls_preregistration_hash": preregistration_hash,
            "candidate_id": candidate.candidate_id,
            "recipe_id": candidate.recipe_id,
            "program_set_hash": candidate.program_set_hash,
            "external_skill_source_receipt_hash": (
                candidate.external_skill_source_receipt_hash
            ),
            "arm": arm,
            **semantics,
            "same_frozen_offline_verifier": True,
            "offline_verifier_network": "none",
            "performance_gate_bound": False,
        }
    )


@dataclass(frozen=True)
class PreparedControlPlanV1:
    plan: ControlPlanV1
    targets: tuple[ControlTargetBindingV1, ...]
    reuse_receipt: Mapping[str, Any]


def build_control_plan_from_replication_c_v1(
    *,
    preregistration: Mapping[str, Any],
    replication_c_execution_freeze: Mapping[str, Any],
    replication_c_report: Mapping[str, Any],
    measurement_view: Mapping[str, Any],
    candidate: FixedContractCandidateV2,
    protocol: PaperProtocol,
) -> PreparedControlPlanV1:
    """Reconstruct the exact eight targets without opening trial artifacts."""

    prereg_hash = _verify_preregistration(preregistration)
    candidate.verify()
    view = verify_measurement_view(measurement_view)
    freeze_hash = _verify_self_hash(
        replication_c_execution_freeze,
        field="manifest_hash",
        label="Replication-C execution freeze",
    )
    report_hash = _verify_self_hash(
        replication_c_report,
        field="report_hash",
        label="Replication-C report",
    )
    bindings = preregistration.get("freeze_bindings")
    if not isinstance(bindings, Mapping):
        raise ControlsRunnerError("preregistration bindings are missing")
    candidate_binding = bindings.get("candidate")
    cohort_binding = bindings.get("cohort")
    report_binding = bindings.get("measurement_report")
    c_freeze_binding = bindings.get("replication_c_execution_freeze")
    provider_binding = bindings.get("provider")
    if not all(
        isinstance(row, Mapping)
        for row in (
            candidate_binding,
            cohort_binding,
            report_binding,
            c_freeze_binding,
            provider_binding,
        )
    ):
        raise ControlsRunnerError("preregistration closure is incomplete")
    assert isinstance(candidate_binding, Mapping)
    assert isinstance(cohort_binding, Mapping)
    assert isinstance(report_binding, Mapping)
    assert isinstance(c_freeze_binding, Mapping)
    assert isinstance(provider_binding, Mapping)
    if (
        c_freeze_binding.get("freeze_hash") != freeze_hash
        or report_binding.get("report_hash") != report_hash
        or cohort_binding.get("measurement_view_hash")
        != view.get("measurement_view_hash")
        or cohort_binding.get("item_count") != 8
        or candidate_binding.get("candidate_id") != candidate.candidate_id
        or candidate_binding.get("recipe_id") != candidate.recipe_id
        or candidate_binding.get("program_set_hash")
        != candidate.program_set_hash
        or candidate_binding.get("external_skill_source_receipt_hash")
        != candidate.external_skill_source_receipt_hash
        or provider_binding.get("provider_label") != "plus"
        or provider_binding.get("model") != protocol.payload.get("model")
        or provider_binding.get("api_origin")
        != protocol.payload.get("provider_endpoint_origin")
        or provider_binding.get("pro_fallback_authorized") is not False
    ):
        raise ControlsRunnerError("preregistration identity binding drifted")

    c_candidate = replication_c_execution_freeze.get("candidate")
    c_treatment = replication_c_execution_freeze.get("treatment")
    c_plan = replication_c_execution_freeze.get("plan")
    evaluation = (
        c_treatment.get("evaluation_binding")
        if isinstance(c_treatment, Mapping)
        else None
    )
    if (
        c_candidate != candidate.safe_payload(Path(candidate.candidate_skill_source).parents[1])
        and not (
            isinstance(c_candidate, Mapping)
            and c_candidate.get("candidate_id") == candidate.candidate_id
            and c_candidate.get("recipe_id") == candidate.recipe_id
            and c_candidate.get("program_set_hash") == candidate.program_set_hash
        )
    ):
        # The first equality is used when the project root is the direct
        # candidate parent; the explicit identities keep this helper usable
        # in relocated, read-only test workspaces.
        raise ControlsRunnerError("Replication-C candidate identity drifted")
    if (
        not isinstance(c_treatment, Mapping)
        or not isinstance(evaluation, Mapping)
        or not isinstance(c_plan, Mapping)
        or not _is_sha256(evaluation.get("period_out_treatment_id"))
        or c_treatment.get("recipe_id") != candidate.recipe_id
        or c_treatment.get("program_set_hash") != candidate.program_set_hash
        or c_treatment.get("external_skill_source_receipt_hash")
        != candidate.external_skill_source_receipt_hash
        or not isinstance(c_treatment.get("evaluator_epoch"), str)
        or c_treatment.get("evaluator_epoch")
        != preregistration["control_design"]["evaluation"]["evaluator_epoch"]
        or c_plan.get("plan_hash") != replication_c_report.get("plan_hash")
    ):
        raise ControlsRunnerError("Replication-C treatment binding drifted")

    view_items = view.get("measurement_items")
    if not isinstance(view_items, list) or len(view_items) != 8:
        raise ControlsRunnerError("measurement view target set drifted")
    item_by_instruction_hash: dict[str, Mapping[str, Any]] = {}
    item_by_hash: dict[str, Mapping[str, Any]] = {}
    for item in view_items:
        if not isinstance(item, Mapping):
            raise ControlsRunnerError("measurement item is malformed")
        item_id = item.get("item_id")
        instruction_hash = item.get("instruction_sha256")
        fold = item.get("fold")
        if (
            not isinstance(item_id, str)
            or not item_id
            or not _is_sha256(instruction_hash)
            or isinstance(fold, bool)
            or not isinstance(fold, int)
            or fold not in range(4)
        ):
            raise ControlsRunnerError("measurement item identity drifted")
        item_hash = stable_hash({"item_id": item_id})
        if instruction_hash in item_by_instruction_hash or item_hash in item_by_hash:
            raise ControlsRunnerError("measurement item identity is duplicated")
        item_by_instruction_hash[str(instruction_hash)] = item
        item_by_hash[item_hash] = item

    results = replication_c_report.get("results")
    pair_rows = results.get("pairs") if isinstance(results, Mapping) else None
    evidence_rows = replication_c_report.get("semantic_runtime_evidence")
    report_plan = replication_c_report.get("plan")
    report_work = (
        report_plan.get("work_units") if isinstance(report_plan, Mapping) else None
    )
    if (
        not isinstance(pair_rows, list)
        or len(pair_rows) != 8
        or not isinstance(evidence_rows, list)
        or len(evidence_rows) != 8
        or replication_c_report.get("semantic_runtime_evidence_set_hash")
        != stable_hash(evidence_rows)
        or not isinstance(report_work, list)
        or len(report_work) != 16
    ):
        raise ControlsRunnerError("Replication-C evidence set drifted")
    pair_by_item = {
        str(row.get("item_id_hash")): row
        for row in pair_rows
        if isinstance(row, Mapping)
    }
    candidate_work_by_hash = {
        str(row.get("work_unit_hash")): row
        for row in report_work
        if isinstance(row, Mapping) and row.get("arm") == "candidate"
    }
    if len(pair_by_item) != 8 or len(candidate_work_by_hash) != 8:
        raise ControlsRunnerError("Replication-C pair/work identity drifted")

    evidence_by_item: dict[str, Mapping[str, Any]] = {}
    for wrapper in evidence_rows:
        if not isinstance(wrapper, Mapping):
            raise ControlsRunnerError("Replication-C evidence row is malformed")
        work_hash = wrapper.get("work_unit_hash")
        evidence = wrapper.get("evidence")
        frozen_work = candidate_work_by_hash.get(str(work_hash))
        if not isinstance(evidence, Mapping) or not isinstance(frozen_work, Mapping):
            raise ControlsRunnerError("Replication-C evidence work binding drifted")
        evidence_body = dict(evidence)
        evidence_hash = evidence_body.pop("evidence_hash", None)
        extraction = evidence.get("extraction_receipt")
        if not isinstance(extraction, Mapping):
            raise ControlsRunnerError("Replication-C extraction receipt is missing")
        extraction_body = dict(extraction)
        extraction_hash = extraction_body.pop("receipt_hash", None)
        item = item_by_instruction_hash.get(str(extraction.get("instruction_sha256")))
        if not isinstance(item, Mapping):
            raise ControlsRunnerError("Replication-C evidence is outside the cohort")
        item_hash = stable_hash({"item_id": item["item_id"]})
        query_receipt = evidence.get("query_receipt")
        if (
            evidence_hash != stable_hash(evidence_body)
            or extraction_hash != payload_hash(extraction_body)
            or frozen_work.get("item_id_hash") != item_hash
            or frozen_work.get("request_hash") != evidence.get("request_hash")
            or evidence.get("plan_hash") != extraction.get("plan_hash")
            or evidence.get("extraction_receipt_hash") != extraction_hash
            or evidence.get("program_id") != candidate.recipe_id
            or evidence.get("treatment_hash")
            != evaluation.get("period_out_treatment_id")
            or evidence.get("external_skill_source_receipt_hash")
            != candidate.external_skill_source_receipt_hash
            or evidence.get("candidate_id") != candidate.candidate_id
            or evidence.get("asset_manifest_hash")
            != candidate.asset_manifest_hash
            or evidence.get("operator_source_sha256")
            != candidate.operator_source_sha256
            or evidence.get("answers_payload_persisted") is not False
            or evidence.get("raw_instruction_persisted") is not False
            or evidence.get("gold_content_accessed") is not False
            or evidence.get("pack_content_accessed") is not False
            or not isinstance(query_receipt, Mapping)
        ):
            raise ControlsRunnerError("Replication-C semantic evidence drifted")
        validated_query = validate_operator_only_query_receipt_v1(
            query_receipt,
            expected_plan_hash=str(evidence["plan_hash"]),
            expected_candidate_id=candidate.candidate_id,
            expected_asset_manifest_hash=candidate.asset_manifest_hash,
            expected_contract_hash=str(evidence["contract_hash"]),
            expected_operator_source_sha256=candidate.operator_source_sha256,
        )
        if (
            evidence.get("query_receipt_hash")
            != validated_query.get("receipt_hash")
            or evidence.get("output_sha256")
            != validated_query.get("post_output_sha256")
            or item_hash in evidence_by_item
        ):
            raise ControlsRunnerError("Replication-C output evidence drifted")
        evidence_by_item[item_hash] = evidence

    targets: list[ControlTargetBindingV1] = []
    for item_hash, item in sorted(item_by_hash.items()):
        pair = pair_by_item.get(item_hash)
        evidence = evidence_by_item.get(item_hash)
        if not isinstance(pair, Mapping) or not isinstance(evidence, Mapping):
            raise ControlsRunnerError("control target lacks prior evidence")
        target = ControlTargetBindingV1(
            item_id=str(item["item_id"]),
            fold_id=f"measurement-fold-{int(item['fold'])}",
            prior_pair_id=str(pair.get("pair_id") or ""),
            prior_raw_observation_hash=str(
                pair.get("raw_observation_hash") or ""
            ),
            prior_candidate_observation_hash=str(
                pair.get("candidate_observation_hash") or ""
            ),
            prior_raw_success=bool(pair.get("raw_success")),
            prior_candidate_success=bool(pair.get("candidate_success")),
            candidate_output_sha256=str(evidence.get("output_sha256") or ""),
            typed_plan_hash=str(evidence.get("plan_hash") or ""),
            extraction_receipt_hash=str(
                evidence.get("extraction_receipt_hash") or ""
            ),
        )
        target.verify()
        if (
            pair.get("fold_id") != target.fold_id
            or pair.get("raw_valid") is not True
            or pair.get("candidate_valid") is not True
        ):
            raise ControlsRunnerError("control target prior pair drifted")
        targets.append(target)

    plan = build_control_plan_v1(
        targets=targets,
        controls_preregistration_hash=prereg_hash,
        prior_measurement_report_hash=report_hash,
        prior_measurement_plan_hash=str(replication_c_report["plan_hash"]),
        evaluator_epoch=str(c_treatment["evaluator_epoch"]),
        candidate_recipe_id=candidate.recipe_id,
        candidate_program_set_hash=candidate.program_set_hash,
        candidate_treatment_hash=str(evaluation["period_out_treatment_id"]),
        skill_only_treatment_hash=_control_treatment_hash(
            preregistration_hash=prereg_hash,
            candidate=candidate,
            arm="skill_only",
        ),
        operator_only_treatment_hash=_control_treatment_hash(
            preregistration_hash=prereg_hash,
            candidate=candidate,
            arm="operator_only",
        ),
        external_skill_source_receipt_hash=(
            candidate.external_skill_source_receipt_hash
        ),
        candidate_skill_source=candidate.candidate_skill_source,
        agent_id=str(protocol.payload["agent_id"]),
        model=str(protocol.payload["model"]),
        max_steps=int(protocol.payload["max_steps"]),
        codex_agent_execution_policy_hash=(
            protocol.codex_agent_execution_policy.policy_hash
        ),
    )
    reuse = validate_prior_measurement_reuse_v1(
        replication_c_report,
        targets=targets,
        expected_report_hash=report_hash,
    )
    return PreparedControlPlanV1(
        plan=plan,
        targets=tuple(targets),
        reuse_receipt=reuse,
    )


def _arm_row(
    *,
    arm: str,
    item_id_hash: str,
    fold_id: str,
    observation_hash: str,
    valid: bool,
    success: bool,
    score: float,
    output_sha256: str | None = None,
    candidate_output_hash_match: bool | None = None,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "item_id_hash": item_id_hash,
        "fold_id": fold_id,
        "observation_hash": observation_hash,
        "valid": valid,
        "success": success if valid else None,
        "score": float(score) if valid else None,
        "output_sha256": output_sha256,
        "candidate_output_hash_match": candidate_output_hash_match,
        "raw_content_persisted": False,
    }


def descriptive_controls_results_v1(
    *,
    plan: ControlPlanV1,
    reuse_receipt: Mapping[str, Any],
    execution: ControlExecutionV1,
) -> dict[str, Any]:
    """Create the preregistered four-arm, fold, and contrast description."""

    plan.verify()
    if execution.plan != plan:
        raise ControlsRunnerError("control execution differs from its plan")
    work_by_hash = {row.work_unit_hash: row for row in plan.work_units}
    target_by_item = {
        row.target.item_id_hash: row.target for row in plan.work_units
    }
    if len(target_by_item) != 8:
        raise ControlsRunnerError("control target set is incomplete")
    reuse_rows = reuse_receipt.get("rows")
    if (
        not isinstance(reuse_rows, list)
        or len(reuse_rows) != 16
        or reuse_receipt.get("row_set_hash") != stable_hash(reuse_rows)
        or reuse_receipt.get("executions_performed") != 0
        or reuse_receipt.get("model_calls_performed") != 0
    ):
        raise ControlsRunnerError("prior reuse receipt drifted")
    rows: list[dict[str, Any]] = []
    for row in reuse_rows:
        if not isinstance(row, Mapping):
            raise ControlsRunnerError("prior reuse row is malformed")
        item_hash = str(row.get("item_id_hash") or "")
        target = target_by_item.get(item_hash)
        arm = row.get("arm")
        if target is None or arm not in {"raw", "full"}:
            raise ControlsRunnerError("prior reuse row escaped frozen arms")
        rows.append(
            _arm_row(
                arm=str(arm),
                item_id_hash=item_hash,
                fold_id=target.fold_id,
                observation_hash=_require_sha256(
                    row.get("observation_hash"), "prior observation hash"
                ),
                valid=row.get("valid") is True,
                success=row.get("success") is True,
                score=1.0 if row.get("success") is True else 0.0,
            )
        )
    for result in execution.results:
        work = work_by_hash.get(result.work_unit_hash)
        if work is None:
            raise ControlsRunnerError("incremental result escaped control plan")
        result.verify(work)
        rows.append(
            _arm_row(
                arm=result.arm,
                item_id_hash=work.target.item_id_hash,
                fold_id=work.target.fold_id,
                observation_hash=result.observation_hash,
                valid=result.valid,
                success=result.success,
                score=result.score,
                output_sha256=result.output_sha256,
                candidate_output_hash_match=(
                    result.candidate_output_hash_match
                ),
            )
        )
    rows.sort(key=lambda row: (row["item_id_hash"], row["arm"]))
    by_arm_item = {
        (str(row["arm"]), str(row["item_id_hash"])): row for row in rows
    }
    if len(rows) != 32 or len(by_arm_item) != 32:
        raise ControlsRunnerError("four-arm result grid did not close")

    arm_summaries: list[dict[str, Any]] = []
    for arm in ("raw", "full", "skill_only", "operator_only"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        valid = [row for row in arm_rows if row["valid"]]
        summary = {
            "arm": arm,
            "item_count": len(arm_rows),
            "valid_count": len(valid),
            "invalid_count": len(arm_rows) - len(valid),
            "success_count": sum(row["success"] is True for row in valid),
            "failure_count": sum(row["success"] is False for row in valid),
            "score_total": sum(float(row["score"]) for row in valid),
            "rows": arm_rows,
            "row_set_hash": stable_hash(arm_rows),
        }
        arm_summaries.append(summary)

    contrast_summaries: list[dict[str, Any]] = []
    contrast_rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for name, left_arm, right_arm in _CONTRASTS:
        contrast_rows: list[dict[str, Any]] = []
        for item_hash in sorted(target_by_item):
            left = by_arm_item[(left_arm, item_hash)]
            right = by_arm_item[(right_arm, item_hash)]
            paired_valid = left["valid"] is True and right["valid"] is True
            delta = (
                int(left["success"] is True) - int(right["success"] is True)
                if paired_valid
                else None
            )
            relation = (
                "gain"
                if delta == 1
                else "harm"
                if delta == -1
                else "tie"
                if delta == 0
                else "invalid"
            )
            contrast_rows.append(
                {
                    "contrast": name,
                    "item_id_hash": item_hash,
                    "fold_id": target_by_item[item_hash].fold_id,
                    "paired_valid": paired_valid,
                    "delta": delta,
                    "relation": relation,
                    "raw_content_persisted": False,
                }
            )
        contrast_rows_by_name[name] = contrast_rows
        valid_rows = [row for row in contrast_rows if row["paired_valid"]]
        contrast_summaries.append(
            {
                "contrast": name,
                "left_arm": left_arm,
                "right_arm": right_arm,
                "item_count": 8,
                "paired_valid_count": len(valid_rows),
                "invalid_pair_count": 8 - len(valid_rows),
                "gain_count": sum(row["relation"] == "gain" for row in valid_rows),
                "harm_count": sum(row["relation"] == "harm" for row in valid_rows),
                "tie_count": sum(row["relation"] == "tie" for row in valid_rows),
                "net_delta": sum(int(row["delta"]) for row in valid_rows),
                "rows": contrast_rows,
                "row_set_hash": stable_hash(contrast_rows),
                "descriptive_only": True,
            }
        )

    fold_rows: list[dict[str, Any]] = []
    for fold_id in sorted({target.fold_id for target in target_by_item.values()}):
        fold_arm_rows: list[dict[str, Any]] = []
        for arm in ("raw", "full", "skill_only", "operator_only"):
            selected = [
                row for row in rows if row["fold_id"] == fold_id and row["arm"] == arm
            ]
            valid = [row for row in selected if row["valid"]]
            fold_arm_rows.append(
                {
                    "arm": arm,
                    "item_count": len(selected),
                    "valid_count": len(valid),
                    "invalid_count": len(selected) - len(valid),
                    "success_count": sum(row["success"] is True for row in valid),
                    "failure_count": sum(row["success"] is False for row in valid),
                }
            )
        fold_contrasts: list[dict[str, Any]] = []
        for name, _, _ in _CONTRASTS:
            selected = [
                row
                for row in contrast_rows_by_name[name]
                if row["fold_id"] == fold_id
            ]
            valid = [row for row in selected if row["paired_valid"]]
            fold_contrasts.append(
                {
                    "contrast": name,
                    "item_count": len(selected),
                    "paired_valid_count": len(valid),
                    "invalid_pair_count": len(selected) - len(valid),
                    "gain_count": sum(row["relation"] == "gain" for row in valid),
                    "harm_count": sum(row["relation"] == "harm" for row in valid),
                    "tie_count": sum(row["relation"] == "tie" for row in valid),
                    "net_delta": sum(int(row["delta"]) for row in valid),
                }
            )
        fold_rows.append(
            {
                "fold_id": fold_id,
                "item_count": sum(
                    target.fold_id == fold_id for target in target_by_item.values()
                ),
                "arms": fold_arm_rows,
                "contrasts": fold_contrasts,
                "descriptive_only": True,
            }
        )

    item_relations: list[dict[str, Any]] = []
    for item_hash in sorted(target_by_item):
        item_relations.append(
            {
                "item_id_hash": item_hash,
                "fold_id": target_by_item[item_hash].fold_id,
                "arm_outcomes": [
                    {
                        "arm": arm,
                        "valid": by_arm_item[(arm, item_hash)]["valid"],
                        "success": by_arm_item[(arm, item_hash)]["success"],
                    }
                    for arm in ("raw", "full", "skill_only", "operator_only")
                ],
                "contrast_relations": [
                    {
                        "contrast": name,
                        "relation": next(
                            row["relation"]
                            for row in contrast_rows_by_name[name]
                            if row["item_id_hash"] == item_hash
                        ),
                    }
                    for name, _, _ in _CONTRASTS
                ],
                "raw_content_persisted": False,
            }
        )

    result = {
        "four_arm_item_count": 8,
        "record_count": 32,
        "arms": arm_summaries,
        "arm_set_hash": stable_hash(arm_summaries),
        "contrasts": contrast_summaries,
        "contrast_set_hash": stable_hash(contrast_summaries),
        "folds": fold_rows,
        "fold_set_hash": stable_hash(fold_rows),
        "item_relations": item_relations,
        "item_relation_set_hash": stable_hash(item_relations),
        "descriptive_only": True,
        "performance_thresholds": [],
        "performance_gate_bound": False,
        "raw_content_persisted": False,
    }
    _reject_raw_report_content(result)
    return result


def _require_output_root(path: str | Path) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.exists() or unresolved.is_symlink():
        raise FileExistsError(unresolved)
    parent = unresolved.parent.resolve(strict=True)
    destination = parent / unresolved.name
    destination.mkdir(mode=0o700)
    return destination


def _failure_disposition_v1(execution_started: bool) -> str:
    return (
        "executed_incomplete_no_retry"
        if execution_started
        else "execution_failed_no_retry"
    )


def _durable_failure_snapshot_v1(
    *,
    plan: ControlPlanV1 | None,
    worker_root: Path | None,
) -> dict[str, Any]:
    """Hash-only terminal-state scan; it never resumes or invokes a worker."""

    if plan is None or worker_root is None:
        return {
            "expected_work_unit_count": 16,
            "durable_state_available": False,
            "rows": [],
            "row_set_hash": stable_hash([]),
            "execution_claimed_count": 0,
            "observation_finalized_count": 0,
            "unresolved_work_unit_count": 16,
            "worker_artifact_closure": None,
            "physical_call_count_known": False,
            "raw_content_persisted_in_receipt": False,
        }
    rows: list[dict[str, Any]] = []
    for work in plan.work_units:
        state = worker_root / work.work_unit_hash / "durable"
        try:
            chain = load_durable_stage_chain_v2(
                state,
                stage_order=CONTROL_STAGE_ORDER_V1,
                work_unit_hash=work.work_unit_hash,
                request_hash=work.request_hash,
            )
            stages = [row.stage for row in chain]
            state_valid = True
            error_type = None
        except Exception as exc:
            chain = ()
            stages = []
            state_valid = False
            error_type = type(exc).__name__
        rows.append(
            {
                "work_unit_hash": work.work_unit_hash,
                "arm": work.arm,
                "durable_state_present": state.exists(),
                "durable_state_valid": state_valid,
                "durable_state_error_type": error_type,
                "stage_count": len(stages),
                "latest_stage": stages[-1] if stages else None,
                "latest_stage_hash": chain[-1].stage_hash if chain else None,
                "execution_claimed": "execution_claimed" in stages,
                "agent_completed": "agent_completed" in stages,
                "operator_completed": "operator_completed" in stages,
                "verifier_completed": "verifier_completed" in stages,
                "observation_finalized": "observation_finalized" in stages,
                "raw_content_persisted": False,
            }
        )
    rows.sort(key=lambda row: row["work_unit_hash"])
    finalized = sum(row["observation_finalized"] is True for row in rows)
    closure: Mapping[str, Any] | None = None
    closure_error: dict[str, Any] | None = None
    if worker_root.is_dir() and not worker_root.is_symlink():
        try:
            closure = _legacy._artifact_closure(worker_root)
        except Exception as exc:
            closure_error = {
                "error_type": type(exc).__name__,
                "error_message_hash": stable_hash({"message": str(exc)}),
                "raw_error_persisted": False,
            }
    return {
        "expected_work_unit_count": 16,
        "durable_state_available": True,
        "rows": rows,
        "row_set_hash": stable_hash(rows),
        "execution_claimed_count": sum(
            row["execution_claimed"] is True for row in rows
        ),
        "agent_completed_count": sum(
            row["agent_completed"] is True for row in rows
        ),
        "operator_completed_count": sum(
            row["operator_completed"] is True for row in rows
        ),
        "verifier_completed_count": sum(
            row["verifier_completed"] is True for row in rows
        ),
        "observation_finalized_count": finalized,
        "unresolved_work_unit_count": 16 - finalized,
        "worker_artifact_closure": closure,
        "worker_artifact_closure_error": closure_error,
        "physical_call_count_known": False,
        "no_recovery_or_replay_performed": True,
        "raw_content_persisted_in_receipt": False,
    }


def _verify_bound_runtime_inputs(
    *,
    project: Path,
    preregistration_path: Path,
    preregistration: Mapping[str, Any],
    controls_freeze: Mapping[str, Any],
    controls_freeze_path: Path,
    prewarm_path: Path,
    benchmark_root: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    del controls_freeze_path  # the manifest is self-hashed by its validator
    prereg_bound = controls_freeze.get("preregistration")
    c_freeze_bound = controls_freeze.get("replication_c_execution_freeze")
    prewarm_bound = controls_freeze.get("prewarm")
    reuse_bound = controls_freeze.get("prior_measurement_reuse")
    if not all(
        isinstance(row, Mapping)
        for row in (prereg_bound, c_freeze_bound, prewarm_bound, reuse_bound)
    ):
        raise ControlsRunnerError("controls freeze file closure is incomplete")
    assert isinstance(prereg_bound, Mapping)
    assert isinstance(c_freeze_bound, Mapping)
    assert isinstance(prewarm_bound, Mapping)
    assert isinstance(reuse_bound, Mapping)
    _bound_project_path(
        project,
        prereg_bound,
        label="controls preregistration",
        supplied=preregistration_path,
    )
    c_freeze_path = _bound_project_path(
        project,
        c_freeze_bound,
        label="Replication-C execution freeze",
    )
    bound_prewarm = _bound_project_path(
        project,
        prewarm_bound,
        label="controls prewarm",
        supplied=prewarm_path,
    )
    reuse_path = _bound_project_path(
        project,
        reuse_bound,
        label="prior measurement reuse receipt",
    )
    bindings = preregistration.get("freeze_bindings")
    if not isinstance(bindings, Mapping):
        raise ControlsRunnerError("preregistration file closure is absent")
    report_binding = bindings.get("measurement_report")
    cohort_binding = bindings.get("cohort")
    if not isinstance(report_binding, Mapping) or not isinstance(
        cohort_binding, Mapping
    ):
        raise ControlsRunnerError("preregistration prior evidence is absent")
    report_path = _bound_project_path(
        project,
        report_binding,
        label="Replication-C report",
    )
    view_binding = {
        "relative_path": cohort_binding.get("measurement_view_relative_path"),
        "file_sha256": cohort_binding.get("measurement_view_file_sha256"),
    }
    view_path = _bound_project_path(
        project,
        view_binding,
        label="Replication-C measurement view",
    )
    c_freeze = _read_object(c_freeze_path, "Replication-C execution freeze")
    materialization = c_freeze.get("materialization")
    if not isinstance(materialization, Mapping):
        raise ControlsRunnerError("Replication-C materialization is absent")
    relative_root = materialization.get("benchmark_root_relative_path")
    if (
        not isinstance(relative_root, str)
        or Path(relative_root).is_absolute()
        or ".." in Path(relative_root).parts
    ):
        raise ControlsRunnerError("Replication-C benchmark binding is malformed")
    frozen_benchmark = project / relative_root
    supplied_benchmark = benchmark_root.expanduser()
    if (
        supplied_benchmark.is_symlink()
        or not supplied_benchmark.is_dir()
        or supplied_benchmark.resolve(strict=True)
        != frozen_benchmark.resolve(strict=True)
    ):
        raise ControlsRunnerError("supplied benchmark differs from freeze")
    return c_freeze_path, bound_prewarm, reuse_path, report_path, view_path


def run_controls_v1(
    *,
    project_root: str | Path,
    controls_preregistration_path: str | Path,
    controls_freeze_path: str | Path,
    benchmark_root: str | Path,
    prewarm_path: str | Path,
    output_root: str | Path,
    env_file: str | Path,
    backend_factory_override: Callable[[ControlWorkUnitV1], Any] | None = None,
    cache_preflight: bool = True,
) -> dict[str, Any]:
    """Execute the frozen grid once; test overrides must never be used formally."""

    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    destination = _require_output_root(output_root)
    execution_started = False
    controls_freeze_hash: str | None = None
    controls_freeze_commit: str | None = None
    controls_freeze_file_sha256: str | None = None
    plan_hash: str | None = None
    active_plan: ControlPlanV1 | None = None
    worker_root_for_failure: Path | None = None
    try:
        project = Path(project_root).expanduser().resolve(strict=True)
        prereg_path = _regular_file_input(
            controls_preregistration_path,
            "controls preregistration",
        )
        freeze_path = _regular_file_input(
            controls_freeze_path,
            "controls execution freeze",
        )
        warm_path = _regular_file_input(prewarm_path, "controls prewarm")
        benchmark_unresolved = Path(benchmark_root).expanduser()
        if benchmark_unresolved.is_symlink() or not benchmark_unresolved.is_dir():
            raise ControlsRunnerError("controls benchmark is not a regular tree")
        benchmark = benchmark_unresolved.resolve(strict=True)
        prereg = _read_object(prereg_path, "controls preregistration")
        controls_freeze = _read_object(freeze_path, "controls execution freeze")
        controls_freeze_commit = _require_head_committed_file(
            project,
            freeze_path,
            "controls execution freeze",
        )
        controls_freeze_file_sha256 = sha256_file(freeze_path)
        (
            c_freeze_path,
            bound_prewarm,
            reuse_path,
            report_path,
            view_path,
        ) = _verify_bound_runtime_inputs(
            project=project,
            preregistration_path=prereg_path,
            preregistration=prereg,
            controls_freeze=controls_freeze,
            controls_freeze_path=freeze_path,
            prewarm_path=warm_path,
            benchmark_root=benchmark,
        )
        c_freeze = _read_object(c_freeze_path, "Replication-C execution freeze")
        c_report = _read_object(report_path, "Replication-C report")
        view = _read_object(view_path, "Replication-C measurement view")
        prewarm = _read_object(bound_prewarm, "controls prewarm")
        frozen_reuse = _read_object(reuse_path, "prior reuse receipt")
        candidate = load_fixed_contract_candidate_v2(project)
        protocol_binding = c_freeze.get("paper_protocol")
        if not isinstance(protocol_binding, Mapping):
            raise ControlsRunnerError("frozen paper protocol is missing")
        protocol_path = _bound_project_path(
            project,
            protocol_binding,
            label="paper protocol",
        )
        protocol = PaperProtocol.read(protocol_path)
        if protocol.protocol_hash != protocol_binding.get("protocol_hash"):
            raise ControlsRunnerError("paper protocol hash drifted")
        prepared = build_control_plan_from_replication_c_v1(
            preregistration=prereg,
            replication_c_execution_freeze=c_freeze,
            replication_c_report=c_report,
            measurement_view=view,
            candidate=candidate,
            protocol=protocol,
        )
        plan = prepared.plan
        active_plan = plan
        plan_hash = plan.plan_hash
        controls_freeze_hash = validate_controls_execution_freeze_v1(
            controls_freeze,
            expected_plan=plan,
            project_root=project,
            validate_live_files=True,
        )
        if frozen_reuse != dict(prepared.reuse_receipt):
            raise ControlsRunnerError("frozen prior reuse receipt drifted")

        loaded_provider = load_provider_environment_v1(env_file)
        _, verified_provider = _validate_execution_freeze_v2(
            c_freeze,
            project_root=project,
            candidate=candidate,
            env_file=env_file,
        )
        controls_provider = controls_freeze.get("provider")
        if not isinstance(controls_provider, Mapping):
            raise ControlsRunnerError("controls provider binding is missing")
        if (
            loaded_provider.get("api_key_hmac_sha256")
            != verified_provider.get("api_key_hmac_sha256")
            or loaded_provider.get("model") != verified_provider.get("model")
            or loaded_provider.get("api_origin")
            != verified_provider.get("api_origin")
            or verified_provider.get("provider_label") != "plus"
            or controls_provider.get("provider_binding_hash_from_replication_c")
            != verified_provider.get("binding_hash")
            or controls_provider.get("model") != verified_provider.get("model")
            or controls_provider.get("api_origin")
            != verified_provider.get("api_origin")
            or controls_provider.get("pro_fallback_authorized") is not False
        ):
            raise ControlsRunnerError("Plus provider identity differs from freeze")
        if (
            protocol.payload.get("model") != verified_provider.get("model")
            or protocol.payload.get("provider_endpoint_origin")
            != verified_provider.get("api_origin")
        ):
            raise ControlsRunnerError("provider differs from frozen protocol")
        _configure_environment(protocol)

        c_materialization = c_freeze.get("materialization")
        if not isinstance(c_materialization, Mapping):
            raise ControlsRunnerError("frozen materialization is missing")
        tree_hash = _require_sha256(
            c_materialization.get("benchmark_tree_hash"),
            "benchmark tree hash",
        )
        _require_benchmark_tree_hash_v2(
            benchmark,
            expected_tree_hash=tree_hash,
            stage="controls initial validation",
        )
        verified_view = verify_measurement_view(view)
        item_ids = [
            str(item["item_id"]) for item in verified_view["measurement_items"]
        ]
        prewarm_rows = _prewarm_by_item_v2(
            prewarm=prewarm,
            measurement_view_hash=str(verified_view["measurement_view_hash"]),
            benchmark_tree_hash=tree_hash,
            expected_item_ids=item_ids,
        )

        atomic_write_hashed_json_v2(
            destination / PLAN_FILENAME,
            {
                "runner_version": CONTROLS_RUNNER_VERSION,
                "controls_execution_freeze_hash": controls_freeze_hash,
                "controls_execution_freeze_file_sha256": (
                    controls_freeze_file_sha256
                ),
                "controls_execution_freeze_git_commit": controls_freeze_commit,
                "controls_preregistration_hash": plan.controls_preregistration_hash,
                "control_plan_hash": plan.plan_hash,
                "control_plan_receipt": plan.safe_payload(),
                "prior_reuse_receipt_hash": prepared.reuse_receipt["receipt_hash"],
                "new_physical_work_unit_count": 16,
                "new_model_call_count": 8,
                "new_operator_call_count": 8,
                "raw_content_persisted": False,
                "sealed_content_accessed": False,
            },
            hash_field="receipt_hash",
            refuse_existing=True,
        )
        atomic_write_hashed_json_v2(
            destination / REUSE_FILENAME,
            {
                "reuse_receipt_hash": prepared.reuse_receipt["receipt_hash"],
                "prior_measurement_report_hash": plan.prior_measurement_report_hash,
                "reused_observation_count": 16,
                "reexecuted_observation_count": 0,
                "new_model_calls": 0,
                "new_operator_calls": 0,
                "new_verifier_calls": 0,
                "raw_content_persisted": False,
            },
            hash_field="receipt_hash",
            refuse_existing=True,
        )

        event_sink = JsonlEventSink(destination / EVENTS_FILENAME)
        cache = None
        offline_cache = None
        if cache_preflight:
            cache, offline_cache = _legacy._verify_formal_local_cache(
                benchmark_root=benchmark,
                item_ids=item_ids,
                prewarm_rows=prewarm_rows,
                event_sink=event_sink,
            )
            if cache.cache_only is not True or not isinstance(
                offline_cache, SkillLearnOfflineVerifierRuntimeCache
            ):
                raise ControlsRunnerError("formal controls require local caches")
        elif backend_factory_override is None:
            raise ControlsRunnerError(
                "formal controls may not skip the cache preflight"
            )
        _require_benchmark_tree_hash_v2(
            benchmark,
            expected_tree_hash=tree_hash,
            stage="controls cache preflight",
        )
        atomic_write_hashed_json_v2(
            destination / BATCH_FILENAME,
            {
                "runner_version": CONTROLS_RUNNER_VERSION,
                "controls_execution_freeze_hash": controls_freeze_hash,
                "control_plan_hash": plan.plan_hash,
                "new_physical_work_unit_count": 16,
                "maximum_concurrent_work_units": 16,
                "new_model_call_count": 8,
                "maximum_concurrent_model_calls": 8,
                "new_operator_call_count": 8,
                "all_futures_submitted_before_results_read": True,
                "provider_label": "plus",
                "retry_authorized": False,
                "replay_authorized": False,
                "recovery_authorized": False,
                "resampling_authorized": False,
                "offline_evaluation_only": True,
                "online_judge_calls": 0,
            },
            hash_field="receipt_hash",
            refuse_existing=True,
        )
        worker_root = destination / "worker_state"
        worker_root_for_failure = worker_root
        limiter = SkillLearnModelInferenceLimiter(8)
        provider_circuit_ids: set[int] = set()
        backend_ids: set[int] = set()

        if backend_factory_override is None:
            assert cache is not None and offline_cache is not None
            operator_shared = prepare_operator_only_shared_runtime_v1(
                project_root=project,
                benchmark_root=benchmark,
                asset_path=candidate.operator_asset_path,
                prewarm=prewarm,
                expected_program_id=candidate.recipe_id,
                expected_treatment_hash=plan.operator_only_treatment_hash,
                expected_external_skill_source_receipt_hash=(
                    candidate.external_skill_source_receipt_hash
                ),
                event_sink=event_sink,
            )

            def backend_factory(work: ControlWorkUnitV1) -> Any:
                state = worker_root / work.work_unit_hash / "durable"
                if work.arm == "operator_only":
                    backend: Any = FrozenOperatorOnlyBackendV1(
                        work,
                        shared=operator_shared,
                    )
                else:
                    request = work.request
                    if request is None:
                        raise ControlsRunnerError("skill-only request is missing")
                    circuit = SkillLearnProviderCircuit()
                    if id(circuit) in provider_circuit_ids:
                        raise ControlsRunnerError("provider circuit was reused")
                    provider_circuit_ids.add(id(circuit))
                    backend = DurableSkillOnlyBackendV1(
                        benchmark,
                        control_work=work,
                        durable_state_root=state,
                        agent_id=request.agent_id,
                        model=request.model,
                        max_steps=request.max_steps,
                        provider_mode="openai_compatible",
                        trials_dir=worker_root / work.work_unit_hash / "trials",
                        record_upstream=True,
                        prebuilt_cache=cache,
                        offline_verifier_cache=offline_cache,
                        provider_circuit=circuit,
                        model_inference_limiter=limiter,
                        codex_agent_execution_policy=(
                            protocol.codex_agent_execution_policy
                        ),
                        event_sink=event_sink,
                    )
                if id(backend) in backend_ids:
                    raise ControlsRunnerError("control backend instance was reused")
                backend_ids.add(id(backend))
                return backend

        else:
            backend_factory = backend_factory_override

        execution_started = True
        with future_terminal_semantics_v2():
            execution = execute_control_plan_once_v1(
                plan=plan,
                worker_root=worker_root,
                backend_factory=backend_factory,
            )
        if backend_factory_override is None and (
            len(provider_circuit_ids) != 8 or len(backend_ids) != 16
        ):
            raise ControlsRunnerError("independent backend closure drifted")
        stage_heads: list[dict[str, Any]] = []
        skill_only_network_rows: list[dict[str, Any]] = []
        for work in plan.work_units:
            chain = load_durable_stage_chain_v2(
                worker_root / work.work_unit_hash / "durable",
                stage_order=CONTROL_STAGE_ORDER_V1,
                work_unit_hash=work.work_unit_hash,
                request_hash=work.request_hash,
            )
            if [row.stage for row in chain] != list(CONTROL_STAGE_ORDER_V1):
                raise ControlsRunnerError("control durable stage chain is incomplete")
            stage_heads.append(
                {
                    "work_unit_hash": work.work_unit_hash,
                    "arm": work.arm,
                    "final_stage_hash": chain[-1].stage_hash,
                    "stage_count": len(chain),
                    "raw_content_persisted": False,
                }
            )
            if work.arm == "skill_only":
                before = read_hashed_json_v2(
                    worker_root
                    / work.work_unit_hash
                    / "durable"
                    / SKILL_ONLY_VERIFIER_NETWORK_BEFORE_FILENAME,
                    hash_field="receipt_hash",
                )
                after = read_hashed_json_v2(
                    worker_root
                    / work.work_unit_hash
                    / "durable"
                    / SKILL_ONLY_VERIFIER_NETWORK_AFTER_FILENAME,
                    hash_field="receipt_hash",
                )
                verifier_payload = chain[-2].payload
                if (
                    before.get("arm") != "skill_only"
                    or before.get("phase") != "before_verifier"
                    or before.get("attached_network_count") != 0
                    or before.get("verifier_network") != "none"
                    or before.get("model_secret_env_available_to_verifier")
                    is not False
                    or before.get("raw_content_persisted") is not False
                    or after.get("arm") != "skill_only"
                    or after.get("phase") != "after_verifier"
                    or after.get("attached_network_count") != 0
                    or after.get("verifier_network") != "none"
                    or after.get("model_secret_env_available_to_verifier")
                    is not False
                    or after.get("raw_content_persisted") is not False
                    or verifier_payload.get("verifier_network") != "none"
                    or verifier_payload.get(
                        "verifier_network_before_receipt_hash"
                    )
                    != before.get("receipt_hash")
                    or verifier_payload.get(
                        "verifier_network_after_receipt_hash"
                    )
                    != after.get("receipt_hash")
                ):
                    raise ControlsRunnerError(
                        "skill-only verifier network-none evidence drifted"
                    )
                skill_only_network_rows.append(
                    {
                        "work_unit_hash": work.work_unit_hash,
                        "verifier_network": "none",
                        "before_receipt_hash": before["receipt_hash"],
                        "after_receipt_hash": after["receipt_hash"],
                        "before_attached_network_count": 0,
                        "after_attached_network_count": 0,
                        "raw_content_persisted": False,
                    }
                )
        stage_heads.sort(key=lambda row: row["work_unit_hash"])
        skill_only_network_rows.sort(key=lambda row: row["work_unit_hash"])
        if len(skill_only_network_rows) != 8:
            raise ControlsRunnerError(
                "skill-only verifier network-none closure is incomplete"
            )
        results = descriptive_controls_results_v1(
            plan=plan,
            reuse_receipt=prepared.reuse_receipt,
            execution=execution,
        )
        incremental_valid = all(row.valid for row in execution.results)
        operator_identity_valid = all(
            row.candidate_output_hash_match is True
            for row in execution.results
            if row.arm == "operator_only"
        )
        disposition = (
            "executed_complete"
            if incremental_valid and operator_identity_valid
            else "executed_incomplete_no_retry"
        )
        _require_benchmark_tree_hash_v2(
            benchmark,
            expected_tree_hash=tree_hash,
            stage="controls execution completion",
        )
        closure = _legacy._artifact_closure(worker_root)
        body = {
            "report_version": CONTROLS_REPORT_VERSION,
            "runner_version": CONTROLS_RUNNER_VERSION,
            "execution_completed": True,
            "execution_disposition": disposition,
            "evidence_valid": incremental_valid and operator_identity_valid,
            "controls_execution_freeze_hash": controls_freeze_hash,
            "controls_execution_freeze_file_sha256": (
                controls_freeze_file_sha256
            ),
            "controls_execution_freeze_git_commit": controls_freeze_commit,
            "controls_preregistration_hash": plan.controls_preregistration_hash,
            "control_plan_hash": plan.plan_hash,
            "control_plan_receipt": plan.safe_payload(),
            "prior_measurement_report_hash": plan.prior_measurement_report_hash,
            "prior_measurement_plan_hash": plan.prior_measurement_plan_hash,
            "prior_reuse_receipt_hash": prepared.reuse_receipt["receipt_hash"],
            "prior_reused_observation_count": 16,
            "prior_reexecuted_observation_count": 0,
            "incremental_execution": execution.safe_payload(),
            "results": results,
            "durable_stage_heads": stage_heads,
            "durable_stage_head_set_hash": stable_hash(stage_heads),
            "skill_only_verifier_network": "none",
            "skill_only_verifier_network_evidence": skill_only_network_rows,
            "skill_only_verifier_network_evidence_set_hash": stable_hash(
                skill_only_network_rows
            ),
            "worker_artifact_closure": closure,
            "new_physical_work_unit_count": 16,
            "new_model_call_count": execution.model_call_count,
            "new_operator_call_count": execution.operator_call_count,
            "new_offline_verifier_call_count": len(execution.results),
            "model_inference_slot_limit": 8,
            "maximum_concurrent_model_calls": (
                limiter.maximum_active if backend_factory_override is None else None
            ),
            "maximum_concurrent_physical_work_units": (
                execution.maximum_active_backend_calls
            ),
            "all_futures_submitted_before_results_read": True,
            "independent_provider_circuit_count": (
                len(provider_circuit_ids)
                if backend_factory_override is None
                else None
            ),
            "independent_backend_instance_count": execution.backend_instance_count,
            "provider_label": "plus",
            "provider_binding_hash": verified_provider["binding_hash"],
            "provider_identity_sidecar_hash": verified_provider[
                "identity_sidecar_hash"
            ],
            "provider_selection_receipt_hash": verified_provider[
                "selection_receipt_hash"
            ],
            "provider_retry_count": 0,
            "model_replay_count": 0,
            "operator_replay_count": 0,
            "verifier_replay_count": 0,
            "resampling_used": False,
            "mid_batch_provider_switch_used": False,
            "offline_evaluation_only": True,
            "offline_judge_only": True,
            "online_judge_calls": 0,
            "network_fallback_used": False,
            "benchmark_tree_rehashed_after_execution": True,
            "benchmark_tree_hash": tree_hash,
            "prewarm_hash": prewarm["prewarm_hash"],
            "prebuilt_cache_only": True,
            "performance_gate_applied": False,
            "performance_thresholds_bound": False,
            "operator_output_hash_match_is_protocol_identity_check": True,
            "operator_output_hash_match_is_performance_gate": False,
            "controls_are_mechanism_characterization": True,
            "generalization_claim_authorized": False,
            "skill_only_vs_reused_arms_contemporaneously_randomized": False,
            "promotion_status_changed": False,
            "candidate_mutation_authorized": False,
            "sealed_evaluation_authorized": False,
            "sealed_content_accessed": False,
            "answers_payload_persisted_in_report": False,
            "expected_output_content_persisted_in_report": False,
            "raw_content_persisted_in_report": False,
            "private_worker_trace_retained": True,
            "private_verifier_ctrf_retained": True,
            "public_archive_must_exclude_private_worker_tree": True,
            "secret_value_persisted": False,
            "retry_authorized": False,
            "replay_authorized": False,
            "recovery_authorized": False,
        }
        _reject_raw_report_content(body)
        return atomic_write_hashed_json_v2(
            destination / REPORT_FILENAME,
            body,
            hash_field="report_hash",
            refuse_existing=True,
        )
    except Exception as exc:
        durable_snapshot = _durable_failure_snapshot_v1(
            plan=active_plan,
            worker_root=worker_root_for_failure,
        )
        failure = {
            "report_version": "financial_sec13f_contract_controls_failure_v1",
            "runner_version": CONTROLS_RUNNER_VERSION,
            "execution_completed": False,
            "execution_started": execution_started,
            "execution_disposition": _failure_disposition_v1(
                execution_started
            ),
            "failure_kind": (
                "runtime_exception_before_grid_closure"
                if execution_started
                else "preexecution_validation_failure"
            ),
            "controls_execution_freeze_hash": controls_freeze_hash,
            "controls_execution_freeze_file_sha256": (
                controls_freeze_file_sha256
            ),
            "controls_execution_freeze_git_commit": controls_freeze_commit,
            "control_plan_hash": plan_hash,
            "error_type": type(exc).__name__,
            "error_message_hash": stable_hash({"message": str(exc)}),
            "raw_error_persisted": False,
            "durable_failure_snapshot": durable_snapshot,
            "invalid_or_unresolved_results_retained_by_hash": True,
            "private_worker_trace_may_be_retained": execution_started,
            "public_archive_must_exclude_private_worker_tree": True,
            "retry_authorized": False,
            "replay_authorized": False,
            "recovery_authorized": False,
            "resampling_authorized": False,
            "candidate_mutation_authorized": False,
            "sealed_evaluation_authorized": False,
            "sealed_content_accessed": False,
            "secret_value_persisted": False,
        }
        try:
            atomic_write_hashed_json_v2(
                destination / FAILURE_FILENAME,
                failure,
                hash_field="report_hash",
                refuse_existing=True,
            )
        except Exception as write_exc:
            raise ControlsRunnerError(
                "atomic controls failure receipt could not be persisted"
            ) from write_exc
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--controls-preregistration", type=Path, required=True)
    parser.add_argument("--controls-freeze", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--prewarm", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_controls_v1(
            project_root=args.project_root,
            controls_preregistration_path=args.controls_preregistration,
            controls_freeze_path=args.controls_freeze,
            benchmark_root=args.benchmark_root,
            prewarm_path=args.prewarm,
            output_root=args.output_root,
            env_file=args.env_file,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "execution_completed": False,
                    "error_type": type(exc).__name__,
                    "error_message_hash": stable_hash({"message": str(exc)}),
                    "retry_authorized": False,
                    "replay_authorized": False,
                    "recovery_authorized": False,
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "execution_disposition": report["execution_disposition"],
                "evidence_valid": report["evidence_valid"],
                "new_model_call_count": report["new_model_call_count"],
                "new_operator_call_count": report["new_operator_call_count"],
                "retry_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0 if report["execution_disposition"] == "executed_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTROLS_REPORT_VERSION",
    "CONTROLS_RUNNER_VERSION",
    "ControlsRunnerError",
    "PreparedControlPlanV1",
    "build_control_plan_from_replication_c_v1",
    "descriptive_controls_results_v1",
    "main",
    "run_controls_v1",
]
