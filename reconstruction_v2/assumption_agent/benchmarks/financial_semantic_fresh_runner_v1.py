from __future__ import annotations

"""Prospective fresh-validation runner for the frozen financial candidate.

The module deliberately separates metadata-only planning from execution.  A
plan contains nine RAW work units and exactly one physically executed semantic
work unit.  Candidate outcomes on the other eight, predeclared inactive routes
are exact projections of their RAW observations; they never consume model
calls.  Official HippoRAG is recorded as non-applicable and is never
substituted with a similarly named local baseline.
"""

import argparse
import concurrent.futures
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..events import JsonlEventSink
from ..models import SplitName, stable_hash
from ..secure_env import (
    configured_api_origin,
    configured_model,
    configured_skilllearn_provider_mode,
)
from .docker_egress import DockerEgressPolicy, configured_trial_network_byte_limit
from .financial_semantic_integration_v1 import (
    FinancialSemanticSubprocessBackendV1,
    SharedFinancialSemanticPlannerV1,
)
from .financial_semantic_treatment_freeze_v1 import (
    FINANCIAL_SEMANTIC_TREATMENT_FREEZE_VERSION,
    HIPPORAG_STATUS_NOT_APPLICABLE,
    load_financial_semantic_treatment_freeze_v1,
)
from .offline_verifier import (
    SkillLearnOfflineVerifierRuntimeCache,
    offline_verifier_profile_for_family,
)
from .paper_protocol import PaperProtocol
from .prewarm import (
    FrozenTaskInputPrebuiltImageCache,
    validate_development_prewarm_receipt,
)
from .skilllearn_compiler import (
    NO_SKILL_TREATMENT_HASH,
    verify_skill_source_tree,
)
from .skilllearn_lifecycle import (
    SkillLearnModelInferenceLimiter,
    SkillLearnProviderCircuit,
    SkillLearnSubprocessBackend,
    SkillLearnTrialObservation,
    SkillLearnTrialRequest,
    TrialVariant,
)
from .task_input_freeze import (
    expected_prewarm_closure_rows,
    load_frozen_task_input_closure,
)
from .train_execution_contract_development_v2 import (
    SKILLLEARN_BENCHMARK_RELATIVE_ROOT,
    V320_MANIFEST_RELATIVE_PATH,
)
from .train_typed_assignment_crossfit_v3 import (
    TYPED_ASSIGNMENT_PROVIDER_POLICY,
    _verify_provider_selection_receipt_v3,
)
from .v320_train_candidate_material_v2 import V320_MODEL
from .v320_train_candidate_material_v2 import V320_SOURCE_RELATIVE_ROOT
from ..splits import SplitManifest


FINANCIAL_SEMANTIC_FRESH_RUNNER_VERSION = (
    "financial_semantic_fresh_flat_paired_runner_v1"
)
FRESH_SPLIT_RELATIVE_PATH = (
    "manifests/skilllearn_fresh_provenance_split_v1.json"
)
V320_PROTOCOL_RELATIVE_PATH = (
    "manifests/skilllearn_paper_protocol_v3_20_ruoli_gpt54mini.json"
)
V320_PREWARM_RELATIVE_PATH = (
    f"{V320_SOURCE_RELATIVE_ROOT}/development_prewarm.json"
)
FRESH_SPLIT_VERSION = "skilllearn_fresh_provenance_split_v1"
ACTIVE_FRESH_ITEM_ID = "financial-analysis-4"
ACTIVE_FAMILY = "financial-analysis"
EXPECTED_FRESH_ITEM_COUNT = 9
EXPECTED_RAW_EXECUTION_COUNT = 9
EXPECTED_SEMANTIC_EXECUTION_COUNT = 1
EXPECTED_PHYSICAL_WORK_UNIT_COUNT = 10
EXPECTED_INACTIVE_PROJECTION_COUNT = 8
DEFAULT_MODEL_INFERENCE_SLOTS = EXPECTED_PHYSICAL_WORK_UNIT_COUNT
DEFAULT_OUTER_WORKERS = EXPECTED_PHYSICAL_WORK_UNIT_COUNT
EXECUTION_EVENTS_FILENAME = "execution.events.jsonl"
ASSET_PREFLIGHT_FILENAME = "asset_preflight.report.json"
REPORT_FILENAME = "fresh_paired.report.json"
FAILURE_FILENAME = "fresh_paired.failure.json"
INACTIVE_PROJECTION_POLICY = "exact_raw_inactive_route_projection_v1"
OFFICIAL_HIPPORAG_STATUS = HIPPORAG_STATUS_NOT_APPLICABLE


class FinancialSemanticFreshRunnerError(PermissionError):
    """A frozen prospective execution boundary was crossed."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise FinancialSemanticFreshRunnerError(
            "required frozen file is not a regular file"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FinancialSemanticFreshRunnerError(
            f"{label} is not a regular file"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinancialSemanticFreshRunnerError(
            f"{label} is unreadable"
        ) from exc
    if not isinstance(payload, dict):
        raise FinancialSemanticFreshRunnerError(
            f"{label} must contain one object"
        )
    return payload


def _resolve_project_path(
    project: Path,
    relative_path: object,
    *,
    directory: bool,
) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise FinancialSemanticFreshRunnerError(
            "frozen project-relative path is missing"
        )
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise FinancialSemanticFreshRunnerError(
            "frozen project-relative path is unsafe"
        )
    resolved = (project / relative).resolve(strict=True)
    if project != resolved and project not in resolved.parents:
        raise FinancialSemanticFreshRunnerError(
            "frozen project-relative path escaped project root"
        )
    if directory and not resolved.is_dir():
        raise FinancialSemanticFreshRunnerError(
            "frozen project-relative directory is unavailable"
        )
    if not directory and not resolved.is_file():
        raise FinancialSemanticFreshRunnerError(
            "frozen project-relative file is unavailable"
        )
    return resolved


def _worker_artifact_closure(worker_root: Path) -> dict[str, Any]:
    """Hash persisted trial logs while omitting raw locator text.

    Upstream may leave a private ``agent/tmp`` staging directory containing
    task inputs.  It is intentionally outside the post-hoc agent/verifier log
    closure; the auditable files are the Codex trace, action-budget receipt and
    offline-verifier artifacts.
    """

    root = worker_root.resolve(strict=True)
    rows: list[dict[str, Any]] = []
    for current, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in directory_names:
            candidate = current_path / name
            relative = candidate.relative_to(root)
            if candidate.is_symlink():
                raise FinancialSemanticFreshRunnerError(
                    "worker artifact tree contains a linked directory"
                )
            if len(relative.parts) >= 2 and relative.parts[-2:] == (
                "agent",
                "tmp",
            ):
                continue
            kept_directories.append(name)
        directory_names[:] = kept_directories
        for name in sorted(file_names):
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise FinancialSemanticFreshRunnerError(
                    "worker artifact tree contains a linked or special file"
                )
            relative = path.relative_to(root).as_posix()
            rows.append(
                {
                    "locator_hash": stable_hash(
                        {"relative_path": relative}
                    ),
                    "file_sha256": _sha256_file(path),
                    "file_size": path.stat().st_size,
                }
            )
    rows.sort(key=lambda row: row["locator_hash"])
    if not rows:
        raise FinancialSemanticFreshRunnerError(
            "recorded worker artifact closure is empty"
        )
    return {
        "closure_policy": "agent_and_offline_verifier_artifact_hashes_v1",
        "file_count": len(rows),
        "files": rows,
        "file_set_hash": stable_hash(rows),
        "raw_locator_text_persisted_in_report": False,
        "raw_artifacts_retained_under_worker_state": True,
    }


@dataclass(frozen=True)
class FreshSplitMetadataV1:
    manifest_hash: str
    item_ids: tuple[str, ...]
    family_by_id: Mapping[str, str] = field(compare=False, repr=False)

    def verify(self) -> None:
        if (
            not _is_sha256(self.manifest_hash)
            or len(self.item_ids) != EXPECTED_FRESH_ITEM_COUNT
            or len(set(self.item_ids)) != len(self.item_ids)
            or tuple(sorted(self.item_ids)) != self.item_ids
            or set(self.family_by_id) != set(self.item_ids)
            or self.family_by_id.get(ACTIVE_FRESH_ITEM_ID) != ACTIVE_FAMILY
        ):
            raise FinancialSemanticFreshRunnerError(
                "fresh split metadata drifted"
            )


def load_fresh_split_metadata_v1(path: Path) -> FreshSplitMetadataV1:
    payload = _read_json(path, "fresh provenance split")
    without_hash = dict(payload)
    declared_hash = without_hash.pop("manifest_hash", None)
    item_ids = payload.get("fresh_validation_ids")
    family_by_all_id = payload.get("family_by_id")
    if (
        declared_hash != stable_hash(without_hash)
        or payload.get("manifest_version") != FRESH_SPLIT_VERSION
        or payload.get("fresh_validation_content_accessed") is not False
        or payload.get("residual_sealed_content_accessed") is not False
        or payload.get("sealed_test") is not True
        or not isinstance(item_ids, list)
        or not isinstance(family_by_all_id, Mapping)
        or payload.get("counts", {}).get("fresh_validation")
        != EXPECTED_FRESH_ITEM_COUNT
    ):
        raise FinancialSemanticFreshRunnerError(
            "fresh provenance split is not frozen"
        )
    normalized_ids = tuple(sorted(str(item_id) for item_id in item_ids))
    metadata = FreshSplitMetadataV1(
        manifest_hash=str(declared_hash),
        item_ids=normalized_ids,
        family_by_id={
            item_id: str(family_by_all_id.get(item_id) or "")
            for item_id in normalized_ids
        },
    )
    metadata.verify()
    return metadata


@dataclass(frozen=True)
class FrozenFinancialTreatmentV1:
    manifest_hash: str
    recipe_id: str
    program_set_hash: str
    treatment_id: str
    candidate_id: str
    candidate_manifest_hash: str
    external_skill_source_receipt_hash: str
    candidate_skill_source: str
    fresh_item_id: str
    fresh_split_manifest_hash: str
    evaluator_epoch: str
    operator_asset_path: str
    minilm_runtime_asset_path: str
    qa_runtime_asset_path: str

    @property
    def treatment_hash(self) -> str:
        return self.treatment_id

    def verify(self, split: FreshSplitMetadataV1) -> None:
        hashes = (
            self.manifest_hash,
            self.recipe_id,
            self.program_set_hash,
            self.treatment_id,
            self.candidate_id,
            self.candidate_manifest_hash,
            self.external_skill_source_receipt_hash,
            self.fresh_split_manifest_hash,
        )
        if (
            not all(_is_sha256(value) for value in hashes)
            or self.program_set_hash
            != stable_hash({"recipe_ids": [self.recipe_id]})
            or self.fresh_item_id != ACTIVE_FRESH_ITEM_ID
            or self.fresh_item_id not in split.item_ids
            or self.fresh_split_manifest_hash != split.manifest_hash
            or self.evaluator_epoch
            != f"financial-semantic-fresh-{split.manifest_hash[:12]}"
            or not self.candidate_skill_source
            or not self.operator_asset_path
            or not self.minilm_runtime_asset_path
            or not self.qa_runtime_asset_path
        ):
            raise FinancialSemanticFreshRunnerError(
                "financial treatment identity drifted"
            )


def load_frozen_financial_treatment_v1(
    *,
    project_root: Path,
    benchmark_root: Path,
    path: Path,
    split: FreshSplitMetadataV1,
) -> FrozenFinancialTreatmentV1:
    try:
        payload = load_financial_semantic_treatment_freeze_v1(
            path,
            project_root=project_root,
            benchmark_root=benchmark_root,
        )
    except Exception as exc:
        raise FinancialSemanticFreshRunnerError(
            "financial treatment freeze failed live validation"
        ) from exc
    if payload.get("manifest_version") != (
        FINANCIAL_SEMANTIC_TREATMENT_FREEZE_VERSION
    ):
        raise FinancialSemanticFreshRunnerError(
            "financial treatment freeze version drifted"
        )
    split_source = payload["fresh_split_source"]
    operator_asset = payload["operator_asset"]
    runtime_assets = payload["runtime_assets"]
    treatment = FrozenFinancialTreatmentV1(
        manifest_hash=str(payload["manifest_hash"]),
        recipe_id=str(payload.get("recipe_id") or ""),
        program_set_hash=str(payload.get("program_set_hash") or ""),
        treatment_id=str(payload.get("treatment_id") or ""),
        candidate_id=str(payload.get("candidate_id") or ""),
        candidate_manifest_hash=str(
            payload.get("candidate_manifest_hash") or ""
        ),
        external_skill_source_receipt_hash=str(
            payload.get("external_skill_source_receipt_hash") or ""
        ),
        candidate_skill_source=str(
            payload.get("candidate_skill_source") or ""
        ),
        fresh_item_id=str(payload.get("fresh_item_id") or ""),
        fresh_split_manifest_hash=str(split_source["manifest_hash"]),
        evaluator_epoch=str(payload.get("evaluator_epoch") or ""),
        operator_asset_path=str(operator_asset["relative_path"]),
        minilm_runtime_asset_path=str(
            runtime_assets["minilm"]["relative_path"]
        ),
        qa_runtime_asset_path=str(
            runtime_assets["qa"]["relative_path"]
        ),
    )
    treatment.verify(split)
    return treatment


@dataclass(frozen=True)
class FreshPhysicalWorkUnitV1:
    arm: str
    request: SkillLearnTrialRequest
    skill_source_dir: Path | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    @property
    def item_id(self) -> str:
        return self.request.item_id

    @property
    def family(self) -> str:
        return self.request.family

    @property
    def work_unit_hash(self) -> str:
        return stable_hash(
            {
                "runner_version": FINANCIAL_SEMANTIC_FRESH_RUNNER_VERSION,
                "arm": self.arm,
                "request_hash": self.request.request_hash,
            }
        )

    def safe_payload(self) -> dict[str, Any]:
        return {
            "work_unit_hash": self.work_unit_hash,
            "arm": self.arm,
            "item_id_hash": stable_hash({"item_id": self.item_id}),
            "family_hash": stable_hash({"family": self.family}),
            "request_hash": self.request.request_hash,
            "candidate_source_required": self.skill_source_dir is not None,
            "raw_content_persisted": False,
        }


@dataclass(frozen=True)
class FreshExecutionPlanV1:
    split: FreshSplitMetadataV1
    treatment: FrozenFinancialTreatmentV1
    physical_work_units: tuple[FreshPhysicalWorkUnitV1, ...]
    raw_requests_by_item: Mapping[str, SkillLearnTrialRequest] = field(
        compare=False,
        repr=False,
    )
    candidate_requests_by_item: Mapping[str, SkillLearnTrialRequest] = field(
        compare=False,
        repr=False,
    )

    @property
    def plan_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        work_rows = [row.safe_payload() for row in self.physical_work_units]
        pair_rows = [
            {
                "item_id_hash": stable_hash({"item_id": item_id}),
                "raw_request_hash": self.raw_requests_by_item[
                    item_id
                ].request_hash,
                "candidate_request_hash": self.candidate_requests_by_item[
                    item_id
                ].request_hash,
                "candidate_execution": (
                    "physical"
                    if item_id == self.treatment.fresh_item_id
                    else "exact_raw_projection"
                ),
            }
            for item_id in self.split.item_ids
        ]
        return {
            "runner_version": FINANCIAL_SEMANTIC_FRESH_RUNNER_VERSION,
            "fresh_split_manifest_hash": self.split.manifest_hash,
            "treatment_manifest_hash": self.treatment.manifest_hash,
            "work_units": work_rows,
            "work_unit_set_hash": stable_hash({"rows": work_rows}),
            "pairs": pair_rows,
            "pair_set_hash": stable_hash({"rows": pair_rows}),
            "raw_execution_count": EXPECTED_RAW_EXECUTION_COUNT,
            "semantic_execution_count": EXPECTED_SEMANTIC_EXECUTION_COUNT,
            "inactive_projection_count": EXPECTED_INACTIVE_PROJECTION_COUNT,
            "official_hipporag_status": OFFICIAL_HIPPORAG_STATUS,
            "official_hipporag_execution_count": 0,
            "raw_content_persisted": False,
        }

    def verify(self) -> None:
        self.split.verify()
        self.treatment.verify(self.split)
        raw = [row for row in self.physical_work_units if row.arm == "raw"]
        semantic = [
            row for row in self.physical_work_units if row.arm == "semantic"
        ]
        if (
            len(self.physical_work_units)
            != EXPECTED_PHYSICAL_WORK_UNIT_COUNT
            or len({row.work_unit_hash for row in self.physical_work_units})
            != EXPECTED_PHYSICAL_WORK_UNIT_COUNT
            or len(raw) != EXPECTED_RAW_EXECUTION_COUNT
            or {row.item_id for row in raw} != set(self.split.item_ids)
            or len(semantic) != EXPECTED_SEMANTIC_EXECUTION_COUNT
            or semantic[0].item_id != self.treatment.fresh_item_id
            or set(self.raw_requests_by_item) != set(self.split.item_ids)
            or set(self.candidate_requests_by_item)
            != set(self.split.item_ids)
        ):
            raise FinancialSemanticFreshRunnerError(
                "fresh flat execution grid drifted"
            )
        for item_id in self.split.item_ids:
            raw_request = self.raw_requests_by_item[item_id]
            candidate_request = self.candidate_requests_by_item[item_id]
            if (
                raw_request.variant is not TrialVariant.POLICY_OFF
                or raw_request.treatment_hash != NO_SKILL_TREATMENT_HASH
                or raw_request.program_id is not None
                or raw_request.external_skill_source_receipt_hash
                or candidate_request.variant is not TrialVariant.POLICY_ON
                or candidate_request.program_id != self.treatment.recipe_id
                or candidate_request.program_set_hash
                != self.treatment.program_set_hash
                or candidate_request.treatment_hash
                != self.treatment.treatment_hash
                or candidate_request.external_skill_source_receipt_hash
                != self.treatment.external_skill_source_receipt_hash
                or raw_request.pair_id != candidate_request.pair_id
                or raw_request.item_id != candidate_request.item_id
                or raw_request.family != candidate_request.family
                or raw_request.split is not SplitName.VALIDATION
                or candidate_request.split is not SplitName.VALIDATION
            ):
                raise FinancialSemanticFreshRunnerError(
                    "fresh request identity drifted"
                )


def build_fresh_execution_plan_v1(
    *,
    split: FreshSplitMetadataV1,
    treatment: FrozenFinancialTreatmentV1,
    candidate_skill_source: Path,
    agent_id: str,
    model: str,
    max_steps: int,
    codex_agent_execution_policy_hash: str,
) -> FreshExecutionPlanV1:
    split.verify()
    treatment.verify(split)
    source_receipt = verify_skill_source_tree(candidate_skill_source)
    if source_receipt.receipt_hash != (
        treatment.external_skill_source_receipt_hash
    ):
        raise FinancialSemanticFreshRunnerError(
            "candidate source differs from frozen external source receipt"
        )
    raw_requests: dict[str, SkillLearnTrialRequest] = {}
    candidate_requests: dict[str, SkillLearnTrialRequest] = {}
    physical: list[FreshPhysicalWorkUnitV1] = []
    for item_id in split.item_ids:
        family = split.family_by_id[item_id]
        pair_id = "fresh-pair-" + stable_hash(
            {
                "split_manifest_hash": split.manifest_hash,
                "item_id": item_id,
                "repeat": 0,
            }
        )[:20]
        shared = {
            "item_id": item_id,
            "family": family,
            "split": SplitName.VALIDATION,
            "evaluator_epoch": treatment.evaluator_epoch,
            "pair_id": pair_id,
            "repeat": 0,
            "agent_id": agent_id,
            "model": model,
            "max_steps": max_steps,
            "manifest_hash": split.manifest_hash,
            "codex_agent_execution_policy_hash": (
                codex_agent_execution_policy_hash
            ),
        }
        raw_request = SkillLearnTrialRequest(
            **shared,
            variant=TrialVariant.POLICY_OFF,
            treatment_hash=NO_SKILL_TREATMENT_HASH,
        )
        candidate_request = SkillLearnTrialRequest(
            **shared,
            variant=TrialVariant.POLICY_ON,
            program_id=treatment.recipe_id,
            program_set_hash=treatment.program_set_hash,
            treatment_hash=treatment.treatment_hash,
            external_skill_source_receipt_hash=(
                treatment.external_skill_source_receipt_hash
            ),
        )
        raw_requests[item_id] = raw_request
        candidate_requests[item_id] = candidate_request
        physical.append(
            FreshPhysicalWorkUnitV1(
                arm="raw",
                request=raw_request,
            )
        )
        if item_id == treatment.fresh_item_id:
            physical.append(
                FreshPhysicalWorkUnitV1(
                    arm="semantic",
                    request=candidate_request,
                    skill_source_dir=candidate_skill_source,
                )
            )
    plan = FreshExecutionPlanV1(
        split=split,
        treatment=treatment,
        physical_work_units=tuple(
            sorted(
                physical,
                key=lambda row: (row.item_id, row.arm),
            )
        ),
        raw_requests_by_item=raw_requests,
        candidate_requests_by_item=candidate_requests,
    )
    plan.verify()
    return plan


class _RunnableBackend(Protocol):
    def run(
        self,
        request: SkillLearnTrialRequest,
        *,
        skill_source_dir: Path | None,
        trace_id: str,
    ) -> SkillLearnTrialObservation: ...


BackendFactory = Callable[[FreshPhysicalWorkUnitV1], _RunnableBackend]


@dataclass(frozen=True)
class FreshPhysicalResultV1:
    work: FreshPhysicalWorkUnitV1
    observation: SkillLearnTrialObservation = field(
        compare=False,
        repr=False,
    )
    backend_instance_token: str
    semantic_runtime_evidence: tuple[Mapping[str, Any], ...] = field(
        default=(),
        compare=False,
        repr=False,
    )

    def safe_payload(self) -> dict[str, Any]:
        return {
            **self.work.safe_payload(),
            "observation": self.observation.to_dict(),
            "observation_hash": self.observation.observation_hash,
            "backend_instance_token": self.backend_instance_token,
            "semantic_runtime_evidence": [
                dict(row) for row in self.semantic_runtime_evidence
            ],
            "semantic_runtime_evidence_set_hash": stable_hash(
                {
                    "rows": [
                        dict(row) for row in self.semantic_runtime_evidence
                    ]
                }
            ),
        }


@dataclass(frozen=True)
class FreshInactiveProjectionV1:
    item_id_hash: str
    raw_observation_hash: str
    candidate_request_hash: str
    projected_observation_hash: str
    raw_success: bool
    projected_success: bool
    raw_error_type: str | None
    projected_error_type: str | None

    def safe_payload(self) -> dict[str, Any]:
        return {
            "projection_policy": INACTIVE_PROJECTION_POLICY,
            "item_id_hash": self.item_id_hash,
            "raw_observation_hash": self.raw_observation_hash,
            "candidate_request_hash": self.candidate_request_hash,
            "projected_observation_hash": self.projected_observation_hash,
            "raw_success": self.raw_success,
            "projected_success": self.projected_success,
            "raw_error_type": self.raw_error_type,
            "projected_error_type": self.projected_error_type,
            "behavior_identical_by_predeclared_inactive_route": True,
            "model_calls": 0,
            "online_judge_calls": 0,
            "raw_content_persisted": False,
        }


@dataclass(frozen=True)
class FreshBatchExecutionV1:
    plan: FreshExecutionPlanV1
    physical_results: tuple[FreshPhysicalResultV1, ...]
    inactive_projections: tuple[FreshInactiveProjectionV1, ...]
    maximum_concurrent_calls: int

    @property
    def result_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def safe_payload(self) -> dict[str, Any]:
        physical_rows = [row.safe_payload() for row in self.physical_results]
        projection_rows = [
            row.safe_payload() for row in self.inactive_projections
        ]
        return {
            "plan_hash": self.plan.plan_hash,
            "physical_results": physical_rows,
            "physical_result_set_hash": stable_hash(
                {"rows": physical_rows}
            ),
            "inactive_projections": projection_rows,
            "inactive_projection_set_hash": stable_hash(
                {"rows": projection_rows}
            ),
            "maximum_concurrent_calls": self.maximum_concurrent_calls,
        }

    def verify(self) -> None:
        self.plan.verify()
        if (
            len(self.physical_results)
            != EXPECTED_PHYSICAL_WORK_UNIT_COUNT
            or len(
                {
                    row.work.work_unit_hash
                    for row in self.physical_results
                }
            )
            != EXPECTED_PHYSICAL_WORK_UNIT_COUNT
            or len(
                {
                    row.backend_instance_token
                    for row in self.physical_results
                }
            )
            != EXPECTED_PHYSICAL_WORK_UNIT_COUNT
            or len(self.inactive_projections)
            != EXPECTED_INACTIVE_PROJECTION_COUNT
            or self.maximum_concurrent_calls <= 0
            or self.maximum_concurrent_calls
            > EXPECTED_PHYSICAL_WORK_UNIT_COUNT
        ):
            raise FinancialSemanticFreshRunnerError(
                "fresh batch execution evidence drifted"
            )
        results_by_work = {
            row.work.work_unit_hash: row for row in self.physical_results
        }
        if set(results_by_work) != {
            row.work_unit_hash for row in self.plan.physical_work_units
        }:
            raise FinancialSemanticFreshRunnerError(
                "fresh physical result coverage drifted"
            )
        semantic_rows = [
            row for row in self.physical_results if row.work.arm == "semantic"
        ]
        if (
            len(semantic_rows) != 1
            or len(semantic_rows[0].semantic_runtime_evidence) > 1
            or any(
                row.semantic_runtime_evidence
                for row in self.physical_results
                if row.work.arm == "raw"
            )
        ):
            raise FinancialSemanticFreshRunnerError(
                "financial runtime evidence is not isolated"
            )


def execute_fresh_plan_v1(
    *,
    plan: FreshExecutionPlanV1,
    backend_factory: BackendFactory,
    max_workers: int = DEFAULT_OUTER_WORKERS,
) -> FreshBatchExecutionV1:
    """Submit every physical work unit before waiting for any one result."""

    plan.verify()
    if max_workers < len(plan.physical_work_units):
        raise FinancialSemanticFreshRunnerError(
            "fresh runner must expose one worker per physical work unit"
        )
    lock = threading.Lock()
    active = 0
    maximum_active = 0
    retained_backends: list[_RunnableBackend] = []

    backend_rows: list[
        tuple[FreshPhysicalWorkUnitV1, _RunnableBackend, str]
    ] = []
    for work in plan.physical_work_units:
        backend = backend_factory(work)
        if any(existing is backend for existing in retained_backends):
            raise FinancialSemanticFreshRunnerError(
                "backend factory reused an instance across physical futures"
            )
        retained_backends.append(backend)
        backend_rows.append(
            (
                work,
                backend,
                stable_hash(
                    {
                        "work_unit_hash": work.work_unit_hash,
                        "backend_object_identity": id(backend),
                    }
                ),
            )
        )

    def run_one(
        work: FreshPhysicalWorkUnitV1,
        backend: _RunnableBackend,
        token: str,
    ) -> FreshPhysicalResultV1:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            observation = backend.run(
                work.request,
                skill_source_dir=work.skill_source_dir,
                trace_id=f"fresh-financial-v1:{work.work_unit_hash[:20]}",
            )
            evidence = tuple(
                getattr(backend, "financial_runtime_evidence", ())
            )
            return FreshPhysicalResultV1(
                work=work,
                observation=observation,
                backend_instance_token=token,
                semantic_runtime_evidence=evidence,
            )
        finally:
            with lock:
                active -= 1

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(plan.physical_work_units)
    ) as executor:
        futures = {
            executor.submit(run_one, work, backend, token): work
            for work, backend, token in backend_rows
        }
        results = tuple(future.result() for future in futures)
    ordered_results = tuple(
        sorted(results, key=lambda row: row.work.work_unit_hash)
    )
    raw_by_item = {
        row.work.item_id: row.observation
        for row in ordered_results
        if row.work.arm == "raw"
    }
    projections: list[FreshInactiveProjectionV1] = []
    for item_id in plan.split.item_ids:
        if item_id == plan.treatment.fresh_item_id:
            continue
        raw = raw_by_item[item_id]
        candidate_request = plan.candidate_requests_by_item[item_id]
        projected = raw.as_variant(candidate_request)
        projections.append(
            FreshInactiveProjectionV1(
                item_id_hash=stable_hash({"item_id": item_id}),
                raw_observation_hash=raw.observation_hash,
                candidate_request_hash=candidate_request.request_hash,
                projected_observation_hash=projected.observation_hash,
                raw_success=raw.success,
                projected_success=projected.success,
                raw_error_type=raw.error_type,
                projected_error_type=projected.error_type,
            )
        )
    result = FreshBatchExecutionV1(
        plan=plan,
        physical_results=ordered_results,
        inactive_projections=tuple(
            sorted(projections, key=lambda row: row.item_id_hash)
        ),
        maximum_concurrent_calls=maximum_active,
    )
    result.verify()
    return result


def _configure_environment(protocol: PaperProtocol) -> None:
    payload = protocol.payload
    execution = payload.get("execution")
    if not isinstance(execution, Mapping):
        raise FinancialSemanticFreshRunnerError(
            "paper protocol execution contract is missing"
        )
    expected_origin = str(payload.get("provider_endpoint_origin") or "")
    configured_origin = configured_api_origin()
    if configured_origin and configured_origin != expected_origin:
        raise FinancialSemanticFreshRunnerError(
            "configured provider origin differs from frozen protocol"
        )
    if not os.environ.get("ASSUMPTION_V2_API_KEY", "").strip():
        raise FinancialSemanticFreshRunnerError(
            "selected provider key is absent from the process environment"
        )
    os.environ["ASSUMPTION_V2_API_BASE"] = expected_origin
    os.environ["ASSUMPTION_V2_MODEL"] = str(payload["model"])
    os.environ["ASSUMPTION_V2_SKILLLEARN_PROVIDER_MODE"] = str(
        payload["trial_provider_mode"]
    )
    os.environ["ASSUMPTION_V2_API_ALLOWED_IPV4S"] = ",".join(
        str(value) for value in payload["provider_endpoint_ipv4s"]
    )
    os.environ["ASSUMPTION_V2_SKILLLEARN_CACHE_ONLY"] = "1"
    os.environ["ASSUMPTION_V2_TRIAL_NETWORK_BYTE_LIMIT"] = str(
        execution["trial_network_byte_limit"]
    )
    if (
        configured_model() != V320_MODEL
        or configured_skilllearn_provider_mode()
        != payload["trial_provider_mode"]
        or configured_api_origin() != expected_origin
        or configured_trial_network_byte_limit()
        != execution["trial_network_byte_limit"]
    ):
        raise FinancialSemanticFreshRunnerError(
            "selected provider batch configuration drifted"
        )
    egress = DockerEgressPolicy.from_env()
    if (
        egress.endpoint_origin != expected_origin
        or tuple(egress.allowed_ipv4s)
        != tuple(payload["provider_endpoint_ipv4s"])
    ):
        raise FinancialSemanticFreshRunnerError(
            "selected provider egress authority drifted"
        )


def _provider_selection_receipt(
    *,
    provider_label: str,
    provider_selection_receipt_path: Path,
    selected_canary_report_path: Path,
    plus_canary_report_path: Path | None,
    plus_transport_failure_receipt_path: Path | None,
    plus_failure_event_ledger_path: Path | None,
    plus_expected_canary_report_path: Path | None,
) -> dict[str, Any]:
    if provider_label not in {"plus", "pro"}:
        raise FinancialSemanticFreshRunnerError(
            "selected provider label must be plus or pro"
        )
    try:
        return _verify_provider_selection_receipt_v3(
            selection_receipt_path=provider_selection_receipt_path,
            selected_canary_report_path=selected_canary_report_path,
            provider_label=provider_label,
            plus_canary_report_path=plus_canary_report_path,
            plus_transport_failure_receipt_path=(
                plus_transport_failure_receipt_path
            ),
            plus_failure_event_ledger_path=plus_failure_event_ledger_path,
            plus_expected_canary_report_path=(
                plus_expected_canary_report_path
            ),
        )
    except Exception as exc:
        raise FinancialSemanticFreshRunnerError(
            "Plus-to-Pro pre-task provider selection is invalid"
        ) from exc


@dataclass(frozen=True)
class _RuntimeAssetsV1:
    prebuilt_cache: FrozenTaskInputPrebuiltImageCache = field(
        compare=False,
        repr=False,
    )
    offline_cache: SkillLearnOfflineVerifierRuntimeCache = field(
        compare=False,
        repr=False,
    )
    provider_circuit: SkillLearnProviderCircuit = field(
        compare=False,
        repr=False,
    )
    model_limiter: SkillLearnModelInferenceLimiter = field(
        compare=False,
        repr=False,
    )
    preflight_report: Mapping[str, Any]


def _prepare_runtime_assets_v1(
    *,
    project: Path,
    destination: Path,
    benchmark_root: Path,
    protocol: PaperProtocol,
    split: FreshSplitMetadataV1,
    event_sink: JsonlEventSink,
    task_input_cache_root: Path | None,
) -> _RuntimeAssetsV1:
    primary_manifest = SplitManifest.read(
        project / V320_MANIFEST_RELATIVE_PATH
    )
    frozen = load_frozen_task_input_closure(
        protocol.payload,
        project_root=project,
    )
    if frozen is None:
        raise FinancialSemanticFreshRunnerError(
            "frozen task input closure is unavailable"
        )
    prewarm_path = project / V320_PREWARM_RELATIVE_PATH
    prewarm = _read_json(prewarm_path, "v3.20 prewarm receipt")
    execution = protocol.payload["execution"]
    assert isinstance(execution, Mapping)
    validate_development_prewarm_receipt(
        prewarm,
        manifest=primary_manifest,
        expected_version=str(execution["development_prewarm"]),
        frozen_task_inputs=frozen,
    )
    prebuilt_cache = FrozenTaskInputPrebuiltImageCache(
        benchmark_root,
        frozen_task_inputs=frozen,
        expected_prewarm_rows=expected_prewarm_closure_rows(prewarm),
        cache_only=True,
        event_sink=event_sink,
        task_input_cache_root=task_input_cache_root,
    )
    offline_cache = SkillLearnOfflineVerifierRuntimeCache(
        event_sink=event_sink
    )
    circuit = SkillLearnProviderCircuit()
    limiter = SkillLearnModelInferenceLimiter(
        DEFAULT_MODEL_INFERENCE_SLOTS
    )

    def preflight(item_id: str) -> dict[str, Any]:
        family = split.family_by_id[item_id]
        if offline_verifier_profile_for_family(family) is None:
            raise FinancialSemanticFreshRunnerError(
                "fresh family has no offline verifier profile"
            )
        backend = SkillLearnSubprocessBackend(
            benchmark_root,
            agent_id=str(protocol.payload["agent_id"]),
            model=str(protocol.payload["model"]),
            max_steps=int(protocol.payload["max_steps"]),
            provider_mode=str(protocol.payload["trial_provider_mode"]),
            record_upstream=False,
            prebuilt_cache=prebuilt_cache,
            offline_verifier_cache=offline_cache,
            provider_circuit=circuit,
            model_inference_limiter=limiter,
            codex_agent_execution_policy=(
                protocol.codex_agent_execution_policy
            ),
            event_sink=event_sink,
        )
        image, verifier = backend.prewarm_trial_environment(
            family=family,
            item_id=item_id,
            trace_id=(
                "financial-fresh-asset-preflight:"
                + stable_hash({"item_id": item_id})[:20]
            ),
        )
        if (
            not prebuilt_cache.cache_only
            or verifier is None
            or verifier.profile.profile_id
            != offline_verifier_profile_for_family(family).profile_id
        ):
            raise FinancialSemanticFreshRunnerError(
                "fresh offline asset preflight drifted"
            )
        return {
            "item_id_hash": stable_hash({"item_id": item_id}),
            "family_hash": stable_hash({"family": family}),
            "prebuilt_image_key_hash": stable_hash(
                {"prebuilt_image_key": image.cache_key}
            ),
            "prebuilt_image_id_hash": stable_hash(
                {"prebuilt_image_id": image.image_id}
            ),
            "offline_verifier_profile_id_hash": stable_hash(
                {"profile_id": verifier.profile.profile_id}
            ),
            "offline_verifier_runtime_key_hash": stable_hash(
                {"runtime_key": verifier.runtime_key}
            ),
            "cache_only": True,
            "verifier_runtime_network": "none",
            "model_calls": 0,
            "online_judge_calls": 0,
            "raw_content_persisted": False,
        }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=EXPECTED_FRESH_ITEM_COUNT
    ) as executor:
        rows = tuple(executor.map(preflight, split.item_ids))
    ordered = tuple(sorted(rows, key=lambda row: row["item_id_hash"]))
    without_hash = {
        "preflight_policy": (
            "fresh_all_items_cache_only_offline_verifier_preflight_v1"
        ),
        "passed": True,
        "fresh_split_manifest_hash": split.manifest_hash,
        "item_count": len(ordered),
        "items": list(ordered),
        "item_set_hash": stable_hash({"rows": list(ordered)}),
        "cache_only": True,
        "offline_evaluation_only": True,
        "model_calls": 0,
        "online_judge_calls": 0,
        "raw_content_persisted": False,
    }
    report = {**without_hash, "report_hash": stable_hash(without_hash)}
    (destination / ASSET_PREFLIGHT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _RuntimeAssetsV1(
        prebuilt_cache=prebuilt_cache,
        offline_cache=offline_cache,
        provider_circuit=circuit,
        model_limiter=limiter,
        preflight_report=report,
    )


def run_financial_semantic_fresh_v1(
    *,
    project_root: Path,
    output_root: Path,
    treatment_manifest_path: Path,
    minilm_snapshot_root: Path,
    qa_snapshot_root: Path,
    provider_label: str,
    provider_selection_receipt_path: Path,
    selected_canary_report_path: Path,
    plus_canary_report_path: Path | None = None,
    plus_transport_failure_receipt_path: Path | None = None,
    plus_failure_event_ledger_path: Path | None = None,
    plus_expected_canary_report_path: Path | None = None,
    task_input_cache_root: Path | None = None,
) -> Mapping[str, Any]:
    """Run one fixed-provider, maximally parallel 10-call fresh batch."""

    project = project_root.expanduser().resolve(strict=True)
    destination = output_root.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("fresh paired output root already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    provider_verified = False
    fresh_assets_opened = False
    try:
        protocol = PaperProtocol.read(project / V320_PROTOCOL_RELATIVE_PATH)
        if (
            protocol.payload.get("protocol_version") != "3.20.0"
            or protocol.payload.get("model") != V320_MODEL
            or protocol.payload.get("max_steps") != 100
        ):
            raise FinancialSemanticFreshRunnerError(
                "frozen execution protocol drifted"
            )
        provider_receipt = _provider_selection_receipt(
            provider_label=provider_label,
            provider_selection_receipt_path=(
                provider_selection_receipt_path.resolve(strict=True)
            ),
            selected_canary_report_path=(
                selected_canary_report_path.resolve(strict=True)
            ),
            plus_canary_report_path=(
                plus_canary_report_path.resolve(strict=True)
                if plus_canary_report_path
                else None
            ),
            plus_transport_failure_receipt_path=(
                plus_transport_failure_receipt_path.resolve(strict=True)
                if plus_transport_failure_receipt_path
                else None
            ),
            plus_failure_event_ledger_path=(
                plus_failure_event_ledger_path.resolve(strict=True)
                if plus_failure_event_ledger_path
                else None
            ),
            plus_expected_canary_report_path=(
                plus_expected_canary_report_path.resolve(strict=True)
                if plus_expected_canary_report_path
                else None
            ),
        )
        provider_verified = True
        _configure_environment(protocol)
        benchmark_root = (
            project / SKILLLEARN_BENCHMARK_RELATIVE_ROOT
        ).resolve(strict=True)
        split_path = project / FRESH_SPLIT_RELATIVE_PATH
        split = load_fresh_split_metadata_v1(split_path)
        treatment = load_frozen_financial_treatment_v1(
            project_root=project,
            benchmark_root=benchmark_root,
            path=treatment_manifest_path.resolve(strict=True),
            split=split,
        )
        candidate_source = _resolve_project_path(
            project,
            treatment.candidate_skill_source,
            directory=True,
        )
        plan = build_fresh_execution_plan_v1(
            split=split,
            treatment=treatment,
            candidate_skill_source=candidate_source,
            agent_id=str(protocol.payload["agent_id"]),
            model=str(protocol.payload["model"]),
            max_steps=int(protocol.payload["max_steps"]),
            codex_agent_execution_policy_hash=(
                protocol.codex_agent_execution_policy.policy_hash
            ),
        )
        event_sink = JsonlEventSink(
            destination / EXECUTION_EVENTS_FILENAME
        )
        assets = _prepare_runtime_assets_v1(
            project=project,
            destination=destination,
            benchmark_root=benchmark_root,
            protocol=protocol,
            split=split,
            event_sink=event_sink,
            task_input_cache_root=task_input_cache_root,
        )
        fresh_assets_opened = True
        operator_asset = _resolve_project_path(
            project,
            treatment.operator_asset_path,
            directory=False,
        )
        minilm_runtime_asset = _resolve_project_path(
            project,
            treatment.minilm_runtime_asset_path,
            directory=False,
        )
        qa_runtime_asset = _resolve_project_path(
            project,
            treatment.qa_runtime_asset_path,
            directory=False,
        )
        planner = SharedFinancialSemanticPlannerV1(
            asset_path=operator_asset,
            minilm_runtime_asset_path=minilm_runtime_asset,
            minilm_snapshot_root=minilm_snapshot_root.resolve(strict=True),
            qa_runtime_asset_path=qa_runtime_asset,
            qa_snapshot_root=qa_snapshot_root.resolve(strict=True),
        )
        if (
            planner.asset.get("candidate_id") != treatment.candidate_id
            or planner.asset.get("manifest_hash")
            != treatment.candidate_manifest_hash
        ):
            raise FinancialSemanticFreshRunnerError(
                "planner asset differs from frozen treatment"
            )
        trials_root = destination / "worker_state"

        def backend_factory(
            work: FreshPhysicalWorkUnitV1,
        ) -> _RunnableBackend:
            common: dict[str, Any] = {
                "agent_id": work.request.agent_id,
                "model": work.request.model,
                "max_steps": work.request.max_steps,
                "provider_mode": str(
                    protocol.payload["trial_provider_mode"]
                ),
                "trials_dir": trials_root / work.work_unit_hash,
                "record_upstream": True,
                "prebuilt_cache": assets.prebuilt_cache,
                "offline_verifier_cache": assets.offline_cache,
                "provider_circuit": assets.provider_circuit,
                "model_inference_limiter": assets.model_limiter,
                "codex_agent_execution_policy": (
                    protocol.codex_agent_execution_policy
                ),
                "event_sink": event_sink,
            }
            if work.arm == "semantic":
                return FinancialSemanticSubprocessBackendV1(
                    benchmark_root,
                    planner=planner,
                    expected_program_id=treatment.recipe_id,
                    expected_treatment_hash=treatment.treatment_hash,
                    expected_external_skill_source_receipt_hash=(
                        treatment.external_skill_source_receipt_hash
                    ),
                    **common,
                )
            if work.arm != "raw":
                raise FinancialSemanticFreshRunnerError(
                    "unregistered fresh physical arm"
                )
            return SkillLearnSubprocessBackend(benchmark_root, **common)

        batch = execute_fresh_plan_v1(
            plan=plan,
            backend_factory=backend_factory,
            max_workers=DEFAULT_OUTER_WORKERS,
        )
        artifact_closure = _worker_artifact_closure(trials_root)
        physical = list(batch.physical_results)
        raw_by_item = {
            row.work.item_id: row.observation
            for row in physical
            if row.work.arm == "raw"
        }
        semantic_row = next(
            row for row in physical if row.work.arm == "semantic"
        )
        active_raw = raw_by_item[treatment.fresh_item_id]
        invalid_count = sum(not row.observation.valid for row in physical)
        runtime_evidence = [
            dict(value)
            for value in semantic_row.semantic_runtime_evidence
        ]
        without_hash: dict[str, Any] = {
            "runner_version": FINANCIAL_SEMANTIC_FRESH_RUNNER_VERSION,
            "execution_completed": True,
            "evidence_valid": invalid_count == 0
            and len(runtime_evidence) == 1
            and all(
                row.observation.raw_trial_artifacts_persisted
                for row in physical
            ),
            "provider_policy": TYPED_ASSIGNMENT_PROVIDER_POLICY,
            "provider_selection": provider_receipt,
            "selected_provider_label_hash": stable_hash(
                {"provider_label": provider_label}
            ),
            "selected_provider_fixed_for_complete_batch": True,
            "mid_batch_provider_switch_authorized": False,
            "plan": plan.safe_payload(),
            "plan_hash": plan.plan_hash,
            "treatment_manifest_hash": treatment.manifest_hash,
            "fresh_split_manifest_hash": split.manifest_hash,
            "asset_preflight_report_hash": assets.preflight_report[
                "report_hash"
            ],
            "batch": batch.safe_payload(),
            "batch_result_hash": batch.result_hash,
            "worker_artifact_closure": artifact_closure,
            "raw_execution_count": EXPECTED_RAW_EXECUTION_COUNT,
            "semantic_execution_count": EXPECTED_SEMANTIC_EXECUTION_COUNT,
            "inactive_projection_count": (
                EXPECTED_INACTIVE_PROJECTION_COUNT
            ),
            "physical_work_unit_count": (
                EXPECTED_PHYSICAL_WORK_UNIT_COUNT
            ),
            "physical_backend_instance_count": len(
                {
                    row.backend_instance_token for row in physical
                }
            ),
            "outer_worker_limit": DEFAULT_OUTER_WORKERS,
            "maximum_concurrent_runner_calls": (
                batch.maximum_concurrent_calls
            ),
            "model_inference_slot_limit": DEFAULT_MODEL_INFERENCE_SLOTS,
            "maximum_concurrent_model_calls": (
                assets.model_limiter.maximum_active
            ),
            "invalid_physical_result_count": invalid_count,
            "active_pair": {
                "item_id_hash": stable_hash(
                    {"item_id": treatment.fresh_item_id}
                ),
                "raw_observation_hash": active_raw.observation_hash,
                "candidate_observation_hash": (
                    semantic_row.observation.observation_hash
                ),
                "raw_success": active_raw.success,
                "candidate_success": semantic_row.observation.success,
                "raw_error_type": active_raw.error_type,
                "candidate_error_type": (
                    semantic_row.observation.error_type
                ),
            },
            "financial_runtime_evidence": runtime_evidence,
            "financial_runtime_evidence_set_hash": stable_hash(
                {"rows": runtime_evidence}
            ),
            "official_hipporag": False,
            "hipporag_status": OFFICIAL_HIPPORAG_STATUS,
            "official_hipporag_execution_count": 0,
            "hipporag_proxy_substitution_used": False,
            "offline_evaluation_only": True,
            "online_judge_calls": 0,
            "network_fallback_used": False,
            "cache_only_prebuilt_images": True,
            "promotion_gate_applied": False,
            "promotion_authorized": False,
            "sealed_test_accessed": False,
            "raw_trial_artifacts_persisted": True,
            "secret_value_persisted": False,
        }
        report = {**without_hash, "report_hash": stable_hash(without_hash)}
        (destination / REPORT_FILENAME).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    except Exception as exc:
        failure_without_hash = {
            "runner_version": FINANCIAL_SEMANTIC_FRESH_RUNNER_VERSION,
            "execution_completed": False,
            "provider_selection_verified": provider_verified,
            "fresh_assets_opened": fresh_assets_opened,
            "error_type": type(exc).__name__,
            "error_message_hash": stable_hash({"message": str(exc)}),
            "raw_error_persisted": False,
            "secret_value_persisted": False,
        }
        failure = {
            **failure_without_hash,
            "report_hash": stable_hash(failure_without_hash),
        }
        (destination / FAILURE_FILENAME).write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen financial semantic candidate against the fresh "
            "validation cohort in one maximally parallel offline-verifier batch."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--treatment-manifest", type=Path, required=True)
    parser.add_argument("--minilm-snapshot-root", type=Path, required=True)
    parser.add_argument("--qa-snapshot-root", type=Path, required=True)
    parser.add_argument(
        "--provider-label",
        choices=("plus", "pro"),
        required=True,
    )
    parser.add_argument(
        "--provider-selection-receipt",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--selected-canary-report",
        type=Path,
        required=True,
    )
    parser.add_argument("--plus-canary-report", type=Path)
    parser.add_argument(
        "--plus-transport-failure-receipt",
        type=Path,
    )
    parser.add_argument("--plus-failure-event-ledger", type=Path)
    parser.add_argument("--plus-expected-canary-report", type=Path)
    parser.add_argument("--task-input-cache-root", type=Path)
    args = parser.parse_args()
    report = run_financial_semantic_fresh_v1(
        project_root=args.project_root,
        output_root=args.output_root,
        treatment_manifest_path=args.treatment_manifest,
        minilm_snapshot_root=args.minilm_snapshot_root,
        qa_snapshot_root=args.qa_snapshot_root,
        provider_label=args.provider_label,
        provider_selection_receipt_path=(
            args.provider_selection_receipt
        ),
        selected_canary_report_path=args.selected_canary_report,
        plus_canary_report_path=args.plus_canary_report,
        plus_transport_failure_receipt_path=(
            args.plus_transport_failure_receipt
        ),
        plus_failure_event_ledger_path=args.plus_failure_event_ledger,
        plus_expected_canary_report_path=(
            args.plus_expected_canary_report
        ),
        task_input_cache_root=args.task_input_cache_root,
    )
    print(
        json.dumps(
            {
                "execution_completed": report["execution_completed"],
                "evidence_valid": report["evidence_valid"],
                "active_pair": report["active_pair"],
                "maximum_concurrent_runner_calls": report[
                    "maximum_concurrent_runner_calls"
                ],
                "maximum_concurrent_model_calls": report[
                    "maximum_concurrent_model_calls"
                ],
                "hipporag_status": report["hipporag_status"],
                "online_judge_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
