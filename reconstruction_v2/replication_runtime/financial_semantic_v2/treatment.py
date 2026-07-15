from __future__ import annotations

"""Bindings for reusing the frozen financial candidate in a new evaluator.

The period-out extension changes the task pack and the evaluator wrapper, not
the candidate.  Candidate-facing requests therefore keep the exact opaque
recipe, source-tree, and asset identities.  A new treatment ID binds the new
pack/evaluator while retaining the parent treatment ID as provenance.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from assumption_agent.benchmarks.financial_semantic_fresh_runner_v1 import (
    FRESH_SPLIT_RELATIVE_PATH,
    FrozenFinancialTreatmentV1,
    load_fresh_split_metadata_v1,
    load_frozen_financial_treatment_v1,
)
from assumption_agent.benchmarks.skilllearn_compiler import (
    verify_skill_source_tree,
)
from assumption_agent.benchmarks.train_execution_contract_development_v2 import (
    SKILLLEARN_BENCHMARK_RELATIVE_ROOT,
)
from assumption_agent.models import stable_hash


PARENT_TREATMENT_RELATIVE_PATH = (
    "manifests/financial_semantic_treatment_freeze_v1.json"
)
REPLICATION_EVALUATOR_BINDING_VERSION = (
    "financial_semantic_sec13f_period_out_evaluator_binding_v1"
)


class FinancialSemanticReplicationTreatmentError(PermissionError):
    """A period-out run no longer matches the fixed parent candidate."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_candidate_identity_shape(
    candidate: "FixedFinancialCandidateIdentityV1",
) -> None:
    hashes = (
        candidate.parent_manifest_hash,
        candidate.candidate_id,
        candidate.candidate_manifest_hash,
        candidate.recipe_id,
        candidate.program_set_hash,
        candidate.parent_treatment_id,
        candidate.external_skill_source_receipt_hash,
    )
    if (
        not all(_is_sha256(value) for value in hashes)
        or candidate.program_set_hash
        != stable_hash({"recipe_ids": [candidate.recipe_id]})
    ):
        raise FinancialSemanticReplicationTreatmentError(
            "fixed candidate identity is malformed"
        )


@dataclass(frozen=True)
class FixedFinancialCandidateIdentityV1:
    parent_manifest_hash: str
    candidate_id: str
    candidate_manifest_hash: str
    recipe_id: str
    program_set_hash: str
    parent_treatment_id: str
    external_skill_source_receipt_hash: str
    candidate_skill_source: Path
    operator_asset_path: Path
    minilm_runtime_asset_path: Path
    qa_runtime_asset_path: Path

    def safe_payload(self, *, project_root: Path) -> dict[str, Any]:
        project = project_root.resolve(strict=True)

        def relative(path: Path) -> str:
            return path.resolve(strict=True).relative_to(project).as_posix()

        return {
            "parent_manifest_hash": self.parent_manifest_hash,
            "candidate_id": self.candidate_id,
            "candidate_manifest_hash": self.candidate_manifest_hash,
            "recipe_id": self.recipe_id,
            "program_set_hash": self.program_set_hash,
            "parent_treatment_id": self.parent_treatment_id,
            "external_skill_source_receipt_hash": (
                self.external_skill_source_receipt_hash
            ),
            "candidate_skill_source": relative(self.candidate_skill_source),
            "operator_asset_path": relative(self.operator_asset_path),
            "minilm_runtime_asset_path": relative(
                self.minilm_runtime_asset_path
            ),
            "qa_runtime_asset_path": relative(self.qa_runtime_asset_path),
            "candidate_recipe_and_source_identity_reused_exactly": True,
        }


def load_fixed_financial_candidate_identity_v1(
    project_root: str | Path,
) -> FixedFinancialCandidateIdentityV1:
    project = Path(project_root).expanduser().resolve(strict=True)
    benchmark = (project / SKILLLEARN_BENCHMARK_RELATIVE_ROOT).resolve(
        strict=True
    )
    split = load_fresh_split_metadata_v1(
        (project / FRESH_SPLIT_RELATIVE_PATH).resolve(strict=True)
    )
    treatment: FrozenFinancialTreatmentV1 = load_frozen_financial_treatment_v1(
        project_root=project,
        benchmark_root=benchmark,
        path=(project / PARENT_TREATMENT_RELATIVE_PATH).resolve(strict=True),
        split=split,
    )

    def project_path(relative: str) -> Path:
        raw = Path(relative)
        if raw.is_absolute() or ".." in raw.parts:
            raise FinancialSemanticReplicationTreatmentError(
                "parent treatment path is unsafe"
            )
        resolved = (project / raw).resolve(strict=True)
        try:
            resolved.relative_to(project)
        except ValueError as exc:
            raise FinancialSemanticReplicationTreatmentError(
                "parent treatment path escaped the project"
            ) from exc
        return resolved

    source = project_path(treatment.candidate_skill_source)
    source_receipt = verify_skill_source_tree(source)
    if source_receipt.receipt_hash != (
        treatment.external_skill_source_receipt_hash
    ):
        raise FinancialSemanticReplicationTreatmentError(
            "candidate skill source no longer matches its frozen receipt"
        )
    identity = FixedFinancialCandidateIdentityV1(
        parent_manifest_hash=treatment.manifest_hash,
        candidate_id=treatment.candidate_id,
        candidate_manifest_hash=treatment.candidate_manifest_hash,
        recipe_id=treatment.recipe_id,
        program_set_hash=treatment.program_set_hash,
        parent_treatment_id=treatment.treatment_id,
        external_skill_source_receipt_hash=(
            treatment.external_skill_source_receipt_hash
        ),
        candidate_skill_source=source,
        operator_asset_path=project_path(treatment.operator_asset_path),
        minilm_runtime_asset_path=project_path(
            treatment.minilm_runtime_asset_path
        ),
        qa_runtime_asset_path=project_path(treatment.qa_runtime_asset_path),
    )
    _validate_candidate_identity_shape(identity)
    return identity


def build_replication_evaluator_binding_v1(
    *,
    candidate: FixedFinancialCandidateIdentityV1,
    preregistration_hash: str,
    runtime_source_closure_hash: str,
    pack_commitment_hash: str | None = None,
) -> dict[str, Any]:
    """Bind a new evaluator without minting a new candidate recipe.

    ``pack_commitment_hash`` is intentionally optional at preregistration time;
    it becomes mandatory before model execution, after deterministic pack
    materialization from the predeclared SEC archives.
    """

    _validate_candidate_identity_shape(candidate)
    for label, value in (
        ("preregistration_hash", preregistration_hash),
        ("runtime_source_closure_hash", runtime_source_closure_hash),
    ):
        if not _is_sha256(value):
            raise FinancialSemanticReplicationTreatmentError(
                f"{label} must be a sha256 identity"
            )
    if pack_commitment_hash is not None and not _is_sha256(
        pack_commitment_hash
    ):
        raise FinancialSemanticReplicationTreatmentError(
            "pack commitment must be a sha256 identity"
        )
    body: dict[str, Any] = {
        "binding_version": REPLICATION_EVALUATOR_BINDING_VERSION,
        "parent_manifest_hash": candidate.parent_manifest_hash,
        "candidate_id": candidate.candidate_id,
        "candidate_manifest_hash": candidate.candidate_manifest_hash,
        "recipe_id": candidate.recipe_id,
        "program_set_hash": candidate.program_set_hash,
        "parent_treatment_id": candidate.parent_treatment_id,
        "external_skill_source_receipt_hash": (
            candidate.external_skill_source_receipt_hash
        ),
        "preregistration_hash": preregistration_hash,
        "runtime_source_closure_hash": runtime_source_closure_hash,
        "pack_commitment_hash": pack_commitment_hash,
        "candidate_recipe_and_source_identity_reused_exactly": True,
        "evaluator_wrapper_is_not_candidate_content": True,
        "performance_gate_bound": False,
        "promotion_authorized": False,
    }
    period_out_treatment_id = stable_hash(
        {
            "binding_version": REPLICATION_EVALUATOR_BINDING_VERSION,
            "parent_treatment_id": candidate.parent_treatment_id,
            "recipe_id": candidate.recipe_id,
            "program_set_hash": candidate.program_set_hash,
            "external_skill_source_receipt_hash": (
                candidate.external_skill_source_receipt_hash
            ),
            "preregistration_hash": preregistration_hash,
            "runtime_source_closure_hash": runtime_source_closure_hash,
            "pack_commitment_hash": pack_commitment_hash,
        }
    )
    body["period_out_treatment_id"] = period_out_treatment_id
    return {**body, "binding_hash": stable_hash(body)}


def validate_replication_evaluator_binding_v1(
    payload: Mapping[str, Any],
    *,
    candidate: FixedFinancialCandidateIdentityV1,
    require_pack_commitment: bool,
) -> str:
    _validate_candidate_identity_shape(candidate)
    expected_fields = {
        "binding_version",
        "parent_manifest_hash",
        "candidate_id",
        "candidate_manifest_hash",
        "recipe_id",
        "program_set_hash",
        "parent_treatment_id",
        "external_skill_source_receipt_hash",
        "preregistration_hash",
        "runtime_source_closure_hash",
        "pack_commitment_hash",
        "candidate_recipe_and_source_identity_reused_exactly",
        "evaluator_wrapper_is_not_candidate_content",
        "performance_gate_bound",
        "promotion_authorized",
        "period_out_treatment_id",
        "binding_hash",
    }
    if set(payload) != expected_fields:
        raise FinancialSemanticReplicationTreatmentError(
            "replication evaluator binding fields drifted"
        )
    body = dict(payload)
    declared = body.pop("binding_hash", None)
    expected_identity = {
        "parent_manifest_hash": candidate.parent_manifest_hash,
        "candidate_id": candidate.candidate_id,
        "candidate_manifest_hash": candidate.candidate_manifest_hash,
        "recipe_id": candidate.recipe_id,
        "program_set_hash": candidate.program_set_hash,
        "parent_treatment_id": candidate.parent_treatment_id,
        "external_skill_source_receipt_hash": (
            candidate.external_skill_source_receipt_hash
        ),
    }
    if (
        payload.get("binding_version")
        != REPLICATION_EVALUATOR_BINDING_VERSION
        or any(payload.get(key) != value for key, value in expected_identity.items())
        or payload.get("candidate_recipe_and_source_identity_reused_exactly")
        is not True
        or payload.get("evaluator_wrapper_is_not_candidate_content") is not True
        or payload.get("performance_gate_bound") is not False
        or payload.get("promotion_authorized") is not False
        or not _is_sha256(payload.get("preregistration_hash"))
        or not _is_sha256(payload.get("runtime_source_closure_hash"))
        or (require_pack_commitment and not _is_sha256(payload.get("pack_commitment_hash")))
        or (
            payload.get("pack_commitment_hash") is not None
            and not _is_sha256(payload.get("pack_commitment_hash"))
        )
        or not _is_sha256(payload.get("period_out_treatment_id"))
        or payload.get("period_out_treatment_id")
        != stable_hash(
            {
                "binding_version": REPLICATION_EVALUATOR_BINDING_VERSION,
                "parent_treatment_id": candidate.parent_treatment_id,
                "recipe_id": candidate.recipe_id,
                "program_set_hash": candidate.program_set_hash,
                "external_skill_source_receipt_hash": (
                    candidate.external_skill_source_receipt_hash
                ),
                "preregistration_hash": payload.get("preregistration_hash"),
                "runtime_source_closure_hash": payload.get(
                    "runtime_source_closure_hash"
                ),
                "pack_commitment_hash": payload.get(
                    "pack_commitment_hash"
                ),
            }
        )
        or declared != stable_hash(body)
    ):
        raise FinancialSemanticReplicationTreatmentError(
            "replication evaluator binding failed closed"
        )
    return str(declared)
