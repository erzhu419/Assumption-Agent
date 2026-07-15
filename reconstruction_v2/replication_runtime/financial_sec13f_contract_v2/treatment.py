from __future__ import annotations

"""Opaque candidate and evaluation-treatment identities for contract v2."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from assumption_agent.benchmarks.financial_sec13f_contract_operator_v2 import (
    load_contract_asset_v2,
    sha256_file,
)
from assumption_agent.benchmarks.skilllearn_compiler import (
    verify_skill_source_tree,
)
from assumption_agent.models import stable_hash


CANDIDATE_IDENTITY_VERSION = "financial_sec13f_contract_candidate_identity_v2"
EVALUATION_TREATMENT_VERSION = (
    "financial_sec13f_contract_fresh_evaluation_treatment_v2"
)
CANDIDATE_SOURCE_RELATIVE = (
    "candidates/financial_sec13f_contract_operator_v2"
)
ASSET_RELATIVE = "manifests/financial_sec13f_public_contract_asset_v2.json"
OPERATOR_RELATIVE = (
    "assumption_agent/benchmarks/financial_sec13f_contract_operator_v2.py"
)


class ContractTreatmentError(PermissionError):
    """A candidate or evaluation identity no longer matches its source."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class FixedContractCandidateV2:
    candidate_id: str
    asset_manifest_hash: str
    asset_file_sha256: str
    operator_source_sha256: str
    external_skill_source_receipt_hash: str
    recipe_id: str
    program_set_hash: str
    base_treatment_id: str
    candidate_skill_source: Path
    operator_asset_path: Path

    def verify(self) -> None:
        if not all(
            _is_sha256(value)
            for value in (
                self.candidate_id,
                self.asset_manifest_hash,
                self.asset_file_sha256,
                self.operator_source_sha256,
                self.external_skill_source_receipt_hash,
                self.recipe_id,
                self.program_set_hash,
                self.base_treatment_id,
            )
        ):
            raise ContractTreatmentError("candidate identity is malformed")
        expected_recipe = stable_hash(
            {
                "identity_version": CANDIDATE_IDENTITY_VERSION,
                "candidate_id": self.candidate_id,
                "asset_manifest_hash": self.asset_manifest_hash,
                "asset_file_sha256": self.asset_file_sha256,
                "operator_source_sha256": self.operator_source_sha256,
                "external_skill_source_receipt_hash": (
                    self.external_skill_source_receipt_hash
                ),
            }
        )
        if self.recipe_id != expected_recipe:
            raise ContractTreatmentError("candidate recipe identity drifted")
        if self.program_set_hash != stable_hash(
            {"recipe_ids": [self.recipe_id]}
        ):
            raise ContractTreatmentError("candidate program set drifted")
        expected_base = stable_hash(
            {
                "identity_version": CANDIDATE_IDENTITY_VERSION,
                "recipe_id": self.recipe_id,
                "program_set_hash": self.program_set_hash,
                "operator_is_candidate_content": True,
                "post_agent_pre_verifier_treatment": True,
            }
        )
        if self.base_treatment_id != expected_base:
            raise ContractTreatmentError("base treatment identity drifted")

    def safe_payload(self, project_root: str | Path) -> dict[str, Any]:
        self.verify()
        project = Path(project_root).resolve(strict=True)
        return {
            "identity_version": CANDIDATE_IDENTITY_VERSION,
            "candidate_id": self.candidate_id,
            "asset_manifest_hash": self.asset_manifest_hash,
            "asset_file_sha256": self.asset_file_sha256,
            "operator_source_sha256": self.operator_source_sha256,
            "external_skill_source_receipt_hash": (
                self.external_skill_source_receipt_hash
            ),
            "recipe_id": self.recipe_id,
            "program_set_hash": self.program_set_hash,
            "base_treatment_id": self.base_treatment_id,
            "candidate_skill_source": self.candidate_skill_source.relative_to(
                project
            ).as_posix(),
            "operator_asset_path": self.operator_asset_path.relative_to(
                project
            ).as_posix(),
            "operator_is_candidate_content": True,
            "candidate_recipe_reused_from_v1": False,
        }


def load_fixed_contract_candidate_v2(
    project_root: str | Path,
) -> FixedContractCandidateV2:
    project = Path(project_root).expanduser().resolve(strict=True)
    source = (project / CANDIDATE_SOURCE_RELATIVE).resolve(strict=True)
    asset_path = (project / ASSET_RELATIVE).resolve(strict=True)
    operator_path = (project / OPERATOR_RELATIVE).resolve(strict=True)
    for path in (source, asset_path, operator_path):
        try:
            path.relative_to(project)
        except ValueError as exc:
            raise ContractTreatmentError(
                "candidate path escaped the project"
            ) from exc
    if not source.is_dir() or not asset_path.is_file() or not operator_path.is_file():
        raise ContractTreatmentError("candidate source or asset is unavailable")
    source_receipt = verify_skill_source_tree(source)
    asset = load_contract_asset_v2(asset_path)
    if (
        asset["candidate_skill_source_receipt_hash"]
        != source_receipt.receipt_hash
        or asset["operator_source_sha256"] != sha256_file(operator_path)
    ):
        raise ContractTreatmentError("candidate asset/source binding drifted")
    asset_file_sha256 = sha256_file(asset_path)
    recipe_id = stable_hash(
        {
            "identity_version": CANDIDATE_IDENTITY_VERSION,
            "candidate_id": asset["candidate_id"],
            "asset_manifest_hash": asset["manifest_hash"],
            "asset_file_sha256": asset_file_sha256,
            "operator_source_sha256": asset["operator_source_sha256"],
            "external_skill_source_receipt_hash": source_receipt.receipt_hash,
        }
    )
    program_set_hash = stable_hash({"recipe_ids": [recipe_id]})
    base_treatment_id = stable_hash(
        {
            "identity_version": CANDIDATE_IDENTITY_VERSION,
            "recipe_id": recipe_id,
            "program_set_hash": program_set_hash,
            "operator_is_candidate_content": True,
            "post_agent_pre_verifier_treatment": True,
        }
    )
    candidate = FixedContractCandidateV2(
        candidate_id=asset["candidate_id"],
        asset_manifest_hash=asset["manifest_hash"],
        asset_file_sha256=asset_file_sha256,
        operator_source_sha256=asset["operator_source_sha256"],
        external_skill_source_receipt_hash=source_receipt.receipt_hash,
        recipe_id=recipe_id,
        program_set_hash=program_set_hash,
        base_treatment_id=base_treatment_id,
        candidate_skill_source=source,
        operator_asset_path=asset_path,
    )
    candidate.verify()
    return candidate


def build_evaluation_treatment_v2(
    *,
    candidate: FixedContractCandidateV2,
    execution_source_closure_hash: str,
    measurement_view_hash: str,
    benchmark_tree_hash: str,
) -> dict[str, Any]:
    candidate.verify()
    bindings = {
        "execution_source_closure_hash": execution_source_closure_hash,
        "measurement_view_hash": measurement_view_hash,
        "benchmark_tree_hash": benchmark_tree_hash,
    }
    if not all(_is_sha256(value) for value in bindings.values()):
        raise ContractTreatmentError("evaluation treatment binding is malformed")
    body: dict[str, Any] = {
        "treatment_version": EVALUATION_TREATMENT_VERSION,
        "candidate_id": candidate.candidate_id,
        "recipe_id": candidate.recipe_id,
        "program_set_hash": candidate.program_set_hash,
        "base_treatment_id": candidate.base_treatment_id,
        "external_skill_source_receipt_hash": (
            candidate.external_skill_source_receipt_hash
        ),
        **bindings,
        "operator_is_candidate_content": True,
        "performance_gate_bound": False,
        "promotion_authorized": False,
    }
    body["period_out_treatment_id"] = stable_hash(body)
    return {**body, "binding_hash": stable_hash({**body})}


def validate_evaluation_treatment_v2(
    value: Mapping[str, Any],
    *,
    candidate: FixedContractCandidateV2,
) -> str:
    candidate.verify()
    expected_fields = {
        "treatment_version",
        "candidate_id",
        "recipe_id",
        "program_set_hash",
        "base_treatment_id",
        "external_skill_source_receipt_hash",
        "execution_source_closure_hash",
        "measurement_view_hash",
        "benchmark_tree_hash",
        "operator_is_candidate_content",
        "performance_gate_bound",
        "promotion_authorized",
        "period_out_treatment_id",
        "binding_hash",
    }
    if set(value) != expected_fields:
        raise ContractTreatmentError("evaluation treatment fields drifted")
    body = dict(value)
    declared_binding = body.pop("binding_hash", None)
    declared_treatment = body.get("period_out_treatment_id")
    treatment_body = dict(body)
    treatment_body.pop("period_out_treatment_id")
    if (
        declared_binding != stable_hash(body)
        or declared_treatment != stable_hash(treatment_body)
        or value.get("treatment_version") != EVALUATION_TREATMENT_VERSION
        or value.get("candidate_id") != candidate.candidate_id
        or value.get("recipe_id") != candidate.recipe_id
        or value.get("program_set_hash") != candidate.program_set_hash
        or value.get("base_treatment_id") != candidate.base_treatment_id
        or value.get("external_skill_source_receipt_hash")
        != candidate.external_skill_source_receipt_hash
        or value.get("operator_is_candidate_content") is not True
        or value.get("performance_gate_bound") is not False
        or value.get("promotion_authorized") is not False
    ):
        raise ContractTreatmentError("evaluation treatment failed closed")
    for field in (
        "execution_source_closure_hash",
        "measurement_view_hash",
        "benchmark_tree_hash",
        "period_out_treatment_id",
        "binding_hash",
    ):
        if not _is_sha256(value.get(field)):
            raise ContractTreatmentError("evaluation treatment hash is malformed")
    return str(declared_binding)
