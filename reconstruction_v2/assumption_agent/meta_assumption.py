"""Content-addressed meta-assumption sidecars.

This module deliberately leaves :class:`HypothesisProgram` unchanged.  The
objects below live upstream of executable treatments and bind a frozen
ontology, a problem-specific claim, TRAIN-only diagnostic evidence, and the
compiler output without changing the identity of existing programs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import re
from typing import Any, Mapping, Protocol, Sequence

from .evaluation import hypothesis_program_behavior_hash
from .models import (
    ActionNode,
    HypothesisKind,
    HypothesisProgram,
    SplitName,
    stable_hash,
)


META_ASSUMPTION_SIDECAR_VERSION = "meta_assumption_sidecar_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_.-]{2,127}\Z")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _valid_identifier(value: object) -> bool:
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def _strict_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _unique_nonempty(values: Sequence[str]) -> bool:
    return bool(values) and len(set(values)) == len(values) and all(
        isinstance(value, str) and value.strip() == value and bool(value)
        for value in values
    )


def _unique_strings(values: Sequence[str]) -> bool:
    return len(set(values)) == len(values) and all(
        isinstance(value, str) and value.strip() == value and bool(value)
        for value in values
    )


class AssumptionRole(str, Enum):
    WORLD_CLAIM = "world_claim"
    REPRESENTATION_PRIOR = "representation_prior"
    REGULARIZER = "regularizer"
    GOVERNANCE_RULE = "governance_rule"
    DECISION_RULE = "decision_rule"


class CompilerTarget(str, Enum):
    TASK_PROGRAM = "task_program"
    POLICY_PROGRAM = "policy_program"
    EVALUATOR_ARTIFACT = "evaluator_artifact"
    IMPLEMENTATION_CONTRACT = "implementation_contract"
    NO_DIRECT_TREATMENT = "no_direct_treatment"


class ProbeDisposition(str, Enum):
    SUPPORTED = "supported"
    FALSIFIED = "falsified"
    INCONCLUSIVE = "inconclusive"


class TreatmentDisposition(str, Enum):
    ACTIVE_PROGRAM = "active_program"
    PRESERVE_BASELINE = "preserve_baseline"
    EVALUATOR_ARTIFACT = "evaluator_artifact"


@dataclass(frozen=True)
class CompilerTrustAnchor:
    """Harness-owned compiler identity used to verify untrusted receipts."""

    compiler_id: str
    compiler_version: str
    implementation_hash: str
    primary_metric: str

    @property
    def anchor_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.compiler_id):
            issues.append("compiler_trust_anchor_id_invalid")
        if not _valid_identifier(self.compiler_version):
            issues.append("compiler_trust_anchor_version_invalid")
        if not _is_sha256(self.implementation_hash):
            issues.append("compiler_trust_anchor_hash_invalid")
        if (
            not isinstance(self.primary_metric, str)
            or not self.primary_metric.strip()
        ):
            issues.append("compiler_trust_anchor_metric_missing")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "compiler_id": self.compiler_id,
            "compiler_version": self.compiler_version,
            "implementation_hash": self.implementation_hash,
            "primary_metric": self.primary_metric,
        }


@dataclass(frozen=True)
class OntologyRoot:
    root_id: str
    title: str
    description: str

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.root_id):
            issues.append("ontology_root_id_invalid")
        if not self.title.strip():
            issues.append("ontology_root_title_missing")
        if not self.description.strip():
            issues.append("ontology_root_description_missing")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "title": self.title,
            "description": self.description,
        }


@dataclass(frozen=True)
class DiagnosticProbePlan:
    probe_id: str
    observable_ids: tuple[str, ...]
    support_rule_id: str
    counter_rule_id: str
    max_evaluations: int
    train_only: bool = True

    @property
    def plan_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.probe_id):
            issues.append("diagnostic_probe_id_invalid")
        if not _unique_nonempty(self.observable_ids):
            issues.append("diagnostic_probe_observables_invalid")
        if not _valid_identifier(self.support_rule_id):
            issues.append("diagnostic_probe_support_rule_invalid")
        if not _valid_identifier(self.counter_rule_id):
            issues.append("diagnostic_probe_counter_rule_invalid")
        if self.support_rule_id == self.counter_rule_id:
            issues.append("diagnostic_probe_rules_not_distinct")
        if (
            not isinstance(self.max_evaluations, int)
            or isinstance(self.max_evaluations, bool)
            or self.max_evaluations <= 0
        ):
            issues.append("diagnostic_probe_budget_invalid")
        if self.train_only is not True:
            issues.append("diagnostic_probe_not_train_only")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "observable_ids": list(self.observable_ids),
            "support_rule_id": self.support_rule_id,
            "counter_rule_id": self.counter_rule_id,
            "max_evaluations": self.max_evaluations,
            "train_only": self.train_only,
        }


@dataclass(frozen=True)
class MetaAssumptionTemplate:
    template_id: str
    primary_parent_id: str
    parent_ids: tuple[str, ...]
    roles: tuple[AssumptionRole, ...]
    claim_schema: str
    admissible_variable_types: tuple[str, ...]
    support_signatures: tuple[str, ...]
    counter_signatures: tuple[str, ...]
    probe_plan: DiagnosticProbePlan
    compiler_targets: tuple[CompilerTarget, ...]
    not_applicable_conditions: tuple[str, ...]
    invariances: tuple[str, ...] = ()

    @property
    def template_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def validate(self, *, root_ids: frozenset[str]) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.template_id):
            issues.append("assumption_template_id_invalid")
        if self.primary_parent_id not in root_ids:
            issues.append("assumption_template_primary_parent_unknown")
        if (
            not _unique_nonempty(self.parent_ids)
            or self.primary_parent_id not in self.parent_ids
            or any(parent not in root_ids for parent in self.parent_ids)
        ):
            issues.append("assumption_template_parents_invalid")
        if (
            not self.roles
            or len(set(self.roles)) != len(self.roles)
            or any(not isinstance(role, AssumptionRole) for role in self.roles)
        ):
            issues.append("assumption_template_roles_invalid")
        if not self.claim_schema.strip():
            issues.append("assumption_template_claim_schema_missing")
        for values, issue in (
            (
                self.admissible_variable_types,
                "assumption_template_variable_types_invalid",
            ),
            (
                self.support_signatures,
                "assumption_template_support_signatures_invalid",
            ),
            (
                self.counter_signatures,
                "assumption_template_counter_signatures_invalid",
            ),
            (
                self.not_applicable_conditions,
                "assumption_template_not_applicable_invalid",
            ),
        ):
            if not _unique_nonempty(values):
                issues.append(issue)
        if set(self.support_signatures) & set(self.counter_signatures):
            issues.append("assumption_template_support_counter_overlap")
        if (
            not self.compiler_targets
            or len(set(self.compiler_targets)) != len(self.compiler_targets)
            or any(
                not isinstance(target, CompilerTarget)
                for target in self.compiler_targets
            )
        ):
            issues.append("assumption_template_compiler_targets_invalid")
        if len(set(self.invariances)) != len(self.invariances) or any(
            not isinstance(value, str)
            or value.strip() != value
            or not value
            for value in self.invariances
        ):
            issues.append("assumption_template_invariances_invalid")
        issues.extend(self.probe_plan.validate())
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "primary_parent_id": self.primary_parent_id,
            "parent_ids": list(self.parent_ids),
            "roles": [role.value for role in self.roles],
            "claim_schema": self.claim_schema,
            "admissible_variable_types": list(
                self.admissible_variable_types
            ),
            "support_signatures": list(self.support_signatures),
            "counter_signatures": list(self.counter_signatures),
            "probe_plan": self.probe_plan.safe_payload(),
            "probe_plan_hash": self.probe_plan.plan_hash,
            "compiler_targets": [
                target.value for target in self.compiler_targets
            ],
            "not_applicable_conditions": list(
                self.not_applicable_conditions
            ),
            "invariances": list(self.invariances),
        }


@dataclass(frozen=True)
class LegacyAssumptionAlias:
    alias_id: str
    target_template_ids: tuple[str, ...]

    def validate(self, *, template_ids: frozenset[str]) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.alias_id):
            issues.append("legacy_assumption_alias_id_invalid")
        if (
            not _unique_nonempty(self.target_template_ids)
            or any(
                template_id not in template_ids
                for template_id in self.target_template_ids
            )
        ):
            issues.append("legacy_assumption_alias_targets_invalid")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "alias_id": self.alias_id,
            "target_template_ids": list(self.target_template_ids),
        }


@dataclass(frozen=True)
class UniversalAssumptionOntology:
    version: str
    roots: tuple[OntologyRoot, ...]
    templates: tuple[MetaAssumptionTemplate, ...]
    legacy_aliases: tuple[LegacyAssumptionAlias, ...]

    @property
    def ontology_hash(self) -> str:
        return stable_hash(self.safe_payload(include_hash=False))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.version):
            issues.append("assumption_ontology_version_invalid")
        root_ids = tuple(root.root_id for root in self.roots)
        template_ids = tuple(row.template_id for row in self.templates)
        alias_ids = tuple(row.alias_id for row in self.legacy_aliases)
        if not root_ids or len(root_ids) != len(set(root_ids)):
            issues.append("assumption_ontology_roots_invalid")
        if not template_ids or len(template_ids) != len(set(template_ids)):
            issues.append("assumption_ontology_templates_invalid")
        if not alias_ids or len(alias_ids) != len(set(alias_ids)):
            issues.append("assumption_ontology_aliases_invalid")
        issues.extend(issue for root in self.roots for issue in root.validate())
        root_set = frozenset(root_ids)
        issues.extend(
            issue
            for template in self.templates
            for issue in template.validate(root_ids=root_set)
        )
        template_set = frozenset(template_ids)
        issues.extend(
            issue
            for alias in self.legacy_aliases
            for issue in alias.validate(template_ids=template_set)
        )
        return tuple(sorted(set(issues)))

    def require_template(self, template_id: str) -> MetaAssumptionTemplate:
        if self.validate():
            raise PermissionError("assumption ontology is invalid")
        matches = [
            row for row in self.templates if row.template_id == template_id
        ]
        if len(matches) != 1:
            raise KeyError(f"unknown assumption template: {template_id}")
        return matches[0]

    def safe_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "sidecar_version": META_ASSUMPTION_SIDECAR_VERSION,
            "version": self.version,
            "roots": [
                row.safe_payload()
                for row in sorted(self.roots, key=lambda item: item.root_id)
            ],
            "templates": [
                {
                    **row.safe_payload(),
                    "template_hash": row.template_hash,
                }
                for row in sorted(
                    self.templates, key=lambda item: item.template_id
                )
            ],
            "legacy_aliases": [
                row.safe_payload()
                for row in sorted(
                    self.legacy_aliases, key=lambda item: item.alias_id
                )
            ],
            "template_count": len(self.templates),
            "legacy_alias_count": len(self.legacy_aliases),
        }
        if include_hash:
            payload["ontology_hash"] = self.ontology_hash
        return payload


@dataclass(frozen=True)
class HypothesisClaim:
    claim_id: str
    ontology_hash: str
    template_ids: tuple[str, ...]
    scope_hash: str
    mechanism_statement: str
    bound_variable_types: tuple[str, ...]
    observable_predictions: tuple[str, ...]
    counter_predictions: tuple[str, ...]
    competing_claim_ids: tuple[str, ...]
    description_length_bits: int
    evidence_receipt_hashes: tuple[str, ...]
    formation_split: SplitName = SplitName.TRAIN
    lineage_claim_ids: tuple[str, ...] = ()

    @property
    def claim_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def validate(
        self, ontology: UniversalAssumptionOntology
    ) -> tuple[str, ...]:
        issues: list[str] = []
        selected_templates: list[MetaAssumptionTemplate] = []
        if not _valid_identifier(self.claim_id):
            issues.append("hypothesis_claim_id_invalid")
        if self.ontology_hash != ontology.ontology_hash:
            issues.append("hypothesis_claim_ontology_mismatch")
        if (
            not _unique_nonempty(self.template_ids)
            or tuple(sorted(self.template_ids)) != self.template_ids
        ):
            issues.append("hypothesis_claim_template_ids_invalid")
        else:
            for template_id in self.template_ids:
                try:
                    selected_templates.append(
                        ontology.require_template(template_id)
                    )
                except (KeyError, PermissionError):
                    issues.append("hypothesis_claim_template_unknown")
        if not _is_sha256(self.scope_hash):
            issues.append("hypothesis_claim_scope_hash_invalid")
        if not self.mechanism_statement.strip():
            issues.append("hypothesis_claim_statement_missing")
        for values, issue in (
            (
                self.bound_variable_types,
                "hypothesis_claim_bound_variables_invalid",
            ),
            (
                self.observable_predictions,
                "hypothesis_claim_predictions_invalid",
            ),
            (
                self.counter_predictions,
                "hypothesis_claim_counter_predictions_invalid",
            ),
            (
                self.evidence_receipt_hashes,
                "hypothesis_claim_evidence_receipts_invalid",
            ),
        ):
            if not _unique_nonempty(values):
                issues.append(issue)
        if selected_templates and self.bound_variable_types:
            admissible_union = {
                variable_type
                for template in selected_templates
                for variable_type in template.admissible_variable_types
            }
            if any(
                variable_type not in admissible_union
                for variable_type in self.bound_variable_types
            ):
                issues.append(
                    "hypothesis_claim_bound_variable_not_admissible"
                )
            if any(
                not set(self.bound_variable_types).intersection(
                    template.admissible_variable_types
                )
                for template in selected_templates
            ):
                issues.append("hypothesis_claim_template_unbound")
        if selected_templates:
            if any(
                not set(self.observable_predictions).intersection(
                    template.support_signatures
                )
                for template in selected_templates
            ):
                issues.append(
                    "hypothesis_claim_template_support_signature_uncovered"
                )
        if selected_templates:
            if any(
                not set(self.counter_predictions).intersection(
                    template.counter_signatures
                )
                for template in selected_templates
            ):
                issues.append(
                    "hypothesis_claim_template_counter_signature_uncovered"
                )
        if set(self.observable_predictions) & set(self.counter_predictions):
            issues.append("hypothesis_claim_prediction_overlap")
        if any(
            not _is_sha256(value) for value in self.evidence_receipt_hashes
        ):
            issues.append("hypothesis_claim_evidence_receipt_hash_invalid")
        if (
            len(set(self.competing_claim_ids))
            != len(self.competing_claim_ids)
            or self.claim_id in self.competing_claim_ids
            or any(
                not _valid_identifier(value)
                for value in self.competing_claim_ids
            )
        ):
            issues.append("hypothesis_claim_competitors_invalid")
        if (
            not isinstance(self.description_length_bits, int)
            or isinstance(self.description_length_bits, bool)
            or self.description_length_bits <= 0
        ):
            issues.append("hypothesis_claim_description_length_invalid")
        if self.formation_split is not SplitName.TRAIN:
            issues.append("hypothesis_claim_not_train_formed")
        if (
            len(set(self.lineage_claim_ids)) != len(self.lineage_claim_ids)
            or self.claim_id in self.lineage_claim_ids
            or any(
                not _valid_identifier(value)
                for value in self.lineage_claim_ids
            )
        ):
            issues.append("hypothesis_claim_lineage_invalid")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "ontology_hash": self.ontology_hash,
            "template_ids": list(self.template_ids),
            "scope_hash": self.scope_hash,
            "mechanism_statement": self.mechanism_statement,
            "bound_variable_types": list(self.bound_variable_types),
            "observable_predictions": list(self.observable_predictions),
            "counter_predictions": list(self.counter_predictions),
            "competing_claim_ids": list(self.competing_claim_ids),
            "description_length_bits": self.description_length_bits,
            "evidence_receipt_hashes": list(
                self.evidence_receipt_hashes
            ),
            "formation_split": self.formation_split.value,
            "lineage_claim_ids": list(self.lineage_claim_ids),
        }


@dataclass(frozen=True)
class ProbeTrustAnchor:
    """Harness-owned identity for one closed probe/rule implementation."""

    verifier_id: str
    verifier_version: str
    implementation_hash: str
    probe_id: str
    support_rule_id: str
    counter_rule_id: str

    @property
    def anchor_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for value, issue in (
            (self.verifier_id, "probe_trust_anchor_verifier_id_invalid"),
            (
                self.verifier_version,
                "probe_trust_anchor_verifier_version_invalid",
            ),
            (self.probe_id, "probe_trust_anchor_probe_id_invalid"),
            (
                self.support_rule_id,
                "probe_trust_anchor_support_rule_id_invalid",
            ),
            (
                self.counter_rule_id,
                "probe_trust_anchor_counter_rule_id_invalid",
            ),
        ):
            if not _valid_identifier(value):
                issues.append(issue)
        if not _is_sha256(self.implementation_hash):
            issues.append("probe_trust_anchor_implementation_hash_invalid")
        if self.support_rule_id == self.counter_rule_id:
            issues.append("probe_trust_anchor_rules_not_distinct")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "verifier_id": self.verifier_id,
            "verifier_version": self.verifier_version,
            "implementation_hash": self.implementation_hash,
            "probe_id": self.probe_id,
            "support_rule_id": self.support_rule_id,
            "counter_rule_id": self.counter_rule_id,
        }


@dataclass(frozen=True)
class ProbeObservationStatistic:
    """Private statistic values committed jointly to one observation hash."""

    observation_hash: str
    statistic_values: tuple[tuple[str, str], ...]

    @property
    def statistic_commitment_hash(self) -> str:
        return stable_hash(self.private_payload())

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if not _is_sha256(self.observation_hash):
            issues.append("probe_observation_hash_invalid")
        keys = tuple(key for key, _ in self.statistic_values)
        if (
            not self.statistic_values
            or len(keys) != len(set(keys))
            or tuple(sorted(keys)) != keys
            or any(
                not isinstance(key, str)
                or not key
                or key.strip() != key
                or not isinstance(value, str)
                or value.strip() != value
                for key, value in self.statistic_values
            )
        ):
            issues.append("probe_observation_statistics_invalid")
        return tuple(sorted(set(issues)))

    def private_payload(self) -> dict[str, Any]:
        return {
            "observation_hash": self.observation_hash,
            "statistic_values": [
                {"statistic_id": key, "canonical_value": value}
                for key, value in self.statistic_values
            ],
        }

    def safe_commitment_payload(self) -> dict[str, Any]:
        return {
            "observation_hash": self.observation_hash,
            "statistic_commitment_hash": self.statistic_commitment_hash,
            "statistic_payload_persisted": False,
        }


@dataclass(frozen=True)
class ProbeEvidenceBundle:
    """TRAIN evidence whose private statistics are content committed."""

    bundle_id: str
    ontology_hash: str
    claim_hash: str
    template_id: str
    probe_plan_hash: str
    train_split_hash: str
    observation_statistics: tuple[ProbeObservationStatistic, ...]
    formation_split: SplitName = SplitName.TRAIN
    source_payload_access_count: int = 0
    validation_or_test_access_count: int = 0
    online_or_api_evaluation_count: int = 0

    @property
    def observation_hashes(self) -> tuple[str, ...]:
        return tuple(
            row.observation_hash for row in self.observation_statistics
        )

    @property
    def statistic_commitment_hashes(self) -> tuple[str, ...]:
        return tuple(
            row.statistic_commitment_hash
            for row in self.observation_statistics
        )

    @property
    def evidence_bundle_hash(self) -> str:
        return stable_hash(self.safe_payload())

    @property
    def expected_bundle_id(self) -> str:
        return "probe-evidence." + stable_hash(self.binding_payload())[:24]

    def binding_payload(self) -> dict[str, Any]:
        return {
            "sidecar_version": META_ASSUMPTION_SIDECAR_VERSION,
            "ontology_hash": self.ontology_hash,
            "claim_hash": self.claim_hash,
            "template_id": self.template_id,
            "probe_plan_hash": self.probe_plan_hash,
            "train_split_hash": self.train_split_hash,
            "observation_hashes": list(self.observation_hashes),
            "statistic_commitment_hashes": list(
                self.statistic_commitment_hashes
            ),
            "observation_count": len(self.observation_statistics),
            "formation_split": self.formation_split.value,
            "source_payload_access_count": self.source_payload_access_count,
            "validation_or_test_access_count": (
                self.validation_or_test_access_count
            ),
            "online_or_api_evaluation_count": (
                self.online_or_api_evaluation_count
            ),
        }

    def validate(
        self,
        *,
        ontology: UniversalAssumptionOntology,
        claim: HypothesisClaim,
    ) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.bundle_id):
            issues.append("probe_evidence_bundle_id_invalid")
        elif self.bundle_id != self.expected_bundle_id:
            issues.append("probe_evidence_bundle_id_binding_mismatch")
        if self.ontology_hash != ontology.ontology_hash:
            issues.append("probe_evidence_ontology_mismatch")
        if self.claim_hash != claim.claim_hash:
            issues.append("probe_evidence_claim_mismatch")
        if self.template_id not in claim.template_ids:
            issues.append("probe_evidence_template_not_claimed")
            template = None
        else:
            try:
                template = ontology.require_template(self.template_id)
            except (KeyError, PermissionError):
                template = None
                issues.append("probe_evidence_template_unknown")
        if template is not None and (
            self.probe_plan_hash != template.probe_plan.plan_hash
            or len(self.observation_statistics)
            > template.probe_plan.max_evaluations
        ):
            issues.append("probe_evidence_plan_mismatch")
        if not _is_sha256(self.train_split_hash):
            issues.append("probe_evidence_train_split_hash_invalid")
        if (
            not self.observation_statistics
            or len(set(self.observation_hashes))
            != len(self.observation_hashes)
            or tuple(sorted(self.observation_hashes))
            != self.observation_hashes
        ):
            issues.append("probe_evidence_observations_invalid")
        issues.extend(
            issue
            for row in self.observation_statistics
            for issue in row.validate()
        )
        for value, issue in (
            (
                self.source_payload_access_count,
                "probe_evidence_source_access_count_invalid",
            ),
            (
                self.validation_or_test_access_count,
                "probe_evidence_heldout_access_count_invalid",
            ),
            (
                self.online_or_api_evaluation_count,
                "probe_evidence_online_evaluation_count_invalid",
            ),
        ):
            if not _strict_nonnegative_int(value):
                issues.append(issue)
        if self.formation_split is not SplitName.TRAIN:
            issues.append("probe_evidence_not_train_only")
        if self.validation_or_test_access_count != 0:
            issues.append("probe_evidence_heldout_accessed")
        if self.online_or_api_evaluation_count != 0:
            issues.append("probe_evidence_online_evaluation_used")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            **self.binding_payload(),
            "observation_statistic_commitments": [
                row.safe_commitment_payload()
                for row in self.observation_statistics
            ],
        }


@dataclass(frozen=True)
class ProbeVerificationResult:
    observed_support_signature_ids: tuple[str, ...] = ()
    observed_counter_signature_ids: tuple[str, ...] = ()

    def validate(
        self,
        *,
        template: MetaAssumptionTemplate,
        claim: HypothesisClaim,
    ) -> tuple[str, ...]:
        issues: list[str] = []
        if not _unique_strings(self.observed_support_signature_ids):
            issues.append("probe_verification_support_signatures_invalid")
        if not _unique_strings(self.observed_counter_signature_ids):
            issues.append("probe_verification_counter_signatures_invalid")
        if set(self.observed_support_signature_ids).intersection(
            self.observed_counter_signature_ids
        ):
            issues.append("probe_verification_signature_overlap")
        authorized_support = set(template.support_signatures).intersection(
            claim.observable_predictions
        )
        authorized_counter = set(template.counter_signatures).intersection(
            claim.counter_predictions
        )
        if not set(self.observed_support_signature_ids).issubset(
            authorized_support
        ):
            issues.append("probe_verification_support_not_authorized")
        if not set(self.observed_counter_signature_ids).issubset(
            authorized_counter
        ):
            issues.append("probe_verification_counter_not_authorized")
        return tuple(sorted(set(issues)))


@dataclass(frozen=True)
class ProbeReceipt:
    receipt_id: str
    ontology_hash: str
    claim_hash: str
    template_id: str
    probe_id: str
    probe_plan_hash: str
    probe_trust_anchor_hash: str
    evidence_bundle_hash: str
    train_split_hash: str
    observation_hashes: tuple[str, ...]
    support_count: int
    counter_count: int
    observation_count: int
    budget_used: int
    budget_limit: int
    disposition: ProbeDisposition
    observed_support_signature_ids: tuple[str, ...] = ()
    observed_counter_signature_ids: tuple[str, ...] = ()
    formation_split: SplitName = SplitName.TRAIN
    source_payload_access_count: int = 0
    validation_or_test_access_count: int = 0
    online_or_api_evaluation_count: int = 0

    @property
    def receipt_hash(self) -> str:
        return stable_hash(self.safe_payload())

    @property
    def expected_receipt_id(self) -> str:
        return "probe-receipt." + stable_hash(self.binding_payload())[:24]

    @property
    def falsified(self) -> bool:
        return self.disposition is ProbeDisposition.FALSIFIED

    @property
    def support_score_fraction(self) -> tuple[int, int]:
        denominator = self.observation_count or 1
        return (self.support_count - self.counter_count, denominator)

    def validate(
        self,
        *,
        ontology: UniversalAssumptionOntology,
        claim: HypothesisClaim,
    ) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.receipt_id):
            issues.append("probe_receipt_id_invalid")
        elif self.receipt_id != self.expected_receipt_id:
            issues.append("probe_receipt_id_binding_mismatch")
        if self.ontology_hash != ontology.ontology_hash:
            issues.append("probe_receipt_ontology_mismatch")
        if self.claim_hash != claim.claim_hash:
            issues.append("probe_receipt_claim_mismatch")
        if self.template_id not in claim.template_ids:
            issues.append("probe_receipt_template_not_claimed")
            template = None
        else:
            try:
                template = ontology.require_template(self.template_id)
            except (KeyError, PermissionError):
                template = None
                issues.append("probe_receipt_template_unknown")
        if template is not None and (
            self.probe_id != template.probe_plan.probe_id
            or self.probe_plan_hash != template.probe_plan.plan_hash
            or self.budget_limit != template.probe_plan.max_evaluations
        ):
            issues.append("probe_receipt_plan_mismatch")
        if not _is_sha256(self.probe_trust_anchor_hash):
            issues.append("probe_receipt_trust_anchor_hash_invalid")
        if not _is_sha256(self.evidence_bundle_hash):
            issues.append("probe_receipt_evidence_bundle_hash_invalid")
        if not _unique_strings(self.observed_support_signature_ids):
            issues.append(
                "probe_receipt_observed_support_signatures_invalid"
            )
        if not _unique_strings(self.observed_counter_signature_ids):
            issues.append(
                "probe_receipt_observed_counter_signatures_invalid"
            )
        if set(self.observed_support_signature_ids).intersection(
            self.observed_counter_signature_ids
        ):
            issues.append("probe_receipt_observed_signature_overlap")
        if template is not None:
            authorized_support = set(
                template.support_signatures
            ).intersection(claim.observable_predictions)
            authorized_counter = set(
                template.counter_signatures
            ).intersection(claim.counter_predictions)
            if not set(self.observed_support_signature_ids).issubset(
                authorized_support
            ):
                issues.append(
                    "probe_receipt_support_signature_not_authorized"
                )
            if not set(self.observed_counter_signature_ids).issubset(
                authorized_counter
            ):
                issues.append(
                    "probe_receipt_counter_signature_not_authorized"
                )
        if not _is_sha256(self.train_split_hash):
            issues.append("probe_receipt_train_split_hash_invalid")
        if (
            len(set(self.observation_hashes))
            != len(self.observation_hashes)
            or any(
                not _is_sha256(value) for value in self.observation_hashes
            )
        ):
            issues.append("probe_receipt_observation_hashes_invalid")
        for value, issue in (
            (self.support_count, "probe_receipt_support_count_invalid"),
            (self.counter_count, "probe_receipt_counter_count_invalid"),
            (
                self.observation_count,
                "probe_receipt_observation_count_invalid",
            ),
            (self.budget_used, "probe_receipt_budget_used_invalid"),
            (self.budget_limit, "probe_receipt_budget_limit_invalid"),
            (
                self.source_payload_access_count,
                "probe_receipt_source_access_count_invalid",
            ),
            (
                self.validation_or_test_access_count,
                "probe_receipt_heldout_access_count_invalid",
            ),
            (
                self.online_or_api_evaluation_count,
                "probe_receipt_online_evaluation_count_invalid",
            ),
        ):
            if not _strict_nonnegative_int(value):
                issues.append(issue)
        if self.observation_count <= 0:
            issues.append("probe_receipt_observations_empty")
        if (
            self.observation_count != len(self.observation_hashes)
            or self.observation_count <= 0
            or self.support_count
            != len(self.observed_support_signature_ids)
            or self.counter_count
            != len(self.observed_counter_signature_ids)
            or self.support_count + self.counter_count
            > self.observation_count
            or self.budget_used != self.observation_count
            or self.budget_used > self.budget_limit
        ):
            issues.append("probe_receipt_count_contract_invalid")
        if self.formation_split is not SplitName.TRAIN:
            issues.append("probe_receipt_not_train_only")
        if self.validation_or_test_access_count != 0:
            issues.append("probe_receipt_heldout_accessed")
        if self.online_or_api_evaluation_count != 0:
            issues.append("probe_receipt_online_evaluation_used")
        if (
            self.disposition is ProbeDisposition.SUPPORTED
            and self.support_count <= self.counter_count
        ):
            issues.append("probe_receipt_supported_without_net_support")
        elif (
            self.disposition is ProbeDisposition.FALSIFIED
            and self.counter_count <= self.support_count
        ):
            issues.append("probe_receipt_falsified_without_net_counter")
        elif (
            self.disposition is ProbeDisposition.INCONCLUSIVE
            and self.support_count != self.counter_count
        ):
            issues.append("probe_receipt_inconclusive_with_net_evidence")
        return tuple(sorted(set(issues)))

    def binding_payload(self) -> dict[str, Any]:
        numerator, denominator = self.support_score_fraction
        return {
            "sidecar_version": META_ASSUMPTION_SIDECAR_VERSION,
            "ontology_hash": self.ontology_hash,
            "claim_hash": self.claim_hash,
            "template_id": self.template_id,
            "probe_id": self.probe_id,
            "probe_plan_hash": self.probe_plan_hash,
            "probe_trust_anchor_hash": self.probe_trust_anchor_hash,
            "evidence_bundle_hash": self.evidence_bundle_hash,
            "train_split_hash": self.train_split_hash,
            "observation_hashes": list(self.observation_hashes),
            "observed_support_signature_ids": list(
                self.observed_support_signature_ids
            ),
            "observed_counter_signature_ids": list(
                self.observed_counter_signature_ids
            ),
            "support_count": self.support_count,
            "counter_count": self.counter_count,
            "observation_count": self.observation_count,
            "support_score_fraction": {
                "numerator": numerator,
                "denominator": denominator,
            },
            "budget_used": self.budget_used,
            "budget_limit": self.budget_limit,
            "disposition": self.disposition.value,
            "formation_split": self.formation_split.value,
            "source_payload_access_count": self.source_payload_access_count,
            "validation_or_test_access_count": (
                self.validation_or_test_access_count
            ),
            "online_or_api_evaluation_count": (
                self.online_or_api_evaluation_count
            ),
        }

    def safe_payload(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            **self.binding_payload(),
        }


class ProbeVerifier(Protocol):
    verifier_id: str
    verifier_version: str
    implementation_hash: str
    probe_id: str
    support_rule_id: str
    counter_rule_id: str

    def match_signatures(
        self,
        *,
        template: MetaAssumptionTemplate,
        claim: HypothesisClaim,
        evidence: ProbeEvidenceBundle,
    ) -> ProbeVerificationResult: ...


class ProbeVerifierRegistry:
    """Closed harness registry that derives receipts from committed evidence."""

    def __init__(self) -> None:
        self._verifiers: dict[
            tuple[str, str, str], ProbeVerifier
        ] = {}
        self._trust_anchors: dict[
            tuple[str, str, str], ProbeTrustAnchor
        ] = {}

    @staticmethod
    def _key(
        probe_id: str,
        support_rule_id: str,
        counter_rule_id: str,
    ) -> tuple[str, str, str]:
        return (probe_id, support_rule_id, counter_rule_id)

    @staticmethod
    def _observed_anchor(verifier: ProbeVerifier) -> ProbeTrustAnchor:
        return ProbeTrustAnchor(
            verifier_id=verifier.verifier_id,
            verifier_version=verifier.verifier_version,
            implementation_hash=verifier.implementation_hash,
            probe_id=verifier.probe_id,
            support_rule_id=verifier.support_rule_id,
            counter_rule_id=verifier.counter_rule_id,
        )

    def register(
        self,
        verifier: ProbeVerifier,
        *,
        trust_anchor: ProbeTrustAnchor,
    ) -> None:
        observed_anchor = self._observed_anchor(verifier)
        if observed_anchor.validate():
            raise PermissionError("probe verifier identity is invalid")
        if trust_anchor.validate():
            raise PermissionError("probe trust anchor is invalid")
        if observed_anchor != trust_anchor:
            raise PermissionError(
                "probe verifier does not match trust anchor"
            )
        key = self._key(
            trust_anchor.probe_id,
            trust_anchor.support_rule_id,
            trust_anchor.counter_rule_id,
        )
        existing_anchor = self._trust_anchors.get(key)
        existing_verifier = self._verifiers.get(key)
        if existing_anchor is not None and (
            existing_anchor != trust_anchor
            or (
                existing_verifier is None
            )
            or (
                existing_verifier is not verifier
                and self._observed_anchor(existing_verifier)
                != observed_anchor
            )
        ):
            raise PermissionError("probe verifier registry conflict")
        self._verifiers[key] = verifier
        self._trust_anchors[key] = trust_anchor

    def _trusted_verifier(
        self,
        template: MetaAssumptionTemplate,
    ) -> tuple[ProbeVerifier, ProbeTrustAnchor]:
        plan = template.probe_plan
        key = self._key(
            plan.probe_id,
            plan.support_rule_id,
            plan.counter_rule_id,
        )
        verifier = self._verifiers.get(key)
        anchor = self._trust_anchors.get(key)
        if verifier is None or anchor is None:
            raise PermissionError(
                "probe verifier is not registered for template plan"
            )
        if (
            anchor.validate()
            or self._observed_anchor(verifier) != anchor
        ):
            raise PermissionError(
                "registered probe verifier no longer matches trust anchor"
            )
        return verifier, anchor

    def require_trust_anchor(
        self,
        template: MetaAssumptionTemplate,
    ) -> ProbeTrustAnchor:
        _, anchor = self._trusted_verifier(template)
        return anchor

    def _derive_result(
        self,
        *,
        ontology: UniversalAssumptionOntology,
        claim: HypothesisClaim,
        evidence: ProbeEvidenceBundle,
    ) -> tuple[
        MetaAssumptionTemplate,
        ProbeTrustAnchor,
        ProbeVerificationResult,
    ]:
        evidence_issues = evidence.validate(
            ontology=ontology,
            claim=claim,
        )
        if evidence_issues:
            raise PermissionError(
                f"probe evidence is invalid: {list(evidence_issues)}"
            )
        template = ontology.require_template(evidence.template_id)
        verifier, anchor = self._trusted_verifier(template)
        result = verifier.match_signatures(
            template=template,
            claim=claim,
            evidence=evidence,
        )
        if not isinstance(result, ProbeVerificationResult):
            raise PermissionError(
                "probe verifier returned an invalid result type"
            )
        result_issues = result.validate(
            template=template,
            claim=claim,
        )
        if result_issues:
            raise PermissionError(
                f"probe verifier result is invalid: {list(result_issues)}"
            )
        return template, anchor, result

    @staticmethod
    def _receipt_from_result(
        *,
        ontology: UniversalAssumptionOntology,
        claim: HypothesisClaim,
        evidence: ProbeEvidenceBundle,
        template: MetaAssumptionTemplate,
        anchor: ProbeTrustAnchor,
        result: ProbeVerificationResult,
    ) -> ProbeReceipt:
        support_count = len(result.observed_support_signature_ids)
        counter_count = len(result.observed_counter_signature_ids)
        if support_count > counter_count:
            disposition = ProbeDisposition.SUPPORTED
        elif counter_count > support_count:
            disposition = ProbeDisposition.FALSIFIED
        else:
            disposition = ProbeDisposition.INCONCLUSIVE
        receipt = ProbeReceipt(
            receipt_id="probe-receipt.pending",
            ontology_hash=ontology.ontology_hash,
            claim_hash=claim.claim_hash,
            template_id=template.template_id,
            probe_id=template.probe_plan.probe_id,
            probe_plan_hash=template.probe_plan.plan_hash,
            probe_trust_anchor_hash=anchor.anchor_hash,
            evidence_bundle_hash=evidence.evidence_bundle_hash,
            train_split_hash=evidence.train_split_hash,
            observation_hashes=evidence.observation_hashes,
            support_count=support_count,
            counter_count=counter_count,
            observation_count=len(evidence.observation_statistics),
            budget_used=len(evidence.observation_statistics),
            budget_limit=template.probe_plan.max_evaluations,
            disposition=disposition,
            observed_support_signature_ids=(
                result.observed_support_signature_ids
            ),
            observed_counter_signature_ids=(
                result.observed_counter_signature_ids
            ),
            formation_split=evidence.formation_split,
            source_payload_access_count=(
                evidence.source_payload_access_count
            ),
            validation_or_test_access_count=(
                evidence.validation_or_test_access_count
            ),
            online_or_api_evaluation_count=(
                evidence.online_or_api_evaluation_count
            ),
        )
        return replace(receipt, receipt_id=receipt.expected_receipt_id)

    def issue_receipt(
        self,
        *,
        ontology: UniversalAssumptionOntology,
        claim: HypothesisClaim,
        evidence: ProbeEvidenceBundle,
    ) -> ProbeReceipt:
        template, anchor, result = self._derive_result(
            ontology=ontology,
            claim=claim,
            evidence=evidence,
        )
        receipt = self._receipt_from_result(
            ontology=ontology,
            claim=claim,
            evidence=evidence,
            template=template,
            anchor=anchor,
            result=result,
        )
        issues = receipt.validate(ontology=ontology, claim=claim)
        if issues:
            raise PermissionError(
                f"trusted probe receipt is invalid: {list(issues)}"
            )
        return receipt

    def verify_receipt(
        self,
        receipt: ProbeReceipt,
        *,
        ontology: UniversalAssumptionOntology,
        claim: HypothesisClaim,
        evidence: ProbeEvidenceBundle,
    ) -> tuple[str, ...]:
        issues = list(
            receipt.validate(ontology=ontology, claim=claim)
        )
        try:
            template, anchor, result = self._derive_result(
                ontology=ontology,
                claim=claim,
                evidence=evidence,
            )
            expected = self._receipt_from_result(
                ontology=ontology,
                claim=claim,
                evidence=evidence,
                template=template,
                anchor=anchor,
                result=result,
            )
            if receipt != expected:
                issues.append(
                    "probe_receipt_trusted_recomputation_mismatch"
                )
        except (KeyError, PermissionError, TypeError, ValueError):
            issues.append("probe_receipt_trusted_verification_failed")
        return tuple(sorted(set(issues)))


def action_node_semantics_hash(action: ActionNode) -> str:
    """Return the canonical executable semantics of one real action node."""

    return stable_hash(
        {
            "operation": action.operation,
            "target": action.target,
            "value": action.value,
            "depends_on": sorted(action.depends_on),
        }
    )


@dataclass(frozen=True)
class RecipeActionBinding:
    """Bind one opaque recipe identifier to one actual action semantics."""

    recipe_id: str
    action_id: str
    action_semantics_hash: str

    @property
    def binding_hash(self) -> str:
        return stable_hash(self.safe_payload())

    def validate(
        self, *, program: HypothesisProgram
    ) -> tuple[str, ...]:
        issues: list[str] = []
        if (
            not isinstance(self.recipe_id, str)
            or not self.recipe_id
            or self.recipe_id.strip() != self.recipe_id
        ):
            issues.append("recipe_action_binding_recipe_id_invalid")
        if (
            not isinstance(self.action_id, str)
            or not self.action_id
            or self.action_id.strip() != self.action_id
        ):
            issues.append("recipe_action_binding_action_id_invalid")
        if not _is_sha256(self.action_semantics_hash):
            issues.append("recipe_action_binding_semantics_hash_invalid")
        matches = tuple(
            action
            for action in program.action_graph
            if action.id == self.action_id
        )
        if len(matches) != 1:
            issues.append("recipe_action_binding_action_missing")
        elif (
            action_node_semantics_hash(matches[0])
            != self.action_semantics_hash
        ):
            issues.append("recipe_action_binding_semantics_mismatch")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "action_id": self.action_id,
            "action_semantics_hash": self.action_semantics_hash,
        }


@dataclass(frozen=True)
class CompiledTreatment:
    disposition: TreatmentDisposition
    program: HypothesisProgram | None = None
    recipe_ids: tuple[str, ...] = ()
    evaluator_artifact_hash: str = ""
    recipe_action_bindings: tuple[RecipeActionBinding, ...] = ()

    @property
    def treatment_behavior_hash(self) -> str:
        if self.disposition is TreatmentDisposition.ACTIVE_PROGRAM:
            if self.program is None:
                raise PermissionError("active treatment has no program")
            return hypothesis_program_behavior_hash(self.program)
        if self.disposition is TreatmentDisposition.EVALUATOR_ARTIFACT:
            if not _is_sha256(self.evaluator_artifact_hash):
                raise PermissionError("evaluator artifact hash is invalid")
            return self.evaluator_artifact_hash
        return stable_hash(
            {"treatment_disposition": TreatmentDisposition.PRESERVE_BASELINE.value}
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.disposition is TreatmentDisposition.ACTIVE_PROGRAM:
            if self.program is None:
                issues.append("compiled_treatment_program_missing")
            else:
                issues.extend(
                    f"compiled_treatment_{issue}"
                    for issue in self.program.validate()
                )
            if not _unique_nonempty(self.recipe_ids):
                issues.append("compiled_treatment_recipe_ids_invalid")
            binding_recipe_ids = tuple(
                binding.recipe_id
                for binding in self.recipe_action_bindings
            )
            binding_action_ids = tuple(
                binding.action_id
                for binding in self.recipe_action_bindings
            )
            if (
                not self.recipe_action_bindings
                or len(set(binding_recipe_ids))
                != len(binding_recipe_ids)
                or len(set(binding_action_ids))
                != len(binding_action_ids)
                or set(binding_recipe_ids) != set(self.recipe_ids)
            ):
                issues.append(
                    "compiled_treatment_recipe_action_coverage_invalid"
                )
            elif self.program is not None:
                issues.extend(
                    issue
                    for binding in self.recipe_action_bindings
                    for issue in binding.validate(program=self.program)
                )
            if self.evaluator_artifact_hash:
                issues.append("compiled_treatment_unexpected_evaluator_hash")
        elif self.disposition is TreatmentDisposition.PRESERVE_BASELINE:
            if (
                self.program is not None
                or self.recipe_ids
                or self.recipe_action_bindings
            ):
                issues.append("compiled_noop_contains_active_treatment")
            if self.evaluator_artifact_hash:
                issues.append("compiled_noop_contains_evaluator_hash")
        elif self.disposition is TreatmentDisposition.EVALUATOR_ARTIFACT:
            if (
                self.program is not None
                or self.recipe_ids
                or self.recipe_action_bindings
            ):
                issues.append("compiled_evaluator_contains_program")
            if not _is_sha256(self.evaluator_artifact_hash):
                issues.append("compiled_evaluator_artifact_hash_invalid")
        else:
            issues.append("compiled_treatment_disposition_invalid")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "recipe_ids": list(self.recipe_ids),
            "recipe_action_bindings": [
                {
                    **binding.safe_payload(),
                    "binding_hash": binding.binding_hash,
                }
                for binding in sorted(
                    self.recipe_action_bindings,
                    key=lambda row: (row.recipe_id, row.action_id),
                )
            ],
            "recipe_action_binding_hashes": list(
                self.recipe_action_binding_hashes
            ),
            "treatment_behavior_hash": self.treatment_behavior_hash,
            "program_kind": (
                self.program.kind.value if self.program is not None else None
            ),
            "evaluator_artifact_hash": (
                self.evaluator_artifact_hash or None
            ),
            "program_payload_persisted": False,
        }

    @property
    def recipe_action_binding_hashes(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                binding.binding_hash
                for binding in self.recipe_action_bindings
            )
        )


def _compiler_target_for_treatment(
    treatment: CompiledTreatment,
) -> CompilerTarget:
    if treatment.disposition is TreatmentDisposition.PRESERVE_BASELINE:
        return CompilerTarget.NO_DIRECT_TREATMENT
    if treatment.disposition is TreatmentDisposition.EVALUATOR_ARTIFACT:
        return CompilerTarget.EVALUATOR_ARTIFACT
    assert treatment.program is not None
    if treatment.program.kind is HypothesisKind.TASK:
        return CompilerTarget.TASK_PROGRAM
    if treatment.program.kind is HypothesisKind.POLICY:
        return CompilerTarget.POLICY_PROGRAM
    if treatment.program.kind is HypothesisKind.EVALUATOR:
        return CompilerTarget.EVALUATOR_ARTIFACT
    raise PermissionError("compiled program kind is unsupported")


@dataclass(frozen=True)
class CompilationReceipt:
    receipt_id: str
    ontology_hash: str
    template_hashes: tuple[str, ...]
    claim_hash: str
    probe_receipt_hashes: tuple[str, ...]
    compiler_id: str
    compiler_version: str
    compiler_implementation_hash: str
    primary_metric: str
    compiler_trust_anchor_hash: str
    compiler_target: CompilerTarget
    treatment_disposition: TreatmentDisposition
    recipe_ids: tuple[str, ...]
    recipe_action_binding_hashes: tuple[str, ...]
    treatment_behavior_hash: str
    formation_split: SplitName = SplitName.TRAIN

    @property
    def receipt_hash(self) -> str:
        return stable_hash(self.safe_payload())

    @property
    def expected_receipt_id(self) -> str:
        return "compilation." + stable_hash(self.binding_payload())[:24]

    def binding_payload(self) -> dict[str, Any]:
        return {
            "sidecar_version": META_ASSUMPTION_SIDECAR_VERSION,
            "ontology_hash": self.ontology_hash,
            "template_hashes": list(self.template_hashes),
            "claim_hash": self.claim_hash,
            "probe_receipt_hashes": list(self.probe_receipt_hashes),
            "compiler_id": self.compiler_id,
            "compiler_version": self.compiler_version,
            "compiler_implementation_hash": (
                self.compiler_implementation_hash
            ),
            "primary_metric": self.primary_metric,
            "compiler_trust_anchor_hash": (
                self.compiler_trust_anchor_hash
            ),
            "compiler_target": self.compiler_target.value,
            "treatment_disposition": self.treatment_disposition.value,
            "recipe_ids": list(self.recipe_ids),
            "recipe_action_binding_hashes": list(
                self.recipe_action_binding_hashes
            ),
            "treatment_behavior_hash": self.treatment_behavior_hash,
            "formation_split": self.formation_split.value,
        }

    def validate(
        self,
        *,
        ontology: UniversalAssumptionOntology,
        claim: HypothesisClaim,
        probes: Sequence[ProbeReceipt],
        probe_evidence_bundles: Sequence[ProbeEvidenceBundle],
        probe_verifier_registry: ProbeVerifierRegistry,
        treatment: CompiledTreatment,
        trust_anchor: CompilerTrustAnchor,
    ) -> tuple[str, ...]:
        issues: list[str] = []
        if not _valid_identifier(self.receipt_id):
            issues.append("compilation_receipt_id_invalid")
        elif self.receipt_id != self.expected_receipt_id:
            issues.append("compilation_receipt_id_binding_mismatch")
        anchor_issues = trust_anchor.validate()
        issues.extend(anchor_issues)
        if not anchor_issues:
            if self.compiler_id != trust_anchor.compiler_id:
                issues.append("compilation_receipt_compiler_id_mismatch")
            if self.compiler_version != trust_anchor.compiler_version:
                issues.append(
                    "compilation_receipt_compiler_version_mismatch"
                )
            if (
                self.compiler_implementation_hash
                != trust_anchor.implementation_hash
            ):
                issues.append("compilation_receipt_compiler_hash_mismatch")
            if self.primary_metric != trust_anchor.primary_metric:
                issues.append(
                    "compilation_receipt_trusted_metric_mismatch"
                )
            if (
                self.compiler_trust_anchor_hash
                != trust_anchor.anchor_hash
            ):
                issues.append(
                    "compilation_receipt_trust_anchor_hash_mismatch"
                )
        if self.ontology_hash != ontology.ontology_hash:
            issues.append("compilation_receipt_ontology_mismatch")
        expected_templates = tuple(
            sorted(
                ontology.require_template(template_id).template_hash
                for template_id in claim.template_ids
            )
        )
        if self.template_hashes != expected_templates:
            issues.append("compilation_receipt_template_hashes_mismatch")
        if self.claim_hash != claim.claim_hash:
            issues.append("compilation_receipt_claim_mismatch")
        expected_probes = tuple(
            sorted(probe.receipt_hash for probe in probes)
        )
        if self.probe_receipt_hashes != expected_probes:
            issues.append("compilation_receipt_probe_hashes_mismatch")
        if not _valid_identifier(self.compiler_id):
            issues.append("compilation_receipt_compiler_id_invalid")
        if not _valid_identifier(self.compiler_version):
            issues.append("compilation_receipt_compiler_version_invalid")
        if not _is_sha256(self.compiler_implementation_hash):
            issues.append("compilation_receipt_compiler_hash_invalid")
        if not self.primary_metric.strip():
            issues.append("compilation_receipt_primary_metric_missing")
        treatment_issues = treatment.validate()
        issues.extend(treatment_issues)
        expected_target = (
            _compiler_target_for_treatment(treatment)
            if not treatment_issues
            else None
        )
        if expected_target is not None and self.compiler_target is not expected_target:
            issues.append("compilation_receipt_target_mismatch")
        if self.treatment_disposition is not treatment.disposition:
            issues.append("compilation_receipt_disposition_mismatch")
        if self.recipe_ids != treatment.recipe_ids:
            issues.append("compilation_receipt_recipe_ids_mismatch")
        if (
            self.recipe_action_binding_hashes
            != treatment.recipe_action_binding_hashes
        ):
            issues.append(
                "compilation_receipt_recipe_action_bindings_mismatch"
            )
        if (
            not treatment_issues
            and self.treatment_behavior_hash
            != treatment.treatment_behavior_hash
        ):
            issues.append("compilation_receipt_behavior_hash_mismatch")
        if (
            treatment.program is not None
            and treatment.program.expected_effect.metric
            != self.primary_metric
        ):
            issues.append("compilation_receipt_primary_metric_mismatch")
        if expected_target is not None and not any(
            expected_target
            in ontology.require_template(template_id).compiler_targets
            for template_id in claim.template_ids
        ):
            issues.append("compilation_receipt_target_not_authorized")
        if self.formation_split is not SplitName.TRAIN:
            issues.append("compilation_receipt_not_train_formed")
        evidence_by_template: dict[
            str, list[ProbeEvidenceBundle]
        ] = {}
        for evidence in probe_evidence_bundles:
            evidence_by_template.setdefault(
                evidence.template_id, []
            ).append(evidence)
        if (
            len(probe_evidence_bundles) != len(probes)
            or any(
                len(rows) != 1
                for rows in evidence_by_template.values()
            )
        ):
            issues.append(
                "compilation_receipt_probe_evidence_coverage_mismatch"
            )
        for probe in probes:
            matching_evidence = evidence_by_template.get(
                probe.template_id, []
            )
            if len(matching_evidence) != 1:
                issues.append(
                    "compilation_receipt_probe_evidence_coverage_mismatch"
                )
            else:
                issues.extend(
                    probe_verifier_registry.verify_receipt(
                        probe,
                        ontology=ontology,
                        claim=claim,
                        evidence=matching_evidence[0],
                    )
                )
            if probe.falsified:
                issues.append("compilation_receipt_used_falsified_probe")
            if (
                treatment.disposition
                is not TreatmentDisposition.PRESERVE_BASELINE
                and probe.disposition is not ProbeDisposition.SUPPORTED
            ):
                issues.append(
                    "compilation_receipt_active_treatment_without_support"
                )
        if set(probe.template_id for probe in probes) != set(
            claim.template_ids
        ):
            issues.append("compilation_receipt_probe_coverage_mismatch")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            **self.binding_payload(),
        }


class HypothesisSpaceCompiler(Protocol):
    compiler_id: str
    compiler_version: str
    implementation_hash: str
    primary_metric: str

    def compile(
        self,
        *,
        ontology: UniversalAssumptionOntology,
        claim: HypothesisClaim,
        probes: Sequence[ProbeReceipt],
    ) -> CompiledTreatment: ...


class HypothesisSpaceCompilerRegistry:
    """Closed, harness-owned claim-to-treatment compiler registry."""

    def __init__(
        self,
        *,
        probe_verifier_registry: ProbeVerifierRegistry,
    ) -> None:
        self._compilers: dict[tuple[str, str], HypothesisSpaceCompiler] = {}
        self._trust_anchors: dict[
            tuple[str, str], CompilerTrustAnchor
        ] = {}
        self._probe_verifier_registry = probe_verifier_registry

    def register(
        self,
        compiler: HypothesisSpaceCompiler,
        *,
        trust_anchor: CompilerTrustAnchor,
    ) -> None:
        observed_anchor = CompilerTrustAnchor(
            compiler_id=compiler.compiler_id,
            compiler_version=compiler.compiler_version,
            implementation_hash=compiler.implementation_hash,
            primary_metric=compiler.primary_metric,
        )
        if observed_anchor.validate():
            raise PermissionError("hypothesis-space compiler identity is invalid")
        anchor = trust_anchor
        if anchor.validate():
            raise PermissionError("compiler trust anchor is invalid")
        if observed_anchor != anchor:
            raise PermissionError(
                "hypothesis-space compiler does not match trust anchor"
            )
        key = (anchor.compiler_id, anchor.compiler_version)
        existing = self._compilers.get(key)
        existing_anchor = self._trust_anchors.get(key)
        if existing is not None and (
            existing_anchor != anchor
            or (
                existing is not compiler
                and (
                    existing.implementation_hash
                    != compiler.implementation_hash
                    or existing.primary_metric != compiler.primary_metric
                )
            )
        ):
            raise PermissionError(
                "hypothesis-space compiler registry conflict"
            )
        self._compilers[key] = compiler
        self._trust_anchors[key] = anchor

    def require_trust_anchor(
        self, compiler_id: str, compiler_version: str
    ) -> CompilerTrustAnchor:
        anchor = self._trust_anchors.get((compiler_id, compiler_version))
        if anchor is None:
            raise PermissionError(
                "hypothesis-space compiler is not registered"
            )
        return anchor

    def compile(
        self,
        *,
        compiler_id: str,
        compiler_version: str,
        ontology: UniversalAssumptionOntology,
        claim: HypothesisClaim,
        probes: Sequence[ProbeReceipt],
        probe_evidence_bundles: Sequence[ProbeEvidenceBundle],
    ) -> tuple[CompiledTreatment, CompilationReceipt]:
        if ontology.validate():
            raise PermissionError("assumption ontology is invalid")
        claim_issues = claim.validate(ontology)
        if claim_issues:
            raise PermissionError(
                f"hypothesis claim is invalid: {list(claim_issues)}"
            )
        key = (compiler_id, compiler_version)
        compiler = self._compilers.get(key)
        trust_anchor = self._trust_anchors.get(key)
        if compiler is None or trust_anchor is None:
            raise PermissionError("hypothesis-space compiler is not registered")
        observed_anchor = CompilerTrustAnchor(
            compiler_id=compiler.compiler_id,
            compiler_version=compiler.compiler_version,
            implementation_hash=compiler.implementation_hash,
            primary_metric=compiler.primary_metric,
        )
        if (
            observed_anchor.validate()
            or observed_anchor != trust_anchor
            or trust_anchor.validate()
        ):
            raise PermissionError(
                "registered compiler no longer matches trust anchor"
            )
        canonical_probes = tuple(
            sorted(probes, key=lambda row: row.receipt_hash)
        )
        if not canonical_probes:
            raise PermissionError("claim compilation requires probe receipts")
        canonical_evidence = tuple(
            sorted(
                probe_evidence_bundles,
                key=lambda row: row.evidence_bundle_hash,
            )
        )
        evidence_by_template: dict[
            str, list[ProbeEvidenceBundle]
        ] = {}
        for evidence in canonical_evidence:
            evidence_by_template.setdefault(
                evidence.template_id, []
            ).append(evidence)
        probe_issues: tuple[str, ...] = ()
        if (
            len(canonical_evidence) != len(canonical_probes)
            or any(
                len(rows) != 1
                for rows in evidence_by_template.values()
            )
        ):
            probe_issues = (
                "claim_compilation_probe_evidence_coverage_invalid",
            )
        probe_issues = (
            *probe_issues,
            *(
                issue
                for probe in canonical_probes
                for evidence in evidence_by_template.get(
                    probe.template_id, ()
                )
                for issue in self._probe_verifier_registry.verify_receipt(
                    probe,
                    ontology=ontology,
                    claim=claim,
                    evidence=evidence,
                )
            ),
        )
        if any(
            len(evidence_by_template.get(probe.template_id, ())) != 1
            for probe in canonical_probes
        ):
            probe_issues = (
                *probe_issues,
                "claim_compilation_probe_evidence_coverage_invalid",
            )
        if probe_issues:
            raise PermissionError(
                f"claim probe receipt is invalid: {sorted(set(probe_issues))}"
            )
        if any(probe.falsified for probe in canonical_probes):
            raise PermissionError("falsified claim cannot be compiled")
        if set(probe.template_id for probe in canonical_probes) != set(
            claim.template_ids
        ):
            raise PermissionError(
                "claim compilation probe coverage is incomplete"
            )
        treatment = compiler.compile(
            ontology=ontology,
            claim=claim,
            probes=canonical_probes,
        )
        treatment_issues = treatment.validate()
        if treatment_issues:
            raise PermissionError(
                f"compiled treatment is invalid: {list(treatment_issues)}"
            )
        if (
            treatment.disposition
            is not TreatmentDisposition.PRESERVE_BASELINE
            and any(
                probe.disposition is not ProbeDisposition.SUPPORTED
                for probe in canonical_probes
            )
        ):
            raise PermissionError(
                "active or evaluator treatment requires supported probes"
            )
        target = _compiler_target_for_treatment(treatment)
        receipt_body = {
            "ontology_hash": ontology.ontology_hash,
            "template_hashes": tuple(
                sorted(
                    ontology.require_template(template_id).template_hash
                    for template_id in claim.template_ids
                )
            ),
            "claim_hash": claim.claim_hash,
            "probe_receipt_hashes": tuple(
                probe.receipt_hash for probe in canonical_probes
            ),
            "compiler_id": trust_anchor.compiler_id,
            "compiler_version": trust_anchor.compiler_version,
            "compiler_implementation_hash": (
                trust_anchor.implementation_hash
            ),
            "primary_metric": trust_anchor.primary_metric,
            "compiler_trust_anchor_hash": trust_anchor.anchor_hash,
            "compiler_target": target,
            "treatment_disposition": treatment.disposition,
            "recipe_ids": treatment.recipe_ids,
            "recipe_action_binding_hashes": (
                treatment.recipe_action_binding_hashes
            ),
            "treatment_behavior_hash": treatment.treatment_behavior_hash,
        }
        receipt = CompilationReceipt(
            receipt_id="compilation.pending",
            **receipt_body,
        )
        receipt = replace(receipt, receipt_id=receipt.expected_receipt_id)
        issues = receipt.validate(
            ontology=ontology,
            claim=claim,
            probes=canonical_probes,
            probe_evidence_bundles=canonical_evidence,
            probe_verifier_registry=self._probe_verifier_registry,
            treatment=treatment,
            trust_anchor=trust_anchor,
        )
        if issues:
            raise PermissionError(
                f"claim-treatment binding is invalid: {list(issues)}"
            )
        return treatment, receipt


def verify_compilation_receipt(
    receipt: CompilationReceipt,
    *,
    ontology: UniversalAssumptionOntology,
    claim: HypothesisClaim,
    probes: Sequence[ProbeReceipt],
    probe_evidence_bundles: Sequence[ProbeEvidenceBundle],
    probe_verifier_registry: ProbeVerifierRegistry,
    treatment: CompiledTreatment,
    trust_anchor: CompilerTrustAnchor,
) -> None:
    issues = receipt.validate(
        ontology=ontology,
        claim=claim,
        probes=probes,
        probe_evidence_bundles=probe_evidence_bundles,
        probe_verifier_registry=probe_verifier_registry,
        treatment=treatment,
        trust_anchor=trust_anchor,
    )
    if issues:
        raise PermissionError(
            f"claim-treatment binding verification failed: {list(issues)}"
        )
