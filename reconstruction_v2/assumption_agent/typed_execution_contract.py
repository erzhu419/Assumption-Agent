from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping, Sequence

from .models import ResidualExample, SplitName, stable_hash
from .typed_operator_grammar import (
    FamilyCapabilityGraph,
    OperatorKind,
    TypedRecipe,
    WorkflowKind,
)


TYPED_EXECUTION_CONTRACT_VERSION = (
    "train_supported_closed_task_execution_contract_v1"
)
MIN_INDEPENDENT_TRAIN_SUPPORT = 2
MAX_SUPPORTS_PER_INVARIANT = 16
MAX_ACTION_STARTS = 100
MAX_MUTATIONS = 32
MAX_COMPLETION_CHECKS = 32
MAX_SEARCH_CANDIDATES = 256

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_RECIPE_ID = re.compile(r"^recipe_[0-9a-f]{20}$")


class RuntimeRole(str, Enum):
    """Closed task-local roles; no path or model-authored role is accepted."""

    PRIMARY_ARTIFACT = "primary_artifact"
    SOURCE_COLLECTION = "source_collection"
    WORKING_STATE = "working_state"
    DECLARED_OUTPUT = "declared_output"
    FINAL_MATERIALIZED_OUTPUT = "final_materialized_output"
    EFFECT_RECEIPT = "effect_receipt"


class InvariantKind(str, Enum):
    PRIMARY_ARTIFACT_READ_BEFORE_MUTATION = (
        "primary_artifact_read_before_mutation"
    )
    TASK_DELTA_ONLY = "task_delta_only"
    PRESERVE_UNTARGETED_CONTENT = "preserve_untargeted_content"
    EACH_SOURCE_ITEM_ASSIGNED_EXACTLY_ONCE = (
        "each_source_item_assigned_exactly_once"
    )
    SOURCE_COLLECTION_EMPTY_AFTER_SUCCESS = (
        "source_collection_empty_after_success"
    )
    INPUT_DERIVATION_PRESERVED = "input_derivation_preserved"
    OBSERVABLE_INTERACTION_POSTCONDITION = (
        "observable_interaction_postcondition"
    )
    FINITE_SEARCH_SPACE_DECLARED = "finite_search_space_declared"
    FINAL_METRICS_FROM_FINAL_OUTPUT = "final_metrics_from_final_output"
    FINAL_OUTPUT_REOPENED = "final_output_reopened"


class CompletionCheckKind(str, Enum):
    REOPEN_FINAL_OUTPUT = "reopen_final_output"
    VERIFY_DECLARED_OUTPUT_EXISTS = "verify_declared_output_exists"
    VERIFY_ALL_INVARIANTS = "verify_all_invariants"
    RECOMPUTE_FROM_FINAL_OUTPUT = "recompute_from_final_output"
    VERIFY_SOURCE_COLLECTION_EMPTY = "verify_source_collection_empty"
    REPLAY_OBSERVABLE_INTERACTION = "replay_observable_interaction"
    VERIFY_SEARCH_EVALUATION_COUNT = "verify_search_evaluation_count"
    EMIT_EFFECT_RECEIPT = "emit_effect_receipt"


class CompletionPhaseKind(str, Enum):
    APPLY_REGISTERED_MUTATION = "apply_registered_mutation"
    REOPEN_MATERIALIZED_OUTPUT = "reopen_materialized_output"
    CHECK_CLOSED_INVARIANTS = "check_closed_invariants"
    BOUNDED_REPAIR = "bounded_repair"
    FINALIZE_EFFECT_RECEIPT = "finalize_effect_receipt"


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True)
class TrainSupportRef:
    """Content-free binding to an independent TRAIN policy-off failure."""

    family_hash: str
    transition_id_hash: str
    task_id_hash: str
    evidence_hash: str

    @classmethod
    def from_residual(cls, residual: ResidualExample) -> TrainSupportRef:
        issues = residual.validate()
        if issues:
            raise PermissionError(
                f"execution-contract support is not admissible TRAIN data: {issues}"
            )
        if residual.split is not SplitName.TRAIN:
            raise PermissionError("execution-contract support must be TRAIN")
        if residual.baseline_success is not False:
            raise PermissionError(
                "execution-contract support must be a policy-off failure"
            )
        family_hash = stable_hash({"family": residual.family})
        transition_id_hash = stable_hash(
            {"transition_id": residual.transition_id}
        )
        task_id_hash = stable_hash({"task_id": residual.task_id})
        evidence_hash = stable_hash(
            {
                "family_hash": family_hash,
                "transition_id_hash": transition_id_hash,
                "task_id_hash": task_id_hash,
                "split": SplitName.TRAIN.value,
                "policy_off_failure": True,
            }
        )
        return cls(
            family_hash=family_hash,
            transition_id_hash=transition_id_hash,
            task_id_hash=task_id_hash,
            evidence_hash=evidence_hash,
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for value, issue in (
            (self.family_hash, "support_family_hash_invalid"),
            (self.transition_id_hash, "support_transition_hash_invalid"),
            (self.task_id_hash, "support_task_hash_invalid"),
            (self.evidence_hash, "support_evidence_hash_invalid"),
        ):
            if not _is_sha256(value):
                issues.append(issue)
        expected = stable_hash(
            {
                "family_hash": self.family_hash,
                "transition_id_hash": self.transition_id_hash,
                "task_id_hash": self.task_id_hash,
                "split": SplitName.TRAIN.value,
                "policy_off_failure": True,
            }
        )
        if self.evidence_hash != expected:
            issues.append("support_evidence_hash_mismatch")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "family_hash": self.family_hash,
            "transition_id_hash": self.transition_id_hash,
            "task_id_hash": self.task_id_hash,
            "evidence_hash": self.evidence_hash,
            "raw_content_persisted": False,
        }


@dataclass(frozen=True)
class ExecutableInvariantSpec:
    kind: InvariantKind
    input_role: RuntimeRole
    output_role: RuntimeRole
    operation: OperatorKind
    supports: tuple[TrainSupportRef, ...]

    @property
    def invariant_id(self) -> str:
        return "invariant_" + stable_hash(self._identity_payload())[:20]

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value
            if isinstance(self.kind, InvariantKind)
            else None,
            "input_role": self.input_role.value
            if isinstance(self.input_role, RuntimeRole)
            else None,
            "output_role": self.output_role.value
            if isinstance(self.output_role, RuntimeRole)
            else None,
            "operation": self.operation.value
            if isinstance(self.operation, OperatorKind)
            else None,
            "support_evidence_hashes": [
                row.evidence_hash for row in self.supports
            ],
        }

    def validate(
        self,
        *,
        target_family_hash: str,
        recipe_operator_kinds: frozenset[OperatorKind],
    ) -> tuple[str, ...]:
        issues: list[str] = []
        if not isinstance(self.kind, InvariantKind):
            issues.append("invariant_kind_not_closed")
        if not isinstance(self.input_role, RuntimeRole):
            issues.append("invariant_input_role_not_closed")
        if not isinstance(self.output_role, RuntimeRole):
            issues.append("invariant_output_role_not_closed")
        if not isinstance(self.operation, OperatorKind):
            issues.append("invariant_operation_not_closed")
        elif self.operation not in recipe_operator_kinds:
            issues.append("invariant_operation_outside_recipe")
        if not self.supports:
            issues.append("invariant_support_empty")
        if len(self.supports) > MAX_SUPPORTS_PER_INVARIANT:
            issues.append("invariant_support_exceeds_limit")
        issues.extend(issue for row in self.supports for issue in row.validate())
        canonical = tuple(
            sorted(
                self.supports,
                key=lambda row: (
                    row.transition_id_hash,
                    row.task_id_hash,
                    row.evidence_hash,
                ),
            )
        )
        if canonical != self.supports:
            issues.append("invariant_support_order_not_canonical")
        if any(row.family_hash != target_family_hash for row in self.supports):
            issues.append("invariant_support_family_mismatch")
        transition_hashes = {row.transition_id_hash for row in self.supports}
        task_hashes = {row.task_id_hash for row in self.supports}
        evidence_hashes = {row.evidence_hash for row in self.supports}
        if min(
            len(transition_hashes),
            len(task_hashes),
            len(evidence_hashes),
        ) < MIN_INDEPENDENT_TRAIN_SUPPORT:
            issues.append("invariant_independent_support_insufficient")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "invariant_id": self.invariant_id,
            "kind": self.kind.value
            if isinstance(self.kind, InvariantKind)
            else None,
            "input_role": self.input_role.value
            if isinstance(self.input_role, RuntimeRole)
            else None,
            "output_role": self.output_role.value
            if isinstance(self.output_role, RuntimeRole)
            else None,
            "operation": self.operation.value
            if isinstance(self.operation, OperatorKind)
            else None,
            "supports": [row.safe_payload() for row in self.supports],
            "model_authored_fields": [],
        }


@dataclass(frozen=True)
class CompletionContractSpec:
    final_output_role: RuntimeRole
    self_evaluation_source_role: RuntimeRole
    effect_receipt_role: RuntimeRole
    phase_order: tuple[CompletionPhaseKind, ...]
    checks: tuple[CompletionCheckKind, ...]

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.final_output_role is not RuntimeRole.FINAL_MATERIALIZED_OUTPUT:
            issues.append("completion_final_output_role_invalid")
        if (
            self.self_evaluation_source_role
            is not RuntimeRole.FINAL_MATERIALIZED_OUTPUT
        ):
            issues.append("completion_self_evaluation_not_single_source")
        if self.effect_receipt_role is not RuntimeRole.EFFECT_RECEIPT:
            issues.append("completion_effect_receipt_role_invalid")
        expected_phase_order = (
            CompletionPhaseKind.APPLY_REGISTERED_MUTATION,
            CompletionPhaseKind.REOPEN_MATERIALIZED_OUTPUT,
            CompletionPhaseKind.CHECK_CLOSED_INVARIANTS,
            CompletionPhaseKind.BOUNDED_REPAIR,
            CompletionPhaseKind.FINALIZE_EFFECT_RECEIPT,
        )
        if self.phase_order != expected_phase_order:
            issues.append("completion_phase_order_invalid")
        if any(
            not isinstance(row, CompletionPhaseKind)
            for row in self.phase_order
        ):
            issues.append("completion_phase_not_closed")
        if not self.checks:
            issues.append("completion_checks_empty")
        if len(self.checks) != len(set(self.checks)):
            issues.append("completion_checks_duplicate")
        if tuple(sorted(self.checks, key=lambda row: row.value)) != self.checks:
            issues.append("completion_checks_order_not_canonical")
        if any(not isinstance(row, CompletionCheckKind) for row in self.checks):
            issues.append("completion_check_not_closed")
        required = {
            CompletionCheckKind.REOPEN_FINAL_OUTPUT,
            CompletionCheckKind.VERIFY_DECLARED_OUTPUT_EXISTS,
            CompletionCheckKind.VERIFY_ALL_INVARIANTS,
            CompletionCheckKind.RECOMPUTE_FROM_FINAL_OUTPUT,
            CompletionCheckKind.EMIT_EFFECT_RECEIPT,
        }
        if not required.issubset(set(self.checks)):
            issues.append("completion_required_check_missing")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "final_output_role": self.final_output_role.value,
            "self_evaluation_source_role": (
                self.self_evaluation_source_role.value
            ),
            "effect_receipt_role": self.effect_receipt_role.value,
            "phase_order": [row.value for row in self.phase_order],
            "checks": [row.value for row in self.checks],
            "single_source_self_evaluation_required": True,
            "effect_receipt_required": True,
            "model_authored_fields": [],
        }


@dataclass(frozen=True)
class FiniteSearchSpaceSpec:
    candidate_hashes: tuple[str, ...] = ()
    evaluation_limit: int = 0

    @property
    def search_space_hash(self) -> str:
        return stable_hash(
            {"ordered_candidate_hashes": list(self.candidate_hashes)}
        )

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if len(self.candidate_hashes) > MAX_SEARCH_CANDIDATES:
            issues.append("search_candidate_count_exceeds_limit")
        if any(not _is_sha256(value) for value in self.candidate_hashes):
            issues.append("search_candidate_hash_invalid")
        if tuple(sorted(set(self.candidate_hashes))) != self.candidate_hashes:
            issues.append("search_candidate_set_not_canonical")
        if (
            not isinstance(self.evaluation_limit, int)
            or isinstance(self.evaluation_limit, bool)
            or self.evaluation_limit < 0
            or self.evaluation_limit > len(self.candidate_hashes)
        ):
            issues.append("search_evaluation_limit_invalid")
        if bool(self.candidate_hashes) != bool(self.evaluation_limit):
            issues.append("search_space_zero_state_mismatch")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "candidate_hashes": list(self.candidate_hashes),
            "candidate_count": len(self.candidate_hashes),
            "evaluation_limit": self.evaluation_limit,
            "search_space_hash": self.search_space_hash,
            "raw_candidate_values_persisted": False,
        }


@dataclass(frozen=True)
class ResourceBudgetSpec:
    max_action_starts: int
    max_mutations: int
    max_repair_attempts: int
    max_completion_checks: int
    max_search_evaluations: int

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        for value, upper, issue in (
            (
                self.max_action_starts,
                MAX_ACTION_STARTS,
                "resource_action_start_limit_invalid",
            ),
            (
                self.max_mutations,
                MAX_MUTATIONS,
                "resource_mutation_limit_invalid",
            ),
            (
                self.max_completion_checks,
                MAX_COMPLETION_CHECKS,
                "resource_completion_check_limit_invalid",
            ),
        ):
            if not _is_positive_int(value) or value > upper:
                issues.append(issue)
        if (
            not isinstance(self.max_search_evaluations, int)
            or isinstance(self.max_search_evaluations, bool)
            or self.max_search_evaluations < 0
            or self.max_search_evaluations > MAX_SEARCH_CANDIDATES
        ):
            issues.append("resource_search_evaluation_limit_invalid")
        if (
            not isinstance(self.max_repair_attempts, int)
            or isinstance(self.max_repair_attempts, bool)
            or self.max_repair_attempts < 0
            or self.max_repair_attempts >= self.max_mutations
        ):
            issues.append("resource_repair_attempt_limit_invalid")
        return tuple(sorted(set(issues)))

    def safe_payload(self) -> dict[str, Any]:
        return {
            "max_action_starts": self.max_action_starts,
            "max_mutations": self.max_mutations,
            "max_repair_attempts": self.max_repair_attempts,
            "max_completion_checks": self.max_completion_checks,
            "max_search_evaluations": self.max_search_evaluations,
            "action_span_compute_is_budgeted": True,
            "runtime_receipt_required": True,
        }


@dataclass(frozen=True)
class TypedExecutionContract:
    contract_version: str
    graph_hash: str
    target_family_hash: str
    recipe_id: str
    workflow: WorkflowKind
    invariants: tuple[ExecutableInvariantSpec, ...]
    completion: CompletionContractSpec
    search_space: FiniteSearchSpaceSpec
    resources: ResourceBudgetSpec

    @property
    def contract_id(self) -> str:
        return "execution_contract_" + stable_hash(
            self.safe_payload(include_hash=False, include_id=False)
        )[:20]

    @property
    def contract_hash(self) -> str:
        return stable_hash(self.safe_payload(include_hash=False))

    def validate(self, graph: FamilyCapabilityGraph) -> tuple[str, ...]:
        issues: list[str] = []
        graph_issues = graph.validate()
        if graph_issues:
            issues.append("execution_contract_graph_invalid")
        if self.contract_version != TYPED_EXECUTION_CONTRACT_VERSION:
            issues.append("execution_contract_version_mismatch")
        if not _is_sha256(self.graph_hash) or self.graph_hash != graph.graph_hash:
            issues.append("execution_contract_graph_hash_mismatch")
        if (
            not _is_sha256(self.target_family_hash)
            or self.target_family_hash != graph.target_family_hash
        ):
            issues.append("execution_contract_family_hash_mismatch")
        if not _OPAQUE_RECIPE_ID.fullmatch(self.recipe_id):
            issues.append("execution_contract_recipe_id_not_opaque")
        recipes = {row.recipe_id: row for row in graph.recipes}
        recipe = recipes.get(self.recipe_id)
        if recipe is None:
            issues.append("execution_contract_recipe_missing")
            recipe_operator_kinds: frozenset[OperatorKind] = frozenset()
        else:
            recipe_operator_kinds = frozenset(row.kind for row in recipe.nodes)
            if self.workflow is not recipe.workflow:
                issues.append("execution_contract_workflow_mismatch")
        if not isinstance(self.workflow, WorkflowKind):
            issues.append("execution_contract_workflow_not_closed")
        if not self.invariants:
            issues.append("execution_contract_invariants_empty")
        if tuple(
            sorted(
                self.invariants,
                key=lambda row: (
                    row.kind.value
                    if isinstance(row.kind, InvariantKind)
                    else "",
                    row.operation.value
                    if isinstance(row.operation, OperatorKind)
                    else "",
                ),
            )
        ) != self.invariants:
            issues.append("execution_contract_invariant_order_not_canonical")
        if len({row.kind for row in self.invariants}) != len(self.invariants):
            issues.append("execution_contract_invariant_duplicate")
        issues.extend(
            issue
            for row in self.invariants
            for issue in row.validate(
                target_family_hash=self.target_family_hash,
                recipe_operator_kinds=recipe_operator_kinds,
            )
        )
        issues.extend(self.completion.validate())
        issues.extend(self.search_space.validate())
        issues.extend(self.resources.validate())
        if len(self.completion.checks) > self.resources.max_completion_checks:
            issues.append("completion_checks_exceed_resource_budget")
        if self.resources.max_repair_attempts + 1 > self.resources.max_mutations:
            issues.append("repair_attempts_exceed_mutation_budget")
        if (
            self.search_space.evaluation_limit
            != self.resources.max_search_evaluations
        ):
            issues.append("search_space_resource_budget_mismatch")
        if self.workflow is WorkflowKind.CONFIGURE_AND_RUN:
            if not self.search_space.candidate_hashes:
                issues.append("configure_run_search_space_missing")
            if (
                CompletionCheckKind.VERIFY_SEARCH_EVALUATION_COUNT
                not in self.completion.checks
            ):
                issues.append("configure_run_search_receipt_check_missing")
        elif self.search_space.candidate_hashes:
            issues.append("non_search_workflow_has_search_space")
        expected_kinds = {
            row[0] for row in _workflow_invariant_rows(self.workflow)
        } if isinstance(self.workflow, WorkflowKind) else set()
        if {row.kind for row in self.invariants} != expected_kinds:
            issues.append("execution_contract_workflow_invariants_mismatch")
        return tuple(sorted(set(issues)))

    def safe_payload(
        self,
        *,
        include_hash: bool = True,
        include_id: bool = True,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract_version": self.contract_version,
            "graph_hash": self.graph_hash,
            "target_family_hash": self.target_family_hash,
            "recipe_id": self.recipe_id,
            "workflow": self.workflow.value
            if isinstance(self.workflow, WorkflowKind)
            else None,
            "invariants": [row.safe_payload() for row in self.invariants],
            "completion": self.completion.safe_payload(),
            "search_space": self.search_space.safe_payload(),
            "resources": self.resources.safe_payload(),
            "model_authored_primitive_fields": [],
            "raw_content_persisted": False,
            "runtime_enforcement_claimed": False,
        }
        if include_id:
            payload["contract_id"] = self.contract_id
        if include_hash:
            payload["contract_hash"] = self.contract_hash
        return payload


_WORKFLOW_INVARIANTS: Mapping[
    WorkflowKind,
    tuple[tuple[InvariantKind, RuntimeRole, RuntimeRole, OperatorKind], ...],
] = {
    WorkflowKind.DERIVE_TASK_OUTPUT: (
        (
            InvariantKind.PRIMARY_ARTIFACT_READ_BEFORE_MUTATION,
            RuntimeRole.PRIMARY_ARTIFACT,
            RuntimeRole.WORKING_STATE,
            OperatorKind.READ_REGISTERED_ARTIFACT,
        ),
        (
            InvariantKind.TASK_DELTA_ONLY,
            RuntimeRole.WORKING_STATE,
            RuntimeRole.DECLARED_OUTPUT,
            OperatorKind.DERIVE_TASK_DELTA,
        ),
        (
            InvariantKind.FINAL_OUTPUT_REOPENED,
            RuntimeRole.DECLARED_OUTPUT,
            RuntimeRole.FINAL_MATERIALIZED_OUTPUT,
            OperatorKind.CHECK_TASK_LOCAL_RESULT,
        ),
    ),
    WorkflowKind.TRANSFORM_IN_PLACE: (
        (
            InvariantKind.PRIMARY_ARTIFACT_READ_BEFORE_MUTATION,
            RuntimeRole.PRIMARY_ARTIFACT,
            RuntimeRole.WORKING_STATE,
            OperatorKind.READ_REGISTERED_ARTIFACT,
        ),
        (
            InvariantKind.PRESERVE_UNTARGETED_CONTENT,
            RuntimeRole.WORKING_STATE,
            RuntimeRole.DECLARED_OUTPUT,
            OperatorKind.DERIVE_TASK_DELTA,
        ),
        (
            InvariantKind.FINAL_OUTPUT_REOPENED,
            RuntimeRole.DECLARED_OUTPUT,
            RuntimeRole.FINAL_MATERIALIZED_OUTPUT,
            OperatorKind.CHECK_TASK_LOCAL_RESULT,
        ),
    ),
    WorkflowKind.ORGANIZE_COLLECTION: (
        (
            InvariantKind.EACH_SOURCE_ITEM_ASSIGNED_EXACTLY_ONCE,
            RuntimeRole.SOURCE_COLLECTION,
            RuntimeRole.WORKING_STATE,
            OperatorKind.DERIVE_ORGANIZATION_PLAN,
        ),
        (
            InvariantKind.SOURCE_COLLECTION_EMPTY_AFTER_SUCCESS,
            RuntimeRole.SOURCE_COLLECTION,
            RuntimeRole.FINAL_MATERIALIZED_OUTPUT,
            OperatorKind.CHECK_TASK_LOCAL_RESULT,
        ),
        (
            InvariantKind.FINAL_OUTPUT_REOPENED,
            RuntimeRole.DECLARED_OUTPUT,
            RuntimeRole.FINAL_MATERIALIZED_OUTPUT,
            OperatorKind.CHECK_TASK_LOCAL_RESULT,
        ),
    ),
    WorkflowKind.BUILD_VISUALIZATION: (
        (
            InvariantKind.INPUT_DERIVATION_PRESERVED,
            RuntimeRole.PRIMARY_ARTIFACT,
            RuntimeRole.WORKING_STATE,
            OperatorKind.DERIVE_VISUALIZATION_SPEC,
        ),
        (
            InvariantKind.OBSERVABLE_INTERACTION_POSTCONDITION,
            RuntimeRole.WORKING_STATE,
            RuntimeRole.FINAL_MATERIALIZED_OUTPUT,
            OperatorKind.CHECK_TASK_LOCAL_RESULT,
        ),
        (
            InvariantKind.FINAL_OUTPUT_REOPENED,
            RuntimeRole.DECLARED_OUTPUT,
            RuntimeRole.FINAL_MATERIALIZED_OUTPUT,
            OperatorKind.CHECK_TASK_LOCAL_RESULT,
        ),
    ),
    WorkflowKind.CONFIGURE_AND_RUN: (
        (
            InvariantKind.FINITE_SEARCH_SPACE_DECLARED,
            RuntimeRole.WORKING_STATE,
            RuntimeRole.DECLARED_OUTPUT,
            OperatorKind.DERIVE_TASK_DELTA,
        ),
        (
            InvariantKind.FINAL_METRICS_FROM_FINAL_OUTPUT,
            RuntimeRole.FINAL_MATERIALIZED_OUTPUT,
            RuntimeRole.EFFECT_RECEIPT,
            OperatorKind.INSPECT_GENERATED_OUTPUT,
        ),
        (
            InvariantKind.FINAL_OUTPUT_REOPENED,
            RuntimeRole.DECLARED_OUTPUT,
            RuntimeRole.FINAL_MATERIALIZED_OUTPUT,
            OperatorKind.INSPECT_GENERATED_OUTPUT,
        ),
    ),
}


def _workflow_invariant_rows(
    workflow: WorkflowKind,
) -> tuple[tuple[InvariantKind, RuntimeRole, RuntimeRole, OperatorKind], ...]:
    return _WORKFLOW_INVARIANTS[workflow]


def _completion_checks(workflow: WorkflowKind) -> tuple[CompletionCheckKind, ...]:
    checks = {
        CompletionCheckKind.REOPEN_FINAL_OUTPUT,
        CompletionCheckKind.VERIFY_DECLARED_OUTPUT_EXISTS,
        CompletionCheckKind.VERIFY_ALL_INVARIANTS,
        CompletionCheckKind.RECOMPUTE_FROM_FINAL_OUTPUT,
        CompletionCheckKind.EMIT_EFFECT_RECEIPT,
    }
    if workflow is WorkflowKind.ORGANIZE_COLLECTION:
        checks.add(CompletionCheckKind.VERIFY_SOURCE_COLLECTION_EMPTY)
    elif workflow is WorkflowKind.BUILD_VISUALIZATION:
        checks.add(CompletionCheckKind.REPLAY_OBSERVABLE_INTERACTION)
    elif workflow is WorkflowKind.CONFIGURE_AND_RUN:
        checks.add(CompletionCheckKind.VERIFY_SEARCH_EVALUATION_COUNT)
    return tuple(sorted(checks, key=lambda row: row.value))


def _recipe_for_id(
    graph: FamilyCapabilityGraph,
    recipe_id: str,
) -> TypedRecipe:
    if graph.validate():
        raise PermissionError("execution-contract graph is invalid")
    matches = [row for row in graph.recipes if row.recipe_id == recipe_id]
    if len(matches) != 1:
        raise PermissionError("execution-contract recipe is not registered")
    return matches[0]


def derive_train_execution_contract(
    *,
    graph: FamilyCapabilityGraph,
    recipe_id: str,
    residuals: Sequence[ResidualExample],
    search_candidate_hashes: Sequence[str] = (),
    max_action_starts: int = MAX_ACTION_STARTS,
    max_mutations: int = 8,
) -> TypedExecutionContract:
    """Build a closed contract using only independent TRAIN failure receipts.

    Residual feedback, context, feature values, task text, paths, commands, and
    evaluator literals are deliberately not consulted or persisted.  The v1
    typed graph remains unchanged; this is an opt-in companion object.
    """

    recipe = _recipe_for_id(graph, recipe_id)
    supports_by_identity: dict[tuple[str, str], TrainSupportRef] = {}
    for residual in residuals:
        if residual.split is not SplitName.TRAIN:
            raise PermissionError(
                "execution-contract derivation cannot receive non-TRAIN data"
            )
        if residual.validate():
            raise PermissionError(
                "execution-contract derivation received inadmissible TRAIN data"
            )
        if residual.family != graph.target_family:
            continue
        if residual.baseline_success is not False:
            continue
        support = TrainSupportRef.from_residual(residual)
        supports_by_identity[
            (support.transition_id_hash, support.task_id_hash)
        ] = support
    supports = tuple(
        sorted(
            supports_by_identity.values(),
            key=lambda row: (
                row.transition_id_hash,
                row.task_id_hash,
                row.evidence_hash,
            ),
        )[:MAX_SUPPORTS_PER_INVARIANT]
    )
    if (
        len({row.transition_id_hash for row in supports})
        < MIN_INDEPENDENT_TRAIN_SUPPORT
        or len({row.task_id_hash for row in supports})
        < MIN_INDEPENDENT_TRAIN_SUPPORT
    ):
        raise PermissionError(
            "execution-contract derivation needs two independent same-family "
            "TRAIN failures"
        )

    candidate_hashes = tuple(sorted(set(search_candidate_hashes)))
    if any(not _is_sha256(value) for value in candidate_hashes):
        raise PermissionError("search candidates must be opaque SHA-256 values")
    if recipe.workflow is WorkflowKind.CONFIGURE_AND_RUN:
        if not candidate_hashes:
            raise PermissionError(
                "configure-and-run contract requires a finite search space"
            )
    elif candidate_hashes:
        raise PermissionError(
            "search candidates are only valid for configure-and-run recipes"
        )
    search_space = FiniteSearchSpaceSpec(
        candidate_hashes=candidate_hashes,
        evaluation_limit=len(candidate_hashes),
    )
    checks = _completion_checks(recipe.workflow)
    contract = TypedExecutionContract(
        contract_version=TYPED_EXECUTION_CONTRACT_VERSION,
        graph_hash=graph.graph_hash,
        target_family_hash=graph.target_family_hash,
        recipe_id=recipe.recipe_id,
        workflow=recipe.workflow,
        invariants=tuple(
            sorted(
                (
                    ExecutableInvariantSpec(
                        kind=kind,
                        input_role=input_role,
                        output_role=output_role,
                        operation=operation,
                        supports=supports,
                    )
                    for kind, input_role, output_role, operation in (
                        _workflow_invariant_rows(recipe.workflow)
                    )
                ),
                key=lambda row: (row.kind.value, row.operation.value),
            )
        ),
        completion=CompletionContractSpec(
            final_output_role=RuntimeRole.FINAL_MATERIALIZED_OUTPUT,
            self_evaluation_source_role=(
                RuntimeRole.FINAL_MATERIALIZED_OUTPUT
            ),
            effect_receipt_role=RuntimeRole.EFFECT_RECEIPT,
            phase_order=(
                CompletionPhaseKind.APPLY_REGISTERED_MUTATION,
                CompletionPhaseKind.REOPEN_MATERIALIZED_OUTPUT,
                CompletionPhaseKind.CHECK_CLOSED_INVARIANTS,
                CompletionPhaseKind.BOUNDED_REPAIR,
                CompletionPhaseKind.FINALIZE_EFFECT_RECEIPT,
            ),
            checks=checks,
        ),
        search_space=search_space,
        resources=ResourceBudgetSpec(
            max_action_starts=max_action_starts,
            max_mutations=max_mutations,
            max_repair_attempts=min(2, max_mutations - 1),
            max_completion_checks=max(len(checks), 8),
            max_search_evaluations=search_space.evaluation_limit,
        ),
    )
    issues = contract.validate(graph)
    if issues:
        raise PermissionError(
            f"derived execution contract is invalid: {list(issues)}"
        )
    return contract


def verify_execution_contract_payload(
    contract: TypedExecutionContract,
    payload: Mapping[str, Any],
    *,
    graph: FamilyCapabilityGraph,
) -> None:
    issues = contract.validate(graph)
    if issues:
        raise PermissionError(
            f"execution contract is invalid: {list(issues)}"
        )
    if dict(payload) != contract.safe_payload():
        raise PermissionError("execution contract payload or hash drifted")
