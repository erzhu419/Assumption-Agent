from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .models import (
    ActionNode,
    ExpectedEffect,
    FeaturePredicate,
    HypothesisKind,
    HypothesisProgram,
    ResidualExample,
    TriggerSpec,
    VerifierContract,
    stable_hash,
)


CAUSAL_ACTION_SPAN_EVIDENCE_VERSION = (
    "train_policy_off_chronological_action_span_evidence_v1"
)
TYPED_OPERATOR_GRAMMAR_VERSION = "closed_typed_operator_artifact_graph_v1"
TYPED_RECIPE_SELECTION_VERSION = "opaque_recipe_id_only_selection_v1"
TYPED_OPERATOR_LOWERING_VERSION = "harness_owned_typed_recipe_lowering_v1"
TYPED_PROGRAM_BINDING_VERSION = "harness_owned_typed_program_binding_v2"
TYPED_SELECTION_SNAPSHOT_LEDGER_VERSION = (
    "externally_frozen_typed_selection_snapshot_ledger_v1"
)
TYPED_SELECTION_FREEZE_AUTHORIZATION_VERSION = (
    "formal_result_receipt_bound_typed_selection_freeze_v1"
)
TYPED_SELECTION_EXECUTION_AUTHORIZATION_VERSION = (
    "validated_protocol_lock_bound_typed_selection_execution_v1"
)
TYPED_REPAIR_BRANCH_ID_POLICY_VERSION = (
    "parent_snapshot_scoped_typed_repair_id_v1"
)
TYPED_PROGRAM_STATEMENT = (
    "Apply a harness-owned typed task-local artifact workflow for this "
    "TRAIN-supported family."
)
TYPED_PROGRAM_VERIFIER_CHECKS = (
    "task-local result structure is complete",
)
TYPED_PROGRAM_VERIFIER_ANCHOR = "offline-post-agent-verifier"

_TYPED_SELECTION_EXECUTION_AUTHORIZATION_SEAL = object()

MAX_TRACE_ACTION_SPANS = 100
MAX_REGISTERED_ARTIFACTS_PER_FAMILY = 6

KNOWN_ARTIFACT_EVIDENCE_RELATIONS = frozenset(
    {
        "declared_task_local_path",
        "copied_task_artifact",
        "environment_source_artifact",
        "successful_command_span",
        "failed_command_span_cooccurrence",
        "unknown_command_span_cooccurrence",
        "observed_file_change",
    }
)
POSITIVE_ARTIFACT_EVIDENCE_RELATIONS = frozenset(
    {
        "declared_task_local_path",
        "copied_task_artifact",
        "environment_source_artifact",
        "successful_command_span",
        "observed_file_change",
    }
)


class TypedGraphUnavailableError(RuntimeError):
    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class SpanOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class PrimitiveAssessmentClass(str, Enum):
    OPERATIONALLY_OBSERVED = "operationally_observed"
    RECOVERED_AFTER_FAILURE = "recovered_after_failure"
    FAILURE_COOCCURRENCE_ONLY = "failure_cooccurrence_only"
    INSUFFICIENT = "insufficient"
    DO_NOT_RECOMMEND_EXACT_SIGNATURE = "do_not_recommend_exact_signature"


class ArtifactFormat(str, Enum):
    TABULAR = "tabular"
    STRUCTURED_RECORD = "structured_record"
    OFFICE_DOCUMENT = "office_document"
    PDF = "pdf"
    SCIENTIFIC_ARRAY = "scientific_array"
    CONFIGURATION = "configuration"
    WEB_ASSET = "web_asset"
    IMAGE = "image"
    VIDEO = "video"
    ARCHIVE = "archive"
    DIRECTORY = "directory"
    TEXT = "text"
    GENERIC = "generic"


class CapabilityKind(str, Enum):
    TABULAR_DATA = "tabular_data"
    STRUCTURED_RECORD = "structured_record"
    OFFICE_DOCUMENT = "office_document"
    PDF_DOCUMENT = "pdf_document"
    SCIENTIFIC_DATA = "scientific_data"
    CONFIGURATION = "configuration"
    WEB_ASSET = "web_asset"
    IMAGE_MEDIA = "image_media"
    VIDEO_MEDIA = "video_media"
    ARCHIVE = "archive"
    FILESYSTEM_COLLECTION = "filesystem_collection"
    TEXT_DOCUMENT = "text_document"
    GENERIC_LOCAL_ARTIFACT = "generic_local_artifact"


class WorkflowKind(str, Enum):
    DERIVE_TASK_OUTPUT = "derive_task_output"
    TRANSFORM_IN_PLACE = "transform_in_place"
    ORGANIZE_COLLECTION = "organize_collection"
    BUILD_VISUALIZATION = "build_visualization"
    CONFIGURE_AND_RUN = "configure_and_run"


class OperatorKind(str, Enum):
    READ_REGISTERED_ARTIFACT = "read_registered_artifact"
    PARSE_WITH_REGISTERED_CAPABILITY = "parse_with_registered_capability"
    DERIVE_TASK_DELTA = "derive_task_delta"
    SERIALIZE_WITH_REGISTERED_CAPABILITY = (
        "serialize_with_registered_capability"
    )
    WRITE_BACK_REGISTERED_ARTIFACT = "write_back_registered_artifact"
    WRITE_TASK_DECLARED_OUTPUT = "write_task_declared_output"
    INSPECT_REGISTERED_METADATA = "inspect_registered_metadata"
    DERIVE_ORGANIZATION_PLAN = "derive_organization_plan"
    MOVE_WITHIN_TASK_ROOT = "move_within_task_root"
    DERIVE_VISUALIZATION_SPEC = "derive_visualization_spec"
    RENDER_LOCAL_VISUALIZATION = "render_local_visualization"
    INVOKE_REGISTERED_LOCAL_CAPABILITY = (
        "invoke_registered_local_capability"
    )
    INSPECT_GENERATED_OUTPUT = "inspect_generated_output"
    CHECK_TASK_LOCAL_RESULT = "check_task_local_result"


FORBIDDEN_OPERATOR_KINDS = frozenset(
    {
        "arbitrary_shell_command",
        "free_text_directive",
        "network_fetch",
        "package_install",
        "read_external_verifier",
        "read_validation_or_test_content",
        "read_secret_or_credential",
        "write_outside_task_root",
    }
)
ALLOWED_OPERATOR_KINDS = frozenset(kind.value for kind in OperatorKind)


_LOCAL_PATH = re.compile(r"/root(?:/[A-Za-z0-9._+@%=-]+)+")
_FORBIDDEN_PATH_PARTS = frozenset(
    {
        ".env",
        ".ssh",
        "api_key",
        "auth",
        "authorization",
        "credentials",
        "credential",
        "id_rsa",
        "password",
        "passwd",
        "private_key",
        "secret",
        "secrets",
        "solution",
        "solutions",
        "test",
        "tests",
        "token",
        "verifier",
        "validation",
    }
)


@dataclass(frozen=True)
class PrimitiveRef:
    kind: str
    value: str

    @property
    def primitive_id(self) -> str:
        return "primitive_" + stable_hash(
            {"kind": self.kind, "value": self.value}
        )[:20]

    def safe_payload(self) -> dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "kind": self.kind,
            "value_hash": stable_hash(
                {"kind": self.kind, "value": self.value}
            ),
            "raw_value_persisted": False,
        }


@dataclass(frozen=True)
class ActionSpanEvidence:
    trial_id_hash: str
    family_hash: str
    span_index: int
    start_event_index: int | None
    completion_event_index: int
    exact_command_hash: str
    command_signature_hash: str
    outcome: SpanOutcome
    executable: PrimitiveRef
    flags: tuple[PrimitiveRef, ...]
    artifacts: tuple[PrimitiveRef, ...]
    baseline_success: bool
    trace_complete: bool
    is_last_allowlisted_span: bool = False
    later_exact_success_span_hashes: tuple[str, ...] = ()
    later_same_executable_success_span_hashes: tuple[str, ...] = ()
    later_shared_artifact_success_span_hashes: tuple[str, ...] = ()

    @property
    def span_id(self) -> str:
        return stable_hash(self._base_identity_payload())

    @property
    def span_hash(self) -> str:
        return stable_hash(self._identity_payload())

    @property
    def recovered(self) -> bool:
        return bool(
            self.later_exact_success_span_hashes
            or self.later_same_executable_success_span_hashes
            or self.later_shared_artifact_success_span_hashes
        )

    def _base_identity_payload(self) -> dict[str, Any]:
        return {
            "policy": CAUSAL_ACTION_SPAN_EVIDENCE_VERSION,
            "trial_id_hash": self.trial_id_hash,
            "family_hash": self.family_hash,
            "span_index": self.span_index,
            "start_event_index": self.start_event_index,
            "completion_event_index": self.completion_event_index,
            "exact_command_hash": self.exact_command_hash,
            "command_signature_hash": self.command_signature_hash,
            "outcome": self.outcome.value,
            "executable_id": self.executable.primitive_id,
            "flag_ids": [row.primitive_id for row in self.flags],
            "artifact_ids": [row.primitive_id for row in self.artifacts],
            "baseline_success": self.baseline_success,
            "trace_complete": self.trace_complete,
            "task_effect_attributed": False,
            "primitive_inadmissibility_attributed": False,
            "raw_command_persisted": False,
            "command_output_used": False,
            "model_message_used": False,
        }

    def _identity_payload(self) -> dict[str, Any]:
        return {
            **self._base_identity_payload(),
            "span_id": self.span_id,
            "is_last_allowlisted_span": self.is_last_allowlisted_span,
            "later_exact_success_span_hashes": list(
                self.later_exact_success_span_hashes
            ),
            "later_same_executable_success_span_hashes": list(
                self.later_same_executable_success_span_hashes
            ),
            "later_shared_artifact_success_span_hashes": list(
                self.later_shared_artifact_success_span_hashes
            ),
        }

    def safe_payload(self) -> dict[str, Any]:
        return {**self._identity_payload(), "span_hash": self.span_hash}


@dataclass(frozen=True)
class TrialTraceEvidence:
    trial_id_hash: str
    family_hash: str
    trace_hash: str
    action_budget_receipt_hash: str
    action_event_hash: str
    baseline_success: bool
    action_budget_limit: int
    trace_complete: bool
    action_start_count: int
    command_span_count: int
    discarded_command_count: int
    changed_artifacts: tuple[PrimitiveRef, ...]
    spans: tuple[ActionSpanEvidence, ...]

    @property
    def evidence_hash(self) -> str:
        return stable_hash(self.safe_payload(include_hash=False))

    def safe_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "policy": CAUSAL_ACTION_SPAN_EVIDENCE_VERSION,
            "trial_id_hash": self.trial_id_hash,
            "family_hash": self.family_hash,
            "trace_hash": self.trace_hash,
            "action_budget_receipt_hash": self.action_budget_receipt_hash,
            "action_event_hash": self.action_event_hash,
            "baseline_success": self.baseline_success,
            "action_budget_limit": self.action_budget_limit,
            "trace_complete": self.trace_complete,
            "action_start_count": self.action_start_count,
            "command_span_count": self.command_span_count,
            "discarded_command_count": self.discarded_command_count,
            "changed_artifact_ids": [
                row.primitive_id for row in self.changed_artifacts
            ],
            "changed_artifact_set_hash": stable_hash(
                {
                    "artifact_ids": [
                        row.primitive_id for row in self.changed_artifacts
                    ]
                }
            ),
            "span_hashes": [row.span_hash for row in self.spans],
            "span_set_hash": stable_hash(
                {"span_hashes": [row.span_hash for row in self.spans]}
            ),
            "allowlisted_span_chronology_preserved": True,
            "allowlisted_command_occurrences_deduplicated_or_truncated": False,
            "signature_task_path_limit": 12,
            "full_raw_command_coverage_claimed": False,
            "raw_content_persisted": False,
            "command_output_used": False,
            "model_message_used": False,
            "verifier_content_used": False,
            "validation_or_test_content_used": False,
        }
        if include_hash:
            payload["evidence_hash"] = self.evidence_hash
        return payload


@dataclass(frozen=True)
class PrimitiveAssessment:
    scope: str
    primitive_id: str
    classification: PrimitiveAssessmentClass
    failure_span_count: int
    success_span_count: int
    independent_train_task_count: int
    recovery_contradiction_count: int
    successful_trial_cooccurrence_count: int
    complete_trace_count: int
    evidence_span_set_hash: str

    @property
    def observationally_inadmissible(self) -> bool:
        return False

    def safe_payload(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "primitive_id": self.primitive_id,
            "classification": self.classification.value,
            "failure_span_count": self.failure_span_count,
            "success_span_count": self.success_span_count,
            "independent_train_task_count": self.independent_train_task_count,
            "recovery_contradiction_count": self.recovery_contradiction_count,
            "successful_trial_cooccurrence_count": (
                self.successful_trial_cooccurrence_count
            ),
            "complete_trace_count": self.complete_trace_count,
            "evidence_span_set_hash": self.evidence_span_set_hash,
            "task_effect_attributed": False,
            "observationally_inadmissible": False,
            "raw_value_persisted": False,
        }


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_id: str
    locator: str
    format: ArtifactFormat
    support_count: int
    evidence_relations: tuple[str, ...]
    provenance_hash: str

    @property
    def access(self) -> str:
        return "task_local_read_write"

    def model_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "format": self.format.value,
            "access": self.access,
            "support_count": self.support_count,
            "evidence_relations": list(self.evidence_relations),
            "locator_disclosed": False,
        }

    def safe_payload(self) -> dict[str, Any]:
        return {
            **self.model_payload(),
            "locator_hash": stable_hash({"locator": self.locator}),
            "provenance_hash": self.provenance_hash,
            "raw_locator_persisted": False,
        }


@dataclass(frozen=True)
class CapabilitySpec:
    capability_id: str
    kind: CapabilityKind
    artifact_format: ArtifactFormat
    provenance_hash: str

    def payload(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "kind": self.kind.value,
            "artifact_format": self.artifact_format.value,
            "directive_text_owned_by_harness": True,
            "capability_requirement_derived_from_artifact_format": True,
            "capability_implementation_verified": False,
            "executable_disclosed": False,
            "model_selection_free_text_arguments_allowed": False,
            "runtime_agent_argument_surface_restricted": False,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class TypedOperatorNode:
    node_id: str
    kind: OperatorKind
    artifact_id: str
    capability_id: str
    input_type: str
    output_type: str
    depends_on: tuple[str, ...] = ()

    def payload(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "operator_id": self.kind.value,
            "artifact_ref": self.artifact_id,
            "capability_ref": self.capability_id,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "depends_on": list(self.depends_on),
            "model_authored_primitive_fields": [],
        }


@dataclass(frozen=True)
class TypedRecipe:
    recipe_id: str
    workflow: WorkflowKind
    primary_artifact_id: str
    capability_id: str
    nodes: tuple[TypedOperatorNode, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "recipe_id": self.recipe_id,
            "workflow": self.workflow.value,
            "primary_artifact_ref": self.primary_artifact_id,
            "capability_ref": self.capability_id,
            "operator_ids": [row.kind.value for row in self.nodes],
            "node_graph_hash": stable_hash(
                {"nodes": [row.payload() for row in self.nodes]}
            ),
        }


@dataclass(frozen=True)
class FamilyCapabilityGraph:
    target_family: str
    source_evidence_hash: str
    artifacts: tuple[ArtifactSpec, ...]
    capabilities: tuple[CapabilitySpec, ...]
    recipes: tuple[TypedRecipe, ...]

    @property
    def target_family_hash(self) -> str:
        return stable_hash({"family": self.target_family})

    @property
    def graph_hash(self) -> str:
        return stable_hash(self.safe_payload(include_hash=False))

    def model_catalog(self) -> dict[str, Any]:
        issues = self.validate()
        if issues:
            raise PermissionError(
                f"typed capability graph is invalid: {list(issues)}"
            )
        return {
            "grammar_version": TYPED_OPERATOR_GRAMMAR_VERSION,
            "selection_contract": TYPED_RECIPE_SELECTION_VERSION,
            "target_family_hash": self.target_family_hash,
            "artifacts": [row.model_payload() for row in self.artifacts],
            "capabilities": [row.payload() for row in self.capabilities],
            "recipes": [row.payload() for row in self.recipes],
            "model_output_schema": selection_schema(self),
            "model_authored_primitive_fields": [],
            "raw_artifact_locators_disclosed": False,
            "raw_executables_disclosed": False,
        }

    def safe_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "grammar_version": TYPED_OPERATOR_GRAMMAR_VERSION,
            "selection_contract": TYPED_RECIPE_SELECTION_VERSION,
            "lowering_version": TYPED_OPERATOR_LOWERING_VERSION,
            "target_family_hash": self.target_family_hash,
            "source_evidence_hash": self.source_evidence_hash,
            "artifacts": [row.safe_payload() for row in self.artifacts],
            "capabilities": [row.payload() for row in self.capabilities],
            "recipes": [row.payload() for row in self.recipes],
            "allowed_operator_kinds": sorted(ALLOWED_OPERATOR_KINDS),
            "forbidden_operator_kinds": sorted(FORBIDDEN_OPERATOR_KINDS),
            "model_authored_primitive_count": 0,
            "raw_content_persisted": False,
        }
        if include_hash:
            payload["graph_hash"] = self.graph_hash
        return payload

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if (
            not self.target_family.strip()
            or self.target_family != self.target_family.strip()
        ):
            issues.append("target_family_missing")
        if not _is_sha256(self.source_evidence_hash):
            issues.append("source_evidence_hash_invalid")
        artifact_ids = [row.artifact_id for row in self.artifacts]
        capability_ids = [row.capability_id for row in self.capabilities]
        recipe_ids = [row.recipe_id for row in self.recipes]
        if not self.artifacts:
            issues.append("artifact_registry_empty")
        if not self.capabilities:
            issues.append("capability_registry_empty")
        if not self.recipes:
            issues.append("recipe_registry_empty")
        if len(self.artifacts) > MAX_REGISTERED_ARTIFACTS_PER_FAMILY:
            issues.append("artifact_registry_exceeds_limit")
        if len(artifact_ids) != len(set(artifact_ids)):
            issues.append("duplicate_artifact_id")
        if len(capability_ids) != len(set(capability_ids)):
            issues.append("duplicate_capability_id")
        if len(recipe_ids) != len(set(recipe_ids)):
            issues.append("duplicate_recipe_id")
        artifact_locators = [row.locator for row in self.artifacts]
        if len(artifact_locators) != len(set(artifact_locators)):
            issues.append("duplicate_artifact_locator")
        if tuple(sorted(self.artifacts, key=_artifact_rank)) != self.artifacts:
            issues.append("artifact_registry_order_not_canonical")
        for artifact in self.artifacts:
            if _canonical_task_locator(artifact.locator) != artifact.locator:
                issues.append("artifact_locator_not_canonical")
            if _artifact_format(artifact.locator) is not artifact.format:
                issues.append("artifact_format_not_canonical")
            if artifact.support_count <= 0:
                issues.append("artifact_support_invalid")
            if not _is_sha256(artifact.provenance_hash):
                issues.append("artifact_provenance_hash_invalid")
            relation_set = set(artifact.evidence_relations)
            if (
                not artifact.evidence_relations
                or tuple(sorted(relation_set)) != artifact.evidence_relations
            ):
                issues.append("artifact_relations_not_canonical")
            if not relation_set.issubset(KNOWN_ARTIFACT_EVIDENCE_RELATIONS):
                issues.append("artifact_relation_unknown")
            if not relation_set.intersection(
                POSITIVE_ARTIFACT_EVIDENCE_RELATIONS
            ):
                issues.append("artifact_lacks_positive_availability_evidence")
            if artifact.artifact_id != _artifact_id(
                target_family=self.target_family,
                locator=artifact.locator,
                artifact_format=artifact.format,
                support_count=artifact.support_count,
                evidence_relations=artifact.evidence_relations,
                provenance_hash=artifact.provenance_hash,
            ):
                issues.append("artifact_id_not_canonical")
        known_artifacts = set(artifact_ids)
        known_capabilities = set(capability_ids)
        artifact_by_id = {row.artifact_id: row for row in self.artifacts}
        capability_by_id = {
            row.capability_id: row for row in self.capabilities
        }
        for capability in self.capabilities:
            if not _is_sha256(capability.provenance_hash):
                issues.append("capability_provenance_hash_invalid")
        expected_capabilities = tuple(
            _capability_spec(
                target_family=self.target_family,
                artifact=artifact,
            )
            for artifact in self.artifacts
        )
        if self.capabilities != expected_capabilities:
            issues.append("capability_registry_not_canonical")
        for recipe in self.recipes:
            if recipe.primary_artifact_id not in known_artifacts:
                issues.append("recipe_unknown_artifact")
            if recipe.capability_id not in known_capabilities:
                issues.append("recipe_unknown_capability")
            if not recipe.nodes:
                issues.append("recipe_nodes_empty")
            artifact = artifact_by_id.get(recipe.primary_artifact_id)
            capability = capability_by_id.get(recipe.capability_id)
            if artifact is not None and capability is not None:
                if capability.artifact_format is not artifact.format:
                    issues.append("capability_artifact_format_mismatch")
                if capability.kind is not _capability_for_format(artifact.format):
                    issues.append("capability_kind_mismatch")
            node_ids = [row.node_id for row in recipe.nodes]
            known_nodes = set(node_ids)
            if len(node_ids) != len(known_nodes):
                issues.append("duplicate_operator_node_id")
            for node in recipe.nodes:
                if node.kind.value not in ALLOWED_OPERATOR_KINDS:
                    issues.append("operator_not_allowlisted")
                if node.kind.value in FORBIDDEN_OPERATOR_KINDS:
                    issues.append("forbidden_operator_expressible")
                if node.artifact_id not in known_artifacts:
                    issues.append("operator_unknown_artifact")
                if node.artifact_id != recipe.primary_artifact_id:
                    issues.append("operator_recipe_artifact_mismatch")
                if node.capability_id not in known_capabilities:
                    issues.append("operator_unknown_capability")
                if node.capability_id != recipe.capability_id:
                    issues.append("operator_recipe_capability_mismatch")
                if any(dep not in known_nodes for dep in node.depends_on):
                    issues.append("operator_unknown_dependency")
                if not node.depends_on and node.input_type != "artifact_ref":
                    issues.append("operator_root_input_type_invalid")
                for dependency in node.depends_on:
                    parent = next(
                        (
                            candidate
                            for candidate in recipe.nodes
                            if candidate.node_id == dependency
                        ),
                        None,
                    )
                    if parent is not None and parent.output_type != node.input_type:
                        issues.append("operator_port_type_mismatch")
            if _has_node_cycle(recipe.nodes):
                issues.append("operator_graph_cycle")
        expected_recipes = tuple(
            sorted(
                (
                    _build_recipe(
                        target_family=self.target_family,
                        artifact=artifact,
                        capability=capability,
                        workflow=workflow,
                    )
                    for artifact, capability in zip(
                        self.artifacts,
                        expected_capabilities,
                    )
                    for workflow in _workflows_for_format(artifact.format)
                ),
                key=lambda row: row.recipe_id,
            )
        )
        if self.recipes != expected_recipes:
            issues.append("recipe_registry_not_canonical")
        return tuple(sorted(set(issues)))


@dataclass(frozen=True)
class TypedRecipeSelectionSnapshot:
    """Frozen graph/catalog commitments used by the production selector.

    The graph retains harness-only artifact locators in memory.  Only
    ``model_payload`` may cross the model boundary, and that payload is derived
    from ``FamilyCapabilityGraph.model_catalog`` where locators and executable
    values are absent.
    """

    graph: FamilyCapabilityGraph
    expected_graph_hash: str
    expected_model_catalog_hash: str

    @property
    def snapshot_hash(self) -> str:
        return stable_hash(self.safe_payload(include_hash=False))

    def validate(self) -> tuple[str, ...]:
        issues = list(self.graph.validate())
        if (
            not _is_sha256(self.expected_graph_hash)
            or self.graph.graph_hash != self.expected_graph_hash
        ):
            issues.append("typed_snapshot_graph_hash_mismatch")
        model_catalog_hash = (
            stable_hash(self.graph.model_catalog()) if not issues else ""
        )
        if not _is_sha256(self.expected_model_catalog_hash) or (
            model_catalog_hash
            and model_catalog_hash != self.expected_model_catalog_hash
        ):
            issues.append("typed_snapshot_model_catalog_hash_mismatch")
        return tuple(sorted(set(issues)))

    def safe_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "grammar_version": TYPED_OPERATOR_GRAMMAR_VERSION,
            "selection_contract": TYPED_RECIPE_SELECTION_VERSION,
            "target_family_hash": self.graph.target_family_hash,
            "source_evidence_hash": self.graph.source_evidence_hash,
            "graph_hash": self.expected_graph_hash,
            "model_catalog_hash": self.expected_model_catalog_hash,
            "recipe_count": len(self.graph.recipes),
            "raw_artifact_locators_persisted": False,
            "raw_primitive_values_persisted": False,
        }
        if include_hash:
            payload["snapshot_hash"] = self.snapshot_hash
        return payload

    def model_payload(self) -> dict[str, Any]:
        issues = self.validate()
        if issues:
            raise PermissionError(
                f"typed selection snapshot is invalid: {list(issues)}"
            )
        return {
            **self.safe_payload(),
            "catalog": self.graph.model_catalog(),
        }


def freeze_typed_recipe_selection_snapshot(
    graph: FamilyCapabilityGraph,
) -> TypedRecipeSelectionSnapshot:
    issues = graph.validate()
    if issues:
        raise PermissionError(
            f"typed recipe graph is invalid: {list(issues)}"
        )
    snapshot = TypedRecipeSelectionSnapshot(
        graph=graph,
        expected_graph_hash=graph.graph_hash,
        expected_model_catalog_hash=stable_hash(graph.model_catalog()),
    )
    snapshot_issues = snapshot.validate()
    if snapshot_issues:
        raise PermissionError(
            f"typed selection snapshot is invalid: {list(snapshot_issues)}"
        )
    return snapshot


@dataclass(frozen=True)
class TypedSelectionSnapshotLedger:
    """Externally anchored ordered snapshot set for production selection.

    Snapshot self-hashes establish internal consistency only.  This ledger
    additionally binds the exact feasibility decision, frozen TRAIN source,
    split manifest, and ordered graph/catalog set authorized by the execution
    protocol.  File and receipt verification is performed by the benchmark
    loader before this immutable object is constructed.
    """

    ledger_version: str
    feasibility_preregistration_hash: str
    feasibility_result_receipt_sha256: str
    feasibility_decision_hash: str
    feasibility_report_hash: str
    manifest_hash: str
    source_train_receipt_hash: str
    graph_set_hash: str
    model_catalog_set_hash: str
    expected_target_family_hashes: tuple[str, ...]
    snapshots: tuple[TypedRecipeSelectionSnapshot, ...]

    @property
    def snapshot_hashes(self) -> tuple[str, ...]:
        return tuple(row.snapshot_hash for row in self.snapshots)

    @property
    def snapshot_set_hash(self) -> str:
        return stable_hash({"snapshot_hashes": list(self.snapshot_hashes)})

    @property
    def ledger_hash(self) -> str:
        return stable_hash(self.safe_payload(include_hash=False))

    def validate(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.ledger_version != TYPED_SELECTION_SNAPSHOT_LEDGER_VERSION:
            issues.append("typed_snapshot_ledger_version_mismatch")
        for value, issue in (
            (
                self.feasibility_preregistration_hash,
                "typed_snapshot_ledger_preregistration_hash_invalid",
            ),
            (
                self.feasibility_result_receipt_sha256,
                "typed_snapshot_ledger_result_receipt_hash_invalid",
            ),
            (
                self.feasibility_decision_hash,
                "typed_snapshot_ledger_decision_hash_invalid",
            ),
            (
                self.feasibility_report_hash,
                "typed_snapshot_ledger_report_hash_invalid",
            ),
            (
                self.manifest_hash,
                "typed_snapshot_ledger_manifest_hash_invalid",
            ),
            (
                self.source_train_receipt_hash,
                "typed_snapshot_ledger_source_receipt_hash_invalid",
            ),
            (
                self.graph_set_hash,
                "typed_snapshot_ledger_graph_set_hash_invalid",
            ),
            (
                self.model_catalog_set_hash,
                "typed_snapshot_ledger_catalog_set_hash_invalid",
            ),
        ):
            if not _is_sha256(value):
                issues.append(issue)
        if not self.snapshots:
            issues.append("typed_snapshot_ledger_empty")
        issues.extend(
            issue for row in self.snapshots for issue in row.validate()
        )
        snapshot_hashes = self.snapshot_hashes
        if len(set(snapshot_hashes)) != len(snapshot_hashes):
            issues.append("typed_snapshot_ledger_duplicate_snapshot")
        target_hashes = tuple(
            row.graph.target_family_hash for row in self.snapshots
        )
        if (
            not self.expected_target_family_hashes
            or target_hashes != self.expected_target_family_hashes
            or any(not _is_sha256(value) for value in target_hashes)
        ):
            issues.append("typed_snapshot_ledger_target_order_mismatch")
        if len(set(target_hashes)) != len(target_hashes):
            issues.append("typed_snapshot_ledger_duplicate_target")
        computed_graph_set_hash = stable_hash(
            {
                "outcomes": [
                    {
                        "target_family_hash": row.graph.target_family_hash,
                        "graph_hash": row.expected_graph_hash,
                        "availability_error_hash": None,
                    }
                    for row in self.snapshots
                ]
            }
        )
        if computed_graph_set_hash != self.graph_set_hash:
            issues.append("typed_snapshot_ledger_graph_set_mismatch")
        computed_catalog_set_hash = stable_hash(
            {
                "catalog_hashes": [
                    row.expected_model_catalog_hash
                    for row in self.snapshots
                ]
            }
        )
        if computed_catalog_set_hash != self.model_catalog_set_hash:
            issues.append("typed_snapshot_ledger_catalog_set_mismatch")
        return tuple(sorted(set(issues)))

    def safe_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "ledger_version": self.ledger_version,
            "feasibility_preregistration_hash": (
                self.feasibility_preregistration_hash
            ),
            "feasibility_result_receipt_sha256": (
                self.feasibility_result_receipt_sha256
            ),
            "feasibility_decision_hash": self.feasibility_decision_hash,
            "feasibility_report_hash": self.feasibility_report_hash,
            "manifest_hash": self.manifest_hash,
            "source_train_receipt_hash": self.source_train_receipt_hash,
            "graph_set_hash": self.graph_set_hash,
            "model_catalog_set_hash": self.model_catalog_set_hash,
            "expected_target_family_hashes": list(
                self.expected_target_family_hashes
            ),
            "snapshot_hashes": list(self.snapshot_hashes),
            "snapshot_set_hash": self.snapshot_set_hash,
            "snapshots": [row.safe_payload() for row in self.snapshots],
            "raw_content_persisted": False,
        }
        if include_hash:
            payload["ledger_hash"] = self.ledger_hash
        return payload


@dataclass(frozen=True)
class TypedSelectionFreezeAuthorization:
    """Receipt-derived authority to freeze a fresh typed protocol.

    This is deliberately separate from the TRAIN snapshot ledger.  A ledger
    reconstructed by the offline integration diagnostic has no authorization;
    only the production loader may attach this object after exact verification
    of the formal result receipt and its canonical artifacts.
    """

    authorization_policy: str
    result_receipt_stable_hash: str
    result_receipt_file_sha256: str
    source_binding_hash: str
    snapshot_ledger_hash: str
    fresh_development_protocol_freeze_eligible: bool
    development_task_execution_authorized: bool

    def validate_for(
        self,
        ledger: TypedSelectionSnapshotLedger,
    ) -> tuple[str, ...]:
        issues: list[str] = []
        if self.authorization_policy != (
            TYPED_SELECTION_FREEZE_AUTHORIZATION_VERSION
        ):
            issues.append("typed_selection_freeze_authorization_policy_invalid")
        for value, issue in (
            (
                self.result_receipt_stable_hash,
                "typed_selection_result_receipt_hash_invalid",
            ),
            (
                self.result_receipt_file_sha256,
                "typed_selection_result_receipt_file_hash_invalid",
            ),
            (
                self.source_binding_hash,
                "typed_selection_source_binding_hash_invalid",
            ),
            (
                self.snapshot_ledger_hash,
                "typed_selection_authorized_ledger_hash_invalid",
            ),
        ):
            if not _is_sha256(value):
                issues.append(issue)
        if self.snapshot_ledger_hash != ledger.ledger_hash:
            issues.append("typed_selection_authorized_ledger_mismatch")
        if self.fresh_development_protocol_freeze_eligible is not True:
            issues.append("typed_selection_protocol_freeze_not_authorized")
        # The integration receipt grants protocol-freeze authority only.  Task
        # execution additionally requires the separately validated protocol
        # lock at the experiment boundary.
        if self.development_task_execution_authorized is not False:
            issues.append("typed_selection_receipt_scope_overclaimed")
        return tuple(sorted(set(issues)))

    @property
    def authorization_hash(self) -> str:
        return stable_hash(self.safe_payload(include_hash=False))

    def safe_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "authorization_policy": self.authorization_policy,
            "result_receipt_stable_hash": self.result_receipt_stable_hash,
            "result_receipt_file_sha256": self.result_receipt_file_sha256,
            "source_binding_hash": self.source_binding_hash,
            "snapshot_ledger_hash": self.snapshot_ledger_hash,
            "fresh_development_protocol_freeze_eligible": (
                self.fresh_development_protocol_freeze_eligible
            ),
            "development_task_execution_authorized": (
                self.development_task_execution_authorized
            ),
            "raw_content_persisted": False,
        }
        if include_hash:
            payload["authorization_hash"] = self.authorization_hash
        return payload


class TypedSelectionExecutionAuthorization:
    """Process-local capability issued only after protocol-lock validation.

    The formal result receipt authorizes freezing a protocol, not executing a
    task.  This separate sealed type is created by the paper-protocol boundary
    only after the claim-eligible lock and its embedded freeze receipt have
    both been revalidated.  Diagnostic callers can reconstruct snapshots and
    compile them offline, but cannot manufacture this execution type from a
    JSON payload or a collection of plausible hashes.
    """

    __slots__ = (
        "authorization_policy",
        "protocol_id",
        "protocol_hash",
        "protocol_lock_hash",
        "manifest_hash",
        "snapshot_ledger_hash",
        "freeze_authorization_hash",
        "task_execution_authorized",
        "_seal",
    )

    def __init__(
        self,
        *,
        authorization_policy: str,
        protocol_id: str,
        protocol_hash: str,
        protocol_lock_hash: str,
        manifest_hash: str,
        snapshot_ledger_hash: str,
        freeze_authorization_hash: str,
        task_execution_authorized: bool,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _TYPED_SELECTION_EXECUTION_AUTHORIZATION_SEAL:
            raise PermissionError(
                "typed selection execution authorization is factory-only"
            )
        object.__setattr__(self, "authorization_policy", authorization_policy)
        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "protocol_hash", protocol_hash)
        object.__setattr__(self, "protocol_lock_hash", protocol_lock_hash)
        object.__setattr__(self, "manifest_hash", manifest_hash)
        object.__setattr__(
            self,
            "snapshot_ledger_hash",
            snapshot_ledger_hash,
        )
        object.__setattr__(
            self,
            "freeze_authorization_hash",
            freeze_authorization_hash,
        )
        object.__setattr__(
            self,
            "task_execution_authorized",
            task_execution_authorized,
        )
        object.__setattr__(self, "_seal", _seal)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            "typed selection execution authorization is immutable"
        )

    def validate_for(
        self,
        ledger: TypedSelectionSnapshotLedger,
        freeze_authorization: TypedSelectionFreezeAuthorization,
        *,
        manifest_hash: str | None = None,
        protocol_hash: str | None = None,
    ) -> tuple[str, ...]:
        issues: list[str] = []
        if self._seal is not _TYPED_SELECTION_EXECUTION_AUTHORIZATION_SEAL:
            issues.append("typed_selection_execution_authorization_unsealed")
        if self.authorization_policy != (
            TYPED_SELECTION_EXECUTION_AUTHORIZATION_VERSION
        ):
            issues.append(
                "typed_selection_execution_authorization_policy_invalid"
            )
        if not isinstance(self.protocol_id, str) or not self.protocol_id:
            issues.append("typed_selection_execution_protocol_id_invalid")
        for value, issue in (
            (
                self.protocol_hash,
                "typed_selection_execution_protocol_hash_invalid",
            ),
            (
                self.protocol_lock_hash,
                "typed_selection_execution_lock_hash_invalid",
            ),
            (
                self.manifest_hash,
                "typed_selection_execution_manifest_hash_invalid",
            ),
            (
                self.snapshot_ledger_hash,
                "typed_selection_execution_ledger_hash_invalid",
            ),
            (
                self.freeze_authorization_hash,
                "typed_selection_execution_freeze_hash_invalid",
            ),
        ):
            if not _is_sha256(value):
                issues.append(issue)
        if self.snapshot_ledger_hash != ledger.ledger_hash:
            issues.append("typed_selection_execution_ledger_mismatch")
        if (
            self.freeze_authorization_hash
            != freeze_authorization.authorization_hash
        ):
            issues.append("typed_selection_execution_freeze_mismatch")
        if freeze_authorization.validate_for(ledger):
            issues.append("typed_selection_execution_freeze_invalid")
        if manifest_hash is not None and self.manifest_hash != manifest_hash:
            issues.append("typed_selection_execution_manifest_mismatch")
        if protocol_hash is not None and self.protocol_hash != protocol_hash:
            issues.append("typed_selection_execution_protocol_mismatch")
        if self.task_execution_authorized is not True:
            issues.append("typed_selection_task_execution_not_authorized")
        return tuple(sorted(set(issues)))

    @property
    def authorization_hash(self) -> str:
        return stable_hash(self.safe_payload(include_hash=False))

    def safe_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "authorization_policy": self.authorization_policy,
            "protocol_id": self.protocol_id,
            "protocol_hash": self.protocol_hash,
            "protocol_lock_hash": self.protocol_lock_hash,
            "manifest_hash": self.manifest_hash,
            "snapshot_ledger_hash": self.snapshot_ledger_hash,
            "freeze_authorization_hash": self.freeze_authorization_hash,
            "task_execution_authorized": self.task_execution_authorized,
            "raw_content_persisted": False,
        }
        if include_hash:
            payload["authorization_hash"] = self.authorization_hash
        return payload


def _issue_typed_selection_execution_authorization(
    *,
    protocol_id: str,
    protocol_hash: str,
    protocol_lock_hash: str,
    manifest_hash: str,
    ledger: TypedSelectionSnapshotLedger,
    freeze_authorization: TypedSelectionFreezeAuthorization,
) -> TypedSelectionExecutionAuthorization:
    """Issue one sealed execution capability after protocol validation."""

    authorization = TypedSelectionExecutionAuthorization(
        authorization_policy=(
            TYPED_SELECTION_EXECUTION_AUTHORIZATION_VERSION
        ),
        protocol_id=protocol_id,
        protocol_hash=protocol_hash,
        protocol_lock_hash=protocol_lock_hash,
        manifest_hash=manifest_hash,
        snapshot_ledger_hash=ledger.ledger_hash,
        freeze_authorization_hash=(
            freeze_authorization.authorization_hash
        ),
        task_execution_authorized=True,
        _seal=_TYPED_SELECTION_EXECUTION_AUTHORIZATION_SEAL,
    )
    issues = authorization.validate_for(
        ledger,
        freeze_authorization,
        manifest_hash=manifest_hash,
        protocol_hash=protocol_hash,
    )
    if issues:
        raise PermissionError(
            "typed selection execution authorization is invalid: "
            f"{list(issues)}"
        )
    return authorization


def freeze_typed_selection_snapshot_ledger(
    snapshots: Sequence[TypedRecipeSelectionSnapshot],
    *,
    feasibility_preregistration_hash: str,
    feasibility_result_receipt_sha256: str,
    feasibility_decision_hash: str,
    feasibility_report_hash: str,
    manifest_hash: str,
    source_train_receipt_hash: str,
    expected_graph_set_hash: str,
    expected_model_catalog_set_hash: str,
    expected_target_family_hashes: Sequence[str],
) -> TypedSelectionSnapshotLedger:
    ledger = TypedSelectionSnapshotLedger(
        ledger_version=TYPED_SELECTION_SNAPSHOT_LEDGER_VERSION,
        feasibility_preregistration_hash=feasibility_preregistration_hash,
        feasibility_result_receipt_sha256=(
            feasibility_result_receipt_sha256
        ),
        feasibility_decision_hash=feasibility_decision_hash,
        feasibility_report_hash=feasibility_report_hash,
        manifest_hash=manifest_hash,
        source_train_receipt_hash=source_train_receipt_hash,
        graph_set_hash=expected_graph_set_hash,
        model_catalog_set_hash=expected_model_catalog_set_hash,
        expected_target_family_hashes=tuple(expected_target_family_hashes),
        snapshots=tuple(snapshots),
    )
    issues = ledger.validate()
    if issues:
        raise PermissionError(
            f"typed snapshot ledger is invalid: {list(issues)}"
        )
    return ledger


def canonical_typed_recipe_selection_request(
    *,
    snapshot: TypedRecipeSelectionSnapshot,
    snapshot_ledger: TypedSelectionSnapshotLedger,
    evaluator_epoch: str,
    selection_round: int,
    excluded_recipe_ids: Sequence[str] = (),
    parent_program_hash: str | None = None,
    parent_recipe_id: str | None = None,
    failed_checks: Sequence[Mapping[str, Any]] = (),
    repair_depth: int = 0,
) -> dict[str, Any]:
    """Build the one canonical, sanitized opaque recipe-selection request.

    Both the proposer and the harness-owned binding registry call this builder.
    A binding therefore cannot merely carry caller-supplied receipt hashes: the
    registry independently reconstructs the request those hashes must describe.
    """

    snapshot_issues = snapshot.validate()
    if snapshot_issues:
        raise PermissionError(
            f"typed selection snapshot is invalid: {list(snapshot_issues)}"
        )
    ledger_issues = snapshot_ledger.validate()
    if ledger_issues:
        raise PermissionError(
            f"typed snapshot ledger is invalid: {list(ledger_issues)}"
        )
    if snapshot.snapshot_hash not in snapshot_ledger.snapshot_hashes:
        raise PermissionError("typed snapshot is outside the frozen ledger")
    if not isinstance(evaluator_epoch, str) or not evaluator_epoch.strip():
        raise ValueError("typed recipe selection evaluator epoch is missing")
    if (
        not isinstance(repair_depth, int)
        or isinstance(repair_depth, bool)
        or repair_depth < 0
    ):
        raise ValueError("typed repair depth must be a nonnegative integer")
    repair = bool(
        repair_depth or parent_program_hash is not None or parent_recipe_id is not None
    )
    if repair != bool(repair_depth):
        raise ValueError("typed repair depth and parent must be paired")
    if repair:
        if (
            not _is_sha256(parent_program_hash)
            or not isinstance(parent_recipe_id, str)
            or not parent_recipe_id
        ):
            raise PermissionError("typed repair parent receipt is malformed")
    elif parent_program_hash is not None or parent_recipe_id is not None:
        raise PermissionError("typed root parent receipt is malformed")
    elif failed_checks:
        raise PermissionError("typed root cannot carry failed-check receipts")

    graph_recipe_ids = {
        row.recipe_id for row in snapshot.graph.recipes
    }
    if any(
        not isinstance(recipe_id, str) or not recipe_id
        for recipe_id in excluded_recipe_ids
    ):
        raise PermissionError("typed selection exclusions are malformed")
    canonical_excluded = tuple(sorted(set(excluded_recipe_ids)))
    if not set(canonical_excluded).issubset(graph_recipe_ids):
        raise PermissionError("typed selection exclusions are malformed")
    expected_selection_round = len(canonical_excluded) + 1
    if (
        not isinstance(selection_round, int)
        or isinstance(selection_round, bool)
        or selection_round != expected_selection_round
    ):
        raise PermissionError(
            "typed selection round does not match the complete exclusion scope"
        )

    output_schema = selection_schema(snapshot.graph)
    allowed_recipe_ids = [
        recipe_id
        for recipe_id in output_schema["properties"]["recipe_id"]["enum"]
        if recipe_id not in canonical_excluded
    ]
    if not allowed_recipe_ids:
        raise PermissionError("typed selection snapshot has no untried recipe")
    output_schema = {
        **output_schema,
        "properties": {
            "recipe_id": {
                "type": "string",
                "enum": allowed_recipe_ids,
            }
        },
    }
    catalog = dict(snapshot.graph.model_catalog())
    catalog["model_output_schema"] = output_schema
    request_kind = (
        "select_typed_repair_recipe"
        if repair
        else "select_typed_root_recipe"
    )
    payload: dict[str, Any] = {
        "request_kind": request_kind,
        "contract_version": TYPED_RECIPE_SELECTION_VERSION,
        "grammar_version": TYPED_OPERATOR_GRAMMAR_VERSION,
        "evaluator_epoch": evaluator_epoch,
        "selection_scope": {
            "selection_round": selection_round,
            "excluded_recipe_ids": list(canonical_excluded),
            "excluded_recipe_set_hash": stable_hash(
                {"recipe_ids": list(canonical_excluded)}
            ),
            "excluded_recipe_count": len(canonical_excluded),
        },
        "selection_snapshot": {
            **snapshot.safe_payload(),
            "catalog": catalog,
        },
        "selection_authority": snapshot_ledger.safe_payload(),
        "output_schema": output_schema,
        "constraints": {
            "model_output_fields": ["recipe_id"],
            "primitive_values_model_authored": False,
            "artifact_locators_model_authored": False,
            "free_text_actions_model_authored": False,
            "harness_owned_materialization": True,
        },
    }
    if repair:
        assert parent_program_hash is not None
        assert parent_recipe_id is not None
        failed_check_rows: list[dict[str, str]] = []
        for row in failed_checks:
            if not isinstance(row, Mapping):
                raise PermissionError("typed failed-check receipt is malformed")
            failed_check_rows.append(
                {
                    "check": str(row.get("check") or ""),
                    "reason": str(row.get("reason") or ""),
                }
            )
        payload["repair_context"] = {
            "parent_program_hash": parent_program_hash,
            "parent_recipe_id": parent_recipe_id,
            "repair_depth": repair_depth,
            "failed_checks": failed_check_rows,
            "failed_check_set_hash": stable_hash(failed_check_rows),
            "parent_action_graph_disclosed": False,
            "free_text_repair_fields_allowed": False,
        }
    return payload


def canonical_typed_recipe_selection_response(
    recipe_id: str,
) -> dict[str, str]:
    """Return the complete canonical opaque selector response envelope."""

    if not isinstance(recipe_id, str) or not recipe_id:
        raise PermissionError("typed recipe response is malformed")
    return {"recipe_id": recipe_id}


@dataclass(frozen=True)
class TypedProgramBinding:
    binding_policy: str
    request_kind: str
    snapshot_hash: str
    graph_hash: str
    model_catalog_hash: str
    snapshot_ledger_hash: str
    recipe_id: str
    evaluator_epoch: str
    program_id: str
    program_executable_hash: str
    request_hash: str
    response_hash: str
    selection_round: int
    excluded_recipe_ids: tuple[str, ...] = ()
    repair_depth: int = 0
    failed_checks: tuple[tuple[str, str], ...] = ()
    parent_program_id: str | None = None
    parent_program_hash: str | None = None
    parent_recipe_id: str | None = None
    parent_binding_hash: str | None = None
    lineage_program_ids: tuple[str, ...] = ()
    lineage_recipe_ids: tuple[str, ...] = ()

    @property
    def program_identity_hash(self) -> str:
        """Status-invariant identity used by descendant request receipts."""

        return stable_hash(
            {
                "program_id": self.program_id,
                "evaluator_epoch": self.evaluator_epoch,
                "program_executable_hash": self.program_executable_hash,
            }
        )

    @property
    def binding_hash(self) -> str:
        return stable_hash(self.safe_payload(include_hash=False))

    def safe_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "binding_policy": self.binding_policy,
            "request_kind": self.request_kind,
            "snapshot_hash": self.snapshot_hash,
            "graph_hash": self.graph_hash,
            "model_catalog_hash": self.model_catalog_hash,
            "snapshot_ledger_hash": self.snapshot_ledger_hash,
            "recipe_id": self.recipe_id,
            "evaluator_epoch": self.evaluator_epoch,
            "program_id": self.program_id,
            "program_executable_hash": self.program_executable_hash,
            "program_identity_hash": self.program_identity_hash,
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "selection_round": self.selection_round,
            "excluded_recipe_ids": list(self.excluded_recipe_ids),
            "excluded_recipe_set_hash": stable_hash(
                {"recipe_ids": list(self.excluded_recipe_ids)}
            ),
            "repair_depth": self.repair_depth,
            "failed_checks": [
                {"check": check, "reason": reason}
                for check, reason in self.failed_checks
            ],
            "failed_check_set_hash": stable_hash(
                [
                    {"check": check, "reason": reason}
                    for check, reason in self.failed_checks
                ]
            ),
            "parent_program_id": self.parent_program_id,
            "parent_program_hash": self.parent_program_hash,
            "parent_recipe_id": self.parent_recipe_id,
            "parent_binding_hash": self.parent_binding_hash,
            "lineage_program_ids": list(self.lineage_program_ids),
            "lineage_recipe_ids": list(self.lineage_recipe_ids),
            "raw_content_persisted": False,
        }
        if include_hash:
            payload["binding_hash"] = self.binding_hash
        return payload


def validate_typed_selection_history_payloads(
    history_payload: Mapping[str, Any],
    *,
    snapshot_ledger: TypedSelectionSnapshotLedger,
) -> tuple[dict[str, Any], ...]:
    """Rebuild every history-only receipt without trusting caller hashes.

    Attempt rows may have no archived HypothesisProgram because behavior
    deduplication intentionally happens after selection.  The frozen recipe,
    evaluator epoch and parent binding still determine every executable and
    provenance field, so the reader can reconstruct the canonical binding
    directly and reject a self-rehashed forged attempt.
    """

    if snapshot_ledger.validate():
        raise PermissionError("typed selection history ledger is invalid")
    snapshots = {
        row.snapshot_hash: row for row in snapshot_ledger.snapshots
    }
    pending: list[tuple[int, str, dict[str, Any]]] = []
    for key, raw_row in history_payload.items():
        if not isinstance(key, str) or not isinstance(raw_row, Mapping):
            raise PermissionError(
                "typed selection history row is malformed"
            )
        row = dict(raw_row)
        repair_depth = row.get("repair_depth")
        if (
            not isinstance(repair_depth, int)
            or isinstance(repair_depth, bool)
            or repair_depth < 0
        ):
            raise PermissionError(
                "typed selection history repair depth is malformed"
            )
        pending.append((repair_depth, key, row))

    bindings_by_hash: dict[str, TypedProgramBinding] = {}
    canonical_rows: list[dict[str, Any]] = []
    for repair_depth, key, row in sorted(pending):
        snapshot = snapshots.get(str(row.get("snapshot_hash") or ""))
        if snapshot is None:
            raise PermissionError(
                "typed selection history snapshot is outside the ledger"
            )
        evaluator_epoch = row.get("evaluator_epoch")
        recipe_id = row.get("recipe_id")
        excluded_recipe_ids = row.get("excluded_recipe_ids")
        failed_checks = row.get("failed_checks")
        if (
            not isinstance(evaluator_epoch, str)
            or not evaluator_epoch
            or not isinstance(recipe_id, str)
            or not isinstance(excluded_recipe_ids, list)
            or not all(
                isinstance(value, str)
                for value in excluded_recipe_ids
            )
            or not isinstance(failed_checks, list)
            or any(
                not isinstance(value, Mapping)
                or set(value) != {"check", "reason"}
                or not isinstance(value.get("check"), str)
                or not isinstance(value.get("reason"), str)
                for value in failed_checks
            )
        ):
            raise PermissionError(
                "typed selection history selection receipt is malformed"
            )
        canonical_exclusions = tuple(
            sorted(set(excluded_recipe_ids))
        )
        if canonical_exclusions != tuple(excluded_recipe_ids):
            raise PermissionError(
                "typed selection history exclusions are not canonical"
            )
        selection_round = row.get("selection_round")
        if (
            not isinstance(selection_round, int)
            or isinstance(selection_round, bool)
            or selection_round != len(canonical_exclusions) + 1
        ):
            raise PermissionError(
                "typed selection history round is malformed"
            )
        canonical_failed_checks = tuple(
            (str(value["check"]), str(value["reason"]))
            for value in failed_checks
        )
        base_program = materialize_recipe_selection(
            {"recipe_id": recipe_id},
            graph=snapshot.graph,
            evaluator_epoch=evaluator_epoch,
            expected_graph_hash=snapshot.expected_graph_hash,
            expected_model_catalog_hash=(
                snapshot.expected_model_catalog_hash
            ),
        )
        executable_hash = stable_hash(
            _typed_program_executable_payload(base_program)
        )

        parent_binding_hash = row.get("parent_binding_hash")
        parent_binding = (
            bindings_by_hash.get(parent_binding_hash)
            if isinstance(parent_binding_hash, str)
            else None
        )
        if repair_depth == 0:
            if parent_binding_hash is not None:
                raise PermissionError(
                    "typed root selection history has a parent"
                )
            program_id = base_program.id
            parent_program_id = None
            parent_program_hash = None
            parent_recipe_id = None
            lineage_program_ids: tuple[str, ...] = ()
            lineage_recipe_ids: tuple[str, ...] = ()
        else:
            if parent_binding is None:
                raise PermissionError(
                    "typed repair selection history parent is missing"
                )
            if (
                parent_binding.snapshot_hash != snapshot.snapshot_hash
                or parent_binding.evaluator_epoch != evaluator_epoch
                or repair_depth != parent_binding.repair_depth + 1
            ):
                raise PermissionError(
                    "typed repair selection history parent drifted"
                )
            parent_program_id = parent_binding.program_id
            parent_program_hash = parent_binding.program_identity_hash
            parent_recipe_id = parent_binding.recipe_id
            lineage_program_ids = (
                *parent_binding.lineage_program_ids,
                parent_binding.program_id,
            )
            lineage_recipe_ids = (
                *parent_binding.lineage_recipe_ids,
                parent_binding.recipe_id,
            )
            canonical_content = base_program.to_dict()
            canonical_content.pop("id", None)
            branch_identity_hash = stable_hash(
                {
                    "policy": TYPED_REPAIR_BRANCH_ID_POLICY_VERSION,
                    "parent_program_hash": parent_program_hash,
                    "snapshot_hash": snapshot.snapshot_hash,
                    "snapshot_ledger_hash": snapshot_ledger.ledger_hash,
                    "recipe_id": recipe_id,
                    "repair_depth": repair_depth,
                    "canonical_program_without_id": canonical_content,
                }
            )
            program_id = f"repair_{branch_identity_hash}"

        expected_request = canonical_typed_recipe_selection_request(
            snapshot=snapshot,
            snapshot_ledger=snapshot_ledger,
            evaluator_epoch=evaluator_epoch,
            selection_round=selection_round,
            excluded_recipe_ids=canonical_exclusions,
            parent_program_hash=parent_program_hash,
            parent_recipe_id=parent_recipe_id,
            failed_checks=[
                {"check": check, "reason": reason}
                for check, reason in canonical_failed_checks
            ],
            repair_depth=repair_depth,
        )
        expected_response = canonical_typed_recipe_selection_response(
            recipe_id
        )
        binding = TypedProgramBinding(
            binding_policy=TYPED_PROGRAM_BINDING_VERSION,
            request_kind=str(expected_request["request_kind"]),
            snapshot_hash=snapshot.snapshot_hash,
            graph_hash=snapshot.expected_graph_hash,
            model_catalog_hash=snapshot.expected_model_catalog_hash,
            snapshot_ledger_hash=snapshot_ledger.ledger_hash,
            recipe_id=recipe_id,
            evaluator_epoch=evaluator_epoch,
            program_id=program_id,
            program_executable_hash=executable_hash,
            request_hash=stable_hash(expected_request),
            response_hash=stable_hash(expected_response),
            selection_round=selection_round,
            excluded_recipe_ids=canonical_exclusions,
            repair_depth=repair_depth,
            failed_checks=canonical_failed_checks,
            parent_program_id=parent_program_id,
            parent_program_hash=parent_program_hash,
            parent_recipe_id=parent_recipe_id,
            parent_binding_hash=(
                parent_binding.binding_hash
                if parent_binding is not None
                else None
            ),
            lineage_program_ids=lineage_program_ids,
            lineage_recipe_ids=lineage_recipe_ids,
        )
        canonical = binding.safe_payload()
        if key != binding.binding_hash or canonical != row:
            raise PermissionError(
                "typed selection history is not canonical"
            )
        bindings_by_hash[binding.binding_hash] = binding
        canonical_rows.append(canonical)
    return tuple(canonical_rows)


class TypedProgramBindingRegistry:
    """Harness-owned provenance registry shared by proposer and compiler."""

    def __init__(
        self,
        *,
        snapshot_ledger: TypedSelectionSnapshotLedger | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._bindings: dict[str, TypedProgramBinding] = {}
        self._binding_keys_by_program_id: dict[str, str] = {}
        self._bindings_by_hash: dict[str, TypedProgramBinding] = {}
        self._programs_by_binding_hash: dict[str, HypothesisProgram] = {}
        self._snapshots: dict[str, TypedRecipeSelectionSnapshot] = {}
        self._snapshot_ledger: TypedSelectionSnapshotLedger | None = None
        if snapshot_ledger is not None:
            self.bind_snapshot_ledger(snapshot_ledger)

    @property
    def snapshot_ledger_hash(self) -> str:
        with self._lock:
            ledger = self._snapshot_ledger
        return ledger.ledger_hash if ledger is not None else ""

    def bind_snapshot_ledger(
        self,
        ledger: TypedSelectionSnapshotLedger,
    ) -> None:
        issues = ledger.validate()
        if issues:
            raise PermissionError(
                f"typed snapshot ledger is invalid: {list(issues)}"
            )
        with self._lock:
            existing = self._snapshot_ledger
            if existing is not None and existing.ledger_hash != ledger.ledger_hash:
                raise PermissionError("typed snapshot ledger binding conflict")
            self._snapshot_ledger = ledger

    def require_snapshot_ledger(
        self,
        snapshot: TypedRecipeSelectionSnapshot | None = None,
    ) -> TypedSelectionSnapshotLedger:
        with self._lock:
            ledger = self._snapshot_ledger
        if ledger is None:
            raise PermissionError("typed snapshot ledger binding is missing")
        if ledger.validate():
            raise PermissionError("typed snapshot ledger binding is invalid")
        if snapshot is not None and snapshot.snapshot_hash not in (
            ledger.snapshot_hashes
        ):
            raise PermissionError("typed snapshot is outside the frozen ledger")
        return ledger

    def register(
        self,
        program: HypothesisProgram,
        *,
        snapshot: TypedRecipeSelectionSnapshot,
        recipe_id: str,
        request_kind: str,
        request_hash: str,
        response_hash: str,
        selection_round: int,
        excluded_recipe_ids: Sequence[str] = (),
        parent: HypothesisProgram | None = None,
        failed_checks: Sequence[Mapping[str, Any]] = (),
        repair_depth: int = 0,
    ) -> TypedProgramBinding:
        ledger = self.require_snapshot_ledger(snapshot)
        if snapshot.validate():
            raise PermissionError("typed program snapshot is invalid")
        if typed_recipe_id_for_program(program, snapshot=snapshot) != recipe_id:
            raise PermissionError(
                "typed program is not the frozen recipe materialization"
            )
        if request_kind not in {
            "select_typed_root_recipe",
            "select_typed_repair_recipe",
        }:
            raise PermissionError("typed program request kind is malformed")
        if (request_kind == "select_typed_repair_recipe") != (
            parent is not None
        ):
            raise PermissionError(
                "typed repair registration requires a bound parent"
            )
        graph_recipe_ids = {
            row.recipe_id for row in snapshot.graph.recipes
        }
        if any(
            not isinstance(value, str) or not value
            for value in excluded_recipe_ids
        ):
            raise PermissionError("typed selection exclusions are malformed")
        canonical_excluded_recipe_ids = tuple(
            sorted(set(excluded_recipe_ids))
        )
        if (
            not set(canonical_excluded_recipe_ids).issubset(graph_recipe_ids)
            or recipe_id in canonical_excluded_recipe_ids
        ):
            raise PermissionError("typed selection exclusions are malformed")
        parent_binding: TypedProgramBinding | None = None
        parent_recipe_id: str | None = None
        if parent is None:
            if program.parent_id is not None or program.lineage:
                raise PermissionError(
                    "typed repair registration requires a bound parent"
                )
        else:
            parent_binding = self.require(parent)
            parent_recipe_id = typed_recipe_id_for_program(
                parent,
                snapshot=snapshot,
            )
            if parent_binding.snapshot_hash != snapshot.snapshot_hash:
                raise PermissionError(
                    "typed repair parent snapshot binding mismatch"
                )
            if parent.evaluator_epoch != program.evaluator_epoch:
                raise PermissionError("typed repair crossed evaluator epochs")
            if (
                parent_recipe_id is None
                or parent_recipe_id != parent_binding.recipe_id
            ):
                raise PermissionError(
                    "typed repair parent recipe binding mismatch"
                )
            if program.parent_id != parent.id or tuple(program.lineage) != (
                *parent.lineage,
                parent.id,
            ):
                raise PermissionError(
                    "typed repair ancestry does not match its bound parent"
                )
            required_lineage_recipe_ids = {
                *parent_binding.excluded_recipe_ids,
                *parent_binding.lineage_recipe_ids,
                parent_binding.recipe_id,
            }
            if not required_lineage_recipe_ids.issubset(
                canonical_excluded_recipe_ids
            ):
                raise PermissionError(
                    "typed repair exclusions omit inherited or lineage recipes"
                )
        expected_selection_round = len(canonical_excluded_recipe_ids) + 1
        if (
            not isinstance(selection_round, int)
            or isinstance(selection_round, bool)
            or selection_round != expected_selection_round
        ):
            raise PermissionError(
                "typed selection round does not match the complete exclusion scope"
            )
        canonical_failed_checks = tuple(
            (
                str(row.get("check") or ""),
                str(row.get("reason") or ""),
            )
            for row in failed_checks
            if isinstance(row, Mapping)
        )
        if len(canonical_failed_checks) != len(failed_checks):
            raise PermissionError("typed failed-check receipt is malformed")
        expected_request = canonical_typed_recipe_selection_request(
            snapshot=snapshot,
            snapshot_ledger=ledger,
            evaluator_epoch=program.evaluator_epoch,
            selection_round=selection_round,
            excluded_recipe_ids=canonical_excluded_recipe_ids,
            parent_program_hash=(
                parent_binding.program_identity_hash
                if parent_binding is not None
                else None
            ),
            parent_recipe_id=parent_recipe_id,
            failed_checks=[
                {"check": check, "reason": reason}
                for check, reason in canonical_failed_checks
            ],
            repair_depth=repair_depth,
        )
        if (
            not _is_sha256(request_hash)
            or request_hash != stable_hash(expected_request)
        ):
            raise PermissionError(
                "typed selection request receipt does not match the canonical request"
            )
        expected_response = canonical_typed_recipe_selection_response(
            recipe_id
        )
        if (
            not _is_sha256(response_hash)
            or response_hash != stable_hash(expected_response)
        ):
            raise PermissionError(
                "typed selection response receipt does not match the canonical response"
            )
        executable_hash = stable_hash(
            _typed_program_executable_payload(program)
        )
        key = _typed_program_binding_key(program)
        binding = TypedProgramBinding(
            binding_policy=TYPED_PROGRAM_BINDING_VERSION,
            request_kind=request_kind,
            snapshot_hash=snapshot.snapshot_hash,
            graph_hash=snapshot.expected_graph_hash,
            model_catalog_hash=snapshot.expected_model_catalog_hash,
            snapshot_ledger_hash=ledger.ledger_hash,
            recipe_id=recipe_id,
            evaluator_epoch=program.evaluator_epoch,
            program_id=program.id,
            program_executable_hash=executable_hash,
            request_hash=request_hash,
            response_hash=response_hash,
            selection_round=selection_round,
            excluded_recipe_ids=canonical_excluded_recipe_ids,
            repair_depth=repair_depth,
            failed_checks=canonical_failed_checks,
            parent_program_id=parent.id if parent is not None else None,
            parent_program_hash=(
                parent_binding.program_identity_hash
                if parent_binding is not None
                else None
            ),
            parent_recipe_id=parent_recipe_id,
            parent_binding_hash=(
                parent_binding.binding_hash
                if parent_binding is not None
                else None
            ),
            lineage_program_ids=tuple(program.lineage),
            lineage_recipe_ids=(
                (
                    *parent_binding.lineage_recipe_ids,
                    parent_binding.recipe_id,
                )
                if parent_binding is not None
                else ()
            ),
        )
        with self._lock:
            existing_snapshot = self._snapshots.get(snapshot.snapshot_hash)
            if (
                existing_snapshot is not None
                and existing_snapshot.safe_payload()
                != snapshot.safe_payload()
            ):
                raise PermissionError("typed snapshot hash collision")
            existing = self._bindings.get(key)
            if existing is not None and existing != binding:
                raise PermissionError("typed program binding conflict")
            existing_key = self._binding_keys_by_program_id.get(program.id)
            if existing_key is not None and existing_key != key:
                raise PermissionError("typed program ID binding conflict")
            self._snapshots[snapshot.snapshot_hash] = snapshot
            self._bindings[key] = binding
            self._binding_keys_by_program_id[program.id] = key
            self._bindings_by_hash[binding.binding_hash] = binding
            self._programs_by_binding_hash[binding.binding_hash] = program
        return binding

    def require(self, program: HypothesisProgram) -> TypedProgramBinding:
        key = _typed_program_binding_key(program)
        with self._lock:
            binding = self._bindings.get(key)
            snapshot = (
                self._snapshots.get(binding.snapshot_hash)
                if binding is not None
                else None
            )
            parent_binding = (
                self._bindings_by_hash.get(binding.parent_binding_hash)
                if binding is not None
                and binding.parent_binding_hash is not None
                else None
            )
        if binding is None or snapshot is None:
            raise PermissionError("typed program binding is missing")
        ledger = self.require_snapshot_ledger(snapshot)
        expected_request = canonical_typed_recipe_selection_request(
            snapshot=snapshot,
            snapshot_ledger=ledger,
            evaluator_epoch=binding.evaluator_epoch,
            selection_round=binding.selection_round,
            excluded_recipe_ids=binding.excluded_recipe_ids,
            parent_program_hash=binding.parent_program_hash,
            parent_recipe_id=binding.parent_recipe_id,
            failed_checks=[
                {"check": check, "reason": reason}
                for check, reason in binding.failed_checks
            ],
            repair_depth=binding.repair_depth,
        )
        if (
            binding.binding_policy != TYPED_PROGRAM_BINDING_VERSION
            or binding.program_id != program.id
            or binding.evaluator_epoch != program.evaluator_epoch
            or binding.program_executable_hash
            != stable_hash(_typed_program_executable_payload(program))
            or binding.program_identity_hash != key
            or snapshot.expected_graph_hash != binding.graph_hash
            or snapshot.expected_model_catalog_hash
            != binding.model_catalog_hash
            or binding.snapshot_ledger_hash
            != ledger.ledger_hash
            or typed_recipe_id_for_program(program, snapshot=snapshot)
            != binding.recipe_id
            or binding.parent_program_id != program.parent_id
            or binding.lineage_program_ids != tuple(program.lineage)
            or binding.request_kind != expected_request["request_kind"]
            or binding.request_hash != stable_hash(expected_request)
            or binding.response_hash
            != stable_hash(
                canonical_typed_recipe_selection_response(binding.recipe_id)
            )
        ):
            raise PermissionError("typed program binding does not validate")
        if binding.parent_binding_hash is None:
            if (
                binding.parent_program_id is not None
                or binding.parent_program_hash is not None
                or binding.parent_recipe_id is not None
                or binding.lineage_program_ids
                or binding.lineage_recipe_ids
                or binding.repair_depth != 0
                or binding.failed_checks
            ):
                raise PermissionError("typed root ancestry binding is malformed")
        elif (
            parent_binding is None
            or parent_binding.binding_hash != binding.parent_binding_hash
            or parent_binding.program_id != binding.parent_program_id
            or parent_binding.snapshot_hash != binding.snapshot_hash
            or parent_binding.recipe_id != binding.parent_recipe_id
            or parent_binding.program_identity_hash
            != binding.parent_program_hash
            or binding.lineage_recipe_ids
            != (*parent_binding.lineage_recipe_ids, parent_binding.recipe_id)
        ):
            raise PermissionError("typed repair parent binding does not validate")
        return binding

    def require_for_snapshot(
        self,
        program: HypothesisProgram,
        snapshot: TypedRecipeSelectionSnapshot,
    ) -> TypedProgramBinding:
        binding = self.require(program)
        if binding.snapshot_hash != snapshot.snapshot_hash:
            raise PermissionError(
                "typed program snapshot binding mismatch"
            )
        return binding

    def lineage_recipe_ids(
        self,
        program: HypothesisProgram,
        *,
        snapshot: TypedRecipeSelectionSnapshot,
    ) -> tuple[str, ...]:
        """Return every bound recipe on the root-to-program typed branch."""

        binding = self.require_for_snapshot(program, snapshot)
        return (*binding.lineage_recipe_ids, binding.recipe_id)

    def matches_registered_snapshot(
        self,
        program: HypothesisProgram,
    ) -> bool:
        """Whether executable fields match any snapshot known to the registry."""

        with self._lock:
            snapshots = tuple(self._snapshots.values())
        return any(
            typed_recipe_id_for_program(program, snapshot=snapshot)
            is not None
            for snapshot in snapshots
        )

    def safe_binding(self, program: HypothesisProgram) -> dict[str, Any]:
        return self.require(program).safe_payload()

    def restore_safe_payload(
        self,
        program: HypothesisProgram,
        binding_payload: Mapping[str, Any],
    ) -> TypedProgramBinding:
        """Restore one persisted binding under the currently frozen ledger.

        Repair parents must be restored first.  Registration reconstructs both
        opaque selector receipts from the sanitized payload, and the exact
        generated safe payload (including its binding hash) must then match the
        persisted row byte-for-structure.  A failed restore leaves no new
        binding in the registry.
        """

        if not isinstance(binding_payload, Mapping):
            raise PermissionError("typed binding restore payload is malformed")
        payload = dict(binding_payload)
        ledger = self.require_snapshot_ledger()
        snapshot_hash = payload.get("snapshot_hash")
        snapshots = tuple(
            row
            for row in ledger.snapshots
            if row.snapshot_hash == snapshot_hash
        )
        if len(snapshots) != 1:
            raise PermissionError(
                "typed binding restore snapshot is outside the frozen ledger"
            )
        snapshot = snapshots[0]

        excluded_recipe_ids = payload.get("excluded_recipe_ids")
        failed_checks = payload.get("failed_checks")
        if not isinstance(excluded_recipe_ids, list) or not isinstance(
            failed_checks, list
        ):
            raise PermissionError("typed binding restore payload is malformed")
        if any(
            not isinstance(row, Mapping)
            or set(row) != {"check", "reason"}
            or not isinstance(row.get("check"), str)
            or not isinstance(row.get("reason"), str)
            for row in failed_checks
        ):
            raise PermissionError(
                "typed binding restore failed-check receipt is malformed"
            )

        parent: HypothesisProgram | None = None
        parent_binding_hash = payload.get("parent_binding_hash")
        if parent_binding_hash is not None:
            if not _is_sha256(parent_binding_hash):
                raise PermissionError(
                    "typed binding restore parent receipt is malformed"
                )
            with self._lock:
                parent = self._programs_by_binding_hash.get(
                    parent_binding_hash
                )
            if parent is None:
                raise PermissionError(
                    "typed binding restore requires its parent first"
                )
            self.require(parent)

        key = _typed_program_binding_key(program)
        with self._lock:
            preexisting = self._bindings.get(key)
        try:
            restored = self.register(
                program,
                snapshot=snapshot,
                recipe_id=payload.get("recipe_id"),
                request_kind=payload.get("request_kind"),
                request_hash=payload.get("request_hash"),
                response_hash=payload.get("response_hash"),
                selection_round=payload.get("selection_round"),
                excluded_recipe_ids=excluded_recipe_ids,
                parent=parent,
                failed_checks=failed_checks,
                repair_depth=payload.get("repair_depth"),
            )
            if restored.safe_payload() != payload:
                raise PermissionError(
                    "typed binding restore payload does not match canonical provenance"
                )
            return self.require(program)
        except Exception:
            if preexisting is None:
                with self._lock:
                    inserted = self._bindings.get(key)
                    if inserted is not None:
                        self._bindings.pop(key, None)
                        if self._binding_keys_by_program_id.get(program.id) == key:
                            self._binding_keys_by_program_id.pop(
                                program.id,
                                None,
                            )
                        self._bindings_by_hash.pop(
                            inserted.binding_hash,
                            None,
                        )
                        self._programs_by_binding_hash.pop(
                            inserted.binding_hash,
                            None,
                        )
                        if not any(
                            row.snapshot_hash == inserted.snapshot_hash
                            for row in self._bindings.values()
                        ):
                            self._snapshots.pop(
                                inserted.snapshot_hash,
                                None,
                            )
            raise

    def binding_count(self) -> int:
        with self._lock:
            return len(self._bindings)


def _typed_program_binding_key(program: HypothesisProgram) -> str:
    return stable_hash(
        {
            "program_id": program.id,
            "evaluator_epoch": program.evaluator_epoch,
            "program_executable_hash": stable_hash(
                _typed_program_executable_payload(program)
            ),
        }
    )


def extract_trial_trace_evidence(
    trace_path: str | Path,
    *,
    containment_root: str | Path,
    trial_id_hash: str,
    family_hash: str,
    trace_hash: str,
    action_budget_receipt_hash: str,
    expected_action_start_count: int,
    expected_action_event_hash: str,
    baseline_success: bool,
    expected_action_budget_limit: int = MAX_TRACE_ACTION_SPANS,
) -> TrialTraceEvidence:
    """Extract every chronological allowlisted command occurrence.

    Raw commands, outputs and model prose are used only transiently for the
    existing allowlisted signature parser.  They never enter the returned
    evidence object.
    """

    from .benchmarks.skilllearn_lifecycle import (
        _allowlisted_action_trace_command,
        _allowlisted_action_trace_root_path,
    )

    path = Path(trace_path)
    root = Path(containment_root).resolve(strict=True)
    if path.is_symlink():
        raise PermissionError("action-span trace cannot be a symlink")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError("action-span trace escaped containment") from exc
    if not resolved.is_file():
        raise PermissionError("action-span trace is not a regular file")

    if not all(
        _is_sha256(value)
        for value in (
            trial_id_hash,
            family_hash,
            trace_hash,
            action_budget_receipt_hash,
            expected_action_event_hash,
        )
    ):
        raise PermissionError("action-span provenance hash is malformed")
    if (
        not isinstance(expected_action_start_count, int)
        or isinstance(expected_action_start_count, bool)
        or expected_action_start_count < 0
        or expected_action_start_count > expected_action_budget_limit
    ):
        raise PermissionError("action-span receipt action count is malformed")
    trace_bytes = resolved.read_bytes()
    if hashlib.sha256(trace_bytes).hexdigest() != trace_hash:
        raise PermissionError("action-span trace hash mismatch")
    if not isinstance(baseline_success, bool):
        raise PermissionError("action-span baseline outcome is malformed")

    started: dict[str, tuple[str, int, str]] = {}
    action_start_ids: set[str] = set()
    action_events: list[dict[str, Any]] = []
    raw_spans: list[ActionSpanEvidence] = []
    changed: dict[str, PrimitiveRef] = {}
    discarded_command_count = 0
    terminal_event_index: int | None = None
    terminal_type = ""
    action_budget_rows: list[tuple[int, str, str, int]] = []

    for event_index, raw_line in enumerate(
        trace_bytes.decode("utf-8", errors="replace").splitlines()
    ):
        try:
            row = json.loads(raw_line)
        except (TypeError, json.JSONDecodeError) as exc:
            if raw_line.strip():
                raise PermissionError(
                    "action-span trace contains malformed JSON"
                ) from exc
            continue
        if not isinstance(row, Mapping):
            continue
        event_type = str(row.get("type") or "")
        if event_type == "assumption.action_budget.started":
            if (
                action_budget_rows
                or action_events
                or terminal_event_index is not None
            ):
                raise PermissionError("action-span budget event is out of order")
            limit = row.get("limit")
            if not isinstance(limit, int) or isinstance(limit, bool):
                raise PermissionError("action-span budget limit is malformed")
            action_budget_rows.append(
                (
                    event_index,
                    str(row.get("policy") or ""),
                    str(row.get("unit") or ""),
                    limit,
                )
            )
            continue
        if event_type == "turn.completed":
            if terminal_event_index is not None:
                raise PermissionError("action-span trace has duplicate terminal")
            terminal_event_index = event_index
            terminal_type = event_type
            continue
        if event_type == "turn.failed":
            if terminal_event_index is not None:
                raise PermissionError("action-span trace has duplicate terminal")
            terminal_event_index = event_index
            terminal_type = event_type
            continue
        if event_type == "item.started":
            if terminal_event_index is not None:
                raise PermissionError("action-span action starts after terminal")
            if not action_budget_rows:
                raise PermissionError("action-span action precedes budget event")
            item = row.get("item")
            if not isinstance(item, Mapping):
                raise PermissionError("action-span action start is malformed")
            item_id = str(item.get("id") or "")
            item_type = str(item.get("type") or "")
            if (
                not item_id
                or not item_type
                or item_id in action_start_ids
            ):
                raise PermissionError(
                    "action-span action ID is missing or duplicate"
                )
            action_start_ids.add(item_id)
            action_events.append(
                {
                    "event_index": len(action_events) + 1,
                    "item_id": item_id,
                    "item_type": item_type,
                    "malformed": False,
                }
            )
            if len(action_events) > expected_action_budget_limit:
                raise PermissionError(
                    "action-span trace exceeds frozen action budget"
                )
            relevant_action = item_type in {
                "command_execution",
                "file_change",
            }
            if not relevant_action:
                continue
            command_identity = (
                stable_hash({"command": str(item.get("command") or "")})
                if item_type == "command_execution"
                else ""
            )
            started[item_id] = (item_type, event_index, command_identity)
            continue
        item = row.get("item")
        if not isinstance(item, Mapping):
            continue
        item_id = str(item.get("id") or "")
        item_type = str(item.get("type") or "")
        relevant_action = item_type in {"command_execution", "file_change"}
        if event_type != "item.completed" or not relevant_action:
            continue
        if terminal_event_index is not None:
            raise PermissionError("action-span action completes after terminal")
        if not action_budget_rows:
            raise PermissionError("action-span action precedes budget event")
        started_row = started.pop(item_id, None)
        if started_row is None or started_row[0] != item_type:
            raise PermissionError("action-span action start/completion is unpaired")
        start_event_index = started_row[1]
        if item_type == "command_execution" and started_row[2] != stable_hash(
            {"command": str(item.get("command") or "")}
        ):
            raise PermissionError("action-span command changed before completion")
        if item_type == "file_change":
            changes = item.get("changes")
            if not isinstance(changes, list):
                continue
            for change in changes:
                if not isinstance(change, Mapping):
                    continue
                safe = _allowlisted_action_trace_root_path(
                    str(change.get("path") or "")
                )
                if safe:
                    ref = PrimitiveRef("artifact", safe)
                    changed.setdefault(ref.primitive_id, ref)
            continue
        if item_type != "command_execution":
            continue
        signature = _allowlisted_action_trace_command(
            str(item.get("command") or "")
        )
        if signature is None:
            discarded_command_count += 1
            continue
        outcome = _span_outcome(item.get("status"), item.get("exit_code"))
        executable = PrimitiveRef(
            "executable", str(signature["executable_basename"])
        )
        flags = tuple(
            PrimitiveRef("flag", str(value))
            for value in signature.get("safe_flags", [])
        )
        artifacts = tuple(
            PrimitiveRef("artifact", str(value))
            for value in signature.get("task_local_paths", [])
        )
        exact_hash = str(signature["original_command_hash"])
        signature_hash = stable_hash(
            {
                "exact_command_hash": exact_hash,
                "executable_id": executable.primitive_id,
                "flag_ids": [row.primitive_id for row in flags],
                "artifact_ids": [row.primitive_id for row in artifacts],
            }
        )
        raw_spans.append(
            ActionSpanEvidence(
                trial_id_hash=trial_id_hash,
                family_hash=family_hash,
                span_index=len(raw_spans),
                start_event_index=start_event_index,
                completion_event_index=event_index,
                exact_command_hash=exact_hash,
                command_signature_hash=signature_hash,
                outcome=outcome,
                executable=executable,
                flags=flags,
                artifacts=artifacts,
                baseline_success=baseline_success,
                trace_complete=False,
            )
        )

    if len(action_budget_rows) != 1 or action_budget_rows[0][1:] != (
        "codex_jsonl_action_start_budget_v1",
        "codex_action_start_v1",
        expected_action_budget_limit,
    ):
        raise PermissionError("action-span trace budget identity mismatch")
    if started:
        raise PermissionError("action-span trace has unclosed actions")
    if terminal_event_index is None or terminal_type != "turn.completed":
        raise PermissionError("action-span trace terminal is incomplete")
    if len(raw_spans) > expected_action_budget_limit:
        raise PermissionError("action-span trace exceeds frozen action budget")
    action_event_hash = stable_hash(action_events)
    if (
        len(action_events) != expected_action_start_count
        or action_event_hash != expected_action_event_hash
    ):
        raise PermissionError(
            "action-span trace does not match action-budget receipt"
        )
    trace_complete = True
    spans = [replace(row, trace_complete=trace_complete) for row in raw_spans]
    enriched: list[ActionSpanEvidence] = []
    for index, span in enumerate(spans):
        later = spans[index + 1 :]
        exact = tuple(
            row.span_id
            for row in later
            if row.outcome is SpanOutcome.SUCCEEDED
            and row.exact_command_hash == span.exact_command_hash
        )
        same_executable = tuple(
            row.span_id
            for row in later
            if row.outcome is SpanOutcome.SUCCEEDED
            and row.executable.primitive_id == span.executable.primitive_id
        )
        artifact_ids = {row.primitive_id for row in span.artifacts}
        shared_artifact = tuple(
            row.span_id
            for row in later
            if row.outcome is SpanOutcome.SUCCEEDED
            and artifact_ids.intersection(
                item.primitive_id for item in row.artifacts
            )
        )
        enriched.append(
            replace(
                span,
                is_last_allowlisted_span=index == len(spans) - 1,
                later_exact_success_span_hashes=exact,
                later_same_executable_success_span_hashes=same_executable,
                later_shared_artifact_success_span_hashes=shared_artifact,
            )
        )
    return TrialTraceEvidence(
        trial_id_hash=trial_id_hash,
        family_hash=family_hash,
        trace_hash=trace_hash,
        action_budget_receipt_hash=action_budget_receipt_hash,
        action_event_hash=action_event_hash,
        baseline_success=baseline_success,
        action_budget_limit=expected_action_budget_limit,
        trace_complete=trace_complete,
        action_start_count=len(action_events),
        command_span_count=len(enriched),
        discarded_command_count=discarded_command_count,
        changed_artifacts=tuple(
            changed[key] for key in sorted(changed)
        ),
        spans=tuple(enriched),
    )


def assess_observed_primitives(
    trials: Sequence[TrialTraceEvidence],
) -> tuple[PrimitiveAssessment, ...]:
    spans_by_trial = {row.trial_id_hash: row.spans for row in trials}
    trials_by_id = {row.trial_id_hash: row for row in trials}
    scopes: dict[tuple[str, str], list[ActionSpanEvidence]] = {}
    for trial in trials:
        for span in trial.spans:
            scopes.setdefault(
                ("exact_command", span.exact_command_hash), []
            ).append(span)
            scopes.setdefault(
                ("executable", span.executable.primitive_id), []
            ).append(span)
            for artifact in span.artifacts:
                scopes.setdefault(
                    ("artifact", artifact.primitive_id), []
                ).append(span)
    rows: list[PrimitiveAssessment] = []
    for (scope, primitive_id), spans in sorted(scopes.items()):
        failures = [row for row in spans if row.outcome is SpanOutcome.FAILED]
        successes = [
            row for row in spans if row.outcome is SpanOutcome.SUCCEEDED
        ]
        recovery_count = sum(
            any(
                later.completion_event_index > failure.completion_event_index
                and later.outcome is SpanOutcome.SUCCEEDED
                and _span_has_scoped_primitive(
                    later,
                    scope=scope,
                    primitive_id=primitive_id,
                )
                for later in spans_by_trial[failure.trial_id_hash]
            )
            for failure in failures
        )
        independent_tasks = {row.trial_id_hash for row in spans}
        complete_traces = {row.trial_id_hash for row in spans if row.trace_complete}
        successful_trials = {
            row.trial_id_hash for row in spans if row.baseline_success
        }
        if failures and recovery_count > 0:
            classification = PrimitiveAssessmentClass.RECOVERED_AFTER_FAILURE
        elif successes:
            classification = PrimitiveAssessmentClass.OPERATIONALLY_OBSERVED
        elif (
            scope == "exact_command"
            and len({row.trial_id_hash for row in failures}) >= 2
            and failures
            and all(row.trace_complete for row in failures)
            and all(row.is_last_allowlisted_span for row in failures)
            and all(
                trials_by_id[row.trial_id_hash].discarded_command_count == 0
                for row in failures
            )
            and recovery_count == 0
        ):
            classification = (
                PrimitiveAssessmentClass.DO_NOT_RECOMMEND_EXACT_SIGNATURE
            )
        elif failures:
            classification = PrimitiveAssessmentClass.FAILURE_COOCCURRENCE_ONLY
        else:
            classification = PrimitiveAssessmentClass.INSUFFICIENT
        rows.append(
            PrimitiveAssessment(
                scope=scope,
                primitive_id=primitive_id,
                classification=classification,
                failure_span_count=len(failures),
                success_span_count=len(successes),
                independent_train_task_count=len(independent_tasks),
                recovery_contradiction_count=recovery_count,
                successful_trial_cooccurrence_count=len(successful_trials),
                complete_trace_count=len(complete_traces),
                evidence_span_set_hash=stable_hash(
                    {"span_hashes": sorted(row.span_hash for row in spans)}
                ),
            )
        )
    return tuple(rows)


def _span_has_scoped_primitive(
    span: ActionSpanEvidence,
    *,
    scope: str,
    primitive_id: str,
) -> bool:
    if scope == "exact_command":
        return span.exact_command_hash == primitive_id
    if scope == "executable":
        return span.executable.primitive_id == primitive_id
    if scope == "artifact":
        return any(row.primitive_id == primitive_id for row in span.artifacts)
    raise ValueError(f"unknown primitive assessment scope: {scope}")


def build_family_capability_graph(
    *,
    target_family: str,
    failures: Sequence[ResidualExample],
    action_profiles: Mapping[str, Mapping[str, Any]],
    trial_evidence: Mapping[str, TrialTraceEvidence],
    minimum_support: int = 2,
    maximum_artifacts: int = MAX_REGISTERED_ARTIFACTS_PER_FAMILY,
) -> FamilyCapabilityGraph:
    if minimum_support <= 0 or maximum_artifacts <= 0:
        raise ValueError("typed graph bounds must be positive")
    if any(key != row.trial_id_hash for key, row in trial_evidence.items()):
        raise PermissionError("typed graph trial-evidence key mismatch")
    family_failures = [
        row
        for row in failures
        if not row.baseline_success and row.family == target_family
    ]
    if len(family_failures) < minimum_support:
        raise TypedGraphUnavailableError("insufficient_family_support")
    expected_family_hash = stable_hash({"family": target_family})
    relation_by_locator: dict[str, set[str]] = {}
    support_by_locator: dict[str, set[str]] = {}
    provenance_by_locator: dict[str, set[str]] = {}
    for residual in family_failures:
        profile_hash = str(
            residual.context.get("action_context_profile_hash") or ""
        )
        profile = action_profiles.get(profile_hash)
        if not isinstance(profile, Mapping):
            raise PermissionError("typed graph action profile is missing")
        if stable_hash(dict(profile)) != profile_hash:
            raise PermissionError("typed graph action profile hash mismatch")
        observations = _profile_artifacts(profile)
        trial_hash = stable_hash({"item_id": residual.task_id})
        trial = trial_evidence.get(trial_hash)
        if trial is None:
            raise PermissionError("typed graph trial evidence is missing")
        if (
            trial.family_hash != expected_family_hash
            or trial.baseline_success is not residual.baseline_success
            or not trial.trace_complete
        ):
            raise PermissionError("typed graph trial evidence binding mismatch")
        for span in trial.spans:
            relation = (
                "successful_command_span"
                if span.outcome is SpanOutcome.SUCCEEDED
                else (
                    "failed_command_span_cooccurrence"
                    if span.outcome is SpanOutcome.FAILED
                    else "unknown_command_span_cooccurrence"
                )
            )
            for artifact in span.artifacts:
                observations.setdefault(artifact.value, set()).add(
                    relation
                )
        for artifact in trial.changed_artifacts:
            observations.setdefault(artifact.value, set()).add(
                "observed_file_change"
            )
        for locator, relations in observations.items():
            safe = _canonical_task_locator(locator)
            if safe is None:
                continue
            relation_by_locator.setdefault(safe, set()).update(relations)
            support_by_locator.setdefault(safe, set()).add(trial_hash)
            provenance_by_locator.setdefault(safe, set()).add(profile_hash)
            provenance_by_locator[safe].add(trial.evidence_hash)

    candidates: list[ArtifactSpec] = []
    for locator in sorted(support_by_locator):
        support = len(support_by_locator[locator])
        if support < minimum_support:
            continue
        fmt = _artifact_format(locator)
        provenance_hash = stable_hash(
            {
                "profile_or_trace_hashes": sorted(
                    provenance_by_locator[locator]
                ),
                "supporting_trial_hashes": sorted(
                    support_by_locator[locator]
                ),
                "relations": sorted(relation_by_locator[locator]),
            }
        )
        relations = tuple(sorted(relation_by_locator[locator]))
        artifact_id = _artifact_id(
            target_family=target_family,
            locator=locator,
            artifact_format=fmt,
            support_count=support,
            evidence_relations=relations,
            provenance_hash=provenance_hash,
        )
        candidates.append(
            ArtifactSpec(
                artifact_id=artifact_id,
                locator=locator,
                format=fmt,
                support_count=support,
                evidence_relations=relations,
                provenance_hash=provenance_hash,
            )
        )
    candidates.sort(key=_artifact_rank)
    artifacts = tuple(candidates[:maximum_artifacts])
    if not artifacts:
        raise TypedGraphUnavailableError("no_supported_task_local_artifact")
    capabilities: list[CapabilitySpec] = []
    recipes: list[TypedRecipe] = []
    for artifact in artifacts:
        capability = _capability_spec(
            target_family=target_family,
            artifact=artifact,
        )
        capabilities.append(capability)
        for workflow in _workflows_for_format(artifact.format):
            recipes.append(
                _build_recipe(
                    target_family=target_family,
                    artifact=artifact,
                    capability=capability,
                    workflow=workflow,
                )
            )
    source_evidence_hash = stable_hash(
        {
            "target_family_hash": stable_hash({"family": target_family}),
            "failure_transition_hashes": sorted(
                stable_hash({"transition_id": row.transition_id})
                for row in family_failures
            ),
            "artifact_provenance_hashes": [
                row.provenance_hash for row in artifacts
            ],
        }
    )
    graph = FamilyCapabilityGraph(
        target_family=target_family,
        source_evidence_hash=source_evidence_hash,
        artifacts=artifacts,
        capabilities=tuple(capabilities),
        recipes=tuple(sorted(recipes, key=lambda row: row.recipe_id)),
    )
    issues = graph.validate()
    if issues:
        raise PermissionError(f"typed capability graph invalid: {list(issues)}")
    return graph


def selection_schema(graph: FamilyCapabilityGraph) -> dict[str, Any]:
    issues = graph.validate()
    if issues:
        raise PermissionError(
            f"typed recipe graph is invalid: {list(issues)}"
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["recipe_id"],
        "properties": {
            "recipe_id": {
                "type": "string",
                "enum": sorted(row.recipe_id for row in graph.recipes),
            }
        },
    }


def materialize_recipe_selection(
    selection: Mapping[str, Any],
    *,
    graph: FamilyCapabilityGraph,
    evaluator_epoch: str,
    expected_graph_hash: str,
    expected_model_catalog_hash: str,
) -> HypothesisProgram:
    graph_issues = graph.validate()
    if graph_issues:
        raise PermissionError(
            f"typed recipe graph is invalid: {list(graph_issues)}"
        )
    if (
        not _is_sha256(expected_graph_hash)
        or graph.graph_hash != expected_graph_hash
        or not _is_sha256(expected_model_catalog_hash)
        or stable_hash(graph.model_catalog()) != expected_model_catalog_hash
    ):
        raise PermissionError(
            "typed recipe graph does not match frozen selection snapshot"
        )
    if set(selection) != {"recipe_id"}:
        raise PermissionError("typed recipe selection fields are not closed")
    recipe_id = selection.get("recipe_id")
    if not isinstance(recipe_id, str):
        raise PermissionError("typed recipe selection ID is malformed")
    matches = [row for row in graph.recipes if row.recipe_id == recipe_id]
    if len(matches) != 1:
        raise PermissionError("typed recipe selection references unknown ID")
    recipe = matches[0]
    artifacts = {row.artifact_id: row for row in graph.artifacts}
    capabilities = {row.capability_id: row for row in graph.capabilities}
    artifact = artifacts[recipe.primary_artifact_id]
    capability = capabilities[recipe.capability_id]
    actions = tuple(
        _lower_operator_node(
            node,
            artifact=artifact,
            capability=capability,
        )
        for node in recipe.nodes
    )
    program = HypothesisProgram(
        id="typed-policy-" + stable_hash(
            {
                "graph_hash": graph.graph_hash,
                "recipe_id": recipe.recipe_id,
            }
        )[:18],
        kind=HypothesisKind.POLICY,
        statement=TYPED_PROGRAM_STATEMENT,
        trigger=TriggerSpec(
            all_of=(
                FeaturePredicate(
                    key="family", op="eq", value=graph.target_family
                ),
            )
        ),
        anti_trigger=TriggerSpec(),
        action_graph=actions,
        expected_effect=ExpectedEffect(),
        verifier=VerifierContract(
            checks=TYPED_PROGRAM_VERIFIER_CHECKS,
            required_evidence=(),
            anchor_id=TYPED_PROGRAM_VERIFIER_ANCHOR,
        ),
        evaluator_epoch=evaluator_epoch,
    )
    issues = program.validate()
    if issues:
        raise PermissionError(
            f"typed recipe materialized invalid program: {issues}"
        )
    return program


def is_typed_recipe_materialization(
    program: HypothesisProgram,
) -> bool:
    """Recognize the immutable harness-owned typed program envelope."""

    return bool(
        program.kind is HypothesisKind.POLICY
        and program.statement == TYPED_PROGRAM_STATEMENT
        and program.verifier.checks == TYPED_PROGRAM_VERIFIER_CHECKS
        and program.verifier.anchor_id == TYPED_PROGRAM_VERIFIER_ANCHOR
    )


def typed_recipe_id_for_program(
    program: HypothesisProgram,
    *,
    snapshot: TypedRecipeSelectionSnapshot,
) -> str | None:
    """Return the unique frozen recipe whose harness lowering produced a program.

    Harness-owned lineage and status fields are intentionally ignored so the
    same check binds both root candidates and recursively repaired children.
    Every executable trigger/action/verifier field remains part of the match.
    """

    if snapshot.validate():
        return None
    candidate = _typed_program_executable_payload(program)
    matches: list[str] = []
    for recipe in snapshot.graph.recipes:
        expected = materialize_recipe_selection(
            {"recipe_id": recipe.recipe_id},
            graph=snapshot.graph,
            evaluator_epoch=program.evaluator_epoch,
            expected_graph_hash=snapshot.expected_graph_hash,
            expected_model_catalog_hash=(
                snapshot.expected_model_catalog_hash
            ),
        )
        if _typed_program_executable_payload(expected) == candidate:
            matches.append(recipe.recipe_id)
    return matches[0] if len(matches) == 1 else None


def typed_program_snapshot_binding(
    program: HypothesisProgram,
    snapshots: Sequence[TypedRecipeSelectionSnapshot],
) -> tuple[TypedRecipeSelectionSnapshot, str] | None:
    matches: list[tuple[TypedRecipeSelectionSnapshot, str]] = []
    for snapshot in snapshots:
        recipe_id = typed_recipe_id_for_program(
            program,
            snapshot=snapshot,
        )
        if recipe_id is not None:
            matches.append((snapshot, recipe_id))
    return matches[0] if len(matches) == 1 else None


def _typed_program_executable_payload(
    program: HypothesisProgram,
) -> dict[str, Any]:
    payload = program.to_dict()
    for key in (
        "id",
        "parent_id",
        "lineage",
        "status",
        "created_from_transition_ids",
    ):
        payload.pop(key, None)
    return payload


def canonical_recipe(graph: FamilyCapabilityGraph) -> TypedRecipe:
    graph_issues = graph.validate()
    if graph_issues:
        raise PermissionError(
            f"typed recipe graph is invalid: {list(graph_issues)}"
        )
    if not graph.recipes:
        raise ValueError("typed graph has no recipe")
    workflow_priority = {
        WorkflowKind.BUILD_VISUALIZATION: 0,
        WorkflowKind.CONFIGURE_AND_RUN: 1,
        WorkflowKind.ORGANIZE_COLLECTION: 2,
        WorkflowKind.DERIVE_TASK_OUTPUT: 3,
        WorkflowKind.TRANSFORM_IN_PLACE: 4,
    }
    artifacts = {row.artifact_id: row for row in graph.artifacts}
    return min(
        graph.recipes,
        key=lambda row: (
            workflow_priority[row.workflow],
            _artifact_rank(artifacts[row.primary_artifact_id]),
            row.recipe_id,
        ),
    )


def _span_outcome(status: Any, exit_code: Any) -> SpanOutcome:
    normalized = str(status or "").strip().lower()
    if normalized in {"failed", "failure", "error", "timed_out", "timeout"}:
        return SpanOutcome.FAILED
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        return SpanOutcome.SUCCEEDED if exit_code == 0 else SpanOutcome.FAILED
    if normalized in {"completed", "complete", "succeeded", "success", "ok"}:
        return SpanOutcome.SUCCEEDED
    return SpanOutcome.UNKNOWN


def _profile_artifacts(
    profile: Mapping[str, Any],
) -> dict[str, set[str]]:
    observations: dict[str, set[str]] = {}
    environment = profile.get("runtime_environment")
    if isinstance(environment, Mapping):
        fields = {
            "declared_task_local_paths": "declared_task_local_path",
            "copied_task_files": "copied_task_artifact",
            "environment_source_files": "environment_source_artifact",
        }
        for field, relation in fields.items():
            values = environment.get(field)
            if not isinstance(values, (list, tuple)):
                continue
            for value in values:
                locator = str(value or "").strip()
                if locator and not locator.startswith("/"):
                    locator = "/root/" + locator.lstrip("/")
                if locator:
                    observations.setdefault(locator, set()).add(relation)
    return observations


def _canonical_task_locator(value: str) -> str | None:
    from .benchmarks.skilllearn_lifecycle import (
        _allowlisted_action_trace_root_path,
    )

    candidate = str(value or "").strip().rstrip(".,;:)]}\"'")
    if not candidate or len(candidate) > 300:
        return None
    if not _LOCAL_PATH.fullmatch(candidate):
        return None
    if _allowlisted_action_trace_root_path(candidate) is None:
        return None
    raw_components = candidate.split("/")[2:]
    if any(part in {"", ".", ".."} for part in raw_components):
        return None
    parts = PurePosixPath(candidate).parts
    if not parts or parts[0] != "/" or parts[1] != "root":
        return None
    components = parts[2:]
    lowered = {part.lower() for part in components}
    lowered_stems = {
        PurePosixPath(part.lower()).stem for part in components
    }
    if (lowered | lowered_stems).intersection(_FORBIDDEN_PATH_PARTS):
        return None
    return candidate


def _artifact_format(locator: str) -> ArtifactFormat:
    name = PurePosixPath(locator).name.lower()
    suffix = PurePosixPath(locator).suffix.lower()
    if suffix in {".csv", ".tsv", ".xls", ".xlsx"}:
        return ArtifactFormat.TABULAR
    if suffix in {".json", ".jsonl", ".yaml", ".yml", ".xml"}:
        return ArtifactFormat.STRUCTURED_RECORD
    if suffix in {".docx", ".pptx", ".odt", ".odp"}:
        return ArtifactFormat.OFFICE_DOCUMENT
    if suffix == ".pdf":
        return ArtifactFormat.PDF
    if suffix in {".nc", ".h5", ".hdf5", ".npy", ".npz"}:
        return ArtifactFormat.SCIENTIFIC_ARRAY
    if suffix in {".toml", ".ini", ".cfg", ".conf", ".nml"}:
        return ArtifactFormat.CONFIGURATION
    if suffix in {".html", ".htm", ".css", ".js", ".svg"}:
        return ArtifactFormat.WEB_ASSET
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff"}:
        return ArtifactFormat.IMAGE
    if suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return ArtifactFormat.VIDEO
    if suffix in {".zip", ".tar", ".gz", ".tgz", ".bz2"}:
        return ArtifactFormat.ARCHIVE
    if suffix in {".txt", ".md", ".log"}:
        return ArtifactFormat.TEXT
    if not suffix or name in {"data", "output", "papers", "bcs"}:
        return ArtifactFormat.DIRECTORY
    return ArtifactFormat.GENERIC


def _capability_for_format(fmt: ArtifactFormat) -> CapabilityKind:
    mapping = {
        ArtifactFormat.TABULAR: CapabilityKind.TABULAR_DATA,
        ArtifactFormat.STRUCTURED_RECORD: CapabilityKind.STRUCTURED_RECORD,
        ArtifactFormat.OFFICE_DOCUMENT: CapabilityKind.OFFICE_DOCUMENT,
        ArtifactFormat.PDF: CapabilityKind.PDF_DOCUMENT,
        ArtifactFormat.SCIENTIFIC_ARRAY: CapabilityKind.SCIENTIFIC_DATA,
        ArtifactFormat.CONFIGURATION: CapabilityKind.CONFIGURATION,
        ArtifactFormat.WEB_ASSET: CapabilityKind.WEB_ASSET,
        ArtifactFormat.IMAGE: CapabilityKind.IMAGE_MEDIA,
        ArtifactFormat.VIDEO: CapabilityKind.VIDEO_MEDIA,
        ArtifactFormat.ARCHIVE: CapabilityKind.ARCHIVE,
        ArtifactFormat.DIRECTORY: CapabilityKind.FILESYSTEM_COLLECTION,
        ArtifactFormat.TEXT: CapabilityKind.TEXT_DOCUMENT,
        ArtifactFormat.GENERIC: CapabilityKind.GENERIC_LOCAL_ARTIFACT,
    }
    return mapping[fmt]


def _artifact_id(
    *,
    target_family: str,
    locator: str,
    artifact_format: ArtifactFormat,
    support_count: int,
    evidence_relations: Sequence[str],
    provenance_hash: str,
) -> str:
    return "artifact_" + stable_hash(
        {
            "family": target_family,
            "locator": locator,
            "format": artifact_format.value,
            "support_count": support_count,
            "evidence_relations": list(evidence_relations),
            "provenance_hash": provenance_hash,
        }
    )[:20]


def _capability_spec(
    *,
    target_family: str,
    artifact: ArtifactSpec,
) -> CapabilitySpec:
    kind = _capability_for_format(artifact.format)
    capability_id = "capability_" + stable_hash(
        {
            "family": target_family,
            "artifact_id": artifact.artifact_id,
            "kind": kind.value,
            "artifact_provenance_hash": artifact.provenance_hash,
        }
    )[:20]
    return CapabilitySpec(
        capability_id=capability_id,
        kind=kind,
        artifact_format=artifact.format,
        provenance_hash=artifact.provenance_hash,
    )


def _artifact_rank(artifact: ArtifactSpec) -> tuple[Any, ...]:
    relation_priority = min(
        (
            {
                "observed_file_change": 0,
                "copied_task_artifact": 1,
                "declared_task_local_path": 2,
                "successful_command_span": 3,
                "environment_source_artifact": 4,
                "failed_command_span_cooccurrence": 5,
                "unknown_command_span_cooccurrence": 6,
            }.get(relation, 9)
            for relation in artifact.evidence_relations
        ),
        default=9,
    )
    generic_penalty = int(
        artifact.format in {ArtifactFormat.GENERIC, ArtifactFormat.DIRECTORY}
    )
    return (
        -artifact.support_count,
        generic_penalty,
        relation_priority,
        stable_hash({"locator": artifact.locator}),
    )


def _workflows_for_format(fmt: ArtifactFormat) -> tuple[WorkflowKind, ...]:
    workflows = [
        WorkflowKind.DERIVE_TASK_OUTPUT,
        WorkflowKind.TRANSFORM_IN_PLACE,
    ]
    if fmt in {ArtifactFormat.DIRECTORY, ArtifactFormat.OFFICE_DOCUMENT}:
        workflows.append(WorkflowKind.ORGANIZE_COLLECTION)
    if fmt is ArtifactFormat.TABULAR:
        workflows.append(WorkflowKind.BUILD_VISUALIZATION)
    if fmt in {ArtifactFormat.CONFIGURATION, ArtifactFormat.SCIENTIFIC_ARRAY}:
        workflows.append(WorkflowKind.CONFIGURE_AND_RUN)
    return tuple(workflows)


def _build_recipe(
    *,
    target_family: str,
    artifact: ArtifactSpec,
    capability: CapabilitySpec,
    workflow: WorkflowKind,
) -> TypedRecipe:
    operator_sequences: Mapping[
        WorkflowKind, tuple[tuple[OperatorKind, str, str], ...]
    ] = {
        WorkflowKind.DERIVE_TASK_OUTPUT: (
            (OperatorKind.READ_REGISTERED_ARTIFACT, "artifact_ref", "bytes"),
            (
                OperatorKind.PARSE_WITH_REGISTERED_CAPABILITY,
                "bytes",
                "typed_content",
            ),
            (OperatorKind.DERIVE_TASK_DELTA, "typed_content", "typed_content"),
            (
                OperatorKind.SERIALIZE_WITH_REGISTERED_CAPABILITY,
                "typed_content",
                "bytes",
            ),
            (OperatorKind.WRITE_TASK_DECLARED_OUTPUT, "bytes", "artifact_mutation"),
            (OperatorKind.CHECK_TASK_LOCAL_RESULT, "artifact_mutation", "check_result"),
        ),
        WorkflowKind.TRANSFORM_IN_PLACE: (
            (OperatorKind.READ_REGISTERED_ARTIFACT, "artifact_ref", "bytes"),
            (
                OperatorKind.PARSE_WITH_REGISTERED_CAPABILITY,
                "bytes",
                "typed_content",
            ),
            (OperatorKind.DERIVE_TASK_DELTA, "typed_content", "typed_content"),
            (
                OperatorKind.SERIALIZE_WITH_REGISTERED_CAPABILITY,
                "typed_content",
                "bytes",
            ),
            (OperatorKind.WRITE_BACK_REGISTERED_ARTIFACT, "bytes", "artifact_mutation"),
            (OperatorKind.CHECK_TASK_LOCAL_RESULT, "artifact_mutation", "check_result"),
        ),
        WorkflowKind.ORGANIZE_COLLECTION: (
            (OperatorKind.INSPECT_REGISTERED_METADATA, "artifact_ref", "metadata"),
            (OperatorKind.DERIVE_ORGANIZATION_PLAN, "metadata", "organization_plan"),
            (OperatorKind.MOVE_WITHIN_TASK_ROOT, "organization_plan", "artifact_mutation"),
            (OperatorKind.CHECK_TASK_LOCAL_RESULT, "artifact_mutation", "check_result"),
        ),
        WorkflowKind.BUILD_VISUALIZATION: (
            (OperatorKind.READ_REGISTERED_ARTIFACT, "artifact_ref", "bytes"),
            (
                OperatorKind.PARSE_WITH_REGISTERED_CAPABILITY,
                "bytes",
                "typed_content",
            ),
            (OperatorKind.DERIVE_VISUALIZATION_SPEC, "typed_content", "visualization_spec"),
            (OperatorKind.RENDER_LOCAL_VISUALIZATION, "visualization_spec", "artifact_mutation"),
            (OperatorKind.CHECK_TASK_LOCAL_RESULT, "artifact_mutation", "check_result"),
        ),
        WorkflowKind.CONFIGURE_AND_RUN: (
            (OperatorKind.READ_REGISTERED_ARTIFACT, "artifact_ref", "bytes"),
            (
                OperatorKind.PARSE_WITH_REGISTERED_CAPABILITY,
                "bytes",
                "typed_content",
            ),
            (OperatorKind.DERIVE_TASK_DELTA, "typed_content", "typed_content"),
            (
                OperatorKind.SERIALIZE_WITH_REGISTERED_CAPABILITY,
                "typed_content",
                "bytes",
            ),
            (OperatorKind.WRITE_BACK_REGISTERED_ARTIFACT, "bytes", "artifact_mutation"),
            (
                OperatorKind.INVOKE_REGISTERED_LOCAL_CAPABILITY,
                "artifact_mutation",
                "generated_output",
            ),
            (OperatorKind.INSPECT_GENERATED_OUTPUT, "generated_output", "check_result"),
        ),
    }
    recipe_id = "recipe_" + stable_hash(
        {
            "family": target_family,
            "artifact_id": artifact.artifact_id,
            "capability_id": capability.capability_id,
            "workflow": workflow.value,
            "artifact_provenance_hash": artifact.provenance_hash,
            "capability_provenance_hash": capability.provenance_hash,
        }
    )[:20]
    nodes: list[TypedOperatorNode] = []
    previous: str | None = None
    for index, (kind, input_type, output_type) in enumerate(
        operator_sequences[workflow]
    ):
        node_id = f"{recipe_id}_n{index + 1}"
        nodes.append(
            TypedOperatorNode(
                node_id=node_id,
                kind=kind,
                artifact_id=artifact.artifact_id,
                capability_id=capability.capability_id,
                input_type=input_type,
                output_type=output_type,
                depends_on=(previous,) if previous else (),
            )
        )
        previous = node_id
    return TypedRecipe(
        recipe_id=recipe_id,
        workflow=workflow,
        primary_artifact_id=artifact.artifact_id,
        capability_id=capability.capability_id,
        nodes=tuple(nodes),
    )


def _lower_operator_node(
    node: TypedOperatorNode,
    *,
    artifact: ArtifactSpec,
    capability: CapabilitySpec,
) -> ActionNode:
    locator = artifact.locator
    capability_name = capability.kind.value.replace("_", " ")
    templates = {
        OperatorKind.READ_REGISTERED_ARTIFACT: (
            f"Read the registered task-local artifact at {locator}."
        ),
        OperatorKind.PARSE_WITH_REGISTERED_CAPABILITY: (
            f"Parse the registered artifact as {artifact.format.value} using "
            f"an available task-local {capability_name} method."
        ),
        OperatorKind.DERIVE_TASK_DELTA: (
            "Derive the concrete content change required by the current task "
            "from the parsed artifact; do not substitute a generic check."
        ),
        OperatorKind.SERIALIZE_WITH_REGISTERED_CAPABILITY: (
            "Serialize the changed content in the task-required artifact format."
        ),
        OperatorKind.WRITE_BACK_REGISTERED_ARTIFACT: (
            f"Write the serialized task-required result back within {locator}."
        ),
        OperatorKind.WRITE_TASK_DECLARED_OUTPUT: (
            "Write the result to the output artifact declared by the current task."
        ),
        OperatorKind.INSPECT_REGISTERED_METADATA: (
            f"Inspect task-local metadata for the registered artifact at {locator}."
        ),
        OperatorKind.DERIVE_ORGANIZATION_PLAN: (
            "Derive a deterministic organization plan from task-local metadata "
            "and the current task requirements."
        ),
        OperatorKind.MOVE_WITHIN_TASK_ROOT: (
            "Apply the organization plan using only moves within the task root."
        ),
        OperatorKind.DERIVE_VISUALIZATION_SPEC: (
            "Derive the required visual encodings and interactions from the "
            "parsed table and current task requirements."
        ),
        OperatorKind.RENDER_LOCAL_VISUALIZATION: (
            "Render a self-contained task-local visualization artifact without "
            "network fetches."
        ),
        OperatorKind.INVOKE_REGISTERED_LOCAL_CAPABILITY: (
            f"Use an available task-local {capability_name} method on the "
            "updated artifact without installing packages or using the network."
        ),
        OperatorKind.INSPECT_GENERATED_OUTPUT: (
            "Inspect the generated task-local output for structural completeness."
        ),
        OperatorKind.CHECK_TASK_LOCAL_RESULT: (
            "Check the resulting task-local artifacts for the required structure "
            "before completion."
        ),
    }
    operation = (
        "check_condition"
        if node.kind
        in {
            OperatorKind.CHECK_TASK_LOCAL_RESULT,
            OperatorKind.INSPECT_GENERATED_OUTPUT,
        }
        else "execute_step"
    )
    return ActionNode(
        id=node.node_id,
        operation=operation,
        target=f"typed operator {node.kind.value}",
        value=templates[node.kind],
        depends_on=node.depends_on,
    )


def _has_node_cycle(nodes: Iterable[TypedOperatorNode]) -> bool:
    by_id = {row.node_id: row for row in nodes}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        visiting.add(node_id)
        for dependency in by_id[node_id].depends_on:
            if dependency in by_id and visit(dependency):
                return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    return any(visit(node_id) for node_id in by_id)


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
