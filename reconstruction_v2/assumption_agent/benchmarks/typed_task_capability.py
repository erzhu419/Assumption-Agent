from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from ..models import stable_hash
from ..typed_execution_contract import (
    TypedExecutionContract,
    load_typed_execution_contract,
)
from ..typed_operator_grammar import (
    ArtifactFormat,
    BoundTypedRecipe,
    CapabilityKind,
    FamilyCapabilityGraph,
    OperatorKind,
)


PORTABLE_ARTIFACT_ROLE_POLICY_VERSION = (
    "current_item_public_instruction_primary_input_v1"
)
OFFICE_COLLECTION_ROLE_POLICY_VERSION = (
    "current_task_runtime_root_office_collection_discovery_v1"
)
RESTRICTED_TASK_CAPABILITY_POLICY_VERSION = (
    "harness_owned_readonly_delimited_table_profile_v1"
)
TASK_CAPABILITY_EFFECT_RECEIPT_VERSION = (
    "exact_input_pre_post_and_task_local_output_effect_v1"
)
PORTABLE_TASK_CAPABILITY_COMPILER_VERSION = (
    "receipt_bound_portable_task_capability_compiler_v1"
)
PORTABLE_TASK_CAPABILITY_METADATA_VERSION = (
    "harness_owned_pre_agent_task_capability_metadata_v1"
)
PORTABLE_TASK_CAPABILITY_METADATA_WITH_EXECUTION_CONTRACT_VERSION = (
    "harness_owned_pre_agent_task_capability_metadata_v2_execution_contract"
)
PORTABLE_CAPABILITY_GRAPH_PROJECTION_VERSION = (
    "complete_recipe_portable_capability_projection_v1"
)
OFFICE_COLLECTION_INVENTORY_POLICY_VERSION = (
    "harness_owned_readonly_office_collection_inventory_v1"
)
OFFICE_COLLECTION_EFFECT_RECEIPT_VERSION = (
    "exact_collection_tree_pre_post_and_task_local_output_effect_v1"
)

TASK_DECLARED_PRIMARY_INPUT_ROLE = "task_declared_primary_input"
PROFILE_DELIMITED_TABLE_CAPABILITY = "profile_delimited_table"
TASK_DECLARED_OFFICE_COLLECTION_ROLE = (
    "task_declared_office_document_collection"
)
INVENTORY_OFFICE_DOCUMENT_COLLECTION_CAPABILITY = (
    "inventory_office_document_collection"
)
CAPABILITY_OUTPUT_ROOT = PurePosixPath(
    "/root/.assumption-v2/capabilities"
)

MAX_TABULAR_INPUT_BYTES = 64 * 1024 * 1024
MAX_TABULAR_RECORDS = 1_000_000
MAX_TABULAR_COLUMNS = 10_000
MAX_OFFICE_COLLECTION_FILES = 10_000
MAX_OFFICE_COLLECTION_BYTES = 8 * 1024 * 1024 * 1024
MAX_RUNTIME_DISCOVERY_ENTRIES = 50_000
MAX_RUNTIME_DISCOVERY_TEXT_BYTES = 8 * 1024 * 1024

_TASK_PATH = re.compile(r"/root(?:/[A-Za-z0-9._+-]+)+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_OFFICE_COLLECTION_FILENAME = re.compile(r"[A-Za-z0-9._+@%=-]+")
_FORBIDDEN_COMPONENTS = frozenset(
    {
        ".env",
        ".ssh",
        "auth",
        "credentials",
        "oracle",
        "password",
        "private_key",
        "secret",
        "solution",
        "solutions",
        "test",
        "tests",
        "token",
        "validation",
        "verifier",
    }
)
_SUPPORTED_TABULAR_SUFFIXES = frozenset({".csv", ".tsv"})
_SUPPORTED_OFFICE_COLLECTION_SUFFIXES = frozenset(
    {".docx", ".odp", ".odt", ".pdf", ".pptx"}
)
SUPPORTED_PORTABLE_TASK_CAPABILITY_FAMILIES = frozenset(
    {
        "organize-messy-files",
        "stock-data-visualization",
        "temperature-simulation",
    }
)
_TABULAR_PORTABLE_FAMILIES = frozenset(
    {"stock-data-visualization", "temperature-simulation"}
)
_OFFICE_COLLECTION_PORTABLE_FAMILIES = frozenset(
    {"organize-messy-files"}
)


class PortableArtifactResolutionError(PermissionError):
    """A current item could not be bound to exactly one safe artifact."""


class RestrictedCapabilityExecutionError(RuntimeError):
    """A fixed task-local capability could not produce an exact receipt."""


@dataclass(frozen=True)
class PortableArtifactRoleSpec:
    role: str
    artifact_format: ArtifactFormat
    capability: str
    source_graph_hash: str
    source_recipe_id: str

    @property
    def role_spec_hash(self) -> str:
        return stable_hash(self.safe_payload(include_hash=False))

    def safe_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "policy": _role_policy_for_spec(self),
            "role": self.role,
            "artifact_format": self.artifact_format.value,
            "capability": self.capability,
            "source_graph_hash": self.source_graph_hash,
            "source_recipe_id": self.source_recipe_id,
            "source_artifact_locator_disclosed": False,
            "model_authored_locator_allowed": False,
            "model_authored_capability_arguments_allowed": False,
        }
        if include_hash:
            payload["role_spec_hash"] = self.role_spec_hash
        return payload


@dataclass(frozen=True)
class PortableFamilyCapabilityGraph(FamilyCapabilityGraph):
    """Canonical selectable graph backed by the restricted runtime registry."""

    source_graph_hash: str
    behavioral_alias_count: int
    behavioral_alias_set_hash: str
    behavioral_treatment_signature_hashes: tuple[str, ...]

    def _projected_capability_payloads(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for capability in self.capabilities:
            payload = capability.payload()
            payload.update(
                {
                    "pre_agent_artifact_evidence_capability_verified": True,
                    "pre_agent_evidence_argument_surface_restricted": True,
                    "runtime_agent_capability_invocation_available": False,
                    "full_recipe_operator_execution_by_capability": False,
                    "portable_registry_version": (
                        PORTABLE_CAPABILITY_GRAPH_PROJECTION_VERSION
                    ),
                }
            )
            rows.append(payload)
        return rows

    def portable_projection_payload(self) -> dict[str, Any]:
        return {
            "projection_version": (
                PORTABLE_CAPABILITY_GRAPH_PROJECTION_VERSION
            ),
            "source_graph_hash": self.source_graph_hash,
            "selectable_recipe_set_hash": stable_hash(
                {"recipe_ids": [row.recipe_id for row in self.recipes]}
            ),
            "all_selectable_recipes_artifact_evidence_backed": True,
            "pre_agent_evidence_argument_surface_restricted": True,
            "capability_execution_covers_full_recipe_operator_plan": False,
            "non_access_operators_are_fixed_agent_plan": True,
            "model_authored_locator_allowed": False,
            "model_authored_capability_arguments_allowed": False,
            "source_recipe_literal_locator_used": False,
            "complete_recipe_sets_only": True,
            "source_executable_artifact_count": (
                len(self.artifacts) + self.behavioral_alias_count
            ),
            "retained_behavioral_artifact_count": len(self.artifacts),
            "behavioral_alias_artifact_count": self.behavioral_alias_count,
            "behavioral_alias_set_hash": self.behavioral_alias_set_hash,
            "behavioral_treatment_signature_hashes": list(
                self.behavioral_treatment_signature_hashes
            ),
            "behavioral_aliases_deduplicated": True,
            "diversity_counts_behavioral_signature_only": True,
        }

    def model_catalog(self) -> dict[str, Any]:
        payload = super().model_catalog()
        payload["capabilities"] = self._projected_capability_payloads()
        payload["portable_capability_projection"] = (
            self.portable_projection_payload()
        )
        payload["portable_selector_context"] = {
            "target_family": self.target_family,
            "target_family_scope": "frozen_train_routing_label",
            "target_family_used_as_task_content": False,
            "selector_output_fields": ["recipe_id"],
            "opaque_recipe_id_only_output": True,
        }
        return payload

    def safe_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = super().safe_payload(include_hash=False)
        payload["capabilities"] = self._projected_capability_payloads()
        payload["portable_capability_projection"] = (
            self.portable_projection_payload()
        )
        if include_hash:
            payload["graph_hash"] = self.graph_hash
        return payload

    def validate(self) -> tuple[str, ...]:
        issues = list(super().validate())
        if not _SHA256.fullmatch(self.source_graph_hash):
            issues.append("portable_projection_source_graph_hash_invalid")
        if not self.recipes:
            issues.append("portable_projection_recipe_registry_empty")
        if self.behavioral_alias_count < 0:
            issues.append("portable_projection_alias_count_invalid")
        if not _SHA256.fullmatch(self.behavioral_alias_set_hash):
            issues.append("portable_projection_alias_set_hash_invalid")
        if (
            not self.behavioral_treatment_signature_hashes
            or tuple(sorted(set(self.behavioral_treatment_signature_hashes)))
            != self.behavioral_treatment_signature_hashes
            or len(self.behavioral_treatment_signature_hashes)
            != len(self.artifacts)
            or any(
                not _SHA256.fullmatch(value)
                for value in self.behavioral_treatment_signature_hashes
            )
        ):
            issues.append("portable_projection_treatment_signatures_invalid")
        return tuple(sorted(set(issues)))


@dataclass(frozen=True)
class ResolvedTaskArtifact:
    spec: PortableArtifactRoleSpec
    item_id_hash: str
    public_instruction_hash: str
    container_locator: str
    host_path: Path
    input_sha256: str
    input_size: int
    input_entry_count: int = 1

    @property
    def binding_hash(self) -> str:
        return stable_hash(self.safe_payload(include_hash=False))

    def safe_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        is_collection = (
            self.spec.capability
            == INVENTORY_OFFICE_DOCUMENT_COLLECTION_CAPABILITY
        )
        payload = {
            "policy": _role_policy_for_spec(self.spec),
            "item_id_hash": self.item_id_hash,
            "public_instruction_hash": self.public_instruction_hash,
            "role_spec_hash": self.spec.role_spec_hash,
            "role": self.spec.role,
            "artifact_format": self.spec.artifact_format.value,
            "capability": self.spec.capability,
            "candidate_count": 1,
            "resolved_locator_hash": stable_hash(
                {"locator": self.container_locator}
            ),
            "input_sha256": self.input_sha256,
            "input_size": self.input_size,
            "resolution_source": (
                "current_task_runtime_root_discovery"
                if is_collection
                else "current_item_public_instruction"
            ),
            "source_recipe_literal_locator_used": False,
            "raw_locator_persisted": False,
        }
        if is_collection:
            payload.update(
                {
                    "input_kind": "directory_tree",
                    "input_entry_count": self.input_entry_count,
                    "document_content_persisted": False,
                }
            )
        if include_hash:
            payload["binding_hash"] = self.binding_hash
        return payload


@dataclass(frozen=True)
class PortableTaskArtifactFingerprint:
    kind: str
    sha256: str
    size: int
    entry_count: int


@dataclass(frozen=True)
class OfficeCollectionCapabilityEffectReceipt:
    binding: ResolvedTaskArtifact
    invocation_id: str
    output_container_locator: str
    output_host_path: Path
    input_before_sha256: str
    input_after_sha256: str
    input_size_before: int
    input_size_after: int
    input_entry_count_before: int
    input_entry_count_after: int
    output_sha256: str
    output_size: int
    file_count: int
    total_size: int
    extension_counts: tuple[tuple[str, int], ...]
    inventory_hash: str

    @property
    def receipt_hash(self) -> str:
        return stable_hash(self.safe_payload(include_hash=False))

    @property
    def agent_payload(self) -> dict[str, Any]:
        return {
            "policy": OFFICE_COLLECTION_INVENTORY_POLICY_VERSION,
            "role": self.binding.spec.role,
            "artifact_format": self.binding.spec.artifact_format.value,
            "profile_locator": self.output_container_locator,
            "effect_receipt_hash": self.receipt_hash,
            "profile_created_before_agent_start": True,
            "source_artifact_locator_disclosed": False,
        }

    def safe_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "receipt_version": OFFICE_COLLECTION_EFFECT_RECEIPT_VERSION,
            "capability_policy": OFFICE_COLLECTION_INVENTORY_POLICY_VERSION,
            "artifact_binding_hash": self.binding.binding_hash,
            "item_id_hash": self.binding.item_id_hash,
            "role_spec_hash": self.binding.spec.role_spec_hash,
            "invocation_id": self.invocation_id,
            "input_locator_hash": stable_hash(
                {"locator": self.binding.container_locator}
            ),
            "input_before_tree_sha256": self.input_before_sha256,
            "input_after_tree_sha256": self.input_after_sha256,
            "input_size_before": self.input_size_before,
            "input_size_after": self.input_size_after,
            "input_entry_count_before": self.input_entry_count_before,
            "input_entry_count_after": self.input_entry_count_after,
            "input_unchanged": (
                self.input_before_sha256 == self.input_after_sha256
                and self.input_size_before == self.input_size_after
                and self.input_entry_count_before
                == self.input_entry_count_after
            ),
            "output_locator_hash": stable_hash(
                {"locator": self.output_container_locator}
            ),
            "output_sha256": self.output_sha256,
            "output_size": self.output_size,
            "file_count": self.file_count,
            "total_size": self.total_size,
            "extension_counts": [
                {"extension": extension, "count": count}
                for extension, count in self.extension_counts
            ],
            "inventory_hash": self.inventory_hash,
            "effect_kind": "created_task_local_collection_inventory",
            "task_input_mutated": False,
            "network_accessed": False,
            "subprocess_invoked": False,
            "package_install_attempted": False,
            "verifier_content_accessed": False,
            "test_content_accessed": False,
            "solution_content_accessed": False,
            "model_authored_argument_count": 0,
            "raw_input_locator_persisted": False,
            "document_content_profiled": False,
            "document_content_persisted": False,
            "document_content_exposed_to_agent": False,
            "content_bytes_used_only_for_integrity_hash": True,
        }
        if include_hash:
            payload["receipt_hash"] = self.receipt_hash
        return payload


@dataclass(frozen=True)
class TaskCapabilityEffectReceipt:
    binding: ResolvedTaskArtifact
    invocation_id: str
    output_container_locator: str
    output_host_path: Path
    input_before_sha256: str
    input_after_sha256: str
    input_size_before: int
    input_size_after: int
    output_sha256: str
    output_size: int
    delimiter_kind: str
    record_count: int
    column_count: int
    minimum_record_width: int
    maximum_record_width: int
    header_hash: str

    @property
    def receipt_hash(self) -> str:
        return stable_hash(self.safe_payload(include_hash=False))

    @property
    def agent_payload(self) -> dict[str, Any]:
        """The only runtime surface intended for the treated task agent."""

        return {
            "policy": RESTRICTED_TASK_CAPABILITY_POLICY_VERSION,
            "role": self.binding.spec.role,
            "artifact_format": self.binding.spec.artifact_format.value,
            "profile_locator": self.output_container_locator,
            "effect_receipt_hash": self.receipt_hash,
            "profile_created_before_agent_start": True,
            "source_artifact_locator_disclosed": False,
        }

    def safe_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "receipt_version": TASK_CAPABILITY_EFFECT_RECEIPT_VERSION,
            "capability_policy": RESTRICTED_TASK_CAPABILITY_POLICY_VERSION,
            "artifact_binding_hash": self.binding.binding_hash,
            "item_id_hash": self.binding.item_id_hash,
            "role_spec_hash": self.binding.spec.role_spec_hash,
            "invocation_id": self.invocation_id,
            "input_locator_hash": stable_hash(
                {"locator": self.binding.container_locator}
            ),
            "input_before_sha256": self.input_before_sha256,
            "input_after_sha256": self.input_after_sha256,
            "input_size_before": self.input_size_before,
            "input_size_after": self.input_size_after,
            "input_unchanged": (
                self.input_before_sha256 == self.input_after_sha256
                and self.input_size_before == self.input_size_after
            ),
            "output_locator_hash": stable_hash(
                {"locator": self.output_container_locator}
            ),
            "output_sha256": self.output_sha256,
            "output_size": self.output_size,
            "delimiter_kind": self.delimiter_kind,
            "record_count": self.record_count,
            "column_count": self.column_count,
            "minimum_record_width": self.minimum_record_width,
            "maximum_record_width": self.maximum_record_width,
            "header_hash": self.header_hash,
            "effect_kind": "created_task_local_derived_profile",
            "task_input_mutated": False,
            "network_accessed": False,
            "subprocess_invoked": False,
            "package_install_attempted": False,
            "verifier_content_accessed": False,
            "test_content_accessed": False,
            "solution_content_accessed": False,
            "model_authored_argument_count": 0,
            "raw_input_rows_persisted": False,
            "raw_input_locator_persisted": False,
            "derived_header_available_only_in_task_local_profile": True,
        }
        if include_hash:
            payload["receipt_hash"] = self.receipt_hash
        return payload


@dataclass(frozen=True)
class CompiledPortableTaskCapability:
    """Locator-free compiler-to-lifecycle contract for one routed item.

    The only locator in this payload is a harness-chosen *output* path.  The
    source role is resolved later from the current item's public instruction;
    neither the frozen TRAIN locator nor a model-authored locator is persisted.
    """

    role_spec: PortableArtifactRoleSpec
    item_id_hash: str
    program_id_hash: str
    typed_binding_hash: str
    bound_recipe_hash: str
    output_container_locator: str
    execution_contract: TypedExecutionContract | None = None

    @property
    def metadata_hash(self) -> str:
        return stable_hash(self.safe_payload(include_hash=False))

    def safe_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "metadata_version": (
                PORTABLE_TASK_CAPABILITY_METADATA_WITH_EXECUTION_CONTRACT_VERSION
                if self.execution_contract is not None
                else PORTABLE_TASK_CAPABILITY_METADATA_VERSION
            ),
            "compiler_mode": PORTABLE_TASK_CAPABILITY_COMPILER_VERSION,
            "item_id_hash": self.item_id_hash,
            "program_id_hash": self.program_id_hash,
            "typed_binding_hash": self.typed_binding_hash,
            "bound_recipe_hash": self.bound_recipe_hash,
            "role_spec": self.role_spec.safe_payload(),
            "role_spec_hash": self.role_spec.role_spec_hash,
            "execution_phase": "before_agent_start",
            "resolver_policy": _role_policy_for_spec(self.role_spec),
            "capability_policy": RESTRICTED_TASK_CAPABILITY_POLICY_VERSION,
            "effect_receipt_version": TASK_CAPABILITY_EFFECT_RECEIPT_VERSION,
            "output_container_locator": self.output_container_locator,
            "public_instruction_required": True,
            "runtime_root_required": True,
            "task_input_must_preexist": True,
            "source_artifact_locator_disclosed": False,
            "source_recipe_literal_locator_used": False,
            "model_authored_locator_allowed": False,
            "model_authored_capability_arguments_allowed": False,
            "harness_executes_capability": True,
            "agent_start_requires_verified_effect_receipt": True,
            "raw_content_persisted": False,
        }
        if self.execution_contract is not None:
            payload.update(
                {
                    "execution_contract": (
                        self.execution_contract.safe_payload()
                    ),
                    "execution_contract_hash": (
                        self.execution_contract.contract_hash
                    ),
                    "execution_contract_runtime_enforcement_claimed": False,
                }
            )
        if include_hash:
            payload["metadata_hash"] = self.metadata_hash
        return payload


def build_compiled_portable_task_capability(
    role_spec: PortableArtifactRoleSpec,
    *,
    item_id: str,
    program_id: str,
    typed_binding_hash: str,
    bound_recipe_hash: str,
    execution_contract: TypedExecutionContract | None = None,
) -> CompiledPortableTaskCapability:
    """Build one deterministic pre-agent hook without an input locator."""

    _validate_role_spec(role_spec)
    if not isinstance(item_id, str) or not item_id.strip():
        raise PermissionError("portable capability item identity is missing")
    if not isinstance(program_id, str) or not program_id.strip():
        raise PermissionError("portable capability program identity is missing")
    for value, label in (
        (typed_binding_hash, "typed binding"),
        (bound_recipe_hash, "bound recipe"),
    ):
        if not _SHA256.fullmatch(str(value or "")):
            raise PermissionError(
                f"portable capability {label} hash is malformed"
            )
    output_locator = deterministic_portable_capability_output_locator(
        role_spec_hash=role_spec.role_spec_hash,
        typed_binding_hash=typed_binding_hash,
    )
    if execution_contract is not None:
        issues = execution_contract.validate_closed()
        if issues:
            raise PermissionError(
                f"portable execution contract is invalid: {list(issues)}"
            )
        if execution_contract.graph_hash != role_spec.source_graph_hash:
            raise PermissionError(
                "portable execution contract graph binding mismatch"
            )
        if execution_contract.recipe_id != role_spec.source_recipe_id:
            raise PermissionError(
                "portable execution contract recipe binding mismatch"
            )
    metadata = CompiledPortableTaskCapability(
        role_spec=role_spec,
        item_id_hash=stable_hash({"item_id": item_id}),
        program_id_hash=stable_hash({"program_id": program_id}),
        typed_binding_hash=typed_binding_hash,
        bound_recipe_hash=bound_recipe_hash,
        output_container_locator=output_locator,
        execution_contract=execution_contract,
    )
    validate_compiled_portable_task_capability(metadata.safe_payload())
    return metadata


def validate_compiled_portable_task_capability(
    payload: Mapping[str, Any],
) -> CompiledPortableTaskCapability:
    """Reconstruct and exactly validate one persisted hook contract."""

    if not isinstance(payload, Mapping):
        raise PermissionError("portable capability metadata is malformed")
    raw_role = payload.get("role_spec")
    if not isinstance(raw_role, Mapping):
        raise PermissionError("portable capability role spec is malformed")
    try:
        role_spec = PortableArtifactRoleSpec(
            role=str(raw_role.get("role") or ""),
            artifact_format=ArtifactFormat(
                str(raw_role.get("artifact_format") or "")
            ),
            capability=str(raw_role.get("capability") or ""),
            source_graph_hash=str(raw_role.get("source_graph_hash") or ""),
            source_recipe_id=str(raw_role.get("source_recipe_id") or ""),
        )
    except ValueError as exc:
        raise PermissionError(
            "portable capability role spec is malformed"
        ) from exc
    _validate_role_spec(role_spec)
    if dict(raw_role) != role_spec.safe_payload():
        raise PermissionError("portable capability role spec is not canonical")

    metadata = CompiledPortableTaskCapability(
        role_spec=role_spec,
        item_id_hash=str(payload.get("item_id_hash") or ""),
        program_id_hash=str(payload.get("program_id_hash") or ""),
        typed_binding_hash=str(payload.get("typed_binding_hash") or ""),
        bound_recipe_hash=str(payload.get("bound_recipe_hash") or ""),
        output_container_locator=str(
            payload.get("output_container_locator") or ""
        ),
        execution_contract=(
            load_typed_execution_contract(payload["execution_contract"])
            if isinstance(payload.get("execution_contract"), Mapping)
            else None
        ),
    )
    if ("execution_contract" in payload) != (
        metadata.execution_contract is not None
    ):
        raise PermissionError(
            "portable capability execution contract is malformed"
        )
    if metadata.execution_contract is not None:
        if (
            metadata.execution_contract.graph_hash
            != metadata.role_spec.source_graph_hash
        ):
            raise PermissionError(
                "portable capability execution contract graph drifted"
            )
        if (
            metadata.execution_contract.recipe_id
            != metadata.role_spec.source_recipe_id
        ):
            raise PermissionError(
                "portable capability execution contract recipe drifted"
            )
    for value in (
        metadata.item_id_hash,
        metadata.program_id_hash,
        metadata.typed_binding_hash,
        metadata.bound_recipe_hash,
    ):
        if not _SHA256.fullmatch(value):
            raise PermissionError(
                "portable capability metadata hash is malformed"
            )
    expected_output = deterministic_portable_capability_output_locator(
        role_spec_hash=role_spec.role_spec_hash,
        typed_binding_hash=metadata.typed_binding_hash,
    )
    if metadata.output_container_locator != expected_output:
        raise PermissionError(
            "portable capability output locator is not deterministic"
        )
    if dict(payload) != metadata.safe_payload():
        raise PermissionError("portable capability metadata is not canonical")
    return metadata


def deterministic_portable_capability_output_locator(
    *,
    role_spec_hash: str,
    typed_binding_hash: str,
) -> str:
    for value in (role_spec_hash, typed_binding_hash):
        if not _SHA256.fullmatch(str(value or "")):
            raise PermissionError(
                "portable capability output identity is malformed"
            )
    output_id = stable_hash(
        {
            "compiler_mode": PORTABLE_TASK_CAPABILITY_COMPILER_VERSION,
            "role_spec_hash": role_spec_hash,
            "typed_binding_hash": typed_binding_hash,
        }
    )[:24]
    return str(CAPABILITY_OUTPUT_ROOT / f"portable-{output_id}.json")


def portable_role_spec_for_recipe(
    graph: FamilyCapabilityGraph,
    recipe_id: str,
) -> PortableArtifactRoleSpec:
    """Map one frozen typed recipe to a locator-free executable role.

    Unsupported formats fail closed instead of falling back to prompt text,
    shell, a TRAIN locator, or model-authored capability arguments.
    """

    issues = graph.validate()
    if issues:
        raise PermissionError(
            f"portable role received an invalid typed graph: {list(issues)}"
        )
    if graph.target_family not in SUPPORTED_PORTABLE_TASK_CAPABILITY_FAMILIES:
        raise PermissionError(
            "typed recipe family has no supported portable task capability"
        )
    matches = [row for row in graph.recipes if row.recipe_id == recipe_id]
    if len(matches) != 1:
        raise PermissionError("portable role references an unknown recipe")
    recipe = matches[0]
    artifacts = {row.artifact_id: row for row in graph.artifacts}
    capabilities = {row.capability_id: row for row in graph.capabilities}
    artifact = artifacts[recipe.primary_artifact_id]
    capability = capabilities[recipe.capability_id]
    operator_kinds = {row.kind for row in recipe.nodes}
    if graph.target_family in _TABULAR_PORTABLE_FAMILIES:
        if (
            artifact.format is not ArtifactFormat.TABULAR
            or capability.kind is not CapabilityKind.TABULAR_DATA
            or OperatorKind.PARSE_WITH_REGISTERED_CAPABILITY
            not in operator_kinds
        ):
            raise PermissionError(
                "typed recipe has no supported restricted task capability"
            )
        return PortableArtifactRoleSpec(
            role=TASK_DECLARED_PRIMARY_INPUT_ROLE,
            artifact_format=ArtifactFormat.TABULAR,
            capability=PROFILE_DELIMITED_TABLE_CAPABILITY,
            source_graph_hash=graph.graph_hash,
            source_recipe_id=recipe.recipe_id,
        )
    if graph.target_family in _OFFICE_COLLECTION_PORTABLE_FAMILIES:
        if (
            artifact.format is not ArtifactFormat.OFFICE_DOCUMENT
            or capability.kind is not CapabilityKind.OFFICE_DOCUMENT
            or not operator_kinds.intersection(
                {
                    OperatorKind.PARSE_WITH_REGISTERED_CAPABILITY,
                    OperatorKind.INSPECT_REGISTERED_METADATA,
                }
            )
        ):
            raise PermissionError(
                "typed recipe has no supported restricted task capability"
            )
        return PortableArtifactRoleSpec(
            role=TASK_DECLARED_OFFICE_COLLECTION_ROLE,
            artifact_format=ArtifactFormat.OFFICE_DOCUMENT,
            capability=INVENTORY_OFFICE_DOCUMENT_COLLECTION_CAPABILITY,
            source_graph_hash=graph.graph_hash,
            source_recipe_id=recipe.recipe_id,
        )
    raise PermissionError(
        "typed recipe family has no supported portable task capability"
    )


def project_portable_family_capability_graph(
    graph: FamilyCapabilityGraph,
) -> PortableFamilyCapabilityGraph:
    """Project a valid graph to artifacts with complete portable lowering.

    Artifacts are atomic projection units: either every canonical recipe for
    an artifact lowers to a closed pre-agent artifact-evidence role plus a
    fixed agent plan, or the artifact, its capability, and all of its recipes
    are omitted.  The evidence capability does not claim to execute the
    recipe's write/render/transform operators.  The result remains a canonical
    ``FamilyCapabilityGraph`` and is idempotent.
    """

    issues = graph.validate()
    if issues:
        raise PermissionError(
            f"portable projection received an invalid typed graph: {list(issues)}"
        )
    if isinstance(graph, PortableFamilyCapabilityGraph):
        for recipe in graph.recipes:
            portable_role_spec_for_recipe(graph, recipe.recipe_id)
        return graph
    retained_artifact_ids: set[str] = set()
    behavioral_groups: dict[str, list[str]] = {}
    for artifact in graph.artifacts:
        recipes = tuple(
            row
            for row in graph.recipes
            if row.primary_artifact_id == artifact.artifact_id
        )
        if not recipes:
            continue
        try:
            lowered_rows: list[dict[str, Any]] = []
            for recipe in recipes:
                role_spec = portable_role_spec_for_recipe(
                    graph,
                    recipe.recipe_id,
                )
                lowered_rows.append(
                    {
                        "workflow": recipe.workflow.value,
                        "operator_kinds": [
                            node.kind.value for node in recipe.nodes
                        ],
                        "role": role_spec.role,
                        "artifact_format": role_spec.artifact_format.value,
                        "evidence_capability": role_spec.capability,
                        "non_access_operators_are_fixed_agent_plan": True,
                    }
                )
        except PermissionError:
            continue
        behavioral_signature = stable_hash(
            {
                "projection": PORTABLE_CAPABILITY_GRAPH_PROJECTION_VERSION,
                "target_family": graph.target_family,
                "recipe_treatments": sorted(
                    lowered_rows,
                    key=lambda row: (
                        row["workflow"],
                        tuple(row["operator_kinds"]),
                    ),
                ),
            }
        )
        group = behavioral_groups.setdefault(behavioral_signature, [])
        group.append(artifact.artifact_id)
        if len(group) == 1:
            retained_artifact_ids.add(artifact.artifact_id)
    if not retained_artifact_ids:
        raise PermissionError(
            "typed graph has no complete portable capability recipe set"
        )
    artifacts = tuple(
        row
        for row in graph.artifacts
        if row.artifact_id in retained_artifact_ids
    )
    retained_capability_ids = {
        row.capability_id
        for row in graph.recipes
        if row.primary_artifact_id in retained_artifact_ids
    }
    capabilities = tuple(
        row
        for row in graph.capabilities
        if row.capability_id in retained_capability_ids
    )
    recipes = tuple(
        row
        for row in graph.recipes
        if row.primary_artifact_id in retained_artifact_ids
    )
    alias_groups = [
        {
            "behavioral_treatment_signature_hash": signature,
            "artifact_ids": artifact_ids,
        }
        for signature, artifact_ids in sorted(behavioral_groups.items())
        if len(artifact_ids) > 1
    ]
    behavioral_alias_count = sum(
        len(row["artifact_ids"]) - 1 for row in alias_groups
    )
    behavioral_alias_set_hash = stable_hash(
        {"alias_groups": alias_groups}
    )
    behavioral_treatment_signature_hashes = tuple(
        sorted(behavioral_groups)
    )
    if len(retained_artifact_ids) == len(graph.artifacts):
        projected_source_evidence_hash = graph.source_evidence_hash
    else:
        projected_source_evidence_hash = stable_hash(
            {
                "projection": PORTABLE_CAPABILITY_GRAPH_PROJECTION_VERSION,
                "source_graph_hash": graph.graph_hash,
                "retained_artifact_ids": [
                    row.artifact_id for row in artifacts
                ],
                "behavioral_alias_set_hash": behavioral_alias_set_hash,
            }
        )
    projected = PortableFamilyCapabilityGraph(
        target_family=graph.target_family,
        source_evidence_hash=projected_source_evidence_hash,
        artifacts=artifacts,
        capabilities=capabilities,
        recipes=recipes,
        source_graph_hash=graph.graph_hash,
        behavioral_alias_count=behavioral_alias_count,
        behavioral_alias_set_hash=behavioral_alias_set_hash,
        behavioral_treatment_signature_hashes=(
            behavioral_treatment_signature_hashes
        ),
    )
    projected_issues = projected.validate()
    if projected_issues:
        raise PermissionError(
            "portable capability projection is not canonical: "
            f"{list(projected_issues)}"
        )
    for recipe in projected.recipes:
        portable_role_spec_for_recipe(projected, recipe.recipe_id)
    return projected


def portable_role_spec_for_bound_recipe(
    bound_recipe: BoundTypedRecipe,
) -> PortableArtifactRoleSpec:
    """Lower only the exact registry-resolved recipe, never a prompt guess."""

    if (
        bound_recipe.snapshot.validate()
        or (
            bound_recipe.binding.snapshot_hash
            != bound_recipe.snapshot.snapshot_hash
        )
        or (
            bound_recipe.binding.graph_hash
            != bound_recipe.snapshot.graph.graph_hash
        )
        or bound_recipe.binding.recipe_id != bound_recipe.recipe.recipe_id
        or bound_recipe.recipe not in bound_recipe.snapshot.graph.recipes
    ):
        raise PermissionError("bound typed recipe is not canonical")
    return portable_role_spec_for_recipe(
        bound_recipe.snapshot.graph,
        bound_recipe.recipe.recipe_id,
    )


def resolve_portable_artifact_role(
    spec: PortableArtifactRoleSpec,
    *,
    item_id: str,
    public_instruction: str,
    runtime_root: str | Path,
) -> ResolvedTaskArtifact:
    """Resolve from the current item only; TRAIN locators are not consulted."""

    _validate_role_spec(spec)
    if not isinstance(item_id, str) or not item_id.strip():
        raise PortableArtifactResolutionError("current item identity is missing")
    if not isinstance(public_instruction, str) or not public_instruction.strip():
        raise PortableArtifactResolutionError(
            "current item public instruction is missing"
        )
    root_argument = Path(runtime_root).expanduser()
    if root_argument.is_symlink():
        raise PortableArtifactResolutionError("runtime root is not a directory")
    root = root_argument.resolve(strict=True)
    if not root.is_dir():
        raise PortableArtifactResolutionError("runtime root is not a directory")

    candidates: list[
        tuple[str, Path, PortableTaskArtifactFingerprint]
    ] = []
    for locator in portable_artifact_locators_from_public_instruction(
        spec,
        public_instruction=public_instruction,
        runtime_root=root,
    ):
        is_collection = (
            spec.capability
            == INVENTORY_OFFICE_DOCUMENT_COLLECTION_CAPABILITY
        )
        host_path = _contained_runtime_path(
            root,
            locator,
            require_file=not is_collection,
            require_directory=is_collection,
        )
        if host_path is None:
            continue
        fingerprint = portable_task_artifact_fingerprint(spec, host_path)
        candidates.append((locator, host_path, fingerprint))
    candidates = sorted(
        {row[0]: row for row in candidates}.values(),
        key=lambda row: row[0],
    )
    if len(candidates) != 1:
        raise PortableArtifactResolutionError(
            "current item role resolution must yield exactly one safe "
            f"artifact, found {len(candidates)}"
        )
    locator, host_path, fingerprint = candidates[0]
    return ResolvedTaskArtifact(
        spec=spec,
        item_id_hash=stable_hash({"item_id": item_id}),
        public_instruction_hash=stable_hash(
            {"public_instruction": public_instruction}
        ),
        container_locator=locator,
        host_path=host_path,
        input_sha256=fingerprint.sha256,
        input_size=fingerprint.size,
        input_entry_count=fingerprint.entry_count,
    )


def portable_artifact_locators_from_public_instruction(
    spec: PortableArtifactRoleSpec,
    *,
    public_instruction: str,
    runtime_root: str | Path | None = None,
) -> tuple[str, ...]:
    """Return canonical candidates using only the current task surface.

    Tabular candidates are explicit public-instruction locators.  Collection
    candidates intentionally ignore recipe/TRAIN locators and instead use a
    bounded, read-only listing of the current runtime root.
    """

    _validate_role_spec(spec)
    if not isinstance(public_instruction, str) or not public_instruction.strip():
        raise PortableArtifactResolutionError(
            "current item public instruction is missing"
        )
    if spec.capability == PROFILE_DELIMITED_TABLE_CAPABILITY:
        candidates: list[str] = []
        for raw_locator in _TASK_PATH.findall(public_instruction):
            locator = _canonical_current_item_locator(raw_locator)
            if locator is None or locator.startswith("/root/output/"):
                continue
            if PurePosixPath(locator).suffix.lower() not in (
                _SUPPORTED_TABULAR_SUFFIXES
            ):
                continue
            candidates.append(locator)
        return tuple(sorted(set(candidates)))
    if spec.capability == INVENTORY_OFFICE_DOCUMENT_COLLECTION_CAPABILITY:
        if runtime_root is None:
            raise PortableArtifactResolutionError(
                "office collection resolution requires the current runtime root"
            )
        root_argument = Path(runtime_root).expanduser()
        if root_argument.is_symlink():
            raise PortableArtifactResolutionError(
                "runtime root is not a directory"
            )
        root = root_argument.resolve(strict=True)
        if not root.is_dir():
            raise PortableArtifactResolutionError(
                "runtime root is not a directory"
            )
        return portable_collection_locators_from_runtime_entries(
            spec,
            _runtime_root_entries(root),
        )
    raise PortableArtifactResolutionError(
        "portable artifact role spec is outside the closed registry"
    )


def portable_collection_locators_from_runtime_entries(
    spec: PortableArtifactRoleSpec,
    entries: Sequence[tuple[str, str]],
) -> tuple[str, ...]:
    """Select flat document collections from a bounded metadata-only tree.

    ``entries`` contain only a kind marker (``d``, ``f``, or ``l``) and a
    container locator.  No file contents or model-authored arguments enter
    collection selection.
    """

    _validate_role_spec(spec)
    if spec.capability != INVENTORY_OFFICE_DOCUMENT_COLLECTION_CAPABILITY:
        raise PortableArtifactResolutionError(
            "runtime-tree discovery is not registered for this role"
        )
    if len(entries) > MAX_RUNTIME_DISCOVERY_ENTRIES:
        raise PortableArtifactResolutionError(
            "runtime-tree discovery exceeds the fixed entry bound"
        )
    normalized: dict[str, str] = {}
    for raw_kind, raw_locator in entries:
        kind = str(raw_kind or "")
        locator = _canonical_current_item_locator(str(raw_locator or ""))
        if kind not in {"d", "f", "l"} or locator is None:
            continue
        previous = normalized.setdefault(locator, kind)
        if previous != kind:
            raise PortableArtifactResolutionError(
                "runtime-tree discovery has conflicting entry kinds"
            )
    children: dict[str, list[tuple[str, str]]] = {}
    for locator, kind in normalized.items():
        parent = str(PurePosixPath(locator).parent)
        children.setdefault(parent, []).append((locator, kind))
    candidates: list[str] = []
    for locator, kind in sorted(normalized.items()):
        if kind != "d" or locator.startswith("/root/output/"):
            continue
        direct = children.get(locator, [])
        if not direct or any(child_kind != "f" for _, child_kind in direct):
            continue
        if not all(
            PurePosixPath(child).suffix.lower()
            in _SUPPORTED_OFFICE_COLLECTION_SUFFIXES
            for child, _ in direct
        ):
            continue
        candidates.append(locator)
    return tuple(candidates)


def portable_task_artifact_fingerprint(
    spec: PortableArtifactRoleSpec,
    path: str | Path,
) -> PortableTaskArtifactFingerprint:
    """Return a fixed integrity fingerprint without exposing input content."""

    _validate_role_spec(spec)
    current = Path(path)
    if spec.capability == PROFILE_DELIMITED_TABLE_CAPABILITY:
        if current.is_symlink() or not current.is_file():
            raise PortableArtifactResolutionError(
                "current item tabular input is not a safe file"
            )
        size = current.stat().st_size
        if size > MAX_TABULAR_INPUT_BYTES:
            raise PortableArtifactResolutionError(
                "current item tabular input exceeds the fixed capability bound"
            )
        return PortableTaskArtifactFingerprint(
            kind="file",
            sha256=_sha256_file(current),
            size=size,
            entry_count=1,
        )
    if spec.capability == INVENTORY_OFFICE_DOCUMENT_COLLECTION_CAPABILITY:
        snapshot = _office_collection_tree_snapshot(current)
        return PortableTaskArtifactFingerprint(
            kind="directory_tree",
            sha256=str(snapshot["tree_sha256"]),
            size=int(snapshot["total_size"]),
            entry_count=int(snapshot["file_count"]),
        )
    raise PortableArtifactResolutionError(
        "portable artifact role spec is outside the closed registry"
    )


def execute_restricted_task_capability(
    binding: ResolvedTaskArtifact,
    *,
    runtime_root: str | Path,
    required_output_container_locator: str | None = None,
) -> TaskCapabilityEffectReceipt | OfficeCollectionCapabilityEffectReceipt:
    """Execute one fixed capability and bind its exact pre/post effect."""

    _validate_role_spec(binding.spec)
    if (
        binding.spec.capability
        == INVENTORY_OFFICE_DOCUMENT_COLLECTION_CAPABILITY
    ):
        return _execute_office_collection_inventory(
            binding,
            runtime_root=runtime_root,
            required_output_container_locator=(
                required_output_container_locator
            ),
        )
    if binding.spec.capability != PROFILE_DELIMITED_TABLE_CAPABILITY:
        raise RestrictedCapabilityExecutionError(
            "restricted capability is not registered"
        )
    root_argument = Path(runtime_root).expanduser()
    if root_argument.is_symlink():
        raise RestrictedCapabilityExecutionError(
            "runtime root is not a safe directory"
        )
    root = root_argument.resolve(strict=True)
    current_input = _contained_runtime_path(
        root,
        binding.container_locator,
        require_file=True,
    )
    if current_input is None or current_input != binding.host_path:
        raise RestrictedCapabilityExecutionError(
            "resolved task artifact no longer matches the runtime root"
        )
    before_size = current_input.stat().st_size
    before_sha256 = _sha256_file(current_input)
    if (
        before_sha256 != binding.input_sha256
        or before_size != binding.input_size
    ):
        raise RestrictedCapabilityExecutionError(
            "resolved task artifact changed before capability execution"
        )

    suffix = PurePosixPath(binding.container_locator).suffix.lower()
    if suffix not in _SUPPORTED_TABULAR_SUFFIXES:
        raise RestrictedCapabilityExecutionError(
            "resolved task artifact is not a delimited table"
        )
    delimiter = "\t" if suffix == ".tsv" else ","
    delimiter_kind = "tab" if suffix == ".tsv" else "comma"
    profile = _profile_delimited_table(current_input, delimiter=delimiter)

    invocation_id = stable_hash(
        {
            "policy": RESTRICTED_TASK_CAPABILITY_POLICY_VERSION,
            "binding_hash": binding.binding_hash,
            "capability": binding.spec.capability,
        }
    )[:24]
    output_locator = (
        str(CAPABILITY_OUTPUT_ROOT / f"{invocation_id}.json")
        if required_output_container_locator is None
        else str(required_output_container_locator)
    )
    output_path = _prepare_capability_output_path(
        root,
        output_locator,
    )
    profile_payload = {
        "profile_version": RESTRICTED_TASK_CAPABILITY_POLICY_VERSION,
        "role": binding.spec.role,
        "artifact_format": binding.spec.artifact_format.value,
        "capability": binding.spec.capability,
        "invocation_id": invocation_id,
        "source_locator_hash": stable_hash(
            {"locator": binding.container_locator}
        ),
        "source_sha256": before_sha256,
        "delimiter_kind": delimiter_kind,
        "record_count": profile["record_count"],
        "column_count": profile["column_count"],
        "minimum_record_width": profile["minimum_record_width"],
        "maximum_record_width": profile["maximum_record_width"],
        "columns": profile["columns"],
        "raw_input_rows_persisted": False,
        "raw_input_locator_persisted": False,
    }
    output_bytes = (
        json.dumps(
            profile_payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with output_path.open("xb") as handle:
            handle.write(output_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RestrictedCapabilityExecutionError(
            "restricted capability output already exists"
        ) from exc

    after_size = current_input.stat().st_size
    after_sha256 = _sha256_file(current_input)
    if before_size != after_size or before_sha256 != after_sha256:
        output_path.unlink(missing_ok=True)
        raise RestrictedCapabilityExecutionError(
            "restricted capability changed its input artifact"
        )
    receipt = TaskCapabilityEffectReceipt(
        binding=binding,
        invocation_id=invocation_id,
        output_container_locator=output_locator,
        output_host_path=output_path,
        input_before_sha256=before_sha256,
        input_after_sha256=after_sha256,
        input_size_before=before_size,
        input_size_after=after_size,
        output_sha256=hashlib.sha256(output_bytes).hexdigest(),
        output_size=len(output_bytes),
        delimiter_kind=delimiter_kind,
        record_count=int(profile["record_count"]),
        column_count=int(profile["column_count"]),
        minimum_record_width=int(profile["minimum_record_width"]),
        maximum_record_width=int(profile["maximum_record_width"]),
        header_hash=stable_hash({"columns": profile["columns"]}),
    )
    verify_task_capability_effect(receipt, runtime_root=root)
    return receipt


def _execute_office_collection_inventory(
    binding: ResolvedTaskArtifact,
    *,
    runtime_root: str | Path,
    required_output_container_locator: str | None,
) -> OfficeCollectionCapabilityEffectReceipt:
    root_argument = Path(runtime_root).expanduser()
    if root_argument.is_symlink():
        raise RestrictedCapabilityExecutionError(
            "runtime root is not a safe directory"
        )
    root = root_argument.resolve(strict=True)
    current_input = _contained_runtime_path(
        root,
        binding.container_locator,
        require_file=False,
        require_directory=True,
    )
    if current_input is None or current_input != binding.host_path:
        raise RestrictedCapabilityExecutionError(
            "resolved task artifact no longer matches the runtime root"
        )
    try:
        before = _office_collection_tree_snapshot(current_input)
    except PortableArtifactResolutionError as exc:
        raise RestrictedCapabilityExecutionError(str(exc)) from exc
    if (
        before["tree_sha256"] != binding.input_sha256
        or before["total_size"] != binding.input_size
        or before["file_count"] != binding.input_entry_count
    ):
        raise RestrictedCapabilityExecutionError(
            "resolved task artifact changed before capability execution"
        )

    inventory_entries = list(before["inventory_entries"])
    extension_counts_map: dict[str, int] = {}
    for row in inventory_entries:
        extension = str(row["extension"])
        extension_counts_map[extension] = (
            extension_counts_map.get(extension, 0) + 1
        )
    extension_counts = tuple(sorted(extension_counts_map.items()))
    inventory_hash = stable_hash({"files": inventory_entries})
    invocation_id = stable_hash(
        {
            "policy": OFFICE_COLLECTION_INVENTORY_POLICY_VERSION,
            "binding_hash": binding.binding_hash,
            "capability": binding.spec.capability,
        }
    )[:24]
    output_locator = (
        str(CAPABILITY_OUTPUT_ROOT / f"{invocation_id}.json")
        if required_output_container_locator is None
        else str(required_output_container_locator)
    )
    output_path = _prepare_capability_output_path(root, output_locator)
    profile_payload = {
        "profile_version": OFFICE_COLLECTION_INVENTORY_POLICY_VERSION,
        "role": binding.spec.role,
        "artifact_format": binding.spec.artifact_format.value,
        "capability": binding.spec.capability,
        "invocation_id": invocation_id,
        "source_locator_hash": stable_hash(
            {"locator": binding.container_locator}
        ),
        "source_tree_sha256": before["tree_sha256"],
        "file_count": before["file_count"],
        "total_size": before["total_size"],
        "extension_counts": [
            {"extension": extension, "count": count}
            for extension, count in extension_counts
        ],
        "files": inventory_entries,
        "inventory_hash": inventory_hash,
        "document_content_profiled": False,
        "document_content_persisted": False,
        "raw_input_locator_persisted": False,
    }
    output_bytes = (
        json.dumps(
            profile_payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with output_path.open("xb") as handle:
            handle.write(output_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RestrictedCapabilityExecutionError(
            "restricted capability output already exists"
        ) from exc

    try:
        after = _office_collection_tree_snapshot(current_input)
    except PortableArtifactResolutionError as exc:
        output_path.unlink(missing_ok=True)
        raise RestrictedCapabilityExecutionError(str(exc)) from exc
    if (
        before["tree_sha256"] != after["tree_sha256"]
        or before["total_size"] != after["total_size"]
        or before["file_count"] != after["file_count"]
    ):
        output_path.unlink(missing_ok=True)
        raise RestrictedCapabilityExecutionError(
            "restricted capability changed its input artifact"
        )
    receipt = OfficeCollectionCapabilityEffectReceipt(
        binding=binding,
        invocation_id=invocation_id,
        output_container_locator=output_locator,
        output_host_path=output_path,
        input_before_sha256=str(before["tree_sha256"]),
        input_after_sha256=str(after["tree_sha256"]),
        input_size_before=int(before["total_size"]),
        input_size_after=int(after["total_size"]),
        input_entry_count_before=int(before["file_count"]),
        input_entry_count_after=int(after["file_count"]),
        output_sha256=hashlib.sha256(output_bytes).hexdigest(),
        output_size=len(output_bytes),
        file_count=int(before["file_count"]),
        total_size=int(before["total_size"]),
        extension_counts=extension_counts,
        inventory_hash=inventory_hash,
    )
    verify_task_capability_effect(receipt, runtime_root=root)
    return receipt


def verify_task_capability_effect(
    receipt: TaskCapabilityEffectReceipt | OfficeCollectionCapabilityEffectReceipt,
    *,
    runtime_root: str | Path,
) -> dict[str, Any]:
    """Re-open both artifacts and exactly verify a capability effect receipt."""

    if isinstance(receipt, OfficeCollectionCapabilityEffectReceipt):
        return _verify_office_collection_effect(
            receipt,
            runtime_root=runtime_root,
        )
    root_argument = Path(runtime_root).expanduser()
    if root_argument.is_symlink():
        raise PermissionError("capability effect runtime root is invalid")
    root = root_argument.resolve(strict=True)
    input_path = _contained_runtime_path(
        root,
        receipt.binding.container_locator,
        require_file=True,
    )
    output_path = _contained_runtime_path(
        root,
        receipt.output_container_locator,
        require_file=True,
    )
    if input_path is None or input_path != receipt.binding.host_path:
        raise PermissionError("capability effect input binding is invalid")
    if output_path is None or output_path != receipt.output_host_path:
        raise PermissionError("capability effect output binding is invalid")
    if (
        _sha256_file(input_path) != receipt.input_after_sha256
        or input_path.stat().st_size != receipt.input_size_after
        or receipt.input_before_sha256 != receipt.input_after_sha256
        or receipt.input_size_before != receipt.input_size_after
    ):
        raise PermissionError("capability effect input receipt does not verify")
    output_bytes = output_path.read_bytes()
    if (
        hashlib.sha256(output_bytes).hexdigest() != receipt.output_sha256
        or len(output_bytes) != receipt.output_size
    ):
        raise PermissionError("capability effect output hash does not verify")
    try:
        profile = json.loads(output_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PermissionError(
            "capability effect profile is not canonical JSON"
        ) from exc
    if not isinstance(profile, Mapping):
        raise PermissionError("capability effect profile is malformed")
    expected = {
        "profile_version": RESTRICTED_TASK_CAPABILITY_POLICY_VERSION,
        "role": receipt.binding.spec.role,
        "artifact_format": receipt.binding.spec.artifact_format.value,
        "capability": receipt.binding.spec.capability,
        "invocation_id": receipt.invocation_id,
        "source_locator_hash": stable_hash(
            {"locator": receipt.binding.container_locator}
        ),
        "source_sha256": receipt.input_before_sha256,
        "delimiter_kind": receipt.delimiter_kind,
        "record_count": receipt.record_count,
        "column_count": receipt.column_count,
        "minimum_record_width": receipt.minimum_record_width,
        "maximum_record_width": receipt.maximum_record_width,
        "columns": list(profile.get("columns") or []),
        "raw_input_rows_persisted": False,
        "raw_input_locator_persisted": False,
    }
    if dict(profile) != expected or stable_hash(
        {"columns": expected["columns"]}
    ) != receipt.header_hash:
        raise PermissionError("capability effect profile content drifted")
    canonical_bytes = (
        json.dumps(
            expected,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if output_bytes != canonical_bytes:
        raise PermissionError("capability effect profile encoding drifted")
    payload = receipt.safe_payload()
    if not _SHA256.fullmatch(str(payload.get("receipt_hash") or "")):
        raise PermissionError("capability effect receipt hash is malformed")
    return payload


def _verify_office_collection_effect(
    receipt: OfficeCollectionCapabilityEffectReceipt,
    *,
    runtime_root: str | Path,
) -> dict[str, Any]:
    root_argument = Path(runtime_root).expanduser()
    if root_argument.is_symlink():
        raise PermissionError("capability effect runtime root is invalid")
    root = root_argument.resolve(strict=True)
    input_path = _contained_runtime_path(
        root,
        receipt.binding.container_locator,
        require_file=False,
        require_directory=True,
    )
    output_path = _contained_runtime_path(
        root,
        receipt.output_container_locator,
        require_file=True,
    )
    if input_path is None or input_path != receipt.binding.host_path:
        raise PermissionError("capability effect input binding is invalid")
    if output_path is None or output_path != receipt.output_host_path:
        raise PermissionError("capability effect output binding is invalid")
    try:
        current = _office_collection_tree_snapshot(input_path)
    except PortableArtifactResolutionError as exc:
        raise PermissionError(
            "capability effect input receipt does not verify"
        ) from exc
    if (
        current["tree_sha256"] != receipt.input_after_sha256
        or current["total_size"] != receipt.input_size_after
        or current["file_count"] != receipt.input_entry_count_after
        or receipt.input_before_sha256 != receipt.input_after_sha256
        or receipt.input_size_before != receipt.input_size_after
        or receipt.input_entry_count_before
        != receipt.input_entry_count_after
    ):
        raise PermissionError("capability effect input receipt does not verify")
    output_bytes = output_path.read_bytes()
    if (
        hashlib.sha256(output_bytes).hexdigest() != receipt.output_sha256
        or len(output_bytes) != receipt.output_size
    ):
        raise PermissionError("capability effect output hash does not verify")
    try:
        profile = json.loads(output_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PermissionError(
            "capability effect profile is not canonical JSON"
        ) from exc
    if not isinstance(profile, Mapping):
        raise PermissionError("capability effect profile is malformed")
    inventory_entries = list(current["inventory_entries"])
    current_extension_counts_map: dict[str, int] = {}
    for row in inventory_entries:
        extension = str(row["extension"])
        current_extension_counts_map[extension] = (
            current_extension_counts_map.get(extension, 0) + 1
        )
    current_extension_counts = tuple(
        sorted(current_extension_counts_map.items())
    )
    expected = {
        "profile_version": OFFICE_COLLECTION_INVENTORY_POLICY_VERSION,
        "role": receipt.binding.spec.role,
        "artifact_format": receipt.binding.spec.artifact_format.value,
        "capability": receipt.binding.spec.capability,
        "invocation_id": receipt.invocation_id,
        "source_locator_hash": stable_hash(
            {"locator": receipt.binding.container_locator}
        ),
        "source_tree_sha256": receipt.input_before_sha256,
        "file_count": receipt.file_count,
        "total_size": receipt.total_size,
        "extension_counts": [
            {"extension": extension, "count": count}
            for extension, count in receipt.extension_counts
        ],
        "files": inventory_entries,
        "inventory_hash": receipt.inventory_hash,
        "document_content_profiled": False,
        "document_content_persisted": False,
        "raw_input_locator_persisted": False,
    }
    if (
        dict(profile) != expected
        or stable_hash({"files": inventory_entries})
        != receipt.inventory_hash
        or receipt.file_count != int(current["file_count"])
        or receipt.total_size != int(current["total_size"])
        or receipt.extension_counts != current_extension_counts
        or receipt.binding.input_sha256 != receipt.input_before_sha256
        or receipt.binding.input_size != receipt.input_size_before
        or receipt.binding.input_entry_count
        != receipt.input_entry_count_before
    ):
        raise PermissionError("capability effect profile content drifted")
    canonical_bytes = (
        json.dumps(
            expected,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if output_bytes != canonical_bytes:
        raise PermissionError("capability effect profile encoding drifted")
    payload = receipt.safe_payload()
    if not _SHA256.fullmatch(str(payload.get("receipt_hash") or "")):
        raise PermissionError("capability effect receipt hash is malformed")
    return payload


def _validate_role_spec(spec: PortableArtifactRoleSpec) -> None:
    registered = (
        (
            spec.role == TASK_DECLARED_PRIMARY_INPUT_ROLE
            and spec.artifact_format is ArtifactFormat.TABULAR
            and spec.capability == PROFILE_DELIMITED_TABLE_CAPABILITY
        )
        or (
            spec.role == TASK_DECLARED_OFFICE_COLLECTION_ROLE
            and spec.artifact_format is ArtifactFormat.OFFICE_DOCUMENT
            and spec.capability
            == INVENTORY_OFFICE_DOCUMENT_COLLECTION_CAPABILITY
        )
    )
    if (
        not registered
        or not _SHA256.fullmatch(spec.source_graph_hash)
        or not re.fullmatch(r"recipe_[0-9a-f]{20}", spec.source_recipe_id)
    ):
        raise PortableArtifactResolutionError(
            "portable artifact role spec is outside the closed registry"
        )


def _role_policy_for_spec(spec: PortableArtifactRoleSpec) -> str:
    if spec.capability == PROFILE_DELIMITED_TABLE_CAPABILITY:
        return PORTABLE_ARTIFACT_ROLE_POLICY_VERSION
    if spec.capability == INVENTORY_OFFICE_DOCUMENT_COLLECTION_CAPABILITY:
        return OFFICE_COLLECTION_ROLE_POLICY_VERSION
    raise PortableArtifactResolutionError(
        "portable artifact role spec is outside the closed registry"
    )


def _canonical_current_item_locator(value: str) -> str | None:
    locator = str(value or "").strip().rstrip(".,;:)]}\"'")
    if not _TASK_PATH.fullmatch(locator) or len(locator) > 300:
        return None
    path = PurePosixPath(locator)
    if len(path.parts) < 3 or path.parts[:2] != ("/", "root"):
        return None
    components = path.parts[2:]
    if any(part in {"", ".", ".."} for part in components):
        return None
    lowered = {part.lower() for part in components}
    stems = {PurePosixPath(part.lower()).stem for part in components}
    if (lowered | stems).intersection(_FORBIDDEN_COMPONENTS):
        return None
    return str(path)


def _contained_runtime_path(
    root: Path,
    container_locator: str,
    *,
    require_file: bool,
    require_directory: bool = False,
) -> Path | None:
    if require_file and require_directory:
        return None
    locator = _canonical_current_item_locator(container_locator)
    if locator is None:
        return None
    relative = PurePosixPath(locator).relative_to("/root")
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            return None
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return None
    if require_file and not resolved.is_file():
        return None
    if require_directory and not resolved.is_dir():
        return None
    return resolved


def _runtime_root_entries(root: Path) -> tuple[tuple[str, str], ...]:
    entries: list[tuple[str, str]] = []
    for current_text, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_text)
        directory_names.sort()
        file_names.sort()
        retained_directories: list[str] = []
        for name in directory_names:
            child = current / name
            kind = "l" if child.is_symlink() else "d"
            entries.append((kind, _container_locator_for_host_path(root, child)))
            if kind == "d":
                retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in file_names:
            child = current / name
            kind = "l" if child.is_symlink() else "f"
            entries.append((kind, _container_locator_for_host_path(root, child)))
        if len(entries) > MAX_RUNTIME_DISCOVERY_ENTRIES:
            raise PortableArtifactResolutionError(
                "runtime-tree discovery exceeds the fixed entry bound"
            )
    return tuple(entries)


def _container_locator_for_host_path(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise PortableArtifactResolutionError(
            "runtime-tree entry escaped the current runtime root"
        ) from exc
    return str(PurePosixPath("/root").joinpath(*relative.parts))


def _office_collection_tree_snapshot(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise PortableArtifactResolutionError(
            "office document collection is not a safe directory"
        )
    try:
        children = sorted(path.iterdir(), key=lambda row: row.name)
    except OSError as exc:
        raise PortableArtifactResolutionError(
            "office document collection could not be listed"
        ) from exc
    if not children or len(children) > MAX_OFFICE_COLLECTION_FILES:
        raise PortableArtifactResolutionError(
            "office document collection count is outside the fixed bound"
        )
    tree_entries: list[dict[str, Any]] = []
    inventory_entries: list[dict[str, Any]] = []
    total_size = 0
    for child in children:
        suffix = child.suffix.lower()
        if (
            child.is_symlink()
            or not child.is_file()
            or suffix not in _SUPPORTED_OFFICE_COLLECTION_SUFFIXES
            or not _OFFICE_COLLECTION_FILENAME.fullmatch(child.name)
            or len(child.name.encode("utf-8")) > 1024
        ):
            raise PortableArtifactResolutionError(
                "office document collection contains an unsupported entry"
            )
        try:
            size = child.stat().st_size
            content_sha256 = _sha256_file(child)
            if child.stat().st_size != size:
                raise PortableArtifactResolutionError(
                    "office document collection changed during fingerprinting"
                )
        except OSError as exc:
            raise PortableArtifactResolutionError(
                "office document collection entry could not be fingerprinted"
            ) from exc
        total_size += size
        if total_size > MAX_OFFICE_COLLECTION_BYTES:
            raise PortableArtifactResolutionError(
                "office document collection exceeds the fixed byte bound"
            )
        inventory = {
            "filename": child.name,
            "extension": suffix,
            "size": size,
        }
        inventory_entries.append(inventory)
        tree_entries.append(
            {
                **inventory,
                "content_sha256": content_sha256,
            }
        )
    return {
        "tree_sha256": stable_hash({"entries": tree_entries}),
        "total_size": total_size,
        "file_count": len(tree_entries),
        "inventory_entries": inventory_entries,
    }


def _prepare_capability_output_path(
    root: Path,
    output_locator: str,
) -> Path:
    locator = _canonical_current_item_locator(output_locator)
    if locator is None or not locator.startswith(
        str(CAPABILITY_OUTPUT_ROOT) + "/"
    ):
        raise RestrictedCapabilityExecutionError(
            "capability output escaped its fixed task-local root"
        )
    relative = PurePosixPath(locator).relative_to("/root")
    current = root
    for component in relative.parts[:-1]:
        current = current / component
        if current.exists():
            if current.is_symlink() or not current.is_dir():
                raise RestrictedCapabilityExecutionError(
                    "capability output parent is not a safe directory"
                )
        else:
            current.mkdir(mode=0o700)
    output = current / relative.name
    try:
        output.parent.resolve(strict=True).relative_to(root)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise RestrictedCapabilityExecutionError(
            "capability output parent escaped the runtime root"
        ) from exc
    if output.exists() or output.is_symlink():
        raise RestrictedCapabilityExecutionError(
            "capability output already exists"
        )
    return output


def _profile_delimited_table(
    path: Path,
    *,
    delimiter: str,
) -> dict[str, Any]:
    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="strict",
            newline="",
        ) as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            try:
                columns = next(reader)
            except StopIteration as exc:
                raise RestrictedCapabilityExecutionError(
                    "delimited table has no header"
                ) from exc
            if not columns or len(columns) > MAX_TABULAR_COLUMNS:
                raise RestrictedCapabilityExecutionError(
                    "delimited table column count is outside the fixed bound"
                )
            minimum_width = len(columns)
            maximum_width = len(columns)
            record_count = 0
            for row in reader:
                record_count += 1
                if record_count > MAX_TABULAR_RECORDS:
                    raise RestrictedCapabilityExecutionError(
                        "delimited table record count exceeds the fixed bound"
                    )
                minimum_width = min(minimum_width, len(row))
                maximum_width = max(maximum_width, len(row))
                if maximum_width > MAX_TABULAR_COLUMNS:
                    raise RestrictedCapabilityExecutionError(
                        "delimited table record width exceeds the fixed bound"
                    )
    except (UnicodeDecodeError, csv.Error, OSError) as exc:
        raise RestrictedCapabilityExecutionError(
            "delimited table could not be parsed by the registered capability"
        ) from exc
    return {
        "columns": columns,
        "record_count": record_count,
        "column_count": len(columns),
        "minimum_record_width": minimum_width,
        "maximum_record_width": maximum_width,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
