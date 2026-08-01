"""Evidence governance, conservative certification, and version authorization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from datetime import datetime
from math import isfinite
import re
from types import MappingProxyType

from .hashing import stable_hash
from .schema import (
    AuthorityRole,
    EvidenceKind,
    EvidenceReceipt,
    EvidenceSplit,
    FrameworkStatus,
    Observation,
    PatchCoordinate,
    PromotionDecision,
    ReductionMap,
    TheoryPatch,
    TheoryState,
    payload_dict,
    require_tuple,
)


SEMANTIC_METRICS = frozenset(
    {
        "semantic_retrieval_score",
        "embedding_similarity",
        "llm_self_reported_confidence",
        "legacy_fixture_pass",
    }
)
CLAIM_EVIDENCE_KINDS = frozenset(
    {
        EvidenceKind.PROOF,
        EvidenceKind.EXECUTABLE_TEST,
        EvidenceKind.PHYSICAL_OR_SIMULATION,
        EvidenceKind.HELDOUT_HUMAN,
    }
)
SEMANTIC_EVIDENCE_KIND = EvidenceKind.SEMANTIC_RETRIEVAL
SEMANTIC_ACTOR_ROLE = AuthorityRole.GENERATOR
SEMANTIC_ALLOWED_SPLITS = frozenset({EvidenceSplit.TRAIN})
EMPIRICAL_EXECUTION_REQUIRED_METRICS = frozenset(
    {"unseen_prediction_success", "hard_negative_rejection"}
)
EMPIRICAL_FORBIDDEN_EVIDENCE_KINDS = frozenset({EvidenceKind.PROOF})
GATE_POLICY_SCHEMA_VERSION = "hegel-machine-gate-policy/1"
GATE_POLICY_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class SealedHoldoutManifest:
    theory_version_id: str
    candidate_id: str
    patch_content_id: str
    evaluator_epoch: str
    evaluator_version: str
    gate_policy_id: str
    probe_registry_id: str
    observations: tuple[Observation, ...]
    train_observation_ids: tuple[str, ...]
    validation_observation_ids: tuple[str, ...]
    old_success_observation_ids: tuple[str, ...]
    holdout_observation_ids: tuple[str, ...]
    hard_negative_ids: tuple[str, ...]
    registered_at: str
    opened_at: str
    independent_custodian_id: str
    generator_excluded: bool
    opening_nonce: str

    def __post_init__(self) -> None:
        for name in (
            "observations",
            "train_observation_ids",
            "validation_observation_ids",
            "old_success_observation_ids",
            "holdout_observation_ids",
            "hard_negative_ids",
        ):
            require_tuple(getattr(self, name), f"manifest {name}")
        if not all(
            (
                self.theory_version_id,
                self.candidate_id,
                self.patch_content_id,
                self.evaluator_epoch,
                self.evaluator_version,
                self.gate_policy_id,
                self.probe_registry_id,
                self.opening_nonce,
            )
        ):
            raise ValueError("sealed manifest has an empty theory or policy binding")
        partitions = (
            set(self.train_observation_ids),
            set(self.validation_observation_ids),
            set(self.old_success_observation_ids),
            set(self.holdout_observation_ids),
            set(self.hard_negative_ids),
        )
        if any(not partition for partition in partitions):
            raise ValueError("sealed manifest needs every required evidence partition")
        sequences = (
            self.train_observation_ids,
            self.validation_observation_ids,
            self.old_success_observation_ids,
            self.holdout_observation_ids,
            self.hard_negative_ids,
        )
        if any(len(sequence) != len(set(sequence)) for sequence in sequences):
            raise ValueError("sealed manifest repeats an observation id")
        if any(
            re.fullmatch(r"observation_[0-9a-f]{64}", observation_id) is None
            for sequence in sequences
            for observation_id in sequence
        ):
            raise ValueError(
                "sealed manifest partitions require content-addressed observations"
            )
        for index, left in enumerate(partitions):
            if any(left.intersection(right) for right in partitions[index + 1 :]):
                raise ValueError("sealed manifest partitions overlap")
        observation_map = {
            observation.content_id: observation for observation in self.observations
        }
        if len(observation_map) != len(self.observations):
            raise ValueError("sealed manifest repeats observation content")
        committed_ids = set().union(*partitions)
        if set(observation_map) != committed_ids:
            raise ValueError(
                "sealed manifest partitions do not bind the observation registry"
            )
        expected_splits = {
            **{
                observation_id: EvidenceSplit.TRAIN
                for observation_id in self.train_observation_ids
            },
            **{
                observation_id: EvidenceSplit.VALIDATION
                for observation_id in self.validation_observation_ids
            },
            **{
                observation_id: EvidenceSplit.OLD_SUCCESS
                for observation_id in self.old_success_observation_ids
            },
            **{
                observation_id: EvidenceSplit.HOLDOUT
                for observation_id in self.holdout_observation_ids
            },
            **{
                observation_id: EvidenceSplit.HARD_NEGATIVE
                for observation_id in self.hard_negative_ids
            },
        }
        for observation_id, observation in observation_map.items():
            if observation.split is not expected_splits[observation_id]:
                raise ValueError("observation content appears in the wrong split")
            if re.fullmatch(r"[0-9a-f]{64}", observation.provenance_hash) is None:
                raise ValueError("sealed observation lacks a provenance SHA-256")
        if not self.independent_custodian_id or not self.generator_excluded:
            raise ValueError("sealed manifest needs an independent custodian")
        if _parse_time(self.registered_at) >= _parse_time(self.opened_at):
            raise ValueError("holdout manifest was not sealed before opening")

    @property
    def manifest_id(self) -> str:
        return stable_hash(self, prefix="sealed_manifest_")


@dataclass(frozen=True, slots=True)
class EvidenceLedger:
    theory_version_id: str
    candidate_id: str
    evaluator_epoch: str
    data_cutoff: str
    source_manifest_id: str
    holdout_manifest: SealedHoldoutManifest | None
    receipts: tuple[EvidenceReceipt, ...]

    def __post_init__(self) -> None:
        require_tuple(self.receipts, "ledger receipts")
        if not self.receipts:
            raise ValueError("evidence ledger cannot be empty")
        if not self.source_manifest_id or not self.data_cutoff:
            raise ValueError("ledger needs a source manifest and data cutoff")
        if self.holdout_manifest is not None:
            if self.source_manifest_id != self.holdout_manifest.manifest_id:
                raise ValueError("ledger source id does not match sealed manifest")
            if _parse_time(self.holdout_manifest.opened_at) > _parse_time(
                self.data_cutoff
            ):
                raise ValueError("holdout opened after the ledger data cutoff")
        receipt_ids: set[str] = set()
        content_ids: set[str] = set()
        for receipt in self.receipts:
            if receipt.theory_version_id != self.theory_version_id:
                raise ValueError("ledger mixes theory versions")
            if receipt.candidate_id != self.candidate_id:
                raise ValueError("ledger mixes candidates")
            if receipt.evaluator_epoch != self.evaluator_epoch:
                raise ValueError("cross-epoch evidence aggregation is forbidden")
            if receipt.data_cutoff != self.data_cutoff:
                raise ValueError("ledger mixes data cutoffs")
            if receipt.receipt_id in receipt_ids:
                raise ValueError("ledger contains a duplicate receipt id")
            if receipt.content_id in content_ids:
                raise ValueError("ledger contains duplicate evidence content")
            receipt_ids.add(receipt.receipt_id)
            content_ids.add(receipt.content_id)

    @property
    def ledger_id(self) -> str:
        return stable_hash(self, prefix="evidence_ledger_")

    @property
    def sealed_holdout(self) -> bool:
        return self.holdout_manifest is not None

    def measured(self, metric: str) -> tuple[EvidenceReceipt, ...]:
        if metric in SEMANTIC_METRICS:
            raise ValueError("semantic scores cannot be queried by the promotion gate")
        return tuple(
            receipt
            for receipt in self.receipts
            if receipt.metric == metric
            and receipt.kind is not EvidenceKind.SEMANTIC_RETRIEVAL
        )

    def conservative_value(self, metric: str, *, higher_is_better: bool) -> float:
        receipts = self.measured(metric)
        if not receipts:
            raise KeyError(f"missing measured gate evidence: {metric}")
        values = [receipt.value for receipt in receipts]
        return min(values) if higher_is_better else max(values)


@dataclass(frozen=True, slots=True)
class GateThresholds:
    residual_explanation: float = 0.75
    old_success_preservation: float = 0.95
    limiting_case_reduction: float = 0.90
    expressivity_gain: float = 0.01
    compression_gain: float = 0.01
    unseen_prediction_success: float = 0.60
    hard_negative_rejection: float = 0.95
    regression_cost: float = 0.02
    complexity_cost: float = 1.0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(
                    f"gate threshold {item.name} must be finite and in [0, 1]"
                )

    @property
    def policy_id(self) -> str:
        """Return the complete policy id while preserving the v0.1 API."""

        return _policy_with_thresholds(self).policy_id


DEFAULT_GATE_THRESHOLDS = GateThresholds()


@dataclass(frozen=True, slots=True)
class MetricPolicy:
    splits: frozenset[EvidenceSplit]
    actor_role: AuthorityRole
    probe_ids: frozenset[str]
    higher_is_better: bool
    require_independent: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.splits, frozenset) or not isinstance(
            self.probe_ids, frozenset
        ):
            raise TypeError("metric policy splits and probes must be frozensets")
        if not self.splits or not self.probe_ids:
            raise ValueError("metric policy needs allowed splits and probes")
        if any(not isinstance(split, EvidenceSplit) for split in self.splits):
            raise TypeError("metric policy contains a non-EvidenceSplit value")
        if not isinstance(self.actor_role, AuthorityRole):
            raise TypeError("metric policy actor must be an AuthorityRole")
        if any(
            not isinstance(probe_id, str) or not probe_id
            for probe_id in self.probe_ids
        ):
            raise TypeError("metric policy probe ids must be nonempty strings")
        if not isinstance(self.higher_is_better, bool) or not isinstance(
            self.require_independent, bool
        ):
            raise TypeError("metric policy flags must be booleans")


METRIC_POLICIES: Mapping[str, MetricPolicy] = MappingProxyType({
    "residual_explanation": MetricPolicy(
        frozenset({EvidenceSplit.TRAIN, EvidenceSplit.VALIDATION}),
        AuthorityRole.EVALUATOR,
        frozenset({"probe_exact_residual"}),
        True,
    ),
    "old_success_preservation": MetricPolicy(
        frozenset({EvidenceSplit.OLD_SUCCESS}),
        AuthorityRole.EVALUATOR,
        frozenset({"probe_exact_residual"}),
        True,
    ),
    "limiting_case_reduction": MetricPolicy(
        frozenset({EvidenceSplit.OLD_SUCCESS, EvidenceSplit.VALIDATION}),
        AuthorityRole.EVALUATOR,
        frozenset({"probe_exact_residual"}),
        True,
    ),
    "expressivity_gain": MetricPolicy(
        frozenset({EvidenceSplit.TRAIN, EvidenceSplit.VALIDATION}),
        AuthorityRole.EVALUATOR,
        frozenset({"probe_exact_residual"}),
        True,
    ),
    "compression_gain": MetricPolicy(
        frozenset({EvidenceSplit.TRAIN, EvidenceSplit.VALIDATION}),
        AuthorityRole.EVALUATOR,
        frozenset({"probe_exact_residual"}),
        True,
    ),
    "unseen_prediction_success": MetricPolicy(
        frozenset({EvidenceSplit.HOLDOUT}),
        AuthorityRole.EVALUATOR,
        frozenset({"probe_exact_residual"}),
        True,
    ),
    "hard_negative_rejection": MetricPolicy(
        frozenset({EvidenceSplit.HARD_NEGATIVE}),
        AuthorityRole.FALSIFIER,
        frozenset({"probe_hard_negative"}),
        True,
    ),
    "regression_cost": MetricPolicy(
        frozenset({EvidenceSplit.OLD_SUCCESS}),
        AuthorityRole.EVALUATOR,
        frozenset({"probe_exact_residual"}),
        False,
    ),
    "complexity_cost": MetricPolicy(
        frozenset({EvidenceSplit.TRAIN, EvidenceSplit.VALIDATION}),
        AuthorityRole.EVALUATOR,
        frozenset({"probe_exact_residual"}),
        False,
    ),
})

SEALED_METRIC_SPLITS: Mapping[str, EvidenceSplit] = MappingProxyType({
    "residual_explanation": EvidenceSplit.VALIDATION,
    "old_success_preservation": EvidenceSplit.OLD_SUCCESS,
    "limiting_case_reduction": EvidenceSplit.OLD_SUCCESS,
    "expressivity_gain": EvidenceSplit.VALIDATION,
    "compression_gain": EvidenceSplit.VALIDATION,
    "unseen_prediction_success": EvidenceSplit.HOLDOUT,
    "hard_negative_rejection": EvidenceSplit.HARD_NEGATIVE,
    "regression_cost": EvidenceSplit.OLD_SUCCESS,
    "complexity_cost": EvidenceSplit.VALIDATION,
})
SEALED_PARTITION_SPLITS = frozenset(
    {
        EvidenceSplit.TRAIN,
        EvidenceSplit.VALIDATION,
        EvidenceSplit.OLD_SUCCESS,
        EvidenceSplit.HOLDOUT,
        EvidenceSplit.HARD_NEGATIVE,
    }
)


@dataclass(frozen=True, slots=True)
class GatePolicySpec:
    """Canonical, immutable description of every promotion-gate rule.

    A sealed manifest binds this object rather than only the numeric thresholds.
    All mapping-like inputs are normalized into sorted immutable tuples before
    hashing so rule order cannot change the identifier.
    """

    schema_version: str
    policy_version: str
    thresholds: GateThresholds
    metric_policies: tuple[tuple[str, MetricPolicy], ...]
    sealed_metric_splits: tuple[tuple[str, EvidenceSplit], ...]
    sealed_partition_splits: frozenset[EvidenceSplit]
    claim_evidence_kinds: frozenset[EvidenceKind]
    semantic_metrics: frozenset[str]
    semantic_evidence_kind: EvidenceKind
    semantic_actor_role: AuthorityRole
    semantic_allowed_splits: frozenset[EvidenceSplit]
    empirical_execution_required_metrics: frozenset[str]
    empirical_forbidden_evidence_kinds: frozenset[EvidenceKind]

    def __post_init__(self) -> None:
        require_tuple(self.metric_policies, "gate policy metric policies")
        require_tuple(
            self.sealed_metric_splits,
            "gate policy sealed metric splits",
        )
        if not self.schema_version or not self.policy_version:
            raise ValueError("gate policy needs schema and policy versions")
        if not isinstance(self.thresholds, GateThresholds):
            raise TypeError("gate policy thresholds must be GateThresholds")
        for name in (
            "claim_evidence_kinds",
            "sealed_partition_splits",
            "semantic_metrics",
            "semantic_allowed_splits",
            "empirical_execution_required_metrics",
            "empirical_forbidden_evidence_kinds",
        ):
            if not isinstance(getattr(self, name), frozenset):
                raise TypeError(f"gate policy {name} must be a frozenset")
        if not isinstance(self.semantic_evidence_kind, EvidenceKind):
            raise TypeError("semantic evidence kind must be an EvidenceKind")
        if not isinstance(self.semantic_actor_role, AuthorityRole):
            raise TypeError("semantic actor role must be an AuthorityRole")
        metric_names = tuple(name for name, _ in self.metric_policies)
        if not metric_names or len(metric_names) != len(set(metric_names)):
            raise ValueError("gate policy metric names must be unique and nonempty")
        if self.metric_policies != tuple(sorted(self.metric_policies)):
            raise ValueError("gate policy metric policies must be canonicalized")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(metric_policy, MetricPolicy)
            for name, metric_policy in self.metric_policies
        ):
            raise TypeError("gate policy contains an invalid metric policy")
        sealed_names = tuple(name for name, _ in self.sealed_metric_splits)
        if (
            len(sealed_names) != len(set(sealed_names))
            or self.sealed_metric_splits != tuple(sorted(self.sealed_metric_splits))
        ):
            raise ValueError("sealed metric splits must be canonical and unique")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(split, EvidenceSplit)
            for name, split in self.sealed_metric_splits
        ):
            raise TypeError("gate policy contains an invalid sealed split")
        threshold_names = {item.name for item in fields(self.thresholds)}
        if set(metric_names) != threshold_names or set(sealed_names) != threshold_names:
            raise ValueError(
                "thresholds, metric policies, and sealed splits must cover one schema"
            )
        metric_policy_map = dict(self.metric_policies)
        if self.sealed_partition_splits != SEALED_PARTITION_SPLITS:
            raise ValueError(
                "gate policy must bind the fixed five-partition manifest schema"
            )
        if self.semantic_metrics.intersection(metric_names):
            raise ValueError("semantic metrics and promotion metrics must be disjoint")
        for metric, split in self.sealed_metric_splits:
            if split not in self.sealed_partition_splits:
                raise ValueError(
                    "sealed metric split is absent from the five-partition manifest"
                )
            if split not in metric_policy_map[metric].splits:
                raise ValueError(
                    "sealed metric split is not allowed by its metric policy"
                )
        unseen_policy = metric_policy_map["unseen_prediction_success"]
        if (
            unseen_policy.splits != frozenset({EvidenceSplit.HOLDOUT})
            or not unseen_policy.require_independent
            or dict(self.sealed_metric_splits)["unseen_prediction_success"]
            is not EvidenceSplit.HOLDOUT
        ):
            raise ValueError(
                "unseen prediction evidence must be independent sealed holdout"
            )
        if (
            not self.claim_evidence_kinds
            or any(
                not isinstance(kind, EvidenceKind)
                for kind in self.claim_evidence_kinds
            )
            or self.semantic_evidence_kind in self.claim_evidence_kinds
        ):
            raise ValueError("claim and semantic evidence kinds must be disjoint")
        if not self.semantic_metrics or any(
            not isinstance(metric, str) or not metric
            for metric in self.semantic_metrics
        ):
            raise ValueError("semantic metric exclusions must be nonempty names")
        if not self.semantic_allowed_splits or any(
            not isinstance(split, EvidenceSplit)
            for split in self.semantic_allowed_splits
        ):
            raise ValueError("semantic evidence needs explicit allowed splits")
        if not self.empirical_execution_required_metrics.issubset(
            set(metric_names)
        ):
            raise ValueError("empirical execution rules cite an unknown metric")
        if not self.empirical_forbidden_evidence_kinds or any(
            not isinstance(kind, EvidenceKind)
            for kind in self.empirical_forbidden_evidence_kinds
        ):
            raise ValueError("empirical evidence-kind exclusions are invalid")
        # Force canonical serialization at construction time.
        stable_hash(self)

    @classmethod
    def from_components(
        cls,
        *,
        thresholds: GateThresholds,
        metric_policies: Mapping[str, MetricPolicy] | None = None,
        sealed_metric_splits: Mapping[str, EvidenceSplit] | None = None,
        sealed_partition_splits: frozenset[EvidenceSplit] = (
            SEALED_PARTITION_SPLITS
        ),
        claim_evidence_kinds: frozenset[EvidenceKind] = CLAIM_EVIDENCE_KINDS,
        semantic_metrics: frozenset[str] = SEMANTIC_METRICS,
        semantic_evidence_kind: EvidenceKind = SEMANTIC_EVIDENCE_KIND,
        semantic_actor_role: AuthorityRole = SEMANTIC_ACTOR_ROLE,
        semantic_allowed_splits: frozenset[EvidenceSplit] = SEMANTIC_ALLOWED_SPLITS,
        empirical_execution_required_metrics: frozenset[str] = (
            EMPIRICAL_EXECUTION_REQUIRED_METRICS
        ),
        empirical_forbidden_evidence_kinds: frozenset[EvidenceKind] = (
            EMPIRICAL_FORBIDDEN_EVIDENCE_KINDS
        ),
        schema_version: str = GATE_POLICY_SCHEMA_VERSION,
        policy_version: str = GATE_POLICY_VERSION,
    ) -> "GatePolicySpec":
        metric_policies = (
            METRIC_POLICIES if metric_policies is None else metric_policies
        )
        sealed_metric_splits = (
            SEALED_METRIC_SPLITS
            if sealed_metric_splits is None
            else sealed_metric_splits
        )
        return cls(
            schema_version=schema_version,
            policy_version=policy_version,
            thresholds=thresholds,
            metric_policies=tuple(sorted(metric_policies.items())),
            sealed_metric_splits=tuple(sorted(sealed_metric_splits.items())),
            sealed_partition_splits=frozenset(sealed_partition_splits),
            claim_evidence_kinds=frozenset(claim_evidence_kinds),
            semantic_metrics=frozenset(semantic_metrics),
            semantic_evidence_kind=semantic_evidence_kind,
            semantic_actor_role=semantic_actor_role,
            semantic_allowed_splits=frozenset(semantic_allowed_splits),
            empirical_execution_required_metrics=frozenset(
                empirical_execution_required_metrics
            ),
            empirical_forbidden_evidence_kinds=frozenset(
                empirical_forbidden_evidence_kinds
            ),
        )

    @property
    def metric_policy_map(self) -> dict[str, MetricPolicy]:
        return dict(self.metric_policies)

    @property
    def sealed_metric_split_map(self) -> dict[str, EvidenceSplit]:
        return dict(self.sealed_metric_splits)

    @property
    def policy_id(self) -> str:
        return stable_hash(self, prefix="gate_policy_")


DEFAULT_GATE_POLICY = GatePolicySpec.from_components(
    thresholds=DEFAULT_GATE_THRESHOLDS
)


def _policy_with_thresholds(thresholds: GateThresholds) -> GatePolicySpec:
    if thresholds == DEFAULT_GATE_POLICY.thresholds:
        return DEFAULT_GATE_POLICY
    return replace(DEFAULT_GATE_POLICY, thresholds=thresholds)


def _policy_measured(
    ledger: EvidenceLedger,
    metric: str,
    policy: GatePolicySpec,
) -> tuple[EvidenceReceipt, ...]:
    if metric in policy.semantic_metrics:
        raise ValueError("semantic scores cannot be queried by the promotion gate")
    return tuple(
        receipt
        for receipt in ledger.receipts
        if receipt.metric == metric
        and receipt.kind is not policy.semantic_evidence_kind
    )


def _policy_conservative_value(
    ledger: EvidenceLedger,
    metric: str,
    policy: GatePolicySpec,
) -> float:
    metric_policy = policy.metric_policy_map[metric]
    receipts = _policy_measured(ledger, metric, policy)
    if not receipts:
        raise KeyError(f"missing measured gate evidence: {metric}")
    values = [receipt.value for receipt in receipts]
    return (
        min(values)
        if metric_policy.higher_is_better
        else max(values)
    )


@dataclass(frozen=True, slots=True)
class GateCheck:
    name: str
    measured_value: float | None
    threshold: float | None
    passed: bool
    reason: str

    def __post_init__(self) -> None:
        if self.measured_value is not None and not isfinite(self.measured_value):
            raise ValueError("gate check value must be finite")
        if self.threshold is not None and not isfinite(self.threshold):
            raise ValueError("gate check threshold must be finite")
        if not self.name or not self.reason:
            raise ValueError("gate check needs a name and rationale")


@dataclass(frozen=True, slots=True)
class ConservativeExtensionCertificate:
    parent_version_id: str
    candidate_id: str
    patch_id: str
    patch_content_id: str
    evaluator_epoch: str
    reduction_map_id: str
    reduction_map_content_id: str
    receipt_ids: tuple[str, ...]
    evidence_ledger_id: str
    gate_policy_id: str
    proposed_child_version_id: str | None
    checks: tuple[GateCheck, ...]
    decision: PromotionDecision
    required_next_tests: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("receipt_ids", "checks", "required_next_tests"):
            require_tuple(getattr(self, name), f"certificate {name}")
        identifiers = (
            self.parent_version_id,
            self.candidate_id,
            self.patch_id,
            self.patch_content_id,
            self.evaluator_epoch,
            self.reduction_map_id,
            self.reduction_map_content_id,
            self.evidence_ledger_id,
            self.gate_policy_id,
        )
        if any(not identifier for identifier in identifiers):
            raise ValueError("certificate has an empty content binding")
        if not self.checks or not self.receipt_ids:
            raise ValueError("certificate needs checks and measured receipt ids")
        if len({check.name for check in self.checks}) != len(self.checks):
            raise ValueError("certificate repeats a gate check")
        if self.decision is PromotionDecision.ACTIVE_SCOPED:
            if not all(check.passed for check in self.checks):
                raise ValueError("active certificate contains a failed check")
            if not self.proposed_child_version_id:
                raise ValueError("active certificate must bind the proposed child")
        # Force canonical serialization at construction time.
        stable_hash(self)

    @property
    def certificate_id(self) -> str:
        return stable_hash(self, prefix="extension_cert_")


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 evidence timestamp: {value}") from exc
    if parsed.utcoffset() is None:
        raise ValueError(f"evidence timestamp lacks a timezone: {value}")
    return parsed


def compile_patch(parent: TheoryState, patch: TheoryPatch) -> TheoryState:
    """Compile exactly one supported coordinate into a proposed child.

    Phase 2 authorizes only a scoped extension. Other patch schemas exist for
    later phases but cannot mutate a theory through this compiler.
    """

    if patch.parent_version_id != parent.version_id:
        raise ValueError("patch parent mismatch")
    if patch.coordinate is not PatchCoordinate.SCOPE:
        raise ValueError(f"no active compiler for coordinate: {patch.coordinate.value}")
    payload = payload_dict(patch.payload)
    if set(payload) - {"operation", "scope", "law_library"}:
        raise ValueError("scope patch payload contains another coordinate")
    if payload.get("operation") != "add_scope":
        raise ValueError("scope compiler only supports add_scope")
    scope = payload.get("scope")
    if not isinstance(scope, str) or patch.scope != (scope,):
        raise ValueError("scope payload and patch contract disagree")
    if payload.get("law_library", "unchanged") != "unchanged":
        raise ValueError("scope patch may not alter the law library")

    child = replace(
        parent,
        parent_version_id=parent.version_id,
        scope=tuple(sorted(set(parent.scope) | {scope})),
        conditional_description_length=(
            parent.conditional_description_length
            + patch.conditional_description_length
        ),
    )
    allowed_differences = {
        "parent_version_id",
        "scope",
        "conditional_description_length",
    }
    for item in fields(parent):
        if item.name not in allowed_differences and getattr(parent, item.name) != getattr(
            child, item.name
        ):
            raise AssertionError(f"scope compiler changed {item.name}")
    return child


def _evidence_contract_checks(
    *,
    parent: TheoryState,
    patch: TheoryPatch,
    ledger: EvidenceLedger,
    policy: GatePolicySpec,
) -> tuple[GateCheck, ...]:
    thresholds = policy.thresholds
    metric_policies = policy.metric_policy_map
    registered_probes = {
        (probe.probe_id, probe.version): probe for probe in parent.probes
    }
    actors = {
        assignment.role: assignment.actor_id
        for assignment in patch.authority_assignments
    }
    failures: list[str] = []
    for receipt in ledger.receipts:
        if (receipt.probe_id, receipt.probe_version) not in registered_probes:
            failures.append(f"{receipt.receipt_id}:unregistered_probe")
        if receipt.data_cutoff != parent.data_cutoff:
            failures.append(f"{receipt.receipt_id}:cutoff_mismatch")
        if receipt.kind is policy.semantic_evidence_kind:
            if receipt.actor_id != actors[policy.semantic_actor_role]:
                failures.append(f"{receipt.receipt_id}:semantic_actor")
            if receipt.split not in policy.semantic_allowed_splits:
                failures.append(f"{receipt.receipt_id}:semantic_holdout_access")
            continue
        metric_policy = metric_policies.get(receipt.metric)
        if metric_policy is None:
            failures.append(f"{receipt.receipt_id}:unknown_claim_metric")
            continue
        if not 0 <= receipt.value <= 1:
            failures.append(f"{receipt.receipt_id}:metric_out_of_unit_range")
        if receipt.split not in metric_policy.splits:
            failures.append(f"{receipt.receipt_id}:wrong_split")
        if receipt.kind not in policy.claim_evidence_kinds:
            failures.append(f"{receipt.receipt_id}:wrong_evidence_kind")
        if metric_policy.require_independent and not receipt.independent:
            failures.append(f"{receipt.receipt_id}:not_independent")
        if receipt.actor_id != actors[metric_policy.actor_role]:
            failures.append(f"{receipt.receipt_id}:wrong_authority")
        if receipt.kind in policy.empirical_forbidden_evidence_kinds and (
            receipt.metric in policy.empirical_execution_required_metrics
        ):
            failures.append(f"{receipt.receipt_id}:empirical_metric_requires_execution")
        if receipt.probe_id not in metric_policy.probe_ids:
            failures.append(f"{receipt.receipt_id}:wrong_probe")
        expected_threshold = getattr(thresholds, receipt.metric)
        if receipt.higher_is_better is not metric_policy.higher_is_better:
            failures.append(f"{receipt.receipt_id}:wrong_metric_direction")
        if receipt.threshold != expected_threshold:
            failures.append(f"{receipt.receipt_id}:wrong_receipt_threshold")
        if not receipt.passed:
            failures.append(f"{receipt.receipt_id}:failed_receipt")

    hard_negative_receipts = _policy_measured(
        ledger,
        "hard_negative_rejection",
        policy,
    )
    covered_negatives = {
        observation_id
        for receipt in hard_negative_receipts
        for observation_id in receipt.observation_ids
    }
    missing_negatives = set(patch.hard_negative_ids) - covered_negatives
    if missing_negatives:
        failures.append("missing_hard_negatives:" + ",".join(sorted(missing_negatives)))

    manifest = ledger.holdout_manifest
    predictions = {prediction.content_id: prediction for prediction in patch.predictions}
    evaluated_prediction_ids = {
        receipt.preregistration_id
        for receipt in _policy_measured(
            ledger,
            "unseen_prediction_success",
            policy,
        )
        if receipt.preregistration_id is not None
    }
    if evaluated_prediction_ids != set(patch.prediction_ids):
        failures.append("incomplete_preregistered_prediction_coverage")
    for receipt in _policy_measured(
        ledger,
        "unseen_prediction_success",
        policy,
    ):
        prediction = predictions.get(receipt.preregistration_id or "")
        if prediction is None:
            failures.append(f"{receipt.receipt_id}:unknown_preregistration")
            continue
        if _parse_time(prediction.registered_at_cutoff) >= _parse_time(
            receipt.data_cutoff
        ):
            failures.append(f"{receipt.receipt_id}:prediction_not_preregistered")
        if manifest is not None and _parse_time(
            prediction.registered_at_cutoff
        ) >= _parse_time(manifest.opened_at):
            failures.append(
                f"{receipt.receipt_id}:prediction_registered_after_holdout_open"
            )

    if manifest is not None:
        expected_manifest_bindings = (
            manifest.theory_version_id == parent.version_id
            and manifest.candidate_id == patch.candidate_id
            and manifest.patch_content_id == patch.content_id
            and manifest.evaluator_epoch == parent.evaluator.epoch
            and manifest.evaluator_version == parent.evaluator.version
            and manifest.gate_policy_id == policy.policy_id
            and manifest.probe_registry_id
            == stable_hash(parent.probes, prefix="probe_registry_")
        )
        if not expected_manifest_bindings:
            failures.append("sealed_manifest_context_binding_mismatch")
        if any(
            observation.data_cutoff != parent.data_cutoff
            for observation in manifest.observations
        ):
            failures.append("sealed_observation_cutoff_mismatch")
        if manifest.independent_custodian_id in set(actors.values()):
            failures.append("holdout_custodian_is_an_assigned_authority")
        observed_holdout = {
            observation_id
            for receipt in _policy_measured(
                ledger,
                "unseen_prediction_success",
                policy,
            )
            for observation_id in receipt.observation_ids
        }
        if observed_holdout != set(manifest.holdout_observation_ids):
            failures.append("holdout_receipts_not_in_sealed_partition")
        if set(patch.hard_negative_ids) != set(manifest.hard_negative_ids):
            failures.append("patch_negatives_do_not_match_sealed_partition")
        partition_ids = {
            EvidenceSplit.TRAIN: set(manifest.train_observation_ids),
            EvidenceSplit.VALIDATION: set(manifest.validation_observation_ids),
            EvidenceSplit.OLD_SUCCESS: set(
                manifest.old_success_observation_ids
            ),
            EvidenceSplit.HOLDOUT: set(manifest.holdout_observation_ids),
            EvidenceSplit.HARD_NEGATIVE: set(manifest.hard_negative_ids),
        }
        for receipt in ledger.receipts:
            allowed_ids = partition_ids.get(receipt.split)
            if allowed_ids is None or not set(receipt.observation_ids).issubset(
                allowed_ids
            ):
                failures.append(f"{receipt.receipt_id}:outside_sealed_partition")
        observed_by_split = {
            split: {
                observation_id
                for receipt in ledger.receipts
                if receipt.split is split
                for observation_id in receipt.observation_ids
            }
            for split in partition_ids
        }
        for split, expected_ids in partition_ids.items():
            if split is EvidenceSplit.TRAIN:
                # Train data is committed for leakage auditing, not required
                # to appear in a promotion receipt. Semantic retrieval remains
                # optional and never becomes claim evidence.
                continue
            if observed_by_split[split] != expected_ids:
                failures.append(f"incomplete_sealed_partition:{split.value}")
        for metric, required_split in policy.sealed_metric_splits:
            metric_observation_ids = {
                observation_id
                for receipt in _policy_measured(ledger, metric, policy)
                for observation_id in receipt.observation_ids
            }
            if metric_observation_ids != partition_ids[required_split]:
                failures.append(f"incomplete_metric_partition:{metric}")

    return (
        GateCheck(
            "evidence_contract",
            None,
            None,
            not failures,
            "registered_probes_splits_actors_and_cutoffs"
            if not failures
            else ";".join(failures),
        ),
    )


def _measured_check(
    ledger: EvidenceLedger,
    name: str,
    threshold: float,
    *,
    policy: GatePolicySpec,
) -> GateCheck:
    try:
        value = _policy_conservative_value(ledger, name, policy)
    except KeyError as exc:
        return GateCheck(name, None, threshold, False, str(exc))
    higher_is_better = policy.metric_policy_map[name].higher_is_better
    passed = value >= threshold if higher_is_better else value <= threshold
    return GateCheck(
        name,
        value,
        threshold,
        passed,
        "measured_pass" if passed else "measured_fail",
    )


def evaluate_conservative_extension(
    *,
    parent: TheoryState,
    patch: TheoryPatch,
    ledger: EvidenceLedger,
    reduction_map: ReductionMap,
    thresholds: GateThresholds = DEFAULT_GATE_THRESHOLDS,
    policy: GatePolicySpec | None = None,
) -> ConservativeExtensionCertificate:
    """Build a bound certificate from registered, independently measured receipts."""

    if policy is None:
        gate_policy = _policy_with_thresholds(thresholds)
    else:
        if (
            thresholds != DEFAULT_GATE_THRESHOLDS
            and thresholds != policy.thresholds
        ):
            raise ValueError("explicit gate policy and thresholds disagree")
        gate_policy = policy
    thresholds = gate_policy.thresholds
    try:
        proposed_child = compile_patch(parent, patch)
        compiler_ok = True
        compiler_reason = "single_coordinate_scope_compiler"
    except ValueError as exc:
        proposed_child = None
        compiler_ok = False
        compiler_reason = str(exc)

    structural = [
        GateCheck(
            "parent_binding",
            None,
            None,
            patch.parent_version_id == parent.version_id
            and ledger.theory_version_id == parent.version_id,
            "parent_versions_match"
            if patch.parent_version_id == parent.version_id
            and ledger.theory_version_id == parent.version_id
            else "parent_version_mismatch",
        ),
        GateCheck(
            "candidate_binding",
            None,
            None,
            patch.candidate_id == ledger.candidate_id,
            "candidate_ids_match"
            if patch.candidate_id == ledger.candidate_id
            else "candidate_id_mismatch",
        ),
        GateCheck(
            "evaluator_epoch",
            None,
            None,
            ledger.evaluator_epoch == parent.evaluator.epoch,
            "epoch_frozen"
            if ledger.evaluator_epoch == parent.evaluator.epoch
            else "evaluator_epoch_mismatch",
        ),
        GateCheck(
            "single_coordinate_compiler",
            None,
            None,
            compiler_ok,
            compiler_reason,
        ),
        GateCheck(
            "reduction_map",
            reduction_map.maximum_error,
            thresholds.regression_cost,
            reduction_map.parent_version_id == parent.version_id
            and reduction_map.child_candidate_id == ledger.candidate_id
            and reduction_map.reduction_id == patch.reduction_map_id
            and reduction_map.maximum_error <= thresholds.regression_cost,
            "reduction_bound_and_identity_supplied"
            if reduction_map.parent_version_id == parent.version_id
            and reduction_map.child_candidate_id == ledger.candidate_id
            and reduction_map.reduction_id == patch.reduction_map_id
            and reduction_map.maximum_error <= thresholds.regression_cost
            else "missing_or_mismatched_reduction",
        ),
    ]
    structural.extend(
        _evidence_contract_checks(
            parent=parent,
            patch=patch,
            ledger=ledger,
            policy=gate_policy,
        )
    )
    structural.append(
        GateCheck(
            "sealed_holdout",
            None,
            None,
            ledger.sealed_holdout,
            "sealed_manifest_confirmed"
            if ledger.sealed_holdout
            else "controlled_or_unsealed_qualification",
        )
    )
    structural.append(
        GateCheck(
            "external_trust_root",
            None,
            None,
            False,
            "external_custodian_signature_verifier_not_implemented",
        )
    )

    checks = structural + [
        _measured_check(
            ledger,
            "residual_explanation",
            thresholds.residual_explanation,
            policy=gate_policy,
        ),
        _measured_check(
            ledger,
            "old_success_preservation",
            thresholds.old_success_preservation,
            policy=gate_policy,
        ),
        _measured_check(
            ledger,
            "limiting_case_reduction",
            thresholds.limiting_case_reduction,
            policy=gate_policy,
        ),
        _measured_check(
            ledger,
            "expressivity_gain",
            thresholds.expressivity_gain,
            policy=gate_policy,
        ),
        _measured_check(
            ledger,
            "compression_gain",
            thresholds.compression_gain,
            policy=gate_policy,
        ),
        _measured_check(
            ledger,
            "unseen_prediction_success",
            thresholds.unseen_prediction_success,
            policy=gate_policy,
        ),
        _measured_check(
            ledger,
            "hard_negative_rejection",
            thresholds.hard_negative_rejection,
            policy=gate_policy,
        ),
        _measured_check(
            ledger,
            "regression_cost",
            thresholds.regression_cost,
            policy=gate_policy,
        ),
        _measured_check(
            ledger,
            "complexity_cost",
            thresholds.complexity_cost,
            policy=gate_policy,
        ),
    ]

    unseen = _policy_measured(
        ledger,
        "unseen_prediction_success",
        gate_policy,
    )
    unseen_policy = gate_policy.metric_policy_map["unseen_prediction_success"]
    unseen_split = gate_policy.sealed_metric_split_map[
        "unseen_prediction_success"
    ]
    preregistered_holdout = bool(unseen) and all(
        receipt.split is unseen_split
        and (not unseen_policy.require_independent or receipt.independent)
        and receipt.preregistration_id in patch.prediction_ids
        for receipt in unseen
    )
    checks.append(
        GateCheck(
            "independent_preregistered_holdout",
            None,
            None,
            preregistered_holdout,
            "independent_holdout_is_preregistered"
            if preregistered_holdout
            else "missing_independent_preregistered_holdout",
        )
    )

    all_pass = all(check.passed for check in checks)
    seal_check = next(check for check in checks if check.name == "sealed_holdout")
    external_trust_check = next(
        check for check in checks if check.name == "external_trust_root"
    )
    all_except_external_trust = all(
        check.passed for check in checks if check.name != "external_trust_root"
    )
    all_except_seal_and_external_trust = all(
        check.passed
        for check in checks
        if check.name not in {"sealed_holdout", "external_trust_root"}
    )
    old_success = next(
        check for check in checks if check.name == "old_success_preservation"
    )
    residual = next(check for check in checks if check.name == "residual_explanation")
    if all_pass:
        decision = PromotionDecision.ACTIVE_SCOPED
        next_tests = ("fresh_recheck_in_new_epoch", "survival_across_domain")
    elif all_except_external_trust and not external_trust_check.passed:
        decision = PromotionDecision.CANDIDATE
        next_tests = ("implement_and_verify_external_trust_root",)
    elif (
        all_except_seal_and_external_trust
        and not seal_check.passed
        and not external_trust_check.passed
    ):
        decision = PromotionDecision.CANDIDATE
        next_tests = (
            "run_sealed_external_holdout",
            "implement_and_verify_external_trust_root",
        )
    elif not old_success.passed:
        decision = PromotionDecision.REJECT
        next_tests = ("repair_old_success_regression",)
    elif residual.passed:
        decision = PromotionDecision.BRANCH_ONLY
        next_tests = tuple(
            f"repair:{check.name}" for check in checks if not check.passed
        )
    else:
        decision = PromotionDecision.REJECT
        next_tests = ("explain_residual_with_independent_evidence",)

    receipt_ids = tuple(
        sorted(
            receipt.content_id
            for receipt in ledger.receipts
            if receipt.kind is not gate_policy.semantic_evidence_kind
        )
    )
    return ConservativeExtensionCertificate(
        parent_version_id=parent.version_id,
        candidate_id=ledger.candidate_id,
        patch_id=patch.patch_id,
        patch_content_id=patch.content_id,
        evaluator_epoch=ledger.evaluator_epoch,
        reduction_map_id=reduction_map.reduction_id,
        reduction_map_content_id=reduction_map.content_id,
        receipt_ids=receipt_ids,
        evidence_ledger_id=ledger.ledger_id,
        gate_policy_id=gate_policy.policy_id,
        proposed_child_version_id=(
            proposed_child.version_id if proposed_child is not None else None
        ),
        checks=tuple(checks),
        decision=decision,
        required_next_tests=next_tests,
    )


LEGAL_TRANSITIONS: dict[FrameworkStatus, frozenset[FrameworkStatus]] = {
    FrameworkStatus.DRAFT: frozenset(
        {FrameworkStatus.CANDIDATE_BRANCH, FrameworkStatus.REJECTED}
    ),
    FrameworkStatus.CANDIDATE_BRANCH: frozenset(
        {
            FrameworkStatus.BRANCH_ONLY,
            FrameworkStatus.CANDIDATE,
            FrameworkStatus.REJECTED,
        }
    ),
    FrameworkStatus.BRANCH_ONLY: frozenset(
        {FrameworkStatus.CANDIDATE, FrameworkStatus.REJECTED}
    ),
    FrameworkStatus.CANDIDATE: frozenset(
        {
            FrameworkStatus.ACTIVE_SCOPED,
            FrameworkStatus.BRANCH_ONLY,
            FrameworkStatus.REJECTED,
        }
    ),
    FrameworkStatus.ACTIVE_SCOPED: frozenset(
        {
            FrameworkStatus.GENERAL,
            FrameworkStatus.DEMOTED,
            FrameworkStatus.DEPRECATED,
            FrameworkStatus.CONTRADICTED,
        }
    ),
    FrameworkStatus.GENERAL: frozenset(
        {
            FrameworkStatus.DEMOTED,
            FrameworkStatus.DEPRECATED,
            FrameworkStatus.CONTRADICTED,
        }
    ),
    FrameworkStatus.DEMOTED: frozenset(
        {FrameworkStatus.CANDIDATE, FrameworkStatus.DEPRECATED}
    ),
    FrameworkStatus.DEPRECATED: frozenset(),
    FrameworkStatus.REJECTED: frozenset(),
    FrameworkStatus.CONTRADICTED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class BranchRecord:
    branch_id: str
    parent_version_id: str
    candidate_id: str
    patch_content_id: str
    status: FrameworkStatus

    def __post_init__(self) -> None:
        if not all(
            (
                self.branch_id,
                self.parent_version_id,
                self.candidate_id,
                self.patch_content_id,
            )
        ):
            raise ValueError("branch record has an empty content binding")


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    branch_id: str
    from_status: FrameworkStatus
    to_status: FrameworkStatus
    patch: TheoryPatch
    ledger: EvidenceLedger
    reduction_map: ReductionMap
    certificate: ConservativeExtensionCertificate
    gate_policy: GatePolicySpec = DEFAULT_GATE_POLICY

    def __post_init__(self) -> None:
        if not self.branch_id:
            raise ValueError("evaluation record needs a branch id")


@dataclass(frozen=True, slots=True)
class TheoryVersionGraph:
    """Validated shadow lifecycle graph for the current Phase-2 kernel.

    Active state append is deliberately disabled until an external signature
    trust root exists. Every non-active transition stores enough inputs for
    deterministic replay instead of trusting a caller-supplied certificate.
    """

    states: tuple[TheoryState, ...]
    branches: tuple[BranchRecord, ...] = ()
    evaluation_records: tuple[EvaluationRecord, ...] = ()

    def __post_init__(self) -> None:
        require_tuple(self.states, "version graph states")
        require_tuple(self.branches, "version graph branches")
        require_tuple(
            self.evaluation_records,
            "version graph evaluation records",
        )
        state_map = {state.version_id: state for state in self.states}
        if len(state_map) != len(self.states):
            raise ValueError("version graph contains duplicate immutable states")
        if len(self.states) != 1 or self.states[0].parent_version_id is not None:
            raise ValueError(
                "the current graph requires exactly one genesis theory; "
                "active edges need an external trust root"
            )
        branch_map = {branch.branch_id: branch for branch in self.branches}
        if len(branch_map) != len(self.branches):
            raise ValueError("lifecycle graph contains duplicate branch ids")
        events_by_branch: dict[str, list[EvaluationRecord]] = {
            branch_id: [] for branch_id in branch_map
        }
        for record in self.evaluation_records:
            if record.branch_id not in events_by_branch:
                raise ValueError("evaluation record references an absent branch")
            events_by_branch[record.branch_id].append(record)

        targets = {
            PromotionDecision.REJECT: FrameworkStatus.REJECTED,
            PromotionDecision.BRANCH_ONLY: FrameworkStatus.BRANCH_ONLY,
            PromotionDecision.CANDIDATE: FrameworkStatus.CANDIDATE,
        }
        for branch_id, branch in branch_map.items():
            records = events_by_branch[branch_id]
            if not records:
                if branch.status not in {
                    FrameworkStatus.DRAFT,
                    FrameworkStatus.CANDIDATE_BRANCH,
                }:
                    raise ValueError(
                        "advanced branch status lacks a replayable evaluation record"
                    )
                continue
            current = records[0].from_status
            if current not in {
                FrameworkStatus.DRAFT,
                FrameworkStatus.CANDIDATE_BRANCH,
            }:
                raise ValueError("evaluation history starts from an invalid branch state")
            for record in records:
                if record.from_status is not current:
                    raise ValueError("evaluation history has a discontinuous transition")
                if (
                    record.patch.parent_version_id != branch.parent_version_id
                    or record.patch.candidate_id != branch.candidate_id
                    or record.patch.content_id != branch.patch_content_id
                ):
                    raise ValueError("evaluation record and branch binding disagree")
                try:
                    parent = state_map[branch.parent_version_id]
                except KeyError as exc:
                    raise ValueError("branch parent is absent from the graph") from exc
                recomputed = evaluate_conservative_extension(
                    parent=parent,
                    patch=record.patch,
                    ledger=record.ledger,
                    reduction_map=record.reduction_map,
                    policy=record.gate_policy,
                )
                if recomputed != record.certificate:
                    raise ValueError("stored certificate fails deterministic replay")
                if recomputed.decision is PromotionDecision.ACTIVE_SCOPED:
                    raise ValueError(
                        "active transition lacks an external trust-root adapter"
                    )
                expected_target = targets[recomputed.decision]
                if record.to_status is not expected_target:
                    raise ValueError("evaluation decision and lifecycle target disagree")
                if expected_target not in LEGAL_TRANSITIONS[current]:
                    raise ValueError(
                        f"illegal lifecycle transition: {current} -> {expected_target}"
                    )
                current = expected_target
            if branch.status is not current:
                raise ValueError("branch status disagrees with replayed evaluation history")

    @property
    def edges(self) -> tuple[()]:
        return ()

    @property
    def authoritative(self) -> bool:
        """The current in-process shadow ledger has no signed writer identity."""

        return False

    @property
    def branch_statuses(self) -> tuple[tuple[str, FrameworkStatus], ...]:
        return tuple(
            sorted((branch.branch_id, branch.status) for branch in self.branches)
        )

    @property
    def negative_evidence_ids(self) -> tuple[str, ...]:
        negatives: set[str] = set()
        for record in self.evaluation_records:
            negatives.update(record.patch.hard_negative_ids)
            for receipt in record.ledger.receipts:
                if (
                    receipt.kind is not record.gate_policy.semantic_evidence_kind
                    and not receipt.passed
                ):
                    negatives.update(receipt.observation_ids)
        return tuple(sorted(negatives))

    @property
    def certificate_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    record.certificate.certificate_id
                    for record in self.evaluation_records
                }
            )
        )

    def record_evaluation(
        self,
        *,
        branch_id: str,
        parent: TheoryState,
        patch: TheoryPatch,
        ledger: EvidenceLedger,
        reduction_map: ReductionMap,
        policy: GatePolicySpec = DEFAULT_GATE_POLICY,
    ) -> "TheoryVersionGraph":
        """Recompute and retain a non-active decision with replay inputs.

        This method never appends a theory state and therefore is not an
        authorization path. Its records remain local, non-authoritative shadow
        history until evaluator/falsifier writer signatures are implemented.
        """

        branches = {branch.branch_id: branch for branch in self.branches}
        if branch_id not in branches:
            raise ValueError("branch is absent from the lifecycle ledger")
        branch = branches[branch_id]
        if (
            branch.parent_version_id != parent.version_id
            or branch.candidate_id != patch.candidate_id
            or branch.patch_content_id != patch.content_id
        ):
            raise ValueError("branch does not bind this parent and patch")
        certificate = evaluate_conservative_extension(
            parent=parent,
            patch=patch,
            ledger=ledger,
            reduction_map=reduction_map,
            policy=policy,
        )
        if certificate.decision is PromotionDecision.ACTIVE_SCOPED:
            raise ValueError("active decisions require an external trust-root adapter")
        target = {
            PromotionDecision.REJECT: FrameworkStatus.REJECTED,
            PromotionDecision.BRANCH_ONLY: FrameworkStatus.BRANCH_ONLY,
            PromotionDecision.CANDIDATE: FrameworkStatus.CANDIDATE,
        }[certificate.decision]
        if target not in LEGAL_TRANSITIONS[branch.status]:
            raise ValueError(
                f"illegal lifecycle transition: {branch.status} -> {target}"
            )
        updated_branches = tuple(
            replace(item, status=target) if item.branch_id == branch_id else item
            for item in self.branches
        )
        record = EvaluationRecord(
            branch_id,
            branch.status,
            target,
            patch,
            ledger,
            reduction_map,
            certificate,
            policy,
        )
        return replace(
            self,
            branches=updated_branches,
            evaluation_records=self.evaluation_records + (record,),
        )


@dataclass(frozen=True, slots=True)
class PromotionResult:
    graph: TheoryVersionGraph
    child: TheoryState
    certificate: ConservativeExtensionCertificate


def authorize_promotion(
    *,
    graph: TheoryVersionGraph,
    branch_id: str,
    parent: TheoryState,
    patch: TheoryPatch,
    ledger: EvidenceLedger,
    reduction_map: ReductionMap,
    promoter_actor_id: str,
    policy: GatePolicySpec = DEFAULT_GATE_POLICY,
) -> PromotionResult:
    """Reject active writes until an external signature trust root exists."""

    branches = {branch.branch_id: branch for branch in graph.branches}
    branch = branches.get(branch_id)
    if branch is None or branch.status is not FrameworkStatus.CANDIDATE:
        raise ValueError("branch must be candidate_framework before promotion")
    promoter = next(
        assignment.actor_id
        for assignment in patch.authority_assignments
        if assignment.role is AuthorityRole.PROMOTER
    )
    if promoter_actor_id != promoter:
        raise ValueError("caller is not the assigned promotion authority")
    known = {state.version_id for state in graph.states}
    if parent.version_id not in known:
        raise ValueError("parent theory is absent from the graph")
    if (
        branch.parent_version_id != parent.version_id
        or branch.candidate_id != patch.candidate_id
        or branch.patch_content_id != patch.content_id
    ):
        raise ValueError("branch does not bind this parent and patch")

    certificate = evaluate_conservative_extension(
        parent=parent,
        patch=patch,
        ledger=ledger,
        reduction_map=reduction_map,
        policy=policy,
    )
    if certificate.decision is not PromotionDecision.ACTIVE_SCOPED:
        raise ValueError(
            "active promotion is disabled until an external trust root is verified"
        )
    raise ValueError(
        "active promotion is disabled until an external trust root is verified"
    )
