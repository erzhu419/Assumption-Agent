"""Core data model for the Recursive Assumption Agent.

The project used to store related ideas in separate forms: strategy JSON,
wisdom entries, residual analyses, and Exp82 typed hypotheses.  This module
defines the shared substrate those artifacts can be lifted into: every useful
piece of agent behavior is an assumption with evidence, risks, verifiers, and
residuals.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_id(prefix: str, *parts: object, length: int = 12) -> str:
    raw = "\x1f".join(str(p) for p in parts if p is not None)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


class AssumptionType(str, Enum):
    OBJECT = "object"
    METHOD = "method"
    EVALUATOR = "evaluator"
    MEMORY = "memory"
    WORLD_MODEL = "world_model"
    ALIGNMENT = "alignment"
    SELF_MODIFICATION = "self_modification"
    HARNESS = "harness"
    RETRIEVAL = "retrieval"
    STRATEGY = "strategy"
    RESIDUAL = "residual"
    CASE = "case"
    VERIFIER = "verifier"


class HypothesisKind(str, Enum):
    CLAIM = "claim"
    FEATURE = "feature"
    CONSTRAINT = "constraint"
    DECOMPOSITION = "decomposition"
    VERIFICATION = "verification"
    HP_CHANGE = "hp_change"
    RETRIEVAL_POLICY = "retrieval_policy"
    EVALUATOR_POLICY = "evaluator_policy"
    FORMAL_MAPPING = "formal_mapping"


class EdgeType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SPECIALIZES = "specializes"
    GENERALIZES = "generalizes"
    IS_ANALOGY_OF = "is_analogy_of"
    IS_FORMAL_ISOMORPHISM_OF = "is_formal_isomorphism_of"
    USES_EVALUATOR = "uses_evaluator"
    GENERATED_FROM_RESIDUAL = "generated_from_residual"
    FAILED_BECAUSE = "failed_because"
    EXECUTION_LAPSE_OF = "execution_lapse_of"
    REPLACES = "replaces"
    DEPENDS_ON = "depends_on"
    DERIVED_FROM = "derived_from"
    HAS_CASE = "has_case"
    HAS_VERIFIER = "has_verifier"
    HAS_RESIDUAL = "has_residual"


class ResidualType(str, Enum):
    NO_RESIDUAL = "no_residual"
    EXECUTION_LAPSE = "execution_lapse"
    OPTIMIZATION = "optimization"
    ASSUMPTION_DEFECT = "assumption_defect"
    DISCOVERY = "discovery"
    EVALUATOR_DEFECT = "evaluator_defect"
    MEMORY_DEFECT = "memory_defect"
    SIMULATOR_DEFECT = "simulator_defect"
    UNKNOWN = "unknown"


class TrialStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    OBSERVED = "observed"
    FAILED = "failed"


@dataclass
class AssumptionNode:
    id: str
    type: AssumptionType | str
    claim: str
    kind: HypothesisKind | str = HypothesisKind.CLAIM
    formal_form: dict[str, Any] | None = None
    context_conditions: list[str] = field(default_factory=list)
    predicted_effects: list[str] = field(default_factory=list)
    risk_predictions: list[str] = field(default_factory=list)
    verifiers: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    residual_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5
    metaproductivity: float = 0.0
    status: str = "active"
    tags: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_timestamp)
    updated_at: str = field(default_factory=utc_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return _enum_to_value(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssumptionNode":
        d = dict(data)
        d["type"] = AssumptionType(d["type"])
        d["kind"] = HypothesisKind(d.get("kind", HypothesisKind.CLAIM))
        return cls(**d)


@dataclass
class AssumptionEdge:
    source: str
    target: str
    type: EdgeType | str
    weight: float = 1.0
    evidence: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_timestamp)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source, self.target, str(self.type))

    def to_dict(self) -> dict[str, Any]:
        return _enum_to_value(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AssumptionEdge":
        d = dict(data)
        d["type"] = EdgeType(d["type"])
        return cls(**d)


@dataclass
class EvidenceRecord:
    node_id: str
    source: str
    outcome: str
    metric: str | None = None
    value: float | None = None
    split: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    evidence_id: str = field(default_factory=lambda: stable_id("ev", uuid.uuid4().hex))
    timestamp: str = field(default_factory=utc_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return _enum_to_value(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceRecord":
        return cls(**dict(data))


@dataclass
class TrialManifest:
    """Falsifiable contract for one agent decision.

    This follows the reconstruction/AHE requirement: every retrieval, prompt,
    evaluator call, scaffold edit, or self-modification records what it assumes,
    why it was selected, what it predicts, and what actually happened.
    """

    problem_id: str
    action_type: str
    assumption: str
    why_selected: str
    expected_effect: str
    assumption_ids: list[str] = field(default_factory=list)
    component: str | None = None
    predicted_regressions: list[str] = field(default_factory=list)
    verifier: str | None = None
    verification_plan: str | None = None
    rollback_condition: str | None = None
    cost: float = 0.0
    observed_effect: str | None = None
    residual: str | None = None
    residual_type: ResidualType | str | None = None
    status: TrialStatus | str = TrialStatus.PENDING
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_trial_id: str | None = None
    trial_id: str = field(default_factory=lambda: stable_id("trial", uuid.uuid4().hex))
    timestamp: str = field(default_factory=utc_timestamp)

    def observe(
        self,
        observed_effect: str,
        *,
        residual: str | None = None,
        residual_type: ResidualType | str | None = None,
        status: TrialStatus | str = TrialStatus.OBSERVED,
    ) -> None:
        self.observed_effect = observed_effect
        self.residual = residual
        self.residual_type = residual_type
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return _enum_to_value(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrialManifest":
        d = dict(data)
        if d.get("residual_type"):
            d["residual_type"] = ResidualType(d["residual_type"])
        d["status"] = TrialStatus(d.get("status", TrialStatus.PENDING))
        return cls(**d)


@dataclass
class ActivatedSubgraph:
    query: str
    seed_ids: list[str]
    nodes: list[AssumptionNode]
    edges: list[AssumptionEdge]
    scores: dict[str, float]
    cases: list[AssumptionNode] = field(default_factory=list)
    residuals: list[AssumptionNode] = field(default_factory=list)
    verifiers: list[AssumptionNode] = field(default_factory=list)


def _enum_to_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _enum_to_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_enum_to_value(v) for v in value]
    return value
