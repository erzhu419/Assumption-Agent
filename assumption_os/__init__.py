"""Recursive Assumption Agent core package."""

from .graph_memory import JsonlGraphStore, SimpleAssumptionGraph
from .schema import (
    AssumptionEdge,
    AssumptionNode,
    AssumptionType,
    EdgeType,
    EvidenceRecord,
    HypothesisKind,
    ResidualType,
    TrialManifest,
    TrialStatus,
)
from .selector import MetaproductivitySelector, SelectionScore, SelectionWeights
from .conditioned_eval import (
    ConditionedEvalRow,
    ConditionedNodeSummary,
    GateDecision,
    GateThresholds,
    RouteLabel,
    evaluate_node,
    route_problem_to_node,
)

__all__ = [
    "AssumptionEdge",
    "AssumptionNode",
    "AssumptionType",
    "ConditionedEvalRow",
    "ConditionedNodeSummary",
    "EdgeType",
    "EvidenceRecord",
    "GateDecision",
    "GateThresholds",
    "HypothesisKind",
    "JsonlGraphStore",
    "MetaproductivitySelector",
    "ResidualType",
    "RouteLabel",
    "SelectionScore",
    "SelectionWeights",
    "SimpleAssumptionGraph",
    "TrialManifest",
    "TrialStatus",
    "evaluate_node",
    "route_problem_to_node",
]
