"""Hegel Machine: bounded structural recognition and theory evolution."""

from .milestones import PHASE2A, PHASE2B, PHASE2R, PHASE3A, PHASE3B, PHASE3C
from .phase2b_selector import (
    CandidateEvaluation,
    CandidateGridCommitment,
    CandidateGridCell,
    TypedSelectorDecision,
    select_typed_candidate_evaluations,
)
from .phase2b_wire import PredictionBundle, PublicEvidenceBundle

from .recognition import (
    RecognitionDecision,
    RecognitionDisposition,
    RecognitionPolicy,
    StructuralProjection,
    UnboundStructuralEpisode,
    recognize_structural_law,
    replay_recognition_decision,
)

from .schema import (
    EvidenceReceipt,
    LawKind,
    RelationLaw,
    TheoryPatch,
    TheoryState,
)

__all__ = [
    "EvidenceReceipt",
    "LawKind",
    "PHASE2A",
    "PHASE2B",
    "PHASE2R",
    "PHASE3A",
    "PHASE3B",
    "PHASE3C",
    "PredictionBundle",
    "PublicEvidenceBundle",
    "RecognitionDecision",
    "RecognitionDisposition",
    "RecognitionPolicy",
    "RelationLaw",
    "StructuralProjection",
    "TheoryPatch",
    "TheoryState",
    "TypedSelectorDecision",
    "UnboundStructuralEpisode",
    "CandidateEvaluation",
    "CandidateGridCommitment",
    "CandidateGridCell",
    "recognize_structural_law",
    "replay_recognition_decision",
    "select_typed_candidate_evaluations",
]

__version__ = "0.2.0"
