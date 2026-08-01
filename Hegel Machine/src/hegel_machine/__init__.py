"""Hegel Machine: bounded structural recognition and theory evolution."""

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
    "RecognitionDecision",
    "RecognitionDisposition",
    "RecognitionPolicy",
    "RelationLaw",
    "StructuralProjection",
    "TheoryPatch",
    "TheoryState",
    "UnboundStructuralEpisode",
    "recognize_structural_law",
    "replay_recognition_decision",
]

__version__ = "0.2.0"
