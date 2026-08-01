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
from .phase2b_freeze_v1 import (
    Phase2BExactFreeze,
    frozen_phase2b_exact_freeze,
)
from .phase3_certificate_v1 import OutsideFrozenClosureClaim
from .phase3_closure_preflight import phase3_closure_capacity_preflight_report
from .phase3_dsl_v1 import (
    OBSERVED_OMITTED_SINK_CONTROL,
    ODD_REDUCTION_TARGET,
    OLD_DSL_V1,
)

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
    "Phase2BExactFreeze",
    "PredictionBundle",
    "PublicEvidenceBundle",
    "OLD_DSL_V1",
    "ODD_REDUCTION_TARGET",
    "OBSERVED_OMITTED_SINK_CONTROL",
    "OutsideFrozenClosureClaim",
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
    "frozen_phase2b_exact_freeze",
    "phase3_closure_capacity_preflight_report",
]

__version__ = "0.2.0"
