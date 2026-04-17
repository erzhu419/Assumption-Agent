"""Phase 2 configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
PHASE0_DIR = PROJECT_ROOT.parent / "phase zero"
PHASE1_DIR = PROJECT_ROOT.parent / "phase one"

EXECUTIONS_DIR = PHASE0_DIR / "experience_log" / "executions"
DISTILLED_DIR = PHASE0_DIR / "experience_log" / "distilled"
PENDING_REVIEW = DISTILLED_DIR / "pending_review"
APPLIED = DISTILLED_DIR / "applied"
REJECTED = DISTILLED_DIR / "rejected"
PENDING_HUMAN = DISTILLED_DIR / "pending_human"

KB_DIR = PHASE0_DIR / "kb" / "strategies"
CHANGE_HISTORY_DIR = PHASE0_DIR / "change_history"

# Evaluator thresholds
INFO_SCORE_THRESHOLD = 0.3     # below => drop record
DIMINISHING_BASE = 1.0         # 1 / (1 + ln(1 + similar_count))

# Distiller thresholds
MIN_SUPPORT_PER_CANDIDATE = 2  # cluster size required
CLUSTER_SIM_THRESHOLD = 0.75   # cosine similarity for clustering new conditions

# Integrator thresholds (by stability tier)
STABILITY_TIERS = {
    "foundational": {
        "min_evidence_to_modify": 20,
        "max_confidence_delta_per_update": 0.05,
        "auto_apply_eligible": False,
    },
    "empirical": {
        "min_evidence_to_modify": 5,
        "max_confidence_delta_per_update": 0.15,
        "auto_apply_eligible": True,
    },
    "tentative": {
        "min_evidence_to_modify": 2,
        "max_confidence_delta_per_update": 0.20,
        "auto_apply_eligible": True,
    },
}

NEW_CONDITION_DEFAULT_TIER = "tentative"
NEW_CONDITION_DEFAULT_CONFIDENCE = 0.65
