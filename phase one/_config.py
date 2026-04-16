"""
Phase 1 configuration: strategy IDs, action space, hyperparameters.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
PHASE0_DIR = PROJECT_ROOT.parent / "phase zero"
PHASE_HALF_DIR = PROJECT_ROOT.parent / "phase half"
KB_DIR = PHASE0_DIR / "kb" / "strategies"
COMP_DIR = PHASE0_DIR / "kb" / "compositions"
PROBLEMS_DIR = PHASE0_DIR / "benchmark" / "problems"
ANNOTATIONS_DIR = PHASE0_DIR / "benchmark" / "annotations"

# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------
STRATEGY_IDS = [
    "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10",
    "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20",
    "S21", "S22", "S23", "S24", "S25", "S26", "S27",
]

COMPOSITION_IDS = ["COMP_001", "COMP_002", "COMP_003", "COMP_004", "COMP_005"]

SPECIAL_ACTIONS = ["SPECIAL_GATHER_INFO"]

# Full action space: 27 strategies + 5 compositions + 1 special = 33
ACTION_SPACE = STRATEGY_IDS + COMPOSITION_IDS + SPECIAL_ACTIONS
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_SPACE)}
IDX_TO_ACTION = {i: a for a, i in ACTION_TO_IDX.items()}
NUM_ACTIONS = len(ACTION_SPACE)

# ---------------------------------------------------------------------------
# Feature dimensions
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 768          # sentence-transformer output dim
STRUCTURAL_FEATURES = 10     # coupling, decomp, baseline, info, reversibility, etc.
KB_MATCH_FEATURES = NUM_ACTIONS  # one match score per strategy
CROSS_PROBLEM_FEATURES = NUM_ACTIONS  # recent success rate per strategy

# Total input dim for MLP dispatcher
INPUT_DIM = EMBEDDING_DIM + STRUCTURAL_FEATURES + KB_MATCH_FEATURES + CROSS_PROBLEM_FEATURES

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
PPO_CONFIG = {
    "learning_rate": 1e-4,
    "clip_epsilon": 0.2,
    "value_loss_coef": 0.5,
    "entropy_coef": 0.05,
    "entropy_decay": 0.995,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "num_epochs_per_update": 4,
    "batch_size": 64,
    "max_episodes": 50000,
    "early_stopping_patience": 2000,
    "early_stopping_min_episodes": 5000,
    "early_stopping_threshold": 0.01,
}

# Model-based RL (world model integration)
MODEL_BASED_CONFIG = {
    "real_ratio": 0.1,          # 10% real execution
    "calibration_interval": 500,
    "cold_start_real_ratio": 0.5,  # 50% real during cold start
    "cold_start_episodes": 2000,
}

# Reward weights
REWARD_WEIGHTS = {
    "completion": 0.35,
    "consistency": 0.10,
    "efficiency": 0.10,
    "progress": 0.10,
    "step_progress": 0.20,
    "calibration": 0.15,
}

# Strategy collapse monitoring
COLLAPSE_CONFIG = {
    "window": 500,
    "threshold": 0.6,  # top-3 strategies > 60% = collapse
    "entropy_boost": 1.5,  # multiply entropy_coef by 1.5
    "boost_duration": 2000,
}
