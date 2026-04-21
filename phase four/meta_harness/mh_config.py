"""Phase 4 Meta-Harness configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PHASE0_DIR = PROJECT_ROOT.parent / "phase zero"
PHASE1_DIR = PROJECT_ROOT.parent / "phase one"
PHASE2_DIR = PROJECT_ROOT.parent / "phase two"
PHASE3_DIR = PROJECT_ROOT.parent / "phase three"

KB_DIR = PHASE0_DIR / "kb" / "strategies"
COMP_DIR = PHASE0_DIR / "kb" / "compositions"
EXECUTIONS_DIR = PHASE0_DIR / "experience_log" / "executions"

META_HARNESS_DIR = Path(__file__).parent
HARNESSES_DIR = META_HARNESS_DIR / "harnesses"
LOGS_DIR = META_HARNESS_DIR / "logs"

# Search configuration
NUM_ITERATIONS = 10
SEARCH_SET_SIZE = 20
HELD_OUT_SIZE = 100
JUDGE_MAX_TOKENS = 256
HARNESS_MAX_TOKENS = 2400

# History budget — proposer sees this many prior harnesses
MAX_HISTORY_HARNESSES = 8
