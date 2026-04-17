"""Phase 3 configuration: discrete state/action space, thresholds."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
PHASE0_DIR = PROJECT_ROOT.parent / "phase zero"
PHASE1_DIR = PROJECT_ROOT.parent / "phase one"
PHASE2_DIR = PROJECT_ROOT.parent / "phase two"

KB_DIR = PHASE0_DIR / "kb" / "strategies"
EXECUTIONS_DIR = PHASE0_DIR / "experience_log" / "executions"
FORMAL_KB_DIR = PROJECT_ROOT / "formal_kb"
KERNELS_DIR = FORMAL_KB_DIR / "kernels"

# ---------------------------------------------------------------------------
# Problem feature discretization (from Phase 3 dev doc §1.3)
# State space size: 3 × 2 × 2 × 3 × 3 = 108 discrete states
# ---------------------------------------------------------------------------
FEATURE_BINS = {
    "coupling_estimate":        [("low", 0.0, 0.30), ("medium", 0.30, 0.70), ("high", 0.70, 1.01)],
    "decomposability":          [("low", 0.0, 0.40), ("high", 0.40, 1.01)],
    "has_baseline":             [("no", False), ("yes", True)],
    "information_completeness": [("low", 0.0, 0.40), ("medium", 0.40, 0.70), ("high", 0.70, 1.01)],
    "component_count":          [("few", 0, 4), ("moderate", 4, 8), ("many", 8, 10**6)],
}

# Ordered list of feature names (determines state encoding)
FEATURE_ORDER = list(FEATURE_BINS.keys())

# ---------------------------------------------------------------------------
# Action space (16 abstract action categories, Phase 3 dev doc §1.3)
# ---------------------------------------------------------------------------
ACTION_SPACE = [
    "decompose",            # 分解问题为子问题
    "isolate_variable",     # 隔离单一变量
    "simplify",             # 简化/降维
    "analogize",            # 类比已知问题
    "negate",               # 反向推理/反证
    "test_boundary",        # 测试边界条件
    "enumerate",            # 枚举/穷举
    "estimate_update",      # 估计后更新（贝叶斯式）
    "build_incrementally",  # 增量构建
    "compare_cases",        # 对比案例（求同/求异）
    "abstract",             # 抽象化/泛化
    "relax_constraint",     # 松弛约束
    "seek_falsification",   # 寻找反例
    "evaluate_select",      # 评估并选择（满意化/最优化）
    "switch_perspective",   # 切换视角
    "no_action",            # 无特定行动
]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_SPACE)}
NUM_ACTIONS = len(ACTION_SPACE)

# ---------------------------------------------------------------------------
# Isomorphism detection thresholds (will be calibrated in 3.4)
# ---------------------------------------------------------------------------
ISO_THRESHOLDS = {
    "fisher_max":       0.50,   # Fisher-Rao distance below this = similar
    "spectral_max":     0.20,
    "kl_asymmetry_min": 0.60,   # |log(KL_ab/KL_ba)| < this => symmetric
    "kl_asymmetry_max": 1.67,
}

# Known expected isomorphism pairs (from Phase 3 dev doc §6.2)
EXPECTED_ISO_PAIRS = [
    ("S01", "S15"),  # controlled variable ↔ incremental build (partial)
    ("S16", "S03"),  # method of agreement ↔ analogical reasoning (partial)
    ("S06", "S09"),  # special-to-general ↔ dimensional reduction (partial)
]
