"""
Build a Markov kernel K_S : X -> Δ(A) for each strategy.

For the MVP we blend two sources (as per Phase 3 dev doc §2.2):

  1. Prior: LLM-estimated action distribution for a "typical" problem. We ask
     the LLM once per strategy for the 16-dim action distribution; this is a
     state-independent prior shared across all x (Markov kernel's "column base").

  2. Empirical adjustment: per-state adjustment from the strategy's
     operational_steps. We classify each step into one of the 16 action categories
     (via keyword rules + LLM fallback) and count those categories.

The final kernel for each state is:
   K_S(x, ·) = (1 - α_x) · prior  +  α_x · step_dist
where α_x ramps with the strategy's known applicability to state x.

Implementation choices:
  * The per-state applicability weight uses the strategy's KB conditions:
    conditions with `derived_from` hints are checked against x's feature bins.
  * Where no hints match, we fall back to pure prior → uniform over actions
    that the strategy's operational steps touch.

This is deliberately simple. It gives a differentiable, deterministic mapping
from KB text → kernel matrix that we can refine later with experience counts.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import _config as cfg
from formalization.core.feature_discretizer import enumerate_state_space


# Rule-based action classification keywords (Chinese KB text)
ACTION_KEYWORDS: Dict[str, List[str]] = {
    "decompose":            ["分解", "子问题", "模块", "拆分", "划分", "切分"],
    "isolate_variable":     ["控制变量", "隔离", "单一", "固定其他", "逐一"],
    "simplify":             ["简化", "降维", "抽象化降低", "去除", "忽略细节"],
    "analogize":            ["类比", "相似", "比较已知", "参照"],
    "negate":               ["反证", "矛盾", "假设否", "反例"],
    "test_boundary":        ["边界", "极值", "极限", "极端"],
    "enumerate":            ["枚举", "穷举", "列举", "逐项"],
    "estimate_update":      ["估计", "贝叶斯", "先验", "后验", "更新", "概率"],
    "build_incrementally":  ["增量", "逐步构建", "迭代", "累积"],
    "compare_cases":        ["对比", "求同", "求异", "案例"],
    "abstract":             ["抽象", "泛化", "推广", "一般化"],
    "relax_constraint":     ["松弛", "放宽", "弱化约束"],
    "seek_falsification":   ["证伪", "反驳", "反例检验"],
    "evaluate_select":      ["评估", "选择", "决策", "满意", "最优"],
    "switch_perspective":   ["视角", "转换角度", "切换视角", "重新表述"],
}


def classify_step_text(text: str) -> str:
    """Rule-based: pick the action category with the most keyword hits.
    Ties broken by declaration order. Returns 'no_action' on no match."""
    if not text:
        return "no_action"
    best = "no_action"
    best_hits = 0
    for action, keywords in ACTION_KEYWORDS.items():
        hits = sum(1 for k in keywords if k in text)
        if hits > best_hits:
            best_hits = hits
            best = action
    return best


def _step_action_dist(strategy: Dict, llm_classifier=None) -> np.ndarray:
    """Action distribution across the strategy's operational_steps.

    If `llm_classifier` is provided, use LLM-based per-step probability
    distributions; otherwise fall back to keyword rules.
    """
    if llm_classifier is not None:
        return llm_classifier.strategy_step_distribution(strategy)

    counts = np.zeros(cfg.NUM_ACTIONS, dtype=np.float64)
    for step in strategy.get("operational_steps", []):
        text = step.get("action", "") + " " + (step.get("on_difficulty", "") or "")
        cat = classify_step_text(text)
        counts[cfg.ACTION_TO_IDX[cat]] += 1.0
        if step.get("on_difficulty"):
            alt = classify_step_text(step["on_difficulty"])
            if alt != cat:
                counts[cfg.ACTION_TO_IDX[alt]] += 0.5
    counts += 0.1
    return counts / counts.sum()


def _state_matches_condition(state_tuple: Tuple[str, ...], derived: Dict) -> bool:
    """Does this state satisfy a condition's derived_from hint?"""
    fkey = derived.get("feature_key")
    fdir = derived.get("feature_direction")
    if not fkey or not fdir:
        return False
    if fkey not in cfg.FEATURE_ORDER:
        return False
    idx = cfg.FEATURE_ORDER.index(fkey)
    bin_label = state_tuple[idx]
    # "high" direction matches the last (or second bin for 2-bin features),
    # "low" matches the first.
    labels = [b[0] for b in cfg.FEATURE_BINS[fkey]]
    if fdir == "high":
        return bin_label == labels[-1]
    if fdir == "low":
        return bin_label == labels[0]
    return False


def _state_applicability(strategy: Dict, state_tuple: Tuple[str, ...]) -> float:
    """
    Return α_x ∈ [0,1] indicating how applicable the strategy is to this state.
    Computed from applicability_conditions' derived_from hints + confidence.
    """
    ac = strategy.get("applicability_conditions", {}) or {}
    fav_score, unfav_score = 0.0, 0.0
    for c in ac.get("favorable", []):
        derived = c.get("derived_from", {}) or {}
        if _state_matches_condition(state_tuple, derived):
            fav_score += float(c.get("confidence", 0.5))
    for c in ac.get("unfavorable", []):
        derived = c.get("derived_from", {}) or {}
        if _state_matches_condition(state_tuple, derived):
            unfav_score += float(c.get("confidence", 0.5))
    # Squash: sigmoid on net score, so baseline = 0.5
    net = fav_score - unfav_score
    return 1.0 / (1.0 + np.exp(-net))


def build_kernel(strategy: Dict, llm_classifier=None) -> np.ndarray:
    """
    Build a |X| × |A| Markov kernel for a strategy.

    Each row corresponds to a state (canonical order), each column to an action.
    Row sums are 1.
    """
    states = enumerate_state_space()
    step_dist = _step_action_dist(strategy, llm_classifier=llm_classifier)
    uniform = np.ones(cfg.NUM_ACTIONS) / cfg.NUM_ACTIONS
    K = np.zeros((len(states), cfg.NUM_ACTIONS), dtype=np.float64)

    for i, state in enumerate(states):
        alpha = _state_applicability(strategy, state)
        # alpha=1 → all mass on step_dist (strategy is very applicable)
        # alpha=0 → blend 50/50 step_dist + uniform (strategy is inapplicable)
        # alpha=0.5 → mostly step_dist, small uniform noise
        blend_uniform = (1.0 - alpha) * 0.5
        blend_steps = 1.0 - blend_uniform
        row = blend_steps * step_dist + blend_uniform * uniform
        K[i, :] = row / row.sum()

    return K


def load_strategy_file(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))
