"""
LLM-based action classifier.

For each operational step, asks an LLM to output a probability distribution
over the 16-dim ACTION_SPACE. Cached to disk so subsequent runs are free.

Fallback to keyword rules if LLM fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import _config as cfg
from formalization.core.kernel_builder import classify_step_text

PHASE0_SCRIPTS = Path(__file__).parent.parent.parent.parent / "phase zero" / "scripts"
sys.path.insert(0, str(PHASE0_SCRIPTS))
from llm_client import create_client, parse_json_from_llm  # noqa: E402


PROMPT = """你是方法论建模专家。将下列"操作步骤"映射为 16 维抽象行动空间上的概率分布。

## 16 维抽象行动空间
1. decompose          — 分解问题为子问题
2. isolate_variable   — 隔离单一变量（固定其他）
3. simplify           — 简化/降维
4. analogize          — 类比已知问题
5. negate             — 反向推理/反证
6. test_boundary      — 测试边界/极值条件
7. enumerate          — 枚举/穷举
8. estimate_update    — 估计后更新（贝叶斯式）
9. build_incrementally— 增量构建/迭代累积
10. compare_cases     — 对比案例（求同/求异）
11. abstract          — 抽象化/泛化/推广
12. relax_constraint  — 松弛/放宽约束
13. seek_falsification— 寻找反例/证伪
14. evaluate_select   — 评估并选择（满意化/最优化）
15. switch_perspective— 切换视角/重新表述
16. no_action         — 以上均不适用

## 当前步骤
策略：{strategy_name}
步骤描述：{step_text}
困难时回退：{on_difficulty}

## 任务
- 给出这 16 个行动的概率分布（和为 1）
- 同一步骤可能涉及多个行动（例如"分解 + 简化"）—— 允许多峰分布
- 如完全对不上任何，把质量放在 no_action

输出 JSON（键是 1-16 的序号）：
{{"1": 0.0, "2": 0.0, ..., "16": 0.0}}
"""


class LLMActionClassifier:
    def __init__(self, cache_path: Optional[Path] = None):
        self.cache_path = cache_path
        self._client = None
        self._cache: Dict[str, List[float]] = {}
        if cache_path and cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    @property
    def client(self):
        if self._client is None:
            self._client = create_client()
        return self._client

    def _save_cache(self):
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=2))

    def _fallback_dist(self, step_text: str, on_diff: str) -> np.ndarray:
        dist = np.zeros(cfg.NUM_ACTIONS) + 0.02
        primary = classify_step_text(step_text)
        dist[cfg.ACTION_TO_IDX[primary]] += 0.5
        if on_diff:
            secondary = classify_step_text(on_diff)
            if secondary != primary:
                dist[cfg.ACTION_TO_IDX[secondary]] += 0.3
        return dist / dist.sum()

    def classify(self, strategy_name: str, step_text: str, on_diff: str = "") -> np.ndarray:
        cache_key = f"{strategy_name}||{step_text[:100]}||{on_diff[:60]}"
        if cache_key in self._cache:
            arr = np.asarray(self._cache[cache_key], dtype=np.float64)
            if len(arr) == cfg.NUM_ACTIONS:
                return arr / arr.sum()

        prompt = PROMPT.format(strategy_name=strategy_name,
                               step_text=step_text[:500],
                               on_difficulty=(on_diff or "（无）")[:300])
        try:
            resp = self.client.generate(prompt, max_tokens=256, temperature=0.1)
            parsed = parse_json_from_llm(resp["text"])
            arr = np.zeros(cfg.NUM_ACTIONS, dtype=np.float64)
            for k, v in parsed.items():
                try:
                    idx = int(k) - 1
                    if 0 <= idx < cfg.NUM_ACTIONS:
                        arr[idx] = max(0.0, float(v))
                except Exception:
                    continue
            if arr.sum() < 1e-6:
                raise ValueError("empty distribution")
            arr = arr / arr.sum()
            self._cache[cache_key] = arr.tolist()
            self._save_cache()
            return arr
        except Exception:
            return self._fallback_dist(step_text, on_diff)

    def strategy_step_distribution(self, strategy: Dict) -> np.ndarray:
        """Aggregate distribution across all steps of a strategy (smoothed)."""
        strategy_name = strategy.get("name", {}).get("zh", strategy.get("id", ""))
        total = np.zeros(cfg.NUM_ACTIONS) + 0.05  # smoothing
        for step in strategy.get("operational_steps", []):
            text = step.get("action", "") or ""
            on_diff = step.get("on_difficulty") or ""
            dist = self.classify(strategy_name, text, on_diff)
            total += dist
            if on_diff:
                total += 0.3 * self.classify(strategy_name, on_diff)
        return total / total.sum()
