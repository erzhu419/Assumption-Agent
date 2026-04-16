"""
LLM-based World Model Simulator (Route B from Phase 0.5 dev doc).
Uses LLM prompting to predict strategy execution outcomes.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

# Import shared LLM client from Phase 0
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase zero" / "scripts"))
from llm_client import create_client, parse_json_from_llm

from .statistical_model import WorldModelPrediction
from .feature_discretizer import DiscreteState


SIMULATION_PROMPT = """你是一个经验丰富的问题解决专家。

## 任务
请**不要**解决下面的问题。而是**预测**如果按照指定策略来解决，会发生什么。

## 问题特征
- 组件间耦合度: {coupling}
- 可分解性: {decomposability}
- 是否有基准: {has_baseline}
- 信息完整度: {info_completeness}
- 组件数量: {component_count}

## 选定策略
{strategy_name}: {strategy_description}

## 之前的尝试（如有）
{previous_attempts}

## 预测要求
1. 按这个策略走，大概会在哪一步遇到困难？
2. 最终成功解决问题的概率有多大？(0.0-1.0)
3. 最可能的失败原因是什么？
4. 大概需要多少步才能完成？

输出 JSON（不要代码块标记）：
{{"predicted_success_probability": 0.0到1.0, "predicted_bottleneck_step": "...", "predicted_failure_modes": ["原因1", "原因2"], "predicted_steps": 整数}}"""


class LLMSimulator:
    """
    LLM-based strategy outcome predictor.
    More expensive than statistical model but can handle unseen combinations.
    """

    def __init__(self, strategy_kb: Dict = None):
        """
        Args:
            strategy_kb: Dict mapping strategy_id to strategy info dict.
                         If None, predictions will use generic descriptions.
        """
        self._client = None  # Lazy init
        self.strategy_kb = strategy_kb or {}
        self.correction_factor: float = 0.0  # Optimism bias correction

    @property
    def client(self):
        if self._client is None:
            self._client = create_client()
        return self._client

    def predict(self, state: DiscreteState, strategy_id: str,
                strategy_name: str = "", strategy_description: str = "",
                previous_attempts: str = "") -> WorldModelPrediction:
        """Predict strategy outcome using LLM simulation."""

        # Get strategy info from KB if available
        if strategy_id in self.strategy_kb:
            info = self.strategy_kb[strategy_id]
            strategy_name = strategy_name or info.get("name", {}).get("zh", strategy_id)
            strategy_description = strategy_description or info.get("description", {}).get("one_sentence", "")

        prompt = SIMULATION_PROMPT.format(
            coupling=state.coupling,
            decomposability=state.decomposability,
            has_baseline=state.has_baseline,
            info_completeness=state.info_completeness,
            component_count=state.component_count,
            strategy_name=strategy_name or strategy_id,
            strategy_description=strategy_description or "未知策略",
            previous_attempts=previous_attempts or "无",
        )

        try:
            response = self.client.generate(prompt, max_tokens=512, temperature=0.3)
            result = parse_json_from_llm(response["text"])

            prob = float(result.get("predicted_success_probability", 0.5))
            # Apply optimism bias correction (from calibrator)
            prob = max(0.0, min(1.0, prob + self.correction_factor))

            return WorldModelPrediction(
                predicted_success_probability=prob,
                prediction_confidence=0.4,  # LLM predictions are moderate confidence
                predicted_failure_modes=result.get("predicted_failure_modes", []),
                predicted_steps_to_complete=int(result.get("predicted_steps", 10)),
                source="llm",
            )
        except Exception as e:
            # Fallback on LLM error
            return WorldModelPrediction(
                predicted_success_probability=0.5,
                prediction_confidence=0.1,
                predicted_failure_modes=[f"LLM simulation error: {e}"],
                source="llm_error",
            )
