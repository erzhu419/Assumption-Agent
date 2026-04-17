"""
LLM-based condition writer: replace template text with natural-language
conditions grounded in the actual supporting executions.

The writer receives a candidate + a few of its supporting executions and is
prompted to produce a novel, generalizable condition description — one that
would not be trivially derivable from the problem's structural features alone
(to avoid being redundant with what the dispatcher already sees).

Falls back to the template text on any LLM failure — the integrator still works.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

PHASE0_SCRIPTS = Path(__file__).parent.parent.parent / "phase zero" / "scripts"
sys.path.insert(0, str(PHASE0_SCRIPTS))
from llm_client import create_client, parse_json_from_llm  # noqa: E402


PROMPT = """你是一个方法论知识库的资深维护者。当前需要为策略 {strategy_id} 添加一条{placement_zh}适用条件。

## 背景
{strategy_name} — {strategy_one_sentence}

## 结构特征线索
所有支持证据的问题共享的特征：{feature_summary}

## 支持证据（执行记录摘要，最多 {n_show} 条）
{evidence_block}

## 你的任务
写一条"{placement_zh}适用条件"——用一句话描述为何该策略在这些问题上{outcome_verb}。

**关键约束（必读）**
1. 禁止重复结构特征本身（如"耦合度≥0.65"之类的阈值语言）。要描述**现象**或**机制**，而不是特征值。
2. 条件要**可跨领域复用**——不能只针对某个具体问题。
3. 30 字以内，汉语。
4. 如果证据不足以支撑一个具体、非平凡的条件，返回 null。

输出 JSON（不要代码块）：
{{
  "condition": "...",      // 30 字以内的条件描述，或 null
  "reasoning": "...",       // 一句话解释你为什么这样写
  "generalizable": true,    // 该条件是否跨领域适用
  "redundant_with_features": false  // 是否与结构特征重复
}}
"""


PLACEMENT_ZH = {"favorable": "有利", "unfavorable": "不利"}
OUTCOME_VERB = {"favorable": "有效", "unfavorable": "失效"}


class ConditionWriter:
    def __init__(self, kb_dir: Path, executions_dir: Path, max_evidence_shown: int = 3):
        self.kb_dir = Path(kb_dir)
        self.executions_dir = Path(executions_dir)
        self.max_evidence_shown = max_evidence_shown
        self._client = None
        self._strategies: Dict[str, dict] = {}
        self._executions_cache: Dict[str, dict] = {}

    @property
    def client(self):
        if self._client is None:
            self._client = create_client()
        return self._client

    def _load_strategy(self, strategy_id: str) -> dict:
        if strategy_id not in self._strategies:
            for f in self.kb_dir.glob(f"{strategy_id}_*.json"):
                self._strategies[strategy_id] = json.loads(f.read_text(encoding="utf-8"))
                break
        return self._strategies.get(strategy_id, {})

    def _load_execution(self, exec_id: str) -> dict:
        if exec_id in self._executions_cache:
            return self._executions_cache[exec_id]
        p = self.executions_dir / f"{exec_id}.json"
        if not p.exists():
            return {}
        d = json.loads(p.read_text(encoding="utf-8"))
        self._executions_cache[exec_id] = d
        return d

    def _feature_summary(self, evidences: List[dict]) -> str:
        """Describe structural feature profile of the supporting set."""
        if not evidences:
            return "无"
        feats: Dict[str, List[float]] = {}
        domains: List[str] = []
        for e in evidences:
            rec = self._load_execution(e.get("execution_id", ""))
            task = rec.get("task", {})
            comp = task.get("complexity_features", {})
            for k, v in comp.items():
                if isinstance(v, (int, float)):
                    feats.setdefault(k, []).append(float(v))
            d = task.get("domain")
            if d:
                domains.append(d)
        parts = []
        for k, vs in feats.items():
            if vs:
                avg = sum(vs) / len(vs)
                parts.append(f"{k}≈{avg:.2f}")
        dom_str = ",".join(sorted(set(domains))) if domains else "-"
        return f"领域={dom_str}；{'; '.join(parts[:6])}"

    def _evidence_block(self, evidences: List[dict], limit: int) -> str:
        lines = []
        for i, e in enumerate(evidences[:limit]):
            rec = self._load_execution(e.get("execution_id", ""))
            task = rec.get("task", {})
            desc = task.get("description", "")[:120]
            outcome = rec.get("outcome", {})
            outcome_str = "成功" if outcome.get("success") else "失败"
            reason = outcome.get("failure_reason") or "无特定原因"
            lines.append(f"  {i+1}. [{outcome_str}] {desc}…（失败原因：{reason}）")
        return "\n".join(lines) if lines else "  (无可读证据)"

    def rewrite(self, candidate: dict) -> Dict:
        """Return {condition, redundant_with_features, generalizable, reasoning}
        or the template fallback if LLM disapproves/fails."""
        strategy = self._load_strategy(candidate["target_strategy"])
        strategy_name = (strategy.get("name", {}).get("zh")
                        or strategy.get("name", {}).get("en", candidate["target_strategy"]))
        strategy_one = strategy.get("description", {}).get("one_sentence", "")
        placement = candidate["placement"]
        evidences = candidate.get("supporting_evidence", [])

        prompt = PROMPT.format(
            strategy_id=candidate["target_strategy"],
            strategy_name=strategy_name,
            strategy_one_sentence=strategy_one,
            placement_zh=PLACEMENT_ZH[placement],
            feature_summary=self._feature_summary(evidences),
            n_show=min(self.max_evidence_shown, len(evidences)),
            evidence_block=self._evidence_block(evidences, self.max_evidence_shown),
            outcome_verb=OUTCOME_VERB[placement],
        )

        try:
            resp = self.client.generate(prompt, max_tokens=256, temperature=0.2)
            parsed = parse_json_from_llm(resp["text"])
            cond = parsed.get("condition")
            if not cond or parsed.get("redundant_with_features"):
                return {"condition": None, "rejected": True, "raw": parsed}
            if not parsed.get("generalizable", True):
                return {"condition": None, "rejected": True, "raw": parsed}
            return {
                "condition": str(cond).strip()[:50],
                "reasoning": parsed.get("reasoning", ""),
                "rejected": False,
                "raw": parsed,
            }
        except Exception as e:
            return {"condition": None, "rejected": True, "error": str(e)}
