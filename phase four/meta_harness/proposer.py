"""
LLM-based harness proposer.

Reads prior harness source + eval scores + judge reasoning excerpts, then
proposes a new harness (Python code) that aims to beat prior best on the
search set.

The proposer is given structured context:
  - 27 KB strategies (one-sentence each)
  - Summary of 50 trace samples (domain × strategy × success)
  - All prior harnesses' full source + eval results + judge reasoning examples
  - Explicit "write a complete Python file" instruction
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase zero" / "scripts"))
from llm_client import create_client  # noqa: E402

from runtime import load_kb, load_trace_samples
import mh_config as cfg


PROPOSE_PROMPT = """你是一位 LLM harness 工程师。你的任务：编写一个 Python 函数，包装一个 LLM 调用客户端，让它在"问题解决"基准上的 judge 胜率（对抗 baseline）超过所有先前版本。

## Harness 接口约定
你的代码必须：
1. 定义顶层函数 `def solve(problem: str, ctx) -> str`
2. 返回字符串（最终答案）
3. 通过 `ctx.generate(prompt, max_tokens=N, temperature=T)` 调用 LLM（返回字符串）
4. 在单次 solve 内**最多调用 ctx.generate 6 次**（超过会抛异常）
5. 可以使用 `ctx.kb` — 一个 dict，key 是策略 ID（S01-S27），value 是完整策略字典
6. 可以使用 `ctx.trace_samples` — List[Dict]，每项含 task_id/domain/difficulty/selected_strategy/success/features
7. 可以 `ctx.log(msg)` 记录诊断信息

## 可用的 KB 摘要
{kb_summary}

## 可用的 trace 摘要（按 domain 统计各策略成功率）
{trace_summary}

## 先前版本历史（按时间倒序，最多 {max_history} 个）
{history}

## 你的任务
写一个**新的**、与历史版本在策略上**不同**的 harness 实现。考虑：
- Judge 从哪些维度评分（问题理解、分析深度、结构化程度、实用性）
- 先前失败的模式（查看 judge 理由样例）
- 能否用 KB 策略、trace 先验、多步自检、角色设定、结构化输出等改进

**直接输出完整的 Python 代码（不加 markdown 代码块围栏）**，从 docstring 或 `def solve` 开始。
不要加解释文字，只给代码。

## 代码约束
- **控制在 60 行以内**，代码越精炼越好
- **必须完整**，不能有未闭合的 if/else/try 分支 —— 简短优于完备
- 避免过度复杂的分类/检索逻辑 —— 每次 solve 最多 3 次 LLM 调用

## 设计建议（可选参考）
- 先看 problem 的领域/难度，决定是否需要拆分
- 用 KB 策略时，只选 1-2 条最相关的
- 若需多步，第一步可以让 LLM 自己分析 problem 性质，第二步按分析结果选具体做法
- 最终输出必须直接回答问题，不要只给"方法论"
- 警惕"过度结构化" —— baseline 就是直接答，太 meta 反而输
"""


@dataclass
class HistoryEntry:
    version: str
    source_code: str
    win_rate: float
    mean_delta: float
    wins: int
    losses: int
    ties: int
    by_domain: Dict[str, Dict]
    judge_reasoning_examples: List[str]


def build_kb_summary(kb: Dict[str, Dict]) -> str:
    lines = []
    for sid in sorted(kb.keys()):
        s = kb[sid]
        name = s.get("name", {}).get("zh", sid)
        one = s.get("description", {}).get("one_sentence", "")
        lines.append(f"- {sid} {name}: {one}")
    return "\n".join(lines)


def build_trace_summary(trace_samples: List[Dict]) -> str:
    """Aggregate: per (domain, strategy) success count."""
    from collections import defaultdict
    agg = defaultdict(lambda: {"n": 0, "success": 0})
    for t in trace_samples:
        key = (t.get("domain", "?"), t.get("selected_strategy", "?"))
        agg[key]["n"] += 1
        if t.get("success"):
            agg[key]["success"] += 1
    lines = []
    for (dom, sid), v in sorted(agg.items(), key=lambda x: -x[1]["n"])[:20]:
        sr = v["success"] / v["n"] if v["n"] else 0
        lines.append(f"  {dom} × {sid}: n={v['n']}, success_rate={sr:.0%}")
    return "\n".join(lines) if lines else "(no trace samples)"


def build_history(entries: List[HistoryEntry], max_n: int) -> str:
    if not entries:
        return "(no prior harnesses; write a creative first attempt)"
    recent = entries[-max_n:][::-1]
    blocks = []
    for e in recent:
        dom_block = "  ".join(
            f"{d}={v.get('harness', 0)}w/{v.get('baseline', 0)}b"
            for d, v in e.by_domain.items()
        )
        judge_lines = "\n".join(f"      - {r}" for r in e.judge_reasoning_examples[:3])
        block = (f"### {e.version} — win_rate={e.win_rate:.1%} "
                 f"(w/l/t = {e.wins}/{e.losses}/{e.ties}) mean_Δ={e.mean_delta:+.2f}\n"
                 f"    by domain: {dom_block}\n"
                 f"    judge reasoning examples:\n{judge_lines}\n"
                 f"    source code:\n{e.source_code}")
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


class HarnessProposer:
    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = create_client()
        return self._client

    def propose(self, history: List[HistoryEntry], iteration: int) -> str:
        """Returns raw Python source code for a new harness."""
        kb = load_kb()
        traces = load_trace_samples()
        prompt = PROPOSE_PROMPT.format(
            kb_summary=build_kb_summary(kb),
            trace_summary=build_trace_summary(traces),
            history=build_history(history, cfg.MAX_HISTORY_HARNESSES),
            max_history=cfg.MAX_HISTORY_HARNESSES,
        )
        resp = self.client.generate(prompt, max_tokens=cfg.HARNESS_MAX_TOKENS, temperature=0.7)
        code = resp["text"]

        # Strip markdown fences if present
        code = code.strip()
        if code.startswith("```"):
            m = re.match(r"```(?:python)?\n(.*)```\s*$", code, flags=re.DOTALL)
            if m:
                code = m.group(1)
            else:
                # fallback: strip first line and trailing ```
                lines = code.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].rstrip() == "```":
                    lines = lines[:-1]
                code = "\n".join(lines)
        return code
