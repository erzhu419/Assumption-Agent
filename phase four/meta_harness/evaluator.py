"""
Evaluator: run harness + baseline on N problems, judge A/B, return win rate.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase zero" / "scripts"))
from llm_client import parse_json_from_llm  # noqa: E402

from runtime import HarnessContext, run_harness, make_context
import mh_config as cfg


JUDGE_PROMPT = """你是方法论评审专家。下面是同一个问题的两个解答，请客观评判。

## 问题
{problem}

## 解答 A
{answer_a}

## 解答 B
{answer_b}

## 评审任务
从以下四个维度综合打分并给出胜者：
1. **问题理解**：是否准确抓住了问题的核心关键
2. **分析深度**：推理链条是否严谨、有逻辑结构
3. **结构化程度**：步骤是否清晰、可追溯、可审查
4. **实用性**：最终建议是否可操作、切中要害

输出 JSON（不要代码块）：
{{"winner": "A"或"B"或"tie", "score_a": 整数1-10, "score_b": 整数1-10,
  "reasoning": "简短说明胜者为什么更好（不超过 80 字）"}}
"""


@dataclass
class TrialResult:
    problem_id: str
    domain: str
    difficulty: str
    judge_winner: str          # "harness" | "baseline" | "tie"
    score_harness: int
    score_baseline: int
    harness_was: str           # "A" or "B"
    harness_calls: int
    judge_reasoning: str
    harness_error: Optional[str] = None


@dataclass
class EvalResult:
    harness_path: str
    n: int
    wins: int                  # harness wins
    losses: int
    ties: int
    win_rate: float            # wins / (wins + losses)
    mean_delta: float
    mean_harness_calls: float
    errors: int
    by_domain: Dict[str, Dict] = field(default_factory=dict)
    trials: List[TrialResult] = field(default_factory=list)


def judge(ctx: HarnessContext, problem: str, a: str, b: str) -> Dict:
    prompt = JUDGE_PROMPT.format(problem=problem, answer_a=a, answer_b=b)
    # Use ctx._client directly to bypass call counting for judging
    resp = ctx._client.generate(prompt, max_tokens=cfg.JUDGE_MAX_TOKENS, temperature=0.1)
    try:
        return parse_json_from_llm(resp["text"])
    except Exception:
        return {"winner": "tie", "score_a": 5, "score_b": 5, "reasoning": "parse_failure"}


def evaluate(harness_path: Path, problems: List[Dict], ctx: HarnessContext,
             baseline_path: Path, seed: int = 42) -> EvalResult:
    random.seed(seed)
    trials: List[TrialResult] = []
    errors = 0

    for p in problems:
        desc = p.get("description", "")
        pid = p["problem_id"]
        domain = p.get("domain", "unknown")
        difficulty = p.get("difficulty", "medium")

        # Fresh ctx per solve (resets call counter)
        ctx.max_calls = 10
        hr = run_harness(harness_path, desc, ctx)
        # Baseline also uses the runtime (so call limits apply to it too)
        br = run_harness(baseline_path, desc, ctx)

        if hr.get("error"):
            errors += 1
            # Still record as loss for this trial
            harness_ans = hr["answer"]
        else:
            harness_ans = hr["answer"]
        baseline_ans = br["answer"]

        # Random side-swap
        if random.random() < 0.5:
            a_text, b_text, harness_was = harness_ans, baseline_ans, "A"
        else:
            a_text, b_text, harness_was = baseline_ans, harness_ans, "B"

        v = judge(ctx, desc, a_text, b_text)
        winner_raw = v.get("winner", "tie")
        if winner_raw == "tie":
            winner = "tie"
        elif winner_raw == harness_was:
            winner = "harness"
        else:
            winner = "baseline"

        score_a = int(v.get("score_a", 5))
        score_b = int(v.get("score_b", 5))
        s_h = score_a if harness_was == "A" else score_b
        s_b = score_b if harness_was == "A" else score_a

        trials.append(TrialResult(
            problem_id=pid, domain=domain, difficulty=difficulty,
            judge_winner=winner,
            score_harness=s_h, score_baseline=s_b,
            harness_was=harness_was,
            harness_calls=hr.get("llm_calls", 0),
            judge_reasoning=v.get("reasoning", ""),
            harness_error=hr.get("error"),
        ))

    wins = sum(1 for t in trials if t.judge_winner == "harness")
    losses = sum(1 for t in trials if t.judge_winner == "baseline")
    ties = sum(1 for t in trials if t.judge_winner == "tie")
    decided = wins + losses
    wr = wins / decided if decided else 0.5
    mean_delta = (sum(t.score_harness - t.score_baseline for t in trials) / max(len(trials), 1))
    mean_calls = sum(t.harness_calls for t in trials) / max(len(trials), 1)

    # By domain
    by_dom: Dict[str, Dict[str, int]] = {}
    for t in trials:
        d = by_dom.setdefault(t.domain, {"harness": 0, "baseline": 0, "tie": 0})
        d[t.judge_winner] += 1

    return EvalResult(
        harness_path=str(harness_path.name),
        n=len(trials), wins=wins, losses=losses, ties=ties,
        win_rate=round(wr, 3), mean_delta=round(mean_delta, 2),
        mean_harness_calls=round(mean_calls, 2), errors=errors,
        by_domain=by_dom, trials=trials,
    )


def log_eval(result: EvalResult, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "harness": result.harness_path,
            "n": result.n,
            "wins": result.wins, "losses": result.losses, "ties": result.ties,
            "win_rate": result.win_rate,
            "mean_delta": result.mean_delta,
            "mean_calls": result.mean_harness_calls,
            "errors": result.errors,
            "by_domain": result.by_domain,
        }, ensure_ascii=False) + "\n")
