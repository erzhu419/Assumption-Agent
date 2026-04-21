"""
Core validation: does the dispatcher + KB stack actually make the LLM better?

Experiment design:
  Condition A (baseline): problem -> LLM -> answer
  Condition B (guided):   problem -> dispatcher (RE-SAC + KB matcher) -> strategy
                          -> LLM (with strategy operational_steps injected) -> answer

Then a BLIND judge LLM scores A vs B on each of 100 test problems.
Random side-swap per problem, so the judge can't tell which is which.

Win rate ≥ 55% = method actually helps LLM solve problems.
Win rate ≈ 50% = whole project thesis is unsupported.

Metrics reported:
  - B_wins / A_wins / ties  (direct)
  - Win rate B over A, 95% CI
  - Breakdown by difficulty and domain
  - Judge reasoning excerpts for qualitative inspection
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

import _config as cfg
from llm_client import create_client, parse_json_from_llm


SYSTEM_SOLVER = """你是一位严谨的问题解决者。针对下面给出的问题，给出一个清晰、结构化的解决方案。

要求：
1. 先简要重述问题核心
2. 给出你的分析和推理步骤
3. 给出最终建议/解答
4. 语言精炼，不超过 400 字

"""

GUIDED_SOLVER = """你是一位严谨的问题解决者。针对下面给出的问题，请**按照给定的方法论框架**来解决。

## 方法论框架：{strategy_name_zh}
**核心思想：** {strategy_description}

**操作步骤：**
{operational_steps}

## 要求
1. 严格按照上述方法论的步骤展开分析（每一步都要显式体现）
2. 先简要重述问题核心
3. 给出最终建议/解答
4. 语言精炼，不超过 400 字

## 问题
"""


JUDGE_PROMPT = """你是方法论评审专家。下面是同一个问题的两个解答，请客观评判。

## 问题
{problem_description}

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
{{
  "winner": "A" 或 "B" 或 "tie",
  "score_a": 整数1-10,
  "score_b": 整数1-10,
  "reasoning": "简短说明胜者为什么更好（不超过 80 字）"
}}
"""


@dataclass
class TrialResult:
    problem_id: str
    domain: str
    difficulty: str
    strategy_selected: str
    baseline_answer: str
    guided_answer: str
    judge_winner: str     # "A"/"B"/"tie" in randomized positions
    baseline_was: str     # "A" or "B" — which side baseline was
    score_baseline: int
    score_guided: int
    judge_reasoning: str


def _format_steps(strategy: Dict) -> str:
    steps = strategy.get("operational_steps", [])
    lines = []
    for s in steps:
        step_n = s.get("step", len(lines) + 1)
        action = s.get("action", "")
        lines.append(f"  {step_n}. {action}")
    return "\n".join(lines) if lines else "(操作步骤未定义)"


def _format_composition(comp: Dict, strategies: Dict) -> Dict:
    """Build a composite strategy dict from a COMP_* definition."""
    seq = comp.get("sequence", [])
    name = comp.get("name", {}).get("zh", comp.get("composition_id", ""))
    steps_combined = []
    for sid in seq:
        if sid in strategies:
            for s in strategies[sid].get("operational_steps", []):
                steps_combined.append({
                    "step": len(steps_combined) + 1,
                    "action": f"[{sid}] {s.get('action', '')}",
                })
    desc = comp.get("transition_condition") or name
    return {
        "id": comp.get("composition_id"),
        "name": {"zh": name},
        "description": {"one_sentence": desc},
        "operational_steps": steps_combined,
    }


def run_solver(client, problem: str, strategy: Optional[Dict]) -> str:
    if strategy is None:
        prompt = SYSTEM_SOLVER + f"## 问题\n{problem}"
    else:
        prompt = (
            GUIDED_SOLVER.format(
                strategy_name_zh=strategy.get("name", {}).get("zh", strategy.get("id", "")),
                strategy_description=strategy.get("description", {}).get("one_sentence", ""),
                operational_steps=_format_steps(strategy),
            )
            + problem
        )
    resp = client.generate(prompt, max_tokens=800, temperature=0.3)
    return resp["text"].strip()


def run_judge(client, problem: str, a: str, b: str) -> Dict:
    prompt = JUDGE_PROMPT.format(problem_description=problem, answer_a=a, answer_b=b)
    resp = client.generate(prompt, max_tokens=256, temperature=0.1)
    try:
        return parse_json_from_llm(resp["text"])
    except Exception:
        return {"winner": "tie", "score_a": 5, "score_b": 5, "reasoning": "parse_failure"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of problems")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="test")
    ap.add_argument("--algo", default="resac")
    ap.add_argument("--output", default="llm_ab_results.json")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Load environment and sample problems
    print("Loading data...")
    strategy_kb = {}
    for f in sorted(cfg.KB_DIR.glob("S*.json")):
        d = json.load(open(f, encoding="utf-8"))
        strategy_kb[d["id"]] = d

    compositions = {}
    for f in sorted(cfg.COMP_DIR.glob("COMP_*.json")):
        d = json.load(open(f, encoding="utf-8"))
        compositions[d["composition_id"]] = d

    from task_env.base_env import TaskEnvironment
    env = TaskEnvironment(strategy_kb=strategy_kb)
    problems = env.get_all_problems(args.split)
    random.shuffle(problems)
    problems = problems[: args.n]
    print(f"  Sampled {len(problems)} problems from split={args.split}")

    # 2. Load dispatcher + matcher + features
    from dispatcher.resac_discrete import RESACDiscreteDispatcher
    from dispatcher.feature_extractor import FeatureExtractor
    from dispatcher.kb_matcher import KBMatcher

    dispatcher = RESACDiscreteDispatcher(
        input_dim=cfg.INPUT_DIM, num_actions=cfg.NUM_ACTIONS, ensemble_size=5
    )
    dispatcher.load(str(PROJECT / "checkpoints" / f"dispatcher_{args.algo}.pt"))
    dispatcher.training = False

    extractor = FeatureExtractor(use_llm=False)
    matcher = KBMatcher(cfg.KB_DIR, cfg.STRATEGY_IDS)

    feat_cache = json.loads((PROJECT / "cache" / "features.json").read_text())

    # 3. Solve + judge
    client = create_client()
    results: List[TrialResult] = []
    t0 = time.time()

    for i, p in enumerate(problems):
        pid = p["problem_id"]
        desc = p.get("description", "")
        domain = p.get("domain", "unknown")
        difficulty = p.get("difficulty", "medium")

        # Select strategy via dispatcher
        feats = extractor.extract(desc, problem_id=pid)
        cached = feat_cache.get(pid, {})
        if cached:
            feats.update(cached)
        kb_scores = matcher.compute_scores(feats, cfg.ACTION_SPACE)
        vec = extractor.features_to_vector(feats, kb_match_scores=kb_scores)
        action = dispatcher.select_action(vec, cfg.ACTION_SPACE)
        sid = action.strategy_id

        # Resolve strategy object
        if sid in strategy_kb:
            strategy = strategy_kb[sid]
        elif sid in compositions:
            strategy = _format_composition(compositions[sid], strategy_kb)
        else:
            # SPECIAL_GATHER_INFO or unknown: fall back to baseline-like prompt
            # but we still tag the selected id for later analysis.
            strategy = None

        # Solve both
        try:
            base = run_solver(client, desc, None)
            if strategy is None:
                # If dispatcher picked SPECIAL_GATHER_INFO, "guided" is the same prompt
                # but we note this. Still send through judge as-is; baseline wins if tied.
                guided = run_solver(client, desc, None)
            else:
                guided = run_solver(client, desc, strategy)
        except Exception as e:
            print(f"  [skip] {pid}: solver error {e}")
            continue

        # Blind judge with randomized side assignment
        if random.random() < 0.5:
            a, b = base, guided
            baseline_was = "A"
        else:
            a, b = guided, base
            baseline_was = "B"

        try:
            verdict = run_judge(client, desc, a, b)
        except Exception as e:
            print(f"  [skip-judge] {pid}: {e}")
            continue

        winner_raw = verdict.get("winner", "tie")
        # Map judge verdict back to baseline/guided
        if winner_raw == "tie":
            real_winner = "tie"
        elif winner_raw == baseline_was:
            real_winner = "baseline"
        else:
            real_winner = "guided"

        score_a = int(verdict.get("score_a", 5))
        score_b = int(verdict.get("score_b", 5))
        score_baseline = score_a if baseline_was == "A" else score_b
        score_guided = score_b if baseline_was == "A" else score_a

        results.append(TrialResult(
            problem_id=pid, domain=domain, difficulty=difficulty,
            strategy_selected=sid,
            baseline_answer=base[:200] + ("..." if len(base) > 200 else ""),
            guided_answer=guided[:200] + ("..." if len(guided) > 200 else ""),
            judge_winner=real_winner, baseline_was=baseline_was,
            score_baseline=score_baseline, score_guided=score_guided,
            judge_reasoning=verdict.get("reasoning", ""),
        ))

        if (i + 1) % 10 == 0:
            n_b = sum(1 for r in results if r.judge_winner == "guided")
            n_a = sum(1 for r in results if r.judge_winner == "baseline")
            n_t = sum(1 for r in results if r.judge_winner == "tie")
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(problems)}] guided={n_b} base={n_a} tie={n_t}  {elapsed:.0f}s")

    # 4. Report
    print(f"\n{'='*60}")
    print(f"LLM A/B VALIDATION — {len(results)} problems")
    print(f"{'='*60}")

    n = len(results)
    n_guided = sum(1 for r in results if r.judge_winner == "guided")
    n_base = sum(1 for r in results if r.judge_winner == "baseline")
    n_tie = sum(1 for r in results if r.judge_winner == "tie")
    # Win rate excluding ties
    decided = n_guided + n_base
    win_rate = n_guided / decided if decided > 0 else 0.5
    # Wilson 95% CI
    if decided > 0:
        z = 1.96
        p = n_guided / decided
        denom = 1 + z**2 / decided
        center = (p + z**2 / (2 * decided)) / denom
        halfw = (z / denom) * (np.sqrt(p * (1 - p) / decided + z**2 / (4 * decided**2)))
        ci_low, ci_high = center - halfw, center + halfw
    else:
        ci_low = ci_high = 0.5

    print(f"\n  guided wins : {n_guided}/{n}  ({n_guided/n:.1%})")
    print(f"  baseline wins: {n_base}/{n}  ({n_base/n:.1%})")
    print(f"  ties         : {n_tie}/{n}  ({n_tie/n:.1%})")
    print(f"  win rate (excl. ties): {win_rate:.1%}")
    print(f"  95% CI: [{ci_low:.1%}, {ci_high:.1%}]")

    # Mean scores
    mean_base = np.mean([r.score_baseline for r in results]) if results else 0
    mean_guided = np.mean([r.score_guided for r in results]) if results else 0
    print(f"  mean judge score: baseline={mean_base:.2f}, guided={mean_guided:.2f}, Δ={mean_guided - mean_base:+.2f}")

    # Breakdown by difficulty
    print("\n  By difficulty:")
    for diff in ["easy", "medium", "hard"]:
        sub = [r for r in results if r.difficulty == diff]
        if not sub:
            continue
        g = sum(1 for r in sub if r.judge_winner == "guided")
        b = sum(1 for r in sub if r.judge_winner == "baseline")
        t = sum(1 for r in sub if r.judge_winner == "tie")
        d = g + b
        wr = g / d if d else 0.5
        print(f"    {diff:>6}: guided={g}, base={b}, tie={t}, win_rate={wr:.1%}  (n={len(sub)})")

    # Breakdown by domain
    print("\n  By domain:")
    by_dom = defaultdict(lambda: [0, 0, 0])  # guided, base, tie
    for r in results:
        if r.judge_winner == "guided":
            by_dom[r.domain][0] += 1
        elif r.judge_winner == "baseline":
            by_dom[r.domain][1] += 1
        else:
            by_dom[r.domain][2] += 1
    for dom, (g, b, t) in sorted(by_dom.items()):
        d = g + b
        wr = g / d if d else 0.5
        print(f"    {dom:<22}: guided={g}, base={b}, tie={t}, win_rate={wr:.1%}  (n={g+b+t})")

    # Save
    out = PROJECT.parent / "phase two" / "analysis" / args.output
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "n_total": n,
        "n_guided_wins": n_guided,
        "n_baseline_wins": n_base,
        "n_ties": n_tie,
        "win_rate_excl_ties": win_rate,
        "wilson_ci_95": [ci_low, ci_high],
        "mean_score_baseline": float(mean_base),
        "mean_score_guided": float(mean_guided),
        "results": [asdict(r) for r in results],
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
