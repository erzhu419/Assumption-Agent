"""
Ablation over guided-prompt design.

Runs THREE variants against the same baseline LLM, on the same 100 problems:
  v1 (hard):  strict step-by-step injection (original)  — already tested, 43.4%
  v2 (soft):  same 7 steps but framed as "reference, adapt freely"
  v3 (hint):  name + one_sentence description only, no steps

Each trial uses a separate judge call. Random side-swap per trial.
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

# v2: soft framing, keeps steps but as reference
GUIDED_SOFT = """你是一位严谨的问题解决者。针对下面给出的问题，给出一个清晰、结构化的解决方案。

## 方法论参考（可选参考，不必严格遵守）
某些类似问题上，下面这套方法论可能有帮助：

**{strategy_name_zh}：** {strategy_description}

**参考步骤：**
{operational_steps}

你可以采纳其中有用的部分，也可以忽略整个方法论，按你自己的判断展开分析。重要的是**解决问题**，而不是套用方法论。

## 要求
1. 先简要重述问题核心
2. 给出你的分析和推理步骤
3. 给出最终建议/解答
4. 语言精炼，不超过 400 字

## 问题
"""

# v3: just a one-sentence hint, no operational_steps
GUIDED_HINT = """你是一位严谨的问题解决者。针对下面给出的问题，给出一个清晰、结构化的解决方案。

**一个思考提示（可参考）：** 类似问题常用「{strategy_name_zh}」的思路 —— {strategy_description}

要求：
1. 先简要重述问题核心
2. 给出你的分析和推理步骤
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
    variant: str
    judge_winner: str    # "guided" | "baseline" | "tie"
    baseline_was: str
    score_baseline: int
    score_guided: int
    judge_reasoning: str


def _format_steps(strategy: Dict) -> str:
    steps = strategy.get("operational_steps", [])
    lines = []
    for s in steps:
        lines.append(f"  {s.get('step', len(lines) + 1)}. {s.get('action', '')}")
    return "\n".join(lines) if lines else "(未定义)"


def _resolve_strategy(sid: str, strategy_kb: Dict, compositions: Dict) -> Optional[Dict]:
    if sid in strategy_kb:
        return strategy_kb[sid]
    if sid in compositions:
        comp = compositions[sid]
        seq = comp.get("sequence", [])
        name = comp.get("name", {}).get("zh", sid)
        steps = []
        for s in seq:
            if s in strategy_kb:
                for step in strategy_kb[s].get("operational_steps", []):
                    steps.append({
                        "step": len(steps) + 1,
                        "action": f"[{s}] {step.get('action', '')}",
                    })
        return {
            "id": sid,
            "name": {"zh": name},
            "description": {"one_sentence": comp.get("transition_condition", name)},
            "operational_steps": steps,
        }
    return None


def build_prompt(variant: str, problem: str, strategy: Optional[Dict]) -> str:
    if strategy is None:
        return SYSTEM_SOLVER + f"## 问题\n{problem}"
    name = strategy.get("name", {}).get("zh", strategy.get("id", ""))
    desc = strategy.get("description", {}).get("one_sentence", "")
    if variant == "soft":
        return GUIDED_SOFT.format(
            strategy_name_zh=name, strategy_description=desc,
            operational_steps=_format_steps(strategy),
        ) + problem
    if variant == "hint":
        return GUIDED_HINT.format(
            strategy_name_zh=name, strategy_description=desc,
        ) + problem
    raise ValueError(variant)


def run_llm(client, prompt: str) -> str:
    return client.generate(prompt, max_tokens=800, temperature=0.3)["text"].strip()


def run_judge(client, problem, a, b) -> Dict:
    p = JUDGE_PROMPT.format(problem_description=problem, answer_a=a, answer_b=b)
    try:
        return parse_json_from_llm(client.generate(p, max_tokens=256, temperature=0.1)["text"])
    except Exception:
        return {"winner": "tie", "score_a": 5, "score_b": 5, "reasoning": "parse_failure"}


def run_variant(variant, client, problems, strategy_kb, compositions,
                dispatcher, extractor, matcher, feat_cache, action_space) -> List[TrialResult]:
    results = []
    t0 = time.time()
    for i, p in enumerate(problems):
        pid = p["problem_id"]
        desc = p.get("description", "")
        domain = p.get("domain", "unknown")
        difficulty = p.get("difficulty", "medium")

        feats = extractor.extract(desc, problem_id=pid)
        cached = feat_cache.get(pid, {})
        if cached:
            feats.update(cached)
        kb_scores = matcher.compute_scores(feats, action_space)
        vec = extractor.features_to_vector(feats, kb_match_scores=kb_scores)
        action = dispatcher.select_action(vec, action_space)
        strategy = _resolve_strategy(action.strategy_id, strategy_kb, compositions)

        try:
            base = run_llm(client, SYSTEM_SOLVER + f"## 问题\n{desc}")
            if strategy is None:
                guided = run_llm(client, SYSTEM_SOLVER + f"## 问题\n{desc}")
            else:
                guided = run_llm(client, build_prompt(variant, desc, strategy))
        except Exception as e:
            print(f"  [skip {variant}] {pid}: {e}")
            continue

        if random.random() < 0.5:
            a, b, baseline_was = base, guided, "A"
        else:
            a, b, baseline_was = guided, base, "B"

        verdict = run_judge(client, desc, a, b)
        winner_raw = verdict.get("winner", "tie")
        if winner_raw == "tie":
            real = "tie"
        elif winner_raw == baseline_was:
            real = "baseline"
        else:
            real = "guided"

        score_a = int(verdict.get("score_a", 5))
        score_b = int(verdict.get("score_b", 5))
        sb = score_a if baseline_was == "A" else score_b
        sg = score_b if baseline_was == "A" else score_a

        results.append(TrialResult(
            problem_id=pid, domain=domain, difficulty=difficulty,
            strategy_selected=action.strategy_id, variant=variant,
            judge_winner=real, baseline_was=baseline_was,
            score_baseline=sb, score_guided=sg,
            judge_reasoning=verdict.get("reasoning", ""),
        ))

        if (i + 1) % 10 == 0:
            g = sum(1 for r in results if r.judge_winner == "guided")
            b_ = sum(1 for r in results if r.judge_winner == "baseline")
            t = sum(1 for r in results if r.judge_winner == "tie")
            print(f"  [{variant} {i+1}/{len(problems)}] g={g} b={b_} t={t} {time.time() - t0:.0f}s")
    return results


def report(variant: str, results: List[TrialResult]):
    if not results:
        return
    n = len(results)
    g = sum(1 for r in results if r.judge_winner == "guided")
    b = sum(1 for r in results if r.judge_winner == "baseline")
    t = sum(1 for r in results if r.judge_winner == "tie")
    decided = g + b
    wr = g / decided if decided else 0.5
    mg = np.mean([r.score_guided for r in results])
    mb = np.mean([r.score_baseline for r in results])
    print(f"\n  [{variant}] guided={g}, base={b}, tie={t}, win_rate={wr:.1%}, mean Δ={mg - mb:+.2f}")
    # By domain
    by_dom = defaultdict(lambda: [0, 0, 0])
    for r in results:
        idx = 0 if r.judge_winner == "guided" else 1 if r.judge_winner == "baseline" else 2
        by_dom[r.domain][idx] += 1
    for dom, (g_, b_, t_) in sorted(by_dom.items()):
        d = g_ + b_
        wr_ = g_ / d if d else 0.5
        print(f"    {dom:<22}: g={g_:>2} b={b_:>2} t={t_} wr={wr_:.1%} (n={g_+b_+t_})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", nargs="+", default=["soft", "hint"])
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    strategy_kb = {json.load(open(f, encoding="utf-8"))["id"]: json.load(open(f, encoding="utf-8"))
                   for f in sorted(cfg.KB_DIR.glob("S*.json"))}
    compositions = {json.load(open(f, encoding="utf-8"))["composition_id"]: json.load(open(f, encoding="utf-8"))
                    for f in sorted(cfg.COMP_DIR.glob("COMP_*.json"))}

    from task_env.base_env import TaskEnvironment
    env = TaskEnvironment(strategy_kb=strategy_kb)
    problems = env.get_all_problems("test")
    random.shuffle(problems)
    problems = problems[: args.n]
    print(f"Sampled {len(problems)} problems")

    from dispatcher.resac_discrete import RESACDiscreteDispatcher
    from dispatcher.feature_extractor import FeatureExtractor
    from dispatcher.kb_matcher import KBMatcher
    dispatcher = RESACDiscreteDispatcher(
        input_dim=cfg.INPUT_DIM, num_actions=cfg.NUM_ACTIONS, ensemble_size=5
    )
    dispatcher.load(str(PROJECT / "checkpoints" / "dispatcher_resac.pt"))
    dispatcher.training = False
    extractor = FeatureExtractor(use_llm=False)
    matcher = KBMatcher(cfg.KB_DIR, cfg.STRATEGY_IDS)
    feat_cache = json.loads((PROJECT / "cache" / "features.json").read_text())

    client = create_client()
    all_results = {}
    for variant in args.variants:
        print(f"\n{'='*60}\n  Running variant: {variant}\n{'='*60}")
        # Reset RNG so same problems get same random side-swap across variants
        random.seed(args.seed + hash(variant) % 1000)
        res = run_variant(variant, client, problems, strategy_kb, compositions,
                          dispatcher, extractor, matcher, feat_cache, cfg.ACTION_SPACE)
        all_results[variant] = res
        report(variant, res)

    out = PROJECT.parent / "phase two" / "analysis" / "ab_ablations.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {v: [asdict(r) for r in rs] for v, rs in all_results.items()},
        indent=2, ensure_ascii=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
