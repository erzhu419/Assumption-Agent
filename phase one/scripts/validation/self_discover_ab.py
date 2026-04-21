"""
Self-Discover minimal replacement: our 27 strategies vs baseline.

Three-step SELECT/ADAPT/IMPLEMENT flow (Zhou et al. 2024):
  SELECT    : LLM picks 3-5 relevant strategies from the 27-strategy list
              (we provide only one-sentence descriptions, not operational_steps)
  ADAPT     : LLM rephrases each selected strategy to fit the current task
  IMPLEMENT : LLM produces a JSON reasoning plan tailored to the task
  EXECUTE   : LLM solves the task following its own plan

Baseline: single-call direct solve.

Two optional modes for SELECT:
  --filter=none : LLM sees all 27 strategies (pure Self-Discover)
  --filter=mlp  : LLM sees only the top-K strategies ranked by RE-SAC dispatcher

The second mode asks: does our trained dispatcher add value ON TOP of
Self-Discover, by pre-filtering the strategy list?
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
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


SELECT_PROMPT = """以下是多条方法论"推理模块"，每条一句话描述其核心思想。

## 任务
{task}

## 候选推理模块
{modules}

从上面选出 3-5 条**对解决本任务最关键**的模块。按重要性排序。

输出 JSON（不要代码块）：
{{"selected": ["模块编号1", "模块编号2", ...], "reasoning": "简述为什么选这些"}}
"""


ADAPT_PROMPT = """针对下面的具体任务，把选中的推理模块**改写得更贴合该任务**。不是复述，而是改成能直接指导该任务求解的具体说法。

## 任务
{task}

## 选中的推理模块（原始）
{selected_text}

输出 JSON（不要代码块）：
{{"adapted": [
  {{"original_id": "...", "adapted_description": "针对该任务的具体化说法（30-60 字）"}}
]}}
"""


IMPLEMENT_PROMPT = """把已经改写的推理模块，组装成**针对该任务的 step-by-step 推理计划**。

## 任务
{task}

## 改写后的推理模块
{adapted_text}

生成一个 JSON 推理计划：每一步是一个 "step_description" + "expected_output"。步骤按自然执行顺序排列，总共 3-7 步。

输出 JSON（不要代码块）：
{{"plan": [
  {{"step_description": "...", "expected_output": "..."}},
  ...
]}}
"""


EXECUTE_PROMPT = """你将按照下面的推理计划解决问题。依次完成每一步，最终给出解答。

## 任务
{task}

## 推理计划
{plan}

要求：
1. 按计划顺序执行，每步都明示"Step N: ..."
2. 最终给出解答
3. 语言精炼，总共不超过 500 字
"""


BASELINE_PROMPT = """你是一位严谨的问题解决者。针对下面给出的问题，给出一个清晰、结构化的解决方案。

要求：
1. 先简要重述问题核心
2. 给出你的分析和推理步骤
3. 给出最终建议/解答
4. 语言精炼，不超过 400 字

## 问题
{task}
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
{{"winner": "A"或"B"或"tie", "score_a": 整数1-10, "score_b": 整数1-10,
  "reasoning": "简短说明胜者为什么更好（不超过 80 字）"}}
"""


@dataclass
class TrialResult:
    problem_id: str
    domain: str
    difficulty: str
    selected_strategies: List[str]
    judge_winner: str
    baseline_was: str
    score_baseline: int
    score_guided: int
    judge_reasoning: str


def build_module_list(strategy_kb: Dict, strategy_ids: List[str]) -> str:
    """One-sentence-only bullet list, as Self-Discover does."""
    lines = []
    for sid in strategy_ids:
        s = strategy_kb.get(sid)
        if not s:
            continue
        name = s.get("name", {}).get("zh", sid)
        one = s.get("description", {}).get("one_sentence", "")
        lines.append(f"{sid} ({name}): {one}")
    return "\n".join(lines)


def extract_selected_ids(raw: List[str], valid_ids: set) -> List[str]:
    """Parse LLM's selected list, tolerate 'S07' / '07' / 'S07 控制' etc."""
    out = []
    for item in raw:
        s = str(item).strip().upper()
        # look for Sxx
        import re
        m = re.search(r"S\d{2}", s)
        if m and m.group(0) in valid_ids:
            out.append(m.group(0))
    return list(dict.fromkeys(out))[:5]  # dedupe, cap at 5


def self_discover_solve(client, task: str, strategy_kb: Dict,
                        candidate_ids: List[str]) -> Dict:
    """Run SELECT -> ADAPT -> IMPLEMENT -> EXECUTE. Returns dict with 'answer' and metadata."""
    module_list = build_module_list(strategy_kb, candidate_ids)
    valid = set(candidate_ids)

    # SELECT
    sel_resp = client.generate(
        SELECT_PROMPT.format(task=task, modules=module_list),
        max_tokens=300, temperature=0.2
    )
    try:
        sel = parse_json_from_llm(sel_resp["text"])
        selected = extract_selected_ids(sel.get("selected", []), valid)
    except Exception:
        selected = candidate_ids[:3]
    if not selected:
        selected = candidate_ids[:3]

    selected_text = "\n".join(
        f"- {sid} ({strategy_kb[sid]['name']['zh']}): "
        f"{strategy_kb[sid]['description']['one_sentence']}"
        for sid in selected if sid in strategy_kb
    )

    # ADAPT
    ad_resp = client.generate(
        ADAPT_PROMPT.format(task=task, selected_text=selected_text),
        max_tokens=400, temperature=0.2
    )
    try:
        ad = parse_json_from_llm(ad_resp["text"])
        adapted = ad.get("adapted", [])
    except Exception:
        adapted = [{"original_id": sid,
                    "adapted_description": strategy_kb[sid]["description"]["one_sentence"]}
                   for sid in selected if sid in strategy_kb]
    adapted_text = "\n".join(
        f"- [{a.get('original_id', '?')}] {a.get('adapted_description', '')}"
        for a in adapted
    )

    # IMPLEMENT
    imp_resp = client.generate(
        IMPLEMENT_PROMPT.format(task=task, adapted_text=adapted_text),
        max_tokens=500, temperature=0.2
    )
    try:
        imp = parse_json_from_llm(imp_resp["text"])
        plan = imp.get("plan", [])
    except Exception:
        plan = [{"step_description": a.get("adapted_description", ""),
                 "expected_output": ""} for a in adapted]
    plan_text = "\n".join(
        f"  Step {i+1}: {p.get('step_description', '')}\n"
        f"         期望输出: {p.get('expected_output', '')}"
        for i, p in enumerate(plan)
    )

    # EXECUTE
    exe_resp = client.generate(
        EXECUTE_PROMPT.format(task=task, plan=plan_text),
        max_tokens=800, temperature=0.3
    )
    return {
        "answer": exe_resp["text"].strip(),
        "selected": selected,
        "plan_text": plan_text,
    }


def baseline_solve(client, task: str) -> str:
    resp = client.generate(
        BASELINE_PROMPT.format(task=task), max_tokens=800, temperature=0.3
    )
    return resp["text"].strip()


def judge(client, problem: str, a: str, b: str) -> Dict:
    resp = client.generate(
        JUDGE_PROMPT.format(problem_description=problem, answer_a=a, answer_b=b),
        max_tokens=256, temperature=0.1
    )
    try:
        return parse_json_from_llm(resp["text"])
    except Exception:
        return {"winner": "tie", "score_a": 5, "score_b": 5, "reasoning": "parse_failure"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--filter", choices=["none", "mlp"], default="mlp",
                    help="how to pick the candidate pool for SELECT")
    ap.add_argument("--top-k", type=int, default=8,
                    help="when filter=mlp, keep top-K dispatcher picks")
    ap.add_argument("--output", default="self_discover_results.json")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Load KB
    strategy_kb = {}
    for f in sorted(cfg.KB_DIR.glob("S*.json")):
        d = json.load(open(f, encoding="utf-8"))
        strategy_kb[d["id"]] = d
    strategy_ids = sorted(strategy_kb.keys())
    print(f"Loaded {len(strategy_kb)} strategies, filter={args.filter}")

    # Load problems
    from task_env.base_env import TaskEnvironment
    env = TaskEnvironment(strategy_kb=strategy_kb)
    problems = env.get_all_problems("test")
    random.shuffle(problems)
    problems = problems[:args.n]
    print(f"Sampled {len(problems)} problems")

    # Optional: MLP filter
    dispatcher = None
    extractor = None
    matcher = None
    feat_cache = {}
    if args.filter == "mlp":
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
    t0 = time.time()
    results: List[TrialResult] = []

    for i, p in enumerate(problems):
        pid = p["problem_id"]
        desc = p.get("description", "")
        domain = p.get("domain", "unknown")
        difficulty = p.get("difficulty", "medium")

        # Build candidate pool
        if dispatcher is not None:
            feats = extractor.extract(desc, problem_id=pid)
            cached = feat_cache.get(pid, {})
            if cached:
                feats.update(cached)
            kb_scores = matcher.compute_scores(feats, cfg.ACTION_SPACE)
            vec = extractor.features_to_vector(feats, kb_match_scores=kb_scores)
            # Get top-K action indices among S01-S27 only (policy returns (probs, log_probs))
            device = next(dispatcher.policy.parameters()).device
            with torch.no_grad():
                v = torch.as_tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
                probs, _ = dispatcher.policy(v)
                probs = probs.squeeze(0).cpu().numpy()
            top_indices = probs[:27].argsort()[::-1][:args.top_k]
            candidates = [cfg.STRATEGY_IDS[i] for i in top_indices]
        else:
            candidates = strategy_ids

        try:
            sd_result = self_discover_solve(client, desc, strategy_kb, candidates)
            baseline_ans = baseline_solve(client, desc)
        except Exception as e:
            print(f"  [skip] {pid}: {e}")
            continue

        # Blind judge with random side
        if random.random() < 0.5:
            a_text, b_text, baseline_was = baseline_ans, sd_result["answer"], "A"
        else:
            a_text, b_text, baseline_was = sd_result["answer"], baseline_ans, "B"

        try:
            v = judge(client, desc, a_text, b_text)
        except Exception as e:
            print(f"  [skip-judge] {pid}: {e}")
            continue

        winner_raw = v.get("winner", "tie")
        if winner_raw == "tie":
            real = "tie"
        elif winner_raw == baseline_was:
            real = "baseline"
        else:
            real = "guided"

        score_a = int(v.get("score_a", 5))
        score_b = int(v.get("score_b", 5))
        sb = score_a if baseline_was == "A" else score_b
        sg = score_b if baseline_was == "A" else score_a

        results.append(TrialResult(
            problem_id=pid, domain=domain, difficulty=difficulty,
            selected_strategies=sd_result["selected"],
            judge_winner=real, baseline_was=baseline_was,
            score_baseline=sb, score_guided=sg,
            judge_reasoning=v.get("reasoning", ""),
        ))

        if (i + 1) % 10 == 0:
            g = sum(1 for r in results if r.judge_winner == "guided")
            b = sum(1 for r in results if r.judge_winner == "baseline")
            t_ = sum(1 for r in results if r.judge_winner == "tie")
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(problems)}] guided={g} base={b} tie={t_}  {elapsed:.0f}s")

    # Report
    n = len(results)
    g = sum(1 for r in results if r.judge_winner == "guided")
    b = sum(1 for r in results if r.judge_winner == "baseline")
    t = sum(1 for r in results if r.judge_winner == "tie")
    decided = g + b
    wr = g / decided if decided else 0.5
    print(f"\n{'='*60}\n  SELF-DISCOVER A/B ({n} problems, filter={args.filter})\n{'='*60}")
    print(f"  guided wins: {g}  baseline wins: {b}  ties: {t}")
    print(f"  win rate: {wr:.1%}")
    print(f"  mean Δ (guided - baseline): "
          f"{np.mean([r.score_guided - r.score_baseline for r in results]):+.2f}")

    # By domain
    print("\n  By domain:")
    by_dom = defaultdict(lambda: [0, 0, 0])
    for r in results:
        idx = 0 if r.judge_winner == "guided" else 1 if r.judge_winner == "baseline" else 2
        by_dom[r.domain][idx] += 1
    for dom, (g_, b_, t_) in sorted(by_dom.items()):
        d = g_ + b_
        wr_ = g_ / d if d else 0.5
        print(f"    {dom:<22}: g={g_:>2} b={b_:>2} t={t_} wr={wr_:.1%}  (n={g_+b_+t_})")

    # Save
    out = PROJECT.parent / "phase two" / "analysis" / args.output
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "filter": args.filter,
        "top_k": args.top_k if args.filter == "mlp" else None,
        "n": n, "guided": g, "baseline": b, "ties": t, "win_rate": wr,
        "results": [asdict(r) for r in results],
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
