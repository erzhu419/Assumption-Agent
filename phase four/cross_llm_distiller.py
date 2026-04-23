"""Cross-LLM distillation (脑洞 G).

When v20 (gemini-3-flash + our scaffolding) loses a problem by a wide
margin, a stronger generator often "saw something" that 3-flash missed.
We capture that:

  1. Pick N problems where our best variant LOST the judge by gap ≥ GAP
  2. Re-solve each with GPT-5.4 (stronger cross-model)
  3. Judge GPT-5.4's new answer vs our original losing answer
  4. Keep problems where GPT-5.4 beats our answer by gap ≥ GAP
  5. Single distillation pass: GPT-5.4 reads all surviving pairs and
     names the RECURRING orientation its solutions use that our
     scaffolding never foregrounded
  6. Novelty-check the distilled wisdom against current library
  7. Emit candidate(s) to cross_llm_candidates.json for later A/B

Pairs naturally with validate_parallel.py — any candidate produced here
is just another row to include next run.

Usage:
  python cross_llm_distiller.py --judgments phase2_v20_vs_phase2_v16.json
  python cross_llm_distiller.py --judgments xxx.json --top-k 6 --gap 2
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase one" / "scripts" / "validation"))
sys.path.insert(0, str(PROJECT / "phase four"))

from llm_client import create_client, parse_json_from_llm
from gpt5_client import GPT5Client
from cached_framework import judge_pair, _save_content_cache
from wisdom_registry import load_or_init_registry


CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_CANDIDATES = AUTO_DIR / "cross_llm_candidates.json"
OUT_LOG = AUTO_DIR / "cross_llm_log.json"


SOLVE_PROMPT = """你是一位资深的跨域研究者。以下是一道难题，请用最严谨的方法论给出 300-500 字的结构化解答。

## 问题
{problem}

## 输出
直接给答卷正文，不要元注释。"""


DISTILL_PROMPT = """你是 wisdom library 设计师。下面是 **{n} 组**同一现象的证据：对每道题，我们有

  A: gemini-3-flash + 当前 wisdom 脚手架的解答（输了）
  B: GPT-5.4 的独立解答（赢了）

A 用的是 v20 框架（75+2 条 wisdom，含 v19a/v19b 的 reframe 层）。B 没看到任何脚手架，是纯靠模型自身能力。

判官一致判 B 胜的共同原因 → 暗示 **一个现有 library 没明确激活的 orientation**。

## 当前 library（确认不是重复）
{wisdom_brief}

## 证据（题目 + A 输 + B 赢 + 判词）
{evidence}

## 你的任务
判断这 {n} 组证据是否共享 **一个 coherent 的 orientation**，而这 orientation 在现有 library 中没有被 articulate。若是：

1. 起名 + aphorism (≤35 中文字符)
2. source (真实作者+作品，或 "民间谚语"，**不得编造**)
3. signal (15-30 字，什么情境激活)
4. unpacked_for_llm (60-120 字 scenario+self-question)
5. 两个 cross_domain_examples

## 输出 JSON（不要代码块）
若不 coherent 或和现有重复：`{{"proposal": null, "reason": "..."}}`

若 coherent 且 novel：
{{"proposal": {{
  "aphorism": "≤35 字",
  "source": "作者+作品 / 民间谚语",
  "signal": "15-30 字",
  "unpacked_for_llm": "60-120 字",
  "cross_domain_examples": [
    {{"domain": "...", "scenario": "30-50 字"}},
    {{"domain": "不同域", "scenario": "..."}}
  ],
  "rationale": "为什么这条 orientation 是 3-flash 脚手架遗漏的 (60-100 字)",
  "evidence_pids": [{pid_list}]
}}}}
"""


def cache_load(p, default=None):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return default if default is not None else {}
    return default if default is not None else {}


def cache_save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def load_judgment_file(name):
    """name is relative to phase two/analysis/cache/judgments/"""
    p = CACHE / "judgments" / name
    if not p.exists():
        raise FileNotFoundError(f"judgment file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def parse_judgment_file(name):
    """Return {pid: {winner_variant, score_us, score_them, gap, reasoning, ...}}.

    The judgment file has a_was/winner as either 'A'/'B' or the variant name.
    Our variant is the 'A' side (first token in the filename before '_vs_').
    """
    # filename format: {ours}_vs_{them}.json
    stem = Path(name).stem
    if "_vs_" not in stem:
        raise ValueError(f"can't parse variant names from {name}")
    ours, theirs = stem.split("_vs_", 1)
    # Trim trailing descriptor like _holdout50
    ours = ours.replace("_holdout50", "")
    theirs = theirs.replace("_holdout50", "")

    data = load_judgment_file(name)
    parsed = {}
    for pid, v in data.items():
        winner = v.get("winner", "tie")
        sa = v.get("score_a", 0)
        sb = v.get("score_b", 0)
        a_was = v.get("a_was", "A")  # which side "ours" was on when judged
        if a_was == "A":
            score_us, score_them = sa, sb
        else:
            score_us, score_them = sb, sa
        # Normalize winner to 'ours' | 'theirs' | 'tie'
        if winner == "tie":
            who_won = "tie"
        elif (winner == "A" and a_was == "A") or (winner == "B" and a_was == "B") \
             or (winner == ours):
            who_won = "ours"
        else:
            who_won = "theirs"
        parsed[pid] = {
            "who_won": who_won,
            "score_us": score_us,
            "score_them": score_them,
            "gap": score_them - score_us,  # positive = we lost by this much
            "reasoning": v.get("reasoning", ""),
            "domain": v.get("domain", "?"),
            "difficulty": v.get("difficulty", "?"),
            "ours": ours,
            "theirs": theirs,
        }
    return parsed, ours, theirs


def pick_hard_losses(parsed, top_k, gap):
    losses = [(pid, d) for pid, d in parsed.items()
              if d["who_won"] == "theirs" and d["gap"] >= gap]
    losses.sort(key=lambda x: -x[1]["gap"])
    return losses[:top_k]


def load_problem_pool():
    """Return {pid: problem_dict} across all known samples."""
    pool = {}
    for sample_name in ("sample_100.json", "sample_holdout_50.json"):
        p = CACHE / sample_name
        if not p.exists():
            continue
        for prob in json.loads(p.read_text(encoding="utf-8")):
            if "problem_id" in prob:
                pool[prob["problem_id"]] = prob
    return pool


def load_answers(variant):
    p = CACHE / "answers" / f"{variant}_answers.json"
    return cache_load(p)


def solve_with_gpt5(problem_text, client, pid):
    prompt = SOLVE_PROMPT.format(problem=problem_text)
    try:
        resp = client.generate(prompt, max_tokens=1200, temperature=0.3)
        return resp["text"].strip()
    except Exception as e:
        print(f"    [solve err {pid}] {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", required=True,
                    help="e.g. phase2_v20_vs_phase2_v16.json")
    ap.add_argument("--top-k", type=int, default=6,
                    help="how many hard-loss problems to re-solve")
    ap.add_argument("--gap", type=int, default=1,
                    help="minimum score gap; judges typically separate by 1. "
                         "Use 2+ for strictly-severe but most runs will have 0 such.")
    ap.add_argument("--novelty-max-sim", type=float, default=0.78)
    args = ap.parse_args()

    print(f"=== cross_llm_distiller ===")
    print(f"Judgment file: {args.judgments}")

    # 1. Parse judgments
    parsed, ours, theirs = parse_judgment_file(args.judgments)
    print(f"Variants: ours={ours}, theirs={theirs}; total pids={len(parsed)}")

    # 2. Pick hard losses
    losses = pick_hard_losses(parsed, args.top_k, args.gap)
    print(f"Hard losses (gap >= {args.gap}, top {args.top_k}): {len(losses)}")
    for pid, d in losses:
        print(f"  [{pid}] gap={d['gap']} ({d['score_us']} vs {d['score_them']}) "
              f"[{d['domain']}/{d['difficulty']}]")

    if not losses:
        print("No severe losses — nothing to distill.")
        return

    # 3. Load problem text + our answer
    pool = load_problem_pool()
    our_ans = load_answers(ours)
    print(f"\nLoaded our answers: {len(our_ans)} problems")

    # 4. Re-solve each with GPT-5.4
    gpt = GPT5Client()
    pairs = []  # (pid, problem, ours_ans, gpt5_ans, loss_reasoning)
    print(f"\n[solve] re-solving {len(losses)} problems with GPT-5.4...")
    for pid, d in losses:
        if pid not in pool:
            print(f"  [skip {pid}] not in problem pool")
            continue
        prob = pool[pid]["description"]
        ans_a = our_ans.get(pid)
        if not ans_a:
            print(f"  [skip {pid}] no 'ours' answer cached")
            continue
        ans_b = solve_with_gpt5(prob, gpt, pid)
        if not ans_b:
            continue
        pairs.append((pid, prob, ans_a, ans_b, d["reasoning"], d))
        print(f"  [{pid}] GPT-5.4 solved ({len(ans_b)} chars)")

    if not pairs:
        print("No pairs to distill.")
        return

    # 5. Confirm GPT-5.4 beats ours on each (gate out regressions)
    print(f"\n[confirm] A/B judge: ours vs GPT-5.4 on {len(pairs)} problems...")
    judge_client = create_client()
    surviving = []
    for pid, prob, ans_a, ans_b, loss_reason, d in pairs:
        rng = random.Random(hash(pid) % (2**32))
        if rng.random() < 0.5:
            left, right, a_was = ans_a, ans_b, "A"  # "A" is ours
        else:
            left, right, a_was = ans_b, ans_a, "B"  # "A" is gpt5
        try:
            v = judge_pair(judge_client, prob, left, right)
        except Exception as e:
            print(f"  [judge err {pid}] {e}")
            continue
        w = v.get("winner", "tie")
        sa = v.get("score_a", 0); sb = v.get("score_b", 0)
        # Normalize: is ours winner?
        if w == "tie":
            continue
        ours_won = (w == a_was)
        if a_was == "A":
            score_ours, score_gpt = sa, sb
        else:
            score_ours, score_gpt = sb, sa
        gap = score_gpt - score_ours
        if not ours_won and gap >= args.gap:
            print(f"  [{pid}] ✓ GPT-5.4 > ours by {gap}; reasoning: {v.get('reasoning','')[:60]}")
            surviving.append({
                "pid": pid,
                "problem": prob,
                "ours_answer": ans_a,
                "gpt5_answer": ans_b,
                "orig_loss_reasoning": loss_reason,
                "confirmed_judge_reasoning": v.get("reasoning", ""),
                "confirmed_gap": gap,
                "domain": d["domain"],
                "difficulty": d["difficulty"],
            })
        else:
            print(f"  [{pid}] ✗ ours won or gap<{args.gap} — drop")
    _save_content_cache()

    if not surviving:
        print("No confirmed GPT-5.4 wins — no distillation signal.")
        return

    print(f"\nSurvived: {len(surviving)} confirmed GPT-5.4 wins")

    # 6. Single-pass distillation
    registry = load_or_init_registry()
    active = [w for w in registry["wisdoms"] if w.get("status") == "active"]
    wisdom_brief = "\n".join(
        f"[{w['id']}] {w['aphorism']} — {w.get('signal','')[:45]}"
        for w in active
    )

    evidence_block = "\n\n---\n\n".join(
        f"## [{s['pid']}] [{s['domain']}/{s['difficulty']}]\n"
        f"问题: {s['problem'][:300]}\n\n"
        f"A (gemini-3-flash+scaffolding, 输):\n{s['ours_answer'][:600]}\n\n"
        f"B (GPT-5.4, 赢, gap={s['confirmed_gap']}):\n{s['gpt5_answer'][:600]}\n\n"
        f"判词 (原/复测): {s['orig_loss_reasoning'][:150]} / {s['confirmed_judge_reasoning'][:150]}"
        for s in surviving
    )

    pid_list_str = ", ".join(f'"{s["pid"]}"' for s in surviving)
    prompt = DISTILL_PROMPT.format(
        n=len(surviving), wisdom_brief=wisdom_brief,
        evidence=evidence_block, pid_list=pid_list_str,
    )
    print(f"\n[distill] GPT-5.4 naming recurring orientation...")
    resp = gpt.generate(prompt, max_tokens=1500, temperature=0.35)
    parsed_resp = parse_json_from_llm(resp["text"])
    proposal = parsed_resp.get("proposal")
    if not proposal:
        reason = parsed_resp.get("reason", "no proposal")
        print(f"  [rejected by GPT-5.4] {reason[:120]}")
        return

    # 7. Novelty check
    print(f"\n[novelty] computing embedding similarity...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    existing_texts = [w["unpacked_for_llm"] for w in active]
    existing_embs = model.encode(existing_texts, normalize_embeddings=True,
                                  show_progress_bar=False)
    new_emb = model.encode([proposal.get("unpacked_for_llm", "")],
                           normalize_embeddings=True)[0]
    max_sim = float(np.max(existing_embs @ new_emb))
    proposal["novelty_sim"] = max_sim
    proposal["_source"] = "cross_llm"
    proposal["_evidence_pids"] = [s["pid"] for s in surviving]

    print(f"  aphorism: {proposal.get('aphorism','?')}")
    print(f"  source:   {proposal.get('source','?')}")
    print(f"  novelty:  max_sim={max_sim:.2f}  threshold={args.novelty_max_sim}")

    if max_sim > args.novelty_max_sim:
        print(f"  [REJECT] too similar to existing wisdom")
        return

    # 8. Persist
    cands = cache_load(OUT_CANDIDATES, default=[])
    cands.append(proposal)
    cache_save(OUT_CANDIDATES, cands)

    log = cache_load(OUT_LOG, default=[])
    log.append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "judgment_file": args.judgments,
        "hard_losses_attempted": len(losses),
        "gpt5_solved": len(pairs),
        "gpt5_confirmed_wins": len(surviving),
        "candidate_aphorism": proposal.get("aphorism"),
        "novelty_sim": max_sim,
        "evidence_pids": [s["pid"] for s in surviving],
    })
    cache_save(OUT_LOG, log)

    print(f"\n=== Candidate produced ===")
    print(f"  {proposal.get('aphorism')} ({proposal.get('source')})")
    print(f"  signal: {proposal.get('signal','')}")
    print(f"  Saved → {OUT_CANDIDATES.name}")
    print(f"  Next step: merge into success_distilled_candidates pool or run validate_parallel directly")


if __name__ == "__main__":
    main()
