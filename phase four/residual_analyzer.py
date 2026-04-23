"""
Phase 4 v2 MVP Step 1: Analyze v16's residual failures via GPT-5.4.

For every problem where v16 lost or tied (against baseline_long / ours_27 / v13_reflect),
extract:
  - what orientation v16 missed
  - nearest existing wisdom (and how well it covers this)
  - proposed refinement OR "novel pattern needed"
  - cluster_tag for later aggregation

Outputs: phase four/residuals/v16_residuals.json

This is the Phase 4 v2 entry point — after this analysis, we'll human-check
cluster quality, then decide between Mode A (refine existing) or Mode B
(propose new wisdom).
"""

import json
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
from llm_client import parse_json_from_llm
from gpt5_client import GPT5Client


CACHE = PROJECT / "phase two" / "analysis" / "cache"
JUDGMENTS = CACHE / "judgments"
ANSWERS = CACHE / "answers"
WISDOM_PATH = CACHE / "wisdom_library.json"
SAMPLE = CACHE / "sample_100.json"
SELECTIONS = CACHE / "phase2_v3_selections.json"

OUT_DIR = PROJECT / "phase four" / "residuals"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "v16_residuals.json"

# Which comparisons to extract v16 losses from
V16_JUDGMENT_FILES = [
    ("phase2_v16_vs_baseline_long.json", "baseline_long"),
    ("phase2_v16_vs_ours_27.json", "ours_27"),
    ("phase2_v16_vs_phase2_v13_reflect.json", "phase2_v13_reflect"),
]


ANALYZE_PROMPT = """你是 wisdom library residual 分析专家。下面这道题，v16 架构 (cases + audit) 输给了对手架构。你的任务是诊断 v16 为什么输——**不是表面原因，是它 miss 的 orientation**。

## 问题
{problem}

## v16 选用的 wisdoms
{v16_wisdom_list}

## v16 答案
{v16_answer}

## 对手答案（{winner_variant}）
{winner_answer}

## 判官的理由
{judge_reason}

## 完整 wisdom library（75 条，只给名称 + signal，便于定位）
{wisdom_brief}

## 分析任务
输出 JSON（不要代码块）：
{{
  "what_v16_missed": "v16 miss 了什么 orientation/视角 (30-60 字)，要 specific",
  "nearest_existing_wisdom": "W0XX 或 null（如果 library 里真的没有近似的）",
  "wisdom_applicability": 0-10 整数,  // 现有 wisdom 能覆盖这 miss 的程度 (10=完美覆盖，0=完全不沾)
  "proposed_refinement": "如果 W0XX 存在但 miss 了某个应用方向，具体怎么扩展 signal/unpacked (40-80 字)。若是 novel pattern，写 null",
  "novel_orientation_needed": "如果现有 wisdom 都不覆盖，描述需要的新 orientation (30-60 字)。若不需要，写 null",
  "cluster_tag": "6-15 字的短标签，用于聚类 (如 '隐含权衡盲区', '时序 dependency 忽视')"
}}
"""


def main():
    # Load everything
    judgments = {}
    for fname, opponent in V16_JUDGMENT_FILES:
        j = json.loads((JUDGMENTS / fname).read_text(encoding="utf-8"))
        for pid, v in j.items():
            winner = v.get("winner", "")
            # v16 lost or tied
            if winner != "phase2_v16":
                if pid not in judgments:
                    judgments[pid] = []
                judgments[pid].append({
                    "opponent": opponent,
                    "winner": winner,
                    "reasoning": v.get("reasoning", ""),
                })

    print(f"v16 residuals (lost/tied): {len(judgments)} unique problems")

    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    pid_to_info = {p["problem_id"]: p for p in sample}
    wisdom = json.loads(WISDOM_PATH.read_text(encoding="utf-8"))
    wid_to_entry = {w["id"]: w for w in wisdom}
    selections = json.loads(SELECTIONS.read_text(encoding="utf-8"))

    # Compact wisdom brief for prompt
    wisdom_brief_lines = [
        f"[{w['id']}] {w['aphorism']} — {w.get('signal', '')[:40]}"
        for w in wisdom
    ]
    wisdom_brief = "\n".join(wisdom_brief_lines)

    # Load answers
    v16_ans = json.loads((ANSWERS / "phase2_v16_answers.json").read_text(encoding="utf-8"))
    opp_ans = {}
    for _, opponent in V16_JUDGMENT_FILES:
        opp_ans[opponent] = json.loads((ANSWERS / f"{opponent}_answers.json").read_text(encoding="utf-8"))

    # Resume
    out = {}
    if OUT_PATH.exists():
        try:
            out = json.loads(OUT_PATH.read_text(encoding="utf-8"))
            print(f"Resuming: {len(out)} already analyzed")
        except Exception:
            out = {}

    client = GPT5Client()
    t0 = time.time()
    new_count = 0
    errors = 0

    for i, (pid, losses) in enumerate(judgments.items()):
        if pid in out:
            continue

        problem_info = pid_to_info.get(pid)
        if not problem_info:
            continue

        # Pick the strongest opponent for analysis (v13_reflect > ours_27 > baseline_long)
        preferred_order = ["phase2_v13_reflect", "ours_27", "baseline_long"]
        losses_sorted = sorted(losses,
                               key=lambda x: preferred_order.index(x["opponent"])
                               if x["opponent"] in preferred_order else 99)
        primary_loss = losses_sorted[0]
        opponent = primary_loss["opponent"]

        v16_a = v16_ans.get(pid, "")
        opp_a = opp_ans[opponent].get(pid, "")
        if not v16_a or not opp_a:
            continue

        v16_wisdom_ids = selections.get(pid, [])[:2]
        v16_wisdom_list = "\n".join(
            f"  - [{wid}] {wid_to_entry[wid]['aphorism']}: {wid_to_entry[wid].get('signal', '')[:50]}"
            for wid in v16_wisdom_ids if wid in wid_to_entry) or "  (无)"

        prompt = ANALYZE_PROMPT.format(
            problem=problem_info.get("description", "")[:500],
            v16_wisdom_list=v16_wisdom_list,
            v16_answer=v16_a[:800],
            winner_variant=opponent,
            winner_answer=opp_a[:800],
            judge_reason=primary_loss["reasoning"][:300],
            wisdom_brief=wisdom_brief,
        )

        for attempt in range(3):
            try:
                resp = client.generate(prompt, max_tokens=600, temperature=0.3)
                parsed = parse_json_from_llm(resp["text"])
                # Validate required fields
                required = {"what_v16_missed", "nearest_existing_wisdom",
                            "wisdom_applicability", "cluster_tag"}
                if not required.issubset(parsed.keys()):
                    raise ValueError(f"missing fields: {required - parsed.keys()}")
                out[pid] = {
                    "problem_id": pid,
                    "domain": problem_info.get("domain", "?"),
                    "difficulty": problem_info.get("difficulty", "?"),
                    "primary_opponent": opponent,
                    "all_opponents": [l["opponent"] for l in losses],
                    **parsed,
                }
                new_count += 1
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [fail {pid}] {e}")
                    errors += 1
                    break
                time.sleep(3)

        if new_count % 5 == 0 and new_count > 0:
            OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
            print(f"  [{pid}] {new_count} analyzed, {time.time()-t0:.0f}s")

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n=== Summary ===")
    print(f"  Analyzed: {len(out)} / {len(judgments)} residuals")
    print(f"  Errors: {errors}")
    print(f"  Time: {time.time()-t0:.0f}s")

    # Cluster statistics
    print("\n=== Cluster tags ===")
    tag_c = Counter(r.get("cluster_tag", "?") for r in out.values())
    for tag, n in tag_c.most_common(20):
        print(f"  {n:>3}x  {tag}")

    # Wisdom applicability distribution
    print("\n=== Wisdom applicability distribution ===")
    apps = [r.get("wisdom_applicability", 0) for r in out.values()]
    buckets = Counter()
    for a in apps:
        if a <= 3: buckets["low (0-3: Mode B candidate)"] += 1
        elif a <= 6: buckets["mid (4-6: Mode A candidate)"] += 1
        else: buckets["high (7-10: wisdom exists, refine)"] += 1
    for k, v in buckets.most_common():
        print(f"  {v:>3}  {k}")

    # Most common nearest wisdom (for Mode A targeting)
    print("\n=== Most-invoked 'nearest existing wisdom' ===")
    nw_c = Counter(r.get("nearest_existing_wisdom", "?") for r in out.values()
                   if r.get("nearest_existing_wisdom"))
    for wid, n in nw_c.most_common(10):
        if wid in wid_to_entry:
            print(f"  {n:>3}x  [{wid}] {wid_to_entry[wid]['aphorism']}")


if __name__ == "__main__":
    main()
