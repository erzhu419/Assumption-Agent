"""
Phase 2 改造 — Failure-mined awareness triggers.

For each case where our system lost to baseline, ask LLM:
  - What did baseline notice that we missed?
  - What attention prior would have caught this before answering?

Output: trigger_library.json keyed by (domain, difficulty), each entry a list of
  awareness triggers (short sentences starting with "注意..." or "警觉..."
  or similar orientation phrases).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

from llm_client import create_client, parse_json_from_llm  # noqa: E402

CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
TRIGGERS_PATH = CACHE / "trigger_library.json"


EXTRACT_PROMPT = """你要从一次失败案例里挖出 **"我们本该察觉但错过的早期信号"**。

## 问题
{problem}

## 我们的解答（输了）
{our_answer}

## 对手的解答（赢了）
{baseline_answer}

## 评委理由
{judge_reasoning}

## 你的任务
**不是**再解一遍问题。而是：找出"对手注意到的，我们没注意到的东西"。这个东西必须是一个**意识朝向**（注意力先验），不是具体技巧。

好的 triggers 例子（意识朝向）：
- "注意：实用建议前，先问'当事人的约束条件是什么'"
- "警觉：推理前，先识别'这是真冲突还是假冲突'"
- "反问：我给出的答案是否回避了问题中最尖锐的部分？"

坏的 triggers 例子（技巧/方案，不采用）：
- "使用 SWOT 分析" （这是技巧）
- "第一步做 X，第二步做 Y" （步骤）
- "答案应该包括 A、B、C" （内容清单）

输出 JSON（不要代码块）：
{{"triggers": [
  "意识朝向 1（30-50字，反问或警觉式表达）",
  "意识朝向 2",
  "（最多 3 条；如果对手没有明显优势，可以返回空列表）"
]}}
"""


def _gen(client, prompt: str, max_retries: int = 6) -> str:
    for attempt in range(max_retries):
        try:
            return client.generate(prompt, max_tokens=400, temperature=0.2)["text"]
        except Exception as e:
            msg = str(e).lower()
            transient = any(k in msg for k in ["503", "429", "500", "unavailable",
                "resource_exhausted", "overloaded", "high demand", "disconnect",
                "remoteprotocol", "timeout", "connection"])
            if not transient or attempt == max_retries - 1:
                raise
            wait = min(2 ** attempt * 5, 120)
            print(f"    [retry {attempt+1}/{max_retries}] {str(e)[:80]}... wait {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="orient_hybrid_vs_baseline",
                    help="judgment cache to mine losses from")
    ap.add_argument("--loser", default="orient_hybrid",
                    help="variant name to mine losses OF")
    args = ap.parse_args()

    # Load judgments
    judgments_path = CACHE / "judgments" / f"{args.source}.json"
    judgments = json.loads(judgments_path.read_text(encoding="utf-8"))

    # Load answers
    our_answers = json.loads((CACHE / "answers" / f"{args.loser}_answers.json").read_text(encoding="utf-8"))
    base_answers = json.loads((CACHE / "answers" / "baseline_answers.json").read_text(encoding="utf-8"))

    # Load sample for problem descriptions
    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))
    prob_by_id = {p["problem_id"]: p for p in sample}

    # Identify losses (winner == "baseline", i.e. orient_hybrid lost)
    losses = []
    for pid, j in judgments.items():
        if j.get("winner") == "baseline":
            if pid in our_answers and pid in base_answers and pid in prob_by_id:
                losses.append({
                    "problem_id": pid,
                    "domain": j.get("domain", "?"),
                    "difficulty": j.get("difficulty", "?"),
                    "our_answer": our_answers[pid],
                    "baseline_answer": base_answers[pid],
                    "judge_reasoning": j.get("reasoning", ""),
                    "problem": prob_by_id[pid].get("description", ""),
                })

    print(f"Mining {len(losses)} loss cases (loser={args.loser}) from {args.source}")

    # Load existing triggers (if resuming)
    existing = {}
    if TRIGGERS_PATH.exists():
        existing = json.loads(TRIGGERS_PATH.read_text(encoding="utf-8"))

    # Mine
    client = create_client()
    by_category: Dict[str, List[str]] = defaultdict(list)
    # Seed from existing
    for cat_key, triggers in existing.items():
        by_category[cat_key].extend(triggers)

    t0 = time.time()
    mined = 0
    for i, loss in enumerate(losses):
        try:
            prompt = EXTRACT_PROMPT.format(
                problem=loss["problem"][:600],
                our_answer=loss["our_answer"][:500],
                baseline_answer=loss["baseline_answer"][:500],
                judge_reasoning=loss["judge_reasoning"][:200],
            )
            raw = _gen(client, prompt)
            parsed = parse_json_from_llm(raw)
            triggers = parsed.get("triggers", [])
            cat_key = f"{loss['domain']}__{loss['difficulty']}"
            for t in triggers:
                if isinstance(t, str) and 10 < len(t) < 200:
                    by_category[cat_key].append(t.strip())
                    mined += 1
        except Exception as e:
            print(f"  [skip] {loss['problem_id']}: {e}")
            continue

        if (i + 1) % 10 == 0:
            TRIGGERS_PATH.write_text(json.dumps(dict(by_category), ensure_ascii=False, indent=2))
            print(f"  [{i+1}/{len(losses)}] mined {mined} triggers; {time.time()-t0:.0f}s")

    # Save
    TRIGGERS_PATH.write_text(json.dumps(dict(by_category), ensure_ascii=False, indent=2))

    # Summary
    print(f"\n=== Trigger library summary ===")
    for cat_key in sorted(by_category.keys()):
        trigs = by_category[cat_key]
        print(f"  {cat_key}: {len(trigs)} triggers")
        for t in trigs[:2]:
            print(f"    - {t[:100]}")

    print(f"\nTotal: {sum(len(v) for v in by_category.values())} triggers "
          f"across {len(by_category)} categories")
    print(f"Saved to {TRIGGERS_PATH.name}")


if __name__ == "__main__":
    main()
