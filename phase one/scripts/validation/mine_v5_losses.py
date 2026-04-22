"""
Iterate on v5: mine triggers from v5's 48 remaining losses (vs baseline).

These are the cases where our current best variant STILL loses. Mining these
should produce triggers targeting specific residual failure modes.

Uses SAME pipeline as original mine_triggers.py (Gemini Flash, 2-3 triggers
per loss, category-keyed). Output merged with existing trigger_library.json
into trigger_library_v6.json (for v6 variant).

Complementary to (but distinct from) wisdom library — these are category-
specific paraphrases in the proven-effective form.
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

sys.path.insert(0, str(Path(__file__).parent))

CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
EXISTING_TRIGGERS = CACHE / "trigger_library.json"
OUT_PATH = CACHE / "trigger_library_v6.json"


EXTRACT_PROMPT = """你要从一次失败案例中提炼"我们本该察觉但错过的早期信号"。这是在 v5 版本（已经有 Stage-1 priors + 161 triggers + wisdom library）基础上**仍然失败**的残余 loss —— 提炼的 trigger 需要更 targeted、更针对性。

## 问题
{problem}

## 我们的解答（v5 版本，已经很强，但仍输了）
{our_answer}

## 对手的解答（赢了）
{baseline_answer}

## 评委理由
{judge_reasoning}

## 你的任务

找出 "对手注意到而 v5 没注意到" 的 **specific 警觉或反问**，形式为**场景触发 + 具体自问**（这是经过验证最适合 LLM 消化的形式）。

好的 trigger 例子：
- "当发现自己反复朝同一个方向优化时，问自己：优化目标本身是不是定错了？"
- "在给出建议前，反问：对方真正想要的是我给的答案，还是一个能让他自己做决定的框架？"

不要生成：
- 模板化 meta 反思（"是否充分识别"类）
- 过度具体的 instance-level trigger（"本案例中..."类）
- Aphorism 或警句（≤20 字不够 LLM 消化）

输出 JSON（不要代码块）：

{{"triggers": [
  "场景触发 + 具体自问 (40-60 字)，要比我们现有 triggers 更 targeted",
  "...最多 3 条。如果该 loss 没新 insight 可挖，返回空列表"
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
    judgments_path = CACHE / "judgments" / "phase2_v5_vs_baseline.json"
    judgments = json.loads(judgments_path.read_text(encoding="utf-8"))

    v5_ans = json.loads((CACHE / "answers" / "phase2_v5_answers.json").read_text(encoding="utf-8"))
    base_ans = json.loads((CACHE / "answers" / "baseline_answers.json").read_text(encoding="utf-8"))
    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))
    prob_by_id = {p["problem_id"]: p for p in sample}

    losses = []
    for pid, j in judgments.items():
        if j.get("winner") != "baseline":
            continue
        if pid in v5_ans and pid in base_ans and pid in prob_by_id:
            losses.append({
                "problem_id": pid,
                "domain": j.get("domain", "?"),
                "difficulty": j.get("difficulty", "?"),
                "our_answer": v5_ans[pid],
                "baseline_answer": base_ans[pid],
                "judge_reasoning": j.get("reasoning", ""),
                "problem": prob_by_id[pid].get("description", ""),
            })

    print(f"v5 losses to mine: {len(losses)}")
    by_cat = defaultdict(int)
    for l in losses:
        by_cat[f"{l['domain']}__{l['difficulty']}"] += 1
    for k, v in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    # Start with existing triggers
    existing = json.loads(EXISTING_TRIGGERS.read_text(encoding="utf-8"))
    by_category: Dict[str, List[str]] = defaultdict(list)
    for cat_key, trigs in existing.items():
        by_category[cat_key].extend(trigs)

    original_count = sum(len(v) for v in by_category.values())
    print(f"\n  Starting from {original_count} existing triggers across {len(by_category)} categories")

    client = create_client()
    mined = 0
    t0 = time.time()

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
                if isinstance(t, str) and 20 < len(t) < 200:
                    # Check not already in library
                    existing_lower = [x.lower() for x in by_category[cat_key]]
                    if t.lower()[:30] not in [x[:30] for x in existing_lower]:
                        by_category[cat_key].append(t.strip())
                        mined += 1
        except Exception as e:
            print(f"  [skip] {loss['problem_id']}: {e}")
            continue

        if (i + 1) % 10 == 0:
            OUT_PATH.write_text(json.dumps(dict(by_category), ensure_ascii=False, indent=2))
            print(f"  [{i+1}/{len(losses)}] new mined {mined}; {time.time()-t0:.0f}s")

    OUT_PATH.write_text(json.dumps(dict(by_category), ensure_ascii=False, indent=2))
    new_total = sum(len(v) for v in by_category.values())
    print(f"\n=== Summary ===")
    print(f"  {original_count} original + {mined} new = {new_total} total")
    print(f"  Saved to {OUT_PATH.name}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
