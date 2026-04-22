"""
baseline_long: give plain baseline the SAME token/length budget as v12c on math/sci.

Answer user's question: if baseline writes at v12c's length, is v12c's +14pp real
reasoning or just budget/length artifact?

Routing:
  - math:    BASELINE_PROMPT-math (550 char 可放宽, max_tokens=1100)
  - science: BASELINE_PROMPT-sci  (500 char,       max_tokens=900)
  - others:  original BASELINE_PROMPT (400 char,   max_tokens=800)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

import _config as cfg  # noqa: E402
from llm_client import create_client  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from cached_framework import _generate_with_retry, BASELINE_PROMPT


CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE / "answers"


BASELINE_MATH_LONG = """你是一位严谨的问题解决者。针对下面的数学问题，给出清晰、完整的解答。

要求：
1. 先简要重述问题核心
2. 给出你的分析和推理步骤
3. 给出最终建议/解答
4. 不超过 550 字（数学题可适当放宽）

## 问题
{problem}
"""

BASELINE_SCIENCE_LONG = """你是一位严谨的问题解决者。针对下面的科学问题，给出清晰、结构化的解答。

要求：
1. 先简要重述问题核心
2. 给出你的分析和推理步骤
3. 给出最终建议/解答
4. 不超过 500 字

## 问题
{problem}
"""


def cache_load(p):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def cache_save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline_long")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    answers = cache_load(answers_path)

    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))[: args.n]

    client = create_client()
    t0 = time.time()
    new = hit = 0
    math = sci = other = 0

    for i, p in enumerate(sample):
        pid = p["problem_id"]
        if pid in answers:
            hit += 1
            continue
        dom = p.get("domain", "?")
        problem = p.get("description", "")

        if dom == "mathematics":
            prompt = BASELINE_MATH_LONG.format(problem=problem)
            max_tok = 1100
            math += 1
        elif dom == "science":
            prompt = BASELINE_SCIENCE_LONG.format(problem=problem)
            max_tok = 900
            sci += 1
        else:
            prompt = BASELINE_PROMPT.format(problem=problem)
            max_tok = 800
            other += 1

        try:
            resp = _generate_with_retry(client, prompt, max_tokens=max_tok, temperature=0.3)
            answers[pid] = resp["text"].strip()
        except Exception as e:
            print(f"  [err] {pid}: {e}")
            continue

        new += 1
        if new % 10 == 0:
            cache_save(answers_path, answers)
            print(f"  [{args.variant}] {i+1}/{len(sample)} math={math} sci={sci} other={other} {time.time()-t0:.0f}s")

    cache_save(answers_path, answers)
    print(f"\n  [{args.variant}] done: math={math} sci={sci} other={other} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
