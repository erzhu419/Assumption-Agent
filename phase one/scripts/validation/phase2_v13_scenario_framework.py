"""
Phase 2 v13-scenario — Scenario-Mediated Reasoning (MVP, 2-turn compressed).

Based on user's insight: meta-principles (Russell, biblical, sunk cost) don't
directly apply to problems — human cognition parses them into 2-3 hypothetical
"rehearsal scenarios" specific to the problem's structure, then projects the
actual situation onto those scenarios (with continuous interpolation, not
dogmatic match to one).

This is closer to Kahneman System 2 mental simulation than to Self-Refine.

Turn 1 — generate scenarios + their signal weights:
  "Given problem P and meta-principles M, generate 3 hypothetical scenarios
   that instantiate M on P's structure. For each, give a 'weight' reflecting
   how much P's actual details hint at it."

Turn 2 — synthesize answer conditioned on weighted scenarios:
  "Use the weighted scenarios as structure. Don't match one rigidly — reason
   in between them. Produce final answer."

Math/sci still bypass to v12c hygiene (user's world-model-weak domains).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

import _config as cfg  # noqa: E402
from llm_client import create_client  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from cached_framework import _generate_with_retry
from phase2_v12_framework import (EXECUTE_MATH, EXECUTE_SCIENCE, format_priors,
                                   format_wisdom, MATH_SCI)


CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE / "answers"
STRUCTURES_DIR = CACHE / "structures"
WISDOM_PATH = CACHE / "wisdom_library.json"
SELECTIONS_PATH = CACHE / "phase2_v3_selections.json"


# Turn 1: scenario generation with self-estimated weights
SCENARIO_PROMPT = """你要为一个问题生成"演练场景"，用于后续的推理投影。

## 问题
{problem}

## 本类别的 meta-orientation（可参考，非教条）
{priors_block}

{wisdom_block}

## 任务
生成 **3 个假设性演练场景**，每个基于上面的 meta-principle 在本问题结构上具体化。

每个场景要：
- 30-80 字，具体到"在本问题的这种情况下会怎样展开"
- 3 个场景**路径上互不重复**（不同的可能方向）
- 附上 `signal_hit` 打分（0-10）：问题文本里多大比例的信号暗示这条路径**真的在发生**

## 输出 JSON（不要代码块）
{{"scenarios": [
  {{"description": "场景 1 ...", "signal_hit": 7}},
  {{"description": "场景 2 ...", "signal_hit": 3}},
  {{"description": "场景 3 ...", "signal_hit": 5}}
]}}
"""


# Turn 2: synthesize conditioned on scenarios
SYNTHESIZE_PROMPT = """你要解决下面的问题。在给出答案前，已经预先演练了 3 个假设场景，附带 signal 权重（眼前问题在各场景上有多像）。

## 问题
{problem}

## 预演场景（不是固定答案，而是参考路径）
{scenarios_block}

## 任务
- 用权重**引导**你的答案焦点（权重高的场景 → 需要重点考虑；权重低也有可能变化）
- **不要**"本题 more likely S1, 所以..."这种教条套用
- 做**连续化**：如果问题 partially 落在 S1 和 S2 之间，答案要 reflect 这个中间态
- 不超过 600 字

最终答案：
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


def parse_json_safe(text):
    from llm_client import parse_json_from_llm
    try:
        return parse_json_from_llm(text)
    except Exception:
        return None


def format_scenarios(scenarios):
    if not scenarios:
        return "  (未能生成，请直接作答)"
    lines = []
    for i, s in enumerate(scenarios, 1):
        desc = s.get("description", "")
        hit = s.get("signal_hit", 5)
        lines.append(f"  S{i} [signal={hit}/10]: {desc}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="phase2_v13_scenario")
    ap.add_argument("--base", default="orient_hybrid")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    scenarios_path = ANSWERS_DIR / f"{args.variant}_scenarios.json"
    struct_path = STRUCTURES_DIR / f"{args.variant}_structures.json"
    answers = cache_load(answers_path)
    scenarios_cache = cache_load(scenarios_path)

    if not struct_path.exists():
        base_struct = cache_load(STRUCTURES_DIR / f"{args.base}_structures.json")
        if base_struct:
            cache_save(struct_path, base_struct)
    structures = cache_load(struct_path)

    library = json.loads(WISDOM_PATH.read_text(encoding="utf-8"))
    lib_by_id = {e["id"]: e for e in library}
    selections = cache_load(SELECTIONS_PATH)
    print(f"  {args.variant}: wisdom={len(library)}")
    print(f"  math/sci -> v12c hygiene; others -> scenario-branch 2-turn")

    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))[: args.n]

    client = create_client()
    t0 = time.time()
    new = hit = scenario_count = hygiene_count = 0

    for i, p in enumerate(sample):
        pid = p["problem_id"]
        if pid in answers:
            hit += 1
            continue
        dom = p.get("domain", "?")
        diff = p.get("difficulty", "?")
        problem = p.get("description", "")

        if dom in MATH_SCI:
            if dom == "mathematics":
                prompt = EXECUTE_MATH.format(problem=problem)
                max_tok = 1100
            else:
                prompt = EXECUTE_SCIENCE.format(problem=problem)
                max_tok = 900
            try:
                resp = _generate_with_retry(client, prompt, max_tokens=max_tok, temperature=0.3)
                answers[pid] = resp["text"].strip()
            except Exception as e:
                print(f"  [err] {pid}: {e}")
                continue
            hygiene_count += 1
        else:
            key = f"{dom}__{diff}"
            struct = structures.get(key, {"attention_priors": []})
            priors = struct.get("attention_priors", [])
            sel_ids = selections.get(pid, [])
            wisdom_entries = [lib_by_id[sid] for sid in sel_ids if sid in lib_by_id]

            priors_fmt = format_priors(priors)
            wisdom_fmt = format_wisdom(wisdom_entries)
            if wisdom_fmt.strip():
                wisdom_block = f"## 本类别的 wisdom（可参考）\n{wisdom_fmt}\n"
            else:
                wisdom_block = ""

            # Turn 1: scenarios
            if pid in scenarios_cache:
                scenarios = scenarios_cache[pid]
            else:
                try:
                    r1 = _generate_with_retry(client, SCENARIO_PROMPT.format(
                        problem=problem, priors_block=priors_fmt,
                        wisdom_block=wisdom_block),
                        max_tokens=600, temperature=0.4)
                    parsed = parse_json_safe(r1["text"])
                    if parsed and isinstance(parsed.get("scenarios"), list):
                        scenarios = parsed["scenarios"][:3]
                    else:
                        scenarios = []
                    scenarios_cache[pid] = scenarios
                except Exception as e:
                    print(f"  [err scenario {pid}]: {e}")
                    scenarios = []
                    scenarios_cache[pid] = []

            # Turn 2: synthesize
            try:
                r2 = _generate_with_retry(client, SYNTHESIZE_PROMPT.format(
                    problem=problem, scenarios_block=format_scenarios(scenarios)),
                    max_tokens=900, temperature=0.3)
                answers[pid] = r2["text"].strip()
            except Exception as e:
                print(f"  [err synth {pid}]: {e}")
                continue
            scenario_count += 1

        new += 1
        if new % 10 == 0:
            cache_save(answers_path, answers)
            cache_save(scenarios_path, scenarios_cache)
            print(f"  [{args.variant}] {i+1}/{len(sample)} hyg={hygiene_count} scn={scenario_count} {time.time()-t0:.0f}s")

    cache_save(answers_path, answers)
    cache_save(scenarios_path, scenarios_cache)
    print(f"\n  [{args.variant}] done: hyg={hygiene_count} scn={scenario_count} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
