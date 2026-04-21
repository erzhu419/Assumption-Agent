"""
Track C: LLM auto-detects form per task category before SELECT stage.

Stage 0 (NEW): For each (domain, difficulty) task category, given 3 representative
problems, LLM classifies:
  "procedural" → problem needs step-by-step rigor (use technique-form modules)
  "orientational" → problem needs attention priors (use orientation-form modules)

Based on Stage 0 output, the variant uses either:
  - technique form (original 27 strategies via cached_framework pipeline)
  - orientation form (12 orient + 15 technique via orientation_framework)

Key property: decision is made AUTOMATICALLY — no human domain labels.
If classifier accuracy is high, result should match domain_gated.
If classifier finds finer distinctions (e.g. "engineering hard" is procedural
while "engineering medium" is orientational), it may even beat domain_gated.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

import _config as cfg  # noqa: E402
from llm_client import create_client, parse_json_from_llm  # noqa: E402

# Reuse pipelines from the two existing frameworks
sys.path.insert(0, str(Path(__file__).parent))
from cached_framework import (
    _generate_with_retry,
    discover_structure as discover_technique,
    generate_self_discover_answer as execute_technique,
    load_ours_27_modules,
)
from orientation_framework import (
    discover_structure_orient as discover_orientation,
    execute_orient as execute_orientation,
    load_orient_modules,
)


CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE / "answers"
STRUCTURES_DIR = CACHE / "structures"
CLASSIFIER_CACHE = CACHE / "form_classifier.json"


DETECT_PROMPT = """# 你是推理模式分析专家。

下面是同一类任务（领域+难度）的 3 个代表问题。请判断这类任务**更需要哪种推理模式**：

**procedural（程序性）** — 需要严格的步骤、精确的计算、证明的严谨性。典型如：数学证明、算法推导、物理公式应用、代码实现的正确性论证。

**orientational（意识朝向）** — 需要多维思考、反思性觉知、对潜在盲点的警觉。典型如：商业决策、人生建议、工程权衡、跨利益方的判断。

## 任务类别
领域: {domain}
难度: {difficulty}

## 3 个代表问题
{examples}

## 你的任务
基于这 3 个问题的**思考模式需求**，判断归类。不是看题目来自哪个领域 —— 有的"工程"问题本质是决策（orientational），有的"日常生活"问题本质是严密推理（procedural）。

输出 JSON（不要代码块）：
{{"form": "procedural" 或 "orientational",
  "confidence": 0.0-1.0,
  "reasoning": "为什么归入这一类（50字内）"}}
"""


def classify_category(client, domain: str, difficulty: str,
                      examples: List[Dict]) -> Dict:
    ex_text = "\n\n".join(
        f"示例 {i+1}: {p.get('description', '')[:400]}"
        for i, p in enumerate(examples)
    )
    resp = _generate_with_retry(client,
        DETECT_PROMPT.format(domain=domain, difficulty=difficulty, examples=ex_text),
        max_tokens=200, temperature=0.1)
    try:
        return parse_json_from_llm(resp["text"])
    except Exception:
        return {"form": "orientational", "confidence": 0.5, "reasoning": "parse_failure"}


def cache_load(p: Path):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def cache_save(p: Path, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="auto_detect")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    struct_path = STRUCTURES_DIR / f"{args.variant}_structures.json"
    classifier_cache = cache_load(CLASSIFIER_CACHE)
    answers = cache_load(answers_path)
    structures = cache_load(struct_path)

    # Load sample (fixed seed=42 100 problems)
    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))[: args.n]

    # Group train problems by category for seed examples
    from task_env.base_env import TaskEnvironment
    kb = {}
    for f in sorted(cfg.KB_DIR.glob("S*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        kb[d["id"]] = d
    env = TaskEnvironment(strategy_kb=kb)
    train_pool = env.get_all_problems("train")
    by_cat = defaultdict(list)
    for p in train_pool:
        by_cat[(p.get("domain", "?"), p.get("difficulty", "?"))].append(p)

    needed_cats = set()
    for p in sample:
        needed_cats.add((p.get("domain", "?"), p.get("difficulty", "?")))

    client = create_client()
    t0 = time.time()

    # Stage 0: classify each category
    print("\n[Stage 0] Classifying categories...")
    technique_modules = load_ours_27_modules()
    orientation_modules = load_orient_modules(hybrid=True)

    for cat in sorted(needed_cats):
        key = f"{cat[0]}__{cat[1]}"
        if key in classifier_cache:
            continue
        if cat not in by_cat or len(by_cat[cat]) < 3:
            print(f"  [skip-category] {cat}")
            continue
        rng = random.Random(hash(("classify", cat)) & 0xFFFFFFFF)
        examples = rng.sample(by_cat[cat], 3)
        result = classify_category(client, cat[0], cat[1], examples)
        classifier_cache[key] = {
            "form": result.get("form", "orientational"),
            "confidence": float(result.get("confidence", 0.5)),
            "reasoning": result.get("reasoning", ""),
        }
        cache_save(CLASSIFIER_CACHE, classifier_cache)
        print(f"  {cat}: {classifier_cache[key]['form']} "
              f"(conf={classifier_cache[key]['confidence']:.2f}) "
              f"— {classifier_cache[key]['reasoning'][:60]}")

    # Stage 1: structure discovery per (category, form)
    print("\n[Stage 1] Discovering structures based on classifier decision...")
    for cat in sorted(needed_cats):
        key = f"{cat[0]}__{cat[1]}"
        if key in structures:
            continue
        form = classifier_cache.get(key, {}).get("form", "orientational")
        if cat not in by_cat or len(by_cat[cat]) < 3:
            continue
        rng = random.Random(hash(("seed", cat)) & 0xFFFFFFFF)
        examples = rng.sample(by_cat[cat], 3)
        print(f"  [{form}] discovering {cat}")
        if form == "procedural":
            # Use technique-form pipeline
            struct = discover_technique(client, technique_modules, cat[0], cat[1], examples)
            structures[key] = {"form": "procedural", **struct}
        else:
            struct = discover_orientation(client, orientation_modules, cat[0], cat[1], examples)
            structures[key] = {"form": "orientational", **struct}
        cache_save(struct_path, structures)

    # Stage 2: answer each problem using its category's form
    print("\n[Stage 2] Generating answers with form-matched execute...")
    new_count = 0
    for i, p in enumerate(sample):
        pid = p["problem_id"]
        if pid in answers:
            continue
        key = f"{p.get('domain', '?')}__{p.get('difficulty', '?')}"
        struct = structures.get(key)
        if struct is None:
            continue
        form = struct.get("form", "orientational")
        if form == "procedural":
            # Build structure dict matching execute_technique's expected format
            structure_inner = struct.get("structure", {})
            if "structure" in struct:
                # struct wraps
                answers[pid] = execute_technique(client, p.get("description", ""), struct["structure"])
            else:
                answers[pid] = execute_technique(client, p.get("description", ""), structure_inner)
        else:
            answers[pid] = execute_orientation(client, p.get("description", ""), struct)
        new_count += 1
        if new_count % 10 == 0:
            cache_save(answers_path, answers)
            print(f"  [{args.variant}] {i+1}/{len(sample)} new={new_count} {time.time()-t0:.0f}s")
    cache_save(answers_path, answers)

    # Report classifier decisions
    print("\n=== Classifier decisions ===")
    for key, v in sorted(classifier_cache.items()):
        print(f"  {key}: {v['form']}  conf={v['confidence']:.2f}")

    # Compare against domain-gated expectation
    USE_ORIENT_DOM = {"business", "daily_life", "software_engineering", "engineering"}
    agree = 0
    total = 0
    for key, v in classifier_cache.items():
        dom = key.split("__")[0]
        expected = "orientational" if dom in USE_ORIENT_DOM else "procedural"
        if v["form"] == expected:
            agree += 1
        total += 1
    print(f"\nClassifier agreement with naive domain rule: {agree}/{total} = {agree/max(total,1):.0%}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
