"""
Phase 3 改造 — hierarchical problem isomorphism via archetypes.

For each task category, LLM detects which universal archetypes
(from phase zero/kb/archetypes.json) apply. Archetype wisdom is then
injected as a 3rd layer of attention prior:

  Layer 1 (base): attention_priors from orientation Stage 1
  Layer 2 (Phase 2): failure-mined triggers
  Layer 3 (NEW): archetype wisdom (cross-civilizational)

Reuses phase2_triggers answers' Stage 0/1 structures — ONLY extends EXECUTE.
Variant: phase3_archetypes.
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
from llm_client import create_client, parse_json_from_llm  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from cached_framework import _generate_with_retry
from phase2_framework import select_triggers_for_category


CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
ARCHETYPES_PATH = cfg.PHASE0_DIR / "kb" / "archetypes.json"
TRIGGERS_PATH = CACHE / "trigger_library.json"
CATEGORY_ARCHETYPES_PATH = CACHE / "category_archetypes.json"

ANSWERS_DIR = CACHE / "answers"
STRUCTURES_DIR = CACHE / "structures"


DETECT_ARCHETYPE_PROMPT = """# 你是古典智慧应用专家。

## 任务类别
{domain} / {difficulty}

## 该类别的代表问题
{examples}

## 下面是 20 个跨文明的"原型"（Layer 3 universals，不是技巧是觉知）：
{archetype_list}

## 你的任务
从 20 个原型里选出 **对该类别问题最有 illuminating 作用的 2-3 个**。

选择标准：
- 这个原型的"核心智慧"能帮 LLM 在解答时多一个维度的觉知
- 不是"能套得上"，是"如果没有它，LLM 容易忽视重要的东西"
- 同一类别问题常常触及同一个 Layer 3 模式

输出 JSON（不要代码块；reasoning 限 **50 字内**，避免截断）：
{{"selected": ["A0X", "A0Y"], "reasoning": "简短理由（<50字）"}}
"""


EXECUTE_WITH_ARCHETYPES = """# 你要解决下面的问题。

## 觉知层级（越深越本质）

### Layer 3: 跨时代/跨文明的古典原型警觉 ——
{archetype_block}

### Layer 2: 从历史失败里积累的 category-specific 警觉 ——
{triggers_block}

### Layer 1: 本任务类别的通用 attention priors ——
{priors_block}

## 问题
{problem}

## 要求
- **不要**用 "Step 1、Step 2" 格式
- 带着上面三层觉知直接回答
- 如果某个 Layer 3 原型让你发现了问题的本质结构（例如识别出这是"沉没成本谬误"），在答案中明确指出这个关联
- 如果某个 Layer 2 警觉让你发现盲点，在结尾点出
- 语言精炼，不超过 500 字

请开始：
"""


def cache_load(p: Path) -> Dict:
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def cache_save(p: Path, obj: Dict):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def load_archetypes() -> List[Dict]:
    return json.loads(ARCHETYPES_PATH.read_text(encoding="utf-8"))["archetypes"]


def format_archetype_list(archetypes: List[Dict]) -> str:
    return "\n".join(
        f"- **{a['id']} {a['name']}**: {a['wisdom']}"
        for a in archetypes
    )


def detect_archetypes_for_category(client, domain: str, difficulty: str,
                                    examples: List[Dict],
                                    archetypes: List[Dict]) -> List[str]:
    ex_text = "\n\n".join(
        f"示例 {i+1}: {p.get('description', '')[:400]}"
        for i, p in enumerate(examples[:3])
    )
    valid_ids = {a["id"] for a in archetypes}
    resp = _generate_with_retry(client, DETECT_ARCHETYPE_PROMPT.format(
        domain=domain, difficulty=difficulty, examples=ex_text,
        archetype_list=format_archetype_list(archetypes),
    ), max_tokens=600, temperature=0.2)
    try:
        parsed = parse_json_from_llm(resp["text"])
        ids = parsed.get("selected", [])
        return [s for s in ids if isinstance(s, str) and s in valid_ids][:3]
    except Exception as e:
        # Last-ditch: regex-hunt A01-A20 from raw text (works even if JSON truncated)
        import re
        found = re.findall(r"A\d{2}", resp.get("text", ""))
        return [s for s in found if s in valid_ids][:3]


def execute_phase3(client, problem: str, priors: List[str],
                    triggers: List[str], archetype_info: List[Dict]) -> str:
    archetype_block = "\n".join(
        f"  - 【{a['name']}】 {a['wisdom']}" for a in archetype_info
    ) if archetype_info else "  (无特别原型适用)"
    triggers_block = "\n".join(f"  - {t}" for t in triggers) if triggers else "  (无历史警觉)"
    priors_block = "\n".join(f"  {i+1}. {p}" for i, p in enumerate(priors))
    r = _generate_with_retry(client,
        EXECUTE_WITH_ARCHETYPES.format(
            archetype_block=archetype_block,
            triggers_block=triggers_block,
            priors_block=priors_block,
            problem=problem,
        ), max_tokens=900, temperature=0.3)
    return r["text"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="phase3_archetypes")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    struct_path = STRUCTURES_DIR / f"{args.variant}_structures.json"
    answers = cache_load(answers_path)

    # Reuse phase2_triggers structures (same Stage 0/1)
    phase2_struct_path = STRUCTURES_DIR / "phase2_triggers_structures.json"
    if not struct_path.exists() and phase2_struct_path.exists():
        cache_save(struct_path, json.loads(phase2_struct_path.read_text(encoding="utf-8")))
        print(f"  [reused] copied phase2_triggers structures")
    structures = cache_load(struct_path)

    # Load supporting data
    archetypes = load_archetypes()
    archetype_by_id = {a["id"]: a for a in archetypes}
    triggers_db = cache_load(TRIGGERS_PATH)
    cat_archetypes = cache_load(CATEGORY_ARCHETYPES_PATH)
    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))[: args.n]

    print(f"  archetypes: {len(archetypes)}")
    print(f"  triggers library: {sum(len(v) for v in triggers_db.values())} across "
          f"{len(triggers_db)} cats")

    # Group train problems for seed examples
    from task_env.base_env import TaskEnvironment
    kb = {}
    for f in sorted(cfg.KB_DIR.glob("S*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        kb[d["id"]] = d
    env = TaskEnvironment(strategy_kb=kb)
    from collections import defaultdict
    import random
    train_pool = env.get_all_problems("train")
    by_cat = defaultdict(list)
    for p in train_pool:
        by_cat[(p.get("domain", "?"), p.get("difficulty", "?"))].append(p)

    needed_cats = set()
    for p in sample:
        needed_cats.add((p.get("domain", "?"), p.get("difficulty", "?")))

    client = create_client()
    t0 = time.time()

    # Step 1: detect archetypes per category (cache)
    print("\n[Stage 1.5] Detecting archetypes per category...")
    for cat in sorted(needed_cats):
        key = f"{cat[0]}__{cat[1]}"
        if key in cat_archetypes:
            continue
        if cat not in by_cat:
            continue
        rng = random.Random(hash(("arche", cat)) & 0xFFFFFFFF)
        examples = rng.sample(by_cat[cat], min(3, len(by_cat[cat])))
        ids = detect_archetypes_for_category(
            client, cat[0], cat[1], examples, archetypes)
        cat_archetypes[key] = ids
        cache_save(CATEGORY_ARCHETYPES_PATH, cat_archetypes)
        print(f"  {cat}: {ids} -> " +
              ", ".join(archetype_by_id[i]["name"] for i in ids if i in archetype_by_id))

    # Step 2: generate answers with all 3 layers
    print("\n[Stage 2] Generating phase3_archetypes answers...")
    new_count = 0
    hit_count = 0
    for i, p in enumerate(sample):
        pid = p["problem_id"]
        if pid in answers:
            hit_count += 1
            continue
        dom = p.get("domain", "?")
        diff = p.get("difficulty", "?")
        key = f"{dom}__{diff}"
        struct = structures.get(key)
        if struct is None:
            print(f"  [skip] {pid}: no structure for {key}")
            continue
        priors = struct.get("attention_priors", [])
        triggers = select_triggers_for_category(triggers_db, dom, diff)
        arche_ids = cat_archetypes.get(key, [])
        arche_info = [archetype_by_id[aid] for aid in arche_ids if aid in archetype_by_id]
        answers[pid] = execute_phase3(
            client, p.get("description", ""), priors, triggers, arche_info)
        new_count += 1
        if new_count % 10 == 0:
            cache_save(answers_path, answers)
            print(f"  [{args.variant}] {i+1}/{len(sample)} "
                  f"(new={new_count} hit={hit_count}) {time.time()-t0:.0f}s")
    cache_save(answers_path, answers)
    print(f"\n  total: new={new_count} hit={hit_count} {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
