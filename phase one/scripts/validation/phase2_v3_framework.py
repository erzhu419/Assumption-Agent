"""
Phase 2 v3 — Wisdom Library (75 cross-civ entries) + per-problem selection.

Pipeline:
  Stage A (per-problem selection, Gemini Flash):
    Given problem + compact list of 75 aphorisms (with signals),
    pick 3-5 most relevant.
  Stage B (execute, Gemini Flash):
    Inject PRE-UNPACKED version of selected entries as priors.
    Aphorism + source shown as soft attribution (not forced citation).

Addresses Phase 2 v1/v2 shortcomings:
  - Wider corpus (not just our losses)
  - LLM-consumable unpacked form (not raw aphorism)
  - Soft reference (not forced citation)
  - Dynamic per-problem selection (not fixed category retrieval)
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


CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
WISDOM_PATH = CACHE / "wisdom_library.json"
ANSWERS_DIR = CACHE / "answers"
SELECTIONS_PATH = CACHE / "phase2_v3_selections.json"


SELECT_PROMPT = """# 智慧库选择任务

## 问题
{problem}

## 智慧库（共 {n} 条，格式：[ID] 警句 — signal）
{library_brief}

## 你的任务
从 75 条 wisdom entries 中挑选 **3-5 条最能帮助解决当前问题**的。
判断标准：
- signal 描述的情境**与当前问题的结构对得上**（不是字面对得上）
- 它的 orientation 如果被激活，**能让回答真的不同**（不只是装饰）
- 不要选字面贴合但实际不适用的

输出 JSON（不要代码块）：
{{"selected_ids": ["W00X", "W0XX", ...], "reason": "一句话说为什么这几条"}}
"""


EXECUTE_PROMPT = """# 你要解决下面的问题。

## 思维背景（可参考，不强制引用）

以下是几条可能适用的 orientation，已为你 pre-unpacked 成可直接自问的形式。
**读问题时让它们在脑中浮现，但不要为引用而引用**。

{priors_block}

## 问题
{problem}

## 要求
- **不要**用 Step 1/2 格式，带着上面的 orientation 直接思考和作答
- 答案本身是重点；引用某条 aphorism 仅在它**真的塑形了你的思考**时再提（结尾附一句即可，不是必须）
- 语言精炼，不超过 500 字
- 如果没有一条 orientation 真的 fire，就直接答题，不要勉强

开始：
"""


def cache_load(p: Path):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def cache_save(p: Path, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def build_brief_library(library: List[Dict]) -> str:
    """Compact one-liner per entry: [ID] aphorism — signal."""
    return "\n".join(
        f"[{e['id']}] {e['aphorism']} — {e.get('signal','')[:60]}"
        for e in library
    )


def format_priors(selected_entries: List[Dict]) -> str:
    """Show selected entries with full unpacked content."""
    parts = []
    for e in selected_entries:
        parts.append(
            f"• 【{e['id']} — {e['source']}】《{e['aphorism']}》\n"
            f"    {e['unpacked_for_llm']}"
        )
    return "\n\n".join(parts)


def select_entries(client, problem: str, library: List[Dict]) -> List[str]:
    brief = build_brief_library(library)
    resp = _generate_with_retry(client, SELECT_PROMPT.format(
        problem=problem[:600], library_brief=brief, n=len(library)),
        max_tokens=400, temperature=0.2)
    try:
        parsed = parse_json_from_llm(resp["text"])
        ids = parsed.get("selected_ids", [])
        valid_ids = {e["id"] for e in library}
        return [s for s in ids if isinstance(s, str) and s in valid_ids][:5]
    except Exception:
        import re
        found = re.findall(r"W\d{3}", resp.get("text", ""))
        valid_ids = {e["id"] for e in library}
        return [s for s in found if s in valid_ids][:5]


def execute_with_wisdom(client, problem: str, selected_entries: List[Dict]) -> str:
    priors_block = format_priors(selected_entries) if selected_entries else "  (本题没有特别适用的 orientation)"
    resp = _generate_with_retry(client, EXECUTE_PROMPT.format(
        priors_block=priors_block, problem=problem),
        max_tokens=900, temperature=0.3)
    return resp["text"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="phase2_v3_wisdom")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    library = json.loads(WISDOM_PATH.read_text(encoding="utf-8"))
    lib_by_id = {e["id"]: e for e in library}
    print(f"Wisdom library: {len(library)} entries")

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    answers = cache_load(answers_path)
    selections = cache_load(SELECTIONS_PATH)

    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))[: args.n]

    client = create_client()
    t0 = time.time()
    new_count = 0
    hit_count = 0

    for i, p in enumerate(sample):
        pid = p["problem_id"]
        if pid in answers:
            hit_count += 1
            continue
        desc = p.get("description", "")

        # Stage A: select (cache selections for reuse)
        if pid in selections:
            sel_ids = selections[pid]
        else:
            try:
                sel_ids = select_entries(client, desc, library)
            except Exception as e:
                print(f"  [select-error] {pid}: {e}")
                continue
            selections[pid] = sel_ids
            if len(selections) % 10 == 0:
                cache_save(SELECTIONS_PATH, selections)

        # Stage B: execute with pre-unpacked priors
        selected_entries = [lib_by_id[sid] for sid in sel_ids if sid in lib_by_id]
        try:
            answers[pid] = execute_with_wisdom(client, desc, selected_entries)
        except Exception as e:
            print(f"  [execute-error] {pid}: {e}")
            continue
        new_count += 1
        if new_count % 10 == 0:
            cache_save(answers_path, answers)
            cache_save(SELECTIONS_PATH, selections)
            print(f"  [{args.variant}] {i+1}/{len(sample)} "
                  f"(new={new_count} hit={hit_count}) {time.time()-t0:.0f}s "
                  f"last-selected={sel_ids}")

    cache_save(answers_path, answers)
    cache_save(SELECTIONS_PATH, selections)
    print(f"\n  [{args.variant}] total: new={new_count} hit={hit_count} "
          f"{time.time()-t0:.0f}s")

    # Summary: which entries got picked most
    from collections import Counter
    picks = Counter()
    for sel_ids in selections.values():
        for sid in sel_ids:
            picks[sid] += 1
    print("\n  Top 10 most-selected wisdom entries:")
    for sid, n in picks.most_common(10):
        if sid in lib_by_id:
            print(f"    [{sid}] picked {n}x: {lib_by_id[sid]['aphorism']}")


if __name__ == "__main__":
    main()
