"""
Phase 4 v2 Mode A' Step 1: re-run wisdom selection with improved prompt.

Diagnosis (from residual_analyzer): 7 high-frequency "should-have-applied"
wisdoms (W025, W031, W064, W061, W066, W050, W023) hit top-2 in only 1/43
residual problems. The wisdom content is fine; the SELECT prompt is missing
paradigm-framing signal recognition.

This script re-runs v3's select_entries per problem with an improved prompt
that explicitly flags paradigm/framing/stakeholder cues. Output saved to
`phase2_v3_selections_v2.json` (doesn't overwrite original).

Then can compare hit rates on residuals and decide whether to regen v16.
"""

import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase one" / "scripts" / "validation"))

from llm_client import create_client, parse_json_from_llm
from cached_framework import _generate_with_retry


CACHE = PROJECT / "phase two" / "analysis" / "cache"
WISDOM_PATH = CACHE / "wisdom_library.json"
SAMPLE = CACHE / "sample_100.json"
OUT_PATH = CACHE / "phase2_v3_selections_v2.json"


# Improved SELECT_PROMPT with explicit paradigm-framing recognition
SELECT_PROMPT_V2 = """# 智慧库选择任务

## 问题
{problem}

## 智慧库（共 {n} 条，格式：[ID] 警句 — signal）
{library_brief}

## 你的任务
从 75 条 wisdom entries 中挑选 **3-5 条最能帮助解决当前问题**的。

### 判断标准（按重要度排序）

**1. 范式识别优先**：问题表面看像 A 类（如"算法选择"/"性能优化"/"代码重构"），实际核心是 B 类（如"在什么棋盘上下子"/"什么才算好答案"/"谁来定义问题"）。
   - 凡问题涉及 **多 stakeholder、监管、投入成本、利益冲突、研究方向、范式选型** 任何一项，
   - **W025 (范式不只给答案，还规定何为问题)** 和 **W031 (提好问题，常胜过急着作答)** 几乎必选。
   - 看似技术题而实为策略题时，**W064 (若终点已到，来路便会开口)** 也常适用。

**2. 信号结构对齐**：signal 描述的情境**与当前问题的结构对得上**（不是字面对得上）。

**3. 激活后能使回答不同**：它的 orientation 如果被激活，**能让回答真的不同**（不只是装饰）。

**4. 不要选字面贴合但实际不适用的**。

### 输出 JSON（不要代码块）
{{"selected_ids": ["W00X", "W0XX", ...], "reason": "一句话说为什么这几条"}}
"""


def build_brief_library(library):
    return "\n".join(
        f"[{e['id']}] {e['aphorism']} — {e.get('signal','')[:60]}"
        for e in library
    )


def select_entries(client, problem, library):
    brief = build_brief_library(library)
    resp = _generate_with_retry(client, SELECT_PROMPT_V2.format(
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


def main():
    library = json.loads(WISDOM_PATH.read_text(encoding="utf-8"))
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))

    # Resume
    out = {}
    if OUT_PATH.exists():
        try:
            out = json.loads(OUT_PATH.read_text(encoding="utf-8"))
            print(f"Resuming: {len(out)} cached")
        except Exception:
            out = {}

    client = create_client()
    t0 = time.time()
    new_count = 0
    errors = 0

    for i, p in enumerate(sample):
        pid = p["problem_id"]
        if pid in out:
            continue
        desc = p.get("description", "")
        try:
            sel_ids = select_entries(client, desc, library)
            out[pid] = sel_ids
            new_count += 1
        except Exception as e:
            print(f"  [err {pid}] {e}")
            errors += 1
            continue

        if new_count % 10 == 0:
            OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
            print(f"  [{pid}] {new_count}/{len(sample)} done, {time.time()-t0:.0f}s")

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n=== Summary ===")
    print(f"  Selections: {len(out)}/{len(sample)}, errors: {errors}, time: {time.time()-t0:.0f}s")

    # Compare with original
    orig = json.loads((CACHE / "phase2_v3_selections.json").read_text(encoding="utf-8"))
    residuals = json.loads((PROJECT / "phase four" / "residuals" / "v16_residuals.json").read_text(encoding="utf-8"))

    print("\n=== Residual top-2 hit comparison (v1 → v2 selection) ===")
    print(f"{'wisdom':<8} {'residuals':>9} {'v1 top2':>8} {'v2 top2':>8} {'Δ':>5}")
    for target_w in ['W025', 'W031', 'W064', 'W061', 'W066', 'W050', 'W023']:
        pids = [pid for pid, x in residuals.items() if x.get('nearest_existing_wisdom') == target_w]
        v1_hit = sum(1 for pid in pids if target_w in orig.get(pid, [])[:2])
        v2_hit = sum(1 for pid in pids if target_w in out.get(pid, [])[:2])
        delta = v2_hit - v1_hit
        print(f"{target_w:<8} {len(pids):>9} {v1_hit:>8} {v2_hit:>8} {delta:>+5}")


if __name__ == "__main__":
    main()
