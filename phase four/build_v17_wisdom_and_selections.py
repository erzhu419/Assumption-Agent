"""
Phase 4 v2 Mode B+C combined: conditional paradigm selection (B) +
refined W025 content with object-level bail-out (C).

Rationale: v16_sel_v2 showed Mode A' (unconditional paradigm push) lifts
engineering/sw_eng but HURTS math/hard. Root cause: W025 activates on every
'seemingly ambiguous' problem, including hard math where it adds noise.

Fix (both at content and selection level):
  - C: W025 unpacked gets explicit "if problem is object-level (proof/
        computation/implementation), DO NOT trigger" bail-out.
  - B: Selection prompt distinguishes paradigm-level signals (stakeholder,
        regulation, investment cost) from object-level (proof, exact
        computation, algorithm). Only force W025/W031 when paradigm-level.

Outputs:
  - wisdom_library_v17.json (with refined W025)
  - phase2_v3_selections_v17.json (from new SELECT_PROMPT_V3)
"""

import copy
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
WISDOM_ORIG = CACHE / "wisdom_library.json"
WISDOM_OUT = CACHE / "wisdom_library_v17.json"
SAMPLE = CACHE / "sample_100.json"
SELECTIONS_OUT = CACHE / "phase2_v3_selections_v17.json"


# ---- Part C: refined wisdom entries ----

# W025: add object-level bail-out
W025_REFINED = {
    "aphorism": "范式不只给答案，还规定何为问题",
    "source": "Kuhn, 《科学革命的结构》",
    "signal": "当问题涉及多 stakeholder、监管、投入成本、利益冲突、研究方向，且胜负在范式层而非对象层时触发。纯对象层问题（数学证明、精确计算、算法实现）**不触发**此条。",
    "unpacked_for_llm": "先辨认层次：(a) 若问题有多方评价标准、合规/投入约束、阶段决策——优先审视 \"在哪个棋盘下子 / 什么才算好答案\"；(b) 若问题就是确定输入-输出的对象层（证明、计算、实现），此 wisdom 不适用，不要强行提升层次。context-gated，非 unconditional。",
    "cross_domain_examples": [
        {
            "domain": "business",
            "scenario": "高管讨论效率 vs 韧性：表面讲方案，实际在争'什么算好问题'——此时触发。"
        },
        {
            "domain": "mathematics",
            "scenario": "要求证明具体定理时：问题层次清晰，别提升到'这定理重不重要'——此时 NOT 触发。"
        }
    ],
    "abstraction_check": "触发条件用抽象词（stakeholder、监管、投入、范式/对象层），无 domain 术语。",
    "cluster": "popper_russell_wittgenstein",
    "id": "W025"
}

# W031 similar: add "不要用在纯计算题上"
W031_REFINED = {
    "aphorism": "提好问题，常胜过急着作答。",
    "source": "Polya, 《How to Solve It》",
    "signal": "当问题的表述本身可能误导、或存在更好的重新表述时触发。但若问题已经是清晰的对象层任务（精确计算/证明具体命题），不再重问，直接解。",
    "unpacked_for_llm": "先问：题目的表述本身是否精确？它对不对称、不必要的约束、或把真正的问题隐藏在表面之下？如果问题本身已经清楚（给定输入、要求具体输出），就不要再反问\"真正要解什么\"，直接解。",
    "cross_domain_examples": [
        {
            "domain": "business",
            "scenario": "用户说'加个功能'——先问'真正要解决的用户问题是什么'——此时触发。"
        },
        {
            "domain": "engineering",
            "scenario": "要求'计算梁在 100N 载荷下的挠度'——边界清楚，不必重问，直接算——NOT 触发。"
        }
    ],
    "abstraction_check": "无 domain 术语。",
    "cluster": "polya_heuristics",
    "id": "W031"
}


# ---- Part B: conditional paradigm selection prompt ----

SELECT_PROMPT_V3 = """# 智慧库选择任务

## 问题
{problem}

## 智慧库（共 {n} 条，格式：[ID] 警句 — signal）
{library_brief}

## 你的任务
从 75 条 wisdom entries 中挑选 **3-5 条最能帮助解决当前问题**的。

### 判断标准（按顺序）

**第 1 步 — 判断问题层次**

- **对象层** (Object-level): 有明确输入输出的计算 / 证明 / 实现类
  - 特征：纯数学证明、精确计算、算法实现、已知输入求确定输出
  - **不要选 W025 (范式) / W031 (重新提问) / W066 (换说法)** — 它们会把答案拉离具体计算
  - 应优先选对象层 wisdom：如 W012 (欲速则不达)、W023 (找反例)、W029 (尽量简单)、W050 (磨刀不误砍柴工)

- **范式层** (Paradigm-level): 胜负不在"怎么算"而在"在哪个棋盘下子"
  - 触发条件（出现任一即应考虑 W025/W031）：
    - 多 stakeholder / 利益冲突 / 监管 / 合规 / 审计
    - 投入成本、沉没成本、阶段决策、是否转向
    - "什么才算好答案"有争议 / 目标有冲突
    - 表面像技术题，但评判标准由商业/合规/战略定义

**第 2 步 — signal 结构对齐**

signal 描述的情境与当前问题的结构对得上（不是字面对得上）。

**第 3 步 — 激活后能使回答真的不同**

不只是装饰，而是改变分析角度或结论。

### 输出 JSON（不要代码块）
{{"selected_ids": ["W00X", "W0XX", ...], "layer": "object | paradigm", "reason": "一句话"}}
"""


def build_brief_library(library):
    return "\n".join(
        f"[{e['id']}] {e['aphorism']} — {e.get('signal','')[:60]}"
        for e in library
    )


def select_entries(client, problem, library):
    brief = build_brief_library(library)
    resp = _generate_with_retry(client, SELECT_PROMPT_V3.format(
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
    # Part C: build refined wisdom library
    orig_lib = json.loads(WISDOM_ORIG.read_text(encoding="utf-8"))
    refined_lib = []
    refined_count = 0
    for e in orig_lib:
        if e["id"] == "W025":
            refined_lib.append(W025_REFINED)
            refined_count += 1
        elif e["id"] == "W031":
            refined_lib.append(W031_REFINED)
            refined_count += 1
        else:
            refined_lib.append(e)
    WISDOM_OUT.write_text(json.dumps(refined_lib, ensure_ascii=False, indent=2))
    print(f"Part C: refined {refined_count} wisdom entries (W025, W031) → {WISDOM_OUT.name}")

    # Part B: re-run selection with refined library + conditional prompt
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    out = {}
    if SELECTIONS_OUT.exists():
        try:
            out = json.loads(SELECTIONS_OUT.read_text(encoding="utf-8"))
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
            sel_ids = select_entries(client, desc, refined_lib)
            out[pid] = sel_ids
            new_count += 1
        except Exception as e:
            print(f"  [err {pid}] {e}")
            errors += 1
            continue

        if new_count % 10 == 0:
            SELECTIONS_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
            print(f"  [{pid}] {new_count}/{len(sample)} done, {time.time()-t0:.0f}s")

    SELECTIONS_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n=== Selection Summary ===")
    print(f"  {len(out)}/{len(sample)}, errors={errors}, time={time.time()-t0:.0f}s")

    # Compare with v1 and v2
    from collections import Counter
    v1 = json.loads((CACHE / "phase2_v3_selections.json").read_text(encoding="utf-8"))
    v2 = json.loads((CACHE / "phase2_v3_selections_v2.json").read_text(encoding="utf-8"))
    residuals = json.loads((PROJECT / "phase four" / "residuals" / "v16_residuals.json").read_text(encoding="utf-8"))

    # Compare W025 top-2 hit rates across residual problems (domain-split)
    pid_to_info = {p["problem_id"]: p for p in sample}
    target_ws = ["W025", "W031"]
    for target_w in target_ws:
        rp = [pid for pid, x in residuals.items() if x.get("nearest_existing_wisdom") == target_w]
        print(f"\n{target_w} residuals ({len(rp)} total):")
        print(f"  {'domain':<22} {'count':>6} {'v1top2':>8} {'v2top2':>8} {'v3top2':>8}")
        dom_split = Counter((pid_to_info[p]["domain"] for p in rp if p in pid_to_info))
        for dom in sorted(dom_split):
            dom_pids = [p for p in rp if pid_to_info.get(p, {}).get("domain") == dom]
            v1h = sum(1 for p in dom_pids if target_w in v1.get(p, [])[:2])
            v2h = sum(1 for p in dom_pids if target_w in v2.get(p, [])[:2])
            v3h = sum(1 for p in dom_pids if target_w in out.get(p, [])[:2])
            print(f"  {dom:<22} {len(dom_pids):>6} {v1h:>8} {v2h:>8} {v3h:>8}")

    # Also check: did v3 correctly AVOID W025 on object-level math problems?
    print(f"\n=== Object-level avoidance check ===")
    # Count W025 top-2 hits on non-residual math problems (should be low/zero)
    residual_pids = set(residuals.keys())
    math_non_residual = [p["problem_id"] for p in sample
                          if p.get("domain") == "mathematics" and p["problem_id"] not in residual_pids]
    w025_v1 = sum(1 for p in math_non_residual if "W025" in v1.get(p, [])[:2])
    w025_v2 = sum(1 for p in math_non_residual if "W025" in v2.get(p, [])[:2])
    w025_v3 = sum(1 for p in math_non_residual if "W025" in out.get(p, [])[:2])
    print(f"  math non-residual ({len(math_non_residual)} problems) — W025 in top-2:")
    print(f"    v1: {w025_v1}, v2: {w025_v2}, v3: {w025_v3}")
    print(f"  (hope: v3 ≤ v1, meaning v3 avoids forcing W025 on pure math)")


if __name__ == "__main__":
    main()
