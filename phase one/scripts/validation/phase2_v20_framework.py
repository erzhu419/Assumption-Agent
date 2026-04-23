"""
v20: Combined v19a (frame directive) + v19b (problem rewrite).

Hypothesis: v19a (+10pp) and v19b (+14pp) both won by foregrounding frame
decision. If we do BOTH — rewrite the problem AND place frame as PRIMARY
directive — the effects should compose.

Architecture:
  Turn 0 (frame + rewrite combined):
    Produce JSON with {frame, critical_reframe, anti_patterns, evaluation_criteria,
                        rewritten_problem, what_changed}
  Turn 1 (solve):
    PRIMARY FRAME block (v19a-style) + rewritten problem (v19b-style)
    + secondary wisdom/cases
  Turn 2 (audit):
    Check both frame adherence and whether solved rewritten vs original

Math/sci bypass to v12c hygiene.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

import _config as cfg  # noqa: E402
from llm_client import create_client, parse_json_from_llm  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from cached_framework import _generate_with_retry
from phase2_v12_framework import (EXECUTE_MATH, EXECUTE_SCIENCE, format_priors,
                                   MATH_SCI)
from phase2_v15_exemplar_framework import (build_same_domain_exemplar,
                                            format_wisdom_with_cases)


CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE / "answers"
STRUCTURES_DIR = CACHE / "structures"
WISDOM_PATH_DEFAULT = CACHE / "wisdom_library.json"
SELECTIONS_PATH_DEFAULT = CACHE / "phase2_v3_selections.json"
EXEMPLARS_PATH = CACHE / "wisdom_diverse_exemplars.json"
EMB_PATH = CACHE / "signal_embeddings.npz"
V13_REFLECT = ANSWERS_DIR / "phase2_v13_reflect_answers.json"
OURS_27 = ANSWERS_DIR / "ours_27_answers.json"


FRAME_REWRITE_PROMPT = """# 问题诊断 + 重写任务

读下面问题，**同时** 做两件事：
1. 诊断问题的 frame（对象层 / 范式层 / 混合）及 critical reframe
2. 把问题重写成一个更精确、显式化了真正该回答维度的版本

## 原始问题
{problem}

## 输出 JSON（不要代码块）
{{
  "frame": "object" | "paradigm" | "hybrid",
  "critical_reframe": "如果误读会被误读成什么？真正该解的问题说清楚 (30-60 字)",
  "anti_patterns": ["要避免的常见错误做法 1", "2", ...],
  "evaluation_criteria": "什么样答案算好 (30-50 字)",
  "rewritten_problem": "重写版本（完整段落，可比原文稍长，显式化 stakeholder/criteria/constraints）",
  "what_changed": "相比原文凸显了什么 (20-50 字)"
}}

## 指导
- **对象层** (object): 纯计算/证明/实现，保留原问题为重写版，rewritten ≈ original
- **范式层** (paradigm): 显式化 stakeholder / 监管 / 投入 / criteria，rewrite 可大改
- **hybrid**: 两层都显式化
"""


EXECUTE_V20 = """# 你要解决下面的问题。

## ⚡ PRIMARY FRAME (顶层 directive，必须遵守)

**问题类型**: {frame}
**Critical reframe**: {critical_reframe}
**评判标准**: {evaluation_criteria}
**要避免的反模式**:
{anti_patterns_block}

---

## 📝 你要解决的问题（已经被诊断并重写，显式化真正该回答的维度）

### 原始问题
{problem}

### 重写版本（基于上面 frame 的显式化）
{rewritten_problem}

*重写点评*: {what_changed}

**针对重写版本作答**；若原问题本身清晰（object-level），重写 ≈ 原文，直接按原文答。

---

## 次要参考（仅在与 PRIMARY FRAME 一致时纳入）

### 本类别 attention priors
{priors_block}

### Wisdom + 判例集
{wisdom_case_block}

---

## 要求
- 完全服从 PRIMARY FRAME 与重写版本的 framing
- 次要参考是补充，不是主体
- 不超过 600 字

开始：
"""


REFLECT_PROMPT_V20 = """草稿已产。

## 原问题
{problem}

## Primary Frame (Turn 0 已定)
{frame_summary}

## 重写版本
{rewritten_problem}

## 草稿
{draft}

## 自检
1. 草稿是否按 PRIMARY FRAME 作答？滑回了错误 frame 吗？
2. 草稿是针对 original 还是 rewritten？如果重写凸显了什么但 draft 没 address，补上。
3. 踩了 anti_patterns 里的某条吗？

输出最终答案（不要 audit 过程）。不超过 650 字。
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


def format_anti_patterns(aps):
    if not aps:
        return "  (无)"
    return "\n".join(f"  - {ap}" for ap in aps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="phase2_v20")
    ap.add_argument("--base", default="orient_hybrid")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-wisdoms", type=int, default=2)
    ap.add_argument("--sample", default="sample_100.json")
    ap.add_argument("--selections", default=None,
                    help="override selections file (e.g. phase2_v3_selections_v2.json)")
    ap.add_argument("--wisdom", default=None,
                    help="override wisdom library file")
    args = ap.parse_args()

    selections_path = (CACHE / args.selections) if args.selections else SELECTIONS_PATH_DEFAULT
    wisdom_path = (CACHE / args.wisdom) if args.wisdom else WISDOM_PATH_DEFAULT

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    meta_path = ANSWERS_DIR / f"{args.variant}_meta.json"  # frame + rewrite combined
    drafts_path = ANSWERS_DIR / f"{args.variant}_drafts.json"
    struct_path = STRUCTURES_DIR / f"{args.variant}_structures.json"
    answers = cache_load(answers_path)
    meta = cache_load(meta_path)
    drafts = cache_load(drafts_path)

    if not struct_path.exists():
        base_struct = cache_load(STRUCTURES_DIR / f"{args.base}_structures.json")
        if base_struct:
            cache_save(struct_path, base_struct)
    structures = cache_load(struct_path)

    library = json.loads(wisdom_path.read_text(encoding="utf-8"))
    lib_by_id = {e["id"]: e for e in library}
    selections = cache_load(selections_path)
    diverse_exs = cache_load(EXEMPLARS_PATH)
    v13 = json.loads(V13_REFLECT.read_text(encoding="utf-8"))
    ours = json.loads(OURS_27.read_text(encoding="utf-8"))

    emb = np.load(EMB_PATH, allow_pickle=True)
    prob_emb = emb["problem_emb"]
    prob_ids_emb = emb["problem_ids"].tolist()
    pid_to_emb_idx = {pid: i for i, pid in enumerate(prob_ids_emb)}

    sample = json.loads((CACHE / args.sample).read_text(encoding="utf-8"))[: args.n]

    client = create_client()
    t0 = time.time()
    new = hit = hyg = full = 0

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
                print(f"  [err {pid}] {e}")
                continue
            hyg += 1
        else:
            # Turn 0: combined frame + rewrite
            if pid in meta:
                m = meta[pid]
            else:
                try:
                    r0 = _generate_with_retry(client, FRAME_REWRITE_PROMPT.format(problem=problem),
                                              max_tokens=700, temperature=0.2)
                    m = parse_json_from_llm(r0["text"])
                    required = {"frame", "critical_reframe", "anti_patterns",
                                "evaluation_criteria", "rewritten_problem", "what_changed"}
                    if not required.issubset(m.keys()):
                        raise ValueError(f"missing: {required - m.keys()}")
                    meta[pid] = m
                except Exception as e:
                    print(f"  [err meta {pid}] {e}")
                    m = {"frame": "hybrid", "critical_reframe": "", "anti_patterns": [],
                         "evaluation_criteria": "", "rewritten_problem": problem,
                         "what_changed": "fallback"}
                    meta[pid] = m

            key = f"{dom}__{diff}"
            struct = structures.get(key, {"attention_priors": []})
            priors = struct.get("attention_priors", [])

            sel_ids = selections.get(pid, [])[: args.max_wisdoms]
            wisdom_entries = [lib_by_id[sid] for sid in sel_ids if sid in lib_by_id]
            wisdom_entries = [w for w in wisdom_entries if w["id"] in diverse_exs]

            w_blocks = []
            for w in wisdom_entries:
                pre_mined = diverse_exs.get(w["id"], [])
                same_dom_ex = build_same_domain_exemplar(
                    pid, dom, sample, prob_emb, pid_to_emb_idx, v13, ours)
                cross = [e for e in pre_mined if e.get("domain") != dom][:3]
                if len(cross) < 3:
                    cross += [e for e in pre_mined if e not in cross][:3 - len(cross)]
                w_blocks.append(format_wisdom_with_cases(w, cross, same_dom_ex))
            wisdom_case_block = "\n\n---\n\n".join(w_blocks) if w_blocks else "  (无)"

            if pid in drafts:
                draft = drafts[pid]
            else:
                try:
                    r1 = _generate_with_retry(client, EXECUTE_V20.format(
                        frame=m.get("frame", "hybrid"),
                        critical_reframe=m.get("critical_reframe", ""),
                        evaluation_criteria=m.get("evaluation_criteria", ""),
                        anti_patterns_block=format_anti_patterns(m.get("anti_patterns", [])),
                        problem=problem,
                        rewritten_problem=m.get("rewritten_problem", problem),
                        what_changed=m.get("what_changed", ""),
                        priors_block=format_priors(priors),
                        wisdom_case_block=wisdom_case_block,
                    ), max_tokens=1100, temperature=0.3)
                    draft = r1["text"].strip()
                    drafts[pid] = draft
                except Exception as e:
                    print(f"  [err draft {pid}] {e}")
                    continue

            # Turn 2 audit
            frame_summary = f"{m.get('frame', '?')}: {m.get('critical_reframe', '')}"
            try:
                r2 = _generate_with_retry(client, REFLECT_PROMPT_V20.format(
                    problem=problem, frame_summary=frame_summary,
                    rewritten_problem=m.get("rewritten_problem", problem),
                    draft=draft
                ), max_tokens=1100, temperature=0.3)
                answers[pid] = r2["text"].strip()
            except Exception as e:
                print(f"  [err audit {pid}] {e}")
                continue
            full += 1

        new += 1
        if new % 10 == 0:
            cache_save(answers_path, answers)
            cache_save(meta_path, meta)
            cache_save(drafts_path, drafts)
            print(f"  [{args.variant}] {i+1}/{len(sample)} hyg={hyg} full={full} {time.time()-t0:.0f}s")

    cache_save(answers_path, answers)
    cache_save(meta_path, meta)
    cache_save(drafts_path, drafts)
    print(f"\n  [{args.variant}] done: hyg={hyg} full={full} ({time.time()-t0:.0f}s)")

    from collections import Counter
    fc = Counter(x.get("frame", "?") for x in meta.values())
    print(f"  Frame distribution: {dict(fc)}")


if __name__ == "__main__":
    main()
