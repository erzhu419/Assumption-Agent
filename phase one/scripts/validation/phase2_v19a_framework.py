"""
v19a: Frame-First Architecture (Turn 0 frame decision → foregrounded directive).

Breaking v16 ceiling by changing attention hierarchy:
- Turn 0: LLM judges problem frame (object/paradigm/hybrid) + reframe + anti-patterns
- Turn 1: Frame output placed in TOP section as "PRIMARY FRAME (必须遵守)",
         wisdom/cases in SECONDARY section
- Turn 2: audit (same as v16)

Design principle (from user): "识别结果不应和案例平权塞入，应该支配后续推理"
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
                                   format_wisdom, MATH_SCI)
from phase2_v15_exemplar_framework import (build_same_domain_exemplar,
                                            format_wisdom_with_cases)


CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE / "answers"
STRUCTURES_DIR = CACHE / "structures"
WISDOM_PATH = CACHE / "wisdom_library.json"
SELECTIONS_PATH = CACHE / "phase2_v3_selections.json"
EXEMPLARS_PATH = CACHE / "wisdom_diverse_exemplars.json"
EMB_PATH = CACHE / "signal_embeddings.npz"
V13_REFLECT = ANSWERS_DIR / "phase2_v13_reflect_answers.json"
OURS_27 = ANSWERS_DIR / "ours_27_answers.json"


FRAME_PROMPT = """# 问题层次诊断

你是问题诊断专家。读下面这个问题，判断它真正的类型与 frame：

## 问题
{problem}

## 诊断任务
输出 JSON（不要代码块）：
{{
  "frame": "object" | "paradigm" | "hybrid",
  "frame_reasoning": "为什么是这个 frame (20-40 字)",
  "critical_reframe": "如果这个问题被误读，会被误读成什么？把真正该解的问题说清楚 (30-60 字)",
  "anti_patterns": ["要避免的常见错误做法 1", "要避免的 2", "..."],
  "evaluation_criteria": "什么样的答案才算 \"好\"（从 problem 的语境推断，30-50 字）"
}}

**判断标准**：
- object-level: 有明确输入输出的计算/证明/实现（唯一或少数正确答案）
- paradigm-level: 胜负在 "在哪个棋盘下子 / 什么才算好答案"（多 stakeholder / 监管 / 投入 / 范式冲突）
- hybrid: 有明确对象层要求但 frame 选择也关键
"""


EXECUTE_V19A = """# 你要解决下面的问题。

## ⚡ PRIMARY FRAME (这是顶层 directive，必须遵守)

**问题类型**: {frame}
**Frame 推理**: {frame_reasoning}
**关键 reframe**: {critical_reframe}
**评判标准**: {evaluation_criteria}
**要避免的反模式**:
{anti_patterns_block}

---

## 次要参考（仅当与上面的 frame 一致时考虑）

### 本类别 attention priors
{priors_block}

### Wisdom + 判例集
{wisdom_case_block}

---

## 问题
{problem}

## 要求
- **必须在 PRIMARY FRAME 指定的评判标准下作答**
- 次要参考只是补充，不是主体
- 不超过 600 字

开始：
"""


REFLECT_PROMPT_V19A = """你刚给出了一个草稿答案。

## 问题
{problem}

## Primary Frame (Turn 0 已定)
{frame_summary}

## 你的草稿
{draft}

## 自检任务
1. 草稿是否真正按 Primary Frame 作答？还是滑回了错误的 frame？
2. 是否踩到 anti_patterns 里的任一条？
3. 有没有 1-2 个 critical_reframe 暗示的角度没被展开？

输出最终答案（不要列出 audit 过程），不超过 650 字：
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
    ap.add_argument("--variant", default="phase2_v19a")
    ap.add_argument("--base", default="orient_hybrid")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-wisdoms", type=int, default=2)
    args = ap.parse_args()

    answers_path = ANSWERS_DIR / f"{args.variant}_answers.json"
    frames_path = ANSWERS_DIR / f"{args.variant}_frames.json"
    drafts_path = ANSWERS_DIR / f"{args.variant}_drafts.json"
    struct_path = STRUCTURES_DIR / f"{args.variant}_structures.json"
    answers = cache_load(answers_path)
    frames = cache_load(frames_path)
    drafts = cache_load(drafts_path)

    if not struct_path.exists():
        base_struct = cache_load(STRUCTURES_DIR / f"{args.base}_structures.json")
        if base_struct:
            cache_save(struct_path, base_struct)
    structures = cache_load(struct_path)

    library = json.loads(WISDOM_PATH.read_text(encoding="utf-8"))
    lib_by_id = {e["id"]: e for e in library}
    selections = cache_load(SELECTIONS_PATH)
    diverse_exs = cache_load(EXEMPLARS_PATH)
    v13 = json.loads(V13_REFLECT.read_text(encoding="utf-8"))
    ours = json.loads(OURS_27.read_text(encoding="utf-8"))

    emb = np.load(EMB_PATH, allow_pickle=True)
    prob_emb = emb["problem_emb"]
    prob_ids_emb = emb["problem_ids"].tolist()
    pid_to_emb_idx = {pid: i for i, pid in enumerate(prob_ids_emb)}

    sample = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))[: args.n]

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

        # math/sci bypass
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
            # Turn 0: frame decision
            if pid in frames:
                frame_data = frames[pid]
            else:
                try:
                    r0 = _generate_with_retry(client, FRAME_PROMPT.format(problem=problem),
                                              max_tokens=500, temperature=0.2)
                    frame_data = parse_json_from_llm(r0["text"])
                    # Validate shape
                    required = {"frame", "frame_reasoning", "critical_reframe",
                                "anti_patterns", "evaluation_criteria"}
                    if not required.issubset(frame_data.keys()):
                        raise ValueError(f"missing fields: {required - frame_data.keys()}")
                    frames[pid] = frame_data
                except Exception as e:
                    print(f"  [err frame {pid}] {e}")
                    # Fallback to generic
                    frame_data = {
                        "frame": "hybrid",
                        "frame_reasoning": "fallback",
                        "critical_reframe": "",
                        "anti_patterns": [],
                        "evaluation_criteria": "",
                    }
                    frames[pid] = frame_data

            # Turn 1: solve with frame as top directive
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
                    r1 = _generate_with_retry(client, EXECUTE_V19A.format(
                        frame=frame_data.get("frame", "hybrid"),
                        frame_reasoning=frame_data.get("frame_reasoning", ""),
                        critical_reframe=frame_data.get("critical_reframe", ""),
                        evaluation_criteria=frame_data.get("evaluation_criteria", ""),
                        anti_patterns_block=format_anti_patterns(frame_data.get("anti_patterns", [])),
                        priors_block=format_priors(priors),
                        wisdom_case_block=wisdom_case_block,
                        problem=problem,
                    ), max_tokens=1000, temperature=0.3)
                    draft = r1["text"].strip()
                    drafts[pid] = draft
                except Exception as e:
                    print(f"  [err draft {pid}] {e}")
                    continue

            # Turn 2: audit (w/ frame reminder)
            frame_summary = f"{frame_data.get('frame', '?')}: {frame_data.get('critical_reframe', '')}"
            try:
                r2 = _generate_with_retry(client, REFLECT_PROMPT_V19A.format(
                    problem=problem, frame_summary=frame_summary, draft=draft
                ), max_tokens=1100, temperature=0.3)
                answers[pid] = r2["text"].strip()
            except Exception as e:
                print(f"  [err audit {pid}] {e}")
                continue
            full += 1

        new += 1
        if new % 10 == 0:
            cache_save(answers_path, answers)
            cache_save(frames_path, frames)
            cache_save(drafts_path, drafts)
            print(f"  [{args.variant}] {i+1}/{len(sample)} hyg={hyg} full={full} {time.time()-t0:.0f}s")

    cache_save(answers_path, answers)
    cache_save(frames_path, frames)
    cache_save(drafts_path, drafts)
    print(f"\n  [{args.variant}] done: hyg={hyg} full={full} ({time.time()-t0:.0f}s)")

    # Frame distribution
    from collections import Counter
    fc = Counter(f.get("frame", "?") for f in frames.values())
    print(f"  Frame distribution: {dict(fc)}")


if __name__ == "__main__":
    main()
