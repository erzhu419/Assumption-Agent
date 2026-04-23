"""
v19d: Frame-triggered Persona Switch.

Turn 0: detect frame (same as v19a)
Turn 1: based on frame, use DIFFERENT persona/system prompt
  - object: 技术专家 persona (direct computation/implementation focus)
  - paradigm: 战略顾问 persona (stakeholder + criteria analysis focus)
  - hybrid: 融合 persona (balance both)
Turn 2: audit (same as v16)

Hypothesis: changing LLM's ROLE via persona prompt flips its whole output mode,
much stronger than adding content hints. Frame decision drives persona, wisdoms
are secondary.
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
from phase2_v19a_framework import FRAME_PROMPT


CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
ANSWERS_DIR = CACHE / "answers"
STRUCTURES_DIR = CACHE / "structures"
WISDOM_PATH = CACHE / "wisdom_library.json"
SELECTIONS_PATH = CACHE / "phase2_v3_selections.json"
EXEMPLARS_PATH = CACHE / "wisdom_diverse_exemplars.json"
EMB_PATH = CACHE / "signal_embeddings.npz"
V13_REFLECT = ANSWERS_DIR / "phase2_v13_reflect_answers.json"
OURS_27 = ANSWERS_DIR / "ours_27_answers.json"


PERSONA_OBJECT = """你是一名**精通领域细节的技术执行者**。你的风格：
- 直接输出计算结果、算法步骤、代码实现、公式推导
- 不反问"真正要解什么"；遇到明确要求就直接开做
- 优先精确、完整、可操作
- 避免空洞的战略建议、利益相关者分析、stakeholder 语言"""

PERSONA_PARADIGM = """你是一名**资深战略顾问 + 利益相关者分析师**。你的风格：
- 先问"谁来评判这个答案的好坏 + 按什么标准"
- 识别问题背后的 multi-stakeholder 冲突、监管/合规、投入代价
- 给出方案时明确"在哪个棋盘下子"、"什么算好答案"
- 避免陷入纯技术细节 — 先让 frame 清楚，技术方案在 frame 下展开"""

PERSONA_HYBRID = """你是一名**架构师级综合思考者**。你同时兼顾：
- 对象层：精确的技术答案（计算/实现/验证）
- 范式层：stakeholder、评判标准、真正要优化什么
- 在单一回答里展示两层联系，不让任何一层被吞并"""


EXECUTE_V19D = """{persona}

---

## 本类别 attention priors
{priors_block}

## Wisdom + 判例集
{wisdom_case_block}

## 问题
{problem}

## 要求
- 按上面 persona 的风格作答
- 不超过 600 字

开始：
"""


REFLECT_PROMPT_V19D = """你刚给出了一个草稿答案。原始 persona 是: {persona_label}.

## 问题
{problem}

## 草稿
{draft}

## 自检
1. 草稿是否真正符合 persona 风格？还是回到了中性/通用模式？
2. 有没有 1-2 个 persona 本该强调但 draft 里弱化的角度？

输出最终答案（不要 audit 过程），不超过 650 字：
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
    ap.add_argument("--variant", default="phase2_v19d")
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

    persona_map = {"object": PERSONA_OBJECT, "paradigm": PERSONA_PARADIGM, "hybrid": PERSONA_HYBRID}

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
            # Turn 0: frame decision
            if pid in frames:
                frame_data = frames[pid]
            else:
                try:
                    r0 = _generate_with_retry(client, FRAME_PROMPT.format(problem=problem),
                                              max_tokens=500, temperature=0.2)
                    frame_data = parse_json_from_llm(r0["text"])
                    if "frame" not in frame_data:
                        raise ValueError("no frame field")
                    frames[pid] = frame_data
                except Exception as e:
                    print(f"  [err frame {pid}] {e}")
                    frame_data = {"frame": "hybrid"}
                    frames[pid] = frame_data

            frame = frame_data.get("frame", "hybrid")
            if frame not in persona_map:
                frame = "hybrid"
            persona = persona_map[frame]

            # Turn 1: solve with persona
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
                    r1 = _generate_with_retry(client, EXECUTE_V19D.format(
                        persona=persona,
                        priors_block=format_priors(priors),
                        wisdom_case_block=wisdom_case_block,
                        problem=problem,
                    ), max_tokens=1000, temperature=0.3)
                    draft = r1["text"].strip()
                    drafts[pid] = draft
                except Exception as e:
                    print(f"  [err draft {pid}] {e}")
                    continue

            # Turn 2: audit
            try:
                r2 = _generate_with_retry(client, REFLECT_PROMPT_V19D.format(
                    persona_label=frame, problem=problem, draft=draft
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

    from collections import Counter
    fc = Counter(f.get("frame", "?") for f in frames.values())
    print(f"  Frame distribution: {dict(fc)}")


if __name__ == "__main__":
    main()
