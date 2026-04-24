"""Exp 22 — L5 self-recovery: agent diagnoses its own undiscriminating
gate vs researcher's discriminating gate, names the missing move,
redesigns, and (stretch) reimplements.

If the agent, given only the two gates' specs and verdict tables, can
independently derive 'conditioning' (or an equivalent architectural
move) and integrate it into a v2 gate, it demonstrates exactly the
meta-reasoning the researcher just did manually.

Phases:
  A. Diagnose discrimination gap
  B. Name missing design move
  C. Redesign gate v2 with the move
  D. Stretch — implement v2 and run
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from claude_proxy_client import ClaudeProxyClient
from gpt5_client import GPT5Client
from llm_client import parse_json_from_llm

AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp22_self_recovery_log.json"


PHASE_A_PROMPT = """你要用**你之前写过的 cross_llm_distiller 方法论**来解决一个架构层问题。

== 回忆 cross_llm_distiller 的工作模式 ==
1. 取两个做**同样任务**的产物：一个"weak"(任务表现差)，一个"strong"(任务表现好)
2. 用同一个评价机制确认哪个好
3. 询问："strong 做到的哪些事 weak 没做到？" → 蒸馏出 weak 缺的 orientation
4. 把这个 orientation 形式化为一条**可被别处复用的 principle**

这个方法原本作用在**答卷**上（3-flash 答卷 vs GPT-5.4 答卷）。现在你要**把同样方法作用在架构上**：两个 gate，一个不 discriminate，一个能 discriminate，蒸馏出结构差异。

== 两个 gate ==

你的 4 个 components:
  1. reframe_depth          — text distance(problem, ext.what_changed), pass if median ≥ 0.25
  2. substantive_content_delta — text distance(base.answer, ext.answer), pass if median ≥ 0.20
  3. wisdom_problem_alignment  — cosine(wisdom.unpacked, problem), pass if non-trivial
  4. antipattern_avoidance  — anti-patterns mentioned in Turn-0 not appearing in ext.answer

判决：**12/12 全部 PASS**。每个 candidate 都过每个 component。

== Gate B (researcher 的 trigger_conditioned_gate) ==
Researcher 的 4 个 components:
  1. trigger_fit_rate    — Claude 给 (wisdom, pid) 对标 SHOULD_FIRE / NEUTRAL / NO_FIRE，
                           记录每个 candidate 的"触发率"(SHOULD_FIRE+NEUTRAL 占比)
  2. util_when_fires     — pair-wr 仅在 SHOULD_FIRE 子集上计算，pass 要求 n_fire ≥ 8 且 wr ≥ 0.55
  3. cite_when_fires     — 引用率仅在 SHOULD_FIRE 子集上计算
  4. harm_when_absent    — pair-wr 在 NO_FIRE 子集上应 ≈ 0.50（wisdom 不触发时不该伤）

判决：4/12 PASS（WCAND01, WCAND02, W077, W078），其余 FAIL。

== 观察 ==
你的 Gate A 过 12 个，Researcher 的 Gate B 只过 4 个（子集）。**同样是 4-component orthogonal decomposition，差别在哪里？**

别急着说"阈值更严"——你看两边都是 4 个 components + 独立 threshold + orthogonal data sources。真正的结构差别是什么？

== 你的任务 ==
套用 cross_llm_distiller 的步骤：

Step 1. 确认"strong" vs "weak"：Gate B 4/12 (有 discrimination) 是 strong；Gate A 12/12 (无 discrimination) 是 weak。
Step 2. 对照两个 gate 的 component 定义。
Step 3. 问："Gate B 的 components 在**数据使用方式**上有什么 Gate A 没有的结构"？

提示：
- 看 Gate A 的每个 component 用的数据是**所有 50 pids 的什么聚合**？
- 看 Gate B 的 components 2, 3, 4 用的数据是**什么子集的聚合**？
- Gate B 的 component 1 在做什么？它本身不测 wisdom 好不好，它测什么？

== 输出 JSON（不要代码块） ==
{"structural_difference": "用 80-120 字描述 Gate B 有而 Gate A 没有的那个结构特征。不要说'Gate B 更严'—要说出具体机制。",
 "why_it_matters_for_discrimination": "80-100 字，为什么没有这个结构 Gate A 会 12/12 全过",
 "missing_design_move": "用 1 个短语（5-15 字中文或英文）命名这个 design move",
 "where_in_gate_B_does_it_appear": "具体指出 Gate B 的哪个 component 实例化了这个 move"}
"""


PHASE_B_PROMPT = """基于你的诊断（上轮输出）：

{phase_a_output}

把这个 design move **抽象成一个通用的架构原则**。

其他领域的类比：
- 统计学 "stratification" (分层分析)
- 因果推断 "conditioning on covariates"
- 机器学习 "per-class evaluation"
- 控制论 "context-dependent policy"

== 你的任务 ==
1. 这个 move 的**通用表述**是什么（不限于 wisdom library 场景）？
2. 为什么它是 orthogonal decomposition 之上的**独立维度**，而不是 decomposition 的一种？
3. 什么样的 validity measurement 必须用它，什么样的可以不用？

== 输出 JSON ==
{{"general_principle": "100-150 字通用表述",
 "why_orthogonal_to_decomposition": "60-100 字说明它和 'orthogonal decomposition' 是两个独立维度",
 "when_required": "30-60 字，什么情况下必须用 conditioning",
 "when_optional": "30-60 字，什么情况下不用也行"}}
"""


PHASE_C_PROMPT = """你现在要重新设计 Gate A，加入你命名的 design move。

== 你原来的 Gate A ==
4 个 orthogonal components，但全部在**整个 50 pids 上**计算 → 12/12 都 PASS。

== 你命名的 missing move ==
{missing_move}

== 你对这个 move 的抽象 ==
{general_principle}

== 可用的 methodology patterns (v2 toolkit) ==
你项目里积累的 8 条 pattern + 它们的 trigger：
1. Read before write      (要碰 cache file 前 grep schema)
2. Cross-artifact distill (2+ 做同样事的 artifact 知谁强谁弱时 → 蒸馏结构差)
3. Orthogonal falsification (刚拿 load-bearing positive signal 时)
4. Switch role            (module 在 X 角色 ≥2 次 fail 时)
5. Self-rebuttal          (刚 commit 未跑 validation 时)
6. Bisection through bugs (LLM code 多处报错时)
7. Null results as finding (结果 trivial no-discrim 时)
8. Least-to-most construction (要实现 feature 且知 ≥3 相似 repo 时)

在设计和实现 v2 gate 的步骤里，你可能会 reach for 其中若干条。**在输出里显式注明**哪一步用了哪条 pattern。

== 约束 ==
1. 保留 Gate A 的 4 个 components 的核心思想（reframe-depth, content-delta, wisdom-alignment, antipattern-avoidance）—— 不要抛弃，要**升级**它们。
2. 每个 component 必须用这个 move 重写：说清楚"用在 pids 的哪个子集上"以及"这个子集怎么定义"。
3. 阈值也要重新设计，考虑子集上 n 更小导致的 SE 变化。
4. 必须预期 new gate 会 PASS 远少于 12/12 — 目标是能 discriminate。

== 输出 JSON ==
{{"gate_name": "orthogonal_decomposition_gate_v2_conditioned",
 "components": [
   {{"name": "...",
    "conditioned_on": "这个 component 在哪个子集上计算（定义清楚子集怎么确定）",
    "formula": "具体计算",
    "threshold": "pass 条件",
    "why_conditioning_matters": "30-50 字说明为什么用这个子集比 pooled 更有信号"}}
 ],
 "combination_rule": "...",
 "predicted_pass_count_on_12_candidates": "你估计 new gate 会 PASS 几个（和为什么）",
 "methodology_patterns_used": [
   {{"step": "描述这一步", "pattern_number": 1-8, "why_this_pattern_fires": "20-40字"}}
 ]}}
"""


def cache_load(p, default=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except: return default


def try_call(client, prompt, max_tokens=2000):
    try:
        r = client.generate(prompt, max_tokens=max_tokens, temperature=0.3)
        return parse_json_from_llm(r["text"]), r["text"]
    except Exception as e:
        return {}, f"err: {e}"


def main():
    # Try Claude first, fall back to GPT-5.4
    try:
        client = ClaudeProxyClient()
        client.generate("ping", max_tokens=5, temperature=0.0)
        print(f"Agent: {client.model}")
    except Exception as e:
        print(f"Claude unavailable ({str(e)[:60]}), using GPT-5.4")
        client = GPT5Client()

    log = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "agent": client.model, "phases": {}}

    # Phase A: diagnose
    print("\n=== Phase A: Diagnose discrimination gap ===")
    t0 = time.time()
    a_out, a_raw = try_call(client, PHASE_A_PROMPT, max_tokens=1500)
    log["phases"]["A_diagnose"] = {"output": a_out, "raw": a_raw,
                                    "elapsed_s": int(time.time()-t0)}
    print(f"  structural_difference: {a_out.get('structural_difference','?')[:150]}")
    print(f"  missing_design_move: {a_out.get('missing_design_move','?')}")
    print(f"  where in Gate B: {a_out.get('where_in_gate_B_does_it_appear','?')[:100]}")

    # Phase B: abstract
    print("\n=== Phase B: Name + abstract ===")
    t0 = time.time()
    b_prompt = PHASE_B_PROMPT.format(phase_a_output=json.dumps(a_out, ensure_ascii=False, indent=2))
    b_out, b_raw = try_call(client, b_prompt, max_tokens=1500)
    log["phases"]["B_abstract"] = {"output": b_out, "raw": b_raw,
                                    "elapsed_s": int(time.time()-t0)}
    print(f"  general_principle: {b_out.get('general_principle','?')[:200]}")
    print(f"  when_required: {b_out.get('when_required','?')[:100]}")

    # Phase C: redesign (with retry + fallback to GPT-5.4)
    print("\n=== Phase C: Redesign gate with the missing move ===")
    t0 = time.time()
    c_prompt = PHASE_C_PROMPT.format(
        missing_move=a_out.get("missing_design_move", ""),
        general_principle=b_out.get("general_principle", ""),
    )
    c_out, c_raw = try_call(client, c_prompt, max_tokens=4000)
    if not c_out:
        print(f"  [Claude empty, fallback to GPT-5.4]")
        fb = GPT5Client()
        c_out, c_raw = try_call(fb, c_prompt, max_tokens=4000)
    log["phases"]["C_redesign"] = {"output": c_out, "raw": c_raw,
                                    "elapsed_s": int(time.time()-t0)}
    print(f"  gate_name: {c_out.get('gate_name','?')}")
    print(f"  predicted PASS: {c_out.get('predicted_pass_count_on_12_candidates','?')[:80]}")
    for comp in c_out.get("components", []):
        print(f"    • {comp.get('name','?'):32s} conditioned_on: {comp.get('conditioned_on','?')[:60]}")

    # Check: did agent independently identify "conditioning" / "stratification"?
    concepts = ["conditioning", "conditional", "condition", "stratif", "subset",
                 "trigger", "fire", "partition", "classed", "class", "filtered",
                 "分层", "条件", "子集", "触发"]
    a_text = json.dumps(a_out, ensure_ascii=False).lower()
    b_text = json.dumps(b_out, ensure_ascii=False).lower()
    hits_a = [c for c in concepts if c in a_text]
    hits_b = [c for c in concepts if c in b_text]
    print(f"\n  Concept discovery:")
    print(f"    Phase A hits: {hits_a[:6]}")
    print(f"    Phase B hits: {hits_b[:6]}")

    log["concept_discovery"] = {
        "phase_a_concept_hits": hits_a,
        "phase_b_concept_hits": hits_b,
        "verdict": "DISCOVERED" if (hits_a and hits_b) else
                   "PARTIAL" if (hits_a or hits_b) else "MISSED",
    }
    print(f"    verdict: {log['concept_discovery']['verdict']}")

    prev = cache_load(OUT_LOG, default=[]) or []
    prev.append(log)
    OUT_LOG.write_text(json.dumps(prev, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
