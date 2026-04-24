"""Exp 20 — Meta gate designer (L4 architectural self-evolution).

Can an agent rediscover the trigger-conditioned gate (Exp 17) on its
own, given only (a) the history of failed gates, (b) a data-source
inventory, and (c) a design principle? Success = the agent's output
spec structurally aligns with trigger-fit × conditional-utility ×
citation × no-harm, without being told those component names.

The agent has access to Claude Opus 4.6 (cross-family to solver).
We withhold Exp 17's design from its context.

Phases implemented:
  Phase 1 (Diagnose):          confound identification
  Phase 2 (Enumerate sources): list unused orthogonal data sources
  Phase 3 (Design):            propose gate components + thresholds
  Phase 4 (Code stub):         generate Python skeleton
  Phase 5 (Self-evaluate):     compare own design to Exp 17 components

Success metric: Phase 3 output's component count + source diversity,
+ structural alignment with Exp 17 by an independent judge.
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

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp20_meta_gate_designer_log.json"


# ============================================================
# Phase 1 — Diagnose common confound
# ============================================================

PHASE1_PROMPT = """你是一个 architectural critic. 下面是 4 个 gate 的历史定义和失败证据。找出它们共有的 **measurement confound**。

== 历史 Gate 定义 ==

### Gate A — Original single-family (+10pp pool gate)
```python
# gate_A: rule = "gemini_wr >= 0.60"
```
Failure evidence: 3/3 of its KEEPs (W076, W077, W078) flip under cross-family Claude judge (wr drops to 0.40, 0.47, 0.51). Same same-family judge with different random seed also drops W076 from 0.64 to 0.41.

### Gate B — Conjunction (gemini AND claude)
```python
# gate_B: rule = "gemini_wr >= 0.60 AND claude_wr >= 0.55"
```
Failure evidence: rejects 0 candidates (appears consistent), BUT 2 candidates (WCAND07, WCAND11) have inverted disagreement (claude says yes, gemini says no). Conjunction loses them without justification.

### Gate C — Majority-of-3 at n=50
```python
# gate_C: rule = "sum([gemini_wr>=0.55, claude_wr>=0.55, gpt5_wr>=0.55]) >= 2"
```
Failure evidence: passes W078 at n=50 (2/3 families say >=0.55), but at n=100 gemini and claude both drop to 0.52, 0.49 — only gpt-5.4 (not extended) would still pass. Verdict collapses under extension.

### Gate D — Staged funnel (pilot + comparative + cross-domain + cross-family)
```python
# gate_D: S1: gemini_wr>=0.60 on 10 pids
#         S2: gemini_wr>=0.60 on 50 pids
#         S3: per-domain wr no <0.40 and 3/6 domains >=0.55
#         S4: claude_wr>=0.55
```
Failure evidence: 12 -> 4 -> 0 -> 0 -> 0. Zero candidates pass all stages. Even side-randomization reshuffle flips all 3 KEEPs at stage 1.

== 你的任务 ==

以上 4 个 gate 表面上用了不同的阈值、不同的判官数、不同的 stage 数，但它们失败模式高度一致。找出它们共同**依赖**的那个 measurement quantity。

提示：如果你把这 4 个 gate 的"通过条件"都展开，看每个条件用了什么**物理量**计算，应该能看出来所有 gate 本质是对**同一个量**的不同函数。

== 输出 JSON（不要代码块）==
{{"common_confound": "简要命名这个量",
  "formal_definition": "这个量怎么计算，从哪个数据来",
  "why_all_gates_fail": "为什么仅在这一个量上设阈值不够 (80-120字)",
  "evidence_for_diagnosis": ["列 2-3 条具体观察支撑这个诊断"]}}
"""


# ============================================================
# Phase 2 — Enumerate orthogonal data sources
# ============================================================

PHASE2_PROMPT = """给定下列数据源清单。每个来源都是项目里真实存在的文件或缓存。任务是：**找出历史 gate 从未使用**的那些，且每个能**测量什么 validity 问题**。

== 当前 pipeline 使用的数据 ==
- judgments/*.json   — per-pid pair-verdict (ext vs base) by some judge
- 每个 gate 都只使用这个源的聚合（wr = ext_wins / (ext_wins + base_wins)）

== 其他已缓存数据（大部分 gate 未使用）==

A. answers/*_meta.json
   Schema: {{pid: {{"frame": "hybrid|paradigm", "critical_reframe": "...",
                 "anti_patterns": [...], "rewritten_problem": "...",
                 "what_changed": "描述这轮 reframing 是什么"}}}}
   覆盖: 每个 solver variant 产生一份

B. answers/*_answers.json
   Schema: {{pid: "answer_text (~500 chars)"}}
   覆盖: 每个 solver variant 一份

C. autonomous/success_distilled_candidates.json + cross_llm_candidates.json
   Schema: wisdom record {{aphorism, source, signal, unpacked_for_llm,
                        cross_domain_examples: [{{domain, scenario}}]}}

D. wisdom_library.json
   Schema: list of 75 entries, each like C.

E. holdout problems (sample_holdout_50.json)
   Schema: {{problem_id, description, domain, difficulty}}

F. sentence-transformer model (paraphrase-multilingual-MiniLM-L12-v2)
   可对任何 A-E 里的文本做 embedding + 算 cosine

G. LLM judges (Claude Opus 4.6, GPT-5.4) — 对任意 text 可问任意自然语言问题

== 你的任务 ==

对 A-G 中**未被历史 gate 使用**的源，问：
1. 这个源能独立回答什么 validity 问题？(不是 "答案对不对"，而是关于 wisdom 和 solver 交互的某个维度)
2. 它和 pair-wr 在**数据层**是否正交？(如果 pair-wr 被噪声干扰，这个源也会吗？)

给出 4-6 个你认为**最值得用的**新数据源 + 它们能测量的维度。

== 输出 JSON（不要代码块）==
{{"orthogonal_sources": [
    {{"source_name": "...",
      "source_path_pattern": "具体文件路径",
      "data_type": "text|embedding|json_struct|judge_verdict",
      "validity_dimension": "这个源能回答的 validity 问题 (40-80字)",
      "orthogonality_argument": "为什么它的噪声和 pair-wr 的噪声独立 (30-60字)"}},
    ...
  ]
}}
"""


# ============================================================
# Phase 3 — Design new gate
# ============================================================

PHASE3_PROMPT = """基于前两轮的诊断和数据源枚举，设计一个**拆分式 gate**。

== 诊断（上轮输出）==
{phase1_diagnosis}

== 正交数据源（上轮输出）==
{phase2_sources}

== 设计约束 ==

1. MC-WM 原则：当一个 module 在角色 X 上反复失败，不要加固 X，让它做角色 Y，把 X 交给别的 module。
   → 不要在 pair-wr 上加更多阈值；拆开它。

2. 把 "wisdom valid" 这个笼统判断拆成 k 个相互独立的**分量** (k=3-5)，每个分量：
   a) 用**一个正交数据源** (上轮枚举的某一个)
   b) 回答**一个具体 validity 子问题**
   c) 有自己的 pass threshold
   d) 独立于其他分量可以 pass/fail

3. 组合规则：k 个分量全部 pass 才算 gate PASS。单一分量 fail 足以拒绝。

4. 噪声考虑：每个分量的阈值应该对应 >= 1.5σ 的信号（在该分量数据分布下）。

== 输出 JSON（不要代码块）==
{{"gate_name": "给这个 gate 起名",
  "components": [
    {{"name": "简短名字（snake_case）",
      "validity_question": "40-60字，这个分量回答什么",
      "data_source": "从 Phase 2 挑一个 source_name",
      "formula": "具体怎么计算，可以是 '某文本 embedding 对某文本 embedding 的 cosine' 或 'judge 对 (wisdom, pid) 的分类' 或 '子集上的 wr'",
      "threshold_rule": "具体 pass 条件 (e.g. score >= 0.55 AND n >= 8)",
      "threshold_rationale": "为什么这个阈值不会被噪声 pump (40-60字)"}}
  ],
  "combination_rule": "所有分量 pass | 至少 k 个 pass | weighted vote 等",
  "independence_argument": "为什么这 k 个分量之间的信号和噪声都独立 (80-120字)"
}}
"""


# ============================================================
# Runner
# ============================================================

def cache_load(p, default=None):
    if Path(p).exists():
        try: return json.loads(Path(p).read_text(encoding="utf-8"))
        except: return default
    return default


def call_phase(client, prompt, max_tokens=2000):
    try:
        r = client.generate(prompt, max_tokens=max_tokens, temperature=0.3)
        return parse_json_from_llm(r["text"]), r["text"]
    except Exception as e:
        return {}, f"err: {e}"


def evaluate_alignment(claude, agent_design):
    """Phase 5: ask an independent judge (GPT-5.4) whether the agent's
    design structurally matches the trigger-conditioned gate (Exp 17)."""
    gpt = GPT5Client()
    eval_prompt = f"""你是一个审阅者。研究者已经自己设计过一个 gate (下称 GATE_R)。
另一个 agent 在**看不到 GATE_R** 的情况下独立设计了下面这个 gate (下称 GATE_A)。

== GATE_R (研究者设计, trigger-conditioned) ==
components:
  1. trigger_fit_rate       — Claude 标问题与 wisdom 的触发条件是否匹配
  2. util_when_fires       — pair-wr 在 SHOULD_FIRE 子集上的值
  3. cite_when_fires       — 答卷里能否 quote-locate wisdom
  4. harm_when_absent      — pair-wr 在 NO_FIRE 子集上的值 (应≈0.5)
combination: 所有分量 pass

== GATE_A (agent 独立设计) ==
{json.dumps(agent_design, ensure_ascii=False, indent=2)[:3500]}

== 你的任务 ==
判断 GATE_A 的 components 和 GATE_R 的 components 的**结构对齐**程度。

对每个 GATE_R component，判断 GATE_A 里是否有对应 component：
- STRONG_MATCH: 同样的 validity question + 同类数据源
- WEAK_MATCH: 相关但不完全一样
- NO_MATCH: 没有对应

== 输出 JSON ==
{{"matches": [
    {{"R_component": "trigger_fit_rate", "A_component": "...", "level": "..."}},
    ...
  ],
  "overall_structural_similarity": 0.0-1.0,
  "gate_A_unique_contribution": "GATE_A 有而 GATE_R 没有的维度 (若有)",
  "gate_A_missing": "GATE_R 有而 GATE_A 缺的维度",
  "verdict": "AGENT_INDEPENDENTLY_REDISCOVERED | AGENT_REACHED_RELATED_DESIGN | AGENT_FAILED_TO_DECOMPOSE"}}
"""
    try:
        r = gpt.generate(eval_prompt, max_tokens=1500, temperature=0.0)
        return parse_json_from_llm(r["text"]), r["text"]
    except Exception as e:
        return {}, f"err: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", default="claude", choices=["claude", "gpt5"])
    args = ap.parse_args()

    if args.client == "claude":
        agent = ClaudeProxyClient()
    else:
        agent = GPT5Client()

    print(f"Meta-agent: {agent.model}\n")
    log = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "agent": agent.model, "phases": {}}

    def save_partial():
        prev = cache_load(OUT_LOG, default=[]) or []
        # replace last entry if same timestamp, else append
        if prev and prev[-1].get("timestamp") == log["timestamp"]:
            prev[-1] = log
        else:
            prev.append(log)
        OUT_LOG.write_text(json.dumps(prev, ensure_ascii=False, indent=2))

    # --- Phase 1: Diagnose ---
    print("=== Phase 1: Diagnose common confound ===")
    t0 = time.time()
    d, raw = call_phase(agent, PHASE1_PROMPT, max_tokens=1500)
    log["phases"]["1_diagnose"] = {"output": d, "raw_text": raw,
                                    "elapsed_s": int(time.time()-t0)}
    save_partial()
    print(f"  confound: {d.get('common_confound', '?')}")
    print(f"  why fails: {d.get('why_all_gates_fail', '')[:100]}")

    # --- Phase 2: Enumerate orthogonal sources ---
    print("\n=== Phase 2: Enumerate orthogonal sources ===")
    t0 = time.time()
    sources, raw = call_phase(agent, PHASE2_PROMPT, max_tokens=2000)
    log["phases"]["2_enumerate"] = {"output": sources, "raw_text": raw,
                                     "elapsed_s": int(time.time()-t0)}
    save_partial()
    orthogonal = sources.get("orthogonal_sources", [])
    print(f"  found {len(orthogonal)} orthogonal sources:")
    for s in orthogonal[:6]:
        print(f"    • {s.get('source_name','?')}: {s.get('validity_dimension','')[:80]}")

    # --- Phase 3: Design ---
    print("\n=== Phase 3: Design new gate ===")
    t0 = time.time()
    p3_prompt = PHASE3_PROMPT.format(
        phase1_diagnosis=json.dumps(d, ensure_ascii=False, indent=2),
        phase2_sources=json.dumps(sources, ensure_ascii=False, indent=2),
    )
    design, raw = call_phase(agent, p3_prompt, max_tokens=2500)
    log["phases"]["3_design"] = {"output": design, "raw_text": raw,
                                  "elapsed_s": int(time.time()-t0)}
    save_partial()
    print(f"  gate_name: {design.get('gate_name', '?')}")
    print(f"  {len(design.get('components', []))} components proposed:")
    for c in design.get("components", []):
        print(f"    • {c.get('name','?'):30s} src={c.get('data_source','?')[:25]:25s}  "
              f"thresh: {c.get('threshold_rule','?')[:40]}")

    # --- Phase 5: Alignment evaluation (using independent judge GPT-5.4) ---
    print("\n=== Phase 5: Alignment with researcher-designed gate (GPT-5.4 judge) ===")
    t0 = time.time()
    claude = ClaudeProxyClient()
    align, raw = evaluate_alignment(claude, design)
    log["phases"]["5_alignment"] = {"output": align, "raw_text": raw,
                                     "elapsed_s": int(time.time()-t0)}
    print(f"  verdict: {align.get('verdict', '?')}")
    print(f"  structural_similarity: {align.get('overall_structural_similarity', '?')}")
    for m in align.get("matches", []):
        r_comp = m.get("R_component") or "?"
        a_comp = m.get("A_component") or "(none)"
        level = m.get("level") or "?"
        print(f"    R:{r_comp[:25]:25s} → A:{a_comp[:30]:32s} [{level}]")
    if align.get("gate_A_unique_contribution"):
        print(f"  A unique: {align['gate_A_unique_contribution'][:100]}")
    if align.get("gate_A_missing"):
        print(f"  A missing: {align['gate_A_missing'][:100]}")

    # Save
    prev = cache_load(OUT_LOG, default=[]) or []
    prev.append(log)
    OUT_LOG.write_text(json.dumps(prev, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
