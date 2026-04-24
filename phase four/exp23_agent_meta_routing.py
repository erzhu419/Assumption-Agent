"""Exp 23 — Agent independently proposes the meta-routing table for
methodology patterns (what v1 Section 9 was hand-written by researcher).

If agent can produce a comparable table, the last hand-written piece
of METHODOLOGY.md becomes agent-derivable.

Phases:
  P1. Show agent 8 patterns (description only, NO triggers from v2).
  P2. Ask: for each pattern, what external signal would a person-in-
      project notice that should make them reach for this pattern?
  P3. Compare with researcher's Section 9.
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
OUT_LOG = AUTO_DIR / "exp23_agent_meta_routing_log.json"


PATTERNS_INPUT = """== 8 个 methodology patterns (摘自实际项目经验) ==

Pattern 1: Read before write
  要点: 写任何新 exp 代码前，先 grep/Read 相关 meta/answer/log/md 文件看真实 schema。
  用过后的感受: 80% 实现 bug 是 data-contract 假设错误。

Pattern 2: Cross-artifact distillation
  要点: 手上有 2+ 个做同样任务的 artifact，其中一个 strong 一个 weak。
        蒸馏 "strong 做到的哪些事 weak 没做到" → 升华为可复用 principle。
  用过后的感受: 这是本项目最通用的 primitive，能作用在 answer / architecture / method 三层。

Pattern 3: Orthogonal falsification
  要点: 任何 positive signal 必须从**数据流不同源**的测量再证实一次。
  用过后的感受: 不 orthogonal 的 falsification 只是换几家 judge，依然共享 judge-bias noise。

Pattern 4: Switch role when module fails (MC-WM)
  要点: 一个 module 在角色 X 上反复失败，不要加固 X，让它做 Y，把 X 交给别人。
  用过后的感受: 省掉"无限加严参数"的死循环，跳出到 meta。

Pattern 5: Agent self-rebuttal as prediction
  要点: 让 agent 在任何 validation 跑**之前**提 3 条反驳。
        把反驳记下来当 falsifiable prediction。
  用过后的感受: agent 往往能列出人会忽略的 confound (e.g. library interaction).

Pattern 6: Bisection through bug layers
  要点: LLM 生成的代码失败时，不要盲目 re-prompt rewrite。读生成的代码，识别具体哪一层 bug。
  用过后的感受: 盲目重写会产生新 bug。Read-then-patch 省很多轮次。

Pattern 7: Null results are findings
  要点: 测量结果 trivial (all-PASS / all-FAIL / no-discrim) 时，要命名这个发现，
        不要继续调参直到"看起来对"。
  用过后的感受: chasing higher pass count 会 over-claim。

Pattern 8: Least-to-most construction
  要点: 要实现 feature 时，不从零写；找 ≥3 个现成类似 repo，选最近的 1 个，
        明确 "我要换的是哪 1 个 module"，其他保留。
  用过后的感受: 省 80% 实现时间，前提是能 articulate 替换点。

== 你的任务 ==

想象一个 agent（或人类 developer）在项目里工作。**当下 external signal** 是
某个**可外显观察**的事件（e.g. "我要打开一个 cached json 文件了"、"我刚
commit 了 3 条假设"、"我的 code 连续 3 个 attempt fail"）。

对每条 pattern，回答：**什么样的 external signal 会触发"现在该 reach for 这条
pattern"**？signal 必须：
- 可外显观察（不是 agent 的内心状态）
- 具体（不是"当情况需要时"这种空话）
- 互斥（同一 signal 不应同时触发多条 pattern）

把 8 条 pattern 的 triggers 整理成一张 meta-routing table。

== 输出 JSON（不要代码块）==
{"routing_table": [
    {"pattern_number": 1,
     "external_signal": "具体外显事件 (30-60 字)",
     "counter_example_signal": "什么 signal **不**应触发这条 pattern (防止误用)"},
    ... (8 条)
  ],
  "hardest_disambiguation": "在你看来最容易混淆的两条 pattern 是哪两条？为什么？",
  "missing_patterns_you_noticed": "按项目经验，还有哪些没列入的 pattern 值得加 (可选)"
}
"""


# Researcher's hand-written Section 9 for comparison
RESEARCHER_ROUTING = {
    1: "要开写新代码碰缓存文件",
    2: "面前有 2+ 个 artifact 且知道谁强谁弱",
    3: "刚拿到一个 load-bearing positive signal",
    4: "某 module 在 X 角色 ≥ 2 次 fail",
    5: "刚 commit 一个 KEEP 且还没跑 validation",
    6: "LLM code 失败且 traceback 指向多处",
    7: "结果 trivial / no discrimination",
    8: "要实现 feature 且知道 ≥ 3 相似 repo",
}


COMPARE_PROMPT = """Researcher 手写的 meta-routing table:

{researcher}

Agent 独立产出的 meta-routing:

{agent}

对每条 pattern (1-8)，判断 agent 的 trigger 和 researcher 的对齐度：
- STRONG_MATCH: agent 的 signal 和 researcher 的在**触发条件**上一致
- WEAK_MATCH: 方向对但具体触发点有偏差
- DIVERGED: agent 识别了一个不同的 trigger（可能更好或更差）
- MISSED: agent 根本没意识到这种情况

== 输出 JSON ==
{{"alignments": [
    {{"pattern": 1, "level": "...", "comment": "30-60 字"}},
    ...
  ],
  "strong_match_count": N,
  "overall_similarity": 0.0-1.0,
  "agent_insight_researcher_missed": "agent 提到的、researcher 没想到的 trigger 角度（若有）",
  "researcher_insight_agent_missed": "researcher 的 trigger 里 agent 没 cover 的（若有）"
}}
"""


def cache_load(p, default=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except: return default


def try_call(client, prompt, max_tokens=2500):
    try:
        r = client.generate(prompt, max_tokens=max_tokens, temperature=0.3)
        return parse_json_from_llm(r["text"]), r["text"]
    except Exception as e:
        return {}, f"err: {e}"


def main():
    # Try Claude first, fall back
    try:
        client = ClaudeProxyClient()
        client.generate("ping", max_tokens=5, temperature=0.0)
        print(f"Agent: {client.model}")
    except Exception as e:
        print(f"Claude unavailable ({str(e)[:60]}); using GPT-5.4")
        client = GPT5Client()

    log = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "agent": client.model}

    # P1+P2 combined: agent produces routing
    print("\n=== Phase 1+2: agent produces meta-routing ===")
    t0 = time.time()
    out, raw = try_call(client, PATTERNS_INPUT, max_tokens=2500)
    log["agent_routing"] = {"output": out, "raw_text": raw,
                             "elapsed_s": int(time.time()-t0)}
    routing = out.get("routing_table", [])
    print(f"  Produced {len(routing)} trigger entries:")
    for r in routing:
        p_num = r.get("pattern_number")
        sig = r.get("external_signal", "?")
        print(f"    [P{p_num}] {sig[:75]}")
    if out.get("hardest_disambiguation"):
        print(f"\n  Hardest disambiguation: {out['hardest_disambiguation'][:150]}")
    if out.get("missing_patterns_you_noticed"):
        print(f"  Missing patterns suggested: {out['missing_patterns_you_noticed'][:150]}")

    # P3: compare with researcher's
    print("\n=== Phase 3: compare with researcher's Section 9 ===")
    researcher_str = "\n".join(f"  Pattern {k}: {v}" for k, v in RESEARCHER_ROUTING.items())
    agent_str = "\n".join(
        f"  Pattern {r.get('pattern_number','?')}: {r.get('external_signal','?')}"
        for r in routing
    )
    t0 = time.time()
    # Use same client to judge (could swap to different for cleanliness, but stays in family here)
    cmp_prompt = COMPARE_PROMPT.format(researcher=researcher_str, agent=agent_str)
    cmp_out, cmp_raw = try_call(client, cmp_prompt, max_tokens=2000)
    log["alignment"] = {"output": cmp_out, "raw_text": cmp_raw,
                         "elapsed_s": int(time.time()-t0)}
    print(f"  alignment: {cmp_out.get('overall_similarity','?')}")
    print(f"  strong_match_count: {cmp_out.get('strong_match_count','?')}/8")
    for a in cmp_out.get("alignments", []):
        print(f"    P{a.get('pattern','?')}: {a.get('level','?'):12s} "
              f"{a.get('comment','')[:70]}")
    if cmp_out.get("agent_insight_researcher_missed"):
        print(f"\n  Agent insight researcher missed: {cmp_out['agent_insight_researcher_missed'][:150]}")
    if cmp_out.get("researcher_insight_agent_missed"):
        print(f"  Researcher insight agent missed: {cmp_out['researcher_insight_agent_missed'][:150]}")

    prev = cache_load(OUT_LOG, default=[]) or []
    prev.append(log)
    OUT_LOG.write_text(json.dumps(prev, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
