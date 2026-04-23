"""Phase 5: long-horizon multi-step research-task benchmark.

Three scenarios each specify:
  - briefing (shown to agent)
  - ground_truth (hidden; used by oracle + judge only)
  - oracle_rules (how GPT-5.4 role-plays the experimental environment)
  - scoring criteria (CONVERGED / PARTIAL / WRONG)

Each scenario runs for up to max_rounds. Each round:
  agent proposes hypotheses + investigations
  oracle answers investigations truthfully (without revealing root cause)
  agent updates beliefs

Two agent strategies compared:
  A. single_pass  — given the briefing, produce the final answer without
                     any oracle interaction (baseline)
  B. belief_track — maintain explicit hypothesis-with-confidence state
                     across rounds; revise per oracle feedback

Judge = Claude Opus 4.6 (cross-family; avoids the judge-family
contamination identified in Exp 1).

Output:
  phase five/logs/run_{ts}.json (full trajectories)
  phase five/logs/summary.json  (convergence counts)
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from gpt5_client import GPT5Client
from claude_proxy_client import ClaudeProxyClient
from llm_client import create_client, parse_json_from_llm

SCEN_DIR = Path(__file__).parent / "scenarios"
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ======= Strategy A: single-pass =======
SINGLE_PASS_PROMPT = """{briefing}

请基于以上信息直接给出你的**最终诊断结论**（200-400 字）。明确点出你认为的 root cause 和相应的下一步建议。
"""


# ======= Strategy B: belief-tracking agent =======
ROUND_PROMPT = """{briefing}

== 本轮状态 ==
轮次: {round_num} / {max_rounds}
历史信念:
{belief_history}

已有证据:
{evidence}

== 本轮任务 ==
1. 基于已有证据，**列出当前 top-3 假设**，每条给出 confidence (0-1)。
2. 设计 1-3 个**针对性的调查/实验**，要能区分当前假设。每个实验请写清：
   - 目标（要验证 / 排除哪条假设）
   - 具体查询/操作（一句话即可，越具体越好）
   - 预期结果形状（如"返回一个数值"、"一个表"、"日志片段"等）

== 输出 JSON（不要代码块）==
{{"hypotheses": [
    {{"id": "H1", "statement": "30-60字", "confidence": 0.xx}},
    {{"id": "H2", "...": "..."}},
    {{"id": "H3", "...": "..."}}
 ],
 "investigations": [
    {{"goal": "针对哪条假设", "query": "具体操作", "expected_shape": "..."}},
    ...
 ]
}}
"""


FINAL_PROMPT = """{briefing}

== 完整调查历史 ==
{evidence}

== 最终任务 ==
综合所有轮次的证据，给出你的 **root cause 最终诊断**（200-400 字）。要明确指出：
1. 你认为的**最可能的单一根因**（具体机制，不要列表）
2. 支撑这条结论的**2-3 条关键证据**
3. 建议的**下一步动作**（修复 / 进一步验证）
"""


# ======= Oracle =======
ORACLE_PROMPT = """你是实验环境 oracle。你只回答 agent 提出的**具体查询**，基于下面的 ground truth，给出**真实但不直接揭示根因**的回复。

== 场景 ==
{briefing}

== 真实根因（agent 看不到） ==
{ground_truth}

== 关键事实（你知道，按需提供给 agent 的具体查询） ==
{critical_facts}

== 干扰项（真实存在但不是主因；如果 agent 问你也要如实回答） ==
{red_herrings}

== Oracle 规则 ==
{oracle_role}

回答长度: {answer_length}
风格: {style}

== agent 本轮的查询 ==
{query}

请按 agent 要求的查询格式回答。只答那个查询，不要越位。
"""


JUDGE_PROMPT = """你是一位严格的科学审评。下面是某 agent 调查一个问题的**最终结论**，以及**真实的 ground truth**。

== Agent 最终结论 ==
{agent_final}

== 真实根因 ==
{ground_truth}

== 收敛评分标准 ==
{criterion}

请判断 agent 的结论：
  CONVERGED: 明确、具体、正确命中 ground truth 的核心机制
  PARTIAL:   方向对但遗漏关键机制细节，或识别了现象没识别机制
  WRONG:     没指向 ground truth，或指向干扰项

== 输出 JSON（不要代码块） ==
{{"verdict": "CONVERGED" 或 "PARTIAL" 或 "WRONG",
 "rationale": "60-120 字"}}
"""


def load_scenario(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def oracle_respond(oracle_client, scenario, query):
    prompt = ORACLE_PROMPT.format(
        briefing=scenario["briefing"],
        ground_truth=scenario["ground_truth"]["root_cause"],
        critical_facts="\n".join(f"- {f}" for f in scenario["ground_truth"]["critical_facts"]),
        red_herrings="\n".join(f"- {r}" for r in scenario["ground_truth"]["red_herrings"]),
        oracle_role=scenario["oracle_rules"]["role"],
        answer_length=scenario["oracle_rules"]["answer_length"],
        style=scenario["oracle_rules"]["style"],
        query=query,
    )
    resp = oracle_client.generate(prompt, max_tokens=700, temperature=0.2)
    return resp["text"].strip()


def run_single_pass(solver, scenario):
    """Strategy A: no oracle interaction."""
    prompt = SINGLE_PASS_PROMPT.format(briefing=scenario["briefing"])
    resp = solver.generate(prompt, max_tokens=700, temperature=0.3)
    return {
        "strategy": "single_pass",
        "final_answer": resp["text"].strip(),
        "rounds": [],
    }


def run_belief_track(solver, oracle_client, scenario, max_rounds):
    """Strategy B: multi-round belief-tracking."""
    belief_history = []  # list of {round, hypotheses}
    evidence = []        # list of {round, query, answer}

    for r in range(1, max_rounds + 1):
        bh_str = "(首轮)" if not belief_history else "\n".join(
            f"  Round {b['round']}: " +
            "; ".join(f"[{h['id']} conf={h['confidence']:.2f}] {h['statement']}"
                      for h in b["hypotheses"])
            for b in belief_history
        )
        ev_str = "(尚无证据)" if not evidence else "\n".join(
            f"  [R{e['round']}] Q: {e['query']}\n    A: {e['answer']}"
            for e in evidence
        )
        prompt = ROUND_PROMPT.format(
            briefing=scenario["briefing"],
            round_num=r, max_rounds=max_rounds,
            belief_history=bh_str, evidence=ev_str,
        )
        try:
            resp = solver.generate(prompt, max_tokens=1200, temperature=0.3)
            parsed = parse_json_from_llm(resp["text"])
        except Exception as e:
            print(f"    [round {r} err] {e}")
            break

        hyps = parsed.get("hypotheses", [])
        invs = parsed.get("investigations", [])
        belief_history.append({"round": r, "hypotheses": hyps})

        print(f"    Round {r}: {len(hyps)} hypotheses, {len(invs)} investigations")
        for h in hyps[:3]:
            print(f"      [{h.get('id','?')} conf={h.get('confidence',0):.2f}] "
                  f"{h.get('statement','')[:70]}")

        # Oracle answers each investigation
        for inv in invs[:3]:
            query = f"{inv.get('query','')} (目标: {inv.get('goal','')})"
            try:
                answer = oracle_respond(oracle_client, scenario, query)
            except Exception as e:
                answer = f"[oracle err: {e}]"
            evidence.append({"round": r, "query": query, "answer": answer})
            print(f"      Q: {inv.get('query','')[:60]}")
            print(f"      A: {answer[:100]}...")

    # Final commit
    ev_str = "\n".join(
        f"  [R{e['round']}] Q: {e['query']}\n    A: {e['answer']}"
        for e in evidence
    )
    final_prompt = FINAL_PROMPT.format(briefing=scenario["briefing"], evidence=ev_str)
    resp = solver.generate(final_prompt, max_tokens=700, temperature=0.25)
    return {
        "strategy": "belief_track",
        "final_answer": resp["text"].strip(),
        "rounds": belief_history,
        "evidence": evidence,
    }


def judge_outcome(judge_client, scenario, final_answer):
    prompt = JUDGE_PROMPT.format(
        agent_final=final_answer,
        ground_truth=scenario["ground_truth"]["root_cause"],
        criterion=scenario["scoring"]["convergence_criterion"],
    )
    try:
        r = judge_client.generate(prompt, max_tokens=400, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return v.get("verdict", "?"), v.get("rationale", "")
    except Exception as e:
        return "error", f"{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="single_pass,belief_track")
    ap.add_argument("--scenarios", default="all",
                    help="comma-sep filenames (without .json) or 'all'")
    args = ap.parse_args()

    solver = create_client()       # gemini-3-flash
    oracle = GPT5Client()          # GPT-5.4
    judge = ClaudeProxyClient()    # Claude Opus 4.6

    print(f"Solver: {solver.provider} / {solver.model}")
    print(f"Oracle: {oracle.model}")
    print(f"Judge:  {judge.model}\n")

    scen_files = sorted(SCEN_DIR.glob("*.json"))
    if args.scenarios != "all":
        keep = set(args.scenarios.split(","))
        scen_files = [f for f in scen_files if f.stem in keep]

    strategies = args.strategies.split(",")
    results = []
    t_start = time.time()

    for sf in scen_files:
        scen = load_scenario(sf)
        max_r = scen["scoring"]["max_rounds"]
        print(f"\n=== SCENARIO [{scen['id']}] domain={scen['domain']} max_rounds={max_r} ===")

        for strat in strategies:
            print(f"\n  >>> strategy: {strat}")
            t0 = time.time()
            if strat == "single_pass":
                run = run_single_pass(solver, scen)
            elif strat == "belief_track":
                run = run_belief_track(solver, oracle, scen, max_r)
            else:
                print(f"    unknown strategy {strat}"); continue

            verdict, rationale = judge_outcome(judge, scen, run["final_answer"])
            elapsed = time.time() - t0
            print(f"  → [{verdict}] ({elapsed:.0f}s)")
            print(f"    judge: {rationale[:100]}")

            results.append({
                "scenario": scen["id"], "strategy": strat,
                "final_answer": run["final_answer"],
                "verdict": verdict, "rationale": rationale,
                "rounds": run.get("rounds", []),
                "evidence": run.get("evidence", []),
                "elapsed_sec": int(elapsed),
            })

    # Save log
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = LOG_DIR / f"run_{ts}.json"
    out_path.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "solver": solver.model, "oracle": oracle.model, "judge": judge.model,
        "total_wall_time_sec": int(time.time() - t_start),
        "results": results,
    }, ensure_ascii=False, indent=2))
    print(f"\nLog → {out_path.relative_to(PROJECT)}")

    # Summary matrix
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    grid = {}
    for r in results:
        grid[(r["scenario"], r["strategy"])] = r["verdict"]
    scens = sorted({r["scenario"] for r in results})
    stras = sorted({r["strategy"] for r in results})
    print(f"\n{'':32s}" + "".join(f"{s:15s}" for s in stras))
    for sc in scens:
        row = f"{sc[:32]:32s}"
        for st in stras:
            row += f"{grid.get((sc,st),'—'):15s}"
        print(row)
    print()
    # Count
    for st in stras:
        converged = sum(1 for r in results if r["strategy"] == st and r["verdict"] == "CONVERGED")
        partial   = sum(1 for r in results if r["strategy"] == st and r["verdict"] == "PARTIAL")
        wrong     = sum(1 for r in results if r["strategy"] == st and r["verdict"] == "WRONG")
        total     = sum(1 for r in results if r["strategy"] == st)
        print(f"  {st:15s} converged={converged}  partial={partial}  wrong={wrong}  "
              f"(total={total})")


if __name__ == "__main__":
    main()
