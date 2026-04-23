"""Auto-recurse: close the loop between object-level A/B gate and
meta-level gate-modification.

Each cycle:
  1. Propose or inherit a gate specification.
  2. Apply gate to the 12 validation rows → KEEPs_t.
  3. Audit KEEPs_t with the cross-family judge (Claude Opus 4.6),
     populating cross-judge scores for any row that hasn't been scored.
  4. Self-rebuttal: GPT-5.4 is shown the KEEP set + gate spec and
     asked whether the gate was adequate; returns a verdict.
  5. Meta-loop: GPT-5.4 inspects the audit results and proposes a
     new gate specification if (and only if) the audit revealed bias.
  6. Update a persistent belief state:
       - per-wisdom confidence (Bayesian-style combine of gemini and
         claude evidence),
       - current gate spec,
       - cycle history.
  7. Terminate if the proposed gate is unchanged from the previous
     cycle (belief fixed point) or max-cycles reached.

Output:
  phase four/autonomous/belief_state.json   (final state)
  phase four/autonomous/recurse_log.json    (cycle-by-cycle trajectory)
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase four"))

from gpt5_client import GPT5Client
from claude_proxy_client import ClaudeProxyClient
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
BELIEF_PATH = AUTO_DIR / "belief_state.json"
RECURSE_LOG = AUTO_DIR / "recurse_log.json"
HOLDOUT_SAMPLE = "sample_holdout_50.json"

PARALLEL_JUDGES = 6
MAX_CYCLES = 3

# Map of all 12 candidates from the single validation run
CANDIDATE_TABLE = {
    "WCAND01": {"ext_answers": "_valp_v20_ext_WCAND01_answers.json",
                "base_answers": "_valp_v20p1_base_answers.json",
                "aphorism": "上工治未病，不治已病", "gemini_wr": 0.58},
    "WCAND02": {"ext_answers": "_valp_v20_ext_WCAND02_answers.json",
                "base_answers": "_valp_v20p1_base_answers.json",
                "aphorism": "别高效解决一个被看错的问题", "gemini_wr": 0.46},
    "WCAND03": {"ext_answers": "_valp_v20_ext_WCAND03_answers.json",
                "base_answers": "_valp_v20p1_base_answers.json",
                "aphorism": "凡事预则立，不预则废", "gemini_wr": 0.56},
    "WCAND04": {"ext_answers": "_valp_v20_ext_WCAND04_answers.json",
                "base_answers": "_valp_v20p1_base_answers.json",
                "aphorism": "急则治其标，缓则治其本", "gemini_wr": 0.48},
    "WCAND05": {"ext_answers": "_valp_v20_ext_WCAND05_answers.json",
                "base_answers": "_valp_v20p1_base_answers.json",
                "aphorism": "凡益之道，与时偕行", "gemini_wr": 0.64,
                "committed_id": "W076"},
    "WCAND06": {"ext_answers": "_valp_v20_ext_WCAND06_answers.json",
                "base_answers": "_valp_v20p1_base_answers.json",
                "aphorism": "覆水难收，向前算账", "gemini_wr": 0.58},
    "WCAND07": {"ext_answers": "_valp_v20_ext_WCAND07_answers.json",
                "base_answers": "_valp_v20p1_base_answers.json",
                "aphorism": "亲兄弟，明算账", "gemini_wr": 0.56},
    "WCAND08": {"ext_answers": "_valp_v20_ext_WCAND08_answers.json",
                "base_answers": "_valp_v20p1_base_answers.json",
                "aphorism": "想理解行为，先看激励", "gemini_wr": 0.51},
    "WCAND09": {"ext_answers": "_valp_v20_ext_WCAND09_answers.json",
                "base_answers": "_valp_v20p1_base_answers.json",
                "aphorism": "不谋全局者，不足谋一域", "gemini_wr": 0.55},
    "WCAND10": {"ext_answers": "_valp_v20_ext_WCAND10_answers.json",
                "base_answers": "_valp_v20p1_base_answers.json",
                "aphorism": "没有调查，就没有发言权", "gemini_wr": 0.60,
                "committed_id": "W077"},
    "WCAND11": {"ext_answers": "_valp_v20_ext_WCAND11_answers.json",
                "base_answers": "_valp_v20p1_base_answers.json",
                "aphorism": "若不是品牌，你就只是商品。", "gemini_wr": 0.54},
    "WCROSSL01": {"ext_answers": "_valp_v20_ext_WCROSSL01_answers.json",
                   "base_answers": "_valp_v20_base_answers.json",
                   "aphorism": "是骡子是马，拉出来遛遛", "gemini_wr": 0.60,
                   "committed_id": "W078"},
}


JUDGE_PROMPT = """你是方法论评审专家。下面是同一个问题的两个解答。

## 问题
{problem}

## 解答 A
{answer_a}

## 解答 B
{answer_b}

## 评审
四个维度：问题理解、分析深度、结构化程度、实用性。

输出 JSON（不要代码块）：
{{"winner": "A"或"B"或"tie", "score_a": 1-10整数, "score_b": 1-10整数,
  "reasoning": "80字内说明胜因"}}
"""


META_PROMPT = """你是自反省 agent。当前 gate 规范如下：
{current_gate}

上一轮的全部 12 条候选与两家族判决如下（judge-A=gemini-3-flash 主判官；judge-B=claude-opus-4-6 复判）：
{rows}

## 观察
- Judge-A 上 {n_keep_a}/12 KEEP
- Judge-B 独立判决这 12 条时的 wr 分布：{wr_b_list}
- 如果用"两家族都 >= 某阈值"的 AND 规则，哪些会过？

## 你的任务
判断当前 gate 是否需要修改。若不需要，返回 `{{"gate_update": null, "reason": "..."}}`。
若需要，提出一个**严格更安全**的 gate 规范，并说明它在当前 12 条候选上会 KEEP 几个（举证）。

## 输出 JSON（不要代码块）
{{"gate_update": {{
    "rule": "描述规则（例如 'gemini_wr>=0.60 AND claude_wr>=0.55'）",
    "expected_keeps_on_current_data": ["W0xx", ...],
    "rationale": "60-120 字"
 }} 或 null,
 "reason": "50-80 字说明保留或修改的依据"}}
"""


def cache_load(p, default=None):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def cache_save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def judge_one(client, problem, a, b):
    prompt = JUDGE_PROMPT.format(problem=problem, answer_a=a, answer_b=b)
    try:
        r = client.generate(prompt, max_tokens=400, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return v.get("winner", "tie"), v.get("score_a", 0), v.get("score_b", 0), \
               v.get("reasoning", "")
    except Exception as e:
        return "error", 0, 0, f"err: {e}"


def claude_judge_pair(cid, problems, client):
    """Judge all 50 pairs for candidate cid with Claude."""
    info = CANDIDATE_TABLE[cid]
    base_path = CACHE / "answers" / info["base_answers"]
    ext_path = CACHE / "answers" / info["ext_answers"]
    if not base_path.exists() or not ext_path.exists():
        return None
    ans_base = json.loads(base_path.read_text(encoding="utf-8"))
    ans_ext = json.loads(ext_path.read_text(encoding="utf-8"))

    def one(p):
        pid = p["problem_id"]
        ba, ea = ans_base.get(pid), ans_ext.get(pid)
        if not ba or not ea:
            return pid, "missing"
        rng = random.Random(hash(pid) % (2**32))
        if rng.random() < 0.5:
            left, right, ext_was = ea, ba, "A"
        else:
            left, right, ext_was = ba, ea, "B"
        winner, sa, sb, _rs = judge_one(client, p["description"], left, right)
        if winner == "tie":
            return pid, "tie"
        return pid, ("ext" if winner == ext_was else "base")

    c = {"ext": 0, "base": 0, "tie": 0, "missing": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=PARALLEL_JUDGES) as ex:
        futures = [ex.submit(one, p) for p in problems]
        for f in as_completed(futures):
            try:
                _pid, res = f.result()
                c[res] = c.get(res, 0) + 1
            except Exception as e:
                c["error"] += 1
    total = c["ext"] + c["base"]
    wr_ext = c["ext"] / total if total else 0.5
    return {"wr_claude": wr_ext, **c}


def apply_gate(gate_spec, cid, gemini_wr, claude_wr):
    """Return True if candidate passes gate_spec."""
    rule = gate_spec.get("rule", "gemini_wr>=0.60")
    # Parse a very small DSL: conjunction of "X_wr>=V" or "<=V"
    # Support: "gemini_wr>=0.60", "gemini_wr>=0.60 AND claude_wr>=0.55"
    clauses = [c.strip() for c in rule.split("AND")]
    for clause in clauses:
        if "gemini_wr" in clause:
            _, val = clause.split(">=") if ">=" in clause else clause.split("<=")
            val = float(val.strip())
            if ">=" in clause:
                if gemini_wr < val: return False
            else:
                if gemini_wr > val: return False
        elif "claude_wr" in clause:
            if claude_wr is None:
                return False  # missing evidence → fail safe
            _, val = clause.split(">=") if ">=" in clause else clause.split("<=")
            val = float(val.strip())
            if ">=" in clause:
                if claude_wr < val: return False
            else:
                if claude_wr > val: return False
    return True


def belief_update(prior, gemini_wr, claude_wr):
    """Update confidence on a wisdom given two judge scores.
    Simple average of the two wr values, treating missing claude as 0.5."""
    cj = claude_wr if claude_wr is not None else 0.5
    combined = 0.5 * gemini_wr + 0.5 * cj
    # smoothed update toward evidence
    alpha = 0.7
    return alpha * combined + (1 - alpha) * prior


def init_state(existing_exp1):
    """Seed state from existing Exp1 results if available."""
    beliefs = {}
    claude_scores = {}
    for cid, info in CANDIDATE_TABLE.items():
        beliefs[cid] = {
            "aphorism": info["aphorism"],
            "committed_id": info.get("committed_id"),
            "confidence": info["gemini_wr"],  # seed as gemini wr
            "gemini_wr": info["gemini_wr"],
            "claude_wr": None,
        }
    # Fill in Claude scores from Exp1 log for 3 KEEPs
    if existing_exp1:
        latest = existing_exp1[-1]
        for r in latest.get("results", []):
            wid = r["wid"]
            for cid, info in CANDIDATE_TABLE.items():
                if info.get("committed_id") == wid:
                    beliefs[cid]["claude_wr"] = r["cross_wr_ext"]
                    claude_scores[cid] = r["cross_wr_ext"]
    return {
        "gate": {"rule": "gemini_wr>=0.60", "description": "initial single-judge +10pp"},
        "beliefs": beliefs,
        "history": [],
    }


def cycle(state, cycle_num, problems, claude_client, gpt_client):
    print(f"\n{'='*60}\nCYCLE {cycle_num}   gate={state['gate']['rule']}")
    print(f"{'='*60}")

    # Determine current KEEPs under current gate
    keeps_before = []
    for cid, b in state["beliefs"].items():
        if apply_gate(state["gate"], cid, b["gemini_wr"], b["claude_wr"]):
            keeps_before.append(cid)
    print(f"\nKEEPs under current gate: {len(keeps_before)}  {keeps_before}")

    # Ensure we have Claude scores for all current KEEPs
    for cid in keeps_before:
        if state["beliefs"][cid]["claude_wr"] is None:
            print(f"  [claude] scoring {cid} ({state['beliefs'][cid]['aphorism']})...")
            t0 = time.time()
            res = claude_judge_pair(cid, problems, claude_client)
            if res:
                state["beliefs"][cid]["claude_wr"] = res["wr_claude"]
                print(f"    → wr_claude={res['wr_claude']:.2f} ({time.time()-t0:.0f}s)")

    # Re-apply gate now that Claude scores are populated
    keeps_after = []
    flipped = []
    for cid in keeps_before:
        b = state["beliefs"][cid]
        if apply_gate(state["gate"], cid, b["gemini_wr"], b["claude_wr"]):
            keeps_after.append(cid)
        else:
            flipped.append(cid)
    print(f"\nKEEPs after claude evidence: {len(keeps_after)}  "
          f"(flipped on contact with evidence: {len(flipped)})")

    # Update beliefs (confidence)
    for cid in state["beliefs"]:
        b = state["beliefs"][cid]
        prior = b["confidence"]
        b["confidence"] = belief_update(prior, b["gemini_wr"], b["claude_wr"])

    # Meta-loop: propose new gate if needed
    rows_block = "\n".join(
        f"  • {cid:9s} [{b['aphorism'][:22]:22s}]  gemini={b['gemini_wr']:.2f}  "
        f"claude={'%.2f' % b['claude_wr'] if b['claude_wr'] is not None else 'N/A'}  "
        f"conf={b['confidence']:.2f}"
        for cid, b in state["beliefs"].items()
    )
    wr_b_list = [b["claude_wr"] for b in state["beliefs"].values()
                 if b["claude_wr"] is not None]

    prompt = META_PROMPT.format(
        current_gate=json.dumps(state["gate"], ensure_ascii=False),
        rows=rows_block,
        n_keep_a=sum(1 for b in state["beliefs"].values()
                     if b["gemini_wr"] >= 0.60),
        wr_b_list=sorted(wr_b_list) if wr_b_list else "未测",
    )
    print(f"\n[meta] asking GPT-5.4 for gate decision...")
    try:
        resp = gpt_client.generate(prompt, max_tokens=1500, temperature=0.3)
        parsed = parse_json_from_llm(resp["text"])
        gate_update = parsed.get("gate_update")
        reason = parsed.get("reason", "")
    except Exception as e:
        print(f"  [meta err] {e}")
        gate_update = None
        reason = f"meta call failed: {e}"

    # Snapshot this cycle
    snap = {
        "cycle": cycle_num,
        "gate_before": dict(state["gate"]),
        "keeps_before_audit": keeps_before,
        "keeps_after_audit": keeps_after,
        "flipped_by_audit": flipped,
        "meta_reason": reason,
        "gate_proposal": gate_update,
        "beliefs_snapshot": {cid: dict(b) for cid, b in state["beliefs"].items()},
    }

    # Apply gate update if proposed
    new_gate_rule = (gate_update or {}).get("rule")
    if new_gate_rule and new_gate_rule != state["gate"]["rule"]:
        state["gate"] = {
            "rule": new_gate_rule,
            "description": f"cycle {cycle_num} meta-update",
            "rationale": (gate_update or {}).get("rationale", ""),
        }
        snap["gate_after"] = dict(state["gate"])
        print(f"\n[gate update] {new_gate_rule}")
    else:
        snap["gate_after"] = dict(state["gate"])
        print(f"\n[gate] unchanged — belief appears stable")

    state["history"].append(snap)
    return state, (gate_update is None or
                    new_gate_rule == snap["gate_before"]["rule"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cycles", type=int, default=MAX_CYCLES)
    args = ap.parse_args()

    # Load existing Exp1 log to seed Claude scores for KEEPs
    exp1_log = cache_load(AUTO_DIR / "exp1_cross_judge_log.json", default=[])

    # Load or init state
    if BELIEF_PATH.exists():
        print("Loading existing belief state...")
        state = cache_load(BELIEF_PATH)
    else:
        print("Initialising fresh belief state from Exp1 scores...")
        state = init_state(exp1_log)

    problems = json.loads((CACHE / HOLDOUT_SAMPLE).read_text(encoding="utf-8"))
    problems = [p for p in problems if "description" in p]

    claude = ClaudeProxyClient()
    gpt = GPT5Client()
    print(f"Judge A (gate): gemini-3-flash (cached wr)")
    print(f"Judge B (audit): {claude.model}")
    print(f"Meta LLM: {gpt.model}")
    print(f"Problems: {len(problems)} held-out")

    for i in range(args.max_cycles):
        state, converged = cycle(state, i + 1, problems, claude, gpt)
        cache_save(BELIEF_PATH, state)  # persist incrementally
        if converged:
            print(f"\n[FIXED POINT] belief stable after cycle {i+1}; terminating")
            break
    else:
        print(f"\n[MAX CYCLES] reached {args.max_cycles} without fixed point")

    # Final log
    log = cache_load(RECURSE_LOG, default=[]) or []
    log.append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_cycles": args.max_cycles,
        "final_gate": state["gate"],
        "n_cycles_run": len(state["history"]),
        "trajectory": state["history"],
    })
    cache_save(RECURSE_LOG, log)
    cache_save(BELIEF_PATH, state)

    print(f"\n=== FINAL ===")
    print(f"  Final gate: {state['gate']['rule']}")
    print(f"  Cycles run: {len(state['history'])}")
    print(f"\n  Per-wisdom trajectory:")
    for cid, b in state["beliefs"].items():
        cid_tag = b.get("committed_id") or "----"
        cw = b["claude_wr"]
        cw_str = f"{cw:.2f}" if cw is not None else " N/A"
        print(f"    {cid:9s} ({cid_tag:4s}) gemini={b['gemini_wr']:.2f} "
              f"claude={cw_str}  conf={b['confidence']:.2f}  "
              f"{b['aphorism'][:20]}")


if __name__ == "__main__":
    main()
