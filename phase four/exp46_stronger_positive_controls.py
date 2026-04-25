"""Exp 46 — Stronger positive controls for the audit stack.

Exp 44 found that all 3 controls (placebo / random / generic-useful)
scored wr 0.06-0.18, suggesting the gate has high specificity but
not testing whether the gate has SENSITIVITY (would it accept a
genuinely useful library addition?).

We design THREE stronger positive controls that target specific
failure modes a methodology paper would care about:

  C-MATH-NUMERIC: a math/quant-specific hint about explicit unit-
    tracking and order-of-magnitude sanity-check. Evaluated ONLY
    on math + science problems where this is plausibly useful.

  C-CASE-STRUCTURE: a structural hint about laying out
    'situation / complication / question / answer' (the SCQA
    framework) explicitly before solving. Generic but operational.

  C-LIBRARY-DUPLICATE: a duplicate of an EXISTING base-library
    wisdom that v20 is already known to use successfully. Hardest
    positive control: if the gate doesn't accept this, the gate
    can't accept anything (since the wisdom is already part of
    what gives v20 its 0.86 win rate).

All three eval on sample_extend_50 (where cached base lives).
3 cheap-tier judges. Cost: ~$8, ~30 min.
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))


def _load_api_keys():
    if os.environ.get("RUOLI_GPT_KEY") and os.environ.get("RUOLI_BASE_URL"):
        return
    keyfile = Path.home() / ".api_keys"
    if not keyfile.exists():
        return
    pat = re.compile(r'^\s*export\s+(\w+)=("([^"]*)"|\'([^\']*)\'|(\S+))')
    for line in keyfile.read_text().splitlines():
        m = pat.match(line)
        if not m: continue
        name = m.group(1)
        val = m.group(3) if m.group(3) is not None else (
              m.group(4) if m.group(4) is not None else m.group(5))
        os.environ.setdefault(name, val)
        if name == "RUOLI_BASE_URL":
            base = val + "/v1" if not val.endswith("/v1") else val
            os.environ.setdefault("CLAUDE_PROXY_BASE_URL", base)
            os.environ.setdefault("GPT5_BASE_URL", base)
            os.environ.setdefault("GEMINI_PROXY_BASE_URL", base)
        if name == "RUOLI_GEMINI_KEY":
            os.environ.setdefault("GEMINI_PROXY_API_KEY", val)
        if name == "RUOLI_GPT_KEY":
            os.environ.setdefault("GPT5_API_KEY", val)
        if name == "RUOLI_CLAUDE_KEY":
            os.environ.setdefault("CLAUDE_PROXY_API_KEY", val)

_load_api_keys()

from model_router import cheap, cheap_panel
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
ANS = CACHE / "answers"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp46_stronger_positive_controls_log.json"

PARALLEL = 6


# ---- 3 stronger positive controls ----
CONTROLS = [
    {
        "tid": "C-MATH-NUMERIC",
        "label": "math/quant-specific: explicit units + order-of-magnitude sanity",
        "aphorism": "先核单位，再核量级，最后核数字",
        "source": "（人工设计的方法论控制 — 量化推理）",
        "signal": "面对数学/量化估算/工程计算问题",
        "unpacked_for_llm":
            "对量化问题：(1) 在每一步显式标注单位（米、秒、人/天等），"
            "用单位一致性作为初步检验；(2) 在最终结果出来前，先估算"
            "数量级（10^?），用与已知锚点比较确认合理；"
            "(3) 最后再呈现具体数字。"
            "适用于数学、物理、工程、量化估算类问题；不适用于业务策略、"
            "社会科学、文学等非数值推理问题。",
        "eval_filter": ["mathematics", "science", "engineering"],  # only on quant-likely domains
    },
    {
        "tid": "C-SCQA",
        "label": "SCQA: situation / complication / question / answer",
        "aphorism": "先理 situation，再点 complication，定 question，给 answer",
        "source": "Barbara Minto's Pyramid Principle (consultancy SCQA)",
        "signal": "面对开放性、需要清晰呈递的咨询/分析/汇报型问题",
        "unpacked_for_llm":
            "在解答前，先用 SCQA 框架展开问题：(S) Situation：现状/背景"
            "中可观察的事实是什么？(C) Complication：当前出现了什么变化"
            "或冲突？(Q) Question：基于 S+C，需要回答的具体问题是什么？"
            "(A) Answer：回答这个具体问题，并按金字塔结构展开论据。"
            "这是一个有充分文献支持的分析-呈递框架（Barbara Minto, "
            "McKinsey 等），适用于咨询、商业分析、政策建议等开放型问题。",
        "eval_filter": None,  # all domains
    },
    {
        "tid": "C-LIBRARY-DUP",
        "label": "duplicate of an existing base-library wisdom (W016)",
        # Use W016 from the actual wisdom library — pick one we know v20 uses
        "aphorism": "兼听则明，偏信则暗",
        "source": "《资治通鉴》",
        "signal": "面对单一信息源、当事人、单一视角的判断时",
        "unpacked_for_llm":
            "在做判断前，主动收集与当前结论矛盾的证据或不同立场的视角。"
            "如果当前判断只基于单一信息源、单一当事人、或单一立场，"
            "在做最终决策前，应至少考虑一个反方意见或独立的证据来源。"
            "这是 v20 base library 中已经在用的方法论；作为正向控制，"
            "如果 audit stack 拒绝它，则 audit stack 的 sensitivity 极低。",
        "eval_filter": None,
    },
]


FRAME_PROMPT = """对下面问题产生 frame + 重写。

## 原题
{problem}

## 输出 JSON
{{"frame": "object_level/paradigm/hybrid",
  "critical_reframe": "30-80字",
  "rewritten_problem": "120-250字"}}
"""

EXECUTE_PROMPT = """# 解决问题

## PRIMARY FRAME
- frame: {frame}
- critical reframe: {critical_reframe}

## 问题（重写）
{rewritten_problem}

## 次要参考 wisdom
{wisdom_block}

## 要求：≤ 500 字
开始："""

JUDGE_PROMPT = """方法论评审.

## 问题
{problem}

## 解答 A
{answer_a}

## 解答 B
{answer_b}

Output JSON: {{"winner": "A"/"B"/"tie", "score_a": 1-10, "score_b": 1-10,
  "reasoning": "80 chars max"}}
"""


def cache_load(p, default=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except: return default


def load_problems():
    pid_to_prob = {}
    for f in (PROJECT / "phase zero" / "benchmark" / "problems").glob("*.json"):
        for q in json.loads(f.read_text()):
            pid_to_prob[q["problem_id"]] = q.get("description") or q.get("problem") or ""
    return pid_to_prob


def domain_of(pid):
    return pid.rsplit("_", 1)[0]


def solve(client, problem, wisdoms):
    try:
        r = client.generate(FRAME_PROMPT.format(problem=problem),
                            max_tokens=500, temperature=0.2)
        m = parse_json_from_llm(r["text"])
    except Exception as e:
        return f"[turn0 err: {e}]"
    if wisdoms:
        wb = "\n".join(f"• {w['aphorism']}: {w.get('unpacked_for_llm','')[:200]}"
                        for w in wisdoms)
    else:
        wb = "(无)"
    try:
        r = client.generate(EXECUTE_PROMPT.format(
            frame=m.get("frame", "object_level"),
            critical_reframe=m.get("critical_reframe", ""),
            rewritten_problem=m.get("rewritten_problem", problem),
            wisdom_block=wb), max_tokens=900, temperature=0.2)
        return r["text"].strip()
    except Exception as e:
        return f"[turn1 err: {e}]"


def judge_one(client, problem, a, b):
    try:
        r = client.generate(JUDGE_PROMPT.format(problem=problem, answer_a=a, answer_b=b),
                            max_tokens=300, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return v.get("winner", "tie")
    except Exception:
        return "err"


def main():
    pid_to_prob = load_problems()
    eval_raw = json.loads((CACHE / "sample_extend_50.json").read_text())
    eval_set = [p for p in eval_raw if "description" in p]
    print(f"Eval set (sample_extend_50): {len(eval_set)} pids\n")

    base_ans = json.loads((ANS / "_exp10_v20_base_answers.json").read_text())
    common_pids = sorted(set(p["problem_id"] for p in eval_set) & set(base_ans))
    print(f"Base coverage: {len(common_pids)}\n")

    judges = cheap_panel()
    solver = cheap("gemini")
    print(f"Solver: {solver.model}, judges: {[j.model for j in judges]}\n")

    # Generate ext per (control, pid) — but filter to control's eval_filter if specified
    all_tasks = []
    for c in CONTROLS:
        if c["eval_filter"]:
            pids = [pid for pid in common_pids if domain_of(pid) in c["eval_filter"]]
            print(f"  {c['tid']}: filtered to {len(pids)} pids in {c['eval_filter']}")
        else:
            pids = common_pids
        for pid in pids:
            all_tasks.append((c, pid))

    print(f"\n[1/2] Generating {len(all_tasks)} ext answers...")
    ext_answers = {c["tid"]: {} for c in CONTROLS}

    def gen(c, pid):
        prob = next((p for p in eval_set if p["problem_id"] == pid), None)
        if prob is None: return c["tid"], pid, "[no prob]"
        return c["tid"], pid, solve(solver, prob["description"], [c])

    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(gen, c, pid) for c, pid in all_tasks]
        for f in as_completed(futs):
            tid, pid, ans = f.result()
            ext_answers[tid][pid] = ans
            done += 1
            if done % 30 == 0:
                print(f"  gen {done}/{len(all_tasks)} ({time.time()-t0:.0f}s)")

    # Judge each (control, pid) under all 3 cheap judges
    print(f"\n[2/2] Judging across 3 cheap families...")
    verdicts = {c["tid"]: {j.family: {} for j in judges} for c in CONTROLS}

    def jt(judge, c, pid):
        prob = next((p for p in eval_set if p["problem_id"] == pid), None)
        if prob is None: return judge.family, c["tid"], pid, "missing"
        b = base_ans.get(pid, ""); e = ext_answers[c["tid"]].get(pid, "")
        if not b or not e or b.startswith("[") or e.startswith("["):
            return judge.family, c["tid"], pid, "missing"
        rng = random.Random((hash(pid) ^ hash(c["tid"])) % (2**32))
        if rng.random() < 0.5:
            left, right, ext_was = e, b, "A"
        else:
            left, right, ext_was = b, e, "B"
        w = judge_one(judge, prob["description"], left, right)
        if w == "tie": v = "tie"
        elif w in ("A", "B"): v = "ext" if w == ext_was else "base"
        else: v = "err"
        return judge.family, c["tid"], pid, v

    jtasks = [(j, c, pid) for j in judges for c in CONTROLS
                for pid in (ext_answers[c["tid"]].keys())]
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(jt, *t) for t in jtasks]
        for f in as_completed(futs):
            fam, tid, pid, v = f.result()
            verdicts[tid][fam][pid] = v
            done += 1
            if done % 50 == 0:
                print(f"  judge {done}/{len(jtasks)} ({time.time()-t0:.0f}s)")

    # Aggregate
    print(f"\n=== Stronger positive control results ===")
    print(f"{'tid':18s} {'label':50s}  {'gemini':>8s} {'haiku':>8s} {'mini':>8s}  3-fam mean")
    print("-" * 100)
    summary = []
    for c in CONTROLS:
        wrs = []
        line = f"{c['tid']:18s} {c['label'][:48]:50s}  "
        per_fam = {}
        for j in judges:
            v = verdicts[c["tid"]][j.family]
            ne = sum(1 for x in v.values() if x == "ext")
            nb = sum(1 for x in v.values() if x == "base")
            nt = sum(1 for x in v.values() if x == "tie")
            tot = ne + nb
            wr = ne / tot if tot else 0.5
            wrs.append(wr)
            per_fam[j.family] = {"wr": wr, "ext": ne, "base": nb, "tie": nt, "n_eff": tot}
            line += f"  {wr:>6.2f}"
        mean_wr = sum(wrs) / len(wrs)
        line += f"   {mean_wr:.2f}"
        gate = "PASS" if any(w >= 0.60 for w in wrs) else "REVERT"
        line += f"  ({gate})"
        print(line)
        summary.append({"tid": c["tid"], "label": c["label"],
                          "aphorism": c["aphorism"],
                          "per_family": per_fam,
                          "3_fam_mean": mean_wr,
                          "any_family_above_0.60": any(w >= 0.60 for w in wrs)})

    print(f"\n=== Sensitivity verdict ===")
    n_pass = sum(1 for s in summary if s["any_family_above_0.60"])
    print(f"  Controls with ≥1 family wr ≥ 0.60: {n_pass}/{len(CONTROLS)}")
    if n_pass == 0:
        print(f"  Audit specificity high (all controls rejected); SENSITIVITY remains untested.")
    else:
        passing = [s["tid"] for s in summary if s["any_family_above_0.60"]]
        print(f"  Audit DOES accept some genuinely useful additions: {passing}")
        print(f"  This shows the gate has discrimination beyond rejecting random text.")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "controls": CONTROLS, "summary": summary, "verdicts": verdicts}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
