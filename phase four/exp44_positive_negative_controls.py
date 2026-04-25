"""Exp 44 — Positive / negative controls for the audit stack.

Closes review weakness #4 (no audit-stack calibration). We construct
three control "wisdoms" with different known properties and run each
through the inner-loop +10pp gate AND the L1 cross-family audit:

  C-PLACEBO    : a methodologically vacuous aphorism unrelated to the
                  problem domain (filler text matched in length to a
                  real wisdom record). Should NOT pass the gate; if
                  the inner-loop gate accepts it, the gate has high
                  false-positive rate.

  C-RANDOM     : a length-matched random literary quote with no
                  methodological content. Same expectation.

  C-USEFUL     : a deliberately useful generic methodological hint
                  ("write down all assumptions before solving;
                  identify the type of problem; check units").
                  Should plausibly improve answers if the gate has
                  any utility-detection power.

For each control we report:
  - inner-loop wr_ext under +10pp gate (gemini judge, n=50)
  - L1 cross-family wr (claude-haiku judge, the cheap-tier
    re-judging used elsewhere)

If C-USEFUL has wr_inner < 0.60 (rejected) AND C-PLACEBO/RANDOM
also have wr_inner < 0.60 (correctly rejected), the audit's main
finding strengthens: even a USEFUL control fails the +10pp gate
under same-family judging.

If C-USEFUL passes the gate but C-PLACEBO does not, the gate has
discrimination, and the original three KEEPs' failures are more
likely judge-fragility than "the gate would reject anything."

Cost: 3 controls x 50 pids x 1 (ext) generations + 3 x 50 x 2 judges
= ~450 calls. ~$5, ~30 min.
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

from model_router import cheap
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
ANS = CACHE / "answers"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp44_positive_negative_controls_log.json"

PARALLEL = 6


# ---- 3 control "wisdoms" ----
CONTROLS = [
    {
        "tid": "C-PLACEBO",
        "label": "placebo (vacuous methodological-sounding nonsense)",
        "aphorism": "万事互通，举一反三",  # all things connect, learn three from one
        "source": "民间俗语",
        "signal": "面对任何问题",  # "face any problem" — vacuously universal
        "unpacked_for_llm":
            "在思考问题时保持开放的心态，将不同领域的知识联系起来，"
            "通过类比和迁移学习以达到融会贯通的境界。",
        # ^ filler that sounds wise but says nothing operational
    },
    {
        "tid": "C-RANDOM",
        "label": "random literary quote (no methodological content)",
        "aphorism": "落霞与孤鹜齐飞，秋水共长天一色",  # Tang poetry — pure aesthetic, no methodology
        "source": "王勃《滕王阁序》",
        "signal": "审美场景",
        "unpacked_for_llm":
            "在思考时欣赏问题本身的美学结构，让灵感自然涌现。"
            "保持心境的开阔与自由。",
    },
    {
        "tid": "C-USEFUL",
        "label": "deliberately useful methodological hint",
        "aphorism": "解题三步：列前提、定指标、再推论",  # solve in 3 steps: list premises, define metrics, then infer
        "source": "（人工设计的方法论控制）",
        "signal": "面对需要明确推理链路的问题",
        "unpacked_for_llm":
            "解题前明确列出：(1) 已知的事实和约束（前提），(2) 衡量解答质量的"
            "可观察指标（成功标准），(3) 在前提下基于指标做推论。"
            "避免在隐含假设上跳跃；避免没有衡量标准的推荐。",
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


def solve(client, problem, wisdoms):
    try:
        r = client.generate(FRAME_PROMPT.format(problem=problem),
                            max_tokens=500, temperature=0.2)
        m = parse_json_from_llm(r["text"])
    except Exception as e:
        return f"[turn0 err: {e}]"
    if wisdoms:
        wb = "\n".join(f"• {w['aphorism']}: {w.get('unpacked_for_llm','')[:180]}"
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


def wilson(k, n):
    from scipy.stats import binomtest
    if n == 0: return (0.5, 0.5)
    r = binomtest(k, n).proportion_ci(method="wilson")
    return r.low, r.high


def main():
    pid_to_prob = load_problems()
    # The original Exp 10 evaluated on sample_extend_50 (cached base lives there);
    # use the same eval set so we can reuse cached base answers
    eval_raw = json.loads((CACHE / "sample_extend_50.json").read_text())
    holdout = [p for p in eval_raw if "description" in p]
    print(f"Eval set (sample_extend_50, where Exp 10 base lives): {len(holdout)} pids\n")

    base_ans = json.loads((ANS / "_exp10_v20_base_answers.json").read_text())
    common_pids = sorted(set(p["problem_id"] for p in holdout) & set(base_ans))
    print(f"Base coverage on eval set: {len(common_pids)}\n")

    solver = cheap("gemini")  # same as inner loop
    haiku = cheap("claude_haiku")  # for L1 cross-family
    print(f"Solver: {solver.model}, L1 audit judge: {haiku.model}\n")

    # Generate ext answers per control
    print(f"[1/2] Generating ext answers ({len(CONTROLS)} x {len(common_pids)} = "
          f"{len(CONTROLS)*len(common_pids)} calls)...")
    ext_answers = {c["tid"]: {} for c in CONTROLS}

    def gen(c, pid):
        prob = next((p for p in holdout if p["problem_id"] == pid), None)
        if prob is None: return c["tid"], pid, "[missing prob]"
        return c["tid"], pid, solve(solver, prob["description"], [c])

    tasks = [(c, pid) for c in CONTROLS for pid in common_pids]
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(gen, c, pid) for c, pid in tasks]
        for f in as_completed(futs):
            tid, pid, ans = f.result()
            ext_answers[tid][pid] = ans
            done += 1
            if done % 30 == 0:
                print(f"  gen {done}/{len(tasks)} ({time.time()-t0:.0f}s)")

    # Judge each ext vs base with both gemini (inner-loop) and claude-haiku (L1)
    print(f"\n[2/2] Judging ({len(CONTROLS)} x {len(common_pids)} x 2 judges = "
          f"{len(CONTROLS)*len(common_pids)*2} calls)...")
    verdicts = {c["tid"]: {"gemini": {}, "haiku": {}} for c in CONTROLS}

    def jt(judge, jname, c, pid):
        prob = next((p for p in holdout if p["problem_id"] == pid), None)
        if prob is None: return c["tid"], jname, pid, "missing"
        b = base_ans.get(pid, ""); e = ext_answers[c["tid"]].get(pid, "")
        if not b or not e or b.startswith("[") or e.startswith("["):
            return c["tid"], jname, pid, "missing"
        rng = random.Random((hash(pid) ^ hash(c["tid"])) % (2**32))
        if rng.random() < 0.5:
            left, right, ext_was = e, b, "A"
        else:
            left, right, ext_was = b, e, "B"
        w = judge_one(judge, prob["description"], left, right)
        if w == "tie": v = "tie"
        elif w in ("A", "B"): v = "ext" if w == ext_was else "base"
        else: v = "err"
        return c["tid"], jname, pid, v

    jtasks = [(judge, jname, c, pid)
                for c in CONTROLS for pid in common_pids
                for judge, jname in [(solver, "gemini"), (haiku, "haiku")]]
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(jt, *t) for t in jtasks]
        for f in as_completed(futs):
            tid, jname, pid, v = f.result()
            verdicts[tid][jname][pid] = v
            done += 1
            if done % 50 == 0:
                print(f"  judge {done}/{len(jtasks)} ({time.time()-t0:.0f}s)")

    # Aggregate
    print(f"\n=== Audit-stack calibration on 3 controls ===")
    print(f"{'tid':14s}  {'label':45s}  {'wr_inner':>10s}  {'wr_L1':>8s}  {'inner gate':12s}")
    print("-" * 100)
    summary = []
    for c in CONTROLS:
        v_g = verdicts[c["tid"]]["gemini"]
        v_h = verdicts[c["tid"]]["haiku"]
        ne_g = sum(1 for x in v_g.values() if x == "ext")
        nb_g = sum(1 for x in v_g.values() if x == "base")
        nt_g = sum(1 for x in v_g.values() if x == "tie")
        ne_h = sum(1 for x in v_h.values() if x == "ext")
        nb_h = sum(1 for x in v_h.values() if x == "base")
        nt_h = sum(1 for x in v_h.values() if x == "tie")
        wr_g = ne_g / (ne_g + nb_g) if ne_g + nb_g else 0.5
        wr_h = ne_h / (ne_h + nb_h) if ne_h + nb_h else 0.5
        gate = "PASS" if wr_g >= 0.60 else "REVERT"
        print(f"{c['tid']:14s}  {c['label'][:43]:45s}  {wr_g:>10.2f}  {wr_h:>8.2f}  "
              f"{gate:12s}")
        summary.append({
            "tid": c["tid"], "label": c["label"], "aphorism": c["aphorism"],
            "wr_inner_gemini": wr_g, "wr_L1_haiku": wr_h,
            "inner_gate_decision": gate,
            "n_eff_inner": ne_g + nb_g, "ties_inner": nt_g,
            "n_eff_L1": ne_h + nb_h, "ties_L1": nt_h,
            "ext_inner": ne_g, "base_inner": nb_g,
            "ext_L1": ne_h, "base_L1": nb_h,
        })

    print(f"\n=== Calibration interpretation ===")
    placebo_passes = any(s["inner_gate_decision"] == "PASS" and s["tid"].startswith("C-PLACEBO") or s["tid"] == "C-RANDOM" for s in summary)
    useful = next((s for s in summary if s["tid"] == "C-USEFUL"), None)
    placebo = next((s for s in summary if s["tid"] == "C-PLACEBO"), None)
    randomc = next((s for s in summary if s["tid"] == "C-RANDOM"), None)

    print(f"  C-PLACEBO inner-loop wr: {placebo['wr_inner_gemini']:.2f}, "
          f"gate: {placebo['inner_gate_decision']}")
    print(f"  C-RANDOM inner-loop wr: {randomc['wr_inner_gemini']:.2f}, "
          f"gate: {randomc['inner_gate_decision']}")
    print(f"  C-USEFUL inner-loop wr: {useful['wr_inner_gemini']:.2f}, "
          f"gate: {useful['inner_gate_decision']}")

    print(f"\n  Compare to original KEEPs:")
    print(f"    W076 (kept): inner=0.64, L1 (Opus)=0.40")
    print(f"    W077 (kept): inner=0.60, L1 (Opus)=0.475")
    print(f"    W078 (kept): inner=0.60, L1 (Opus)=0.51")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "controls": CONTROLS,
           "summary": summary,
           "verdicts": verdicts}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
