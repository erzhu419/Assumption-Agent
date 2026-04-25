"""Exp 40 — Cross-solver L4 FULL audit on the 3 KEEPs.

Closes review weakness #6 partially: every previous answer text in
the audit chain came from gemini-3-flash. We regenerate base + ext
answers for W076/W077/W078 with two different solver families
(claude-haiku-4.5 and gpt-5.4-mini) on the same 50 holdout pids,
then judge each (base, ext) pair with the same 3-family cheap
panel (gemini-3-flash, claude-haiku, gpt-5.4-mini).

For each (KEEP, solver, judge) triple we report wr_ext + 95%
Wilson CI. The headline question: does cross-family-judge
fragility persist across solver families, or is it a gemini-
solver artefact?

Cost: 3 KEEPs x 2 solvers x 50 pids x 2 (base+ext) = 600 gen
calls, plus 3 KEEPs x 2 solvers x 50 pids x 3 judges = 900
judge calls. Cheap-tier throughout. Estimated: ~$15, ~45 min.
"""

import json
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from model_router import cheap, cheap_panel
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
ANS = CACHE / "answers"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp40_cross_solver_full_audit_log.json"

PARALLEL = 6

# 3 KEEPs
KEEPS = [
    {"id": "W076", "cid": "WCAND05", "aphorism": "凡益之道，与时偕行",
     "source": "《周易》",
     "signal": "环境/认知/工艺正在变迁但旧规则还在沿用",
     "unpacked_for_llm": "判断当前问题是否处在一个旧框架的合理边界正在收窄"
                          "的时刻；如果是，先让框架本身随时代调整，再在新框架"
                          "内做对象级决策。"},
    {"id": "W077", "cid": "WCAND10", "aphorism": "没有调查，就没有发言权",
     "source": "毛泽东《反对本本主义》",
     "signal": "多因混杂、观察噪声大、没定位清楚就已经要决策时",
     "unpacked_for_llm": "在多因素混杂、观察噪声大的情境里，先建复现/归因"
                          "链路再下结论；调研充分前不要做大决策。"},
    {"id": "W078", "cid": "WCROSSL01", "aphorism": "是骡子是马，拉出来遛遛",
     "source": "民间俗语",
     "signal": "当多种方案/模型/工具看起来差不多但需要择一时",
     "unpacked_for_llm": "列候选、定指标、做检验、再决定；用便宜的discriminating"
                          "test 替代昂贵的全量试验，避免在不可观测的口头论证里"
                          "提前 commit。"},
]

# 2 new solver families (NOT gemini)
SOLVER_FAMILIES = ["claude_haiku", "gpt_mini"]

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
    except Exception as e:
        return f"err:{str(e)[:30]}"


def wilson(k, n):
    from scipy.stats import binomtest
    if n == 0: return 0.5, 0.5
    r = binomtest(k, n).proportion_ci(method="wilson")
    return r.low, r.high


def main():
    # Load 50 cached gemini base/ext pid set so we use the same pid sample
    base_orig = json.loads((ANS / "_exp10_v20_base_answers.json").read_text())
    pid_to_prob = load_problems()
    pids = sorted(set(base_orig) & set(pid_to_prob))
    print(f"Using {len(pids)} pids from cached Exp 10 holdout sample\n")

    judges = cheap_panel()
    print(f"Judge panel: {[j.model for j in judges]}\n")

    # Stage 1: generate base + ext answers per (solver, keep, pid)
    answers = {sf: {kp["id"]: {"base": {}, "ext": {}} for kp in KEEPS}
                for sf in SOLVER_FAMILIES}

    def gen_task(sf, kp, pid, kind):
        client = cheap(sf)
        wisdoms = [] if kind == "base" else [kp]
        return sf, kp["id"], pid, kind, solve(client, pid_to_prob[pid], wisdoms)

    tasks = []
    for sf in SOLVER_FAMILIES:
        for kp in KEEPS:
            for pid in pids:
                tasks.append((sf, kp, pid, "base"))
                tasks.append((sf, kp, pid, "ext"))

    print(f"[1/2] Generating {len(tasks)} answers across {len(SOLVER_FAMILIES)} "
          f"solver families x {len(KEEPS)} KEEPs x {len(pids)} pids x 2 (base+ext)")
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(gen_task, sf, kp, pid, kind) for sf, kp, pid, kind in tasks]
        for f in as_completed(futs):
            sf, kid, pid, kind, ans = f.result()
            answers[sf][kid][kind][pid] = ans
            done += 1
            if done % 50 == 0:
                print(f"  gen {done}/{len(tasks)} ({time.time()-t0:.0f}s)")

    # Stage 2: judge each (KEEP, solver) pair triple with all 3 cheap judges
    print(f"\n[2/2] Judging {len(SOLVER_FAMILIES)*len(KEEPS)*len(pids)*len(judges)} pairs")
    verdicts = {sf: {kp["id"]: {j.family: {} for j in judges} for kp in KEEPS}
                 for sf in SOLVER_FAMILIES}

    def judge_task(judge, sf, kp, pid):
        kid = kp["id"]
        b = answers[sf][kid]["base"].get(pid, "")
        e = answers[sf][kid]["ext"].get(pid, "")
        if not b or not e or b.startswith("[") or e.startswith("["):
            return judge.family, sf, kid, pid, "missing"
        rng = random.Random((hash(pid) ^ hash(sf) ^ hash(kid)) % (2**32))
        if rng.random() < 0.5:
            left, right, ext_was = e, b, "A"
        else:
            left, right, ext_was = b, e, "B"
        w = judge_one(judge, pid_to_prob[pid], left, right)
        if w == "tie": v = "tie"
        elif w in ("A", "B"): v = "ext" if w == ext_was else "base"
        else: v = "err"
        return judge.family, sf, kid, pid, v

    jtasks = [(j, sf, kp, pid) for j in judges for sf in SOLVER_FAMILIES
                for kp in KEEPS for pid in pids]
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(judge_task, *t) for t in jtasks]
        for f in as_completed(futs):
            fam, sf, kid, pid, v = f.result()
            verdicts[sf][kid][fam][pid] = v
            done += 1
            if done % 100 == 0:
                print(f"  judge {done}/{len(jtasks)} ({time.time()-t0:.0f}s)")

    # Aggregate
    print(f"\n=== Cross-solver L4 FULL audit: wr_ext per (KEEP, solver, judge) ===")
    print(f"{'KEEP':6s} {'solver':14s}  " +
          "  ".join(f"{j.family[:12]:>12s}" for j in judges) + "  mean(3-family)")
    print("-" * 90)
    summary = {}
    for sf in SOLVER_FAMILIES:
        for kp in KEEPS:
            kid = kp["id"]
            fam_wrs = {}
            for j in judges:
                v = verdicts[sf][kid][j.family]
                ne = sum(1 for x in v.values() if x == "ext")
                nb = sum(1 for x in v.values() if x == "base")
                tot = ne + nb
                wr = ne / tot if tot else 0.5
                lo, hi = wilson(ne, tot) if tot else (0.5, 0.5)
                fam_wrs[j.family] = {"wr": wr, "lo": lo, "hi": hi, "n": tot, "ext": ne, "base": nb,
                                       "tie": sum(1 for x in v.values() if x == "tie")}
            mean_wr = sum(d["wr"] for d in fam_wrs.values()) / len(fam_wrs)
            summary[(sf, kid)] = {"per_family": fam_wrs, "mean_wr": mean_wr}
            line = f"{kid:6s} {sf:14s}  "
            for j in judges:
                line += f"  {fam_wrs[j.family]['wr']:>10.2f}  "
            line += f"  {mean_wr:.2f}"
            print(line)

    # Compare to gemini-solver baseline (Exp 10 + Exp 36 cheap-judge re-runs)
    print(f"\n=== For comparison: gemini-solver inner-loop and cheap re-runs ===")
    print(f"  W076 (gemini solver, gemini judge): inner-loop 0.64; haiku 0.41 (Exp 1); haiku re-run 0.41 (Exp 36)")
    print(f"  W077 (gemini solver): inner 0.60; haiku 0.42; haiku re-run 0.50")
    print(f"  W078 (gemini solver): inner 0.60; haiku 0.51; haiku re-run 0.51")

    # Headline: across the 18 (KEEP, solver, judge) cells, how many >= 0.60?
    n_cells_above_keep = sum(1 for d in summary.values()
                              for j in d["per_family"]
                              if d["per_family"][j]["wr"] >= 0.60)
    n_total_cells = sum(1 for d in summary.values() for _ in d["per_family"])
    print(f"\n=== Headline ===")
    print(f"  ({n_cells_above_keep}/{n_total_cells}) (KEEP, solver, judge) cells reach wr_ext >= 0.60")
    print(f"  ({sum(1 for d in summary.values() if all(f['wr'] >= 0.60 for f in d['per_family'].values()))}"
          f"/{len(summary)}) (KEEP, solver) pairs have ALL three judges >= 0.60")

    # Save
    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n_pids": len(pids),
           "solver_families": SOLVER_FAMILIES,
           "judges": [j.model for j in judges],
           "summary": {f"{sf}__{kid}":
                        {"per_family": {f: {k:v for k,v in d.items()}
                                          for f, d in info["per_family"].items()},
                         "mean_wr": info["mean_wr"]}
                        for (sf, kid), info in summary.items()},
           "verdicts": verdicts}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
