"""Exp 54 — Fresh n=100 evaluation under v20-STYLE exemplar protocol
(closes the standing protocol-mismatch concern: Exp 53 used a clean
no-exemplar solver, but the original cached gate ran v20 with
same-domain in-pool exemplars).

The reviewer's persistent question was: would the n=100 null still
hold under the original v20+exemplar pipeline, or is the cached signal
specifically exemplar-dependent? This experiment answers it on the
SAME 100 fresh pids as Exp 53.

Protocol (Exp 53 + same-domain exemplar augmentation):
  - Reuse the 100 fresh pids and base answers from Exp 53
  - Per pid: find most-similar SAME-DOMAIN neighbour in the 100-pid
    fresh pool (excluding self); use its Exp 53 base answer as the
    in-pool exemplar (mirrors v20's `build_same_domain_exemplar`)
  - Generate base + ext answers WITH the exemplar block in BOTH
    conditions (preserving symmetry between base and ext, matching
    v20's design)
  - Inner judge: gemini-3-flash; L1 judge: claude-haiku
  - Frozen thresholds: inner >= 0.60, L1 >= 0.55 (preregistered)

Comparison to Exp 53 (no exemplar) on the SAME pids isolates the
exemplar mechanism's contribution to win rate.

Cost: 100 base-with-exemplar + 1200 ext-with-exemplar + 1200 inner
judge + (depends on inner-passers) L1 judge. ~$20 cheap-tier.
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
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp54_n100_v20style_exemplar_log.json"

INNER_THRESHOLD = 0.60
L1_THRESHOLD = 0.55
PARALLEL = 6


def cache_load(p, default=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except: return default


# ---------- prompts ----------
FRAME_PROMPT = """对下面问题产生 frame + 重写。

## 原题
{problem}

## 输出 JSON
{{"frame": "object_level/paradigm/hybrid",
  "critical_reframe": "30-80字",
  "rewritten_problem": "120-250字"}}
"""

EXECUTE_PROMPT_WITH_EXEMPLAR = """# 解决问题

## PRIMARY FRAME
- frame: {frame}
- critical reframe: {critical_reframe}

## 问题（重写）
{rewritten_problem}

## 同域参考案例（来自相似问题的解题思路示范）
- 相似问题: {exemplar_problem}
- 参考解答摘要: {exemplar_answer_snippet}

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


def domain_of(pid):
    return pid.rsplit("_", 1)[0]


def find_same_domain_exemplar(pid, fresh_pids, embeddings, pid_to_idx):
    """Return (other_pid) — most similar same-domain pid (excluding self)."""
    dom = domain_of(pid)
    same_dom = [p for p in fresh_pids if domain_of(p) == dom and p != pid]
    if not same_dom:
        # Fall back: cross-domain nearest neighbour
        same_dom = [p for p in fresh_pids if p != pid]
    qv = embeddings[pid_to_idx[pid]]
    best_sim, best_pid = -1.0, None
    for op in same_dom:
        sim = float(qv @ embeddings[pid_to_idx[op]])
        if sim > best_sim:
            best_sim, best_pid = sim, op
    return best_pid


def solve_with_exemplar(client, problem, exemplar_problem, exemplar_answer,
                          wisdom):
    try:
        r = client.generate(FRAME_PROMPT.format(problem=problem),
                            max_tokens=500, temperature=0.2)
        m = parse_json_from_llm(r["text"])
    except Exception as e:
        return f"[turn0 err: {e}]"
    if wisdom:
        wb = f"• {wisdom['aphorism']}: {wisdom.get('unpacked_for_llm', '')[:200]}"
    else:
        wb = "(无)"
    try:
        r = client.generate(EXECUTE_PROMPT_WITH_EXEMPLAR.format(
            frame=m.get("frame", "object_level"),
            critical_reframe=m.get("critical_reframe", ""),
            rewritten_problem=m.get("rewritten_problem", problem),
            exemplar_problem=exemplar_problem[:200],
            exemplar_answer_snippet=exemplar_answer[:500],
            wisdom_block=wb), max_tokens=900, temperature=0.2)
        return r["text"].strip()
    except Exception as e:
        return f"[turn1 err: {e}]"


def judge_one(client, problem, a, b):
    try:
        r = client.generate(JUDGE_PROMPT.format(problem=problem,
                                                  answer_a=a, answer_b=b),
                            max_tokens=300, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return v.get("winner", "tie")
    except Exception:
        return "err"


def load_candidates():
    val = json.loads((AUTO_DIR / "validation_log_parallel.json").read_text())
    cands = []
    for entry in val:
        for r in entry["results"]:
            cands.append({"tid": r["tid"], "aphorism": r["candidate"],
                          "source": r.get("source", "")})
    sd = cache_load(AUTO_DIR / "success_distilled_candidates.json", default=[])
    cl = cache_load(AUTO_DIR / "cross_llm_candidates.json", default=[])
    aph_to_rec = {r["aphorism"]: r for r in sd + cl if "aphorism" in r}
    for c in cands:
        rec = aph_to_rec.get(c["aphorism"], {})
        c["unpacked_for_llm"] = rec.get("unpacked_for_llm", c["aphorism"])
    return cands


def main():
    print(f"=== Exp 54: n=100 fresh evaluation under v20-style exemplar "
          f"protocol ===")

    # Load Exp 53 fresh pids and base answers (we mirror Exp 53's split)
    e53 = cache_load(AUTO_DIR / "exp53_n100_fresh_audit_log.json")
    if e53 is None or "fresh_pids" not in e53:
        print("ERROR: exp53 log not found"); return
    fresh_pids_list = e53["fresh_pids"]
    print(f"  Reusing the same {len(fresh_pids_list)} fresh pids as Exp 53")

    # Load problem descriptions
    pid_to_prob = {}
    for f in (PROJECT / "phase zero" / "benchmark" / "problems").glob("*.json"):
        for q in json.loads(f.read_text()):
            pid_to_prob[q["problem_id"]] = q.get("description", "")
    fresh_pids = [pid for pid in fresh_pids_list if pid in pid_to_prob]

    # Reuse Exp 53 BASE answers as the candidate exemplar source
    # (same-domain neighbour's base answer = "what a typical good
    # answer to a similar problem looks like")
    e53_base = e53.get("base_answers")
    if e53_base is None:
        # Try loading from inner verdicts log structure if needed
        # If not stored, regenerate base answers below
        pass

    # Embed problems for similarity search
    print(f"  Embedding {len(fresh_pids)} problems...")
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    texts = [pid_to_prob[pid] for pid in fresh_pids]
    embeddings = np.array(model.encode(texts, normalize_embeddings=True,
                                          show_progress_bar=False))
    pid_to_idx = {pid: i for i, pid in enumerate(fresh_pids)}

    # Pick same-domain exemplar pid for each fresh pid
    exemplar_map = {}
    for pid in fresh_pids:
        ex_pid = find_same_domain_exemplar(pid, fresh_pids, embeddings,
                                              pid_to_idx)
        exemplar_map[pid] = ex_pid

    # Step 1: regenerate BASE answers with exemplar augmentation
    # (Exp 53 base answers were generated WITHOUT exemplar; for v20-style
    # we need new base-with-exemplar answers).
    solver = cheap("gemini")
    haiku = cheap("claude_haiku")
    print(f"  Solver: {solver.model}; L1 judge: {haiku.model}\n")

    print(f"[1/4] Generating {len(fresh_pids)} BASE answers with same-domain "
          f"exemplar (no wisdom)...")
    base_answers = {}

    def gen_base(pid):
        ex_pid = exemplar_map[pid]
        ex_prob = pid_to_prob.get(ex_pid, "")
        # For the very first base round, exemplar answers don't exist yet.
        # Bootstrap: use a placeholder same-domain hint string. Then in
        # round 2 use actual base answers we just generated. We do a
        # 2-pass: first pass with no-exemplar (= Exp 53 reuse), second
        # pass uses pass-1's outputs as exemplars.
        ex_ans = pass1_base.get(ex_pid, "(同域参考案例: 关注问题的核心约束并按合理顺序展开)")
        return pid, solve_with_exemplar(solver, pid_to_prob[pid], ex_prob,
                                          ex_ans, None)

    # Pass 1: regenerate base WITHOUT exemplar (just to seed pass 2)
    # — actually, we can directly reuse Exp 53's base verdicts to build
    # base answers. But Exp 53 verdicts log doesn't store full answers.
    # So we generate pass 1 fresh.
    print(f"  pass-1 (no-exemplar bootstrap base, parallel)...")
    pass1_base = {}
    def gen_pass1(pid):
        try:
            r = solver.generate(FRAME_PROMPT.format(problem=pid_to_prob[pid]),
                                  max_tokens=500, temperature=0.2)
            m = parse_json_from_llm(r["text"])
        except Exception as e:
            return pid, f"[turn0 err: {e}]"
        try:
            r = solver.generate(
                "# 解决问题\n## frame\n{}\n## 问题\n{}\n## 要求：≤500字\n开始：".format(
                    m.get("critical_reframe", ""),
                    m.get("rewritten_problem", pid_to_prob[pid])),
                max_tokens=900, temperature=0.2)
            return pid, r["text"].strip()
        except Exception as e:
            return pid, f"[turn1 err: {e}]"

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(gen_pass1, pid) for pid in fresh_pids]
        for i, f in enumerate(as_completed(futs), 1):
            pid, ans = f.result()
            pass1_base[pid] = ans
            if i % 25 == 0:
                print(f"    pass1 base {i}/{len(fresh_pids)} ({time.time()-t0:.0f}s)")

    print(f"  pass-2 (base WITH same-domain exemplar from pass-1)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(gen_base, pid) for pid in fresh_pids]
        for i, f in enumerate(as_completed(futs), 1):
            pid, ans = f.result()
            base_answers[pid] = ans
            if i % 25 == 0:
                print(f"    base {i}/{len(fresh_pids)} ({time.time()-t0:.0f}s)")

    # Step 2: ext answers with same-domain exemplar AND wisdom
    cands = load_candidates()
    print(f"\n[2/4] Generating {len(cands)} × {len(fresh_pids)} = "
          f"{len(cands)*len(fresh_pids)} ext answers (exemplar + wisdom)...")
    ext_answers = {c["tid"]: {} for c in cands}

    def gen_ext(c, pid):
        ex_pid = exemplar_map[pid]
        ex_prob = pid_to_prob.get(ex_pid, "")
        ex_ans = pass1_base.get(ex_pid, "(同域参考案例: 关注问题的核心约束并按合理顺序展开)")
        return c["tid"], pid, solve_with_exemplar(solver, pid_to_prob[pid],
                                                    ex_prob, ex_ans, c)

    tasks = [(c, pid) for c in cands for pid in fresh_pids]
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(gen_ext, c, pid) for c, pid in tasks]
        for f in as_completed(futs):
            tid, pid, ans = f.result()
            ext_answers[tid][pid] = ans
            done += 1
            if done % 60 == 0:
                print(f"  ext {done}/{len(tasks)} ({time.time()-t0:.0f}s)")

    # Step 3: inner judge
    print(f"\n[3/4] Inner-loop judge (gemini) on {len(cands)} × "
          f"{len(fresh_pids)} pairs...")
    inner_verdicts = {c["tid"]: {} for c in cands}

    def judge_inner(c, pid):
        b = base_answers.get(pid, ""); e = ext_answers[c["tid"]].get(pid, "")
        if not b or not e or b.startswith("[") or e.startswith("["):
            return c["tid"], pid, "missing"
        rng = random.Random((hash(pid) ^ hash(c["tid"])) % (2**32))
        if rng.random() < 0.5:
            left, right, ext_was = e, b, "A"
        else:
            left, right, ext_was = b, e, "B"
        w = judge_one(solver, pid_to_prob[pid], left, right)
        if w == "tie": v = "tie"
        elif w in ("A", "B"): v = "ext" if w == ext_was else "base"
        else: v = "err"
        return c["tid"], pid, v

    inner_tasks = [(c, pid) for c in cands for pid in fresh_pids]
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(judge_inner, c, pid) for c, pid in inner_tasks]
        for f in as_completed(futs):
            tid, pid, v = f.result()
            inner_verdicts[tid][pid] = v
            done += 1
            if done % 60 == 0:
                print(f"  inner judge {done}/{len(inner_tasks)} ({time.time()-t0:.0f}s)")

    inner_summary = []
    for c in cands:
        v = inner_verdicts[c["tid"]]
        ne = sum(1 for x in v.values() if x == "ext")
        nb = sum(1 for x in v.values() if x == "base")
        nt = sum(1 for x in v.values() if x == "tie")
        n_eff = ne + nb
        wr = ne / n_eff if n_eff else 0.5
        inner_pass = wr >= INNER_THRESHOLD
        inner_summary.append({"tid": c["tid"], "aphorism": c["aphorism"],
                                "ext": ne, "base": nb, "tie": nt,
                                "n_eff": n_eff, "wr_inner": wr,
                                "inner_pass": inner_pass})

    print(f"\n=== Inner-loop verdict on n=100 fresh data WITH exemplar ===")
    print(f"{'tid':12s} {'wr_inner_v20':>12s}  {'wr_inner_53':>11s}  delta  pass?")
    n_inner_pass = 0
    e53_inner = {s["tid"]: s["wr_inner"] for s in e53["summary"]}
    for s in inner_summary:
        wr_v20 = s["wr_inner"]
        wr_53 = e53_inner.get(s["tid"], None)
        delta = wr_v20 - wr_53 if wr_53 is not None else 0
        marker = "PASS" if s["inner_pass"] else "REVERT"
        print(f"  {s['tid']:12s} {wr_v20:>12.3f}  {wr_53:>11.3f}  "
              f"{delta:+.3f}  {marker}")
        if s["inner_pass"]:
            n_inner_pass += 1
    print(f"\n  {n_inner_pass}/{len(cands)} candidates pass inner gate at "
          f"{INNER_THRESHOLD}")

    # Step 4: L1 on inner-passers (or top-3 if none pass)
    candidates_for_l1 = [s for s in inner_summary if s["inner_pass"]]
    if not candidates_for_l1:
        candidates_for_l1 = sorted(inner_summary, key=lambda s: -s["wr_inner"])[:3]
        print(f"\n  No candidates passed inner gate; running L1 on top-3.")

    print(f"\n[4/4] L1 cross-family (claude-haiku) on {len(candidates_for_l1)} "
          f"× {len(fresh_pids)} pairs...")
    l1_verdicts = {s["tid"]: {} for s in candidates_for_l1}
    l1_tasks = [(s, pid) for s in candidates_for_l1 for pid in fresh_pids]

    def judge_l1(s, pid):
        b = base_answers.get(pid, ""); e = ext_answers[s["tid"]].get(pid, "")
        if not b or not e or b.startswith("[") or e.startswith("["):
            return s["tid"], pid, "missing"
        rng = random.Random((hash(pid) ^ hash(s["tid"])) % (2**32))
        if rng.random() < 0.5:
            left, right, ext_was = e, b, "A"
        else:
            left, right, ext_was = b, e, "B"
        w = judge_one(haiku, pid_to_prob[pid], left, right)
        if w == "tie": v = "tie"
        elif w in ("A", "B"): v = "ext" if w == ext_was else "base"
        else: v = "err"
        return s["tid"], pid, v

    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(judge_l1, s, pid) for s, pid in l1_tasks]
        for f in as_completed(futs):
            tid, pid, v = f.result()
            l1_verdicts[tid][pid] = v
            done += 1
            if done % 30 == 0:
                print(f"  l1 judge {done}/{len(l1_tasks)} ({time.time()-t0:.0f}s)")

    print(f"\n=== Final n=100 v20-style verdict ===")
    print(f"{'tid':12s} {'wr_inner':>9s} {'inner':>6s}  {'wr_L1':>8s} "
          f"{'L1@.55':>7s} {'L1@.60':>7s}")
    print("-" * 78)
    final_results = []
    for s in inner_summary:
        v_l1 = l1_verdicts.get(s["tid"], {})
        ne_l1 = sum(1 for x in v_l1.values() if x == "ext")
        nb_l1 = sum(1 for x in v_l1.values() if x == "base")
        n_eff_l1 = ne_l1 + nb_l1
        wr_l1 = ne_l1 / n_eff_l1 if n_eff_l1 else None
        l1_pass_55 = (wr_l1 is not None) and (wr_l1 >= 0.55)
        l1_pass_60 = (wr_l1 is not None) and (wr_l1 >= 0.60)
        wr_l1_s = f"{wr_l1:.3f}" if wr_l1 is not None else "---"
        i_mark = "PASS" if s["inner_pass"] else "FAIL"
        l55 = "PASS" if l1_pass_55 else ("---" if wr_l1 is None else "FAIL")
        l60 = "PASS" if l1_pass_60 else ("---" if wr_l1 is None else "FAIL")
        print(f"  {s['tid']:12s} {s['wr_inner']:>9.3f} {i_mark:>6s}  "
              f"{wr_l1_s:>8s} {l55:>7s} {l60:>7s}")
        s["wr_l1"] = wr_l1; s["n_eff_l1"] = n_eff_l1
        s["l1_pass_55"] = l1_pass_55; s["l1_pass_60"] = l1_pass_60
        final_results.append(s)

    n_pre = sum(1 for s in final_results
                 if s["inner_pass"] and s.get("l1_pass_55"))
    n_strict = sum(1 for s in final_results
                    if s["inner_pass"] and s.get("l1_pass_60"))
    print(f"\n=== Headline ===")
    print(f"  v20-style with same-domain exemplar at n=100 fresh:")
    print(f"  At preregistered policy (inner>=0.60 AND L1>=0.55): "
          f"{n_pre}/{len(cands)} pass")
    print(f"  At strict policy       (inner>=0.60 AND L1>=0.60): "
          f"{n_strict}/{len(cands)} pass")
    print(f"\n  Comparison vs Exp 53 (no-exemplar n=100): "
          f"Exp 53 was {e53['n_pass_pre']}/12 (pre) and "
          f"{e53['n_pass_strict']}/12 (strict).")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "n_fresh": len(fresh_pids),
           "fresh_pids_source": "Exp 53 (seed=2027)",
           "frozen_thresholds": {"inner": INNER_THRESHOLD,
                                  "L1_pre": 0.55, "L1_strict": 0.60},
           "judges": [solver.model, haiku.model],
           "exemplar_protocol": "v20-style same-domain neighbour from "
                                  "fresh pool, base answers from pass-1 used "
                                  "as exemplar text",
           "summary": final_results,
           "n_pass_pre": n_pre, "n_pass_strict": n_strict,
           "comparison_to_exp53": {"exp53_n_pass_pre": e53["n_pass_pre"],
                                    "exp53_n_pass_strict": e53["n_pass_strict"]},
           "exemplar_map": exemplar_map,
           "inner_verdicts": inner_verdicts,
           "l1_verdicts": l1_verdicts}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
