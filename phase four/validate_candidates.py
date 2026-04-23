"""A/B validate success-distilled candidates on held-out 50.

For each candidate:
  1. Build extended library (current 75 + candidate)
  2. Mine 3 cross-domain exemplars for new wisdom (via build_diverse_exemplars_v15 logic)
  3. Run v20 with extended library on sample_holdout_50
  4. Compare to v20_holdout50 base (we may need to generate this first)
  5. A/B judge; KEEP if +10pp

Top-K candidates selected by cluster size.
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase one" / "scripts" / "validation"))
sys.path.insert(0, str(PROJECT / "phase four"))

from llm_client import create_client, parse_json_from_llm
from gpt5_client import GPT5Client
from cached_framework import judge_pair, _save_content_cache
from wisdom_registry import load_or_init_registry, save_registry, append_wisdom


CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
CANDIDATES_PATH = AUTO_DIR / "success_distilled_candidates.json"
VALIDATION_LOG = AUTO_DIR / "validation_log.json"

V20_SCRIPT = PROJECT / "phase one" / "scripts" / "validation" / "phase2_v20_framework.py"
EXEMPLARS_PATH = CACHE / "wisdom_diverse_exemplars.json"

HOLDOUT_SAMPLE = "sample_holdout_50.json"
KEEP_THRESHOLD = 0.10  # +10pp to KEEP


# Reuse diverse exemplar mining prompt
EXEMPLAR_PROMPT = """给下面这条 wisdom 从 50 个候选问题里挑 3 个**跨域最远**的判例。

## Wisdom
aphorism: {aphorism}
source: {source}
signal: {signal}
unpacked: {unpacked}

## 候选 problems (50 个)
{problems_brief}

## 规则
1. 挑 3 个 pid
2. 跨 2-3 个不同 domain
3. 本 wisdom 真的在那道题上 fire

## 输出 JSON (不要代码块)
{{"selected": [
  {{"pid": "xxx", "why_applies": "20-40字"}},
  ... (3 条)
]}}
"""


def cache_load(p, default=None):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return default if default is not None else {}
    return default if default is not None else {}


def cache_save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def mine_exemplars_for_candidate(cand, sample, v13_ans, ours_ans):
    """Mine 3 cross-domain exemplars for a new wisdom using held-out 50 problems."""
    pid_to_info = {p["problem_id"]: p for p in sample}
    problems_brief = "\n".join(
        f"[{p['problem_id']}] [{p.get('domain','?')}/{p.get('difficulty','?')}] "
        f"{p.get('description','')[:80]}"
        for p in sample
    )
    client = GPT5Client()
    prompt = EXEMPLAR_PROMPT.format(
        aphorism=cand["aphorism"], source=cand["source"],
        signal=cand["signal"], unpacked=cand["unpacked_for_llm"],
        problems_brief=problems_brief,
    )
    try:
        resp = client.generate(prompt, max_tokens=500, temperature=0.3)
        parsed = parse_json_from_llm(resp["text"])
        selected = parsed.get("selected", [])
    except Exception as e:
        print(f"    [exemplar error] {e}")
        return []

    result = []
    MATH_SCI = {"mathematics", "science"}
    for item in selected[:3]:
        pid = item.get("pid", "").strip()
        if pid not in pid_to_info:
            continue
        info = pid_to_info[pid]
        dom = info.get("domain", "?")
        ans = ours_ans.get(pid) if dom in MATH_SCI else v13_ans.get(pid)
        ans = ans or v13_ans.get(pid) or ours_ans.get(pid) or ""
        result.append({
            "pid": pid, "domain": dom,
            "difficulty": info.get("difficulty", "?"),
            "problem_sketch": info.get("description", "")[:350],
            "why_applies": item.get("why_applies", ""),
            "answer_snippet": ans[:700],
            "answer_source": "ours_27" if dom in MATH_SCI else "v13_reflect",
        })
    return result


def run_v20(variant, sample_file, wisdom_file, n=50):
    cmd = [
        "python", "-u", str(V20_SCRIPT),
        "--variant", variant, "--n", str(n),
        "--sample", sample_file,
        "--wisdom", wisdom_file,
    ]
    print(f"    [v20 {variant}] ...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=4200)
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[-400:]}")
        return {}
    return cache_load(CACHE / "answers" / f"{variant}_answers.json")


def judge_ab(batch, ans_a, ans_b, label_a, label_b):
    client = create_client()
    wins_a = wins_b = ties = 0
    for p in batch:
        pid = p["problem_id"]
        a, b = ans_a.get(pid), ans_b.get(pid)
        if not a or not b:
            continue
        rng = random.Random(hash(pid) % (2**32))
        if rng.random() < 0.5:
            left, right, a_was = a, b, "A"
        else:
            left, right, a_was = b, a, "B"
        v = judge_pair(client, p.get("description", ""), left, right)
        w = v.get("winner", "tie")
        if w == "tie":
            ties += 1
        elif w == a_was:
            wins_a += 1
        else:
            wins_b += 1
    _save_content_cache()
    total_decided = wins_a + wins_b
    wr_a = wins_a / total_decided if total_decided else 0.5
    return {"wins_a": wins_a, "wins_b": wins_b, "ties": ties, "wr_a": wr_a}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()

    candidates = cache_load(CANDIDATES_PATH, default=[])
    # Dedupe by aphorism
    seen_aph = set()
    deduped = []
    for c in candidates:
        a = c.get("aphorism", "")
        if a in seen_aph:
            continue
        seen_aph.add(a)
        deduped.append(c)
    # Sort by cluster size
    deduped.sort(key=lambda c: -c.get("_cluster_size", 0))
    top = deduped[:args.top_k]

    print(f"Loaded {len(candidates)} candidates, {len(deduped)} unique, testing top {len(top)}")
    for c in top:
        print(f"  • {c['aphorism']} (cluster n={c.get('_cluster_size','?')})")

    # Load context
    sample = json.loads((CACHE / HOLDOUT_SAMPLE).read_text(encoding="utf-8"))
    sample = [p for p in sample if "description" in p]
    registry = load_or_init_registry()
    v13_ans = cache_load(CACHE / "answers" / "phase2_v13_reflect_answers.json")
    ours_ans = cache_load(CACHE / "answers" / "ours_27_answers.json")

    # Step 1: ensure v20 base answers on holdout_50 exist
    META_STRIP = {"status", "created_at", "last_activated", "activation_count",
                   "contribution_gain", "source", "keep_reason", "gain_samples",
                   "deprecated_at", "deprecation_reason", "removed_at", "removal_reason"}
    base_export = [{k: v for k, v in w.items() if k not in META_STRIP}
                   for w in registry["wisdoms"] if w.get("status") == "active"]
    base_lib_filename = "_validation_base_library.json"
    cache_save(CACHE / base_lib_filename, base_export)

    base_variant = "_validation_v20_base"
    if not (CACHE / "answers" / f"{base_variant}_answers.json").exists():
        print("\n[base] Generating v20 base on held-out 50...")
        run_v20(base_variant, HOLDOUT_SAMPLE, base_lib_filename, len(sample))
    ans_base = cache_load(CACHE / "answers" / f"{base_variant}_answers.json")
    print(f"  base answers: {len(ans_base)}")

    # Step 2: for each candidate, mine exemplars + run ext + judge
    results = []
    exemplars_all = cache_load(EXEMPLARS_PATH, default={})

    for i, cand in enumerate(top):
        aphorism = cand["aphorism"]
        print(f"\n=== [{i+1}/{len(top)}] {aphorism} ===")

        # Mine exemplars for this candidate
        print(f"  [exemplars] mining 3 cross-domain...")
        ex = mine_exemplars_for_candidate(cand, sample, v13_ans, ours_ans)
        if len(ex) != 3:
            print(f"    [skip] mined only {len(ex)} exemplars")
            continue

        # Build extended library
        tentative_id = f"WCAND{i+1:02d}"
        cand_entry = {k: v for k, v in cand.items() if not k.startswith("_")}
        cand_entry.pop("novelty_sim", None)
        cand_entry.pop("covers_batch_pids", None)
        cand_entry.pop("rationale", None)
        cand_entry["id"] = tentative_id
        ext_lib = base_export + [cand_entry]

        ext_lib_filename = f"_validation_ext_{tentative_id}.json"
        cache_save(CACHE / ext_lib_filename, ext_lib)

        # Add to diverse exemplars cache so v20 can use them
        exemplars_all[tentative_id] = ex
        cache_save(EXEMPLARS_PATH, exemplars_all)

        # Run v20 with ext library
        ext_variant = f"_validation_v20_ext_{tentative_id}"
        ans_ext = run_v20(ext_variant, HOLDOUT_SAMPLE, ext_lib_filename, len(sample))
        if not ans_ext:
            print("    [skip] ext run failed")
            continue

        # A/B judge
        print(f"  [judge] ext vs base on {len(sample)} held-out problems...")
        ab = judge_ab(sample, ans_ext, ans_base, "extended", "base")
        print(f"    ext={ab['wins_a']}, base={ab['wins_b']}, ties={ab['ties']}, "
              f"wr_ext={ab['wr_a']:.2f}")

        decision = "KEEP" if ab["wr_a"] >= 0.5 + KEEP_THRESHOLD else "REVERT"
        print(f"    → {decision}")

        result = {
            "candidate": aphorism,
            "source": cand.get("source", ""),
            "cluster_size": cand.get("_cluster_size", 0),
            "novelty_sim": cand.get("novelty_sim", 0),
            "tentative_id": tentative_id,
            "exemplar_pids": [e["pid"] for e in ex],
            "ab": ab,
            "decision": decision,
        }
        if decision == "KEEP":
            real_id = append_wisdom(
                registry, cand_entry, "success_distilled",
                f"A/B on held-out 50: wr_ext={ab['wr_a']:.2f}"
            )
            result["committed_id"] = real_id
            # Transfer exemplars to real id
            exemplars_all[real_id] = ex
            del exemplars_all[tentative_id]
            cache_save(EXEMPLARS_PATH, exemplars_all)
            save_registry(registry)
            print(f"    → committed as {real_id}")

        results.append(result)

    log = cache_load(VALIDATION_LOG, default=[])
    log.append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "top_k": args.top_k,
        "results": results,
    })
    cache_save(VALIDATION_LOG, log)

    print(f"\n=== Validation Summary ===")
    kept = [r for r in results if r["decision"] == "KEEP"]
    print(f"  Tested: {len(results)}")
    print(f"  KEPT:   {len(kept)}")
    for r in kept:
        print(f"    ✅ {r['candidate']} → {r.get('committed_id','?')} "
              f"(wr_ext={r['ab']['wr_a']:.2f})")
    rev = [r for r in results if r["decision"] == "REVERT"]
    for r in rev:
        print(f"    ❌ {r['candidate']} (wr_ext={r['ab']['wr_a']:.2f})")


if __name__ == "__main__":
    main()
