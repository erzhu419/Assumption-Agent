"""Exp 82 v2 small ablation: 3 wisdoms × 5 kinds × ~15 trigger pids × ~5 outside pids.

Per (wisdom, kind):
  1. Use exp17 trigger labels to pick trigger_subset (SHOULD_FIRE) and
     outside_subset (NO_FIRE).
  2. Propose one Hypothesis via the kind-specific LLM proposer.
  3. Evaluate it via the kind-specific A/B harness on (trigger ∪ outside),
     with and without GENERIC warning.
  4. Apply accept_decide() and persist.

Aggregate per kind:
  - accept rate
  - mean Δ(EXT-BASE) on trigger
  - mean Δ(EXT-BASE) on outside (specificity check)
  - mean Δ(EXT-GENERIC) on trigger (does it beat 'be careful'?)

Output:
  - phase six/exp82/hypotheses.jsonl  (full Hypothesis records)
  - phase six/exp82/forensic.jsonl    (per-call forensics)
  - phase six/exp82/ablation_summary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from model_router import cheap  # noqa: E402
from hypothesis import Hypothesis, accept_decide, KINDS, HYPO_LOG  # noqa: E402
from proposers import propose  # noqa: E402
from evaluators import evaluate, load_gold, load_base_cache  # noqa: E402

EXP_DIR = Path(__file__).parent
SUMMARY = EXP_DIR / "ablation_summary.json"
TRIGGER_LABELS = PROJECT / "phase four" / "autonomous" / "exp17_trigger_labels.json"

# Default = top 3 (the small-ablation seeds). Override via env var SEED_CIDS
# (comma-sep) or --all flag for the full 12-wisdom run.
import os as _os
_default_seeds = ("WCAND10", "WCAND09", "WCAND01")
SEED_CIDS = tuple(_os.environ.get("SEED_CIDS", ",".join(_default_seeds)).split(","))
MIN_TRIGGER_PIDS = int(_os.environ.get("MIN_TRIGGER_PIDS", "5"))


def load_wisdoms() -> dict:
    M = json.loads((EXP_DIR / "verdict_matrix.json").read_text(encoding="utf-8"))
    return {c["cid"]: c for c in M["candidates"]}


def load_trigger_labels() -> dict:
    return json.loads(TRIGGER_LABELS.read_text(encoding="utf-8"))


def pick_subsets(labels_for_cid: dict, gold: dict, max_trigger: int = 15,
                  max_outside: int = 5) -> tuple[list, list]:
    """From exp17 labels for one wisdom, pick pids that are also in gold.
    Cap to max_trigger / max_outside.
    """
    fire = [pid for pid, e in labels_for_cid.items()
            if e.get("verdict") == "SHOULD_FIRE" and pid in gold]
    no_fire = [pid for pid, e in labels_for_cid.items()
                if e.get("verdict") == "NO_FIRE" and pid in gold]
    return fire[:max_trigger], no_fire[:max_outside]


def main():
    print(f"Loading wisdoms, gold, base_cache, trigger labels...", flush=True)
    wisdoms = load_wisdoms()
    gold = load_gold()
    base_cache = load_base_cache()
    labels = load_trigger_labels()
    print(f"  wisdoms: {len(wisdoms)}, gold: {len(gold)}, base_cache: {len(base_cache)}", flush=True)

    seeds_raw = [(cid, wisdoms[cid], labels.get(cid, {})) for cid in SEED_CIDS if cid in wisdoms]
    seeds = []
    for cid, w, lab in seeds_raw:
        fire, no_fire = pick_subsets(lab, gold)
        if len(fire) < MIN_TRIGGER_PIDS:
            print(f"  SKIP {cid}: '{w['aphorism']}' — only {len(fire)} trigger pids (< {MIN_TRIGGER_PIDS})", flush=True)
            continue
        seeds.append((cid, w, lab))
        print(f"  {cid}: '{w['aphorism']}' — trigger {len(fire)}, outside {len(no_fire)}", flush=True)
    print(f"  → {len(seeds)} usable wisdoms × 5 kinds = {len(seeds)*5} hypotheses", flush=True)

    solver = cheap("gpt_mini")
    judge = cheap("gpt_mini")
    print(f"  solver=gpt_mini, judge=gpt_mini", flush=True)

    # ─── Phase 1: propose 3 × 5 = 15 hypotheses ────────────────────────
    print(f"\n[Phase 1] Proposing 15 hypotheses (3 wisdoms × 5 kinds)...", flush=True)
    hypotheses = []
    for cid, w, lab in seeds:
        fire, no_fire = pick_subsets(lab, gold)
        for kind in KINDS:
            t0 = time.time()
            try:
                h = propose(w, kind, llm_client=solver,
                             trigger_subset=fire, outside_subset=no_fire)
                hypotheses.append(h)
                print(f"  {cid}/{kind:14s} OK ({time.time()-t0:.1f}s) — {h.claim[:80]}", flush=True)
            except Exception as e:
                print(f"  {cid}/{kind:14s} FAIL ({time.time()-t0:.1f}s): {str(e)[:120]}", flush=True)

    # persist all hypotheses (deferred state)
    for h in hypotheses:
        h.persist()

    # ─── Phase 2: evaluate each ───────────────────────────────────────
    print(f"\n[Phase 2] Evaluating {len(hypotheses)} hypotheses...", flush=True)
    for i, h in enumerate(hypotheses, 1):
        t0 = time.time()
        print(f"\n  [{i}/{len(hypotheses)}] {h.seed_cid}/{h.kind}: {h.claim[:80]}", flush=True)
        try:
            ev = evaluate(h, gold, base_cache, solver, judge,
                            solver_name="gpt_mini",
                            with_generic=True, n_trials=1, max_workers=4)
            decision, reason = accept_decide(ev)
            h.record_outcome(ev, decision, reason)
            n_t = ev["n_trigger"]
            n_o = ev["n_outside"]
            base_t = ev["base_correct_trigger"]
            ext_t = ev["ext_correct_trigger"]
            base_o = ev["base_correct_outside"]
            ext_o = ev["ext_correct_outside"]
            d_eb = ev.get("delta_ext_base", 0)
            d_ob = ev.get("outside_delta_ext_base", 0)
            d_eg = ev.get("delta_ext_generic", None)
            print(f"    n_trig={n_t} n_out={n_o}  trig: BASE {base_t}/{n_t} → EXT {ext_t}/{n_t} (Δ={d_eb:+.2f})", flush=True)
            print(f"                          out:  BASE {base_o}/{n_o} → EXT {ext_o}/{n_o} (Δ={d_ob:+.2f})", flush=True)
            if d_eg is not None:
                gen_t = ev.get("generic_correct_trigger", 0)
                print(f"                          gen:  GEN  {gen_t}/{n_t}                          (Δ EXT-GEN={d_eg:+.2f})", flush=True)
            if h.kind == "feature":
                ftr = ev.get("feature_fire_rate_trigger", 0)
                fout = ev.get("feature_fire_rate_outside", 0)
                print(f"                          feature fire-rate: trigger {ftr:.0%} / outside {fout:.0%}", flush=True)
            print(f"    decision: {decision} ({reason or '-'})  [{time.time()-t0:.1f}s]", flush=True)
        except Exception as e:
            print(f"    EVAL FAIL ({time.time()-t0:.1f}s): {str(e)[:200]}", flush=True)
            h.record_outcome({}, "deferred", "eval_error")

    # re-persist with outcomes
    HYPO_LOG.write_text("")  # truncate
    for h in hypotheses:
        h.persist()

    # ─── Phase 3: aggregate by kind ───────────────────────────────────
    print(f"\n[Phase 3] Per-kind aggregation\n", flush=True)
    by_kind = {}
    for h in hypotheses:
        by_kind.setdefault(h.kind, []).append(h)

    summary = {"by_kind": {}, "n_hypotheses": len(hypotheses), "seeds": list(SEED_CIDS)}
    print(f"  {'kind':14s} | {'n':>2s} | {'accept':>6s} | {'Δ(E-B) trig':>12s} | {'Δ(E-B) out':>11s} | {'Δ(E-G) trig':>12s}", flush=True)
    print(f"  {'-'*14}-+-{'-'*2}-+-{'-'*6}-+-{'-'*12}-+-{'-'*11}-+-{'-'*12}", flush=True)
    for kind in KINDS:
        hs = by_kind.get(kind, [])
        if not hs:
            continue
        n_acc = sum(1 for h in hs if h.decision == "accepted")
        d_eb_l = [h.evidence.get("delta_ext_base") for h in hs if "delta_ext_base" in h.evidence]
        d_ob_l = [h.evidence.get("outside_delta_ext_base") for h in hs if "outside_delta_ext_base" in h.evidence]
        d_eg_l = [h.evidence.get("delta_ext_generic") for h in hs if "delta_ext_generic" in h.evidence]
        avg_eb = (sum(d_eb_l) / len(d_eb_l)) if d_eb_l else 0
        avg_ob = (sum(d_ob_l) / len(d_ob_l)) if d_ob_l else 0
        avg_eg = (sum(d_eg_l) / len(d_eg_l)) if d_eg_l else 0
        kind_summary = {
            "n": len(hs),
            "n_accepted": n_acc,
            "accept_rate": n_acc / len(hs),
            "mean_delta_ext_base_trigger": avg_eb,
            "mean_outside_delta_ext_base": avg_ob,
            "mean_delta_ext_generic": avg_eg,
            "hypothesis_ids": [h.hid for h in hs],
        }
        summary["by_kind"][kind] = kind_summary
        print(f"  {kind:14s} | {len(hs):2d} | {n_acc}/{len(hs)}    | {avg_eb:+10.2%}   | {avg_ob:+9.2%}   | {avg_eg:+10.2%}", flush=True)

    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved → {SUMMARY.relative_to(PROJECT)}", flush=True)
    print(f"       {HYPO_LOG.relative_to(PROJECT)}", flush=True)
    print(f"       {(EXP_DIR / 'forensic.jsonl').relative_to(PROJECT)}", flush=True)


if __name__ == "__main__":
    main()
