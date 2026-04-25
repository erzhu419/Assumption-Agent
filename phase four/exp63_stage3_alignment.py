"""Exp 63 — Stage 3 minimal alignment: same problem in different
languages should be matched to the same prior by the scheduler.

If a methodological prior is genuinely about the structure of a
problem (not its surface language), then the same problem
expressed in English, Spanish, French, German, and Japanese
should be assigned the SAME prior by the scheduler. Inconsistency
across languages would mean the scheduler picks based on surface
features, not underlying structure --- which would be the
opposite of an alignment layer.

This is the simplest possible operationalisation of Stage 3:
implicit cross-language alignment. The scheduler does not need to
explicitly recognise ``these two problems are the same'' --- it
just needs to make consistent picks. We measure pairwise
agreement of scheduler picks across language versions.

Data: Exp 61's 60 problems × 5 languages (en + es + fr + de + ja).
Scheduler picks per (problem, language) are already in Exp 61's
output. Compute:

  Per-problem agreement: mode pick across 5 languages, fraction
    of languages picking the mode.
  Pairwise agreement: for each pair of languages, fraction of
    problems where both pick the same prior.
  Family-correct agreement: per-family, fraction of all 5 langs
    picking the family-optimal prior.

If agreement is significantly above chance (~25% for 4-prior
random), Stage 3 implicit alignment is empirically demonstrated.
"""
import json, os, sys, time
from pathlib import Path
PROJECT = Path(__file__).parent.parent
AUTO = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO / "exp63_stage3_alignment_log.json"

def main():
    print(f"=== Exp 63: Stage 3 minimal alignment (cross-language pick consistency) ===")
    e61_path = AUTO / "exp61_crosslanguage_log.json"
    if not e61_path.exists():
        print(f"ERROR: Exp 61 log not found. Run Exp 61 first.")
        return
    e61 = json.loads(e61_path.read_text())
    picks_by_lang = e61["scheduler_picks_per_language"]
    langs = list(picks_by_lang.keys())
    print(f"  Languages: {langs}")
    pids = sorted(picks_by_lang[langs[0]].keys())
    print(f"  Problems: {len(pids)}\n")

    # Per-problem mode + agreement
    print(f"=== Per-problem mode-pick agreement ===")
    problem_summary = {}
    full_agreement_count = 0
    fam_optimal = {"A_decompose": "decompose", "B_restate": "restate",
                    "C_estimate": "estimate", "D_constraints": "constraints"}
    fam_agree_optimal = {f: 0 for f in fam_optimal}
    fam_pid_count = {f: 0 for f in fam_optimal}

    for pid in pids:
        picks = [picks_by_lang[lc].get(pid, "none") for lc in langs]
        # Find mode
        from collections import Counter
        mode_pick, mode_count = Counter(picks).most_common(1)[0]
        agreement = mode_count / len(langs)
        family = pid.split("_")[0]
        full = "A_decompose" if family == "A" else "B_restate" if family == "B" \
                else "C_estimate" if family == "C" else "D_constraints"
        fam_pid_count[full] += 1
        if mode_pick == fam_optimal[full]:
            fam_agree_optimal[full] += 1
        if mode_count == len(langs):
            full_agreement_count += 1
        problem_summary[pid] = {"family": full, "picks": picks,
                                  "mode_pick": mode_pick,
                                  "mode_count": mode_count,
                                  "agreement_fraction": agreement}

    avg_agreement = sum(s["agreement_fraction"] for s in problem_summary.values()) / len(problem_summary)
    print(f"  Average per-problem agreement (fraction of langs picking mode): "
          f"{avg_agreement:.1%}")
    print(f"  Full agreement (all 5 langs identical): "
          f"{full_agreement_count}/{len(pids)} = {full_agreement_count/len(pids):.1%}")
    print(f"  Random-pick baseline expected agreement: ~25% per problem")

    # Pairwise agreement
    print(f"\n=== Pairwise language-pair agreement ===")
    n_pids = len(pids)
    pair_agree = {}
    print(f"{'lang_pair':16s} {'agreement':>10s}")
    print("-" * 32)
    for i, l1 in enumerate(langs):
        for l2 in langs[i+1:]:
            agree = sum(1 for pid in pids if picks_by_lang[l1].get(pid) == picks_by_lang[l2].get(pid))
            pair_agree[f"{l1}-{l2}"] = agree / n_pids
            print(f"{l1}-{l2:14s} {agree/n_pids:>10.3f}")

    # Per-family mode-correct rate
    print(f"\n=== Per-family mode-pick = optimal rate ===")
    print(f"  (How often is the cross-lang mode pick the family-optimal prior?)")
    for f in fam_optimal:
        if fam_pid_count[f] > 0:
            rate = fam_agree_optimal[f] / fam_pid_count[f]
            print(f"  {f:14s} optimal={fam_optimal[f]:14s} mode_correct: "
                  f"{fam_agree_optimal[f]}/{fam_pid_count[f]} = {rate:.1%}")

    # Per-language pick correctness
    print(f"\n=== Per-language scheduler-pick correctness ===")
    for lc in langs:
        correct = sum(1 for pid in pids
                       if picks_by_lang[lc].get(pid) == fam_optimal[problem_summary[pid]["family"]])
        print(f"  {lc}: {correct}/{n_pids} = {correct/n_pids:.1%}")

    # Conclusion
    if avg_agreement > 0.6:
        print(f"\n  → Stage 3 implicit alignment validated: scheduler is consistent "
              f"across surface language ({avg_agreement:.1%} mode agreement). "
              f"The picks reflect underlying problem structure, not surface language.")
    else:
        print(f"\n  → Stage 3 alignment not validated: {avg_agreement:.1%} agreement is "
              f"close to or below random.")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "n_problems": n_pids,
           "languages": langs,
           "average_per_problem_mode_agreement": avg_agreement,
           "full_agreement_fraction": full_agreement_count / n_pids,
           "pairwise_agreement": pair_agree,
           "per_family_mode_correct_rate": {f: fam_agree_optimal[f] / fam_pid_count[f]
                                                if fam_pid_count[f] > 0 else 0
                                              for f in fam_optimal},
           "per_problem_summary": problem_summary}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
