# Summary

The paper presents a fully logged, retrieval-level “wisdom library” self-improvement loop for an LLM solver, then argues that its single-family \(n=50\) A/B acceptance gate is judge-fragile. The main empirical claim is a null result: three wisdoms accepted by the inner loop do not survive a multi-layer audit involving cross-family re-judgment, side/order perturbations, sample extension, cross-solver checks, fresh-domain/GSM8K probes, and faithfulness-style diagnostics. The paper positions the audit stack, rather than the wisdom loop itself, as the methodological contribution, while repeatedly emphasizing that the evidence comes from one loop cycle and three accepted candidates.

# Strengths

- **The paper studies an important and under-audited failure mode.** The central L1 experiment in “Experiment 1 — Cross-judge stability” re-judges identical cached answer pairs with a different model family, holding content fixed. This is a simple but valuable diagnostic for LLM-as-judge self-improvement claims, and the paper makes a convincing case that such checks are often missing in adjacent work.

- **The authors are unusually explicit that the main result is negative and scoped.** The abstract’s “Honest scope” paragraph and the “Scope and non-claims” paragraph in the Introduction state that the evidence is three accepted wisdoms from one loop cycle, with no human ground truth. This restraint strengthens the submission relative to many self-improvement papers that would overclaim from similar evidence.

- **The loop is concretely specified and logged.** Sections 3.1–3.4 describe the solver, mutable registry, candidate generators, pruner, and A/B gate in enough detail to understand the system’s state transitions. The versioned registry and provenance claims are especially useful for auditing whether a purported library mutation actually occurred.

- **The paper reports several uncomfortable negative findings instead of hiding them.** Examples include the failure generator never firing in Section 4.4, the collapse of original KEEPs under L1/L3, the retraction of the Exp. 27 F1 framing, and the failure of the trigger-conditioned gate to generalize in Exp. 15/33. These admissions increase trust in the authors’ willingness to falsify their own narrative.

- **The conceptual decomposition in Section 3.5 is appropriately caveated.** The paper explicitly says the \(Z^{\mathrm{specific}}, Z^{\mathrm{generic}}, Z^{\mathrm{style}}\) decomposition is informal and non-identifiable from one experiment. That is the right level of theory for this setting and helps organize the audit layers without pretending to prove identifiability.

- **The audit goes beyond one kind of perturbation.** Even though the layers are not independent, the paper does attempt cross-family judging, sample extension, cross-solver regeneration, fresh-domain testing, standard-benchmark probing, and non-pairwise faithfulness checks. This breadth is a strength for a case-study methodology paper.

- **The reproducibility intent is strong.** The main body promises release of code, cached judgments, registry states, figures, prompt templates, costs, seeds, proxy details, and logs. For a paper whose main contribution is an audit procedure, such artifacts are important.

# Weaknesses

- **The audit stack is not validated as an audit method; it only rejects this loop’s candidates.**  
  Reference: Discussion, “No positive / negative controls for the audit stack,” and Experiments 9–15.  
  For a methodology paper, showing that an audit rejects three weak or unstable KEEPs is not enough; the method must also be shown to accept genuinely useful interventions and reject known-placebo interventions at an interpretable rate. Without positive controls, length-matched random-context controls, deliberately useful task-specific hints, or human-curated improvements, the audit stack may simply be too conservative or miscalibrated.  
  **Fix:** Add controlled interventions: random aphorisms, shuffled wisdoms, length-matched placebo context, intentionally useful task-specific hints, and human-curated library additions. Report false-positive and false-negative rates of each audit layer.

- **The entire audit protocol is post-hoc and adaptively expanded.**  
  Reference: Section 4, “A note on staged stress testing (not pre-registration),” and Discussion, “Gate-design freedom / hindsight” and “Audit stack itself is post-hoc.”  
  The paper states that L1–L6 were added through staged reviewer simulation after the inner-loop result, and that several gate designs were tried before the trigger-conditioned gate. This is a serious weakness for a top-venue methodology claim, because the final “0/3 survive six layers” result is vulnerable to layer-selection hindsight: keep adding tests until the accepted candidates fail.  
  **Fix:** Freeze the audit stack and thresholds in advance, then run it prospectively on a fresh loop cycle or fresh candidate batch. The included preregistration template is not a substitute for a prospective result.

- **There are serious internal inconsistencies in the reported statistics.**  
  Reference: Abstract expensive-tier claim; Exp. 35 Table 1/caption; Fig. 5-judge forest caption; Section 4.18 “Binomial confidence intervals”; Exp. 42 GSM8K table.  
  The abstract says no expensive-tier CI upper bound reaches 0.60, but Exp. 35 reports \(W_{076}\) under GPT-5.4 as \(0.54\) with CI \([0.40,0.67]\), and \(W_{077}\) under Opus as \([0.39,0.67]\). The five-judge forest caption says no CI upper bound reaches the KEEP threshold across all 15 measurements, but the inner-loop Gemini CIs for the original KEEPs plainly exceed 0.60. Exp. 42 reports base \(28/30\) and \(+W078=29/30\) while still labeling \(W078\) as “\(-1\), HARMS,” apparently mixing runs.  
  **Fix:** Regenerate all tables from a single source of truth, report exact ext/base/tie counts and effective \(n\), and correct all captions/abstract claims before resubmission.

- **Tie handling is inconsistent with the paper’s own stated protocol.**  
  Reference: Section 4.1 “Tie handling and effective \(n\),” Experiment 1 table, Section 4.18 Table of CIs, Exp. 32 GSM8K.  
  The paper says ties are excluded and Wilson CIs use the non-tie effective \(n\), but the CI table lists \(n=50\) for L1 rows where Exp. 1 has many ties, e.g. \(W077=19:21:10_{\text{tie}}\), whose effective \(n\) is 40, not 50. In GSM8K, subjective win rates are computed after excluding 20–26 ties out of 30, leaving extremely small effective denominators, yet the narrative treats the resulting wr values as strong evidence.  
  **Fix:** Report all CIs using the actual non-tie denominator, and preferably model ternary outcomes directly or count ties as 0.5 in a sensitivity analysis.

- **The main A/B gate appears to leak held-out information through exemplar mining.**  
  Reference: Section 3.4 “The A/B validation gate,” item 1: “Mine 3 cross-domain exemplars from the held-out 50,” then evaluate on all 50 held-out problems.  
  If the wisdom record used during evaluation contains exemplars selected from the same held-out set being evaluated, the held-out gate is contaminated. This is especially problematic because the paper’s core story depends on the original gate producing plausible KEEPs that are later audited.  
  **Fix:** Mine exemplars only from training/development pools disjoint from the evaluation set, or exclude exemplar PIDs from evaluation. Rerun the original gate under a nested split.

- **The statistical evidence is underpowered and often overinterpreted.**  
  Reference: Section 4.1 Measurement; Section 4.18 CI table; Exp. 31–42.  
  At \(n=50\), a wr of 0.60 has a Wilson interval including 0.50, as the paper itself notes. Many later tests use \(n=20\) or \(n=30\), often with high tie rates, and many claims are threshold-crossing claims rather than statistically significant paired differences. The paper should not say “fail every audit axis” or “actively harms” without more careful uncertainty treatment.  
  **Fix:** Use larger samples, paired tests for identical answer pairs, hierarchical models over problem/candidate/judge, and equivalence/non-inferiority framing where appropriate.

- **The audit layers are repeatedly described as independent despite shared data and dependencies.**  
  Reference: Introduction list says “each layer attacks a different failure mode and shares no data with the others”; abstract says they reuse the same KEEPs and cached pairs; Exp. 9 calls orthogonal tests “statistically independent.”  
  The same three KEEPs, overlapping problem pools, cached answer pairs, and judge families recur across many layers. This does not invalidate the case study, but it invalidates language suggesting independent evidence streams.  
  **Fix:** Consistently call them conditionally distinct stress tests, not independent tests, and use a dependency-aware analysis if combining evidence.

- **Cross-family disagreement is sometimes overinterpreted as lack of utility.**  
  Reference: Abstract “0/3 KEEPs survive,” Exp. 1 interpretation, Exp. 38 “rules out” stylistic-disagreement rebuttal, Conclusion.  
  The paper correctly states in places that L1 only falsifies robustness to judge choice, not substantive utility. But elsewhere it slides into stronger claims such as “final validated library delta +0,” “falsified,” and “rules out” alternative explanations using another LLM judge. Low \(\kappa\) among judges shows instability, not which judge is correct.  
  **Fix:** Use strictly robustness/non-validation language unless supported by human labels or objective task metrics. Treat gpt-5.5 as another model judge, not a human substitute that can rule out stylistic disagreement.

- **The side-randomization result is confounded with temporal drift.**  
  Reference: Experiment 5 “The side-stability finding” and its caveat; Section 4.17 \(\kappa\) bonus finding.  
  The paper initially presents \(0.64 \to 0.41\) as caused by side seed, but later acknowledges the rerun was not in the same batch and hosted-model drift is large even at temperature 0.0. Therefore L2, as executed, does not isolate side-position bias.  
  **Fix:** Run same-batch AB/BA counterbalancing for every pair, ideally with both orders judged back-to-back and position-bias correction estimated directly.

- **The faithfulness layer is not yet methodologically sound.**  
  Reference: Experiments 9–11 and 14; Discussion “L6 operationalization is under-specified.”  
  Comparing a sentence-embedding difference vector of two “what_changed” strings to a wisdom embedding has no validated interpretation as causal faithfulness. LLM-judged citation strictness is also prompt-sensitive, as Exp. 17 shows when requiring explicit citations drives target-citation to 95% without proving utility.  
  **Fix:** Validate L6 on synthetic examples with known inserted wisdom effects, include human faithfulness labels, and add placebo/context controls.

- **The benchmark and task distribution are insufficiently calibrated.**  
  Reference: Section 4.1 Setup; Limitations “Language scope”; Exp. 39 English replication.  
  The primary benchmark is a Chinese open-ended pool with unclear provenance, no human calibration, and LLM-judged outcomes. The English replication is only 30 MT-Bench prompts with a 6-entry library, and GSM8K is a small arithmetic probe that may not reflect the intended open-ended setting.  
  **Fix:** Provide detailed benchmark construction and validation, include public open-ended benchmarks with human annotations, and run a full English closed-loop replication rather than a small static-library probe.

- **The solver/proposer scope remains narrow despite cross-solver audits.**  
  Reference: Limitations “Single LLM family for solver”; Exp. 40.  
  The original candidate generation, solving, and gate are all centered on Gemini. Exp. 40 audits the three KEEPs with other solvers, but does not rerun the full self-improvement loop with another solver family. This limits claims about “default gates” beyond this specific solver-judge configuration.  
  **Fix:** Run at least one full loop cycle with a non-Gemini solver and judge, then apply the frozen audit stack.

- **Several baseline analyses are not meaningful.**  
  Reference: Exp. 27 and Exp. 33/Section 4.18 “random 30% baseline.”  
  A random 30% inclusion rate is not a principled baseline for utility or audit correctness, and pass-rate comparisons do not establish that a gate is better or worse without external labels. The paper partially retracts the Exp. 27 F1 framing, but still uses “below random baseline” language elsewhere.  
  **Fix:** Define baselines in terms of external utility labels or controlled interventions, not arbitrary pass-rate targets.

- **The presentation is too sprawling and contains too many chronological dead ends for a main-track paper.**  
  Reference: Experiments 1–42 are not ordered logically; Exp. 7 is retained as an “intermediate verdict subsequently overturned”; Exp. 25–42 appear before Exp. 12–19; repeated phrases such as “first time,” “fatal objections closed,” and “strict-reviewer closure.”  
  The paper reads like a lab notebook plus rebuttal log rather than a polished methodology paper. This makes it hard to identify the final protocol, final claims, and final evidence.  
  **Fix:** Move chronological history and overturned intermediate analyses to the appendix. In the main paper, present a fixed audit protocol, a single final results table, corrected statistics, and concise limitations.

- **Reproducibility remains limited by proprietary hosted models and a third-party proxy.**  
  Reference: Limitations “Reproducibility via third-party proxy”; Section 3.1 footnote; Section 4.17 observed temporal drift.  
  The paper itself observes substantial drift on identical content, and exact-token reproduction depends on proxy routing state. Cached artifacts help, but independent reproduction of the audit conclusions remains uncertain.  
  **Fix:** Replicate key results through official vendor endpoints and/or open-weight models with frozen checkpoints, and distinguish “reproduce from cached logs” from “rerun the experiment.”

# Questions to the authors

1. Did the A/B gate really mine cross-domain exemplars from the same 50 held-out problems used for evaluation? If yes, what happens when exemplars are mined from a disjoint development pool or exemplar PIDs are excluded?

2. Can you provide a corrected single table with ext/base/tie counts, effective \(n\), and Wilson CIs for every headline cell, especially Exp. 1, Exp. 35, Exp. 32, and Exp. 42?

3. If the six-layer audit stack and all thresholds are frozen now, what happens on a fresh loop cycle or fresh candidate batch with no further gate/layer changes?

4. Can the audit stack accept known-useful interventions and reject known-placebo interventions? For example: length-matched random wisdoms, shuffled wisdom descriptions, and human-authored task-specific hints.

5. On a stratified subset of the Chinese open-ended pairs, how do human annotators rate base vs. ext, and how do their preferences correlate with Gemini, Claude, GPT, and gpt-5.5 judgments?

6. How much of L2 is true side-position bias versus temporal/model drift? Can you rerun same-batch AB/BA counterbalancing for all original KEEPs?

7. Would a full non-Gemini self-improvement loop—candidate generation, solving, judging, and audit—show the same judge-fragility pattern?

# Rating

Reject

The paper is interesting and potentially valuable as a case study, but it is not yet a reliable top-venue methodology contribution. The most important problems are the lack of positive/negative controls for the audit stack, the post-hoc adaptive construction of the audit layers, apparent held-out leakage through exemplar mining, and multiple serious statistical/reporting inconsistencies. The core L1 idea is strong and the authors are commendably transparent about many limitations, but the current paper overstates what judge disagreement proves and contains enough internal contradictions that the main empirical narrative cannot be trusted without substantial correction and prospective validation.

# Confidence

4. I am familiar with LLM self-improvement loops, LLM-as-judge evaluation and cross-judge reliability issues, and reproducibility/preregistration concerns in empirical ML, though I am not a specialist in this particular Chinese wisdom-library benchmark.