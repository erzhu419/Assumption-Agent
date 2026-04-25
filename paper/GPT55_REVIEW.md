# Summary

The paper builds a closed-loop LLM “wisdom library” system that proposes, gates, and prunes retrieval-level methodological advice, then audits the loop’s three accepted additions using a multi-layer re-evaluation stack. Its main empirical story is deliberately differentiated: the original cached-data KEEP decisions fail to reproduce under stricter/cross-family cached audits, but a later preregistered fresh-data re-evaluation of the original 12 candidates recovers 2/3 original KEEPs at a laxer L1 threshold and finds one originally rejected candidate. The authors argue that cross-family re-judgment, selection-bias modeling, and preregistered fresh-data re-evaluation should become routine for LLM-judged self-improvement claims.

# Strengths

- **Clear and unusually honest scoping of the main claim**  
  The abstract and Conclusion repeatedly state that this is “a paper about audit methodology, not about whether self-improving LLM loops work,” and that the evidence is a single-loop case study with small \(n\). This restraint strengthens the submission because many self-improvement papers overgeneralize from self-judged gains, whereas this paper explicitly narrows its claim.

- **Good negative-results culture and willingness to revise the headline**  
  Several sections report results that undermine earlier interpretations: Exp. 7’s majority-of-3 positive verdict is later “subsequently overturned,” Exp. 27 retracts an earlier F1 framing, and Exp. 47 softens the earlier “0/3 survive” cached-data story. This transparency is a real strength for an audit-methodology paper because the object of study is precisely post-hoc overclaiming.

- **Useful operational audit checklist**  
  The L1–L6 stack in the Introduction and Method is concretely specified: cross-family re-judgment, side reseed, sample extension, cross-solver replication, fresh-domain/GSM8K, and faithfulness probes. Even if not novel individually, the operational bundling is practically useful and could help other authors stress-test self-improvement loops.

- **Selection-bias analysis is an important conceptual addition**  
  Exp. 45 explicitly models winner’s curse/top-3-of-12 regression-to-the-mean and concludes that L1 drops alone cannot distinguish judge fragility from selection bias. This is one of the paper’s strongest methodological points because it prevents an overly simple “cross-family drop = spurious wisdom” interpretation.

- **The preregistered fresh-data re-evaluation is a meaningful improvement over purely cached auditing**  
  Exp. 47 holds the original 12 candidates fixed, samples a disjoint 30-pid split, freezes thresholds in advance, and reports both point-estimate and posterior readings. This is much stronger than only rejudging cached answer pairs, and it materially changes the conclusion.

- **Artifacts and provenance appear unusually complete**  
  The main body describes released code, judgment files, registry states, prompt/log provenance, and preregistration commit hashes in Exp. 47; the prompt also indicates appendices with model identities, seeds, costs, prompts, and schema examples. For a case study, this level of logging substantially improves auditability.

- **The paper identifies several concrete failure modes in LLM-as-judge pipelines**  
  The same-family drift in Sec. “Experiment 37,” the side-randomization caveat in Exp. 5, the low cross-judge \(\kappa\), and the tie-handling sensitivity discussion are all practically relevant. These observations strengthen the methodological argument that a single \(n=50\) same-family A/B gate is fragile.

# Weaknesses

- **The main empirical evidence is still far too small and narrow for the paper’s methodological ambitions**  
  The core loop produces only 12 candidates and 3 original KEEPs, and the strongest fresh-data replication in Exp. 47 uses only 30 problems with \(n_{\mathrm{eff}}\) often 26–29. For a top-venue methodology paper arguing for a general audit stack, one loop cycle on one scaffold/solver family is not enough to establish that the stack reliably distinguishes selection bias from genuine non-replication. A convincing fix would be at least one fully fresh, preregistered loop replication—rerunning candidate generation, pruning, gating, and audit—preferably across multiple solver families or task distributions.

- **The audit stack was largely post-hoc, and the preregistered experiment is not a preregistered full-loop validation**  
  Sec. “A note on staged stress testing” admits that L1–L6 were added through staged reviewer simulation rather than preregistered before the inner loop. Exp. 47 is valuable, but it fixes the original 12 candidates and does not rerun the self-improvement loop; the paper itself emphasizes this. For this class of paper, post-hoc layer construction creates substantial garden-of-forking-paths risk. The fix is a locked audit protocol applied prospectively to a newly generated candidate set from a fresh loop run.

- **The fresh-data “2/3 replicate” headline is threshold- and uncertainty-sensitive**  
  Exp. 47 reports that W077 and WCAND03 pass L1 only at the preregistered \(0.55\) threshold, not at \(0.60\), and the joint posterior for the exact preregistered rule is only 0.370 for W077 and 0.424 for WCAND03; even W078 is 0.539. Thus the “2/3 original KEEPs replicate” claim is a point-estimate statement with weak individual evidence. For a methodology paper claiming diagnostic separation, this is not sufficient. The fix is to scale Exp. 47 to \(n\ge 100\), report preregistered decision uncertainty as primary, and avoid binary “replicate/collapse” labels unless posterior support is high.

- **No human ground truth on the main open-ended task undermines the validity of the audit target**  
  The paper relies almost entirely on LLM judges, including gpt-5.5 as a “human-annotation substitute” in Exp. 38/41/48. Cross-family disagreement is informative about judge robustness, but it does not establish which answer is actually better or whether the wisdom has substantive value. For an LLM-as-judge audit-methods paper, some human validation or objective-task calibration is essential. The fix is a human study on a stratified subset of open-ended pairs, plus known-effect synthetic controls where the correct intervention is known.

- **Positive controls do not establish audit-stack sensitivity**  
  Exp. 44 and Exp. 46 are explicitly negative: all six controls, including deliberately useful and duplicate-library controls, fail the gate. The paper then treats Exp. 47’s WCAND03 as an “empirical positive control,” but that candidate is not independently known to be useful; it is just another candidate that passes a later LLM-judged protocol. This is a serious weakness because an audit stack that mostly rejects additions may have unknown false-negative rate. A proper fix would include synthetic or human-validated positive controls that should pass by construction, e.g. tasks where a specific missing rule deterministically improves objective correctness.

- **Several statistical analyses are mislabeled or overinterpreted**  
  Exp. 49 calls DerSimonian–Laird random-effects pooling a “proper hierarchical model” and reports \(P(\theta>t)\) as posterior probabilities, but DerSimonian–Laird is a frequentist random-effects meta-analysis, not a Bayesian posterior model. With \(k=1\)–2 fresh judge cells, between-judge variance is essentially not estimable, so the fresh DL probabilities are not strong evidence. The fix is either to relabel these as approximate frequentist/meta-analytic sensitivity calculations or fit an actual hierarchical Bayesian model on the full pid-by-judge verdict array.

- **The selection-bias model is useful but not decisive, and the paper sometimes treats it as more explanatory than warranted**  
  Exp. 45 fits an empirical-Bayes prior to the same 12 observed gate scores, assumes exchangeability, and models remeasurement under a noisy selection process, while the observed L1 drops also change judge family and therefore change the estimand. The paper later notes that controls show the 12 candidates are not exchangeable with random text, which weakens the fitted-prior story. This analysis supports “L1 alone is ambiguous,” but not a strong causal decomposition of cached drops into selection bias versus judge effects. The fix is a prespecified hierarchical selection model with separate same-judge retest and cross-judge components, ideally estimated on more candidates.

- **The paper’s core theoretical decomposition is explicitly non-identifiable and contributes little formal methodology**  
  Sec. “What the A/B gate actually measures” introduces \(Z^{\mathrm{specific}}, Z^{\mathrm{generic}}, Z^{\mathrm{style}}\), but repeatedly states it is informal, non-unique, and not estimable from the experiments. This is fine as intuition, but for a top-venue methodology paper it leaves the audit stack without a formal estimand or decision-theoretic criterion. A fix would define target estimands for each layer, specify decision rules and error rates, and show under what assumptions the combined audit improves false-positive/false-negative tradeoffs.

- **The evidence streams are not independent, and the paper still sometimes rhetorically benefits from their count**  
  The abstract and Introduction correctly say the layers are “conditionally distinct” rather than independent, but many later phrases—“six audit layers,” “every conditionally-distinct axis,” “fifth independent layer,” “sixth audit layer”—risk overstating cumulative evidence. Many layers reuse the same KEEPs, cached answer pairs, problem pools, judges, or solver outputs. The fix is to present a dependency graph and use a single integrated analysis rather than counting layers as quasi-independent confirmations.

- **Low judge agreement is both a finding and a threat to the paper’s conclusions**  
  Exp. 37 reports mean \(\kappa\) around 0.18–0.26, and Exp. 25 finds poor reliability for some labels, especially \(S_{\text{anti}}\) with mean \(\kappa=0.14\). The paper uses this to argue against same-family judging, but it also means the audit verdicts are noisy and weakly validated. For an audit methodology, the judge panel itself must be validated. The fix is calibration against human labels or objective outcomes, plus reliability-aware aggregation rather than thresholding individual noisy judges.

- **There are internal inconsistencies and confusing references that reduce trust**  
  Exp. 31 says W078 was the sole candidate surviving majority-of-3 at \(n=50\) “Exp 11,” but the majority-of-3 result is Exp. 7. The Limitations section says Exp. 39 has \(0/6\) English wisdoms clearing \(0.60\) under any single family across two runs, but Exp. 39 itself states an earlier run had EW06 with gemini \(0.63\). These are not central, but in a paper with many experiments and changing conclusions, such inconsistencies make the audit trail harder to trust. The fix is a consistency pass and a compact canonical results table.

- **Handling of missing/invalid outputs is under-specified**  
  Exp. 47 has zero ties but \(n_{\mathrm{eff}}\) ranges from 26–29 out of 30 due to “dropped responses per cell,” yet the main decision rule treats the remaining valid outputs as the denominator. If invalid outputs correlate with base/ext condition or candidate difficulty, this can bias win rates at small \(n\). The fix is to preregister missingness handling, report invalid counts by condition and candidate, and run conservative sensitivity analyses counting invalid ext responses as losses or rerunning failed calls.

- **The same-domain exemplar exposure weakens claims about held-out evaluation**  
  The Limitations section discloses that, at solve time, each evaluation pid receives a same-domain exemplar selected from other pids in the evaluation pool. The authors argue this is symmetric between base and ext, which helps the pairwise comparison, but it still contaminates the evaluation distribution and may affect scaffold-vs-baseline and generalization claims. The fix is a clean rerun with strict pool separation and no in-pool exemplar retrieval.

- **The paper is structurally unwieldy and reads like a revision log rather than a clean methodology paper**  
  The main body contains dozens of experiments, reviewer-response language, “strict-reviewer closure,” retractions, caveats, and intermediate results. This transparency is admirable, but it obscures the central contribution and makes it hard to identify the final protocol. A top-venue version should move most exploratory and superseded experiments to appendices, present one locked audit algorithm, one primary prospective evaluation, and a small number of secondary analyses.

- **Some novelty claims are overstated or unsupported**  
  Statements such as “to our knowledge, the first time…” in Exp. 3 and the broad recommendation that L1 should become routine are plausible but not strongly supported by a systematic literature review or broad empirical study. The related-work table is useful, but absence of reported L1 in a few adjacent papers is not enough to establish novelty or necessity. The fix is to tone down priority claims and frame the contribution as a well-documented case study plus reusable protocol.

# Questions to the authors

1. Can you run the full closed loop—not just re-evaluate the original 12 candidates—on a fresh problem split with the L1–L6 audit protocol and thresholds fixed before candidate generation?

2. For Exp. 47, what are the invalid/dropped outputs by candidate and by condition, and do the pass/fail decisions change under conservative missingness rules?

3. Can you provide human annotations for a stratified subset of the cached and fresh base/ext answer pairs, especially W077 and W078, to calibrate whether the cross-family or original judge is closer to human preference?

4. Can you construct a true known-positive control where adding a wisdom is guaranteed or independently validated to improve objective performance, and show that the audit stack accepts it?

5. Why was \(0.55\) chosen as the preregistered L1 threshold for Exp. 47, given that the original gate threshold was \(0.60\), and what evidence existed before the preregistration that might have made \(0.55\) attractive?

6. Do W077 and W078 still pass the fresh-data protocol if Exp. 47 is scaled to \(n=100\) with the same thresholds and judges?

7. Can you replace the DerSimonian–Laird “posterior” analysis with an actual hierarchical Bayesian model over pid-level correlated judge verdicts, and does it change the W077/W078 fresh-data interpretation?

# Rating

Weak Reject

The submission is unusually transparent and contains a useful practical audit checklist, but the evidence is not yet strong enough for a top-venue methodology paper. The most important issues are the single-loop/small-\(n\) empirical base, the post-hoc construction of most of the audit stack, the lack of human or known-positive validation, and the threshold-sensitive fresh-data replication. I also found the statistical framing in Exp. 49 and some of the binary “replicate/collapse” language too strong relative to the uncertainty. I would be much more positive on a shorter version centered on a fully preregistered fresh-loop replication with human/known-effect calibration.

# Confidence

4. I am familiar with LLM-as-judge evaluation, self-improvement/self-refinement loops, selection effects in adaptive evaluation, and reproducibility/preregistration issues, though I am not a specialist in this particular “wisdom library” style of retrieval scaffold.