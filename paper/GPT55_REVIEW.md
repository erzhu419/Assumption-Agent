# Summary

The paper builds and audits a minimal closed-loop “wisdom library” agent: an LLM proposes Chinese methodological aphorisms, validates them using a same-family \(+10\)pp A/B gate at \(n=50\), and commits accepted entries. The inner gate accepts 3/12 candidates, but a sequence of audits—cross-family re-judgment, side re-randomization, sample extension, cross-solver checks, fresh-domain tests, and faithfulness probes—finds that none robustly clears the original acceptance threshold, with preregistered \(n=100\) and \(n=200\) fresh replications yielding 0/12 passes. The paper argues that this calibrated negative result shows same-family LLM-as-judge A/B gating is insufficient for retrieval/prompt-level self-improvement, and uses the failure to motivate a broader six-stage architecture for self-hypothesizing/self-validating agents.

# Strengths

- **Honest and useful negative result** (§Experiments, especially §§3.1–3.5; Conclusion).  
  The central empirical finding is not framed as “self-improvement works” but as “the apparent improvement does not survive audit.” This is valuable for a field where positive self-improvement claims are often vulnerable to judge leakage, overfitting, and selection effects.

- **Clear threat model for LLM-as-judge gates** (§2.5, “What the A/B gate actually measures”).  
  The decomposition into specific content, generic extra-context effects, and judge stylistic preference is explicitly labeled as non-identifiable and used as an organizing taxonomy rather than as a causal model. This restraint strengthens the paper’s methodological credibility.

- **Multiple independent audit axes rather than a single re-test** (§Introduction; audit stack L1–L6; Fig. 3).  
  Cross-family re-judgment, side reseeding, \(n\)-extension, cross-solver replication, fresh-domain testing, and non-pairwise faithfulness probes target different plausible failure modes. Even though the layers are post-hoc and not independent, the breadth of checks makes the null more convincing than a simple failed replication.

- **Preregistered fresh-data follow-ups address post-hoc audit concerns** (§3.5).  
  The authors acknowledge that the original audit stack was not preregistered (§3, “A caveat on the audit stack itself”) and then give more inferential weight to preregistered \(n=100\), \(n=100\)+exemplar, and \(n=200\)+exemplar replications. This is exactly the right corrective move.

- **Selection-bias analysis directly targets winner’s curse** (§3.3).  
  Modeling the “top-3-of-12 noisy measurements” process is appropriate for this setting and prevents over-interpreting the cross-family drops as purely judge-family fragility. The paper correctly distinguishes regression-to-the-mean from evidence that the gate is “wrong” in a stronger sense.

- **Good artifact/reproducibility orientation** (§2.2 registry; Contribution 7; Reproducibility paragraph).  
  The versioned registry, logged mutations, cached answer pairs, judgment files, model identities/seeds in the appendix, and preregistration document are strong practices for this class of empirical methodology paper.

- **The paper explicitly lists unresolved objections** (§Discussion, “Objections we have not closed”).  
  The authors are unusually candid about gate-design hindsight, post-hoc audit-layer selection, solver-family scope, lack of human ground truth, incomplete positive controls, and L6 under-specification. This transparency improves trust in the empirical narrative.

- **The \(n=30\) false-positive episode is pedagogically useful** (§3.4–3.5).  
  Reporting the small-\(n\) “positive” result and then showing that it disappears at \(n=100/200\) is a useful case study in how hierarchical Bayes plus small fresh samples can still be overconfident when protocol covariates are omitted.

# Weaknesses

- **The main contribution is empirically narrow relative to the paper’s architectural claims** (§1, §5, Conclusion).  
  The strongest evidence concerns one Chinese wisdom-library loop, one primary solver family, one candidate-generation cycle, and 12 candidates. The paper then motivates a broad six-stage architecture for “self-hypothesizing, self-validating agents,” including world models, schedulers, formal alignment layers, and new-prior generation. For a methodology/case-study paper, a scoped recommendation about auditing LLM-as-judge retrieval updates is well supported; the broader architecture reads as speculative extrapolation.  
  **Fix:** Split the paper or sharply demote the roadmap to discussion/future work; make the accepted contribution the audited negative result and concrete reporting checklist.

- **The six-stage roadmap validation is much weaker than the text sometimes claims** (§5.1, Table 1).  
  Table 1 says Stage 3 is only a “minimal proxy” and “formal alignment not tested,” yet the following paragraph says “Three stages work (priors library, experience feedback, alignment).” This is internally inconsistent: cross-language scheduler pick consistency is not evidence for formal prior-equivalence detection across domains. Similarly, Stage 0 “works” is based on four priors in a synthetic environment, not on a validated general prior library.  
  **Fix:** Rephrase the stage verdicts conservatively: Stage 3 is not validated; Stage 0 is probed only in a toy setting; the roadmap experiments are diagnostics, not architectural validation.

- **The \(n=50\), \(+10\)pp gate is repeatedly described as “strict,” but statistically it is weak** (§2.4; Discussion “The +10pp gate is strict by design”).  
  At \(n=50\), clearing 0.60 means roughly 30/50 wins, which is not strong evidence against 0.5 under a binomial model; the standard error near 0.5 is about 0.07. The paper eventually demonstrates this empirically, but still calls the original gate strict and says \(50\) problems is “enough to distinguish \(+10\)pp from noise” in the Limitations. That statement is false for conventional error control.  
  **Fix:** State plainly that \(n=50\), \(+10\)pp is a common but underpowered operational gate, not a statistically strict one; report exact binomial/Wilson intervals and type-I error properties for the gate.

- **The positive control does not establish positive sensitivity of the gate** (§3.6, Exp 67).  
  The “POSITIVE” intervention yields 0/16 objective accuracy, same as BASE and NEGATIVE, so it is not a true positive control. The pairwise judge prefers POSITIVE over NEGATIVE and rejects NEGATIVE, but that only shows the judge can distinguish reasoning-rich from reasoning-suppressed text, not that the gate accepts objectively useful content. The authors admit this, but the abstract still says the audit stack is “sharply sensitive” and “operating as designed,” which overstates the evidence.  
  **Fix:** Add a genuine positive control where an intervention improves objective task accuracy by \(\geq 10\)pp and test whether the gate accepts it; until then, call Exp 67 a negative-control/style-sensitivity check, not a positive-and-negative control.

- **Human ground truth is missing for the main open-ended task distribution** (§Discussion; Limitations).  
  The central conclusion depends on LLM judges disagreeing with the original same-family gate and on LLM-substitute proxies such as GPT-5.5 ratings and GSM8K transfer. For open-ended Chinese methodological tasks, cross-family disagreement is not equivalent to lack of substantive utility. This is especially important because the paper argues about judge stylistic preference, yet uses other LLM judges as the primary correction.  
  **Fix:** Include a human evaluation on a stratified subset of cached and fresh pairs, ideally blinded and with Chinese-fluent raters, to estimate whether cross-family LLM judgments track human preferences or objective usefulness.

- **The same-domain in-pool exemplar mechanism changes the estimand and may interact with candidate wisdoms** (§Limitations, “Some held-out exposure”; §3.5).  
  The authors argue the exposure is symmetric because base and ext receive the same same-domain example, but symmetry does not imply cancellation: the new wisdom may interact with the exemplar or retrieval context. The later no-exemplar/with-exemplar sensitivity is useful, but the main original gate accepted candidates under a contaminated protocol relative to standard held-out evaluation.  
  **Fix:** Make the strict pool-separated, no in-pool-exemplar protocol the primary evaluation from the start, or report the original gate as measuring “incremental value conditional on in-pool exemplar prompting,” not general held-out utility.

- **Selection-bias model details are under-specified and the fitted prior looks pathological** (§3.3).  
  The empirical-Bayes Beta prior has \(\hat\alpha+\hat\beta \approx 6\times 10^7\), implying essentially no between-candidate variability. With only 12 candidates and \(n=50\) per candidate, that estimate is likely an artifact of marginal likelihood at the boundary rather than a reliable population prior. The qualitative point—winner’s curse—is correct, but the percentile claims rely on a fragile model.  
  **Fix:** Present sensitivity analyses with weakly informative priors, finite-concentration hierarchical models, nonparametric bootstrap over candidates, and explicit raw counts for all 12 candidates.

- **The audit stack is post-hoc and the paper’s own strongest recommendation was not applied prospectively to a new loop** (§3 caveat; Discussion objections).  
  The authors are transparent that L1–L6 were added sequentially in response to objections. The preregistered fresh-data replications test the original 12 candidates, but they do not rerun candidate generation, selection, and audit end-to-end under a fully preregistered protocol. For a methodology paper advocating a default audit standard, this limits causal/inferential strength.  
  **Fix:** Run a second full loop with the audit protocol and thresholds fixed before candidate generation, then report the resulting acceptance/rejection trajectory.

- **Only one full evolution cycle is studied** (§Discussion; Limitations).  
  The title and framing evoke autonomous self-improvement, but the empirical case is one cycle with final library delta \(+0\). This is enough for a negative audit case study, not for claims about loop dynamics, convergence, drift, or long-horizon accumulation.  
  **Fix:** Either remove “growing”/closed-loop improvement framing or add multi-cycle experiments with staged held-out splits and audit at each cycle.

- **Solver-family scope remains limited despite cross-solver audits** (§Discussion objections; Limitations).  
  The original loop’s candidates, gate, and scaffold are all built around gemini-3-flash. Cross-solver L4 audits on accepted candidates help, but they do not show whether a non-Gemini solver would propose different candidates, pass different gates, or fail differently. Since the claim concerns same-family gate unreliability, solver–judge family entanglement is central.  
  **Fix:** Run at least one full candidate-generation/gating/audit cycle with a different solver family and preferably a different default judge family.

- **Related-work comparison risks overgeneralizing from “not reported” to methodological deficiency** (§Related Work, Table 1; Discussion “What this implies for the rest of the literature”).  
  The table is useful for LLM-judged retrieval/prompt-level loops, but the text also gestures toward self-rewarding LMs, STaR-style bootstrapping, self-play, and weight-level training. Those systems may use reward models, environment feedback, or downstream benchmarks, so cross-family re-judgment of cached outputs is not always the relevant audit. The paper sometimes acknowledges this, but the discussion still implies a broad indictment.  
  **Fix:** Restrict the recommendation to systems whose primary acceptance signal is an LLM preference verdict over open-ended outputs; treat weight-update and environment-reward systems separately.

- **L6 faithfulness probes are too weak to support “faithfulness” claims** (§2.5; Discussion “What is actually novel”; limitations).  
  Embedding-direction alignment and LLM-judged citation strictness are plausible diagnostics but not validated measures of whether the wisdom causally influenced reasoning. The authors admit this, but L6 is still included in the six-layer stack and abstract-level toolbox.  
  **Fix:** Rename L6 to “auxiliary descriptive probes” unless validated on synthetic cases with known causal wisdom effects and human labels.

- **Some numerical statements are confusing or inconsistent across the paper**.  
  Examples: the scaffold win rate is reported as 86% vs 74% in the abstract/intro figures, 0.64 vs v16 and 0.88 vs baseline in Contribution 3, and the final conclusion mentions \(n=100\) fresh replications but omits the \(n=200\) result that the abstract emphasizes. These may refer to different comparisons, but the main body does not make the distinctions easy to track.  
  **Fix:** Add a compact table of all headline comparisons with rows for task split, solver, judge, \(n\), baseline, metric, and whether audited.

- **The paper is sprawling and mixes case study, audit methodology, roadmap manifesto, synthetic diagnostics, and pilot studies**.  
  For a top-venue main-track paper, the argument would be stronger if it were narrower and more disciplined. The multi-step research-task pilot with \(N=3\), the broad philosophical introduction, and the six-stage architecture distract from the strong negative-result/audit contribution.  
  **Fix:** Move the pilot and much of the roadmap to appendix or future-work discussion; foreground the audit protocol, preregistered replications, and statistical analysis.

# Questions to the authors

1. For all 12 original candidates, what are the raw inner-gate counts, including ties, and the exact binomial/Wilson intervals under the original \(n=50\) gate?

2. How sensitive is the selection-bias conclusion to alternative hierarchical priors that do not collapse to \(\alpha+\beta \approx 6\times 10^7\)? In particular, do the “drop percentile” results hold under weakly informative finite-concentration priors?

3. Were any human raters used informally during development to inspect the Chinese open-ended answer pairs? If so, how did their preferences compare to gemini, Claude, and GPT-family judges?

4. In the \(n=100\) with-exemplar and no-exemplar replications, were the same candidate prompts, retrieval decisions, and side-randomization procedures used exactly, except for the exemplar ablation? Any candidate-specific prompt changes would materially affect interpretation.

5. Can the authors provide a single consolidated table of the three preregistered fresh replications showing, for each candidate, inner win rate, L1 win rate, counts, ties, and confidence intervals?

6. What would happen if the original acceptance gate were replaced prospectively by “inner \(\geq 0.60\) and cross-family L1 \(\geq 0.55\)” on a newly generated candidate set rather than retrospectively on the original 12?

7. For Exp 67, can the authors construct or identify a task set where the POSITIVE intervention actually improves objective accuracy? Without that, the positive-control question remains unresolved.

# Rating

**Weak Reject**

The paper contains a valuable and unusually transparent negative case study showing that one same-family LLM-as-judge retrieval-update loop produces false-looking accepts under audit. However, the evidence is still too narrow and partially post-hoc for the broader methodological and architectural claims being made. The most rating-relevant weaknesses are the absence of human ground truth on the main open-ended task, the lack of a genuine positive control demonstrating gate sensitivity to objectively useful interventions, the post-hoc audit-stack construction without a fully prospective second loop, and the overextended six-stage roadmap. I would be much more positive on a tighter version framed as an audited negative result plus preregistered replication protocol, with the speculative architecture sharply reduced.

# Confidence

**4**

I am familiar with LLM self-improvement loops, LLM-as-judge evaluation failure modes, winner’s-curse/selection-bias issues, and reproducibility/preregistration standards for empirical ML papers, though I am not a specialist in Chinese-language aphorism-based prompting specifically.