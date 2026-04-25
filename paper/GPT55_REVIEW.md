# Summary

The paper builds a closed-loop “wisdom library” curation system for a scaffolded LLM solver, lets it propose and gate new retrieval entries, and then audits the three entries accepted by the inner-loop same-family A/B gate. Its main empirical claim is a null result: the accepted wisdoms do not robustly survive a multi-layer audit involving cross-family re-judging, sample extension, cross-solver tests, fresh-domain probes, expensive judges, faithfulness analyses, and objective GSM8K checks. The claimed methodological contribution is the audit stack and the case-study evidence that a default single-family LLM-as-judge gate is fragile.

# Strengths

- **Clear high-level scoping of the claim.** The abstract and “Scope and non-claims” in the Introduction explicitly state that this is a methodology/case-study paper, not a field-level claim that self-improving LLM loops fail. This restraint is important because the empirical base is small and the paper repeatedly acknowledges that cross-family disagreement is not the same as human-ground-truth falsification.

- **Valuable negative result with unusually extensive self-audit.** The paper reports that the loop’s apparent \(3/12\) KEEP outcome collapses under multiple probes, rather than only reporting the inner-loop success. The chronological retention of intermediate positive-looking results, e.g. Exp. 7’s majority-of-3 apparent survival of \(W_{078}\) and its later overturning in Exp. 8/31/32/34, strengthens the submission by showing the authors did not simply hide failed hypotheses.

- **Operationally concrete L1 audit.** Experiment 1 re-judges the exact same cached answer pairs with a different model family, holding answer content fixed. This is a simple, cheap, and practically useful audit intervention for LLM-as-judge self-improvement papers, and the reported \(0.64 \to 0.40\), \(0.60 \to 0.47\), \(0.60 \to 0.51\) drops are directly interpretable as lack of robustness to judge family.

- **Good acknowledgement of statistical fragility.** Section “Binomial confidence intervals on headline claims” reports Wilson intervals and explicitly notes that many CIs overlap both parity and the \(0.60\) threshold. This is much better than the common practice of treating \(n=50\) point estimates as decisive.

- **Reproducibility-oriented artifacts.** The main body claims release of code, candidate records, cached answer pairs, judgment files, registry states, and logs, and the prompt says the appendix includes model identities, temperatures, seeds, proxy details, cost breakdown, prompt templates, and schemas. For a methodology paper, this kind of provenance is a genuine strength.

- **Attempts to separate preference from faithfulness.** Experiments 9–14 and 15–17 try to move beyond scalar pairwise win rate by measuring perturbation magnitude, embedding/LLM faithfulness, citation, trigger-fit, and conditional utility. The operationalizations are imperfect, but the motivation is right: pairwise preference alone is not enough to establish that a retrieved “wisdom” caused the claimed reasoning improvement.

- **Honest limitation reporting.** The Limitations section admits serious issues, including held-out contamination through exemplar mining, post-hoc audit-stack construction, single-cycle scope, no human ground truth, no positive controls, proxy-based reproducibility, and L6 under-specification. This candor makes the paper more trustworthy, even though several of these issues remain fatal for acceptance.

# Weaknesses

- **Held-out contamination undermines the inner-loop gate being audited.** In the Limitations section, the authors disclose that candidate exemplars were mined from the same 50-problem held-out set used for the A/B gate, and may even include the evaluated PID itself. For this class of paper, the central object of study is the gate’s KEEP decision; if the gate was partially exposed to the evaluation distribution, then the original \(0.64/0.60/0.60\) KEEP scores are not clean held-out evidence. A clean rerun with exemplars mined only from a separate development pool and a frozen evaluation set is necessary.

- **The audit stack is post-hoc and not validated on a fresh loop.** Section “A note on staged stress testing” says the six layers were added in response to reviewer-simulation objections and were not pre-registered. That is acceptable for exploration, but the paper presents the stack as the main methodological contribution and recommends routine reporting based on it. A convincing methodology paper should apply a frozen audit stack to a fresh loop or at least to a fresh batch of candidates, with all thresholds and survival criteria fixed in advance.

- **No positive controls, negative controls, or human-ground-truth calibration.** The Discussion explicitly admits there are no positive/negative controls for the audit stack and no human ground truth on the open-ended Chinese pool. This is severe: the paper shows the stack rejects the three KEEPs, but not that it would accept a genuinely useful library addition or reject placebo/random context at a known rate. The fix is to include placebo wisdoms, length-matched generic context, deliberately useful task-specific hints, human-curated improvements, and human or objective labels on a representative subset.

- **Small, selected sample makes regression-to-the-mean a major alternative explanation.** The headline \(0/3\) survival result is computed on three candidates selected from the right tail of 12 initial candidates, with many measurements at \(n=50\) or \(n=20\). The paper discusses CIs but does not model the winner’s-curse/selection effect: under a noisy gate, selected high point estimates are expected to drop on remeasurement even without any special “judge-fragility.” The authors should provide a selection-adjusted analysis, e.g. a hierarchical model or simulation of expected post-selection shrinkage under null and alternative assumptions.

- **Several central claims are internally inconsistent.** The abstract says “None of the three KEEPs reaches \(\mathrm{wr}_{\mathrm{ext}}\geq0.60\) as a point estimate under any audit-stack measurement we report,” but Experiment 40 reports 7/18 cross-solver cells at or above \(0.60\), including \(W_{078}\) with a 3-family mean of \(0.63\) under the Claude solver and \(0.61\) under the GPT-mini solver. Similarly, the Introduction says the six layers “share no data with the others,” while the abstract correctly says L1/L2 reuse cached answer pairs and L3/L4 overlap in solver/problem pool. The paper needs a single precise definition of “survive” and should remove or correct all stronger inconsistent statements.

- **The interpretation of cross-family disagreement is too strong in places.** The theory section carefully says cross-family re-judgment falsifies robustness to judge choice, not substantive utility. But Experiment 1’s interpretation says the gate is “measuring gemini-3-flash’s preference for answers that activate certain rhetorical structures,” and the abstract/conclusion use “validated library delta \(+0\).” Without human or objective calibration, another judge family may simply have different or worse preferences; the supported claim is non-robustness of the LLM-judge verdict, not absence of utility on the original open-ended task. The fix is to calibrate judges against human labels or task-grounded outcomes and consistently phrase the claim as robustness failure.

- **Tie handling is fragile and judge-dependent.** Section “Tie handling and effective \(n\)” excludes ties from the denominator, but tie rates differ substantially across judges, e.g. Claude has many ties in Exp. 1 and GSM8K subjective judgments have 20+ ties out of 30. Excluding ties changes the estimand and can make judges with different tie propensities look incomparable. The paper should report sensitivity analyses treating ties as half-wins, as a third multinomial outcome, and via ordinal/preference models.

- **Hosted-model temporal drift confounds several claimed interventions.** Section “Experiment 37” reports that the same gemini family at temperature 0 drifts from \(0.64\to0.47\) on \(W_{076}\), and Experiment 42 reports base GSM8K accuracy drifting from 30/30 to 28/30. This weakens claims that L2 isolates side-randomization or that repeated measurements differ only by a named nuisance variable. The fix is repeated adjudication with multiple independent runs, official versioned endpoints where possible, and modeling run-to-run variance explicitly.

- **The L6 faithfulness measurements are not yet validated.** Experiments 9–11 use embedding alignment between Turn-0 `what_changed` deltas and wisdom text, plus a single LLM’s YES/PARTIAL/NO judgments. These are plausible probes but not established measures of causal faithfulness. For a methodology paper, L6 needs validation on synthetic cases with known inserted effects, human faithfulness annotations, and placebo/length-matched controls.

- **The random-30% baseline is arbitrary and sometimes overinterpreted.** Experiments 33 and 15 compare pass rates such as \(0/9\), \(3/21\), or \(4/21\) to a “random-inclusion” baseline of 30%, while also acknowledging that 30% is arbitrary. This does not establish utility or gate quality; it only says the observed pass rate is below an arbitrary reference. A meaningful baseline would compare against always-accept, random wisdoms, length-matched generic context, human-curated candidate entries, and pre-registered alternative gates on the same external criterion.

- **The cross-solver and fresh-domain audits are uneven.** Exp. 26 tests only \(W_{076}\) on 20 fresh PIDs; Exp. 31 tests only \(W_{078}\); Exp. 33 tests the 9 new candidates with \(n=20\) and gemini-only judging; Exp. 40 is fuller but yields mixed positive cells. This unevenness makes the “six-layer” story look more like accumulated probes than a balanced factorial audit. A cleaner design would test all KEEPs and a control set across the same PIDs, solvers, judges, and sample sizes.

- **The architectural recovery story is confusing and partly inconsistent.** The abstract says scaling to 21 candidates in Exp. 33 yields \(3/21\) PASS, while the Introduction and Exp. 15 say the trigger-conditioned gate locally rescues \(4/12\), prospectively gets \(0/9\), and thus gives \(4/21\). These appear to refer to different gates, but the text sometimes presents them as the same “architectural recovery.” The authors should separate naive-gate pass rate, trigger-conditioned-gate training performance, and prospective trigger-conditioned performance in one table.

- **The scaffold performance claims are under-supported and internally unclear.** The abstract claims v20 is \(+28\)pp vs. v16 on 100 problems and \(+76\)pp vs. a budget-matched baseline on 50 held-out, while Figure 1/Figure 2 text elsewhere mentions \(86\%\) vs. \(74\%\) and \(0.86\to0.88\). Since the paper’s audit depends on having a strong scaffold as instrument, these numbers should be made consistent, tabulated, and clearly labeled as same-family-judged rather than independently validated.

- **The English replication does not test the same claim.** Exp. 39 uses a fixed six-entry English wisdom library on MT-Bench, not a full English closed-loop candidate-generation/gating/audit process. It is useful as an extra stress test, but it cannot remove the “Chinese-only” limitation for the self-improvement loop. A real replication would need English candidate generation, English exemplar mining, English gate decisions, and the same pre-registered audit stack.

- **The paper is overloaded and hard to audit as a main-track methodology contribution.** The main body includes dozens of experiments, retractions, intermediate verdicts, architectural probes, and reviewer rebuttals, many of which are exploratory. This makes the central contribution difficult to verify and contributes to inconsistent claims. The paper should move exploratory chronology to appendices and present one clean frozen protocol, one primary result table, and one limitations table in the main text.

# Questions to the authors

1. If you rerun the inner loop cleanly with exemplars mined only from a separate development pool, do any of \(W_{076}, W_{077}, W_{078}\) still pass the original same-family gate, and does the audit-stack conclusion remain \(0/3\)?

2. How much of the L1/L3 drop would be expected from post-selection regression to the mean under a noisy \(n=50\) gate? Can you provide a simulation or hierarchical Bayesian analysis conditioning on selecting the top three of twelve candidates?

3. What is the human-preference or task-grounded calibration of the five LLM judges on a subset of the Chinese open-ended answer pairs? If human raters prefer the original gemini KEEPs, the interpretation of “audit failure” would change substantially.

4. What positive controls has the audit stack been tested on? For example, would it accept a human-written task-specific hint known to improve GSM8K or a deliberately useful domain-specific retrieval entry?

5. Please define “survive” formally. Does survival require every judge in a panel to exceed \(0.60\), a majority, the mean, a CI lower bound, or something else? How does this definition reconcile with Exp. 40’s cells and means above \(0.60\)?

6. What fraction of the held-out evaluation PIDs were used as exemplars for each candidate, and did any candidate include the same PID being evaluated as an exemplar? This matters for assessing the size of the contamination.

7. Can you provide a tie-aware reanalysis of the main L1/L3/Exp. 35 results, treating ties as half-wins and also as a third multinomial outcome?

8. How were the proxy-served model identities verified, and how stable are repeated judgments on official endpoints or repeated proxy calls? The reported temporal drift is large enough to affect several conclusions.

# Rating

Weak Reject

The submission is interesting, unusually honest, and potentially useful as an audit case study, but it is not yet a solid top-venue methodology paper. The most important issues are the contaminated inner-loop held-out gate, the post-hoc and unvalidated audit stack, the absence of human/positive-control calibration, and the very small selected sample with no selection-adjusted inference. Internal inconsistencies around what counts as “survival” and the mixed Exp. 40 cross-solver results further weaken the headline \(0/3\) claim. A clean pre-registered rerun with controls and calibrated evaluation could make this much stronger.

# Confidence

4 — I am familiar with LLM-as-judge evaluation, self-improvement/agent-loop methodology, and reproducibility/preregistration issues; I have somewhat less direct experience with this exact “wisdom-library” retrieval formulation, but the evaluation concerns are squarely within my area.