# Summary

The paper builds a self-improving retrieval-scaffolded LLM loop that proposes “wisdom” entries, gates them by same-family LLM-judged A/B win rate, and then audits the accepted entries using a six-layer stress-test stack. Its main empirical story is differentiated: the original three KEEPs fail to reproduce on cached answer pairs under stricter cross-family / extended / expensive-judge audits, but a preregistered fresh-data re-evaluation of the fixed original 12 candidates recovers W077 and W078 at a laxer L1 threshold while W076 fails. The paper argues that cross-family re-judgment, selection-bias modeling, and preregistered fresh-data replication should be routine for LLM-judged self-improvement claims.

# Strengths

- **Clear and important target failure mode: same-family self-adjudication.** The Introduction and Experiment 1 focus on a concrete and relevant risk: a generator and judge from the same model family may reward stylistic alignment rather than substantive improvement. This is a timely issue for LLM-as-judge self-improvement papers and makes the case study potentially useful beyond the specific “wisdom library” system.

- **Honest reporting of negative and mixed results.** The abstract, Contribution 2, Exp. 45, and Conclusion repeatedly acknowledge that cached-data L1 drops are compatible with regression to the mean, that Exp. 47 is not a fresh full loop, and that the “2/3 replicate” result is point-estimate / threshold-sensitive. This transparency is a major strength relative to many self-improvement papers.

- **Operational audit layers are concretely specified.** The six layers in the abstract / Introduction and detailed experiments give actionable checks: cross-family re-judgment, side reseeding, sample extension, cross-solver replication, fresh-domain tests, and faithfulness probes. Even if not all layers are novel, the paper provides a reusable checklist for auditing similar systems.

- **Selection-bias analysis materially improves the interpretation.** Exp. 45 explicitly models the top-3-of-12 winner’s-curse effect and correctly narrows the interpretation of cached L1 drops. This prevents an overstrong “judge fragility” conclusion from L1 alone and strengthens the paper’s scientific honesty.

- **Preregistered fresh-data replication is a meaningful corrective to post-hoc auditing.** Exp. 47 freezes thresholds and a fresh split before evaluation, and the paper carefully clarifies that candidate generation and pruning are not rerun. This is the strongest evidence in the paper and directly addresses a central hindsight-bias concern.

- **The paper surfaces reproducibility instability rather than hiding it.** Sections on Exp. 37, Exp. 39, and Exp. 42 document hosted-model drift and run-to-run variation even at nominally fixed settings. This is useful evidence for the broader LLM evaluation community.

- **Artifact-oriented contribution appears substantial.** The Conclusion and Contribution 7 promise code, logs, candidate records, answer pairs, audit harnesses, and registry states. For a case-study / methodology paper, preserving provenance and releasing audit traces is important.

# Weaknesses

- **The central empirical basis is far too small for the claimed diagnostic separation.** The main differentiated conclusion rests on only three original KEEPs and a fresh replication with about 26–30 effective examples per candidate in Exp. 47. This is especially problematic for a methodology paper claiming the audit stack can “separate” selection-driven cached drops from genuine non-replication: with three positives, one failure, and threshold-sensitive point estimates, the evidence is only anecdotal. A fix would require applying a frozen audit protocol to multiple independently generated loops, more candidates, and fresh splits of at least \(n \ge 100\) per candidate.

- **The audit stack is mostly post-hoc and not validated as a methodology.** The “note on staged stress testing” in Sec. Experiments states that L1–L6 were added in response to reviewer-simulation objections, not preregistered before the loop. This is acceptable as exploratory case analysis, but not sufficient to establish the audit stack as a reliable method, because layer selection itself is outcome-adaptive. The fix is a prospective study: freeze L1–L6, thresholds, tie handling, and decision rules; then run a new full loop including candidate generation, pruning, and audit.

- **Exp. 47 does not replicate the self-improving loop, only re-evaluates fixed candidates.** The paper clarifies this in Exp. 47 and the Conclusion, but the title and broad framing still invite a stronger reading than the evidence supports. Since candidate generation, pruning, and the full loop are not rerun, Exp. 47 cannot validate the full closed-loop methodology or the proposed audit stack end-to-end. The fix is a genuine fresh-loop replication with all candidate-generation modules rerun on new data and then audited under the frozen protocol.

- **Threshold sensitivity undermines the “2/3 replicate” headline.** In Exp. 47, W077’s L1 score is 0.57 and WCAND03’s L1 score is 0.57; both pass only because the preregistered L1 threshold is 0.55 rather than the original 0.60. The paper acknowledges that at 0.60 only W078 clears both gates, but the abstract and contribution still foreground “2/3 replicate.” The fix is to make the threshold-sensitive result the main headline, report both thresholds symmetrically, and avoid treating W077 as a robust replication without larger fresh \(n\).

- **The posterior analyses are internally inconsistent in their decision criteria.** Exp. 47 reports the exact joint posterior for the preregistered rule: W078 0.539, W077 0.370, WCAND03 0.424. Exp. 49 then labels W077 as “replicates” using a DerSimonian–Laird pooled \(P(\theta>0.55)=0.808\), which is not the same as the preregistered joint rule requiring inner \(>0.60\) and L1 \(>0.55\). For this class of audit paper, changing from a joint gate to a pooled probability after the fact is a serious interpretive inconsistency. The fix is to use one primary decision estimand throughout, with all alternative posteriors clearly secondary.

- **Independence assumptions for judge posteriors are not justified.** Exp. 47’s joint posterior assumes independence between the inner and L1 judges on the same answer pairs, and Exp. 49 pools judge cells despite shared pids, shared answer content, and correlated judge errors. The paper notes this limitation, but still uses the resulting probabilities to support replicate/collapse labels. The fix is a hierarchical model over pid-level paired outcomes with judge-family random effects, or a conservative bootstrap over pids that treats judge verdicts on the same pid as correlated.

- **No human ground truth for the main open-ended task.** Nearly all core measurements are LLM preferences, with gpt-5.5 used as a “human-substitute” in Exp. 38/41/48 and GSM8K used only as a small objective side probe. For auditing self-improvement claims, cross-family LLM disagreement is useful but does not establish which answer is actually better. The fix is a blinded human evaluation on a representative subset of the Chinese open-ended tasks, ideally with expert and non-expert raters and preregistered rubrics.

- **Positive controls do not establish sensitivity of the audit stack.** Exp. 44 and Exp. 46 construct placebo, random, generic-useful, math-specific, SCQA, and duplicate controls, but all fail. The paper later treats WCAND03 in Exp. 47 as an empirical positive control, but this is not a known-positive intervention; it is another candidate selected by the same evaluation process. A methodology paper needs known positives and known negatives to characterize false positive and false negative behavior. The fix is to design synthetic tasks where a specific inserted wisdom deterministically helps, or use independently human-validated wisdoms as positive controls.

- **There are concrete internal contradictions and stale claims.** In Sec. theory, the paper says the cached-data null survives because “no audit layer at any threshold assigns a cached-data wr \(\ge 0.60\) point estimate to the three KEEPs,” but Exp. 40 reports cross-solver cells and even 3-family means above 0.60 for W078 under new solver families. Exp. 39 says an earlier English run had EW06 with gemini 0.63, then states the invariant headline is that “no English wisdom clears the gate under any family in either run.” These contradictions reduce confidence in the paper’s synthesis. The fix is a full consistency pass that distinguishes cached original-solver audits, cross-solver audits, single-judge cells, means, and consensus rules.

- **The theoretical decomposition is only a taxonomy, but the prose sometimes implies causal isolation.** Sec. “What the A/B gate actually measures” correctly says the \(Z^{\mathrm{specific}}, Z^{\mathrm{generic}}, Z^{\mathrm{style}}\) decomposition is informal and non-identifiable. However, the later bullet list says layers “cancel” specific nuisance terms, e.g. L1 “cancels \(J\)-specific stylistic preferences,” which is stronger than justified. For this paper type, causal language matters because the contribution is an audit methodology. The fix is to replace “cancel” with “stress” or “probe,” define explicit estimands, and avoid causal interpretations unless supported by placebo and known-effect interventions.

- **The same-domain exemplar exposure weakens the held-out interpretation.** The Limitations section states that at solve time the prompt includes a same-domain exemplar selected from the evaluation pool’s other pids, with cached v15-class answer text. The authors argue this is symmetric between base and ext, but it still contaminates held-out evaluation and may interact asymmetrically with the added wisdom. For a paper about auditing subtle gate effects, this is not a minor issue. The fix is a clean rerun with strict pool separation and no in-pool exemplars.

- **The paper’s presentation is sprawling and contains too many overturned intermediate analyses.** Experiments 1–49, many of which are chronological “reviewer objection” responses, make the main contribution hard to evaluate. Sections such as Exp. 7 explicitly present an “intermediate verdict” later overturned, and Exp. 27 retracts an earlier F1 framing. This transparency is appreciated, but a top-venue paper should separate exploratory logbook material from the core preregistered evidence. The fix is to move most chronological experiments to appendices and center the paper on a small number of frozen protocols and results.

- **Several novelty claims are overstated.** The paper states that L1 and L6 are the core methodological novelty, while also admitting that L2–L5 are standard statistical hygiene. Cross-family judging, sample extension, fresh-domain testing, and cross-solver testing are sensible but not new; the novelty is mainly packaging plus a single case study. The fix is to either substantially validate the package across systems or narrow the contribution to “case-study evidence motivating a reporting checklist.”

- **Reliance on hosted models through a third-party proxy limits reproducibility.** The Limitations section notes ruoli.dev proxy routing and observed temporal drift in Exp. 37/39/42. This is especially problematic because many key effects are on the order of 0.05–0.10, comparable to observed drift. The fix is to cache and release all answer/judge texts, rerun critical cells through official endpoints or open-weight judges where possible, and report repeated-run variance as part of the primary uncertainty.

# Questions to the authors

1. If Exp. 47 were rerun at \(n=100\) fresh pids with the original L1 threshold of 0.60, do you expect W077 to pass? What sample size would be needed for W077’s exact joint posterior under the preregistered rule to exceed 0.95?

2. Why should Exp. 49’s DerSimonian–Laird pooled \(P(\theta>0.55)\) supersede Exp. 47’s exact joint posterior for the preregistered decision rule? Which of these is the primary inferential target?

3. Can you provide a clean full-loop replication plan in which candidate generation, pruning, gate decisions, and all six audit layers are frozen before any fresh-loop data are observed?

4. How much do the main cached and fresh results change if same-domain in-pool exemplars are removed entirely from both base and ext prompts?

5. Do the released artifacts include all raw answer texts and judge rationales needed to let an external human or LLM judge re-score the same pairs without relying on the ruoli.dev proxy?

6. Can you construct a true known-positive control for the audit stack, e.g. a synthetic task family where one inserted methodological instruction is guaranteed or independently verified to improve outputs?

7. For the main Chinese open-ended pool, what is the agreement between human raters and the inner/audit LLM judges on a representative subset of candidate answer pairs?

# Rating

Weak Reject

The paper is interesting, unusually transparent, and potentially valuable as a cautionary case study, but it is not yet a strong top-venue methodology contribution. The main problems are the tiny empirical base, the largely post-hoc construction of the audit stack, lack of human or known-positive validation, and threshold-sensitive fresh-data claims. The statistical interpretation is also not fully consistent: the exact preregistered joint rule, point-estimate threshold crossings, and later pooled Bayesian labels support different strengths of conclusion. I would be much more positive after a frozen, prospective full-loop replication with larger fresh \(n\), clean pool separation, and at least some human / known-positive calibration.

# Confidence

4 — I am familiar with LLM self-improvement and retrieval-loop evaluation, LLM-as-judge failure modes including cross-family judging and position bias, and reproducibility / preregistration issues in empirical ML, though I am not a specialist in this specific “wisdom library” implementation.