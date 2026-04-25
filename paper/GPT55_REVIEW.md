# Summary

The paper presents a case study of an autonomous retrieval-library “wisdom” improvement loop for an LLM solver, then audits the loop’s three accepted wisdoms using cross-family judging, reseeding, sample extension, cross-solver tests, fresh-domain probes, and faithfulness measurements. Its main empirical story is deliberately differentiated: on cached answer pairs none of the three KEEPs reproduces the original \(0.60\) gate as a point estimate, but in a preregistered fresh-data re-evaluation of the original 12 candidates, two of the three original KEEPs pass a laxer L1 threshold of \(0.55\), while W076 collapses. The paper argues that the combination of cross-family rejudgment, explicit selection-bias modeling, and fresh-data replication is a useful audit methodology for self-improving LLM loops, though the evidence is from one loop, three KEEPs, and small \(n\).

# Strengths

- **Clear scoped framing of the empirical claim** — The abstract and “Honest scope” paragraph explicitly state that this is “audit methodology, not about whether self-improving LLM loops work” and that the empirical base is “twelve candidates from one loop cycle on one solver family.” This helps prevent the paper from overselling a single-loop case study as a field-level result.

- **Strong use of cached-answer rejudgment as a clean audit intervention** — In Experiment 1 / L1, the paper rejudges the *same* base/ext answer strings with a different family, holding content fixed. This is a valuable methodological point because it isolates at least one major nuisance variable—judge family—without requiring re-solving.

- **Honest selection-bias analysis that weakens the authors’ earlier interpretation** — The Exp 45 “winner’s curse / selection-adjusted analysis” explicitly concludes that the L1 drops are statistically consistent with regression to the mean on top-3-of-12 noisy \(n=50\) selections. This is a commendable correction of a tempting but unsupported “judge fragility alone” interpretation.

- **Preregistered fresh-data replication is a meaningful improvement over purely post-hoc audits** — Exp 47 provides commit hashes, a timestamped preregistration, a fixed seed, and frozen thresholds before running the fresh split. Even though it is not a fresh full loop, this is substantially stronger than only auditing cached data.

- **Threshold sensitivity is reported rather than hidden** — Exp 47 explicitly shows that at the preregistered L1 threshold \(0.55\), W077/W078/WCAND03 pass, but at \(0.60\), only W078 clears both gates. This transparency materially improves the credibility of the empirical reporting.

- **The paper reports many negative and null results** — Examples include the \(0/9\) new Exp 33 candidates, the failed constructed controls in Exp 44/46, W078’s GSM8K harm direction in Exp 32, and the prospective trigger-conditioned gate failure on new candidates. This is valuable in a literature where self-improvement claims are often selectively positive.

- **Artifact and provenance emphasis is unusually strong** — The main text repeatedly points to logs, cached answer pairs, registry states, code paths, preregistration files, and proxy/model details, with appendices covering prompts, cost, seeds, and reproducibility. This strengthens the paper as a reusable audit testbed even if the scientific claims remain limited.

# Weaknesses

- **The central “diagnostic separation” claim is not demonstrated** — The abstract and conclusion claim that the audit stack plus Exp 45/47 produces a pattern “consistent with separating” selection-driven cached drops from genuine non-replication, with W076 diagnosed as real non-replication and W077/W078 as regression-to-mean cached drops. For this class of methodology paper, a diagnostic procedure needs either a prespecified classifier, known ground truth, or validation across multiple loops/candidates; here the separation is inferred post hoc from only three KEEPs. A convincing fix would preregister an explicit decision rule and evaluate it on multiple independently generated loops or on synthetic/known-effect candidates where the true status is known.

- **Exp 47 is not a fresh full-loop replication, despite being used as the strongest evidence** — Sec. Exp 47 correctly clarifies that candidate generation, success/failure distillation, cross-LLM distillation, and pruning are not rerun; the original 12 candidates are held fixed. This matters because selection effects in candidate generation and pruning are part of the self-improving-loop claim. The fix is a genuinely preregistered fresh loop: rerun candidate generation and pruning on fresh data, then apply the frozen audit stack to the newly selected KEEPs.

- **The main positive fresh-data result is threshold-sensitive and statistically weak** — Exp 47 shows W077’s L1 is \(0.57\), which passes \(0.55\) but fails the original \(0.60\) gate; the joint posterior table gives W077 only \(0.370\) probability for the exact preregistered rule \(\theta_{\text{inner}}>0.60 \land \theta_{\text{L1}}>0.55\). The conclusion still says “two of the three original KEEPs clear both gates” and “recover,” which reads stronger than the posterior evidence supports. The fix is to make posterior uncertainty the primary result, avoid binary “replicate” labels at \(n=30\), and rerun the fresh split at \(n\ge 100\).

- **The audit stack is largely post-hoc and layer-selected** — The “note on staged stress testing” states that the full stack was not preregistered and that layers were added in response to reviewer-simulation objections. This is not fatal for a case study, but it weakens any claim that the stack itself provides an unbiased audit procedure rather than a sequence of exploratory probes. The fix is to freeze L1–L6, thresholds, and stopping rules before running a new loop.

- **No external ground truth exists for the open-ended Chinese tasks** — The paper relies on LLM judges, LLM structured ratings, and LLM faithfulness judgments; Exp 38/41/48 use gpt-5.5 as a “human-annotation substitute,” but this is still another LLM. For a methodology paper about auditing LLM self-improvement, lack of human or objective validation makes it impossible to know which judge family is closer to actual quality. A fix would include blinded human expert ratings on a representative subset, or a larger objective benchmark where correctness is externally defined.

- **The statistical modeling is not strong enough for the claims made from it** — Exp 45’s empirical-Bayes Beta prior is fit to only 12 observed wrs, which are themselves prefiltered candidates, and assumes exchangeability; Exp 44 later says this exchangeability is “approximately wrong.” Exp 49’s DerSimonian–Laird pooling has only \(k=1\)–2 judge cells for fresh data and does not model within-pid correlation among judges. A fix would fit a hierarchical model on per-pid, per-judge verdicts with candidate-level and judge-level effects, report sensitivity to priors, and avoid treating posterior \(>0.5\) as a replication decision.

- **The paper sometimes double-counts dependent evidence streams** — The abstract and audit-stack description acknowledge that L1/L2 reuse cached pairs and L3/L4 overlap, but later language such as “0/5 independent stability checks” in Exp 5 and claims that orthogonal tests are “statistically independent” in Exp 9–10 are too strong. For an audit-methodology contribution, dependence among probes determines how much cumulative evidence the stack supplies. The fix is to present a dependency graph or explicitly separate “new answer generation,” “new judge,” “new problem,” and “new metric” as non-independent factors rather than independent evidence streams.

- **Positive controls do not establish audit sensitivity** — Exp 44 and Exp 46 report that all six constructed controls fail, including supposedly useful controls and a base-library duplicate. The later claim that Exp 47 establishes sensitivity by accepting WCAND03 is weaker because WCAND03 is not a known-positive control; it is another candidate selected from the same exploratory pipeline. A stronger fix is a synthetic benchmark or ablation where a known-useful wisdom is removed from the base library and then reintroduced, or a human-validated positive intervention with expected benefit.

- **The L6 faithfulness layer is undervalidated** — Experiments 9–14 use embedding deltas, LLM citation judgments, and prospective citation fields as proxies for “faithfulness.” These are plausible diagnostics, but embedding cosine between \(\Delta\texttt{what\_changed}\) and a wisdom description is a weak semantic measure, and forced citations in Exp 17 show citation can be increased by prompt format without proving causal utility. The fix is to validate L6 on known inserted effects and placebo wisdoms, ideally with human faithfulness labels.

- **The base scaffold and its headline improvements are not audited to the paper’s own standard** — Sec. “Contribution 3” admits v20’s \(0.64\) vs v16 and \(0.88\) vs baseline are same-family judged and not re-audited. Since the self-improvement loop operates on top of this scaffold, uncertainty about the base scaffold’s true gain complicates interpretation of the loop’s candidate utility. A fix would cross-family or human-audit the scaffold comparisons, or clearly demote them to background motivation only.

- **Evaluation-pool exposure through dynamic same-domain exemplars is a real contamination concern** — The Limitations section says the solver dynamically selects a same-domain exemplar from the evaluation pool’s other pids at solve time, then argues this is symmetric between base and ext. Symmetry helps for relative wr, but it still means the evaluated system has access to distribution-specific examples from the evaluation pool, which is problematic for claims about held-out generalization and fresh-domain behavior. A clean rerun with strict pool separation would remove this concern.

- **The paper is structurally sprawling and difficult to evaluate** — Experiments are numbered up to 49, appear nonchronologically in the main text, and include many intermediate results that are later overturned. There are also distracting phrases such as “Preregistered preregistered fresh-data replication” in the abstract and many strong intermediate claims later softened. For a top-venue methodology paper, the main body should be reorganized around a small set of prespecified questions, with exploratory experiments moved to appendix.

- **Claims about adjacent literature are too broad for the evidence** — Table 1 and the discussion suggest that adjacent self-improvement work lacks cross-family rejudgment and that same-family \(+N\)pp should be treated as preliminary. This is a reasonable recommendation for LLM-judged retrieval-library loops, but the text repeatedly gestures toward weight-level self-improvement and broader self-improving LMs despite acknowledging different acceptance signals. The fix is to restrict the normative claim to LLM-judged retrieval/corpus-level loops unless a broader survey is provided.

- **Model and proxy reproducibility remain fragile** — The Limitations section notes ruoli.dev proxy routing, hosted-model temporal drift, and lack of exact-token reproducibility. The appendices apparently specify model identities, seeds, and proxy details, which helps, but many core claims depend on proprietary, drifting judge models. The fix is to release all cached outputs/verdicts as primary artifacts and, where possible, replicate key audits with stable open-weight judges or official vendor endpoints.

# Questions to the authors

1. What exact preregistered decision rule maps the combination of cached audit results, Exp 45 selection-bias analysis, and Exp 47 fresh results into the labels “selection-driven cached-data drop” vs “genuine non-replication”? Would the same rule classify W077 as replicated if the threshold were \(0.60\) or if the fresh split were expanded to \(n=100\)?

2. Can you run a fully preregistered fresh *full loop*—including candidate generation, pruning, inner gate, L1, and the final decision rule—rather than only re-evaluating the original 12 candidates?

3. What happens if L1 in Exp 47 is run on all 12 candidates, not only the inner-gate passers? This would help quantify false negatives of the inner gate under the preregistered fresh split.

4. Do human raters agree more with gemini, haiku, gpt-mini, opus, gpt-5.4, or gpt-5.5 on a blinded subset of the Chinese open-ended answer pairs? This would materially affect the interpretation of “judge fragility.”

5. Can you construct a true positive control with known benefit, e.g. remove an active base-library wisdom and reinsert it as a candidate, or use a synthetic task family where a specific wisdom deterministically helps?

6. How robust are the Exp 49 posterior labels under a hierarchical model fit to the full per-pid verdict array, including judge-family effects and pid-level correlation, rather than DerSimonian–Laird pooling over aggregate proportions?

7. Why is \(0.55\) the correct L1 replication threshold in Exp 47 in utility terms, rather than merely a laxer statistical bound? What practical library decision would differ between \(0.55\) and \(0.60\)?

# Rating

Weak Reject

The paper is unusually transparent and contains several useful audit ideas, especially cached cross-family rejudgment, explicit selection-bias modeling, and the preregistered fresh-data re-evaluation. However, the main methodological claim—that the stack can distinguish selection-driven cached drops from true non-replication—is not demonstrated beyond a post-hoc interpretation of three KEEPs with small \(n\), no human/objective ground truth, and threshold-sensitive fresh results. The audit stack itself was mostly developed post hoc, and the strongest replication is not a fresh full-loop replication. I see this as a valuable artifact and case study, but not yet a sufficiently rigorous or clean main-track contribution.

# Confidence

4 — I am familiar with LLM-as-judge evaluation, self-improvement / self-refinement loops, and reproducibility concerns around preregistration, selection bias, and post-hoc audit design, though I have not worked specifically with this exact “wisdom library” formulation.