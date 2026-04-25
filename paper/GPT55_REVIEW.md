# Summary

The paper presents a closed-loop LLM “wisdom library” system that proposes, gates, prunes, and audits retrieval-level methodological entries. Its main empirical pattern is mixed: the original three same-family-gated KEEPs fail strict cached-data audits at the original \(0.60\) threshold, but a later preregistered fresh-data re-evaluation of the same 12 candidates at a laxer L1 threshold recovers two of those three KEEPs plus one originally rejected candidate. The authors argue that cross-family re-judgment, selection-bias modeling, and preregistered fresh-data re-evaluation form a useful audit methodology for separating regression-to-the-mean from genuine non-replication, while carefully scoping the result as a single-loop case study.

# Strengths

- **Important and timely problem framing** — The Introduction and Related Work focus on a real weakness of LLM self-improvement papers: same-family LLM-as-judge acceptance signals. This is a valuable target for a methodology/case-study paper because many agentic/self-improving systems currently rely on weak or endogenous validation.

- **Operationally concrete audit stack** — The six audit layers are specified in enough procedural detail to be reproducible in principle: fixed cached answer pairs for L1/L2, \(n=50\to100\) extension for L3, solver-family changes for L4, fresh-domain/GSM8K tests for L5, and faithfulness proxies for L6. This concreteness makes the paper more useful than a purely rhetorical critique of LLM-as-judge evaluation.

- **Unusually candid reporting of negative and contradictory results** — The paper reports that the original cached-data headline collapses, that the trigger-conditioned gate does not generalize to the 9 new candidates, that positive controls fail, that several CIs overlap thresholds, and that Exp~27’s earlier F1 framing is retracted. This honesty strengthens the submission relative to many self-improvement papers that only report monotone gains.

- **Explicit selection-bias analysis** — Exp~45 is a valuable addition: it directly asks whether the L1 drops could be explained by top-3-of-12 regression to the mean. Even if the model is limited, including this analysis substantially improves the paper because it prevents the authors from over-interpreting cross-family drops as necessarily “judge fragility.”

- **Preregistered fresh-data re-evaluation with threshold and missing-data sensitivity** — Exp~47 includes preregistration commit metadata, a disjoint fresh split, threshold sensitivity, missing-data sensitivity, and posterior calculations for the joint decision rule. This is a strong practice and materially changes the interpretation of the cached-data null.

- **Reproducibility orientation** — The main text repeatedly identifies logs, scripts, cached answer pairs, model settings, and decision traces, and the appendices reportedly include prompt templates, model identities, temperatures, seeds, proxy details, and compute/cost breakdowns. For an audit-methodology paper, this artifact discipline is a major positive.

- **Good internal caveating in several places** — The theory section explicitly says the \(Z^{\mathrm{specific}}/Z^{\mathrm{generic}}/Z^{\mathrm{style}}\) decomposition is not identifiable; the audit-layer section says layers are “conditionally distinct” rather than independent; Exp~47 says it is not a fresh full loop. These caveats improve scientific clarity.

# Weaknesses

- **Single-loop, single-cycle case study is too small to support the methodological recommendation** — The abstract, Scope paragraph, and Conclusion acknowledge that the empirical base is 12 candidates from one loop cycle and only three original KEEPs. For a methodology paper advocating a default audit stack for self-improving LLM loops, one loop on one main solver family cannot establish operating characteristics, generality, or reliability of the proposed audit. A convincing fix would be a preregistered application to multiple independent loops/tasks/solver families, ideally including at least one fresh full loop where candidate generation, pruning, gating, and auditing are all rerun under a fixed protocol.

- **No external ground truth for the main open-ended task** — The Setup, Limitations, Exp~38/41/48, and Discussion make clear that the core Chinese open-ended evaluations are judged by LLMs, with gpt-5.5 used only as a stronger LLM substitute rather than human truth. This is a serious weakness for an audit paper about LLM-as-judge fragility: cross-family disagreement does not tell us which judge is right, and a stronger LLM reference is not an external criterion. The fix is human expert annotation on a substantial subset, objective task outcomes where possible, or synthetic known-effect tasks where the “correct” audit decision is known.

- **The audit stack is largely post-hoc and has substantial researcher degrees of freedom** — The Experiments preamble explicitly says the full stack was not preregistered and grew through staged reviewer simulation; many later experiments are framed as rebuttals or closures. This is acceptable for exploratory work, but it weakens claims that the stack itself diagnoses failure modes rather than being adapted after seeing surprising results. The fix is a fully preregistered audit protocol, including thresholds, missing-data handling, layer order, and reporting rules, applied to a genuinely fresh full loop.

- **The central “replication” headline is threshold- and missing-data-sensitive** — Exp~47 shows that at the preregistered \(0.55\) L1 threshold, W077/W078/WCAND03 pass as point estimates, but at \(0.60\) only W078 passes L1; the missing-data sensitivity says W078’s inner wr drops from \(0.63\) to \(0.57\) and fails the inner gate if invalid outputs are counted as base wins. This matters because the paper’s differentiated conclusion depends heavily on Exp~47. A fix would be to make the posterior joint decision rule, not point-estimate threshold crossing, the primary headline and rerun fresh evaluation at \(n\ge100\) with preregistered missing-data treatment.

- **The paper uses conflicting inferential targets for “replication”** — Exp~47 correctly says the primary target is the joint event \(\theta_{\text{inner}}>0.60 \land \theta_{\text{L1}}>0.55\), with joint posteriors W078 \(0.539\), W077 \(0.370\), WCAND03 \(0.424\). Exp~49 then labels W077 as “replicates” using a DerSimonian–Laird pooled \(P(\theta>0.55)=0.808\), even while acknowledging DL is secondary and not Bayesian hierarchical. For this class of paper, the decision rule must be coherent; otherwise the audit becomes another source of post-hoc threshold rhetoric. The fix is to choose one preregistered inferential target and report all labels according to it.

- **Several statistical analyses ignore dependence and selection structure** — The paper often pools across judges, pids, and candidates despite repeated use of the same pids and answer pairs; Exp~49’s DL analysis has \(k=1\)–2 for fresh cells and cannot estimate between-judge variance; Exp~45 models top-3-of-12 selection but not the full candidate-generation/filtering process. These assumptions are especially consequential because most effects are near thresholds. The fix is a hierarchical model with candidate, pid, judge-family, and solver-family effects, clustered uncertainty by pid, and explicit selective-inference or pipeline-level simulation.

- **The selection-bias argument is useful but incomplete and partly contradicted by the controls** — Exp~45 fits an empirical-Bayes prior to the 12 observed candidate wrs, but Exp~44/46 explicitly show the 12 candidates are not exchangeable with random text and that the gate may be structurally anti-additive. Thus the “top-3-of-12 noisy measurements” model is not the full null model of the loop. To support the paper’s key “selection-driven vs true non-replication” distinction, the authors need a simulation or analysis of the entire proposal-generation/deduplication/gating pipeline, not only the final 12 wr values.

- **Positive-control sensitivity remains unresolved** — Exp~44 and Exp~46 construct six controls, including generic-useful, math-specific, SCQA, and a base-library duplicate; all fail badly. The paper later says Exp~47’s WCAND03 demonstrates nonzero sensitivity, but WCAND03 is not a known positive; it is merely a candidate that passed the same LLM-based fresh gate. For an audit methodology, sensitivity to true improvements is as important as specificity against false positives. A fix would be a synthetic or objective benchmark where an inserted intervention is known to improve outcomes, or a remove-and-reinsert test of an independently validated base wisdom.

- **LLM/proxy drift confounds several claimed nuisance interventions** — Exp~37, Exp~39, Exp~42, and the Limitations section document substantial temporal drift even at nominally fixed model identifiers and temperature settings. This directly weakens L2 side-reseed claims and any interpretation of small differences around \(0.55\)–\(0.60\). The fix is contemporaneous paired reruns, official or open-weight frozen model snapshots, repeated judge calls to estimate drift, and treating model-time as a random effect.

- **Some language around L1 remains too strong relative to Exp~45** — Exp~1 and the Discussion contain strong claims such as same-family \(+N\)pp being “until further notice” judge-preference alignment and cross-family rejudge being “cheap and decisive.” But Exp~45 later says pure regression-to-the-mean is statistically sufficient to explain the L1 drops. The fix is to consistently frame L1 as a robustness check on the original verdict, not as evidence of judge bias or lack of methodological utility without fresh-data/objective corroboration.

- **Faithfulness layer L6 is under-validated** — Exp~9–14 and Exp~17 use embedding-direction alignment, LLM citation judgments, and explicit solver citations. These are plausible probes, but they are not validated against causal faithfulness; explicit citation can reflect prompt compliance rather than actual causal use. For an audit methodology, L6 should be calibrated on synthetic cases with known inserted causal effects, placebo wisdoms, and human faithfulness labels.

- **The scaffold/baseline claims are not audited under the paper’s own standards** — Sections on v13\(\to\)v20 and held-out generalization report large scaffold gains using same-family LLM judging, while Contribution 3 admits these are not re-audited. The Limitations section also notes symmetric same-domain exemplar exposure from the evaluation pool. Since the scaffold is the substrate that generates candidates and signals, unaudited same-family scaffold gains weaken the empirical foundation. The fix is to apply at least L1/human/objective checks to the scaffold comparisons or remove these results from the evidential chain.

- **The six-layer stack’s incremental value over simpler protocols is not established** — The Discussion itself says L2–L5 are mostly standard statistical hygiene and L1/L6 carry the main methodological claim. The paper does not show that all six layers are necessary, cost-effective, or predictive of fresh/human/objective outcomes beyond, say, cross-family rejudgment plus fresh-data replication. A fix would be an ablation or decision-analysis showing which layers add predictive information and under what sample sizes.

- **Experimental coverage is unbalanced across candidates and layers** — Some probes test only W076, some only W078, some all three KEEPs, some all 12, and the English replication uses a different 6-entry library. The paper often acknowledges this, but the narrative still aggregates them into a broad “audit stack pattern.” A stronger design would use a balanced factorial layout over candidates \(\times\) judges \(\times\) solvers \(\times\) datasets, at common \(n\).

- **Presentation is overly long and chronologically difficult to audit** — The main body contains dozens of experiments, intermediate verdicts later overturned, retractions, reviewer-response-style “closures,” and multiple decision rules. For a top-venue methodology paper, this makes it hard to identify the confirmatory evidence versus exploratory debugging. The fix is to move exploratory chronology to appendices and make the main paper center on one clean preregistered protocol, one primary estimand, and a small number of decisive analyses.

# Questions to the authors

1. Can you run a fully preregistered fresh **full loop**—including candidate generation, pruning, gating, and audit—rather than only re-evaluating the original 12 candidates? This would directly address the largest external-validity concern.

2. Do you have any human expert annotations on the core Chinese open-ended answer pairs, even for a small subset, to calibrate whether gemini, haiku, opus, GPT-family, or gpt-5.5 judgments track human preferences?

3. For Exp~47, what are the exact conclusions under the conjunction of the two conservative choices: invalid outputs counted as candidate losses **and** L1 threshold \(0.60\)? This appears to substantially weaken the “2/3 KEEPs replicate” headline.

4. Why should WCAND03 be treated as evidence of audit-stack sensitivity rather than just another LLM-gated candidate? Can you construct or report a true positive control with independently known utility?

5. Can you provide a single hierarchical model over candidate, pid, judge, solver, and dataset effects that reports posterior probabilities for the exact preregistered joint decision rule? This would likely change the interpretation of W077 and WCAND03.

6. How many gate designs, thresholds, candidate filters, and audit variants were tried but not emphasized in the final narrative? A table of all attempted protocols would clarify the degree of post-hoc selection.

7. What is the empirical or decision-theoretic justification for using \(0.55\) as the preregistered L1 replication threshold while keeping the inner gate at \(0.60\)? How sensitive are conclusions to alternative thresholds chosen before Exp~47?

8. Can the key cached and fresh results be reproduced using official vendor endpoints or frozen open-weight judges, rather than the ruoli.dev proxy, given the documented temporal drift?

# Rating

**Weak Reject**

The submission is interesting, unusually transparent, and contains useful artifacts and negative results, but it is not yet a strong top-venue methodology contribution. The main barriers are the single-loop/small-\(n\) empirical base, lack of external ground truth, post-hoc construction of much of the audit stack, and threshold-sensitive Exp~47 conclusions. The paper’s own analyses show that the most important positive “replication” labels are probabilistic and weak for W077/WCAND03, while the cached-data null can be explained by selection effects. I would be much more positive after a preregistered fresh full-loop replication with human/objective calibration and a coherent primary statistical model.

# Confidence

**4** — I am familiar with LLM-as-judge evaluation, agentic/self-improvement loops, selection bias in adaptive evaluation, and reproducibility/preregistration issues; my confidence is slightly below 5 because the paper relies on specific proprietary/proxy model identities and extensive unpublished appendices/logs.