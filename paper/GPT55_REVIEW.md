# Summary

The paper presents a case study of a retrieval/prompt-level “self-improving” LLM loop that proposes Chinese-language methodological “wisdoms,” accepts candidates via a same-family LLM-as-judge A/B gate at \(n=50\), and then audits the accepted candidates. The main empirical claim is negative: the loop accepts \(3/12\) candidates, but these do not survive cross-family re-judgment, larger fresh-data replications, or selection-bias analysis. The paper then uses this failure to motivate a broader six-stage architecture for self-hypothesizing/self-validating agents and reports small diagnostic probes of those stages.

# Strengths

- **Clear and valuable negative result, scoped more honestly than many self-improvement papers.**  
  In the abstract, Introduction “What we found,” and Experiments §§3.1–3.5, the authors explicitly state that the loop does *not* demonstrate self-improvement and that the final library delta is \(+0\). This strengthens the submission because negative results about self-improvement loops are important and underreported.

- **Auditing same-family LLM-as-judge gates is a timely and concrete target.**  
  The paper focuses on a real methodological vulnerability: generator/judge family coupling in LLM preference evaluation. The six audit layers listed in the Introduction and §2.5 target plausible failure modes—judge-family preference, side bias, sample size, solver interaction, domain transfer, and faithfulness—making the case study practically relevant.

- **The cached-audit vs fresh-data distinction is handled relatively carefully.**  
  In §3, the authors explicitly acknowledge that the six audit layers were post-hoc and assign more inferential weight to the preregistered \(n=100\) and \(n=200\) fresh-data replications. This is an important strength: the paper does not simply present a post-hoc audit as if it were confirmatory.

- **Selection bias / winner’s curse is directly modeled rather than merely asserted.**  
  §3.3 fits an empirical-Bayes model to the 12 candidate win rates and simulates the “measure 12, accept top 3, re-measure” process. Even if the modeling choices need scrutiny, explicitly asking whether the observed drops are expected under top-\(k\) selection is exactly the right diagnostic for this class of loop.

- **The authors disclose many unresolved problems instead of hiding them.**  
  The Discussion “Objections we have not closed” section is unusually candid: it lists gate-design hindsight, post-hoc audit-stack construction, solver-family scope, missing human ground truth, lack of positive controls, and under-specified L6 faithfulness. This improves trust in the empirical narrative.

- **Artifact and provenance emphasis is strong.**  
  §§2.2, 2.3, Conclusion “Reproducibility,” and the stated appendices describe versioned library state, candidate evidence trails, logs, prompts, seeds, proxy details, and preregistration documents. For a methodology/case-study paper, this level of provenance is valuable.

- **The paper avoids claiming that cross-family re-judgment identifies causal content utility.**  
  §2.5 explicitly says the decomposition into content/generic-context/style components is not identifiable and that cross-family re-judgment is only a falsifier of judge-family robustness. This is a good correction to a common overclaim in LLM-evaluation work.

# Weaknesses

- **The central empirical scope is too narrow for the breadth of the claims.**  
  The main null result is one loop, one primary solver family, one Chinese open-ended benchmark distribution, one evolution cycle, and 12 candidates (§§3.1–3.5; Limitations). This is useful as a case study, but the title, abstract, Introduction, and Conclusion repeatedly generalize to “same-family A/B gates” and “retrieval-level self-improvement claims” broadly. For a top-venue methodology paper, one audited loop can motivate hypotheses, but it cannot establish a general methodological prescription as strongly as the paper presents it.  
  **Fix:** Either substantially narrow the claims throughout, or add at least one fully independent loop replication: different solver family, different language/domain, fresh candidate generation, same preregistered audit stack.

- **The six-stage “self-hypothesizing agent” roadmap is weakly connected to the empirical contribution.**  
  The empirical study tests a retrieval-library acceptance gate, but §4 expands to a broad architecture involving world models, schedulers, experience feedback, category-theoretic alignment, and new prior generation. The bridge from “\(n=50\) same-family A/B gate fails” to “these six architectural stages are necessary” is mostly argumentative, not demonstrated. This matters because the paper presents itself as both a case study and a roadmap; the roadmap currently reads more like speculative positioning than a contribution established by the experiments.  
  **Fix:** Reframe the roadmap as discussion/future work, or provide stronger evidence that each missing stage would specifically remedy the observed failure modes.

- **The roadmap validation experiments are too small and synthetic to support the stated verdicts.**  
  §4.1 uses 60 synthetic problems across 4 task families, with “known-optimal prior” labels and simple probes. Yet Table 1 labels stages as “works,” “fails,” or “conditional,” and the following paragraph says “Three stages work.” Cross-language pick consistency is treated as an “alignment” probe, but it does not test formal cross-domain equivalence detection, category-theoretic alignment, or transfer of structurally equivalent priors. For a main-track ML paper, these are weak diagnostics relative to the architectural claims.  
  **Fix:** Downgrade the language to “toy probes,” remove “works” labels, and avoid claiming validation of Stage 3 alignment unless an actual equivalence-detection/transfer experiment is run.

- **Internal inconsistency in stage counting and terminology.**  
  The abstract says “six-stage roadmap,” the Introduction says “six moving parts” but “stages 0 through 4” with stage 0.5, while §4 says “five sequential stages” and then includes Stage 0.5. Later §4.1 says “three of the six stages work,” while Table 1 says Stage 3 is only an operational probe. These inconsistencies make the roadmap appear under-edited and weaken the methodological framing.  
  **Fix:** Use one consistent numbering scheme and one consistent claim: e.g., “six components: 0, 0.5, 1, 2, 3, 4,” with Stage 3 clearly labeled as “not validated.”

- **The paper’s own positive-control results undermine the audit/gate interpretation more than the main text admits.**  
  In Discussion “Positive controls remain unresolved,” all \(6/6\) positive/placebo/useful controls fail the gate with wr \(\leq 0.25\), and the authors say the parsimonious null is that the gate is “structurally anti-additive.” If the gate rejects even constructed positive controls, then the experiment may primarily show that this scaffold/gate setup is pathological, not that same-family A/B gates are generally unreliable. This is a major issue for a methodology paper because calibration of the measurement instrument is essential.  
  **Fix:** Include a genuine positive-control environment where a known-useful inserted wisdom reliably improves objective outcomes and verify that the gate and audit stack detect it.

- **The selection-bias model is not sufficiently justified and may be misspecified.**  
  §3.3 fits an empirical-Bayes Beta prior and obtains concentration \(\hat\alpha+\hat\beta \approx 6\times 10^7\), effectively assuming almost no between-candidate heterogeneity. This boundary-like result is then used to argue the accepted candidates are upward fluctuations. But the model treats candidate cells as independent Beta-Binomial measurements, despite shared problems, shared judges, common solver artifacts, and possibly correlated candidate effects. It also describes the process as “top-3-of-12,” whereas the gate rule is threshold-based \(\mathrm{wr}\ge 0.60\), not literally top-3 selection.  
  **Fix:** Provide sensitivity analyses with hierarchical logistic models over candidates/problems/judges, correlated candidate effects, and threshold-selection rather than top-\(k\) selection; report whether the regression-to-mean conclusion is robust.

- **Statistical language around the \(+10\)pp gate is misleading.**  
  §2.4 says the \(+10\)pp threshold is “intentionally strict” and refuses candidates “undistinguishable from sampling noise.” But at \(n=50\), observing 30/50 wins gives a wide interval; under a binomial null around 0.5, this is not strong evidence. The paper later recognizes this, but the gate description and some contribution text still treat \(0.60\) as a meaningful success threshold.  
  **Fix:** State explicitly near the gate definition that \(0.60\) at \(n=50\) is a heuristic operating threshold, not a statistically stringent acceptance criterion; include exact binomial/Wilson intervals and false-positive rates under multiple testing over 12 candidates.

- **Same-domain exemplar exposure is not guaranteed to “cancel” by symmetry.**  
  In Limitations, the authors argue that using a same-domain example from the held-out evaluation pool applies symmetrically to base and ext, so it cancels in the relative comparison. This is not generally valid: the new wisdom can interact with the exemplar, retrieval, prompt framing, or solver behavior, and indeed §3.5 reports that exemplar ablation shifts win rates by 0.06–0.18. For an audit paper, evaluation-pool exposure is a serious design flaw even if later ablated.  
  **Fix:** Make the no-exemplar fresh replication the primary confirmatory result and avoid defending the original exposure as cancelled; future runs should use strict split separation from the start.

- **The distinction between “judge fragility” and “true quality difference” remains unresolved without human or objective ground truth.**  
  Cross-family re-judgment shows lack of robustness to judge family, but it does not tell whether Gemini was over-rewarding style, Claude was under-rewarding useful content, or both were noisy. The paper acknowledges no human ground truth in Limitations but still leans on cross-family drops as strong evidence against the gate’s accepted wisdoms. For open-ended Chinese reasoning tasks, judge disagreement is not enough to establish lack of substantive utility.  
  **Fix:** Add human evaluations on a stratified subset, or use tasks with objective outcomes where possible; report judge-human correlations for the audit panel.

- **The hierarchical Bayesian analysis is presented as a contribution but then shown to be overconfident and wrong.**  
  §3.4 reports a multilevel logistic model giving \(P(\text{pass})=0.763\) for W078, then §3.5 says the \(n=100\) data falsifies this. The authors candidly diagnose the missing exemplar covariate and flat candidate prior, but this means the hierarchical analysis is not supporting the final conclusions except as a cautionary tale. Its role in the paper is unclear.  
  **Fix:** Either remove it from the claimed contributions or refit a final preregistered hierarchical model including exemplar/protocol covariates and candidate-pool shrinkage.

- **The paper sometimes overstates “replication” because protocols differ.**  
  The fresh evaluations differ in sample size, exemplar mechanism, seed, possibly protocol details, and thresholds (§3.5). The \(n=100\) no-exemplar run is not a direct replication of the original v20-style gate; the \(n=100\) with-exemplar and \(n=200\) with-exemplar are closer. Calling the whole sequence “replicated three times” risks obscuring these design changes.  
  **Fix:** Separate “direct replication under original protocol” from “sensitivity replication under modified protocol,” and reserve “replication” for protocol-matched runs.

- **Some numerical phrasing is confusing or incorrect.**  
  The abstract and Introduction describe “exemplar boost \(\sim 0.10\)--\(0.20\) absolute win-rate” and elsewhere “\(+0.10\)--\(0.20\) pp”; \(0.10\) absolute win rate is 10 percentage points, not 0.10 percentage points. Contribution 3 says \(0.88\) vs baseline is “+76pp above parity,” which is unconventional because parity is 0.50, so it is +38 percentage points over parity, not +76pp unless using margin over loss. Such phrasing creates ambiguity in reported effect sizes.  
  **Fix:** Standardize all effects as either “absolute win-rate units” or “percentage points,” and avoid “above parity” unless precisely defined.

- **The paper contains duplicated/garbled text in the theory section.**  
  In §2.5, the paragraph “What cross-family re-judgment can and cannot deliver” cuts off mid-sentence—“two families may share style preferences that”—and is immediately followed by another paragraph with overlapping content. This suggests the main body was not fully edited and makes the theoretical argument harder to follow.  
  **Fix:** Remove the duplicate fragment and consolidate the cross-family re-judgment caveats.

- **The related-work comparison risks being too checklist-based.**  
  Table 1 marks adjacent papers as lacking L1/L3/L4/L6 audit practices. This is useful, but the paper sometimes implies that absence of cross-family re-judgment is a general deficiency even for works whose acceptance signals, tasks, or evaluation regimes differ substantially. The table does include “n/a” for environment/weight-level cases, but the Discussion later broadens again to many self-improvement papers.  
  **Fix:** Restrict the comparison to papers whose primary acceptance signal is same-family LLM preference over open-ended outputs, and avoid extrapolating to weight-level or environment-reward systems without direct evidence.

- **Reproducibility remains limited by third-party proxy and model identities.**  
  The Limitations section notes that API access is via ruoli.dev and exact-token reproduction depends on proxy routing state. For an audit-methodology paper whose conclusions depend on judge-family behavior and model identity, this is a nontrivial reproducibility limitation.  
  **Fix:** Replicate key cells through official vendor endpoints or provide cached inputs/outputs sufficient for third parties to re-judge with current public models.

# Questions to the authors

1. **What happens under a fully preregistered, fresh candidate-generation loop?**  
   The current audit stack was developed post-hoc after the original loop. If you rerun candidate generation, gating, and auditing from scratch with the final audit protocol frozen, do you again obtain \(0\) validated additions?

2. **Can you provide a genuine positive-control task where an inserted wisdom has known objective utility and show that the gate plus audit detects it?**  
   The failed positive controls are currently one of the most serious threats to interpreting the results.

3. **How robust is the selection-bias conclusion to a hierarchical model with shared problem effects and threshold selection rather than top-3 selection?**  
   The \(\alpha+\beta\approx 6\times10^7\) empirical-Bayes fit seems boundary-like; sensitivity analyses would materially affect my confidence.

4. **Do human judges agree more with the original same-family judge or the cross-family audit panel on a stratified subset of Chinese open-ended outputs?**  
   This would help distinguish judge-family fragility from true quality differences.

5. **For the \(n=200\) with-exemplar run, what are the confidence intervals and exact counts for the top candidates, especially W078 and wcand03?**  
   Values such as 0.590 and 0.580 are close to the 0.60 threshold, so uncertainty and exact binomial counts matter.

6. **How many gate/audit/gate-redesign variants were tried before the final reported protocol, and which results were excluded from the main narrative?**  
   The paper acknowledges gate-design freedom; a complete design-search log would clarify the extent of hindsight.

7. **Does the no-exemplar \(n=100\) run use exactly the same prompts and solver settings except for exemplar removal?**  
   Since exemplar interaction is central to the interpretation, any additional protocol changes would matter.

# Rating

**Weak Reject**

The paper addresses an important problem and provides a candid, artifact-backed negative case study, but the evidence is not yet strong enough for the breadth of the methodological and architectural claims. The most serious issues are the single-loop/single-primary-solver scope, the post-hoc audit-stack construction, the unresolved lack of positive controls and human/objective ground truth, and the weak connection between the empirical null and the large six-stage roadmap. I would be much more positive on a narrower paper framed strictly as an auditable case study with preregistered fresh-loop replication, calibrated positive controls, and reduced claims about general self-validating architectures.

# Confidence

**4** — I am familiar with LLM self-improvement loops, LLM-as-judge evaluation failure modes, cross-model auditing, and reproducibility/preregistration concerns, though I am not a specialist in Chinese-language wisdom-library benchmarks specifically.