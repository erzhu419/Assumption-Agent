# Summary

The paper presents a self-improving retrieval-library loop for LLM “wisdom” entries and, more centrally, an audit methodology for checking whether accepted entries survive re-judgment. The main empirical story is that the loop’s original three KEEP decisions pass a same-family \(n=50\) gate but fail many cached-data audits at the original \(0.60\) threshold, while a later preregistered fresh-data re-evaluation of the original 12 candidates recovers W077 and W078 at a laxer L1 threshold of \(0.55\). The authors argue that the combination of cross-family re-judgment, selection-bias modeling, and fresh-data replication can distinguish regression-to-the-mean cached-data drops from genuine non-replication, but the evidence comes from one loop, three original KEEPs, small \(n\), and heavily post-hoc audit development.

# Strengths

- **Honest negative-result framing and scope caveats throughout.** The abstract, Introduction, Exp. 45, Exp. 47, Discussion, and Limitations repeatedly state that this is a case study rather than a field-level claim, and that cached L1 drops alone cannot prove judge fragility. This strengthens the paper because many self-improvement-loop papers would have stopped at the original \(3/12\) KEEP result.

- **Operationally concrete audit layers.** Section “The six-layer audit stack” and Experiments 1, 5, 8, 40, 31/32/42, and 9–14 specify concrete interventions: judge-family swap, side reseed, sample extension, solver-family swap, fresh-domain / GSM8K probes, and non-pairwise faithfulness checks. Even if not all layers are novel, the paper gives enough procedural detail to make the audit stack implementable.

- **Selection-bias analysis is a valuable correction to the initial narrative.** Exp. 45 explicitly models the winner’s curse from selecting top candidates out of 12 noisy \(n=50\) measurements, and correctly concludes that L1 drops are consistent with regression to the mean. This is one of the strongest parts of the paper because it narrows an otherwise overconfident “judge fragility” interpretation.

- **Preregistered fresh-data re-evaluation is a meaningful addition.** Exp. 47 freezes thresholds and evaluates the original 12 candidates on 30 disjoint fresh problems, recovering W077/W078 and WCAND03 while W076 collapses. Although small and threshold-sensitive, this is substantially more informative than only re-judging cached answer pairs.

- **The paper reports uncertainty and tie handling rather than only point estimates.** Section “Tie handling and effective \(n\),” Table 1 / Table \(\ref{tab:ci}\), Exp. 35, and Exp. 49 report Wilson intervals, effective non-tie \(n\), and sensitivity to tie conventions. This improves interpretability and exposes that many threshold crossings are statistically fragile.

- **The authors preserve and report many failed fixes.** The trigger-conditioned gate in Exp. 15, constructed controls in Exp. 44/46, English replication in Exp. 39, GSM8K probes in Exp. 32/42, and architectural redesign attempts are mostly negative or mixed. Keeping these in the paper makes the case study more credible than a selectively positive narrative.

- **Artifact and provenance emphasis is strong.** The main text repeatedly references logs, code, registry states, commit hashes for Exp. 47, model identities, seeds, costs, and cached answer pairs. For a methodology/audit paper, this kind of provenance is important and should help independent re-analysis.

# Weaknesses

- **The central claim remains too strong for the evidence.**  
  The abstract and Conclusion claim the audit stack “suggests a separation” between selection-driven cached-data drops and genuine non-replication, with W077/W078 treated as recovered and W076 as genuinely non-replicating. But this classification is based on only three original KEEPs and a fresh \(n=30\) split, with W077’s fresh L1 score only \(0.57\) and wide intervals. For an audit-methodology paper, demonstrating classification ability requires known positives/negatives, multiple loops, or at least substantially larger fresh samples. A fix would be to either downgrade the claim to “illustrates how such a pattern could be diagnosed” or run the full preregistered audit on several independent loops with enough power to classify candidates.

- **Audit-stack development is largely post-hoc, and the preregistration only partially addresses this.**  
  The “note on staged stress testing” in \S\ref{sec:experiments} says the layers were added after reviewer-simulation objections, not preregistered before the inner loop. Exp. 47 is preregistered, but it occurs after dozens of audits, uses the original candidate set rather than a fresh full loop, and adopts a laxer L1 threshold of \(0.55\). For this class of paper, post-hoc layer selection creates a serious risk that the audit stack is tailored to observed failures. A fix would be a genuinely preregistered fresh loop: rerun candidate generation, pruning, and all audit layers with primary endpoints fixed before seeing any results.

- **Threshold dependence is severe and not resolved.**  
  Exp. 47 shows that at the preregistered L1 threshold \(0.55\), W077/W078/WCAND03 pass, but at the original \(0.60\) L1 threshold only W078 passes. Many headline labels—“REPLICATE,” “COLLAPSE,” “FLIP,” “+0,” “+3”—change under small threshold shifts. This is a weakness because the paper’s contribution is an audit methodology, and the methodology’s conclusions should not hinge on a 0.55 vs. 0.60 convention without a principled utility or error-rate justification. A fix would be to define loss functions or decision costs, report continuous posterior probabilities as primary, and treat binary threshold labels as secondary.

- **The paper repeatedly uses categorical language despite overlapping uncertainty.**  
  Table \(\ref{tab:ci}\) shows many CIs include both parity and the decision threshold; Exp. 47 explicitly notes Wilson CIs at \(n=30\) include both \(0.50\) and \(0.60\). Nevertheless, the text frequently says “FLIP,” “COLLAPSE,” “NULL,” “HARM,” and “REPLICATE.” For an audit paper, such language can mislead readers into thinking threshold crossings are statistically decisive. The fix is to replace categorical labels with calibrated uncertainty statements, e.g. posterior probability of exceeding each threshold and probability of sign relative to parity.

- **No human ground truth, and LLM-judge reliability is low.**  
  The paper’s primary evidence is still LLM-as-judge. Exp. 37 reports only fair agreement: within-cheap \(\kappa=0.18\), within-expensive \(\kappa=0.26\), cross-tier \(\kappa=0.25\). Exp. 38/41/48 use gpt-5.5 as a “human-substitute,” but that remains another LLM. For a methodology paper about auditing LLM self-improvement, low judge agreement undermines the ability to say which audit verdict is closer to truth. A fix would be a human evaluation study on a stratified subset, or objective tasks with known outcomes, used to calibrate the audit stack’s sensitivity/specificity.

- **The Bayesian/statistical analyses are not yet adequate for the claims.**  
  Exp. 49 calls the DerSimonian–Laird random-effects pooling a “proper hierarchical model,” but DL with \(k=1\)–2 fresh cells cannot estimate between-judge variance meaningfully and does not model within-pid dependence. Also, pooling W077’s fresh inner and L1 cells to get \(P(\theta>0.55)=0.808\) is not the same estimand as the preregistered decision “inner \(\geq0.60\) and L1 \(\geq0.55\).” Exp. 45’s empirical-Bayes prior is fit to only 12 prefiltered candidates and later contradicted by Exp. 44’s observation that the 12 are not exchangeable with random controls. A fix would be a preregistered hierarchical model over candidate, judge, problem, and run effects, with posterior probabilities for the exact decision rule.

- **Positive controls and sensitivity remain unresolved.**  
  Exp. 44 and Exp. 46 construct six controls, including “useful” and “stronger positive” controls, and all fail. The paper then treats WCAND03 in Exp. 47 as an “empirical positive control,” but WCAND03 is not a known true positive; it is simply a candidate selected from the same original candidate-generation process that happened to pass on the fresh split. For an audit methodology, demonstrating that the audit can accept known-good interventions is essential. A fix would be to create synthetic or objective task families where an inserted intervention is known to help, or remove an existing validated wisdom from the base library and test whether restoring it is accepted.

- **Internal inconsistencies from accumulated revisions make the scientific story hard to trust.**  
  Several passages appear inconsistent with later results. In Contribution 4, the paper says each of W076/W077/W078 is “subsequently falsified by the audit stack,” while Exp. 47 and the Conclusion say W077/W078 replicate on fresh data. In \S\ref{sec:theory}, the text says “no audit layer at any threshold and no non-pair-wr measurement assigns lift to the three KEEPs,” but later L4 has \(7/18\) single-judge cells \(\geq0.60\), Exp. 14 finds W078 citable, and Exp. 47 recovers W077/W078. These inconsistencies matter because the paper’s claim is already nuanced; contradictory phrasing obscures the actual result. A fix would be a major rewrite with one canonical results table and removal of stale claims.

- **The methodology’s novelty is limited relative to the amount of machinery.**  
  The Discussion admits L2–L5 are standard statistical hygiene and that L1/L6 are the main methodological claims. Cross-family re-judgment and fresh held-out validation are useful, but not by themselves a top-venue methodological advance unless validated systematically. The paper currently presents a large bundle of reasonable checks rather than a new statistical method or a validated audit protocol. A fix would be to formalize the audit as a decision procedure with error guarantees, cost/benefit analysis, or demonstrated performance across multiple independent loops.

- **The main loop itself is not independently validated.**  
  The v20 scaffold gains in \S\ref{sec:ratchet} and \S\ref{sec:holdout} are same-family LLM-judged, and the paper explicitly says they are not re-audited. Since the audit stack is motivated by failures of same-family judging, the substrate’s strength remains uncertain. For a case study, this weakens the interpretation of the loop’s candidate generation and the claim that the setup is a strong instrument. A fix would be to cross-family or human-audit the scaffold-vs-baseline improvements, or clearly remove scaffold strength from the contribution.

- **Data leakage / held-out exposure is real and not fully neutralized by symmetry.**  
  The Limitations section states that the solver dynamically selects same-domain exemplars from the evaluation pool’s other pids, with cached v15-class answer text. The authors argue this is symmetric between base and ext, but interactions with the added wisdom could still change how leaked in-pool information is used. For a paper about auditing subtle \(+10\)pp effects, this is a serious confound. A fix is a clean rerun with strict split separation and no evaluation-pool exemplars.

- **The paper overgeneralizes from one highly specific setup.**  
  The strongest general statements—e.g. in Experiment 1 and Discussion, that same-family \(+N\)pp improvements should be read as judge-preference alignment “until further notice”—are based on one Chinese open-ended retrieval-wisdom loop with one original solver family. Exp. 39 is not a full English self-improvement loop and does not validate language generality. A fix would be to restrict all general recommendations to retrieval-level LLM-judged loops and present cross-family re-judgment as a hypothesis-generating audit, not a broad verdict on self-improvement literature.

- **The paper is far too diffuse and hard to review as a main-track submission.**  
  The main body includes dozens of experiments up to Exp. 49, non-monotonic numbering, multiple “closure” sections, appendix-like material in the main text, and repeated revisions of the headline. This makes it difficult to identify the primary endpoint, preregistered claims, and final evidence. A fix would be to restructure around: setup, original gate, selection-bias model, preregistered fresh evaluation, and a compact audit-stack ablation; move exploratory experiments to appendices.

- **Several baselines are arbitrary or acknowledged as invalid.**  
  The \(30\%\) random-inclusion reference in Exp. 15/33 and \S\ref{sec:stats} is explicitly “not a principled utility baseline,” and Exp. 27 retracts an earlier F1 baseline as circular. Without a principled baseline, it is hard to judge whether the audit stack improves decision quality versus simpler alternatives such as larger \(n\), cross-family majority vote, or human spot checks. A fix would be to compare predefined audit policies under a common cost budget and external validation target.

# Questions to the authors

1. What is the exact primary endpoint you want reviewers to judge: cached-data non-reproduction at \(0.60\), fresh-data replication at \(0.55\), or classification of candidates into selection-driven vs. genuine non-replication?

2. For Exp. 47, why was \(0.55\) chosen as the L1 replication threshold rather than preserving the original \(0.60\) threshold, and was this threshold considered before any knowledge of cached L1 failures or W077/W078 behavior?

3. Can you report posterior probabilities for the exact Exp. 47 decision rule—\(P(\theta_{\text{inner}}>0.60 \land \theta_{\text{L1}}>0.55)\)—rather than pooled \(P(\theta>0.55)\) across inner and L1 judges?

4. Do you have any human annotation, even on a small stratified subset, comparing base vs. ext answers for W076/W077/W078 and WCAND03? If not, why should gpt-5.5 be treated as an adequate reference for calibration?

5. Can you run a clean no-leakage rerun where evaluation prompts do not use same-domain exemplars from the evaluation pool, at least for the three original KEEPs and WCAND03?

6. What would happen if the entire candidate-generation loop—not just evaluation of the original 12 candidates—were rerun under the preregistered audit protocol? Is there any preliminary evidence from such a fresh full loop?

7. Can you construct a known-positive control by removing an existing high-value wisdom from the base library and testing whether the audit stack accepts its restoration?

8. Which experiments are confirmatory and which are exploratory? Can you provide a single table marking preregistered vs. post-hoc, fresh vs. cached, and primary vs. secondary analyses?

# Rating

Reject

The paper is interesting, unusually transparent, and contains several valuable audit ideas, but it does not yet meet the bar for a top-venue methodology contribution. The main issues are the post-hoc construction of the audit stack, the very small number of original KEEPs, the threshold-sensitive fresh replication, lack of human or known-ground-truth validation, and statistical analyses that are not strong enough to support the claimed diagnostic separation between regression-to-the-mean and genuine non-replication. The submission also contains internal inconsistencies and too many exploratory experiments in the main narrative, making the final scientific claim difficult to pin down. I would encourage resubmission after a preregistered fresh full-loop replication with larger \(n\), known-positive controls, and a cleaner statistical decision framework.

# Confidence

4. I am familiar with LLM self-improvement / self-refinement loops, LLM-as-judge evaluation and its reliability issues, and reproducibility / preregistration concerns, though I have not worked specifically on this exact “wisdom library” formulation.