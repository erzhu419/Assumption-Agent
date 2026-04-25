# Summary

The paper presents a case study of a closed-loop LLM “wisdom library” agent that proposes Chinese-language methodological priors, accepts them via a same-family LLM-judged A/B gate at \(n=50\), and then audits the accepted candidates using cross-family re-judgment, sample extension, solver/domain perturbations, and faithfulness probes. The central empirical claim is a calibrated negative result: the loop’s original \(3/12\) accepted candidates do not survive more robust evaluation, with three preregistered fresh-data replications at \(n=100,100,200\) all yielding \(0/12\) candidates passing the joint gate. The paper argues that this failure exposes why same-family LLM-as-judge gates are insufficient for retrieval/prompt-level self-improvement, and motivates a six-stage architecture for self-hypothesizing/self-validating agents, with preliminary diagnostics for each stage.

# Strengths

- **Honest negative result with clear scope** (§Abstract; §Introduction “What this paper is/found”; §Experiments).  
  The paper explicitly states that “self-improvement works” is *not* the contribution, and instead frames the result as a failure of a minimal loop. This is valuable for a methodology/case-study paper because the field has many positive self-improvement claims with limited auditing.

- **Strong emphasis on preregistered fresh-data replications** (§3.5 “Three preregistered fresh-data replications”).  
  The authors correctly recognize that the post-hoc cached audit is not enough and give more inferential weight to fresh \(n=100\), \(n=100\), and \(n=200\) evaluations. This substantially strengthens the null claim relative to a purely retrospective audit.

- **Explicit treatment of selection bias / winner’s curse** (§3.3 “Selection bias”).  
  Modeling the “top-3-of-12” selection process is exactly the right concern for this setup. The paper avoids the common mistake of interpreting post-selection drops as direct evidence of judge failure, and instead notes that the observed drops are compatible with regression to the mean.

- **Candid discussion of post-hoc audit design** (§Experiments caveat; §Discussion “Objections we have not closed”).  
  The paper admits that the six audit layers were not preregistered and were added in response to objections. This transparency is unusually good and helps the reader separate exploratory diagnostics from confirmatory evidence.

- **Operational audit stack is concrete and reusable** (§Introduction L1–L6 list; §Theory; §Discussion “What is actually novel”).  
  Cross-family re-judgment, side reseeding, sample extension, cross-solver replication, fresh-domain testing, and non-pairwise faithfulness probes are specified at a level that future authors could implement. For a methodology paper, this practical specificity is a real asset.

- **The paper distinguishes what cross-family re-judgment can and cannot prove** (§2.5 “What cross-family re-judgment can and cannot deliver”).  
  The authors correctly state that judge swapping is a falsifier of robustness, not an identified causal decomposition of “style” versus “content.” This prevents overclaiming from one of the paper’s core diagnostics.

- **Release/reproducibility posture appears strong** (§Contribution 7; §Conclusion “Reproducibility”; stated appendices).  
  The submission claims release of code, logs, prompt templates, model identities, temperatures, seeds, proxy details, and preregistration documents. For this kind of audit paper, artifact completeness is central, and the described release would materially improve reproducibility.

- **The roadmap diagnostics include failures, not only successes** (§4.1 Table “Per-stage operational verdict”).  
  The authors report that the naive cheap world model has AUROC \(0.40\) and that naive new-prior generation beats baseline on \(0/5\) holdouts. Reporting these failures makes the roadmap more credible than a purely aspirational architecture proposal.

# Weaknesses

- **The core evidence is still a single-loop, single-substrate case study, but some claims are stated too broadly** (§Abstract final paragraph; §Conclusion recommendations).  
  The paper’s strongest empirical result is about one Chinese wisdom-library loop, one main solver family, one candidate-generation cycle, and a particular same-family gate. Yet the conclusion recommends that “any future paper proposing retrieval-level self-improvement” report a fairly specific five-item audit package, and the abstract says same-family \(n=50\) A/B is “an unreliable substitute” for audit-based acceptance. For a top-venue methodology paper, the general recommendation needs either multiple independent loops/configurations or a tighter statement that the evidence motivates, rather than establishes, the audit norm.  
  **Fix:** Add at least one additional full closed-loop replication on a different solver family or English task substrate, or substantially narrow the normative language throughout to “in this loop / this class of loop, this audit revealed failure.”

- **The paper’s main target shifts between an empirical audit case study and an architectural roadmap, weakening the contribution focus** (§Abstract; §Roadmap; §4.1).  
  The first half is a careful audit of one failed self-improvement loop; the second half proposes a broad six-stage architecture involving world models, schedulers, category-theoretic alignment, and new-prior generation. The empirical support for the roadmap is a small synthetic 60-problem diagnostic environment, which is far less mature than the audit study. For a methodology paper, this risks diluting the strong negative result with a speculative architecture that is not comparably validated.  
  **Fix:** Make the audit stack/null result the primary paper and move the roadmap to a shorter discussion/future-work section, or expand the stage-validation experiments into a separate, adequately powered contribution.

- **The “six-stage roadmap” overclaims validation from weak operational probes** (§4.1 Table; paragraph “What this calibrates”).  
  Table §4.1 labels Stage 3 as “operational probe passes” based on cross-language pick consistency, but the roadmap’s Stage 3 is “alignment of priors via category theory and information geometry,” i.e., formal cross-domain equivalence detection. Cross-language consistency of a scheduler is at best a weak invariance probe and does not test equivalence detection, merging, transfer, or formal alignment. Similarly, Stage 0 “works” is inferred from wrong priors hurting, not from a demonstrated library benefit under robust evaluation.  
  **Fix:** Rename these as “minimal proxies” and avoid statements like “three stages work”; report them as preliminary diagnostics only, with explicit hypotheses each proxy does and does not test.

- **Statistical treatment of the empirical-Bayes selection-bias model is questionable and underexplained** (§3.3).  
  The fitted Beta prior concentration \(\hat\alpha+\hat\beta \approx 6\times10^7\) is extreme and essentially collapses the prior to a point near 0.555. That may be a boundary/identifiability artifact of fitting 12 noisy binomial observations with little between-candidate variation, not strong evidence that all candidates have “near-identical true win-rates.” For a statistical audit paper, interpreting such an extreme empirical-Bayes fit requires diagnostics, sensitivity to priors, and comparison to alternative hierarchical models.  
  **Fix:** Provide sensitivity analyses with weakly informative hyperpriors, bounded concentration, nonparametric/bootstrap selection simulations, and show that the “drops are expected under selection” conclusion does not depend on the degenerate concentration estimate.

- **Decision thresholds are sometimes treated as substantive despite being arbitrary and near-miss sensitive** (§Gate; §3.5 Exp 66).  
  The inner gate threshold is fixed at \(0.60\), and Exp 66 reports W078 at inner \(0.590\) / L1 \(0.573\), wcand03 at \(0.580\) / \(0.574\), and wcand07 at \(0.565\) / \(0.558\). The paper correctly says the preregistered joint rule yields \(0/12\), but the narrative “clean null” can obscure that some candidates are close to the arbitrary threshold and pass L1. For a methodology paper, binary pass/fail should be secondary to estimated effects and uncertainty.  
  **Fix:** Present posterior/CI estimates for all candidates in the \(n=200\) run, explicitly discuss near-threshold sensitivity, and avoid implying that \(0.590\) versus \(0.600\) is substantively decisive.

- **The paper repeatedly uses point-estimate threshold failure as “the gate’s own criterion,” but the original gate’s statistical validity is itself criticized** (§3.2; §Contribution 1; §Contribution 4).  
  The argument says no candidate “clears \(0.60\) as a point estimate,” but the paper also argues that \(n=50\) point-estimate gating is statistically fragile. There is a tension in using the flawed criterion to declare audit failure, rather than using uncertainty-aware decision rules throughout.  
  **Fix:** Recast cached-audit results in terms of posterior probabilities / confidence intervals / predictive replication probabilities, and reserve point-threshold language only for reproducing the original loop’s decision rule.

- **Human ground truth is absent for the primary open-ended Chinese evaluation, leaving judge disagreement hard to interpret** (§Discussion “No human ground truth”; §Limitations).  
  The paper’s core empirical object is LLM-judged preference over open-ended answers. Cross-family disagreement shows lack of robustness, but it does not establish which judge family, if any, better tracks human utility. The authors acknowledge this, but for a top-venue audit-method paper, some human calibration would greatly strengthen the claim that the original accepts were spurious rather than merely under-valued by the audit panel.  
  **Fix:** Add a human evaluation on a stratified subset of cached and fresh pairs, ideally including all three original KEEPs and near-threshold Exp 66 candidates, with inter-rater agreement and comparison to judge families.

- **Positive controls are unresolved, undermining audit sensitivity** (§Discussion “Positive controls remain unresolved”).  
  The authors report that six constructed controls all fail the gate and admit that “no positive control we constructed actually functions as a positive.” This is a serious weakness for an audit stack: without a known-useful intervention, it is hard to know whether the stack detects true improvements or simply rejects additions in this scaffold. The authors even suggest the gate may be “structurally anti-additive,” which changes the interpretation of the whole case study.  
  **Fix:** Construct a synthetic or semi-synthetic task family where a specific inserted prior is known to improve objective accuracy, or use an independently human-validated intervention, and verify that the gate/audit stack can recover it.

- **The scaffold’s own claimed improvement is not audited under the paper’s standards** (§Contribution 3; Figures 1–2).  
  The paper motivates the loop using a scaffolded solver rising from \(74\%\) to \(86\%\), and Figure 1 says v20 reaches \(86\%\) vs \(74\%\). Contribution 3 later notes these gains are themselves measured by same-family LLM judging and should be read only under standard reporting conventions. This is honest, but it weakens the setup: the substrate producing candidates may itself be evaluated by the unreliable mechanism under critique.  
  **Fix:** Either audit the scaffold-vs-baseline comparison with the same cross-family/fresh-data standards, or remove the scaffold performance as a motivating empirical claim and present it only as background.

- **The exemplar mechanism creates a protocol ambiguity that is only partially resolved** (§3.5; §Limitations “Some held-out exposure”).  
  The solver retrieves same-domain examples from the held-out evaluation pool itself, and the paper argues this exposure is symmetric between base and ext. Symmetry helps relative comparison, but it does not eliminate distributional contamination, and the paper later shows the exemplar mechanism adds \(0.06\)–\(0.18\) absolute win-rate in the fresh runs. This is a major confound for interpreting the original cached signal.  
  **Fix:** Make the no-exemplar and with-exemplar protocols separate estimands from the beginning, and emphasize the cleanest strictly pool-separated result as primary.

- **The hierarchical Bayesian analysis is ultimately shown to be misleading, but remains prominently presented** (§3.4).  
  The \(n=30\) hierarchical model gives W078 posterior \(0.763\) for passing the joint rule, which is later “empirically falsified” at \(n=100\). The authors explain hindsight reasons, but this section still occupies substantial narrative space and may confuse the evidence hierarchy. For a methodology paper, a Bayesian model that fails because it omits a key protocol covariate should be treated as an error analysis, not a core contribution.  
  **Fix:** Move this analysis to a cautionary subsection or appendix, and provide the final Bayesian analysis including the \(n=100/n=200\) data and exemplar covariate as the main model.

- **The candidate-generation process and problem distribution are not sufficiently characterized in the main body** (§Method; §3.1).  
  The paper says there are 1768 Chinese open-ended problems, 100 training, 50 held-out, 50 extension, later fresh pids, and a mix of domains, but the main body gives little detail on task taxonomy, difficulty, answer format, or how “win” judgments relate to actual correctness/usefulness. For a case study, readers need to know what kind of generalization the null result is about.  
  **Fix:** Add a concise dataset/task table in the main body: domains, examples, split construction, prompt length, answer type, and judge rubric.

- **Some related-work claims are too sweeping and may be inaccurate in framing** (§Related Work Table 1; §Discussion implications).  
  The table says adjacent papers “almost never report” cross-family re-judgment and marks several with dashes. That may be true for the primary protocols, but different papers use different evaluators, human studies, environment rewards, or downstream metrics. The discussion then broadens to self-rewarding LMs, self-play, instruction backtranslation, and STaR-style methods, although the paper’s evidence is about retrieval-level prompt/library updates.  
  **Fix:** Tighten the comparison set to LLM-preference-gated retrieval/prompt updates, and avoid implying that weight-level self-improvement papers face the same failure mode unless their acceptance signal is directly same-family preference judging.

- **L6 faithfulness probes are weak and not validated** (§Theory L6; §Discussion L6).  
  The authors acknowledge that embedding-direction alignment and LLM-judged citation strictness are weak proxies and can fail in either direction. Since L6 is part of the advertised six-layer audit stack, its preliminary status should be clearer in the contributions and abstract.  
  **Fix:** Demote L6 from a core audit layer to an exploratory diagnostic unless validated on synthetic known-causal cases or human faithfulness labels.

- **Terminology and self-presentation are sometimes grandiose relative to evidence** (§Introduction “what people mean when they say intelligence”; §Roadmap category theory discussion).  
  Phrases such as “It is what people mean when they say ‘intelligence’” and the broad appeal to category theory/information geometry are not needed for the empirical audit and can read as overreach. Top-venue methodology papers benefit from precise, evidence-bound framing.  
  **Fix:** Remove or soften grand claims and keep the introduction focused on the specific evaluation failure and audit methodology.

# Questions to the authors

1. **Human calibration:** Do you have any human preference labels for the cached or fresh A/B pairs, especially the three original KEEPs and the near-threshold Exp 66 candidates? If so, how do human judgments align with gemini, Claude, and GPT-family judges?

2. **Final Bayesian model:** What is the posterior probability that each candidate passes the preregistered joint rule after including all \(n=100\), \(n=100\), and \(n=200\) fresh runs, with exemplar/no-exemplar as an explicit covariate?

3. **Selection-bias sensitivity:** Does the conclusion that cached drops are consistent with top-3-of-12 selection hold under non-degenerate hyperpriors or bounded Beta concentration, rather than the \(\alpha+\beta\approx 6\times10^7\) empirical-Bayes fit?

4. **Positive control:** Can the audit stack recover a known-useful inserted wisdom on a synthetic or objective task family? If not, how should readers distinguish “the candidate wisdoms are not useful” from “the scaffold/gate is anti-additive to any insertion”?

5. **Task distribution:** What are the domains, difficulty levels, and answer types in the 1768-problem Chinese pool and in the fresh pid samples? Were fresh pids sampled iid from the same pool or generated/filtered differently?

6. **Solver-family replication:** Have you run a complete candidate-generation → same-family gate → audit cycle with a non-gemini solver, rather than only cross-solver audits of the original KEEPs?

7. **Near-threshold Exp 66 interpretation:** For W078 at \(0.590/0.573\), what are the confidence intervals or posterior intervals, and how sensitive is the “0/12” conclusion to reasonable alternative preregistered thresholds such as inner \(\geq0.58\) or a posterior-probability rule?

# Rating

**Weak Reject**

The submission contains an interesting and unusually candid negative result, and the preregistered \(n=100/n=200\) replications make the central failure of this particular loop fairly convincing. However, the work is not yet a top-venue methodology contribution because the empirical base is still one main loop/substrate, there is no human ground truth or functioning positive control, and the audit stack itself was post-hoc before the fresh replications. The broad architectural roadmap and claims that several stages “work” are under-supported relative to the careful audit study and distract from the strongest contribution. I would be much more positive if the paper either narrowed itself to a rigorously scoped case study, or added human calibration, a known-positive control, and at least one independent full-loop replication.

# Confidence

**4**

I am familiar with LLM self-improvement loops, LLM-as-judge evaluation/auditing issues, and statistical concerns around preregistration, post-selection bias, and reproducibility; my confidence is somewhat limited by not seeing the appendices or artifacts.