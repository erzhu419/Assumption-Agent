# Summary

The paper presents a case study of an autonomous “wisdom-library” LLM loop that proposes methodological aphorisms, validates them with a same-family LLM A/B gate at \(n=50\), and commits candidates passing a \(+10\)pp threshold. The central empirical claim is negative: the loop’s \(3/12\) accepted candidates do not survive cross-family re-judgment, larger fresh-data replications, or selection-bias analysis, so same-family A/B gating is unreliable for this kind of self-improvement claim. The paper then uses this null result to motivate a six-stage architecture for self-hypothesizing/self-validating agents and reports a small synthetic diagnostic campaign suggesting which stages are easy, conditional, or currently failing.

# Strengths

- **Clear and valuable negative result on a concrete self-improvement loop**  
  In the abstract, Introduction, and Experiments summary, the authors explicitly state that “self-improvement works” is *not* the contribution; the main result is that the minimal loop’s \(3/12\) KEEP decisions collapse under audit. This is a useful corrective to a literature where positive self-improvement claims are often accepted on weak LLM-as-judge evidence.

- **Multiple audit axes target distinct failure modes**  
  The six-layer audit stack in the Introduction and Method—cross-family re-judgment, side randomization, sample extension, cross-solver replication, fresh-domain testing, and faithfulness probes—is well motivated by concrete nuisance sources: judge preference, side bias, sample noise, solver interaction, distribution shift, and lack of causal use. Even if not all layers are novel, bundling them into a reusable checklist is practically valuable.

- **Honest treatment of post-hoc analysis and limitations**  
  The Experiments caveat explicitly states that the audit layers were not preregistered and were added in response to objections. The Discussion similarly lists gate-design hindsight, post-hoc audit selection, solver-family scope, lack of human ground truth, and missing positive controls as unresolved objections. This transparency strengthens the paper considerably.

- **Fresh-data replications are a major improvement over cached-only audit**  
  The progression from cached audit to \(n=30\), then \(n=100\) without exemplars, \(n=100\) with exemplars, and \(n=200\) with exemplars is one of the strongest parts of the submission. The triple \(0/12\) fresh-data result in Section “Three preregistered fresh-data replications” is much more persuasive than the initial cross-family cached re-judgment alone.

- **Selection-bias / winner’s curse analysis addresses an important ambiguity**  
  Section “Selection bias” correctly notes that cached drops after selecting the top 3 of 12 noisy measurements are ambiguous. Modeling the expected shrinkage under top-\(k\) selection is exactly the right kind of diagnostic for this setting, even though the implementation details need more scrutiny.

- **The authors avoid overinterpreting cross-family re-judgment as ground truth**  
  Section “What cross-family re-judgment can and cannot deliver” carefully states that a judge-family swap cannot identify content-specific utility and may conflate calibration, verbosity preference, tie propensity, and competence. This is a nuanced and correct framing.

- **Reproducibility orientation is unusually strong**  
  The main body describes released code, logs, model identities, seeds, registry states, and judgment files, and the user notes that the appendix includes prompts, compute/cost, schema examples, and reproducibility details. For a methodology/case-study paper, this materially improves auditability.

# Weaknesses

- **The paper’s main empirical scope is one loop, one primary solver family, one task distribution, yet some claims are written too broadly**  
  The Introduction and Conclusion repeatedly describe same-family \(n=50\) A/B gating as “the worst possible substitute” or “the field’s current proxy” having “none” of the required ingredients. For a top-venue methodology paper, a single-loop Chinese wisdom-library case study cannot support such broad field-level conclusions. The fix is to rephrase the claims as scoped recommendations, or add at least one independently implemented loop with a different solver, generator, judge, and task distribution.

- **No functioning positive control means the audit stack’s sensitivity is unknown**  
  The Discussion explicitly states that Exp 44 and Exp 46 tried six controls and all \(6/6\) failed the gate, so “no positive control we constructed actually functions as a positive.” This is severe: if the gate/audit stack rejects even known-useful or intended-useful controls, then \(0/12\) may reflect an anti-additive scaffold, an overly harsh evaluation design, or task insensitivity rather than failure of candidate wisdom content. A fix would be to include synthetic tasks with deterministic known-useful interventions, or human-validated helpful wisdoms with objective outcomes, and show that the audit stack accepts them.

- **The same-domain in-pool exemplar mechanism is a serious evaluation contamination risk, and “symmetry” does not fully solve it**  
  In Limitations, the solver retrieves a same-domain example from the held-out evaluation pool itself, including that pid’s cached prior-version answer. The authors argue this applies symmetrically to base and ext, so it cancels in the relative comparison. But symmetric exposure need not cancel if the added wisdom interacts with the exemplar, changes retrieval use, or changes how the solver copies/adapts the example. The \(n=100\) no-exemplar and with-exemplar sensitivity helps, but the clean fix is to run the full original gate and audit under strict pool separation from the start.

- **Several internal inconsistencies weaken confidence in experimental accounting**  
  The abstract says there are three preregistered fresh-data replications including \(n=200\), while the Introduction’s “What we found” and roadmap still say “two preregistered fresh \(n=100\) replications.” Section “Three preregistered fresh-data replications” says the \(n=200\) top three are “below both inner-\(0.60\) and L1-\(0.55\) thresholds,” but the listed L1 values are \(0.573\), \(0.574\), and \(0.558\), all above \(0.55\). These may be editing errors, but in a paper where exact decision rules are central, they need correction throughout.

- **The \(+10\)pp gate is described as statistically strict, but at \(n=50\) it is not a strong statistical criterion**  
  Section “The A/B validation gate” says the threshold is “intentionally strict” and refuses candidates indistinguishable from sampling noise. But \(30/50=0.60\) versus \(0.50\) is not a strong signal; a one-sided binomial test is weak, and with 12 candidates the multiple-comparison/winner’s-curse issue is substantial. The paper later recognizes this empirically, but the gate should not be described as strict in the statistical sense. The fix is to distinguish “operationally high threshold” from “statistically well-powered threshold,” and define gates using preregistered confidence/posterior criteria.

- **The selection-bias model is useful but under-specified in the main body**  
  Section “Selection bias” says an empirical-Bayes Beta prior was fitted to the 12 candidate win rates and used to simulate “measure 12, accept top 3, re-measure.” But the main body does not specify the fitted prior, whether uncertainty in the prior is propagated, whether candidates are exchangeable, whether judge-family effects are modeled, or whether the “top 3” matches the actual threshold-based acceptance process. A fix would be to include the model equations, fitted hyperparameters, and sensitivity to alternative priors in the main text or a compact table.

- **The hierarchical Bayesian analysis is ultimately misleadingly prominent given that it is falsified by later data**  
  Section “Hierarchical Bayes on the small-\(n\) fresh data” reports posterior support \(0.763\) for W078 under the preregistered decision rule, then the next section says larger \(n\) empirically falsifies this. This is honest, but the paper should more directly diagnose why the model failed—prior choice, pooling assumptions, omitted exemplar/protocol factors, pid effects, or distribution shift. Otherwise the reader learns that the hierarchical model was overconfident but not how to use it safely in future audits.

- **The roadmap validation experiments are too small and synthetic to support the architectural claims made around them**  
  Section “Empirical validation of each stage” uses 60 problems across 4 synthetic task families, 5 languages, and 3 solver families. This is useful as a diagnostic toy environment, but claims like “Stage 3 strongly works” and “the scheduler operates on underlying problem structure” are too strong. Cross-language pick consistency is not the same as the formal alignment layer involving category theory, Markov categories, or Blackwell equivalence described earlier. The fix is to label these as toy operational probes, not validation of the proposed architectural stages.

- **The Stage 3 theory is largely disconnected from the experiment actually run**  
  The roadmap describes alignment via category theory and information geometry, detecting equivalences such as Le Chatelier/Lenz or Markov-category isomorphisms. The experiment tests whether an LLM scheduler picks the same prior after translating problems into five languages. That is cross-lingual robustness, not formal alignment of priors. Either remove the category-theoretic framing or add an experiment that actually tests equivalence detection/merging across formally analogous domains.

- **Scaffold performance claims are based on the same kind of same-family judging the paper criticizes**  
  Contribution 3 says the v20 scaffold reaches strong win rates against baselines, but acknowledges these were “measured by a same-family pairwise LLM judge” and are not independently audited. Since the scaffold is the substrate for all candidate generation and evaluation, uncertainty about its real performance matters. The fix is either to audit the scaffold-vs-baseline claims with the same L1/L3/L4 stack or to remove performance numbers not needed for the audit contribution.

- **The candidate-generation loop is not re-run under the preregistered audit protocol**  
  The stronger fresh-data replications evaluate the original 12 candidates under better protocols, but the full closed loop—generation, selection, pruning, commitment—is not rerun prospectively with the final audit stack fixed. The Discussion acknowledges this. For a methodology paper about auditing self-improving loops, the strongest validation would be a fully preregistered second loop where all gate/audit choices are frozen before candidate generation.

- **The paper sometimes conflates point-estimate threshold failure with statistical evidence of no effect**  
  Several sections emphasize that no candidate “clears” \(0.60\) as a point estimate, while also noting that Wilson intervals sometimes touch \(0.60\). This is acceptable for evaluating the gate’s own decision rule, but not equivalent to showing no content-specific utility. The paper mostly says this, but phrases like “clean null” and “collapses” can overstate the evidence. The fix is to consistently distinguish “fails the operational acceptance rule” from “evidence of zero or negative effect.”

- **The multi-step research-task pilot is too small to contribute meaningfully**  
  Section “Multi-step research-task pilot” reports \(N=3\), with two scenarios favoring multi-turn + wisdom and one tie. This does not validate extension to sequential tasks and is too small for a main-body result. It should be moved entirely to appendix or framed only as an implementation smoke test.

- **The related-work table risks overstating the novelty of the audit recommendation**  
  Table “Audit practices in adjacent LLM-judged self-improvement papers” marks many papers as lacking L1/L3/L4/L6. The observation is useful, but the comparison mixes quite different systems and acceptance signals. The paper partially handles this with “n/a” rows, but the broader discussion still implies a general gap across self-improvement literature. The fix is to limit the claim to retrieval/prompt-level LLM-judged acceptance loops, or add a more systematic survey.

- **Reproducibility via a third-party proxy limits exact replication**  
  Limitations state that model access is mediated by ruoli.dev and exact-token reproduction depends on proxy routing state. This is not fatal, especially with cached outputs, but it weakens claims about re-running the full loop. The fix is to provide cached answer/judgment artifacts as primary reproducibility targets and, ideally, replicate key cells on official vendor endpoints.

# Questions to the authors

1. Can you provide a functioning positive control where a known-useful inserted wisdom/intervention is accepted by the gate and survives the audit stack? If not, how should readers distinguish “all candidates are useless” from “the evaluation setup is insensitive or anti-additive”?

2. For the \(n=200\) replication, why does the text say the top candidates are below the L1-\(0.55\) threshold when the listed L1 values are \(0.573\), \(0.574\), and \(0.558\)? Was the joint rule inner \(\geq 0.60\) *and* L1 \(\geq 0.55\), or was a different L1 criterion used?

3. What exactly was preregistered for the \(n=100\) and \(n=200\) runs: candidate set, thresholds, exemplar condition, seeds, judge families, exclusion rules, and analysis code? Were any analyses or candidate subsets added after seeing results?

4. In the empirical-Bayes winner’s-curse analysis, what are the fitted Beta hyperparameters, and how sensitive are the simulated drop percentiles to alternative priors or to modeling the actual threshold rule rather than top-3 selection?

5. Why did the hierarchical Bayesian model assign W078 posterior support \(0.763\) before being contradicted by the \(n=100\) and \(n=200\) data? Which modeling assumption failed, and what revised model would you recommend for future users of the audit stack?

6. Can the full closed loop be rerun prospectively with strict pool separation and the final audit protocol fixed in advance, rather than only re-evaluating the original 12 candidates?

7. For the roadmap experiments, what evidence shows that the Stage 3 cross-language consistency probe measures “alignment of priors” rather than ordinary translation robustness of the scheduler prompt?

# Rating

**Weak Reject**

The paper is interesting, unusually transparent, and contains a valuable negative result, but it is not yet a convincing top-venue methodology contribution. The biggest issues are the lack of a functioning positive control, the single-loop/single-primary-solver scope, the contaminated same-domain exemplar mechanism, and the fact that the final audit protocol is not applied prospectively to a fresh closed-loop run. I also find the roadmap section substantially overclaimed relative to the toy diagnostics, especially the formal alignment claims. With a clean preregistered second loop, a working positive control, corrected inconsistencies, and tighter scoping, this could become a strong case-study paper.

# Confidence

**4**

I am familiar with LLM self-improvement loops, LLM-as-judge evaluation fragility, cross-model judging, winner’s-curse effects in adaptive experimentation, and reproducibility/preregistration issues, though I am less specialized in the category-theoretic alignment framing invoked in the roadmap.