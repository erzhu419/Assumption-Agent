# Summary

The paper presents a case study of a closed-loop LLM “wisdom library” system that proposes new methodological priors, validates them with a same-family LLM A/B gate, and then audits the accepted candidates. The central empirical claim is a negative result: the inner loop’s \(n=50\), \(+10\)pp same-family gate accepts 3/12 candidates, but those accepts do not survive cross-family re-judgment, fresh-data replication at larger \(n\), or several other audit probes. The paper then uses this failure to motivate a broader six-stage architecture for self-hypothesizing/self-validating agents and reports a small synthetic diagnostic campaign for each proposed stage.

# Strengths

- **Clear and valuable negative result on same-family LLM-as-judge gating.**  
  The core finding in Sections 3.1–3.5 is useful: a same-family \(n=50\) A/B gate accepted 3/12 candidate “wisdoms,” but larger and cross-family evaluations found \(0/12\) under the relevant thresholds. This is a practically important cautionary result for retrieval-level self-improvement loops.

- **The authors are unusually candid about post-hoc analysis and non-claims.**  
  The caveat at the start of Section 3 explicitly states that the six audit layers were not preregistered and were added sequentially in response to objections. The “Scope and non-claims” paragraph and “Objections we have not closed” section are also commendably honest and strengthen the credibility of the negative result.

- **The audit stack targets concrete nuisance variables.**  
  L1–L6 in the Introduction and Section 2.5 separately stress judge-family dependence, side-position effects, sample-size noise, solver-family interactions, domain shift, and faithfulness. Even if not all layers are equally validated, this decomposition is useful as an operational checklist for future LLM-as-judge self-improvement work.

- **Fresh-data replications substantially strengthen the main null.**  
  The paper does not rely only on cached cross-family re-judgment. Section 3.5 reports preregistered fresh evaluations at \(n=100\) without exemplars, \(n=100\) with exemplars, and \(n=200\) with exemplars, all yielding \(0/12\). This is much stronger evidence than the initial post-hoc audit alone.

- **Selection-bias / winner’s-curse analysis is directly relevant.**  
  Section 3.3 models top-3-of-12 selection on noisy \(n=50\) measurements and shows that the observed drops are plausible under regression to the mean. This is exactly the right statistical concern for a loop that selects the best-looking candidates from multiple trials.

- **The paper distinguishes cached-audit evidence from preregistered replication evidence.**  
  In Sections 3 and 4, the authors repeatedly state that the cached audit is post-hoc and that the fresh \(n=100/200\) replications carry more inferential weight. This separation prevents some common overinterpretation of exploratory audit results.

- **Reproducibility appears unusually strong for this kind of agentic paper.**  
  The main body describes released code, logs, registry states, judgment files, seeds, and preregistration documents, and the prompt notes that appendices include model identities, temperatures, seeds, proxy details, cost breakdowns, and prompt templates. This is a major positive for a methodology/case-study submission.

- **The paper identifies its own strongest missing controls.**  
  The “Positive controls remain unresolved” item in Discussion is especially important: all six attempted controls fail, leaving audit sensitivity untested. Acknowledging this directly is a strength, even though the issue remains severe.

# Weaknesses

- **The central empirical contribution is a single-loop case study, but many claims are phrased as field-level conclusions.**  
  The paper repeatedly moves from “our one wisdom-library loop failed” to broad claims such as “a same-family A/B gate at \(n=50\) is the worst possible substitute” in the abstract and conclusion, and “the field’s current proxy … has none of them” in the abstract/introduction. For a top-venue methodology paper, the evidence base is too narrow for such general language. A fix would be to run the audit protocol on multiple independently designed loops, at least one non-Gemini solver loop, and preferably one benchmark with objective rewards, or to substantially narrow the claims to “in this loop.”

- **No functioning positive control means the audit stack’s sensitivity is unvalidated.**  
  The Discussion states that Exp 44 and Exp 46 tried six controls and all failed the gate, and admits that “no positive control we constructed actually functions as a positive.” This is a serious issue: if the validation/audit pipeline cannot detect a known-useful intervention, then \(0/12\) may reflect an anti-additive scaffold/gate setup rather than absence of wisdom utility. The fix is essential: include a synthetic task family or independently human-validated intervention where the inserted wisdom is known to improve objective outcomes, and show the gate/audit stack accepts it.

- **The original evaluation pool has same-domain exemplar exposure, and “symmetry” does not fully remove the concern.**  
  In Limitations, the authors state that v20 retrieves a same-domain example from the held-out evaluation pool itself at solve time, but argue this applies symmetrically to base and ext. Symmetric exposure does not guarantee cancellation, because the new wisdom can interact with the retrieved exemplar differently than the base condition, and the paper itself later finds an exemplar boost of \(0.06\)–\(0.18\). The fix is to make the strictly separated-pool/no-exemplar protocol the primary result from the beginning, and to treat all in-pool-exemplar results as contaminated sensitivity analyses.

- **The “candidate’s own utility” decomposition is overinterpreted.**  
  Section 3.5 concludes that cached signal is consistent with exemplar boost plus top-3-of-12 selection bias and says “none of the three is the candidate’s own utility.” But the experiments do not identify candidate-specific utility; the theory section itself correctly says the components are not identifiable. The fix is to avoid exclusionary wording and instead say the data are consistent with non-specific explanations and do not establish content-specific utility.

- **Several numerical/statistical statements are imprecise or incorrect in wording.**  
  Section 3.3 says W076’s observed drop of “0.20 percentage points,” but from 0.64 to roughly 0.44 this is 20 percentage points or 0.20 absolute win-rate units, not 0.20 percentage points. Similar language appears elsewhere: “exemplar boost \(0.10\)--\(0.20\) pp” in the abstract/intro appears to mean 10–20 percentage points, not 0.10–0.20 pp. The fix is to consistently use “absolute win-rate units” or “percentage points” correctly throughout.

- **The gate threshold is described as statistically strict, but \(n=50\) and \(+10\)pp is not reliably strict after selecting among 12 candidates.**  
  Section 2.4 says the \(+10\)pp threshold is “intentionally strict” and refuses candidates indistinguishable from sampling noise, but a 0.60 win rate over 50 trials has a wide interval and is particularly fragile under 12-fold selection. The later results prove this. The fix is to revise the methodological description: the gate was operationally strict relative to a heuristic threshold, but statistically underpowered for selected candidates.

- **The hierarchical Bayesian analysis is not sufficiently specified in the main body for the weight placed on it.**  
  Section 3.4 reports a multilevel logistic model with random effects for candidate, judge family, and pid via Metropolis-within-Gibbs, yielding posterior decision probabilities such as 0.763 for W078. But the main body does not specify priors, convergence diagnostics, posterior predictive checks, or how dependence among cached and fresh verdict arrays is handled. Since this analysis is used to explain the \(n=30\) positive and its later falsification, the main body should include enough detail to assess it, with the appendix carrying only additional diagnostics.

- **The roadmap contribution is much less mature than the audit contribution.**  
  Section 4 proposes a broad six-stage architecture involving priors, world models, schedulers, experience, formal alignment, and new-prior generation. But the empirical validation in Section 4.1 is a small synthetic 60-problem environment with simple one-sentence priors and mostly LLM-judged/proxy outcomes. This is not enough to substantiate claims about a “working architecture” for self-hypothesizing agents. The fix is either to demote the roadmap to discussion/speculation or to provide a real integrated prototype where multiple stages interact on a nontrivial benchmark.

- **Stage 3 “alignment” is not actually the formal alignment layer described.**  
  Section 4 describes Stage 3 using category theory, Markov categories, Blackwell equivalence, and information geometry. But Section 4.1 tests only cross-language scheduler consistency over translated problems. Cross-language pick agreement does not validate formal equivalence detection among priors such as Le Chatelier/Lenz or RC discharge/radioactive decay. The fix is to either remove the category-theoretic claims or add experiments where the system detects and merges structurally equivalent priors across domains with ground-truth equivalence labels.

- **The paper has internal inconsistencies about the number and framing of stages.**  
  The Introduction says “six moving parts” called stages 0 through 4 with 0.5 as a half-stage; Section 4 says “five sequential stages”; later it again says “six stages including 0.5.” This is minor but symptomatic of an overgrown paper. The fix is to standardize terminology: either six components or five stages plus one auxiliary component.

- **The related-work table risks overstating novelty by reducing adjacent papers to missing audit checkmarks.**  
  Table 1 marks several adjacent works as not reporting cross-family re-judgment, sample extension, cross-solver, or faithfulness. This is useful, but the comparison is somewhat unfair because those works often have different acceptance signals, tasks, and evaluation regimes. The paper partly acknowledges this for STaR/SPIN/Voyager, but still makes broad claims about “current self-improvement literature.” The fix is to restrict the table to directly comparable LLM-preference-gated retrieval/prompt-update systems or provide more nuanced columns for objective rewards/human eval/held-out task performance.

- **The scaffold’s claimed baseline improvement is not independently audited, yet it motivates the whole system.**  
  Contribution 3 says v20 reaches strong pairwise win rates versus baselines but explicitly notes these gains were themselves measured by same-family judging and are not re-audited. Since the loop’s candidate generation and exemplar effects depend on this scaffold, uncertainty about whether the scaffold genuinely improves performance weakens the case study. The fix is to audit the scaffold-vs-baseline comparison with the same cross-family/fresh/objective checks, or remove scaffold-performance claims from the main argument.

- **The multi-step research-task pilot is too small to support even a secondary claim.**  
  Section 3 reports \(N=3\) scenarios, with two favoring multi-turn + wisdom and one tie. This is fine as an appendix demo, but in the main body it distracts from the core audit result and cannot meaningfully validate extension to sequential tasks. The fix is to move this entirely to the appendix or expand it into a benchmark-sized evaluation.

- **The paper’s rhetoric sometimes exceeds its evidence.**  
  Phrases such as “It is what people mean when they say ‘intelligence’,” “none of these is optional,” and “worst possible self-validator” are not necessary and read as advocacy rather than evidence. This is especially problematic for a methodology paper whose strongest contribution is careful negative evidence. The fix is to use narrower, evidence-tethered language.

- **There are signs of editing/LaTeX inconsistency that obscure the theory section.**  
  In Section 2.5, a paragraph ends mid-thought: “two families may share style preferences that” before a new paragraph restarts “What this means for cross-family re-judgment.” This duplicated/truncated text should be fixed. The paper is long and contains repeated summaries of the same result; tightening would improve clarity.

# Questions to the authors

1. Can you provide a true positive-control experiment where a known-useful inserted wisdom/intervention improves objective task outcomes, and show that your gate plus audit stack accepts it?

2. In the \(n=100\) and \(n=200\) fresh replications, are the problem samples fully disjoint not only from the original held-out set but also from any examples used in retrieval, exemplar mining, candidate generation, and prompt construction?

3. What priors, convergence diagnostics, and posterior predictive checks were used for the hierarchical Bayesian model in Section 3.4, and how sensitive are the posterior decision probabilities to alternative priors and correlation structures?

4. If the same audit stack is applied to the v20 scaffold-vs-baseline comparison in Figure 1/Figure 2, do the claimed 74% to 86% or 0.88-vs-baseline gains survive cross-family judging and fresh-data replication?

5. Can the entire closed loop—candidate generation, same-family gate, and audit—be rerun from scratch with a non-Gemini solver/judge family to test whether the failure mode is specific to this solver-judge configuration?

6. For the Stage 3 alignment claim, do you have any experiment where the system identifies structural equivalence between different priors across domains, rather than merely making consistent scheduler choices across translations?

7. How often do ties occur for each judge family across the main cached and fresh evaluations, and are tie-handling rules identical across the inner gate and audit layers?

# Rating

**Weak Reject**

The paper contains an interesting and useful negative result, and the authors deserve credit for unusually candid reporting, preregistered fresh replications, and strong artifact orientation. However, for a top ML venue main track, the contribution is still too narrow and internally uneven: it is a single-loop case study, lacks a functioning positive control, relies on an originally contaminated/exemplar-exposed protocol, and overextends from the case study to broad claims about self-improving LLM loops. The six-stage roadmap is much weaker than the audit result and is supported only by toy diagnostics, with the formal alignment/world-model/new-prior claims especially underdeveloped. I would be more positive on a shorter paper framed squarely as an audit case study with positive controls and more restrained claims.

# Confidence

**4** — I am familiar with LLM-as-judge evaluation, self-reflection/self-improvement loop papers, and reproducibility/preregistration issues; I am less expert in the category-theoretic alignment material, but that material is not central to my rating.