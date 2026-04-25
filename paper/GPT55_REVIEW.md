# Summary

The paper presents a fully logged case study of an autonomous LLM “wisdom library” loop: the loop proposes candidate retrieval entries, gates them with a same-family LLM A/B test, and then audits the accepted entries with cross-family judges, larger samples, cross-solver tests, fresh domains, faithfulness probes, and a preregistered fresh split. Its central empirical story is that the original same-family \(n=50\) gate is unstable: on cached data the three accepted wisdoms do not reproduce at the original \(0.60\) threshold, but a later preregistered fresh split recovers two of the three under a laxer \(0.55\) L1 threshold. The claimed contribution is therefore not that the loop improves itself, but that a multi-layer audit stack can distinguish selection/regression-to-mean artifacts from more robust candidate signals.

# Strengths

- **The paper is unusually explicit about scope and negative results**  
  In the abstract and “Honest scope” paragraph, the authors state that this is a case study, not evidence that self-improving LLM loops generally work. They also foreground the null/partial-null results rather than hiding them. This honesty strengthens the submission relative to many self-improvement papers that report only inner-loop gains.

- **Identical-answer cross-family re-judgment is a clean and useful audit primitive**  
  Experiment 1 rejudges the same 150 cached base/ext answer pairs with Claude Opus 4.6, changing only the judge family. This is a strong design choice for isolating judge-family dependence and is practically cheap enough to recommend as an audit item.

- **The authors explicitly acknowledge and model winner’s-curse / selection bias**  
  The Exp~45 selection-adjusted analysis is important: it shows that the observed L1 drops are statistically consistent with regression to the mean after selecting the top 3 of 12 noisy \(n=50\) measurements. This substantially improves the paper by preventing the over-simple conclusion that all L1 drops are “judge fragility.”

- **The paper reports confidence intervals and tie-handling conventions**  
  Section “Tie handling and effective \(n\)” and Table~\ref{tab:ci} specify that ties are excluded and Wilson intervals use non-tie effective \(n\). The authors also correctly note that many intervals include both parity and the threshold. This makes the statistical fragility visible.

- **The audit stack probes multiple nuisance variables rather than only rerunning the same gate**  
  L1--L6 cover judge family, side randomization, sample extension, solver family, domain transfer, and non-pairwise faithfulness. Even though these are not independent evidence streams, the breadth is valuable for a methodology case study.

- **The preregistered fresh split is a meaningful correction to the post-hoc audit concern**  
  Exp~47 is the strongest experiment in the paper because thresholds are frozen before the fresh 30-pid run. It also usefully complicates the story: two original KEEPs recover, one fails, and one original REVERT is rescued.

- **The paper preserves failures of its own attempted fixes**  
  The trigger-conditioned gate in Exp~15 looks promising on the original 12 candidates but fails prospectively on the 9 new candidates from Exp~29/33. Reporting this failed generalization strengthens the credibility of the case study.

- **Artifact orientation is strong**  
  The conclusion and reproducibility paragraph promise code, cached answers, judgment files, model settings, registry states, and prompts, with appendices covering prompts and reproducibility details. For an audit methodology paper, this artifact trail is a major positive.

# Weaknesses

- **The primary claim is internally inconsistent after Exp~47**  
  The abstract opens with “zero of its committed wisdoms survive” and later says “2/3 of the original KEEPs do replicate under the preregistered protocol.” Contribution 4 says “Each is subsequently falsified by the audit stack,” while the conclusion says W077 and W078 recover and only W076 is a true non-replication. For this class of methodology paper, the primary endpoint must be unambiguous; otherwise the audit stack appears to be reinterpreted after each new result.  
  **Fix:** Rewrite the paper around one primary estimand, e.g. “cached-data strict \(0.60\) non-reproduction” versus “fresh-split \(0.55\) replication,” and remove all remaining claims that “0/3 survive” universally.

- **The audit stack is largely post-hoc, and the paper does not adequately control the garden of forking paths**  
  Section “A note on staged stress testing” admits the full stack was not preregistered and that layers were added sequentially in response to reviewer-simulation objections. Thresholds being fixed before each layer’s data is not enough, because the choice of which layers to add and which analyses to emphasize remains data- and objection-dependent. For a top-venue methodology paper, the method itself needs prospective validation.  
  **Fix:** Preregister the full audit protocol and apply it to a genuinely new loop or at least a second full cycle, reporting all attempted analyses and primary/secondary endpoints.

- **There is no reliable external ground truth for the main open-ended Chinese task**  
  Most claims are based on LLM-as-judge preferences, while the paper repeatedly documents low inter-judge agreement and temporal drift. GPT-5.5 structured ratings are explicitly “not human annotation,” and GSM8K is a small \(n=30\) ceiling-effect probe far from the main task. Without human or objective labels, it is hard to know whether the audit stack is correcting the original judge or merely replacing one unstable preference system with another.  
  **Fix:** Add blinded human evaluation on a stratified subset of the Chinese open-ended pairs, or design synthetic/objective tasks where the correct effect of a wisdom is known.

- **The statistical evidence is weaker than the narrative often implies**  
  Table~\ref{tab:ci} shows that the original KEEPs’ CIs often include parity and that several audit CIs include the \(0.60\) threshold. The paper frequently uses point-estimate language such as “FLIP,” “collapse,” and “zero survive,” even when intervals overlap substantially. The p-values against an arbitrary \(p_0=0.30\) “random inclusion” reference are not meaningful evidence of gate quality.  
  **Fix:** Replace threshold-crossing rhetoric with a hierarchical uncertainty model over candidates, judges, domains, and solvers; report posterior probabilities of clearing thresholds rather than binary point-estimate decisions.

- **The selection-bias analysis is important but not sufficient for the conclusions drawn from it**  
  Exp~45 models regression to the mean for the top 3 of 12 noisy gate scores, but the L1 audit also changes the estimand by switching judge family and tie behavior. Later claims that “other audit layers, which selection bias does not predict, carry signal beyond regression-to-the-mean” are not justified: sample extension, fresh-domain tests, and cross-solver reruns can all show drops after selecting noisy top candidates.  
  **Fix:** Simulate the entire selection-and-audit pipeline under a null model with candidate generation, judge random effects, solver/domain effects, and tie propensities, rather than modeling only the top-3 gate scores.

- **The “six layers” are not independent, and the paper sometimes still counts them rhetorically as independent confirmation**  
  The abstract correctly says the layers are “conditionally distinct” and reuse KEEPs/cached pairs, but later sections describe “six independent confirming data streams” and “every conditionally-distinct axis” as if the evidence compounds. Many layers share the same candidates, answer pairs, judges, or problem pools.  
  **Fix:** Quantify dependence among audit outcomes, avoid language implying independent replication, and present the layers as diagnostic stress tests rather than additive evidence.

- **The theoretical decomposition is not identifiable and occasionally overclaims despite caveats**  
  Section~\ref{sec:theory} says the decomposition into \(Z^{\mathrm{specific}}, Z^{\mathrm{generic}}, Z^{\mathrm{style}}\) is informal and not identifiable, but later says layers “cancel” terms and argues “by elimination” that \(Z^{\mathrm{specific}}\) is near zero. That inference is invalid because the interventions do not isolate components and because judge, solver, domain, and generic-context effects can interact.  
  **Fix:** Remove the causal/elimination language and frame the decomposition purely as a taxonomy of possible confounds; validate any causal claims with placebo, length-matched, and known-effect interventions.

- **Exp~47 is underpowered and is not clearly a “fresh-loop re-execution”**  
  Section~\ref{sec:exp47} says the entire loop is re-executed from scratch, but the table evaluates the original 12 candidates “unchanged” on 30 fresh pids. It is therefore a fresh validation split for existing candidates, not a full fresh loop with new candidate generation, pruning, and gate decisions. Also, \(n=30\) gives very wide Wilson intervals, and the L1 threshold is relaxed to \(0.55\).  
  **Fix:** Rename it accurately, or actually rerun the full candidate-generation loop on fresh data; increase to \(n\geq100\) and justify or preregister a single threshold policy.

- **Positive controls do not establish audit sensitivity**  
  Exp~44 and Exp~46 construct six controls, including “useful” controls and a duplicate of a base-library wisdom, but all fail. The authors correctly say sensitivity remains untested, but this is a major unresolved issue: an audit stack that rejects all constructed positives may be overly conservative or the task setup may be anti-additive.  
  **Fix:** Use a synthetic task where a specific intervention is known to help, or remove a demonstrably useful existing wisdom from the base library and test whether reinserting it is accepted.

- **Evaluation-pool exposure through dynamic same-domain exemplars is not fully neutralized by symmetry**  
  The Limitations section states that, at solve time, each evaluation pid receives a same-domain exemplar selected from other pids in the evaluation pool. Even if base and ext both receive the same exposure, the added wisdom may interact with the exemplar differently, so relative wr is not guaranteed exposure-neutral. For an audit methodology paper, this weakens claims about held-out validation.  
  **Fix:** Rerun the key inner gate and Exp~47-style audit with strict pool separation and no evaluation-pool exemplars.

- **The scope is too narrow for the strength of the recommendation**  
  The main loop is one Chinese open-ended wisdom-library cycle with one primary solver family. Exp~39 English and Exp~32/42 GSM8K probes are small and not full closed-loop replications. Yet the paper recommends default reporting practices for retrieval-level self-improvement claims broadly.  
  **Fix:** Apply the audit stack to at least one additional independently designed loop, preferably in English and with another solver family, before making field-level reporting recommendations.

- **L6 faithfulness measures are weakly validated**  
  Embedding cosine between Turn-0 “what_changed” deltas and wisdom descriptions is a very indirect measure of causal faithfulness. The LLM citation/YES/PARTIAL rubric is also prompt-sensitive, as Exp~17 shows by increasing citations through an explicit `used_wisdoms` field without proving utility. For a paper claiming a faithfulness audit layer, this is underdeveloped.  
  **Fix:** Calibrate L6 on synthetic cases with known inserted wisdom effects, include human faithfulness labels, and separate “citation,” “causal use,” and “utility conditional on use.”

- **The organization is extremely hard to follow and contains numbering/name collisions**  
  The main body includes dozens of experiments, out-of-order numbering, “L1” both as an audit layer and as “measurement redesign,” and references such as “Exp 11 majority-of-3” even though Exp~11 is also LLM-judged faithfulness. This makes the methodology difficult to reproduce from the main paper.  
  **Fix:** Move chronological/reviewer-response experiments to an appendix; in the main text, present a clean protocol, a predesignated result table, and a small number of confirmatory analyses.

- **Reproducibility remains compromised by hosted-model drift and third-party proxy routing**  
  The Limitations section and Exp~37/42 explicitly document temporal drift even at temperature 0.0 or on identical GSM8K inputs. Access is through `ruoli.dev`, not official endpoints, so exact reruns may be impossible. This is especially problematic because the paper’s central object is measurement stability.  
  **Fix:** Archive all cached prompts/responses/verdicts, use official endpoints where possible, include open-model replications, and make claims about the cached artifact rather than reproducible model behavior unless independently rerun.

- **The tone sometimes overstates novelty and closure**  
  Phrases such as “first paper,” “strict-reviewer closure,” and “fatal objections closed” appear repeatedly, while the paper itself later reopens several of those objections. Cross-family rejudgment and multi-judge audits are not wholly new in LLM evaluation, even if applying them to self-improving retrieval loops is useful.  
  **Fix:** Tone down self-praise, state novelty narrowly, and compare more carefully to existing LLM-as-judge reliability and evaluation-audit literature.

# Questions to the authors

1. **What is the primary endpoint of the paper after Exp~47?** Is it cached-data non-reproduction at \(0.60\), fresh-split replication at \(0.55\), or diagnosis of W076 as the only true non-replicating KEEP?

2. **Was Exp~47 a full fresh loop or only a fresh validation of the original 12 candidates?** If candidate generation/pruning were not rerun, please rename it and clarify exactly which parts of the loop were executed.

3. **Can you provide a complete audit-trail table of all analyses attempted, including failed or abandoned ones, with timestamps and threshold decisions?** This would materially affect how much to trust the post-hoc audit stack.

4. **Do you have any blinded human judgments on the Chinese open-ended answer pairs?** Even a modest human panel on the three KEEPs and key REVERTs would substantially clarify whether the audit panel or the original judge is closer to human preference.

5. **Can Exp~45 be extended to simulate the full selection-and-audit protocol, including cross-family judge effects, tie propensities, domain shifts, and the fresh split?** The current selection model is useful but too narrow for the conclusions drawn.

6. **Can you construct a true positive control?** For example, remove a known useful base wisdom and test reinsertion, or build a synthetic task where one wisdom deterministically helps.

7. **What happens if the key inner-loop gate and Exp~47 validation are rerun without same-domain exemplars drawn from the evaluation pool?** The symmetry argument is plausible but not decisive.

8. **Are the exact cached answer pairs and judge prompts sufficient to reproduce every table without querying the proxy?** If not, which claims require live model calls and are therefore subject to temporal drift?

# Rating

Weak Reject

The submission is interesting, unusually honest, and potentially valuable as an audit case study, but it is not yet a solid top-venue methodology contribution. The most serious issues are the inconsistent primary claim after Exp~47, the largely post-hoc construction of the audit stack, the lack of external ground truth for the main task, and statistical over-reliance on threshold-crossing point estimates with small \(n\). I also do not think the current evidence validates the audit stack’s sensitivity, because all constructed positive controls fail and the only prospective recovery is an underpowered \(n=30\) fresh validation with a relaxed threshold. A tighter, preregistered second-loop study with human/objective validation and a true positive control could make this substantially stronger.

# Confidence

4. I am familiar with self-improving / self-refining LLM loops, LLM-as-judge reliability issues, and reproducibility/preregistration concerns in ML evaluation, though I am not a specialist in Chinese aphorism-based retrieval libraries specifically.