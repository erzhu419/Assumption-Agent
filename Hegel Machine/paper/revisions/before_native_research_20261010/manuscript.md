# Hegel Machine: A Framework for Recursive Hypothesis Formation, Testing, and Revision

## Abstract

An agent's hypotheses depend on what it can represent, observe and test. We present Hegel Machine, a framework for revising these conditions of research alongside predictive explanations. An unresolved supporting assumption becomes a child hypothesis; its evidence can change the parent's next experiment, representation or explanation. The framework treats research procedures and meta-priors as revisable objects; our implementations revise finite response models, experiment procedures and search preferences. A functional-analytic perspective makes the coupling explicit: available observations induce a geometry on candidate predictions, and experiments that cannot separate alternatives leave directions of disagreement invisible. Research can therefore change the observations as well as the fitted hypothesis. Finite implementations demonstrate supporting evidence changing a pending experiment and new measurement access restoring prediction in 30/30 cases versus 0/30 without it. A language-model-authored procedure revision improves frozen transfer from 10/18 to 18/18. Preserving a refuted program also improves later repair and return, with 90/90 continuation successes versus 69/90 after deletion. These mechanisms do not establish overall agent superiority: matched flat research attains equal accuracy, and complete Code, textual-memory and Hegel agents solve 29/30, 29/30 and 26/30 targets. The contribution is a framework for cumulative, recursive hypothesis research and bounded evidence for its constituent operations.

## 1. Introduction

A useful hypothesis abstracts from details. Even a simple prediction assumes that a selected representation contains enough information to determine an outcome. A conservation law, a separability assumption and a research heuristic make different commitments, but each proposes a regularity that may carry beyond the cases from which it was formed. Learning can improve an answer within that abstraction, or expose a reason to revise the abstraction itself.

A failed prediction does not identify which commitment failed. An instrument may have changed, a formerly negligible interaction may now matter, or a response may depend on a condition absent from the current observations. Refitting can be appropriate; so can acquiring a new measurement, restricting an applicability claim, or investigating a supporting assumption. Even an experiment that once resolved a question can lose its distinguishing power after a change in coordinates. Research therefore needs to change its questions and methods as well as its answers.

**Hegel Machine treats the conditions under which hypotheses are formed and tested as further objects of hypothesis research.** If a parent explanation depends on an uncertain instrument meaning, the instrument becomes a child question. Its answer returns to change the parent investigation. If the difficulty lies in a research procedure, that procedure becomes the object of revision. Acquired explanations and methods then change what later research can express and accomplish. This is a recursive organization of inquiry: supporting assumptions can themselves require support, while empirically settled subproblems can be reused.

The functional-analytic perspective is central to this formulation. A hypothesis can be viewed through its predicted responses to experiments, and an observation through the distinctions it can reveal among those predictions. Together they determine an observable geometry. Two explanations may differ internally yet coincide under every current test. Repeating those tests cannot expose the missing distinction; a different probe, measurement or representation may do so. The framework consequently pairs revision of the hypothesis space with revision of the observations used to investigate it. Section 3 makes this precise, including the conditions under which observations are continuous linear functionals.

The longer-term objective is to learn abstractions that explain both new evidence and the valid scope of earlier explanations. This involves expansion and simplification: an agent may introduce uncertainty when a deterministic description fails, or adopt a simpler effective model when the neglected detail makes no relevant difference. Across tasks, a proposed correspondence between structural roles must itself have testable consequences. A broader explanation should preserve earlier successes where warranted, while changing the assumptions responsible for failure.

Cumulative research requires separating **accepting a prediction, retaining a computation, and reusing it in a repair**. Suppose acquired components $A$ and $B$ support $C=A+0.7B$. When the event component changes to $B'$, $C$ fails as a current predictor, yet its executable structure can participate in

$$
D=C+0.7(B'-B)=A+0.7B'.
$$

It can also be useful when its original conditions return. This preservation mechanism serves the broader research process: neither retained code nor a successful earlier argument automatically certifies a new application.

Program libraries, belief revision, active experiment design and software repair supply important precedents. Our contribution is an explicit organization of hypotheses, observations and supporting research, together with interventions that test which acquired objects change subsequent behavior. We distinguish predictor reuse from method reuse, observable equivalence from representational sufficiency, and controller operations from choices authored by a language model.

The experiments follow that argument. We first examine child-to-parent evidence and the acquisition of missing distinctions, then the revision and transfer of research procedures, and finally preservation through failure and return. A constrained-optimization application tests whether learned priors change downstream decisions. Each study implements a bounded part of the framework; the complete common-model comparison also tests the gap between available mechanisms and autonomous use.

![Hypothesis spaces and observations coupled through recursive supporting research.](figures/revisable_knowledge_framework.png)

**Figure 1. Hypothesis spaces and recursive research.** a, Candidate explanations and available experiments jointly determine observable signatures. Equal signatures leave explanations indistinguishable under those tests; an informative added experiment can separate them. b, An unresolved premise opens a supporting investigation. Its evidence changes the parent investigation, whose revised consequences are tested in earlier valid conditions and new ones. Both explanations and research procedures accumulate and can be revised. Each reported implementation realizes a bounded subset of this organization.

## 2. Relation to earlier work

**Program libraries and repair.** DreamCoder jointly learns reusable program components and a search policy, showing how acquired abstractions make later synthesis more accessible (Ellis et al., 2021). GenProg uses executable tests to guide program repair (Le Goues et al., 2012). These are direct precedents for computational retention and test-guided revision. A new callable name is not by itself new semantic knowledge: its body may already be expressible in the base language. Here its value is measured by later predictions and research costs, including interventions that remove the acquired object.

**Rules and experiments.** Piriyakulkij et al. (2024) combine rule revision with active experiments. POPPER generates subordinate falsification tests for supplied hypotheses and provides a sequential statistical testing framework (Huang et al., 2025). Our finite confirmation checks serve a different operational purpose: a selected executable candidate must predict a newly acquired trajectory. The experiments examine how a supporting answer changes a parent investigation and how a failed research method is revised. Formal sequential error control is outside the present implementation.

**Experience and reasoning procedures.** Reflexion stores textual feedback that informs later attempts (Shinn et al., 2023). SELF-DISCOVER composes reasoning structures from available modules (Zhou et al., 2024). These motivate comparisons between text, executable procedures and ordinary code investigation. In our method-learning experiments, a model edits a Python action-selection program; frozen consumers then run it with shared numerical operators. The executable change and its subsequent effect can therefore be inspected separately from the author's explanation.

**Belief and dependency revision.** AGM theory distinguishes contraction and revision of commitments (Alchourrón et al., 1985), while assumption-based truth maintenance tracks the premises supporting conclusions (de Kleer, 1986). We distinguish these support relations from computational calls. A program can remain executable after the evidence for applying it changes. Support is associated with a scope, binding and version, and revised parent predictions require their own empirical test.

**Table 1. Mechanism-level positioning.** The final column states the additional question tested here, rather than a claim that the preceding work lacks every related capability.

| Precedent | Established mechanism | Question examined here |
|---|---|---|
| Program library learning and repair | Reusable code; test-guided modification | Does an observed-failing response remain useful for repair and return? |
| Active rule learning and POPPER | Hypothesis revision; informative or subordinate tests | Does a supporting answer change the unexecuted parent experiment? |
| Reflexion and SELF-DISCOVER | Reusable experience; composed reasoning procedures | Does a changed executable research method improve frozen downstream execution? |
| Belief revision and truth maintenance | Context-dependent commitments and support | Which computational resources remain usable when a predictive commitment is withdrawn? |

These mechanisms can also be implemented in an ordinary executable archive. The current evidence isolates several knowledge-availability effects, but does not isolate the entire dependency organization against one unified, equally capable archive. Section 6 states this remaining attribution gap.

## 3. Hypothesis spaces, observations and recursive research

### 3.1 A scoped abstraction

Let an experiment $e$ specify controlled inputs, a trajectory length and requested measurements. A predictive hypothesis is $h=(\phi,p,s,\Gamma)$: a representation $\phi$, executable program $p$, applicability scope $s$, and supporting premises $\Gamma$. For a deterministic response it predicts $\hat y_n=p(\phi(x_{\leq n}))$; an uncertainty model instead specifies conditional predictive distributions. The same statement may be a hypothesis under investigation, a working assumption in another inference, or a question reopened by a counterexample.

The research state contains representations, an executable language, experiment constructors, procedures, knowledge, evidence and remaining resources:

$$
\mathcal R=(\Phi,\mathcal L,\mathcal E,\Pi,K,D,b).
$$

Learned definitions extend the effective language available to bounded search; learned procedures change how that search proceeds. The environment interface, initial primitive families, resource ceilings and evaluation loss are supplied by the protocol. Acquired coefficients, coordinate bindings, compositions, empirical scopes and executable method edits are the learned objects. The controlled implementations do not invent unrestricted new sensors or arbitrary mathematical primitives.

We distinguish three objects by what they change. A predictive program $p$ maps represented inputs to predicted outcomes. A research procedure $\pi(D,K)$ selects research actions from acquired evidence and knowledge. A meta-prior $\eta$ biases which candidate structures or research directions are generated, ranked or reused. A learned $p$ can serve as a prior feature in a downstream optimizer, as in Section 5.4; an authored revision of $\pi$ changes executable research control flow, as in Section 5.2. Preferences among existing methods can also change with accumulated returns: in Appendix E.1, retaining return records changes method selection in 12/12 cases, although neither arm completes the subsequent demands or production tasks. This is evidence that research preferences respond to experience, with no demonstrated completion benefit in that comparison.

A procedure is itself a hypothesis about useful action selection. Its empirical consequences include which tasks it resolves, what it wrongly confirms, which earlier abilities survive, and what resources it consumes. Testing a research preference likewise requires tracing its effect on subsequent choices and outcomes. Table B2 identifies the controller, learned object and fixed interface for each implemented operation.

### 3.2 A functional-analytic view: what can an observation distinguish?

Fix a nonempty domain of legal physical experiments $\mathcal E$. A hypothesis $h$ determines a response function $f_h$ on that domain. More generally, an experiment and a statistic of its outcome define an observation map $F_{e,\psi}(h)$: for example, the predicted reading, mean response or variability under a controlled trajectory. A finite family $Q$ of such maps gives the observable signature $S_Q(h)=(F(h))_{F\in Q}$. Hypotheses with identical signatures are observationally equivalent under $Q$. Here $Q$ may describe observations already acquired or a planned test family; it is distinct from the entire legal experiment domain.

A classical linear instance clarifies the geometry. Let $V$ be a real linear subspace of the bounded scalar response functions on $\mathcal E$, equipped with the supremum norm. The embedded hypotheses need not themselves form a linear space. Evaluation at an experiment, $\delta_e(f)=f(e)$, is a continuous linear functional because $|\delta_e(f)|\leq\|f\|_\infty$. For a nonempty finite family $Q\subset V^*$ of continuous linear observations, define

$$
p_Q(f)=\max_{q\in Q}|q(f)|,\qquad
K_Q=\bigcap_{q\in Q}\ker q.
$$

The seminorm $p_Q$ measures disagreement visible to these observations. Its kernel $K_Q$ contains the directions they cannot detect. Consequently, $d_Q(f,g)=p_Q(f-g)$ is a pseudometric on $V$, and $\|f+K_Q\|_Q=p_Q(f)$ is a norm on the observable quotient $V/K_Q$. This is the norm induced by $p_Q$, not necessarily the quotient norm inherited from the original supremum norm. Seminorms, continuous duals and quotient spaces are standard functional-analysis constructions (Melrose, 2020; Zsák, 2023). Here they provide an organizing formulation: observation-induced indistinguishability becomes explicit, making it possible to ask which supporting answer would change the distinctions available to an investigation.

Adding observations $Q\subseteq Q'$ shrinks the invisible subspace: $K_{Q'}\subseteq K_Q$. It reveals a previously missing distinction exactly when an added functional is nonzero on some old invisible difference. For example, on a bounded control domain the responses $f(z)=z^2$ and $g(z)=|z|$ agree at $z=-1$, whereas at $z=-0.6$ they predict $0.36$ and $0.6$. A new probe can expose their difference without changing either response law. Figure 2 records this situation in an executed investigation.

General experiment maps, conditional variances and research success criteria need not be linear functionals. Their signatures still define observational equivalence, and pulling back a metric on those signatures gives a pseudometric on hypotheses. Appendix A gives the linear argument and a distributional version. The numerical controllers use entropy reduction, disagreement scores and task losses for action selection; no common norm is assumed for all of them.

This formulation separates two research failures. An **observation limitation** leaves competing explanations indistinguishable under current tests. A **representation limitation** prevents the predictor from expressing a relevant difference in its inputs: if $\phi(x_1)=\phi(x_2)$ but $y_1\ne y_2$, every deterministic predictor of $\phi$ incurs average squared error at least $(y_1-y_2)^2/4$ on that pair. More fitting in the same representation cannot remove this error. A new measurement may resolve it; uncertainty modeling may instead be appropriate when the residual variation persists. Choosing between these possibilities is itself an empirical question.

### 3.3 Revising the space, the observations and the scope

The paired view suggests three coupled operations: change candidate explanations, change the observations that discriminate them, or change the representation and its supporting interpretation. An instrument or coordinate hypothesis specifies how a physical intervention becomes an input to a response law. Revising that binding can therefore change which experiment is informative, even when the candidate laws stay fixed. Comparisons across bindings must refer to common physical experiments; an arbitrary change of coordinates must not count as a new empirical success. The terminal evaluation contract remains fixed while the research procedure changes.

Model expansion and simplification fit the same picture. Expansion introduces a missing variable, interaction, condition or uncertainty family. Simplification removes distinctions irrelevant to the current question while retaining their valid range. A proposed generalization may take the form $p_{\mathrm{new}}(x,c)=p_{\mathrm{old}}(x)$ on supported conditions $c\in s$, with a revised law elsewhere. Both preservation and new predictions require testing. The aim is a wider explanation of why an earlier model worked and where it fails, rather than an ever-growing list of exceptions. The implemented conditional branches and residual compositions provide finite instances; general discovery of such explanations remains a research objective.

Structural transfer adds another hypothesis: a mapping from the roles in one task to those in another preserves the relevant relation. Matching words or numerical coordinates alone cannot establish it. A proposed binding should predict the outcome of a target intervention. These correspondences are typed: a world constraint restricts possible outcomes, a complexity preference directs search, and an evidence rule governs support. They require different tests and have different consequences when revised. The present experiments test renamed, permuted and recalibrated controls, component changes and old-condition returns within supplied task families.

There are direct implementations of the paired view. The online researcher groups candidates by finite response signatures, selects experiments by predicted reduction of uncertainty, and rewrites a retained probe after supporting calibration. The growth researcher purchases candidate measurement channels and compares the representations on common acquired records. It also fits conditional variance and compares simpler response programs. These are finite operational instances of changing observable distinctions and effective model classes. Appendix B.2 maps them to the actual study sources; the framework does not require a generic symbolic implementation of a continuous dual space.

### 3.4 Computational dependence and empirical support

Let $E_{\mathrm{call}}$ contain edges $p\to q$ when $p$ executes a call to $q$. Let $E_{\mathrm{support}}$ contain edges from a claim to the assumptions whose support it needs. The graphs answer different questions: what computation runs, and what warrants its application. A fit can be executable while one of its instrument assumptions is unresolved.

The relation $\operatorname{Support}(p;s,D,v)$ records empirical support for program $p$ under scope $s$, evidence $D$ and version/binding $v$. This is a relation to particular evidence, not an assertion of universal truth. Implementations realize it through acquired records, dependencies, case-specific practices and confirmation outcomes. The notation makes explicit why support for an earlier source cannot silently certify an edited source or a new coordinate binding.

**Table 2. Research-state transitions.** Retention and predictive acceptance are separate decisions.

| Event | Predictive/support consequence | Computational consequence |
|---|---|---|
| A current application is refuted | Withdraw or restrict its current support; retain the counterexample | Keep the program available for testing, composition or repair |
| A supporting instrument is unresolved | Suspend the dependent claim; investigate the instrument | Preserve unaffected definitions and alternative bindings |
| Code, scope or binding changes | Reassess the affected support relation and dependent claim | Create or select the revised executable candidate |
| A frozen replacement passes a new test | Record support for that candidate in the tested conditions | Make the confirmed result available to later tasks |
| Earlier conditions return | Compare retained predictions on current evidence | Call the old program or replay a method if supported |

The first transition is the retention experiment's intervention. The instrument transition is tested by the online support experiment. Method-edit practice tests the version transition. They share a framework but use distinct controllers and protocols.

### 3.5 A research loop with supporting questions

The following algorithm describes the common organization. Each study supplies concrete candidate generators and stopping rules; it is not a claim that one implementation executes every operation.

```text
Input: public laboratory, initial operators, acquired knowledge K,
       paid evidence D, resource and research-round ceilings
Output: a tested predictor, or the incumbent and unresolved questions;
        updated knowledge, counterexamples and resource ledger

1. Compare current fits and available retained programs on paid evidence.
2. While an eligible research action remains within its ceilings:
   a. Locate the unresolved distinction: response, representation,
      applicability condition, supporting assumption, or method.
   b. If a parent decision needs an unresolved supporting answer,
      open that child question and run its investigation.
      Return its measured answer to revise the pending parent action.
   c. Construct or adapt candidates using the shared operators.
      If current observations cannot resolve a relevant distinction,
      propose a new probe, measurement or supported representation.
      Retain useful earlier computations independently of acceptance.
   d. Choose a distinguishing experiment or a proposed confirmation;
      acquire and charge its observations and measurements.
   e. If a frozen candidate passes fresh confirmation, record the
      candidate-specific support and return the response.
   f. Otherwise retain the counterexample, revise the affected claim,
      and reopen a direction when new evidence makes it informative.
3. Stop when supported completion, a research/step limit, absence of an
   informative permitted action, or a resource limit ends the visit.
   Preserve unresolved status rather than promote a merely fitted answer.
```

The numerical implementations include sparse and composed response fits, relation-boundary probes, instrument calibration, bounded temporal alternatives and retained definition calls. Some older nested-relation paths fit components first and then compile an answered dependency graph. Those graphs describe a decomposition of completed computation. The online laboratory instead opens a child before its decisive observation and consumes the answer before choosing the parent's next experiment.

Figure 2 shows the distinction. The parent needs a coordinate mapping; a child investigates sensor calibration. Returned calibration and coordinate evidence identifies the second control as the driver. The retained probe $(-0.6,-1)$ then aliases quadratic and absolute-value responses, both predicting 1. The method changes the probe to $(-1,-0.6)$ before acquiring the response, separating all four public alternatives. A deterministic intervention changing only the returned calibration evidence changes the parent interpretation and probe; fresh confirmation rejects the resulting incorrect branch.

![Supporting questions and evidence change the pending parent experiment.](figures/online_support_inquiry.png)

**Figure 2. An executed child-to-parent update.** Chain 79101013 opens supporting questions before their resolving observations. Returned evidence changes the response experiment before execution. The candidate predictions use the recorded language and acquired binding. This trace demonstrates the mechanism; matched-control performance is reported separately in Section 5.1.

For learned Python methods, the model edits `select_actions`, while the host supplies action execution, paid practice and publication checks. Frozen consumers execute the resulting program with the same numerical operators. Automatic practice after an edit and host fallback to an older source are explicit experimental conditions, not actions attributed to the model. In the autonomous recall intervention, host fallback is disabled and restoring a source must be selected by the model.

## 4. Evaluation design and resource interpretation

We report interventions within fixed cohorts. An apparatus or complete source chain is the statistical unit where its tasks share learned knowledge. Repeated author attempts on one state are not independent environments, and running two programs on the same targets does not double the number of target contracts. Terminal evaluation follows research and is not returned for model selection. Fresh confirmation is a research-time check after freezing the candidate; threshold crossing and independently correct prediction are reported separately.

**Table 3. Evidence map.** All numerical studies are offline except the explicitly model-authored phases. Table B2 separates who initiates each operation, what changes and what remains fixed. The supplement and artifact index identify each frozen protocol and retained failure record.

| Evidence package | Controller and learned object | Independent unit and status | Question answered |
|---|---|---|---|
| Observation and model-class revision | Fixed growth researcher; acquired view and response/variance model | 30 apparatus sequences with paired access and model controls | Value of missing measurements; expansion and simplification |
| Retention after refutation | Fixed numerical researcher; acquired response library | 30 new apparatus sequences under a fixed protocol | Effect of retained computation; conditional marking ablation |
| Procedure availability | Fixed growth researcher; acquired research steps | 30 shared-prefix paired successors | Marginal value of the procedure book |
| Method revision | LLM edits; generic Python frozen execution | Separate 18-, 32- and 56-target cohorts after source development | Effect of each frozen control-flow revision |
| Online support | Fixed recursive and flat controllers | 30 new source chains; a separate 30-pair temporal cohort | Child-to-parent operation; matched flat performance |
| Complete agents | Same model, tools and allowances | 30 source–target chains per arm | Overall completion and predictive outcomes |
| Autonomous recall | LLM selects or edits archived methods | 20 attempts on one regressed state | Local selection/publication effect of supported history |
| Executable hypothesis delegation | LLM proposes a basis; fixed researcher tests and revises it | Six continuations of one paid development failure | Adoption under two research assignments; physical completion |
| SC-KG application | Numerical prior search and a frozen optimizer | Nine historical cases; retrospective matched rerun | Downstream decision effect under the historical contract |

The object studies evaluate four independent 64-row trajectories per task, using terminal MSE below $10^{-6}$ and research confirmation below $10^{-7}$. The initial task allowance is 384 observations, 4,608 measurement units, 360,000 charged computation units and six research rounds. Retention continuations use separate one-task sessions in every arm. The online finite-language study permits 24 observations and uses MSE below $10^{-10}$ for both fresh and terminal tests. Temporal and growth studies retain their own stated thresholds and ceilings in the supplement; counts from these contracts are not pooled.

Resource use is a vector: acquired response observations, requested measurement units, charged program computation, measured CPU or execution time, and model calls/tokens where applicable. Charged computation is an implementation-specific ledger, not FLOPs or elapsed time. Candidate evaluation, fitting and screening are charged according to the frozen runner's operations; Python orchestration, lookup and some host work are represented only in timing. For example, response fitting charges candidate-row and fitted-column work. A learned call actually executes its body, but its main scoring charge is a candidate-row pass, not the body's expanded primitive count. Scoring recalled bindings is charged; pure library lookup and general host bookkeeping are not uniformly priced. Growth studies separately report terminal primitive steps. The precise contracts and code locations are given in Appendix B.1. Research costs and terminal evaluation costs remain separate, and unsuccessful attempts remain in total expenditures.

A transfer saving does not imply that learning has already paid for itself. For a resource with acquisition cost $C$ and comparable positive per-task saving $\Delta c$, a break-even count would be $C/\Delta c$. This calculation is meaningful only within the same resource and workload. We do not trade model tokens against observation rows or infer end-to-end runtime from charged operations. The shared-prefix interventions include the same acquisition cost on both sides; their absolute downstream differences survive that inclusion. In the selective-method study, both programs solve 56/56 targets. The two relevant author phases, including the unsuccessful predecessor and new counterexample, cost 118,797 charged units; mean subsequent saving is 7,407.21 per target. Repeating that fixed task mixture would cover these incremental charged units after about 17 targets. Observation savings are negative, measurement savings are zero, and transfer uses no model tokens in either arm. This conditional projection therefore supplies no overall or full-history break-even claim.

## 5. Results

### 5.1 Restoring distinctions and returning supporting evidence

**Online support and strong flat controls.** In 30 source chains, three researchers receive the same acquired state for four independent continuations: same or changed response structure crossed with same or changed surface. Surface changes rename and permute controls and alter calibration. All four response primitives remain public. Recursive, flat active and frozen-method-with-fallback research each solve 120/120 targets, with no false confirmation. The recursive method rewrites 47 experiments before execution and uses retained definitions on the 60 same-structure targets, as do both controls.

The resource tradeoff is mixed. Recursive and flat research use 852 versus 647 observations, and 388,496 versus 735,048 charged computations. The former buys more evidence while scoring fewer candidate-action combinations. In a separate temporal comparison, matched instrument candidates, response fits, probes and ceilings give 60/60 correct targets for both separately confirmed child research and a flat controller. For recurrent cases, flat uses 159 rather than 163 observations and 45 rather than 57 measurement units per target, but fits three instrument-conditioned responses rather than one. These are strong local null results for an accuracy advantage from explicit child organization.

**Measurement and model-class revision.** In a separate joint-growth intervention, access to an informative measurement channel gives correct terminal predictions in 30/30 cases, versus 0/30 for controls alone. The latter still produce 30 internal confirmations of stochastic descriptions, with mean terminal MSE $0.1442$. Thus a researcher can confirm an uncertainty description while missing an available explanatory variable. The added channel costs a mean 47.2 observations, 240 measurement units and 50,656 charged computation units per target. This contrast tests the value of observable information, rather than a different fit to identical inputs.

Scope revision provides a related test: learned conditions yield 29/30 terminal successes versus 0/30 for the frozen old response. In the 27 cases with an identifiable conditional branch, applying the new branch everywhere also fails every case. The successful explanations need both the earlier behavior and its revised scope. One learned scope passes internal confirmation but fails terminal evaluation.

The same campaign also exercises expansion and simplification. On the stochastic task, all 30 runs introduce a variance model and 19 select a conditional one; variance mean absolute error falls from $0.01461$ for the constant model to $0.01103$. When the task's true response becomes linear, both simpler and dense models solve 30/30, but terminal prediction uses 69,120 versus 194,560 primitive steps. All 30 subsequent nonlinear returns restore a supported, correct response. The latter comparison concerns execution of the final predictors, not total research cost. Interval calibration and the limits of uncertainty identification are reported in Section 6 and Appendix H.3.7.

**Complete common-model comparison.** Ordinary Code/ReAct, textual memory and Hegel share a 24-observation, 16-call and 300,000-reported-token allowance per task, the same public laboratory and the requested provider model alias `gpt-6-luna` at medium reasoning. Token limits are checked before the next call and can be exceeded by that call. Model identity here is the logged request alias, rather than an independently verified backend architecture.

**Table 4. Complete agent outcomes.** Each arm contains thirty registered source–target chains. Service and budget failures remain in the denominator.

| Knowledge condition | Source success | Target success |
|---|---:|---:|
| Ordinary Code/ReAct | 29/30 | 29/30 |
| Textual memory | 30/30 | 29/30 |
| Hegel executable research knowledge | 26/30 | 26/30 |

Every submitted predictor passes independent terminal evaluation. Failures involve service interruption, token exhaustion, or consuming the observations required for confirmation. Hegel makes 36 inherited experimental-program calls, including 15 with unchanged source, but completes fewer tasks overall. The experiment uses 894 model calls and at least 20,395,181 reported tokens; 70.3% of known input tokens are cached. Table 4 is the direct answer to whether the mechanisms improve a complete agent on this task family: they do not.

The available four-primitive environment is nearly saturated by ordinary code investigation. Harder temporal development tasks expose different failures: an agent can buy an informative instrument observation without analyzing it or revising the parent. Executable post-acquisition analysis repairs this in individual traces, while persistent-advice comparisons still tie at 4/6 attempted targets in each condition. These records are retained in Appendix F rather than combined with the 30-chain cohort.

### 5.2 Research procedures can be revised and transferred

**Procedure availability.** In the joint-growth study, deleting only the acquired procedure book while retaining the same predictive program library leaves accuracy at 30/30 in both conditions. Keeping the procedures reduces successor observations from 5,088 to 4,028 and charged computation from 5,219,559 to 3,944,154; both use 7,200 measurement units. The paired mean observation difference is $-35.33$, interval $[-45.47,-25.20]$; the computation difference is $-42{,}514$, interval $[-49{,}246,-35{,}687]$. Reused steps generate fitting questions for the current instrument rather than copy its earlier index. This intervention measures knowledge about investigation separately from the availability of response programs.

Across the full eight-task sequence, persistent research passes both fresh confirmation and independent evaluation on 232/240 tasks versus 215/240 for equally expressive task-local research. Charged computation falls 22.42%, while CPU rises 17.92%. These results provide context for the procedure intervention; they do not isolate every revised object at once (Appendix D).

**Method revision.** The LLM author receives an actual failure, edits a Python procedure, and practices it on source conditions. The published source is frozen before evaluation on other archives. Consumers use the ordinary `code_react` Python executor and the same instrument, response-fitting, experiment and confirmation operators. There are no target-time model calls. Consequently, these experiments test learned control flow in a generic executable archive, not an additional effect of Hegel-specific runtime gates.

**Table 5. Separate method-revision cohorts.** Each row has a different protocol and target set. They are not stages of a single success curve or a pooled estimate.

| Failed procedural assumption | Executed revision | Frozen target result |
|---|---|---|
| A supported instrument suffices for the response | Preserve the instrument; refit and test the parent response | Old 10/18; revised 18/18 |
| All response records share the current mechanism | Fit on current instrument-supported records | Old 20/32; revised 32/32 |
| New-case repair preserves prior abilities | Test old prediction, then retain or reopen its support | Initial 36/44; corrected and refit-both each 56/56 |

The first author makes one edit and three source practices in five calls. All eight failed response-only transfers are repaired, and all ten earlier successes survive. Transfer observations fall from 2,928 to 2,544 and measurements remain 3,888, while charged computation rises from 183,048 to 200,712. The later selective revision illustrates why preservation needs testing: its initial program repairs twelve new cases but breaks eight older ones. The corrected method preserves all 56 target outcomes; relative to refit-both it reduces charged computation by 35.48%, leaves measurements unchanged and adds 384 observations. Appendix F retains the unsuccessful author phases and acquisition costs.

**Recall and new frozen transfer.** A separate intervention resumes one regressed method state in 20 author attempts, disabling host fallback. With supported history, all ten authors select the old source and eight publish through the six-case practice gate; without it, two of ten publish. Seven publications are identical to the earlier supported source. The three other publications contain changed code. These are repeated decisions on one state, and the conditions have different future confirmation streams because their available practice histories differ. The contrast establishes local selection and completion value, not a rate of discovery across independent tasks.

We deduplicate the ten published sources by literal equality and freeze all four distinct programs before a new, zero-model-call transfer study. Each program and an existing fixed repair procedure receive the same 24 targets: three archived donor states crossed with eight conditions covering response changes, instrument changes, their combination, delayed changes and old-condition return. One donor participated in source authoring; two did not. New trajectories and intervention combinations extend the evaluation within the same finite families. All four published programs pass fresh confirmation and terminal prediction on 24/24 targets with zero false confirmation. The fixed repair procedure passes 21/24; all three failures follow a late response change. Thus the three new sources now have downstream evidence, while none improves accuracy over the unchanged successful program. The published programs use 277,728–280,032 charged units and 33.65–35.24 CPU seconds, versus 115,893 and 18.23 for the less successful fixed repair. The five executions per target are paired measurements, not 120 independent tasks. Appendix F.12 gives the complete protocol and cost vectors.

The original recall study incurs 123 model calls and 5,405,786 reported tokens, including failed authors. Its 8/10 versus 2/10 publication result and the new frozen-target measurements answer different questions. Source equality prevents the seven repeated publications from being counted as seven independently acquired methods. The actual common executor also makes clear which gains an ordinary archive can already obtain.

**Executable hypothesis delegation after refutation.** A local crystal-control fixture adds a model-authored structural proposal followed by physical counterexamples and numerical revision. Six Hegel continuations share one complete failed state: 184 acquired observations, nine inherited declarations and the same prior refutations. With only the research assignment changed, an open request for an explanation yields 0/3 completions; a historical assignment explicitly asking for an executable basis submitted to the existing fitter and controller yields 2/3. Both successful authors propose signal, squared-signal and cubed-signal terms. Their first fitted prediction, 7,715.405 Hz, is refuted. The fixed researcher then selects the cubic-only support and jointly revises its coefficients and connected activation band. It requests 7,905 Hz; the simulated reactor activates at 7,906 Hz during adjustment, completing the task. Each success uses one model call: the model supplies the structural proposal, while the subsequent counterexample-driven revision is performed by the existing research program.

Across all three attempts per assignment, open versus executable delegation uses 30 versus 11 model calls, 120 versus 225 world actions and 106,084 versus 933,930 research computation units. Four unsuccessful authors voluntarily stop. In a separate zero-model-call replay of one recorded proposal, reusing already established exclusions of old families preserves completion while reducing new actions from 99 to 35 and research computation from 466,965 to 420,819. These results locate both an adoption bottleneck and a concrete use of retained negative knowledge. The requested 7,905-Hz point remains formally unconfirmed. Appendix F.13 gives the protocols, historical controls and limitations of these operator-started repetitions on one development state.

### 5.3 An observed-failing program can remain useful

Each apparatus first learns a conserved response $A$, an event response $B_r$ and their composition $C=A+0.7B_r$. One paid 24-observation challenge tests the frozen acquired $C$ after the event boundary changes. All 30 challenges refute it. Before any replacement is learned, three continuations branch from the same prefix: conditional retention, retention without the challenge's conditional feedback, and removal of $C$, its 17 dependent abstraction/definition entries and its meta-prior. Ancestors $A$ and $B_r$, generic methods and other research experience remain. All arms may subsequently learn and confirm a replacement.

The three successor tasks are the changed mechanism, the original-condition return, and the changed-condition return. The intervention therefore concerns **90 continuation tasks per arm**; another 90 prefix tasks are shared. The prespecified primary contrast is the original-condition return, not the first changed task.

**Table 6. Retention after refutation.** Terminal accuracy by phase, followed by complete-sequence charged resources. Shared prefix and challenge costs are included in each arm's sequence totals. Conditional and unmarked retention have identical terminal errors and charged costs in every paired continuation.

| Outcome | Conditional retention | Unmarked retention | Global discard |
|---|---:|---:|---:|
| First changed task | 30/30 | 30/30 | 10/30 |
| Original-condition return (primary) | 30/30 | 30/30 | 30/30 |
| Changed-condition return | 30/30 | 30/30 | 29/30 |
| All continuation tasks | 90/90 | 90/90 | 69/90 |
| Full sequence, including shared prefix | 180/180 | 180/180 | 159/180 |
| Research observations | 17,987 | 17,987 | 22,484 |
| Measurement units | 73,680 | 73,680 | 120,864 |
| Charged computation | 8,553,189 | 8,553,189 | 11,651,123 |

All three arms succeed on the original return, but retained responses use 24 observations per return versus 40 after deletion. Return computation totals are 30,930 versus 138,476; the paired mean difference is $-3{,}584.9$ units, with a 95% apparatus-bootstrap interval of $[-3{,}868.0,-3{,}273.9]$. Thus the primary endpoint supports easier recovery at equal accuracy. The accuracy difference occurs mainly on the first revision. Of its 20 deletion failures, 16 remain unresolved and four pass internal confirmation but fail terminal evaluation. These failures retain substantial numerical budgets; the stopping analysis below examines their actual control-flow cause.

A replay of all twenty first-revision failures with the original frozen runtime reproduces their statuses, costs, confirmation losses and terminal errors. The sixteen unresolved cases stop at `relation_ambiguous`: fourteen after the first confirmation attempt and two after the second, below the six-round limit. Rival relation candidates still agree with the short confirmation. Fifteen runs fail to generate a predicted distinguishing probe; one exhausts a local 6,000-unit probe-generation cap while retaining global resources. The four false confirmations instead terminate normally after accepting a candidate that later fails terminal evaluation. No session revisit occurs. Thus the loss is mediated by the available relation search and stopping policy, rather than depletion of the overall allowance. Another random acquisition and candidate comparison would fit the remaining budgets, but its ability to repair these cases has not been tested. Appendix H.3.4 records the replay.

Across the complete sequence, retention reduces observations by 20.00%, measurements by 39.04% and charged computation by 26.59%. The computation difference per sequence is $-103{,}264.5$, interval $[-132{,}874.2,-78{,}700.7]$. Conditional-retention CPU time is nearly equal to deletion: the paired difference is $-0.62$ seconds, interval $[-12.93,10.79]$. These intervals resample complete apparatuses, stratified by the two generator profiles, with 10,000 paired resamples.

![Successor research costs separated by the changed task and the two return conditions.](figures/scoped_retention_comparison.png)

**Figure 3. Where retention changes research expenditure.** Each panel shows the three successor stages across 30 apparatuses. The shared acquisition prefix and common challenge are excluded from these bars and included in Table 6. Retention and unmarked retention coincide. The primary original-return accuracy is tied; its observations and charged computation decrease.

The conditional marking adds no measured benefit over unmarked retention. Moreover, unmarked retention uses the same researcher and library machinery; it is not a general archive baseline with the entire support organization removed. The result identifies a benefit from retaining executable knowledge under this controller, not a unique benefit from attaching conditional semantics.

Three checks distinguish inheritance from a textual or syntactic trace. We inspect exact called definitions, test the repaired response on independent trajectories, and intervene on availability. A direct call proves that code was executed; an algebraically expanded equivalent could preserve the same meaning without that call. In a separate component-change study, a gain change replays a method using $A$ and $B$ without calling $C$, whereas a boundary change retains $C$ inside a residual correction. Both and the unchanged condition solve all 60 successor/return tasks in each branch. These observations establish usable preservation without treating a particular syntax as indispensable.

The complete four-algorithm composition study is supporting evidence: full research solves 180/180 versus 179/180 for task-local research, with 76.25% less charged computation. Memory and unused sequence-resource allocation both change in that contrast, so it is a complete-system comparison. The dense fixed-basis control's 0/180 result is an implementation diagnostic: it fits 185 coefficients from 48 observations and even misses a representable linear task. It is not the principal competitive baseline. Appendix H.3 gives the full comparisons and failures.

### 5.4 Downstream application: learned priors for constrained optimization

SC-KG supplies a decision-level application. A learned numerical program becomes a GP feature: objective and constraint surrogates can use $(1,q)$ and $(1,p)$ instead of a fixed coordinate-quadratic basis. Residual discrepancies can reopen research. This is bounded numerical program search, using a three-dimensional optimization domain, rather than a language-model agent or a test of the original very-high-dimensional motivation.

We rerun nine historical cases against the same historical SC-KG backend and 32-observation prefix, followed by 32 optimization calls. The shared prefix was originally selected by the Hegel researcher. The controls are native SC-KG and its full configuration combining low-frequency structure, orthogonality, sparsity and additivity. The original model-selection gate is retained. Frozen Hegel v0.53 research supplies the third system.

**Table 7. Historical common-prefix SC-KG comparison.** Lower terminal regret is better. Every final recommendation is feasible. The paired cases were already used in the historical Hegel evaluation, so this is a retrospective comparator addition.

| Optimizer prior system | Mean terminal regret | Feasible recommendations | Hegel paired wins |
|---|---:|---:|---:|
| Native SC-KG | 0.100000 | 9/9 | 9/9 |
| Historical full four-property configuration | 0.129478 | 9/9 | 9/9 |
| Frozen Hegel v0.53 | 0.017944 | 9/9 | — |

Native and Hegel exactly reproduce their historical endpoints. The full control selects its learned coordinate model at both recorded initial and final gates in every case; enabling four properties does not mean that all four were deployed. The three mean regrets cover stable conditions, a moving optimum and joint center and objective-interaction change. Optimizer wall times are 24.40, 624.82 and 26.56 seconds respectively; the full control adds 8.45 seconds of preparation. Hegel's historical prefix-research cost is not included in these rerun times, so they are not end-to-end learning-time comparisons.

Appendix G preserves two important boundaries. An earlier same-paid-data feature comparison found no independent gain from the learned prior, and a later component revision improved calibration without changing recommendations. The current nine-case win is a complete-system result on a low-dimensional, shared-prefix contract. It does not isolate recursion, or reproduce the full prior's original independent-source high-dimensional training setting.

## 6. Discussion and limitations

The framework connects abstraction to the experiments that give it empirical meaning. A hypothesis specifies which differences matter; an observation determines which of those differences can be exposed; recursive supporting research revises the connection when it is uncertain. The experiments instantiate parts of this cycle: calibration changes an unexecuted probe, measurement access resolves missing information, and procedure revision changes what later research does. Retention makes these revisions cumulative by preserving earlier computations without preserving their refuted applications.

This perspective also connects several familiar research settings. For a reinforcement-learning agent, a state representation and transition model are hypotheses about what suffices to predict consequences. Robust modeling expands the considered family of consequences; simplifying assumptions restrict it. Causal inquiry asks which interventions distinguish competing explanations, and representation learning changes the variables in which those explanations are expressed. These are connections within the proposed framework, rather than additional evaluated domains. Appendix H.1 develops them and the philosophical motivation: a more general explanation can preserve the insight of earlier ones by identifying their valid conditions.

**Attribution.** The retention intervention identifies value in executable knowledge availability. Conditional marking ties unmarked retention; strong flat research ties explicit recursion in accuracy; and frozen Python method transfer already runs in an ordinary code archive. A unified same-history comparison with revalidation, rollback, residual repair and identical tools, differing only in support and recursive organization, remains absent. The present result is therefore a framework with demonstrated mechanisms, not proof that its organizational vocabulary is uniquely necessary.

**Finite discovery and transfer.** Operators, candidate families and measurement interfaces are supplied. The functional-analytic construction organizes the hypothesis–observation relation; it does not replace the study-specific search procedures or provide a general discovery guarantee. The implementation does not learn arbitrary infinite-dimensional spaces, new sensor primitives or a general topology from data. Some dependency graphs are compiled after fitting, and only the online experiments establish pre-action support return. New parameterized programs and changed action-selection code extend accessible research within those interfaces. A useful abstraction can be expressible in the old primitive language while changing what bounded research can accomplish. Learned models used as prior features, revised procedures and experience-dependent method preferences do not by themselves establish invention of a new cross-task structural principle. Unrestricted experimental-primitive invention, cross-domain abstraction discovery and general recursive self-improvement remain open. The recall cohort has one authoring state; additional target executions do not turn it into multiple independent method discoveries. The executable-delegation study also reuses one previously studied state: its two rediscoveries and model-free execution repair establish local adoption and completion, while automatic initiation, independently confirmed response laws and unseen-task discovery remain untested by that comparison.

**Task-dependent simplification.** The simplification study changes the true response to a linear law and later restores the nonlinear law. It tests adaptation to a simpler response and recovery of an earlier model. It does not test coarse-graining one unchanged complex environment because a different task or scale makes some distinctions irrelevant. Learning that cheaper effective model together with an empirical boundary for its approximation would more directly address task-dependent idealization.

**Statistical support and confirmation.** Fresh trajectories separate candidate selection from that individual check, but thresholds are not calibrated significance levels. Repeated revision and testing have no formal sequential type-I error guarantee here. False confirmations in the deletion, passive and growth studies show that limited confirmation coverage matters. In particular, improved variance prediction does not establish calibrated uncertainty: nominal 95% intervals cover 91.09% of fresh stochastic responses, below the constant model's 92.81%. Controls-only confirmations also fail to detect missing explanatory measurements. The current uncertainty results do not establish robust decision guarantees. Bootstrap intervals concern the stated apparatus distribution; they cannot turn related tasks or repeated authors into independent samples. Method-development cohorts are reported separately from frozen transfer, and the historical SC-KG comparison is retrospective.

**Cost and autonomous control.** Charged computation omits host work and can fall while CPU rises. Learning and unsuccessful author attempts must be paid for before claiming long-run savings. No stable open-ended net-benefit claim follows from these sequences. Complete agents still fail to use available procedures or to finish valid confirmation, and the main common-model comparison favors the ordinary controls. Model-request aliases and service outcomes also limit extrapolation to other providers or architectures.

The next discriminating experiment is a prospective method-transplant comparison across genuinely different donor failures, retaining failed donors and holding initial archive contents, numerical operators, experiment permissions and budgets fixed. It should compare support-organized research, a strong ordinary executable archive, and a condition with frozen research methods but the same predictive knowledge. Each researcher may choose different experiments: equality concerns access and evaluation, not the observations it decides to acquire. Tasks should include surface changes that preserve structure and superficially familiar changes that invalidate it. The decisive trace links new supporting evidence to an actual change of research action and then to independent target outcomes or a measured resource tradeoff. That experiment would test the remaining organizational claim.

## 7. Conclusion

Hegel Machine treats predictive explanations, their supporting assumptions and the procedures used to investigate them as revisable hypotheses. Its functional-analytic perspective pairs a space of possible explanations with the observations that make their differences visible. Recursive research can revise either side: a child answer changes the parent's experiment, a new measurement enriches an inadequate representation, and a failed method becomes a candidate for revision and transfer. Controlled studies demonstrate finite instances and benefits from preserving earlier executable knowledge. General abstraction invention, a unique advantage of recursive organization and overall autonomous-agent superiority remain open. The resulting framework makes those broader claims concrete enough to test.

## Experimental materials

The [companion supplement](supplementary.pdf) retains complete protocols, intermediate failures, detailed costs and the mathematical derivations. The [artifact index](ARTIFACT_INDEX.md) maps each study to its saved source snapshot, protocol, results and analysis commands, and distinguishes numerical replay from reauthoring with a language model. The distribution includes the English Markdown, editable TeX, PDFs and a selected evidence archive. Tables 4–7 are regenerated directly from the included result JSON during manuscript construction, without model calls. The archive also supplies a runnable frozen numerical study and the four published programs with their 24 transfer targets. Replay dependencies for the other studies are identified in the index. Public archival deposition and an immutable release identifier remain to be supplied before external publication.

## References

Alchourrón, C. E., Gärdenfors, P., and Makinson, D. (1985). On the logic of theory change: Partial meet contraction and revision functions. *The Journal of Symbolic Logic*, 50(2), 510–530. [doi:10.2307/2274239](https://doi.org/10.2307/2274239).

de Kleer, J. (1986). An assumption-based TMS. *Artificial Intelligence*, 28(2), 127–162. [doi:10.1016/0004-3702(86)90080-9](https://doi.org/10.1016/0004-3702(86)90080-9).

Ellis, K., Wong, C., Nye, M., Sablé-Meyer, M., Morales, L., Hewitt, L., Cary, L., Solar-Lezama, A., and Tenenbaum, J. B. (2021). DreamCoder: Bootstrapping inductive program synthesis with wake-sleep library learning. *Proceedings of PLDI*. [Conference record](https://www.pldi21.org/poster_pldi.355.html).

Huang, K., Jin, Y., Li, R., Li, M. Y., Candès, E., and Leskovec, J. (2025). Automated hypothesis validation with agentic sequential falsifications. *Proceedings of ICML*, PMLR 267, 25372–25437. [Proceedings](https://proceedings.mlr.press/v267/huang25n.html).

Le Goues, C., Nguyen, T., Forrest, S., and Weimer, W. (2012). GenProg: A generic method for automatic software repair. *IEEE Transactions on Software Engineering*, 38(1), 54–72. [doi:10.1109/TSE.2011.104](https://doi.org/10.1109/TSE.2011.104).

Melrose, R. (2020). *Functional Analysis: Lecture Notes for 18.102, Spring 2020*. Massachusetts Institute of Technology. Chapter 1, sections 2, 4 and 5. [Lecture notes](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/3d4cc88026d44a01f936cd6a0aa995cb_MIT18_102s20_lec_FA.pdf).

Piriyakulkij, W. T., Langenfeld, C., Le, T. A., and Ellis, K. (2024). Doing experiments and revising rules with natural language and probabilistic reasoning. *Advances in Neural Information Processing Systems*, 37, 53102–53137. [Proceedings](https://papers.nips.cc/paper_files/paper/2024/hash/5f1b350fc0c2affd56f465faa36be343-Abstract-Conference.html).

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., and Yao, S. (2023). Reflexion: Language agents with verbal reinforcement learning. *Advances in Neural Information Processing Systems*, 36. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html).

Zhou, P., et al. (2024). SELF-DISCOVER: Large language models self-compose reasoning structures. *Advances in Neural Information Processing Systems*, 37. [Proceedings](https://proceedings.nips.cc/paper_files/paper/2024/hash/e41efb03e20ca3c231940a3c6917ef6f-Abstract-Conference.html).

Zsák, A. (2023). *Functional Analysis, Part III, Lent 2023*. University of Cambridge; notes taken by Leonard Tomczak. Sections 1.4 and 3.2. [Course notes](https://math.berkeley.edu/~ltomczak/notes/Lent2023/FuncAna_Notes.pdf).
