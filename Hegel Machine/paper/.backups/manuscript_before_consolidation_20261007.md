# Hegel Machine: A Framework for Revisable Research Knowledge

*English working manuscript — consolidated framework and evidence*

## Abstract

Scientific revision can change what is measured, which assumptions support a prediction, and how those assumptions are tested. We present Hegel Machine, a framework for learning scoped, executable abstractions together with their research dependencies. It treats prediction, representation, applicability and research procedure as possible objects of revision. Conditional reasoning connects a parent question to subordinate hypotheses, while persistent knowledge separates accepting a prediction, retaining its computation, and using it in a replacement. Bounded implementations learn programs and procedures, acquire distinguishing observations, and confirm revisions on fresh evidence. A recorded trajectory follows an acquired hypothesis through failure, nested relation repair, independent confirmation and later direct reuse. A language-model-authored research-method revision preserves a supported instrument after parent-prediction failure and raises frozen transfer success from 10/18 to 18/18 without further model calls. In an object-process intervention, retaining an observed-failing response solves 180/180 tasks versus 159/180 after deletion; additional conditional feedback provides no further benefit. Across 30 eight-task sequences, persistent research passes both confirmation and independent evaluation on 232/240 tasks, versus 215/240 for an equally expressive task-local researcher. Charged computation falls 22.42%, while CPU time rises 17.92%. Retaining learned procedures reduces successor observations by 20.83% and computation by 24.44% after instrument roles change. A separate intervention finds a new concept redundant and costly to consider. In a finite online-inquiry study, supporting answers change parent experiments and 47 of 120 continuations rewrite a retained step before execution. Recursive, flat and frozen-method controls all solve 120/120 continuations. These results establish bounded mechanisms of hypothesis revision, online supporting investigation and reuse; unrestricted discovery, calibrated uncertainty and stable long-horizon benefit remain open.

## 1. Introduction

A useful assumption makes a problem easier to solve. Treating a process as deterministic reduces the outcomes that must be considered. Assuming separability makes a multivariable response easier to fit. Choosing a coarse observation can remove detail that is irrelevant to a decision. Each simplification also creates a possible failure: an unusual outcome, an interaction, or an unobserved state can invalidate the explanation. A research agent therefore needs a way to investigate its assumptions as well as use them.

Even an elementary prediction contains such a commitment. It selects which details to represent and proposes a relation that carries from observed cases to another case. A more ambitious hypothesis may concern an invariant, an intervention, a model's operating conditions, or the effectiveness of a research procedure. We use **scoped abstraction** for this common structure: a compressed account of a problem, together with the conditions and consequences that make it meaningful. Learning can improve the answer within that account or revise the account itself.

This investigation involves several distinct choices. An error may be corrected by refitting an existing model, but it may instead reveal that the present observations cannot distinguish states with different consequences. An earlier rule may fail only in a new operating region. A combined explanation may remain partly correct, so replacing the entire model would discard useful knowledge. Even the experiment used to settle an earlier question may lose its ability to distinguish the current alternatives. Effective learning requires identifying which of these changes is justified by the available evidence. Resource limits constrain the investigation; its primary purpose is to acquire a correct explanation and recognize when the available evidence cannot support one.

Existing approaches provide important parts of this process. Program library learning acquires abstractions that improve subsequent synthesis. Active rule learning interleaves hypothesis revision and experimental design. Belief revision and truth maintenance provide foundations for changing commitments while tracking their dependencies. These precedents motivate a more specific question: **can an agent accumulate executable research knowledge whose applicability, composition, and investigation procedure can themselves be revised when tasks change?**

Our starting point is that knowledge has two roles. It supplies predictions about the current problem, and it changes how later problems are investigated. A learned object relation might appear directly in a response program. The same experience might also suggest which measurement to request, which boundary to probe, or which residual to investigate first. These roles can have different values. A correct program can be expensive to recover; a useful research procedure can produce a different program after refitting; a successful local revision can increase the cost of a later task. We therefore assess knowledge through the predictions and research decisions it supports on subsequent tasks. Acquisition and maintenance costs describe the resources required for those capabilities.

The central distinction is between an **explanatory commitment** and an **executable computation**. A claim that a program predicts the current response can be false even when the computation remains useful. For example, a response $C=A+0.7B$ fails when the event component changes to $B'$. Its computation can still support the repair $D=C+0.7(B'-B)$, and $C$ can again predict correctly when the original conditions return. The agent must acquire evidence for these uses; neither accepting every old prediction nor erasing every failed program realizes this form of learning.

Hegel Machine organizes these roles in an executable research loop. Hypotheses predict observable responses. Research procedures specify how candidate explanations are constructed from evidence and subproblems. A persistent library stores confirmed programs, their source dependencies, and evidence about their applicability. When a prediction fails, the agent can revise a binding or construct a replacement while retaining an earlier program for conditions in which it remains useful. New observations participate in further research, and subsequent tasks can call or adapt what was learned.

The broader question is how an agent decides what must be investigated next. A prediction can depend on a measurement interpretation; that interpretation can depend on an instrument model; testing the instrument can require a further hypothesis about an intervention. We call this **recursive research** when an unresolved supporting question becomes a research problem whose result changes its parent's justification. Reinforcement learning, robust optimization and causal inference expose different forms of these dependencies: the adequacy of a state, the range of possible environments, and the assumptions needed to identify an intervention effect. Section 3 develops these connections as a research framework.

The name expresses a Hegel-inspired motivation: a developed account should explain the achievements and boundaries of earlier accounts. In this framework, a history of hypotheses is useful knowledge about what can be computed, under which conditions it worked, and why it failed. We operationalize this motivation through predictions, observed conditions, and subsequent research costs. The relevant empirical object is a sequence containing acquisition, transfer, failure, repair, and return. Success on the final task alone would leave several of these requirements untested.

This paper develops three contributions. First, it formulates hypothesis revision jointly with representation, experimental distinguishability, and an evolving executable library. Second, it gives finite implementations that store predictive programs and research procedures and revise their use through measured counterexamples. Third, it tests this mechanism through actual program calls, fresh predictions, knowledge-removal interventions and complete research costs. Three 30-instance object-process studies examine retention, failure feedback and component-specific revision. A separate 30-instance joint-growth study tests representation changes, learned procedures and uncertainty within the same research loop. The broader formulation identifies what can be revised; the experiments establish specific benefits and failures of maintaining that research state.

## 2. Related work

**Program and library learning.** DreamCoder learns reusable program abstractions together with neural guidance for subsequent program search. Its learned concepts are compositional and support transfer across tasks. LAPS extends this approach through language-guided abstraction and search, including generalization comparisons under matched search budgets. These systems establish that learning a useful language of programs is already a substantive research direction. Hegel Machine focuses on the lifecycle of such knowledge when an acquired explanation becomes partially inapplicable and its investigation procedure must be revisited. [Ellis et al., 2020](https://arxiv.org/abs/2006.08381); [Wong et al., 2021](https://proceedings.mlr.press/v139/wong21a.html).

**Program synthesis and repair.** SKETCH uses counterexamples from verification to guide synthesis of finite programs. GenProg searches for repairs using tests that encode a defect and the functionality to preserve. These approaches already make failure evidence and preservation central to program revision. The setting considered here adds an evolving empirical task, experiments that must be acquired, and a persistent library whose subsequent research value is measured. [Solar-Lezama et al., 2006](https://people.eecs.berkeley.edu/~sseshia/pubdir/asplos06-final.pdf); [Le Goues et al., 2012](https://doi.org/10.1109/TSE.2011.104).

**Active hypothesis formation.** Piriyakulkij et al. combine language-model proposals with sequential Monte Carlo inference and information-based experiment selection. Their natural-language hypotheses are translated into executable code, and their revision procedure retains better explanations while revising poorer ones. They also examine computational budgets. The question pursued here concerns persistent applicability and component-level repair across task sequences, together with the reuse of research procedures. An advantage on this question requires direct experimental comparison. [Piriyakulkij et al., 2024](https://papers.nips.cc/paper_files/paper/2024/file/5f1b350fc0c2affd56f465faa36be343-Paper-Conference.pdf).

**Generated hypotheses and sequential falsification.** Hypothesis-Driven Reasoning extracts factors and iteratively generates and verifies rules that support new predictions. POPPER starts from a supplied hypothesis and constructs executable subordinate tests, checking their relevance and combining their statistical evidence. These are direct precedents for explicit hypothesis memory and supporting investigations. Our empirical question concerns whether an unresolved supporting condition changes the next experiment, whether its answer returns to revise a parent explanation, and whether the resulting research program remains useful when tasks change. A graph of completed fits alone does not answer this question. [Agarwal and Yamada, 2026](https://doi.org/10.1609/aaai.v40i3.37146); [Huang et al., 2025](https://arxiv.org/abs/2502.09858).

**Experience and research procedures.** Reflexion uses textual feedback to guide subsequent attempts; ExpeL separates retrieved experience from induced insights; Voyager accumulates executable skills; SELF-DISCOVER composes reasoning structures from existing modules. These approaches motivate separate interventions on the availability of a procedure and on its modification after failure. They also motivate a common-model comparison with ordinary code-based investigation and textual experience, using the same accessible experiments. [Shinn et al., 2023](https://arxiv.org/abs/2303.11366); [Zhao et al., 2023](https://arxiv.org/abs/2308.10144); [Wang et al., 2023](https://arxiv.org/abs/2305.16291); [Zhou et al., 2024](https://arxiv.org/abs/2402.03620).

**Belief revision and assumption maintenance.** The AGM account formalizes contraction and revision of theories. Assumption-based truth maintenance tracks the assumption sets supporting conclusions and permits reasoning across incompatible contexts. Our empirical setting adds executable predictors, acquired measurements, and resource costs: an apparent contradiction must be investigated, its conditions matter, and a replacement must make successful predictions. These are connections to established approaches to revisable knowledge. [Alchourrón et al., 1985](https://doi.org/10.2307/2274239); [de Kleer, 1986](https://doi.org/10.1016/0004-3702(86)90080-9).

**Representation and observability.** Representation learning studies how explanatory factors become accessible through a learned representation. Identifiability results in causal representation learning make explicit the role of assumptions and interventions. We use this connection to distinguish fitting a response within a representation from investigating whether the representation preserves the distinctions required by the task. [Bengio et al., 2013](https://arxiv.org/abs/1206.5538); [Varici et al., 2024](https://proceedings.mlr.press/v238/varici24a.html).

**Scientific and self-improving agents.** The AI Scientist-v2 organizes scientific experimentation through agentic search; AlphaEvolve searches over executable algorithmic proposals; the Darwin Gödel Machine changes agent code and evaluates the resulting agents. Hegel Machine takes executable scientific assumptions and their research procedures as its main units of accumulated knowledge. The present implementation makes these units inspectable in controlled numerical environments. [Yamada et al., 2025](https://arxiv.org/abs/2504.08066); [Novikov et al., 2025](https://arxiv.org/abs/2506.13131); [Zhang et al., 2025](https://arxiv.org/abs/2505.22954).

The Red Queen Gödel Machine further co-evolves agents and their evaluators, studying how changes to the search criterion affect the resulting agents. Hegel Machine investigates a different evolving object: the supporting assumptions and procedures used to establish an explanation. A measurement or test procedure can change while the external predictive question remains fixed. This distinction lets an experiment ask whether a revised investigation discovers a better explanation under the same outcome criterion. [Iacob et al., 2026](https://arxiv.org/abs/2606.26294).

These precedents supply the ingredients for our formulation. Its organizing distinction separates accepting a prediction, retaining a callable computation, and using that computation in a new explanation. The empirical question is whether maintaining these distinctions improves research across acquisition, failure, revision and return, after paying for the original knowledge. Source-availability interventions and complete-sequence costs make that question testable in the implemented system.

## 3. A research state that can be revised

### 3.1 Hypotheses, evidence, and effective languages

We distinguish a **hypothesis**, which is a claim being investigated, from an **assumption**, which is a premise provisionally used in an inference. The same proposition can occupy either role: a fitted relation is initially a hypothesis, becomes a working premise in a composition, and returns to being a research question after failure. Its role can change without its status becoming certain.

A representation $\phi$ groups detailed situations by the information it retains. Writing $x\sim_\phi x'$ when $\phi(x)=\phi(x')$ makes the abstraction explicit. A predictor defined on these groups can succeed only to the extent that the discarded distinctions are irrelevant to its query and scope, or are represented through uncertainty. For a causal query, relevance also depends on the interventions considered. Thus abstraction is task-relative: a coarser description can be adequate for one prediction and inadequate for another. A constant forecast is already a predictive hypothesis; a useful learned law must additionally survive tests beyond the cases used to construct it.

An agent encounters tasks indexed by $t$. Within a task, it chooses experiments $e$ and observes outcomes $y$. An experiment specifies controllable inputs, a trajectory length, and requested measurement channels. Acquiring and processing the resulting data incurs a resource vector

$$
c(e)=(c_{\mathrm{obs}},c_{\mathrm{meas}},c_{\mathrm{comp}}).
$$

We describe its research state as

$$
\mathcal R_t=(\Phi_t,\mathcal L_t,\mathcal E_t,\Pi_t,K_t,D_t,b_t),
$$

where $\Phi_t$ contains available observation representations, $\mathcal L_t$ contains executable primitives and learned definitions, $\mathcal E_t$ contains available experiments, $\Pi_t$ contains research procedures, $K_t$ stores knowledge and its dependencies, $D_t$ is acquired evidence, and $b_t$ is the remaining budget. This notation describes the objects that can participate in research; different implementations may permit different subsets to change.

A predictive hypothesis contains a representation $\phi$, a program $p$, an applicability description $s$, and any explicit supporting premises $\Gamma$. We write $h=(\phi,p,s,\Gamma)$ to expose these components. Given an observed history $x_{\leq n}$, its deterministic form predicts

$$
\hat y_n=p\bigl(\phi(x_{\leq n})\bigr).
$$

The task objective and resource accounting are fixed by the experimental protocol. An applicability description can include an executable condition, an instrument binding, or recorded empirical coverage. Coverage describes where evidence has been obtained; extending that coverage requires additional investigation. A knowledge entry also records supporting observations, counterexamples, source dependencies, and the procedure used to construct its program.

The framework also admits distributional predictions and structural claims such as conditional independence or invariance. These claims have different criteria for support. A research-method hypothesis concerns whether a procedure resolves a question at an acceptable cost; a meta-prior expresses a revisable preference over candidate structures or research directions. Keeping these types distinct prevents a useful search preference from being mistaken for an established law of the environment.

Learning a new callable definition changes the agent's **effective research language**: expressions can be proposed, evaluated, and combined using an acquired abstraction. In a fixed underlying programming language, the definition may already be expressible by expanding its body. Its value can nevertheless be measured through improved access to solutions under finite search and acquisition budgets. This distinction lets us evaluate knowledge growth through actual use rather than through the number of newly introduced names.

### 3.2 Structural relations as testable assumptions

A structural relation specifies a pattern together with the roles and conditions under which it can be tested. Monotonicity involves an ordering and an inequality; symmetry involves a transformation and a prediction-preservation condition; conservation involves a collection of quantities and an allowed transition. These constraints differ in arity and in the observations needed to assess them.

For example, a conservation candidate for an exchange process can predict that the total quantity after a transition equals the previous total plus externally supplied input. A local event candidate may depend on which objects lie within a relational boundary. A response composed from these candidates can fail because the boundary changes while conservation continues to hold. Representing the two components explicitly makes this a question about local repair and retained dependencies.

The corresponding search bias is also revisable. A preference for separable models can guide an inexpensive first fit; persistent interaction residuals can motivate coupled candidates. In the implementation, these meta-priors are structured search and applicability records. They influence candidate generation and research reuse, and are evaluated by the consequences of using them.

### 3.3 Experiment-induced distinguishability

The same pair of hypotheses can agree under one set of observations and disagree under another. Let $P_h^e$ denote the outcome distribution predicted by hypothesis $h$ under experiment $e$. For a finite experiment set $E$, define

$$
h\sim_E g
\quad\Longleftrightarrow\quad
P_h^e=P_g^e\quad\text{for every }e\in E.
$$

With strictly positive weights $w_e$, a compatible pseudometric is

$$
d_E(h,g)=\sum_{e\in E}w_e\,\mathrm{TV}(P_h^e,P_g^e).
$$

It becomes a metric on the corresponding observational equivalence classes. Adding an experiment refines these classes whenever it distinguishes hypotheses that were previously equivalent. Appendix A gives the elementary argument. This perspective captures the useful part of the functional viewpoint: the tests available to an agent help determine which distinctions matter to its research.

The implementation below uses finite prediction-disagreement scores and task losses. The distributional formulation explains the role of experiments; it does not require a calibrated posterior over programs. Likewise, candidates that happen to agree on a small acquired sample remain empirically unresolved. Population-level observational equivalence is a stronger statement.

### 3.4 Revision and preservation

Let $D_{\mathrm{res}}$ contain observations that motivate revision and let $D_{\mathrm{old},s}$ contain earlier successes within an applicable scope. An empirical preservation objective asks for a revised hypothesis $h'$ satisfying

$$
\widehat R_{D_{\mathrm{res}}}(h')
<\widehat R_{D_{\mathrm{res}}}(h),
\qquad
\widehat R_{D_{\mathrm{old},s}}(h')
\leq\widehat R_{D_{\mathrm{old},s}}(h)+\epsilon,
$$

together with accurate predictions on additional observations. Preservation can also be realized by retaining $h$ for its old conditions and learning $h'$ for the changed conditions. The task sequences below explicitly return to both conditions to evaluate this possibility.

The equations define an evaluation objective. In the implemented loop, newly acquired trajectories confirm current candidates, while returned tasks test whether useful earlier knowledge remains available. Resource comparisons then determine whether this form of preservation reduces the total cost of subsequent research.

This formulation separates three decisions about an acquired program $p$: whether to accept $p$ as the current predictor, whether to retain $p$ as a callable computation, and whether to use $p$ when constructing a replacement. A counterexample bears directly on the first decision. The other two depend on subsequent predictive and research value. In particular, a replacement can contain a call to $p$ together with a learned correction even when $p$ alone is inaccurate. Preserving the computation then preserves an accessible research resource, while confirmation determines whether the new explanatory commitment is supported.

### 3.5 Recursive research and conditional justification

The object of recursion is an unresolved obligation supporting a conclusion. Suppose a parent question asks whether a response law explains an observed discrepancy. A subordinate question may ask whether a sensor is calibrated, whether an interaction boundary changed, or whether the proposed experiment distinguishes two explanations. If answering that question requires an unknown coordinate binding, the binding becomes a further question. A result flows back to the parent as a usable computation, a supported condition, a counterexample, or an unresolved dependency. The parent's conclusion must respect those dependencies.

This gives a reusable research pattern: identify the blocked inference, formulate a subordinate claim with observable or deductive consequences, investigate it, and update the parent. The same pattern can be applied at another level when needed. Depth is justified by the problem's dependencies and limited by available evidence and resources. A parent can remain conditional when a child is unresolved. Reopening a parent after a counterexample creates a new research episode over accumulated evidence.

Least squares and Newton's method clarify the smallest ingredients of this pattern. Least squares chooses parameters to minimize $F(\theta)=\tfrac12\sum_i(y_i-f_\theta(x_i))^2$. A Newton step solves a local quadratic model through $\nabla^2F(\theta_k)d_k=-\nabla F(\theta_k)$ and updates $\theta_{k+1}=\theta_k+\alpha_kd_k$. In full-rank linear least squares, the quadratic model is exact and an undamped step reaches the minimizer. For more general objectives, curvature conditions and step selection matter. These are numerical operations within specified premises. A research-level question arises when the agent investigates whether the chosen response family, error description, or local approximation is adequate. Optimization success settles the fitted objective; empirical adequacy requires consequences that can fail on new evidence. [Boyd and Vandenberghe, 2004](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).

Adding a condition is another legitimate research operation. A difficult question $q$ may become tractable under an additional assumption $a$:

$$
\Gamma\cup\{a\}\vdash q
\quad\Longrightarrow\quad
\Gamma\vdash(a\Rightarrow q).
$$

The result is a conditional theorem in a deductive setting. For an observable applicability condition, its empirical counterpart restricts evaluation to $S_a=\{x:a(x)\}$. A structural or distributional assumption instead restricts the admissible model class to $\mathcal M_a=\{M:M\text{ satisfies }a\}$. In either case, support concerns the resulting conditional claim. The added condition should identify a meaningful, nonempty set of cases or models and yield consequences on newly acquired evidence. Whether $a$ applies to the current situation is itself a possible subordinate question. If its validity cannot be established, the answer remains explicitly conditional. If later evidence violates $a$, the agent can investigate a broader model or a different condition.

For example, a global linear response may fail because two operating regimes have different mechanisms. A conditional model can ask whether linearity holds within a measurable regime, and another question can investigate how to identify that regime before predicting. Conversely, if a proposed measurement cannot distinguish the regimes, the agent must address the missing information or retain uncertainty. Narrowing the scope, enriching the representation, and widening the model family are different responses to the same failed commitment.

### 3.6 Reinforcement learning, robustness and causal identification

These fields provide concrete instances of assumptions that a research agent can expose and reconsider. Their objectives and evidential requirements remain distinct; the shared framework concerns which commitments support an answer and what action can resolve a failed commitment.

**Reinforcement learning: assumptions behind a decision state.** In a stationary Markov decision model, $s_t=\phi(x_{\leq t})$ is taken to contain the history relevant to the next transition and reward. Value learning estimates the consequences of actions under a policy and environment; a model-free algorithm can do this without explicitly fitting a transition model. A research question arises when histories mapped to the same state systematically have different consequences. The agent can investigate memory, a new observation, or a different state abstraction, then test the resulting prediction or control performance. This separates learning within a decision model from studying whether its representation is adequate. The reward or task objective supplies the criterion for that study. [Sutton and Barto, 2018](https://mitpress.mit.edu/9780262039246/reinforcement-learning/).

**Robustness: revising which environments are possible.** A nominal decision uses one model $M_0$. A robust decision can instead optimize a lower envelope,

$$
\max_\pi\inf_{M\in\mathcal U}J(\pi;M),
\qquad
\mathcal U=\{M_0\}\ \Longrightarrow\ \max_\pi J(\pi;M_0).
$$

Here $J(\pi;M)$ denotes expected return under policy $\pi$ and environment model $M$. Enlarging $\mathcal U$ admits more possible environments and can make the decision more conservative. An uncertainty-set hypothesis concerns what belongs in $\mathcal U$, how its coverage is justified, and which nominal performance is sacrificed. Robust dynamic programming additionally depends on the structure of that set; rectangularity is a substantive condition for the decomposition studied by Iyengar. Within our framework, a failed nominal prediction can motivate investigation of variability or model uncertainty, while later evidence may justify a simpler description. [Iyengar, 2005](https://doi.org/10.1287/moor.1040.0129).

**Causal inference: assumptions behind an intervention claim.** Observational agreement can coexist with different predictions for $P(Y\mid\operatorname{do}(X=x))$. Identification asks whether the causal query is fixed by the available distribution and the assumed causal structure. A research agent can seek an intervention, investigate a supporting assumption, or report the remaining range of answers. Pearl's instrumental-variable example illustrates conditional tractability: a causal effect that is generally not identified nonparametrically becomes identified under appropriate linear structural and instrumental-variable assumptions, with a nonzero instrument--treatment covariance. The extra assumptions create a more restricted identification problem. They also become dependencies of the answer. [Pearl, 2009](https://escholarship.org/content/qt36w8n7pg/qt36w8n7pg.pdf).

The common operations are therefore specific: expose a premise, obtain discriminating evidence, revise its scope or representation, propagate the change to dependent conclusions, and assess the consequences. Research can also target the experiment-selection procedure itself when it repeatedly fails to expose relevant distinctions. This organization makes it possible to study assumptions about the world and assumptions about how to study the world within one persistent state.

### 3.7 Generalization, structural correspondence and the history of hypotheses

A broader account can explain an earlier law as a special case, a scoped approximation, or a component inside a larger mechanism. If $\rho$ binds the old variables and conditions into a new model $h'$, preservation can be expressed as

$$
\sup_{x\in S}\left|h'(\rho(x))-h(x)\right|\leq\epsilon,
$$

for a specified old scope $S$, with a corresponding limiting statement when appropriate. The bound states what a proposed generalization would preserve; demonstrating it requires a proof or evidence suited to the claim. The broader model also needs consequences outside that scope. These requirements give generalization content beyond storing several unrelated answers. When unification is unsupported, separately scoped models can remain available.

For an illustrative extension of the composition example, a family $G(x;\lambda)=A(x)+0.7B_\lambda(x)$ contains $C$ at an old boundary $\lambda=r$ and a revised response at $\lambda=r'$. Learning the family, identifying the applicable $\lambda$, and showing that it predicts new cases are separate research achievements. The implemented additive repair $D=C+0.7(B_{r'}-B_r)$ preserves useful computation even before a compact, scope-selecting family has been learned.

Reuse across tasks similarly requires a proposed structural correspondence. A relation can involve an ordering, a transformation, several conserved quantities, or a conditional dependency. The agent must bind the current quantities to the relation's roles and test the resulting consequences. For predictive programs, a useful correspondence can be written schematically as $p_{\mathrm{new}}(x)\approx T[p_{\mathrm{old}}(\rho(x))]$, where $\rho$ maps input roles and $T$ accounts for a justified output transformation. The mapping and its scope are hypotheses too. Successful reuse executes the acquired relation with the current binding; failed reuse can prompt revision of the binding, the relation, or the experiment used to assess it.

This gives a precise engineering interpretation of the Hegelian motivation. Hegel's account of the history of philosophy treats earlier systems as moments in the development of later thought. We take from it a research principle: the history of successful abstractions and their limitations can inform the construction of a more adequate account. A retained hypothesis carries executable content, empirical scope, failure evidence and dependencies. Earlier conclusions can be recovered where warranted, and their failures can explain why further structure was needed. [Hegel, 1892, Introduction A.3](https://www.gutenberg.org/files/51635/51635-h/51635-h.htm).

Thus the unit of accumulated knowledge is a scoped abstraction together with the research dependencies supporting its use. Its history changes both the answers available to the agent and the investigations that become accessible next.

![A revisable research state, response-program preservation and repair, and research-procedure generation, reuse and revision.](figures/revisable_knowledge_framework.png)

**Figure 1. Revisable research knowledge and its implemented revision paths.** **a**, The framework admits changes to representations, hypotheses, experiments, procedures, and applicability or search priors; each implementation provides bounded revision operators. **b**, A response that fails as a complete prediction can remain an executable component of a repaired explanation and be reassessed when earlier conditions return. **c**, Paid research traces yield procedures with roles and subquestions. These procedures are rebound to new tasks, replayed or revised after failure, and tested on fresh observations. Measured outcomes also update applicability and search priorities. [Vector PDF](figures/revisable_knowledge_framework.pdf).

## 4. The Hegel Machine research loop

We instantiate the framework with two bounded controllers built on a common
executable language and predictive knowledge library. The object-composition
controller tests preservation and local repair of learned definitions. The
joint-growth controller combines expression, representation, scope, procedure
and uncertainty revision. Sections 4.1–4.5 describe the research operations;
Section 4.6 explains their joint-growth instantiation. Each study fixes its own
controller and protocol. A retained definition can be called unchanged, while
a retained procedure can construct a newly fitted response; the experiments
record these forms of reuse separately.

### 4.1 Executable knowledge and paid evidence

The current implementation uses an abstract syntax tree language containing numerical operations, predicates, object and relation aggregation, temporal operators, and calls to learned definitions. Temporal operators include lagged values, windows, event differences, and accumulating state. Programs can contain nested calls to earlier definitions, making their source dependencies executable.

The environment exposes control variables and a finite measurement interface. Acquired observations are stored with their experimental context. A supported projection can reuse a measurement in another view without purchasing the same observation again. Candidate fitting, validation, and further research operate on this accumulated evidence. Observation acquisition, measurement, and charged program computation are recorded separately.

Confirmed knowledge stores a response program and, where available, its parameterized abstraction and research procedure. A retained response can be called directly. A retained procedure can instead reconstruct a response from the current data, refitting its parameters. These operations serve different purposes and are recorded separately in the evaluation.

### 4.2 Competing explanations and dependent research

Research begins with inexpensive response fits in the current coordinates and relevant retained knowledge. Available directions include calibration, alternative measured views, conditional responses, object relations, residual composition, and temporal structure. The candidate generators use finite families, including affine and sparse responses, separable and coupled terms, low-dimensional projections, and bounded history.

A candidate can depend on a subproblem whose answer supplies a coordinate, a component, or a residual explanation. A research graph records these dependencies, and executable answers can be composed into the parent response. In the nested relation-revision path, the generator first fits candidates and then compiles its completed operations into an answered dependency graph. The controller subsequently obtains distinguishing observations and refits within a bounded loop. These operations implement finite dependent revision; Section 5.9 traces their execution. General composition search uses a small beam and a limited number of expansion rounds and fitted terms.

Fitting a more complex model is only one possible response to failure. The agent can purchase a different view, reconsider a condition, or revisit a previously unsuccessful direction after new evidence becomes available. These choices compete within the resource budget. The session controller can also revisit an earlier task when newly learned knowledge creates a potentially useful research opportunity.

### 4.3 Choosing and revising experiments

For candidates that predict directly from controllable coordinates, the agent constructs a finite pool of probes and compares their predicted responses. One implemented selection rule is

$$
e^*=\arg\max_{e\in\mathcal E_t^{\mathrm{pool}}}
\frac{1}{|e|}\sum_{x\in e}
\left[\max_{h\in\mathcal H_t}h(x)-\min_{h\in\mathcal H_t}h(x)\right].
$$

The pool includes random sequences and structured variations such as alternating endpoints and piecewise-constant inputs. Other research directions use probes for instrument calibration, relational boundaries, or temporal conditions. An experiment's construction therefore depends on the current candidates and the problem being investigated. Required measurements are acquired through the environment interface.

The interpretation of a failed experiment also matters. If an earlier structure fit was based on too little evidence, a new counterexample can justify fitting that direction again using the expanded data. Treating a direction as permanently exhausted after its first attempt would confuse the outcome of one fit with the limits of the hypothesis family. Conversely, revisiting a direction without new evidence or a changed candidate can consume resources without adding a distinction.

### 4.4 Fresh confirmation and scoped revision

A selected program is frozen before its confirmation trajectory is acquired. Successful confirmation permits it to enter the reusable library. An unsuccessful trajectory becomes evidence for subsequent research, after which a revised candidate requires new confirmation. The distinction between candidate construction and confirmation is maintained even when the research process resumes an earlier task.

Retained programs are evaluated on current paid observations. Their failures are recorded even when another candidate wins the comparison. In the ordinary transfer path, a losing retained call adds a counterexample to its empirical prior and changes that record to conditional. Its executable knowledge remains available for later reassessment. Current observations can therefore reject an old response on a changed task and support it when the earlier conditions return. Section 5.4 separately tests the consequences of retaining the response and of adding this conditional feedback.

Composition makes preservation more precise. If a combined response fails while one of its source definitions remains useful, the revision can retain that definition and change the residual contribution. The experiments inspect the resulting program, test its predictions, and intervene on source availability. These observations address different aspects of the same revision: structural retention, predictive adequacy, and subsequent value.

### 4.5 Learning research procedures

A research procedure records selected transformations, dependencies, and fitting operations. Its reuse transfers the structure of an investigation to another task, where coefficients or conditions can be fitted again. A successful direct call and a successful procedure replay therefore need not produce the same execution trace, even when they ultimately produce the same predictive program.

Research procedures and their applicability are stored with supporting evidence and counterexamples. The session controller uses available knowledge and the remaining budgets to price further visits. Its portfolio search compares affordable single visits and selected pairs, executes an available action, and updates the plan from the measured result. This provides a concrete mechanism for allocating research across dependent tasks.

When an unfinished task resumes, the controller also considers retained concepts
that were unavailable to its earlier visit. Screening uses that task's purchased
observations and charges the adaptation and comparison work. A squared-error
candidate that meets the fit tolerance on these observations and improves the
incumbent can be frozen for one new confirmation trajectory. This permits a
revisit without purchasing a second selection trajectory; failure ends the quoted
package and retains the new evidence. Appendix E.3 tests this continuation under
an unchanged resource ceiling.

The overall loop is summarized below. Each operation refers to the bounded candidate generators, evidence store, and research actions described above.

```text
Input: task interface, reusable knowledge K, resource budget b
Acquire initial observations and compare current fits with applicable knowledge.
While an affordable research action remains:
    Construct or adapt candidates using paid evidence and dependent subproblems.
    Compare candidates, required measurements, and remaining research cost.
    Select a candidate and an experiment appropriate to the unresolved distinction.
    Freeze the candidate; acquire a fresh confirmation trajectory.
    If confirmation succeeds:
        Store the executable result, its dependencies, and its research procedure.
        Make the new knowledge available to later tasks and affordable revisits.
        Return the supported response for this visit.
    Otherwise:
        Retain the counterexample and revise the record of the failed application.
        Reconsider candidates and research directions using the enlarged evidence.
Return the best supported response and the unresolved research state.
```

### 4.6 Joint growth of the research state

A second finite instantiation makes the joint revision loop explicit through a
single persistent researcher. It shares the executable language and knowledge
library with the preceding implementation, while adding an observation-driven
expression search, evidence-dependent procedure compilation, coordinate transfer,
and a repeated-measurement uncertainty model. This instantiation is evaluated
separately from the object-composition experiments below.

**Effective vocabulary and applicability.** Candidate search first tests
previously acquired definitions under empirically compared role bindings. Failed
predictions generate residuals from which products, absolute-value relations and
thresholded linear terms are proposed. Both persistent and task-local search can
construct these expression families. A retained definition provides an additional
starting point; its absence does not remove conditional models from the
task-local hypothesis language. An old response $p$ can be embedded in a new
response $q(x)=\mathbf{1}_{c(x)}p(x)+\mathbf{1}_{\neg c(x)}r(x)$, where the
condition and replacement are learned from acquired observations. The old model
is then exactly preserved on the learned branch. Nearby observations bracket a
proposed threshold, and additional probes target that unresolved interval.

**Roles and representations.** A callable definition separates its body from its
observed arguments. Candidate argument permutations are assessed on current
responses. Public coordinate ranges additionally propose affine changes of
units and origin; the measurements determine whether a proposed correspondence
predicts the new task. A learned procedure can directly compare paid instrument
views on current pilot data. If no applicable procedure resolves the mismatch,
the fallback investigation repeats fixed controls before requesting that
comparison. Selecting a view changes the inputs available to subsequent
explanation search. The public ranges and instrument catalogue constrain what
it can seek.

**Generated research procedures.** Acquisitions produce evidence handles, fits
consume the observations actually used, and confirmations consume a frozen
candidate. The ancestor graph of a successful confirmation becomes a reusable
procedure. Failed confirmations remain usable as counterexample acquisitions;
unsuccessful or unused search branches are omitted from the stored procedure.
A successful instrument comparison becomes a subproblem generator: on a later
task it constructs a fitting question for each currently available instrument,
then selects a view by measured validation loss. The recorded winning instrument
index is not assumed to retain its meaning. Subsequent failure can restrict the
procedure's applicability and produce a child with different executed steps.
Primitive laboratory commands remain fixed, while their dependencies, bindings
and composition are learned from completed investigations.

**Uncertainty and reversible simplification.** Repeated residuals at fixed
controls estimate within-setting variation. A misspecified deterministic mean
that produces the same residual on every repeat does not by itself generate a
positive noise estimate. Constant and coordinate-dependent variance proposals
are compared by leaving complete repeated settings out, rather than treating
replicates at one setting as independent validation locations. A fresh response
batch then assesses the mean prediction relative to the estimated noise scale.
Simpler predictive expressions compete under the same task objective; their
adoption preserves earlier complex definitions for later reassessment. Evaluation
separately measures mean error and variance calibration, so a larger declared
variance cannot make a poor mean predictor pass the external task criterion.

These mechanisms connect changes to effective vocabulary, explanatory scope,
representation, procedure and uncertainty in one loop. Their operational scope
and separate evidence contrasts are reported in Section 5.7 and Appendix D.

### 4.7 Online supporting investigations

A third, finite instantiation makes supporting questions operational before
their evidence is collected. A calibrated-response laboratory exposes two
anonymous controls and three experiment types: observe the response, measure
an unknown physical coordinate, and calibrate the instrument against a known
reference. Its response is $y=f(o u_j)$ and its instrument reads
$r=a(o u_j)+b$. The public language contains linear, quadratic, absolute-value
and threshold responses, two control bindings, two orientations, two calibration
signs and three offsets: 96 joint hypotheses. These primitives and their allowed
parameters are shared by all researchers.

The recursive researcher opens a response question. Unresolved coordinate
bindings produce a supporting question; unresolved instrument parameters can
produce its calibration subquestion. Observations filter the shared hypothesis
set. A returned answer changes the parent's remaining alternatives and therefore
its choice of experiment. When a retained research step distinguishes fewer
current alternatives than another available step, the researcher rewrites that
step before acquiring its observation. The revised executable procedure is
retained after confirmation. Questions already answered by current evidence need
not be expanded.

Two controls isolate this organization. A flat active researcher selects from
the complete experiment menu using information gain about the response law,
without a support tree. A frozen-method researcher uses the same recursive
organization and source knowledge, attempts the retained steps unchanged, and
then permits fresh adaptive fallback. Freezing a procedure therefore removes
step rewriting without withholding the decisive measurements or the candidate
language. Responses are represented in the existing Program language; a
retained Definition can be called with a newly established coordinate binding.
This is executable reuse after reassessment, not an exemption from research.

### 4.8 Common-model code investigation

The language-model comparison uses the same external model within each run
for three knowledge conditions: gpt-6.1-sol through Responses, and a separate
qwen3.8-flash run through native Chat Completions. Each can buy any public laboratory experiment,
write and run Python, evaluate on purchased observations, and submit a predictor.
The host supplies observations and executes code; it supplies no hypothesis
fitter or hidden task parameters. A submission is frozen before three new
confirmation observations are collected. Their targets remain outside every
model-readable research file during scoring. Failed confirmation then makes
the observations available as counterexamples.

The ordinary Code/ReAct condition starts each task without persistent knowledge.
The textual-memory condition carries the source agent's own lesson. The Hegel
condition carries the source agent's executable predictors, experimental
programs and editable hypothesis record. A stored experimental program must
actually have selected source experiments before it can be retained; a question
is recorded before those observations are acquired. Existing evidence may resolve
a task without additional supporting questions. New tasks can execute or revise
the retained files through the same Python interface available to all three arms.
Executed entry points are distinguished from agent-authored dependency references.

This comparison tests representation and persistence of research knowledge under
a common proposal model. The LLM authors its programs and decisions; it is not
a language front end to the finite researcher in Section 4.7. Both controllers
use the same public laboratory and independent prediction criterion, allowing
their evidence to address complementary questions.

## 5. Controlled studies

### 5.1 Evaluation questions and protocol

Evaluation follows three questions. First, can acquired computations survive
counterexamples and support appropriate local repair? Sections 5.3–5.5 compare
complete research algorithms, intervene on response retention, and vary which
component changes. Second, can several objects of research change within one
persistent process? Section 5.7 combines task sequences with interventions on
programs, procedures and measurement access. Third, does available knowledge help
an unfinished consumer, and does newly produced knowledge contribute separately?
Section 5.8 tests this distinction under the remaining research budget.
Section 5.10 separately tests online supporting investigations and source-held-out
structural transfer, with correct explanations and predictions as primary outcomes.

Three 30-instance object-process studies and one 30-instance joint-growth study
provide the main evidence. Appendix C contains earlier 12-instance mechanism
studies; Appendix E contains the subsequent 12-instance integration studies and
a post-hoc product-removal intervention on the same continuation cases. Each
study has its own frozen implementation and protocol. Comparisons are made
within studies, with all registered successes and failures retained.

In the composition studies, each complete learning sequence is followed by independent evaluation on four trajectories of 64 observations per task. A task passes when its mean squared error is below $10^{-6}$. Research-time confirmation uses a threshold of $10^{-7}$. The reported resource totals include source learning, candidate search, unsuccessful research, and confirmation; terminal evaluation is performed separately. Charged computation is the implementation's operation accounting, whereas observations and measurements represent separate acquisition costs. The joint-growth study has its own deterministic and stochastic criteria, specified in Appendix D.

Source-deletion arms begin from the same actually learned prefix as the retention arm. The selected source and its dependent knowledge are removed, and both arms continue with the same researcher settings and budget limits. Deleted knowledge may be relearned. The contrast measures the total downstream effect of making that source available at the intervention point. All arms complete learning before terminal evaluation is requested.

A procedural criterion supplements prediction accuracy. It asks whether sources were learned from an empty library, subsequently used, contradicted by acquired evidence, revised, and available when the original conditions return. The composition study additionally requires actual calls to both components, preservation of the unchanged source definition, and later calls to the original and revised compositions. These criteria make particular knowledge-lifecycle events observable.

### 5.2 Task family and supporting mechanisms

The main experiments use a controlled object-exchange family with selectable coarse and fine measurements. The total-quantity response $A$ is conserved through internal exchange, whereas the event response $B_r$ depends on a relation radius $r$. Their combined response is $C=A+0.7B_r$. The six-task sequence learns $A$, learns $B_r$, learns $C$, changes the radius to $0.45$, returns to $C$, and returns to the changed response. Each initial task allows 384 observations, 4,608 measurement units, 360,000 charged computation units and six research rounds. The agent learns the components from observations and can later call their acquired definitions.

Three earlier mechanism studies provide supporting evidence (Appendix C). The single-source study shows a small complete-sequence saving from retention, but its inexpensive fixed-library learner solves all 48 tasks with much less computation than full research. The two-source study identifies positive, separately measurable contributions from two acquired sources. The object-composition study preserves the exact unchanged component inside a repair and reduces complete-sequence computation by 41.11% relative to deletion of that component. Its procedural criterion passes in 11/12 instances, while both arms solve all 72 terminal tasks: the remaining instance recovers the right program by replaying a procedure rather than directly calling the acquired revision. These findings motivate the prospective comparisons below and establish that prediction accuracy, actual reuse and total research cost need separate assessment.

### 5.3 Complete research algorithms on the composition sequence

We compare four algorithms on the six-task sequence in Section 5.2. The full researcher retains knowledge across tasks. A task-local researcher runs the same hypothesis search, instrument learning, confirmation, and within-task memory, but starts an independent session with an empty library for every task. The fixed-library learner fits a predeclared quadratic and two-step-history basis. The passive researcher learns representations and shared knowledge from four precommitted random trajectories of 24 observations per task, without adaptive acquisition or revisits. Both fixed controls use the finest legal measurement view. All arms access the same public laboratory interface.

Initial task budgets match Section 5.2, giving sequence ceilings of 2,304 observations, 27,648 measurement units, and 2,160,000 charged computation units. The full researcher can allocate unused sequence resources to revisits. Each task-local session keeps its own budget, so this contrast evaluates the combined benefit of cross-task knowledge and resource allocation. Actual expenditure is reported separately from these common ceilings.

After repairing the passive researcher's selection of mutually exclusive measurement views on a development instance, we froze the implementation and registered 30 new apparatus seeds, with 15 assigned to each existing generator profile. Both profile groups were run prospectively under that freeze; their labels do not denote training and test partitions in this experiment. Every arm finishes learning before any arm receives terminal scores. Evaluation uses four common new control trajectories per task and each program's required public measurements. Uncertainty is computed from 10,000 paired bootstrap resamples of apparatus instances, stratified by generator profile. The unit of resampling is a complete six-task sequence.

**Table 1. Four-arm composition comparison.** All 30 registered instances completed. Each arm is evaluated on 180 tasks; resource totals include complete research sequences. CPU time is measured process time in seconds.

| Arm | Tasks passing | Observations | Measurements | Charged computation | CPU seconds |
|---|---:|---:|---:|---:|---:|
| Full | 180/180 | 17,489 | 75,192 | 9,078,989 | 1,939.14 |
| Dense fixed basis | 0/180 | 17,520 | 210,240 | 11,074,896 | 406.06 |
| Passive experiments | 87/180 | 17,280 | 207,360 | 7,025,428 | 1,457.42 |
| Task-local memory | 179/180 | 42,677 | 298,572 | 38,234,202 | 2,557.32 |

Relative to task-local research, the full system reduces cumulative observations by 59.02%, measurements by 74.82%, and charged computation by 76.25%. In all 29 instances where both solve every task, full uses fewer resources in all three dimensions. The mean paired differences per six-task instance are $-839.6$ observations (95% interval $[-856.7,-813.1]$), $-7{,}446$ measurement units ($[-7{,}589.6,-7{,}226.4]$), and $-971{,}840$ computation units ($[-1{,}013{,}700,-918{,}906]$). Accuracy differs by one task: the mean solved-count difference is $0.033$ per instance, with interval $[0,0.100]$. The main observed advantage is resource use at nearly equal task accuracy.

The task-local failure occurs on the first changed composition in apparatus 538840. Its confirmation error is $6.11\times10^{-5}$, and terminal MSE is $8.60\times10^{-5}$. It spends 359,841 of its 360,000 computation units while leaving observation and measurement resources unused. On the same task, full succeeds using previously learned components and 101,698 computation units. Thus the independent researcher correctly rejects its unsuccessful candidate but does not complete the task within its research allowance.

The full system's measured training CPU time is 24.17% lower in aggregate, but its paired mean difference of $-20.61$ seconds has interval $[-33.84,2.22]$. Apparatus 548184 accounts for the only full-system revisit and requires 361.77 CPU seconds, compared with 73.22 for task-local research. The strong saving in charged computation therefore coexists with an unresolved runtime advantage and a substantial tail cost.

Passive research solves 87 tasks using less computation than full, while purchasing more instrument measurements. It confirms 155 candidates internally, of which 70 fail terminal evaluation on independent trajectories. This directly exposes limited confirmation coverage under its fixed four-trajectory design. The dense fixed-basis learner fails all tasks, including the conserved-total task, whose response lies in its linear subspace. In this object interface it fits 185 coefficients to 48 training observations and stops after failed confirmation. Its failure therefore reflects the implemented fitting and selection procedure as well as the limited basis available for event responses.

![Four-arm composition results: terminal accuracy, separate research costs relative to task-local learning, and terminal instrument measurements.](figures/composed_offline_comparison.png)

**Figure 2. Accuracy and distinct resource dimensions across 30 apparatus instances.** Panel A counts tasks with terminal MSE below $10^{-6}$. Panel B normalizes each cumulative research resource separately by the task-local total. Panel C reports instrument measurements used during terminal prediction: full uses 0, task-local uses 3,072, and both fixed controls use 552,960. Every arm additionally incurs 46,080 evaluation observations and 46,080 charged evaluation computation units. Measured evaluation CPU totals are 66.37, 54.15, 11.50, and 18.17 seconds for full, dense fixed basis, passive experiments, and task-local memory, respectively. Research and evaluation costs are recorded separately. [Vector PDF](figures/composed_offline_comparison.pdf).

### 5.4 Retaining a response after an observed failure

The previous comparisons establish effects of knowledge availability. We next ask whether a response that has just failed remains useful for subsequent research, and whether recording that failure as conditional-prior feedback independently changes the outcome. These are two distinct interventions.

Each apparatus first shares the same naturally learned $A,B,C$ prefix. We identify $C$ by equality between the newly learned concept's expanded program and the confirmed response, using the measured program if instrument compilation subsequently changed its coordinates. We then freeze this response and purchase one predetermined random trajectory of 24 observations on the changed task. The measurement view is the least expensive public view that supplies the program's variables. This challenge uses acquired responses and the research-time error threshold; it does not use terminal evaluation.

After a measured failure, three continuations begin before any replacement $D$ is learned. **Conditional retention** retains $C$ and records the failure using the ordinary conditional meta-prior update. **Unmarked retention** retains the same knowledge but omits this challenge's feedback record. **Global discard** removes $C$ and its dependent knowledge, preserving its ancestors $A$ and $B$; the removed response can be learned again from new evidence. Normal research and feedback continue in every arm. A retained response is eligible for later paid reassessment; the conditional record does not supply an executable general validity predicate.

The suffix contains the first changed task, a return to the original condition, and a return to the changed condition. Each suffix task starts a fresh one-task session carrying only its arm's persistent library. This removes cross-task session caches and resource borrowing from all three arms. The common challenge is charged to every arm and deducted from the first changed task's allowance. Its outcomes support the feedback intervention; they are not supplied as an additional free fitting sample. Prefix, challenge, research, confirmation and terminal evaluation costs are kept distinct.

We specified 30 new apparatus instances before execution, split evenly between the two generator profiles. All cases remain in the analysis, including any that do not learn an exact confirmed $C$ or whose challenge does not refute it. The primary contrast is performance and research cost on the original-condition return. First-revision, changed-return and complete-sequence outcomes show where the effects arise. Paired uncertainty again resamples complete apparatus instances within profile, using 10,000 bootstrap samples.

Every registered case completed, and all 30 challenges refuted the learned response, activating the intervention. Both retention arms solve all 180 tasks and satisfy fresh confirmation on every task. Global discard solves 159 tasks and confirms 160 internally. Table 2 includes the common acquisition prefix once in each counterfactual arm, as well as the challenge and subsequent research.

**Table 2. Retention after measured failure.** Each arm comprises 30 six-task sequences. The first three tasks share a learned prefix; the final three use separate continuations.

| Arm | Tasks passing | Observations | Measurements | Charged computation | CPU seconds |
|---|---:|---:|---:|---:|---:|
| Conditional retention | 180/180 | 17,987 | 73,680 | 8,553,189 | 2,678.81 |
| Unmarked retention | 180/180 | 17,987 | 73,680 | 8,553,189 | 2,581.20 |
| Global discard | 159/180 | 22,484 | 120,864 | 11,651,123 | 2,697.45 |

The primary original-return contrast has equal accuracy: all three arms solve all 30 return tasks. Both retention arms use 24 observations per return, compared with 40 after deletion. Aggregate charged computation on these returns is 30,930 versus 138,476. The paired mean difference is $-16$ observations, with interval $[-16,-16]$, and $-3{,}584.9$ computation units, with interval $[-3{,}868.0,-3{,}273.9]$. All 30 retained responses are selected through direct calibrated calls; deletion instead requires a research program to recover the response.

The quality difference arises principally during the first revision. Retention solves all 30 changed tasks, while deletion solves 10. Of the 20 deletion failures, 16 remain internally unresolved and four pass internal fresh confirmation but fail terminal evaluation. Three other deletion cases pass terminal evaluation while remaining unresolved, demonstrating that accuracy and research completion measure different properties. Each failed first-revision case retains at least 96 observation units, 2,064 measurement units, and 146,261 computation units. Thus these failures occur before exhaustion of the three resource accounts. The compact records do not identify a common stopping cause. On the final changed-condition return, retention solves 30 tasks and deletion solves 29; the remaining deletion failure is unresolved and also retains substantial resources.

Across complete sequences, retention uses 20.00% fewer observations, 39.04% fewer measurement units, and 26.59% less charged computation than deletion. The paired mean computation difference per sequence is $-103{,}264.5$, with interval $[-132{,}874.2,-78{,}700.7]$. The first-revision computation difference alone has interval $[-58{,}778.5,5{,}844.6]$, so its computation saving is less certain than the whole-sequence effect. Measured training CPU is nearly equal: the paired conditional-retention-minus-deletion difference is $-0.62$ seconds, with interval $[-12.93,10.79]$. Each arm additionally uses 46,080 observations and 46,080 computation units for terminal evaluation; retention uses no terminal instrument measurements, compared with 70,656 after deletion.

Conditional and unmarked retention produce exactly equal terminal errors and charged costs in every instance and successor phase. The recorded intervention is active: the former starts with a conditional prior and one counterexample for $C$, while the latter retains its supported prior without that counterexample. Both arms actually call $C$ inside the selected response at all three successor stages in every instance. The measured benefit therefore comes from preserving executable knowledge and its downstream uses. This additional feedback record has no independent measured benefit under the tested protocol; normal later feedback remains active in both arms.

![Successor research costs for conditional retention, unmarked retention, and global discard, separated into the first changed task, original return, and changed return.](figures/scoped_retention_comparison.png)

**Figure 3. Where retention changes later research costs.** Each bar sums one successor phase over 30 apparatus instances. The two retention arms have identical charged costs. These phase totals exclude the shared three-task prefix and common challenge, which are included in Table 2; terminal evaluation is reported separately. [Vector PDF](figures/scoped_retention_comparison.pdf).

### 5.5 A boundary test: changing different components or neither

This study asks whether the retained composition supports different revisions while remaining reusable when nothing changes. After a common naturally learned $A,B,C$ prefix, each apparatus forks into three conditions: unchanged $C=A+0.7B$; a total-response gain change to $1.35A+0.7B$; and a relation-radius change to $A+0.7B_{0.45}$. Each condition is followed by a return to the original response. The gain intervention changes the total component's contribution without changing the event component, physical transitions or instrument readings. The radius intervention changes the event component while preserving the total. Both use the same public task description and measurement interface.

The researcher receives acquired observations through that interface. Branch labels and hidden intervention parameters configure the laboratory; they are not supplied to hypothesis search. All three conditions use matched probe seeds, the same resource allowances as Section 5.4, and fresh one-task sessions carrying their own copies of the prefix library. The agent algorithm is unchanged, and no predetermined challenge is inserted. The retained response can be accepted, refitted, or used in further research according to the observations encountered by the ordinary loop.

We specified 30 new apparatus instances, with 15 in each generator profile. All learning finishes before evaluation. The prefix is evaluated once per apparatus, followed by six successor evaluations, for 270 research outcomes in total. We also evaluate the frozen prefix response on each successor's terminal controls, with its own measurement and evaluation costs. This reference tests whether revision was necessary for prediction; it is a fixed predictor rather than another research algorithm. The recorded actions, transfer attempts and exact called definitions describe how the agent investigates each condition.

All 30 instances completed, with all 90 prefix tasks and all 180 successor tasks passing terminal evaluation and fresh confirmation. The fixed original response passes every unchanged task and every old-condition return, but fails every first successor in both changed conditions. Its mean MSE is $0.2002$ after the gain change and $0.02525$ after the radius change. Accurate successor predictions therefore require adaptation in the changed conditions.

**Table 3. Research behavior under three matched conditions.** Each condition contains 30 current tasks and 30 original-condition returns. Observations and charged computation are mean research costs for the current task; no successor purchases auxiliary instrument measurements.

| Condition | Current tasks passing | Returns passing | Frozen C on current task | Current observations | Current computation |
|---|---:|---:|---:|---:|---:|
| Unchanged | 30/30 | 30/30 | 30/30 | 24.00 | 427.00 |
| Total-response gain 1.35 | 30/30 | 30/30 | 0/30 | 40.00 | 529.33 |
| Event radius 0.45 | 30/30 | 30/30 | 0/30 | 113.07 | 107,719.20 |

All unchanged tasks select the original response through a calibrated call and add no new response concept. All gain-change tasks replay a research procedure and construct a new response using the exact original $A$ and $B$ definitions, without calling the old complete response $C$. They refit the composition using 16 more observations than in the unchanged condition and conduct no relation-boundary comparisons. All radius-change tasks instead perform comparisons between relation candidates, retain the exact original component definitions, and call $C$ inside the repaired response. Preserving a computation thus makes it available; the resulting explanation need not use every retained program.

On all 90 original-condition returns, the selected response again calls the original $C$, with 24 observations per task. The gain change and radius change incur different research costs because they require different operations. Relative to unchanged tasks, their paired mean first-successor computation increases are 102.3 units (95% interval $[98.3,109.0]$) and 107,292.2 units ($[101{,}952.6,113{,}324.9]$), respectively. These are differences between environmental conditions, not comparisons of competing research algorithms.

The common acquisition prefix costs 12,600 observations, 73,800 measurement units and 5,573,655 computation units across all instances. Including this prefix in each five-task branch yields totals of 14,040/73,800/5,599,275 for unchanged, 14,520/73,800/5,603,785 for the gain change, and 16,712/73,800/8,836,713 for the radius change, in the same resource order. Physical execution shares the prefix once, for 20,072 observations, 73,800 measurements and 8,892,463 charged computation units. Independent evaluation uses 69,120 observations and computation units, while the fixed-response reference adds 46,080 of each; both use zero instrument measurements.

### 5.6 What the composition studies establish

Across the three mechanism protocols, learned programs are used in later tasks, counterexamples lead to revised responses, and earlier successful knowledge can remain usable after revision. The independent-source experiment identifies two positive marginal contributions. The composition experiment adds preservation of an unchanged definition within a repaired response and a positive total effect of retaining source $A$. The subsequent four-arm study establishes substantial charged-resource savings relative to the same researcher restarted on every task, with nearly equal terminal accuracy. The retention intervention shows that even an observed-failing response can remain useful for repair and return. Its unmarked-retention control locates the gain in retained executable knowledge; it does not establish an additional benefit from the tested conditional-feedback update.

The component-change study extends this result to a different coefficient revision and to the decision to reuse an unchanged response. It connects the form of the observed change to distinct research behavior within the available search families. Together, the studies establish a concrete retention-and-revision mechanism: learned computation survives a failed application, participates in successful subsequent research, and remains available when earlier conditions return.

The resource results answer different questions. Source deletion measures knowledge availability and its downstream learning consequences. The task-local comparison evaluates complete research with and without cross-task knowledge and resource allocation. The fixed-library results show that the usefulness of a particular solver changes across task families. Direct calls and method replay distinguish two ways that accumulated knowledge can influence a later task. CPU measurements expose costs that the internal computation accounting does not fully capture.

### 5.7 Joint revision of expressions, representations and research procedures

The second instantiation tests whether several objects of revision can participate
in one persistent research process. Thirty new apparatus instances, balanced over
quadratic, cubic and absolute-value response families, each undergo eight tasks:
source acquisition, changed coordinates, a local scope change, a hidden input,
changed instrument roles, stochastic responses, simplification and return. The
persistent and task-local arms have identical expression primitives, candidate
search and per-task budget ceilings. In particular, the task-local arm can learn
conditional responses from scratch. Five additional continuations per instance
remove specific knowledge or measurement access from a common acquired prefix.
The frozen campaign contains 630 research runs; all 30 instances completed.
Appendix D specifies the protocol and maps the eight conceptual capabilities to
their operational tests.

**Complete research sequences.** Persistent research passes independent evaluation
on 238 of 240 tasks, versus 218 for task-local research. Requiring both this
external criterion and successful research-time confirmation gives 232 versus
215. The latter paired difference is 0.567 tasks per apparatus, with a 95%
family-stratified bootstrap interval of $[0.333,0.800]$. Source acquisition,
coordinate transfer, simplification and nonlinear return pass in both arms on
all 30 instances. After instrument roles change, terminal passes are 30 versus
20; on the stochastic task they are 29 versus 18. The scope-change task favors
task-local search slightly, with 29 versus 30 passes.

**Table 4. Whole-sequence joint-growth results.** Costs include all eight tasks,
including source acquisition, unsuccessful search and confirmation. Confirmation
and terminal evaluation are separate outcomes; neither count alone is their
intersection.

| Outcome | Persistent | Task-local |
|---|---:|---:|
| Independent evaluation passes | 238/240 | 218/240 |
| Research-time confirmations | 234/240 | 218/240 |
| Both confirmation and independent evaluation | **232/240** | **215/240** |
| Observations | 23,780 | 23,840 |
| Measurement units | 21,840 | 22,800 |
| Charged computation | 17,068,659 | 22,000,724 |
| Process CPU seconds | 177.18 | 150.26 |

Charged computation decreases by 22.42%, with a saving in every apparatus. Its
paired mean difference is $-164{,}402$ units per sequence, with interval
$[-189{,}932,-139{,}616]$. Observation and measurement totals decrease by 0.25%
and 4.21%, but their paired intervals include zero. Actual process CPU increases
by 17.92%; its paired difference is $+0.897$ seconds, interval $[0.597,1.191]$.
The empirical advantage is therefore predictive success and charged computation,
with an actual runtime penalty.

**Which retained knowledge helps?** In the new-expression branch, the target adds
a cubic term to the acquired source. Retaining and deleting program knowledge
both solve and confirm all 30 tasks. Retention changes the mean computation by
$-2{,}303$ units, interval $[-6{,}019,188]$; only seven instances save computation.
The new abstraction is usable in a new composition, but its independent resource
benefit is heterogeneous. For changed coordinates, both arms again solve and
confirm all tasks; program retention reduces computation in all 30 pairs, by a
mean 5,568 units with interval $[2{,}373,8{,}399]$.

The procedure intervention retains the exact same program library and clears
only learned methods before instrument names, units and ordering change. Both
continuations solve and confirm 30/30 tasks. Keeping the methods reduces
observations from 5,088 to 4,028 and computation from 5,219,559 to 3,944,154;
both purchase 7,200 measurement units. The paired mean observation difference is
$-35.33$, interval $[-45.47,-25.20]$, and computation difference is $-42{,}514$,
interval $[-49{,}246,-35{,}687]$. These contrasts share an actually learned prefix,
so including its acquisition cost leaves the absolute savings unchanged. The
transferred procedures generate new fitting subquestions for the current
instruments instead of copying the previously successful index. This supplies
separate evidence that learned research steps can reduce later work.

**Scope and observable distinctions.** Learned scope revisions pass 29/30 tasks,
whereas the frozen source response passes none. On the 27 cases containing an
identifiable conditional branch, applying the new branch everywhere also fails
every case; three equivalent hinge expressions do not support that diagnostic.
Both old and new behavior are needed in this constructed setting. The remaining
scope failure passes internal confirmation but has terminal MSE
$5.00\times10^{-5}$, above the fixed $10^{-5}$ criterion.

When a hidden input affects the response, allowing paid instruments yields
30/30 confirmed terminal passes; withholding them yields 0/30 terminal passes.
Access costs an additional 47.2 observations, 240 measurement units and 50,656
computation units per successor on average. All controls-only runs internally
confirm stochastic descriptions. Such descriptions can represent unresolved
variation, but cannot supply the full-response point predictions required by
this deterministic task. This contrast demonstrates the value of the available
distinguishing measurement, including its acquisition cost.

**Uncertainty, simplification and return.** The researcher introduces variance
models in all 30 stochastic tasks and selects coordinate-dependent variance in
19. Against a constant variance estimate formed from the same paid repeats, the
mean variance-estimation error decreases from 0.01461 to 0.01103. Normal intervals
labelled 95% cover 91.09% of fresh responses, compared with 92.81% for the constant
estimate. Better variance fitting therefore does not imply better coverage. Six
persistent runs meet the external stochastic criterion but do not complete
internal confirmation before exhausting the budget; one confirmed run misses
the external mean-error criterion. On the linear task, learned simple programs
and dense cubic fits to the same paid data both pass all 30 instances. Executing
the learned programs on terminal data takes 69,120 primitive steps, versus
194,560 for the dense fits. All 30 subsequent returns recover the nonlinear
response. This is a prediction-execution contrast, not a second complete research
algorithm comparison.

![Joint-growth task outcomes, paired acquisition differences and whole-sequence resource trade-offs.](figures/joint_research_growth.png)

**Figure 4. Joint growth and its costs.** **a**, Independent-evaluation pass rates
over 30 instances per stage; Table 4 separately gives confirmation and their
intersection. **b**, Mean paired observation differences and 95% seed-bootstrap
intervals for knowledge/access interventions and the whole sequence. Measurement
access improves accuracy while increasing acquisition, so its positive cost
is not an equal-accuracy inefficiency. **c**, Whole-sequence resource ratios;
charged-computation savings coexist with higher CPU time.
[Vector PDF](figures/joint_research_growth.pdf).

### 5.8 Knowledge availability and marginal value

The integration studies connect learned procedures to the legacy controller's
acquisition, confirmation and revisit operations (Appendix E). A continuation
comparison starts both arms from identical post-production states with only
32 observation rows left. Selecting a candidate on the consumer's already paid
data and purchasing one fresh 24-row confirmation completes 12/12 previously
unfinished consumers, versus 0/12 for the earlier controller. Whole-sequence
charged computation increases by 6.57%, within the original ceilings.

All successful offers derive from earlier prefix concepts. A post-hoc
intervention retains or removes the newly produced concept while preserving
those earlier sources, procedures and incurred costs. Both arms produce the
same final programs and confirmations in all 12 cases; retaining the new product
adds 1,920 screening computation units per case. The controller therefore makes
useful older knowledge available to an unfinished task, while this additional
product provides no marginal predictive benefit. This contrast distinguishes
the value of resuming research from the value of the new knowledge produced
before that resumption.

### 5.9 An observed hypothesis life cycle

Apparatus 812676 from the confirmation profile of Section 5.5 supplies a retrospective illustration of the recorded mechanism. The acquired library contains a total-response program $A$, an event-response program $B_r$, and their learned composition $C=A+0.7B_r$. On the changed-boundary task, the first eight-row validation of the retained $C$ gives MSE $0.0463443$. The frozen $C$ also fails independent evaluation, with MSE $0.0280351$. The discrepancy motivates further research with the old computation still available.

The nested-relation generator constructs a correction inside the acquired composition. Its schematic form is

$$
D=C+0.7(B_{r'}-B_r).
$$

The actual fitted program retains the original definitions and adds an executable correction. Its compiled research record links preserving the composed call, revising the internal relation, and refitting the answer. Those nodes describe completed operations: the initial retained fit has loss $0.0201398$, while the final corrected fit has loss $1.69\times10^{-19}$. This is a concrete decomposition of the repair within a prescribed operation family.

The radius remains uncertain after a successful fit. The controller therefore proposes experiments on which competing relation boundaries disagree, acquires their outcomes, and fits again. Recorded refinements narrow the remaining interval as follows:

$$
[0.4499800,0.4575664]\ \longrightarrow\ [0.449999655,0.450000288].
$$

These bounds describe candidates consistent with acquired evidence; they are not statistical confidence intervals. After 100 research observations, the selected program is frozen. A further 24 observations confirm it at loss $2.01\times10^{-19}$, and an independent 256-row evaluation gives MSE $1.32\times10^{-19}$. The result enters the library as `concept_0004`.

When the original mechanism returns, the selected response directly calls the acquired $C$ (`concept_0003`). No response concept is added. This task uses 24 observations in total, including eight for fresh confirmation, and passes the independent test at MSE $1.52\times10^{-19}$. Matching is operational: a stored program is bound to the current inputs and compared on current evidence. The experiment thus shows new hypothesis construction, an experimentally resolved internal uncertainty, fresh confirmation, and later use of an earlier hypothesis. The [complete case record](../artifacts/paper_revision_choice_20261005/case_812676_confirmation.json) contains the candidate failures, acquired probes, parameter refinements and selected calls.

Direct reuse of the newly acquired revision is visible in the separate six-task trace for [apparatus 26003](../artifacts/composed_local_cycle_20261004_r3/case_26003_development.json) in Appendix C.3. Task 4 constructs $D$; task 5 calls $C$ when the original condition returns; task 6 calls $D$ when the changed condition recurs. Both return tasks acquire 24 observations and introduce no new response concept. All six tasks pass fresh confirmation and independent evaluation. The full cohort's 11/12 direct-call criterion and 72/72 terminal result, reported in Appendix C.3, distinguish this successful illustration from the aggregate outcomes.

![A recorded hypothesis revision followed by reuse, and the contraction of its relation-parameter interval under acquired evidence.](figures/hypothesis_revision_trace.png)

**Figure 5. From a failed hypothesis to a confirmed revision and later reuse.** **a**, In apparatus 812676, the acquired composition $C$ fails on the changed task. A nested relation correction produces $D$, which passes fresh confirmation and independent evaluation. When the original condition returns, the agent directly calls $C$. **b**, Recorded parameter refinements reduce the width of the remaining relation-radius interval as observations accumulate. The shaded segment marks 24 further observations acquired after freezing $D$; they confirm the chosen program rather than refine the interval. Returning to $C$ uses a separate task's 24 observations, including eight for fresh confirmation. [Vector PDF](figures/hypothesis_revision_trace.pdf).

### 5.10 Online recursion and frozen structural transfer

We prespecify six development and 30 separate evaluation source chains.
Each chain first learns one source. Three researchers then receive identical
copies of that acquired state for four independent continuations: unchanged
structure and surface, unchanged structure with a changed surface, changed
structure on the old surface, and changes to both. Surface changes rename and
permute controls and change instrument calibration. Structural changes replace
the response primitive. Each source-to-target composition is new to that source
history; all four primitives remain available in the public research language.
The four consumers never share newly learned knowledge.

Each task allows 24 observations and one million candidate-prediction
computations, equally across arms. Correct executable hypotheses and independent
predictions are the primary outcomes. Three new response observations confirm a
frozen candidate; 256 independent observations evaluate it after research.
Both use an MSE threshold of $10^{-10}$. The 30 source tasks all pass. Each
continuation arm passes confirmation and independent evaluation on 120/120
tasks, with no false confirmation and no runtime failure (Table 5).

**Table 5. Online inquiry under matched access to evidence.** Outcomes concern
120 continuations per arm, grouped in 30 independent source chains. The common
source acquisition is shared by the intervention. Reuse counts actual calls to
retained definitions after reassessment. Rewrites occur before the changed
experiment is executed.

| Researcher | Confirmed and independently correct | False confirmations | Retained-definition calls | Online method rewrites |
|---|---:|---:|---:|---:|
| Recursive, revisable method | 120/120 | 0 | 60 | 47 |
| Flat active research | 120/120 | 0 | 60 | 0 |
| Recursive, frozen method | 120/120 | 0 | 60 | 0 |

The 60 same-structure continuations in each arm use the retained response
definition, including all 30 with a changed surface. The 60 changed-structure
continuations identify a replacement instead. Recursive research rewrites a
retained step on 23 same-structure/new-surface and 24 changed-structure/new-surface
tasks. No rewrite occurs in either unchanged-surface condition. All four
conditions pass in every arm, so this study establishes appropriate behavior
within the finite language without an accuracy advantage over flat research
or frozen procedures with fallback.

Figure 6 follows the first registered chain. The parent asks for a coordinate
binding, whose child investigates calibration. After the coordinate answer
identifies the second control as the driver, the retained response probe
$(-0.6,-1)$ aliases the quadratic and absolute-value hypotheses: both predict 1.
Before taking the next observation, the method changes the probe to
$(-1,-0.6)$, which makes all four remaining laws predict different outcomes.
Thus the supporting answer participates in choosing an informative parent
experiment. A separate deterministic intervention test changes only the returned
calibration evidence: the parent coordinate interpretation and response probe
change, and fresh confirmation rejects the resulting incorrect branch. This
intervention is a mechanism test, separate from the 30-chain performance cohort.

![Online supporting questions and their effect on the parent experiment.](figures/online_support_inquiry.png)

**Figure 6. Supporting evidence changes the experiment before execution.**
**a**, In chain 79101013, questions open before the observations that resolve
them, and answers return to the parent. The star marks the method rewrite before
the next response probe. **b**, Candidate predictions explain why that rewrite
restores a distinction lost by the old step. The table is evaluated from the
recorded candidate language and returned binding, not from a hidden mechanism
label. [Vector PDF](figures/online_support_inquiry.pdf).

Resource totals describe a trade-off rather than the primary result:
recursive, flat and frozen-method continuations use 852, 647 and 883
observations, respectively, and 388,496, 735,048 and 390,108 charged computations.
The recursive organization acquires more observations than the flat control
while scoring fewer candidate-action combinations. The [complete protocol and
results](../artifacts/online_inquiry_20261005_main/summary.json) retain all
registered chains and all three controls.

### 5.11 Common-model study and task saturation

The completed comparison uses gpt-6-luna at medium reasoning effort through
Codex authenticated with ChatGPT. All 90 registered source-to-target chains
finish their research loops and independent evaluation. Each task has the same
24-observation, 16-call and 300,000-reported-token ceilings, with three paid
fresh observations required for confirmation. The larger token allowance follows
a retained development trial in which repeated input overhead prevented
submission; cached input remains charged to the common allowance.

| Knowledge condition | Source successes / 30 | Target successes / 30 |
|---|---:|---:|
| Code/ReAct | 29 | 29 |
| Textual memory | 30 | 29 |
| Executable research knowledge | 26 | 26 |

All submitted models also pass the 256-point independent evaluation. One Code
source times out and its target is unattempted. Textual memory has one target
token-budget failure. Hegel has four source token-budget failures; its target
failures comprise three exhausted confirmation reserves followed by token
exhaustion and one model-capacity error. On the 29 targets attempted by all arms,
Code solves 29, textual memory 28 and Hegel 25. Hegel performs 36 successful
inherited experimental-program calls on targets, including 15 unchanged calls,
but this execution does not translate into superior task completion.

The [completed run](../artifacts/online_inquiry_luna_subscription_20261005_main_30/RUN_STATUS.md)
contains 894 calls and at least 20,395,181 reported tokens; 70.3% of known input
tokens are cached. Earlier GPT, Qwen and interrupted Gemini cohorts remain
separate service/interface records, not additional samples in this comparison.

This task is saturated for ordinary code investigation: it publicly enumerates
four response laws, and direct response probes can identify them without
resolving the instrument. The completed comparison establishes no advantage for
executable research knowledge. We retain it as a negative comparative result
and a bounded execution example. A subsequent development environment expands the
legal trajectory horizon: two physical explanations agree on every legal short
experiment and diverge on longer experiments.

### 5.12 A measured child repairs a temporal representation

The temporal apparatus expands its legal horizon from three to 24 steps while
preserving its physical parameters. A bounded offline controller first extends
a quadratic control-history model. When paid trajectory error persists, it
opens an instrument question before acquiring auxiliary readings. Existing
finite-history and continuous-state fitters propose instrument explanations;
their predicted disagreement selects the next measured experiment. The selected
instrument program becomes an executable coordinate of a general quadratic
parent response, evaluated from control history alone.

Across 12 new paired apparatuses, all 24 targets pass 48 fresh confirmation rows
and a separate 256-row evaluation at MSE below $10^{-10}$. All 12 finite cases
use history revision without measuring the instrument. All 12 recurrent cases
acquire 15 short measured rows followed by a four-step distinguishing probe;
the returned state program changes the parent predictor. Their maximum terminal
MSE is $7.36\times10^{-17}$. A three-lag quadratic refitted on exactly the same
preconfirmation response rows fails all 12 recurrent cases, with MSE
0.00762--0.316. Target acquisition uses 144 rows for finite cases and 163 rows
plus 57 measurement units for recurrent cases. The
[prospective records](../artifacts/temporal_recursive_research_20261005/RUN_STATUS.md)
contain the acquired trajectories and selected programs. Each final predictor
is fixed before confirmation and terminal evaluation. This experiment uses no
LLM calls.

A subsequent [matched-repertoire comparison](../artifacts/temporal_matched_repertoire_20261006/RUN_STATUS.md)
uses 30 new paired apparatuses. A flat adaptive researcher has the same
instrument candidates, response basis, experiment planner and observation limits.
It fits instrument-conditioned response candidates on 63 paid rows and selects
on two other complete trajectories totaling 48 rows. Both policies confirm and
independently solve 60/60 targets, with zero false confirmations. In every
recurrent case, flat selects a successful response before the instrument alone
is uniquely supported. It uses 159 target rows and 45 measurement units, compared
with 163 and 57 for recursive research, while making three response fits instead
of one. Both policies use 144 rows and no meter readings on finite cases. Thus
the common representation repertoire achieves the same predictive success
without separate child confirmation. The computation cap covers instrument
search; these results do not compare total computation.

### 5.13 LLM selection of shared research operators

We expose the individual history, instrument, response-fitting, disagreement
and assessment operations to all three gpt-6-luna knowledge conditions. The
LLM chooses acquisitions, operator calls, returned predictors and submission;
the operators do not execute a complete research policy. One new apparatus draw
supplies finite and recurrent continuations, each preceded by short-source
learning. Each task allows 240 observations, 24 model calls and 600000 reported
tokens checked between calls, with 48 fresh confirmation rows.

All six sources pass. Code and textual memory each confirm and independently
solve 2/2 targets, while Hegel completes 0/2 targets before its token boundary.
Hegel actually re-executes a retained short experiment. On the recurrent target
it purchases a proposed distinguishing measurement but does not subsequently
assess or refit the instrument. Each unsuccessful target has a rejected fresh
confirmation; no submitted model is falsely confirmed. No recorded question
has a parent ID. The [complete pilot](../artifacts/temporal_shared_operators_luna_20261006/RUN_STATUS.md)
contains 98 calls and 3668858 reported tokens; 43.9% of known input is cached.
The textual recurrent target finishes at 865992 tokens because its final call
exceeds the between-call threshold. Interface repairs made afterward retain
long training trajectories and compact large source files in the prompt;
the reported model outcomes are unchanged.

A subsequent [development retest](../artifacts/temporal_evidence_continuation_luna_20261006/RUN_STATUS.md)
adds explicitly agent-authored post-acquisition operator sequences. One new
paired apparatus yields 2/2 correct sources and 2/2 correct targets in every
arm. Hegel's recurrent target buys two six-step meter experiments and immediately
assesses the inherited instrument candidates: the finite candidate is refuted
(MSE 0.008481), while the recurrent candidate is supported (7.12e-19). The LLM
then replaces the inaccurate parent response (paid MSE 0.001031) with a response
using that coordinate; fresh-confirmation and terminal MSE are 1.18e-17 and
9.26e-18. On the finite target, the frozen source predictor succeeds unchanged.
The recurrent target needs 72 observations versus 96 for either control;
including source learning reverses that difference, to 192 versus 174. Across
both sequences Code, text and Hegel use 342, 378 and 360 observations and 27, 19
and 33 model calls, respectively. All 79 calls report 1905425 tokens, with
50.34% of input tokens cached.

The retest establishes successful use of measured evidence to select a retained
supporting hypothesis and revise its dependent response. Its limitations are
specific: the new apparatus and multiple interface changes do not isolate an
effect of continuations, and both ordinary controls also succeed. The LLM writes
continuations in query arguments but does not incorporate them into its retained
research program. Its online questions remain roots; a later dependency manifest
does not establish recursive problem decomposition. That pilot does not test
learning and transferring the complete repair procedure.

### 5.14 Learning executable research procedures

We distinguish an acquisition plan from a research procedure that analyzes its
observations, chooses whether to revise an earlier explanation, and submits a
predictor. All procedures below are authored by `gpt-6-luna` using the shared
numerical operators. The initial donor is the successful recurrent target's paid
experience; terminal evaluation is excluded from learning.

Early development establishes the difficulty of preserving a useful procedure.
[Retrospective consolidation](../artifacts/temporal_method_learning_retest_luna_20261006/RUN_STATUS.md)
produces interface errors and then a short acquisition-only plan. Paid practice
executes an inherited program, revises it and freshly confirms its own exported
predictor. The [first two attempts](../artifacts/temporal_method_practice_feedback_luna_20261006/RUN_STATUS.md)
fail confirmation at MSE 0.646 and 0.261 without revising the changed instrument;
a [later attempt](../artifacts/temporal_method_dependency_feedback_luna_20261006/RUN_STATUS.md)
refutes that instrument but exhausts observations before rebuilding it.

An [expanded practice](../artifacts/temporal_method_dependency_extended_luna_20261006/RUN_STATUS.md)
refutes the inherited instrument at MSE 0.01497, then learns a procedure that fits
a new instrument and its dependent response. Its predictor passes 48-row fresh
confirmation at MSE 4.62e-19. Practice uses 288 observations and 11 model calls.
The frozen procedure acquires three 24-step measured trajectories and refits on
current data, without carrying practice-fitted coefficients into transfer. A
retention/deletion intervention restores the original predictive snapshot and
shares the new verbal lesson. Both conditions confirm and independently solve
2/2 targets. Retention/deletion use 264/216 observations, 30665/4348 charged
computation and 15/16 model calls. The recurrent retained agent copies the exact
learned acquisitions and fitting steps into an inline query. Thus this case
establishes behavioral transfer with equal accuracy and higher data/computation
costs when the procedure is available.

Two [conditional-practice attempts](../artifacts/temporal_conditional_method_retest_luna_20261006/RUN_STATUS.md)
then allow programs to read actual operator feedback and analyze existing paid
records. Neither publishes a freshly confirmed method. Repairing inconsistent
feedback formats permits different stages of one unchanged source to execute,
but its decisions concern record counts and result presence. The last source edit
is unexecuted. These attempts also begin by refitting the apparatus, removing the
initial contrast between valid and invalid prior knowledge.

We therefore introduce paired practice. Both cases receive identical earlier
calibration and the same public interface; the mechanism remains unchanged in
one and changes in the other. A program starts with empty current observations,
chooses experiments and analysis steps, and explicitly declares its final model.
Ordinary fresh confirmation either accepts it or returns paid counterexamples.
The next invocation uses this actual feedback. No LLM chooses intermediate actions
once the program is frozen, and no initial refit is supplied by the host. The
first paired learner confirms both cases in five model calls. Its fixed program
also solves four frozen cases across two earlier calibrations, but always buys
72 measured observations and refits: predictive success alone leaves reuse untested.

A further learning task explicitly requires confirmed reuse of the unchanged
predictor in one case and confirmed revision in another, without revealing which
case supports either choice. Its proposed prior-first submissions expose a runtime
error: the host requires paid training rows before allowing fresh confirmation.
We interrupt this run and repair that precondition. An initial submission now
skips only the absent training diagnostic; actual 48-row confirmation remains
mandatory. We then [freeze the last recorded LLM-authored candidate](../artifacts/temporal_paired_candidate_replay_20261006/RUN_STATUS.md)
and replay it from empty workspaces, without restoring the erroneous feedback.

The unchanged program first submits the inherited predictor. On both unchanged
mechanisms, fresh confirmation passes and execution ends with that exact predictor,
using 48 observations, no meter readings and 48 charged computation units per case.
On the changed mechanisms, the inherited predictors fail at MSE 0.5322 and 2.3136.
The program then purchases 48 measured observations and fits instrument and response
hypotheses. It selects a response using paid instrument consistency and whole-trajectory
validation, then requests a new confirmation. Both revisions pass, with terminal
MSE 7.25e-18 and 4.62e-19. All four cases pass confirmation and a separate 256-row
terminal evaluation, performed only after all four executions stop. The second
calibration was unavailable during authoring and reverses which mechanism is
unchanged. Its supplied predictor role is used without editing the procedure.

Across these four matched cases, the prior-first procedure uses 384 observations,
288 measurement units and 15820 charged computation, compared with 480, 864 and
93828 for the earlier always-refitting program. Both execute without LLM calls and
solve 4/4 cases. Changed cases cost 144 observations each, versus 120 for fixed
refitting, because the rejected prior test is also paid; unchanged cases cost only
48. These are observed execution costs of two development candidates. Acquisition
patterns also differ, so the comparison does not isolate knowledge retention's
causal contribution.

A [subsequent online attempt](../artifacts/temporal_paired_online_repaired_luna_20261006/RUN_STATUS.md)
uses the repaired runtime but ends after 17 calls and 1,022,484 reported tokens
without publishing. It achieves three confirmed revisions and one confirmed reuse
across different program versions, repeatedly changing executable branches between
trials. The final source is confirmed on only one case. We therefore expose each
current program's completed and missing practice cases directly in write/practice
feedback. A [development continuation](../artifacts/temporal_paired_coverage_continuation_luna_20261006/RUN_STATUS.md)
starts from the last actual candidate, with new empty practice records. It confirms
a revision at MSE 1.90e-18 and prior reuse at 1.06e-17, but an intervening edit again
leaves the current source confirmed on only one case. Its sixth provider call fails
because the selected model is at capacity. It reports 194,933 tokens before that
failure, whose usage is unavailable. Neither attempt supplies a published method
or a completed retention/deletion comparison.

A third phase restores the saved candidate and paid trial history after the
provider interruption. Confirmation on case_b remains valid for that exact source;
the earlier case_a result belongs to another version and is excluded from its
coverage. The LLM performs two new case_a trials without changing the source, then
publishes the procedure. This phase uses three calls, 91,617 reported tokens,
240 observations and 44,514 charged computation units. The published program
assesses the inherited instrument on current measured trajectories. Supported
instrument hypotheses lead to reuse of the unchanged predictor; refuted hypotheses
lead to instrument and response refitting, followed by fresh predictive confirmation.
It passes all four frozen cases across the two calibrations, with maximum terminal
MSE 8.77e-16 and no execution-time LLM calls. Each case uses 120 observations and
216 measurement units; unchanged/changed cases use 192/22,257 charged computation
units respectively. This completes publication through continued development,
including the earlier failed attempts. The [saved results](../artifacts/temporal_paired_state_resume_luna_20261006/summary.json)
keep new-phase costs separate from retained trial evidence.

The completed frozen replay establishes an executable prior-first confirmation-and-repair
procedure in this bounded family. Online publication is now complete, while the
added state feedback has no isolated efficacy estimate. Reuse and revision are
explicit task objectives, roles are supplied, and numerical fitting families remain
developer authored. The earlier frozen program reacts to predictive failure without
independently localizing its cause to the instrument; the subsequently published
procedure assesses measured instrument support before selecting reuse or refitting.
Two previously studied calibrations do not establish population-level reliability,
open-ended method invention, or stable long-horizon net benefit.

### 5.15 Availability of a learned research procedure

We retain or delete the published procedure while holding original predictive
memory, instrument roles and its final verbal lesson fixed. Both conditions use
the same Luna model and single-invocation procedure interface. The LLM can call,
modify or reconstruct a procedure, or choose individual research actions. Four
paired cases cover stable and changed mechanisms under the two calibrations.
Each run has limits of 240 observations, 48-row fresh confirmation, 100,000 charged
computation units, 24 model calls and 600,000 reported tokens, with the token limit
checked between calls. All eight agents stop before terminal evaluation.

The initial pilot and a subsequent fresh-agent development rerun are reported
separately below. The rerun keeps the published source, predictive memory, lessons,
tasks, seeds, model and limits fixed while giving both conditions the repaired
submission interface described below. Each task starts with no current observations.

| Measure | Pilot retained | Pilot deleted | Rerun retained | Rerun deleted |
|---|---:|---:|---:|---:|
| Confirmed and correct | 3/4 | 3/4 | 4/4 | 4/4 |
| False confirmations | 0 | 0 | 0 | 0 |
| Model calls | 24 | 34 | 18 | 16 |
| Reported tokens | 1,378,181 | 1,856,123 | 794,795 | 848,375 |
| Observations | 408 | 396 | 408 | 372 |
| Measurement units | 792 | 612 | 648 | 396 |
| Computation units | 42,363 | 12,373 | 30,224 | 25,028 |

The pilot failures occur on different changed mechanisms. The deleted failure
does not refit a refuted instrument and incurs a costly output-decoding failure.
The retained failure selects a strong candidate but repeatedly edits its manifest
after a missing-path rejection, never retrying confirmation. We repair direct
consumer submission to register an explicitly selected model path while preserving
existing research entries and paid fresh confirmation. Separate post-hoc completion
of the unchanged, previously selected candidate passes 48 new rows at MSE 4.23e-20
without an LLM call or refit. The original pilot remains 3/4 in each condition.

In the rerun, all eight agents autonomously confirm a prediction that also passes
terminal evaluation. Retained agents invoke the unchanged learned procedure ten
times across all four tasks; its final invocation submits every accepted model.
Both stable cases reuse the original predictor. On one changed case, a response
fit initially lacks a separate validation trajectory; the LLM acquires another
trajectory and the same procedure completes the revision. Deleted agents submit
all four refitted predictors through direct finish. Neither condition edits a
manifest or encounters a submission rejection, provider failure or output-decoding
failure. The formerly blocked retained case confirms at MSE 1.23e-19 and obtains
terminal MSE 1.69e-19.

Retention uses 6.3% fewer reported tokens in this rerun, but more model calls,
observations, measurements and charged computation. Python execution totals are
2.78 versus 6.40 seconds. Thus learned-method use and autonomous completion are
observed without an overall comparative advantage. The original
[pilot record](../artifacts/temporal_paired_retention_luna_20261006/RUN_STATUS.md)
and [rerun record](../artifacts/temporal_paired_submission_retest_luna_20261006/RUN_STATUS.md)
retain failed learning costs in the common ledger; their exact token total remains
unknown. The same-case rerun is a development result, and stochastic changes in
research trajectories prevent attributing the improvement solely to the repair.
Both calibrations were previously studied; unseen-task generalization remains open.

### 5.16 Revising a research method from a parent counterexample

The preceding tasks usually change instrument and response together. We separate
them using independently seeded response parameters, preserving the old default
laboratory behavior and public interface. The original two donor memories and
published procedure remain fixed. In an initial frozen screen, the procedure
solves both stable cases and all 24 new-instrument cases, but fails on both
response-only changes: 26/28 overall, with no false confirmations. Each failure
correctly supports the old instrument, then submits the same incorrect parent
predictor three times despite fresh counterexamples, spending 216/240 observations.
Instrument validity does not establish the validity of a response using it.

A preselected six-pair live retention/deletion pilot completes three cases per arm,
with one retained confirmation-budget failure and five service-capacity interruptions.
All six confirmed outputs pass terminal scoring. One retained agent recovers by
bypassing the procedure and refitting the response from paid evidence. Another
buys 96 extra measured rows and refits both components, leaving too little budget
for confirmation despite low candidate validation error. Neither rewrites the
method. These trajectories distinguish local recovery from a reusable procedural
revision. [Initial screen and live pilot](../artifacts/temporal_calibration_transfer_luna_20261006/RUN_STATUS.md).

We then explicitly ask Luna to revise the executable method from one recorded
response-only failure. The author receives the first donor's original knowledge,
the old source, and that trial's paid observations and feedback. It chooses how
to edit and practice on three opaque cases: stable, joint change and the observed
response change. Publication requires the same final source to pass fresh
confirmation on all three, with actual unchanged-predictor reuse and revision.
The second donor and transfer outcomes are unavailable during authoring.

Luna makes one source edit, practices all three cases and publishes in five calls,
using 166,174 reported tokens, 408 observations, 648 measurement units and 25,233
charged computation units. The new source reads failed predictive-confirmation
feedback. In response-only practice it first supports the original instrument
on 72 measured rows, then refutes the old parent prediction on 48 fresh rows.
It retains that instrument and fits the response using the 120 already paid rows,
with whole-trajectory validation, then passes another 48-row confirmation at
MSE 4.57e-18. No instrument refit or additional measurement occurs in this repair.

We freeze the old and newly published methods on the same 18 transfer tasks,
using identical original numerical priors, confirmation seeds and limits. The
response-only tasks use four new response seeds under each of two priors; four
new instrument seeds each supply an instrument-only and a joint-change task.
Only the method and its description transfer from authoring. All subsequent
execution is programmatic, with no LLM selecting intermediate actions.

| Calibration change | Original method | Revised method |
|---|---:|---:|
| Stable | 2/2 | 2/2 |
| Response only | 0/8 | 8/8 |
| Instrument only | 4/4 | 4/4 |
| Instrument and response | 4/4 | 4/4 |
| Total confirmed and correct | 10/18 | 18/18 |

Both methods produce zero false confirmations. Every revised response-only run
executes the same branch: support the instrument, refute the old prediction,
refit only the response from existing evidence, and confirm on 48 new rows.
Terminal MSE ranges from 5.79e-18 to 8.59e-16. All ten other tasks retain exactly
the old method's fresh and terminal scores, including direct reuse on stable
cases. Frozen observations fall from 2,928 to 2,544; measurement units remain
3,888 and computation rises from 183,048 to 200,712. Authoring costs are additional.
This establishes a measured counterexample changing an executable research rule,
followed by preserved earlier behavior and successful transfer of the revision.
[Method-learning and matched frozen results](../artifacts/temporal_counterexample_learning_luna_20261006/RUN_STATUS.md).

### 5.17 Evidence scope after a calibration checkpoint

We next test a limitation of the learned trigger: supporting measurements can
precede the change that invalidates a parent prediction. The apparatus now
discloses a possible recalibration after 72 observations and asks for predictions
under its subsequently fixed physics. Stable, response-only, instrument-only and
joint-change cases share their public interface and initial observations. The
change occurs between whole trajectories. Instrument and response operators can
fit selected complete paid records, retaining the remaining history and its costs.

The published method still passes all 18 preceding stationary tasks and both
stable checkpoint controls, but fails all 12 changed-checkpoint cases: 20/32
confirmed and terminal-correct, with zero false confirmations. Its earlier
instrument assessment concerns the pre-change evidence. Repeated response fits
do not establish whether that instrument remains applicable afterwards.

An initial author phase supplies a real instrument-change failure and four opaque
practice cases. Luna makes 11 source edits, four inspections and five practices
in 20 calls, reaching the 1,208,226-token ceiling without publication. An earlier
source confirms the stable case; four response-change practices fail. The last
two acquire new meter readings and fit the instrument on those records, but
response training and refitting still mix earlier and current labels. Their
fresh-confirmation MSEs are 0.00455 and 0.00507. No executed program reassesses
the old instrument on the new measurements. This phase uses 1,032 observations,
1,512 measurement units and 58,290 charged computation units, without a provider
error or timeout. A separate programmatic execution of the unedited final draft
on the four practice cases confirms only the stable case. This diagnostic replay
does not publish the draft or establish transfer.
[Failure-selectivity experiment](../artifacts/temporal_failure_selectivity_luna_20261006/RUN_STATUS.md).

A new author phase starts from the last successfully published method and one
failed practice from that run. Its feedback now exposes a compact chronology of
the already paid evidence: support on rows 0–71, a failed prediction on rows
72–119, new measurements on rows 120–191, and response refitting on rows 0–191.
The recorded failed source is identified separately from the starting method.
No replacement program or hidden change label is supplied. Luna authors one
source revision, executes six practices and publishes in nine calls, using
554,008 reported tokens, 1,080 observations, 1,944 measurement units and 68,067
computation units. The same final source confirms all four cases, retaining the
unchanged predictor in the stable case. The preceding unsuccessful phase and
its costs remain separate.

The revised program buys new paired meter/response trajectories after the old
prediction fails, fits an instrument on those records, and explicitly restricts
response fitting to the same records. It then confirms on fresh observations.
Thus the evidence used to fit a supporting computation also bounds the evidence
used to fit its parent response. We freeze this program on the same 32 cases,
with the same original priors and limits and no LLM calls during transfer.

| Task group | Previous published method | Evidence-scoped revision |
|---|---:|---:|
| Previous stationary tasks | 18/18 | 18/18 |
| Stable checkpoint | 2/2 | 2/2 |
| Response changes after checkpoint | 0/4 | 4/4 |
| Instrument changes after checkpoint | 0/4 | 4/4 |
| Both change after checkpoint | 0/4 | 4/4 |
| Total confirmed and terminal-correct | 20/32 | 32/32 |

Both methods produce zero false confirmations. All twelve repaired tasks finish
within 240 observations. Observations rise from 5,376 to 6,816, measurement units
from 6,912 to 12,960, and computation from 237,096 to 629,340. Stable tasks still
reuse the original predictor; earlier changed tasks preserve accuracy with more
acquisition and refitting. The method repairs evidence scope, but refits both
instrument and response after every predictive failure. It has not learned to
retest the old instrument and preserve it when only the response changes.
[Evidence-scope learning and frozen comparison](../artifacts/temporal_evidence_scope_learning_luna_20261006/RUN_STATUS.md).

### 5.18 Selective repair while preserving earlier capabilities

We explicitly ask the author to demonstrate two responses to a failed parent
prediction: preserve the inherited instrument when new measurements support it,
and reopen instrument research when they refute it. The selected response must
use the corresponding instrument and pass fresh confirmation. These outcomes are
identified from paid execution traces, without exposing hidden change labels.
The same source must also retain unchanged-predictor reuse and solve every
practice case. Operators, hypothesis families and per-task limits remain fixed.

The first learned revision satisfies this objective in eight calls (450,058
reported tokens) and succeeds
on all twelve additional calibration cases, with four instrument-retention and
eight reopening trajectories. However, it regresses on eight earlier tasks:
an instrument already invalid at the start triggers repeated measurement because
refitting is conditioned on an earlier failed parent confirmation. The strong
baseline passes 44/44, whereas this revision passes 36/44. Its four practice cases
all began with a valid instrument, leaving that entry condition untested.
[Initial selective-revision result](../artifacts/temporal_selective_revision_luna_20261006/RUN_STATUS.md).

We add a fifth opaque practice case reproducing this initial-refutation failure,
with a stable checkpoint that preserves its physics and the common public
interface. Its actual failed replay supplies the evidence for another author
phase. All preceding 44 tasks become regression controls; twelve further
calibration cases are fixed before authoring. The comparator remains the earlier
fully successful refit-both method, rather than the regressed revision.

Luna makes two source edits and confirms all five cases in eleven calls, using
732,491 reported tokens, 1,656 observations, 3,240 measurement units and 71,427
charged computation units. The final program changes the research order. After
its initial acquisition, it tests the inherited predictor directly. A failed
prediction leads to fresh measured trajectories and an assessment of the old
instrument. Support leads to response-only fitting; refutation leads to instrument
and response fitting. This also handles initially invalid instruments by obtaining
a parent counterexample before the supporting investigation. The same source
and original numerical priors then execute without further LLM calls.

Both methods confirm and pass terminal evaluation on all 56 tasks, with zero
false confirmations. This preserves all 44 regression controls and succeeds on
all twelve new calibration cases. The final program retains the instrument in
20 tasks, reopens it after measured refutation in 32, and directly reuses the old
predictor in four. Among the new twelve, these repair counts are four and eight.

| Frozen execution cost, 56 tasks | Refit-both baseline | Selective revision |
|---|---:|---:|
| Observations | 12,576 | 12,960 |
| Measurement units | 23,328 | 23,328 |
| Charged computation | 1,169,268 | 754,464 |

Computation falls by 35.48%. The 384 additional observations come from eight
initially changed-instrument tasks, where testing the inherited predictor adds
a 48-row failed confirmation before recovery. On the twelve new calibration
cases, both observation and measurement costs remain equal, while computation
falls from 269,964 to 186,792 (30.81%). Authoring costs are additional: the two
selective-learning phases together use 19 calls and 1,182,549 reported tokens.
The earlier 36/44 regression remains recorded. These results establish an
executable method that conditionally preserves a supported component, repairs a
refuted component, and recovers its earlier predictive capability after revision.
[Selective preservation and matched results](../artifacts/temporal_selective_preservation_luna_20261006/RUN_STATUS.md).

**Historical experiment methods under live LLM control.** We also expose the
measured v0.76 method book to Luna on the original window/event task family.
An initial two-source pilot solves both tasks through self-designed queries but
produces no operator-managed method entries, so its planned transfer pairs do
not start. We then directly retain or remove the historical book in two matched
target pairs. Both conditions receive identical current paid calibration,
supplied response candidates, tools and limits.

All four targets pass fresh confirmation and the original 128-row terminal
activation/exit evaluation with zero MSE. In the retained event task, Luna
requests a distinguishing experiment. The planner recalls the old temporal
recipe, rebinds its control under the new calibration and threshold, and skips
fresh synthesis. Four purchased observations return $[0,0,1,-1]$, uniquely
supporting the event response among five supplied candidates. Measured feedback
updates the historical method with cross-task support. The other targets use
informative self-designed queries without explicit method execution.

Across these two pairs, retention reduces observations from 336 to 252 but
increases measurement units from 432 to 468, charged computation from 23,540 to
25,946, and reported tokens from 173,825 to 283,230 (four versus nine calls).
The retained event agent first fits uninformative all-zero data. The two-pair
pilot establishes live consumption of an existing measured method, with mixed
cost effects. Historical acquisition costs remain separate.
[Protocol, traces and results](../artifacts/temporal_historical_methods_luna_20261006/RUN_STATUS.md).

We subsequently expose the existing research-action policy as a paid-evidence
choice operator, with executable plans from the same experiment library. In a
second four-run pilot on those tasks, every Luna agent requests this comparison,
purchases four distinguishing observations, and confirms the uniquely supported
candidate. Both retained runs execute historical methods; both removed runs
reconstruct a method. All four frozen predictions again pass terminal scoring
with zero MSE. Both conditions use 248 observations, 456 measurement units and
six calls. Retention reduces charged computation from 31,042 to 26,329 (15.2%)
but increases reported tokens from 179,812 to 206,813 (15.0%). The redundant
zero-response fit does not recur.
[Research-choice traces and matched results](../artifacts/temporal_historical_actions_luna_20261006/RUN_STATUS.md).

An offline session study connects paid inquiry episodes to the existing revisit
scheduler. In 12 fixed apparatus pairs, the complete learned library is retained
or removed after identical consumer observations and source research. All training
precedes final scoring. Retention triggers 12 revisits and produces 12/12 consumers
passing fresh confirmation and terminal evaluation, versus 0/12 after removal;
source tasks pass 12/12 in both conditions. All retained consumer programs call
the source concept. Whole-sequence observations are 5,907 versus 3,744,
measurement units 2,592 in both, and charged computation 1,361,114 versus 920,268.
Each successful callable retains its actual source template and role binding
through revision. A fresh native comparison supplies provisional source support;
the Agent's subsequent fresh confirmation compares the unchanged transferred
explanation and its incumbent again. All 12 consumers obtain confirmed source
returns that update the producing research procedure. Two feedback stages per
apparatus yield 24 historical records. These results use the original 12
apparatuses after repairing source propagation, with unchanged final programs,
observations and measurement costs; the additional paired comparisons cost 576
computation units. [Session protocol and results](../artifacts/callable_source_returns_20261006/RUN_STATUS.md).

A subsequent same-model pilot gives Luna autonomous control of this session
interface on four of those apparatuses. Both knowledge conditions share the
paid consumer prefix, tools and limits. Retention produces three completed targets
and one call-limit exhaustion; task-local learning produces three completed targets
and one service-capacity interruption. All six submitted predictors pass fresh
confirmation and terminal evaluation. Among the three pairs without service
interruption, retention completes 2/3 targets versus 3/3 for task-local learning.
Both use 288 consumer observations; retention uses 72 additional measurement
units and 597,151 versus 524,046 reported tokens. Shared source production remains
charged once per condition in the complete ledger.

No Agent selects native research, a revisit or explicit source confirmation.
The retained condition invokes the ordinary instrument, response and history
operators once each. Its unsuccessful run finds a near-exact trajectory-held-out
quadratic fit, then explores weaker features without submitting before its call
limit. This pilot exposes a research-control failure and shows that the earlier
offline scheduler ablation does not establish an advantage over an ordinary
adaptive Agent. The source knowledge was produced by the native offline researcher,
and the apparatuses were previously studied. [Live-session protocol and results](../artifacts/legacy_session_luna_20261006/RUN_STATUS.md).

We next connect the existing evaluated-model snapshot and research-action policy
to ordinary Python candidates. New observations require a paid reassessment of
the retained predictor using its original fitting data. A fixed offline controller
tests this mechanism on 12 new finite-history apparatus instances: learn an initial
response, change it, then return to the original response. Candidate coefficients
are frozen before assessment. Retaining all confirmed explanations and retaining
only the latest each solve 24/24 targets, with all 12 shared sources also passing.
On return, retention selects the original learned code in every case without
refitting; latest-only refits in every case. Including initial acquisition once,
charged computation is 59,040 versus 84,672 (30.27% lower), with 3,456 observations
and no measurement units in either condition. Actual training times are nearly
equal, 18.37 versus 18.47 seconds.

Separately, a disclosed development replay of the unsuccessful Luna apparatus
completes in 12 calls. Its first fresh confirmation fails; after incorporating the
24 counterexample rows, it compares the old snapshot and revised predictor on
the same 96 paid rows, with MSEs $1.82\times10^{-7}$ and $9.73\times10^{-30}$,
respectively. The final predictor passes a new confirmation and 128-row terminal
evaluation. This single replay does not isolate the repair's causal effect, and
the LLM still does not select the research-choice or revisit operators. The
offline experiment establishes candidate reuse under a fixed controller;
this motivates a live changed-and-returned sequence.
[Candidate-retention experiments](../docs/hegel_machine_candidate_retention_20261006.md).

A subsequent live pilot uses two new apparatus instances. Luna learns each source
once, then each knowledge condition continues through a changed response and its
return. Archived Python predictors carry their original fitting data and dependencies;
each branch inherits its own confirmed state. All ten research episodes end before
terminal evaluation. Both sources and all four targets per condition pass fresh
confirmation and the separate 128-row terminal test.

On both returns, retention submits the unchanged original predictor in two calls;
latest-only researches anew in 11 and 8 calls. In one retained return, the Agent
asks the research-choice operator to compare the original and changed predictors,
obtaining paid MSEs of $2.46\times10^{-29}$ and $0.38385$, then confirms the original.
The other evaluates and confirms the original directly. Including common source
acquisition once per condition, retention uses 30 versus 42 calls, 697,061 versus
1,188,554 reported tokens, and 576 versus 600 observations. Latest-only's failed
submission on one return purchases counterexamples and leads to a successful
revision. Physical usage is 61 calls with complete usage reporting; cached input
accounts for 54.28% of reported input tokens.
[Live retention protocol and traces](../artifacts/candidate_retention_cycle_luna_20261006/RUN_STATUS.md).

A follow-up diagnostic pilot reuses these two learned sources, each on a parameter
change and a change to recurrent dynamics. An explicit same-source fitting tool
holds out whole paid trajectories. Under a fixed controller, refitting repairs
both parameter changes; its validation errors of 0.116 and 0.029 on the recurrent
tasks instead lead to the existing instrument and response investigations.
All four targets pass fresh confirmation and terminal evaluation. Live Luna
solves the two parameter changes but exhausts its 12-call limit on both recurrent
tasks. This first live run omitted the numerical confirmation criterion from the
public interface. After publishing the unchanged $10^{-10}$ MSE threshold, a
development replay of all four tasks again yields 2/4, with no false confirmations.
The replay uses 41 calls and 1,260,301 reported tokens. One failed investigation
explicitly recognizes underdetermination and buys meter readings, but its next
action still fits control history without using those readings. Both parameter
tasks succeed through instrument/response research; neither live run selects
the new refit tool. The remaining operational gap is turning evidence of failure
into an effective research branch. That version of the Python research-choice interface
offered retention and confirmation, whereas revision planning already existed for
the internal program language.
[Diagnostic pilot and preserved failures](../artifacts/python_refit_criteria_luna_20261006/RUN_STATUS.md).

We subsequently expose the existing refit, instrument and response investigations
through the Python chooser. A deterministic consumer executes only its proposed
next action, without selecting branches from private task labels or numerical
errors. On the same four development targets, both parameter changes require
refit and confirmation; both recurrent changes additionally require measured
instrument validation and response revision. All four pass fresh confirmation
and terminal evaluation. The 12 proposed actions are all executed, using 480
observations, 288 measurement units and 10,598 charged computation units, with
no LLM calls. Replayed control trajectories remain together when splitting
response training and validation. This result establishes an executable bridge
to prior research mechanisms.

The subsequent live replay passes all four targets, with no false confirmations.
Both recurrent cases acquire instrument readings and invoke the existing
instrument and response operators; the second does so at its tenth model call,
after unsuccessful control-history fitting. The two parameter changes are solved
by refitting a three-lag model. Neither the research-choice nor the refit operator
is invoked by the live Agent. Total usage is 33 calls, 954,230 reported tokens,
504 observations and 144 measurement units; cached input is 54.59% of reported
input. These traces establish live use of the older research branches.
[Routing implementation and live traces](../artifacts/python_research_routing_final_luna_20261006/RUN_STATUS.md).

We next place concrete recommendations directly in action feedback and compare
showing versus withholding them on the same four development targets. Both
conditions compute the recommendations and retain the same tools, including the
explicit chooser; they share model, prompt, initial observations and a 12-call
limit. The visible condition completes 3/4 targets, compared with 2/4 when advice
is withheld, with no false confirmations. Both solve the parameter changes;
only the visible condition solves one recurrent change. In that pair, both
spend 144 measurement units, but the successful Agent tests an instrument
distinction, assesses the acquired readings and uses the supported instrument
to revise its response. Terminal MSE is $1.09\times10^{-18}$. Thus a supporting
investigation actually changes the parent revision in this live trace.

Across the four targets, visible/hidden advice uses 32/36 calls,
968,505/1,128,135 reported tokens, 456/696 observations, 216/144 measurement units
and 17,086/22,488 charged computation units. The 240-row observation difference
comes entirely from a pair in which both Agents fail. Six of 15 visible
recommendations with a subsequent event exactly match that next action and its
required arguments. In the remaining visible-condition failure, instrument
ambiguity is identified, but the Agent returns to high-order history fitting;
near-zero paid-data error becomes $4.18\times10^{-4}$ on fresh observations.
The completed experiment uses 68 model calls, with all usage reported and 49.76%
cached input. [Advice intervention and traces](../artifacts/python_research_advice_ablation_luna_20261006/RUN_STATUS.md).

We then keep unresolved advice available after ordinary Python actions and in
the research state. On four new apparatuses, live source learning succeeds on
three; the failed source blocks four registered target episodes. Persistent and
event-only advice both complete 4/6 attempted targets (4/8 registered), with no
false confirmations. Removing a pair affected by service capacity leaves both
at 4/5; persistent advice uses 4.91% more tokens, 16.67% more observations and
75% more measurement units. In one failure the same distinguishing purchase
remains visible across four ordinary actions but is never executed. In another,
the event-only Agent finds a low-error candidate but uses its last call for
evaluation, leaving fresh confirmation unfinished. Continued presentation is
operational, while effective adoption and completion remain unresolved.
[Frozen persistence study](../artifacts/python_pending_advice_luna_20261006/RUN_STATUS.md).

We subsequently expose the existing procedure executor as one bounded live action.
Both conditions receive the same editable, developer-authored adapter to the
research policy; the intervention allows repeated execution in one model call.
Explicit offline execution completes all six targets. In the live availability
comparison, the bundled/atomic conditions complete 3/6 and 5/6; excluding two
service-capacity-interrupted pairs leaves 2/4 and 4/4. No Agent invokes the supplied
program in either condition. The two research failures stop after paid candidate
evaluation and after instrument discrimination, respectively, without fresh
confirmation. Thus making a complete procedure available does not establish its
adoption or benefit. [Execution intervention and traces](../artifacts/python_procedure_luna_20261006/RUN_STATUS.md).

A follow-up places the procedure option directly in current feedback. Both
conditions complete 5/6, or 4/4 after removing two service-interrupted pairs;
the latter comparison uses 36 versus 20 calls. None of 24 displayed options is
selected. This presentation is therefore retained only as an experimental
condition. [Feedback-option intervention](../artifacts/python_procedure_offer_luna_20261006/RUN_STATUS.md).

We then reuse the method-authoring and paid-practice loop with actual archived
Python predictors, retaining their training files and dependencies. Development
exposes mismatched practice and runtime trace formats and an incomplete description
of paid evaluation versus fresh confirmation. After correcting these interfaces,
continued authoring produces a conditional procedure: confirm the inherited
predictor, refit after rejection, and investigate the instrument when refitting
fails separate-trajectory validation. Instrument evidence then supports response
revision and fresh confirmation. The authoring process ends before testing its
last edit, so we freeze that exact candidate for an externally scheduled replay.

The unchanged source passes all three practice conditions. On two other archived
predictors, each tested under unchanged, response-parameter and recurrent-instrument
conditions, both the learned candidate and fixed procedure complete fresh
confirmation and independent terminal evaluation on **6/6 targets**. The candidate
directly reuses the old predictor twice, refits it twice, and completes measured
instrument research followed by response revision twice. Its maximum terminal
MSE is $6.34\times10^{-19}$. It uses 768 versus 672 observation rows and 11,606
versus 10,406 computation units; both programs use 288 measurement units. The
candidate's initial fresh rejection tests make it more expensive in this comparison.
Replay uses no model calls. All preceding authoring phases, including failures,
total 49 calls and 2,015,212 known tokens, with one call's usage unreported.
[Frozen-candidate replay and retained authoring outcomes](../artifacts/python_method_candidate_replay_20261006/RUN_STATUS.md).

A subsequent workflow makes an author's explicit submission complete missing paid
practices before applying the same publication gate. Restoring the actual author
checkpoint, Luna edits the program once and submits within six calls, consuming
165,997 tokens (36.10% of reported input cached). The submitted program passes all
three practices and is published. Frozen execution on six targets with advanced
trajectory streams again yields **6/6 for both learned and fixed programs**, with
two unchanged-prior reuses and no false confirmations. The new source replaces
selective refitting with measured instrument reconstruction after every prior
failure. Consequently, observations rise to 960 versus 672, measurement units to
864 versus 288, and computation to 37,392 versus 10,406. This completes the
author-requested practice, publication and transfer chain, while exposing a loss
of selective repair. Cumulative authoring, including failed phases, uses 55 calls
and 2,181,209 known tokens plus one unreported call; source acquisition and the
earlier offline replay remain separately charged.
[Publication continuation and costs](../artifacts/python_method_publication_luna_20261006/RUN_STATUS.md).

To preserve selective repair while testing recovery, we add the existing laboratory's
public 96-row response-change checkpoint. The earlier selective program validates
a refit, encounters a fresh counterexample and repeatedly resubmits the rejected
candidate. Paid feedback supports a model-authored revision. Following a service
interruption and exhausted author phase, an explicitly added phase with completion
guidance submits the last authored source unchanged. All four practices pass:
parameter changes retain zero-measurement refitting, while fresh rejection after
validation triggers further research using newly measured response records.
On eight frozen cross-source targets, the revised program solves **8/8**, versus
6/8 for the earlier selective source, 8/8 for the previous published program and
6/8 for the fixed controller. The latter confirms two models before the checkpoint
that fail settled-state evaluation. Against the previous 8/8 program, observations
decrease from 1,344 to 1,248, measurement units from 1,296 to 864, and computation
from 61,172 to 23,160; measured Python execution time increases from 6.00 to 8.35
seconds. New authoring, including the interrupted call and failed practice phase,
uses 13 calls and 369,866 known tokens plus one unreported call. The 24 completed
reference episodes are reused by the evaluator and never supplied to the author.
[Selective repair and counterexample recovery](../artifacts/python_selective_method_completion_luna_20261006/RUN_STATUS.md).

We next retain these cases and add an instrument-family change at the same
announced checkpoint. The previous published program solves all ten cross-source
targets. Across five further author phases, the final candidate confirms all five
practice cases but retains an absolute fresh-loss cutoff for choosing scoped
refitting. It demonstrates no zero-measurement recovery after refit rejection,
so publication fails and its ten transfer slots remain unexecuted. A
researcher-arranged diagnostic then tests the same current-record refit in both
change conditions. A response change with fresh MSE 0.447 yields scoped-validation
MSE $1.87\times10^{-29}$ and passes subsequent confirmation; an instrument change
with smaller fresh MSE 0.0284 retains validation MSE 0.0237. This demonstrates why
loss magnitude cannot identify which repair is adequate in these cases. The
diagnostic costs 312 observations and no meter readings. Supplying it to the
author does not produce another revision: the last phase spends its calls reading
evidence. All five phases together consume 40 calls and 1,521,964 tokens, with
27.01% of reported input cached. The sixteen unchanged reference outcomes and
four newly acquired reference outcomes are each charged once.
[Scoped-evidence comparison and failed method adoption](../artifacts/python_scoped_method_comparison_luna_20261006/RUN_STATUS.md).

A subsequent engineering integration reuses the evidence-dependent repair rule
already present in earlier implementations. It replaces the candidate's
fresh-loss cutoff with current-record refitting and separate-trajectory validation;
the other branches and numerical operators remain unchanged. All five source
practices and ten cross-source targets pass. A later offline replay of the exact
unmodified candidate also solves all ten targets, allowing a direct development
comparison: observations fall from 1,722 to 1,628, measurement units from 1,278 to
852, and computation from 50,388 to 29,020. Both response-change targets recover
without meter readings; failed scoped validation in both instrument-change targets
leads to measured repair. Those structural cases incur 48 additional computation
units in total. The five source and twenty new target executions use no model
calls; the twenty earlier reference outcomes are reused.
[Historical repair integration](../artifacts/python_current_evidence_bridge_20261006/RUN_STATUS.md).

We then test adoption of this supplied program. Three independent Luna authors
start with the unpublished candidate as their current file and receive both
programs, their provenance and compact source-practice evidence. Candidate
positions alternate and practice confirmation streams advance. Each author
copies the historical repair program using ordinary Python and submits it on
the next call; all fifteen fresh practices pass the unchanged publication gate.
The six calls consume 145,243 tokens, with no reported cached input. Each final
source is identical to the supplied program, so its existing ten target outcomes
are reused as historical evidence; no new target execution is counted.
[Adoption of supplied research programs](../artifacts/python_method_adoption_luna_20261006/RUN_STATUS.md).


A further task changes the response after an instrument-supported response has
passed validation. The supplied program then stops after fresh refutation and
solves ten of twelve cross-source targets. Three Luna authors given this source
counterexample publish no revision: two spend their calls inspecting and editing;
one confirms three original cases before a provider-capacity interruption. We
then connect each source edit to the existing paid practice engine, immediately
returning execution feedback on the added case. Two of three authors publish
programs satisfying all six source conditions; each frozen program solves all
twelve targets without false confirmation. In one successful trajectory, the
author repairs repeated acquisition and stale-record mixing, then fits a current
measured batch and reduces fresh MSE from 2.03 to $3.04\times10^{-19}$. Both
programs preserve earlier reuse and scoped-repair behavior. The third author
passes the added case once but breaks it in a later edit and does not publish.
All 25 practices, including ten failures, are retained. This condition uses
29 calls and 1,750,532 tokens, with 27.16% of reported input cached. Its 24 new
target executions evaluate two programs on the same twelve tasks; the unchanged
controls are reused. The first published program uses 2,296 target observations
versus the unchanged program's 2,010.
[Method revision with immediate experimental feedback](../artifacts/python_method_edit_practice_luna_20261006/RUN_STATUS.md).

## 6. Discussion

The central design issue is the unit that persists between research episodes. A stored answer helps when the next task is sufficiently similar. An executable concept also supports new compositions. A research procedure can remain useful when its earlier fitted answer no longer applies. Evidence about applicability determines which of these uses is justified. Hegel Machine puts these units into the same research loop so that their consequences can be evaluated across task changes.

The framework's broader contribution is to make the support of an abstraction a possible object of research. A residual can challenge a coefficient, the representation in which that coefficient is meaningful, a boundary condition, or the procedure used to collect evidence. These alternatives explain why adding model capacity alone is an incomplete account of scientific revision. Section 5.9 exhibits one narrow path through this space: failure of a composed prediction leads to research on an internal relation, followed by confirmation and reuse. The RL, robust-decision and causal examples in Section 3.6 describe further settings in which the same dependency-oriented questions are meaningful.

The online study separates this organization from its comparative advantage.
Supporting investigations precede their evidence and change a parent's next
experiment, yet a flat adaptive researcher with the complete joint language
also solves every task. Recursion is therefore an operational way to organize
research in this setting, with no demonstrated accuracy advantage. Compared
with hypothesis-generation and experience-memory systems, the relevant object
of evolution is a linked explanation, its supporting conditions and the
procedure used to investigate it. Each object requires evidence of actual use
and an intervention on its consequences. A fixed external prediction task
supplies that evidence even when the agent revises its own research criteria.

An executable program can remain useful as an ingredient after it fails as a complete explanation. In the composition family, the retained response supplies a structured starting point for a correction, while unchanged definitions remain available within the resulting expression. Every proposed correction must then predict newly acquired observations. The distinction between retaining a computation and accepting its current predictions is central to this interpretation of revisable knowledge. The retention intervention establishes its practical value, but conditional feedback leaves behavior unchanged relative to unmarked retention. Applicability feedback must change a useful research decision to contribute independently. Recording a failure is insufficient when both states admit the same successful reassessment.

The joint study extends this distinction from predictive programs to research
procedures. A successful instrument choice is stored as a comparison to repeat,
with fitting subquestions bound to the current catalogue. Removing those
procedures increases later acquisition and search even when predictive knowledge
is unchanged. The transfer concerns how to investigate a response, rather than
only which response to predict. Conversely, the expression-composition contrast
shows why generating a callable abstraction does not by itself establish a
stable downstream saving. Knowledge types require their own interventions.

The continuation intervention supplies a concrete redundancy case: existing
knowledge already provides the required structure, and an additional useful-looking
candidate only increases screening work. Knowledge growth is therefore assessed
against the demand it serves and the cost of considering it.

A deterministic model is itself a modeling commitment. In the joint study,
repeated outcomes support a stochastic description, while a later linear task
supports a cheaper expression without discarding nonlinear knowledge needed on
return. These are operational examples of both expansion and simplification.
Variance estimation and calibrated prediction intervals remain distinct targets:
improved variance error here coexists with interval undercoverage. The relevant
criterion is improved investigation and prediction under the task's conditions,
taking acquisition and maintenance costs into account.

The experiment-induced perspective explains why candidate generation alone is insufficient. Two proposed explanations may remain indistinguishable under the observations that the agent currently purchases. Additional fitting of those observations can leave the ambiguity intact. Appendix A gives a simple error floor for a representation that aliases two distinct deterministic outcomes. An effective research action must either expose the missing distinction, change the prediction model to represent uncertainty, or acknowledge the unresolved question.

The component-change study shows distinct research responses to three situations: direct reuse when nothing changes, reconstruction from existing components after a gain change, and new relation comparisons after a boundary change. This supplies a concrete instance of a research procedure changing its work according to current evidence. The same system's deletion failures show why this capability remains conditional on the knowledge and search directions available. Some candidates pass internal confirmation yet fail terminal prediction, and some unresolved investigations stop with resources left. Wider revision choice requires investigations that distinguish these situations reliably.

The larger research agenda concerns how far these changes can extend and when
maintaining a research state pays. Knowledge acquisition is an investment whose
payoff depends on recurrence, task change, and the cost of deciding whether it
applies. A family with cheap fixed models may leave little room for an elaborate
research loop. A family with expensive repeated discovery may make retention
valuable. The joint study exhibits both sides: learned methods reduce a later
task's work, while maintaining and searching the complete state increases actual
CPU over the sequence. Separate resource dimensions and whole-sequence outcomes
make this trade-off visible.

## 7. Limitations

The live hypothesis-retention cycle contains two parameter instances within one
family. It tests historical retention's incremental value and establishes neither
a population-level accuracy gain nor stable long-horizon benefit. Archived fit
callbacks can rebuild parameters on their original rows; the two retained returns
do not fit on current target labels. Predictive success also leaves structural
explanations unresolved: one latest-only Agent attributed an old-coefficient
failure to insufficient history and adopted nine lags, although a three-lag model
also solved that task. Research on parameter failure versus structural inadequacy
remains necessary. The subsequent diagnostic pilot uses the same two sources;
its replay is neither an independent sample nor an isolated causal estimate of
the interface repair. Failure of one fitting procedure does not establish that
every model in its family is inadequate.

The later 4/4 routing replay reuses the same known tasks and does not isolate the
new chooser's effect: the live Agent never invokes it. Its instrument components
are not separately assessed on prospective readings, although the composed
response passes fresh confirmation and terminal prediction. An earlier startup
was operator-interrupted after concurrent tests exposed interface errors; its
seven saved calls and 178,244 reported tokens remain separately recorded, with
unreported in-flight usage unknown. It supplies no task success/failure outcomes.

The subsequent advice intervention has one discordant pair out of four known
tasks derived from two apparatuses. It isolates visibility within the shared
tool environment, rather than the complete framework versus a vanilla Agent.
Each condition is sampled once per target. The observed additional success and
lower aggregate token use do not establish a stable comparative advantage, and
measurement cost increases. The later persistence study uses three successfully
learned sources out of four new apparatuses and retains the blocked targets.
One provider call has unknown usage, so its complete token total is unavailable.
That study also exposes a routing dead end after fitting an instrument without
paired readings. A subsequent repair restores acquisition and passes offline
confirmation; it is outside the frozen live comparison.

The archived-Python method studies use two previously studied apparatus families.
The initial replay tests an unpublished last candidate; the later author-requested
publication uses a revised source and host-scheduled completion of missing practices.
Those stationary comparisons match fixed-controller accuracy at higher observation
and computation costs. The subsequent selective-repair study uses an announced
change checkpoint, explicit behavioral publication requirements, and an additional
prompted completion phase. Its eight targets do not establish autonomous change-point
detection. Fitting families remain supplied and experimental sequences are embedded
in the source. The learned revision still purchases meter data after a validated
refit fails fresh confirmation. The subsequent five-case study and manually
arranged scope comparison establish that current-record refitting can separate
two repair needs, but its author fails to adopt the demonstrated choice. The
diagnostic is researcher-directed evidence, not an autonomously selected experiment.
The subsequent successful integration is a manual edit, evaluated on the same
five conditions and two target archives. Its exact-parent comparison is added
after the integration run; it measures a local development intervention, not
autonomous method learning or independent confirmation on unseen task families.
Interface clarifications and additional author allowances are part of these
development interventions. The adoption study supplies a human-integrated program
and its source-practice evidence; three successful authors establish a narrow
adoption result, not autonomous invention or general retrieval from a large library.
It also repairs stale author context after an ordinary Python source edit. The
separate effects of this repair and the supplied library are not isolated.
The subsequent response-reopening study remains within these known families and
an announced checkpoint. The host initiates practice after an edit; it does not
supply the repair or initiate publication. The three development attempts do not
establish a reliable population-level advantage, and one control author is
interrupted by provider capacity. The successful programs rebuild the instrument
as well as the response, increasing acquisition cost. Token stopping is checked
before a call and may be exceeded by that call's reported usage. Retention of
previously successful method revisions remains unresolved.
Instrument-assessment failure and open-ended
acquisition revision remain untested. Development changes do not isolate the
contributions of interface repairs, prompting and requested reasoning effort.

The historical-method LLM pilot contains two development pairs with supplied
response candidates. The book is also visible in the prompt, so its language-level
influence is not isolated from executable reuse. The observed successful reuse
does not establish stable net benefit or new hypothesis invention.

The temporal controller in Section 5.12 uses a developer-authored revision
procedure and existing candidate families. Its same-data history control tests
the value of a learned instrument coordinate. The subsequent flat comparison
finds equal accuracy with fewer measurements and more response fits; it does
not establish an independent benefit of separate child confirmation.
Earlier raw-Python temporal LLM pilots complete zero of two targets in both Code
and Hegel arms. The shared-operator pilot in Section 5.13 is a separate,
single-apparatus development comparison. It measures selection of prescribed
operations, not invention of their fitting algorithms, and still establishes
no advantage of executable memory. The later repaired-interface comparison in
Section 5.15 completes all four cases in both conditions. It tests two previously
studied calibrations with supplied fitting operators and does not isolate a
causal effect of the interface repair or establish overall method-retention gains.
Section 5.16 independently varies instrument and response parameters within that
same recurrent family. Paired worlds share parameters and are not independent
population samples. Five capacity interruptions prevent a complete initial live
comparison; its three completed cases per arm cannot be interpreted as equivalent
algorithmic success rates. The subsequent successful method-learning run is
explicitly prompted to repair a supplied counterexample, with prescribed fitting
operators and publication cases. Its learned trigger responds to any preceding
failed confirmation. Section 5.17 repairs stale evidence scoping through a second,
explicitly prompted author phase. Its refit-both strategy does not establish
selective diagnosis of which supporting component failed. The two author phases
differ in their supplied failure, feedback presentation and stochastic trajectory,
so their outcomes do not isolate the causal effect of the compact chronology.
Section 5.18 subsequently demonstrates instrument retention versus reopening
under an explicit selective-repair objective. Its initial regression and added
practice case show that preserving earlier capabilities is not automatic.
The program still uses prescribed instrument-assessment operators and fixed
acquisition patterns. It does not determine whether response coefficients also
changed when an instrument is refuted. The disclosed checkpoint does not test
detection of arbitrary unannounced drift or noisy failure sources.
The frozen contrast compares old and revised methods, not ordinary Code agents or
an additional live-LLM ablation. Parameter transfer does not establish new structural
families, and additional authoring/compute costs prevent an overall cost-dominance claim.

The general recursive-research formulation is broader than the demonstrated algorithms. The calibrated-response study establishes live two-level supporting investigations, but its question types, dependencies and 96-hypothesis language are prescribed. Its structural transfer holds out source-to-target compositions, not primitives from the researcher, and all three researchers reach equal accuracy under the common ceiling. The earlier nested-relation case uses prescribed revision operators and a bounded experimental loop; its dependency graph is compiled from completed fitting steps. It does not establish autonomous generation of arbitrary-depth research subproblems, formal theorem proving, or self-invention of the rules of justification. The RL, robust-optimization and causal-identification connections are conceptual instantiations, with no reported end-to-end experiments in those domains. Numerical convergence, empirical confirmation and deductive proof supply different kinds of support. An additional assumption supports a conditional claim, and some assumptions cannot be tested from the available observations. The Hegel-inspired preservation principle concerns earlier valid content within its scope; a theory containing every historical claim need not exist. Priors can improve search efficiency but cannot guarantee freedom from overfitting.

The reported results are controlled studies of finite scalar and object-process
families. New apparatus instances vary realizations within those families;
they do not establish transfer to unseen scientific domains or arbitrary new
structural compositions. The object interface supplies a schema, variable roles,
and measurement options. The joint-growth interface anonymizes roles and changes
units, but public ranges constrain candidate coordinate transformations and the
response families remain algebraic analogs. Primitive operators, instrument
catalogues and search bounds are designed in advance. The library changes the
effective search language; learned procedures compose existing laboratory
commands and fitting subquestions. Neither mechanism establishes unrestricted
invention of representations or experimental tools.

The studies do not isolate every action-selection mechanism or establish reliable
autonomous choice among all revision levels. A preserved composition can retain
an obsolete component with a cancelling correction, so successful repair need
not produce a minimal expression. The response-retention intervention uses a
preselected shared challenge; it does not test autonomous choice of that challenge.
Its additional conditional-feedback record provides no benefit. Joint growth
learns executable scope conditions in a separate family, but this does not repair
the earlier null feedback result or establish general validity predicates. The
separate target-binding withdrawal mechanism also cannot distinguish old and
changed regimes when their public task identity is the same.

The component-change study covers a response-gain change, a relation-radius change and an unchanged condition. The gain change alters a contribution to the readout rather than the physical exchange law. Both revisions use existing program and research families; the result does not establish general discovery of new mechanisms or optimal selection among arbitrary research actions. This study contains no unresolved outcome, so it does not explain or repair the earlier deletion arm's stopping failures.

A successful fresh trajectory is finite evidence. It does not establish logical
conservative extension, a uniform guarantee over the old scope, or statistical
immunity to adaptive overfitting. The 11/12 procedural result and the 72/72
task-accuracy result measure different properties. The mechanism studies contain
12 instances each. The three subsequent object-process studies and the
joint-growth study each contain 30 apparatus instances, not 30 scientific
domains. In the retention intervention, deletion produces four internally
confirmed first-revision candidates that fail terminal evaluation. Joint growth
also has two such persistent-arm failures and six externally successful but
unconfirmed tasks. Its stochastic pass rule is deliberately broad and does not
certify calibrated uncertainty; measured interval coverage remains below nominal.

The task-local controls retain the corresponding researcher's within-task
capabilities, including conditional-expression search in the joint study. The
other controls cover particular algorithms: the dense fixed-basis learner lacks
adaptive model selection, and passive research removes adaptive revisits. A
stronger fixed-family learner and an independent code agent remain useful
comparisons for the earlier task families. The new calibrated-response study
includes a strong flat active control. Its completed common-model comparison
also exhibits a ceiling for ordinary Code/ReAct and no Hegel advantage (Section
5.11). The public finite grammar and direct response access make it a weak test
of autonomous representation or research-method discovery. The replacement
trajectory environment remains a development task.
Source removal identifies knowledge availability and its downstream
consequences, not a complete attribution of all sources and interactions. Method
deletion isolates a procedure-book effect, but not the contribution of each
compiled step. Static branch and dense-fit diagnostics are not complete-agent
baselines. The new-expression resource interval includes zero.

The inquiry-session comparison is offline and uses the same native scheduler
without available transfer knowledge as its removal control. Its 12 successful
consumers require more research expenditure. Source returns compare a fitted
transferred explanation against its frozen incumbent; they do not separately
identify the causal contributions of binding, fitting and the learned call.
The same-apparatus repair rerun adds no independent sample. These outcomes do not
establish the complete combined loop under autonomous LLM control.

Charged computation differs from measured process CPU time, which also depends
on the machine and workload. Main-arm order is randomized in the joint study,
while retention and additional deletion branches run in fixed orders. Their CPU
differences can include cache and order effects. Joint-growth computation savings
coexist with higher actual CPU, and acquisition differences have intervals that
include zero. The results establish neither a universally favorable scalar cost
trade-off nor stable long-horizon net benefit. The comparative performance
gains come from offline numerical and symbolic research; the completed
language-model comparison shows no overall gain.

The studies use separate frozen runtime snapshots from the development sequence. In particular, the source-deletion composition snapshot precedes subsequent observation-view repairs. The four-arm study uses one later snapshot for all arms and includes every registered instance. Comparisons between the studies cannot identify the effects of intervening software changes because their apparatus cohorts also differ.

## 8. Conclusion

Hegel Machine treats a hypothesis as a scoped abstraction whose representation,
supporting premises and investigation procedure can themselves become research
questions. The broader framework links these questions through conditional
dependencies and retains the history needed to reuse or revise their answers.
Its bounded implementations separate accepting an explanation from preserving
its executable content. A recorded trajectory connects a failed prediction,
internal relation repair, discriminating experiments, fresh confirmation and
later direct reuse. Controlled interventions show
that an acquired computation can survive a failed application, contribute to a
repair and reduce subsequent research costs. A second finite instantiation
connects expression, scope, representation, procedure and uncertainty revisions.
Persisted procedures reduce work after instrument roles change, and the complete
research state improves confirmed predictive success over equally expressive
task-local learning. The observed runtime penalty, uncertain abstraction payoff
and incomplete calibration delimit these gains. Online supporting investigations
also change subsequent experiments and revise retained procedures, while matching
a strong flat control in predictive accuracy. The framework supplies a way to
study research knowledge through the experiments it changes and the subsequent
outcomes it enables.

## Appendix A. Two elementary consequences of the formulation

### A.1 Additional experiments refine observational equivalence

For a finite set $E$, each term $\mathrm{TV}(P_h^e,P_g^e)$ is nonnegative, symmetric, and satisfies the triangle inequality. Their positively weighted sum is therefore a pseudometric. It is zero exactly when the predicted distributions agree for every experiment in $E$, so it defines a metric on the quotient by $\sim_E$.

If $E\subseteq E'$, then $h\sim_{E'}g$ implies $h\sim_Eg$ by restriction. Thus every $E'$-equivalence class is contained in an $E$-equivalence class. The refinement is strict if some newly available experiment distinguishes a pair that was equivalent under $E$. This statement concerns predicted distributions and available tests; finding a useful additional experiment remains a research decision.

### A.2 Representation aliasing creates a fitting error floor

Suppose two deterministic cases have the same representation, $\phi(x_1)=\phi(x_2)$, but responses $y_1\ne y_2$. Any predictor depending only on that representation must output the same value $a$ on both. Its average squared error satisfies

$$
\frac{(a-y_1)^2+(a-y_2)^2}{2}
=\left(a-\frac{y_1+y_2}{2}\right)^2
+\frac{(y_1-y_2)^2}{4}
\geq\frac{(y_1-y_2)^2}{4}.
$$

Refitting such a deterministic predictor cannot remove the error. A representation containing a relevant distinguishing observation could do so. In a stochastic setting, repeated outcomes can instead support an uncertainty model; the pair alone does not identify the cause. These elementary observations motivate alternative research actions without supplying a general discovery guarantee.

## Appendix B. Implementation and experimental material

The current implementation is organized around [executable programs](../src/hegel_machine/cognition/language.py), [persistent knowledge](../src/hegel_machine/cognition/memory.py), [research procedures](../src/hegel_machine/cognition/research_programs.py), the [research controller](../src/hegel_machine/cognition/unified_research.py), and [task sequences and revisits](../src/hegel_machine/cognition/unified_session.py). The [object laboratory](../src/hegel_machine/cognition/unified_object_lab.py) provides the composition study's interface.

The following material contains the exact protocols, per-instance outcomes and
frozen runtime snapshots. The first three studies are supporting mechanism
cohorts. Later cohorts were registered before execution; the final product-removal
intervention is explicitly post-hoc on previously observed continuation cases.

| Study | Protocol | Recorded results |
|---|---|---|
| Single source | [Protocol](../artifacts/narrow_research_cycle_20261004_r2/protocol.json) | [Summary](../artifacts/narrow_research_cycle_20261004_r2/summary.json) |
| Two sources | [Protocol](../artifacts/two_source_cycle_20261004/protocol.json) | [Summary](../artifacts/two_source_cycle_20261004/summary.json) |
| Composition and local revision | [Protocol](../artifacts/composed_local_cycle_20261004_r3/protocol.json) | [Summary](../artifacts/composed_local_cycle_20261004_r3/summary.json) |
| Four-arm composition comparison | [Protocol](../artifacts/paper_composed_comparison_20261005/protocol.json) | [Summary](../artifacts/paper_composed_comparison_20261005/summary.json) |
| Retention after an observed failure | [Protocol](../artifacts/paper_scoped_retention_20261005/protocol.json) | [Summary](../artifacts/paper_scoped_retention_20261005/summary.json) |
| Different component changes and no change | [Protocol](../artifacts/paper_revision_choice_20261005/protocol.json) | [Summary](../artifacts/paper_revision_choice_20261005/summary.json) |
| Joint research-state growth | [Protocol](../artifacts/growth_paper_20261006_main_30/protocol.json) | [Summary](../artifacts/growth_paper_20261006_main_30/summary.json), [derived joint outcomes](../artifacts/growth_paper_20261006_main_30/derived_analysis.json) |
| Procedure integration | [Protocol](../artifacts/workflow_bridge_20261006_main_12/results/protocol.json) | [Summary](../artifacts/workflow_bridge_20261006_main_12/results/summary.json) |
| Demand-conditioned procedure choice | [Protocol](../artifacts/workflow_demand_20261006_main_12/protocol.json) | [Summary](../artifacts/workflow_demand_20261006_main_12/results/summary.json) |
| Complete research budgets | [Protocol](../artifacts/workflow_budget_20261006_main_12/protocol.json) | [Summary](../artifacts/workflow_budget_20261006_main_12/summary.json) |
| Paid-demand continuation | [Protocol](../artifacts/paid_revisit_20261006_main_12/protocol.json) | [Summary](../artifacts/paid_revisit_20261006_main_12/summary.json) |
| New-product contribution (post-hoc) | [Protocol](../artifacts/new_product_value_20261006_posthoc_12/protocol.json) | [Summary](../artifacts/new_product_value_20261006_posthoc_12/summary.json) |

The joint-growth [frozen source bundle](../artifacts/growth_paper_20261006_frozen_source/)
contains the actual runner and runtime used for the 30-instance campaign. Current
entry points are [joint research](../src/hegel_machine/cognition/research_growth.py),
[expression search](../src/hegel_machine/cognition/growth_models.py),
[procedure compilation](../src/hegel_machine/cognition/growth_procedures.py), and
[variance learning](../src/hegel_machine/cognition/growth_uncertainty.py).
The separate [post-run analysis](../scripts/analyze_growth_paper_study.py) derives
confirmation/evaluation intersections and prefix-inclusive costs directly from
the saved case files; it does not alter the frozen protocol or fitted models.

The composition study's procedural failure is analyzed in the [instance 26107 diagnostic](../artifacts/composed_local_cycle_20261004_r3/diagnostic_26107_method_reuse.json). Earlier development cohorts remain available in the corresponding artifact directories. The prospective experiments are specified separately in the [research plan](RESEARCH_PLAN.md).

## Appendix C. Supporting mechanism studies

### C.1 Single-source acquisition, failure, and return

The first sequence learns a response, reuses it, changes the mechanism, and returns to the original condition. Initial visits have caps of 96 observations, 384 measurements, 24,000 charged computation units, and one research round. The session can allocate unused resources to revisits within the summed sequence cap. All 12 instances satisfy the complete procedural criterion. Table C1 reports complete-sequence totals.

**Table C1. Single-source study.** Each arm is evaluated on 48 tasks. The fixed-library, passive, and memory-disabled arms are algorithm controls; source deletion is a separate intervention on a shared learned prefix.

| Arm | Tasks passing | Observations | Measurements | Charged computation |
|---|---:|---:|---:|---:|
| Full | 48/48 | 2,592 | 2,448 | 318,674 |
| Delete learned source | 48/48 | 2,592 | 2,448 | 323,966 |
| Fixed library | 48/48 | 2,304 | 9,216 | 20,736 |
| Passive experiments | 48/48 | 4,608 | 18,432 | 591,702 |
| Memory disabled | 48/48 | 2,736 | 1,296 | 549,219 |

Retaining the learned source saves 5,292 computation units, or 1.63% of the deletion arm's complete-sequence cost, with equal observations and measurements. The first successor uses 195 rather than 636 computation units, illustrating how a large local saving can become modest after acquisition and revision are included.

The fixed-library control provides an important comparison: it solves every task with much less computation, while requiring more measurements. It uses a predeclared quadratic and two-step-history basis. The passive control uses precommitted experiments and also changes adaptive revisiting behavior. The original memory-disabled control retains active research but sets `use_memory=False`, which also disables within-task instrument learning and changes the confirmation reserve. It is therefore a broader memory ablation. The task-local control in Section 5.3 preserves these within-task capabilities. Their results establish a resource trade-off and distinguish the shared-prefix source effect from differences between entire research algorithms.

### C.2 Two independent learned sources

The second sequence learns source $A$, learns source $B$, revisits $A$ and $B$, changes the mechanism associated with $B$, and returns to $A$ and $B$. The source tasks involve distinct control-history and delayed-measurement dependencies. The first two tasks naturally acquire the separate sources. Budgets and terminal criteria match the single-source study. Four arms retain both sources, remove $A$, remove $B$, or remove both after the common two-task prefix.

**Table C2. Two-source study.** All arms use 5,328 observations and 5,184 measurements across 84 evaluated tasks.

| Knowledge retained | Tasks passing | Charged computation |
|---|---:|---:|
| $A,B$ | 84/84 | 721,080 |
| $B$ only | 84/84 | 727,908 |
| $A$ only | 84/84 | 723,348 |
| Neither | 84/84 | 730,176 |

All 12 sequences satisfy the procedural criterion. The contributions of $A$ and $B$ are 6,828 and 2,268 computation units, respectively, and their sum is the observed 9,096-unit saving against deletion of both. The measured interaction term is zero. This study demonstrates separately useful retained knowledge; the next study tests two sources within the same response.

### C.3 Composition and local mechanism change

The object environment exposes controllable object variables and coarse and fine measurement channels. Its response can combine a total-quantity component $A$ with an event component $B_r$, where $r$ controls an interaction radius. The combined target is

$$
C=A+0.7B_r.
$$

The sequence is: learn the total component; learn the event component; learn their combined response; change the event radius to $0.45$; return to the original combination; return to the changed combination. Initial visits have caps of 384 observations, 4,608 measurements, 360,000 charged computation units, and six research rounds; revisits draw on the remaining sequence budget. The source-deletion arm removes $A$ after the shared two-task learning prefix while retaining $B$.

The learned combined program actually calls both sources. Under the mechanism change, the original combination encounters a measured counterexample. A successful revised expression preserves the exact definition of $A$ and compensates for the changed event contribution, with the schematic form

$$
D=C+0.7B_{0.45}-0.7B_r.
$$

The experiment thereby tests preservation inside an executable composition, followed by return to both the original and changed conditions.

**Table C3. Composition study.** Resource totals cover 12 complete six-task sequences per arm, including the common source-learning prefix.

| Arm | Tasks passing | Observations | Measurements | Charged computation |
|---|---:|---:|---:|---:|
| Retain $A,B$ | 72/72 | 7,002 | 29,484 | 3,415,926 |
| Delete $A$ | 72/72 | 10,334 | 57,804 | 5,800,741 |

Both arms meet the same terminal accuracy criterion on every task. Retention reduces all three charged resources in every instance. The cumulative reductions are 32.24% for observations, 48.99% for measurements, and 41.11% for computation. The complete procedural criterion passes in 11 of 12 instances: six of six under the development profile and five of six under the confirmation profile.

The remaining instance separates program recovery from direct calling. On the final task, it uses the stored research procedure to recover exactly the same program body as the earlier revision, but does not call the new definition $D$ directly. It uses 48 observations and 2,686 computation units; the other 11 retention instances use 24 observations and 2,575–2,583 computation units on that task. The retained transfer record and corresponding control flow indicate that a smaller response passed the short validation sample before being rejected on fresh evidence; this intermediate sequence is a reconstruction from the compact record. The subsequent method replay succeeds. This is a failure of the stipulated direct-call criterion and an instance of avoidable research cost, despite correct final prediction and actual procedure reuse.

## Appendix D. Joint-growth protocol and operational scope

### D.1 Response families and task changes

The joint study uses three finite response families,

$$
f_1(x,y)=a x^2+bxy,\qquad
f_2(x,y)=a x^3+bxy,\qquad
f_3(x,y)=a|x|+bxy,
$$

with apparatus-specific coefficients sampled in $a\in[0.8,1.2]$ and
$b\in[0.3,0.7]$. Each apparatus belongs to one family. Thirty new seeds are
balanced across the three families. The first surface implements the product
directly; later surfaces implement it as
$xy=((x+y)^2-x^2-y^2)/2$ and use different anonymous control names, coordinate
origins and units. A transfer must discover an empirically adequate binding.
These are controlled algebraic analogs, rather than independent real-world
scientific domains.

The eight-task sequence learns the source response, transfers it across a
surface change, adds $0.8\max(0,x)$ in one region, introduces an unobserved input,
changes the instrument names and their ordering, introduces random response
variation, simplifies the response to $0.8x-0.3y$, and returns to the original
nonlinear law. In the hidden-input tasks, the response contains
$z\sim\operatorname{Uniform}[-0.6,0.6]$. One auxiliary instrument measures an
affine transformation of $z$ and another measures an independent distractor.
Both have the same per-row price. The later instrument task reverses the
informative instrument's catalogue position, so replaying an old index cannot
by itself recover the appropriate view. In the stochastic task,
$\epsilon\mid x\sim\mathcal N(0,(0.08+0.16|x|)^2)$ is added to the response.
The researcher receives purchased observations and the public interface; family,
stage, latent means, and true noise scales are available only to the experiment
driver and evaluator.

Each task has ceilings of 512 observations, 1,536 measurement units, 300,000
candidate-row computation units and six rounds. The two main arms use identical
expression primitives and search algorithms. The persistent arm retains its
knowledge; the task-local arm starts each task without prior knowledge. Whole
sequences include all source acquisition, failed search and confirmation costs.
Process CPU time is measured separately from charged computation. Main-arm
execution order is randomized within each seed.

All research and prediction-diagnostic fitting finish before terminal data are
requested. Evaluation uses four fresh trajectories of 64 rows per task. A
deterministic task passes at MSE $\leq10^{-5}$. The stochastic task requires mean
prediction MSE $\leq0.03$ and mean predicted variance between one quarter and four
times the actual noise variance. The latter is a broad noise-detection criterion;
normal-interval coverage, standardized squared residuals, and variance estimation
error are reported separately. There are no external model calls.

### D.2 Interventions and eight operational capabilities

Additional branches copy an actually acquired prefix before changing one form
of knowledge or access. Program deletion removes the source concepts and their
derived program knowledge while retaining the method book. Method deletion
clears only the method book. The controls-only laboratory preserves the response
process but exposes no auxiliary measurement channel. All branches have the
same per-task resource caps. The prefix is physically acquired once and its cost
is included when reporting each counterfactual whole sequence.

| Capability in the formulation | Operational test in this instantiation |
|---|---|
| New abstractions after prior failure | Add $0.35y^3$ after source acquisition; compare retained and removed program knowledge under the same expression search. |
| Applicability and preservation | Learn the local change; compare its predictions with the frozen old response and, when identifiable from paid data, its new branch applied without the learned guard. |
| Structural correspondence | Transfer across anonymous role order, units and origin; compare retention with program deletion from the same source prefix. |
| Observation insufficiency | Compare access to paid instruments with a controls-only view of the same hidden-input process. |
| Research-procedure generation and revision | From the same measured prefix, retain or remove the procedure book before changing instrument roles; inspect and execute the generated fitting subquestions. |
| Uncertainty and reversible simplification | Assess a repeated-evidence variance model on fresh outcomes; compare simple-model execution with a dense fit on the same paid data, then test the nonlinear return. |
| Revising how hypotheses are proposed and tested | Combine the definition and procedure interventions with the complete persistent/task-local sequence. |
| Recovering experimental distinguishability | Test whether paid view changes and boundary-directed probes lead to fresh predictive success. |

The last two rows are properties of the connected loop, not two additional
independent algorithms. The static scope contrast identifies an old branch by
agreement on already acquired observations; it does not consult terminal targets.
When an equivalent hinge expression contains no identifiable conditional branch,
that diagnostic is recorded as unavailable. The dense cubic fit in the
simplification diagnostic uses the same paid observations, and its additional
fitting and evaluation costs are reported. These prediction diagnostics are
distinct from complete competing research algorithms.

Paired intervals resample apparatus seeds within each response family, using
10,000 bootstrap replicates and percentile endpoints at 2.5% and 97.5%. Tasks
within a sequence are not treated as independent replicates. All unsuccessful
research remains in the reported cohort. CPU for additional deletion branches
is not order-balanced against the retained prefix continuation; the primary
runtime comparison is the randomized complete-sequence comparison.

## Appendix E. Learned procedures in the legacy researcher

The growth study in Appendix D uses its own research loop. A subsequent
integration exposes the same procedure book through the legacy knowledge library.
The legacy agent records actual acquisition, fitting and fresh confirmation
steps, executes learned procedures through its existing acquisition and search
functions, and returns failed procedure evidence to its observation-revision
controller. Produced concepts retain their procedure origin, so later source
returns can reach the procedure book. Execution feedback and downstream knowledge
returns remain distinct.

The integration comparison uses 12 new apparatus seeds: six finite-history and
six recurrent instruments, with six tasks per sequence. Two source tasks are
acquired once; their complete session is cloned before retaining or removing only
the learned procedure list. Predictive concepts, instrument models, legacy
methods, experience and method-identity counters are identical at the fork. Both
arms can subsequently learn new procedures. Research caps are 384 observations,
1,536 measurement units and 360,000 charged computation units per task, with the
same session pooling rules. Whole-sequence costs include the common prefix in
each counterfactual arm. Terminal evaluation occurs after both arms finish and
uses four fresh random trajectories plus positive and negative impulse tests;
trajectory length respects each task's public limit. A joint pass requires
internal confirmation at MSE $\leq10^{-7}$ and both terminal tests at
MSE $\leq10^{-6}$. Paired intervals resample apparatuses within instrument mode.

Both arms pass confirmation and terminal evaluation on 72/72 tasks. Retaining
source procedures uses 4,334 observations, 9,006 measurement units and 1,038,795
charged computation units, compared with 4,092, 8,328 and 1,011,121 after removal:
increases of 5.91%, 8.14% and 2.74%. Training CPU totals are 59.21 versus 54.44
seconds. Actual procedure replay occurs 26 versus 33 times; the removal arm
learns new procedures and performs two procedure revisions. Neither arm requires
a task revisit in this cohort. All three paired resource-difference intervals
include zero (per-sequence observation difference: $[-3.33,42.83]$; measurement:
$[-9.25,120.75]$; computation: $[-8998.75,13306.92]$). The integration therefore
establishes execution through the legacy controller without demonstrating an
additional benefit from retaining its source procedures. All registered cases
are retained, with no execution failures or external model calls.

A separate existing continuation regression starts one task under a restricted
initial computation budget. A later task supplies learned knowledge, after which
the existing portfolio scheduler selects a revisit and the measured source return
reaches its producing procedure. The paid comparison loss falls from 0.3338 to
$6.64\times10^{-20}$, followed by fresh confirmation at
$4.28\times10^{-19}$. This is an operational integration test. The
producer and consumer have different measured response signatures. In this first
integration revision, strict condition matching does not use that return to choose
a procedure. These
results therefore separate shared storage, executable reuse, receipt delivery and
an economic benefit of that delivery. General object and scalar-continuation
procedure execution remain outside this integration experiment. Source and raw
results are preserved in `artifacts/workflow_bridge_20261006_main_12/`; the earlier
paper studies retain their frozen implementations and outcomes.

### E.1 Demand-conditioned procedure choice

A subsequent revision separates procedure applicability from the conditions under
which its knowledge helped a consumer. Unfinished visits supply demands through
their paid observations. The selector matches confirmed source returns to those
demands after filtering procedures for executable steps and acquisition and
confirmation budgets.

Twelve new paired cases first acquire a shared three-task prefix containing two
learned procedures and a confirmed source return. An independent researcher then
supplies an unfinished consumer visit under a 512-unit initial computation cap.
The next production task has a different response condition. The arms differ
only in retaining or deleting procedure return records; both retain predictive
knowledge and can acquire new returns. Each scheduled task has caps of 192
observations, 768 measurement units and 120,000 computation units. Terminal
evaluation uses two fresh 24-step trajectories after both arms finish.

Retention changes the native choice from `method_1` to `method_0` in 12/12 cases.
Both arms pass the 36 shared-prefix tasks but complete neither the 12 incoming
demands nor the 12 production tasks. Whole-sequence research costs are 9,888
observations and 11,808 measurement units in each arm, and 2,497,834 versus
2,497,809 computation units. The paired mean computation difference is +2.08
units (95% bootstrap interval [−3.08, 7.33]); the mean suffix MSE difference is
−0.00474 [−0.05159, 0.04572]. All cases are retained. This experiment establishes
that actual returns influence procedure choice, without a demonstrated research
benefit. It repeats a known consumer condition and tests choice inside an already
scheduled task; the historical return does not measure the marginal value of
producing more knowledge. Frozen source and results are in
`artifacts/workflow_demand_20261006_main_12/`.

### E.2 Complete research budgets and state identification

The next revision bounds all fitting callbacks within a replay by one shared
computation allowance. Observation-revision options specify enforceable packages
for fitting, distinguishing observations and fresh confirmation. When a measured
alternative is affordable, preliminary mechanism revision cannot consume the
observations needed to execute it. An existing continuous-state identifier also
enters the research-action policy with a complete quote for first- and second-order
identification, executable fitting and scoring up to six candidates. This allows
a small temporal hypothesis to compete before broad representation search uses
its budget. Identified programs replay input-driven state without past target
values; adoption still requires fresh confirmation.

A paired experiment uses 12 new seeds and the task family and caps of E.1. The
frozen prior implementation acquires each shared prefix and incoming unfinished
visit once. Both continuations retain identical predictive knowledge, procedures
and return records; only the subsequent implementation differs. All research
finishes before terminal scoring. Research costs include the common prefix in
each counterfactual bill, while terminal costs are reported separately.

The revised implementation completes fresh confirmation and terminal evaluation
on 12/12 production tasks, compared with 0/12 for the baseline. Mean terminal MSE
on these tasks falls from 0.2344 to $6.27\times10^{-18}$. Both arms retain 36/36
successful shared-prefix tasks and leave all 12 incoming demands unresolved.
Whole-sequence research costs are 9,600 versus 9,888 observations, 11,808
measurement units in both arms, and 2,043,146 versus 2,496,666 computation units:
2.91% fewer observations and 18.17% less charged computation. The paired mean
computation difference is −37,793 units, with a 95% interval of
[−40,445, −35,167] from 10,000 bootstrap resamples of the 12 cases.

At the first remeasurement, every revised run retains 104 observation rows,
compared with 80 in the baseline. All revised runs execute state identification
and confirm the resulting response. Each then has 32 observation rows left in
the session pool. In that implementation, the revisit scheduler requires 48 rows before screening an
unfinished scalar task, so neither source screening nor revisiting occurs.
The remaining demands therefore expose an allocation gate before candidate
evaluation; these cases do not measure whether the new knowledge could help them.

This comparison evaluates the combined allocation and execution-order revision.
It does not isolate a benefit of knowledge-return feedback or extend the model
language. The unsuccessful first development revision, which enabled earlier
measurement without completing the response, is retained alongside the successful
second development case. Protocol, frozen source and all cases are preserved in
`artifacts/workflow_budget_20261006_main_12/`.

### E.3 Reopening a paid demand within the remaining budget

The next comparison tests the continuation rule in Section 4.5. Screening includes
retained knowledge that the unfinished consumer has not previously encountered.
A candidate selected on paid observations is frozen for one fresh confirmation;
failed confirmation ends this package. Completed short-history fits can also be
challenged by later knowledge that predicts a distinguishable longer response.

Twelve new seeds use the same task family and ceilings as E.2. The original
implementation acquires the common prefix and unfinished consumer; the frozen
budget repair then finishes production. Both arms resume copies of this exact
post-production state, including all knowledge and costs, with 32 observation
rows remaining. The runner invokes the native session controller without choosing
a source or candidate. Both arms finish research before terminal evaluation.

The revised controller completes and passes terminal evaluation on 12/12 incoming
demands, versus 0/12 for the baseline. Production remains 12/12 in both arms,
and the shared prefix passes 36/36. Each successful revisit uses the consumer's
192 paid rows for selection, then buys one 24-row confirmation trajectory. It
adds 11,165 computation units, including screening and portfolio selection, and
no measurements. Mean incoming terminal MSE falls from 0.8538 to
$5.60\times10^{-19}$. Whole-sequence observations increase from 9,600 to 9,888
(3%), computation from 2,040,654 to 2,174,634 (6.57%), and measurement spending
remains 11,808. No case fails execution and no external model is called.

All selected offers derive from two retained prefix concepts. Protocol, frozen
implementation and outcomes are in `artifacts/paid_revisit_20261006_main_12/`.

To isolate the newly produced concept's contribution, a post-hoc intervention
reuses all 12 post-production checkpoints. Under the same frozen controller, one
arm retains that concept and the other removes it and its associated library
entries. Earlier concepts, research procedures, paid observations, production
reports and all incurred costs remain identical. Both continuations finish
before terminal scoring. Incoming demands and production tasks each pass 12/12
in both arms; final programs and fresh confirmations are identical in every
pair. Observations and measurements are unchanged. Retention increases whole
research computation from 2,151,594 to 2,174,634, exactly 1,920 additional
screening units per case. The new product therefore provides no marginal
predictive benefit for these demands and adds search cost. All cases complete
without an execution failure or external model call. The intervention and
receipts are in `artifacts/new_product_value_20261006_posthoc_12/`.

This attribution uses previously observed cases and preserves production costs;
it does not estimate the value of skipping production. The two prefix sources'
individual contributions and long-horizon net benefit remain open.

## References

Agarwal, A. K., and Yamada, M. (2026). Hypothesis-driven reasoning for large language models. *Proceedings of the AAAI Conference on Artificial Intelligence*, 40(3), 1686–1693. [doi:10.1609/aaai.v40i3.37146](https://doi.org/10.1609/aaai.v40i3.37146).

Alchourrón, C. E., Gärdenfors, P., and Makinson, D. (1985). On the logic of theory change: Partial meet contraction and revision functions. *The Journal of Symbolic Logic*, 50(2), 510–530. [doi:10.2307/2274239](https://doi.org/10.2307/2274239).

Bengio, Y., Courville, A., and Vincent, P. (2013). Representation learning: A review and new perspectives. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 35(8), 1798–1828. [doi:10.1109/TPAMI.2013.50](https://doi.org/10.1109/TPAMI.2013.50).

Boyd, S., and Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press. [Author-hosted book](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).

de Kleer, J. (1986). An assumption-based TMS. *Artificial Intelligence*, 28(2), 127–162. [doi:10.1016/0004-3702(86)90080-9](https://doi.org/10.1016/0004-3702(86)90080-9).

Ellis, K., Wong, C., Nye, M., Sable-Meyer, M., Cary, L., Morales, L., Hewitt, L., Solar-Lezama, A., and Tenenbaum, J. B. (2020). DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning. *arXiv:2006.08381*. [Preprint](https://arxiv.org/abs/2006.08381).

Hegel, G. W. F. (1892). *Lectures on the History of Philosophy*, Vol. I. Translated by E. S. Haldane. Kegan Paul, Trench, Trübner & Co. Introduction A.3; consulted in the 1955 reprint. [Public-domain text](https://www.gutenberg.org/files/51635/51635-h/51635-h.htm).

Huang, K., Jin, Y., Li, R., Li, M. Y., Candès, E., and Leskovec, J. (2025). Automated hypothesis validation with agentic sequential falsifications. *arXiv:2502.09858*. [Preprint](https://arxiv.org/abs/2502.09858).

Iacob, A., et al. (2026). The Red Queen Gödel Machine: Co-evolving agents and their evaluators. *arXiv:2606.26294*. [Preprint](https://arxiv.org/abs/2606.26294).

Iyengar, G. N. (2005). Robust dynamic programming. *Mathematics of Operations Research*, 30(2), 257–280. [doi:10.1287/moor.1040.0129](https://doi.org/10.1287/moor.1040.0129).

Le Goues, C., Nguyen, T., Forrest, S., and Weimer, W. (2012). GenProg: A generic method for automatic software repair. *IEEE Transactions on Software Engineering*, 38(1), 54–72. [doi:10.1109/TSE.2011.104](https://doi.org/10.1109/TSE.2011.104).

Novikov, A., et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery. *arXiv:2506.13131*. [Preprint](https://arxiv.org/abs/2506.13131).

Pearl, J. (2009). Causal inference in statistics: An overview. *Statistics Surveys*, 3, 96–146. [doi:10.1214/09-SS057](https://doi.org/10.1214/09-SS057).

Piriyakulkij, W. T., Langenfeld, C., Le, T. A., and Ellis, K. (2024). Doing experiments and revising rules with natural language and probabilistic reasoning. *Advances in Neural Information Processing Systems*, 37, 53102–53137. [Proceedings](https://papers.nips.cc/paper_files/paper/2024/hash/5f1b350fc0c2affd56f465faa36be343-Abstract-Conference.html).

Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K., and Yao, S. (2023). Reflexion: Language agents with verbal reinforcement learning. *arXiv:2303.11366*. [Preprint](https://arxiv.org/abs/2303.11366).

Solar-Lezama, A., Tancau, L., Bodík, R., Seshia, S. A., and Saraswat, V. A. (2006). Combinatorial sketching for finite programs. *Proceedings of ASPLOS*, 404–415. [Author publication page](https://people.eecs.berkeley.edu/~sseshia/pubs/b2hd-solar-asplos06.html).

Sutton, R. S., and Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press. [Publisher](https://mitpress.mit.edu/9780262039246/reinforcement-learning/).

Varici, B., Acartürk, E., Shanmugam, K., and Tajer, A. (2024). General identifiability and achievability for causal representation learning. *Proceedings of AISTATS*, PMLR 238, 2314–2322. [Proceedings](https://proceedings.mlr.press/v238/varici24a.html).

Wang, G., et al. (2023). Voyager: An open-ended embodied agent with large language models. *arXiv:2305.16291*. [Preprint](https://arxiv.org/abs/2305.16291).

Wong, L., Ellis, K. M., Tenenbaum, J., and Andreas, J. (2021). Leveraging language to learn program abstractions and search heuristics. *Proceedings of ICML*, PMLR 139, 11193–11204. [Proceedings](https://proceedings.mlr.press/v139/wong21a.html).

Yamada, Y., Lange, R. T., Lu, C., Hu, S., Lu, C., Foerster, J., Clune, J., and Ha, D. (2025). The AI Scientist-v2: Workshop-level automated scientific discovery via agentic tree search. *arXiv:2504.08066*. [Preprint](https://arxiv.org/abs/2504.08066).

Zhang, J., Hu, S., Lu, C., Lange, R. T., and Clune, J. (2025). Darwin Gödel Machine: Open-ended evolution of self-improving agents. *arXiv:2505.22954*. [Preprint](https://arxiv.org/abs/2505.22954).

Zhao, A., Huang, D., Xu, Q., Lin, M., Liu, Y.-J., and Huang, G. (2023). ExpeL: LLM agents are experiential learners. *arXiv:2308.10144*. [Preprint](https://arxiv.org/abs/2308.10144).

Zhou, P., et al. (2024). SELF-DISCOVER: Large language models self-compose reasoning structures. *arXiv:2402.03620*. [Preprint](https://arxiv.org/abs/2402.03620).
