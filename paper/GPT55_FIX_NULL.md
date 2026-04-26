### Short answer

The evidence says the loop did **not** discover “wisdom”; it discovered **same-family stylistic preference under noisy n=50 selection**. The positive controls show the system is not intrinsically anti-additive: when given a **task-specific, procedural, error-targeted intervention**, both Gemini and GPT solvers improve in judged quality, and cross-family judges agree.

So the redesign should stop trying to evolve broad aphorisms over broad open-ended tasks. It should instead mine concrete solver failure modes, generate compact procedural “patches” for those failure modes, and gate them on fresh, preregistered, cross-family, slice-specific evaluations.

---

## (1) Bottleneck diagnosis

Ranked:

### 1. Candidate generator / wisdom form: primary bottleneck

The orchestrator is producing paraphrases of existing aphoristic advice: “consider alternatives,” “state assumptions,” “verify,” etc. These are too generic. Strong LLMs already contain these priors. Adding another aphorism gives mostly style, not capability.

The positive controls worked because they were **specific**:

> If Bayesian/base-rate problem, instantiate counts, separate likelihood from posterior, compute explicitly, verify against intuition.

That is not an aphorism. It is an executable cognitive patch.

### 2. Task distribution: second bottleneck

“Chinese open-ended problems” are too heterogeneous and noisy. A generic aphorism cannot have a large enough average treatment effect across that mixture. If a candidate helps 10% of problems by +20 points but hurts/does nothing elsewhere, n=50 pairwise judging will be dominated by noise and style.

The right unit is not “all Chinese open-ended problems.” The right unit is a **failure slice**: Bayesian base rates, hidden constraints, adversarial wording, multi-hop arithmetic, quantifier traps, causal confusion, etc.

### 3. Gate: measurement bottleneck, not capability bottleneck

Same-family Gemini judging at n=50 selected artifacts. Cross-family and n=200 killed them. So the gate was over-sensitive to Gemini-style outputs and winner’s curse.

But the gate did accept real interventions in Exp 68/69. Therefore the gate is not fundamentally impossible; it just needs preregistered fresh data, larger n, cross-family judging, and objective metrics when possible.

### 4. Solver: not main bottleneck

The solver is strong, sometimes ceilinged, but Exp 68/69 show it still responds to strong interventions. The issue is not “LLMs cannot use wisdom.” The issue is “generic aphorisms are already internalized and too weak.”

---

## (2) Minimal redesign likely to produce `wisdom_count > 0` at n=200 cross-family

Do not rebuild everything. Change the loop target from broad aphorism discovery to **failure-mode patch discovery**.

### Step A: Build failure slices

Create 5–10 preregistered task slices where baseline models are imperfect but not floor/ceiling.

Examples:

1. Bayesian/base-rate reasoning  
2. Conditional probability wording traps  
3. Constraint satisfaction / scheduling  
4. Hidden quantifier logic: “all/some/not necessarily”  
5. Chinese legal/administrative reading with exceptions  
6. Multi-step arithmetic embedded in prose  
7. Causal vs correlational explanation  
8. Counterfactual reasoning  
9. Ambiguous instruction conflict resolution  
10. Planning with resource constraints

For each slice, collect maybe 300–500 problems. Split into:

- discovery/train failures,
- dev gate,
- preregistered fresh test n=200.

Do not let the generator see the final test.

### Step B: Generate from errors, not from aphorisms

For each slice:

1. Run baseline solver.
2. Collect wrong or low-quality answers.
3. Cluster errors.
4. Ask generator to produce candidate interventions that specifically prevent those errors.

Candidate format should be constrained:

```text
Name:
Trigger: When the problem contains...
Failure prevented:
Micro-procedure:
  1.
  2.
  3.
Verification check:
Bad pattern to avoid:
Optional tiny example:
```

This is the key change. The unit of evolution becomes an **error-correcting procedure**, not a proverb.

### Step C: Gate on matched fresh slice, not broad mixture

For candidate `C` targeting slice `S`, test on fresh problems from `S`.

Use at least three measurements:

1. **Objective accuracy**, if possible.
2. **Cross-family pairwise win rate**, e.g. Gemini, Claude, GPT judge.
3. **Negative-control / ablation**, e.g. remove the procedural core and keep style constant.

Acceptance rule could be:

```text
KEEP if:
- n=200 fresh preregistered
- cross-family mean win rate >= 0.60
- lower 80% or 90% interval above 0.55
- no judge family below 0.56
- objective accuracy nonnegative and preferably positive
- ablated/stylistic version does not pass
```

For slices with objective answers, require actual accuracy lift, not only pairwise preference.

### Step D: Maintain routing/indexing

A library of 500 generic wisdoms dumped into every prompt will fail. Each wisdom needs a router.

At inference:

```text
classify problem slice -> retrieve 1–3 relevant patches -> apply patch
```

A Bayesian template should not be injected into poetry interpretation. A legal-exception checklist should not be injected into arithmetic.

### Step E: Use exemplars deliberately

Exp 68/69 show exemplars matter. But do not let exemplars be the whole intervention. Compare:

1. template only,
2. examples only,
3. template + examples.

Keep only candidates where the reusable mechanism survives, or explicitly label the wisdom as an exemplar-backed skill.

### Minimal viable redesign

If you want the smallest change likely to produce survivors:

> Replace the 75-aphorism seed library with 20 hand-written “micro-procedure cards” for known failure slices, have the generator mutate those using failure traces, and gate each candidate on n=200 fresh slice-specific tasks with cross-family judges.

I would expect >0 survivors from Bayesian, constraint, and quantifier slices.

If you keep the broad Chinese open-ended distribution and aphorism form, I would expect 0 again.

---

## (3) A different kind of wisdom that could work

Yes: **triggered cognitive patches**.

They are between aphorisms and full algorithmic templates.

They are not broad sayings like:

> “兼听则明，偏信则暗.”

And they are not full worked curricula.

They are compact, conditional, executable habits.

Example:

```text
Name: Denominator Lock

Trigger:
The problem gives percentages, base rates, or conditional probabilities
across different groups.

Patch:
Before comparing probabilities, choose one denominator, preferably 1,000
or 10,000 cases. Convert every rate into counts on that denominator.
Only then compute the requested probability.

Verification:
Check that you did not confuse P(evidence | hypothesis) with
P(hypothesis | evidence).

Failure prevented:
Base-rate neglect and likelihood/posterior inversion.
```

Another:

```text
Name: Exception First

Trigger:
A rule is followed by “except,” “unless,” “only if,” “provided that,”
or Chinese equivalents like “但,” “除非,” “仅当.”

Patch:
Rewrite the main rule and the exception as separate if-then clauses.
Evaluate the exception before applying the default rule.

Verification:
Ask: “Is this case inside the exception set?”
```

Another:

```text
Name: Quantifier Swap Check

Trigger:
The answer depends on words like all, some, none, must, may, necessarily,
可能, 必然, 所有, 至少.

Patch:
Translate the claim into symbolic form. Then test whether reversing
the quantifier changes meaning. Do not infer “all A are B” from
“all B are A.”

Verification:
Construct a two-item counterexample.
```

These are “wisdom” in a useful sense: compressed procedural knowledge with a trigger, an operation, and a check.

The loop should evolve these.

---

### Bottom line

The failed loop optimized vague, high-level advice over a noisy, broad distribution using a same-family stylistic judge. The successful positive controls used narrow, mechanistic, task-specific procedures with examples.

So the redesign is:

> Mine failures → cluster into slices → generate triggered micro-procedures → route them selectively → evaluate on fresh n=200 slice-specific tasks with cross-family judges and objective metrics.

That is the minimal path to real surviving wisdoms.