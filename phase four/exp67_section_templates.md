# Exp 67 §3.6 Templates — pick based on outcome

## Outcome A — audit stack works
> §3.6 — Positive control on objectively-graded tasks: the audit stack is sensitive
>
> The most consistent reviewer concern was that 6/6 prior positive-control wisdoms (Exp 44, Exp 46) failed the gate, leaving the audit stack's sensitivity unverified. To address this we ran a positive control where correctness is independently verifiable. We took 30 hand-curated problems from Exp 62 with known numerical gold answers — 15 Bayesian/probabilistic-trap problems (Family E) and 15 multi-step arithmetic problems (Family F) — and produced three answers per problem using the same solver and prior-injection mechanism: BASE (no prior), POSITIVE (the family's optimal prior — `constraints` for E, `decompose` for F), and NEGATIVE (a deliberately wrong prior — `estimate` for E, `restate` for F). We graded each answer objectively against gold, then ran pairwise A/B judging with the *exact same* `judge_pair` function the original wisdom-loop gate used (gemini-3-flash) and a cross-family L1 audit (claude-haiku).
>
> Objective accuracy: BASE = X%, POSITIVE = Y%, NEGATIVE = Z%, with Δacc(POS−BASE) = +N pp and Δacc(NEG−BASE) = −M pp. Pairwise A/B verdicts: POSITIVE-vs-BASE wr_inner = A, wr_L1 = B; NEGATIVE-vs-BASE wr_inner = C, wr_L1 = D. The gate accepted the objectively-helpful prior (wr ≥ 0.60) and rejected the objectively-harmful one (wr < 0.40); the L1 cross-family audit confirmed both directions. The audit stack is therefore demonstrably sensitive to known-positive insertions on objectively-graded tasks. The original 12-candidate null at n ≥ 100 cannot be attributed to a structurally anti-additive gate; the candidates are most parsimoniously explained as not actually delivering content-specific utility on the original Chinese open-ended task domain.

## Outcome B — same-family judge has style bias; L1 saves it
> §3.6 — Positive control on objectively-graded tasks: same-family A/B has rhetorical bias; L1 audit corrects
>
> [same setup paragraph as Outcome A]
>
> Objective accuracy: BASE = X%, POSITIVE = Y%, NEGATIVE = Z%, with Δacc(POS−BASE) = +N pp. The objective signal is therefore real and positive: the prior demonstrably improves correctness. Pairwise verdicts diverge: POSITIVE-vs-BASE wr_inner = A (same-family) is BELOW 0.55 even though the prior helps objectively, while wr_L1 = B (cross-family) is at or above 0.55. NEGATIVE-vs-BASE shows wr_inner = C, wr_L1 = D.
>
> Interpretation: the same-family judge has a rhetorical preference against insertions — likely because base answers are slightly more concise than prior-augmented ones, and gemini-3-flash equates concision with quality on this prompt format. The cross-family L1 layer detects the objective improvement that the same-family judge misses. **This is a strong empirical case for L1 as a non-optional audit layer, not a redundant secondary check.** It also explains why the 6/6 prior positive-control wisdoms (Exp 44/46) all failed the same-family gate — they may have been objectively useful but the gate was not capable of detecting that.

## Outcome C — gate is structurally anti-additive
> §3.6 — Positive control on objectively-graded tasks: the gate is structurally anti-additive
>
> [same setup paragraph]
>
> Objective accuracy: BASE = X%, POSITIVE = Y%, NEGATIVE = Z%, with Δacc(POS−BASE) = +N pp. The prior demonstrably helps on the ground-truth metric. **Yet both the same-family gate and the cross-family L1 audit reject the objectively-better answer**: POSITIVE-vs-BASE wr_inner = A < 0.55, wr_L1 = B < 0.55. NEGATIVE-vs-BASE shows wr_inner = C, wr_L1 = D.
>
> This is the strongest possible vision-paper result. **A pairwise LLM-judge gate cannot be relied on to detect known-positive interventions**, even when (i) the intervention measurably improves correctness on objectively-graded ground truth, (ii) the gate uses the exact same judge function as the paper's original wisdom loop, (iii) the cross-family audit panel sees the same answer pairs. The 0/12 null on the original 12 candidates is therefore not informative about the candidates themselves — it tells us only that the gate-and-audit architecture as a whole is fundamentally unsuited to detecting useful content-level interventions. Same-family A/B at n=50, even when supplemented by the full audit stack, is unfit for the role of validator in any LLM self-improvement loop. The roadmap of §5 then follows directly: priors, scheduler, world-model verification, and alignment — not better gates.

## Outcome D — positive control failed
> §3.6 — Positive control attempt on objectively-graded tasks: prior selection failure
>
> [same setup paragraph]
>
> Objective accuracy: BASE = X%, POSITIVE = Y%, NEGATIVE = Z%. Δacc(POS−BASE) = +N pp ≈ 0; the chosen prior did not measurably improve objective correctness. We cannot use this run as a positive control. Two non-mutually-exclusive explanations: (i) the solver model (gemini-3-flash) already self-applies the optimal prior on these problems regardless of explicit instruction, so adding an explicit prior is informationally redundant; or (ii) the priors as natural-language hints are too generic to lock the solver into the correct reasoning track on these specific problem types. Constructing a stronger positive control — e.g. inserting a problem-specific solving template or using a model with weaker baseline performance — remains a key open task for future work.
>
> [In this case, leave the Discussion's "positive controls remain unresolved" framing intact.]
