# v13 Final Results — Simple Reflection Wins

Written 2026-04-22 after all v13 judges completed. This is the post-mortem on what works and why.

## TL;DR

**v13-reflect (simple 2-pass: draft → audit priors → revise) ties Self-Discover at 50-50 overall, and wins on soft domains.**

- First architecture from our line to match Self-Discover
- Mechanism: 2-pass reflection + prior-application audit. NOT scenario construction.
- v13-scenario (your original scenario-branch proposal) underperforms v13-reflect by 34pp.

## Full result matrix (gemini-3-flash generator + gemini-3-flash judge)

| A vs B | A wins | B wins | A wr |
|---|---|---|---|
| **v13-reflect vs ours_27** | 50 | 50 | **50%** |
| **v13-reflect vs baseline_long** | 74 | 26 | **74%** |
| **v13-reflect vs v12c** | 83 | 17 | **83%** |
| **v13-reflect vs v11** | 84 | 16 | **84%** |
| v13-scenario vs ours_27 | 29 | 71 | 29% |
| v13-scenario vs baseline_long | 59 | 41 | 59% |
| v13-scenario vs v11 | 61 | 39 | 61% |
| v13-scenario vs v12c | 65 | 34 | 66% |
| v13-scenario vs v13-reflect | 33 | 67 | 33% |
| ours_27 vs baseline | 81 | 19 | 81% |
| v11 vs ours_27 | 24 | 76 | 24% |
| v12c vs ours_27 | 26 | 74 | 26% |

## Domain-level v13-reflect vs ours_27 (the key comparison)

| Domain | v13-reflect | ours_27 | Winner |
|---|---|---|---|
| business | 9 | 6 | **v13-reflect 60%** |
| daily_life | 11 | 4 | **v13-reflect 73%** |
| engineering | 8 | 7 | v13-reflect 53% |
| software_engineering | 12 | 13 | ~ tie |
| science | 6 | 9 | ours_27 60% |
| mathematics | 4 | 11 | **ours_27 73%** |

**Interpretation**: v13-reflect dominates on "advisory / soft" problems — exactly where meta-knowledge application matters. ours_27 keeps advantage on pure computation (math/science) because its rigid JSON-fill structure enforces step-by-step rigor.

This validates the original meta-knowledge application hypothesis:
- Meta-knowledge helps on soft problems (v13-reflect uses it via audit)
- Meta-knowledge doesn't help on pure computation (ours_27's structure is more useful)

## Why v13-reflect beats v13-scenario

Expected: scenario construction explicitly unpacks meta → LLM applies more robustly.
Actual: scenario construction acts as a noisy constraint that the Turn 2 LLM must work around.

Hypotheses for why:
1. **Scenarios introduce framing errors**: Turn 1's scenarios instantiate meta-principles to ONE interpretation of problem structure. If that interpretation is off, Turn 2 inherits the error.
2. **Continuous interpolation is hard**: the instruction "don't dogmatically match one scenario, interpolate between them" is hard for LLM to follow. It tends to pick the highest-weight scenario and run with it.
3. **v13-reflect's audit IS implicit scenario generation**: when the LLM self-audits "did I apply prior X?", it's internally imagining what applying X would look like. That's a compressed scenario generation. Making it explicit (v13-scenario) adds steps without improving quality.
4. **Turn 1 token cost**: v13-scenario spends tokens on scenarios (~400 chars), leaving less cognitive budget for Turn 2. v13-reflect's Turn 2 gets the full budget.

## The broader insight

The mechanism that actually delivers gains is:
```
LLM produces draft → LLM audits its own draft against stated priors → LLM revises
```

**This is already well-known in literature as Self-Refine.** What our experiments add:

1. On-domain quantification: on 3-flash, simple reflection is ~equal to Self-Discover's 4-stage pipeline
2. Anti-pattern discovery: elaborate scenario generation (v13-scenario) underperforms simple reflection
3. Domain differentiation: reflection wins soft problems, Self-Discover wins math

## The paper direction

The original scaffold story ("phase2 beats baseline via prior injection") is dead on 3-flash. The real story is:

**Title direction**: *Simple Self-Audit Nearly Matches Structured Multi-Stage Reasoning in Modern LLMs — The Scaffold Ceiling Problem*

Contributions:
1. Demonstrate scaffold approaches (priors + triggers + wisdom) hit a ceiling on modern LLMs (3-flash era) — they can't beat a well-budgeted baseline
2. Show that 2-pass self-audit recovers most of the value of 4-stage pipelines at half the inference cost
3. Propose a hybrid router: reflection for soft domains, structured fill for computation
4. Counter-intuitive: elaborate scenario construction underperforms simple reflection
5. Release a clean eval set + protocol showing these patterns

## Practical recommendation

For production use on 3-flash-class models:

**For advisory / open-ended problems (business, daily_life, engineering, sw_eng):**
→ Use v13-reflect: priors + triggers + wisdom in Turn 1, then audit/revise in Turn 2

**For pure computation (math, science):**
→ Use Self-Discover-style JSON structure fill (ours_27 pattern) OR v12c hygiene

**Routing**: dispatch by domain detection at the start.

This combined architecture should beat any single approach. Test as v14-hybrid when priority allows.

## What we learned that matters

1. **"Inject more content" is dead.** Our 14-iteration v1-v12 line proved this.
2. **"Let LLM use full budget alone" (baseline_long) beats ad-hoc scaffolds.** The baseline itself is stronger than expected.
3. **"Add a reasoning pass" beats "add more content".** Structural change > content change.
4. **Simple > complex, for reasoning passes.** v13-scenario's scenarios were elaborate but hurt performance.
5. **Domain routing matters.** No single architecture wins everywhere; hybrid is correct.

## What's still uncertain

1. Is v13-reflect stable under:
   - Different random seeds?
   - Different judge models?
   - Held-out problems?
   (Only original 100 sample tested so far)

2. Does it scale to stronger generators (Claude Opus, GPT-5.4)?

3. Is there a further improvement via 3-turn (draft → audit → revise → final polish)? Or is 2-turn the sweet spot?

4. What's the latency/cost tradeoff? v13-reflect is ~2x baseline inference time. Worth it for +48pp?

## Next steps

See `experiment_roadmap.md` for priority order. Critical:
- P0: hold-out 50 validation of v13-reflect
- P0: regenerate v13-reflect on Opus/GPT-5.4 to test cross-model portability
- P1: build v14-hybrid (domain-routed: reflect for soft, ours_27 for hard)
