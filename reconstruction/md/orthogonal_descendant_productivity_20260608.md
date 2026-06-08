# Orthogonal Descendant Productivity 2026-06-08

## Question

The previous ON/OFF experiment showed that the same live-positive seed is
retained differently:

- orthogonal ON: `orthogonal_new_family` plus `orthogonal_to`;
- orthogonal OFF: `specialization` plus `specializes`.

The remaining question was whether this graph difference matters after the next
generation.  This experiment lets both retained graphs continue for three
generations of descendant proposals.

## Setup

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_descendant_productivity_20260608.json`

Seed:

`prop_9975d9e28cd7`, the scoped execution-contract hypothesis.

Seed evidence:

- same-model live acceptance;
- trigger: 3 wins, 2 ties, 0 losses;
- controls: 8 no-op ties;
- acceptance decision: `accept`.

Both branches start from the same seed evidence.  The only intended difference
is the retained graph topology.

## Three-Generation Loop

Each branch runs:

1. descendant proposal generation;
2. novelty/integration classification;
3. trigger/control acceptance gate;
4. gated temp apply of accepted descendants;
5. ACP/metaproductivity learning from accepted/rejected descendants.

No main graph mutation is performed.

## Result

Orthogonal ON:

- accepted descendants: 5;
- reject-harm descendants: 1;
- productivity score: 0.9569;
- max learned ACP score: 0.75;
- old-parent descendant labels: 0.

Orthogonal OFF:

- accepted descendants: 2;
- reject-harm descendants: 2;
- productivity score: 0.2708;
- max learned ACP score: 0.3448;
- old-parent descendant labels: 6.

Comparison:

- accepted descendant delta: +3;
- reject-harm delta, ON minus OFF: -1;
- productivity score delta: +0.6861;
- ACP score delta: +0.4052;
- old-parent label delta, OFF minus ON: +6;
- main graph mutation delta: 0;
- pass: true.

## Interpretation

This is stronger than the previous retention-only ablation.  The previous test
showed that ON/OFF changes how the accepted seed is represented.  This test
shows why the representation matters: once the seed is a separate orthogonal
execution-harness family, later proposals specialize metric calibration, owner
handoff, rollback, abstention, and traceable execution manifests under that
family.  When the same seed is folded back into `strategy_S01`, later proposals
are generated as mixed strategy/checklist repairs, creating more over-routing
and harm rejects.

The result is still bounded: descendant outcomes are deterministic
falsification fixtures derived from the live seed judge reasons, not fresh LLM
answer generation for every descendant.  The next empirical escalation is to
run the strongest ON descendants as live answer variants against the same-model
baseline.
