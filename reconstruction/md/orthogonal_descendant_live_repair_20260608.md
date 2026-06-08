# Orthogonal Descendant Live Repair - 2026-06-08

## What Was Missing

The previous 3-generation orthogonal descendant benchmark proved recursive
productivity with deterministic falsification fixtures, but it had not yet shown
that a generated descendant could pass real same-model answer/judge validation.

## Retained-Graph Setup

The accepted execution-contract seed is not written into the main graph by
default.  To test its descendant honestly, the queue now builds a frozen retained
graph snapshot:

- base graph: `phase four/assumption_graph`
- retained snapshot: `phase four/assumption_graph/paper_readiness_20260604/orthogonal_descendant_live_graph_20260608`
- retained seed: `cand_39de0aeae8a3`
- descendant parent: `cand_39de0aeae8a3`
- main graph mutation: false

Queue-level validation passed:

- proposal count: 1
- novelty classification: specialization
- specializes edge count: 1
- trigger rows: 5
- active trigger rows: 5
- route-scoped controls: 8
- outside activation: 0
- fixture readback: accept
- fixture temp apply: writes candidate

## V1 Negative Result

The first live descendant was a compact execution-manifest variant.  It completed
the live answer/judge pipeline but failed benefit:

- proposal: `prop_f6ea5d74b785`
- decision: reject_benefit
- trigger winners: candidate 0, baseline 3, ties 2
- controls: route-scoped no-op ties

Failure pattern:

- it sometimes changed a hard user constraint instead of preserving it
- it omitted task-specific bridges such as secrecy/access control, option
  comparison, role/KPI alignment, and waiting-variance control
- it was a generic execution checklist rather than a constraint-preserving
  execution bridge

## V2 Repair

The repaired descendant is:

- proposal: `prop_d7abf65010d2`
- candidate: `cand_f8ca2582dbc4`
- claim: preserve hard user constraints and core assets first, bridge to
  task-specific risks, then add compact execution fields

Live same-model validation:

- status: live_positive_acceptance
- judged trigger pairs: 5
- route-scoped no-op controls: 8
- trigger outcomes: 4 wins, 1 loss
- trigger utility: 0.8
- trigger LCB90: 0.5710
- control outcomes: 8 ties
- control loss UCB90: 0.0
- decision: accept

Repair delta:

- candidate trigger wins: 0 -> 4
- baseline trigger wins: 3 -> 1
- acceptance: reject_benefit -> accept

## Interpretation

This is the first live-positive descendant after the orthogonal seed retention
line.  The recursive loop is now stronger than a fixture-only demonstration:

failure -> repaired descendant hypothesis -> live ablation -> judge ->
acceptance gate.

The accepted improvement is not "more checklist"; it is a stricter assumption:
preserve the problem's hard constraints before adding execution commitments.

## Live Readback

The accepted v2 judgment was also fed back through the bounded daemon without
rerunning the model:

- input judgment: `proposal_d7abf65010d2` vs
  `phase2_v20_claude_opus_execution_baseline`
- recomputed acceptance: accept
- daemon readback accept count: 1
- readback apply count: 0
- node mutation without apply: false
- gated temp apply count: 1
- applied candidate status: active
- applied edge type: specializes
- retained graph snapshot mutated: false

This closes the operational loop for this descendant:

real live judgment -> acceptance gate -> recursive daemon readback -> gated temp
apply.
