# Orthogonal Ablation - 2026-06-08

## Question

Does the `orthogonal_new_family` gate add measurable behavior beyond the generic
novelty/integration gate?

The specific risk is new-axis collapse: a candidate that explains the same
residual through a different mechanism axis can be misintegrated as a
`specialization` of the old parent family.  That loses the distinction between
"same family, narrower scope" and "same residual, different explanatory axis".

Le Chatelier's principle and Lenz's law are not orthogonal in this sense. They
belong to the same negative-feedback family, or should be connected through
`is_formal_isomorphism_of` / `is_analogy_of`. Orthogonality applies when a new
hypothesis is low-overlap with the old family while still grounded in the same
failure/residual, such as moving from controlled-variable reasoning to
evaluator-drift diagnosis.

## Implementation

New code:

- `assumption_os/novelty_integration.py`: added `enable_orthogonal` so the gate
  can be toggled without changing the proposal batch or other thresholds.
- `assumption_os/orthogonal_ablation.py`: runs the same fixture with
  orthogonal ON/OFF and reports classification, integration-edge, axis-retention,
  family-contamination, and recursive metaproductivity proxy metrics.
- `tests/test_assumption_os.py`: added a performance validation test for the
  ON/OFF comparison.

## Validation Scope

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_ablation_20260608.json`

This is a deterministic recursive-retention proxy, not a live downstream QA or
LLM-judge ablation. It is stronger than fixture correctness because it toggles
the mechanism on the same proposal set and measures the integration consequence
of disabling it.

## Expected Interpretation

With orthogonal ON:

- the evaluator-drift proposal is classified as `orthogonal_new_family`;
- it receives an `orthogonal_to` edge;
- the new explanation axis is retained as its own family;
- recursive descendant outcomes remain separated by family.

With orthogonal OFF:

- the same proposal falls back to `specialization`;
- it receives no `orthogonal_to` edge;
- the evaluator-drift axis collapses into the controlled-variable parent;
- the recursive proxy sees family-axis contamination.

## Paper Claim Boundary

This supports the claim that the system can represent and preserve orthogonal
new hypothesis families during recursive self-evolution. It does not yet prove
that orthogonality improves live QA, heldout solver score, or long-run
autonomous discovery under LLM calls. The next stronger validation is a live
recursive benchmark with orthogonal ON/OFF over generated residual clusters.

## Surface False-Positive Guard

Follow-up artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_surface_ablation_20260608.json`

The first real generated surface-proposal check found an important edge case:
surface tags can use aliases of the same family, for example
`world_model_screen` vs `world_model`, or `recursive_assumption_runner` vs
`recursive_runner`. These must not be treated as orthogonal.

The novelty gate now canonicalizes such family aliases before testing
orthogonality. On the real surface proposal batch, the expected behavior is zero
`orthogonal_to` edges: these are same-family repairs/scopes, not new orthogonal
families. The fixture ablation still validates true orthogonal retention.

## Judged Downstream Negative Control

Follow-up artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_downstream_ablation_20260608.json`

The existing cached acceptance artifacts contain real judged outcomes for six
non-orthogonal candidate proposals. Toggling the orthogonal gate on these rows
must not change novelty classification or downstream accept/reject decisions.

Result:

- judged_proposal_count: 6
- judged_classification_change_count: 0
- judged_false_orthogonal_count: 0
- enabled_orthogonal_edge_count_all_proposals: 0

This is a downstream safety check, not a positive live orthogonal-benefit check.
No cached answer/judgment artifact currently exists for a proposal that is
actually classified as `orthogonal_new_family`; that remains the next live run.

## Positive Live-Ready Queue

Follow-up artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_positive_queue_20260608.json`

The missing positive arm is now queued rather than only described.  The artifact
constructs a real candidate under `strategy_S01`:

- candidate axis: evaluator/rubric drift rather than controlled-variable method
  repair;
- grounding: same residual/parent via `generated_from_residual`;
- novelty ON: `orthogonal_new_family`, with one `orthogonal_to` edge;
- novelty OFF: collapses to a non-new-family specialization;
- preflight: `ready_for_fresh_ablation`;
- trigger rows: 4;
- control rows: 8;
- no no-fire exposure;
- command hints use environment variable placeholders only.

This validates that the system can produce a positive orthogonal new-family
candidate and route it into fresh ablation.  It still does not claim a live
downstream answer-quality win; that requires running the emitted answer and
pairwise-judgment commands with API credentials supplied through environment
variables.

## Daemon Readback Bridge

Follow-up artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_positive_readback_20260608.json`

The positive queue now connects to the bounded recursive daemon path:

- daemon dry-run consumes one ready leaf;
- dry-run records planned execution without node mutation;
- fixture judgments in the same `candidate_acceptance` format produce one
  accepted proposal and resume the recursive parent;
- gated apply on a temporary graph writes the candidate node and its
  `orthogonal_to` edge;
- main graph remains untouched unless a later real acceptance payload is applied
  through the existing gate.

This removes the engineering gap after live judgments are produced. The only
remaining empirical gap for this positive arm is the live answer-quality
judgment itself.

## Multi-Cluster Positive Arm

Follow-up artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_multi_cluster_20260608.json`

The positive arm is now tested on three distinct real graph parents rather than
a single curated candidate:

- `strategy_S01`: evaluator/rubric drift as a new axis;
- `strategy_S25`: simulator-scale/surrogate gap as a new axis;
- `strategy_S26`: provenance/archive memory gap as a new axis.

Performance validation result:

- proposal_count: 3
- distinct_parent_count: 3
- enabled_orthogonal_count: 3
- disabled_orthogonal_count: 0
- enabled_orthogonal_edge_count: 3
- disabled_orthogonal_edge_count: 0
- preflight_ready_count: 3
- min_trigger_count: 4
- min_control_count: 8
- dry_planned_leaf_count: 3
- readback_accept_count: 3
- readback_applied_count: 0
- apply_applied_count: 3
- temp_orthogonal_edge_count: 3
- pass: true

This is still a mechanism validation, not a live answer-quality result.  The
important improvement is breadth: the gate now preserves several orthogonal
new explanatory axes and the recursive daemon can consume, read back, and
gated-apply them.  A real ON/OFF downstream utility claim still requires
running the emitted fresh-ablation and pairwise-judge commands.

## Execution-Contract Live Positive Arm

Follow-up artifacts:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_execution_queue_20260608.json`

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_execution_live_same_model_20260608.json`

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_execution_scope_repair_20260608.json`

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_execution_scope_repair_live_same_model_20260608.json`

The first same-model live positive arm is now present, with an important
negative boundary:

- Broad execution-contract hypothesis: same-model live run passed execution
  integrity but was rejected by benefit gate, with 4 wins, 3 ties, and 1 loss
  on 8 triggers.
- Recursive scope repair: narrowed to practical operations/transition rows,
  then passed same-model live acceptance with 3 wins, 2 ties, 0 losses on 5
  triggers and 8 route-scoped no-op control ties.

This shows the intended recursive behavior: do not promote an over-broad
orthogonal hypothesis; read the residual, repair the activation scope, rerun
fresh ablation, and only retain the scoped hypothesis after acceptance.

## Recursive ON/OFF Retention Ablation

Follow-up artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_recursive_ablation_20260608.json`

The scoped live-positive execution-contract hypothesis was then replayed through
the recursive runner with orthogonal novelty toggled ON/OFF.  The downstream
judgment evidence was held constant, so this is not an answer-quality retest; it
is a recursive retention test.

Result:

- live acceptance reused: 3 wins, 2 ties, 0 losses on trigger rows; 8 no-op
  control ties;
- orthogonal ON: classified as `orthogonal_new_family`, temp apply wrote
  `orthogonal_to`;
- orthogonal OFF: classified as `specialization`, temp apply wrote
  `specializes` and no `orthogonal_to`;
- downstream utility delta: 0.0;
- recursive retention delta: +2.25;
- main graph mutation delta: 0;
- pass: true.

This answers the narrower ON/OFF question: the gate's immediate benefit is not
that it improves the same accepted answer again, but that it keeps the accepted
hypothesis as a separate execution-harness family for future descendants rather
than folding it into the old parent strategy.

## Three-Generation Descendant Productivity

Follow-up artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_descendant_productivity_20260608.json`

The retained ON/OFF graphs were then allowed to generate and evaluate three
generations of descendant proposals.  Both branches start from the same
live-positive seed.  The descendant loop runs proposal generation,
novelty/integration, trigger/control acceptance, gated temp apply, and ACP
learning.

Result:

- orthogonal ON accepted descendants: 5;
- orthogonal OFF accepted descendants: 2;
- accepted descendant delta: +3;
- reject-harm delta, ON minus OFF: -1;
- productivity score delta: +0.6861;
- ACP score delta: +0.4052;
- old-parent label delta, OFF minus ON: +6;
- main graph mutation delta: 0;
- pass: true.

This gives the first bounded evidence that orthogonal retention changes later
self-evolution behavior, not just the first retained edge.  The ON branch keeps
the execution-contract seed as a family and produces more productive
descendants; the OFF branch folds descendants into `strategy_S01`, causing more
mixed-axis proposals and harm rejects.
