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
