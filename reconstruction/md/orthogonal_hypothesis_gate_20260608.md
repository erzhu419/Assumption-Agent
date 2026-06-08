# Orthogonal Hypothesis Gate - 2026-06-08

## Current Hypothesis Representation

The current system represents hypotheses at three levels:

- graph substrate: `AssumptionNode` plus `AssumptionEdge`;
- bounded diagram payload: `AssumptionNode.formal_form` and `payload`, which can store objects, morphisms, invariants, functor/kernel checks, transfer predictions, and negative controls;
- falsifiable lifecycle record: proposal payloads, trial manifests, verifier summaries, acceptance/rejection records, and novelty/integration metadata.

So a hypothesis is not only text. It is a graph node with typed context, expected effects, risks, evidence, residuals, optional structural diagram, and integration edges.

## Gap To A Full Category-Theory Reasoning Engine

The current layer is still a bounded structural morphism layer, not a complete category-theory engine. The remaining gaps are:

- no general category object model with verified identities, composition closure, functors, natural transformations, limits/colimits, adjunctions, or monoidal structure;
- no theorem prover for commutative diagrams or naturality conditions;
- no probabilistic Markov-category or information-geometry engine that proves Blackwell/Fisher/KL relations;
- diagram extraction is heuristic and finite, not a proof-producing parser;
- transfer is validated empirically through gates and ablations, not formally guaranteed;
- the orthogonality relation is a deterministic engineering gate, not an inner-product proof in a formally defined hypothesis Hilbert space.

The safe manuscript claim remains: `category-inspired bounded structural morphism layer`.

## Orthogonal Hypothesis Idea

The MC-WM observation was: a newly proposed hypothesis can be especially useful when it is orthogonal to the existing hypothesis set rather than a near-duplicate or minor local edit.

This is now implemented in `novelty_integration.py` as `orthogonal_new_family`.

The rule is intentionally narrow:

- the candidate must be grounded in the same residual, parent, or `generated_from_residual` edge;
- it must remain low-overlap with the best existing node and with the parent node;
- it must not share a substantive family tag or formal family key with the parent;
- it must not already qualify as duplicate, specialization, formal isomorphism, or ordinary analogy;
- accepted candidates are integrated with an `orthogonal_to` edge and a new-family action;
- `orthogonal_to` is counted as a productive edge for clade/metaproductivity tracing.

This turns "orthogonal novelty" into a measurable integration policy rather than a prompt slogan.

This also clarifies the Le Chatelier / Lenz-law case: those are not orthogonal. They are two realizations of the same negative-feedback family and should be represented with `is_formal_isomorphism_of`, `is_analogy_of`, or family membership. Orthogonality applies when a new candidate explains the same residual through a different mechanism axis, such as switching from a strategy-defect hypothesis to an evaluator-drift hypothesis.

## Performance Validation

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/novelty_integration_gate_20260604.json`

Result:

- proposal_count: 6
- classified_count: 6
- gold_accuracy: 1.0
- required_classes_present: true
- formal_edges_recommended: true
- analogy_edges_recommended: true
- orthogonal_edges_recommended: true
- orthogonal_rows_are_new_families: true
- pass: true

The fixture now covers:

- duplicate
- specialization
- formal_isomorphism
- analogy
- genuinely_new_family
- orthogonal_new_family

Unit tests also include a negative control: a candidate that declares itself orthogonal but shares the parent family tag is classified as `specialization`, not `orthogonal_new_family`.

## Next Validation

This is functionally validated, but not yet proven as a downstream performance gain. The next experiment should toggle the orthogonal gate inside a real recursive benchmark line:

`failure cluster -> candidate set -> novelty/orthogonality gate -> ablation + controls -> retention -> next generation`

The relevant metric is whether orthogonal candidates have higher long-run clade metaproductivity than near-duplicate or incremental candidates after controlling for immediate utility and cost.
