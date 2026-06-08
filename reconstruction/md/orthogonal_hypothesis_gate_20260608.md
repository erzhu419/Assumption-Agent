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
- it must remain low-overlap with the best existing node and with the parent node under an informative-token similarity check that filters generic stopwords, IDs, and lifecycle metadata;
- it must not share a substantive family tag, canonical family alias, or formal family key with the parent;
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

## Orthogonal Ablation Update

Follow-up artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_ablation_20260608.json`

The gate is now toggled ON/OFF on the same proposal batch.  With the gate ON,
the evaluator-drift candidate is retained as `orthogonal_new_family` with an
`orthogonal_to` edge.  With the gate OFF, the same candidate collapses into the
old controlled-variable family as a `specialization`.

This validates the mechanism as a recursive-retention proxy: the ON condition
preserves the new explanation axis and avoids family-axis contamination.  It is
still not a live downstream QA or LLM-judge ablation.

Second follow-up artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_surface_ablation_20260608.json`

This checks the opposite failure mode on real generated surface proposals. It
confirms that same-family aliases such as `world_model_screen` / `world_model`
and `recursive_assumption_runner` / `recursive_runner` do not become orthogonal
new families. On that batch, zero `orthogonal_to` edges is the correct behavior.

The next stronger experiment is to run orthogonal ON/OFF on live generated
residual clusters with answer-quality or judge-based downstream outcomes.

Third follow-up artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_downstream_ablation_20260608.json`

This uses existing real judged candidate-acceptance artifacts as a downstream
negative control. Six non-orthogonal judged proposals keep the same novelty
classification and downstream accept/reject decision under orthogonal ON/OFF.
So the gate is downstream-safe on the current judged non-orthogonal proposal
line. The positive live orthogonal-benefit test still requires fresh judgments
for an actual `orthogonal_new_family` proposal.

Fourth follow-up artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_positive_queue_20260608.json`

The system now builds a positive orthogonal candidate that is ready for live
fresh ablation.  It uses the real graph parent `strategy_S01`, but proposes a
different explanatory axis: before editing controlled-variable reasoning, test
whether the residual is caused by evaluator/rubric drift.  This is exactly the
intended meaning of orthogonality here: same failure context, different
mechanism family.

Validation:

- orthogonal ON classification: `orthogonal_new_family`;
- orthogonal OFF classification: non-new-family specialization;
- `orthogonal_to` edge count: 1 when enabled, 0 when disabled;
- preflight readiness: `ready_for_fresh_ablation`;
- trigger/control split: 4 trigger rows and 8 control rows;
- secret handling: commands contain only `<set-in-env>` placeholders.

The remaining step is live answer-quality judging.  The code does not mutate the
graph from this queue unless later judgments pass the existing gated acceptance
path.

Fifth follow-up artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_positive_readback_20260608.json`

The queued positive candidate now also passes the daemon/readback bridge.  A
bounded daemon dry-run consumes the leaf, fixture judgments in the same format
as real pairwise judge output resume the parent frame, and a gated apply on a
temporary graph writes both the candidate node and `orthogonal_to` edge.

This closes the operational path around the positive candidate.  The honest
remaining claim boundary is empirical: run the queued live answer and judge
commands, then let the same readback path accept/reject from real judgments.

Sixth follow-up artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_multi_cluster_20260608.json`

The positive arm has been expanded from one example to three different
residual/parent clusters:

- `strategy_S01` -> evaluator/rubric drift;
- `strategy_S25` -> simulator-scale/surrogate gap;
- `strategy_S26` -> provenance/archive memory gap.

Validation:

- proposal_count: 3;
- distinct_parent_count: 3;
- orthogonal ON classification: 3 `orthogonal_new_family`;
- orthogonal OFF classification: 0 `orthogonal_new_family`, 3 specializations;
- `orthogonal_to` edges: 3 when enabled, 0 when disabled;
- preflight readiness: 3 `ready_for_fresh_ablation`;
- minimum trigger/control rows: 4 trigger rows, 8 control rows;
- daemon dry-run: 3 planned executable leaves;
- fixture readback: 3 accepted candidates, no graph mutation without apply;
- gated temporary apply: 3 candidate nodes and 3 `orthogonal_to` edges.

One useful negative finding from this run: a first S26 memory candidate was
correctly rejected as a specialization because its wording had too much
lexical overlap with existing nodes.  After rewriting it as a shorter
provenance/archive axis, it passed.  This supports the intended definition of
orthogonality: the candidate must be a genuinely different explanatory axis,
not only a relabeled version of the old family.

The empirical boundary remains unchanged: this validates multi-cluster
mechanism readiness, not a live downstream answer-quality win.  The emitted
commands are ready for fresh ablation once API credentials are supplied through
environment variables.
