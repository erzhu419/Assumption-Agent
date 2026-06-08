# Orthogonal Recursive Ablation 2026-06-08

## Question

After the scoped execution-contract hypothesis passed same-model live
acceptance, the next question was whether the orthogonal gate actually changes
recursive self-evolution, or whether it is only a label.

This experiment holds downstream answer evidence constant and toggles only the
novelty/integration gate:

- orthogonal gate ON;
- orthogonal gate OFF.

The same proposal, same preflight rows, same live judgments, same acceptance
thresholds, and same recursive runner are used in both arms.

## Evidence Used

Queue artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_execution_scope_repair_20260608.json`

Live artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_execution_scope_repair_live_same_model_20260608.json`

Judgment artifact:

`phase two/analysis/cache/judgments/proposal_9975d9e28cd7_vs_phase2_v20_claude_opus_execution_baseline_orthogonal_execution_scope_repair_live_same_model_20260608_prop_9975d9e28cd7.json`

The live acceptance result is:

- trigger wins: 3;
- trigger ties: 2;
- trigger losses: 0;
- trigger utility: 0.80;
- trigger LCB90: 0.5710;
- control losses: 0;
- control loss UCB90: 0.0;
- acceptance decision: `accept`.

## ON/OFF Result

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/orthogonal_recursive_ablation_20260608.json`

Orthogonal ON:

- novelty classification: `orthogonal_new_family`;
- recursive frame exposes `apply_accepted_candidate_if_requested`;
- daemon rebuilds acceptance from the live judgment set;
- temp apply writes candidate node;
- temp apply writes `generated_from_residual`;
- temp apply writes `orthogonal_to`;
- retention score: 2.0.

Orthogonal OFF:

- novelty classification: `specialization`;
- recursive frame still exposes `apply_accepted_candidate_if_requested`;
- daemon rebuilds the same acceptance from the same live judgment set;
- temp apply writes candidate node;
- temp apply writes `generated_from_residual`;
- temp apply writes `specializes`;
- no `orthogonal_to` edge;
- retention score: -0.25.

Comparison:

- downstream utility delta: 0.0;
- orthogonal edge delta: +1;
- specializes edge delta: -1;
- recursive retention delta: +2.25;
- main graph mutation delta: 0;
- pass: true.

## Interpretation

The orthogonal gate does not make the already accepted answer better in this
specific comparison.  That is intentional: the same live judgments are reused in
both arms.

The observed gain is recursive retention quality.  With the gate ON, the
accepted execution-contract hypothesis starts an independent execution-harness
family through `orthogonal_to`.  With the gate OFF, the same accepted evidence is
folded back into `strategy_S01` as a specialization.  That means future
descendants would be searched, evaluated, and attributed under the old method
family rather than under a newly discovered orthogonal axis.

This is the first clean ON/OFF evidence that orthogonal novelty changes the
recursive hypothesis graph after a live-positive candidate, without relying on
main-graph mutation or answer-leakage.

## Remaining Gap

This still does not prove that orthogonal retention improves long-horizon
downstream QA/decision utility over 3-5 generations.  The next stronger
benchmark should let descendants actually branch under the ON/OFF retained
graphs, then compare accepted descendant productivity across generations.
