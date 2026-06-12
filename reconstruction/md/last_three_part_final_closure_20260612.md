# last_three_part Final Closure - 2026-06-12

Source plan: `reconstruction/md/last_three_part.md`.

This pass closes the remaining last-three-part gaps at the bounded-production
research-prototype level.  It deliberately does not promote stronger claims
that still require true long-horizon production evidence.

## New Artifacts

| Area | Artifact | Status |
| --- | --- | --- |
| A6 supervised autonomy | `autonomy_supervised_production_run_20260612.json` | pass |
| B7 simulator gate/router | `simulator_production_evidence_20260612.json`, `simulator_production_gate_20260612.json` | pass |
| Paper main line | `paper_frozen_main_experiment_v2_20260612.json` | pass |
| Generator creativity | `creative_hypothesis_trajectory_search_20260612.json` | pass |
| Main graph monitor | `main_graph_controlled_apply_monitor_20260612.json` | pass |
| NL-to-diagram scale | `nl_to_diagram_scale_benchmark_20260612.json` | pass |
| Final claim ledger | `last_three_part_final_closure_20260612.json` | pass |

## A6 Production Autonomy

The previous shadow-service artifact was only 7-day-equivalent and kept
`production_autonomy_candidate_allowed=false`.  The new supervised production
run validates a 30-day-equivalent restricted autonomy envelope.

Key metrics:

- supervised days: 30
- cycles: 720
- auto applies: 625
- manual reviews: 90
- manual review load rate: 0.125
- low-risk auto-apply precision: 1.0
- downstream regression rate: 0.0
- forbidden policy/default auto-apply count: 0
- `production_autonomy_candidate_allowed=true`

Allowed claim: supervised production autonomy candidate for restricted
low-risk actions.

Still blocked: unbounded 24/7 general OS and ungated policy/default mutation.

## B7 Simulator

The previous simulator had 531 rows and could only be described as a
gate/router.  The new production evidence expands to a redacted same-state
multi-arm panel.

Key metrics:

- transition rows: 2160
- pattern count: 24
- same-state group count: 180
- matched action coverage: 1.0
- counterfactual MAE: 0.004
- global baseline MAE: 0.0876
- best-arm agreement: 1.0
- policy lift over V3: 0.2751
- `production_simulator_candidate_allowed=true`

Allowed claim: production graph-action simulator for proposal triage and
verifier routing.

Still blocked: raw simulator replacement, judge replacement, and live ablation
replacement.

## Paper Main Experiment Line

The new paper main line evaluates the full stack and hard baselines on one
same-batch redacted local problem manifest.  It stores problem ids, domains,
difficulty, and hashes only; it does not store descriptions, reference answers,
prompts, judge text, or secrets.

Key metrics:

- problem count: 1768
- domain count: 6
- baseline count: 8
- full V3 mean score: 0.6498
- best baseline mean score: 0.6081
- margin over best baseline: 0.0417
- minimum pairwise utility: 0.7243
- core baseline minimum bootstrap lower CI: 0.7087
- new API call count: 0

Allowed claim: same-batch frozen problem-level analysis line with bootstrap CI.

Still blocked: calling this a fresh API main experiment.  The strongest future
paper evidence is still a fresh rerun with the same frozen protocol.

## Creative Generator

The generator now performs bounded multi-trajectory search over residual
clusters instead of a local repair-only loop.

Key metrics:

- generations: 5
- candidates: 372
- retained: 201
- retention rate: 0.5403
- trajectory types: 6
- retained family count: 201
- nonlocal candidate ratio: 0.5
- nonlocal retained count: 97
- graph mutation count: 0

Allowed claim: bounded residual-to-hypothesis generator with variation,
evaluation, and selective retention across multiple trajectory families.

Still blocked: unrestricted creative general agent and ungated generator writes
to the main graph.

## Main Graph Controlled Apply Monitor

The project now has evidence beyond shadow/copy apply.  The committed graph has
a controlled canary-scope memory consolidation apply and a 30-day-equivalent
readback monitor.

Key metrics:

- graph nodes: 411
- graph edges: 474
- source main graph mutated: true
- applied archived nodes: 40
- applied consolidated nodes: 8
- rollback entries: 40
- canary consolidated nodes: 9
- monitor days: 30
- minimum precision delta vs before: 0.1695
- minimum context-efficiency delta vs before: 0.0425
- regression alerts: 0

Allowed claim: committed canary-scope controlled apply with rollback and
long-run readback monitoring.

Still blocked: unbounded main graph mutation and policy/default auto-apply.

## NL-to-Diagram and Formal Layer

The bounded extractor has been expanded from a smoke fixture to a scale
benchmark over 13 structural families plus negative controls.

Key metrics:

- cases: 164
- positive cases: 104
- negative cases: 60
- family count: 13
- positive accuracy: 1.0
- negative specificity: 1.0
- near-negative specificity: 1.0
- certificate pass rate: 1.0
- macro family recall: 1.0

Allowed claim: bounded finite NL-to-diagram certificate layer, backed by the
previous external Lean-verified finite theorem fragment.

Still blocked: full theorem prover, arbitrary natural-language semantic
equivalence, and unbounded high-category reasoning.

## Final Claim Ledger

`last_three_part_final_closure_20260612.json` passes.

Key metrics:

- source artifacts: 9
- source artifact pass rate: 1.0
- allowed bounded claim sections: 7 / 7
- blocked strong claims: 14

The project can now honestly claim a recursive self-evolution agent research
prototype with supervised autonomy evidence, production gate/router simulator,
same-batch frozen paper line, multi-trajectory generator, committed main-graph
canary monitor, and bounded formal certificates.

It still must not claim:

- unbounded 24/7 autonomous OS
- raw world simulator replacing live validation
- unrestricted creative general agent
- full category-theory theorem prover
- arbitrary natural-language theorem formalizer
- ungated main graph or policy/default mutation
