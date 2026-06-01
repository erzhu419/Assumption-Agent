# Recursive Assumption Agent Core

This package is the reconstruction layer for the project. It keeps the old
experiment scripts intact, but lifts their artifacts into one shared substrate:
an Assumption Graph.

## What Changed

- `schema.py` defines assumptions, edges, evidence, and trial manifests.
- `adapters.py` converts existing `strategies/`, `wisdom_library.json`,
  `v16_residuals.json`, and Exp82 typed hypotheses into graph nodes.
- `graph_memory.py` stores the graph as JSONL and performs HippoRAG-style
  spreading retrieval over assumption nodes instead of plain passage chunks.
- `record_phase2_eval.py` writes pairwise Phase2 evaluation outcomes back as
  `TrialManifest` records, evidence, confidence updates, and residual links.
- `retrieval_policy.py` adds domain-aware reranking and execution checks for
  prompt injection. The first policy targets software-engineering negative
  transfer observed in the 21-50 heldout audit.
- `selector.py` ranks assumptions with a metaproductivity-aware score inspired
  by HGM, so a method can be useful because its descendants are productive, not
  only because it won one A/B test.
- `residuals.py` implements the EmbodiSkill distinction between an assumption
  being wrong and the executor simply failing to apply a valid assumption.
- `context.py` formats activated subgraphs for future v16/v20/Exp82 prompt
  integration.
- `proposal_overlay.py` and `candidate_eval.py` let candidate nodes enter a
  temporary graph overlay for preflight and fresh ablation, without mutating
  the committed graph store.
- `candidate_acceptance.py` converts fresh proposal judgments into accept /
  reject / defer decisions and can apply only accepted candidates to the graph.
- `recursive_runner.py` turns one-shot evolution artifacts into a recursive
  assumption tree with parent hypotheses, child verification subproblems,
  argument maps, return updates, and optional manifest writeback.
- `manifest_logger.py` records LLM calls, retrievals, judge calls, tool-use,
  simulator rollouts, and daemon iterations as redacted `TrialManifest`s.
- `harness_observer.py` audits existing judge/meta/log artifacts and backfills
  bounded manifest coverage for files that were produced before direct logging.
- `world_model.py` is the cheap verifier: it predicts candidate acceptance
  probability, regression risk, verifier tier, and next action before expensive
  ablations.
- `trajectory_search.py` performs multi-path rollout over recursive frontier
  actions using world-model scores.
- `recursive_daemon.py` runs the bounded execute/read/resume loop and can apply
  only accepted candidates when explicitly requested.
- `residual_clusterer.py` clusters systematic residuals and synthesizes
  candidate method hypotheses plus heldout validation plans.

## Build The Graph

```bash
python3 -m assumption_os.build_graph --out "phase four/assumption_graph"
```

Use `--fresh` when the JSONL graph should be rebuilt from source artifacts
before replaying evaluation writebacks:

```bash
python3 -m assumption_os.build_graph --out "phase four/assumption_graph" --fresh
```

Current build ingests:

- `phase zero/kb/strategies/S*.json`
- `phase two/analysis/cache/wisdom_library.json`
- `phase four/residuals/v16_residuals.json`
- `phase six/exp82/hypotheses.jsonl`

The build writes:

- `nodes.jsonl`
- `edges.jsonl`
- `evidence.jsonl`
- `trials.jsonl`
- `build_summary.json`

## Write Back Evaluation Outcomes

After generating and judging a graph-augmented Phase2 variant, replay the
judgment cache into the graph:

```bash
python3 -m assumption_os.record_phase2_eval \
  --graph-dir "phase four/assumption_graph" \
  --sample "phase two/analysis/cache/sample_100.json" \
  --meta "phase two/analysis/cache/answers/phase2_v20_ag_filt_gpt55_meta.json" \
  --judgments \
    "phase two/analysis/cache/judgments/phase2_v20_ag_filt_gpt55_vs_phase2_v20_gpt55.json" \
    "phase two/analysis/cache/judgments/phase2_v20_gpt55_vs_phase2_v20_ag_filt_gpt55.json" \
  --intervention phase2_v20_ag_filt_gpt55 \
  --baseline phase2_v20_gpt55 \
  --eval-id phase2_v20_ag_filt_gpt55_vs_gpt55_n20 \
  --summary-out "phase four/assumption_graph/eval_phase2_v20_gpt55_n20_writeback.json"
```

The writeback intentionally skips rows without intervention meta by default;
for v20 this excludes math/science hygiene-bypass rows where graph context was
not injected.

## Phase2 v20 Integration Notes

`phase one/scripts/validation/phase2_v20_framework.py` accepts
`--assumption-graph` to inject activated graph context into Turn 1. After the
21-50 heldout audit, software engineering is gated off by default because graph
context caused negative transfer there:

```bash
python3 "phase one/scripts/validation/phase2_v20_framework.py" \
  --variant phase2_v20_ag_next \
  --assumption-graph "phase four/assumption_graph"
```

To force graph injection for all non-bypass domains:

```bash
python3 "phase one/scripts/validation/phase2_v20_framework.py" \
  --variant phase2_v20_ag_full \
  --assumption-graph "phase four/assumption_graph" \
  --assumption-graph-skip-domains ""
```

For math/science bypass rows, use `--math-science-bridge` to keep ordinary
proof/mechanism rows strict while routing research-bridge and science-decision
rows to concrete bridge/action prompts:

```bash
python3 "phase one/scripts/validation/phase2_v20_framework.py" \
  --variant phase2_v20_ms_bridge_next \
  --sample proposal_samples/math_science_21_50_sample.json \
  --math-science-bridge
```

On the 21-50 heldout math/science rows, this intent-aware bypass beat current
`phase2_v20_gpt55` 16-2 bidirectionally and beat the raw `baseline` 18-0. The
report is in
`phase four/assumption_graph/math_science_bridge_eval_gpt55_21_50.md`.

On the larger 30-row math/science slice from `sample_100`, the same bridge beat
current `phase2_v20` bidirectionally by 53 wins / 5 losses / 2 ties, and beat
the raw baseline by 55 wins / 2 losses / 3 ties. The report is in
`phase four/assumption_graph/math_science_bridge_eval_gpt55_ms100.md`.

The software-engineering reranker is available when the skip gate is disabled,
but it is not yet enabled by default. In the targeted rerun it improved
retrieval strongly but only reached neutral answer quality, so the default gate
remains conservative.

When `--assumption-graph` is supplied, v20 also enables a small domain execution
template layer from `assumption_os.domain_templates`. For software engineering,
this layer stays active even while graph context is skipped, so SE tasks get
concrete execution constraints without attributing the result to retrieved graph
nodes. On the 10 heldout SE problems from `sample_21_50`, this template-only
intervention moved the bidirectional result from the learned-graph failure
case's 20.0% decisive win rate to 62.5%. Use
`--disable-domain-execution-template` to turn this layer off.

Combining learned graph answers for non-SE domains with the SE template answers
moves the full 21-50 heldout bidirectional result from 45.7% to 59.6% decisive
win rate against `phase2_v20_gpt55`.

Against the plain raw `baseline` prompt on the same 21-50 heldout slice, the
combined policy scores 55 wins, 5 losses, and 0 ties bidirectionally, for a
91.7% decisive win rate.

## Conditioned Evaluation Gate

`assumption_os.conditioned_eval` implements the first self-evolution gate after
writeback: route each judged problem into `should_fire`, `no_fire`, or `neutral`
for a candidate node, then compute benefit only on active `should_fire` rows and
harm only on active `no_fire` rows.  The gate can output `promote`, `keep`,
`expand_retrieval`, `narrow_scope`, `revise`, or `insufficient_evidence`.

```bash
python3 -m assumption_os.conditioned_eval \
  --graph-dir "phase four/assumption_graph" \
  --sample "phase two/analysis/cache/sample_21_50.json" \
  --meta "phase two/analysis/cache/answers/phase2_v20_ag_learned_gpt55_meta.json" \
  --judgments \
    "phase two/analysis/cache/judgments/phase2_v20_ag_learned_gpt55_vs_phase2_v20_gpt55.json" \
    "phase two/analysis/cache/judgments/phase2_v20_gpt55_vs_phase2_v20_ag_learned_gpt55.json" \
  --intervention phase2_v20_ag_learned_gpt55 \
  --baseline phase2_v20_gpt55 \
  --summary-out "phase four/assumption_graph/conditioned_eval_phase2_v20_gpt55_21_50.json"
```

`assumption_os.lifecycle` turns conditioned decisions into auditable lifecycle
actions and pending manifests.  It deliberately plans edits instead of applying
them directly.

```bash
python3 -m assumption_os.lifecycle \
  --conditioned-summary "phase four/assumption_graph/conditioned_eval_phase2_v20_gpt55_21_50.json" \
  --eval-id phase2_v20_ag_learned_gpt55_vs_gpt55_21_50_conditioned \
  --summary-out "phase four/assumption_graph/lifecycle_plan_phase2_v20_gpt55_21_50.json"
```

For a single conservative entry point, `assumption_os.evolution_cycle` now runs
the whole planning loop in dry-run mode: writeback preview, conditioned gate,
formal mapping audit, lifecycle actions, failure-derived hypotheses, candidate
proposals, candidate preflight, regression prediction, sequential
falsification, cheap world-model rollout, Bayesian policy scoring, and policy
update plan. It does not mutate the graph unless `--writeback`,
`--apply-accepted`, or the gated `--autonomous-apply` mode is explicitly
supplied.

```bash
python3 -m assumption_os.evolution_cycle \
  --graph-dir "phase four/assumption_graph" \
  --sample "phase two/analysis/cache/sample_21_50.json" \
  --meta "phase two/analysis/cache/answers/phase2_v20_ag_learned_gpt55_meta.json" \
  --judgments \
    "phase two/analysis/cache/judgments/phase2_v20_ag_learned_gpt55_vs_phase2_v20_gpt55.json" \
    "phase two/analysis/cache/judgments/phase2_v20_gpt55_vs_phase2_v20_ag_learned_gpt55.json" \
  --intervention phase2_v20_ag_learned_gpt55 \
  --baseline phase2_v20_gpt55 \
  --eval-id phase2_v20_ag_learned_gpt55_vs_gpt55_21_50_cycle \
  --policy-rerank \
  --assumption-graph-skip-domains software_engineering \
  --summary-out "phase four/assumption_graph/evolution_cycle_dryrun_phase2_v20_gpt55_21_50.json"
```

For unattended graph edits, use `--autonomous-apply` with candidate judgments.
This automatically enables writeback and applies only candidates that pass the
acceptance gate and are not blocked by the formal-mapping gate. If no candidate
acceptance payload is supplied, it only performs the evaluation writeback.

The current dry run processed 22 writeback-preview rows and scanned skipped
rows for failures (`missing_meta=18`, `policy_skipped=20`). It produced 12
conditioned summaries, planned 8 lifecycle actions, generated 8 lifecycle
proposals plus 14 failure-derived hypothesis proposals, and identified 13
candidates ready for fresh ablation. Its sequential falsification gate produced
`manifest_only=4`, `ready_for_ablation=13`, and `blocked_underpowered=5`. The
Bayesian scorer ranked those 13 ready candidates as `run_ablation`; five
underpowered candidates were ranked as `collect_evidence`. The same dry run
also audits 45 typed formal nodes into 9 complete formal mappings and applies
22 proposal-level formal gates with 0 blocked. The report is in
`phase four/assumption_graph/evolution_cycle_dryrun_phase2_v20_gpt55_21_50.md`.

Failure-hypothesis generation defaults to materializing every grouped loss. Use
`--failure-hypothesis-top-n` only as an explicit proposal-budget cap, or `0` to
disable this source.

`assumption_os.recursive_runner` is the first recursive outer loop. It consumes
the evolution-cycle payload, builds a task stack, and expands each high-priority
candidate into an argument map plus a child verification/evidence/repair
subproblem. Open child frames define exactly what observation is needed before
their parent hypothesis can be updated. When a candidate-acceptance payload is
available, the same runner resumes the tree by propagating each child
verification result back to the parent candidate: accepted candidates become
eligible for gated apply, harmful candidates become scope-repair/reject actions,
weak candidates become revise/reject actions, and underpowered candidates
become more-judgment actions.

```bash
python3 -m assumption_os.recursive_runner \
  --graph-dir "phase four/assumption_graph" \
  --problem "Phase2 v20 graph self-evolution has judged losses and skipped math/science/software rows; decide which assumptions should be tested, repaired, or written back next." \
  --goal "Build a recursive assumption tree where each candidate hypothesis has explicit support, objections, falsification tests, and child subproblems that return updates to the parent." \
  --problem-id phase2_v20_gpt55_21_50_recursive \
  --eval-id recursive_runner_phase2_v20_gpt55_21_50 \
  --evolution-payload "phase four/assumption_graph/evolution_cycle_dryrun_phase2_v20_gpt55_21_50.json" \
  --max-children 8 \
  --max-depth 3 \
  --summary-out "phase four/assumption_graph/recursive_runner_phase2_v20_gpt55_21_50.json"
```

After fresh proposal judgments exist for the same proposal IDs, add
`--acceptance-payload path/to/candidate_acceptance_summary.json` to resume the
tree from child verification results instead of leaving those children as
`run_fresh_ablation` leaves.

`assumption_os.recursive_executor` is the executor/resume layer for that
frontier. In dry-run mode it records which recursive leaf commands are ready to
run. With candidate judgments, it builds the acceptance payload and resumes the
recursive tree so returned verification children update their parent candidate
frames.

```bash
python3 -m assumption_os.recursive_executor \
  --recursive-payload "phase four/assumption_graph/recursive_runner_phase2_v20_gpt55_21_50.json" \
  --evolution-payload "phase four/assumption_graph/evolution_cycle_dryrun_phase2_v20_gpt55_21_50.json" \
  --eval-id recursive_executor_phase2_v20_gpt55_21_50 \
  --summary-out "phase four/assumption_graph/recursive_executor_phase2_v20_gpt55_21_50.json"
```

The executor does not spend model calls by default. Add `--execute` only when
the frontier commands should actually run. Add `--candidate-judgments`,
`--candidate-variant`, `--candidate-baseline`, and `--proposal-ids` for a single
acceptance resume, or `--judgment-bundle` for multiple per-proposal judgment
runs.

`assumption_os.recursive_daemon` wraps the executor in a bounded loop. It can
dry-run or execute frontier `command_hint`s, ingest judgment sets, resume the
recursive tree, record daemon/tool-use manifests, and apply only candidates
accepted by the gate when `--apply-accepted` is explicitly supplied.

```bash
python3 -m assumption_os.recursive_daemon \
  --graph-dir "phase four/assumption_graph" \
  --recursive-payload "phase four/assumption_graph/recursive_runner_phase2_v20_gpt55_21_50.json" \
  --evolution-payload "phase four/assumption_graph/evolution_cycle_dryrun_phase2_v20_gpt55_21_50.json" \
  --eval-id recursive_daemon_phase2_v20_gpt55_21_50 \
  --max-iterations 1 \
  --writeback-manifests \
  --summary-out "phase four/assumption_graph/recursive_daemon_phase2_v20_gpt55_21_50.json"
```

`assumption_os.world_model` is the cheap verifier/simulator. It consumes the
proposal, preflight, falsification, acceptance, regression, and formal-gate
payloads, then predicts acceptance probability, regression risk, verifier tier,
and a recommended next action. It also emits simulator `TrialManifest`s; use
real ablation/judge evidence to override it before promotion.

```bash
python3 -m assumption_os.world_model \
  --graph-dir "phase four/assumption_graph" \
  --proposals "phase four/assumption_graph/evolution_cycle_proposals.json" \
  --preflight "phase four/assumption_graph/candidate_preflight_phase2_v20_gpt55_21_50.json" \
  --acceptance "phase four/assumption_graph/recursive_positive_ms_bridge_acceptance.json" \
  --train-calibration-out "phase four/assumption_graph/world_model_calibration_phase2_v20.json" \
  --raw-prediction-out "phase four/assumption_graph/world_model_raw_phase2_v20.json" \
  --eval-id world_model_phase2_v20_gpt55_21_50 \
  --summary-out "phase four/assumption_graph/world_model_phase2_v20_gpt55_21_50.json"
```

The calibration artifact is reusable: pass it back through
`--calibration` in `world_model`, or through `--world-model-calibration` in
`evolution_cycle`. When candidate acceptance evidence is available, the cycle
can also train and persist it inline with `--train-world-model-calibration` and
`--world-model-calibration-out`. The expanded reconstruction validation stores
its reusable 16-label model at
`phase four/assumption_graph/world_model_calibration_reconstruction_gap_20260601_expanded.json`.

`assumption_os.trajectory_search` uses those predictions to keep multiple
hypothesis futures alive: promote-after-verification, repair-then-retest,
evidence-first, or reject-and-synthesize. This is the multi-path search layer
for the recursive runner's frontier.

```bash
python3 -m assumption_os.trajectory_search \
  --recursive-payload "phase four/assumption_graph/recursive_runner_phase2_v20_gpt55_21_50.json" \
  --world-model-payload "phase four/assumption_graph/world_model_phase2_v20_gpt55_21_50.json" \
  --eval-id trajectory_search_phase2_v20_gpt55_21_50 \
  --beam-width 5 \
  --summary-out "phase four/assumption_graph/trajectory_search_phase2_v20_gpt55_21_50.json"
```

The first recursive dry run produced `root_problem=1`,
`candidate_hypothesis=8`, and `verification_subproblem=8`, with 16 recursion
edges and 8 leaf `run_fresh_ablation` next actions. By default this is a dry
run; `--writeback` appends each recursive frame as a `TrialManifest` without
applying candidates.

`assumption_os.formal_mapping` is the executable GRAM/category-style bridge for
typed formal forms. It groups Exp82-style `feature`, `constraint`,
`decomposition`, `verification`, and `hp_change` nodes by source seed, then
checks whether each group preserves the operational invariants needed for a
safe morphism:

- `trigger_detector`
- `constraint_operator`
- `decomposition_operator`
- `verification_operator`
- `runtime_policy`

```bash
python3 -m assumption_os.formal_mapping \
  --graph-dir "phase four/assumption_graph" \
  --summary-out "phase four/assumption_graph/formal_mapping_audit_phase2_graph.json"
```

The first audit found `complete=9`, `partial=0`, and `unsafe=0`. This is still a
bounded audit layer: it checks that generated formal bundles are executable and
constraint-preserving, and `assumption_os.evolution_cycle` now feeds the result
into a proposal-level formal mapping gate. `partial` or `unsafe` mappings block
promotion-sensitive policy actions until the formal bundle is repaired. The
latest cycle has `not_applicable=10` proposal gates and `blocked=0`, meaning the
current proposal set does not target typed formal bundles. The gate does not yet
synthesize new mappings by itself.

The same module now also supports solver-time formal search. `search_formal_mappings`
matches problem text against complete mapping triggers, then returns executable
constraints, ordered steps, verifier instructions, and runtime hints. Phase2
graph retrieval injects these hits through `format_policy_context` as a `Formal
Mapping Reasoning` section. The first five-query search audit passed 5 / 5; see
`phase four/assumption_graph/formal_mapping_search_eval_phase2_graph.md`.

Add `--formal-metrics` to also produce a finite categorical /
information-geometry payload. The metric layer represents each formal bundle as
a stochastic kernel over `feature`, `constraint`, `decomposition`,
`verification`, and `hp_change`, then reports row KL divergence, total
variation, Frobenius distance, and a Blackwell-style dominance proxy against
the reference verifier pipeline.

```bash
python3 -m assumption_os.formal_mapping \
  --graph-dir "phase four/assumption_graph" \
  --formal-metrics \
  --summary-out "phase four/assumption_graph/formal_mapping_metrics_phase2_graph.json"
```

`assumption_os.manifest_logger` is the generic log bridge for events outside
graph mutation: LLM calls, retrievals, judge calls, tool-use, and simulator
rollouts. It redacts secret-looking values before writing manifests.

```bash
python3 -m assumption_os.manifest_logger \
  --graph-dir "phase four/assumption_graph" \
  --events "phase four/assumption_graph/component_events.jsonl" \
  --eval-id component_events_phase2_v20 \
  --writeback
```

`assumption_os.harness_observer` is the artifact coverage bridge. It converts
existing judgment JSON, answer meta JSON, and run logs into bounded
`TrialManifest` events, then reports which artifact files are covered after
writeback.

```bash
python3 -m assumption_os.harness_observer \
  --graph-dir "phase four/assumption_graph" \
  --artifacts "phase two/analysis/cache/judgments/phase2_v20_gpt55_vs_phase2_v20_ms_bridge_gpt55_21_50.json" \
              "phase two/analysis/cache/answers/phase2_v20_ms_bridge_gpt55_21_50_meta.json" \
              "phase four/assumption_graph/recursive_scoped_judge_run_gpt55_21_50.log" \
              "phase six/autonomous/exp80_run.log" \
  --eval-id harness_observer_backfill_20260601 \
  --writeback \
  --summary-out "phase four/assumption_graph/harness_observer_backfill_20260601.json"
```

`assumption_os.residual_clusterer` closes the systematic-residual synthesis
loop. It clusters residual manifests by residual type and signature, then emits
candidate method hypotheses with heldout trigger/control validation plans. A
callable LLM synthesizer can be injected by code; the CLI stays deterministic
and environment-key-free.

```bash
python3 -m assumption_os.residual_clusterer \
  --graph-dir "phase four/assumption_graph" \
  --eval-id residual_clusters_phase2_v20 \
  --min-cluster-size 2 \
  --summary-out "phase four/assumption_graph/residual_clusters_phase2_v20.json"
```

`assumption_os.performance_validation` runs the non-smoke validation suite for
the reconstruction-gap mechanisms. It uses real candidate acceptance
payloads and positive controls where available, then reports ranking quality,
calibration, multi-path coverage, daemon gated-apply behavior, manifest
throughput/redaction, harness artifact coverage, residual synthesis coverage,
and formal metric coverage.

```bash
python3 -m assumption_os.performance_validation \
  --graph-dir "phase four/assumption_graph" \
  --eval-id reconstruction_gap_perf_20260601_expanded \
  --summary-out "phase four/assumption_graph/reconstruction_gap_perf_20260601_expanded.json" \
  --report-out "phase four/assumption_graph/reconstruction_gap_perf_20260601_expanded.md"
```

The expanded performance validation passes all seven sections. The initial run found
one real issue: post-acceptance world-model probabilities stayed too high after
rejected evidence, with Brier score 0.2767. The calibrated version now scores
Brier 0.0081 on the expanded 2 accepted / 14 rejected labeled set, while
preserving pre-acceptance ranking (`AUC=1.0`). The calibration payload also
reports leave-one-out Brier 0.0064 versus raw 0.5316. Manifest validation now
parses 12 real events from existing run/judge logs in addition to synthetic
redaction probes. Those 12 real events are also persisted through
`real_log_manifest_ingest_20260601` as observed trials in the graph. Harness
observer discovers 19 real artifact events from judgment/meta/log files and
backfilled the 10 previously uncovered judgment/meta events, leaving full
artifact-file coverage after writeback. Full report:
`phase four/assumption_graph/reconstruction_gap_perf_20260601_expanded.md`.

`assumption_os.failure_hypotheses` converts loss rows into candidate assumptions
and manifests. It now uses two sources: attributed graph losses from
`writeback_summary.processed_trials`, and raw judgment losses for rows that
writeback skipped because meta was missing or the current policy bypassed the
domain. Skipped-source candidates carry `source_skipped_reason` in the node
payload and manifest.

In the current dry run it generated 14 failure hypotheses: two processed graph
losses plus 12 net skipped losses from math/science missing-meta bypasses and
software-engineering policy skips. Failure-hypothesis generation defaults to
materializing every grouped loss; use `--failure-hypothesis-top-n` as an
explicit proposal-budget cap, or `0` to disable this source.

`assumption_os.proposals` then turns lifecycle actions into candidate nodes and
experiment manifests. Retrieval-policy candidates copy the parent's trigger
surface into a candidate retrieval node; revision candidates are narrower child
assumptions with concrete acceptance criteria. It is a proposal queue, not an
automatic mutation step; use `--apply` only after reviewing and validating the
generated candidates.

```bash
python3 -m assumption_os.proposals \
  --graph-dir "phase four/assumption_graph" \
  --lifecycle-plan "phase four/assumption_graph/lifecycle_plan_phase2_v20_gpt55_21_50.json" \
  --eval-id phase2_v20_ag_learned_gpt55_vs_gpt55_21_50_proposals \
  --summary-out "phase four/assumption_graph/proposals_phase2_v20_gpt55_21_50.json"
```

`assumption_os.candidate_eval` preflights proposal candidates before spending
fresh solver/judge calls. It overlays the candidate in memory, routes each
problem into trigger/control subsets, and can model the actual ablation mode
where the proposal target is forced only on its own `should_fire` rows.
For failure hypotheses generated from skipped rows, preflight probes the source
problem even when the normal run would skip missing-meta math/science rows or
the software-engineering domain.

```bash
python3 -m assumption_os.candidate_eval \
  --graph-dir "phase four/assumption_graph" \
  --proposals "phase four/assumption_graph/proposals_phase2_v20_gpt55_21_50.json" \
  --sample "phase two/analysis/cache/sample_21_50.json" \
  --meta "phase two/analysis/cache/answers/phase2_v20_ag_learned_gpt55_meta.json" \
  --eval-id phase2_v20_ag_learned_gpt55_vs_gpt55_21_50_candidate_preflight \
  --force-proposal-route \
  --summary-out "phase four/assumption_graph/candidate_preflight_phase2_v20_gpt55_21_50.json"
```

For the v20 solver, candidate experiments can be run with a temporary overlay:

```bash
python3 "phase one/scripts/validation/phase2_v20_framework.py" \
  --variant proposal_3a5cf90b1010 \
  --sample sample_21_50.json \
  --assumption-graph "phase four/assumption_graph" \
  --assumption-graph-skip-domains "" \
  --assumption-proposals "phase four/assumption_graph/proposals_phase2_v20_gpt55_21_50.json" \
  --assumption-proposal-ids prop_3a5cf90b1010 \
  --assumption-force-proposal-route
```

`--assumption-force-proposal-route` is deliberately route-conditioned: retrieval
policy proposals force the parent node only on the parent's trigger subset;
revision/scope proposals force the candidate child only on the child's trigger
subset. Neutral and no-fire rows remain unforced controls.

Proposal forcing can now be narrowed further by route metadata:

```bash
python3 "phase one/scripts/validation/phase2_v20_framework.py" \
  --variant proposal_ready_combo_se_hard_next \
  --sample proposal_samples/proposal_se_mh_sample.json \
  --assumption-graph "phase four/assumption_graph" \
  --assumption-graph-skip-domains "" \
  --assumption-proposals "phase four/assumption_graph/proposals_phase2_v20_gpt55_21_50.json" \
  --assumption-proposal-ids prop_3a5cf90b1010 prop_e9c8ee2fa09b prop_b7fd42179967 prop_ad2c1f2b1cad prop_16571ee152bc prop_66a126a35878 \
  --assumption-force-proposal-route \
  --assumption-force-proposal-domains software_engineering \
  --assumption-force-proposal-difficulties hard
```

`--assumption-route-scope-proposals` is also available for stricter candidate
isolation: it removes proposal candidate nodes outside their own routed
`should_fire` subset. The first screening found that this was safer outside the
target route but too suppressive for software-engineering gains, so it remains
an experimental guard rather than the default policy.

In the first mini-model screening pass, the six ready proposals were also tested
as a combo route-conditioned policy (`proposal_ready_combo_gpt54mini`) against a
same-model v20 baseline on the union proposal sample. The combo won 26-12
bidirectionally, mostly on software-engineering rows. Individual acceptance
remained conservative and did not apply any candidate to the graph.

The follow-up routed policy evaluation is recorded in
`phase four/assumption_graph/route_policy_ablation_gpt54mini_gpt55.md`. Mini
screening preferred `software_engineering` + `medium|hard`, but GPT-5.5
confirmation showed the medium rows were not stable. The current recommended
gate is therefore `software_engineering` + `hard` + proposal `should_fire`,
which scored 6 wins, 2 losses, and 10 fallback ties against `phase2_v20_gpt55`
on the nine SE confirmation rows.

After fresh judgments are available, `assumption_os.candidate_acceptance`
separates trigger benefit from control harm. By default it only writes a JSON
decision report; `--apply-accepted` is required to mutate the graph.

```bash
python3 -m assumption_os.candidate_acceptance \
  --graph-dir "phase four/assumption_graph" \
  --proposals "phase four/assumption_graph/proposals_phase2_v20_gpt55_21_50.json" \
  --preflight "phase four/assumption_graph/candidate_preflight_phase2_v20_gpt55_21_50.json" \
  --judgments "phase two/analysis/cache/judgments/proposal_3a5cf90b1010_vs_phase2_v20_gpt55.json" \
  --candidate-variant proposal_3a5cf90b1010 \
  --baseline-variant phase2_v20_gpt55 \
  --proposal-ids prop_3a5cf90b1010 \
  --eval-id proposal_3a5cf90b1010_acceptance \
  --summary-out "phase four/assumption_graph/acceptance_proposal_3a5cf90b1010.json"
```

## Why This Shape

The reconstruction documents and the reference papers point to the same failure
mode: wisdom-as-prompt text is too weak. The system needs explicit lifecycle
objects:

1. record the assumption behind every key action,
2. retrieve related assumptions, cases, residuals, and verifiers as a graph,
3. test assumptions with manifest-style predictions,
4. classify failures before editing the library,
5. preserve assumptions that failed only because of execution lapse,
6. prefer assumption families with long-run productive descendants.

This package is the first code-level boundary for that lifecycle.
