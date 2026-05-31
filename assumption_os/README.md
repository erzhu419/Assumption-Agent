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
falsification, Bayesian policy scoring, and policy update plan. It does not mutate the graph unless `--writeback` or
`--apply-accepted` is explicitly supplied.

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
  --failure-hypothesis-top-n 8 \
  --summary-out "phase four/assumption_graph/evolution_cycle_dryrun_phase2_v20_gpt55_21_50.json"
```

The first dry run processed 22 writeback-preview rows, produced 12 conditioned
summaries, planned 8 lifecycle actions, generated 8 lifecycle proposals plus 2
failure-derived hypothesis proposals, and identified 3 candidates ready for
fresh ablation. Its sequential falsification gate produced `manifest_only=4`,
`ready_for_ablation=3`, and `blocked_underpowered=3`. The Bayesian scorer ranked
those 3 ready candidates as `run_ablation`; three underpowered candidates were
ranked as `collect_evidence`. The same dry run also audits 45 typed formal nodes
into 9 complete formal mappings and applies 10 proposal-level formal gates with
0 blocked. The report is in
`phase four/assumption_graph/evolution_cycle_dryrun_phase2_v20_gpt55_21_50.md`.

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

`assumption_os.failure_hypotheses` converts attributed loss rows from
`writeback_summary.processed_trials` into candidate assumptions and manifests.
In the current dry run it generated two candidates:
`daily_life_0161` under `strategy_S03` for a memory-defect residual, and
`business_0199` under `strategy_S12` for an optimization residual.

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
