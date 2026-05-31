# Conditioned Lifecycle Pass: Phase2 v20 GPT-5.5 21-50

Date: 2026-05-31

## Inputs

- Graph: `phase four/assumption_graph`
- Sample: `phase two/analysis/cache/sample_21_50.json`
- Meta: `phase two/analysis/cache/answers/phase2_v20_ag_learned_gpt55_meta.json`
- Judgments:
  - `phase2_v20_ag_learned_gpt55_vs_phase2_v20_gpt55.json`
  - `phase2_v20_gpt55_vs_phase2_v20_ag_learned_gpt55.json`

This pass uses the learned-graph heldout result, not the later SE-template
synthetic result, because the purpose is to audit graph-node evidence.

## Conditioned Gate

Artifact:

- `phase four/assumption_graph/conditioned_eval_phase2_v20_gpt55_21_50.json`

Rows processed: 42 non-bypass judged rows.

Decision counts:

- `expand_retrieval`: 4
- `revise`: 5
- `insufficient_evidence`: 11
- `promote`: 0

Important implementation note: strategy nodes no longer use broad lexical
fallback routing. `strategy_Sxx` routes to `should_fire` only through explicit
activation metadata or coverage/gold tags; otherwise it remains neutral. This
prevents the conditioned gate from degenerating into pooled evaluation.

Wisdom nodes now also use activation profiles built from `signal`,
`unpacked_for_llm`, examples, domains, and keywords instead of broad lexical
fallback. This tightened `wisdom_W020` from an over-broad route to 8 routed
should-fire rows across the bidirectional judged rows.

## Lifecycle Plan

Artifact:

- `phase four/assumption_graph/lifecycle_plan_phase2_v20_gpt55_21_50.json`

Action counts:

- `expand_retrieval`: 2
- `revise_assumption`: 5
- `keep_collect_evidence`: 2

Top actions:

- Expand retrieval for `wisdom_W020`.
- Revise `strategy_S26`, `strategy_S27`, `strategy_S01`, `strategy_S14`, `strategy_S21`.
- Expand retrieval for `strategy_S08`.
- Collect more active evidence for `strategy_S12` and `strategy_S11` before editing retrieval.

No graph mutation was applied from this plan.

## Proposal Queue

Artifact:

- `phase four/assumption_graph/proposals_phase2_v20_gpt55_21_50.json`

Proposal counts:

- `retrieval_policy`: 2
- `assumption_revision`: 5
- `evidence_request`: 2

The proposal queue creates candidate nodes and pending manifests in JSON only.
It was generated without `--apply`, so `nodes.jsonl`, `edges.jsonl`, and
`trials.jsonl` were not mutated by this pass.

The five revision candidates are semantic children, not copies of the failed
parent:

- `strategy_S26`: path dependency only when lock-in mechanisms, switching
  costs, accumulated assets/skills, staged migration evidence, and no-go
  thresholds are explicit.
- `strategy_S27`: incentive analysis only when stakeholders, measurable
  incentives, constraints, decision rights, and interventions are mapped.
- `strategy_S01`: controlled-variable reasoning only when baseline,
  one-factor intervention, controlled environment/data, and causal confirmation
  criterion are explicit.
- `strategy_S14`: boundary-condition analysis only when edge cases become
  concrete tests, thresholds, monitored failure modes, and fallback actions.
- `strategy_S21`: dead-end recognition only when failure thresholds,
  opportunity cost, sunk-cost separation, and a higher-level alternative are
  explicit.

## Candidate Preflight

Artifact:

- `phase four/assumption_graph/candidate_preflight_phase2_v20_gpt55_21_50.json`

This pass overlays proposal candidates in memory and models the next ablation
mode: force the proposal target only on its own routed `should_fire` subset.

Readiness counts:

- `ready_for_fresh_ablation`: 6
- `needs_more_trigger_rows`: 1
- `manifest_only`: 2

Ready for fresh ablation:

- `wisdom_W020` retrieval-policy parent injection: 4 trigger rows.
- `strategy_S26` revision child: 3 trigger rows.
- `strategy_S27` revision child: 3 trigger rows.
- `strategy_S01` revision child: 6 trigger rows.
- `strategy_S14` revision child: 3 trigger rows.
- `strategy_S21` revision child: 6 trigger rows.

Not ready:

- `strategy_S08` retrieval policy has only 2 trigger rows in this sample, so
  it needs more evidence before a meaningful quality test.
- `strategy_S12` and `strategy_S11` are evidence-request manifests, not
  candidate nodes.

## Acceptance Gate

Fresh mini-model screening has now been run:

- Per-candidate proposal variants were generated with `gpt-5.4-mini` on
  proposal-specific trigger/control samples.
- Strict individual acceptance rejected all six tested candidates:
  five failed trigger-row benefit LCB90, and `strategy_S21` passed trigger
  benefit but failed the conservative control-harm UCB gate.
- No candidate was applied to the persistent graph.

The route-conditioned combo policy was also tested:

- Artifact: `phase four/assumption_graph/candidate_ablation_summary_gpt54mini_21_50.json`
- Combo variant: `proposal_ready_combo_gpt54mini`
- Baseline: `phase2_v20_gpt54mini_prop_union`
- Bidirectional result: 26 combo wins, 12 baseline wins, 0 ties.
- Decisive win rate: 68.4%.
- Main gain: software engineering, 16 combo wins vs 2 baseline wins.
- Main regression area: easy/daily-life rows.

Acceptance gate behavior:

- `assumption_os.candidate_acceptance` reads proposal JSON, preflight JSON, and
  one or more pairwise judgment files.
- It scores trigger rows separately from controls.
- Acceptance requires enough judged trigger rows, trigger-row benefit LCB90 to
  clear threshold, and control-row loss UCB90 to stay below the harm threshold.
- It writes a decision report by default; `--apply-accepted` is required before
  accepted candidate nodes, edges, and accepted trial manifests are committed
  into `nodes.jsonl`, `edges.jsonl`, and `trials.jsonl`.

## Interpretation

The graph is not yet ready for automatic promotion on this heldout slice. The
right next step is candidate testing:

1. keep the combo policy as an experimental route-conditioned policy, not a
   graph writeback,
2. tighten the combo policy to software-engineering / medium-hard routes before
   broader use,
3. hold `strategy_S08` until a larger trigger subset exists,
4. collect additional active examples for `strategy_S12` and `strategy_S11`,
5. only apply candidates to the graph after a fresh conditioned gate beats the
   parent and does not increase no-fire harm.
