# Evolution Cycle Dry Run: phase2 v20 gpt55 21-50

Date: 2026-05-31

## Command

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
  --conditioned-top-n 12 \
  --proposal-top-n 12 \
  --proposal-artifact-out "phase four/assumption_graph/evolution_cycle_dryrun_phase2_v20_gpt55_21_50_proposals.json" \
  --summary-out "phase four/assumption_graph/evolution_cycle_dryrun_phase2_v20_gpt55_21_50.json"
```

No graph mutation was performed. `writeback=false` and `dry_run=true`.

## Pipeline Result

- Writeback preview processed rows: 22
- Conditioned decisions: `promote=1`, `expand_retrieval=7`, `insufficient_evidence=4`
- Lifecycle actions: `promote_assumption=1`, `expand_retrieval=4`, `keep_collect_evidence=3`
- Candidate proposals: `promotion_record=1`, `retrieval_policy=4`, `evidence_request=3`
- Candidate preflight: `manifest_only=4`, `ready_for_fresh_ablation=1`, `needs_more_trigger_rows=3`
- Sequential falsification gate: `manifest_only=4`, `ready_for_ablation=1`, `blocked_underpowered=3`
- Bayesian policy scorer: `run_ablation=1`, `collect_evidence=3`, `record_only=4`
- Formal mapping audit: `complete=9`, `partial=0`, `unsafe=0`
- Formal mapping proposal gate: `not_applicable=8`, `blocked=0`

## Policy Plan

The dry run generated three policy-action classes:

- `run_fresh_ablation_before_promotion`: one retrieval-policy candidate is ready for a fresh ablation.
- `collect_more_evidence`: three candidates need more trigger rows before ablation.
- `record_manifest_only_no_graph_policy_change`: promotion/evidence-request manifests do not mutate retrieval policy.

The ready candidate is a retrieval policy for `wisdom_W020`:

- proposal: `prop_2ec0255facee`
- candidate: `cand_55693b29e986`
- routed trigger rows: `daily_life_0161`, `daily_life_0173`, `daily_life_0018`
- control rows: 8 non-trigger rows
- regression prediction: low preflight risk; fresh ablation still required
- Bayesian priority: `1.6421`
- Bayesian expected value: `0.73`
- Bayesian information value: `0.3873`

## Interpretation

This closes the orchestration gap for the self-evolution loop. The system now
has a single dry-run entry point that turns judged failures and wins into
conditioned gates, lifecycle actions, candidate hypotheses, falsification
preflight, regression predictions, formal mapping diagnostics, and a graph
policy update plan. Promotion or graph mutation remains gated behind explicit
acceptance evidence and opt-in write flags.

The Bayesian scorer is only a budget-allocation layer. It ranks which gated
candidate is worth testing next; it does not override the sequential
falsification or acceptance gates.

The formal mapping audit is a bounded safety check for typed Exp82 bundles. It
does not synthesize new mappings yet, but the cycle now turns incomplete or
unsafe formal bundles into proposal-level policy blocks before promotion or
accepted-candidate application.
