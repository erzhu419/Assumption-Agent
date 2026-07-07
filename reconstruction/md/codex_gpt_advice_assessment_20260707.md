# Codex Assessment of GPT Advice 2026-07-07

## Reference Intake

Downloaded local reference bundle:

```text
reference/self_evo_continual_20260707/
  papers/      arXiv PDFs for refs 2-4 and 6-23
  pages/       arXiv abs pages plus Wikipedia reader snapshots
  repos/       shallow clones for 21 related repos
  metadata/    references and repo manifests
```

Notes:

- FST / `Learning, Fast and Slow` has a public code URL, but it currently returns "Code coming soon"; no FST repo was cloned.
- FLEX exposes a project page; the cloned repo is the project-page repo, not confirmed full training code.
- Self-Rewarding repos are searched implementations rather than an official paper repo.

## Agreement

I agree with the main GPT advice.

The current HLE branch has real engineering improvements, but it is still too much of a patch chain:

```text
source/operator tweak
  -> tiny HLE probe
  -> new failure bucket
  -> another source/operator tweak
```

That is not yet continual learning. The missing abstraction is a fast layer that turns failures into reusable, promoted-or-shadowed policies while a slow baseline stays available as fallback.

The project already contains many pieces:

- OperatorSpec and operator policy.
- HLE option/source/solver lanes.
- Conservative promotion ideas in the framework evolution modules.
- Regression/debug cohort discipline.

The gap is that HLE-specific experience is not yet represented as first-class fast-weight policy memory with trigger, anti-trigger, utility, harm, evidence, failure rows, and promotion status.

## Implemented First Step

Added `assumption_os.fast_policy_memory`:

```text
FastPolicyHypothesis
  kind: operator | source_binding | solver_lane | fallback_policy
  trigger_terms
  anti_trigger_terms
  expected_utility
  expected_harm
  evidence_rows
  failure_rows
  promotion_status
  fallback_behavior

select_fast_policies(...)
  deterministic trigger/anti-trigger scoring
  promoted policies only by default
  raw text is hashed, not persisted

evaluate_fast_policy_promotion(...)
  fixed-regression non-regression
  unseen gain or stable non-inferiority
  selected-label stability
  no_fallback_count = 0
  cost cap unless clear accuracy gain
  tracked failure-bucket non-worsening
```

Also connected the HLE option lane router to carry a fast-policy trace:

```text
route_option_lanes(..., fast_policy_decision=...)
  fast_policy_memory.selected_policy_ids
  fast_policy_memory.selected_actions
  slow_baseline_required = true
```

This deliberately does not make fast policies override the conservative router yet. It creates the audit surface needed for transition datasets and promotion gates.

Added `assumption_os.hle_transition_dataset`:

```text
transition_record_from_hle_row(...)
  question/run hash
  domain/category
  action path
  selected label hash
  gold after-run label hash
  correctness
  failure bucket
  cost/latency
  path hashes
  option feature hashes
  fast policy ids

build_transition_dataset(...)
  summary accuracy
  action counts
  failure-bucket counts
  no_fallback_count
  latency/cost sums
```

This implements the Week 1 "state-action-outcome dataset" idea without using gold in the decision path.

Extended `assumption_os.hle_transition_dataset` to read real HLE artifacts:

```text
load_hle_result_rows_from_path(...)
  JSON / JSONL
  top-level rows
  aggregate shards -> recursive shard loading

python3 -m assumption_os.hle_transition_dataset <hle-json> --out <out.transition_dataset.json>
```

The normalizer now supports the hash-only fields emitted by HLE runs:

```text
problem_id_hash
question_hash
prediction_hash
answer_hash
component_efficacy.selection.verified_or_abstain_gate
component_efficacy.flags
call_metadata.variant_watchdog
```

It keeps raw question/option content out of the persisted rows.

Added `assumption_os.hle_fast_policy_miner`:

```text
mine_fast_policy_hypotheses(...)
  transition rows -> candidate FastPolicyHypothesis records
  candidate_generation_missed_gold -> source_binding hypothesis
  source/directness gaps -> source_binding hypothesis
  verified_or_abstain no_fallback -> fallback_policy hypothesis
  high latency -> solver_lane hypothesis
```

All mined policies are `promotion_status = candidate`; they are not selected by the live fast-policy selector unless a later fixed/unseen promotion gate explicitly promotes them.

Generated local, ignored artifacts from the latest fixed-offset HLE run:

```text
phase four/assumption_graph/paper_readiness_20260604/hle_parallel_runs/
  hle_fixedoffsets_mc12_router_attempt_cap20_cacheonly_mini_20260707.transition_dataset.json
  hle_fixedoffsets_mc12_router_attempt_cap20_cacheonly_mini_20260707.policy_mining.json
```

Extracted summary:

```text
record_count = 12
accuracy = 2/12 = 0.1667
verified_or_abstain_gate_status_counts = {abstained: 6, no_fallback: 6}
primary failure buckets:
  candidate_generation_missed_gold = 7
  verified_or_abstain no_fallback = 2
  gold_option_direct_source_insufficient = 1
mined candidate policies = 3
promotion blockers:
  insufficient_unseen_transition_rows_min_24
  no_fallback_present
  missing_fair_control_or_split_metadata
```

This agrees with the GPT advice: the current evidence says to learn from transition rows and fix candidate/source/fallback lifecycle, not to promote another narrow HLE seed patch.

## Next Implementation Direction

Next should be a small continual-learning benchmark over this transition format:

```text
Round 0: raw / f577 / current on unseen 24
Round 1: distill fast policies from failures
Round 2: apply promoted policies only on new unseen 24
Round 3: repeat on held-out failure families
```

The important change is not another HLE rule; it is making promotion explicit and conservative.
