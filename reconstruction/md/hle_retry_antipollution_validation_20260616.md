# HLE Retry And Anti-Pollution Validation 2026-06-16

## Scope

This pass addressed two failure modes seen in recent HLE runs:

- endpoint/concurrency noise: live runs were dominated by endpoint `RuntimeError` and long-hanging requests;
- pollution: generic graph harness context and weak transient evidence were being activated and could be mistaken for useful agent cognition.

No raw HLE questions, gold answers, rationales, canaries, prediction text, or API keys are persisted in the artifacts below.

## Code Changes

- Added model-router retry controls:
  - `MODEL_ROUTER_PER_ATTEMPT_TIMEOUT`
  - `MODEL_ROUTER_BACKOFF_BASE_SEC`
  - `MODEL_ROUTER_GLOBAL_CONCURRENCY`
  - `MODEL_ROUTER_GLOBAL_CONCURRENCY_DIR`
  - `MODEL_ROUTER_GLOBAL_SLOT_TTL_SEC`
  - `MODEL_ROUTER_GLOBAL_SLOT_WAIT_SEC`
- Changed HLE parallel runner default `--shard-size` to `1` so one slow problem cannot contaminate a multi-sample shard.
- Added `pollution_audit` to HLE aggregate artifacts:
  - fresh problem hash accounting
  - cache/live separation
  - endpoint error separation
  - context pollution summary
  - selection credit assignment
  - claim scope downgrade when endpoint noise exists
- Blocked generic graph harness context from entering prompts or exact-answer repair:
  - generic graph context is still counted as retrieved;
  - it is now marked as discarded instead of injected.
- Blocked broad evidence bridge for math-domain HLE items:
  - math should route through math tool / candidate claim verifier rather than Wikipedia-style evidence.

## Validation

Unit regression:

```bash
python3 -m unittest tests.test_hle_smoke_eval tests.test_hle_parallel_shard_runner
```

Result: `58 tests OK`.

Endpoint probe:

- model: `gpt-5.4-mini`
- retry path: global slot + per-attempt timeout
- result: returned in ~3 seconds.

Shard-size stress finding:

- `hle_parallel_triad_live_n6_retry_limited_antipollution_seed2400_stride37_gpt54mini_20260616`
- endpoint errors: `0`
- process timeouts: `1`
- conclusion: retry/concurrency fix removed endpoint errors, but `shard_size=2` can still let a slow item contaminate a shard.

Clean shard-size-1 validation:

- `hle_parallel_triad_live_n4_retry_limited_antipollution_shard1_seed2200_stride23_gpt54mini_20260616`
- pass: `true`
- paper clean pass: `true`
- pollution pass: `true`
- distinct samples: `4`
- live calls resolved: `12/12`
- top-level endpoint errors: `0`
- process timeouts: `0`
- accuracy: agent `0.0`, raw `0.0`, HippoRAG `0.0`
- conclusion: runner quality improved, but this batch does not support an HLE capability-improvement claim.

Same-batch anti-pollution before/after:

- before math evidence gate:
  - `hle_parallel_triad_live_n2_antipollution_gate_shard1_seed2292_stride23_gpt54mini_20260616`
  - evidence context used: `2`
  - evidence context wrong: `2`
  - generic graph context discarded: `2`
- after math evidence gate:
  - `hle_parallel_triad_live_n2_antipollution_gate_v2_shard1_seed2292_stride23_gpt54mini_20260616`
  - evidence context used: `0`
  - evidence context wrong: `0`
  - generic graph context discarded: `2`
  - graph context used: `0`
  - endpoint errors: `0`
  - process timeouts: `0`
  - pollution pass: `true`

## Interpretation

This pass fixes measurement quality and two concrete pollution paths. It does not improve HLE accuracy on the small validation batches: the correct claim is that endpoint noise and context pollution are now separated and gated, not that HLE task performance improved.

Next HLE work should use `shard_size=1`, fresh unseen seed windows, and the pollution table as a hard claim guard before reporting agent-vs-baseline improvements.
