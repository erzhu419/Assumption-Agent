# HLE Expanded Domain-Rule Validation Summary

Date: 2026-06-16

This expands the prior targeted 3-item HLE validation to two parallel fresh shards:

- `hle_triad_expanded_domain_rule_shard_a_gpt54mini_n3_seed1800_20260616`
- `hle_triad_expanded_domain_rule_shard_b_gpt54mini_n3_seed2200_20260616`

The run used the same three-way protocol:

- raw `gpt-5.4-mini`
- HippoRAG-style transient retrieval baseline
- `assumption_agent_recursive_verify`

No raw HLE question text, gold answers, rationales, canaries, or image payloads are persisted.

## Aggregate Result

| variant | n | correct | accuracy | top-level errors |
| --- | ---: | ---: | ---: | ---: |
| `assumption_agent_recursive_verify` | 6 | 4 | 0.6667 | 0 |
| `raw` | 6 | 2 | 0.3333 | 1 |
| `hipporag_baseline` | 6 | 1 | 0.1667 | 2 |

## Clean Shared Subset

Clean shared subset means all three variants returned without top-level API error.

| variant | clean n | correct | accuracy |
| --- | ---: | ---: | ---: |
| `assumption_agent_recursive_verify` | 4 | 2 | 0.5000 |
| `raw` | 4 | 1 | 0.2500 |
| `hipporag_baseline` | 4 | 1 | 0.2500 |

## Interpretation

The expanded smoke is positive for the Assumption Agent, but not paper-clean because the endpoint produced top-level RuntimeError rows for controls. The result supports continuing larger validation, but it should not be reported as a final HLE claim.

The main engineering bottleneck is now runner reliability and throughput: long single-call waits and endpoint errors can dominate small HLE batches. The next validation should run sharded/parallel jobs with explicit per-call heartbeat, soft timeout, and error-stratified reporting.
