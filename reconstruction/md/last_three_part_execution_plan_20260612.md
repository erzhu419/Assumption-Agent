# Last Three Part Execution Plan

Source plan: `reconstruction/md/last_three_part.md`.

## Claim Ladder

- L1 bounded mechanism: fixed inputs, fixed budget, fixed queue, fixed validation protocol.
- L2 robust bounded system: repeated cycles, replay, checkpoint, recovery, negative controls, fault injection.
- L3 production candidate: default-enabled only in restricted real flows with gated apply.
- L4 unbounded/general claim: 24/7 autonomy, broad cross-domain replacement of validation, or full theorem proving.

Current target is L2 -> L3, not L4.

## Track Order

1. Track A: bounded autonomy envelope -> supervised production autonomy.
2. Track B: calibrated simulator candidate -> graph-action simulator for triage/routing.
3. Track C: finite category proof engine -> bounded formal reasoning stack.
4. Integrated mini-loop: connect A/B/C into replayable recursive self-evolution episodes.

## Immediate Tickets

| Ticket | Module | Status | Validation |
| --- | --- | --- | --- |
| A1 | `assumption_os/autonomy_journal.py` | implemented | `tests/test_autonomy_journal.py`, `autonomy_journal_replay_20260612.json` |
| A2 | `assumption_os/autonomy_queue.py` | implemented | `tests/test_autonomy_queue.py`, `autonomy_queue_lease_20260612.json` |
| B1 | `assumption_os/simulator_transition_schema.py` | implemented | `tests/test_simulator_transition_schema.py`, `simulator_transition_schema_validation_20260612.json` |
| B2 | `simulator_eval_splits.py` | pending | leave-domain/pattern/artifact splits |
| C1 | `finite_category_certificate.py` | pending | certificate schema + obligation checks |
| I1 | `integrated_recursive_episode.py` | pending | residual -> proposal -> simulator -> formal gate -> ablation -> replay |

## A1 Completion Snapshot

- Append-only JSONL journal.
- `event_id`, `cycle_id`, `idempotency_key`, graph before/after hashes.
- Deterministic replay.
- Duplicate event no-op.
- Idempotency conflict blocked.
- Crash-mid-cycle recovery replay.
- Graph hash divergence detection.

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/autonomy_journal_replay_20260612.json`

Metrics:

- `replay_same_journal_same_state=true`
- `duplicate_event_no_double_apply=true`
- `crash_mid_cycle_recoverable=true`
- `graph_hash_divergence_detected=true`

## Next Step

Implement B2 leave-pattern/domain/artifact split evaluation after the transition dataset is frozen.

## A2 Completion Snapshot

- Lease-based checkpoint queue.
- Task states: `pending`, `leased`, `completed`, `failed`, `deferred`, `blocked`, `expired`.
- Worker lease ownership and TTL.
- Retry-bounded requeue after crash/timeout.
- Terminal expiry after retry budget is exhausted.
- Completed task idempotency.
- Blocked task isolation: timeout processing cannot auto-unblock it.
- Atomic JSON checkpoint reload.
- Optional A1 journal writeback for every mutating queue operation.

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/autonomy_queue_lease_20260612.json`

Metrics:

- `double_lease_blocked_for_original_task=true`
- `worker_crash_releases_lease=true`
- `expired_task_requeues=true`
- `same_task_not_executed_twice=true`
- `retry_limit_expires_terminal=true`
- `blocked_task_not_auto_unblocked=true`
- `checkpoint_reload_same_state=true`
- `journal_replay_divergence_detected=false`

## B1 Completion Snapshot

- Frozen simulator transition schema `simulator_transition_schema_v0`.
- Materialized current Phase13 345 first-party transition-like rows into JSONL.
- Sources:
  - Phase10 reliability observed-arm rows: 51
  - residual fresh live judgments: 18
  - live multigeneration transition-like rows: 36
  - blinded recursive live judgments: 240
- Required row sections: `state`, `action`, `prediction`, `outcome`, `provenance`.
- Split labels: `train`, `validation`, `test`.
- Provenance hash on every row.
- Redaction check blocks prompt/answer/secret payloads.
- Invalid rows are written to quarantine.

Artifacts:

`phase four/assumption_graph/paper_readiness_20260604/simulator_transition_schema_v0.json`

`phase four/assumption_graph/paper_readiness_20260604/simulator_transition_dataset_v0.jsonl`

`phase four/assumption_graph/paper_readiness_20260604/simulator_transition_quarantine_v0.jsonl`

`phase four/assumption_graph/paper_readiness_20260604/simulator_transition_schema_validation_20260612.json`

Metrics:

- `raw_row_count=345`
- `valid_row_count=345`
- `invalid_row_count=0`
- `quarantine_row_count=0`
- `redacted_row_count=345`
- `split_counts={"train":255,"validation":37,"test":53}`
- `provenance_hash_unique=true`
- `secret_or_prompt_payload_detected=false`
