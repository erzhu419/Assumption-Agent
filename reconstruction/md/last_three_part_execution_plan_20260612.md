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
| A2 | `assumption_os/autonomy_queue.py` | next | lease/retry/no-double-execute tests |
| B1 | `simulator_transition_schema.py` | pending | validate 345 redacted transition-like rows |
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

Implement A2 lease-based queue semantics before touching simulator policy. This keeps the autonomy substrate replayable and crash-safe before new decision logic is added.
