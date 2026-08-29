# Phase-3A Q0.5b attempt-2 Target-D campaign v2

Status: **implemented, tested, not preflighted, not executed**.  This additive
successor does not change the old V1 attempt-2 bridge or the v2r20 shadow
campaign.  A successful v2r20 shadow slot remains historical gate evidence;
it is not the formal actual result.

## Authority and source

The exact owner text is `授权 Q0.5b attempt-2 actual`.  It authorizes one
logical attempt-2 and at most one Target-D supervisor `Popen`.  The campaign
binds the clean runtime HEAD/tree, requires the exact post-Meta ancestor
`a1e7119baab55faecac837ddb3dec0d336078cf6`, and reuses the Target-D supervisor
lineage rooted at `5b38581f43c9fd88692b4f3c2cf8f13e8a0ca24b`.

The formal scope is Q0.5b qualification only.  Q1 remains `NOT_RUN`; attempt-3,
retry, network, unrelated production targets, global Meta adoption, ACTIVE,
promotion, and current-pointer writes are not authorized.

## Real Meta V11 input

Both entrypoints require `--meta-chain-manifest` naming an absolute ordinary
JSON file with schema
`hegel-phase3a-q05b-attempt2-meta-v11-chain-manifest/2`.  It contains exactly:

```json
{
  "schema_version": "hegel-phase3a-q05b-attempt2-meta-v11-chain-manifest/2",
  "artifacts": {
    "competition_input": {"path": "/absolute/file.json", "bytes": 1, "sha256": "..."}
  },
  "expected_digests": {
    "expected_competition_contract_digest": "sha256:..."
  }
}
```

`artifacts` must contain the contract's 30 names in exact order and
`expected_digests` its 29 names in exact order.  Every row is replayed from its
real file.  The frozen public Meta verifier must return
`VERIFIED_QUALIFIED_FRESH_POST_TRANSITION_EVALUATOR_EPOCH`.  A missing,
incomplete, unqualified, or not-yet-readable real chain returns exit 75 with
`PENDING_ATTEMPT2_TARGET_D_V2_NOT_CONSUMED`; synthetic pytest objects cannot
serve as formal input.

## Entry sequence

Preflight:

```bash
/usr/bin/python3.10 -I -S -B \
  tools/phase3_q05b_attempt2_target_d_campaign_v2.py \
  --preflight \
  --source-commit FULL_CLEAN_HEAD_40_HEX \
  --meta-chain-manifest /absolute/meta-v11-chain.json
```

After preflight returns a capsule digest, the only run form is:

```bash
/usr/bin/python3.10 -I -S -B \
  tools/phase3_q05b_attempt2_target_d_campaign_v2.py \
  --run \
  --source-commit THE_SAME_FULL_CLEAN_HEAD_40_HEX \
  --meta-chain-manifest /absolute/meta-v11-chain.json \
  --capsule-digest sha256:THE_PREFLIGHT_DIGEST
```

Every RUN first acquires the stable campaign lock named below and holds it
through residue handling, ledger replay, all fresh gates, consumption, the sole
`Popen`, and terminal publication.  A concurrent RUN waits for that lifecycle
to finish; once admitted, it replays the terminal or consumed ledger instead of
cleaning an active entry or invoking a second supervisor.

Preflight and a fresh locked RUN replay, in order: exact authorization; path/format;
clean source lineage; the real Meta V11 chain; Target-D raw source and Git
object closure; isolated Python 3.10, vendored Tomli, and Cargo.lock; immutable
v2r20 live-gauntlet and decision-audit evidence; exact CPU affinity with
restore; the real scheduling admission; and formal-root absence.  These gates
all finish before consumption or `Popen`.

The v2r20 replay source is loaded from the historical contract's frozen clone
origin and exact campaign commit.  This preserves the old tool's path-bound
`WORKTREE_ROOT` interpretation when the additive formal successor runs from
the m3 worktree; the historical tool, contract, and evidence remain unchanged.

## PENDING and terminal boundary

Repairable pre-consumption path, file, format, permission, dependency, CPU, or
scheduling conditions return exit 75 without authorization consumption,
campaign execution, ledger record, retry transition, or supervisor `Popen`.
Transient scheduling probes and known-empty setup roots are removed, their
parents fsynced, and absence proved before PENDING is returned.

Consumption is intentionally the first durable semantic ledger record:

1. `attempt-2.consumed.json`
2. `attempt-2.terminal.json`

This removes an intent-to-consumption disk-failure gap.  The consumed record
contains the exact capsule and is installed immediately before the sole
`Popen`.  If a ledger already exists, the lock holder replays it before any
fresh gate.  The terminal outcome digest, decision-audit digest, decision
predicates, the byte-decoded external outcome, and the identical audit copies
in external outcome and terminal body are all compared or recomputed before
its status is returned.  Consumed-without-terminal is
`ATTEMPT2_TARGET_D_V2_CONSUMED_OUTCOME_UNKNOWN_TERMINAL_NO_RESUME`.  No missing
manifest or later scheduling outage can turn a consumed attempt into PENDING.

The external `attempt-2.outcome.json` and ledger terminal body persist the
same decision and decision-audit object from one core return.  After `Popen`,
success, method failure, process failure, interruption, or publication
uncertainty is terminal and cannot authorize another invocation.

## Fixed namespaces

- Stable campaign lock: `/home/erzhu419/.local/state/hegel-machine/q05b-attempt2-target-d-v2.campaign.lock`
- Ledger: `/home/erzhu419/.local/state/hegel-machine/q05b-attempt2-target-d-v2-ledger`
- Evidence: `/home/erzhu419/.local/state/hegel-machine/q05b-attempt2-target-d-v2-evidence`
- Trajectory: `/home/erzhu419/.local/state/hegel-machine/q05b-attempt2-target-d-v2-trajectory`
- Scheduling probe: `/home/erzhu419/.local/state/hegel-machine/q05b-attempt2-target-d-v2-scheduling-probe`
- Artifact: `/home/erzhu419/.local/state/hegel-machine/m3-start-runtime-worktree/Hegel Machine/artifacts/phase3_q05b_dual_qualification_v2.json`

The lock is a persistent coordination file rather than an attempt output root.
No fixed lock or formal output namespace was created while implementing or
testing this successor.
