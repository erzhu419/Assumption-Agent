# Hegel Machine Phase-3A M2.5 — A8 R7 Attempt-7 Post-stage Recovery Audit

Status: **IMPLEMENTATION QUALIFIED / ATTEMPT-7 NOT RUN**

This amendment is a sealed continuation of one already-staged formal
transaction. It does not authorize a new ceremony or a second attempt at the
pre-stage core.

## Frozen parent evidence

- Parent amendment commit: `e10fa89575af19c85e9744533e16d648463be451`
- Formal basis commit: `0af65964235390ce2bebefea7379eaa9c50eda24`
- Run ID: `e4af9f57c38fb298462ec628c4ed8a03`
- Ledger ID: `ec849e2f1e2e1163cfc450370b25b484`
- R6 terminal audit: 10 exact ordered records
- R6 terminal chain root: `d17b2fc442226b1800f7f4900b52dbca824f5391ca0ab0ec1d4f6fc034711de2`
- R6 terminal failure: `FAIL_M25_FORMAL_CUSTODY_STATE` at
  `COMPLETE_ONLY_FORMAL_CORE`
- Fixed staged prestage-intent SHA-256:
  `89d8414cd68adaa084b3dfe865abf5d9245806764d89413dc7d1503e6dffc0ab`
- Fixed R6 source-admission SHA-256:
  `c2003bcc77db04c2672d59d1aada6e45baeb4275df6f8a3f8304c68f8ef26828`

The current admitted recovery shape is exactly `STAGED_PENDING / PENDING`,
with zero run-labelled containers and four retained run-labelled role volumes.
No claim in this document says that ordinal 7 has been consumed.

## R7 authority boundary

R7 may only:

1. replay the already admitted R6 `source_admission` with the already staged
   prestage intent;
2. rehydrate the exact staged transaction and acquire its recovery locks;
3. recheck the five immutable pre-attempt audit records by bytes and inode;
4. install `attempt-start.json` while the rehydrated transaction lock is held;
5. call `_continue_post_stage_transaction_recovery_core_v1` once;
6. emit exactly one terminal `finalize.json` or `failure.json` after admission.

The five-record authorization prefix is:

1. `preflight.json`
2. `incident-diagnostic.json`
3. `poststage-qualification.json`
4. `authorization-request.json`
5. `authorization.json`

`attempt-start.json` is the unique ordinal-7 consumption token. The authorized
owner phrase is:

`AUTHORIZE_A8_R7_ATTEMPT_7_FIXED_STAGED_POSTSTAGE_PENDING_IDEMPOTENT_CONTINUATION_ONLY`

### Historical R6 replay bridge

The ordinary fixed-validator path still requires `HEAD` to equal the admitted
amendment commit. R7 uses a separate public-only bridge because its committed
source is necessarily one commit after the terminal R6 source admission. That
bridge accepts only when all of the following are simultaneously true:

- the canonical R6 source-admission hash is exactly `c2003bcc...`;
- the expected child identity is the R7 commit already verified by preflight;
- current `HEAD` equals that expected child and has R6 as its sole parent;
- R6 still has `0024f811...` as its sole parent;
- the validator `ls-tree` row is byte-identical in R6 and R7;
- the live validator remains a regular `0644` file whose bytes equal the
  admitted R6 Git blob.

This is not an ancestor allowance. A sibling, merge commit, grandchild,
different admission, or validator mode/blob/path drift is rejected before the
isolated validator or gate evaluator runs.

## Explicit prohibitions

R7 does not permit:

- a new seed draw or raw-seed read/hash by the orchestrator;
- new keys, signatures, run IDs, ledger IDs, or formal identities;
- the pre-stage core, ordinary execute, or a new static/source qualification;
- a replacement staged payload or a different source admission;
- network pulls or non-local runtime artifacts;
- `phase3-m3-start` or any transition away from `NOT_RUN`.

On a successful post-stage continuation, the expected result is 24/24 formal
gates with child state still `NOT_RUN`. Gate readiness does not start M3.

## Current verification

The R7 implementation and its focused unit tests were constructed without
executing the real recovery. The tests freeze the R6 terminal root, exact
source admission and staged intent, post-stage-only policy, audit-prefix inode
recheck, lock-before-token ordering, CLI argument boundary, and the historical
bridge's admission/commit/parent/tree anti-drift matrix. A seed-free read-only
qualification also replays the exact staged public evidence through the
historical bridge and obtains the exact staged promotion bytes: gates advance
from 14 to 24 while the child state remains `NOT_RUN` and all 15 M3 output
slots remain null.

The final source manifest and its source hashes are present and verified.
External authorization records and real ordinal-7 execution remain
intentionally absent at this status.
