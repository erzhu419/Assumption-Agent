# Phase-3A M2.5 A8 R5 Attempt-5 Recovery Audit

## Status and scope

R5 is a code amendment and a one-shot authorization path for recovery attempt ordinal 5. It does not itself execute the real ceremony, read or hash `split_master_seed.bin`, mutate the four retained key volumes, create a formal actor container, publish formal roots, or start M3.

The fixed owner phrase is:

`AUTHORIZE_A8_R5_ATTEMPT_5_RECOVERY_START_ORDER_AND_COMPOSITE_FAILURE_COMPLETE_ONLY_REAL_PENDING_RESUME`

The audit namespace is `recovery-audit-r5-e4af9f57c38fb298462ec628c4ed8a03-attempt-5`. A visible `attempt-start.json` consumes ordinal 5. A visible `failure.json` or `finalize.json` is terminal, and a second execution is rejected.

## Frozen R4r2 parent

R5 accepts only a clean committed HEAD whose sole parent is `f24bae3c4fd1f4480e0aa9ecba69ac945779828d`. Before preparation and again before execution it verifies the exact eight-file R4r2 terminal inventory, regular-file metadata, raw SHA-256, full 64-hex self receipt, byte size, semantic links and chain root:

`9c0c0b8f05e97ec6b87c0ac9b4a36823f5338ce69053f442e9b1cbf1137f00d5`

The R4r2 canonical record proves ordinal 4 was consumed, no finalize exists, and the terminal code is `FAIL_M25_FORMAL_CONTAINER_RUNTIME`. It does not preserve the original primary exception. R5 therefore records `UNRECOVERABLE_FROM_CANONICAL_R4_RECORD` as the authoritative historical status.

Read-only forensics and a seed-free synthetic replay indicate the missing-seccomp start-order defect and no formal actor start. This is explicitly `NON_CANONICAL_READ_ONLY_FORENSIC_INFERENCE`. The synthetic replay is marked `UNATTESTED_NONCANONICAL_DIAGNOSTIC`; its claim is limited to exact seven names and modes plus the seed size, and it is not a formal custody receipt.

## Admission and live-state gates

The executor accepts only the exact 43-key `hegel-phase3-m25-a8-r5-source-admission/1` object. It binds the R5 commit, sole R4r2 parent, fixed A8/run/ledger identity, all eight R4 raw and self-receipt identities, the R4 failure tuple, isolated A8 validation and the unchanged 95-input A8 closure. Preflight separately binds all 13 runtime-exception sources to HEAD bytes, Git mode, worktree mode and normal index flags, so an excluded source cannot hide drift with `assume-unchanged` or `skip-worktree`.

The live preflight must observe all of the following without opening the seed:

- marker `PENDING` and journal `RESERVED`;
- public evidence, promotion and publication receipt absent;
- zero run-labelled containers, exactly four retained labelled volumes, and no network operation;
- seed `lstat` metadata: regular name, 32 bytes, mode `0600`, with `raw_bytes_read=false` and `sha256_computed=false`;
- zero additional formal-identity entropy draws and M3 disabled.

The R5 child extends the historical runtime-exception registries only to name its two later source artifacts; the frozen parent commit and frozen 95-file A8 root remain unchanged. Every one of those 95 files is re-read through the same descriptor/inode-stable HEAD verifier before admission.

## Start and cleanup correction

The recovery executor no longer performs a premature custody reclaim before its frozen runtime inputs are prepared. Recovery start accepts only the caller-owned mode-`0700` custody directory and leaves the handoff flag false until a real handoff occurs. Cleanup reclaims only after a handoff; a failed reclaim is retained as typed invalid custody rather than followed by unsafe probes or volume destruction.

## Failure evidence

`A8R5RecoveryAmendmentError` is a formal executor error. All failure records use the executor's bounded recursive evidence walker (maximum depth 32 and maximum nodes 256). The record preserves:

- the terminal primary code and detail hash;
- every typed cleanup code and detail hash;
- a separate final-close summary, including when final-close is the sole failure;
- the complete bounded evidence tree and its canonical hash;
- the R4 canonical-primary absence disclosure and non-canonical inference labels.

Raw exception details are never persisted. A cleanup, final-close, visibility check, hidden-file discard, canonical retry, or failure-audit durability error is combined with—not substituted for—the primary failure. A concurrently visible `failure.json` is accepted only if its exact canonical bytes equal the locally reconstructed record.

## Fault qualification

The R5 test matrix covers attempt-start, admission, finalize and failure installation both before and after the durable link; post-primary visibility and discard faults; exact and conflicting concurrent failure records; primary-plus-final-close and sole-final-close failures; bounded deep composites; successful finalize and one-shot rejection. Tests use synthetic paths and actors only. They do not invoke the formal custody transaction, seed, retained volumes or real Docker roles.

## Commands after a committed manifest exists

Run `python -m hegel_machine.phase3_m25_a8_recovery_cli_r5_v1 preflight ...` first. Preparation installs only the four-record prefix. The owner action installs authorization using the exact phrase above. `recover-fixed-complete-seed` is a separate explicit action and is the only command permitted to consume ordinal 5.

CLI stderr is diagnostic transport, not formal evidence. Formal executor errors retain their typed code/detail; non-formal `OSError` and `ValueError` paths disclose only the exception type and never invoke an arbitrary exception stringifier.
