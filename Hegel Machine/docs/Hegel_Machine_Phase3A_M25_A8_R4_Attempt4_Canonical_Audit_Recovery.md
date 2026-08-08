# Phase-3A M2.5 A8 R4 Attempt-4 Canonical-Audit Recovery

Status: narrow recovery amendment for a still-PENDING/RESERVED formal
transaction after recovery attempt ordinal 3 was consumed and terminalized.
It is not a retry of ordinal 3, not a new formal basis and not an M3 start.

## R3.1 terminal outcome

R3.1 commit `6c1b73064d292d57d5a9c35fd83c75caff57c300`
passed its clean committed-source preflight and received a fresh fixed-transaction
authorization. The formal command was invoked exactly once. It atomically
published a complete canonical `attempt-start.json`, so ordinal 3 is consumed.
The command then terminalized before source admission or actor execution.

The cause was a post-publication verifier defect, not different evidence.
`runtime_artifact_metadata` is a tuple of three dictionaries in the typed
builder and a JSON list after decoding. The builder and stored record had the
same canonical bytes and the same raw SHA-256
`09bbc99ad2b33930a043b0178bc5c1ebc3f71dfb09b025a412fbb00224493312`,
but direct Python object equality returned false.

The failure receipt is canonical and self-hashed. It records:

- phase `ATTEMPT_START_DURABILITY`;
- admission `null`;
- raw SHA-256
  `90c176985d83780440007d2111577c0dc5ffbae5430eae523919653b7b6b0153`;
- raw-seed read/hash false;
- M3 start false.

The transaction remains marker `PENDING`, journal `RESERVED`, zero run-labelled
containers, four original key volumes, three unchanged public reservations and
no formal public output. The raw seed was not opened or hashed.

## Frozen terminal chain

The R3.1 audit directory is immutable and read-only for R4. It contains exactly
seven mode-0600, single-link records in this order:

1. `preflight.json`
2. `incident-diagnostic.json`
3. `a8-validation-receipt.json`
4. `authorization-request.json`
5. `authorization.json`
6. `attempt-start.json`
7. `failure.json`

There is no hidden `.next`, admission or finalize record. Using rows with exact
keys `name`, `raw_sha256`, `receipt_sha256`, `size_bytes` and `mode_octal`, the
canonical terminal-chain root is:

`d4bb2c5984405d127537bde1e973f175b630a16bcaa8ec4fe15617e665400093`

R4 replays every raw/self-receipt hash, mode, owner, group, link count, internal
provenance link and failure-semantic field before it may prepare authorization.

## Exact installer correction

All authoritative audit installation uses three checks:

1. before publication, supplied raw bytes must equal canonical JSON of the
   self-receipted typed builder;
2. after publication, the independent reader must accept canonical JSON and
   its self-receipt;
3. the observed raw bytes must equal the supplied raw bytes exactly.

Decoded Python container shape is not an identity layer and is never compared.
Content changes, trailing bytes, non-canonical JSON, invalid self receipts and
pre-existing different visible records remain fail-closed.

The attempt-4 qualification drives the complete execution state machine through
pre-link and post-link faults at `attempt-start`, `admission`, `finalize` and
`failure`, plus complete-only core failure, final-public-replay failure and the
successful path. A non-authoritative hidden `.next` is either removed before a
terminal record is accepted or, only for an unconsumed pre-link attempt-start,
discarded and recreated by the next invocation. An exact visible attempt-start
always consumes ordinal 4; an exact visible finalize always resolves as the
already-verified success. Re-entry after either consumption outcome is rejected.

## Attempt-4 identity and authorization

R4 must be one clean committed sole child of R3.1 commit
`6c1b73064d292d57d5a9c35fd83c75caff57c300`. Its source admission is a new
exact schema `hegel-phase3-m25-a8-r4-source-admission/1` with ordinal 4. It
retains all R1/R2 anchors and binds:

- R3 commit `52a4a61934a73c70dc09b919cae377db166eaedf`;
- R3.1 commit `6c1b73064d292d57d5a9c35fd83c75caff57c300`;
- the current committed R4 HEAD as its sole child;
- the R3 pre-attempt prefix root and R3.1 seven-record terminal root;
- R3.1 attempt/failure raw and self-receipt hashes, failure phase/detail and
  null admission;
- the exact current incident, A8 validation, unchanged-input closure and
  actor/errata/live-bundle hashes.

R4 uses the new repo-external namespace:

`/home/erzhu419/.local/state/hegel-machine/phase3-m25-0af65964235390ce2bebefea7379eaa9c50eda24/recovery-audit-r4-e4af9f57c38fb298462ec628c4ed8a03-attempt-4`

The fresh revision ID is `R4_CANONICAL_AUDIT_INSTALLER_V1`. Authorization must
use exactly:

`AUTHORIZE_A8_R4_ATTEMPT_4_CANONICAL_BYTES_COMPLETE_ONLY_REAL_PENDING_RESUME`

Preparation and authorization do not consume ordinal 4. Only an exact visible
`attempt-start.json` in the R4 namespace consumes it. Once visible, R4 is never
reinvoked even if a later phase fails.

## Preserved prohibitions and success condition

R4 does not authorize ordinary execute, abort, redraw, post-stage recovery,
raw-seed read/hash, network access, identity regeneration or `phase3-m3-start`.
All role containers remain digest-pinned, offline and independently keyed.

Success is only a fully replayed `24/24` gate report with child state `NOT_RUN`,
M3 not started and all 15 M3 output roots null. A separate future action is
still required to start M3.
