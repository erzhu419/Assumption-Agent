# Phase-3A M2.5 A8 R3.1 Pre-Attempt Canonical-Bytes Erratum

Status: narrow implementation erratum for the still-unconsumed recovery
attempt ordinal 3. It is not a new formal basis, not ordinal 4, not a retry of
a consumed attempt and not an M3 start action.

## Frozen pre-attempt outcome

R3 commit `52a4a61934a73c70dc09b919cae377db166eaedf` was prepared and freshly
authorized. Its recovery command then rejected before actor construction,
formal-lock acquisition, source admission and publication of
`attempt-start.json`. The R3 audit directory contains exactly five immutable
records: preflight, incident, isolated-A8 validation, authorization request and
authorization. It contains no hidden temp, attempt-start, admission, failure
or finalize record. No run-labelled container or public output was created;
the four pre-existing key volumes remain.

The fixed five-row prefix root is
`9771b20bf63f1095456618d3ccd4c9db0c54c693307314b8aea72afa18249999`.
Each row binds only name, raw SHA-256, self-receipt SHA-256, byte length and
mode. Device, inode and timestamps are deliberately excluded from the portable
root but regular-file identity, owner, group, mode and exact directory
inventory are checked live.

## Defect and exact correction

The stored R3 incident and a fresh reconstruction are byte-for-byte identical:
both are 12,586 bytes with raw SHA-256
`d0b27d5c7f1f00a74873bac2394f05fb6666a29e07fdbf9886999f0dddbebc21`.
The rejection came only from direct Python object equality: JSON arrays in the
stored record deserialize as lists, while nine typed diagnostic fields in the
builder are tuples. R3.1 replaces that comparison with equality of the exact
canonical self-receipted bytes:

```python
if incident_raw != _receipt_record_bytes_v1(incident_now):
    _fail("stored R3 incident differs before attempt")
```

No generic coercion, field omission or relaxed comparison is allowed.

## Provenance and authorization

R3.1 must be one clean committed sole child of R3 commit
`52a4a61934a73c70dc09b919cae377db166eaedf`. Its manifest binds only the
R3.1 delta. The original five records remain untouched at their old path and
are revalidated and embedded in the new incident as
`PRE_ATTEMPT_SUPERSEDED_IMPLEMENTATION_DEFECT`. A new repo-external audit path
is fixed to:

```text
/home/erzhu419/.local/state/hegel-machine/
phase3-m25-0af65964235390ce2bebefea7379eaa9c50eda24/
recovery-audit-r31-e4af9f57c38fb298462ec628c4ed8a03-
attempt-3-revision-1
```

The revision ID is `R31_CANONICAL_INCIDENT_BYTES_V1`. A fresh owner action must
use exactly:

`AUTHORIZE_A8_R31_ATTEMPT_3_REVISION_1_CANONICAL_BYTES_COMPLETE_ONLY_REAL_PENDING_RESUME`

The executor-facing source-admission schema and action remain the already
frozen ordinal-3 wire,
`hegel-phase3-m25-a8-r3-source-admission/1` and
`CODE_AMENDMENT_RECOVERY_CONTINUATION`. R3.1 does not widen that formal wire.
Its source admission binds the new incident raw hash; the incident in turn
binds the revision ID, defect code and old-prefix root. The executor separately
requires the admitted HEAD to be a single child of the fixed R3 commit and the
validator bytes to equal that committed blob.

## Consumption and prohibited operations

`attempt-start.json` remains the sole consumption edge, so ordinal 3 is still
unconsumed. The new preparation and authorization records do not consume it.
Once the new audit publishes an exact visible attempt-start, ordinal 3 can
never be invoked again. Success must remain `24/24`, child state `NOT_RUN`, M3
not started and all 15 M3 output roots null.

R3.1 does not authorize ordinary execute, abort, seed redraw, raw-seed read or
hash, network access, post-stage recovery, identity regeneration, or
`phase3-m3-start`.
