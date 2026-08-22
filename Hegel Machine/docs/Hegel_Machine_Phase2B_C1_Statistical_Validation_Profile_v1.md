# Hegel Machine Phase-2B C1 Statistical Validation Profile v1

## Status and evidence boundary

This companion is a pure candidate-mechanics skeleton.  Its strongest success
claim is exactly
`STATISTICAL_VALIDATION_CANDIDATE_MECHANICS_VERIFIED_NOT_R08`.  It can classify
a closed caller-supplied candidate as `COMPLETE_PASS`, `COMPLETE_FAIL`, or
`REJECTED`, but even `COMPLETE_PASS` does not satisfy R08.

There is deliberately no profile config in this construction slice.  The
final Formal Wire V3 and R06 generator qualification identities are not pinned
here.  The verifier instead requires the original canonical R06 candidate and
result bytes and the original canonical R07 candidate and result bytes, binds
their hashes into the candidate result, and keeps `r06_qualified`,
`r07_frozen`, and `r08_satisfied` false.  Canonical binding is not authority,
qualification, durable custody, or proof that an actual run occurred.

The module is computation-only.  It performs no filesystem or network I/O,
does not invoke Docker, does not consult a clock or process RNG, does not
generate a key or signature, does not write a CAS object, and does not execute
the recognizer or any baseline.  Public-key signature verification operates
only on caller-supplied bytes in memory.

## Closed cohort and systems

Each validation version has two fixed batches:

- `PRIMARY_960`: 720 main rows plus 240 semantic-conflict challenge rows;
- `PRESERVATION_VARIANTS_572`: 496 legal variants plus 76 invalid controls.

The system order is Hegel recognizer, embedding nearest-prototype, frozen LLM
semantic-only, and flat learned typed.  Every complete attempt must bind all
1,532 rows for all four systems.  Attempt 1, when permitted, reuses the exact
cohort object, byte digests, batch roots, generation-receipt root, and all five
seed roots from attempt 0.

The cohort object also cross-binds the exact candidate profile ID and the hash
root of the four original R06/R07 candidate/result byte strings.  Consequently,
a signed attempt cannot be transplanted onto a different prerequisite graph
merely by replacing the top-level graph fields.

The five seed-root purposes are latent generator, answer, shuffle, identifier,
and padding.  Each root is exactly 32 bytes, has a purpose-domain commitment,
and is distinct within a validation version.  If validation version 2 is used,
all five roots must be fresh and disjoint from version 1.

## Current metric closure

The profile contains the current twelve overall gates and expands the three
family/scale slice gate bases to 24 slice results.  Wilson lower bounds use the
fixed binary64 constant `0x1.a515209676ab8p+0`.

Metrics are not accepted as free-standing aggregate claims.  A complete
attempt carries four closed evaluator manifests: exactly 720 ordered main
rows, 240 ordered challenge rows, 572 ordered preservation rows, and the exact
240-row answerable projection used for scale regret.  The verifier recomputes
every numerator and denominator from those rows.  It rejects duplicate or
overlapping row IDs, quota/order drift, or a reported aggregate that differs
by one count.

The overall gates are, in order: family exact, binding exact, scale-set
accuracy, joint exact, hard-negative rejection, binding-counterfactual
rejection, scale-counterfactual rejection, sign/invariant-break rejection,
abstention specificity, fail-closed rate, preservation consistency, and
nonidentifiable-scale abstention.  The slice bases are answerable joint exact,
all-control rejection, and abstention specificity for each of six families and
two scales.

For an answerable row, joint exactness is exactly the conjunction of family,
binding, scale-set, and decision exactness, so joint cannot exceed any
component.  For a control or nonidentifiable row, component exactness is
absent and decision exactness is identical to abstention.  The six family
slices and two scale slices must independently sum back to the same overall
joint, control-rejection, and abstention-specificity counts.

Preservation has two simultaneous conditions.  Legal variants require the
existing point and Wilson gates (`>=0.97` and lower bound `>=0.94`), hence at
least 482 correct of 496.  Invalid controls require exactly 76 of 76.  The two
counts are folded into the reported preservation result; an invalid-control
error cannot be hidden by high legal accuracy.

Fail-closed has a nonzero denominator equal to all signed lifecycle events in
the applicable validation version and requires exact accuracy 1.  The result
also publishes every per-version denominator and an aggregate denominator for
the whole graph; both layers must have the same exact numerator and denominator.
It therefore cannot pass with an empty event set, a skipped event record, or a
denominator that silently ignores the infrastructure attempt of a permitted
retry.

## Exact scale regret and challenge gates

Scale regret contains 240 reduced nonnegative rational values with denominator
at most 1,024, arranged as 12 family-by-scale clusters of 20.  The exact point
mean must be at most `1/20`.
The paired-cluster bootstrap has 10,000 replicates and its nearest-rank
one-sided 95% upper value at zero-based sorted index 9,499 must be at most
`0.08`.

The challenge has 120 low-overlap structural positives and 120 high-overlap
structural negatives.  The system needs at least 109/120 low-overlap correct,
with Wilson lower bound at least 0.85, and at least 114/120 high-overlap
correct, with Wilson lower bound at least 0.90.

Semantic-only advantage is

```text
system correct - max(embedding correct, frozen-LLM correct)
```

over the 240 challenge rows.  The point advantage must be at least 0.15.  In
each bootstrap replicate, the maximum of the two resampled baseline totals is
taken inside that replicate.  The one-sided 95% lower value at zero-based
sorted index 499 must be at least 0.10.

Both bootstraps use the fixed master seed `411876909552964556` and a pure
HMAC-SHA256 uint32 stream.  Sampling uses rejection above the largest multiple
of the cluster count below `2^32`; it does not introduce modulo bias and does
not consult external RNG state.

## Attempt, signature, CAS, and time mechanics

A validation version contains attempt index 0, or indices 0 then 1.  Attempt 1
is permitted only with action `REEXECUTE`, exact
`any_raw_prediction_byte_produced=false`, and zero raw output bytes.  The only
four retry reasons and terminal codes are `CONTAINER_START_FAILURE`,
`HOST_OOM_BEFORE_FIRST_PREDICTION_BYTE`,
`HOST_TERMINATION_BEFORE_FIRST_PREDICTION_BYTE`, and
`INPUT_TRANSFER_CHECKSUM_MISMATCH`.  Parser, verifier, model, or statistical
failure—and any failure after even one raw prediction byte—cannot trigger a
retry.  The second attempt must produce a statistical `COMPLETE_PASS` or
`COMPLETE_FAIL`; there is no third attempt.

Six distinct candidate public keys have the fixed purposes custodian, runtime,
observer, evaluator, store, and time.  Each attempt has six contiguous signed
events in that purpose order:

1. cohort bound;
2. runtime execution recorded;
3. observer outputs bound;
4. evaluator metrics bound;
5. store-if-absent recorded;
6. attempt finalized.

The event chain binds the previous event ID, an exact contiguous logical time,
safe Unix seconds, nanoseconds in `[0,999999999]`, and strictly increasing
declared timestamps.  These are declared-time mechanics only; the verifier
does not consult or authenticate a clock.

Each attempt also has a store-signed candidate CAS receipt.  Its object key is
fixed by validation version and attempt, and its only permitted transition is
generation 0 to generation 1.  An upload retry may occur once, but both upload
digests must be byte-identical to the committed value digest.  This separate
action is named `REUPLOAD_COMMITTED_OUTPUT`, and its only retry reason is
`OUTPUT_UPLOAD_FAILURE_AFTER_OUTPUT_HASH_COMMITTED`: it cannot authorize
`REEXECUTE`, change cohort bytes, or change a committed archive hash.  This verifies a candidate receipt
signature and content binding, not the existence,
durability, independence, or rollback resistance of an external store.

## Validation version 2 bump and NO-GO

Version 2 is not an implicit retry or an editable continuation of version 1.
It is permitted only after a signed and CAS-bound version-1
`COMPLETE_FAIL`.  A content-addressed bump record must bind that immutable
failure archive and CAS receipt.  It also binds an ordered full-version root
covering every version-1 attempt archive, signed-event ID sequence, CAS subject,
CAS signature, and committed value digest, so an earlier infrastructure attempt
or receipt cannot be omitted or reordered.  The exact protocol, selector, and validation
IDs all move from their v1 ID to their v2 ID in the same record; a partial or
unchanged ID pair is rejected.  All five version-2 cohort byte/root digests and
all five seed roots must be fresh and disjoint from version 1.  A passing
version cannot be bumped.

Arbitrary nonempty JSON is not a development or issuance record.  The bump has
a content-addressed core, one closed public v1 failure-development report with
schema `hegel-machine-phase2b-c1-public-failure-development-report/1`, and one
closed v2 issuance/authority record with schema
`hegel-machine-phase2b-c1-version-two-issuance-authority/1`.  Both schemas
internally repeat and bind the immutable failure root, failure CAS receipt,
ordered full-version attempt/archive/CAS root, bump-core ID, old/new
three-version IDs, v1/v2 cohort roots, and v1/v2 seed
roots.  The public report additionally binds the public development change-set
digest and states that only public or synthetic inputs were used.  The
issuance states candidate-only authority and keeps authenticated authority
false.  The verifier reconstructs both canonical byte strings exactly.

If version 2 also ends in `COMPLETE_FAIL`, the result carries an exact
`NO-GO` marker.  No version 3 or unrecorded extra tuning cycle exists in this
skeleton.  The authority bytes are still only caller-supplied candidate bytes;
their canonical binding does not authenticate an authority.

## Atomic results

`COMPLETE_PASS` means the statistical gates pass for the terminal permitted
validation version—v1 when there is no valid bump, otherwise v2—while every
supplied lifecycle event, signature, candidate CAS receipt, declared-time
relation, and per-version gate record was structurally verified.  It is not a
claim that every historical version passed.  `COMPLETE_FAIL` means that
terminal permitted version was structurally valid but its statistical outcome
did not pass.  `REJECTED` means structural,
canonical, signature, lifecycle,
resource, or mutation verification failed.  A rejection publishes no partial
metric or readiness result.

The result content address covers every field except its own ID, including all
gate vectors, exact rational endpoints, evaluator row cardinalities,
per-version and aggregate fail-closed counts, version-bump/NO-GO state, and all
positive and negative claim booleans.  The unambiguous field
`final_validation_version_all_statistical_gates_passed` applies only to that
terminal permitted version.  The detailed gate vectors and exact rational
endpoints likewise describe that final version.  A separate per-version outcome/gate-root vector
preserves v1 failure and v2 outcome without upgrading a v2 pass into a
whole-graph pass claim.  The result object remains cooperative
and forgeable.  Downstream code must call the fresh result verifier with the
original graph; that verifier replays the graph and compares strict canonical
projections without caller-defined equality.  Before any recursive projection,
it performs an exact, bounded, non-recursive DTO shape check; recursion or memory
failure anywhere in fresh verification returns false.  Copying a `COMPLETE_PASS` result
or changing any claim field is not accepted as evidence.

Before allocating canonical snapshot bytes, the verifier enforces zero-or-four
system-run containers, bounded declared output sizes, 256-byte ASCII IDs, and
a conservative pre-canonical upper bound against the exact 20 MiB snapshot
ceiling.  The exact canonical byte length is checked again after encoding.
