# Phase-3A-Q5B Purpose-5 trust amendment v1

## Result boundary

Q5A produced three fresh Purpose-5 Ed25519 public signer identities and sealed
their repository-external private-key custody.  Its genesis and key-status
objects were explicitly candidates: no root was pinned, no signer was current,
and no certificate was signed.

Q5B is the separate public amendment that may pin those exact three identities
and accept their epoch-1 status.  Its success state is exactly:

```text
Q5B_PURPOSE5_TRUST_AMENDMENT_ADMITTED_CURRENT_NOT_SIGNED_NOT_ISSUED
```

After success, the Purpose-5 status has three active signers.  This is not the
project-wide `ACTIVE` transition: global activation, adoption, promotion,
production and current-pointer mutation remain false.

## Exact Q5A predecessor

The controller replays five Q5A authority objects and one audit object:

1. capsule;
2. owner authorization;
3. authorization consumption;
4. successful terminal;
5. public candidate artifact;
6. the seven-event trajectory.

The capsule, authorization, consumption and terminal semantic digests are
recomputed under the Q5A domains.  Raw identities are also bound by the frozen
Q5B preregistration.  The Q5A capsule's embedded Q4 predecessor receipt is
cross-checked against the Q4 lineage inside the candidate artifact.  Q4
predecessor identities and Q5A outer authority identities remain separate
levels; neither is relabelled as the other.

The trajectory must replay the exact durable sequence from capsule preparation
through the three one-shot public receipts and the Q5A success terminal.  Its
final action audit remains worker 3, OpenSSL 6, key generation 3, public
derivation 3, public receipt 3 and signature 0.

Q5B never opens a Q5A custody root or private key.  The six selected inputs are
all public artifacts.

## Purpose-5 public wire

Q5B introduces three strict numeric-array canonical-CBOR objects:

- `PinnedTrustGenesisV1` binds the Q5A candidate root, the Q5A unpinned genesis
  root, all three ordered signer manifests and all three key IDs;
- `CurrentKeyStatusBundleV1` binds the pinned genesis, current epoch 1, the
  three active signer manifests, an empty revoked set and an empty prior-status
  chain;
- `Q5bPurpose5TrustAmendmentReceiptV1` binds the complete Q5A outer authority,
  Q5A artifact identity, Q5A/Q5B source lineages and both new Q5B roots.

The three roles remain, in order:

1. `FINAL_CERTIFICATE_CUSTODIAN`;
2. `FINAL_CERTIFICATE_PYTHON_REPLAY_ATTESTER`;
3. `FINAL_CERTIFICATE_RUST_REPLAY_ATTESTER`.

Future key-status transitions require 2-of-3 eligible current signers.  That
transition threshold is distinct from Q6 certificate issuance, which requires
all three Purpose-5 signatures.

The Q6 signature preimage policy remains
`hegel-q6-purpose5-pinned-status-bound-signature-preimage/1`.  Q4's body-only
signature message is not an issuance gate.

## M25 separation

The existing M25 trust-genesis wire freezes purposes 1 through 4 and rejects a
different purpose set.  Its workers and replacement policy likewise do not
admit Purpose 5.  Q5B therefore uses a new Purpose-5 wire; it does not reuse an
M25 key, M25 trust object, repository test key or legacy
`phase3_certificate_v1` object.

## State and PENDING

Capsule preparation and authorization recording do not consume authorization
or perform the amendment.  Before consumption, the controller replays the live
source closure and all six sealed Q5A inputs on every entry, including entries
where the capsule or authorization already exists.  A public path, format or
source problem may therefore remain same-capsule `PENDING` with campaign and
consumption counts both zero.

After consumption, `PENDING` is forbidden.  A valid sealed Q5B artifact may be
used only to finish its terminal.  A missing or unverifiable artifact, an
unverifiable consumption state or any inconsistent terminal is invalidated and
cannot be resumed.

The Q5B trajectory is reconciled from durable state rather than blindly
appended.  Its normal sequence is capsule prepared, authorization recorded and
successful amendment.  An invalidated terminal replaces the final success row.
If the terminal is durable but its trajectory suffix is missing, only that
exact suffix is restored; the amendment is not executed again.

## Explicit zero surface

Q5B has no internal process or cryptographic-key action.  Popen, worker,
OpenSSL, actor, Docker, measurement, target/truth/split/role access, Q1-Q4 or
archive reread, custody-root access, private-key access, key generation/import,
signature, certificate envelope publication, certificate issuance, network,
synthesis, MDL, external scientific claim, adoption, promotion, global
`ACTIVE`, production and current-pointer mutation are all zero or false.

Q5B remains `LOCAL_DISCLOSED_DEVELOPMENT_CONTROL`.  Q6 signing and issuance
require a new capsule and separate owner authorization.
