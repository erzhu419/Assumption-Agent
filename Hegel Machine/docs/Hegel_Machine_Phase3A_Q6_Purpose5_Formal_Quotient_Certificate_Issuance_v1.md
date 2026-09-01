# Phase-3A-Q6 Purpose-5 formal quotient-certificate issuance v1

## Result boundary

Q5B pinned the three Purpose-5 signer identities and accepted their epoch-1
current status.  Q6 is the first stage allowed to use the retained
repository-external private keys.  Its successful terminal is exactly:

```text
Q6_PURPOSE5_THREE_OF_THREE_FORMAL_QUOTIENT_CERTIFICATE_ISSUED
```

Success means that one bounded quotient-certificate envelope was signed by all
three current Purpose-5 signers, independently verified, published, and sealed
in one aggregate issuance receipt.  It does not unlock the legacy
`ClosureAssessment` issuance path, project-wide `ACTIVE`, synthesis, MDL,
adoption, promotion, production, a current pointer, external authority, or an
external scientific claim.

## Frozen public predecessor

Every controller entry producer-free replays the exact Q5B capsule,
authorization, consumption, successful terminal, public artifact, and
three-event trajectory.  It also replays the exact sealed Q4 capability
artifact because Q5B binds the Q4 body hash and capability root but does not
embed the complete 34-field Q4 body required by the envelope.

The Q4 replay must recover:

- capability artifact raw SHA-256
  `8c04217f9ffa8619c9a827dff7af84791318fee61432217e4efc929419d9c00d`;
- capability receipt root
  `f45a3d0d37df747683d7de5c72f279a86fd7fc4220f3a782bb33ee0d36c9f0c8`;
- quotient body hash
  `873b10cb7ac9b6eec2c00543ccc755c86e2c20c7180101842b74a2d6856c03de`.

The Q5B replay must recover:

- receipt root
  `46d8c8182720a1557995331a01480294ee98ef24e16b5fc6e9973124efe21aef`;
- pinned trust-genesis root
  `8a772a4c79ab7b6139eeaebc397d123e02248a2a307c58c558a3dc2f70f11d19`;
- current key-status root
  `0eba7a69d2c77328537bc8cd5ee1c7935dc45caedb726ac91b1c304033e67c3c`;
- current epoch 1 and exactly three active signers.

The Q4 receipt root and body hash in the Q5B lineage must equal the values
recomputed from the separate Q4 artifact.  Q6 does not reread Q1, Q2, Q3, the
Q4 ledger, the Q5A ledger, or any target, truth, split, or role input.

## Signature preimage

Q4's body-only `signature_message` is not an issuance gate.  Q6 signs one
strict numeric-array canonical-CBOR preimage under policy:

```text
hegel-q6-purpose5-pinned-status-bound-signature-preimage/1
```

The canonical preimage binds the certificate kind, one capsule-frozen creation time,
current epoch 1, the complete unsigned Q4 body and its body hash, Q4 capability
receipt root, Q5B amendment receipt root, pinned genesis root, current status
root, and the three role-ordered key IDs.  The creation time is frozen once in
the capsule; it is not redrawn on controller re-entry.  All three signer intents
bind identical preimage bytes, preimage root, and signing-message identity.  The
exact Ed25519 message is the frozen Q6 ASCII domain prefix plus NUL followed by
that 32-byte preimage root; the canonical preimage and the signing message are
published as separate public files and separately bound by the capsule.

## Signer and envelope order

Signing actions occur in role order:

1. `FINAL_CERTIFICATE_CUSTODIAN`;
2. `FINAL_CERTIFICATE_PYTHON_REPLAY_ATTESTER`;
3. `FINAL_CERTIFICATE_RUST_REPLAY_ATTESTER`.

Their role-ordered key IDs are `568e0c23e25dd11be10ac41bbf8bd6cb`,
`b43d7064dafbf4151eb054d7ff281c5c`, and
`6266c47bea6e2fc8c2aad410cf32d18a`.  The Q4 envelope requires bytewise key-ID
order, so its records are emitted in the distinct order role 1, role 3, role 2.
The Q6 verifier maps every record back to exactly one current signer and accepts
only the complete three-of-three set.

## Key-bearing worker boundary

Engineering freeze, tests, capsule preparation, authorization recording, and
startup smoke do not open a custody root or private key and do not execute
OpenSSL.  Before authorization consumption, the controller may inspect the
fixed public predecessor/source files plus process identity, bound public
parameters, and `/usr/bin/openssl` executable metadata; it does not read a
custody root, private key, or signing-message file.  Custody/private-key metadata is
admitted only after consumption and before the next signer intent, including the
contract-bound maximum private PEM size of exactly 16384 bytes, so a local
path, ownership, mode, or permission problem can enter a clean-checkpoint
`PENDING` without starting that signer.

For each signer the controller then:

1. durably publishes and reads back one intent;
2. invokes one one-shot signing worker;
3. permits that worker to pass the fixed custody PEM to one
   `/usr/bin/openssl pkeyutl -sign -rawin` process;
4. seals one 64-byte signature and one public receipt;
5. durably publishes and reads back one verification intent;
6. independently verifies the signature exactly once with the Q5B public key
   and durably checkpoints that verified signer before extending the completed
   prefix.

The controller never reads private-key content.  Q6 generates, imports,
exports, copies, replaces, or revokes no key and does not change key status.

## PENDING and no-resume boundary

Before consumption, source, public predecessor, startup-smoke, publication, or
fresh-state problems may enter same-capsule `PENDING_NOT_CONSUMED` with campaign,
consumption, private-key-load, and signature counts all zero.

Every PENDING occurrence is also appended and fsynced as one canonical-JSONL
event at the capsule-bound `pending-history.jsonl` path.  Occurrence numbers
start at 1 and remain contiguous.  Each event binds its pending digest, failure
phase/code/detail, capsule/authorization/consumption identities, and completed
prefix.  `latest.pending.json` may be cleared when work resumes, but the
append-only history is never deleted after success; both successful and
invalidated terminal outcomes bind its final raw SHA-256, byte count, and event
count.

After consumption, only clean completed prefixes `[]`, `[1]`, and `[1,2]` may
pause before the next intent.  A durable intent is one-shot.  A missing or
unverifiable post-intent signature, unknown worker outcome, or contradictory
state is terminal no-resume; the signer is never invoked again.  A valid durable
signature with no verification intent may proceed to its one verification.  A
verification intent without a verified checkpoint is an unknown verification
outcome and is terminal no-resume; verification is never repeated.

If that sealed signature and public worker receipt are both exact but a public
source/predecessor gate is temporarily unavailable before the verification
intent exists, the controller may instead record the exact inflight status
`KNOWN_SEALED_WORKER_RESULT_AWAITING_VERIFICATION`.  The PENDING object binds
the signer ordinal, signing-intent digest, worker-receipt and signature raw
hashes, and an explicitly null verification-intent digest.  Its completed
prefix remains the verified-only prefix, while its observed audit includes the
known worker/signature action.  Recovery replays the public gates and performs
the first and only verification; it never invokes that signing worker again.

The complete prefix `[1,2,3]` already means that all three exact producer-free
verifications and their durable per-signer checkpoints succeeded.  It forbids
further signing or cryptographic re-verification and may only bind the envelope
and artifact identities in the complete checkpoint.  The aggregate issuance
receipt binds that complete checkpoint's digest; the checkpoint in turn binds
all three verified-signature receipt digests.  This one digest is the
non-cryptographic continuation proof used by artifact-to-terminal recovery.
Envelope and
receipt codecs are structural and do not silently repeat cryptographic
verification.  Later deterministic finalization replays that checkpoint and
the bound raw identities instead of verifying an already settled signature
again.  It may only finish the envelope, aggregate artifact, terminal, or
trajectory suffix.
Once the artifact is sealed, only artifact-to-terminal partial finalization is
allowed.  Once the terminal is sealed, only terminal-to-trajectory suffix
reconciliation is allowed.

## Exact successful surface

One successful campaign has authorization consumption 1, logical campaign 1,
signing-intent publication 3, signing-worker Popen 3, OpenSSL sign Popen 3,
logical private-key load 3, producer-free signature verification 3, verified
checkpoint publication 3, valid signature 3, envelope publication 1, and
certificate issuance 1.  The three retained keys and accepted current status
remain unchanged.

Key generation/import/export/copy, key-status mutation, actor, Docker,
measurement, target/truth/split/role access, Q1-Q3 or archive reread, network,
synthesis, MDL, external scientific claim, external authority, adoption,
promotion, global `ACTIVE`, production, and current-pointer mutation remain zero
or false.
