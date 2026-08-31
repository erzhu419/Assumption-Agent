# Hegel Machine Phase-3A-Q4 formal quotient-certificate capability admission v1

## Position

Q3 is sealed as
`Q3_BOUNDED_QUOTIENT_CERTIFICATE_CANDIDATE_COMPLETE`.  Its artifact is an
unsigned `LOCAL_DISCLOSED_DEVELOPMENT_CONTROL` candidate.  Q4 admits a strict,
producer-free formal body codec and replay path for that exact candidate.  It
does not sign or issue a certificate.

```text
sealed Q3 capsule / authorization / consumption / terminal / candidate
-> producer-free strict replay
-> QuotientCertificateBodyV1
-> unsigned capability receipt
-> later, separately authorized key-status and issuance stage
```

The old v1.0.2 syntactic-closure certificate body is not reused.  The quotient
line changed the closure identity, completion semantics, archive identity and
claim vocabulary.  In particular, Q4 never relabels quotient-class counts as
canonical-program counts or Q1 roots as old diagnostic-formal bridge roots.

## Exact predecessor

The only runtime inputs are the five sealed Q3 files listed below.

| Input | Raw SHA-256 | Semantic identity |
|---|---|---|
| capsule | `00308146ef5219077838a6e27940f9517a65992b7094cfcb18af8f1015b31d35` | `sha256:f8459ae97648ae316a96a2d60da80e4c20bced20aeb518f400bc981f24ac7fc1` |
| authorization | `8b9aad56e982d85ae6a8cbc59f44256ac60f03e47d0d26477cb97df9d9964d08` | `sha256:a72e3a8e644b43762561472b92465c2ea7344a8faf2ae495bf5a8a26e0a239a5` |
| consumption | `275cf4e779a8b215b94d703a363406cfa0bb7ce34b9da51b99e4fb073daca759` | `sha256:ee0f1a1bd626e90ff466bb339df31783d758c22c5b426b7f8a701436ce998354` |
| terminal | `71363770e4052f0215edb925e36fdc8b79e6ea7789a61e5645e3f185367aa38b` | `sha256:bc0200feb3684c27975b6e212308c9d8b794f41b80dd7c666571e9474aab4624` |
| candidate | `659ec3d60467a6c21737c21230c6566dd1f1f1a10f89d5fcfc43dcaf8acb8ca2` | `sha256:8964461f54c8c45028d97b292e2d265b1ea00caf7b8df23a639f95b660005d8f` |

Their exact total is 19,924 bytes.  Q4 does not read Q1 archives, Q2 role
streams, target truth or a split.  The Q1 and Q2 evidence remains transitively
bound through the Q3 candidate.

## Formal body wire

`QuotientCertificateBodyV1` uses tag `0x3E00`, schema ID
`hegel-q4-formal-quotient-certificate-body/1`, and content-hash domain
`HEGEL/Q4/FORMAL_QUOTIENT_CERTIFICATE_BODY/V1`.

Its strict array is:

```text
[
  1, 0x3E00, schema_id,
  claim_kind, dsl_version, closure_semantics_version,
  outside_input_signature_id, outside_universe_root, outside_truth_root,
  equivalence_id,
  q1_completion_receipt_root, q1_closure_bundle_root,
  q1_dual_replay_agreement_root, q1_target_blind_access_ledger_root,
  q2_capsule_digest, q2_authorization_digest, q2_consumption_record_digest,
  q2_terminal_record_digest, q2_artifact_root,
  outside_role_result_root, 418, 0, rfc6962_empty_root,
  [2, null_universe_root, null_truth_root, null_role_result_root,
   89, 1, [null_matching_class_id]],
  q3_capsule_digest, q3_authorization_digest, q3_consumption_record_digest,
  q3_terminal_record_digest, q3_candidate_root,
  q3_repository_commit_id, q3_source_closure_root,
  q4_repository_commit_id, q4_source_closure_root,
  LOCAL_DISCLOSED_DEVELOPMENT_CONTROL
]
```

The claim is exactly
`OUTSIDE_FROZEN_QUOTIENT_CLOSURE` for `hegel-old-dsl-v1.6.0`, quotient
semantics `hegel-quotient-closure-v1.0.1`, and equivalence
`exact_behavior_vector_with_bottom`.  It remains a disclosed local development
control and is not an unbounded language claim or untouched C2 evidence.

The Q3 repository/source pair identifies the sealed scientific predecessor.
The separate Q4 repository/source pair identifies the exact producer-free
formal bridge that generated the body; neither pair substitutes for the other.

The structural future envelope reserves quotient certificate kind `2` and the
three-signature array shape.  Q4a exercises its decoder only with test vectors;
it does not publish an envelope or treat structural decoding as signature or
key-status verification.

## Capability receipt and missing live authority

The unsigned capability receipt uses tag `0x3E01`, schema ID
`hegel-q4-formal-quotient-certificate-capability-receipt/1`, and content-hash
domain `HEGEL/Q4/FORMAL_QUOTIENT_CERTIFICATE_CAPABILITY_RECEIPT/V1`.  It binds
the complete body and body hash, the Q3 candidate root, future actor purpose
`5`, and the exact absence of live issuance prerequisites.

The four missing live prerequisites are a pinned purpose-5 genesis trust
anchor, a current purpose-5 key-status bundle, three active distinct purpose-5
signer keys, and three valid certificate signatures.  Consequently the receipt
must encode zero active signers, zero signatures, required signature count
three, `NOT_READY_FOR_ISSUANCE`, and false values for envelope publication, signing,
issuance, external authority and synthesis release.

Purpose 5 is the existing `FINAL_CERTIFICATE_SIGNER_RESERVED_FOR_M4` purpose.
This stage does not instantiate it, read a private key, create a live key epoch,
or claim that repository-local bytes are a trust anchor.  A future issuance
capsule must bind an independently pinned genesis, replay a continuous current
status chain, resolve three distinct active keys and separately authorize the
three signing actions.

## State and authority

Engineering freeze ends at a capsule-ready, not-authorized state.  A later
exact authorization may execute one producer-free campaign and publish only:

```text
Q4_FORMAL_QUOTIENT_CERTIFICATE_CAPABILITY_ADMITTED_NOT_ISSUED
```

Pre-consumption local path, format or source failures may enter same-capsule
`PENDING_NOT_CONSUMED`.  After consumption there is no PENDING; only an exact
sealed capability artifact may be finalized into a terminal.

Every Popen, actor, Docker, measurement, target open, role evaluation, split
access, Q1 archive read, private-key access, signature and issuance count stays
zero.  Network, external scientific claim, C2 PASS, synthesis, MDL, adoption,
promotion, ACTIVE, production and current-pointer update remain false.

## Downstream lock

Capability admission is not certificate issuance and does not release
synthesis.  Issuance requires a new capsule and new authorization after the
live purpose-5 trust and status evidence exists.  Synthesis/MDL remains a
further independently authorized stage after any valid bounded local
certificate disposition.
