# Hegel Machine Phase-3A-Q3 bounded quotient-certificate admission v1

## Position

Q1 is sealed `COMPLETE` at `20/20`. Q2 is sealed
`Q2_ROLE_MEMBERSHIP_PASS`: the disclosed Generic Odd role has zero matches
among 418 frozen quotient classes, while the null control has one match among
89 classes.

Q3 is a completion-only, producer-free replay of those already sealed Q2
bytes. It creates an unsigned bounded predecessor-admission candidate for a
later certificate-capability stage. It does not reopen target truth, read a
split, enumerate or reread a Q1 archive, or execute a membership fold.

```text
sealed Q2 capsule / authorization / consumption / terminal / artifact
-> producer-free strict JSON and CBOR replay
-> exact bounded claim and null-control projection
-> unsigned local admission candidate
-> later formal quotient-certificate capability and issuance stage
```

## Exact claim boundary

The bounded candidate carries:

```text
OUTSIDE_FROZEN_QUOTIENT_CLOSURE(
  dsl_version = hegel-old-dsl-v1.6.0,
  closure_semantics_version = hegel-quotient-closure-v1.0.1,
  input_signature = 1,
  frozen_universe_root = sha256:b7e6eed1174ceee1944bd540cbb0ace4c5c7fc0dffea79dd956a51f7a2410a05,
  equivalence = exact_behavior_vector_with_bottom,
  target_truth_root = sha256:f5bbdc26bec62f9966e5ef31eaa800190ed52dedc73ee61545e0f9c122a1a506
)
```

It also binds the null-control `IN_FROZEN_QUOTIENT_CLOSURE` result. The
Generic Odd target and its split were disclosed before Q2, so the candidate is
only `LOCAL_DISCLOSED_DEVELOPMENT_CONTROL`. It is not untouched C2 evidence,
an unbounded `OUTSIDE_LANGUAGE` claim, an externally authorized certificate,
or a formal M4 issuance.

`outside_certificate_issued`, `formal_certificate_signed`,
`external_authority`, and `synthesis_released` remain false. Q2 quotient-class
IDs are not reinterpreted as formal certificate canonical-AST program hashes.

## Producer-free input closure

The target-free preflight reads exactly five sealed Q2 files, in order:

1. capsule;
2. owner authorization;
3. consumption record;
4. terminal record;
5. Q2 membership artifact.

Their frozen total is 22,898 bytes. Q2 source rows remain transitively bound
by the sealed Q2 capsule, but Q3 does not import or replay that source closure;
doing so would pull the historical target-row producer into the Q3 process.

The Q3 runtime source closure contains exactly:

- `src/hegel_machine/phase3_q3_bounded_quotient_certificate_v1.py`;
- `src/hegel_machine/phase3_q3_completion_replay_v1.py`;
- `src/hegel_machine/strict_cbor_v1.py`;
- `tools/phase3_q3_bounded_quotient_certificate_v1.py`.

The core decoder imports only `strict_cbor_v1`. In particular it cannot import
Q2 membership/campaign/admission/archive modules, target rows, split code, Q1
archive materializers, or legacy M3 start code.

## State and authority

Engineering freeze ends at:

```text
Q3_BOUNDED_QUOTIENT_CERTIFICATE_CAPSULE_READY_NOT_AUTHORIZED
```

Capsule preparation has logical-campaign count 0 and authorization-consumption
count 0. It creates no authorization, consumption record, Q3 artifact, or
terminal.

A later exact authorization may permit one completion-only campaign:

```text
CAPSULE_READY_NOT_AUTHORIZED
-> AUTHORIZATION_RECORDED_NOT_CONSUMED
-> AUTHORIZATION_CONSUMED
-> Q3_BOUNDED_QUOTIENT_CERTIFICATE_CANDIDATE_COMPLETE
```

Every Q3 action count remains zero: internal Popen, actor, Docker, measurement,
target open, target-truth access, role evaluation, split access, Q1 archive
read, legacy M3 read, synthesis, MDL, and certificate signature. Network,
certificate issuance, external scientific claim, adoption, promotion, ACTIVE,
production, and current-pointer update remain false.

Before consumption, local path/source/codec failures may enter same-capsule
`PENDING_NOT_CONSUMED`. After consumption, no PENDING is allowed. Only an
already sealed exact Q3 artifact may be finalized into a terminal; a missing
or unverifiable artifact is terminal-no-resume.

## Downstream lock

This Q3 candidate does not satisfy the historical final-certificate signature
gate and does not itself unlock Phase-3B synthesis. A producer-independent
formal quotient-certificate body, required replay evidence, current key-status
resolution, and the frozen authority/signature policy must be implemented and
authorized in a separate future capsule. Synthesis, MDL, integration, and any
theory mutation remain separately authorized downstream stages.
