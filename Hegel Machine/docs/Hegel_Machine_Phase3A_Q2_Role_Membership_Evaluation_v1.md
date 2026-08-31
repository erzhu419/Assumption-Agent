# Hegel Machine Phase-3A-Q2 role membership evaluation v1

## Position and current state

Phase-3A-Q1 is sealed as `Q1_COMPLETE_20_OF_20`: Q1 is `COMPLETE`, its gate
count is `20/20`, its mask is `1048575`, and all eight ordered output roots
are non-null.  The completion-only v8c campaign did not start Q2, open target
truth, read a split, evaluate a role, issue a certificate, or mutate the v7r1
actor outputs.

This document freezes the next transition:

```text
sealed Q1 quotient closure
-> exact OUTSIDE_TARGET membership
-> exact IN_LANGUAGE_NULL control membership
-> bounded Q2 result
-> later certificate or synthesis stage
```

The preregistration state is `Q2_SPEC_FROZEN_NOT_AUTHORIZED`; the actual
controller and contract are `IMPLEMENTED_NOT_EXECUTED`.  Preflight and
authorization recording are target-blind.  No Q2 actual campaign has been
authorized or executed by this freeze.

## Exact Q1 predecessor

Q2 admits only the v8c completion selected in
`config/phase3_q2_role_membership_v1.json`:

- capsule digest
  `sha256:e6eafec73cb97b8048fc6322c21466b9d2f17e8f6f8f6f92d586d00bc361e0c0`;
- terminal record digest
  `sha256:2163d03e5b295b3bb93418e3b05e7ae216254b264272ab682a26f9d70745ba9b`;
- completion artifact raw digest
  `sha256:623bdfa6ce60b244d3dff9d0ed274dda2ef52048b3992f7dae07ce8983c3ce18`;
- completion receipt root
  `sha256:c75d08e038dc17c01aa4c9c6b69a2d4ef95034901a55e7982fb6a93d78f7798a`.

The admission replays the terminal record digest, decodes the strict Q1
completion receipt, derives slot 8 through the receipt's self-content-root
policy, and compares all eight ordered slots.  It does not replay the v7r1
actors, Q1 archive trees, or Q1 resource measurements.  Those facts are
already sealed by v8c.

| Slot | Meaning | Root |
|---:|---|---|
| 1 | Odd signature archive manifest | `sha256:496582bc6685de4323e372247bc23dc1a89999c80c1e1ba54ef6f4febb2a9200` |
| 2 | Odd saturation state | `sha256:e5f901ca83f90b80c8c1a8f1a8836947a18ed50c4c5bee47c0db69f2a5030a6a` |
| 3 | Sink signature archive manifest | `sha256:a774cfedd950e648adee8a7f42de4bab1d1a90da700586cf9bf53fafa4f672bb` |
| 4 | Sink saturation state | `sha256:a82cd0197c9bed90af04974ff3348c35058af7d08c4855415ef8f1bd9ded9026` |
| 5 | Q1 closure bundle | `sha256:4c011a8b75cca0c52994340475be5cf3b29c5079a57f1b372d5fa59d22b1fe51` |
| 6 | Q1 dual replay agreement | `sha256:2a94390022a2df730b5791cc139a576811da64365b456a1541ab67c439938cd0` |
| 7 | Q1 target-blind access ledger | `sha256:d0c96cfb6eb877a9b75367bc061a8721da1c167d6fe34cc33879023ed4cdec58` |
| 8 | Q1 completion receipt | `sha256:c75d08e038dc17c01aa4c9c6b69a2d4ef95034901a55e7982fb6a93d78f7798a` |

## Frozen role contract

The only matcher is `BOOL_BIT_EXACT_PREDICATE_MATCH_V1`.  A class matches a
role only when every row matches.  A defined `BOOL` cell compares by its exact
boolean value, a defined `BIT` cell compares by its exact uint `0/1` value,
and bottom never matches.  This adapter does not merge Bool and Bit quotient
classes and does not alter a Q1 class ID.

| Order | Role | Rows | Universe root | Truth root | PASS condition |
|---:|---|---:|---|---|---|
| 1 | `OUTSIDE_TARGET` | 480 | `sha256:b7e6eed1174ceee1944bd540cbb0ace4c5c7fc0dffea79dd956a51f7a2410a05` | `sha256:f5bbdc26bec62f9966e5ef31eaa800190ed52dedc73ee61545e0f9c122a1a506` | zero matching Q1 classes |
| 2 | `IN_LANGUAGE_NULL` | 85 | `sha256:1a46f9967ad48df6ec1d9be609de701413ce86171fb0f92495a982bef0f40ff5` | `sha256:9c0f5d75ea3c31f6cb1ea9917346a7a3f480ae9ce0ac0cb3bb21aac9d3bd7808` | at least one matching Q1 class |

Q2 does not need the historical split to evaluate full-table membership.
`split_accessed` therefore remains false throughout this stage.

## Result and terminal vocabulary

Each role result binds the role, input signature, universe and truth roots,
the corresponding Q1 manifest and saturation roots, the ordered matching
class IDs, evaluated class count, match count, and one of:

- `IN_FROZEN_QUOTIENT_CLOSURE`;
- `OUTSIDE_FROZEN_QUOTIENT_CLOSURE`.

The complete campaign statuses are:

- `Q2_ROLE_MEMBERSHIP_PASS`: Odd match count is zero and Sink match count is
  greater than zero;
- `Q2_ROLE_MEMBERSHIP_COMPLETE_FAIL`: the evaluation completed exactly but
  the preregistered scientific disposition did not hold;
- `Q2_ROLE_MEMBERSHIP_INVALIDATED_TERMINAL_NO_RESUME`: a consumed campaign has
  unknown or unverifiable output.

A complete fail is a scientific result, not an infrastructure retry.  Q2 only
publishes bounded membership.  It does not itself issue an outside
certificate, adopt or promote a relation, enter `ACTIVE` or production, or
update a current pointer.

The single CBOR result artifact also binds the exact capsule, authorization,
consumption, Q1 slots 5--8, per-role target-open/evaluation counts, and observed
wall/RSS audit.  A terminal accepts it only when the role results also bind Q1
slots 1--4 and the capsule-bound class archive roots with exact 418/89 class
counts.

## Reader and representation boundary

The Q1 class streams are the only archive streams required for Q2 membership:
the actual Odd stream has 418 class records in 852,581 bytes and the Sink
stream has 89 class records in 55,084 bytes.  They can be read in place from
the sealed Q1 output; no copy, hard link, external sort, or large scratch area
is needed.

The target-blind actual reader replay took 45.65 seconds and peaked at
71,274,496 bytes RSS.  Its eight selected files total 919,757 bytes, including
907,665 framed class-stream bytes.  The frozen local envelope is therefore
deliberately small: at most 1,048,576 selected-reader bytes, 565 role rows,
1,048,576 result bytes, 268,435,456 bytes peak RSS, 180 seconds wall time, and
4,194,304 bytes fresh persistent storage.  Internal `Popen`, actor, and Docker
counts remain zero.  This replaces any temptation to carry Q1's multi-GiB
enumeration envelope into Q2.

The Q1 archive stores `Q1BehaviorCellV1`, while the already-qualified matcher
accepts the Q0 `BehaviorCellV1` Python type.  The Q2 implementation must use a
narrow canonical-cell bridge: replay the Q1 cell, construct the corresponding
Q0 cell, require identical canonical cell objects under the same sort, and
then call the frozen matcher.  It must not create a Q0 behavior identity or
change the Q1 class identity.

## Pending and authorization boundary

Before authorization consumption, missing local files, unreadable sealed Q1
inputs, or a failed target-blind reader smoke check may produce
`Q2_ROLE_MEMBERSHIP_PENDING_NOT_CONSUMED` under the same future capsule digest.
Such a state does not open target truth, run the matcher, consume authorization,
or count a logical campaign.  The cooperating operator may correct the path,
format, or local resource problem and re-enter at that point.

Consumption is the sole measurement intent.  It is durably published and read
back before the first target generator.  After consumption there is no role or
target retry: a complete sealed artifact permits artifact-to-terminal-only
finalization; an absent or unverifiable artifact produces an invalidated
terminal no-resume.  The fsynced trajectory records each target-open and role
evaluation intent/return so an invalidated terminal can retain the per-role
conservative action denominator and exact failure phase.

Opening either truth table and executing the membership fold requires a new,
exact Q2 capsule and separate owner authorization.  Network access remains
forbidden.  No Q1 actor, Q1 measurement, or v7r1 work product may be rerun or
copied as a Q2 actor product.

## Claim limitation

The Generic Odd target, its 480-row universe, and its split were disclosed
before this Q2 stage.  A future PASS is therefore a bounded development,
regression, and false-invention-control result.  It cannot be reported as an
untouched or independently hidden C2 success and cannot establish an
unbounded `OUTSIDE_LANGUAGE` claim.
