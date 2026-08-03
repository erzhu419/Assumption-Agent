# Phase-3A M2.5 implementation status

Date: 2026-08-03

Machine freeze: `hegel-freeze-p2b-p3-v1.1.2`

Child DSL: `hegel-old-dsl-v1.1.0`

State: `DUAL_EXACT_WIRE_ERRATA_GOLDEN_PASS / EXTERNAL_GENESIS_NOT_EXECUTED / 14/24 / NOT_RUN`

M2.5 purpose 1–4 的当前 authority class 是
`OWNER_ACCEPTED_CONTAINER_TECHNICAL_ACTORS_V1`。它要求 purpose-separated、离线、live-probed
Docker technical actors，不要求不同真人或组织；所有 publication 必须完整披露：
`same_admin_controller=true`、`organizational_independence=false`、
`independent_human_actors=false`、`technical_role_independence=true`、
`owner_accepted_threat_model=true`、`remote_attestation=false`、
`hardware_key_nonexportability=false`。该 owner amendment 不改 formal wire，也不接管
Phase-2B sealed-holdout 的 custodian 要求。

## Outcome

E1–E12 are resolved deterministic prerequisites, not current specification
blockers. The exact-wire layer now has:

- 81 unique numeric tags/schemas with strict numeric-array CBOR construction,
  decode guards, enum registries, root preimages and state constraints;
- 21 sorted candidate object vectors, 8 sorted candidate record-tree vectors
  and 15 sorted production-validator negative vectors;
- byte-exact Python/Rust agreement and exact agreement with the checked golden
  fixture;
- a fresh qualification from detached Commit-A source rather than a live
  worktree or caller-supplied executable;
- a frozen repository secret-policy audit with zero findings.

The implementation-basis Commit A is
`d293507048d39323462e5c9033905b352ae07cb2`. Python executed the explicit
minimal exact-wire module closure from its private Git-archive snapshot and
did not execute the broad package `__init__`. Rust used a fresh empty target,
locked/offline Cargo, 16 lock-checksummed `.crate` archives that Cargo unpacked
inside the isolated environment, verified absence of visible Cargo configs,
and a committed approved-toolchain policy. The runner hashed and executed the
same open output-binary inode and verified source, registry and toolchain
stability after replay.

The resulting artifact is
[phase3_m25_errata_qualification_v1.json](../artifacts/phase3_m25_errata_qualification_v1.json):

- status: `DUAL_EXACT_WIRE_ERRATA_GOLDEN_PASS`;
- kind: `DETERMINISTIC_CANDIDATE_NON_AUTHORITATIVE`;
- golden SHA-256:
  `a0e8ce77f3655d484fdc40506f81034fd4d103b458af3ce9f586fe42cc369ae1`;
- Python/Rust compact sorted response: 20,308 bytes, SHA-256
  `9c855290ad9f9a6e3e523107e0162e6e3c363afec09224245c5e35075ad8ab4c`;
- checked artifact SHA-256:
  `a9b84c1dbe10238a57d565e2f61ec98b8cab9bd889d466f50d7c412178eb8b32`;
- external-genesis start guard: `10/10`, side-effect-free authorization only;
- formal-root, gate and state delta: none.

The repository audit examined 1,407 ancestor commits, 11 subtree path states,
476 unique blobs (574,786,726 bytes) and 76 JSON blobs, with zero findings.
Its claim is limited to the frozen filename, private-key-header and non-null
forbidden-JSON-key policy. It is not a universal proof that arbitrary bytes
cannot conceal a secret.

No formal root, real seed, real private key, marker, external signature,
custody attestation, parent-absence audit claim, M3 execution identity or M3
transition has been created.

## Earlier typed-row candidate values

The earlier v1.1.2 caller-supplied/unattested row replay remains useful as a
candidate-value regression, but it is not the current exact-wire build
qualification:

| Role | Rows | Candidate universe root | Candidate truth root |
|---|---:|---|---|
| odd | 480 | `b7e6eed1174ceee1944bd540cbb0ace4c5c7fc0dffea79dd956a51f7a2410a05` | `f5bbdc26bec62f9966e5ef31eaa800190ed52dedc73ee61545e0f9c122a1a506` |
| sink | 85 | `1a46f9967ad48df6ec1d9be609de701413ce86171fb0f92495a982bef0f40ff5` | `9c0f5d75ea3c31f6cb1ea9917346a7a3f480ae9ce0ac0cb3bb21aac9d3bd7808` |

These candidate roots exercise the repaired DAG but are not formal roots.
Formal identity still requires external custody, audit, opaque-ID persistence
and signatures.

## Gate state

| Gates | State | Reason |
|---:|---|---|
| 1–14 | satisfied | unchanged shrink-1 qualification |
| 15–24 | blocked | deterministic errata qualified; live owner-accepted technical-actor evidence absent |

`24/24` would mean ready but still `NOT_RUN`. It would not automatically start
closure; `NOT_RUN -> RUNNING/CANONICAL_ENUMERATION` remains a separate action.

## Resolved deterministic errata

The twelve byte/root/trust/state conflicts are closed by
[the exact-wire resolution](Hegel_Machine_Phase3A_M25_Exact_Wire_Errata_Resolution.md)
and
[the implementation closure addendum](Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md).
The original questions document remains historical review input, not current
normative state. Synthetic/candidate bridge wire has been dual-qualified;
externally instantiated bridge roots have not been generated.

## External actor boundary and next hard gate

A successful fresh Commit-A replay may return a side-effect-free authorization
to begin the owner-accepted technical-actor external-genesis workflow. A checked/stored JSON
artifact, its self-hash, or the earlier caller-supplied binary cannot authorize
that workflow by itself. The exact operator sequence is frozen in
[the external-genesis runbook](phase3_m25_external_genesis_operator_runbook.md), as superseded
for actor eligibility by the
[owner amendment](Hegel_Machine_Owner_Accepted_Container_Technical_Actor_Eligibility_Amendment_v1.md).

The next hard gate is live execution by the four qualified, purpose-separated container actors:
one-shot seed genesis, purpose-separated actor keys, parent-history audit, opaque-ID
persistence, formal roots and the required envelopes. That work has not been
executed in this repository. This is technical-role independence under the disclosed same-admin
threat model, not organizational or independent-human custody.

The source-level admission path is no longer permanently blocked by an
unfinished pre-stage recovery flag. Its intent/checkpoint continuation,
host-readable recovery anchor, exact UID-65534 reclaim, keyless retained-seed
verification and immutable-plan pre-seed abort matrices have passed, so a fresh
qualified execution attempt may proceed. This is only implementation
eligibility: no real seed/key/signature/formal root was produced by that test
work, and the recorded external state therefore remains `14/24 / NOT_RUN`.

An exact pre-seed abort leaves a canonical terminal tombstone derived from the
evidence path. It preserves run/ledger and Docker-absence identity across a
crash after the last lock unlink. Three deterministic, role-independent
retirement markers permanently retire the original evidence, promotion and
derived publication-receipt physical paths. A later attempt must choose a
completely fresh output triple; changing roles or combining any retired path
with fresh paths cannot bypass retirement. The tombstone and markers are
diagnostic recovery evidence, not formal gate artifacts.

## v2 counterevidence

Commit `4861b2d8` remains a protocol-valid negative for the frozen SCAR hard
selector. Its primary failure mechanism is hard structural eligibility causing
coverage collapse, with additional evidence that the extracted structural
signal is noisy. It does not affect M2.5 byte identity or bounded closure, but
it prohibits treating v2 thresholds/weights or hard eligibility as verified
positive priors in Phase-3B. See
[v2_scar_negative_impact_assessment.md](v2_scar_negative_impact_assessment.md).

## Evidence and claim boundary

Allowed claim:

> A fresh detached Commit-A Python/Rust replay reproduced 21 candidate objects,
> 8 candidate record trees and 15 negative guard codes over the 81-schema
> registry. This deterministic, non-authoritative result permits only the
> separately governed owner-accepted technical-actor genesis workflow; that workflow has not run.

Not allowed: external genesis completed, formal M2.5 completion, Gates 15–24
passed, `24/24`, M3 started, complete closure, odd outside verdict, sink
mechanism recovery, autonomous relation invention, outside/MDL certificate or
ACTIVE promotion. Formal roots remain null and the child remains `NOT_RUN`.
