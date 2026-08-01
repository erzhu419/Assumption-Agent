# Phase-3A M2.5 implementation status

Date: 2026-08-01

Machine freeze under review: `hegel-freeze-p2b-p3-v1.1.2`

Child DSL: `hegel-old-dsl-v1.1.0`
State: `EXACT_ERRATA_REQUIRED_EXTERNAL_GENESIS_BLOCKED / 14/24 / NOT_RUN`

## Outcome

The v1.1.2 deterministic wire is now implemented and replayable, but external
genesis is deliberately blocked before any CSPRNG call or marker creation.
The repository establishes the following limited result:

- 58 unique numeric object tags/schemas and the amendment's enum tables are
  represented by strict numeric-array CBOR construction and decode guards;
- Python and Rust generate the complete 480-row odd role and 85-row sink role,
  including typed input objects, row bytes, input hashes, leaf hashes, two-row
  roots, and full roots;
- both endpoints reproduce the checked-in public fixture and all four roots
  printed in the amendment;
- odd `192/96/192` and sink `39/20/26` quota allocation is implemented as a
  deterministic pure function, with typed replay deriving index/hash/stratum
  from validated role rows rather than trusting caller-supplied strata, and
  without instantiating the real split seed;
- authoritative root APIs remain fail-closed and are distinct from candidate
  root APIs;
- secret-state, FD-3, one-shot marker, actor-key separation, public-output, and
  Commit-B path policies have read-only/pure fault-injection tests.

The Rust replay executable is caller supplied and unattested. The diagnostic
report binds its binary SHA-256 and the repository source snapshot but
explicitly sets `binary_source_binding_claim=false`; those source hashes are
not a build attestation. The Rust crate is also compiled and tested from the
checked-in source during repository verification.

No formal root, real seed, real private key, external signature, custody
attestation, parent-absence claim, execution identity, or M3 transition has
been created. The gate count therefore remains exactly `14/24`.

## Reproduced candidate values

| Role | Rows | Candidate universe root | Candidate truth root |
|---|---:|---|---|
| odd | 480 | `b7e6eed1174ceee1944bd540cbb0ace4c5c7fc0dffea79dd956a51f7a2410a05` | `f5bbdc26bec62f9966e5ef31eaa800190ed52dedc73ee61545e0f9c122a1a506` |
| sink | 85 | `1a46f9967ad48df6ec1d9be609de701413ce86171fb0f92495a982bef0f40ff5` | `9c0f5d75ea3c31f6cb1ea9917346a7a3f480ae9ce0ac0cb3bb21aac9d3bd7808` |

These are deterministic candidate values. They do not become the formal
Gate-20/21 roots until the exact preimage DAG is repaired and the required
independent actors attest the resulting objects.

## Gate state

| Gates | State | Reason |
|---:|---|---|
| 1–14 | satisfied | unchanged shrink-1 qualification |
| 15–24 | blocked | exact E1–E12 errata plus independent external actors are still required |

`24/24` would mean **ready but still `NOT_RUN`**. It would not automatically
start closure; `NOT_RUN -> RUNNING/CANONICAL_ENUMERATION` remains a separate,
explicitly authorized transition.

## Exact blockers

The v1.1.2 prose calls the wire complete, but implementation-level replay
found twelve groups that still change bytes, roots, trust, or state:

1. 15 named M3 output slots versus prose saying 16;
2. bridge signatures ordered before the execution root they sign;
3. no unique 3/3 bridge statement/envelope and run-identity binding;
4. missing signed-envelope and dual-agreement hash domains;
5. actor-key trust roots and signer purposes not transitively anchored;
6. FD-3 calculators versus a genesis-only hidden-access ledger;
7. incomplete parent path/history/two-source absence-audit wire;
8. ambiguous M3RunGenesis initial-state and target-role enums;
9. incomplete root preimages, nested schemas, numeric IDs, and the repository
   path-with-space versus `IdDigestV1` syntax conflict;
10. no replayable append-only opaque-ID registry evidence;
11. sink witness assigned in prose to a nonexistent role-binding field;
12. custodian envelope coverage disagrees with the topology.

The answer-shaped, machine-field request is
[questions_for_gpt_phase3_m25_wire_completion_errata.md](questions_for_gpt_phase3_m25_wire_completion_errata.md).
The implementation does not treat its recommended options as approvals.

## External actor boundary

`assert_external_genesis_start_allowed()` has no override and currently raises
`FAIL_M25_EXACT_ERRATA_REQUIRED` before CSPRNG or marker operations. After an
exact successor amendment resolves E1–E12, the intended sequence is:

1. implement and dual-verify the repaired schemas, domains, registries, and
   root DAG;
2. commit the deterministic implementation and public vectors (Commit A);
3. let an independent custodian create the first seed/key material outside the
   repository, and an independent auditor perform the historical audit;
4. run the constrained Python/Rust calculators through the frozen secret
   transport and collect the required signatures;
5. verify and publish only allowlisted public manifests/roots (Commit B);
6. reach 24/24 while remaining `NOT_RUN`;
7. start M3 only through a separate explicit transition.

## v2 counterevidence

Commit `4861b2d8` remains a protocol-valid negative for the frozen SCAR hard
selector. Its primary failure mechanism is hard structural eligibility causing
coverage collapse, with additional evidence that the extracted structural
signal is noisy. It has no effect on M2.5 byte identity or bounded closure, but
it prohibits treating v2 thresholds/weights or hard eligibility as verified
positive priors in Phase-3B. It remains a negative control alongside
no-prior, semantic-only, frozen-v2, and Hegel-invented matched arms.

See [v2_scar_negative_impact_assessment.md](v2_scar_negative_impact_assessment.md).

## Evidence and claim boundary

- Current deterministic evidence:
  [v1.1.2 typed-row qualification](../artifacts/phase3_m25_wire_completion_qualification_v112.json)
  and [external-genesis preflight](../artifacts/phase3_m25_external_preflight_v1.json).
- Historical v1.1.1 20-vector evidence remains pinned to commit `d772b844` and
  is no longer described as current-source bound.
- The current v1.1.2 typed-row qualification and external-preflight artifacts
  are diagnostic and non-authoritative.
- `formal roots = null`, `M3 execution manifest = null`, all run-produced roots
  are null, `M3 = NOT_RUN`.

Allowed claim:

> The v1.1.2 deterministic typed-row/candidate-root implementation reproduces
> the amendment's public values, while exact errata and independent external
> genesis remain fail-closed.

Not allowed: formal M2.5 qualification, 15–24 gates passed, M3 started,
complete closure, odd outside verdict, sink mechanism recovery, autonomous
relation invention, or an outside/MDL certificate.
