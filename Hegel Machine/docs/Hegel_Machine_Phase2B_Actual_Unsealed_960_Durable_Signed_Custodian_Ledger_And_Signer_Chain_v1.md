# Hegel Machine Phase2B Actual-Unsealed-960 Package 0 Mechanics v1

## 1. Status and claim boundary

**LOCAL, OFFLINE, PRODUCER-FREE SCHEMA/VERIFIER/KAT MECHANICS ONLY.**

This artifact implements the presented-byte mechanics for authoritative-evidence
catalog package ordinal 0, `durable_signed_custodian_ledger_and_signer_chain`.
It accepts only raw bytes, exact tuples of raw bytes, and mandatory out-of-band
byte pins. It does not access a network, credentials, a live service, or an
external system. It does not generate keys, signatures, identifiers, events,
receipts, checkpoints, anchors, time, or evidence.

The strongest successful disposition is:

```text
VERIFIED_PRESENTED_PACKAGE0_MECHANICS_NOT_R10
```

The following remain false:

- package 15 and package 16 full verification;
- gate admission and R10 satisfaction;
- authority currentness and owner non-revocation;
- external key custody, durability, non-equivocation, and global anti-fork;
- execution authorization, actual-960 verification, and C1 satisfaction.

The separately named real verifier remains exactly `NOT_IMPLEMENTED`. Synthetic
KAT success cannot be promoted into operational evidence, authority, R10,
actual execution, or C1.

## 2. Frozen local artifact snapshot

| Artifact | SHA-256 | Size |
|---|---|---:|
| `src/hegel_machine/phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_v1.py` | `50788cc2ad4c3e3d204379db2789e6ba1477d8bfda94b339418fa2fa7fd310ba` | 3,011 lines / 125,458 bytes |
| `config/phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_v1.json` | `22e9ca200ed6ad4b6a4daabf706a68cd6c384d4188fe6d73df3b52b4b97de799` | 14,269 bytes, no trailing LF |
| `tests/test_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_v1.py` | `b8ff42149ad327a16f7c2b6cd363e873f8760fc9ceda1c3ad6cfa15a30895c3f` | 2,003 lines / 73,406 bytes |

The mechanics contract identifiers are:

```text
contract_id = phase2b_actual_unsealed_960_package0_mechanics_contract_v1_1922ccf326d5938ac55acc6d3d86fa4fbde3275205741b38a65dabd09f5387dd
contract_payload_sha256 = 22e9ca200ed6ad4b6a4daabf706a68cd6c384d4188fe6d73df3b52b4b97de799
```

The config bytes are exactly the module's RFC8785 `CONTRACT_PAYLOAD_V1`; they
contain no trailing line feed. This document records the inspected snapshot but
is not an input pin and cannot backfill or alter the machine contract.

## 3. Frozen dependencies and local capture

The verifier binds the committed authoritative-evidence catalog:

```text
catalog source SHA-256 = ead2e2e5a7b03a804148fbd329e67561c1367375b2c05bf4ab2cd691f20487c6
catalog config SHA-256 = 62f46cfba8a86f7ad969a7ea584c1828f0b34bb1d95571b1cc64259ac01a588a
catalog ID = phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1_a9f1b10bf7df69065f9847a69b935613ba1c368832c21ea5ac60e863961272cd
```

At import, every mutable catalog surface used by the verifier is projected back
to the exact catalog payload. This includes the signer registry, gate, stage,
package, bundle-index, ABI, result, dependency, and DAG surfaces. Descriptor
signature cardinalities are independently recomputed to 35 total and 20 owned
by package 0. Pre-import catalog mutation therefore fails closed.

The RFC8785 dependency is locally inspected at exact source SHA-256:

```text
16dcc6524a1c4a507edb9f85455c72b00fbc6c0c1b67471f03b67b2f25d32654
```

The module reads only those local dependency source bytes at definition time,
checks the SHA-256, and privately executes and captures that same byte buffer.
The path is not a caller input. This mechanism records implementation identity;
it does not claim source currentness, authority, or external attestation.

Ed25519 verification requires the declared runtime dependency
`cryptography>=41`. Absence of the backend fails closed.

## 4. Public raw ABI

The presented and real entry points share exactly 12 positional parameters:

1. catalog payload;
2. signer-registry payload;
3. package-0 envelope;
4. exact tuple of ten ledger-event payloads;
5. exact tuple of ten stage-event payloads;
6. store-if-absent receipt;
7. logical-time token;
8. governance checkpoint;
9. custodian checkpoint;
10. external anchor;
11. package-15 envelope;
12. package-16 envelope.

They also share exactly 16 mandatory keyword-only byte pins for catalog,
registry, registry-root, governance-root, custodian-root, readiness-DAG,
progression-plan, legacy-gap, statistical-validation, campaign, and attempt
identity. There are no defaults, decoded mappings, paths, private keys,
callbacks, sockets, clocks, nonces, `*args`, or `**kwargs`.

The local stricter mechanics boundary caps every exact byte parameter or tuple
item at 16 MiB, rejects bool-as-int, and freezes generic content-ID, campaign-ID,
and attempt-ID ASCII grammars. These restrictions are explicit in the contract
and are not represented as a catalog reissue.

## 5. Verified presented graph

The accepted graph is strictly backward-bound:

```text
catalog + OOB pins
  -> signer registry
  -> shallow package 15 and package 16 envelopes
  -> ten stage envelopes
  -> ten signed custodian-ledger envelopes
  -> signed-stage-event chain root and local chain ID
  -> store receipt + time token + two checkpoints + external anchor
  -> package 0 envelope
  -> private 47-field mechanics result
```

The verifier checks exact RFC8785 bytes, keysets, wire types, content IDs, raw
SHA-256 values, signature preimages, signature order, role/purpose/epoch/key
bindings, predecessor links, campaign/attempt lineage, chain root, five-object
referenced graph, and terminal package-15/package-16 atomic binding.

The exact signature accounting is:

| Scope | Count |
|---|---:|
| package-0 outer | 4 |
| package-0 ledger events | 10 |
| package-0 referenced artifacts | 6 |
| signer-registry roots | 2 |
| stage events | 10 |
| package-15 shallow outer | 1 |
| package-16 shallow outer | 2 |
| **Total** | **35** |

Stage 0--8 cannot forward-reference package 0, package 17, bundle index, any
stage event, custodian ledger event, chain, referenced artifact, or the future
package-15/package-16 terminal pair by either artifact kind or content-ID
prefix. Stage 9 must bind the exact presented package-15/package-16 kind, ID,
and raw SHA pair.

## 6. Deliberately shallow scope

This Package0-only ABI does not contain complete gate-predecessor and raw
subject graphs for all ten stages. Accordingly:

- nonterminal subject artifact payloads are not recursively verified;
- raw predecessor gate evidence is not recursively verified;
- package 15 and package 16 nested gate/result records are not verified;
- package 15 and package 16 contribute only exact shallow outer-envelope,
  lineage, content-ID, and signature mechanics;
- `package15_verified`, `package16_verified`, and every operational or elevated
  result remain exact false.

`stage_subject_binding_verified=true` means only the frozen shallow stage
envelope, ledger-to-stage projection, predecessor syntax, no-forward-reference
rules, and exact terminal package pair were verified. It does not mean that a
stage occurred in reality.

## 7. Private result and tests

The result has exactly 47 fields. Twelve presented mechanics booleans may be
true after full success; thirteen nonclaim booleans must be exact false. The
constructor, issuer registry, and verifier implementation are private. Direct
construction, forgery, copying, deep-copying, pickling, and post-issuance
mutation are rejected by the public `require_*` boundary.

The focused suite passes 16 test instances. The dependency-integrated local
suite covering RFC8785, the readiness DAG, legacy gap inventory, statistical
validation candidate mechanics, the authoritative-evidence catalog, and this
Package0 slice passes 208 test instances. The focused coverage includes:

- a deterministic complete 35-signature positive graph;
- exact 12+16 ABI and stable real `NOT_IMPLEMENTED` checks;
- 47-field private-lineage and mutation rejection;
- type, canonicality, pin, signature, predecessor, and graph failures;
- validly re-signed and fully cascaded lineage substitutions;
- package-15/package-16 early forward-reference kind-only and ID-only cases;
- 16 MiB + 1 and identifier grammar boundaries;
- pre-import catalog-schema/ABI/result mutation isolation;
- pre-import targeted RFC8785 wrapper isolation;
- post-import module-global rebind stability.
- exact no-trailing-LF config-to-contract byte equality.

The deterministic synthetic corpus fingerprint is:

```text
30e3a9a4b5ba847e07d8121d507f8650ca33dc7584bad20d760ccb43ea1c10d4
```

This fingerprint and the green KATs prove local verifier mechanics only. They
do not evidence a real producer, a live role, external custody, durable media,
fresh identifiers, an actual run, or scientific success.

## 8. Remaining hard gate

Package 0 mechanics are one local construction slice inside the frozen
18-package evidence catalog. The earliest external blocker remains R01 current,
active, non-revoked authority. Package 0 itself cannot become authoritative
until externally produced and independently supplied registry, stage, ledger,
receipt, token, checkpoint, anchor, package-15, and package-16 bytes exist and a
separately versioned real admission path verifies the complete operational
lineage.

Therefore this artifact does not advance R01--R16 from `0/16`, allocate a
campaign or attempt ID, authorize execution, run actual 720+240/496+76 cohorts,
perform inference/reveal/scoring, or satisfy C1. Those later hard gates remain
ordered exactly as frozen by the authoritative progression plan.
