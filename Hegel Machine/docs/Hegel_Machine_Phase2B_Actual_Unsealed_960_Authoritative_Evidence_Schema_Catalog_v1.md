# Hegel Machine Phase2B Actual-Unsealed-960 Authoritative Evidence Schema Catalog v1

## 1. Status and claim boundary

**PROSPECTIVE SCHEMA/RUNBOOK DOCUMENTATION ONLY.**

This document describes the frozen machine contract for a future C1 evidence
bundle. It is not an authority receipt, readiness receipt, execution record,
operational-evidence bundle, actual-corpus result, or C1 PASS.

The frozen current state is:

- R01--R16 satisfied: `0/16`;
- catalog disposition: `RUNBOOK_FROZEN_NOT_ADMITTED`;
- all 16 gate verifier availabilities: `SCHEMA_FROZEN_NOT_VERIFIABLE`;
- real evidence-bundle verifier: `NOT_IMPLEMENTED`;
- campaign and attempt identifiers: not allocated;
- actual generation, inference, reveal, scoring, and terminal consumption: not
  executed.

Catalog verification proves only that the presented catalog bytes match the
frozen schema/runbook contract. It cannot promote schema mechanics into
evidence, current authority, operational readiness, or scientific validity.

## 2. Inspected artifact snapshot

The following identities record the local bytes inspected for this document:

| Artifact | SHA-256 | Size |
|---|---|---:|
| `src/hegel_machine/phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1.py` | `c75bfe81e5d77d5062da2babe9adc193ca2bdf6a9943aad442ea8214f3e89dff` | 9,355 lines / 403,738 bytes |
| `config/phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1.json` | `71473b38f5fd4df165884c06134f4fbef6d94bb8e307fa71703d946d3a49b0ed` | 1,527,500 bytes, no trailing LF |
| `tests/test_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1.py` | `0e32c0b5183112b7f3ef429e683b6e6999ef2703a57537f96fa50312d61202b2` | 2,291 lines / 97,684 bytes |

The catalog content identifier is:

```text
phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1_accb3c3739394cc36f2e89fc420f58799fbe71a1f33c21ee31906a3624e23082
```

This document is deliberately not an input pin of the catalog. It may explain
the frozen bytes but cannot replace them or backfill an ancestor. If prose and
machine contract ever disagree, the source/config bytes above are the
candidate-mechanics contract and the discrepancy must fail closed.

## 3. Frozen catalog inventory

The catalog freezes:

- 16 readiness-gate schemas in exact R01--R16 order;
- 16 gate-native-source receipt schemas;
- 16 gate-support schemas;
- 10 signed stage-event schemas ending only in `TERMINAL_CONSUMED`;
- 18 evidence-package schemas;
- one signer-registry schema, one preallocation-lineage schema, and one bundle
  index schema;
- 79 pairwise distinct top-level schema and artifact content-ID domain/prefix
  pairs;
- 187 pairwise distinct descriptor signature domains;
- 210 required authorization-policy rows in the signer-registry schema, not
  issued or live authorizations;
- exact raw-byte, RFC8785, content-ID, detached-signature, length-framed root,
  dependency-pin, and private-lineage rules.

Every gate-source predecessor list must equal the matching gate dependency
list, remain pairwise distinct, and refer only to an earlier gate. Self,
forward, duplicate, partial, or cross-gate substitutions are invalid. Each
gate also has its own exact NUL-terminated source-root domain:

```text
HEGEL/PHASE2B/ACTUAL960/AUTH_EVIDENCE/GATE/{1..16}/SOURCE_ROOT/V1\0
```

These are schemas and content identifiers only. No source-receipt, support,
gate-evidence, stage-event, package-instance, or bundle-instance runtime object
has been issued by this construction.

## 4. Three-layer gate boundary

The catalog makes a signed self-report insufficient at all three layers.

### 4.1 Native-source receipt

The source receipt binds descriptor-frozen native kinds, exact raw payload
SHA-256 values and byte counts, predecessor receipts, lineage, and a
gate-specific source root. Its claim is raw provenance only. Its body does not
contain `passed`, `satisfied`, `current`, `non_revoked`, or
`evidence_observed`.

The four signature slots describe a future provenance contract for
`GOVERNANCE_ANCHOR`, `CUSTODIAN`, `STORE_RECEIPT`, and `TIME_ATTESTOR`.
They do not assert that signatures, external custody, currentness, or authority
exist now.

### 4.2 Gate support

For the current catalog version, every prospective gate-support object
accepted under this schema fixes all four truth-bearing fields to false:

```text
evidence_observed = false
current = false
non_revoked = false
satisfied = false
first_failure_code = GATE_VERIFIER_NOT_IMPLEMENTED
```

Each future gate requires a separately versioned, public, bytes-only,
producer-free dispatch over the descriptor-frozen native input graph before a
true result can become representable. No such authority-complete dispatch is
implemented for R01--R16 in this version.

### 4.3 Outer gate evidence

Every outer gate descriptor fixes:

```text
current_catalog_gate_success_allowed = false
current_catalog_gate_success_failure_code = GATE_VERIFIER_NOT_IMPLEMENTED
```

Neither the source receipt nor four-role support signatures can override that
boundary.

## 5. Local verification record and its limits

The focused catalog suite passes 61 tests. The dependency-integrated suite
covering the readiness DAG, legacy gap inventory, statistical-validation
candidate, and catalog passes 164 tests.

The focused suite independently checks all 79 schema content IDs, length-framed
schema roots, exact field manifests, 187 signature domains, the 16 source-root
domains, raw ABI, OOB pins, private result lineage, and stable real-API
`NOT_IMPLEMENTED`. It also rejects source self-dependency, forward dependency,
duplicate dependency, unique-but-wrong source-root domain, binding-rule drop,
truth-field injection, support `satisfied=true`, and outer-gate success
elevation.

Those results establish local schema/static/KAT mechanics only. They do not
prove that:

- any operational evidence has been supplied or verified;
- owner authorization is current and non-revoked;
- six live roles, external key custody, durable CAS, external time,
  non-equivocation, or rollback resistance exist;
- campaign or attempt identifiers were allocated;
- actual 720+240 or 496+76 artifacts were generated or audited;
- recognizer or baseline inference ran;
- predictions were committed, answers revealed, scoring performed, or a
  terminal report consumed;
- execution is authorized or C1 is satisfied.

All corresponding result fields remain exact boolean `false`.

## 6. Next construction slice: package 0 verifier mechanics

The next local construction slice is package ordinal 0:

```text
name: durable_signed_custodian_ledger_and_signer_chain
purpose: authenticate_one_shot_append_only_custody
required gate: R10_DURABLE_CUSTODY_AND_LEDGER_VERIFIED
producer stage: PREREGISTERED
finalization stage: TERMINAL_CONSUMED
```

The slice should implement a producer-free raw-bytes verifier and matching
synthetic/negative KAT for:

1. the exact ten-stage signed ledger chain;
2. stage-subject raw-byte equality;
3. content IDs, raw payload SHA-256 values, predecessor links, and chain root;
4. store-if-absent receipt, logical-time token, governance checkpoint,
   custodian checkpoint, and external-anchor envelopes;
5. referenced-artifact graph and terminal atomic cross-binding;
6. catalog/campaign/attempt lineage;
7. four aggregate package signatures, ten ledger-event signatures, and six
   referenced-artifact signatures, for an exact total of 20 signature slots.

This next slice must not generate keys, sign records, allocate identifiers,
write a live ledger, issue receipts/checkpoints/anchors/events, or connect to a
live authority. Synthetic KAT success must not mark R10, package 0, or C1 as
passed. Until the complete raw public dispatch and externally supplied
authority/custody evidence exist, the global real bundle verifier remains
`NOT_IMPLEMENTED` and R01--R16 remain `0/16`.
