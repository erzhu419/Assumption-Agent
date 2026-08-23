# Hegel Machine Phase2B Actual-Unsealed-960 R01 Presented Mechanics v1

## 1. Status and claim boundary

**LOCAL, OFFLINE, PRODUCER-FREE PRESENTED-MECHANICS ONLY.**

This downstream slice implements a separate versioned byte contract for the
catalog dispatch name:

```text
VERIFY_R01_REAL_CURRENT_CUT_ACTIVE_NONREVOKED_V1
```

It does not register that dispatch in the authoritative-evidence catalog, does
not backfill a verifier source SHA-256 into the catalog, and does not change the
catalog's admission state. It accepts only caller-supplied exact bytes and
performs deterministic local computation. It does not access a network,
credentials, a live authority, a producer, a clock, a key store, a receipt
service, durable storage, or any other external system. It does not generate,
sign, issue, persist, fetch, or refresh evidence.

The strongest successful disposition is:

```text
VERIFIED_PRESENTED_R01_MECHANICS_NOT_R01
```

with reason:

```text
EXACT_24_SOURCE_AND_FULL_RAW_LEAF_GRAPH_VERIFIED
```

Success establishes only that one fully presented synthetic byte graph obeys
this downstream contract. It does not establish external currentness,
authority, owner non-revocation, gate admission, R01, readiness, execution
authorization, an actual-960 run, or C1. The separately named real entry point
remains exactly `NOT_IMPLEMENTED` and never issues a result.

## 2. Base and frozen-ancestor boundary

Construction began from clean local HEAD:

```text
95fb08e1fe97c7d268f97f5b5967d3842f3e9c3a
```

The authoritative-evidence catalog v1 and Package 0 ancestors are not modified
or reissued by this slice. Their inspected byte identities remain:

| Frozen ancestor artifact | SHA-256 |
|---|---|
| `src/hegel_machine/phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1.py` | `ead2e2e5a7b03a804148fbd329e67561c1367375b2c05bf4ab2cd691f20487c6` |
| `config/phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1.json` | `62f46cfba8a86f7ad969a7ea584c1828f0b34bb1d95571b1cc64259ac01a588a` |
| `tests/test_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1.py` | `3f37bef4defd4acd5d1b0e94d89486bbc577186eb358575122d42b16a22f5bb7` |
| `docs/Hegel_Machine_Phase2B_Actual_Unsealed_960_Authoritative_Evidence_Schema_Catalog_v1.md` | `a243825d5f3816f62bf609bea488d773ebc91db51ca451bc7e958b125bbbcbf1` |
| `src/hegel_machine/phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_v1.py` | `50788cc2ad4c3e3d204379db2789e6ba1477d8bfda94b339418fa2fa7fd310ba` |
| `config/phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_v1.json` | `22e9ca200ed6ad4b6a4daabf706a68cd6c384d4188fe6d73df3b52b4b97de799` |
| `tests/test_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_v1.py` | `b8ff42149ad327a16f7c2b6cd363e873f8760fc9ceda1c3ad6cfa15a30895c3f` |
| `docs/Hegel_Machine_Phase2B_Actual_Unsealed_960_Durable_Signed_Custodian_Ledger_And_Signer_Chain_v1.md` | `d3b411fd48079dae5e9c116c87808fb2fefb5f05c44888db8a5ded4b6b8f29a2` |

In particular, the catalog's R01 descriptor still says:

```text
gate_ordinal = 1
gate_id = R01_AUTHORIZATION_VERIFIED
verifier_dispatch_id = VERIFY_R01_REAL_CURRENT_CUT_ACTIVE_NONREVOKED_V1
verification_availability = SCHEMA_FROZEN_NOT_VERIFIABLE
verifier_source_sha256_or_none = null
allowed_native_source_cardinalities_in_exact_order = [24]
```

The catalog remains `RUNBOOK_FROZEN_NOT_ADMITTED`, all readiness gates remain
R01--R16 `0/16`, and current catalog gate success remains disallowed. A
downstream mechanics implementation is not a catalog mutation, dispatch
registration, gate-support receipt, or admission result.

## 3. Versioned downstream artifacts

The new machine contract schema is:

```text
hegel-machine-phase2b-actual-unsealed-960-verify-r01-real-current-cut-active-nonrevoked-contract/1
```

Its final content identifier and canonical contract-payload SHA-256 are:

```text
contract_id = phase2b_actual_unsealed_960_verify_r01_presented_contract_v1_a0d14b4bfdb485785ca87c51b146e7c3149e5c9bd22fbd60f8c1ab2d447bd190
contract_payload_sha256 = 7db897dafc652843cc331137c2218e2bccb29be941bc2fa3a1248bacf50ba67a
```

The contract status and claim level are respectively:

```text
DOWNSTREAM_PRESENTED_MECHANICS_IMPLEMENTED_REAL_NOT_IMPLEMENTED
LOCAL_OFFLINE_PRODUCER_FREE_PRESENTED_MECHANICS_ONLY_NOT_CURRENTNESS_NOT_AUTHORITY_NOT_OWNER_NONREVOCATION_NOT_R01_NOT_READINESS_NOT_EXECUTION_NOT_ACTUAL960_NOT_C1
```

The final source/config/KAT identities are:

| New artifact | Final SHA-256 | Final size |
|---|---|---:|
| `src/hegel_machine/phase2b_actual_unsealed_960_verify_r01_real_current_cut_active_nonrevoked_v1.py` | `59bf7b4b0e37c25fd97acfab1e112e31a0906f2518040bd52a0d9dc610873f34` | `65505` |
| `config/phase2b_actual_unsealed_960_verify_r01_real_current_cut_active_nonrevoked_v1.json` | `7db897dafc652843cc331137c2218e2bccb29be941bc2fa3a1248bacf50ba67a` | `19139` |
| `tests/test_phase2b_actual_unsealed_960_verify_r01_real_current_cut_active_nonrevoked_v1.py` | `bb51405d59e7c17b363a9c8a567f22423a7e33ac2e4d135eba467d8a608720d4` | `55361` |

The config bytes must be exactly the module's RFC8785 `CONTRACT_PAYLOAD_V1`
with no trailing line feed. This document describes the contract but is not an
input pin and cannot replace the source or canonical config. Its own digest is
not embedded here because that would be self-referential; it belongs in the
post-commit handoff record.

## 4. Public exact-bytes ABI

The presented verifier is:

```python
verify_phase2b_actual_unsealed_960_r01_real_current_cut_active_nonrevoked_presented_mechanics_v1(
    catalog_payload,
    raw_native_source_payloads_in_exact_order,
    raw_composite_leaf_payload_tuples_in_exact_native_source_order,
    *,
    out_of_band_expected_catalog_id_bytes,
    out_of_band_expected_catalog_payload_sha256_digest,
)
```

The positional inputs are exactly:

1. `catalog_payload`: exact built-in `bytes`;
2. `raw_native_source_payloads_in_exact_order`: an exact built-in tuple of
   exactly 24 exact built-in `bytes` objects;
3. `raw_composite_leaf_payload_tuples_in_exact_native_source_order`: an exact
   built-in outer tuple of exactly 24 exact built-in tuple rows, containing
   exact built-in `bytes` leaves.

Both keyword-only inputs are mandatory exact built-in `bytes`:

1. `out_of_band_expected_catalog_id_bytes` must equal the catalog ID encoded
   as ASCII;
2. `out_of_band_expected_catalog_payload_sha256_digest` must equal the raw
   32-byte SHA-256 digest of the catalog payload.

There are no defaults, decoded mappings, paths, keys, callbacks, sockets,
clocks, nonces, `*args`, or `**kwargs`. Every byte parameter or tuple item must
be nonempty, at most 16,777,216 bytes, and must not end in LF. Structured
payloads must be exact RFC8785 encodings whose canonical round trip is
byte-identical.

The real entry point has the same ABI:

```python
verify_real_phase2b_actual_unsealed_960_r01_current_cut_active_nonrevoked_v1(...)
```

It unconditionally raises
`FAIL_R01_REAL_CURRENT_CUT_ACTIVE_NONREVOKED_V1_NOT_IMPLEMENTED`. The stated
missing evidence is external currentness, authority, custody, durability, and
non-revocation evidence.

## 5. Exact catalog R01 native-source graph

The verifier consumes the catalog R01 order exactly once, using zero-based
ordinals. It checks every predecessor ordinal, predecessor content ID, and
predecessor raw-payload SHA-256 in the listed order; reordering or substituting
an otherwise well-formed source fails closed.

| Ordinal | Native source kind | Exact predecessor ordinals | Raw composite row |
|---:|---|---|---|
| 0 | `SIGNED_OWNER_AUTHORIZATION` | `[]` | empty |
| 1 | `STAGE_B_TRUST_LOCK` | `[0]` | empty |
| 2 | `AUTHORITY_PROFILE` | `[1]` | empty |
| 3 | `SIGNER_REGISTRY` | `[2]` | empty |
| 4 | `OWNER_STATUS_LEDGER_BUNDLE` | `[0,1,2,3]` | six owner-status leaves |
| 5 | `CURRENT_CUT_RESPONSE` | `[4]` | empty |
| 6 | `CURRENT_CUT_STATEMENT` | `[5]` | empty |
| 7 | `CURRENT_CUT_LOGICAL_TIME_CHAIN` | `[5,6]` | five logical-time leaves |
| 8 | `EXTERNAL_WITNESS_POLICY` | `[3]` | empty |
| 9 | `SIGNED_ANTI_FORK_ANCHOR` | `[6,7,8]` | empty |
| 10 | `EXTERNAL_ANCHOR_RECEIPT` | `[9]` | empty |
| 11 | `LIVE_RUNTIME_CLOSURE_REPORT_0` | `[2,3]` | empty |
| 12 | `LIVE_RUNTIME_CLOSURE_REPORT_1` | `[2,3]` | empty |
| 13 | `LIVE_RUNTIME_CLOSURE_REPORT_2` | `[2,3]` | empty |
| 14 | `LIVE_RUNTIME_CLOSURE_REPORT_3` | `[2,3]` | empty |
| 15 | `LIVE_RUNTIME_CLOSURE_REPORT_4` | `[2,3]` | empty |
| 16 | `LIVE_RUNTIME_CLOSURE_BUNDLE` | `[11,12,13,14,15]` | five report payloads, byte-equal to sources 11--15 |
| 17 | `SERVICE_KEY_CUSTODY_ATTESTATION_0` | `[3,11]` | empty |
| 18 | `SERVICE_KEY_CUSTODY_ATTESTATION_1` | `[3,12]` | empty |
| 19 | `SERVICE_KEY_CUSTODY_ATTESTATION_2` | `[3,13]` | empty |
| 20 | `SERVICE_KEY_CUSTODY_ATTESTATION_3` | `[3,14]` | empty |
| 21 | `SERVICE_KEY_CUSTODY_ATTESTATION_4` | `[3,15]` | empty |
| 22 | `CAS_COMPLETE_SESSION` | `[4,7,16,17,18,19,20,21]` | 176 CAS leaves |
| 23 | `DUAL_SIGNED_R01_RECEIPT` | `[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]` | one raw receipt-body leaf |

Every native source is an exact envelope with fields
`schema_version`, `body`, and `native_source_id`. Its body has exactly, in
contract order:

```text
schema_version
status
claim_level
catalog_id
catalog_payload_sha256
gate_ordinal
gate_id
verifier_dispatch_id
native_source_ordinal
native_source_kind
predecessor_ordinals_in_exact_order
predecessor_source_ids_in_exact_order
predecessor_source_payload_sha256s_in_exact_order
presented_state_profile
composite_leaf_manifest_or_none
nonclaims
```

An atomic source must have `composite_leaf_manifest_or_none = null` and an
exact empty leaf tuple at its row in the 24-row leaf matrix.

## 6. Full raw composite expansion

The only composite source ordinals are exactly `[4,7,16,22,23]`. A manifest,
content ID, SHA-256, byte count, or root without every required raw leaf is not
accepted. The exact composite leaf total is 193.

### 6.1 Owner-status ledger bundle, source 4

The exact six-leaf order is:

```text
OWNER_STATUS_GENESIS_HEAD
OWNER_STATUS_ACTIVE_DECISION
OWNER_STATUS_ACTIVE_SIGNED_EVENT
OWNER_STATUS_ACTIVE_CHECKPOINT
OWNER_STATUS_ACTIVE_HEAD
OWNER_STATUS_ACTIVE_PROJECTION
```

All six caller-supplied raw leaves must be present and independently bound into
the source-4 manifest.

### 6.2 Current-cut logical-time chain, source 7

The exact five-leaf order is:

```text
CURRENT_CUT_LOGICAL_TIME_GENESIS_HEAD
CURRENT_CUT_LOGICAL_TIME_REQUEST
CURRENT_CUT_LOGICAL_TIME_SIGNED_TOKEN
CURRENT_CUT_LOGICAL_TIME_CHECKPOINT
CURRENT_CUT_LOGICAL_TIME_CURRENT_HEAD
```

All five caller-supplied raw leaves must be present and independently bound
into the source-7 manifest.

### 6.3 Five-report runtime closure bundle, source 16

The exact five-leaf order is:

```text
LIVE_RUNTIME_CLOSURE_REPORT_0
LIVE_RUNTIME_CLOSURE_REPORT_1
LIVE_RUNTIME_CLOSURE_REPORT_2
LIVE_RUNTIME_CLOSURE_REPORT_3
LIVE_RUNTIME_CLOSURE_REPORT_4
```

Each leaf must be byte-for-byte equal to the complete raw native-source
payload at ordinal 11, 12, 13, 14, or 15 respectively. The manifest therefore
reuses each report source's verified content ID, raw-payload SHA-256, and byte
count. A re-encoded equivalent object is not equal and is rejected.

### 6.4 Complete main CAS session, source 22

The main CAS object count is exactly 29 and the full raw leaf order is:

```text
CAS_RETENTION_PLAN                                      # 1
CAS_REQUEST_000 ... CAS_REQUEST_028                     # 29
CAS_OBJECT_000 ... CAS_OBJECT_028                       # 29
CAS_SIGNED_RECEIPT_000 ... CAS_SIGNED_RECEIPT_028       # 29
CAS_CHECKPOINT_000 ... CAS_CHECKPOINT_028               # 29
CAS_HEAD_000 ... CAS_HEAD_029                           # 30
CAS_RESPONSE_000 ... CAS_RESPONSE_028                   # 29
```

The total is `1 + 29 + 29 + 29 + 29 + 30 + 29 = 176` raw leaves. Every ordinal
in every range is required; an object count, manifest, ID list, SHA list, byte
count list, or root alone cannot stand in for any raw leaf.

### 6.5 Dual-signed R01 receipt, source 23

The exact one-leaf order is:

```text
R01_RECEIPT_BODY
```

The caller must supply the raw receipt-body leaf payload and bind it into the
source-23 manifest. An opaque dual-receipt ID or SHA-256 without that raw leaf
is rejected.

### 6.6 Leaf and manifest mechanics

Except for the byte-equal runtime-report reuse described above, every raw leaf
is an exact `raw-composite-leaf/1` envelope with fields
`schema_version`, `body`, and `raw_leaf_id`. Its body has exactly:

```text
schema_version
status
claim_level
parent_source_ordinal
parent_source_kind
leaf_ordinal
leaf_kind
payload_label
nonclaims
```

`payload_label` must be byte-for-byte the exact `leaf_kind` literal; an
arbitrary descriptive label is not accepted.

Each composite manifest has exactly:

```text
schema_version
profile_id
leaf_kinds_in_exact_order
leaf_ids_in_exact_order
leaf_payload_sha256s_in_exact_order
leaf_payload_byte_counts_in_exact_order
leaf_root_sha256
```

The profile ID is exactly `R01_SOURCE_{ordinal:02d}_FULL_RAW_LEAVES_V1`.
Native-source and raw-leaf content IDs are domain-separated SHA-256 values over
the RFC8785 body. Composite roots bind exact leaf kinds, IDs, raw-payload
SHA-256 values, and byte counts. The final source-graph root binds all 24
source kinds, source IDs, raw source SHA-256 values, and the five composite
roots. No opaque ID/SHA-only composite path exists.

These envelope names and state-profile literals are presented descriptors.
This verifier does not treat the words `SIGNED`, `ACTIVE`, `CURRENT`,
`CUSTODY`, or `RECEIPT` as proof that an external producer performed those
acts. In particular,
`presented_active_zero_revocation_profile_literal_mechanics_verified = true`
means only that the required presented profile literals and their mechanical
bindings matched. It does not make external currentness, owner non-revocation,
or authority true.

## 7. Exact result and permanent nonclaims

Successful presented verification privately issues an exact frozen 38-field
`VerifiedPhase2BActualUnsealed960R01PresentedMechanicsV1` result. Its 17
identity/count/root fields are followed by twelve mechanics booleans that must
all be exact `true`:

```text
exact_raw_transport_verified
catalog_r01_descriptor_verified
rfc8785_envelopes_verified
source_content_ids_verified
native_source_order_verified
source_predecessor_graph_verified
raw_payload_sha256s_and_byte_counts_verified
composite_manifests_verified
full_raw_composite_leaf_expansion_verified
presented_active_zero_revocation_profile_literal_mechanics_verified
permanent_nonclaims_verified
private_lineage_verified
```

The following nine nonclaim booleans occur in every native source, every
non-runtime raw leaf, and the issued result, and must all be exact `false`:

```text
external_currentness_verified = false
owner_non_revocation_verified = false
authority_verified = false
gate_admission_verified = false
r01_satisfied = false
readiness_verified = false
execution_authorized = false
actual_960_verified = false
c1_satisfied = false
```

The result class uses frozen slots with public construction disabled. The
issuing closure retains an identity-and-snapshot registry. The public
`require_verified_phase2b_actual_unsealed_960_r01_presented_mechanics_v1`
boundary requires the exact result class, issued object identity, unchanged
field snapshot, all twelve mechanics booleans true, and all nine nonclaims
false. Direct construction, forged instances, copying, deep-copying,
serialization, and post-issuance mutation fail closed.

## 8. Stable fail-closed semantics

All verifier failures use
`Phase2BActualUnsealed960R01PresentedMechanicsErrorV1`, a `ValueError` subtype
with stable `code` and `detail` attributes. The stable code families are:

| Code | Boundary |
|---|---|
| `FAIL_R01_PRESENTED_MECHANICS_V1_TYPE` | exact built-in byte/tuple types, nonempty input, and 16 MiB bound |
| `FAIL_R01_PRESENTED_MECHANICS_V1_CATALOG` | exact catalog bytes and two independent out-of-band pins |
| `FAIL_R01_PRESENTED_MECHANICS_V1_CANONICAL` | exact RFC8785 bytes, no trailing LF, and byte-identical round trip |
| `FAIL_R01_PRESENTED_MECHANICS_V1_SCHEMA` | exact keysets, schema versions, text, arrays, and safe integers |
| `FAIL_R01_PRESENTED_MECHANICS_V1_CONTENT_ID` | domain-separated native-source or leaf content ID |
| `FAIL_R01_PRESENTED_MECHANICS_V1_BINDING` | catalog, ordinal, kind, profile, predecessor ID/SHA, or graph binding |
| `FAIL_R01_PRESENTED_MECHANICS_V1_COMPOSITE` | manifest, cardinality, raw-leaf equality, byte count, SHA, or root |
| `FAIL_R01_PRESENTED_MECHANICS_V1_CLAIM_BOUNDARY` | missing, extra, non-boolean, or non-false permanent nonclaim |
| `FAIL_R01_PRESENTED_MECHANICS_V1_RESULT` | private result type, lineage, snapshot, or truth-vector violation |
| `FAIL_R01_REAL_CURRENT_CUT_ACTIVE_NONREVOKED_V1_NOT_IMPLEMENTED` | unconditional real-entry boundary |

The module also asserts at import that the committed catalog payload, catalog
ID, R01 gate descriptor, exact 24-source order/cardinality, and public catalog
spec still match the inspected v1 surface. Any drift fails before verification
rather than silently changing this downstream contract.

## 9. Focused and dependency-integrated QA record

The focused command is:

```bash
PYTHONPATH=src python3 -m pytest -q --capture=sys \
  tests/test_phase2b_actual_unsealed_960_verify_r01_real_current_cut_active_nonrevoked_v1.py
```

Dependency-integrated QA covers RFC8785, the frozen catalog and Package 0,
the offline authority/owner/time/runtime/store/evidence cores, the prior R01
presented overlays, readiness/nonclaim ancestors, and this downstream slice.
Each file is intentionally run in a fresh process with its own owner-private
`0700` local temporary directory: legacy negative KATs deliberately rebind
module globals, and the CAS fixtures correctly reject a non-private temp root.
The exact command form for each file is:

```bash
env TMPDIR=<owner-private-0700-local-temp-subdirectory> \
  PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  python3 -B -m pytest -q --capture=sys -p no:cacheprovider \
  tests/<one-integrated-test-file>
```

The 16 independently executed files were `test_rfc8785_v1.py`,
`test_phase2b_formal_wire_covert_step5_authority_v2.py`,
`test_phase2b_formal_wire_covert_step5_owner_status_replay_core_v1.py`,
`test_phase2b_formal_wire_covert_step5_logical_time_replay_core_v1.py`,
`test_phase2b_formal_wire_covert_step5_runtime_closure_v1.py`,
`test_phase2b_formal_wire_covert_step5_store_replay_core_v2.py`,
`test_phase2b_formal_wire_covert_step5_r01_evidence_contract_v1.py`,
`test_phase2b_formal_wire_covert_step5_r01_pure_replay_v1.py`,
`test_phase2b_formal_wire_covert_step5_r01_external_witness_admission_contract_v2.py`,
`test_phase2b_formal_wire_covert_step5_r01_live_runtime_overlay_v2.py`,
`test_phase2b_actual_unsealed_960_readiness_dag_v1.py`,
`test_phase2b_actual_unsealed_960_replay_admission_evidence_contract_v2.py`,
`test_phase2b_c1_statistical_validation_v1.py`,
`test_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1.py`,
`test_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_v1.py`,
and the new `test_phase2b_actual_unsealed_960_verify_r01_real_current_cut_active_nonrevoked_v1.py`.

The unchanged-ancestor check is:

```bash
git diff --exit-code 95fb08e1fe97c7d268f97f5b5967d3842f3e9c3a -- \
  src/hegel_machine/phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1.py \
  config/phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1.json \
  tests/test_phase2b_actual_unsealed_960_authoritative_evidence_schema_catalog_v1.py \
  docs/Hegel_Machine_Phase2B_Actual_Unsealed_960_Authoritative_Evidence_Schema_Catalog_v1.md \
  src/hegel_machine/phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_v1.py \
  config/phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_v1.json \
  tests/test_phase2b_actual_unsealed_960_durable_signed_custodian_ledger_and_signer_chain_v1.py \
  docs/Hegel_Machine_Phase2B_Actual_Unsealed_960_Durable_Signed_Custodian_Ledger_And_Signer_Chain_v1.md
```

The KAT source uses an independent literal 24-source/predecessor/profile oracle
and a deterministic full 193-leaf corpus builder. Its focused cases cover:

- the complete positive graph, exact source-graph root, and length-framed
  synthetic-corpus fingerprint;
- exact public ABI, exported surface, catalog schema-frozen/null-SHA boundary,
  and the 38-field private truth vector;
- byte subclasses, lists in place of tuples, missing/extra/swapped/duplicated
  sources, and exact JSON keyset/array-shape failures;
- an IDs/SHAs/counts/root-only composite with its raw leaves removed;
- missing, reordered, mutated, and runtime-copy-drift leaves across the owner,
  time, runtime, CAS, and dual-receipt expansion;
- independently recomputed source/leaf IDs, raw SHA-256 values, byte counts,
  leaf roots, predecessor links, and cross-corpus splice rejection;
- active/zero-revocation profile drift and every missing, extra, non-boolean,
  or true permanent nonclaim;
- direct-result forgery, copy, deep-copy, pickle, and post-issuance mutation;
- same-ABI real `NOT_IMPLEMENTED`, exact config bytes, post-import rebinding,
  controlled pre-import catalog poisoning, offline/producer-free AST and fresh
  import checks, and the 16 MiB + 1 pre-decode boundary.

These are coverage statements about the KAT source. The regenerated final
source/config/KAT bytes pass all 47 focused instances. The 16-file isolated
dependency matrix passes all 702 instances, and all eight frozen catalog-v1
and Package0 source/config/test/doc artifacts remain byte-identical to HEAD
`95fb08e1fe97c7d268f97f5b5967d3842f3e9c3a`.

The final QA record is:

```text
focused result = 47 passed
focused test instances = 47
dependency-integrated result = 702 passed, 0 failed, 16 isolated processes
dependency-integrated test instances = 702
unchanged-ancestor result = PASS, 8 artifacts byte-identical to 95fb08e1
synthetic corpus fingerprint = 41a3219b57968272c89548e92b1f8c01256e3ede5e08c8a3106f4edc121165cd
synthetic source-graph root = sha256:59727ae147c0035e62219603810c4e9845e5d5f9a9c2687e61a2b60fb0806f18
```

## 10. Remaining hard gate

This slice does not advance the readiness DAG. The earliest unresolved hard
gate remains externally current, authoritative, active, non-revoked R01
evidence under a separately versioned real verifier. Package 1, network
security policy, account configuration, live actors, external execution, and
the actual-960 campaign remain outside this construction.

The next hard-gate review must therefore preserve the exact distinction:

```text
presented byte mechanics: locally verifiable
external currentness and authority: not verified
R01 gate admission: false
R01--R16 readiness: 0/16
real entry: NOT_IMPLEMENTED
execution / actual-960 / C1: false
```
