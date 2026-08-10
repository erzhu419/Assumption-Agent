# Hegel Machine Phase-3A-Q0 Dual Qualification Result v1

## 1. Verdict

The frozen Phase-3A-Q0 micro projection has passed all **14/14 Q0 readiness
gates** under the dual isolated Python/Rust qualification profile.  The exact
artifact status is:

```text
status: DUAL_EXHAUSTIVE_ORACLE_EQUIVALENCE_PASS
claim_scope: Q0_DUAL_ENGINEERING_QUALIFICATION_ONLY
q0_state: QUALIFIED_NOT_Q1_RUN
readiness_gates: 14/14
readiness_gate_mask: 0x3fff
```

This result qualifies the exact quotient mechanics on
`hegel-q0-micro-projection-v1`.  It is **not** a Q1 complete quotient-closure
result, a Q2 target-role result, an M3 formal execution result, or an outside
language certificate.

The qualified frozen identities are:

| Object | Frozen identity |
|---|---|
| DSL | `hegel-old-dsl-v1.6.0` |
| closure semantics | `hegel-quotient-closure-v1.0.1` |
| Q0 freeze | `hegel-freeze-p3a-q0-v1.0.1` |
| projection | `hegel-q0-micro-projection-v1` |
| source commit | `ae201905526747980ff8e46e70f9ec96de93d3eb` |
| formal artifact | `artifacts/phase3_q0_dual_exhaustive_oracle_qualification_v1.json` |
| artifact SHA-256 | `1d8d530cbccf133af97da900164b89c9385db21d873bfff039ee7ce07c45f28d` |
| artifact byte length | `1,443,257` |

## 2. Artifact and receipt identity

The checked artifact is canonical JSON with no duplicate keys.  Its source
manifests were replayed against the full commit above, and its evidence roots
were independently recomputed from their canonical preimages.

| Evidence object | Root |
|---|---|
| pre-dual gate evidence | `sha256:918683fd91ef1f201ae52e2cf9c182f2ffcc0a8fde792f931c60f66cbdc3bd9d` |
| pre-receipt gate evidence | `sha256:e19c16e2843237c46b47f188d5f5a4a636d23c219947505eca9b7ac5b8cad569` |
| final 14-gate evidence | `sha256:0d3550b5bcf7566cbd187094134436d84f30a80269b5afcb8878919d9515c6d1` |
| isolation evidence | `sha256:e51530fd93086ae716a1de79f19875427c032102a693ef3634100a8eba573d1d` |
| 40-field saturation receipt | `sha256:ee198614e94cf425202f9c667836fc6ad61fda02c9439a689eb90012c5798ad2` |

The receipt was decoded as strict canonical CBOR and re-encoded byte for
byte.  Its tail is exactly:

```text
[gate_total=14, gate_mask=0x3fff,
 q1_status_id=0, q1_output_root=null,
 q2_status_id=0, role_evaluation_performed=false,
 m3_formal_roots=null, outside_certificate_issued=false]
```

The receipt binds the Python and Rust implementation roots, both equal
endpoint roots, and the host-replayed class root.  Gate 14 was promoted only
after this candidate receipt was replayed; the pre-receipt evidence retained
Gate 14 as pending.

## 3. Exact endpoint, state, and preimage agreement

| Object | Qualified root or digest |
|---|---|
| probe universe | `sha256:2c960bcc229175afe6d5e106a34410216669bfe66b14d5c85103762c596f4192` |
| projection manifest | `sha256:2f39aa248f1305eeaf20a724f6d690cf2b13003f86620d09d2753815831f7ad1` |
| semantic binding | `sha256:b7ec5e860a007469b8a1b3930f17c130f59a800d2a832dfd438d18a75538ff99` |
| syntax program archive | `sha256:bd1a59f816bd6648d0dd73b9a1622f2bb88bb9aeca1489a0d876fbc9dbf0c829` |
| syntax class archive | `sha256:a2f0dacf4524fdb8725d29a2c3883a7ebd78fa686cb2030ac0d0608710176cf1` |
| direct quotient class archive | `sha256:a2f0dacf4524fdb8725d29a2c3883a7ebd78fa686cb2030ac0d0608710176cf1` |
| syntax operator coverage | `sha256:6953f39dc97f17288850b524ca8b04dbb2f6ddd3d53eaf4cb8e4e6465bcd840c` |
| direct quotient operator coverage | `sha256:a9a0b6fdc97c475323ccae31fba14a6df411307220efd8538c7971fe9c38c1fd` |
| syntax saturation state | `sha256:7028819d133c4da6071c06a0bfca2d0b91622e106207d0b0f081148f41c0826a` |
| direct quotient saturation state | `sha256:d87ef33d9d7010ded284b55acfa71aab4d7d991e3d7703c30f1db2caf5893933` |
| common 43-field endpoint state | `sha256:d33e54dd99e6cbe8aacc541fc0877af9657a553be58523670cce5c474006d4d2` |

Both implementations produced byte-identical canonical saturation-state
preimages:

| Preimage | Bytes | Raw SHA-256 |
|---|---:|---|
| exhaustive syntax state | 127,439 | `9df6cc3101a75d5d3f1d7f2707761ee4fa523b82bcf3f480a74950ebeab45507` |
| direct quotient state | 125,153 | `cf2d964cc482e3ae99cc5e7ed3ed22b386d22894631f07fc33f65066c72fa42f` |

The two endpoint JSON streams differ only in implementation identity and
per-implementation diagnostic material.  Their stream digests are
`538a1d6508ae0b5d19d773f0f786b7f55155daeb938a9f16e6cd708651c54e90`
for Python and
`1856f74aa518b965cca0a64c0e64af620580aab151d913fa5d73de452a38797f`
for Rust; the implementation-neutral 43-field endpoint CBOR and its root are
identical.

## 4. Qualified micro-projection counts

| Quantity | Exhaustive syntax | Direct quotient |
|---|---:|---:|
| frozen leaves | 15 | 15 |
| raw operator applications | 567 | 545 |
| strict-admitted applications | 567 | 545 |
| rewrite collapses | 30 | 30 |
| continuation-bank points | 251 | 251 |
| maximum bank points per behavior class | 43 | 43 |
| saturation rounds | 3 | 3 |

The exhaustive path contained 537 canonical syntax programs.  Both paths
closed to 69 exact behavior classes, 122 visible Pareto-frontier points, and a
maximum visible frontier size of 4 per class.  The terminal round had an empty
work queue and zero class, frontier, and continuation-bank deltas.

These are counts for the frozen **Q0 micro projection**.  They are not the
cardinality of the full v1.6 quotient closure and must not be reused as a Q1
capacity or completeness claim.

## 5. Gate ledger

| Gate | Name | Result |
|---:|---|---|
| 1 | `NORMATIVE_DIRECTION_BYTES_BOUND` | PASS |
| 2 | `V16_DSL_TYPING_AND_REGISTRY_ROOTS_QUALIFIED` | PASS |
| 3 | `INPUT_SIGNATURE_OBSERVATION_ADAPTERS_QUALIFIED` | PASS |
| 4 | `BEHAVIOR_AND_BOTTOM_CODEC_QUALIFIED` | PASS |
| 5 | `UNIVERSE_ONLY_BINDINGS_QUALIFIED` | PASS |
| 6 | `EXACT_EQUIVALENCE_CONTRACT_QUALIFIED` | PASS |
| 7 | `CONSTRUCTION_SIGNATURE_QUALIFIED` | PASS |
| 8 | `PARETO_DOMINANCE_AND_MDL_QUALIFIED` | PASS |
| 9 | `PER_OPERATOR_CONGRUENCE_QUALIFIED` | PASS |
| 10 | `STRUCTURAL_INDUCTION_COMPLETENESS_QUALIFIED` | PASS |
| 11 | `EXHAUSTIVE_MICRO_ORACLE_EQUALITY_QUALIFIED` | PASS |
| 12 | `COLLISION_BOTTOM_SORT_ADVERSARIAL_VECTORS_PASS` | PASS |
| 13 | `TARGET_TRUTH_AND_SPLIT_INPUT_ISOLATION_PASS` | PASS |
| 14 | `DUAL_HOST_AGREEMENT_Q1_OUTPUTS_NULL_NOT_RUN` | PASS |

All final rows have `pending_dual=false`.  Passing Gate 14 means that the Q0
dual agreement and the downstream `NOT_RUN`/`null` guard were jointly
verified; it does not mean Q1 was run.

## 6. Isolation evidence and trust boundary

The Python and Rust endpoints were run concurrently from distinct read-only
source snapshots with:

- Docker `--network=none` and `--pull=never`;
- pinned local image IDs
  `python@sha256:e031123e3d85762b141ad1cbc56452ba69c6e722ebf2f042cc0dc86c47c0d8b3`
  and
  `rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89`;
- no endpoint-output exchange;
- target, truth, and split sources absent from both endpoint snapshots;
- an offline, read-only Cargo identity containing exactly 21 lock-selected
  registry dependencies, 1,033 verified extracted dependency files, and 1,099
  sealed Cargo-home files;
- all seven isolation prerequisites replayed as PASS.

The implementation roots are:

| Role | Implementation root |
|---|---|
| Python endpoint | `sha256:66f881531292e8dac1a2fa85a337ca5d9dc517f069d6c39e89042c268a2619cd` |
| Rust endpoint | `sha256:136bb507fb3849fc0c12b0094cd553770b5c8eef2af98a132f47ab7502236273` |
| host replay/issuer | `sha256:1fd55bf792709aea6be28198bcd6ff75444c45e254714cfe68087aa370185ae2` |

The trust boundary must remain explicit.  Python and Rust have technical
process/filesystem isolation, but all three roles remain under the same
administrative controller; organizational independence is false.  The host is
a **trusted target-blind replay and receipt issuer**, not a third independent
endpoint and not a filesystem-hard-isolated external authority.  Its exact
import manifest is source-bound, including the pre-dual gate qualification
module loaded before host replay.  No external signatures were produced.

Accordingly, this result supports a dual engineering qualification with a
trusted host replay.  It does not support a claim of three independent
organizations, a signed formal certificate, or an adversarially independent
external audit.

## 7. Round-history fixture incident and correction

An earlier test fixture carried the implicit assumption that the Python and
Rust implementations must expose identical intermediate `direct_rounds`
mutation counters.  That assumption is too strong: legal enumeration and
frontier-maintenance schedules can differ while producing the same complete
semantic state.

The formal run made the permitted difference concrete:

| Round | Python frontier/bank mutations | Rust frontier/bank mutations |
|---:|---:|---:|
| 1 | 46 / 82 | 37 / 68 |
| 2 | 73 / 222 | 70 / 182 |
| 3 | 0 / 0 | 0 / 0 |

The correction at the qualified source commit was:

1. exclude `direct_rounds` from cross-implementation shared-field equality
   and from the implementation-neutral 43-field semantic-state identity;
2. validate each implementation's round indices, non-negative counters, and
   final full zero-delta row independently;
3. retain byte equality for the complete state preimages, endpoint-state
   CBOR/root, class archive, and all semantic counts and roots;
4. update the Rust test fixture to its real intermediate mutation counts and
   add the regression
   `test_dual_comparison_allows_only_intermediate_round_history_drift`.

This was a diagnostic-fixture/comparison-boundary defect, not a mismatch in
the qualified quotient state.  The repair prevents harmless schedule drift
from causing a false failure without weakening exact terminal equivalence.

## 8. Closed downstream state

The exact state after this qualification is:

```text
Q0: QUALIFIED_NOT_Q1_RUN
Q1 status: NOT_RUN (id 0)
Q1 output root: null
Q2 status: NOT_RUN (id 0)
role evaluation performed: false
M3 formal roots: null
outside certificate issued: false
signatures: null
```

No odd-target or hidden-sink role verdict was computed.  No target truth or
split was opened.  No `OUTSIDE_FROZEN_QUOTIENT_CLOSURE(...)` certificate was
issued, and this artifact neither performs nor authorizes an `ACTIVE`
transition.  Q1 requires a separate, explicit target-blind full quotient
closure execution; Q2 remains downstream of a completed Q1.

## 9. Independent artifact replay performed for this result

Before writing this document, the artifact was replayed against the committed
source with the actual execution ordering.  The replay verified:

- canonical JSON bytes and the artifact SHA-256;
- exact Python, Rust, and host source manifests against the full Git commit;
- live pre-dual evidence and its source binding;
- strict validation of both endpoint schemas, coverage rows, state preimages,
  endpoint CBOR, and endpoint roots;
- host target-blind state and class-archive replay;
- canonical reconstruction of isolation, pre-receipt, and final-gate evidence
  roots;
- strict CBOR decode/re-encode and root replay of the 40-field receipt;
- the exact `14/14`, `0x3fff`, downstream `NOT_RUN`/`null` boundary.

The replay passed.  Its scope is artifact integrity and Q0 engineering
qualification only; it does not manufacture any downstream execution result.
