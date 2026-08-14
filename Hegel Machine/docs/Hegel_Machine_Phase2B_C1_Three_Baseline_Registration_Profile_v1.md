# Phase-2B C1 three-baseline registration mechanics profile v1

**Profile ID:** `phase2b_c1_three_baseline_registration_profile_v1_27cde92a67d2d92f83e52c9b60c84ace1138df7df134756b38a4cbfb98f2166d`

**Claim level:** `MECHANICS_ONLY_NOT_REGISTRATIONS`

## Exact boundary

This slice freezes pure, in-memory mechanics for three caller-supplied baseline
registration candidate graphs. It does not download or inspect model artifacts,
build or run a container, generate a key, create a production signature, write a
CAS object or registry snapshot, consult a clock, allocate campaign or attempt
identifiers, generate holdout cases, or issue an actual registration.

The checked-in JSON artifact uses the literal encoding
`RFC8785_CANONICAL_BODY_PLUS_ONE_LF`: its body is RFC 8785 canonical JSON and the
artifact appends exactly one LF. The file including that LF is not itself
described as RFC 8785 bytes. The profile content ID is independently recomputed
from the canonical profile body, not from an unverified checked-in ID.

The supplied R05 value is a predecessor *candidate*. Its exact shape and content
address are checked, but no authority is established. R05 remains required
before R07; R06 may run in parallel with R07 after R05; R08 still requires both
R06 and R07. A mechanics success does not satisfy any of those readiness nodes.

## Exact three-baseline closure

The aggregate order is closed and exact:

1. `embedding_nearest_prototype`, specification
   `phase2b_baseline_spec_d14fa83a6843b92d1a502bba97b982323d1de8bf0abe22763d8c1cb67c97cfa4`,
   implementation `sentence-transformers/all-mpnet-base-v2`;
2. `frozen_llm_semantic_only`, specification
   `phase2b_baseline_spec_c07005a2e32423f5e11a2160367b089d8f69208289239bfc18821fb351f13fa8`,
   implementation `Qwen/Qwen2.5-7B-Instruct`;
3. `flat_learned_typed`, specification
   `phase2b_baseline_spec_0e7b098030bf7e5a36f2deb0befd40fbabc55ce538a78765be2aa33d6b7af6d2`,
   implementation `sklearn.ensemble.HistGradientBoostingClassifier`.

The two Hugging Face candidates require exact lowercase 40-hex revisions. The
HGB candidate requires an exact lowercase 40-hex repository commit and a
SHA-256 dependency lock. The HGB manifest contains exactly one
`DEPENDENCY_LOCK` role whose digest equals that lock; the HF manifests cannot
carry the role.

Each artifact manifest is a bounded, exact tuple. Relative paths are unique,
ascending, portable, and traversal-free. Artifact roles are unique and equal
the exact per-baseline role set. Sizes, counts, and their aggregate are exact
RFC 8785 safe integers, at most `9007199254740991`. The implementation
recomputes the manifest root as:

`SHA256(HEGEL/PHASE2B/C1/THREE_BASELINE/ARTIFACT_FILE_MANIFEST/V1\x00 || RFC8785(entries))`.

The remaining source/archive/configuration/decoder/SBOM/image/development-state
and offline-qualification digests are deliberately named caller-supplied
candidate bindings. Their syntax and cross-component equality are checked, but the
underlying artifacts are not replayed here. Therefore
`artifact_roots_replayed=false` even on mechanics success.

## Five-purpose candidate signature graph

The exact purpose order is:

1. `BASELINE_RUNTIME_BUILDER_ATTESTATION`;
2. `BASELINE_OFFLINE_QUALIFIER_ATTESTATION`;
3. `BASELINE_REGISTRATION_CUSTODIAN_ATTESTATION`;
4. `STORE_IF_ABSENT_RECEIPT`;
5. `DECLARED_TIME_TOKEN`.

The input contains five epoch-1 Ed25519 candidate public keys. Their raw 32-byte
values and content-addressed key IDs must both be pairwise distinct. Each
baseline supplies five caller-provided 64-byte signatures, giving fifteen
purpose/domain/preimage-specific verifications in baseline-then-purpose order.
Cryptographic success proves only that those supplied public bytes verify those
supplied signatures. It does not prove signer identity, origin, organizational
independence, external authority, or a governance role.

The store receipt candidate binds its exact record and custodian candidate,
derived object key, record-content digest, `IF_ABSENT`, and generation `0 -> 1`.
The three store object keys and receipt IDs must not replay across baselines.

Declared-time candidates use two exact integers: `declared_unix_seconds` in
`0..9007199254740991` and `declared_nanosecond` in `0..999999999`. Logical
sequences are exactly `0, 1, 2`; predecessor tokens form
`null -> token[0] -> token[1]`; `(seconds, nanosecond)` pairs are strictly
increasing. This is a signed causal declaration, not evidence that an external
clock observed any event.

## Snapshot and atomic verification

Before snapshot traversal, a fixed, nonrecursive shell guard requires one exact
R05 candidate, exactly five purpose-ordered key candidates, and exactly three
baseline-ordered final candidates. The verifier then walks that graph into exact
immutable tagged primitives. The walker applies exact-type, depth, node, tuple,
UTF-8 text, raw-byte, and safe-integer bounds before descending; rejects cycles;
counts repeated DAG occurrences; and makes real byte copies.

The cumulative ceilings are 32,768 nodes, depth 32, 8,192 tuple entries, 1 MiB
of UTF-8 input, 1 MiB of raw bytes, and 2 MiB combined leaf bytes. Before a
tagged JSON projection is materialized or RFC 8785 runs, a conservative size
estimator enforces a 4 MiB canonical-byte ceiling. The snapshot then receives a
domain-separated SHA-256 commitment. Validation, all preimages, all IDs, the
registry, and projections are derived only from detached objects thawed from
that snapshot.

After each of the fifteen real Ed25519 verifications, the original graph is
re-snapshotted and compared exactly with the initial tree, canonical bytes,
commitment, and node count. Mutation at any verification boundary rejects with
no candidate IDs, projections, or partial output. Only after all checks does the
verifier derive the ordered three-ID registry root and registry ID. No aggregate
CAS receipt or aggregate time token exists, so
`aggregate_snapshot_persisted=false`.

## Cooperative result and legacy isolation

`ThreeBaselineRegistrationMechanicsV1` is an ordinary cooperative computation
DTO. Python callers can construct or mutate objects; the object itself is not
unforgeable evidence. It therefore carries `result_object_unforgeable=false`
and `downstream_fresh_reverification_required=true`. The downstream
`verify_three_baseline_registration_mechanics_result_v1` API first reruns the
complete graph and all fifteen real signature checks, then applies an exact
fixed-field result guard before the same cumulative snapshot budgets and final
canonical comparison. Every public result field and claim polarity must match.

No helper emits the existing execution protocol's accepted baseline
registration DTO. The only compatibility-shaped value is
`LegacyBaselineProjectionCandidateV1`, with
`eligible_for_execution_freeze=false` and `r07_registration_verified=false`.
Its enum and field closure are intentionally incompatible with the old
execution-freeze consumer. It cannot bypass R07.

## Qualification fixture boundary

The tests contain five fixed raw public keys, fifteen fixed 64-byte signatures,
and fifteen fixed preimage hashes. Under explicit fixture-only authorization,
the literals were generated once in memory from deterministic
`TEST_ONLY_NON_AUTHORITY` seeds. No private bytes were persisted, production
code contains no private-key or signing surface, and pytest never signs. The
fixture keys and signatures must never be reused for actual registration, CAS,
or ceremony work.

Even a successful mechanics result leaves all substantive claims false:

- R05 authority, R07 completion, and R08 unlock;
- actual baseline registrations or execution-freeze eligibility;
- artifact-root replay and aggregate snapshot persistence;
- external trust, signer identity, organizational independence, durable CAS,
  rollback resistance, authoritative time, or observed chronology;
- campaign/attempt allocation, holdout generation, actual execution, network
  access, Docker invocation, key generation, or signature generation.
