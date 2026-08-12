# Phase-2B formal-track preregistration status

## Canonical name and present status

The canonical future milestone name remains:

> **Phase-2B Sealed Typed-Evidence Structural Identification Qualification**

The reserved capability claim is **Sealed Typed-Evidence Structural Law
Identification and Verification**. Neither is a description of the current
result. The checked-in result remains **Phase-2A Controlled Typed-Selector
Mechanics Qualification**; Phase-2B is shadow-only preregistration and
implementation infrastructure.

The exact protocol choices are no longer open questions. They are frozen by
`Hegel_Machine_Phase2B_Phase3_Exact_Freeze_Decisions.md` plus the
[v1.0.2 strict canonical/certificate amendment](Hegel_Machine_Strict_Canonical_AST_CBOR_Certificate_Bridge_Freeze_v1.0.2.md)
under `hegel-freeze-p2b-p3-v1.0.2`. v1.0.2 inherits the v1.0.1 seed correction:
`411876909552964556` remains the master/bootstrap seed, while sklearn's
executable `random_state` remains the domain-separated SHA-256-to-uint32 value
`2611585425`. It additionally completes the Phase-3 strict acceptance and
certificate specification. The subsequent M1 artifact verifies both strict
implementations against the shared vectors, and M2 establishes a bounded
`DSL_TOO_LARGE` status for the old DSL; neither result generates, seals, runs,
audits, scores, or consumes a Phase-2B holdout. The following states therefore
remain false:

```text
formal_phase2b_exit_claim
sealed_holdout_generated
sealed_holdout_consumed
independent_custodian_attested
external_isolation_attested
formal_covert_channel_audit_passed
recognizer_image_built_from_allowlist
recognizer_entrypoint_implemented
formal_recognizer_run_runnable
```

The Phase-3 distinction must not be reused as Phase-2B evidence:

```text
strict specification frozen                         = true
strict Python/Rust shared vectors                    = 48/48 PASS each
accepted strict canonical count                     = 64,680 unique each
hegel-old-dsl-v1.0.0 under 50,000 syntactic budget  = DSL_TOO_LARGE
complete closure / extensional target verdict       = false / unavailable
formal roots                                        = null
hidden-sink formal verdict                          = false
outside or MDL certificate issued                   = false
Phase-2B sealed qualification                       = false
ACTIVE promotion                                    = false
```

The Phase-3 evidence is the
[dual strict gate](../artifacts/phase3_dual_strict_gate_v1.json) and
[dual strict capacity replay](../artifacts/phase3_dual_strict_capacity_replay_v1.json).
Both implementations produced diagnostic set commitment
`sha256:c1a02a66a8d6d8f75204cb3daf03ab0b01c2b3b8e486d0ab3d481ee3be43c930`;
the ordinal-50,001 AST hash is
`sha256:7c7f786c2cc57d31506b3c61d162d175c7f69a2878a089c72c9d053694cba948`.
Neither value is a formal archive root or an extensional target verdict.
Its required next action is to publish a new old-DSL version, apply frozen
shrink step 1 by removing `mean_v1`, `min_v1`, and `max_v1`, regenerate target
and validation commitments, and restart the new version from `NOT_RUN`; see the
[readiness resolution](Hegel_Machine_Phase3_Freeze_Readiness_Resolution.md).
Phase-2B formal exit remains an independent NO-GO until the sealed-data,
custodian, runtime, covert-audit, baseline and statistical gates below pass.

## Exact sample freeze

The main sealed holdout contains 720 independent latent cases:

```text
6 families × 2 scales × 60 cases = 720
```

Each `family × scale` cell has the exact case quota:

| Case type | Per cell | Total |
|---|---:|---:|
| `unique_scale_answerable` | 19 | 228 |
| `admissible_scale_set_answerable` | 1 | 12 |
| `wrong_family_hard_negative` | 8 | 96 |
| `binding_counterfactual` | 8 | 96 |
| `scale_counterfactual` | 8 | 96 |
| `sign_or_invariant_break` | 8 | 96 |
| `insufficient_or_nonidentifiable` | 8 | 96 |
| **Total** | **60** | **720** |

The exact per-cell margin strata are `21 / 18 / 12 / 9` for
`clear_interior / moderate / near_boundary_identifiable /
nonunique_or_insufficient`. The final stratum contains the eight insufficient
cases and the one set-valued answerable case. Set-valued cases are answerable:
they enter family, binding, scale-set and joint-exact denominators, but not
unique-scale, abstention-specificity, or nonidentifiability-abstention
denominators. `joint_exact` requires exact family, binding and scale set plus
`ANSWER_SET`; a singleton, strict subset, strict superset, or abstention is
wrong.

An additional 240-case sealed semantic-conflict challenge is frozen alongside
the main 720:

```text
12 cells × (10 low-semantic-overlap structural positives
            + 10 high-semantic-overlap structural negatives) = 240
```

It shares the freeze/reveal event but is not part of the main accuracy
denominator and may not tune thresholds. Therefore the formal data commitment
is **720 main + 240 challenge**, not “720 including the challenge.”

The derived preservation/sensitivity suite is also separate from the 720
latent cases:

```text
496 legal preservation pairs + 76 invalid-transform controls = 572 pairs
```

The six canonical family IDs, transformation applicability matrix, pair counts,
footprint classes (`P2_PAIR`, `P3_CHAIN`, `P4_STAR`, `PSET_AGGREGATE`), grouped
permutation-importance gates, baseline configurations, 10,000-replicate paired
cluster bootstrap, rerun allow/deny list, and validation-version failure policy
are all frozen. Actual external model revisions and artifact hashes still have
to be pinned before a run.

## Process and trust boundary

```text
public protocol + frozen code/image/config/baselines
  → independent custodian generates new latent cases
  → trusted wire builder assigns shuffled IDs and fixed envelopes
  → covert-channel audit over channel-only fields
  → public typed-evidence root + private salted answer commitment
  → OCI-isolated untrusted recognizer runs once
  → prediction/audit archive hashes are committed
  → custodian reveals the answer manifest
  → independent evaluator scores once
  → immutable report marks the holdout consumed
```

Phase-2A fixtures remain useful for tests and adapter regression, but may not
enter validation, calibration, confidence intervals, baselines, or the sealed
holdout. The public selector re-enumerates family, role-binding, and scale
hypotheses from the bound evidence and registry; it does not trust a
caller-provided candidate grid.

A Python dataclass, OCI argv, or Boolean flag cannot prove OS isolation,
custodian independence, or one-shot consumption. Formal evidence needs a
detached runtime attestation bound to the image, run specification, public
input root, and prediction archive. The current process-local lifecycle guard
is not a durable signed append-only custodian ledger.

## Public wire and uncertainty boundary

The allowlist rejects explicit answers such as `law_family`, `correct_binding`,
`correct_scale`, PASS/FAIL labels, ranks, oracle margins, and candidate-private
payloads. That is only a syntactic boundary. UUID bytes, provenance values,
record order, missingness, lengths, padding, unused identifiers, and eligibility
masks can still carry an answer.

The frozen covert-channel audit requires independent `K_shuffle`, `K_id`, and
`K_padding`; post-shuffle HMAC-derived UUIDs; a 65,536-byte envelope; channel
feature tests; 10,000 stratified label permutations; Holm–Bonferroni
`FWER = 0.01`; `NMI <= 0.02`; balanced-accuracy advantage `<= 0.05`; 32
consistent global renamings; 32 case-order permutations; and 16 within-case
observation-order permutations. Fixed-envelope prefix/suffix feature extraction,
NMI/LOO balanced-accuracy statistics, one global Holm family, and 32/32/16
invariance mechanics now have a deterministic non-authoritative implementation.
Stage-A wire mechanics also provide a schema-closed accepted-JCS profile, an
explicit frozen-10 plus V2-extension-6 UUID path manifest, and a replayable
65,536-byte envelope with public test padding. Stage-B mechanics now accept only
an exact authority tuple, exact run ID, and three pairwise-distinct 32-byte IKM
values; derive purpose-separated shuffle/ID/padding keys; perform unbiased
whole-batch Fisher–Yates; assign case-local post-shuffle HMAC UUIDv4 values with
batch-global counters/collision accounting; recanonicalize the renamed schema;
compile public base provenance plus shared exact-validator-native derived provenance
before framing; emit secret-HMAC-padded envelopes atomically; and support byte-exact
replay from the supplied custodian inputs. The Stage-C closed typed codec losslessly
decodes/re-encodes all eight certificate profiles, binary64 atoms, and exact rationals.
Each emitted payload authority directly obtains exact-transform `COMPLETE` without
post-decode mutation. Its whole-batch replay also binds Stage-B membership, source
authority order, the supplied-secret receipt, and every transform result. Given exact
top-level API types, batch/custodian/typed semantic drift returns one batch `ABSTAIN`
with no partial roots; a top-level type-contract violation raises `TypeError` before
replay.

The post-Stage-C recognizer-input archive mechanics are not a recognizer.  A
custodian-gated issuer compiles each `PublicRecognizerRegistryV1` read-only from
the same live Stage-B shuffle/HMAC allocation that emits its adjacent envelope;
it neither reruns an approximate allocator nor retains a private allocation
sidecar.  The source registry must exactly rebuild the frozen six-law,
15-role, 35-observable theory with the 50,000 candidate cap.  Fixed public
bridge-family UUID aliases remain distinct from the F01--F06 canonical family
identities.  Registry entity/role/quantity scope must exactly match the strict
typed payload authority, all source UUIDs must be disjoint from all public UUIDs,
and public rows may not reuse authority UUIDs.  The archive exposes no old IDs,
source indices, per-case source-authority roots or authorities, renaming maps,
permutations, or keys.  The sole exception is one global opaque
`source_registry_id` commitment in metadata; it does not establish source
projection or trust.  Exact top-level type misuse raises `TypeError` before any
archive root; other source, custodian, or semantic drift returns one atomic
`ABSTAIN` with no partial archive.

Successful issuance still returns only `DecodedRecognizerInputArchiveV1`, the
same false-secret-claim type produced by the public decoder.  It replays bounded
framing, the closed registry schema, registry/envelope scope bijection, and direct
payload exact-transform completion.  It does not verify batch membership, source
projection, secret custody, origin, formal covert audit, sealed eligibility,
recognizer execution, prediction evaluation, or C1 exit.  Opaque commitments are
not a durable self-verifying trusted receipt; no such public receipt or recognizer
effect evidence is implemented.

Those V1 identities and conclusions remain frozen. The compact V2 path has an
independent lossless codec schema and policy, batch `/3` payload schema, V2 envelope
magic and policy, public typed-replay policy, and V2 registry/archive schemas and
content-ID domains. Its private issuer observes one live allocation and requires all
global source UUIDs to be disjoint from all public UUIDs, including fixed aliases;
that private gate is not a durable public claim.
The public V2 decoder replays only bounded archive and row structure, registry schema
and authority scope, compact typed authority, direct exact-transform semantics, and
cross-row disjointness only for unlinkable public UUIDs (authority plus registry
role/quantity UUIDs, excluding fixed aliases that may repeat). Batch/source
projection, source-public and single-allocation verification, secret custody, origin,
formal audit, sealed eligibility, recognizer/prediction execution, capacity, and C1
remain false.

The V1 prediction-archive slice remains non-authoritative mechanics. An exact
`PublicRunContextV1` binds the decoded input archive ID/SHA, current protocol ID,
exact execution-freeze-manifest commitment, ordered input-row root, and expected
count 960. Each of the 960 public prediction records is independently length-
framed bounded accepted-JCS; the manifest binds canonical input-row and record
roots, while each record transitively binds its prediction-content root. The issuer
reruns `initial_theory()`, the adjacent public registry, and the exact derived
bridge as an internal derived-to-prediction gate.
The durable public decoder does not re-prove that gate: it verifies only structural
archive, canonical framing, closed record schema, and row-root coverage. Input
membership, manifest authority, derived mapping, runtime, capacity, origin, sealed/
formal audit, scoring, effect, and C1 claims remain false. Recognizer-facing decoded
semantic fields and values contain no split, gold, index, ordinal, or case-position
labels; this is not a claim about arbitrary substrings in binary hash bytes.

The separate unsealed evaluator is evaluator-side only. It proves that exact 720
and 240 row-ID lists are sorted, disjoint, exhaustive, and rooted in the same 960
prediction archive, then returns only `STRUCTURALLY_COMPLETE_NOT_SCORED`. It has no
score callback, metrics, effect claim, recognizer runtime, or capacity evidence.
The runner similarly freezes only the exact total of 960, not an executable
recognizer entrypoint or an actual 960-case run.

The historical V1 verbose-profile size gate was `125,582 > 65,424`. In one
constructed positive regression, compact V2 represents the same exact logical
authority in a 50,255-byte payload with 15,169 bytes of payload-cap headroom and
15,201 bytes of secret padding in the unchanged 65,536-byte fixed envelope. That row
replays the exact transform and V2 recognizer-input archive and preserves derived-bridge compilation
and decision parity. This closes the old payload-size P0 for one mechanics case only;
it is not prediction E2E or capacity evidence. The V1 prediction archive rejects a
V2 input archive by exact type and policy. The independent V2 single-row mapper now
consumes an exact `TrustedRecognizerInputRowV2` and current execution-freeze manifest,
then uses compact typed replay, the adjacent public-registry adapter, the frozen exact
derived bridge, and a closed decision/reason map to return a privately issued,
process-local, ephemeral V2 outcome plus the generic `PredictionBundle`. V1 types are
rejected fail-closed. The same constructed positive regression preserves decision,
bundle identity, family/binding/scales, and input/protocol/freeze-root parity. The
single-row mapping remains ephemeral and the structural archive is not a
durable trusted receipt. There is still no backed actual-960 production run,
recognizer runtime, scoring, or effect.

The independent V2 prediction-archive codec now freezes V2-only run-context,
record, archive, policy, and content identities under `HGP2PA2\0`. Its closed
16-field manifest binds the context, ordered input-row root, and ordered record
root; exactly 960 bounded accepted-JCS prediction records are length-framed in
input-archive wire order. The builder invokes the committed V2 row mapper in that
order and abstains atomically on contract-covered validation, mapping, encoding,
or public-decoding failures. Public decode verifies only structural
archive, canonical framing, closed record schema, and ordered row-root coverage.
The synthetic 1 ANSWER + 1 ANSWER_SET + 958 ABSTAIN fixture is structural codec
evidence, not an actual 960-case execution. Input membership, batch membership,
source projection/disjointness, single allocation, secret custody, execution-
manifest authority, derived mapping, runtime, capacity, origin/formal audit,
sealed eligibility, scoring, effect, and C1 remain false. The next slice is
`unsealed_prediction_evaluator_v2_structural_720_240_partition_replay`, followed
later by CLI, formal scoring, and actual unsealed replay.

Pairwise distinct IKM values do not attest independent generation. The raw-envelope
diagnostic verifies typed payload identity and direct transform replay but not batch
membership or secret padding; supplied-secret replay proves only that the supplied
authorities/run/IKM reproduce the bytes. All of these receipts remain
`NON_AUTHORITATIVE_MECHANICS_ONLY`. Consequently this is still not the complete trusted
RFC-8785 builder or a formal namespace/covert audit, it has no origin or one-shot
custody attestation, and it has not run on the formal corpus. No audit-pass claim
is available. The 1..1024 authority cap is enforced, but the 1024-authority
worst-case wall-time/RSS has not been qualified.

Until all frozen `standard_error` semantics and Student-t/Bonferroni conversion
rules are implemented and tested, formal selector input is strictly:

```text
allowed uncertainty kind = absolute_bound
standard_error = STANDARD_ERROR_UNSUPPORTED
```

This is an execution restriction, not merely a documentation preference.

The implemented exact compiler enforces it at the whole-bundle boundary: one
`standard_error` observation or one endpoint outside the literal-pinned 663-point
RationalValue grid yields an abstaining receipt with no partial compiled rows.
For `absolute_bound`, all arithmetic starts from exact `Fraction.from_float`
representations of the wire's normalized binary64 values and rounds only outward.
The root/identity exact bridge consumes that receipt without converting it to the
existing float selector interval. Its authoritative entrypoint accepts only the raw
bundle, theory, and adapter registry, then recomputes both the uncertainty receipt
and complete candidate grid internally. All six law families use conservative
natural rational intervals for residuals, tolerances, normalized scores, and the
structural margin. Explicit entity/role/quantity/channel/membership, transform,
observation, scale, edge, vector-width, total-component, candidate, adapter-scan,
exact-operation, and fraction-bit-length budgets fail closed before or during
evaluation. Before those hashes, a bounded recursive walk also requires every
nested bundle/theory/registry dataclass, enum, tuple, and primitive to have its exact
frozen type; subclasses cannot split compiled values from committed mappings, and
tree nodes, text, and integer bit lengths are capped. Every preflight rejection
occurs before authority content hashing and
receives no content root, run ID, or committed downstream receipt; only a
post-preflight compilation and decision carry complete provenance.

The next mechanics layer uses a content-addressed
`PublicTransformEvidenceBundleV2`. Eight distinct wire-operation certificate types
drive exact sparse interval or discrete kernels; no unknown operation is inferred
from its name or silently treated as identity. A separate authoritative derived
witness bridge reruns uncertainty and transform compilation internally, reconstructs
the complete strict-scope law-by-binding-by-scale-by-support-slice grid, and takes a
conservative hull over every slice at a scale before selection. Missing, ambiguous,
unused, or provenance-inconsistent observations fail closed without a partial grid.

This is not a complete preservation implementation. Declared unit ratios, weights,
sampling points, split/merge inverses, and coarse sparse matrices are checked only
under their narrow typed contracts; dimensioned derived verifier semantics and
multi-root merge are unsupported. The eight wire operations are not the same
taxonomy as the eight formal `PreservationTransform` classes, and no 496+76
preservation suite has been executed.

## Real blockers after the exact freeze

`ready_for_holdout_generation` and formal exit remain false because
implementation and external-evidence blockers remain:

- a bounded-binary64 projection mechanics slice, bundle-atomic exact
  RationalValue-grid `absolute_bound` compiler, and a provenance-bound six-law
  exact rational residual/tolerance/selector bridge are implemented. Typed exact
  kernels now cover all eight wire operations and a derived witness bridge consumes
  their outputs, but the narrow operation contracts, dimensionless verifier boundary,
  incomplete transform-to-law coverage and unexecuted formal preservation suite keep
  the broad projection compiler and typed evidence-to-prediction pipeline incomplete;
- fixed-envelope feature/statistics/invariance, keyed batch shuffle/ID/
  recanonicalization/native-provenance/secret-padding/replay, and strict typed
  direct/whole-batch replay mechanics are implemented only as non-authoritative
  receipts. Origin authentication, a complete trusted RFC-8785 builder, formal namespace-aware
  field/UUID audit, formal-corpus resource execution and independent formal audit
  are not implemented or executed;
- exact 40-hex external baseline revisions, image/SBOM digests, and artifact
  hashes are not registered;
- an independent latent generator has not produced validated 720 + 240 sealed
  artifacts or the 572 derived pairs;
- a functional recognizer CLI, signed minimal image, formal scoring/sealed evaluator,
  and detached isolation attestation do not exist as completed evidence;
- no durable signed append-only custodian/CAS ledger enforces one-shot use
  across processes and restarts;
- no validation version has passed, so sealed holdout generation is forbidden.

These are implementation and external-execution blockers, not unresolved quota
or statistical-design questions. Until they are satisfied, Phase-2B remains
shadow-only protocol/readiness infrastructure and cannot block or authorize
ACTIVE theory mutation.
