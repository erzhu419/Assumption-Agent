# Phase-2B actual-unsealed-960 generator construction profile v1

## Scope

This artifact is a static, non-generating construction contract.  The only
runtime operation defined by its companion module is fresh verification of
caller-supplied canonical profile bytes.  It contains no case constructor,
materializer, entropy source, identifier allocator, filesystem loader,
network/Docker operation, key or signature operation, receipt issuer,
commitment issuer, recognizer, or scorer.

The sole success disposition is
`PROFILE_VERIFIED_NOT_GENERATOR_NOT_GENERATED`.  It means that exact profile
bytes match this static contract.  It does not satisfy R06, authorize R14,
generate any of the 960 primary rows or 572 preservation variants, authenticate
an origin, or constitute execution evidence.  Verification results are
ordinary publicly constructible cooperative-computation DTOs; they are not
unforgeable.  A consumer must freshly verify the original bytes.

## Normative-status split

The following totals are derived from the existing exact freeze:

- 6 families × 2 scale cells × 60 main cases = 720;
- 12 cells × (10 low-overlap structural positives + 10 high-overlap
  structural negatives) = 240 semantic-conflict cases;
- 496 legal preservation variants + 76 invalid-transform controls = 572;
- per-cell case quotas `19/1/8/8/8/8/8`;
- per-cell margin quotas `21/18/12/9`;
- the eight-transform applicability and aggregate count matrix;
- absolute-bound uncertainty only and whole-batch rejection of
  `standard_error`;
- normalized scale-regret point estimate at most 0.05 and one-sided bootstrap
  upper bound at most 0.08.

The complete case-by-margin joint matrix, exact control oracles, private scale
class semantics, descriptor token-slot construction, preservation base
selection/no-reuse schedule, taxonomy-to-Wire operation mapping, disjointness
fingerprints, challenge-to-R08 binding, constant eligibility signatures, and future
receipt/manifest schemas, atomic seven-key durable CAS, final-root signature key
registry, and qualification-cache non-evidence policy are new prospective
companion decisions.  They are
not represented as historical freeze facts.

The canonical machine body carries a total `section_provenance_in_order`
table.  Every normative section is marked as existing/derived, new prospective,
Wire-implemented dependency, owner-authorization-derived, or a stated mixture;
no section inherits provenance from a document-wide label.

## Main case and margin matrix

Every family × private-scale cell has the same matrix:

| case type | clear `m>=3` | moderate `1<=m<3` | near `1/4<=m<1` | nonunique | total |
|---|---:|---:|---:|---:|---:|
| `unique_scale_answerable` | 11 | 8 | 0 | 0 | 19 |
| `admissible_scale_set_answerable` | 0 | 0 | 0 | 1 | 1 |
| `wrong_family_hard_negative` | 3 | 2 | 3 | 0 | 8 |
| `binding_counterfactual` | 3 | 2 | 3 | 0 | 8 |
| `scale_counterfactual` | 2 | 3 | 3 | 0 | 8 |
| `sign_or_invariant_break` | 2 | 3 | 3 | 0 | 8 |
| `insufficient_or_nonidentifiable` | 0 | 0 | 0 | 8 | 8 |
| **total** | **21** | **18** | **12** | **9** | **60** |

Singleton answerables use
`m_select = min(nonselected.lower) - selected.upper`.  All-fail controls use
`m_reject = min(candidate.lower) - 1`.  A unique closest negative must be
separated from every other candidate by at least `1/1024`.  Set-valued and
nonidentifiable cases use exact ties (`m=0`) rather than a forced top-1 choice.
No accepted construction oracle may contain `ERROR` or `INCONCLUSIVE`.

The set-valued row has one family/binding and exactly the passing scale set
`{S0,S1}`.  Each insufficient/nonidentifiable row has exactly two equal passing
groups that differ in family or binding, so it cannot be mistaken for a legal
scale set.  All five control classes have expected decision `ABSTAIN`.

## Semantic-conflict descriptor

Every constructive semantic descriptor has kind `natural_language`, exactly
six segments, and exactly 24 UTF-8 bytes per segment.  Each segment is one
lowercase ASCII token followed only by U+0020 to the fixed width; trimming is
right-strip-U+0020 only.  Segment payload is therefore exactly 144 bytes.

Main cases and low-overlap positives use the fixed first six neutral tokens:

```text
amber cedar cobalt driftwood ember flint
```

The complete neutral bank is:

```text
amber cedar cobalt driftwood ember flint garnet harbor indigo juniper
kestrel lantern marble nectar opal pebble quartz raven saffron topaz
umber velvet willow zephyr
```

High-overlap negatives use all six tokens from the relevant family row:

```text
F01 symmetry equivariance invariance permutation reflection rotation
F02 monotonicity ordering increasing decreasing isotonic rank
F03 conservation balance inflow outflow storage continuity
F04 additivity complementarity synergy interaction union combined
F05 locality markov blanket conditional neighborhood separation
F06 feedback stability disturbance response control mitigation
```

Overlap is exact set intersection with that six-token family lexicon divided
by six: low positives are `0/6`; high negatives are `6/6`.  The descriptor
contains no label, truth, case type, stratum, scale, filename, or order
metadata.  Low positives are clear-interior singleton answers; high negatives
are clear-interior all-fail abstentions.  Each ten-row half-cell has footprint
counts `3/3/2/2` for `P2_PAIR/P3_CHAIN/P4_STAR/PSET_AGGREGATE`.

This profile owns no challenge threshold, Wilson computation, or bootstrap
result.  A future exact R08 receipt must content-address the validation version,
challenge threshold table, one-sided Wilson contract, and paired-cluster
bootstrap contract.  Until that binding exists, all such evaluation remains
absent and R08 remains false.

## Scale and regret oracle

The two private construction classes are:

```text
S0_FINE_LOCAL        root
S1_COARSE_AGGREGATE  non-root
```

They never appear as public labels.  Every graph has two source scale UUIDs,
and the answer is projected through the exact private source-to-public UUID
mapping.  A legal scale transform creates fresh target scales and the exact
bijection `S0->T0, S1->T1`, retaining the fine-before-coarse order.  It is not
a hidden scale swap.

Each answerable main case must eventually carry two exact nonnegative rational
scale losses, one strictly positive exact normalizer, and an oracle scale set
equal to the exact argmin.  Normalized regret is
`(chosen_loss - oracle_min_loss) / normalizer`.  Set-valued rows have equal
losses and still require exact `ANSWER_SET`; joint exactness, not zero regret,
rejects strict subsets.  Controls and challenge rows do not silently enter the
main scale-regret denominator.  This profile defines the future oracle schema
only and evaluates no regret.

## Private UUID label projection

The prerequisite namespace order is exactly:

```text
bundle_id observation_id entity_id role_candidate_id quantity_id context_id
task_id scale_id aggregate_map_id transform_id
```

Every future private alignment link must have complete source-to-public mapping
rows for all ten namespaces, global injectivity, and a fresh public-envelope
cross-check.  Canonical private label bytes are projected only with the final
Wire single-row or source-order batch projection API.  Foreign namespace,
cross-row, stale mapping, and splice inputs reject atomically.  The mechanics
do not authenticate truth or origin.

## Preservation manifest

| transform | applicable families | legal | invalid |
|---|---|---:|---:|
| entity alpha renaming | F01–F06 | 72 | 12 |
| observation reorder | F01,F03,F04,F05 | 48 | 8 |
| irrelevant entity augmentation | F01–F06 | 72 | 12 |
| unit conversion | F01–F06 | 96 | 12 |
| coordinate affine | F01,F02,F05,F06 | 64 | 8 |
| equivalent aggregation split/merge | F03,F04,F05 | 48 | 6 |
| nontrivial scale map | F01–F06 | 60 | 12 |
| sign convention reparameterization | F02,F03,F06 | 36 | 6 |

The formal taxonomy is not the Wire operation enum.  Alpha renaming,
observation reorder, and irrelevant augmentation use private exact witnesses;
unit, coordinate, and split/merge use their typed Wire operations; scale maps
use temporal/spatial/sampling/coarse operations in a per-family `3/3/2/2`
split; sign reparameterization uses a one-dimensional coordinate affine map
`A=-1,b=0`, not a negative unit-conversion factor.

The ten nontrivial-scale-map ordinals are exact and split five per source
scale.  S0 ordinals 0–4 are temporal-sum/group-2,
spatial-weighted-mean/group-2, temporal-sampling/stride-2-offset-0,
coarse-sum/group-2, and temporal-weighted-mean/group-3.  S1 ordinals 5–9 are
temporal-sum/group-3, spatial-sum/group-2,
spatial-sampling/stride-2-offset-1, coarse-weighted-mean/group-2, and
spatial-weighted-mean/group-3.  The aggregate operation multiplicity is
exactly temporal/spatial/sampling/coarse `3/3/2/2`; all coefficients are exact
rationals, missingness rejects, aggregation/coarse boundaries are exact
partitions, and sampling uses `VALID_ONLY`.

Main slots use `p(f,s,k)=(17*k+11*f+7*s) mod 60`.  In fixed transform order,
each applicable operation consumes its legal base slots and then one invalid
base per scale.  A main base is never reused within a cell; the maximum is 54
of 60 in F05.  A future 572-row manifest must bind base, variant, taxonomy,
operation/witness, legal/invalid status, expected projected answer, and exact
oracle result.  This profile issues no such manifest.

## Eligibility and disjointness

The primary eligibility signature is fixed at 12 available observations, six
entities, five role candidates, and two scales.  The derived signature differs
only by one neutral padding entity.  Primary target multiplicities are family
`6x160`, construction scale `2x480`, binding `4x240`, answerable 360,
abstain 600, and decisions `ANSWER=348, ANSWER_SET=12, ABSTAIN=600`.

Every source must first adapt to a closed identifier-free canonical latent-case
form.  Exact and structural fingerprints are domain-separated SHA-256 over
RFC8785 bytes; structural fingerprints additionally clear the semantic
descriptor.  The primary 960 must have no exact or structural duplicate,
main/challenge must be disjoint, and primary rows must not intersect the
complete forbidden-source manifest.  A declared forbidden source that cannot
be completely parsed is `UNKNOWN -> NO_GO`.  Intended preservation equality is
allowed only through one explicit pair link; unlinked duplicate variants
reject.  No Bloom filter or quadratic pairwise corpus comparison is permitted.

## Authorization, five fresh secrets, custody, and replay

The profile binds the committed prospective owner authorization exactly:

```text
phase2b_c1_owner_actual_unsealed_960_authorization_v1_dd2cd6a98b429fe01f172daf16ea70404c91356e40e9e42daed4cb5416a5c50d
```

That binding is not a live revocation check, owner signature, or execution
authorization.  Its source/config content hashes are also frozen so a similarly
named replacement cannot be substituted.

A future live custodian must create exactly five independent, pairwise-distinct
32-byte secret values for `latent_generator_seed`, `answer_seed`, `K_shuffle`,
`K_id`, and `K_padding`.  R11 stores only purpose-separated domain-hash
commitments.  R14 later cross-binds those commitments to the verified launch
overlay and fresh campaign/attempt IDs.  Distinct values alone do not prove
statistical independence; a separate source attestation is required for each
purpose.  Reusing one raw value for two purposes rejects before commitment, and
the custody attestation must bind pairwise raw-value distinctness.  Repeating one
commitment under two purposes rejects before the durable store even though the
purpose-separated state-key hashes would differ.  Each of the five future
receipt rows binds its purpose, commitment, independent-source attestation root,
and custody-receipt root; raw values and manifest contents remain absent.

Generator source, the five-secret manifest, private latent cases, and the
private answer/alignment manifest remain custodian-only.  Neither recognizer nor
any baseline may receive generator source, seed material/manifests, or answer
plaintext.  Before its prediction is committed, the scorer also receives no
answer plaintext.  Post-commit scoring may receive only the answer/alignment
opening, never generator source or seeds.  A future custody-only replay must
reproduce the private/public roots byte-for-byte from the frozen source archive,
dependency lock, image, profile, Wire binding, verified launch overlay, fresh
campaign/attempt IDs, and all five secrets.

## Three UUID domains

The complete private source UUID set is every UUID leaf in every source graph,
not one representative per namespace.  The complete primary-public target set
is every target in the 960 source-to-public mappings.  The complete
variant-public target set is every fresh target in all 572 full-domain variant
renaming schedules.  These three sets must be pairwise disjoint both within
each of the ten namespaces and globally.  Each set is globally injective;
variant schedules cover every base-public UUID and may reuse no private or
primary-public UUID.

## Synthetic/dev R06 qualification contract

R06 qualification is a future, non-generating, synthetic/dev-only operation.
It must bind an exact source commit, source archive root, dependency-lock hash,
runtime image digest, and synthetic fixture root.  Two fresh-process replays
must be byte-identical.  Required fixtures include a quota-shaped synthetic 960,
the synthetic 572 preservation manifest, a 1024-case acceptance, and an atomic
1025-case rejection.  A closed cache record may bind its public synthetic seed
manifest, KAT report, replay roots, and measured resources, but it is explicitly
`NON_EVIDENCE_CACHE_ONLY`.  Even a syntactically valid, signed, all-pass cache
record cannot authenticate its own execution or satisfy R06.

The future limits are 16 logical CPUs, 1,800 seconds, 12 GiB maximum RSS, and
128 MiB output.  Actual input count must be zero.  Admission must ignore cached
pass/fail and resource values and freshly rerun every KAT twice under a qualifier
identity, source/archive/lock/image, entrypoint, trusted qualifier registry, and
admission-controller key pinned before execution.  The admission controller must
directly observe process exit and resources.  Neither cache nor fresh-admission
evidence was issued; `FUTURE_REQUIRED_NOT_ISSUED` remains in force and R06 is
false.

## Future generation receipt schema

The receipt status is exactly `FUTURE_REQUIRED_NOT_ISSUED`.  Its future wire is
a closed, typed, bounded RFC 8785 wrapper with no trailing LF, duplicate or
unknown fields, float/non-finite values, or integers outside the binary64-safe
range.  The body ID is domain-separated over the canonical body; nested roots
are independently domain-separated over authorization, profile, overlay,
campaign, attempt, purpose, and nested payload.

The receipt binds the owner authorization, readiness DAG, launch-overlay
receipt, fresh prefixed campaign and attempt content IDs, all R01–R13 evidence roots, generator
source/archive/lock/image manifests, all five secret commitments, the exact
private `720+240=960` partition, public Wire `720+240=960` partition, 572 public
variants, answer alignment, preservation/disjointness/scale-regret manifests,
and a deterministic replay root.  Private source, answer, and seed manifest
plaintext never appears; only content addresses do.

The durable store stage is one atomic seven-key transaction.  Its attempt key
is domain-separated over authorization, profile, overlay, campaign, and attempt;
that key moves generation `0→1` from absent and stores a
`new_unsigned_receipt_root` exactly equal to the body-signed root.  Five more
keys are separately domain-separated over each exact `(purpose, commitment)`
pair in frozen purpose order.  The seventh key is derived only from the
domain-separated ordered five-pair set—never authorization, profile, overlay,
campaign, or attempt.  Each of the six consumption rows moves
`UNUSED→CONSUMED` and binds its purpose, winning campaign, winning attempt,
body-signed root, and final receipt root; the full-set row uses purpose literal
`FULL_FIVE_COMMITMENT_SET`.

All seven keys must be pairwise distinct and all seven absent/unused
preconditions are checked before any mutation.  If any key was used, none of
the seven mutations applies and no store receipt or final root issues.  Thus a
second attempt reusing any same-purpose subset of one through five commitments
hits at least one durable per-purpose key and rejects atomically; exact five-way
reuse also hits the full-set key.  A full-set-only key is explicitly
insufficient because it would allow partial secret reuse.

The body-signed root covers every substantive body field before store output.
To avoid a root cycle, the atomic request-root normalization omits only the six
derived `winning_final_receipt_root` leaves while retaining every purpose,
commitment, winner campaign/attempt, and body root.  The request root fixes the
transaction ID; a deterministic store-receipt root binds that request,
transaction ID, `COMMITTED_ALL_SEVEN`, and exact key count seven.  The resulting
transaction root determines the final receipt root.  That final root is then
inserted into all six winner rows, freshly recomputed, and verified before the
single atomic commit.  All seven durable records require fresh readback before
purpose signatures.  A deterministic transcript alone is not execution
evidence without the store-purpose signature and that readback.

The final receipt root therefore covers the complete normalized atomic store
request, store receipt, status, and transaction root.  Store transcript
splicing is inside what is signed rather than merely adjacent to a signature.
A signed contiguous logical-time prefix orders R11, R12, R13, then R14; it is
not UTC authority.

The launch overlay must pin one exact purpose-key registry root and epoch.  Its
four rows map governance freeze, custodian generation, store CAS, and logical
time to distinct Ed25519 key IDs and distinct public keys, with a fresh
revocation snapshot and exact `ACTIVE_NOT_REVOKED` status.  Each signature
preimage is domain-separated and contains purpose, key ID, registry epoch,
registry root, and final receipt root.  Rogue registries, revoked keys, purpose
relabeling, CAS/store-receipt splices, foreign attempts, reuse of any one-to-five
same-purpose commitments, cross-purpose raw-value or commitment reuse, and
readiness-root splices reject atomically.  `standard_error_count` must be zero
and atomic success true.

No cache, admission evidence, receipt, root, commitment, signature, registry
verification, CAS transition, time token, campaign, or attempt is issued by
this verifier.

## Readiness boundary

R05 is the required predecessor of both R06 and R07.  R06 and R07 are parallel;
R08 requires both.  Profile verification does not satisfy R06.
Actual cohort materialization is permitted only at R14 after the independent
launch overlay has been verified and fresh identifiers have been allocated.  R15 is the
actual-corpus covert audit and R16 is inference admission.  This artifact
satisfies none of R06, R08, R14, R15, or R16.
It also does not satisfy R03 or R04, execute a Wire replay, verify custody or
technical-role independence, or issue either seed commitment.

## Final profile identity and Wire dependency binding

The final construction-profile ID is
`phase2b_actual_unsealed_960_generator_construction_profile_v1_3d5175cbfb3749ce7f225a55e3d3d045156adf53ef7aea726ea592d4aa82d531`;
the exact canonical config SHA-256 is
`8d14629f1e449d9e31bdc9a65b38375e3a994cd6c4a0b8e80006248d0d32b4aa`.
The config is RFC 8785 bytes with no trailing LF.

The profile binds Formal Wire V3 source, test, config, and documentation
SHA-256 values `c8120920f0e31b959043f63a0f672471fcbea50d05ca2cd8eaf7fd0c29089765`,
`253001d11d7603456b80cf43837ed71de567bfd2cd6613cc78b21ca23f7db0f0`,
`4f82ab060e2d53912cd17c8966a681a53d11e96ce3c6bb02ea3a42c5d2e85784`,
and `704a937967441ebbb9b2a77a865f83890e241d37031e2b0a87f1da2877b3456a`.
It binds graph-completeness policy
`phase2b_formal_case_graph_completeness_policy_v3_7585a9f9e58823d766906296905efcd24b1e350330ca5c3f64d75361e9ad90e5`
and field-classification manifest
`phase2b_formal_wire_field_manifest_v3_12916eefe10770f54c3d6c71f4b8b2c14ebbd671642e96ff4082dd298e5088dc`.
Those four files were freshly hash-checked, and the independent Wire red team
reported final static GO with P0=0 and P1=0.  The current 73-test Wire STOP is bound
as upstream implementation evidence but was not executed by this profile
verifier.  This is a final Wire dependency binding for this construction
profile only: it does not satisfy R03, R04, R05, or R06 and is not actual-corpus
or execution evidence.  Any Wire drift requires another profile-ID/config-SHA
repin.
