# Phase-2B rich formal trusted-wire V3

**Status:** `PRODUCTION_CANDIDATE_SYNTHETIC_QUALIFICATION_ONLY`
**Claim level:** `FORMAL_WIRE_PUBLIC_STRUCTURE_ONLY_NOT_KEY_TRANSCRIPT_NOT_COVERT_AUDIT_NOT_EXECUTION`

## 1. Result and boundary

`phase2b_formal_trusted_wire_v3.py` implements a closed, generator-facing
formal public-case schema and a fixed-envelope trusted-wire builder. It is not
a wrapper around the older V2 authority, because V2 carries six identifier
namespaces that are outside the immutable exact-ten freeze. V3 instead owns
the needed semantics inside ten frozen namespaces and non-UUID structured
values.

This implementation may establish only structural mechanics: closed-schema
validation, builder-input catalog checks, cooperative key-receipt consistency,
secret shuffle mechanics, public UUID assignment, padding, envelope replay,
and a non-public cooperative row alignment plus internal structural variant
context. A successful builder call
returns a private `FormalTrustedWireBuildSuccessV3`; its `public_batch` is the
only public archive DTO and contains neither alignment nor variant context. It does
not load or generate actual-960,
run Docker, authenticate key origin, establish independent key generation,
run a recognizer, execute metadata-predictivity statistics, complete a covert
audit, reveal answers, or score predictions.

## 2. Closed rich case graph

Every public case contains all of the following typed records:

1. entities and their role-candidate scope;
2. quantities with a value kind and seven exact SI exponents;
3. contexts with closed temporal/spatial reference kinds, coordinate labels,
   exact temporal/spatial supports, and orientation;
4. observations with entity/role/quantity/context/scale references, ordered
   component descriptors (label, semantic axis, and value role), exact
   rational or Boolean values, uncertainty, and
   recomputed public provenance;
5. a task target;
6. a dedicated closed semantic descriptor with an evidence-carrier kind and
   one through sixteen already-NFC UTF-8 evidence segments;
7. scales and an acyclic, root-complete aggregation graph;
8. aggregate maps, structured component references, exact groups, and
   quotient labels;
9. exactly the used transform contracts, including typed sparse rows,
   discrete mappings, and an operation-tagged certificate union for all eight
   frozen transform operations;
10. the recognizer theory/family/law/observable registry; and
11. an eligibility mask.

`FormalSemanticDescriptorV3` has exactly `kind` and `segments`. Kinds are
`natural_language`, `symbolic`, or `mixed`; they describe the evidence carrier,
not a positive/negative or overlap stratum. Each segment is NFC before entry,
allows ordinary U+0020 padding for later fixed-width profiles, rejects Unicode categories
`Cc/Cf/Cs/Co/Cn/Zl/Zp`, is at most 1,024 UTF-8 bytes, and the
descriptor total is at most 4,096 bytes. RFC 8785 performs no Unicode
normalization. UUID-shaped or `sha256:` digest tokens anywhere in a segment
reject, preventing identifier/root duplication through evidence text.
The restricted benchmark-text profile is pinned to Python 3.10.12 Unicode
13.0.0, the official UCD file hashes, 4,173 frozen
`Default_Ignorable_Code_Point` scalars, the exact range-table hash, and a
reject-all-noncharacters rule. It is deliberately not an unrestricted
multilingual-text claim.
Challenge label, overlap stratum, private truth, filename, and
order-metadata fields remain absent. Whether a descriptor meets a later
semantic-overlap challenge is deliberately not frozen or claimed here.

The identifier-leakage rule is not limited to the semantic descriptor. An
immutable DTO/field routing manifest covers every direct `str`, optional
`str`, and `tuple[str, ...]` field. Free semantic and binding text uses the
same NFC/UCD-13 safety profile and rejects embedded canonical UUIDs,
`sha256:<64hex>`, and bare 64-hex tokens. Typed UUID, digest, decimal, schema,
and content-ID fields use field-specific exact validators. A newly added or
otherwise unrouted string field fails closed, and this routing manifest is
content-bound into the graph-completeness policy ID.

All nullable fields use exactly `{"available": bool, "value": ...}`. An
unavailable value is exactly `{"available": false, "value": null}`. There is
no parallel string missingness state. Raw binary64 values, Boolean-as-integer
values, unknown fields, missing fields, unresolved references, cyclic scale
graphs, incomplete or mismatched typed certificates, duplicate kernel terms, aggregate-group
partitions that disagree with transform inputs/outputs, cross-group kernel
terms, reducer-clause or exact-coefficient disagreements, and unused per-case
transforms are rejected. Context support availability/dimensions and numeric
uncertainty vector length/nonnegativity are also exact graph invariants.
Numeric observations cannot encode missingness: a missing component must use
the explicit unitless Boolean/missing-compatible quantity sentinel, a missing
value role, and `not_applicable` uncertainty. Boolean/missing-compatible
quantities have zero SI exponents and exact unit sentinel `(none, 1, 0)`;
numeric quantities cannot use that sentinel.

Every used edge is projected into the separately versioned pure exact-transform
core. The projection binds full clock/frame reference semantics, coordinate
labels, inline unit scale/offset, component roles, group boundaries, sparse
row input/output identities, and claimed target values. The core freshly
checks all eight operation equations and exact absolute-bound propagation
`r_out = sum(abs(a_i) * r_i)`. Temporal, spatial, and coarse weighted means
state explicitly that coefficients are caller-declared rather than inferred
from durations or volumes. Split/coarse output-label bindings are determined
by the public group inputs, output component label, and quotient label; there
is no generic clause bag or unlabeled dense-matrix entry point.

The bound core is
`hegel-machine-phase2b-pure-exact-transform-core/1`, source SHA-256
`e0fa517b4275a1d2daee1fd1f877b61a8f79ca745a702979615b2130e53bdc5f`.
Wire invokes `verify_exact_transform_core_outputs_v1`, so validation includes
fresh computation and comparison against every claimed public output. A core
`COMPLETE` result remains a computation DTO only; it is not authority,
provenance, key-origin, or execution proof.

The exact UUID namespaces are:

```text
bundle_id observation_id entity_id role_candidate_id quantity_id
context_id task_id scale_id aggregate_map_id transform_id
```

Clock and frame semantics are closed context fields; unit semantics are an
inline semantic binding, exact scale/offset-to-SI, and SI exponents; a component is an observation/quantity/context/scale plus
bounded axis/index; quotient semantics belong to an aggregate map. Therefore
`source_channel`, `clock`, `frame`, `unit`, `component`, and `quotient_class`
never become public UUID namespaces.
The certificate field `quotient_aggregate_map_id` is explicitly routed to the
frozen `aggregate_map_id` namespace; it is not a hidden eleventh namespace.

## 3. Source-to-public boundary

`FormalSourceCaseV3` contains private construction symbols. The current
source DTO requires canonical UUIDv4 syntax so the same UUID-closure validator
can inspect every use site, but those values are private local symbols, not
public IDs. A source UUID has exactly one source-case owner. Reusing the same
namespace/UUID pair in another case rejects the whole batch.

The builder canonicalizes all set-like source order, sorts the private cases by
their canonical construction content, applies the batch-global secret shuffle,
and only then assigns public UUIDv4 values. Every UUID use site is renamed by
namespace. Replay rejects both cross-namespace reuse and reuse of any public
UUID in a second envelope. Successful batch and replay DTO constructors do the
same check from freshly decoded envelope bytes, require every embedded batch
context to equal the cached batch context, and enforce the realizable collision
retry bound; directly constructed or replaced cache objects cannot bypass it.

The frozen ID input is exactly:

```text
domain || run_id || u16(namespace_len) || namespace || u64(counter) || u8(retry)
```

Counters are namespace-local; assigned UUID values are globally unique. Retry
values 0 through 10 are bounded and tested. The complete private source UUID
set is also treated as occupied, so no public UUID may equal a source UUID in
any case or namespace.

## 4. Global transform catalog

The caller cannot supply a decorative digest. The trusted builder accepts a
private `FormalGlobalTransformCatalogV3` and its
`FormalGlobalTransformCatalogBindingReceiptV3`. The catalog root is recomputed
from the RFC 8785 canonical sorted catalog entries, and the binding receipt is
content-address checked.

Each entry commits one exact canonical source
`FormalSourceTransformUseV3(aggregate_map, transform)`, including its source
UUID symbols. Every source aggregate-map/transform use must be a member before
public renaming. This digest is intentionally not claimed to be alpha-invariant
and is not a reusable semantic whitelist across independently UUID-named
corpora. Public bytes contain only the single literal field
`/batch_context/transform_catalog_root`; no per-case catalog, unused-transform
list, or catalog receipt is serialized.

Standalone envelope decode and batch replay prove structure and the bound root
value only, so their `catalog_membership_verified` value is always false. The
cooperative builder-input membership bit exists only on the non-public full
build success after replay of its public batch, private alignment, immutable
base bytes, and private catalog evidence. It is not an origin-authenticated
claim. There is no exported or accepted standalone variant capability.

## 5. Canonical envelope and channel projection

The wire imports the finalized versioned RFC surface directly:

```python
canonicalize_rfc8785_v1
decode_and_require_canonical_rfc8785_v1
RFC8785Error
```

Each envelope is exactly 65,536 bytes: an 80-byte header, canonical public
payload, and at least 32 padding bytes. The header binds payload length,
payload SHA-256, and padding SHA-256. Strict decode requires canonical RFC
8785 bytes, exact closed fields, recomputed observation/case provenance, all
graph invariants, and exact cached DTO replay. A `DecodedFormalEnvelopeV3`
constructor freshly parses and normalizes its raw canonical payload bytes,
reconstructs the context and public case, and exact-compares those objects plus
the payload mapping, classified leaves, and channel projection against every
cached field.
An iterative preflight caps JSON nesting at 64 and container tokens at 16,384
before recursive JSON decoding. Public decode, replay, build, and variant
boundaries convert recursion or memory exhaustion into fixed small fail-closed
errors rather than diagnostics proportional to the rejected input.

`ChannelProjectionV3` is immutable and contains only channel observations:
payload and padding lengths, padding digest/prefix/suffix, absolute JSON
pointers for every namespace UUID and nullable missingness wrapper, record and
component order tokens, observation-array paths, eligibility-mask tokens, and
public provenance. The field manifest classifies every concrete public leaf
exactly once as `EVIDENCE_BEARING` or `CHANNEL_ONLY`; its content ID binds the
full nested DTO field manifest and path-pattern precedence.
Semantic-descriptor leaves follow the closed-graph default and are
`EVIDENCE_BEARING` in the public leaf partition. Separately, the immutable
channel projection exposes descriptor kind/count, per-segment UTF-8 length,
trailing U+0020 count, boundary scalars, lexical-token count, ordered
non-alphanumeric scalar sequence, Unicode-category run signature, and segment
order digest. This makes representation choices observable to the later
auditor; Wire does not claim descriptor representation or semantics are
covert-channel free.

## 6. Keys and private audit alignment

Shuffle, ID, and padding each require a distinct 32-byte IKM. The transcript
receipt binds their commitments and rejects equality with any latent or answer
commitment. This proves only consistency of the supplied transcript; it does
not prove key-source independence or origin. Public batch and public replay
keep `key_source_transcript_verified=false` and
`key_source_independence_verified=false`. A caller retaining the run ID, three
key sources, source cases, and exact catalog can call
`verify_formal_trusted_wire_build_success_v3`; that verifier recomputes the
receipt and deterministically rebuilds shuffle, UUID allocation, padding,
envelopes, alignment, and the internal variant context before returning true.
The public batch ID binds the exact key-source receipt ID in addition to the
batch context and ordered envelope IDs. Public replay accepts the receipt ID
explicitly and binds it into its batch ID while keeping transcript verification
false.

After successful shuffle and envelope construction, the builder returns a
non-serialized `FormalPrivateAuditAlignmentV3` as a cooperative content-linkage
DTO. It has no hidden module token and its constructor is not an unforgeable
issuer. Each public-order link binds batch/run/key context, envelope
SHA-256, source commitment, source-row index, label-binding ID, and a complete
mapping of every source UUID across all ten namespaces to its public UUID. The verifier fresh-replays the
public batch and requires namespace-preserving exact public coverage and global
injectivity. Missing, duplicate, stale, spliced, foreign, or cross-namespace
views reject. Public batch bytes, `asdict(public_batch)`, and standalone replay
contain none of this private state.
Each per-case private UUID map is additionally capped at 40,960 entries, which
is ten frozen namespaces times the per-record schema cap. This is a collection
budget, not a claim that each case contains exactly ten UUID rows.

`project_private_source_label_to_public_v3` projects one canonical RFC 8785
private-label object; `project_private_source_labels_to_public_order_v3`
projects and shuffles a full source-order label batch with one success
validation. UUID leaves must use a frozen namespace field and be present in the
issued row map. `issue_projected_private_audit_rows_v3` combines projection,
source-commitment/label-binding issuance, and public-order alignment so each
returned private row carries the exact projected label bytes. The private-row content seal establishes only fresh byte/source
linkage. Neither the DTO nor these digests authenticate origin, establish
cryptographic unforgeability, or prove that a
label is semantically true.

## 7. Public API

The production-candidate surface is:

```python
issue_formal_global_transform_catalog_v3(source_cases=...)
verify_formal_global_transform_catalog_binding_v3(catalog=..., binding_receipt=...)
build_formal_trusted_wire_batch_v3(
    source_cases=...,
    global_transform_catalog=...,
    transform_catalog_binding_receipt=...,
    run_id=...,
    key_sources=...,
    key_source_receipt=...,
)
# -> FormalTrustedWireBuildSuccessV3(public_batch, private alignment, private context)
verify_formal_trusted_wire_build_success_v3(
    build_success=..., run_id=..., key_sources=..., source_cases=...,
    global_transform_catalog=..., transform_catalog_binding_receipt=...,
)
decode_and_audit_formal_envelope_v3(envelope)
replay_formal_trusted_wire_batch_v3(envelopes, key_source_receipt_id=...)
make_formal_variant_payloads_v3(build_success=..., schedule=...)
normalize_formal_variant_payload_v3(
    payload, build_success=..., schedule=..., envelope_id=...
)
reencode_formal_variant_payload_envelope_v3(
    payload=..., padding_bytes=..., build_success=..., schedule=..., envelope_id=...
)
reencode_formal_variant_batch_v3(build_success=..., schedule=...)
reencode_formal_variant_suite_v3(build_success=..., schedules=(...))
verify_private_audit_alignment_v3(batch=..., alignment=...)
issue_formal_private_audit_rows_v3(build_success=..., private_payloads_in_source_order=...)
issue_projected_private_audit_rows_v3(build_success=..., canonical_source_label_bytes_in_source_order=...)
align_private_rows_to_public_v3(...)
project_private_source_label_to_public_v3(...)
project_private_source_labels_to_public_order_v3(...)
```

There is no raw public root-only reencoder. A `FormalVariantScheduleV3` must
provide an exact case permutation, an exact per-envelope observation
permutation, and either identity or a total global namespace-preserving UUID
bijection. Each variant API requires the full build success. It first copies
all primitive bytes, strings, and tuples once, reconstructs and validates a
fresh public batch, alignment, catalog, and base snapshot, and then reads only
that snapshot. The internal context holds immutable canonical payload and
padding bytes and is content-bound to the private alignment ID, catalog root,
catalog receipt, and batch. It inverse-maps UUIDs, restores observation order,
and requires exact base-payload equality before reencoding. Self-minted
contexts, mutable aliases, post-validation replacement, and semantic
transform/catalog-root mutation are outside the variant surface.
Every explicit variant target UUID must also be disjoint from the complete
private source UUID domain captured by the freshly validated alignment. Thus a
total public-to-source UUID schedule cannot turn a variant into a source-ID
disclosure channel; rejection occurs before any envelope is framed.
For a nonempty explicit full rename, the target domain is also disjoint from
the base public UUID domain; the empty rename remains the sole identity form.

`reencode_formal_variant_suite_v3` is the only batch-of-schedules entrypoint.
It snapshots the complete primitive schedule tuple before build state, then
reconstructs and validates the full public batch, key receipt, private
alignment, private catalog evidence, immutable base bytes, and inverse catalog
membership exactly once. Each snapshotted schedule independently repeats the
case/UUID/observation permutation checks, inverse semantic equality,
provenance normalization, framing, canonical candidate-wrapper replay, padding
equality, context equality, global UUID ownership, and retry-bound closure.
Pure case-order schedules and rows unaffected by an observation permutation
reuse immutable base payload, padding, and envelope bytes that were fully
replayed during the one-time build snapshot. The function has no generator,
file write, validated handle, token, or reusable cache capability; failure
returns no partial result.

The suite is bounded to 80 unique schedules, 81,920 output envelopes,
5,368,709,120 output bytes, and 8,388,608 primitive schedule tokens. Its
resource preflight first requires the outer schedule, case-order, UUID-row,
and observation-row containers to be exact tuples and applies O(1) length
caps. For each observation-order row, preflight may inspect only the exact DTO
type and the O(1) length of its exact observation-ID tuple while checked-adding
the complete primitive-token budget. It does not traverse any case ID, UUID
row field, or observation ID value until that whole budget has passed.
Oversized or forged containers therefore fail closed before those element
contents are accessed.

Its private runtime result stores immutable nested envelope tuples. Every
batch also carries a freshly reconstructed exact schedule preimage; its base
envelope order must equal that schedule's case-order tuple. `schedule_root` is
not a constructor input: the batch freshly derives it from the preimage and
rejects any preexisting cached root drift. The suite then derives ordered
schedule roots, per-batch envelope roots, and one suite root. These
roots content-address atomic candidate bytes; they do not prove structural or
semantic verification, key origin, catalog authority, covert audit, or
execution. The suite deliberately does not repeat exact-transform-core work on
every output. Every consumer must call
`decode_and_audit_formal_envelope_v3` on every returned envelope before passing
it to a recognizer or auditor. The public decoder is the verification boundary;
all suite result claim flags remain false. Directly constructed batch and
suite DTOs freshly recompute schedule roots, variant envelope IDs, envelope
hashes, per-batch roots, and the suite root, and reject any caller-supplied
derived cache that differs.

## 8. Atomic failures and qualification evidence

Any `standard_error` observation rejects the whole batch before source roots,
shuffle, ID allocation, envelopes, or batch roots. Counts outside 1 through
1,024 reject with empty envelopes and no batch ID. Payloads beyond the fixed
envelope budget, catalog membership failures, key overlap, UUID ownership
failures, and post-build replay failures also return atomic empty rejection.
Noncanonical aggregate-group order rejects before coarse quotient labels can
be positionally rebound or normalized. When a valid graph is globally renamed,
canonical group reordering rebuilds the coarse quotient-label tuple from each
group's explicit output-bound quotient label rather than retaining stale
positions.

The focused test suite covers RFC duplicate/noncanonical/unknown/missing/type
failure, the full nested DTO manifest, graph closure, eight typed operation certificates,
eight separate actual build/decode/public-replay paths,
catalog membership, source-order erasure, same-local-ID attacks, purpose-key
KATs, collision retry boundaries, standard-error atomicity, payload and
1,024/1,025 capacity boundaries, alignment attacks, 32 generated UUID renames,
16 genuinely distinct observation-order replays, 32 genuinely distinct
five-case order schedules, combined case/observation/UUID variants, private
label projection attacks, semantic descriptor/UCD-13/representation-channel
closure, fresh caller-evidence verifier attacks, fresh header/cache replay,
fixed envelopes, and exact-one leaf classification. Additional attacks cover
every routed non-ID text leaf, all eight operation-literal paths under
`standard_error`, source/public UUID collision, receipt replacement, deep
bounded JSON, self-minted variant contexts, immutable-base mutation, and
after-validation TOCTOU hooks. It also covers direct decoded/batch/replay cache
splices, global cross-envelope UUID reuse, excessive retry counts, the exact
JSON container-token boundary, the private mapping cap, and a full-domain
variant map back to private source UUIDs. Suite tests independently freeze all
three root domains, reject duplicate/partial/base-domain schedules and bounded
resource attacks, exercise schedule/build TOCTOU and atomic failure, prove one
full-build snapshot per suite, explicitly public-decode every returned envelope
in the focused consumer test, execute the exact 80-schedule surface over a
32-case rich batch, and execute the maximum 80-by-1,024 surface (81,920 logical
fixed envelopes) under the implementation resource bound. Those are
implementation tests, not an independent corpus or covert-audit qualification.
The independent qualification layer must retain all execution, corpus,
recognizer, scorer, and audit claims as false until it runs its own bounded
evidence procedure.
