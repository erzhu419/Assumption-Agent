# Hegel Machine Phase-3A-Q0 Exact Quotient Semantics Engineering Freeze v1

## 0. Authority, identity, and present state

This document is the executable engineering freeze for the first qualification
stage of the post-shrink-6 quotient direction.  It implements the direction of
the following byte-exact normative document without changing that document:

```text
human_document_id = hegel-phase3-post-shrink6-quotient-direction-v1
path = Hegel Machine/docs/Hegel_Machine_Phase3_Post_Shrink6_Quotient_Direction_Decision.md
sha256 = 1df8d3ff3ede2cbead98e7901a3e82b91c460ad1d5eb0d1af78938e7b2d23b95
```

The identities frozen by this engineering amendment are:

```yaml
dsl_version: hegel-old-dsl-v1.6.0
dsl_freeze_version: hegel-freeze-p2b-p3-v1.6.0
closure_semantics_version: hegel-quotient-closure-v1.0.1
q0_freeze_version: hegel-freeze-p3a-q0-v1.0.1
q0_qualification_id: hegel-phase3a-q0-exact-quotient-qualification-v1
q0_projection_id: hegel-q0-micro-projection-v1

historical_source_y_commit: 5217568303d5c7f902682c092750f637c64f080a
historical_evidence_z_commit: ea98157f5d6eb2930ab28dda8f3a6839b343673c

q0_execution_state: NOT_RUN
q0_readiness: 0/14
q0_formal_roots: null
q0_formal_roots_generated: false
qualified_q0_class_archive_root: null
q0_saturation_receipt_root: null
q0_projection_manifest_root: 2f39aa248f1305eeaf20a724f6d690cf2b13003f86620d09d2753815831f7ad1
q0_semantic_binding_root: b7ec5e860a007469b8a1b3930f17c130f59a800d2a832dfd438d18a75538ff99
q1_execution_state: NOT_RUN
q2_execution_state: NOT_RUN
target_truth_access_allowed_in_q0: false
split_access_allowed_in_q0: false
role_evaluation_allowed_in_q0: false
historical_m3_gate_count_inherited: 0
```

Changing closure representation does **not** create `hegel-old-dsl-v1.7.0`.
The language remains v1.6.0.  The new identity belongs to closure semantics and
qualification, not to the grammar.

The historical syntactic M3 execution and its `DSL_TOO_LARGE` terminal result
remain immutable evidence.  `NOT_RUN` in this document means that the new Q0,
Q1, and Q2 quotient track has not run; it does not say that historical M3 never
ran.

---

## 1. Exact claim boundary

Q0 is a finite, target-blind mechanism qualification.  It proves at most:

> Independent Python and Rust implementations, followed by host replay,
> produce the same exact behavior quotient and the same MDL-aware continuation
> frontier as an exhaustive syntax-to-quotient oracle on the frozen four-row
> Q0 micro projection under the frozen depth-two, four-node grammar projection.

Q0 does not prove any of the following:

- complete quotient closure of `hegel-old-dsl-v1.6.0` on the 480-row odd
  universe or the 85-row sink universe;
- Q1 readiness, Q1 execution, or Q1 completion;
- odd-cardinality or sink role membership;
- `OUTSIDE_FROZEN_QUOTIENT_CLOSURE`;
- autonomous scale inference, relation invention, or an MDL invention
  certificate;
- that the finite-universe equivalence is equality on every possible input;
- that any Q0 diagnostic root is an M3 formal root.

A Q0 PASS is named exactly:

```text
DUAL_EXHAUSTIVE_ORACLE_EQUIVALENCE_PASS
```

It must never be abbreviated to `COMPLETE`.

---

## 2. Q0 object and enum namespace

Q0 uses a new append-only namespace.  It does not reinterpret any existing M3
tag, `EquivalenceModeId`, state ID, receipt, or archive root.

### 2.1 Numeric object tags

| Tag | Object |
|---:|---|
| `0x3601` | `BehaviorBlobV1` |
| `0x3602` | `FutureAdmissibilitySignatureV1` |
| `0x3603` | `FrontierEntryV1` |
| `0x3604` | `QuotientClassRecordV1` |
| `0x3605` | `Q0SaturationReceiptV1` |
| `0x3606` | `Q0ProbeInputV1` |

These tags are reserved for this Q0 namespace.  Tombstones, if ever needed,
must not be reused.

### 2.2 Output sort and cell tags

```text
OutputSortId:
  1 BOOL
  2 BIT
  3 SIGN
  4 BOUNDED_INT
  5 RATIONAL_VALUE

CellTag:
  0 UNDEFINED
  1 DEFINED
```

### 2.3 Q0 terminal status

```text
0 NOT_RUN
1 RUNNING
2 DUAL_EXHAUSTIVE_ORACLE_EQUIVALENCE_PASS
3 INCONCLUSIVE_RESOURCE_LIMIT
4 FAIL_SEMANTICS_MISMATCH
5 FAIL_IMPLEMENTATION_DISAGREEMENT
```

No Q0 status is an alias for an M3 state or closure status.

### 2.4 Resource-guard IDs

```text
1 RAW_OPERATOR_APPLICATIONS
2 CANONICAL_SYNTAX_PROGRAMS
3 BEHAVIOR_CLASSES
4 TOTAL_FRONTIER_POINTS
5 FRONTIER_POINTS_PER_CLASS
6 SATURATION_ROUNDS
7 WALL_TIME
8 RESIDENT_MEMORY
9 OUTPUT_BYTES
10 TOTAL_CONTINUATION_BANK_POINTS
11 CONTINUATION_BANK_POINTS_PER_CLASS
```

The ID is evidence of which frozen guard prevented completion; it is never a
ranking or a license to retry with an unpublished larger value.  The ordered
`(uint ID, ASCII name)` registry is part of `Q0ProjectionManifestV1`; every
`INCONCLUSIVE_RESOURCE_LIMIT` endpoint error must carry exactly one registered
`resource_guard_id`.

---

## 3. Behavior identity and explicit bottom

### 3.1 Canonical behavior cells

```text
UndefinedCell = [0]
DefinedCell   = [1, CanonicalValue]
```

`CanonicalValue` is sort-specific:

| Sort | Exact canonical payload |
|---|---|
| `BOOL` | CBOR `false` or `true` |
| `BIT` | CBOR uint `0` or `1` |
| `SIGN` | CBOR int `-1`, `0`, or `1` |
| `BOUNDED_INT` | CBOR integer in `[-8,8]` |
| `RATIONAL_VALUE` | reduced `[numerator, denominator]` belonging to `{Fraction(n,d) | -64 <= n <= 64, 1 <= d <= 8}` |

Python `bool` must not pass an integer check accidentally.  In particular,
`false`, Bit `0`, BoundedInt `0`, Sign `0`, RationalValue `0/1`, and bottom are
six different typed states.

### 3.2 Behavior blob

```text
BehaviorBlobV1 =
[
  1,
  0x3601,
  b"hegel-q0-behavior-blob/1",
  input_signature_id,
  frozen_universe_root,
  output_sort_id,
  row_count,
  [cell_0, ..., cell_(row_count-1)]
]
```

The authoritative semantic identity is:

```text
behavior_id = ContentHash(
  "HEGEL/Q0/BEHAVIOR_ID/V1",
  BehaviorBlobV1
)
```

The complete canonical bytes, not only the digest, must be retained during
collision checking.  Two distinct preimages with the same SHA-256 digest cause
an unconditional `FAIL_SHA256_PREIMAGE_COLLISION`.

### 3.3 Exact quotient relation

For one frozen input signature and universe, two admitted ASTs are equivalent
if and only if their `BehaviorBlobV1` canonical bytes are identical.  Thus:

- output sort is part of equality;
- universe identity and row order are part of equality;
- every defined value is bit exact;
- bottom is explicit and sort-preserving through the blob's output sort;
- bottom at different row positions yields different behavior;
- undefined bitmaps are derived audit data and cannot replace behavior bytes.

Q0 symbolic or syntactic normalization may reduce work, but it is never a
semantic identity.  Every quotient merge must end in exact behavior-byte
equality.

---

## 4. Frozen production observation-adapter rules

The production Odd and Sink adapters are target-independent evaluators.  Their
implementation lives in separately source-bound code, but the following rules
are normative.  A missing observable produces typed bottom.  It must never be
imputed as `false`, `0`, an empty set, or an empty measurement.

```text
authoritative_adapter_source_path =
Hegel Machine/src/hegel_machine/phase3_q0_input_adapter_v1.py
```

Gate 3 must bind the committed source bytes and independent replay evidence;
the path alone is not an implementation qualification.

### 4.1 Odd input signature

For `OddInputV1`:

- `set_size` is the exact size in `5..8`;
- `bit_at(i)` returns the exact Bit for an in-range slot;
- out-of-range `bit_at(i)` returns typed Bit bottom;
- `scalar_const` returns its frozen exact rational value;
- every aggregate leaf returns its declared output-sort bottom because the
  Odd input carries no quantity measurement, scope metadata, or orientation;
- `context_flag(c)` returns Bool bottom for every context ID;
- `task_flag(t)` returns Bool bottom for every task ID;
- no implicit Bit-to-RationalValue coercion is introduced by the adapter.

### 4.2 Sink input signature

For `SinkInputV1(a,b,c,d)`:

- the observed entity-set cardinality is exactly four, so `set_size = 4`;
- `bit_at(i)` returns typed Bit bottom for every index because this signature
  carries no Bit sequence;
- the four q0 measurements are exact rationals `a/1,b/1,c/1,d/1`;
- role IDs are `0,1,2,3` in field order;
- orientations are `+1,+1,-1,-1` in field order;
- only `scope_id = 3`, `control_volume_all_observed_v1`, has frozen membership
  for all four entities;
- scope IDs `0`, `1`, and `2` return typed aggregate bottom; the adapter does
  not infer all/primary/boundary membership from field names;
- q1 is missing and therefore every q1 aggregate returns typed bottom;
- every non-empty scope extension returns typed bottom because its context
  observables are not carried by the production input;
- with empty extension, scope 3, and q0, active map 0 `sum`, active map 1
  `count_nonzero`, and active map 5 `signed_balance` are evaluated exactly;
- an exact rational aggregate result outside the frozen RationalValue domain is
  bottom;
- `context_flag(c)` and `task_flag(t)` return Bool bottom for every ID;
- strict bottom propagation applies to every parent operator.

The designated signed-balance expression evaluates to exact zero on the frozen
sink universe under these rules, but Q0 does not execute the sink role or read
its truth table.

---

## 5. Production universe ordering and Q0 isolation

The production universe ordering remains unchanged:

- Odd: set size ascending from 5 through 8, then bitstring numeric value
  ascending, bits MSB-first;
- Sink: enumerate `a,b,c,d` over `0..4` in tuple-lexicographic ascending order
  and retain exactly rows satisfying `d = a + b - c`;
- `universe_index` is contiguous from zero and strictly ascending.

These rules yield exactly 480 Odd rows and 85 Sink rows.  Q0 neither regenerates
nor reorders either production universe.

Q0 does not mount production truth rows, split assignments, role-match records,
or any target-conditioned score.  Existing truth and split commitments may be
known historically; the precise isolation claim is that Q0 admissible inputs,
runtime mounts, imports, and branch predicates exclude them.  Q0 must not claim
that no person or earlier process has ever seen a committed truth table.

---

## 6. Q0 four-row micro projection

### 6.1 Projection identity and structural limits

```yaml
projection_id: hegel-q0-micro-projection-v1
input_signature_id: 0x7001
input_object_tag: 0x3606
row_count: 4
leaf_count: 15
maximum_ast_depth: 2
maximum_ast_node_count: 4
maximum_top_level_clauses: 2
maximum_aggregate_leaves: 1
maximum_scalar_parameter_occurrences: 3
maximum_scope_clauses: 2
maximum_distinct_bit_slots: 4
boolean_composition: AND2_only
```

`input_signature_id = 0x7001` identifies only the ordered composite Q0
qualification universe.  It is not a third production input schema and does
not replace the source signature carried by any row.  Every row is decoded by
the same production Odd or Sink adapter used outside Q0.

### 6.2 Exact four rows

The row order and decoded canonical source objects are exactly:

| Row | Source signature | Source tag | Canonical typed-input object |
|---:|---:|---:|---|
| R0 | `1` (`OddInputV1`) | `0x3401` | `[1, 0x3401, b"hegel-odd-input/1", 5, [0,1,0,1,0]]` |
| R1 | `1` (`OddInputV1`) | `0x3401` | `[1, 0x3401, b"hegel-odd-input/1", 8, [1,0,1,0,1,0,1,0]]` |
| R2 | `2` (`SinkInputV1`) | `0x3402` | `[1, 0x3402, b"hegel-sink-input/1", 0, 0, 0, 0]` |
| R3 | `2` (`SinkInputV1`) | `0x3402` | `[1, 0x3402, b"hegel-sink-input/1", 4, 1, 2, 3]` |

For R3, `3 = 4 + 1 - 2`, so the row satisfies the frozen Sink input balance
constraint.  The Q0 universe wrapper preserves, for each row, its source
signature ID, numeric source tag, and complete canonical source object.  It
must not translate either input into a synthetic primary/auxiliary entity
schema.

The composite input object and its identity are frozen as:

```text
Q0ProbeInputV1 =
[
  1,
  0x3606,
  b"hegel-q0-probe-input/1",
  0x7001,
  4,
  [
    [1, 0x3401, <canonical R0 object>],
    [1, 0x3401, <canonical R1 object>],
    [2, 0x3402, <canonical R2 object>],
    [2, 0x3402, <canonical R3 object>]
  ]
]

q0_micro_universe_root = ContentHash(
  "HEGEL/Q0/PROBE_UNIVERSE_ROOT/V1",
  Q0ProbeInputV1
)
```

Each `<canonical Rn object>` is the decoded canonical source array shown in the
table, nested directly rather than wrapped in a byte string.  Strict wrapper
encoding must reproduce the golden 172-byte CBOR below exactly; decode and
re-encode drift is rejected.  The root is the `frozen_universe_root` in every
Q0 `BehaviorBlobV1`; production universe, target, truth, seed, split, and
assignment roots remain untouched.

The canonical probe encoding is 172 bytes.  Its frozen golden values are:

```text
canonical_probe_cbor_hex =
860119360656686567656c2d71302d70726f62652d696e7075742f3119700104848301193401850119340151686567656c2d6f64642d696e7075742f31058500010001008301193401850119340151686567656c2d6f64642d696e7075742f31088801000100010001008302193402870119340252686567656c2d73696e6b2d696e7075742f31000000008302193402870119340252686567656c2d73696e6b2d696e7075742f3104010203

q0_micro_universe_root_hex =
2c960bcc229175afe6d5e106a34410216669bfe66b14d5c85103762c596f4192
```

Observation behavior follows Section 4 without a micro-only adapter:

- R0 and R1 expose exact `set_size` and Bit slots; every aggregate, context
  flag, and task flag is bottom;
- R2 and R3 expose exact `set_size = 4`; Bit slots, context flags, and task
  flags are bottom;
- on R2 and R3, only scope 3, q0, empty-extension aggregates may be defined;
  every scope 0/1/2, q1, or non-empty-extension aggregate is bottom.

### 6.3 Exact fifteen leaves

The syntax oracle and quotient engine use exactly the following leaves:

1. `scalar_const(RationalParameterId=1)` = `-1`;
2. `scalar_const(RationalParameterId=3)` = `0`;
3. `scalar_const(RationalParameterId=5)` = `+1`;
4. `bit_at(Index=0)`;
5. `bit_at(Index=1)`;
6. `set_size()`;
7. `aggregate(map=0 sum, scope=3 control_volume_all_observed, q0, [])`;
8. `aggregate(map=1 count_nonzero, scope=3 control_volume_all_observed, q0, [])`;
9. `aggregate(map=5 signed_balance, scope=3 control_volume_all_observed, q0, [])`;
10. `aggregate(map=0 sum, scope=0 all, q0, [])`;
11. `aggregate(map=0 sum, scope=3 control_volume_all_observed, q1, [])`;
12. `aggregate(map=0 sum, scope=3 control_volume_all_observed, q0, [(c0,true)])`;
13. `aggregate(map=1 count_nonzero, scope=1 primary, q0, [])`;
14. `context_flag(c0)`;
15. `task_flag(t0)`.

The corresponding leaf-only golden behavior vectors, in row order R0–R3,
are:

| Leaf | Output sort | Exact vector |
|---:|---|---|
| 1 | `RATIONAL_VALUE` | `[-1,-1,-1,-1]` |
| 2 | `RATIONAL_VALUE` | `[0,0,0,0]` |
| 3 | `RATIONAL_VALUE` | `[1,1,1,1]` |
| 4 | `BIT` | `[0,1,⊥,⊥]` |
| 5 | `BIT` | `[1,0,⊥,⊥]` |
| 6 | `BOUNDED_INT` | `[5,8,4,4]` |
| 7 | `RATIONAL_VALUE` | `[⊥,⊥,0,10]` |
| 8 | `BOUNDED_INT` | `[⊥,⊥,0,4]` |
| 9 | `RATIONAL_VALUE` | `[⊥,⊥,0,0]` |
| 10 | `RATIONAL_VALUE` | `[⊥,⊥,⊥,⊥]` |
| 11 | `RATIONAL_VALUE` | `[⊥,⊥,⊥,⊥]` |
| 12 | `RATIONAL_VALUE` | `[⊥,⊥,⊥,⊥]` |
| 13 | `BOUNDED_INT` | `[⊥,⊥,⊥,⊥]` |
| 14 | `BOOL` | `[⊥,⊥,⊥,⊥]` |
| 15 | `BOOL` | `[⊥,⊥,⊥,⊥]` |

Every numeric entry is encoded through its declared sort; for example leaf 7
uses reduced rational pairs while leaf 8 uses bounded integers.  The table's
`⊥` is the corresponding sort's `[0]` cell, never a shared untyped value.

The canonical operator projection consists of all v1.6-surviving typed unary
operators, canonical binary operator IDs `1,2,3,5,6`, `approx_equal` with the
surviving nonzero tolerance IDs `1,2`, and canonical AND2.  Removed operators,
tombstones, source aliases, tolerance-zero aliases, AND1, AND3, and any implicit
coercion are excluded before syntax counting.

The fifteen canonical leaf arrays, canonical operator IDs, tolerance IDs,
AND arity, all structural limits, all resource guards, sort-specific cohort
capacities, continuation-bank policies, and coverage registry are committed by:

```text
q0_projection_manifest_root = ContentHash(
  "HEGEL/Q0/PROJECTION_MANIFEST/V1",
  Q0ProjectionManifestV1
)

Q0ProjectionManifestV1 schema = b"hegel-q0-projection-manifest/1"
q0_projection_manifest_root_hex =
2f39aa248f1305eeaf20a724f6d690cf2b13003f86620d09d2753815831f7ad1
```

Its exact 17-field array order is:

```text
[
  1, schema, projection_id,
  ordered_15_canonical_leaf_nodes,
  [0,1,2,3], [1,2,3,5,6], [1,2], 2,
  [max_depth,max_nodes,max_top_clauses,max_aggregates,
   max_scalar_parameters,max_scope_clauses,max_distinct_bits],
  [raw,syntax,classes,frontier_total,frontier_per_class,bank_total,
   bank_per_class,rounds,output_bytes,wall_seconds,memory_bytes],
  [[sort_id,witness_capacity] for sort_id 1..5],
  b"LEX_MIN_REAL_AST_UP_TO_SORT_CAPACITY",
  b"EXPAND_EACH_BANK_REP_ONCE_REGARDLESS_OF_VISIBLE_DOMINANCE",
  b"PUBLIC_CLASS_ARCHIVE_VISIBLE_FRONTIER_ONLY",
  ordered_resource_guard_registry,
  ordered_coverage_code_registry,
  6
]
```

An ID string is not a substitute for this content root.  A strict frontier
decoder recursively admits only this exact subgrammar; a strict-v1.6 leaf such
as `bit_at(7)`, `context_flag(c3)`, or `task_flag(t1)` remains outside Q0 even
though the full old DSL admits it.

The qualified Q0 semantic binding additionally commits the version IDs,
normative-document SHA-256, projection manifest, production adapter ID, probe
root, and the following already sealed non-formal shrink-6 v1.6 root family:

```text
child_dsl_spec_root = da5ed2db33a88a0912d5003999f787cc26ba18564876615773a82bb742d9f8ae
operator_semantics_root = 922e48ada22dfa8621a4d516e07ec9aa7dc8fc10c165d1cafc963575aed5ec03
identifier_registry_root = 64c9415f7759eec140e439030c5a5374851b9024d7d4849b52b995704ba76ff1
canonical_ast_schema_root = 5de72fc51e27e5501561ffda6b05522f4941d1a13c4b324f5edcc15fa584a0bd
canonical_cbor_profile_root = ef0008912962de9da322eaeea6e421e1e58d16be152f968298774af0fd3249ab

q0_semantic_binding_root = ContentHash(
  "HEGEL/Q0/SEMANTIC_BINDING/V1",
  Q0SemanticBindingV1
)
Q0SemanticBindingV1 schema = b"hegel-q0-semantic-binding/1"
q0_semantic_binding_root_hex =
b7ec5e860a007469b8a1b3930f17c130f59a800d2a832dfd438d18a75538ff99
```

Its exact 17-field array order is:

```text
[
  1, schema, dsl_version, dsl_freeze_version,
  closure_semantics_version, q0_freeze_version, qualification_id,
  normative_document_sha256_bytes, projection_manifest_root,
  child_dsl_spec_root, operator_semantics_root, identifier_registry_root,
  canonical_ast_schema_root, canonical_cbor_profile_root,
  adapter_schema_id, projection_id, probe_universe_root
]
```

These are diagnostic qualification bindings, not the reset production/formal
Q1 roots.  Both independent endpoints must reconstruct both Q0 roots from the
frozen preimages; copying only the golden digests is insufficient.

---

## 7. Future-admissibility signature

### 7.1 Exact schema

```text
FutureAdmissibilitySignatureV1 =
[
  1,
  0x3602,
  b"hegel-q0-construction-signature/1",
  output_sort_id,
  ast_depth,
  ast_node_count,
  scalar_parameter_occurrence_count,
  aggregate_leaf_count,
  distinct_bit_slot_bitmap,
  scope_clause_count,
  top_level_clause_count,
  old_law_composition_depth,
  normalization_profile_id,
  mdl_length_q32
]
```

The distinct-slot field is an exact eight-bit set:

```text
bit i = 1 iff bit_at(i) occurs in the canonical AST
unused bits = 0
```

A cardinality alone is forbidden because `{0}` and `{1}` have different future
union behavior.

`old_law_composition_depth` is exactly zero for every admitted v1.6 program AST:
the grammar contains no old-law-composition node.  A nonzero value fails with:

```text
REJECT_Q0_UNREPRESENTED_LAW_COMPOSITION
```

This field is retained to make the historical limit explicit; it is not
silently inferred from ordinary AST depth.

### 7.2 Normalization profiles

```text
0 GENERAL
1 ABSOLUTE_ROOT
2 CONST_NEGATIVE_ONE       # active RationalParameterId 1
3 CONST_ZERO               # active RationalParameterId 3
4 CONST_POSITIVE_ONE       # active RationalParameterId 5
5 TOP_LEVEL_AND2
```

The profile records the syntax-sensitive normalization/continuation category
needed by the bounded construction proof.  It is not part of behavior equality.

### 7.3 MDL-aware Pareto dominance

Two signatures may be compared only when output sort and normalization profile
are equal.  Signature `a` dominates signature `b` if and only if:

```text
bitmask(a) is a subset of bitmask(b)

and, componentwise:
  ast_depth(a)                         <= ast_depth(b)
  ast_node_count(a)                    <= ast_node_count(b)
  scalar_parameter_occurrence_count(a) <= scalar_parameter_occurrence_count(b)
  aggregate_leaf_count(a)              <= aggregate_leaf_count(b)
  scope_clause_count(a)                <= scope_clause_count(b)
  top_level_clause_count(a)            <= top_level_clause_count(b)
  old_law_composition_depth(a)          <= old_law_composition_depth(b)
  mdl_length_q32(a)                     <= mdl_length_q32(b)

and at least one relation is strict.
```

MDL is part of dominance.  A structurally larger but shorter admitted
representative cannot be dropped merely because another representative uses
fewer structural resources.

The MDL table remains `hegel-mdl-prefix-v1.0.0`; lengths are unsigned Q32.
No symbolic normal form supplies an MDL length.

### 7.4 Distinct normalization-witness multiplicity amendment

The pre-execution v1.0.0 draft kept only one representative for an exact
signature.  Independent syntax/direct replay rejected that rule with the
following target-blind counterexample:

```text
syntax quotient: 69 classes / 95 frontier points
direct quotient: 69 classes / 94 frontier points
missing AST: AND2(context_flag(c0), task_flag(t0))
missing AST CBOR: 82018204828300040083000500
```

Both leaves have the same all-bottom Bool behavior, the same GENERAL
signature, and the same MDL.  Keeping only the lexicographically first leaf
leaves no second distinct child.  Applying AND to that one AST twice triggers
the frozen AND deduplication rewrite and collapses to AND1, so the direct path
cannot construct the admitted distinct-child AND2.

This is a future-admissibility failure, not a behavior-equivalence failure.
The v1.0.0 draft produced no committed, signed, or formal object and is
non-authoritative.  It is superseded before first execution by
`hegel-quotient-closure-v1.0.1` and `hegel-freeze-p3a-q0-v1.0.1`.

The exact representative capacity per signature cohort is:

| Output sort | Capacity | Identity-sensitive surviving rewrite |
|---|---:|---|
| `BOOL` | 2 | AND flatten/deduplicate |
| `RATIONAL_VALUE` | 2 | `difference(x,x) -> 0` |
| `BIT` | 1 | none |
| `SIGN` | 1 | none |
| `BOUNDED_INT` | 1 | none |

Two is sufficient because every surviving operator has at most two AST child
positions that can be compared for canonical identity; `approx_equal` has two
AST children plus a registry tolerance, and v1.6 admits AND2 only.  Within an
exact signature cohort, distinct canonical AST bytes are sorted and retained
up to that sort's capacity.  Ranks are contiguous from zero.  Cohort `A` may
dominate cohort `B` only when signature `A` dominates signature `B` under
Section 7.3 **and** retained multiplicity of `A` is at least that of `B`.
This multiplicity condition is part of continuation state and fixed-point
saturation; it is not part of behavior identity.

A second pre-execution replay exposed a subtler non-monotonicity.  Recomputing
the frontier globally over all 537 admitted syntax ASTs retained 122 points,
whereas incrementally updating only the previously visible frontier retained
120.  A one-witness cohort can be dominated and invisible, then become
nondominated after a second distinct witness arrives.  Therefore every class
keeps a target-blind exact-signature **continuation bank**: the
lexicographically least real admitted ASTs up to the sort capacity are retained
even while their cohort is dominated.  Every new bank representative is
queued and expanded exactly once regardless of visible dominance.  The final
public class archive contains only the globally recomputed visible frontier;
the fixed-point state separately binds the complete bank.  The discarded
120-point draft root is non-authoritative.

---

## 8. Frontier records and deterministic representatives

```text
FrontierEntryV1 =
[
  1,
  0x3603,
  b"hegel-q0-frontier-entry/1",
  FutureAdmissibilitySignatureV1,
  normalization_witness_rank,
  representative_ast_cbor_bytes,
  representative_ast_hash
]

frontier_entry_id = ContentHash(
  "HEGEL/Q0/FRONTIER_ENTRY_ID/V1",
  FrontierEntryV1
)
```

Every representative must be admitted by the strict v1.6 boundary.  Its full
signature, output sort, metrics, exact bit-slot mask, normalization profile,
MDL, CBOR bytes, and AST hash are recomputed from strict replay.  Within one
exact signature cohort, the lexicographically smallest distinct canonical AST
bytes are retained up to the capacity in Section 7.4 and assigned ranks
`0..capacity-1` without gaps.

Strict v1.6 admission is necessary but not sufficient: recursive replay must
also admit every node under the exact Q0 projection manifest and enforce all
micro and inherited limits from Section 6.

Within a class, retained frontier entries are ordered by:

```text
canonical signature CBOR bytes ascending
then normalization_witness_rank ascending
then representative AST CBOR bytes ascending
```

Classes are ordered by behavior ID bytes ascending, then by complete canonical
behavior-blob CBOR bytes ascending.  The second key is also the mandatory
collision preimage check.  Discovery order, Python dict order, process
scheduling, and hash-table iteration are forbidden identity inputs.

For Q0 the quotient class is keyed by `behavior_id` and carries its entire
MDL-aware frontier.  AST bytes remain representative/provenance evidence; they
are not semantic class identity.

### 8.1 Quotient class record

There is deliberately no second semantic class digest.  The exact behavior
preimage plus `behavior_id` is the class identity.  Its archive record is:

```text
QuotientClassRecordV1 =
[
  1,
  0x3604,
  b"hegel-q0-quotient-class/1",
  class_index,
  BehaviorBlobV1,
  behavior_id,
  frontier_count,
  [FrontierEntryV1_0, ..., FrontierEntryV1_(n-1)],
  minimum_mdl_q32
]

quotient_class_record_id = ContentHash(
  "HEGEL/Q0/QUOTIENT_CLASS_RECORD_ID/V1",
  QuotientClassRecordV1
)
```

`behavior_id` is intentionally a content identity, not a standalone closure
claim.  Its semantic use is always the tuple:

```text
(
  dsl_version,
  dsl_freeze_version,
  closure_semantics_version,
  q0_projection_manifest_root,
  q0_semantic_binding_root,
  input_signature_id,
  frozen_universe_root,
  behavior_id,
  behavior_blob_cbor_bytes
)
```

Gate 2 must replay the complete semantic binding, including all five sealed
v1.6 roots and all four DSL/freeze/closure/Q0 version identities, before an
archive can qualify.  Consequently, equal content IDs under a future DSL or
operator registry do not make the two closure claims identical.

The decoder must:

1. strictly decode the nested `BehaviorBlobV1` and retain its complete
   canonical CBOR bytes;
2. recompute `behavior_id` and compare all 32 bytes;
3. strictly decode every frontier entry and recompute its entry ID, AST hash,
   complete signature, and normalization-witness capacity from the AST;
4. verify every frontier signature has the behavior blob's output sort and
   every cohort rank is contiguous from zero;
5. recompute cohort multiplicity, multiplicity-aware Pareto nondominance,
   lexicographic representative selection, frontier order, `frontier_count`,
   and the minimum `mdl_length_q32` over all retained entries, then compare
   `minimum_mdl_q32`;
6. require `class_index` to equal the zero-based position in the class archive;
7. evaluate every frontier AST through the production target-blind adapter on
   all four probe rows and require its complete sort-bound behavior bytes to
   equal the nested `BehaviorBlobV1`; also require input signature `0x7001`,
   the exact probe root, and exactly four cells.

The complete behavior bytes are mandatory even though the digest is stored.
A repeated behavior ID with different behavior bytes is the collision failure
in Section 3; a repeated class with identical bytes is a duplicate-record
failure.

### 8.2 Q0 class archive and root

Records are sorted by `(behavior_id, complete behavior CBOR bytes)`, assigned
contiguous `class_index`, canonically encoded, and committed directly as
RFC6962 leaves:

```text
leaf_hash(record_bytes) = SHA256(0x00 || record_bytes)
node_hash(left, right)   = SHA256(0x01 || left || right)
empty_tree_hash          = SHA256(b"")
```

Non-power-of-two trees use the RFC6962 largest-power-of-two split.  No length
prefix, ContentHash domain, diagnostic JSON, chunk wrapper, or program archive
is inserted between a class record and its leaf hash.

The Python exhaustive syntax oracle, Rust exhaustive syntax oracle, direct
Python quotient engine, direct Rust quotient engine, and host replay each
compute a class archive root from their own decoded records.  A PASS requires
all roots and all record bytes to be identical.  Their common value is named:

```text
qualified_q0_class_archive_root
```

This is a Q0 micro-qualification root.  It is not an M3 formal root, a Q1
production quotient root, or an input to an outside-language certificate.  At
the freeze state of this document it is `null` because Q0 is `NOT_RUN`.

---

## 9. Saturation state and completeness condition

The qualified state is not a bare set of behavior IDs or visible classes.  It
is the resource-labelled tuple:

```text
S_t = (
  all_seen_canonical_program_records,
  behavior_id -> complete exact-signature continuation bank R_t,
  behavior_id -> globally recomputed visible MDL-aware frontier F_t,
  work_queue / coverage / fixed-point metadata
)
```

The work queue contains every new real representative entering `R_t`, even if
its cohort is currently dominated in `F_t`.  Each such representative is
expanded exactly once.  A third same-signature representative beyond a
sort-capacity of two, or a second beyond a capacity of one, is not admitted to
the bank and creates no work item.

Q0 PASS requires all of the following:

```text
S_(t+1) == S_t
work_queue_empty == true
every admitted operator × eligible frontier tuple closed == true
no new behavior class == true
no new nondominated frontier point == true
no new continuation-bank point == true
exhaustive syntax oracle complete == true
syntax-oracle quotient == direct quotient engine == true
syntax/direct/host complete continuation-bank rows == true
Python result == Rust result == host replay == true
no resource guard hit == true
```

No-new-class and no-new-visible-frontier are both insufficient.  A latent bank
point in an old class can unlock either the second witness that revives its
cohort or a later admissible composition, so bank delta and queue closure must
also be zero before PASS.

The exhaustive syntax oracle enumerates every strict canonical micro program
within the frozen micro structural limits, then groups by exact behavior bytes
and independently computes the MDL-aware frontier.  The direct quotient engine
constructs classes and frontiers without using target truth.  The visible class
archives must be byte-identical after canonical ordering, and each path's
complete saturation state must independently satisfy the same fixed-point
contract.

### 9.1 Host-only dual PASS receipt

Numeric tag `0x3605` belongs only to the host-issued dual PASS receipt.  Its
canonical object has exactly 40 fields in this order:

```text
Q0SaturationReceiptV1 =
[
  1,                                      # 0
  0x3605,                                 # 1
  b"hegel-q0-saturation-receipt/1",       # 2
  b"hegel-phase3a-q0-exact-quotient-qualification-v1", # 3
  b"hegel-old-dsl-v1.6.0",                # 4
  b"hegel-quotient-closure-v1.0.1",       # 5
  b"hegel-freeze-p3a-q0-v1.0.1",          # 6
  b"hegel-q0-micro-projection-v1",        # 7
  q0_micro_universe_root,                 # 8
  2,                                      # 9 PASS only
  syntax_raw_operator_application_count,  # 10
  quotient_raw_operator_application_count,# 11
  canonical_syntax_program_count,         # 12
  behavior_class_count,                   # 13
  frontier_point_count,                   # 14
  maximum_frontier_points_per_class,      # 15
  saturation_round_count,                 # 16
  true,                                   # 17 zero-delta full round
  true,                                   # 18 work queue empty
  true,                                   # 19 all typed operator/frontier tuples closed
  true,                                   # 20 no resource guard hit
  true,                                   # 21 exhaustive syntax oracle complete
  syntax_program_archive_root,            # 22
  syntax_oracle_class_archive_root,        # 23
  quotient_engine_class_archive_root,      # 24
  syntax_operator_coverage_root,           # 25
  quotient_operator_coverage_root,         # 26
  python_implementation_root,              # 27
  rust_implementation_root,                # 28
  python_endpoint_output_root,             # 29
  rust_endpoint_output_root,               # 30
  host_replay_class_archive_root,          # 31
  14,                                      # 32 Q0 gate count
  0x3fff,                                  # 33 all fourteen gate bits
  0,                                       # 34 Q1 NOT_RUN
  null,                                    # 35 Q1 output root
  0,                                       # 36 Q2 NOT_RUN
  false,                                   # 37 role evaluation not performed
  null,                                    # 38 M3 formal roots
  false                                    # 39 outside certificate not issued
]

q0_saturation_receipt_root = ContentHash(
  "HEGEL/Q0/SATURATION_RECEIPT/V1",
  Q0SaturationReceiptV1
)
```

The syntax oracle and direct quotient engine keep independent raw counts,
class roots, and per-operator coverage roots.  Their class roots and the host
replay class archive root must be byte-identical.  Field 31 is not a
saturation-state root.  The two implementation roots bind
the independent Python and Rust source/implementation closures; the two
endpoint roots bind their complete emitted endpoint states.  Every root field
is exactly 32 bytes.

For Q0 qualification, an implementation closure is not source files alone.
The Python implementation root additionally binds the pinned OCI image and
exact isolated interpreter invocation.  The Rust implementation root binds
the pinned OCI image, target triple, exact offline Cargo invocation, every
path-crate source byte, `Cargo.lock`, every lock-selected registry archive and
its checksum-verified regular-file expansion, and the deterministic sealed
Cargo home used by the run.  That Cargo home is materialized before execution
and mounted read-only; mutable global Cargo databases and network resolution
are excluded.  The outer host evidence separately binds the commit-tree
verification, isolation prerequisites, and host gate-evidence root.

The seven numeric counts are exact positive integers within the guards in
Section 10.  Additionally:

```text
behavior_class_count <= canonical_syntax_program_count
behavior_class_count <= frontier_point_count
maximum_frontier_points_per_class <= frontier_point_count
```

The five consecutive `true` fields are acceptance guards, not descriptive
claims that may be set before replay.  Status 2 is the only status admitted by
this record.  At `NOT_RUN`, during a partial run, after a guard hit, or after
any mismatch, the receipt does not exist and its root is `null`; endpoint
counts must never be placed into a fabricated receipt.  The Q1/Q2/M3/role and
certificate guards at fields 34–39 are mandatory even on a Q0 PASS.

### 9.2 Endpoint diagnostic and saturation state

Each implementation may emit a target-neutral endpoint diagnostic state under
the untagged byte-string schema:

```text
b"hegel-q0-oracle-endpoint-state/1"
```

Its state root uses:

```text
HEGEL/Q0/ORACLE_ENDPOINT_STATE/V1
```

The endpoint canonical object has exactly 43 fields:

```text
[
  1, endpoint_schema, q0_freeze_version, dsl_version,
  closure_semantics_version, projection_id, probe_universe_root,
  projection_manifest_root, semantic_binding_root,
  single_endpoint_status,
  syntax_raw, quotient_raw, syntax_strict_admitted,
  quotient_strict_admitted, syntax_rewrites, quotient_rewrites,
  canonical_syntax_count, behavior_class_count,
  visible_frontier_count, maximum_visible_frontier_per_class,
  syntax_bank_count, quotient_bank_count,
  maximum_syntax_bank_per_class, maximum_quotient_bank_per_class,
  saturation_round_count, work_queue_empty, zero_delta_full_round,
  final_class_delta, final_frontier_delta, final_bank_delta,
  syntax_program_archive_root, syntax_class_archive_root,
  direct_class_archive_root, syntax_coverage_root, direct_coverage_root,
  syntax_state_root, direct_state_root, all_guards_respected,
  target_truth_accessed, split_accessed, role_evaluation_performed,
  formal_roots_generated, authoritative_claim_allowed
]
```

The fields occupy indices `0..42` exactly in the displayed order.  The
endpoint root is `ContentHash("HEGEL/Q0/ORACLE_ENDPOINT_STATE/V1", object)`.
The independently replayable saturation-state preimage is exactly:

```text
(
  tuple(program_records),
  tuple(complete_continuation_bank_rows),
  tuple(visible QuotientClassRecordV1 records),
  tuple(path OperatorCoverageRowV1 records),
  FixedPointStateV1
)

syntax_state_root = ContentHash("HEGEL/Q0/SYNTAX_STATE/V1", preimage)
direct_state_root = ContentHash("HEGEL/Q0/DIRECT_QUOTIENT_STATE/V1", preimage)
```

Each isolated endpoint must additionally emit the diagnostic-only fields
`syntax_saturation_state_preimage_cbor_hex` and
`direct_saturation_state_preimage_cbor_hex`.  Each field is the strict
canonical CBOR encoding of the complete five-tuple above; neither field enters
the 43-field endpoint object.  The host strict-decodes and byte-exactly
re-encodes both blobs, independently recomputes the program, class, coverage,
and path-state roots, and requires the corresponding Python, Rust, and host
preimage bytes to agree.  Root equality without these replayable preimage
bytes is insufficient for Q0 PASS.

Each program record is the untagged array:

```text
[
  1,
  b"hegel-q0-syntax-program-record/1",
  program_index,
  canonical_ast_cbor_bytes,
  canonical_ast_hash,
  output_sort_id,
  mdl_length_q32
]
```

Program records are ordered by `(ast_depth, ast_node_count, output_sort_id,
root_operator_id, canonical_ast_cbor_bytes)` before assigning contiguous
`program_index`.

The syntax program archive root is the RFC6962 root over the canonical CBOR of
these seven-field records.  A raw-AST-only tree is a different, rejected root.

Each continuation-bank row is:

```text
[
  behavior_id,
  complete_behavior_blob_cbor_bytes,
  FutureAdmissibilitySignatureV1,
  [[rank, representative_ast_cbor, representative_ast_hash], ...]
]
```

Rows are sorted by behavior ID, complete behavior bytes, and canonical
signature bytes.  Bank entries are sorted by rank and AST bytes.  Visible
class records remain the Section 8 archive objects and do not expose dominated
bank cohorts.

The fixed-point metadata is the exact eleven-field array:

```text
[
  1,
  b"hegel-q0-fixed-point-state/1",
  path_id,
  saturation_round_count,
  work_queue_empty,
  zero_delta_full_round,
  all_eligible_tuples_covered,
  final_new_program_delta,
  final_class_delta,
  final_frontier_delta,
  final_bank_delta
]
```

PASS requires all three booleans to be true and all four final deltas to be
zero.  Per-round discovery history is diagnostic only and is excluded so that
different legal schedules do not change semantic state identity.
The supervisor validates each implementation's round indices, bounds, and
terminal zero-delta row independently, but it must not require intermediate
round-history counters to be byte-identical across implementations.

The two exact `path_id` byte strings are:

```text
b"hegel-q0-exhaustive-syntax-path/1"
b"hegel-q0-direct-quotient-path/1"
```

The following values are dual-implementation golden expectations for the
frozen source; they become qualification evidence only after isolated endpoint
execution and host replay:

| Object | Expected SHA-256/root hex |
|---|---|
| syntax program archive | `bd1a59f816bd6648d0dd73b9a1622f2bb88bb9aeca1489a0d876fbc9dbf0c829` |
| visible class archive (both paths) | `a2f0dacf4524fdb8725d29a2c3883a7ebd78fa686cb2030ac0d0608710176cf1` |
| syntax saturation state | `7028819d133c4da6071c06a0bfca2d0b91622e106207d0b0f081148f41c0826a` |
| direct saturation state | `d87ef33d9d7010ded284b55acfa71aab4d7d991e3d7703c30f1db2caf5893933` |
| syntax coverage | `6953f39dc97f17288850b524ca8b04dbb2f6ddd3d53eaf4cb8e4e6465bcd840c` |
| direct coverage | `a9a0b6fdc97c475323ccae31fba14a6df411307220efd8538c7971fe9c38c1fd` |
| single endpoint output | `d33e54dd99e6cbe8aacc541fc0877af9657a553be58523670cce5c474006d4d2` |

Expected counts are `567/545` raw and strict-admitted applications,
`30/30` rewrites, 537 syntax programs, 69 behavior classes, 122 visible
frontier points (maximum 4 per class), and 251 continuation-bank points on
each path (maximum 43 per class), converging in three saturation rounds.

Each path also emits one coverage row per canonical operator code:

```text
OperatorCoverageRowV1 =
[
  operator_code,
  eligible_raw_count,
  strict_admitted_count,
  rewrite_collapse_count,
  canonical_duplicate_count,
  new_canonical_program_count
]
```

The append-only numeric coverage-code registry is:

```text
0x0000..0x000e  fifteen leaves in Section 6.3 manifest order
0x1000..0x1003  unary operator IDs 0..3
0x2001           difference
0x2002           equal_exact
0x2003           less_equal
0x2005           same_sign
0x2006           opposite_sign
0x3001           approx_equal tolerance ID 1
0x3002           approx_equal tolerance ID 2
0x4002           canonical AND2
```

Rows are ordered by `operator_code` ascending and committed directly as an
RFC6962 tree of their canonical CBOR encodings.  The syntax and direct quotient
paths compute separate coverage roots; neither may copy the other's counters.

The endpoint places `q0_projection_manifest_root` and
`q0_semantic_binding_root` immediately after the probe-universe root and before
its single-endpoint status.  Python and Rust independently reconstruct these
roots; they may not merely copy golden digest literals.  Their syntax and
direct state roots are explicit endpoint fields.

An endpoint that completes its own two-path equality check reports exactly
`SINGLE_IMPLEMENTATION_EXHAUSTIVE_ORACLE_EQUIVALENCE_PASS`.  It must not emit
numeric status 2 or the word `DUAL`; those belong only to the host receipt
after Python, Rust, and host replay agree.  Endpoint diagnostics remain
non-authoritative until their complete source-bound state, both independent
outputs, host replay, and all 14 gates have qualified.  They populate no M3,
Q1, Q2, role, certificate, or host receipt root slot merely by being non-null.

---

## 10. Q0 resource guards

```yaml
maximum_raw_operator_applications: 5000
maximum_canonical_syntax_programs: 2000
maximum_behavior_classes: 2000
maximum_total_frontier_points: 2000
maximum_frontier_points_per_class: 64
maximum_total_continuation_bank_points: 2000
maximum_continuation_bank_points_per_class: 64
maximum_saturation_rounds: 4
maximum_wall_time_seconds: 300
maximum_memory_bytes: 536870912       # 512 MiB
maximum_output_bytes: 67108864        # 64 MiB
```

The 5,000 raw-application budget applies independently to the exhaustive
syntax path and the direct quotient path; the two counters are never pooled.
Wall time, resident memory, and output bytes are likewise enforced per isolated
endpoint.  A value equal to a numeric maximum is admissible.  The next event
that would make it exceed the maximum triggers the guard before the event is
accepted.

Visible frontier and continuation bank have independent total and per-class
guards; the receipt's frontier counts describe only the visible frontier.
Round zero is leaf initialization and is not a saturation round.  Rounds one
through four process newly admitted bank work.  PASS requires a completed
zero-delta round with an empty queue no later than round four.  If work remains
or any program/class/frontier/bank delta is nonzero after round four, the
rounds guard produces `INCONCLUSIVE_RESOURCE_LIMIT`.

The raw counter increments once for one canonical operator token applied to one
type-legal ordered child/frontier tuple, before normalization, rejection,
deduplication, or quotient merging.  Commutative tuples are counted only in the
frozen canonical child order.

Every limit is fail-closed.  The implementation must test a limit before
discarding or truncating a syntax program, behavior class, or frontier point.
Any hit yields exactly:

```text
INCONCLUSIVE_RESOURCE_LIMIT
```

It cannot yield PASS, COMPLETE, outside-language evidence, or authorization for
Q1.  Partial diagnostic bytes may be retained but do not populate formal roots.

---

## 11. Explicit Bool/Bit predicate role matcher

The existing target and truth rows remain Bit-valued and their roots do not
change.  Q0 class identity remains strictly sort-sensitive.  Q0 reuses the
explicit role matcher already reserved by the old freeze; it does not invent a
new coercion rule.  The only permitted future Q2 role comparison profile is:

```text
BOOL_BIT_EXACT_PREDICATE_MATCH_V1
```

For each row this matcher returns true only when:

1. the target cell is CBOR uint Bit `0` or `1`;
2. the program cell is defined;
3. either:
   - the program sort is `BOOL`, the program value has exact bool type, and
     `int(program_bool) == target_bit`; or
   - the program sort is `BIT`, the program value has exact uint Bit type, and
     `program_bit == target_bit`.

All rows must satisfy the rule for a role match.  Bottom never matches.

This is a role-level predicate comparison profile, not:

- a Bool/Bit quotient merge;
- an implicit DSL coercion;
- an operator available to old-language programs;
- a change to target, truth, universe, seed, split, or assignment roots.

Q0 and Q1 do not execute this matcher.  It remains reserved for Q2 after a
sealed Q1 quotient closure.

---

## 12. Fourteen Q0 readiness gates

Q0 starts at `0/14`.  The gates are a new registry and inherit zero passes from
the historical M3 `24/24` record.

| Gate | Exact name |
|---:|---|
| 1 | `NORMATIVE_DIRECTION_BYTES_BOUND` |
| 2 | `V16_DSL_TYPING_AND_REGISTRY_ROOTS_QUALIFIED` |
| 3 | `INPUT_SIGNATURE_OBSERVATION_ADAPTERS_QUALIFIED` |
| 4 | `BEHAVIOR_AND_BOTTOM_CODEC_QUALIFIED` |
| 5 | `UNIVERSE_ONLY_BINDINGS_QUALIFIED` |
| 6 | `EXACT_EQUIVALENCE_CONTRACT_QUALIFIED` |
| 7 | `CONSTRUCTION_SIGNATURE_QUALIFIED` |
| 8 | `PARETO_DOMINANCE_AND_MDL_QUALIFIED` |
| 9 | `PER_OPERATOR_CONGRUENCE_QUALIFIED` |
| 10 | `STRUCTURAL_INDUCTION_COMPLETENESS_QUALIFIED` |
| 11 | `EXHAUSTIVE_MICRO_ORACLE_EQUALITY_QUALIFIED` |
| 12 | `COLLISION_BOTTOM_SORT_ADVERSARIAL_VECTORS_PASS` |
| 13 | `TARGET_TRUTH_AND_SPLIT_INPUT_ISOLATION_PASS` |
| 14 | `DUAL_HOST_AGREEMENT_Q1_OUTPUTS_NULL_NOT_RUN` |

Every gate is scoped to the Q0 micro projection.  In particular, Gates 9–11 do
not assert full-v1.6 production closure completeness.

Gate 12 is produced by the source-bound host Python contract replay.  Its
evidence must state `producer_scope = HOST_PYTHON_CONTRACT_REPLAY` and
`dual_adversarial_execution_claimed = false`.  It qualifies the frozen
collision, bottom, sort, and canonical-wire controls; it does not claim that
both isolated endpoints independently executed injected abnormal paths.

The Python and Rust endpoints receive the hard filesystem/process isolation
described by the dual-isolation profile.  The host replay is a source-bound,
target-blind trusted issuer, not a third independent endpoint: it is not
filesystem-hard-isolated and must not be presented as an external role.  Gate
13 therefore binds its complete loaded-module manifest and verifies that all
host-local module files are the exact clean-commit files, while the
`target_truth_split_sources_present = false` isolation statement applies to
the two endpoint snapshots.

Gate 14 requires:

```text
q0 terminal status = DUAL_EXHAUSTIVE_ORACLE_EQUIVALENCE_PASS
Python/Rust visible class records and class archive root byte-identical
syntax/direct path states independently fixed-point-valid and host-replayed
Python and Rust endpoint roots separately source-bound and host-verified
host-only 40-field Q0SaturationReceiptV1 exists
q0_saturation_receipt_root is non-null and replay-valid
gate count/bitmap = 14/0x3fff
Q1 state = NOT_RUN
Q2 state = NOT_RUN
all Q1/Q2 output slots = null
M3 formal roots = null
role evaluation performed = false
certificate issued = false
```

Reaching 14/14 authorizes preparation of a separate Q1 engineering freeze.  It
does not automatically start Q1.

---

## 13. Provenance reuse and mandatory reset

The following may be rebound only as bit-identical historical provenance:

- strict canonical AST/CBOR profiles;
- surviving v1.6 AST bytes and hashes;
- old MDL code table;
- production universe rows and roots;
- target, truth, seed, split, custody, trust, and access-history commitments;
- Source-Y and Evidence-Z syntactic capacity evidence.

They do not pass a Q0 gate merely by existing.

The following must be new Q0/Q1 objects or evidence:

- quotient closure-semantics binding;
- observation-adapter source binding;
- exact behavior/bottom codec qualification;
- construction-signature and dominance policy;
- class/frontier identities and canonical order;
- micro oracle and quotient implementation bindings;
- saturation and resource-limit receipts;
- Q0 dual agreement and host replay;
- later Q1 genesis, execution manifest, completion receipts, and archive roots.

No old `canonical_program_archive_root`, output archive root, exhaustion receipt,
execution manifest, or M3 gate count is reused as a quotient output.

---

## 14. Failure and publication rules

The following statements are allowed before execution:

```text
Q0_SPEC_FROZEN_NOT_RUN
Q0_READINESS_0_OF_14
Q1_NOT_RUN
Q2_NOT_RUN
```

After a valid Q0 PASS, the strongest allowed statement is:

```text
Q0_MICRO_PROJECTION_DUAL_EXHAUSTIVE_ORACLE_EQUIVALENCE_QUALIFIED
```

The following remain forbidden throughout Q0:

- `QUOTIENT_CLOSURE_COMPLETE` for either production universe;
- `OUTSIDE_FROZEN_QUOTIENT_CLOSURE`;
- parity outside/in-language verdict;
- sink control pass/fail verdict;
- a new DSL version or target-conditioned language shrink;
- a new split seed, redraw, or assignment;
- formal M3 root or certificate issuance;
- using a subset, unchanged class count, elapsed time, or an exhausted guard as
  a completeness proof.

Q0 artifacts must disclose whether they are pre-execution specs, partial
diagnostics, single-implementation results, dual results, or host-replayed
qualification.  A stored report or passing unit test alone is not 14/14.

---

## 15. Authorized successor

The only automatic successor to a genuine Q0 14/14 PASS is drafting and freezing
the separate:

```text
Phase-3A-Q1 — Complete Frozen Quotient Closure
```

Q1 must freeze production resource envelopes, formal roots, independent
implementation identities, full class/frontier archives, and a new run genesis
while still excluding target truth and role matching.  Q2 may begin only after
Q1 has reached its own exact `COMPLETE` terminal state and sealed its quotient
closure.

No Q0 result alone authorizes Path B.  Path B remains conditional on a later,
target-blind production quotient-capacity result under a separately frozen Q1
resource contract.
