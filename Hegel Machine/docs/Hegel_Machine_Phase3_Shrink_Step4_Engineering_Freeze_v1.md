# Hegel Machine Phase-3 shrink step 4 engineering freeze v1

Status: **NORMATIVE ENGINEERING FREEZE; NON-FORMAL AND NOT RUN**

This document freezes shrink step 4 as a child-language construction. It does
not execute closure, publish an observed enumeration output, create a formal
root, evaluate a target role, issue a certificate, sign an object, or change
ACTIVE governance.

## 1. Trigger and parent evidence

Shrink step 4 is admitted only by Evidence Commit N:

```text
parent evidence commit       c286732c140bd9adcfd3eef2b1788b3eac0eb3e9
parent implementation basis  d17b03e14f3f3e8a63c924706086f17367fbc0d6
parent evidence record ID     phase3_shrink3_dual_complete_enumeration_diagnostic_3030ad10f2cd4f767a8397597be1ab3ed6cac7cd71975d69f59cc5abec6a4f5a
parent diagnostic status      DUAL_DSL_TOO_LARGE_HOST_REPLAY_PASS
parent claim level            NON_FORMAL_DUAL_CHILD_DIAGNOSTIC
parent DSL/freeze             hegel-old-dsl-v1.3.0 / hegel-freeze-p2b-p3-v1.3.0
parent execution state        NOT_RUN
parent formal roots           null
```

That record establishes a bounded, dual diagnostic rank-50,001 witness and a
host replay. It does not establish complete closure cardinality and is not a
formal M3 terminal. Its sole routing authority here is the preregistered
engineering transition from three to two maximum root conjunction clauses.

The child begins and remains:

```text
execution_state                  = NOT_RUN
formal_roots_generated            = false
formal_roots                      = null
formal_state_transition_allowed   = false
engineering_only                  = true
sealed_dual_strict_qualification  = NOT_RUN
```

## 2. Child identity and sole language delta

```text
parent DSL       hegel-old-dsl-v1.3.0
parent freeze    hegel-freeze-p2b-p3-v1.3.0
child DSL        hegel-old-dsl-v1.4.0
child freeze     hegel-freeze-p2b-p3-v1.4.0
amendment        hegel-freeze-p2b-p3-v1.4.0-shrink-step4
shrink step      SHRINK_STEP_4_REDUCE_MAX_TOP_LEVEL_CLAUSES_3_TO_2
```

The only language change is:

```text
maximum_top_level_clauses: 3 -> 2
```

The limit applies to the normalized root `top_level_AND` node (formal tag 4).
It does not recursively count nested nodes that merely have Bit output. A
non-AND root has `top_level_clause_count = 0`; a normalized root AND has the
number of its canonical children. A source expression is first admitted and
normalized under all inherited rules, then the child maximum is checked. Thus
a three-input source that canonically collapses to at most two distinct
clauses may survive, while a canonical three-clause root does not.

The exact failure for an otherwise legal normalized root with more than two
clauses is:

```text
REJECT_STRUCTURAL_LIMIT
```

No special shrink-4 removal code is introduced. The report surface must carry
`maximum_top_level_clauses = 2` so a recognizer cannot claim the child version
while silently retaining the parent limit.

## 3. Inherited surface and rejection order

All remaining semantics are byte-for-byte inherited from v1.3.0, including:

- aggregate active IDs `[0,1,5]` and tombstones `[2,3,4]`;
- rational active IDs `[1,3,5]`, tombstones `[0,2,4,6]`, and reserved ID 7;
- binary source IDs `[1,2,3,4,5,6]`, formal IDs `[1,2,3,5,6]`, source alias
  ID 4, tombstone ID 0, and reserved ID 7;
- source alias `greater_equal(left,right) -> less_equal(right,left)`;
- aggregate, rational, then binary tombstone priority;
- implicit Bit-to-RationalValue coercion prohibition, bottom semantics,
  normalization, depth, node-count and scope-clause limits;
- AST numeric tags, deterministic CBOR, `HEGEL/AST/V1`, MDL codewords,
  closure budgets, exact extensional equality, and target-independent program
  identity.

Structural or typing errors are still resolved before legal-tree tombstone
passes. Inherited tombstones and noncanonical rewrites are resolved before the
new normalized-root clause limit. Consequently a legal three-clause source
containing a removed aggregate, rational parameter, or `add` returns the
inherited removal code, not `REJECT_STRUCTURAL_LIMIT`. Malformed payloads are
never searched for tombstone-looking bytes.

## 4. Cross-version program identity

A parent v1.3.0 canonical program survives exactly when its already-normalized
root is not an AND of more than two clauses. Every survivor retains identical
canonical AST CBOR bytes, `HEGEL/AST/V1` hash, operator IDs, registry IDs, and
MDL codewords. No canonical program is repaired, truncated, split, renumbered,
or automatically migrated.

Byte stability does not permit reuse of a parent semantic/archive identity.
Any later child archive or formal object must bind v1.4.0 and newly constructed
child roots. A legacy-reader report must say that automatic migration was not
performed.

## 5. Five non-formal diagnostic roots

The following SHA-256 values are synthetic engineering bindings. Their
SHRINK4 domains deliberately make all five new even where the inherited AST
or registry bytes are unchanged. They are not formal CBOR/RFC6962 roots.

| field | exact UTF-8 preimage, with `\\0` meaning one NUL | SHA-256 hex |
|---|---|---|
| `child_dsl_spec_root` | `HEGEL/M3/SHRINK4/DIAGNOSTIC_BINDING/V1\\0CHILD_DSL_SPEC\\0hegel-old-dsl-v1.4.0\\0hegel-freeze-p2b-p3-v1.4.0\\0shrink-step4\\0maximum-top-level-clauses:2` | `736c9cf98749d9a9d2d98596d15a5b09329e1d6eb74d4bee172837fdd34e876f` |
| `operator_semantics_root` | `HEGEL/M3/SHRINK4/DIAGNOSTIC_BINDING/V1\\0OPERATOR_SEMANTICS\\0hegel-old-dsl-v1.4.0\\0hegel-canonical-ast-v1\\0hegel-mdl-prefix-v1.0.0\\0binary-active-formal:1,2,3,5,6\\0binary-tombstones:0\\0binary-source-alias:4\\0maximum-top-level-clauses:2` | `45fe7c575759b6955eb6b52ad954a9ca6561083dbdb67155f9731e795c6fe050` |
| `identifier_registry_root` | `HEGEL/M3/SHRINK4/DIAGNOSTIC_BINDING/V1\\0IDENTIFIER_REGISTRY\\0hegel-old-dsl-v1.4.0\\0aggregate-active:0,1,5\\0aggregate-tombstones:2,3,4\\0rational-active:1,3,5\\0rational-tombstones:0,2,4,6\\0rational-reserved:7\\0binary-source-active:1,2,3,4,5,6\\0binary-formal-active:1,2,3,5,6\\0binary-source-alias:4\\0binary-tombstones:0\\0binary-reserved:7` | `1f9b886480ace19440469267abd06f24c65ef61fb9c734c5ef5ff8ae7e981fd3` |
| `canonical_ast_schema_root` | `HEGEL/M3/SHRINK4/DIAGNOSTIC_PROFILE/V1\\0CANONICAL_AST_SCHEMA\\0hegel-canonical-ast-v1\\0strict-numeric-tag-cbor-array\\0maximum-top-level-clauses:2` | `f9b02ddad69f04f1f9137501dccfdcefa111d0402570197b68b98c11ebcb4eda` |
| `canonical_cbor_profile_root` | `HEGEL/M3/SHRINK4/DIAGNOSTIC_PROFILE/V1\\0CANONICAL_CBOR_PROFILE\\0hegel-cbor-det-v1\\0RFC8949-deterministic-no-map-text-float-tag-indefinite` | `b7fd10722f31d780d53b2f490c92491872ffc749b4cb5cdfccc3eebd5f18837f` |

The program archive remains target-independent. These roots contain no odd or
sink target, split, seed, key, signature, holdout, or formal-run identity.

## 6. Qualification admission

Before any shrink-4 enumerator is admitted, independent Python and Rust
recognizers must be driven by one commit-bound supervisor over:

1. an ordered sealed strict vector manifest covering surviving identity,
   source/formal three-clause rejection, post-normalization collapse, inherited
   tombstone priority, malformed/noncanonical priority, and boundary cases;
2. the complete inherited 2,160-source constructive survivor set, with exact
   parent/child CBOR and hash identity and a common accepted-set commitment;
3. target-free module closure and pinned offline container controls; and
4. exact `maximum_top_level_clauses = 2` report binding on both implementations.

Equal category counts are insufficient: normalized per-vector outcomes,
ordered outcome roots, capacity fields, and accepted-set commitment must all
match. The sealed matrix contains 22 exact wires (category counts
`3/2/2/5/1/1/8`) with manifest root
`sha256:f84035e632bf5a655a9ebd636a0cafe7ab1097c45be87d4db944a0012f52aa90`
and outcome root
`sha256:c19341f08ac5f5759c2cdcb3681a37d958de362b81d02c184f7e2413dca18d7c`.
The exact capacity schemas contain 37 Python / 36 Rust fields and the golden
schemas contain 33 Python / 32 Rust fields. In each pair the only Python-only
field is `loaded_hegel_modules`. The supervisor rejects any missing, aliased,
extra, or substituted key, including omission of
`maximum_top_level_clauses`.

The final qualification report applies the same exact-key rule recursively to
repository, vector, capacity, runtime, Docker and Cargo-seed receipts. The
offline Cargo `cache`/`index` transport is committed as ordered regular-file
rows `(path, mode, size, sha256)` under
`HEGEL/SHRINK4/CARGO_SEED_MANIFEST/V1` and must remain byte-identical before
and after the build.

The 2,160 programs are a survivor subset, not a closure enumeration. A passing
dual strict result authorizes only development and later qualification of
independent complete enumerators. It cannot start M3.

## 7. Claim boundary

No source-freeze or dual-strict artifact may contain observed shrink-4
enumeration counts, witness hashes, archive roots, chunk roots, bucket roots,
target matches, or closure verdicts. Those can exist only after a later source
commit and independent run.

Nothing here claims shrink-4 `COMPLETE`, full closure cardinality, odd-target
outside status, hidden-sink mechanism recognition, relation invention, MDL
success, `OUTSIDE_FROZEN_CLOSURE(...)`, M4 signatures, Phase-3 exit, Phase-2B
exit, or ACTIVE governance.
