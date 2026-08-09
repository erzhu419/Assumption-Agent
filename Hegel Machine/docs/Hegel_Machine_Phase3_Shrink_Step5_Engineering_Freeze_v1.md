# Hegel Machine Phase-3 shrink step 5 engineering freeze v1

Status: **NORMATIVE ENGINEERING FREEZE; NON-FORMAL AND NOT RUN**

This document freezes shrink step 5 as a child-language construction. It does
not execute closure, create a formal root, evaluate a target role, instantiate
a seed or key, sign a certificate, or move M3 out of `NOT_RUN`.

## 1. Sole admission authority

The only routing authority is Evidence Commit R:

```text
parent evidence commit       1bbdae8f3131625621c0bc1cfdfe5d7da6035e13
parent implementation basis  103eb6ad2d8500024580193b895809784d894609
parent record ID              phase3_shrink4_dual_complete_enumeration_diagnostic_5693b38315689969a1a525b75bec2917f95af1aa54951267797a0319afc60521
parent artifact SHA-256       2d653f667d8d43e0e8e68c54d6f0a939aab57bf6ba3add9b334809ca17745058
parent diagnostic status      DUAL_DSL_TOO_LARGE_HOST_REPLAY_PASS
parent claim level            NON_FORMAL_DUAL_CHILD_DIAGNOSTIC
parent DSL/freeze             hegel-old-dsl-v1.4.0 / hegel-freeze-p2b-p3-v1.4.0
parent execution state        NOT_RUN
parent formal roots           null
```

Evidence R authorizes exactly the preregistered route:

```text
operation                         reduce max_total_node_count from 7 to 6
preregistered_shrink_order_step   5
maximum_top_level_clauses_remains 2
only_open_route                   true
authority                         ENGINEERING_ONLY
formal_status_promotion_allowed   false
```

The shrink-5 source commit must be a single-parent direct child of Evidence R.
An ancestor-only or byte-substituted parent binding is rejected.

The child begins and remains:

```text
execution_state                  = NOT_RUN
formal_roots_generated           = false
formal_roots                     = null
formal_state_transition_allowed  = false
engineering_only                 = true
sealed_dual_strict_qualification = NOT_RUN
```

## 2. Child identity and sole language delta

```text
parent DSL     hegel-old-dsl-v1.4.0
parent freeze  hegel-freeze-p2b-p3-v1.4.0
child DSL      hegel-old-dsl-v1.5.0
child freeze   hegel-freeze-p2b-p3-v1.5.0
amendment      hegel-freeze-p2b-p3-v1.5.0-shrink-step5
step           SHRINK_STEP_5_REDUCE_MAX_TOTAL_NODE_COUNT_7_TO_6
```

The only language change is:

```text
maximum_ast_node_count / max_total_node_count: 7 -> 6
maximum_top_level_clauses:                       remains 2
```

Both exact fields, `maximum_ast_node_count = 6` and
`maximum_top_level_clauses = 2`, must appear on every strict, golden and
capacity report. Neither spelling is an alias for the other.

The source recognizer first runs the entire v1.4.0 parse, typing, registry,
tombstone, rewrite and canonical normalization pipeline. The formal decoder
first establishes an exact valid v1.4.0 canonical AST. Only then does v1.5.0
enforce the six-node limit on the normalized tree. Therefore a raw expression
whose redundant structure collapses to six or fewer nodes survives, while an
otherwise legal canonical seven-node tree returns:

```text
REJECT_STRUCTURAL_LIMIT
```

No shrink-5-specific error code is introduced.

## 3. Inherited semantics

Everything else is inherited byte-for-byte from v1.4.0, including:

- aggregate active IDs `[0,1,5]` and tombstones `[2,3,4]`;
- rational active IDs `[1,3,5]`, tombstones `[0,2,4,6]`, reserved ID `7`;
- binary source IDs `[1,2,3,4,5,6]`, formal IDs `[1,2,3,5,6]`, source
  alias ID `4`, tombstone ID `0`, and reserved ID `7`;
- `greater_equal(left,right) -> less_equal(right,left)`;
- inherited error priority, explicit coercions, typing, bottom semantics,
  scope rules, AND normalization, depth and two-clause limits;
- strict numeric-tag CBOR arrays, deterministic CBOR, `HEGEL/AST/V1`, the
  Q32 MDL code, exact extensional equality, and target-independent program
  identity.

Malformed or ill-typed inputs do not become node-limit rejections. Inherited
tombstones and source rewrites retain their frozen priority. The formal decoder
does not search arbitrary malformed bytes for shapes that resemble a legal
seven-node tree.

## 4. Cross-version identity

A canonical v1.4.0 program survives exactly when its normalized node count is
at most six. Every survivor retains identical canonical AST CBOR bytes,
`HEGEL/AST/V1` hash, operator and registry IDs, and MDL length. A seven-node
parent program is rejected; it is never truncated, repaired, split, or
automatically migrated.

Stable program bytes do not permit reuse of a semantic/archive identity. Any
future v1.5.0 archive or formal object must bind the new DSL/freeze roots.

## 5. Non-formal diagnostic bindings

These roots are synthetic engineering bindings, not formal CBOR/RFC6962 roots:

| field | SHA-256 hex |
|---|---|
| `child_dsl_spec_root` | `3340b3278caf562b560cc30cd14d3cd5f1d628e222b43d29d9d1e41b379f5675` |
| `operator_semantics_root` | `5d2700884ae7125b9712a2bd06aa929feaf2fad1d4bfcd4fa5953c157a720ee1` |
| `identifier_registry_root` | `1b0c141126b278778009d3ebbbf49f5de231ad0166a88a8a9caf367b35bff8ef` |
| `canonical_ast_schema_root` | `828fdcc9f16ebd590702ff4297cac6f6ffa19b01299ea7a93753a4fced0961c5` |
| `canonical_cbor_profile_root` | `0ccbd740c0b1f6a39fb8151ea56e114561093ee4fccb228bf83a9294e0bae783` |

Their exact preimages live in
`phase3_m3_shrink5_diagnostic_profile_v1.py`. They bind no target, split,
seed, key, signature, holdout, or formal-run identity.

## 6. Qualification obligations

Before a shrink-5 enumerator is admitted, independent Python and Rust
recognizers must be driven from one immutable source commit over:

1. the sealed ordered 22-vector strict manifest `S01..F10`, covering survivor
   identity, normalization before the node check, source/formal seven-node
   rejection, and inherited malformed/tombstone/noncanonical priorities;
2. the complete inherited 175-source atom survivor set and the complete
   inherited 2,160-source AND2 boundary set;
3. exact report schemas and both structural-limit fields; and
4. commit-bound, target-free, offline container controls.

The 175 atom sources (15 constant comparisons, 16 rational aggregates and 144
mixed comparisons) exercise byte/hash/MDL-stable survivors at six or fewer
nodes. All 2,160 inherited boundary sources have canonical v1.4.0 node count seven.
This is an observed property of the preregistered source set that the sealed
qualification must independently replay in both implementations. The required
child disposition for the boundary set is therefore 0 accepted and 2,160
`REJECT_STRUCTURAL_LIMIT` results at each of the source and formal boundaries.
The report must bind the 175-survivor accepted set, the exact 2,160 parent-only
input set, and the ordered source/formal rejection outcomes. The boundary set
is not a survivor set and neither set is a complete closure execution.

Equal aggregate counts are insufficient. Per-vector normalized outcomes,
ordered roots, exact capacity fields, input/rejection commitments and all
repository/runtime receipts must match. Unknown, missing, aliased, or extra
fields fail closed.

## 7. Isolation and two-commit rule

The supervisor executes only a committed Git archive. Python and Rust use
distinct digest-pinned images, `--pull=never`, `--network=none`, a read-only
root filesystem, dropped capabilities, no-new-privileges, fixed resource
limits and fresh tmpfs. Rust builds `--release --locked --offline` from
read-only Cargo cache/index bytes committed by a pre/post manifest; no
pre-unpacked registry source tree is mounted.

This is technical process/container independence under one administrative
controller. It is not independent-human or organizational custody.

The source and observation must be separate local commits:

1. Source Commit S freezes implementation, vectors, profiles, tests and this
   protocol with no observed qualification result.
2. Qualification runs only from immutable Source S.
3. Evidence Commit T may record the Source-S-bound successful report.

A pass authorizes only development and later qualification of independent
shrink-5 complete enumerators. It cannot start M3.

## 8. Claim boundary

No source-freeze or strict-qualification artifact may claim a shrink-5 closure
cardinality, witness, target match, outside-language result, formal root,
certificate, signature, Phase-3 exit, Phase-2B exit, or ACTIVE governance
change. The next route, if any, is determined only by a later independently
replayed complete diagnostic under the already frozen shrink order.
