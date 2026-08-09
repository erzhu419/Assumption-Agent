# Hegel Machine Phase-3 shrink step 3 engineering freeze v1

Status: **NORMATIVE ENGINEERING FREEZE; NON-FORMAL AND NOT RUN**

This document freezes the implementation metadata for shrink step 3. It does
not create child formal roots, execute closure, evaluate a target role, issue a
certificate, sign an object, or authorize ACTIVE governance.

## 1. Diagnostic trigger and state boundary

Shrink step 3 is reached only through the preregistered engineering route in
the shrink-2 diagnostic result:

```text
diagnostic result commit     d9334589343554841d9f9fd30456a7402bcc7d33
implementation basis         f94cf1fb27c6734f24d4510efba0ca3726132706
evidence record ID           phase3_shrink2_dual_complete_enumeration_diagnostic_e118f3809b2f5eef0ebd1c97936da746472a4188e0cc3feecc3e01688922b966
diagnostic status             DUAL_DSL_TOO_LARGE_HOST_REPLAY_PASS
claim level                  NON_FORMAL_DUAL_CHILD_DIAGNOSTIC
parent diagnostic DSL        hegel-old-dsl-v1.2.0
parent diagnostic freeze     hegel-freeze-p2b-p3-v1.2.0
parent execution state       NOT_RUN
parent formal roots          null
```

Commit `d9334589343554841d9f9fd30456a7402bcc7d33` records a dual diagnostic
result against the Commit-H basis;
it is not a formal terminal, formal closure execution, or full closure
cardinality. Its only authority here is the previously frozen engineering
route: remove `add`, retain `difference`, and do not promote formal state.

The shrink-3 child begins and remains:

```text
execution_state                 = NOT_RUN
formal_roots_generated           = false
formal_roots                     = null
formal_state_transition_allowed  = false
engineering_only                 = true
sealed_dual_strict_outcome_replay = NOT_RUN
```

The diagnostic hashes produced by the Python metadata modules are not formal
CBOR/RFC6962 roots and may never be relabelled as such.

## 2. Child identity and sole delta

```text
parent DSL          hegel-old-dsl-v1.2.0
parent freeze       hegel-freeze-p2b-p3-v1.2.0
child DSL           hegel-old-dsl-v1.3.0
child freeze        hegel-freeze-p2b-p3-v1.3.0
human amendment     hegel-freeze-p2b-p3-v1.3.0-shrink-step3
shrink step         SHRINK_STEP_3_REMOVE_ADD_RETAIN_DIFFERENCE
```

The only language-surface change is:

```text
remove add        BinaryOperatorId/v1 = 0
retain difference BinaryOperatorId/v1 = 1
```

ID 0 remains allocated and becomes a permanent tombstone. It is never
renumbered, reused, migrated, or replaced by another operator. In particular,
an `add` source is rejected even when its operands could have folded to an
active rational constant. ID 1 and the semantics and normalization of
`difference` are unchanged.

The shrink-1 aggregate registry and shrink-2 rational-parameter registry are
inherited exactly:

```text
AggregateMapId active       [0,1,5]
AggregateMapId tombstone    [2,3,4]
RationalParameterId active  [1,3,5]
RationalParameterId tombstone [0,2,4,6]
```

Typing, implicit-coercion prohibition, bottom semantics, exact extensional
equivalence, scope, structural limits, closure budgets, target/control
payloads, MDL code table, AST schema, deterministic-CBOR profile, and AST hash
domain remain unchanged.

## 3. Sparse `BinaryOperatorId/v1` registry

The inherited three-bit allocation is sparse-preserving:

| ID | Name | Source admission | Formal-canonical admission |
|---:|---|---|---|
| 0 | `add` | TOMBSTONE | TOMBSTONE |
| 1 | `difference` | ACTIVE | ACTIVE |
| 2 | `equal_exact` | ACTIVE | ACTIVE |
| 3 | `less_equal` | ACTIVE | ACTIVE |
| 4 | `greater_equal` | ACTIVE | SOURCE ALIAS ONLY |
| 5 | `same_sign` | ACTIVE | ACTIVE |
| 6 | `opposite_sign` | ACTIVE | ACTIVE |
| 7 | reserved | NOT A SOURCE TOKEN | NONCANONICAL |

The exact registry sets are:

```text
active source IDs             = [1,2,3,4,5,6]
active formal-canonical IDs   = [1,2,3,5,6]
source-alias IDs              = [4]
tombstone IDs                 = [0]
reserved IDs                  = [7]
wire width                    = 3 bits
ID compaction allowed         = false
tombstone reuse allowed       = false
automatic migration allowed   = false
removed source/formal error   = REJECT_REMOVED_BINARY_OPERATOR
unallocated registry ID error = REJECT_REGISTRY_INDEX_OUT_OF_RANGE
source numeric operator ID    = not representable in the named source AST
unknown source name error     = REJECT_UNKNOWN_EXPRESSION
formal reserved ID 7 error    = REJECT_NONCANONICAL_AST
```

Source alias ID 4 keeps its inherited rewrite:

```text
greater_equal(left, right) -> less_equal(right, left)
```

Consequently ID 4 is admitted at the source boundary but is never admitted as
a formal-canonical binary node. Its formal noncanonical disposition is checked
at the final rewrite/canonicality stage described below, not before tombstone
priority has been resolved.

## 4. Exact rejection priority

No tombstone scanner may inspect arbitrary malformed source or CBOR payloads.
The complete parent structural/typing/registry validation pass precedes every
tombstone pass:

1. generic source syntax or deterministic-CBOR decoding/profile errors;
2. AST envelope, node structure, tag, arity, and field-shape errors;
3. parent child-sort and implicit-coercion typing errors;
4. parent registry type, range, reserved-ID, and value-grid errors;
5. inherited aggregate tombstone: `REJECT_REMOVED_AGGREGATE_MAP`;
6. inherited rational tombstone: `REJECT_REMOVED_RATIONAL_PARAMETER`;
7. new binary-operator tombstone: `REJECT_REMOVED_BINARY_OPERATOR`;
8. remaining normalization, structural-limit, noncanonical-rewrite, or
   re-encoding error.

Items 1--4 inherit the parent's existing left-to-right error rules; this freeze
does not invent a new relative priority among distinct parent validation
errors. Items 5--8 apply only after the source or formal AST is structurally
legal under that parent validation surface. Within that legal AST, the
tombstone passes are whole-tree passes in the displayed order. Thus an
inherited aggregate tombstone wins over a rational tombstone, a rational
tombstone wins over removed `add`, and removed `add` wins over a remaining
formal rewrite. The formal ID-4 source alias check is part of item 8.

This ordering also applies when normalization would later expose a whole-AST
depth or node-count limit. A source that passes all local shape checks but is
whole-AST oversized and contains `add` returns
`REJECT_REMOVED_BINARY_OPERATOR`; it is not normalized first and does not
return `REJECT_STRUCTURAL_LIMIT`. A locally invalid scope extension or other
field remains an item 1--4 error and is not covered by this rule.

Representative mixed cases are normative:

| Structurally legal content | Required result |
|---|---|
| removed aggregate nested under `add` | `REJECT_REMOVED_AGGREGATE_MAP` |
| removed rational nested under `add` | `REJECT_REMOVED_RATIONAL_PARAMETER` |
| otherwise legal `add` requiring a fold | `REJECT_REMOVED_BINARY_OPERATOR` |
| formal ID 4 containing an inherited tombstone | inherited tombstone code |
| formal ID 4 without a tombstone | `REJECT_NONCANONICAL_AST` |
| malformed/out-of-range node that also contains tombstone-looking bytes | parent malformed/range code |

Source and formal boundaries must both return the exact removal code
`REJECT_REMOVED_BINARY_OPERATOR` for a structurally legal use of ID 0. Neither
boundary may rewrite the removed node, silently drop it, reinterpret it as
`difference`, or automatically migrate a legacy parent program.

## 5. Identity and cross-version stability

A parent v1.2 program survives exactly when it is admitted by the inherited
aggregate and rational registries and contains no `add` node. Every surviving
program retains byte-identical canonical AST CBOR and the same
`HEGEL/AST/V1` hash. Numeric IDs and fixed MDL codewords are not compacted.

Byte/hash stability does not make semantic identity or an archive root
cross-version reusable. Any later formal child qualification must bind newly
constructed v1.3 DSL, operator-semantics, identifier-registry, and run objects.
Parent archives and roots are transition evidence only. A legacy reader must
report:

```text
parent_dsl_version                       = hegel-old-dsl-v1.2.0
parent_effective_freeze_version          = hegel-freeze-p2b-p3-v1.2.0
automatic_operator_migration_performed   = false
```

## 6. Engineering qualification requirements

Before a shrink-3 capacity or enumeration diagnostic may be used, independent
Python and Rust implementations must agree on at least:

- all registry IDs, aliases, tombstones, diagnostic commitments, and inherited
  aggregate/rational dispositions;
- source and formal rejection of ID 0 with the exact removal code;
- the source boundary's named-only operator syntax (numeric operator tokens
  are malformed and unknown names are unknown expressions), plus formal ID 7
  rejection as noncanonical rather than as a tombstone;
- the full mixed-error priority matrix in section 4;
- source ID 4 rewriting to formal ID 3 with swapped children;
- rejection of formal ID 4 only after inherited/new tombstone checks;
- byte/hash stability for every sampled surviving parent AST;
- absence of automatic `add` migration or pre-rejection folding; and
- target-free module closure, network-free pinned execution, and identical
  accepted-set/accounting commitments.

The strict profile is an ordered sealed manifest of 36 vectors: 8 survivor
identity cases, 4 source-add rejections, 6 source-priority cases, 3 formal-add
rejections, 6 formal-priority cases, 6 formal-shape-priority cases, and 3
formal alias/reserved/registry cases. Its diagnostic manifest root is:

```text
sha256:e091e08f33be8bbfa579b6d333f618326b4ed2ebae6d2830d3adc0df7a6333b5
```

Dual qualification must replay the exact ordered input wires and expected
dispositions in that manifest and compare the framed outcome root. Equal
aggregate counts alone are insufficient evidence that the implementations
received the same vectors.

At this source freeze, Python has locally replayed the sealed manifest and the
Rust endpoint has passed its independent 36-vector built-in self-test, but an
external supervisor has not yet driven all sealed wires through both endpoints.
Therefore `SEALED_DUAL_STRICT_OUTCOME_REPLAY` remains `NOT_RUN`; equal 36/36
counts are not a sealed dual qualification result.

Qualification outputs remain diagnostic. A constructive subset cannot prove
closure `COMPLETE`, and a bounded rank-50,001 witness cannot establish the full
closure cardinality. Any mismatch or raw-budget failure is inconclusive and
does not authorize another shrink.

At this freeze commit the complete enumerator is not yet dual-qualified. A
reduced test budget may report only
`DIAGNOSTIC_PREFIX_BUDGET_EXCEEDED`; `DSL_TOO_LARGE` is reserved for the frozen
50,000-program budget with a fully closed boundary bucket and rank-50,001
witness. No Python-only prefix or probe has routing authority.

## 7. Formal requalification and claim boundary

This engineering freeze does not inherit or manufacture child formal roots.
Before any formal v1.3 M3 start, all child DSL/registry/operator roots,
target-role binding manifests, implementation bindings, archive emitters,
bridge objects, execution manifest, run genesis, and entry gates must be
regenerated and independently replayed. Even a future 24/24 gate result means
ready but `NOT_RUN`; a separate explicit start remains mandatory.

Nothing in this document claims a shrink-3 closure result, full closure
cardinality, odd-target outside status, hidden-sink recognition, relation
invention, MDL success, `OUTSIDE_FROZEN_CLOSURE(...)`, Phase-3 exit, M4
signatures, Phase-2B exit, or ACTIVE governance.
