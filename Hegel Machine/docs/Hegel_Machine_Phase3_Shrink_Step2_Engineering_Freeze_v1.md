# Hegel Machine Phase-3 shrink step 2 engineering freeze v1

Status: normative engineering freeze for implementation and diagnostic qualification. This document does not create child formal roots, reuse a parent archive, start a child M3 run, evaluate a target role, or issue a certificate.

## 1. Trigger and parent evidence

Shrink step 2 is activated only by the committed parent terminal carrier:

```text
parent result carrier commit  db612e403bb46e6a295fed01e85649f8af0924b4
parent DSL                    hegel-old-dsl-v1.1.0
parent effective freeze       hegel-freeze-p2b-p3-v1.1.2
parent run ID                 e4af9f57c38fb298462ec628c4ed8a03
parent terminal state         DSL_TOO_LARGE/NONE
parent terminal file SHA-256  4f631224383297f6f30d70dbcefc15ed1c1296ba634a604e5a59562d11e67aed
parent terminal self-hash     973214b278e0bd3af474fa0b095e518e1ea8323917845b856e8b5de72913c67c
parent terminal state root    1f54925f8187c955ae4e7b2c9bb83144ce6627468153d8e1696b27c4705d23dd
parent dual agreement root    48454aef57c3b560ee2f05e46b1dae1f4cd3e9fd0de52b392083a7c8b2359d83
```

The parent run is terminal and has no same-version transition to role evaluation. A cross-version shrink transition must use `parent_status=DSL_TOO_LARGE` and initialize the child at `NOT_RUN/NONE`.

## 2. Child identity and sole language delta

```text
child DSL          hegel-old-dsl-v1.2.0
child freeze       hegel-freeze-p2b-p3-v1.2.0
human amendment    hegel-freeze-p2b-p3-v1.2.0-shrink-step2
shrink step        SHRINK_STEP_2_REDUCE_RATIONAL_PARAMETER_TO_NEG1_ZERO_POS1
```

The only admitted surface change is:

```text
RationalParameter {-2,-1,-1/2,0,1/2,1,2}
               -> {-1,0,1}
```

Typing, bottom semantics, equivalence, scope, aggregate-map admission, structural limits, closure budgets, target/control payloads, MDL code table, AST schema, canonical CBOR profile, and AST hash domain remain unchanged.

## 3. Sparse RationalParameter registry

The inherited three-bit `RationalParameterId/v1` allocation is never compacted:

| ID | Exact value | Child state |
|---:|---:|---|
| 0 | -2 | TOMBSTONE |
| 1 | -1 | ACTIVE |
| 2 | -1/2 | TOMBSTONE |
| 3 | 0 | ACTIVE |
| 4 | 1/2 | TOMBSTONE |
| 5 | 1 | ACTIVE |
| 6 | 2 | TOMBSTONE |
| 7 | reserved | UNALLOCATED |

The machine rules are:

```text
active_ids                  = [1,3,5]
tombstone_ids               = [0,2,4,6]
wire_width_bits             = 3
dense_reindex_allowed       = false
tombstone_reuse_allowed     = false
removed source/formal error = REJECT_REMOVED_RATIONAL_PARAMETER
reserved/unknown error      = REJECT_REGISTRY_INDEX_OUT_OF_RANGE
```

The shrink-1 sparse aggregate registry is inherited exactly: IDs `0,1,5` are active and `2,3,4` remain permanent tombstones.

## 4. Canonicalization consequence

Surviving parent programs keep identical canonical AST CBOR bytes and `HEGEL/AST/V1` hashes. The fixed three-bit MDL codeword remains attached to the original numeric ID.

The existing constant-fold rule is interpreted against the child active grid. A fold is emitted as `scalar_const` only when its result is active. An inactive result retains the normalized operator AST:

```text
add(1,-1)        -> scalar_const(0 / ID 3)
add(1,1)         -> add(ID 5, ID 5)
add(-1,-1)       -> add(ID 1, ID 1)
difference(1,-1) -> difference(ID 5, ID 1)
difference(-1,1) -> difference(ID 1, ID 5)
```

This is not permission to reintroduce IDs `0,2,4,6`, automatically migrate a removed literal, or reject an otherwise valid active-input operator program merely because the parent would have folded it to a removed literal.

## 5. Strict dual-implementation qualification

Python and Rust must independently verify at least:

- source and formal rejection of IDs `0,2,4,6` with the exact removal code;
- ID `7` rejection as out of range, not as a tombstone;
- inherited aggregate tombstone rejection;
- bytes/hash stability for surviving parent programs;
- child canonical round-trip for retained inactive-fold operator ASTs;
- active-result constant folding;
- malformed input error priority without scanning arbitrary payloads;
- identical canonical CBOR and accepted-set commitments.

Neither implementation may use the other implementation's output as its canonicalization authority.

The exact shared rejection matrix is:

| Boundary input | Required code |
|---|---|
| source malformed node/arity, non-integer rational-pair component, or non-integer `approx_equal` shorthand | `REJECT_MALFORMED_SOURCE_AST` |
| source registry/index field with an invalid type or value, including negative or host-width overflow | `REJECT_REGISTRY_INDEX_OUT_OF_RANGE` |
| source rational denominator not positive or exact pair outside the frozen grid | `REJECT_REGISTRY_INDEX_OUT_OF_RANGE` |
| source empty `top_level_AND` | `REJECT_EMPTY_CONJUNCTION` |
| source `add`/`difference`/exact comparison receiving any `Bit` child | `REJECT_IMPLICIT_COERCION` |
| other source child-sort mismatch | `REJECT_TYPE_MISMATCH` |
| source or formal `new_symbol_call` in the old DSL | `REJECT_NEW_SYMBOL_IN_OLD_DSL` |
| formal schema version other than 1 | `REJECT_UNKNOWN_AST_SCHEMA` |
| formal unknown AST-node or leaf tag | `REJECT_UNKNOWN_EXPRESSION` |
| formal registry/operator field negative or out of range | `REJECT_REGISTRY_INDEX_OUT_OF_RANGE` |
| formal binary ID 4 or 7, malformed shape, noncanonical ordering, or invalid arity | `REJECT_NONCANONICAL_AST` |
| formal child-sort mismatch, including an unconverted `Bit` | `REJECT_TYPE_MISMATCH` |
| truncated, reserved-additional-info, undefined, or unapproved-simple formal CBOR | `REJECT_TRUNCATED_CBOR`, `REJECT_RESERVED_CBOR`, `REJECT_CBOR_UNDEFINED`, or `REJECT_CBOR_SIMPLE`, respectively |
| structurally legal mixed tombstones | aggregate-map rejection before rational-parameter rejection |

Rational-pair source aliases use exact integer arithmetic rather than host-width arithmetic. Arbitrarily wide JSON integers that reduce exactly to a frozen grid value are admitted as that immutable numeric ID; other numeric values fail with the range code above. The Rust source boundary parses left-to-right directly; it must not pre-transform an unvisited sibling because doing so would alter the frozen failure priority. The Rust formal boundary performs the bit-exact CBOR preflight and validates each completed child subtree before visiting later siblings or tolerance fields.

The shared golden suite contains 59 vectors: 7 surviving-identity checks, 7 fold/operator checks, 27 source rejections, 15 source boundary checks, 3 arbitrary-width source checks, 1 source-malformed check, 1 mixed-tombstone priority check, 17 formal rejections, and 12 formal failure-code checks. The latter categories overlap their source/formal rejection totals.

## 6. Constructive subset precommitment

The target-independent shrink-2 capacity subset is frozen before it is used as a gate:

```text
active rational constants    3
constant comparison atoms    C(3+1,2) + 3^2 = 15
rational aggregate leaves    2 maps x 4 scopes x 2 quantities = 16
mixed comparison atoms       3 x 16 x 3 directions = 144
source AND2 candidates       15 x 144 = 2,160
```

The generator uses original parameter IDs `1,3,5`, rational aggregate IDs `0,5`, empty scope extensions, `equal_exact`, ordered `less_equal`, and canonical two-clause `top_level_AND` construction. It contains no target, split, seed, key, evaluator, or role input.

Python qualification must invoke `phase3_shrink2_capacity_entrypoint_v1.py` directly under `python -I -S -B`. The entrypoint installs a minimal package shell and fails if its exact nine-module target-free dependency closure expands. An ordinary `import hegel_machine...` is not isolation evidence because the historical public package initializer intentionally exports target-bearing APIs. Rust qualification uses `cargo --locked --offline` in `rust@sha256:38bc5a86d998772d4aec2348656ed21438d20fcdce2795b56ca434cf21430d89`, with `network=none`, `pull=never`, a read-only source mount, and no Git dependency.

Even exact dual acceptance of all 2,160 unique candidates means only that this constructive subset is admitted consistently. By construction it cannot test the 50,001-program boundary. It is never `COMPLETE`, a closure cardinality, an adequacy result, or an outside-language verdict.

## 7. Publication and formal requalification gates

After strict and subset qualification, the diagnostic child publication still carries `formal_roots=null` and `execution_state=NOT_RUN`. Before any child M3 start, all child-specific identities must be regenerated and independently replayed:

- DSL spec, operator semantics, parameter registry, aggregate registry, AST/CBOR and bounded-universe roots;
- odd and sink DSL-binding manifests while retaining byte-identical payload content IDs;
- split/custodian/seed-continuity manifests without redraw when the existing seed remains uncompromised;
- append-only hidden-access ledger continuity;
- Python/Rust implementation bindings and archive emitters;
- bridge, attestations, execution manifest, run genesis, run ID, and 24/24 entry gates.

All new run output slots remain null at 24/24. A separate explicit start is still required. Parent program archives, receipts, roots, run ID, and execution manifest are evidence for the transition only and are forbidden as child formal outputs.

## 8. Legal successor states and claim boundary

The 2,160-source subset cannot expose a 50,001st unique program and has no terminal routing authority. After subset qualification, an independently implemented complete child enumeration is still required. Only a closed frontier at or below 50,000 programs may enter role evaluation. A full-enumeration 50,001 witness routes to the preregistered shrink step 3: remove `add`, retain `difference`. Raw-budget, semantic, implementation, or replay failures remain fail-closed and carry no match verdict.

This shrink-2 freeze does not claim `COMPLETE`, odd-target outside status, hidden-sink recognition, relation invention, MDL success, Phase-3 exit, M4 signatures, Phase-2B exit, or ACTIVE governance.
