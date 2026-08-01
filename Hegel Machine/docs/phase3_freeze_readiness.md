# Phase-3 old-language freeze readiness（historical pre-v1.0.2 snapshot）

> **STATUS: `SUPERSEDED_FOR_NORMATIVE_DECISIONS`**
>
> **SUPERSEDED BY:**
> [`hegel-freeze-p2b-p3-v1.0.2`](Hegel_Machine_Strict_Canonical_AST_CBOR_Certificate_Bridge_Freeze_v1.0.2.md)
> and the
> [Phase-3 readiness resolution](Hegel_Machine_Phase3_Freeze_Readiness_Resolution.md).
>
> **HISTORICAL VALUE:** this file preserves the implementation-audit state that
> exposed the strict-identity gaps before target outcomes were observed. The
> historical snapshot beginning after the M1/M2 overlay below is not the
> current normative truth.

## Current authoritative state after M1/M2 execution

| Layer | Current state |
|---|---|
| surface and strict specification | **FROZEN** in v1.0.2 |
| Python/Rust strict implementation and shared golden vectors | **VERIFIED: 48/48 each** |
| bounded M2 capacity replay | **`DSL_TOO_LARGE`** for `hegel-old-dsl-v1.0.0` under the 50,000 syntactic budget |
| complete closure / extensional target verdict | **NOT AVAILABLE**; this status is not `COMPLETE` |
| formal roots / outside or MDL certificate | **NOT GENERATED / NOT ISSUED** |

```json
{
  "freeze_version": "hegel-freeze-p2b-p3-v1.0.2",
  "surface_parameter_freeze_complete": true,
  "strict_acceptance_contract_complete": true,
  "normative_parameter_freeze_complete": true,
  "strict_acceptance_implementation_verified": true,
  "python_shared_vectors_passed": "48/48",
  "rust_shared_vectors_passed": "48/48",
  "accepted_strict_canonical_count_python": 64680,
  "accepted_strict_canonical_count_rust": 64680,
  "accepted_unique_count_python": 64680,
  "accepted_unique_count_rust": 64680,
  "diagnostic_set_commitment": "sha256:c1a02a66a8d6d8f75204cb3daf03ab0b01c2b3b8e486d0ab3d481ee3be43c930",
  "first_out_of_budget_ordinal": 50001,
  "first_out_of_budget_ast_hash": "sha256:7c7f786c2cc57d31506b3c61d162d175c7f69a2878a089c72c9d053694cba948",
  "executed_closure_status": "DSL_TOO_LARGE",
  "complete_closure_enumerated": false,
  "extensional_target_verdict": null,
  "formal_archive_roots_generated": false,
  "formal_roots": null,
  "hidden_sink_formal_verdict_allowed": false,
  "outside_certificate_issued": false,
  "mdl_certificate_allowed": false,
  "phase2b_formal_exit": false,
  "active_promotion_allowed": false,
  "required_next_action": "PUBLISH_SHRUNK_OLD_DSL_VERSION_USING_FROZEN_STEP_1"
}
```

The evidence is recorded in the
[dual strict gate artifact](../artifacts/phase3_dual_strict_gate_v1.json) and
[dual strict capacity replay artifact](../artifacts/phase3_dual_strict_capacity_replay_v1.json).
Both implementations accepted all 64,680 source candidates as 64,680 unique
strict canonical ASTs. The shared set commitment is diagnostic, not a formal
RFC6962 archive root. The ordinal-50,001 hash is an overflow witness, not a
closure cardinality or target match result.

The frozen next action is
`PUBLISH_SHRUNK_OLD_DSL_VERSION_USING_FROZEN_STEP_1`: publish a new old-DSL
version, remove `mean_v1`, `min_v1`, and `max_v1`, regenerate all target and
validation commitments, and start the new version's closure state from
`NOT_RUN`. The current v1.0.0 result must never be rewritten to `COMPLETE`.

Current NO-GO items are an extensional target verdict, target synthesis, a
hidden-sink formal verdict, outside/MDL certificate issuance, Phase-2B formal
exit, and ACTIVE promotion. Formal roots remain `null`.

---

## Historical snapshot: frozen surface, unresolved canonical acceptance

The overall audited contract is `hegel-freeze-p2b-p3-v1.0.1`; it supersedes
v1.0.0 because the original 64-bit sklearn `random_state` was not executable.
The master/bootstrap seed remains `411876909552964556`, while sklearn uses the
frozen domain-separated SHA-256-to-uint32 value `2611585425`.

`Hegel_Machine_Phase2B_Phase3_Exact_Freeze_Decisions.md` freezes the Phase-3
DSL surface as `hegel-old-dsl-v1.0.0` and the MDL parameters as
`hegel-mdl-prefix-v1.0.0` with unsigned Q32 lengths. The following are decided:

- primitive finite domains and identifier registries;
- four scopes, six aggregates, and four adapter-only transforms;
- leaf/operator typing, strict-bottom propagation, and exact extensional
  comparison;
- AST, node, clause, bit-slot, aggregate, scope, composition, and parameter
  limits;
- the 50,000 syntactically canonical program budget, 5,000,000 raw-application
  budget, traversal keys, and shrink order;
- target/control universes, support, MDL code-table parameters, replay threshold,
  and the high-level dual-replay/3-of-3 trust design.

This does not yet freeze which candidate syntax is accepted as one strict
canonical AST. The node-level canonical CBOR schema, allowed normalization and
rewrite rules, and several certificate wire identities remain unresolved.
The exact machine status is therefore:

```text
milestone = Phase-3A Bounded Frozen-Closure Adequacy
surface_parameter_freeze_complete = true
strict_acceptance_contract_complete = false
normative_parameter_freeze_complete = false
```

Consequently Phase-3 remains shadow-only, no hidden experiment has started,
the LANGUAGE compiler and ACTIVE promotion are disabled, and no outside or MDL
certificate has been issued.

## Capacity preflight is conditional, not an executed closure

The budget counts syntactically canonical programs before extensional quotient.
The normative state transitions are:

```text
formal enumerator produces canonical program 50,001 → DSL_TOO_LARGE
raw cap is hit before frontier exhaustion               → INCONCLUSIVE_BUDGET
no accepted canonicalizer/enumerator run exists         → NOT_RUN
an executed run detects incomplete semantics            → INCONCLUSIVE_SEMANTICS
```

They do not imply that either budget status has already occurred.

The current Python capacity preflight constructs a deliberately small subset:

```text
77 constant-only atoms × 840 one-aggregate atoms
  = 64,680 diagnostic AND2 candidate ASTs

status = CONDITIONAL_CAPACITY_LOWER_BOUND_EXCEEDS_BUDGET
executed_closure_status = NOT_RUN
```

Every diagnostic candidate is typed and satisfies the provisional depth-2,
7-node, two-clause, one-aggregate, and at-most-three-parameter limits. The
diagnostic replay finds all 64,680 representations distinct, but it uses a
tuple-AST plus canonical JSON. It is not formal canonical CBOR and it does not
apply a still-unfrozen strict canonicalizer. It deliberately excludes most of
the grammar and therefore is a conditional lower bound, not a complete closure
or an extensional target comparison.

The repository must not report `DSL_TOO_LARGE` yet. First freeze and implement
the strict canonical AST/CBOR acceptance rules, then replay this subset. If the
formal canonicalizer accepts the witnesses without collapsing them below the
budget, the transition is:

```text
DSL_TOO_LARGE
  → publish a new DSL version
  → apply frozen shrink step 1: remove mean_v1, min_v1, max_v1
  → regenerate target and validation commitments
```

If canonicalization rejects or merges some witnesses, that fact must follow a
precommitted syntactic rule, not a post-hoc algebraic or extensional reduction.
The full closure still has to be enumerated under the resulting strict schema.

## Receipt bindings and the 50,001st witness

`ClosureEnumerationReceipt` is an untrusted, structurally validated replay
claim for exactly one `target_role`. Every receipt must bind:

```text
contract_id
dsl_spec_id
operator_semantics_id
equivalence_contract_id
enumerator_implementation_id
target_id and target_role
the role-appropriate diagnostic universe and truth-table content IDs
```

The outside target and null control have independent evaluation domains:

| Role | Universe/root binding | Truth binding |
|---|---|---|
| `OUTSIDE_TARGET` | 480-row `bounded_universe_diagnostic_id` | `target_table_diagnostic_id` |
| `IN_LANGUAGE_NULL` | 85-row `hidden_sink_universe_diagnostic_id` | `hidden_sink_target_table_diagnostic_id` |

The evaluator selects the expected diagnostic IDs from `target_role`; using the
outside IDs for the null control, or vice versa, is a binding mismatch and must
fail closed. These canonical-JSON content IDs are not the formal
canonical-CBOR/RFC6962 roots; the formal roots and their preimage/tree bridge
remain unresolved.

The wire can represent a bounded-grammar overflow without claiming full
cardinality. `DSL_TOO_LARGE` requires exactly 50,000 accepted canonical programs
plus a non-null `first_out_of_budget_program_id` for the 50,001st witness. It
forbids closed-frontier flags, a closure cardinality, and raw/wall-clock abort
flags. This is support for a future replay result, not a current result: the
sealed verifier has not recomputed any receipt, the capacity evidence remains
conditional, and executed closure status is still `NOT_RUN`.

## Frozen Phase-3A target

The first target is:

```text
TARGET_P3A_GENERIC_ODD_REDUCTION_V1
Generic Odd-Cardinality Reduction over Bounded Entity Sets
```

It is one permutation-invariant relation over bit-valued `EntitySet` inputs of
size 5–8. The complete universe has `2^5 + 2^6 + 2^7 + 2^8 = 480` rows. The
agent-facing split is 192 discovery / 96 validation / 192 sealed prediction
rows; the full truth table is hidden from synthesis and reserved for exact
old-closure comparison.

Binary XOR remains `TARGET_DESIGN_SANITY_ONLY` until executable closure replay
proves a canonical old-DSL expression on the complete four-row universe. If an
old-DSL program exactly matches the 480-row target, the target becomes
`IN_LANGUAGE_POSITIVE_CONTROL` and the precommitted fallback registry applies.

The source witness also has a typing conflict: it applies `difference` directly
to two `Bit` results, while the frozen signature requires two `RationalValue`
children and provides explicit `bit_to_scalar`. The machine contract therefore
records the source spelling as non-typechecking and separately records the
type-explicit candidate with two conversions. Whether the source intended
implicit coercion is an unresolved canonical-acceptance decision; no executable
XOR verdict is issued from either string.

## Frozen observed omitted-channel null control

`CONTROL_P3A_OBSERVED_OMITTED_SINK_V1` contains four observed channels. The
auxiliary outflow is omitted only by the initial scope and is never latent.
With values in `0..4`, `d = a + b - c`, and `0 <= d <= 4`, its bounded universe
has 85 rows. The successful relation uses `signed_balance_v1`,
`control_volume_all_observed_v1`, `q0`, and zero tolerance; false-invention rate
must remain zero.

The source prose calls the baseline `control_volume_primary_only_v1`, while the
four-member catalog contains `scope_primary_only_v1`. The machine contract
currently uses `scope_primary_only_v1`, records the former as a source alias,
and does not add a fifth scope. This still needs explicit confirmation.

## Canonicalization decisions still required

Before capacity or closure status can advance, a machine-readable amendment
must specify:

1. the canonical CBOR schema for every AST node kind, including tags/field IDs,
   child representation, registry references, clause boundaries, and
   `root_operator_id` extraction;
2. the complete syntactic normalization list: commutative child ordering,
   operator aliases, associative flattening, duplicate/idempotent clauses,
   constant folding, `greater_equal` operand reversal, `approx_equal(..., 0)`
   versus `equal_exact`, AND1 unwrapping, and every permitted algebraic rewrite;
3. whether any algebraic rewrite is allowed before the syntactic-program count;
   extensional quotienting remains forbidden as a completeness shortcut;
4. exact node counting for aggregate leaves and their map/scope/quantity/scope-
   extension payloads, tolerance arguments, and top-level AND wrappers/clauses.

Until this amendment is frozen, diagnostic tuple/JSON identity cannot be
substituted for strict canonical AST/CBOR identity.

## Certificate and MDL boundary

The only allowed formal claim is:

```text
OUTSIDE_FROZEN_CLOSURE(
  dsl_version,
  bounded_universe_root,
  target_truth_table_root,
  equivalence = exact_extensional
)
```

The unbounded shorthand `OUTSIDE_LANGUAGE` is prohibited.

High-level issuance still requires canonical CBOR, RFC-6962 Merkle roots,
4,096-record chunks, independent complete Python/Rust replay, and 3-of-3
Ed25519 signatures. The following strict interfaces remain unresolved and are
grouped into nine specification items:

1. canonical-CBOR backend, version, and exact acceptance profile;
2. strict canonical AST schema and root-operator extraction;
3. program-output blob record, encoding, archive, and root schema;
4. relation and naming among canonical program archive root, program archive
   root, chunk manifest root, and their Merkle preimages;
5. exact identity represented by every `match_program_hash`;
6. exhaustion-receipt root preimage and self-field exclusion rule;
7. final certificate envelope/timestamp plus repository-commit hash algorithm
   and wire format;
8. key-status discovery/trust anchor, Ed25519 key/signature encoding, and exact
   revocation-manifest fields;
9. MDL AST/new-symbol wire schema, the literal 16-bit `NEW_REDUCER_V1` header,
   dual-replay receipt/certificate envelope, and cross-language Q32 `log2`
   reference algorithm.

Schema, Merkle, signature-verification, and Q32 helper code does not resolve
these acceptance questions. No complete Python closure replay, independent
Rust replay, trusted key-status path, 3/3 certificate, or complete formal MDL
AST scorer replay has run.

## Historical go/no-go (pre-v1.0.2)

| Work item | Status |
|---|---|
| Phase-3A Bounded Frozen-Closure Adequacy surface parameters | GO / recorded |
| normative parameter freeze | false / strict amendments required |
| diagnostic capacity subset | `CONDITIONAL_CAPACITY_LOWER_BOUND_EXCEEDS_BUDGET` |
| executed complete closure | `NOT_RUN` |
| strict canonical AST/CBOR amendment | REQUIRED before closure status |
| target synthesis run | NO-GO |
| formal outside or MDL certificate | NO-GO |
| shadow candidate records | GO |
| ACTIVE promotion | NO-GO |

At this historical snapshot the next decision point was canonical acceptance,
not target outcome. v1.0.2 later froze that specification and M1/M2 later
executed it; the current shrink-step next action is recorded in the overlay at
the top of this file.
