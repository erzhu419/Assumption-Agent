# Phase-3A M2.5 implementation status

Date: 2026-08-01
Machine freeze under review: `hegel-freeze-p2b-p3-v1.1.1`
Child DSL: `hegel-old-dsl-v1.1.0`
State: `M25_FOUNDATION_IMPLEMENTED_NORMATIVE_COMPLETION_REQUIRED / NOT_RUN`

## Outcome

M2.5 construction has started, but authoritative commitment has not. The
repository now separates two claims that must not be collapsed:

1. deterministic formal-wire and split-crypto primitives can be implemented,
   replayed, and tested without secrets;
2. authoritative roots require a bit-exact completion amendment plus real,
   independent custodian and auditor actions.

Consequently the M3 count remains `14/24`. No formal root, split seed,
signature, hidden-access ledger claim, execution manifest, or state transition
has been minted.

## Implemented foundation

- Python strict numeric-array object registry for the schemas whose prefix,
  field order, and ContentHash domain are explicit;
- fail-closed handling for every schema/domain conflict or omitted definition;
- Python HKDF-SHA256 role-key derivation, HMAC row rank, seed commitment,
  Ed25519 key-ID/signature-preimage helpers, all as pure functions;
- an independent Rust strict-CBOR/ContentHash/RFC6962/HKDF/HMAC implementation
  and JSON replay CLI;
- 20 synthetic shared vectors: 7 positive primitive cases plus 13 strict-CBOR
  rejection/precedence cases with exact Python/Rust error-code parity; these never
  generate or store a real seed or private key;
- an executable diagnostic readiness report that asserts all 15 run-output
  roots are null and rejects any attempted authority/state escalation;
- an append-only binding of the later v2 SCAR negative result, without changing
  the frozen v2 legacy snapshot.

The implementation intentionally does not construct odd/sink formal roots yet:
their `canonical_input_object`, input-signature enum, hash rule, and bridge
profile still require normative completion. A plausible encoding is not a
formal identity.

## Gate state

| Gate | State | Reason |
|---:|---|---|
| 1–14 | satisfied | unchanged shrink-1 qualification |
| 15 seed genesis | blocked | independent custodian/key/one-shot seed plus missing root preimages |
| 16 ledger genesis | blocked | depends on gate 15 and real custodian signature |
| 17 parent absence | blocked | independent audit and signature wire not frozen |
| 18 formal bindings | blocked | missing custodian core and root DAG definitions |
| 19 spec/registry roots | blocked | ID/enums/content preimages incomplete |
| 20 odd universe/truth | blocked | canonical row payload/hash/profile incomplete |
| 21 sink universe/truth | blocked | canonical row payload/hash/profile incomplete |
| 22 split roots | blocked | no real seed; sink 85-row exhaustive partition not frozen |
| 23 state/receipt goldens | blocked | state prefix/closure enum/agreement construction unresolved |
| 24 execution identity | blocked | prior gates, output-null carrier, and implementation qualification unresolved |

## Normative blockers that change bytes or state

- `CustodianBindingCoreV1` is referenced but has no tag/schema/fields/domain.
- `BucketAccountingRecordV1` has a tag but no record schema.
- many required `*_id_digest` values and numeric enums have no derivation or
  registry.
- required ContentHash roots have names but no canonical preimage objects.
- odd/sink `canonical_input_object` and `canonical_input_hash` are not unique.
- sink specifies only minimum support for 32 rows, not an exhaustive allocation
  of all 85 rows; role/stratum/partition IDs are also missing.
- the state-record example conflicts with the global object prefix convention.
- role receipts require a completed enumeration agreement, while the only
  completed dual-agreement form requires role receipts, creating a cycle.
- no formal object carries the 15 required pre-run null output slots.
- v1.1.1's replacement gate list no longer explicitly qualifies both complete
  enumerators, traversal/bucket accounting, and archive emitters; non-null
  source/contract roots alone cannot prove executable implementations.
- the bridge-specific 3/3 signature requirement from v1.0.2 is not clearly
  superseded by the decision to defer final certificate signatures to M4.

The full, answer-shaped request for the next normative amendment is
[questions_for_gpt_phase3_m25_wire_completion.md](questions_for_gpt_phase3_m25_wire_completion.md).

## External actor boundary

Codex has not declared itself an independent custodian or auditor. Tests may
use public deterministic byte strings, but a test seed/key can never satisfy a
gate. After the specification is completed, the recommended choreography is:

1. commit the completed specification, implementations, and shared golden
   vectors;
2. an independent custodian runs a one-shot command outside the repository,
   generating and retaining the key/seed under the frozen storage policy;
3. an independent auditor signs the historical absence attestation;
4. Python and Rust replay the sealed allocation through an approved secret
   transport and publish only permitted public roots/envelopes;
5. commit public manifests and the diagnostic 24/24 qualification artifact;
6. only then create the initial `NOT_RUN -> RUNNING/CANONICAL_ENUMERATION`
   record and begin M3.

The exact two-commit choreography, actor independence, secret transport, and
signature wires are themselves among the questions that must be frozen.

## Effect of v2 commit `4861b2d8`

The v2 result is a protocol-valid negative for one concrete SCAR
operationalization: fixed extractor/binder, hard structural eligibility, and
length-2 composition. It did not execute T01–T22 one by one, and the 13 legacy
items are aliases rather than a second set of independent priors.

Therefore it has no effect on M2.5 byte identity, M3 closure, the odd/sink
targets, the split seed, or the formal outside-certificate predicate. It does
affect Phase-3B/3C experimental design: v2 thresholds and hard eligibility must
not be imported as verified positive priors, and later synthesis must include
matched no-prior, frozen-v2, semantic-only, and Hegel-invented arms with
coverage/no-op and old-success-preservation controls.

The source-bound assessment is
[v2_scar_negative_impact_assessment.md](v2_scar_negative_impact_assessment.md).

## Current claim boundary

Allowed:

> M2.5 deterministic foundations are implemented or under synthetic dual
> replay, and the missing normative/external inputs are explicitly fail-closed.

Not allowed:

- formal M2.5 qualified;
- split seed first-instantiated;
- 15–24 gates passed;
- M3 started or closure complete;
- odd target outside or sink control inside;
- a new relation invented;
- an outside/MDL certificate issued.
