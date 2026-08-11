# Hegel Machine Goal-to-Evidence Gap Ledger v1

## 1. Purpose and precedence

This ledger maps the original research goal to current executable evidence.  It
exists because historical documents reuse phase numbers for different
capabilities.  Capability IDs `C1` through `C8`, not a bare `Phase N` label,
control claim order here.

The source goal was audited against these four documents in full:

| Source | SHA-256 |
|---|---|
| `markdown/GPT_advice&roadmap.md` | `5387c8ba991ad65e1d0351b3cd1bcdb77d264542b7f52346a56454c2ee94e607` |
| `markdown/Hegel_assumption.md` | `a1dcd527a3b150c92b86af7709c833a63fa75c2026ae92d3c39d1171ddd006c8` |
| `markdown/黑格尔机.md` | `42559c60fd4852e03e2c8f41a608fa01d6a32c2d1be8de72b834a8dcbe4fe4db` |
| `markdown/黑格尔机和泛函分析.md` | `4a8d9a74db75c3788e3b2dc5cff27fd99e1b9db4f04f9439f8f4baab4f3c2b53` |

The implementation baseline reviewed for this ledger is local commit
`d55c017fb5ae16a83ff57758cdb3e9b57c29afb9`.  Evidence is classified as:

- **OBSERVED**: replayable artifact or test result supports the narrow claim;
- **MECHANICS**: executable machinery exists, but no independent effect result;
- **SPECIFIED**: exact contract exists, but execution or implementation is absent;
- **NOT RUN**: runnable or partly runnable path has no qualifying execution;
- **MISSING**: the required capability is not implemented.

Compilation, fixture success, a protocol freeze, a capacity count, or an
engineering qualification must never be promoted to an effect claim.

## 2. Deduplicated goal

The target is not merely to enumerate more hypotheses.  It is to detect when
the current hypothesis language is inadequate, exhaust cheaper explanations,
propose a minimal executable and falsifiable language change, make a new
prediction, and conservatively version, replay, reduce, or roll back that
change.  The state transition of interest is therefore
`RelationLanguageSnapshot_t -> RelationLanguageSnapshot_t+1`.

The dependency order is:

```text
G0 evidence and claim boundary
  -> G1 immutable theory state and separated authorities
  -> G2 initial ontology, probes, scales, identifiability
  -> C1 known structural-law identification
  -> G4 inadequacy diagnosis and minimal operator choice
  -> C2 bounded relation/meta-prior invention
  -> C3 conservative multigeneration evolution
  -> C4 open-world theory-language evolution
  -> C5-C8 mathematical/scientific/human-comparison programs
```

`C2` cannot inherit a pass from syntactic enumeration, and `C3+` cannot inherit
a pass from one bounded invention fixture.

## 3. Current capability map

| Capability | Current status | What is actually supported | Missing exit evidence |
|---|---|---|---|
| G0 claim/evidence boundary | OBSERVED / strong | frozen schemas, content IDs, replay records, explicit non-claims, fail-closed receipts | one canonical cross-generation status ledger was missing; this file starts it |
| G1 theory/governance kernel | MECHANICS | `TheoryState`, typed observations, laws, probes, patches, reduction maps, separated governance decisions, branch/version scaffolds | independent promotion trust root and real lifecycle evidence |
| G2 initial language/geometry | MECHANICS | 22 assumption templates, six executable active laws, task-relative probes/quotients and scale contracts | one first-class relation-language snapshot manifest; complete executable 22-leaf library; identifiability/gauge intervention evidence |
| C1 known-law identification | PARTIAL | Phase-2A: six verifier families, 43 synthetic cases, 24 answerable/19 abstain, complete 24-projection controlled grid, semantic fields excluded from acceptance | formal Phase-2B typed-evidence projection, sealed independent data, covert audit, baselines, custodian/runtime evidence and statistical exit |
| G4 inadequacy diagnosis | MECHANICS | ordered refit/noise/scope/mixture/composition diagnosis and robustification/idealization evaluator prototypes | diagnosis derived from real residual clusters; executable probe-addition and operator-selection evidence |
| C2 bounded relation invention | MISSING effect | strict DSL/certificate and bounded-enumeration infrastructure; old DSL reached `DSL_TOO_LARGE` under its budget | incomplete-ontology benchmark, at least three outside-library relations, non-equivalence/MDL, hard negatives, counterfactuals and unseen predictions |
| C3 conservative multigeneration | MISSING effect | certificate and version-graph schemas | real old-success preservation, limiting cases, reduction maps, demotion/rollback, negative memory and multigeneration survival |
| C4 open-world evolution | MISSING | no qualifying effect evidence | stream unknown rejection, drift, merge/split, forgetting audit, confidence decay, reopen and bounded language growth |
| C5-C8 long-horizon claims | FUTURE | no qualifying evidence | mathematical concept invention, computational/empirical science discovery, external reproduction and matched human comparison |

## 4. Engineering tracks that must not be confused with the goal DAG

- Legacy formal M3 executed and ended `DSL_TOO_LARGE` after the bounded
  canonical-program budget.  It is not a closure-complete target verdict or
  invention result.
- Q0 is a 14/14 dual micro-projection engineering qualification, not C1/C2.
- Q0.5a freezes Q1 archive/projection machinery; its authority remains
  `SPECIFICATION_FROZEN_IMPLEMENTATION_NOT_QUALIFIED`.
- The only Q0.5b actual attempt failed closed at Stage 7 with 0/20 and no
  artifact.  The adapter bug is fixed in local commit `d55c017...`, but another
  exactly-once actual run requires separate explicit authorization.
- Q1 remains `NOT_RUN`, 0/20, with eight null formal roots.  Q2 and the
  outside/null-control verdicts are also not effect evidence.

These tracks can harden a future C2 certificate, but they do not bypass C1.

## 5. C1 critical path from the present repository

The closest missing big-goal gate is **C1 known structural-law identification
without semantic acceptance**.  The dependency order is:

1. **Projection mechanics** — deterministic public evidence -> complete
   family/binding/scale grid -> verifier evaluations; no caller-signed subset.
2. **Formal numeric semantics** — bundle-atomic exact RationalValue-grid outward
   uncertainty receipts and exact residual/tolerance/normalized-interval/margin
   propagation are implemented for all six laws, but only through the
   root/identity, dimensionless, support-aligned bridge.  `standard_error`
   remains unsupported until its full Student-t/Bonferroni contract exists.
3. **Complete transform semantics** — temporal/spatial aggregation, sampling,
   unit conversion, affine coordinates, split/merge and coarse graining must
   have frozen executable meanings and preservation controls.
4. **Recognizer and archive** — strict CLI, complete prediction archive,
   challenge/preservation evaluator and deterministic replay.
5. **Trusted wire and anti-leak audit** — independent shuffle/ID/padding keys,
   fixed 65,536-byte envelopes, global renaming/order invariance, NMI and
   stratified permutation/Holm-Bonferroni gates.
6. **Independent data** — validated 720 main + 240 semantic-conflict cases and
   496 legal + 76 invalid derived pairs, generated only after validation passes.
7. **Execution authority** — pinned baselines/image/SBOM, detached runtime
   attestation and a durable signed one-shot custodian/CAS ledger.
8. **Formal C1 exit** — consumed sealed report passes overall and family/scale
   slice thresholds.  Only then may C2 relation invention open.

## 6. Implemented mechanics deltas

`phase2b_projection_compiler.py` implements a deliberately narrow first item:

- replays the adapter's complete candidate grid;
- matches each verifier observable to exact quantity, role/entity witness and
  source-observation provenance;
- compiles Boolean values and binary64 `absolute_bound` values/intervals;
- uses directed binary64 rounding, never midpoint substitution;
- permits verifier scoring only for dimensionless observations with identical
  temporal and spatial support and a conservative binary64 arithmetic domain;
- evaluates only degenerate point envelopes; non-degenerate numeric intervals
  are explicit full-grid errors because corner scores do not generally bound a
  nonlinear verifier residual;
- supports root and explicit identity scale paths only;
- rejects `standard_error` at bundle preflight;
- represents missing/ambiguous witnesses, shape drift, unsupported transforms
  and verifier abstention as fail-closed outcomes without a partial grid.

`phase2b_uncertainty_compiler.py` adds the next narrow edge without weakening
that boundary:

- compiles the complete `PublicEvidenceBundle` atomically, not a caller-selected
  favorable subset;
- treats each normalized binary64 as the exact value returned by
  `Fraction.from_float`, rather than reconstructing a lost JSON decimal lexeme;
- expands point and interval `absolute_bound` values exactly and rounds lower
  bounds down / upper bounds up to the literal-pinned 663-point RationalValue
  grid;
- preserves Boolean and missing observations as typed rows;
- rejects any `standard_error` or out-of-grid endpoint for the whole bundle,
  returning no compiled siblings;
- binds bundle, observation, exact-freeze, grid and policy identities into
  immutable result roots; exact-type checks reject subclass commitment spoofing;
- contains no selector/projection import and no float interval bridge.

`phase2b_exact_bridge_v1.py` now closes the next root/identity-only edge:

- its authoritative API accepts only the raw `PublicEvidenceBundle`, theory and
  adapter registry, then recomputes the exact uncertainty receipt and complete
  adapter grid internally; callers cannot inject receipts, grids, evaluations or
  selections;
- binds bundle, observation-compilation, uncertainty, adapter, bridge-policy,
  exact-verifier and exact-selector identities through every post-preflight
  compilation and decision; an uncommitted preflight rejection has no content
  root or run ID and cannot enter the selector;
- implements conservative natural rational intervals for all six frozen law
  residuals using exact add/subtract/multiply/absolute/max/positive-division and
  fail-closed discrete-branch checks, without corner enumeration or conversion
  to float; a negative-feedback interval crossing the zero branch is an error
  cell rather than an invented continuous interpolation;
- keeps tolerance, normalized intervals and structural-margin decisions exact;
- freezes entity/role/quantity/channel/membership, transform,
  observation/scale/edge/vector-width/total-component/candidate, adapter-scan,
  operation-count, Fraction bit-length, theory/tree-node, authority-text and
  integer-bit-length resource budgets; a bounded recursive exact-type walk
  prevents nested schema subclasses from separating compiled values from
  committed mappings. Rejected authorities stop before content hashing and
  receive no committed downstream receipt;
- rejects unused or missing transform-catalog entries, nonunique paths,
  non-identity transforms, dimensioned observations, support mismatch and
  uncertain verifier preconditions fail-closed.

Accordingly these narrow flags are true:

```text
formal_rational_grid_uncertainty_compiler_implemented
absolute_bound_uncertainty_semantics_compiler_implemented
bundle_atomic_exact_uncertainty_receipt_implemented
root_identity_six_law_exact_rational_residual_interval_semantics_implemented
exact_rational_selector_bridge_implemented
authoritative_exact_bridge_recomputes_uncertainty_and_adapter_internally
exact_uncertainty_receipt_consumed_by_root_identity_bridge
oversized_bundle_theory_or_registry_rejected_before_content_hash
nested_authority_exact_type_enforced_before_content_hash
```

while the broad flags below remain false:

```text
projection_compiler_implemented
uncertainty_semantics_compiler_implemented
standard_error_uncertainty_semantics_compiler_implemented
exact_uncertainty_receipt_consumed_by_projection_compiler
exact_rational_residual_interval_semantics_implemented
typed_evidence_to_prediction_pipeline_complete
formal_recognizer_run_runnable
covert_channel_audit_implemented
ready_for_holdout_generation
formal_phase2b_exit_claim
active_promotion_enabled
```

This is implementation progress, not sealed C1 evidence. The new exact bridge is
an internal root/identity recognizer core, not the frozen recognizer CLI or a sealed
selector execution. The older binary64 projection compiler still does not consume
the exact receipt, and its broad completion flag therefore remains false. Full
transform semantics, `task_target`/unused-observation coverage, trusted wire,
covert-channel audit, archive evaluator, signed image/runtime evidence and durable
one-shot custody remain formal-pipeline blockers.

## 7. Next authorized construction slice

The next non-actual construction slice is **Phase-2B Complete Transform Semantics
v1**. Add one frozen executable operation at a time—unit conversion, coordinate
affine, temporal/spatial aggregation, sampling resolution, equivalent split/merge
and coarse-graining—with exact interval propagation, dimensional/support contracts,
legal-preservation pairs and invalid-transform controls. No operation may be
inferred from its name or silently treated as identity. In parallel, the runner
inventory and recognizer CLI still need to include the exact bridge, but CLI/image,
trusted wire and archive-evaluator work must remain separate from claims about
transform correctness or sealed execution.

No Q0.5b actual rerun, Docker execution, holdout generation, or ACTIVE mutation
is authorized by this ledger.
