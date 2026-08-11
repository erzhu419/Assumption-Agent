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
   uncertainty receipts are implemented for `absolute_bound`; the still-missing
   edge is exact residual/tolerance/verifier propagation.  `standard_error`
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

Accordingly these narrow flags are true:

```text
formal_rational_grid_uncertainty_compiler_implemented
absolute_bound_uncertainty_semantics_compiler_implemented
bundle_atomic_exact_uncertainty_receipt_implemented
```

while the broad flags below remain false:

```text
projection_compiler_implemented
uncertainty_semantics_compiler_implemented
standard_error_uncertainty_semantics_compiler_implemented
exact_uncertainty_receipt_consumed_by_projection_compiler
exact_rational_residual_interval_semantics_implemented
exact_rational_selector_bridge_implemented
typed_evidence_to_prediction_pipeline_complete
formal_recognizer_run_runnable
covert_channel_audit_implemented
ready_for_holdout_generation
formal_phase2b_exit_claim
active_promotion_enabled
```

This is implementation progress, not sealed C1 evidence.

Neither mechanics result is yet a sealed selector input: the current public
selector still consumes caller-supplied evaluation objects and only replays
their grid identities, while the exact uncertainty receipt is not consumed by
the projection compiler.  Binding exact receipt and evaluation provenance to
the complete grid is therefore an additional formal-pipeline blocker.  The
mechanics slice also does not yet prove that `task_target` and every unused
observation have complete projection coverage; that belongs to the trusted
wire/coverage contract.

## 7. Next authorized construction slice

The next non-actual commit is **Phase-2B Root/Identity Exact Rational
Residual-Tolerance-Selector Bridge v1**.  It should carry the exact RationalValue
intervals through an exact residual/tolerance/verifier result type and bind that
receipt to the complete adapter grid and selector input.  It must prove
conservative arithmetic for each supported law and normalized interval operation;
it must not convert the receipt back to the existing float `ClosedInterval`.
Unsupported nonlinear or transform cases remain full-grid abstentions.  After
that bridge is replayable, transform semantics can be added one operation at a
time, each with legal preservation and invalid-transform tests.

No Q0.5b actual rerun, Docker execution, holdout generation, or ACTIVE mutation
is authorized by this ledger.
