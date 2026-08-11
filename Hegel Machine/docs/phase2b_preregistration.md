# Phase-2B formal-track preregistration status

## Canonical name and present status

The canonical future milestone name remains:

> **Phase-2B Sealed Typed-Evidence Structural Identification Qualification**

The reserved capability claim is **Sealed Typed-Evidence Structural Law
Identification and Verification**. Neither is a description of the current
result. The checked-in result remains **Phase-2A Controlled Typed-Selector
Mechanics Qualification**; Phase-2B is shadow-only preregistration and
implementation infrastructure.

The exact protocol choices are no longer open questions. They are frozen by
`Hegel_Machine_Phase2B_Phase3_Exact_Freeze_Decisions.md` plus the
[v1.0.2 strict canonical/certificate amendment](Hegel_Machine_Strict_Canonical_AST_CBOR_Certificate_Bridge_Freeze_v1.0.2.md)
under `hegel-freeze-p2b-p3-v1.0.2`. v1.0.2 inherits the v1.0.1 seed correction:
`411876909552964556` remains the master/bootstrap seed, while sklearn's
executable `random_state` remains the domain-separated SHA-256-to-uint32 value
`2611585425`. It additionally completes the Phase-3 strict acceptance and
certificate specification. The subsequent M1 artifact verifies both strict
implementations against the shared vectors, and M2 establishes a bounded
`DSL_TOO_LARGE` status for the old DSL; neither result generates, seals, runs,
audits, scores, or consumes a Phase-2B holdout. The following states therefore
remain false:

```text
formal_phase2b_exit_claim
sealed_holdout_generated
sealed_holdout_consumed
independent_custodian_attested
external_isolation_attested
formal_covert_channel_audit_passed
recognizer_image_built_from_allowlist
recognizer_entrypoint_implemented
formal_recognizer_run_runnable
```

The Phase-3 distinction must not be reused as Phase-2B evidence:

```text
strict specification frozen                         = true
strict Python/Rust shared vectors                    = 48/48 PASS each
accepted strict canonical count                     = 64,680 unique each
hegel-old-dsl-v1.0.0 under 50,000 syntactic budget  = DSL_TOO_LARGE
complete closure / extensional target verdict       = false / unavailable
formal roots                                        = null
hidden-sink formal verdict                          = false
outside or MDL certificate issued                   = false
Phase-2B sealed qualification                       = false
ACTIVE promotion                                    = false
```

The Phase-3 evidence is the
[dual strict gate](../artifacts/phase3_dual_strict_gate_v1.json) and
[dual strict capacity replay](../artifacts/phase3_dual_strict_capacity_replay_v1.json).
Both implementations produced diagnostic set commitment
`sha256:c1a02a66a8d6d8f75204cb3daf03ab0b01c2b3b8e486d0ab3d481ee3be43c930`;
the ordinal-50,001 AST hash is
`sha256:7c7f786c2cc57d31506b3c61d162d175c7f69a2878a089c72c9d053694cba948`.
Neither value is a formal archive root or an extensional target verdict.
Its required next action is to publish a new old-DSL version, apply frozen
shrink step 1 by removing `mean_v1`, `min_v1`, and `max_v1`, regenerate target
and validation commitments, and restart the new version from `NOT_RUN`; see the
[readiness resolution](Hegel_Machine_Phase3_Freeze_Readiness_Resolution.md).
Phase-2B formal exit remains an independent NO-GO until the sealed-data,
custodian, runtime, covert-audit, baseline and statistical gates below pass.

## Exact sample freeze

The main sealed holdout contains 720 independent latent cases:

```text
6 families × 2 scales × 60 cases = 720
```

Each `family × scale` cell has the exact case quota:

| Case type | Per cell | Total |
|---|---:|---:|
| `unique_scale_answerable` | 19 | 228 |
| `admissible_scale_set_answerable` | 1 | 12 |
| `wrong_family_hard_negative` | 8 | 96 |
| `binding_counterfactual` | 8 | 96 |
| `scale_counterfactual` | 8 | 96 |
| `sign_or_invariant_break` | 8 | 96 |
| `insufficient_or_nonidentifiable` | 8 | 96 |
| **Total** | **60** | **720** |

The exact per-cell margin strata are `21 / 18 / 12 / 9` for
`clear_interior / moderate / near_boundary_identifiable /
nonunique_or_insufficient`. The final stratum contains the eight insufficient
cases and the one set-valued answerable case. Set-valued cases are answerable:
they enter family, binding, scale-set and joint-exact denominators, but not
unique-scale, abstention-specificity, or nonidentifiability-abstention
denominators. `joint_exact` requires exact family, binding and scale set plus
`ANSWER_SET`; a singleton, strict subset, strict superset, or abstention is
wrong.

An additional 240-case sealed semantic-conflict challenge is frozen alongside
the main 720:

```text
12 cells × (10 low-semantic-overlap structural positives
            + 10 high-semantic-overlap structural negatives) = 240
```

It shares the freeze/reveal event but is not part of the main accuracy
denominator and may not tune thresholds. Therefore the formal data commitment
is **720 main + 240 challenge**, not “720 including the challenge.”

The derived preservation/sensitivity suite is also separate from the 720
latent cases:

```text
496 legal preservation pairs + 76 invalid-transform controls = 572 pairs
```

The six canonical family IDs, transformation applicability matrix, pair counts,
footprint classes (`P2_PAIR`, `P3_CHAIN`, `P4_STAR`, `PSET_AGGREGATE`), grouped
permutation-importance gates, baseline configurations, 10,000-replicate paired
cluster bootstrap, rerun allow/deny list, and validation-version failure policy
are all frozen. Actual external model revisions and artifact hashes still have
to be pinned before a run.

## Process and trust boundary

```text
public protocol + frozen code/image/config/baselines
  → independent custodian generates new latent cases
  → trusted wire builder assigns shuffled IDs and fixed envelopes
  → covert-channel audit over channel-only fields
  → public typed-evidence root + private salted answer commitment
  → OCI-isolated untrusted recognizer runs once
  → prediction/audit archive hashes are committed
  → custodian reveals the answer manifest
  → independent evaluator scores once
  → immutable report marks the holdout consumed
```

Phase-2A fixtures remain useful for tests and adapter regression, but may not
enter validation, calibration, confidence intervals, baselines, or the sealed
holdout. The public selector re-enumerates family, role-binding, and scale
hypotheses from the bound evidence and registry; it does not trust a
caller-provided candidate grid.

A Python dataclass, OCI argv, or Boolean flag cannot prove OS isolation,
custodian independence, or one-shot consumption. Formal evidence needs a
detached runtime attestation bound to the image, run specification, public
input root, and prediction archive. The current process-local lifecycle guard
is not a durable signed append-only custodian ledger.

## Public wire and uncertainty boundary

The allowlist rejects explicit answers such as `law_family`, `correct_binding`,
`correct_scale`, PASS/FAIL labels, ranks, oracle margins, and candidate-private
payloads. That is only a syntactic boundary. UUID bytes, provenance values,
record order, missingness, lengths, padding, unused identifiers, and eligibility
masks can still carry an answer.

The frozen covert-channel audit requires independent `K_shuffle`, `K_id`, and
`K_padding`; post-shuffle HMAC-derived UUIDs; a 65,536-byte envelope; channel
feature tests; 10,000 stratified label permutations; Holm–Bonferroni
`FWER = 0.01`; `NMI <= 0.02`; balanced-accuracy advantage `<= 0.05`; 32
consistent global renamings; 32 case-order permutations; and 16 within-case
observation-order permutations. This specification is frozen, but the trusted
wire builder and audit have **not been executed**. No audit-pass claim is
available.

Until all frozen `standard_error` semantics and Student-t/Bonferroni conversion
rules are implemented and tested, formal selector input is strictly:

```text
allowed uncertainty kind = absolute_bound
standard_error = STANDARD_ERROR_UNSUPPORTED
```

This is an execution restriction, not merely a documentation preference.

The implemented exact compiler enforces it at the whole-bundle boundary: one
`standard_error` observation or one endpoint outside the literal-pinned 663-point
RationalValue grid yields an abstaining receipt with no partial compiled rows.
For `absolute_bound`, all arithmetic starts from exact `Fraction.from_float`
representations of the wire's normalized binary64 values and rounds only outward.
The exact receipt is not converted to the existing float selector interval.

## Real blockers after the exact freeze

`ready_for_holdout_generation` and formal exit remain false because
implementation and external-evidence blockers remain:

- a bounded-binary64, dimensionless, support-aligned root/identity projection
  mechanics slice and a separate bundle-atomic exact RationalValue-grid
  `absolute_bound` compiler are implemented, but the exact receipt is not yet
  connected to conservative residual/tolerance/verifier intervals.  Non-degenerate
  residual intervals therefore still fail closed; complete transform semantics,
  receipt provenance binding and the formal typed evidence-to-prediction pipeline
  remain incomplete;
- trusted wire builder and the full covert-channel auditor are not implemented
  or executed;
- exact 40-hex external baseline revisions, image/SBOM digests, and artifact
  hashes are not registered;
- an independent latent generator has not produced validated 720 + 240 sealed
  artifacts or the 572 derived pairs;
- a functional recognizer CLI, signed minimal image, strict archive evaluator,
  and detached isolation attestation do not exist as completed evidence;
- no durable signed append-only custodian/CAS ledger enforces one-shot use
  across processes and restarts;
- no validation version has passed, so sealed holdout generation is forbidden.

These are implementation and external-execution blockers, not unresolved quota
or statistical-design questions. Until they are satisfied, Phase-2B remains
shadow-only protocol/readiness infrastructure and cannot block or authorize
ACTIVE theory mutation.
