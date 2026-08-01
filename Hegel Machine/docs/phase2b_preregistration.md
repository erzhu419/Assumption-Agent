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
`Hegel_Machine_Phase2B_Phase3_Exact_Freeze_Decisions.md` under
`hegel-freeze-p2b-p3-v1.0.1`. This implementation-audit amendment supersedes
v1.0.0: `411876909552964556` remains the master/bootstrap seed, while sklearn's
executable `random_state` is the frozen domain-separated SHA-256-to-uint32 value
`2611585425`. This freezes a contract; it does not generate,
seal, run, audit, score, or consume a holdout. The following states therefore
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

## Real blockers after the exact freeze

`ready_for_holdout_generation` and formal exit remain false because
implementation and external-evidence blockers remain:

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
