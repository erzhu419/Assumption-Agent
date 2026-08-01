# Phase-2B formal-track preregistration status

## Canonical name and present status

The canonical milestone name is:

> **Phase-2B Sealed Typed-Evidence Structural Identification Qualification**

The reserved capability claim is **Sealed Typed-Evidence Structural Law
Identification and Verification**. Neither name is a description of the current
result. The repository currently contains a family-neutral-shaped wire contract,
a statistical protocol, immutable seal → prediction commitment → reveal →
consume lifecycle records with a process-local fork guard, and an OCI launch
specification. It does not contain a durable one-shot custodian, functional
recognizer image/CLI, or sealed holdout result.

The only valid status for the checked-in artifact is:

```text
preregistration_candidate_with_open_freeze_questions
```

The corresponding artifact is `artifacts/phase2b_preregistration_v1.json` and
must keep all of the following false until external evidence exists:

```text
formal_phase2b_exit_claim
sealed_holdout_generated
sealed_holdout_consumed
independent_custodian_attested
external_isolation_attested
recognizer_image_built_from_allowlist
recognizer_entrypoint_implemented
formal_recognizer_run_runnable
```

## Process and trust boundary

```text
public protocol + frozen code/image/config/baselines
  → independent custodian generates new latent cases
  → public family-neutral-shaped typed evidence root + side-channel audit
     + private salted answer commitment
  → OCI-isolated untrusted recognizer runs once
  → predictions and audit archive hashes are committed
  → custodian reveals the answer manifest
  → independent evaluator scores once
  → immutable report marks the holdout consumed
```

The recognizer input image must exclude `phase2_exit`, public development
generators, custodian code, evaluator code and answer manifests. Phase-2A
fixtures remain useful for unit tests and adapter regression, but may not enter
validation, calibration, confidence intervals, baselines or the sealed holdout.

A Python dataclass or Boolean flag cannot prove OS isolation or custodian
independence. Formal evidence therefore requires a detached external runtime
attestation bound to the OCI run specification and prediction archive hash.
The current attestation object is only structurally checked; it has no signature
verifier. The exact module inventory contract has no built/signed minimal-image
SBOM. The frozen entrypoint is a reserved path, not an implemented executable,
and no strict evaluator yet parses exactly 720 unique `PredictionBundle`s and
recomputes the archive root.

## Public input boundary

The public wire schema uses UUIDv4-shaped identifiers and a field allowlist. It may
carry typed observations, quantity/unit information, uncertainty, temporal and
spatial support, role/entity candidates, a task target, an aggregation graph,
an admissible transform catalog and a missingness mask.

It rejects unknown fields and, in particular:

```text
law_family
correct_binding
correct_scale
expected_pass / expected_fail
candidate_rank
oracle_margin
candidate-private witness or payload
family-specific observable names
```

This is a syntactic boundary, not proof of semantic neutrality. Allowed UUIDs,
provenance hashes, role-candidate sets, missingness patterns and unused
transforms can still encode an answer. Formal generation therefore also needs
independent randomized/global-shuffled ID assignment, consistent-renaming
invariance, allowed-field answer-correlation tests and a side-channel audit.
Those checks are not implemented. Likewise, `standard_error` and
`absolute_bound` radii remain typed inputs; no frozen compiler currently turns
both models into formal closed residual intervals.

The public input is evidence, not a caller-provided family × binding × scale
projection grid. Formal Phase-2B must generate role and scale hypotheses inside
the recognizer. The public selector therefore re-enumerates the adapter from the
bound evidence bundle and frozen registry; it does not accept a caller-signed
grid commitment. The old two-tag capability remains only **Scale-Indexed
Candidate Projection Selection**.

## Frozen sample and gate skeleton

The independent latent-case target is:

```text
6 families × 2 scale cells × 60 cases = 720 cases
```

Per family × scale cell:

| Case type | Count |
|---|---:|
| answerable positive | 20 |
| wrong-family hard negative | 8 |
| binding counterfactual | 8 |
| scale counterfactual | 8 |
| sign/invariant-break | 8 |
| insufficient/genuinely ambiguous | 8 |

Preservation pairs use a separate denominator. Binary gates use a 95% one-sided
Wilson lower confidence bound. Each family and each scale must pass its own
marginal slice gates. The implementation uses the stricter values where the
answer document provided two scale tables; this tie-break remains visible for
confirmation before holdout generation.

## Why generation is still blocked

`ready_for_holdout_generation` is intentionally false. Open freeze questions
include:

- the 15% ambiguous margin stratum requests 108 cases, while the case table
  allocates only 96 ambiguous cases;
- the law-family applicability matrix and total count for preservation pairs;
- exact model/image/prompt pins for three mandatory baselines;
- bootstrap seed, iteration count and resampling unit;
- the allowed retry policy for infrastructure failure before answer reveal;
- shared-footprint taxonomy and the formal family-discrimination statistic;
- whether the semantic-conflict challenge is a slice of the 720 or additional.

These are protocol-definition gaps, not failed benchmark results. No secret
holdout should be generated until they are resolved and a new protocol ID is
produced.

## External prerequisites that cannot be self-certified

- an independent custodian and secret master seed;
- a new 720-case holdout generated only after all hashes are frozen;
- a real no-network/read-only/no-repository recognizer runtime;
- a prediction commitment that predates answer reveal;
- detached external attestation;
- organizational enforcement that a consumed holdout is never debugged or
  rerun;
- frozen external embedding, LLM semantic-only and flat typed baselines.
- a durable append-only CAS ledger that atomically consumes each parent across
  processes/restarts and verifies custodian signatures.

Without all of them the current result remains protocol/readiness
infrastructure. A later end-to-end development run may be called an unsealed
pipeline validation, never a formal Phase-2 exit.
