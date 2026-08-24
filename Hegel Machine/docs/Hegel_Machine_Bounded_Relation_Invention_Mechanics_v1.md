# Hegel Machine bounded relation-invention mechanics v1

## Status

This slice closes one local implementation gap: an ordered
`OntologyInadequacyReport` can now gate a bounded program search that emits an
executable `RelationLawCandidate`, preregisters a prediction before a holdout
outcome is opened, runs hard-negative falsification, binds a reduction map, and
stops at a shadow candidate branch.

Its exact success status is:

```text
SHADOW_RELATION_CANDIDATE_VERIFIED_PUBLIC_SYNTHETIC_C2_LOCKED
```

This is **not C2 evidence**. The corpus is public and synthetic, the holdout
vault is in the same Python process, no external custodian or trust root is
present, C1 remains false, and the language compiler deliberately remains
closed. The module therefore cannot mutate `TheoryState` or authorize
promotion.

## Why this slice exists

Before this change, `RelationLawCandidate` was an immutable schema with no
producer in the repository. `diagnostics.py` could reach `ONTOLOGY_DEFECT`, but
there was no executable path from that verdict to a new relation candidate.
The existing vertical slice changes scope while keeping the six-law library
unchanged; it does not invent a relation.

The new path is deliberately narrower than the authoritative C2 protocol:

```text
frozen old-language comparison
  -> comparator and residual rule frozen without validation-label selection
  -> validation transfer of the residual mask
  -> synthetic profile replay through the ordered diagnostic ladder
  -> bounded executable-program quotient search
  -> one relation candidate commitment
  -> holdout prediction commitment
  -> process-local holdout opening
  -> hard-negative and exact prediction replay
  -> candidate branch only; active compiler remains closed
```

## Frozen public fixture

The fixture models a stateful threshold relation:

```text
next_on = drive >= 2 or (prior_on and drive >= 0)
```

The synthesizer is not given this expression. It receives labeled discovery
and validation rows and enumerates a frozen grammar containing Boolean
constants, `prior_on`, six `drive >= threshold` atoms, atom negation, and
commutative `and`/`or`, up to five AST nodes. The enumeration contains 1,980
canonical programs and 84 distinct behavior classes on the discovery plus
validation inputs.

The frozen old-language comparison is the same grammar limited to three AST
nodes. Its representative is predeclared as `drive>=0`; validation labels do
not select it. The implementation requires it to tie the best discovery
accuracy before it may be used, and separately rejects the case where any old
program fits discovery and validation exactly. Its conservative split accuracy
is `5/6`. The smallest exact behavior class requires five nodes. Its canonical
representative is:

```text
and(drive>=0,or(drive>=2,prior_on))
```

The accepted expression contains no observation ID, input ID, or outcome
lookup. An exact truth-vector quotient chooses a single canonical
representative among equivalent syntax.

## Synthetic diagnostic-profile ordering

Language synthesis requires an exact `OntologyInadequacyReport`; callers
cannot omit it. The report is recomputed from and bound to the same discovery
and validation partitions before synthesis begins.

The predeclared old comparator's discovery errors define a frozen residual
probe:

```text
0 <= drive < 2 and prior_on is false
```

That error mask is fixed from discovery and predicts the validation error mask
exactly. The diagnostic profile then records frozen public-fixture assertions:
zero gain for refit, scope, mixture, low-order composition, idealization,
robustification, and added-probe repairs, plus positive persistence metrics.
The existing decision function replays those values through every ordered
threshold and reaches `ONTOLOGY_DEFECT` before search opens.

This slice does **not** execute eight independent repair evaluators, and the
discovery residual is not an externally timestamped preregistration. The
result therefore claims only synthetic-profile decision mechanics, not
empirical exclusion of cheaper repairs.

A low-order target such as `drive>=2` is stopped before synthesis because the
old comparison language already fits it exactly. A report from another data
generation cannot be replayed against changed partitions.

## Prediction and falsification chronology

The successful chronology is fixed as:

```text
synthetic_profile_diagnostic_ladder_replayed
candidate_synthesis_committed
holdout_prediction_preregistered
process_local_holdout_opened
holdout_scored_without_candidate_change
```

The holdout inputs contain half-step drive values unseen by the synthesis API.
Their prediction vector is committed before the process-local vault releases labels. The
candidate scores `1.0`; the old comparison scores `0.75`. Three explicit hard
negatives cover prior-state and threshold decoys and are all rejected.
The vault rejects malformed vectors and requires the prediction to bind the
exact synthesis commitment and the exact ordered input generation before it
marks itself opened.

This ordering demonstrates API mechanics only. A process-local vault cannot
prove that an operator, debugger, or earlier process did not inspect its
private outcomes.

## Candidate and conservative boundary

The result constructs the repository's first produced
`RelationLawCandidate` whose `kind` is `None`: it is not silently mislabeled as
one of the six active `LawKind` values. Candidate-only violation functionals
therefore permit `law_kind=None`, while `TheoryState` still requires every
active law and active functional to share a concrete registered kind.
Candidate/function-kind mismatches fail closed.

The candidate carries:

- an executable program and a declared zero/one-error violation functional;
- positive examples and hard negatives;
- a preregistered holdout-vector prediction;
- a finite-universe identifiability certificate;
- a declared reduction-map intent to avoid candidate dispatch outside the new
  synthetic scope; this slice does not execute the formal reduction gate;
- five distinct process-local Generator/Formalizer/Falsifier/Evaluator/
  Promoter role IDs, not independent authorities.

The report contains only computed synthetic discovery/validation/holdout/
hard-negative accuracies, expressivity gain, and description-compression gain.
Formal old-success preservation, limiting-case reduction, and regression cost
are explicitly listed as unmeasured rather than assigned a success value.
The description gain uses the fixture's fixed two-bits-per-node and
four-bits-per-error accounting; it is not formal MDL and does not include a
new-symbol code cost.

The corresponding language `TheoryPatch` binds the ontology report, but
`compile_patch` rejects it with
`LANGUAGE_COMPILER_CLOSED_CANDIDATE_BRANCH_ONLY`. Canonical parent theory bytes
and all six active law IDs are unchanged.

## Replay

Run the deterministic report without network or state writes:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
python3 -B -m hegel_machine.bounded_relation_invention_v1
```

The canonical stdout must equal:

```text
artifacts/bounded_relation_invention_mechanics_v1.json
```

The report validator recomputes the frozen replay and rejects missing, extra,
or contradictory fields even when a caller recomputes the report content ID.

Focused verification:

```bash
TMPDIR=/tmp TMP=/tmp TEMP=/tmp PYTHONPATH=src \
PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -p no:cacheprovider -q \
tests/test_bounded_relation_invention_v1.py
```

## Permanent nonclaims

- The public synthetic fixture is not independently generated hidden evidence.
- The process-local holdout is not external custody or non-access proof.
- Bounded program search is not unrestricted open-world discovery.
- Finite-fixture non-equivalence is not universal non-expressibility.
- Frozen synthetic profile values are not executed cheaper-repair evidence.
- A discovery-frozen residual probe is not external preregistration.
- Distinct process-local role IDs are not independent authorities.
- A declared reduction map is not executed formal reduction evidence.
- Fixed-bit description gain is not formal MDL or a new-symbol cost.
- The candidate branch is not C1 PASS, C2 PASS, or ACTIVE theory.
- Shadow byte preservation is not multigeneration retention evidence.
- No network, container, external actor, or real scientific outcome is used.
- The authoritative C2 route remains locked until C1 passes and the frozen
  Q1/Q2, custody, sealed-test, MDL, negative-control, and reduction requirements
  are satisfied.
