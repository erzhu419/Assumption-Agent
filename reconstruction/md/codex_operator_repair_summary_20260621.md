# Operator Fidelity Repair Summary 20260621

## Mechanism

- Added programmatic OperatorSpec slot verifier.
- Added one bounded answer-time repair pass when required slots are missing.
- Default operator gate remains `daily_life`; max specs = 2.

## Fidelity

- Programmatic daily-life operator pass rate: `1.0`.
- Programmatic daily-life fidelity mean: `0.9`.
- Decorative use rate: `0.0`.
- Repair attempts: `1` on `['daily_life_0216']`.
- Repaired fidelity before/after: `{'daily_life_0216': {'before': 0.675, 'after': 1.0, 'decorative_after': 0}}`.

## Utility Judges

- Repair vs selective-live forward: `{'repair': 7, 'selective_live': 2, 'tie': 3}`, mean repair delta `0.5`.
- Selective-live vs repair reverse: `{'repair': 6, 'tie': 4, 'selective_live': 2}`, mean repair delta `0.3333`.
- Combined repair vs selective-live: `{'repair': 6, 'selective_live': 1, 'tie': 3, 'mixed_tie': 1, 'direction_conflict': 1}`.

- Repair vs ctx-only forward: `{'ctxonly': 3, 'repair': 7, 'tie': 2}`, mean repair delta `0.3333`.
- Ctx-only vs repair reverse: `{'ctxonly': 3, 'repair': 6, 'tie': 3}`, mean repair delta `0.1667`.
- Combined repair vs ctx-only: `{'ctxonly': 2, 'mixed_tie': 1, 'repair': 6, 'direction_conflict': 1, 'tie': 2}`.

## Decision

This clears the local push gate for the repair mechanism on the n=12 triggered sample: fidelity improved, decorative use was removed on the repaired case, and pairwise utility improved against both selective-live and ctx-only. The business domain remains a non-operator generation-loss area and should stay outside operator activation.
