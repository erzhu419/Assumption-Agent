# Selective Daily-Life Operator Live A/B 20260621

## Setup

- Variant: `codex_op_ab_selective_daily_live_20260621` vs `codex_op_ab_ctxonly_20260621`.
- Turn0/meta was seeded from ctx-only, then Turn1/Turn2 were regenerated live.
- OperatorSpec default gate: only `daily_life`; max specs = 2.
- Existing graph skip still excludes `software_engineering` from graph retrieval.

## Gate Check

- `daily_life`: enabled, 2 operators each, n=3.
- `business`: not selected, 0 operators, n=3.
- `engineering`: not selected, 0 operators, n=3.
- `software_engineering`: graph skipped, 0 operators, n=3.

## Judge Results

- Forward selective vs ctx-only: `{'ctxonly': 5, 'selective_live': 5, 'tie': 2}`, mean selective-ctx delta `+0.08`.
- Reverse ctx-only vs selective: `{'ctxonly': 5, 'selective_live': 5, 'tie': 2}`, mean selective-ctx delta `+0.08`.
- Combined directional outcomes: `{'ctxonly': 4, 'mixed_tie': 2, 'selective_live': 4, 'tie': 1, 'direction_conflict': 1}`.

## By Domain

- Forward: `{'business': {'ctxonly': 2, 'tie': 1}, 'daily_life': {'selective_live': 2, 'ctxonly': 1}, 'engineering': {'ctxonly': 1, 'tie': 1, 'selective_live': 1}, 'software_engineering': {'selective_live': 2, 'ctxonly': 1}}`.
- Reverse: `{'business': {'ctxonly': 3}, 'daily_life': {'selective_live': 2, 'ctxonly': 1}, 'engineering': {'ctxonly': 1, 'tie': 2}, 'software_engineering': {'selective_live': 3}}`.
- Combined: `{'business': {'ctxonly': 2, 'mixed_tie': 1}, 'daily_life': {'selective_live': 2, 'ctxonly': 1}, 'engineering': {'mixed_tie': 1, 'tie': 1, 'ctxonly': 1}, 'software_engineering': {'selective_live': 2, 'direction_conflict': 1}}`.

## Interpretation

Live selective is overall neutral on this n=12 sample: both judge directions are 5/5/2. The gate itself works as intended. The stable signal remains that daily-life benefits more often than not, while business still loses under fresh generation even without operators, so remaining non-daily variance is mostly answer-generation noise rather than operator injection.
