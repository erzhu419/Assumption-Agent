# Selective Daily-Life Operator Counterfactual 20260621

## Policy

- Enable OperatorSpec only on `daily_life`.
- Reuse ctx-only answers on all other domains.
- Daily-life rows reuse the already generated full-operator answers and existing live judgments.

## Counts

- selective vs ctx-only: `{'tie': 10, 'selective': 2}`
- ctx-only vs selective: `{'tie': 10, 'selective': 2}`
- by domain forward: `{'business': {'tie': 3}, 'daily_life': {'selective': 2, 'tie': 1}, 'engineering': {'tie': 3}, 'software_engineering': {'tie': 3}}`

## Interpretation

On this n=12 sample, the selective policy improves only the domain where OperatorSpec won before and avoids the business/software losses by construction.
