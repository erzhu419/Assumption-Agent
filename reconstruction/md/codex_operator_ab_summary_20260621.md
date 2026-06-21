# Codex OperatorSpec A/B Summary 20260621

## Setup
- sample: `phase two/analysis/cache/proposal_samples/codex_operator_ab_non_bypass_n12_20260621.json`
- operator variant: `codex_op_ab_operator_20260621`
- ctx-only variant: `codex_op_ab_ctxonly_20260621`
- both variants used the same Turn 0 frame/rewrite meta; only OperatorSpec injection differed.

## Results
- forward: `{'ctxonly': 5, 'operator': 5, 'tie': 2}`
- reverse: `{'ctxonly': 5, 'operator': 5, 'tie': 2}`
- bidirectional combined: `{'ctxonly': 5, 'operator': 4, 'mixed_tie': 2, 'tie': 1}`
- by domain: `{'business': {'ctxonly': 2, 'operator': 1}, 'daily_life': {'operator': 2, 'tie': 1}, 'engineering': {'ctxonly': 1, 'mixed_tie': 2}, 'software_engineering': {'operator': 1, 'ctxonly': 2}}`

## Activation
- problems_with_operator: `12`
- total_operator_specs: `24`
- total_required_slots: `106`
- avg_operator_chars_when_on: `2076.75`
- avg_context_chars: `6649.67`
- operator_source_counts: `{'strategy_S08': 3, 'strategy_S15': 2, 'strategy_S21': 6, 'strategy_S27': 3, 'strategy_S14': 2, 'strategy_S25': 1, 'strategy_S22': 3, 'strategy_S26': 1, 'cand_50e44c655f61': 3}`

## Per Problem
| problem | domain | diff | fwd | rev | combined | operator ids |
| --- | --- | --- | --- | --- | --- | --- |
| `business_0097` | `business` | `medium` | `ctxonly` | `ctxonly` | `ctxonly` | `strategy_S15, strategy_S08` |
| `daily_life_0183` | `daily_life` | `hard` | `operator` | `operator` | `operator` | `strategy_S21, strategy_S27` |
| `engineering_0244` | `engineering` | `medium` | `ctxonly` | `ctxonly` | `ctxonly` | `strategy_S08, strategy_S14` |
| `engineering_0183` | `engineering` | `hard` | `operator` | `tie` | `mixed_tie` | `strategy_S27, strategy_S25` |
| `business_0146` | `business` | `medium` | `operator` | `operator` | `operator` | `strategy_S27, strategy_S21` |
| `engineering_0152` | `engineering` | `hard` | `tie` | `operator` | `mixed_tie` | `strategy_S21, strategy_S23` |
| `business_0192` | `business` | `hard` | `ctxonly` | `ctxonly` | `ctxonly` | `strategy_S22, strategy_S26` |
| `daily_life_0197` | `daily_life` | `medium` | `operator` | `operator` | `operator` | `strategy_S21, strategy_S22` |
| `software_engineering_0142` | `software_engineering` | `medium` | `operator` | `operator` | `operator` | `cand_50e44c655f61, strategy_S21` |
| `software_engineering_0265` | `software_engineering` | `easy` | `ctxonly` | `ctxonly` | `ctxonly` | `cand_50e44c655f61, strategy_S08` |
| `daily_life_0216` | `daily_life` | `medium` | `tie` | `tie` | `tie` | `strategy_S08, strategy_S22` |
| `software_engineering_0337` | `software_engineering` | `hard` | `ctxonly` | `ctxonly` | `ctxonly` | `cand_50e44c655f61, strategy_S15` |
