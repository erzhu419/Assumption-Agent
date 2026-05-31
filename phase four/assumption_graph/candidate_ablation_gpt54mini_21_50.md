# Candidate Proposal Ablation: gpt-5.4-mini 21-50

Date: 2026-05-31

## Setup

- Baseline: `phase2_v20_gpt54mini_prop_union`
- Combo proposal variant: `proposal_ready_combo_gpt54mini`
- Sample: `phase two/analysis/cache/proposal_samples/proposal_ready_union_sample.json`
- Proposals forced only on routed `should_fire` rows.
- No candidate was applied to the persistent graph.

## Result

Bidirectional pairwise judge over 19 rows x 2 directions:

- Combo wins: 26
- Baseline wins: 12
- Ties: 0
- Decisive win rate: 68.4%
- Mean score delta for combo: +0.47

By domain:

- `business`: {'combo_win': 2, 'baseline_win': 2}
- `daily_life`: {'baseline_win': 5, 'combo_win': 3}
- `engineering`: {'combo_win': 5, 'baseline_win': 3}
- `software_engineering`: {'combo_win': 16, 'baseline_win': 2}

By difficulty:

- `easy`: {'baseline_win': 5, 'combo_win': 1}
- `hard`: {'combo_win': 7, 'baseline_win': 3}
- `medium`: {'combo_win': 18, 'baseline_win': 4}

## Interpretation

The combo route-conditioned proposal policy improved the mini-model screening result, especially on software-engineering rows. Individual candidate acceptance remains conservative: no single proposal passed both trigger LCB and control-harm UCB thresholds, so `--apply-accepted` was not run.
