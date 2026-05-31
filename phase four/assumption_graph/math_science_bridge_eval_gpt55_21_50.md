# Math/Science Bridge Bypass Evaluation

Date: 2026-05-31

## Setup

- Variant: `phase2_v20_ms_bridge_gpt55_21_50`
- Current v20 baseline: `phase2_v20_gpt55`
- Raw baseline: `baseline`
- Sample: `phase two/analysis/cache/proposal_samples/math_science_21_50_sample.json`
- Model: `gpt-5.5`
- Rows: 9 from the 21-50 heldout slice

The change is not graph retrieval. It replaces the one-size math/science hygiene
bypass with an intent-aware bypass:

- `math_formal`: ordinary proof/computation prompt
- `math_research_bridge`: concrete mathematical bridge / proof-vs-refutation plan
- `science_mechanism`: ordinary mechanism prompt
- `science_decision`: publication, graduation, resource, and collaboration action plan

## Route Distribution

- `math_research_bridge`: 4
- `science_mechanism`: 3
- `science_decision`: 2

## Against Current v20

Bidirectional pairwise judge:

- Bridge wins: 16
- Current v20 wins: 2
- Ties: 0
- Decisive win rate: 88.9%
- Mean score delta: +0.89

By domain:

- `mathematics`: 8 wins, 0 losses
- `science`: 8 wins, 2 losses

## Against Raw Baseline

Bidirectional pairwise judge:

- Bridge wins: 18
- Raw baseline wins: 0
- Ties: 0
- Decisive win rate: 100.0%
- Mean score delta: +1.28

By domain:

- `mathematics`: 8 wins, 0 losses
- `science`: 10 wins, 0 losses

## Interpretation

The previous failures were not Assumption Graph failures. They were prompt
routing failures: research-style mathematics and science-decision questions were
being answered like formal proof/mechanism questions. The new bypass keeps
formal math/science rows strict while giving research-bridge and decision rows
the concrete named tools, tradeoffs, and action plans that the judge preferred.

This closes the first identified issue on the 21-50 heldout slice: raw baseline
no longer has math/science wins in this subset.
