# Math/Science Bridge Stability: sample100 slice

Date: 2026-05-31

## Setup

- Variant: `phase2_v20_ms_bridge_gpt55_ms100`
- Sample: `phase two/analysis/cache/proposal_samples/math_science_sample100.json`
- Size: 30 math/science rows from `sample_100`
- Solver/judge model: `gpt-5.5` through the configured proxy environment

The variant uses the intent-aware math/science bypass:

- `math_formal`
- `math_research_bridge`
- `science_mechanism`
- `science_decision`

## Route Counts

- `science_mechanism`: 12
- `science_decision`: 3
- `math_research_bridge`: 8
- `math_formal`: 7

## Pairwise Results

Against current `phase2_v20`, bidirectional:

- Bridge wins: 53
- `phase2_v20` wins: 5
- Ties: 2
- Decisive win rate: 91.4%

Against raw `baseline`, bidirectional:

- Bridge wins: 55
- Baseline wins: 2
- Ties: 3
- Decisive win rate: 96.5%

Single-direction checks:

- `phase2_v20_ms_bridge_gpt55_ms100` vs `phase2_v20`: 27 / 2 / 1
- `phase2_v20` vs `phase2_v20_ms_bridge_gpt55_ms100`: 3 / 26 / 1
- `phase2_v20_ms_bridge_gpt55_ms100` vs `baseline`: 28 / 1 / 1
- `baseline` vs `phase2_v20_ms_bridge_gpt55_ms100`: 1 / 27 / 2

## Interpretation

The larger 30-row math/science slice confirms that the intent-aware bypass is
not just a 21-50 artifact. It beats both the current v20 math/science bypass and
the raw baseline by a wide margin. The remaining losses are concentrated in a
small set of math rows, so future work should target formal proof/computation
precision rather than replacing the bridge policy.
