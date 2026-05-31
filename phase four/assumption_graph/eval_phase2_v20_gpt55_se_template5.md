# Phase2 v20 GPT-5.5 SE Template-Only Regression (n=5)

Date: 2026-05-31

## Setup

- Baseline: `phase2_v20_gpt55`
- Variant: `phase2_v20_ag_se_template_gpt55`
- Sample: `phase two/analysis/cache/sample_se_regression_5.json`
- Model: `gpt-5.5` through environment variables only
- Intervention: `--assumption-graph "phase four/assumption_graph"` with the default `--assumption-graph-skip-domains software_engineering`

This means software-engineering graph context was still gated off. The active
intervention was the prompt-side domain execution template in
`assumption_os/domain_templates.py`, not retrieved graph assumptions.

## Result

Forward judge (`template` vs `baseline`):

- Template: 2
- Baseline: 1
- Tie: 2
- Decisive win rate: 66.7%
- Mean delta: +0.20

Reverse judge (`baseline` vs `template`), normalized:

- Template: 2
- Baseline: 2
- Tie: 1
- Decisive win rate: 50.0%
- Mean delta from baseline-as-A report: -0.20

Combined bidirectional result:

- Template: 4
- Baseline: 3
- Tie: 3
- Decisive win rate: 57.1%

## Per-Problem Read

- `software_engineering_0355` release quality: tie / baseline. Baseline remained stronger on defect handling and regression specificity.
- `software_engineering_0225` startup GTM: template / tie. Template improved wedge customer, responsibility boundary, and Go/No-Go closure.
- `software_engineering_0186` adapter discovery: template / template. This is the clearest gain; the template added authorization boundary, adapter contract, stopping criteria, and demo fallback.
- `software_engineering_0364` platform migration: baseline / template. Mixed result; both answers captured TCO and path dependency.
- `software_engineering_0365` SCADA migration: tie / baseline. Baseline was slightly preferred for governance and quantitative acceptance.

## Interpretation

The domain execution template turned the targeted SE negative-transfer set from
stable baseline wins into a slightly positive/mostly neutral result. The effect
is useful but not strong enough to re-enable software-engineering graph context
by default.

Keep the conservative default:

- graph context skips `software_engineering`
- software-engineering reranker remains opt-in through `--assumption-graph-skip-domains ""`
- domain execution templates remain enabled for graph experiment variants unless explicitly disabled

No graph writeback was performed for this run, because this was a template-only
intervention and there were no active retrieved assumption IDs to credit or
penalize.
