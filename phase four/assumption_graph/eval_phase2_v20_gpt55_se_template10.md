# Phase2 v20 GPT-5.5 SE Template-Only Heldout Subset (n=10)

Date: 2026-05-31

## Setup

- Baseline: `phase2_v20_gpt55`
- Variant: `phase2_v20_ag_se_template_gpt55`
- Sample: `phase two/analysis/cache/sample_21_50_software_engineering.json`
- Model: `gpt-5.5` through environment variables only
- Intervention: `--assumption-graph "phase four/assumption_graph"` with default `--assumption-graph-skip-domains software_engineering`

Software-engineering graph context stayed gated off. The active intervention was
only the prompt-side execution template in `assumption_os/domain_templates.py`.

## Result

Forward judge (`template` vs `baseline`):

- Template: 5
- Baseline: 2
- Tie: 3
- Decisive win rate: 71.4%
- Mean delta: +0.20

Reverse judge (`baseline` vs `template`), normalized:

- Template: 5
- Baseline: 4
- Tie: 1
- Decisive win rate: 55.6%
- Mean delta from baseline-as-A report: -0.30

Combined bidirectional result:

- Template: 10
- Baseline: 6
- Tie: 4
- Decisive win rate: 62.5%

## Comparison To Previous SE Negative Transfer

On the same 10 heldout software-engineering problems, the earlier learned-graph
variant scored:

- Learned graph: 3
- Baseline: 12
- Tie: 5
- Decisive win rate: 20.0%

The template-only intervention therefore reverses the main SE failure mode while
keeping graph assumption injection disabled for this domain.

## Per-Problem Combined Outcomes

- `software_engineering_0355` release quality: template 0, baseline 1, tie 1.
- `software_engineering_0322` safety/latency: template 2, baseline 0, tie 0.
- `software_engineering_0225` startup GTM: template 1, baseline 0, tie 1.
- `software_engineering_0186` adapter discovery: template 2, baseline 0, tie 0.
- `software_engineering_0086` online/offline gap: template 0, baseline 2, tie 0.
- `software_engineering_0050` legacy increment: template 1, baseline 0, tie 1.
- `software_engineering_0364` platform migration: template 1, baseline 1, tie 0.
- `software_engineering_0365` SCADA migration: template 0, baseline 1, tie 1.
- `software_engineering_0062` online/offline gap: template 2, baseline 0, tie 0.
- `software_engineering_0288` tech choice: template 1, baseline 1, tie 0.

## Decision

Keep the default policy:

- `software_engineering` graph context remains skipped by default.
- Software-engineering reranker remains opt-in for explicit graph-injection runs.
- Domain execution templates stay enabled when `--assumption-graph` is supplied,
  unless `--disable-domain-execution-template` is passed.

No graph writeback was performed, because this run did not retrieve or apply
software-engineering assumption IDs. The result should be treated as prompt
harness evidence, not graph-node evidence.
