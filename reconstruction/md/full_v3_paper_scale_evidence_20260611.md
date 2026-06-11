# Full V3 Paper-Scale Evidence Aggregation - 2026-06-11

## Purpose

This run consolidates the current full-v3 evidence into one paper-facing artifact. It does not make new API calls. It aggregates existing first-party live/cached traces, problem-level paper statistics, retrieval baselines, toggle baselines, v2/v3 phase validations, and the vertical recursive slice.

Output artifact:

`phase four/assumption_graph/paper_readiness_20260604/full_v3_paper_scale_evidence_20260611.json`

## Performance Validation Result

Command:

```bash
python3 -m assumption_os.full_v3_paper_scale_evidence \
  --root . \
  --eval-id full_v3_paper_scale_evidence_20260611 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/full_v3_paper_scale_evidence_20260611.json'
```

Result: pass.

Key metrics:

- Required artifact pass rate: 1.0 over 22 artifacts
- V3 mechanism pass rate: 1.0 over 8 mechanism artifacts
- Raw first-party live events: 6403
- Valid judge events: 2818
- Main problem-level n: 100
- Structural vs base utility: 0.625
- Structural vs base bootstrap CI lower: 0.53
- Structural vs base sign-test p-value: 0.0124006
- Structural vs placebo utility: 0.705
- Structural vs placebo bootstrap CI lower: 0.61
- Structural vs placebo sign-test p-value: 0.00003114
- Retrieval margin over best baseline: 0.70
- Key toggle minimum margin: 0.08
- Long-run downstream win rate: 0.75
- Long-run capability improvement: 0.1945
- Fresh guarded heldout300 active interventions: 11/300
- Fresh guarded full-remaining active interventions: 21/556
- Fresh cue-repair selective active interventions: 31/556
- Fresh cue-repair selective vs base utility / CI lower: 0.5144 / 0.5054
- Fresh cue-repair selective vs placebo utility / CI lower: 0.5153 / 0.5063
- Phase8 creative candidates: 8
- Phase8 nonlocal candidate ratio: 0.35
- Phase8 world-model quality AUROC / Brier: 1.0 / 0.1156
- Phase8 selected quality profile: quality_v4
- Phase8 selected coverage profile: coverage_v6
- Phase8 coverage profile active gain: +4, utility 0.5108 / 0.5135
- Prompt/answer payload stored: false
- Secret leak detected: false
- Boundary case count: 1

## Interpretation

The evidence chain is now less fragmented: the paper-facing claim can point to one artifact that combines the frozen main experiment, hard baselines, retrieval baselines, first-party trace scale, phase validations, and long-run/vertical recursive results.

The artifact now includes the fresh live guarded heldout300, strict full-remaining, and cue-repair selective expansion runs.  The fresh reruns are still small-effect validations, but the latest retained profile improves both active coverage and problem-level utility over the previous selective guard.

Phase8 adds a separate post-v3 bottleneck artifact for generator creativity, world-model profile selection, and coverage exploration.  It does not promote the broader coverage profile as default because quality_v4 still has higher base/placebo utility; instead it records coverage_v6 as a positive but lower-utility exploration profile.
