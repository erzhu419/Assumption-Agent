# Full V3 Phase 8 Creativity / World Model / Coverage - 2026-06-11

## Goal

Address the three current bottlenecks after the recursive self-evolution loop became operational:

- hypothesis generator creativity: avoid only producing local repair candidates;
- world-model prediction: separate quality retention from coverage exploration;
- fresh-live coverage: expand active interventions without replacing a higher-utility default profile.

## Implementation

New module:

`assumption_os/full_v3_phase8_creativity_world_coverage.py`

This builder does not call APIs and does not mutate the graph.  It reads compact first-party artifacts, including the fresh-live v4 quality profile and the phase8 conditional S24 coverage run.

## Generator Upgrade

The phase8 generator evidence adds 8 live-residual-derived candidates:

- hotspot bottleneck subpolicy;
- route-boundary world model;
- S25 emergence-boundary negative controls;
- nonlocal residual-axis generation rule;
- math counterexample abstention;
- business S08 placebo guard;
- S06 special-case transfer;
- profile manifest/tracing update.

Metrics:

- baseline phase4 candidate count: 12
- phase8 creative candidate count: 8
- combined candidate count: 20
- nonlocal new-family count: 7
- nonlocal candidate ratio: 0.35
- residual cluster coverage: 1.0

## World-Model Selector

The selector scores fresh-live profiles separately for:

- quality retention: should this become the default guarded policy?
- coverage exploration: does it expand active rows while staying positive?

Result:

- selected quality profile: `quality_v4`
- selected coverage profile: `coverage_v6`
- quality-world-model AUROC: 1.0
- quality-world-model Brier: 0.1156
- base-rate Brier: 0.2222

## Fresh-Live Coverage Result

Coverage v6 tested a conditional S24 software-engineering guard.  It only activates on profiling/hotspot bottleneck cues such as CPU time, processing time, abnormal slowness, and explicit backtracking hotspots.

Result:

- active interventions: 31 -> 35
- structural vs base utility: 0.5108
- structural vs placebo utility: 0.5135
- active gain over quality profile: +4

Decision:

Coverage v6 is positive but not retained as the default because quality v4 is stronger:

- quality v4: base 0.5144, placebo 0.5153
- coverage v6: base 0.5108, placebo 0.5135

So the default remains v4.  v6 is retained as a coverage-exploration profile and as training evidence for the world-model selector.

## Validation

Command:

```bash
python3 -m assumption_os.full_v3_phase8_creativity_world_coverage \
  --root . \
  --out 'phase four/assumption_graph/paper_readiness_20260604/full_v3_phase8_creativity_world_coverage_20260611.json'
```

Result: pass.
