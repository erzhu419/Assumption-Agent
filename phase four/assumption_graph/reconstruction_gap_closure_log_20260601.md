# Reconstruction Gap Closure Log - 2026-06-01

## Scope

Target: close the remaining gaps called out in `reconstruction/md/reconstruction.md` while keeping the existing Assumption Graph and recursive runner architecture intact.

Current gaps being addressed:

1. No executable world model / simulator as a cheap verifier for candidate assumptions.
2. No multi-path hypothesis trajectory search over candidate futures.
3. No daemon-like loop that can execute frontier `command_hint`, read results, resume recursion, and continue.
4. Not every LLM call, retrieval, judge call, or tool-use has a TrialManifest-shaped log.
5. No residual clustering pipeline that turns systematic residuals into new method hypotheses plus heldout validation plans.
6. Formal mapping is an audit/search layer, not yet an executable finite categorical / information-geometry reasoning layer.

## Design Constraints

- Graph mutation remains gated. Default commands plan and log; writeback/apply require explicit flags.
- No API keys are written to code, docs, logs, or test data. Provider credentials remain environment-only.
- New modules use the existing `TrialManifest`, `JsonlGraphStore`, candidate payload, and recursive runner contracts.
- The formal layer is implemented as a bounded finite-kernel engine rather than an over-claimed full theorem prover.

## Implementation Plan

1. Add a component manifest logger for LLM/retrieval/judge/tool events.
2. Add a cheap assumption world model that predicts candidate acceptance/regression risk and emits simulator manifests.
3. Add multi-path trajectory search that rolls out proposal action paths with world-model scores.
4. Add a recursive daemon driver that plans/executes frontier commands, ingests judgment bundles, resumes recursion, and optionally applies accepted candidates.
5. Add residual clustering and synthesis from systematic residual groups into candidate method hypotheses and validation plans.
6. Extend formal mapping with finite stochastic-kernel metrics: categorical objects/morphisms, row KL, total variation, Frobenius distance, and a Blackwell-style dominance proxy.
7. Add unit tests and README documentation for the new loop.

## Progress

- 2026-06-01: Confirmed existing schema already has `AssumptionType.WORLD_MODEL`, `ResidualType.SIMULATOR_DEFECT`, and reusable `TrialManifest`.
- 2026-06-01: Confirmed existing recursive runner/executor already supports planned frontier commands and gated acceptance resume; daemon can build on this instead of duplicating it.
- 2026-06-01: Confirmed existing formal mapping layer has role audit/search but no metric kernel payload yet.
- 2026-06-01: Added `assumption_os.manifest_logger` for redacted LLM/retrieval/judge/tool/simulator manifests.
- 2026-06-01: Added `assumption_os.harness_observer` to audit existing judge/meta/log artifacts and backfill manifest coverage for pre-existing harness files.
- 2026-06-01: Added `assumption_os.world_model` and integrated it into `evolution_cycle` as the cheap verifier/simulator stage.
- 2026-06-01: Added `assumption_os.trajectory_search` for multi-path promote/repair/evidence/reject futures over recursive frontier actions.
- 2026-06-01: Added `assumption_os.recursive_daemon` for bounded command planning/execution, judgment ingestion, recursive resume, manifest logging, and gated apply.
- 2026-06-01: Added `assumption_os.residual_clusterer` for systematic residual clusters -> synthesized method candidates -> heldout validation plans.
- 2026-06-01: Extended `assumption_os.formal_mapping` with finite stochastic-kernel categorical/info-geometry metrics.
- 2026-06-01: Added unit coverage for all six reconstruction gaps. Verification command: `python3 -m unittest tests.test_assumption_os` -> 36 tests OK.
- 2026-06-01: Added `assumption_os.performance_validation` and ran non-smoke validation for the six gap closures. First run exposed world-model post-acceptance miscalibration (`brier_score=0.2767`) because rejected evidence still left high predicted acceptance probabilities.
- 2026-06-01: Calibrated world-model probabilities after real acceptance evidence: accepted candidates floor to high confidence, rejected-harm/rejected-benefit candidates cap to low acceptance probability, and priority now uses scaled raw priority instead of a 1.0 clamp. Rerun passed all six sections with `world_model.post_calibration.brier_score=0.0359`.
- 2026-06-01: Expanded validation coverage by adding the proposal-screening payload/preflight to the labeled set, training an explicit world-model calibration payload, adding leave-one-out calibration metrics, and parsing existing run/judge logs into component manifests.
- 2026-06-01: Promoted world-model calibration from report-only output into reusable artifacts: `world_model` can now train/save calibration from raw pre-acceptance predictions, and `evolution_cycle` can load or inline-train that calibration before screening candidate trajectories.
- 2026-06-01: Persisted the expanded 16-label calibration artifacts: `world_model_raw_reconstruction_gap_20260601_expanded.json` and `world_model_calibration_reconstruction_gap_20260601_expanded.json`.
- 2026-06-01: Harness observer discovered 19 real artifact events from one judgment JSON, one answer-meta JSON, and two run logs; it backfilled the 10 previously uncovered judgment/meta events into `trials.jsonl` and skipped 9 already covered log events.
- 2026-06-01: Current verification command: `python3 -m unittest tests.test_assumption_os` -> 39 tests OK.

## Closure Notes

- The world model is now executable and logged, but remains a cheap verifier. Fresh ablation/judge evidence is still the promotion authority.
- Multi-path search is active as a ranked trajectory planner; it does not silently choose and mutate the graph.
- The daemon can run frontier commands with `--execute`, but defaults to dry-run and requires `--apply-accepted` for graph mutation.
- Component manifests now cover arbitrary LLM/retrieval/judge/tool events through a shared logger; call sites can adopt it incrementally.
- Harness observer now covers pre-existing judgment/meta/log artifacts through bounded backfill, reducing black-box cache gaps while avoiding full prompt/answer import.
- Residual synthesis supports an injectable LLM synthesizer in code and a deterministic CLI path for reproducible tests.
- Formal mapping now has a real finite metric engine, scoped to finite stochastic kernels over typed formal roles rather than unrestricted theorem proving.

## Performance Validation - 2026-06-01

Command:

```bash
python3 -m assumption_os.performance_validation \
  --root . \
  --graph-dir "phase four/assumption_graph" \
  --eval-id reconstruction_gap_perf_20260601_expanded \
  --summary-out "phase four/assumption_graph/reconstruction_gap_perf_20260601_expanded.json" \
  --report-out "phase four/assumption_graph/reconstruction_gap_perf_20260601_expanded.md"
```

Results:

- Overall: PASS.
- Manifest logger: 112 events, including 12 parsed real run/judge-log events, no secret leak; the 12 real events are persisted in `trials.jsonl` via `real_log_manifest_ingest_20260601`.
- Harness observer: 4 artifact files, 19 discovered events, 10 newly backfilled events, 9 already-covered events skipped, full artifact-file coverage after writeback, no secret leak; persisted via `harness_observer_backfill_20260601`.
- World model: 16 matched labels from 2 accepted / 14 rejected proposal outcomes; raw pre-acceptance Brier 0.5316, trained calibration Brier 0.0060, leave-one-out Brier 0.0064, post-acceptance Brier 0.0081.
- Trajectory search: 10 frontier actions, 26 trajectories, multi-path rate 0.8, top-path label hit rate 1.0.
- Recursive daemon: 2 positive-control accepted candidates applied in a temp graph, dry-run applied 0, gated apply applied 2, manifests written.
- Residual clusterer: 109 residual records, 7 clusters, 3 synthesized candidate proposals with validation plans.
- Formal metrics: 9 complete mappings, 9/9 finite kernels same-shape, 0 warnings.
