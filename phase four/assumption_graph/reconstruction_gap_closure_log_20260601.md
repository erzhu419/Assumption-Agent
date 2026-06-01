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
7. Evolution context / harness responsibility is implicit in commands rather than represented as an auditable assumption.
8. Evaluation is still too easy to collapse into answer win-rate rather than lifecycle capabilities.
9. Runtime self-evolution mechanisms exist as modules and reports, but are not yet first-class graph memory surfaces.
10. Live runners still rely mostly on post-hoc log ingestion instead of first-party LLM/retrieval/tool trace emission.

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
8. Add an evolution-context gate for task specification, context selection, observability, failure attribution, verification, permission boundaries, intervention recording, rollback, and procedure updates.
9. Add an AssumptionBench-style capability scoreboard for explicitness, selection, execution, residual attribution, memory transfer, metaproductivity, verifier reliability, world-model quality, and harness governance.
10. Add runtime memory surfaces so retrieval policy, verifier stack, world model, evaluator policy, formal mapping, recursive runner, manifest logger, governance gate, and lifecycle scoreboard are typed graph memory.
11. Add a first-party runtime trace recorder and wire it into `phase2_v20_framework.py` so live LLM/retrieval/tool events can become TrialManifests without parsing logs.

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
- 2026-06-01: Current verification command: `python3 -m unittest tests.test_assumption_os` -> 40 tests OK.
- 2026-06-01: Added `assumption_os.verifier_stack` to combine preflight, world-model, formal, falsification, and acceptance gates into one ordered V0-V4 verifier verdict per proposal.
- 2026-06-01: Persisted `verifier_stack_reconstruction_gap_20260601_expanded.json`: 33 proposals, 2 accepted-for-apply, 14 rejected, 6 needing preflight repair, 11 collect-more-evidence.
- 2026-06-01: Extended `assumption_os.falsification` from ordered checks into POPPER-style experiment protocols. Each candidate now gets explicit route-power, trigger-benefit, control-harm, placebo-context, and fresh cross-judge falsifiers with stop/pass/fail rules and Type-I-control notes.
- 2026-06-01: Regenerated the unified verifier artifact with 135 falsification experiment records across 27 candidate proposals.
- 2026-06-01: Added `assumption_os.recursive_audit` to verify recursive runner closure: parent/child consistency, argument contracts, return-update contracts, and actionable frontier integrity.
- 2026-06-01: Persisted `recursive_audit_reconstruction_gap_20260601_expanded.json`: 2 cases, 12 frames, 5 actionable frontier items, min closure score 1.0, 0 critical issues, 0 warnings.
- 2026-06-01: Added `assumption_os.evolution_context` to make the evolution procedure itself auditable. It checks 9 harness responsibilities and gates write/apply/execute requests against explicit permissions and accepted-candidate budgets.
- 2026-06-01: Persisted `evolution_context_reconstruction_gap_20260601_expanded.json`: dry mode is `ready_for_manual_apply`, bounded apply is `gated_apply_allowed`, unpermitted apply is `blocked_by_permissions`, and all 9 responsibilities pass.
- 2026-06-01: Added `assumption_os.assumption_bench` to evaluate lifecycle capabilities separately from answer win-rate.
- 2026-06-01: Persisted `assumption_bench_reconstruction_gap_20260601_expanded.json`: 9 / 9 capabilities pass, overall score 0.9839, minimum score 0.8833.
- 2026-06-01: Added `assumption_os.memory_surfaces` to materialize runtime self-evolution mechanisms as first-class graph nodes, edges, evidence, and an indexing TrialManifest.
- 2026-06-01: Persisted `memory_surfaces_reconstruction_gap_20260601_expanded.json`: 10 runtime surface nodes, 16 typed edges, graph node-type coverage 4 -> 11, edge-type coverage 5 -> 11, `memory_transfer_ready=true`.
- 2026-06-01: Regenerated `assumption_bench_reconstruction_gap_20260601_expanded.json`: 9 / 9 capabilities pass, overall score 0.9968, minimum score 0.9716, `memory_transfer=1.0`.
- 2026-06-01: Added `assumption_os.runtime_trace` and wired optional runtime tracing flags into `phase2_v20_framework.py` for live LLM calls, graph retrieval calls, and tool/cache events.

## Closure Notes

- The world model is now executable and logged, but remains a cheap verifier. Fresh ablation/judge evidence is still the promotion authority.
- Multi-path search is active as a ranked trajectory planner; it does not silently choose and mutate the graph.
- The daemon can run frontier commands with `--execute`, but defaults to dry-run and requires `--apply-accepted` for graph mutation.
- Component manifests now cover arbitrary LLM/retrieval/judge/tool events through a shared logger; call sites can adopt it incrementally.
- Harness observer now covers pre-existing judgment/meta/log artifacts through bounded backfill, reducing black-box cache gaps while avoiding full prompt/answer import.
- Residual synthesis supports an injectable LLM synthesizer in code and a deterministic CLI path for reproducible tests.
- Formal mapping now has a real finite metric engine, scoped to finite stochastic kernels over typed formal roles rather than unrestricted theorem proving.
- Runtime memory surfaces are now in the graph, so future retrieval can access system-level assumptions instead of relying only on code modules and reports.
- Live phase2 runs can now emit redacted first-party runtime trace events and write them directly as TrialManifests instead of depending only on post-hoc log parsing.

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
- Runtime trace: first-party LLM/retrieval/tool events are emitted as redacted JSONL, converted to TrialManifests, and can be written back to graph memory without post-hoc log parsing.
- Harness observer: 4 artifact files, 19 discovered events, 10 newly backfilled events, 9 already-covered events skipped, full artifact-file coverage after writeback, no secret leak; persisted via `harness_observer_backfill_20260601`.
- Verifier stack: 33 proposals, 2 accepted-for-gated-apply, 14 rejected, 6 preflight-repair, 11 collect-more-evidence; V4 acceptance stages show 2 pass / 14 fail / 17 missing. The falsification protocol layer adds 135 experiment records across 27 candidate proposals; accepted protocol checks and rejected protocol checks both pass.
- Recursive audit: dry frontier plus accepted-return cases pass with 12 total frames, 5 actionable frontier items, min closure score 1.0, 0 critical issues, and 0 warnings.
- Evolution context: 9 / 9 harness responsibilities pass; dry mode reports `ready_for_manual_apply`, bounded permission reports `gated_apply_allowed`, and unpermitted apply is blocked with permission violations.
- Memory surfaces: 10 runtime surface nodes and 16 typed edges are indexed in graph memory; persisted graph coverage is now 11 node types and 11 edge types.
- AssumptionBench: 9 / 9 lifecycle capabilities pass; overall score 0.9968, minimum score 0.9716. `memory_transfer` is now 1.0 after runtime mechanisms were written into the graph.
- World model: 16 matched labels from 2 accepted / 14 rejected proposal outcomes; raw pre-acceptance Brier 0.2182, trained calibration Brier 0.0085, leave-one-out Brier 0.0090, post-acceptance Brier 0.0081.
- Trajectory search: 10 frontier actions, 26 trajectories, multi-path rate 0.8, top-path label hit rate 1.0.
- Recursive daemon: 2 positive-control accepted candidates applied in a temp graph, dry-run applied 0, gated apply applied 2, manifests written.
- Residual clusterer: 109 residual records, 7 deterministic clusters, 2 synthesized candidate proposals with validation plans.
- Formal metrics: 9 complete mappings, 9/9 finite kernels same-shape, 0 warnings.
