# HLE Option Evidence Gate and Module Ablation

Date: 2026-06-17

## Scope

Implemented the next HLE anti-pollution steps:

- executable-style multiple-choice option evidence gate
- verified-or-abstain integration for strong option-specific evidence
- strict ambiguous / weak-support blocking
- automatic module ablation runner with same seed/window profiles
- real module-off toggles for graph retrieval, structural morphism, recursive runner, world-model router, evidence bridge, agent HippoRAG, MC option evidence, and verified-or-abstain

Artifacts store only hashes, counts, status fields, and aggregate metrics. Raw HLE questions, answers, rationales, canaries, prediction text, and API keys are not persisted.

## Implementation Notes

`mc_option_evidence_scorer` now emits a verified candidate only when all are true:

- top score is high enough
- margin over runner-up is high enough
- top option has at least two discriminative supporting docs
- top option has more supporting docs than runner-up
- top option has no ambiguous docs that also support competing labels

If these conditions fail, the scorer remains diagnostic and cannot override raw/majority selection.

New selection method:

- `verified_option_evidence_priority`

New blocker statuses:

- `blocked_non_discriminative_option_evidence`
- `blocked_ambiguous_option_evidence`
- `blocked_weak_support_count`

New ablation module:

- `assumption_os.hle_module_ablation_runner`

Default profiles:

- `full`
- `verified_gate_off`
- `no_option_evidence`
- `no_agent_hipporag`
- `no_evidence`
- `no_graph`
- `no_morphism`
- `no_recursive_runner`
- `no_world_model_router`

## Validation

Unit and compile validation:

- `python3 -m unittest tests.test_hle_smoke_eval tests.test_hle_parallel_shard_runner`
- result: 69 tests OK
- `python3 -m py_compile assumption_os/hle_smoke_eval.py assumption_os/hle_parallel_shard_runner.py assumption_os/hle_module_ablation_runner.py`
- result: pass

Dry-run module ablation:

- eval id: `hle_module_ablation_dryrun_current_gate_seed3100_20260617`
- profiles: 8
- pass: true
- failed gates: none

Live negative-control finding before the final gate tightening:

- eval id: `hle_module_ablation_live_n1_option_evidence_seed0_20260617`
- profile result: process / paper-clean / pollution gates passed
- downstream accuracy: raw, HippoRAG, and agent were all wrong on this one problem
- important finding: old option evidence gate selected `verified_option_evidence_priority`, but `candidate_correct_for_eval=false`
- conclusion: one supporting doc plus score margin was not strict enough

Fixed-gate live smoke:

- eval id: `hle_updated_option_gate_live_n1_agent_seed0_20260617`
- pass: true
- paper-clean pass: true
- pollution pass: true
- sample count: 1
- resolved live calls: 1 / 1
- option evidence status: `blocked_weak_support_count`
- candidate emitted: false
- selection method: `verified_or_abstain_direct_fallback`
- downstream accuracy on this single item: 0.0

This means the new gate did not improve that problem, but it correctly prevented a false verified override that the previous version allowed.

## Claim Boundary

This step supports the claim that the HLE option-evidence path is now safer and auditable. It does not support a claim of improved HLE accuracy yet.

The next performance step should run the automatic ablation matrix on fresh n=12/n=30 windows after the remaining evidence modules are strengthened, and should require problem-level clean-shared reporting before any downstream improvement claim.
