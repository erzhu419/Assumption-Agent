# GPT Revise V3 Gap Audit - 2026-06-11

This note records the current status against `GPT_revise_v3.md` after the
latest verifier-protocol pass.

## Implemented

- Learned selector path: Phase10 added a discrete graph-action world-model
  selector; raw selector remains shadow when uncalibrated, while the calibrated
  residual guard is promoted for production scheduling.
- Phase naming / capability audit: Phase11 separates implementation levels and
  marks fixture, shadow, live-derived, and production artifacts explicitly.
- Contract checker: Phase0 production contract gate validates overlay admission
  before proposal application.
- Memory consolidation: Phase1 has a JSONL sleep job with dry-run/apply modes,
  archive semantics, and consolidation records.
- Residual clusterer: live residual clustering is connected to Phase4 proposal
  seed generation.
- Graph-action world model: Phase10 uses redacted state bits, action arms, and a
  calibrated guard instead of directly promoting an uncalibrated predictor.
- Verifier protocol: `verifier_stack.py` now emits an explicit
  `VerifierProtocol` per proposal type and a `protocol_report` for conformance.

## VerifierProtocol coverage

The stack now distinguishes these protocols:

- `method_hypothesis_candidate`
- `retrieval_policy_candidate`
- `bounded_structural_morphism_candidate`
- `world_model_calibration_candidate`
- `memory_consolidation_candidate`
- `prompt_guard_candidate`

Each protocol records required verifier stages, negative controls, objective
evidence, acceptance thresholds, manual-review policy, default next action, and
blocked claims. Accepted graph mutations must still pass manual gated apply.

## Validation

- Targeted verifier tests: 3 tests passed.
- Full unit suite: 149 tests passed.
- Performance validation: `overall_pass=true`.
- Reconstruction progress in performance validation:
  - structure: 86.4%
  - behavior: 78.1%
- Verifier stack remained passing after protocolization.

## Still not fully solved

- Phase10 learned selector still needs a fresh, same-batch V1/V3/no-world-model
  comparison on a larger heldout slice.
- Phase1 memory consolidation still needs more first-party before/after
  retrieval evidence beyond the current job/probe path.
- Phase3 rollout search is still not a strong learned simulator; it needs
  training on live manifest trajectories.
- Phase7 daemon is bounded and gated, not a long-running unattended production
  scheduler.
- The residual generator can produce live seeds, but multi-generation
  residual-cluster -> LLM synthesis -> trajectory search -> ablation loops
  should be expanded on fresh tasks.
- The formal/morphism layer remains a bounded structural gate, not a complete
  category-theory theorem prover.

## Next best work

Run a same-batch fresh benchmark that toggles:

- V3 calibrated residual guard
- V3 no world model
- V3 no recursive runner
- V3 no morphism
- V1 baseline

Report problem-level utility, active coverage, bootstrap confidence intervals,
and domain breakdown. This is the cleanest remaining evidence gap for paper
claims.
