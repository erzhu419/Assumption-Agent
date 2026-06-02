# Reconstruction Gap Backlog - 2026-06-02

This backlog records the remaining distance to `reconstruction.md` after the
controlled artifact-readback and external V5 objective-gate validation passes.

## Current Completion Snapshot

- Structure: 86.4%
- Behavior: 78.1%
- Weighted: 81.9%
- Completed items: 6 / 9
- Strongest component: recursive execution loop, currently 0.92 structure / 0.84 behavior on bounded and cached real readback.

## Remaining Gaps

1. World model data is not yet 1000+ independent first-party live traces.
   The current 1000+ rows are mostly distilled transitions from a smaller
   first-party seed plus artifact replay. The next acceptance bar is larger
   raw runtime coverage from real recursive/daemon executions.

2. Trace-policy proposals need broader real fresh ablation with controls.
   The latest accepted real example proves artifact readback can resume the
   recursive runner, but a 3-trigger-only sample is too narrow for graph
   promotion. Future real readbacks must report trigger benefit, control
   coverage, and no control harm before they count as controlled promotion.

3. V5 objective gate has an external benchmark hook, but not live heldout tasks.
   The verifier now requires accepted candidates to pass external objective
   tasks when an objective benchmark is supplied. Current validation uses
   deterministic positive-control tasks; the remaining bar is live heldout
   objective tasks with real model outputs and judge results.

4. Residual labels need broader calibration.
   Curated-gold agreement tests exist, but the remaining bar is a larger
   human/LLM-adjudicated failure set covering execution, optimization,
   assumption, memory, evaluator, and simulator residuals.

5. ACP/metaproductivity needs larger learned histories.
   ACP currently learns from accepted/rejected descendants, but cost, risk,
   novelty, and scheduler allocation are still partly hand-weighted.

6. Formal mapping is still a bounded audit/gate.
   It supports finite-kernel deduplication and transfer probes, not a full
   category-theoretic or information-geometric reasoning engine.

7. The daemon is bounded and gated, not fully unattended.
   This is intentional for safety, but the reconstruction target eventually
   needs continuous queue execution, readback, residual logging, and gated
   proposal generation under explicit budget and mutation policies.

## Next Work Order

1. Harden real artifact readback so accepted evidence must include trigger and
   control quality metrics. Done in `0a21462`.
2. Re-read existing cached proposal artifacts to find at least one accepted
   controlled promotion without spending new model calls. Done in `0a21462`.
3. Add an external V5 objective-task gate so accepted proposals are not promoted
   solely by trigger/control acceptance. Done in current external-V5 pass.
4. Replace deterministic V5 positive controls with live heldout objective tasks
   once API environment variables are available.
5. Run a small fresh GPT ablation with
   trigger and control rows once API environment variables are available.
6. Feed the controlled result through recursive daemon readback, performance
   validation, reconstruction progress, and assumption bench.
7. Only push when the validation evidence improves or exposes a stricter,
   useful gate with passing tests.
