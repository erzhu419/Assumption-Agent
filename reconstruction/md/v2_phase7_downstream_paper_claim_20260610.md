# V2 Phase 7: Frozen Downstream Mechanism Claim

Date: 2026-06-10

## Scope

This phase builds a clean frozen benchmark line for the v2 reconstruction.  It compares the full v2 recursive assumption-graph stack against required local baselines and module ablations on typed process-alignment tasks.

This is a mechanism benchmark, not a full HippoRAG QA or broad unseen reasoning benchmark.

## Implementation

- Added `assumption_os/downstream_paper_claim_v2.py`.
- Sources:
  - Phase 5 formal alignment certificates;
  - Phase 3 graph-action world-model cost reduction;
  - Phase 6 residual-triggered hypothesis generation coverage.
- Compared systems:
  - ordinary RAG / semantic retrieval proxy;
  - HippoRAG-style graph retrieval proxy;
  - v16/v20 case-backed reflection proxy;
  - no formal alignment;
  - no world model;
  - no recursive generator;
  - full recursive assumption graph v2.

Mechanism utility is defined as:

```text
0.55 * alignment_accuracy
+ 0.20 * heldout_residual_coverage
+ 0.15 * screen_cost_reduction
+ 0.10 * negative_control_safety
```

## Performance Validation

Command:

```bash
python3 -m assumption_os.downstream_paper_claim_v2 \
  --root . \
  --eval-id downstream_paper_claim_v2_20260610 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/downstream_paper_claim_v2_20260610.json'
```

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/downstream_paper_claim_v2_20260610.json`

Main metrics:

- problem count: 16
- system count: 7
- full accuracy: 1.0000
- best retrieval/no-formal accuracy: 0.8125
- accuracy margin over retrieval/no-formal: 0.1875
- full mechanism utility: 0.9156
- best non-full mechanism utility: 0.8500
- utility margin over best non-full: 0.0656
- full negative-control safety: 1.0000
- full residual coverage: 1.0000
- full screen cost reduction: 0.4375
- bootstrap accuracy-margin mean: 0.1892
- bootstrap 95% CI: [0.0000, 0.3750]
- paired full wins over no-formal: 3
- paired full losses over no-formal: 0

System rows:

- ordinary RAG / semantic retrieval proxy: accuracy 0.5625, utility 0.4094
- HippoRAG-style graph retrieval proxy: accuracy 0.6875, utility 0.4781
- v16/v20 case-backed reflection proxy: accuracy 0.8125, utility 0.5469
- no formal alignment: accuracy 0.8125, utility 0.5469
- no world model: accuracy 1.0000, utility 0.8500
- no recursive generator: accuracy 1.0000, utility 0.7156
- full recursive assumption graph v2: accuracy 1.0000, utility 0.9156

Unit validation:

```bash
python3 -m unittest tests.test_assumption_os.AssumptionOSTest.test_downstream_paper_claim_v2_builds_frozen_mechanism_line
```

Result: `OK`.

## Interpretation

The full v2 stack beats retrieval/no-formal baselines in alignment accuracy and beats module ablations in mechanism utility.  The world model matters by reducing screen cost; the recursive residual generator matters by preserving heldout residual coverage; the formal layer matters by improving process-alignment accuracy over graph/semantic/case proxies.

## Boundary

The bootstrap lower bound touches zero because the local fixture has only 16 process-pair problems.  This validates the v2 mechanism line and produces a clean table for a methods claim, but it is not yet sufficient for a broad paper claim that the system beats HippoRAG/RAG/one-shot self-improve on full unseen QA or reasoning benchmarks.
