# Full V2 Phase 1: Shadow Graph Memory Bypass

Date: 2026-06-11

## Scope

This phase adds a shadow graph-memory retrieval policy without replacing existing `graph_memory` behavior.  The goal is to activate the correct assumption subgraph, not merely the most lexically similar nodes.

## Implementation

- Added `assumption_os/full_v2_phase1_graph_memory_bypass.py`.
- Uses a frozen fixture of assumption/method/verifier/risk nodes.
- Compares semantic-only retrieval against full-v2 scoring.
- Full-v2 score combines:
  - semantic relevance;
  - domain match;
  - residual match;
  - graph centrality;
  - confidence;
  - ACP / metaproductivity;
  - verifier availability;
  - regression-risk penalty;
  - context-cost penalty.

## Performance Validation

Command:

```bash
python3 -m assumption_os.full_v2_phase1_graph_memory_bypass \
  --root . \
  --eval-id full_v2_phase1_graph_memory_bypass_20260611 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/full_v2_phase1_graph_memory_bypass_20260611.json'
```

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/full_v2_phase1_graph_memory_bypass_20260611.json`

Metrics:

- query count: 5
- semantic top-k precision: 0.4667
- full top-k precision: 0.6667
- semantic top-1 accuracy: 1.0000
- full top-1 accuracy: 1.0000
- semantic negative-transfer rate: 0.4000
- full negative-transfer rate: 0.0000
- risky node top-k count: 0
- semantic context efficiency: 2.4240
- full context efficiency: 4.3214
- residual retrieval accuracy: 1.0000

Unit validation:

```bash
python3 -m unittest tests.test_assumption_os.AssumptionOSTest.test_full_v2_phase1_graph_memory_bypass_demotes_risky_context
```

Result: `OK`.

## Interpretation

The full-v2 retriever demotes lexical distractors and risky graph-context nodes while preserving correct top-1 activation.  This is the intended Phase 1 upgrade: conditional graph context, not more graph context.

## Boundary

This is a frozen fixture and a shadow scoring policy.  It does not yet perform large-scale PPR over the committed Assumption Graph or sleep-stage memory consolidation.
