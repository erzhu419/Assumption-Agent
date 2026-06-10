# V2 Phase 6: Residual-Triggered Hypothesis Generation

Date: 2026-06-10

## Scope

This phase implements disciplined hypothesis generation: no candidate is synthesized unless a systematic residual cluster exists.  The validation fixture uses formal-alignment baseline failures as residuals, then generates candidate repair hypotheses with manifests, graph overlays, world-model screening, heldout replay, and outside negative controls.

## Implementation

- Added `assumption_os/residual_hypothesis_generator_v2.py`.
- Source residuals come from Phase 5 baseline errors:
  - semantic aligner false negatives;
  - graph-edit role-similarity false negatives;
  - trajectory JS-similarity false negatives.
- Each systematic residual cluster produces exactly one candidate hypothesis.
- Each candidate includes:
  - `AssumptionManifestV2`;
  - graph overlay ops;
  - novelty / duplicate / conflict check;
  - world-model preflight screen;
  - heldout residual replay;
  - outside negative-control replay.

## Performance Validation

Command:

```bash
python3 -m assumption_os.residual_hypothesis_generator_v2 \
  --root . \
  --eval-id residual_hypothesis_generator_v2_20260610 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/residual_hypothesis_generator_v2_20260610.json'
```

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/residual_hypothesis_generator_v2_20260610.json`

Metrics:

- source formal alignment pass: true
- residual count: 15
- cluster count: 3
- clustered residual count: 15
- clustered residual fraction: 1.0000
- proposal count: 3
- random proposal count: 0
- duplicate claim count: 0
- conflict count: 0
- world-model accepted proposals: 3
- heldout total: 9
- heldout covered: 9
- heldout residual coverage: 1.0000
- outside-control harm count: 0
- manifest validation issue count: 0

Cluster signatures:

- `llm_semantic_aligner_proxy:missed_positive_alignment`: 7
- `graph_edit_role_similarity:missed_positive_alignment`: 5
- `trajectory_js_similarity:missed_positive_alignment`: 3

Generated hypotheses:

- Require a typed process-family bridge when lexical semantic alignment misses a formally accepted process pair.
- Allow sparse role overlap only when shared family, finite diagram, invariant preservation, and causal-mask support all pass.
- Treat trajectory information geometry as supporting evidence rather than a hard gate when typed invariants and negative controls support alignment.

Unit validation:

```bash
python3 -m unittest tests.test_assumption_os.AssumptionOSTest.test_residual_hypothesis_generator_v2_requires_systematic_clusters
```

Result: `OK`.

## Interpretation

This phase implements the variation step in a safer form: variation is triggered by systematic residuals, then immediately constrained by evaluation and selective-retention gates.  The output is ready for the recursive runner, but it is still a candidate queue; it does not mutate the committed graph by itself.
