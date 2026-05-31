# Recursive Assumption Agent Core

This package is the reconstruction layer for the project. It keeps the old
experiment scripts intact, but lifts their artifacts into one shared substrate:
an Assumption Graph.

## What Changed

- `schema.py` defines assumptions, edges, evidence, and trial manifests.
- `adapters.py` converts existing `strategies/`, `wisdom_library.json`,
  `v16_residuals.json`, and Exp82 typed hypotheses into graph nodes.
- `graph_memory.py` stores the graph as JSONL and performs HippoRAG-style
  spreading retrieval over assumption nodes instead of plain passage chunks.
- `record_phase2_eval.py` writes pairwise Phase2 evaluation outcomes back as
  `TrialManifest` records, evidence, confidence updates, and residual links.
- `retrieval_policy.py` adds domain-aware reranking and execution checks for
  prompt injection. The first policy targets software-engineering negative
  transfer observed in the 21-50 heldout audit.
- `selector.py` ranks assumptions with a metaproductivity-aware score inspired
  by HGM, so a method can be useful because its descendants are productive, not
  only because it won one A/B test.
- `residuals.py` implements the EmbodiSkill distinction between an assumption
  being wrong and the executor simply failing to apply a valid assumption.
- `context.py` formats activated subgraphs for future v16/v20/Exp82 prompt
  integration.

## Build The Graph

```bash
python3 -m assumption_os.build_graph --out "phase four/assumption_graph"
```

Use `--fresh` when the JSONL graph should be rebuilt from source artifacts
before replaying evaluation writebacks:

```bash
python3 -m assumption_os.build_graph --out "phase four/assumption_graph" --fresh
```

Current build ingests:

- `phase zero/kb/strategies/S*.json`
- `phase two/analysis/cache/wisdom_library.json`
- `phase four/residuals/v16_residuals.json`
- `phase six/exp82/hypotheses.jsonl`

The build writes:

- `nodes.jsonl`
- `edges.jsonl`
- `evidence.jsonl`
- `trials.jsonl`
- `build_summary.json`

## Write Back Evaluation Outcomes

After generating and judging a graph-augmented Phase2 variant, replay the
judgment cache into the graph:

```bash
python3 -m assumption_os.record_phase2_eval \
  --graph-dir "phase four/assumption_graph" \
  --sample "phase two/analysis/cache/sample_100.json" \
  --meta "phase two/analysis/cache/answers/phase2_v20_ag_filt_gpt55_meta.json" \
  --judgments \
    "phase two/analysis/cache/judgments/phase2_v20_ag_filt_gpt55_vs_phase2_v20_gpt55.json" \
    "phase two/analysis/cache/judgments/phase2_v20_gpt55_vs_phase2_v20_ag_filt_gpt55.json" \
  --intervention phase2_v20_ag_filt_gpt55 \
  --baseline phase2_v20_gpt55 \
  --eval-id phase2_v20_ag_filt_gpt55_vs_gpt55_n20 \
  --summary-out "phase four/assumption_graph/eval_phase2_v20_gpt55_n20_writeback.json"
```

The writeback intentionally skips rows without intervention meta by default;
for v20 this excludes math/science hygiene-bypass rows where graph context was
not injected.

## Phase2 v20 Integration Notes

`phase one/scripts/validation/phase2_v20_framework.py` accepts
`--assumption-graph` to inject activated graph context into Turn 1. After the
21-50 heldout audit, software engineering is gated off by default because graph
context caused negative transfer there:

```bash
python3 "phase one/scripts/validation/phase2_v20_framework.py" \
  --variant phase2_v20_ag_next \
  --assumption-graph "phase four/assumption_graph"
```

To force graph injection for all non-bypass domains:

```bash
python3 "phase one/scripts/validation/phase2_v20_framework.py" \
  --variant phase2_v20_ag_full \
  --assumption-graph "phase four/assumption_graph" \
  --assumption-graph-skip-domains ""
```

The software-engineering reranker is available when the skip gate is disabled,
but it is not yet enabled by default. In the targeted rerun it improved
retrieval strongly but only reached neutral answer quality, so the default gate
remains conservative.

When `--assumption-graph` is supplied, v20 also enables a small domain execution
template layer from `assumption_os.domain_templates`. For software engineering,
this layer stays active even while graph context is skipped, so SE tasks get
concrete execution constraints without attributing the result to retrieved graph
nodes. On the 10 heldout SE problems from `sample_21_50`, this template-only
intervention moved the bidirectional result from the learned-graph failure
case's 20.0% decisive win rate to 62.5%. Use
`--disable-domain-execution-template` to turn this layer off.

Combining learned graph answers for non-SE domains with the SE template answers
moves the full 21-50 heldout bidirectional result from 45.7% to 59.6% decisive
win rate against `phase2_v20_gpt55`.

Against the plain raw `baseline` prompt on the same 21-50 heldout slice, the
combined policy scores 55 wins, 5 losses, and 0 ties bidirectionally, for a
91.7% decisive win rate.

## Conditioned Evaluation Gate

`assumption_os.conditioned_eval` implements the first self-evolution gate after
writeback: route each judged problem into `should_fire`, `no_fire`, or `neutral`
for a candidate node, then compute benefit only on active `should_fire` rows and
harm only on active `no_fire` rows.  The gate can output `promote`, `keep`,
`expand_retrieval`, `narrow_scope`, `revise`, or `insufficient_evidence`.

```bash
python3 -m assumption_os.conditioned_eval \
  --graph-dir "phase four/assumption_graph" \
  --sample "phase two/analysis/cache/sample_21_50.json" \
  --meta "phase two/analysis/cache/answers/phase2_v20_ag_learned_gpt55_meta.json" \
  --judgments \
    "phase two/analysis/cache/judgments/phase2_v20_ag_learned_gpt55_vs_phase2_v20_gpt55.json" \
    "phase two/analysis/cache/judgments/phase2_v20_gpt55_vs_phase2_v20_ag_learned_gpt55.json" \
  --intervention phase2_v20_ag_learned_gpt55 \
  --baseline phase2_v20_gpt55 \
  --summary-out "phase four/assumption_graph/conditioned_eval_phase2_v20_gpt55_21_50.json"
```

## Why This Shape

The reconstruction documents and the reference papers point to the same failure
mode: wisdom-as-prompt text is too weak. The system needs explicit lifecycle
objects:

1. record the assumption behind every key action,
2. retrieve related assumptions, cases, residuals, and verifiers as a graph,
3. test assumptions with manifest-style predictions,
4. classify failures before editing the library,
5. preserve assumptions that failed only because of execution lapse,
6. prefer assumption families with long-run productive descendants.

This package is the first code-level boundary for that lifecycle.
