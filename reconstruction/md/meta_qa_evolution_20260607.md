# Meta-QA Evolution Probe - 2026-06-07/08

## Purpose

HippoRAG QA probing showed a specific gap: the structural morphism layer is useful for cross-domain structural analogy, but direct morphism retrieval is not a factual multi-hop QA retriever by itself. This update tests the missing adapter: turn QA residuals, pre-reconstruction method-layer "philosophies", and HippoRAG-style context edges into solve-time retrieval policies, evaluate them, retain only non-regressive policies, and then measure QA retrieval and reader effects.

The current claim is narrow:

- meta-QA does not replace HippoRAG;
- direct morphism still mostly abstains on factual QA;
- generalized assumption edges plus gated/learned policy selection improve supporting-evidence retrieval on real HippoRAG reproduction QA files;
- answer-level reader validation is positive on local extractive reader slices, but not yet a full external LLM QA benchmark.

## Mechanism

Inputs used at runtime:

- question text;
- dataset name;
- corpus titles and text;
- deterministic BM25/retrieval diagnostics;
- candidate trigger flags;
- assumption-edge route.

Inputs excluded from runtime ranking/selection:

- gold answers;
- gold titles;
- supporting facts.

The controller evaluates eight hypotheses:

| hypothesis | role |
|---|---|
| `qa_hyp_comparison_dual_anchor` | retrieve both sides of a binary comparison |
| `qa_hyp_anchor_preserve_insert` | preserve BM25 and insert one title anchor |
| `qa_hyp_named_anchor_bridge` | broad named-anchor bridge, rejected when regressive |
| `qa_hyp_generic_prf` | pseudo-relevance feedback, rejected when it amplifies wrong hops |
| `qa_hyp_representation_title_normalization` | canonicalize noisy surface/title mentions |
| `qa_hyp_decomposition_bridge_entity` | anchor page -> role-labeled bridge entity -> bridge page |
| `qa_hyp_controlled_bridge_insert` | bounded bridge insert with BM25 guard |
| `qa_hyp_assumption_edge_policy_selector` | lift HippoRAG context edges into high-precision assumption-edge routes |

The deterministic retention gate keeps accepted hypotheses only when support-chain and answer-coverage utility are positive under bounded harm caps. The 2026-06-08 learned selector adds a cross-fit bucketed policy world model: each fold trains on heldout-fold-excluded retrieval utility/harm labels, then conservatively overrides the deterministic retained policy only if predicted utility clears the harm/benefit thresholds.

This implements the desired variation -> evaluation -> selective retention loop for QA:

1. generate multiple retrieval hypotheses from residuals and method priors;
2. evaluate each hypothesis against support-chain coverage and answer coverage;
3. retain only accepted policies;
4. learn a conservative selector from the accepted/rejected policy utilities;
5. validate retrieval and reader performance without using gold data at runtime.

## Artifacts

- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_heldout60_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_heldout300_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_heldout600_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_full3000_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_learned_heldout300_20260608.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_learned_full3000_20260608.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_reader15_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_reader60_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_reader60_learned_20260608.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_reader300_learned_20260608.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_llm_reader15_gpt54mini_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_llm_reader60_gpt54mini_20260607.json`

Datasets:

- `reference/repos/HippoRAG/reproduce/dataset/hotpotqa.json`
- `reference/repos/HippoRAG/reproduce/dataset/musique.json`
- `reference/repos/HippoRAG/reproduce/dataset/2wikimultihopqa.json`

## Retrieval Performance

All runs use top-k 5. Metrics are problem-level. Bootstrap CI resamples problem rows.

| slice | n | BM25 all | fixed meta all | learned all | learned delta vs BM25 | BM25 fraction | learned fraction | learned delta | BM25 answer | learned answer | learned delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 15 | 0.1333 | 0.6667 | 0.6667 | +0.5334 | 0.5111 | 0.8444 | +0.3333 | 0.2667 | 0.6000 | +0.3333 |
| heldout60 | 60 | 0.3333 | 0.5500 | 0.5500 | +0.2167 | 0.6056 | 0.7681 | +0.1625 | 0.5167 | 0.6167 | +0.1000 |
| heldout300 | 300 | 0.3367 | 0.4533 | 0.4533 | +0.1166 | 0.6064 | 0.7125 | +0.1061 | 0.5133 | 0.5767 | +0.0634 |
| full3000 | 3000 | 0.3077 | 0.3627 | 0.3630 | +0.0553 | 0.5988 | 0.6384 | +0.0396 | 0.4730 | 0.5033 | +0.0303 |

Full3000 learned vs fixed meta controller:

- any gold recall delta: `+0.0074`;
- all gold recall delta: `+0.0003`;
- mean support fraction delta: `+0.0039`;
- answer coverage delta: `+0.0003`.

Full3000 learned-vs-fixed bootstrap CI:

- any gold recall delta: point `+0.0073`, 95% CI `[+0.0043, +0.0107]`;
- all gold recall delta: point `+0.0003`, 95% CI `[-0.0010, +0.0017]`;
- mean support fraction delta: point `+0.0038`, 95% CI `[+0.0022, +0.0057]`;
- answer coverage delta: point `+0.0003`, 95% CI `[-0.0010, +0.0017]`.

Full3000 learned vs BM25:

- any gold recall delta: `+0.0097`;
- all gold recall delta: `+0.0553`;
- mean support fraction delta: `+0.0396`;
- answer coverage delta: `+0.0303`.

Full3000 learned selector selected:

- `ordinary_bm25`: 2349 rows;
- `assumption_edge_policy_selector`: 381 rows;
- `comparison_dual_anchor`: 153 rows;
- `anchor_preserve_insert`: 117 rows.

Changed-row harm: 8 harms among 651 changed rows, harm rate `0.0123`. This is within the learned selector's bounded-risk policy and gives positive full3000 net utility.

## Reader Validation

Local extractive reader:

- model: `distilbert-base-cased-distilled-squad`;
- no raw reader answers stored;
- answer strings are represented only by hashes and aggregate metrics.

Reader60 learned slice:

| retriever | EM | F1 | contains |
|---|---:|---:|---:|
| BM25 | 0.1167 | 0.2428 | 0.1500 |
| RAG-to-Memory PPR | 0.1000 | 0.1472 | 0.1000 |
| fixed meta | 0.1500 | 0.2945 | 0.1833 |
| learned meta | 0.1500 | 0.2945 | 0.1833 |

Reader60 learned meta vs BM25: EM `+0.0333`, F1 `+0.0517`, contains `+0.0333`.

Reader300 learned slice:

| retriever | EM | F1 | contains |
|---|---:|---:|---:|
| BM25 | 0.1167 | 0.1883 | 0.1467 |
| RAG-to-Memory PPR | 0.0867 | 0.1512 | 0.1033 |
| fixed meta | 0.1267 | 0.2039 | 0.1633 |
| learned meta | 0.1267 | 0.2039 | 0.1633 |

Reader300 learned meta vs BM25: EM `+0.0100`, F1 `+0.0156`, contains `+0.0166`.

External LLM reader was not rerun in this update because the relevant API environment variables were not set in the shell. The code and repro pack list env var names only; no secret values are written to code or artifacts.

## Interpretation

The meta-cognitive advantage is not that every QA row receives a longer answer or more abstract morphism prompt. The gain comes from a bounded selector that detects when a structural policy should fire, applies it only on scoped rows, and abstains elsewhere. This explains why QA gains are smaller than the cross-domain morphism benchmark, but still positive and measurable:

- deterministic fixed meta improves support-chain retrieval over BM25 and PPR;
- learned meta is conservative on small heldout slices and adds a small full3000 gain over fixed meta;
- local reader EM/F1 improves when fed meta-retrieved evidence;
- MuSiQue remains harder and mostly neutral, which should be reported as a boundary.

## Reproduction

```bash
python3 -m assumption_os.meta_qa_evolution --root . \
  --eval-id meta_qa_evolution_learned_heldout300_20260608 \
  --samples-per-dataset 100 \
  --workers 6 \
  --bootstrap-iterations 400 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_learned_heldout300_20260608.json'

python3 -m assumption_os.meta_qa_evolution --root . \
  --eval-id meta_qa_evolution_learned_full3000_20260608 \
  --samples-per-dataset 1000 \
  --workers 6 \
  --bootstrap-iterations 400 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_learned_full3000_20260608.json'

python3 -m assumption_os.meta_qa_evolution --root . \
  --eval-id meta_qa_evolution_reader60_learned_20260608 \
  --samples-per-dataset 20 \
  --workers 3 \
  --bootstrap-iterations 200 \
  --run-extractive-reader \
  --reader-samples-per-dataset 20 \
  --reader-slice dataset_balanced \
  --out 'phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_reader60_learned_20260608.json'

python3 -m assumption_os.meta_qa_evolution --root . \
  --eval-id meta_qa_evolution_reader300_learned_20260608 \
  --samples-per-dataset 100 \
  --workers 6 \
  --bootstrap-iterations 200 \
  --run-extractive-reader \
  --reader-samples-per-dataset 100 \
  --reader-slice dataset_balanced \
  --out 'phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_reader300_learned_20260608.json'

python3 -m unittest tests.test_assumption_os.AssumptionOSTest.test_meta_qa_evolution_retains_only_beneficial_retrieval_hypotheses
```
