# Meta-QA Evolution Probe - 2026-06-07

## Purpose

HippoRAG QA probing showed a specific gap: the structural morphism layer is useful for cross-domain structural analogy, but it is not a factual multi-hop QA retriever by itself. This update turns the pre-reconstruction method-layer "philosophies" and HippoRAG-style context edges into solve-time retrieval policies, then evaluates them with variation / evaluation / selective retention.

The current claim is narrow:

- meta-QA does not replace HippoRAG;
- direct morphism still mostly abstains on factual QA;
- generalized assumption edges and gated policy selection improve supporting-evidence retrieval on real HippoRAG reproduction QA files.

## Mechanism

Inputs used for ranking:

- question text;
- corpus titles;
- corpus text;
- deterministic retrieval diagnostics.

Inputs excluded from ranking:

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

The bounded-risk gate now has two levels:

- broad policies can retain at most `min(3, 1%)` harms and must have positive all-support, support-fraction, and answer-coverage utility;
- narrow scoped policies with 100-250 activations can retain at most 4 harms if net support gain is large and answer coverage does not regress.

This is what allows full3000 to retain `comparison_dual_anchor` and `assumption_edge_policy_selector`, while rejecting broad over-routing policies.

## Artifacts

- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_heldout60_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_heldout300_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_heldout600_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_full3000_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_reader15_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_reader60_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_llm_reader15_gpt54mini_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_llm_reader60_gpt54mini_20260607.json`

Datasets:

- `reference/repos/HippoRAG/reproduce/dataset/hotpotqa.json`
- `reference/repos/HippoRAG/reproduce/dataset/musique.json`
- `reference/repos/HippoRAG/reproduce/dataset/2wikimultihopqa.json`

## Retrieval Performance

All runs use top-k 5. Metrics are problem-level, not pairwise-row pseudoreplication. Bootstrap CI resamples problem rows.

| slice | n | BM25 all | meta all | delta all | BM25 fraction | meta fraction | delta fraction | BM25 answer | meta answer | delta answer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 15 | 0.1333 | 0.6667 | +0.5334 | 0.5111 | 0.8444 | +0.3333 | 0.2667 | 0.6000 | +0.3333 |
| heldout60 | 60 | 0.3333 | 0.5500 | +0.2167 | 0.6056 | 0.7681 | +0.1625 | 0.5167 | 0.6167 | +0.1000 |
| heldout300 | 300 | 0.3367 | 0.4533 | +0.1166 | 0.6064 | 0.7125 | +0.1061 | 0.5133 | 0.5767 | +0.0634 |
| heldout600 | 600 | 0.3017 | 0.3700 | +0.0683 | 0.5894 | 0.6332 | +0.0438 | 0.4800 | 0.5200 | +0.0400 |
| full3000 | 3000 | 0.3077 | 0.3627 | +0.0550 | 0.5988 | 0.6345 | +0.0357 | 0.4730 | 0.5030 | +0.0300 |

Full3000 bootstrap CI vs BM25:

- any gold recall delta: `+0.0023`, 95% CI `[+0.0003, +0.0043]`;
- all gold recall delta: `+0.0550`, 95% CI `[+0.0473, +0.0633]`;
- mean support fraction delta: `+0.0357`, 95% CI `[+0.0309, +0.0403]`;
- answer coverage delta: `+0.0300`, 95% CI `[+0.0240, +0.0360]`.

Full3000 vs RAG-to-Memory-style PPR:

- all gold recall delta: `+0.2354`;
- mean support fraction delta: `+0.2511`;
- answer coverage delta: `+0.2073`.

## Full3000 Retention

Accepted:

| hypothesis | decision | activations | harms | harm cap |
|---|---|---:|---:|---:|
| `qa_hyp_comparison_dual_anchor` | `accept_retain_bounded_risk` | 162 | 4 | 4 |
| `qa_hyp_assumption_edge_policy_selector` | `accept_retain_bounded_risk` | 381 | 2 | 3 |

Rejected:

| hypothesis | reason | activations | harms | harm cap |
|---|---|---:|---:|---:|
| `qa_hyp_anchor_preserve_insert` | regression | 2936 | 55 | 3 |
| `qa_hyp_named_anchor_bridge` | regression | 2148 | 210 | 3 |
| `qa_hyp_generic_prf` | regression | 3000 | 344 | 3 |
| `qa_hyp_representation_title_normalization` | regression | 2936 | 10 | 3 |
| `qa_hyp_decomposition_bridge_entity` | regression | 1133 | 56 | 3 |
| `qa_hyp_controlled_bridge_insert` | regression | 1215 | 65 | 3 |

This is the key behavioral point: the controller improves QA retrieval by retaining a small number of useful structural policies, not by making every answer longer or more structured.

## By Dataset

Full3000 meta vs BM25:

| dataset | BM25 all | meta all | BM25 fraction | meta fraction | BM25 answer | meta answer |
|---|---:|---:|---:|---:|---:|---:|
| 2WikiMultiHopQA | 0.2930 | 0.4340 | 0.6122 | 0.7060 | 0.4660 | 0.5520 |
| HotpotQA | 0.4810 | 0.5030 | 0.7175 | 0.7290 | 0.6380 | 0.6400 |
| MuSiQue | 0.1490 | 0.1510 | 0.4667 | 0.4686 | 0.3150 | 0.3170 |

The gain is largest on 2Wiki-style entity relation chains. Hotpot gains are smaller but positive. MuSiQue is essentially neutral, which is acceptable because the retained policies abstain or preserve BM25 when they do not match.

## Reader Proxies

The reader artifacts remain answer-level validation proxies:

- local extractive reader: `distilbert-base-cased-distilled-squad`;
- live GPT reader: `gpt-5.4-mini`;
- raw reader answers, prompts, API keys, and gold answer strings are not stored.

Earlier reader60 result:

- local extractive reader vs BM25: EM `+0.0333`, mean F1 `+0.0517`;
- live GPT reader vs BM25: EM `+0.0333`, mean F1 `+0.0602`, contains-gold prediction `+0.0667`.

The current frozen full3000 line is retrieval-first. A paper-ready answer-level full benchmark should rerun the reader on the same frozen controller if cost/time allow.

## Reproduction

```bash
python3 -m assumption_os.meta_qa_evolution --root . \
  --out 'phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_20260607.json'

python3 -m assumption_os.meta_qa_evolution --root . \
  --samples-per-dataset 20 \
  --workers 3 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_heldout60_20260607.json'

python3 -m assumption_os.meta_qa_evolution --root . \
  --samples-per-dataset 100 \
  --workers 6 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_heldout300_20260607.json'

python3 -m assumption_os.meta_qa_evolution --root . \
  --samples-per-dataset 200 \
  --workers 6 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_heldout600_20260607.json'

python3 -m assumption_os.meta_qa_evolution --root . \
  --samples-per-dataset 1000 \
  --workers 6 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_full3000_20260607.json'

python3 -m unittest tests.test_assumption_os.AssumptionOSTest.test_meta_qa_evolution_retains_only_beneficial_retrieval_hypotheses
```
