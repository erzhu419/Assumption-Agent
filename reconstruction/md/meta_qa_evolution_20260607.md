# Meta-QA Evolution Probe - 2026-06-07

## Why

HippoRAG QA probe showed a real gap: bounded structural morphism improves cross-domain structural analogy retrieval, but it does not directly act as a factual multi-hop QA retriever. On HotpotQA / MuSiQue / 2Wiki samples, the safe policy falls back to BM25, so QA scores barely change.

This probe tests the missing solve-time metacognition adapter:

`QA failure -> multiple retrieval hypotheses -> evidence evaluation -> selective retention -> guarded QA retrieval policy`

It does not use live API calls or reader answers. It evaluates retrieval evidence with supporting-fact titles and answer-string coverage from the local HippoRAG reproduction datasets.

## Setup

Artifact:

- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_20260607.json`

Data:

- `reference/repos/HippoRAG/reproduce/dataset/hotpotqa.json`
- `reference/repos/HippoRAG/reproduce/dataset/musique.json`
- `reference/repos/HippoRAG/reproduce/dataset/2wikimultihopqa.json`

Config:

- 5 sampled rows per dataset, 15 total
- retrieval top-k = 5
- no raw model answers stored
- ranking inputs: question, corpus titles, corpus text, retrieval residual type
- ranking excludes: gold answers, gold titles, supporting facts

## Variation

The controller generated four retrieval hypotheses:

| hypothesis | decision | activated rows | reason |
|---|---:|---:|---|
| `qa_hyp_comparison_dual_anchor` | accept | 1 | fixed a binary comparison row by forcing both named entities into evidence retrieval |
| `qa_hyp_anchor_preserve_insert` | reject | 15 | aggregate evidence improved, but one row regressed, so it is not retained without a narrower gate |
| `qa_hyp_named_anchor_bridge` | reject | 13 | caused support-chain regression on at least one row |
| `qa_hyp_generic_prf` | reject | 15 | had aggregate upside but also a support-chain regression |

This is the important behavior: the system does not keep every plausible self-generated idea. It keeps a narrow no-regression policy and rejects broader policies that look intuitively reasonable but damage evidence coverage.

## Performance

| retriever | all support recall@5 | mean support fraction@5 | answer coverage@5 |
|---|---:|---:|---:|
| ordinary BM25 | 0.1333 | 0.5111 | 0.2667 |
| RAG-to-Memory-style PPR | 0.1333 | 0.4889 | 0.2000 |
| meta-QA controller | 0.2000 | 0.5444 | 0.2667 |

Delta vs BM25:

- all support recall@5: `+0.0667`
- mean support fraction@5: `+0.0333`
- answer coverage@5: `+0.0000`

Delta vs PPR:

- all support recall@5: `+0.0667`
- mean support fraction@5: `+0.0555`
- answer coverage@5: `+0.0667`

## Interpretation

This is not enough to claim superiority over HippoRAG on full QA. It is enough to show where the recursive self-evolution capability appears in QA:

1. The system diagnoses that direct morphism is not applicable to factual QA.
2. It generates multiple retrieval-policy hypotheses from incomplete-support residuals.
3. It evaluates each candidate against evidence-chain metrics.
4. It selectively retains only a narrow policy with measured benefit and no answer-coverage regression.
5. It falls back to BM25 everywhere else.

So the current result is a small solve-time metacognition gain, not a broad QA breakthrough. The next hard target is to scale this from 15 rows and one accepted narrow policy to a heldout multi-generation QA evolution line.

## Reproduction

```bash
python3 -m assumption_os.meta_qa_evolution --root . \
  --out 'phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_20260607.json'

python3 -m unittest tests.test_assumption_os.AssumptionOSTest.test_meta_qa_evolution_retains_only_beneficial_retrieval_hypotheses
```
