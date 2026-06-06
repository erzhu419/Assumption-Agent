# HippoRAG QA Probe - 2026-06-06

## Why

前面的 RAG-to-Memory baseline comparison 只证明了一件事：在 10 个跨域结构同构 retrieval case 上，bounded structural morphism 明显强于 KG/PPR/embedding 类 baseline。

这不能直接外推到 HippoRAG 2 全论文 QA benchmark。HotpotQA、MuSiQue、2Wiki 主要考察 factual retrieval、多跳实体链路和 reader answer generation；morphism layer 不是事实 QA retriever。

## Setup

数据来自本地 HippoRAG repo：

- `reference/repos/HippoRAG/reproduce/dataset/hotpotqa.json`
- `reference/repos/HippoRAG/reproduce/dataset/musique.json`
- `reference/repos/HippoRAG/reproduce/dataset/2wikimultihopqa.json`

Probe:

- 每个数据集抽 5 题，共 15 题；
- corpus 使用对应 `*_corpus.json`；
- retrieval top-k = 5；
- 比较 `ordinary_bm25`、`rag_to_memory_style_ppr`、`current_safe_policy_bm25_fallback`、`structural_morphism_direct`；
- `structural_morphism_direct` 在 factual QA 上标为不可直接适用；
- 额外用 gpt-5.5 reader 跑每个数据集 2 题，共 6 题 x 2 retrievers = 12 次调用；
- artifact 不存 API key，不存 raw model answer，只存 answer hash、长度、latency、gold-answer hit。

Artifact:

- `phase four/assumption_graph/paper_readiness_20260604/hipporag_qa_probe_20260606.json`

## Retrieval/Coverage Result

15 题 top-5 retrieval/coverage:

| retriever | any gold recall@5 | all gold recall@5 | mean gold fraction@5 | answer coverage@5 |
|---|---:|---:|---:|---:|
| ordinary BM25 | 0.8667 | 0.1333 | 0.5111 | 0.2667 |
| RAG-to-Memory-style PPR | 0.8667 | 0.1333 | 0.4889 | 0.2000 |
| current safe policy fallback | 0.8667 | 0.1333 | 0.5111 | 0.2667 |
| structural morphism direct | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Interpretation:

- PPR-style graph retrieval did not beat BM25 on this small QA slice.
- It was slightly worse on mean supporting-doc fraction and answer coverage.
- Current morphism layer is not directly applicable to factual QA retrieval; the only safe behavior is fallback to ordinary retrieval unless a separate QA routing layer is added.

## Live Reader Result

gpt-5.5 reader, 6 sampled QA rows, 12 calls:

| retriever context | reader n | gold answer hit rate | failed calls |
|---|---:|---:|---:|
| ordinary BM25 | 6 | 0.0000 | 0 |
| RAG-to-Memory-style PPR | 6 | 0.1667 | 0 |

This is too small to claim PPR is better. The main signal is risk: even when a supporting passage appears in top-5, reader answer generation can still fail if the full multi-hop chain is incomplete or the context has insufficient bridge evidence.

## Answer To The Concern

Yes, the earlier effect looked very good partly because the benchmark was aligned to the morphism contribution. On a real QA benchmark slice:

- the current morphism layer does not transfer as a direct QA retriever;
- RAG-to-Memory-style PPR is not dominated by morphism in QA because morphism is mostly abstaining/falling back;
- the system needs a separate QA benchmark line before any paper claim about HotpotQA/MuSiQue/2Wiki performance.

The safe paper claim should stay narrow:

> The morphism layer improves cross-domain structural analogy retrieval and can complement memory retrieval, but it is not a replacement for HippoRAG-style factual/multi-hop QA retrieval.

## Reproduction

No-reader probe:

```bash
python3 -m assumption_os.hipporag_qa_probe --root . --out 'phase four/assumption_graph/paper_readiness_20260604/hipporag_qa_probe_20260606.json'
```

Optional live reader probe:

```bash
RUOLI_GPT_KEY=<set-in-env> RUOLI_BASE_URL=<set-in-env> GPT55_MODEL=gpt-5.5 \
python3 -m assumption_os.hipporag_qa_probe --root . --run-reader --reader-samples-per-dataset 2 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/hipporag_qa_probe_20260606.json'
```
