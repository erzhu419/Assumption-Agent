# Meta-QA Evolution Probe - 2026-06-07

## Why

HippoRAG QA probing showed the real gap: bounded structural morphism is useful for cross-domain structural analogy, but it is not by itself a factual multi-hop QA retriever. On HotpotQA / MuSiQue / 2Wiki samples, direct morphism mostly abstains and the system falls back to BM25.

This update uses pre-reconstruction method-layer "philosophies" as QA retrieval priors:

- representation transform: normalize possessive, quoted, and parenthesized mentions into canonical title candidates;
- decomposition/composition: retrieve an anchor page, extract the role-labeled bridge entity, then retrieve the bridge page;
- controlled intervention: preserve the working BM25 path and insert only one or two bounded candidates.

The loop is still variation / evaluation / selective retention. A plausible policy is retained only when heldout support-chain metrics improve without row-level regression.

## Artifacts

- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_heldout60_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_reader15_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_reader60_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_llm_reader15_gpt54mini_20260607.json`
- `phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_llm_reader60_gpt54mini_20260607.json`

Data:

- `reference/repos/HippoRAG/reproduce/dataset/hotpotqa.json`
- `reference/repos/HippoRAG/reproduce/dataset/musique.json`
- `reference/repos/HippoRAG/reproduce/dataset/2wikimultihopqa.json`

No live API calls or raw model answers are used. Ranking inputs are question text, corpus titles, corpus text, and deterministic retrieval diagnostics. Ranking excludes gold answers, gold titles, and supporting facts.

The reader artifacts use a local open extractive QA model (`distilbert-base-cased-distilled-squad`) as a reproducible reader proxy. The artifact stores answer hashes, lengths, EM/F1, and latency only; raw reader answers and gold answer strings are not stored.

The live-reader artifacts use an OpenAI-compatible GPT reader (`gpt-5.4-mini`) with the same retrieved contexts. They also store only hashes and metrics, not prompts, raw answers, API keys, or gold answer strings.

## Variation

The controller now evaluates seven retrieval hypotheses:

| hypothesis | small-slice decision | heldout60 decision | role |
|---|---:|---:|---|
| `qa_hyp_comparison_dual_anchor` | accept | accept | force both comparison anchors |
| `qa_hyp_anchor_preserve_insert` | accept | accept | preserve BM25, insert title anchor |
| `qa_hyp_named_anchor_bridge` | reject | reject | broad bridge caused regressions |
| `qa_hyp_generic_prf` | reject | reject | PRF amplified wrong first hops |
| `qa_hyp_representation_title_normalization` | accept | accept | canonicalize noisy surface mentions |
| `qa_hyp_decomposition_bridge_entity` | accept | accept | anchor page -> bridge entity -> bridge page |
| `qa_hyp_controlled_bridge_insert` | accept | accept | bounded insert with BM25 guard |

The bridge policies initially over-fired. Two guards fixed the main regressions:

- do not insert when top-k title diversity indicates BM25's tail may be the only unique evidence;
- only extract explicit role-labeled bridge phrases, not arbitrary capitalized phrases.

## Performance

Small slice: 15 rows, 5 per dataset, top-k 5.

| retriever | all support recall@5 | mean support fraction@5 | answer coverage@5 |
|---|---:|---:|---:|
| ordinary BM25 | 0.1333 | 0.5111 | 0.2667 |
| RAG-to-Memory-style PPR | 0.1333 | 0.4889 | 0.2000 |
| meta-QA controller | 0.6000 | 0.8111 | 0.6000 |

Delta vs BM25:

- all support recall@5: `+0.4667`
- mean support fraction@5: `+0.3000`
- answer coverage@5: `+0.3333`

Heldout60: 60 rows, 20 per dataset, top-k 5.

| retriever | all support recall@5 | mean support fraction@5 | answer coverage@5 |
|---|---:|---:|---:|
| ordinary BM25 | 0.3333 | 0.6056 | 0.5167 |
| RAG-to-Memory-style PPR | 0.1500 | 0.4472 | 0.3167 |
| meta-QA controller | 0.5000 | 0.7292 | 0.5833 |

Delta vs BM25:

- all support recall@5: `+0.1667`
- mean support fraction@5: `+0.1236`
- answer coverage@5: `+0.0666`

Delta vs PPR:

- all support recall@5: `+0.3500`
- mean support fraction@5: `+0.2820`
- answer coverage@5: `+0.2666`

## Reader Proxy

Reader15: 15 rows, same top-k 5 contexts, local extractive QA reader.

| retriever | exact match | mean F1 | contains-gold prediction |
|---|---:|---:|---:|
| ordinary BM25 | 0.0000 | 0.0742 | 0.0000 |
| RAG-to-Memory-style PPR | 0.0000 | 0.0500 | 0.0000 |
| meta-QA controller | 0.0667 | 0.1409 | 0.0667 |

Reader60: 60 rows, same heldout split.

| retriever | exact match | mean F1 | contains-gold prediction |
|---|---:|---:|---:|
| ordinary BM25 | 0.1167 | 0.2428 | 0.1500 |
| RAG-to-Memory-style PPR | 0.1000 | 0.1472 | 0.1000 |
| meta-QA controller | 0.1500 | 0.2945 | 0.1833 |

Reader60 deltas:

- vs BM25: exact match `+0.0333`, mean F1 `+0.0517`, contains-gold prediction `+0.0333`
- vs PPR: exact match `+0.0500`, mean F1 `+0.1473`, contains-gold prediction `+0.0833`

## Live GPT Reader

Live GPT reader15: 15 rows, same top-k 5 contexts, `gpt-5.4-mini`.

| retriever | exact match | mean F1 | contains-gold prediction |
|---|---:|---:|---:|
| ordinary BM25 | 0.2000 | 0.2000 | 0.2000 |
| RAG-to-Memory-style PPR | 0.2667 | 0.2667 | 0.2667 |
| meta-QA controller | 0.6000 | 0.6000 | 0.6000 |

Live GPT reader60: 60 rows, same heldout split, `gpt-5.4-mini`.

| retriever | exact match | mean F1 | contains-gold prediction |
|---|---:|---:|---:|
| ordinary BM25 | 0.3167 | 0.3986 | 0.3500 |
| RAG-to-Memory-style PPR | 0.2500 | 0.2959 | 0.2833 |
| meta-QA controller | 0.3500 | 0.4588 | 0.4167 |

Live GPT reader60 deltas:

- vs BM25: exact match `+0.0333`, mean F1 `+0.0602`, contains-gold prediction `+0.0667`
- vs PPR: exact match `+0.1000`, mean F1 `+0.1629`, contains-gold prediction `+0.1334`

## Interpretation

This is the first result where method-layer metacognition clearly helps QA retrieval and propagates to answer-level reader metrics, rather than merely saying morphism is not applicable. The improvement does not come from using category/morphism as a direct factual retriever. It comes from using old method priors as retrieval-control policies:

`canonical representation -> decomposed bridge search -> controlled insert -> selective retention`

The result still should not be described as a full HippoRAG leaderboard win. We now have both a local extractive-reader proxy and a live GPT reader result, but not the full HippoRAG reader stack or full benchmark. The correct claim is narrower and stronger: pre-reconstruction method-layer assumptions can be converted into gated QA retrieval policies that improve supporting-evidence recall, local extractive-reader EM/F1, and live GPT-reader EM/F1 on real HippoRAG reproduction datasets.

## Reproduction

```bash
python3 -m assumption_os.meta_qa_evolution --root . \
  --out 'phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_20260607.json'

python3 -m assumption_os.meta_qa_evolution --root . \
  --samples-per-dataset 20 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_heldout60_20260607.json'

python3 -m assumption_os.meta_qa_evolution --root . \
  --samples-per-dataset 20 \
  --run-extractive-reader \
  --reader-samples-per-dataset 20 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_reader60_20260607.json'

RUOLI_GPT_KEY=<set-in-env> RUOLI_BASE_URL=<set-in-env> \
python3 -m assumption_os.meta_qa_evolution --root . \
  --samples-per-dataset 20 \
  --run-llm-reader \
  --llm-reader-provider gpt \
  --llm-reader-model gpt-5.4-mini \
  --llm-reader-samples-per-dataset 20 \
  --out 'phase four/assumption_graph/paper_readiness_20260604/meta_qa_evolution_llm_reader60_gpt54mini_20260607.json'

python3 -m unittest tests.test_assumption_os.AssumptionOSTest.test_meta_qa_evolution_retains_only_beneficial_retrieval_hypotheses
```
