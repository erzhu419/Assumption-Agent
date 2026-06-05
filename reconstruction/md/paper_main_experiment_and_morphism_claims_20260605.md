# Paper Main Experiment and Morphism Claim Audit - 2026-06-05

## Why This Exists

这次补的是论文口径里的两块硬证据：

1. 一条干净的 frozen 主实验线，避免论文证据看起来像多个 artifact 事后拼装。
2. 范畴论/morphism 表述收紧，明确这里只能主张 `category-inspired bounded structural morphism layer`，不能主张完整范畴论定理证明器。

对应 artifact：

- `phase four/assumption_graph/paper_readiness_20260604/paper_main_experiment_20260605.json`
- `phase four/assumption_graph/paper_readiness_20260604/morphism_claim_bundle_20260605.json`

## Frozen Main Experiment

主实验线已经固定为：

`tasks -> hypothesis_generation -> novelty_integration -> ablation_controls -> recursive_resume -> gated_retention -> next_generation`

最终 frozen run：

- source: `phase four/assumption_graph/structural_live_ablation_20260603/structural_live_all_repairs_margin100_v2_gpt54mini_gpt55_20260604_summary.json`
- forensic: `phase four/assumption_graph/structural_live_ablation_20260603/structural_live_all_repairs_margin100_v2_gpt54mini_gpt55_20260604_forensic.jsonl`
- judge rows are collapsed to one outcome per `problem_id` per pair.
- no prompt/answer/raw judge text is stored in the paper artifact.

Problem-level main table:

| Pair | N | Win/Loss/Tie | Utility | 95% bootstrap CI | Exact sign-test p |
| --- | ---: | --- | ---: | --- | ---: |
| structural vs raw LLM/base | 100 | 59 / 34 / 7 | 0.625 | [0.530, 0.715] | 0.0124006 |
| structural vs no-morphism placebo | 100 | 68 / 27 / 5 | 0.705 | [0.610, 0.790] | 0.00003114 |

统计口径：

- unit of analysis: `problem_id`，不是 raw judge row。
- utility: win = 1, tie = 0.5, loss = 0。
- bootstrap: 2000 resamples, seed 20260605。
- paired test: exact two-sided sign test over non-tie outcomes。
- domain breakdown 已在 artifact 中按 business / daily_life / engineering / mathematics / science / software_engineering 记录。
- seed/run variance 只作为 diagnostic，不作为独立问题数：13 条大样本 structural live runs，base utility mean 0.5878, stdev 0.0531；placebo utility mean 0.5970, stdev 0.0592。

## Baselines Covered

主 artifact 现在覆盖这些 baseline family：

- `raw_llm_baseline`: frozen 100 题 pairwise control。
- `long_prompt_placebo_no_morphism`: frozen 100 题 no-morphism placebo control。
- `ordinary_kg_triple_retrieval`: morphism benchmark 中的 KG triple retrieval。
- `embedding_retrieval`: morphism benchmark 中的 lexical embedding-style retrieval。
- `ordinary_rag_bm25_full_text`: full-text BM25 RAG-style retrieval。
- `full_text_tfidf_vector_retrieval`: full-text TF-IDF vector retrieval。
- `sentence_transformer_embedding`: sentence-transformer embedding retrieval。
- `no_morphism_structural_placebo`: frozen run 内的无 morphism 对照。
- `no_world_model_trace_policy`: same 100 problem_ids 的 matched toggle-off summary。
- `no_recursive_runner_one_shot`: same 100 problem_ids 的 matched toggle-off summary。
- `no_novelty_gate_incremental_addition`: same 100 problem_ids 下 final novelty/integration gate 前的 incremental addition toggle。

这能支持论文里最关键的 claim：不是简单“更长 prompt”或者普通 RAG/KG/embedding retrieval 带来的收益。

## Morphism Claim Boundary

推荐短表述：

`category-inspired bounded structural morphism layer`

推荐论文表述：

`We implement a category-inspired bounded structural morphism layer: finite typed objects, morphism/operator labels, composition cues, preserved invariants, negative controls, and transfer gates for cross-domain analogy retrieval and task routing.`

明确不能写成：

- complete category-theory theorem prover
- general categorical reasoning engine
- exact Blackwell order engine
- exact Fisher information geometry engine
- morphism proves semantic equivalence
- morphism guarantees causal transfer

## Morphism Evidence

Cross-domain retrieval benchmark:

| Scorer | Top-1 hit rate |
| --- | ---: |
| morphism | 1.0 |
| KG triple | 0.1 |
| embedding proxy | 0.1 |

Other evidence:

- morphism margin over best baseline: 0.9
- nonlexical structural success rate: 0.8
- KG/embedding miss but morphism succeeds: 8 / 10 cases
- domain pairs include chemistry -> electromagnetism, world_model -> geophysics, control -> macro_economics, software_engineering -> biochemistry, mathematics -> aerospace_control。

Invariant and anti-hallucination evidence:

- expected candidate count: 10
- preserved invariant count: 20
- expected broken invariant count: 0
- formal negative-control application count: 288

Downstream effect:

- downstream transfer AUC: 0.9833
- answer-quality mean delta: 0.9125
- guided win rate: 1.0
- answer-quality probe count: 9

## Current Status

Both new gates pass:

- `paper_main_experiment_20260605.pass = true`
- `morphism_claim_bundle_20260605.pass = true`

This does not mean the paper is finished. It means the two missing paper-facing evidence structures are now explicit, reproducible, and bounded in claim scope.
