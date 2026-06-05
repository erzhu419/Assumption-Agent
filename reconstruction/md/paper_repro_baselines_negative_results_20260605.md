# Paper Baselines, Negative Results, and Repro Pack - 2026-06-05

## What Was Added

这次补齐四个审稿级缺口：

1. matched frozen toggle-off baselines
2. real full-text RAG/vector retrieval baselines
3. negative-results and boundary-condition artifact
4. reproducibility package manifest

Artifacts:

- `phase four/assumption_graph/paper_readiness_20260604/paper_baseline_hardening_20260605.json`
- `phase four/assumption_graph/paper_readiness_20260604/paper_retrieval_baselines_20260605.json`
- `phase four/assumption_graph/paper_readiness_20260604/paper_negative_results_20260605.json`
- `phase four/assumption_graph/paper_readiness_20260604/paper_repro_pack_20260605.json`

## Matched Toggle-Off Baselines

这些 baseline 不再只是 loose historical proxy，而是同一批 100 个 `problem_id` 的 tracked summary-level toggle audit。

| Baseline | Toggle utility vs raw | Final minus toggle | Toggle utility vs placebo | Final minus toggle |
| --- | ---: | ---: | ---: | ---: |
| no_world_model_trace_policy | 0.545 | +0.080 | 0.550 | +0.155 |
| no_recursive_runner_one_shot | 0.545 | +0.080 | 0.550 | +0.155 |
| no_novelty_gate_incremental_addition | 0.530 | +0.095 | 0.585 | +0.120 |
| no_final_margin_retention_gate | 0.680 | -0.055 | 0.590 | +0.115 |

Interpretation:

- The three key toggles all lose to the final frozen pipeline on both raw and placebo comparisons.
- `no_final_margin_retention_gate` is retained as a tradeoff: it scores higher vs raw but lower vs placebo, so the paper should not oversell margin gating as universally monotonic.

## Full-Text RAG / Vector Retrieval Baselines

The retrieval baseline artifact uses actual retrieval scorers over full candidate text, not prompt length:

| Retriever | Top-1 hit rate |
| --- | ---: |
| ordinary_rag_bm25_full_text | 0.300 |
| full_text_tfidf_vector_retrieval | 0.000 |
| sentence_transformer_embedding | 0.000 |
| structural_morphism | 1.000 |

Morphism margin over best full-text retrieval baseline: `0.700`.

This strengthens the morphism claim: KG/embedding proxy was not enough by itself, so now the benchmark also includes BM25-style ordinary RAG and sentence-transformer retrieval over the same cases.

## Negative Results and Boundaries

Negative-results artifact now keeps these boundaries explicit:

Domain-level weak spots:

- `mathematics`: utility vs raw `0.5333`, flagged `weak_vs_raw`
- `science`: utility vs raw `0.4667`, flagged `weak_vs_raw`

Historical repair failures:

- `bottleneck_first_margin_failure`: pass `false`, utility vs raw `0.700`, utility vs placebo `0.300`
- `signal_first_repair_failure`: pass `false`, utility vs raw `0.125`, utility vs placebo `0.250`
- `natural_one_shot_failure`: pass `false`, utility vs raw `0.545`, utility vs placebo `0.550`

Formal/morphism boundary:

- The allowed claim remains `category-inspired bounded structural morphism layer`.
- The artifact explicitly forbids claiming a complete theorem prover, exact Blackwell/Fisher engine, semantic equivalence proof, or causal-transfer guarantee.
- The policy is: require preserved invariants, negative controls, and downstream behavior gain before promotion.

## Repro Pack

The reproducibility package includes:

- 8 exact commands
- frozen config summary
- redacted data card
- API/env var names only, no secret values
- 18 artifact/code manifest rows with SHA256 hashes
- one-click main table command

Redaction policy:

- No API keys are written.
- Raw model answers and forensic judge raw text are excluded.
- Summary/audit JSON and Markdown are retained.

## Validation

Targeted validation passed:

`python3 -m unittest tests.test_assumption_os.AssumptionOSTest.test_paper_main_experiment_freezes_problem_level_stats_and_baselines tests.test_assumption_os.AssumptionOSTest.test_paper_baseline_hardening_uses_matched_frozen_toggle_offs tests.test_assumption_os.AssumptionOSTest.test_paper_retrieval_baselines_include_real_full_text_rag tests.test_assumption_os.AssumptionOSTest.test_paper_negative_results_records_boundaries_and_failures tests.test_assumption_os.AssumptionOSTest.test_paper_repro_pack_records_commands_hashes_and_env_names_only`

Result: `5 tests OK`.
