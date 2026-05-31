# Phase2 v20 Software-Engineering Reranker Regression, GPT-5.5, n=5

Date: 2026-05-31

## Setup

- Sample: `phase two/analysis/cache/sample_se_regression_5.json`.
- Rows: the five `software_engineering` cases from sample 21-50 where the learned graph variant lost to same-model baseline in both judge directions.
- Baseline: `phase2_v20_gpt55`.
- Intervention: `phase2_v20_ag_se_rerank_gpt55`.
- Intervention used `--assumption-graph-skip-domains ""`, so graph context was injected for software engineering.
- Solver/Judge model: `gpt-5.5`.
- API credentials were passed only through shell environment variables.

## Change Tested

Implemented `assumption_os/retrieval_policy.py`, a domain-aware reranker for Phase2 graph retrieval.

For `software_engineering`, it routes problems into specific subtypes:

- `release_quality`
- `adapter_discovery`
- `platform_migration`
- `online_offline_gap`
- `performance_regression`
- `legacy_increment`
- `tech_choice`
- `startup_gtm`
- `safety_latency`

The reranker adds route-specific method boosts and injects domain execution checks, such as acceptance metrics, Go/No-Go thresholds, rollout/rollback plans, adapter contracts, and MVP validation criteria.

## Offline Retrieval Audit

On all 10 software-engineering rows in sample 21-50:

| Retrieval mode | Gold strategy hit@8 | Recall | MRR |
|---|---:|---:|---:|
| generic primary retrieval | 9 / 10 | 90.0% | 0.482 |
| software reranker | 10 / 10 | 100.0% | 0.950 |

The reranker fixed the main retrieval miss (`software_engineering_0355`) and moved most gold strategies to rank 1 or 2.

## GPT-5.5 Regression Result

| Direction | AG reranker wins | Baseline wins | Ties | AG win rate |
|---|---:|---:|---:|---:|
| `phase2_v20_ag_se_rerank_gpt55` vs `phase2_v20_gpt55` | 2 | 3 | 0 | 40.0% |
| reverse naming/order check | 3 | 2 | 0 | 60.0% |
| combined | 5 | 5 | 0 | 50.0% |

Case-level summary:

| PID | Outcome Across Two Directions | Notes |
|---|---|---|
| `software_engineering_0355` | baseline 2-0 | still needs richer bug-by-bug execution detail |
| `software_engineering_0225` | reranker 2-0 | startup/wedge route helped |
| `software_engineering_0186` | baseline 2-0 | adapter route found methods, but answer still too governance-heavy |
| `software_engineering_0364` | split 1-1 | platform migration route improved but unstable |
| `software_engineering_0365` | reranker 2-0 | platform migration route helped |

## Interpretation

The reranker eliminated stable negative transfer on this targeted set, but it did not create a stable positive result. It should remain behind the existing software-engineering skip gate until the prompt can force more concrete execution detail for adapter discovery and release-quality triage.

The next software-engineering fix should be prompt-side, not only retrieval-side:

- require per-item triage tables for QA/release problems,
- require adapter contracts and concrete discovery steps for undocumented API problems,
- require explicit spike evidence, no-go thresholds, and phased migration plans for platform migration problems.

## Writeback

The reranker regression results were replayed with policy-aware attribution:

```bash
python3 -m assumption_os.record_phase2_eval \
  --graph-dir "phase four/assumption_graph" \
  --sample "phase two/analysis/cache/sample_se_regression_5.json" \
  --meta "phase two/analysis/cache/answers/phase2_v20_ag_se_rerank_gpt55_meta.json" \
  --judgments \
    "phase two/analysis/cache/judgments/phase2_v20_ag_se_rerank_gpt55_vs_phase2_v20_gpt55.json" \
    "phase two/analysis/cache/judgments/phase2_v20_gpt55_vs_phase2_v20_ag_se_rerank_gpt55.json" \
  --intervention phase2_v20_ag_se_rerank_gpt55 \
  --baseline phase2_v20_gpt55 \
  --eval-id phase2_v20_ag_se_rerank_gpt55_vs_gpt55_regression5 \
  --top-k 8 \
  --policy-rerank \
  --assumption-graph-skip-domains "" \
  --summary-out "phase four/assumption_graph/eval_phase2_v20_gpt55_se_rerank5_writeback.json"
```

Writeback summary:

| Item | Count |
|---|---:|
| processed trials | 10 |
| AG wins | 5 |
| baseline wins | 5 |
| no-residual successes | 5 |
| optimization residuals | 5 |

Current graph after all writebacks:

| Artifact | Rows |
|---|---:|
| `nodes.jsonl` | 391 |
| `edges.jsonl` | 448 |
| `evidence.jsonl` | 701 |
| `trials.jsonl` | 82 |

## Cached Artifacts

- `phase two/analysis/cache/sample_se_regression_5.json`
- `phase two/analysis/cache/answers/phase2_v20_ag_se_rerank_gpt55_answers.json`
- `phase two/analysis/cache/answers/phase2_v20_ag_se_rerank_gpt55_meta.json`
- `phase two/analysis/cache/judgments/phase2_v20_ag_se_rerank_gpt55_vs_phase2_v20_gpt55.json`
- `phase two/analysis/cache/judgments/phase2_v20_gpt55_vs_phase2_v20_ag_se_rerank_gpt55.json`
- `phase four/assumption_graph/eval_phase2_v20_gpt55_se_rerank5_writeback.json`
