# Phase2 v20 Learned-Graph Heldout Evaluation, GPT-5.5, sample 21-50

Date: 2026-05-31

## Setup

- Heldout sample: `phase two/analysis/cache/sample_21_50.json`, copied from `sample_100.json[20:50]`.
- Sample size: 30 problems.
- Domain mix: engineering 5, mathematics 4, daily_life 4, business 2, software_engineering 10, science 5.
- Non-bypass problems: 21. Math/science 9 rows use the v20 hygiene bypass and do not receive graph context.
- Solver/Judge model: `gpt-5.5` through OpenAI-compatible proxy.
- Baseline: `phase2_v20_gpt55`, same v20 scaffold, no graph.
- Intervention: `phase2_v20_ag_learned_gpt55`, using the graph after the n=20 writeback.

API credentials were passed only through shell environment variables.

## Commands

```bash
python3 "phase one/scripts/validation/phase2_v20_framework.py" \
  --variant phase2_v20_gpt55 \
  --sample sample_21_50.json \
  --n 30

python3 "phase one/scripts/validation/phase2_v20_framework.py" \
  --variant phase2_v20_ag_learned_gpt55 \
  --sample sample_21_50.json \
  --n 30 \
  --assumption-graph "phase four/assumption_graph"

python3 "phase one/scripts/validation/cached_framework.py" \
  --sample sample_21_50.json \
  --n 30 \
  --judge phase2_v20_ag_learned_gpt55 phase2_v20_gpt55

python3 "phase one/scripts/validation/cached_framework.py" \
  --sample sample_21_50.json \
  --n 30 \
  --judge phase2_v20_gpt55 phase2_v20_ag_learned_gpt55
```

## Results

| Direction | AG wins | Baseline wins | Ties | AG win rate excl. ties |
|---|---:|---:|---:|---:|
| `phase2_v20_ag_learned_gpt55` vs `phase2_v20_gpt55` | 9 | 14 | 7 | 39.1% |
| reverse naming/order check | 12 | 11 | 7 | 52.2% |
| combined | 21 | 25 | 14 | 45.7% |

Non-bypass only:

| Subset | AG wins | Baseline wins | Ties | AG win rate excl. ties |
|---|---:|---:|---:|---:|
| non-bypass 21 problems, combined directions | 16 | 14 | 12 | 53.3% |

By-domain combined result:

| Domain | AG | Baseline | Tie | Notes |
|---|---:|---:|---:|---|
| engineering | 7 | 0 | 3 | strong positive |
| daily_life | 5 | 1 | 2 | positive |
| business | 1 | 1 | 2 | neutral |
| software_engineering | 3 | 12 | 5 | strong negative |
| science, bypass | 4 | 4 | 2 | not graph-attributable |
| mathematics, bypass | 1 | 7 | 0 | not graph-attributable |

## Retrieval Audit

On the 21 non-bypass heldout problems:

| Visible top-k | Gold strategy hit | Recall | MRR |
|---:|---:|---:|---:|
| 5 | 12 / 21 | 57.1% | 0.355 |
| 8 | 15 / 21 | 71.4% | 0.373 |
| 10 | 16 / 21 | 76.2% | 0.379 |

This is materially weaker than the post-writeback n=20 audit, where top8 recall was 86.7%. The heldout result is therefore not a clean generalization win. The graph helps engineering and daily_life, but software_engineering routing is too generic and often misses the exact risk/prior family.

## Loss Pattern

Repeated two-direction baseline wins were concentrated in:

- `software_engineering_0355`: QA game-release prioritization; graph missed `S24/S12/S11`.
- `software_engineering_0225`: medical AI startup commercialization; graph answer was less wedge/metric-specific.
- `software_engineering_0186`: undocumented local API adapter; graph over-weighted governance versus adapter/MVP details.
- `software_engineering_0364` and `software_engineering_0365`: platform migration decisions; graph answer was less concrete on migration metrics and staged execution.

Math losses are excluded from graph attribution because v20 did not inject graph context for math/science.

## Graph Writeback

Heldout judged outcomes were replayed into the graph:

```bash
python3 -m assumption_os.record_phase2_eval \
  --graph-dir "phase four/assumption_graph" \
  --sample "phase two/analysis/cache/sample_21_50.json" \
  --meta "phase two/analysis/cache/answers/phase2_v20_ag_learned_gpt55_meta.json" \
  --judgments \
    "phase two/analysis/cache/judgments/phase2_v20_ag_learned_gpt55_vs_phase2_v20_gpt55.json" \
    "phase two/analysis/cache/judgments/phase2_v20_gpt55_vs_phase2_v20_ag_learned_gpt55.json" \
  --intervention phase2_v20_ag_learned_gpt55 \
  --baseline phase2_v20_gpt55 \
  --eval-id phase2_v20_ag_learned_gpt55_vs_gpt55_21_50 \
  --top-k 8 \
  --summary-out "phase four/assumption_graph/eval_phase2_v20_gpt55_21_50_writeback.json"
```

Writeback summary:

| Item | Count |
|---|---:|
| processed trials | 42 |
| skipped missing graph meta | 18 |
| AG wins | 16 |
| baseline wins | 14 |
| ties | 12 |
| no-residual successes | 16 |
| optimization residuals | 11 |
| memory-defect residuals | 3 |

Current graph after both writebacks:

| Artifact | Rows |
|---|---:|
| `nodes.jsonl` | 386 |
| `edges.jsonl` | 408 |
| `evidence.jsonl` | 621 |
| `trials.jsonl` | 72 |

## Interpretation

The n=20 result was too optimistic. On 21-50 heldout, learned graph is only marginally positive on non-bypass rows and negative overall because math bypass rows and software_engineering dominate the loss mass.

Next change should not be another blind n=100 run. The practical fix is a routing gate:

- inject graph only when retrieval has sufficient confidence or gold-like method specificity,
- add a software_engineering-specific retrieval/reranking policy that favors concrete execution metrics, rollout criteria, rollback paths, and adapter/MVP detail,
- keep math/science bypass separate until graph context is explicitly designed for those domains.

Implemented immediate conservative gate after this evaluation:

- `phase2_v20_framework.py` now defaults `--assumption-graph-skip-domains software_engineering`.
- Pass `--assumption-graph-skip-domains ""` to restore full graph injection.
- This is a stop-loss gate, not a final solution. It prevents the observed software_engineering negative transfer while preserving graph injection for engineering, daily_life, and business.

Follow-up reranker result:

- Added `assumption_os/retrieval_policy.py` with software-engineering route-specific reranking.
- Offline SE hit@8 improved from 9/10 to 10/10; MRR improved from 0.482 to 0.950.
- A targeted GPT-5.5 rerun on the five stable SE losses improved from stable negative transfer to neutral: AG 5 / baseline 5 / tie 0 across two judge directions.
- Because the rerun is neutral rather than positive, the default software-engineering skip gate remains in place.

See `phase four/assumption_graph/eval_phase2_v20_gpt55_se_rerank5.md`.

## Cached Artifacts

- `phase two/analysis/cache/sample_21_50.json`
- `phase two/analysis/cache/answers/phase2_v20_gpt55_answers.json`
- `phase two/analysis/cache/answers/phase2_v20_ag_learned_gpt55_answers.json`
- `phase two/analysis/cache/answers/phase2_v20_ag_learned_gpt55_meta.json`
- `phase two/analysis/cache/judgments/phase2_v20_ag_learned_gpt55_vs_phase2_v20_gpt55.json`
- `phase two/analysis/cache/judgments/phase2_v20_gpt55_vs_phase2_v20_ag_learned_gpt55.json`
- `phase four/assumption_graph/eval_phase2_v20_gpt55_21_50_writeback.json`
