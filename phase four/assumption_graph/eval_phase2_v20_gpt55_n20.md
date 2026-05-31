# Phase2 v20 Assumption Graph Evaluation, GPT-5.5, n=20

Date: 2026-05-31

## Setup

- Sample: `phase two/analysis/cache/sample_100.json` first 20 problems, seed 42 order.
- Solver/Judge model: `gpt-5.5` through OpenAI-compatible proxy.
- API credentials were passed only through shell environment variables, not written to files.
- Baseline: `phase2_v20_gpt55`, same v20 scaffold, no Assumption Graph.
- Tested graph variant after retrieval filtering: `phase2_v20_ag_filt_gpt55`.
- Existing historical comparison target: cached `phase2_v20`.

## Commands

```bash
export LLM_PROVIDER=gemini
export GEMINI_BASE_URL=https://ruoli.dev/v1
export GEMINI_MODEL=gpt-5.5
export GEMINI_API_KEY=<runtime-secret>

python3 "phase one/scripts/validation/phase2_v20_framework.py" \
  --variant phase2_v20_gpt55 \
  --n 20

python3 "phase one/scripts/validation/phase2_v20_framework.py" \
  --variant phase2_v20_ag_filt_gpt55 \
  --n 20 \
  --assumption-graph "phase four/assumption_graph"

python3 "phase one/scripts/validation/cached_framework.py" \
  --n 20 \
  --judge phase2_v20_ag_filt_gpt55 phase2_v20_gpt55

python3 "phase one/scripts/validation/cached_framework.py" \
  --n 20 \
  --judge phase2_v20_gpt55 phase2_v20_ag_filt_gpt55
```

## Results

### Historical Cached Baseline

This comparison is useful as a sanity check but confounds scaffold change with model change.

| Comparison | Result |
|---|---:|
| `phase2_v20_ag_gpt55` vs historical `phase2_v20` | 19 / 0 / 1, win rate 100.0% |

### Fair Same-Model Baseline

Both sides used `gpt-5.5`; the only intended scaffold difference is Assumption Graph context injection.

| Direction | AG wins | Baseline wins | Ties | AG win rate excl. ties |
|---|---:|---:|---:|---:|
| `phase2_v20_ag_filt_gpt55` vs `phase2_v20_gpt55` | 11 | 6 | 3 | 64.7% |
| reverse naming/order check | 12 | 3 | 5 | 80.0% |
| combined | 23 | 9 | 8 | 71.9% |

Non-bypass subset only, excluding math/science because v20 routes those through the hygiene bypass without graph context:

| Subset | AG wins | Baseline wins | Ties | AG win rate excl. ties |
|---|---:|---:|---:|---:|
| non-bypass 15 problems, combined directions | 17 | 5 | 8 | 77.3% |

By-domain combined result:

| Domain | AG | Baseline | Tie |
|---|---:|---:|---:|
| business | 6 | 0 | 2 |
| daily_life | 4 | 2 | 2 |
| engineering | 4 | 1 | 1 |
| software_engineering | 3 | 2 | 3 |
| mathematics, bypass | 4 | 0 | 0 |
| science, bypass | 2 | 4 | 0 |

## Retrieval Audit

Initial graph context was case-heavy: wisdom cross-domain case nodes occupied the top ranks, so strategy recall against `coverage_tags` was 0 at top 10.

Fix applied:

- `phase2_v20_framework.py` now retrieves primary Assumption Graph context with candidate types restricted to method/harness/retrieval/world_model/evaluator/alignment nodes.
- Formatting now shows up to 8 primary assumption nodes instead of case nodes.

Post-fix retrieval audit on the same non-bypass 15 problems:

| Visible top-k | Gold strategy hit | Recall | MRR |
|---:|---:|---:|---:|
| 5 | 11 / 15 | 73.3% | 0.391 |
| 8 | 11 / 15 | 73.3% | 0.391 |
| 10 | 12 / 15 | 80.0% | 0.399 |

## Interpretation

The result is directionally positive: same-model AG injection beats same-model v20 baseline on n=20, especially on the non-bypass task families where graph context is actually used.

The current weak spot is not answer quality collapse, but retrieval sharpness. Several misses retrieve plausible generic strategies but not the exact gold strategies.

## Graph Writeback

The non-bypass judged outcomes were replayed into the graph with:

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
  --top-k 8 \
  --summary-out "phase four/assumption_graph/eval_phase2_v20_gpt55_n20_writeback.json"
```

Writeback summary:

| Item | Count |
|---|---:|
| processed trials | 30 |
| skipped missing graph meta | 10 |
| AG wins | 17 |
| baseline wins | 5 |
| ties | 8 |
| no-residual successes | 17 |
| optimization residuals | 4 |
| memory-defect residuals | 1 |

The writeback uses a frozen retrieval pass before applying confidence/residual
updates, so trial attribution is not order-dependent.

Post-writeback retrieval audit on the same non-bypass 15 problems:

| Visible top-k | Gold strategy hit | Recall | MRR |
|---:|---:|---:|---:|
| 5 | 11 / 15 | 73.3% | 0.374 |
| 8 | 13 / 15 | 86.7% | 0.395 |
| 10 | 13 / 15 | 86.7% | 0.395 |

## Cached Artifacts

- Answers:
  - `phase two/analysis/cache/answers/phase2_v20_gpt55_answers.json`
  - `phase two/analysis/cache/answers/phase2_v20_ag_filt_gpt55_answers.json`
- Meta/drafts:
  - `phase two/analysis/cache/answers/phase2_v20_gpt55_meta.json`
  - `phase two/analysis/cache/answers/phase2_v20_ag_filt_gpt55_meta.json`
  - `phase two/analysis/cache/answers/phase2_v20_gpt55_drafts.json`
  - `phase two/analysis/cache/answers/phase2_v20_ag_filt_gpt55_drafts.json`
- Judgments:
  - `phase two/analysis/cache/judgments/phase2_v20_ag_filt_gpt55_vs_phase2_v20_gpt55.json`
  - `phase two/analysis/cache/judgments/phase2_v20_gpt55_vs_phase2_v20_ag_filt_gpt55.json`
- Graph writeback:
  - `phase four/assumption_graph/eval_phase2_v20_gpt55_n20_writeback.json`
  - `phase four/assumption_graph/trials.jsonl`
