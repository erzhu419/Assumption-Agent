# RAG-to-Memory Baseline Comparison - 2026-06-06

## Question

如果把 `From RAG to Memory: Non-Parametric Continual Learning for Large Language Models` / HippoRAG 2 的方法作为 baseline，现在的 bounded structural morphism layer 比它强多少？

## Scope

这次比较不是完整复现 HippoRAG 2 的全论文 QA benchmark。完整复现需要它的多数据集 corpus、OpenIE extractor、NV-Embed-v2、triple filter LLM、QA reader 和官方 evaluation。

这里做的是同任务、同候选集的机制对齐 baseline：

- 使用已有 morphism benchmark 的 10 个跨域结构同构 case；
- baseline 只能读取 `label`、`domain`、`surface_text`、`kg_triples`；
- baseline 不读取当前方法的 `objects`、`morphisms`、`composition_laws`、`invariants`、`negative_invariants`；
- 将 HippoRAG 2 的检索机制映射为 OpenIE-style triples、phrase nodes、passage nodes、relation edges、context edges、synonym edges、query-to-triple recognition filter、Personalized PageRank passage ranking；
- 当前方法使用 bounded category-inspired structural morphism scorer。

## Result

Artifact:

- `phase four/assumption_graph/paper_readiness_20260604/rag_to_memory_baseline_20260606.json`

Performance:

| metric | RAG-to-Memory/PPR baseline | structural morphism | margin |
|---|---:|---:|---:|
| top-1 hit rate | 0.300 | 1.000 | +0.700 |
| top-2 recall | 0.400 | 1.000 | +0.600 |

Relative top-1 multiplier:

- `1.000 / 0.300 = 3.333x`

So on this cross-domain structural analogy benchmark, the current method is:

- +70 percentage points top-1 over the RAG-to-Memory-style graph-memory baseline;
- +60 percentage points top-2 recall over that baseline;
- 3.33x the top-1 hit rate.

## What The Baseline Did Catch

The RAG-to-Memory-style baseline correctly retrieved 3 / 10 cases:

- `morph_resnet_kalman`
- `morph_ci_enzyme_bottleneck`
- `morph_budget_mass_balance`

These are cases where lexical KG triples and synonym/context edges already expose enough of the bridge.

## What It Missed

It missed 7 / 10 cases:

- `morph_le_chatelier_lenz`
- `morph_jepa_seismic`
- `morph_thermostat_stabilizer`
- `morph_strangler_bridge`
- `morph_abtest_clinical_trial`
- `morph_counterexample_flight_envelope`
- `morph_compiler_assembly`

The typical failure mode is exactly the motivation for morphism: graph memory can activate local lexical or synonym-neighborhood matches, but it does not know that two domains share the same higher-order structural invariant unless that invariant is explicitly represented.

Examples:

- Le Chatelier vs Lenz law: KG/PPR is pulled toward same-domain chemistry words like `temperature` and `reaction`, while morphism matches `perturb -> opposing response -> restore constraint`.
- JEPA vs seismic autocorrelation: KG/PPR is pulled toward pixel/noise surface overlap, while morphism matches `stable signal separated from stochastic nuisance`.
- Strangler migration vs bridge retrofit: KG/PPR is pulled toward `legacy service` inventory/rewrite surface text, while morphism matches `slice -> verify -> expand_or_rollback while preserving old path`.

## Graph-Memory Health Check

The baseline was not empty or strawman:

- average nodes per case: 75.3
- average edges per case: 150.0
- average phrase nodes per case: 66.3
- average triple nodes per case: 6.0
- average synonym edges per case: 46.8
- average filtered triples per case: 4.4
- fallback case count: 0

## Interpretation

This supports the narrower claim:

> On cross-domain structural analogy retrieval, a bounded structural morphism representation retrieves cases that a HippoRAG-2-style graph memory/PPR baseline misses when the baseline is restricted to passage text and KG triples.

This does not mean the current system is better than HippoRAG 2 on factual QA, multi-hop QA, or long-context sense-making. HippoRAG 2 is designed for non-parametric continual knowledge retrieval. The current method is stronger here because the tested target is different: finding reusable structural invariants across domains.

## Reproduction

Commands:

```bash
python3 -m assumption_os.rag_to_memory_baseline --out 'phase four/assumption_graph/paper_readiness_20260604/rag_to_memory_baseline_20260606.json'
python3 -m assumption_os.paper_repro_pack --root . --out 'phase four/assumption_graph/paper_readiness_20260604/paper_repro_pack_20260605.json'
python3 -m unittest tests.test_assumption_os.AssumptionOSTest.test_rag_to_memory_baseline_quantifies_morphism_margin
```

Validation status:

- baseline artifact gate: pass
- repro pack gate: pass
- targeted performance validation: pass
