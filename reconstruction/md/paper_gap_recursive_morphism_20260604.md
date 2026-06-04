# Paper gap note: recursive self-evolution and morphism benchmark

Date: 2026-06-04

This note records the two paper-level gaps selected for immediate implementation.

## Current gap

递归自进化证明现在能说“机制已闭环、有 gated repair、有 recursive runner”，但论文主张如果是“自我提出假设并递归论证”，就要展示多轮：

`failure -> hypothesis -> ablation -> accept/reject -> next hypothesis`

连续 3-5 代，整体性能或局部 clade productivity 明显改善。

范畴论/morphism 的独立贡献现在 structural morphism 有用，但还要证明它不是“更长更具体 prompt”。需要一个专门 benchmark：跨领域同构关系，比如勒夏特列原理/楞次定律、ResNet/skip connection、反馈控制/负反馈经济机制。对比 KG triple 和 embedding retrieval，证明 morphism 能找出 KG 找不到的结构相似性。

## Implementation target

1. Add a recursive self-evolution proof payload that turns the existing live ablation sequence into an auditable multi-generation trace. The trace must include failure diagnosis, generated hypothesis, ablation result, accept/reject decision, and the next hypothesis that follows from residuals.
2. Add a morphism-specific benchmark where the correct match is cross-domain structural similarity and lexical/KG surface cues point to distractors. This tests whether role/invariant morphism matching contributes beyond longer prompt text.

## Validation criteria

Recursive proof passes only if:

- at least 5 mainline generations are present;
- each generation has failure, hypothesis, ablation evidence, and decision fields;
- the sequence contains both accepted improvements and at least one rejected branch;
- global utility improves over the root failure on base or placebo comparison;
- local branch productivity improves on the weak structural clades.

Morphism benchmark passes only if:

- at least 8 cross-domain analogy cases are evaluated;
- morphism top-1 hit rate is at least 0.80;
- morphism beats both KG-triple and lexical embedding-proxy baselines by at least 0.20 absolute hit rate;
- non-lexical success rate is at least 0.75.

When a neural embedding backend is available, the benchmark also records a
real sentence-embedding retrieval baseline over the same surface text. The
current local validation uses `sentence-transformers/all-MiniLM-L6-v2`.

## Why this is aligned with reconstruction.md

This implements the missing paper evidence for recursive assumption evolution and category-inspired structural transfer without claiming a full category-theory theorem prover. The morphism layer is still deliberately bounded: objects, role transitions, composition hints, invariants, and negative controls are treated as finite diagrams for engineering validation.

## 2026-06-04 validation result

Artifacts:

- `phase four/assumption_graph/paper_readiness_20260604/recursive_self_evolution_proof_20260604.json`
- `phase four/assumption_graph/paper_readiness_20260604/morphism_independent_benchmark_20260604.json`

Recursive self-evolution proof:

- pass: true
- mainline generations: 5
- branch tests: 3
- rejected branches: 1
- best base utility delta over root: +0.135
- final placebo utility delta over root: +0.155
- bottleneck branch placebo delta: +0.500
- signal branch placebo delta from trace: +0.7083

Morphism independent benchmark:

- pass: true
- cases: 10
- morphism top-1 hit rate: 1.000
- KG-triple top-1 hit rate: 0.100
- embedding-proxy top-1 hit rate: 0.100
- neural embedding top-1 hit rate: 0.000
- morphism margin over best baseline: +0.900
- morphism margin over neural embedding: +1.000
- nonlexical structural success rate: 0.800

Test command:

`python3 -m unittest tests.test_assumption_os`

Result: 76 tests OK.
