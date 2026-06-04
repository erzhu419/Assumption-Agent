# Novelty / Integration Gate 2026-06-04

## 要解决的问题

之前系统已经能做 variation -> evaluation -> selective retention：

- variation: residual / world-model / trace-policy / structural morphism 生成多个 candidate proposals。
- evaluation: preflight、fresh ablation、judge、candidate acceptance、formal/structural gate。
- selective retention: accepted candidates gated apply，rejected 保留为审计记录。

缺口是：每个新假设在进入 retention 前，缺少统一机制判断它到底是旧假设的重复、旧假设族的特化、跨域类比、范畴/结构同构，还是应创建全新假设族。

## 新增机制

新增 `assumption_os/novelty_integration.py`，对所有 candidate proposals 生成统一 payload：

- `duplicate`: normalized claim 或同族高相似，推荐 `merge_with_existing`，accepted 时不新增候选节点。
- `specialization`: scope narrowing / revision / parent edge 指向旧族，推荐 `specializes`。
- `formal_isomorphism`: structural morphism candidate 通过 functor/kernel gate，推荐 `is_formal_isomorphism_of`。
- `analogy`: structural transfer 或跨类型结构相似但未证明同构，推荐 `is_analogy_of`。
- `genuinely_new_family`: 未落入旧族/同构/类比，作为新假设族保留。
- `manifest_only`: 没有 candidate node，不参与图结构写入。

这层不替代 benefit/harm acceptance；它只回答“被接受后应该如何整合进图”。

## 闭环接入

- `evolution_cycle` 自动产出 `novelty_integration` payload。
- `policy_update_plan` 每个 proposal 显示 novelty classification 和 recommended action。
- `apply_accepted_candidates` 支持传入 novelty payload：
  - duplicate accepted 不新增候选节点，避免污染主图。
  - specialization / formal_isomorphism / analogy accepted 会附加 gate 推荐边。
  - manifest metadata 记录 novelty classification、match basis、match score。
- `recursive_daemon` 的 accepted apply 路径同样传入 novelty payload，避免递归执行绕过该 gate。

## Performance Validation

Artifact:

`phase four/assumption_graph/paper_readiness_20260604/novelty_integration_gate_20260604.json`

Fixture 覆盖 5 类 candidate：

- duplicate
- specialization
- formal_isomorphism
- analogy
- genuinely_new_family

结果：

- proposal_count: 5
- classified_count: 5
- gold_accuracy: 1.0
- required_classes_present: true
- formal_edges_recommended: true
- analogy_edges_recommended: true
- pass: true

对应单测：

- `test_novelty_integration_gate_classifies_candidate_family`
- `test_novelty_integration_performance_validation_passes`
- `test_apply_accepted_candidates_uses_novelty_integration_edges`

## 与“递归式自主提出假设并自我论证”的关系

现在新假设进入图前多了一层结构化归属判断：

1. variation 产生多个 candidate。
2. novelty/integration gate 判断 candidate 与旧图的关系。
3. evaluator / ablation / judge 判断 candidate 是否有实际收益。
4. accepted candidate 按 novelty gate 的推荐方式进入图。
5. rejected candidate 保留为后续 residual / trajectory search 的失败证据。

因此“新假设是否落到旧假设内，还是成为新假设族”已经有统一实现，不再只是局部 heuristic。
