# V17.0 M3 HippoRAG + Horizontal Wormhole Scaling Claims & Evidence (`claims_and_evidence.md`)

## 1. 大规模异质跨领域森林构建（`K = 36` 源与近失工作区，`22` 类可执行 Python 机理）
- 4 个独立冻结评测簇上，`36/36` 源与反例近失工作区 100% 严格通过，自动收敛为 **36 个活跃概念节点**、**4 条树内纵向辩证细化边（`local_implication_edges`）** 与 **11 条跨树中间层横向虫洞（`HorizontalWormholeEdge`）**。
- 查询索引严格执行 `validate_observable_query_signatures` 零标签泄漏校验（仅允许实测 `sig:*` 几何与协议特征）。

## 2. 单因子受控检索与契约对比（`672` 个未见目标工作区，全部 `96/96` 严格通过、`0` 假确认）
- **相对同元数据 `Flat_BM25_EqualMetadata`**：目标期复合成本差 `-12.4879`（端到端含建索引 `-10.2904`）；授权 `Recall@1` `0.6957` vs `0.7826`，每簇错误候选试算 `5.00` vs `6.00`。
- **相对关闭横向虫洞的纵向森林 `Forest_VerticalPPR_NoWormholes`**：目标期复合成本差 `-69.8999`（物理测量费差 `-60.0000`），授权 `Recall@1` `0.6957` vs `0.5652`。
- **召回 != 授权（相对 `HippoPPR_UncheckedRecall`）**：目标期复合成本差 `-29.7650`，`Hegel_M3_HippoPPR` 在零子进程开销下提前拦截 `108` 次超界/排除/跨尺度候选，而未检召回发生 `36` 次无效试算。
- **相对全历史强基线 `O_strong_AllValidatedPriors` 与条件技能库 `O_HExA_CodeSkills`**：目标期净省 `-130.5946`（端到端含源期与索引构建净省 `-104.6371`，`4/4` 簇全负）及 `-311.6509`（端到端 `-296.8934`）。
