# Hegel V17.0（里程碑 M3）：大规模异质哲学森林（`K = 36`）、HippoRAG 特异性加权 PPR 与横向虫洞单因子实测报告

## 1. M3 核心科学命题与架构兑现

在 V16.9（M2.5 红队闭环）中，我们在小规模候选库（`K = 7`）上诚实记录了两个关键边界：(1) 当候选库仅有 7 个节点且无枢纽特征碰撞时，保留完整路径界与排除条件的扁平表与图检索仅相差常数扫描开销，关闭横边但保留局部跨领域读取亦打平；(2) 全历史强基线 `O_strong_AllValidatedPriors` 在单批 10 道目标上尚未摊平源期反例检验多投的 `+15.0` 开销。我们当时坚定保留了 **`PhilosophicalForest` + `HorizontalWormholeEdge` + 纵向 `local_implication_edges`** 骨架，正是为了在 **M3 大规模异质干扰森林（`K = 36`）** 中检验其真正的规模化因果机制。

`V17.0 (M3)` 将开放 Python 工作区基准扩展至 **`K = 36` 个跨学科源与反例近失工作区（覆盖 `23` 类互不相同的可执行 Python 数学物理/工程机理）**，接入 **HippoRAG 特异性加权个性化 PageRank（`HippoRAGWormholeIndex`）**，并在 **`2` 个开发校准簇 + `4` 个独立冻结评测簇 × `24` 道未见目标 × `7` 个严格单因子受控臂 = `672` 个目标工作区 + `8` 个域外诚实性工作区**上完成了端到端实测：

1. **零标签泄漏可观测索引（`validate_observable_query_signatures`）**：图特征词与查询种子仅允许由对称 pilot 实验实测几何不变量、校准读出极性及公开 `ScaleMap` 生成的 `sig:*` 词元（`27` 个共享几何/协议特征词），严禁任何领域名、特征名或真值机理名进入索引。
2. **概率守恒的概念森林 + 横向虫洞马尔可夫游走**：
   - **特异性种子重置**：按 $\sigma \in Q$ 在活跃节点上的度 $\deg(\sigma)$ 赋予权重 $s(\sigma) = 1 / \deg(\sigma)^{1.15}$ 并均分至关联概念节点，彻底阻断高频枢纽词（如 `sig:arity_2`）在随机游走中途充当跨树泄漏桥梁。
   - **纵横双轴传播**：树内沿反例驱动的辩证扬弃边 `Parent -> RefinedChild` 向上汇聚，树间沿经纯数据流 AST 互验的 `HorizontalWormholeEdge` 双向传播（权重按实测 `hop_error_budget` 衰减），平稳分布严格满足 $\sum_v p(v) = 1$。
3. **绝对公平的同元数据单因子对照与“召回 $\neq$ 授权”硬隔离**：
   - `Hegel_M3_HippoPPR`、`Flat_BM25_EqualMetadata`、`Flat_Overlap_EqualMetadata`、`Forest_VerticalPPR_NoWormholes` 与 `HippoPPR_UncheckedRecall` 消费 **同一个冻结 `PhilosophicalForest` 实例的精确克隆**（100% 相同的节点、`ClaimContract`、`ObservationMap`、`ScaleMap`、`ALIGN` 排除条件、非传递路径误差界与 `Stage-1` 门控），**唯一自变量仅为检索算子或契约审查开关**！

---

## 2. 冻结 4 评测簇（`672` 目标工作区 + `8` 域外对照）核心实测结果

| 受控实验臂 | 严格通过率 | 假确认 | 源期+建索引成本 | 目标物理测量费 (`C_meas`) | 目标逻辑计算 (`C_comp`) | 实测 Python 墙钟 (s) | 目标复合成本 (`λ=0.05`) | 端到端总复合成本 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Hegel_M3_HippoPPR`** | **`96 / 96` (100%)** | **`0`** | `1980.96` | **`830.00`** | **`817.49`** | **`1.85s`** | **`870.87`** | **`2851.83`** |
| `Flat_BM25_EqualMetadata` | `96 / 96` (100%) | `0` | `1978.76` | `840.00` | `867.24` | `1.76s` | `883.36` | `2862.12` |
| `Flat_Overlap_EqualMetadata` | `96 / 96` (100%) | `0` | `1978.76` | `840.00` | `867.24` | `1.77s` | `883.36` | `2862.12` |
| `Forest_VerticalPPR_NoWormholes` | `96 / 96` (100%) | `0` | `1980.88` | `890.00` | `1015.48` | `2.26s` | `940.77` | `2921.65` |
| `HippoPPR_UncheckedRecall` | `96 / 96` (100%) | `0` | `1980.96` | `855.00` | `912.79` | `2.09s` | `900.64` | `2881.60` |
| `O_strong_AllValidatedPriors` | `96 / 96` (100%) | `0` | `1955.00` | `856.25` | `2904.38` | `2.02s` | `1001.47` | `2956.47` |
| `O_HExA_CodeSkills` | `96 / 96` (100%) | `0` | `1966.20` | `1095.00` | `1750.50` | `3.73s` | `1182.53` | `3148.73` |

### 关键成对差值（`Hegel_M3_HippoPPR - Baseline`，`N = 4` 独立评测簇均值）

| 对比维度 | 目标测量费差 `ΔMeas` (`λ=0`) | 目标复合差 `ΔTarget` (`λ=0.05`) | 端到端总差 `ΔE2E`（含源期+建图） | `4/4` 簇全负 | 端到端回本目标数 `N*` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **vs `Flat_BM25_EqualMetadata`（纯检索结构效应）** | **`-10.0000`** | **`-12.4879`** | **`-10.2904`** | **Yes (`4/4`)** | **`N* = 13`** |
| **vs `Flat_Overlap_EqualMetadata`（IDF 余弦）** | **`-10.0000`** | **`-12.4879`** | **`-10.2904`** | **Yes (`4/4`)** | **`N* = 13`** |
| **vs `Forest_VerticalPPR_NoWormholes`（横向虫洞纯效应）** | **`-60.0000`** | **`-69.8999`** | **`-69.8174`** | **Yes (`4/4`)** | **`N* = 8`** |
| **vs `HippoPPR_UncheckedRecall`（召回 $\neq$ 授权效应）** | **`-25.0000`** | **`-29.7650`** | **`-29.7650`** | **Yes (`4/4`)** | **`N* = 5`** |
| **vs `O_strong_AllValidatedPriors`（全历史强基线）** | **`-26.2500`** | **`-130.5946`** | **`-104.6371`** | **Yes (`4/4`)** | **`N* = 7`** |
| **vs `O_HExA_CodeSkills`（条件技能+错误库）** | **`-265.0000`** | **`-311.6509`** | **`-296.8934`** | **Yes (`4/4`)** | **`N* = 1`** |

> **$\lambda$ 全区间稳健性**：因为 `Hegel_M3_HippoPPR` 在**纯物理测量费（`λ = 0.00`）与逻辑计算费（`ΔCompute < 0`）两个正交分量上同时严格低于全部 6 个对照臂**，故对任意 $\lambda \in [0.00, 0.40]$，`Hegel_M3_HippoPPR` 均严格占优（无任何交叉反转点）。

---

## 3. 检索诊断、真实权衡与域外诚实性边界

1. **`Recall@1` 与 `Recall@2` 的真实图扩散权衡（`index_and_retrieval_diagnostics.json`）**：
   - `Flat_BM25_EqualMetadata`：授权 `Recall@1 = 0.7826`，但授权 **`Recall@2 = 0.9130`**（在两跳非传递桥接目标上被单域反号干扰项挤出 Top-2 试用窗口，触发全量探索，每簇多付 `+10.0` 物理测量费）。
   - `Hegel_M3_HippoPPR`：授权 `Recall@1 = 0.6957`（图扩散在 `T_Assoc` 上将部分一跳节点的第 1 名锐度分流至相邻高连通节点，导致 `T_Assoc` 子类比 `BM25` 高 `+6.33`），但授权 **`Recall@2 = 1.0000`（`92 / 92` 100% 命中 Top-2 窗口！）**，每簇错误候选试算全场最低（`5.0` vs `BM25` `6.0` vs `Unchecked` `9.0` vs `NoWormholes` `13.0`），在 `T_MultiHopGuard` 子类净省 `-17.67`/簇，实现整体物理测量与端到端成本双赢。
2. **契约预筛与域外诚实性（`out_of_scope_honesty.json`）**：
   - `Hegel_M3_HippoPPR` 通过 Tier-2 契约、`compose_wormhole_chain` 与 `ScaleMap` 零子进程开销提前过滤 **`108` 次**违约候选。
   - 在 23 类机理之外的 `OOS01_pure_sinusoid_wave`（确认 `RMSE ≈ 1.83`）与 `OOS02_chirp_oscillator_2d`（确认 `RMSE ≈ 1.06`）共 `8/8` 个工作区上，100% 诚实返回 `status = "scoped_unknown"`，`0` 假确认。

---

## 4. 新增模块、测试与证据归档清单

- 核心检索索引模块：[hipporag_wormhole_index.py](../../src/hegel_machine/cognition/hipporag_wormhole_index.py)
- 大规模 36 源 / 23 机理开放工作区宇宙与受控求解器：[m3_scale_universe.py](../../src/hegel_machine/cognition/m3_scale_universe.py)
- M3 单元测试套件（连同 M0/M1/M2/V16.9 共 `20/20` 全部通过）：[test_v17_0_m3_hipporag_wormhole_scaling.py](../../tests/test_v17_0_m3_hipporag_wormhole_scaling.py)
- M3 正式评测脚本：[run_v17_0_m3_hipporag_scale_study.py](../../scripts/run_v17_0_m3_hipporag_scale_study.py)
- 静态只读证据包目录（独立于 V16.8 与 V16.9）：`docs/evidence/v17_0_m3_hipporag_wormhole_scale_20261119/`
