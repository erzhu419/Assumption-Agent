# Hegel V17.2 (`M4`)：开放组合式 AST 科学发现、主动多尺度粗粒化重整化（$\Pi_s$）与 $K=64$ 自生长保真压缩森林实测报告

> **里程碑定位**：本报告标志 `Hegel V16.5 -> M0 -> M1 -> M2 -> M2.5 -> M3 -> M3.1 -> M4` 通用多尺度认知架构主线的全面合流与收官落地。在严格保持 `docs/evidence/v16_8_*`、`v16_9_*`、`v17_0_*`、`v17_1_*` 历史静态证据零改动的前提下，`V17.2 (M4)` 证据包与持久化作者代码完整归档于 `docs/evidence/v17_2_m4_multiscale_open_discovery_20261119/`（全量运行耗时 `103.49s`，`M0–M4` 全套 `28/28` 单元测试 `100%` 通过）。

---

## 一、`M4` 三大核心突破：把“管中窥豹 + 多尺度普适类”化为可证伪工程闭环

1. **支柱一：零预设复合公式表的开放组合式 AST 演化器（`m4_compositional_discovery.py`）**
   - 彻底告别从固定公式枚举表里挑答案：`OpenCompositionalCodeProvider` 在原生 `CodeAgent`（`query / write / evaluate / run / finish`）循环中仅从原子算子原语（`linear_1d`, `michaelis_1d`, `exp_relax_2d`, `murray_power_2_3_1d`, `variational_cellular_2d`）出发；当基元拟合超限时，运行诊断脚本 `diag_residual.py` 读取真实残差几何 `stdout`，按需动态施加结构组合子（`denom_quadratic_inhibition`、`variational_ceiling_cap`、`additive_secondary_shunt`、`oscillatory_damping_modulation`、`fast_mode_transient_correction`），支持**多轮迭代复合**（如 `Monod -> Haldane 1D -> Haldane + Shunt 2D`），自动编译纯 Python 阻尼 Levenberg-Marquardt `fit(observations)` 与纯函数 `predict(row)`，并通过 `canonicalize_python_model_ast` 纯 AST 骨架互验。
   - **完整满足审计 `§10.2` 六项可反驳工程判据**：
     - **反事实见证分叉（`R2`）**：同一个 `S1` 单调饱和源面对 3 种不同物理反例见证（`W_haldane_shunt`, `W_variational_cap`, `W_pure_shunt`）与时间弛豫族 2 种反例见证（`W_underdamped_ringing`, `W_fast_mode_excess`），生成 **5 个互异的 `VERIFIED_PURE_AST` 骨架摘要**，全部达到 `confirm_rmse = 0.0`、`heldout_rmse = 0.0`、`strict_pass = True`，且全部保留原 `S1` 低底物作用域零退化（`backward_s1_preserved = True`，`R3`）。
     - **历史产物因果可删可恢复（`R1`）**：在未见逆向传感器重排目标上，保留持久化 `authored_method.json` 时 `1` 次候选评估即收敛（`ran_diagnostic = False`, 算力 `22.0`）；**删除该持久化文件后**被迫触发 2 轮残差诊断重合成（`ran_diagnostic = True`, 评估 `4` 次，算力升至 `46.0`）；**恢复文件字节后** `100%` 恢复 `22.0` 算力与单步迁移行为，且严格优于同初始 Pilot 的全历史普通作者（算力 `52.0`，节省 `-57.7%`）。

2. **支柱二：带付费微宏观双通道的主动多尺度重整化与尺度索引虫洞（`m4_multiscale_renormalization.py`）**
   - 直接将**“同一系统在不同尺度拥有完全不同的跨领域同构邻居”**建成双通道付费实验台 `MultiScaleWorkspaceTaskLab`（廉价宏观粗粒化通道 `coarse_macro` $\Pi_s$：`4.0 / 点` vs 昂贵微观精细通道 `micro_fine`：`12.0 / 点`）：
     - **微观分子/快弛豫尺度 $S_{\text{micro}}$**：水分子极性快模弛豫 `aq_micro_polar` $\longleftrightarrow$ 氨偶极溶剂化 `chem_ammonia_micro` $\longleftrightarrow$ 分布式队列重试背压瞬态 `sys_queue_micro`（而疏水蜂蜡 `bio_wax_hydrophobic_micro` 在此尺度非极性互斥，**不开虫洞**！）；
     - **介观界面变分尺度 $S_{\text{meso}}$（经 $\Pi_{\text{micro}\to\text{meso}}$ 消去快模后涌现）**：雪花/对流胞六边形面积-周长变分极小 `aq_meso_hex_cell` $\longleftrightarrow$ **蜜蜂蜂巢密铺 `bio_honeycomb_meso`** $\longleftrightarrow$ 分布式分片边界通信极小 `sys_shard_meso`；
     - **宏观分形输运尺度 $S_{\text{macro}}$（经 $\Pi_{\text{meso}\to\text{macro}}$ 空间树聚合涌现）**：河流汇水网 `aq_macro_river_tree` $\longleftrightarrow$ **生物血管 Murray $2/3$ 幂律网络 `bio_vascular_macro`** $\longleftrightarrow$ 跨区域 CDN 骨干树 `sys_cdn_macro`（含高流量饱和封顶子规律 `bio_vascular_macro_capped`）。
   - **主动尺度闭合探针（`run_active_scale_closure_probe`）与辩证下钻（Escalation）**：先做 1 对微宏观边界比对测量 $|\hat{y}_{\text{micro}} - \hat{y}_{\text{coarse}}|$；闭合成立时签发 `ScaleMap` 证书并全程走 $3\times$ 更便宜的 `coarse_macro` 通道；闭合破裂（快淬火非绝热区）时自动阻断宏观定律误用并沿 `coarse_graining_edges` 逆向下钻至 $S_{\text{micro}}$！

3. **支柱三：$K = 16 \to 32 \to 48 \to 64$ 持续自生长、保真压缩与过期虫洞重验（Continual Lifecycle）**
   - 在持续涌入跨领域同构副本、高频诱饵干扰源与中间层节点经验修订的 4 阶段流中，在线执行 `compress_compatible_nodes`（零冲突、零假新拒绝 `FidelityCompression`）、`revalidate_wormhole_after_revision` 与 Minton 效用账本归档，将 `59` 个注册节点压缩保持在 **`17` 个活跃节点（体积压缩 `-71.2%`）**，保持 **`0.0%` 历史遗忘率**！

---

## 二、`V17.2 (M4)` 预注册实测总表

### 1. 支柱一：反事实见证结构分叉与持久化方法因果性验证

| 实验检验项 | 观测诊断签名 / 控制条件 | 合成算子族 (`VERIFIED_PURE_AST`) | AST 骨架摘要 | 确认 / 留出 RMSE | 原 `S1` 作用域保持 | 算力 / 复合成本 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `W_haldane_shunt`（底物抑制+旁路） | 第1轮 `turnover_drop=True` $\to$ 第2轮 `secondary_corr=0.809` | `michaelis__comb_quad_inh__comb_shunt` | `c2ced29ba2c345` | `0.000` / `0.000` | **`True`** | — |
| `W_variational_cap`（变分硬封顶） | `kink_plateau_ceiling=True` | `michaelis__comb_cap1d` | `176ae5aef4e4e4` | `0.000` / `0.000` | **`True`** | — |
| `W_pure_shunt`（纯副通道分流） | `secondary_active=True` (`corr=0.854`) | `michaelis__comb_shunt` | `fa3cf16ad597b5` | `0.000` / `0.000` | **`True`** | — |
| `W_underdamped_ringing`（欠阻尼振铃） | `oscillatory_overshoot=True` | `exp_relax_2d__comb_osc_ringing` | `5ab363f56cb6c7` | `0.000` / `0.000` | — | — |
| `W_fast_mode_excess`（快模瞬态余项） | `fast_transient_decay=True` | `linear_1d__comb_fast_micro` | `a6eb5d9ace81d3` | `0.000` / `0.000` | — | — |
| **目标迁移：`Hegel_Intact_Artifact`** | 加载持久化 `authored_method.json`（`ran_diag=False`, 评估 `1` 次） | 直接复用合成算子并重拟合逆向透镜 | `c2ced29ba2c345` | `0.000` / `0.000` | — | **`22.0` / `41.10`** |
| **删除控制：`Hegel_Deleted_Artifact`** | **删除持久化产物文件**（被迫跑诊断重合成：`ran_diag=True`, 评估 `4` 次） | 重新经 2 轮残差诊断合成 | `c2ced29ba2c345` | `0.000` / `0.000` | — | `46.0` / `42.30` |
| **恢复控制：`Hegel_Restored_Artifact`** | **恢复原文件字节**（`ran_diag=False`, 评估 `1` 次） | `100%` 恢复单次命中与原成本 | `c2ced29ba2c345` | `0.000` / `0.000` | — | **`22.0` / `41.10`** |
| **同工具对照：`Ordinary_AllHistory_EqualPilot`** | 相同初始 Pilot（零强制加点）+ 遍历全历史先验（评估 `6` 次） | 全历史顺序筛选 | `c2ced29ba2c345` | `0.000` / `0.000` | — | `52.0` / `42.60` |

---

### 2. 支柱二：多尺度重整化 × 尺度索引虫洞 × 两跳游走七臂对比（4 冻结簇 × 18 目标 = 每臂 `72` 个工作区 + `8` 个域外工作区）

| 受控实验臂 | `strict_pass` | 假确认 | 总召回确认 (`72`) | **`T_MultiScale_Walk` 两跳游走确认 (`16`)** | **`T_Closure_Breakdown` 微观下钻 (`16`)** | **`T_Adiabatic` 平均测量费** | **`T_Breakdown` 复合成本** | **`T_Walk` 复合成本** | 全集平均测量费 | 全集平均算力费 | **全集平均复合成本** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Hegel_M4_MultiScale_HippoWalk`** | **`1.000`** | **`0`** | **`72/72`** | **`16/16` (`100%`)** | **`16/16` (`100%`)** | **`42.000`** | **`100.656`** | **`44.953`** | **`55.778`** | **`15.747`** | **`56.565`** |
| `HippoPPR_Alpha0_MultiScalePool` | `1.000` | `0` | `56/72` | **`0/16` (`0%`)** | `16/16` | `42.000` | `100.599` | `57.799` | `58.444` | `18.642` | `59.377` |
| `HippoPPR_ScaleBlind_Walk` | `1.000` | `0` | `56/72` | `16/16` | **`0/16` (`0%`)** | `34.000` | **`133.556`** | `44.953` | `58.444` | `19.747` | `59.432` |
| `Flat_BM25_WormholePool_EqualContent` | `1.000` | `0` | `56/72` | **`0/16` (`0%`)** | `16/16` | `42.000` | `100.642` | `57.842` | `58.444` | `19.507` | `59.420` |
| `Forest_VerticalOnly_NoWormholes` | `1.000` | `0` | `56/72` | **`0/16` (`0%`)** | `16/16` | `42.000` | `100.640` | `57.839` | `58.444` | `19.467` | `59.418` |
| `Continuing_AllHistory_CodeAuthor` | `1.000` | `0` | `0/72` | `0/16` | `16/16` | `46.000` | `112.500` | `53.100` | `62.444` | `19.667` | `63.428` |
| `MicroOnly_AlwaysFine_Forest` | `1.000` | `0` | `52/72` | `0/16` | `0/16` | **`80.000`** | `84.656` | `109.253` | `88.000` | `17.413` | `88.871` |

- **域外诚实性**：`8/8` 个混沌突变域外工作区全部返回 `scoped_unknown`，`false_confirmations = 0`。
- **DEV 门控校准独立入账**：`dev_total_composite_cost = 1382.872`（按单目标净节约相对 `Continuing_AllHistory` 约 `201` 道目标摊平，相对 `MicroOnly` 约 `43` 道目标摊平）。

---

### 3. 支柱三：$K = 16 \to 32 \to 48 \to 64$ 持续演化、保真压缩与零遗忘曲线

| 持续学习臂 | Stage 1 ($K=16$)<br>活跃节点 / 复合成本 | Stage 2 ($K=32$)<br>活跃节点 / 复合成本 | Stage 3 ($K=48$)<br>活跃节点 / 复合成本 | Stage 4 ($K=64$)<br>**活跃节点 / 算力 / 复合成本** | 历史遗忘率 (`Forgetting`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Hegel_M4_FullLifecycle`** | **`17` / `57.796`** | **`17` / `57.797`** | **`17` / `57.797`** | **`17` (`-71.2%`) / `15.930` / `57.797`** | **`0.0000`** |
| `Forest_NoCompression_NoPruning` | `17` / `57.796` | `31` / `57.814` | `45` / `57.831` | `59` / `16.973` / `57.849` | `0.0000` |
| `Flat_BM25_WormholePool_Continual` | `17` / `60.993` | `31` / `61.049` | `45` / `61.105` | `59` / `23.220` / `61.161` | `0.0000` |
| `Frozen_Stage1_Memory` | `16` / `61.010` | `16` / `61.010` | `16` / `61.010` | `16` / `20.200` / `61.010` | `0.0000`（缺失新演化分支，召回卡在 `75%`） |
| `Continuing_AllHistory_CodeAuthor` | `17` / `64.950` | `31` / `64.950` | `45` / `64.950` | `59` / `19.000` / `64.950` | `0.0000` |
