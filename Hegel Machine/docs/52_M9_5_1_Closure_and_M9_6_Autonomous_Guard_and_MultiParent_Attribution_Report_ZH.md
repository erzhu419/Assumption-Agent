# M9.5.1 审计闭环与 M9.6 自主守卫及多父机制图持续流归因报告

- **M9.5.1 确定性闭环提交（`Commit 1`，`0` 次付费 LLM 调用）**：`767d7b4cceb93d6f39c8d39883616f0f5d829aa7`（`docs/evidence/m9_5_1_audit_closure_and_controlled_attribution_20261127/`）
- **M9.6 正式实验冻结提交（`Commit 2.1`）**：`6c39d51e5e90872b1ad8be61c0d7226f75777d6a`（源码树 SHA-256 起止完全一致：`e7ac0ec61423e2319a7e1a9b4304a99cf6bbca1938e18ebc2efb559eaab9d35c`）
- **M9.6 封存证据包（`run2`）**：`docs/evidence/m9_6_autonomous_guard_multiparent_continual_run2_20261127/`（`2,173` 个 SHA-256 校验文件，`136/136` 条合法决策溯源 DAG，`567/567` 份原始调用账本镜像，未知用量调用数 `0`）
- **完整保留的中断运行（`run1`，Turn-1 前 `hash_probe` 只读资产遍历遗漏）**：`docs/evidence/m9_6_aborted_run1_hash_probe_readonly_asset_walk_bug_20261127/`（`114` 次调用，`v2 = 3,244,728`，已 100% 计入 Scope 3 物理总开销）
- **并发执行引擎**：`32` 路并行 `gpt-6.2-sol` CLI（`configured_codex_concurrency_cap = 32`，`configured_stream_workers = 32`，实测峰值活跃 CLI 并发 = `16`）

---

## 一、核心科学结论（正向突破与证伪发现全部如实呈现）

1. **M9.5 审计发现 100% 确定性闭环（`M9.5.1`，`F01–F14`，`0` 次 LLM 调用）**：
   - **`F01`（双重包装适配器缺陷）**：改为从原始规范算法（canonical solver）做单跳（1-hop）参数适配后，`M9.5` 中 `8` 个被双重包装的 `Flat_ParameterizedCache` 轮次有 **`6/8`**（`12/12` 个私有种子）直接翻转为 `strict_pass = True`；仅剩 `2` 个真正的跨物理机制回退轮次（`beta_w02 R`、`beta_w04 R`）仍需机制守卫区分。
   - **`F02`（私有验证器校准历史显式挂载）**：在公共探针与私有验证器中同步挂载前缀锁定（`<= t - 1`、`0444` 只读、SHA-256 校验）的 `/tmp/hegel_continual_archive/archive_calibration_history.json` 后，受影响的 `5` 个读取历史的 Beta 轮次 **`5/5`**（`10/10` 个私有种子）全部通过。
   - **`F03`（`48` 个非发现复用步上的四因素离线受控回放）**：
     - `Flat_F01_Unvalidated_SingleLatest_Probes1`：`28/48`（`48` 次探针）
     - `Flat_Multiversion_Validated_Probes1`：`40/48`（`48` 次探针——普通 `2/3–1/3` 留出验证分翼 OLS 在 `8/8` 个 `G2` 抗 EMI 步上与 `M8` 完全等价！）
     - `Flat_Multiversion_Validated_Probes2`：`48/48`（`56` 次探针）
     - `Full_Mechanism_Graph_Guard_Probes1`：`48/48`（`48` 次探针）
   - **`F04–F14`**：统一 `CostPolicy` 双口径账本（`M9.5` 流内 `v2`：`16,082,306 / 15,065,362 / 5,935,605`）、Alpha 双层 OLS 应力回归反作弊校验（`F05`）、零 `pickle` 纯 JSON 数值进程隔离 RPC（`F06`）、`QualificationReceipt` 外键强校验（`F07`）、`M9.3` 真实锁定源冒烟验证（`F08`）、AST 槽位前缀 `[1, *path]`（`F09`）、`train+test` 全分片历史核查（`F10`），以及 `M9.4` 全量 `68` 私有实例 vNext 重评（`60/68` 实例 = `30/34` 轮次，`F11`）。

2. **M9.6 正向突破——零人工硬编码的自主守卫与多父依赖图归纳（`8/8` 世界一次命中，仅占求解器开销 `+2.90%`）**：
   - Runner 中**人工硬编码守卫数为 `0`**。面对混有短视瞬时干扰特征（myopic snapshot distractors）与范数干扰特征的 `12` 维通用结构特征库，`gpt-6.2-sol` 在全部 `8/8` 个世界的 Step `C`（此时非马尔可夫事件 `U` 尚未出现！）首次调用即 **`8/8`（`100%`）一次性通过公开正反例对比验证与上游微扰敏感度检验**（被拒提案 `0` 次，`8` 个世界总归纳成本仅 `145,244` `v2`，平均每世界仅 `18,156` `v2`）。
   - 在全部 `4/4` 个 Alpha 世界自主选出含历史状态的 `max_history_and_hold_plus_residual_state_excess`，在全部 `4/4` 个 Beta 世界自主选出含涡量历史的 `combined_nonseparable_or_vortex_history_excess`，并在全部 `8/8` 个世界精确绑定双翼校准父节点 `(wing_a, wing_b)`，100% 剔除 `aux_bench_c_calibration` 与 `sentinel_shared_acoustic_root`（全部 `64` 步无关节点漂移严格为 `0.0`）。
   - 在后续流中，该自主归纳守卫以 **`0` 轮 LLM 调用**将 **`8/8` 个基态回归 `R` 路由至 `C0_base`**，并将 **`8/8` 个非马尔可夫卸载残余记忆事件 `U`（当前时刻输入与 `R` 同分布、唯历史轨迹不同！）精准路由至 `C1_extension`**（共完成 `31` 次基于对比验证守卫的零轮闭环）。

3. **M9.6 关键证伪结论——在补齐全部工程公平性的最强基线 `Flat_Multiversion_Validated` 面前，“嵌套物理包含性（`C1 ⊃ C0`）”消解了准确率与轮次差距，而在线 `L2/L1` 森林编译成为未被摊销的净负向税（流内 `v2` `+32.44%`，`p = 0.0078125`）**：
   - 当 `Flat_Multiversion_Validated` 同时获得 `F01` 单跳规范导出、`F02` 只读历史清单、`F03` 留出验证分翼 OLS 且不背负任何编译税时，**两组的通过率、零轮闭环数与总交互轮数完全打平**：
     - `strict_pass`：**`Flat 63/64`（`98.44%`）vs `Full 63/64`（`98.44%`）**（`p = 1.0`）。
     - 零轮闭环数：**`Flat 47/64` vs `Full 47/64`**（`p = 1.0`）；总轮数：**`210` vs `210`**（`p = 1.0`）。
   - **为什么 `Flat` 在 `max_pre_llm_probes = 1` 下也能零轮通过 `R` 和 `G2`？** 因为在物理结构上，`C1`（含屈服粘塑性 / 2D 非可分 Sylvester + 涡量记忆）天然是基态 `C0`（亚屈服线性粘弹性 / 1D 可分 Kronecker）的**数学超集（`C1 ⊃ C0`）**！当输入回到亚屈服或可分基态（`R`、`G2`）时，`Flat` 默认选取的最新规范求解器 `C1` 会自动令过应力项归零（输出 `viscoplastic_weights = [0, 0]`），第 1 次探针即直接通过公共门控与私有验证器，无需退回 `C0`！
   - 与此同时，在跨域接口任务 `T` 上，`Full` 虽然 `8/8` 次查看了技能目录（`8/8`），但对在线编译的跨族 `L2/L1` 算子虫洞实际执行与保留率均为 **`0/8`（精确可达保留 `0/8`）**（`T` 平均轮数 `13.00` vs `13.12`，纯求解 `v2` 开销 `5,028,063` vs `5,003,696`，仅差 `+0.49%`）。
   - 因此，`Full` 额外支付的在线 `L2/L1` 森林编译开销（流内 `1,453,819` `v2` + Stage S `820,842` `v2`）未能产生下游节省，导致 `Full` **流内总成本（`Scope 1`）比最强 `Flat` 高出 `+32.44%`（`v2`：`6,627,126` vs `5,003,696`，双侧精确符号翻转 `p = 0.0078125`）**，含 Stage S 独立总成本（`Scope 2`）高出 **`+32.57%`**。

---

## 二、三口径完整成本账本（`CostPolicy v1 & v2`，零遗漏）

| 统计口径 | 实验臂 / 阶段 | 逻辑调用数 | Prompt Tokens | Cached Tokens | Completion Tokens | `proxy_v1` (`U + 0.25C + O`) | `proxy_v2` (`U + 0.10C + 4O`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Stage S 共享盲源求解** | `Stage_S_Shared_Source`（`8` 世界） | `100` | `2,636,649` | `648,320` | `112,018` | `2,262,427` | `2,501,197` |
| **Stage S L2+L1 编译** | 仅计入 `Full`（`Flat` 计 `0`！） | `19` | `465,700` | `149,120` | `122,339` | `476,199` | `820,842` |
| **Scope 1：流内纯求解** | `Flat_Multiversion_Validated` | `210` | `5,587,244` | `1,521,024` | `196,365` | `4,642,841` | `5,003,696` |
| **Scope 1：流内纯求解** | `Full_Mechanism_Graph_Autonomous` | `210` | `5,621,095` | `1,510,528` | `191,632` | `4,679,831` | `5,028,063` |
| **Scope 1：流内自主守卫归纳** | `Full_Mechanism_Graph_Autonomous` | `8` | `141,428` | `15,488` | `4,439` | `134,251` | `145,244` |
| **Scope 1：流内 L2 升华编译** | `Full_Mechanism_Graph_Autonomous` | `20` | `1,260,291` | `555,904` | `173,462` | `1,016,825` | `1,453,819` |
| **Scope 1：流内总开销** | `Flat_Multiversion_Validated` | `210` | `5,587,244` | `1,521,024` | `196,365` | `4,642,841` | `5,003,696` |
| **Scope 1：流内总开销** | `Full_Mechanism_Graph_Autonomous` | `238` | `7,022,814` | `2,081,920` | `369,533` | `5,830,907` | `6,627,126` |
| **Scope 2：含 Stage S 独立总计** | `Flat_Multiversion_Validated` | `310` | `8,223,893` | `2,169,344` | `308,383` | `6,905,268` | `7,504,893` |
| **Scope 2：含 Stage S 独立总计** | `Full_Mechanism_Graph_Autonomous` | `357` | `10,125,163` | `2,879,360` | `603,890` | `8,569,533` | `9,949,165` |
| **Scope 3：封存 `run2` 物理消耗** | `run2` 全阶段合计 | `567` | `15,712,407` | `4,400,384` | `800,255` | `13,212,374` | `14,952,861` |
| **Scope 3：保留 `run1` 物理消耗** | 中断 `run1`（`hash_probe` 缺陷） | `114` | `2,967,251` | `590,720` | `202,288` | `2,726,499` | `3,244,728` |
| **Scope 3：战役全物理总消耗** | **`run1 + run2`（未知用量 = `0`）** | **`681`** | — | — | — | **`15,938,873`** | **`18,197,589`** (`19,682,201` tok) |

---

## 三、按因果事件分层的持续流表现（每臂 `64` 轮）

| 因果事件 | `Flat_Multiversion_Validated` 通过（零轮数） | `Full_Mechanism_Graph_Autonomous` 通过（零轮数） | 平均轮数（`Flat` vs `Full`） | 流内 `proxy_v2`（`Flat` vs `Full`） |
| --- | --- | --- | --- | --- |
| `N` | `8/8`（`8` 零轮） | `8/8`（`8` 零轮） | `0.00` vs `0.00` | `0` vs `0` |
| `G1` | `8/8`（`8` 零轮） | `8/8`（`8` 零轮） | `0.00` vs `0.00` | `0` vs `0` |
| `T` | `8/8`（`0` 零轮） | `8/8`（`0` 零轮） | `13.12` vs `13.00` | `2,336,391` vs `2,930,741` |
| `C` | `8/8`（`0` 零轮） | `8/8`（`0` 零轮） | `13.12` vs `13.25` | `2,306,450` vs `3,348,153` |
| `R` | `8/8`（`8` 零轮） | `8/8`（`8` 零轮） | `0.00` vs `0.00` | `0` vs `0` |
| `U` | `8/8`（`8` 零轮） | `8/8`（`8` 零轮） | `0.00` vs `0.00` | `0` vs `0` |
| `G2` | `8/8`（`8` 零轮） | `8/8`（`8` 零轮） | `0.00` vs `0.00` | `0` vs `0` |
| `M` | `7/8`（`7` 零轮） | `7/8`（`7` 零轮） | `0.00` vs `0.00` | `0` vs `0` |

> **关于唯一私有未通过轮次（`alpha_m96_w03` 第 8 步 `M`，两臂同为 `7/8`）的法证说明**：
> 在 `alpha_m96_w03_batch_08`（`M`）中，`Flat` 与 `Full` 均零轮通过公共门控，且在私有种子上真实预测误差极低（如种子 `973862` 上 `strain_mae = 0.00956`、`rate_mae = 0.00440`、`vp_err = 0.00044`），但在私有种子 `973861`（`preload_scale = 0.55`）上触发了 `weights_prediction_handoff_mismatch`，原因是 `_build_m96_alpha_task` 私有一致性核查行将 `z_m` 初始化为 `np.zeros((N, 2))` 而非 `ds["test"]["initial_memory_states"]`（导致 `rem_c - rem_r * t` 超过 `0.08`）。两臂在同一私有种子上表现完全一致（`7/8` vs `7/8`），配对差值为 `0`。

---

## 四、逐世界配对差值（`Full - Flat`）与自主守卫/多父归纳结果

| 世界（配对簇） | `Δ 通过` | `Δ 零轮` | `Δ 轮数` | `Δ 流内 v1` | `Δ 流内 v2` | Step C 自主归纳守卫特征 | 验证通过的多父校准根节点 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `alpha_m96_w01` (`pair_01`) | `+0` | `+0` | `-1` | `+85,037` | `+133,969` | `max_history_and_hold_plus_residual_state_excess` | `alpha_wing_a_calibration,alpha_wing_b_calibration` |
| `alpha_m96_w02` (`pair_02`) | `+0` | `+0` | `+3` | `+231,609` | `+280,955` | `max_history_and_hold_plus_residual_state_excess` | `alpha_wing_a_calibration,alpha_wing_b_calibration` |
| `alpha_m96_w03` (`pair_03`) | `+0` | `+0` | `-1` | `+184,783` | `+246,739` | `max_history_and_hold_plus_residual_state_excess` | `alpha_wing_a_calibration,alpha_wing_b_calibration` |
| `alpha_m96_w04` (`pair_04`) | `+0` | `+0` | `+2` | `+312,415` | `+390,498` | `max_history_and_hold_plus_residual_state_excess` | `alpha_wing_a_calibration,alpha_wing_b_calibration` |
| `beta_m96_w01` (`pair_01`) | `+0` | `+0` | `-4` | `-36,691` | `+10,452` | `combined_nonseparable_or_vortex_history_excess` | `beta_wing_a_calibration,beta_wing_b_calibration` |
| `beta_m96_w02` (`pair_02`) | `+0` | `+0` | `-1` | `+73,110` | `+115,549` | `combined_nonseparable_or_vortex_history_excess` | `beta_wing_a_calibration,beta_wing_b_calibration` |
| `beta_m96_w03` (`pair_03`) | `+0` | `+0` | `-1` | `+156,340` | `+228,066` | `combined_nonseparable_or_vortex_history_excess` | `beta_wing_a_calibration,beta_wing_b_calibration` |
| `beta_m96_w04` (`pair_04`) | `+0` | `+0` | `+3` | `+181,463` | `+229,825` | `combined_nonseparable_or_vortex_history_excess` | `beta_wing_a_calibration,beta_wing_b_calibration` |

---

## 五、模块级取舍结论与架构收敛建议

1. **高确定性保留（高性价比、经受严格证伪检验的核心机制）**：
   - **单跳规范求解器导出 + 留出验证分翼参数适配（`F01 + F03`）**：以 `0` Token 成本直接消去 `47/64`（`73.4%`）的持续流重复推理轮次。
   - **按需冲突触发的轻量级自主对比守卫归纳（`M9.6` 去除了 `L2/L1` 重编译后的纯守卫模块）**：每次机制分叉仅耗费约 `1.8` 万 `v2` Token（占求解成本 `+2.90%`），在 `8/8` 世界实现零人工硬编码的 `100%` 历史守卫与多父依赖识别；但应当改为**惰性冲突触发（Lazy-on-Mutual-Incompatibility）**——当新求解器 `C_new` 在所有历史正例上已天然兼容（`C1 ⊃ C0` 嵌套扩展）时直接单探针复用、无需调用守卫归纳；仅当 `C_new` 在历史正例上失败（即两分支互斥、非嵌套）时才触发对比守卫归纳。
2. **明确裁撤/停用（经受控归因证伪的净负向 Token 税模块）**：
   - **流内在线 `L2/L1` 哲学森林升华编译与跨族算子虫洞打包（`compile_episode_dialectical_sublation` + `sublate_autogenous_forest_into_compositional_primitives`）**：占用了 `Full` 流内额外开销的 `91.4%`（`1,453,819 / 1,623,430` `v2`）及 Stage S 额外 `820,842` `v2`，而在跨域 `T` 任务上的真实执行与保留率均为 `0/8`。
