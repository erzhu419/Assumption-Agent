# Hegel M9.9.1 独立审计闭环、30 次断网容器交叉补证、取证根因校准与正式勘误报告（`Q01–Q14` + 原编号 `R01–R16`）

- **日期**：`2026-11-28`
- **被审计基线提交**：`d4b1cd7f06a08db8d967a43913003527d21a0354`（`Hegel_M9_9_Audit`）
- **本轮 `M9.9.1` 付费 LLM 调用数**：**`0`**（纯确定性契约硬化 + 真实断网 Docker 容器补证）
- **原封存证据包只读承诺**：`docs/evidence/m9_8_1_*`、`docs/evidence/m9_9_dev_live_calibration_20261128` 与 `docs/evidence/m9_9_autonomous_decomposition_and_interaction_withdrawal_20261128` 字节级零改动；本轮新增补证全部独立封存于 `docs/evidence/m9_9_1_audit_closure_and_forensic_reverification_20261128/`（含 `SHA256SUMS.txt`）。

---

## 一、`Hegel_M9_9_Audit`（`Q01–Q14`）逐项修复与勘误总表

| 编号 | 优先级 | 审计诊断问题 | `M9.9.1` 代码修复、容器补证与声明校准 | 状态 |
|---|---|---|---|---|
| `Q01` | `P1` | 原分解验证器仅实跑对角正格（`u0+d0 @ S`, `u0+d1 @ E`），向 `d0d1_gind` 传入 `30` 个未执行交叉负约束并写入 `8` 条未执行 `Hegel` `d0@E` 科学负证人。 | 升级 `verify_m99_autonomous_decomposition_in_containers` 完整执行 `2×2` 四格（`matrix_2x2`）；新增强制由已完成 `ProbeReceipt` 构造证人的 `record_tested_witness_from_probe_receipt`；在断网容器中实跑补齐全部 **`30/30` 次历史交叉公开门控**（`0` 通过、`30` 科学/域拒绝、`0` 接口拒绝、`0` 超时/基础设施），并对原 `8` 条未执行证人建立显式 `supersedes_unexecuted_witness_id` 勘误映射。 | **已闭环（代码 + 30 次容器补证）** |
| `Q02` | `P1` | `alpha_m99_c02/E7`（`0.071267`）原归因于上游牛顿外推失准，实测 `up_b01` 载荷 RMSE 仅 `1.4e-5`，真因是下游 `_chamber_features(test)` 仅用 `test['ref_coupons']` 独立标准化且未加权填补 `NaN`。 | 在共享核查内核新增零真值泄露的公开跨切分交换不变性检查 `verify_alpha_pooled_ref_coupon_invariance`（确定性拦截 `c02/dn_b02` 并放行联合变换实现）；在 `docs/55_*` 第八节更正根因（仅修联合预处理即 `0.071267 -> 0.004857`）。 | **已闭环（代码 + 勘误）** |
| `Q03` | `P1` | `alpha_m99_c04/E7` 前缀 `step_06` 库中已有合格 `bind_alpha_m99_c04_up_b02`（`RMSE = 7.65e-5`），因过窄守卫 `up_stat_03 > 1.289054`（`E7` 为 `1.274262`）被排除。 | 在断网容器中实跑核验 `up_b02 + dn_b03`（直接探针 `RMSE = 0.00007655`，公开三层门控 `accepted = True`，官方私判 `strict_pass = True`；仅作事后归因补证，不改写原 `M9.9` 确认分数或费用）；对称 `1` 备选候选回退机制已接入 `M10.0`。 | **已闭环（容器实证 + M10.0 接入）** |
| `Q04` | `P1` | `c02` 延迟至 `E2_down` 分解后丢失 `E1_down` 可执行旧单体；`c03` 缺失 `d2_sep` 向 `E6` 传播。 | 在 `docs/55_*` 第八节校准说明，并在 `M10.0` 对双模块化臂对称实现单体可执行兜底与按需惰性下游回投（全额计入 `Scope-1`）。 | **已闭环（勘误 + M10.0 接入）** |
| `Q05` | `P1` | `--finalize-existing` 覆盖了摘要中的 `run_id`、`wall_sec` 与源码树哈希（`be7190fb...` vs 冻结树 `34b01d19...`）。 | Runner 分离只写一次的 `execution_manifest.json` 与 `finalization_manifest.json`；在补证账本精确重建 `7624c43bf` 树哈希（`915f469dc3f50a05...`）与原始镜像调用时间包络（确认 `3,718.765s`、开发 `2,259.938s`）。 | **已闭环（代码 + 来源账本）** |
| `Q06` | `P1` | 将 `Hegel - Plain`（`p = 0.0625`）写成显著获胜、将 `50/56` vs `54/56`（`p = 0.375`）写成质量持平。 | 全文收窄为精确 `p` 值与效应区间，单列未知 `x` 敏感性阶梯（`x > 53,450` 时 `p < 0.05`），并在 `M10.0` 预登记显式质量非劣门控。 | **已闭环** |
| `Q07` | `P1` | `hydrate_separability_withdrawals_from_graph` 未强制校验代码 SHA、`probe_id` 及历史碰撞正证人/当前负证人外键。 | 补齐完整 6 链路外键与时间先后校验（单测确认合成错 SHA/缺 probe 反例全部拒绝，正式 `7/7` 条撤回回执全部通过）。 | **已闭环** |
| `Q08` | `P2` | 显式旧模式降权后二次 `save -> reload` 摘要不幂等；旧 `M9.8` 高 `r²` 推定定位仍可写下游科学负例。 | 迁移后规范化重算 `v2` 摘要与 `unknown_requires_reverification` 回执 ID（`snap_1 == snap_2` 往返幂等且零升权），并阻断 `compatibility_inference` 写入下游科学负例。 | **已闭环** |
| `Q09` | `P1` | 零预置源码被泛化表述为零预置分解/一般拓扑发现。 | 统一修正为**零预置模块源码、给定两阶段接口与父拓扑下的在线模块形成**（未知层数 DAG 另轨）。 | **已闭环** |
| `Q10` | `P1` | 固定 `dn_obs` 路由基线与 `step_05` 消融被泛化为所有普通注册表上界。 | 封存普通 `dict` 在相同 `up+dn` 特征下 `16/16` 候选 SHA 一致回放，明确限定为固定 `dn_obs` 路由对比选择性条件扩展策略。 | **已闭环** |
| `Q11` | `P1` | 理想未见格点公式被写成精确 LLM 成本定律。 | 修正为附条件格点计数 `N_{sep,unseen} = |(U x D) \ (R u X)|`，并在 `M10.0` 实测质量约束下的 `ΔC(K)` 与 `K*`。 | **已闭环** |
| `Q12` | `P2` | 冻结前取消的首轮开发诊断运行未列入研发总账。 | 独立单列不可核验项 `C_cancelled > 0`（全周期研发总账 `45,683,620 + x + C_cancelled`，绝不填零）。 | **已闭环** |
| `Q13` | `P2` | `E4/E5` 增益标签颠倒、`E1`（`7/8, 6/8`）与累计（`8/8, 7/8`）混写、墙钟耗时（总 `1.421×` / 覆盖子项 `3.320×`）误写为 CPU。 | 已在报告生成器、中英文 `docs/55_*` 与能力总入口同步修正。 | **已闭环** |
| `Q14` | `P2` | `R01–R16` 编号漂移及能力总入口 `54b` 坏链接。 | 在 `docs/55_*` 第七节按原 `R01–R16` 编号逐一对应（明确保留 `R14` 退役与 `R15` 开放边界），并修复 `docs/hegel_machine_reuse_hub_20261006.md` 中的 `54b` 链接。 | **已闭环** |

---

## 二、历史 `15` 次已接受分解共 `30` 个交叉格断网容器实跑补证（`Q01`）

- **封存账本**：`01_m99_q01_30_cross_diagonal_decomposition_probes.json`（总耗时 `198,916.24 ms`，`0` 次付费 LLM 调用）
- **`30` 次交叉公开门控实跑统计**：
  - 真实通过（`accepted = True`）：**`0/30`**
  - 真实科学/域拒绝（`scientific_or_numerical` / `domain_precondition`）：**`30/30`**
  - 结构/接口拒绝（`schema_or_interface`）：**`0/30`**
  - 超时或基础设施错误：**`0/30`**
  - 建立显式勘误替代映射的原 `Hegel` 未执行 `d0@E` 证人数：**`8/8`**

| 实验臂 | 确认簇（分解步） | 交叉核查格 | `probe_id` | 执行状态 | 是否通过 | 失败分类 | 耗时 (`ms`) | 替代原未执行证人 ID |
|---|---|---|---|---|---|---|---:|---|
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c01` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_0add1ca13ca24097` | `completed` | `False` | `scientific_or_numerical` | `6804.4` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c01` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_660905d1d0874005` | `completed` | `False` | `scientific_or_numerical` | `6804.6` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c02` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_9601f2bb2ec74795` | `completed` | `False` | `scientific_or_numerical` | `6707.6` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c02` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_d24659a3178241e1` | `completed` | `False` | `scientific_or_numerical` | `6783.5` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c04` (`s2`) | `u0_plus_d0_on_current_E` | `xprobe_58dca380e2cd49fc` | `completed` | `False` | `scientific_or_numerical` | `6899.5` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c04` (`s2`) | `u0_plus_d1_on_Stage_S` | `xprobe_c335eeca7c114268` | `completed` | `False` | `scientific_or_numerical` | `6799.4` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c01` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_9e43c86c78ce470b` | `completed` | `False` | `scientific_or_numerical` | `6274.5` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c01` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_4bc69683c58e45eb` | `completed` | `False` | `scientific_or_numerical` | `6257.1` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c02` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_e23ac84a895443df` | `completed` | `False` | `scientific_or_numerical` | `6432.1` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c02` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_bd41a5b78d324e17` | `completed` | `False` | `scientific_or_numerical` | `6711.5` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c03` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_d2cccfe762164390` | `completed` | `False` | `scientific_or_numerical` | `6422.2` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c03` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_b68f0a6bcdce4ced` | `completed` | `False` | `scientific_or_numerical` | `6332.1` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c04` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_dcd2c3efe94d43c7` | `completed` | `False` | `scientific_or_numerical` | `6150.6` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c04` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_d063352996064440` | `completed` | `False` | `scientific_or_numerical` | `6221.2` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c01` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_bcb66ea656944fa9` | `completed` | `False` | `scientific_or_numerical` | `6621.6` | `wit_0002_bind_alpha_m99_c01_dn_b00_5a38161b` |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c01` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_231ce31383604457` | `completed` | `False` | `scientific_or_numerical` | `6590.1` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c02` (`s2`) | `u0_plus_d0_on_current_E` | `xprobe_bbc23825e55a42c4` | `completed` | `False` | `scientific_or_numerical` | `6512.7` | `wit_0002_bind_alpha_m99_c02_dn_b00_ed77e0c2` |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c02` (`s2`) | `u0_plus_d1_on_Stage_S` | `xprobe_10fa95a6a4be4936` | `completed` | `False` | `scientific_or_numerical` | `6611.7` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c03` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_7dee6323006c405b` | `completed` | `False` | `scientific_or_numerical` | `6589.3` | `wit_0002_bind_alpha_m99_c03_dn_b00_03b4a8a3` |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c03` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_28c126ae49d4445b` | `completed` | `False` | `scientific_or_numerical` | `6774.5` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c04` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_77d4acd78740449e` | `completed` | `False` | `scientific_or_numerical` | `6791.2` | `wit_0002_bind_alpha_m99_c04_dn_b00_3b352674` |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c04` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_080483c2d29f4a76` | `completed` | `False` | `scientific_or_numerical` | `6619.5` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c01` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_46a3596f507449a2` | `completed` | `False` | `scientific_or_numerical` | `6529.4` | `wit_0002_bind_beta_m99_c01_dn_b00_dc9859ed` |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c01` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_ec0c29b4ae0445ef` | `completed` | `False` | `scientific_or_numerical` | `6638.4` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c02` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_94606e0c92e54451` | `completed` | `False` | `scientific_or_numerical` | `7091.9` | `wit_0002_bind_beta_m99_c02_dn_b00_c95627f7` |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c02` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_0acbdf212bfd4d1f` | `completed` | `False` | `scientific_or_numerical` | `6786.9` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c03` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_45e352d518f346f6` | `completed` | `False` | `scientific_or_numerical` | `6783.1` | `wit_0002_bind_beta_m99_c03_dn_b00_83808d89` |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c03` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_0d3bdab1c2e240f9` | `completed` | `False` | `scientific_or_numerical` | `6858.0` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c04` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_c867ac26ea5549f6` | `completed` | `False` | `scientific_or_numerical` | `6666.3` | `wit_0002_bind_beta_m99_c04_dn_b00_387320b1` |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c04` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_b234458c7732465d` | `completed` | `False` | `scientific_or_numerical` | `6851.3` | — |

---

## 三、来源身份核对（`Q05`）、未知 `x` 精确符号翻转敏感性阶梯（`Q06`）与 `c04/E7` 容器验证（`Q03`）

1. **Git 源码树与执行时间包络核对（`Q05`）**：
   - 声明付费运行冻结提交 `7624c43bf72d283e82442504eff35ff6ee02b40d`（`2700` 个 `.py` 文件）：`915f469dc3f50a05c84e5065f1b03e99cdd80cf661bf4826dab16ff8e98af2c0`
   - `m9_9_continual_summary.json` 记录的赛后补算树哈希：`be7190fbfab8df60c5da023c3133f59fa0f71c6dde95ebc967367de3fd9d40bc`
   - 原始镜像调用时间包络下界：确认集 `3,718.765s`（运行前缀 `m99_confirmation_20261127T193020Z` vs 赛后汇总 `219.26s`）；开发集 `2,259.938s`（运行前缀 `m99_dev_20261127T183538Z`）。
2. **`alpha_m99_c04 / E7` 前缀已有模块 `bind_alpha_m99_c04_up_b02` 断网容器验证（`Q03`）**：
   - 直接上游 `56/24` 留出探针：`execution_status = completed`，**`chosen_up_holdout_rmse = 0.00007655`**（`<= 0.03`）。
   - 组合 `(bind_alpha_m99_c04_up_b02 + bind_alpha_m99_c04_dn_b03)` 公开三层门控：**`accepted = True`**（`6331.4 ms`）。
   - 官方私判评测器（仅作事后诊断补证，不改动原确认集 `50/56` 得分与费用）：**`strict_pass = True`**（`5192.31 ms`）。
3. **未知超时尝试 `x`（`call_uuid = 0391d702b0c547a488608847b5dff224`）双侧精确符号翻转 `p` 值阶梯**：

| `Plain` 未知尝试成本 `x` (`v2`) | 双侧精确符号翻转 `p`（`Hegel - Plain`） |
|---:|---:|
| `0` | `0.06250000` |
| `17,652` | `0.06250000` |
| `17,653` | `0.05468750` |
| `53,450` | `0.05468750` |
| `53,451` | `0.04687500` |
| `148,364` | `0.04687500` |
| `148,365` | `0.03906250` |
| `184,162` | `0.03906250` |
| `184,163` | `0.03125000` |
| `217,083` | `0.03125000` |
| `217,084` | `0.02343750` |
| `252,881` | `0.02343750` |
| `252,882` | `0.01562500` |

4. **非 LLM 代理探针流逝墙钟比值（`Q13`）**：全代理探针累计流逝墙钟比值 `Monolithic / Hegel = 1.420765×`（`1,271,833.04 ms / 895,174.81 ms`）；经验覆盖探针子项累计墙钟比值 `= 3.319855×`（`689,259.6 ms / 207,604.6 ms`）。
