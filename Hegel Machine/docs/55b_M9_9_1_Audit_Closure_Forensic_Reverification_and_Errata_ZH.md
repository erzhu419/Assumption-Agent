# Hegel M9.9.1 独立审计闭环、30 次断网容器交叉补证、取证根因校准与正式勘误报告（`Q01–Q14` + 原编号 `R01–R16`）

- **日期**：`2026-11-28`
- **被审计基线提交**：`d4b1cd7f06a08db8d967a43913003527d21a0354`（`Hegel_M9_9_Audit`）
- **本轮 `M9.9.1` 付费 LLM 调用数**：**`0`**（纯确定性契约硬化 + 真实断网 Docker 容器补证）
- **原封存证据包只读承诺**：`docs/evidence/m9_8_1_*`、`docs/evidence/m9_9_dev_live_calibration_20261128` 与 `docs/evidence/m9_9_autonomous_decomposition_and_interaction_withdrawal_20261128` 字节级零改动；本轮新增补证全部独立封存于 `docs/evidence/m9_9_1_audit_closure_and_forensic_reverification_20261128/`（含 `SHA256SUMS.txt`）。

---

## 一、`Hegel_M9_9_Audit`（`Q01–Q14`）逐项修复与勘误总表

| 编号 | 优先级 | 审计诊断问题 | `M9.9.1` 代码修复、容器补证与声明校准 | 状态 |
|---|---|---|---|---|
| `Q01` | `P1` | 分解时两格交叉约束来自推断。 | 历史补跑完成 30/30 格，其中 30 格为科学或域拒绝，匹配旧证人 8/8。新联合守卫使用实测格。 | 历史回放 |
| `Q02` | `P1` | alpha c02/E7 未过。 | 冻结下游的联合标准化与插补修正使误差从 0.071267 降到 0.004857。公开交换检查覆盖联合标准化的一项必要性质，冻结回放同时修改了插补方式，无法分出其独立贡献。 | 冻结组件归因 |
| `Q03` | `P1` | c04/E7 的守卫排除了已有的合格上游。 | 前缀组合 up_b02 + dn_b03 公开通过 True，私判通过 True。新回退的在线收益待测。 | 冻结候选回放 |
| `Q04` | `P1` | 延迟分解使旧单体无法调用。消融还把缺失的可分分支指向了耦合分支。 | 新执行路径支持归档回退，前缀消融显式记录缺失分支。原始实验保留当时实际执行的路由规则。 | 工程修复，收益待测 |
| `Q05` | `P1` | `--finalize-existing` 覆盖了摘要中的 `run_id`、`wall_sec` 与源码树哈希（`be7190fb...` vs 冻结树 `34b01d19...`）。 | Runner 新增一次性执行起止清单与独立后处理清单；在补证账本精确重建 `7624c43bf` 树哈希（`34b01d19ef125692...`）与原始镜像调用时间包络（确认 `3,718.765s`、开发 `2,259.938s`）。 | **已闭环（代码 + 来源账本）** |
| `Q06` | `P1` | 八个配对簇。 | Hegel 相对单体质量 p = 0.5，成本 p = 0.765625。相对 Plain 在 x = 0 时成本 p = 0.0625。阶梯使用同一组八个配对。 | 统计口径修正 |
| `Q07` | `P1` | 旧撤回记录可通过内部外键校验，即使签发时未留存科学负回执。 | 签发时留存 0/7 条联合科学回执，独立回放得到 7/7 条身份匹配的科学负例。新授权要求可验证回执。 | 原始记录与回放分别列账 |
| `Q08` | `P2` | 显式旧模式降权后二次 `save -> reload` 摘要不幂等；旧 `M9.8` 高 `r²` 推定定位仍可写下游科学负例。 | 迁移后规范化重算 `v2` 摘要与 `unknown_requires_reverification` 回执 ID（`snap_1 == snap_2` 往返幂等且零升权），并阻断 `compatibility_inference` 写入下游科学负例。 | **已闭环** |
| `Q09` | `P1` | 零预置源码被泛化表述为零预置分解/一般拓扑发现。 | 统一修正为**零预置模块源码、给定两阶段接口与父拓扑下的在线模块形成**（未知层数 DAG 另轨）。 | **已闭环** |
| `Q10` | `P1` | 冻结消融限制了固定选择器可用的特征。 | 在同一宿主重建的策略输入上，普通字典复现 16/16 个候选摘要。实际容器输入另行登记。 | 前缀回放 |
| `Q11` | `P1` | 理想未见格点公式被写成精确 LLM 成本定律。 | 修正为附条件格点计数 `N_{sep,unseen} = card((U x D) \ (R u X))`，未来需要实测质量约束下的 `ΔC(K)` 与 `K*`。 | **已闭环** |
| `Q12` | `P2` | 被取消的开发运行曾从总成本中遗漏。 | 找回 333 次调用，已知 8,220,922 v2，另有 6 次未结。总计 53904542 + x + y。 | 部分账单恢复 |
| `Q13` | `P2` | `E4/E5` 增益标签颠倒、`E1`（`7/8, 6/8`）与累计（`8/8, 7/8`）混写、墙钟耗时（总 `1.421×` / 覆盖子项 `3.320×`）误写为 CPU。 | 已在报告生成器、中英文 `docs/55_*` 与能力总入口同步修正。 | **已闭环** |
| `Q14` | `P2` | `R01–R16` 编号漂移及能力总入口 `54b` 坏链接。 | 在 `docs/55_*` 第七节按原 `R01–R16` 编号逐一对应（明确保留 `R14` 退役与 `R15` 开放边界），并修复 `docs/hegel_machine_reuse_hub_20261006.md` 中的 `54b` 链接。 | **已闭环** |

---

## 二、历史 `15` 次已接受分解共 `30` 个交叉格断网容器实跑补证（`Q01`）

- **封存账本**：`01_m99_q01_30_cross_diagonal_decomposition_probes.json`（总耗时 `146,840.91 ms`，`0` 次付费 LLM 调用）
- **`30` 次交叉公开门控实跑统计**：
  - 真实通过（`accepted = True`）：**`0/30`**
  - 真实科学/域拒绝（`scientific_or_numerical` / `domain_precondition`）：**`30/30`**
  - 结构/接口拒绝（`schema_or_interface`）：**`0/30`**
  - 超时或基础设施错误：**`0/30`**
  - 建立显式勘误替代映射的原 `Hegel` 未执行 `d0@E` 证人数：**`8/8`**

| 实验臂 | 确认簇（分解步） | 交叉核查格 | `probe_id` | 执行状态 | 是否通过 | 失败分类 | 耗时 (`ms`) | 替代原未执行证人 ID |
|---|---|---|---|---|---|---|---:|---|
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c01` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_207e31f71a9a4bfd` | `completed` | `False` | `scientific_or_numerical` | `4952.0` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c01` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_ebe609a20564467c` | `completed` | `False` | `scientific_or_numerical` | `4904.9` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c02` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_284af950f9464dcf` | `completed` | `False` | `scientific_or_numerical` | `4964.0` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c02` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_c4dfb183d1714fca` | `completed` | `False` | `scientific_or_numerical` | `4963.1` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c04` (`s2`) | `u0_plus_d0_on_current_E` | `xprobe_1e633a14a73d49c4` | `completed` | `False` | `scientific_or_numerical` | `5029.2` | — |
| `Flat_PlainModuleRegistry_Online` | `alpha_m99_c04` (`s2`) | `u0_plus_d1_on_Stage_S` | `xprobe_8c0458eeabb74ca8` | `completed` | `False` | `scientific_or_numerical` | `5432.3` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c01` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_7341a09e5b54400b` | `completed` | `False` | `scientific_or_numerical` | `4753.4` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c01` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_ca69051b451e46e5` | `completed` | `False` | `scientific_or_numerical` | `4766.6` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c02` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_54cfc758b4ff4a33` | `completed` | `False` | `scientific_or_numerical` | `4883.2` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c02` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_17979b37c7264b54` | `completed` | `False` | `scientific_or_numerical` | `4770.1` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c03` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_bd0601848b7c4811` | `completed` | `False` | `scientific_or_numerical` | `5085.5` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c03` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_c9c1a1f1d45a43fc` | `completed` | `False` | `scientific_or_numerical` | `5313.0` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c04` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_2da92e05d2564f54` | `completed` | `False` | `scientific_or_numerical` | `4771.7` | — |
| `Flat_PlainModuleRegistry_Online` | `beta_m99_c04` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_2b4eba4007e3482d` | `completed` | `False` | `scientific_or_numerical` | `4807.3` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c01` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_b83f6acb78b74a3c` | `completed` | `False` | `scientific_or_numerical` | `4937.4` | `wit_0002_bind_alpha_m99_c01_dn_b00_5a38161b` |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c01` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_3ca15e58450842ca` | `completed` | `False` | `scientific_or_numerical` | `4966.2` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c02` (`s2`) | `u0_plus_d0_on_current_E` | `xprobe_34560cf5d69b460f` | `completed` | `False` | `scientific_or_numerical` | `5121.8` | `wit_0002_bind_alpha_m99_c02_dn_b00_ed77e0c2` |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c02` (`s2`) | `u0_plus_d1_on_Stage_S` | `xprobe_4b152e625ff045b5` | `completed` | `False` | `scientific_or_numerical` | `5079.4` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c03` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_30803ee5863a46b0` | `completed` | `False` | `scientific_or_numerical` | `4545.1` | `wit_0002_bind_alpha_m99_c03_dn_b00_03b4a8a3` |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c03` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_a039987d7dc64097` | `completed` | `False` | `scientific_or_numerical` | `4537.6` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c04` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_47532080a84b48fc` | `completed` | `False` | `scientific_or_numerical` | `5119.5` | `wit_0002_bind_alpha_m99_c04_dn_b00_3b352674` |
| `Hegel_AutonomousInteractionGraph_Online` | `alpha_m99_c04` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_24384cd5773c41af` | `completed` | `False` | `scientific_or_numerical` | `5148.2` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c01` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_1cb070deb60e4a2e` | `completed` | `False` | `scientific_or_numerical` | `4623.3` | `wit_0002_bind_beta_m99_c01_dn_b00_dc9859ed` |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c01` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_be0d9b11609b4276` | `completed` | `False` | `scientific_or_numerical` | `4497.6` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c02` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_1375835636ba4d81` | `completed` | `False` | `scientific_or_numerical` | `5004.8` | `wit_0002_bind_beta_m99_c02_dn_b00_c95627f7` |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c02` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_5ea807d92f224f76` | `completed` | `False` | `scientific_or_numerical` | `4990.9` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c03` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_aa2aa8723a38492d` | `completed` | `False` | `scientific_or_numerical` | `4884.7` | `wit_0002_bind_beta_m99_c03_dn_b00_83808d89` |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c03` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_5664ada0fc764d48` | `completed` | `False` | `scientific_or_numerical` | `4955.8` | — |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c04` (`s1`) | `u0_plus_d0_on_current_E` | `xprobe_bbf71289896548e7` | `completed` | `False` | `scientific_or_numerical` | `4529.1` | `wit_0002_bind_beta_m99_c04_dn_b00_387320b1` |
| `Hegel_AutonomousInteractionGraph_Online` | `beta_m99_c04` (`s1`) | `u0_plus_d1_on_Stage_S` | `xprobe_c4d8b74456b443be` | `completed` | `False` | `scientific_or_numerical` | `4503.6` | — |

---

## 三、来源身份核对（`Q05`）、未知 `x` 精确符号翻转敏感性阶梯（`Q06`）与 `c04/E7` 容器验证（`Q03`）

1. **Git 源码树与执行时间包络核对（`Q05`）**：
   - 声明付费运行冻结提交 `7624c43bf72d283e82442504eff35ff6ee02b40d`（`2700` 个 `.py` 文件）：`34b01d19ef12569240f013bd137e5818c449d68bdcb2ebb285697099c1234cdd`
   - `m9_9_continual_summary.json` 记录的赛后补算树哈希：`be7190fbfab8df60c5da023c3133f59fa0f71c6dde95ebc967367de3fd9d40bc`
   - 原始镜像调用时间包络下界：确认集 `3,718.765s`（运行前缀 `m99_confirmation_20261127T193020Z` vs 赛后汇总 `219.26s`）；开发集 `2,259.938s`（运行前缀 `m99_dev_20261127T183538Z`）。
2. **`alpha_m99_c04 / E7` 前缀已有模块 `bind_alpha_m99_c04_up_b02` 断网容器验证（`Q03`）**：
   - 直接上游 `56/24` 留出探针：`execution_status = completed`，**`chosen_up_holdout_rmse = 0.00007655`**（`<= 0.03`）。
   - 组合 `(bind_alpha_m99_c04_up_b02 + bind_alpha_m99_c04_dn_b03)` 公开三层门控：**`accepted = True`**（`4977.58 ms`）。
   - 官方私判评测器（仅作事后诊断补证，不改动原确认集 `50/56` 得分与费用）：**`strict_pass = True`**（`4227.89 ms`）。
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
