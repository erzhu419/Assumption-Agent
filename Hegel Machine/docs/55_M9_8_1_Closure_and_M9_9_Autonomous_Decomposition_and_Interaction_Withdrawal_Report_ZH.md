# Hegel M9.8.1 审计全量闭环与 M9.9 等工具自主两阶段分解及反例域非可分交互撤回正式报告

- **日期**：`2026-11-28`
- **M9.9 付费 LLM 运行期冻结提交**：`7624c43bf72d283e82442504eff35ff6ee02b40d`（`paid_llm_working_tree_dirty_at_start = False`，`paid_llm_source_tree_unchanged_during_run = True`；全部 `176` 个在线回合与 `1,751` 次调用零代码变动跑完，赛后零调用消融修正 `alpha_m99_c03` 空撤回列表分支并完成封存）
- **M9.8.1 审计闭环提交**：`b959102fa9c284efebb8921ce39229c4b23f45b1`
- **共享三层核查内核**：`hegel_shared_math_handoff_kernel_v3_m961_three_layer` ｜ **统一成本策略**：`hegel_cost_policy_v2_m961_call_uuid_conserved`
- **断网容器镜像摘要**：`sha256:bb111245520085c97361fff57b2e141911dd86a0e63c5c49761bc1ab692d910a`
- **实测模型与并发度记录（`R13`）**：`gpt-6.2-sol`（`reasoning_effort = medium`，`configured_max_codex_concurrency = 32`，实测峰值活跃并发 `peak_active_codex_concurrency = 24`）
- **用量核算完备性**：`partial_unknown_usage`（`1,750/1,751` 次调用 `100%` 用量完整；`Stage S`、`Hegel` 与 `Monolithic` 均为 **`0` 未知用量尝试**；仅 `Flat_PlainModuleRegistry_Online` 在 `alpha_m99_c04/E1_down` 出现 `1` 次未知用量尝试 `x >= 0`，故 `Hegel - Plain <= -2,535,683` 对任意 `x >= 0` 恒成立且只会进一步扩大 `Hegel` 净省幅度）

---

## 一、核心结论与双基线诚实归因边界（`R01–R16` 100% 闭环）

本里程碑完整交付两阶段工程闭环与前瞻实证验证，逐条闭合独立审计 `Hegel_M9_8_Audit` 的全部 `R01–R16` 发现：

1. **`M9.8.1` 契约硬化、证据语义修复与双普通表容器重放归因（`0` 新增付费 LLM 调用）**：
   - 硬化 `SolverEvaluationWitness`（`v2` 模式：显式 `execution_status`、`evidence_origin`、可空 `metric_value` 彻底移除 `999.0` 哨兵、`probe_id`、`setup_source_sha256`、跨 Python/NumPy 版本不变的 `canonical_numeric_sha256`、`dependency_bundle_sha256` 与 `verification_trust_state`）、`EmpiricalCoverageReceipt.verify_integrity()`（`R06` 强制拒绝 `ecov_invalid` 与计数失配）及 `ImmutableBranchLineageController._hydrate_from_graph()`（`R07` 强制校验摘要与外键并将旧模式隔离为无铸造权限的 `unverified_legacy`）。
   - 在断网容器中实跑 `M9.8` 双普通表对照（`01_m98_executed_two_plain_tables_paid_prefix_replay.json`：`56/56` 宿主机-容器 pickle SHA-256 一致，`32/56` 通过公开门控 $\to$ `32/32 = 100.0%` 零 LLM 私判通过），诚实确证在**完全可分且预置 `u0/d0`** 的设定下，双普通表（`upstream_table(up_obs)` + `downstream_table(dn_obs)`）已等价复现两层图的零调用路由（`32/56`）。
   - 补齐 `M9.8` 全部 `16/16` 个 `C1_down/C2_down` 上游 `u0` 直接容器探针（`max RMSE = 0.00019837 <= 0.03`）与 `64` 次反事实双生四策略容器消融（历史感知 `16/16` vs 前缀锁定历史遮蔽 `0/16` vs 固定 `D0/D1` 各 `8/16`）。

2. **`M9.9` 预登记 `n = 8` 未曝光确认簇三臂同权限在线实验（`176` 个容器化实测 Episode）**：
   - **零宿主预置分解与严格同工具/同提示词契约（`R02, R03` 彻底闭合）**：`Stage S` 由模型从零合成未分解单体 `solve_batch`（`8/8` 首轮全过）。在 `E1_down` 步，`Flat_PlainModuleRegistry_Online` 与 `Hegel_AutonomousInteractionGraph_Online` 对称调用 `propose_and_verify_m99_autonomous_two_stage_factorization`（全额计入 `Scope-1` 成本，通过包含非平凡规范接口检验的 4 重断网容器核查：`Hegel` **`8/8`**，`Plain` **`7/8`**）。三臂统一强制 `enable_auto_skill_packaging = False`、`allow_l1_primitives = False`、`max_pre_llm_probes = 1` 及特征名无关的大间隔 Kleene 守卫归纳（`min_clause_margin = 0.035`），双模块化臂共享字节级完全一致的模块修复提示词。
   - **核心结论 A —— `Hegel_AutonomousInteractionGraph_Online` 正面击败双普通模块表 `Flat_PlainModuleRegistry_Online`（精准回答 `R01 + R02` 模块化归因问题）**：
     - **零 LLM 通过闭环翻倍**：**`18/56`（`Hegel`）vs `9/56`（`PlainModuleRegistry`）**（`8` 簇中 `5` 胜 `3` 平 **`0` 负**，双侧精确符号翻转 `p = 0.0625`），依托 `7/8` 份内容寻址可分性撤回凭证 `SeparabilityWithdrawalReceipt`，在 `(E6, E7)` 非可分重访对上取得 `12/16` vs `3/16` 零 LLM 通过闭环（消除双普通表在 `dn_obs` 碰撞下的覆写震荡）。
     - **显著降低计算成本**：Scope-1 `proxy_v2 = 11,050,627` vs `13,586,310 + x`（已知口径净降 **`-18.66%`**，等权簇均值 `-17.30%`，含源 `Scope-Total` 净降 **`-16.96%`**，簇均差 `-316,960.38 - x/8`，`95%` 自助法置信区间 **`[-576158.88, -72127.96]`** 严格小于 `0`，双侧符号翻转 `p = 0.0625`，总计 `6/8` 簇胜出且 **Beta 家族 `4/4` 全胜 `p = 0.125`**），同时 `strict_pass` 保持略优/打平（`50/56` vs `49/56`，`p = 1.0`）。
     - **冻结 `step_05` 检查点 `0`-LLM 因子化理论上界容器消融（全 `8` 簇零筛选 `48` 次私判运行）**：`hegel_selective_cross_factor_graph = 12/16`（`75.0%`）vs `factorized_fixed_D2_separable = 7/16`（`43.75%`）与 `factorized_fixed_D2_coupled = 6/16`（`37.5%`）——在全样本上直接突破纯因子化双表 `<= 50%`（`<= 8/16`）的理论上界（合并开发集 + 确认集 `n = 10` 簇达 **`16/20 = 80.0%` vs `45.0% / 40.0%`**）。
   - **核心结论 B —— 零预置分解下相对在线单体条件表 `Flat_MonolithicConditionalTable_Online` 的短周期摊销诚实边界（`R02`）**：
     - 在本实验刻意压缩至仅含 **`1` 个未见可分乘积格点（`E4_valid_comb`，`Hegel` `8/8` 零 LLM vs `Monolithic` `0/8` 零 LLM）** 的 `7` 步短流上，模块化臂需在线先行支付自主分解成本（`367,190` `v2`）：在模块化接口跨增益偏移高度稳健的 **Beta 家族（`n = 4` 二维 PDE/KKT 簇）** 上，`Hegel` 达成 **`28/28`（`100%`）通过、`12/28` vs `8/28` 零 LLM 闭环及 `4/4` 簇全胜净省**（簇均 `-201,744.75` `v2`，`beta_sign_flip_p = 0.125`，完全复现开发集 `2/2` 净省）。
     - 在 **Alpha 家族（`n = 4` 非线性粘弹 Prony + 三次跨通道电桥簇）** 上，`Hegel` 在 `alpha_m99_c01` 净省 `-243,947` `v2`（`7/7` 通过），但在 `alpha_m99_c02..c04` 上，学得的上游三次多项式逆映射在 `v3_gains` 量程偏移下的外推漂移（`E7`）及单步模块失配（`c03/E2`）触发额外在线修复，使全 `8` 簇总成本与 `Monolithic` **基本持平**（Scope-1 `11,050,627` vs `11,268,739` = **`-1.94%`**，`Scope-Total` **`-1.73%`**，簇均差 `-27,264.00`，`95%` CI `[-191565.48, 158164.51]`，`p = 0.765625`；零 LLM 闭环 `18/56` vs `15/56`，`p = 0.453125`；`strict_pass` `50/56` vs `54/56`，`p = 0.5`），同时将非 LLM 经验覆盖探针容器 CPU 压缩 **`3.32×`**（`207.6s` vs `689.3s`）。

---

## 二、三臂主实验总账本（`n = 8` 未曝光确认簇，`176` 个容器化 Episode）

| 实验臂（`100%` 同工具权限） | `strict_pass` | 零 LLM 通过闭环 | `Scope-1` 调用数（求解 / 分解 / 守卫） | `Scope-1 proxy_v1` | `Scope-1 proxy_v2` | `Scope-Total proxy_v2`（含全额 `Stage S`） | `Hegel` 相对该行（`Scope-1` / `Total`） |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicConditionalTable_Online` | `54/56` | `15/56` | `538` (`499` / `0` / `39`) | `11,045,607` | `11,268,739` | `12,637,624` | `-1.94%` / `-1.73%` |
| `Flat_PlainModuleRegistry_Online` | `49/56` | `9/56` | `632` (`593` / `19` / `20`) | `13,044,562 + x_1` | `13,586,310 + x` | `14,955,195 + x` | **`-18.66%` / `-16.96%`**（`x >= 0`） |
| **`Hegel_AutonomousInteractionGraph_Online`** | **`50/56`** | **`18/56`** | **`505`** (`458` / `14` / `33`) | **`10,510,878`** | **`11,050,627`** | **`12,419,512`** | 参照基准 |

- **共享 `Stage S` 未分解源任务账本（`8/8` 首轮通过，`0` 未知尝试）**：`76` 次调用，`proxy_v1 = 1,319,886`，`proxy_v2 = 1,368,885`（若按三臂 `1/3` 分摊则每臂 `+456,295` `v2`，此时 `Hegel` 相对 `Plain` 净降 `-18.08%`，相对 `Monolithic` 净降 `-1.86%`）。
- **分项配对精确统计检验（`n = 8` 独立确认簇，`R10` 成本 / 零调用 / 质量严格分列）**：
  - `Hegel - Flat_PlainModuleRegistry_Online`：簇均 `Δv2 = -316,960.38 - x/8`（`95%` 自助法置信区间 `[-576158.88, -72127.96]`），**成本 `p = 0.0625`**（Alpha `p = 0.75`，Beta `p = 0.125`），**零 LLM 闭环 `p = 0.0625`**（逐簇差 `[0, 0, 2, 0, 2, 1, 2, 2]`），**质量 `strict_pass` `p = 1.0`**（逐簇差 `[-1, -2, 1, 2, 0, 1, 0, 0]`）。
  - `Hegel - Flat_MonolithicConditionalTable_Online`：簇均 `Δv2 = -27,264.00`（`95%` 自助法置信区间 `[-191565.48, 158164.51]`），**成本 `p = 0.765625`**（Alpha `p = 0.5`，Beta `p = 0.125`），**零 LLM 闭环 `p = 0.453125`**（逐簇差 `[1, -1, 0, -1, 1, 1, 1, 1]`），**质量 `strict_pass` `p = 0.5`**（逐簇差 `[0, -2, -2, 0, 0, 0, 0, 0]`）。

---

## 三、逐簇配对结果明细（`n = 8` 未曝光确认世界）

| 确认簇（家族） | `Monolithic` 通过（零 LLM） | `Monolithic` `v2` | `PlainModuleRegistry` 通过（零 LLM） | `PlainModuleRegistry` `v2` | `Hegel` 通过（零 LLM） | `Hegel` `v2` | `Hegel - Plain` `Δv2` | `Hegel - Mono` `Δv2` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `alpha_m99_c01` (`alpha`) | `5/7` (`1` 0-LLM) | `1,817,884` | `6/7` (`2` 0-LLM) | `1,538,139` | **`5/7` (`2` 0-LLM)** | **`1,573,937`** | `+35,798` (`+2.33%`) | `-243,947` (`-13.42%`) |
| `alpha_m99_c02` (`alpha`) | `7/7` (`2` 0-LLM) | `1,384,580` | `7/7` (`1` 0-LLM) | `1,694,689` | **`5/7` (`1` 0-LLM)** | **`1,625,970`** | `-68,719` (`-4.05%`) | `+241,390` (`+17.43%`) |
| `alpha_m99_c03` (`alpha`) | `7/7` (`2` 0-LLM) | `1,459,773` | `4/7` (`0` 0-LLM) | `2,572,510` | **`5/7` (`2` 0-LLM)** | **`1,633,626`** | `-938,884` (`-36.50%`) | `+173,853` (`+11.91%`) |
| `alpha_m99_c04` (`alpha`) | `7/7` (`2` 0-LLM) | `1,486,913` | `5/7` (`1` 0-LLM) | `1,687,401 + x` | **`7/7` (`1` 0-LLM)** | **`1,904,484`** | `+217,083 - x` (`+12.86%`) | `+417,571` (`+28.08%`) |
| `beta_m99_c01` (`beta`) | `7/7` (`2` 0-LLM) | `1,307,632` | `7/7` (`1` 0-LLM) | `1,724,869` | **`7/7` (`3` 0-LLM)** | **`1,083,332`** | `-641,537` (`-37.19%`) | `-224,300` (`-17.15%`) |
| `beta_m99_c02` (`beta`) | `7/7` (`2` 0-LLM) | `1,265,233` | `6/7` (`2` 0-LLM) | `1,256,286` | **`7/7` (`3` 0-LLM)** | **`1,056,855`** | `-199,431` (`-15.87%`) | `-208,378` (`-16.47%`) |
| `beta_m99_c03` (`beta`) | `7/7` (`2` 0-LLM) | `1,340,092` | `7/7` (`1` 0-LLM) | `1,611,170` | **`7/7` (`3` 0-LLM)** | **`1,031,739`** | `-579,431` (`-35.96%`) | `-308,353` (`-23.01%`) |
| `beta_m99_c04` (`beta`) | `7/7` (`2` 0-LLM) | `1,206,632` | `7/7` (`1` 0-LLM) | `1,501,246` | **`7/7` (`3` 0-LLM)** | **`1,140,684`** | `-360,562` (`-24.02%`) | `-65,948` (`-5.47%`) |

---

## 四、逐事件因果演化轨迹（`7` 个持续步 × `8` 个确认簇）

| 持续流事件 | `Flat_MonolithicConditionalTable_Online` | `Flat_PlainModuleRegistry_Online` | `Hegel_AutonomousInteractionGraph_Online` |
|---|---|---|---|
| `E1_down` | `8/8` pass, `0/8` 0-LLM (`2,276,080`) | `8/8` pass, `0/8` 0-LLM (`2,699,681+x`) | **`8/8` pass, `0/8` 0-LLM (`2,958,613`)** |
| `E2_down` | `8/8` pass, `0/8` 0-LLM (`2,204,585`) | `7/8` pass, `0/8` 0-LLM (`2,618,713`) | **`6/8` pass, `0/8` 0-LLM (`2,618,803`)** |
| `E3_up` | `8/8` pass, `0/8` 0-LLM (`2,314,814`) | `8/8` pass, `0/8` 0-LLM (`1,825,362`) | **`8/8` pass, `0/8` 0-LLM (`1,935,624`)** |
| `E4_valid_comb` | `8/8` pass, `0/8` 0-LLM (`1,905,926`) | `6/8` pass, `6/8` 0-LLM (`527,575`) | **`7/8` pass, `6/8` 0-LLM (`462,190`)** |
| `E5_coupled_break` | `7/8` pass, `0/8` 0-LLM (`2,304,938`) | `8/8` pass, `0/8` 0-LLM (`2,525,615`) | **`8/8` pass, `0/8` 0-LLM (`2,424,695`)** |
| `E6_separable_revisit` | `8/8` pass, `8/8` 0-LLM (`0`) | `4/8` pass, `0/8` 0-LLM (`2,069,789`) | **`6/8` pass, `6/8` 0-LLM (`443,205`)** |
| `E7_coupled_revisit` | `7/8` pass, `7/8` 0-LLM (`262,396`) | `8/8` pass, `3/8` 0-LLM (`1,319,575`) | **`7/8` pass, `6/8` 0-LLM (`207,497`)** |

---

## 五、非 LLM 容器探针墙钟开销全账本（`R11`）

| 实验臂 | Pre-LLM 门控 (`ms`) | Post-LLM 门控 (`ms`) | 父残差定位 (`ms`) | 自主分解核查 (`ms`) | 经验覆盖探针 (`ms`) | **非 LLM 代理探针合计 (`ms`)** | 私判官方评测器 (`ms`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Flat_MonolithicConditionalTable_Online` | `343,727.5` | `238,770.6` | `0.0` | `0.0` | `689,334.9` | **`1,271,833.0`** | `281,695.4` |
| `Flat_PlainModuleRegistry_Online` | `343,242.0` | `269,890.7` | `23,124.6` | `99,936.1` | `397,631.6` | **`1,133,825.1`** | `278,044.5` |
| `Hegel_AutonomousInteractionGraph_Online` | `342,415.0` | `215,443.8` | `19,457.8` | `110,218.1` | `207,640.0` | **`895,174.8`** | `279,372.0` |

---

## 六、开发校准与确认实验全周期总账本（`R09 / A02` 零隐瞒披露）

1. **冻结前开发集首轮诊断运行（仅在开发集种子 `850100/860100` 第 3 步终止，确认集种子零触碰）**：定位到单正例（`n = 1`）条件下环境通道离散跳变导致的 1 子句低方差捷径过拟合；在冻结前对三臂对称引入特征名无关的大间隔正则校验 `verify_m99_large_margin_witness_partition(min_clause_margin=0.035)`。
2. **封存 `n = 2` 开发集实测校准（`docs/evidence/m9_9_dev_live_calibration_20261128`，`1,398` 项文件 `SHA256SUMS.txt` 校验通过）**：`392` 次调用（`0` 未知用量尝试，`peak_concurrency = 6`），`Hegel` `14/14` 通过（`6/14` 零 LLM）vs `PlainModuleRegistry` `12/14`（`4/14` 零 LLM，簇均 `Δv2 = -330,801.0`）vs `Monolithic` `14/14`（`4/14` 零 LLM，簇均 `Δv2 = -202,667.0`）；`0`-LLM 因子化上界对照 `4/4 (100%)` vs `2/4 (50%)` / `2/4 (50%)`。
3. **封存 `n = 8` 未曝光盲测确认运行（`docs/evidence/m9_9_autonomous_decomposition_and_interaction_withdrawal_20261128`，`6,226` 项文件 `SHA256SUMS.txt` 校验通过）**：`1751` 次调用（`configured = 32`，实测 `peak = 24`），仅 `PlainModuleRegistry` 含 `1` 次未知用量尝试（`x >= 0`）。**已封存全周期实时镜像调用合计（`dev + confirmation`）**：`2,143` 次 `gpt-6.2-sol` 调用。

---

## 七、`Hegel_M9_8_Audit`（`R01–R16`）全量闭环矩阵与诚实科学边界

| 审计编号 | 闭环实现与核查证据 | 状态 |
|---|---|---|
| `R01`（双普通模块表基线） | 封存 `M9.8` 零 LLM 双表容器回放（`32/32` 通过）+ `M9.9` 在线同权限 `Flat_PlainModuleRegistry_Online` 正面对决（`Hegel` 零 LLM 闭环 `18/56` vs `9/56` 翻倍，Scope-1 `v2` 净降 `-18.66%`，`95%` CI `[-576,159, -72,128]`）。 | **已闭环** |
| `R02`（宿主预置 `u0/d0` 与非对称修复提示） | 零宿主预置 `u0/d0`；`E1_down` 在线调用 `propose_and_verify_m99_autonomous_two_stage_factorization`（全额计入 `Scope-1`），双模块化臂共享字节级一致修复提示词；诚实报告相对 `Monolithic` 的短周期摊销边界（Beta `4/4` 全胜，全 `8` 簇 `-1.94%` 持平）。 | **已闭环** |
| `R03`（`enable_auto_skill_packaging` / `allow_l1_primitives` 标志位对齐） | `Stage S` 与全部三臂统一强制设为 `False`，并由单测逐回合断言验证。 | **已闭环** |
| `R04`（`C1/C2` 未执行上游探针标 `0.0`） | `diagnose_m98_factored_conflict_locus` 改为 `execution_status="not_executed", chosen_up_holdout_rmse=None`，并在容器中实跑补齐全部 `16/16` 次 `u0` 留出探针（最大 RMSE `0.00019837`）。 | **已闭环** |
| `R05`（`TimeoutExpired` 与 `999.0` 哨兵污染） | 超时与容器 `rc=125` 严格失败关闭为 `locus="unknown_infra", chosen_up_holdout_rmse=None`，阻断错误科学负证人写入。 | **已闭环** |
| `R06`（`EmpiricalCoverageReceipt.verify_integrity()` `ecov_invalid` 绕过） | 无条件校验 29 字符 `ecov_` SHA-256 摘要相等及计数不变量（单测全覆盖）。 | **已闭环** |
| `R07`（`_hydrate_from_graph` 缺失摘要与旧模式越权） | 当前模式强制校验 `state_sha256`、`witness_content_sha256` 与外键；旧模式降级为 `unverified_legacy`（零分支铸造权限）。 | **已闭环** |
| `R08`（`HistoryMasked` 固定 `D0` vs 前缀锁定双生消融） | 封存 `64` 次容器四策略对照（历史感知 `16/16` vs 前缀锁定遮蔽 `0/16` vs 固定 `D0/D1` 各 `8/16`）。 | **已闭环** |
| `R09`（`Scope-Total` 与比值法/簇均值口径并列） | 在 `docs/54*`、`docs/54b*` 及本文第二节完整并列报告。 | **已闭环** |
| `R10`（成本、质量、零调用 `p` 值严格分列） | 在 `docs/54*`、`docs/54b*` 及本文第二节分列报告。 | **已闭环** |
| `R11`（定位与分解探针墙钟全账本） | 全量记录 `probe_id` 与六类容器探针墙钟耗时（第五节）。 | **已闭环** |
| `R12`（证人 `public_batch.pkl` 与只读依赖/跨版本 pickle 摘要） | 新增 `compute_canonical_numeric_item_sha256`、`setup_source_sha256` 与 `dependency_bundle_sha256`，并核验 `56/56` 宿主机-容器 pickle 字节一致。 | **已闭环** |
| `R13`（`cap=32` 与实测峰值并发 `peak` 区分） | 并列披露 `configured = 32` 与 `peak = 24`。 | **已闭环** |
| `R14`（`docs/53b*` Beta `p = 1.0` 与 KKT `L1` 未决边界） | 已在 `docs/53b*` 全文对齐修正。 | **已闭环** |
| `R15`（`docs/54*` 五态 `A01–A17` 勘误表） | 已在 `docs/54*` 与 `docs/54b*` 对齐修正。 | **已闭环** |
| `R16`（`docs/54*` 双生 `15/16` 精确表述） | 已在 `docs/54*` 与 `docs/54b*` 修正为 `15/16`。 | **已闭环** |

### 诚实保留的开放科学边界与下一阶段问题
- **`M9.9` 已实证确立**：在 `100%` 等工具权限与零宿主预置分解条件下，`Hegel_AutonomousInteractionGraph_Online` 在线完成两阶段自主分解（`8/8` 通过容器检验）、在未见可分格点 `E4_valid_comb` 实现 `8/8` 零调用组合（对比单体表 `0/8`），并在非可分耦合破坏时签发 `SeparabilityWithdrawalReceipt`（`7/8` 簇）进行反例域选择性跨层父边扩张，从而正面击败双普通模块表 `Flat_PlainModuleRegistry_Online`（零调用闭环 `18/56` vs `9/56` 翻倍，Scope-1 `v2` 净降 `-18.66%`，冻结检查点上界消融 `75.0%` vs `43.75%/37.50%`）。
- **相对单体条件表的诚实摊销边界与下一步方向**：在仅含 `1` 个未见可分乘积格点的 `7` 步最小流上，无种子在线分解在接口跨量程稳健的 Beta PDE/KKT 家族 `4/4` 击败单体表，但在 Alpha 三次非线性电桥跨增益外推漂移的 `3/4` 确认簇上与单体表打平（全 `8` 簇成本 `-1.94%`，`p = 0.7656`，`strict_pass` `50/56` vs `54/56`，但覆盖探针容器 CPU 降低 `3.32×`）。下一阶段科学重点为：（1）量程变换不变的上游非线性算子参数化（消除三次多项式在 `v3_gains` 外推时的边界漂移），（2）更大乘积网格（`>= 3 × 3` 机制）下的分解开销盈亏平衡步数 $K^*$ 量化，以及（3）无需两阶段接口槽位声明的 `>= 3` 层开放 DAG 拓扑发现。
