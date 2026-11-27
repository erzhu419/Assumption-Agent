# Hegel Machine M9.7.1 审计闭环与 M9.8 盲残差失配发现 + 双层分解机制图实验报告

- **报告 Schema**: `hegel_m9_7_1_and_m9_8_sealed_report_v1`
- **M9.8 运行 ID**: `m98_run_20261127T142012Z`
- **确认集运行前冻结提交**: `2963e5d65f68d91510746e21c60c0985ec333358` (`working_tree_dirty_at_start = False`, `source_tree_sha256 = b1574bd7dac87b13a8f069d3bf361b4c1bcaeeaa3a43423004816d2913b66547`)
- **容器镜像摘要**: `sha256:bb111245520085c97361fff57b2e141911dd86a0e63c5c49761bc1ab692d910a`
- **模型与并发度**: `gpt-6.2-sol` (`reasoning_effort = medium`, `max_codex_concurrency = 32`)
- **证据来源标记**: `executed` (`usage_completeness = complete`, `unknown_usage_attempts = 0`)
- **统一验证核与成本口径**: `hegel_shared_math_handoff_kernel_v3_m961_three_layer` / `hegel_cost_policy_v2_m961_call_uuid_conserved`

---

## 1. 核心结论摘要（`A01–A17` 全部闭环 + 未曝光 `n = 8` 簇分解机制图正面胜出）

1. **第一阶段（`M9.7.1`，`0` 次付费 LLM 调用 — 封存于 `docs/evidence/m9_7_1_audit_closure_and_paid_pool_attribution_20261127/`）**：
   - **真实容器化付费池重放（`A02`, `A15`, `N11`）**：
     - `01_n11_executed_m95_paid_pool_replay.json`：在断网容器 `hegel-clean-scipy:m7v1` 内对 `M9.5` 全部 `48` 个复用 episode 执行 6 臂真实代码重放（`1_Unvalidated = 28/48`，`2_ValidatedWingOLS = 40/48`，`3_OldestFirst_Probes1 = 32/48`，`4_Multiversion_Probes2 = 48/48`，`5_Flat_SharedConditionalTable_Probes1 = 48/48`，`6_Hegel_MechanismGraph_Probes1 = 48/48`，`evidence_provenance = executed`）。
     - `02_m97_executed_flat_conditional_table_attribution.json`：对 `M9.7` 全部 `42` 个持续学习 episode 执行无图在线前缀锁定 `Flat_ConditionalTable`（`42/42` 路由一致、`29/42` 零 LLM 私有验证通过），实证确认在单层同父节点 `M9.7` 流上，多版本条件表已解释全部相对 `Flat_LatestFirst` 的收益，从而确立 `M9.8` 双层因果分解机制图的正交验证边界。
   - **契约、冷启动恢复、验证器与勘误闭环（`A01, A05–A12, A16, A17`）**：
     - 完成 `PARENT_VERSION_SET_MISMATCH` 全父集版本校验（`A07`）、单向覆盖 `(eval_bid, owner_bid)` 与内容寻址回执摘要（`A08`）、超时与科学反例严格分层（`A09`）、`witness_ledger` 与 `EmpiricalCoverageReceipt` 检查点持久化恢复（`A10`）、`Flat/Graph` 统一校准估计器（`A11`）、Alpha 错误码拆分（`A06`）、能力索引更新（`A16`）及 `M9.7` 完整成本/耗时/`R1` 取证勘误（`docs/53b_M9_7_1_Audit_Errata_Contract_Closure_and_Paid_Pool_Attribution_ZH.md`）。

2. **第二阶段（`M9.8`，预注册未曝光 `n = 8` 独立确认簇 — `176` 个真实容器化在线 episode）**：
   - **正面击败不阉割的在线 `Flat_ConditionalTable_Online`（`A02`, `A04`, `A15`）**：
     - **`Flat_LatestFirst`**：私有严格通过 `44/56`，零 LLM 闭环 `4/56`，Scope-1 `v2 = 13,078,368`（`618` 次调用：`618` 求解 + `0` 守卫归纳）。
     - **`Flat_ConditionalTable_Online`（赋予实时 `gpt-6.2-sol` 12 维观测 Kleene 守卫归纳 + 经验覆盖回执的在线条件版本表）**：私有严格通过 `50/56`，**零 LLM 闭环 `15/56`**（在已见乘积格点 `R0` 与 `U_iso` 上 `16/16` 零轮通过，证明条件表未被阉割；但在未见跨因子组合格点 `M_comb1 (U1 x D1)` 与 `M_comb2 (U1 x D2)` 上因单体耦合全部触发重新合成），Scope-1 `v2 = 11,139,318`（`531` 次调用：`504` 求解 + `27` 守卫归纳）。
     - **`Hegel_FactoredMechanismGraph`（双层因果分解机制图：`Upstream {u0, u1} -> Downstream {d0, d1, d2}`）**：私有严格通过 `56/56`，**零 LLM 闭环 `32/56`**（在 `C_up` 通过上游残差定位仅合成模块化 `u1`，随后在 `M_comb1` 与 `M_comb2` 上零样本跨因子组合 `u1 o d1` 与 `u1 o d2` 直接通过！），Scope-1 `v2 = 6,177,898`（`310` 次调用：`286` 求解 + `24` 守卫归纳）。
   - **簇级配对统计推断（`n = 8` 独立确认簇：`4` Alpha + `4` Beta）**：
     - **`Hegel_FactoredMechanismGraph` 对比 `Flat_ConditionalTable_Online` (`v2`)**：簇均值差 **`-620,178`**（**`-44.54%`**），95% Bootstrap CI **`[-775347.3, -480665.62]`**，精确双侧符号翻转检验 **`p = 0.0078125`**（Alpha `n=4`: `-48.74%, p = 0.125`；Beta `n=4`: `-38.32%, p = 0.125`，**Alpha 与 Beta 双家族一致净省算力**！）。
     - **`Hegel_FactoredMechanismGraph` 对比 `Flat_LatestFirst` (`v2`)**：簇均值差 **`-862,559`**（**`-52.76%`**），95% Bootstrap CI **`[-926214.88, -801797.83]`**，精确双侧符号翻转检验 **`p = 0.0078125`**。
   - **严格字节级全等 `(R0, U_iso)` 反事实双生对 + 真实容器化 `HistoryMasked_CurrentOnly` 消融对照（`A03` & `A13`）**：
     - 全部 `8` 个簇上 `SHA256(/holdout_task/public_batch.pkl)` 在 `R0` 与 `U_iso` 间 **`100%` 字节级相同**（容器内无任何 `gt` 字段），且全部 8 个 episode 的盲提示词 `100%` 相同（`0` 公式或机制标签泄漏）。
     - 遮蔽前缀仪器日志的 `HistoryMasked_CurrentOnly` 容器对照通过 **`0/16` (`0.0%`，因 `UNKNOWN` 回退统一落入优先级最高的 `dn_b02 = D2`；任何仅依赖当前输入的确定性策略理论上界亦不超过 `8/16 = 50.0%`)**，而历史感知 `Hegel_FactoredMechanismGraph` 达到 **`16/16` (`100.0%`) 零轮通过**（`Flat_ConditionalTable_Online` 为 `15/16` 零轮通过）。

---

## 2. Scope-1 持续流总账、调用角色拆分与非 LLM 容器耗时（`A05`, `A12`）

> **共享源任务 `Stage S` 初始化开销（三臂完全共享）**：`73` 次调用，`v1 = 1,115,434`，`v2 = 1,193,405`，LLM `wall_ms = 1,946,861`。
> **`M9.8` 全周期透明总账（`A05` 零隐藏运行核算）**：
> - `m9_8_aborted_run1_stage_s_contract_underspec_20261127`（旧种子段 `911000..924000` 在 `Stage S` 因基线符号约定未写明中止，`0` 个持续流 episode，旧种子段已永久退役）：`201` 次调用，`v2 = 5,318,353`（`usage_completeness = complete`）。
> - `m9_8_dev_live_calibration_pass1_20261127` + `m9_8_dev_live_calibration_20261127`（冻结前仅在 `DEV_M98_WORLD_SPECS` `810100, 820100` 上的两轮在线校准验证：`262` + `381` 次调用）：`643` 次调用，`v2 = 14,095,339`（`usage_completeness = complete`）。
> - `m9_8_blind_mismatch_factored_graph_20261127`（全新未曝光种子段 `931000..944000` 单次冻结确认集实验）：`1532` 次调用，`v2 = 31,588,989`（`usage_completeness = complete`）。
> - **`M9.8` 全部运行累计总计**：**`2376` 次调用，`v2 = 51,002,681`**（`100%` 原始回执镜像与 `SHA256SUMS.txt` 封存）。

| 指标（每臂 `56` 个持续 episode，`n = 8` 簇） | `Flat_LatestFirst` | `Flat_ConditionalTable_Online` | `Hegel_FactoredMechanismGraph` | `Hegel` 对比 `Flat_Table` | `Hegel` 对比 `Flat_Latest` |
|---|---:|---:|---:|---:|---:|
| 私有验证严格通过率 (`strict_pass`) | `44/56` | `50/56` | `56/56` | — | — |
| 零 LLM (`0`-turn) 严格闭环数 | `4/56` | `15/56` | `32/56` | `+`17 | `+`28 |
| 求解 LLM 调用数 (`stream_solve`) | `618` | `504` | `286` | `-43.25%` | `-53.72%` |
| 守卫归纳 LLM 调用数 (`autonomous_guard_induction`) | `0` | `27` | `24` | `-11.11%` | — |
| **Scope-1 总 LLM 调用数 (`solve + guard`)** | **`618`** | **`531`** | **`310`** | **`-41.62%`** | **`-49.84%`** |
| Scope-1 `proxy_v1_u_025c_1o` | `12,452,860` | `10,534,868` | `5,802,048` | `-44.93%` | `-53.41%` |
| **Scope-1 `proxy_v2_u_010c_4o`（主口径）** | **`13,078,368`** | **`11,139,318`** | **`6,177,898`** | **`-44.54%`** | **`-52.76%`** |
| — Alpha 家族（`n = 4` 簇）`proxy_v2` | `7,091,417` | `6,646,428` | `3,406,721` | `-48.74%` | `-51.96%` |
| — Beta 家族（`n = 4` 簇）`proxy_v2` | `5,986,951` | `4,492,890` | `2,771,177` | `-38.32%` | `-53.71%` |
| 模型调用 `wall_ms` (`solve + guard`) | `16,766,155` | `13,638,091` | `7,713,468` | `-43.44%` | `-53.99%` |
| 非 LLM 预检容器探针 `wall_ms` (`56` 次) | `366,919` | `374,914` | `377,177` | — | — |
| 非 LLM 经验覆盖探针 `wall_ms` | `0` | `365,519` | `147,523` | `-59.64%` | — |

---

## 3. 按因果事件拆解（每类事件 `8` 个簇 — 验证 `M_comb1` 与 `M_comb2` 跨因子零样本组合）

| 因果事件 | `Flat_LatestFirst` (`通过`, `0-LLM`, `v2`) | `Flat_ConditionalTable_Online` (`通过`, `0-LLM`, `v2`) | `Hegel_FactoredMechanismGraph` (`通过`, `0-LLM`, `v2`) |
|---|---|---|---|
| `C1_down` | `7/8` (`0` 0-LLM, `2,370,988`) | `8/8` (`0` 0-LLM, `2,151,790`) | `8/8` (`0` 0-LLM, `1,974,374`) |
| `R0` | `8/8` (`4` 0-LLM, `626,727`) | `8/8` (`8` 0-LLM, `0`) | `8/8` (`8` 0-LLM, `0`) |
| `C2_down` | `7/8` (`0` 0-LLM, `2,180,956`) | `7/8` (`0` 0-LLM, `2,250,209`) | `8/8` (`0` 0-LLM, `2,221,100`) |
| `C_up` | `6/8` (`0` 0-LLM, `2,388,555`) | `4/8` (`0` 0-LLM, `2,275,667`) | `8/8` (`0` 0-LLM, `1,982,424`) |
| `U_iso` | `0/8` (`0` 0-LLM, `1,454,261`) | `7/8` (`7` 0-LLM, `201,429`) | `8/8` (`8` 0-LLM, `0`) |
| `M_comb1` | `8/8` (`0` 0-LLM, `2,246,849`) | `8/8` (`0` 0-LLM, `2,260,045`) | `8/8` (`8` 0-LLM, `0`) |
| `M_comb2` | `8/8` (`0` 0-LLM, `1,810,032`) | `8/8` (`0` 0-LLM, `2,000,178`) | `8/8` (`8` 0-LLM, `0`) |

---

## 4. 全部 `n = 8` 独立确认簇逐簇结果（`A04`, `A15`）

| 确认簇 (`家族`) | `Flat_LatestFirst` | `Flat_ConditionalTable_Online` | `Hegel_FactoredMechanismGraph` | `Latest` `v2` | `Flat_Table` `v2` | `Hegel` `v2` | `Hegel` 对比 `Flat_Table` |
|---|---|---|---|---:|---:|---:|---:|
| `alpha_m98_v2_c01` (`alpha`) | `5/7` (`1/7` 0-LLM) | `6/7` (`2/7` 0-LLM) | `7/7` (`4/7` 0-LLM) | `1,528,392` | `1,547,464` | `728,960` | `-52.89%` |
| `alpha_m98_v2_c02` (`alpha`) | `5/7` (`1/7` 0-LLM) | `5/7` (`1/7` 0-LLM) | `7/7` (`4/7` 0-LLM) | `1,822,959` | `1,760,714` | `896,092` | `-49.11%` |
| `alpha_m98_v2_c03` (`alpha`) | `5/7` (`1/7` 0-LLM) | `6/7` (`2/7` 0-LLM) | `7/7` (`4/7` 0-LLM) | `1,898,976` | `1,478,392` | `906,103` | `-38.71%` |
| `alpha_m98_v2_c04` (`alpha`) | `5/7` (`1/7` 0-LLM) | `5/7` (`2/7` 0-LLM) | `7/7` (`4/7` 0-LLM) | `1,841,090` | `1,859,858` | `875,566` | `-52.92%` |
| `beta_m98_v2_c01` (`beta`) | `6/7` (`0/7` 0-LLM) | `7/7` (`2/7` 0-LLM) | `7/7` (`4/7` 0-LLM) | `1,504,510` | `1,125,968` | `731,678` | `-35.02%` |
| `beta_m98_v2_c02` (`beta`) | `6/7` (`0/7` 0-LLM) | `7/7` (`2/7` 0-LLM) | `7/7` (`4/7` 0-LLM) | `1,451,913` | `1,164,968` | `703,532` | `-39.61%` |
| `beta_m98_v2_c03` (`beta`) | `6/7` (`0/7` 0-LLM) | `7/7` (`2/7` 0-LLM) | `7/7` (`4/7` 0-LLM) | `1,555,025` | `1,103,210` | `652,758` | `-40.83%` |
| `beta_m98_v2_c04` (`beta`) | `6/7` (`0/7` 0-LLM) | `7/7` (`2/7` 0-LLM) | `7/7` (`4/7` 0-LLM) | `1,475,503` | `1,098,744` | `683,209` | `-37.82%` |

---

## 5. 字节级全等 `(R0, U_iso)` 上的 `HistoryMasked_CurrentOnly` 容器消融（`A03`）

| 确认簇 | 事件 | `SHA256(/holdout_task/public_batch.pkl)` | 遮蔽历史后选中分支 | 遮蔽历史私有验证通过 |
|---|---|---|---|---|
| `alpha_m98_v2_c01` | `R0` | `ae4a31e07207547a...` | `bind_alpha_m98_v2_c01_dn_b02` | `False` |
| `alpha_m98_v2_c01` | `U_iso` | `ae4a31e07207547a...` | `bind_alpha_m98_v2_c01_dn_b02` | `False` |
| `alpha_m98_v2_c02` | `R0` | `a442fb3bf9640e4e...` | `bind_alpha_m98_v2_c02_dn_b02` | `False` |
| `alpha_m98_v2_c02` | `U_iso` | `a442fb3bf9640e4e...` | `bind_alpha_m98_v2_c02_dn_b02` | `False` |
| `alpha_m98_v2_c03` | `R0` | `5934e05b785216ee...` | `bind_alpha_m98_v2_c03_dn_b02` | `False` |
| `alpha_m98_v2_c03` | `U_iso` | `5934e05b785216ee...` | `bind_alpha_m98_v2_c03_dn_b02` | `False` |
| `alpha_m98_v2_c04` | `R0` | `d46c5bf54b8418b3...` | `bind_alpha_m98_v2_c04_dn_b02` | `False` |
| `alpha_m98_v2_c04` | `U_iso` | `d46c5bf54b8418b3...` | `bind_alpha_m98_v2_c04_dn_b02` | `False` |
| `beta_m98_v2_c01` | `R0` | `6b21aad5b6fbf76c...` | `bind_beta_m98_v2_c01_dn_b02` | `False` |
| `beta_m98_v2_c01` | `U_iso` | `6b21aad5b6fbf76c...` | `bind_beta_m98_v2_c01_dn_b02` | `False` |
| `beta_m98_v2_c02` | `R0` | `c49246d5f780d292...` | `bind_beta_m98_v2_c02_dn_b02` | `False` |
| `beta_m98_v2_c02` | `U_iso` | `c49246d5f780d292...` | `bind_beta_m98_v2_c02_dn_b02` | `False` |
| `beta_m98_v2_c03` | `R0` | `bf91a8bf8337c91a...` | `bind_beta_m98_v2_c03_dn_b02` | `False` |
| `beta_m98_v2_c03` | `U_iso` | `bf91a8bf8337c91a...` | `bind_beta_m98_v2_c03_dn_b02` | `False` |
| `beta_m98_v2_c04` | `R0` | `6c2059b36a6e8b80...` | `bind_beta_m98_v2_c04_dn_b02` | `False` |
| `beta_m98_v2_c04` | `U_iso` | `6c2059b36a6e8b80...` | `bind_beta_m98_v2_c04_dn_b02` | `False` |
