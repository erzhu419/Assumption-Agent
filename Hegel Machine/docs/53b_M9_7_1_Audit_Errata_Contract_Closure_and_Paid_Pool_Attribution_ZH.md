# M9.7.1 审计勘误、契约加固、冷启动恢复与已付代码池真实执行归因报告（`0` 次付费 LLM 调用）

- **封存证据目录**：`docs/evidence/m9_7_1_audit_closure_and_paid_pool_attribution_20261127/`（`SHA256SUMS.txt` 全量校验通过；全部历史 `M9.4`–`M9.7` 证据目录 `100%` 原样只读保留）
- **证据来源纪律（`A01` 闭合）**：所有统计与回放表强制标注 `evidence_provenance`（`executed` 真实隔离执行、`recomputed_from_raw` 原始账本重算、`derived_expectation` 理论推导），绝不将未执行常量包装为实跑回执。

## 一、M9.7 统计口径、调用计数、未知用量与非 LLM 容器耗时勘误（`A04, A05, A12, A15, A17`）

1. **实验性质重分类与全物理总账（`A04`）**：
   - 由于 `CONFIRMATION_M97_WORLD_SPECS` 在提交 `27b8cb97c` 前已于中止排障运行 `run1`（OpenBLAS 容器线程争用）与 `run2`（Beta 域异常证人分类）中曝光，`M9.7` `run3` 正式归类为**修复后冻结复制机制实验（已曝光种子）**，未曝光盲测确认留交全新种子的 `M9.8`。
   - `run1 + run2 + run3` 全物理账本合计：**`1,564` 次唯一 `call_uuid`**，已知 `proxy_v2 = 36,439,824`（Alpha 在三次运行中节省幅度高度一致：`-64.24%`、`-65.70%`、`-66.02%`）。
2. **调用分项与部分未知用量声明（`A04, A05`）**：
   - `Hegel_DemandGated_Graph` 持续流调用数：**`148` 次求解调用 + `6` 次守卫归纳调用 = 共 `154` 次持续流 LLM 调用**，对比 `Flat` **`307` 次**（求解调用 `-51.79%`，**总持续流调用 `-49.84%`**）。
   - 用量完整度：**`partial_unknown_usage`**（`alpha_m97_c01:C1` 含 `1` 次 `360,015.06 ms` 超时尝试 `1b028fe68e4745c692589f62c402f78b`），真实差值精确写作 $C_{\text{Hegel}} - C_{\text{Flat}} = -3{,}741{,}297 + x$（$x \ge 0$）。
3. **分族分层结果与 Beta 准确表述（`A05, A15`）**：
   - **Alpha（`n = 3`）**：`20/21` vs `20/21` 私判通过，`14/21` vs `3/21` 零调用通过闭环，已知 `Scope 1 proxy_v2` 为 `1,957,198` vs `5,759,604`（**`-66.02%`**，`3/3` 同向下降，双侧符号翻转 `p = 0.25`；Alpha 占 `Flat` 总 `v2` 的 `80.67%`）。
   - **Beta（`n = 3`）**：`21/21` vs `21/21` 私判通过、`15/21` vs `15/21` 零调用通过闭环持平，`0` 次守卫归纳调用、`0` 次流内 L2 编译调用，已知 `Scope 1 proxy_v2` 为 **`1,441,450` vs `1,380,341`（`+4.43%`，`+61,109`）**（精确双侧符号翻转 `p = 1.0`；更正原报告“exact tie / 零税”及旧硬编码 `0.50` 的不精确表述）。
   - **合并簇级（`n = 6`）**：簇均值差 `-623,549.50`，逐簇相对变化均值 **`-28.90%`**，双侧符号翻转 `p = 0.15625`。
4. **完整非 LLM 容器墙钟时间账本（`A05`，严格按 `wall_ms` 统计，不与 CPU 秒混用）**：
   - `Flat_EqualInfo_Multiversion`：`42` 次预检容器探针 = **`311,517.44 ms`**（`0` 次覆盖探针）。
   - `Hegel_DemandGated_Graph`：`42` 次预检容器探针（`319,824.16 ms`）+ `18` 次 `EmpiricalCoverageReceipt` 容器探针（`115,258.55 ms`）= **`435,082.71 ms`**（多出 `+123,565.27 ms` 容器墙钟耗时，另有 `6` 次守卫归纳中的 `24` 次父扰动容器敏感度运行）。
5. **`R0` vs `U_iso` 口径收窄（`A03, A13`）、编译策略命名（`A12`）与旧 KKT `L1` 原语缺口保留声明（`A14`）**：
   - `M9.7` 的 `R0` 与 `U_iso` 严格匹配测试协变量（`obs_02..08 == 0.0` 差值）、公开阈值、标定扫描与末端快照，但训练观测曲线（最大差 `17.78`）、Beta `y_obs_raw`（`0.0325`）与任务标题不同；`M9.7` 中同契约流上的编译门控实际等价于 `disable_eager_l2_when_canonical_binding_covers_contract`。
   - **`A14`（`retained_open_gap`）**：整程序任务求解器虽已通过完整 KKT 主对偶校验，但历史 `M9.4/M9.6` `L1` 原语编译器的完整对偶证书返回与模型实际采用仍保留为 `retained_open_gap`（本报告不声称其已调用或闭合）。
6. **`alpha_m97_c02 / R1` 精确数值取证与候选退役声明（`A06`，状态 = `diagnosis_and_error_code_closed_numerical_candidate_retired`）**：
   - 对 SHA 锁定的 `2b098388239b...` 复现证实两个私判种子的全部 5 个输出数组均 `100%` 有限（`all_outputs_finite = True`）；种子 `991672` 实际因极点翻转触发 **`PARAMETER_ORDER`**（`estimated_taus = [1.630723, 0.280998]` vs 真值 `[0.66, 4.35]`）、**`STRAIN_ERROR`**（`0.081975 > 0.070`）与 **`RATE_ERROR`**（`0.162279 > 0.055`）。验证器错误码已拆分闭合，历史封存成绩 `41/42` 原样保留，该失稳候选正式退役（不将该诊断闭合包装为已修复数值优化器）。

## 二、已付代码池真实隔离执行归因（`M9.5` `A01` + `M9.7` `A02`，`evidence_provenance = executed`）

| 代码池回放实验 | 受控策略臂 | 需要机制图？ | 公共门控通过 | 零 LLM 私判 `strict_pass` | 探针数 | 证据来源 |
|---|---|---:|---:|---:|---:|---|
| `M9.5`（`48` 个复用步，`A01`） | `1_SingleLatest_Unvalidated_Probes1` | 否 | `28/48` | `28/48` | `48` | `executed` |
| `M9.5`（`48` 个复用步，`A01`） | `2_SingleLatest_ValidatedWingOLS_Probes1` | 否 | `40/48` | `40/48` | `48` | `executed` |
| `M9.5`（`48` 个复用步，`A01`） | `3_Multiversion_OldestFirst_Probes1` | 否 | **`32/48`** | **`32/48`** | `48` | `executed` |
| `M9.5`（`48` 个复用步，`A01`） | `4_Multiversion_ValidatedWingOLS_Probes2` | 否 | `48/48` | `48/48` | `56` | `executed` |
| `M9.5`（`48` 个复用步，`A01`） | `5_Flat_SharedConditionalTable_Probes1` | 否 | **`48/48`** | **`48/48`** | `48` | `executed` |
| `M9.5`（`48` 个复用步，`A01`） | `6_Hegel_MechanismGraph_Guard_Probes1` | 是 | `48/48` | `48/48` | `48` | `executed` |
| `M9.7`（`42` 个持续步，`A02`） | `Flat_LatestFirst_Probes1`（`run3`） | 否 | `18/42` | `18/42` | `42` | `recomputed_from_raw` |
| `M9.7`（`42` 个持续步，`A02`） | **`Flat_ConditionalTable_Executed_Probes1`** | **否** | **`30/42`** | **`29/42`** | `42` | **`executed`（Docker）** |
| `M9.7`（`42` 个持续步，`A02`） | `Hegel_DemandGated_Graph_Probes1`（`run3`） | 是 | `30/42` | `29/42` | `42` | `recomputed_from_raw` |

- **核心表征归因结论（`A02`）**：在 `M9.7` 全部 `42` 个持续步上，严格前缀锁定的普通条件版本表 `Flat_ConditionalTable`（`0` 个 `PersistentMechanismGraph` / `M94BindingRegistry` / `Forest` 对象）在隔离 Docker 实跑中与 `Hegel_DemandGated_Graph` 达成 **`42/42` 路由一致**、**`30/42` 公共门控一致**与 **`29/42` 零 LLM 私判通过一致**（同时单列 `run3` 中已付出的结构形成成本：`6` 次守卫调用 = `98,267 v2` + `115,258.55 ms` 覆盖探针）。原因非常清晰：`M9.7` 所有分支共享完全相同的 `{cal_node_p0, cal_node_p1}` 父集合；要真正检验图拓扑超越 `Flat_ConditionalTable` 的独特价值，必须测试**分支特异的异构多父依赖与局部选择性失效/重校准**（见 `M9.8`）。

## 三、契约加固、冷启动恢复与全仓能力索引（`A07, A08, A09, A10, A11, A16`）

- **`A07`**：`verify_binding_identity_and_foreign_keys` 强制 `set(parent_parameter_versions.keys()) == set(expected_parents)`（`PARENT_VERSION_SET_MISMATCH`），旧残缺记录安全迁移为 `unknown_requires_reverification`。
- **`A08`**：`verify_m97_multibranch_witness_partition` 强制单向 `(eval_bid, owner_bid)` 覆盖语义，并为 `EmpiricalCoverageReceipt` 增加内容寻址完整性重算校验。
- **`A09`**：`classify_public_probe_failure` 严格执行 `INFRA_FAILURE -> SCHEMA_FAILURE -> DOMAIN_REJECTION -> SCIENTIFIC_FAILURE` 四级优先级，确保超时绝不成为科学反例。
- **`A10`**：`ImmutableBranchLineageController` 将 `witness_ledger` 与 `coverage_receipts` 内容寻址持久化进 `PersistentMechanismGraph` 检查点；`C2` 后冷停止并重载检查点可精确复现 `R1 / U_iso / G_perm / M_couple` 完整路由且 `0` 额外 LLM 调用。
- **`A11`**：`estimate_validated_channel_affine_calibration` 统一 `Flat` 与 `Graph` 的分通道分层 `2/3–1/3` 估计器（全开发集差值精确为 `0.0`）。
- **`A16`**：`historical_capabilities.json`（完整保留 `181` 条历史版本并新增 `m6_m97_continual_mechanism_lineage_and_verification` 路由）与 `docs/hegel_machine_reuse_hub_20261006.md` 已同步索引 `M6–M9.7.1`。
