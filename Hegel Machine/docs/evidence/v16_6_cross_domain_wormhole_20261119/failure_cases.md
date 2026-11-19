# Failure Cases, Negative-Transfer Traps & Honest Non-Gain Cases (`failure_cases.md`)

依据《06_给Gemini的实施合同.md》§9 与 §11，如实报告本轮观察到的无收益簇、被撤回的弱父节点及近失例拦截（全部数值由实测结果动态导出）：

## 1. 簇 A（有序离散边界）：Hegel 在目标成本上未击败 `O_strong`（诚实记录，不禁用基线二分！）
- **实测现象**：在 4 个独立冻结测试簇（共 20 道簇 A 目标）上，`Hegel_V16_6` 与 `O_strong` 严格通过率均为 `20/20`（零非法放宽 `earliest_global_threshold`），观测探针数完全一致，但簇 A 目标复合成本差 `Hegel - O_strong = +0.7500`（Hegel 多付每次 `3.0` compute 的森林检索与合同校验费，折合每簇 `+0.75` 复合成本）。
- **原因与结论边界**：因为 `O_strong` 被公平允许直接使用折半查找与前缀校验原语，簇 A 证明的是**自主形成、近失例分裂（`some_adjacent_flip` vs `earliest_global_threshold`）与零降标谎报生命周期成立**，而非在已知基础算法上宣称效率突破。

## 2. 诚实的“父命题仍成立但无净收益 → 主动降级退出”案例（`sys_root_concave_limit` on `T_B5_unbounded_sqrt_concave`）
- **实测现象**：在无界凹增长近失例 `nm_B2_unbounded_sqrt` 与未见目标 `T_B5_unbounded_sqrt_concave`（$y = a\sqrt{u}+b$）上，中间层有理饱和模板因第 3 点尾部残差超标被 `ALIGN` 排除谓词拦截；此时局部树根节点 `sys_root_concave_limit`（单调边际递减）在目标上**依然为真**。
- **治理动作**：仅凭“单调凹”无法比共享开放语法省掉采样点（`Hegel` 与 `O_strong` 测量成本均为 `42.0`，复合成本 `44.77` vs `44.23`）。系统将其标记为 `archived_low_utility = True` 并执行 **Clean Exit**（零浪费复用已购 3 个预解点转入开放语法直接拟合），相比生搬硬套有理饱和后再补救的 `O_HExA_inspired`（测量成本 `48.0`）每题节省 `6.0` 测量成本。

## 3. 跨尺度误用拦截案例（`T_C2_service_burst_tail_latency`）
- **实测现象**：同一结算微服务在 `T_C1`（稳态平均吞吐）下合法通过横向虫洞复用有理饱和模型；但在 `T_C2`（突发负载 p99 尾延迟）下，`ScaleMap(window_average)` 丢弃了突发到达聚类信息（违反闭合条件）。
- **对比差异**：`O_HExA_inspired` 因表面同属该服务响应而先试跑有理饱和再触发错误恢复（平均耗费 `8.0` 次探针 = `48.0` 测量成本），而 `Hegel_V16_6` 在 0 附加探针时即由 `authorize_abstraction_use` 拒绝 `window_average` 尺度映射（`scale_map_does_not_preserve_task_view`），直接以 `7.0` 次探针（`42.0` 测量成本）完成 $1/u^2$ 尾部模型辨识。

## 4. 源形成与近失例修订开销及端到端对账
- **实测账本**：每簇源与近失例阶段均值复合成本：`Hegel_V16_6 = 296.38`，`O_strong = 305.73`，`O_HExA_inspired = 308.82`（在纯单源形成步骤上 Hegel 多付对比归纳与版本注册 compute，而在近失例 `nm_B1` 上因 CEGAR 联合探针比普通网格回退少买探针而抵消了部分开销）。计入全部源阶段与近失例开销后，端到端全生命周期总成本差 `Hegel - O_strong = -64.7000`，目标阶段净差 `Hegel - O_strong = -55.3500`。
