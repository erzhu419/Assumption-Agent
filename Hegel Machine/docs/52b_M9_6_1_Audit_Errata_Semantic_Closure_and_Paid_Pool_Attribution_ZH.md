# M9.6.1 审计勘误、语义闭合与已付代码池受控归因报告（`0` 次付费 LLM 调用）

- **证据包路径**：`docs/evidence/m9_6_1_semantic_closure_and_paid_pool_attribution_20261127/`（`SHA256SUMS.txt` 全量通过；原 `M9.4`、`M9.5`、`M9.5.1`、`M9.6` 封存证据保持原状）
- **统一成本策略版本**：`hegel_cost_policy_v2_m961_call_uuid_conserved`（同时对 `v1` 与 `v2` 强制断言 `sum(episodes) == sum(events) == sum(worlds) == sum(disjoint_pairs) == arm_ledger`）
- **统一三层科学核查内核**：`hegel_shared_math_handoff_kernel_v3_m961_three_layer`（结构核查 `validate_output_structure` + 估计坐标见证核查 `validate_public_witness` + 隐藏真值精度核查 `validate_truth_accuracy`）

## 一、M9.6 逐事件、逐世界与独立配对簇成本勘误（`N01` 闭合）

- **按事件修正后的真实 `proxy_v2`（完全从 `call_uuid` 原始三元组重算）**：
  - `T`：`Flat = 2,531,206` vs `Full = 3,095,591`（其余零轮事件 `N, G1, R, U, G2, M = 0`）
  - `C`：`Flat = 2,472,490` vs `Full = 3,531,535`
  - **守恒核对**：`Flat = 5,003,696`（严格等于 `Scope 1 = 5,003,696`），`Full = 6,627,126`（严格等于 `Scope 1 = 6,627,126`）。流内 L2 编译占额外 `v2` 总差值的精确比例为 **`89.55%`**（`1,453,819 / 1,623,430`）。
- **主推断层级（`n = 4` 个独立来源配对簇）**：
  - 四个独立 pair 的 `v2` 差值（`Full - Flat`）：`[122073, 401432, 474908, 625017]`（`4/4` 同向更高，精确双侧符号翻转 `p = 0.125`）。
- **次级描述性敏感度（`n = 8` 个世界）**：
  - 逐世界 `v2` 差值：`{'alpha_m96_w01': 129381, 'alpha_m96_w02': 291293, 'alpha_m96_w03': 243453, 'alpha_m96_w04': 393279, 'beta_m96_w01': -7308, 'beta_m96_w02': 110139, 'beta_m96_w03': 231455, 'beta_m96_w04': 231738}`（**`7/8` 个世界更高**，`beta_m96_w01 = -7,308`；均值差 `202,928.75`，95% Bootstrap CI `[119139.49, 280925.88]`，双侧符号翻转 `p = 0.015625`）。
- **零模型调用轮次口径区分**：两臂各有 `48/64` 个零模型调用 episode，其中成功通过私有验证器的零轮闭环各为 `47/64`（不能将零 LLM 调用等同于零容器/CPU 成本）。

## 二、`alpha_m96_w03 / M` 真实根因订正（`N02`）与 M9.4 `68/68` 全量重评勘误（`N03`）

- **`N02`（`alpha_m96_w03 / M` 误诊订正）**：原 `M9.6` 正式封存成绩保持 `63/64` vs `63/64` 不变。该 `M` 任务并未传入非零初态（`preload_scale = 0.0`）；种子 `973861` 的 `0.1128 > 0.08` 残差源于旧私判用隐藏真值应力检验候选的代数自洽关系。在三层契约内核 `hegel_shared_math_handoff_kernel_v3_m961_three_layer` 下，第 2 层估计坐标恒等式残差低至 `<= 2.01e-14`（同时 `100%` 拦截伪造标定见证），第 3 层隐藏真值物理误差全部达标。
- **`N03`（`M9.4` 全量 `68` 私有实例统一内核重评）**：修复 `terminal_field_B` 双通道分派并由 `hegel_shared_math_handoff_kernel_v3_m961_three_layer` 完整独立重跑 `M9.4` 全部 `34` 个 episode（`68` 个私有实例）后，通过数为 **`38/68` 私有实例（`17/34` 轮次）**。

## 三、资格身份（`N04`）、多冲突不可变谱系（`N07`）、重建漏斗（`N09/N10`）与代码池控制（`N11`）

- **`N04`**：按 `executable_kind` 唯一绑定实际执行代码 SHA（拒绝未执行模块凭据冒用：`True`），重算内容寻址 `receipt_id`（拒绝篡改内容：`True`），并校验完整多父版本向量 `parent_parameter_versions`（拒绝次父节点过期版本：`True`）。
- **`N07`**：`ImmutableBranchLineageController` 拒绝无实测科学反例的分叉（`True`），禁止由选中 `C1` 自造 `C0` 负见证（`True`），并在确定性状态测试中完整通过 `C0 -> C1 -> C2 -> 恢复 C0 -> 恢复 C1`（`{'restore_C0_x_0_2': 'bind_dev_b00', 'restore_C1_x_1_1': 'bind_dev_b01', 'route_C2_x_2_4': 'bind_dev_b02'}`）。
- **`N09 + N10`**：从 `04d_skill_contract_inspections.jsonl` 重建 `Full` 在 `T` 上的真实漏斗：新 `L1` 搜索/描述为 `5/8`、`3/8`；新 `L2` 搜索/描述为 `6/8`、`4/8`；至少描述任一新模块为 `7/8`；动态调用与精确保留仍为 `0/8`、`0/8`（同时明确区分 `4` 个 Alpha 跨域 KKT 机会与 `4` 个 Beta 同族多预算机会，以及部分 L1 缺失完整对偶见证字典的契约缺口）。
- **`N11`**：顺序受控代码池回放区分了 `OldestFirst_Probes1`（`32/48`）与 `LatestFirst_Probes1`（`40/48`），并证明在固定两翼接线的 `M9.5` 上 `Flat_SharedConditionalTable_Probes1`（`48/48`）与 `Hegel` 图守卫（`48/48`）等价——直接引出下一阶段 `M9.7` 对**非嵌套多次冲突 + 严格同现异史 + 置换多父与真实接线干扰父 + 按需编译**的检验。
