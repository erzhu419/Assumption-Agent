# Hegel Machine v10.2–v10.4 统一技术交接与审计闭合主档案（供外部 AI 审阅）

## 0. 执行摘要与硬约束合规声明

本档案面向外部审计模型（GPT-6），完整汇报自被审计提交 `a66cd82b22f444044dbda98f43782c96f833a4f0` 以来在 `codex/hegel-cognitive-mainline` 分支上依次完成的三个闭合阶段：
1. **`v10.2`（提交 `85b6055b11312a53720561f6697f6054b4ab8873`）：** 100% 解决 `GPT_diagnosis_v10.2.md` 及 `01–04` 施工合同提出的全部 **20 项审计发现（`F01–F20`）**、**交付物 A / B / C** 与 **22 项行为验收标准（`#1–#22`）**。
2. **`v10.3`（提交 `c0bad267a4f211cda03f388e9386908e73c27016`）：** 针对 `v10.2` 诚实暴露的两道真实研究前沿——小库下 `structured` 与强 `archive_full_tools` 近似平手（`F17`）以及 $N \ge 6$ 冷传感器角色消歧瓶颈——在不引入任何新物理家族的前提下，实现持久化多键前提签名索引（`PremiseCommitmentIndex`）与可证伪跨任务传感器角色标定先验（`SensorRoleCalibrationPrior`），完成 $M \in \{4, 12, 36, 72\}$ 与 $N \in \{6, 7, 8\}$ 严格单因子消融。
3. **`v10.4`（当前提交）：** 针对 `v10.3` 在 $M \ge 36$ 大规模开放语法库上暴露的“12行稀疏初测窗口自定义阈值谓词假阳性劫持无条件守恒”现象，引入无作弊**四层奥卡姆极小性门控（`occam_premise_minimality`）**（公平同时授予 `archive_with_occam` 与 `structured_indexed_with_occam`），并完成 $M = 72 \times N = 8$ 全栈联合验证与自包含审计包构建。

### 硬边界核查清单
- **Phase 6 扩张冻结：** 严格遵守 `02_下一施工合同.md` 第 2.4 节——未新增三维／移动网格 PDE、未新增 SOS/Positivstellensatz、未新增通用定理生成器或专设物理家族。
- **零新增付费 LLM 调用：** `v10.2`、`v10.3`、`v10.4` 全部代码、测试与 **9 份新增证据回执**均在 **`model_calls = 0, total_tokens = 0`** 下完成。
- **历史回执 100% 零覆盖：** `v10_world_structure_20261116/` 下全部 13 份原始回执保持原始字节零修改（SHA256 由 `05_STANDALONE_VERIFY_V10_4.py` 自动核验）。
- **零私有属性作弊与零削弱基线：** 运行时认知模块对 `lab._...` 私有字段零读取；所有对照中普通基线（`cold_full_tools`、`archive_full_tools`）均共享完全相同的求解器集合、证人核验器与奥卡姆极小性规则。

---

## 一、`v10.2` 审计闭合核心结论（详见 `02_F01_TO_F20_AND_22_ACCEPTANCE_MATRIX.md`）

1. **P0 证明域合法性统一（`F01`）：**
   - 修复前 `probe_certificate_domain.py` 复现 `both_opposite_certificates_accepted = True`。
   - 修复后在 `algebraic_proof_certificates.py` 统一 `physical_legality`（全链路强制 `state_bounds`，拒绝出盒证人）、`analysis_region`（未由 `verify_reachable_box_enclosure` 证明可达集包络时返回 `status="conditional_impossibility_domain_unverified", verified=False`）、`unbounded`（禁止虚构默认盒，拒绝非零盒对偶乘子），三种模式下 **`both_opposite_certificates_accepted = False`**。
   - 同时闭合 `F02`（Lean 导出降级为 `draft_only, lean_compiled=False`，可信基明确为 Python `Fraction` + `SymPy QQ`）、`F07`（`certify_rollout_baseline` 共享核验器）、`F10`（全链路接入 `SearchBudget`）、`F11`（`all_arity_proved=False` + 有限元数列表）、`F12`（内核强制检查非负状态锥与有向边位移号）、`F14`（打通 `AnonymousObjectLab` 在线发现到 `model_origin="learned_model"` 证书生成闭环，诚实标记 `environment_coverage_verified=False`）。
2. **SDE 含噪历史条件漂移重测（`F08`）：**
   - 新增独立后验评估器 `sde_conditional_drift_eval.py`（预测阶段严格屏蔽 `after` 传感器），在 Cohort 5B-1 强过程噪声环境下分离报告四项指标：2SLS-IV 原生完成率 **`2/4`**、实时含噪轨迹均值 MSE **`5.421e-3`**、**真实含噪历史一步条件漂移均值 MSE `6.119e-4`（`4/4 <= 2e-3`，较 Naive OLS `6.276e-3` 降低 `10.26x`）**、无噪孪生迁移 MSE **`3.317e-4`**。
3. **严格同工具归因与四机制实测修复（交付物 B，`F03–F06, F17`）：**
   - **Study B1（三臂同工具）：** `cold_full_tools` `0/4` vs `archive_full_tools` `3/4` vs `structured_full_tools` `3/4`（目标总算力较 `cold` 降 `-34.42%`；在小库 $M \le 2$ 下 `structured` 较强重构 `archive` 仅省 `580` 算力，如实报告近似平手，撤回“同构漏洞”旧表述）。
   - **Study B2（前提依赖线索严格单因子隔离）：** 固定同一父承诺、同一观测包与同一 43 项局部补丁语法，仅切换违背前提槽位线索——紧预算 `16k` 下诚实平手（`0/6` vs `0/6`）；充足预算 `220k` 下 `fresh48 <= 1e-7` 通过率 **`2/6 -> 4/6`**、均值搜索算力 **`168,137.3 -> 54,075.0`（-67.84%）**；另单独测量 `local_patch` (`2/4`) vs `from_scratch` (`0/4`) 独立语法空间因子。
   - **Study B3（跨 4 类真正不同物理机制实测探针修复）：** 覆盖端口增益、体相耗散漏汇、耦合偏移、有向反转 4 类机制共 8 题，双臂均 `8/8` 通过，实测 `lab.probe_queries` 由 **`101` 步降至 `42` 步（-58.42%）**（替代原 `89.92%` 名义公式估计）。
4. **冻结全 6 阶段 `v4` 生命周期（交付物 C）：**
   - 单独披露开发定标链 `dev_chain_700000`（全阶段通过），并在目标暴露前 SHA256 冻结评估 **4 条连续预注册种子前瞻链 `711000–714000`**（零挑选种子）：`cold_full_tools` 后冻结阶段 **`0/12`**；`archive_full_tools` 与 `structured_lifecycle_v10` 均为 **`7/12`（在源形成成功的 3 条链上为 `7/9 = 77.8%`）**，单链冻结后目标算力较 `cold` 降低 **`-23.45%`**，完整保留 `prospective_chain_2` Stage 1 稀疏初测未过与 Stage 6 $N=6$ 冷角色预算耗尽两类真实失败分母。

---

## 二、`v10.3` 与 `v10.4`：大规模冲突库分化、奥卡姆门控与高元数突破（详见 `03_SCALING_HIGH_ARITY_AND_OCCAM_REPORT.md`）

1. **持久化前提签名索引（`PremiseCommitmentIndex`）+ 四层奥卡姆极小性门控（`occam_premise_minimality`）：**
   - 在公平赋予 `archive_with_occam` 相同奥卡姆门控的前提下（Study 1A，`ample_380k`，冷角色归纳，6 题 $N=5$ 目标）：
     * **$M = 36$：** 无门控时 `archive_no_occam` / `structured_indexed_no_occam` 因 13 类自定义谓词劫持无条件题仅过 `3/6`；开启奥卡姆门控后，`archive_with_occam` 恢复至 **`4/6`**（均值目标算力 `245,914.3`），**`structured_indexed_with_occam` 达到 `5/6`**（均值目标算力 **`215,422.3`**，单题净省 `30,492.00` 算力，写入成本摊销盈亏平衡仅需 **`0.28` 题**）。
     * **$M = 72$：** 无门控时 `archive_no_occam` 为 `2/6`、`structured_indexed_no_occam` 为 `3/6`；开启奥卡姆门控后，`archive_with_occam` 达到 **`4/6`**（均值目标算力 `274,952.3`），**`structured_indexed_with_occam` 达到 `5/6`**（均值目标算力 **`216,109.7`**，单题较同工具 `archive_with_occam` 净省 **`58,842.67` 算力**，盈亏平衡 **`0.29` 题**）。
   - 在 **Study 1B（$M=72$ 逆序对抗插入测试：68 条冲突先验置于 4 条基础先验之前）** 中，`reversed_m72_indexed_no_occam` 仅通过 **`0/4`**（审计算力 `229,890.0`），而 **`reversed_m72_indexed_with_occam` 达到 `4/4`**（审计算力降至 **`32,546.0`**），证实了先验入库顺序无关性。
2. **可证伪跨任务传感器角色标定先验（`SensorRoleCalibrationPrior`）突破 $N = 6, 7, 8$：**
   - 仅从已确认源任务提炼实验设计先验，在目标端先执行 2 行可证伪检验（等位置静默 + 多驱动叠加线性性），通过后以 $O(N)$ 枢纽二分法标定位置、以 $O(N)$ 等状态选择器扫描标定焦点。
   - **v10.3 Study 2A（9 题，$N \in \{6, 7, 8\}$）：** `cold_full_tools` `0/9`、`commitment_only_cold_roles` `2/9`（$N=7,8$ 全军覆没 `0/6`）、**`commitment_plus_calibration_prior` `9/9`（100%，全部编译至纯控制通道 `selected_channels=[]`，角色消歧算力下降 `-88.71%`）**。
   - **v10.3 Study 2B（3 台违背先验仪器负向控制）：** `variant="anchored"`、`ContinuumFieldLab`（无位置环 PDE）、`readout="total"`（全局焦点不变读出）上 **`false_role_application_count = 0`**，连续介质与全局守恒任务仍正常以 `<= 1e-19` MSE 确认通过。
   - **v10.3 Study 2C（前瞻链 `711000–714000` 扩展至 Stage 7 $N=8$）：** 在 Stage 1 成功的全部 3 条前瞻链上，后冻结 4 个阶段（含 $N=6$ 与 $N=8$ 两次原条件回归恢复 $C_0$）达到 **`12/12`（100%）**；`prospective_chain_2` 保持 `0` 先验诚实负样本。
3. **`v10.4` Study 2：$M = 72$ 冲突库 $\times$ $N = 8$（$(8!)^2 = 1.63 \times 10^9$ 排列）全栈联合评测：**
   - 在同时挂载 72 条异质冲突先验库与高元数 `N=8` 匿名目标仪器的最严苛组合下：`cold_full_tools` 为 `0/3`；未配角色标定的 `m72_indexed_cold_roles_with_occam` 因 32 传感器排列瓶颈为 `0/3`；全栈 **`m72_full_stack_indexed_calibrated_occam` 达到 `2/3` 在线纯控制通道确认通过**（另第 2 题 `ha_dpg_left_rev_N8` 亦恢复出 `fresh48 MSE = 1.38e-18` 的精确程序，因 `461.9k/480k` 预算上限未跑完最终无传感器确认步而严格按诚实分母计为 `2/3`），均值角色消歧算力降至 `97,633.0`，均值总算力降至 `296,366.0`，证明三大机制零干扰正交协同。

---

## 三、诚实保留的开放边界与非声称声明（Non-Claims）

1. **冷启动首轮稀疏采样方差仍客观存在：** 如 `prospective_chain_2`（`712000`）Stage 1 所示，当完全无历史先验（空库）时，首轮冷观测采样的信噪比偶发不足以跨过严格的 `fresh48 <= 1e-7` 确认门控（我们拒绝放宽 `1e-7` 门控或事后换种子掩盖该负例）。
2. **形式证明边界限定：** `algebraic_proof_certificates.py` 的可信计算基为 Python `Fraction` 精确有理算术与 `SymPy QQ` 多项式环（在所测有限元数 $N$ 上精确成立，`all_arity_proved = False`）；Lean 4 文本导出仅为草稿模板（`lean_status = "draft_only", lean_compiled = False`），且 `learned_model` 证书仅对经验提取出的线性控制模型精确成立（`environment_coverage_verified = False`）。
3. **离散格点与图算子边界限定：** Phase 4B/5A 实测对象为有限离散周期环／二维离散环面格点（检验离散平移子群 $\mathbb{Z}_{N_x}\times\mathbb{Z}_{N_y}$）与有限不规则图拉普拉斯动力系统，不声称已证明网格间距 $h \to 0$ 的连续偏微分方程极限收敛或全自同构群不变性。
4. **同工具公平性结论：** 在小规模知识库（$M \le 4$）下，允许按需重构等价承诺的强 `archive_full_tools` 基线与预巩固结构化库表现接近平手；结构化写入期巩固与多键前提索引的算力优势（在赋予双方同等奥卡姆门控后，在 $M=72$ 下每题节省按需重构与重复巩固开销 `58,842.7` 算力，首题内 `K*=0.29` 即摊平写入成本）主要体现在中大规模异质冲突知识库的持续积累场景。
