# Hegel M8.9.1 审计勘误、完备 Box-L0 证书核与封存 HS01 独立重评增补报告（`docs/44b_ZH`）

- **增补范围**：针对 `docs/44_M8_9_Progressive_Disclosure_and_Net_Value_Staging_Report_ZH.md`（及英文版）与封存证据目录 `docs/evidence/m8_9_progressive_disclosure_and_net_value_ablation_20261126/` 的非破坏性勘误及确定性正确性闭环（回应 `markdown/Hegel_M8_9_Audit` 全部可复现反例 `R1..R7`）。
- **封存不改性原则**：已封存的 `M8.9` 证据目录与原始 `docs/44_*` 报告保持 `0` 字节修改；所有独立重评、真实逐任务留一交叉验证（`LOTO`）及台账核查回执均独立存放在 `docs/evidence/m8_9_1_audit_closure_and_regrade_20261126/`。

---

## 1. `docs/44` 正式勘误对照表（台账事实 vs 报告叙述对齐）

| `docs/44` 条目 | `docs/44` 原文表述 | 封存台账真实数值（`M8.9.1` 审计核实） | 处置与代码闭环 |
| :--- | :--- | :--- | :--- |
| **主分支轮数预算（`max_turns`）** | 正文误写为 `18 轮主分支预算`。 | `M8.9` 全部 `24/24` 个 episode 的实际主分支预算均为 `max_turns = 28`（与 `run_m8_9_progressive_disclosure_and_net_value_ablation.py` 默认 `--max-turns 28` 一致），三组实验臂预算完全公平对称。 | 在本勘误及 `m8_9_1_ledger_errata_audit.json` 中正式更正。 |
| **渐进披露函数卡片长度** | 正文简述为 `约 120 字符函数索引卡`。 | `M8.9` 归档卡片实际字符长度：Arm 2（`126` 张）`min = 324`、`median = 441.0`、`mean = 462.71`、`max = 755`；Arm 3（`91` 张）`min = 324`、`median = 424.0`、`mean = 467.73`、`max = 755`。Turn-1 清单总长压缩（`41,929.5 -> 6,210.0` 字符，`-85.19%`）完全精确。 | `m6_4_executable_wormhole_library.py` 现直接在诊断字典中输出精确 `card_char_stats`。 |
| **Arm 3 `mean_staged_l1_count`（`4.00 -> 2.625`）口径** | 将 `4.00 -> 2.625` 统称为已挂载 `L1` 数下降，未区分 Turn-1 推荐集与容器磁盘/目录挂载集。 | 在 `M8.9` 中，Arm 2 在磁盘上挂载了 `4.00` 个领域过滤 `L1` 模块（推荐 `4.00` 个）；Arm 3 将 **Turn-1 推荐 `L1` 模块数** 从 `4.00` 降至 `2.625`（清单 `6,210 -> 4,769.2` 字符），但同时把全部 `6.00` 个合格 `L1` 模块写入 `/tmp/hegel_wormhole_skills` 供 `--describe` 按需回退查询。 | `M8.9.1` 将各渐进披露臂的按需 `--describe` 目录统一对称对齐为全部 `6` 个合格 `L1` 模块，并分别显式记录 `recommended_turn1_l1_count`、`mounted_on_disk_l1_count` 与 `queryable_catalog_l1_count`。 |
| **`hs01` 私有实例维度 $p$** | 正文写为 $p \in [14, 16]$。 | 3 个私有种子 `(99101, 99102, 99103)` 实际维度为 $p = (14, 14, 15)$（合计 `65,536` 个支撑子集）。 | 在本勘误及 `m8_9_hs01_independent_regrade_receipt.json` 中精确标注。 |
| **`hs01` 证书验证器完备性（`len >= 2` 与叶节点下界验证）** | 原 `hs01` 验证器强制 `2 <= len(leaves) <= 64`（导致 3 个实验臂在根结点 `gap = 0` 的合法单叶精确证书全部被误杀），且 `min_on_subcube` 仅枚举 $2^m$ 顶点而未求解连续箱约束凸 QP 下界或校验叶节点声称 `lb` 的真实性（存在伪造高 `lb` 假阳性漏洞）。 | `M8.9.1` 新增独立完备证书核 `verify_box_l0_subcube_certificate`（`src/hegel_machine/cognition/m8_9_1_box_l0_certificate_kernel.py`）：(a) 接受 `1 <= len(leaves) <= max_leaves`；(b) 对每个叶节点通过全部 $2^p$ 支撑上的精确连续凸 QP（$p \le 16$）或携带证明的 Fenchel 透视对偶（`dual_nu`）/ PSD 对角下界（`minorant_diagonal`, `minorant_center`）见证（任意 $p$）严格验证下界；(c) 正交分离输出 `solution_correct`、`partition_valid`、`leaf_bounds_valid`、`gap_satisfied` 与 `certificate_valid`。 | 已在 `R1` 闭环；下方给出 `M8.9` 全部封存提交物的独立重评结果（`18/18` 全通过，伪造下界负控严格拦截）。 |
| **路由器校准与真实 `LOTO` / `< 0.96` 机制归因** | 正文将 `< 0.96` 放宽归因于 `VERIFIED`，并将样本内回代结果与 `LOTO` 混述。 | `VERIFIED` 早于 `M8.8` 已豁免 `< 0.96`；`M8.8 -> M8.9` 实际放宽的是 `SUPPORTED_IN_PARTIAL_SCOPE`（改变 `tb__sparse-network-assimilation`）。同时 `m8_9_container_router_calibration.json` 为样本内回代（`14/20`）。 | `m8_9_1_router_true_loto_receipt.json` 补齐 `m7_4` 真实 20 折 `LOTO`：当前特征取得 **`14/20`（触发 `12` 次第二分支，`648` 轮）**，而移除该规则在 `LOTO` 下退化为 **`13/20`（`10` 次触发，`596` 轮）**（`m7_5` `LOTO` 均为 `12/12`）。同时修正了 `CONTAINER_STATE_FEATURE_NAMES` 第 `2..3` 列命名顺序并绑定 `feature_builder_sha256` 校验。 |
| **单种子 `+5.31%` / `-5.67%` Token 差值表述** | 报告了 Arm 3 相对 Arm 2 的单次运行 Token 差值。 | 在每个 `(arm, task)` 仅单种子（$N = 1$）条件下，主分支 `+5.31%` 与整策略 `-5.67%` 均处于单次采样随机波动范围内（主要由 `uf03` 分支波动驱动），不构成统计显著的净增益或退化。 | 本勘误采用中性校准表述；`M9.0a` 引入每单元多独立重复（`R = 2`）实验设计。 |

---

## 2. 封存 `M8.9` `HS01` 提交物的非破坏性独立重评回执

基于 `verify_box_l0_subcube_certificate`（`m891_hs01_private_verifier_sha256 = f25bcccb05f5b5ae61e452e6f52121833c39e9974c928fd8da97dffd419978c0`，详见 `docs/evidence/m8_9_1_audit_closure_and_regrade_20261126/m8_9_hs01_independent_regrade_receipt.json`）：
1. **GPT-6 伪造下界负控反例**（`reproduce_hs01_unsound_certificate.py`）：在真实最优值为 `0.2` 的实例上返回次优可行解 `true_objective = 0.6` 并伪造两片叶下界 `lb = 0.59999999`。**重评结论**：`strict_pass = False`、`leaf_bounds_valid = False`、`certificate_valid = False`、`solution_correct = False`（因 `unsound_leaf_lower_bound(leaf=1,claimed=0.600000>verified=0.300000)` 被确定性拦截）。
2. **`M8.9` 全部 `6` 份唯一 SHA-256 封存 `hs01` 产物**：在原始 3 个私有实例（`seeds = 99101, 99102, 99103`，$p = 14, 14, 15$，共 `65,536` 次精确连续凸 QP 求解）上，**全部 `18/18` 个 `(artifact, instance)` 配对均通过重评（`strict_pass = True, leaf_bounds_valid = True, solution_correct = True`）**，相对精确全局下界最大数值偏差 $\le 4.25 \times 10^{-7}$。

### 原始 `M8.9` 标签与 `M8.9.1` 独立重评并列对照（每臂 `8` 题）

| 实验臂 | 原始 `M8.9` 主分支 `1x` 通过数 | 原始 `M8.9` 整策略通过数 | `M8.9.1` 重评主分支 `1x` 通过数 | `M8.9.1` 重评整策略通过数 | 主分支 `hs01` 产物 SHA-256（前 12 位） | 整策略 `hs01` 产物 SHA-256（前 12 位） |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dedup_full_l1_l2`** | `7/8` (`87.5%`) | `8/8` (`100.0%`) | **`8/8` (`100.0%`)** | **`8/8` (`100.0%`)** | `7822e1f6498b` | `cd1bf2f4609b` |
| **`progressive_all_l1_l2`** | `7/8` (`87.5%`) | `7/8` (`87.5%`) | **`8/8` (`100.0%`)** | **`8/8` (`100.0%`)** | `b0c38f8e47ff` | `b0c38f8e47ff` |
| **`progressive_net_value_gated`** | `7/8` (`87.5%`) | `7/8` (`87.5%`) | **`8/8` (`100.0%`)** | **`8/8` (`100.0%`)** | `3ad794e2bf16` | `3ad794e2bf16` |

---

## 3. `M8.9.1` 七项确定性工程闭环摘要（`R1..R7`）

1. **`R1`（完备证书核 `m8_9_1_box_l0_certificate_kernel.py` + `hs01` + `PT06`）**：支持小维度全 $2^p$ 支撑精确凸 QP 验证（$p \le 16$）与任意维度携带证明的 Fenchel 透视对偶（`dual_nu`）/ PSD 对角下界（`minorant_diagonal`, `minorant_center`）多项式时间验证，合法接受 `1 <= len(leaves) <= max_leaves` 并彻底阻断伪造下界假阳性。
2. **`R2`（内容寻址缓存与严格资格门控）**：`enrich_and_harden_l1_primitive_contracts` 按 `(stem, code_sha256, test_sha256)` 寻址并仅向字节一致副本回写回执；`sublate_autogenous_forest_into_compositional_primitives` 按 `(filename, recomputed_source_sha256)` 收集多版本历史源码；`verify_l1_module_qualification_record` 对缺失/空/失配 SHA 回执严格 fail-closed 拦截；`M8` 容器修复注入真实前置条件摘要并标注 `"repaired_digest_kind": "raw_single_file_sha256"`。
3. **`R3`（两阶段原子回写与双向留存判定）**：`auto_package_wormhole_imports_in_container` 升级为主机端全文件预构建 + SHA 校验通过后再向容器提交的两阶段回写；`inspect_final_deliverable_skill_retention` 升级为有序辅助闭包双向集合相等 + `SimpleNamespace` 导出映射核验 + 作用域感知的入口可达静态调用图，并为全部任务补齐 `declared_callable_entrypoints`（`8/8` 全覆盖）。
4. **`R4`（实质性验证与命题绑定 `nogood` 消解）**：`is_substantive_verification_command` 拒绝纯 `python3 -c "import sol"` 冒烟命令；`nogood` 消解见证按 `risk_kind`（如 `target_identifiability` 强制要求显式锚点/规范不变性检验）与 `scope` 绑定校验。
5. **`R5`（路由器特征对齐与 `LOTO` 归档）**：对齐 `CONTAINER_STATE_FEATURE_NAMES` 列名与特征向量顺序，在 `from_json_dict` 中强制校验 `(feature_schema_version, feature_names, feature_builder_sha256)`，并归档真实 `LOTO` 回执。
6. **`R6`（持久化规划器签名纯化与时序探针）**：从 incumbent 签名 `inc_sig` 中剥离后验账本字段 `dependent_task_ids` / `inapplicable_task_ids`，并在 `select_discriminating_observation_probe` 中支持时序 `ObservationContext`。
7. **`R7`（数值规模门控与仅知识参照臂）**：引入显式数值未知量规模解析（`>= 4000 unknowns`）消除单词 `"iterative"` 误翻转，对齐渐进披露目录容量，并新增无可执行 Python 包的 `progressive_knowledge_only`（`/tmp/hegel_knowledge_ref`）模式以支撑 `M9.0a` 知识隔离对照实验。
