# `M10.0.1` 审计闭环与探针原始回执落盘勘误（`T01–T11`，`0` 次付费模型调用）

> **独立封存事后复核目录**：`docs/evidence/m10_0_1_audit_closure_and_probe_reverification_20261128/`（共 `44` 个经 `SHA256SUMS.txt` 校验的文件）。全程零修改任何既有 `m9_9*` 或 `m10_0_*` 封存目录及历史得分（`72/74/77`）。

## 1. `T01` 缺失原始探针回执清单与 `41/41` 真实 Docker 事后补测

- **缺失回执清点（`01_m100_probe_persistence_inventory.json`）**：
  - `M10.0` 开发集：`19/107` 条证人标记为 `recorded_without_raw_measurement`（`{'pprobe': 15, 'uprobe': 4}`）。
  - `M10.0` 确认集：**`41/216`** 条证人标记为 `recorded_without_raw_measurement`（`{'pprobe': 32, 'uprobe': 9}`），其中包含 **`8/8`** 条撤回前置正证人。
- **真实容器并行事后补测（`02_m100_41_missing_probes_docker_backverification.json`）**：
  - 全部 **`41`** 条缺失确认探针（`32` 条 `pprobe` + `9` 条 `uprobe`）已在 `hegel-clean-scipy:m7v1` 容器中按封存代码、参数纪元与只读依赖原样重放（耗时 `96.09s`，`0` 次付费模型调用）。
  - **补测结论**：**`41/41`（`100.0%`）** 与原证人记录裁决完全一致，包含 **`8/8`（`100.0%`）** 撤回前置正证人。所有补测文件严格标注 `raw_measurement_status = "backfilled_posthoc_replay"`、新 `backfill_*` ID 与 `supersedes_probe_id`，绝不伪装为原运行期落盘。
- **前向架构强制闭环**：新增写一次 `ProbeReceiptStore`、`SolverEvaluationWitness` v3 外部回执摘要字段、将 `run_m10_0` 全部裸证人写入迁移至 `record_tested_witness_from_probe_receipt`、保留 `B2` 主/备双诊断，并在撤回签发与冷加载中强制三方外部回执外键（`tests/test_m10_0_1_audit_closure_suite.py` 已复现并严格拒绝 GPT-6 的伪造 `prior.probe_id` 负控）。

## 2. 归档求解器零调用 AST 提升与反合一覆盖盘点（`03_m100_archived_solvers_lifting_coverage_inventory.json`）

- 对 `32` 个 `M10.0` 确认集归档单体求解器运行旧版 `M6.8` 标量 AST 提升器与 `anti_unify_programs`：
  - 分类统计：`{'extracted_unadopted': 0, 'lifter_unsupported': 32, 'no_shared_skeleton': 0}`；
  - 标量提升程序数：`0`；跨簇非平凡反合一骨架对数：`0`。
- **工程结论**：科学求解器主体为命令式 NumPy/SciPy 数值例程（落入 `lifter_unsupported`），印证 `M10.1` 开放多模块计算图发现必须直接建立在可执行 Python 模块节点与实测 I/O 契约之上，而非单纯依赖旧版标量 AST 提升。

## 3. 全部 `41/41` 缺失确认探针事后补测明细表

| 源簇 ID | 证人 ID | 证据来源 | 原 `probe_id` | 补测 `probe_id` | 撤回前置正证人？ | 原记录 vs 重放 | 一致？ |
|---|---|---|---|---|---:|---|---|
| `alpha_m100_c01` | `wit_0005_bind_alpha_m100_c01_up_b01_e8ce6588` | `joint_task_probe` | `pprobe_9cfe4611634442db` | `backfill_pprobe_9cfe4611634442db` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c01` | `wit_0006_bind_alpha_m100_c01_dn_b02_cfe8ecc8` | `joint_task_probe` | `pprobe_b7b53350516d4026` | `backfill_pprobe_b7b53350516d4026` | `True` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c01` | `wit_0008_bind_alpha_m100_c01_up_b01_2d495579` | `direct_component_probe` | `uprobe_563241be8db84ff5` | `backfill_uprobe_563241be8db84ff5` | `False` | `failed` vs `failed` | **`True`** |
| `alpha_m100_c01` | `wit_0009_bind_alpha_m100_c01_up_b02_2d495579` | `joint_task_probe` | `pprobe_381c2b6144df4513` | `backfill_pprobe_381c2b6144df4513` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c01` | `wit_0011_bind_alpha_m100_c01_dn_b03_31b8c081` | `joint_task_probe` | `pprobe_c0041f8fcfca4c5a` | `backfill_pprobe_c0041f8fcfca4c5a` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c02` | `wit_0004_bind_alpha_m100_c02_up_b01_2e36a86d` | `joint_task_probe` | `pprobe_12cd75eb32924c45` | `backfill_pprobe_12cd75eb32924c45` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c02` | `wit_0007_bind_alpha_m100_c02_dn_b02_7a35fe59` | `joint_task_probe` | `pprobe_8f0a149cff7a4561` | `backfill_pprobe_8f0a149cff7a4561` | `True` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c02` | `wit_0008_bind_alpha_m100_c02_up_b01_a00ea06d` | `direct_component_probe` | `uprobe_78b5844273284d8f` | `backfill_uprobe_78b5844273284d8f` | `False` | `failed` vs `failed` | **`True`** |
| `alpha_m100_c02` | `wit_0009_bind_alpha_m100_c02_up_b02_a00ea06d` | `joint_task_probe` | `pprobe_a57b3696f1374dd8` | `backfill_pprobe_a57b3696f1374dd8` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c02` | `wit_0010_bind_alpha_m100_c02_dn_b03_b74aecdd` | `joint_task_probe` | `pprobe_ed8002da7b1e4748` | `backfill_pprobe_ed8002da7b1e4748` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c03` | `wit_0005_bind_alpha_m100_c03_up_b01_35a330cf` | `joint_task_probe` | `pprobe_2fcef5a9fc4e4bc1` | `backfill_pprobe_2fcef5a9fc4e4bc1` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c03` | `wit_0006_bind_alpha_m100_c03_dn_b02_c2b83bba` | `joint_task_probe` | `pprobe_44d9575412c24720` | `backfill_pprobe_44d9575412c24720` | `True` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c03` | `wit_0008_bind_alpha_m100_c03_up_b01_3e170931` | `direct_component_probe` | `uprobe_eb9046b06c144754` | `backfill_uprobe_eb9046b06c144754` | `False` | `failed` vs `failed` | **`True`** |
| `alpha_m100_c03` | `wit_0009_bind_alpha_m100_c03_up_b02_3e170931` | `joint_task_probe` | `pprobe_1df4bf8e0c564897` | `backfill_pprobe_1df4bf8e0c564897` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c03` | `wit_0011_bind_alpha_m100_c03_dn_b03_d6f2f32c` | `joint_task_probe` | `pprobe_7e00ebbcf9864a9c` | `backfill_pprobe_7e00ebbcf9864a9c` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c04` | `wit_0004_bind_alpha_m100_c04_up_b01_f2244f00` | `joint_task_probe` | `pprobe_7641e285f0654580` | `backfill_pprobe_7641e285f0654580` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c04` | `wit_0007_bind_alpha_m100_c04_dn_b02_b493573f` | `joint_task_probe` | `pprobe_1fdc31a3cd7e4d7d` | `backfill_pprobe_1fdc31a3cd7e4d7d` | `True` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c04` | `wit_0008_bind_alpha_m100_c04_up_b01_dd8649e5` | `direct_component_probe` | `uprobe_6a8bd8fc6c304956` | `backfill_uprobe_6a8bd8fc6c304956` | `False` | `failed` vs `failed` | **`True`** |
| `alpha_m100_c04` | `wit_0009_bind_alpha_m100_c04_up_b02_dd8649e5` | `joint_task_probe` | `pprobe_270fea190d7c40c8` | `backfill_pprobe_270fea190d7c40c8` | `False` | `passed` vs `passed` | **`True`** |
| `alpha_m100_c04` | `wit_0010_bind_alpha_m100_c04_dn_b03_f3c69c5d` | `joint_task_probe` | `pprobe_f35231f9373048a8` | `backfill_pprobe_f35231f9373048a8` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c01` | `wit_0005_bind_beta_m100_c01_up_b01_7528934e` | `joint_task_probe` | `pprobe_dd600d0799114516` | `backfill_pprobe_dd600d0799114516` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c01` | `wit_0006_bind_beta_m100_c01_dn_b02_27b4710e` | `joint_task_probe` | `pprobe_3dbd6d2de2ca4a2d` | `backfill_pprobe_3dbd6d2de2ca4a2d` | `True` | `passed` vs `passed` | **`True`** |
| `beta_m100_c01` | `wit_0008_bind_beta_m100_c01_up_b01_e7ec38f6` | `direct_component_probe` | `uprobe_31a67c992f0e4e67` | `backfill_uprobe_31a67c992f0e4e67` | `False` | `failed` vs `failed` | **`True`** |
| `beta_m100_c01` | `wit_0009_bind_beta_m100_c01_up_b02_e7ec38f6` | `joint_task_probe` | `pprobe_1000490ab5074d6a` | `backfill_pprobe_1000490ab5074d6a` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c01` | `wit_0011_bind_beta_m100_c01_dn_b03_3f8467fe` | `joint_task_probe` | `pprobe_bfde598286af47bb` | `backfill_pprobe_bfde598286af47bb` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c02` | `wit_0004_bind_beta_m100_c02_up_b01_a06703f6` | `joint_task_probe` | `pprobe_0d6e020f6216497b` | `backfill_pprobe_0d6e020f6216497b` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c02` | `wit_0007_bind_beta_m100_c02_dn_b02_b9a60bfe` | `joint_task_probe` | `pprobe_195e88b7b5ca4610` | `backfill_pprobe_195e88b7b5ca4610` | `True` | `passed` vs `passed` | **`True`** |
| `beta_m100_c02` | `wit_0008_bind_beta_m100_c02_up_b01_c4142fb8` | `direct_component_probe` | `uprobe_4665ecb98718492c` | `backfill_uprobe_4665ecb98718492c` | `False` | `failed` vs `failed` | **`True`** |
| `beta_m100_c02` | `wit_0009_bind_beta_m100_c02_up_b02_c4142fb8` | `joint_task_probe` | `pprobe_a299909c771b4923` | `backfill_pprobe_a299909c771b4923` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c02` | `wit_0010_bind_beta_m100_c02_dn_b03_e563d504` | `joint_task_probe` | `pprobe_c746c9e36f9a4da5` | `backfill_pprobe_c746c9e36f9a4da5` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c03` | `wit_0005_bind_beta_m100_c03_up_b01_ddbc9e27` | `joint_task_probe` | `pprobe_85004a39ab544feb` | `backfill_pprobe_85004a39ab544feb` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c03` | `wit_0006_bind_beta_m100_c03_dn_b02_7d7e5c94` | `joint_task_probe` | `pprobe_7e6426367ca8408e` | `backfill_pprobe_7e6426367ca8408e` | `True` | `passed` vs `passed` | **`True`** |
| `beta_m100_c03` | `wit_0008_bind_beta_m100_c03_up_b01_72168d3c` | `direct_component_probe` | `uprobe_149f45eb92984c2b` | `backfill_uprobe_149f45eb92984c2b` | `False` | `failed` vs `failed` | **`True`** |
| `beta_m100_c03` | `wit_0009_bind_beta_m100_c03_up_b02_72168d3c` | `joint_task_probe` | `pprobe_08b46660d7914687` | `backfill_pprobe_08b46660d7914687` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c03` | `wit_0011_bind_beta_m100_c03_dn_b03_6f9fba05` | `joint_task_probe` | `pprobe_cbb6e782e2864ca8` | `backfill_pprobe_cbb6e782e2864ca8` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c04` | `wit_0004_bind_beta_m100_c04_up_b01_fc22afcb` | `joint_task_probe` | `pprobe_25457f9a1bed4d16` | `backfill_pprobe_25457f9a1bed4d16` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c04` | `wit_0007_bind_beta_m100_c04_dn_b02_7cda1fab` | `joint_task_probe` | `pprobe_e719d2a817fe479e` | `backfill_pprobe_e719d2a817fe479e` | `True` | `passed` vs `passed` | **`True`** |
| `beta_m100_c04` | `wit_0008_bind_beta_m100_c04_up_b01_54615d04` | `direct_component_probe` | `uprobe_0421fbdfa59846b5` | `backfill_uprobe_0421fbdfa59846b5` | `False` | `failed` vs `failed` | **`True`** |
| `beta_m100_c04` | `wit_0009_bind_beta_m100_c04_dn_b03_3b4d43a5` | `joint_task_probe` | `pprobe_4a1046e8970e4fbe` | `backfill_pprobe_4a1046e8970e4fbe` | `False` | `passed` vs `passed` | **`True`** |
| `beta_m100_c04` | `wit_0010_bind_beta_m100_c04_up_b01_bb129c73` | `direct_component_probe` | `uprobe_fa8a98379f344079` | `backfill_uprobe_fa8a98379f344079` | `False` | `failed` vs `failed` | **`True`** |
| `beta_m100_c04` | `wit_0011_bind_beta_m100_c04_up_b02_bb129c73` | `joint_task_probe` | `pprobe_16f03b58993a4f13` | `backfill_pprobe_16f03b58993a4f13` | `False` | `passed` vs `passed` | **`True`** |
