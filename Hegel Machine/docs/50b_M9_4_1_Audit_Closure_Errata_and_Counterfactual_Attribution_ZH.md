# Hegel M9.4.1 确定性正确性门闭环、机器生成勘误与反事实归因补遗

> **机器自动生成声明**：本文档所有数字均由 `scripts/generate_m9_4_1_audit_closure_and_counterfactual_receipts.py` 直接从封存 JSON 证据计算并经断言校验生成（`0` 人工手抄数字，`0` 付费模型调用）。

## 1. `docs/48b` 与 `docs/50` 机械复核勘误表（`A10` 闭环）

| 勘误条目 | 机器核实的精确真值 | 权威证据来源 |
| :--- | :--- | :--- |
| **`M9.3` 8 局后继计费代理总开销（`B / K / E`）** | `B = 1,950,809`，`K = 1,909,591`，`E = 1,927,667`（`E - B = -23,142`，即 `-1.1863%`；`E - K = +18,076`） | `m9_3_online_source_freeze_lifecycle_20261126/**/00_episode_summary.json` |
| **`M9.3` 新源 `L2` 正向调用局明细** | `[{"episode": "rep_01/equal_source_executable_library/rec02_R__original_separable_kronecker_kkt_heldout_recovery", "arm": "stage_R", "calls": 3}, {"episode": "rep_01/suc01_C__viscoplastic_bingham_overstress_counterexample_revision", "arm": "equal_source_executable_library", "calls": 1018}, {"episode": "rep_02/suc02_C__nonseparable_vortex_cross_coupling_counterexample_revision", "arm": "equal_source_executable_library", "calls": 3}]`（第二处 `3` 次调用来自 `rep_02/suc02_C`，而非 `suc02_T`） | `m9_3_source_aware_l1_l2_adoption_funnel.json` |
| **`M9.3` 未知用量超时账本定位** | `[{"rel_path": "stage_S_sources/attempt_01/src02__kronecker_sylvester_multishift_and_box_simplex_kkt_source/01b_provider_attempts/attempts_003.json", "call_seq": 3, "usage_unknown_attempt_count": 1}, {"rel_path": "stage_S_sources/attempt_01/src02__kronecker_sylvester_multishift_and_box_simplex_kkt_source/01b_provider_attempts/attempts_012.json", "call_seq": 12, "usage_unknown_attempt_count": 1}, {"rel_path": "stage_TC_successors/rep_01/equal_source_knowledge_only/suc01_C__viscoplastic_bingham_overstress_counterexample_revision/01b_provider_attempts/attempts_011.json", "call_seq": 11, "usage_unknown_attempt_count": 1}]` | `m9_3_online_source_freeze_lifecycle_20261126/**/attempts_*.json` |
| **`M9.3` 历史反例分裂父节点 ID** | 旧错误父节点：`node_auto_07_control_panel_deconfounding`；真实 `src01` 源父节点：`node_auto_10_hereditary_spectrum_identifi` | `m9_3_post_counterexample_bifurcated_bundle.json` |
| **`M9.4` 原始调用账本覆盖边界** | 全局共 `321` 条调用摘要：其中 `291` 份求解 `attempts_*.json` 已随包封存；另有 `30` 条编译调用仅存摘要与临时路径（已在 `M9.4.1` 通过 `immediate_archive_dir` 原子直写修复） | `all_provider_call_receipts.json` |
| **`M9.4` 早期开发轮（`d44b3314`）诚实披露** | 提交 `d44b3314f557`：`superseded_development_run_evidence_unrecoverable_on_disk` —— 因 `ep01` 瞬态子机制残留抑制 `AncestorReweight` 在修正前目录被覆盖，原始文件不可恢复，其研发成本未知且不混入最终单轮 `6,825,968` 代理值 | `gate_c_reconciliation_and_errata.json` |

## 2. `M9.4.1` 四道正确性门（`Gates A–D`）验证摘要

- **门 A（共享隔离验收核与完整 7 维系数交接契约）**：
  - `hereditary_weights = zeros(7)` 被公开门拒绝：`True`，被私有门拒绝：`True`。
  - `hereditary_weights[2:7] = 1_000_000` 被公开门拒绝：`True`，被私有门拒绝：`True`。
  - 对封存 `M9.4` 全部 `34` 个 episode 的离线 vNext 紧化重评：`34 / 34` 通过完整 7 维系数与标准 KKT 共享核（不一致数：`0`）。
- **门 B（严格三值守卫、绑定外键身份链、AST 定义域守卫、按函数分层 `F08`、`F05` 精确且可达）**：
  - 6 组守卫边界控制输出：`{"valid_subyield": {"state": "TRUE", "measured_value": 0.0}, "missing_yield": {"state": "UNKNOWN", "measured_value": null}, "nan_history_valid_hold": {"state": "UNKNOWN", "measured_value": null}, "nan_yield_valid_history": {"state": "UNKNOWN", "measured_value": null}, "missing_history_current_below": {"state": "UNKNOWN", "measured_value": null}, "valid_history_residual_damage": {"state": "FALSE", "measured_value": 0.4}}`。
  - 篡改声明 SHA 被拒绝：`True`；无效资格 ID 被拒绝：`True`；`TypedMechanism` 外键真实注册：`True`。
  - AST `Program("domain_guard", ...)` 域外抛出 `DomainViolationError`：`True`。
  - 按公共函数分层捕获 `F08` 拒绝 `24 次稳定调用 + 1 次无种子随机函数`：`True`。
  - `F05` 改写后可达模块归类为 `adapted_closure_reachable`（不再误标 `exact`）：`True`。
- **门 D（真实 `Stage S` 源码反事实归因结论）**：
  - 实测证实当每批均附完整校准表时，真实 `Stage S` 求解器自带的本批校准已可通过 `Alpha G / Alpha R / Beta G`（`Beta R` 加同一公开加性势变换亦通过），因此所谓“旧 `Stage S` 必败”仅适用于固定旧参数适配器，不适用于原源自动重放。`M9.5` 据此为全部三臂提供对等的自动重放与公开门服务，并在统一输出契约下区分“当批信息充分负控”与“稀疏当批证据 + 历史必需”两类条件。
