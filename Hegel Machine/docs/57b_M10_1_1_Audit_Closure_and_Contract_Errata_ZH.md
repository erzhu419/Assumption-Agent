# `M10.1.1` 审计闭环、证据契约升级与机制归因勘误报告（`U01–U12`，`0` 付费调用）

> **封存目录**：`docs/evidence/m10_1_1_audit_closure_and_contract_reverification_20261128/`（零修改既有 `m10_0*` / `m10_1_*` 原件与 `60/60/61` 得分）。

## 1. 对 `M10.0.1 + M10.1` 既有实现缺陷的完整清点与代码闭环

1. **`U02` 可交叠有效域误送互斥分类器（已闭环）**：原 `_induce_guards_for_table` 未做碰撞预检且未排除同输入双阳性重叠，导致 `8/8` 簇在首次分解后产生 **`86` 次调用 / `1,663,837 v2`** 零采纳守卫支出。现已合入 `build_feasible_domain_negative_witnesses` + `check_axis_aligned_all_of_guard_satisfiability`，在全部 `43` 个历史碰撞步实现 **`0` LLM 调用即时识别**，并支持可交叠候选集公开路由。
2. **`U03 + U04` 四链撤回语义外键与候选采纳事务分离（已闭环）**：升级 `verify_m101_cut_withdrawal_external_fk` 强绑定 `world_id / arm / old_dag_node_code_shas / new_dag_node_code_shas / public_input_sha256 / scientific_or_numerical` 及 `ProbeReceiptStore` 排他 `os.link` 写一次与非空图索引校验；对历史 `8` 张票复核确认 Hegel `2/6` 代码一致（`a02/S2, a04/S6`）、`4/6` 代码版本错配被新验证器严格拦截，全部 `8/8` 关系与并发负控拦截通过；同时将 `candidate_public_adoption` 与 `withdrawal_proof_issuance` 解耦，避免 `a03/S1`（已隔离回退账 `10` 调用 / `193,777 v2`）丢弃公开合格图。
3. **`U05 + U06` 运行时边消费/反包装空壳强校验与 `a02/S7, S8` 隔离容器诊断（已闭环）**：升级 `compile_m101_dag_to_instrumented_solver_code` 强制校验每条入边实际读取（`_TrackedPayloadDict`）与非汇节点新增数值变换，`3/3` 拒绝 `audit_semantic_shell_diamond`、`audit_unused_branch_diamond` 与 `audit_exact_identity`，同时 `a04/S4` 真实四节点菱形在容器中 `100%` 通过。在隔离 Docker 容器内对封存候选做事后诊断重放（注：宿主与容器 Python 版本不同致 `pickle` 原始字节摘要不同 `exact_container_public_sha_match = False`，但实际选中 `dag_07` 的公开/私判六位小数指标与原容器回执完全一致；仅作机制诊断，不改动原 `61/64` 得分）：`S7` 上纯公开 CV 选择器选中 `dag_06`（公开 `y_mae = 0.027161` < `dag_07` `0.035996`，诊断私判 `max_y_mae = 0.028308 <= 0.045`），`S8` 上保留 `dag_07`（公开 `0.034104` < `0.288225`）。
4. **`U09 + U11 + U12` 生产门接线与完整在线状态冷恢复（已闭环）**：`T05/T06` 已接入 `validate_public_witness` 与 `M10.0` `validate_public_schema.py`（开发正控 `2/2` 零误拒，原曝光失败候选 `alpha_m100_c03/S4` 与 `beta_m100_c04/S5` 被新生产门直接拦截，`alpha_m100_c03/S1` 保持临界通过，Layer 2 严格预算门实测拦截紧裕度 Beta KKT 证书）；`persist_m101_full_prefix_state` / `hydrate_m101_full_prefix_state` 持久化完整在线前缀并强验外键；`32/32` 标量 AST 提升普查明确界定为 `8 Stage S + 24 Mono` 定向样本与既有 `M6.8` 标量提升器范围。

## 2. 历史 `M10.1` 撤回票新验证器逐票复核表（`U03`）

| 世界 ID | 实验臂 | 票据声明转换 | prior/fail 节点代码集一致 | 新 v2 语义验证器裁决 |
|---|---|---|---|---|
| `conf_m101_a02` | `Hegel_UnknownCut_GraphDiscovery_Online` | `dag_01 -> dag_03` | `True` | **`strict_semantic_valid`** |
| `conf_m101_a02` | `Hegel_UnknownCut_GraphDiscovery_Online` | `dag_02 -> dag_06` | `False` | **`strict_semantic_rejected_historical_mismatch`** |
| `conf_m101_a02` | `Hegel_UnknownCut_GraphDiscovery_Online` | `dag_05 -> dag_07` | `False` | **`strict_semantic_rejected_historical_mismatch`** |
| `conf_m101_a03` | `Hegel_UnknownCut_GraphDiscovery_Online` | `dag_05 -> dag_07` | `False` | **`strict_semantic_rejected_historical_mismatch`** |
| `conf_m101_a04` | `Hegel_UnknownCut_GraphDiscovery_Online` | `dag_06 -> dag_07` | `True` | **`strict_semantic_valid`** |
| `conf_m101_b01` | `Fixed_TwoStage_Modular_Online` | `dag_06 -> dag_07` | `True` | **`strict_semantic_valid`** |
| `conf_m101_b01` | `Hegel_UnknownCut_GraphDiscovery_Online` | `dag_01 -> dag_05` | `False` | **`strict_semantic_rejected_historical_mismatch`** |
| `conf_m101_b02` | `Fixed_TwoStage_Modular_Online` | `dag_07 -> dag_10` | `False` | **`strict_semantic_rejected_historical_mismatch`** |
