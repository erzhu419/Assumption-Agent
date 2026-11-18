# 来源清单、39 份证据回执索引与核查范围（v15）

- **历史冻结回执（35 份，100% 原始字节零改动）：** `v10..v14` 全量保留，`1001000–1046000` 全部退居已曝光开发／诊断材料；
- **`v15` 全新严格 JSON（`allow_nan=False`）回执（4 份，零新增 `src/` 模块）：**
  1. `deliverable_1_v14_errata_and_factual_reexport_20261118.json`（V14 10 项勘误 + 已曝光 6 条 D3 链真实生产事件与状态感知重导出）；
  2. `deliverable_2_removal_replay_parity_and_exposed_diagnostics_20261118.json`（GPT-6 V15 探针修复 + §3.3 三项单版本移除与计费重建控制 + 8 源簇 24 对 K/R 末态与调度零分歧 + §5.2 六个受影响案例回放）；
  3. `deliverable_3_prospective_e1_same_history_organization_20261118.json`（`1051000–1056000` 前瞻 `E1`：同原始历史、同一状态转移函数下 `K` vs `R` vs 真实修复版本移除且允许重建 vs `cold`，`k=1,3,6` 前缀全账）；
  4. `deliverable_4_prospective_e2_end_to_end_strategy_20261118.json`（`1051000–1056000` 前瞻 `E2`：零拼接端到端策略比较，强普通方法用自身实际形成的 `L^Ord_c` 对阵递归方法 `L^K_c`）。

## 一键零依赖沙箱核验
```bash
PYTHONPATH="" python3 05_STANDALONE_VERIFY_V15.py --output verify_v15_report.json
```

| 阶段 | 回执文件 | 角色分类 (`study_role`) | 字节数 | SHA256 前16位 |
| :--- | :--- | :--- | ---: | :--- |
| `v10_world_structure_20261116` | `paid_position_round1_census_20261116.json` | `historical_read_only_workspace_census` | `124,838` | `2423df4dcb372709…` |
| `v10_world_structure_20261116` | `v10_ablation_and_cross_scale_successor_20261116.json` | `capability_ablation_and_cross_scale_fixture` | `26,920` | `87bdac7844f40f21…` |
| `v10_world_structure_20261116` | `v10_cow_602_603_new_ledger_20261116.json` | `historical_development_continuation_comparison` | `21,587` | `3983109cffbd0892…` |
| `v10_world_structure_20261116` | `v10_phase2a_expanded_revisions_20261116.json` | `solver_and_premise_revision_capability_ablation` | `28,075` | `f92f536962911a54…` |
| `v10_world_structure_20261116` | `v10_phase2b_open_grammar_llm_20261116.json` | `workflow_multi_factor_llm_discovery_and_frozen_transfer` | `44,923` | `17a9e2bde572473a…` |
| `v10_world_structure_20261116` | `v10_phase2c_active_threshold_and_pomdp_20261116.json` | `active_probe_and_solver_capability_ablation` | `17,828` | `250f8ec38ea0adee…` |
| `v10_world_structure_20261116` | `v10_phase3a_higher_order_and_coupled_pomdp_20261116.json` | `solver_model_class_ablation` | `15,975` | `132227729217b7a0…` |
| `v10_world_structure_20261116` | `v10_phase3b_open_grammar_pomdp_llm_20261116.json` | `workflow_multi_factor_llm_pomdp_discovery_and_frozen_transfer` | `40,832` | `cc9803d4c22eb318…` |
| `v10_world_structure_20261116` | `v10_phase4a_curved_hypersurface_probing_20261116.json` | `active_probe_and_boundary_capability_study` | `30,827` | `69ffe79989b1ae42…` |
| `v10_world_structure_20261116` | `v10_phase4b_continuum_pde_and_vdp_20261116.json` | `solver_and_representation_capability_ablation_with_holdout_negative` | `24,259` | `e7a116f63e546e82…` |
| `v10_world_structure_20261116` | `v10_phase5a_manifold_and_graph_pde_20261116.json` | `solver_and_representation_capability_ablation` | `27,641` | `485a3a0934499de3…` |
| `v10_world_structure_20261116` | `v10_phase5b_stochastic_sde_and_resonance_20261116.json` | `estimator_and_active_probe_capability_ablation` | `23,591` | `d63a5b9e8d17a76f…` |
| `v10_world_structure_20261116` | `v10_phase5c_algebraic_proofs_and_impossibility_20261116.json` | `supplied_model_solver_and_checker_qualification` | `13,227` | `a1ae89bd9da06205…` |
| `v10_2_audit_remediation_20261116` | `equal_tool_attribution_20261116.json` | `historic_untouched_receipt` | `65,358` | `edd01975a655431b…` |
| `v10_2_audit_remediation_20261116` | `errata_and_sha256_ledger_20261116.json` | `historic_untouched_receipt` | `25,161` | `efe3a5076e11259c…` |
| `v10_2_audit_remediation_20261116` | `frozen_v4_lifecycle_20261116.json` | `historic_untouched_receipt` | `90,893` | `ec58960838665b48…` |
| `v10_2_audit_remediation_20261116` | `post_fix_domain_probe_20261116.json` | `historic_untouched_receipt` | `4,056` | `03f50cc968fc836b…` |
| `v10_2_audit_remediation_20261116` | `pre_fix_a66cd82b_domain_probe_20261116.json` | `historic_untouched_receipt` | `2,279` | `9550076632152645…` |
| `v10_2_audit_remediation_20261116` | `sde_and_proof_remediation_20261116.json` | `historic_untouched_receipt` | `34,035` | `592ae8f01bb83acb…` |
| `v10_3_scaling_and_calibration_20261117` | `high_arity_sensor_calibration_20261117.json` | `historic_untouched_receipt` | `34,739` | `734b0d70b5b27d3a…` |
| `v10_3_scaling_and_calibration_20261117` | `library_scaling_curve_20261117.json` | `historic_untouched_receipt` | `132,233` | `ceff49583506b5d7…` |
| `v10_4_occam_and_gpt6_bundle_20261117` | `occam_minimality_ablation_20261117.json` | `historic_untouched_receipt` | `175,738` | `9370ca4482318e59…` |
| `v11_branch_lifecycle_and_km_20261117` | `deliverable_a_credibility_and_baselines_20261117.json` | `historic_untouched_receipt` | `65,123` | `cc955898d7f18133…` |
| `v11_branch_lifecycle_and_km_20261117` | `deliverable_b_branch_lifecycle_20261117.json` | `historic_untouched_receipt` | `58,705` | `5fa9c4d9c0a887d5…` |
| `v11_branch_lifecycle_and_km_20261117` | `deliverable_c_knowledge_vs_method_2x2_20261117.json` | `historic_untouched_receipt` | `51,792` | `1da74922233f761e…` |
| `v11_branch_lifecycle_and_km_20261117` | `deliverable_c_v11_1_method_remediation_2x2_20261117.json` | `historic_untouched_receipt` | `56,369` | `ea0ccfebab86cd55…` |
| `v12_autonomous_reconciliation_and_utility_20261117` | `deliverable_a_v12_semantics_and_errata_20261117.json` | `historic_untouched_receipt` | `12,306` | `01621d5553ffd052…` |
| `v12_autonomous_reconciliation_and_utility_20261117` | `deliverable_b_v12_autonomous_lifecycle_20261117.json` | `historic_untouched_receipt` | `4,096,786` | `bcc6422a4e693542…` |
| `v12_autonomous_reconciliation_and_utility_20261117` | `deliverable_c_v12_knowledge_utility_20261117.json` | `historic_untouched_receipt` | `12,400,526` | `7004fda92deeef10…` |
| `v13_evidence_recursion_and_routing_20261117` | `pr_a_evidence_and_recursion_controls_20261117.json` | `historic_untouched_receipt` | `606,215` | `afa86eb33307030c…` |
| `v13_evidence_recursion_and_routing_20261117` | `pr_b_routing_vs_applicability_and_cache_20261117.json` | `historic_untouched_receipt` | `18,965` | `3f455dfc761900f4…` |
| `v13_evidence_recursion_and_routing_20261117` | `pr_c_tightened_source_paired_study_20261117.json` | `historic_untouched_receipt` | `10,380,190` | `03409d0aa986e2ad…` |
| `v14_continuous_object_and_fair_comparison_20261118` | `deliverable_1_errata_and_replays_a_to_f_20261118.json` | `historic_v14_untouched` | `24,061` | `0e1e7ba58f3926f7…` |
| `v14_continuous_object_and_fair_comparison_20261118` | `deliverable_2_unconfounded_noH_source_paired_study_20261118.json` | `historic_v14_exposed_retired` | `9,071,184` | `9076698f6e8314b2…` |
| `v14_continuous_object_and_fair_comparison_20261118` | `deliverable_3_continuous_repaired_object_study_20261118.json` | `historic_v14_exposed_retired` | `13,594,125` | `6c9e09a6f2fb2211…` |
| `v15_true_intervention_and_strategy_20261118` | `deliverable_1_v14_errata_and_factual_reexport_20261118.json` | `v15_deliverable_1_errata_and_status_aware_reexport` | `1,789,649` | `90f6e08845354f71…` |
| `v15_true_intervention_and_strategy_20261118` | `deliverable_2_removal_replay_parity_and_exposed_diagnostics_20261118.json` | `v15_deliverable_2_controls_parity_and_exposed_replays` | `627,325` | `987d24691a7a5c3e…` |
| `v15_true_intervention_and_strategy_20261118` | `deliverable_3_prospective_e1_same_history_organization_20261118.json` | `v15_deliverable_3_prospective_e1_same_history` | `17,078,013` | `ebef638056d9bceb…` |
| `v15_true_intervention_and_strategy_20261118` | `deliverable_4_prospective_e2_end_to_end_strategy_20261118.json` | `v15_deliverable_4_prospective_e2_end_to_end` | `13,725,854` | `f53eb81e84c550c6…` |
