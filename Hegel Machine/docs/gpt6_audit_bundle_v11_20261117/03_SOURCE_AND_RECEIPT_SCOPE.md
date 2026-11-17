# 来源清单、25 份证据回执分级索引与核查范围（v11）

本包完整收录：
- **历史冻结回执（22 份，100% 原始字节零修改）：** `v10`（13 份）、`v10.2`（6 份）、`v10.3`（2 份）、`v10.4`（1 份）；
- **`v11` 新增实证回执（3 份）：** `deliverable_a_credibility_and_baselines_20261117.json`、`deliverable_b_branch_lifecycle_20261117.json`、`deliverable_c_knowledge_vs_method_2x2_20261117.json`；
- **随包完整未截断源码／脚本／测试（23 个文件）：** 含 `unified_research.py` 全文、新增 `branch_reconciliation.py`、`discriminating_experiment_policy.py` 及 `03_fingerprint_probe.py`。

## 一、25 份 JSON 证据回执全量索引

| 阶段 | 回执文件 | 实验角色分类 (`study_role`) | 独立物理源数 | 字节数 | SHA256 前16位 |
| :--- | :--- | :--- | :---: | ---: | :--- |
| `v10_world_structure_20261116` | `paid_position_round1_census_20261116.json` | `historical_read_only_workspace_census` | `0` | `124,838` | `2423df4dcb372709…` |
| `v10_world_structure_20261116` | `v10_ablation_and_cross_scale_successor_20261116.json` | `capability_ablation_and_cross_scale_fixture` | `6` | `26,920` | `87bdac7844f40f21…` |
| `v10_world_structure_20261116` | `v10_cow_602_603_new_ledger_20261116.json` | `historical_development_continuation_comparison` | `2` | `21,587` | `3983109cffbd0892…` |
| `v10_world_structure_20261116` | `v10_phase2a_expanded_revisions_20261116.json` | `solver_and_premise_revision_capability_ablation` | `6` | `28,075` | `f92f536962911a54…` |
| `v10_world_structure_20261116` | `v10_phase2b_open_grammar_llm_20261116.json` | `workflow_multi_factor_llm_discovery_and_frozen_transfer` | `4` | `44,923` | `17a9e2bde572473a…` |
| `v10_world_structure_20261116` | `v10_phase2c_active_threshold_and_pomdp_20261116.json` | `active_probe_and_solver_capability_ablation` | `8` | `17,828` | `250f8ec38ea0adee…` |
| `v10_world_structure_20261116` | `v10_phase3a_higher_order_and_coupled_pomdp_20261116.json` | `solver_model_class_ablation` | `4` | `15,975` | `132227729217b7a0…` |
| `v10_world_structure_20261116` | `v10_phase3b_open_grammar_pomdp_llm_20261116.json` | `workflow_multi_factor_llm_pomdp_discovery_and_frozen_transfer` | `4` | `40,832` | `cc9803d4c22eb318…` |
| `v10_world_structure_20261116` | `v10_phase4a_curved_hypersurface_probing_20261116.json` | `active_probe_and_boundary_capability_study` | `6` | `30,827` | `69ffe79989b1ae42…` |
| `v10_world_structure_20261116` | `v10_phase4b_continuum_pde_and_vdp_20261116.json` | `solver_and_representation_capability_ablation_with_holdout_negative` | `8` | `24,259` | `e7a116f63e546e82…` |
| `v10_world_structure_20261116` | `v10_phase5a_manifold_and_graph_pde_20261116.json` | `solver_and_representation_capability_ablation` | `12` | `27,641` | `485a3a0934499de3…` |
| `v10_world_structure_20261116` | `v10_phase5b_stochastic_sde_and_resonance_20261116.json` | `estimator_and_active_probe_capability_ablation` | `8` | `23,591` | `d63a5b9e8d17a76f…` |
| `v10_world_structure_20261116` | `v10_phase5c_algebraic_proofs_and_impossibility_20261116.json` | `supplied_model_solver_and_checker_qualification` | `8` | `13,227` | `a1ae89bd9da06205…` |
| `v10_2_audit_remediation_20261116` | `equal_tool_attribution_20261116.json` | `equal_tool_knowledge_and_single_factor_attribution` | `19` | `65,358` | `edd01975a655431b…` |
| `v10_2_audit_remediation_20261116` | `errata_and_sha256_ledger_20261116.json` | `read_only_sha256_and_errata_index` | `0` | `25,161` | `efe3a5076e11259c…` |
| `v10_2_audit_remediation_20261116` | `frozen_v4_lifecycle_20261116.json` | `frozen_six_stage_prospective_lifecycle` | `5` | `90,893` | `ec58960838665b48…` |
| `v10_2_audit_remediation_20261116` | `post_fix_domain_probe_20261116.json` | `remediated_domain_soundness_probe` | `0` | `4,056` | `03f50cc968fc836b…` |
| `v10_2_audit_remediation_20261116` | `pre_fix_a66cd82b_domain_probe_20261116.json` | `historical_bug_reproduction_probe` | `0` | `2,279` | `9550076632152645…` |
| `v10_2_audit_remediation_20261116` | `sde_and_proof_remediation_20261116.json` | `domain_semantics_learned_bridge_and_sde_drift_reaudit` | `5` | `34,035` | `592ae8f01bb83acb…` |
| `v10_3_scaling_and_calibration_20261117` | `high_arity_sensor_calibration_20261117.json` | `single_factor_high_arity_role_calibration_and_falsifiability` | `16` | `34,739` | `734b0d70b5b27d3a…` |
| `v10_3_scaling_and_calibration_20261117` | `library_scaling_curve_20261117.json` | `historical_v10_3_scaling_receipt` | `4` | `132,233` | `ceff49583506b5d7…` |
| `v10_4_occam_and_gpt6_bundle_20261117` | `occam_minimality_ablation_20261117.json` | `historical_v10_4_occam_receipt` | `9` | `175,738` | `9370ca4482318e59…` |
| `v11_branch_lifecycle_and_km_20261117` | `deliverable_a_credibility_and_baselines_20261117.json` | `v11_deliverable_a_cache_credibility_and_strong_baselines` | `4` | `65,123` | `cc955898d7f18133…` |
| `v11_branch_lifecycle_and_km_20261117` | `deliverable_b_branch_lifecycle_20261117.json` | `v11_deliverable_b_8_stage_branch_reconciliation_lifecycle` | `20` | `58,705` | `5fa9c4d9c0a887d5…` |
| `v11_branch_lifecycle_and_km_20261117` | `deliverable_c_knowledge_vs_method_2x2_20261117.json` | `v11_deliverable_c_2x2_world_knowledge_vs_induced_method` | `36` | `51,792` | `1da74922233f761e…` |

## 二、23 个随包附带核心源码、实验脚本与验收测试清单（`bundled_sources/`）

| 仓库相对路径 | 行数 | 字节数 | Git Blob SHA1 前12位 | SHA256 前16位 |
| :--- | ---: | ---: | :--- | :--- |
| `src/hegel_machine/cognition/models.py` | `261` | `9,597` | `9a56752e811c` | `c7933a0f788e028c…` |
| `src/hegel_machine/cognition/unified_research.py` | `7991` | `531,426` | `c66dc6966e31` | `efe7a4f609db9e54…` |
| `src/hegel_machine/cognition/premise_commitment_index.py` | `957` | `39,733` | `efa68c57b8e2` | `a84791e7ad7af6d9…` |
| `src/hegel_machine/cognition/branch_reconciliation.py` | `1604` | `68,047` | `786de7bc14d7` | `0367325a29d91780…` |
| `src/hegel_machine/cognition/discriminating_experiment_policy.py` | `694` | `27,463` | `3b9bcb7ee84a` | `8bee1bf607356627…` |
| `src/hegel_machine/cognition/premise_dependency_audit.py` | `5661` | `250,652` | `3d5d5fe89ff5` | `6ca9535b4c78c199…` |
| `src/hegel_machine/cognition/sensor_role_calibration.py` | `776` | `35,852` | `7a14408e533d` | `3d4a2c0f1b65fa7e…` |
| `src/hegel_machine/cognition/equal_tool_attribution.py` | `1227` | `54,009` | `1db014d235e0` | `46ffcacb217e29ae…` |
| `src/hegel_machine/cognition/world_structure_commitments.py` | `3938` | `163,687` | `9346b3496922` | `03c09399a8303e8a…` |
| `src/hegel_machine/cognition/algebraic_proof_certificates.py` | `2730` | `120,826` | `6b62872e211a` | `0289ebd60fea7b14…` |
| `src/hegel_machine/cognition/sde_conditional_drift_eval.py` | `304` | `13,340` | `c0c9f7c40850` | `345a20396c60d5cd…` |
| `src/hegel_machine/cognition/stochastic_premise_audit.py` | `1163` | `50,419` | `f5ecfdb350da` | `1c74a1d6c77fc4eb…` |
| `scripts/run_v11_deliverable_a_reaudit.py` | `673` | `30,413` | `73674409536b` | `bd14fb2be1cc58ea…` |
| `scripts/run_v11_branch_lifecycle_study.py` | `1181` | `50,921` | `43e5c333ffb4` | `0ee33187d6879db2…` |
| `scripts/run_v11_knowledge_vs_method_2x2.py` | `995` | `41,675` | `59fd8cea145f` | `21bc960adc9d8b93…` |
| `scripts/run_v10_3_library_scaling_study.py` | `758` | `32,009` | `b6ad95814afc` | `b126a9a39e1862c2…` |
| `scripts/run_v10_4_occam_minimality_study.py` | `477` | `20,001` | `32bfcf1dbc83` | `a207390f0d6a4313…` |
| `scripts/run_v10_2_frozen_lifecycle_study.py` | `685` | `31,687` | `b0f1e1e2cca9` | `131ecbceb1084b41…` |
| `tests/test_v11_audit_and_branch_lifecycle.py` | `594` | `25,318` | `e72f5963e0d6` | `1704a33fde4da3d8…` |
| `tests/test_v10_4_occam_minimality.py` | `221` | `8,330` | `ca4574bf18eb` | `6b2e0ff1c3d25239…` |
| `tests/test_v10_3_scaling_and_calibration.py` | `245` | `9,765` | `7aa3a963a014` | `1d521082e62a8156…` |
| `tests/test_v10_2_audit_acceptance.py` | `606` | `28,321` | `4baa6946dc08` | `1927f9da2edc0894…` |
| `markdown/hegel_v11/03_fingerprint_probe.py` | `121` | `5,482` | `4d6723543a11` | `df6c1413b52cae69…` |
