# 来源清单、22 份证据回执分级索引与核查范围

本清单将 `v10`（13 份原始回执，100% 字节未改动）、`v10.2`（6 份审计修复与冻结生命周期回执）、`v10.3`（2 份知识库规模化与高元数传感器标定回执）以及 `v10.4`（1 份奥卡姆极小性消融与 `M=72 x N=8` 全栈回执）共 **22 份 JSON 回执**及 **22 个核心源码／脚本／测试文件**全部完整打包入本 ZIP（见 `receipts/` 与 `bundled_sources/`），供审阅端在本地沙箱直接全量读取与校验，消除超长文件截断盲区。

## 一、22 份 JSON 证据回执全量索引（含实验角色与独立物理源计数 `F16`）

| 阶段 | 回执文件（ZIP 内路径 `receipts/<阶段>/<文件名>`） | 实验角色分类 (`study_role`) | 独立物理源数 | 字节数 | SHA256 前16位 |
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
| `v10_3_scaling_and_calibration_20261117` | `library_scaling_curve_20261117.json` | `single_factor_library_scaling_curve` | `26` | `132,233` | `ceff49583506b5d7…` |
| `v10_4_occam_and_gpt6_bundle_20261117` | `occam_minimality_ablation_20261117.json` | `single_factor_occam_minimality_and_m72_x_n8_fullstack` | `9` | `175,738` | `9370ca4482318e59…` |

## 二、22 个随包附带核心源码、实验脚本与验收测试清单（`bundled_sources/`）

| 仓库相对路径（ZIP 内路径 `bundled_sources/<路径>`） | 行数 | 字节数 | Git Blob SHA1 前12位 | SHA256 前16位 |
| :--- | ---: | ---: | :--- | :--- |
| `src/hegel_machine/cognition/models.py` | `260` | `9,557` | `7ea84ea2cfa1` | `341e8a5318fea0f3…` |
| `src/hegel_machine/cognition/algebraic_proof_certificates.py` | `2730` | `120,826` | `6b62872e211a` | `0289ebd60fea7b14…` |
| `src/hegel_machine/cognition/sde_conditional_drift_eval.py` | `304` | `13,340` | `c0c9f7c40850` | `345a20396c60d5cd…` |
| `src/hegel_machine/cognition/stochastic_premise_audit.py` | `1163` | `50,419` | `f5ecfdb350da` | `1c74a1d6c77fc4eb…` |
| `src/hegel_machine/cognition/equal_tool_attribution.py` | `1011` | `45,847` | `4a7b86134ca3` | `32118d04cec7ee9f…` |
| `src/hegel_machine/cognition/premise_commitment_index.py` | `451` | `19,791` | `358f462b7455` | `c87447d011e65db6…` |
| `src/hegel_machine/cognition/sensor_role_calibration.py` | `743` | `34,245` | `08de391396c2` | `55a4baca902a1536…` |
| `src/hegel_machine/cognition/premise_dependency_audit.py` | `5581` | `246,633` | `905c3f6e5e2a` | `736695b88e49acb9…` |
| `src/hegel_machine/cognition/world_structure_commitments.py` | `3938` | `163,687` | `9346b3496922` | `03c09399a8303e8a…` |
| `src/hegel_machine/cognition/relation_uncertainty.py` | `983` | `58,250` | `c0735089d4fe` | `5f57d167d3a22505…` |
| `src/hegel_machine/cognition/competing_object_experiments.py` | `455` | `26,385` | `c5b5a0d6b68a` | `3943b99a7e8e3fe0…` |
| `src/hegel_machine/cognition/condition_questions.py` | `300` | `16,574` | `d825991d77d1` | `d0d0176beb571cee…` |
| `scripts/run_v10_2_sde_and_proof_remediation.py` | `556` | `24,144` | `36cf4d497266` | `8619eb3f53183249…` |
| `scripts/run_v10_2_equal_tool_attribution.py` | `1079` | `44,791` | `03886354f901` | `9d45ee25d46087eb…` |
| `scripts/run_v10_2_frozen_lifecycle_study.py` | `655` | `30,135` | `6a9de6691984` | `2ec31b80f1997ae9…` |
| `scripts/build_v10_2_errata_and_sha256_ledger.py` | `409` | `27,538` | `65387da13bd7` | `bf66619f1b8052d6…` |
| `scripts/run_v10_3_library_scaling_study.py` | `740` | `31,125` | `90d232e28a88` | `ff4fc36843fe9521…` |
| `scripts/run_v10_3_high_arity_calibration_study.py` | `672` | `29,437` | `97b9e479ffb9` | `f5a1469a374e42e9…` |
| `scripts/run_v10_4_occam_minimality_study.py` | `475` | `19,776` | `7745e5081646` | `0fe6f1ed01505325…` |
| `tests/test_v10_2_audit_acceptance.py` | `606` | `28,321` | `4baa6946dc08` | `1927f9da2edc0894…` |
| `tests/test_v10_3_scaling_and_calibration.py` | `245` | `9,765` | `7aa3a963a014` | `1d521082e62a8156…` |
| `tests/test_v10_4_occam_minimality.py` | `221` | `8,330` | `ca4574bf18eb` | `6b2e0ff1c3d25239…` |

## 三、核查边界声明（遵循 `02_下一施工合同.md` #16–#22）

- **零在线付费 API 调用：** `v10.2`、`v10.3`、`v10.4` 的所有代码修复、同工具对照、前瞻生命周期、规模曲线与高元数实验均在 `model_calls = 0, total_tokens = 0` 下实测产生。
- **历史回执零覆盖：** `v10_world_structure_20261116/` 下全部 13 份回执与 `v10_2`（6份）、`v10.3`（2份）回执均保持原始 SHA256 不变（可由 `05_STANDALONE_VERIFY_V10_4.py` 一键核验）。
- **测试边界诚实披露：** 专项验收套件（`test_v10_*.py` + `test_v10_2_audit_acceptance.py` + `test_v10_3_scaling_and_calibration.py` + `test_v10_4_occam_minimality.py` + 历史能力与条件区间套件）与仓库历史参数化测试集（收集规模 `17,520` 项）严格分开报告，绝不将专项测试通过混称为普适科学定律。
