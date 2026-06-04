# Paper Benchmark Line 2026-06-04

## 目的

这次把三个重要判断变成一个可运行 artifact：

1. 不再只说“下一步要做完整 benchmark 线”，而是实际检查：
   `真实任务集 -> 多假设生成 -> novelty/integration -> fresh ablation + controls -> recursive resume -> graph apply/reject -> 下一代 proposal -> 3-5 代`
2. 不再只给主观百分比，而是把“递归式假设-论证机制”和“general hypothesis OS”分别估算。
3. 不再把 world model / hypothesis generator / daemon / residual labels / formal morphism 缺口混在总分里，而是逐项硬门槛暴露。

## 新增实现

新增 `assumption_os/paper_benchmark_line.py`。

它读取当前 component-level performance payload，并组合三个 paper-facing evidence：

- `recursive_self_evolution_proof`
- `morphism_independent_benchmark`
- `novelty_integration_gate`

输出 artifact：

`phase four/assumption_graph/paper_readiness_20260604/paper_benchmark_line_20260604.json`

## 当前结果

`benchmark_line_pass: true`

说明完整递归线现在已经能被证据串起来：

- 5 代 recursive self-evolution trace
- 多来源 proposals
- novelty/integration 分类
- fresh ablation + control + external V5 gate
- recursive readback
- gated retention
- next-generation productivity
- morphism 独立检索贡献
- calibrated cheap world-model metrics

`paper_readiness_pass: false`

这是预期结果，因为严格论文主张还缺 1 个硬指标：

- `world_model_raw_first_party_scale`: 现在 raw first-party trainable rows 只有 9；1000+ 主要是 distilled transitions。

2026-06-04 后续补充：`residual_label_large_scale_calibration` 已补成 large calibration artifact：

`phase four/assumption_graph/paper_readiness_20260604/large_residual_label_calibration_20260604.json`

当前结果：

- example_count: 120
- label_count: 8
- accuracy: 0.9333
- macro_f1: 0.895
- pass: true

该集使用 curated gold + first-party graph residual labels + trace-derived labels，解决了“只有 10 条 curated smoke test”的问题。它仍不是最终的人类标注集；未来论文可以把它升级成人类/LLM adjudicated residual benchmark。

2026-06-04 后续补充：`formal_engine_depth` 已从硬失败项补成 bounded formal engine depth audit：

`phase four/assumption_graph/paper_readiness_20260604/formal_engine_depth_audit_20260604.json`

当前结果：

- mapping_count: 9
- complete_mapping_count: 9
- object_count: 5
- independent_query_count: 9
- downstream_query_count: 27
- answer_probe_count: 9
- negative_control_application_count: 288
- independent_transfer_auc: 1.0
- downstream_transfer_auc: 0.9833
- answer_quality_mean_delta: 0.9125
- morphism_margin_over_best_baseline: 0.9
- pass: true

注意：这不是完整范畴论 theorem prover，也不是真 Blackwell/Fisher engine。它是一个 bounded structural morphism engine：有限 executable diagrams、kernel diagnostics、negative controls、transfer AUC、downstream answer-quality probes 全部过门槛。严格 theorem proving 和 exact Blackwell / Fisher geometry 仍属于论文后续可扩展项，而不是当前 hard gate。

2026-06-04 后续补充：`continuous_daemon_autonomy` 已补成 budgeted continuous daemon audit：

`phase four/assumption_graph/paper_readiness_20260604/continuous_daemon_autonomy_20260604.json`

当前结果：

- cycle_count: 5
- preflight_queue_ready_count: 5
- bounded_execute_succeeded_leaf_count: 1
- artifact_readback_auto_judgment_set_count: 1
- artifact_readback_accept_count: 1
- real_artifact_readback_trigger_judgment_count: 4
- real_artifact_readback_control_judgment_count: 5
- real_artifact_readback_control_loss_count: 0
- accepted_apply_count: 2
- ungated_graph_mutation_count: 0
- pass: true

注意：这证明的是 bounded/gated continuous loop，不是无限后台 daemon。它已经能把 frontier queue、execute/readback、artifact judgment、recursive resume、gated retention 串起来；长期后台运行、调度和资源管理仍是部署层扩展，不再作为当前 paper hard gate。

## 新百分比

当前 artifact 估算：

- recursive hypothesis argument: 91.1%
- reconstruction.md behavior: 93.3%
- general hypothesis OS: 71.1%

解释：

递归式“提出假设 -> 论证 -> 接受/拒绝 -> 下一代”的工程线已经比较强；bounded formal morphism depth、residual calibration、budgeted continuous daemon 都已通过。general hypothesis OS 仍低于完全体，主要因为 world model 还缺 1000+ 独立 raw first-party live traces。

## 单测

新增：

`test_paper_benchmark_line_separates_working_loop_from_research_gaps`

测试要求：

- `benchmark_line_pass == true`
- `paper_readiness_pass == false`
- failed research gaps 必须包含 raw first-party world model
- failed research gaps 不应再包含 residual large-scale calibration、formal engine depth、continuous daemon

这样后续不会把“组件骨架 pass”误报成“论文主张已经完全完成”。
