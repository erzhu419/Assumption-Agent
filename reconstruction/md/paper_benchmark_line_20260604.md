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

这是预期结果，因为严格论文主张还缺 4 个硬指标：

- `world_model_raw_first_party_scale`: 现在 raw first-party trainable rows 只有 9；1000+ 主要是 distilled transitions。
- `continuous_daemon_autonomy`: 现在是 bounded/gated/readback，不是持续后台无人值守循环。
- `residual_label_large_scale_calibration`: curated label set 只有 10 条，虽 macro-F1=1.0，但不是大规模 adjudicated set。
- `formal_engine_depth`: morphism benchmark 过了，但不是完整范畴论 theorem prover，也不是真 Blackwell/Fisher engine。

## 新百分比

当前 artifact 估算：

- recursive hypothesis argument: 91.1%
- reconstruction.md behavior: 93.3%
- general hypothesis OS: 64.9%

解释：

递归式“提出假设 -> 论证 -> 接受/拒绝 -> 下一代”的工程线已经比较强；但 general hypothesis OS 的分数仍低，因为它要求跨长期真实 trace 的世界模型、持续 daemon、大规模 residual 标注、严格形式推理。这些不是再加一个轻量模块能诚实补掉的。

## 单测

新增：

`test_paper_benchmark_line_separates_working_loop_from_research_gaps`

测试要求：

- `benchmark_line_pass == true`
- `paper_readiness_pass == false`
- failed research gaps 必须包含 raw first-party world model、continuous daemon、大规模 residual labels、formal engine depth

这样后续不会把“组件骨架 pass”误报成“论文主张已经完全完成”。
