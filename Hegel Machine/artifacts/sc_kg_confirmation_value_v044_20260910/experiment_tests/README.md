本目录保存未采用为默认的 v0.44 短序列估值实验测试，与同实验目录 `../method_source_v044/` 中 9 个冻结模块配套。

复现使用包含既有 scripts、历史 artifacts 与本地 SC-KG 的独立项目副本：将冻结模块放回 `src/hegel_machine/cognition/`，将本目录测试放回该副本的 `tests/`。测试依赖原项目相对布局，不直接在归档目录执行。

原 84 项检查命令：

```bash
PYTHONPATH=src python3 -m pytest -o addopts='' -q \
  tests/test_cognition_sc_kg_environment.py \
  tests/test_cognition_supported_exploration.py \
  tests/test_cognition_terminal_offers.py \
  tests/test_cognition_objective_research.py \
  tests/test_cognition_objective_identification.py \
  tests/test_cognition_research_value.py
```

原始日志完整保留：`research_sequence_integrated_v044_20260910.log` 为 83 passed、1 failed；唯一失败是旧测试未将新增序列成本计入共用预算。更新该计费断言后，`research_sequence_precision_cost_v044_20260910.log` 的定向复跑为 1 passed。因此 84 项检查最终通过，未重新执行整套，也没有单次“84 passed”日志。`research_sequence_adapter_v044_20260910.log` 是新增两项实际集成测试最初通过记录。

`test_cognition_value_precision.py` 一并归档，但不计入上述 84 项。默认源码已恢复此前决策行为，默认 SC 测试恢复原 25 项；实验新增的两个 SC 测试与 `test_cognition_research_value.py` 仅留在本目录。原生 SC-KG 对照测试未修改。上述测试的计算与实际模拟开销单列为验证成本，不计入 355 个正式比较案例。
