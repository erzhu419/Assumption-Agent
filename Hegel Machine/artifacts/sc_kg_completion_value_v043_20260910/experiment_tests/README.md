这些测试对应同一实验目录 `../method_source_v043/` 中的 8 个冻结运行模块，验证的是未采用为默认的 v0.43 完成计划实验。`passed_78_tests.log` 保留该实验源码下的完整通过记录。

复现时使用包含既有 scripts、历史 artifacts 和本地 SC-KG 的独立项目副本：把 `method_source_v043/` 内模块放回 `src/hegel_machine/cognition/` 的同名位置，把本目录测试放回该副本的 `tests/`。测试中的相对路径以原项目布局为准，不能直接从归档目录运行。

当时 78 项通过的命令：

```bash
PYTHONPATH=src python3 -m pytest -o addopts='' -q \
  tests/test_cognition_sc_kg_environment.py \
  tests/test_cognition_terminal_completion.py \
  tests/test_cognition_supported_exploration.py \
  tests/test_cognition_terminal_offers.py \
  tests/test_cognition_objective_research.py \
  tests/test_cognition_objective_identification.py
```

`test_cognition_value_precision.py` 一并保存；它不计入上述 78 项。当前默认代码已恢复 v0.42 决策行为，默认 tests 不再包含完成计划实验契约；真实原生 SC-KG 对照的测试独立保留。
