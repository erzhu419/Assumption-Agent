# 可调用条件覆盖与重命名复用（20261004）

本轮修复了统一 composition 流程在重命名任务中重复研究条件假设的问题。重命名只改变测量字段绑定；当已有 callable 的先验在同一 command contract 上、经过 fresh trajectory 验证并标记为 supported 时，条件覆盖直接复用。共享验证已经选定的 callable 也会作为当前 transfer 的条件研究对象，竞争的旧 callable 不再触发无效的付费区分实验。若 source、command contract 或 fresh-trajectory 支持条件不匹配，原有 paid condition planner 仍然运行。

同时修复了完整条件覆盖收据的去重路径：program/predicate 是结构化字典，现以 canonical JSON key 去重，避免在完全覆盖时将字典放入 set 导致运行时异常。

## 验证证据

- `PYTHONPATH=src pytest -q tests/test_cognition_unified_composition.py`：`2 passed`。
- 互补作用域回归中的条件收据最小测试也通过：重复写入同一覆盖只保留一条，`KnowledgeLibrary.to_dict()` / `from_dict()` 后内容保持一致。
- 条件研究、程序实验和 object research 聚焦回归：`tests/test_cognition_transfer_condition_plans.py`、`tests/test_cognition_transfer_condition_runtime.py`、`tests/test_cognition_program_experiment_methods.py`、`tests/test_cognition_object_research_stopping.py`、`tests/test_cognition_object_action_selection.py`：全部通过。
- composition 直接 fixture 的五个任务观测行数为 `[192, 183, 40, 40, 24]`；重命名任务由 8 行短试验、8 行验证和 8 行确认组成，不再产生条件实验 acquisition。修复前该任务为 75 行观测。
- 直接 fixture 的新旧损失差均保持在 `1e-7` 以下；`py_compile` 与 `git diff --check` 通过。
- 全部验证均为本地离线执行，没有调用外部模型/API，也没有拉取服务器数据。

## 适用边界

纯重命名的24行观测结果保留。2026-10-04 审计发现的旧覆盖跳过当前反例已修复：复用覆盖前检查当前已付行的实测误差，真实反例优先；对应条件反例和纯重命名回归均通过。新窄链完成机制变化后的反驳与替代学习，见[修复结果](hegel_machine_narrow_cycle_repair_20261004.md)；更广机制与条件覆盖仍待验证。
