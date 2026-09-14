# 0.24.0 共享前置研究

2026-09-09。[十项目标完成度](hegel_machine_advice_goal_status_20260909.md)是当前总进度入口。

`ResearchAgenda` 复用 `ResearchGraph`：后继问题通过 `depends_on` 声明前置答案，多个后继共同形成研究需求；共享已完成成果，避免各后继重复研究和付费，知识由实际后继任务检验并调用。未解决前置会阻塞后继；新证据、知识撤销或修改依赖会重开受影响的问题。仪器研究重开会继续原工作区并计入新增费用。

公开 `cognition agenda` 请求可以先登记 `questions`，再用 `dependencies: [{"question_id": "consumer", "depends_on": ["source"]}]` 连接或修改研究顺序。继续使用同一 checkpoint 保存观测、知识、图和费用。

[实际示例](../artifacts/hegel_machine_v024_shared_prerequisite_20260909/summary.json)：前置概念被两个后继实际 `abstract_reuse`，经历 JSON 保存恢复。共享研究用 65 行观测、5920 计算；各自重复前置用 96 行、10256 计算，共省 31 行、4336 计算。两个后继独立评测误差均小于 1e-18。

运行示例：`PYTHONPATH=src python3 scripts/run_shared_research_agenda.py --output artifacts/shared_research_demo`。

[559 项认知测试通过](../artifacts/cognition_v024_shared_research_20260909.log)，含实际共享调用、依赖修改、未决前置阻塞、付费新证据重开、仪器继续及公开 CLI 保存恢复。

本批扩展原清单第 1、2、9 项：跨任务共享子问题、按后继需求安排投入、依赖变化后的持续研究。校准的下一步是让行动分界驱动条件实验，随后重新拟合与代表轨迹比较，作为可执行研究程序接入这条链。

## Limitations

共享依赖由调用方给出，需求数量仅决定未知收益机会的探索次序；尚未自动发现多个问题应共享什么知识，也不代表已预测其未来净收益。当前长期效果仍以 0.23.1 的已完成配对为准，结果并不稳定。后续建设优先补方法与代理评价器自身修订、未覆盖的结构族和长期普通代码 Agent 对照。
