# 0.25：把研究收益评价器变成可修订程序

2026-09-09。本批推进原清单第 5 项：`ResearchEvaluator` 从已完成续研的真实收益合成可执行函数，以后续预先给出的预测和实际反馈比较新旧评价器，支持采用、改写与撤销。它已接入 `ResearchSession` 和 `ResearchAgenda`，参与实际研究选择；模型与待反馈预测随工作区保存。推断、比较和拟合消耗总预算，并为下一次完整研究留出费用。

[构造功能例](../artifacts/research_evaluator_demo_20260909.json)执行 11 次真实续研：前六次反馈提出函数，随后四次支持采用。面对两个待研究问题，旧均值都预测收益 0.33，新函数预测 0.125／0.605，改变了选择；被选问题的实际归一化收益为 0.605。准备、续研和评价器合计使用 208 行观测、3516 计算。复现入口：`PYTHONPATH=src python3 scripts/run_research_evaluator_demo.py --output artifacts/research_evaluator_demo_new.json`。

[两种子对照](../artifacts/research_evaluator_v024_study_note_20260909.md)则未产生有效新评价器：10 次后续选择全部相同，最终损失相同，多用 44595 计算。该批在版本提升前运行，原 `v024` 产物保留；后来修复的调度回退分支未在其中触发。修复后完成一次实际续研，并以针对性用例覆盖该分支。

[574 项认知测试通过](../artifacts/cognition_v025_research_evaluator_20260909.log)，覆盖函数表示改变、后续比较、撤销、恢复、实际反馈与调度。核心实现：[评价器](../src/hegel_machine/cognition/research_evaluator.py)、[研究机会](../src/hegel_machine/cognition/research_opportunities.py)、[会话](../src/hegel_machine/cognition/session.py)、[议程](../src/hegel_machine/cognition/agenda.py)。

## Limitations 与下一步

函数与输入特征仍受现有语言限制；构造例说明真实反馈能够改变选择，两种子对照尚无净收益。评价器修订预测研究收益的函数，任务目标和最终评价准则保持固定。实验模拟模型自身的持续修订仍未接通，因此[十项目标](hegel_machine_advice_goal_status_20260909.md)仍为 7 项有界闭环、3 项部分完成。

下一批补第 5 项的实验模拟模型：把模拟与实测的差异转成研究问题，修订模型后改变实验选择；同时推进第 7 项缺失的结构发现。第 10 项的跨任务普通 Python Agent 对照与稳定长期净收益继续保留为未完成目标。
