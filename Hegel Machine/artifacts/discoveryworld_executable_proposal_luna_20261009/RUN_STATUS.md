# 解释提案接入已有数值研究器（2026-10-09）

复用旧Program提案与多观测研究器，补`bind → comparison_action/research_update、research_action/research`出口；保留原`contrast_action`。父研究先比较已付观测，再自行选择实验。相关28项测试通过，无测试模型调用。

真实试跑沿用699动作、33调用的同一现场，`gpt-6-luna / medium`、累计65调用上限。提案阶段主动用6/8调用结束，返回C4旧扫描分析和C5探索扫描，未生成`bind`。父研究随后采用`contrast`，自行修复Python类型错误，并将C4判定改为C5新观测分析。现已以`model_call_budget_exhausted`结束：新增32调用、95动作，总794动作，仍15/29、未完成任务；父研究程序未改。模型报告12,444,102 tokens，其中缓存输入615,168。见[终值](brief.json)与[协议](protocol.json)；终值的`episode_status=running`表示环境任务未结束，`study_status=finished`表示本轮试跑已停止。

科学内容均由模型生成，运行中未添加科学提示或修改其程序。复用登记已同步；下一算法问题见[网页端方案问题](../../markdown/Hegel_Machine_Web_GPT6_Design_Questions_20261009.md)。

## Limitations

28项测试证明当时接口接通；真实新数值出口未被采用。提案阶段由宿主发起，不能单独归因于出口改动。当时数值研究仍冻结同题候选，新族需独立问题，复用旧物性的新响应仍受旧确认规则限制。后续修复见[解释族小实验](../response_family_pilot_20261009/RUN_STATUS.md)，本试跑未使用后续改动。
