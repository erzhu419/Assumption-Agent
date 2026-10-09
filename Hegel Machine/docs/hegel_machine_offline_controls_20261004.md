# 当前离线控制臂

2026-10-04 修复后：原seed12901、1→2→3的12k／24k两档四臂均3/3；full计算分别19,443／35,341，固定库均1,296。下文1/3和重复预留属于修复前记录；最新空库学习→机制变化→再利用窄链另有12/12冻结确认与同前缀来源删除对照，见[修复结果](hegel_machine_narrow_cycle_repair_20261004.md)。

本轮补上了当前认知主线的四臂离线配对入口：完整研究、固定库、固定实验设计和无跨任务复用。入口是 [`scripts/run_current_offline_ablation.py`](../scripts/run_current_offline_ablation.py)，每臂使用相同任务序列、每任务观测/测量/候选行上限；每臂从独立空 `KnowledgeLibrary` 开始，所有学习结束后才建立独立终测轨迹。

四臂的含义保持清楚：`full` 使用当前 UnifiedResearchSession；`fixed_library` 使用已有 fixed 策略且不学习库；`passive_experiments` 预先提交四条轨迹和全部仪器，使用共享表示研究但不做自适应回访；`no_reuse` 保留当前主动研究、关闭跨任务记忆。它们是算法控制臂，不是普通 Code Agent 或同一 LLM 的证据。

已跑通一批单任务离线 smoke cohort（task 0，训练预算 96 行观测、384 测量、12000 候选行，独立评分 16 行）：四臂均完成并得到可执行模型，完整臂和无复用臂各花费 48 行观测、0 测量、185 候选行；固定库臂花费 48 行观测、192 测量、432 候选行；固定实验臂花费 96 行观测、384 测量、288 候选行。该批只是接口和预算合同的运行证据，不能据此声称完整系统优于任一控制。

默认 runner 现在使用 source-first 的 `1→2→3` 序列，确保后继任务确实能看到前序概念；原来的 `2→3→4` 仍保留为无复用压力的协议负例。source-first 缩小 cohort 使用每任务 96/384/12000、独立评分 16 行：fixed_library 达标 3/3，full 达标 1/3，passive_experiments 达标 3/3，no_reuse 达标 1/3。完整臂在第 3 题能看到 `concept_0001`，但没有接受 successor；固定实验臂在第 3 题实际使用 `concept_0002`，说明“来源可见”与“投入足以完成确认”仍是两个问题。原 `2→3→4` cohort 中 full 仅 1/3，且来源只在最后题产生，不能作为跨任务复用收益证据。两批都只能定位研究投入瓶颈，不能声称完整系统优于固定基底。

2026-10-04 审计按保存的 seed12901 协议复跑：每任务计算上限从 12000 提到 24000 后，full 在第 3 题确认并使用 `concept_0002`，四臂均为 3/3 达标；当前 full 计算为 48480，fixed_library 为 1296，passive_experiments 为 48616，no_reuse 为 56756。最窄学习→复用途径已跑通，当前策略在12000下的研究投入仍有问题；固定库在两档预算都能低成本完成。此前 full 48551 和48000单臂未决属于此前代码状态的开发记录，本次未复跑48000。当前 receipt 的 `refuted_transfers` 统计 accepted=False，包含未接受情形，不能一律解释为取得反例后的合理拒绝。

在 seed12901、12000 的当前 full 运行中，第 2 题完成 `validate_frozen_program` 但未被接受，仍未决。内层确认预留恢复了最后比较的可达性，但外层 session 又将初访总上限扣去48，这部分不会在本次确认门释放；两层存在重复预留。仅移除外层后，mean MSE 从 .427841 降至 .388016，达标仍为1/3。该实现需要局部修复，详见[审计](hegel_machine_post_0181_audit_20261004.md)。

这项预留只作用于 `use_memory=True` 且不是 fixed 策略的统一研究臂；固定库、无复用和被动实验控制仍按原预算语义运行。seed12901 当前 full 成本为24516；此前24510来自不同的 seed73，两批必须分开。当前重跑汇总见[审计收据](../artifacts/post_0181_audit_20261004/results.json)，原保存结果保留不覆盖。

当前仍缺普通 Code Agent 和同一 LLM 模块消融。现有 `CodeComparisonSession` 与 `UnifiedCodeSession` 能承载这两臂，但需要单独冻结模型、token/调用预算和不读 Hegel 结果的 provider 后再运行，不能用本地确定性 provider 冒充普通 Code 结果。
