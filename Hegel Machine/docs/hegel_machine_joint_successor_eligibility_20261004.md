# 联合后继候选资格（2026-10-04）

本轮把真实 successor 的联合候选接入未完成的普通数值任务。联合搜索不是额外开一轮实验：它复用同一批已付响应，先完成原有 singleton replay，再检查两个不同来源的单源候选是否共同解释剩余误差。

资格条件是：至少两个不同来源的 singleton 候选已经在当前付费响应上完成测量；最佳 singleton 仍有非零残差；用两个可执行来源共同拟合后，训练损失严格低于最佳 singleton；联合程序没有被当前任务的已有反例撤回。只有同时满足这些条件，候选才进入原有 successor 报价和后续新响应确认。singleton 队列的顺序和预算访问保持不变，联合拟合使用剩余计算预算，失败则不产生候选。

联合候选保留两个来源的 executable call、`joint_source_ids`、lineage 和共享研究程序，因此后续回报可以归因到联合 binding；机制变化时仍由现有 transfer counterexample 逻辑撤回联合程序，不撤回两个 singleton。

验证：`test_joint_successor_requires_residual_and_keeps_two_source_lineage` 通过；普通 successor、来源回传、调度器和互补来源套件共 31 项测试通过。当前 fixture 验证的是联合候选资格和 lineage，完整真实实验序列的长期净收益、多任务独占贡献和同预算普通 Code 对照仍未完成。
