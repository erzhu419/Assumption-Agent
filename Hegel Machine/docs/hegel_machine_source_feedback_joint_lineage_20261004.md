# 联合来源 lineage 与共享成本收据（0.181.0续研）

本轮修复了单个可执行程序由多个来源支持时的归因重复：来源候选在同一程序上合并为一个 joint lineage，各来源仍保留局部记录，新增 `joint_returns` 只记录一次共享适配和比较成本。实验把 `concept_0002` 的真实正来源复制为一个独立来源 `concept_0003`，两者经过真实 successor continuation 都取得 `confirmed` 正回报；坏来源的负回报在掩码前删除，避免撤回事件混入价值估计。

运行命令：

```bash
PYTHONPATH=src:tests python3 scripts/run_source_feedback_joint_lineage.py
```

四个掩码使用同一已付前缀和同一目标响应：保留 A+B 联合候选、删除 A 后执行 B、删除 B 后执行 A、删除 A+B 后保留基线。A+B 报价含两个 `transfer_evidence`，但执行访问仍为 `24` 行观测、`120` 计算；联合记录把共享适配记为 `72`、共享比较记为 `48`，而不是把两份来源局部记录的 `45+45` 和 `48+48` 再当成联合成本。

联合候选和两个单来源候选的未见损失都为约 `2.90e-20`，删除 A+B 后基线未见损失为 `0.9766025984148944`。因此 `v(AB)=v(A)=v(B)=0.9766025984148944`，两来源的边际贡献均为零，交互项 `v(AB)-v(A)-v(B)+v(∅)=-0.9766025984148944`，表示完全冗余而非协同。这个负结果很重要：当前系统已经能识别共享 lineage 和避免成本双计，但还没有证明联合来源能产生互补收益。

本轮没有把来源概念判错；删除掩码只删除条件回报，A、B 及其单来源候选仍可执行。互补续研已经完成：A=current term、B=lag term 的真实联合 call 在同一 lineage 中取得两个正边际贡献，并在机制变化后只撤回联合绑定。结果见[互补联合收据](hegel_machine_source_feedback_complementary_bundle_20261004.md)。

与 `0.181.0` 的关系仍是同一未发布续研，没有改版本号。结果见[联合 lineage 收据](../artifacts/source_feedback_joint_lineage_20261004/)。
