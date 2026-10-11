# 冻结策略的真实 Luna 采用检查（2026-10-10）

**Luna 首次调用即自主采用现成控制器返回的4734 Hz实验，随后实现目标激活并自主分析为点预测成立。旧夹具错误保留了false完成标志；修复计分同步后，相同动作重放成功。** 两段运行均已结束，未继续追跑。

从原576观测／160模型调用现场恢复全部研究与费用，保留当前16项声明及两份作者表示源码。科学策略与上一轮修复一致；模型为订阅`gpt-6-luna / medium`。实验驱动未指定频率、公式或实验动作，首次选择见[真实模型输出](agent/code_action_001.json)。

| 阶段 | 新模型调用 | 新物理动作 | 结果 |
|---|---:|---:|---|
| 采用检查；模型总额度160→164，其他上限不变 | 4 | 10 | 首次选中实验；控制器9步后触及累计1200秒Python上限；模型直接操作到4731，随后两次分析被额度阻断 |
| 显式续跑；Python总额度1200→1600，模型164→172 | 4 | 3 | 分析中间结果，自主操作至4734并确认激活，随后主动finish；剩4次调用未用 |

合计8调用、13动作、646678研究计算、60.286秒Python、2481502报告token；本批缓存命中token为0。世界动作与任务墙钟上限均未增加，前次费用完整保留。三份科学源码均未变。[汇总证据](adoption_summary.json)、[续跑结果](../response_incumbent_adoption_luna_20261010_completion/brief.json)。

最终观测588显示全部五个反应器激活，模型的`reference_response_5_30`分析为supported。旧`ReferenceLab`只在Exit时刷新完成标志；[官方任务评分](</home/erzhu419/.cache/hegel-machine/discoveryworld/discoveryworld/TaskScorer.py:2803>)每世界tick按激活状态更新，不要求退出菜单。这是夹具同步缺陷，不是模型的科学或收尾失败。

整批冻结实验结束后，仅在`TransferSignalLab`复用既有`ImmediateCompletionLab`的逐动作同步方式，不改变激活谓词、阈值或动作。589条原动作重放的物理观察与反馈全部相同，仅最后观测588的完成标志变为true，仍在菜单内，没有补Exit。[同步修复重放](../response_incumbent_adoption_luna_20261010_completion/score_sync_replay.json)独立保存，原始false结果不覆盖。

恢复时发现此前实际导入生成的`__pycache__`被当文本读取，导致首次准备失败，未调用模型。复用既有文件过滤方式排除Python缓存，恢复回归通过；失败保留在相邻`response_incumbent_adoption_luna_20261010/setup_failure.json`。

## Limitations与下一步

这是一个已消费简化任务的自主采用与点预测验证，不是官方DiscoveryWorld成绩、新结构发现、独立泛化或递归优势。第二段有显式额度增加；修复重放也不是一次新模型运行，不能据此做同预算性能比较。计分同步会改变未来运行的终止时刻。原恢复协议的顶层旧study/prefix字段属于继承记录，实际阶段起点与额度见各自`resume`及extension字段。

独立参数条件另见[冻结配对面板](../response_incumbent_frozen_transfer_20261010/RUN_STATUS.md)：两臂均2/6，新修订未触发，无独立收益。后续优先复用旧响应研究接续条件，修复先前目标以操作搜索成功后、下一目标又退回逐项点预测的分配问题。
