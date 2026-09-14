# 先验采用条件与真实优化

2026-09-10，0.31.0。后续共同失配修复见 [0.32 研究结果](hegel_machine_joint_model_revision_20260910.md)。

本批继续解决“已学先验改变了优化行为，却没有独立收益”。[原三臂对照](hegel_machine_meta_prior_research_20260910.md)的数据回传修复已经有效；进一步[只读诊断](../artifacts/sc_kg_prior_adoption_diagnostic_v030_20260910/diagnostic.json)发现，原 RZDT2 的默认二次表示本来就包含学得规律。两种模型均值最大差不到 4e-9，预测方差却明显不同；SC 的经验初始化随参数化改变了协方差。这不能单凭维数降低认定先验有益。

## 本批实现

[PriorAdoption](../src/hegel_machine/cognition/prior_adoption.py)从默认模型开始，用两种模型对后续真实观测的事前预测比较误差。至少四个新预测后，配对改善超过误差容差及采样不确定性才采用；反向证据可撤回，未决证据继续积累。两者均满足既定精度时保留默认模型。产生候选的研究数据用于拟合，不充当采用依据。

[SC-KG 接口](../src/hegel_machine/cognition/sc_kg_environment.py)维护两套真实约束 GP/HVD，使用完全相同的已付数据和新实测更新；切换发生在当前更新完成后，下次采集与最终建议随之改变。额外模型拟合、预测、更新和运行时间单独记录，比较本身不增加模拟调用。知识库保存部署时的采用与撤回证据。

同时增加 T11 的公开坐标差折点候选 `abs(x_i-x_j)`：原平滑模型仍有残差时才拟合其贡献。新增真实 `StatePolicyRZDT1(d=3)` 装置，模型从实测发现非二次约束，问题类别不进入学习侧。它用于检验默认表示确实失配时的新先验价值。

## 对照结果

[固定协议](../artifacts/sc_kg_prior_adoption_v031_20260910/protocol.json)包含 15 个病例、自动采用／始终学得／始终默认三臂。RZDT2 保留原三种子，再加入三个预先固定的新种子，分别测试噪声 0 和 .03；三个新种子的折点约束单列。每臂总预算 64：32 次研究＋32 次优化，完整已付数据全部继承。

[45 次完整优化](../artifacts/sc_kg_prior_adoption_v031_20260910/summary.json)全部完成且推荐可行。平均目标越低越好：

| 三种子一组 | 自动采用 | 始终学得 | 始终默认 |
|---|---:|---:|---:|
| 原 RZDT2，无噪声 | 1.281809 | 1.286606 | 1.281809 |
| 原 RZDT2，噪声 .03 | 1.286606 | 1.286606 | 1.324615 |
| 新种子 RZDT2，无噪声 | 1.320459 | 1.320459 | 1.320459 |
| 新种子 RZDT2，噪声 .03 | 1.493466 | 1.560967 | 1.493466 |
| 新种子折点约束 | .378840 | .378840 | .378840 |

自动策略实际采用 5 次：2 个带噪 RZDT2 和 3 个折点任务。相对始终默认，15 个病例中 1 个改善、14 个持平；相对始终学得，2 个改善、13 个持平。新种子的带噪退步被避免，原带噪病例中一次采用带来改善。静态研究未发生撤回。

三个折点任务均通过实测发现 T11 表示。后续真实采集点上，默认模型预测 MSE 为 4.66e-6～1.01e-5，学得模型为 6.29e-18～7.07e-16；最终优化目标却相同，预测收益尚未转为优化收益。

实际新增研究调用 384 次、优化调用 1440 次，另外复用历史研究 96 次。自动比较新增模拟为零；15 个自动臂额外拟合 15 个模型、完成 480 次影子 GP 更新及 960 次 HVD 更新，记录的额外处理时间合计约 .92 秒。

## 运行

研究检查点可以接着优化，每完成一个优化任务保存结果，下次继续剩余任务：

```bash
PYTHONPATH=src python3 -m hegel_machine cognition prior --checkpoint artifacts/sc_prior_session.json --request config/cognition_sc_kg_prior.json
PYTHONPATH=src python3 -m hegel_machine cognition prior --checkpoint artifacts/sc_prior_session.json --optimize --optimizer-config config/cognition_sc_kg_optimizer.json --optimizer-steps 1
PYTHONPATH=src python3 -m hegel_machine cognition prior --checkpoint artifacts/sc_prior_session.json --optimize
```

检查点保存采用策略与逐次预测；知识库保存采用／撤回证据及费用。Python 入口为 `initialize_sc_kg_optimizer(problem, config, adoption_policy=PriorAdoption())`。

[完整认知回归](../artifacts/cognition_v031_release_20260910.log)657 项通过，后续[优化任务暂停与续跑测试](../artifacts/prior_adoption_resume_v031_20260910.log)另行验证完成任务不重复执行。

[实际命令续跑](../artifacts/sc_kg_prior_adoption_session_v031_20260910_summary.json)在原四题研究数据上完成四次优化，总计账 256 次，其中新优化调用 176 次。两次按单任务暂停后继续，完成任务保持原结果；采用证据已写回共享知识库。

## Limitations 与下一步

[独立机制变化实验](../artifacts/sc_kg_prior_mechanism_change_v031_20260910/summary.json)使用预定新种子：先学得折点表示，再实际执行 12 次 StatePolicy 和 20 次 RZDT2 响应。真正采用策略在第 9 次新观测后采用，变化后却没有撤回；此时默认 MSE .599108、学得 MSE .164960，两个解释均已失配。相对优势不足以判定一个模型值得继续使用。下一步优先补上共同失配时重开假设研究的条件。脚本化切换测试只证明接口切换，不能替代这个失败案例。

采用目前依据约束预测改善，尚未把不确定性压缩本身的决策价值学成可检验条件；比较两个模型需要额外计算。六个带噪研究候选仍处于未决状态，其中两个由后续预测证据获得采用支持。折点搜索仍局限于公开坐标差，22 项先验没有全部通用化。跨任务采用经验已保存，尚未转为泛化条件。优化任务之间可以继续；单个 SC 优化运行中途的双模型检查点恢复尚未接通。长期总收益仍需扩大研究。
