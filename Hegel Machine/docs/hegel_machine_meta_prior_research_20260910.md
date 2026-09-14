# 元先验自主研究与 SC-KG 接通

2026-09-10，0.30.0。

本次检查确认：SC-KG 已有固定先验的参数学习和消费，但原真实 RZDT 流程停在“需要新实验”，没有继续形成新解释；另一个到达 G11 的运行是预设转换的正对照，`hegel_invoked=false`。旧流程分别见 [RZDT 状态](/home/erzhu419/.local/state/kg-op/meta-prior-local-rzdt-v11-runs/q05b-attempt2-v1/driver_state.json)与[正对照状态](/home/erzhu419/.local/state/kg-op/meta-prior-staged-positive-control-v11-runs/q05b-attempt2-v2/driver_state.json)。因此此前“元先验已能自主提出并递归更新”的说法不能覆盖这条实际联动。

本批把这条循环接入认知主线：公开控制与实测 → 稀疏、低维、低阶交互和有限历史假设 → 根据候选分歧及投影丢失方向设计实验 → 保存反例 → 学习残差子答案并组成新程序 → 条件修订旧先验 → 后继任务实际调用。旧成功程序保留；原始坐标中的简单模型已经解释现有证据时，先验证它，再决定是否追加递归研究。

实现位于 [meta_prior_claims](../src/hegel_machine/cognition/meta_prior_claims.py)、[meta_prior_research](../src/hegel_machine/cognition/meta_prior_research.py)、[meta_prior_session](../src/hegel_machine/cognition/meta_prior_session.py) 和 [SC-KG 接口](../src/hegel_machine/cognition/sc_kg_environment.py)。`KnowledgeLibrary.meta_priors` 保存支持、反例和来源，实际程序仍通过共同概念库进入后续搜索。SC-KG 接口调用真实 `simulate`，返回的程序可作为其 `ParametricGPR` 的基函数。

## 实际结果

[测量修订研究](../artifacts/measurement_revision_v030_20260910/summary.json)完成三组各 24 题、共享／空库共 144 项。原失败题在同一通道上重选表示，独立 MSE 从 0.02446 降至 1.97e-13。空库 72 题平均 MSE 从 3.445e-4 降至 4.761e-6，少用 2 行观测、66 测量，增加 13323 搜索计算；共享组误差、观测与测量保持，计算增加 0.27%。

[通用研究](../artifacts/meta_prior_research_efficient_v030_20260910.json)包含两类变化、三种子、共享／空库两臂，共 48 题，全部学习结束后评分。

| 每臂 24 题 | 共享知识 | 每题空库 |
|---|---:|---:|
| 完成 | 24/24 | 18/24 |
| 实测行数 | 672 | 960 |
| 搜索计算 | 19620 | 47028 |
| 平均独立 MSE | 1.25e-16 | 0.7734 |

三个嵌套读出反例任务的最终程序实际执行递归子答案，并调用旧机制。简化研究选择后，共享计算相对[本批初版](../artifacts/meta_prior_research_v030_20260910.json)减少 16.15%，所有最终程序与误差保持一致；额外构造的残差图从 12 个减到 3 个。

[真实 SC-KG 研究](../artifacts/sc_kg_meta_prior_efficient_v030_20260910/summary.json)为三组四题、共享／空库两臂，24/24 完成。共享 12 题用了 240 次模拟、4491 搜索计算；空库为 312、15042，分别节省 **23.1% 实测与 70.1% 计算**，独立平均 MSE 分别约 1.51e-20、9.11e-18。九个共享后继任务全部实际调用已学知识。

冻结程序交给真实 SC-KG GP 后，共用 8 个新训练点和 64 个新评价点，共享臂平均 MSE 为 5.92e-21，SC 默认二次 basis 为 0.00931。默认 basis 有 11 个参数方向，学得程序压缩为 2 维；这里测量的是少样本预测收益。

[完整优化器初次对照](../artifacts/sc_kg_optimizer_smoke_v030_20260910/summary.json)运行原 `SingleOLHKGAlgorithm.run()`，学得程序实际改变采集点。总预算每臂 64 次，学得臂包括 32 次研究调用；两臂都三题可行，但学得臂仅一胜两负，平均目标 1.6172，默认 1.4431（越低越好）。据此补上数据继承：同次模拟的完整目标与约束输出、重复采样均保存，直接装入 SC 的真实后验，继续执行原采集算法。

[数据继承后的三臂比较](../artifacts/sc_kg_optimizer_data_reuse_v030_20260910/summary.json)重新取得完整实测数据，每组每臂总预算仍为 64。两个继承臂都将全部 32 行研究观测装入目标与约束后验，再取得 32 行新观测；默认从头开始取得 64 行。三种子结果如下，目标越低越好：

| 优化方式 | 平均目标 | 最终可行 |
|---|---:|---:|
| 学得先验＋已付数据 | 1.286606 | 3/3 |
| 默认先验＋相同数据 | 1.281809 | 3/3 |
| 默认先验从头开始 | 1.443123 | 3/3 |

数据继承使学得臂在三组中均优于原“只回传程序”版本。相对默认从头开始为两胜一负；相同数据下，学得先验为零胜一负两平。**数据回传修复有效，先验本身的独立优化增益尚未体现。** 两个继承臂分别如实计入研究的 32 次调用；物理执行共新增 480 次模拟，共享数据没有重复获取。

## 直接运行

```bash
PYTHONPATH=src python3 -m hegel_machine cognition prior --checkpoint artifacts/prior_session.json --research-steps 2
PYTHONPATH=src python3 -m hegel_machine cognition prior --checkpoint artifacts/prior_session.json --evaluate
PYTHONPATH=src python3 -m hegel_machine cognition prior --checkpoint artifacts/sc_prior_session.json --request config/cognition_sc_kg_prior.json --evaluate
```

前两条在通用装置上保存并继续研究；第三条使用实际本地 SC-KG 装置，[配置](../config/cognition_sc_kg_prior.json)可修改项目路径和任务顺序。`--memory PATH` 导入／导出公共概念库。`frontier` 新增元先验阶段；已有会话可以续接，保存的实验与已完成任务继续使用。

同一个 SC-KG `environment` 恢复研究结果后，可以直接把程序和已付观测交回优化器。以下函数来自 `cognition.sc_kg_environment`，`config` 使用 SC-KG 原 `SingleOLHKGConfig`，其中 `N` 包含此前研究已付调用：

```python
environment.restore(result["research_state"]["environment_state"])
problem = compile_sc_kg_program(environment, Program.from_dict(result["program"]))
optimizer = initialize_sc_kg_optimizer(problem, config)
optimization = optimizer.run()
```

[SC-KG 命令续跑](../artifacts/sc_kg_prior_data_session_v030_20260910_summary.json)完成四题，80 次实测、1497 搜索计算，80 行完整模拟输出均随检查点保留；[整机续接](../artifacts/frontier_meta_prior_v030_20260910_summary.json)从原 24 阶段会话运行新增阶段，25 阶段完成、零待办。旧阶段保留原版本结果。[认知回归](../artifacts/cognition_v030_release_20260910.log)645 项通过，数据继承改动后的 [10 项接口与续跑测试](../artifacts/cognition_v030_sc_data_integration_20260910.log)通过。

## Limitations 与下一步

本轮无机制标签的编译器覆盖 T02、T03、T05、T08 及 T09 组合；22 项先验目录仍未全部成为通用自主编译器。嵌套读出对空库更难，部分收益来自已学程序扩大可搜索语言；简单二次序列共享组计算仍高 40.7%。测量研究仍有最大 MSE 1.74e-4 的冷启动任务，处于现有 .002 停止阈值内。当前主要证据来自有限、无噪声装置；完整优化是三种子小规模运行，SC-KG 原生产流程仍是历史状态。下一步研究何时值得采用新先验，减少简单任务的迁移研究费用，并在带噪和结构变化任务中检验长期总收益。
