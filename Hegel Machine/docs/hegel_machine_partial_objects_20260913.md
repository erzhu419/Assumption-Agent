# 部分对象知识与角色修订

2026-09-13，0.66.0。已确认的数量、驱动和位置可直接形成预测候选与仪器模型；新观测支持当前解释时，系统保留其余归属未决，后继需要焦点角色时再补齐。部分图随知识库保存和恢复，实际用于后续调用。

角色解释同时比较纯加性输入、仅改变相互作用的坐标、以及兼有直接作用的坐标。已学区分实验在后继测量中检验旧解释；即使控制到读数的校准相同，转移依赖改变仍可撤回旧定义并重建对象关系。

已学观测函数还可逐对象重绑定选择参数、汇总输出，再与差分和窗口组合。这让单对象知识服务于后继总量问题；原函数与来源保留，组合结果接受新观测选择。

## 实际结果

每组从空知识按总量→稳定总量→焦点→事件→组合→窗口→位置增加直接作用→窗口后继运行。两版本每题同为512观测、6144测量、480000搜索计算预算，每题终测使用4条独立64步轨迹。开发中旧0.65和首版新实现均仅7/8，末题窗口MSE分别为0.0361和0.0309，[两组失败](../artifacts/partial_objects_20260913/development_v1/summary.json)保留。补上逐对象调用后，新版完整8/8，末题MSE为6.82e-19。

三组新装置首轮新版仅20/24，独立冷启动2/3。失败装置的角色全部恢复正确，但边界实验固定选择了待测对象对之外的对象，12轮均未触及关键相互作用；错误半径随后传播到组合任务。现已用实付控制与读数配对确定实际焦点，边界、置换和局部性实验均按这一关系干预。首轮结果保留于[完整序列](../artifacts/partial_objects_20260913/comparison_v1/summary.json)和[独立起步](../artifacts/partial_objects_20260913/anchored_comparison_v1/summary.json)。

修复后的**相同三组装置复测**结果如下，两个版本均已完成全部任务。新版终测全部免测量；原失败装置仍使用12轮×5行边界实验，三个问题的区间宽度均缩小4096倍，两个焦点和两个窗口问题全部恢复。

| 24题完整序列 | 0.65 | 0.66 |
|---|---:|---:|
| 终测通过 | 19/24 | 24/24 |
| 学习观测 | 5347 | 4790 |
| 学习测量 | 41298 | 37443 |
| 搜索计算 | 3780537 | 1721208 |

序列观测、测量和搜索计算分别减少10.42%、9.33%、54.47%。独立空知识起步另计：通过率0/3→3/3，观测794→1090、测量5808→8100、搜索计算1009176→663895。结果见[修后序列](../artifacts/partial_objects_20260913/comparison_v2/summary.json)及[修后独立起步](../artifacts/partial_objects_20260913/anchored_comparison_v2/summary.json)。

实际执行中，三次按当前响应补齐部分图、三次因角色依赖反例重开研究，五题采用逐对象调用组合，包含三个末尾窗口，见[实际采用记录](../artifacts/partial_objects_20260913/comparison_v2/behavior_summary.json)。三组各自前两题使用部分图使测量减少44.88%；三个归属补齐任务的测量则增加14.46%。

实现：[部分知识](../src/hegel_machine/cognition/partial_object_knowledge.py)、[角色竞争](../src/hegel_machine/cognition/object_induction.py)、[依赖实验](../src/hegel_machine/cognition/object_interventions.py)、[仪器学习](../src/hegel_machine/cognition/instrument_models.py)、[调用组合](../src/hegel_machine/cognition/research_programs.py)、[结构实验](../src/hegel_machine/cognition/structural_discovery.py)与[共同主循环](../src/hegel_machine/cognition/unified_research.py)。组合与部分会话22项、结构实验32项回归通过，另覆盖角色撤销与部分仪器学习。入口：[运行配置](../config/cognition_partial_objects.json)、[序列脚本](../scripts/run_partial_object_study.py)、[独立起步脚本](../scripts/run_anchored_object_study.py)。

## Limitations 与下一步

本轮仍使用有限角色与依赖解释族，逐对象推广从无历史观测函数开始；两个对象的对称归属及更一般潜在对象仍可能不可识别。依赖复测需要实际投入，分阶段成本可能增加。本批为版本比较，没有隔离各新增机制的独立贡献，也没有新增真实LLM对照。

下一方向是让仍不可确定的对象划分产生可执行的等价解释，并依据任务需要选择新的测量或干预，而非要求每个问题都先确定唯一对象身份。原构想进展见[目标复核](hegel_machine_advice_goal_status_20260909.md)。
