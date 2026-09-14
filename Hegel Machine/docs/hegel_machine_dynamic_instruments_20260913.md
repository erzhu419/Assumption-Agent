# 动态仪器的学习、修订与免测量复用

2026-09-13，0.75.0。普通仪器现在可以从实测中学习有限控制历史的读数模型。更换延迟或增益后，系统购买读数、修订标定并重新绑定原机制；仅响应机制变化时保留正确标定。独立命令轨迹确认后，预测可免去仪器。动态条件求解使用各时刻的真实控制量，历史替换同时保留延迟和重置边界。

三组新种子各执行“首次研究→相同任务→仪器变化→仅机制变化→返回”五题，每题间恢复完整知识库；两臂没有提供固定候选或真实仪器公式。与0.74均15/15，最大终测MSE分别2.79e-15、2.78e-15。新版15个最终程序均免测量，旧版全部需要仪器，见[正式结果](../artifacts/dynamic_instruments_20260913/comparison_v1/summary.json)。

| 成本 | 0.74 | 0.75 |
|---|---:|---:|
| 学习观测行数 | 1272 | 1392（+9.43%） |
| 学习测量单位 | 2520 | 2160（−14.29%） |
| 学习计算单位 | 154876 | 158110（+2.09%） |
| 独立终测3840行的测量单位 | 11520 | 0 |

三次仪器变化均取得字段反例并更新模型；三次仅机制变化没有误撤销字段模型。相同任务和返回任务的六次学习均免测量。首次校准及仪器变化需要额外观测，部署端测量单列，未并入学习收益。90项相关测试通过，覆盖完整恢复、重标定后原机制复用、风险回归及动态窄边界实际求解。

开发组5/5，学习计算55956→51330；正式组计算增加，保留该差异。启动自由项曾在单轨迹中过拟合，现先检验统一有限历史多项式，仍有残差才扩展启动行为；[原失败](../artifacts/dynamic_instruments_20260913/startup_overfit_v1.json)及[首轮开发](../artifacts/dynamic_instruments_20260913/development_v1/summary.json)保留。

## Limitations 与下一步

当前覆盖有限历史多项式和同名测量接口，递归隐状态仪器、一般结构生成与开放研究能力尚未完成。新增时序求解轨迹可以执行，但其步骤尚未进入跨任务方法库。下一步把时序控制依赖、准备历史和区分步骤保存为可重绑定的研究程序，再用新任务的实际收益决定复用、修订或撤回。

运行：[普通会话配置](../config/cognition_dynamic_instrument_session.json)、[正式对照脚本](../scripts/run_dynamic_instrument_study.py)及[协议](../config/cognition_dynamic_instruments.json)。实现：[仪器模型](../src/hegel_machine/cognition/instrument_models.py)、[统一研究](../src/hegel_machine/cognition/unified_research.py)、[历史替换](../src/hegel_machine/cognition/scale_bindings.py)、[动态实验](../src/hegel_machine/cognition/temporal_control_probes.py)。
