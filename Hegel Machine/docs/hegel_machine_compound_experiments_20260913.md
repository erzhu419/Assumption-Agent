# 组合条件实验与方法适用范围

2026-09-13，0.71.0。实验求解同时满足多个窄窗口、嵌套分支及共享控制条件；程序方法保存完整条件关系，按当前阈值和标定重新执行。标定反例只限制失败字段的当前映射，修订后恢复原方法；未区分的实验限制对应任务与假设条件。一般可控候选也接入实际采购、比较和重新选择，新增反例参与之后的采用判断。

开发固定有限先验任务旧版3/6、新版6/6；正式三组新种子旧0.70为10/18、新版18/18，新版MSE均为0。旧版始终剩两个兼容候选，正确题为首项恰好正确；新版18题全部由新实测唯一识别。观测90→186、测量225→465、计算76698→22116（减少71.16%）。终测分别覆盖全域、激活、非激活及候选分歧区域。24次组合实测包含18次区分与6次标定反例；原方法修订适用条件后及改名后继成功复用12次。见[正式结果](../artifacts/compound_experiment_methods_20260913/comparison_v1/summary.json)。

既有对象主线只新增运行0.71的15题，复用上批已完成的0.70对照：两版均15/15，观测4785、测量19206不变；计算2249359→2247861（减少1498），跨任务方法支持47→69次，方法学习次数5→3。见[主线比较](../artifacts/compound_experiment_methods_20260913/object_regression_v1/summary.json)。82项相关测试通过，包括普通研究入口的实测修订、保存恢复、坏标定重测和条件端点。

实现：[组合求解](../src/hegel_machine/cognition/program_condition_sets.py)、[方法适用条件](../src/hegel_machine/cognition/program_experiment_methods.py)、[共同主循环](../src/hegel_machine/cognition/unified_research.py)。运行：[任务配置](../config/cognition_compound_experiment_methods.json)、[实验脚本](../scripts/run_compound_experiment_methods_study.py)。

## Limitations 与下一步

完整知识库的快速程序迁移仍会漏检窄激活区域：已保存的[后继反例](../artifacts/compound_experiment_methods_20260913/short_transfer_gap.json)中，短验证误差为0，真实激活处预测0、实测1，组合方法未被调用。下一步把组合实验接入快速迁移的采用判断。上述新任务给定有限门条件与仿射仪器先验，标定漂移挑战是受控适用性检验；路径与数值搜索有界，不代表一般结构语法、任意研究操作或自主长期投入已完成。本批未新增真实普通Code对照。
