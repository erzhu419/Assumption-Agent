# 下一核心的历史接线

本轮 shifted_2/gain_higher 的2167请求被反驳，途中2169阳性随后被恢复，任务完成。10月8日已实现此流程：沿途证据筛选旧假设，成功设置用于恢复及后继拟合，原错误预测仍保留反证。因此少一次请求确认不是恢复能力缺失；不应另造“2169会成功”的题目补数字。历史证据见 `artifacts/discoveryworld_observed_settings_20261008/RUN_STATUS.md`。

下一窄缺口是实验规划与执行的粒度不同。0.69/0.76已有包含准备历史的完整轨迹实验；当前DW适配只比较终点Boolean，执行却逐步调节并利用沿途反馈。本轮未实现路径预测或路径计费。

入口（均相对项目根目录）：

- `src/hegel_machine/cognition/world_reference_recalibration.py:217` `_tune`：从当前公开菜单选一步；791–829行的 `select_actions`保存沿途成功设置并恢复。
- `src/hegel_machine/cognition/world_reference_recalibration.py:355` `answer`：全部实测筛候选；异于请求点的成功不确认原预测。
- `src/hegel_machine/cognition/world_experiment_choice.py:24` `select_experiment`：每项procedure适配为单行Probe，只表示终点。
- `src/hegel_machine/cognition/models.py:25` `Probe`及`src/hegel_machine/cognition/probing.py:525` `choose_probe`：现有多步序列、逐步预测和长度费用接口。

最窄实验：先用已保存的一组真实菜单与动作轨迹，检验按公开菜单预告的下一段调节能否逐步复现实际设置；不得假定未见对象网格或菜单恒定。仅对可重建段比较终点选择与序列选择，保留原模型、预算和真实恢复动作，记录实际动作、反驳及任务结果。不添加宿主提供的正确频率或新答案。
