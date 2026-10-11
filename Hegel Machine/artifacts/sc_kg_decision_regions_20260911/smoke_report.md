# 决策区域：首轮烟雾结果

5例均可行；相对默认0.55为0改善、3相同、2退步，停止扩展。新增5次优化、160次模拟，未使用新确认种子。

|任务|0.55|区域方案|Native|区域识别请求|安装|目标计算|
|---|---:|---:|---:|---:|---:|---:|
|unknown_confirmation_3_center_interaction_change|0.35102|0.35516|0.36368|4|1|19995|
|goal_completion_2_center_interaction_change|0.36765|0.36765|0.50777|4|0|19239|
|goal_recompetition_3_center_interaction_change|0.35638|0.35638|0.53448|0|0|18391|
|joint_identification_3_stable_interaction|0.41669|0.41669|0.39516|0|0|3519|
|joint_identification_3_center_interaction_change|0.35862|0.48393|0.40532|4|0|16176|

平均最优值差距（目标下界0.35）：0.020072→0.045962；Native为0.091282。共12次区域识别请求，唯一安装为加性模型，无交互模型安装。

最大退步任务的区域实验集中在x3=1的模型假极小附近，未改善最终解。下一步保留互反实验方向，检验当前低值区域是否被错误模型限制。
