# 已有反例与模型指定实验接回现成控制器

旧`query_response_family`已在上一轮接到新声明提交前；本轮复用原安装、调频、实测与参考恢复，向子研究开放`reference_response`的`experiment:{number,setting}`。同点续跑沿用问题；改选保留旧题未决记录并执行新点。实际到达设置、阴阳响应和证据直接返回原LLM循环。

旧task02第29事件明确要求晶体4@1012，却提交1646动作、耗尽剩余262动作。从相同250观测前缀重放这个已记录意图，新入口6动作到达1012，实测阴性；没有把途中响应或恢复状态当作该设置的成功。28项原回归、6项新增测试通过。

复现：`PYTHONPATH=src python3 artifacts/response_authored_control_20261009/replay.py`。实测见`result.json`。这是操作意图的宿主转写；模型自主采用另见`../response_controller_adoption_20261009/`。

## Limitations

控制器正确执行并不保证实验有区分力；不可达设置返回公开网格，不自行替换科学意图。本次重放不证明1012是有价值的实验。
