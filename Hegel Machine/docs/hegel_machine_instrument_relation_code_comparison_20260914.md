# 输入与关系修订：真实普通 Code 对照

2026-09-14。双方获得相同源规律、校准、付费数据及首次九行实验；每题512行观测、6144测量费用。Code固定 `gpt-5.6-sol` / `ultra`、12次调用及240000调用间token阈值。0.85已统一运行、拟合和评估的测量视图，并提供可直接评估的细测源模型；16项相关测试通过。

[前三批调用记录](../artifacts/instrument_relation_code_comparison_20260914/execution.json)保留原故障与预算停止。[第四批开发](../artifacts/instrument_relation_code_comparison_20260914/development_v4/summary.json)两题均停研：共7次调用、617352已报告token，工具执行无新错误，但0/2完成模型评估或交付。

[预算末自写程序的补充评分](../artifacts/instrument_relation_code_comparison_20260914/development_v4/supplementary_written_models.json)原样测试唯一写出的model.py：随机MSE 0.00312145、边界MSE 0.00165340，均未达标；另一题未写模型。额外337行终测、0测量费用、337计算单列，未改变原协议结果。

## Limitations 与下一步

本批是同一已知装置上的两个相关开发任务，正式36题未派。比较范围为给定有限源先验后的修订，计算单位与模型用量分列。下一步需要改善真实Code的实验选择与预算内交付，再扩大比较；目前不能据此宣称黑格尔机普遍优于普通Agent。认知主线见[跨任务反例研究](hegel_machine_relation_trajectory_evidence_20260914.md)。
