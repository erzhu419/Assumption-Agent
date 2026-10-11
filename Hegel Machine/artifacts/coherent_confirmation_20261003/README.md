# 0.172 Coherent confirmation：完成

仅统一pair确认的reading residual分布与响应期望；普通研究、固定fresh、gain及停研语义保留。
52项定向测试通过，一次冻结；24条长链全部学完后完成336项终测，12 workers，外部API调用0。
同源控制仅调用冻结0.171 pair planner，168/168题及12/12完整账本精确复现0.171；另复用0.164参照。
两臂均137/168、MSE 0.02238404；后7均70/84、MSE 0.01765989，逐题与逐链全部持平，无新增失守或恢复。
full/control全资源：87441/85833/44914279 vs 87345/85737/44729929，增加96观测、96测量、184350计算。
本次实测CPU3015.26 vs 3204.36秒；不据此宣称稳定速度收益。
显式pair41 vs 40次，实际区分14 vs 13批、326 vs 302行，27次零增益均未减少。
预期分歧合计614.36 vs 864.45；多一次正向局部区分未转化为终测质量收益。

限制：已见固定cohort，root12 p9退化仍未解决。开发三例两臂各24行全0区分，新臂额外947计算；公共state重建计费，非精确历史恢复。事后逐点复算未反馈Agent。
详细结果见execution_summary.json、compact_results.json、paired_details.json；每链research_evidence.json.gz保留CDF basis、forecast分量与实际fresh差。
