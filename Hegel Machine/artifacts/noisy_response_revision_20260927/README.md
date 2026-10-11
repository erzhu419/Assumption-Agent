# 0.127 含噪响应修订

最终版离线比较0.126：空库累计学习静态平方→窗口平方，σ=0.01。每题研究上限240观测/24万计算，链共480/48万，回访使用余量；无额外来源。各组全部学习后统一4×64终评，噪声任务以独立均值误差为主，不设1e−7达标门槛。

| 最终组 | 均值MSE：旧→新 | 胜/平/负 | 完成：旧→新 | 学习观测：旧→新 | 学习计算：旧→新 |
|---|---|---|---|---|---|
| 已知5装置开发，10题 | 8.8789e−5→4.4432e−5 | 6/3/1 | 9→9 | 1400→1897 | 812051→624082 |
| 新12装置9913601–12，24题 | 5.4537e−5→4.1596e−5 | 9/11/4 | 23→24 | 3489→3666 | 2086506→1577613 |
| 无噪保持2装置，4题 | 同为2.1419e−16 | 0/4/0 | 4→4 | 416→416 | 60010→60010 |

正式均值误差−23.73%、独立有噪MSE−7.97%，学习观测+5.07%、计算−24.39%。按装置合并为7胜/3平/2负；窗口均值−32.14%，静态+1.20%。含终评总费用为9633→9810观测、2098794→1589901计算；全部来源/学习/终评/CPU见[fresh_replication_noise_support/summary.json](fresh_replication_noise_support/summary.json)。

首正式9913401–12保留在fresh_replication：0胜/23平/1负，root08窗口均值误差升至3.97685仍报完成。付费历史定位为未决来源缺失与大误差获噪声支持；修复后该已知反例只作为开发，窗口均值为7.5792e−5并完成。最初validation_loss接口失败及中间开发快照均保留，未覆盖为正结果。

四个最终正式负题均未采用响应重拟合：root02与root12静态因提前重复支持停止，省去后续研究但误差增加；root02变化的来源概念传到窗口；root10窗口追加重复改变后续训练与保留机制，虽然由未决变完成，误差仍变差。见[negative_cases.json](fresh_replication_noise_support/negative_cases.json)。开发root02窗口的refit通过独立收益确认，但重复残差下界3.489e−7仍超过1e−7，216/240观测后保留未决。

从项目根目录复算汇总：
```bash
python3 artifacts/noisy_response_revision_20260927/compare_results.py
PYTHONHASHSEED=0 python3 scripts/run_nonlinear_mechanism_revision.py --output artifacts/noisy_response_revision_20260927/fresh_replication_noise_support --phase learn
PYTHONHASHSEED=0 python3 scripts/run_nonlinear_mechanism_revision.py --output artifacts/noisy_response_revision_20260927/fresh_replication_noise_support --phase evaluate
PYTHONHASHSEED=0 python3 scripts/run_nonlinear_mechanism_revision.py --output artifacts/noisy_response_revision_20260927/fresh_replication_noise_support --phase summarize
```
各组protocol.json/runtime保存任务与运行快照，已完成checkpoint和终评会跳过重复执行。

局限：收益限于固定静态/窗口族及一种噪声幅度，仍有静态退步和概念估计误差向后继传播；完成数不能代替均值误差。下一重点是噪声下研究停止与后继概念质量之间的取舍。
