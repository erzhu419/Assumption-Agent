# 0.129 自主时间尺度修订

固定12个新装置9914001–12，静态→窗口3→5→7→2→3恢复，权重0.35、σ=0.01。旧0.128累计、新0.129累计、新0.129每题独立三臂，共216题；跨度不提供给Agent。全部学习结束后统一4×64终评，完全离线，外部模型调用0。

每题首次研究上限240观测/24万计算，回访使用链剩余池，链总1440/144万；独立臂保留题内记忆，无跨题传递或资源池。无额外来源任务。

| 正式组（每组72题） | 均值响应MSE | completed | 学习观测 | 学习计算 | 研究CPU秒 | 终评CPU秒 |
|---|---:|---:|---:|---:|---:|---:|
| 旧累计 | 0.9853883 | 43/72 | 15743 | 8721814 | 16557.31 | 3118.93 |
| 新累计 | 3.451243e-05 | 56/72 | 15317 | 7813794 | 9304.64 | 128.36 |
| 新独立 | 15.00469 | 12/72 | 15432 | 3218507 | 144.86 | 4.61 |

新累计相对旧累计：34胜33平5负，12条装置链均值全部改善；学习观测−2.71%、计算−10.41%、研究CPU−43.80%，含终评CPU−52.06%。相对新独立为60胜12平0负，但研究CPU为64.23倍。每臂终评另计18432观测/36864计算；完整有噪误差和全部费用见[summary](fresh_replication/summary.json)。

| 阶段（各12题） | 旧→新累计均值MSE | 旧→新 completed | 新独立MSE / completed |
|---|---:|---:|---:|
| 窗口5 | 0.953274 → 3.84554e−5 | 0 → 5 | 27.1526 / 0 |
| 窗口7 | 4.958917 → 3.54606e−5 | 0 → 8 | 33.0696 / 0 |
| 窗口2 | 4.52665e−5 → 4.62015e−5 | 10 → 9 | 6.01198 / 0 |

最终各12份窗口5/7/2程序均含对应跨度；部分仍同时包含其他跨度。既有报告记录44个尺度候选、30次付费区分检验、27次选作待确认假设。30次检验均留下counterexample标记；这是有噪观测下的实验判定，不等于30个已证实的确定性机制错误。最终直接保留尺度修订且fresh supported的2题为窗口7的3→7与4→7，其余多数继续经删除/重拟合形成最终程序。[具体宽度、检验及最终状态](fresh_replication/temporal_scale_attribution.json)。

开发仅为已知9913802/05/10的原六阶段18题：均值MSE 0.714203→3.61201e−5，completed14→15，5胜11平2负；窗口5三题0→1完成。学习观测+2.36%、计算−0.35%、研究CPU+18.38%，不计作新装置复现。

## Limitations

新累计仍16题未决；窗口2均值误差增2.07%，首次窗口3增6.30%。最差退步为root12窗口2：1.85463e−5→6.22331e−5，双方均completed。研究耗时长尾仍存在：新root03返回窗口3单次530.16秒；旧root09窗口7回访1803.64秒，保留958节点程序。独立冷启动仍未解决历史响应发现。本轮是固定装置家族内的版本与积累比较；独立臂同时去除了跨题传递和回访资源池。CPU围绕研究调用计时，不含检查点序列化，程序规模不当作函数级性能剖析。

[逐题、逐装置与最差pair](fresh_replication/paired_comparison.json) · [研究动作](fresh_replication/attribution.json) · [逐visit CPU及程序规模](fresh_replication/cpu_attribution.json)

复现入口（项目根目录；已有完整检查点会跳过学习）：

```bash
PYTHONHASHSEED=0 python3 scripts/run_nonlinear_mechanism_revision.py --output artifacts/temporal_scale_revision_20260927/fresh_replication --phase learn --workers 12
PYTHONHASHSEED=0 python3 scripts/run_nonlinear_mechanism_revision.py --output artifacts/temporal_scale_revision_20260927/fresh_replication --phase evaluate --workers 12
python3 scripts/run_nonlinear_mechanism_revision.py --output artifacts/temporal_scale_revision_20260927/fresh_replication --phase summarize
python3 artifacts/temporal_scale_revision_20260927/compare_results.py
python3 artifacts/temporal_scale_revision_20260927/attribute_temporal_scales.py
python3 artifacts/temporal_scale_revision_20260927/attribute_cpu.py
```
