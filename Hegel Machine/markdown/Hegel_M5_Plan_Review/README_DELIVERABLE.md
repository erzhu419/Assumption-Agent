# Hegel M5 计划评估附件

固定提交：e2b0fa97af3c05e878a1c7c165ee19ad0c213d2b。

完整报告：[Hegel M5 多尺度时间与共同子树计划评估](https://chatgpt.com/space/page_17e5fb5fb3408191946cad3c5ff04b83)。`PLAN_REVIEW_ZH.md` 为相同内容的离线副本。

## 文件

- `PLAN_REVIEW_ZH.md`：完整中文评估，包括理论反例、引用、有限施工建议及核验边界。
- `integration_evidence_recalculation.json`：固定提交证据的 SHA 校验、五臂账本加总及逐题 Flat 成本归因。
- `evidence_readonly_check_summary.json`：上述结果的简表。
- `midpoint_scope_cold_control.json`：原保存实例在 b=1.40 与 b=1.80 的真实冷载分发控制。该实例与 S02 参数不同，不能将源实例对 S02 的差异当作新的同实例假确认。
- `recompute_evidence.py`：只读复算与两次冷载分发脚本。不会重跑长基准、调用 fit 或购测；输出写入单独目录。
- `SHA256SUMS.txt`：本附件文件的校验和。

本附件不复制仓库源代码或原提交证据包。复算直接从本地 Git 的固定提交对象导出原证据到临时目录，校验并使用后清理。

## 复算

在已安装该项目依赖的 Python 环境中，将仓库切换至上述固定提交，然后执行：

```bash
python recompute_evidence.py --repo '/absolute/path/Assumption-Agent/Hegel Machine' --output-dir '/absolute/path/review-output'
```

`--repo` 指向 Hegel Machine 子目录。脚本核对 HEAD，读取 Git 原始证据，并使用当前固定提交的运行源码。本轮已在 Python 3.12 环境执行成功。

预期要点：144/144 SHA 匹配；140 条主流回执；测量费差只出现在 S27，为 49 M；S27 之外 Flat 比 PoolOn 合计便宜 8.0853 C；两次冷载分发均成功且无 fit 调用。

另一个本轮已执行的聚焦核验，是仓库现成 `tests/test_forecast_experiment_value.py`，4 项通过。它验证任务风险选择器能避开大幅度但无关的预测分歧，具体输入与断言见报告引用的源文件。该测试不包含在上面的复算脚本里。

## 验证范围

报告的线上版本已完整回读，忽略空白后与离线正文一致；197 个内容块、15 个数学块均保留。本轮没有可用的页面视觉预览，未声称完成视觉排版检查。附件 ZIP 已校验可读与 CRC，完整主基准未在本轮重跑。
