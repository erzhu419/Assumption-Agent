# Hegel v12 / 6e963206 独立审阅包

## 阅读顺序

1. `01_系统审计与主目标判定.md`：核心判断及原因。
2. `02_下一轮最小施工与实验合同.md`：有限后续工作。
3. `03_来源与复核范围.md`：本轮真正阅读、执行与没有执行的内容。
4. `recomputed_results.json`：完整保存回执的独立复算。
5. `probe_results.json`：实际源码片段的局部控制结果。

本包不包含用户全部原代码/证据；输入是用户提供的 `Hegel_v12_GPT6_Audit_Handoff_20261117.zip`。
没有修改仓库，没有重跑科学来源与后继，没有付费模型调用。

## 本地重现

使用 Python 3.10 或更高版本，将原审计ZIP解压到一个目录，保留 `bundled_sources/`、`receipts/` 等结构。脚本只使用标准库。

```bash
export HEGEL_V12_BUNDLE="/absolute/path/to/extracted/Hegel_v12_GPT6_Audit_Handoff_20261117"
python review_probes.py
python recompute_receipts.py
```

`review_probes.py`只抽取指定函数及过滤块，使用显式最小适配器；它不是全仓导入或完整Agent重放。
`recompute_receipts.py`只读取保存JSON；它不拟合模型，不运行仿真环境。

随包作者校验器的本轮重新执行输出保存在 `bundle_verifier_reproduced.json`；其 `all_checks_passed` 只覆盖该校验器的实际检查，不意味着所有科学结论被独立复现。

## 不改变的边界

H*删除打平是应保留的阴性结果。后续不得削弱基线、放宽世界门槛或增加隐藏提示以强迫它获胜。
优化普通档案后若仍打平，应缩小科学主张；若分支组织继续有效，那已可以是原目标的一项有限成果。
