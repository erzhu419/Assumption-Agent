# 审计边界 交付完整性与复现

固定源码提交为 `4001bef56ffa77520c248b853e7fed3535ee9fe4`。审阅没有修改该仓库，没有调用付费模型，没有重新进行官方科学评测或启动原工作机容器。保存的成绩是对现有证据的独立重算；程序反例均在临时对象/副本和明确的离线桩上运行。

## 自动审计器四个范围控制

运行 `audit_controls/auditor_scope_controls.py` 可复现以下结果。该脚本只在临时副本中做改动，不修改原回执；为隔离“语义覆盖”与“文件摘要”问题，每个测试副本都重新生成一致的 SHA 清单。Docker 检查显式替换为空容器列表的成功返回，因此这些结果不声称重查了任何真实 Docker 服务。

| 控制 | 临时副本中的改变 | 当前审计器结果 | 原因 |
|---|---|---|---|
| AUD01 | 删除全部 12 份完整调用 JSONL | 八项仍通过 | 只有文件存在时才做 summary 与 JSONL 对账 |
| AUD02 | 删除一份 TB 合题官方 verifier 报告 | 八项仍通过 | 只有 summary 和 verifier 同时存在时才比较 |
| AUD03 | 改变 AMR 反题 post-lock strict_pass | 八项仍通过 | 未对 post-lock 子分支成绩与合题 summary 做原生账本对账 |
| AUD04 | 在 BioMystery 子分支已保存命令中放入反溯源域名字面标记并重算 DAG | 八项仍通过 | 反溯源检查只看顶层合题 DAG；其中没有真实 CONTAINER_EXECUTION 记录 |

AUD04 中的命令仅为字符串，未执行任何网络访问。对真实最新 40 条子分支的 834 条容器命令补做指定反溯源域名及私有路径字面扫描，未命中。这支持“当前可见命令没有这些直接访问”的有限判断；无法据此排除所有间接访问、外部开发信息或先前已知反馈。

这些控制的意义是：哈希清单完整时，审计器仍可能没有审到宣称的性质。修复应增加必需文件数量/类型合同、递归分支账本对账、真实执行路径扫描、候选与评分绑定，不应修改摘要算法来容忍缺项。

## 当前 Git 与原便携 ZIP

最新 SHA 清单四个 pyc 文件未入 Git，当前能重算 2,099 个存在文件，四个缺失。报告的 2,103/2,103 不能从交付复现。两个旧 M6.9 pyc 同样缺失，M6.8 的 899 与 M6.8.1 的 813 项则完整。见 `results/recomputed_results.json`。

原 ZIP 的 CRC 完整，但它只含摘要和部分回执：138 个成员中，只有 111 个对应最新 SHA 清单；不含完整 branch/post-lock/JSONL/producer 链。准确名称应是“摘要审阅包”。完整复现包应连同清单内必要文件、源码依赖、版本与环境说明一起导出，并在新目录执行一次文件与导入闭包检查。对可重建 pycache 可以显式排除，再生成新的发布清单并保留旧清单及勘误。

本审计交付包是诊断、复算脚本和核查结果，不包含整个原仓库、官方私有测试、原始大型中间日志或原便携 ZIP。重算需另有固定提交的仓库内容。这一点与被审报告所声称“自包含完整证据包”的性质不同。

## 本审计如何复现

将所需固定提交 checkout 到本机，并提供完整 `Hegel Machine/src`、`scripts`、`tests` 和 `docs/evidence/m6*`。脚本使用 `HEGEL_AUDIT_REPO`，统一建议指向 `Hegel Machine` 目录：

```bash
export HEGEL_AUDIT_REPO='/absolute/path/Assumption-Agent/Hegel Machine'
python results/recompute_results.py --repo-root "$HEGEL_AUDIT_REPO" --output-dir /tmp/hegel-m691-results
python capabilities/recompute_capabilities.py
python audit_controls/auditor_scope_controls.py
```

其他控制按目录 README 执行：

```bash
python capabilities/qualification_control.py
python capabilities/library_regression_controls.py
python capabilities/structural_controls.py
python capabilities/tracka_regression_controls.py
python synthesis/harness_and_selector_controls.py
python synthesis/cache_identity_control.py
```

结果复算脚本只使用 Python 标准库，不导入项目模块。源码行为控制会导入被审项目及其所需数学库，需与项目环境兼容。Track A 使用本提交中保存的 M6.3 历史现场；不需要重新付费生成这些数据。能力脚本会写同目录结果，应在可写副本执行。

原环境没有 pytest。本次直接执行了六个 synthesis 原测试函数和一个相关复合 bridge/harness 原测试函数，七项通过；没有把它称作重新执行全部 31 项 pytest。原尝试日志与离线替代方式在 `synthesis/existing_tests.txt` 与专项报告中。

## 已证实事实与不同强度的结论

| 级别 | 本次例子 | 可推出什么 |
|---|---|---|
| 原始结果复算 | 18/20 合题、17/20 分支、4141 completed 调用 | 这些保存结果和计数成立 |
| 实际运行缺陷 | 64 次 qualification 全失败、symbolic 未写 OOF、30 份空 build recipe | 本次相应接口或证据链没有发挥宣称作用 |
| 已保存因果顺序 | D3 后官方评分，14 题有独立时间戳佐证 | 支持本题选择前未读两分支官方分数 |
| 离线反例 | NaN/缺入口仍验收、cutoff 回退、长期 Turn 截断、缓存不含技能 body | 存在可重复合同边界漏洞；不表示本轮每个实例都遭遇 |
| 解释性推断 | 提示词减负、物理协议处理、模型随机性共同影响增益 | 需要同预算、同历史的配对消融，不能由版本先后直接断言因果 |
| 尚待确认 | 真实共享结构的跨任务收益、独立 1x 17分、稳定90% | 当前结果不足以确认，下一轮按具体合同补证据 |

此前常见的混淆是从“代码存在→挂载→调用→采用→泛化收益”一路自动升级。新审计要求每一步有其自己的证据，同时保留已经成立的前一步。
