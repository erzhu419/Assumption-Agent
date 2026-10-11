# 机制删除与恢复：0.121 正式结果

正式批 `fresh_replication_v2` 使用预登记的 12 个新 9989 根、每根两套装置。216 次初访及 9 次自然回访全部结束后，统一完成 216 个任务的初访／最终终测。新版 72/72 达标并正确完成，12/12 整链通过；旧版整链 10/12，空库起步整链 8/12。没有修改正式种子、预算或选择规则。

| 臂 | 达标／正确完成／误报 | 学习：观测／测量／计算 | 全成本：观测／测量／计算 | 来源＋研究 CPU 秒 |
|---|---:|---:|---:|---:|
| 旧 0.120 累计 | 70／58／0 | 12373／113256／9378103 | 35251／209520／12187572 | 907.259 |
| 新 0.121 累计 | 72／72／0 | 11561／96828／2845377 | 33415／134724／5654846 | 198.008 |
| 新版空库起步后累计 | 66／50／3 | 12777／124860／8757692 | 32489／161724／8794556 | 645.641 |

全成本包括来源、学习和终测。每个 seeded 臂来源合计为 3422 观测、37896 测量、2772605 计算、46.803 CPU 秒；空库来源为零。终测 CPU 未单独记录。新版相对旧版全计算 −53.60%、全测量 −35.70%；相对空库全计算 −35.70%、全测量 −16.70%，但全观测 +2.85%。终测测量为旧／新／空库 58368／0／36864；新版最终只需已知控制量，终测仍支付 18432 次响应观测。

新版 48 次真实机制区分实验共 424 行、48 次原程序反例；最终 22 次窗口删除、26 次完整重拟合均经新轨迹确认，另有 11 次继承范围区分实验。按 `mechanism_revision` 元数据归因，包含改写后类别为 `research_program` 的候选。两个恢复阶段均 12/12 通过。预测配对相对旧版 2 胜 0 负，相对空库 6 胜 0 负。

六阶段为完整窗口组合 → 去窗口 → 再去事件 → 事件恢复 → 第二装置与匿名接线 → 完整组合恢复。各阶段上限交替为观测 256/384、计算 180000/240000、测量 3072/4608；完整 session 总上限为 1920/1260000/23040，筛选与回访只花未用余池。Agent 只见普通匿名任务接口与付费观测。全程 `PYTHONHASHSEED=0`，终测每任务四条 64 步新轨迹、MSE 阈值 `1e-7`，全离线、外部 API 调用为零。

结果见 `fresh_replication_v2/{summary,mechanism_outcomes,cost_totals,next_gap}.json`。下一主线是冷启动仪器误差与关系范围联合修订，特别是校准变化后重评依赖它的范围证据。空库 root07 新学过窄范围，后继校准变化后仍沿用，三个任务错误完成；新版有来源臂未出现此问题。其余 5 个终测未达标任务均在计算耗尽附近停止，仍有正的确认误差；具体证据与归因边界保留于 `next_gap.json`。

开发记录原样保留：初版 `development`；9988 的 `fresh_replication` 因范围研究元数据丢失停止于 39 次初访、未终测，未完成调用费用未知；修复 `development_v2` 完成 36 初访＋2 回访，通过旧／新／空库 12／12／11。9988 不计正式结果。

复现正式批（完整检查点会复用；开发改为 `development_v2`）：

```bash
PYTHONHASHSEED=0 python3 scripts/run_knowledge_update_revisits.py --output artifacts/mechanism_revision_20260926/fresh_replication_v2 --phase learn
PYTHONHASHSEED=0 python3 scripts/run_knowledge_update_revisits.py --output artifacts/mechanism_revision_20260926/fresh_replication_v2 --phase evaluate
PYTHONHASHSEED=0 python3 scripts/run_knowledge_update_revisits.py --output artifacts/mechanism_revision_20260926/fresh_replication_v2 --phase summary
PYTHONHASHSEED=0 python3 scripts/summarize_mechanism_revision_study.py --output artifacts/mechanism_revision_20260926/fresh_replication_v2
```

局限：收益来自有限匿名对象 MSE 任务族与已有真实来源库，空库臂不是普通 LLM。新累计臂未采用事件删除或单支路删除，去事件任务由已有仿射解释解决。最终命令空间程序与原测量空间程序不可直接比较节点数；同一机制实验内的候选大小才可比较。生产保持冻结，所有学习、终测及汇总进程已结束。
