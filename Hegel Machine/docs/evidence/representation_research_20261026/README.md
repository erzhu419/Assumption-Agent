# 表示修订与区分实验：2026-10-26小证据包

`evidence.json` 汇总本轮事实和新增费用：743193＋194＋930＋930＋5905＋43939＝**795091计算**，新增4行观测、0次provider调用。历史费用通过原账本引用保留，不重复相加。`manifest.json` 将每份材料映射到外置原现场；不包含方法图、完整候选池或快照。

`models/` 保留实际新7／新9源码。`predictions/` 包含旧97行两稿预测、训练465行旧259／新9预测和验证144行旧259／新9预测。验证向量是已存结果，打包没有执行模型。旧512行及259／455源码复用相邻终测证据包。

按17个原record恢复材料和固定split：

```python
import json
from pathlib import Path
base = Path("docs/evidence/representation_research_20261026")
binding = json.loads((base / "material/records17_split.json").read_text())
records = []
for item in binding["record_material"]:
    ref = item["rows_ref"]
    rows = json.loads((base / ref["file"]).read_text())
    for part in ref["json_path"]:
        rows = rows[part]
    if "slice" in ref:
        rows = rows[slice(*ref["slice"])]
    records.append(rows)
training = [row for i in binding["split"]["training_records"] for row in records[i]]
validation = [row for i in binding["split"]["validation_records"] for row in records[i]]
assert (len(training), len(validation)) == (465, 144)
```

`distinguishing_selection.json` 保留原并列和预登记的运营选择；`distinguishing_purchase.json` 保存实际4行，`measured_question_returns.json` 保存正确绑定后的两个`observed_false`答案。合同补交中的失败及费用也保留。

## 证据范围

原`1e-10`门槛未通过。144行验证只直接比较两份父稿与15份native提案；3561稿强池的完整比较使用609行，不能据此宣布完整强池的验证胜出。当前没有学出合格适用条件，也没有本轮父稿再次修订、独立确认、独立迁移或因果PR3收益。本包已追加新7递归尝试：在更严格的150秒主机时限中断，未导出、比较或采纳。原生时间与外部Python计量不同，校正子账单列；原内部拟合值没有落盘，无法补回。后补代码仅保证今后已完成拟合值可保留和零拟合恢复，不称本次已恢复。

后继已沿既有入口正式准备并完成冷载入：`successor_registration.json`、`successor_cold_acceptance.json`和`successor_budget_contract.json`。3578稿和原执行上下文完整保留；B／C共用旧20对象，最新单来源对象效应不适用。原150万计算不足，未加额度或启动科学执行。584份源码冻结留在外置盘，包内只存小回执。另有14项prepare接线测试通过，见`successor_prepare_tests.json`。
