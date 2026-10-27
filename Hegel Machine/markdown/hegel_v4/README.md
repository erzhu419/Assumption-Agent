# 本次交付：从问题语义到理论谱生长

当前实现与待决问题的集中说明见[GPT6交接文档](../../docs/hegel_machine_gpt6_handoff_20261021.md)，其中包括真实主线、完整旧基线、历轮负结果及最新联合搜索。下面保留原规格包的来源与阅读顺序；历史审阅快照不代表当前实现状态。

2026-10-21 更新：当前施工依据为原 [v4](../GPT_diagnosis_v4.md)、下列01／02／03，以及新增 [v5](../GPT6_diagnosis_v5.md)、[诊断与恢复计划](Hegel_5c945310_Diagnosis_and_Recovery_Plan_ZH.md)、[三项交付](Codex_Next_Three_Deliverables_ZH.md)。新建议明确修正“保留并列即禁止行动”的解释，先修合法探索、再做父锚定修订与研究池分离，最后在执行输入到位后验证当前版本科学链。原文件及其历史审阅快照保留；实现进展只记入[原总报告](../../docs/hegel_machine_executable_evidence_repair_20261013.md)。

建议阅读顺序：

1. `01_Concepts_Mathematics_and_Research_ZH.md`：区分问题、假设、泛函、先验与收缩；外部研究的可用部件和限制。
2. `02_Codex_Executable_Question_and_Theory_Spec_ZH.md`：实际数据流、编译、判同、路由、父返回、理论归纳、文件接线及28项行为验收。第15节可直接作为首轮Codex指令。
3. `03_References_and_Source_Boundaries.md`：21组外部文献、当前源码与原始构思的来源及范围。

`reference_math_checks.py` 与 `reference_math_results.json` 只用于核对8个简单数学／语义例子，不是Hegel项目测试，不构成新科学结果。运行：

```bash
python reference_math_checks.py
```

本次没有修改GitHub仓库、运行Luna、购买实验或改变旧论文结果。算法主体是施工规格与伪代码，需要在已有工程中实施和评测。

固定审阅快照：`468256bfc7e9d0c6077f29a0475e7e065d24c6d6`。
