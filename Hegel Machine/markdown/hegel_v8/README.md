# 3806d239审阅包

基线：`3806d239b5c53b9b78ac23fcf7fa4c74f16c27ed`，2026-10-27。

阅读顺序：

1. `Hegel_3806_Current_Diagnosis_and_Mainline_Plan_ZH.md`：当前事实、归因、主线取舍与研究设计。
2. `Codex_Minimal_Method_Research_Contract_ZH.md`：首个最小单位的输入、文件接线、生成/选择、完整预算和24项拟议验收。
3. `Mainline_Status_Matrix_ZH.md`：v4—v7整体需求状态概览。

`source_index.json`给出固定提交的来源定位与本轮读取范围；`mainline_matrix.json`保存结构化需求矩阵。

`check_readonly_diagnostics.py`仅重算便携回执中的数字，并运行一个合成引用依赖示例。它不导入项目、不执行预测器、不拟合、不购买数据，也不是项目回归测试。运行：

```bash
python3 check_readonly_diagnostics.py
```

输出`readonly_diagnostic_results.json`。原始精度依赖已提交回执，本包未独立重现外置科学工作区。

本包不包含新实验授权；未修改远端仓库。所列函数扩展、候选上限、开发四格及后继数量属于建议合同，不冒称已实施能力。旧实验的失败、未知费用、原阈值与强基线应继续保留。
