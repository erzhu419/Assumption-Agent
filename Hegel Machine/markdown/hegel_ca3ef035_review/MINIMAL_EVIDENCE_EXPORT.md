# 最小原始证据导出：不需要完整工作区

## 用途

量化本报告发现的数值与语义问题，在相同来源、相同split、相同固定结构上复核。导出不新增观测、不调用模型、不读取隐藏环境定义、不覆盖原回执。

## 输入来源

优先从现有 `research_effectiveness_boundary_20261021.json` 的 `sparse_joint_parameter_induction` 解析原文件引用，尤其：

- `comparison_ref`
- `failure_diagnosis_ref`
- `fit_failure_readout_ref`
- 各condition的`summary_ref`
- 原source records和各before／after指针。

不得根据缺失数据自行补造值。

## 最小包

### 1. manifest.json

记录源码提交、实际加载模块版本、源数据引用、提取时间、用途和未取得字段。报告实际字节大小。所有文件只包含本问题需要的切片。

### 2. source_revisions.jsonl

每个实际修订一条：

- physical source、episode、before／after引用与时间顺序；
- before／after完整小Program；
- 实际参数路径、值、对应基函数角色、单位和作用域；
- 拟合所用行集合和权重的引用、正则／算法／初始化说明（有则取，无则标unknown）；
- 原训练、确认、源终测状态；
- 原失败记录和费用引用，不复制整条大轨迹。

### 3. metadata_learning_rows.json

为三类构造核保存实际13输入、7输出及其名称和角色（具体维数以现有对象为准，不强行补齐），split索引、physical source身份、已付证据去重身份。输入应完全来自拟合前可用信息。

### 4. selected_fits.json / matrices.npz

先选四类：原最佳train、一个条件良好的仿射、一个低变化或近共线、报告中的输入冲突。

对每个固定结构提供：原Program与parameter_paths、实际线性设计矩阵、target、初始化、原返回参数、训练／held预测、原loss/容差／status、rank和条件数若已计算。没有保存的设计矩阵可以由同一公开Program及同一元数据重建，记录这是新诊断计算；不得使用未来世界标签。

若小批证实数值问题，再一次性按相同规则导出／重评所有适用的线性结构。没有必要重做候选生成。

### 5. alias_and_units.json

保留`1`／`dimensionless`原始声明来源、系数与基函数的单位关系、明确别名和未解析项。只公开合同，不读取真实机制。

## 必须保留的界限

- 这些是开发历史，不会因为导出就重新变成独立测试。
- 同一个原始反例的多个引用不增加来源数。
- 不把after当世界真值。
- 不从最终正确解补早期未知输入。
- 缺失行、缺失矩阵、缺失拟合状态分别记录，不以零填充。
- 返回的数据不自动写回原研究库的科学支持状态。
