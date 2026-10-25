# 来源与证据等级

固定提交：`ca3ef0357bf4a4345b173a6d9b27a570f7efbc9d`。外部检索与代码分析均在2026-10-21完成。

## 仓库

|编号|路径|已读重点|证据性质|
|---|---|---|---|
|S1|[Hegel Machine/docs/hegel_machine_gpt6_handoff_20261021.md](https://github.com/erzhu419/Assumption-Agent/blob/ca3ef0357bf4a4345b173a6d9b27a570f7efbc9d/Hegel%20Machine/docs/hegel_machine_gpt6_handoff_20261021.md)|最新总览、目标、结果、外置原始数据边界|报告内容，非重跑|
|S2|[Hegel Machine/docs/hegel_machine_executable_evidence_repair_20261013.md](https://github.com/erzhu419/Assumption-Agent/blob/ca3ef0357bf4a4345b173a6d9b27a570f7efbc9d/Hegel%20Machine/docs/hegel_machine_executable_evidence_repair_20261013.md)|最新联合前沿、来源和历史负结果|报告内容，已与代码核对|
|S3|[Hegel Machine/docs/evidence/research_effectiveness_boundary_20261021.json](https://github.com/erzhu419/Assumption-Agent/blob/ca3ef0357bf4a4345b173a6d9b27a570f7efbc9d/Hegel%20Machine/docs/evidence/research_effectiveness_boundary_20261021.json)|4090–4500附近的sparse_joint及前序、冲突数值、成本|机器摘要；外置逐例文件未取得|
|S4|[Hegel Machine/src/hegel_machine/cognition/revision_parameter_relations.py](https://github.com/erzhu419/Assumption-Agent/blob/ca3ef0357bf4a4345b173a6d9b27a570f7efbc9d/Hegel%20Machine/src/hegel_machine/cognition/revision_parameter_relations.py)|1–394：输入/目标、split、sample/unit gate、训练和held checks|静态源码核对|
|S5|[Hegel Machine/src/hegel_machine/cognition/revision_evidence.py](https://github.com/erzhu419/Assumption-Agent/blob/ca3ef0357bf4a4345b173a6d9b27a570f7efbc9d/Hegel%20Machine/src/hegel_machine/cognition/revision_evidence.py)|全文：first_measured_parent_failure摘要|静态源码核对|
|S6|[Hegel Machine/src/hegel_machine/cognition/revision_relations.py](https://github.com/erzhu419/Assumption-Agent/blob/ca3ef0357bf4a4345b173a6d9b27a570f7efbc9d/Hegel%20Machine/src/hegel_machine/cognition/revision_relations.py)|1–300、370–650、770–1175：联合洞、源匹配、参数学习、单位、派生参数消费|静态源码核对|
|S7|[Hegel Machine/src/hegel_machine/cognition/abstractions.py](https://github.com/erzhu419/Assumption-Agent/blob/ca3ef0357bf4a4345b173a6d9b27a570f7efbc9d/Hegel%20Machine/src/hegel_machine/cognition/abstractions.py)|1–450：模板、fitting_shape、结构项和_fit_slots|静态源码及独立重建线性分支|
|S8|[Hegel Machine/src/hegel_machine/cognition/synthesis.py](https://github.com/erzhu419/Assumption-Agent/blob/ca3ef0357bf4a4345b173a6d9b27a570f7efbc9d/Hegel%20Machine/src/hegel_machine/cognition/synthesis.py)|1–220：_solve、_fit_linear、_UnfittedProposals|静态源码；_solve叶函数转录复核|
|S9|[Hegel Machine/tests/test_revision_parameter_relations.py](https://github.com/erzhu419/Assumption-Agent/blob/ca3ef0357bf4a4345b173a6d9b27a570f7efbc9d/Hegel%20Machine/tests/test_revision_parameter_relations.py)|1–290：1e-8合同、精确affine、sparse13输入fixtures|阅读测试；未运行完整pytest|
|S10|[Hegel Machine/src/hegel_machine/cognition/finite_native_author.py](https://github.com/erzhu419/Assumption-Agent/blob/ca3ef0357bf4a4345b173a6d9b27a570f7efbc9d/Hegel%20Machine/src/hegel_machine/cognition/finite_native_author.py)|1–260：导出、比较复用、预算、有限作者|静态源码核对|
|S11|[Hegel Machine/src/hegel_machine/cognition/question_selection.py](https://github.com/erzhu419/Assumption-Agent/blob/ca3ef0357bf4a4345b173a6d9b27a570f7efbc9d/Hegel%20Machine/src/hegel_machine/cognition/question_selection.py)|350–540：并列操作选择及逐计划可计算性|静态源码核对|

## 独立复核

`diagnostic_checks.py`和`diagnostic_results.json`是本报告中独立执行的有限检查。没有重新运行723个真实结构、全部1446个历史Program或新的科学来源。代码使用公开小型合成例和机器摘要中的四个数值；最小消元函数按固定源码转录。

## 外部原始文献

- R1：LAPACK DGELSD官方说明：https://www.netlib.org/lapack/lapack-3.1.1/html/dgelsd.f.html 。只引用其SVD最小二乘和有效秩处理，不依赖此历史文档判断最新软件版本。
- R2：Andrychowicz et al. (2016), Learning to learn by gradient descent by gradient descent：https://arxiv.org/abs/1606.04474 。支持“学习优化过程”是一项不同且合法的目标，不能作为本项目当前有效性证据。
- R3：Brunton, Proctor & Kutz (2016), Discovering governing equations from data by sparse identification of nonlinear dynamical systems：https://doi.org/10.1073/pnas.1517384113 。只借用变量／坐标／稀疏基选择的重要性，不将其保证套到当前元参数数据。
- R4：Bowers et al., Top-Down Synthesis for Library Learning：https://arxiv.org/abs/2211.16605 。共享程序结构与压缩是计算抽象，不等于独立世界支持。

## 未核验范围

没有访问用户本地 `/mnt/hegel-snapshot-cow/...` 或 `/mnt/d/Hegel-History-Archive-20261012/...` 的完整文件；Git只给出引用时保留这一边界。没有用旧论文成功结果补记当前版本的运行。没有穷尽仓库每一个旧模块，覆盖的是本次失败的主要生成、拟合、消费、选择和验证链。