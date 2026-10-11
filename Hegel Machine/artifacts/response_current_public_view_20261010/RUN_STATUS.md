# 实时公开观测接回历史对象视图（2026-10-10）

复用10月7日的 `public_object_context/public_object_relations`，将相同投影接到实时模型输入、程序 `public_spec` 和 `current_observation.json`。原观测字段与索引记录保留，两臂接口相同。

原r3分析源码在同一归档实时观测上重放：旧视图 `refuted/unknown`，修后 `supported/activated`，两臂一致，0新API、0新物理动作。这里只恢复C4状态读取；归档任务仍未完成。见[重放](analysis_replay.json)、[补丁](current_public_view.patch)。24项观测/分析及84项父子回传/上下文检查通过；初次命令遗漏仓库导入路径的两项已用正确路径单独重跑。

另一例早停已有未完成状态和阴性观测，模型仍把候选预测写成实际成功；未发现完成反馈丢失，不强制追加研究。[原记录诊断](early_finish_diagnosis.json)。

[机制变化六次复测已启动](../response_current_public_view_live_20261010/RUN_STATUS.md)：Hegel／同程序Code各3次，从零观测、相同9声明及原预算开始。沿用订阅gpt-6-luna/medium和缓存记账，模型自行提出/运行/修订。该批属于开发复测，不能以接口重放替代自主收益。
