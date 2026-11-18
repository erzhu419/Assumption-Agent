# c71e65f7 / V13 审计与下一轮方案

阅读顺序：

1. `01_V13_系统审计与主目标判定.md`：真实进步、主对照混杂、剩余语义与接口缺口。
2. `02_下一轮最小施工与实验合同.md`：现有函数的最小修改、六个生产路径回放、公平对照和同对象连续实验。
3. `03_来源与复核范围.md`：来源、实际执行、限制与文献。
4. `recomputed_results.json`：从完整原始回执独立复算的机器结果。

## 复核脚本

需要用户原始 `Hegel_v13_GPT6_Audit_Handoff_20261117.zip` 的解压目录；该目录应直接包含 `05_STANDALONE_VERIFY_V13.py`、`bundled_sources/` 与 `receipts/`。本审计包不重复打包原项目源码和大型原始回执。

以下脚本仅需 Python 标准库，不调用网络、模型 API 或完整科学任务：

```bash
python probes/recompute_v13.py --bundle /absolute/path/to/v13_bundle --output recomputed_results.json
python probes/probe_acquisition_contract.py --bundle /absolute/path/to/v13_bundle --output acquisition_probe.json
python probes/probe_question_identity.py --bundle /absolute/path/to/v13_bundle --output question_identity_probe.json
python probes/probe_scope_and_routing.py --bundle /absolute/path/to/v13_bundle --output scope_routing_probe.json
```

源码摘取探针使用明确的最小依赖替身。请按每份结果的 scope/limitation 字段解释，勿将其视为完整 Agent 回放。完整报告中已经区分作者实测、本次算术复核、本次源码控制和下一轮尚待实验的推断。

`evidence/source_verification.json` 保存关键文件与指定提交对应的 SHA；`evidence/author_standalone_verification.json` 是本次运行作者校验器的输出；其余 JSON 是本次实际探针结果。

**总体判断：V13 修通了真实的同次失败递归修复，但“同一个新假设在公平强普通重建条件下改善未见研究”的完整证据仍未完成。**
