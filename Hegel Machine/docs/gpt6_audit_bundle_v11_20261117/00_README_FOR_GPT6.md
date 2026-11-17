# Hegel Machine v11 & v11.1 GPT-6 审计交接包

## 阅读顺序

1. `01_V11_AUDIT_REMEDIATION_AND_LIFECYCLE_DOSSIER.md` —— 对 `v10.4` 过度表述的正式撤回、交付 A（原子版本索引＋强普通持久化缓存平手复核＋暂定奥卡姆）、交付 B（8 阶段可执行分支调和研究链 `B1–B10`）、交付 C（`v11` 原样保留诊断回执 + `v11.1` 全新种子段 `985000–990000` 候选生成器专项改进 `2×2` 验证）主报告。
2. `02_ACCEPTANCE_TRACEABILITY_A_B_C.md` —— `02_下一轮实施与验收.md` 全部验收条款逐条代码／回执／单测映射表。
3. `03_SOURCE_AND_RECEIPT_SCOPE.md`（配套 `04_SOURCE_AND_RECEIPT_MANIFEST.json`）—— 全部 **26 份完整 JSON 回执**（22 份冻结旧回执 + 3 份 `v11` 回执 + 1 份 `v11.1` 前瞻改进回执）与 **24 个完整源码／脚本／测试文件**清单。

## 一键零依赖沙箱独立核验

```bash
python3 05_STANDALONE_VERIFY_V11.py --output verify_report.json
```

仅使用 Python 标准库即可独立验证：
- AST 提取 `_prior_state_fingerprint` 通过全部 6 类语义修改测试（`undetected_semantic_mutations == 0`）；
- 全部 26 份回执与 24 份源码 SHA256 完整一致，且 `v10` 13 份原始回执与 `v11` `971000–976000` 诊断回执零字节改动；
- `v11` 与 `v11.1` 四份回执满足 `model_calls == 0`、单题巩固无双重计费（`<= 1`）、68 条压力夹具隔离、暂定奥卡姆状态、`B1–B10` 全通过、三分区及跨版本种子零重叠与评估器不可变；
- 随包全部 9 个运行时认知模块（含完整 `unified_research.py`）对 `lab._...` 私有属性零读取。
