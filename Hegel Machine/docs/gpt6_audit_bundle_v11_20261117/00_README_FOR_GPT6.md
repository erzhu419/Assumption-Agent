# Hegel Machine v11 GPT-6 审计交接包

## 阅读顺序

1. `01_V11_AUDIT_REMEDIATION_AND_LIFECYCLE_DOSSIER.md` —— 对 `v10.4` 过度表述的正式撤回、交付 A（原子版本索引＋强普通持久化缓存平手复核＋暂定奥卡姆）、交付 B（8 阶段可执行分支调和研究链 `B1–B10`）、交付 C（`K0/K1 × M0/M1` 正交 `2×2` 与有界 `M1->M2` 检验）主报告。
2. `02_ACCEPTANCE_TRACEABILITY_A_B_C.md` —— `02_下一轮实施与验收.md` 全部验收条款逐条代码／回执／单测映射表。
3. `03_SOURCE_AND_RECEIPT_SCOPE.md`（配套 `04_SOURCE_AND_RECEIPT_MANIFEST.json`）—— 全部 **25 份完整 JSON 回执**（22 份冻结旧回执 + 3 份 `v11` 新回执）与 **23 个完整源码／脚本／测试文件**清单。

## 一键零依赖沙箱独立核验

```bash
python3 05_STANDALONE_VERIFY_V11.py --output verify_report.json
```

仅使用 Python 标准库即可独立验证：
- AST 提取 `_prior_state_fingerprint` 通过全部 6 类语义修改测试（`undetected_semantic_mutations == 0`）；
- 全部 25 份回执与 23 份源码 SHA256 完整一致，且 `v10` 13 份原始回执零字节改动；
- `v11` 三份回执满足 `model_calls == 0`、单题巩固无双重计费（`<= 1`）、68 条压力夹具隔离、暂定奥卡姆状态、`B1–B10` 全通过、三分区种子零重叠与评估器不可变；
- 随包全部 9 个运行时认知模块（含完整 `unified_research.py`）对 `lab._...` 私有属性零读取。
