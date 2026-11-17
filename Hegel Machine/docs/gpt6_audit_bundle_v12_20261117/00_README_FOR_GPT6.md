# Hegel Machine v12 GPT-6 审计交接包

## 阅读顺序

1. `01_V12_AUTONOMOUS_LIFECYCLE_AND_UTILITY_DOSSIER.md` —— `v12` 科学定位校准、交付 A（确认契约／可观测范围／增量反统一／量纲与遥测修复＋`v11` 只读重分类）、交付 B（单一普通入口 `UnifiedResearchAgent.run()` 完整自主调和链＋删除 $H^*$ 行为对照）、交付 C（固定 `M0`、全组共享相同 Pass-0 标定 API 的五臂知识效用与两类反事实干预实验）主报告。
2. `02_ACCEPTANCE_TRACEABILITY_V12.md` —— `GPT_diagnosis_v12.md`（`F01–F11, E01–E02`）与 `02_下一轮最小施工与实验.md` 逐条源码与回执映射表。
3. `03_SOURCE_AND_RECEIPT_SCOPE.md`（配套 `04_SOURCE_AND_RECEIPT_MANIFEST.json`）—— 全部 **29 份完整 JSON 回执**（26 份零字节改动的历史回执 + 3 份 `v12` 新回执）与 **22 个完整源码／脚本／测试文件**清单。

## 一键零依赖沙箱独立核验

```bash
python3 05_STANDALONE_VERIFY_V12.py --output verify_v12_report.json
```

仅使用 Python 标准库即可独立验证：
- 直接从随包 `branch_reconciliation.py` AST 提取函数体并复跑 GPT-6 的 `source_excerpt_probes.py` 三项探针（`0 vs 5e-10` 非等价、`.22/.46/.34` 三分支增量反统一成功、无证据新根为 `unverified_missing_support`）；
- 复跑 `03_fingerprint_probe.py` 6 类语义修改检测与有界物理域反例检查；
- 校验全部 29 份回执与 22 份源码 SHA256 完整一致，且历史 `v10`（13 份）与 `v11` 回执零字节改动；
- 扫描随包运行时模块与 `v12` 实验脚本：零 `lab._...` 私有读取、零伪造探针名字符串、零 `if relation_condition ==` 隐藏条件分流、`model_calls == 0`。
