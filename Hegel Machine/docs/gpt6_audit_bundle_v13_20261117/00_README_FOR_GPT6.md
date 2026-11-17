# Hegel Machine v13 GPT-6 审计交接包

## 阅读顺序
1. `01_V13_EVIDENCE_RECURSION_AND_ROUTING_DOSSIER.md` —— `v13` 主报告（含合同 §7 六项简明主结果、PR-A/B 控制与 PR-C 8 源链簇配对对比）。
2. `02_ACCEPTANCE_TRACEABILITY_V13.md` —— `N1–N6` 与 16 项行为验收（`#1–#16`）逐条追溯表。
3. `03_SOURCE_AND_RECEIPT_SCOPE.md`（配套 `04_SOURCE_AND_RECEIPT_MANIFEST.json`）—— 全量 **32 份 JSON 回执**（29 份零改动历史回执 + 3 份 `v13` 新回执）与 **23 个完整源码／脚本／测试文件**清单。

## 一键零依赖沙箱核验
```bash
python3 05_STANDALONE_VERIFY_V13.py --output verify_v13_report.json
```
