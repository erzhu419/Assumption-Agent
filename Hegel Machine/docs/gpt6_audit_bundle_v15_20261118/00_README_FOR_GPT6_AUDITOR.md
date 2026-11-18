# Hegel Machine v15 GPT-6 独立审计自包含交付包（2026-11-18）

## 交付内容概览
1. `00_README_FOR_GPT6_AUDITOR.md`：本导航说明；
2. `01_EXECUTIVE_AUDIT_REPORT_V15.md`：100% 由逐题 JSON 自动渲染的 v15 主审计报告（含 V14 10 项勘误、三项真实移除控制、K/R 统一状态转移、`E1` 同历史对照、`E2` 端到端策略比较、成本剖分及 §10 停止规则结论）；
3. `02_CONTRACT_CLOSURE_MATRIX_V15.md`：`markdown/hegel_v15/02_下一轮最小施工与实验合同.md` 全部条款逐项闭环核查表；
4. `03_SOURCE_AND_39_RECEIPTS_INDEX.md`：39 份严格 JSON 回执（35 份历史零改动 + 4 份 v15 新增）及全部源码／探针索引；
5. `04_MANIFEST.json`：完整机器可读 SHA-256 与 Git Blob 清单；
6. `05_STANDALONE_VERIFY_V15.py`：零第三方依赖沙箱一键核验器（同时提供 `05_STANDALONE_VERIFY_V14.py` 兼容垫片供 `probe_scope_replay.py` 直接加载）；
7. `06_SHA256_SUMS.txt`：包内所有文件的逐文件 SHA-256 校验表；
8. `probes/`：GPT-6 V14 三探针 + GPT-6 V15 四探针（`probe_d3_object_removal.py`、`probe_scope_replay.py`、`compare_replay_receipts.py`、`recompute_v14.py`）；
9. `receipts/`：全部 39 份证据回执；
10. `bundled_sources/`：包含 `memory.py`、`branch_reconciliation.py`、`premise_commitment_index.py`、`unified_research.py` 在内的完整源码、测试与实验脚本。

## 沙箱一键复核命令
```bash
PYTHONPATH="" python3 05_STANDALONE_VERIFY_V15.py --output verify_v15_report.json
```
