# Hegel Machine v10.2–v10.4 GPT-6 统一审计交接包

## 推荐阅读顺序

1. `01_UNIFIED_AUDIT_DOSSIER_v10_2_to_v10_4.md` —— `v10.2` 审计全量闭合、`v10.3` 规模索引与高元数传感器标定、`v10.4` 四层奥卡姆极小性门控及开放边界总览。
2. `02_F01_TO_F20_AND_22_ACCEPTANCE_MATRIX.md` —— `GPT_diagnosis_v10.2.md` 全部 **20 项发现（`F01–F20`）** 与 `02_下一施工合同.md` **22 项行为验收（`#1–#22`）** 逐条代码／回执／测试追溯表。
3. `03_SCALING_HIGH_ARITY_AND_OCCAM_REPORT.md` —— 由回执 JSON 自动生成的 $M \in \{4, 12, 36, 72\}$、四臂同工具奥卡姆消融、逆序对抗测试与 $M=72 \times N=8$ 全栈实测表。
4. `04_SOURCE_AND_RECEIPT_SCOPE.md`（配套机器可读 `04_SOURCE_AND_RECEIPT_MANIFEST.json`）—— 全部 **22 份完整 JSON 回执**（含实验角色与独立物理源计数）与 **22 个随包源码／脚本／测试文件**的 SHA256 与 Git Blob SHA1 清单。

---

## 在审阅端沙箱中一键运行独立核验（零第三方依赖）

本 ZIP 包内已完整附带 `receipts/`（全部 22 份原始与新增 JSON 回执）和 `bundled_sources/`（全部 22 个核心模块、实验脚本与验收测试完整未截断源码）。解压后可直接运行：

```bash
python3 05_STANDALONE_VERIFY_V10_4.py --output verify_report.json
```

该脚本仅依赖 Python 标准库（`fractions`, `hashlib`, `json`, `re`, `pathlib`），零网络、零付费调用，自动核验：
1. 包内全部 Markdown、JSON 回执与源码文件的 `SHA256SUMS.json` 完整性；
2. `v10_world_structure_20261116/` 全部 13 份历史回执相对冻结账本零字节改动；
3. GPT-6 盒域反例在 `physical_legality`、`analysis_region`、`unbounded` 三种统一语义下均满足 `both_opposite_certificates_accepted == False`；
4. `v10.2`、`v10.3`、`v10.4` 全部回执满足 `model_calls == 0` 及关键定量不变量；
5. 随包运行时认知模块静态扫描确认对 `lab._...` 私有属性零读取、API Key 零残留。

本轮构建时的标准核验输出已保存在 `sandbox_verify_stdout.json` 供逐字段比对。若在装有 `numpy`/`scipy`/`sympy` 的完整仓库环境中，亦可附加 `--repo "/path/to/Hegel Machine"` 直接调用仓库内核实时复核。
