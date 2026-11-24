# 黑格尔机 M6.9.1 独立核查包

固定提交：`4001bef56ffa77520c248b853e7fed3535ee9fe4`。核查日期：2026 年 11 月 23 日。

**核心判断：原生合题 18/20、反题分支 17/20 成立；独立端到端 1x、完整预算匹配、全部旧缺陷闭环和真实自动抽象迁移等更强结论仍有具体缺口。** 本包提供完整诊断、实施调整、二十题复盘和离线复算控制。

## 阅读顺序

1. `01_overall_diagnosis.md`：总判断、成立的成绩、优先缺陷和哲学树实现差距。
2. `02_adjustment_plan.md`：沿现有模块修复的施工表、验收合同、真实多尺度和共同结构接线、实验安排。
3. `results/twenty_task_diagnosis.md`：全部二十题的正反合成绩、实际过程、失败原因与后继建议。
4. `results/results_diagnosis.md`：各版本分数、成本、时间顺序、哈希和便携 ZIP 复核。
5. `capabilities/capabilities_diagnosis.md`：Track A、可执行技能、生产闭包、多尺度与共同子树专项。
6. `synthesis/synthesis_and_harness_diagnosis.md`：公共选择器、任务接口、验证事务和缓存专项。
7. `03_audit_boundaries_and_reproduction.md`：审计器范围控制、证据强度与运行说明。
8. `04_literature_and_reuse.md`：直接相关论文与已存在能力的复用位置。
9. `findings_ledger.json` 与 `coverage_notes.md`：可供下一轮逐项跟踪的缺陷清单和本次覆盖范围。

所有 GitHub 来源固定到被审提交，避免分支继续前进后无法对应。MD 内的本地相对路径指本包文件，完整程序与原始回执仍在固定提交仓库中。

## 使用脚本

将完整固定提交下载到本机，令 `HEGEL_AUDIT_REPO` 指向其 `Hegel Machine` 目录。详见 `03_audit_boundaries_and_reproduction.md` 和各专项 README。结果复算脚本不导入项目代码；其他源代码控制需要项目依赖。

本包没有重新运行在线模型、没有执行官方科学评测，也没有修改项目源码。临时控制中出现的非法候选、修改后的日志标记和离线 Docker 桩，均为范围检查，不能当作实际模型作弊证据。原始 18/20 不因这些边界漏洞被追减。

## 文件完整性

`PACKAGE_SHA256SUMS.txt` 校验本诊断包自身文件；它与被审项目的 SHA256SUMS 不同。大型中间轨迹索引和原仓库 ZIP 镜像未重复放入，必要诊断、脚本、结果和逐题证据链接均已保留。
