# M6.3–M6.7 本次审阅覆盖说明

日期：2026-11-22（Asia/Shanghai）。机器可读清单：`coverage_inventory.json`。

## 固定版本

| 角色 | 固定提交 |
|---|---|
| M6–M6.2 历史基线 | `b1d0c3e99f1e6b12cbc62ca8311f1a25d66f2dbf` |
| M6.3 | `0f6bd7d6f0df069cc32d7c517eb6d31d0be68019` |
| M6.7 最新审阅版本 | `948e2d57a48b74369158ba2fbdee87bd32102403` |

该清单分别记录 baseline→M6.3、M6.3→M6.7、baseline→M6.7 的 `git diff --name-status/--numstat/--stat` 结果。范围限定为相关源码、运行/审计脚本、测试和多精度计划目录。完整新增回执文件不逐项混入代码变更表，它们通过各证据包 manifest 与 SHA 清单单独核验。

## 相关变更文件（baseline→M6.7）

共 **16 个文件，7034 行新增、144 行删除**。下表是变更范围，不把它自动等同于每个文件均逐行全读；实际阅读与验证范围见后文。

| 变更 | 类型 | 文件 | diff 行数 |
|---|---|---|---|
| `A` | plan_or_report | `Hegel Machine/markdown/hegel_v16_5_multi_precision_plan/29_M6_3_Sovereign_Audit_Synthesis_and_Frontier_Breakthrough_Report.md` | +150 / −0 |
| `A` | plan_or_report | `Hegel Machine/markdown/hegel_v16_5_multi_precision_plan/30_M6_4_and_M6_5_Executable_Wormhole_Push_Pull_Spectrum_Report.md` | +81 / −0 |
| `A` | script | `Hegel Machine/scripts/audit_m6_3_sovereign_bundle.py` | +181 / −0 |
| `A` | script | `Hegel Machine/scripts/audit_m6_4_executable_wormhole_bundle.py` | +210 / −0 |
| `A` | script | `Hegel Machine/scripts/run_m6_3_frontier_28turn_study.py` | +223 / −0 |
| `A` | script | `Hegel Machine/scripts/run_m6_3_sovereign_ast_study.py` | +271 / −0 |
| `A` | script | `Hegel Machine/scripts/run_m6_4_frontier_28turn_study.py` | +271 / −0 |
| `A` | script | `Hegel Machine/scripts/run_m6_5_frontier_28turn_study.py` | +332 / −0 |
| `A` | script | `Hegel Machine/scripts/run_m6_6_unified_frontier_28turn_study.py` | +276 / −0 |
| `A` | script | `Hegel Machine/scripts/run_m6_7_host_gate_wormhole_28turn_study.py` | +293 / −0 |
| `M` | source | `Hegel Machine/src/hegel_machine/cognition/m5_live_gpt_and_provenance.py` | +87 / −34 |
| `A` | source | `Hegel Machine/src/hegel_machine/cognition/m6_3_sovereign_causal_wormhole_engine.py` | +1347 / −0 |
| `A` | source | `Hegel Machine/src/hegel_machine/cognition/m6_4_executable_wormhole_library.py` | +1861 / −0 |
| `M` | source | `Hegel Machine/src/hegel_machine/cognition/m6_frontier_real_benchmarks.py` | +832 / −110 |
| `A` | test | `Hegel Machine/tests/test_m6_3_sovereign_audit_controls.py` | +320 / −0 |
| `A` | test | `Hegel Machine/tests/test_m6_4_executable_wormhole_library.py` | +299 / −0 |

## 三个专项报告的实际覆盖

### Track A / M6.3

完整阅读 M6.3 引擎、M6.3 新测试和审计脚本；核对 shared Python runner 的新增 ABI/隔离/重试实现、Track A study 入口、M6.1 因果森林与提取/写回依赖、Bayes-ball/回溯投影 helper，读 Plan29 与两份用户上传总结，并参照上一轮已封存审计。复算 96 条新 episode，核验 **522/522 SHA、96/96 DAG**；实际执行五个现有非 Docker 测试函数。另做 AST、归属、Pareto、投影、d-sep 消融、ABI 和仅读取本审计生成 marker 的隔离反例。详见 `tracka/tracka_diagnosis.md` 及其控制脚本与 JSON。

### 可执行库 / 宿主门控

完整阅读 `m6_4_executable_wormhole_library.py`（1861 行）、对应审计器和测试；追踪 Frontier 检索、挂载、提示、门控、快照、遥测与结果写出 caller。读 M6.4/5/5a/6/7 的全部 TB episode summary、全部 skill harvest、Wave1 巩固事件；对 symbolic、wave、clinical、sparse-network、nano、certificate 的相关轨迹和产物定向定位。实际执行两个现有纯本地测试、十一组便宜控制；另外三个涉及 Docker/外部任务/并发锁的现有测试没有宣称全部重跑。详见 `library_gate/library_gate_diagnosis.md`。

### Frontier 结果 / 归因

遍历 M6.1–M6.7 保存的 Frontier 目录及 M6.5a 中断消融，复算 **309 个 episode occurrence、169 个不同 DAG 哈希**；309 条链有效，**1306 条 SHA 清单项**一致。M6.3–M6.7 五个完整新包自身含 956 条 SHA 项。复用 Bare 的 occurrence 不作为新独立样本；M6.5a 未提供单独 SHA/manifest。核对逐题 summary/verifier/DAG/token、关键时间序列与产物哈希；以官方 v0.1.0 固定版本的 nano 测试和属性表重算保存 CSV。详见 `results/frontier_results_diagnosis.md` 及复算输出。

## 重要依赖与历史复用

已读项目 `AGENTS.md`，从 reuse hub 和现有 historical registry 查询相关能力，继续沿当前 m6/m5/forest/index 路径追踪；这些查询不等于重新逐行审阅全部 181 个历史版本。当前修改并未改变原多精度计划的所有章节；新 Plan29、30 的承诺需与它们实际调用到的模块分开评价。本次以已封存上一轮审计为历史基线，不重复声称阅读每篇参考文献全文。

## 明确没有做的事情

- 没有新增付费模型调用或重跑 M6.3–M6.7 在线评测。
- 没有完整重跑 Docker 基准、官方 AMR/verifier 或全部 pytest 套件。
- 没有将局部反例、源码适配或保存产物的离线重算追加为新端到端通过分数。
- 没有据 SHA、DAG 完整性直接推断因果归因、零泄漏、统计独立性或稳定胜率。
- 没有声称已重新读完整个历史仓库及所有参考文献全文。

## 工作树与方案意见

核对时 checkout HEAD 为 `948e2d57a48b74369158ba2fbdee87bd32102403`，`git status --porcelain` 为空；本次诊断脚本、报告与结果写在仓库外的审计目录。

赞成主方案 **B′：先修完整程序保留、目标适配、可核验选择与恢复，再做 A 的 2× 总预算公平评估**。这不是退回“少调用几个函数”的 B。应保留复杂多尺度与部分共性结构，让候选及其依赖闭包在主执行路径可被验证消费；2×双轨的比较必须包含同 2×预算强普通控制，选择器用公开证据，公开代理的不确定性与择错率另记。现有证据不支持承诺固定 16–17/20，也不支持把第二轨本身称作已证的新哲学能力。
