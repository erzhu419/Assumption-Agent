# Hegel M6.3 至 M6.7 审计交付索引

审阅日期：2026-11-22（Asia/Shanghai）。最新固定提交：`948e2d57a48b74369158ba2fbdee87bd32102403`。

## 阅读顺序

| 文件 | 内容 |
|---|---|
| `01_overall_diagnosis.md` | 核心判断、纠正分数、计划逻辑、实施断点、重点日志、A/B决策 |
| `02_plan_and_implementation_changes.md` | 可直接落实到现有模块的调整计划、数据契约、施工顺序及验收 |
| `03_harness_and_evidence_controls.md` | 提交门、候选归档、宿主状态、并发锁、缓存和审计范围 |
| `04_literature_and_reuse_notes.md` | LILO、DreamCoder、Skill∞、test-time compute及两篇2026年11月RSI论文的针对性比较 |
| `tracka/tracka_diagnosis.md` | M6.3修复的真实收益、AST/Pareto/投影/d-sep/ABI/对照残余问题 |
| `library_gate/library_gate_diagnosis.md` | 六固定算子、程序提取、调用遥测、宿主门控和工具契约 |
| `results/frontier_results_diagnosis.md` | 全版本分数、重评口径、成本、同轨迹反证及双轨上界 |
| `results/twenty_task_appendix.md` | 全20题、五个新完整版本、100条episode的诊断和固定源码链接 |
| `coverage_notes.md` / `coverage_inventory.json` | 真实覆盖、16个相关变更文件、源码哈希、边界 |

## 机器可读证据

- `results/task_pass_matrix_compact.csv`：20题原生通过矩阵，另列Bare声明重评。
- `results/episode_matrix.csv`：309条episode occurrence的分数、费用与来源；不是309个独立运行。
- `results/frontier_recomputed.json`：聚合、哈希/DAG检查、版本矩阵和来源。
- `results/oracle_pair_coverage.json`：已有候选集合的题级并集，不代表可部署分数。
- `results/nano_property_recount.json`：固定官方v0.1.0容差下的事后属性核算。
- 各 `*_controls_results.json`：离线反例、保存轨迹校验与原有纯测试结果。
- `SHA256SUMS.txt`：本交付包文件完整性；它不替代源码或科学结论验证。

完整原始日志通过逐题附录中的固定commit链接引用。本交付不重复打包整个Git仓库、全部历史原始轨迹、论文全文或官方隐藏truth文件。原始回执和源码没有被修改。

## 复现离线诊断

在拥有固定提交的完整checkout及项目Python依赖的环境中，将以下路径换成实际仓库根目录，执行所需脚本。它们仅写本审计目录的结果；无在线模型费用，不运行完整Docker评测。

```bash
export HEGEL_M67_REPO=/path/to/Assumption-Agent
python3 results/recompute_frontier.py
python3 results/trace_claim_controls.py
python3 tracka/tracka_controls.py
python3 library_gate/run_library_controls.py
python3 harness_controls.py
```

Track A隔离反例仅读取脚本自行创建的无敏感marker，说明Python审计钩子的边界；未发现原模型利用此路径。它不是新增benchmark成绩。相同地，并发锁控制证明一个潜在交错，不证明保存的失分运行发生该死锁。

Nano属性计数是对既有CSV的事后检查，源码和固定官方来源见结果报告；它不能作为下一轮生产选择器的真值输入。官方truth/test镜像未打包。

## 结论强度

报告分别标识已观察事实、可复现接口缺陷、潜在风险与待检验改进。未重跑付费在线研究、完整Docker基准或全部pytest；没有估计未测量的自然方差，没有承诺16–17/20，也没有把原题离线重放当作新泛化。

建议先修B′的完整能力继承、合法适配、验证和选择，再公平评估A的2×预算；继续保留跨时刻多尺度与跨域部分共性结构。
