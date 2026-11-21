# Hegel M6 与 v18 全面诊断资料包

固定源码：

- M6–M6.2：`b1d0c3e99f1e6b12cbc62ca8311f1a25d66f2dbf`
- v18.1–v18.4：`922b95e194d17c9259f1993706440e0602a43db9`
- 共同基线：`63b940a03205d0f5b36bf2403d274d74709a17fe`

本包只包含本轮原创审计、派生指标和离线诊断脚本；原始仓库、模型输出与论文不重复打包。原始证据均有固定提交链接。项目源码未修改，未新增付费模型调用；原分数没有被局部修复结果改写。

## 阅读顺序

| 入口 | 内容 |
|---|---|
| `overall_diagnosis.md` | 总判断、计划逻辑、关键问题、结果、研究借鉴和分阶段实施方案 |
| `v18_diagnosis.md` | v18.1–18.4，14项诊断，上轮F01–F11闭合核验，所有模型/机制族结果与现有模块复用 |
| `m6_runtime_diagnosis.md` | M6/M6.1 AST及运行器，12组发现，Luna九个失败与代码恢复控制 |
| `frontier_diagnosis.md` | 真实科学任务全版本比较，9组运行/机制问题，20题逐题事实、诊断和改进 |
| `literature_research.md` | 15篇原始论文的机制、局限、项目接口映射和实际阅读覆盖；官方benchmark版本/规则核验 |
| `v18_failure_appendix.md` | v18.4全部104个域内失分episode逐题索引与固定SHA源码/DAG链接 |
| `frontier_source_index.md` | Sol五组20题状态矩阵；全部160条保存记录及原summary/trajectory/verifier链接 |

`P0`表示下一轮可信结论前应先修的状态/执行/评测问题，不表示已观察到外部攻击或所有历史实验无效。事实、反事实控制、科学机制推断和未验证风险在报告中分开。

## 机器可读结果

- `v18_recomputed.json`、`v18_all_episode_metrics.csv`：2310个新live episode及102个deterministic链的复算。
- `v18_targeted_controls_results.json`：写回来源、权限、边覆盖、HEC、历史、ABI与符号域控制。
- `v18_replayed_runtime_errors.json`：60份原始错误程序的源码哈希、原错误摘要及本轮重放完整traceback；本轮15秒上限与原25秒不同。
- `m6_controls/controls_results.json`：AST新192+旧288 episodes、1510个SHA条目、七个原测试函数、归属/顺序/隔离/依赖/门控控制。
- `m6_controls/fit_contract_control.json`、`luna_c02_order_control.json`：仅复用已有付费代码的恢复结果。
- `frontier_recomputed.json`、`frontier_episode_index.json`：160个保存episode、140个不同DAG的账本和索引；不同哈希不等于统计独立。
- `frontier_trace_digest.json`：扫描用的派生动作摘要，保留原证据截断限制，不冒充完整未截断输出。
- `frontier_controls_results.json`：单文件快照扩展名反例、真实完成顺序反转、实际stress gate触发题。
- `m6_changed_paths.tsv`、`v18_changed_paths.tsv`：相对共同基线的变更文件清单。
- `audit_environment.json`：本轮Python/数值库环境与执行边界。
- `delivery_manifest.json`、`SHA256SUMS.txt`：本交付包内容身份。它们只验证内容一致性，不代替报告的语义核验。

## 离线复算

先准备两个固定提交的本地checkout，并确保相关 `Hegel Machine/src`、`tests` 和报告所列 evidence 包已完整下载。代码已有的数值依赖需能导入；本轮版本见 `audit_environment.json`。仓库不打包进本报告。

以下环境变量指向仓库根目录，不是其中的 `Hegel Machine` 子目录。脚本只读取项目；会在本审计目录更新派生结果或使用临时目录。没有模型请求，也不会启动真实 frontier Docker。

```bash
export HEGEL_V18_REPO='/absolute/path/to/v18_checkout'
export HEGEL_M6_REPO='/absolute/path/to/m6_checkout'
python3 v18_recompute.py
python3 v18_targeted_controls.py
python3 m6_controls/run_controls.py
python3 m6_controls/run_saved_model_rescues.py
python3 frontier_recompute.py
python3 frontier_controls.py
```

可选的既有错误程序回放及附件再生：

```bash
python3 v18_replay_runtime_errors.py
python3 v18_build_failure_appendix.py
```

`v18_replay_runtime_errors.py`执行已保存作者程序，使用原训练行和无标签预测输入；只接收本轮核对过的数值库导入，单个进程15秒诊断上限。它不等于强隔离沙箱，也不用于执行未来不受信任的新程序。对本轮60份固定源码重放的结果已在包内，无需为了阅读报告再次运行。

## 数据解释约定

1. 原版主榜、离线程序恢复、假设的未来方案，分开报告；不把本地修正后的程序当成新一次模型成功。
2. 通过数分母按各任务协议解释。v18主表216域内+24 OOS；AST每臂20域内+4 OOS；frontier20题中含15个TBScience和5个Bio任务。
3. M6.1和M6.2复用同一20条Bare28轨迹。复用不是额外独立样本；多版本并集不是单版本准确率。
4. token/effective token来自主agent保存账本；部分失败usage未知，Bio额外语义judge未完整入账；C是代理成本，不是实际美元。
5. benchmark未来版本不倒灌旧规则。AMR资源/证书cap核验固定到TBScience `v0.1.0`；Bio preview的冻结版本仍需项目补齐。
6. 文献报告明确逐篇阅读范围，未宣称穷尽所有近期论文、完整审计所有论文源码或重跑其模型实验。

## 下一轮验收建议

优先运行已经触发真实损失的最小反例：fit闭包与fitted状态、owner顺序、C02多候选、numeric permission拒绝、HEC停止、历史中段扰动、单文件快照、无关exit-0验证债务。它们修复后，再用冻结新任务和明确对照检验结构组织、跨题迁移与改进器生产率。

复杂跨时刻状态、多尺度折叠、部分共享子树和开放分支应保留；修复目标是使它们真正参与后继执行，并把实际收益与其他因素分开。
