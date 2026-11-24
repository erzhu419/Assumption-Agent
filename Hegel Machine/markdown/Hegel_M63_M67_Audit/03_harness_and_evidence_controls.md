# Frontier 运行器与证据链诊断

固定源码：`948e2d57a48b74369158ba2fbdee87bd32102403`。本文件补充 Track A、可执行库和结果三份专项，关注计划怎样落实成真实提交协议。所有控制均为本地、无付费模型、无 Docker 的最小反例；没有修改原仓库。

## H01 提交门是一次提示而不是候选验证事务

当前 `m6_frontier_real_benchmarks.py` 的提交流程使用 `candidate_has_verified`、`pullback_gate_triggered` 等 episode 级布尔值。第一次不满足条件的 submit 可被拦截；已经触发过门后再次 submit 可以结束。最后一轮又受预算条件限制，能够不经过该门结束。因此“强制通过机械验证才能提交”不是实际保证。

`harness_controls.py` 直接执行原 `run_frontier_episode`，只模拟模型、容器及官方评测外部调用，得到三个控制：

| 控制 | 实际结果 | 说明 |
|---|---|---|
| 连续两次 submit，中间没有验证 | 第二轮结束，`strict_pass=False`，DAG 仍合法 | 一次告警不是验证状态机 |
| 运行 `python3 -c "print('solver.py')"` 后 submit | 无门控即结束，`strict_pass=False` | 命令包含文件名不等于验证了候选 |
| `max_turns=1` 时直接 submit | 无门控即结束，`strict_pass=False` | 预算边界可跳过提示门 |

其中 `is_substantive_verification_command`（1503 行起）会接受上述 print 和无关的 `/tmp/test_unrelated.py`，却会拒绝名称不在关键词集合中的真实 `/tmp/validate_saved_candidate.py`。它识别的是命令字串，未绑定实际产物、输入、验证程序及结果。

另一方面，在同一 shell 命令中“改产物并立即验算”的正常操作，因文件变更分支与验证状态更新分支互斥，可能仍留下验证债。M6.7 symbolic 第 26 轮已经安装并做隔离验证，第 27 轮又被通用门拦住，第 28 轮重复检查。这里同时存在弱门可绕过和正常验证不被认可两个方向的错误。

**修改位置：**沿用 `run_frontier_episode`、现有候选快照和 DAG；增加候选级 `VerificationReceipt`，绑定完整产物哈希、生产程序及依赖哈希、公开输入版本、验证器哈希、执行结果、适用契约。宿主统一执行已登记验证器，并从执行结果生成回执，不由模型在 shell 命令中声称成功。提交成为 `commit(candidate_hash)`：候选无已接受回执时返回具体缺项，候选改动后旧回执失效；预算用尽时提交已经确认的 incumbent。已经对精确版本完成的检查不需要再消耗一个模型回合。

这里的“确认”只代表指定公开契约与验证集，不等于私有评测必过。探索工作区仍允许失败、暂时退步及高复杂度候选；保护的是提交版本和已保存能力。

## H02 候选档案保存了文件摘要，没有保存完整可执行能力

现有 `/tmp/.hegel_candidate_archive` 提供版本留存基础，但没有接成跨运行的完整档案；容器最终删除。终稿快照受文件后缀及 512 KB 上限限制，未普遍包含 `/tmp` 的 producer、C++ 源码和编译参数、NPZ/NPY 状态、大证书 JSON、训练配置与输入版本。

结果专项对 M6.3–M6.7 的复核：certificate 与 sparse-network 的 `saved_artifact_snapshots` 全部为空；Koopman 每版只保留 `results/report.json`；clinical/nano 主要保留最终 TSV/CSV。DAG 中保留了不少生成命令，可以恢复部分过程，但它们还不是冷启动后能直接挂载、重拟合和调用的能力对象。

**修改：**在删除容器前，把真实 producer、完整依赖、状态及公开重建配方写入现有 checkpoint/content-addressed storage。较大参数和证书存内容寻址引用，不能因超过文本快照上限而静默丢弃。终稿答案、可复现求解配方、可移植程序、跨域抽象分别标记；完整程序先保存，能做部分子树抽取时再建立衍生关系。

可复用 `code_agent.py` 中 `retain_evaluated_python_model`、`archive_confirmed_python_model` 的“已评估留存/确认后晋升”语义，以及 `checkpoint_storage.py` 的实际写入方法；Frontier 文件 ABI 与 typed Program 的差异通过 adapter 解决，不假定旧接口天然覆盖任意 Python/C++。

## H03 宿主门控未计入完整、可归因的成本

M6.7 新增门控在一个外层 Docker exec 中串行执行全部探针，最终才输出一个 JSON。wrapper 将不适用、无报告、非零退出、超时和解析失败都折叠成空列表，且调用方随即把 executed 置真。调用记录没有每个探针的 elapsed、stderr 或明确取消状态；精确定位热点和评估收益因此受阻。

不能从整题 4,792 秒及 437 个函数调用推导出函数每次耗时，也不能从一个空数组推导出 75 秒全部耗在 100k 叶子枚举。Python 客户端超时是否连同容器内工作进程终止，当前接口也未提供可核实回执。

**修改：**每个适用探针单独返回结构化结果：`not_applicable / completed_no_issue / diagnostic / failed / timed_out / cancelled`；绑定候选哈希并保存预算、输入摘要、实际耗时、错误和部分输出。确需重试时按原因和剩余预算决定，不能用一次 episode 布尔值永久封死。运行费用分别登记模型 token、工具 CPU/elapsed、锁等待、宿主 probe、候选形成、选择和官方 judge；保留并行重叠，不能把 episode wall 求和当整场 elapsed。

## H04 并发计时锁有可复现的嵌套读死锁交错

M6.5 起脚本中的 `_HeavyTimingRWLock` 允许 writer 自身再次读，但没有跟踪 reader 重入。一般 verifier 外层持有读锁，再进入被包装的 Docker exec 申请第二次读；若这之间已有 heavy writer 排队，第二次读会等待 writer，writer 又等待该线程仍持有的第一次读释放。

本地只执行原锁类，构造这一合法交错：`readers=1`、`waiting_writers=1`，reader 无法获得嵌套读，writer 也无法前进。控制脚本有受控退出，没有遗留进程。

**该缺陷是潜在执行风险，不是当前已完成失分轨迹的已证根因。** 本次未从保存日志证明某一题发生此死锁，不能拿它替所有慢运行解释。

**修改：**优先取消同一执行链的双重加锁，只由调度边界决定资源类别；如保留读写锁，则显式记录线程读深度、定义 writer 公平性，允许已有 reader 的重入。把锁等待和实际执行时间分别写回。AMR 候选与 oracle 必须在相同隔离条件下配对计时，并记录全部重复值及官方聚合，不能用最小值替代中位数。

## H05 缓存身份没有覆盖完整实验状态

Frontier 的部分 snapshot/cache 身份仅覆盖 nogood 与学习到的模块文件名，没有绑定所有代码内容、初始固定库、提示、输入数据、容器镜像、图结构及实际运行配置。Track A 也有候选数量而非所有候选内容的摘要路径。实验恢复后可能把不同版本当成同一上下文。

本次没有查出已发生的缓存混用；这里应修的是后续可重复性边界。

**修改：**由同一 manifest 生成 study identity：源码提交与工作树状态、容器 digest、任务数据版本、完整森林/技能内容哈希、提示/策略哈希、模型配置、随机性参数、预算、并发和资源隔离配置。恢复必须检查该身份；合法重评另建事件，不覆盖原 run。

## H06 完整性审计与科学结论审计需要分开

本次 Frontier 1,306 项 SHA、309 条 DAG occurrence 都通过内部检查；Track A 522 项 SHA、96 条 DAG 也通过。它们确实支持“读到的文件与清单匹配”“事件链结构未损坏”，不能证明 summary 所称的结果、调用收益或零信息流泄漏。

具体例子是复制的 Bare AMR：新 summary 写通过，原 official verifier 和原 DAG 保持超时失败。七项检查仍可全部通过。更正分数的正确方法是追加完整重评事件并让聚合器读取；不能只改一列布尔值和说明文字。

`audit_m6_4_executable_wormhole_bundle.py` 的静态无泄漏检查主要扫固定模块与巩固记录，没有完整扫新的 host runner；activation 校验比较同源总数字段；docker ps 未检查退出码也可误认为没有残留。它是有价值的工程检查集，应按真实范围命名。

需要增加四条有限而明确的语义检查：

1. 每个最终分数能追溯到官方 verifier 或合法重评事件，且绑定同一个最终候选。
2. 每个“改进”陈述包含改前/改后产物、共同评估条件、选择行为；无产物变化不得声称 gate 修复了产物。
3. 每个精确调用总数能从持久化完整事件重算，启动自测、失败调用和被最终候选采纳分别记账。
4. 每个自动继承成功声明对应可重放的 producer 与目标域资格回执，不能以导出文件数代替。

## 本地复现

在拥有指定提交和现有 Python 依赖的环境中：

```bash
HEGEL_M67_REPO=/path/to/Assumption-Agent python3 harness_controls.py
```

运行结果写入同目录 `harness_controls_results.json`。这不是新的性能测试，也不会更改 benchmark 分数。

## 源码定位

- [Frontier 运行器](https://github.com/erzhu419/Assumption-Agent/blob/948e2d57a48b74369158ba2fbdee87bd32102403/Hegel%20Machine/src/hegel_machine/cognition/m6_frontier_real_benchmarks.py)：验证识别 1503 行起；提交与门控约 2694 行起；文件变化与验证债约 2861 行起；调用摘要约 2990 行起。
- [宿主探针及执行 wrapper](https://github.com/erzhu419/Assumption-Agent/blob/948e2d57a48b74369158ba2fbdee87bd32102403/Hegel%20Machine/src/hegel_machine/cognition/m6_4_executable_wormhole_library.py)：宿主调用 1834 行起。
- [M6.5 并发和资源调度](https://github.com/erzhu419/Assumption-Agent/blob/948e2d57a48b74369158ba2fbdee87bd32102403/Hegel%20Machine/scripts/run_m6_5_frontier_28turn_study.py)。
- [M6.7 runner](https://github.com/erzhu419/Assumption-Agent/blob/948e2d57a48b74369158ba2fbdee87bd32102403/Hegel%20Machine/scripts/run_m6_7_host_gate_wormhole_28turn_study.py)。
- [七项审计器](https://github.com/erzhu419/Assumption-Agent/blob/948e2d57a48b74369158ba2fbdee87bd32102403/Hegel%20Machine/scripts/audit_m6_4_executable_wormhole_bundle.py)。

具体行号、离线控制及其实际输出共同构成证据；潜在风险与已观察失分已在各条分别标记。
