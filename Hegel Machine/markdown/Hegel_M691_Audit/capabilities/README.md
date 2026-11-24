# 能力专项复现说明

被审提交：`4001bef56ffa77520c248b853e7fed3535ee9fe4`。

主报告：`capabilities_diagnosis.md`。

## 环境与路径

测试环境为 Python 3.12；需要该仓库运行这些 cognition 模块的依赖，包括 NumPy、SciPy、scikit-learn。`recompute_capabilities.py` 只使用 Python 标准库。无需 Docker、API key、模型服务或网络。

所有控制支持 `HEGEL_AUDIT_REPO`。该变量既可以指向仓库根（其中有 `Hegel Machine`），也可以直接指向 `Hegel Machine` 目录。默认值是本次审计旁的 `repo_m691_4001bef56`。旧 library/Track A 控制兼容 `HEGEL_M67_REPO`，但新变量优先。

将路径替换成自己的固定提交 checkout：

```bash
export HEGEL_AUDIT_REPO='/absolute/path/to/Assumption-Agent'
python recompute_capabilities.py
python qualification_control.py
python library_regression_controls.py
python tracka_regression_controls.py
python structural_controls.py
```

直接指向 Hegel 目录也可：

```bash
export HEGEL_AUDIT_REPO='/absolute/path/to/Assumption-Agent/Hegel Machine'
python recompute_capabilities.py
```

脚本路径相对本目录。输出 JSON 与脚本同目录；临时示例写入系统临时目录后自动清理，不修改仓库源码。Python 导入可能生成正常 `__pycache__`；需要避免时可设置 `PYTHONDONTWRITEBYTECODE=1`。

## 必需 fixtures

1. 本轮实证重算使用提交内的：
   `Hegel Machine/docs/evidence/m6_9_1_a_two_branch_frontier_40ep_20261123/`。
   可通过 `HEGEL_AUDIT_BUNDLE` 改 evidence 目录名，但针对不同 bundle 的数字不应沿用本报告。
2. Track A 旧反例复核使用同一最新提交内保留的：
   `Hegel Machine/docs/evidence/m6_3_sovereign_ast_causal_96ep_20261121/all_episode_receipts.jsonl`。
   它调用当前实现及当前 `build_m6_causal_24_waves(seed=20261121)`，用旧保存 receipt 检查当前行为；**无须 checkout 旧提交，也没有重做旧模型生成**。
3. 其余三组控制使用当前源代码及脚本生成的公开合成数组/程序，无真实任务隐藏答案依赖。

## 已保存结果与解释范围

| 脚本 | 输出 | 关键观察 |
|---|---|---|
| `recompute_capabilities.py` | `empirical_capabilities_recomputed.json` | 4,141 次 completed/non-smoke 调用、6 个任务；15 个资格检查共 64 个模块、0 个 qualified；30 个分支 manifest 构建命令全为空；W1 TB 来源 Thesis 4 / Anti 4；4 个缺失不同 blob 引用 |
| `qualification_control.py` | `qualification_control.json` | 本地执行原生成 shell 命令即可复现 SyntaxError；只导入 `math` |
| `library_regression_controls.py` | `library_regression_controls.json` | 旧闭包/NaN/minimax/LOCO/Fourier/短程记录等改善；LOPO 公共偏差仍是适用范围限制 |
| `tracka_regression_controls.py` | `tracka_controls_results.json` | 真实旧成功程序不再丢、AST 区分、owner、ABI、旧公式敏感性；实际旧样本的 d-separation 消融 archive 不变 |
| `structural_controls.py` | `structural_controls.json` | 共享代理对源规律变化不敏感；缺失切片仍可提升；显式 cutoff 被覆盖；np.load 未被排除；旧投影均值抵消；symbolic term schema 错配；两个 AST 上下文边界 |

`empirical_capabilities.json` 是最初读取结果；新重算结果使用更详细的缺失引用 dict 格式，关键计数一致。`structural_controls_stdout.txt` 保存较早完整运行的 stdout，规范结果使用 `.json`。

这些控制故意记录错误而不统一断言成功。脚本正常退出意味着成功观测被审行为，并不意味着所测产品合同全部通过。历史控制里若干 key 沿用旧缺陷名，例如 `consensus_not_minimax`；应读数值与当前报告，不能仅凭 key 名判断当前版本仍有旧错。

`audit_hook_isolation_benign_marker` 只尝试读取脚本刚创建的数字 marker，不读取任何用户文件；它验证旧已知入口已拒绝，并不证明完整 OS 隔离。资格控制不启动 Docker，而是用同一 shell payload 在本地复现 quoting 行为。结构控制没有将潜在边界归因为本轮实际失分。

## 固定源码与历史复用查询

本次已按仓库 AGENTS 要求检查 reuse hub、历史注册表与实际原实现。查询输出在 `historical_reuse_queries.json`，包括 `Pi_fold`、`subtree`、`reconciliation`、`retention`、`retained`。精确关键词未命中不意味着源码中不存在相应算法；报告明确区分缺算法、缺桥接、缺执行采纳证据。

本专项没有重新执行在线基准，也没有修改仓库源码或提交 PR。报告中的 GitHub 链接固定到上述 commit，不使用会漂移的分支头。
