# M4.2 独立审计复现实验包

固定源码提交：fb4b23aaf802c20b4e2d0018436abab73b0cca06。比较基线：99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c。

先读 M4_2_AUDIT.md，再按问题查 closure、pi、unified 三份专项报告。所有控制 JSON 都保留了正例、失败例与反方向结果。原仓库源码和提交证据只读；本包不是源码修复补丁。

## 已完成的执行

- 九个指定测试文件共 37 tests：退出码 0。
- 完整 M4.2 study：四种子、九臂、792 条统一回执，退出码 0。
- Part2 432 行与提交逐字节一致。
- Part4 792 行除了 13 条 selected_model_sha256，其余字段逐项一致；成本、召回、成功率、误差、完整曲线一致。完整差异保留在 replay/replay_comparison.json。
- 本轮对照均为本地离线有限控制，没有在线模型调用。

replay/evidence 是完整原始外部重放输出，包含其 SHA 清单要求的五个字节码缓存。不要先删缓存再声称原 SHA 清单仍全部通过。

## 重放提交本身

先将一个独立源码 checkout 固定到上述提交。以下变量自行填入绝对路径，不修改本包的原始结果：

```bash
export HEGEL_SOURCE='/path/to/Assumption-Agent/Hegel Machine'
export HEGEL_REPLAY_OUTPUT='/path/to/new-m42-replay'
cd "$HEGEL_SOURCE"
PYTHONPATH="$HEGEL_SOURCE/src" PYTHONHASHSEED=0 python3 scripts/run_v17_4_m4_2_unified_study.py --out-dir "$HEGEL_REPLAY_OUTPUT"
```

37 tests 对应文件清单记录在 audit_manifest.json。依赖按固定提交的项目配置安装。此次运行环境中的 anyio 插件有一条 pytest 弃用警告，不影响退出状态。

## 只复算账本敏感性

此操作无需重新进行科学实验：

```bash
python3 recalculate_cost_sensitivity.py \
  --evidence "$HEGEL_SOURCE/docs/evidence/v17_4_m4_2_unified_closure_and_pi_lifecycle_20261120" \
  --output /path/to/new-accounting-sensitivity.json
```

它保留原决策、原裁剪 Frozen、原 upfront，只重计 Pi 单价与空诊断条目；不是修复后的九臂主结果。输出应与 cost/accounting_sensitivity.json 的 JSON 值一致。

## 在新路径重放审计控制

原始控制脚本保留了本次运行的绝对路径，原 pickle 也含本次 artifact 引用。跨机器请在新的审计运行目录重建共享 builder，不要把 pickle 内旧绝对路径当成当前 artifact。

可以复制下面八个小脚本到新目录，保持相对结构：

- unified/capture_shared_build.py
- unified/probe_unified_controls.py
- unified/analyze_receipts.py
- pi/probe_projection_dependency.py
- pi/probe_dispatcher_scope.py
- closure/probe_m42_contract_controls.py
- closure/probe_fresh_budget_edges.py
- cost_controls.py

只在脚本副本中做两次路径替换：

1. 原源码根 /workspace/scratch/3f14ad2da0ed/repo_m42_fb4b23aaf/Hegel Machine → 新 HEGEL_SOURCE。
2. 原审计根 /workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf → 新审计运行目录。

设置 PYTHONPATH 为新源码根的 src。依次运行 capture_shared_build.py、probe_unified_controls.py、analyze_receipts.py；然后运行两个 pi probes、两个 closure probes 和根目录 cost_controls.py。共享 build 只执行一次，后续用其原始返回与真实 Stage0 快照，避免重复形成相同历史。

保留本包原脚本与 JSON；新结果写新目录。原样旧 probe 位于 closure/original_probes，其硬编码旧 commit 标签作为历史字节的一部分保留，本次实际导入来源见 closure/review_manifest.json。

## 目录说明

| 路径 | 内容 |
|---|---|
| M4_2_AUDIT.md | 完整综合诊断与验收建议 |
| replay/ | 测试日志、研究输出及逐字段比较 |
| cost/ | 五项实际执行价格／重发现对照与敏感性 |
| closure/ | 参数绑定、OR 分支、fresh 与原 probe |
| pi/ | 合法证书扩盒、最终查询、角色改名、Act3 和轨迹依赖 |
| unified/ | 真 Stage0、已付确认 trace、DEV、两分岔、维护差额及共享构建 |
| changed_paths.tsv、change_scope.json、core_patch.diff | 固定提交的差异范围 |
| audit_manifest.json | 来源身份、测试文件、完整性与原证据只读检查 |
| PACKAGE_SHA256SUMS.txt | 本审计包内容清单，不包括清单自身 |

代码与原始数据的来源均为用户所指固定提交；报告中的新数值是对应本地控制或明确标注的账本重计。页版报告与 Markdown 的主体内容相同；页版内容和结构会回读验证，未进行视觉预览。

