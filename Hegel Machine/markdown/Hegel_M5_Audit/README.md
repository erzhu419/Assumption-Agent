# Hegel M5 63b940a03 审计附件

审计提交：63b940a03205d0f5b36bf2403d274d74709a17fe。完整结论见 AUDIT_REPORT_ZH.md；机器可读结果保存在 results/，独立复算和最小控制在 controls/。

## 范围

本次审计没有修改项目源码，没有重新运行完整 640 条主基准或全部历史 51 项测试。完成六文件 SHA256、640 条回执复算，重跑新增包含 C01–C12 的一个 pytest 测试入口，并执行所附局部控制。数值能力和成本配对使用真实拟合；错误标签阈值单独使用固定预测桩隔离逻辑。结果和报告中的限制应一并保留。

## 环境

本次使用 Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0；单项 pytest 使用 9.2.0。原提交通过 hash((seed, arm)) 派生部分随机数，未指定 PYTHONHASHSEED，不能保证主基准 nfev 逐数重现。控制内部配对使用同一个显式 RNG。

## 准备

使用该提交的现有 checkout。以下路径由复现者替换，不要在原证据目录输出审计结果。完整 checkout 应已包含 v18 证据；稀疏 checkout 可用 git archive 或 git show 从同一提交只读取出，放在单独目录，再设置 HEGEL_AUDIT_EVIDENCE_DIR。

```bash
export HEGEL_AUDIT_REPO_ROOT=/path/to/Assumption-Agent
export HEGEL_AUDIT_EVIDENCE_DIR="$HEGEL_AUDIT_REPO_ROOT/Hegel Machine/docs/evidence/v18_0_m5_final_temporal_multiscale_and_base_subtree_20261120"
export PYTHONPATH="$HEGEL_AUDIT_REPO_ROOT/Hegel Machine/src"
export PYTHONDONTWRITEBYTECODE=1
export HEGEL_AUDIT_RUN_DIR=/path/to/unpacked-audit
```

## 只读账本复算

```bash
python3 "$HEGEL_AUDIT_RUN_DIR/controls/recompute_evidence.py" --repo-root "$HEGEL_AUDIT_REPO_ROOT" --evidence-dir "$HEGEL_AUDIT_EVIDENCE_DIR" --output-dir "$HEGEL_AUDIT_RUN_DIR/recomputed"
```

输出与 results/evidence_recomputed.json 比较。检查 6/6 SHA、640 唯一 arm/seed/task、空 summary_mismatches 与逐臂成本。

## 定向控制

在解压附件副本运行，避免覆盖保存的原结果。contracts/cold/legacy 三个脚本会在自身目录写新 JSON；legacy 还会在自身目录临时展开固定提交的 W4 实例与快照。它们不更改仓库或原始证据。

```bash
python3 "$HEGEL_AUDIT_RUN_DIR/controls/evidence_targeted_controls.py" --repo-root "$HEGEL_AUDIT_REPO_ROOT" --evidence-dir "$HEGEL_AUDIT_EVIDENCE_DIR" --output-dir "$HEGEL_AUDIT_RUN_DIR/recomputed"
python3 "$HEGEL_AUDIT_RUN_DIR/controls/temporal_runtime_controls.py" > "$HEGEL_AUDIT_RUN_DIR/recomputed/temporal_runtime_controls.json"
python3 "$HEGEL_AUDIT_RUN_DIR/controls/warm_pipeline_control.py" > "$HEGEL_AUDIT_RUN_DIR/recomputed/warm_pipeline_control.json"
python3 "$HEGEL_AUDIT_RUN_DIR/controls/contracts_controls.py"
python3 "$HEGEL_AUDIT_RUN_DIR/controls/cold_subtree_controls.py"
python3 "$HEGEL_AUDIT_RUN_DIR/controls/legacy_gray_control.py"
```

legacy_gray_control.py 通过 git show 读取固定提交中的少量旧证据，因而需要 checkout 的 Git 对象库保留本 SHA。cold_subtree_controls.py 直接读取本提交 post_j16 快照。两者只测试报告中所述的依赖/授权结论，不把 S02 解析遗漏项当成另一实例的真实误差。

## 新增原测试入口

```bash
cd "$HEGEL_AUDIT_REPO_ROOT/Hegel Machine"
python3 -m pytest -q tests/test_v18_0_m5_temporal_multiscale_and_base_subtree.py::test_m5_all_12_mathematical_and_redteam_controls
```

本轮该测试入口通过。另两个新增测试入口会重复运行完整或半量 benchmark，本轮未执行。

## 保存内容

附件不重复打包 Git 仓库及原提交大证据，依据固定 SHA 获取即可。results/ 保留本轮实际输出，其中 source_snapshot 可能记录审计机原始路径，属于运行来源记录；可移植脚本的路径已经由环境变量/参数指定。脚本完成可移植入口改写后做了语法检查，未用新运行覆盖原始观察结果。

SHA256SUMS.txt 覆盖本附件每个成员（校验文件自身除外）。REVIEW_MANIFEST.json 保存固定 SHA、27 项变更列表、审阅与执行范围、环境版本及未执行项目。

