# Hegel M3.1 与 M4 审计包

主报告：`M4_AUDIT.md`。固定提交：`11961f702739ab272661ff59b77fb5bc8e3589b8`。比较基线：`612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8`。日期：2026-11-19（Asia/Shanghai）。

## 内容

- `M4_AUDIT.md`：完整结论、复现结果、代码问题、证据边界和最小 M4.1 建议。
- `m31/M31_AUDIT.md`、`author/AUTHOR_AUDIT.md`、`multiscale/MULTISCALE_LIFECYCLE_AUDIT.md`：三个专项的精确源码行号、正反控制、局限及命令。
- `replay_m31/`、`replay_m4/`：用最终提交运行原研究入口生成的完整静态输出，未覆盖仓库内原证据。
- `m31_replay_comparison.json`、`m4_replay_comparison.json`：与提交内结果的逐项比较；分别说明新增零尺度边计数字段和一处 0.001 算力均值差异。
- `original_ledger_audit.json`：原始逐题和源账本重算；包括摘要 SHA 检查及成本口径。
- `m31/`、`author/probe_run/`、`multiscale/probe_workspaces/`：所有定点控制、工作区、模型和观测。
- `audited_source/`：23 个重点源码、脚本、测试、文档与依赖快照，方便核对；不是完整可安装仓库。
- `changed_paths.tsv`、`tree_11961f702_inventory.json`、`focused_patch.diff`：范围和差异；文件清单不代表逐字重读所有历史文件。
- `run_environment.json`、`focused_tests.log`：环境及 28 项回归成功记录。
- `USER_HANDOFF.md`：本轮用户交接书副本。
- `SHA256SUMS`：本包文件校验清单。

## 完整重放

先取得给定仓库并检出固定提交，按项目要求准备 Python 依赖。不要直接运行原脚本的默认入口覆盖历史证据；使用本包的输出重定向 launcher。

```bash
export HEGEL_REPO='/absolute/path/Assumption-Agent/Hegel Machine'
export HEGEL_AUDIT_OUT='/absolute/path/new-audit-output'
export AUDIT_BUNDLE='/absolute/path/Hegel_M4_Audit_11961f702_20261119'
export PYTHONPATH="$HEGEL_REPO/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=0
python3 "$AUDIT_BUNDLE/replay_study.py" m31
python3 "$AUDIT_BUNDLE/replay_study.py" m4
```

审计运行中 M3.1 约 481.88 秒、M4 约 147.8 秒，存在其他本地并行工作；这些用时不用于声称速度优势。重放使用原入口，只改变证据输出目录，没有改变函数、任务、臂、门控、seed 或费用。

## 小控制

作者脚本显式支持仓库与输出目录：

```bash
python3 "$AUDIT_BUNDLE/author/probe_author.py" --repo "$HEGEL_REPO" --out "$HEGEL_AUDIT_OUT/author-controls"
```

M3.1 和多尺度脚本将输出写到脚本所在目录。为保留原审计证据，先复制这些脚本到一个新的目录，再以固定提交的 PYTHONPATH 运行：

```bash
mkdir -p "$HEGEL_AUDIT_OUT/m31-controls" "$HEGEL_AUDIT_OUT/multiscale-controls"
cp "$AUDIT_BUNDLE/m31/probe_closure_original.py" "$AUDIT_BUNDLE/m31/probe_boundaries.py" "$AUDIT_BUNDLE/m31/probe_factorization.py" "$HEGEL_AUDIT_OUT/m31-controls/"
cp "$AUDIT_BUNDLE/multiscale/probe_multiscale_lifecycle.py" "$HEGEL_AUDIT_OUT/multiscale-controls/"
python3 "$HEGEL_AUDIT_OUT/m31-controls/probe_closure_original.py"
python3 "$HEGEL_AUDIT_OUT/m31-controls/probe_boundaries.py"
python3 "$HEGEL_AUDIT_OUT/m31-controls/probe_factorization.py"
python3 "$HEGEL_AUDIT_OUT/multiscale-controls/probe_multiscale_lifecycle.py"
```

`probe_closure_original.py` 为旧审计脚本字节不变副本。其结果中写死的旧 commit 字符串不是运行时 provenance；本轮运行代码是新提交。原样脚本 SHA 为 `2a0a787c742592ada1b590ebeb47cbb78fbfb1020f3d56d2ac33d32a6884186c`。其余新控制记录当前固定提交。

## 回归

在 `Hegel Machine` 下，以同一 PYTHONPATH 和 pytest 运行：

```bash
python3 -m pytest -q \
  tests/test_v16_6_m0_contracts_and_wormholes.py \
  tests/test_v16_6_m1_cross_domain_wormhole_study.py \
  tests/test_v16_8_m2_open_workspace_wormholes.py \
  tests/test_v16_9_m2_audit_closure_and_open_revision.py \
  tests/test_v17_0_m3_hipporag_wormhole_scaling.py \
  tests/test_v17_1_m3_1_audit_closure_and_walk.py \
  tests/test_v17_2_m4_multiscale_and_open_discovery.py
```

本轮 28 项全部通过，仅有 pytest 插件兼容性弃用警告。没有因此继续扩跑无关测试。

## 证据解释

新增反事实是对边界或因果归因的诊断，不混入原预登记主目标。数值确认成功、权限合法、方法程序被消费、全生命周期成本有净收益属于不同验收对象；主报告逐项区分。

历史和参考材料的覆盖边界见主报告第 2 节。当前工作树保持 clean，没有修改源码、远程写入或调用在线付费模型。原 M2/M3 历史证据目录差异为空。

网页报告另行以内容读回核对标题、正文和链接；未在浏览器做视觉排版预览。ZIP 内 Markdown、JSON、代码和日志是可下载的完整交付。
