# Hegel V17.3 M4.1 独立审计包

固定提交：99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c。基线：11961f702739ab272661ff59b77fb5bc8e3589b8。审计日期：2026-11-20。

## 先读什么

- `M4_1_AUDIT.md`：完整中文诊断、数值、边界、优先级和 M5 建议。
- `closure/CLOSURE_AUDIT.md`：工程闭环、真实契约消费和 artifact 身份。
- `pi/PI_AUDIT.md`：快慢动力学、区域盒、匹配购测控制。
- `unified/UNIFIED_AUDIT.md`：作者入林、静态排序、已付前缀、R3 和指标。
- `replay/comparison_to_committed.json`：完整研究与提交逐字段比较。
- `part2/deduplicated_results.json`：去重后的四格；`all_496_records.json` 保留原全部调用。
- `ledger_and_metric_review.json`：336 条记录与摊销重算。

审计未修改源码和原证据，没有使用在线商业模型。33 项指定回归通过；四份分项 JSON 和 336 条目标记录与提交一致，只有 summary 的总墙钟时间不同。新控制的正面、负面结果都保留。

## 复现环境

使用固定提交的独立 checkout，按该仓库自身依赖安装 Python 环境；本次为 Python 3.12，使用已有 NumPy、SciPy 和 pytest。`environment.json` 记录实际版本。目录名中的空格需要 shell 引号。

下面以解压后审计目录为当前目录。将路径替换成自己的真实位置；输出必须放在原仓库和本审计包之外的新目录。

```bash
export HEGEL_AUDIT_REPO='/path/to/Assumption-Agent/Hegel Machine'
export HEGEL_AUDIT_OUT='/path/to/new-audit-output'
mkdir -p "$HEGEL_AUDIT_OUT"
```

### 完整研究

```bash
PYTHONHASHSEED=0 python3 replay_full_study.py \
  --repo "$HEGEL_AUDIT_REPO" \
  --out "$HEGEL_AUDIT_OUT/full-study"
```

该 wrapper 只改变输出目录；它检查提交且拒绝向 checkout 内部或非空输出目录重写。直接运行仓库默认 study 入口会覆盖其证据目录，不属于只读复现。

### 原有 33 项回归

```bash
PYTHONPATH="$HEGEL_AUDIT_REPO/src" PYTHONHASHSEED=0 python3 -m pytest \
  "$HEGEL_AUDIT_REPO/tests/test_v16_6_m0_contracts_and_wormholes.py" \
  "$HEGEL_AUDIT_REPO/tests/test_v16_6_m1_cross_domain_wormhole_study.py" \
  "$HEGEL_AUDIT_REPO/tests/test_v16_8_m2_open_workspace_wormholes.py" \
  "$HEGEL_AUDIT_REPO/tests/test_v16_9_m2_audit_closure_and_open_revision.py" \
  "$HEGEL_AUDIT_REPO/tests/test_v17_0_m3_hipporag_wormhole_scaling.py" \
  "$HEGEL_AUDIT_REPO/tests/test_v17_1_m3_1_audit_closure_and_walk.py" \
  "$HEGEL_AUDIT_REPO/tests/test_v17_2_m4_multiscale_and_open_discovery.py" \
  "$HEGEL_AUDIT_REPO/tests/test_v17_3_m4_1_unified_chain_and_audit_closure.py" -q
```

### 四格去重

```bash
PYTHONHASHSEED=0 python3 replay_part2_deduplicate.py \
  --repo "$HEGEL_AUDIT_REPO" \
  --out "$HEGEL_AUDIT_OUT/part2"
```

通过 Python profiling 在原 Part2 函数返回时保存局部逐题列表，不修改原函数。按 cluster_id/target 去重，并逐项核对重复记录是否完全相同。

### 动力学和区域盒控制

```bash
PYTHONHASHSEED=0 python3 pi/probe_pi_paths.py \
  --repo "$HEGEL_AUDIT_REPO" \
  --out "$HEGEL_AUDIT_OUT/pi"
```

该脚本在原函数第一次调用前捕获 solver 闭包；仪表记录全部购测、fit 和授权。购测策略控制及越界控制分别标记。所有更改只在本进程内，最后恢复被包装的方法。

### 工程闭环和统一链控制

这两份脚本在自身所在目录写输出。先复制到新的独立目录，避免改写包内证据：

```bash
mkdir -p "$HEGEL_AUDIT_OUT/closure" "$HEGEL_AUDIT_OUT/unified"
cp closure/probe_current_semantics.py closure/probe_relative_escape.py "$HEGEL_AUDIT_OUT/closure/"
cp unified/probe_unified.py "$HEGEL_AUDIT_OUT/unified/"
PYTHONPATH="$HEGEL_AUDIT_REPO/src" PYTHONHASHSEED=0 python3 "$HEGEL_AUDIT_OUT/closure/probe_current_semantics.py"
PYTHONPATH="$HEGEL_AUDIT_REPO/src" PYTHONHASHSEED=0 python3 "$HEGEL_AUDIT_OUT/closure/probe_relative_escape.py"
PYTHONPATH="$HEGEL_AUDIT_REPO/src" PYTHONHASHSEED=0 python3 "$HEGEL_AUDIT_OUT/unified/probe_unified.py"
```

`HEGEL_AUDIT_REPO` 仅选择代码和已提交 artifact 所在目录。relative-path 控制只生成并执行自己的常数模型，结果依赖默认临时目录布局，不读取任何其他用户数据。统一链控制中的策略替换只保留已付确认并排除重复确认，不向源码写补丁。

## 原始探针的口径

`closure/original_probes/` 保存上一轮四个探针的逐字节副本及此次输出。probe 内写死的旧 commit 标签仍保留；实际执行的新 commit、导入目录和 clean 状态见 `closure/replay_execution_manifest.json`。旧 author probe 内嵌旧 loader 表达式，所以它的 metadata-only 布尔值不能证明新 loader 有同一个漏洞；新 loader 结论来自 current_semantics 控制。

本轮只给部分新审计 harness 增加了 `--repo/--out` 或 `HEGEL_AUDIT_REPO` 以便异地重放，不改变默认控制逻辑；这些配置修改没有触及仓库源码。

## 限定与保存状态

- 静态排序控制为主种子 12 题，是审计阶段的敏感性分析，没有把它称为预注册盲测。
- 已付前缀控制为真实 SameHistory 的一个目标，不冒充新 48 题均值。
- 区域盒外控制不反驳原盒内结果；它核验越界接口承诺。
- 累计回本点是明示假设下的代理费用推算，不是独立长期学习实验。
- 未重新逐字阅读全部历史文件或无法获得全文的文献；范围盘点、直接源码和历史复用位置见主报告。
- 报告页面按原生 Markdown 标题、表格和链接生成，并回读核对内容；未做浏览器视觉预览。ZIP 内 Markdown 为完整可复查版本。

`MANIFEST_SHA256.txt` 覆盖审计包中的其他全部交付文件。仓库本身不包含在此包中；按固定提交重新取得即可。
