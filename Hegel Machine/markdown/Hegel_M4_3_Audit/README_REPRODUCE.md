# V17.5 M4.3 独立审计复现包

源码必须固定于 `3461535364fcdb4e9ae9c87ccba502b7e6b56fe8`。本包不修改源仓库，不调用模型 API。主报告是 `M4_3_AUDIT.md`，三个专项报告位于 `instance/`、`scale/`、`evaluation/`。

## 环境与路径

本次审计环境：Python 3.12.14，NumPy 2.3.5，SciPy 1.17.0，threadpoolctl 3.6.0，cryptography 50.0.0，pytest 9.2.0。详细环境见 `environment.json`。pytest 的一个插件弃用警告不影响 40 项通过；日志保留原警告。

建议在已有仓库创建独立固定提交工作树，或使用同 SHA 的干净完整 checkout。需要源码、tests、提交内 M4.2/M4.3 evidence、`markdown/Hegel_M4_2_Audit`，以及这些入口引用的旧证据。不要把复现输出写回原仓库。

以下示例从解压后的审计目录运行。`HEGEL_AUDIT_REPO` 统一指向 checkout 内的 `Hegel Machine` 子目录，`HEGEL_AUDIT_OUT` 使用一个新的输出目录。

```bash
export HEGEL_AUDIT_REPO='/path/to/Assumption-Agent/Hegel Machine'
export HEGEL_AUDIT_OUT='/path/to/new-audit-output'
export PYTHONPATH="$HEGEL_AUDIT_REPO/src"
export PYTHONDONTWRITEBYTECODE=1
```

源项目依赖以其 `pyproject.toml` 为准；若要逐字数值重现，先安装上述固定版本。这里只使用本地数学模拟和子进程，不需要服务密钥。

## 完整官方研究

```bash
python3 "$HEGEL_AUDIT_REPO/scripts/run_v17_5_m4_3_unified_study.py" \
  --out-dir "$HEGEL_AUDIT_OUT/replay/evidence"
```

这会重建一次闭合验证森林及一次主研究森林，运行 C1–C6、四 seed 的 11 臂主表、附加噪声对照，以及临时目录删除后的冷恢复。执行时长取决于机器，本次实际时长见新 evidence 的 `study_manifest.json`，不要把墙钟与 CostMeter 的 K 混同。

运行相关 10 个测试文件、共 40 项（固定本提交的通配符集合）：

```bash
cd "$HEGEL_AUDIT_REPO"
python3 -m pytest tests/test_v16_*.py tests/test_v17_*.py
```

本包保留本次分开的 37 项历史回归与 3 项新增测试日志，分别为 `replay/prior_37_tests.log` 和 `replay/m43_tests.log`。这不是对全仓库所有测试文件的执行声明。

## 有限诊断控制

回到解压后的审计目录执行：

```bash
python3 instance/probe_same_ast_instance_swap.py
python3 instance/probe_instance_freshness.py
python3 scale/run_scale_controls.py
python3 scale/verify_archive_identity.py
python3 evaluation/analyze_evaluation.py
python3 evaluation/replay_fork_quality.py
python3 probe_cost_events.py \
  --repo "$HEGEL_AUDIT_REPO" \
  --out "$HEGEL_AUDIT_OUT/cost_events.json"
```

前六项支持上面的环境变量。instance/scale 也接受仓库根目录并自动补 `Hegel Machine`；为避免不同脚本产生歧义，推荐统一使用子目录。fork quality 逐行 append 留存进度，重复执行请换新的输出目录，不能把两次追加记录混算为一轮。

这些控制复用提交中的可迁移森林，没有再次运行主研究 builder。其中实例控制会用原生 host 校准函数生成第二个真实合法同 AST 实例，并只替换审计目录副本。scale 控制会在内存副本上测试证书/角色关系，保留正常 T21 和 quench 正对照。

费用观察仅包裹原函数记录调用，不改变算法选择。其返回的 `fallback_pool_specs=[]` 只用于四个正常召回即成功的目标；记录均成功召回，不把该小控制当作主表 fallback 比较。

## 重跑证据比对

```bash
python3 verify_replay.py \
  --committed "$HEGEL_AUDIT_REPO/docs/evidence/v17_5_m4_3_channel_bound_lifecycle_20261120" \
  --replay "$HEGEL_AUDIT_OUT/replay/evidence" \
  --out "$HEGEL_AUDIT_OUT/replay_verification.json"
```

该脚本检查原始和重跑 SHA256 清单、文件集合、字节差异、JSON 逐字段差异、1,144 条回执唯一性、账本算术、11 臂/回本/噪声摘要是否相同。它保留数值差异，没有用宽松容差或静默忽略参数/摘要来制造“完全一致”。运行时间、临时路径及平台相关源码摘要的差异需要据实际字段解释。

## 证据索引

| 路径 | 内容 |
|---|---|
| `M4_3_AUDIT.md` | 主结论、修复矩阵、两组 P1、局部问题、结果与下一步 |
| `instance/INSTANCE_FRESHNESS_AUDIT_ZH.md` | 实例身份、旧入口、OR、步骤次序、已付观测 |
| `scale/SCALE_DISPATCH_AUDIT_ZH.md` | 角色/作用域、证书、fine 矩阵、摘要与冷载 |
| `evaluation/EVALUATION_AUDIT.md` | 11 臂、维护分解、噪声、Compiled/PPR、动态索引 |
| `instance/probe_same_ast_instance_swap.json` | P1 实例错绑三组控制及节点字段 inventory |
| `scale/scale_controls.json` | P1 轴映射、fresh 盒、正常目标及 R3 验证 |
| `cost_events.json` | 正常 T01/T21/T23/T24 的实际函数调用记录 |
| `evaluation/evaluation_analysis.json` | 全回执统计、通道和版本控制 |
| `evaluation/fork_quality/fork_quality_receipts.jsonl` | 32 条噪声质量补核 |
| `replay/evidence/` | 本次完整独立重跑生成的 evidence |
| `replay_verification.json` | 逐字段比对与账本核验 |
| `changed_paths.tsv`、`change_scope.json` | 本地 Git 差异范围 |
| `core_patch.diff` | 本轮执行源码与测试变更，便于审阅 |
| `SHA256SUMS_AUDIT.txt` | 此交付包的内容校验清单 |

生成的模型、观察行、确认包和失败轨迹保留在各控制工作区，便于追溯。包中未收录下载缓存、源仓库副本、Python 字节码及冗长的重复历史检索响应。

本报告文本和表格完成事实核对；页面版保存后还会核对正文、标题和链接。页面视觉版式未通过浏览器预览检查。
