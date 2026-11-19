# Hegel M3 审计复现包

固定提交：`612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8`  
比较基线：`9115f9adf9dad1f52f106a2ab14babe8b68243dd`

先读 [M3_AUDIT.md](M3_AUDIT.md)。它包含结论、证据边界、代码引用、修复次序与 M4 验收建议。三个专项报告提供更细的行号和实验解释。

## 文件组织

- `M3_AUDIT.md`：完整主报告。
- `closure/`：开放作者、确认文件隔离、AST 与收据边界；附三份脚本及完整小工作区。
- `mechanism/`：固定 eval04 图的检索因子分解与九次实际 target 控制。
- `results/`：全量收据复算、两套原研究完整重放、费用控制与候选消费分析。
- `committed_evidence/`：固定提交原始 V16.9 和 M3 证据副本。
- `source_snapshots/`：本次 16 个核心变化文件及若干关联文件；用于对照阅读，不是完整可安装仓库。
- `INPUT_HANDOFF.md`：本次用户提供的交接材料。
- `changed_paths.tsv`、`focused_patch.diff`、`tree_612d8eea5_inventory.json`：固定版本变化和树清单。
- `AUDIT_PROVENANCE.json`、`ENVIRONMENT.json`、`FILE_MANIFEST.json`：版本、环境及 SHA256 清单。
- `pytest.log`：20 个里程碑测试通过；8 个存储测试通过；1 个真实 7z 测试因环境缺可执行文件未通过。

没有修改被审计仓库，没有远程写入或付费模型调用。主实验重放只调用原有确定性入口。新控制是事后诊断，不能代替新版本的冻结评测。

## 准备

需要固定提交的完整仓库和其 Python 依赖。将下面 HEGEL_REPO 改成该 checkout 中的 Hegel Machine 目录；从本复现包根目录执行。路径值仅为需要替换的示例。

```bash
export HEGEL_REPO="/absolute/path/Assumption-Agent/Hegel Machine"
export PYTHONPATH="$HEGEL_REPO/src"
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p rerun/results rerun/mechanism rerun/closure
```

本包的原始输出保留在各目录。下面将复算输出放入 rerun；不要把原始证据覆盖后再与它自身比较。仓库官方研究脚本默认输出其 evidence 目录；本包 replay wrapper 会显式指定独立输出，适合审计重放。

## 仅对原始收据重算

无需科学模型拟合：

```bash
AUDIT_OUTPUT_DIR="$PWD/rerun/results" python3 results/recompute_receipts.py
```

它核对费用、模块哈希、簇汇总、Recall、固定成本顺序置换与 DEV 摊销。

## 小型失败见证与机制控制

closure 的脚本按自身目录保存完整工作区，因此先复制脚本到独立输出位置。三份脚本通过 PYTHONPATH 使用固定仓库，不需修改其中科学逻辑。

```bash
cp closure/probe_closure.py closure/probe_open_forced_queries.py closure/probe_m3_ast_rejection.py rerun/closure/
python3 rerun/closure/probe_closure.py
python3 rerun/closure/probe_m3_ast_rejection.py
python3 rerun/closure/probe_open_forced_queries.py
AUDIT_OUTPUT_DIR="$PWD/rerun/mechanism" python3 mechanism/probe_retrieval_factorization.py
AUDIT_OUTPUT_DIR="$PWD/rerun/mechanism" python3 mechanism/probe_target_controls.py
AUDIT_OUTPUT_DIR="$PWD/rerun/results" python3 results/probe_cost_accounting.py
```

检索控制只覆盖提交唯一持久化的 eval04 图。target 控制将 T13/T20 原来已实际发生的 Stage-1 admission 固定为 true；其余使用原消费者，并核对原臂结果与原证据一致。开放链删除额外查询控制仅覆盖 seed 5026101。费用归一化没有重校准 gate，也没有给出新 N*。

## 完整原实验重放

本次环境耗时约每项三分钟；用户环境与依赖不同会改变墙钟。

```bash
AUDIT_OUTPUT_DIR="$PWD/rerun/results/replay" python3 results/replay_study.py
AUDIT_OUTPUT_DIR="$PWD/rerun/results/replay_v16_9" python3 results/replay_v16_9.py
AUDIT_OUTPUT_DIR="$PWD/rerun/results" python3 results/derive_baseline_and_contribution.py
```

最后一项读取 replay 的 target 收据，以复算候选可达性、T13 贡献、匹配分母 Recall 及原/重放墙钟。原始两套完整重放已经包括在 results/replay 和 results/replay_v16_9 中。

## 证据解释与文档检查

- 同一任务不同 arm 的运行数不等于独立机制数。
- Recall 是授权后的生成 schema 字符串命中；有效的其他公式也可能严格通过任务。
- compute 是代理计算单位，measurement 是任务定义的测量费用；两者不是实测秒数。
- 所有新诊断均标明范围；归一化费用、固定成本置换与冻结图消融，不冒充重新冻结的新主实验。
- 报告的数字、代码链接、证据引用和各 JSON 已做核对。在线报告做内容回读；没有使用浏览器视觉预览。

FILE_MANIFEST.json 包含包内除自身以外每个文件的大小与 SHA256。固定源码与原始提交以 Git SHA 为最终标识。

