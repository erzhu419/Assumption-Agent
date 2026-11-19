# Hegel M2 独立审计与复现包

审计对象为 `erzhu419/Assumption-Agent`，固定提交：

```text
9115f9adf9dad1f52f106a2ab14babe8b68243dd
```

本包保存审计结论、原始回执的复算、原研究脚本的独立重放，以及明确标注为事后诊断的控制实验。它不修改原仓库、不调用 LLM/provider、不调用网络，也不把诊断控制当作新的预注册性能实验。

## 阅读入口

- [M2_AUDIT.md](M2_AUDIT.md)：主审计报告。
- [在线主报告](https://chatgpt.com/space/page_ff9f70d9bcb08191ab03c9c42491107c)：同一结论的可阅读版本。
- [reports/MECHANISM_AUDIT.md](reports/MECHANISM_AUDIT.md)：机制、接口缺陷、具体反例和最小修复验收。
- [reports/M2_RESULTS_AUDIT.md](reports/M2_RESULTS_AUDIT.md)：原结果复算、公平性控制、成本及统计范围。
- [results_index.json](results_index.json)：结果文件逐项说明、来源类别和摘要哈希。
- [provenance.json](provenance.json)：固定提交、26份核心文件的 SHA256/Git blob SHA、便携包装差异与旧回归输出头部勘误。

历史与参考文献审计另见本包历史专项报告；其覆盖级别与无法取得的全文按该报告注明，不据文件清单冒称逐行阅读或全部历史重跑。

## 文件布局

| 位置 | 内容 |
|---|---|
| `scripts/` | 5个便携化审计脚本、路径初始化及原研究重放入口 |
| `mechanism/` | AST、运行边界、确认训练污染三个 focused probe 的原始 JSON |
| `results/receipt_recomputation.json` | 原始 ledger、配对结果及生命周期复算 |
| `results/controls_results.json` | 事后对照的逐题结果和簇汇总 |
| `results/replay/` | 原 M2 研究脚本在固定9115上独立重放的完整12份回执 |
| `evidence/published_m2/` | 固定提交内作者发布的原始12份回执 |
| `evidence/mechanism_workspaces/` | focused probes 的实际 `model.py`、观测、合成器配置与脚本 |
| `m0_recheck.json`、`boundary_recheck.json` | 在9115重新执行历史语义/边界探针的结果，旧头部勘误见下文 |
| `history/fixed_commit_file_inventory.json` | 固定提交的全树文件路径与Git对象清单；不代表逐文件精读 |

原始 JSON、JSONL、日志逐字保留。它们可能记载原运行时间、原位置或旧脚本常量；这些是证据文本，不是复现时使用的路径依赖。便携脚本只读取 `HEGEL_REPO`，并只向 `AUDIT_OUTPUT_DIR` 写本轮结果。大型事后控制工作区未全部捆绑，完整逐题 JSON 已保存，工作区可由相应脚本重新生成。

## 准备环境

使用 Python 3.10 或更新版本；本次实际复核使用 Python 3.12。需要一个已经检出的固定9115仓库，以及该仓库的 Python 依赖：其 `pyproject.toml` 声明 `cryptography>=41`、`scipy>=1.11`、`threadpoolctl>=3.1`；仅在额外运行仓库 pytest 时需要 `pytest>=8`。本包本身不新增第三方依赖；包内脚本使用标准库并导入仓库模块。原模型评分会启动同一 Python 解释器的本地子进程。

`HEGEL_REPO` 应指向包含 `src/hegel_machine` 的 **`Hegel Machine` 目录**。`run_audit.py --repo` 也接受其上一层 Assumption-Agent 仓库目录并自动定位该子目录。

先只检查版本和26份核心文件，不运行任何实验：

```bash
python run_audit.py --repo '/path/to/Assumption-Agent/Hegel Machine' --check-only
```

若当前 Git HEAD 不等于固定提交，或核心文件 hash 不同，入口停止。没有 Git 元数据的源码副本也可检查26份文件hash，但会明确标记只校验了文件。完整复现仍应使用指定提交的完整 checkout。

## 运行命令

### 1. 默认 focused 审计与原回执复算

```bash
python run_audit.py --repo '/path/to/Assumption-Agent/Hegel Machine'
```

默认执行 `ast confirmation runtime receipts`，输出到本包新建的 `rerun/`。保留随包原始结果不覆盖。也可以显式指定：

```bash
python run_audit.py --repo '/path/to/Assumption-Agent/Hegel Machine' \
  --output './my_rerun' --steps ast confirmation runtime receipts
```

### 2. 完整重放作者原 M2 实验

```bash
python run_audit.py --repo '/path/to/Assumption-Agent/Hegel Machine' \
  --output './original_study_rerun' --steps replay
```

它直接调用固定源码中的 `run_m2_study(output_dir)`，不修改研究逻辑。会运行2个开发簇、4个评估簇和6个评估臂；输出12份回执至指定输出目录的 `replay/`。本次已完成的独立重放原始证据在 `results/replay/`。

### 3. 运行事后诊断控制

```bash
python run_audit.py --repo '/path/to/Assumption-Agent/Hegel Machine' \
  --output './posthoc_controls_rerun' --steps controls
```

控制脚本在独立进程中构造派生类或运行时方法，不改仓库源码；三项干预是：

1. Flat 消费其构造时已复制的同一个 Stage1 gate；
2. 关闭 horizontal edges 时保留现有 local 节点作为可确认的历史候选；
3. 普通基线使用所有已经验证的同元数历史 schema，并向额外候选收费。

这些控制用于定位原归因的脆弱处，不能取代新的冻结公平对照，也不能据此主张普通方法在所有任务上普遍更优。原科学数据、阈值、失败、测量费用及普通求解器均保留。

所有步骤一起运行：

```bash
python run_audit.py --repo '/path/to/Assumption-Agent/Hegel Machine' --steps all
```

### 4. 直接运行单个脚本

根入口会设置两个环境变量，也可以自行设置：

```bash
export HEGEL_REPO='/path/to/Assumption-Agent/Hegel Machine'
export AUDIT_OUTPUT_DIR='/path/to/new_audit_output'
python scripts/probe_ast.py
python scripts/probe_confirmation_training_leak.py
python scripts/probe_runtime.py
python scripts/recompute_receipts.py
python scripts/run_controls.py
```

这五个脚本的便携化只调整仓库路径、输出路径、导入初始化和 AST 结果落盘，不改探针输入、公式、阈值、对照策略或被审计函数。原脚本与包装脚本的 hash 对应关系在 `provenance.json`。

## 如何理解复现结果

- `probe_ast.py` 使用原规范化函数定义，演示旁注与执行程序分离、未解析变量和指数符号被抹除的情形；不把它解释成原40个Hegel目标必然出错。
- `probe_runtime.py` 的列顺序和未支持函数控制采用默认 gate，是新的 focused 诊断，成本不应混入作者的冻结配对汇总。
- `probe_confirmation_training_leak.py` 使用完全合法的 `fit/predict` 模型证明确认标签进入训练；原实验模板的 `fit` 为空操作，因此这不是对已报告40题存在训练泄漏的指控。
- 原始研究里 HExA 的并列模板选择依赖 `set` 迭代顺序。本包入口对子进程声明 `PYTHONHASHSEED=0`；可以用 `--python-hash-seed 3` 检验边界。原作者及已有重放中未冻结的 hash 顺序可能让 HExA 某些汇总不同，这一限制已保留，不承诺所有字段逐字一致。
- 真实墙钟会随机器变化。代码中合成时间固定记0.02秒是被审计的问题之一，不能把原 `python_execution_seconds` 当成完整真实子进程计时。
- `cost_sensitivity.json` 是对保存成本表的代数推导；不是新增科学执行。`edge_reproducibility.json` 保存额外的 fork/hash 顺序控制观测。

## 旧 M0/边界结果的版本头部勘误

`m0_recheck.json` 和 `boundary_recheck.json` 头部仍写旧提交 `675f52612a9d0f1fb208da8bdb0552de3c571758`。原因是仓库历史探针把该字符串写死；**本轮实际 `--repo` 指向9115 checkout**。原 JSON 没有悄悄改写，`provenance.json` 将旧头部与实际执行提交分开记录，并列出当前源文件hash。

若需要重复这两组历史控制，直接调用指定提交中的原脚本：

```bash
python "$HEGEL_REPO/markdown/Hegel_V166_diagnosis/scripts/probe_m0_semantics.py" \
  --repo "$HEGEL_REPO" --output "$AUDIT_OUTPUT_DIR/m0_recheck.json"
python "$HEGEL_REPO/markdown/Hegel_V166_diagnosis/scripts/probe_boundary_contract.py" \
  --repo "$HEGEL_REPO" --output "$AUDIT_OUTPUT_DIR/boundary_recheck.json"
```

其输出旧头部仍会出现，同样应按实际传入源码的hash解释，不能把它误判为本轮执行了旧版本。原M0 probe的某些方法用受控夹具或AST隔离，属于语义边界回归，不等于完整在线研究器实测。

## 本包验证范围

便携化后只进行了 Python 编译检查、入口帮助及固定提交/hash检查；没有为了包装再次运行全部科学实验。已有原始重放、focused probes、事后控制的结果均来自本轮先前已完成的实际执行，并原样保留。
