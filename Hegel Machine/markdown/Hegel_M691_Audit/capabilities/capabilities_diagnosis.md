# M6.9.1 能力、可执行继承与多尺度结构专项审计

审计对象：固定提交 `4001bef56ffa77520c248b853e7fed3535ee9fe4`。结果范围：`m6_9_1_a_two_branch_frontier_40ep_20261123`。本专项不重复总报告的合题候选提交、评测隔离与成本审计。所有控制均离线运行，没有模型付费调用、没有重新运行真实容器评测、没有修改被审仓库。

## 1. 结论

本轮确实修好了多项旧实现缺陷，并显著加强了实际程序、最终产物和调用回执的保留。但“旧缺陷已百分之百闭环”“已经以六个学习 AST 加一个共享子树完成实际多尺度继承”“17/20 是完全独立的 1x 管线”三种说法，仍超过现有证据。

需要区分三个层次：

1. **已有静态科学算子被执行**：证据充分。完整 JSONL 可以重算出 4,141 次完成调用，含 4,104 次 agent 调用和 37 次 host probe 调用。
2. **Wave 1 源程序被保留、抽取、挂载到 Wave 2**：部分链路真实存在，但启动资格检查全部失败；完整 producer 的构建命令全部为空；实际记录没有学习模块 API 调用。
3. **系统从真实跨领域程序抽出部分共同机制，再通过多尺度读出影响后继求解**：当前主要是接口接线和合成示例，尚未闭合为结果贡献证据。共享子树的输入甚至不是源程序语义，而是“函数个数 × 驱动 + 状态”的人工代理。

这不要求砍掉跨时刻、多尺度或跨领域部分共性的计划。相反，下一步应把真实复杂程序、状态与时间语义接入已有抽象器，同时允许暂时无法结构化的 Python/C++ 程序作为可验证的不透明算子保留。继续保留程序的丰富性；修的是输入语义、继承身份和执行接口。

## 2. 审计方法与历史复用

已阅读 `Hegel Machine/AGENTS.md`、历史 reuse hub，并查询现有 `historical_capabilities` 注册表。查询结果保存在 `historical_reuse_queries.json`：`subtree` 命中 20 个版本，`retention` 命中 22 个版本；`Pi_fold`、`reconciliation` 精确词查询没有命中。这不代表算法不存在：实际源码已有 M5 的多尺度读出、共享子树提取、语义/角色/时间对齐及撤回传播，另有 `code_agent.retain_evaluated_python_model`、`checkpoint_storage` 等完整持久化基础设施。问题主要在新桥接层如何喂入真实对象、保留执行上下文，以及调用结果是否真正进入后继产物。

检查范围包括整个 `m6_8_b_prime_capability_bridge.py`、整个虫洞算子库、Track A 引擎变更、M5 执行 ABI 变更、相关 runner/调用点/测试及最新证据。还复跑了旧审计中的有针对性的库与 Track A 反例，并增加少量直接验证新桥接语义的控制。控制只说明所构造边界的行为；没有把离线反例直接归因为本轮某题失分。

## 3. 先确认已经修好的部分

| 旧问题 | 当前控制结果 | 可以确认的范围 |
|---|---|---|
| 纯函数抽取丢掉 helper 闭包、嵌套函数 | 两类均可抽取 | 常见闭包提取能力改善 |
| 内层名字错误授权外层自由变量 | 反例被拒绝 | 该作用域边界修复 |
| `np.loadtxt` 被当纯函数 | 已拒绝 | 所测 I/O 入口被封；不代表所有 I/O 都被封 |
| NaN parity 被误判相等 | 现在返回不相等 | 非有限数门控修复 |
| minimax 实际选 medoid | `[-5,0,1,10]` 正确选 1 | 离散最坏偏差选择修复 |
| LOCO 把缺失标签 -1 纳入训练 | 当前排除 -1 | 所测标签缺失处理修复 |
| Fourier 字典缺 Nyquist / 重复频率 | 16 点、8 模式生成 16×16 满秩字典；Gram 误差约 1.57e-15 | 所测正交构造修复 |
| 失败函数也记成功调用 | `status=failed` 且 `call_count=0` | 失败调用不再抬高 completed 次数 |
| 中途进程退出丢调用记录 | 100 次后 `os._exit` 仍保留 100 次 | 所测短程实时记录修复；不等于任意聚合边界零丢失 |
| Track A 动态 key/alias 导致真实成功代码被丢弃 | 保存的 M6.3 成功源程序不再被丢弃 | 旧实证丢失问题修复 |
| 同变量不同机制被去重吞掉 | `x` 与 `x**2` 均保留 | 去重键开始区分程序结构 |
| 可选 `state=None` ABI 没有收到 fit state | 正确识别 `predict_row_and_fit_state`，RMSE 0 | 所测 ABI 修复 |
| 历史 L1 投影完全不看旧模型 | 换成 ±1e6 旧公式，投影数值与符号随之变化，并拒绝一致 | 已不是原来的固定演示投影 |
| 低覆盖 owner 仍冒充原分支 | 外来变量占主导时合成 composite branch | 所测 owner 规则改善 |
| Python audit guard 被 ctypes 读文件绕过 | 对本审计生成的无害数字 marker，open 与 ctypes 均被拒绝 | 旧已知入口修复；不宣称 Python hook 是 OS 级隔离证明 |

LOPO 公共速度偏差仍不能由路径删一稳定性识别，但当前 API 明确标注 `subset_jackknife_precision_only_not_common_velocity_bias`。这是正确收窄能力声明，应当保留。以上结果见 `library_regression_controls.json`、`tracka_controls_results.json`。不应因为仍有新边界就否认这些实际修复。

## 4. P1：64 个模块全部没有通过启动资格检查

**观察到的事实**：最新 15 个 TB Antithesis episode 的 `D1C_SKILL_BINDING_AND_SMOKE_QUALIFICATION`，合计 64 个模块，全部标为 `qualification_exec_failed`，`qualified_count` 全为 0。随后确实有实际 skill 调用，所以问题不是“库根本不可用”，而是预期的 import/qualification 流程没有运行成功。

原因直接定位到 [bridge L936–951](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/src/hegel_machine/cognition/m6_8_b_prime_capability_bridge.py#L936)：多行 Python 代码用 `json.dumps(qual_py)` 拼进 `bash -lc`。JSON 字符串转义不能充当 shell 参数转义。实际传给 `python3 -c` 的代码含字面 `\n`，触发 `SyntaxError: unexpected character after line continuation character`。

**复现**：`qualification_control.py` 只把 Docker 执行替换成运行相同 shell 命令，导入无害的 `math`；同样得到 return code 1 和上述错误。没有使用真实任务或隐藏数据。

**修法**：改用直接参数数组，例如 `docker exec -e PYTHONPATH=/tmp <cid> python3 -c <真实多行代码> <模块列表 JSON>`；或写脚本文件后执行。保留 stderr、return code、模块源码 hash。导入成功与合同样例成功必须分成两个状态。现在函数名说“contract qualification”，实际就算 quoting 修好也主要做 import 和 callable 枚举，尚不是参数/状态/单位/时间语义的合同验证。

**验收**：在现有真实保存模块上执行同一入口，确保各模块进入正确的 imported/contract-checked/failed 状态；失败原因不应归一成没有细节的 `qualification_exec_failed`。这一步只需已有模块和公开合同样例，不需要再跑 20 题付费评测。

## 5. P1：源文件保留不是完整 producer 闭包

### 5.1 构建/调用命令采集存在明确字段错配

[bridge L607–611](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/src/hegel_machine/cognition/m6_8_b_prime_capability_bridge.py#L607) 从 `turn["decision"]["bash_command"]` 读命令；实际 [frontier L3010–3017](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/src/hegel_machine/cognition/m6_frontier_real_benchmarks.py#L3010) 的日志字段是顶层 `turn["bash_command"]`。

**实证**：最新 30 个 TB 分支 manifest，`build_and_invoke_commands` 全部为空。C++ 源码被保存，并不意味着其编译参数、链接环境、运行命令和输入工件已封装成可重建技能。

**修法**：统一消费真实 turn schema，保存有顺序、有 cwd、有退出状态、能关联产物的 recipe；不要只截最后 12 条且每条最多 500 字符的命令片段来声称完整。先复用既有 execution-context / checkpoint 设施；从最终候选 hash 反向关联实际生成命令及输入。

### 5.2 `retained_evaluated_model` 的选择没有证明它是被评价的求解器

[bridge L589–599](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/src/hegel_machine/cognition/m6_8_b_prime_capability_bridge.py#L589) 取 `py_prods[-1]`，再传入固定 `training_mse=0.0`。最新 sparse Antithesis 的 primary 是 `app__data__check_outputs.py`，certified Antithesis 是 `tmp__verify_sparse.py`：这两个选择不能当作已确认求解模型的身份证明。

原 `code_agent` 保留工具负责复制上下文，它信任调用者给出的“被评价文件”。应从候选产生链选出实际入口，再绑定真实公开验证 receipt；没有 MSE 评价的任务用明确的验证状态，不能填写无依据的 0.0。

### 5.3 中间状态、依赖与截断仍未闭合

[bridge L501–546](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/src/hegel_machine/cognition/m6_8_b_prime_capability_bridge.py#L501) 仅保存特定源码后缀、单文件不超过 2MB，并静默截为前 40 个；声明产物另存，未枚举真实程序读取的所有状态。Sparse 的源程序引用 `/tmp/sign_prior.npz`、多组 warm-start `.npz` 等，manifest 没有把这些完整列成执行依赖。Spin 的 Antithesis 达到 40 个上限并混入 `/tmp/spin_deps/pysat` 包文件。不能由“有 40 个文件”反推闭包完整，也不能仅据达到上限断言某个自写 solver 必然丢失。

Blob 限制已从旧的很小阈值提高至 15MB，这有实际价值。本轮 certified 的 10,017,387 字节 `.bin` 确实在 `branch_trajectories/content_addressed_blobs/` 中；不要误报为丢失。所有 manifest 共出现 89 次 blob 引用、56 个不同引用，其中 8 次引用对应 4 个缺失 `.pyc`，总计 69,931 字节。它们影响“每个引用都可解析”的封存完整性声明，不能直接等同为主要科学产物丢失。

**具体改法**：保留“opaque_full_producer_retained”这一有价值的方向，但把状态拆成 `sources_captured`、`dependency_closure_declared`、`restore_verified`、`public_contract_verified`；只有在干净容器里恢复源、build recipe、状态及环境后能产生规定输出，才赋予可执行闭包资格。不能结构化的 C++/Python 应继续保留和调用，无需为可解释性强制转换成小 AST。

## 6. P1：共享子树目前来自人工代理，与真实程序语义脱钩

这是本专项最重要的计划—实现差距。

[bridge L779–792](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/src/hegel_machine/cognition/m6_8_b_prime_capability_bridge.py#L779) 给每个 Wave 1 源构造同一种 `Program`：

```text
add(mul(const=max(1, number_of_extracted_functions), feature=u_drive),
    feature=memory_state_or_slow_state)
```

`memory_state` 或 `slow_state` 又取决于源码字符串关键词判断。然后 [L849–869](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/src/hegel_machine/cognition/m6_8_b_prime_capability_bridge.py#L849) 拿前两个条目调用已有 `anti_unify_programs` 和 M5 子树工具，手工填写 drive_response/state_carryover、角色与探针签名。

因此目前的“一棵真实 SharedBaseSubtreePrior”，证明了注册接口被调用，不能证明从 AMR 与波传播程序中发现了实际共同机制。最新记录的子树是 `subtree_wave1_amr_poisson_optimize__baseline_free_localization`；它的选择来源是列表顺序。

**离线语义变更控制**：把两个源的实际公式从线性律改成 `sin(a*x*x)`，保持抽取函数数量与 temporal label 相同，得到的共享子树完全相同，hash 均为 `6e7ad7b15aafa2d61c47f1016dfbc2efad87bb34ed9b02f55ba75b8675fc106c`。见 `structural_controls.json`。该结果直接验证了“当前共享子树对被替换的真实公式不敏感”。

**已有能力与缺口**：M5 的 `extract_shared_base_subtree_from_sources` 接收两个真实 `Program` 并进行 anti-unification；它不是空壳。缺口是新桥接器没有传入真实机制结构。

**修法**：

1. 对可识别的真实程序局部建立结构 IR：运算、数据依赖、状态更新、forcing、单位、时间差、边界条件、模型参数、记忆核/阶段变量。只提升有证据的部分，保留未解析主体和原始程序 hash。
2. 对不透明算子声明输入/输出/状态合同，利用公开干预与反事实测试识别可迁移不变量；不能解析的部分继续作为 opaque 节点参与组合。
3. 两来源只共有一部分时，对共有部分 anti-unify；差异留在条件分支、剩余槽位、适用范围和后继补全中。不要把两个完整机制硬压成一个总公式。
4. 候选对子应来自角色、单位、因果依赖、时间读出兼容性，而不是列表最前两个。保留多个非支配共性，不必只保留一棵树。
5. 用“源机制被干预，抽象相应变化；共同机制不变，源特有部分不污染共性”的反事实控制验证抽象语义。随后检查后继是否实际引用该子树并改变程序。

这才保留用户要求的跨时刻、多尺度和逐层部分共性；当前的人工代理反而已经把它们大幅简化。

## 7. Pi_fold 和分支撤回：有接口、有调用，但研究闭环尚未完成

### 7.1 M5 readiness 标志来自固定三点示例

[bridge L889–921](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/src/hegel_machine/cognition/m6_8_b_prime_capability_bridge.py#L889) 每次用相同三点序列调用历史 M5 temporal/Pi_fold，用 `Program(feature=q)` 且空 source 列表调用语义对齐器。这证明导入与函数接口可运行，不能说明 Frontier 的真实 producer 已获得角色、单位与多尺度对齐。

### 7.2 Nano 的三次 Pi_fold 是打印诊断

本轮三次 agent Pi_fold 调用真实存在。但保存的 [nano `tmp__indent_reduce.py` L73–82](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/docs/evidence/m6_9_1_a_two_branch_frontier_40ep_20261123/branch_trajectories/wave_1__gpt-6.2-sol__tb__nanoindentation-property-extraction/branch_2_antithesis_wormhole/04b_producer_skill_manifest/producers/tmp__indent_reduce.py#L73) 仅对 sample_06/08/10 调用并打印前 180 字符，没有把返回模型用于卸载拟合或校准计算。

这允许一种较弱解释：诊断文本可能影响后续思考。不能凭此说完全没有影响；同样不能把三次调用直接计作 Pi_fold 数值机制对最终结果的已证明增益。

### 7.3 新 Pi_fold helper 的两项具体合同缺口

[bridge L200–205](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/src/hegel_machine/cognition/m6_8_b_prime_capability_bridge.py#L200)：用户显式指定 `cutoff_t0` 后，如果前缀不足四点，代码静默切成“最后两点以外都训练”，同时仍报告原 cutoff。离线例 `t=0..9, cutoff=0` 本应只有 1 个训练点，却改用 8 点。这违反因果截点合同；不代表本轮 nano 默认截点发生同样问题。

[L224–249](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/src/hegel_machine/cognition/m6_8_b_prime_capability_bridge.py#L224)：同一 forward suffix 既挑选 memory tau/模式，又被称为 confirmation 分数。可把它当模型选择集，不能再当独立确认。返回值主要是模式/误差/tau，没有可直接在新 horizon 继续执行的 fitted coefficients/state；慢 ODE 又按离散步而非真实 dt 运行，和处理 dt 的记忆核比较时要约束时间网格。

**修法**：不足合同数据时返回 insufficient causal context 或显式要求重新采样，不能悄悄越过截点。保存可执行拟合状态、训练时窗与选型时窗；在另一后续 horizon 确认。使用现有 M5 的真实时间读出接口，接入轨迹段、单位、状态与强迫，保留多模式而非强制每题一个模式。

### 7.4 撤回传播作用在临时图，未更新活跃继承状态

[bridge L872–887](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/src/hegel_machine/cognition/m6_8_b_prime_capability_bridge.py#L872) 新建一个 claim list，所有 `supported_by=[]`，调用撤回传播后返回摘要，没有把返回状态写回真实 archive/forest 的技能资格。最新 12 个节点全部变为 `pending_empirical_support`，不是从真实前提依赖中选择性撤回受影响后继。

**修法**：以实际 premise/claim/artifact ID 建支持图；把撤回落到 active retrieval/dispatch 的资格状态，保留未受影响部分、分支版本与历史 receipt。不要只生成新报告字段。撤回某个源特有前提不应自动撤掉仍有独立支持的共享子树。

## 8. 4,141 次调用、6 个 AST 和 1 棵子树分别证明什么

完整 JSONL 的独立重算如下：

| 任务 | 完成调用 | agent / host | 含义 |
|---|---:|---:|---|
| nanoindentation | 4,121 | 4,097 / 24 | 主要是 breakpoint 和 Feng–Ngan 反复计算；其中 Pi_fold 3 次打印诊断 |
| clinical | 9 | 1 / 8 | cohort aligner 确实调用 |
| symbolic | 5 | 0 / 5 | 全是宿主探针，没有 agent scanner 调用 |
| baseline-free | 4 | 4 / 0 | envelope 与 tomography |
| guided-wave | 1 | 1 / 0 | tomography |
| virtual-baseline | 1 | 1 / 0 | tomography |
| 合计 | 4,141 | 4,104 / 37 | 6 个任务；nano 占 99.52% |

总量是真实算子执行数，但不能代表同等数量的独立洞见、能力改进或跨领域继承。调用次数不宜作为主要进度 KPI；应同时报告任务覆盖、函数覆盖、来源种类、产物变化、消耗和验证收益。

Wave 1 的旧抽取路径生成 6 个 `wave1_sublated_*` 模块，新 manifest 路径另生成 5 个 `wave1_manifest_*` 模块。记录其模块名/函数名以及挂载位置有价值。**完整调用日志里，两类学习模块的 API 调用均为 0**。这表明该路径的执行性 adoption 尚未证明；不能进一步断言“没有文本学习或代码复制”，因为 LLM 仍可能阅读 briefing 或复制源片段。

“10 个 full manifest 事件”也应拆开：8 个 TB 实体 manifest，加 2 个 BioMystery 从历史 DAG/heredoc 回收的事件；后者没有对应的完整 `04b` manifest 文件。回收函数识别到源码片段可用于后续重建，但不能仅凭 `recovered_file_count>0` 证明编译、后续 sed 修改、运行状态与产物都能完整重放。

本节源记录：[Wave 1→2 consolidation](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/docs/evidence/m6_9_1_a_two_branch_frontier_40ep_20261123/wave1_to_wave2_consolidation_gpt-6.2-sol.json)。离线重算脚本为 `recompute_capabilities.py`。

## 9. 17/20 的分支成绩与 1x 谱系应分开

Runner 在 [L245–262](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/scripts/run_m6_9_a_two_branch_study.py#L245) 用 Wave 1 **合题胜者**的 top-level summaries/episode dirs 做 consolidation，再把这个 archive 给 Wave 2 Antithesis。

实际 top-level manifest 的源分支：

| Wave 1 任务 | 被继承的实际分支 |
|---|---|
| AMR、navigation、ODE、sparse | Thesis |
| baseline-free、nano、reactor、traffic | Antithesis |

因此 Antithesis 的 17/20 可以作为本次两分支实验内的分支终局统计；它不能直接当作完全单独运行、从 Wave 1 到 Wave 2 都只有 1x 成本的系统成绩。即便后继没有调用学习 API，合题胜者的文本经验、来源选择及前一波额外计算也进入了可获得上下文。独立 1x claim 需要 Anti-only Wave 1 继承链，或者单列共享预训练/库建设成本。

结果专项还定位到 6 个 top-level summary 的 `producer_skill_manifest_summary` 仍保留 Anti 身份，而实际复制的 manifest 已为 Thesis。应以实际 manifest 文件和生成/选择 hash 为准，把摘要同步为同一来源，不能让报告与继承体分别指向不同程序。这里不重复合题事务审计。

## 10. 仍需修的结构边界：不能说百分之百闭环

以下均有小型复现；只有前述 startup、manifest、调用覆盖和共享代理直接对应本次实际证据。

### 10.1 少数切片保护会忽略消失的切片

[bridge L264](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/src/hegel_machine/cognition/m6_8_b_prime_capability_bridge.py#L264) 使用两个损失字典的交集。Incumbent 为 `{majority:1.0, minority:0.1}`，challenger 只提供 `{majority:0.5}`，仍能被提升。应要求覆盖预声明的同一切片集合；缺失切片标为不可比较/未验证，不能悄悄忽略。保护局部共性与少数条件时尤其重要。

### 10.2 AST 角色翻译仍把“可能是 row”当成“一定是 row”

条件分支里 `r=row` 后改成局部 `{'x':7}`，或者同一个 helper 同时接收 row 与参数 dict，翻译仍返回 `ok=True, binding_status=bound`，把局部字典的 `x` 改成 `target_x`，运行抛 KeyError。旧直线 alias、关键字 helper 等已改善；这两种控制流/多调用上下文是剩余边界。

应保留不确定绑定：局部 specialization，或仅改写被证明属于 row 的访问；不能确定时保留原始 opaque 程序并等待合同适配，不要标成功后静默接错参数。

### 10.3 纯函数检查仍允许 `np.load`

虽然 `np.loadtxt` 已拒绝，`return np.load(path)` 仍被抽作纯函数。需要按声明效果与依赖分类；有 I/O 的可执行技能仍可有价值，但不能包装成无状态纯函数。与其不断扩大黑名单，不如记录 effect/文件输入，并在相应执行合同下使用。

### 10.4 历史投影会被单箱均值抵消

当前旧模型确实被执行，是明显改善。但在五个点 `x=-2,-1,0,1,2` 上，旧律 `y=-x`、新律 `y=x`，新投影仍返回 `passive_shadow_consistent=True`、最大 bin 缺陷 0；逐点差异 RMSE 是 2.828。原因是小样本退为单个箱，正负误差抵消。

应验证对应条件上的投影，而不只验证全局平均；保持旧模型身份及旧参数的冻结语义，明确何时允许重拟合。小样本不够辨识时返回“未确认”比“已一致”准确。相关实现：[Track A L941–988](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/src/hegel_machine/cognition/m6_3_sovereign_causal_wormhole_engine.py#L941)。

### 10.5 Symbolic host probe 的 scanner 结构合同不一致

scanner 返回单变量候选 `term=(kind,j)`；host probe 却读 `h['col']`。用公开合成二分类强单变量数据运行真实 scanner/probe，会得到 `[MECHANICAL_WORMHOLE::SYMBOLIC_SCAN_NOTE] KeyError: 'col'`。异常被包成 NOTE，因此外层可以正常返回而没有成功完成该诊断。最新 symbolic 日志有 scanner 调用，并不证明这个特定错误在本轮实际触发；本控制只说明它是一个可触发缺陷。修法是统一 term schema、显式记录 probe error，不能把错误归类为没有问题。调用点：[library L1667–1685](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/src/hegel_machine/cognition/m6_4_executable_wormhole_library.py#L1667)。

### 10.6 d-separation 的独立作用仍需按数据区分

当前构造性的单测可以证明 d-separation 具备阻止特定写入的能力；但在保存的真实 M6.3 Sol Wave 1 receipt 上，禁用该判定后最终 archive 完全相同，写回成本均为 10.5，blocked targets 由 2 变 0。不能因合成例有效就断言历史实际增益已经归因于 Bayes-ball；也不能据一个实际样本无变化就认定该方法无用。此处报告“能力存在，本样本独立贡献未检出”。

## 11. 后续调整：先闭合真实对象，再评估智能增益

### 第一层：修当前证据链，使用已有数据验收

修 qualification shell 参数、build recipe 字段、producer 身份选择、摘要与 manifest 来源一致性、缺失 blob 引用；把失败/不适用/未知状态分开。这些问题都有现成回执或小型控制，无需新大评测。

### 第二层：把完整 producer 作为真实继承单位

继续复用 `code_agent` 与 `checkpoint_storage`。每个最终候选绑定其 source、state、build/invoke graph、输入合同、有效条件和公开验证 receipt。先证明 source→artifact 可重建，再声明 learned skill；对 C++ 和有状态复杂程序不要求降为纯函数。

### 第三层：把历史 M5 接入真实任务对象

使用已有 M5 机制而非重新发明另一组同名示例。真实轨迹切片→角色/单位/时间对齐→候选 Pi_fold/state kernel→部分共同 IR→适用前提和补全槽位→后继执行。多尺度是模型结构与适用范围的一部分，不是简单加一个 temporal 标签。把撤回传播写入真正的 archive dispatch 状态，保留未被推翻的部分结构与分支。

### 第四层：建立 adoption 与结果的最短证据链

每个后继引用记录 source artifact/hash、所用函数或 opaque entry、输入形状/角色/时间范围、返回值去向、最终产物 hash、公开验证变化。对于“只读 briefing”“复制程序”“直接 API 调用”分别记录实际通道，避免把没有 API 调用误说成完全没有传承，也避免把模块挂载当成已使用。

冻结已有 W1 产物后，先做少量能够区分机制的配对检查：文本经验、静态库、学习 producer、学习部分子树分别增加了什么；保持同一公开数据和运行预算，不重新挑最有利的题。只有存在新、可执行、在后继有可测贡献的能力时，再讨论递归改进强度。

### 第五层：把研究主张与预算主张分开登记

维护单轨独立继承、双轨共享继承、合题提交三个口径。当前 17/20 分支统计与实际 18/20 合题结果可以并列保留，但不能把前者自动标为独立 1x。新实验应只解决尚缺的这一个因果或成本问题；不要为完善叙事反复重跑已解决例子。

## 12. 可复现材料

所有脚本支持 `HEGEL_AUDIT_REPO`，可设为固定提交的仓库根，也可直接设为 `Hegel Machine` 目录。完整命令、依赖与 fixture 在本目录 `README.md`。

| 文件 | 用途 |
|---|---|
| `recompute_capabilities.py` / `empirical_capabilities_recomputed.json` | 从本轮提交的 DAG、manifest、完整 JSONL 重算资格、调用、谱系、缺失引用 |
| `qualification_control.py` / `.json` | 复现同一 shell quoting 错误，无 Docker |
| `library_regression_controls.py` / `.json` | 对旧算子缺陷定向复核 |
| `tracka_regression_controls.py` / `tracka_controls_results.json` | Track A 旧缺陷、历史真实 receipt 消融和 ABI/隔离边界 |
| `structural_controls.py` / `.json` | 共享子树语义、因果 cutoff、切片保护、AST 边界、历史投影、symbolic schema |
| `historical_reuse_queries.json` | 现有历史注册表的相关查询记录 |

本报告不将小型边界例充作本轮错误轨迹，也不以调用数代替性能或因果归因。已有通过记录、已修缺陷和有用的复杂程序都应保留。下一轮最值得完成的是：让已有高维、跨时刻、跨领域部分共性方案真正消费和生成真实程序，而不是继续增加同名状态字段。
