# M6–M6.2 实现、合成证据与运行器专项诊断

审计对象：`b1d0c3e99f1e6b12cbc62ca8311f1a25d66f2dbf`。本文覆盖 M6 Phase 0、旧 288 episode 合成实验、M6.1 的 192 episode AST 实验，以及 M6/M6.1/M6.2 共用的 frontier 运行器。20 道真实题的逐题复盘、官方 benchmark 配置和全部 frontier 原始账本由主报告另行处理；本文不把未实际运行的 Docker/付费模型实验说成复现。

## 一、判断先行

**M6.1 确实修正了旧 M6 中很关键的结构泄漏，Sol 的 20/20、Luna 的 11/20 和对应 token 数可从已提交回执重算。但目前代码仍不能支撑“零泄漏”“经过因果屏蔽的跨树传播”“最小修订半径”“机械保证单调棘轮”这些强表述。尤其有两个已经在现有日志中发生、并能离线复现的执行错误：把检索第一名当成新知识的拥有者，以及忽略生成程序 `fit()` 的返回模型。它们把已经付费产生的正确结构丢掉、放错树，甚至作为反例否定。**

这不是要求删掉跨领域的部分共性、跨时刻和多尺度能力。相反，目前最有价值的修复，是使已有丰富结构能够通过正确的实例绑定、参数状态接口、传播条件和证据回传真正生效。现有失败里，至少三类已有可执行改善空间：

1. **归属/调度失配**：同一批正确代码只改变完成顺序，目标 heldout RMSE 可以从 `0` 变成 `3.165737`；另一个真实来源顺序交换，可以让 Luna 两个 C02 目标的既有代码 refit 到 `0/0/0`。
2. **程序接口失配**：Luna C05 已写对形式并求得参数，只因 `fit` 返回的闭包未被使用而失败；接通返回值后，同一代码、同一数据的 train/confirm/heldout 全为 `0`。
3. **模型未采纳有效线索**：即便部分 Wave-2 目标已收到可用 AST，Luna 仍会只做被动 probe 后直接 abstain。这一类需要改候选执行与选择闭环，不能再归咎于“森林没有跨域知识”。

这些离线控制证明的是**失效原因和潜在可恢复能力**，不应把恢复结果直接加到原始端到端 pass 分数；原始分数仍然是 20/20 与 11/20。

### 优先级

本文的 P0 指“下一轮结果可信度/知识状态正确性必须先修”，不是声称存在对外攻击事件。

| 编号 | 优先级 | 发现 | 证据强度 |
|---|---|---|---|
| M6-01 | P0 | 检索首位直接成为写回 owner；异任务代码相互覆盖，完成顺序改变知识 | 原始回执 + 原程序离线重放 |
| M6-02 | P0 | 源节点 first-success 去重吞掉后来更适合迁移的已验证模型 | 原始 Luna C02 两份代码 + 目标 refit |
| M6-03 | P0 | `fit()` 返回模型被忽略，软件接口错误被解释成科学假设反证 | 原始 Luna C05/C03 程序 + 仅适配接口的控制 |
| M6-04 | P0（评测边界） | 合成 worker 能导入私有生成器，不能机械保证零泄漏 | 最小反事实 payload 在当前 runner 获得三段 RMSE 0；未发现原轨迹利用 |
| M6-05 | P1 | AST 字符串替换没有做到语义依赖提取与角色绑定 | 两个最小控制：遗漏 helper 依赖、误改参数字典 |
| M6-06 | P1 | 因果/投影证书、可运输性、多跳误差与递归闭环部分停留在 fixture 或提示 | 源码逐段核对 |
| M6-07 | P1 | 非干预的 selected probe 被标成 regime-breaking；重复 confirmation 仍叫 fresh | 反事实 + 实际多轮轨迹 |
| M6-08 | P0（frontier 闸门） | 任意不改 artifact 的 exit-0 命令可清掉验证债务；M6.2 没有实际 incumbent 比较 | 真实 runner 控制外部 I/O 后复现 + 源码 |
| M6-09 | P1 | frontier 的 Bayes-ball 两个固定布尔值没有逐条筛知识，RUNG-1 shell lesson 仍跨树传播 | 源码 + shell-only lesson 控制 |
| M6-10 | P1 | 源任务官方 grader 结果进入 Wave-2 lesson；基线没有等内容记忆对照 | 数据流核对；不等于泄漏 Wave-2 标签 |
| M6-11 | P1 | 审计脚本声明明显超过实际检查；hash 有效不等于因果/隔离有效 | 脚本 8–19 行与 61–119 行比较 |
| M6-12 | P1/P2 | 学习快照缓存、完成顺序、日志截断与外部依赖固定不足 | 源码及已保存证据结构 |

## 二、版本边界与已有进步

M6 提交和 `922b95e19` 的 v18 提交都是 `63b940a03` 的后继分支，不能仅凭用户叙述把它们当成顺序升级。但是两棵树内下列文件的 blob 完全相同，说明这些共享修复实际被复制过去了：

| 文件 | 两提交共同 blob |
|---|---|
| `m5_temporal_multiscale_and_base_subtree.py` | `ea9bb580fa67f447902e3dc3aa40e97bfbb03d47` |
| `wormhole_forest.py` | `09d6c17ad653c65bedc9c0ddccec534676dd1d33` |
| `m5_live_gpt_and_provenance.py` | `1dbd747c3af17d7d98b865bf7920116a832d108e` |

应肯定的修正包括：

- Report 27 第 11 行明确撤回旧 `59/60` 对自主发现的证明力，把 Report 26 降为给定结构参数拟合和管线 smoke test。这是合理的纠错，而不是应该继续使用旧高分证明新算法。
- M6.1 不再按 C01/C02/C03 的子题标签直接拼目标公式；成功程序的 Python AST 确实进入写回与角色翻译。删掉 `winning_python_code` 后不能再凭成功状态产生公式，变量替换测试也有实际意义。
- C03 的第一轮 warning 不再直接给出 `log`/`sqrt` 目标形式；C04 不再直接在 prompt 中塞一条预组装的完整根公式。它仍提供组件知识，这是知识复用实验允许的输入，应单独量化而不是一概视为作弊。
- M6.1 四个 arm 的行单价统一到 `1.5`，且报告把写回成本列出来。M5 基线增加了本地 0-hop AST 写回，优于旧版完全没有等价更新的比较。
- Phase 0 的 fork/chain/collider d-separation 基本实现确实经过纯测试；真实 frontier 的 agent 与 verifier 分容器，架构上比合成 host subprocess 更接近标签隔离。
- SHA 和 provenance chain 的完整性检查在本次复核中通过。问题主要在“这些字节表达了什么、执行了什么”，不是存在已证实的篡改。

计划与报告出处：`markdown/hegel_v16_5_multi_precision_plan/26_M6_Causal_Markov_Blanket_and_Recursive_Pullback_Live_Report.md:13–28`、`27_M6_Frontier_Benchmarks_TBScience_and_BioMystery_PostMortem_Report.md:11,27`、`28_M6_1_Zero_Leakage_AST_and_28Turn_Frontier_Audit_Report.md:13–32`。所有源码行号均以本提交固定内容为准。

## 三、M6.1 AST 192 episode 重算

每个模型每个 arm 有 24 episodes，其中 20 个 in-scope，4 个 OOS。总计 `2 × 4 × 24 = 192`。下表 total C 包括一次 wave consolidation 成本；`tokens` 采用回执的同名累计字段，不把它当成实际货币支出。

| 模型 | arm | in-scope | episode C | writeback C | total C | tokens |
|---|---|---:|---:|---:|---:|---:|
| Sol | Strong Ordinary | 14/20 | 299.70215 | 0 | 299.70215 | 945,115 |
| Sol | EqualAuth Flat | 14/20 | 316.46975 | 0 | 316.46975 | 1,190,371 |
| Sol | M5 Full | 17/20 | 324.84980 | 6.0 | 330.84980 | 1,048,796 |
| Sol | M6.1 | **20/20** | 324.77045 | 8.4 | 333.17045 | **822,129** |
| Luna | Strong Ordinary | 10/20 | 307.63035 | 0 | 307.63035 | 1,233,923 |
| Luna | EqualAuth Flat | 6/20 | 330.40610 | 0 | 330.40610 | 1,259,670 |
| Luna | M5 Full | 4/20 | 335.31420 | 3.0 | 338.31420 | 1,476,732 |
| Luna | M6.1 | **11/20** | 340.44470 | 7.8 | 348.24470 | **1,182,846** |

Sol M6.1 相对 M5 的 token 少约 `21.61%`，total C 多约 `0.70%`；相对 Strong token 少约 `13.01%`，total C 多约 `11.17%`。所以可以说此特定任务集上它提高成功数并减少语言调用负担，不能笼统说所有资源指标都更低。

所有 arm 的 32 个 OOS episodes 都正确 abstain、false confirmation 为 0；但 OOS 使用明显的 `high_freq_sign_oscillation` signature 并配有明确 scope 指令，这证明有用的拒答遵从，尚不足以证明一般性的范围外发现或未知机制识别。

合成任务主要是五类固定机制、两次重复、两波分布变化。它没有充分施压跨时间演化、部分结构逐层组合、深度变化、带非平稳噪声的观测模型，因此不能凭这一组满分说多尺度哲学树已发挥最大潜力。

### 完整性和可复现检查

| evidence 根 | SHA 条目 | 不匹配 | DAG | 无效 DAG |
|---|---:|---:|---:|---:|
| `m6_1_zero_leakage_ast_causal_192ep_20261121` | 719 | 0 | 192 | 0 |
| `m6_live_gpt_causal_aufhebung_288ep_20261121` | 790 | 0 | 288 | 0 |
| `m6_phase0_causal_wormhole_sandbox_20261121` | 1 | 0 | 0 | 0 |

还直接调用了仓库七个相关纯测试函数，均通过：fork/chain/collider、Phase 0 motif/multihop/recursive fixture、K40 与 24-task 生成、旧 M6 vs M5 consolidation、空代码/变量替换 AST 测试、C03/C04 spoiler 检查、frontier CTRF/rung 排序。环境未安装 pytest，因此这里精确称“测试函数直接执行通过”，不称跑过完整 pytest suite。

## 四、M6-01：检索首位被当作新机制 owner，是已发生的知识串树

### 代码路径

`m6_1_zero_leakage_ast_causal_benchmark.py:669` 直接执行：

```python
primary_nid = rec_ids[0] if rec_ids else None
```

这个值通过 episode receipt 的 `primary_recalled_node_id` 进入 `consolidate_wave1_via_ast_wormholes`，后者在 `:460–482` 以该节点作为源 owner，写入赢得确认的完整 Python 程序。这里缺少“新程序实际采用了哪一个实例/子假设、对哪个目标的哪个角色成立”的绑定验证。

检索相关性与知识归属是不同关系。PPR 为寻找启发保留多个候选本来是合理的；第一名恰好来自相近领域，也不意味着把本任务新解放进它的树是合理的。

### 原始证据

| 任务 | 实际首位 owner | 正确含义与错误 |
|---|---|---|
| C01 TreeA | `node_C01_TreeA_prime_electro` | 第一名已是相邻域节点，而不是源 TreeA 的已采用实例 |
| C05 collider-shielded direct X | 同一个 `node_C01_TreeA_prime_electro` | C05 的 direct-X 负对照解被放进 C01 的节点 |
| C04 root pullback | `node_M6_dist_03` | 完整根组合模型被写进 distractor 节点；成功不等于写回成功 |

例如 C05 检索中，错误 owner 的分数约 `.404138`，正确 `node_C05_TreeG_orth_direct` 约 `.337676`。C04 的 distractor 约 `.125`，真正 root 约 `.043360`。这些不是“阈值可能不合适”的猜测，而是实际 receipt 已写出的 owner。

Luna 的两次 C01 Wave-1 都失败，而 C05 一次成功。因此实际 `ast_wave1_consolidation_gpt-6-luna.json` 把 C05 的 `x_ind,u_ctrl` 程序写到 C01 prime，再传播到 TreeA 和 TreeF doubleprime。与此同时 event 仍声称把 C05 TreeG 作为 collider-shielded target 阻挡了。**屏蔽了去往 G 的某条边，不能挽救从错误 owner 出发的另一条污染路径。**

### 最小离线控制

保持 Sol 已有 Wave-1 成功 receipts 的全部内容不变，只交换 C01 和 C05 的进入顺序，再调用原 consolidation，并在同一 Wave-2 Fdouble 目标的原始数据上执行其生成的代码：

| 源顺序 | 目标 AST 实际 active keys | train RMSE | confirm RMSE | heldout RMSE |
|---|---|---:|---:|---:|
| C01 先 | `u_ctrl,z_2` | 0 | 0 | 0 |
| C05 先 | `u_ctrl,x_ind` | 1.145122 | 3.456152 | 3.165737 |

总写回成本都仍是 `8.4`，events 的 collider-shielded 标签也仍存在。也就是说，花了同样的账面成本，有同样的概念标签，却因调度顺序得到相反的知识质量。

完成顺序不是控制脚本人为制造的唯一入口：`run_m6_1_zero_leakage_ast_causal_study.py:149` 通过 `as_completed` 收集 Wave-1；后续原序送入 consolidation，因此网络/模型时延能影响科学知识状态。

### 修法与历史复用

- 为赢得确认的候选记录显式 `adopted_instance_id / source_node_id / target_role_binding / contract_digest / confirming_bundle`，再验证写回 owner；没有可证明归属的新发现，进入新候选节点或未归属层，不能强行挂到 PPR 第一名。
- 保留 PPR 的丰富召回，不要改成题号直接路由的 oracle。检索负责提出候选，实例绑定负责确认“哪一部分被用了”。
- 复用 `wormhole_forest.py:233` 的 `validate_claim_certificate_instance_identity`、`:275` 的 `authorize_abstraction_use`，以及 `m4_4_instance_role_closure_and_online_writeback.py:783` 的 `_write_back_discovered_mechanism_to_forest`。历史路径已经有明确 node/tree/operator/host bundle 的写回语义，不必新造一套靠 rank 绑定的短路。

## 五、M6-02：first-success 去重丢失有用的部分共性

`consolidate_wave1_via_ast_wormholes:460–475` 在首次找到一个成功程序后就把 source 加到 `seen_source_nodes`。同节点之后的成功程序直接跳过。这里把“节点有一个成功程序”误当成“其所有跨域接口已经被最好地概括”。

实际 Luna C02 Wave-1 两份成功模型：

- R1 的预测用了 `m_2,u_raw,t_proc`，多带一个热变量系数；当前 wormhole role map 不含 `t_proc`。
- R2 的预测只用 `m_2,u_raw`，能够被映射并在新域正确 refit。

实际 R1 先完成，`:531` 的 `used_keys ⊆ role_map.keys()` 不成立，所以不传播；R2 被去重丢弃。两个 Wave-2 C02 都失败。

离线只把真实 R2 receipt 放在 R1 前：目标 `node_C02_TreeM2_analog` 得到程序，在 W2R1 与 W2R2 原数据上分别 refit，二者 train/confirm/heldout 都为 `0/0/0`。没有修改公式、购买新样本或调用新模型。

这正是项目想利用“跨领域部分共性”但被缓存策略抹掉的实例。正确方向不是禁用带额外变量的丰富 R1：应保存同一 node 的多候选前沿，区分结构等价、可迁移部分、未绑定依赖与参数不确定性；R1 可保留为较宽模型，R2 可作为当前目标的完整可执行候选。不能仅因 R1 的额外系数在某次 fit 中接近 0 就擅自删掉变量；是否退化为子结构要有参数/证据判据。

历史可复用 `branch_reconciliation.py:716` 的语义/角色/单位/时间对齐，`:1844`、`:1961`、`:2072` 的 anti-unification 程序族和 fresh-confirmation 流程，以及 `:3888`、`:4038` 的证据驱动 reconciliation。这里需要的是“保留并选择候选族”，不是再写一个 first-success 分支。

## 六、M6-03：`fit()` 状态未接通，被误判成科学公式失败

共享 `m5_live_gpt_and_provenance.py:1126–1129` 的生成 runner：

```python
if hasattr(mod, 'fit'):
    mod.fit(data['train_rows'])
tr_p = [float(mod.predict(...)) for ...]
```

`fit` 的返回值完全被忽略。提示要求实现 `fit`/`predict`，但没有足够明确地区分“必须修改模块全局状态”“返回 fitted callable”“返回参数对象”三种协议。Luna 实际生成了常见的返回闭包写法。

### C05：只接接口即全部恢复

实际程序：`W1R2__C05_ColliderShielded_DirectX/model_step_01.py`，模型形式是正确的两个特征 `x/(1.1+x)` 与 `u`，`fit` 返回带已拟合系数的预测闭包；模块级 `predict` 仍用默认系数 `5,.7`。

| 执行方式 | train | confirm | heldout |
|---|---:|---:|---:|
| 仓库原 runner | 0.229492 | 0.196291 | 0.220350 |
| 只令 `predict = fit(rows)`（返回值为 callable 时） | 0 | 0 | 0 |

原轨迹接到错误数值后把它理解成结构失败，改写成更复杂的 decoy 模型，随后出现未定义 `_MODEL_COEFFICIENTS`，最后 abstain。这里的首次“反证”应该标注为 fitted-state 接口错误，而不是把正确机制加入 Nogood。

### C03：同样接口错误之外还存在真正的公式偏差

Luna `W1R1__C03.../model_step_02.py` 原结果 `1.588758 / 1.387670 / 1.613468`；只接闭包后变成 `.034541 / .055989 / .035168`。确认阈值是 `.05`，因此仍未通过 confirm：其分母常数也写错了。这个控制把软件错误与剩余的科学误差分开，不能写成“C03 已经被全部救回”。

### 最小修复

规定并在执行前验证统一的 fitted-model 协议：例如 `fit(rows)->FittedModel`、`predict(fitted,row)`，或兼容现有两种合法写法的 adapter。每个确认回执应记录实际调用的 `predict` 与 fitted-state 摘要。训练残差和参数初始化状态异常先归为实现/接口错误，修复后才允许把高误差用于机制反证。

这项修复跨 M5、M6 和 v18 共享文件；主报告另一部分已独立找到相同现象。适配器不应偷偷改公式，更不能因训练误差低就直接确认：仍须原有外推确认和私有 heldout。

## 七、M6-04/M6-05：“纯 AST”与“零泄漏”是两种不同承诺

### 7.1 当前合成 worker 无法强制隔离私有真值

`m5_live_gpt_and_provenance.py:1136–1139` 在宿主文件系统执行任意生成 Python，并把项目 `src` 放入 `PYTHONPATH`。虽然 payload 的 confirm/heldout 输入去掉了 `y`，同一进程仍能导入含完整生成规律、固定 seed 和数据构造函数的项目模块。

本次离线控制提交一个不做学习的 Python 模型：导入 `m6_live_gpt_causal_aufhebung_benchmark.build_m6_causal_24_waves`，用公开 task ID 和默认 seed 重构本题的全部行，再按输入查表预测。当前正式 executor 给出 train/confirm/heldout 三段 RMSE `0`。

**这是权限边界反例，不是对原始 192 episodes 作弊的指控。** 对保存的 `model_step_*.py` 做相关导入、文件访问、`sys.argv` 等全局扫描，未看到相同利用。现有分数可按已保存正常程序解释，但“百分之百零泄漏”没有被运行时保证。

最小修复是只给 worker 公开训练/候选输入和允许运行库，把私有生成器、确认/heldout 标签与评分放到不能被候选导入的独立进程/容器；worker 不挂载整个仓库。加入上述攻击性控制验证隔离，而不是再加一次 prompt 字符串筛查。

### 7.2 AST 提取器不是完整的程序依赖分析

`m6_1_zero_leakage_ast_causal_benchmark.py:80–173` 的 collector 对所有字符串 subscript/`.get` 取 key，不验证被访问对象是否为 `row`；主要只访问 `predict`，只有完全没取到 key 才退到全模块。

反例 A：真实依赖藏在 helper。

```python
def hidden(row):
    return row['z_2']
def predict(row):
    return hidden(row) - .3 * row['u_ctrl']
```

提取结果只有 `u_ctrl`，漏掉 `z_2`。这样接口完整性判断可能基于不完整依赖成立，helper 内也未完成正确目标绑定。

反例 B：参数字典 key 被误当作观测角色。

```python
P = {'z_2': 4.0}
def predict(row):
    return P['z_2'] * row['u_ctrl']
```

把 `z_2` 映射为 `z_4` 后，代码变成 `P['z_4']`，但字典字面量仍只有 `z_2`，会 KeyError。AST 语法树转换成功，不等于保持程序语义。

应建立可达调用图与 row provenance/角色数据流，标明不可静态解析的依赖，并对翻译后的程序做执行契约核验。对动态部分暂记 unknown 或请求目标 probe，不必为了让 AST 简单而禁止 helper、丰富结构或跨尺度模块化程序。旧“把 z2 文本改成 z4”的单测试依然有用，只是远不足以证明语义可运输。

## 八、M6-06：因果外推、回溯解释、多跳与根递归的证明层次混在一起

### 8.1 d-connection 不是可运输机制的等价证明

M6.1 consolidation 确实做了图搜索、活跃变量集合、role map 子集与 d-separation 检查。但 `d-connected` 只表明在给定图和条件下未被屏蔽，不能单独证明两个域共享同一结构方程、系数可重用或在 selection shift 后可运输。当前 role maps 与变量-DAG 映射主要是预注册的任务结构，没有独立检查目标 selection diagram 上对应机制的不变性。

这里应该把“有希望的结构提示”“可执行角色绑定”“目标验证过的机制”“可迁移参数”分成不同权限。弱证据仍可引导候选生成，并不需要扔掉跨域灵感；只是不能以一个布尔 connected 将其直接写成强事实。

共享 `wormhole_forest` 已有权限层级/实例身份接口，但这条 M6.1 直接 consolidation 路径未完整接通相应权限和多跳误差合成。另一个审计分支已在共同模块发现省略 edge binding、numeric authorization 与实际提示不一致等独立问题，见配套 `v18_targeted_controls_results.json`；这些修复也必须统一纳入，不能仅在 M6 新函数外补标签。

### 8.2 回溯投影证书既不够强，也没有进入新写回路径

`m6_causal_wormhole_sandbox.py:126–159` 声称验证：

\[
\sup_x \left|E_{Z\mid x}[f_{new}(x,Z)]-f_{old}(x)\right|\leq\epsilon.
\]

实际实现是最多 8 个分位 bin，在每个 bin 的均值 `x_mid` 上计算经验平均，再取最大偏差；无覆盖保证、置信区间或连续域误差上界。有限点经验一致性是有用诊断，不能称为数学证明全域上确界。

更直接的是：M6.1 文件第 50 行导入这个函数，但新 AST consolidation 没有调用。因此“旧哲学作为新机制的观测阴影被验证保留”的思想，在这条 live 更新里没有完成相应的数值认证。新程序更新成功与旧解释被合法降级，仍是两个不同事件。

修法：保存旧机制及其适用观测分布，实际计算有明确统计边界的 retrospective consistency 回执；失败时保留候选冲突，而不是自动证明式降级。先诚实标注经验校验即可，不必为了强证书删减理论丰富度。

### 8.3 Phase 0 半径与递归结果是可解释 fixture，不是一般性算法证实

`m6_causal_wormhole_sandbox.py` 的 multihop fixture 把需受影响的树放在固定子群，背景接口被设为正交/不变；因此 touched 节点为常数的结果，证明的是这个构造上的筛选器有效。它没有证明随任意图增长的最优修订半径，也没有把扫描成本、非独立邻居和递归多路径的不确定性一并上界。`visited_states=(tree,kind)` 的去重是避免循环，不等于 ATMS 最小冲突集算法。

`:535–648` 的 recursive benchmark 则直接列出四种策略的选项、代价和乘法收益，例如 `19.0 C` 对 `3.7 C`。没有一般 AND/OR 搜索器实际从未知叶、备选 probe 和根效用中求出这个最优选择；代价优势是 fixture 设定带来的示范，不是自动规划器端到端发现的效率测量。

M6.1 的 C04 比此 fixture 更进一步：precondition 不满足时会做最多 3 层图 BFS，给 LLM 组件，而不是完整根方程。但 BFS 由程序执行、组件组合由 LLM 完成、根验证是否真实闭合，需要分别给出证据，不能统称整个递归机器已完成。

另外，`:712–722` 把 `pilot_means` 用于前提判断；若一半行满足、一半行严重不满足，平均值可能掩盖情境分裂。当前固定 `p_weather=.12` 的数据没触发这一点；这是混合 regime 下明确的后续风险。应按上下文/切片判定前提，保留多情境局部树，而不是用一个均值吞掉它们。

## 九、M6-07：probe 语义与 freshness 仍会误导模型

### selected observational probe 被称为 regime-breaking

`detect_observational_identifiability_hazards:594–598` 通过：

```python
probe_kind != 'observational_L1'
```

选择所谓能打破共线/selection 的 probe。但 C03 被动增样标记是 `observational_L1_selected`，于是也进入候选。最小控制只留下这个 selected probe，warning 仍建议用 `probe_selected_catalog_more` 打破 regime。

这在科学上有误导：多采样同一个 selection slice 不会自动打破 collider bias。实际 Luna C03 后续确实追加 selected 数据并请求不存在的 `open_branch_discriminating_probe`，在 5-turn 限制内消耗机会。

修复不应仅补一个字符串黑名单。probe 的 schema 应有干预对象、可变 regime、覆盖条件、成本和辨别哪些候选的契约；基于这些字段计算候选能否被区分。在仍不可识别时保留等价候选和不确定性。

### 重复使用同一 confirmation，却叫 fresh

同一 episode 的多次候选一直用同一 `ds.confirm_rows`。本次数据有 9 个 episodes 多次使用 confirm：Luna M6 为 2，Flat 为 5，M5 为 1，Strong 为 1。后续 `refuted_on_fresh_confirmation` 状态在字面上不成立。

这不意味着独立 heldout 分数必然失效：实际正常程序的 heldout 仍没有作为反馈给 agent。但反复暴露同一 confirm 误差属于开发验证集自适应，应如实命名；fresh final confirmation 应另留一批或采用顺序有效的检验。账本同时区分第一次购入数据与重复计算，避免重复确认成本掩盖同一数据被重用的事实。

## 十、Luna M6.1 的九个失败：从机制到执行区分归因

下表简写省略共同 arm/model 路径，完整 task IDs 可在 `m6_controls/controls_results.json` 查到。

| task | 原轨迹观察 | 主要归因及可做的下一步 |
|---|---|---|
| W1R1 C01 | 取了可辨别 probe，仍拟合旧 x-Hill；train `.951642`，confirm `4.734426`，第三轮弃答 | probe 已到，候选搜索没切换到 z 结构；需要执行候选比较，不能再增加同类提示标签 |
| W1R2 C01 | 同样有 probe，但继续 x 变体；confirm `3.45755`、`3.509721`，第四轮弃答 | 科学候选探索/采用不足，未发现前两次失败的最小结构差异 |
| W1R1 C03 | 先取 unselected probe；闭包拟合结果被忽略，同时分母有偏；后续取 selected、请求不存在 probe | 接口错误 + 剩余机制误差 + probe 能力描述错位，不能全算基础模型能力不足 |
| W1R2 C05 | 第一份程序形式正确；runner 用默认参数，随后引入 decoy 和未定义变量 | 只修接口即可使原 first candidate 三段误差 0 |
| W2R1 C01 | 继承被 C05 污染的 direct-X 写回；train `.995529`，confirm `3.342476`，第二轮弃答 | 错 owner/错误传播与采纳共同作用 |
| W2R2 C01 | 同样受污染；confirm `3.456152` 后 `4.525792`，第五轮弃答 | 修归属可避免错误高置信先验；不要把错误转移视为新域真的没有共性 |
| W2R1 C02 | 只取被动 probe，第二轮不写代码即弃答 | 可迁移 R2 被 first-success 去重，缺了既有可用结构 |
| W2R2 C02 | 被动拟合 train 0、confirm `.767029`，第三轮弃答 | 同上；真实 R2 已有程序在此目标 refit 为 0/0/0 |
| W2R2 C03 | 目标已可收到 transport 代码，但仅取被动 probe 然后第二轮弃答 | 有效建议没有被执行，是候选采纳闭环不足，不是没有跨域 AST |

Sol 的成功说明这些任务及迁移思路可被更强基础模型利用；Luna 的上述差异说明算法对模型的程序接口与规划能力仍有高依赖。两者并不矛盾。要展示自改进减少这种依赖，应让先验候选、实际执行、局部验证、反馈分类和采纳形成可观测链，而不只是增加 `Nogood` 数量。

## 十一、M6-08：frontier 的根验证闸门和 M6.2 棘轮并不保证所说的性质

### 11.1 exit-0 与“当前 artifact 已验证”混为一谈

`m6_frontier_real_benchmarks.py:2186–2198` 依据 artifact snapshot 是否改变及命令 exit code 更新状态。只要命令成功且未改变 artifact，就令：

```python
artifacts_modified_since_last_verify = False
```

最小控制使用真实 `run_frontier_episode`，只替换 Docker/LLM 外部 I/O：第一轮写入错误 artifact；第二轮执行 `true`；第三轮 submit。结果为 `ended_by_finish=true`、`pullback_gate_triggered=false`、最终官方 strict pass=false。`true` 没验证任何根约束，却清了债务。

需要绑定验证回执与候选 hash、测试/约束版本、测量值和通过状态。普通读文件/环境命令仍可自由执行，只是不能改变“已验证”的资格。

### 11.2 单调棘轮仍靠自然语言

M6.2 的 `latest_m6_2_metric_note` 在 `:1907,1947,2034` 是自由字符串；没有持久 incumbent hash、结构化 worst-slice 指标、固定验证协议、候选对比与回滚事务。模型得到“atomic promotion/monotone ratchet”的指令，并不等于系统强制了这些性质。

`stress_challenge_gate` 在 `:2096–2130` 每 episode 最多触发一次，只是在还有至少 5 turns 时拦下第一次 submit 并插入提示。其后是否做了有效压力测试没有新证书要求；precommit root gate 在 `:2131–2180` 也只拦一次。因此“被提醒过”与“验证后满足了”没有被区分。

lean ≤5 Nogood 也是提示约束；append 路径没有机械容量策略。此次没有据此发现实际 >5 的 M6.2 episode，故这里是实现未保证，而不是虚构已观察到爆仓。

最小修复：保存不可变候选、验证器规范与测量回执，允许探索分支变差，但 incumbent 只在固定规则下升级；保留不满足当前 acceptance 的探索候选及其知识，避免把“最终制品不可退步”错误扩大为“科学探索不能走暂时更差的路线”。这能保留项目的探索灵活性。

主报告另行审查实际任务中是否发生退化及哪些日志支持；不要把这个控制当成所有失败的总解释。

## 十二、M6-09/M6-10：frontier 跨波知识写回没有逐条因果授权

### 固定图上两个布尔检查不筛实际 lesson

`consolidate_wave1_into_m6_forest:608–614` 在一个固定方法论 DAG 上计算“机制→泛化连通”和“观测拟合→选择 collider←离散化被屏蔽”两个布尔值。后续 `:649–661` 将 lesson 写到匹配 family 的节点及所有一跳邻居，没有把这条 lesson 的实际变量、领域假设、适用接口送进 d-separation，也没有使用每条 wormhole 的角色映射去验证 lesson。

控制输入只有一条 `RUNG_1_SHELL_OR_ENV_QUIRK`：`python alias exists -> use python3`。它仍被传播到符号动力学、算子反演、多尺度 PDE、可识别性证书等多个节点，结果同时输出 `bayes_ball_mechanism_channel_active=true` 和 `bayes_ball_collider_channel_blocked=true`。

shell lesson 可以有用，并非必须删除；问题是本来只是可迁移操作经验，却被这两个固定布尔字段包装成因果筛选通过。应给不同类别知识各自适用范围，以具体作用域/证据区分，而不是统一叫 causal reinterpretation。

`Nogood` 当前主要是模型自填的 `assumption_refuted / empirical_witness / causal_blanket_diagnosis / pivot` 文本。它可能促进有价值的反思，但没有实现一般 ATMS 的最小矛盾假设集、证据蕴含检查与依赖撤销。后续应优先把自由文本 witness 绑定实际 artifact、命令输出和假设版本，再逐步增加最小冲突处理。

### 源波官方 grader 的结果确实进入 lesson

`:619–646` 在官方 verifier 已经评分后，读取 `strict_pass` 和 `failure_taxonomy`，把 `PASSED_AFTER_PIVOT` 或 `FAILED(category)` 写进跨波 lesson。因此 Report 27 的“只巩固模型自己的容器观察”不精确。

这在明确的“Wave-1 是带反馈训练集”设定下完全可以允许；它不等于泄漏 Wave-2 的测试答案。但应列出源反馈种类与预算，并让比较组拥有等价训练反馈。当前 Bare 没有同内容 Wave-1 记忆，Strong/Flat 也不自动复用自己的成功程序，结果同时包含“有跨题记忆”和“记忆的哲学树组织方式”两种效应。至少需要 flat learned-memory、普通反思记忆和现结构的等内容对照。

### lesson 截断再引入完成顺序偏差

每节点 bucket 在 `:660` 只留先到的 4 条 lesson；原 runner 用 `as_completed`。后到的更相关/更高 rung 经验可因满额被忽略，且 event 的 `updated_nodes` 仍列出目标节点集合，即使 bucket 已满没有实际 append。应分别记录 offered、accepted、rejected-with-reason 和新的 snapshot hash，按证据质量/适用性/冗余性做有记录的选择。

主审计补充的真实 M6.2 完成顺序反转控制显示：本批各桶未超过 4 条，**只改变 lesson 顺序，没有改变保留集合**。因此桶截断是此实现的风险，不是本批已证实的知识丢失；AST 部分的 owner/first-success 丢失则有实际回执与数值反事实支持，两者不可混为一谈。参见 `frontier_controls_results.json`。

## 十三、M6-11：独立审计脚本的自述超过实际做的检查

`scripts/audit_m6_1_zero_leakage.py:8–19` 声称检查 z2→z4 反事实、完整 writeback 成本、C03/C04 prompt spoiler、frontier `/tests`/`/solution`/rubric 字符串、equal max_turns/timeout 等。

实际 `audit_track_a:61–94` 做的是 SHA、manifest 自报 auth/zero_proxy、unit_row、DAG hash chain、空代码不能写回；没有运行变量替换，没有核对所有 prompts，没有重算写回总价。变量替换等测试另在 tests 存在，不能据此说本审计入口本身已全部执行。

实际 `audit_track_b:97–119` 做 SHA、manifest 自报 auth、DAG，以及 consolidation 文本不能出现 `answer_rubric`/`ground_truth` 两个字符串；没有检查 `/tests`/`/solution`、prompt 全文、相等 turn/timeout，也没有查询任何 Docker 容器。Report 28 开头把“0 leaked Docker containers”紧邻该 audit exit-0 描述，脚本本身并不能证明这个结论。

manifest 写着 `zero_ruoli_proxy_verified=true` 不能单独证实网络没有经过代理；同理，不含 `ground_truth` 字面量不能证明没有 grader-derived feedback。应把每条主张与具体已执行检查、结果和局限一一对应；没有执行的项不得使用 verified 状态。

本次独立复算确认了 719/192 的完整性等真实部分，并补了反事实控制。结论不是否定密码学审计的价值，而是其证明对象必须精确：它保证保留下来的内容未被悄然更改，不保证 omitted 内容完整、语义正确或私有真值不可达。

## 十四、M6-12：可重复性、外部隔离和日志完整性

### 14.1 cache 没有绑定学习状态

`run_m6_frontier_scout_study.py:198–206` 见到 cached summary 的 `provenance_verified=True` 就接受，没有用当前源代码、模型参数、prompt、image digest、task版本、forest snapshot 一起构成 cache key。AST study `:99–104` 同样依赖保存的 chain verification 标志。

当 Wave-1 的完成顺序已经影响写回，重启只复用部分 Wave-2 的情况尤其需要 snapshot binding；否则一个 run 目录可能混入不同知识状态下的结果。这里尚未证明已提交包必然发生了这种混合，因此应列为可复现性风险并补核验，不能直接宣布旧数据无效。

### 14.2 “lossless older summary” 实际截断

`m6_frontier_real_benchmarks.py:1535` 注释称完整最近窗口和无损旧摘要，但实际按字符截 stdout/stderr：近轮约 6000/2500，旧轮约 1400/700，命令也截到 500。写给人的 Markdown 只保留约 stdout 2500/stderr 1500；DAG command preview 与 stdout/stderr 也更短。

M6.2 的 `latest_m6_2_metric_note` 虽进入下一轮决策，却没写入对应 `turn_entry`/DAG/Markdown 的结构化字段。原始模型返回曾在运行内存里出现，不等于 evidence bundle 保存了完整决策依据。

需要把完整原始 command/output/assistant JSON 存为内容寻址 artifact，并让 prompt 使用有预算的视图、让 DAG 引用原始哈希。给模型的上下文可以压缩；用于审计的原始证据不应随着相同截断丢失。主报告还发现单文件 artifact snapshot 扩展名过滤的问题，另有专项控制，不在此重复推导。

### 14.3 verifier 隔离有进展，但不等于完全 hermetic

TB 的 agent/verifier 分容器、只复制提交 artifact，是实质性的正确边界。与此同时 `start_detached_container:1141` 未设置 `--network none`；Bio 输入的 chmod 并不是只读挂载，`allowed_domains` 元数据也没有在执行层强制。没有发现原轨迹借此读取答案，因此应准确说“隔离未完全保证/缺少断言”，不要凭理论可能性判作弊。

`exec_container_bash:1203–1232` 的客户端 `TimeoutExpired` 没有明确清理容器内进程组；长命令是否继续后台运行需要 Docker 行为实测。不能仅凭杀了客户端进程就把“计算资源已经终止”当成已证事实，也不应在未实测时说一定留下进程。

外部 benchmark repositories 与构建依赖没有完整纳入这次提交的可离线运行环境，manifest 也未固定所有外部 commit 与镜像 digest。因此本次不能复跑官方 Docker tests。主报告已进一步核对 AMR 官方资源时限等重要差异；这些差异要在统一结果面板中区别成 infra-invalid 与 candidate failure。

Bio judge 的语义判分 fallback 使用 `gpt-6-luna`，不完全等于上游原始 judge 的固定实现/版本；涉及 judge 调用的成本未完整进入 episode ledger。精确匹配结果不受此影响，语义 judge 分数则应注明协议、模型和失败处理，避免用“official”一词隐去比较差异。

## 十五、保留项目核心能力的改进顺序

### 第一阶段：不花新模型费用，先修已证实的执行错接

1. 修 `fit` 协议；重放保存的全部 authored models，列出接口修复造成的数值变化，但保留原版端到端分数作为历史结果。
2. 将检索、采纳、确认、归属与写回分开；绑定真实采用实例，不通过 taskID 偷塞正确 owner。
3. 保留同源多模型/部分接口，修 first-success 去重；所有 consolidation 都对输入顺序确定，明确保留和淘汰规则。
4. 把执行验证从 exit-0 改为 candidate-hash-bound receipt，真正实现制品 incumbent 的 promotion/rollback。
5. 记录真实完整决策与 artifact，校正 audit 自述及统计面板的不可评价/超时分类。

此阶段优先复用仓库已有的 instance certificate、online writeback、anti-unification、evidence reconciliation，不另建只有同名概念的平行实现。

### 第二阶段：把“部分共性”变成有条件、可组合的实际执行对象

一个候选应携带结构 AST/IR、已绑定角色、未绑定变量、单位/时间尺度、前提与适用域、来源机制实例、证据等级、参数后验或区间、误差预算。不同层次的共性可以继续被利用：

- 仅有结构相似：提出候选或安排 probe；
- 角色/单位/时间映射成立：构造目标可执行骨架；
- 目标条件满足但参数未知：只拟合目标参数；
- 部分模块得到验证：局部提交，其余作为开放子假设；
- 多跳组合：合成误差和前提，在根目标上重新确认。

这种设计允许丰富的跨尺度和跨领域组合，同时避免“一个 boolean causal=true 就把所有层的权力都开满”。它强化项目的能力，而不是为了可解释性把结构剪简单。

### 第三阶段：用小而有区分力的实验验证真的进步

不建议再马上堆更多近似同分布 repetitions。先冻结上述接口后，用同源经验、同样 token/工具预算比较：

| 比较 | 识别的问题 |
|---|---|
| 普通 learned flat memory vs 哲学树记忆 | 收益来自有记忆，还是来自组织/组合方式 |
| 正确绑定 vs 原 rank-1 绑定 | 知识 owner 和权限闭环的影响 |
| 多候选保留 vs first-success | 部分共性被丢弃的代价 |
| 有 target role/certificate vs 仅 d-connected | 结构灵感与可执行可运输的差别 |
| 相同学习快照的随机完成顺序 | 实验是否真正与调度无关 |
| rich temporal modules 开/关，其他内容固定 | 多尺度动态是否被下游实际使用 |
| 有真实 ratchet vs 只有提示 | incumbent 保持能力与探索能力是否同时改善 |

选择涉及 mixed regime、多条同源候选、helper/dynamic AST、跨时间单位变换、部分接口映射以及可误导 proxy 的题，比再重复简单已满分题更能判断潜力。先做离线固定代码回放，再做少量按事前规则冻结的端到端试验，避免凭后验改 prompt 挑成功结果。

## 十六、证据文件、执行范围与复用入口

### 本次新增离线控制

- `m6_controls/run_controls.py`：完整性重算、192 回执汇总、七个原测试、AST helper/key 控制、真实来源顺序控制、private-generator 隔离反例、probe warning 反例、frontier RUNG-1 传播和 exit-0 验证债务控制。
- `m6_controls/controls_results.json`：上述实际执行结果。
- `m6_controls/run_saved_model_rescues.py`：只使用保存过的付费模型代码，重放 fit 协议与 Luna C02 来源顺序控制；已实际执行完成。
- `m6_controls/fit_contract_control.json`、`m6_controls/luna_c02_order_control.json`：可核验原值与反事实值。

从 `/workspace/scratch/3f14ad2da0ed` 可离线运行：

```bash
python audit_m6_v18/m6_controls/run_controls.py
python audit_m6_v18/m6_controls/run_saved_model_rescues.py
```

控制脚本会在临时目录构造模型/回执，不更改被审查仓库源码，不发起付费 LLM 调用，不启动真实 frontier Docker 实验。frontier 验证债务控制明确 mock 外部 I/O，只验证原编排器的状态逻辑。

### 固定源码入口

下列路径均位于固定提交的 `Hegel Machine/src/hegel_machine/cognition/`：

| 源码 | 本文重点位置 |
|---|---|
| `m6_1_zero_leakage_ast_causal_benchmark.py` | 80–173 AST；438–562 consolidation；594–598 probe；669 owner；712–768 precondition/递归提示；后半 episode 确认执行 |
| `m5_live_gpt_and_provenance.py` | 1087–1177 authored Python runner/私有边界 |
| `m6_causal_wormhole_sandbox.py` | 126–159 retrospective projection；multihop fixture；535–648 recursive fixture |
| `m6_frontier_real_benchmarks.py` | 594–676 lesson consolidation；1141–1232 container；1530 起 history；1905 起 loop；2096–2198 gates；2300 起 summary |
| `wormhole_forest.py` | 233 instance identity；275 authorization |
| `m4_4_instance_role_closure_and_online_writeback.py` | 783 explicit-owner writeback |
| `branch_reconciliation.py` | 716 alignment；1844/1961/2072 anti-unification；3888/4038 evidence reconciliation；6044 autonomous episode |

历史检索以 `docs/hegel_machine_reuse_hub_20261006.md` 为入口，并查询了 premise/结构/研究程序相关路线。检索未命中的关键词不被当成“仓库没有该功能”的证据；以上建议指向现有源文件，避免因为新 harness 没接某能力就误称整个项目从未实现。

本报告没有声称重跑了官方 Docker、完整阅读仓库每一个历史版本的每个字节或再次购买模型计算。它所标明的实证来自固定提交的源码、已保存的完整回执集合及列出的可重复离线控制；关于一般化、复杂度、最优性和风险的判断与已发生故障明确分开。
