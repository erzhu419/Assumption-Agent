## 结论

**修订后的计划总体成立，值得继续按现在的方向实现。跨时刻动态、多尺度时间折叠、跨领域部分共同结构，以及“软参数热启动＋宽域回退”，都应该保留。**

但按这份新计划自己规定的验收标准，**`63b940a03` 还不足以认定“M5 已完整收官”**。主要原因有两类：

1. **核心研究闭环仍有未接通的环节。**一些部件已经实现，但主实验通过预置机制签名、预先填写的共享结构、固定候选模型和按任务名执行的修订，跨过了计划要求它们完成的工作。
2. **存在已复现的功能错误，以及影响收益归因的对照混杂。**其中最直接的是：热启动训练通过、目标确认失败后，不会重新尝试同模型的宽域拟合；另外，探针选择没有真正控制取得哪份观测，J 流保存的新知识也没有决定后继行为。

完整报告和复现实验已整理好：

- [完整审计报告](https://chatgpt.com/space/page_d10ef83244cc81919565da55fc7aeb5a)
- Hegel_M5_Audit_63b940a03_20261120.zip[Hegel_M5_Audit_63b940a03_20261120.zip](sandbox:/workspace/scratch/3f14ad2da0ed/Hegel_M5_Audit_63b940a03_20261120.zip)

附件包含完整正文、七个复算或定向控制脚本、七份机器可读结果、复现说明和校验清单。以下把关键判断完整展开。

---

## 一、先评估改过的计划

我支持这次计划中的几个主要调整。

| 设计选择 | 我的判断 |
|---|---|
| 快动态经过条件、范围和尺度归一化验证，也能进入共同基底 | 合理，应保留 |
| 时间折叠可以产生常数、慢动力学、记忆核或潜状态摘要 | 合理，应保留 |
| 区分采样步长、内生时间常数、提取窗口、预测时距和因果截止点 | 是必要的执行语义 |
| 共同子结构同时支持因果链与并列 AND 前提 | 更符合“部分共性逐层利用” |
| 用任务风险 VOI 选择探针，并结合 HEC 停止 | 合理，且已有代码可以复用 |
| 反例可以修订最小受影响的上游子图 | 保留了理论生长和修订能力 |
| 无量纲软参数先验，加宽域回退 | 合理，不应该退回完全无信息的搜索 |
| 潜状态不强求人类命名，M5 采用有限验收范围 | 都可以接受 |

这些选择已经写进新版计划的裁决矩阵、两条核心支柱和验收合同。**没有完全采用上一轮建议，不构成问题；判断标准应当是设计是否自洽、执行是否兑现、证据是否支持结论。**[修订计划](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md#L12-L21)

计划层面还需要明确三个边界，但不需要因此削减表达力。

**第一，“先验收紧”应主要约束优先搜索，而不能意外变成最终可接受模型域的边界。**计划中的候选空间收缩与宽域兜底可以兼容：先优先搜索历史支持区域，遇到反驳再扩大搜索。当前实现中的热启动失败，正好说明这层区别不能只停留在文字上。

**第二，灰区之外的范围授权需要相应的范围前提。**一个支持点和一个反例点，不能自动证明支持点以下的整个半轴都安全；这需要已有单调性、连通作用域或覆盖证据。把条件写入现有 scope contract 即可，不需要再建一个庞大的证明系统。

**第三，有限收官可以成立，但验收项目必须在有限实验中真实发生。**四种子、固定任务族、指定封存集中的零假确认，是合理的里程碑目标；“从原始历史形成”“自动提炼共同结构”“同内容组织消融”“恢复后消费所学知识”，也应当在这个有限范围内真正执行。[生成消费链与验收要求](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/20_Hegel_M5_Final_Master_Plan_Temporal_MultiScale_and_Shared_Base_Subtree.md#L63-L133)

---

## 二、这次确实做成了什么

先把应该保留的成果说清楚。

我复算了提交中的 **六个受 SHA256 校验的文件、640 条主基准回执及八臂汇总**，全部对应得上。新增的 C01–C12 所在 pytest 测试入口也重新运行通过。此次主要验证采用定向控制，没有重新运行完整八臂基准或全部 51 项历史测试。

主基准确实进行了 `least_squares` 和 RK4 数值求解；训练、确认、heldout 的输入行有分离检查；最终 heldout 在候选通过确认后才被评价，未发现它直接反馈进该次拟合或候选选择。

几个主要实验臂的原始结果如下：

| 实验臂 | 域内通过 | 域外 unknown | 回执假确认 | 候选数 | nfev | 测量费 M | 目标复合成本 C |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hegel Full | 72/72 | 8/8 | 0 | 152 | 2,952 | 1,096 | 1,164.1795 |
| Strong AllHistory | 72/72 | 8/8 | 0 | 408 | 9,056 | 1,920 | 2,076.6160 |
| Forest Subtree PPR | 72/72 | 8/8 | 0 | 152 | 2,947 | 1,096 | 1,164.1245 |
| EqualAuth Flat BM25 | 72/72 | 8/8 | 0 | 192 | 4,087 | 1,124 | 1,203.2665 |

这里的 \(C=M+0.05K\) 是项目定义的复合计量单位。**表内数字能够对账，但这些差额由哪些机制造成，还需要另外检验。**此外，组 II 在绝热任务上的粗通道测量费确实都为 `64 M`，这项平权修订已经生效。[提交回执](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/docs/evidence/v18_0_m5_final_temporal_multiscale_and_base_subtree_20261120/part2_track1_anonymous_receipts.jsonl)

---

## 三、最重要的实现问题

### 1. 主基准已经拿到了它本来需要从历史中识别的部分机制信息

这是当前最影响实验解释的一点。

任务生成器根据真实任务族，直接填写：

- 记忆任务的 `endpoint_matched_history_gap`、`coupled_transient_lag`；
- 反转任务的 `endogenous_adaptation_candidate`；
- 外漏任务的 `mass_sum_drift`；
- 共同基底任务的 C1、C2 签名。

主求解器随后直接用这些签名进行检索、授权、选择首试模型、启用热先验和决定探针类别。**中间没有从已购买的 pilot 观测推导这些签名的过程。**[任务签名生成](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1254-L1323) · [运行期消费](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1692-L1743)

记忆任务还有一个更具体的问题：

> `mem_conv` 在生成器中被独立随机抽取，生成真值时直接作为隐藏状态 `z0`；拟合器又把同一个字段作为已知的 `z0`。

所以这部分实验验证了**给定状态特征后的动态拟合**，没有验证从过去的可观测轨迹恢复记忆状态。`tau_scale` 也直接由任务参数传入。[数据构造](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L988-L1080)

我做了两个依赖控制。

**控制 A：禁用新增时间入口。**

把 `extract_causal_temporal_regimes` 和 `predict_via_pi_fold_mode` 同时替换为“一调用就抛错”，记忆任务仍然通过：

- confirmation RMSE：`0.000022`；
- heldout RMSE：`0.000074`。

这证明主基准求解没有消费这两个新增入口。

**控制 B：只删除机制签名，保持所有数值数据不变。**

| 条件 | 候选数 | nfev | M | C | 最终结果 |
|---|---:|---:|---:|---:|---|
| 原有预置签名 | 1 | 8 | 14 | 14.54825 | 通过 |
| 删除签名 | 5 | 117 | 12 | 13.83700 | 通过 |

这个结果的准确含义是：签名已经替系统完成了一部分机制路由；删除签名后，这一例仍能靠模型枚举求解。它也说明预设增加的探针，在该例中并非达标必需。

因此，“匿名化”目前没有解决关键的信息边界。这里不能据此指控大模型预训练背题——主实验本身没有在线 LLM 发现步骤——但**清洗领域名词，并没有清洗实验器直接提供的机制标签**。

建议保留目前这个“机制特征已给定”的条件作为部件基线，同时把实际主链接到：

**已购前缀 → 状态/尺度估计 → 可追溯机制特征 → 检索与候选。**

这不要求潜状态变得浅显，也不要求取消复杂动态模型。

### 2. 共享子树的“自动抽取”，目前没有产生实际执行的共享结构

`extract_shared_base_subtree_from_sources()` 确实调用了 `anti_unify_programs()`，这一点成立。

但它接收的以下内容，都是调用者另外填写的：

- `shared_nodes`；
- `shared_relations`；
- 角色 schema；
- 开放分支；
- 来源信息；
- 参数样本。

反统一得到的 Program 主要保存为 `repr` 和参数槽位元数据，没有用于生成后面实际执行的模型。[抽取函数](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L230-L263)

主 seed 中的两个源程序只是：

```text
7.5 × u_norm
5.2 × u_norm
```

它们可以给出带参数槽的线性共同骨架，但不能自行推出 C1∧C2、`u→x→y`、饱和项、平方根补全或外漏分支。这些结构在旁边被手工注册，实际 M6 求解器则是另一个预置模型。[源程序、子树和先验构造](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1524-L1558)

此外，`source_formation_cost = 28.0` 是常数，没有对应的源探索、拟合和确认账本。

因此，目前可以支持：

> 给定已形成的结构库和参数样本之后，检索、排序及拟合能够产生收益。

目前还不能支持：

> 系统从相同原始源历史中自动提炼了这些共同结构，并已测得完整形成加复用的生命周期收益。

修复方向应当沿用已有的 `branch_reconciliation`，让已确认源工作区的 Program、前提、角色关系和参数估计真正进入抽取器。仓库已经有经确认的反统一家族和观测驱动修订接口；这里主要需要完成适配和消费。[既有反统一提案接口](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/branch_reconciliation.py#L2072-L2094)

### 3. VOI 选中什么探针，没有决定实际使用什么数据

价值器本身确实调用了现有 `value_forecast_experiments`。断点在调用它之后。

共同子树路径中：

- 如果选中 nuisance，增加 nuisance 的费用；
- 随后始终增加 `p_open` 的费用；
- 始终把 `p_open["row"]` 加入训练；
- 即使返回 HEC stop，这条路径也仍然使用 `p_open`。

反转路径中，除 HEC stop 外，也固定加入 slow probe 和两条额外动态观测；选择 nuisance 主要表现为多记一笔费用。[子树路径](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1745-L1770) · [反转路径](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1779-L1806)

我固定任务和随机数，只强制改变选择器输出：

| 同一子树任务的选择 | 实际训练数据 | nfev | heldout RMSE | M |
|---|---|---:|---:|---:|
| 有关探针 | 6 行 | 8 | 0 | 14 |
| 无关 nuisance | 与上一行逐字一致 | 8 | 0 | 17.5 |
| HEC stop，不选探针 | 与上一行逐字一致 | 8 | 0 | 14 |

反转任务也一样：强制选 slow 或 nuisance，实际都得到相同的 8 行训练数据，nfev 都是 7，M 只是从 `19.5` 变成 `23`。

所以当前 RawVariance 消融主要比较的是：

> 获得同样有效数据之前，是否被额外收取一笔 nuisance 费用。

这不能验证错误选针后的信息不足、候选更新和重新规划。

**最小修复很明确：实际采集接口只执行 `selected_probe_id`，只返回这个实验的观测。**如果第一次买了无关探针，就应该根据这次真实响应重新计算风险并选择下一针。HEC stop 之后可以继续必要的目标确认，但不能无条件购买已经决定停止的分歧探针。

### 4. J 流中新长出的知识，没有决定后继行为

这部分直接关系到项目最重要的“历史积累与持续扬弃”。

当前 J 流的执行方式是：

- J01 之前，用手写源程序和关系注册 `sub_joint_fast_slow`；
- J09 到来时，在任何 solver 执行之前，按精确 task ID 触发修订；
- 固定填写归因原因，追加 `confirmed_external_sink_leak`；
- 后继任务根据预置的 `mass_sum_drift` 直接选择原来就有的 LinearLeak 模型。

新增开放槽并没有进入实际候选生成。[J 的初始化](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1887-L1923) · [J09 修订触发](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1987-L2005)

这里我使用了**提交内真实的 post_j16 快照**做控制：

1. 经公开 revision API 删除 J09 新增的 leak 分支；
2. 删除 `sub_joint_fast_slow`；
3. 对相同 J19 数据、相同 RNG 运行求解。

结果：

| 指标 | 原恢复状态 | 删除新增知识 |
|---|---:|---:|
| 候选数 | 1 | 1 |
| nfev | 44 | 44 |
| confirmation RMSE | 0 | 0 |
| heldout RMSE | 0 | 0 |
| C | 12.71425 | 12.71425 |

**完整 receipt 和拟合调用都相同。**

这不否定 J19 的数值求解成功，但它说明：**该成功不依赖被称为“在线学习成果”的新增分支和联合子树。**

冷恢复还有一层同类问题。子进程确实读取了快照，但随后又执行 `_build_seed_forest_and_priors()`，重造参数先验。[冷恢复代码](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L2027-L2048)

另外两个控制显示：

- 清空恢复子树的来源、证据和多样性，J17 仍使用完全相同的 warm box；
- 经公开 API 将子树降权为合法的 `retrieve_candidate`，J17 仍执行完全相同的 warm prior；
- 两者仍是相同的 9 nfev 和完整回执。

因此，**恢复的来源、版本和权限还没有真正约束先验消费。**

我建议继续使用现在的 J，不必增加大量任务。先完成一条明确的链：

> 实际源经历形成结构 → 目标失败产生反例 → 竞争归因决定修订 → 新分支确认写回 → 后继真正消费 → 冷恢复仍依赖这项知识。

然后对一项明确受它帮助的后继做删除或回退版本对照，观察候选、探针、成本或完成率的可解释变化。

### 5. 当前胜幅不能直接归因于哲学树结构

这里需要分开看两组比较。

#### 组 I：Strong 的测量政策与价格不同

Strong 每题固定购买：

- 5 行 pilot；
- 4 行 extra；
- 3 行 confirm；
- 每行 `2 M`，每题固定 `24 M`。

Hegel 初始购买 8 行，同一 dataset 中的行却按普通 `1.5 M`、coarse `1 M` 计价。Strong 也没有先看已购数据，再决定是否需要 extra 的自适应过程。[Strong 策略](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1648-L1681) · [森林计价与采样](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1699-L1776)

复算 Strong−Full 的目标成本差：

\[
912.4365
=
824\text{（测量费差）}
+
88.4365\text{（加权计算费差）}.
\]

**其中 90.31% 来自测量费。**

进一步只做会计敏感性分析：保持 Strong 原来的 12 行和已经记录的拟合结果不变，仅按 Hegel 各族的单价重计，Strong 的 M 从 `1920` 降到 `1344`。

也就是说，**单价这一项就解释了原差额中的 `576 C`，占 63.13%。**

剩余差额仍然混有采样量、候选顺序和冷启动政策。这不是公平重跑的预测结果，但足以说明不能把原来的全部胜幅归因于哲学树。

#### 组 II：同证书还没有做到同发现内容、同执行政策

在 C1+C2 签名命中的任务上，Flat/Single 同时：

- 失去 warm prior；
- 被强制增加两条训练观测；
- Flat 还使用不同的预设回退候选顺序。

所以这组可以作为**完整子树模块的能力消融**，但目前不能叫“纯组织消融”。

Flat−PPR 的差额是 `39.142 C`：

- `28 M` 来自采样费用差，占 **71.54%**；
- 剩余 `11.142 C` 仍混合检索、参数先验和候选排序。

如果目标是测纯组织收益，就应把相同发现结论、参数先验、证书和分叉候选，以平坦记录同样提供给 Flat，再只替换索引或关系导航。

#### 暖冷拟合的停止规则也不对称

warm 的一次局部拟合结束后，只要训练误差达标就返回；cold 即使第一个局部解已经非常精确，也固定执行其余四个起点。[拟合器](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L409-L443)

同一个模型、同五行数据的控制：

- warm：`8 nfev`；
- cold：`10 + 17 + 11 + 15 + 9 = 62 nfev`；
- 第一个冷解训练 RMSE 已约 `1.07e−9`。

这个例子在共同达标规则下支持的是 **8 对 10**，不能把 **8 对 62** 全部解释成历史先验减少了搜索难度。参数先验仍然有价值，应该把它的收益与停止政策的收益分开。

### 6. 热启动确认失败后，宽域兜底没有执行——这是已复现的功能错误

这一点比表述范围更直接。

目前的逻辑是：

1. warm 训练 RMSE 小于 `0.85 × tolerance`，返回该结果；
2. 独立 confirmation 失败；
3. 跳到下一个模型类；
4. 不以 wide bounds 重试这个原本正确的模型类。

我构造了一个仍完全属于现有 C 模型族的任务，参数全部在 hard bounds 内：

```text
vmax = 7.8
rho = 0.14
local_branch_coeff = 0.32
```

pilot 和 probe 的 `s_grad` 在 1 附近小幅变化，独立确认覆盖更宽范围。两次使用同样的数据和随机数：

| 指标 | 原 warm 流程 | 仅关闭 warm box，直接宽域 |
|---|---:|---:|
| 正确 M6 模型训练 RMSE | 0.03309146 | 约 3.63e−16 |
| 正确 M6 模型确认 RMSE | 0.63387170 | 0 |
| 最终尝试模型数 | 8 | 1 |
| 最终状态 | scoped_unknown | confirmed_ok |
| heldout RMSE | 未提交 | 0 |
| 总 C | 23.64475 | 17.59025 |

这说明，**保留一个 `wide_fallback_box` 字段，并不等于实际保留了逃逸路径。**当前流程在训练点上被 warm 区域的近似解满足，确认反驳却没有重新打开同模型参数域。[训练回退条件](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L391-L446) · [确认后的控制流](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L1844-L1883)

这次失败仍诚实返回 unknown，不是一次假确认。最小修复是：

> 确认反驳 warm 后，先执行同模型 wide refit，必要时补充辨识实验，再决定是否放弃这个结构。

**软热启动应保留，回退路径需要补全。**

### 7. 灰区修复没有接回实际旧写回合同

新增灰区 guard 对显式带 `gray_zone_bounds` 的合同有效，C10 的局部测试有意义。

但旧的 `OutcomeDrivenForestConsolidator` 仍把支持点和反例点的中点写成安全上界，没有写入灰区元数据。J 中的二分只是对已知解析式计算一个数，也没有回写真实合同。[新增守卫](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L428-L473) · [仍使用中点的写回器](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_0_closure_and_continual_sublation.py#L439-L461)

我从本提交读取真实旧快照和 W4 frozen instance，使用**当前源码**执行：

```text
s = 4
b = 1.8
```

仍得到：

```text
status = dispatched_ok
authorized_tier = local_solver_guidance
fit_called = False
```

实际合同依旧是 `b <= 1.85`，没有 `gray_zone_bounds`。

这里严格只说明：**位于先前支持界 1.45 与反例界 2.25 之间的点，仍被旧合同授权。**本控制没有为该固定实例绑定同参数目标真值，因此不能把 C10 的解析遗漏项当成这个实例的实际预测误差。

修复应进入原写回器和旧状态恢复路径，不能仅靠一个手工构造灰区合同的测试关闭这项问题。

---

## 四、另有几项具体合同和数学问题

这些问题不意味着原来 640 条回执都已经出错，但会在接通真实输入之后影响可靠性。

### 联合绑定需要区分“部分相容”和“足以激活这一层先验”

当前绑定器有三个可复现行为：

- `satisfied_shared_node_ids=[]` 被解释为所有共享节点都满足；
- 只满足 C1、缺少 C2，仍返回 `valid=True`；
- 删除因果链共有的 `x_pool` 中间绑定，仍返回 `valid=True`，只是 active relation 从 3 降到 1。

时间尺度还只是检查一个 ratio 是否为有限正数，没有实际校验每条边的读出、时间锚点和尺度映射。[绑定器实现](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L2216-L2314)

当前主基准外层会检查两个签名并硬填两个 satisfied IDs，所以不能据这些 API 负控制声称原表已经发生缺 C2 的假确认。

修复时也不应把部分共性改成“整棵树全有或全无”。应当返回：

- 哪些子结构已经匹配；
- 哪些条件实际成立；
- 可以激活哪一层先验；
- 哪些开放槽仍待确认。

仅 C1 成立，可以继续使用 C1 的候选先验；C1∧C2 的强化先验，需要两者的证据。

### 同源换皮仍能提高 D_eff，并改变参数 pooling 决策

两个 lineage A、B 原本有相同 cue 时：

```text
D_eff = 1.1111111111
```

增加一个 lineage 仍然是 A、只是表面 cue 改名的记录后：

```text
独立 lineage 数仍为 2
D_eff = 1.9047619048
```

这跨过 `1.35` 门槛，实际将模式从 skeleton-only 改为 numeric warm start。原因是同 lineage 的 cue 取并集后，改变了它与其他 lineage 的 Jaccard 距离。[多样性计算及门槛](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L2148-L2205)

此外，任意对称、元素在 `[0,1]` 的相似度矩阵不一定是合法 Gram 矩阵。当前接受的三来源输入能够产生：

- `D_eff = 5.2857 > 3`；
- `D_eff = −12.5421`；
- 矩阵奇异异常。

需要修 kernel 构造或校验，以及 lineage 的稳定表示。单纯裁剪最终数值不能修复奇异问题。这些负例没有被原基准固定来源集合触发，应作为局部数学缺陷处理。

### 记忆折叠对绝对时间原点敏感

初态项使用 `exp(-t0/tau)`，没有减去该初态所属的历史起点。

控制保持观测值、参数、采样间隔和预测时距完全不变，只把所有时间戳与 `t0` 加 100：

| 条件 | 预测 |
|---|---:|
| 原时间原点 | 0.60653066，正确 |
| 全部时间戳平移 100 | 0 |
| 平移后正确结果 | 仍应为 0.60653066 |

这是明确的时间锚点错误。[记忆初态项](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L186-L212)

C03 没覆盖它，因为 C03 恰好使用斜率重构分支，卷积估计被完全覆盖，随后运行二维状态模型。**增广潜状态执行指数记忆核本身合理**；需要分别检验卷积路径、斜率路径和时间锚点。

### 子树摘要覆盖范围还不完整

当前摘要没有覆盖来源、多样性、探针签名等字段，也没有逐版本验证共享子树历史。修改这些字段，现有 verifier 仍报告零 mismatch。

这是完整性覆盖范围问题，不应夸大为已有攻击或造假。它应当与“从持久状态恢复真实 prior”的修复一起处理，使决定先验的来源、估计实例、撤回状态和版本能被核对。[快照校验范围](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m4_4_instance_role_closure_and_online_writeback.py#L236-L270)

---

## 五、测试和报告口径需要小幅修正

| 问题 | 准确影响 |
|---|---|
| Strong/Static 用 `1.5×tol` 判假确认，Hegel 用 `tol` | 相同边界误差会得到不同标签；但原表 Strong/Static 没有落入分歧区间的样本，因此不改变本批零计数 |
| Blind 真实拟合 M1，却报告 nfev=0 | 是优化次数漏记；K 已计入拟合费，不能说全部成本漏计 |
| Static 第二候选成功也报告 candidate_count=1 | 低估部分尝试次数 |
| 用 Python `hash((seed, arm))` 派生 RNG | 跨进程受 hash salt 影响，四个任务 seed 不足以保证 nfev 精确复现 |
| J 对 Strong 的条件表达式断言 | 当前 else 分支只 assert 一个正成本，未真正比较两者 |
| 同三条 confirmation 被多个候选重复使用 | 它承担 validation 职能；最终 heldout 仍独立 |

其中，J 比较断言是一个很容易修的小问题；当前证据中 Hegel 确实更便宜，但这个断言没有守住未来回归。[测试代码](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/tests/test_v18_0_m5_temporal_multiscale_and_base_subtree.py#L95-L100)

还要调整对“12 项控制全过”的解释。部分控制只证明了局部 API 行为：

- C09 用固定差值、固定归因调用修订；
- C10 对已知解析式二分；
- C11 检查反统一槽位；
- C12 直接构造误差并填写零假确认标志。

它们不能替代实际调用链中的自动归因、真实探测、结构消费和最终确认。**测试通过与本轮反例并不矛盾，因为覆盖的行为不同。**

语义孪生展示也应继续保留，但宜定位为**语义化的合成同构示例**。目前三个例子复用了模拟、VOI、绑定和拟合部件，没有全部运行同一条匿名主 pipeline；选择探针后，也没有全部完成真实执行、候选更新和分叉写回。[展示代码](https://github.com/erzhu419/Assumption-Agent/blob/63b940a03205d0f5b36bf2403d274d74709a17fe/Hegel%20Machine/src/hegel_machine/cognition/m5_temporal_multiscale_and_base_subtree.py#L2203-L2324)

---

## 六、下一轮应怎样收敛

我建议把下一轮限定为四个工作包，沿用现在的任务族和历史资产。

### A. 让实验选择真正控制取得的信息

统一 acquisition 接口，只返回实际购买的观测；从前缀形成状态、尺度和机制特征；HEC stop 停止对应探针。先用现有的一条记忆任务、一条反转任务、一条共同基底任务完成接线。

### B. 让源经历、共享结构与后继求解真正相连

从已确认源 Program 和前提关系生成子树及开放槽；实际失败经过归因再修订；后继候选引用子树 ID、版本、绑定和分支。冷恢复从持久源估计构造 prior，避免重新填入固定源参数。

### C. 修复已经复现的回退和范围问题

优先处理：

- warm 被确认反驳后的同模型 wide 重试；
- 生产写回和旧恢复状态的灰区合同；
- 部分绑定与完整先验激活的区分；
- 记忆初态时间锚点。

D_eff 和持久化覆盖随来源先验链一起修复。

### D. 在原有限评测上重新做收益归因

统一测量价格、优化停止、错误阈值和稳定 RNG；源形成费用来自实际账本；组 II 为 Flat 提供同样的结论、参数先验和证书。

在这些依赖正确之前，增加更多种子或任务数量不会解决验收问题。接线完成后，现有的 80 题/臂与 J 流足以作为下一轮主体，不需要再开启一整代无限扩张的工程。

**我对当前版本的阶段定位是：多尺度与共享子树的多个可执行部件已经具备，预置机制知识条件下的集成演示也有正面结果；自动源形成、观测驱动的联合成长，以及同内容净收益的验收仍需闭合。**

下一轮最有价值的进展，是让你希望保留的那两条核心思路——**跨时刻与多尺度的结合、部分共同结构的逐层利用**——真正决定系统买什么实验、保留哪些候选、如何修订，以及下一次为什么能够做得更好。