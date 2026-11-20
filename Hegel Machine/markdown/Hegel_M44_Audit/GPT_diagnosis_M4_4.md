**这轮的总体判断是：V17.6 / M4.4 有实质进步，上一轮两个主要 P1 问题的原始反例已经修复，在线结构写回和后续复用也确实发生了。当前适合做一轮范围明确的收尾，然后进入 M5。**

还需要补齐的地方，集中在五个具体问题：**新建粗粒化边的证书绑定、S01 的作用域修订语义、在线状态持久化、OR 动作的统一消费，以及显式角色映射的接口约束。** 我没有在本次正常主表路径中复现新的 P1 假确认；这些问题应按各自触发条件处理，不应把已经成立的结果一并否定。

同时，本轮证据支持“可验证的在线结构记忆”，但还不足以支持“所有入口完全闭合”“从两个独立领域自动形成了新同构”或更强意义上的递归自我改进。

完整材料已经整理好：

- [完整审计报告：Assumption Agent V17.6 M4.4 审计与后续建议](https://chatgpt.com/space/page_73485f480b788191bd49ca66bbd5eb53)
- Hegel_M4_4_Audit_b965f022b_20261120.zip[Hegel_M4_4_Audit_b965f022b_20261120.zip](sandbox:/workspace/scratch/3f14ad2da0ed/Hegel_M4_4_Audit_b965f022b_20261120.zip)：包含完整报告、三份专项审计、官方完整重放、独立控制脚本、账本复算和差异核验，共 313 个文件。

下面展开最影响下一步决策的部分。

## 一、本次审计和复现覆盖了什么

审计固定在你给出的 [`b965f022b`](https://github.com/erzhu419/Assumption-Agent/commit/b965f022b214f47a1341559f20041da6a224d66d)，比较基线是 M4.3 的 `346153536`。上传的工作总结与提交内对应文件逐字节一致。

本次完整审阅了十个发生变化的源码、runner 和测试文件，沿实际调用链回读其依赖，并检查了新增证据。历史部分使用提交中保留的审计、反例和能力登记继续追踪；历史 181 个版本的登记用于定位已有机制，不代表本轮重新执行了全部历史版本。

相对 M4.3 的 854 个变化路径中，主体是：

| 类别 | 数量 |
|---|---:|
| 源码、runner、测试 | 10 |
| 本轮新增证据文件 | 130 |
| 导入的上一轮审计文件 | 711 |
| 旧审计 ZIP、索引和工作总结 | 3 |

我还完整运行了相关测试和官方实验 runner，并重新核对逐条账本：

| 核验对象 | 本次结果 |
|---|---|
| V16/V17 相关测试 | 11 个测试文件，**42/42 通过** |
| 官方完整 runner | 三部分完成，退出码 0 |
| Part1 闭合控制 | 与提交 JSON **逐字节一致** |
| Part2 主基准 | **1,144 条**唯一回执，费用、状态、误差、候选数一致 |
| 噪声 fork | **192 条**唯一回执，上述结果一致 |
| Part3 在线流 | **80 条**唯一回执，费用、通过、误差、候选数一致 |
| 提交及重放的 SHA 清单 | 两套分别 **129/129 通过** |
| 全部 1,416 条回执的账本 | 分阶段费用、观测数加和及 \(C=M+0.05K\) 无不一致 |

这里需要准确区分“结果重现”和“文件完全相同”。

两套证据包各有 130 个文件，其中 93 个逐字节相同。拟合参数及其衍生汇总有极小数值差异，最大绝对值约为 \(4\times10^{-13}\)，对应的模型、实例和语义摘要也会变化。Part3 另外有两处 S03 路由差异：提交选择 W6 主节点，本次选择同一 action、同一 instance 的 W6 twin；费用和数值表现不变。这个具体排名差异的来源没有继续隔离，原始差异已经保留。

所以，本轮可以说**主表、维护收益、噪声对照、在线成本与主要行为均已重现**，不能说整个实验目录完全确定性重现。也不应为了让跨运行 SHA 相同而舍入参数、削弱实例身份检查。

审计没有修改仓库源码或提交内的实验结果。上述执行和核验记录均在复现包中。

## 二、上一轮哪些问题已经真正修好

这一部分值得先明确，因为它决定了后面的建议应该有多大范围。

| 上一轮问题 | 本次核实结果 | 判断 |
|---|---|---|
| R3 只绑定 AST，同 AST 的另一参数实例可替换 | 14 个持产物节点及 2 个 OR 分支，共 16 个动作：正确实例锚接受，错误锚拒绝；缺锚也拒绝 | **原 P1 已修** |
| 规范坐标盒检查与拟合使用不同角色排列 | 同名规范轴锁定恒等；合法映射限制拟合排列，选定映射继续约束 fresh 盒、购测和查询 | **原 P1 已修** |
| fresh confirmation 可能在验证盒外购测 | acquisition 前有守卫，确认点使用目标坐标中的合法盒交集 | 已修 |
| 细尺度矩阵和冗余预拟合 | `A12=g_sf` 接回真实方程；丢弃的 generic prefit 为零；本地计算和校验入账 | 已修 |
| 错误证书重验证先改状态再失败 | 错误实例在 mutation 前拒绝，森林状态不变；合法同步替换通过 | **重验证入口已修** |
| 迁移后摘要或 artifact 路径失配 | 三份快照磁盘和冷载检查均为零 mismatch，历史及 OR 路径可解析 | Part2 已修 |
| legacy 兼容分支绕过 strict flags | strict 模式拒绝无 bundle 产物且零 refit；宽松模式明确标记 | 已修 |
| 旧 ScaleBlind 升级重复购买已付确认点 | 原入口保留并合并已付细尺度确认点 | 已修 |
| 正式 revision 后索引仍使用旧内容 | PPR、Flat、Compiled 均刷新并收费；没有变化时不重复收费 | 已修 |
| 噪声 fork、DEV 条件不齐 | 四种子、两噪声、三臂；DEV 的噪声和 Pi 场景已对齐 | 已修 |

特别是前两项，已经进入实际消费者，不能继续沿用“只加了字段、没有接线”的旧判断。实例校验和受限角色拟合的接线，可以直接在 [R3 实例与动作处理](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L562-L678)及[角色受限拟合与确认](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py#L1420-L1478)中看到。

但“重验证入口已修”和“所有新建、加载、消费入口都满足相同约束”，仍是两个不同的完成度。下面的问题主要就落在这里。

## 三、当前应处理的五个具体问题

以下按 **P2：已实证、范围明确的接口或交付缺口**处理。我会同时说明触发条件和对现有主表的影响。

### 1. 新建粗粒化边还可以绑定另一个实例的证书

`revalidate_vertical_edges_with_certificate` 已经检查节点、OR 分支与证书的实例一致性。

但 `add_coarse_graining_edge` 没有共享这一检查。新建时，它使用类似 `node_digest or certificate_digest` 的方式给边填写实例摘要，随后直接加入森林。后续 `fine_scale_children_of` 主要检查端点版本和 stale 状态，也没有阻止这类关系失配。[新建及消费入口](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L904-L960)；[已修的重验证入口](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L1284-L1308)。

我用的控制没有伪造证书，而是取提交中两个真实产物：

- Stage0 Pi 证书对应实例 `416e0bd1…`。
- Stage2 Pi 节点对应实例 `058680e5…`。

在独立森林副本中移除原 Pi 粗边，再用正式新增接口把旧证书接到新节点，调用成功，边被视为 live。此时 node/edge 指向一个实例，内嵌 certificate 指向另一个实例。

这条错误关系还会影响实际执行：

| 原始 T21 控制 | 选择通道 | 测量费 M | heldout RMSE |
|---|---|---:|---:|
| 没有 Pi 粗边 | `micro_fine` | 70 | 0 |
| 加入跨实例证书的粗边 | `coarse_macro` | 21 | 0.008633 |

**它确实能凭不匹配的证据开通廉价粗通道。** 本例的两个模型比较接近，误差仍在容差内，所以这不是新发现的假确认；正常提交中的粗边也没有这个身份失配。

另一个值得补齐的点是：把这个错误绑定的森林交给 `verify_forest_snapshot_digests`，仍会得到 mismatch 为零。当前 verifier 检查的是各对象自身摘要，尚未检查对象之间的关系。

**建议：** 直接复用现有重验证中的纯校验逻辑，让新建、重验证、归档关系检查共享同一个 node—edge—certificate 一致性约束。失败应发生在边列表、版本和 epoch 变化之前。现有 S0/S2 控制足以验收，不需要新造一套证书体系。

### 2. S01 实际没有完成所称的作用域修订，而且失败回执也会触发更新

S01 的意图是加入 `b<=1.6` 的约束，元数据中也写了 `b_bleed_ceiling_safe`。

但执行路径没有消费这个条件。节点 preconditions 保持不变，实际新增的是：

```python
protocol_spec.online_bleed_verified_max = 1.6
```

这个字段在当前 src/tests 中没有授权消费者。并且该分支没有像 W6/W7 写回分支一样，以 `rec["confirmed_ok"]` 为提交前提。[S01 写回路径](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/m4_4_instance_role_closure_and_online_writeback.py#L1099-L1127)。

两个控制把问题分开验证了：

1. **失败条件控制：** 仅把 S01 回执的 `confirmed_ok` 改为 `False`，在进入 S02 前停止，W4 仍然由 v1 升至 v2。
2. **实际授权控制：** 对真实 W4 运行授权器和 R3，修订前后，`s=4,b=1.5` 都获授权；超过所称边界的 `b=1.7` 也都获授权，预测相同。

所以，S01 已经实现了一个真实的版本变化和索引刷新事件，但没有实现一个会改变合法行为的作用域变化。

此外，S01 的 pilot 中 b 最大只有 1.1，heldout 中最大只有 1.0。现有观测本身也没有把 1.6 识别成经验边界。

**建议根据这个步骤真正要证明什么来收敛：**

- 如果目的是验证索引能响应在线元数据变化，就把它准确命名为“协议元数据更新”。
- 如果目的是验证经验驱动的作用域修订，就需要实际边界证据、可执行 predicate/box、角色映射和合格回执后的原子提交。

仅仅把常量 1.6 从 `protocol_spec` 移到 preconditions，可以补上执行效果，但还没有证明边界是从经验中获得的。

### 3. Part3 的新知识没有持久化，实验结束后无法恢复在线终态

这是当前最影响“持续学习”下一阶段的交付问题。

官方 runner 在 `TemporaryDirectory` 中运行在线流，归档时只复制 Part2 的 Stage0、Stage1、Stage2 workspaces 和三份快照。在线函数返回 summary 和 receipts，没有保存或返回最终森林。临时目录退出后，再写入 Part3 的统计文件。[runner 归档范围](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/scripts/run_v17_6_m4_4_closure_and_writeback_study.py#L85-L173)。

因此，提交的 130 个证据文件中：

- 在线 W6/W7 的 artifact：**0**。
- post-stream forest snapshot：**0**。
- 三份已有快照仍为 **8、15、15 个节点**，没有在线新增节点。

完整重跑也重现了这个归档缺口。

**进程内后继任务的复用是真实的；进程结束后的终态恢复尚未实现。** 重新跑一遍发现过程会重新付费，不能代替恢复已形成的知识。

这里已有足够的基础设施可复用：本版已经把相对定位符和 Part2 冷载修好了。下一步保存最终森林、W6/W7 bundle、来源和写回事件，再删除原临时目录，在新进程中继续一个后继任务即可。

如果 M5 要恢复完整学习进程，还应明确 episode、已付观测、账本、问题队列和流游标由哪个现有 checkpoint 层保存。这个任务可以接现有存储能力完成。

### 4. OR 的 `selected_action` 尚未贯穿重新拟合召回

R3 现在已经先授权，再使用 `selected_action` 加载对应实例。

M4.3 的重新拟合召回仍然先按节点顶层 action 加载 operator spec，然后才调用 gate；取得 `selected_action` 后，也没有改用它。M4.1/M4.2 的旧消费者存在同类模式。[实际调用次序](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py#L1276-L1319)。

有效控制使用普通 Monod OR，不涉及 Pi 或尺度证书：

- A 是当前提交的增益 6 实例。
- B 是提交内旧审计保留的、真实确认过的增益约 12 实例。
- 两者 AST 相同、实例不同，通过正式 `register_node` 建立互斥条件分支。
- 当前目标适用 B。

结果是：

| 条件 | R3 | M4.3 重新拟合召回 |
|---|---|---|
| A/B 都可用 | 正确加载 B | gate 选 B，loader 实际先载 A；同 AST 重新拟合后仍能正确完成 |
| 仅未匹配的 A 不可用，B 完整 | 仍正确加载 B | 在 gate 前就跳过整个节点 |

禁用 fallback 的隔离控制中，后者返回 `scoped_unknown`。这说明适用分支的可用性仍可能被无关分支阻断。

这里的重点是**动作来源和分支独立性**。同 AST 的 A 在重新拟合后可以得到正确参数，所以不能把“加载了 A”直接等同于数值必错。

**建议：** 先形成当前请求的“有效动作”，再让实例摘要、roles、spec、路径和执行计划统一消费它。用“B 完整、A 不可用”的普通 OR 正例验收即可。

### 5. R3 的显式 `role_map` 仍可覆盖规范轴语义

新 helper 已经限制：规范轴同名时只能采用恒等映射。

但 R3 直接接受调用方给出的显式 `role_map`，没有共享该限制。保持物理点和 oracle 不变，在：

```text
u_drive = 2.7
T_horizon = 0.47
```

这个控制点上：

| 调用方式 | 路径及结果 |
|---|---|
| 规范名称，不传冲突映射 | 下钻 fine，预测 3.22880775，正确 |
| 匿名名称，提供正确映射 | 下钻 fine，同样正确 |
| 规范名称，显式互换映射 | 选择 coarse，预测 0.96967258，误差 2.25913517 |

最后一项回执为 `dispatched_ok`，但 `within_tolerance=False`。[R3 盒检查与映射入口](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L498-L563)。

这个反例依赖调用方主动传入与规范名称冲突的映射；当前主表没有这样调用。而且盒检查和执行使用的是同一个映射，**不是上一轮“两阶段各选一个排列”的原 P1 再现**。

建议统一公共角色规范化入口：规范名字的语义不能被冲突映射覆盖；匿名轴继续支持经过验证的合法映射。这样与本版工作总结的承诺一致。

## 四、Part3 的在线学习：真实成果与证据边界

### 1. W6/W7 的结构形成和后续复用是真实的

这是本轮最有价值的正结果。

S02 的 W6 实际由：

```text
atomic_michaelis_1d__comb_quad_inh
```

形成一维 Haldane 抑制结构。

S09 的 W7 实际由：

```text
atomic_exp_relax_2d__comb_shunt
```

形成应力松弛所需结构。

这条路径使用当前已付 prefix、Stage0 原子基元、已有组合子、实际拟合和独立新确认，没有把 W6/W7 的预制复合真值直接放入候选池。新加的 `round2_1d_unresolved` 是通用残差触发，各臂共用。[组合搜索变化](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/m4_compositional_discovery.py#L790-L818)。

后继 12 个目标使用新节点的结构，在当前目标 pilot 上重新拟合，再购买 disjoint fresh confirmation，最后验证 heldout。它证明的是**结构知识复用**。流尾对 W6/W7 原参数的 zero-fit 检查，证明的是另一项能力。

完整 20 步的结果如下，数据来自[在线流汇总](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/docs/evidence/v17_6_m4_4_instance_role_closure_and_online_writeback_20261120/part3_m4_4_online_writeback_stream_summary.json)，并已与[逐条回执](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/docs/evidence/v17_6_m4_4_instance_role_closure_and_online_writeback_20261120/part3_m4_4_online_writeback_receipts.jsonl)复算：

| 指标 | PPR PoolOn | PPR PoolOff | Compiled Live | Frozen Open |
|---|---:|---:|---:|---:|
| 通过 | 20/20 | 20/20 | 20/20 | 20/20 |
| 假确认 | 0 | 0 | 0 | 0 |
| 后继 12 题召回复用 | 12 | 12 | 12 | 0 |
| 自动刷新次数 | 3 | 3 | 3 | 0 |
| 评估候选数 | 42 | 42 | 51 | 156 |
| 总测量费 M | 603 | 603 | 603 | 651 |
| 总计算费 K | 495.58 | 495.58 | 504.67 | 1236.12 |
| 逐行汇总总 C | **627.7785** | **627.7785** | **628.2335** | **712.8068** |

相对 Frozen：

- 两条 PPR live 臂各净省 **85.0283 C**。
- Compiled live 净省 **84.5733 C**。
- 三条 live 臂都少支付 **48 M**，并避免了大量重复搜索。

所以，“新结构进入历史后降低未来求解开销”已经有实际证据。Compiled 也完成了 12 次复用，这进一步说明首先成立的是**保存和消费新结构的价值**。

### 2. PPR 的优势是 0.4550，原因明确，但幅度很小

PPR 相对 Compiled 的流内总成本优势为：

\[
628.2335-627.7785=0.4550
\]

归因也可以从账本直接拆开：

- 少评 9 个候选，节约 **54 K**。
- 检索及通道预检多花 **44.91 K**。
- 净节约 **9.09 K**。

候选差来自 S12–S15 每题少两次，以及 S18 少一次。其余 15 步 Compiled 的费用更低。按未舍入 K 差计算是 0.4545 C，逐条费用四舍五入后再汇总为 0.4550，属于正常舍入差异。

这相当于 Compiled 流内总成本的约 **0.0724%**。适当的表述是：**在当前任务流和代理计价下，PPR 有小幅、可解释的候选排序收益。**

这里还有两点成本口径：

第一，Part3 没有计入流开始前的初始索引构建。补上当前实现的初建费用后：

| 臂 | 含初始索引的总 C |
|---|---:|
| PPR Live | 628.7635 |
| Compiled Live | 629.2535 |

PPR 仍省 **0.4900**，所以这个遗漏没有改变排序。

第二，这个优势取决于候选拟合与检索的相对定价。固定本次执行序列，令候选单价乘以 \(r\)，流内净计算收益为：

\[
54r-44.91
\]

临界点为 \(r\approx0.8317\)：候选单价从 6 降到约 4.99 K 时，优势消失。这是计价敏感性分析，并非观察到了另一轮实验的排名翻转。由于两臂 M 相同，把全部 K 统一乘同一个权重，不会改变两者的优劣符号。

### 3. Pooling 改变了独立查询排名，但尚未产生端到端收益

流尾的三 cue 独立查询中：

- PoolOn 的 Top1 是新的 W6 twin。
- PoolOff/Compiled 的 Top1 是旧 Murray 节点。

这个控制证明特定图上 pooling 可以改变排名。但完整 20 步中，PoolOn 和 PoolOff 的候选数、费用逐条相等，净收益为 **0**。上述 0.4550 也被 PoolOff 同样取得，因此不能归因于 pooling。

工作总结已说明，实际 pilot 产生的额外次线性 cue 使端到端打平；这一限定与回执一致。[独立 pooling 查询的位置](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/m4_4_instance_role_closure_and_online_writeback.py#L1203-L1210)。

下一步若要证明 pooling 的任务价值，应该让这种排名差异进入真实目标的候选试验、拟合、确认和总费用。继续扩大独立 Top1 查询数量，回答不了这个问题。

### 4. Twin 当前代表同一实例的别名，还没有第二领域的独立经验支持

在线 helper 把同一个 action、参数、instance digest 和 artifact 路径复制到两棵树。第二个节点没有独立 source episode，却直接具有：

```text
SUPPORTED_IN_SCOPE
local_solver_guidance
```

角色映射固定为 `identity_canonical`，hop error 固定为 `0.01`，source episode refs 未填写。树名、cue 切分和 distractor hub 由脚本安排。[写回与 twin 构造](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/m4_4_instance_role_closure_and_online_writeback.py#L703-L811)。

这种构造可以合理地表示**同一已确认实例的两个检索别名**，也可以作为受控检索压力场景。它尚未提供第二份独立领域经验。

因此需要在知识表示中分清三个对象：

1. 可以共享的数学结构或方法。
2. 已校准、已确认的具体物理实例。
3. 已在第二领域确认的实例及跨域关系。

当前后继目标的重新拟合和 fresh confirmation 提供了保护；正式 S04 等目标的成功记录成立。风险在于把同源别名进一步解释为“另一个物理实例已经获证实”，并据此直接 zero-fit 授权。

仓库已有更合适的形成路径：M2 的 `open_workspace_wormhole.py` 会分别运行各域 source episode，检查通过和 AST，再按结构分组；至少两个节点才建立关系，误差和 episode refs 来自实际两域证据。在线 helper 可以接回这条路径。[历史两域形成机制](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L1730-L1918)。

另外，12 个后继目标的 `non_direct_walk_win` 全为 false。本轮不能进一步归因出“这些成功必须依赖横向虫洞”。

### 5. 保持性记录和写回自动化还需要更准确的口径

流尾历史 R3 的十个请求包含九个数值请求和一个拒绝请求，覆盖六个节点，没有覆盖本次 S01 修订的 W4。

`all_routes_exact` 检查的是 selected node。数值通过率、最大误差、正确拒绝是另外的返回字段，但 Part3 summary 只保留 routes 与 any_fit，没有保留完整数值质量字段和十条回执。[流尾检查与保存字段](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/m4_4_instance_role_closure_and_online_writeback.py#L1167-L1241)。

本次没有发现实际遗忘。应补的是证据记录：保存已有完整返回值，断言数值、拒绝和零拟合，再加上真正被修改的 W4 边界即可。

写回触发也仍然预登记在 S01/S02/S09，其余步骤 `wb_meta=None`。它已经是一条包含真实发现的在线流原型；推广到一般在线学习时，应由 confirmed artifact、novelty、证据范围和运行结果触发提交。

## 五、Part2 主表仍成立，但收益应按机制解释

每臂 96 个域内目标、8 个域外目标。全部域外目标返回 scoped unknown，训练前缀与确认集的最大重叠为零。以下是[主基准汇总](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/docs/evidence/v17_6_m4_4_instance_role_closure_and_online_writeback_20261120/part2_m4_4_continual_benchmark_summary.json)的完整主要结果：

| 臂 | 域内通过 | 假确认 | 每题 C | 含前期费用的 96 题总 C |
|---|---:|---:|---:|---:|
| Hegel Unified | 96/96 | 0 | 32.4651 | **3768.9262** |
| Walk NoPool | 96/96 | 0 | 32.4651 | 3768.9262 |
| Compiled SameForest | 96/96 | 0 | 32.3946 | **3763.9730** |
| Hegel NoMaintenance | 96/96 | 0 | 37.6469 | 4047.9281 |
| AllHistory BestPilotRMSE | 96/96 | 0 | 41.5625 | 4423.8250 |
| Alpha0 PoolOff | 95/96 | 0 | 39.5510 | 4449.1725 |
| Alpha0 PoolOn | 95/96 | 0 | 39.5552 | 4449.5727 |
| Flat BM25 | 95/96 | 0 | 39.5694 | 4450.9390 |
| Frozen Stage0 OpenRediscovery | 95/96 | 1 | 44.5023 | 4516.2720 |
| Stratified DEV FirstAcceptable | 96/96 | 0 | 38.1667 | 4565.8250 |
| Frozen Stage0 NoDiscovery | 16/96 | 0 | 35.8351 | 3684.2220 |

这里有几个会影响结论的细节。

**第一，Part2 的 Compiled 全周期仍比 Hegel 便宜 4.9532。** 两者均有 87 次 recall、31 次非直接图召回成功，其余九次是全历史候选池回退，回执为 `fallback:op_composed_haldane_shunt_2d`。这九次不应计为新开放发现。

Part2 与 Part3 分别呈现“Compiled 略便宜”和“PPR 略便宜”，对应不同任务流，应并列保留。

**第二，Frozen Open 从 93/96 提高到 95/96，有可追踪的实现原因。** 新增的一维未解决残差通用组合尝试，让两个 MurrayCapped 目标获得了足够通过本题容差的 capped Michaelis 近似。困难目标没有被移除。另一个目标仍保留 false confirmation，heldout RMSE 为 0.049667。

**第三，维护的净收益成立，但高度集中于 Pi 的 hydraulic adiabatic 子域。**

相对 NoMaintenance：

\[
\text{后续毛节省}=497.4519
\]

\[
\text{扣除 Stage2 费用后的净收益}
=497.4519-218.45
=\mathbf{279.0019}
\]

其中：

| 来源 | 毛节省 C |
|---|---:|
| 8 个 hydraulic adiabatic 目标 | 491.2920 |
| 8 个 thermal quench 目标 | 0.0920 |
| 其余 80 个目标合计 | 6.0679 |

Pi 合计贡献 **98.7802%** 的维护毛收益。主要机制是 hydraulic 子域获得了合法、低成本的粗通道。

工作总结已经主动给出这个条件，我同意该限定。若还要把收益分配给某一项归档或 fidelity compression，需要对应单因子对照。当前成立的是联合维护结果。

固定顺序下，相对 NoMaintenance 第 46 个目标回本；相对 Frozen Open 第 41 个目标回本。目标顺序和高收益任务比例都会影响这一数值。

**第四，噪声与 DEV 对齐后的结果值得保留。**

四种子、每种子八个 onset 目标、三臂、两个噪声条件，共 192 条 fork；每个条件下各臂均为 32/32 通过，假确认和重叠为零：

| 条件 | Hegel 平均 M / C | BestPilotRMSE 平均 M / C | DEV 平均 M / C |
|---|---:|---:|---:|
| 无噪声 | 28 / 29.2435 | 28 / 32.9250 | 32 / 33.0250 |
| Gaussian σ=0.002 | 29.25 / 31.2529 | 30 / 37.0250 | 32 / 33.0250 |

噪声下，Hegel 相对 BestPilotRMSE 每题少 0.75 M、5.7721 C；Hegel 自身也因噪声增加了成本。四种子扩展增强了有限受控证据，但这些仍是一条形成历史上的参数目标变体。[完整 fork 回执](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/docs/evidence/v17_6_m4_4_instance_role_closure_and_online_writeback_20261120/part2_m4_4_fork_noise_receipts.jsonl)。

## 六、放回你的哲学目标，本轮推进到了哪一步

按你给出的工程方向，本轮已经建立起一个有实际内容的过程：

**已有原子概念参与组合，形成新结构；新结构通过观测确认后进入历史；后续任务消费这段历史，并减少重新求解的代价。**

这与“新假设来自已有概念和理论的总结与一般化”的设计方向是相符的。后继目标需要重新拟合、重新确认，也完全可以是合理的知识增长：保存的是可迁移结构，实例参数和适用条件继续接受经验约束。

接下来最需要加强的是：**历史中的每一项知识，究竟保留了什么证据、允许什么行为、遇到反例后撤回哪一部分。**

同一个程序复制到两棵树，只增加了表示和检索入口，没有增加第二份经验支持。S01 增加一个边界字段，也只有在授权、购测和执行消费它之后，才构成行为层面的知识修订。持久化则决定这些经验能否跨越当前进程继续成为“历史”。

因此，我对本轮 RSI 进度的判断是：

- **知识层的在线进步已经发生。** 新结构进入记忆，并在后续任务中产生收益。
- **方法层的递归改进尚未由本轮验证。** 组合语法、检索规则、代理计价和写回位置仍由实现预设；还没有展示系统根据自身历史提出一种学习策略改变，再通过后继冻结评价确认其净收益。

“M4.4：可验证的在线结构记忆与 M5 桥接原型”，是与现有证据相称的定位。

## 七、建议的下一步：四个有限修复包，然后做 M5 的一条完整链

我建议保留当前主线，把收尾控制在下面四包。

| 修复包 | 主要工作 | 最小验收 |
|---|---|---|
| **证书关系一致性** | 共享现有重验证的实例检查，新建和归档也消费 | 真实 S0/S2 错误组合在新建、重验均拒绝且无状态变化；归档识别关系失配 |
| **有效动作与角色统一** | `selected_action` 贯穿 loader、实例、roles 和执行；规范轴拒绝冲突映射 | 普通 OR 的 B 完整、A 不可用仍可召回；合法匿名映射仍通过 |
| **在线修订语义与确认条件** | S01 准确命名或实现真实经验范围修订；失败不作为成功写回事由；澄清 twin 权限 | 修订确实改变域内/域外合法行为；别名保留同一实例身份，第二域授权等待第二域确认 |
| **在线终态与保持证据** | 保存最终森林、W6/W7、来源、完整 R3 结果及恢复所需状态 | 删除原临时目录后冷恢复，继续一个后继任务；旧有效知识和新知识均能消费 |

这轮收尾完成后，M5 可以先做一个小而完整的预登记实验：

1. 确认一个新结构，并由运行结果触发写回。
2. 在两个后继任务中复用它。
3. 遇到一个真实反例。
4. 生成可执行的条件修订，检验域内继续使用、域外拒绝或下钻。
5. 冷恢复后再次调用新旧机制，保留已经支付的观测、失败历史和预算。
6. 计入形成、维护、初建、刷新与写回费用，评价整条链的净收益。

优先保留当前 **Live Compiled** 强对照。若要进一步论证图组织相对普通动态结构记忆的价值，再增加能保存同样 AST 和实例的对照；若要论证 pooling，就让它改变真实任务的候选试验顺序，并加入一个跨域映射不成立的负例。

**我建议下一轮把重点放在“真实反例能否改变后续合法行为，以及这种改变能否跨进程保留”。** 这两项补上后，M4.4 已经成立的实例身份、角色约束、在线结构复用和成本账本，就能组合成更有说服力的持续学习过程。