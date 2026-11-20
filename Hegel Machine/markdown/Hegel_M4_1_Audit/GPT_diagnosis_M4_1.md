我固定在 **`99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c`** 完成了这轮审计。我的判断是：

**M4.1 是一次实质升级，但还不能验收为“工程边界全部闭合、三支柱已经统一、图结构收益已经得到充分归因”。应当保留目前的架构，先修几处明确的执行和证据接线，再扩大 M5。**

已经整理好两份交付：

- **[完整中文审计报告](https://chatgpt.com/space/page_a0fb2b6efc2c8191b0d84b58394757f6)**：包括工程闭环、动力学、统一发现链、统计口径、成本账本和 M5 建议。
- **Hegel_M4_1_Audit_99b1fdfeb_20261120.zip[Hegel_M4_1_Audit_99b1fdfeb_20261120.zip](sandbox:/workspace/scratch/3f14ad2da0ed/deliverables/Hegel_M4_1_Audit_99b1fdfeb_20261120.zip)**：包括完整报告、三个专项报告、独立重放、逐题结果和最小复现脚本。

先说明证据强度：**33 项指定回归全部通过；四份分项 JSON 与全部 336 条目标记录，独立重放后与提交内容逐项一致。** `summary.json` 只有墙钟时间不同。因此，下面的问题主要涉及**实验究竟证明了什么，以及检查是否真正控制了执行**，不是报告数字无法复现。

本轮覆盖全部实质代码改动、研究入口、测试、工作总结及直接执行依赖，并复跑上一轮边界探针；未改动的大量历史版本沿用已有审计。完整范围清单在包内，没有把提交树盘点冒充为重新逐字阅读全部历史文件和文献。

---

## 一、这版确实完成了哪些重要进步

这些进展应当明确承认，不能继续套用上一轮的诊断。

### 1. 作者产物确实进入了森林，后继任务也确实消费它

Stage 1 生成的三个复合算子，经 `load_verified_authored_method` 检查后，实际注册进同一个 `PhilosophicalForest`。Stage 3 真正调用检索索引，再根据节点指向的磁盘产物取出算子。删除、篡改、恢复 artifact，会改变召回和费用。

这已经补上了上一轮“作者发现与森林迁移分别演示”的重要断点。确定性 provider 也确实通过 CodeAgent 工具协议读取残差、进行多轮组合；本轮不应因为没有商业 LLM 调用，就否认这一工程能力。[作者实现](https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/m4_compositional_discovery.py) · [统一链实现](https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py)

### 2. 快慢动力学和绝热减阶是真实现

Part 3 的细尺度 RK4 和 Schur 减阶来自同一套方程，消元系数自洽。我用同系统矩阵指数及减阶闭式解做了 15 点独立核验：

| 核验 | 最大绝对误差 |
|---|---:|
| 细 RK4 两个状态，对照矩阵指数 | `4.98691e-7` |
| 粗 RK4 慢状态，对照减阶闭式解 | `3.18884e-8` |

粗粒化边也确实进入 PPR，改变候选可达性。有边时，一个与 query signature 没有直接重合的 coarse 节点可被取回；删边后，它从召回结果中消失。**上一轮“粗粒化边没有实际影响”的具体判断，在这个新实验里已经不适用。**[动力学与实验入口](https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py#L85)

### 3. 多项旧边界修复成立

包括：

- 双 `predict`、顶层 `math.exp` 遮蔽被拒绝；
- 无 receipt、无 `target_id` 的布尔放行被封住；
- 原绝对路径隔离见证修好；
- `osc → shunt` 真正有四个独立参数；
- child 分支补上最终授权；
- fidelity 验证后的对象真正提交；
- 纯协议修订不再无故丢失粗粒化边；
- Frozen 保留完整 Stage 0；
- Stage 3 不再得到真实 `lens_hint` 和人工角色顺序。

问题在于：**旧反例被挡住、字段经过校验、语义已经执行，是三个不同层次。当前总结把它们合并成了“100% 闭合”。**

---

## 二、最应先修的是三处“检查通过，但执行没有兑现”的问题

### 1. `program_steps` 目前仍不是实际执行计划

新 validator 会返回：

- `fresh_points`
- `calibrate_lens`
- `verify_scale_closure`
- `operator_tag`

但旧 M4 把结果存进 `_exec_plan` 后无人读取；新 Part 4 只检查 `plan is None`，随后丢弃；Part 3 也沿用固定流程。

我在旧 M4 evaluator 的同一目标上，只把合法 `fresh_points` 从 **1 改成 2**：

- 两次都召回成功；
- 采样序列完全相同；
- 费用完全相同；
- **实际都只确认 1 个点。**

新 Part 3、Part 4 的同类问题另由源码确认。正常节点本来就声明过 `fresh_points=2`，不需要构造恶意程序才能触发这个不一致。[步骤检查器](https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/m4_multiscale_renormalization.py#L1135) · [新消费者](https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py#L1388)

**建议：**让现有操作函数消费 validated plan；如果标定、闭合本来就是不可省略的固定协议，就明确写成固定协议，删除无效的配置字段。没有必要为此开发通用解释器。

### 2. FidelityCompression 提交了析取契约，真实授权却仍按合取执行

这次“验证后的 merged claim 没提交”确实修好了。但合并器保留了 `disjunctive_branches`，同时把两组前提放到顶层；实际授权器只解释顶层前提，不消费 OR branches。

独立控制使用**完全相同的预测定律**，两个作用域分别为 `scope=A` 和 `scope=B`：

| 阶段 | A 域 | B 域 |
|---|---|---|
| 合并前，真实授权 | 允许 | 允许 |
| 合并后，fidelity 自己的 evaluator | ACC | ACC |
| 合并后，真实授权器 | **拒绝** | **拒绝** |

原因是实际执行成了：

`scope=A AND scope=B`

冗余节点此时已经归档。这是实质性的知识可用性损失。[合并与授权实现](https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py)

当前主实验的三个压缩对象具有相同前提，因此这项控制**不直接推翻那 48 个域内成功结果**。它说明一般作用域析取还没闭合。

最小修复可以很克制：统一契约解释逻辑；或者本阶段只允许等价前提的副本合并，暂时拒绝需要 OR 的组合。

### 3. 区域盒约束没有覆盖 fallback 和最终调用

报告中的 `out_of_box_query_rejected=True` 来自对 helper 的独立测试。实际 solver 只在 recall 分支检查第一个确认点，fallback 没有相同检查，最终终测也没有统一范围限制。

在原绝热盒 `T∈[2.2,5.2]` 上：

- 只把确认点改成 **T=1.2**：recall 被盒检查挡住，随后 fallback 仍通过确认并返回 `completed`。
- 保持训练和确认不变，只把最终终测点，即 `heldout_points`，改成 **u=1.5、T=0.12**：仍返回 `completed`、`confirmed_ok=True`；外部终测发现误差 **0.275920**，记为 false confirmation。[相关执行分支](https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py#L643)

这些点在原声明盒外，**不反驳原盒内成功结果**。它们反驳的是“盒外调用已经被整条执行链严格拦截”的承诺。

应当让 recall、fallback、导出模型共用范围契约。越界可以拒绝、重新验证局部范围或升级细尺度，不需要一律拒绝，更不需要证明任意黑盒在整个空间都正确。

---

## 三、当前最大的归因问题：部分测量收益来自回退分支浪费数据

下面的成本都是项目定义的抽象单位：

**C = 测量费 + 0.05 × 计算代理费。**

### 1. 粗粒化删边对照中，95.36% 的原差额来自购测规则

原结果完整复现；但对齐购测策略后，结果变为：

| 路径 | 测量费 | 计算代理费 | C | 候选数 | heldout RMSE |
|---|---:|---:|---:|---:|---:|
| 原有粗粒化边 | 72 | 12.19 | **72.6095** | 1 | 0.008792 |
| 原删边 | 96 | 35.53 | **97.7765** | 5 | 0.008792 |
| 删边，复用已付 pilot，并与当前 recall 同为一点评估 | 72 | 35.53 | **73.7765** | 5 | 0.008792 |

第三行保留了原 solver、五候选、拟合器、终测和计算计价，只取消无条件额外购测，并匹配当前 recall 的确认点数。

原 fallback：

- 无条件多买五个 pilot，其中两个已经买过；
- 比 recall 多确认一个点。

所以多出的测量费是：

**5×4 + 1×4 = 24**

原总差额是 **25.167**，其中约 **95.36%** 随购测规则对齐消失；剩余差额 **1.167**，约为有边成本的 **1.61%**。

这里应当同时保留两项结论：

- **图边确实影响召回，减少了候选。**
- **原 34.7% 成本增幅不能全部解释为必要的图结构省测收益。**

额外调用确实发生、费用也确实被记账；问题是归因范围过宽。完整调用记录在审计包的 `pi/results/` 中。

### 2. Part 4 的 Alpha0 分支会遗忘已经购买的失败确认

这比“固定 fallback 策略不够好”更明确，是数据流缺陷。

第一次高负载确认失败后，数据只进入局部 `updated_prefix`。只有第二个图候选成功，才写回主 `frozen_prefix`。Alpha0/Flat 没有成功到达 child 时，fallback 又看到最初的低负载 pilot，于是重新买 stress 点，并重复买刚才的确认点。普通历史分支却立即保留了这个反例。[失败确认与回退代码](https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py#L1459)

我重建了与提交索引规模完全一致的 **SameHistory** 森林，在同一 `U_T09` 上只保留已付反例，并让下一次确认使用未见点：

| 路径 | 观测数 | 测量费 | C | 严格通过 |
|---|---:|---:|---:|---|
| 原 Alpha0 SameHistory | 10 | 50 | **52.5363** | 是 |
| 保留已付反例、避免重复确认 | 7 | 35 | **37.3113** | 是 |
| 原 Hegel 同题 | 7 | 35 | **36.4487** | 是 |

候选数仍是 5，没有新增图边、知识或重排。

**该题全部 15 单位的物理测量差消失。** Hegel 仍有候选数和总代理计算成本上的优势，但当前报告的全部测量优势不能直接归因于图游走。

这是一题决定性控制，我没有把它外推成“修复后 48 题的新均值”。应该先修当前入口，再重算已有目标，不需要另造更大的 benchmark。

---

## 四、多机理设计确实改善了基线，但“任何静态顺序都无法绕过”仍然过强

这次不能再沿用上一轮“倒序就追平”的判断。

我保留全部 11 项历史，固定使用：

`Haldane+Shunt → Osc+Shunt → Murray 原子 → Murray+Cap → 其他历史`

所有目标共用同一个顺序，不读取单个目标标签，不改变求解器。在主种子 12 题上：

| 方法 | 平均候选数 | 平均测量费 | 平均 C | 通过 |
|---|---:|---:|---:|---|
| Hegel | **1.6667** | 35 | **36.3477** | 12/12 |
| 固定全历史排序控制 | 2.3333 | 35 | **36.4250** | 12/12 |
| 原逆序 | 3.6667 | 36.6667 | 38.5167 | 12/12 |
| 原正序 | 10.6667 | 35 | 38.9250 | 12/12 |

**Hegel 仍然小胜，应当承认。** 但相对这个合理固定排序，差距只有 **0.0773，约 0.21%**。

因此：

- 三机理均衡设计修复了上一轮的具体薄弱点；
- 原正反序的 2.17–2.58 差额，对列表策略仍然敏感；
- 当前证据支持给定任务与计价下的优势，尚不支持一般性的“数学不可绕过”。[报告中的对应主张](https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/15_V17_3_M4_1_Audit_Closure_Executable_Pi_and_Unified_Chain_Report.md)

这个排序是审计阶段形成的敏感性控制，不是预注册盲测。下一轮在 DEV 冻结合理排序或廉价重排器，再跑独立 TEST，即可检验稳健性；不需要穷举全部排列。

---

## 五、四格补齐成立，但 T25–T28 被重复计数

这里有一个明确的统计错误。

`make_m3_1_factorized_walk_suite` 的 `targets` 已经包含 `walk_targets`。新入口又执行：

```python
all_targets = [*suite["targets"], *suite["walk_targets"]]
```

因此同样的 T25–T28 被追加两遍。四个元组甚至是相同对象，第二次还使用相同工作区路径。[目标构造](https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/m3_1_walk_factorization.py#L822) · [研究入口](https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/scripts/run_v17_3_m4_1_unified_study.py)

独立保存全部 496 条记录后，确认重复记录的结果和费用完全一致。去重结果：

| 检索格 | 原总均 C | 去重后总均 C | 唯一 walk 胜出 |
|---|---:|---:|---:|
| Walk Pool | 38.4009 | **37.4591** | **14/16** |
| Walk NoPool | 38.6065 | **37.6952** | **14/16** |
| Alpha0 Pool | 43.4016 | **40.3052** | **0/16** |
| Alpha0 NoPool | 43.3855 | **40.2867** | **0/16** |

正确口径是：

- 每格 **108 个唯一目标实例**，全部严格通过；
- 四格有 496 次调用，但只有 **432 个不同的臂/簇/目标评测组合**；
- walk 胜出数应为 **14/16**，不能把 28/32 当成双倍独立证据；
- walk 子集的 87.5% 胜出率和均费比较不变；
- 固定 Pool 的总体均费降幅，由原约 **11.52%** 改为约 **7.06%**。

**游走收益的方向没有被推翻。** 但重复加权放大了总体优势。Pool 也仍然是条件性增益：walk 开启时略有利，alpha=0 时略有负担。

另一个确定的指标错误是：**Ordinary Forward 的召回率应从 32/48 改为 48/48。** 第二次历史候选确认成功时，代码漏设 `confirmed_via_recall=True`。其 16 条 SafeWalk 记录均没有 fallback，实际全部通过历史模型完成。这个更正不影响成本和严格通过率，但影响“只有森林完整复用了历史”的叙述。

---

## 六、“统一链”和 R3，目前还差什么

### 1. Part 3 的可执行 Pi 尚未进入 Part 4 持续森林

提交中的 Part 4 记录直接给出：

```text
coarse_graining_edges = 0
```

它有 13 个 concept nodes、3 条 horizontal edges、3 条 vertical tree edges。Part 3 单独建立的 ODE 森林和粗粒化边，没有被 Part 4 消费。[已提交的 Part 4 证据](https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/docs/evidence/v17_3_m4_1_unified_discovery_and_renormalization_20261120/part4_unified_continual_discovery_forest_chain.json)

准确的完成状态是：

- **作者发现 → 入林 → 跨域迁移，已经接通；**
- **真实动力学 → 绝热减阶 → 粗粒化检索，也有有效实验；**
- **两者还没有成为同一条持续对象链。**

另外，尾窗投影 `window_projected_y` 可以计算，但没有进入在线 coarse channel、区域证书和预测的下游消费。在线粗通道使用的是同模型绝热减阶；尾窗投影目前属于诊断支路。

### 2. R3 Tier 3 没有真正运行路由器

当前三层 R3 应当重新区分。

**Tier 2：**新复合族在旧数据上重新拟合，仍能表示旧关系——成立。

**Tier 1：**旧代码与旧训练前缀可以重放——也有意义。但对本次带 `fit` 的作者模型，隔离 evaluator 会重新调用 `fit`，并非冻结拟合参数的零重拟合直测。

S1 恰好把真值设为 primitive 默认参数 `6,2`，不拟合也为零误差。W1 则能清楚区分：

- 文件默认参数直接预测：RMSE **0.65774565**；
- 用旧 prefix 重新 fit：恢复约 `8.8, 2.1, 0.14, 0.76`，RMSE **3.21e-10**。

**Tier 3：**代码只是分别评估 S1、W1，再把两个通过结果 AND 起来，没有把混合旧/新请求送进 forest dispatcher，也没有记录选中的节点。检查甚至发生在新作者节点注册之前，更没有经过之后的压缩与修订。[R3 实现](https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py#L1753)

因此，“旧代码能在旧数据上重放”和“经过知识更新后，森林会自动选回正确旧对象”还不是同一个已验证结果。

最小验收应该是一组交错的旧域、新域、边界请求，统一走维护后的真实 dispatcher，记录 node/version/artifact/参数状态。无需要求不同参数化现场共享一个固定参数模型。

---

## 七、产物验证还有两处绑定缺口，适合在在线 M5 前修掉

### 1. artifact 自身有效，不代表它属于当前节点

把 W1 节点指向的文件替换为提交中另一个**完全有效的 W2 artifact**，当前 loader 会取出 W2 spec，但传入节点仍保留 W1 的 spec 与预期 digest。

缺少的是：

**注册节点的 expected digest = 当前载入 artifact 的 verified digest。**

这只是一个对象身份检查，不需要重新设计森林。[加载路径](https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py#L1280)

### 2. `confirmation_digest` 目前只要求非空

只要 source/spec/hash 内部一致，填：

```text
confirmed_ok = True
confirmation_digest = "no_confirmation_was_run"
```

即使没有执行任何确认，loader 仍会接受。新测试自身也使用固定占位摘要构造有效 artifact。[loader 与测试](https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/m4_compositional_discovery.py#L1089)

这说明当前验证的是**内部完整性**，不是确认来源。实际 Stage 1 的可信宿主确实运行过确认，不应据此指控原数据造假。

进入在线 M5 后，应由宿主保存候选代码不能自行改写的确认 manifest，绑定模型、训练前缀、确认观测、结果与容差；loader 查验即可。

另有一个小修：M4 evaluator 只规范化绝对路径，相对父路径仍可越出工作区。三个入口统一复用已有路径规范化函数即可。完整报告保留了只执行审计自身常数模型的见证，没有把它扩张成原证据已泄漏。

---

## 八、全生命周期账本还不能按目前措辞宣称已经取得净收益

按提交自己的代理费用：

| 主种子 12 题窗口 | 总成本 |
|---|---:|
| Hegel：Stage 0 + 发现 + 维护 + 目标 | **947.5724** |
| Frozen：相同 Stage 0 + 目标 | **763.2760** |
| Hegel 多投入 | **184.2964** |

所以主 12 题窗口尚未摊回发现和维护投入。

若假设发现、维护只支付一次，后继题目比例与单题成本保持不变，不增加维护成本，先只算域内任务，那么相对 Frozen 的**名义回本点约为第 28 题**；相对已经拥有相同发现历史的逆序普通臂，按新增维护费估算约需 **78 题**。

这些是有条件的摊销计算，不是实测长期学习曲线。[账本源码](https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py#L1642)

还需要限定三件事：

1. **Stage 0 的形成费是手填常数。** Stage 2 的测量费和部分维护费也是手工定价，尚非统一事件账本累计。
2. **三轮 DEV 确实执行了隔离拟合，但三轮数据、参数、确认点完全相同。** 它支持“真实失败能触发归档”，不等于三个独立开发现场。
3. **后续三个 seed 没有恢复维护后的同一 forest 快照。** 它们重新加载作者产物并手工建图，没有重放 Stage 2；应称一次发现后的多种子迁移，而不是四条完整持续学习链。

此外，Frozen fallback 直接枚举含三个最终复合式的共同候选表，没有重新调用作者从原子发现。因此它主要比较索引记忆与共同候选库搜索，不能把其代理费用当作开放重新发现的实测成本。

这些边界需要校准措辞，不意味着必须放弃当前实验台。

---

## 九、建议怎么继续：有限修复，然后进入 M5

我建议第一批只收敛到五项：

| 优先工作 | 最小验收 |
|---|---|
| **让合法步骤和范围契约真正控制执行** | fresh_points 改变真实调用；recall、fallback、最终消费遵守相同 scope |
| **统一保留已付证据** | 失败确认不丢失、不重复购买；转入训练后不再冒充 fresh confirmation |
| **让压缩验证与真实授权共享语义** | A/B 合并后实际可用性保留，或明确拒绝暂不支持的析取合并 |
| **绑定节点、artifact 与宿主确认记录** | 有效 W2 不能替换 W1；确认来源可核；路径规范化统一 |
| **修正统计与指标** | 每格 108 个唯一实例、walk 14/16、Forward recall 48/48，结果能追到真实调用 |

这些都是已有模块之间的连接工作，**不需要推倒森林、不需要取消多尺度、不需要重写全部历史能力。**

如果还要验收“三支柱统一”，再补一个真正交叉的 episode：把现有 Pi 算子和实测证书封装成可持久化、可版本化的森林对象，让后继题消费；经历一次范围失配和修订，再通过同一 dispatcher 回访旧域。这个实验对项目主线的解释力，会高于继续添加大量同模板参数变体。

M5 也不必从零造在线接口。仓库已经有 `CodeProvider.next_action`、`CodexCodeProvider`、调用/token/执行预算、模型档案和 paid trace 保留机制。[现有接口与预算](https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/code_agent.py)

接入时最重要的是：

- **模型负责提案，宿主负责确认和知识升级。**
- 允许超出当前 DSL 的数值代码探索，但未获兼容语义证据前，不自动参加等价池化和保真压缩。
- 在 DEV 冻结排序、提示词、阈值和归档规则，再评独立 TEST。
- 外部数据要明确来源、单位、噪声、测量通道与划分；静态表格能检验拟合和迁移，不能单独证明主动实验选择的收益。
- 同起点、同工具、同数据权限的普通研究臂，也应得到相同的证据复用与重试机会。

对原哲学构思而言，本轮最有价值的进步，是“旧原语经经验形成新复合结构，并进入历史供后继取回”已经有了真实执行对应。下一步需要让**经验修订实际改变执行方法与作用域，让旧知识通过真实路由被保留，让累计净收益在同一对象链上出现**。目前最值得投入的工作，就是把这几处连接兑现。