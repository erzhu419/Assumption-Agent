**这次推进是实质性的：M3.1 的主要修复有效，M4 也已经做出了真实的反馈驱动组合作者。但我认为，“M3.1 + M4 全面闭环、六项验收全部完成”的判断仍然过早。** 当前最需要处理的是几个明确的执行错误，以及作者、森林、多尺度和生命周期实验之间尚未接通的数据与权限链。

我建议保留现有森林、PPR、组合作者和全部成功记录，做一次范围明确的 **M4.1 接线与归因修订**，再继续扩大领域和任务规模。

完整材料已经整理好：

- **[完整审计报告](https://chatgpt.com/space/page_0e015321a81c819199ac3314a991e5b9)**：包含逐项诊断、源码依据、上一轮问题闭环表和下一步验收建议。
- **Hegel_M4_Audit_11961f702_20261119.zip[Hegel_M4_Audit_11961f702_20261119.zip](sandbox:/workspace/scratch/3f14ad2da0ed/deliverables/Hegel_M4_Audit_11961f702_20261119.zip)**：包含主报告 Markdown、三个专项报告、全部探针与结果、两套完整重放、实验工作区、重点源码快照及校验清单，共 872 个文件。

下面把最重要的判断展开。

---

## 一、先说明：原实验结果基本能复现，应当保留

我固定在 `11961f702739ab272661ff59b77fb5bc8e3589b8`，对照 `612d8eea5`，审查了新增模块、全部核心差异、研究脚本、测试、工作总结及相关执行依赖，逐一核对了 10 个作者工作区的 40 个持久化文件，并实际完成以下验证：

| 验证 | 结果 |
|---|---|
| M0—M4 相关回归 | **28 项全部通过** |
| 上轮原样 `probe_closure.py` | 原三种确认记忆器、两项 AST 碰撞、四种旧收据绕过均被修复 |
| M3.1 完整研究重放 | **1,080 条主目标记录、300 条孤立 K 记录、开放修订结果、DEV 与源账本业务字段一致** |
| M4 完整研究重放 | **504 条主目标记录逐条一致**，作者摘要、OOS 摘要一致 |
| M4 生命周期重放 | 主要结果一致，仅一处平均 compute 在第三位小数上为 `16.272→16.273` |
| 新增诊断 | 补缺因子格、历史顺序与内容干预、权限绕过、程序步骤、压缩、尺度作用域及冻结记忆对照 |

M3.1 图快照在最终代码下多出了 `coarse_graining_edges=0` 字段；这与 M4 对共享索引的扩展相符，没有改变原目标结果。

因此，原表中这些目标期成本差异可以保留：

| 实验 | Hegel 平均复合成本 | 对照平均复合成本 |
|---|---:|---:|
| M3.1 对普通全历史 | **37.459** | 41.817 |
| M3.1 对 α=0、同 Pool | **37.459** | 40.305 |
| M4 对普通全历史 | **56.565** | 63.428 |
| M4 对同内容 Flat BM25 | **56.565** | 59.420 |
| M4 对 MicroOnly | **56.565** | 88.871 |

这些数值来自固定任务、固定策略和项目的工作量定价；所有这些主实验臂均严格通过、假确认数为零。它们证明了当前实验协议下的结果，后续还需要判断差异究竟由哪个机制产生，以及是否覆盖形成、试错和维护成本。原始证据见 [M3.1 证据目录](https://github.com/erzhu419/Assumption-Agent/tree/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/docs/evidence/v17_1_m3_1_audit_closure_and_walk_20261119) 和 [M4 证据目录](https://github.com/erzhu419/Assumption-Agent/tree/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/docs/evidence/v17_2_m4_multiscale_open_discovery_20261119)。

阅读范围也应明确：本轮集中重读了新增与修改的核心代码、文档及其实际依赖，历史谱系和文献覆盖承接上轮审计；**没有把仓库中所有历史归档和全部参考文献重新逐字通读一遍**。完整报告中把“已索引、已阅读、已运行”分别列明了。

---

## 二、这次有三项进步，应该明确认可

### 1. M3.1 确实修掉了上轮的主要失败见证

原样探针复跑确认：

- 参数、`observations.json`、`model_observations.json` 三类记忆器的确认 RMSE 均为 **9**。
- 完整 CodeAgent 返回 `scoped_unknown`，没有通过这条旧通道假确认。
- 原来的函数遮蔽与 `round` 精度两项 AST 碰撞消失。
- 原四种收据绕过均被拒绝。
- 诊断 stdout 已能改变下一步候选。
- 普通开放修订臂不再无条件先多买两点。
- 显式完整 grammar 与 `allowed_schemas=None` 在同样实际评估 9 个候选时，均收费 **58**，生成模型 SHA 相同；单候选收费 **10**。

这些修复有实际效果。C5 的旧 M3 路径也已经让方法步骤影响校准、合成和确认；C7 的错误经验可以进入消费窗口。具体变化可核对 [M3 共享求解器](https://github.com/erzhu419/Assumption-Agent/blob/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/src/hegel_machine/cognition/m3_scale_universe.py) 和 [开放修订实现](https://github.com/erzhu419/Assumption-Agent/blob/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/src/hegel_machine/cognition/open_python_revision.py)。

### 2. 新 M3.1 已经提供了此前缺失的 Walk 条件收益

上轮 M3 的问题是：α=0 或同 Pool 的 Flat 已能解释大部分收益，没有识别出随机游走的独立贡献。

新 T25—T28 改变了这个局面：

- Walk+Pool：**14/16** 召回确认。
- Walk 无 Pool：**14/16**。
- α=0+Pool：**0/16**。
- Flat 同 Pool：**0/16**。
- 仅纵向边：**0/16**。

这次正确子规律确实与初始低载查询没有直接签名交集，图传播能把它送进候选窗口。**不能继续套用上轮“所有关键候选本来都能直接命中”的诊断。**

我还做了反向核对：恢复子节点完整的实测签名后，四个 safe query 仍然与子节点零直接重叠，α=0 排名没有变化。因此，也不能简单指责 `witness_only_signatures` 覆盖单独制造了全部收益。[任务与图构造源码](https://github.com/erzhu419/Assumption-Agent/blob/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/src/hegel_machine/cognition/m3_1_walk_factorization.py)

### 3. M4 组合作者确实跨过了旧版固定完整公式表的一道门槛

这一点我认为是本轮最有价值的推进。

Haldane+shunt 的两轮确实发生了：

1. 当前 Monod 的残差出现 turnover，组合器添加分母二次抑制项。
2. 新候选成为当前最优后，再对它做诊断。
3. 副通道残差相关约 **0.809**，继续在这个 Haldane AST 上添加 shunt。

五种 witness 会产生不同结构；真实 `run` stdout 进入了候选构造。固定编译器输出的代码也实际经过拟合、独立确认和外部评估。

**离线确定性 provider 本身完全可以作为这一阶段的有效实现。** 上轮建议本来就允许有限、可审计的开放语法，不要求必须调用在线 LLM。[组合作者源码](https://github.com/erzhu419/Assumption-Agent/blob/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/src/hegel_machine/cognition/m4_compositional_discovery.py)

---

## 三、优先级最高的执行错误

### 1. 保真压缩验证的是 merged，实际保留的却还是原 keeper

这是本轮最明确、最应该优先修的 bug 之一。

`fidelity_compress_nodes` 会构造一个带 `disjunctive_branches` 的合并节点，并验证它能否保留 A、B 的接受与拒绝结果。

但新 `compress_compatible_nodes` 在验证通过后：

- 合并 signatures；
- 合并 provenance；
- 归档 B；
- 重绑部分虫洞；

**没有把被验证的 merged contract/program 安装到 keeper。**

本次构造了一个很小的控制：

- A 支持 case A；
- B 支持 case B；
- 合并对象支持两者。

结果是：

| 对象 | 对 case B 的结果 |
|---|---|
| 被 helper 验证的 merged | **ACC** |
| wrapper 执行后的实际 keeper | **UNK** |
| 原 B | **已归档** |

这里的要求很有限：**验证的对象，应当就是提交的对象。** 不涉及要求证明压缩对无限任务永远正确。

修法可以选择其一：

- 将接口明确限定为同程序、同契约的精确去重；
- 或真正提交验证过的合并对象，升版本、更新边，再对已安装对象重放保真面板。

对应实现见 [森林压缩接口](https://github.com/erzhu419/Assumption-Agent/blob/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L1048)。

此外，当前生命周期的 fidelity evaluator 只检查 arity 和少量前提，不检查预测语义。我用同一个 evaluator 请求合并 Murray 幂律和 Monod 饱和律，它也返回了 `accepted=True`。原实验选择的副本本来同式，不能据此说原实验已经压错；但“零冲突”证书目前没有覆盖不同预测规律。

### 2. M4 子规律分支绕过了最终授权

首个候选确认通过后，会生成绑定 target、version、model、confirmation 的 receipt，再申请 `local_solver_guidance`。

但父规律失败、转向 child 的分支，在第二个确认点通过后，直接设置：

```python
confirmed_ok = True
confirmed_via_recall = True
```

没有执行同一最终授权。

我在原 capped child 上加了一个未测量的硬前提：

```python
audit_boundary_measured == True
```

结果：

- 主流程仍然 `confirmed_via_recall=True`；
- Walk 仍成功；
- 外部预测也正确；
- 但直接调用相同最终授权，明确返回拒绝：
  `unknown_precondition_requires_probe_before_guidance`。

另一个控制把所有最终授权调用都改成拒绝，Walk 路径仍成功，因为它根本没调用最终授权。

**预测值正确，不能替代权限和作用域正确。** 建议把首选候选、child、fallback 收敛到同一个确认与提交出口。[child 分支源码](https://github.com/erzhu419/Assumption-Agent/blob/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/src/hegel_machine/cognition/m4_multiscale_renormalization.py#L1292)

### 3. 新 M4 再次只检查 `program_steps` 非空

这里需要区分新旧路径：**M3.1 的旧路径修复成立，新 M4 没有完整继承它。**

新消费者虽然写着 “Interpret program_steps”，实际只是：

```python
if not node.get("program_steps"):
    continue
```

后面的探针、校准、拟合、确认仍在函数体中硬编码。

把所有节点的步骤换成未知 opcode 和无效参数后，五类代表目标的结果、候选数和费用全部不变，只有墙钟波动。

因此，仅证明“清空列表会退回 fallback”，还不能证明程序内容被执行。应复用现有执行机制，让步骤内容影响实际研究行为，并对未知 opcode 明确拒绝。[消费者源码](https://github.com/erzhu419/Assumption-Agent/blob/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/src/hegel_machine/cognition/m4_multiscale_renormalization.py#L1199)

### 4. 合法的再次组合会发生参数编号冲突

`_max_param_idx` 遍历了多个子节点字段，却遗漏 `cos_mul.param`。

对这个合法操作序列：

```text
exp_relax
→ oscillatory_damping
→ additive_shunt
```

振荡频率已经用了 `p2`，随后旁路又被分配为 `p2`，最终生成：

```python
p0 * u * (1 - exp(-t / p1) * cos(p2 * t)) + p2 * t
```

本应有四个独立参数，实际只剩三个，频率和旁路系数被绑死，结果仍标记为 `VERIFIED_PURE_AST`。

当前五例没有在振荡之后追加这个组合，所以五例成功仍成立；但这个问题直接影响“组合子可以继续迭代”的核心能力。修复统一子节点遍历和参数冲突检查即可。[参数遍历源码](https://github.com/erzhu419/Assumption-Agent/blob/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/src/hegel_machine/cognition/m4_compositional_discovery.py#L204)

---

## 四、作者迁移确实消费了历史对象，但其“森林收益”尚未成立

### 1. 应当承认：真实 AST 方法对象有因果作用

实际 loader 消费的是 `authored_method.json` 中的 `operator_spec`。

本次内容干预把依赖定位得很清楚：

| 干预 | 后继表现 |
|---|---|
| 保留正确 AST，去掉历史 Python 和有效认证元数据 | 仍一次评估成功，compute 22，composite 41.1 |
| 保留原 Python 全字节，仅把 JSON AST 改成 linear | 触发诊断，4 次评估，compute 46，composite 42.3 |
| 删除整个历史 artifact | 同样退化为诊断与 4 次评估 |

所以 **R1 在“持久化 AST 方法对象被真实消费”这一层面成立**。AST 作为规范对象、Python 作为编译产物是合理架构，不应因为没有直接执行历史 Python 文件就否定复用。

尚未接上的是：历史对象的确认状态、摘要、来源版本、作用域、森林检索和授权链。当前只取 `operator_spec`，即使元数据写着 `confirmed_ok=False`、摘要错误，也会直接作为 prior 使用。目标端仍有新鲜确认，所以这不等于现有样本已经假确认；它说明历史认证链没有参与消费。[artifact loader](https://github.com/erzhu419/Assumption-Agent/blob/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/src/hegel_machine/cognition/m4_compositional_discovery.py#L1244)

### 2. 普通组只换历史顺序，就消除了全部成本差距

正式案例中：

- Hegel 被直接交付一个已知正确的 Haldane+shunt artifact；
- 普通组拿到六个历史对象，正确项排在第六；
- 这个比较没有运行森林或虫洞索引。

只将普通组同一六个历史对象倒序，其他内容、数据、工具和预算完全保持：

| 条件 | 候选评估数 | 测量费 | 算力费 | 复合成本 |
|---|---:|---:|---:|---:|
| Hegel 手选正确 artifact | 1 | 40 | 22 | **41.1** |
| 普通组原序 | 6 | 40 | 52 | 42.6 |
| 普通组同内容倒序 | 1 | 40 | 22 | **41.1** |

全部严格通过。

这个控制没有证明“倒序是通用最佳检索器”，但证明了**本例的全部差距可以由预先安排的历史顺序解释**。目前成立的是“已知正确历史方法能减少再发现工作”，还不能归因成“森林组织优于强普通全历史”。[普通组历史列表](https://github.com/erzhu419/Assumption-Agent/blob/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/src/hegel_machine/cognition/m4_compositional_discovery.py#L1272)

### 3. “旧域零退化”实际验证的是可重拟合机制族兼容

当前 backward 检查给 child 换上六个旧 S1 标签重新拟合，再验证两个旧点。

这能证明新机制族包含旧族，是有意义的保守扩张：抑制和 shunt 系数可回到零，cap 可移到非激活区域。

但若保持 witness 原来的冻结训练前缀，三个 child 对相同旧域点的 RMSE 分别是：

- Haldane+shunt：**0.716789**
- cap：**0.232777**
- shunt：**0.145234**

阈值为 **0.045**；重新用旧 S1 标签拟合后接近零。

我不认为不同物理域应被要求共用同一组参数。需要修正的是报告层级：

- 原实例模型是否保留；
- 新机制族是否包含旧族；
- 新旧机制如何按作用域选择；

这三件事应分别验证。当前主要通过的是第二项。[旧域检查源码](https://github.com/erzhu419/Assumption-Agent/blob/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/src/hegel_machine/cognition/m4_compositional_discovery.py#L1146)

---

## 五、多尺度实验有实际内容，但尚未证明可执行的跨尺度推导

### 1. 双通道、实测 closure probe、微观外评都是真的

新 M4 确实：

- 调用不同的 coarse/micro 函数；
- 按观测点收费；
- 购买一对观测计算 closure defect；
- 根据结果选择通道；
- 用 micro 真值做独立 heldout；
- 把有效 coarse 边加入 PPR。

这些应当保留。[多尺度实验台](https://github.com/erzhu419/Assumption-Agent/blob/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/src/hegel_machine/cognition/m4_multiscale_renormalization.py#L64)

### 2. 目前的 Pi_s 仍是函数对与拓扑描述

当前 `micro_fn` 和 `coarse_pi_fn` 是分别给定的函数。图上登记了 Pi 名称和预填 defect bound，但尚未执行：

> 同一 fine state → 可执行投影 → coarse state → coarse evolution → 与微观演化投影结果比较。

所谓下钻，当前是直接把通道和尺度设为 micro，再检索；没有沿 `coarse_graining_edges` 执行投影或逆向推理。

五类代表目标关闭 coarse 图边后，质量、测量、候选数、升级和 Walk 都保持，只有 PPR 计价略变。这只说明这些样例尚未建立投影边的必要性，不能推论边永远无用。

因此，报告中的水分子、雪花/蜂巢、河网/血管/CDN 三层，可以作为**带尺度标签的结构类比实验**。目前不能把它们写成已经从同一微观系统经 Pi 导出了介观和宏观规律。

### 3. 一点闭合不能覆盖整个稳态域

当前一个点的 defect 低于阈值，就返回：

```text
closure_verified = True
closure_scope = steady_state_only
```

我在原 macro 的微观函数上加入一个局部偏差：它在指定 probe 和确认区域近乎为零，在留出区域明显非零。结果：

- closure probe 仍通过；
- 粗模型确认仍通过；
- 最终微观 heldout RMSE 为 **0.447214**；
- `false_confirmation=True`、`strict_pass=False`。

这说明闭合证书作用域过宽。**独立外评确实抓住了错误，是正面的保护证据。** 这个新增反例不否定原 504 条记录。

最小修正是区分“在这些点观察到闭合”和“在这个区域有闭合支持”，记录输入域、测量点、误差统计和版本；不必因此要求任意函数的全域形式证明。

此外，ScaleBlind 同时改了 probe、尺度信息、排名和确认通道，属于策略包消融；若要证明尺度索引的独立作用，应拆成同 probe/确认只切排名，以及同索引只切通道策略两项。

---

## 六、K64 生命周期目前主要验证记忆维护，尚未验证持续新知识学习

### 1. Frozen 的起点不等价

代码先复制完整 Stage 1 森林，然后从 Frozen 专门删除已有的 capped child。因此完整臂初始有 17 个节点，Frozen 只有 16 个，差距从 Stage 1 就存在。

同一个 Frozen arm、同一个目标，仅恢复相同初始记忆：

| Frozen 初始内容 | Recall | 候选数 | 测量费 | 复合成本 |
|---|---|---:|---:|---:|
| 原实验删去 child | false | 5 | 56 | 57.8335 |
| 完整 Stage 1 记忆 | true | 2 | 44 | 44.9453 |

这证明了关键记忆被移除后需要更多重建工作，尚未证明共同起点之后持续学习优于冻结记忆。原控制可以保留，但应改名为“缺失结构记忆对照”。[Frozen 初始化](https://github.com/erzhu419/Assumption-Agent/blob/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/src/hegel_machine/cognition/m4_multiscale_renormalization.py#L1561)

### 2. “零遗忘”的定义还不足以测遗忘

后续每阶段主要加入：

- 6 个已有模板副本；
- 8 个诱饵；
- 2 次只修改协议时间戳的 revision。

诱饵的三次失败反馈直接写入，没有通过实际消费产生失败。评价始终使用同一批 12 个既有目标，`backward_forgetting_rate` 定义为：

```python
1.0 - pass_rate
```

这测的是当前固定集合失败率，没有保存每阶段此前学会的知识集合，也没有比较跨阶段能力变化。所有臂，包括无压缩和 Frozen，都为零，因而不能把零遗忘归因给压缩。[生命周期实现](https://github.com/erzhu419/Assumption-Agent/blob/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/src/hegel_machine/cognition/m4_multiscale_renormalization.py#L1420)

### 3. 活跃集合压缩成立，长期净收益尚待核算

活跃节点从 59 压到 17，在固定任务上保持性能，这项结果有价值。

但：

- 59 个注册记录仍保留，71.2% 是**活跃节点数下降**，不是测得的总存储下降；
- Full 与 NoPrune 末阶段复合成本约为 **57.797 与 57.849**，差约 **0.052**；
- 主要差异来自按活跃节点数线性计价；
- 压缩验证、诱饵验证、索引重建维护成本尚未进入阶段目标均值。

K 计数也需勘误：实际注册节点为 `17→31→45→59`，加上六次 revision 是 65 个相应事件；原 `16→32→48→64` 标签与实际计数不一致。

### 4. 尺度边的版本维护漏接了

精确执行 Stage 2 两次节点修订后：

- 横向虫洞都重新验证了；
- 有效 coarse 边却从 **6 条变为 4 条**；
- 相应 meso 节点上下尺度邻居为空；
- `zero_stale_wormholes_left_unrevalidated` 仍然为 true。

旧版本边失效本身是合理保护；缺的是 coarse/implication 边的重验与可达性检查。不能用横边无 stale 代表整个尺度梯级已维护完成。

---

## 七、M3.1 还需要修正的归因与边界

### 1. 正式设计缺少完整 2×2 的第四格

已有 Walk+Pool、Walk 无 Pool、α=0+Pool，缺 **α=0 无 Pool**。Flat 改了评分机制，不能代替这一格。

本次在一个 seed、同一冻结 gate、10 个固定目标上补齐，40 次都通过：

| 因子格 | 召回确认 | 平均复合成本 |
|---|---:|---:|
| Walk + Pool | 9/10 | 43.34415 |
| Walk 无 Pool | 9/10 | 43.88165 |
| α=0 + Pool | 6/10 | 49.68000 |
| α=0 无 Pool | 6/10 | 49.63000 |

结论是：**Walk 条件收益成立；Pool 的边际依赖条件。** 在 α=0 这格，Pool 反而增加约 0.05 成本。因此应修正“三因素均有普遍严格正边际”的措辞。[正式臂表](https://github.com/erzhu419/Assumption-Agent/blob/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/scripts/run_v17_1_m3_1_factorized_walk_study.py#L51)

### 2. 安全窗口是统一策略，尚未由实验台强制

四个 Walk 任务在最初阶段仍然接受 0.90 全范围测量。正式臂共同遵循低载 pilot，比较本身仍有意义；但当前证明的是“低信息初始观测协议下，历史图先验可以减少探索”，尚未证明真实安全约束下 PPR 绝对不可替代。

### 3. 一次父模型失败被转换成四种机制同时为真

正常求解路径中，父候选在高载确认失败后，无条件把封顶、超调、底物抑制、流量回落四个标志全部设为 True。

本次捕获的 T25、T26、T27 授权确实携带这四个 True；一维 T26 也因此被标为观察到二维封顶和暂态超调。

当前被选中的模型仍有新鲜点确认，最终预测也正确；问题是**三值契约中的证据来源失真**。应记录有来源的“父预测被反驳”，尚未辨别的具体机制继续保持 UNKNOWN。[对应更新逻辑](https://github.com/erzhu419/Assumption-Agent/blob/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/src/hegel_machine/cognition/m3_scale_universe.py#L2396)

### 4. C4 的比较对象写错了

报告里的 `53.23 < 74.70`，74.70 来自删节点消融，不是普通组。

正确数值为：

- Hegel：两类目标均 **53.225**；
- 普通组：**55.075、59.075**；
- 普通组平均 **57.075**；
- 平均优势约 **6.75%**；
- 两边初始测量费均为 **50**。

优势仍然存在，应修正比较对象和幅度。[原始开放修订记录](https://github.com/erzhu419/Assumption-Agent/blob/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/docs/evidence/v17_1_m3_1_audit_closure_and_walk_20261119/open_revision_fair_results.json)

### 5. 少数 API 接受边界仍需收紧，但不应夸大成主表污染

新增小探针发现：

- 重复 `def predict`、`math.exp` 属性重绑可导致同摘要异执行；
- 无 receipt 且省略 target_id 时，旧 boolean 分支仍可授 Tier-3；
- 绝对模型路径可使确认执行绕回原工作目录。

当前固定 M4 编译器只生成单个 `predict`，确认也使用相对 `model.py`，没有发现这些边界由其正常触发。因此应做有限语法、路径和字段校验，**不能据此宣称当前全部实验已经泄漏或失效**。

---

## 八、下一步最值得做的是一条真实贯通的 M4.1 链

对你最初的哲学框架而言，本轮最关键的结构性缺口是：

**Pillar 1 形成的新 AST，没有进入 Pillar 2/3 实际使用的森林。**

主脚本先运行组合作者，随后多尺度实验重新手工构造 10 个算子和森林，生命周期继续给这套人工库加入副本和诱饵。三者共享编译器，但没有传递同一个被发现、确认、注册、检索、修订、压缩的方法对象。[M4 研究入口](https://github.com/erzhu419/Assumption-Agent/blob/11961f702739ab272661ff59b77fb5bc8e3589b8/Hegel%20Machine/scripts/run_v17_2_m4_multiscale_open_discovery_study.py)

这也是我不赞成当前就宣布六项全部完成的主要原因。

按上一轮六项标准，当前较准确的判定是：

| 条件 | 当前判断 |
|---|---|
| R1 历史产物有实际作用 | **AST 方法对象层面通过**；身份、契约和森林接线待完成 |
| R2 反例驱动结构变化 | **五个受控实例范围通过**；再组合参数正确性需修 |
| R3 明确保留旧知识 | **机制族可重拟合兼容成立**；实例与作用域保持需分层 |
| R4 未见任务上有改进 | **给定映射、新参数和新变量名下成立**；机制/协议留一尚缺 |
| R5 支付全部必要费用 | 目标期账本成立；形成、作者试错与持续维护未齐 |
| R6 图超越同内容强基线 | M3.1 和多尺度检索有条件证据；作者主例的图归因未成立 |

我建议下一轮按这个顺序收口：

1. **先修执行语义。** 参数编号、最终授权出口、M4 程序步骤、压缩提交对象、尺度边版本维护。
2. **所有臂从相同初始历史开始。** 普通全历史、强 Flat、Forest、Frozen 使用同一作者、工具、历史内容、确认门和预算。
3. **由新 witness 真正生成一个初始库没有的方法。** 独立确认后，把真实 AST、来源版本、作用域、研究程序和 receipt 注册进森林。
4. **让未见目标经真实索引选择并加载它。** 删除和恢复同一个版本对象，验证行为变化；不再由脚本手选正确 prior。
5. **加入局部反例、修订与旧任务回访。** 明确保留旧实例，修改必要的作用域，再验证压缩后的实际安装对象。
6. **补一个真实 Pi_s 案例。** 使用一个已有快慢系统即可，执行投影、比较同一轨迹上的闭合误差，删除投影或边后必须改变预测或实验。
7. **合并完整账本。** 分列目标、源形成、DEV、作者试错和维护成本，同时保存现成的 CodeAgent events、逐题 closure probe 与授权 receipt。

这条链一旦真正贯通，就能更直接回答项目最核心的问题：**历史总结出来的方法，是否使系统下一轮研究得更好；新一般化是否保留了旧知识的有效部分；组织这些历史关系的收益是否超过同内容强基线。**

当前组件已经足够支撑这一步。下一轮的主要价值会来自把这些对象和证据接通，而不是继续增加人工模板、学科名称或更大的 K。