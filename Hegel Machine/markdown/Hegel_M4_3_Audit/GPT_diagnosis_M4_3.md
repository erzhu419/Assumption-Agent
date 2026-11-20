**我的判断是：V17.5（M4.3）确实完成了一轮有效修复，新的主实验结果也能复现；但“全部核心工程缺口已经闭合”仍不能验收。当前有两组应优先修复的 P1：节点没有完整绑定它应当调用的已确认实例，以及证书的坐标映射没有绑定实际执行模型。两组都已经复现出具体错误，不是泛泛的理论担忧。**

实验解释也需要收窄：**维护的收益真实存在，但高度集中于液压 Pi 任务获得合法粗通道；同信息编译关系表在本次实验中仍略优于 PPR。** 这些结果值得保留，也应如实进入项目下一阶段的判断。

完整材料已经整理好：

- [完整审计报告与修复建议](https://chatgpt.com/space/page_6e1bba01b51c819197c3c9ab5538364c)
- Hegel_M4_3_Audit_346153536_20261120.zip[Hegel_M4_3_Audit_346153536_20261120.zip](sandbox:/workspace/scratch/3f14ad2da0ed/Hegel_M4_3_Audit_346153536_20261120.zip)：包含主报告、三个专项报告、复现脚本、实际模型与观测文件、完整重跑证据和差异检查。

下面把关键判断、证据和建议完整展开。

## 一、先交代这次实际核验到了什么

我固定审计了 `3461535364fcdb4e9ae9c87ccba502b7e6b56fe8`，对照上一版 `fb4b23aaf`，检查了全部 **9 个变更代码／测试文件**及其调用依赖、工作总结、新证据，并复用了提交中完整保留的上一轮审计。具体阅读范围列在专项报告里；历史版本和参考文献采用既有审计作为背景，没有把本轮工作冒称为重新逐字精读所有历史版本、全部论文。

实际复现结果如下：

| 核验项 | 结果 |
|---|---|
| 相关历史回归测试 | **37/37 通过** |
| M4.3 新增测试 | **3/3 通过** |
| 完整官方 runner | 成功执行，约 **710 秒** |
| C1–C6 验证摘要 | 与提交文件**逐字节相同** |
| 主实验回执 | **1,144 条，1,144 个唯一运行键** |
| 主回执的费用、状态、误差、成功率、重叠字段 | 与提交一致 |
| 11 臂汇总、生命周期回本、双轨噪声费用 | 与提交一致 |
| 原始和重跑 SHA256 清单 | 两侧各 **126 项全部匹配** |
| 全证据包逐字节一致性 | **93/127 文件相同**，其余差异已保留并解释 |

需要精确区分：**科学指标复现了，整个文件包没有逐字节完全复现。** 主回执只有 41 个 `selected_model_sha256` 不同；外部参数文件有 22 个浮点数的末位差异，最大绝对差约 \(4\times10^{-13}\)，相应参数和实例摘要随之改变，另外还有墙钟及其他摘要差异。没有把这些差异偷偷过滤后宣称“全部相同”。

完整逐字段记录见 replay_verification.json[replay_verification.json](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m43_346153536/replay_verification.json)。

---

## 二、上一轮哪些问题已经真正修好了

这部分应当明确承认，不能继续把 M4.2 的诊断原封不动套在 M4.3 上。

### 1. 多尺度通道现在确实连接到不同观测

新主入口会检查匹配节点、活粗粒化边、作用域和 modality，再选择通道。`coarse_macro` 和 `micro_fine` 实际调用不同的 Schur／RK4 输出，并按 3／10 收费。

因此，上一轮“同一份细尺度真值，只因某个不相关粗边存在就换个价格”的问题，在当前 M4.3 主流程中已经修复。后文的角色映射问题是另一个仍需补齐的接口缺口。  
[相关执行器](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py#L1119-L1197)

### 2. Frozen 和 OpenRediscovery 恢复了真实含义

Frozen 现在恢复真实 Stage0 的 **8 节点、2 粗边、1 虫洞**，没有继续人为裁剪已有能力。

OpenRediscovery 也真正进入原生 CodeAgent，继承已付训练前缀、运行诊断和组合式发现、执行新确认。独立的小控制中，已有两行直接进入训练，只为新增确认多买一行，没有重新购买旧数据。  
[CodeAgent 调用](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py#L1531-L1606)

### 3. 冻结参数文件自身的校验已经有效

现在会核对参数摘要、外部参数文件、manifest 和确认包，并直接使用持久化参数预测原始确认行。上一轮“改了参数快照，loader 却重新拟合出另一组参数来替它通过”的问题确实关闭了。

但这还需要再接上**节点预期实例身份**，这是下面第一组 P1。  
[实例摘要与加载器](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_compositional_discovery.py#L1224-L1555)

### 4. R3 的只读、零 fit 和粗边依赖成立

独立从提交归档冷载后：

- 十条 R3 路由正确；
- Python 调用观察为 **0 次 fit**；
- 完整森林序列化前后相同；
- 删除粗边后，短时升级返回 `escalation_blocked_missing_coarse_edge`。

因此，旧的“验收函数现场补 scope”“删边后仍照样升级”不能继续列为当前未修问题。  
:chatgpt-content-reference{index="2"}[独立控制结果](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m43_346153536/scale/scale_controls.json)

### 5. 其他几项修复也已接入

OR 分支权限、异质法则拒绝合并、`program_steps` 次序、标准路径 fresh 点足额性、失败确认保留、尾窗 drift 参与认证，都有真实实现。

评测侧新增的 **BestPilotRMSE、同森林 Compiled、六族 DEV、Alpha×Pooling 配置**也进入了真实运行。这使本轮结果比上一轮更容易解释和比较。

---

## 三、P1：多数历史节点仍未绑定它应使用的已确认实例

这是我认为最应优先修复的一项。

### 问题发生在哪里

R3 调用 loader 时使用：

```python
expected_instance_digest =
    str(act.get("confirmed_instance_digest", "")) or None
```

但提交的 postStage2 森林中：

- 15 个节点；
- 14 个持有 artifact；
- **只有重认证后的 Pi thermal 节点保存了预期实例摘要**；
- 其余 13 个缺失，其中 **11 个仍是活跃节点**。

所以，loader 可以判断“这个目录内部是一份合法、确认过的实例”，却无法判断“这就是这个历史节点原先承诺的实例”。  
[具体调用位置](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L578-L583)

### 我做的控制不是随便篡改参数

控制使用项目现有的 `_run_metered_atomic_calibration`，生成了另一个**真实购测、真实拟合、独立确认**的 Monod 实例：

- 原实例：`p0=6, p1=2`；
- 新实例：`p0≈12, p1≈2`；
- 两者 **AST 相同、模型源码 SHA 相同**；
- 实例摘要不同；
- 新实例单独经过严格冻结 loader，合法通过。

然后只替换审计副本中的三个 artifact 文件，保留原节点 ID、版本和 AST。

结果如下：

| 控制 | 调度结果 | 原 R3 Q01 的预测 | 对原实例真值的误差 |
|---|---|---:|---:|
| 原实例 | `dispatched_ok`，原节点 v2 | 2.66666667 | 0 |
| 另一合法同 AST 实例 | **仍 `dispatched_ok`，仍原节点 v2** | **5.33333333** | **2.66666667** |
| 同一替换，但节点补原实例摘要 | `scope_unauthorized`，原节点因 `artifact_verification_failed` 被拒 | 无预测 | — |

错绑后的记录仍然标为 `host_confirmed_discovered_instance`，仍获得 `local_solver_guidance`，也仍然零 fit。完整证据见 probe_same_ast_instance_swap.json[同 AST 实例错绑控制](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m43_346153536/instance/probe_same_ast_instance_swap.json)。

**零 fit 不能代替实例身份检查。** 这里要复访的是历史实例；不能用“同一种公式还可以迁移”来解释另一个参数实例被当作原实例。

现有 C1 的 W1/W2 交换控制因为 AST 本来就不同，覆盖不到这个问题。

### 建议怎样修

这项不需要重写 loader，已有比较器就能拒绝反例：

1. 所有支持零 fit 预测的节点，在首次形成和正式修订时保存预期实例摘要。
2. OR 分支、相关边和回执引用实际使用的同一个实例。
3. 实例预测入口缺少摘要时拒绝，或者走明确的旧档案迁移。
4. 结构迁移可以继续取 operator spec 再拟合，但要使用结构复用的语义，不能继承原实例的确认权限。

还应修正回执字段：当前名为 `artifact_sha256` 的值实际只是**模型源码 SHA**，因此两个不同参数实例在该字段上相同。应分别输出源码摘要、参数摘要和实例摘要。

---

## 四、P1：证书盒通过了一种坐标解释，模型却按另一种解释执行

这一项已经导致真实的错误确认。

### 根本原因

`_find_valid_canonical_role_map_for_box` 会尝试角色排列。即使目标已经明确使用规范名称 `u_drive`、`T_horizon`，identity map 失败后，它仍继续尝试交换，只要某个排列能把输入塞进证书盒，就返回那个排列。

随后，拟合器独立寻找最佳 `role_order`，没有要求它与盒检查的映射一致；最终 query guard 又沿用前面的盒映射。  
[盒映射实现](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L2410-L2458)

**角色含义应该先被确定，再检查是否落盒。现在代码允许用“哪个排列能落盒”反过来决定角色含义。**

### 具体反例

保留同一个正常 T21：

- 相同 fine／coarse oracle；
- 相同森林；
- 相同容差 `0.055`。

正常正对照走 coarse，heldout RMSE 为 **0.008633**。

变体将输入点和任务声明范围改为合法短时区域：

- pilot：`u=2.45…2.95`，`T=0.40…1.20`；
- 确认点：`u=2.70,T=0.47`；
- 声明 bounds：`u∈[2.4,3.0],T∈[0.4,1.3]`。

而真实证书要求：

\[
u\in[0.4,3.0],\qquad T\in[2.4,5.2].
\]

实际执行中：

1. 盒 helper 返回 **u↔T** 的交换映射。
2. 拟合器 spy 记录的实际模型仍按 **`['u_drive','T_horizon']`** 执行。
3. 系统以 coarse 通道购测和确认。
4. 最终守卫使用交换映射，没有拦截。
5. 返回 `completed`、`confirmed_ok=true`。
6. 对 fine truth 的 heldout RMSE 为 **0.100316 > 0.055**，即 `false_confirmation=true`。

这仍然是项目自身的连续两模方程，未加入隐藏尖峰或任意模型。实际拟合角色、购测记录、生成源码和最终 scope contract 都保存在 scale_controls.json[尺度控制结果](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m43_346153536/scale/scale_controls.json)。

### fresh 选择后还有一个相关缺口

目前前置检查只看 `confirm_points[:1]`。

保留默认 fresh1，将第一个预声明确认点设为已付 pilot，第二点设为 `u=2.8,T=0.4`：

- 前置检查看到盒内的第一个点；
- fresh helper 正确跳过已付点；
- 实际选到的第二点却没有再次过证书盒；
- 该越盒点仍以 coarse 通道购买和确认。

这个小控制的原 heldout 仍通过，所以它证明的是作用域检查漏接，不是第二个数值错误反例。

### 建议怎样修

应把同一个角色映射贯穿：

**通道资格 → 编译和拟合 → 确认实例 → 实际 fresh 批次 → 最终 query。**

规范名已经明确时使用 identity。匿名映射应作为具体候选身份的一部分，经过验证后冻结。实际确认点必须落在任务域与证书域的交集，不能只检查原列表的第一点。

现有 fitter 已返回 `role_order`，fresh helper 也已有 `verified_box` 参数，这些能力可以直接复用。

---

## 五、主表能复现，但应按什么含义接受

以下都是**提交原始账本**的数值，采用 \(C=M+0.05K\)。每臂 96 道域内题，另 8 道 OOS 均 unknown、无假确认。部分计算事件漏记的问题见下一节，不能把账本算术一致等同于完整计量已验收。  
[原始证据目录](https://github.com/erzhu419/Assumption-Agent/tree/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/docs/evidence/v17_5_m4_3_channel_bound_lifecycle_20261120)

| 臂 | 严格通过 | 假确认 | 平均测量费 M | 平均 C | 96 题全周期 C |
|---|---:|---:|---:|---:|---:|
| Hegel Unified | 96/96 | 0 | 31.3333 | 32.3941 | 3762.1078 |
| Hegel Walk NoPool | 96/96 | 0 | 31.3333 | 32.3941 | 3762.1078 |
| Compiled SameForest | 96/96 | 0 | 31.3333 | 32.3235 | 3757.1530 |
| Hegel NoMaintenance | 96/96 | 0 | 36.2500 | 37.5675 | 4040.3097 |
| Alpha0 PoolOn | 95/96 | 0 | 37.6250 | 39.5131 | 4445.5373 |
| Alpha0 PoolOff | 95/96 | 0 | 37.6250 | 39.5131 | 4445.5373 |
| Flat BM25 | 95/96 | 0 | 37.6250 | 39.5315 | 4447.2990 |
| Ordinary StratifiedDEV | 96/96 | 0 | 37.8333 | 40.2083 | 4761.3750 |
| Ordinary BestPilotRMSE | 96/96 | 0 | 36.1875 | 41.5625 | 4423.8250 |
| Stage0 OpenRediscovery | 93/96 | 1 | 42.1250 | 44.4729 | 4513.4520 |
| Stage0 NoDiscovery | 16/96 | 0 | 34.2917 | 35.8182 | 3682.6020 |

这里有三个必须保留的解释：

- Hegel 的 96 次成功中，**87 次经召回确认，9 次使用全历史 fallback**。不能写成所有任务只靠图召回完成。
- NoDiscovery 总费用低，但只解出 16 题，不能单看费用判断系统优劣。
- OpenRediscovery 有两次 unknown、一次假确认，和它比较费用时不能隐去成功率差异。Hegel 在这个任务流上同时更准确、更便宜，仍然是有效结果。

---

## 六、“46 题维护回本”成立到什么程度

### 1. 数值可复现，收益来源高度集中

逐任务配对后的分解是：

| 部分 | Stage3 毛节省 C |
|---|---:|
| 8 个 hydraulic adiabatic | **+491.1920** |
| 8 个 thermal quench | **−0.0080** |
| 非 Pi 五族，共 80 题 | **+5.4679** |
| 合计 | **+496.6519** |
| 扣除 Stage2 维护投入 218.45 | **净省 278.2019** |

Pi 占 Stage3 毛收益约 **98.899%**。非 Pi 的确不再像上一轮那样完全没有差额，但它的 +5.4679 还不能证明该部分维护独立回本。  
:chatgpt-content-reference{index="5"}[完整配对与归因数据](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m43_346153536/evaluation/evaluation_analysis.json)

### 2. 主要收益来自液压任务获得合法粗通道

形成时：

- thermal 节点有对应粗边，但不授权 hydraulic；
- hydraulic duplicate 有对应 modality，却没有自己的活粗边。

维护后通过 OR 整合和证据连接，液压任务获得合法粗通道。

保持同一个目标函数和坐标，只改 modality，结果是：

| 森林状态 | hydraulic adiabatic | 同一目标改为 thermal |
|---|---|---|
| Stage0／未维护 | micro | coarse |
| 维护后 | coarse | coarse |

粗细实验室输出也确实不同。**这一收益来自合法知识路径和授权可达性变化，不能再归咎于上一轮的虚假价格切换。**

但它也不能主要归功于 quench 范围收缩：当前 quench pilot 从 `T=0.20` 起，早已同时越出旧、新两份证书盒。主表没有构成“旧盒会放行，新盒会拒绝”的配对。

### 3. 46 是当前流的结果，不是系统常数

八个高收益 adiabatic 在流中的位置为：

`21、22、45、46、69、70、93、94`。

第 46 题恰好是第四个高收益任务到来，跨过维护投入。准确表述应是：

> 在当前家族配比、任务顺序、粗细价格比和原始计量代理下，第 46 个目标观察到维护回本。

我认可这个条件性结果。不能从它推出任意任务分布下都会在相近位置回本，也不能把全部节省分别分配给每一种维护操作。

---

## 七、Compiled 对照改变了对 PPR 的解释

这个对照是本轮最值得保留的新增实验之一。

Compiled 和 PPR：

- 都是 **96/96**；
- 测量费都为 **31.3333**；
- 都有 **87 次 recall**；
- 都有 **31 次非直接关系命中**。

PPR 平均候选更少：2.5938 对 2.7604。但在原始账本上，计入额外 **1.818** 编译费后，Compiled 全周期仍便宜 **4.9548**。这不是完整事件计费或墙钟优势的证明，但原账本上的结果很明确。  
:chatgpt-content-reference{index="6"}[评测专项报告](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m43_346153536/evaluation/EVALUATION_AUDIT.md)

这说明：**森林里的关系知识有用，而且可以由不同检索方式消费。** 当前证据支持保留关系和契约，同时应继续单独检验 PPR 本身的增量价值。

另外，两组相同 Alpha 下的 PoolOn／PoolOff 本轮结果完全相同，因此 pooling 在这个条件下的可检测边际效果为零。`NoPool` 关闭的是等价池化，不能误读为删除全历史 fallback。

### 当前 PPR 也没有自动跟随森林更新

正式把 W4 从 v1 修订为 v2，并加入新 scope 条件后：

| 查询对象 | 返回版本 |
|---|---|
| 原 PPR 索引 | **v1，旧 scope** |
| 新建 PPR 索引 | v2 |
| 原 Compiled 表 | **v1，旧 scope** |
| 新编译表 | v2 |

PPR 在构建时固定节点快照和检索结构、预先计算 transitions，没有随正式 revision 刷新。主 Stage3 使用冻结森林，所以没有因此使当前主表失效；但报告里的“免重编动态更新”不能当作现有能力。  
[索引实现](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/hipporag_wormhole_index.py)

进入 M5 在线写回前，应加入版本失效检查。当前小图直接重建并计费即可，不必马上设计复杂的增量 PageRank。

---

## 八、噪声实验总体公平，但结论需要更精确

### 1. 没有发现给 Hegel 更干净数据

同一任务各臂按相同 seed 产生相同 pilot 噪声。附加 stress／confirm 使用同一洁净 oracle 机制，策略决定购测点不同，这属于真实策略差异。

报告缺少附加双轨实验的逐例质量回执，因此我补核了：

**两个噪声设置 × 八个 onset × Hegel／BestPilot 两臂 = 32 条。**

| σ | 臂 | 严格通过 | 假确认 | 平均 M | 平均 C |
|---|---|---:|---:|---:|---:|
| 0 | Hegel | 8/8 | 0 | 28.0 | 29.1623 |
| 0 | BestPilotRMSE | 8/8 | 0 | 28.0 | 32.9250 |
| .002 | Hegel | 8/8 | 0 | 29.5 | 31.6748 |
| .002 | BestPilotRMSE | 8/8 | 0 | 30.0 | 37.0250 |

费用与官方完全相同，overlap 均为 0。**这项成本比较没有隐藏质量退化，可以保留为正证据。**  
:chatgpt-content-reference{index="7"}[质量补核结果](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m43_346153536/evaluation/fork_quality/fork_quality_summary.json)

### 2. “有界噪声”应改为“高斯噪声”

代码调用 `gauss(0,.002)`，没有截断，实际是无界高斯分布。

标准差约为容差的 3.23%，不代表每个实现值都小于容差的 3.3%。主实验 192 个带噪 pilot 的最大绝对扰动为 **0.0052912692**，约为容差的 **8.53%**。修改表述即可，不必为了维护“bounded”这个词重造实验。

### 3. 有噪时的测量优势很小，主要差额仍在计算

有噪时 Hegel 自己也增加了测量：29.5 对无噪的 28。

相对 BestPilot，测量费平均少 **0.5／题**，八题合计少 4 个费用单位，也就是少一次单通道购测。更大的总成本差仍主要来自计算代理。

所以，这支持当前噪声设置下的有限收益，不足以重新建立普遍的信息论不可判别主张。

### 4. DEV 已覆盖六族，但实际没有施加噪声

DEV 虽构造了声明 `.002` 噪声的目标，实际观察行仍直接调用 `oracle_fn`；Pi 选中的两例也都是 adiabatic，没有 quench。

六族覆盖是真修复，但准确说法应是“六家族无噪参数 DEV 的命中频次顺序”。后续若需要同分布 DEV，复用主采样接口即可；目前没有证据证明对齐噪声必然改变排序，也没有必要为此额外铺开大规模扫描。

---

## 九、其他已确认问题：应修，但不应夸大影响

### 1. 新 fine surrogate 的系统矩阵写错了一项

真实 RK4 方程里：

\[
\dot x_s=-x_s/\tau_s+(a_s/\tau_s)u+g_{sf}x_f.
\]

因此矩阵 `A12` 应为 `g_sf=0.35`。新 M4.3 两指数分支用了 `g_sf/tau_s≈0.24137931`。

用三个真实 RK4 点解系数，能补偿标定点，但不能把错误特征值变成同一 ODE 的精确解。

正常 T23 的 heldout RMSE 是 **0.005842**，最大单点误差约 **0.0097155**，仍通过当前 0.055 容差。因此应修正公式与 `exact` 措辞；不能说当前 quench 全部失败，也不能把问题外推到直接执行真实 RK4 的 R3 路径。  
[新 fine 分支](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py#L1297-L1394)

### 2. fine 分支漏计了已经执行的候选工作

实际观察到：

1. 先试拟合一个通用候选；
2. 丢弃该结果；
3. 再生成并训练两指数 surrogate；
4. 最后把 `ev_c=1`。

另外还有 9 次分支内部的 host RK4 计算。

沿用当前每候选 K=6 的约定，仅前一个被丢弃的候选，至少少计 **C=0.3／受影响任务**。Hegel 八个 quench 对应最低 +2.4；NoMaintenance 和 Compiled 的同类分支也有相同遗漏，所以不能凭这一点宣称 46 回本翻转。  
:chatgpt-content-reference{index="8"}[实际调用观察](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m43_346153536/cost_events.json)

正确处理是删掉无效工作，或者如实计费，再更新既有主表。本地可信 solver 的计算应记计算费，不能误称为向未知目标额外购买测量。

### 3. 证书与当前实例仍可错配

真实 Stage0 Pi 证书可被用于真实 Stage2 实例的再验证，API 接受后出现：

- 边实例摘要：当前实例 `0586…`；
- 边内证书实例摘要：旧实例 `416e…`。

两个实例在共同盒内预测仅差约 \(9\times10^{-8}\)，所以这里证明的是证据归属缺口，没有制造数值越界。正常提交的 Stage2 三方本来一致，旧成功链不同步已经修复。

应在复制和重绑前，检查证书、实例、角色和 ScaleMap 的一致性，并使更新原子完成。

### 4. 可迁移归档能执行，但摘要无法按归档内容复算

严格使用生产代码的五个字段重算：

| 快照 | semantic digest 不一致 |
|---|---:|
| Stage0 | 7/8 |
| postStage1 | 14/15 |
| postStage2 | 14/15 |

路径参与身份 hash，而序列化和加载又会改变路径。这不等于归档恢复失败——冷载 R3 已经通过；但“可以恢复执行”和“身份摘要可自验证”是两项不同要求。

建议使用位置无关的内容身份，把存储位置单独解析。  
:chatgpt-content-reference{index="9"}[摘要检查结果](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m43_346153536/scale/archive_digest_checks.json)

### 5. 两项旧入口问题应单独处理

旧无 bundle 的兼容 loader 会提前返回，绕过新增的 `verify_frozen_only` 和 expected-instance 约束；旧 M4 ScaleBlind 在升级到 fine 时仍会丢失已付 fine 确认记录，再购买同一确认点。

**当前严格 R3 不走前一个兼容分支，当前 M4.3 主表也不能因此被判为有训练—确认重叠。** 这两项应作为明确限定范围的兼容修复。  
:chatgpt-content-reference{index="10"}[旧入口控制](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m43_346153536/instance/probe_instance_freshness.json)

---

## 十、放回整个项目里，我怎样看这版的意义

M4.3 更有力地支持了一个具体、可检验的方向：

> 历史形成的规律、角色解释、适用范围、关系路径和已确认实例，可以作为后续求解的可执行资产，减少重复发现和不必要的高成本观测。

这与项目的哲学出发点是连贯的。但这次两个 P1 也暴露出必须保留的区别：

- **同一法则，可以有不同实例。**
- **同一组坐标名称，可以对应不同的角色解释。**
- **过去确认过的东西，必须能说明具体确认了什么、在哪个范围、针对哪个实例。**

如果一般化时抹掉这些区别，系统就会把“某个合法实例”当成“原来的历史实例”，或者把“某个能过盒的排列”当成“物理上正确的解释”。因此，本轮修复实例和角色绑定，会直接加强历史保留、条件化复用与扬弃机制。

同时，当前四个 seed 都来自**一次形成和维护后的森林**；Stage3 不把新结果写回；Pi 内部动力学参数固定，变化主要是读出、任务参数和预定时域。它提供的是一次形成链后的受控迁移证据，还没有展示多条独立形成链上的递归自我改进。

后续如果目标转向真正持续学习，关键变化应是**让新确认实际改变同一森林，并让后续任务真实消费这个改变**，同时记下更新费用、失效旧索引、验证旧能力保留。单纯增加更多测试 seed 不能替代这一步。

## 十一、我建议下一轮只做一个有边界的 M4.3.1

修复顺序可以很明确：

| 顺序 | 工作 | 最小验收 |
|---|---|---|
| 1 | 所有实例节点保存预期实例摘要，证书与实例绑定 | 原实例可用；另一合法同 AST 参数实例拒绝；恢复原实例可用 |
| 2 | 同一角色映射贯穿授权、执行、fresh 与最终 query | 正常 T21 不退化；短时轴交换不能 coarse 错误确认；补选点仍在合法域或升级 |
| 3 | 修正 fine 矩阵，消除或计入被丢弃的候选工作 | 正常 quench 通过；实际候选数与费用对应；更新原主表 |
| 4 | 归档内容身份与路径分离，修复两个兼容入口 | 搬迁后身份可验证；严格 flags 不被绕过；已付 fine 数据不遗失 |
| 5 | 更正文案，并接上动态索引失效 | Gaussian、维护收益来源、同形成链四 seed、Compiled 结果表述准确；在线修订后不返回旧 scope |

前两项是当前“核心闭合”验收的阻断项。其他项都有具体实现位置和可复用接口，没有必要再换一套研究框架。

修复后的回归规模也可以保持有限：**现有 40 项相关测试、这里的有限正负控制、一次既有四 seed 全研究。** 只有具体失败揭示新风险时再扩测。

我对这版的验收意见是：**保留 M4.3 已取得的工程进展和条件性实验收益，完成上述实例与坐标绑定后，再把“核心闭合”作为 M5 的可靠起点。**