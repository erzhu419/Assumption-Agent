我对 **`fb4b23aaf` 固定提交**完成了源码、交接文档、直接依赖和实验回执的复核，也跑完了指定的 **37 项测试和完整四种子研究入口**。

**我的判断是：M4.2 有实质性的集成进展，可以继续作为主线；但“12 项问题 100% 闭合”“第 29／39 题实现全生命周期回本”“图游走的不可替代性已得到证明”，这三项结论目前都需要收窄。**

最值得下一轮集中修复的是五组连接：**真实执行与计费、Frozen 起点与发现过程、已付数据与新鲜确认、确认记录与冻结参数、尺度证书与最终使用范围。** 这些缺口都有具体源码和有限对照支持，修复可以沿用现有组件。

完整材料已经整理好：

- [完整审计报告与施工建议](https://chatgpt.com/space/page_d64073c2276481918331093ca46c03d1)
- Hegel_M4_2_Audit_fb4b23aaf_20261120.zip[Hegel_M4_2_Audit_fb4b23aaf_20261120.zip](sandbox:/workspace/scratch/3f14ad2da0ed/deliverables/Hegel_M4_2_Audit_fb4b23aaf_20261120.zip)：包含完整报告、三个专项审计、控制脚本、原始 JSON、全量重放和完整性清单。

下面把主要判断、证据和建议展开。

## 一、先明确哪些结果已经成立

### 1. 原实验的统计结果确实可以重现

这次复核得到：

| 项目 | 结果 |
|---|---|
| 指定 M0—M4.2 回归套件 | **37 项全部通过** |
| 完整 M4.2 研究入口 | 四种子、九臂完整运行成功 |
| Part2 去重回执 | **432 行逐字节一致** |
| Part4 统一回执 | **792 行的成功率、误差、候选数、召回和成本字段全部一致** |
| 原始回本结果 | 29／39 的账本交叉点能够重现 |
| 提交证据 SHA | **115 项全部通过** |

需要保留一个精确限定：Part4 有 **13 条 `selected_model_sha256` 不同**，其余回执字段一致；汇总文件还有临时路径、墙钟时间和约 \(10^{-13}\) 量级的拟合参数尾数差异。因此可以说“研究统计重现”，不能说整个包逐字节一致。完整差异保存在 replay_comparison.json[重放核对记录](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/replay/replay_comparison.json)。

附件交接书写 120 个受校验文件，提交内交接书已经改为 115。重放多出来的五个文件正好是 Python 字节码缓存，**这不是五份科学证据缺失**。

### 2. 上一轮的几项具体问题确实修好了

我确认了这些有行为证据的进展：

- `fresh_points=1/2` 现在真正改变确认点数量和费用。
- 同法则、同权限的两个作用域合并，真实授权器可以接受 A、B，拒绝 C。
- 仅填写 `confirmed_ok` 和伪确认摘要的自洽 JSON 已被拒绝。
- 真正的 M4.2 W1、W2 产物各自能加载；W2 放入 W1 节点后，expected AST digest 检查会拒绝。
- 相对父路径逃逸已进入 containment 检查。
- 三次 lure DEV 已变为不同公式、领域、特征和点集。
- 失败确认值现在确实保留给后续拟合。
- 每个 seed 都从同一 post-Stage2 快照恢复，OOS 参数也会随 seed 变化。

这些不能再沿用上一轮的否定结论。CLOSURE_AUDIT.md[工程复核结果](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/closure/CLOSURE_AUDIT.md)

### 3. Pi 和实例级 R3 已有真实集成

Pi 节点、初始证书、thermal／hydraulic 合并、短时淬火失配、缩盒重认证，现在确实发生在同一森林构建流程里。

冻结执行也确实恢复了保存的 `_PARAMS`，不是拿默认参数冒充旧实例。旧 Monod、新 Haldane、振荡模型和 capped Murray 的交错重访，有实际工程价值。

不过，“zero fit”应准确称为：

> **零目标重拟合的冻结实例推理。**

因为冷加载 loader 仍可能在历史前缀上重新 fit，以验证来源。末端 helper 返回 `fit_called=False`，不等于整条调用链没有发生任何历史验证拟合。PI_AUDIT.md[Pi 与分发器专项报告](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/pi/PI_AUDIT.md)

---

## 二、最影响当前核心结论的问题：成本没有绑定真实尺度执行

这是本轮最需要优先处理的实验设计问题。

### 1. Pi 观测价格在检索和授权之前就决定了

`evaluate_m4_2_target_under_arm` 对 Pi 目标检查：

1. 森林存在任意一条没有标记 stale 的粗粒化边；
2. 索引是 `HippoRAGWormholeIndex`；
3. `damping_alpha > 0`。

满足就每点收费 4，否则收费 8。

随后两者都创建同一个 `OpenWorkspaceTaskLab`，调用同一个 `target.oracle_fn`。这个价格判断没有要求：

- 那条边对应当前 ODE；
- 本次真的召回或选择了那条边；
- 证书覆盖当前查询；
- 实际执行了粗尺度观测通道。

因此，当前 Hegel 与 Alpha0／BM25 的成本差额，混入了直接按索引条件分配的价格差。[Stage3 计费实现](https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L2271)

### 2. 无关边就能让相同执行半价

我用真实 post-Stage2 森林和原 T17 做了这个控制：

| 森林状态 | 实际观测数 | 最终模型 | heldout RMSE | 测量费 | 复合费 |
|---|---:|---|---:|---:|---:|
| 完整森林 | 7 | Pi coarse 节点 | 0.0 | 28 | 28.950 |
| 删掉 ODE 粗边，只留无关 hydro 粗边 | 8 | exp-relax fallback | 0.005173 | 32 | 34.825 |
| 删除全部粗边 | 8 | 同一个 fallback | 0.005173 | 64 | 66.825 |

**后两行的实际观测阶段、输入点、观测值、候选数、最终模型和误差都相同。** 仅保留一条无关的 hydro 边，就把同一执行从 66.825 降到 34.825。cost_control_summary.json[完整对照数据](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/cost/cost_control_summary.json)

多尺度观测有不同价格本身完全合理。这里需要修的是：**价格必须由实际获得授权并执行的通道产生。**

现有 `MultiScaleWorkspaceTaskLab.acquire_channel` 已经能承载这个修复，不需要重新发明实验室。

### 3. 计算账也有遗漏

同一对照中，PPR 返回 `query_compute_cost=2.745`，但 Stage3 没有把它记入 meter。新执行器主要在末尾按候选数量记一笔合成计算费；冷加载验证、缓存命中等也没有分清。

可以继续采用预登记的计算成本代理，不必把墙钟时间当成唯一标准。但应该让每个代理账项对应实际事件，并在各臂一致消费。[检索费用返回位置](https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/hipporag_wormhole_index.py#L460)

---

## 三、Frozen 和 OpenRediscovery 两个对照仍然不符合其命名

### 1. Frozen 没拿到真实完整的 Stage0

在第一轮 Stage1 开始之前直接截获的真实 Stage0 有：

- **8 个节点；**
- **2 条粗粒化边；**
- **1 条横向 wormhole。**

九臂入口却新建一个空 forest，只复制四个节点，全部边都没有复制。两个 onset 桥和 thermal／hydraulic 两个 Pi coarse 节点也被遗漏。

两个 Frozen 臂仍承担完整的 **244.05** Stage0 费用，却没有拿到这笔费用已经形成的全部能力。[Frozen 构造位置](https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L2600)

恢复真实 Stage0 的两题控制是：

| 目标 | 原四节点 Frozen | 真 Stage0 |
|---|---:|---:|
| T09 Monod onset | 39.50 | 40.10 |
| T17 Pi | 57.55 | 29.55 |

T09 反而略贵，因为多评估两个原子候选；T17 的观测、候选、fallback 和误差全部相同，差额来自恢复粗边后触发半价。

所以我没有从这两例推算“公平 Frozen 的新 80 题均值”。能够确定的是：**当前 Frozen 不是形成完成时的 Stage0 快照，原回本比较的起点不成立。**frozen_snapshot_control.json[快照控制数据](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/unified/frozen_snapshot_control.json)

最小修复就是在真正的 Stage0 边界保存完整 snapshot，后面直接 deepcopy。

### 2. Stage3 的 OpenRediscovery 相对 PoolFallback 只是多记账

这个分支额外执行一笔：

- `obs=1`
- `meas=obs_unit_cost`
- `compute=24`

之后两臂进入同一个 fallback pool、同一个候选拟合函数。没有对应的新 CodeAgent episode、诊断观测或组合生成过程。[相关实现](https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L2510)

原 T01 的执行对照：

| 项目 | PoolFallback | OpenRediscovery |
|---|---:|---:|
| 实际获得观测 | 7 | 7 |
| 账本记录观测 | 7 | 8 |
| 候选评估数 | 10 | 10 |
| 模型、确认和 heldout | 相同并通过 | 相同并通过 |
| 复合费 | 31.65 | 36.85 |

这笔没有对应动作的费用，80 题合计 **480**，正好解释两臂均值的 6.0 差额。cost_control_summary.json[实际执行对照](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/cost/cost_control_summary.json)

这里的判断只针对这条 Stage3 对照。**Stage1 的 CodeAgent 发现确实发生了。**

下一轮应直接复用 `run_compositional_code_agent_episode` 接入真正的开放重发现；如果暂时采用指定费用的代理对照，就应明确称估算，不能列入“实测开放重发现回本”。

### 3. 29／39 的回本数字对这两个账项非常敏感

我保留原决策、观测顺序、候选数、裁剪后的 Frozen 快照和前期投入，只做：

- 各臂 **Pi 目标**观测统一单价；
- 去掉 OpenRediscovery 没有实际动作的诊断账项。

结果为：

| 比较 | 原首次交叉 | 重计后的首次交叉 | 原第 80 题净省 | 重计后净省 |
|---|---:|---:|---:|---:|
| Hegel vs Frozen Pool | 39 | **74** | 482.525 | **34.525** |
| Hegel vs Frozen Open | 29 | **74** | 962.525 | **34.525** |

算术分解是：

\[
482.525-448=34.525
\]

\[
962.525-448-480=34.525
\]

Pi 统一单价取 4 或取 8，都得到同样净差，是因为这批 Pi 任务中两边都获得 112 次观测，同价变化给两边增加相同费用。accounting_sensitivity.json[敏感性计算及完整数组](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/cost/accounting_sensitivity.json)

**74 不是我给出的“正确回本答案”。** 这只是事后账本敏感性，尚未恢复完整 Frozen、修复确认流程、真正执行开放发现，也没有补齐计算计费。这些修复可能改变策略行为，需要之后重新运行主实验。

它已经足够说明：**原 29／39 不能继续作为已完成的架构净收益证明。**

---

## 四、已付失败观测现在保留了，但又被当成新确认

这是本轮已经影响实际运行的协议问题。

新代码将失败确认加入 `frozen_prefix`，这一步是正确修复。但 fallback 在没有剩余确认点时，又强制使用最后一个旧确认点。

结果就是：**模型用这个点拟合，再在这个点上“确认”。**[确认耗尽处理](https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L2526)

这条路径在原 792 条回执中实际发生 **32 次**：

- DEV 冻结臂的 16 个 Murray onset；
- Forward 臂的 16 个 Murray onset。

原 DEV、seed 20261120、T13 的直接 trace：

| 阶段 | 确认点 | 训练前缀行数 | 是否已经用于训练 | RMSE |
|---|---|---:|---|---:|
| ordinary confirm 0 | (11.5, 0.6) | 6 | 否 | 9.7110 |
| ordinary confirm 1 | (8.5, 1.0) | 7 | 否 | 0.55288 |
| fallback confirm | (8.5, 1.0) | 8 | **是** | 2.85e-10 |

这里必须保留公平的解释：**这 32 条的独立 heldout 仍全部通过。** 本例独立五点 heldout RMSE 约 \(7.16\times10^{-9}\)。不能把它们改称 32 个错误预测。

问题是在线确认失去独立性，并重复购买了一次没有新增信息的确定性观测。confirmation_reuse_control.json[调用轨迹](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/unified/confirmation_reuse_control.json)

修复应做到：

- 继续保留全部已付失败数据；
- 在确认前验证查询与训练前缀不重叠；
- 所需新鲜点不足时，获取真正的新输入或返回证据不足；
- 不删除历史，也不重购旧点伪装新确认。

同根的较小边界是：声明 `fresh_points=3` 而只有两个点时，仍默默接受两点确认；旧 M4 子级路径也存在耗尽后复用训练点。参数已经接线，“足额且新鲜”的不变量还没完整接线。

---

## 五、冻结参数确实保存了，但确认的不是实际执行的那组参数

这一项直接关系到实例级 R3 的可信性。

当前流程中：

1. 确认包绑定源码、AST、训练行、确认行和误差；
2. loader 重新 fit 源码，检查它能不能通过；
3. frozen executor 使用 artifact 中保存的 `fitted_params`。

但是 manifest 没有绑定那组参数，loader 也没有使用外部 `confirmed_fitted_params.json` 验证实际冻结态。[保存、加载与冻结执行实现](https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_compositional_discovery.py#L1005)

用提交内真实 W1 完整工作区，同时打开严格 external manifest 和正确 expected digest：

| 条件 | 严格 loader | 原确认集的冻结预测 RMSE |
|---|---|---:|
| 原始参数 | 接受 | \(5.487\times10^{-10}\) |
| 只把 artifact 内的 `p0` 乘 3 | **仍接受** | **7.689592855** |

容差是 **0.045**。源码、AST、bundle、manifest、外部参数文件和确认数据全部没改。

因此，当前 loader 证明“重新拟合能通过”，实际 executor 却可能使用另一份保存状态。contract_controls_results.json[参数绑定控制](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/closure/contract_controls_results.json)

这不意味着当前正常 W1 快照错误；基线正例非常准确。它证明的是保存态错配没有被拦截。

**建议把实际通过确认的参数态，与源码、角色／透镜、范围组成同一个 confirmed instance identity。** 严格实例加载应直接用这组冻结参数预测原确认行，而不是另一次 fit 后间接替代验证。现有归档和 manifest 足够承载这个修复。

---

## 六、尺度授权链尚未贯通到新 Stage3 和最终查询

这是另一组需要合并处理的 P1。它们共同涉及：证书、范围、节点、边、实例和实际查询是否引用同一个授权对象。

### 1. 真正的窄盒证书可以授权扩盒

新重验证接口增加了实质检查，但没有比较证书的盒和拟提交 ScaleMap 的盒。[证书重验证实现](https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L1068)

我使用原函数生成的真实六点证书：

- 认证范围：\(T\in[2.4,5.2]\)；
- 最大闭合缺陷：0.032846；
- 证书内容不变；
- 只把拟提交 ScaleMap 扩为 \([0.05,5.2]\)。

重验证仍 accepted。随后同一 \(u=2,T=0.12\) 查询：

- 原森林走 fine，误差 0；
- 扩盒后走 coarse，获得授权，误差 **0.32400957**。

这要求的是“窄域证据不能自动授权更广域”，不是要求有限网格证明整个连续空间。

### 2. 专门 Pi suite 修好的最终查询守卫，没有进入新 Stage3

Stage3 检查 pilot 和 confirm 的范围，后面直接执行最终查询，没有继承专门 Pi suite 已有的导出范围契约。

只把原 T17 的最终查询改成同一 ODE 的 \(T=0.12\)，其他条件不变：

- 仍返回 `completed`；
- `confirmed_via_recall=True`；
- 外部评价 RMSE 为 **0.218132**；
- `false_confirmation=True`。

外部评价能发现错，不等于执行前已经正确拒绝盒外使用。dispatcher_scope_results.json[这两项范围控制](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/pi/dispatcher_scope_results.json)

### 3. 角色改名会跳过盒检查

新检查使用 `for k in vbox if k in pt`，不存在的字段会被跳过。对同一物理任务将 `u_drive/T_horizon` 等价改名为 `u_probe/t_probe` 后，原本会被排除的越盒 Pi 召回可以获准。

原 80 个域内目标中的 **16 个 Pi 目标**仍使用规范字段名，所以不能声称正式结果已因改名绕过。它是跨域迁移接口上的实际缺口。

正确做法是先映射到证书的规范角色，再验盒；未知映射应拒绝该证书授权。

### 4. 正常 Stage2 的节点、边和证书也没有同步更新

正常 v4 节点的范围已缩为 \([2.4,5.2]\)，边上的 ScaleMap 和原证书仍是 Stage0 的 \([2.2,5.2]\)，仅增加新证书摘要。

同一 ODE 缩小范围时，继续使用旧冻结模型可以合法，不应直接判成数值错误。但应该明确表达：

> 同一个已确认实例，在更小范围重新认证。

这需要把实例身份、新盒、新证据和节点／边版本一起提交，而不是让对象各自表达不同状态。PI_AUDIT.md[详细证据链分析](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/pi/PI_AUDIT.md)

---

## 七、“抽屉原理证明图游走不可替代”的论证过强

这一点值得认真修正文稿，因为它影响项目后续的科学论证方向。

### 1. DEV 没有直接看 TEST，但只覆盖三个家族

DEV 使用独立 seed，这点是真的。问题是代码只取 `dev_targets[:10]`，而生成器按家族分组：

| 家族 | DEV 实际数量 |
|---|---:|
| Haldane | 4 |
| Oscillatory | 4 |
| Monod onset | 2 |
| Murray onset | **0** |
| Pi | **0** |

冻结依据还是成功次数，不是整个顺序的真实执行成本。因此它可以称“独立 DEV 上冻结的一种顺序”，不能称“最优静态序”。[DEV 冻结代码](https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L1755)

我没有发现这里直接读取 TEST 成绩；问题是覆盖和优化目标，不是可以据此指控 test leakage。

### 2. “两个模型都过容差”不等于“已有 pilot 无法区分”

当前候选选择器按顺序拟合，首个达到接受阈值的候选就停止。这种策略容易在两个都过阈值的候选之间选错。

但当前无噪声 pilot 中：

- 正确模型残差常在 \(10^{-10}\) 附近；
- 错误模型虽然也低于应用容差，残差仍约 \(10^{-3}\) 到 \(3\times10^{-2}\)。

这些数据有可用判别信息。

限定事后控制中，我固定只比较报告自己的 W4／W5 两个候选，各试两个角色排列，**只按已有六个 pilot 的最小 RMSE 选择**，之后才揭示确认与 heldout：

- **8/8 首个新确认通过；**
- **8/8 独立 heldout 通过；**
- **额外分辨测量为 0。**posthoc_pilot_pair_control.json[完整控制结果](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/unified/posthoc_pilot_pair_control.json)

这不是新的完整强基线，也没有证明计算成本低于 Hegel：它每题进行了四次候选角色拟合，而且只比较这对候选。

它否定的是更强的论断：

> 当前 pilot 必然无法区分分支，必须依靠图或额外物理测量。

### 3. 关系信息的价值与在线 PPR 的独有价值需要分别检验

两个 onset 家族有宿主提供的不同 domain cue，形成阶段显式登记了对应桥和边。Ordinary 列表不消费这些信息，BM25 也没有展开关系传播。

因此当前结果可以支持：

> 这种结构化关系检索优于当前固定顺序和直接匹配策略。

若要进一步说明在线 PPR 的独有收益，可以保留同一森林，同时增加一个使用相同信息、在 DEV 冻结的 cue 路由或关系闭包查询表，并计入构建和维护成本。

**强控制可以继续使用森林作为知识表示。** 它的目的在于分清关系知识、搜索算法和停止规则各自贡献，不要求改掉项目主架构。UNIFIED_AUDIT.md[统一评测专项分析](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/unified/UNIFIED_AUDIT.md)

---

## 八、R3、轨迹投影和维护收益还需要三个精确限定

### 1. Act3 的 optical 拒绝依赖验证函数现场补规则

原始 post-Stage2 森林对 optical 可以退到 fine 并接受。`run_interleaved_zero_fit_dispatcher_stream` 在开始十条查询前，才现场给 fine 节点加入 thermal／hydraulic OR guard。

这次修改没有走正式 revision，节点版本和 semantic digest 都没变。Stage3 使用的又是补规则前的 snapshot。

因此，10/10 验证成立于那个补过规则的副本；尚不能据此证明生产快照同样拒绝 optical。[Act3 实现](https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L1974)

另外，删除全部 coarse edges 后，十条路由和预测完全不变。当前 Dispatcher 实际是在调用者指定的树里做 leaf→mid→root 合约尝试。它仍然有用，但不能把这项实验当成“粗粒化边支配升级”的因果证据。

最小修复是正式构建／修订 guard，让验证流只读；需要沿边升级时复用现有 `fine_scale_children_of`。

### 2. 轨迹末态门控已经接入，窗口均值仍是诊断

依赖控制显示：

- 只改尾窗非末点、保留末态，窗口投影值最大改变 **67.2**，证书不变；
- 只把末点 `x_fast` 加 1，slaving defect 从 **0.016419** 变 **1.016419**，证书从 True 变 False。

所以准确描述是：**真实轨迹末态的快流形残差已经参与证书；窗口平均值尚未参与授权或尺度选择。** 不能再说轨迹完全无作用，也不能用调用次数 6 证明所有投影输出都被消费。projection_dependency_results.json[轨迹依赖控制](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/pi/projection_dependency_results.json)

### 3. 当前维护没有回本，而且边际改善全部来自 Pi

逐行对齐 Hegel 和 NoMaintenance：

- 64 个非 Pi 目标的成本、候选数和召回完全相同；
- 16 个 Pi 的平均成本为 **28.95 对 32.6234375**；
- Stage3 合计节省 **58.775**；
- Stage2 额外花费 **218.45**；
- 全生命周期 Hegel 仍多花 **159.675**。

即 **2990.325 对 2830.650**。交接书承认维护未回本，这一点应继续保留。

它证明当前 Pi 结构／作用域合并对访问流有用，没有分别证明 Minton、Monod 作用域修订和淬火重认证各自的独立收益。receipts_analysis.json[80 对逐行分析](sandbox:/workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/unified/receipts_analysis.json)

---

## 九、下一轮建议只做一个有限的 M4.3 闭环

我建议按下面顺序推进：

| 顺序 | 工作 | 最小验收条件 |
|---|---|---|
| 1 | 保存真实 Stage0，绑定 confirmed instance | Frozen 与真实边界完整一致；只改保存参数必须被拒绝 |
| 2 | 统一尺度证书和范围消费 | 窄证书不能扩盒；最终越盒拒绝或合法升级；角色改名保持授权结论 |
| 3 | 统一新鲜确认协议 | 确认点足额、与训练前缀不重叠；32 个既有触发场景闭合 |
| 4 | 将成本绑定实际动作 | 无关边不能改价；真实 channel 决定费用；OpenRediscovery 实际运行发现；检索费用入账 |
| 5 | 重跑现有主实验 | DEV 分层覆盖；采用可解释的同信息对照；重新报告生命周期收益及未回本部分 |

另外几项可以随这些改动一起处理：

- OR 压缩近期只允许执行对象、权限和尺度语义相同的 scope 合并；异质分支需要真正返回并执行匹配分支。
- R3 验证流改成只读。
- Stage2 新证书和可重定位的 post-Stage2 森林一起归档。
- `program_steps` 明确作为有限声明式计划，清理没有消费者的字段。
- Stage3 的 Alpha0 同时关闭 pooling，应标成合并消融，或补成真正单因子对照。

本轮暂时不需要同时扩展 PDE、更多随机动力学、全新证明系统或更大的世界库。现有 RK4、Schur、PPR、CodeAgent、paid history、范围检查和归档能力已经足够支持上述施工。

对于你最初“新的哲学假设来自历史概念和理论的总结与一般化”的构思，M4.2 最有价值的推进是：**历史已确认产物开始以可执行实例、范围和版本进入后继选择。** 接下来最关键的是证明：在相同历史、信息和工具下，这些关系确实可靠地改变后继行动，并在真实执行账本上产生净收益。当前这五组连接，是推进这个目标最直接的工作。

完整报告中还有逐项 **C1—C12 验收表、源码定位、失败与反方向控制、复现步骤**。页版内容和结构已回读核对，未做视觉预览。