# Hegel V17.4 M4.2 独立审计与下一轮施工建议

审计对象：[固定提交 fb4b23aaf802c20b4e2d0018436abab73b0cca06](https://github.com/erzhu419/Assumption-Agent/commit/fb4b23aaf802c20b4e2d0018436abab73b0cca06)。比较基线：[V17.3 M4.1 提交 99b1fdfeb](https://github.com/erzhu419/Assumption-Agent/commit/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c)。审计日期：2026 年 11 月 20 日。

## 一 总判断

**M4.2 在对象集成和正常执行路径上有实质进展，可以作为继续施工的主线；但目前不能验收“12 项问题 100% 闭合”“统一架构已在第 29／39 题回本”“两跳图游走具有已证明的不可替代性”这三项强结论。**

最应优先修的不是 RK4、Schur、PPR 或组合语法本身，而是它们之间的五组连接：

1. **测量价格必须由实际选择并授权的尺度通道决定。** 当前 Stage3 在检索前根据索引类型和任意未失效粗粒化边，把同一个 oracle 的观测分成半价和全价。
2. **Frozen 对照必须保留真正的 Stage0 全量状态，OpenRediscovery 必须实际运行发现。** 当前前者被裁成四节点且没有边，后者相对共享池仅新增账项。
3. **已付失败观测应保留，但不能再次作为新鲜确认。** 这条错误路径在本提交实际发生 32 次，影响确认协议和成本口径。
4. **确认与执行必须指向同一组冻结参数。** 当前严格 loader 验证重新拟合的模型，却放行另一组未绑定的保存参数。
5. **证书、ScaleMap、节点、边、实际实例和最终查询必须共享同一个范围约束。** 专门 Pi suite 已修好的最终查询守卫，没有接入新的 Stage3；合法窄盒证书也仍能被绑定到更大的盒。

这些问题有固定提交上的源码和有限控制见证。它们不要求推翻目前的森林架构，也不要求证明任意 Python 安全或把有限区域采样升级为全域数学证明。

## 二 复核范围与可复现性

### 2.1 本轮覆盖

相对 M4.1，Git 全量差异为 **420 个路径**，其中 292 个是归档上一轮审计，116 个是新证据文件，核心实现是 5 个源码文件、1 个研究入口和 2 个测试文件，另有交接书、索引、报告和旧审计压缩包。本轮完整读取交接书和 Report 16，复核这些实现的改动及直接调用链，逐行分析 792 条统一评测回执，并沿用此前已完成的历史版本和原始构思审计。

历史复用按仓库 AGENTS 要求从 reuse hub 和 capability catalogue 入手。现成的 paid history、Python 归档、双通道实验室、范围检查、版本修订与冻结执行都已定位；本报告把“调用没有接上”与“算法不存在”分开。历史引用库和初始构思未在该提交变化，没有把重新下载所有旧文献作为本轮的前置条件，也没有声称本轮逐字重读了全部历史版本和文献。

所有控制在独立副本／内存快照中完成。仓库源代码和提交证据没有修改；没有在线模型调用或远程写操作。全部结果、控制脚本和三个专项报告随审计包提供。

### 2.2 实际重放结果

| 项目 | 本轮结果 | 准确解释 |
|---|---|---|
| M0 至 M4.2 指定回归套件 | 37 项通过，退出码 0 | 现有回归通过，不能替代缺失的对象一致性检查 |
| M4.2 完整研究入口 | 四种子完成，退出码 0 | 九臂结果、成本表和原回本数值均重现 |
| Part2 回执 | 432 行，逐字节一致 | 每格 108 个唯一运行，无旧 walk 重计 |
| Part4 回执 | 792 行；除 13 条模型源码 SHA 外，其余字段逐项一致 | 成功率、候选数、召回、误差、各账项全部一致；不称全字节一致 |
| Part1／Part2 汇总 JSON | 完全一致 | 修复见证及去重结果重现 |
| Part3 汇总 | 仅临时 artifact 路径不同 | 科学数值一致 |
| Part4 汇总 | 路径、墙钟时间、约 1e-13 级参数尾数不同 | 九臂统计与累计成本曲线完全一致 |
| 提交证据完整性 | 115 条 SHA 校验全部通过，共 116 文件 | 包括 SHA 文件本身 |
| 新重放目录 | 120 条 SHA 全部通过，共 121 文件 | 比提交多 5 个自动生成的 Python 字节码缓存 |

附件交接书写 120+1，提交内交接书已更新为 115+1；这两份交接书的差异仅涉及三处文件计数表述。实际重放与提交的五文件差异正好是字节码缓存，不应误报为五份科学证据缺失。另有 13 条源码 SHA 的跨运行差异已原样保存；原提交未归档相应 Stage3 源码，不能仅凭回执把这些源码差异的具体成因全部定死。这里不影响已经逐项一致的研究统计，但以后应归档最终执行 artifact 的可解析引用。

完整核对记录：`replay/replay_comparison.json`；回归日志：`replay/pytest_37.log`；全量重放：`replay/evidence/`。[S1][S2][S3]

## 三 本轮真正完成了什么

### 3.1 工程修复有实际行为变化

- `fresh_points=1/2` 现在实际购买不同数量的确认点。独立旧路径控制的复合成本为 40.6452／44.6452，均成功。这修掉了上轮“合法参数被忽略”的具体见证。
- 同法则、同权限的 A/B 作用域合并，真实授权器现在能通过 A 和 B、拒绝 C。thermal／hydraulic 正例成立。
- 单填 `confirmed_ok` 与伪 confirmation digest 的 JSON 已不能通过新 loader。
- 真正 M4.2 W1 和 W2 产物都能独立加载；把 W2 放入 W1 节点并核对 expected AST digest 会拒绝，结构身份绑定已接通。
- 相对父路径和绝对越界路径都进入 containment 检查，旧逃逸 probe 正确抛出 ValueError。
- 三次 lure DEV 变成了不同公式、领域、特征和点集，消除了上一轮同任务重复三遍的问题。
- 每个测试 seed 从同一 post-Stage2 快照恢复；OOS 参数随 seed 变化；失败确认值确实留在后续拟合前缀里。[S4][S5][S6]

### 3.2 Pi 与实例级 R3 已有真实集成

同一 builder 现在形成原子模型和 Pi 证书，发现 W1—W5，再修订作用域、合并热学／水力分支、记录短时淬火失配、缩小有效盒并重认证。这是对象级的集成进展，不能再沿用“Pi 完全没进统一森林”的旧判断。

冻结执行 helper 确实恢复保存的 `_PARAMS` 并移除 `fit`；旧 Monod、新 Haldane、振荡与 capped Murray 的交错实例重访具有实际意义。正确表述是“零目标重拟合的冻结实例推理”：冷加载 loader 仍可能在历史前缀上重新拟合以验证来源，因此末端返回的 `fit_called=False` 不是整条调用链零 fit 的事件计数。[S4][S7]

### 3.3 去重与专门 Pi 实验的改进应保留

| Part2 臂 | 唯一运行数 | 平均复合成本 | 非直接 walk 胜出 |
|---|---:|---:|---:|
| Walk + Pool | 108 | 37.4591 | 14/16 |
| Walk，无 Pool | 108 | 37.6952 | 14/16 |
| Alpha0，无 Pool | 108 | 40.2867 | 0/16 |
| Alpha0 + Pool | 108 | 40.3052 | 0/16 |

这组真正的 2×2 结果保留了图游走的正向贡献。专门 Pi suite 也已把无边回退的重复 pilot 去掉：有边 72.6095，无边 73.7765；真实差额较旧报告小，但仍然存在。该 suite 的确认越盒升级和最终盒外查询拒绝已进入真实路径。[S8][S9]

## 四 P1 统一成本目前没有绑定实际尺度执行

### 4.1 价格在检索与授权之前决定

`evaluate_m4_2_target_under_arm` 先判断：

- 当前目标属于 `PiS_Adiabatic_CrossDomain_2D`；
- 森林含任意未标 stale 的粗粒化边；
- 索引是 `HippoRAGWormholeIndex`；
- `damping_alpha > 0`。

满足即每观测 4，否则每观测 8。随后两者都构造同一个 `OpenWorkspaceTaskLab`，调用相同的 `target.oracle_fn`。这个条件没有要求粗粒化边与当前 ODE 有关、被检索选中、通过当前范围授权或实际运行粗通道。[S10]

这使 Alpha0／BM25 与 Hegel 的差额不仅包含检索策略，还直接包含价格。多尺度观测可以有不同成本；缺口在于成本必须跟随真实通道动作，而不能跟随实验臂身份。

### 4.2 删除相关边但保留无关边的决定性控制

使用原 builder 的真实 post-Stage2 森林与原 T17 目标：

| 状态 | 实际观测数 | 最终模型 | heldout RMSE | 测量费 | 复合费 |
|---|---:|---|---:|---:|---:|
| 完整森林 | 7 | Pi coarse 节点 | 0.0 | 28 | 28.950 |
| 删 ODE 粗边，只留 hydro 粗边 | 8 | atomic exp-relax fallback | 0.005173 | 32 | 34.825 |
| 删除全部粗边 | 8 | 同一个 fallback | 0.005173 | 64 | 66.825 |

后两行的观测阶段、点和数值摘要完全一致，候选数同为 7，所选模型和独立 heldout 一致。保留一条与 ODE 无关的 hydro 边就让相同执行半价。这直接证明当前价格没有由实际 Pi 路径产生。

证据：`cost/cost_control_summary.json`，每个控制的完整 acquisition 记录也在 `cost/`。

### 4.3 检索费与验证成本还需要归入账本

同一控制里 PPR 检索返回 `query_compute_cost=2.745`，回执没有对应 charge；新执行器仅在末尾按 `2.5 + 6 × evaluated_candidates` 计一次合成计算。loader 的冷验证和缓存命中也没有单列。DEV 的候选拟合次数同样未完整对应其固定计算账项。[S10][S11]

无需把所有机器墙钟都当作唯一标准，但“计量代理”必须有清楚定义，并对每臂的实际事件一致应用。观测事件、检索、候选拟合、冷验证、缓存复用各自有明确归属，才能解释平均成本和长期回本。

**最小修复**：复用现有 `MultiScaleWorkspaceTaskLab.acquire_channel`，只有实际通过证书与范围检查并选择 coarse channel 后才能用粗尺度费用；fine、coarse 的 acquisition receipt 记录 channel、边及版本、证书和单位费。不要再另建第三套实验室。

## 五 P1 Frozen 起点与 OpenRediscovery 没有对应命名的能力

### 5.1 Frozen 丢失已经付费形成的 Stage0 能力

在第一轮 Stage1 调用之前直接捕获的真实 Stage0 有 **8 节点、2 条 coarse edge、1 条 wormhole**。九臂入口却新建一个空 forest，只复制 Monod、Murray、hydro micro、ODE micro 四节点，没有任何边。被删掉的是两个 onset 桥、thermal／hydraulic 两个 Pi coarse 节点，以及全部关系。两个 Frozen 臂仍承担完整 244.05 的 Stage0 费用。[S12]

这不等于“冻结以后不能学习新知识”；它改变了冻结时已经拥有的知识。

在相同 evaluator 和 fallback pool 下，恢复真实 Stage0 的有限控制是：

| 目标 | 原四节点 Frozen | 真 Stage0 | 解释 |
|---|---:|---:|---|
| T09 Monod onset | 39.50 | 40.10 | 同 9 次观测、同结果；多两个原子候选评估 |
| T17 Pi | 57.55 | 29.55 | 同 7 次观测、同 3 次候选评估、同 fallback 与误差；只是触发 8→4 的价格变化 |

恢复完整起点并不保证每个目标更便宜，不能从这两例制造新的 80 题均值。它们已经足以否定原 Frozen 等起点定义。证据：`unified/frozen_snapshot_control.json`。

**最小修复**：在真实 Stage0 边界保存并返回完整 snapshot，后续直接 deepcopy。验收比较节点、边、历史、证书和实例引用，不按节点名挑选重建。

### 5.2 本轮 Stage3 的 OpenRediscovery 相对 PoolFallback 只多记账

`open_code_agent_fallback=True` 的分支仅执行：

```python
lab.meter.charge(
    "open_compositional_rediscovery_diag",
    obs=1, meas=obs_unit_cost, compute=24.0,
)
```

之后两臂用同一个 `fallback_pool_specs` 调同一个 `_autonomous_fit_candidate_specs`。没有额外 CodeAgent episode、诊断观测或组合搜索。[S13]

原 T01 的直接控制：

| 项目 | PoolFallback | OpenRediscovery |
|---|---:|---:|
| 实际获取观测 | 7 | 7 |
| 账本记录观测 | 7 | 8 |
| 候选评估数 | 10 | 10 |
| 所选模型、确认与 heldout | 相同并成功 | 相同并成功 |
| 复合费 | 31.65 | 36.85 |

这一额外记账在 80 题合计 **480** 复合成本，正好解释两臂均值的 6.0 差额。Stage1 的 CodeAgent 确实运行，不能把本发现扩张成“整个项目没有开放发现”；错误只在这条新 Stage3 对照把费用条目当成了发现过程。

**最小修复**：复用已有 `run_compositional_code_agent_episode` 接入真正 Frozen 开放重发现；如果暂时只想用指定费用的代理基线，就如实命名为估算，并把它从“实测回本”结论中分离。

### 5.3 原回本数字的敏感性

保留原执行决策、观测顺序、候选数、被裁剪 Frozen 快照和前期投入，只做两项账本重计：

1. 仅将 Pi 目标的观测统一按单价 4，或统一按单价 8；
2. 去掉 OpenRediscovery 没有对应执行的额外诊断账项。

| 比较 | 原账本首次交叉 | 同价且去掉空记账后首次交叉 | 原第 80 题净省 | 重计后净省 |
|---|---:|---:|---:|---:|
| Hegel vs Frozen Pool | 39 | 74 | 482.525 | 34.525 |
| Hegel vs Frozen Open | 29 | 74 | 962.525 | 34.525 |

分解很直接：`482.525 − 448 = 34.525`；`962.525 − 448 − 480 = 34.525`。两种同价情形净差一样，是因为 Hegel 和 Frozen 在这批 Pi 任务中都获得 112 次观测，单价 4→8 时两边同增 448；不是额外独立的稳健性证据。

**74 和 34.525 只是事后计量敏感性，不是修复后的公平主结果。** 尚未恢复完整 Frozen、修复确认耗尽、执行真正开放发现或补齐费用；这些改动都可能改变决策。它说明原 29／39 对计量设计高度敏感，不能继续作为已证实的架构收益。

同价 4 的原记录均值也从 Hegel／DEV 的 29.22／38.855 变为 29.22／32.855；原 9.635 的差额有 6.0 来自 Pi 定价。仍有剩余差额，不能因此宣布图无收益；需要在对等执行条件下重新测它。证据：`cost/accounting_sensitivity.json`，已由另一审阅分工从原 792 行独立复算。

## 六 P1 已付证据保留后又被用作新确认

新代码把失败确认加入 `frozen_prefix` 是正确修复。但 fallback 在没有未使用确认点时，强制取 `target.confirm_points[-1]`，然后用包含该点的前缀拟合，再在同一点确认。[S13]

这不是只存在于理论边界：提交的 792 条记录中，**DEV 与 Forward 各 16 个 Murray onset，合计 32 次**走这条路径。

原 DEV、seed 20261120、T13 的执行 trace：

| 阶段 | 输入点 | 拟合前缀行数 | 与训练集重叠 | 确认 RMSE |
|---|---|---:|---|---:|
| 第一次 ordinary confirm | (11.5, 0.6) | 6 | 否 | 9.7110 |
| 第二次 ordinary confirm | (8.5, 1.0) | 7 | 否 | 0.55288 |
| fallback confirm | (8.5, 1.0) | 8 | **是** | 2.85e-10 |

这 32 条独立 heldout 仍全部成功，本例五个独立点 RMSE 约 7.16e-9。准确诊断是“在线确认独立性破坏，并重购无新增信息的确定性观测”，**不能改写成 32 条错误预测或 32 条 false confirmation**。

同根的 P2 边界还包括：请求 fresh=3 但仅有两个点仍通过；旧 M4 的父级耗尽两个点后，子级重用已进入前缀的最后一点。fresh 参数已经被消费，但“至少多少新点”这个契约还没有完整执行。

**最小修复**：继续保留已付失败点；确认时记录训练 prefix digest 与查询 digest，并验证所需独立点足额。点池用尽就用现成 probe 机制购新输入，或诚实返回证据不足。不要删除已付历史，也不要把重复购点包装成新证据。

证据：`unified/receipts_analysis.json`、`unified/confirmation_reuse_control.json`、`closure/fresh_budget_results.json`。

## 七 P1 确认实例与冻结执行参数脱钩

### 7.1 严格 loader 没验证真正要执行的那组参数

现在的 provenance 链绑定源码 SHA、AST、训练行、确认行和误差；loader 将源码重新放进临时目录，在历史 prefix 上 fit 后验证。冻结执行则读取 artifact 中的 `fitted_params` 覆盖 `_PARAMS`。manifest 没有参数摘要，外部 `confirmed_fitted_params.json` 也未被 loader 消费。[S4]

因此，“重新拟合能通过”与“保存实例通过”是两个不同命题。

### 7.2 只改一项参数就能通过 loader 并错误执行

以提交内真实 W1 完整工作区为基线，同时开启 `require_external_manifest=True` 和正确 `expected_ast_digest`：

| 条件 | 严格 loader | 原确认集上的冻结预测 RMSE |
|---|---|---:|
| 原始 artifact | 接受 | 5.487e-10 |
| 仅把 artifact 的 p0 乘 3 | **仍接受** | **7.689592855** |

容差为 0.045。源码、AST、bundle、外部 manifest、外部参数文件和确认数据都没有变化。两次 frozen helper 都返回 `fit_called=False`。

这说明保存状态损坏／错配没有被实际实例校验发现；不说明当前正常 W1 已经错误，基线正例非常准确。

**最小修复**：保存并绑定实际通过确认时的参数态，把源码、参数、角色／透镜和范围组成 confirmed instance identity；严格实例 loader 直接用该冻结态预测原确认行，并与 manifest 一致核对。冷加载可以校验缓存，不能用另一次 fit 代替实例验证。现有归档组件即可扩展。

证据：`closure/contract_controls_results.json`。

## 八 P1 尺度授权链在新消费者中仍有断点

### 8.1 合法窄盒证书能被绑定到更大的盒

新重验证函数确实检查了 certified、defect、tolerance、closure_scope 和消元变量，但没有比较证书盒与拟提交 ScaleMap 的盒，也没有绑定具体模型／参数／版本。[S14]

用原函数生成的真实六点证书，T 范围为 [2.4,5.2]，最大闭合缺陷 0.032846。保留整份证书，只把拟提交 ScaleMap 扩成 [0.05,5.2]，重验证仍 accepted，节点到 v5，粗边恢复。

同一个 u=2、T=0.12 的查询，原森林选择 fine、误差为 0；扩大盒后选择 coarse、获得 local solver guidance，误差 **0.32400957**。

这没有要求六点采样排除任意窄 spike，只要求一份已有窄域证据不能自动授权更广的域。当前 builder 正常运行只缩盒，不能把该控制当作原 80 个域内目标已失败。

### 8.2 正常重验证后节点和边仍保存不同范围

正常 post-Stage2 coarse 节点 v4 的盒已是 [2.4,5.2]；相关 edge 的 ScaleMap 和 empirical certificate 仍是 Stage0 的 [2.2,5.2]，仅另写一个新证书摘要。节点 artifact 也继续引用 Stage0；新 Stage2 标定产物没有成为被引用实例。[S14][S15]

同一 ODE 只缩小域时，旧冻结模型可以继续有效，这不是已证数值错误。应明确表示“同一实例在小域重新认证”，将模型身份、新盒、新证据与节点／边版本一起提交。现在三种对象表达的是不同状态。

### 8.3 新 Stage3 没继承专门 Pi suite 的最终范围守卫

Stage3 只检查 pilot 和 confirm 的盒。它在后面直接执行最终 heldout 查询，没有继承专门 Pi suite 已存在的 `model_scope_contract.json` 检查。[S9][S10]

只将原 T17 的最终查询改为同一 ODE 的 u=2、T=0.12，pilot、确认和森林不变：

- 原目标 strict pass；
- 改后仍 `completed`、`confirmed_via_recall=True`；
- 外部评价得到 RMSE **0.218132**、`false_confirmation=True`。

评价器能识别错误不等于授权器已经拒绝盒外使用。最终调用应先拒绝、返回 unknown 或经有效路径升尺度。

### 8.4 角色改名会跳过盒字段

Stage3 与 Dispatcher 使用 `for k in vbox if k in pt`，缺失字段被跳过；空的 `all(...)` 为 True。原 80 个域内目标中的 16 个 Pi 目标仍使用 u_drive／T_horizon，不能指控它们已经因改名绕过。

但对同一物理任务做可逆角色改名 u_probe／t_probe，保留越盒的 T=1.2 确认点：原名时 Pi 召回被排除；改名后 coarse v4 获准并确认。正确迁移应先映射至证书的规范角色，再检查盒；映射未知就拒绝该证书授权。

**最小修复**：复用旧 Pi 模块的严格范围 helper 和现有角色映射，统一召回、fallback、R3、导出／续问的检查。重验证同时维护节点、边、证书与实例身份。把这些同根问题作为一个集成修复包，不必分成数套新系统。

证据：`pi/dispatcher_scope_results.json`。

## 九 R3 分发器的价值与当前证明边界

### 9.1 optical 拒绝来自验证流现场修改的副本

fine 节点原本只要求 active_arity=2。原始 post-Stage2 森林直接处理 optical 会选择 fine 并 `dispatched_ok`。`run_interleaved_zero_fit_dispatcher_stream` 在运行 10 条查询之前，现场加入 thermal／hydraulic OR guard，之后 optical 才被拒绝。它没有走正式 revision，版本仍是 1、semantic digest 未改变。[S7]

主程序把 snapshot 的 deepcopy 交给这个验证流；Stage3 从未加该 guard 的原 snapshot 继续建臂。因此“同一个语义状态在 Act3 和 Stage3 都被验证”尚不成立。

修复很小：把 modality 合约放入正式构建或带证据的版本修订；验证流只读运行，不补规则。

### 9.2 当前升级是给定树内的合约回退

Dispatcher 由调用者指定 candidate_tree_id，然后按 leaf／mid／root 排序尝试同树节点。它没有消费 coarse edges。删除真实森林的全部 coarse edges 后，10 条查询的路由与预测逐项不变。[S7]

这不否定旧新实例共存、范围选择与冻结推理的价值。准确名称是“给定候选树的合约分发与父节点回退”。若要主张沿有效粗粒化边升级，现有 `fine_scale_children_of` 和版本检查可直接复用，并让 stale／删除边产生与声明一致的影响。

fine 分支本身是宿主提供的可信精确 ODE 求解器，并以固定标识和即时构造的成功记录授权；它与经过确认 bundle 验证的发现实例属于不同来源等级。保留可信求解器是合理的，但应记录实际配置 hash 和授权来源，避免把两者混成同一种“已确认发现”。

## 十 对评测设计与理论论证的诊断

### 10.1 DEV 独立成立，最优静态序不成立

DEV 用独立 seed，未发现其读取四个 TEST seed 的成绩。问题是 `dev_targets[:10]` 对按家族分组的生成器，只取到 Haldane 4、Osc 4、Monod onset 2，完全没有 Murray onset 和 Pi。排序依据是候选成功次数，不是完整顺序的真实运行成本。[S16]

因此 38.855 应描述为“在部分家族的独立参数 DEV 上按命中频次冻结的一种顺序”。Reverse 的 36.295 更低但有 1/80 Pi 假确认，不能拿它直接宣布最优；同样不能称 DEV 已是最强可比基线。

近期用五家族分层 DEV、预先冻结评分目标和顺序即可，无需搜索所有阶乘排列。后续若为修正后的主实验发布新泛化结论，保留未参与当前诊断的新 TEST seed。

### 10.2 两个模型均在容差内，不代表已有 pilot 无法辨别

工作总结用“两个分岔模型在低负载都过容差”推出“任何静态列表至少一半必猜错，唯有两跳森林一次命中”。这个推理需要限定为当前 first-acceptable 策略：候选按序拟合，首个低于 0.85×tolerance 即停止。[S17]

已有六个无噪声 pilot 中，正确模型往往残差约 1e-10，错误模型虽仍在应用容差内，却约 1e-3 至 3e-2。接受阈值没有消除这些信息，只是停止规则不再使用它们。

限定事后控制：固定只比较报告自己的 W4／W5 两个算子，各试两种角色排列，使用相同六个 pilot，**仅按 pilot RMSE 最小值选模型**，之后才揭示首个确认和 heldout。seed 20261120 的八个 onset 全部一次新确认通过，独立 heldout 也 8/8，额外分辨测量为 0。

这只是可判别性见证，不是新的完整强基线：每题拟合四个角色候选，没有证明其总计算费低于 Hegel，没有涵盖全历史池，也没有回答噪声情形。它足以否定“这些 pilot 信息论上无法辨别、一定要额外物理测量”的强表述。

证据：`unified/posthoc_pilot_pair_control.json`。

### 10.3 关系信息有用与在线 PPR 不可替代需要区分

两个 onset 家族由宿主提供不同的 domain cue，形成阶段显式登记桥、横边和 specialization。Ordinary 不消费这些 cue；FlatBM25 虽保存节点元数据，打分仍集中在直接 probe signatures，不展开关系传播。[S11][S17][S18][S19]

因此当前证据支持“这种结构化关系检索优于这些固定顺序／直接匹配策略”。要分离关系知识本身与在线 PPR，可在保留森林的条件下，用 DEV 冻结 cue→候选优先级，或把历史关系闭包编译为查询表并计入构建／更新成本。强控制可以保留同一知识表示，不要求把主产品退回扁平列表。

Stage3 的 Alpha0 还同时关闭 pooling，是合并消融；Part2 的正交因子结果不能自动替代 Stage3 的单因子归因。

### 10.4 维护尚未在当前访问流回本

提交已诚实披露 NoMaintenance 全程更便宜。逐行比较又显示：64 个非 Pi 目标两臂的成本、候选数、召回完全一样，全部改善来自 16 个 Pi。

| 项目 | Hegel | NoMaintenance |
|---|---:|---:|
| 非 Pi 四家族 | 每题完全相同 | 每题完全相同 |
| Pi 平均成本 | 28.95 | 32.6234375 |
| Pi 召回确认 | 16/16 | 0/16 |
| 80 题 Stage3 总节省 | 58.775 | 参照 |
| Stage2 额外投入 | 218.45 | 0 |
| 全生命周期总价 | 2990.325 | 2830.650 |

Hegel 仍贵 **159.675**。这支持 Pi 的结构／作用域合并对当前访问有用，没有分别证明 Minton、Monod 作用域修订和淬火重认证各自的独立收益。不能把边际均值更低写成全部维护已经回本。

### 10.5 泛化口径

四个 seed 是一次形成链后的四组参数迁移；Stage3 没把 80 个目标重新学习入林。Pi 迁移随机化读出 scale／sign／offset，内部 ODE 参数仍固定。OOS 随 seed 变化是真改进，但仍为少数固定构造模板的参数变化。W3 确实在 Stage1 被发现并有实例重访，却未作为五个 Stage3 家族之一测试跨域迁移。

这些限制不要求本轮立即扩大世界库。它们确定当前证据是受控机制与参数迁移验证，而不是一般开放世界 RSI 成功。

## 十一 另外几项应收窄的工程声明

### 11.1 OR 分支只控制了 scope

真实控制：同法则 A 具有 bounded_trial 权限、B 只有 retrieve_candidate，互斥作用域合并后 B 被提升为 bounded_trial，因为授权上限仍读 keeper 顶层。另一控制中，互斥域的 Murray 与 capped Murray 被接受合并，B 域授权通过但消费者仍读取 A 的顶层 action。[S5]

当前同法则同权限 thermal／hydraulic 正例不受此反例否定。近期可把 scope 合并限制为相同执行对象、权限、支持状态和尺度语义；若确实要容纳异质分支，就让授权返回完整匹配分支并由执行消费。

### 11.2 轨迹末态残差已接入，窗口平均值仍是诊断

只改尾窗非末点、保留末态时，window_projected_y 最大变化 67.2，证书与两类 defect 不变；只把末点 x_fast 加 1，slaving defect 从 0.016419 变成 1.016419，certified 从 True 变 False。[S15]

应保留“真实轨迹末态快流形门控已完成”，把“窗口平均输出参与授权／尺度决策”收窄为未接通。不能因调用次数为 6 就把所有返回量都视为被消费。若窗口量暂时没有明确研究任务，直接保留诊断字段即可，不必为了审计再增加一个无目的目标。

### 11.3 program_steps 仍是有限声明式计划

当前解释器把列表转成 step map；顺序反转仍返回同一计划。verify_scale_closure 标志也没有相关执行器消费者，尺度保护是宿主固定流程实现的。这可以是一种合法设计，但应称声明式配置，并拒绝或删除无消费字段；不必升级成通用虚拟机。

### 11.4 重验证后的恢复材料不完整

runner 只复制 Stage0 sources 与 Stage1 workspaces，Stage2 完整证书和重定位后的 post-Stage2 forest 没有持久化。报告保留了新盒、计数、accepted，却不能直接恢复当时 v4 的完整证据绑定。[S2]

应复用现有 checkpoint／归档引用保存完整森林、每个实际引用 artifact、参数、确认 bundle 和新证书。若是旧实例在小域重新认证，保存实例引用加新证书即可，不需复制整套历史。

## 十二 对 C1 至 C12 的验收意见

| 项 | 本轮判定 | 剩余范围 |
|---|---|---|
| C1 steps 参数 | 原 fresh=1/2 见证闭合 | fresh 足额、耗尽及声明式语义仍需统一 |
| C2 OR 授权 | 同法则同权限正例闭合 | 分支权限与 action 没随 scope 一起消费 |
| C3 宿主确认 | 无证据自洽 JSON 负例闭合 | 严格实例验证未覆盖保存参数；兼容入口保证较弱 |
| C4 节点摘要绑定 | 已闭合原 W1/W2 调包见证 | 后续可与实例身份合一 |
| C5 路径隔离 | 已闭合原相对逃逸见证 | 无需扩大为任意 Python 安全证明 |
| C6 证书重验证 | 增加实质检查 | 证书域／模型绑定与边证据更新未闭合 |
| C7 去重 | 已闭合 | 432 行逐字节重现 |
| C8 Pi 盒约束 | 专门旧 suite 已闭合 | 新 Stage3 最终查询和角色映射没有继承 |
| C9 轨迹依赖 | 末态 slaving 真接入 | 窗口均值仍诊断 |
| C10 失败确认保留 | 已付数据留存已修 | 实际 32 次重复训练点确认 |
| C11 参数保存和计数 | 参数快照及真实诊断计数存在 | 保存参数未与确认／执行绑定 |
| C12 Stage0／DEV／快照／OOS | 多项具体旧见证已修 | Frozen 快照另被裁剪；成本执行归因仍有问题 |

所以结论是“多项具体旧见证完成，跨消费者的一致性未全部完成”，而非全部失败或全部通过。

## 十三 建议下一轮只做有限的 M4.3 闭环

| 顺序 | 施工对象 | 复用已有能力 | 最小验收 |
|---|---|---|---|
| 1 | 真 Stage0 快照与实例身份 | forest deepcopy、归档 manifest、frozen helper | Frozen 与真实边界完整一致；只改参数必拒绝 |
| 2 | 统一尺度授权对象 | 严格 box helper、角色映射、现有 edge revision | 真窄证书不能扩盒；Stage3 最终越盒拒绝或合法升级；角色改名保持授权 |
| 3 | 全路径新鲜确认 | paid history、现有 probe | 任何确认均足额且不与训练前缀重叠；32 个既有触发场景闭合 |
| 4 | 真实动作计量 | 双通道实验室、CostMeter、检索费用字段 | 相同事件同价；无关边不改价；OpenRediscovery 产生实际 episode |
| 5 | 可比主实验与报告 | 当前九臂 runner、分层 DEV、已有 2×2 | 修复后的生命周期表；同信息控制；明确未回本和适用范围 |

其中 R3 stream 应改为只读，Stage2 证据随快照归档；OR 近期只开放执行语义相同的 scope 合并。它们都能在现有组件中完成。

先把上述已证缺陷接通，再重跑现有主实验即可。不建议现在同时扩展 PDE、随机动力学、新证明系统、更多文献或更大的世界库，也不要求把所有可能边界一次性补完。若之后要做一项泛化扩展，再从有噪 pilot、未见 cue、不同 ODE 参数或合理失败访问流中选择与目标最相关的一项。

对于最初“新假设来自历史概念和理论的总结与一般化”的构思，M4.2 最有价值的推进是：历史已确认产物开始以可执行对象、范围和版本进入后继选择。当前最关键的科学问题是：**同样的历史、信息和工具下，这些关系是否可靠地改变后继行动，并在真实动作账本上带来净收益。** 修复本报告的五条连接，比再增加一组更有利的任务数字更能推进这个问题。

## 十四 证据文件与阅读顺序

完整包包含：

- `M4_2_AUDIT.md`：本报告。
- `replay/`：37 项测试日志、全量研究重放、与提交的逐字段差异。
- `cost/` 与 `cost_controls.py`：实际 acquisition、价格反事实、Open／Pool 对照与账本敏感性。
- `closure/CLOSURE_AUDIT.md`：参数、OR、fresh、路径与旧 probe 的完整复核。
- `pi/PI_AUDIT.md`：证书／盒／边／实例／最终查询、Act3 与轨迹依赖。
- `unified/UNIFIED_AUDIT.md`：真 Stage0、DEV、32 次确认重用、分岔判别与维护差额。
- `changed_paths.tsv`、`change_scope.json`、`core_patch.diff`：固定提交差异与范围。
- 各专项的 JSON、控制脚本、共享构建和森林快照；共享构建避免各专项重复形成同一历史。

原始 JSON 保留控制的失败与反方向结果。便携复现的路径说明与执行顺序见包内 README；历史绝对路径作为原始证据保留，不将其冒充跨机器直接可用的路径。

## 固定源码来源

以下链接均固定到本轮提交，数值控制以包内 JSON 为准。

[S1]: https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/16_V17_4_M4_2_Audit_Closure_Unified_Pi_and_Dispatcher_Report.md
[S2]: https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/scripts/run_v17_4_m4_2_unified_study.py
[S3]: https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/docs/evidence/v17_4_m4_2_unified_closure_and_pi_lifecycle_20261120/part4_unified_execution_receipts.jsonl
[S4]: https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_compositional_discovery.py#L1005
[S5]: https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L248
[S6]: https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_multiscale_renormalization.py#L1141
[S7]: https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L433
[S8]: https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L90
[S9]: https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py#L413
[S10]: https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L2255
[S11]: https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/hipporag_wormhole_index.py#L460
[S12]: https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L2582
[S13]: https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L2498
[S14]: https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L1068
[S15]: https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L252
[S16]: https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L1755
[S17]: https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py#L1400
[S18]: https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L2015
[S19]: https://github.com/erzhu419/Assumption-Agent/blob/fb4b23aaf802c20b4e2d0018436abab73b0cca06/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L831

