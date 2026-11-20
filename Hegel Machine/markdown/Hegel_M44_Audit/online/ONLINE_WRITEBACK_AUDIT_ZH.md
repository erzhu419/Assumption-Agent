# V17.6 / M4.4 Part 3 在线写回、域间证据与保持性专项审计

审计提交：`b965f022b214f47a1341559f20041da6a224d66d`；对比 `3461535364fcdb4e9ae9c87ccba502b7e6b56fe8`。本专项只读源码及提交证据，在审计目录保存控制脚本与输出，不修改仓库、历史结果或远程，不调用模型 API。主审另行运行全部官方测试与研究；本专项不重复完整研究。

## 一、结论

**Part 3 已经做出真实的进程内“先发现结构、写入活森林、后续任务召回该结构”的闭环。后续 12 个目标的调用路径确实消费了 W6/W7 新节点。不能再用 M4.3 的“整个目标流完全只读”描述这一版。** 但“有状态在线写回”目前有三层应分开的含义：

1. **已实现：** 预先指定的 S02、S09 运行原生组合式 CodeAgent，使用真实已付训练与新确认记录形成 AST；当前进程立即入林，索引刷新，后续同族任务重新拟合、确认后通过 heldout。
2. **尚未完整交付：** S01 的所谓作用域修订只改变未消费的 `protocol_spec`，既没有把 `b <= 1.6` 接入 gate，也不以确认成功为前提；新在线知识没有随证据包持久保存，主研究进程结束后只剩汇总回执。
3. **未由该实验证明：** 自动从两个真实领域分别发现同构机制、一般持续流中的自动写回决策、虫洞是 12 次召回成功的必要原因、或所有已学实例均实现数值保持。这些边界不等于现有 20 题预测失败。

其中在线状态持久化问题由本专项核实，索引/成本专项交叉确认，主审负责对完整官方 replay 输出作最终复核。S01 的两项控制及全部 80 条 committed receipts 的重算均在本目录。

## 二、什么是实际成立的

### 2.1 20 步确实都是有 heldout 的预测任务

`run_m4_4_online_writeback_and_multicue_stream` 每一步都调用 `evaluate_m4_3_target_under_arm`，之后才做写回。评估器最后对目标自有 heldout 点运行隔离预测，`strict_pass = confirmed_ok and h_pass`；S01/S02/S09 虽然同时承载事件，也经过这条预测路径，没有用“版本修改成功”冒充一次预测成功。[1][2]

读取全部 80 条提交回执：

| 臂 | 实际回执 | strict pass | false confirmation | 12 个下游目标中 recall 成功 | 下游 `non_direct_walk_win` |
|---|---:|---:|---:|---:|---:|
| PPR PoolOn + LiveWriteBack | 20 | 20 | 0 | 12 | 0 |
| PPR PoolOff + LiveWriteBack | 20 | 20 | 0 | 12 | 0 |
| Compiled + LiveWriteBack | 20 | 20 | 0 | 12 | 0 |
| Frozen PostStage2 + OpenRediscovery | 20 | 20 | 0 | 0 | 0 |

四臂所有确认与训练前缀最大重叠均为 0。PoolOn/PoolOff 的 20 条回执去掉 `arm` 字段后逐项完全相等，不只平均值相等。完整重算见 `online_receipts_analysis.json` 与 `analyze_online_receipts.py`。

### 2.2 W6/W7 的数学结构来自真实 CodeAgent 组合路径

主调用先在活森林检索与拟合，召回失败后创建真实 `OpenWorkspaceTaskLab`，把当前已付前缀转成 observations，进入 `run_compositional_code_agent_episode`。只给 Stage0 原子基元，没有把 W6/W7 真值函数或相应预制复合 spec 作为候选传进去。Provider 在普通 Python 诊断程序读取 `model_observations.json` 后选择已有结构组合子；候选实际拟合和隔离新确认都执行。[3][4]

本版相对 M4.3 在组合器中新增的是通用的 `round2_1d_unresolved` 触发：当一维残差第二轮仍超容差时，尝试二次抑制分母与 ceiling cap。W6 的 committed selected node 为 `open_code_agent:atomic_michaelis_1d__comb_quad_inh`；W7 为 `open_code_agent:atomic_exp_relax_2d__comb_shunt`。因此不能将 W6/W7 说成 host 直接灌入已拟合真值公式，也不能把这个零在线 LLM 的确定性组合搜索说成运行时 LLM 自主发明了新组合语法。[3][4]

注意 W7 实际选择的语法是指数松弛基元加 shunt；报告任务函数是应力松弛超调。两个表征可以在镜头与参数校准后表达相同函数，节点名中的 `cap` 及新模块开头“Michaelis + ceiling cap”描述已与真实执行不一致，应按实际保存的 AST 描述。

### 2.3 后续 12 次 recall 真的消费新节点，但其含义是结构复用后重拟合

三条 live 臂的 S03–S08 最终 selected node 均是 W6 主节点或 twin，S10–S15 均是 W7 主节点或 twin。评估器读取该节点绑定的已验证 authored artifact，取出其 operator spec，然后调用 `_autonomous_fit_candidate_specs` 用当前目标 pilot 重新拟合，另购一条 disjoint fresh confirmation，再测 heldout。[2]

这是真实的结构记忆收益。它同时说明：

- 下游 12 次不是直接拿 S02/S09 的原始参数做 zero-fit 预测。
- 不能把“new W6/W7 zero-fit”那两个流尾单点检查和这 12 次参数变化的迁移混成同一能力。
- Frozen 对照禁止写回，因而每个同族后继重新运行组合搜索；`+85.0283` 首先是允许保存结构相对禁止保存结构的收益。
- live Compiled 也获得 12/12，因此这些成功本身不能隔离 PPR 或虫洞的独立贡献。PPR/Compiled 的 `0.4550` 费用拆分由成本专项给出。

## 三、确定缺口 O1：S01 不是生效的作用域修订，而且确认失败仍写版本

### 3.1 实际代码

任务构造注释称 S01 修订 `b <= 1.6`，`wb_meta` 又给了一个 `b_bleed_ceiling_safe == True` 的 `added_precondition`。但写回分支不读取这个字段，也不修改 `preconditions`；它只加：

```python
upd_cc["protocol_spec"] = {
    **upd_cc.get("protocol_spec", {}),
    "online_bleed_verified_max": 1.6,
}
```

然后调用 `empirical_revise_node` 和边重绑定。这个分支不检查 `rec["confirmed_ok"]`，而后面的 W6/W7 新发现分支才有这项检查。[1]

全仓库 `src`、`tests` 中 `online_bleed_verified_max` 只有这一处写入；`b_bleed_ceiling_safe` 只有任务元数据声明。真实授权器检查显式 preconditions、disjunctive branches、scale map、wormhole alignment 等，没有消费这个新 protocol 字段。[5]

### 3.2 有限控制一：失败回执依然升级版本

`probe_s01_failure.py` 直接调用官方 Part3 函数，仅将 S01 evaluator 结果设为 `confirmed_ok=False, strict_pass=False`，到 S02 前立即停止。无新增观测、拟合或模型调用。观察到：

- W4 从 v1 升至 v2；
- 结论/action 及三条 preconditions 原样；
- protocol 增加 `online_bleed_verified_max: 1.6`；
- 没有增加 `b`、`b_bleed` 或 `b_bleed_ceiling_safe` 的边界条件。

这验证的是失败分支不受确认约束，不声称提交中的 S01 本来失败。

### 3.3 有限控制二：真实 gate 与 zero-fit dispatcher 的边界行为不变

`probe_s01_scope.py` 从 committed postStage2 冷载同一真实 W4 已确认实例，分别使用修订前后节点契约，调用真实 `authorize_abstraction_use` 与 `ForestScopeDispatcher`。保留原先要求的 arity/非 Haldane/secondary bleed 元数据，比较 `s=4.0` 下 `b=1.5` 与 `b=1.7`：

| 输入 | v1 | S01 后 v2 | 预测 |
|---|---|---|---:|
| b=1.5 | bounded trial 与 local guidance 均授权 | 同样授权 | 7.50857151 |
| b=1.7，超过所称 1.6 边界 | bounded trial 与 local guidance 均授权 | 同样授权 | 7.65257151 |

所有预测 `fit_called=False`。这是作用域边界根本没进入调用方的直接结果，不是精度反例。输出见 `s01_scope_control.json`。

**诊断：P2，在线修订语义与报告不一致；不宜把这个事件算作已验证的“经验驱动作域修订”。** 它确实测试了版本变更后的索引刷新，但在当前流里是一次 protocol 元数据更新。Part1 使用真实 precondition 的动态索引测试仍有价值，本发现不否定那项修复。

**最小处理：** 要么据实把 S01 改称“协议元数据变更/索引失效事件”；若确要声称作用域修订，则复用已有 predicate/role-map gate，将实际获得证据支持的边界写入可执行条件，并在成功确认后提交。S01 原始观测的 `b_bleed` 最大值只有 1.1，单靠此任务也没有形成 1.6 边界的经验识别证据，不应只是把常量移入 preconditions 就称为学得作用域。

## 四、确定边界 O2：跨域 twin 和多线索是 harness 注入，尚非两域证据驱动的虫洞发现

### 4.1 新节点中什么是学得，什么是预先给定

数学 operator spec 与拟合参数来自 S02/S09 的真实发现产物。其余关键结构由 `wb_meta` 或 helper 直接提供：[1]

| 内容 | 实际来源 |
|---|---|
| 主/twin 节点名、两棵树名、目标域标签 | 任务构造器硬编码 |
| 主/twin cues 如何切分 | `primary_cues` / `twin_cues` 硬编码 |
| 两节点的参数、模型源码、instance digest | 同一个 `act_dict` 深拷贝，绑定同一个 artifact 路径 |
| twin 的 empirical status | 无第二域 episode，直接设为 `SUPPORTED_IN_SCOPE` |
| twin 权限 | 直接设为 `local_solver_guidance` |
| preconditions | 只有 `active_arity == 1/2` |
| wormhole 映射 | 固定 `identity_canonical` |
| hop error | 固定 `0.01`，不是两域各自误差估计 |
| source episode refs | 此调用不传，默认为空 |

这可以作为**一个已确认校准实例的两个人工别名**。它没有提供“第二领域已独立验证同一机制”的实证基础。AST 身份相同在这里是复制带来的恒真事实；其自身不能使不同物理领域共享同一实例参数。

helper 给 twin 不加源实例/应用域条件，并赋较高权限，意味着若将 twin 当作另一真实领域的历史实例供 R3 分发，证据范围会比现有来源更宽。当前 12 次迁移走了目标重新拟合与新确认，保护了提交中的预测结果；当前流尾只在两个 **primary** 树分别检查一次原参数的 zero-fit，并没有检查 twin 的异域实例授权。[1][6]

### 4.2 多线索对照是有意设计的结构压力测试

正式流开始前，构造器把 W6 两个表面标签一并加入旧 `stage1_W3_murray_capped`，把 W7 两标签及 arity cue 加入 `vascular_2d_hub`，再重绑定原边。后续任务的 domain cue 与新 twins 的 tags 也是同一段测试代码给定；多线索解析是将 `domain_cue` 按 `|` 拆开。这不是从新 paid observations 学出领域语义或歧义词义。[1][7]

人为构造 distractor 并非错误，且四臂接受相同初始森林。应把结论写成“预设 split-cue/hub 图上的 pooling 排序压力测试”，并公开该生成方式。实际 12 个下游成功全部 `non_direct_walk_win=False`，所以不能仅凭存在跨树边便称这些实例必须经过 wormhole 才召回。

### 4.3 已有历史实现可复用，不需要重造发现器

仓库 `open_workspace_wormhole.py` 的历史 M2 路径已经有更严谨的形成过程：每个域分别运行真实 source episode，检查 strict pass 和 pure AST，按 digest 分组；至少两项独立域结果才注册横向边，使用两域确认/评估误差的最大值估计 hop error，并保留两个 `source_episode_refs`。[8]

因此这里的差距是**新在线 helper 缺少历史已实现的证据接线**。最小改法是先将 source 的数学程序作为可试验候选入库；没有第二域证据时把另一域入口标成别名或 `bounded_trial`，待现有 paid downstream confirmation 形成真正的第二域实例时，再复用历史形成路径登记跨域边与其条件。不要复制参数后直接把第二域称为受支持实例。

### 4.4 按约定完成的三次 zero-fit 补证：使用本轮 pytest 产物，非提交归档

主审运行本版官方测试时，最后的 stateful 测试产生了真实 S02 W6 artifact。本专项没有再运行发现、拟合或购测，只调用现有 helper 在隔离森林重建与正式调用相同的主/twin pair，并用真实 `ForestScopeDispatcher` 预测三次。来源文件、参数文件、manifest 三项已最小保留到 `twin_scope_bundle/`，全部与 pytest 原文件字节相同；完整原路径和 SHA256 见 `pytest_w6_artifact_provenance.json`。

**这些文件是本轮审计从新 pytest 运行中另存的证据，原提交官方 evidence 中没有它们。审计者补存不能弥补 runner 未归档 Part3 的事实。** 产物 `authored_method.json` 的 SHA256 为 `389caa76b405c023d430d5004437ef7b8301878a8ccd93d54da9970d10a3a121`，源 task 是 `M44_S02_Discover_W6_Haldane1D`，源域为 `cue_tubule_saturation_a`，包含 7 条训练前缀与 1 条新确认。

取正式 S04 heldout 坐标 `x_load=3.2`，结果为：

| 调用含义 | 预测 | 对应真值 | 绝对误差 | 实际分发 |
|---|---:|---:|---:|---|
| 主节点，原 S02 物理实例 | 6.98458573 | 6.98458574 | 约 1e-8 | `dispatched_ok` / 容差内 |
| twin 作为同一个 S02 实例的别名 | 6.98458573 | 6.98458574 | 约 1e-8 | `dispatched_ok` / 容差内 |
| twin 被当作正式 S04 的另一参数实例 | 6.98458573 | 6.55869399 | 0.42589175 | 仍 `dispatched_ok`，`within_tolerance=False` |

第三行使用的参数正是官方固定 RNG 序列中 S04 的 `vm=14.206, km=1.72, ki=0.1964`；容差为 0.045。三个请求均被授予 `local_solver_guidance`，且全部 `fit_called=False`。两个节点的 action 完全相同，twin precondition 确为仅 arity=1，wormhole 的来源引用为空。

**限定：第三行不是官方 S04 流程失败，也不是“保存参数应自动适应新参数”的要求。官方 S04 会重新拟合和新确认，原回执成功成立。** 此控制仅实证两个已有解释的分界：同源别名的冻结预测完全正常；把该别名升级解释成“另一个真实领域的已确认实例”则没有依据，现有 dispatcher 也没有相应实例上下文去拒绝这种解释。应修正 twin 的声明范围/证据等级，或明确 alias-only；不可把这一数值插入官方 80 条回执重算 false-confirmation rate。

控制输出为 `twin_scope_control.json`，脚本为 `probe_twin_scope.py`。本专项至此冻结，没有扩展更多目标或新机制试验。

## 五、确定缺口 O3：在线产物没有成为可恢复交付物

runner 的 Part3 在 `TemporaryDirectory` 内形成 live bundle。归档逻辑只拷贝 Part2 的 Stage0/Stage1/Stage2 authored workspaces 和三张历史森林；Part3 函数只返回 summary/receipts，不返回活森林。随后临时目录退出删除，再写 Part3 两个统计文件。[9]

提交 evidence 中有 130 个文件（含 SHA256SUMS），但：

- `online_writebacks/` 下 W6/W7 artifact 数为 0；
- post-stream forest snapshot 数为 0；
- 三张 persisted forest 只有 8、15、15 个节点，全部没有任何 `online_` 新节点。

所以当前成功是**进程内有状态写回**；Part2 的冷载验证不能覆盖新 W6/W7 的重启恢复。重跑研究可以重新发现它们，但这不是恢复已经学到的知识。见 `online_receipts_analysis.json` 的 archive 节；主审另以完整官方 replay 输出交叉核实该归档边界。

**最小处理：** 复用本版已修好的相对路径 artifact + 自验证 snapshot 接口，保存 post-stream forest、W6/W7 bundle、真实来源与写回事件；删除原临时目录后，在新进程从这份状态继续消费一条后继任务。不要用重新运行完整研究替代恢复测试。

## 六、自动化与 R3 声明应按实际范围表述

### 6.1 写回由预登记阶段位置开启

外层条件是 `enable_writeback and wb_meta is not None`。只有 S01/S02/S09 的 meta 非空；下游 12 个目标和 5 个历史目标均为 `None`。因此即使日后在其中发现新结构，普通评估结果也不会触发这个 helper；每个后继目标的新参数实例同样没有加入 live forest。[1]

这是预设两次学习事件的持续序列原型。可作为 M5 桥接，但“任意在线流都以确认结果自动驱动 consolidation”尚未完成。下一步接已有 confirmed artifact 的统一出口，而非加入更多按任务名称写死的事件分支。

### 6.2 R3 检验真实但有限，summary 缺少关键质量字段

历史 `run_interleaved_zero_fit_dispatcher_stream` 含十个查询：九个数值请求和一个 optical 拒绝请求。覆盖的不同 historical node 为六个：Stage0 Monod、W1、W2、W3、Pi coarse、Pi fine。W4 和 W5 不在其中；尤其本次被 S01 改动的 W4 没有流尾保持检查。[6]

该函数返回：

- `all_routes_exact`：仅逐条比较 selected node 与 expected node；
- `authorized_pass_rate`：另行统计九个数值请求是否落入容差；
- `max_authorized_abs_error`；
- `unauthorized_cleanly_blocked`；
- 完整十条 receipts。

但 Part3 只保存前者和 `any_fit_called`，未断言、保存后面三个质量指标及原始 R3 行。因此报告不能仅从保存的 `all_routes_exact=True` 推出“10/10 预测精确”或“所有历史能力零遗忘”。这不是断言当前九个数值点实际失败；问题是保存的验收量比文字声明窄。

W6/W7 的流尾字段同时检查了 `dispatched_ok` 和 `within_tolerance`，其数值判定比历史汇总字段完整。但它们各只有一个 primary 树、原 S02/S09 参数的测试点，未覆盖 twin 或 12 个后继校准实例。最小处理是保存现成 R3 返回值并断言其数值/拒绝字段，增加实际改动 W4 的边界回归；不需要为此再开大规模 benchmark。

## 七、建议的优先顺序

1. **先交付可恢复的在线状态。** 不然在线学习收益只能在单次进程内兑现，后继会话无法复用。
2. **修正 S01 的真实语义与提交条件。** 把“协议更新”与“由经验形成的可执行范围修订”明确分开，失败不得作为成功修订证据。
3. **分清 source instance、共享数学方法、第二域实证。** 别名不冒充独立域，已有原生 target 确认可作为新域实例的证据，已有 M2 两源形成器可复用。
4. **把结果驱动写回接到统一出口，并保留现有测试作为回归。** 当前两事件流程已证明必要部件可协同工作，无须重写求解器。
5. **调整报告的归因与保持性字段。** 结构写回收益、PPR 候选排序收益、pooling 三 cue 榜单、zero-fit 原实例保持分别报告，保留现有 positive result 与限定条件。

这一裁决既不要求撤掉 PPR/等价池化，也不否定 M4.4 对 M4.3 的进步。核心工作是把已有进程内闭环变成证据范围清楚、状态可恢复、写回条件由结果驱动的闭环。

## 八、复现与阅读范围

本目录：

- `analyze_online_receipts.py` / `online_receipts_analysis.json`：全部 80 条回执、逐阶段成功与召回、PoolOn/Off equality、归档 inventory。
- `probe_s01_failure.py` / `s01_failed_control.json`：官方 Part3 第一事件失败分支，S02 前停止。
- `probe_s01_scope.py` / `s01_scope_control.json`：修订前后真实授权与 frozen instance 分发。
- `probe_twin_scope.py` / `twin_scope_control.json` / `pytest_w6_artifact_provenance.json`：复用本轮官方 pytest S02 产物的三次 zero-fit 有界控制；`twin_scope_bundle/` 为字节等同的最小来源包，明确不是原提交 evidence。
- `historical_route.json`：按 AGENTS 要求执行的 catalogue 查询；该长关键词组合没有命中，随后沿当前源码与历史审计定位了已有 M2/组合搜索/状态接口。
- `READING_MANIFEST.json`：实际阅读的文件与范围、来源 SHA256。

在该提交的 `Hegel Machine` 目录：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 /workspace/scratch/3f14ad2da0ed/audit_m44_b965f022b/online/analyze_online_receipts.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 /workspace/scratch/3f14ad2da0ed/audit_m44_b965f022b/online/probe_s01_failure.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 /workspace/scratch/3f14ad2da0ed/audit_m44_b965f022b/online/probe_s01_scope.py
```

这些脚本支持 `HEGEL_AUDIT_REPO` 与 `HEGEL_AUDIT_OUT` 指定不同检出和输出目录（twin 脚本从当前 `PYTHONPATH` 加载代码，另外要求 `HEGEL_ONLINE_W6_ARTIFACT` 指定真实 S02 artifact）。S01 scope 控制须先运行 failure 控制生成 before/after 状态。twin 控制可将此环境变量指向本目录保留的 `twin_scope_bundle/persisted_authored_workspaces/online_writebacks/online_W6_michaelis_cap_1d/authored_method.json` 复现，无须重新发现。本专项没有重跑全研究，也没有将未阅读的全部历史版本或参考文献声称为已经通读。

## 源码依据

[1] `src/hegel_machine/cognition/m4_4_instance_role_closure_and_online_writeback.py`：703–811（写回/twin），814–1050（任务、cue、hub 与写回元数据），1082–1164（真实评估后事件提交），1167–1256（R3 与汇总）。

[2] `src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py`：1096–1274（评估/已付pilot），1275–1519（节点验证、重新拟合、fresh confirmation），1715–1777（scope guard、heldout、strict pass）。

[3] 同 [2]：1598–1673（原生 CodeAgent fallback 与参数），及 `m4_2_unified_closure_and_pi_lifecycle.py`：2576–2590（原子基元）。

[4] `src/hegel_machine/cognition/m4_compositional_discovery.py`：255–377（组合子），570–710（已付数据诊断），711–951（Provider），974–1224（原生 episode、新确认、参数快照）。新版 diff 的行为新增在 799–812。

[5] `src/hegel_machine/cognition/wormhole_forest.py`：219–496（真实授权器）；`m4_2_unified_closure_and_pi_lifecycle.py`：480–718（zero-fit dispatcher）。

[6] `src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py`：2113–2253（十查询 R3）；M4.4 模块 1167–1240（保留字段与新 primary 两个单点）。

[7] `src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py`：1197–1273（domain cue 与 empirical pilot cues）；1410–1445（artifact 验证和 spec 加载）。

[8] `src/hegel_machine/cognition/open_workspace_wormhole.py`：1730–1918（独立 paid domain episodes、AST 分组、来源/误差绑定的横向边形成）。

[9] `scripts/run_v17_6_m4_4_closure_and_writeback_study.py`：85–145（临时工作、只归档 Part2），147–173（历史快照冷载），191–195（Part3 只写 summary/receipt）。
