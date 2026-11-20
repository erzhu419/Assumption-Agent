# Assumption Agent V17.6 M4.4 审计与后续建议

审计对象为提交 [`b965f022b214f47a1341559f20041da6a224d66d`](https://github.com/erzhu419/Assumption-Agent/commit/b965f022b214f47a1341559f20041da6a224d66d)，比较基线为 M4.3 的 `3461535364fcdb4e9ae9c87ccba502b7e6b56fe8`。日期：2026年11月20日。上传工作总结与提交内同名文件逐字节一致。

**总体判断：M4.4 有实质进步，可以作为继续建设 M5 的基础。上一轮两个主要 P1 的原始反例已经闭合；Part3 也确实实现了进程内在线结构写回及后续消费。当前需要收敛的是几处证据关系、调用接口和持续状态交付问题。报告中的“所有入口 100% 闭合”“在线经验作用域修订”“跨域新实例已获支持”等较强表述，超过了本次实现和实验已经支持的范围。**

当前最有分量的正结果是：真实组合搜索得到 W6/W7，写入同一活森林后，12 个后继目标复用结构、重新拟合并独立确认；三条 live 臂均成功。相对禁止写回的 Frozen 臂，两条 PPR live 臂的20步代理总成本各减少 `85.0283`，Compiled live 减少 `84.5733`。PPR 相对同样可以写回的 Compiled 臂减少 `0.4550`，其原因是少评九个候选抵消了更高检索费。Pooling 开关的端到端结果完全相同。这三种结论应分别保留。[Part3 完整回执](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/docs/evidence/v17_6_m4_4_instance_role_closure_and_online_writeback_20261120/part3_m4_4_online_writeback_receipts.jsonl)

## 一 本轮覆盖与独立复现

### 1.1 覆盖的具体对象

本提交相对 M4.3 共改变 854 个路径：10 个源码、runner 或测试文件，130 个新证据文件，711 个导入的上一轮审计文件，以及旧审计 ZIP、索引和工作总结三个文档项。逐项审阅了十个代码变更及其实际消费者，完整读取新模块、runner、新测试和本轮工作总结；对历史部分复用提交中保留的审计、反例与能力登记，再沿当前调用链复核。历史 181 版本的登记用于定位已有能力，不代表本轮重新执行了 181 个旧版本。

十个变更代码文件包括 `wormhole_forest.py`、`hipporag_wormhole_index.py`、M4.1/M4.2/M4.3 三个统一链模块、`m4_compositional_discovery.py`、`m4_multiscale_renormalization.py`，以及新的 M4.4 模块、runner 和测试。三组专项分别检查身份与角色消费者、在线写回与证据来源、索引与成本及归档。完整阅读范围见各专项 reading manifest；精确变化清单见 `change_scope.json`。

### 1.2 重放结果

| 核验对象 | 独立结果 |
|---|---|
| V16/V17 相关测试 | 11个文件，42项全部通过，进程退出码0 |
| 官方完整 runner | 三部分全部完成，退出码0，本环境1299.02秒 |
| Part1 闭合控制 JSON | 与提交逐字节相同 |
| Part2 主基准 | 1144条唯一回执；全部费用、状态、误差、候选数一致；42处 selected model SHA 不同 |
| 噪声 fork | 192条唯一回执；全部费用、状态、误差、候选数一致；10处 selected model SHA 不同 |
| Part3 在线流 | 80条唯一回执；费用、通过、误差、候选数一致；18处 selected model SHA 及2处同源别名选择不同 |
| SHA 清单 | 提交包与重放包各129/129通过；两包均130文件，其中93个逐字节相同 |
| 全部1416条科学回执账本 | phase费用及观测数加和、C公式校验无不一致 |

本次是指标与所列行为的重现，并非整个目录逐字节一致。两处额外路由差异都在 S03 的 PPR PoolOn/Off：提交选择 W6 primary，本轮选择同一 action/instance 的 W6 twin，费用和数值结果未变。其精确排序差异来源未进一步隔离，原始差异完整保存。形成产物及汇总中有60处非计时数值字段微差，最大绝对值约 `4.0e-13`，并伴随实例/参数/语义摘要变化。Part2 的主表、维护分解、回本、fork、DEV和冷载指标完全一致；Part3 summary仅六个写回实例摘要不同，其余完全一致。不能为追求跨运行同 SHA 而舍入真实参数或放松实例身份校验。

1299.02秒与提交记录655.61秒来自不同执行环境，本次还与相关测试并行运行；这两个总墙钟不用于算法性能归因。pytest 保留一项 anyio 插件弃用警告，不影响42项通过。官方重放仍未输出任何在线W6/W7产物或post-stream森林，确认第三节的持久化缺口。

完整重放使用官方 `--out-dir` 将结果写到独立目录；源码、已提交实验结果和历史审计均保持不变。环境为 Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0、pytest 9.2.0，固定 `PYTHONHASHSEED=0`。全部科学拟合与控制为本地执行，在线模型 API 调用为零。

执行记录与核验脚本见 `replay/pytest_related.log`、`replay/study_run.log`、`verify_replay.py`、`replay_verification.json`、`reviewed_replay_differences.json`。费用重算读取逐条 receipt 的 phase ledger；没有用已经四舍五入的均值反推总量。`source_integrity.json`确认工作树clean、提交内129项文件摘要仍正确。复现命令见 `README_REPRODUCE.md`。

## 二 上一轮问题实际关闭到了哪里

| 上一轮问题 | 本轮核实 | 判定 |
|---|---|---|
| R3 节点只绑定 AST，合法同 AST 的另一参数实例可替换 | 全部 14 个持产物节点和 2 个 OR 分支共 16 个动作，正确实例锚接受、错误锚拒绝；R3 缺锚拒绝；官方真实增益 6 与约 12 交换控制通过 | 原 P1 已修 |
| 规范坐标盒检查与拟合各选一套角色排列 | 同名规范轴只允许恒等；合法映射限制 `allowed_role_orders`，拟合结果再锁定 fresh 盒交集、购买和最终查询 | 原 P1 已修 |
| fresh helper 可在已验证盒外购测 | coarse acquisition 前有守卫，确认点生成使用目标坐标下的盒交集；原重复 pilot 加越盒确认的反例通过 | 已修 |
| 细尺度特征矩阵和冗余预拟合 | `A12=g_sf` 接回真实方程；原 T23 heldout RMSE 从约 0.005842 到回执精度下的 0；丢弃的 generic prefit 为零；本地 RK4 与 integrity check 入账 | 已修 |
| 跨实例证书重验证和失败前写状态 | 重验证拒绝真实错误证书，整个森林状态不变；合法 action、scale、certificate 同步替换成功 | 重验证入口已修，新建入口仍有下文缺口 |
| 迁移后 semantic/certificate digest 不自洽 | 三份快照磁盘和冷载对象均为零 mismatch；五条历史节点摘要正确；历史与 OR artifact 定位符可解析 | Part2 已修 |
| legacy 严格 flags 被兼容分支绕过 | strict 模式拒绝无 bundle 产物且零 refit；宽松模式明确标记 `legacy_isolated_refit` | 已修 |
| 旧 ScaleBlind 升级丢弃已付细确认点 | 原入口按通道保留并合并，原 C0_T07 不再重复购买相同细确认点 | 已修 |
| 旧索引对象在正式 revision 后仍返回 v1 | PPR、Flat、Compiled 在 node revision、wormhole refinement、负效用归档后均刷新收费；无变化再查询不重复收费 | 已修 |
| Gaussian 名称、单种子 fork 和 DEV 不对齐 | 明确未截断高斯；四种子、两噪声、三臂共 192 条 fork；DEV 包含对应噪声及 adiabatic/quench 各一个 | 已修 |

这些结果足以改变上一轮的判定。特别是实例身份和角色锁已进入生产调用方，不能继续称作“只新增了字段”。当前普通入口的剩余问题应按下文单独定位。[实例与角色接线](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L562-L678)；[角色受限拟合与确认](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py#L1420-L1478)

## 三 应修的具体接口与交付缺口

这里的 P2 指已经实证、范围清楚的完整性或集成问题；不把它们换算为原主表已经发生的失败次数。

### 3.1 新建粗粒化边仍可把另一实例的证书接到当前节点

`revalidate_vertical_edges_with_certificate` 已在 mutation 前比较节点、OR 分支和证书实例；`add_coarse_graining_edge` 尚未共享这个校验。新建入口取 `node_digest or certificate_digest` 写入边，随后直接 append。`fine_scale_children_of` 只检查 endpoint version 和 stale 状态，实际粗通道消费者也没有补充证书身份比较。[新建与消费入口](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L904-L960)；[已修的重验证入口](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L1284-L1308)

独立控制只使用提交内真实产物：Stage0 Pi 证书的实例为 `416e0bd1…`，Stage2 Pi 节点为 `058680e5…`。移除独立森林副本的原 Pi 粗边，再用公开新建接口加入旧证书，调用成功、边 live，node/edge 与内嵌 certificate 分属两个实例。

| 原始 T21 控制 | 实际通道 | 测量费 M | heldout RMSE | 结论 |
|---|---|---:|---:|---|
| 没有 Pi 粗边 | micro_fine | 70 | 0 | 通过 |
| 加入跨实例证书粗边 | coarse_macro | 21 | 0.008633 | 仍通过 |

这证明错误证据绑定能够实际开通廉价粗通道。本例两个已确认模型接近，数值仍在容差内，所以不能把它描述为新的假确认。正常提交的两条粗边身份一致，主表无需因此全部作废。

同一个错误绑定森林交给 `verify_forest_snapshot_digests`，仍得到所有 mismatch 为零。原因是该 verifier 检查各对象自身摘要，尚未检验 node、edge、certificate 的关系一致性。

**修复建议：** 把现有无副作用的实例校验抽成共享函数，新建、重验证和归档关系检查共同消费。失败必须发生在边列表、版本、epoch 修改之前。对旧的无实例 fixture 明确兼容层级，保持历史接口可用。复用本次 S0/S2 控制就能验收，无需再设计一套证书。证据见 `index_archive/certificate_controls.json`。

### 3.2 S01 只更新未消费的协议字段，并未实现所称的作用域修订

S01 的构造注释称要修订 `b<=1.6`，`wb_meta` 声明 `b_bleed_ceiling_safe`；执行时这项元数据没有被读取，preconditions 原样，实际只加入 `protocol_spec.online_bleed_verified_max=1.6`。整个 src/tests 中没有该字段的授权消费者。此分支也没有像 W6/W7 分支一样检查 `rec["confirmed_ok"]`。[S01 提交路径](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/m4_4_instance_role_closure_and_online_writeback.py#L1099-L1127)

两个有限控制已确认：

- 用官方在线函数，仅令 S01 回执 `confirmed_ok=False`，到 S02 前停止，W4 仍从 v1 升至 v2。没有重新发现或新增观测。
- 对提交内真实 W4，使用真实授权器与 R3 检查修订前后：`s=4, b=1.5` 均授权；超过所称边界的 `b=1.7` 也均授权，预测同为 `7.65257151`、零 fit。

因此 S01 确实制造了索引失效事件，却没有改变可执行的作用域。Part1 直接写入真实 precondition 的索引测试仍成立，两者应分开陈述。S01 所用 pilot 的 b 最大仅 1.1，heldout 最大 1.0；当前数据也没有识别 1.6 为真实物理边界。

**修复建议：** 若只要测试动态刷新，准确命名为“协议元数据变更事件”即可。若目标是经验范围修订，则先获得支持边界的实际证据，再把条件通过角色映射接入已有 predicate/box gate，并在合格结果后原子提交。仅把常量 1.6 搬进 preconditions，仍不足以证明边界是学得的。证据见 `online/s01_failed_control.json`、`online/s01_scope_control.json`。

### 3.3 Part3 新知识随临时目录删除，当前交付无法冷恢复 W6 和 W7

官方 runner 把 Part3 放在 `TemporaryDirectory` 中运行。归档时只拷贝 Part2 的 Stage0/Stage1/Stage2 workspaces 和三张快照；在线函数只返回 summary/receipts，没有返回或保存最终森林。退出临时目录后才写入 Part3 的两个统计文件。[runner 归档范围](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/scripts/run_v17_6_m4_4_closure_and_writeback_study.py#L85-L173)

提交的 130 个文件中，W6/W7 `online_writebacks` artifact 数为零，post-stream forest snapshot 数为零；三张快照仍是 8、15、15 个节点，没有新 online 节点。进程内的后继任务确实用过它们；进程结束后，使用交付证据包只能恢复在线流之前的知识。完整重跑可以重新发现，不能替代恢复已经付费形成的历史。

**修复建议：** 复用本版已经修好的相对定位符与 serializer，额外保存 post-stream forest、W6/W7 bundle、真实来源和写回事件。删除原临时目录后，在全新进程中冷载这一状态并继续一个后继目标，确认没有再次运行源发现或重新支付已付观测。持续学习恢复合同还应明确原始 episode、questions、游标和账本由哪个现有 checkpoint 层保存；不要求本轮重建存储系统。

### 3.4 OR 选择出的动作尚未贯穿重新拟合召回

R3 已改成先授权、取 `selected_action`、再载入对应实例。M4.3 的重新拟合召回在 gate 之前先按顶层 action 调用 `_load_node_verified_operator_spec`，得到 gate 的 `selected_action` 后也没有改用它。M4.1/M4.2 旧消费者仍有同一模式。[M4.3 召回次序](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py#L1276-L1319)

有效控制采用普通 Monod OR，不涉及 Pi 或尺度证书：A 是当前 Stage0 增益 6 的实例；B 是本提交旧审计中真实确认的增益约 12 实例；由正式 `register_node` 创建同 AST、不同实例的两条互斥条件分支。目标适用 B。

- A/B 都完整时，R3 加载 B，输出 `5.33333333`；M4.3 的 gate 选择 B，loader 实际加载 A。由于 AST 相同、允许重新拟合，M4.3 仍能正确完成，heldout RMSE 为零。
- 仅把未匹配的 A 定位符设为不可用，B 完整：R3 仍正常；M4.3 在调用 gate 前就跳过节点。禁用 fallback 的隔离控制返回 `scoped_unknown`，暴露了适用分支的可用性被无关动作阻断。

这是分支独立性和动作来源不一致，不是“载 A 后重新拟合必然错误”。当前标准 Pi 分支绑定同一实例，原主表没有这个差异。早期探索的混合 Pi 两实例 OR 不满足当前尺度证书约束，已在原始记录标记排除，不据其 21→88 费用变化提出结论。

**修复建议：** 先形成请求对应的有效动作视图，再让实例、roles、spec、路径和执行计划消费同一视图。保留“B 完整而 A 不可用”普通 OR 正例即可验收。证据见 `identity/generic_or_actions.json`。

### 3.5 R3 显式 role_map 的信任合同需要统一

R3 接受调用方显式 role_map，并按它检查盒、执行预测；它没有共用新 helper 的“规范轴同名时必须恒等”规则。保持原森林、物理点 `u_drive=2.7,T_horizon=0.47` 和 oracle 不变：规范名字默认调用与匿名名字正确映射都转细节点，预测 `3.22880775`；规范名字加显式互换映射则走粗节点，预测 `0.96967258`，误差 `2.25913517`，回执为 `dispatched_ok` 且 `within_tolerance=False`。[R3 盒检查入口](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L498-L563)

该例依赖调用方主动传入矛盾映射，属于 API 合同边界；当前主表不这样调用。盒检查与执行使用的是同一个映射，也不是上一轮两个阶段分别选排列的原 P1 再现。

**修复建议：** 若规范轴名字具有系统承诺的语义，就在公共角色规范化入口拒绝冲突映射，保留匿名坐标合法映射。若显式参数被设计为可信覆盖，则必须缩小“规范轴永不互换”的文档断言，明确覆盖权和调用方责任。本项目当前工作总结的口径更适合前一种处理。证据见 `identity/role_consumer_boundaries.json`。

## 四 Part2 主基准的结论与条件

### 4.1 结果保留

每臂为 96 个域内目标及 8 个域外目标，11 臂共 1,144 条。下表的生命周期总成本包含各臂按现行协议承担的形成、维护或 DEV/编译费用；失败更多的臂不能仅凭费用较低判定更优。所有臂域外 8 个目标都返回 scoped unknown，确认与训练前缀最大重叠为零。[Part2 汇总](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/docs/evidence/v17_6_m4_4_instance_role_closure_and_online_writeback_20261120/part2_m4_4_continual_benchmark_summary.json)

| 臂 | 域内通过 | 假确认 | 每题 C | 含前期的 96 题总 C |
|---|---:|---:|---:|---:|
| Hegel Unified | 96/96 | 0 | 32.4651 | 3768.9262 |
| Walk NoPool | 96/96 | 0 | 32.4651 | 3768.9262 |
| Compiled SameForest | 96/96 | 0 | 32.3946 | 3763.9730 |
| Hegel NoMaintenance | 96/96 | 0 | 37.6469 | 4047.9281 |
| AllHistory BestPilotRMSE | 96/96 | 0 | 41.5625 | 4423.8250 |
| Alpha0 PoolOff | 95/96 | 0 | 39.5510 | 4449.1725 |
| Alpha0 PoolOn | 95/96 | 0 | 39.5552 | 4449.5727 |
| Flat BM25 | 95/96 | 0 | 39.5694 | 4450.9390 |
| Frozen Stage0 OpenRediscovery | 95/96 | 1 | 44.5023 | 4516.2720 |
| Stratified DEV FirstAcceptable | 96/96 | 0 | 38.1667 | 4565.8250 |
| Frozen Stage0 NoDiscovery | 16/96 | 0 | 35.8351 | 3684.2220 |

Hegel 与 Compiled 都有 87 次 recall、31 次非直接图召回成功，九次由全历史候选池回退完成，实际回执均为 `fallback:op_composed_haldane_shunt_2d`；Part2 的 Compiled 全周期仍便宜 `4.9532`。这不要求撤掉 PPR，而是继续保留同信息强对照。此处相对 M4.3 的费用变化包含新校验/RK4 计费，不能直接当成算法退化。

Frozen Open 从上轮 93/96 提升到 95/96。逐目标变化只在两个 MurrayCapped 目标，分别为 seed 20301120 的 T12 和 20311120 的 T09；新增一维未解决残差的通用组合尝试使它们得到 capped Michaelis 近似并过本题容差。不是删除了困难目标。仍保留 seed 20321120、T11 的一次 false confirmation，heldout RMSE `0.049667`。见 `replay/score_changes_vs_m43.json` 和 `replay/part2_receipt_recalculation.json`。

### 4.2 维护收益有明确任务分布条件

维护相对 NoMaintenance 的后续毛节省为 `497.4519`，扣掉 Stage2 费用 `218.45`，净收益 `279.0019`。当前固定顺序第 46 个目标回本；相对 Frozen Open，第 41 个目标回本，96 题净省 `747.3458`。

其中 Pi 贡献 `491.3840`，占毛节省 `98.7802%`：八个 hydraulic adiabatic 目标贡献 `491.2920`，八个 thermal quench 贡献 `0.0920`；其余 80 个目标合计只省 `6.0679`。这一分解与工作总结一致。报告已主动把收益限定到当前低成本粗通道使用频率，值得保留，不再把同一个限制重复升级成新缺陷。

主要机制是 hydraulic 子域获得合法的廉价粗通道。将整个维护收益进一步归给某一项 Minton 归档或 fidelity compression，需要相应单因子对照；当前联合维护结果应按联合处理解释。

回本目标数还依赖高收益任务出现顺序；它是本条序列的观测结果。后续应把任务混合比例、归档/维护政策与费用敏感性一起报告，避免把“第46题回本”变成任意任务流的固定常数。

### 4.3 噪声与 DEV 对齐的修复成立

四种子各八个 onset 目标、三臂、两个噪声条件，共 192 条 fork。两条件三臂全部 32/32 通过，假确认为零，重叠为零：

| 条件 | Hegel 平均 M / C | BestPilotRMSE 平均 M / C | DEV 平均 M / C |
|---|---:|---:|---:|
| 无噪声 | 28 / 29.2435 | 28 / 32.9250 | 32 / 33.0250 |
| 未截断 Gaussian sigma 0.002 | 29.25 / 31.2529 | 30 / 37.0250 | 32 / 33.0250 |

噪声下 Hegel 相对 BestPilotRMSE 每题少 `0.75 M` 和 `5.7721 C`，同时 Hegel 自己的成本也比无噪声上升。DEV 已加入相同 pilot 噪声，且 Pi 由两个 adiabatic 改为一个 adiabatic 加一个 quench，12 题 DEV 费为 `468.0`。这些是有限四种子受控证据；仍是一条形成历史上的参数目标变体，不能视作四条独立形成历史。[完整 fork 回执](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/docs/evidence/v17_6_m4_4_instance_role_closure_and_online_writeback_20261120/part2_m4_4_fork_noise_receipts.jsonl)

## 五 Part3 在线学习实际证明了什么

### 5.1 新结构形成与后继消费是真实的

S02 的 W6 通过 `atomic_michaelis_1d__comb_quad_inh` 形成一维 Haldane 抑制结构；S09 的 W7 通过 `atomic_exp_relax_2d__comb_shunt` 形成应力松弛所需结构。调用链使用当前已付 prefix、Stage0 原子基元、已有组合子、实际 Python 拟合与隔离新确认，未将 W6/W7 的预制复合真值直接放入候选池。此次新增 `round2_1d_unresolved` 是通用残差触发，所有臂共用。[组合搜索变化](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/m4_compositional_discovery.py#L790-L818)

后继 12 题的参数变化时，三条 live 臂取出新节点绑定的 spec，用当前目标 pilot 重新拟合，购买 disjoint fresh confirmation，再测 heldout。因而这是结构知识复用；流尾 W6/W7 各一次原始参数 zero-fit 检查是另一种能力。20 步全部经过实际目标预测评估，S01/S02/S09 的事件没有冒充预测成功。

| 20 步指标 | PPR PoolOn | PPR PoolOff | Compiled Live | Frozen Open |
|---|---:|---:|---:|---:|
| 通过 / 假确认 | 20/20 / 0 | 20/20 / 0 | 20/20 / 0 | 20/20 / 0 |
| 后继 12 题召回复用 | 12 | 12 | 12 | 0 |
| 自动刷新次数 | 3 | 3 | 3 | 0 |
| 评估候选总数 | 42 | 42 | 51 | 156 |
| 总测量费 M | 603 | 603 | 603 | 651 |
| 总计算费 K | 495.58 | 495.58 | 504.67 | 1236.12 |
| 逐行汇总总 C | 627.7785 | 627.7785 | 628.2335 | 712.8068 |

三条 live 臂都比 Frozen 少 48 个测量费用单位，并节省反复搜索开销；两条 PPR 臂各净省 `85.0283`，Compiled 净省 `84.5733`。Compiled 也得到 12 次成功，说明这些成功首先证明保存新结构的价值；图索引相对其它能保存同样结构的记忆系统的收益，要用 live 强对照继续隔离。

### 5.2 PPR 的 0.4550 是小而明确的候选排序收益

PPR 少评九个候选，节约 `54 K`；查询及通道预检相对 Compiled 多 `44.91 K`，净节约 `9.09 K`。九次候选差全部发生在 W7 双 cue 的 S12–S15，每题少两次，以及历史 Murray 的 S18 少一次。其余 15 步 Compiled 的费用更低。

按未舍入的 K 差计算为 `0.4545 C`，逐行四位小数再求和为报告的 `0.4550`，这是正常舍入。相对 Compiled 总成本约 `0.0724%`，应准确称为“在当前计价及任务流下的小幅优势”。全局把 K 统一乘一个权重不会改变这两个 M 相同臂的符号；改变候选拟合与检索的相对单价才会改变归因。固定本次执行序列，按未计初建的流内账本，令候选单价乘 r，净节约为 `54r-44.91 K`；临界 r 为 `0.8316667`。即候选单价从6降到约4.99 K、下降16.8%时，优势消失。这只是显式计价敏感性，不是观察到排名翻转。

流开始前的索引初建没有入账。按当前官方初始化重算，PPR 初建为 `19.7 K`，Compiled 为 `20.4 K`；补计后总成本分别 `628.7635` 与 `629.2535`，PPR 仍省 `0.4900`。Frozen 同样补初始 PPR，Live 对 Frozen 差额不变。另有 writeback 固定 `4.6 K` 与注释“两节点各1.5、一边0.8”不符，当前两次保守多收合计 `0.08 C`；所有 live 臂同额，无排序影响。见 `index_archive/part3_cost_analysis.json`。

### 5.3 Pooling 排名控制与端到端收益应分开

流尾单独查询三条 cue 时，PoolOn 的 Top1 为新 W6 twin，PoolOff/Compiled 为旧 Murray 节点。这证明特定图上 pooling 能改变排名。完整 20 步中，PoolOn/Off 的候选数和 cost 逐条相等，净差为零；本次 `0.4550` 也同时由 PoolOff 达到，不能归给 pooling。

工作总结已经承认实际 pilot 触发的额外次线性 cue 使端到端打平。这条限定是正确的，应保留。下一步最小实验是把这种会改变实际候选试验顺序的查询放入真实目标任务，并从检索到确认逐动作计费；仅扩大图或重复 Top1 查询不能替代这一步。[流尾独立查询](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/m4_4_instance_role_closure_and_online_writeback.py#L1203-L1210)

### 5.4 Twin 是人工别名，尚非第二领域独立实证

写回 helper 将同一个 `act_dict`、同一组参数、同一 instance digest 和同一 artifact 路径复制到两棵命名不同的树。第二节点未运行独立领域 episode，却直接标为 `SUPPORTED_IN_SCOPE` 和 `local_solver_guidance`。角色映射固定为 `identity_canonical`，hop error 固定为 `0.01`，source episode refs 未填写。节点、树、cue 切分和 distractor hub 的标签都来自任务脚本。[写回与 twin 形成](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/m4_4_instance_role_closure_and_online_writeback.py#L703-L811)

这样的构造可作为同一已确认实例的两个检索别名，且四臂拥有相同起点，是有效受控压力场景。它本身不能证明系统从两个独立领域的历史经验发现了同构。12 个后继目标的 `non_direct_walk_win` 全为 false，也不支持“这些成功必须经过横向虫洞”这个更强因果结论。

现有重新拟合与新确认保护了后继目标；若把 twin 进一步用于另一个物理实例的 zero-fit 指导，证据范围必须单独绑定。建议将“共享数学方法”“源校准实例”“已在第二域确认的实例”分别表达。没有第二域证据时给别名或 bounded_trial 地位；真正 target confirmation 成功后，再登记第二域实例、映射、误差和来源。

补充的有限边界控制复用本轮官方 pytest 实际产生的 S02 W6 artifact，包含7条训练前缀和1条独立确认，不重新发现或拟合。在 x=3.2，primary 与同源 alias 都正确输出 `6.98458573`；若将 twin 解释为正式 S04 的不同参数物理实例，R3 仍输出该源值，而该目标真值为 `6.55869399`，误差 `0.42589175`，仍获 `local_solver_guidance`。这只是说明别名不能自动升级成另一已确认实例；正式 S04 有重新拟合保护，成功记录不变。实际回执及三文件来源包在 `online/twin_scope_control.json`、`online/pytest_w6_artifact_provenance.json`。该包由审计保存，明确不属于提交官方 evidence，未替 runner 补齐归档。

仓库已有可复用形成能力：M2 的 `open_workspace_wormhole.py` 先对各域分别运行真实 source episode，检查 strict pass/pure AST，再按结构分组，至少两个节点才建边；hop error 来自实际两域误差，保留双方 episode refs。新在线 helper 缺的是接回这条证据链，不需要另造一个哲学森林或新的发现器。[历史两域形成路径](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L1730-L1918)

### 5.5 保持性与自动化范围需要如实记录

流尾历史 R3 的十个查询包含九个数值请求和一个拒绝请求，覆盖六个历史节点，未覆盖本次 S01 修订的 W4。`all_routes_exact` 只比较 selected node；数值通过率、最大误差和正确拒绝由另外的返回字段表达。Part3 summary 仅保存 routes 与 any_fit，没有保存这些数值质量字段及完整十条回执。W6/W7 则各检查 primary 树一个原参数点并明确检查容差。

当前没有发现实际遗忘；现有交付也不足以支持“所有历史能力零遗忘”的全称断言。最小修复是保存现成 R3 完整返回值，断言九个数值点和拒绝条件，再覆盖本次真正改动的 W4 边界，不需要重开庞大 benchmark。[R3 返回与 Part3 保存字段](https://github.com/erzhu419/Assumption-Agent/blob/b965f022b214f47a1341559f20041da6a224d66d/Hegel%20Machine/src/hegel_machine/cognition/m4_4_instance_role_closure_and_online_writeback.py#L1167-L1241)

写回触发也仍由预登记的 S01/S02/S09 元数据控制，其余步骤 `wb_meta=None`。这是一条含两个真实发现插入事件的持续流原型。要推广为一般在线学习，应把接入条件连接到运行结果中的 confirmed artifact、novelty 和适用域，而不是继续增加按步骤编号分支。

## 六 对项目哲学目标和 RSI 的判断

本轮已经给出一项有意义的工程对应：先前形成的原子概念经组合形成新结构，新结构进入可检索历史，随后改变求解开销。这与项目用历史概念形成新假设的设计方向一致。其价值不依赖所有目标都由零拟合解决；结构迁移后重新确认，本来就是经验知识保留条件的合理方式。

跨域抽象需要更严格地区分“程序同形”与“不同领域各自支持的同构”。复制同一模型到两棵树可以验证检索器，但没有产生第二份经验根据。若系统随后把别名当作新领域事实，历史积累会膨胀出无独立证据的节点。把别名、结构候选、域内确认实例和可执行跨域关系分层，将使哲学框架真正约束系统行为。

本轮主要改进的是对象层知识与调用基础设施：新增 AST、实例锚、角色锁、版本刷新。方法层仍是预设组合语法、检索规则、固定报价和固定写回位置；没有实验显示系统根据自身历史自动提出并验证一项学习策略改变，然后在后继冻结评价中获得净收益。因此更准确的里程碑名称是“可验证的在线结构记忆与 M5 桥接原型”。下一步可把反例驱动的条件修订、保留旧有效实例和状态恢复组合成一条真正长期运行的链。

## 七 有限修复清单与 M5 验收

### 7.1 先完成四个修复包

| 修复包 | 复用当前能力 | 最小验收 |
|---|---|---|
| 证书关系一致性 | 现有 revalidate identity guard | 新建与重验都拒绝真实 S0/S2 跨实例组合且无状态变化；合法同步替换通过；归档检查识别不一致关系 |
| 统一有效动作与角色 | `selected_action`、现有 role helper | 普通 OR 的 B 完整、A 不可用仍能召回；规范轴矛盾映射拒绝；合法匿名映射仍通过 |
| 在线语义和确认条件 | 已有 predicate/box、confirmed artifact 接口 | S01 若仅元数据更新则准确命名；若为作用域修订则须实际改变域内/域外行为；失败不作为成功写回事由 |
| 在线状态和保持回执 | 当前相对定位符、serializer、既有 checkpoint/paid history 基础设施 | 临时源删除后冷恢复 W6/W7 和终态森林，继续一个后继目标；保存真实来源、完整 R3 数值与拒绝记录 |

Twin 的身份与权限应同时在第三、第四包澄清：别名保留同一实例身份；独立域授权等待第二域确认。当前模型语法、PPR、pooling、Compiled 强基线都可以保留。

### 7.2 下一条 M5 链以真实行为变化为中心

建议采用一条有限、预登记的序列：确认一个新结构，完成两个后继复用，收到一个实际反例，生成可执行的条件修订，分别检验域内继续使用与域外拒绝/下钻，冷恢复后再调用新旧机制。让更新由回执触发；预算、已付观测和失败历史连续保留。先把这一条链做成，再扩展到多形成种子，避免用更多静态目标掩盖更新语义。

对照保留当前 live Compiled，另外用能同样保存新 AST 与实例的普通动态记忆隔离“保存结构”与“图组织”的作用。Pooling 的研究问题应进入真实拟合/确认链，并设置一个跨域映射不成立的负例，使未经确认的边不能自动升级权限。费用同时给出 M、各阶段 K、初建/刷新/写回、必要的本地时间测量；代理成本与墙钟分别报告。

验收的关键是：新增经验是否改变后续合法行为，反例是否只撤销超出证据的部分，旧有效知识是否能恢复和继续消费，以及全部前期与维护代价后是否仍有净收益。这些要求可以直接用现有组件完成，不需要无限增加守卫或重写已经成立的体系。

## 八 证据包索引

| 文件或目录 | 内容 |
|---|---|
| `AUDIT_REPORT_ZH.md` | 完整综合诊断 |
| `README_REPRODUCE.md` | 固定提交、环境、官方重放与有限控制命令 |
| `change_scope.json`、`environment.json` | 变更范围和执行环境 |
| `verify_replay.py`、`replay_verification.json` | 两套证据包的逐文件 SHA、逐字段差异、回执唯一性和账本一致性 |
| `recompute_part2.py`、`replay/part2_receipt_recalculation.json` | 家族/尺度维护收益、回本曲线与失败行复算 |
| `replay/evidence/` | 本轮独立执行官方 runner 的完整输出 |
| `identity/` | 16 个动作锚控制、规范/匿名映射、普通 OR selected_action 控制及源码阅读清单 |
| `index_archive/` | 证书新建/重验、三个索引刷新、三快照与历史节点、Part3 费用逐项归因 |
| `online/` | 80 条在线回执复算、S01 失败和实际范围控制、写回来源与归档分析 |

所有对照保留正例和限定条件；探索中不满足生产前提的 Pi 混合分支案例已注明排除，未用来宣称性能失败。报告以源代码与实际回执支持具体结论，不把测试通过表述成全系统不存在剩余问题。
