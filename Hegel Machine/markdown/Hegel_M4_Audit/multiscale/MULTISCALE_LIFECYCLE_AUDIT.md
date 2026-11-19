# M4 多尺度与持续生命周期专项审计

固定提交：`11961f702739ab272661ff59b77fb5bc8e3589b8`。对比基线：`612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8`。

审计源码根：`/workspace/scratch/3f14ad2da0ed/repo_m4_11961f702/Hegel Machine`。本报告中的行号均针对固定提交。代码保持只读；未调用付费模型、未远程写入、未在本专项重跑完整研究。先查阅 AGENTS、reuse hub，并以 `multiscale compression coarse graining`、`compression`、`fidelity` 查询当前 historical 路由，均无匹配，因此继续直接追踪现有实现；不把索引无匹配解释为整个项目没有历史能力。

## 结论

M4 已经建立一个真实执行的合成双通道实验台：粗测、细测分别调用不同的真实函数，按点收费；闭合探针比较实际返回值；候选模型确实经过拟合和隔离执行；最终留出评价使用微观函数；图中确有尺度层、水平虫洞和垂直边。原报告的七臂结果是可执行实验产物，不能贬为仅有文案或预写结果表。

目前最准确的里程碑是：**有限算子族上的主动通道选择、尺度标注检索与受控记忆维护原型完成了一组可复现的组件实验。** 还不能把这一组结果提升为“从同一微观系统经过可执行 Pi_s 导出介观/宏观规律，并通过新发现持续扩大知识库、证明无遗忘”的统一闭环。

需要优先修复的代码问题是：子规律确认路径绕过最终契约授权；新 M4 路径再次只检查 `program_steps` 非空；压缩时实际提交的节点不是保真验证过的节点；节点修订使粗粒化边静默失活。需要重新设计的证据是 Frozen 的等初始记忆对照，以及包含新增知识的阶段性 retention 测量。其余多尺度物理和通用性表述可以先收窄，不必推翻已有实现。

## 已核对的正面结果

- `MultiScaleWorkspaceTaskLab.acquire_channel` 在 `m4_multiscale_renormalization.py:99–123` 根据通道实际调用 `_coarse_pi_fn` 或 `_micro_fn`，记录点数与费用。`run_active_scale_closure_probe:125–140` 确实付费采一对观测并计算绝对差，已不是上一轮被指出的常数探针。
- `evaluate_heldout_micro_truth:142–165` 从 `_micro_fn` 生成留出真值，并传入独立的训练前缀执行模型。本文局部微观偏差反例确实被它抓住，说明外部评价不是用粗模型自证粗模型。
- `hipporag_wormhole_index.py:224–238` 把有效粗粒化边放入 PPR 的双向邻接图；`414–427` 按请求尺度调低不一致节点/虫洞的排名。不能说这些字段完全没有消费者。
- `build_m4_multiscale_forest:627–675` 校验虫洞两端纯 AST 骨架相等；微、介、宏三个尺度的不同邻居确实存在于构建出的图中。只是这些模型和跨尺度对应关系为显式构建，不是本实验从微观动力学推导所得。
- 原 `multiscale_target_runs.json` 有 504 条记录。原摘要报告 Hegel 的 72/72 通过、16/16 walk、16/16 主动微观升级，8/8 OOS 返回 unknown；本专项未把下面的新增反例混入原记录。完整重放由主审计负责，据主审计反馈，全部主要业务结果一致，仅 Stage 2 NoPrune 的平均 compute 四舍五入值 `16.272→16.273`。
- 测试并非全部空洞：`tests/test_v17_2_m4_multiscale_and_open_discovery.py:40–67` 校验实际拓扑；`70–133` 校验通道选择、walk 和对照；`136–148` 校验交换坐标后的签名不变。它们的不足是没有覆盖下列关键反例。

## 1. 子规律成功路径绕过最终授权：明确的正确性问题

**位置：** `m4_multiscale_renormalization.py:1255–1290` 对首个候选制作绑定 target/version/model/confirmation 的 receipt，并调用 `authorize_abstraction_use(local_solver_guidance)`；但父规律被反例否定后的子规律分支 `1292–1331` 在第二个确认点通过后直接设置 `confirmed_ok=True`、`confirmed_via_recall=True`。

这不是仅仅“少记了一张回执”。最终授权检查未知前提、尺度闭合、丢弃变量证据和权限上限。有限试用许可可以在前提未知时成立，最终 solver guidance 不能因此自动成立。

**实测控制：** 保持原 `C0_T11_MultiScaleWalk` 的所有数值、图、候选及预算相同，仅在 capped child 增加未被测量的硬前提 `audit_boundary_measured == True`。结果仍是 `confirmed_via_recall=True`、`non_direct_walk_win=True`、`strict_pass=True`，heldout RMSE 为 0；使用同一 child 契约和有效格式的绑定回执直接调用最终授权，返回 `granted=False`，理由为 `unknown_precondition_requires_probe_before_guidance`。另一个更直接的控制拒绝所有最终授权调用，walk 成功路径仍完成，调用记录只有四次 bounded_trial，根本未调用最终授权。

**修复建议：** 首选候选、子候选、回退候选统一经过一个“数值确认→生成绑定回执→作用域更新→最终授权→记成功”的出口。新增前提未知时继续付费探测或保留 bounded_trial，不把数值拟合成功等同于契约升级。

## 2. 新 M4 路径没有解释执行程序步骤

**位置：** `_register_multiscale_source_node:369–374` 登记四步研究程序；消费者 `evaluate_m4_target_under_arm:1199–1201` 虽注释为 “Interpret program_steps”，实际仅判断列表是否非空。后续的探针、校准、拟合、确认流程全在函数体中硬编码。

**实测控制：** 对整个森林所有节点，将步骤改为非空垃圾 `[{"step":"NOT_AN_IMPLEMENTED_OPCODE","fresh_points":-999}]`。宏观绝热、介观绝热、闭合破裂、两跳 walk、坐标置换五个代表目标的所有返回业务字段、候选数和成本（排除 wall time）与原始结果完全相同。

这限定为新 M4 消费路径的缺陷，不否定 M3.1 在旧路径上的修复。仅测“把列表清空导致回退”也不能证明程序内容被执行，因为非空性本身就是门控条件。

**修复建议：** 复用已经修好的解释器或共享执行器；未知操作码明确拒绝。若本阶段只希望存程序说明而不执行，则将字段与文档降级为计划描述，不声称已形成可执行研究程序迁移。

## 3. 目前 Pi_s 是双函数近似接口，加上尺度拓扑；不是跨尺度动力学推导

**位置：** `MultiScaleWorkspaceTaskLab:67–94` 接受两独立函数。生成器 `737–741` 直接定义 Murray 幂律和一个几乎消失的快模修正；`774–777` 直接定义介观函数与小正弦偏差；`826–830` 直接定义线性慢项与快指数修正。图上的 Pi_s 名称和 defect_bound 在 `457–468`、`533–544`、`614–625` 手工填写。源码没有实现把分子状态映射为胞元面积/长宽比、再映射为树流量的实际投影算子。

报告 14 第 18–20 行所述“经 Pi 消去快模后涌现六边形”“经空间树聚合涌现 Murray”“沿 coarse_graining_edges 逆向下钻”超出了代码。本实验的升级由 `1146–1154` 直接将 `selected_channel` 设为 micro、`active_scale_regime` 设为 micro，再重新检索；并未调用森林的 fine-scale traversal 来决定目标节点，也未执行边上投影。

**去边控制：** 五个代表目标保留节点、算子、虫洞及所有数据，只在建索引时关闭 `include_coarse_graining_edges`。五个目标的质量、测量量、拟合候选数、微观升级和 walk 指标全部保持；PPR 计算费用略有变化，说明边的拓扑确实被计算，但这些样例没有证明它对跨尺度推理必不可少。**不能由五个样例推论这些图边在任何任务上都无用。**

**最低增补：** 选一个已有快慢系统，定义真实 fine state、显式 Pi_s、coarse dynamics 和 preserved query，在同一微观轨迹上计算投影与闭合误差；删除/篡改 Pi 或对应边应改变下一步实验或预测。一个受控实例足以补上机制证据，不需要同时模拟水、蜂巢、河网和 CDN 的全部真实物理。

## 4. 一点 probe 是局部证据，不足以签发整个稳态域的闭合证书

**位置：** `125–138` 用一个点的 defect ≤ `0.85*tolerance` 直接返回 `closure_verified=True`、`closure_scope="steady_state_only"`；节点 ScaleMap 的 `error_bound=0.015` 与 `src_scale_cert:*` 在 `335–352` 固定登记，并不是这次 target 探针估计出的全域误差界。

**实测作用域反例：** 在原宏观目标微观函数上增加

`exp(-((q_flow - 5.8) / 0.20)**2)`。

它在指定 probe 点 `q=4.0` 与确认区域近乎为零，在独立留出点附近有明显偏差。实际得到：probe 仍闭合，系统走 coarse，`confirmed_ok=True`，但微观 heldout RMSE 为 `0.447214`，于是 `false_confirmation=True`、`strict_pass=False`。

**正确解读：** 这是“闭合证书作用域过宽”的反例。独立外评确实发现了问题，因此它不构成原 504 记录作假、holdout 泄漏或评估完全失效的证据。

**修复建议：** 区分 `closure_observed_at_points` 与 `closure_supported_in_region`，记录输入域、测量点、误差统计、模型版本。受控场景可以用少量代表点/边界点和明确的先验光滑性界；不需要声称一个有限 probe 能证明任意函数全域闭合。实际误差界不应被固定的源证书替代。

## 5. Frozen 起点不等价，不能作为持续学习可塑性证据

**位置：** lifecycle 在 `1433–1435` 复制完整 Stage 1 森林；随后 `1561–1565` 特意从 Frozen 删除已经存在的 `bio_vascular_macro_capped`。原报告表中的 Frozen 在 Stage 1 就是 16 个节点，而完整臂已有 17 个；75% recall 差距从 Stage 1 即存在。

**实测控制：** 对同一 `C9_T11_MultiScaleWalk` 使用同一个 Frozen arm，只改变初始内容：

| Frozen 内容 | recall | 候选数 | 测量费 | 复合成本 |
|---|---:|---:|---:|---:|
| 完整 Stage 1 记忆 | True | 2 | 44 | 44.9453 |
| 按原实验删除 child | False | 5 | 56 | 57.8335 |

因此，这组差异证明缺少 child 会增加重建成本；还没有证明发生新知识后，冻结记忆的性能落后于持续更新记忆。

**修复建议：** 所有臂从字节相同的 Stage 1 开始。Stage 2 才引入一个源历史中不存在且被独立确认的新规律，在全部臂都经历相同新数据后，只冻结臂不更新记忆；随后使用独立 target 比较。原删除 child 控制可以保留，但应改名为“缺失结构记忆对照”。

## 6. K64 和零遗忘目前是冗余/诱饵维护测试，不是新知识持续学习测试

**位置：** `1471–1555` 每阶段加 6 个已知模板副本、8 个诱饵，以及 2 次只改 `last_verified_stage` 的修订。诱饵在 `1533–1535` 被直接喂入三条 `saved_cost=0, overhead_cost=8, succeeded=False` 的反馈，没有实际利用失败的试验链。`1437–1444` 的评价集始终是同样 12 个既有目标；注释提到的“四个阶段性 backward retention probes”没有加入。

`1606` 将 `backward_forgetting_rate` 定义为 `1-pass_rate`。这能报告当前固定集合的失败比例；它没有按学习阶段保存旧知识集合，也不是跨阶段能力差值。所有臂，包括无压缩和删节点 Frozen，原表都为零，更不能单独归因给压缩。

另一个计数小问题：Stage 1 实有 11 个核心节点加 6 个干扰节点，共 17，注释/阶段标签写 16。实际注册节点曲线为 `17→31→45→59`；加上六次后续 revision，若以 episode 计则为 65 而非 64。它不影响主要通过率，但 K 的定义需统一。

**成本边界：** Full 和 NoPruning 的原测量均值均为 57，候选成功/recall 也不退化；末阶段 composite `57.797 vs 57.849` 的约 0.052 优势主要来自 index 的显式线性节点计价。压缩验证、诱饵验证及重建索引的维护成本未纳入该阶段均值。当前证据支持“活跃检索集合可被去重归档并保持这 12 个任务性能”，不支持“总长期维护净成本已降低”。59 个注册记录仍全部保留，71.2% 应称活跃节点数下降，而非未经测量的总存储体积压缩。

**建议：** 保留这组便宜的重复/诱饵压力测试；另加最小的新法则→局部反例→保守修订→旧任务回访序列，存每阶段已学任务清单与 raw 结果，并同时报告 recall、fallback 成本、总成功率和维护成本。

## 7. FidelityCompression 有两种不同问题，应分开处理

### 7.1 本实验的 evaluator 不检查预测语义

`m4_multiscale_renormalization.py:1446–1464` 的四个 audit queries 和 evaluator 只检查 arity 与少量 preconditions。它不检查算子/AST、ObservationMap、尺度、程序以及实际预测误差。当前副本由作者预先选择为同式，因此测试本身不会遇到这一漏洞。

**控制结果：** 用完全相同 evaluator 请求压缩 `aq_macro_river_tree` 的 Murray 幂律与 `dist_node_00` 的 Monod 饱和律，仍返回 `accepted=True`、冲突索引为空。

这不意味着原实验确实压错了它生成的 18 个同式副本；它说明“零冲突”证书仅针对当前审计问题集，而且当前问题集没有测算子语义。先做结构/观察映射兼容性检验，再以实际输出和边界任务验证即可。

### 7.2 Wrapper 实际写入的节点不是 helper 验证的节点：明确 API bug

`wormhole_forest.py:452–518` 的 `fidelity_compress_nodes` 会生成含 `disjunctive_branches` 的 `merged_node` 并验证它。新增 wrapper `1048–1096` 成功后只合并 signatures/provenance、归档冗余节点、重绑部分虫洞，没有把 `report["merged_node"]` 的合并规则提交给 keeper。

**控制结果：** 提供一个诚实 evaluator：A 支持 case A，B 支持 case B，合并分支支持两者。helper 正确返回通过、被验证 merged 在 case B 为 `ACC`；wrapper 执行后，实际 keeper 在 case B 为 `UNK`，原 B 已 archive。

这不是要求无限完备的压缩证明，而是要求“被验证的对象等于被提交的对象”。修复时应安装经过验证的 merged contract/program，创建新版本并更新相关边，然后对实际安装对象重放小型 retention 面板。若只允许同一程序的精确去重，应在 API 中明确拒绝不同程序/契约而非悄悄丢弃分支。

## 8. 节点 revision 后粗粒化边没有重验，6 条有效边变成 4 条

`empirical_revise_node` 的版本机制会使旧端点边不再有效；lifecycle `1537–1555` 只处理横向 wormhole。index `227–238` 对 coarse 边要求端点版本完全相等。

**精确重放 Stage 2 的两次 revision：** 有效 coarse 边 `6→4`，`fine_scale_children_of("aq_meso_hex_cell")=[]`，`coarse_scale_parents_of("aq_meso_hex_cell")=[]`；与此同时所有横向 wormhole 的 stale 标志均为 false。因此 `zero_stale_wormholes_left_unrevalidated=True` 没有覆盖已丢失的尺度梯级。

**修复建议：** 为 coarse 与 implication 边建立相同的 stale/revalidate 生命周期；变更只涉及无语义影响的协议时间戳时允许通过明确证据重绑；作用域变化则验证投影条件。增加“应该保留的可达关系是否仍存在”的断言，不能只检查横边布尔标志。

## 9. 消融和边界表述

`ScaleBlind` 同时关掉 `scale_aware`、取消闭合 probe、取消尺度请求值，并使用 micro confirmation；因此七臂比较是多尺度策略包的对照，不是仅测尺度索引的单因素消融。`MicroOnly` 又直接读取 `lab.true_scale_regime`（1132），并缺少 probe 产生的签名。这些不同未必都偏袒 Hegel，但不应混成某一个组件的独立收益。最少拆出“相同 probe 与确认，仅切 index scale_aware”和“相同索引，仅切 probe/通道策略”即可。

`domain_cue`、`candidate_coarse_regime` 和 `scale_boundary_point` 由目标生成器给定。可以把它们定义为公开任务描述，但应明确列入可见输入，不能凭 `sig:` 前缀就证明一切来自自主测量。四个 source cluster 的 seed 在建森林函数中未被实际使用，源图内容一致；target 参数随 seed 改变，故证据主要覆盖四组参数实例。

Pillar 1 的发现结果与 Pillar 2/3 尚未串起来：主脚本先运行 `comp_suite`，随后重新调用手工 `get_m4_operator_library` 构建森林，没有把 Pillar 1 的持久化作者结果导入该森林。因此“组件具备并分别通过”与“统一自生长闭环已打通”仍需区分。

## 建议的最小修复顺序

1. 统一最终确认/授权出口；把 M4 接到真正的程序解释器；修复 validated-merged/committed-keeper 不一致。这三项直接涉及执行语义。
2. 修复粗粒化边版本生命周期；将 closure receipt 改为明确的点/域证据，并保留 independent micro holdout。
3. 复用现有基线修正 Frozen 等初始状态，补一条真正新增规律的后继与旧知识回访；将维护成本加入账本。无需扩大到大量新物理域。
4. 把现报告定位改为受控组件验证，保留已完成七臂结果；当一个真实 Pi_s 和“Pillar 1 作者结果→森林增长→独立迁移”纵向实例接通后，再升级 M4 收官结论。

## 可复现证据

- `probe_multiscale_lifecycle.py`：全部上述轻量控制，约 4.013 秒；共 20 次 target evaluation（15 次原始/垃圾步骤/去边，2 次 child 授权，2 次 Frozen，1 次局部闭合反例），另有纯内存结构检查；零模型调用。
- `probe_results.json`：完整机器可读结果，包含实际 paired probe 值、所有控制、授权理由、节点/边数和成本。
- `probe_workspaces/`：上述目标的候选模型；没有向原源码或历史证据目录写入。

复现命令（在固定 worktree 的 `Hegel Machine` 目录运行）：

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 /workspace/scratch/3f14ad2da0ed/audit_m4_11961f702/multiscale/probe_multiscale_lifecycle.py
```

关键原始材料：`docs/evidence/v17_2_m4_multiscale_open_discovery_20261119/{summary.json,multiscale_target_runs.json,continual_k64_stages.json}`；报告 14 第 3、18–23、47–70 行；两个源码模块及 index 的上述精确行号。当前已有结果无需因本报告被删除；应保留、补齐控制并修订其证明范围。
