# V17.4 / M4.2：统一发现链与评测设计专项审计

固定提交：`fb4b23aaf802c20b4e2d0018436abab73b0cca06`。本报告只涉及统一 Stage 0/1/2/3、对照公平性、DEV 冻结、两跳分岔、确认数据复用与维护价值。完整账本、Pi 实际测量通道、证书/盒外授权、Dispatcher 由主审与其他专项报告负责。

## 一、结论与范围

**M4.2 确实把许多原来分散的能力接进了同一个实验流程；但当前九臂差额、Frozen 回本和“任何静态方法都无法避免多购确认”不能作为已经完成的架构因果证明。** 最直接的工程问题是 Frozen 重新丢失 Stage 0 已经形成的节点与边，以及确认点耗尽后把已进入训练集的旧点重新当确认点。DEV 使用独立 seed 属实，但只调了顺序排列后的前 10 个目标，覆盖分布不完整。分岔对照证明了当前先过容差即停止的顺序策略容易选错，未证明已有 pilot 不能分辨两种机制。

本专项读取了 AGENTS、reuse hub / open_grammar 历史查询、交接文档、report 16、M4.2 新模块对应全过程、M4.1 被复用的候选拟合/签名提取函数、BM25/PPR 检索器、新测试和 runner，并对提交里的全部 792 行 receipt 进行逐行归组与一致性分析。没有修改源码、原始证据或调用线上 API。完整测试与 full study 由主审执行，本专项没有重复。

另做一次共享建林，使用 Python profile 在第一轮 Stage 1 发现调用之前捕获真实 Stage 0 边界，原样保存 Stage 0、Stage 2 前后快照及 builder 返回值；随后仅做两个有限设计控制，以及一条既有确认复用路径的执行跟踪。

文中代码简称：

- **M42**：`src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py`。
- **M41**：`src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py`。
- **INDEX**：`src/hegel_machine/cognition/hipporag_wormhole_index.py`。
- **REPORT16**：`markdown/hegel_v16_5_multi_precision_plan/16_V17_4_M4_2_Audit_Closure_Unified_Pi_and_Dispatcher_Report.md`。

## 二、应保留的真实改进

1. **形成链确实被串起来了。** M42 693–1880 先做真实原子标定与 Pi 认证，再完成五个组合算子发现，接着对同一 forest 修订、压缩、归档和重验证。Stage 1 仍是明确 atomic bases 与有限组合 DSL 驱动的离线 CodeAgent 实验，但不是在 Stage 3 按 target family 临时塞入一个已知答案。新的 W4/W5 算子以产物形式进入后续 forest。
2. **三次 lure DEV 现在不同。** M42 1611–1670 的 chemostat / servo / membrane 使用不同特征、公式、参数、点集，已修复上轮相同任务重复三遍的问题。
3. **同一 post-Stage-2 快照确实在每个 seed 前深拷贝。** M42 2593–2598、2620–2627 还把 Dispatcher 放在另一个 deepcopy 上，避免 Dispatcher 或前一 seed 污染后续 seed。这是正确的控制。
4. **失败确认的值确实被传入后续拟合。** M42 2440–2442、2472–2474、2495–2497 保留了失败点，并排除已经 refute 的 tag。上轮 paid-history 断裂有所修复。下文的问题是保存后又把同一点当“新确认”，不能据此说“失败数据仍然完全丢失”。
5. **九臂记录数与去重标识一致。** 792 个 `(seed, arm, target_id)` 键全部唯一，即 4 seeds × 9 arms × (20 in-scope + 2 OOS)。每臂 80 条域内与 8 条 OOS，不是重复计数得到 792。
6. **W4/W5 非直接召回真实存在。** Hegel 的 48 次非直接命中由 16 个 MonodOnset、16 个 MurrayOnset、16 个 Pi 组成。W4/W5 节点没有目标 domain cue 的直接签名；源桥接点、横边和 local implication 确实构成可运行的两跳关系。不能因为对照不足就抹掉这一点。

## 三、P1：Frozen 不是形成完成时的 Stage 0 快照

### 源码事实

M42 的 Stage 0 在第一次 `_run_discovery_ep` 之前已经含有 **8 个节点、2 条 coarse-graining 边、1 条横向 wormhole**：

| 真实 Stage 0 节点 | 九臂评测的 Frozen 是否保留 |
|---|---|
| `stage0_monod_2d` | 是 |
| `bridge_membrane_carrier_onset` | 否 |
| `stage0_murray_1d` | 是 |
| `hydro_micro_pore_1d` | 是 |
| `bridge_arbor_branch_onset` | 否 |
| `ode_micro_2mode_node` | 是 |
| `ode_meso_schur_pi_thermal` | 否 |
| `ode_meso_schur_pi_hydraulic_dup` | 否 |

其中 carrier bridge 与横边在 831–866，hydro coarse edge 在 929–934，thermal Pi 节点及 coarse edge 在 995–1053，hydraulic 副本在 1055–1086。

然而 M42 **2600–2603** 新建空 forest，只复制四个 node dictionary，所有边都没有复制。两个 Frozen 臂仍承担完整 Stage 0 的 `244.05` 前期费用（2687–2696），但没有获得已付费形成的完整 Stage 0 能力。这不是“冻结后未形成新节点”的合理限制，而是冻结点本身被错误截断。

### 决定性控制

`capture_shared_build.py` 在运行时真实边界截获 `deepcopy(forest)`，没有从测试结果反推节点。分别用原四节点 Frozen 和该真快照，在完全相同 fallback pool 与 evaluator 下评两个代表目标：

| 目标 | 原四节点 Frozen | 真 Stage 0 | 实际差异 |
|---|---:|---:|---|
| T09 MonodOnset | 39.50 | 40.10 | 都 9 次观测、同 Haldane fallback、独立 heldout 通过；真快照多 2 次原子候选评估 |
| T17 Pi | 57.55 | 29.55 | 都 7 次观测、同 `op_atomic_exp_relax_2d` fallback、同 3 次候选评估、同 heldout 通过；只是测量单价由 8 变 4 |

T17 的差额 **28.00** 并非恢复粗尺度执行后自然省钱；它揭示“任何 coarse edge 存在即降价”与错误 Frozen 起点的相互作用。真 Stage 0 本来就已有 coarse edge，所以原实验人为把 Frozen 放进了高价档。主审的实际测量通道与 tariff 专项解释这一层。

T09 变贵这一结果也一并报告：恢复正确快照不保证每个任务都变快，不能据两例推导一个重新计算的 80 目标均值或新回本点。能确定的是，原始 `Frozen_After_Stage0` 命名/起点与形成过程不相符，原回本比较需要在修正快照和测量通道后重跑。

**最小修复**：在 Stage 0 完成、Stage 1 开始之前保存并返回完整 snapshot，九臂 runner 使用它的 deepcopy；不要按节点名重建，也不要遗漏历史、各类边或证书。新增一个边界一致性断言即可，不必重写 forest。

证据：`true_stage0_snapshot.pkl`、`true_stage0_forest.json`、`frozen_snapshot_control.json`。

## 四、P1：32 条真实运行把已用于训练的确认点重新确认为“通过”

M42 **2526–2528** 在没有未使用确认点时执行：

```python
if not rem_cf_pts:
    rem_cf_pts = [target.confirm_points[-1]]
```

上游两轮 ordinary confirm 失败后，两个点都已经加入 `frozen_prefix`。fallback 用包含这些点的 prefix 重新拟合，然后再次购入并评估最后一个旧点。多调用一次 oracle 不会使一个已用于训练的确定性输入变成独立新样本。

这条路径在提交数据中 **实际发生 32 次**：

- DEV 冻结臂，16 个 MurrayOnset 全部发生。
- Forward 臂，16 个 MurrayOnset 全部发生。

对应三种 phase 同时出现：`ordinary_confirm_0`、`ordinary_confirm_1`、`fallback_confirm`。直接重放 `seed=20261120, T13` 的 evaluator 调用轨迹为：

| 确认阶段 | 查询点 `(solute_load,bypass_flow)` | 训练 prefix 行数 | 查询是否已在训练 prefix | 确认 RMSE |
|---|---|---:|---|---:|
| ordinary 0 | `(11.5, 0.6)` | 6 | 否 | 9.7110 |
| ordinary 1 | `(8.5, 1.0)` | 7 | 否 | 0.55288 |
| fallback | `(8.5, 1.0)` | 8 | **是** | 2.85e-10 |

**不要把这改写成“发现 32 个预测假阳性”。** 提交中的这 32 条独立 heldout 都确实通过；本例独立 5 点 heldout RMSE 为 `7.16e-9`。问题是在线确认协议失去独立性，并多计一次无新增信息的确认采购。当前 receipt 的 `strict_pass` 同时依赖 outer heldout，因此这批 32 个值仍为 true；没有额外证据时不能强行把预测结果改为 false。

**最小修复**：确认池用尽时，应通过现有 paid-history / probe 入口获取尚未用于训练的新输入，或返回“证据不足”而非重复确认；需要显式记录确认前的 prefix digest、query digest 与 disjoint 检查。已付失败点继续保留，不能为“恢复独立性”而从历史中删除其信息。

证据：`receipts_analysis.json` 的 32 条完整记录，`confirmation_reuse_control.json` 的真实调用轨迹。

## 五、P2：DEV 独立 seed 是真的，但并非覆盖测试分布的强静态基线

M42 **1755–1790** 的 DEV 使用 `seed=99_112_026`，没有把四个 TEST seed 的结果传入排序器。不能声称这里直接读取了 TEST 成绩。问题是 **1760 只循环 `dev_targets[:10]`**，而 `build_m4_2_target_suite` 固定按五个家族分块排列：

| 家族 | TEST 每 seed | DEV 实际纳入 |
|---|---:|---:|
| HaldaneShunt | 4 | 4 |
| OscShunt | 4 | 4 |
| MonodOnset | 4 | 2 |
| MurrayOnset | 4 | **0** |
| Pi | 4 | **0** |

冻结器按“候选在多少 DEV 任务的 pilot+首确认上成功”排序，再按原 chronological 顺序破平局。它没有优化实际顺序搜索总开销，也没有评估完全缺席的两类。保存的成功次数是 Haldane 6、Osc 4、MonodShunt 2、其余 0。这也解释了 MurrayShunt 被排在最后。

提交结果中 DEV 均值 `38.855`，比不做 DEV 的 Reverse `36.295` 更差。Reverse 有 1/80 Pi 假确认，故它也不是可直接宣称更可靠的最优基线，但这个结果足以反驳“DEV 已冻结最优静态序”的措辞。REPORT16 33–35 的“零窥探 DEV 冻结最优”应降级为“以独立参数 seed 的部分任务、按命中频次冻结的一种静态序”。

DEV 与 TEST 使用相同生成器、点集模板和机制类别；独立 seed 主要带来参数/符号/偏置变化。这是一种有效的受控参数迁移设计，不能等同于机制类别或观测设计完全独立的外部泛化。

**最小修复**：按五家族分层选 DEV，冻结前以同一预算和真实执行成本评价顺序；保存 DEV 目标集合、评分目标和最终冻结摘要。无需搜索所有阶乘排列，也无需按 TEST 真值选顺序。改完后另留未参与这次诊断的新 TEST seed。

## 六、P2：两跳分岔证明了当前策略优势，没有证明 pilot 的信息不足或图游走不可替代

### 报告的推理跳跃

REPORT16 **33–35** 从“两个候选在低负载 pilot 上都小于 tolerance”推出任何静态列表至少一半目标必多购高价确认，并称唯有两跳游走能全部一次命中。

M41 **1400–1441** 的实际选择器是：依列表顺序拟合，只要某候选 RMSE `<= tolerance * 0.85` 即停止。两个不同候选都低于应用容差，当然可以让这种 **first-acceptable** 策略在某些顺序上先选错；但“都可接受”不等于“两种 pilot 数据在统计上不可区分”。当前数据完全无噪声，正确模型常达到 1e-10 量级，错误模型仍有 1e-3 或 1e-2 残差。评分规则主动忽略了这些可用信息。

### 限定的 post-hoc 可判别性控制

这里只针对报告声称不可区分的那一对算子：所有 8 个 onset 目标都固定试同一对 W4/W5，从相同 6 条已付 pilot 拟合，**仅按 pilot RMSE 最小值选择**；之后才揭示首个 confirmation 与 heldout。没有按 `mechanism_family`、目标名称、确认或 heldout 真值选答案。

| 目标 | MonodShunt pilot RMSE | MurrayShunt pilot RMSE | pilot 选出的模型首确认通过 |
|---|---:|---:|---|
| T09 | 3.60e-10 | 0.02727 | 是 |
| T10 | 4.53e-8 | 0.02410 | 是 |
| T11 | 7.64e-10 | 0.02813 | 是 |
| T12 | 1.32e-8 | 0.03102 | 是 |
| T13 | 0.001404 | 1.18e-10 | 是 |
| T14 | 0.001289 | 1.12e-10 | 是 |
| T15 | 0.001378 | 1.39e-10 | 是 |
| T16 | 0.001253 | 1.08e-10 | 是 |

**8/8 首个新确认通过，8/8 独立 heldout 通过，额外分辨测量为 0。** 每个目标对两种模型各试两个 role permutation，合计 4 次候选拟合；它没有证明计算成本比 Hegel 更低，也没有把全部历史候选都纳入，所以不能把它当成重新赢得九臂 benchmark 的完整 baseline。它只否定了“当前六个 pilot 无法识别分支、必需额外物理测量”的强推断。未知噪声条件下哪种策略更好，本控制没有回答。

### 同信息问题还包括先验 cue

M42 2100–2188 的两组 onset 使用相同坐标、点集与容差，但分别附上 `cue_carrier_binding_onset` / `cue_vascular_branch_onset`；M41 1170–1230 把这个外传 domain cue 直接作为 observable signature。Stage 0/1 又人工登记了对应该 cue 的源桥、横边和 specialization 边（M42 831–866、939–963、1372–1482）。这是一种有效的领域上下文先验，必须说明其来源。它不等于系统仅靠六个数值测量自主推断了机制归属。

Ordinary 列表只消费 `(operator_tag, operator_spec)`，不使用这些 cue。FlatBM25 保存完整 node metadata，真实排名却只在 `probe_signatures` 上打分（INDEX 506–524、566–587），不会把桥/蕴含邻接传播为目标候选特征。因此这些对照能衡量“当前直接匹配/固定列表 vs 结构化关系检索”的收益，不能覆盖所有同信息的非在线随机游走路由器。

有限的下一步是加入一个预先冻结的同信息路由控制，例如利用 DEV 学 cue→候选优先级，或者在形成阶段把关系闭包编译成查询表并计入构建/更新成本。它仍然可以保留哲学 forest 作为知识表示；这个控制只区分在线 PPR、关系信息本身与接受阈值策略的贡献。不要把结果解读为应当放弃 graph，也不要宣称任意静态顺序都能替代 graph。

另外，M42 **2631** 的 Alpha0 臂同时设 `damping_alpha=0` 和 `wormhole_equivalence_pooling=False`，而 Hegel 两者均开启，所以 Stage 3 不是单独 alpha 消融。Part 2 的旧任务 2×2 因子实验是真改进，但不能自动把 Stage 3 的这个合并差额归于单一因素。

证据：`posthoc_pilot_pair_control.json` 与 `probe_unified_controls.py`。

## 七、NoMaintenance：有边际改善，但本次只有 Pi 家族贡献

原始 80 对逐行对齐得到：所有 64 个非 Pi 目标，两臂的 measurement cost、compute cost、candidate count、recall 结果均相同。全部 `58.775` 的 Stage 3 总节省来自 16 个 Pi 目标。

| 家族（各 16 条） | Hegel mean cost | NoMaintenance mean cost | Hegel / NoMaintenance recall |
|---|---:|---:|---:|
| HaldaneShunt | 29.10 | 29.10 | 16 / 16 |
| OscShunt | 28.95 | 28.95 | 16 / 16 |
| MonodOnset | 29.55 | 29.55 | 16 / 16 |
| MurrayOnset | 29.55 | 29.55 | 16 / 16 |
| Pi | 28.95 | 32.6234375 | 16 / 0 |

这表明 Stage 2 后的 Pi 结构/作用域合并对当前访问流有用；它没有分别证明 Monod scope revision、Minton lure pruning、quench recertification 各自的下游净收益。维护同时改变多个因素，NoMaintenance 也因此是一个整体维护消融，而不是各维护算子的因果拆分。

考虑 Stage 2 花费 `218.45`，当前 80 目标只省 `58.775`，所以 Hegel 全程总价比 NoMaintenance **多 `159.675`**（`2990.325` vs `2830.65`）。交接文档承认这一点；结论应保留，不能因为边际均值更低就称维护已经回本。若后续工作主张 Minton 的独立价值，只需在正常访问流里加入确会召回 lure 的合理任务或分别禁用单项维护，不必本轮扩展成大规模多因子研究。

完整逐行差额保存在 `receipts_analysis.json` 的 `maintenance_deltas_per_row`；`non_pi_maintenance_differences` 是空列表。

## 八、总体覆盖边界与指标解释

### 原始记录能支持的数字

| 臂 | 域内 strict pass | recall confirmation | 非直接 walk | mean composite |
|---|---:|---:|---:|---:|
| Hegel | 80/80 | 80 | 48 | 29.2200 |
| NoMaintenance | 80/80 | 64 | 32 | 29.9546875 |
| DEV | 80/80 | 64 | 0 | 38.8550 |
| Forward | 80/80 | 64 | 0 | 40.8075 |
| Reverse | 79/80 | 80 | 0 | 36.2950 |
| Alpha0 | 80/80 | 32 | 0 | 39.5200 |
| BM25 | 80/80 | 32 | 0 | 39.5200 |
| Frozen Pool | 80/80 | 0 | 0 | 40.3600 |
| Frozen Open | 80/80 | 0 | 0 | 46.3600 |

这些是提交数据的准确复述，不表示接受其中的通道计价、Frozen 起点或 freshness 协议。所有臂的 8 条 OOS 均 scoped unknown；Reverse 在 `seed=20281120, T17 Pi` 产生一条实际假确认，heldout RMSE `0.047025 > 0.045`，报告已经如实披露。`recall_confirmation_count` 是在线确认计数，不是事后正确预测计数，所以 Reverse 的 80 recall 与 79 strict pass 不矛盾。

### 需要保留的限制

- 四个 seed 是**一次 Stage 0/1/2 形成后的四组参数迁移**，不是四次独立学习链；这点交接文档已经明确。Stage 3 不把新目标的结果重新入林或更新效用，不能把它描述为 80 次持续自我改写形成循环。
- OOS 现在随 seed 改变频率、振幅、相位和跳变幅度，确实不再是同两条任务重复四次；但仍是两个固定函数构造族的参数变化。称“8 个异构 OOS”容易让读者误以为有 8 类不同失配机制。
- Pi 的新目标（M42 2190–2222）随机化的是传感器 scale/sign/offset，内部仍调用同一 `DEFAULT_PI_S_ODE_PARAMS`。这些目标验证观察映射迁移，不足以证明快慢时间尺度、耦合强度、闭包有效区间也发生了未知变化。
- W3 MurrayCap 虽然在 Stage 1 确实发现，但 Stage 3 的五家族是 W1、W2、W4、W5、Pi；没有 W3 跨域迁移家族。不能把五项新发现都解释成有独立 Stage 3 迁移覆盖。
- 新测试的 unified chain 只跑一个 seed，并断言所有 baseline 100% pass；四 seed 的 Reverse 已出现 1 条 false confirmation。这不是造假，而是现有单 seed 测试不能代表完整统计覆盖。

## 九、建议的有限修复顺序

1. **先统一执行事实与证据协议**：真 Stage 0 snapshot；用真实 channel acquisition 记录费用；确认耗尽时不重用训练点；重新核对 OpenRediscovery 的实际执行。后两项通道与 rediscovery 细节见主审报告。
2. **再恢复可解释的对照**：DEV 分层覆盖五家族；Alpha0 与 pooling 分开；所有费用包含对应实际检索/拟合操作。只复用已有 runner 和 historical interfaces，不另建独立评测系统。
3. **修正文稿强结论**：保留“结构关系能在此受控任务上改变候选访问顺序、减少确认浪费”；撤回“彻底击穿一切扁平顺序 / 唯有 graph”的无条件措辞；把全生命周期收益与 NoMaintenance 未回本一起呈现。
4. **随后才做一项有针对性的泛化扩展**：可以选择有噪 pilot、未见 cue、不同 ODE 参数或合理失败流中的一种，预先冻结强控制和接受规则。不要为满足审计把所有想得到的边界一次性全修，也不必放弃当前 forest 架构。

## 十、复现与产物

所有审计产物位于：`/workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/unified/`。

```bash
python /workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/unified/capture_shared_build.py
python /workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/unified/probe_unified_controls.py
python /workspace/scratch/3f14ad2da0ed/audit_m42_fb4b23aaf/unified/analyze_receipts.py
```

共享构建和 controls 需要当前固定 commit 的本地工作树保持存在，产物路径均指向审计目录。主要证据：

- `shared_built.pkl`：原始 builder 完整返回，可供其他专项复用。
- `true_stage0_snapshot.pkl`、`true_stage0_forest.json`：运行时真实 Stage 0 边界。
- `pre_stage2_forest.json`、`post_stage2_forest.json`、`shared_build_metadata.json`。
- `frozen_snapshot_control.json`：两目标、两起点对照，不改 evaluator。
- `posthoc_pilot_pair_control.json`：8 个 onset 的两候选 pilot 残差及独立确认/heldout。
- `confirmation_reuse_control.json`：原版 DEV T13 调用轨迹。
- `receipts_analysis.json`：全部 792 行归组、32 条复用确认、唯一假确认及 80 对维护差额。
- `history_open_grammar.json`：先前能力检索结果。

