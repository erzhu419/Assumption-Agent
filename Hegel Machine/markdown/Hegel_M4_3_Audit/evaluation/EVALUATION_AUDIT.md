# V17.5 M4.3：评测设计、维护归因与检索对照专项审计

审计固定提交：`3461535364fcdb4e9ae9c87ccba502b7e6b56fe8`，前轮基线 `fb4b23aaf802c20b4e2d0018436abab73b0cca06`。本专项只读源码、提交证据及内存副本；没有修改仓库源码、历史证据、远程仓库，也没有调用在线模型。

## 1. 结论先行

**M4.3 对上一轮评测设计的回应是实质性的。真 Stage0 冻结、真实 OpenRediscovery、完整 BestPilotRMSE、六家族 DEV、同森林关系编译表和 Alpha/Pooling 的正交配置都已进入真实执行。不能继续套用 M4.2 的“仅多记账”“裁掉既有 Stage0”“无强普通基线”的旧诊断。**

本轮仍须区分三件事：

1. 按提交的 CostMeter 账本，维护确实在既定流第 46 题首次回本，之后保持正值；其 **98.899% Stage3 毛节省来自 Pi 家族，而 Pi 节省实际来自八个 hydraulic adiabatic 目标**。这主要证明 OR 整合与证书关系使合法粗通道可达，并没有分别证明所有维护操作的独立经济收益。
2. 同信息编译表达成相同质量、相同物理测量费，并且名义全周期成本略低。它支持森林关系知识可以由不同检索算法消费，**尚不支持在线 PPR 的不可替代性或当前实现具有“免重编的动态更新”**。对后者，本专项有正式版本修订后的直接反例。
3. 噪声对所有臂一致，双轨质量也真实可比；但实际噪声是 Gaussian，DEV 没有施加该噪声，双轨附加比较只有一个 seed 的八道 onset。应按实际分布和样本规模解释。

下文所有费用首先是**提交账本的精确重算**。实际拟合、验证、RK4、索引构建等事件是否充分记账由主审独立处理；本专项不以算术重现替代成本完整性验收。

## 2. 覆盖与复用

- 阅读附件 Report 17、M4.3 全部新模块、M4.2 builder/候选池/维护边界与 Stage3 相关调用、HippoRAG/Compiled 索引实现、runner 归档机制，以及上一轮 `Hegel_M4_2_Audit/M4_2_AUDIT.md` 第 10 节及相关原控制。
- 按 `AGENTS.md` 查阅 reuse hub 和 historical catalogue；`ordinary` 查询返回 130 个历史版本，复用的现有入口包括 paid score history、checkpoint、冻结程序、候选比较与生命周期账本，记录见 `historical_reuse.json`。没有重建平行框架。
- 对提交 1,144 行回执逐行配对，键 `(seed, target_id, arm)` 全部唯一；11 臂各 104 行，确认重叠最大值 0。
- 直接加载提交的可重定位 Stage0、未维护 Stage1、维护后 Stage2 三份森林；只做授权/通道预检、实际粗细 oracle 对比和一次正式 scope revision 的索引控制，没有重建形成链。
- 唯一新增目标拟合是报告缺少逐例质量证据的噪声附加对照：两个噪声设置 × 八个 onset × Hegel/BestPilot 两臂，共 32 行。没有再跑 DEV、全研究或回归套件。

主输出：`evaluation_analysis.json`；脚本：`analyze_evaluation.py`。32 行质量记录及模型工作区位于 `fork_quality/`，对应脚本为 `replay_fork_quality.py`。

## 3. 可验收的设计修复

### 3.0 十一臂账本核对

下表均为96道域内题；各臂另外8道OOS均诚实 unknown、无假确认。C采用提交的 `M + 0.05K` 口径。

| 臂 | 通过 | 假确认 | 平均 M | 平均 K | 平均 C | 前置 C | 96题全周期 C |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hegel Unified | 96/96 | 0 | 31.3333 | 21.2150 | 32.3941 | 652.275 | 3762.1078 |
| Hegel Walk NoPool | 96/96 | 0 | 31.3333 | 21.2150 | 32.3941 | 652.275 | 3762.1078 |
| Compiled SameForest | 96/96 | 0 | 31.3333 | 19.8042 | 32.3235 | 654.093 | 3757.1530 |
| Hegel NoMaintenance | 96/96 | 0 | 36.2500 | 26.3508 | 37.5675 | 433.825 | 4040.3097 |
| Alpha0 PoolOn | 95/96 | 0 | 37.6250 | 37.7625 | 39.5131 | 652.275 | 4445.5373 |
| Alpha0 PoolOff | 95/96 | 0 | 37.6250 | 37.7625 | 39.5131 | 652.275 | 4445.5373 |
| Flat BM25 | 95/96 | 0 | 37.6250 | 38.1300 | 39.5315 | 652.275 | 4447.2990 |
| Ordinary StratifiedDev | 96/96 | 0 | 37.8333 | 47.5000 | 40.2083 | 901.375 | 4761.3750 |
| Ordinary BestPilotRMSE | 96/96 | 0 | 36.1875 | 107.5000 | 41.5625 | 433.825 | 4423.8250 |
| Stage0 OpenRediscovery | 93/96 | 1 | 42.1250 | 46.9588 | 44.4729 | 244.050 | 4513.4520 |
| Stage0 NoDiscovery | 16/96 | 0 | 34.2917 | 30.5317 | 35.8182 | 244.050 | 3682.6020 |

NoDiscovery 的账本最低但绝大部分题未解，因此不可用总价列单独排序系统效用。Hegel的96个成功也含9次全历史fallback，不能写成96次都仅凭图召回完成。

### 3.1 Frozen 与发现能力恢复了真实含义

M4.2 builder 在 Stage1 之前直接保存完整 Stage0（8 节点、2 coarse edges、1 wormhole）；M4.3 runner 复制这份 snapshot 给两个 Frozen 臂，而非重新挑选四个节点。OpenRediscovery 在失败后实际运行 `run_compositional_code_agent_episode`，包含诊断、组合候选和新确认；NoDiscovery 使用 Stage0 原子池。相关源码位置：

- `m4_2_unified_closure_and_pi_lifecycle.py:1194–1195`。
- `m4_3_channel_bound_lifecycle_and_audit_closure.py:1531–1606,1724–1726,1845–1862`。

官方 96 题结果也符合这两个能力的差别：OpenRediscovery 93/96 通过，NoDiscovery 16/96 通过。NoDiscovery 的 16 个成功全是 Pi；其余五个复合家族各 0/16。OpenRediscovery 三个失败都属于 capped Murray，其中两次诚实 unknown，一次确认通过但 heldout RMSE 0.049667 高于容差 0.045。不能把 OpenRediscovery 的较低质量隐藏为“严格同质量费用对照”，但 Hegel 在本套任务同时取得更高质量与更低账本费用是表内真实事实。

### 3.2 BestPilotRMSE 不再是假想对手

`_fit_best_pilot_rmse_across_specs` 对每个 spec 使用 `tolerance=0.0`，遍历角色排列，再按训练 RMSE 全局最小值选择；未用 heldout 选模型。失败确认保留于 paid prefix，后续确认选不重叠点。主研究里该臂 96/96 通过、0 假确认。位置：M4.3 `1061–1088,1456–1512,1827–1835`。

这关闭了上一轮“无噪 pilot 明明可判别，却用 first-acceptable 宣称信息论不可判别”的主要实验缺口。

### 3.3 同森林编译表与正交消融具有实际价值

Compiled 从同一森林的节点、wormhole、implication、coarse edges 构造两跳可达表，返回同类候选并进入同一授权、拟合、确认与 heldout 路径。它不是人工另给 target 答案。Alpha0 的 PoolOn/PoolOff 及 Walk 的 PoolOn/PoolOff 也分别设置了参数，消除了上轮同时切换两个因子的混淆。位置：M4.3 `519–625,1763–1817`。

本套任务里 PoolOn/Off 在相同 Alpha 下逐项结果相同，因而应报告 pooling 在此条件下的可检测边际效果为零。`NoPool` 指 wormhole equivalence pooling，不是删除全历史 fallback；四臂都仍有 fallback。

## 4. 维护 46 题回本：数值可重现，来源须精确

### 4.1 六族账本分解

| 家族 | 题数 | Hegel 平均 C | NoMaintenance 平均 C | 维护在 Stage3 省 C |
|---|---:|---:|---:|---:|
| HaldaneShunt | 16 | 28.7167 | 28.7193 | +0.0416 |
| OscShunt4P | 16 | 28.5607 | 28.5609 | +0.0040 |
| MurrayCapped | 16 | 28.5443 | 28.5467 | +0.0384 |
| Onset MonodShunt | 16 | 32.9310 | 33.2680 | +5.3920 |
| Onset MurrayShunt | 16 | 29.4123 | 29.4118 | −0.0081 |
| Pi mixed | 16 | 46.1995 | 76.8985 | +491.1840 |
| 合计 | 96 | 32.3941 | 37.5675 | +496.6519 |

非 Pi 共 +5.4679，说明上轮“所有非 Pi 完全没有差额”确实改变了，但这是很小的毛收益。其主要部分来自九个 Monod onset fallback 各少评估两个候选；其他几族主要只差很小的检索计算代理。不能把非 Pi 的 +5.4679 称为这部分维护独立收回 218.45 的全成本。

### 4.2 Pi 的 491.184 不是淬火盒缩小带来的收益

再分解 16 个 Pi：

| 子类 | 题数 | Hegel 路径 / 平均 C | NoMaintenance 路径 / 平均 C | 合计差额 |
|---|---:|---|---|---:|
| hydraulic adiabatic | 8 | coarse，21.6995 | micro fallback，83.0985 | +491.1920 |
| thermal quench | 8 | exact fine，70.6995 | exact fine，70.6985 | −0.0080 |

形成时 thermal meso 有自己的 ODE coarse edge；hydraulic duplicate 没有这条边。未维护状态下，thermal 节点有边但其 modality 不允许 hydraulic；hydraulic duplicate 有正确 modality，却没有自身活 coarse edge。Stage2 将 hydraulic/thermal 合并，并重认证与重绑，因此现在 hydraulic 目标可从合法节点选择粗通道。

本专项保持同一个 T21 的函数、参数与坐标，只改变 modality 做预检：

| 快照 | adiabatic/hydraulic | 同一 adiabatic/thermal | quench/thermal |
|---|---|---|---|
| 真 Stage0 | micro | coarse | micro |
| 未维护 Stage1 | micro | coarse | micro |
| 维护后 Stage2 | coarse | coarse | micro |

这是**合法授权可达性变化**，不是按实验臂标签虚造半价。实际 `MultiScaleWorkspaceTaskLab.acquire_channel` 调用也确认粗细通道返回不同数值，单位费分别 3 与 10，旧的“同一 oracle 套两种价格”诊断不再适用。

但所有 adiabatic 目标都被设为 hydraulic，所有 quench 都被设为 thermal（M4.3 `945–1028`）。而 quench pilot 从 T=0.20 起，已经同时越出 Stage0 的旧盒及 Stage2 的新盒，未形成“旧盒会放行、新盒会拒绝”的主研究配对。因此当前主表证明 OR 整合/边证据连接在此访问分布上有价值；它不分别估计淬火收盒、Minton、Monod scope 修订的独立回报。那些安全或语义修复可以由自己的控制验收，无需每项都强求盈利。

相关源码：M4.2 `1088–1195,1686–1740,1814–1894`；M4.3 `1139–1175,2000–2041`。完整授权 reason 与 live children 见 JSON `channel_precheck_controls`。

### 4.3 46 是这条固定流的条件性结果

每 seed 按六个家族分块，Pi 排最后；高收益 adiabatic 在整个流的位置为 `21,22,45,46,69,70,93,94`。第 46 题正好是第四个高收益 adiabatic 到来，累计差额跨过 218.45 的维护前置费用。按提交账本，之后净值始终为正，最终 +278.2019，这个算术没有问题。

准确表述应为“在这条固定家族配比与顺序的访问流、当前计量代理下，第 46 个目标回本”。不能把 46 当作不随领域分布、粗细价格比和序列改变的系统常数。也无需为了这一限定再跑任意随机重排或盲目扩大世界库。

## 5. PPR 与编译表：真正得到的结论与一个实际更新缺口

### 5.1 同信息控制应当保留

两臂均 96/96 通过、0 假确认、平均测量费 31.3333、87 次 recall、31 次非直接关系命中。Compiled 的平均账本 C 为 32.3235，PPR 为 32.3941；包含额外 1.818 编译费后，Compiled 的全周期总价 3757.1530 仍略低于 PPR 的 3762.1078，净优势 **4.9548**。PPR 平均候选数更少（2.5938 vs 2.7604），但不足以使其总账本更便宜。

这很有价值：它把“关系知识有用”从“必须在线用 PPR 才能使用知识”中分离出来。当前收益可以通过结构化关系 + 合约 + 合适候选排序实现，不能再把普通 first-acceptable、BestPilot 或 BM25 的不足都归结为哲学表示不可替代。与此同时，Compiled 也确实保留森林，结果并不要求放弃森林结构。

31 次 `non_direct_walk_win` 在 Compiled 上同样出现，因为字段记录的是候选没有直接签名命中，而非在线是否执行了 PageRank；其中包括 8 次 Pi 关系访问，不应把 31 都描述为“两种 onset 的两跳 PPR 成功”。

### 5.2 “免重编的动态多线索扩展”尚未实现

报告第 50、105 行把动态更新优势表述为当前能力。但 `HippoRAGWormholeIndex.__init__` 深拷贝 active nodes、构造签名倒排表和 transitions；`retrieve` 只读取这些缓存，没有对 live forest 的版本刷新或失效检查。Compiled 也使用构建时快照。

本专项对提交森林调用正式 `empirical_revise_node`，在 W4 的既有条件上添加一个新 scope 条件，使 live 节点 v1→v2。使用原有索引查询同一直接签名，结果为：

| 查询对象 | 返回版本 | 是否含新增 scope |
|---|---:|---|
| 原 PPR index | v1 | 否 |
| 新建 PPR index | v2 | 是 |
| 原 Compiled table | v1 | 否 |
| 新编译 table | v2 | 是 |

因此当前 PPR 同样需要重建或显式更新。主 Stage3 不修改森林，这个缺口**没有使现有 1,144 行结果失效**；但它使“当前无需随局部修订更新索引”的声明不成立，并且在 M5 真的把新学习写回同一活图时，会成为旧权限/旧 scope 被继续返回的集成风险。

最小处理是先更正文案；开始动态流时，为索引加 forest revision 检查与失效重建/局部更新，复用现有版本机制。小图阶段直接重建并记账即可，不必现在发明复杂增量 PageRank。源码位置：`hipporag_wormhole_index.py:161–192,277–299,340–474`；控制见 JSON `dynamic_index_control`。

该索引相对 M4.2 的完整 diff 只有第406行：从只读 `target_scale_regime` 改为兼容 `target_scale_regime or scale_regime`。这是尺度字段接线修复；没有新增动态刷新，也没有修改 PageRank 迭代算法或费用常数。M4.3 主研究原本传递的就是 `target_scale_regime`，不应把主表改善归因于一个新 PPR 算法。

## 6. 噪声对照：同样本公平成立，分布/DEV/样本口径应修正

### 6.1 两臂确实看到同一 pilot 噪声

每次 evaluator 按 `target.pilot_noise_seed` 重新创建 RNG，只有 pilot 阶段加噪，因此同一目标各臂取得相同的六个带噪 pilot。追加 stress/confirm 是相同洁净 oracle 机制，点集由各自策略决定，这种信息差来自策略而非暗中偏置。未发现对 Hegel 提供无噪 pilot、对普通臂提供有噪 pilot 的问题。

但调用是 `random.gauss(0.0, pilot_noise_std)`（M4.3 `1177–1195`），没有 clipping/truncation，是无界高斯分布。σ=0.002 为容差 0.062 的约 3.23%，不等于每个噪声实现值都低于容差 3.3%。主 32 个 onset × 6 点的 192 个样本里，最大绝对值为 **0.0052912692**，约为容差 **8.53%**。用“高斯传感器噪声，标准差为容差约 3.2%”即可准确修复文案，不必为了“有界”这个词改变既有实验。

### 6.2 补核双轨质量结果为正

官方附加比较只运行 `seeds[0]` 的 8 个 onset，摘要只保存均值费用，未保存成功率/假确认或逐例回执（M4.3 `1899–1948`）。本专项复用提交森林和原历史 AST，补出32行：

| σ | 臂 | 通过 | 假确认 | 平均测量费 | 平均 C |
|---|---|---:|---:|---:|---:|
| 0 | Hegel | 8/8 | 0 | 28.0 | 29.1623 |
| 0 | BestPilot | 8/8 | 0 | 28.0 | 32.9250 |
| .002 | Hegel | 8/8 | 0 | 29.5 | 31.6748 |
| .002 | BestPilot | 8/8 | 0 | 30.0 | 37.0250 |

所有格子的费用精确重现官方，确认重叠均 0。故此次双轨成本比较没有发生“便宜但质量变差”的隐蔽替换，应当保留为正证据。

同时，有噪时 Hegel 也比无噪多付了测量：29.5 vs 28，而非免疫噪声。相对 BestPilot 的测量优势是平均 0.5/题，8 题共少 4 个费用单位，即少一次单通道测量；更大的总差仍主要来自计算代理。一个 seed 的 8 题不足以建立普遍噪声相变或信息论下界。本轮不需要再加噪声扫频；把当前有限比较说清楚即可。

### 6.3 DEV 覆盖修复成立，但并非 TEST 的同分布噪声版本

DEV 已按六族选 12 题，修掉原 `[:10]` 漏族的问题；没有读取四个 TEST seed 的分数。但 M4.3 `674–681` 虽构造 `onset_pilot_noise_std=.002` 的目标，实际观察行却直接调用 `dt.oracle_fn(pt)`，未加噪。索引 `(20,21)` 对 Pi 只选 adiabatic，没有 quench。排序仍按每个候选命中频次而非完整顺序的实际期望成本。

这些事实不能直接证明冻结顺序一定变坏，本专项没有另外重调 DEV；应把它准确称为“六家族无噪参数 DEV 的命中频次顺序”。若下一轮需要发布更强的“同分布 DEV 成本优化”结论，复用主 evaluator 的 acquisition 口径，并在 Pi 两个子类中各取一例即可。避免另建一套看似相同却跳过噪声的采样逻辑。

## 7. 比较与泛化口径

“AllHistory” 实际是 12 个按时间排列的 AST spec 池，再对多尺度目标加 fast-transient spec。它没有获得森林的 cue→关系检索，也不持有活证书通道授权路径；exact fine 节点走的是宿主已有 ODE 参数求解分支，而普通历史臂会拟合 fast-transient spec。因此 Hegel vs Ordinary 是不同知识组织与执行支持的系统比较，不能仅由总差推出 PPR 这个检索算法的独立效果。Compiled 同森林对照是本轮更严格、也最有信息量的分离。

四个种子来自一次 builder 的同一份 post-Stage2，随后每 seed deepcopy；S3 没有把目标结果学习入林。它们是四组未见参数迁移，并非四条独立形成/维护链。Pi 内部 ODE 参数始终使用 `DEFAULT_PI_S_ODE_PARAMS`，变化的是读出 scale/sign/offset 与预定时域；六族同样是固定机制模板的参数变化。OOS 有八组参数变体，但仍来自同一不连续模板。位置：M4.3 `737–1056,1712–1897`。

这些是当前受控验证的适用范围，不构成要求本轮重测全部181历史版本或全部参考文献的理由，也不否定已有工程收益。它们意味着“开放世界持续 RSI 的一般性成立”仍应留给后继证据。

## 8. 建议只做有限闭环

1. **保留已经真实改好的评测结构。** 不撤回真 Frozen、真实 CodeAgent、BestPilot、Compiled、2×2 消融和六家族主表。
2. **先把主审发现的实际账项完整性修好，再冻结最终 ROI 数字。** 本专项只确认原账本 46/+278.2019 的算术和来源。
3. **纠正三处结论口径。** 维护收益主要来自 hydraulic adiabatic 合法粗通道；噪声为 Gaussian；当前 PPR 不是自动更新的动态图索引。
4. **动态索引更新是进入 M5 前的直接集成任务。** 用现有 version/digest 做失效检查，更新前拒绝旧 scope 的结果，必要时重建并计费；无需增加新理论框架。
5. **DEV 采样与 fork 回执尽量共用既有接口。** 保存现已补出的逐例质量字段，下一轮需要更强分布结论时再对齐 DEV 噪声和 Pi 子型，不必为修辞追加庞大实验。

本专项没有发现理由推翻整个 M4.3 路线。应保留的主判断是：关系与契约确实进入候选执行并有受控收益，许多旧对照缺口已真实修复；更强的动态更新、独立维护收益和一般 RSI 主张仍须按各自证据边界表述。

## 9. 实际阅读清单与证据索引

本专项的“完整阅读”只对下列标明全文的文件作承诺，不冒充逐字重读整个181版本历史。主审另有全变更与其他专项覆盖。

| 文件/对象 | 本专项实际覆盖 |
|---|---|
| `AGENTS.md` | 全文；历史复用与费用/归档约束 |
| 附件与提交内 Report 17 | 全文；两份标题对应同一工作总结 |
| `m4_3_channel_bound_lifecycle_and_audit_closure.py` | 全文2076行；重点497–2075评测/构建/总结 |
| `hipporag_wormhole_index.py` | 全文635行及前后提交精确diff |
| `run_v17_5_m4_3_unified_study.py` | 全文；重定位加载和回执归档 |
| `m4_2_unified_closure_and_pi_lifecycle.py` | 750–885、1010–1205形成边界/节点，1645–1735、1880–1975维护/历史池/账本；相关其余声明与调用定向搜索 |
| `wormhole_forest.py` | `empirical_revise_node`正式版本修订与失效逻辑；本专项并未全文复核该大文件 |
| `Hegel_M4_2_Audit/M4_2_AUDIT.md` | 第1–5节与8–13节，重点10.1–10.5评测诊断及历史对照 |
| reuse hub / capability catalogue | 定向入口与ordinary查询，不声称全文重读巨大汇编；输出historical_reuse.json |
| 11臂提交回执 | 全部1144行程序化配对核验，1144个唯一键 |
| Part2 summary / 三份forest snapshots | 相关统计、节点、scope、边与归档引用；无改写 |

可复现命令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 analyze_evaluation.py --repo '<checkout>/Hegel Machine' --out '<audit>/evaluation'
PYTHONDONTWRITEBYTECODE=1 python3 replay_fork_quality.py --repo '<checkout>/Hegel Machine' --out '<audit>/evaluation/fork_quality'
```

两个脚本也接受可选环境变量 `HEGEL_AUDIT_REPO`（指向checkout内的 `Hegel Machine`）与 `HEGEL_AUDIT_OUT`（指向审计输出根目录）；省略命令行路径时，分别写入根目录下的 `evaluation/` 和 `evaluation/fork_quality/`。两种环境变量均未设置时默认本次工作区路径。

第二个脚本应使用新的输出目录，逐例记录采用append写入以保留运行进度；不要原地重复执行后混算回执行数。
