# Hegel Machine M2 审计：已成立的结果、实际缺陷与最小修复合同

审计日期：2026-11-19。固定提交：`9115f9adf9dad1f52f106a2ab14babe8b68243dd`，分支 `codex/hegel-cognitive-mainline`。本文只针对该快照，不随分支后续变化更新。审计没有修改或推送项目代码，没有进行付费模型调用。便携复现脚本、原结果重算、完整重放和诊断控制随证据包提供。

## 1. 结论先行

**M2 的真实工作区、真实 Python 子进程和有限机制族跨域迁移已经成立；最新项目合同要求的开放作者接入尚未完成。原始结果的算术基本可信，但当前结果不足以证明对充分使用同一历史的普通方法具有净优势，也不足以把收益独立归因于横边或两跳图结构。**

这不要求推翻森林、撤掉多精度对象，也不要求先把 V166 审计提出的所有远期风险修完。当前最值得处理的事情可以分成三类：

1. **直接正确性缺陷**：确认失败仍提交 `completed`；开放 Python 的 `fit` 可以看到刚购得的确认标签；receipt 缺少目标、版本和被测模型绑定。
2. **影响研究结论的比较缺陷**：普通基线过度压缩已验证历史；NoHorizontal 同时禁掉跨域内容使用；Flat 的总差额被全部解释成两跳图收益。
3. **实现与主张不一致**：预写十类公式族被称作开放合成；AST 旁注与实际程序未绑定；近失例修订和两跳误差仍有预设分支；`program_steps` 主要是 schema 路由元数据。

最有判别力的实测结果如下。所有成本差均为 **Hegel − 对照**，负值表示 Hegel 更省；均值单位是一个含十道目标的评测簇，不能误读成单题。

| 项目 | 原结果 | 本轮诊断控制或复核 | 含义 |
|---|---:|---:|---|
| 原冻结套件质量 | 6 臂 × 40 目标全部通过 | 完整重放 240/240，0 假确认 | 当前有限题集的成功应保留 |
| Hegel 对普通基线，目标复合成本 | −41.025 | 普通基线使用全部已验证同维历史后 **+11.55625** | 原净优势依赖历史消费策略 |
| Hegel 对普通基线，含源期成本 | −27.425 | 同历史消费控制后 **+25.15625** | 不是已经确立的强普通净胜 |
| Hegel 对无横边，目标复合成本 | −100.61875 | 无横边但允许同内容 local 候选后 **−1.96875** | 约 98% 差额来自跨域使用资格 |
| Hegel 对 Flat，目标复合成本 | −1.74375 | 其中两跳 C4 仅 **−0.475** | 总差额不能全归为两跳 |
| 新任务确认失败后的状态 | 报告强调零假确认 | `5 + sin(u)` 确认 RMSE 1.289795，仍 `completed` | 当前接口不保证未确认时诚实退出 |

上述额外控制是**事后诊断**，没有被包装成新的冻结基准。它们不证明普通方法普遍优于 Hegel；它们足以否定“原实验已经排除了普通同历史方法”“全部差额由图结构造成”的强归因。

## 2. 对项目目标和当前里程碑的理解

### 2.1 哲学主张在工程上需要怎样的证据

初始五篇长文包含多轮提案、反驳与修正，不能把其中某一段模型建议当作唯一规范。结合用户在《黑格尔机》中的明确要求及随后主计划，项目的可检验目标是：过去的经历形成可执行概念；新反例定位旧概念的边界；修订保留旧经验中仍成立的部分；新的概念、条件、表示或研究程序改变后继任务的搜索和实验，并且相对于会学习的普通方法产生可核算的效用。

这支持局部条件森林与范围内支持，支持把观测协议、尺度与作用域分别表示，也支持跨领域候选映射。它不要求每次修订都发明全新数学，不要求从无先验开始；但“从已列出的十类完整公式中选一类并拟合参数”和“扩展假设语言”需要分别记账。保存知识、复用知识、创造新知识、改善创造知识的方法，也需要分别给证据。

黑格尔原典可以解释项目的思想来源，不能替任何具体工程算法提供经验有效性的保证。这里采用的是一个值得实验的研究纲领，而不是把哲学前提直接当作性能结论。

### 2.2 有限语法合理，M2 合同仍未全部完成

V166 的最小修复合同明确允许通用原语、有限函数族和合成切片。这轮不能因为这些合理简化就宣布失败。

但紧邻当前版本的 V16.7 报告第96行，明确把下一步 M2 定义为：接入开放 Python 工作区与在线 LLM，让候选 `ClaimContract / ObservationMap / ScaleMap / program_steps` 从真实实验残差中自主提议并经反例验证，替代固定插件枚举。V16.8 报告则把阶段定位收缩成真实文件系统、AST 摘要与子进程评测，并在结尾宣布 M0/M1/M2 全部打通。

因此准确状态应为：**M2 的工作区、评测及有限机制族迁移部分完成；开放作者与残差驱动合同修订的接入未完成。** 这由项目自己最新的相邻合同确定，无需额外拔高目标。

来源：[V16.7 的 M2 定义](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/09_V16_7_RedTeam_Response_and_Upgraded_Results.md#L94-L97)；[V16.8 阶段定位与完成声明](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/10_V16_8_M2_Open_Workspace_AST_Wormholes_Report.md#L1-L64)；[V166 最小修复与实验合同](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/markdown/Hegel_V166_diagnosis/02_下一轮最小修复与实验合同.md)。

## 3. 这轮修复与 M2 中已经成立的部分

本轮针对 M0/M1/M2 的三个测试文件共13项测试通过。重新运行 V166 的语义控制12例、boundary 控制4例后，相关旧反例的修复成立：字面内容不再被错误清除；问题版本、过期边、授权边界、AND/OR 支持、UNKNOWN 和执行异常的相关语义已经改善。该结论只覆盖实际检查的控制，不代表每一条历史审计事项都已关闭。

M2 本身的进展也是真实的：

- 观测写入真实工作区，`model.py` 真实生成，拟合与预测真实启动 Python 子进程。
- 源任务和 near-miss 的失败不会一律变成 supported 来源。eval03 的楞次源失败被保留，并从有资格的形成/先验中排除。
- 目标会购测新点，复用不是仅靠源支持就自动通过；失败退出后沿用已付观测，没有清空后重复收费。
- `O_strong` 已实际读取原始历史。旧版“普通组完全不读历史”的批评不能直接照搬。
- 未发现求解器直接读取隐藏目标参数或 heldout 答案来拟合；合成任务由同一实验框架生成，不应仅因这一点被称作答案泄露。
- 原142条源记录、240条目标记录、400条成本记录及12项配对比较重算，没有发现成本算术差错；manifest 列出的四份代码 hash 与该提交一致。
- 原研究脚本完整重放，六臂共240次目标全部严格通过，0假确认。九项非 HExA 比较逐值重现；HExA 三项数值受并列选择的 hash 顺序影响，但方向未反转，见第8节。

证据：包内 `results/receipt_recomputation.json`、`results/replay/`、`m0_recheck.json`、`boundary_recheck.json` 和 provenance；[原始配对结果](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/docs/evidence/v16_8_m2_open_workspace_wormhole_20261119/paired_results.json)。包内目录以 README 的最终索引为准。

## 4. 需要先修的正确性问题

### 4.1 确认失败，仍把候选提交为 completed

`solve_open_workspace_direct()` 在 fresh confirmation 失败后，会再拟合一次全模板库。只要该次脚本没有崩溃，就返回 `status='completed'`；没有新的通过证据。运行成功和科学确认被混为一个状态。

使用相同公开 lab API，给一个模板库外但普通的任务 `y = 5 + sin(u)`，`u∈[1,10]`，容差0.01，得到：

| 检查 | 实测 |
|---|---:|
| fresh confirmation RMSE | 1.2897950867 |
| 返回状态 | completed |
| 31点外评 RMSE | 1.091451 |
| 外评 strict_pass / false_confirmation | false / true |

这里不要求当前有限库必须会解正弦函数。要求只是：**已经观察到确认失败，就不能把没有新证据的模型提交为已完成。** 当前外评器把这种提交计作 false confirmation，因此不能事后把 completed 解释成“只是流程结束”来回避报告的验收口径。

最小修复：返回未确认候选或 `scoped_unknown`；保留已有付费数据，有预算时继续研究。若确认后重拟合产生新模型，旧确认不能自动授权新源码，应冻结已确认模型，或为新模型执行独立确认。另需明确确认阈值为何使用 `tol * 1.2`，使它与声明合同一致。

来源：[失败后无条件 completed](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L789-L825)；包内 `mechanism/runtime_probe_result.json`。

### 4.2 确认数据会进入开放 Python 的 fit

确认点通过 `lab.acquire()` 获得后，标签已经追加到 `lab.observations`。`validate_workspace_model_on_rows()` 又把整个 `lab.observations` 传给评测器作为训练数据。评测器只要看到可调用的 `fit` 就会执行；`require_fit=False` 表示不强制必须提供 fit，不表示禁止调用 fit。

合法的记忆型 `fit/predict` 足以显示问题：先训练 `u=1,2`，然后确认 `u=3,y=9`；模型只记住训练表，陌生输入返回0。

- 经过当前 M2 wrapper，第三行标签进入 fit，确认 RMSE **0**。
- 使用同一个评测器，但只给确认前训练前缀，预测0，确认 RMSE **9**。

**当前十类生成模型的 fit 为空操作，所以不能据此指控原240次结果发生该种泄漏。** 这是接入真正开放 Python 前必须处理的生命周期缺陷。

最小修复：购测确认点前冻结训练前缀、代码和参数；验证器只使用这个冻结上下文，确认标签单独保存。随后若允许训练吸收确认数据，必须产生新的版本身份并重新处理其确认状态。

来源：[M2 验证包装器](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L519-L553)；[评测器的 fit 调用](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/code_agent.py#L600-L612)；包内 `mechanism/confirmation_training_leak_result.json`。

### 4.3 receipt 需要绑定实际被确认的对象

当前 `_validate_evidence_receipt()` 主要检查 passed、claim_id 和有字段时的 observed_error。它没有核对目标身份、节点版本、模型内容 hash、确认数据身份；缺失误差会通过，NaN 也会通过普通大小比较。

focused controls 中，旧目标/旧节点版本的 receipt、缺误差的 receipt、NaN 误差的 receipt 都取得 `local_solver_guidance`。原 M2 主循环每次现场产生正常有限误差，未发现原题利用这些缺陷；问题在于跨任务、跨版本复用的接口保证尚未闭合。

最小修复是添加身份核对和有限值检查，并保留正常 receipt 的正向控制，不需要再设计一层新治理框架。

来源：[receipt 检验](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L155-L170)；包内 runtime probe。

## 5. 普通基线与图消融：最影响结论的部分

### 5.1 O_strong 的问题是已有历史被压缩得过弱

`_reconstruct_prior_by_arity()` 会读取成功历史，但每个 active arity 仅保留一个 schema，且要求至少两个不同源领域支持。于是两个不同的2D机制竞争同一个槽；来自单个 near-miss 的3D memory 永远得不到先验资格。`shared/` 的文件被复制，但求解器未 import 或执行它们。

普通方法当然应在目标上重新确认。问题是“是否允许作为候选去试”被提前过度限制，这与“不能未经确认就当定律使用”不是同一件事。

诊断控制只将历史消费改为：同 arity 的**全部已验证 schema**均可作为候选，包含单源 near-miss。继续使用原合成器、原新点确认、原外评器和同一份源历史；每个额外候选另加10 compute，保留原26 compute的重建费，没有免费增加搜索能力。

| Hegel − 普通同历史控制 | eval01 | eval02 | eval03 | eval04 | 均值/十题簇 |
|---|---:|---:|---:|---:|---:|
| 目标测量费 | +15 | +15 | +30 | +15 | **+18.75** |
| 目标复合成本 | +7.775 | +7.775 | +22.900 | +7.775 | **+11.55625** |
| 源期＋目标成本 | +21.375 | +21.375 | +36.500 | +21.375 | **+25.15625** |

控制40/40通过，四簇都是普通控制更省。可以解释为小候选库里多做候选拟合减少了购测，在当前费用权重下合算；不能把它外推为普通方法在所有规模都更强。但它足以说明，原 `−41.025` 不能作为已经战胜充分同历史消费的普通方法的证据。

HExA 对照另有相似问题：`mistake_bank` 写入 near-miss，却没有在目标求解中读取。原论文的条件技能、错误经验和跨任务重定向不能由一个 arity 多数模板充分代表。可以把现实现准确称作 HExA 启发的简化基线；不应据此宣称击败忠实 HExA。论文采用 static injection，因此本审计没有要求它必须在每个评测 seed 内在线更新。

来源：[普通组历史消费](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L868-L905)；[HExA 的写入和目标入口](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L919-L967)；[HExA 原文](https://arxiv.org/html/2606.29315v2)；包内 `results/controls_results.json` 和便携控制脚本。

### 5.2 NoHorizontal 同时去掉了历史使用资格

关闭横边后，森林仍可返回 local node；但 `_filter_cross_domain()` 随后删除所有源领域与目标领域不同的 local node。由于目标使用新领域名，该臂实际上失去大量历史候选。

控制仍关闭全部横边，只允许同内容 local node 作为跨域候选进入原求解路径。模型、Stage1 gate、购测确认等均沿用原实现：

| Hegel − 对照 | 原 NoHorizontal | 无横边但保留 local 内容 |
|---|---:|---:|
| 目标复合成本/簇 | −100.61875 | **−1.96875** |
| 目标测量费/簇 | −96.25 | **−1.25** |
| 对照通过数 | 40/40 | 40/40 |

约98%的原差额消失。原对照可以保留为“跨域历史复用整体功能开关”的比较；要识别图表示的独立收益，正式消融还应把边上的适用条件和映射信息保存在等内容的非图记录中，只改变检索/组合方式。

来源：[过滤位置](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L1412-L1417)。

### 5.3 Flat 的总差额并不等于两跳差额

原报告把 `Hegel − Flat = −1.74375` 标作两跳非传递与图结构纯消融。逐题分解是：

| 来源 | 均值贡献/十题簇 |
|---|---:|
| C4 两跳任务 | **−0.475** |
| A3 双模蠕变任务 | **−1.44375** |
| 其余八题合计 | +0.175 |
| 总计 | −1.74375 |

C4 两臂测量费都为35；Hegel compute为53.5，Flat为63。收益是少一次受限拟合、扣除额外检索费后的代理计算差额，不能描述成节省目标实验测量，也不能让 A3 差额承担两跳证据。

Flat 确实复制却不调用 Stage1 gate。审计把同一个 gate 接回后，当前四簇数字**完全不变**，因为已有门控结果并未改变这些样本的准入。因此这是正式对照需要统一的设计差异，但不是本次差额已被证明的原因。

包内 `results/receipt_recomputation.json` 给出逐题分解，`results/controls_results.json` 给出加回 gate 的控制。

## 6. 当前 M2 实际运行的机制，与报告中的名称

### 6.1 工作区是真的，开放作者尚未接入

`_WORKSPACE_SYNTHESIZER_SCRIPT` 每次被原样写入工作区，枚举十类预写 schema，使用数值拟合填入常数，再生成模型文件。十类分别为：`linear_1d`、`rational_1d`、`inv_quad_crest_1d`、`congestion_rollover_1d`、`conjugate_relaxation_reset`、`conjugate_relaxation_state_memory`、`bimodal_relaxation_2mode`、`liebig_min_2d`、`liebig_capped_min_2d`、`cobb_douglas_sqrt_2d`。

`allowed_schemas=None` 表示搜索全部这十类，不表示任意 Python 或开放算子组合。M2 引用了 `evaluate_python_model`，没有调用 `CodeAgent.run` 或 `CodeResearchSession` 驱动在线作者。源 Python 模块的存档也没有成为目标真实执行的共享机制；目标主要消费 schema 名称后重新拟合。

这条路径作为受控模板基线很有用。需要调整的是它的证据层级：生成 Python 文件证明可执行产物接线；不能仅凭文件形式证明假设空间由经验扩展。

来源：[固定合成器](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L132-L452)。

### 6.2 AST digest 目前只能作候选召回线索

规范化器优先读取 `_CORE_LAW_EXPR` 字符串；没有该旁注时，只提取 `predict` 的直接 return。它没有追踪中间变量定义和辅助函数，也没有验证旁注与实际执行代码的一致性。同时所有数字、包括决定机制形态的指数及其负号，都可被归一化为 NUM。

原函数上的三个控制均发生 digest 碰撞：

| 程序变化 | 当前结果 | 不能由该结果推出什么 |
|---|---|---|
| 相同旁注，一个 predict 返回 x，另一个返回100000 | 同 digest | 执行行为相同 |
| 无旁注，latent分别为有理式与 `x**5+y`，外层return相同 | 同 digest，角色为空 | 已从程序中提取真实机制或剥离透镜 |
| `x**2` 与 `x**-2` | 同 digest | 只是不同参数实例 |

当前生成器自己写出正确旁注，所以这些控制不直接否定原40题的拟合成功。它们否定的是“规范化器已经支持一般开放代码语义归纳”的结论。

最小修复不需要判定任意程序等价：限定可分析的纯函数子集，追踪已支持的数据流，把可拟合系数与结构常量分开，保留符号、指数和定义域，存储实际程序 hash；超出子集则返回骨架未知。AST 匹配可以提议跨域候选，候选的角色、观测协议、尺度和目标证据仍须独立核验。

来源：[AST 实现](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L58-L126)；包内 `mechanism/ast_probe_result.json`。

### 6.3 角色对齐仍依赖字段顺序

两维 pilot/signature 默认第一个 active feature 是驱动、第二个是时间；三维默认第三个代表 memory，随后把这个顺序作为 forced roles。已有合成器能枚举角色，但源代码实际找到的 role_order 没有在相应形成入口被完整采用。

同一个 T_M2A1，只逆序 `feature_names`，不改变变量名字、数值边界、模拟规律和外评题：

| 条件 | 测量费 | RMSE | 结果 |
|---|---:|---:|---|
| 原字段顺序 | 30 | 0.017834 | pass |
| 字段逆序 | 55 | 0.049067 | fail，容差0.045 |

这是默认 gate 的 focused probe，不是原冻结协议的重新汇总；不能拿这里30与原报告35混比。但它确证当前“坐标无关”表述过强。应复用已有角色枚举、对候选角色做可区分探针，再验证绑定；不必先解决任意维度的全自动符号科学发现。

来源：[pilot 与 signature](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L586-L712)；包内 runtime probe。

### 6.4 近失例路径选择了已有修订，而非发现一般新状态

`form_and_revise_in_workspaces()` 先把 near-miss 作为独立问题交给十类合成器，再按 schema 名进入两个预写分支：memory 分支写入固定排除条件和带记忆节点；capped 分支写入固定 high-corner 条件和新节点。第三个公开变量已经存在，对3个及以上 active features 直接设置 `unreset_third_state_active=True`。

这证明已有状态扩展模板能被选择并保存。它尚未给出完整的“旧模型在 witness 上预测失败 → 确定失败前提/状态闭合缺口 → 提议并比较修订 → 后继验证”的证据链。当前目标遇到新失败主要回退，未把该目标的修订写回森林形成新的连续研究闭环。

ALIGN 对项目很相关的部分是近失例检索、包含/排除条件及后来正例对过强条件的修正；CEGAR 原文则在明确的模型与抽象语义下先判断抽象反例是否可实现，再精化。M2 可以采用它们的思想，但需要自己的实际 witness 与可执行改稿，不能从名称继承论文的保证。这里不要求把整个形式化模型检测器搬进来。

来源：[near-miss 修订](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L1195-L1363)；[ALIGN 原文](https://www.qrg.northwestern.edu/papers/Files/QRG_Dist_Files/QRG_2015/McLure-Friedman-Forbus-AAAI15.pdf)；[CEGAR 原文](https://cse.usf.edu/~haozheng/lib/verification/abstraction/cegar-smc.pdf)。

### 6.5 两跳目前是预置组合检查

初始横边误差预算写死为0.015；rational 分支额外创建一个 relay 节点，未由独立 relay 实验支持就标记 supported，再写入第二跳预算0.018、Lipschitz=1。目标公开 `two_hop_bridge_only=True`，据此触发累计误差检查。没有执行两次实际程序运输、从各自证据估计误差，再由检索自主发现这条路径。

`0.015 + 0.018 = 0.033 > 0.025` 是正确的保守界检查。但是**上界大于容差只能说明该界不足以认证，不能推出实际误差必定超过容差**。选择不试这条路可以是一种合理成本决策；报告中“据此证明试探无效”的因果语言则过强。

Flat 被实现成忽略完整链界的可传递类。这个控制能够显示“保留并消费组合约束”有用，却不能证明“只能用图才能保留该约束”。正式等内容 Flat 完全可以保存端点、路径条件、误差预算和失败记录；应比较的是在相同信息下检索与组合是否更省、更准。

来源：[两跳节点和预算](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L1137-L1193)；[目标路径提示](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_labs.py#L724-L740)；[消费位置](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L1536-L1555)。

### 6.6 program_steps 的执行能力被高估

目标入口将 `program_steps` 转为 op→dict，主要读取 `synthesize_scoped_python_schema.schema_tag`。calibration 和 confirmation/refit 的执行顺序仍写在主机固定流程里。

控制中删除节点声明的 calibration、confirm/refit 步骤，只保留 schema 步骤，结果、动作、每项成本及预测均不变。说明当前保存的是固定执行器的部分配置，而非完整的可学习研究程序。有限执行协议可以保留；若声称学习了方法程序，就应让方法体的合法改动改变动作，并在独立后继任务比较其效用。

来源：[program_steps 消费](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L1570-L1638)；包内 runtime probe。

## 7. 成本、摊销与证据范围

### 7.1 原费用是代理计价，可以保留但要正确命名

原复合费用为 `C = measurement + 0.05 × compute`。restricted synthesis 固定收10 compute，全模板搜索收24。每次合成还把 `python_execution_seconds` 固定增加0.02秒；验证器部分记录的则是真实 subprocess wall time。这个汇总字段因此不是完整实测执行时间。

代理计价不是错误。应分别保留测量费、逻辑 compute、真实墙钟、实际模型调用/token 等账本，不把代理费用下降直接说成真实运行时间提速。项目已经有公共计时设施，本轮研究脚本也记录 solver wall time，无需重造计时框架。

来源：[合成计价与占位计时](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L455-L515)。

### 7.2 给定 gate 的成本结论，不能自动当作全生命周期结论

原 `Hegel − O_strong = −27.425/簇` 确实包含各臂源期与目标期，且明确限定为给定 gate。因此不应指控它算错或隐瞒。开发簇形成与配对校准另花1884.35；如果把这笔一次性成本只向 Hegel 计入并摊到当前四簇，结果是：

`−27.425 + 1884.35 / 4 = +443.6625/簇`。

这是一个明确的分摊情景，并非唯一合理部署口径。将开发设施共用、只形成一次源库、延长目标流，都会改变账本。需要选定部署情景再画实际前缀累计曲线，不应把一次有限实验里的目标节省直接当作长期净收益。

双源相对单源也一样：多付源费 `390 − 250.825 = 139.175`，十个目标省64.19375，因此这一批次端到端为 **+74.98125**。若每题平均收益保持不变，约22个同分布目标可摊回源费；这是外推假设，不是本轮已经观测到的22题结果。与旧 V16.7 的实际前缀地平线实验也不能混为同一结果。

### 7.3 同历史普通控制对计价权重敏感

令 `C=M+λK`，K为当前逻辑 compute 单位。Hegel 相对普通同历史控制：

| 口径 | 差额函数 | 打平权重 |
|---|---|---:|
| 目标期 | `ΔC(λ)=18.75−143.875λ` | λ≈0.13032146 |
| 源期＋目标期 | `ΔC(λ)=28.75−71.875λ` | λ=0.4 |

普通控制多花逻辑计算、少花实验测量；在原λ=0.05下更便宜。这个结果为更大候选库中的检索优势留下研究空间，但新的优势必须在同信息、同确认规则的比较中成立。

证据：包内 `results/cost_sensitivity.json`。

### 7.4 240次工作区不是240种独立机制

当前是四个冻结种子簇，每簇相同十类结构；不同臂重复这些目标。240次真实执行值得报告，但不能当成240个独立的新机制样本，也不能把本套件0假确认推广为开放任务空间的接口保证。早期四簇描述性结果合理，优先修机制与比较，胜过单纯加几十个相同模板种子。

## 8. 其余会影响复现与门控解释的问题

| 问题 | 已证实什么 | 最小处理 |
|---|---|---|
| HExA 并列 schema 用 `max(set(...), key=count)` | 改变 Python hash seed 可在 min/reset 间选择；本轮 Hegel−HExA 目标差从原−40.725变为−36.69375，方向没反转 | 确定性 tie-break 并写入 manifest |
| `deepcopy(lab)` 共享 `_oracle` 闭包 RNG | 两个 fork 的 oracle 是同一函数；相同下一点会顺序消费不同噪声 | 显式复制环境状态或按 query/replicate 固定噪声；验证独立推进 |
| DEV 校准键在高角诊断前记录，目标在诊断后使用 capped 键 | capped 目标走 `exploration_arm_no_prior_guarantee`，未命中专门校准 | 对最终决策状态校准，保留未覆盖探索的真实标签 |
| 研究脚本使用 TemporaryDirectory | 临时工作区及未导出的中间拟合/确认上下文被删除；最终目标模型源码已保存在 `target_episodes.jsonl` 的 `authored_model_py` | 接开放代码时按版本保留实际被验证的完整上下文，receipt指向精确hash |

RNG噪声很小，审计没有证明它改变胜负；capped走探索也不是违规。这里需要修的是可复现合同和报告口径，不能把它们夸大成全部结果无效。

来源：[HExA tie-break](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L938-L940)；[paired fork](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L1434-L1461)；包内 `results/edge_reproducibility.json`。

## 9. 最小修复顺序：接通已有能力，保留有用简化

### 第一步：先让提交与确认状态可靠

只处理三个明确漏洞：确认前缀隔离；失败/未确认不返回成功；receipt绑定目标、节点版本、模型hash及确认数据，并要求有限误差。复用现有 confirmed model/context 保存机制。验收使用本包的正弦、memorizer、旧receipt/NaN反例和正常确认正例，不扩大为无边界的治理重构。

### 第二步：修程序与角色身份，冻结公平对照

限定 AST 可分析子集，验证旁注与执行代码，保留结构常量；复用角色枚举处理字段排列和重命名。普通组保留全部有效历史候选；HExA真正消费错误经验；Flat与图使用同一份映射、作用域、链误差和确认规则；NoWh不要同时禁止候选内容。修 tie-break、RNG fork和最终校准键后，再冻结一轮正式比较。

若这一轮 Hegel 与强普通方法持平或更贵，应如实保留。小库下持平不反证大库条件组织价值，反而给 M3 提供可信基线。

### 第三步：完成一条最小的真实开放 M2 闭环

项目已经有可复用实现：

| 能力 | 当前代码位置 | 接到 M2 时实际需要做的工作 |
|---|---|---|
| 开放 Python 作者与研究会话 | `code_agent.py` 的 CodeAgent、`code_session.py` 的 CodeResearchSession | 适配 OpenWorkspaceTaskLab 的 list acquire/task_view 与原 Probe/spec/channels；由同一 provider 决定 query/write/run/evaluate/finish |
| 有前提定位的开放修订 | `open_grammar_author.py`、`premise_dependency_audit.py` | 将标量观测残差转为作者可消费的证据/前提；其 Program/对象角色 AST 与 fit/predict 不同，需显式转换 |
| Python 研究方法学习与冻结 | `python_method_learning.py` 的 PythonArchiveMethodLearner/PreparedPythonMethodAgent | 绑定具体方法体、版本与后继消费者；记录实际动作改变 |
| 被确认代码与上下文保存 | `archive_confirmed_python_model`、`python_model_context_matches` 等 | 存精确被验证版本；避免旧receipt授权新改稿 |
| 运行与成本追踪 | `experiment_runtime.ExecutionTiming`、checkpoint设施 | 替换0.02占位，保存实际开放代码产物 |

历史原结果也表明这些能力并非空壳：v10 Phase2B 的 premise-gated 开放作者4/4，5次调用、34527 tokens；Phase3B 为3/4，5次调用、29218 tokens；两条后继 N4→N5 转移都4/4、零新增LLM调用。本轮核对了原JSON，没有重跑这些付费实验，也没有把它们升级为通用RSI证明。

最小新验收可以只选少量有区别的任务：旧程序遇到真实反例；作者提出不在现有十类 schema 中的机制/条件/观测或方法改稿；同时保留旧范围成功与新 witness 通过；冻结产物后在独立新目标被真实调用；移除新知识后行为变化；与同工具、同历史的普通会话比较全账。**不要求一次完成四种对象的全部开放创发，也不要求发现全新数学。**

来源：[已有 CodeResearchSession](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/code_session.py#L24-L195)；[CodeAgent 回路](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/code_agent.py#L1127-L1386)；包内 `history/originals/` 与历史子报告。

### 第四步：再评估 M3 的规模与多跳价值

图与森林可以保留。M3应在固定同内容库、相同映射与确认权限下，只改变检索/组合方式；路径来自真实源证据，去掉提示两跳的目标字段；分别计建库/建图、检索、错误候选和最终购测成本。如果当前十类模板下都没有真正平等的基线，扩到K≥32首先放大的是比较差异，未必增加解释力。

HippoRAG/PPR可用于召回关系路径，但召回不等于科学授权；真实多跳的映射有效性和组合误差仍需由当前合同及实验支持。这是基于实验设计的判断，并非声称检索论文已经为 Hegel 提供了科学正确性证明。

## 10. 阅读、访问与复现覆盖

### 10.1 本轮覆盖了什么

- 固定提交的全树路径/hash清单，识别12056个文件，其中 Hegel Machine 7508个；这是文件清单覆盖，不等于逐文件精读。
- 当前M2两个核心模块、研究脚本、测试、相关森林/评测入口、两份相邻报告、V166主要诊断与最小合同、相关主计划及原证据。
- 当前历史能力目录的181个版本、9条路线用于检索；按M2问题追到既有开放作者、Python方法与会话实现，并核对v10原始JSON。这不是181版源码全部逐行读完或重新运行。
- 初始五篇长文进行全篇标题与引文扫描、核心段落细读，五文hash与原覆盖清单相符；没有声称逐行精读全部约33.6万字节的多轮对话。
- 本轮13项目标测试、12个M0语义控制、4个boundary控制；原M2 240次完整重放；四个诊断臂合计160次目标；另外AST、确认、顺序、receipt与RNG的focused controls。

### 10.2 文献覆盖的准确边界

初始五文有52个URL身份，**不是52篇独立论文**。现有存档路由的状态是：

| 状态 | 数量 |
|---|---:|
| 仓库内有归档payload | 18 |
| 仅指向作者另一台机器的external bundle，本工作区未找到 | 28 |
| 项目来源指针 | 2 |
| Cambridge/Wiley全文访问受限 | 2 |
| OpenAI challenge记录及外部渲染快照指针 | 2 |

这张表描述当前存档与工作区状态，不把“外部bundle未找到”等同于“互联网上绝无可访问全文”。逐项链接、初始文档行号、路径和阅读状态见 `history/reference_inventory.md` / `.json`。

本轮关键论文中，ALIGN的7页主文完整读完；CEGAR、Gentner、DreamCoder、LILO、Stitch、HExA和HippoRAG2核查相关关键章节，未标成全部长文/附录全文精读。MathRSI完整读的是官方固定验收文档，未声称存在并读完一篇统一方法论文；AWM为摘要及旧审计定位复核。可获取、已下载、已提取文本、已阅读是不同状态，清单分别记录。

因此，**本轮已形成可复核的当前M2审计，但没有完成字面意义上的“所有历史代码、所有文档和所有参考文献逐字全文阅读”。** 这个未完成边界不能用“已经理解每个细节”掩盖。本文最强结论来自当前固定代码、原回执和实际控制，不依赖把未读文献或未复跑历史当作证据。

### 10.3 如何复现与继续审计

证据包 README 给出 `run_audit.py --repo '/path/to/Hegel Machine'` 的具体模式。代码必须检出本报告固定SHA；脚本只在单独输出目录产生工件，不修改项目或发起在线模型调用。原始审计脚本仅做路径参数化包装，原JSON和原重放作为静态证据保留。

M0语义旧probe输出头部硬编码了历史SHA；本轮实际执行目录为9115，该差异在 provenance明确标注，没有把旧header当作本轮源码身份。原结果、事后控制、focused反例和历史原件也分别标注，不能混成一份冻结实验。

## 11. 建议给当前版本的准确结论

**当前版本完成了开放工作区基础设施和有限机制族的跨域复用实验；原冻结套件全部通过，原费用可复算。当前确认与程序身份接口存在可复现缺陷；开放作者、一般反例修订和真实多跳消费尚未全部接入；对强普通方法及等内容平库的独立净优势仍待公平冻结比较。下一步先闭合确认、身份和历史消费，再复用项目已有作者完成小型真实开放闭环，之后再检验大规模图检索。**

这既保留本轮真实进展，也给下一轮一个范围有限、可执行、可判定完成的研究任务。

