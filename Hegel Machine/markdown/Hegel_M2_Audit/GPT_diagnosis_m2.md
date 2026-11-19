我把审计固定在 **`9115f9adf9dad1f52f106a2ab14babe8b68243dd`**。

**我的判断是：这轮选择性修复有实际效果，M2 的真实 Python 工作区、子进程执行和有限机制族跨域复用也已经跑通；但还不能认定上一版约定的开放 M2 已完整完成。当前最影响下一步的，是确认接口的实际漏洞，以及普通基线、横边消融和两跳比较的归因问题。** 森林、局部作用域、多精度对象和付费目标确认这些设计可以保留。

完整材料已经整理好：

- [完整审计报告：代码、实验与最小修复](https://chatgpt.com/space/page_ff9f70d9bcb08191ab03c9c42491107c)
- Hegel_M2_Audit_9115_20261119.zip[Hegel_M2_Audit_9115_20261119.zip](sandbox:/workspace/scratch/3f14ad2da0ed/deliverables/Hegel_M2_Audit_9115_20261119.zip)：包含主报告、机制与实验专项报告、原始回执、独立重放、诊断控制、文献覆盖清单、代码哈希和便携复现脚本。

先说明阅读边界：我已经追查当前 M2 的主要实现、前后里程碑合同、相关历史能力和原始结果，并实际复跑、复算和构造了控制实验；**没有把“所有历史版本和全部参考文献逐字读完”当作已完成事项。** 初始长文、历史版本及文献的实际阅读层级和缺口都在报告中明确列出。下面的结论以当前固定代码和实测证据为依据。

---

## 一、先确认：这轮哪些进展成立，哪些旧问题不应继续照搬

我复核后认为，不能把 V166 的批评原封不动套到现在。

本轮实际检查的结果是：

| 检查 | 结果 |
|---|---|
| M0、M1、M2 三个相关测试文件 | **13 项通过** |
| V166 语义控制重新执行 | **12 个控制的相关修复成立** |
| 旧 boundary 控制重新执行 | **4 个边界案例符合预期** |
| 原 M2 回执重新计算 | **未发现成本算术差错** |
| 原 M2 研究脚本完整重放 | **240/240 目标严格通过，0 假确认** |
| 额外诊断控制 | **四个臂合计 160/160 目标通过** |

原 M2 的源记录、目标记录、费用 ledger 和配对汇总能够对应起来；manifest 中四份代码哈希也与固定提交一致。九项不涉及 HExA 的配对比较逐值重现；HExA 的三项数值受并列选择的不确定性影响，后面单独说。原始统计见[固定提交的 paired_results.json](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/docs/evidence/v16_8_m2_open_workspace_wormhole_20261119/paired_results.json)，独立重放与复算见审计包。

几项具体改善值得明确认可：

- **真实工作区和真实执行已经接上。** 观测会落盘，模型文件真实生成，拟合和预测确实启动 Python 子进程。
- **失败源不再一律获得支持。** eval03 的楞次源任务失败被记录，并从有资格的形成和先验中排除。
- **普通组确实开始消费历史。** 不能继续说 `O_strong` 完全没有读取 `raw_archive`。
- **目标阶段存在购测确认。** 不是源任务成功，就直接把目标判成功。
- **退出复用时沿用已付观测。** 没有把数据清掉再重新收费。
- 当前未发现求解器直接读取隐藏目标参数或 heldout 答案帮助拟合。合成环境由同一实验框架生成，本身不能据此判定为答案泄露。

这些进展应保留。问题在于：**有限测试集全部成功，与接口在新任务上保证正确，是两件需要分别验证的事。**

---

## 二、最先该修的三个确认问题

### 1. 确认已经失败，仍然返回 `completed`

这是本轮最明确的正确性缺陷。

`solve_open_workspace_direct()` 在 fresh confirmation 失败以后，会再运行一次全模板拟合。只要拟合脚本没有执行错误，就返回：

```python
{"status": "completed", ...}
```

这里没有新的通过证据。代码把“程序成功运行”推进成了“科学任务已完成”。位置在[确认失败后的 fallback 分支](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L789-L825)。

我用相同的公开 lab 接口，给了一个简单的新规律：

\[
y=5+\sin(u),\qquad u\in[1,10],\qquad \text{tolerance}=0.01
\]

结果：

| 检查 | 实测结果 |
|---|---:|
| fresh confirmation RMSE | **1.289795** |
| 返回状态 | **completed** |
| 31 点外评 RMSE | **1.091451** |
| `strict_pass` | false |
| `false_confirmation` | **true** |

这不是要求有限模板库必须会解正弦函数。**它不会解可以；已经确认失败，却继续提交为完成，需要修。**

最小处理就够了：

- 未确认时返回 `unconfirmed`、`scoped_unknown` 或等价状态；
- 保留候选和已付观测，有预算时继续研究；
- 确认后的 refit 如果改变模型，就生成新的版本身份，不能自然继承旧模型的确认。

**这不推翻原 240 次冻结目标通过，但否定了“当前接口已经保证新任务零假确认”的外推。**

### 2. 真正开放 Python 接入后，确认标签会进入 `fit`

这处问题当前模板没有触发，但它恰好阻挡下一步开放作者接入。

流程是：

1. `lab.acquire()` 获取确认点；
2. 确认标签被追加到 `lab.observations`；
3. M2 的验证包装器把全部 `lab.observations` 当作训练数据传给评测器；
4. 模型只要提供了可调用的 `fit`，评测器就会执行它。

`require_fit=False` 的含义是“不要求模型必须有 fit”，并不是“不允许执行 fit”。对应代码见 [M2 验证包装器](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L519-L553)和[评测器 fit 调用](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/code_agent.py#L600-L612)。

我用一个合法的记忆型模型验证：

- 训练点为 `u=1,2`；
- 新确认点为 `u=3,y=9`；
- `fit` 只记住训练表，陌生输入预测 0。

结果：

| 输入给 fit 的数据 | 确认 RMSE |
|---|---:|
| 当前 M2 wrapper 给出的全部数据，包括确认标签 | **0** |
| 仅确认前的训练前缀 | **9** |

**当前生成模板的 `fit` 是空操作，所以不能据此说原实验发生了这种泄漏。** 但把已有 `CodeAgent` 接进来以前，必须冻结确认前的训练上下文、模型代码和参数，再独立验证。

否则，开放作者只要正常编写有状态的 `fit/predict`，就可能无意中得到虚假的确认通过。

### 3. receipt 尚未绑定真正被确认的对象

当前 receipt 检查没有完整核对：

- 当前目标身份；
- 当前节点版本；
- 实际被测模型的内容哈希；
- 确认数据身份。

误差字段缺失会通过；NaN 也会通过普通大小比较。代码见 [receipt validator](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L155-L170)。

focused controls 中，旧目标/旧版本 receipt、缺误差 receipt、NaN receipt 都取得了 `local_solver_guidance`。

原 M2 主循环现场生成了正常 receipt，我没有发现原数据利用这些漏洞。但一旦进入跨目标、跨版本、持久化开放代码复用，这些绑定就很重要。**加上身份核对和有限值检查即可，不需要再扩大治理框架。**

这三组反例的输入、实际生成模型和完整输出都在审计包的 `mechanism/` 与 `evidence/mechanism_workspaces/` 中。

---

## 三、最影响“强普通基线净胜”结论的问题：普通组仍没有充分使用自己的历史

这部分比增加更多种子更值得先处理。

当前 `O_strong` 已经读取历史，但 `_reconstruct_prior_by_arity()` 有两个限制：

1. 每个 active arity 只保留一个 schema；
2. 至少要有两个不同源领域支持，才允许成为先验。

因此：

- 两个不同的二维机制，被压到同一个缓存槽；
- 单个 near-miss 中已经验证的三维 memory 机制，没有候选资格；
- `shared/` Python 文件虽然复制到目标工作区，目标求解器没有实际 import 或执行这些文件。

这些限制可以直接从[普通组历史消费代码](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L868-L905)看到。

这里需要区分：**允许一个历史模型成为待检验候选，不等于允许它未经目标确认就被当成定律。**

### 我做的最小控制

只增强历史消费：

- 同 arity 的全部已验证 schema 都允许成为候选；
- 包含单源 near-miss；
- 沿用原合成器、购测确认和外评器；
- 保留重建费；
- 每个额外候选另收 **10 compute**，没有免费增加搜索。

结果如下。差额均为 **Hegel − 普通组**；负值表示 Hegel 更省；单位是**每个十题簇的平均总成本**。

| 指标 | 原普通基线 | 使用全部有效历史候选的普通控制 |
|---|---:|---:|
| 目标测量费差额 | −35.0000 | **+18.7500** |
| 目标复合成本差额 | −41.0250 | **+11.55625** |
| 含源期的端到端成本差额 | −27.4250 | **+25.15625** |

普通控制 **40/40 通过，而且四个簇都是普通控制更便宜**。逐题数据、代码和额外候选收费见[完整审计第5节及复现材料](https://chatgpt.com/space/page_ff9f70d9bcb08191ab03c9c42491107c)。

这是一项**事后诊断控制**，我不会据此宣布普通方法普遍更强。但它足以说明：

> 原实验尚未证明 Hegel 相对于充分使用同一历史的普通方法具有净优势。当前优势对普通组如何压缩和调用历史非常敏感。

这对项目方向的含义是：需要证明**知识组织、适用条件、检索和消费方式**带来的收益，不能把普通组遗忘了已经验证的候选所造成的损失，全算成哲学一般化的收益。

HExA 对照也有对应问题：`mistake_bank` 写入了 near-miss，目标入口却没有消费它，见[相关写入与调用代码](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L919-L967)。HExA 原方法包含条件技能、错误经验和跨任务重定向；其论文结果采用 static injection，所以不能要求它必须在每个评测 episode 中在线更新，但也不能用未消费错误库的多数模板缓存代表完整能力。当前实现适合称为 **HExA 启发的简化基线**。[arxiv.org](https://arxiv.org/html/2606.29315v2?utm_source=chatgpt.com)

---

## 四、横边和 Flat：目前证明的是功能组合有用，图结构的独立贡献还没分离出来

### 1. NoHorizontal 同时禁掉了跨域历史使用

关闭横边以后，森林仍然可以返回 local node；但 `_filter_cross_domain()` 随后把源领域与目标领域不同的 local node 都过滤掉。由于目标是新的领域名，这个臂失去了大量历史候选。代码见[跨域过滤位置](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L1412-L1417)。

控制实验仍关闭全部横边，只允许同内容 local node 进入原有的候选和确认路径：

| Hegel − 对照，均值/十题簇 | 原 NoHorizontal | 无横边但允许 local 历史候选 |
|---|---:|---:|
| 目标复合成本 | −100.61875 | **−1.96875** |
| 目标测量费 | −96.25 | **−1.25** |
| 对照通过率 | 40/40 | 40/40 |

**约 98% 的原差额消失。**

因此原消融可以保留为“跨域历史复用整体功能开关”。但它不能独立回答“图结构比同内容的非图组织方式好多少”。

正式对照还需要把边上的映射、适用条件和排除条件保存在等内容记录中，然后只改变检索和组合方式。

### 2. Flat 的 `−1.74375`，大部分并不来自两跳任务

逐题分解是：

| 原 Hegel − Flat 差额来源 | 均值/十题簇 |
|---|---:|
| C4 两跳任务 | **−0.475** |
| A3 双模蠕变任务 | **−1.44375** |
| 其余八题合计 | +0.175 |
| 总计 | −1.74375 |

C4 两边的测量费都是 **35**。差别是 Hegel compute 为 **53.5**，Flat 为 **63**：主要少了一次受限拟合的代理计算费用。

所以报告中把整个 `−1.74375` 都解释为“两跳非传递图结构纯增益”，不准确。逐题分解见[审计报告第5.3节](https://chatgpt.com/space/page_ff9f70d9bcb08191ab03c9c42491107c)。

另外，Flat 确实复制了 Stage1 gate 却没有调用。我把同一个 gate 接回后，**当前四簇数字完全不变**。这一设计差异应该统一，但不能把它说成已证实的现有差额原因。

### 3. 当前两跳是预置的组合检查

代码中：

- 第一跳误差预算写死为 `0.015`；
- 第二跳写死为 `0.018`；
- relay 节点没有独立来源实验就被标记 supported；
- 目标公开 `two_hop_bridge_only=True`；
- 求解器据此触发两跳误差检查。

来源是[中继节点与预算构造](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L1137-L1193)和[目标公开属性](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_labs.py#L724-L740)。

这里有一个数学解释也需要修正：

\[
0.015+0.018=0.033>0.025
\]

算术没有问题。但**误差上界大于容差，只说明这个界不足以认证，不能推出实际误差一定超过容差。**

因此，这个例子验证了预置组合约束确实执行，可以保留为门控控制。它还没有证明从历史数据形成两条映射、估计误差、发现路径、实际执行组合之后的自主导航价值。

同内容 Flat 也可以保存路径条件和组合误差。图的优势应通过检索成本、错误候选率或组合效率体现，而不是让 Flat 天生丢掉这些信息。

---

## 五、M2 的“开放、AST、CEGAR、方法程序”四项主张，目前分别到哪一步

### 1. 工作区开放已经成立，假设作者尚未开放

当前 `_WORKSPACE_SYNTHESIZER_SCRIPT` 是固定脚本，枚举十类预写 schema，再拟合常数、生成 `model.py`。

`allowed_schemas=None` 表示搜索这十类，不表示任意 Python，也不表示开放算子组合。M2 调用的是 `evaluate_python_model`，没有接入 `CodeAgent.run` 或 `CodeResearchSession` 的在线作者回路。见[固定合成器实现](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L132-L452)。

**有限语法本身合理，不能把它列为正确性缺陷。** V166 的最小合同也允许这样做。

但 V16.7 报告第96行已经明确约定，M2 要让在线代理从真实残差自主提议 `ClaimContract / ObservationMap / ScaleMap / program_steps`，并替代固定插件枚举。见[上一版的 M2 定义](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/09_V16_7_RedTeam_Response_and_Upgraded_Results.md#L94-L97)。

因此准确状态应是：

**M2 工作区与评测基础完成，有限机制族迁移部分完成；开放作者和残差驱动合同修订的接入未完成。**

### 2. AST digest 目前不能承担可执行语义同一性的证明

规范化器优先读取 `_CORE_LAW_EXPR` 旁注；没有旁注时，仅提取 `predict` 的直接 return，没有解析中间变量和辅助函数。同时数值常量、指数及负号可能被一并归一化。见[AST 规范化代码](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L58-L126)。

原函数上的三个控制都得到相同 digest：

| 两个程序的区别 | 当前结果 |
|---|---|
| 旁注相同，但一个实际预测 x，另一个实际预测 100000 | 同 digest |
| 无旁注，latent 的实际定义不同，但外层 return 相同 | 同 digest |
| `x**2` 与 `x**-2` | 同 digest |

当前模板生成器自己提供正确旁注，所以这不等于原40题必然出错。但它说明，一旦接入真正自由的 Python 改稿，旁注陈旧或数据流不同就会破坏当前归组。

最小修复不需要解决任意程序等价判定：

- 限定可分析的纯函数子集；
- 校验旁注与实际执行代码；
- 区分可拟合系数与结构常量；
- 保留指数、符号和定义域；
- 存实际程序 hash；
- 超出支持范围时返回“骨架未知”。

AST 可以用于召回类比候选；目标上的角色、尺度、观测协议和误差仍然要验证。

### 3. 角色对齐仍依赖字段排列

当前二维 pilot/signature 默认第一个变量是驱动，第二个变量是时间；第三个变量默认承担 memory 角色。

同一个 T_M2A1，只逆序 `feature_names`，其余物理、变量名字、边界和外评不变，得到：

| 条件 | 测量费 | RMSE | 结果 |
|---|---:|---:|---|
| 原字段顺序 | 30 | 0.017834 | pass |
| 逆序 | 55 | 0.049067 | fail，容差0.045 |

这是默认 gate 的单任务诊断，不能把其中30直接和冻结报告的35比较；但两组之间足以说明当前“坐标无关”表述过强。相关顺序假设见[pilot/signature 实现](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L586-L712)。

已有合成器能枚举角色，应先把这项能力接到形成和目标验证，再扩大领域命名数量。

### 4. near-miss 修订和 `program_steps` 的实际作用有限

near-miss 当前先作为独立任务被模板库拟合，然后根据 `memory` 或 `capped` schema 名，进入两个预写修订分支。

这证明已有修订模板被选择、保存并用于后继任务。它还没有完整展示：

> 旧模型在具体 witness 上失败 → 定位失败前提或状态缺口 → 提议竞争修订 → 保留旧范围成功 → 独立验证新预测。

代码见[近失例修订路径](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L1195-L1363)。

ALIGN 对这里真正有用的是近失例检索、包含/排除条件，以及后来正例对过强条件的修正。CEGAR 则在明确的模型和抽象语义下检查反例并精化。M2 可以借用这些思想，但需要自己的真实冲突与可执行改稿，不能从名称继承原方法的保证。[qrg.northwestern.edu](https://www.qrg.northwestern.edu/papers/Files/QRG_Dist_Files/QRG_2015/McLure-Friedman-Forbus-AAAI15.pdf?utm_source=chatgpt.com)

`program_steps` 也有类似边界：当前目标入口主要读取其中的 schema 字段，calibration、confirmation/refit 仍由主机固定流程执行。删除节点声明中的 calibration 和 confirm/refit 步骤，动作、费用和预测完全不变。

因此目前更接近**固定执行器的部分配置**。若要主张学到了研究方法程序，需要让方法体的合法改动确实改变后继动作，并测出其效用。

---

## 六、成本结论需要补齐口径，不能把原数字直接否定

### 1. 原“给定 gate”端到端优势是准确的限定结果

原 `Hegel − O_strong = −27.425/簇` 包含各臂源期和目标期，而且报告明确写了“给定门控”。不能说它算错或隐瞒。

但开发形成与配对校准还花了 **1884.35**。如果把这笔一次性费用只记给 Hegel，并分摊到目前四簇：

\[
-27.425+\frac{1884.35}{4}=+443.6625
\]

这是一个明确的分摊情景，不是唯一合理部署口径。共用开发设施、只形成一次源库、延长目标流，都会改变结果。需要先说明部署情景，再判断生命周期净收益。

双源也一样：

| 双源相对单源 | 成本 |
|---|---:|
| 额外源期投入 | +139.175 |
| 当前十个目标节省 | −64.19375 |
| 该批次端到端差额 | **+74.98125** |

因此当前 M2 的十题批次还没有摊回双源投入。按不变平均收益外推，大约22道类似目标可能摊回；那是外推，不能当成本轮实际观测结果。详细账本见[审计第7节](https://chatgpt.com/space/page_ff9f70d9bcb08191ab03c9c42491107c)。

### 2. compute 是代理费用，部分“执行秒数”仍是占位值

当前每次合成固定收10或24 compute，并固定增加 **0.02秒**；验证阶段记录的才是真实 subprocess 时间。因此汇总的 `python_execution_seconds` 混合了两种含义。见[合成计价与计时代码](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/open_workspace_wormhole.py#L455-L515)。

代理计价可以保留，但应与真实墙钟、模型调用和 token 分账。

对上面的普通同历史控制，令 \(C=M+\lambda K\)，有：

\[
\Delta C_{\text{target}}(\lambda)=18.75-143.875\lambda
\]

在原 \(\lambda=0.05\) 下普通控制更省；到约 \(\lambda=0.1303\) 才打平。准确解释是：**普通控制多花代理计算、少花实验测量，在当前权重下合算。**

### 3. 还有三个范围有限的复现问题

- **HExA 并列选择不确定。** `max(set(schemas), key=schemas.count)` 会受 Python hash 顺序影响。此次 Hegel−HExA 目标差从原 `−40.725` 变为 `−36.69375`，方向没反转。应固定 tie-break。
- **DEV 的两个 deepcopy fork 共享 oracle 的 RNG 闭包。** 相同下一查询会顺序消费不同噪声，不满足严格同随机状态分叉。噪声很小，尚无证据说明它改变胜负。
- **capped 校准键与实际消费键不一致。** DEV 在高角诊断前记录 signature，目标在诊断后使用 capped signature，因此相应目标实际走未校准的探索准入。探索本身合理，报告应按真实路径解释。

这些不需要被夸大成全部结果无效，但下一轮冻结前应修好。代码与 focused 记录见审计第8节。

---

## 七、结合项目历史，下一步应该怎么修，才能避免再造一套框架

我对项目初衷的理解是：**历史中的有效经验要成为后继研究的可执行资源；反例推动概念边界和表示变化；变化保留旧成功，并在未来任务中带来可追溯效用。**

这支持保留森林、局部范围、近失例分支和多精度对象。当前优先处理的是让这些对象真正连接到作者、验证器和后继消费者。

项目已经有：

- `CodeAgent` 的 query/write/run/evaluate/finish 回路；
- `CodeResearchSession` 的共享代码和研究会话；
- `open_grammar_author` 与前提定位修订能力；
- `PythonArchiveMethodLearner` 等研究方法学习与冻结接口；
- 已确认模型和执行上下文的保存设施。

见[CodeResearchSession](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/code_session.py#L24-L195)与[CodeAgent 回路](https://github.com/erzhu419/Assumption-Agent/blob/9115f9adf9dad1f52f106a2ab14babe8b68243dd/Hegel%20Machine/src/hegel_machine/cognition/code_agent.py#L1127-L1386)。我也核对了相关 v10 原始实测 JSON，不能把当前 M2 没接入，写成整个项目从未有过开放作者。

不过旧开放语法路径消费的是对象角色 AST、transition 和 world commitment，M2 当前是标量观测与 `fit/predict` 文件，**仍然需要明确的接口适配，不能只换函数名就算接好了。**

我建议下一轮按这个次序验收：

| 顺序 | 工作 | 最小验收 |
|---|---|---|
| 1 | 确认状态、训练前缀、receipt 身份 | 正弦失败不再 completed；memorizer 无法吃到确认标签；旧版本/NaN receipt 被拒绝 |
| 2 | AST 与角色绑定 | 三个 AST 碰撞不能直接作为机制同一证据；字段排列不破坏角色识别 |
| 3 | 同历史、同信息的对照 | 普通组保留有效候选；HExA 消费错误经验；Flat/NoWh 保留同样条件、映射和确认规则 |
| 4 | 接入现有开放作者，跑一条真实修订链 | 旧假设的实际反例导致新改稿；保留旧范围成功；冻结后在独立新目标真实调用 |
| 5 | 再测 M3 的大库与多跳价值 | 固定同内容库和权限，只改变检索/组合方式，计算建库、检索、错误候选和购测全账 |

第四步可以很小：只做少量真正有区别的机制、条件或方法改稿，不要求一次完成四种对象的全部开放创发，也不要求发现全新数学。

如果公平对照下小库阶段 Hegel 持平或略贵，应如实保留。小库持平不会否定大库条件组织的潜力；它会让后面的 M3 更容易解释。

HippoRAG/PPR 可以帮助召回关系路径，但找到路径不等于映射合法、组合误差合格或目标得到确认。后面这几件事仍需 Hegel 自己的合同和实验来承担。[arxiv.org](https://arxiv.org/html/2502.14802?utm_source=chatgpt.com)

---

## 八、关于你要求的“所有版本、所有细节和参考文献全文”

这一点我需要明确交代，不能用“已经全面理解”遮住实际覆盖缺口。

本轮做到了：

- 固定提交全树的路径/hash清单；
- 历史能力目录的181个版本、9条路线索引，并围绕 M2 追到相关原实现和原始结果；
- 初始五篇长文的全篇标题、引文扫描与核心段落细读；
- 当前 M2 核心实现、相邻合同、相关 V166 诊断及实际消费者追踪；
- 上面列出的重放、复算和诊断控制。

**没有完成181版全部源码逐行精读，也没有完成全部引文全文逐字阅读。**

初始五文中的52个 URL 身份，现有归档状态包括18个仓库内 payload、28个仅指向作者其他机器的外部 bundle，以及项目指针和访问受限项；52也不是52篇独立论文。本轮 ALIGN 主文完整阅读，CEGAR、HExA、HippoRAG2及其他相关方法核查了关键章节，清单没有把“已下载”“已提取文本”和“已读全文”混在一起。具体缺口保存在证据包的 `history/reference_inventory.md` 和 JSON 中。

**当前版本最合适的结论是：有限机制族的工作区迁移实验成立，开放作者与一般修订闭环仍待接入，强普通基线和图结构独立净优势仍待公平冻结比较。** 下一轮先把确认、身份和历史消费这几个明确问题闭合，再用已有作者能力完成一条真正由反例推动、在后继任务消费的修订链，就能继续推进，而且不必推翻已有架构。

本轮没有修改或推送你的仓库代码。