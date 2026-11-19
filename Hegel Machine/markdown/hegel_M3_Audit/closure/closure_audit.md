# 612d8eea5：V16.9 的 F1–F11 闭环与开放作者入口复核

审计固定提交：`612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8`。对照上一轮固定提交 `9115f9adf9dad1f52f106a2ab14babe8b68243dd` 及交付包 `Hegel_M2_Audit_9115_20261119`。本报告行号均指新提交 `Hegel Machine/` 下文件。

## 1. 结论

**这轮是真修复，但“F1–F11 全数闭合”并不成立。** 已有几项关键修复应明确承认：确认失败返回 unknown、成功后不再用未确认重拟合覆盖代码、`fit` 的显式参数只得到确认前缀、三组旧 AST 碰撞消失、角色对称探针、独立 RNG fork、真实计时、普通基线保留全部验证历史、HExA 使用错误库，以及同信息 Flat/NoWh 对照。

剩余问题有两种，不能混说成原结果全错：

1. **当前新实测的解释边界**：所谓开放作者实际使用五个预写公式和脚本 provider；诊断输出不决定新候选；开放链对照的额外测量由脚本强制安排。一次只去掉普通臂额外两点的控制，两个目标仍通过，测量优势全部消失，计算优势仍保留。
2. **开放推广合同还存在具体漏洞**：确认标签仍写进模型的工作目录，合法按文件训练的模型能通过假 fresh confirmation；AST 的受支持调用仍有误合并；M3 又绕过 AST verification 状态，把 9 个 `mismatch:` 节点收进 supported 森林。这 9 个中至少两例已用原源任务重放，确认为规范化器误拒合法代码，而不是物理模型错误。

不能据此要求无限语法、任意程序等价判定或把森林拆掉。当前最小工作是修实际生命周期边界、规范化/准入接口及因果对照，并把未完成的开放作者主张降到实现真实支持的范围。

## 2. 上一轮 F1–F11 逐项判定

| 旧项 | 本轮实际改善 | 仍需保留的边界 | 判定 |
|---|---|---|---|
| F1 开放作者 | 真正调用 `CodeAgent.run`；有 query/write/run/evaluate/finish；生成模块真实执行 scipy `fit` | `ScriptedOpenGrammarCodeProvider` 消费预先给出的候选字符串，renderer 只有五个分支；没有自由作者或残差驱动的算子组合生成 | 接线完成，开放生成未闭合 |
| F2 AST 身份 | 局部赋值内联、正文与旁注核对、指数符号保留；原三例已修 | 函数名绑定未校验，调用 keyword 被丢弃；predict/annotation 仿射处理不一致；M3 不检查失败状态 | 部分修复，存在原路径回归 |
| F3 反例修订 | 旧模型确实先在 witness 上失败；新候选通过新点；旧源族可重拟合；保存修订节点 | 候选、`rev_sig`、修订合同均为宿主预写，诊断输出未进入生成决策；不是一般新机制发明 | 实测受控修订成立，开放主张未闭合 |
| F4 两跳与平库归因 | 去除目标专用 bool；真实第三个 source；第二跳数值来自拟合；Flat 同门控/路径界，NoWh 保留 local 内容 | 拟合残差尚不是跨域映射的误差上界；仍未执行两次实际运输；桥拟合独立 meter 未汇总 | 明显改善，仍是受控路径/误差启发式 |
| F5 失败仍 completed | 原 sinusoid 反例诚实 unknown；取消容差放宽；成功后冻结模型及训练前缀 | 假 fresh confirmation 可由 F11 的文件通道绕过；应归 F11，而非否认 F5 分支修正 | 原缺陷已修 |
| F6 列顺序 | 二维五点与三维六点 pilot 的点集对称；二维镜像角色判断；三维交合成器枚举 | 当前测试只测一个二维反序，不能代替任意语法/所有排列保证；无必要为此无限扩测 | 原聚焦反例已修，范围需准确 |
| F7 receipt | claim 禁通配符、误差必填有限非负、已给的 target/version/hash/digest 能校验 | expected 字段仍可选，`heldout_confirmed=True` 仍可绕开 receipt；新开放链没有 confirmation digest | 主 M2 正常调用改善，接口非强制全绑定 |
| F8 步骤执行 | 删除校准/确认确实改变执行与回退 | 仍转为 op→dict、检查存在性并走固定流程；顺序/重复/自定义过程体没有执行语义 | 固定过程配置已消费，通用解释器未成立 |
| F9 RNG fork | lab 显式持有 Random，deepcopy 克隆 getstate，再绑定新闭包 | 对当前纯 oracle 已足够；不应为无证据的任意可变 oracle 继续扩展要求 | 原缺陷已修 |
| F10 计时与验证上下文 | 合成时间改为实测；code hash 与 prefix 保存 | 完整依赖/拟合上下文未进入 receipt；官方研究脚本仍删除临时工作区，开放链仅导出摘要，不能从原摘要恢复完整作者交互 | 计时已修，开放上下文保存部分完成 |
| F11 fit 泄漏 | `fit(observations)` 参数只收到确认前缀，原 memorizer 正确被挡 | 同一 cwd 的两份标准数据文件仍含确认标签；完整 CodeAgent 新入口可复现 completed 假确认 | 原参数反例已修，隔离合同未闭合 |

主要代码定位：`open_workspace_wormhole.py:790–870, 911–1160, 1168–1276, 1329–1381, 1418–1508, 1780–1826, 2239–2324`；`open_workspace_labs.py:62–119`；`wormhole_forest.py:155–221`。

## 3. 新开放作者入口：真实做了什么

### 3.1 CodeAgent 接线是真的，生成过程是确定性枚举

`open_python_revision.py:99–191` 的 `_render_stateful_least_squares_module` 只接受五个完整机制：Monod、Haldane 一维、Haldane 加旁路、过阻尼阶跃、欠阻尼振铃。`206–294` 的 provider 将宿主给出的 query batches 和 candidate specs 依次执行，唯一自适应分支是评测误差达标后 finish。

`623–635` 在开始实验前就把 O1 的三候选、O2 的两候选及最终签名列好。`686–695` 先渲染所有 witness 候选，随后 `697–706` 的“诊断”只输出样本数、y 的最大/最小值及 span。`next_action` 没有使用 run 的 stdout 选择结构。控制向两个克隆 provider 分别返回“高驱动抑制伴旁路”和“纯振荡无旁路”，下一条 write 的文件内容逐字相同。

因此准确描述是：**接入真实 CodeAgent 工具回路的、五个新公式族的受控修订/复用实验。** “在旧十类库之外”成立；“非预写模板的开放作者”不成立。脚本 provider 的 usage 是固定计数，用于工具协议预算；这不是本轮在线 LLM 调用证据。项目历史 v10 曾有真实在线开放作者，因此这里是已有能力尚未接入新入口，不能写成整个项目从来没有开放作者。

任务生成器与候选选择位于同一模块，宿主按 `chain_key` 将正确机制族预先分派给 provider；这让实验从开始就知道要在何种公式之间修订。未发现拟合器读取目标私有 `_oracle` 参数或外评答案，所以不应称数值答案泄漏；问题是这个设计不能检验作者是否从未知反例自行形成了新机制。

### 3.2 witness 的失败与后继成功是真实的，但修订结构是预写的

`714–719` 真正重测 incumbent 在 witness 上的误差；`504–526` 执行新点确认；`749–791` 注册新节点/虫洞。应保留这些成果。

但是注册的 `operator_ast_kind` 和 `rev_sig` 来自固定候选与预先指定的链。目标命中后 `853–858` 再从宿主 renderer 按 kind 渲染同族，不是执行一般的、历史学出的 Python 模块。它证明了标签/角色/模板族的复用和参数再拟合，没有证明完整开放源码知识的后继迁移。

旧知识保留检查 `721–747` 在旧源训练行上**重新拟合修订后的公式**，然后评旧源确认行。这证明修订公式族仍能表示旧源规律；由于源与 witness 参数本来不同，不要求同一参数向量适用所有任务是合理的。但应称“公式族对旧源的重拟合兼容性”，而不是声称 witness 上已拟合的同一个模型在不变化时保留旧知识。新旧节点同时保留确实属于知识库存层面的保留。

### 3.3 普通臂的额外两点不是数据迫使它买的

`890–913` 对删除修订节点的臂强制先买三点。`925–945` 对 ordinary 强制先买两点；它没有先测试已有八点是否足够。Hegel 则先做同一 pilot、拿专用角色/签名函数，随后直接试一个正确族。

本轮控制只对 ordinary 的 `candidate_specs` 文件名识别，移除第二个 query batch；没有改候选内容、候选顺序、确认点、容差或外评，Hegel/ablation 保持原样。取发表 seed `5026101`：

| 目标/臂 | 原测量费 | 控制测量费 | 原复合费 | 控制复合费 | 控制外评 RMSE | pass |
|---|---:|---:|---:|---:|---:|---|
| O1 Hegel | 50 | 50 | 53.225 | 53.225 | 0.000574 | 是 |
| O1 ordinary | 60 | 50 | 66.225 | 55.075 | 0.000574 | 是 |
| O2 Hegel | 50 | 50 | 53.225 | 53.225 | 0.001468 | 是 |
| O2 ordinary | 60 | 50 | 71.225 | 59.075 | 0.001468 | 是 |

两个未改的 ablation 仍为 65 测量费、74.7/77.45 复合费，全通过。这个单 seed 控制足够反驳“普通臂必需额外两点”的归因；它不能替换四 seed 的正式再评估。原记录可以如实复现，剩余候选枚举的计算收益也仍在：不能因测量差被消除就声称森林没有价值。

最小纠正：所有臂拿到同一 pilot 后先允许测试同等历史候选；只有未过当前质量门时才买新点。Flat/ordinary 对等可用的角色诊断工具也应一致，分别报告“候选排序/检索减少了多少计算”和“实验决策是否真少买数据”。

## 4. F11：文件可见性仍破坏确认前缀隔离

这一问题不是要求防御任意恶意程序。现有 CodeAgent 公共约定就告诉作者可读取 `observations.json` 和 `model_observations.json`（`code_agent.py:66–71`）。

`lab.acquire(confirm, workspace=...)` 会先把确认数据加入 observations 并同步到这两份文件（`open_workspace_labs.py:115–119, 145–157`）。`evaluate_python_model` 虽把 prefix 写入临时训练文件传给 fit，但 subprocess cwd 仍是原工作区（`code_agent.py:739–752`）。因此 `fit` 用显式参数或按标准文件约定读数，会得到不同训练集。

本轮原路径控制：训练为 u=1,2，真实 y=u²，新确认 u=3。

| 普通合法模型的 fit 数据来源 | wrapper 报告 prefix_count | 新点 RMSE |
|---|---:|---:|
| 函数参数 observations | 2 | 9 |
| `json.load(open('observations.json'))` | 2 | **0** |
| `json.load(open('model_observations.json'))` | 2 | **0** |

进一步从 `_run_code_agent_with_prefix_confirm` 完整调用 query→write→evaluate→finish，同一文件读取记忆器得到：`status=completed`、`confirmation_rmse=0`、`confirmed_training_prefix_count=2`。外部再评 u=4 时预测 0，RMSE=16，`false_confirmation=true`。这复现了新开放入口实际可发生的假确认。

**影响范围：** 当前预写的 scipy 模型只用 fit 参数，因此没有证据说发表的 O1/O2 或 M3 有此泄漏。问题是“开放 Python 都被严格隔离”的接口合同不成立。

最小修正：在购买确认前，用已有执行上下文/归档能力创建拟合快照；快照内两份标准训练文件都只含 prefix，模型依赖也被冻结。确认仅把输入特征传入该上下文，在另一记录处保存新标签与误差。随后若用全部已付数据重拟合，生成新 artifact 身份并重新确认。原参数 memorizer 和两种文件 memorizer 必须在确认点均失败，正常 scipy 模型应保持通过。

## 5. F2：受支持 AST 子集仍有两种误合并，M3 又出现实路径误拒

### 5.1 两个很小的受支持代码碰撞

`_SkeletonNormalizer.visit_Call` 按文本名称识别函数，未验证其绑定；最后生成新 Call 时直接 `keywords=[]`（`open_workspace_wormhole.py:136–148`）。

| 两份代码的差别 | 规范化结果 | 真实行为 |
|---|---|---|
| 模块中 `exp(x)` 分别定义为 x² 和 x³，predict 都调用 exp(x) | 都 VERIFIED_PURE_AST，digest 相同 | x=2 时 4 对 8 |
| predict 分别调用 `round(x, ndigits=1)` 与 `round(x, ndigits=3)` | 都 VERIFIED_PURE_AST，digest 相同 | x=1.2345 时 1.2 对 1.234 |

这不是任意 Python 等价难题：对尚未支持的绑定/keyword 返回 UNKNOWN 即可。最小修复是解析允许的导入/内置绑定，拒绝影子重定义；keyword 要么按明确语义规范化，要么拒绝；`ndigits` 等结构参数必须保留具体值，不能仍泛化成同一个 POS_NUM。

### 5.2 M3 图中 9/36 个 supported 节点带 mismatch digest

提交的 `wormhole_forest_graph.json` 包含九个 `mismatch:` 节点：linear 两源、exp 两源、cobb 两源，以及 log、inv-linear、sqrt 三个单源，全部是 `SUPPORTED_IN_SCOPE`。M3 的 source 准入只检查外评通过、digest truthy、schema 存在（`m3_scale_universe.py:1712`），没有像 M2 `open_workspace_wormhole.py:1636–1642` 那样检查 VERIFIED_PURE_AST。

不能简单把这九个说成“九个假模型”。本轮实际重放同 graph seed `7027404` 的 S01/S02；两者 strict_pass=true，digest 分别为 `mismatch:9441462bd127` 和 `mismatch:bcbbf1b14db9`，与提交 graph 逐字一致。生成代码的旁注和 latent 公式完全一致。

根因：predict 的 `_strip_outer_affine` 循环会将 core 内的数值截距和系数也继续剥离；annotation 分支却没有同样的规范化。因而 `latent = a + b*x; return offset + sign*scale*latent` 的 predict 最后是 ROLE_0，annotation 却仍为 POS_NUM + POS_NUM*ROLE_0（`open_workspace_wormhole.py:247–281, 304–305, 347–359`）。同一机制族因此被模型内容 hash 分开，三对本应可归组的源不能用等骨架生成横边。

最小修正不能只有“过滤 mismatch”。应先修 lens 边界/正文旁注一致验证，分开读出映射剥离与核心族规范化；确认正常受控模型得到 verified，再让所有 consumer 用显式 verification_status 控制机制同一性授权。未知骨架仍可保留为单独的已验证求解器候选，但不得用其 digest 充当已验证的抽象同一性证据。最后重跑 formation/index 及对应消融，因为图内容已改变。

## 6. 其余边界：应精确收口，不必过度修复

**receipt。** 当前 M2 主流程会提供 target/version/hash/confirmation digest，故本轮不指称原正常调用用了过期收据。新开放链 `871–882` 的 receipt/task_view 没有 confirmation digest；validator `184–198` 对期望字段采用“给了才检查”；`221` 仍接受 `heldout_confirmed` 布尔旁路。因此报告第 10 行的“强制五元组”需要收窄。最小方案是新入口显式 required receipt mode，由调用上下文独立提供身份与确认记录；旧兼容接口保留但不用于新实验的认证主张。

**program_steps。** 删除步骤确实改变行为，原反例不能继续照搬。但 `2239–2261` 转 op→dict 后仍按固定校准→合成→确认执行，不读声明顺序和任意程序体。当前有限操作合同可以保留；称“由存储步骤配置驱动的固定执行器”即可。若未来研究元方法程序，再接项目已有可执行研究过程，不必为了当前 M3 单独造通用解释器。

**两跳证据。** 第一跳取各源 eval/confirm RMSE 的最大值，第二跳用 rational 族拟合 rollover 已付数据的 train RMSE（`1744–1769, 1780–1826`）。这比写死常数好，但不能直接叫保证误差上界；同数据拟合的平均残差不保证跨域/未见区域行为，更未实际执行两跳映射。明确称经验路由风险估计，并保持目标 fresh confirmation，就能继续做受控实验。若要主张路径非传递的科学必要性，再加每边独立验证点与真实映射组合，没必要全面形式化。

**成本与保存。** 第二跳 probe_lab 的独立 meter 未写回 source logs，故还有小额桥学习费漏账；总账由主审计另核。原脚本 TemporaryDirectory 内完整作者工作区会被删除，`open_author_revision_results.json` 保存的是指标/事件数，不包含完整模型/交互/训练前缀。下一轮至少复用已有归档/checkpoint，保存实际源/修订/确认的代码、付费数据、依赖与事件。没必要为固定模板既有结果重新复制整仓库。

## 7. 建议的最小顺序与验收

1. **先修两个真实接口问题**：训练文件可见性、AST 规范化/准入。验收直接复用本报告三个小脚本，不扩成无限程序安全项目。
2. **修正主张及开放链公平控制**：说明 scripted provider；删除无条件 extra probes，让 ordinary 先测试现有数据；保留原数据并新增控制版本。不要覆盖 24/24 的历史证据。
3. **最小真实开放作者实验**：复用 v10/CodeAgent 已有在线作者或真正组合式作者，至少一条 witness 的下一结构由反馈决定，且结果不在冻结时的预写 candidate_specs；固定先验、保存调用/修订历史、独立确认、保留旧源、后继迁移。无需先拥有无限 Python 语法。
4. **然后重算受影响 M3 图与检索对照**：规范化修正后 source 归组/横边发生变化，应把图差额与 solver/fee schedule 差额分开。

## 8. 证据与复现

本目录为独立诊断工件，不修改仓库。所有 probe 都没有网络或付费模型调用。

| 脚本 | 结果 | 内容 |
|---|---|---|
| `probe_closure.py` | `closure_probe_results.json` | AST 两类新碰撞、函数参数/文件训练对照、完整 CodeAgent 假确认、provider 诊断数据流、receipt 可选绑定 |
| `probe_open_forced_queries.py` | `open_forced_queries_results.json` | seed 5026101 普通臂仅删除强制两点；全部其他流程保持；完整本地工作区保留 |
| `probe_m3_ast_rejection.py` | `m3_ast_rejection_results.json` | 原 M3 graph 的九节点列表；同 seed 原 S01/S02 source 的误拒重放及完整模型 |

执行示例（从本审计父工作区）：

```bash
PYTHONPATH='/workspace/scratch/3f14ad2da0ed/repo_m3_612d8eea5/Hegel Machine/src' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 audit_m3_612d8eea5/closure/probe_closure.py
PYTHONPATH='/workspace/scratch/3f14ad2da0ed/repo_m3_612d8eea5/Hegel Machine/src' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 audit_m3_612d8eea5/closure/probe_open_forced_queries.py
PYTHONPATH='/workspace/scratch/3f14ad2da0ed/repo_m3_612d8eea5/Hegel Machine/src' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 audit_m3_612d8eea5/closure/probe_m3_ast_rejection.py
```

阅读覆盖：AGENTS/reuse hub 与 historical open-grammar 查询；旧 F1–F11 原报告和原证据；新 `open_python_revision.py` 全文、V16.9 研究脚本/测试/报告全文；三个既有 M2 模块的全部差异及关键消费者；新增 M3 AST 生成/准入/原 source 数据。大套件与全 M3 重放由主审计和结果审计负责，此处没有重复跑。报告不声称重新阅读/运行了整个项目的全部历史版本与全部参考文献。
