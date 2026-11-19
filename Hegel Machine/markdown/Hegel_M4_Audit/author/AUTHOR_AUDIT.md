# M4 开放组合作者专项审计：11961f702

审计对象固定为 `11961f702739ab272661ff59b77fb5bc8e3589b8`，对照前版 `612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8`。本专项只判断 M4 支柱一及其与 CodeAgent、归档、确认和后继迁移的实际接线；M3.1 通用修复、多尺度图和生命周期由其他专项复核。

## 1. 结论

**这版有实质进步：已从旧版固定完整候选公式，推进到有限语法中的真实残差驱动表达式组合。应当承认这个里程碑。** 原生 CodeAgent 的 `run` 返回会改变候选 AST，Haldane+shunt 路径确实有两轮诊断；持久化 AST 方法对象也确实改变后继运行。离线确定性 provider 本身不构成失败理由，旧审计 §10.2 也明确允许有限、可审计的语法。

但报告“完整满足六项判据”“全面合流与收官”的表述超出证据。最有实质影响的三个问题是：

1. **开放作者实验的图收益尚未成立。** Hegel 被直接交付手选正确的一个历史 AST，普通组把同一 AST 排在六个历史候选的最后；这一比较没有调用森林/虫洞索引。只把普通组完全相同的六个历史对象倒序，就从 6 次评估、52 算力、42.6 复合成本，变成 1 次评估、22 算力、41.1 复合成本，完全追平 Hegel。
2. **通用二轮组合存在真实参数别名错误。** `oscillatory_damping_modulation` 之后再加旁路，`_max_param_idx` 漏遍历 `cos_mul.param`，给旁路重复分配振荡频率的 `p2`。语法生成结果把两个独立参数绑死，仍被标记 `VERIFIED_PURE_AST`。
3. **历史证书、Python 文件和森林生命周期尚未与此作者的 AST 迁移链合流。** 实际传递的是 JSON 中的 `operator_spec`，且不检查其确认状态、摘要或来源权限。这可以证明“真实 AST 方法对象可重用”，不能等同于“已认证的作者程序通过森林检索和权限闸门被继承”。

“旧域零退化”需要另行校准措辞：当前验证的是**可重拟合机制族的保守扩张**，这是一项有意义的成功；它没有验证同一份 witness 拟合状态同时保留 S1 数值律。不同物理实例本来可以有不同参数，不能据此否定跨域一般化。

## 2. 阅读与实验范围

已完整阅读 `m4_compositional_discovery.py` 1337 行、M4 新测试 163 行、报告 14 的 70 行、运行脚本及作者归档部分；逐一检查 10 个持久化工作区的 40 个文件。相关接口阅读覆盖 `OpenWorkspaceCodeLabAdapter`、CodeAgent 的 `run/evaluate/finish`、`evaluate_python_model`、`_freeze_confirmed_submission`、CostMeter 和实验台采集计费。另读旧 `markdown/hegel_M3_Audit/M3_AUDIT.md` §10.2 六项原文。

遵循 AGENTS 历史复用要求，已读当前 AGENTS、复用枢纽相关段落，并运行 historical 查询。复合关键词无直接命中；`--query open --compact` 返回 11 个历史版本与两条相关复用路线，其中已有 `open_grammar_author.py`、`python_evidence_repair`、既有 CodeAgent 在线 provider 和 `sensor_role_calibration.py`。因此本报告提出的接线和账本建议是复用已有能力，不是断言项目从来没有这些能力。

本专项没有运行整套研究，只有四个目标 episode、三组旧域对照和不调用实验台的 AST/诊断检查；源码只读，全部输出位于本目录。父审计负责完整独立重放。

可携带证据：

- `probe_author.py`：完整离线反事实脚本。
- `probe_run/probe_results.json`：全部结果、10 工作区清单、逐条组合触发和代码一致性记录。
- `probe_run_stdout.json`：便于直接阅读的结果摘要。
- `probe_run/` 下保留各小控制工作区及构造的 AST/代码对象。

## 3. 应确认的真实进步

### 3.1 实际存在反馈驱动的结构组合

源码的原子机制是表达式树：`make_atomic_operator_spec`（110–201 行），六个结构组合分支在 224–341 行，编译器在 380–543 行。没有旧式“初始候选数组中已含五个完整答案”的事实。

诊断脚本在 551–684 行读取训练观测、重新拟合当前最优模型，输出残差/形状特征。`next_action` 在 831–840 行读取成功 `run` 的 stdout；748–814 行按它新增 AST；842–857 行按真实 evaluate 反馈更新最优候选和停止。底层 CodeAgent 确实用独立 Python 子进程执行这些步骤，见 `code_agent.py` 1235–1273 行。

**Haldane+shunt 的两轮是真实的。** 持久化 `W_haldane_shunt/authored_method.json` 记录：

| 次序 | 实际诊断 | 新结构 |
|---|---|---|
| 1 | turnover=true，secondary_active=false，residual_rmse=0.516437 | `michaelis__comb_quad_inh`，摘要 `e0f4e45e60c3d3` |
| 2 | turnover=false，secondary_active=true，corr=0.809，residual_rmse=0.215576 | 上一轮 Haldane 再加 shunt，摘要 `c2ced29ba2c345` |

`_best_base_spec` 会随着 evaluate 成功更新，`additive_secondary_shunt` 施加在该当前最优结构上（788–790、850–852 行），不是所有“多轮”都从同一个原子重新开始。报告举的这一个迭代实例成立。

最小 stdout 反事实也成立：相同 provider 初态喂 `{}` 时新增 0 个表达式，改成 turnover=true、secondary_active=true 时新增 4 个互异表达式。这个检查只验证解析器因果性；完整真实 stdout 路径同时由已存两轮触发记录和原生 CodeAgent 接线支持。

### 3.2 实际保存了可执行对象和独立训练前缀

10/10 工作区各有 `authored_model.py`、`authored_method.json`、`observations.json`、`confirmed_training_observations.json`。

逐一检查结果：10/10 保存的 Python 源码与其 AST 在相应角色/透镜下重编译结果逐字相同；10/10 模型 SHA 和 AST 摘要与 JSON 一致；10/10 observations 与 confirmed training prefix 相同。源 S1 前缀 4 点，其余 9 个工作区均为 6 点。所有作者模型仅导入 `math`，没有被遗漏的本地依赖模块，因此这 10 份归档可以离线执行。

确认与 heldout 的两个标准观测文件只写冻结 prefix；heldout 标签不被传给 provider，传给模型子进程的预测输入也不含 target。`evaluate_isolated_python_model`（53–103 行）与 episode 冻结流程（968–1012 行）确实区别于旧版把确认标签留在 cwd 的路径。未发现这五个成功样本是通过读取 heldout 标签得到的。

### 3.3 持久化 AST 方法对象具备因果作用

正式 loader 在 1244–1260 行读取磁盘 JSON 的 `operator_spec`，provider 会编译、拟合和执行这个对象。因此它不是仅被打印的 schema 名称。

本次内容干预进一步定位了真正的依赖：

| 控制 | 原 Python 文件 | JSON AST | 结果 |
|---|---|---|---|
| 原归档有效 AST | 存在 | Haldane+shunt | 1 evaluate，22 算力，41.1 复合 |
| 只有 AST，确认=false、摘要错误、无源码/receipt | 不存在 | Haldane+shunt | 1 evaluate，22 算力，41.1 复合 |
| 只改 JSON AST，Python 全字节保持正确原版 | 正确且未改 | linear | 4 evaluate，触发诊断，46 算力，42.3 复合 |

最后一项与正式整文件删除控制相同；说明后继行为依赖真实 AST 内容。按旧 §10.2 的“方法/代码对象”表述，**R1 的基本方法对象因果性成立**。不应仅因没有直接执行历史 Python 文件就把它判成失败。需要纠正的是对象类型、认证接线和“森林检索”的表述。

## 4. 已复现的问题与证据边界

### A1：排序先验完全解释本例 57.7% 算力优势

源代码 1205–1207 行显式复制已知 Haldane+shunt 文件给 Hegel；1272–1278 行普通组列表按 S1、cap、shunt、ringing、fast、Haldane+shunt 排列。两者都是“首个拟合达标即结束”（855–857 行）。这里没有 `PhilosophicalForest` 实例、没有 index.search、没有真实历史检索或打分；本模块导入的森林、契约和许可函数未调用。

控制固定同一个目标函数、同一六点 pilot、同一两点 confirmation、同一五点 heldout、同一六个历史表达式的 multiset，只将普通组列表反转：

| 普通组历史顺序 | evaluate | diagnostic | 测量 | 算力 | 复合 | strict |
|---|---:|---|---:|---:|---:|---|
| 提交原序 | 6 | false | 40 | 52 | 42.6 | true |
| 直接倒序 | 1 | false | 40 | 22 | 41.1 | true |

该控制没有证明倒序是普适最优检索器；它严格证明本例“森林优于强普通全历史”的归因不成立。正确项的先验排序已足以消除全部优势。报告可保留“已知正确历史 AST 节省再发现成本”，但图收益需要真实同内容检索器比较和顺序随机化。还应写清，52→22 是算法化算力账本降低 57.7%，复合成本 42.6→41.1 只降低 3.52%，测量完全相同。

### A2：振荡节点之后再次组合会产生参数编号冲突

`_max_param_idx`（204–211 行）只遍历 left/right/arg/base/num/den/a/b，遗漏了 302–304 行 `cos_mul` 保存频率的 `param` 子树。`_collect_params` 同样遗漏（214–221 行），编译器另写 `_walk_cos` 补救搜集，却没有补救新参数分配。

对一个完全合法的操作序列：

```
exp_relax_2d → oscillatory_damping_modulation → additive_secondary_shunt
```

第二步后的最大参数实际是 2，函数却返回 1，第三步再次分配 p2。生成表达式为：

```
p0 * u * (1 - exp(-t / p1) * cos(p2 * t)) + p2 * t
```

预期应有独立频率和旁路系数四个参数；实际只有三个，频率与旁路系数绑在一起，且两处 p2 元信息的边界/初值互相覆盖。结果仍得到 `VERIFIED_PURE_AST`。纯 AST 状态本来只证明约束语法可接受，不证明参数命名和科学模型语义正确。

这不推翻现有五例通过，因为其振荡候选后没有再次添加新参数；它直接阻止把“这个两轮例子成功”推广成“这些组合子可任意安全迭代”。修复可集中在一个统一递归子节点遍历，覆盖 `param/var`，并对重复 idx 的冲突元信息作验证。增加这一条有具体失败见证的回归测试即可，不需要扩建整个框架。

### A3：当前归档证明的是 AST 程序继承，尚未证明认证对象进入森林

1247–1248 行只读取 `loaded['operator_spec']`，不消费 `confirmed_ok`、model SHA、AST 摘要、source version、契约、训练前缀或 receipt。1033–1041 行确实构造了 `frozen_submission`，但它是 episode 返回值，随后在研究脚本摘要中被丢弃；被复制归档的四个文件也没有完整授权 receipt。实际目标仍做新鲜确认，所以这不等于已在现有样本中造成假确认，问题是“源历史认证链”和本次作者迁移没有接上。

Pillar2 只从本模块导入 AST constructors/compiler/evaluator（`m4_multiscale_renormalization.py` 38–44 行）；并没有调用 `OpenCompositionalCodeProvider` 或本作者 episode。共享编译器与实验台基础设施是合理进展，“发现产物进入多尺度森林并在后续生命周期中被实际检索”则需要额外证据。

保留 AST 为规范方法对象完全合理。建议把其内容摘要、来源版本、已确认作用域和依赖组成标准 artifact，沿既有森林注册和授权入口接入；若 Python 只是编译产物，报告应直接说明，避免让保存源码与源码被消费混为一谈。

### A4：旧域保持证明了 family nesting，未证明冻结拟合状态保持

1150–1158 行为每个 child 准备 6 个旧 S1 带标签样本，并将它们作为新的 fit prefix，再评估两个旧域点。可重拟合 family 包含原 Monod 是成立的：抑制项可取 0、shunt 可取 0、cap 可升到不激活区。这正是结构一般化能保留旧模型族的一个有效意义。

为区分层级，本次把同一 witness 的已冻结训练前缀保持不变，再评估报告相同两个旧域点：

| child | 冻结 witness 训练前缀时旧 S1 RMSE | 改用旧 S1 标签重拟合后的 RMSE | 阈值 |
|---|---:|---:|---:|
| Haldane+shunt | 0.716788556 | 8.18e-10 | 0.045 |
| cap | 0.232776623 | 0 | 0.045 |
| shunt | 0.145234268 | 0 | 0.045 |

这些差异由 fixture 中本来不同的物理参数/机制导致，不是“所有跨域模型必须共享参数”的失败证据。应将字段改成 `source_family_refit_compatible` 一类明确层级，并另存原实例模型及作用域；若主张同一系统知识修订后持续保持旧适用域，则需要条件化选择/契约回归测试。当前函数没有执行 docstring 所说的 ALIGN exclusions，也没有修改森林源节点契约。

### A5：映射由宿主提供，未见任务的强度需如实标注

后继目标确实换了系数、变量名、字段顺序和读数仿射透镜，但精确 `target_lens={-1,1.25,14}` 在 1242 行直接提供，正确语义角色顺序在 1255/1285 行直接提供。所有组得到相同信息，所以不是这组相对比较的单边泄漏；但它验证的是 **given-map transfer**，没有验证作者从实验数据识别变量角色或传感器变换。现有 `_estimate_observation_lens`/历史 sensor-role 能力可以接线复用。

五个 witness 的目标和组合子是同一有限 fixture 家族；后继仍为 Haldane+shunt，仅换参数和外部坐标。确认每例 2 点、heldout 每例 5 点；生化 heldout 最远 s=11.0，训练最大 10.5，后继 drive=11.2，训练最大 10.8。没有本作者的机制族留一、协议留一、时间段外推或不匹配机制 OOS 稳健性证据。它们支持有限机制、轻度数值外推和已知映射迁移，不支持全面分布外发现。

没有发现 deterministic provider 按 task_id/surface_domain 分支选公式：它不读取 request 的这些字段；其结构判断来自训练观测和残差。宿主给定的组合子/诊断策略仍是明显的归纳偏置，但有限语法本身为旧验收条件所允许，不能因为语法有限就直接否定进展。

### A6：费用和事件归档尚不足以支持“全生命周期作者收益”

本作者的 compute 是 `4+6*N_evaluate+3*N_diagnostic` 再加实验台采集计算费（1014–1019 行）；这是一种清楚、可复算的 work-price，不是实测所有拟合迭代、残差计算或编译工作。CodeAgent 已记录的 execution time、candidate-row scores 和完整 events 没被保存到 episode 的结果，只有 `agent_step_count`（1057 行）。`wall_clock_seconds` 仅为确认与 heldout 两次 subprocess wall time（1022–1024 行），没有研究期间 fit/run/evaluate 用时，不能称为整条作者链墙钟时间。

源码中的固定 usage=550 tokens 是确定性 provider 的接口填充值（817–824 行），不能冒充实际 LLM tokens；用户和报告已经说明本轮零在线 LLM，故这里不要求商业模型调用，也不把 lab meter 的 0 model_calls 判成漏付 API 钱。

研究脚本 88–92 行只复制四类文件，作者事件未归档；234–273 行将 comp_suite 压缩成摘要，丢掉源形成成本、完整候选评估轨迹和 frozen_submission。现有摘要能核实数值结果，尚不足以核算开放作者自身的形成成本何时被后继节省摊平。应复用 CodeAgent 的现有事件/成本导出，而非另造日志系统。

## 5. 对旧 §10.2 六条件的逐项判定

| 条件 | 本作者专项判定 | 依据/下一步 |
|---|---|---|
| R1 历史产物实际作用 | **有限范围通过** | 持久化 AST 方法对象可删、可恢复、内容干预改变后继；应明示规范对象是 AST，认证和森林接线未闭环。 |
| R2 反例驱动结构变化 | **五个 fixture 范围通过** | 真实 stdout 改变候选，Haldane→shunt 的两轮真实；需要修参数冲突，不能把通过范围外推为任意复合。 |
| R3 旧知识层级明确并保留 | **部分通过，表述需收口** | family 可重拟合兼容已证实；固定实例、条件化原作用域保持及 ALIGN 契约更新未证实。 |
| R4 未见任务上改进 | **部分通过** | 新系数/新字段/给定透镜下可迁移，少量独立留出；缺少机制、协议、范围、时间维度的严格切分。 |
| R5 全部必要费用与持续曲线 | **作者支柱未完成** | work-price 明确且可复算；源形成、作者试错、完整研究时间及累计摊销未进入本支柱对照。其他多尺度账本由独立专项判断。 |
| R6 超过同内容强基线的图收益 | **本支柱不通过** | 没运行图检索；普通列表倒序完全消除差距。多尺度七臂的独立图证据不能自动填补开放作者的合流缺口。 |

因此不能写“六项完整通过”。合理里程碑表述是：**有限组合语法作者、真实反馈分叉、持久化 AST 方法复用和独立确认已落地；需完成参数正确性、历史身份/契约接线与强基线归因，才能把它作为森林驱动持续研究的合流证据。**

## 6. 建议的最小收口顺序

1. 修复统一 AST 参数遍历和冲突检测，用 oscillation→shunt 的失败树作回归，保留当前五例成功轨迹。
2. 以 AST 为正式方法对象接入现有 artifact/forest/receipt 入口；从一次 witness 的实际作者输出生成一个新节点，再让下一目标通过真实索引检索并加载它，保存授权理由及事件。
3. 普通全历史、Flat、Forest 在同一 artifact 集合、同一诊断/角色工具、同一确认门下运行；至少随机化插入顺序，不再手选正确对象给 Hegel。首先允许 Flat/普通方法库强到能正确排序，再判断剩余图收益。
4. 将旧域结果分成“原实例已拟合模型仍保留”“新族可重拟合包含旧族”“新机制适用域契约”三层；不强行要求跨领域参数相同。
5. 保存已有 CodeAgent events、实际研究 Python 时间、源形成账本、AST/代码/训练前缀/证书摘要；在少量留一组合目标上核算累计同质量成本。无需立刻加大真实数据规模或增加大量在线调用。

本次未修改仓库源码，也未调用在线商业模型或执行远程写入。
