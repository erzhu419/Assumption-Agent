# Hegel V17.3 M4.1 独立审计与下一步建议

审计日期：2026 年 11 月 20 日。固定提交：`99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c`；比较基线：`11961f702739ab272661ff59b77fb5bc8e3589b8`。

**结论：M4.1 有实质进展，已打通作者产物进入森林并被后继任务取回的主链，也实现了真实的快慢动力学与绝热减阶。但“11 项全部闭合”“三支柱已成为同一条持续对象链”“成本差已充分归因于图结构”“完整生命周期已经回本”都超过了当前证据。建议保留架构，先做一轮范围明确的接线和口径修复，再扩大在线 M5。**

最值得保留的结果是：在当前有限算子语法、固定任务分布与费用模型下，历史发现确实产生可执行的新复合族，森林能从已付观测线索取回它们；图边也确实影响可达性。这一结论经过独立重放。最需要补齐的是：合法契约必须实际控制执行，回退分支必须共享已经付费的证据，已验证的作用域必须跟随模型到最终调用。

## 一 审计范围与可复现性

### 提交与材料

固定提交树有 13,931 个文件，本轮相对基线有 941 条变更：9 个源码文件、1 个研究入口、1 个测试文件、4 个其他文档文件、53 个新证据文件，以及 873 个上一轮审计归档文件。GitHub compare 返回的前 300 个条目不是完整变更清单，因此本轮使用本地 git 树与差分核对范围。[S1]

本轮逐路径检查新增核心模块、全部变更代码和测试、工作总结、相关历史模块与证据；使用历史能力目录确认已有预算、程序执行、Python 档案与研究续接能力。未改动的历史证据沿用上一轮核验，不把“完成提交树盘点”写成“重新逐字阅读全部历史文件和参考文献”。当前审计结论主要依赖固定源码和独立执行，不用参考文献代替实现证据。

### 独立复现结果

| 项目 | 本轮核验结果 |
| --- | --- |
| M0 至 M4.1 指定回归 | 33 项全部通过，进程退出码 0 |
| 完整四部分 study | 成功完成，278.376 秒；提交记录为 195.909 秒，环境负载不同 |
| Part 1 至 Part 4 四份 JSON | 与提交中的结构和值完全相同 |
| 336 条逐目标 JSONL | 与提交中的全部记录完全相同 |
| summary.json | 仅 elapsed_wall_seconds 不同 |
| 新证据 SHA256 清单 | 52 条列项全部匹配；目录共 53 个文件，包含清单自身；不是文档所写的 55 条 |
| 旧边界探针 | 四份探针原样复制到独立目录运行，均退出 0；旧 harness 中内嵌旧 loader 的结果另行解释 |
| 仓库状态 | HEAD 固定，源工作树 clean；没有修改源码或原证据，没有外部模型调用 |

这排除了“报告中的数字重放不出来”这一担忧。下面的问题集中在数字代表什么，以及验证条件是否进入真实消费者。原始数据、重放日志、逐字段比较、控制脚本和正反结果均在本审计包中。

## 二 上一轮哪些问题确实修好了

| 上轮问题 | M4.1 的实际改进 | 当前判断 |
| --- | --- | --- |
| 双 predict、math.exp 顶层遮蔽 | 旧反例被标为不支持，摘要也不再碰撞 | 原见证已闭合 |
| 无 receipt 又省略 target_id 仍可放行 | 两种调用均拒绝最终指导 | 已闭合 |
| 工作区内绝对模型路径隔离失效 | 原相对和绝对路径控制都得到 RMSE 9 | 原见证已闭合，相对越界仍有小缺口 |
| 一次失败广播四个高负载 flag | 现在最多设置一个对应 flag | 广播错误修好；唯一机理仍属诊断假设 |
| osc 加 shunt 参数 p2 碰撞 | 真实得到 p0、p1、p2、p3，保留多起点 | 已闭合 |
| 作者 JSON 仅有 metadata 也能取 spec | 检查 source、SHA、AST、重编译 AST 一致性 | 完整性改善真实，来源和节点绑定仍不足 |
| 任意 program_steps 非空即可召回 | 五个旧 garbage 控制全部取消召回 | 非法指令拦截修好，合法参数未驱动执行 |
| child 确认跳过最终授权 | 权限上限和未知硬前提都取消 recall | 原旁路已闭合 |
| fidelity 验证对象未提交 | merged claim 现在真正提交，异律重叠拒绝 | 持久化修好，析取语义消费者未接通 |
| 协议修订使 coarse edge 意外丢失 | 纯协议修订维持 6 条，真正变更可标 stale | 版本维护修好，重绑 API 不等于重新验证 |
| M3.1 四格少一格 | Alpha0_NoPool 已加入 | 四格完整，统计列表却重复追加 walk 题 |

因此，不应继续把 M4.1 描述成上一轮完全未改。也不应以这些旧见证通过，推导出相关功能已经实现全部执行语义。详细证据见包内 `closure/CLOSURE_AUDIT.md` 与对应原样探针、新控制。[S2] · [S3] · [S4]

## 三 首要执行问题是契约检查与最终行为不一致

### 1 合法 program_steps 仍然不决定运行

`_validate_and_interpret_m4_program_steps` 返回 `fresh_points`、`calibrate_lens`、`verify_scale_closure` 等计划字段。旧 M4 把它存为 `_exec_plan` 后没有消费者；新 Part 4 只检查 plan 不为 None 后丢弃；Part 3 也一样。三个消费者仍使用硬编码的拟合与确认流程。[S3] · [S5]

在旧 M4 evaluator 的同一 `C0_T01_AdiabaticMacro` 目标上，决定性控制仅把合法 `subprocess_isolated_confirm.fresh_points` 从 1 改为 2。同一森林、同一数据，两次全部采样序列、费用、成功结果都相同，实际都只确认 1 点。新 Part 3、Part 4 丢弃计划字段的问题另由源码确认。正常节点本来就有 `fresh_points=2`，这不是任意恶意 Python 构造。

当前实现应称为步骤格式与一致性校验器。若把它称为可执行研究程序解释器，就必须让合法字段改变实际调用。最小修法是让现有操作函数消费 validated plan；如果标定和闭合其实总是强制步骤，应明确写成固定协议，移除无效的可配置字段。无需为此实现通用编程语言。

### 2 FidelityCompression 提交了 OR，但授权器执行 AND

本轮的提交修复是正确的：验证后的 merged claim 确实进入 keeper。新问题在后续消费者。合并器同时保留 `disjunctive_branches`，并把两组前提合在顶层；真实 `authorize_abstraction_use` 只逐项检查顶层前提，不解释 OR branches。[S4]

使用完全相同的预测定律，作用域分别为 `audit_scope=A` 与 `audit_scope=B`：合并前两域各自都可进行 bounded trial。合并后 fidelity 返回 accepted，自家 branch evaluator 在 A、B 都返回 ACC，冗余节点已归档；但真实授权器对 A、B 全部拒绝。它看到的是 `scope=A AND scope=B`。

这属于实际知识可用性损失。当前 Part 4 的三个压缩对象恰好具有相同前提，所以并未推翻那 48 个域内成功结果；它推翻的是“一般作用域析取已经保真闭合”的主张。

有两种合理的小修：让验证器、授权器和路由器共用同一契约解释逻辑；或者本阶段只合并等价前提的副本，遇到需要析取的组合明确拒绝。后者完全可以作为阶段性实现，不必为了这一版扩大语义内核。

### 3 区域盒检查没有覆盖回退与最终调用

Part 3 的 `out_of_box_query_rejected=True` 来自对 helper 的独立布尔测试。真实 solver 只在 recall 分支检查第一个确认点，fallback 没有相同检查，最终预测也没有逐 query 的范围约束。[S5]

本组控制使用的绝热 Lab 的 `declared_box` 与证书 `verified_input_box` 都是 u 在 0.4 至 3.0、T 在 2.2 至 5.2；没有另外声明的 feature_bounds。以下控制明确在该盒外，不冒充原域内失败：

| 控制 | 实际结果 |
| --- | --- |
| 原训练、证书与终测不变，只把确认时间改为 T=1.2 | recall 被局部盒检查拒绝；fallback 随后通过确认，返回 completed、confirmed_ok=True |
| 原训练与确认不变，只把最终终测 query（Lab 的 heldout_points）改为 u=1.5、T=0.12 | 仍 completed；外部终测误差 0.275920，false_confirmation=True，未提前拒绝或升级 |
| 确认时间改为 T=0.12 | fallback 数值误差超过阈值，返回 unknown；该阴性结果保留 |

原盒内四点测试仍然通过。这里的缺陷是“接口宣称拒绝盒外调用，实际未统一执行”。最小修复是把已验证盒绑定到模型调用契约，让 recall、fallback、导出 predictor 共用范围检查。越界时可拒绝、重新验证局部盒或转到细尺度，不要求永远拒绝，也不要求有限采样证明任意黑盒全域正确。

## 四 作者产物的完整性与知识身份还差两处连接

### 1 有效 artifact 可以与已注册节点脱节

`_load_node_verified_operator_spec` 核验路径中的 artifact 自身后，直接取其 spec，没有比较它与节点注册的 `ast_skeleton_digest`。把 W1 节点所指的独立副本替换为提交内另一个完全有效的 W2 artifact，loader 接受 W2，但传入节点对象仍保留 W1 的 operator_spec 与预期 digest；该加载器也没有更新真实森林的节点版本、签名或前提。[S5]

控制中的两个 digest 分别为 `c2ced29ba2c345` 与 `17d05c4bf44ecf`。这不是 malformed JSON 攻击，而是有效对象替换后，知识身份与实际程序不一致。当前独立确认仍可能发现模型错配，因此本审计不声称此替换已经造成错误最终确认。

最小修复就是比较 verified artifact digest 与注册对象 expected digest，并把 artifact 身份纳入节点版本和执行收据。

### 2 confirmation_digest 当前只检查非空

新 loader 能检查 source、spec、SHA、AST 的内部一致性，这有价值；但确认摘要只需要非空字符串。控制使用现有编译器生成正常模型，未运行任何确认，只填 `confirmed_ok=True` 和 `confirmation_digest="no_confirmation_was_run"`，仍被接受。新回归自身也用固定占位摘要构造有效 artifact。[S2] · [S6]

这意味着“内部完整性”尚不等于“确认来源已验证”。实际 Stage 1 的可信宿主确实执行了确认，原 336 条数据并未因此失效。进入 M5 后，应让宿主写入候选程序不可改的确认 manifest，绑定模型 hash、训练前缀 hash、确认观测 ID、结果、容差和版本；loader 查验这些记录即可，不必部署复杂签名基础设施。

### 3 模型路径还存在一个容易修的小缺口

M2/M3 的路径规范化对绝对、相对输入都做 containment；M4 的 `evaluate_isolated_python_model` 只规范化绝对路径，相对路径原样传递。只使用本审计自身目录里的常数模型，通过相对父路径可以执行临时工作区外的文件，返回码 0。没有读取用户其他文件，也没有据此声称原证据泄漏。[S2]

三个入口复用现有 normalize helper 即可。它适合顺手修掉，没必要扩展成任意 Python 安全证明任务。

## 五 成本收益需要先排除回退分支丢证据和重复购测

所有下列成本都是项目定义的抽象单位：`C = 测量费 + 0.05 × 计算代理费`。它们不是实际金钱或秒数。

### 1 粗粒化边有效，但原 34.7% 成本差大部分来自购测规则

首先确认正面事实：细 RK4 和粗 Schur 减阶来自同一套动力方程；15 点独立矩阵指数/闭式解核验的最大状态误差分别为 `4.98691e-7` 与 `3.18884e-8`。尺度选择使用实际配对粗细观测。粗粒化边确实进入 PPR，无直接 query signature 重合的 coarse 节点，有边时可达、删边后消失。[S5] · [S7]

成本的独立控制如下：

| 路径 | 测量费 | 计算代理费 | C | 候选数 | heldout RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原有粗粒化边 | 72 | 12.19 | 72.6095 | 1 | 0.008792 |
| 原删边 | 96 | 35.53 | 97.7765 | 5 | 0.008792 |
| 删边，复用已付 pilot，确认点数与当前 recall 一致 | 72 | 35.53 | 73.7765 | 5 | 0.008792 |

第三行保留原 solver、五候选、拟合器、终测和计算计价，只不强制增购 fallback pilot，并与当前 recall 同样确认一个点，仍严格通过。原 fallback 无条件多买五点，其中两点早已买过；又比 recall 多确认一点，额外测量费为 `5×4 + 1×4 = 24`。

因此原差额 `25.167` 中的 `24`，约 **95.36%**，来自这套可避免的购测规则。匹配后剩下 `1.167`，约为有边成本的 **1.61%**。边的候选回取价值依然成立；不能把全部 34.7% 都解释为不可替代的图结构省测收益。

此控制为了隔离费用原因，匹配的是当前 recall 的一点评估规则；真正修复 fresh_points 后，两臂应共同遵守声明的确认要求。

### 2 统一链的 Alpha0 对照丢弃了已付失败确认

在 Part 4 中，森林首个模型被高负载确认点否定后，`cf0_rows` 放入局部 updated_prefix。只有 child 确认成功才更新主 frozen_prefix；Alpha0 或 Flat 无法到达 child 时，fallback 回到旧 pilot，重新购买 stress 点，还重复买刚才的确认点。普通历史分支却把失败确认留在 frozen_prefix 中。[S5]

在**与提交索引边计数完全一致的 SameHistory 森林**，对原 `U_T09` 的 Alpha0 路径做最小控制：保留已付 `flow=7.9` 的失败确认，把下一次 fresh confirmation 留给未见的 `flow=6.4`。不加图边、不改候选集合、候选数仍为 5：

| U_T09 路径 | 观测数 | 测量费 | C | strict_pass |
| --- | ---: | ---: | ---: | --- |
| 提交中的 Alpha0 SameHistory | 10 | 50 | 52.5363 | True |
| 保留已付确认并避免重复购买 | 7 | 35 | 37.3113 | True |
| 提交中的 Hegel | 7 | 35 | 36.4487 | True |

该题全部 15 单位的物理测量差消失，Hegel 仍有候选数和总代理计算成本上的较小优势。这是已实现分支的证据续接问题；仓库历史本来就有 paid evidence 复用能力，应接回当前入口。

本控制只报告已实跑的一题，不把它冒充重跑后 48 题新均值。它已经足以说明当前 Hegel 对 Alpha0/Flat 的整段测量优势不能直接全部归因于图游走。

### 3 正反序基线比上一轮更强，但不足以消除排序解释

M4.1 的三机理均衡目标确实修正了上一轮“倒过来恰好把唯一正确模型放首位”的薄弱设计。原正序和逆序的结果都可复现。但“任何静态顺序数学上都无法绕过”不是由两种顺序推出的结论。[S8]

本审计保留全部 11 项历史，用整个评测期固定、与单个目标标签无关的顺序：Haldane+Shunt、Osc+Shunt、Murray 原子、Murray+Cap，然后其余旧历史原序。先试低载可辨识的 Murray 原子，再用失败确认区分 cap，是公开方法结构可以支持的排序。

| 同一主种子的 12 题 | 平均候选数 | 平均测量费 | 平均 C | 通过 |
| --- | ---: | ---: | ---: | --- |
| Hegel | 1.6667 | 35 | 36.3477 | 12/12 |
| 固定全历史排序控制 | 2.3333 | 35 | 36.4250 | 12/12 |
| 提交中的逆序 | 3.6667 | 36.6667 | 38.5167 | 12/12 |
| 提交中的正序 | 10.6667 | 35 | 38.9250 | 12/12 |

**应承认：这次 Hegel 仍然小胜，不能再说“简单倒序就打平”。** 但相对合理固定排序只领先 `0.0773`，约 **0.21%**，而不是以原正反序差额代表稳定的结构净收益。该排序是审计阶段形成的敏感性控制，尚非新独立盲测基线；它证明对排序敏感，不证明所有未来任务上某静态顺序都更好。

下一次可在 DEV 冻结一个合理排序/廉价重排器，再做全新 TEST；不必穷举排列，也不应给基线目标真值或免费答案。

## 六 M3.1 四格补齐成立，但重复题改变总体权重

`make_m3_1_factorized_walk_suite` 返回的 targets 已经是 std_targets 加 walk_targets；新入口又使用 targets 加 walk_targets。四个 walk 元组不仅名称相同，还是同一个对象；第二次还落到相同工作区路径。[S9] · [S10]

每格实际运行 124 次，但只有 **108 个唯一的 cluster_id/target 组合**。四格有 496 次调用、432 个不同的 arm/cluster/target 评测组合；T25–T28 的唯一实例是每格 16 个。独立捕获全部 496 条记录后，重复记录的每项结果和费用都相同。

| 检索格 | 原总均 C | 去重后均 C | 去重后测量费 | 去重后计算费 | 唯一 walk 胜出 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Walk Pool | 38.4009 | **37.4591** | 35.6481 | 36.2192 | **14/16** |
| Walk NoPool | 38.6065 | 37.6952 | 35.8333 | 37.2377 | 14/16 |
| Alpha0 Pool | 43.4016 | 40.3052 | 37.9630 | 46.8444 | 0/16 |
| Alpha0 NoPool | 43.3855 | 40.2867 | 37.9630 | 46.4741 | 0/16 |

四格各 108 个唯一实例仍全部严格通过。walk 子集的均费 `44.757875` 对 `64.3025` 不变，87.5% 胜出率不变；“28/32”应改为“14/16”，不能当作双倍独立证据。

总体效果变得更准确：固定 Pool 比较，原均费降幅约 11.52%，去重后约 **7.06%**。walk 的方向仍清楚，Pool 的收益仍依赖条件：walk 开启时略有利，alpha=0 时略有负担；不应写成无条件独立增益。T01–T06 原均费比较不受这次重复影响。

手工“计数透明说明”也需要改正：27 题是 23 个 standard 目标加 4 个 walk 目标，标准列表没有 T13；不存在另加四个独立 walk 后得到 31 个不同实例的解释。

## 七 已经打通的统一链与尚未打通的部分

### 作者发现进入森林这一连接成立

Stage 1 使用真实 CodeAgent 工具协议，确定性残差驱动 provider 动态合成三个新复合族；不是直接从目标 heldout 取答案。产物经过实际 loader，登记进同一 forest；Stage 3 的取回路径再从磁盘加载并检查 artifact。删除、恢复、metadata-only 控制对行为有影响。这是上一轮明确缺口的实际补齐。[S2] · [S5]

Stage 3 不传 lens_hint 和人工 role_names，研究侧按已购 pilot 做角色置换和仿射标定。Frozen 也复制完整 10 节点 Stage 0，未再手工删掉关键先验。应保留这些正面结论。

### Part 3 的可执行 Pi 尚未进入 Part 4 的持续森林

Part 4 已提交的 `stage2_lifecycle_governance.index_edge_counts.coarse_graining_edges` 是 **0**。它的活森林有 13 个 concept nodes、3 条 horizontal edges、3 条 vertical tree edges；Part 3 单独手工注册的 4 节点 ODE 森林和 Pi 边，没有进入 Part 4 的 Stage 1 至 Stage 3 链。[S5] · [S11]

因此准确的完成状态是“作者发现—入林—跨域迁移链成立，另有一个真实可执行重整化机制实验”。“三个支柱已经熔铸成同一条不断裂对象链”尚不成立。

同样，`apply_executable_pi_s_projection` 的尾窗投影可执行，但在线 coarse channel 不传 fine_trajectory，`window_projected_y` 没有下游消费者。在线粗模型使用的是同方程绝热减阶，尾窗投影是诊断支路。两者都可以保留，只需准确说明哪个结果已进入执行。

### R3 的第三层并没有实际运行 scope dispatcher

Tier 2 的“新复合族在旧数据上重新拟合仍能表达旧关系”成立。Tier 1 使用相同旧 prefix，不把新目标标签混进来，也是一种有效保留检查。但对本次这些带有 fit 的作者模型，`evaluate_isolated_python_model` 会重新调用 fit，故这并非字面上的冻结拟合参数零重拟合。[S2] · [S5]

原 S1 源码初始参数恰好就是旧真值 6、2，使该例难以区分“已冻结状态”与“默认值恰好正确”。W1 则可以明确区分：源码默认参数下高负载 RMSE 为 **0.65774565**；用旧 prefix fit 后恢复 8.8、2.1、0.14、0.76，RMSE 约 **3.21e-10**。

更关键的是，Tier 3 只是把 S1 与 W1 两次分别评估的布尔结果 AND 起来，没有让混合旧/新请求通过 forest dispatcher；没有路由事件，也没有选中节点记录。该检查还发生在新作者节点注册之前，当然也没有经历后面的压缩和修订。因此不能由它推出“旧域自动走旧实例，新域自动走新实例”。

最小验收是一组交错旧域、新域和边界请求，直接调用同一个 dispatcher，记录实际 node/version/parameter-state；旧域使用保存的旧拟合态，新域使用新拟合态。已有代码档案与参数保留机制可以复用。这也更直接对应项目所说的历史知识在扬弃中被保存。

## 八 生命周期与账本还不足以支持强长期收益结论

### 1 发现与维护投资尚未在主 12 题内回本

提交账本为：Stage 0 为 186，Stage 1 为 157.7，Stage 2 为 167.7，Hegel 的 12 题 Stage 3 为 436.1724，总计 **947.5724**。Frozen 的相同 Stage 0 加 12 题约 **763.2760**。因此在这 12 题窗口，Hegel 按该名义账本仍多花 **184.2964**。[S11]

如果假设源发现和维护只支付一次、未来题目构成和单题费用保持相同、不追加维护、先只看域内题，那么相对 Frozen 的名义回本点约为 **第 28 题**。相对已经拥有相同发现历史的逆序普通臂，公共 Stage 1 成本应相消，新增维护 167.7 按每题节省 2.16897 推算，约需 **78 题**。这是带假设的摊销计算，不是已观察到的长期学习曲线；修正回退策略或基线后还需重新估算。

不能把 Hegel 的全生命周期 947.57，与 Frozen 仅 Stage 3 的 577.28 并列后就说已经显示净优势；两边窗口与共同成本必须一致。

### 2 当前账本混合了实测事件和手工定价

Stage 0 的形成费是源码直接填写的常数，不是本 study 运行形成过程累计的计量。Stage 2 的 DEV 数据是宿主按固定公式生成，然后手工加上点数乘单价；并未走共享 Lab.acquire 形成逐观测 paid receipt。维护额外计算 18 也是常量。[S5]

这些预算估计可以用于研究模型，但应标明。计算费还把候选视为固定 6 单位，没有完整计入内部标定、多起点 least-squares、每次隔离 evaluate 重做 fit 的事件数。Part 3 所谓“1 次拟合”实际包括 2 次 lens least-squares 和 3 次模型 fit。它目前是候选数，不是优化调用数。

建议并列保留三层：物理观测及唯一 observation ID、抽象代理费用、实际调用次数/耗时/token。外部终测成本可作为 evaluator 成本单列，没必要强行混进研究预算。

### 3 DEV 诱饵现在真实执行，但不是三份独立现场

两种诱饵确实在隔离子进程拟合并确认失败，改进了上一轮直接灌入失败标签的做法。但三轮使用完全相同的训练点、参数和确认点，dev_i 只改变文件夹名，失败误差三轮分别固定为 1.04962 和 0.49753。[S5] · [S11]

这支持“真实失败能触发归档规则”，尚不支持“跨独立现场的效用统计已稳定”。若三次是同一确定性证据，应去重；若要估计跨现场负效用，应使用不同已冻结 DEV 实例并记录其来源。

后续三个 target seeds 重新加载同一 Stage 1 artifact，并手工重建含三个复合节点及相关边的森林，没有重放 Stage 2 的压缩、DEV 淘汰和修订，也不是从维护后的 forest 快照原样恢复。handoff 已披露一次发现、多种子迁移，应继续保持这个口径；不要把它升级成四条独立持续学习链。现有结果也缺少同信息量、同历史下“不压缩”与“保真压缩”的后继质量与维护成本比较，故不能把零遗忘归因于压缩本身。

Frozen 的 fallback 也没有重新调用作者从原子发现复合式，而是直接枚举包含三个 Stage 1 最终复合式的共同候选表。它可以检验“索引记忆与共同候选库搜索”的差异，不能把这份 fallback 代理费等同于开放作者重新发现的实测成本。当前所谓 chronological all-history 列表还是宿主手写，未收齐全部中间失败稿；建议从真实历史注册器导出，避免叙述超出列表内容。这不是对原 Frozen 起点公平性修复的否定。

## 九 还应更正的指标与证据标签

### Ordinary Forward 的召回率漏记

普通历史第一次确认成功会设置 confirmed_via_recall；第二轮从剩余历史成功确认，只设置 confirmed_ok，漏设前者。原 Forward 全部 16 个 SafeWalk 记录只有 pilot、ordinary_confirm_0、ordinary_confirm_1，没有 fallback，却被记为未召回。[S5]

按当前“通过历史模型完成即 recall”的定义，Forward 应是 **48/48**，不是 32/48。这不改变 strict_pass 或成本，也不会改变 Hegel 的 non_direct_walk 指标，但会改变“只有 Hegel 能完整复用历史”的叙述。Reversed 有真实 fallback，不应把它也自动改为 48/48。

### 其他可直接勘误的项目

- 区域探针实际为 3 对分位点，不是四角加中心；`Regional_5Pt_Box_Probe` 和相应 docstring 需要更名。handoff 一处写 3 点、一处仍写四角中心，应统一。
- `error_bound` 是采样最大差 0.009617，不是已证明的全盒上界；同一无 bump 平滑系统盒角差 0.039603，仍小于当前容差。有限采样限制已经披露，不再要求证明任意窄 spike 不存在。
- Gaussian bump 加在积分后读出 y_micro，没有改变状态 rhs；称局部读出失配更准确，不能扩大成发现新共振动力学。
- 336 行中主种子的 84 行缺少 seed 字段；其余 252 行有 seed。应补齐主种子标识，不影响当前重放。
- composed_rounds 使用 composed_history 长度，实际数的是候选数；W2 的三个候选对应两轮诊断，W3 的两个候选对应一轮，应分开标记。
- OOS 的函数与点位不依赖 seed。每臂 8 次 OOS 运行来自 2 个确定性模板的四次重复；“8/8 unknown”是准确运行计数，不是 8 种独立未知机制。
- 当前 JSONL 是逐目标结果和阶段成本记录，尚非包含 query、selected node/version、训练前缀、确认 digest、artifact 身份的完整执行 receipt。
- handoff 的完整 study 命令默认写回仓库证据目录；本次通过覆盖 EVID_DIR 输出到独立目录。建议入口增加 --out-dir，并把“只读安全”复现说明改成真正不覆盖原证据的命令。

这些属于可定位的指标和文档修正，不应全部提升为架构失败。

## 十 建议的有限修复与 M5 进入条件

### 第一批必须先闭合的五个接口

| 工作项 | 可复用位置 | 最小验收 |
| --- | --- | --- |
| 统一消费步骤、scope 和最终授权 | 现有 step validator、authorize_abstraction_use、ScaleMap gate | 合法 fresh_points 改动改变真实购测；越界在 recall/fallback/export 都同样处理 |
| 保真压缩与执行共用契约语义 | fidelity_compress_nodes、现有前提解释 | 同律 A/B 合并后两域真实授权与预测保持；或明确拒绝暂不支持的 OR 合并 |
| 保留所有已付观测并去重 | 当前 frozen_prefix、历史 paid trace 与 retained fit 路径 | failed confirmation 不丢失；相同 observation 不重复购买；各臂同样 fresh 规则 |
| 节点与 artifact、确认与宿主记录绑定 | verified loader、CodeAgent 档案和确认函数 | 有效 W2 不能悄悄替代 W1；空来源确认不能自行升级；相对路径统一规范化 |
| 修正统计与结果语义 | study 聚合入口、per-target records | 每格唯一 108；walk 14/16；Forward recall 48/48；各条记录可追到真实调用 |

这五项都是现有对象之间的连接工作。完成后重跑受影响控制和主 study 即可，不必重新遍历全部历史版本，也不必添加一大批只复述实现的测试。

### 要继续宣称统一 M4 里程碑，还需一个真正交叉的 episode

把 Part 3 已有的可执行 Pi 算子及实测证书封装为同一森林中可持久化、可版本化的对象，进入 Part 4 的同一 forest；后继题真正消费它，随后遇到一次作用域失配，更新证书/边版本，再让旧域回访通过真实 dispatcher 恢复旧有效模型。一次清楚、有完整对象 ID 的 episode，比再增加数百个同分布参数变体更能回答“多尺度、修订、压缩和历史迁移是否属于同一个系统”。

在这个 episode 完成前，可以开展 M5 的小规模接口试运行，但其名称与结论应限于当前已贯通的 authored-discovery/forest-transfer 链，不宣称可执行 Pi 持续学习也已完成。

### 在线后端应接入已有能力

仓库已有 `CodeProvider.next_action`、`CodexCodeProvider`、`CodeAgentConfig` 的步数/调用/token/执行预算，也有 `retain_evaluated_python_model` 与 paid trace 的历史保留路径。M5 的任务是把这些能力接到新入口并保持同一计量，不是从零再造 provider 或另起一套记忆系统。[S12]

在线试验建议遵循以下边界：

1. **模型提案，宿主确认。** LLM 可以写代码和提出作用域，但不能自行写 confirmed、closure_verified 或有效 receipt。采用什么模型不是当前接口漏洞的解决办法。
2. **允许探索表达范围，同时限制可复用主张。** 不能被当前 AST 规范化的数值代码仍可在隔离试验中评估、保留为不透明程序档案；在获得兼容语义证据前，不将其摘要碰巧相等当作同律、不自动参加等价池化。当前 finite DSL 可继续作为稳定通道。
3. **共享观察账本，分离拟合和确认角色。** 失败确认一旦转入训练，就不再充当下一候选的 fresh confirmation；跨臂复用规则一致。保存 query ID、channel、费用、prefix hash 和确认批次，避免通过重复拟合或反复看测试集获得未计量优势。
4. **先冻结 DEV 决策，再评独立 TEST。** 学习排序、归档阈值、噪声阈值及修改 provider 提示词都发生在 DEV。TEST 至少包含未曝光参数、不同数据来源、真实噪声/单位/测量通道及明确 OOS；不能只更换 RNG seed 而所有模板保持相同。
5. **选择能回答当前问题的外部任务。** 第一批适合可重复评估、测量成本与来源清楚、允许旧知识回访的任务。只给静态表格的基准可以检验表达/拟合和迁移，不能单独证明主动科学发现的实验选择收益。具体外部数据源接入时再冻结版本、来源和划分。
6. **报告完整分布与有限资源失败。** 成功率、错误接受、合理 unknown、累计费用、预算耗尽和超时分开；不把异常统一归为成功的 scoped_unknown。对同一起点、同工具、同数据权限的普通研究臂，给相同修复和重试机会。

这些要求直接服务于当前项目目标。它们不要求形式证明任意 Python、安全地自治无限预算，或在进入 M5 前补齐所有未来科学领域。

## 十一 对项目主线的判断

“新的假设来自已有概念与理论的总结和一般化”在本轮已经有更具体的实验对应：旧原子结构经过残差驱动组合，形成新的可执行族；这些族能进入历史结构，被后继任务按观测线索取回。这比独立脚本中摆放多个手工答案更接近原构思。

但 RSI 的更强命题还需要展示：经验修订了真实执行的方法与作用域；新知识在同一对象链中减少后续研究成本；旧知识不是靠分别重拟合的独立成功来象征性保留；扩大历史后仍有净收益。当前最重要的缺口都在这条因果链上，而不是是否已经使用最强在线模型。

我建议保留 M4.1 作为“真实整合进展、局部实验成立”的版本，撤回过强的全部闭合与不可绕过措辞。先修复有限的消费、证据续接和计数问题，再用一次统一的跨尺度修订与旧域回访，以及一组真正冻结的外部后继任务推进 M5。这样下一轮增加的是项目的可解释能力，而不会只是增加更多可复现但归因不充分的数字。

## 来源与包内证据索引

以下均固定到本轮提交。新增控制的原始数据在审计包中，原始阳性/阴性结果一并保留。

- **S1** 提交及本轮报告。范围：`change_scope.json`、`changed_paths.tsv`、`tree_99b1fdfeb_inventory.json`。
- **S2** `m4_compositional_discovery.py`。新控制：`closure/current_semantics_results.json`、`closure/relative_escape_result.json`。
- **S3** `m4_multiscale_renormalization.py`。步骤执行控制：`closure/probe_current_semantics.py`。
- **S4** `wormhole_forest.py`。作用域析取、版本重绑与真实授权控制同上。
- **S5** 新 `m4_1_unified_discovery_and_renormalization.py`，主体执行、费用、R3 与生命周期。
- **S6** 新 M4.1 回归测试。
- **S7** `hipporag_wormhole_index.py`，粗粒化边与 PPR。
- **S8** 报告 15，包含本次需修正的强主张。
- **S9** `run_v17_3_m4_1_unified_study.py`。重复计数：`part2/all_496_records.json`、`part2/deduplicated_results.json`。
- **S10** `m3_1_walk_factorization.py`。目标对象关系：`part2/suite_identity_inventory.json`。
- **S11** 提交中 Part 4 证据。账本与指标重算：`ledger_and_metric_review.json`。
- **S12** `code_agent.py`，历史 provider、预算与档案接口。历史查询保存在 `history_*.json`。
- 完整重放：`replay/pytest_33.log`、`replay/full_study.log`、`replay/comparison_to_committed.json`、`replay/evidence/`。
- 工程子报告：`closure/CLOSURE_AUDIT.md`。
- 动力学子报告：`pi/PI_AUDIT.md` 与 `pi/results/`。
- 统一链子报告：`unified/UNIFIED_AUDIT.md` 与 `unified/probe_results.json`、`unified/probe_unified.py`。

[S1]: https://github.com/erzhu419/Assumption-Agent/commit/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c
[S2]: https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/m4_compositional_discovery.py
[S3]: https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/m4_multiscale_renormalization.py
[S4]: https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py
[S5]: https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py
[S6]: https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/tests/test_v17_3_m4_1_unified_chain_and_audit_closure.py
[S7]: https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/hipporag_wormhole_index.py
[S8]: https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/15_V17_3_M4_1_Audit_Closure_Executable_Pi_and_Unified_Chain_Report.md
[S9]: https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/scripts/run_v17_3_m4_1_unified_study.py
[S10]: https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/m3_1_walk_factorization.py
[S11]: https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/docs/evidence/v17_3_m4_1_unified_discovery_and_renormalization_20261120/part4_unified_continual_discovery_forest_chain.json
[S12]: https://github.com/erzhu419/Assumption-Agent/blob/99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c/Hegel%20Machine/src/hegel_machine/cognition/code_agent.py
