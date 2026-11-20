# V17.3 / M4.1 B1–B10 工程闭环独立复核

审计提交：`99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c`。比较基线：`11961f702739ab272661ff59b77fb5bc8e3589b8`。本报告只负责工程边界闭环；完整 33 项回归、完整新 study 复跑、统计及动力学区域证书由主审及其他子审负责。

## 结论

本轮存在扎实的实际修复，不能把它描述为仅补测试或仅改文档。旧 AST 双定义/属性遮蔽、无 receipt 的布尔旁路、工作区内绝对路径、四参数编号、未知 opcode、child 最终授权、fidelity 未提交及纯协议修订丢失粗粒化边，都有直接改动和相应复跑改善。

不过，报告 15 的“11 项已 100% 闭合”把三个不同层次混合了：旧反例已挡住、相应字段已经校验、相应语义已经由真实消费者执行。后两层目前仍有差距。最清楚的两项是：`program_steps` 有效字段仍不驱动执行；合并后的析取契约已提交，但真实授权器仍按顶层合取前提执行。另有 artifact 对节点身份的绑定与确认来源检查不完整，以及 M4 evaluator 的相对越界路径仍未统一规范化。

建议在继续扩大 M5 之前做一轮范围有限的接线修复。无需撤回有限 DSL、哲学森林、多尺度结构或图游走路线；本审计也没有把“任意 Python 程序的安全证明”作为验收条件。

## 方法与可复现性

读取了相关 AGENTS/reusehub 约束，调用历史能力查询，检查了上述七份模块差异、新 unified 消费路径、报告 15、新回归测试和 handoff 指定的旧审计。

旧探针四份均逐字节复制到本目录的 `original_probes/`，未修改探针源码，四个进程均 `exit 0`。复制 SHA 记录于 `original_probes/copied_probe_manifest.json`。实际导入均来自新固定 worktree；`replay_execution_manifest.json` 记录实际 commit、PYTHONPATH 与仓库 clean 状态。

必须注意：旧探针内写死的 `commit` 字段仍是旧提交；这是原样保留探针的结果，不代表实际执行旧源码。尤其 `author/probe_author.py` 手工复制了旧 loader 的 `json.loads(...)["operator_spec"]` 数据路径，它的 metadata-only 结果不能用于证明新 `load_verified_authored_method` 失效。新 loader 的判断使用独立新探针。

新增控制全部离线，使用当前有限 ExprNode/step DSL，仅向独立 scratch 工作区写文件：

- `probe_current_semantics.py` / `current_semantics_results.json`：合法 fresh_points、析取域真实授权、确认来源、节点与 artifact 绑定及纵向重绑边界。
- `probe_relative_escape.py` / `relative_escape_result.json`：只执行本审计自身 scratch 中的常数模型，检查相对路径 containment。
- `source_snippets.txt` / `source_manifest.json`：精确行号、对应源码片段和 SHA。

## B1–B10 对照表

| 项目 | 复核结果 | 判断与范围 |
| --- | --- | --- |
| B1 AST 双 predict / math.exp 顶层篡改 | 两组旧反例均为 `UNSUPPORTED_IMPURE_PYTHON`，摘要不同 | 旧见证已闭合；这不等同于任意 Python 的语义或安全认证，而当前有限 DSL 不需要后者 |
| B2 无 receipt 的布尔旁路 | 无 target_id 和有 target_id 两条调用均拒绝最终指导 | 已闭合；receipt 字段一致性与来源可信度仍是不同问题 |
| B3 工作区内绝对路径 | 原相对/绝对确认均 RMSE 9 | 旧见证已闭合；M4 helper 的相对越界入口仍存在，详下 |
| B4 四 flag 同时置真 | 源码变为按维度、对称性及起始签名最多设置一个 flag | 广播错误已闭合；这些新 flag 仍是失败后结构假设，不是独立测出的机理事实 |
| B5 参数编号 | exp→osc→shunt 的真实参数为 p0,p1,p2,p3，distinct=4，AST pure | 旧组合碰撞已闭合，通用子表达式遍历与 multi_start 保留是有效修复 |
| B6 authored artifact 校验 | 未确认 metadata / 原 source 与改 spec 不一致会拒绝 | 完整性改善真实；未验证确认来源，未绑定 node digest 与载入 artifact digest |
| B7 program_steps | 五个旧目标的垃圾 opcode 均取消 recall，成本增加 | 非法指令已挡；合法指令参数未驱动执行，尚非实际解释器 |
| B8 child 最终授权 | deny-final 与未知硬前提都使 confirmed_via_recall=False | 旧旁路已闭合；fallback 独立成功不构成 recall 授权绕过 |
| B9 fidelity 提交 | 原 helper/keeper 不一致已修；Murray/Monod 重叠 ACC 被拒 | 持久化层闭合，真实授权对 disjunctive_branches 的消费未闭合 |
| B10 纵向边版本 | 纯协议修订前后 coarse edges=6，原 6→4 丢边消失 | 版本维护改善；显式 revalidate API 是重绑操作，本身不证明闭合重新成立 |

## 1. B7：合法 program_steps 仍不决定程序执行

### 代码位置

- `m4_multiscale_renormalization.py:1135–1184` 验证步骤并返回 `operator_tag/calibrate_lens/verify_scale_closure/fresh_points`。
- 同文件 `1288–1305` 仅把返回值放入 `_exec_plan`；全源码没有 `_exec_plan` 的读取者。
- 同文件 `1327–1363` 固定做 fitting，并只购买 `target.confirm_points[0]`。
- 新 `m4_1_unified_discovery_and_renormalization.py:1388–1406` 只判断 `plan is None`，随后丢弃计划；`1419` 固定购买第一个点。
- 同新模块 Part 3 `622–625` 同样只检查合法性，`643` 固定 `confirm_pts[:1]`。

### 新控制

在同一 `C0_T01_AdiabaticMacro`、同一森林、同一数据下，把所有节点的合法 `subprocess_isolated_confirm.fresh_points` 分别设为 1 和 2。两次：

- 都 `confirmed_via_recall=True`；
- 全部采样序列及除运行时间外的返回结果完全相同；
- 两次实际 fresh confirmation 都只有 **1 个点**；
- 成本也完全相同。

因此，当前确实实现了 schema validator，但没有把返回的计划连接到执行调度。拒绝 nonsense 与执行合法参数是两个独立要求。默认节点写 `fresh_points=2` 而直接成功路径只验证一个点，已经在正常配置中出现，不需要构造任意恶意代码。

### 最小修复

让已有的固定操作实现接收并消费 validated plan：确认点计数来自 `fresh_points`，不足时申请补点或返回有理由的拒绝；标定和闭合步骤按计划执行，或明确将其列为不可省略的固定协议并移除伪可配置字段。无需引入通用解释器。验收应保留 garbage 拒绝测试，再增加一个合法参数改变应改变调用轨迹的测试。

当前五个旧 garbage 控制的正面结果：完整/垃圾成本分别为 `40.6527/44.9528`、`44.6542/49.2542`、`100.6557/112.6557`、`44.9528/53.2527`、`44.9543/49.5543`，全部完整 recall 真、垃圾 recall 假。此改进应保留。

## 2. B9：析取契约已存入 keeper，但授权器执行了合取

### 代码位置

- `wormhole_forest.py:475–486` 把两节点 preconditions 去重后放到顶层，同时保留 `disjunctive_branches`。
- `1155–1194` 现在真正提交 merged contract，并用传入 evaluator 做提交后检查；这一修复成立。
- 但真实 `authorize_abstraction_use` 的 `263–290` 只读取顶层 preconditions，逐项解释；任一 FALSE 即拒绝。它没有 OR-branch 消费逻辑。
- 全源码只有 compression helper 和新 study 的 `_comp_eval`（`m4_1_unified_discovery_and_renormalization.py:1892–1921`）读取 `disjunctive_branches`。

### 新控制

构造当前有限语法中的**完全相同预测定律**，仅作用域分别为 `audit_scope == A` 与 `audit_scope == B`。非作用域是 UNK，不是对定律的经验反驳；因此两者确实可以由带作用域的 OR 表达式保真保存。

合并前 A、B 各自的真实 `bounded_trial` 都获准。合并后：

- fidelity 返回 accepted=True；
- 已提交对象的自家分支 evaluator 在 A 和 B 上都返回 ACC；
- redundant 已归档；
- 真实授权器对 **A 和 B 都拒绝**，原因均 `precondition_violated`；
- 顶层前提成为 `scope=A AND scope=B`，结果分别 `[TRUE,FALSE]` 和 `[FALSE,TRUE]`。

这是“验证对象与最终执行语义不一致”的实际见证，已经超出旧的“未提交字段”问题。新 Part4 的三个重复节点恰好与 keeper 前提相同，所以当前 48 个域内结果不受本见证直接反驳；但尚不能把实现描述为支持一般作用域析取的保真压缩。

### 最小修复

引入一处共享契约解释逻辑，让 compression evaluator、授权器和执行路由都消费相同的 branch 语义；选中分支后仍用该分支自己的前提、权限上限、作用域与对应程序。若本阶段只需要相同作用域的重复节点去重，更小的修法是暂时仅允许等价前提合并，明确拒绝需要析取的组合，直到分支解释接线完成。

## 3. B6：完整性校验、确认来源和 node binding 仍需分开

### 已改善的部分

`m4_compositional_discovery.py:1089–1141` 确实检查 confirmed_ok、pure AST 状态、source SHA、source AST 和由 spec 重编译后的 AST。现在不能仅保留 operator_spec、填错误摘要就冒充有效 artifact。实际 Stage1 工作流也确实在 `1006–1027` 执行确认并生成记录。

### 仍然缺少的两项

**确认来源：** `confirmation_digest` 在 `1116–1126` 仅要求非空字符串，没有同独立确认记录比对。新回归本身在 `tests/test_v17_3_m4_1_unified_chain_and_audit_closure.py:127–138` 用固定 `abcd1234ef567890` 即构造一个“有效 artifact”。新控制用受支持编译器生成一个正常 Michaelis 程序，没做任何确认、没提供任何观测，仅填 `confirmed_ok=True`、`confirmation_digest="no_confirmation_was_run"` 和正确内部 SHA/AST，loader 仍接受。

这说明的是当前 loader 验证内部一致性，不是证明当前 Stage1 证据造假。当前可信离线宿主生成 artifact 的条件下风险有限；将来由 LLM、外部文件或可变工作区自行提供 metadata 时，必须让确认记录来自宿主控制的数据路径。

**节点身份：** 新 `m4_1_unified_discovery_and_renormalization.py:1280–1291` 只要路径中的 artifact 自身验证通过，就取其 operator_spec；没有比较该 artifact digest 与 node 已注册的 `conclusion_or_action.ast_skeleton_digest`。

新控制把一个 W1 节点指向本审计 scratch 内的 W1 artifact，载入正确；随后仅把路径内容换成提交中**另一个完全有效的 W2 artifact**，再次载入得到 W2，节点契约仍然宣称 W1。W1 digest=`c2ced29ba2c345`，W2 digest=`17d05c4bf44ecf`，两者明确不同。此时节点版本、索引签名、claim 与实际取出的算子已经脱节。控制没有宣称已造成最终错误确认；它证明 artifact 验真后仍需绑定注册对象。

### 最小修复

先补 node expected digest 与 artifact verified digest 的严格相等，并把确认记录 ID、模型 hash、训练前缀 hash、确认点/结果 hash 和 tolerance 放到宿主写入、候选代码不可改写的 manifest。加载器根据 manifest 校验；不必引入公钥基础设施。若继续仅支持可信本地 provider，应把信任边界写清楚，避免称“confirmation digest 已验证”而实际只检查非空。

## 4. B3：旧绝对路径修好，但 M4 helper 相对路径仍可越界

`open_workspace_labs.py:215–228` 的规范化函数对绝对/相对输入都 resolve 并 relative_to；`open_workspace_wormhole.py:917–930` 同样处理两者。M4 的 `evaluate_isolated_python_model`（`m4_compositional_discovery.py:65–74`）却只规范化绝对输入，相对输入原样传入 evaluator。

独立控制仅使用自己的 scratch 常数模型，给 M4 helper 传入相对 `../../../workspace/.../path_outside.py`。在本环境 `/tmp/hegel-m4-sandbox-*/ws` 下它跳出临时工作区，成功执行了工作区之外的文件，`returncode=0, RMSE=0`。

这是剩余 containment 入口，不是对新证据文件已泄漏的指控。正常 cand 路径和旧绝对路径原样复跑均已修好。最小修法是三个入口统一复用现有规范化 helper，对相对/绝对输入都做边界判断，随后只传沙箱中的相对路径；API 对无效路径的返回形式也应统一，报告中“均抛 ValueError”目前并不准确。

## 5. B4 与 B10 的适当表述

### B4：最多一个 flag 是实际改善，机理诊断仍然是推断

`m3_scale_universe.py:2396–2416` 将四旗广播替换为按 arity、对称性、低负载签名选择一面旗。这样不会再把互不相同的 ceiling、overshoot、turnover、rollover 全部宣称已观测。

但任何候选确认失败并不逻辑蕴含某个唯一高负载机理：例如 active_arity==1 就直接把 rollover 标成 True，仍可能只是模型错配的竞争解释之一。当前随后还有 fresh confirmation，所以这不等于已证明新的错误最终授权。建议把这些字段明确记为“待检验诊断/route hint”，或计算实际差分与残差形状后才升级为 observed feature；不要在报告中把结构映射直接称为独立实测机理证据。

### B10：版本重绑闭环成立，revalidate 不自动产生新物理证据

旧探针中 aq_macro / aq_meso 仅更新协议标签，现在 coarse-graining edges 保持 6，fine child 与 coarse parent 都仍可达，horizontal edges 均不 stale。新 `hipporag_wormhole_index.py:209–213,230–234` 也会排除显式 stale 的纵向边。

但 `wormhole_forest.py:966–1014` 的 `revalidate_vertical_edges_after_revision` 只接受 witness_episode_id/reason，把当前两端版本写回并清除 stale，没有测量或证书检验。与新回归相同，先把 ScaleMap 改成 `closure_scope=not_closed` 可使边数 6→4；只给任意文字 label 调该 API 又变 4→6，而 ScaleMap 仍是 not_closed。

这项只表明 API 的责任边界。最终授权还有独立 ScaleMap gate，不能把此控制误报成已发生错误最终指导。在现有纯协议修订调用中，保留原闭合证据可能完全合理。M5 若真的改变投影算子、变量消去、作用域或物理定律，调用方必须先产生可检查的新的证据，再允许重绑；将函数改名为“commit_revalidated_edges”并要求 receipt，或把验证器显式接入，均可避免误解。

## 建议的有限闭环顺序

1. 让合法 `program_steps` 真正驱动采样/标定/闭合行为，补一组参数变化→调用轨迹变化的测试。
2. 让 fidelity 验证与真实授权共用相同 branch 语义；当前阶段可以先限制只合并等价作用域。
3. 绑定 node digest 与 verified artifact digest，并把可信确认来源从 candidate metadata 中分离。
4. 三个 evaluator 共用同一模型路径规范化，关闭相对路径例外。
5. 对 B4 和 B10 准确标注“诊断假设”和“已经外部验证后的重绑”，不要额外扩张成理论证明任务。

这些修复都可以复用现有函数，不需要重写森林、检索器或有限表达式语法。报告应把“旧见证通过率”“契约字段一致性”“真实执行语义”“独立科学证据”分别列出；这样既保留 M4.1 的实际进展，也能给 M5 一个可检验的起点。
