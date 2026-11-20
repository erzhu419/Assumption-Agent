# V17.4（M4.2）工程契约闭环复核

固定提交：`fb4b23aaf802c20b4e2d0018436abab73b0cca06`。对比基线：`99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c`。

## 结论

**上一轮四个具体工程见证得到实质修复，但“全部契约已彻底闭合”仍过强。** 现在 `fresh_points=1/2` 会真实改变采样与费用；同法则、同权限的两个适用域合并后能被真实授权入口接受；元数据自证、异结构合法 artifact 调包和相对父路径越界均被拦截。最应立即补齐的是**保存参数状态与确认对象未绑定**：严格 loader 通过后实际冻结调用仍可使用另一套未经确认的参数。本轮用提交内真实 W1 产物，只改变一个参数、保留全部源码和确认材料，已复现这一对象错位。

另有两个范围较小的工程边界：OR 分支只控制适用域，未控制各分支权限及执行结论；确认点额度耗尽时仍有不足额确认和旧 M4/M4.1 路径重复使用训练点。它们不推翻本提交默认设置下的成功结果，但应纳入后续有效契约定义。

建议优先级：参数状态一致性为 **P1**；通用 OR 分支语义、fresh 预算不变量为 **P2**；声明式 steps 的命名与顺序口径为 **P3**。这里没有要求证明任意 Python 安全，也没有把“攻击者同时改写整个物理世界和全部可信宿主材料”当作工程验收目标。

## 一、复核范围与复现边界

本分工读取了本提交 AGENTS、历史能力索引及 `cognition historical --query code --compact` 返回，重点复用了 `python_archive_retention`、`growth_workflow_bridge`、`learned_research_procedures` 的归档、付费前缀保留和工作流语义背景；阅读了交接书、Report 16、前轮相关诊断、相关源码差异、新测试及改动的旧测试。当前历史目录已经提供 `retain_evaluated_python_model`、`archive_confirmed_python_model` 与共享 workflow，不需要另建平行记忆或确认基础设施。[S7]

所有执行均为离线有限 DSL / 本地模型控制。未修改源码、已提交证据、原审计目录，未调在线模型 API。主审负责全套 37 tests 和完整 M4.2 study；此处没有重复运行它们。Pi 证书、Dispatcher 全路径及生命周期计量由其他分工负责；此报告只给出与其相交的参数一致性结论。

原样复制两份旧探针后执行，副本 SHA 保存在 `original_probes/copy_manifest.json`。原探针内部 `actual_commit` 仍硬编码旧提交，作为字节保留的一部分没有改写；本次实际导入路径与执行提交以 `review_manifest.json` 为准。

## 二、已完成的修复应明确承认

| 旧见证 | 本轮独立结果 | 判断 |
| --- | --- | --- |
| 合法 `fresh_points=1/2` 不改变行为 | T01 实际确认点分别 1、2；复合成本分别 40.6452、44.6452；均 strict pass | 原见证闭合，参数已实际消费 |
| 同法则 A/B 合并后两域都被拒绝 | 合并接受，顶层前提为空，两个 OR 分支；真实 A/B bounded trial 均通过 | 原见证闭合 |
| 仅内部自洽、未确认 JSON 被接受 | 原 `no_confirmation_was_run` 负例 loader 返回 None | 原见证闭合 |
| W1 槽换成合法 W2 | 真正 M4.2 W1/W2 均可独立通过严格 loader；W2 代入 W1 expected digest 时拒绝 | 节点结构绑定闭合 |
| 相对 `../../../` 逃逸 | 原脚本在 containment 检查抛 ValueError，退出码 1 | 原见证闭合；这里的非零退出是预期拦截 |

来源：`original_probes/current_semantics_results.json`、`original_probes/run_manifest.json`、`contract_controls_results.json`。

### 不能把旧 probe 的前提变化误判成新失败

原节点调包探针只把 V17.3 `authored_method.json` 拷到新位置，没有连带旧 `confirmed_training_observations.json` / `authored_model.py`，也没有新 bundle。因此现在它的 `before_returns_registered_W1` 和 `after_accepts_other_verified_W2` 都是 false。单凭这个结果既不能证明“W1 功能坏了”，也不足以证明“仅调包被拒绝”。本轮另用完整真实 M4.2 W1/W2 工作区证明正例接受、错误 expected digest 拒绝，排除了这种混淆。

路径修复把相对与绝对输入统一 resolve，再 `relative_to(ws_root)`，此处是正确而且小范围的修复。[S1]

## 三、P1：确认重放与冻结实例参数不是同一对象

### 3.1 实现链条

1. `run_compositional_code_agent_episode` 先在隔离进程确认 source + training prefix，随后在宿主再次 exec/fit 相同 source，另取 `_PARAMS` 快照，写到 artifact 和 `confirmed_fitted_params.json`。[S2]
2. `host_confirmation_bundle` / `confirmed_manifest.json` 绑定 source SHA、AST、训练行、确认行、误差、容差，但**没有参数快照 hash**。[S2]
3. `load_verified_authored_method` 核验这些字段、可选或必需 external manifest 后，再把 source 放入临时目录，**重新在 training prefix 上 fit**，测试确认行。[S3]
4. `predict_with_frozen_instance` 从 artifact 读取 `fitted_params`，直接覆盖 `_PARAMS` 后 predict；该字段未被 loader 的重放检验。loader 也从未读取外部 `confirmed_fitted_params.json`。[S4]

因此 loader 证明的是“此源码重新拟合能通过确认”，实际执行的是“此源码加另一份保存参数”。AST 绑定解决结构身份，不能替代拟合态身份。

### 3.2 有限决定性控制

使用提交内 `stage1_W1_haldane_shunt` 的完整工作区副本，调用：

```python
load_verified_authored_method(
    path,
    expected_ast_digest=original_digest,
    require_external_manifest=True,
)
```

基线加载通过。仅把 `authored_method.json["fitted_params"]["p0"]` 从 `8.799999987...` 改为 `26.399999961...`，即乘 3。没有改动源码、`model_sha256`、AST digest、training rows、confirm rows、bundle、manifest、外部参数文件或确认文件。严格 loader 仍返回已验证对象。

| 项目 | 原始参数 | 只改 artifact 参数 |
| --- | ---: | ---: |
| 严格 loader 接受 | True | True |
| 冻结预测在原确认集 RMSE | 5.4871488609e-10 | **7.6895928552** |
| 原容差 | 0.045 | 0.045 |
| frozen helper `fit_called` | False | False |
| 外部 manifest / params / observations / source 字节变化 | 无 | **无** |

这不是伪造物理观测，也不是篡改整个可信宿主。它是保存实例损坏、状态错配应被拒绝却未被拒绝的直接见证。**没有据此声称本次原始 W1/W2 等真实快照已经错误**；基线正例准确，说明新快照功能有真实价值。

### 3.3 最小修复

沿现有归档机制，把确认期间实际使用的参数态取回并绑定到同一 manifest。至少增加 canonical `fitted_params_sha256` 或实例 digest，覆盖 source hash、参数、角色/透镜绑定；校验 key 集合与有限数值。loader 的实例验证应使用该冻结态在原确认行上直接 predict，并核对外部 manifest / params 一致性，而不是通过另一次 fit 间接替代实例验证。

这同时使 R3 的“已确认的旧实例被直接调用”更准确。是否把冷加载的 replay fit 纳入零 fit 口径与计量，由 Dispatcher 分工另述；这里不重复将其算作第二个独立缺陷。

## 四、P2：OR 的适用域成立，分支权限与结论尚未随之成立

### 4.1 正面结果成立的条件

当前新测试和 Stage 2 正例合并的是同一法则、同等权限、相容 scale map 的两个域。`authorize_abstraction_use` 现在确实对 `disjunctive_branches[*].preconditions` 做 OR，这修好了上一轮顶层 A AND B 的矛盾。[S5]

但授权上限仍在求值分支之前从**顶层 keeper claim** 读取；支持状态、证据 claim 身份也仍读取顶层。`fidelity_compress_nodes` 从 A deepcopy 整个 node，只有 preconditions/branches 改动；顶层 conclusion/action、program_steps、scale_map 继续保留 A。[S5]

### 4.2 两个合法有限控制

**权限控制：** 两节点同一现成 Murray 法则，互斥 A/B scope；A 的 permission 为 bounded_trial，B 为 retrieve_candidate。语义 OR 的 ACC/UNK fidelity 检查通过。合并前 B 的 bounded trial 被拒绝（exceeds_declared_permission_tier）；合并后 B 获准 bounded trial。两个分支各自的原权限仍保留在 branches 中，但未消费。

**结论控制：** A 为现成 `op_macro_murray_2_3_1d`，B 为现成 `op_macro_murray_capped_1d`，仍是互斥 A/B scope。因为无重叠 ACC，fidelity 接受；B 的真实 gate 通过，但顶层 action 仍是 A 的未封顶 Murray，授权返回也没有 selected branch/action。当前执行消费者从顶层 action 取 spec，因而不能自动执行 B 分支的 capped 法则。[S5][S6]

第二个控制不声称在当前 80 个目标里已产生错误确认；它证明公开通用合并 API 接受了一个尚无完整执行语义的合并。当前同法则 thermal/hydraulic 正例无需因此撤销。

### 4.3 最小修复

近期最小方案是把“仅扩大适用域”的 Fidelity 合并限制为相同执行语义：predictive action/artifact、program_steps、权限和支持状态、observation/scale 语义相同，仅域条件有差异。这样的窄合并已足够保留本次正例。

如果确实要合并不同权限或不同结论，授权器必须返回匹配的完整分支，并由后续执行、receipt 和版本追踪消费该分支；不能只把 scope OR 起来再执行 keeper。无需马上建立一个任意语言的解释器。

## 五、P2：fresh 参数消费已接线，预算耗尽规则仍不完整

### 5.1 不足额确认被默默接受

`fresh_points` 校验允许任意正整数。旧 M4 执行器用 `min(len(confirm_points), requested)`；新 M4.2 用 `unseen_points[:requested]`。二者都没有在可用点不足时返回 inconclusive 或申请新点。[S6]

在 T01 声明 `fresh_points=3`，目标只有两个确认点，实际购两个，仍 confirmed/strict_pass。此问题应定义清楚：若该字段是“至少 n 个新的独立确认点”，就不能静默降到两个；若只是“至多 n 个”，应改名并另设最少确认约束。

### 5.2 旧 M4/M4.1 子级路径在耗尽时重用训练点

对本提交 `C0_T11_MultiScaleWalk`，把合法确认额度设为 2：

- 父模型购 `q_flow=7.8`、`6.2` 两点确认并失败。
- 两行保留到子级 fit 的 prefix，这一步保留已付信息是正确的。
- 此时没有新的预置确认点。代码回退到 `[target.confirm_points[-1]]`，再次把 `q_flow=6.2` 标为 `confirm_1_child`。
- 子级确认使用包含该点的 updated prefix 拟合，再在同一点确认，最后 confirmed/strict_pass 均 true。

源码位置：`m4_multiscale_renormalization.py` 1397–1437；M4.1 unified 同样模式在 1596–1609，fallback 1715–1721。新 M4.2 recall 在 `unseen_pts` 空时 break，已避免直接的子级重用；但仍有不足额切片的问题。[S6]

当前默认已改为 fresh=1，因此上述耗尽控制不自动否定主 study 默认路径。修复应保持已付前缀保留，并在剩余独立确认不足时申请额外可区分点或诚实 unknown；不要丢掉 prefix，也不要重买旧点伪装成新证据。两项控制详见 `fresh_budget_results.json`。

## 六、P3：program_steps 是声明式配置，不是有序程序

解释器把 list 转为 step_map，所以合法顺序完全反转仍返回相同计划；这是本轮直接控制得到的结果。其 docstring 本来称为 declarative steps，因此未把“忽略顺序”列为独立高优先 bug。[S6]

另外，`verify_scale_closure_or_escalate` 只产生 `verify_scale_closure` 布尔字段，该字段在相关三个执行模块没有实际消费者；其 `scale_regime` 参数也不做一致性检查。当前 scale 检查由固定宿主流程实施，不等于缺少 scale 保护，但不能说所有 steps 逐条执行。

两个合理选择：继续明确采用无序声明式计划，拒绝/规范化不能消费的字段；或者规定有限合法偏序并在校验时拒绝逆序。无需为了名称升级为通用程序虚拟机。当前新测试覆盖了 fresh 1/2、非法路径、同法则 OR、节点 digest，却没有覆盖顺序含义、权限异质 OR、参数快照错配或 fresh 耗尽；这些应成为最小验收用例。

## 七、宿主确认信任边界应如何表述

相比旧版本，仅填 `confirmed_ok=True` 已无法通过，真实 rows/prefix/hash 和子进程重放提高了验证强度；这一修复应保留。当前 `require_external_manifest=False` 的默认参数仍允许带 bundle、无外部 manifest 的新 artifact；为兼容旧工作区，还存在用 training rows 重放的 legacy 分支。M4.2 Dispatcher 明确要求 external manifest，而旧统一 spec loader 未要求。[S3][S6]

建议文档分别标明：“当前可信宿主生成流程、严格实例 loader、legacy 兼容读取”的保证，避免笼统称所有入口都强制外部宿主证明。无需把外部 JSON 哈希进一步包装成对任意恶意宿主的可信硬件证明；本轮真正需要修的是上节已经复现的**确认—参数—执行一致性**。

## 八、证据与可复现文件

所有输出位于本报告同一目录：

- `review_manifest.json`：执行提交、源码及新增 probe SHA。
- `original_probes/copy_manifest.json`、`run_manifest.json`：原 probe 字节与执行记录。
- `original_probes/current_semantics_results.json`：fresh 1/2、原 OR、原无确认负例等。
- `probe_m42_contract_controls.py`、`contract_controls_results.json`：真实参数快照、真实 M4.2 digest 调包、OR 权限/结论、顺序控制。
- `probe_fresh_budget_edges.py`、`fresh_budget_results.json`：额度不足与旧子级耗尽重用。
- `historical_code.json`、`core_diff.patch`：历史复用查询与核心差异。

复现新增控制时设置 `PYTHONPATH` 指向固定 checkout 的 `Hegel Machine/src`，执行两个 `probe_*.py`。脚本仅写本目录下副本/结果，运行原版 probe 时另设 `HEGEL_AUDIT_REPO` 指向固定 checkout。所有数据以 JSON 完整保存，stdout 只是摘要。

## 源码定位

以下行号均为固定提交，路径前缀均为 `Hegel Machine/`。

| 引用 | 路径和关键位置 |
| --- | --- |
| S1 | `src/hegel_machine/cognition/m4_compositional_discovery.py` 53–103：相对/绝对路径 containment 与隔离 prefix |
| S2 | 同文件 1005–1077：确认、再次 fit 提取参数、bundle/manifest/artifact 写入 |
| S3 | 同文件 1163–1339：loader、expected digest、legacy、新 bundle、external manifest、重新 fit 确认 |
| S4 | 同文件 1135–1157：冻结 helper 从 artifact 覆盖参数，不读取外部参数文件 |
| S5 | `src/hegel_machine/cognition/wormhole_forest.py` 248–256（顶层权限）；285–305（OR 前提）；420–445（顶层支持状态）；489–591（Fidelity 合并）；1282–1287（正式合并启用 OR） |
| S6 | `src/hegel_machine/cognition/m4_multiscale_renormalization.py` 1141–1190（声明计划）；1336–1357、1397–1437（fresh 消费/耗尽）；`m4_1_unified_discovery_and_renormalization.py` 1382–1397（顶层 action loader）、1596–1609、1715–1721；`m4_2_unified_closure_and_pi_lifecycle.py` 2332–2399（plan 与新 recall 的 fresh 流程） |
| S7 | `src/hegel_machine/cognition/code_agent.py` 971–1004：已有 retained/confirmed Python archive；历史索引 `python_archive_retention` / `growth_workflow_bridge` |
