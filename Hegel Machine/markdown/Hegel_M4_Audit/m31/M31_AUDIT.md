# M3.1 审计：修复闭环、游走归因与适用边界

审计固定提交：`11961f702739ab272661ff59b77fb5bc8e3589b8`。对照提交：`612d8eea5c8104f4a652c8f4b09df5b8b81ca1b8`。

源码根目录：`/workspace/scratch/3f14ad2da0ed/repo_m4_11961f702/Hegel Machine`。

本审计产物目录：`/workspace/scratch/3f14ad2da0ed/audit_m4_11961f702/m31`。

## 1. 结论

**M3.1 的工程修复有实质进展，新的受控实验也确实出现了旧 M3 没有证明的非直接图召回收益。当前需要修正的是契约证据升级、少数 API 边界和因子归因口径，并无理由撤掉哲学森林或多跳图检索。**

可以确认的进展：原三种确认记忆器均失效；原两种 AST 碰撞被修复；原四种旧收据绕过被拒；诊断 stdout 已改变后续动作；普通开放修订臂不再无条件多买两点；同样实际拟合工作量收费相等；`program_steps` 与 HExA 错误库已进入执行决策；新安全 pilot 下正确子规律确实没有直接签名命中，PPR 则能召回它。

需要优先修复的实际问题：

1. **粗候选一次确认失败，被当成四种不同机制都已被观测确认。** 这条代码在正常 T25–T27 执行中实际触发，伤害的是三值契约的证据语义；最终数值确认仍然存在，不能将它表述为本次出现了错误数值确认。
2. **报告宣称完整 walk × pool 2×2 因子设计，正式臂表缺少 α=0、NoPool 格。** 补充的单 seed、10 题控制证实游走条件收益，但 pooling 的边际并非无条件为正。
3. **C2 的有限 AST 接受边界仍可错误接受重复 `predict` 和 `math.exp` 属性重绑；C3 无收据且无目标标识时仍允许旧布尔确认；C1 接受绝对模型路径时可绕回原工作目录。** 这些边界在小探针中可复现，未发现它们由本轮固定 M3/M4 作者正常生成，也没有证据说明污染现有主表。
4. **C4 报告把消融臂的 74.70 误写成普通臂比较值。** 普通臂真实成本为 55.075 / 59.075，Hegel 为 53.225；优势存在，幅度应改正。

“安全低载”目前是所有正式臂共享的初始观测策略，并非实验室强制安全规则。这个事实不否认受控试验的图召回收益，但不足以证明 PPR 在真实安全实验中“绝对不可替代”。

## 2. 证据范围与复现说明

### 2.1 阅读与执行范围

已全文阅读 `m3_1_walk_factorization.py`、`hipporag_wormhole_index.py`、`test_v17_1_m3_1_audit_closure_and_walk.py`、报告 13，以及相关 study 入口；核对 `m3_scale_universe.py`、`open_workspace_wormhole.py`、`open_workspace_labs.py`、`open_python_revision.py`、`wormhole_forest.py`、`cross_domain_arms.py` 的改动与调用链。按项目要求先读 `AGENTS.md` / reuse hub 并执行历史查询，查询结果保存在 `historical_query.json`，此次关键词未返回复用条目。

本子审计实际执行：

| 证据 | 实际工作 | 适用范围 |
| --- | --- | --- |
| `probe_closure_original.py` / `closure_probe_results.json` / `original_closure_replay.log` | 原 `probe_closure.py` 字节不变，复制到独立目录后复跑 | 原 C1–C4 对抗见证是否消除 |
| `probe_boundaries.py` / `boundary_probe_results.json` | 两组新 AST 对比、无收据布尔分支、相对/绝对路径、四个初始高载请求 | API 边界与安全策略层次 |
| `probe_factorization.py` / `factorization_control_results.json` | 一次 36 源形成；同一森林、原始已冻结 DEV gate、seed 8028101 的 10 个 Assoc/Walk 目标 × 4 因子格，共 40 次目标执行 | 缺失因子格的事后诊断；不是新的四 seed 正式研究 |
| 同一控制的额外只读比较 | 恢复子节点完整实测签名、K 三档输入指纹、同观测的 3 次拟合计费比较 | 签名构造因果边界、K 固定输入、计费相等性 |

根代理另行运行了 28 项验收测试和完整研究重放；这些执行不计入本子审计独立完成的工作，也不以本文件替代其完整结果报告。

**原样探针的元数据注意事项：** 原脚本源码内嵌了旧提交 `612d8ee...`，为保留“原样复跑”没有改该字符串。因此新生成的 `closure_probe_results.json` 顶部 `commit` 字段仍是旧值，不能用它判断运行代码版本。实际 `PYTHONPATH` 是固定新 worktree 的 `src`。原文件和复制文件 SHA256 均为：

`2a0a787c742592ada1b590ebeb47cbb78fbfb1020f3d56d2ac33d32a6884186c`。

执行命令（全部离线、仓库只读、输出到本目录）：

```bash
cd '/workspace/scratch/3f14ad2da0ed/repo_m4_11961f702/Hegel Machine'
PYTHONPATH=src python3 /workspace/scratch/3f14ad2da0ed/audit_m4_11961f702/m31/probe_closure_original.py
PYTHONPATH=src python3 /workspace/scratch/3f14ad2da0ed/audit_m4_11961f702/m31/probe_boundaries.py
PYTHONPATH=src python3 /workspace/scratch/3f14ad2da0ed/audit_m4_11961f702/m31/probe_factorization.py
```

下面代码位置均相对于本文件开头的源码根目录，行号对应固定提交。

## 3. C1–C7：哪些已修、哪些需限定

| 项 | 本轮已修与证据 | 结论边界 |
| --- | --- | --- |
| C1 确认文件隔离 | `open_workspace_wormhole.py:899–915`、`open_workspace_labs.py:234–251` 复制工作上下文并覆写两个标准观测文件；正常调用先 `acquire(..., workspace=None)` 再确认。原参数记忆器、`observations.json`、`model_observations.json` 记忆器均 RMSE=9；完整 CodeAgent 返回 `scoped_unknown`、无假确认 | 旧漏洞路径修复成立。仍需拒绝/重定位绝对模型路径，见 §4.4；不宜把临时目录称为任意 Python 的操作系统隔离 |
| C2 AST | `_CORE_LAW_EXPR` 与预测式对称剥离外层仿射，`open_workspace_wormhole.py:414–429`；显式同名函数遮蔽与 round 精度被检查；森林先看 `VERIFIED_PURE_AST`，`m3_scale_universe.py:1776–1782`。原两组碰撞均消除 | 已修的具体见证成立；“纯 AST 已验证”接受域仍有两个简单假阳性，见 §4.2 |
| C3 收据 | `wormhole_forest.py:155–216` 强制 target/version、开放任务模型与确认摘要；提供无效收据时旧布尔值不再覆盖。原四种探针全 False | 无收据且缺 target_id 时仍退回旧布尔分支，见 §4.3。正常 M3 新目标路径附齐字段 |
| C4 stdout / 普通臂 | `open_python_revision.py:241–342` 解析诊断并生成/重排候选；原探针对矛盾 stdout 得到不同下一动作。`1084–1088` 改为初始只 pilot，额外点放 fallback | 改善确实有因果接线。报告 13 第 18 行的普通臂成本取错了消融臂，见 §4.5 |
| C5 方法与排序 | `m3_scale_universe.py:2347–2379` 读取三种操作：决定 schema、是否使用观测透镜、是否进入确认；清空步骤会走不同路径。`hipporag_wormhole_index.py:43–52,444–455,612–623` 用内容哈希平局排序 | 这是有限三操作的声明式执行器：目前将步骤转成 op 字典，不是任意有序程序解释器。该界限不妨碍方法元数据产生真实因果作用。重命名测试实际比较 schema 排序，不能泛称已覆盖所有等内容节点身份顺序 |
| C6 计费 / 桥接 / 尺度 | `m3_scale_universe.py:773–777` 按已评估候选数计费。新增控制：单 schema 实评 1、费用 10；显式全 grammar 与 None 均实评 9、费用 58、同模型 SHA、同训练/验证误差。`1887–1900` 将旧 bridge meter 入账；`1388–1403` 执行真实微轨迹探针 | 是单调的代理工作量计费，不等于 CPU 时间。新 M3.1 没有 bridge 行的原因是根本没执行该 bridge，详见 §6；不是漏账复发 |
| C7 预算 / HExA | `m3_scale_universe.py:2340–2343,2384–2393` 分开试拟合 4 与召回确认 2；`2650–2668` 按观测条件把匹配错误恢复项提前 | 2 是“召回阶段”的确认上限。失败后 direct fallback 还能买确认点；本控制的 T28 实际总确认 3。报告应使用阶段性措辞，避免称全任务最多 2 点。错误库匹配已可达，不再是旧不可达实现 |

## 4. 真实残余问题

### 4.1 正常主路径的契约失真：一次失败同时确认四种机制

位置：`src/hegel_machine/cognition/m3_scale_universe.py:2396–2407`。

只要 `pilot_regime == safe_low_load` 且候选确认误差超限或执行异常，代码将以下四项同时设成 True：

```python
high_corner_ceiling_detected = True
transient_overshoot_detected = True
high_substrate_turnover_detected = True
high_load_rollover_detected = True
```

随后这些值在 `2425–2438` 作为 `target_features` 交给 Tier-3 授权。新探针记录了正常 T25、T26、T27 的真实授权调用：三种不同机制均携带全部四个 True 并被授权，其中 T26 只有一个输入维度，却同时“观测确认”了二维角点封顶、暂态超调与底物支路抑制。

报告 13 第 37 行将这一过程解释为高载证伪父规律后对应条件由 UNKNOWN 升为 TRUE。实际执行只观察到“这个父候选在新点不对”，没有分辨四个机制条件。错误信息不能证明任意一种指定修订机制，更不能同时证明四种。

**影响范围：** 本次被选中的正确修订模型仍另行通过了隔离前缀的新鲜点确认，终评亦成功。本问题破坏的是契约证据来源与未来路由/授权可靠性；不应据此声称现有 1080 个目标已经数值造假。

**最小修法：** 将事件记为有来源的 `parent_prediction_refuted`，附父节点/模型摘要、确认点标识与残差；只有真正运行相应观测判别器时才写具体机制标志。让 UNKNOWN 继续允许 bounded trial，并用候选自己的数值确认与明确的局部有效范围决定下一层授权。这样可保留目前有效的图检索与拟合路径，不需要增设全局理论证明器。

复现证据：`factorization_control_results.json.safe_high_load_authorizations`。

### 4.2 AST 两种新假阳性：有限接受域应进一步收紧

位置：`src/hegel_machine/cognition/open_workspace_wormhole.py:225–270,348–380,397–448`。

**重复定义。** 模块扫描保留第一个 `predict_fn`，后一个同名定义只加入 `shadowed_names`；验证预测式本身并不会调用名字 `predict`，故不会被拒绝。Python 执行时真正生效的是最后一个定义。

```python
def predict(row):
    return row['features']['x']

def predict(row):
    return row['features']['x'] ** 2  # 对比版本改成 **3
```

两版本都被标为 `VERIFIED_PURE_AST` 且 digest 相同；在 x=2 上实际分别输出 4 和 8。

**模块属性重绑。** `math.exp = lambda x: x**2` 与 `math.exp = lambda x: x**3` 的赋值目标为 `ast.Attribute`，当前模块扫描只处理若干 `Name` 绑定；后续 `math.exp(...)` 仍被当成可信白名单调用。两版本同样 verified、同 digest、预测 4 与 8。

**建议：** 重复预测定义直接拒绝；对允许数学模块的属性赋值/删除、复杂顶层控制流绑定、星号导入等不在已支持 DSL 内的构造返回 UNKNOWN/UNSUPPORTED。无须解析任意 Python 的完整动态语义。现成固定编译器只生成一个确定的 `predict`，这些拒绝规则不会限制其有效科学表达。

M4 可达性核对：`m4_compositional_discovery.py:535–538` 的固定 spec 编译器只输出一个 `predict`；本次未发现由其正常操作生成上述例子的途径。开放手写 Python 作者接入时该接受边界才会直接承受更大压力。

证据：`boundary_probe_results.json.ast_boundaries`。

### 4.3 C3 仍有无收据、无 target_id 的布尔放行

位置：`src/hegel_machine/cognition/wormhole_forest.py:236–242`。

提供无效 receipt 的旧绕过确实被堵住。但 `evidence_receipt=None`、目标类型为开放工作区、`task_view` 没有 `target_id` 时，代码落到 `confirmed = bool(heldout_confirmed)`。新探针用有作用域支持的 `m3_open_workspace_ast` contract、空前置条件和 `heldout_confirmed=True`，得到 `local_solver_guidance` granted=True；只加入 `target_id` 则 correctly denied。

这是边界验证顺序问题：缺失身份不应放宽约束。最小修法是按开放任务类型 / family_plugin 决定需要绑定证据；缺 target_id 应返回不满足条件，旧协议若确需布尔捷径应单独显式标记。当前正常 M3 目标确认路径填写了 target/version/model/confirmation 四字段，因此本例不证明本轮主表受影响。

证据：`boundary_probe_results.json.receipt_boundary`。

### 4.4 C1 绝对路径会绕出新复制目录

位置：`open_workspace_wormhole.py:899–915`、`code_agent.py:681–695,712,727,750–753`。

新确认器复制工作目录，却把传入的 `model_rel_path` 原样交给通用 evaluator。通用 `python_model_context` 支持绝对路径；当绝对路径不在新 sandbox 中时，它会把工作目录切到该绝对路径的父目录。于是模型重新进入含确认标签的原 workspace。

与旧见证相同的 `file_memorizer`：相对 `file_memorizer.py` 确认 RMSE=9，绝对 `/.../file_memorizer.py` 确认 RMSE=0。没有使用读父进程内存、恶意网络或其他额外能力。

**建议：** 面向确认的 API 先把模型严格解析为 workspace 内的相对路径，复制后强制在新目录内解析；拒绝外部绝对路径与逃逸路径，必要时只支持已声明的依赖清单。此处的修复目标是契约路径一致性，不要求把这个实验系统扩展成通用恶意代码沙箱。

**可达性边界：** 当前 M3 调用使用相对路径；M4 `run_compositional_code_agent_episode` 在 `m4_compositional_discovery.py:975–980` 直接传相对 `model.py`。正常固定作者不触发此漏洞。报告的常规三记忆器闭环仍成立。

证据：`boundary_probe_results.json.absolute_path_confirmation_boundary`。

### 4.5 C4 对照对象错误，但真实收益仍保留

报告 13 第 18 行声称普通臂比较为 53.23 < 74.70。提交内 `open_revision_fair_results.json` 的四 seed 均为：

| 下游任务 | Hegel | Continuing Ordinary | RemoveRevisedNode 消融 |
| --- | ---: | ---: | ---: |
| T_O1 | 53.225 | 55.075 | 74.700 |
| T_O2 | 53.225 | 59.075 | 77.450 |

Hegel 与 Ordinary 的测量费两题均为 50。74.70 来自测量费为 65 的 RemoveRevisedNode。真实平均比较为 53.225 vs 57.075，优势约 6.75%，主要是拟合/角色搜索开销；不是原文暗示的约 28.75% 普通臂优势。

这应作为文档勘误保留，不能将其误解释为 stdout 修复或普通臂成本拉平没有奏效。

## 5. walk × pool 因子诊断

### 5.1 正式试验缺第四格

`scripts/run_v17_1_m3_1_factorized_walk_study.py:51–60` 的正式列表包括：

| 同一 PPR scorer | Pool | NoPool |
| --- | --- | --- |
| α=0.65 | 已有 | 已有 |
| α=0 | 已有 | **缺少** |

BM25 NoPool 不能代替右下角，因为它还改变 specificity reset、IDF/文档长度归一化与排序公式。`Flat_BM25_WormholePool` 与 `Flat_BM25_EqualMetadata` 可以构成独立的 BM25 pool 配对，但不是 PPR 的第四格。

### 5.2 补缺控制：同一 seed、同一十题、同一门控

本子审计仅使用 seed 8028101、固定 T01–T06 与 T25–T28，四格同源森林、同初始 pilot、同公开能力、同实际费用公式、同原始已冻结 DEV gate。缺失格通过既有 `hipporag_walk_nopool` 模式设 `damping_alpha=0` 构造；没有改源码、增加候选预算或借助真值选模型。

| 格 | strict pass | 召回确认 / 10 | Walk 召回 / 4 | 平均测量费 | 平均复合费 | Walk 平均复合费 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Walk + Pool | 10/10 | 9/10 | 3/4 | 41.0 | 43.34415 | 48.40475 |
| Walk + NoPool | 10/10 | 9/10 | 3/4 | 41.5 | 43.88165 | 48.40475 |
| α0 + Pool | 10/10 | 6/10 | 0/4 | 46.0 | 49.68000 | 64.30250 |
| α0 + NoPool | 10/10 | 6/10 | 0/4 | 46.0 | 49.63000 | 64.30250 |

条件差异：

- 有 Pool 时启用 Walk，平均复合费减少 6.33585；无 Pool 时减少 5.74835。
- 有 Walk 时启用 Pool，平均复合费减少 0.53750；无 Walk 时反而增加 0.05000。
- 四个 Walk 目标上，两种 Pool 设置完全同价；新主要收益来自可将概率输送到未直接命中子节点的图传播。

因此补缺结果支持“在当前任务编码和候选消费策略下，游走产生可测收益”。它不支持“两个因子在所有条件下都严格独立正收益”。尤其正式报告自身 BM25 配对也显示 Pool 更差：84/108 vs 88/108，平均复合费 41.426 vs 40.437。

这些是单 seed、10 题的事后控制，不能据此估计总体成功率或给出新的显著性判断。正式多 seed 主表由根代理独立重放校验。

### 5.3 “仅由图到达”成立，不能归因成通用 PPR 不可替代

本控制四个 query 的低载签名分别为 ridge / reciprocal / shunt / charging onset，加上校准读出与 impedance modality。四个修订子节点都没有 query 直接重叠；PPR 将其排到第 2 位，α0 则没有这些子候选。这是具体可追踪的图行为，不能再沿用旧 M3“所有候选本来都直接命中”的诊断。

同时需区分几项设计选择：

1. `m3_1_walk_factorization.py:139–180` 只在 impedance_port 分支为全范围源补 low-regime onset 签名；对应直接测量域没有同样签名。
2. `498–511` 新增正号 S24P，明确构建 S24(-) ↔ S24P(+) → NM04 路径；`513–515` 删除 S05/S06 等原源，使四个修订节点成为单独的横向分量。
3. `m3_scale_universe.py:2009` 以 witness-only 列表替换子节点完整实测签名。**但本审计恢复完整实测子签名后，四个 safe query 仍全部零交集，α0 排名没有变化。故不能说“只要取消 witness_only 就消除了全部 walk 收益”。**
4. 索引默认赋予父→修订子 1.85、子→父 0.10、终端修订子自环 2.40、普通自环 0.25 的权重（`hipporag_wormhole_index.py:149–155,208–245`）。这是明确的“修订后继值得优先考察”的检索先验，属于机制设计；尚不是自动学会的一般图权重规律。

合适结论是“在人工制定、对各正式臂共用的 observable signature / graph construction 协议下，异质图传播补上了直接检索缺失的路径”。当前没有同元数据的简单邻接展开/反例触发后继检索对照，因此不能把“需要利用关系”进一步证明成“必须采用 Personalized PageRank、不可用其他图搜索”。有限受控结构有验证价值；应提高主张精度，而不是因受控性直接否定它。

### 5.4 安全低载：共享的是 pilot 策略

正式 forest 臂使用 `M3_1FactorizedResearcher` hooks（`m3_1_walk_factorization.py:394–420`），普通/HExA 也通过同 hooks 执行（study `101–113,277–284`）。因此初始低载约束在正式比较中确实对所有臂一致。

实际 pilot 范围：T25 为 [0.04,0.22]；T26 为 [0.04,0.21] 的四点；T27 为 [0.02,0.10]；T28 为 [0.005,0.018]。不能统称四题都在 [0.01,0.22]。

`OpenWorkspaceTaskLab.acquire` 只按全 feature_bounds 裁剪并记费（`open_workspace_labs.py:128–160`）；没有根据 pilot_regime 拒绝高载。四题都能在最初 `phase='presolve'` 请求 0.90 范围并得到读数。确认点函数则直接给出 0.66–0.82 等高载位置（`m3_1_walk_factorization.py:86–122`），fallback 还能用普通高范围 exploration points。

因此当前已证明的是“先给低信息 pilot，随后允许付费高载确认时，图先验可少做探索”的条件收益。尚未验证“危险实验必须凭历史知识才能合法解锁”“主动实验设计者在同一安全规则下没有更便宜的高载观测策略”。本次没有另造主动基线来改变正式试验，也不宣称其必然胜过 PPR。

## 6. K 扩容与桥接账本：应保留的正面判断

### 6.1 K=20/36/52 是实质改进的受控扩容

`make_m3_1_factorized_walk_suite` 先构建相同基础套件，K=20 选择 16 个 essential 源，K=36 为 32 源，K=52 新增 16 源，三档均加相同四个 near misses。新增源循环不消耗后续目标参数使用的 RNG；目标生成在 K 分支之后共用同一逻辑（`m3_1_walk_factorization.py:629–693,695–818`）。

本审计对三档全部 27 目标的公开 task_view、边界、噪声种子与相同审计采样点的 oracle 值做指纹，三档完全一致；同时静态核对目标生成闭包没有随 K 分支改写。正式 K 研究使用固定 10 个 Assoc/Walk 目标、相同两 seed、同合成器与计费（study `335–395`）。因此不应沿用旧跨里程碑比较的“同时改变任务/算法/计费”批评。

仍应保留其范围：这是一种指定四类额外机制、指定 source membership 的抗干扰扩容，不是任意异质知识流、真实语料规模规律或无限持续学习保证；主表 K 曲线只包含指定 10 类目标的两个 seed。

### 6.2 新 M3.1 没有 bridge log 并非漏计复发

旧 M3 在源阶段尝试 rational_monod → congestion_rollover 的近似 bridge，条件是两类源均已存在（`m3_scale_universe.py:1864–1867`）；新增日志在 `1887–1900`。

新 M3.1 删去了原 S05/S06 rollover 源，把 rollover 改为稍后 near-miss 阶段形成的 NM01 子节点。bridge 阶段在 near-miss 之前，因此 `roll_eps` 为空。小控制包装实际 synthesizer 记录 `task_id == m3_bridge_fit`，得到 **0 次调用**；源阶段 phase 为 source=32、source_nearmiss=4、source_scale_calibration=1、index_build=1。

没有执行的工作不应收费。不能因为新 M3.1 `source_episodes.jsonl` 没有 source_bridge_calibration，就报告修复回退或遗漏 meter。旧 M3 closure 测试证明旧 bridge 调用已记账，新 M3.1 则是取消了该特定 source bridge 路径。

## 7. 建议优先级

**下一小轮首先处理：** 修正一次失败令四个机制为真的证据升级；拒绝重复 predict / 数学白名单重绑；严格要求开放任务 receipt 身份；规范确认 API 的相对路径；改正文档 C4 对照对象、2×2 完整性和召回阶段确认预算的措辞。

**随后补最小因果证据：** 将 α0+NoPool 纳入正式登记，并报告 pool × walk 的条件效应；给“只用低载 cue 的关系展开”增加简单、同信息、同计费的图搜索对照，才能比较 PPR 本身的必要性；若主张真实安全约束，则在实验室层定义可执行的动作权限和解除规则，让所有候选策略遵循同一规则。

**继续保留：** 关系图候选召回、可证伪的父子修订、观察透镜、隔离前缀确认、按工作量收费、条件错误恢复库和当前固定 K 控制。它们已经出现真实执行作用。当前发现要求的是把主张约束到已经获得的证据，并收紧几个有限代码边界，不要求回退整个研究路线。
