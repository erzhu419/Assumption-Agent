# M2 机制与代码审计（固定 9115f9adf）

审计代码：`${HEGEL_REPO}`，固定提交 `9115f9adf9dad1f52f106a2ab14babe8b68243dd`。本报告的行号均相对于该目录。已与此前下载的四份核心文件逐字节核对相同。下列反例均在原代码上运行；未改仓库，也未作任何付费模型调用。便携脚本位于 `../scripts/`，原始结果位于 `../mechanism/`。

## 一、主要判断

M2 确实接上了真实 Python 文件生成、子进程拟合/预测、源成功准入、目标购测确认和退出复用已付数据。这是有价值的集成进展，不能因为采用受控机制族便否定。

**不要把有限模型库本身列为 P0。** V166 路线允许有限语法作为初期受控研究范围。当前需要区分的是“有限库实验结果可信且有用途”与“报告把它宣称成已完成开放代码自主研究接线”；主张超出实现和接口中已复现的错误才是本轮修复对象。

但当前实现最准确的名称是“以 10 类固定公式族为候选、使用 AST 模板归组及前置条件路由的受控跨域参数重拟合实验”。它尚不是报告所称的通用开放 Python 科学研究回路、对任意新代码进行语义骨架归纳、残差驱动新状态创造，以及由实测链映射证明的图结构纯增益。很多已有先进能力并非整个项目没有，而是这次 M2 入口未消费。

## 二、已完成且应保留的进展

1. 所有 M2 任务均走 `OpenWorkspaceTaskLab`，购测数据真实写 `observations.json`、模型真实写 `model.py`，合成和评分确实启动独立 Python 子进程（`open_workspace_wormhole.py:455–553`；`open_workspace_labs.py:59–105`）。这不是仅在 JSON 中声明成功。
2. M2 源节点准入检查 `ev['strict_pass']`，未过源外评不会登记 supported AST 节点（`open_workspace_wormhole.py:1039–1040`）；near-miss 也有同类检查（1223–1224）。因此旧版“失败来源仍一律 supported”不能不加区分重复指控。
3. 运行错误以 `EXECUTION_ERROR` 返回，目标评分不会把 Python 崩溃当科学反驳（498–505、540–546）。当前已有测试覆盖 ZeroDivisionError。
4. 目标迁移确实先试探、购测新点、取得误差后调用授权门（1561–1635）；不是仅按源支持就不经目标验证应用。
5. 退出后调用普通求解器时带着同一个 lab 和已付观测，未清空数据重新收费（1637–1638）。
6. 参数拟合不是直接读目标 `_oracle` 参数或 heldout 数值；实验框架的共享模拟器是受控任务生成，不应称作目标答案泄露。

## 三、核心机制缺口

### F1：开放工作区已经成立，“开放假设生成”尚未成立（主张层面的关键缺口）

`_WORKSPACE_SYNTHESIZER_SCRIPT` 为一段固定脚本，每次原样写入工作区再运行。实际可选 schema 恰好 10 个：

- `linear_1d`、`rational_1d`、`inv_quad_crest_1d`、`congestion_rollover_1d`；
- `conjugate_relaxation_reset`、`conjugate_relaxation_state_memory`、`bimodal_relaxation_2mode`；
- `liebig_min_2d`、`liebig_capped_min_2d`、`cobb_douglas_sqrt_2d`。

脚本按 active arity 和 `allow(tag)` 枚举已写好的数值拟合器，`allowed_schemas=None` 的含义是“在这 10 类模板中搜索”，并不是任意 Python 或开放算子组合。生成的 `model.py` 只是把已拟合常数填入预设表达式，`fit` 函数为空操作（`open_workspace_wormhole.py:132–452`、426–437）。目标重新拟合时消费的是字符串 `schema_tag`，不是源 Python 机制模块；`shared_wormhole_modules` 只被写入，从未用于求解。

主程序中没有 `CodeAgent.run`、`CodeResearchSession.run`、真实 provider 决策。引用 `code_agent.evaluate_python_model` 仅复用了评测器，不能等同接入已有自由代码 Agent。

这一阶段保留有限语法完全合理。需要修改的是里程碑口径，以及下一步应把它作为受控基线/工具接回原生研究回路；不应继续声称已跨过开放假设空间这一关。

### F2：AST 规范化相信旁注，而没有从实际可执行程序提取语义（可复现）

`canonicalize_python_model_ast` 优先解析 `_CORE_LAW_EXPR` 字符串；若缺失，只取 `predict` 的直接 return 表达式。它没有把 `latent` 变量定义内联，没有跟踪辅助函数，也没有从任意仿射读出程序中证明/提取所谓透镜（`open_workspace_wormhole.py:58–126`）。当前生成器恰好自己提供正确旁注，因此原 40 题并不会由此自动失效；但是已有自由 Python 改稿后旁注陈旧、或没有这种旁注时，行为就会不正确。

`probe_ast.py` 直接抽取并执行原始规范化器定义，得出三组反例：

| 输入 | 结果 |
|---|---|
| 同一个 `_CORE_LAW_EXPR`，一个 `predict` 返回 x，另一个返回 100000 | digest 完全相同 `780eff9b3b0733` |
| 两个没有旁注的程序，分别 `latent = x/(.28x+1.6)` 与 `latent = x**5+y`，都返回同样仿射包装的 latent | digest 相同 `bc7c4cf63685fd`，提取的角色列表都是空 |
| `x**2` 与 `x**-2` | digest 相同 `7d64aec742b140` |

第三项来源于把所有数值常数、包括指数和负号都抹成 `NUM`（71–81）。把“可拟合的系数”泛化是合理的，把决定机制性质的指数/符号无差别抹掉不合理。

修复边界应很小：当前有限生成器至少校验旁注与实际已执行 core 的一致性，记录模型源码内容 hash；未覆盖的自由代码返回“骨架暂不可提取”，不要假装相同。后续连接已有 Python 回路时逐步支持明确的纯函数子集，不必追求判定任意程序等价。

### F3：近失例修订是预设两分支，未形成报告所说的反例驱动科学修订链

near-miss 在 `form_and_revise_in_workspaces` 中先独立调用普通拟合器，得到最终模板，再按两个预写分支执行修订：

- `unreset_third_state_active and nm_schema == 'conjugate_relaxation_state_memory'` → 写入固定 memory 排除和固定新节点；
- `nm_schema == 'liebig_capped_min_2d'` → 写入固定 high-corner 排除和固定新节点。

位置：`open_workspace_wormhole.py:1195–1363`。

这条路径没有先用被修订的旧模型在当前 witness 上执行预测、记录残差反驳，再定位哪条前提/表示闭合失败；新 memory/cap 结构本来就在合成器候选集中，`m0` 也已在公开特征列表内。`_extract_pilot_signatures` 甚至对任何 3+ 活跃变量直接宣称 `unreset_third_state_active=True`（707–712），并不检验第三变量造成何种历史依赖。因此它证明的是既有状态扩展模板被选择并保存，尚不能证明一般“状态发明”或自主 CEGAR。

此外 `solve_target` 对未见目标失败只回退，不把新失败和改稿写回森林；目标之间没有当前 M2 的新的世界知识积累闭环（1468–1666）。这不必作为 M2 第一版必修，但限制 RSI 论断。

### F4：两跳链预算与路由条件预置，不能承担“首次图结构严格优于等内容平库”的核心证据

所有初始虫洞写死 `hop_error_budget=0.015`、Lipschitz=1（1142–1143）；当 schema 是 `rational_1d`，额外手工造一个 relay 节点，无独立 relay 来源实验就标 `SUPPORTED_IN_SCOPE`，再写死第二跳 0.018（1156–1193）。目标 T_M2C4 明确公开 `two_hop_bridge_only=True`（`open_workspace_labs.py:724–740`），触发人工安排的两跳检查（wormhole 1536–1555）。

求解器没有执行两次实际映射/程序运输，也没有从两跳实验分别估计误差、Lipschitz 或映射作用域。0.033 > 0.025 作为误差界算术/门控单元测试是正确且有价值的；它没有证明当前目标的有效路径必须经过这两条边，或者这两条边的误差界由源实验支持。底层 C4 是作者指定的 congestion rollover 函数，不是实际两跳映射的实测合成。

Flat 还忽略了已复制的 `self.gate`；只检查 pairwise 条件且不消费完整链元数据。因此目前比较混合了额外科学信息/门控，不能视为仅改变数据结构。结果对照由另一审计任务负责，本文仅定位机制。

## 四、会直接导致新任务错误的接口缺陷

### F5：新点确认失败后，回退仍作为 completed 提交（真实复现）

`solve_open_workspace_direct` 在 fresh confirmation 没过、乃至评测执行错误后，又运行一次有限全模板拟合，只要这个脚本不崩就返回 `status='completed'`，没有再次质量门控（`open_workspace_wormhole.py:789–825`）。上游 `run_workspace_python_synthesizer` 也对成功运行的高误差候选返回 `outcome_status='SUPPORTED_IN_SCOPE'`（509–515）。

`probe_runtime.py` 使用相同公开 lab API、同预算，唯一新规律为 y=5+sin(u)，u∈[1,10]，tolerance=.01：

- 真实 fresh confirmation RMSE = **1.2897950866889305**；
- 仍返回 **completed**；
- 31 点外评 RMSE = **1.091451**，`strict_pass=False`，`false_confirmation=True`。

没有使用任何隐藏答案帮助求解；此例只检查“确认明确失败时是否还会宣称完成”。修复可以直接保留模型作为未确认候选，返回 `scoped_unknown`/`unconfirmed`，允许有预算时继续既有研究；失败本身不应删除或重新收费。若还要 refit，旧 receipt 不能自然授权新源码，需要冻结原候选或对新产物再确认。

注意：`completed` 有时可仅表示流程结束，但仓库的 M2 外评把这种错误提交定义为 false_confirmation，报告也把 0 假确认当底线；不能同时用“只是返回了个模型”规避实际验收语义。

### F6：角色对齐取决于公开字段顺序，当前“坐标无关”表述不成立（真实复现）

两维 pilot 和签名默认第 0 个特征是 drive、第 1 个是 time，三维默认第三个是 memory；目标使用这些位置直接作为 `forced_role_order`（586–622、689–712、1582–1586）。源合成器虽然能枚举角色，但源码实际找到的 `role_order` 并未在源记录时被采用：1017 使用 `_extract_pilot_signatures` 的顺序。

在同一个 `make_m2_open_workspace_suite(5026101)` 的 T_M2A1，只把 `feature_names` 列表逆序，保持名字、数值边界、模拟器和评价题全部不变：

| 条件 | 测量费 | 复合费 | 外评 RMSE | 结果 |
|---|---:|---:|---:|---|
| 原字段顺序 | 30 | 31.925 | .017834 | pass |
| 字段逆序 | 55 | 58.5 | .049067 | fail（阈值 .045） |

这里用的是默认 gate 的 focused probe，不是冻结论文协议的再汇总，不应将 30 与报告中的 35 混比。此反例说明列顺序可以导致误分类、不同购测位置和失败；既然初衷是跨领域角色对齐，这比扩到更多命名领域更应先解决。可复用当前合成器已有角色枚举，把作用域探针与映射候选在目标上验证，而非要求所有输入按作者预定语义顺序排列。

### F7：receipt 没有绑定目标、节点版本与实际模型，且可缺失误差（真实复现）

`wormhole_forest._validate_evidence_receipt`（155–170）只检查 passed、claim_id 和有字段时的 observed_error，没检查 target_id、node_version、被测模型内容 hash、确认数据 identity，`claim_id='*'` 也可接受。若 observed_error 缺失，直接通过；NaN > tolerance 为 False，也通过。

三个 focused controls 都取得 `local_solver_guidance`：

1. 新目标/new version 使用旧 target_id 和旧 node_version 的 receipt；
2. receipt 完全没有 observed_error；
3. receipt observed_error 为 NaN。

当前 M2 主循环每次现场构造正常有限误差 receipt，未发现原 40 题利用了这些缺陷；但是报告所谓“收据式四级权限闭合”超出实现。M3 增大记忆库/跨版本复用会让这类误用变成真实路径。最小修复是绑定当前任务、当前节点、当前模型以及 fresh 检查身份，并要求 finite error 与阈值判定；不用更复杂的权限层级。

### F8：program_steps 并未通用执行，只有 schema 参数被消费（真实控制）

1580 附近把 program_steps 转成 op→dict，只读 `synthesize_scoped_python_schema.schema_tag`。calibration 在循环开头固定执行，confirm/refit 在主机代码固定执行。删除节点中所有 `calibrate_observation_lens` 和 `subprocess_confirm_and_refit` 步骤，仅保留 schema 步骤，目标输出、每项成本与所有外评预测完全不变（runtime probe 表三）。

这可以保留为明确的固定执行器协议，但当前 mutation test 只删 schema op，不能证明研究步骤程序真正被解释执行，更不能证明元方法自我修订。

### F9：开发双 fork 并不拥有独立相同起点的随机数流（真实控制）

Lab 的 `_oracle` 为携带 `local_rng` 的闭包（`open_workspace_labs.py:290–295`）。`run_paired_post_pilot_rollout` 通过两个 deepcopy 创建分支（wormhole 1441–1443）；Python deepcopy 函数对象不会复制其闭包状态，两个 lab 的 oracle 仍为同一个函数。

实际结果：`copy_a._oracle is copy_b._oracle == True`；同一点两个“首个”观测分别为 -4.3040725196053335 和 -4.306901387009956。它们按执行先后消费同一 RNG，而非独立克隆的 post-pilot 状态。噪声仅 .002，不应先声称这一定改写胜负；但需要通过已有 laboratory checkpoint/restore 或明确独立 stream fork 修复，之后重新冻结门控。

### F10：合成时间为常数占位、未完整绑定验证上下文（计费与可追溯性）

`run_workspace_python_synthesizer` 计时硬编码 `lab.record_python_wall_time(0.02, ...)`（495–496）。真实 eval 计时是测出来的，但 `python_execution_seconds` 混合了 0.02 合成占位，不能称全部实测子进程时间。compute 的 10/24/6 是定义的逻辑费率，可以用但应报告为 fee schedule，不应与真实 CPU 时间混用。

研究脚本将所有工作区放在 `TemporaryDirectory`（脚本145 起），结束后临时目录及未导出的中间拟合/确认上下文被删除。不过最终目标 `model.py` 的完整源码已经导出到 `target_episodes.jsonl` 的 `authored_model_py` 字段，最终目标观测也保存在 `target_observations`（脚本244–265）；不能说所有模型源码或全部目标数据都丢失。当前确定性模板可根据已导出内容复算。下一步接自由代码时，receipt 仍需绑定实际被验证的模型版本、训练前缀和执行上下文/hash，避免仅靠临时路径或后续重建来推断确认身份。已有 code_agent 保留执行上下文和 checkpoint 基础设施可直接复用。

### F11：fresh confirmation 的标签会进入 Python fit（真实复现；当前模板不受影响）

这一缺陷由 root 定位，本文用合法 `fit`/`predict` 模型验证。M2 调用 `lab.acquire(c_rows)` 后，确认样本已追加到 `lab.observations`；`validate_workspace_model_on_rows` 将这份全部样本传作 `training_observations`（`open_workspace_wormhole.py:525–533`）。`code_agent.py:600–612` 只要模型有可调用的 fit 就执行；`require_fit=False` 的意思是不强制模型必须有 fit，并不是禁止 fit。

在 y=u² 的 lab 中，已付训练为 u=1,2，新的确认点为 u=3。模型只在 fit 中记住输入和训练标签，预测陌生输入返回 0：

- M2 wrapper 给 fit 全部三行，确认预测为9，**RMSE=0**；
- 相同评测器只给确认前两行 prefix，确认预测为0，**MSE=81（RMSE=9）**。

证据：`probe_confirmation_training_leak.py`、`confirmation_training_leak_result.json`。这是无需读取评价文件或私有 oracle 的普通模型生命周期错误。现有 M2 模板的 fit 是空操作，所以不能据此声称原40题发生作弊/泄漏；但当前“已接开放Python且fresh验证隔离完成”不成立。最小修复是在 acquire 前冻结 fitting prefix/模型上下文，评测只对它拟合，再单独把确认数据归档；如果随后重拟合提交新模型，需明确新的验证身份。

## 五、历史复用核查与具体修复路线

已读 `AGENTS.md`、reuse hub，执行 catalogue 查询：`CodeResearchSession` / `CEGAR` 没命中，`open grammar` 返回 v10 路线；随后读当前源码确认旧能力存在。不能把目录检索无结果写成项目不存在能力。

| 当前缺口 | 已存在实现 | 需要接上的部分 |
|---|---|---|
| M2 有工作区但没有自由代码研究循环 | `code_agent.py:1127–1386` 的 CodeAgent；`code_session.py:24–195` 的 CodeResearchSession；历史 `learned_research_procedures` 路线 | 将 OpenWorkspaceTaskLab 的 list acquire / task_view 转接原 `Probe` / `spec` / channels API，并让同一个 provider 驱动 query/write/run/evaluate/finish。有限模板留作工具/基线。 |
| 新状态/新机制不是反驳驱动 | `premise_dependency_audit.py`、`open_grammar_author.py:813+ propose_premise_gated_commitment_revisions`；registry v10 open grammar / POMDP 路线 | 旧模块吃 transition records/world commitment，而 M2 现在是标量监督样本，须明确适配残差、候选/前提与验证输入；不能只换模块名声称复用完成。 |
| 元方法程序只读schema | `python_method_learning.py:124+ PythonArchiveMethodLearner`、`temporal_paired_method.execute_procedure` 与 FrozenMethodAgent | 接真实程序体、实践、旧稿比较、冻结消费与失败保留；现固定流程不用重造通用解释器。 |
| 未确认候选被完成、receipt 版本不绑定 | 旧 confirmed model 保存与 `code_agent.archive_confirmed_python_model`、`python_model_context_matches` | 把确认身份与实际被提交代码绑在一起，复用已有执行上下文保留。 |
| 合成计时和验证上下文保留 | `experiment_runtime.ExecutionTiming`、`checkpoint_storage`、code_agent实际计时 | 用现有公共计时/保存入口替换0.02占位；在现有最终源码和观测导出上补充确认身份与上下文绑定，不改科学预算逻辑。 |

建议先修 F5/F7/F11 的确认契约、F2 的程序—骨架绑定和 F6 的角色绑定，然后修正确归因比较。M3 的 K≥32 可以暂缓科学效用主张；无需把已有有限模板成功删除，更无需先把所有潜在安全/哲学问题全部完善才允许继续。

## 六、可复核证据

- `probe_ast.py` → `ast_probe_result.json`：三种 AST 身份碰撞。
- `probe_runtime.py` → `runtime_probe_result.json`：真实失败确认、列顺序控制、program_steps控制、双fork随机状态、receipt绑定。
- `probe_confirmation_training_leak.py` → `confirmation_training_leak_result.json`：合法 fit 模型受确认标签污染；明确当前固定模板 fit 为空操作。
- `../evidence/mechanism_workspaces/runtime/` 保留该次实际生成的每个模型、购测数据、合成器配置与脚本，便于逐个复核。
- `history_python.json` 是 registry 查询的原始结果，属索引定位，不代表本次把其全部历史实测再次执行。

本报告完整静态读了 M2 的两个核心文件、M2 测试和研究脚本，并追到 code_agent 实际评测、原代码研究接口及相关授权/历史入口。没有宣称重新执行了项目全部181版或所有测试。

原始历史证据另读了 `docs/evidence/v10_world_structure_20261116/v10_phase2b_open_grammar_llm_20261116.json`（本地保存 `original_v10_phase2b_open_grammar.json`）：其确实保存实测模型调用记录与受前提定位约束的生成/验证记录；这证明项目有可复用的旧开放作者路径，不能把 M2 未接入误写成历史从未实现。这里只核对原证据和当前入口，没有重跑该历史付费实验。

## 七、简明 issue 矩阵与最小验收

| 事项 | 对当前40题证据的影响 | 下一步最小动作 | 必须满足的 focused 验收 |
|---|---|---|---|
| F5 确认失败仍completed | 原40题可全部通过，但接口并不保证0假确认 | 候选运行成功与科学确认分离；未确认保留候选/返回unknown | 本次sin案例确认失败后不再completed；原已过任务不退化 |
| F11 确认标签进fit | 当前固定模板fit为空，既有40题未受此影响 | 确认前冻结fitting prefix，验证候选身份 | 合法memorizer在新点应预测0/RMSE9；不再通过 |
| F7 receipt无绑定 | 现场正常receipt未发现误用；跨目标版本复用危险 | 校验目标、节点版本、模型hash、finite误差及确认数据 | 旧目标/旧版本/缺误差/NaN拒绝；当前有效receipt通过 |
| F2 AST依赖旁注 | 受控生成器目前自己写正确旁注；开放接线无法成立 | 源码与core一致性检查；仅泛化可拟合常数；未覆盖代码标unknown | 三个碰撞案例不得作为机制同一证据；合法改名/透镜变换仍归同类 |
| F6 特征角色靠顺序 | 已复现同物理仅逆序就失败 | 复用已有角色枚举并验证映射，不给未验证顺序强制角色 | 同一目标的字段排列/重命名保持同等可用性；失败诚实保留 |
| F1/F3/F8 “开放/CEGAR/程序执行”过称 | 不抹除有限模板实测收益 | 报告按实际缩回；需要完成开放M2时适配已有CodeAgent/修订/方法执行入口 | 至少一条公开生成非现有schema的代码/研究步骤，记录真实旧假设反例→改稿→独立后继；不以改文件后缀充当证据 |
| F4 两跳图优势过称 | 现是作者提供误差界与bool条件的门控演示 | 留作单元控制；构建真实可执行边映射，等内容flat保留同链界与门控 | 每边有独立测量/作用域；合法多跳行为产生累计误差；同信息对照才可归因图组织 |
| F9 fork共享RNG | 可能改变开发门控的细小噪声，未证翻转 | 使用旧实验室状态恢复/显式可复制random state | 两分支同起点同探针噪声一致或明确独立配对协议，顺序交换不互相改变 |
| F10 时间与产物 | compute是定义收费可保留；wall time字段不是真实完整测时 | 用ExecutionTiming和现有保存原语 | 合成墙钟为测量值；缺测标unknown；receipt引用的源码/数据可恢复 |

这些不是要求把所有边界都完善才继续。可以立即保留当前受控实验作为 M2-A，先修实际确认缺陷、修正文稿范围与因果对照，再把已有开放研究器接进同一目标接口，最后讨论大规模 M3。原有限库不应删除，旧成功证据不应清空。

### 便携运行说明

审计包中的脚本已统一使用 `HEGEL_REPO` 与 `AUDIT_OUTPUT_DIR`，通过根目录的 `run_audit.py --repo` 启动；详细命令见根目录 `README.md`。原始结果位于 `mechanism/`，实际模型与观测位于 `evidence/mechanism_workspaces/`。包装仅调整路径、导入位置和结果保存位置，不改探针输入、阈值、政策或被审计函数。全部核查针对固定9115提交，不进行外部或付费模型调用。
