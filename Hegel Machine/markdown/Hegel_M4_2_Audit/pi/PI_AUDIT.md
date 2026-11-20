# V17.4 / M4.2 可执行 Pi、尺度授权链与实例分发器审计

固定提交：`fb4b23aaf802c20b4e2d0018436abab73b0cca06`。比较基线：`99b1fdfebb43cf31b7f1fc5386bd8fa8a21fa75c`。

## 1. 判断与范围

**本轮确实把原先分开的 Pi 节点、真实快慢动力学、析取压缩、经验失效、盒收缩与冻结实例分发接到了一个构建流程。不能继续沿用“Pi 没进森林”“R3 完全没有恢复旧参数”“真实授权根本不读析取分支”等旧判断。**

但 `100% 闭合` 的报告口径仍过强。最重要的剩余缺口是一条共同的尺度授权链：证书没有绑定它所认证的盒和具体实例，重验证后的边保留旧 ScaleMap/旧证据，Stage3 没有继承 Part3 的最终查询盒守卫。这里有真实生产函数上的数值失败见证，不能仅靠正常盒内回归通过来关闭。

另外，Act3 的分发器有实际作用，但其“同一个 post-Stage2 森林、沿粗粒化边升级、完整零 fit”的表述需要精确限定；窗口投影已被调用，但目前真正影响证书的是末态快流形残差，尾窗平均值仍只记入报告。

本分工已读仓库 `AGENTS.md`、相关历史复用 hub、M4.1 Pi 审计、M4.2 handoff/report16、新模块 Pi/Stage0/Stage2/Act3/Stage3 全相关路径、P/W 相关 diff、新测试与已提交 Pi/R3 证据。历史检索 `cognition historical --query pomdp --compact` 保存为 `history_pomdp.json`。RK4、Schur 减阶、收费双通道、边版本检查、冻结重放、角色映射均已有实现，建议以接通现有消费者为主。

本文记号：

- **U** = `src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py`
- **P** = `src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py`
- **W** = `src/hegel_machine/cognition/wormhole_forest.py`
- **C** = `src/hegel_machine/cognition/m4_compositional_discovery.py`
- **R** = `scripts/run_v17_4_m4_2_unified_study.py`

没有修改源码或原证据、没有调用付费 API。完整 37 项回归和全 study 由主审负责，不在本分工重复。生产森林直接复用另一分工调用原 builder 得到的 `../unified/shared_built.pkl`，全部控制只操作其内存副本。

## 2. 应确认完成的修复

| 项目 | 当前判断 | 直接依据 |
|---|---|---|
| Part3 确认越盒后的细尺度升级、最终查询拒绝 | 原专门 Pi suite 中已实际连接 | P:620–644、742–773、413–445；原来只测 helper 的缺口已补成真实 solver 路径 |
| Part3 无边 fallback 重购 pilot | 正常路径已修 | P:734–749 复用 prefix；已提交对照从旧 `97.7765` 回到 `73.7765`，有边 `72.6095`。保留真实图可达性效应，但原 24 测量差不再算图的必要收益 |
| 可执行 Pi 纳入统一森林 | 已完成对象集成 | U:964–1055 注册 fine/coarse 节点与真实初始证书，Stage2 修改的就是同一 coarse 节点 |
| 淬火导致 closure 失效，再标定后恢复 | 正常路径确实执行 | U:1669–1738；报告 quench defect `0.324013`，未闭合重绑被拒绝，节点到 v4 后边恢复 |
| thermal / hydraulic 析取分支真实授权 | 已连接授权器 | U:1572–1605；对应 coarse 节点 `thermal=True, hydraulic=True, optical=False` |
| 冻结参数推理 | 确实恢复已存参数 | C:1135–1160 将 `_PARAMS` 置为 artifact 参数并删除 `fit`；不是再用默认参数冒充旧知识 |
| 轨迹参与证书 | 已新增真实因果门控 | U:306–340 传真实轨迹，末态 slaving 超限可使证书失败；不能再说 `projections_with_fine_trajectory=0` |

这里的“正常路径已修”不等于所有新消费者已统一继承了约束，下一节即为该区别。

## 3. P1：尺度授权链没有闭合——证书、边、实例、最终查询不是同一约束对象

### 3.1 合法窄盒证书可以被用于扩大盒，随后 R3 授权错误的粗预测

`revalidate_vertical_edges_with_certificate` 增加了实质检查：`certified is True`、有限非负 defect、`defect <= tolerance`、`closure_scope != not_closed`、被丢弃变量名字集合完整（W:1082–1123）。这些改进有效。

但它不比较 `closure_certificate.verified_input_box` 与 `updated_scale_map.window_or_partition.verified_input_box`，不核对证书对应的算子/参数/模型/节点版本，也没有把轨迹残差与其阈值重新纳入此接口的检查。之后直接做版本重绑（W:1124–1144）。这不是要求经验采样变成数学证明，而是要求**已取得的某一域内证据只授权该域**。

控制使用原函数生成的真实 6 点证书，ODE 参数和观测均不改：

| 条件 | 原 post-Stage2 森林 | 只扩大 ScaleMap 后 |
|---|---:|---:|
| 证书真实认证 T 盒 | `[2.4, 5.2]` | `[2.4, 5.2]`，未变化 |
| 证书最大 endpoint defect | `0.032846` | `0.032846`，未变化 |
| 拟提交 ScaleMap T 盒 | `[2.4, 5.2]` | `[0.05, 5.2]` |
| 证书重验证结果 | 原状态有效 | `accepted=True`, v5, rebound coarse edge=1 |
| 同一 `(u=2,T=.12)` 查询路由 | fine ODE | coarse Pi 节点 |
| 实际误差 | `0.0` | `0.32400957`，超过 coarse 节点容差 `0.05` |
| 返回执行状态 | `dispatched_ok` | `dispatched_ok`, `local_solver_guidance` |

扩大盒并非当前 builder 的行为，所以不能把这个数值当成已发布 80 个域内目标的错误。它证明的是新“生产级证书重验证”接口尚不能防止一份真证书被错误绑定到更广的作用域；不是伪造 `certified=True` 的无意义攻击。

证据：`dispatcher_scope_results.json` → `certificate_scope_binding`；`probe_dispatcher_scope.py` C 段。

### 3.2 正常 Stage2 重验证后的边仍携带 Stage0 的旧盒、旧证书

原 builder 运行结果中，coarse 节点 v4 的 T 盒为 `[2.4,5.2]`，其 coarse edge 的 `scale_map` 仍为 `[2.2,5.2]`，`empirical_closure_certificate` 也仍是 Stage0 初始证书。原因很直接：W:1022–1066 的底层重绑只更新端点版本、清 stale 和追加 revision provenance；新接口 W:1136–1140 仅另写一个新证书 digest，没有替换边的 ScaleMap 和证书对象。

节点的 `authored_artifact_path` 同样从 U:1039 起一直指向 Stage0 artifact；U:1706–1738 创建新的 Stage2 标定 artifact，却只更新 node ScaleMap，未切换冻结实例。**由于本例 ODE 参数未变、盒只是收缩，继续使用旧已拟合模型可以是合法设计，本分工没有把它断言为数值错误。** 需要修的是身份/证据表达：应明确“同一冻结模型在更小域重认证”，并将该模型 hash、参数快照、盒、证据、节点版本绑定起来；或者切换到新的已确认实例。现在的 node/edge/certificate 三者不一致，使下游消费者难以知道哪份范围才是授权依据。

可直接复用 W:800–813 的 `fine_scale_children_of` 及现有版本检查，不必重写整个图系统。

### 3.3 Stage3 只守 pilot+confirm，最终查询没有 Part3 已有的范围拦截

U:2338–2354 对 `target.pilot_points + target.confirm_points` 检查 vbox；U:2544–2556 随后生成 heldout rows、直接执行模型，没有对应盒检查，也未写入/消费 P:765–770 已有的 `model_scope_contract.json`。

在真实 `evaluate_m4_2_target_under_arm` 中，只把原 `M42_T17_PiSAdiabatic` 的最终查询改为同一 ODE 的 `(u=2,T=.12)`，pilot、确认、模型、参数拟合、森林全部保持：

| 条件 | 原始目标 | 仅最终查询越盒 |
|---|---:|---:|
| 召回节点 | Pi coarse v4 | Pi coarse v4 |
| `confirmed_via_recall` | `True` | `True` |
| `status` | `completed` | `completed` |
| heldout RMSE | `0.0`（输出六位舍入） | `0.218132` |
| `strict_pass` | `True` | `False` |
| `false_confirmation` | `False` | `True` |

这是明确的盒外续问控制，不是把盒外点冒充原有域内结果。外部质量评估发现了失败，但范围契约没有在执行前拒绝、返回 unknown 或升至 fine；这正是上轮 Part3 修好、这轮新 Stage3 没继承的断点。

### 3.4 跨域角色改名还会让局部盒检查 fail-open

U:2344–2349 使用 `for k in vbox if k in pt`，缺失字段被跳过，空迭代的 `all(...)` 为真。分发器自己的 `_point_in_box` 也有同样语义（U:450–459）。P:210–219 的现成 helper 则对缺字段返回 False，两套守卫已分叉。

原 80 目标中的 Pi 家族确实仍使用 `u_drive/T_horizon`（U:2195–2217），**不能声称正常提交数据已经因改名绕过守卫**。控制对同一合法物理任务仅做可逆角色改名：`u_drive → u_probe`、`T_horizon → t_probe`，oracle 做对应字典映射；首个确认点保持物理 `T=1.2`，位于已验证盒外。

- 保留原名：Pi 召回候选被盒门控排除，最终走 fallback；`confirmed_via_recall=False`。
- 等价改名：Pi coarse v4 被召回并确认，`confirmed_via_recall=True`，heldout `strict_pass=True`；相同物理范围条件已失去约束。

数值拟合恰好能在这个近盒外点成功，并不使证书获得更广授权。当前 canonical-role 重编译能力允许新特征名，是合理迁移能力；应先把 query 映射至证书的规范角色坐标，再检查 box。未知映射应拒绝该证书授权，不能按“没有可比较字段”放行。

这些发现应合并为一个 P1：**统一尺度契约的绑定和消费**，而不是把每一个同根断点都包装成独立最高优先级问题。

## 4. Act3：真实分发成立，但验证场景临时补了规则，边并未参与选路

### 4.1 optical 的拒绝不来自原 post-Stage2 森林

原 fine 节点注册时只要求 `active_arity == 2`（U:964–998），没有 modality 限制。`run_interleaved_zero_fit_dispatcher_stream` 的 U:1974–1984 在开始执行 10 条查询前，直接在传入森林上添加 thermal/hydraulic 两个 OR 分支，注释明确说明是为了让 Q10 optical 被拒绝。

实际控制得到：

- 原 builder 的 post-Stage2 森林直接处理 optical：`dispatched_ok → ode_micro_2mode_node`，授权 `local_solver_guidance`。
- 同一森林经过 stream 函数现场补规则后：`scope_unauthorized`。
- fine 节点的合约发生变化，但版本 `1 → 1`，`semantic_digest` 完全不变，也没有走 `empirical_revise_node`。

主 study 在 U:2594 保存 post-Stage2 snapshot，U:2598 只把它的深拷贝交给此 stream，Stage3 又从未补 guard 的 snapshot 分别建臂（U:2623–2626）。因此，Act3 的“未授权模态被生产森林干净拒绝”结论只对经过测试函数补规则的副本成立；它没有验证相同语义状态被后续消费者继承。

修法应把 modality 合约放入正式构建或有证据的版本修订，然后让 stream 作为只读验证器。新增规则要触发版本/digest/edge失效的已有机制。

### 4.2 去掉所有 coarse edges，10 条路由和预测完全不变

U:470–483 取调用者给定 `candidate_tree_id` 内的所有活节点，按 leaf/mid/root 排序；U:488–493 在 coarse box 不满足时跳过它，继续遍历其他同树节点。整个 `dispatch_predict` 没读取 `coarse_graining_edges`，没有调用 `fine_scale_children_of`，也不检查这次升级边的 stale/version/certificate。

控制把真实森林的全部 coarse edges 删除，再跑相同 10 条流，`status/selected_node_id/selected_version/prediction/escalated_from_coarse_node` 全部逐项相同，路由仍 `10/10`。

**这不意味着分发器无用。** 它真实检查 revised predicates、OR 分支、范围，真实加载冻结实例，并在指定树中选择节点；旧 Monod 与新 Haldane 共存和无目标重拟合推理是有效进展。当前实现应称为“给定候选树的合约分发/父节点回退”，还不能把这个实验当成“粗粒化边支配细尺度升级”的因果证据。与 Part3 PPR coarse-edge 可达性控制必须分开讲，后者的边效应仍成立。

### 4.3 实例推理零目标拟合成立；完整冷加载零 fit 与证据绑定仍不成立

C:1135–1160 的 frozen executor 删除 `fit` 并恢复 `_PARAMS`，确实避免在此次 query 上重拟合。应保留这一正面结论。

但 U:506–510 调用的 `load_verified_authored_method(require_external_manifest=True)` 会在冷验证时用历史 prefix 重新 fit 源码并重放 confirmation。`fit_called=False` 只来自最后 executor 的返回标志，不是完整 dispatcher 调用的运行时 fit 事件审计。这不等于目标泄漏；更精确的名称是“零目标重拟合实例推理，允许历史证据校验重放”。

更关键的参数身份问题由契约分工独立控制：真 W1 artifact 只把内嵌 `fitted_params.p0` 放大 3 倍，源码、manifest、bundle、`confirmed_fitted_params.json` 全不变，严格 loader 仍通过；frozen confirmation RMSE 从 `5.49e-10` 变成 `7.68959`，容差为 `.045`。这是“loader验证重新fit出来的实例，executor执行另一份未绑定快照”的对象错位。参见 `../closure/contract_controls_results.json`，不要把它另算成多个重复漏洞。

fine 特例也需明确可信边界：U:500–502 使用字面串 `fine_ode_rk4_exact/fine_ode_cal_0001`，跳过 artifact 加载；U:518–540 为当前 query 现场构造 `passed=True, observed_error=0.0` 供授权，U:547–554 直接执行注册的 ODE 参数。因此这一条是宿主提供的可信精确求解器，不能和经过外部确认绑定的发现实例混称同一来源等级。没有必要放弃 trusted solver，但应记录真实配置 hash、来源和授权类型。

## 5. P2：真正接入的是末态 slaving residual，窗口平均值仍是诊断

P:189–199 的 `window_projected_y` 使用末尾最多 5 点的慢态平均；`fast_manifold_slaving_defect` 只使用最后的 `x_slow[-1], x_fast[-1]`。U:324–335 把前者写入记录，后者进入 `max_slav_defect`，U:340 只比较 endpoint defect 与 slaving defect。粗通道预测仍由独立 Schur ODE 产生。

为隔离数据依赖，本分工仅在内存中改变传入投影函数的轨迹，未改 ODE 观测、粗读出、代码或证书阈值：

| 依赖控制 | 窗口值变化 | 末态 slaving | `certified` |
|---|---:|---:|---:|
| 原轨迹 | 0 | `0.016419` | True |
| 只改尾窗非末点慢态 +100，保留末态 | 最大 `67.2` | `0.016419`，完全不变 | True |
| 只改末点快态 +1 | 不依赖此字段 | `1.016419` | False |

这是依赖隔离控制，不是声称修改后的轨迹仍满足原物理 ODE。它同时证明两个方向：新增 trajectory 参数确实因果影响授权，窗口平均输出目前没有因果影响证书。C9 应写成“新增真实轨迹末态快流形门控，尾窗投影保留作诊断”，或者后续明确窗口残差的消费任务再接入。不能因为调用计数达到 6 就说任意窗口输出已经支配选尺度或后继预测。

本报告不再把“有限网格无法排除任意窄 spike”作为新缺陷；它已披露，也不影响上述具体软件绑定问题。

## 6. P2：归档不足以原样恢复重验证后的证据状态

R:301–318 在临时目录运行 study，然后只复制 `stage0_metered_sources` 和 `persisted_stage1_authored_workspaces`。`stage2_lifecycle/stage2_pi_s_reverified` 的新证书产物没有复制，随后 TemporaryDirectory 删除。

U:1851–1858 的输出只保留 quench defect、重验证 accepted、盒、轨迹次数；不保留完整 Stage2 证书、6 个 projection_records、manifest、frozen snapshot。已提交证据里的 Stage0 artifact path 仍是 `/tmp/hegel-v17-4-run-...`，归档虽然有对应 Stage0 实体，但没有保存重定位后的森林快照及新 Stage2 证据。读者可以重跑重建本例，不能从提交产物直接恢复当时 v4 的全部可核查来源关系。

应归档带相对可重定位路径的 post-Stage2 forest、每个被引用 artifact、对应确认 bundle、参数快照、证书及其绑定。对于“旧冻结实例、小域新认证”的策略，只需保存实例引用和新证书，不必重复整套代码。

另一个静态接口不一致：U:2323 传检索 `scale_regime`，但 `HippoRAGWormholeIndex.retrieve` 在 `hipporag_wormhole_index.py:406,418–429` 读取 `target_scale_regime`。因此这一 Stage3 caller 的 `scale_aware=True` 不会启用预期尺度重排。此项无额外数值探针，不据此推断默认结果一定变化。

## 7. 最小修复顺序与验收

1. **统一尺度授权对象。** 证书绑定规范角色盒、ODE/算子身份、参数 snapshot hash、节点版本和真实证据摘要；updated ScaleMap 必须被证书覆盖。重验证事务同时更新 node/edge/certificate，或明确保存“新证书复核旧实例”的引用。验收即本报告合法窄盒证书扩盒控制必须拒绝。
2. **把现有范围守卫推广到所有消费者。** 复用 P:210 的缺字段拒绝语义，角色映射后统一检查；召回、fallback、frozen dispatch、最终导出/续问共享同一个 scope 决策。越盒可拒绝、重新标定或使用有有效边/合约的 fine solver。验收：Stage3 T=.12 最终查询不再 `completed + false_confirmation`；角色改名不改变授权结论。
3. **让 R3 验证器只读。** 正式构建/修订 fine modality 合约，使用正常版本生命周期；分发升级显式通过现有 `fine_scale_children_of` 或将文档限定为树内回退。删除边控制是否应改变行为，取决于正式语义，但声明与实际消费必须一致。
4. **绑定实际执行的参数并明确零拟合口径。** 将模型源码、角色、透镜、参数、确认记录组成同一 confirmed instance；验证和执行同一对象。允许冷加载历史重放时，单列其事件，不把末端字面 `fit_called=False` 当完整调用无 fit。
5. **补足归档并收窄窗口投影描述。** 保存完整 post-Stage2 状态，保留有效端点 slaving 结论；若窗口量尚无定义明确的任务作用，直接标为诊断即可。

不建议因此重写 RK4、Schur、PPR、CodeAgent 或退回静态列表。本轮大部分需要的是现成机制在同一约束对象上闭合，和让报告声明与实际运行边界一致。

## 8. 可复现文件

- `probe_projection_dependency.py` / `projection_dependency_results.json`：3 个最小轨迹依赖控制。
- `probe_dispatcher_scope.py` / `dispatcher_scope_results.json`：真实 builder 森林的 Act3、合法窄证书扩盒、Stage3 最终范围与角色映射控制。
- `dispatcher_scope_stdout.json`：便于直接阅读的控制摘要。
- `../unified/shared_built.pkl`：同次生产 builder 输出，避免重复研究构建；所有引用 artifact 留在该分工的共享 scratch 目录。
- `../closure/contract_controls_results.json`：独立参数绑定控制，本文引用不重复执行。

运行时在新提交 `Hegel Machine` 目录设置 `PYTHONPATH=src`，先执行 projection probe，再执行 dispatcher probe。所有输出写到本审计目录；共享森林文件需由同一固定提交原 builder 生成。
