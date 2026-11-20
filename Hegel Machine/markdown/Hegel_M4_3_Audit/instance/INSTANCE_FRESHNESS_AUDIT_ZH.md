# V17.5 M4.3：实例身份、冻结加载、OR 权限、确认新鲜性及已付前缀审计

审计对象：`3461535364fcdb4e9ae9c87ccba502b7e6b56fe8`；基线：`fb4b23aaf802c20b4e2d0018436abab73b0cca06`。源码根：`/workspace/scratch/3f14ad2da0ed/repo_m43_346153536/Hegel Machine`。

本分报告仅覆盖分工内的实现、历史接口和有限控制。主审执行 40 项相关测试及完整研究；本报告不重复执行这两项。源代码和提交证据保持只读。先读仓库 AGENTS、reuse hub 相关条目及 historical capability 的 `python_archive_retention`、`growth_workflow_bridge`、`learned_research_procedures` 路线；复用上一轮归档与现有真实校准、CodeAgent、加载器、dispatcher，不写另一套研究算法。

## 结论

V17.5 对上一轮参数快照单点篡改、OR 分支权限、不同法则合并、program_steps 倒序、标准路径确认点耗尽，以及 CodeAgent 已付记录交接，做了真实接线修复。不能把这些仍说成只有报告或只有字段。

但 **R3 多数节点尚未持有其应使用的 `confirmed_instance_digest`**。加载器可以验证“这个文件夹内部是一份真实、确认过的实例”，但节点没有提供它期待的实例摘要时，不能验证“这就是这个历史节点原来确认的实例”。用同一真实 Monod 程序、不同真实校准参数的合法 artifact 替换副本文件夹，R3 仍以原节点和原版本授权，预测翻倍。这是当前核心 P1。

另有两个限于兼容/旧入口的 P2：legacy loader 对两个新增 flag 的语义不一致；旧 M4 ScaleBlind 在跨测量通道升级时仍丢已付 fine 确认记录，并重复购成 fallback confirm。它们不构成对当前 M4.3 四种子主表或严格 R3 零 fit 的全面否定。

## 1. P1：文件内部的实例摘要已绑定，历史节点到实例的绑定尚缺

### 源码原因

- `m4_compositional_discovery.py:1224–1252` 对有限数值参数规范化，计算参数 hash，并把 source SHA、AST、参数 SHA、role names 纳入 `confirmed_instance_digest`。
- 同文件 `1465–1538` 检查 artifact / 外部参数文件 / manifest / bundle，并直接用冻结参数预测原始 confirm rows；`1514–1515` **确实会拒绝** `expected_instance_digest` 不匹配。
- `m4_2_unified_closure_and_pi_lifecycle.py:578–583` 的 R3 入口传入 `str(act.get("confirmed_instance_digest", "")) or None`。节点缺字段时，这个强约束会消失。
- 同文件 `918–923` 创建 `stage0_monod_2d`，`1400–1405` 创建 W1，以及其他多数 Stage 0/1 节点，仅保存 operator、AST 和 artifact path；没有保存确认实例摘要。`1859–1866` 只给 Stage 2 重验证后的 Pi 节点及其 OR 分支补上了实例摘要。

已提交 `post_stage2_forest_snapshot.json` 共 15 节点：14 个持有 artifact，只有 `ode_meso_schur_pi_thermal` 持有 expected instance 摘要，其余 13 个缺失；排除 2 个已归档节点后，12 个活跃 artifact 节点中 11 个缺失。

### 有限现实控制

脚本：`probe_same_ast_instance_swap.py`；完整结果：`probe_same_ast_instance_swap.json`。

控制采用提交中的真实 `stage0_monod_2d`，官方 snapshot 恢复函数，以及现有 `_run_metered_atomic_calibration`。另行创建增益 12 的合法 Monod 校准任务，使用真实 oracle、真实购测、真实 fit 和独立 confirm。没有伪造确认回执，没有修改任意 Python 模型，也没有修改提交证据。

原实例参数为 `p0=6, p1=2`；新实例为 `p0≈11.999999964, p1≈1.999999988`。新任务校准确认 RMSE 为 0，严格冻结 loader 正常接受。**两份 artifact 的 source SHA 和 AST 摘要都相同，实例摘要不同。** 这直接覆盖“相同法则、不同已确认实例”的场景。

将原 artifact 文件夹复制到审计目录，先做原实例 R3 正对照；随后仅替换该副本中的 `authored_method.json`、`confirmed_manifest.json`、`confirmed_fitted_params.json` 为第二个合法实例，保持内存中节点 id/version/AST 不变。

| 控制 | R3 授权/结果 | s=1.6,b=0 预测 | 对原节点真值误差 |
|---|---|---:|---:|
| 原始实例 | `dispatched_ok`，`stage0_monod_2d` v2，`local_solver_guidance` | 2.66666667 | 0 |
| 同 AST、同 source、另一合法参数实例 | 仍 `dispatched_ok`，仍原节点 v2，仍 `local_solver_guidance` | 5.33333333 | 2.66666667 |
| 同一替换，但给节点补原实例摘要 | `scope_unauthorized`，原因 `artifact_verification_failed` | 无预测 | — |

第二行仍标记 `provenance_tier=host_confirmed_discovered_instance`；回执中的 `artifact_sha256` 也与第一行完全相同，因为 `m4_2...:648` 实际存的是模型源码 SHA，不是含参数的完整实例 hash。`within_tolerance` 为 false 是该控制把外部 truth 传给诊断回执后的结果，不是 dispatcher 在输出前拒绝了它；实际状态仍为 `dispatched_ok`。

### 影响边界及修复建议

这不说明本次提交的正常 artifact 已被错换，不应据此把主表正常场景的全部 pass 改成 fail。它说明“原节点精确实例身份可防错绑”的接口保证尚未成立。R3 的用途明确是复访原实例，因此这里不能用一般 operator archetype 的可重拟合复用语义解释。Stage 3 借用结构后对目标重新拟合是另一个合法用途。

最小修复可直接复用已经正确实现的 digest checker：所有要做零 fit 历史实例预测的节点，在首次形成/正式修订时保存 artifact 的实例摘要；R3 若缺该摘要，应拒绝实例预测或进入明确的迁移验证流程。OR 分支和相应依赖边也应绑定实际使用实例；若未来支持同结构不同参数的分支，必须实际执行选中分支的 action。回执增加 `confirmed_instance_digest`、参数 hash，并把源码 hash 的字段名纠正。

新增验收应保留本控制的三个环节：合法原实例可用；合法同 AST 不同参数实例拒绝；恢复原实例可用。现有 C1 的 cross-node swap 使用 W1 与 W2，二者 AST 已不同，不能覆盖本问题。

## 2. 真正修好的部分

### 参数篡改与严格零 fit

旧问题是参数快照被改而加载器重新 fit 验证了另一套参数。新 loader 对源、AST、参数外文件、manifest 与 bundle 做一致性核对，并通过 `predict_with_frozen_instance` 直接评价持久化参数在原 confirm rows 的误差。因此旧 `p0 * 3` 单改 artifact 或同时改参数文件的控制被拒，是实际修复。

严格 R3 显式使用 `require_external_manifest=True`、`verify_frozen_only=True`。`m4_compositional_discovery.py:1535–1538` 在 frozen 验证完成后直接返回，不经过下方 isolated refit。冻结预测删除 `fit` 并要求参数 key 集合与 `_PARAMS` 完全一致。主审/scale 分工负责整个 R3 冷进程的 fit profile，本分工不重复运行该流。

需要准确表述为“严格入口对文件内部的冻结实例验证和零 fit 已成立；多数历史节点的预期实例 anchor 仍缺”。

### OR 权限及异质法则

`wormhole_forest.py:279–326` 将公共 preconditions 与各分支条件一起检查，选中实际可用分支后以其 permission_tier 限制授权，并采用其 evidence_status。`333–348` 返回对应 selected action。`595–615` 在 OR scope 合并中明确拒绝两个不同非空法则 fingerprint。

这正确关闭旧 A 分支 bounded_trial、B 分支 retrieve_candidate 被合并后 B 升权的控制，也关闭 Murray/capped 异质规律沿用 keeper A action 的旧控制。它仍是已有合同及有限 witness 范围内的合并，不应表述为开放世界语义等价证明；现阶段没有必要为这一已承认边界要求全量重写。

### program_steps 次序

`m4_multiscale_renormalization.py:1233–1306` 不再只把步骤收集到字典：检查 opcode 唯一性、允许字段、stage rank 严格递增，要求 compile/fit 与 confirm 存在，验证 operator tag、arity、fresh_points 的类型与正数条件。倒序旧控制会在消费者拟合前被拒。

### 当前 M4.3 确认点与 CodeAgent 已付前缀

新 helper `m4_multiscale_renormalization.py:1163–1230` 先排除当前训练前缀及批次内重复点，不足时合成确定性新点；不足额返回空集。新 M4.3 recall/ordinary 失败确认会进入下一步前缀（例如 `m4_3...:1453–1454`），OpenCodeAgent 的 role permutation 间也保留失败确认（`1595–1606`）。初次新诊断点在选择 episode confirm 前被排除（`1560–1574`）。这确实关闭上一轮同通道 parent/child 点耗尽和当前主流程 prefix-confirm 重叠。

`run_compositional_code_agent_episode` 的 `_PrepaidAwareAdapter`（`m4_compositional_discovery.py:990–1038`）把现有 lab observations 作为第一批输入直接交给 CodeAgent，匹配到的已付记录不再访问 oracle。

独立最小控制使用原生 linear operator 和已付两行 `(x=1,y=2.5),(x=3,y=7.5)`：它们原样进入 episode 训练前缀；整个 oracle 调用序列恰为 `x=1,3,5`，第三行是新 confirm；measurement cost 从 8 增到 12。没有把旧记录重新购测，确认成功。见 `probe_instance_freshness.json` 的 `prepaid_linear`。

确认点的 canonical box/role map 绑定问题由 scale 分工单独审计，不与本节重复计数。

## 3. P2：legacy loader 提前返回绕过新增 flags

`m4_compositional_discovery.py:1386–1418` 在没有 bundle 且未要求外部 manifest 时使用老兼容分支：用 legacy rows 做 isolated refit，然后直接 return；尚未进入后面的 frozen-only 或 expected-instance 检查。

对原封不动归档的 M4.1 W1 workspace，清空 loader cache 后调用：

```python
load_verified_authored_method(
    old_artifact,
    expected_ast_digest=original_ast,
    expected_instance_digest="definitely-not-the-instance-digest",
    verify_frozen_only=True,
)
```

实际成功返回，并调用一次 isolated evaluator，训练/评价各 6 行；`_verification_mode` 缺失。没有修改旧文件。该问题影响调用者对新参数语义的信任，且冷热缓存分支可能表现不一致。

当前 R3 固定 `require_external_manifest=True`，因此不经过这个分支；不能把这个控制写成“当前 R3 冷加载仍 fit”。建议 legacy 分支遇到 `verify_frozen_only` 或 expected instance 时明确拒绝，或完成一次显式旧档案迁移后再走严格路径。

## 4. P2：保留的旧 M4 ScaleBlind 仍会遗失已付 fine 确认点

控制直接运行当前源码中的 `build_m4_cluster_targets(cluster_idx=0)` 的原生 `C0_T07_ClosureBreakdown`，使用原生 `HippoPPR_ScaleBlind_Walk`；没有强制拟合失败、伪造点或改任务。

实际测量顺序：

1. 六个 coarse pilot；
2. micro `confirm_0`：`drive_u=1.8, quench_t=0.16`；
3. micro `confirm_1_child`：`drive_u=2.3, quench_t=0.95`；
4. 升级后重采六个 micro pilot；
5. `fallback_confirm` 再采第 2 步同一个 micro 点。

原因是 `m4_multiscale_renormalization.py:1524–1526,1623–1626` 仅在确认通道等于 pilot 通道时保留相关确认行，随后 `1631–1633` 直接将 `frozen_prefix` 重置为 micro pilot；`1656–1663` 的 helper 只能排除这个被缩减后的前缀，无法知道同通道确认点已经消费。

本例 heldout RMSE 仍为 0、strict_pass=true、false_confirmation=false，付费 observation 总数 15。准确诊断是旧入口的已付证据保留/测量新鲜性不足，并非实证预测错误。它也不推翻新 M4.3 11 臂记录的训练前缀零重叠。

最小修复是独立保留按通道组织的已付行，升级时将已付 micro confirm 与新 micro pilot 合并，并以同通道完整已付记录排除新的 confirm。不同通道同坐标的重新购测可能是必要的，不能用一个跨通道坐标黑名单粗暴禁止。

## 复现与证据清单

- `probe_same_ast_instance_swap.py` / `.json`：主要 P1，含完整正负对照及节点 inventory。
- `same_ast_distinct_gain_calibration/`：由现有 host 函数生成的第二个合法参数实例，含真实训练、确认和参数证据。
- `probe_instance_freshness.py` / `.json`：两个 P2 与已付前缀正控制。
- `prepaid_linear/`：原生 CodeAgent episode 文件。
- `scale_blind_t07/`：旧 M4 单目标执行产物。

这些控制足以确定上述边界；不继续扩大任意 malformed metadata、任意 Python 或完整 benchmark 测试。

## 实际阅读覆盖清单

这里区分整文件阅读、完整函数阅读和依赖定位；本分工不声称独自逐字读完全部仓库或全部版本。其他范围由主审及另外两个分工汇总。

### 整文件阅读

- 当前 `Hegel Machine/AGENTS.md`。
- 用户附件 `17_V17_5_M4_3_Channel_Bound_Lifecycle_and_Audit_Closure_Report.md`，133 行全文。
- 当前 `tests/test_v17_5_m4_3_channel_bound_lifecycle_and_audit_closure.py`，143 行全文。

### 完整函数/类阅读与前后调用点核对

| 文件 | 完整审阅的功能单位 |
|---|---|
| `m4_compositional_discovery.py` | `OpenCompositionalCodeProvider`；`run_compositional_code_agent_episode`；`canonicalize_fitted_params_snapshot`；`compute_confirmed_instance_digest`；`predict_with_frozen_instance`；`load_verified_authored_method` |
| `wormhole_forest.py` | `authorize_abstraction_use`；`fidelity_compress_nodes`；节点注册、实例字段和相关修订/证书接线段 |
| `m4_multiscale_renormalization.py` | `coords_approx_equal`；`select_or_synthesize_disjoint_confirm_points`；`_validate_and_interpret_m4_program_steps`；`evaluate_m4_target_under_arm` |
| `m4_3_channel_bound_lifecycle_and_audit_closure.py` | `evaluate_m4_3_target_under_arm` 全部 recall、ordinary、fallback、scope/heldout 和回执分支；闭环控制 C1–C3 及其断言 |
| `m4_2_unified_closure_and_pi_lifecycle.py` | `_run_metered_atomic_calibration`；R3 loader/预测/回执完整路径；Stage 0/1 action 形成与 Stage 2 Pi 实例字段写入；旧 evaluator 的全部 fresh/CodeAgent/pool 调用分支 |
| `scripts/run_v17_5_m4_3_unified_study.py` | `load_relocated_forest_from_bundle`，并核对 serializer/冷恢复调用位置 |
| `open_python_revision.py` | `_LabAcquireBatch` 与 `OpenWorkspaceCodeLabAdapter`，核对已付适配器接口 |

上述变更对比固定基线 `fb4b23aaf...`；与分工无关的同文件其余机制/生成器没有冒充全部完整审阅。

### 按关键路线定位的历史/依赖资料

- reuse hub 中已付轨迹、archive/研究返回的相关段落；没有把这个长 hub 全文列为本分工完整精读。
- historical CLI 已查询；`historical_capabilities.json` 定位上述三条复用路线的实现入口、状态、证据和已知边界，没有声称读完该 181 版本目录全部长条目。
- `post_stage2_forest_snapshot.json` 对全部节点 action/实例摘要字段做完整 inventory；不是对所有 snapshot 中每一个历史 payload 作全文语义复核。
- 旧 M4.1 W1、当前 Stage 0 Monod、控制生成的另一校准 artifact，以及 `part1_m4_2_audit_closure_verification.json` 都完整解析；语义复核集中于源/参数/AST/实例摘要/确认 bundle 与分工内断言，不冒称逐字人工阅读每个数值 payload。
- 参数/源码、角色、confirmation 的检查以前述固定 commit 函数实现和本报告的真实控制为依据；R3 全流只读/fit profile、证书/role box 以及主研究统计的独立复核由相应分工负责。
