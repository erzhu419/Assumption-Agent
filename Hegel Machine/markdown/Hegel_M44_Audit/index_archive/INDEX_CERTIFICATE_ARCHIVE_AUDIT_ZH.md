# V17.6 / M4.4 动态索引、PPR 成本归因、证书原子性与归档专项审计

固定提交：`b965f022b214f47a1341559f20041da6a224d66d`。对照提交：`3461535364fcdb4e9ae9c87ccba502b7e6b56fe8`。

## 一、结论及范围

本轮确实修好了旧版“原索引对象仍返回旧节点版本”和“迁移后 semantic/certificate digest 与内容不符”。`revalidate_vertical_edges_with_certificate` 也真正增加了节点/OR 分支实例校验，并把 action、ScaleMap、边重绑纳入一次成功提交。不能继续把这些已经修复的具体反例列作未修。

剩余一个明确的 P2 接口缺陷：**新建粗粒化边的 `add_coarse_graining_edge` 没有做报告宣称的证书—实例一致性校验**。用提交中的真实 Stage0 证书与真实 Stage2 节点调用新建入口，仍能得到活跃的、证书和节点分别属于不同实例的边；实际 M4.4 通道资格消费者会使用该边。新的快照 digest verifier 也报告零不一致，因为它检查对象各自摘要，尚未检查跨对象身份相等。

PPR 在 Part3 比 Compiled 净省 `0.4550 C` 是真实的已提交账本结果，但主要来源需要说清：**少评 9 个候选节省 `54 K`，抵消额外 `44.91 K` 的查询与通道预检费用**。Pooling 开关在 20 步端到端账本上完全持平。流后纯三线索 Top1 控制与端到端净收益是不同证据层次。

本专项没有重新建林、没有重复官方全量 study、没有模型 API 调用、没有修改源码和提交内证据。所有控制直接恢复提交的三份快照；只有为确认粗边消费后果而执行的两个原始 T21 小目标，会在专项 `work/` 生成独立工作文件。主审负责完整 42 测试和官方重放；另一专项负责 Part3 持久化缺失、在线发现与手工元数据的因果归属。

## 二、已实证的 P2：重验证入口已修，新建入口仍能建立跨实例粗边

### 2.1 源码路径

- `wormhole_forest.py:904–941`：`add_coarse_graining_edge` 只检查同树和无环。随后 `925–926` 使用 `node_action.confirmed_instance_digest or certificate.confirmed_instance_digest` 选出边的实例摘要；没有比较两者，也没有检查 OR 分支。`939–940` 直接 append 并增加 mutation epoch。
- 同文件 `1284–1307`：重验证入口才执行节点—证书及分支—证书摘要比较。这段修复是真实的。
- 同文件 `943–960`：`fine_scale_children_of` 只检查 stale 标志和当前 endpoint versions，不核对证书实例。
- `m4_3_channel_bound_lifecycle_and_audit_closure.py:1147–1198`，尤其 `1165`：粗通道资格要求存在 live fine child，之后核对节点 scope / box / authorization；未核对该边的 certificate identity。
- 同文件 `1287`：候选资格再次使用相同 live-child 条件。
- `m4_4_instance_role_closure_and_online_writeback.py:148–191`：快照检查对节点五字段和 certificate JSON 分别重算摘要，没有三方一致性校验。

### 2.2 控制使用真实的已确认实例

没有伪造 `certified=True`，没有改数值模型或参数，没有扩大容差。证书来自提交的 `true_stage0_forest_snapshot.json`，节点来自 `post_stage2_forest_snapshot.json`：

| 对象 | confirmed instance digest |
|---|---|
| Stage0 Pi 证书 | `416e0bd1c16725ba97bfe64d3f4e265a83b4f8a39c601e0fc175ecebbeaff070` |
| Stage2 Pi 节点 | `058680e5869a3144da12837f9214eb4c3990efbcc5fe82f1fa037bee6d8d9231` |

对独立 Stage2 森林副本移除已有 Pi 粗边，然后通过公开 `add_coarse_graining_edge` 把这份 Stage0 证书传回：调用成功，边数 `1→2`，`stale_endpoint_revision=False`。节点和边实例是 `0586…`，边内证书实例却为 `416e…`。`fine_scale_children_of` 返回 `ode_micro_2mode_node`，说明它是可消费的 live 边。

实际执行原始 seed `20301120` 的 `M43_T21_PiS_Adiabatic`：

| 控制 | 选中通道 | 测量费 M | heldout RMSE | strict pass | false confirmation |
|---|---|---:|---:|---:|---:|
| 只移除 Pi 粗边 | `micro_fine` | 70 | 0 | true | false |
| 加入上述跨实例 Pi 粗边 | `coarse_macro` | 21 | 0.008633 | true | false |

这说明消费者没有另外补上证书身份防线，错误绑定确实能开通廉价通道。但本例两个真实模型数值接近，仍在当前容差内；**本控制证明身份/证据归属不变量缺口，不构成新的数值假确认反例**。正常已提交 Stage2 两条边的实例均一致，也没有证据要求推翻官方 Part2/Part3 全部结果。

此外，对这个错误绑定森林调用 `verify_forest_snapshot_digests`，仍得到 `semantic_digest_mismatch_count=0`、`certificate_digest_mismatch_count=0`、`missing_expected_instance_nodes=[]`。局部内容摘要自洽不自动意味着跨对象关系合法。

完整证据：`certificate_controls.json` 的 `add_real_cross_instance_edge`、`no_coarse_edge_control` 和 `cross_instance_edge_control`。

### 2.3 建议修复边界

复用当前重验证入口已有检查，不需要新证书算法：把节点/分支/证书身份检查抽成共享的无副作用 validator，新建与重验证两入口都在任何 mutation 前调用。至少当任一方提供实例摘要时，要求两方都有且相等；对保留的旧无实例 fixture 明确兼容语义，避免为修这一缺口破坏已有历史接口。

快照 validator 再增加 node、edge、certificate 三方实例相等检查，作为归档关系一致性门槛；粗边消费端至少拒绝已知跨实例不一致的边。回归用本次提交内真实 S0/S2 组合即可，无须重跑全部形成流程。

## 三、重验证原子性：原问题确实闭合

本专项把整个 `forest.__dict__` 序列化后做 SHA256 前后比对，包含本轮 serializer 未保存的 epoch / questions / raw_episode_archive，避免只比较部分节点掩盖状态变化。

| 控制 | 结果 | 全森林状态 |
|---|---|---|
| 真实 S0 certificate 重验真实 S2 node | `accepted=False`, `certificate_instance_digest_mismatch` | 完全不变 |
| S2 outer action 与 certificate 正常，只将一个 OR 分支 action 换为真实 S0 action | `accepted=False`, `branch_certificate_instance_digest_mismatch` | 完全不变 |
| 提议将 outer action 和全部 OR action 同步切换到已有真实 S0 instance，配合 S0 cert 和被它包含的 S2 ScaleMap | `accepted=True`，version `4→5` | node、edge、cert 全部一致；semantic/certificate digest 校验通过 |

当前正式 builder `m4_2_unified_closure_and_pi_lifecycle.py:1924–1963` 在局部变量中组装新 action / branches / ScaleMap，再一次传入 `revalidate_vertical_edges_with_certificate`；旧版先直接写 action 再调用重验证的路径已经消除。这里只建议补齐新建入口，不能笼统说本版所有证书原子性都失败。

## 四、动态索引：三种索引的正式 mutation 路径都已接上

`forest_state_signature()` 位于 `wormhole_forest.py:701–745`，包含 epoch、节点版本/语义摘要/归档状态/证据状态/cues、wormhole 版本与 endpoint versions、vertical/coarse edge endpoints、stale 标记与 coarse certificate digest。`HippoRAGWormholeIndex`、Flat 和 Compiled 的 `retrieve` 都在查询前调用 `ensure_fresh`。

独立恢复同一 Stage2 森林，将三个索引始终保持为同一对象，依次执行正式 API：

| 事件 | PPR 刷新费 K | Flat 刷新费 K | Compiled 刷新费 K | 结果 |
|---|---:|---:|---:|---|
| W4 增加新前提并正式 revision/rebind | 17.55 | 2.60 | 17.64 | 三者都由 v1 返回 v2，并含新前提 |
| 无 mutation 再查 | 0 | 0 | 0 | 不重复收费 |
| 正式 `refine_wormhole_with_near_miss` | 17.55 | 2.60 | 17.64 | 三者都再刷新一次 |
| W4 连续三次负效用反馈触发归档 | 16.55 | 2.40 | 16.08 | 三者都不再返回该节点 |
| 已归档后无 mutation 再查 | 0 | 0 | 0 | 不重复收费 |

这个控制还说明刷新并非单纯依赖 epoch：wormhole refinement 不增加 epoch，但 edge version 被 signature 捕捉，刷新照样发生。正式 API 的 node、edge、utility 更新已经闭环。相应生产路径为 `hipporag_wormhole_index.py:316–323,373,490–498,585–591,601,673–679` 和 `m4_3_channel_bound_lifecycle_and_audit_closure.py:612–653`。结果在 `dynamic_index_controls.json`。

当前刷新策略是整索引重建：PPR 和 Compiled 都如此。它解决缓存陈旧，尚不证明某个算法拥有一般性的增量维护复杂度优势。这里不把未实现的任意字典就地篡改拦截或大图增量更新列作本轮必须扩展的功能。

## 五、Part3 的 0.4550 到底来自哪里

本专项读取全部 80 条已提交 receipts，并逐行重算四臂费用。没有只依赖工作总结表格。

### 5.1 总量与阶段分解

| 指标，20 步总量 | PPR PoolOn | PPR PoolOff | Compiled | PPR 相对 Compiled |
|---|---:|---:|---:|---:|
| 测量费 M | 603 | 603 | 603 | 完全相同 |
| 评估候选数 | 42 | 42 | 51 | 少 9 个 |
| `candidate_eval_compute` K | 230 | 230 | 284 | 省 54 |
| `retrieval_query` K | 119.71 | 119.71 | 79.17 | 多 40.54 |
| `channel_precheck_retrieval` K | 5.87 | 5.87 | 1.50 | 多 4.37 |
| artifact checks K | 32.75 | 32.75 | 32.75 | 相同 |
| 其他计算 K | 107.25 | 107.25 | 107.25 | 相同 |
| 总 K | 495.58 | 495.58 | 504.67 | 净省 9.09 |
| receipts 中总 C | 627.7785 | 627.7785 | 628.2335 | 净省 0.4550 |

未作逐行四位小数舍入时，`(54−40.54−4.37)×0.05=0.4545`；逐行 C 的舍入后再求和得到报告中的 `0.4550`。差异只有 `0.0005`，不构成造假或实质计费错误。

九个候选优势精确发生在：

- W7 双 cue 的 `S12–S15`：PPR 各评 1 个候选，Compiled 各评 3 个，共省 8 次。
- 历史 Murray 保持题 `S18`：PPR 评 1 个，Compiled 评 2 个，再省 1 次。
- 其余 15 步候选数相同，Compiled 单步复合费更低。

费用是项目声明的代理 `C=M+0.05K`，不是 wall time 或货币费用。本轮数据支持“在这套固定目标、检索评分、候选试验单价和查询定价下 PPR 小幅胜出”，不支持把动态 PPR 的一般计算优势直接作为结论。正式源码按每个普通候选 `6 K` 计费（M4.3 evaluator `1712–1713`），与本次 `9×6=54 K` 完全吻合。

### 5.2 初建成本漏在流外，但补计不会推翻正负号

Part3 在 `m4_4_instance_role_closure_and_online_writeback.py:1070–1079` 构造初始索引，没有将 `index_build_compute_cost` / `compile_compute_cost` 加入任何 receipt 或 upfront 项；mutation 后的三次刷新则被 `retrieve` 计费，并由 evaluator `1158,1273` 纳入 meter。

从提交 Stage2 快照复用同样的 hub 初始化，再创建索引，测得初建费：PPR `19.7 K`、Compiled `20.4 K`。若口径是“从相同森林开始的完整 20 步流”，应同样计入；如果口径是“已有就绪索引后开始的流内服务费用”，应明确声明该起点。

| 补计一次初建后的总 C | 数值 |
|---|---:|
| PPR PoolOn / PoolOff | 628.7635 |
| Compiled | 629.2535 |
| Frozen | 713.7918 |
| Compiled − PPR | 0.4900 |

因此该遗漏不会让 PPR 优势消失，反而微增 `0.0350 C`。Live 与 Frozen 都使用相同初始 PPR，二者 `85.0283 C` 的差额不变。不能把“漏计初建”写成已证明的排名翻转。

小的注释/收费不一致：writeback helper `810–811` 写明每 node `1.5 K`、每 edge `0.8 K`，但本轮两个写回都传 `weaker_parent_id=None`（`916,988`），实际每次只有两 node 加一 wormhole，应为 `3.8 K`，函数固定返回 `4.6 K`。两次共多收 `1.6 K=0.08 C`，是保守多收费，所有 live 臂相同，不影响 live 臂之间排序。这适合校准事件计费时修正，不应夸大为研究主要漏洞。

完整复算与逐步配对在 `part3_cost_analysis.json`。

## 六、Pooling 的三 cue 排序效果与端到端收益要分开

生产实现本轮将同构组件识别从只看 `schema_tag` 扩到 `schema_tag or ast_skeleton_digest`（`hipporag_wormhole_index.py:69–77`），使没有旧 schema tag 的新 authored AST 节点能参与 pooling。概率类分数为同构组件成员 PPR mass 的和（`435–439`），这是实际接入变化。

提交 summary 的流后诊断查询为三条 cue：`sig:cue_tubule_saturation_a`、`sig:cue_reuptake_saturation_b`、`sig:pos_arity_1`。结果为 PoolOn 的 Top1 是 `online_W6_reuptake_twin_1d`，PoolOff 与 Compiled 的 Top1 是 `stage1_W3_murray_capped`。该诊断由源码 `1203–1209` 单独查询，发生在20步流结束之后；它不在流的成本/成功率 receipts 中。

全部 20 对 PoolOn/Off receipts 的 `cost` 和 `evaluated_candidate_count` 逐条相等。因而当前可以说“pooling 改变了特定多线索排名”，不能说“pooling 已产生本次 0.4550 的端到端收益”。同样，三 cue 诊断中的 Top1 命中不是一个经过新购测确认的独立任务胜利。工作报告已承认经验曲线额外触发次线性 cue 导致端到端持平，这一诚实边界应保留。

初始 hub 的 cue 扩充（`828–854`）和 W6/W7 twin 的 cue 配置（`906–917,978–989`）是实验构造的一部分。是否称为实际跨域经验抽象、自然观察 cue 或严格证据支持的 twin，见在线学习专项；本专项不重复把同一元数据归属问题算作额外算法缺陷。

## 七、Part2 位置无关归档：本轮应判定修复成功

新 serializer（M4.4 module `121–131`）直接保存契约，loader（`134–145`）只设置外部 artifact_root。`PhilosophicalForest.resolve_artifact_path`（`680–699`）在执行时解析物理地址，不改契约内容。正式 runner `128–140` 也不再重写 contract/certificate 路径。

本专项按生产 `canonical_digest` 的精确五字段，检查磁盘 JSON 和 cold-loaded 内存对象，并额外覆盖历史版本与 OR 分支的 artifact action：

| 快照 | live 节点数 | live artifact 节点数 | 历史节点数 | artifact action 检查数，含历史/OR |
|---|---:|---:|---:|---:|
| true Stage0 | 8 | 7 | 0 | 7 |
| post Stage1 | 15 | 14 | 0 | 14 |
| post Stage2 | 15 | 14 | 5 | 24 |

三个快照的 live semantic mismatch、certificate hash mismatch、missing expected instance、node/edge/certificate 实例不等全部为 0；五条历史节点 semantic mismatch 也是 0，全部 artifact action 路径可解析且文件存在。旧版 `7/8,14/15,14/15` 不一致已经消除，不能继续沿用旧诊断。

这里验证的是提交中三份 Part2 执行快照。Part3 最终 live forest 和 W6/W7 产物没有随 runner 输出持久保存，是另一项独立问题，由在线学习专项报告。两者不能合并成“本轮归档仍全失败”。

另一个范围说明：当前 serializer 的七个字段不包含 questions / raw_episode_archive / mutation_epoch；现行形成代码没有向这些容器写入科学状态，所以本专项未观察到 Part2 已有问题/原始 episode 被它实际丢失。若把该函数提升为通用持续学习 checkpoint，需要再明确完整恢复合同，但不把未发生的潜在使用情形列为本轮实证失败。

证据：`archive_identity_checks.json`。

## 八、实际阅读与执行清单

先读了本 checkout 的 `AGENTS.md`，查 reuse hub，并按要求运行 historical catalogue 查询 `HippoRAG wormhole index certificate archive M4.4` 与 `wormhole`；两者均无命中。随后复用既有 M4.3 专项原报告和反例事实，沿本轮具体变更追踪当前实现；没有把无命中的 catalogue 当成“从未实现”。

本专项完整读取了上传的 `18_V17_6_M4_4_Instance_Role_Closure_and_Online_WriteBack_Report.md`、当前 `hipporag_wormhole_index.py`、新 runner、新 M4.4 测试文件以及 M4.3 `SCALE_DISPATCH_AUDIT_ZH.md`。其他源码按本专项调用路径读取：

- `wormhole_forest.py` 的全 diff，及 `664–1610` 的注册、边新建、正式修订、重验证、live edge 消费、utility、压缩接口。
- `m4_3_channel_bound_lifecycle_and_audit_closure.py` 的 Compiled `521–654`、builder/初编费用、evaluator `1096–1519` 的通道/召回/确认调用及 `1712–1713` 计费、study `1779–1925` 的初建接法；其他目标族/噪声全量由主审负责。
- `m4_4_instance_role_closure_and_online_writeback.py` 的 serializer/digest `121–191`、证书反例与 archive 验收 `446–490`、dynamic probe `599–620`、完整 online helper/runner `703–1256`。
- `m4_2_unified_closure_and_pi_lifecycle.py` 的 dispatcher `479–620` 和 Stage2 新 action/scale/cert 提交 `1885–1963`。
- 三份提交快照的完整 JSON，经脚本覆盖 live/history/OR action，以及完整 80 条 Part3 receipts 和 Part3 summary。

本专项没有宣称逐字读完全仓库、历史 181 版本或参考文献全文；其余已分工内容由主审综合，不借此扩大实际覆盖范围。

复现命令（默认使用本次固定提交的现有 checkout，输出到脚本所在专项目录）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 python3 \
  /workspace/scratch/3f14ad2da0ed/audit_m44_b965f022b/index_archive/run_index_archive_controls.py
```

其他环境可通过 `HEGEL_AUDIT_REPO` 指定已经检出上述固定 commit 的 git 仓库根目录，通过 `HEGEL_AUDIT_OUT` 指定独立输出目录；仓库变量也接受 `Hegel Machine` 子目录。脚本不切换分支或修改 checkout，调用者应先确认检出版本。

```bash
HEGEL_AUDIT_REPO=/absolute/path/to/Assumption-Agent \
HEGEL_AUDIT_OUT=/absolute/path/to/independent-audit-output \
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 python3 \
  /absolute/path/to/run_index_archive_controls.py
```

输出：`archive_identity_checks.json`、`dynamic_index_controls.json`、`part3_cost_analysis.json`、`certificate_controls.json`。脚本不依赖模型 API、旧临时目录或森林重建。两个有意义的 target controls 均保留实际编译文件与 `scope_contract.json`；再次执行只覆盖所选输出目录下的 authored 工作区。环境变量配置改动仅做语法检查，没有重新运行科学控制。
