# Assumption Agent V17.5 M4.3 独立审计与修复建议

审计提交：`3461535364fcdb4e9ae9c87ccba502b7e6b56fe8`。对照提交：`fb4b23aaf802c20b4e2d0018436abab73b0cca06`。日期：2026 年 11 月 20 日。

**结论：M4.3 已经实质修复上一轮多项核心工程和对照问题，新的受控收益值得保留；但“全部核心不变量已经闭合”仍不能验收。当前最应修复的是两条明确的调用关系：历史节点必须指定它期待的已确认实例；证书使用的坐标映射必须与实际执行模型一致。** 两条都有正常正对照和有限反例，不依赖任意恶意 Python、不连续隐藏尖峰或扩大容差。

维护回本也有真实的新证据。按提交原始账本重算，在当前访问流和项目定义的费用代理下，维护在第 46 题回本，96 题末净省 278.2019；后文另列已观察到的计算事件漏记，尚不能把这份账本称为完整事件计费验收。但其中约 98.9% 的 Stage3 毛收益来自八个液压绝热目标获得合法粗通道；不能把这个结果分别解释为 Minton、范围收缩、所有维护操作的普遍收益。同信息编译关系表则以同质量、同测量费取得略低的原始全周期账本成本。这有助于把森林中的关系知识价值与具体 PPR 算法的价值分开。[S1](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/17_V17_5_M4_3_Channel_Bound_Lifecycle_and_Audit_Closure_Report.md)[S2](https://github.com/erzhu419/Assumption-Agent/tree/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/docs/evidence/v17_5_m4_3_channel_bound_lifecycle_20261120)

## 审计范围与复现结果

这次以固定 SHA 的真实 checkout 为准，核对附件、提交内 Report 17、全部 9 个变更代码/测试文件及其调用依赖，复用提交内上一轮审计和历史复用目录；程序化核对全部 1,144 条主回执及 126 个清单文件，并对关键接口进行独立有限控制。具体逐文件阅读范围保存在三个专项报告中。这里不把“历史资料已索引、已有审计可复用”表述为本轮逐字重读了所有历史版本和全部参考文献全文。

相对上一提交，本地 Git 精确差异为 771 个路径：9 个代码/测试文件、127 个新证据文件、632 个导入的旧审计文件，以及旧审计 ZIP、索引、Report 17 三项。GitHub compare 返回的首批文件清单有数量上限，因此未把该截断清单当作全部差异。审计没有修改源代码、提交证据或远程仓库，没有模型 API 调用。

| 验证项 | 本次结果 |
|---|---|
| 当前相关回归 | 37 项历史回归 + 3 项 M4.3 新测试，合计 40/40 通过 |
| 完整官方研究 | 原 runner 在独立输出目录成功结束，耗时 710.13 秒 |
| 原始及重跑文件清单 | 两侧各 126 个 SHA256 项全部匹配，各有 127 个文件 |
| C1–C6 验证摘要 | 与提交文件逐字节相同 |
| 主回执 | 1,144 条、1,144 个唯一键；费用/状态/误差/成功率/重叠字段全部相同 |
| 11 臂汇总、回本、双轨噪声费用 | 与提交结构化数据完全相同 |
| 全部文件的字节级比较 | 93/127 相同，34 个不同；没有将整个证据包误称为字节级重现 |

不同内容已逐字段保存在 `replay_verification.json`：主回执仅 41 个 `selected_model_sha256` 不同，未出现费用、状态或误差的数值差异。外部参数文件中 22 个浮点数不同，最大绝对差约 4.0e-13，引起对应参数/实例 hash 变化；其他差异包括墙钟、语义/证书摘要及其传播。原账本算术内部一致。这里认可的是科学指标和行为的重现，不要求独立数值拟合产生跨环境字节相同的实例。

另有三组特别重要的独立结果：

- 直接冷载提交内森林，R3 十条路由正确，Python profile 观察到 0 次 `fit`，完整森林序列化前后相同；删除粗边后，短时升级被拒绝。该正结果保存在 `scale/scale_controls.json`。
- 同 AST、同模型源码、不同真实确认参数的实例错绑：R3 原预测 2.66666667 变成 5.33333333，仍按原节点 v2 授权；补上节点预期实例摘要后拒绝。见 `instance/probe_same_ast_instance_swap.json`。
- 同一 Pi 方程的合法短时输入变体：证书盒映射交换 u/T，实际模型仍按原 u/T 执行，最终错误确认，heldout RMSE 0.100316 大于容差 0.055。见 `scale/scale_controls.json`。

这两个负结果均未在原始主表的未扰动运行中触发：前者替换同结构的参数实例，后者改变合法输入范围。因此，正确判断是“正常研究结果可以复现，接口保证仍有缺口”，不能把反例机械外推为 1,144 行全部失败。

## 上一轮哪些问题确实修好了

| 上一轮问题 | 当前实现与证据 | 验收判断 |
|---|---|---|
| 任意不相干粗边也使 Pi 获得便宜收费 | 新主入口查询匹配节点、活粗边、作用域和 modality；真实粗细 oracle 分别返回 Schur/RK4 数值，按 3/10 收费 | 旧的按标签改价问题已修；角色映射仍有另一处缺口 |
| Frozen 只保留裁剪后的 Stage0 | 保存并恢复真实 8 节点、2 粗边、1 虫洞的 Stage0 | 已修 |
| OpenRediscovery 只多记费用、未真正运行发现 | 进入原生 CodeAgent，继承已付 prefix，执行诊断、组合候选和独立确认 | 已修 |
| 同通道确认点耗尽后重复进入训练/确认 | 新 helper 足额选择或合成不重叠点，失败确认进入后续 prefix；主表最大 overlap=0 | 当前 M4.3 主流程已修；实际点仍须加证书盒检查 |
| 参数被改，loader 通过重新 fit 验证另一组参数 | 参数摘要、外部参数文件、manifest、确认包互验，并直接用冻结参数预测原确认行 | 文件内部的冻结实例一致性已修；节点预期实例绑定未全接入 |
| R3 验收函数临时修改 scope，删除边也不影响升级 | Stage2 正式修订 fine 节点，R3 只读；删边后 Q05 被拒 | 已修，且有独立冷载与调用观察 |
| 窄证书可扩盒，正常节点与边范围不同步 | 要求请求盒被证书盒包含，正常 Stage2 node/edge/certificate 同步 | 正常成功链已修；跨实例证书仍未拒绝 |
| tail-window 只记录、不影响认证 | window drift 现在参与 `certified` 判定 | 已修；仍是有限轨迹经验认证 |
| OR 合并使低权限分支升级、异质法则沿用 keeper | 使用匹配分支权限，异质法则 fingerprint 不同则拒绝 | 旧控制已关闭 |
| `program_steps` 只收集字段，不检查次序 | 检查步骤顺序、唯一性、必要步骤和 fresh 点数 | 旧倒序控制已关闭 |
| DEV 漏族、没有正式最小 pilot 残差对照、两因子一起关 | 六族 DEV、BestPilotRMSE、Compiled、Alpha×Pooling 配置进入主运行 | 设计修复成立，解释边界见后文 |

这些修复改变了执行路径，不能继续用 M4.2 的原结论概括 M4.3。尤其是真双通道、真 Frozen、真 CodeAgent 和同森林 Compiled，对后继研究都有直接价值。[S3](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py)[S4](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L578-L583)[S5](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_compositional_discovery.py#L1224-L1555)[S6](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L1140-L1288)

## 必须修复一 历史节点没有完整指定预期实例

### 文件内部真实不等于节点指向原实例

新 `confirmed_instance_digest` 的构造是有用的：它包括源码、AST、有限数值参数及角色名；严格 loader 会比较外部参数文件和 manifest，并使用持久化参数检查原确认误差。这关闭了上一轮单改 `p0` 或同时改参数文件的控制。[S5](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_compositional_discovery.py#L1224-L1555)

但 R3 调用仍使用：

```python
expected_instance_digest=str(act.get("confirmed_instance_digest", "")) or None
```

提交的 postStage2 森林共 15 个节点，其中 14 个有 artifact，只有重认证后的 Pi thermal 节点保存这个预期摘要。其余 13 个缺失，排除两项已归档节点后，仍有 **11 个活跃 artifact 节点缺失**。加载器于是只能确定“这个文件夹是某个合法已确认实例”，不能确定“这就是该历史节点所承诺的实例”。这发生在 R3 本来就要精确复访旧实例的入口，不应与 Stage3 借用结构后重新拟合新目标混为一谈。[S4](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L578-L583)[S5](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_compositional_discovery.py#L1224-L1555)

### 两个真实校准实例的对照

控制复用原生 `_run_metered_atomic_calibration`，为同一个 Monod 结构生成另一份真实观测、真实拟合、独立确认的 artifact。原参数是 `p0=6,p1=2`；新参数约为 `p0=12,p1=2`。两者源码 SHA 和 AST 相同，实例摘要不同。新 artifact 单独经过严格冻结 loader 正常接受，确认 RMSE 为 0。

随后仅在审计副本中替换三个 artifact 文件，保留森林原节点 ID、版本、AST 和路径含义：

| 场景 | R3 状态 | 原 Q01 的预测 | 对原实例真值的误差 |
|---|---|---:|---:|
| 原始实例 | `dispatched_ok`，原节点 v2 | 2.66666667 | 0 |
| 另一合法同 AST 实例 | 仍 `dispatched_ok`，仍原节点 v2 | 5.33333333 | 2.66666667 |
| 同一替换，节点补原实例摘要 | `scope_unauthorized`，原节点因 `artifact_verification_failed` 被拒 | 无预测 | — |

错绑后仍标为 `host_confirmed_discovered_instance`，仍获得 `local_solver_guidance`，也仍然是零 fit。零 fit 只说明没有重新训练，无法代替实例归属检查。现有 C1 使用 W1/W2 的跨节点交换，两者 AST 已不同，因而不能覆盖同结构不同参数的这一情况。

这不要求把任何合法文件的替换都当成恶意攻击。在多代自我修订中，旧路径被新校准覆盖、重用同骨架的另一实例、恢复了错误版本，都是同一个身份不变量要防止的普通错误。完整控制和生成的新校准证据见 `instance/`。

### 最小修复

在所有要支持零 fit 历史预测的节点首次形成及正式修订时，记录 artifact 的预期实例摘要；让匹配的 OR 分支、相关依赖边和 R3 回执引用同一实例。实例预测入口遇到缺失摘要时拒绝，或经过明确的旧档案迁移。结构复用入口仍可只取 operator spec 并重新拟合，但应明确返回结构身份，不能借用原实例的确认权限。

已有 loader 的摘要比较器是正确的，补字段即可让本反例转为拒绝。回执还应把 `artifact_sha256` 改成准确的 `model_source_sha256`，另外输出实例摘要和参数摘要；现在两份不同参数实例在这个所谓 artifact SHA 字段上完全相同，因为记录的只是源码 SHA。[S4](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L578-L583)

## 必须修复二 证书坐标与执行坐标没有绑定

### 有界性检查选了一个能过关的排列

`_find_valid_canonical_role_map_for_box` 即使看见目标已经使用规范名 `u_drive,T_horizon`，也会在 identity map 失败后继续试排列。只要有一个排列使点落在证书盒中，就返回该 map。接下来拟合器独立尝试 role_order，并未要求实际最优模型使用这个证书 map；最终 query guard 又沿用前面那份 map。[S7](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L2410-L2458)[S8](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py)

这把两种不同问题混在一起：角色对应应由模型含义或经过验证的观测映射决定；落盒检查应在这一已确定映射上进行。不能反过来以“哪个排列可以过盒”确定物理含义。

### 同一方程的数值反例

使用正常 seed20301120 的 T21、原始 fine/coarse oracle、原始森林和容差 0.055。正常 T21 正对照走 coarse，拟合角色和 scope map 都为 identity，heldout RMSE 0.008633。

变体将 pilot 的物理坐标改为 `u=2.45…2.95,T=0.40…1.20`，确认点为 `u=2.70,T=0.47`，heldout 也在这一短时区域；任务声明 bounds 同步改为 `u∈[2.4,3.0],T∈[0.4,1.3]`，使查询属于变体的合法任务域。真实证书要求 `u∈[0.4,3.0],T∈[2.4,5.2]`，时间全部不满足。

实际观察：

- 盒 helper 返回 `u_drive→T_horizon,T_horizon→u_drive`。
- 拟合器返回的真实角色顺序仍为 `['u_drive','T_horizon']`，训练 RMSE 约 1.40e-8。
- 系统以 coarse 通道购测和确认，写出的最终 scope contract 保留交换映射。
- 返回 `completed,confirmed_ok=true,out_of_box_query_blocked=false`。
- 对 fine truth 的 heldout RMSE 为 **0.100316 > 0.055**，所以 `false_confirmation=true`。

这个失败仍来自项目自己的连续两模系统。它说明新守卫没有做到“同一个坐标解释贯穿授权和执行”。本例及正常对照的实际购测、拟合角色和模型代码均在 `scale/scale_controls.json` 及对应工作区。

### fresh 选择后也要守同一个盒

另一个更小的接口控制保留默认 fresh1：将第一预声明确认点设为已付 pilot，第二点为 `u=2.8,T=0.4`。前置守卫只看 `confirm_points[:1]`，helper 正确跳过第一点后，实际购测却选了越盒的第二点。调用没有传 helper 已支持的 `verified_box`，购测前也未再次验证。该例原盒内 heldout 仍通过，因此只证明作用域检查漏接，不把它冒称为另一次预测反例。[S3](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py)

### 最小修复

规范名已明确时强制 identity；匿名角色映射应作为具体候选身份的一部分，经验证后与 `role_order`、编译模型、实例摘要和证书一起冻结。通道选择、每个实际 pilot/confirm 批次、补选或合成点，以及最终 query 都使用同一映射。实际 fresh batch 必须位于任务域与证书域的交集；不足额时升级细通道或诚实返回 unknown。

现有 fitter 已返回 `role_order`，helper 已有 `verified_box` 参数，现有 ScopeDispatcher 已有拒绝和升级状态。这里的工作是把已有检查贯穿调用，不是引入一套新的抽象理论。

## 需要一并修正的局部问题

### 证书身份和归档身份

`revalidate_vertical_edges_with_certificate` 现在检查盒包含关系，却不比较传入证书和当前模型实例。直接使用真实 Stage0 Pi certificate 重新验证真实 Stage2 节点，API 接受：边的实例摘要为当前 `0586…`，边内证书实例摘要仍为旧 `416e…`。两份 artifact 均合法、同源码同 AST、参数不同；共同盒内预测只差约 9e-8，因此这是身份归属缺口，不能说该控制已经导致数值误判。当前正常 Stage2 三方原本一致，旧版成功链不同步确已修复。[S6](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L1140-L1288)

建议在一次待提交修订中同时验证实例、角色、ScaleMap、证书及边，再提交版本。当前 builder 先直接更换节点 action，再进行 certificate revalidation，正常成功时一致，但失败路径不应留下半更新状态。

归档方面，内容已经可以迁移并冷载执行，然而按生产代码的五字段 `canonical_digest` 重算，Stage0 7/8、postStage1 14/15、postStage2 14/15 个节点与保存摘要不一致；Stage2 Pi 证书保存摘要也不等于归档内容重算值。路径参与了当前身份 hash，而序列化会把路径改为相对路径、加载会再次改成绝对路径。没有打包前对象，不能宣称路径重写是所有不一致的唯一原因；可以确定的是当前保存摘要无法由实际归档对象复核。见 `scale/archive_digest_checks.json`。[S6](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L1140-L1288)[S9](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/scripts/run_v17_5_m4_3_unified_study.py)

应使用位置无关的 artifact 内容身份，把存储地址单独解析。不要简单对搬迁后的内容重新 hash 然后声称历史身份未变。冷载能预测与摘要可自验证是两个不同的验收项。

### 新 fine 近似不是方程的精确解

真实 RK4 方程是：

```math
\dot x_s=-x_s/\tau_s+(a_s/\tau_s)u+g_{sf}x_f,
\qquad
\dot x_f=-(g_{fs}/\tau_f)x_s-x_f/\tau_f+(a_f/\tau_f)u.
```

因此系统矩阵的 `A12` 应是 `g_sf`，默认 0.35；M4.3 新两指数 surrogate 使用 `g_sf/tau_s`，约 0.24137931。随后用三个真实 RK4 点解三个系数，只能补偿标定点，不能使错误特征值成为同一 ODE 的精确解。[S8](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py)[S10](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py#L1297-L1394)

正常 T23 的 heldout RMSE 是 0.005842，最大单点误差约 0.0097155，仍在 0.055 容差内。该问题不否定这批 quench 的 success，也不影响 R3 直接使用真实 RK4 的 fine 路径；它需要修正方程实现及 `exact` 的措辞，防止后续精度承诺建立在错误公式上。

### fine 候选计数与计算收费漏掉已做的工作

独立对正常 T23/T24 做调用观察，fine 分支实际先调用 `_autonomous_fit_candidate_specs`，完成一个通用候选的训练评价；随后丢弃其结果，再生成并训练两指数 surrogate，最后把 `ev_c=1` 写入计数。还在分支内额外运行了 6 个 pilot 位置和 3 个标定位置的 host RK4 计算。证据见 `cost_events.json`。

记录的 `evaluated_candidate_count=1` 不能表示实际的两个候选训练评价。按当前每候选 K=6 的约定，仅漏掉那个通用候选就至少少计 C=0.3/受影响任务；额外 RK4 是否如何计费也应预先明示。这是计算费用遗漏，不是外部测量泄漏：已有可信模型在本地计算，不应重新算作向未知目标购测。[S10](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py#L1297-L1394)[S11](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/cross_domain_clusters.py#L29-L91)

当前 Hegel 的八个正常 quench 均走该分支，单这一项最低修正为 +2.4 全期 C；NoMaintenance 和 Compiled 的同类分支也受到相同影响。因此不能据此宣称 46 回本整体翻转。合理处理是去掉被丢弃的试拟合，或如实计费；为 host solver 单独记计算事件，并重跑已有研究得到更新表格。其他候选加载也存在冷验证计算，账本应明确按一次 lifecycle 摊销还是逐次计费。

项目当前 C=M+0.05K 是预定义的工作量代理，并非实测墙钟或货币。账本内部加总一致，与实际动作是否全部进入账本，应分开验收。

### 两项旧入口兼容问题

第一，旧无 bundle artifact 的兼容 loader 会在检查新增 flags 前返回：对原封不动的旧 M4.1 W1，同时传错误 `expected_instance_digest` 和 `verify_frozen_only=True`，仍成功并执行一次 isolated refit。当前 R3 使用 `require_external_manifest=True`，不走该分支，不能把它写成当前 R3 的零 fit 失败。兼容模式应遇到这些强约束时拒绝，或先显式迁移。

第二，保留的旧 M4 `ScaleBlind` 原生 T07 在 coarse→fine 升级时，把已付两条 fine 确认从训练 prefix 中遗失，再把其中一条作为 fallback fresh 确认重新购买。该例 heldout 仍为 0、strict pass。修复应按通道保存完整已付记录；跨通道同坐标重新购测可能必要，不能用一个全局坐标黑名单禁止。两项证据都在 `instance/probe_instance_freshness.json`，均限定为旧入口/兼容接口问题。

## 十一臂主结果应该怎样解读

以下沿用提交的 C=M+0.05K 费用代理，每臂 96 道域内题；另 8 道 OOS 各臂均 unknown、无假确认。数值来自逐任务回执和官方汇总，重跑核对见前文。[S2](https://github.com/erzhu419/Assumption-Agent/tree/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/docs/evidence/v17_5_m4_3_channel_bound_lifecycle_20261120)

| 臂 | 严格通过 | 假确认 | 平均 M | 平均 C | 96 题全周期 C |
|---|---:|---:|---:|---:|---:|
| Hegel Unified | 96/96 | 0 | 31.3333 | 32.3941 | 3762.1078 |
| Hegel Walk NoPool | 96/96 | 0 | 31.3333 | 32.3941 | 3762.1078 |
| Compiled SameForest | 96/96 | 0 | 31.3333 | 32.3235 | 3757.1530 |
| Hegel NoMaintenance | 96/96 | 0 | 36.2500 | 37.5675 | 4040.3097 |
| Alpha0 PoolOn | 95/96 | 0 | 37.6250 | 39.5131 | 4445.5373 |
| Alpha0 PoolOff | 95/96 | 0 | 37.6250 | 39.5131 | 4445.5373 |
| Flat BM25 | 95/96 | 0 | 37.6250 | 39.5315 | 4447.2990 |
| Ordinary StratifiedDEV | 96/96 | 0 | 37.8333 | 40.2083 | 4761.3750 |
| Ordinary BestPilotRMSE | 96/96 | 0 | 36.1875 | 41.5625 | 4423.8250 |
| Stage0 OpenRediscovery | 93/96 | 1 | 42.1250 | 44.4729 | 4513.4520 |
| Stage0 NoDiscovery | 16/96 | 0 | 34.2917 | 35.8182 | 3682.6020 |

Hegel 的成功是正证据，但其中只有 87 次经召回确认，另 9 次使用全历史 fallback，不能写成 96 次都只靠图召回完成。NoDiscovery 总价较低而只解出 16 题，不能用单独成本列推断它更优。OpenRediscovery 有两次 unknown 和一次假确认，故与它的费用比较并非严格同成功率对照；Hegel 在本流上同时更便宜且更准确仍是表内真实事实。

### 维护收益的真实来源

| 家族 | 题数 | Stage3 维护毛节省 C |
|---|---:|---:|
| HaldaneShunt | 16 | +0.0416 |
| OscShunt4P | 16 | +0.0040 |
| MurrayCapped | 16 | +0.0384 |
| Onset MonodShunt | 16 | +5.3920 |
| Onset MurrayShunt | 16 | −0.0081 |
| Pi mixed | 16 | +491.1840 |
| 合计 | 96 | +496.6519 |

维护前置 C=218.45，故净收益为 496.6519−218.45=278.2019。非 Pi 的 +5.4679 已经改变上一轮“所有非 Pi 完全不变”的结果，但不足以证明该部分维护单独回本。

再分解 Pi：八个 hydraulic adiabatic，Hegel coarse 平均 C=21.6995，未维护 micro fallback 为 83.0985，每题节省 61.399，合计 +491.192；八个 thermal quench 两边都走 fine，合计反而是 −0.008。因此 Pi 总计 +491.184。

其原因是旧 thermal 节点有活粗边但没有 hydraulic 权限，hydraulic duplicate 有匹配 modality 却没有自己的活粗边。Stage2 合并范围与重绑证据后，液压任务能够通过合法路径访问粗通道。专项保持同一函数和坐标只改 modality，验证 Stage0/未维护状态的 thermal 能 coarse、hydraulic 不能；维护后两者均可。这是知识和授权关系整合带来的真实可执行收益，旧的纯改价解释不适用。

同时，主流把全部 adiabatic 设为 hydraulic，全部 quench 设为 thermal；quench 从 T=0.20 起，早已同时越出新旧证书盒。所以本主表没有单独估计“旧盒放行而新盒拒绝”的范围收缩收益。安全修复可以通过自身控制验收，无需要求每一项都单独盈利。

高收益 adiabatic 在固定流的位置是 21、22、45、46、69、70、93、94；第 46 题正是第四个到来，跨过维护投入。应把 46 表述为此家族配比、顺序和粗细单价下的观测结果。对真 Frozen_Open 的第 41 题回本及 +751.3442 也应保留相同限定。[S2](https://github.com/erzhu419/Assumption-Agent/tree/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/docs/evidence/v17_5_m4_3_channel_bound_lifecycle_20261120)[S3](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py)

### Compiled 是本轮值得保留的强对照

Compiled 和 PPR 取得相同 96/96 质量、相同平均 M=31.3333、相同 87 次 recall 与 31 次非直接关系命中。PPR 平均候选更少，2.5938 对 2.7604，但计算代理更高；按提交原始账本、包含额外 1.818 的编译费后，Compiled 的全周期仍便宜 **4.9548**。该差额不等同于完整事件计费或实测墙钟的优势。

这说明当前关系知识可以被两种执行方式有效使用。它支持保留森林及其契约；也要求把 PPR 本身的增量优势留给确实能区分两者的证据。`non_direct_walk_win` 在 Compiled 上同样出现，表示没有直接签名命中，并不自动等于在线 PageRank 的必要性。Alpha 相同的 PoolOn/Off 本轮逐项结果相同，因而 pooling 的可检测边际效果为零，不能把它继续计为本轮增益来源。

报告对动态更新的措辞也超出了当前实现。两者都在构建时固定节点快照及检索结构，PPR 预先计算 transitions，未随正式 revision 刷新。正式把 W4 从 v1 修订为 v2 后，旧 PPR 与旧 Compiled 都返回 v1；重新构建后才返回 v2。Stage3 冻林不会触发这个风险，M5 真正在线写回时必须检查 forest revision 并刷新索引。小图先直接重建并记账即可。[S12](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/hipporag_wormhole_index.py)

### 噪声对照公平成立 但范围更窄

所有臂按同一目标 seed 获取相同 pilot 噪声，未发现对 Hegel 暗给洁净 pilot。实际分布是 `random.gauss(0,.002)`，没有截断，所以应称高斯噪声，不应称有界噪声。标准差约为容差的 3.23%，不代表每个实现值小于 3.3%；主 192 个带噪 pilot 的最大绝对扰动为 0.0052912692，约为容差的 8.53%。[S3](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py)

官方附加双轨比较只有首个 seed 的八个 onset，并只保存成本摘要。本审计复用提交森林补出 32 条质量记录：

| 噪声 σ | 臂 | 严格通过 | 假确认 | 平均 M | 平均 C |
|---|---|---:|---:|---:|---:|
| 0 | Hegel | 8/8 | 0 | 28.0 | 29.1623 |
| 0 | BestPilotRMSE | 8/8 | 0 | 28.0 | 32.9250 |
| .002 | Hegel | 8/8 | 0 | 29.5 | 31.6748 |
| .002 | BestPilotRMSE | 8/8 | 0 | 30.0 | 37.0250 |

费用精确重现官方、overlap 均为 0。这是可保留的正结果。有噪时 Hegel 也增加测量；相对 BestPilot 的优势为每题 0.5 测量费用，八题合计少一次购测，更大差额主要来自计算代理。它没有重新建立普遍的信息论不可辨别下界。

DEV 的六家族覆盖是真修复，不过实际观察行直接取 `oracle_fn`，没有施加目标对象声明的 .002 噪声；Pi 所选两例也都是 adiabatic。排序依据命中频次，不是完整搜索策略的费用最优化。准确名称应是“六家族无噪参数 DEV 的命中顺序”。后续需要同分布 DEV 时，复用主采样接口即可；当前不需要为了修辞追加庞大噪声扫频。

## 对整个项目定位的判断

本版更有力地支持了一个可操作的研究主张：**历史形成的规律、角色解释、适用范围、关系路径和已确认实例，可以作为后续求解的可执行资产，减少重复发现与不必要的高成本观测。** 这是项目哲学构思在受控环境中的一个具体实现方向。

两条 P1 恰好说明下一步应保持的概念区别。同一法则可以有不同实例；同一表面坐标集合可以有不同物理解释。历史的总结和一般化不能把这两种差异抹掉。结构归纳、实例复访、适用域迁移应该分别知道自己沿用了什么证据、需要补充什么证据。这会加强项目的历史保留与扬弃机制。

当前四个 seed 都从一次形成/维护后的森林复制，Stage3 不写回学习结果；Pi 内部动力学参数固定，变化主要是读出、任务参数和预设时域；OOS 是同一个不连续模板的参数变体。因此它是一次形成链后、多组新参数的受控迁移证据，不能等同于四条独立递归自我改进链，或开放世界的一般 RSI 已经成立。[S3](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py)

普通 AllHistory 与森林还存在执行支持差异：前者主要获得 AST 候选池，森林额外持有关系、证书通道授权及可信 fine solver。因此 Hegel 对 Ordinary 的总差属于整套知识组织与执行支持的系统比较；Compiled 同森林对照才更接近检索器本身的比较。新结果不要求放弃图，也不能据此把当前 PPR 形式定为唯一正确实现。

## 下一轮有限修复计划

建议先完成一个明确边界的 M4.3.1，再推进真正在线学习的 M5。验收不需要重开全部旧版本或添加无限边界世界。

| 顺序 | 工作 | 结束条件 |
|---|---|---|
| 1 | 节点及分支保存预期实例；证书与实例绑定；准确回执字段 | 合法原实例可用，同 AST 另一合法参数实例拒绝，恢复原实例可用；真实旧证书错配被拒 |
| 2 | 同一角色映射贯穿通道、模型、fresh batch 与最终 query | 正常 T21 不退化，短时交换坐标控制不得 coarse 错误确认，fresh 首点耗尽后仍留在合法域或升级 |
| 3 | 修正 fine 系统矩阵，去掉无效试拟合或计费，明确 host solver 计算费 | 同一正常 quench 对照仍通过，实际候选数与账本一致；原有全研究重跑并更新费用 |
| 4 | 稳定内容身份与可迁移路径分离；处理两个兼容入口 | 新位置恢复后可验证身份不变，严格 loader flags 不被 legacy 绕过，旧 fine 已付记录不丢失 |
| 5 | 修改报告主张并准备动态索引刷新 | 将 Gaussian、同形成链四 seed、维护收益来源、Compiled 优势写清；进入在线写回前，旧版 scope 不再由旧索引返回 |

其中前两项是当前“全部核心闭合”验收的阻断项；第三、四项是已确认且修法明确的工程补丁；第五项将文案和后继运行方式与证据对齐。保持现有真 Frozen、真 CodeAgent、BestPilot、Compiled、完整主回执及可恢复森林，不需要再换一套评测骨架。

修复后的最小回归就是：现有相关 40 项测试、上述有限正负控制、一次既有四 seed 全研究。只有某项失败揭示具体新风险时才扩大测试。若后续目标是证明持续学习而非冻林迁移，再让新确认真实写回同一森林，并对这些更新重建/失效索引、记账、检查旧能力保留；不应先靠扩大测试 seed 数代替独立形成链。

## 证据与复现材料

完整包包括本报告、三个专项报告、有限控制脚本及真实输出工作区、完整重跑证据、回归日志、结构化差异检查和 SHA256 清单。没有拷贝整个 Git 仓库。运行环境、入口命令和输出目录约定见 `README_REPRODUCE.md`。

核心定位如下：

- `instance/probe_same_ast_instance_swap.json`：同源码同 AST 的真实参数实例错绑及预期摘要正控制。
- `scale/scale_controls.json`：正常 T21、轴映射反例、实际 fresh 越盒、正常 quench、R3 冷载/只读/删边控制。
- `scale/real_certificate_instances.json`、`scale/archive_digest_checks.json`：真实证书身份交叉与归档摘要复算。
- `cost_events.json`：正常四目标的实际候选评价、冷验证和 host RK4 调用观察。
- `evaluation/evaluation_analysis.json`：1,144 行配对、维护分解、modality 通道控制、索引版本控制。
- `evaluation/fork_quality/fork_quality_receipts.jsonl`：双噪声设置、双臂共 32 条质量回执。
- `replay/`、`replay_verification.json`：完整研究及回归日志，原始与重跑证据差异。

### 固定提交源码与原始证据

- [S1 本轮工作总结](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/17_V17_5_M4_3_Channel_Bound_Lifecycle_and_Audit_Closure_Report.md)
- [S2 原始证据目录与主回执](https://github.com/erzhu419/Assumption-Agent/tree/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/docs/evidence/v17_5_m4_3_channel_bound_lifecycle_20261120)
- [S3 M4.3 主研究与执行器](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py)
- [S4 R3 实例加载调用](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L578-L583)
- [S5 确认实例摘要与 loader](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_compositional_discovery.py#L1224-L1555)
- [S6 森林证书再验证](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/wormhole_forest.py#L1140-L1288)
- [S7 规范坐标盒映射](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py#L2410-L2458)
- [S8 原始动力学与角色拟合](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py)
- [S9 归档与冷恢复 runner](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/scripts/run_v17_5_m4_3_unified_study.py)
- [S10 新 fine 分支与候选计量](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py#L1297-L1394)
- [S11 CostMeter 费用定义](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/cross_domain_clusters.py#L29-L91)
- [S12 HippoRAG 索引实现](https://github.com/erzhu419/Assumption-Agent/blob/3461535364fcdb4e9ae9c87ccba502b7e6b56fe8/Hegel%20Machine/src/hegel_machine/cognition/hipporag_wormhole_index.py)
