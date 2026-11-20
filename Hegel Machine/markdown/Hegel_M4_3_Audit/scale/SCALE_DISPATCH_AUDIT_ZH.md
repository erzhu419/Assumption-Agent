# V17.5 / M4.3 尺度、证书、作用域与 R3 专项诊断

固定提交：`3461535364fcdb4e9ae9c87ccba502b7e6b56fe8`；对照：`fb4b23aaf802c20b4e2d0018436abab73b0cca06`。

本专项只读审计源码、提交内工作总结、M4.2 审计、可重定位快照和真实 authored artifacts。遵循 AGENTS 的历史复用约束，已读 reuse hub，并保存 `historical_reuse_routes.json`。全部新控制直接恢复提交内森林，没有重新建林，没有修改源文件或提交证据，没有模型 API。40 项测试、完整 1144 行 study 和成本因果分解由主审及其他专项负责；本文件不重复这些运行。

## 核心判断

M4.3 在真实双通道、R3 只读、冷载零 fit、实际边依赖、窗口投影参与认证、归档恢复方面有实质改进。旧 M4.2 的“任何粗边都给 Pi 便宜收费”“删粗边仍照样升级”“验收函数临时改作用域”不能继续原样列为未修缺陷。

但新角色映射把“某个排列能落进盒子”误当作“实际预测模型使用的物理坐标映射”。这能在未改 oracle、未改森林、未放大容差的合法短时 Pi 任务中产生真正的错误确认。确认点足额补选还没有接到证书作用域。证书与节点实例的绑定、可重定位后的语义摘要也有剩余缺口。正常 benchmark 的成功率不能等价为这些接口不变量已经全部闭合。

## 发现总表

| 编号 | 判断 | 已观察后果 | 对正常已提交主表的影响范围 |
|---|---|---|---|
| S1 | P1：证书盒角色映射与实际拟合角色不绑定 | 短时点被交换 u/T 后按 coarse 确认；heldout RMSE 0.100316，容差 0.055；`false_confirmation=true`，最终盒守卫未拦 | 有限合法接口变体；不声称原 1144 行已经触发 |
| S2 | P2：实际 fresh confirmation 未重新检查 verified box | 第一预声明点已在 pilot，fresh helper 选到第二点 T=0.4，仍按 coarse 购买和确认 | 正常 fresh1/既有点分布通常不触发；不能声称所有消费者均受盒守卫 |
| S3 | P2：真实证书可与不同 confirmed instance 交叉再验证 | 真实 Stage0 证书绑定 416e…，Stage2 节点绑定 0586…；API 接受并形成 edge/certificate 双重身份 | 当前正常 Stage2 三方实例相同；控制仅证明身份不变量缺口，未制造数值失败 |
| S4 | P2：M43 所谓 exact 两模 surrogate 的系统矩阵写错 | A12 写为 g_sf/tau_s，而 RK4 方程为 g_sf；正常 T23 heldout RMSE 0.005842，最大单点误差 0.0097155 | 当前正常 quench 在 0.055 容差内通过；“exact”及更紧精度承诺不成立 |
| S5 | P2：归档语义摘要不能由归档内容复算 | Stage0 7/8、Stage1 14/15、Stage2 14/15 节点 semantic digest 不符；Pi 证书 digest 也不符 | 正常提交归档直接存在；冷载 R3 仍 10/10，因为当前消费端不检这些摘要 |

另一个专项已经取得更严重的节点实例绑定反例：多数 artifact 节点没有节点侧 `confirmed_instance_digest`，允许相同 AST 的另一真实参数实例被接受。该结果应优先于本专项 S3/S5，勿把“artifact 自身 hash 一致”误写成“与被调用节点的实例身份一致”。

## S1：合法短时 Pi 任务能绕过盒子并错误确认

源码位置：

- `src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py:2410–2458`：`_find_valid_canonical_role_map_for_box`。即使目标已使用 `u_drive`、`T_horizon` 这两个规范名称，identity map 失败后仍枚举其他排列，只要任意排列把检查点塞进盒子就返回。
- `src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py:1141–1177, 1257–1275`：把该 map 用于通道资格和候选资格。
- 同文件 `1384–1391, 1450–1451, 1663–1669`：实际拟合重新独立尝试 role_order，随后却把前面的盒子 map 写进最终 scope contract；没有校验二者一致。
- `src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py:1427–1477`：拟合器实际返回 `role_order`，现成字段已经足够接这条不变量，不需要新算法。

控制使用同一个 fresh seed 20301120 的原始 T21 Pi target、原始 fine/coarse oracle、原始 0.055 容差、同一 Stage2 森林。正常对照完整保留：T21 走 coarse，实际 `role_order=['u_drive','T_horizon']`、guard identity map，heldout RMSE 0.008633，strict pass。

变体只改变合法点分布及声明 bounds：pilot 的 u 为 2.45–2.95，T 为 0.40–1.20；confirm 是 (u=2.70,T=0.47)；heldout 为 (2.50,0.42)、(2.85,0.55)、(2.75,0.75)。真实证书为 u∈[0.4,3.0]、T∈[2.4,5.2]，这些时间全部在证书外。

helper 却返回：

```json
{"u_drive":"T_horizon","T_horizon":"u_drive"}
```

拟合器 spy 记录实际最佳模型仍为 `role_order=['u_drive','T_horizon']`，train RMSE 约 1.40e-8，编译出的 `u_cand_00.py` 也按原物理 u/T 计算。也就是：授权按交换后的坐标，预测按未交换的坐标。最终 heldout 检查再次使用交换 map，因而未拦截。返回 `completed`、`confirmed_ok=true`、`selected_channel='coarse_macro'`、`out_of_box_query_blocked=false`，而 heldout RMSE 0.100316 > 0.055，`false_confirmation=true`。

这不是“任意 Python”或隐藏尖峰要求；控制仍用本项目自己的连续两模 RK4 体系和原始已支持 AST，仅改变已支持任务的查询区间。

建议：固定证书的物理坐标解释，不能用是否落盒来反推角色；把经验证的角色绑定贯穿通道资格、编译/拟合、确认实例与最终 scope。若保留角色排列候选，应把角色排列当成具体候选身份的一部分，某一排列的证书不能授权另一排列的模型。规范名称已明确时，不允许靠交换坐标规避拒绝。

证据：`scale_controls.json` 的 `native_adiabatic_T21_control`、`canonical_axis_role_confusion`，包括 actual fitted candidates、每次真实 acquisition 及 scope contract。

## S2：fresh 的确认点仍可能越出证书

源码 `m4_3_channel_bound_lifecycle_and_audit_closure.py:1152–1155,1262–1264` 只检查 pilot + `confirm_points[:1]`。实际补选在 `1400–1406` 调用 `select_or_synthesize_disjoint_confirm_points`，只传任务 `feature_bounds`，没有传证书盒，也没有在 `_acquire_points` 重新检查实际选出的点。

本控制没有把 fresh_points 改成 2，也没有改程序。保留默认 fresh1，将首个预声明 confirm 设成已经付过的 pilot[0]，第二个 confirm 设为 u=2.8、T=0.4，任务 bounds 明示允许这些物理查询。前置守卫看见的是盒内首点；fresh helper 正确跳过它后，真实采集日志显示越盒的第二点以 `coarse_macro`、`phase='recall_confirm'` 被购买。模型在 coarse oracle 上得到确认，原始盒内 heldout 继续 strict pass，RMSE 0.008633。

因此 fresh disjoint 修复是真的，盒内约束却没有随实际选择结果继续传递。该控制没有预测错误，不能把它升级为第二个 false-confirmation 数值反例；它证明了“fresh 与 scope”两条独立不变量需要同时成立。

建议复用已有 helper 的 `verified_box` 参数，并在规范角色映射后求目标域与证书域的交集；再对实际 acquisition batch 校验。校验失败应缩到合法 fresh 点、升级 fine 或返回 scoped unknown。M4.1 多尺度 suite 在 `m4_1_unified_discovery_and_renormalization.py:708–720` 已有传 verified_box 的接法，不需重新发明。

## S3：盒包含关系已修，证书身份仍未绑定

`wormhole_forest.py:1210–1249` 现在拒绝窄证书授权宽盒，这是正确修复。`1275–1284` 也会把节点 ScaleMap、传入 certificate、节点 action 的实例 digest 分别复制到边。正常提交 Stage2 的 node、edge、certificate 三方实例均为 058680…，其 box 也同步；不能沿用 M4.2“正常链不同步”的旧结论。

问题在于复制前没有验证这些对象属于同一实例。把真实 Stage0 edge 内的证书原封不动传给真实 Stage2 节点，API 返回 accepted。Stage0 证书盒 [2.2,5.2] 包含 Stage2 [2.4,5.2]，所以几何条件成立，但得到：

- edge `confirmed_instance_digest` = 当前节点的 058680…；
- edge certificate `confirmed_instance_digest` = 旧证书的 416e0b…。

两个 artifacts 均通过真实 strict frozen loader，source hash 与 AST 相同，参数快照不同；共同盒内两点的预测仅相差约 9e-8、6e-8。这是实例身份/证据归属缺口，不是数值越界反例。不同参数的真实实例来自已有 Stage0/Stage2，不是伪造 `certified=true`。

此外正常 builder 在 `m4_2_unified_closure_and_pi_lifecycle.py:1855–1868` 先直接写节点 action，再调用再验证函数更新 ScaleMap。正常成功路径最后一致，但这不是一个由单一接口原子验证、提交的更新。失败路径不应留下新 action 与旧 scope 混搭。

建议把 node identity、confirmed instance、canonical roles、fine/coarse model 参数身份、ScaleMap 和 certificate 组成一次待提交记录；先验证全部绑定，再一次性 version/rebind。至少先比对现有 `confirmed_instance_digest`，不要只在成功后分别复制两个不同来源的值。

证据：`authentic_certificate_cross_instance`、`real_certificate_instances.json`。

## S4：两模“exact”表达式实际用了错误矩阵

真实 RK4 的方程在 `m4_1_unified_discovery_and_renormalization.py:104–126`：

\[
\dot x_s=-x_s/\tau_s+(a_s/\tau_s)u+g_{sf}x_f,
\qquad
\dot x_f=-(g_{fs}/\tau_f)x_s-x_f/\tau_f+(a_f/\tau_f)u.
\]

所以矩阵左上行第二列应是 `g_sf`。新 M43 evaluator `1326–1338` 却用 `g_sf/tau_s` 算特征值。默认值下真实 A12=0.35，surrogate A12≈0.24137931。后面用三个真实 RK4 值解三个系数，会在那三个标定点补偿，但不能把错误特征值变成同一 ODE 的精确解。

原始 seed20301120、原始 T23 quench 直接执行：整体 RMSE 0.005842，四个 heldout 的误差依次约 0.002340、0.005820、0.0097155、0.001662。它仍通过当前 0.055 容差；这个 findings 不否定当前 quench success。应修正矩阵，或将该通路如实称为三点校准的两指数近似，并让近似误差受 contract 约束。

注意此错误只在 M43 目标 evaluator 的新 surrogate 分支。R3 的 fine dispatcher 直接调用真实 `_eval_pi_s_fine_scalar`，没有走这段错误特征值公式，不能把 R3 的数值成功一并否定。

## S5：归档已能恢复执行，但摘要不是归档内容的摘要

`scripts/run_v17_5_m4_3_unified_study.py:43–57` 递归把临时绝对路径改成相对 artifact 路径；`74–95` 冷载又改成新绝对路径。node semantic digest 的生产计算在 `wormhole_forest.py:704–710`，包含 claim contract 中的 artifact 路径。certificate digest 在 `1272–1274` 对整个 certificate JSON 求 hash，也包括 artifact_path。

本专项完全按生产的五个字段调用同一 `canonical_digest`：task_view、observation_map、scale_map、claim_contract、program_steps，没有自行增减字段。提交内三个快照不一致数分别为 7/8、14/15、14/15。probe_signatures 不属于这个 digest，因此 M43 builder 对签名的直接修改不能解释该结果。Stage2 Pi 边的 recorded certificate digest 为 759d52b3aa5ce945，当前归档 certificate JSON 重算为 75ab6250ec4156e4。

没有额外重建森林来恢复打包前对象，所以不宣称路径重写是每个不一致的唯一来源；确定事实是归档内容不能重算保存的摘要，而且现有序列化操作本身会改变参与摘要的路径。

这不等于归档恢复失败。独立从新路径加载提交内 postStage2 快照，R3 十条路由全部正确；完整持久化森林前后序列化相等，Python profile 实际观察 `fit` 调用为 0。删去 coarse edges 后 Q05 返回 `escalation_blocked_missing_coarse_edge`，不再默认兜底到 fine。

建议把内容身份和存储地址拆开：摘要绑定稳定 artifact 内容/实例标识，路径通过可重定位映射解析。不要只重算新 hash 并掩盖旧历史身份；恢复时应能验证原身份保持不变。

## 已修复项目与研究解释边界

1. `MultiScaleWorkspaceTaskLab.acquire_channel` 确实依据 channel 选择不同 oracle，返回带 channel 的 rows，按 3/10 单价收费。不是旧版单一 fine truth 改价格；Stage3 最终评估仍对 fine truth。
2. 新 `certify_regional_pi_s_closure_with_trajectory_projection` 同时检查 endpoint closure、终末 fast slaving、窗口投影 drift。窗口 drift 现在进入 certified 布尔值；旧“非末窗口只是记录字段”的批评已被修复。它仍是有限轨迹上的经验认证，不需要为本轮引入全空间形式化证明。
3. Stage2 正式 revision fine 节点至 thermal|hydraulic；R3 函数不再现场改合同。本专项直接 profile 与完整 forest 状态比对支持“只读、零 fit”。
4. 真实可重定位 Stage0、Stage1、Stage2 森林及新 Pi artifact 现已落盘；冷载成功不再依赖被删的临时工作目录。
5. 本轮 Pi 任务的 quench pilot 实际从 T=0.20 开始，早已在旧 [2.2,5.2] 盒外；adiabatic 点也同时落在新旧盒内。因此这些原始 target 不能单独证明 box contraction 的收益。关于 491.184 的具体归因以 evaluation 专项的配对分解为准：收益集中于 hydraulic adiabatic 的 OR scope/通道访问，不能称作主要来自短时盒收缩。

## 建议修复顺序

先合并节点实例 anchor 的 P1 与本专项角色绑定 P1。随后把“实际 fresh batch 也必须在同一作用域”“证书与节点实例相同”“scope/action 原子更新”接入现有接口。修正两模矩阵并保留正常 quench 对照；归档再改为位置无关身份。上述都是现有实现内部的接口修复，不需要重建理论栈、重写已有研究器或增加无界形式证明。

复现脚本：`run_scale_controls.py`、`verify_archive_identity.py`。核心 JSON：`scale_controls.json`、`real_certificate_instances.json`、`archive_digest_checks.json`。全部控制工作区也在本目录，保留实际编译模型、观察行与 scope contract。

## 本专项阅读清单

以下是本专项实际追踪的源路径与接口；其余目标族、全部新源码、完整研究脚本及历史理论综述由主审分工覆盖，不以本专项代替全仓库逐字覆盖声明。

| 路径 | 本专项检查内容 |
|---|---|
| `AGENTS.md`、historical reuse hub/capability catalogue | 当前审计规则、历史组件复用路线；历史路线精简输出另存 JSON |
| `markdown/hegel_v16_5_multi_precision_plan/17_V17_5_M4_3_Channel_Bound_Lifecycle_and_Audit_Closure_Report.md` 及同名上传附件 | 本轮尺度/身份/R3/归档主张 |
| `markdown/Hegel_M4_2_Audit/M4_2_AUDIT.md` 的尺度、实例、R3、轨迹与归档章节 | 对照上一轮真实问题，区分修复与剩余缺口 |
| `src/hegel_machine/cognition/m4_3_channel_bound_lifecycle_and_audit_closure.py` | closure controls；builder变更；Pi target；完整 `evaluate_m4_3_target_under_arm` 的通道、拟合、fresh、fallback、最终scope与heldout链 |
| `src/hegel_machine/cognition/m4_2_unified_closure_and_pi_lifecycle.py` | 默认ODE参数、两通道计算、完整certificate生成、完整dispatcher及R3 stream、Stage2 fine OR revision与重认证提交、角色盒helper、旧M42 evaluator调用 |
| `src/hegel_machine/cognition/m4_1_unified_discovery_and_renormalization.py` | RK4原方程、Schur消元、尾窗/末态projection、strict box helper、candidate role fit及本提交diff |
| `src/hegel_machine/cognition/m4_multiscale_renormalization.py` | 真实双通道lab、fresh选择/合成helper、声明式program约束 |
| `src/hegel_machine/cognition/wormhole_forest.py` | 本提交全部diff；生产semantic digest、注册/修订、CG边活性、证书再验证及对象同步 |
| `src/hegel_machine/cognition/m4_compositional_discovery.py` | strict frozen loader/predict接口经真实实例调用；其内部完整identity审计由instance专项负责 |
| `scripts/run_v17_5_m4_3_unified_study.py` | 可重定位序列化/加载、临时目录退出后的冷恢复验收 |
| 提交内 evidence 的三个 forest snapshot、Stage2 certificate、Stage0/Stage2 Pi artifacts及manifest/params | 恢复、交叉证书、相同AST不同实例、摘要复算及R3只读执行 |

执行：在当前审计目录直接运行 `python3 run_scale_controls.py`、`python3 verify_archive_identity.py`。脚本的固定 ROOT 指向本次 commit checkout；复用者改 ROOT/OUT 即可，逻辑不依赖临时 build 或远程服务。
