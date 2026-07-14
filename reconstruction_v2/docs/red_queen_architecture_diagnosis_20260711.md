# Assumption Agent × Red Queen Gödel Machine：架构诊断与 Reconstruction V2 复核

> - 初版日期：2026-07-11
> - 本次复核：2026-07-14
> - 代码审计基线 revision：`6224bb5a279f50fbcf1f8b36d19cb4ce6cc6c882`
> - 本次实现复核：receipt/runtime provenance 修复提交 `e43670f6`、`18ff3417`；v3.3 execution-policy 提交 `e0b1a33b`；v3.4 model-only/action-budget 主提交 `e491b0af`，runtime-path 修复 `995e6446`，Ruoli 503 分类修复 `ba0f36cf`，host-readable audit artifact 修复 `1df3092a` / `ad66d5a2`；v3.4 max2 v5 canary 已通过、fresh development 因四并发 429 fail closed；v3.5 将所有在线 phase 版本化为 1 worker，repair identity 修复 `96d53a5d`，malformed proposal/claim binding 修复 `d70562de`；v3.6 contrastive evidence / invalid-evidence lifecycle 实现提交 `01608e1e`；v3.7 六路首批 6/6 收到 429；v3.8 两路在 16 valid 后收到 2 个 503；v3.9 固定为 outer item workers=6 / shared model slot=1，并完成首个 clean full development 负结果；v3.10 exact-three/coverage-first fresh run 将 activation 提高到 2/16 但仍 0 gain，并暴露 semantic-diversity hard reject 与 action lowering 丢 target；v3.11 actionability fresh run 证明 treatment 已改变 trace、PDF 与成本但仍 0 gain，同时暴露 repair response shape 未被 generic system contract 可靠约束；v3.12 显式版本化 singular repair response并完成 56/56 clean development trials，但两代仍各仅激活 1/16、0 gain，无 incumbent；误入的空 freeze/partial controls 已隔离并补上 phase prerequisite；v3.13 complementary program-set 已完成 375/375 离线测试和 76/76 valid live development，三套 bundle 均激活 2/16、6 个 policy-on 全部失败、0 gain/0 harm、无 incumbent，同时暴露 G2 cross-arm raw replay 不一致；v3.14 提交 `2229d7af` 完成 411/411 离线测试及 62/62 attempted live development，selector 成功选中 7/7 三-family set、activation=3/16，但 7 个 policy-on 全失败；一条 recursive raw 超 64 MiB 使 primary non-claim，valid baseline replay=31、invalid key 又跨臂执行一次，两份 archive 仍无 incumbent；v3.15 action-quality / terminal-invalid provenance 实现提交 `696a2954`，453/453 离线测试通过，随后 clean lock、86/86 cache-only prewarm、smoke 与 57/57-valid live development 全部完成，但两臂仍 0 gain/0 harm、`incumbent_id=null`
> - 最新 proposal-only 复核：v3.16 family-slot formation 提交 `6ad5c156`；v3.17 artifact-blueprint formation 提交 `4f94e613`。两轮均失败且未启动 benchmark trial
> - 最新表示层进展：commit `b03c643a` 的 causal action-span extractor、closed typed operator/artifact graph 与 opaque recipe-only selection 已完成唯一一次正式离线 decision，9/9 preregistered predicates PASS，既有 report/event/lock 精确复验 PASS；该结果只使另行冻结的 typed-selection integration diagnostic 有资格运行，仍未授权 development
> - RQGM 版本：arXiv:2606.26294v2，2026-06-29
> - legacy 代码范围：`assumption_os/`；legacy 报告范围：`reconstruction/md/` 与对应 artifacts
> - v2 范围：`reconstruction_v2/`

本文从已故障的旧任务“继续调试 assumption”的本地完整记录中恢复了 2026-07-10
的 Red Queen 原始诊断，并用当前代码、测试、实验报告和本地论文重新核验。旧任务
中的凭证、网络连接参数和与架构无关的敏感内容没有复制到本文。

本文同时保留两个时间切片：

1. **legacy 诊断**：解释旧 HLE 系统为什么不是由假设学习闭环驱动；
2. **v2 复核**：判断上述缺口哪些已经修复，哪些只是有了接口或测试，哪些仍会
   阻断论文级结论。

除非明确写成当前状态，legacy 的实验数字和行号只描述诊断时的旧实现，不代表
`reconstruction_v2` 的最新性能。

## 一、执行摘要

### 1.1 最准确的当前结论

旧诊断的主结论仍成立：legacy HLE 把最有研究价值的“假设学习”放在了旁路，
真正控制答案的是一个由 prompt、检索、手写规则、verifier、fallback 和 selector
组成的高维控制面。`trace -> transition -> miner` 虽然存在，却没有可靠地改变下一题
的 runtime，也没有用 policy-off/on 反事实估计因果收益。

但这句话不能原样套到 v2。`reconstruction_v2` 已经在**接口和实验 harness 层**接通：

- 三类结构化 `HypothesisProgram`；
- 内部 effectful runtime；
- 递归 proposal repair；
- paired policy-off/on evaluation；
- train/validation/sealed-test guard；
- archive node、evaluator epoch 和 selective invalidation 骨架；
- SkillLearnBench instance-out/family-out 协议。

因此，当前诊断应更新为：

> **学习闭环在 harness 层已接通；promotion 所有权、外部 backend action/fallback
> 边界和 86-item 离线可运行协议已经闭合。contrastive trigger learning 已在 v3.6 的代码、
> manifest、离线测试和 live train 中运行到真实 paired validation，但串行轮只完成 2/16 pairs
> 后因吞吐主动终止。v3.7 把跨题 worker 从 1 改为 6，但首批 6/6 请求均收到 429，
> 熔断后其余 30 条本地跳过。v3.8 两路完成 16 valid 后又同时收到 2 个 503，熔断跳过 20。
> v3.9 的 6 路题级 pipeline / 1 个在线 agent slot 已 clean 完成，但两代 candidate 均只激活
> 1/16 validation、0 gain/0 harm。v3.10 fresh root 随后完成 38/38 valid train、16/16 valid pairs、
> 56/56 actual trials，0 provider/infra/mismatch；exact-three coverage-first 候选覆盖 2 个 train family、
> 6/6 failure precision，并把 validation activation 提到 2/16，但 candidate/raw 仍同为 3/16、
> 0 gain/0 harm。第二代两次 exact-three response 都因三项 activation signature 坍缩为同一组而被
> 旧合同 terminal reject，两臂 report 因 proposal failure non-claim，archive 仍无 incumbent。
> 离线轨迹复核证明 routing/treatment 实际执行并改变了命令和答案；`selection_change_count=0` 只因
> backend 把 answer 投影为 success 布尔。真正 blocker 是 proposer feedback 硬编码 completion check，
> 以及 lowering v1 在非空 value 时丢掉 `execute_step`/`check_condition` target、把 action 降为含糊
> JSON mode blob。v3.11 因此只修 gate 前的 actionable directive、lowering 与 diversity audit；
> fresh root 的 38/38 train 全部 valid（5 success / 33 residual），exact-three 返回三种 distinct
> activation signature，两个 root 静态通过。入选 court policy 在唯一激活题上真实执行：raw/candidate
> action starts 为 66/16，输出 PDF 内容不同，但二者都失败；no-rec 汇总仍为 4/16 对 4/16、0 gain/
> 0 harm。该 arm 又因一个未激活的 raw poster trial 超过冻结 64 MiB 而 non-claim。recursive arm
> 则在 validation 前失败：repair transport/JSON 成功，却返回 batch 字段 `hypotheses`，而 repair parser
> 需要 singular `hypothesis`。事后调用链复核纠正了最初判断：真实 repair payload 并不含
> `proposal_batch_contract`，只是 generic system contract 没有 versioned repair-specific singular
> override，模型仍复用了 root response shape。v3.12 因而保留 train coverage objective，并新增
> top-level one-object/`hypothesis` response contract；proposer 只作防御性 batch-contract 清理。exact-three root、
> train selection 和所有 evaluator/promotion/split/fuse/retry/sealed 合同不变。fresh root 已完成
> 56/56 valid external trials：38 train 为 8 success / 30 residual，0 provider/infra/action/network/
> mismatch；两代 exact-three 均 3/3 static pass、三种 distinct signature，但每个 root 都只覆盖
> 一个 train family。coverage-first 连续选择 court root，held-out 都仅激活 `court-form-filling-5`
> 这 1/16，raw/candidate 均为 4/16、0 gain/0 harm，action starts 从 raw 9 变为 candidate 32/43。
> 两臂以 consecutive non-promotion 停止，archive 字节相同且 `incumbent_id=null`。由于 12 个
> static node 全部通过，本轮 repair request=0；所以 singular scope 未破坏 full development，但
> 尚不能声称 full-run repair path 已验证。随后旧 runner 错把空 archive 写成 frozen receipt 并误入
> controls；该批仅 2/96 record 且全部 interruption-invalid，现已 quarantine，禁止拼接或作 claim，
> sealed/test 未访问。runner/freeze/control consumer 已统一要求真实 recursive incumbent；这是阶段
> 前置条件，不是新增评分 gate。
> v3.13 随后把互补 root 组合成 program set，并在隔离第一次 partial development 后保留同一冻结
> lock、86/86 prewarm 与 smoke，重新执行正式 development event/work tree。正式轮为 76/76 valid
> attempt-one trials：38 train-off、32 validation-off、6 validation-on，0 provider/infra/action/network/
> mismatch；最大 69/100 actions、62,200,000/67,108,864 network bytes。三套实际 bundle 都由 poster
> 与 court 两个 TRAIN-perfect、0 success-FP 成员组成，只激活 `anthropic-poster-design-2` 与
> `court-form-filling-5`。六个 policy-on 与对应 raw 全部失败，四个 generation decision 因而都是
> 2/16 activation、0 gain/0 harm；两臂均 consecutive non-promotion、`incumbent_id=null`。recursive
> repair 已真实执行且无 response/model failure。证据完整不等于性能提升，因此不 freeze、不跑
> controls/family-out/HippoRAG/sealed。另一个机制缺口是 recursive G2 复用 G1 的 16 条 raw，而
> no-recursive G2 重跑 16 条 raw并得到 4/16（前者 2/16）；两臂内部 pair 有效，但该差异不能纯归因
> recursion。v3.14 随后以提交 `2229d7af` 完成两项有限修订和 411/411 离线测试，再通过新的
> claim-eligible lock、86/86 cache-only prewarm、Plus canary 与 clean smoke。正式 development 完成
> 62/62 attempted trials：38 条 train 全 valid（7 success / 31 residual），0 provider/model/slot/action/
> mismatch。新版 selector 在 G1 按冻结 TRAIN objective 真正选中 7/7 failure support、3 families、
> 0/7 success-FP 的三成员 set，并把 held-out activation 提到 3/16；但 recursive/no-recursive G1 的
> 6 个 on 与 no-recursive G2 的 1 个 on 全部失败，所有可比 pair 都是 0 gain/0 harm。recursive G1
> 一条 court policy-off 使用 68,660,000/67,108,864 bytes，硬 fuse 正确抑制同 request retry，primary
> report 因而 non-claim 并停止；no-recursive 机械上 claim-eligible、两代 non-promotion。valid baseline
> cohort 产生 31 次零执行 replay；但 invalid 不入 evidence cache，导致相同 baseline replay key 在
> no-recursive 又执行一次并得到 valid row，所以该单题不能作严格 cross-arm recursion attribution。
> 两份 archive 都是 `incumbent_id=null`，没有 freeze/controls/family-out/HippoRAG/sealed/test。
> 该结果兑现了预先约定的停止条件：不再迭代 selector。下一问题是 action quality——三条 G1 directive
> 分别缺少实际 HEX、可用的离线漏洞数据源或新的表单操作，基本只是重述 task instruction；不能再靠
> trigger coverage 或 promotion gate 修补。Plus/Pro 都是同一 `gpt-5.4-mini` route，本轮 Plus 全程可用。
> v3.15 已在提交 `696a2954` 把该诊断落实为一个有界的 TRAIN-only action-quality 合同，并通过
> 453/453 离线测试：instruction 明确只是 baseline requirement，候选应补充 exact constant/mapping、
> concrete local tool command 或 artifact-internal manipulation 中至少一种 material delta；proposal 只接收
> 经过 allowlist、containment 与敏感信息过滤的 TRAIN public-environment facts 和 policy-off action-trace facts，
> 不读取 validation outcome、test、solution 或 verifier，也不给 proposal 外部工具、网络或运行时安装权限。
> `proposal_action_delta_audited` 只记录 material-delta/restatement 风险诊断，不拒绝 response、不 retry、
> 不触发 repair、不重排候选、不改变 promotion gate。相同 baseline request 在声明的 same-request retry
> 完成后仍 invalid 时，v3 replay policy 只写 run-scoped terminal tombstone；后续 arm/generation 零执行复用
> 同一 invalid，且明确 `promotion_evidence=false`。首代 checkpoint、action-profile count/set hash 同时进入
> 两臂 report，并由 freeze 逐臂及跨臂核验。正式 v3.15 root 随后通过 clean lock、86/86 cache-only
> prewarm 与 smoke，并完成 57/57 valid actual trials：38 TRAIN policy-off、16 shared validation baseline、
> 3 activated policy-on；8/8 proposal/repair model calls 完成，TRAIN 为 6 success / 32 residual，在线 agent
> 最大并发严格为 1，provider/infra/action-budget/network-cap/pair-mismatch 错误均为 0。recursive G1/G2
> 都只激活 1/16，candidate/raw 均为 4/16、0 gain/0 harm；no-recursive G1 static reject，G2 也只有
> 1/16 activation、0 gain/0 harm。共享 cohort 产生 32 次 zero-execution baseline replay。两臂虽均
> claim-eligible，但这里只表示 clean negative result 可用；它们都以 `consecutive_non_promotion_limit` 结束，
> archive 均为 `incumbent_id=null`，sealed/test=false，未进入任何 downstream phase。13 个 candidate audit
> （9 roots + 4 repairs）中 7 个有 material delta、6 个有 restatement risk，但所有 material delta 都仅是
> `exact_constant_or_mapping`，没有 concrete local tool、artifact manipulation 或 environment primitive；
> 9 个 root 又全部坍缩到 `anthropic-poster`，搜索从 v3.14 G1 的 3-family/7-support 退回单-family/2-support。
> v3.16/v3.17 随后没有直接重跑 development，而是复用冻结的 v3.15 TRAIN receipt 做 proposal-only
> feasibility：38 observations、6 success controls、32 failures、31 profiles、0 source-agent re-execution。
> v3.16 三个 logical call 全部成功，但 9 项标准失败 6 项。v3.17 固定 exact family trigger、空 anti-trigger、
> deterministic reusable artifact 与 read→parse→update→serialize→write-back blueprint 后，distinct single-family
> signature、support 2/2/3、3/3 concrete local tool、2/3 artifact manipulation、0 restatement/self-block 等 8/9
> 均通过；第三个 action 仍绑定两个来自失败 TRAIN command 的 primitive，故唯一剩余项
> `failed_profile_primitive_avoidance_passed=false`，整体仍 fail。一次 `RemoteDisconnected` 在同 request retry
> 后恢复，3 个 logical call 均完成，因此不是 credential/provider-capacity 结论。两轮 backend/evaluator/
> validation/test/verifier/sealed access 全为 0，没有 benchmark trial、promotion 或 archive。该 free-text
> family-slot 路线到此停止，不再做 v3.18 prompt/gate/acceptance patch。
> 仍未成立的是 clean development promotion、
> 跨 family 泛化，以及 Red Queen 式多谱系搜索和
> evaluator co-evolution。v3.3 已把 low reasoning/verbosity、32,768-token
> `body_after_prefix` compaction、10,000-token tool-output limit 和 request-compression
> 变成 protocol-owned treatment；`video-object-counting-1` 从 v3.2 的 71.1 MB hard-cap
> failure 降为 19.69 MB valid failure，full train 的最大流量为 40.6 MB，38/38 均未触发
> cap/provider error，说明本次 batch 未再被 fuse 直接阻塞；但 canary/full 的 1.47/19.69 MB
> 波动也说明跨运行稳定性尚未建立。full train 仍只有 37 valid、
> 9 success、1 invalid：`offer-letter-generator-1` 的真实 Codex JSONL 返回了一次
> `web_search` item，违反冻结的 model-only contract。因 `all_valid_before_proposal_v1`，
> proposal/validation 仍为 0。现已定位根因：Codex 0.144.1 会把兼容键
> `tools.web_search=false` 解析后丢弃，未设置顶层 `web_search` 时默认仍为 `cached`；因此
> v3.3 的 38 个请求都暴露 hosted web search，仅一条实际调用。v3.4 改用权威顶层
> `web_search="disabled"`。零模型 loopback 捕获证明，canonical 请求有 7 个本地工具且
> 0 个 `web_search*`，同配置仅换回旧布尔键的阳性对照有 8 个工具并明确包含
> `web_search(external_web_access=true)`。`max_steps=100` 也已定义为可观测的
> `codex_action_start_v1`：每个 `item.started` 都占一单位，由容器内 supervisor
> 在第 100 个 start 终止，并按 task/TID 清理专用 trial 容器基线后新增的所有 live task；它不是 semantic turn。异常退出、畸形
> start、残留 descendant 和 receipt/trace 不一致均 fail closed。v3.4 clean lock、共享 runtime
> 与 v4 86/86 prewarm 已通过；PATH 作用域和 root-owned `0600` audit artifact 两个本地问题
> 已分别修复。max2 v5 随后完成真实模型推理、2-step 受控截断和本地 verifier：action/tool/
> process/receipt 均 valid，0 web/remote tool。fresh development 因而获准从零启动；38 个 train
> request slot 中 17 条形成有效离线评价（3 success），4 条收到 `provider_rate_limit`，熔断后
> 17 条在本地跳过。没有 cap、action、tool 或 verifier violation，但 all-valid-before-proposal
> 正确阻止 proposal/validation/report/archive。由此可知 API/单次 route 可用，尚未满足的是冻结
> 四并发的持续容量。当时的 v3.5 因而只把五个在线 phase 的 worker 从 4 改为 1，其余合同
> 不变。首轮 serial run 取得 38/38 valid train、9 success、0 provider/cap/tool/action/verifier
> invalid，随后在静态递归验证中因两个不同 repair payload 复用同一 model-declared ID 而
> fail closed；没有 validation-split trial、promotion、report/archive 或 sealed。修复改由父分支、
> depth 与规范化内容派生确定性 repair ID，并保留 archive 冲突硬拒绝。修复后 fresh root 再次
> 得到 38/38 valid、9 success、0 provider/cap/action/tool/verifier invalid；root proposal 返回
> 3 个候选，但第一个 repair 的 transport/JSON 成功后没有 mapping-valued `hypothesis`，裸
> `ValueError` 穿透旧异常边界，仍在 validation trial 前退出。现已把 malformed root/repair
> envelope 与 canonical parse 纳入既有 typed failure isolation，并让 report/freeze 从 generation
> rows 绑定非 claim 状态；没有新增评分 gate、响应 retry 或在线 evaluator。两次 38 条都不能跨进程复用。其后第三个 v3.5 fresh root 首次完成 38/38 train、proposal、真实 repair、双臂 paired validation、两代 lifecycle 与四份 report/archive；两臂均未 promotion，第一代 recursive 为 0 gain/2 harm，no-recursive 为 1 gain/0 harm但 LCB 仍小于 0。第二代 no-recursive 又被一次 Ruoli 503、circuit skip、9 个 invalid pair 与 8 个 budget mismatch 污染，旧 lifecycle 错把它计作普通 non-promotion。v3.6 已一次性把成功 train rows 变成无 instruction/context 的 negative controls，按 train activation precision / false positives / failure support / complexity 选择候选，并把 invalid counterfactual evidence 归类为 terminal non-claim。其串行 live root 完成 38/38 valid train（7 success/31 residual），选出的 root 为 26/27 train activation precision、1 个 success false positive，并完成 2/16 validation pairs 后主动停止；无 report/archive/promotion/freeze。v3.7 的六路 agent 首批 6/6 收到 429；v3.8 的两路 agent 在 16 valid 后同时收到 2 个 503。两轮均被既有 circuit 正确阻止、无可复用 bundle/report。v3.9 不改 evaluator、learning/promotion、model、split、预算、retry 或 circuit；只把调度改为 6 个 outer item pipeline 共用 1 个在线 agent slot，本地容器准备和离线 verifier 仍可并行，同题 variants 仍串行。该调度现已 clean 完成 full development，负结果把 blocker 定位到 proposal diversity 与 prospective coverage，而不是继续补 gate；旧 rows 不复用。**

### 1.2 结论分层

| 命题 | 当前状态 | 证据层级 |
|---|---|---|
| legacy HLE 是高维手写控制面，学习 policy 没有闭环 | 支持 | 代码审计 + 历史 artifacts |
| v2 的 proposal -> repair -> off/on -> gate -> archive 接口已连通 | 支持 | 当前完整离线 suite + v3.15 clean 57/57-valid live development |
| v2 的内部 runtime action 能改变 lane plan | 支持 | 代码 + 单元测试 |
| v2 主 SkillLearn 路径执行了每个 typed action/verifier/fallback 的强语义 | **不支持，且协议已停止这样声称** | 只接受四类显式 prompt/self-check lowering；其余 fail closed |
| promotion threshold 完全由冻结 protocol 所有 | 支持 | protocol-bound spec + 宽松 candidate 对抗测试 |
| 86-item offline-ready runtime 已预验 | 支持 | readiness/preflight `blockers=[]`；v3.4 与三份 v3.5 v4 cache-only prewarm 均 86/86、model 未执行、sealed scoring=false |
| v2 已产生可保留的 promoted incumbent | **不支持** | v3.15 两份 claim-eligible negative report 均停止于 non-promotion，archive 仍为 `incumbent_id=null` |
| v2 稳定优于 raw 或 budget-matched raw | **不支持** | v3.15 recursive G1/G2 都是 candidate/raw 4/16，三次 on 全部 0 gain/0 harm |
| v3.15 已改善真实 action utility | **不支持；clean live 负结果** | 13 个 candidate audit 中 7 material / 6 restatement-risk；material 仅 exact constant/mapping，且 9 roots 全坍缩为 poster 单-family |
| v3.17 family-slot/artifact-blueprint proposal 已达到 trial-feasible | **不支持；proposal-only 负结果** | 8/9 feasibility 通过，但第三候选绑定 2 个 failed TRAIN primitives；0 benchmark/evaluator call |
| v2 已实现 Red Queen 式多 clade 搜索和 evaluator co-evolution | **不支持** | 目前是单 incumbent；evaluator 路径未接主实验 |

### 1.3 潜力判断

研究问题是连贯且可证伪的。显式 `HypothesisProgram` 可能比整块 workspace mutation
更容易做 lineage、activation 和 off/on attribution；但“更可解释”目前仍是待验证
假设，而不是既成事实。它至少需要以下操作化证据：

- schema fidelity；
- action lowering 成功率；
- lineage completeness；
- prospective activation precision；
- paired gain/harm attribution；
- cross-instance 与 cross-family retention。

在结构重复、可程序验证、能复用操作步骤或约束模式的任务上，超过单次 raw 有现实
潜力；在 broad random HLE 上稳定领先的先验较低，因为知识瓶颈、一次性长尾和 source
availability 会与 policy quality 混杂。HLE 更适合作外部 transfer/stress test，而不应
继续作为唯一开发靶子。

## 二、术语、三种“递归”与证据标签

### 2.1 核心术语

| 术语 | 本文含义 |
|---|---|
| assumption / hypothesis | 可证伪的关系、策略或 evaluator 命题，不等同于任意 prompt 建议 |
| `HypothesisProgram` | trigger、anti-trigger、action graph、expected effect、verifier、fallback、lineage 与 evaluator epoch 的结构化程序 |
| activation | 程序在运行前由可用特征命中，且实际改变 treatment 或 execution plan |
| promotion | 只依据冻结 validation 与预注册 gate，把 candidate 变成未来 runtime incumbent |
| archive node | 一组 active programs、runtime version、evaluator epoch 与证据依赖的完整配置 |
| evaluator epoch | 一个 evaluator、artifact protocol 和 scoring rule 保持不变的时期 |
| selective erasure | evaluator 被替换后，仅使依赖旧 evaluator 的 utility/score records 失效；不是删除失败假设的同义词 |
| clean external evidence | split、provider、预算、runtime、verifier、invalid-row policy 和 protocol lock 都满足预注册约束的外部结果 |

### 2.2 三种“递归”必须分开

1. **同题推理递归**：在一道题内展开 assumption tree 或多轮验证；
2. **假设修复递归**：候选未通过静态/训练检查后，生成有 lineage 的 child；
3. **跨代演化递归**：被 promotion 的程序改变 incumbent，再影响下一代 train residual、
   proposal 和未来题的 runtime。

legacy 主要有第 1 种；v2 已实现第 2 种的机制和第 3 种的 harness，但尚未出现真实
promotion，因此还没有观察到完整的跨代能力积累。RQGM 的核心则是跨任务 archive
tree search，不应被简化成“多调用几次模型”。

### 2.3 证据标签

本文使用以下强度顺序：

- **[CODE]**：源码直接可见的事实；
- **[TEST]**：离线测试验证的 wiring/invariant；
- **[ARTIFACT]**：真实运行留下的报告或 event；
- **[INFERENCE]**：由代码和结果支持、但尚无 controlled ablation 的解释；
- **[PROPOSAL]**：建议或验收标准。

## 三、legacy Assumption Agent 的架构诊断

### 3.1 真实行为链路

```text
Assumption Graph
  -> retrieval / OperatorSpec / morphism
  -> multi-prompt candidate generation
  -> source search / span / comparator / many verifiers
  -> fallback / selector
  -> final answer

final trace
  -> transition dataset
  -> fast-policy miner
  -> candidate/shadow policy
  -X-> did not reliably control the next HLE runtime
```

所以 legacy 中真正产生行为变化的主要是 prompt ensemble、手写 domain rule、source、
verifier、fallback 和 selector，而不是“系统自己提出、验证、保留并复用的假设”。

### 3.2 高维控制面的复杂度证据

**[CODE]** 在审计 revision 上，
[`hle_smoke_eval.py`](../../assumption_os/hle_smoke_eval.py) 共 116,808 行；AST 可见
1,401 个顶层函数定义，包含嵌套定义时为 1,604。源码中出现 770 个唯一的 `HLE_*`
配置名。按顶层函数名统计，126 个含 `verifier`，56 个含 `fallback`。

这些数字是复杂度代理指标，不等于 770 个布尔开关、126 个独立 verifier 或 56 个
独立 fallback 行为。它们能直接证明的是：旧系统有很大的配置面和函数面；“归因困难”
则是由此产生、并与历史反复局部回归相一致的**诊断推断**。若要严格量化，仍需要
调用图、交互覆盖与模块消融。

该控制面带来四个可观察风险：

- 一次提升难以归因到 assumption、source、selector、fallback 或额外预算；
- 局部规则可能覆盖另一个局部规则；
- 同 seed 的模型波动可能被误认为代码改进；
- 每次“下一刀”增加自由度，扩大 adaptive overfitting 与不可复现风险。

### 3.3 同题递归没有形成跨题学习

**[CODE]** HLE 在
[`hle_smoke_eval.py:L2963-L2973`](../../assumption_os/hle_smoke_eval.py#L2963-L2973)
调用
[`build_recursive_assumption_run`](../../assumption_os/recursive_runner.py#L73)
时使用 `writeback=False`。该 runner 会构造可审计的同题 assumption tree，但不会把
一道题中新提出并通过验证的程序写回为下一题可调用的 incumbent。

因此 legacy 的“递归”主要是同题内展开，而不是：

```text
propose hypothesis
  -> validate and repair
  -> estimate benefit, harm, and cost
  -> promote or reject
  -> alter future runtime behavior
```

### 3.4 fast policy 没有 effectful semantics

**[CODE]** 旧代码已有
[`fast_policy_memory.py`](../../assumption_os/fast_policy_memory.py) 和
[`hle_fast_policy_miner.py`](../../assumption_os/hle_fast_policy_miner.py)，但 HLE 主文件
对 [`route_option_lanes`](../../assumption_os/hle_lane_router.py#L105) 的四处调用并未把
完整 `fast_policy_decision` 接成动作控制。router 即使收到 policy，也主要把
`selected_policy_ids` / `selected_actions` 写入 metadata，不会据此启停 candidate、
source、solver、verifier 或 final-selection lane。

因此 policy 当时是可审计的 data object，不是可消融的 behavior program。

### 3.5 miner 学的是故障支持度，不是因果收益

**[CODE]** 旧 miner 在
[`_make_policy`](../../assumption_os/hle_fast_policy_miner.py#L185-L207)
里用 `support_count / wrong_count` 构造 `expected_utility`。这回答的是“某个 failure
bucket 出现得多不多”，而不是：

```text
同一题、同一 evaluator、同一预算下，
policy_on 相比 policy_off 修正了多少题，又伤害了多少题？
```

高频故障可以对应无效修复，也可以对应副作用更大的修复。没有 matched off/on，
故障频率不能被解释为净因果收益。

### 3.6 transition data 可审计，但缺 prospective trigger semantics

**[CODE]** [`hle_transition_dataset.py`](../../assumption_os/hle_transition_dataset.py)
为防止泄漏保存了 hash、label、failure bucket、cost 和 path metadata，这是正确的审计
方向；但当时缺少足够的关系类型、约束结构、输出 schema、可验证条件、候选差异和
反触发条件。数据可以证明“发生过一次 transition”，却很难支持 router 学会“什么
新题应触发哪条 policy”。

## 四、从 self-evolution 文献抽取的项目设计约束

本地材料实际包含
[`21 篇 self-evolution/continual-learning PDF`](../reference/self_evo_continual_20260707/papers/)、
2 个背景页面、21 个相关 repo，以及单独保存的
[`RQGM 论文`](<../reference/The Red Queen Gödel Machine Co-Evolving Agents and Their Evaluators.pdf>)。
下表是面向本项目的机制综合，不表示每篇论文都逐字主张所有约束。

| 文献组 | 可迁移机制 | legacy 缺口 |
|---|---|---|
| SkillLearnBench、Voyager、LifelongAgentBench、Fast/Slow | 经验需要编译成可执行、可复用的 skill/fast state，并在未来任务中调用 | OperatorSpec 最接近，但历史 application evidence 不足 |
| Reflexion、ExpeL、FLEX、EvolveR、AgentEvolver | 成功/失败对照、经验抽象、credit assignment 与生命周期 | 主要从 error bucket 生成修补建议 |
| DSPy、GEPA、TextGrad、OPRO、GPTSwarm | 需要小而明确、可优化和可消融的计算图 | 优化表面是 116k 行隐式分支 |
| MemGPT、MemoryBank、A-MEM、HippoRAG 2 | 记忆组织与检索有价值，但 retrieval 不等于 learning | source coverage 投入没有闭合 policy learning |
| Agent-as-a-Judge、Self-Rewarding LM | 中间轨迹评价有价值，但自评需要外部 anchor 与漂移控制 | 多数 verifier 与被评系统共享模型偏差 |
| AI Agents That Matter | held-out、简单基线、预算匹配、成本与复现性必须同时报告 | harness 逐渐接近，历史主实验仍不完整 |

SkillLearnBench 关于 external feedback 优于纯 self-feedback 的结果，为旧项目的递归漂移
提供了一个与观察一致的候选解释；它不是项目内因果证明，仍需 recursive/no-recursive
和 external/self-feedback controlled ablation。

## 五、与 Red Queen Gödel Machine 的精确对照

两者在“生成候选 -> 固定标准评估 -> 保留有效变体 -> 影响后续搜索”的抽象上同构，
但搜索单位不同：Assumption Agent 试图搜索显式 assumption/operator；RQGM 搜索的是
可修改的多代理 workspace/archive node。不能把 RQGM 说成也在直接搜索显式假设。

### 5.1 archive tree 与 clade search

RQGM 把 self-improvement 表示成增长的 archive tree，并按 clade metaproductivity 进行
Thompson sampling；一个节点是共享的多代理 workspace。见
[`RQGM PDF pp. 4-5`](<../reference/The Red Queen Gödel Machine Co-Evolving Agents and Their Evaluators.pdf#page=4>)。

legacy 更接近同一代码主线上的人工 hill climbing：固定/debug seeds 持续影响代码、
prompt、verifier 和 gate，每次局部胜利都被编译进全局复杂度。

### 5.2 evaluator 在 epoch 内冻结

RQGM 在一个 epoch 内冻结 evaluator、artifact-generation protocol 和 binary scoring rule；
只有 epoch boundary 才允许 utility transition。这使每个 epoch 保持为固定标准的搜索
问题，而不是 agent 与 judge 同时漂移。

### 5.3 产生候选与选择候选的数据隔离

RQGM 明确分开创建 node 的 training feedback、驱动 node selection 的 validation，以及
最终 held-out test。训练结果可以指导 meta-agent 修改 node，但不能直接构成 search
utility。见 [`RQGM PDF p. 5`](<../reference/The Red Queen Gödel Machine Co-Evolving Agents and Their Evaluators.pdf#page=5>)。

legacy 中反复使用的 HLE fixed/debug seeds 只能继续作为 regression cohort，不能再作为
泛化证据。

### 5.4 anchor lower bound 与 selective erasure

RQGM challenger evaluator 在固定 ground-truth anchor 上按保守 best-belief/lower-bound
标准与 incumbent 比较。替换 evaluator 后，只清除依赖旧 evaluator slot 的 utility
records，并按需重排 archive；无关信息保留。见
[`RQGM PDF pp. 5-6`](<../reference/The Red Queen Gödel Machine Co-Evolving Agents and Their Evaluators.pdf#page=5>)。

因此：

- 失败 hypothesis 的降级、停用或归档不是 selective erasure；
- selective erasure 只描述 evaluator/utility dependency invalidation。

### 5.5 对 RQGM 性能主张保持边界

RQGM 是 preliminary empirical investigation；主要结果依赖强模型，理论保证是
epoch-local，anchor 偏差仍可能导致 evaluator 漂移。适合借鉴的是 archive、数据隔离、
冻结标准、anchor promotion 和 dependency-aware invalidation，而不是直接继承其性能
或开放式自我改进主张。

## 六、legacy HLE 效果证据 ledger

下表把点估计与 validity 同时报告。它不是 benchmark 排名表，而是说明为什么历史
结果不能支持“Assumption Agent 优于 raw”的主结论。

| 证据 | 点估计 | validity 与归因边界 |
|---|---:|---|
| [`fixed/debug agent-only 12`](../../reconstruction/md/hle_candidateconflictresolve_baseline12_agentonly_mini_20260629.md) | agent 6/12 | `pass=False`、paper-clean false、无同题 raw/Hippo、无 budget controls、operator application verifier 0 |
| [`所谓 unseen12 agent-only`](../../reconstruction/md/hle_docrank_mathbinding_current_unseen12_cacheonly_mini_20260708.md) | agent 5/12 | paper-clean false、无 controls；operator selected/activated 12/12，但 application verifier 0，不能归因于 operator/assumption |
| [`fresh triad promotion report`](../../reconstruction/md/hle_parallelrun_unseen_mc12_fair_policy_promotion_20260707.md) | agent/raw/HippoRAG 均 2/12 | 12 个 triad 齐全，但 promotion `pass=False`；有 control errors、无 selector gain、低于 24-triad gate，缺 budget-matched controls |
| [`controls-only 12`](../../reconstruction/md/hle_freshunseen12_controls_multiglob_fair_cacheonly_mini_20260709.md) | raw 2/12、Hippo 3/12、budget raw 4/12、budget Hippo 3/12 | 4 个不对称 endpoint errors，无 agent；clean-shared n=10 分别为 2/10、3/10、4/10、3/10 |

本表没有把 fixed cohort 的后续更高点估计当成反例隐藏掉：同一 cohort 后续经过更多
adaptive debugging 后出现过更高分，但这只提高 regression value，不提高 generalization
evidence。固定 cohort 越被反复用于决策，越不能承担 sealed claim。

历史证据只能支持：

- 系统有工程性能力和若干有效局部模块；
- 某些改动能改变单题或固定 cohort 的行为；
- 尚无可靠证据证明 agent 稳定优于 raw、HippoRAG 或 budget-matched raw；
- 尚无 attribution 证明正确题来自可迁移 Assumption 机制。

## 七、legacy 缺口到 reconstruction_v2 的 closure delta

### 7.1 已明显改善的部分

| legacy 缺口 | v2 状态 | 证据 | 尚缺 |
|---|---|---|---|
| assumption 没有统一可执行 schema | 已实现三类 `HypothesisProgram` | [`models.py:L221-L275`](../assumption_agent/models.py#L221-L275) | 外部 backend 的 typed lowering |
| policy 不改变 runtime | 内部 `PolicyRuntime` 可启停、排序 lane、设参数和执行 operator step | [`runtime.py:L72-L226`](../assumption_agent/runtime.py#L72-L226) | 主 SkillLearn 路径仍主要是 skill 注入 |
| 无 hypothesis repair lineage | 已实现 failed-check -> child repair tree | [`validation.py`](../assumption_agent/validation.py) | empirical repair benefit |
| utility 来自 failure frequency | promotion 已使用 protocol-owned paired gain/harm/cost/LCB，candidate 只能收紧 | [`evaluation.py`](../assumption_agent/evaluation.py) | 尚缺真实 promotion 与 retained gain |
| train/validation/test 混用 | split guard 与 archive-freeze gate 已实现 | [`splits.py:L220-L267`](../assumption_agent/splits.py#L220-L267) | 一次完整 current-protocol sealed run |
| evaluator 变更无依赖失效 | controller/anchor lower bound/selective invalidation 已实现 | [`archive.py:L291-L370`](../assumption_agent/archive.py#L291-L370) | 尚未接入主 evolution 或真实 challenger |
| HLE 是唯一主战场 | 已转向 86-item offline-ready SkillLearnBench instance-out/family-out | [`BENCHMARK_PROTOCOL.md`](../BENCHMARK_PROTOCOL.md) | v3.15 已完成 clean live development，但 action/search 坍缩为 poster 单-family、仍无 incumbent；尚缺真实 freeze、完整 controls 与 family-out |

### 7.2 当前证据到哪一层

**[TEST]** v3.13 的 `reconstruction_v2` 离线 suite 为 **375/375 通过**，v3.14 为
**411/411 通过**；v3.15 提交 `696a2954` 当时为 **453/453 通过**；加入 v3.16/v3.17 formation、
proposal-only boundary、typed representation 与 single-decision binding 后，当前完整 suite 为 **540/540 通过**。新增覆盖包括 TRAIN-only
action profile 的 containment/allowlist/secret isolation、request-local action-quality prompt、audit-only
不改变 retry/selection/promotion、terminal-invalid memo 的 retry identity 与零执行 replay，以及
report/freeze 的首代 checkpoint/profile provenance。此前 shared immutable valid baseline cohort、
legacy replay compatibility 和 family-count/support tie-break 覆盖仍保留。这些测试证明 schema、wiring、guard、
replay、failure handling 和若干 invariant；不证明真实 benchmark improvement。既有覆盖还包括
protocol threshold ownership、candidate 宽松阈值攻击、backend action lowering v2、exact-three
cardinality 与 audit-only signature diversity、真实/声明 fallback 分离、offline-ready split 不重抽样，
以及离线 verifier receipt 必须绑定 proxy 实际执行的 frozen runtime profile/command。

**[ARTIFACT]** 在第三个 v3.5 fresh-root 运行前，对 `reconstruction_v2/artifacts` 中当时
可读的 v1/v2/v3 smoke、diagnostics 和 development runs 做混合扫描得到：

- 23 份 `*.archive.json`；
- 22 份 `*.report.json`；
- 非空 incumbent：0；
- 这些 report 中 `promoted=true`：0。

这 23/22 不是 23 次 current-protocol 独立实验，也不能作为样本量；它只是 available
artifact tree 的状态审计。结果不是说 gate “失败”；恰恰说明现存 artifacts 没有把
诊断信号包装成 incumbent。但它也意味着系统尚未完成“promoted program 改变下一代
runtime”的实证闭环。

**[ARTIFACT]** 曾有一次 full replay-locked development 出现 raw 4/18、candidate 7/18、
3 gain/0 harm、cost ratio 0.914，但一条 baseline trial 无效，gate 正确拒绝；该结果只能
视为 promising but inadmissible diagnostic，见
[`STATUS.md:L96`](../STATUS.md#L96)。后续 pre-network-hardening 的 685a run 第一代是
raw 4/18、candidate 5/18、2 gain/1 harm，LCB 为负并被拒绝；第二代未完整收束，见
[`development_recursive.events.jsonl`](../artifacts/paper_primary_v3_ruoli_gpt54mini/runs/685a4482_full_development_20260711/development_recursive.events.jsonl)。
这些结果都不能形成性能主张。

**[ARTIFACT]** clean commit `e07913f9` 上当时的 v3.1 protocol smoke 已完成机制验收：
两臂均为 2 个有效 pair、0 invalid、0 provider/budget mismatch，且 behavior-identical
validation 被精确 replay；两臂 candidate/raw 都是 0/2，因此没有 promotion。它只证明
运输、lowering、paired replay 和 fail-closed promotion 能协同工作，不是性能证据，见
[`smoke_recursive.report.json`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/smoke_recursive.report.json)
和
[`smoke_no_recursive.report.json`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/smoke_no_recursive.report.json)。

随后第一次 v3.1 full development 在完整 38-item train 上严格中止，见
[`development_recursive.events.jsonl`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/development_recursive.events.jsonl)：

- 26 个本地 verifier 有效 observation，其中 9 pass、17 fail；
- 2 个已启动 trial 收到 `429 Too Many Requests`，provider circuit 随即打开；
- 9 个尚未启动的 trial 按同一 circuit 在本地跳过，没有继续消耗模型请求；
- `court-form-filling-6` 的长轨迹累计模型流量为 33,730,000 bytes，超过冻结的
  33,554,432-byte fuse，作为 hard-budget invalid 处理；
- 因 12/38 training observations 无效，training evidence 没有写入 replay cache，
  proposal、repair、validation、archive 和 promotion 均未执行；report/archive 也没有落盘；
- sealed split 保持未访问。

这次失败没有调用 online evaluator：task payload 与 verifier 均来自冻结的本地
SkillLearnBench checkout，evaluation 仍由 post-agent offline verifier 完成；唯一在线流量
是预注册的 agent model inference。因而缺口不是“再下载一个 evaluator”或“再补一个
readiness gate”，而是恢复预注册 provider transport 后取得一份完整、0-invalid 的
development evidence。当前进程内 training replay 也不能跨失败进程复用这 26 条有效结果，
所以它们只能作为 transport diagnostic，不能与后续 run 拼接成 claim。

全 run 退出后的单题、5-step、非 claim transport canary 已在同一 provider route 上恢复：
模型请求完成、offline verifier 正常执行，observation 为 `evaluation_valid=1`、
`task_success=0`。这说明 429 已冷却；任务失败不等于 transport 失败。canary 没有读取
validation/sealed，也不进入任何性能汇总，见
[`transport recovery canary`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/transport_recovery_canary/report.json)。

29 个实际启动 trial 的 network receipt 显示总流量中位数约 2.40 MB、p90 约 5.53 MB、
p95 约 22.03 MB；唯一超过 32 MiB 的就是被右删失的 `court-form-filling-6`，且只超出
175,568 bytes（0.52%）。因此 v3.1 没有因单个 train diagnostic 原地抬 cap，而获得最多
一次同协议、全新 run-root 的 clean rerun。该 rerun 在同一 item 上再次触发 hard cap，
这次观测到 38,599,999 bytes；进程在 stop condition 已不可逆后主动中止，没有继续烧完
余下 train。v3.1 因此正式判为 execution-infeasible，而不是继续重跑到碰巧通过。

唯一允许的资源修订已版本化为
[`v3.2 protocol`](../manifests/skilllearn_paper_protocol_v3_2_ruoli_gpt54mini.json)：
统一 provider-only fuse 从 32 MiB 提高到 64 MiB；model、86-item subset、offline evaluator、
dependency policy、4 workers、search budget 和 promotion contract 均不变。64 MiB 是
train-only 已观察最大流量向上取下一 2 的幂，不读取 validation/test，也不复用 v3.1
observations。后续不再允许第二次按题调 cap。

v3.2 在 clean commit `748469b2` 上重新得到 claim-eligible lock 和 86/86 cache-only
prewarm，随后 full development 的 64 MiB budget 没有触发；`court-form-filling-6` 本轮以
4.70 MB 正常完成并通过 verifier-validity 检查。然而同一路由在 8 个有效 train
observation（1 pass、7 fail）后发生独立 transport outage：一个 trial 明确以 429 终止，
其余三个在途 trial 的原始 Codex JSONL 先连续报告 `gpt-5.4-mini` 没有可用 distributor
channel（503），随后也以 turn failure/429 收尾；provider circuit 打开后，29 个尚未启动
请求被本地跳过。training evidence 因 30/38 invalid 再次 fail-closed，未进入 proposal、
validation 或 sealed，见
[`v3.2 development events`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini/development_recursive.events.jsonl)。

随后使用用户提供、仅注入进程且未落盘的新 GPT Pro credential 对同一
`https://ruoli.dev` / `gpt-5.4-mini` route 做最小 canary：models、Responses、chat
端点均为 HTTP 200，真实 Codex container 也完成 agent -> offline verifier，证明
`ruoli` 本身并非永久不可用。这里没有切 model/provider contract，也没有使用 online
evaluator。

第一个新渠道 full root `gptpro01` 完成 38/38 train：36 valid、12 success、2 invalid；
provider/circuit 与 64 MiB cap 错误均为 0，最大 trial 流量 62.9 MB。两个 invalid 都来自
`temperature-simulation-2/3` 的确定性 receipt false-negative：proxy 实际执行的离线
profile 已生成完整 CTRF（各 7 tests、5 pass/2 fail、reward 0），但旧 auditor 错误检查
了未被执行的 upstream `test.sh` 是否含 `--ctrf`，把本应是“有效任务失败”的结果标成
`verifier_execution_receipt_unsupported`。这不是增加 gate 的理由，而是 evidence source
绑定错误。

提交 `e43670f6`、`18ff3417` 之后，receipt 直接绑定 proxy 实际执行的 frozen runtime
profile ID/hash/command hash；profile-backed CTRF 缺失或畸形仍 fail closed，reward 0
仍是 valid failure。该历史提交当时的全套 136/136 tests 通过。最终 clean root `gptpro03` 的 lock 绑定
`18ff3417` 且 claim eligible，prewarm 再次为 86/86、0 failed、无 online build。
真实 run 中两项 temperature receipt 均成为 `pytest_ctrf`、`test_count=7`、valid=true，
证明修复生效而未把失败改成成功。

`gptpro03` 最终 38/38 train 返回：37 valid、9 success、1 invalid；provider error 为 0。
唯一 invalid 是 `video-object-counting-1` 的真实 hard-cap：TX 66.5 MB、RX 4.6 MB、总计
71.1 MB，超过冻结 67,108,864-byte limit，容器被监控终止且禁止 retry。因
`all_valid_before_proposal_v1`，proposal/generation/validation/no-recursive/promotion 均为 0，
report/archive 未生成；sealed/test content 仍未访问，见
[`gptpro03 development events`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini_gptpro03/development_recursive.events.jsonl)。

该 71.1 MB 全部发生在只允许模型端点的 egress 域内。trace 只有 23 次本地命令和 10 次
agent message，可见 shell 输出合计约 13 KB，没有 web/image、包安装或视频上传命令；
15.4 MB 本地视频只被 ffmpeg/OpenCV 读取。同族 `video-object-counting-3` 用 16 次命令、
约 8.79 MB 即完成。因此最可信解释不是依赖缺失或数据外泄，而是 Responses 多轮工具
往返反复发送累计 context/envelope，叠加数次无效能力探测和调试循环，造成 TX 放大。

据此，**v3.2 也正式判为 execution-infeasible**。本项目不会第三次抬 cap、删除该题、
降低验收标准或重跑到碰巧通过。本地 Codex 0.144.1 审计确认请求压缩已经默认开启；
`previous_response_id` 没有公开 ConfigToml 开关，固定 mini route 的最小
`/responses/compact` canary 又返回 503 `model_not_found`。可用且必须 protocol-owned 的
节流手段是 `model_reasoning_effort=low`、更早的本地 auto-compaction，以及低 verbosity；
tool-output cap 对本题约 13 KB 可见输出预计是低杠杆。

下一工作流只允许改进 model transport/trajectory 效率：统一冻结上述配置，并把视频类
轨迹收敛为“一次能力探测、一个整合脚本、一次最终校验”；先在 train-only 长轨迹
canary 上证明仍低于现有 64 MiB，再冻结一次新的 execution-policy revision。
promotion/subset/evaluator/cap 不变，也不拼接任何失败 run 的有效 observation。

该工作流已由 v3.3 完整执行。提交 `e0b1a33b` 新增独立
`codex_low_reasoning_early_local_compaction_v1` policy；v3.1/v3.2 仍解析为旧 catalog-default
treatment，不会被静默套用新配置。v3.3 与 v3.2 删除 protocol ID/version 和这一 policy
字段后逐项相同：64 MiB、86-item subset、4 workers、search/promotion/evaluator/sealed
合同均未改变。该 v3.3 提交当时的全套 150/150 tests 及 Codex 0.144.1 `--strict-config` 的断网解析通过。
claim lock 绑定 clean commit `e0b1a33b`、policy hash 和 67,108,864-byte cap，
`claim_eligible=true`；cache-only prewarm 为 86/86、47 images、7 verifier runtimes，
`online_build_attempted=false`。其中历史字段 `test_content_accessed=false` 表示未执行/评分
test split，也未向模型暴露 test bytes；prewarm 的 infrastructure 路径实际会读取并哈希
test task/image/verifier 文件，v3.4 receipt 已改为显式记录这一区别。

train-only `video-object-counting-1` canary 先得到 valid task failure：总流量 1.47 MB、
`error_type=null`、本地 `common-pytest-ctrf-py312-v1` verifier、0 provider/cap/sealed event。
随后同一 lock/prewarm 的 full development 返回 38/38 train：37 valid、9 success、1 invalid；
38 个 verifier execution receipt 全部 valid，37 个 model-only audit valid、1 个 violated；
38 个 network monitor 均 finalized，最大为 `temperature-simulation-3` 的 40.6 MB，
`video-object-counting-1` 为 19.69 MB，provider/circuit 和 hard-cap error 都为 0。因此
**本次 v3.3 batch 已排除 fuse 作为未进入 proposal 的直接原因**。这不等于跨运行稳定性
已经成立：同一 video task 在 canary/full 中为 1.47/19.69 MB，后续仍须报告这一波动。

唯一 invalid 是 `offer-letter-generator-1`。其 trace 明确包含
`item.started(type=web_search)` 和 `item.completed(type=web_search, query=placeholder)`；
auditor 记录 `remote_tool_call_count=1` 并正确产生
`model_remote_tool_policy_violation`。该 trial 已同时使用 `--ignore-user-config`、
`tools.web_search=false`、disabled `standalone_web_search` 和禁止联网工具的 developer
instruction；容器 egress 又只允许模型端点。故它不是 online evaluator，也不是 benchmark
dependency 下载，而是 Codex/Responses execution boundary 出现了禁止的远程工具 item；
现有证据还不能区分该 item 源于 Codex CLI mapping、provider 还是 model。
按现有合同这是不可重试的真实 invalid，不应通过删 trace、白名单 placeholder 或重采样
改写成普通任务失败。`all_valid_before_proposal_v1` 随即阻止 residual/proposal；0 proposal、
0 counterfactual、0 sealed event，recursive/no-recursive report/archive 均未生成。

本轮还暴露一个与评分 gate 无关的执行预算缺口：`max_steps=100` 目前只参与 request/backend
锁定与一致性校验，上游 Codex run template 没有把它传给 `codex exec` 作为可执行限制。
有效的 `temperature-simulation-3` 运行 2,485 秒、累计约 334 万 token，并产生 241 行 JSONL；
其中只有 93 行是 `item.started`，且它们不能直接等同于 protocol 的 semantic step 或 turn，
所以本 run **不能**证明越过了 100-step/turn cap。它证明的是该 cap 在当前执行链中没有
可审计的 enforcement。该结果本身不授权直接启动新 paper run 或添加 ad-hoc gate；若要
声称 budget-matched，必须先定义并离线验证一个可观测的 action budget。以下 v3.4
revision 正是在完成该非评分诊断后建立，而不是对 v3.3 invalid 的重试。

#### v3.4 零模型定位与最小执行修复

上述两个来源不明的问题现已在不调用模型、不评分、不访问 sealed 的条件下定位并实现
最小修复。

首先，精确审阅 Codex 0.144.1 源码与 loopback 出站请求后确认：旧配置
`tools.web_search=false` 是兼容性 no-op；布尔 `false` 被解析成空值，随后未设置的顶层
`web_search` 回落到 `cached`。自定义 Responses provider 支持 hosted web search，因此
v3.3 的 38 个请求都曾把该工具暴露给模型，不能把其中 37 个“未调用” observation 复用为
合规证据。根因在本地 CLI 配置语义，不是 Ruoli 注入，也无需换 provider、升级 Codex、
重试样本或扩充评分 gate。

v3.4 只合并两项 execution-treatment 修复：使用顶层 `web_search="disabled"`，以及把
`max_steps` 落为 `codex_action_start_v1`。独立的
[`codex_model_only_wire_probe_v1`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/diagnostics/codex_model_only_wire.json)
直接复用实际 provider argv，并用本地 canned Responses server 做阴性/阳性对照：

- canonical v3.4：`POST /v1/responses` 一次，`stream=true`，7 个工具，0 个
  `web_search*`/`web.run`，Codex 正常产生 `turn.completed`；
- stale boolean 对照：同一配置只把 canonical 顶层键换回 `tools.web_search=false`，8 个
  工具中出现 `web_search` 且 `external_web_access=true`；
- 两次请求都只到 loopback，模型推理与评分均为 0；raw request、instruction、Authorization
  和 raw trace 均不落盘。

预算单位不是模糊的 turn：每一条可解析的 `item.started` 都计数，缺失 item/id/type 的
malformed start 也保守占一单位并使 evidence invalid，不能靠畸形事件绕过上限。容器内
Node supervisor 为 Codex 建独立进程组；第 N 个 start 先发 TERM，15 秒为 KILL 的上限。
它直接持有并截断当前 attempt 的 trace，写入随机 nonce，并在 verifier 开始前扫描专用 trial
容器的 `/proc/<tgid>/task/<tid>`，清除基线之后新增的全部 live task；因此 `setsid`/new
session，以及 thread-group leader 已为 zombie 但 worker 仍存活的情况也不能逃逸。任何无法
完成的 task scan 都 fail closed。receipt
绑定 supervisor hash、nonce、full trace hash、action projection hash、limit、实际 steps、
spawn/exit/signal、post-trigger count、严格 token usage、process-group 与 container-scope
confirmation。离线注入已覆盖正常完成、无/空 usage 的 `turn.completed`、exit 42、spawn 失败、
同 chunk N+1、旧 receipt 替换当前 trace、background descendant、`setsid` escape、zombie
leader + live pthread worker，以及忽略 TERM 后的 KILL。自然完成若遗留 task 会清理并
invalid；受控 budget truncation 在所有 live task 清空后可保持 valid。完整 trace 中较晚的
429/鉴权/限额等 fatal provider 事件优先于较早的 generic stream error，避免被误分为普通
receipt invalid 并继续请求 provider。

截断通常没有完整 token usage。为避免 promotion cost ratio 把一臂的 token 与另一臂的
action count 静默混算，v3.4 明确对所有 arms 统一使用 action starts 作为 promotion cost；
token usage 继续作为二级报告指标，并逐 trial 持久化 completeness/truncation。64 MiB cap、
subset、workers、retries、evaluator、promotion、statistics、recursive/no-recursive 定义和
sealed policy 均不变。clean commit/lock、新 runtime cache 与 86/86 v4 cache-only prewarm
已经完成。v4 canary 实际到达模型和 verifier，但 root-owned `0600` trace/receipt 使宿主 auditor
无法读取；`1df3092a` / `ad66d5a2` 将这两类 immutable artifact 显式创建为 `0644`，同时保留
temp-file + rename 原子写入、内容 hash 和 nonce 绑定。对应离线/Docker 回归直接断言生产模式，
不再靠测试侧 `chmod` 掩盖回归。

随后同一路由 `max_steps=2` non-claim/train-only v5 canary 通过：observation
`evaluation_valid=1`、`task_success=0`、`steps=2`、`truncated=true`，action receipt、process
cleanup、model-only tool audit 与 post-agent local verifier 均 valid，0 remote/web tool、0 runtime
install。任务未通过不等于执行机制无效；这个结果满足了启动一次 fresh-root development 的
预设条件。

fresh development 最终覆盖 38 个 train request outcome：17 条 valid（3 success）、4 条已启动
trial 因 `provider_rate_limit` invalid、熔断器打开一次后 17 条在本地跳过。21 个实际启动 trial
均完成 network finalize，最大 47.2 MB、0 hard-cap；17 条完成推理的 action/tool/verifier audit
全部 valid。运行以 `all_valid_before_proposal_v1` fail closed，proposal、paired validation、
recursive/no-recursive report/archive 与 sealed event 均为 0。失败运行中的 17 条 valid observation
不能跨进程拼接。切换 online evaluator 对 model inference 429 没有帮助；下一次运行前需要解决
冻结四并发与 provider 持续容量的矛盾，而不是继续增加评分 gate。

v3.5 随后只把五个在线 phase 的 `parallel_workers` 从 4 改为 1。新 clean lock 与 fresh
86/86 prewarm 均通过；serial train 在约 99 分钟内完成 38/38 valid、9 success，38/38
action/tool/offline-verifier audit 有效，0 provider/circuit、0 cap，最大 finalized traffic
60.28 MB。一次 root proposal 在同 request hash 的 `RemoteDisconnected` 后重试成功并返回
3 个候选；两个静态通过，第三个进入两层 repair。depth 1/2 repair 的 model-declared ID 相同，
但 payload hash 分别不同，archive 因而正确 fail closed。碰撞发生在 validation-split trial 前，
所以仍无 paired counterfactual、promotion、recursive/no-recursive report/archive 或 sealed event。

根因不是 archive gate，而是 repair 把模型字符串当成全局主键。修复后的
`parent_content_scoped_repair_id_v1` 忽略模型 ID/status，使用 parent ID、去除可变 status 的规范化
parent-content hash、repair depth 与去掉 ID 的 canonical candidate child 派生 `repair_<sha256>`；
repair 一律由 harness 以 `candidate` 身份进入既有生命周期，事件记录 policy/hash 与被弃 model ID 的 hash。
同 parent/content replay 保持确定，跨 root 和跨 depth 不再 alias；archive 对真正的同 ID 异内容
仍抛错。离线回归覆盖 sibling roots、此次 depth-1/depth-2 复现和 archive 阴性对照。旧进程已死，
JSONL 不是 checkpoint，38 条 observation 不能拼入修复后运行；必须新 lock/prewarm/root。

repair identity 修复后的 fresh root `repairid01` 绑定提交 `96d53a5d`，新 lock 与 86/86
cache-only prewarm 通过。38 个 serial train 再次全部 valid，其中 9 success；trial duration
合计 5,188.542 秒、最长 574.710 秒，38/38 action/tool/offline-verifier audit 有效，38/38
network monitor finalized、0 超限，最大 22.82 MB。训练后形成 29 residual，root proposal
transport 成功并返回 3 个程序，paired checkpoint 冻结。recursive arm 只验证了第一个 root：
`training_support=false` 且 `runtime_action=false`。repair request 的 HTTP、JSON-object parse 和
provider selection 均成功，但 parsed response 没有 mapping-valued `hypothesis`；事件只保存
response hash，raw response 未落盘，因此不能进一步声称具体错误 envelope。旧 proposer 在
`_complete()` 返回后抛裸 `ValueError`，validator 只捕获 typed `HypothesisProposalCallError`，
进程遂在 `hypothesis_repair_proposed` 前 exit 1。16 个 validation ID 仅被 authorize，实际
counterfactual、promotion、archive、generation complete、report/archive 与 sealed event 均为 0。

提交 `d70562de` 在模型响应语义边界做最小修复。root 的所有 consumed rows 先原子化
canonical parse，成功后才 emit/replay；repair 的 envelope/canonical parse failure 进入同一 typed
candidate-local 通道。事件只写 request/response/key-set hash、字段 presence/type/count 与 phase，
不写 raw。validator 不捕获任意 `ValueError`，所以 archive collision/harness invariant 仍 fail loud。
一个 malformed repair 只终止该 branch，其余 root 继续 static audit，但整代不执行 held-out
validation 或 promotion；malformed root 为两臂保留 terminal non-claim report。report 的
failure count/presence/claim/blockers 从 generation rows 派生，paper freeze 再独立重算，诚实
failure 或 top-level 篡改都被拒绝。该修复落实既有 failure policy，没有改变评分阈值或搜索预算。

response-contract 修复后的 fresh root
`paper_primary_v3_5_offline86_ruoli_gpt54mini_repaircontract01` 绑定 clean commit
`4f772e38`，86/86 cache-only prewarm 再次通过。单 worker train 在同一 invocation 内完成
38/38 valid、5 success，trial duration 合计 6,472.468 秒、最长 691.144 秒；38/38
action/tool/offline-verifier audit 有效，network 全部 finalized、0 hard-cap，最大约 50.2 MB。
root proposal 返回两个候选；其中一个静态通过，另一个 depth-1 repair 成功并以新的
content-scoped ID 注册。训练侧仍按最大 failure support 选择 repair child，因此这次真实经过了
前两次运行未能到达的 paired validation、promotion decision、双臂多代停止与 report/archive
落盘，repair identity 与 response-contract 修复均在生产路径上得到验证。

第一代给出了目前最有信息量的因果对照。recursive arm 选择 29-support repair child，16/16
validation pair 全部 valid，但 activation 13、gain 0、harm 2、mean effect -0.125、90% lower
bound -0.2344，因净收益、harm rate 与 lower bound 三项被拒绝。no-recursive arm 在相同 train
observations、residuals 和 roots 上选择 14-support 的 `artifact-completion-guard-v1`；其 16/16
pair 同样全部 valid，activation 6、gain 1、harm 0、mean effect +0.0625、lower bound
-0.0176，只因效应下界未越过 0 而拒绝。这个结果不能算 promotion，但它直接暴露了当前
selection bias：最大化 failure support 的 repair 覆盖更广却造成伤害，较保守的 root 反而产生
唯一净收益且没有 observed harm。递归本身尚未证明有益；成功训练样本没有进入候选排序，
anti-trigger 的所谓 support 也只来自失败 residual，不能表示对已成功区域的保护。

第二代进一步区分了 architecture failure 与 infrastructure failure。recursive arm 的一个新 root
连续两层 repair 后仍未通过 static checks，因此以 `recursive_validation_rejected` 正常结束，未消费
held-out task trial。no-recursive arm 的新候选通过 static checks，并复用同臂 baseline evidence；
前 4 个实际 candidate trial 正常完成，随后 Ruoli 对一个 policy-on request 返回 HTTP 503，分类为
`provider_model_unavailable`。run-scoped circuit 打开后又跳过 8 个 candidate trial，最终该代
9/16 pair invalid、8 个 budget mismatch。promotion gate 正确以 invalid evidence 等 blocker 拒绝，
但 lifecycle 把这一代错误计成第二次普通 non-promotion：top-level report 仍写
`performance_claim_eligible=true`、`evolution_stop_reason=consecutive_non_promotion_limit`，对应
archive score 也仍为 `valid=true`。这不是 hash 损坏，而是 evidence/claim 语义缺口；provider
故障不应消耗科学性的 non-promotion 次数，也不能产生 valid score。

本轮主事件 2,397 行与 prewarm 340 行的 payload hash/event ID 重算错误均为 0；两份 archive
内部 hash、report 引用、node/score 引用均一致，8 个 secret-like 环境值对 574 个 artifact 文件的
exact-literal 扫描为 0 命中，609 次 split access 为 497 train、112 validation、0 test。两臂
`incumbent_id` 都为 `null`，sealed/test true 为 0。因此本轮应作为“机械闭环成立但 learning 未达到、
且第二代被 provider 污染”的负结果收口，不 freeze 空 control，不运行 validation-controls 或 sealed。
下一版不是放宽 evaluator 或继续增加评分 gate，而是一次性版本化为 v3.6：成功 train rows 作为
anti-trigger negative controls；候选只用 train evidence 按 activation precision、success false-positive、
failure support 与复杂度排序；invalid counterfactual evidence 以 terminal non-claim 停止而不增加
non-promotion counter。model、single worker、split、预算、offline evaluator 与 promotion mapping
保持不变，旧 v3.5 rows 不跨协议复用。

**[ARTIFACT]** 2026-07-13 的 v3.9 fresh root
`paper_primary_v3_9_offline86_ruoli_gpt54mini_outer6_model1_plus01` 首先通过 clean lock 和
86/86 cache-only prewarm，随后完整写出 recursive/no-recursive 两组 report/archive。事件账本记录
56 次实际外部 trial、56 次 model-slot acquire/release、观测最大在线 agent 并发为 1；56 次
trial 全部完成，provider failure、circuit open/skip、infrastructure failure、network-budget failure、
invalid pair、provider mismatch 和 budget mismatch 均为 0。唯一在线环节仍是同一 Ruoli 模型
推理；task 数据与 verifier 均为本地，verifier 在 agent 退出后以 `--network none` 执行；sealed/test
访问保持 false。

共享 train evidence 为 38/38 valid、10 success、28 residual。recursive 第一代 root 的初始
family scope 因 anti-scope support 静态失败，depth-1 repair 通过；它在 train 上只激活 2 个 failure、
0 个 success control。held-out 16 pairs 全部有效，但 prospective activation 只有 1/16，candidate/raw
均为 3/16，gain 0、harm 0、effect lower bound 0、cost ratio 1.038462。第二代未修复 root 也只在
held-out 激活 1/16，仍是 3/16 对 3/16、0 gain/0 harm，cost ratio 1.052885。两代均被原冻结
promotion contract 以 zero net gain、6.25% activation 和 zero effect lower bound 拒绝。no-recursive
两代各返回一个 root，但都未通过 train-only static audit，因而没有消费 held-out treatment trial。
两臂最终均为 `consecutive_non_promotion_limit`、`incumbent_id=null`。

这是首份 clean、完整、可解释的 current-protocol development 结果，也是负学习结果。它排除了
“当前只差 API 恢复”与“离线 evaluator 不可用”这两个解释；真正瓶颈前移到了 candidate search：
协议允许每代最多 3 个 proposal，但该 run 每代只返回 1 个；通过静态审计的两个 recursive
candidate 又都收缩到 2/38 train support 和 1/16 prospective activation。继续放宽 promotion gate、
重复同一 root，或冻结空 archive 都不会接近目标。下一次架构工作应限定在 gate 之前：让 proposer
稳定给出多样化 roots，并把多个高精度、低覆盖的局部程序组成一个可审计 candidate configuration，
或用不读取 validation outcome 的 train-only coverage objective 选择可达到 prospective coverage 的
候选。evaluator、split、预算、sealed policy 与现有 promotion thresholds 不应随之改变。

**[ARTIFACT]** 2026-07-13 的 v3.10 fresh root
`paper_primary_v3_10_offline86_ruoli_gpt54mini_outer6_model1_diverse_plus01` 通过 clean lock 与
86/86 cache-only prewarm，随后完成 56/56 actual trials。事件账本为 56 start / 56 complete、
56 slot wait/acquire/release，最大在线 agent active=1；38/38 train 与 16/16 paired validation 全部
evaluation-valid，provider、circuit、infra、timeout、action/network budget、provider mismatch、
budget mismatch 和 invalid pair 都为 0。train 为 6 success / 32 residual。

第一代一次返回 exact 3 且 3/3 静态通过：poster root 覆盖 1 个 family、2/2 failure；template root
覆盖 2 个 family、3/6 precision；retrieval root 覆盖 2 个 family、6/6 precision，coverage-first
正确选择后者。它在 held-out 激活 2/16，candidate/raw 都为 3/16、0 gain、0 harm、effect/LCB=0、
cost ratio 0.948207，仍被既有 `insufficient_net_gain_count` 与
`paired_effect_lower_bound_below_target` 拒绝。第二代 recursive/no-recursive 各自收到一个 transport、
JSON、exact-count 均成功的三候选 response，但 host 事后计算发现每批三项只有一个 distinct
failed-train activation signature；旧 v3.10 semantic response contract 将两批都原子拒绝。两份 report
因此各有一次 proposal failure、`performance_claim_eligible=false`，两份 archive 字节相同、
`incumbent_id=null`；sealed/test 访问仍为 0。

更关键的非评分诊断推翻了“treatment 没执行”的表面解释。两个激活题上 policy-on/off 的 Codex
command trace 和实际答案均不同；financial candidate 还新增了 PUT/CALL 过滤，但 task success 仍为
0→0。当前 `selection_change_count` 比较的是 `selected_result.answer`，而 SkillLearn 把该字段投影为
`observation.success` 布尔，所以 0 只表示 pass/fail 未翻转，不能表示轨迹相同。compiler lowering v1
又在 `execute_step.value` 非空时丢弃 target，在 `check_condition` 也只保留 value；实际 agent skill
因此出现仅含 `{"mode": "evidence_join_then_compute"}` 的含糊步骤。与此同时训练失败 feedback
硬编码“explicit completion check”，直接把 proposal 引向表面完整性而非任务 operator。这些证据将
blocker 从 coverage/bundle 前移到 action representation；v3.11 先一次性修这一层，不新增评分 gate。

**[ARTIFACT]** 2026-07-13 的 v3.11 fresh root
`paper_primary_v3_11_offline86_ruoli_gpt54mini_outer6_model1_actionable_plus01` 绑定 scoped-clean
commit `a0ca50d8`，通过 86/86 cache-only prewarm（47 images / 7 verifier runtimes）。共享 train
执行 38 次且全部 evaluation-valid：5 success / 33 residual。一次 root proposal 原子返回 exact 3；
activation audit 为 3 个 candidate、3 个 distinct signature、group size `[1,1,1]`，0 error、0 retry。
poster root 因 train support=0 进入 repair；court 与 Mario roots 静态通过，分别为 3/3 与 2/2
failure activation precision，均只覆盖 1 个 train family，未达到既有 2-family coverage target，最终
court 因 support=3 入选。

no-recursive arm 运行 16 个 pair slot，只在 `court-form-filling-5` 激活一次。该 pair 的 treatment
确实执行：编译后的 skill 保留了 form-field scope、`/root/sc100-blank.pdf` 和
`/root/sc100-filled.pdf` 三个 target/value 指令；raw/candidate 分别产生 66/16 个 action start，
observation、treatment 和 PDF hash 均不同，两个外部离线 verifier 结果仍都为失败。因此总计
raw/candidate 均为 4/16，0 gain、0 harm、LCB=0、cost ratio=0.820789；这证明 actionability
修复改变了实际行为和成本，但尚未产生 task-success gain。另一个不激活 candidate 的
`anthropic-poster-design-2` raw/policy-off trial 使用 68,400,000 bytes，超过冻结的 67,108,864-byte
hard cap；容器按合同终止且 hard-budget retry 被抑制。该 pair 因而 invalid，report 以
`invalid_counterfactual_evidence` non-claim 停止，archive `incumbent_id=null`。这里不提高 fuse，
也不把 invalid raw row 拼接或改记为普通错误答案。

recursive arm 更早暴露一个与候选质量无关的 request-contract 缺陷。poster repair 的 provider
transport 与 JSON object 解析在 53.402 秒内成功，但 response 顶层唯一字段为 `hypotheses`；repair
envelope/parser 严格期待 singular `hypothesis`，因此 candidate-local `response_envelope` failure 阻止
整代 held-out validation/promotion。事后逐层复核 `ValidationContext -> RecursiveValidationEngine ->
StructuredHypothesisProposer.revise` 发现，真实 repair capabilities 已不含 `proposal_batch_contract`；
最初的“batch contract 泄漏”判断错误。实际缺口是 repair 只在 nested `output_schema` 中声明 singular
field，而共享 system contract 仅明确了 root batch 情况，没有在 versioned policy 存在时显式要求
one-object/`hypothesis`，模型仍返回了 root-shaped envelope。recursive report 以
`proposal_model_failure` non-claim 停止，archive 同样无 incumbent。这个结果把下一步限定为显式、
协议绑定的 singular repair response，而不是添加 gate、放宽 parser、增加 retry 或修改 evaluator。

**[ARTIFACT]** 2026-07-13 的 v3.12 fresh root
`paper_primary_v3_12_offline86_ruoli_gpt54mini_outer6_model1_repairscope01` 绑定 scoped-clean commit
`9c692b2d` 并通过 86/86 prewarm。development 的 56/56 actual external trials 全部 valid：38 train
policy-off、16 validation raw/policy-off、2 个实际激活的 policy-on；train 为 8 success / 30 residual，
最大 finalized network 为 35,070,000 bytes，0 provider、infrastructure、action、network、budget、
invalid-pair 或 mismatch。两臂均完成两代并以 `consecutive_non_promotion_limit` 停止。

G1/G2 均返回 exact 3、三种 distinct activation signature、3/3 static pass。三个候选分别覆盖
poster、poem、court 的互斥单一 family；TRAIN precision 都为 1、success false positive 都为 0，
coverage target=2 却没有单 root 达到。单体 selector 因 support=3 连续选择 court root。两代都只在
`court-form-filling-5` 激活：raw 为 9 action starts，两个 candidate 分别为 32/43，三者都失败。
因此每代都是 activation 1/16、raw/candidate 4/16、0 gain/0 harm、LCB=0，cost ratio 分别为
1.079038 与 1.116838。recursive/no-recursive 共享或重放同一证据，最终 archive 字节相同且
`incumbent_id=null`。全部 12 个 static node 直接通过，所以本轮 repair request=0；v3.12 singular
contract 已绑定并通过 bounded live canary，但 full development 没有提供 repair-path 样本。

**[ARTIFACT / QUARANTINE]** 旧 `all-development` 紧接着无条件执行 freeze/controls，产生
`frozen=true` 但 `selected_candidate_available=false` 的空 receipt；`promoted_v2` 与 no-rec control
均编译为空 program set/raw alias。进程被立即终止。partial validation 只留下 8 starts、2 completes、
2/96 records，且两条都因 `codex_action_budget_receipt_missing` invalid；没有 control report、family-out、
HippoRAG、sealed journal 或 test access。`validation.partial_admissibility.json` 已把该批标成
diagnostic-only、performance-claim inadmissible、row-reuse forbidden。它不反向污染此前完整有效的
development report/archive。随后增加的是单一 phase-transition invariant：无 promoted recursive
candidate 时 runner 正常结束，freeze producer fail closed，controls consumer 也拒绝旧空 receipt；
没有调整任何 promotion 分数或阈值。

**[ARTIFACT / PROPOSAL-ONLY]** v3.16 与 v3.17 在正式 benchmark 之前增加一次有界的
TRAIN-only feasibility screen，并复用受版本控制的 v3.15 source receipt。该路径重新从 38 份
policy-off result/trace/action receipt 与 public TRAIN environment 重建 38 observations（6 success / 32
failure）和 31 action profiles，`source_agent_trials_reexecuted=0`；不读取 v3.15 development report/events，
不访问 validation/test/verifier，也不构造 task backend 或 evaluator。

v3.16 的 3/3 singular family-slot model calls 均成功，但只通过 root count、profile binding 与 schema 三项；
distinct single-family signature、minimum support、anti-trigger self-block、executable delta、restatement absence
和 failed-primitive avoidance 六项失败。v3.17 随后只作一次结构修订：host 固定 exact family trigger 与空
anti-trigger，按冻结优先级为每个 slot 选一个 support≥2 的 reusable artifact，并给出
read→parse→update→serialize→write-back blueprint。新结果的 support 为 2/2/3，3/3 有 concrete local tool，
2/3 有 artifact manipulation，且 distinct signature、self-block、restatement、schema 等八项全部通过；
唯一失败是第三候选的 `failed_primitive_binding_count=2`。一次 transport `RemoteDisconnected` 在有界 retry
后恢复，最终仍为 3 个 logical success，因此整体失败不能归因于 credential tier 或 route outage。

两轮报告都明确 `backend_call_count=0`、`evaluator_call_count=0`、`validation_task_count=0`、sealed/test=false、
raw response/secret 未落盘。`failure_blocks_future_trial_spend_only=true`，既没有 benchmark row，也没有
promotion/archive。这个结果正确阻止了 preflight/lock/prewarm/smoke/development 支出，并冻结该 free-text
family-slot 路线为负结果；后续不再通过新增 prompt、acceptance predicate、retry、selector 或 promotion gate
继续追逐同一表示。

### 7.3 当前 infrastructure/protocol 状态

全 inventory 的
[`offline verifier coverage audit`](skilllearn_offline_verifier_matrix.md)
给出 credential-independent 任务 **86/95 可运行、9 项 blocked**。本次没有继续追逐
大体积、异构依赖，而是在任何新模型调用前冻结了保留原 split assignment 的 86-item
offline-ready subset：

- instance holdout：38 train / 16 validation / 32 sealed test，16 families；
- family out：48 train / 11 validation / 27 sealed test，9/2/5 families；
- 排除 3 个完整 infrastructure-blocked families、缺权威 verifier 的 GDP item 2，以及
  原先需要 `GH_TOKEN` 的 family；
- 不复用相邻 GDP item 的 oracle，也不把 online evaluator 当作替代品。

新的
[`offline86 verifier matrix`](../artifacts/offline_verifier_matrix_offline86_20260711_v1/matrix.json)
实际得到：7/7 active profiles、15/15 train-family representatives、
`blockers=[]`、`manifest_execution_ready=true`、`passed=true`。随后完整本地 preflight
同样得到 `blockers=[]`、`selected_item_count=86`、`ready_for_live_skill_generation=true`。
两次检查均未执行模型，sealed-test 语义也未暴露给模型。该结果另有受版本控制的精简
[`offline readiness receipt`](../manifests/skilllearn_offline_readiness_receipt_v1.json)，
供 protocol/lock 绑定；不再把 `.gitignore` 下的 matrix artifact 当作唯一证据入口。

这里必须区分三层证据。readiness receipt 绑定的是 7/7 profile contract、15/15 train-family
动态代表探针和 86-item 静态 preflight；它不声称逐项执行了 86 个 verifier。独立的
all-manifest runtime prewarm 才覆盖 train、validation、sealed-test 的全部 86 个 image/runtime。
本次第一次 cache-only 检查暴露 14 个未建镜像；它们在独立准备阶段有界构建后，第二次
cache-only 验收为 **86/86 passed、0 failed、47 个唯一镜像、7 个离线 verifier runtime**，
且最终 receipt 记录 `online_build_attempted=false`。这仍只是零模型、零 sealed scoring 的
基础设施证据，不是 86 项任务准确率。

9 项未进入主协议的原因已经分型，而不是统称“缺缓存”：GDP item 2 在当前官方主分支
仍缺权威 `test_outputs.py` 和 solution；Druid 已有零下载 direct-`javac` 参考 patch 路线，
但缺 vulnerable negative control 与 arbitrary-edit generality；Scala 还需要固定 SBT/Maven
闭包和 CLI verifier adapter；NLP 则需要 Python 3.10 CPU runtime 与约 0.5--1.2 GB 的
最小 ML closure。它们是后续独立 infrastructure workstream，不再阻塞主学习实验。

旧 development lock 仍声明 `network_scope_audit=v1`，v3.1 已升级为 hard-egress v2、
offline-verifier v3、32 MiB/题 hard fuse、prompt-action lowering v1 和 protocol-owned
promotion v2；v3.2 只把同一 fuse 版本化为 64 MiB。旧 live 与 v3.1 observations 因此
都只能作诊断，不能与 v3.2 直接合并。

最新 `gptpro03` protocol lock 绑定 clean commit `18ff3417`、`validation_issues=[]`，对应
prewarm 为 86/86。该 run 的 37 条 valid train observation 与 1 条 hard-cap invalid 也不能
被事后修补或跨进程 replay；它只证明 offline receipt 修复和 provider 稳定性，同时否证
当前 64 MiB execution contract 对所有 train trajectory 的可行性。

sealed test 仍未访问，这是正确状态。

## 八、v2 当前最关键的架构缺口

### 8.1 已关闭的 P0：promotion 标准所有权

**[CODE + TEST]** `PromotionGateSpec` 现在是唯一的 evaluator-owned contract。pairs、
confidence、net gain、activation、minimum effect LCB、maximum harm 和 maximum cost
全部由 [`PaperProtocol`](../assumption_agent/benchmarks/paper_protocol.py) 严格解析；实验 CLI
已移除 `--minimum-pairs` 旁路，recursive/no-recursive 两臂共享同一个 immutable spec，
protocol lock 和 freeze report 都复核完整 promotion mapping。

candidate 的 `ExpectedEffect` 仍可表达更保守的自我约束，但 effective threshold 只能收紧：

```text
effective_min_delta = max(protocol_min_delta, candidate_min_delta)
effective_max_harm = min(protocol_max_harm, candidate_max_harm)
effective_max_cost = min(protocol_max_cost, candidate_max_cost)
```

新增对抗测试证明 `minimum_delta=-1`、`maximum_harm_rate=1`、超大 cost ratio 都不能放宽
protocol 及格线；更严格 candidate 则会生效。promotion decision/event 同时记录 protocol、
candidate 和 effective thresholds。这是一次收敛现有判断所有权的修复，不是新增 gate。

### 8.2 已收紧的 P0：外部 action/verifier/fallback contract

内部 `PolicyRuntime` 的 typed lane action 仍保持 effectful。外部 SkillLearn backend 则不再
冒充同等语义：compiler 只接受 `execute_step`、`produce_artifact`、`request_evidence` 三类
`prompt_directive` 和 `check_condition` 这一类 `agent_local_self_check`。`enable_lane`、
`disable_lane`、`prioritize_lane`、`set_parameter`、`require_verifier`、`abstain` 没有外部
lowering，因而 fail closed；proposal/repair capabilities 也只广告这四类。

编译后的 `SKILL.md` 不再暴露 benchmark external verifier、policy-off/on evidence 或
expected-effect 阈值，并明确 external verifier 只在 agent 退出后运行。compile manifest
绑定完整 program set、每项 lowered treatment、实际 `SKILL.md` content hash、
`external_verifier_exposed_to_agent=false` 和 `baseline_on_nonactivation_only_v1`。输出目录
由 staging tree 整体替换，旧 skill 不会残留；action target/value 引用 hidden verifier、
required evidence 或 policy-off/on 时在 validator/compiler 共用的结构检查中 fail closed。

novelty、counterfactual replay、training replay 和 proposal prior context 现在统一使用实际
lowered-treatment identity，而不是 raw program metadata。只改 expected-effect/verifier 元数据
不能获得新行为身份或重新采样；真正改变注入内容才会改变 treatment hash。generation report
同时绑定实际评价的 candidate treatment；freeze 使用 archive program 重算该 hash，并用与
runtime gate 相同的 summary-blocker 函数重算 promotion，防止替换 treatment 或伪造 allowed。

fallback 的伪证据链也已删除：activated candidate 是独立 treatment，
`baseline_preserved=false`；只有 trigger miss 时 candidate observation 直接 alias baseline，
才记为 observed baseline。SkillLearn promotion 依据冻结的 paired harm/LCB/cost contract，
不再要求一个由字符串声明伪造的 post-verifier rollback。

这没有把 prompt directive 变成 typed program；它做的是把宏观“注入 skill”与细粒度
agent instruction 的证据层级说清楚。真正强类型外部 operator 仍可作为后续研究方向，
但不再是当前论文协议的隐含主张。

### 8.3 P1：failure-only support bias 已关闭，但 precision-first 收缩为 coverage starvation

v3.5 的真实第一代已经给出反例：最大 failure-support 的 recursive repair 在 held-out
validation 造成 0 gain / 2 harm，而较窄的 no-recursive root 为 1 gain / 0 harm。v3.6
因此没有放宽 promotion gate，而是改变 gate 之前的 train-only proposal selection：

- 每个 valid train failure 仍形成带 sanitized failure context 的正例；
- 每个 valid train success 形成 `baseline_success_control`，只含 runtime features 与 label，
  不含 instruction、feedback 或 execution context；
- root/repair/replay request 与双臂第一代 checkpoint 绑定全部 labeled transition IDs；
- static support pass 仍只由 failed rows 决定，success rows 不会把无支持候选洗成通过；
- 同代候选按精确 `failure activations / all train activations`、success false positives、
  failure support、predicate/action complexity 与 payload hash 排序，不读取 validation。

held-out report 另增加 evidence-valid activation、activated gain/harm、precision、harm rate 与
abstention。其分母排除 evaluator-invalid、provider mismatch 与 budget mismatch 的并集；零
valid activation 时 ratio 为 `null` 且 `defined=false`。这些字段是诊断，不进入既有
`PromotionGateSpec`。v3.9 clean development 已验证这套 selection 不再选择 success false
positive，却暴露相反失败模式：两代入选 candidate 均为 train support 2/38、held-out activation
1/16，0 gain/0 harm。precision-first 排序把搜索收缩成了局部 family policy，无法形成足以检验或
晋级的 prospective coverage。下一步应改变 proposer/search/configuration formation，而不是再给
promotion 增加 blocker 或放宽 minimum activation。

v3.10 已完成这个 coverage 假设的 live 检验：exact-three 机制一次给出 3 个静态可执行 root，
coverage-first 从 1-family/2-failure 候选转而选择 2-family/6-failure 且 0 success false positive 的
retrieval root；held-out activation 也确实从 v3.9 的 1/16 提高到 2/16。因此“只要覆盖更宽就会出现
收益”被否定：candidate/raw 仍同为 3/16，0 gain/0 harm。此时继续扩大 bundle 只会扩大含糊
directive 的覆盖面，尚无依据。轨迹和 compiled skill 复核把下一 blocker 定位为 operator actionability：
training feedback 强推 completion check，lowering 又丢 target 并输出 mode JSON。v3.11 因此保留
coverage objective，只修训练提示、action schema 与 agent-facing lowering。

v3.11 live 已把这个假设拆成两部分。action representation 部分通过：三项 root directive 都是
可读的 target/value 指令，入选 court treatment 在 held-out 激活题上把 action starts 从 66 降到 16，
并生成不同 PDF；因此不能再把 0 gain 解释为 skill 未注入或 lowering 未执行。但 task-success 仍为
0→0，说明单个局部 directive 尚未解决缺失日期/checkbox 等真实 operator 细节。另一方面，recursive
arm 没有检验到 repair quality，因为 generic response contract 未显式绑定 singular repair，模型返回
了 root-shaped batch envelope。v3.12 只显式版本化这一 response scope；它不是新的 performance gate。

v3.12 fresh root 随后完整结束，但没有进入 repair：两代各 3 个 root、共 12 个 recursive/no-recursive
static node 全部通过，`hypothesis_repair_requested=0`。因此这轮能说明 singular contract 没有破坏正常
development，却不能说明 live repair quality 已改善，也不应通过人为收紧 static check 来强迫 repair。
更重要的新证据来自 candidate formation：G1/G2 的 poster、poem、court roots 都是 TRAIN precision=1、
success false positive=0，但每个只覆盖一个 family；selector 连续只保留 support=3 的 court root，
把另两个互补、高精度局部 policy 丢到 shadow。held-out 因而仍只有 1/16 activation、0 gain/0 harm。
v3.13 已把 proposal diversity 形成的互补 roots 用冻结的 TRAIN-only subset objective 组合成单一
program-set treatment，并规定只做一次 paired validation；它不能逐个试 validation 后挑最好，也没有
增加或放宽 promotion gate。live 证明 set-level routing/replay 按设计执行：每套 selected bundle 都是
5/5 failure support、2 families、0/8 success FP，held-out 激活也从单一 family 扩为 poster+court 的
2/16。但 6 个真实 policy-on 全部失败，故它否定的是“精确命中 baseline failure 足以预测 action utility”。
no-recursive G2 还有一个 7/7、3-family、0-FP 的三成员 subset，却因为 family deficit 在 2 后封顶且
`bundle_size_asc` 早于 `failure_support_desc` 而只排第四。v3.14 已一次性修这一 TRAIN-only objective
和跨臂 baseline replay：leading precision/capped-deficit/success-FP/overlap 顺序不变，只在这些项相同
时把实际 family count 与 failure support 提到 bundle size 前。live G1 确实选中 7/7、3-family、0-FP
三成员 set，并把 activation 从 2/16 提到 3/16；这证明 selector 改动生效。结果仍是 6 个 G1 on 与
1 个 no-recursive G2 on 全失败、0 gain/0 harm，所以预先声明的停止条件已经满足：不再迭代 selector，
action quality 单列为下一研究问题。valid policy-off evidence 产生 31 次跨臂/多代 replay；但一条 64 MiB
hard-cap invalid 没有形成 terminal memo，随后被另一 arm 以相同 baseline evidence key 重执行。因此
shared valid cohort 有效，terminal invalid 传播仍不完整；它是 attribution correctness，不是新评分 gate。

v3.15 在提交 `696a2954` 只修这两个 gate 前/归因层缺口。`train_only_material_action_delta_prompt_audit_v1`
把 task instruction 标为 baseline requirement，而不是候选自身的 treatment；request-local prompt 要求每个
hypothesis 至少提出 exact constant/mapping、concrete local tool command 或 artifact-internal manipulation
中的一种 material delta，并允许模型使用其静态知识。它的证据面严格限于 TRAIN failure：public
`environment/` 被归一化为 allowlisted package、task-local path、file/operation labels，policy-off trace
只保留 allowlisted executable、flag、`/root` path、成功/失败和 file-change facts。symlink escape、credential/
network syntax、敏感路径、test/verifier/oracle/solution 引用整行丢弃；model prose、command output、raw trace/
environment 文本均不进入 proposal profile。成功 TRAIN control 仍无 instruction/context，validation outcome、
verifier/test content 均不可见，proposal 也没有外部 tool、网络或 runtime install 权限。

`proposal_action_delta_audited` 只回答候选是否看起来提供 material delta、是否有 instruction-restatement 风险。
即使 audit 输入异常，它也不能 reject/重试 response、触发 recursive repair、改变 candidate selection 或进入
promotion decision；因此这是非评分诊断，不是隐藏 gate。完整 profile 通过 hash 引用，首代 checkpoint、
profile count/set hash 写入 recursive/no-recursive report；freeze 同时核验 plan 与 generation、两臂共享值，
并写入 receipt，防止事后替换 action-design cohort。

`behavior_identical_shared_validation_baseline_terminal_outcome_replay_v3` 则在同一冻结 request 已走完或被
明确抑制的 same-request retry 后，把最终 invalid 记为 run-scoped immutable tombstone。key 绑定 baseline
execution/fairness 与 retry policy identity；后续 arm/generation 复用同一 invalid、增加 0 次 baseline execution，
但 tombstone 永远不是 promotion evidence，pair 仍 terminal non-claim，冲突也不覆写 first outcome。
v1/v2 历史语义不变。以上机制在 453/453 离线测试通过后进入正式 live root：clean lock、86/86
cache-only prewarm 和 smoke 均通过，full development 完成 57/57 valid actual trials（38 TRAIN off、
16 个共享 validation baseline、3 个 activated on）与 8/8 proposal/repair model calls。TRAIN 为
6 success / 32 residual；最大 online-agent concurrency=1；provider、infrastructure、action-budget、
network-cap 与 pair-mismatch failure 均为 0。recursive G1/G2 各只激活 1/16，candidate/raw 都是
4/16、0 gain/0 harm；no-recursive G1 在 TRAIN static audit 被拒，G2 的 1/16 activation 同样
0 gain/0 harm。共享 valid cohort 支持 32 次 zero-execution baseline replay。因本轮无 invalid baseline，
terminal-invalid tombstone 没有被 live 触发，其 retry-identity/非评分语义仍只由离线测试证明。

两臂 report 都是 claim-eligible，但这只表示该 clean negative result 可用于诊断，不表示 promotion：
两臂均以 `consecutive_non_promotion_limit` 停止，archive 的 `incumbent_id` 都是 `null`，sealed/test=false，
freeze、controls、family-out、HippoRAG 与 sealed 均未运行。action audit 给出更直接的失败定位：13 个
candidate（9 roots + 4 repairs）中 7 个被检测为有 material delta，6 个有 restatement risk；然而 7 个
material delta 全部只有 `exact_constant_or_mapping`，concrete local tool、artifact manipulation 和
environment primitive 都是 0。更严重的是 9 个 roots 全部指向 `anthropic-poster`，候选形成从 v3.14 G1
的 3-family/7-support 倒退为单-family/2-support。由此可见 action-quality prompt/profile 的 wiring 生效，
但 proposal 仍没有把 profile grounding 转成多 family、可执行 operator；下一改动必须留在 gate 前。

v3.16/v3.17 已把这个 gate 前假设做成独立 proposal-only 试验，而不是再花一轮 38+16 task trials。
结构化 family slot 成功消除了 root family collapse；v3.17 的 exact trigger、空 anti-trigger 和 artifact
blueprint 又消除了 support=0、self-block 与 restatement，并使 3/3 candidate 具有 concrete local tool。
所以 diversity、support 和表面 actionability 已不再是最后 blocker。唯一失败落在第三候选对两个 failed
TRAIN primitives 的绑定。

这里的更深问题不是再加一条“不要使用失败 primitive”的 prompt。第三个 slot 的本地 failed set 是通用
executable `file` 与 `python`；它们来自失败 command 的共现，而不是证明这些工具本身导致失败。自由文本
action 又很自然会在“解析文件”工作流中提到 file/Python。因此当前 taxonomy 同时混合了“在失败轨迹中出现”
与“因果上应禁止”，而模型侧只收到 count/hash，也不可能可靠满足未知的逐值 exclusion。

新的前置结构复核已从 38 条 receipt-bound V3.15 TRAIN trace 中保留 429 个按时间排序的 allowlisted command
occurrences，其中 70 个失败；另有 208 个 non-allowlisted command 被显式计数后丢弃。38 条 trace 均完整，
总 relevant action starts 为 655，单 trace 最大 61，低于冻结的 100-action budget；失败 span 没有一个是最后
allowlisted span，且 observational inadmissibility 计数为 0。这些数字只证明 allowlisted chronology 未去重、
未截断，不声称覆盖完整 raw-command chronology。closed typed graph 已把 proposal output 收窄为一个 opaque、
已注册的 `recipe_id`，使 primitive、locator 与 free-text action 不进入模型输出域；但 materialization 仍经现有
prompt-directive/self-check compiler 交给通用 agent，capability implementation 未验证，也没有 restricted
runtime executor。实现与 preregistration 已就绪，正式的一次性离线 decision 仍未执行；即使 PASS，也只使
另行冻结的 typed-selection integration diagnostic 有资格运行，不授权 development。继续补 prompt 或放宽
acceptance 都会掩盖这一边界；这不是 v3.18 gate patch。

### 8.4 P1：prospective runtime features 仍过粗

当前 SkillLearn feature catalog 主要只有
[`family/category/difficulty/tags/environment_file_count/has_container_environment`](../assumption_agent/benchmarks/skilllearnbench.py)。
完整 train instruction 与 v3.15 的 environment/action-trace profile 可以帮助设计 action，但都被正确禁止
作为 runtime trigger。v3.15 live 的九个 roots 仍全部坍缩到 `anthropic-poster`，也说明“提供 profile”本身
不会自动产生结构化 family diversity 或新的 prospective routing feature。

这比 legacy hash-only data 有进步，却仍缺：

- artifact/output schema；
- constraint signature；
- required capability；
- verification class；
- input modality与工具需求；
- 可在未见 family 上计算的语义特征。

当前很多 candidate 因此只能形成宽泛 family/env-count gate，不足以支撑可信 family-out
prospective routing。

### 8.5 P2：archive 是树形数据结构，不是多 clade 搜索算法

[`ArchiveNode`](../assumption_agent/archive.py#L23-L31) 有 parent、active programs、epoch
和 generation；但 [`PolicyArchive`](../assumption_agent/archive.py#L96-L102) 只有一个
`incumbent_id`。v3.12 及以前每代只把 train-only 排名第一的 `eligible[0]` 送入 validation；
v3.13-v3.15 虽可把多个互补 roots 组成同一 candidate node，但仍只选择一个 program set、只从当前
incumbent 扩展：
[`evolution.py:L325-L453`](../assumption_agent/evolution.py#L325-L453)。

因此 active 算法仍是保守的单 incumbent configuration hill climbing，而不是 RQGM 中多 clade 并行保留、
按 metaproductivity 继续扩展的 archive search。

此外，`ScoreRecord` 只存 candidate successes/total 和 item-set hash，未直接绑定完整 pair
bundle、gain/harm/cost、promotion decision 与 protocol hash。archive 的 provenance 还不够
承担跨 epoch、多分支重排。

### 8.6 P2：evaluator co-evolution 还是独立骨架

`EvaluatorEpochController`、anchor lower bound 和 selective invalidation 有代码与测试；
但主 SkillLearn 实验明确只允许 task/policy hypothesis，evaluator hypothesis 不能编译为
agent skill。当前没有真实 evaluator challenger、epoch transition 或 incumbent re-ranking
artifact。

所以 v2 可以声称“有 evaluator-epoch mechanism skeleton”，不能声称“已经实现 Red Queen
式 agent/evaluator co-evolution”。

### 8.7 P1：递归修复被触发过，但没有因果收益证据

v2 recursive validation 主要修复 schema、trigger support、action vocabulary 和 epoch 等
静态/训练检查。v3.9 已得到完整 clean 对照：recursive 第一代的 depth-1 repair 通过静态审计并在
held-out 真正激活 1/16，但相对 raw 为 0 gain/0 harm；no-recursive 同代 root 因 anti-scope
support 未通过静态审计。该对照证明 repair 能把 candidate 从静态失败变成可执行 treatment，却
没有证明它改善 task success，而且激活范围过窄，无法形成 retained incumbent。

v3.10 第一代 3/3 roots 均直接通过静态检查，没有触发 repair；第二代又在 root response contract
处终止。recursive/no-recursive archive 因而字节相同，report 只在 arm/trace/path provenance 上不同。
这轮没有提供新的 repair 因果样本，也不能把两臂相同解释成 repair 无效。

因此当前可说“递归修复机制会运行并改变候选可执行性”，不能说“递归验证已经改善性能”。

### 8.8 文档与协议漂移正在本次收口

[`ARCHITECTURE.md`](../ARCHITECTURE.md) 和
[`BENCHMARK_PROTOCOL.md`](../BENCHMARK_PROTOCOL.md) 此前曾有段落声称 destination allowlist /
dependency-cache-only 尚未强制；但当前
[`docker_egress.py`](../assumption_agent/benchmarks/docker_egress.py) 和 protocol manifest 已
实现 provider-only hard egress、offline package mode 与 network fuse。本次已同步主
README、benchmark protocol、offline-verifier matrix 和 status 摘要；本轮又把 receipt
runtime provenance、v3.5 serial execution-policy / repair identity / response-contract binding、v3.6 contrastive/invalid-evidence contract、v3.7-v3.9 并发容量/共享 slot、v3.10/v3.11 live 结果，以及 v3.12 singular repair scope、clean 负结果与 empty-incumbent phase prerequisite。历史段落仍
保留为 diagnostic ledger，不能当作当前协议。2026-07-14 又同步 v3.13 clean negative evidence、
v3.14 的 mixed-claim live 结果，以及由此确定的 action-quality 转向；没有把 no-recursive 的机械
claim eligibility 写成 primary performance improvement。当前再同步 v3.15 提交 `696a2954` 的
TRAIN-only action profile、audit-only 边界、terminal-invalid memo/retry identity 与 paired report/freeze
provenance，以及随后 clean 57/57-valid live negative evidence。两臂 claim eligibility 只表明证据可用，
没有被写成 promotion 或 performance improvement。

这种文档漂移本身会破坏 protocol review；重新跑论文实验前必须同步。

## 九、下一步优先级与硬验收标准

| 优先级 | 工作 | 硬验收标准 |
|---|---|---|
| 完成 | 冻结 evaluator-owned promotion policy | 已由 protocol 绑定完整 spec；candidate 只能收紧；对抗测试通过 |
| 完成 | 收紧外部 action/fallback contract | 4 类 prompt/self-check lowering；6 类 unsupported op fail closed；observed fallback 不再由字符串伪造 |
| 完成 | 冻结 offline-ready 范围 | 86-item manifests 保留旧 split；readiness matrix/static preflight 均 `blockers=[]`，无模型调用 |
| 完成（v3.3 历史） | 全 manifest runtime prewarm | cache-only 86/86、47 images、7 verifier runtimes；无 agent、无 sealed scoring；不作为 v3.4 receipt |
| 完成（v3.4 历史） | clean commit、lock 与 v4 prewarm | lock 绑定运行时 scoped clean commit `ad66d5a2`、claim-eligible、0 validation issue；cache-only prewarm 86/86、0 model call、0 sealed scoring |
| 本次排除（非稳定性结论） | 64 MiB fuse 作为本 batch 的直接 blocker | v3.3 38/38 均低于 64 MiB；最大 40.6 MB，video-1 为 19.69 MB；0 cap/provider error。canary/full 波动为 1.47/19.69 MB，尚无跨运行稳定性证据 |
| 完成（零模型） | 定位 model-only execution boundary | 根因为 Codex 0.144.1 丢弃 `tools.web_search=false`；canonical 顶层 disabled 的 loopback 为 7 tools / 0 web，旧键阳性对照为 8 tools / 1 hosted web；未调用模型、未评分 |
| 完成（实现与离线注入） | 执行 action budget | `codex_action_start_v1` 在第 N 个 `item.started` 终止 PGID，并按 task/TID 清除 dedicated-container 基线后的 live task；异常退出、malformed、N+1、`setsid`、zombie leader/live worker、残留 descendant 与 evidence tamper 均 fail closed；所有 arms 统一 action-step cost，不再混合 token/step |
| 完成（机制） | v3.4 clean runtime canary | lock/prewarm、PATH、host-readable receipt 均已验证；max2 v5 为 2-step valid truncation，本地 verifier 有效，0 remote tool，全部 agent task 已退出 |
| 完成（协议版本化） | v3.5 serial execution policy | 五个在线 phase 的 `parallel_workers` 全部由 4 改为 1；其余 model/subset/budget/offline evaluator/retry/circuit/search/promotion/sealed 合同不变 |
| 完成（容量验证） | v3.5 pre-fix serial train | clean lock/prewarm，38/38 valid、9 success、0 provider/cap/action/tool/verifier invalid；约 99 分钟；旧 rows 不跨进程复用 |
| 完成（确定性修复） | repair branch identity | parent ID + status-independent parent hash + depth + canonical candidate content 派生 ID；模型 ID/status 不控制主键或生命周期；真实 depth-2 collision 与 archive fail-closed 对照通过 |
| 完成（异常边界） | malformed proposal isolation + claim binding | post-transport envelope/parse failure typed 化；root 原子 replay、repair branch-local、整代 validation/promotion blocked；report/freeze 防失败 claim 篡改 |
| 完成（第二次容量验证） | v3.5 repairid01 serial train | 38/38 valid、9 success、0 provider/cap/action/tool/verifier invalid；5,188.542 trial-seconds；repair malformed response 在 validation 前暴露，旧 rows 不复用 |
| 完成（首次全闭环，非 clean claim） | response-fix fresh-root development | 新 lock/prewarm、38/38 all-valid train、proposal/repair、双臂 paired validation/promotion/report/archive 均实际完成；第一代 32/32 pair valid、sealed=false；第二代 no-rec 被 503/circuit 污染为 9 invalid，故“整轮 0 invalid”硬标准仍未满足 |
| 完成（v3.6 实现，live 待验） | invalid counterfactual lifecycle/claim | invalid/provider/budget mismatch terminal non-claim，不增加 consecutive non-promotion；archive score invalid；report/freeze 独立重算；legacy 污染报告同样拒绝；mismatch bundle 不缓存 |
| P1 | 递归因果归因 | 两臂共享 train evidence 和 roots，唯一差异是 repair；behavior-identical 时 effect 报 N/A，不重采样 |
| 完成（v3.9 clean 负结果） | contrastive trigger learning 首次 full live 验证 | 38/38 valid train、16/16 valid pairs、0 provider/infra/mismatch；两代 candidate 均仅 1/16 activation、0 gain/0 harm，证明 precision-first 已避开 false positive 但发生 coverage starvation |
| 完成（v3.10 clean 负结果） | proposal diversity 与 train-only coverage selection | exact 3 / 3 static pass；选中 2-family、6/6 precision root；activation 2/16 但 0 gain/0 harm；G2 两批 signature collapse 被 terminal reject；无 incumbent |
| 完成（v3.11 机制有效、性能/claim 失败） | actionable directive lowering 与 audit-only diversity | 38/38 valid train；exact 3 / 3 distinct signatures；court treatment 在激活题改变 trace/PDF 且 66→16 actions，但 0→0；no-rec 因未激活 raw poster 超 64 MiB non-claim；recursive repair 返回 batch envelope 而 non-claim；无 incumbent |
| 完成（v3.12 clean 负结果；repair path 未触发） | singular repair scope + candidate-formation 诊断 | 56/56 valid actual trials、0 provider/infra/mismatch；两代 exact3/3 static pass但所有 root 均只覆盖 1 family，selected court 仅激活 1/16、4/16 对 4/16、0 gain/0 harm；repair request=0；两臂无 incumbent |
| 完成（phase prerequisite 修复） | 禁止空 incumbent freeze/control | 误入的 partial controls 仅 2/96 invalid，已标 diagnostic-only/no-reuse；runner 无 incumbent 即结束，freeze producer 与 controls consumer 双向拒绝空候选；不改评分 gate |
| 完成（v3.13 clean 负结果） | train-only complementary policy bundle | 76/76 valid、0 provider/infra/mismatch；三套两成员 bundle 均为 TRAIN 5/5、2-family、0 success-FP，held-out activation 2/16；6 个 policy-on 全失败、0 gain/0 harm、无 incumbent。program-set routing/replay 正常，但 G2 cross-arm raw 未共享，且 7/7 三-family subset 被 capped target + size-first 排到第四 |
| 完成（v3.14 mixed-claim 负结果） | shared baseline cohort + support-aware tied-set ranking | lock/prewarm/smoke clean；62/62 attempted、38/38 valid train；G1 选中 7/7 三-family set并 activation 3/16，但 7 个 on 全失败、0 gain/0 harm。一条 recursive raw 68.66 MB 超 fuse 使 primary non-claim；31 次 valid baseline replay，invalid key 又跨臂执行一次；两份 archive 无 incumbent |
| 完成（v3.15 clean 负结果） | TRAIN-only material action delta + terminal-invalid attribution | commit `696a2954`、453/453 offline 后，clean lock、86/86 prewarm、smoke 与 57/57 valid live 完成；8/8 model calls、max online concurrency=1、0 provider/infra/budget/network/pair error；recursive G1/G2 与 no-rec G2 均 1/16、0 gain/0 harm，32 次 baseline 零执行 replay；两臂 claim-eligible non-promotion、incumbent null、无 downstream |
| 完成（v3.16/v3.17 proposal-only 负结果） | structural family-stratified proposal formation + artifact blueprint | 复用冻结 TRAIN receipt，0 source rerun、0 benchmark/evaluator call；v3.17 通过 8/9 feasibility，解决 family collapse/support/self-block/restatement，但第三候选仍绑定 2 个 failed primitives，故未授权 development |
| 完成（正式离线 PASS） | typed operator/capability grammar + causal action-span evidence | commit `b03c643a` 的唯一一次 decision 通过 9/9 predicates，既有 report/event/lock 精确复验通过。38 条 receipt-bound trace 含 655 个 action starts（max 61/100）、429 个 chronological allowlisted spans、70 failed、63 later scope-matched recoveries 与 208 discarded commands；3/3 graph/program materialize、9/9 tamper probes fail closed、primitive/locator disclosure=0、live model/backend/evaluator call=0。PASS 仅证明闭合表示的离线 feasibility，不授权 development |
| P0（下一 workstream） | separately frozen typed-selection integration diagnostic | 把 production proposer/evolution 的候选选择面真正收缩为 opaque registered `recipe_id`，并把 snapshot graph/catalog commitment 与 harness materialization 贯穿真实选择路径；禁止模型生成 primitive、locator 或 free-text action。另行预注册、只做非评分 integration diagnostic；通过前不启动 fresh development，且不新增 prompt/gate patch |
| P1 | prospective family-out routing | trigger 不依赖已知 family 或预编译 item ID，只使用冻结、无 gold、运行时可得语义特征 |
| P2 | 多 clade archive | 同 epoch 至少两个 clade 可继续扩展；node 绑定 protocol/evidence/promotion hashes，并报告 retention 与 branch productivity |
| P2 | evaluator co-evolution | 独立 anchor challenger、epoch transition、selective invalidation 和旧 incumbent re-evaluation 实际执行后再作主张 |

近期顺序应是：

1. 已完成：审阅并提交 protocol/action/subset 改动以及 3 个新 manifest/receipt 文件；
2. 已完成：在 clean scoped commit 上重建 claim-eligible lock 和 86-item content-hashed prewarm receipt；
3. 已执行但未形成性能证据：第一次 full development 在 26 个有效 train observation 后，
   被 provider 429/circuit 与一个既有 hard-byte fuse fail-closed；未进入 proposal/validation；
4. 已完成：一个单题、5-step、非 claim transport canary 得到有效 offline-verifier
   observation，确认 provider 已从 429 恢复；
5. 已完成：同协议 fresh-root rerun 再次在 `court-form-filling-6` 超过 32 MiB；按 stop
   rule 中止，v3.1 判 execution-infeasible；
6. 已完成设计：新建 v3.2，仅把统一 fuse 一次性版本化为 64 MiB，其余实验合同不变；
7. 已完成：v3.2 clean lock/prewarm 均通过，64 MiB 未触发；full run 在 8 个有效 train
   observation 后被 provider 的“无可用 distributor channel”503/429 熔断；
8. 已完成：新 GPT Pro credential 的 API/Codex canary 证明同一路由恢复；`gptpro01`
   跑完 38 train，暴露两个 deterministic receipt false-negative，而非 provider/cap 问题；
9. 已完成：receipt auditor 改为绑定实际 runtime profile/command，136/136 tests 通过；
   `gptpro03` 中两项 temperature 均以 7-test CTRF valid failure 完成；
10. 已完成：`gptpro03` 跑完 38 train，但 `video-object-counting-1` 以 71.1 MB 超过
    冻结 64 MiB；37 valid / 1 invalid，proposal 被 fail-closed，v3.2 判 execution-infeasible；
11. 已完成：v3.3 只版本化 Codex execution treatment，v3.1/v3.2 保持旧 mapping；
    150/150 tests、strict-config、claim lock 与 86/86 prewarm 通过；
12. 已完成：video-1 canary 为 1.47 MB valid failure；full run 中 video-1 为 19.69 MB，
    38 个 trial 最大 40.6 MB、0 cap/provider，本次排除 fuse 作为直接 blocker，但尚未证明
    跨运行稳定性；
13. 已执行并 fail-closed：full train 为 37 valid / 1 `web_search` policy invalid；
    proposal/counterfactual/sealed 均为 0，四份 report/archive 未生成。不得重试或通过新 gate
    洗掉 invalid；v3.3 已冻结为不可复用的诊断证据；
    v3.1–v3.4 仅作为 immutable evidence，当前代码仍可按其声明的 schema 验证历史
    receipt，但不承诺这些协议在当前 commit backward-executable；
14. 已完成零模型定位：Codex 0.144.1 把旧 boolean key 丢弃并默认暴露 cached hosted
    search；canonical 顶层 disabled 的真实 wire 捕获无 web，旧键阳性对照稳定检出 web；
15. 已完成 v3.4 最小实现与离线注入：同一 execution policy 同时冻结 model-only tool
    exposure、可执行 action budget、dedicated-container task/TID 清理 receipt、token completeness 和统一 action-step
    promotion cost；未改变 cap/subset/evaluator/promotion/sealed；
16. 已完成：v3.4 clean claim lock、新 shared runtime 与 v4 86/86 cache-only prewarm；receipt
    显式记录 test infrastructure inspected、sealed scoring=false、test bytes exposed to model=false；
17. 已诊断并修复：max2 canary v1 在模型请求前因 shell 的 `PATH=... rm && node` 作用域失败；
    `995e6446` 使用固定 runtime PATH 和 node/codex 绝对路径；
18. 已完成：canary v2/v3 的 no-distributor 503 被 `ba0f36cf` 正确归类；v4 到达模型和 verifier
    后暴露 root-owned `0600` audit artifact，`1df3092a` / `ad66d5a2` 改为显式 `0644` 并补生产断言；
19. 已完成：max2 v5 为 evaluation-valid 的 2-action 截断与本地 verifier failure，0 remote tool，
    action receipt 和 process cleanup 均 valid，因此一次 fresh-root development 获准启动；
20. 已执行并 fail-closed：四并发 development 的 38 个 outcome 为 17 valid（3 success）、4 个
    `provider_rate_limit`、17 个 circuit skip；0 cap/action/tool/verifier violation，未进入 proposal，
    四份 report/archive 未生成，sealed 未触碰，17 条 valid 不得跨 run 拼接；
21. 已完成 v3.5 最小版本化与容量验证：五个在线 phase 的 worker 统一从 4 改为 1；新 lock/
    prewarm 后 serial train 为 38/38 valid、9 success、0 provider/cap/action/tool/verifier invalid；
22. 已执行并 fail-closed：proposal 返回 3 roots，第三个的 depth-1/depth-2 repair 复用同一模型 ID
    但 payload 不同，archive 在 paired validation 前抛 collision；四份 report/archive 未生成，
    sealed 未触碰，38 条 train 不得跨进程拼接；
23. 已完成 repair identity 最小修复：repair ID 改由 parent content/depth/canonical candidate
    content 确定性派生，model ID/status 不控制主键或 lifecycle，archive 冲突保护不放松；
24. 已执行第二个 fresh root：38/38 valid、9 success、0 provider/cap/action/tool/verifier invalid，
    proposal 返回 3 roots；第一个 repair transport 成功但 response envelope malformed，裸 ValueError
    在 validation 前终止；report/archive/sealed 仍为 0，38 条 train 不跨进程复用；
25. 已完成 `d70562de` 与当时的 216/216 离线回归：malformed root/repair 进入既有 typed failure isolation，
    root replay 原子、repair branch-local、整代不 validation/promotion；report/freeze 强制 non-claim。
26. 已执行 response-fix fresh root：clean lock、86/86 prewarm 与 38/38 valid train 后，首次完整
    走通 proposal、真实 repair、recursive/no-recursive paired validation、两代停止以及四份
    report/archive。clean g1 recursive 为 0 gain/2 harm；no-recursive 为 1 gain/0 harm 但 lower bound
    -0.0176，均未 promotion。两臂 incumbent 仍为空，sealed/test 访问为 0；不生成空 control claim。
27. 已诊断 no-rec g2 的 evidence 语义缺口：一次 Ruoli 503 `provider_model_unavailable` 打开 circuit，
    之后 8 个 candidate skip，合计 9 invalid pair，却被旧 lifecycle 计为第二次普通 non-promotion，
    report/score 仍显示 claim-eligible/valid。下一步一次性关闭该分类缺口并版本化 v3.6 contrastive
    trigger learning；不新增 evaluator gate，不改 promotion/split/model/budget，旧 rows 不复用。
    有 retained validation gain 后再做 family-out，最后才增加 multi-clade/evaluator mutation；
28. 已完成 `01608e1e` 与当时的 254/254 离线回归：v3.6 manifest、success controls、exact
    contrastive selection、invalid terminal non-claim、mismatch-safe replay、pair diagnostics、
    legacy/v3.6 report schema 隔离和 freeze 重算均已落地。真实 v3.5 recursive report 可按旧
    schema 解析，污染的 no-recursive report 现因 invalid evidence 明确拒绝；
29. 已运行 v3.6 serial diagnostic：clean lock、86/86 prewarm、38/38 valid train、7 success、
    31 residual；单一 root 的 train activation precision 为 26/27，success false positive 为 1。
    仅完成 2/16 validation pairs（4 个 valid 0→0 trial），第 5 个 trial 中断，无 report/archive/
    promotion/freeze/family-out/sealed。随后一次性版本化 v3.7：五个在线 phase 的跨题 worker
    统一 1→6，invalid retry worker 仍为 1，同题 off/on 仍串行；不新增 gate，不改 evaluator、
    learning/promotion、route、split 或预算，且不复用 v3.6 rows；
30. 已运行 v3.7 six-worker fresh root：clean lock、86/86 prewarm 均通过，首批六个 train
    请求全部返回 `provider_rate_limit`，一次 circuit open 后其余 30 个 request slot 本地跳过。
    无 valid training bundle、proposal、report、archive、family-out 或 sealed access。v3.8 只把五个
    online phase 的 worker 从 6 改为 2，其他字段归一化后完全不变；从新 lock/root 开始，
    不拼接 v3.7 rows，也不新增 gate；
31. 已运行 v3.8 two-worker fresh root：完成 16 个 valid train rows 后，两条同时在途请求均返回
    `provider_model_unavailable`，既有 circuit 跳过其余 20。无 valid bundle/proposal/report/archive/
    sealed。v3.9 采用旧快版真正的两级并发结构：6 个题级 worker 共用 1 个进程级在线 agent
    semaphore，slot 只包围 `docker exec ... codex exec`；docker run、准备、离线 verifier 不取 slot。
    slot policy/count 进入 protocol/lock/plan/fairness/freeze，异常 finally 释放；不新增评分 gate；
32. 已完成 `8d862e8f` 与 289/289 离线回归，v3.9 clean lock 和 86/86 prewarm 均通过。
    随后两个 fresh root 都在第 0 条 benchmark trial 前停止：health probe 的两次 transport retry
    均为 HTTP 503，`skilllearn_trial_started=0`。其后 10 次低频恢复探针仍失败。当前唯一 blocker
    是 Ruoli route availability；不再版本化 worker、gate、retry 或 evaluator。恢复后从新 clean root
    启动，不复用任何 v3.7-v3.9 失败行；
33. 已完成 v3.9 lower-cost credential fresh root：clean lock、86/86 prewarm、38/38 valid train、
    56/56 actual trials、56/56 slot acquire/release，maximum online concurrency=1，0 provider/circuit/
    infra/budget/mismatch failure。recursive 两代均只激活 1/16 validation，candidate/raw 都为 3/16、
    0 gain/0 harm；no-recursive 两代 root 均在 train-only static audit 被拒绝。四份 report/archive
    完整落盘但两臂 `incumbent_id=null`，sealed/test 未访问。当前 blocker 已从 transport 转为
    candidate diversity/configuration coverage；下一步不新增或放宽 gate，不 freeze 空 incumbent；
34. 已实现 v3.10 bounded pre-gate revision 与 309/309 离线回归：单次 root response 必须 exact 3，
    三者在 failed train rows 上的 activation signature 两两不同；少/多/重复均 typed terminal，0 retry。
    proposal response budget 由 protocol 固定为 8,000 tokens。selection 只用 train labels，先最小化
    `ceil(existing minimum_activation_rate × distinct train families)` 的 capped coverage deficit，再比较
    exact precision、success false positives、failure support 与复杂度；同一 objective 也进入 repair
    request。model/provider、6×outer/1×model 调度、offline evaluator、promotion thresholds、split、
    action/network budget、retry、controls 与 sealed policy 均未改变。v3.9 rows 不复用，须新 lock/root。
35. 已完成 v3.10 fresh root：clean lock、86/86 prewarm、38/38 valid train（6 success / 32 residual）、
    56/56 actual trials、16/16 valid pairs、0 provider/infra/budget/mismatch。第一代 exact 3 全部 static
    pass，coverage-first 选中 2-family/6-of-6 failure root并在 validation 激活 2/16，但 candidate/raw
    都为 3/16、0 gain/0 harm。第二代 recursive/no-rec 各自收到 transport/JSON/exact-count 成功的
    三候选 response，却都只有一个 distinct activation signature，旧合同 terminal reject；两臂
    proposal failure non-claim、`incumbent_id=null`、sealed=false；
36. 已实现并运行 v3.11 bounded actionability revision：38/38 valid train（5 success / 33 residual），
    exact-three 得到 3 个 distinct activation signatures；2 个 root 静态通过。选中的 court policy 在
    validation 激活 1/16，实际改变 trace/PDF 并将 action starts 从 66 降到 16，但 task success 仍
    0→0；总计 raw/candidate 均 4/16、0 gain/0 harm。no-rec 又因一个未激活 raw poster trial 超过
    冻结 64 MiB 而 terminal non-claim。recursive repair transport/JSON 成功，却返回 root-shaped
    `hypotheses`；调用链复核确认真实 repair payload 没有 batch contract，但也没有 versioned singular
    response override，整代在 validation 前 non-claim。
    两臂 `incumbent_id=null`、sealed=false，不 freeze、不跑 controls/family-out/HippoRAG；
37. 已实现 v3.12 bounded repair-scope revision：repair 保留 `train_coverage_objective`，并以
    `single_candidate_excludes_root_batch_contract_v1` 绑定 top-level one-object/`hypothesis` response、
    system prompt、plan 与 freeze；若调用方意外提供 root batch contract，`revise()` 防御性删除。
    root exact-three、model/provider、6×outer/1×model 调度、offline evaluator、promotion gate、split、
    action/64 MiB network budget、retry、controls 与 sealed 全部不变；须新 clean lock/root，v3.11 rows
    不复用。
38. 已完成 v3.12 fresh root：clean commit `9c692b2d`、86/86 prewarm、56/56 actual external trials
    全部 valid；38 train 为 8 success / 30 residual，最大网络流量 35,070,000 bytes，0 provider/infra/
    action/network/budget/mismatch。两代 exact-three 均 3/3 static pass且三种 distinct signature，
    但每个 root 都只覆盖一个 train family；selected court treatments 在同一 held-out item 上把 raw 9
    actions 改为 32/43 actions，仍均 0→0。两臂每代都是 1/16 activation、raw/candidate 4/16、
    0 gain/0 harm，最终 archive 字节相同且 `incumbent_id=null`。12 个 static node 全通过，故本轮
    repair request=0，不能声称 full-run singular repair path 已实跑。
39. 同一旧 runner 在 `selected_candidate_available=false` 后仍无条件进入 controls，生成空
    `promoted_v2`/no-rec program set并启动 8 个 trial；立即中止后仅 2/96 records 落盘且两条都因
    缺 action-budget receipt invalid，无 report、family-out、HippoRAG、sealed/test access。该批已有
    machine-readable diagnostic-only/no-reuse marker。现已统一 phase invariant：all-development 无真实
    recursive incumbent 即正常结束，paper freeze 拒绝空 archive，control consumer 也拒绝旧空 receipt。
    345/345 离线回归通过；没有新增或放宽 performance gate。
40. 已实现 v3.13 complementary program-set revision 与 375/375 离线回归。exact-three static-valid
    roots 的最多 7 个非空子集只在 TRAIN 上枚举；排序依次为 union precision、capped family deficit、
    success false positive、overlap、bundle size、failure support、complexity 与 canonical set hash。
    因此低精度成员不会为凑 coverage 被强塞，也没有 minimum bundle size。确定一个 delta set 后才
    进行一次 paired validation。SkillLearn runner 分别绑定 delta/full/per-item matched set，任一新成员
    命中才执行一次 policy-on，否则严格 alias baseline；program-set replay 对顺序不敏感但区分 `{A}`
    与 `{A,B}`。archive rejection 只拒绝 bundle node、成员留 shadow；report/freeze 强校验 selected IDs、
    treatment-set hash、baseline union、node/status。Promotion 仍是
    `evaluator_owned_paired_validation_v2`，成员自约束保守聚合，所有 protocol 数值阈值、model/provider、
    6×outer/1×model、offline evaluator、split、64 MiB fuse、retry、controls 与 sealed 均未改变。
    v3.12 rows 不复用。
41. v3.13 正式 development 已完成：第一次 partial invocation 被独立标为 diagnostic-only，冻结的
    lock、86/86 prewarm 和 smoke 因 Plus/Pro credential 都服务同一 `gpt-5.4-mini` route、model/provider
    identity 不变而保留，正式 development 使用新的
    events/work tree。76/76 trials 全 valid 且均 attempt-1：38 train-off、32 validation-off、6 validation-on，
    0 provider/infra/retry/action/network/mismatch，最大 69/100 actions 与 62,200,000/67,108,864 bytes。
    每套 bundle 都把 poster+court 两个 TRAIN-perfect 成员合并并激活 2/16；6 个 on 与对应 off 全失败，
    四代决策均 0 gain/0 harm。两臂 stop=`consecutive_non_promotion_limit`、`incumbent_id=null`，无 freeze/
    controls/family-out/HippoRAG/sealed/test。recursive repair path 已无错误地实跑。G2 发现 cross-arm
    baseline cache 未共享；no-recursive 的三成员 7/7 subset 又被 capped-family/size-first objective 排第四。
42. 已实现 v3.14 的两项有限修订。`behavior_identical_shared_validation_baseline_arm_replay_v2`
    以 baseline behavior/treatment 与冻结 task/runtime/fairness identity 为 key，去掉 challenger pair 元数据，
    在 recursive/no-recursive 与多代间共享只写一次的 valid policy-off cohort；invalid 不入缓存、冲突不覆写。
    `train_contrastive_complementary_family_support_bundle_precision_first_v2` 保留 precision、capped deficit、
    success false positive、overlap 的 leading order，仅在这些项相同时把 actual family count 与 failure support
    放到 bundle size 前。仍只固定一个 TRAIN set、只做一次 held-out paired validation；model/provider、
    evaluator、promotion thresholds、split、fuse、scheduler、controls/sealed 均不变。由于 protocol/code
    identity 变更，v3.14 使用新 lock、cache-only prewarm、smoke、run root；已有 model/runtime image 与
    依赖没有重下。
43. v3.14 live 已完成并触发停止 selector 的预设条件。commit=`2229d7af`，lock claim eligible，prewarm
    86/86，smoke 8/8 valid；formal development 62/62 attempted、61 valid / 1 hard-budget invalid，0 provider/
    model/slot/action/mismatch，max valid actions=55/100。38 条 train 全 valid（7 success / 31 residual）。G1
    selector 选中 court/dependency/poster 三成员 set，TRAIN 为 7/7、3-family、0/7 success-FP，held-out
    activation=3/16；recursive 两个 valid activation 与 no-recursive 三个 activation 都 0→0，no-recursive
    G2 poster singleton 的一个 activation 也 0→0。合计 7 个 policy-on 全失败。recursive court raw 使用
    68,660,000/67,108,864 bytes，primary report=`invalid_counterfactual_evidence` non-claim；no-recursive
    report 机械上 claim eligible、两代 non-promotion。valid baseline rows 共 replay 31 次，但 invalid key
    因“不入 cache”在 no-recursive 又执行一次，说明 terminal invalid 还没有跨 consumer 传播。两份 archive
    `incumbent_id=null`，无 freeze/controls/family-out/HippoRAG/sealed/test。下一步转向 action content：当前
    directives 没有补充实际 HEX、离线漏洞记录位置或新的表单操作，只是更清楚地复述 instruction。
44. 已实现并完成 clean live：v3.15 提交 `696a2954` 与 453/453 offline regression 通过。它把 material action
    delta 作为 TRAIN-only request-local prompt 与 audit-only 诊断，并以 allowlisted public-environment/
    policy-off trace facts 补充 proposal context；不读取 validation/test/verifier/solution，不允许 proposal
    外部工具，不把 audit 变成 reject/retry/repair/selection/promotion gate。terminal-invalid replay v3 在
    frozen same-request retry identity 完成后共享 non-evidence tombstone，report/freeze 再绑定两臂共同的
    首代 checkpoint 与 action-profile count/set hash。正式 root 通过 clean lock、86/86 cache-only prewarm
    与 smoke，完成 57/57 valid actual trials、8/8 model calls；TRAIN 为 6 success / 32 residual，max online
    concurrency=1，0 provider/infra/action-budget/network-cap/pair-mismatch。recursive G1/G2 都是 1/16、
    candidate/raw 4/16、0 gain/0 harm；no-rec G1 static reject，G2 为 1/16、0 gain/0 harm；32 次 baseline
    replay 零执行。两臂均 claim-eligible negative、`consecutive_non_promotion_limit`、`incumbent_id=null`，
    sealed/test=false，无 downstream。
45. v3.15 的 13 个 candidate audits（9 roots + 4 repairs）中 7 个 material、6 个 restatement-risk；7 个
    material delta 全是 `exact_constant_or_mapping`，concrete local tool、artifact manipulation、environment
    primitive 全为 0。9 个 roots 又全部坍缩到 `anthropic-poster`，从 v3.14 G1 的 3-family/7-support 退回
    单-family/2-support。因此先授权一次 gate 前的 structural family-stratified proposal-only feasibility，
    不直接启动新 development，也不新增/放宽 promotion gate。
46. v3.16 提交 `6ad5c156` 用冻结 v3.15 TRAIN receipt 形成三个 singular family slots。3/3 logical proposal
    calls 完成，0 source-agent/backend/evaluator/validation/test/sealed access，但 9 项 feasibility 中 6 项失败；
    没有 benchmark trial。v3.17 提交 `4f94e613` 只作最后一次结构修订：exact trigger、empty anti-trigger、
    deterministic reusable artifact 和固定 workflow blueprint。新结果通过 8/9：support=2/2/3、3/3 concrete
    tool、2/3 artifact manipulation、0 restatement/self-block；第三候选仍绑定 2 个 failed TRAIN primitives，
    故 `diagnostic_passed=false`。一次 retryable disconnect 已恢复，不改变 semantic negative 结论。
47. free-text family-slot 路线按预设停止。离线重建显示第三 slot 的两个 failed primitives 是通用 `file`/
    `python` executable，暴露了 failed-command 共现不等于 causal inadmissibility，且模型只获 count/hash 无法
    满足未知逐值 exclusion 的表示矛盾。下一 workstream 必须换成 typed operator/capability grammar 或
    artifact-operation graph，并以 causal action-span evidence 定义不可表达项；只允许一次 preregistered
    feasibility decision。没有真实 incumbent 时继续禁止 controls/family-out/HippoRAG/sealed。
48. causal action-span extractor、closed typed operator/artifact graph、opaque recipe-only selection、harness-owned
    materializer 与 single-decision lock/preregistration 已实现并冻结为 commit `b03c643a`。前置结构复核得到
    38/38 complete trace、429 allowlisted occurrences、70 failed、208 discarded、655 action starts、max 61/100。
    该路径使用 stored offline TRAIN outcomes、本地 contract validation 并哈希 unit-test source，但有 0 live
    model/task-backend/evaluator invocation，且未访问 validation/test/sealed split 或 verifier content。该表示只
    闭合 proposal selection；现有 lowering 仍是 prompt directive，非 restricted executor。
49. 唯一一次正式离线 decision 已按预注册命令完成：9/9 predicates PASS，decision hash
    `79acda9b9e393330b8418e5fea15f176236edf8ecf802d310d73862710ba8bfc`，report hash
    `aa1033429980cfc5881aa6b3ccf25609c3d80ce0514c1dc37d1188354789797d`；随后 `--verify-existing`
    对 report、9-event ledger 与 completed lock 精确复验通过。3/3 target graph/program materialize、9/9
    tamper probes fail closed、raw primitive/locator disclosure 均为 0，70 个 failed spans 中 63 个存在后续
    scope-matched recovery。结果收据见
    [`skilllearn_typed_operator_feasibility_result_v1.json`](../manifests/skilllearn_typed_operator_feasibility_result_v1.json)。
    该 PASS 只使 separately frozen typed-selection integration diagnostic freeze-eligible；它不验证 capability
    implementation、restricted executor 或 benchmark gain，也不授权 development。

这比立刻扩展 archive 或继续补 HLE source span 更能降低研究风险。

## 十、建议的实验协议与 claim ladder

### 10.1 主 benchmark

当前最合适的主战场仍是 SkillLearnBench：

- instance holdout：38 train / 16 validation / 32 sealed test；
- family out：48 train / 11 validation / 27 sealed test；
- HLE：只作冻结的 external transfer/stress slice。

如果选择 86-item infrastructure subset，必须在任何模型调用前冻结新 manifests、重新计算
split counts，并说明 9 项排除只由 verifier/toolchain availability 决定，而非 task outcome。

### 10.2 必要 controls

最低矩阵应包含：

```text
raw_no_skill
static_generic_v2              # fixed, no learning
v2_no_recursive_repair         # same evolution budget, repair disabled
promoted_v2                    # frozen incumbent from recursive loop
skilllearn_b1_sonnet           # upstream static reference
human_authored                 # upper reference, not budget matched
```

raw、static、no-recursive 与 promoted arms 必须共享 model、provider policy、step budget、
runtime、verifier isolation 和 invalid-row policy。外部参考与 human upper reference 不应被
伪装成 budget-matched primary controls。

### 10.3 两个不同的因果问题

1. **same-item paired off/on**：相同 item、runtime 和 evaluator 下，candidate treatment
   是否造成 gain/harm；
2. **prospective transfer**：在未见 instance/family 上，router 是否在看不到 outcome 时
   正确激活，并保持净收益。

第 1 个回答局部因果 effect；第 2 个回答假设是否可复用。只做第 1 个不能证明 continual
learning，只做第 2 个而没有 matched controls 又无法归因。

### 10.4 预注册指标

- task success / executable reward；
- gain、harm、net gain 与 exact McNemar；
- effect LCB 与 item-clustered interval；
- prospective activation rate、evidence-valid precision 与 abstention；
- behavior-changing repair count；
- cost ratio、token、latency 与 model calls；
- invalid/error rate、provider/budget/runtime mismatch；
- archive retention、duplicate rate、forgetting 与 cross-family transfer；
- 多比较 Holm correction 与预注册 early stopping。

“hypothesis proposal precision”不能只按 schema pass 定义；更可操作的定义是：候选先通过
train-only static contract，再在 prospective matched validation 中产生正净效应且不超过
harm/cost gate。train-side selection precision 是 failed activation / 全部 labeled train
activation；held-out causal activation precision 的分母则是 evidence-valid 实际激活，正例
来自独立 paired gain，而不是模型自评。两者不得混写。

### 10.5 claim ladder

| 层级 | 可声明内容 | 当前状态 |
|---|---|---|
| L0 wiring | schema、repair、off/on、guard、archive transition 的机械链路已连接 | 达到：当前完整离线 suite 540/540 通过；除 v3.15 的既有 wiring 外，v3.16/v3.17 TRAIN receipt reconstruction、family-slot production path、redacted live ledger、失败结果复核，以及 typed all-item budget/action-span/canonical-graph/semantic-failure/decision-lock boundary 均有覆盖 |
| L1 mechanism live | 真实外部任务中 proposal/repair/treatment/gate 全链路完成 | 达到且最新 benchmark evidence 仍是 v3.15 clean negative：57/57 valid、8/8 model calls、0 infra/fairness error；v3.16/v3.17 未运行 task trial，不提升 L1，也无 promotion claim |
| L2 validation learning | clean held-out validation 上有可晋级净收益 | 未达到 |
| L3 prospective generalization | frozen incumbent 在 unseen instance/family 上保持收益 | 未达到 |
| L4 self-evolution | 多代 retained improvement，且 recursion ablation 有因果贡献 | 未达到 |
| L5 evaluator co-evolution | anchor-guided evaluator replacement 与 selective erasure 改善搜索 | 未达到 |

## 十一、什么才算“真正自我提出并递归验证假设”

以下条件需要同时满足：

1. 候选不是人工预写的唯一答案，而是系统只从 train evidence 提出；
2. candidate selection 同时利用失败与成功对照，不能只奖励 failure support；
3. 假设被编译为当前 backend 能强制或明确审计的程序；
4. activation 在 outcome 前决定，并实际改变 execution treatment；
5. 同一 item/runtime/evaluator 有 policy-off/on paired counterfactual；
6. promotion gate 完全由冻结 protocol/evaluator 所有，candidate 不能放宽阈值；
7. promotion 不读取 sealed test；
8. 通过的程序进入 archive，并在未来未见题上被 prospective router 调用；
9. 失败程序能降级、停用或归档；evaluator epoch 改变时只使旧依赖证据失效；
10. recursive repair 的收益用共享 root/evidence 的 no-recursive arm 做因果消融；
11. 最终提升能归因到该程序，而不是额外预算、fallback、provider 或重采样；
12. 至少一次真实 promotion 改变下一代 incumbent，并在后续任务上保留净收益。

在满足这些条件前，“自我提出并递归验证”仍应被称为研究机制或 harness，而不是已经
证实的 self-evolving capability。

## 十二、最终结论

旧 Assumption Agent 的主要问题不是“没有足够多假设”，而是假设没有稳定编译成可
执行、可路由、可反事实验证并可跨题保留的 policy。legacy 优化的是“怎样更复杂地
回答 HLE”，而不是“哪些假设值得在未来任务中保留，以及它们是否因果性地改善行为”。

`reconstruction_v2` 已经完成了重要转向：它把三层 hypothesis、paired evaluation、split
guard、archive 和 evaluator epoch 做成了清晰的小型系统。这使研究问题第一次真正可
证伪，也比继续给 legacy HLE monolith 加规则更有价值。

本次已经关闭四个会让后续结果先天不可解释的 P0：candidate 不能控制 promotion
及格线；外部 backend 不再把 prompt/verifier/fallback 声明伪装成 typed/observed 事实；
86-item offline-ready manifests 通过 readiness，且 v3.4 v4 prewarm 为 86/86；verifier receipt 现在绑定 proxy
实际执行的 runtime profile/command，完整 CTRF 的任务失败不再被误标成 infrastructure
failure。新 GPT Pro route 证明 Ruoli 模型调用在该 v3.3 batch 中可用，但不能外推持续稳定；离线 evaluator 从未需要
替换为 online evaluator。

v3.3 在本次 batch 中排除了原先的 64 MiB 直接阻塞：38/38 train 的最大流量为 40.6 MB，
`video-object-counting-1` 从 71.1 MB invalid 降为 19.69 MB valid；没有 provider 或 hard-cap
error。代价不是修改 evaluator、子集或 promotion gate，而是把低 reasoning / verbosity
与更早的本地 history compaction 作为 protocol-owned agent treatment。canary/full 的
1.47/19.69 MB 差异表明重复稳定性仍未建立，不能把单次 batch 外推成稳定完成。

最近的两个 execution blocker 已在零模型层定位并修复。`web_search` 不是 provider/model
注入，而是 Codex 0.144.1 对旧 boolean config 的兼容性 no-op；canonical 顶层 disabled 已由
真实 wire 阴性/阳性对照确认。nominal `max_steps` 也不再被称为 semantic turn，而是冻结为
可流式观察的 `codex_action_start_v1`，由容器内 supervisor 终止 PGID、清理 dedicated-container
基线后的全部 live task 并生成 receipt。
这不是通过重试或放松 auditor 把 v3.3 invalid 洗成 performance evidence；v3.3 的 37 条
valid observation 仍全部不可复用，sealed 仍未触碰。

因此当前距离目标的第一段不再是修 gate 或证明单次 API 连通性。v3.4 clean lock、新 runtime
cache、86/86 cache-only prewarm 和 max2 v5 action-budget canary 均已通过；v5 同时证明 actual
wire 无 web、budget receipt valid、全部 agent task 已退出且本地 verifier 在 agent 后执行。
fresh development 也已真实启动，但冻结四并发在 17 条有效离线结果之后触发四个 429，随后
17 个 slot 被 circuit 本地跳过。API credential 和 bounded inference 可用，持续四并发容量不可用；
online evaluator 无法修复该问题。

v3.5 的第三个 fresh root 已在同一 invocation 内取得 38/38 all-valid train，并首次完成 proposal、
真实 repair、双臂 paired validation/promotion decision、两代停止与四份 report/archive。clean g1
给出的最强信号不是 promotion，而是 selection 反例：recursive 最大-support repair 为 0 gain/2 harm，
no-recursive 保守 root 为 1 gain/0 harm但 lower bound 尚为负。g2 随后被一次 Ruoli 503 与 circuit
skip 污染；旧 lifecycle 又把 9 invalid pair 当成普通 non-promotion 并写出 valid score。因此这份
run 只能保留为 L1 机械闭环与 contrastive-learning 动机，不能作为 clean full-development claim，
也不能 freeze 空 incumbent 或进入 sealed。identity/response/invalid-evidence 修复都不是放宽评分 gate；
archive 冲突硬拒绝、promotion contract 和 evaluator 均未改变。v3.6 live 已从零完成 38/38
contrastive train 并进入真实 paired validation，但串行执行只完成 2/16 pairs 后主动终止，因而
没有完整 development claim。v3.7 的固定六路被首批 6/6 429 否决，v3.8 的固定两路又在
16 valid 后被 2 个同时 503 否决。v3.9 随后用 6 个题级 pipeline 配 1 个共享在线 agent slot，
在 2026-07-13 首次完成 clean full development：38/38 valid train、56/56 actual trials、0 provider/
infra/mismatch failure，并写出四份完整 report/archive。这个结果不是 promotion：recursive 两代
都只激活 1/16 validation，candidate/raw 均为 3/16、0 gain/0 harm；no-recursive roots 均在
train-only static audit 被拒绝，两臂 `incumbent_id` 都为空。

因此距离目标最近的缺口不再是 transport、offline evaluator 或更多 promotion gate。v3.10 已证明
exact-three/coverage-first 能把 activation 从 1/16 提高到 2/16，却不能产生 gain；第二代又证明用文本
要求模型满足 host 事后计算的 pairwise signature 不是可靠 response contract。v3.11 随后证明新的
imperative action/lowering 确实改变 agent trace、PDF 和 action cost，但唯一激活仍为 0→0；它同时因
repair model 返回 root-shaped batch envelope 而没有完成 recursive quality 检验。调用链确认 batch
contract 实际未进入 repair，因此 v3.12 只新增 protocol-bound singular response override，保留
exact-three root、8,000-token budget、train-only coverage selection 和 audit-only signature policy，
也没有放宽 evaluator/promotion 或修改 split/fuse/retry/sealed。

v3.12 的 clean 结果排除了“先等 repair 触发”作为下一步：所有 static node 直接通过，repair=0，
而 exact-three 已连续两代产生三个互补、高精度、零 success-FP 的单-family roots。真正的信息损失发生
在 selector 把其中两个丢弃。v3.13 已把这一点一次性版本化为 train-only complementary program-set，
且 live 证实 program-set routing、per-item match、nonactivation alias 和 G1 cross-arm replay 都正确。
正式轮 76/76 valid，但三套 bundle 的 6 个 policy-on 全失败；两臂四代均 activation 2/16、0 gain/0 harm、
无 incumbent。故“精确命中 TRAIN failure”只能证明 trigger precision，不能证明 action utility。

v3.14 的机制修复不是再加 gate：它共享 recursive/no-recursive 与多代的 valid raw baseline cohort，
并在 precision、capped deficit、success-FP、overlap 相同的 TRAIN-only subsets 中把实际 family count 与
failure support 放到 bundle size 前。live 已证明后者按设计选择 7/7、3-family set，activation 也从 2/16
增至 3/16；但 7 个 policy-on 仍全失败、0 gain/0 harm。因此 selector 迭代到此结束。valid baseline
evidence 的 31 次 replay 也证明共享路径有效；一条 hard-cap invalid 因不入 cache 又被另一 arm 执行，
说明未来若还做 cross-arm claim，应共享 terminal invalid memo，而不是重新采样。该修复属于 evidence
identity，不改变任何 promotion threshold。

下一 workstream 必须学习**可执行 action delta**，而不是更宽 trigger 或更多 gate。当前三个 G1 action
program 虽然语法清楚，却分别只说“使用品牌色”“收集权威离线漏洞记录”“只填写必要表单字段”；它们
没有给出任务缺失的实际 HEX、可访问数据源/路径或新操作步骤。TRAIN failure precision 只能证明这些
instruction 出现在失败样本中，不能证明 directive 提供了 baseline agent 原本不知道的知识。v3.15
提交 `696a2954` 已把 instruction-restatement 与 material executable knowledge 的区别写入 request-local
prompt，并用受限 TRAIN environment/policy-off trace profile 提供具体但非 oracle 的设计上下文；audit
仍严格为非评分、不能改变 response lifecycle 或 promotion。terminal-invalid memo 又关闭了 v3.14 的
跨臂重采样归因缺口，paired report/freeze 则绑定共同 action-profile provenance。正式 live root 已通过
clean lock、86/86 cache-only prewarm 与 smoke，并以 57/57 valid actual trials、8/8 model calls、最大
在线 agent 并发 1、0 provider/infra/budget/network/pair error 完成。38 条 TRAIN 中有 6 success / 32
residual；16 条共享 baseline 支撑两臂/两代共 32 次零执行 replay，实际只运行 3 个 candidate-on。
recursive G1/G2 均为 activation 1/16、candidate/raw 4/16、0 gain/0 harm；no-recursive G1 static reject、
G2 仍为 1/16 与 0 gain/0 harm。两臂 claim-eligible 只意味着 clean negative evidence 成立；二者都
`consecutive_non_promotion_limit`、`incumbent_id=null`、sealed/test=false，完全没有 downstream。

action audit 解释了为什么机制更干净却没有更强：13 个 candidates（9 roots + 4 repairs）中 7 个有
material delta、6 个有 restatement risk，但 7 个 material 全部只是 `exact_constant_or_mapping`；concrete
local tool、artifact manipulation 与 environment primitive 均为 0。九个 roots 又全部坍缩为
`anthropic-poster`，从 v3.14 G1 的 3-family/7-support 退到 single-family/2-support。v3.16/v3.17 因而先把
structural family stratification 与 artifact grounding 放进 proposal-only screen，而不是再跑完整 development。
v3.17 确实把三项 proposal 分到不同 family，取得 support 2/2/3、3/3 concrete local tool、2/3 artifact
manipulation、0 restatement/self-block；但第三项仍绑定 `file`/`python` 两个 failed-command primitives，
因此 8/9 pass 仍是整体 fail。0 backend/evaluator/benchmark trial 是正确的 spend-control 结果。

这也终止了“继续写更强 prompt”的路线：failed command 中出现通用 executable 并不证明它在因果上应被
禁用，而只把 count/hash 给模型又无法要求其避开未知具体值。causal action-span taxonomy 与 closed typed
operator/artifact graph 已实现，并在唯一一次正式离线 decision 中 9/9 PASS、精确复验 PASS。这个结果不再
触发任何 prompt/gate patch；它只使一个另行冻结的 typed-selection integration diagnostic 有资格运行，
因为当前 production selection 尚未接入 proposer/evolution，runtime 仍是通用 prompt-directive agent。

只有 feasibility 通过、typed-selection integration 经独立冻结诊断验证、随后新的 clean development 产生
retained validation gain 和真实 incumbent，才按既定顺序进入 freeze、完整 controls、family-out、sealed test；
在此之前不跑 HippoRAG/raw transfer，
更不谈 multi-clade 或 evaluator co-evolution。v3.12 空 freeze/partial-control rows、v3.14 mixed-claim rows 与
v3.16/v3.17 proposal-only artifacts 都不能拼入 performance evidence；v3.15 两臂虽 claim-eligible，archive
仍为空，同样不得绕过 phase prerequisite。
最诚实的论文级表述是：

> **显式 HypothesisProgram 是一个有希望、可能更易归因的 self-evolution 搜索表示；
> v2 已证明协议所有权、离线 evaluator 和学习环 wiring 可运行，但尚未证明它在冻结、
> 干净的外部 benchmark 上产生稳定净收益，更未证明 Red Queen 式多谱系或 evaluator
> 共演化。**

## 附录 A：关键证据索引

- legacy 代码：[`assumption_os/`](../../assumption_os/)
- legacy 自我演化评估：
  [`codex_gpt_advice_assessment_20260707.md`](../../reconstruction/md/codex_gpt_advice_assessment_20260707.md)
- self-evolution bundle：
  [`reference/self_evo_continual_20260707/`](../reference/self_evo_continual_20260707/)
- RQGM PDF：
  [`The Red Queen Gödel Machine`](<../reference/The Red Queen Gödel Machine Co-Evolving Agents and Their Evaluators.pdf>)
- v2 architecture：[`ARCHITECTURE.md`](../ARCHITECTURE.md)
- v2 benchmark protocol：[`BENCHMARK_PROTOCOL.md`](../BENCHMARK_PROTOCOL.md)
- v2 current status：[`STATUS.md`](../STATUS.md)
- last completed claim-bearing benchmark protocol：
  [`skilllearn_paper_protocol_v3_15_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_15_ruoli_gpt54mini.json)
- latest failed proposal-only diagnostic protocol：
  [`skilllearn_paper_protocol_v3_17_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_17_ruoli_gpt54mini.json)
- immutable v3.16 proposal-only diagnostic protocol：
  [`skilllearn_paper_protocol_v3_16_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_16_ruoli_gpt54mini.json)
- immutable v3.10 proposal-diversity diagnostic protocol：
  [`skilllearn_paper_protocol_v3_10_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_10_ruoli_gpt54mini.json)
- immutable v3.9 clean negative-development protocol：
  [`skilllearn_paper_protocol_v3_9_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_9_ruoli_gpt54mini.json)
- immutable v3.8 two-worker capacity diagnostic protocol：
  [`skilllearn_paper_protocol_v3_8_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_8_ruoli_gpt54mini.json)
- immutable v3.7 six-worker capacity diagnostic protocol：
  [`skilllearn_paper_protocol_v3_7_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_7_ruoli_gpt54mini.json)
- immutable v3.6 contrastive/serial diagnostic protocol：
  [`skilllearn_paper_protocol_v3_6_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_6_ruoli_gpt54mini.json)
- immutable v3.5 execution/learning diagnostic protocol：
  [`skilllearn_paper_protocol_v3_5_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_5_ruoli_gpt54mini.json)
- immutable v3.4 execution diagnostic protocol：
  [`skilllearn_paper_protocol_v3_4_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_4_ruoli_gpt54mini.json)
- immutable v3.3 execution diagnostic protocol：
  [`skilllearn_paper_protocol_v3_3_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_3_ruoli_gpt54mini.json)
- immutable v3.2 diagnostic protocol：
  [`skilllearn_paper_protocol_v3_2_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_2_ruoli_gpt54mini.json)
- immutable v3.1 diagnostic protocol：
  [`skilllearn_paper_protocol_v3_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_ruoli_gpt54mini.json)
- frozen offline-ready manifests：
  [`instance holdout`](../manifests/skilllearnbench_instance_holdout_offline_ready_v1.json)；
  [`family out`](../manifests/skilllearnbench_family_out_offline_ready_v1.json)
- version-controlled readiness evidence：
  [`skilllearn_offline_readiness_receipt_v1.json`](../manifests/skilllearn_offline_readiness_receipt_v1.json)
- local ignored diagnostics（非 clone 中的主证据）：
  [`offline verifier matrix`](../artifacts/offline_verifier_matrix_offline86_20260711_v1/matrix.json)；
  [`86-item runtime prewarm receipt`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/development_prewarm.json)；
  [`mechanism smoke`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/smoke_recursive.report.json)；
  [`full-development fail-closed events`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/development_recursive.events.jsonl)；
  [`v3.1 clean-rerun cap recurrence`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini_rerun01/development_recursive.events.jsonl)；
  [`v3.2 claim lock`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini/protocol_lock.json)；
  [`v3.2 86-item prewarm`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini/development_prewarm.json)；
  [`v3.2 provider-capacity failure`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini/development_recursive.events.jsonl)；
  [`GPT Pro Codex canary`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini/gpt_pro_transport_canary/report.json)；
  [`gptpro01 receipt false-negative run`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini_gptpro01/development_recursive.events.jsonl)；
  [`gptpro03 final 64 MiB hard-cap run`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini_gptpro03/development_recursive.events.jsonl)；
  [`video-object-counting-1 failed trace`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini_gptpro03/development_recursive/upstream_trials/no_skill/video-object-counting/video-object-counting-1/v2_policy_off_66599fccf924efd4c6/agent/codex.txt)；
  [`v3.3 claim lock`](../artifacts/paper_primary_v3_3_offline86_ruoli_gpt54mini/protocol_lock.json)；
  [`v3.3 86-item prewarm`](../artifacts/paper_primary_v3_3_offline86_ruoli_gpt54mini/development_prewarm.json)；
  [`v3.3 video-1 canary`](../artifacts/paper_execution_policy_v3_3_video_object_counting_1_canary01/report.json)；
  [`v3.3 full-train events`](../artifacts/paper_primary_v3_3_offline86_ruoli_gpt54mini/development_recursive.events.jsonl)；
  [`forbidden web-search trace`](../artifacts/paper_primary_v3_3_offline86_ruoli_gpt54mini/development_recursive/upstream_trials/no_skill/offer-letter-generator/offer-letter-generator-1/v2_policy_off_a99904ddf5496bed16/agent/codex.txt)；
  [`v3.3 video-1 valid trace`](../artifacts/paper_primary_v3_3_offline86_ruoli_gpt54mini/development_recursive/upstream_trials/no_skill/video-object-counting/video-object-counting-1/v2_policy_off_26f7d1bd8b10776c43/agent/codex.txt)；
  [`long temperature-3 trace`](../artifacts/paper_primary_v3_3_offline86_ruoli_gpt54mini/development_recursive/upstream_trials/no_skill/temperature-simulation/temperature-simulation-3/v2_policy_off_970cfa8b6418033bd2/agent/codex.txt)；
  [`v3.4 zero-model wire probe`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/diagnostics/codex_model_only_wire.json)；
  [`v3.4 runtime preparation`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/codex_runtime_preparation.json)；
  [`v3.4 claim lock`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/protocol_lock.json)；
  [`v3.4 86-item prewarm`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/development_prewarm.json)；
  [`max2 v1 pre-model PATH failure`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/diagnostics/max2_offer_letter_canary_v1.json)；
  [`max2 v2 raw 503 diagnosis`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/diagnostics/max2_offer_letter_canary_v2.json)；
  [`max2 v3 classified provider blocker`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/diagnostics/max2_offer_letter_canary_v3.json)；
  [`max2 v4 host-permission diagnostic`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/diagnostics/max2_offer_letter_canary_v4.json)；
  [`max2 v5 passing action-budget canary`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/diagnostics/max2_offer_letter_canary_v5.json)；
  [`v3.4 four-worker provider-capacity failure`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/development_recursive.events.jsonl)；
  [`v3.5 38/38 train then repair-ID collision`](../artifacts/paper_primary_v3_5_offline86_ruoli_gpt54mini/development_recursive.events.jsonl)；
  [`v3.5 repairid01 38/38 train then malformed repair envelope`](../artifacts/paper_primary_v3_5_offline86_ruoli_gpt54mini_repairid01/development_recursive.events.jsonl)；
  [`v3.5 repaircontract01 recursive report`](../artifacts/paper_primary_v3_5_offline86_ruoli_gpt54mini_repaircontract01/development_recursive.report.json)；
  [`v3.5 repaircontract01 no-recursive contaminated report`](../artifacts/paper_primary_v3_5_offline86_ruoli_gpt54mini_repaircontract01/development_no_recursive.report.json)；
  [`v3.9 clean recursive report`](../artifacts/paper_primary_v3_9_offline86_ruoli_gpt54mini_outer6_model1_plus01/development_recursive.report.json)；
  [`v3.9 clean recursive archive`](../artifacts/paper_primary_v3_9_offline86_ruoli_gpt54mini_outer6_model1_plus01/development_recursive.archive.json)；
  [`v3.9 clean no-recursive report`](../artifacts/paper_primary_v3_9_offline86_ruoli_gpt54mini_outer6_model1_plus01/development_no_recursive.report.json)；
  [`v3.9 clean no-recursive archive`](../artifacts/paper_primary_v3_9_offline86_ruoli_gpt54mini_outer6_model1_plus01/development_no_recursive.archive.json)；
  [`v3.10 recursive report`](../artifacts/paper_primary_v3_10_offline86_ruoli_gpt54mini_outer6_model1_diverse_plus01/development_recursive.report.json)；
  [`v3.10 recursive archive`](../artifacts/paper_primary_v3_10_offline86_ruoli_gpt54mini_outer6_model1_diverse_plus01/development_recursive.archive.json)；
  [`v3.10 no-recursive report`](../artifacts/paper_primary_v3_10_offline86_ruoli_gpt54mini_outer6_model1_diverse_plus01/development_no_recursive.report.json)；
  [`v3.10 no-recursive archive`](../artifacts/paper_primary_v3_10_offline86_ruoli_gpt54mini_outer6_model1_diverse_plus01/development_no_recursive.archive.json)；
  [`v3.15 claim lock`](../artifacts/paper_primary_v3_15_offline86_ruoli_gpt54mini_outer6_model1_actiondelta01/protocol_lock.json)；
  [`v3.15 86-item prewarm`](../artifacts/paper_primary_v3_15_offline86_ruoli_gpt54mini_outer6_model1_actiondelta01/development_prewarm.json)；
  [`v3.15 recursive report`](../artifacts/paper_primary_v3_15_offline86_ruoli_gpt54mini_outer6_model1_actiondelta01/development_recursive.report.json)；
  [`v3.15 recursive archive`](../artifacts/paper_primary_v3_15_offline86_ruoli_gpt54mini_outer6_model1_actiondelta01/development_recursive.archive.json)；
  [`v3.15 no-recursive report`](../artifacts/paper_primary_v3_15_offline86_ruoli_gpt54mini_outer6_model1_actiondelta01/development_no_recursive.report.json)；
  [`v3.15 no-recursive archive`](../artifacts/paper_primary_v3_15_offline86_ruoli_gpt54mini_outer6_model1_actiondelta01/development_no_recursive.archive.json)；
  [`v3.15 action-audit/event ledger`](../artifacts/paper_primary_v3_15_offline86_ruoli_gpt54mini_outer6_model1_actiondelta01/development_recursive.events.jsonl)；
  [`v3.16 failed proposal-only report`](../artifacts/paper_primary_v3_16_offline86_ruoli_gpt54mini_outer6_model1_familyslots01/train_proposal_diagnostic.report.json)；
  [`v3.16 proposal-only event ledger`](../artifacts/paper_primary_v3_16_offline86_ruoli_gpt54mini_outer6_model1_familyslots01/train_proposal_diagnostic.events.jsonl)；
  [`v3.17 failed proposal-only report`](../artifacts/paper_primary_v3_17_offline86_ruoli_gpt54mini_outer6_model1_familyslots02/train_proposal_diagnostic.report.json)；
  [`v3.17 proposal-only event ledger`](../artifacts/paper_primary_v3_17_offline86_ruoli_gpt54mini_outer6_model1_familyslots02/train_proposal_diagnostic.events.jsonl)

## 附录 B：复杂度统计口径

legacy 数字按以下口径复核：

```text
lines:
  Python source splitlines

functions:
  AST module.body 中 FunctionDef / AsyncFunctionDef
  nested-inclusive count 使用 ast.walk

HLE configuration surface:
  source 中唯一正则 token HLE_[A-Z0-9_]+

verifier / fallback proxy:
  顶层函数名分别包含 verifier / fallback
```

这些统计用于描述控制面规模，不应被当作独立行为数量或性能指标。
