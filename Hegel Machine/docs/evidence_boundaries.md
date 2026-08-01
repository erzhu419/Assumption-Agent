# Evidence and claim boundaries

## 当前可以说什么

当前实现和测试支持两个递进层级：

> 在一个冻结的有限定律库和受控离线样例中，系统能够在已提供 typed law
> binding 的条件下，不用语义相似度作为接受依据，计算 structural residual，
> 拒绝缺观测、虚构实体、角色交换、符号翻转和尺度不兼容，并把测得的证据送入
> 不可变、受 evaluator epoch 约束的保守候选评估流程。

以及：

> 在统一 outer schema、冻结且完整的 family/role/scale projection 集合中，系统在
> 不接收 answer key、不读取 semantic metadata 的条件下，能够从统一的
> verifier-ready synthetic witness bundle 经 frozen adapter 重放 24 个候选，并
> 选择唯一通过的 family–binding–scale 组合；目标 binding/scale counterfactual 必须完成计算，
> 不同 family 的距离按 verifier tolerance 归一化后才比较。遇到多解、缺证据、
> competitor 未完成或全部违反时按冻结 policy abstain，并生成绑定 law、role map、
> scale、epoch 和 residual drift 的跨 episode witness。

第一层叫 verifier/integration qualification；第二层的唯一里程碑名称是
**Phase-2A Controlled Typed-Selector Mechanics Qualification**，一般能力名为
**Explicit-Projection Typed Structural Selection**，scale 子能力名为
**Scale-Indexed Candidate Projection Selection**。它们都不叫 raw-evidence law-family discovery，也不是
正式 Phase-2 exit evidence。v0.2 corpus 有 43 个 synthetic case（24 个
answerable、19 个应 abstain），每例 24 个 projections，六族均跨两种 scale；
24 个正例形成 12 个 preservation pair。当前 controlled data 只是代码内合成与
adapter replay。Phase-2 selector 报告没有 sealed manifest，输出的是内部工程标签
`controlled_api_selector_qualified`；另一个 governance vertical slice 才输出
`candidate_framework`。两者都不授权 active graph mutation。即使 governance
manifest 结构检查通过，当前版本也没有外部签名可信根，不会晋升 ACTIVE。

此外可以说：Phase-2B 已有 family-neutral-shaped field-allowlisted wire、内部
role/scale candidate enumeration、绑定完整 adapter grid 的 interval selector core、
统计协议、immutable lifecycle record/进程内防分叉 guard 和 OCI launch-spec contract。
这些是正式轨道的实现基础，不是效果证据，也不证明允许字段无 oracle side channel、
跨进程 one-shot 或 runtime 隔离。Phase-2B 精确合同已冻结为 720 个 main cases +
240 个独立 semantic-conflict challenge，以及 496 legal + 76 invalid = 572 个 derived
pairs；case/margin 配额、baseline config、bootstrap、rerun、footprint 与 covert-audit
规则均不再是 open question。overall contract 是 `hegel-freeze-p2b-p3-v1.0.2`；它继承
v1.0.1 的 seed 修正，以 `2611585425` 作为 domain-separated SHA-256 → uint32 的 sklearn
`random_state`，并保留 `411876909552964556` 作为 master/bootstrap seed。在完整 standard-error
语义实现前，formal selector 只允许
`absolute_bound`。这些数字和限制是 preregistration，不是已生成、已审计或已通过的
holdout evidence；custodian/runtime 未 attested，covert audit 未执行，projection compiler
与完整 pipeline 仍未完成。

此外可以说：Phase-3A 的 v1.0.2 strict specification 已冻结，Python/Rust shared vectors
各 48/48 PASS。M2 两端都接受 64,680 个 source candidates，并得到 64,680 个 unique strict
canonical AST；共同 diagnostic set commitment 是
`sha256:c1a02a66a8d6d8f75204cb3daf03ab0b01c2b3b8e486d0ab3d481ee3be43c930`，ordinal
50,001 的 AST hash 是
`sha256:7c7f786c2cc57d31506b3c61d162d175c7f69a2878a089c72c9d053694cba948`。因此可准确
报告：`hegel-old-dsl-v1.0.0` 在 50,000 syntactic canonical-program budget 下的 bounded
status 为 `DSL_TOO_LARGE`。证据见
[v1.0.2 strict canonical/certificate freeze](Hegel_Machine_Strict_Canonical_AST_CBOR_Certificate_Bridge_Freeze_v1.0.2.md)、
[Phase-3 readiness resolution](Hegel_Machine_Phase3_Freeze_Readiness_Resolution.md)、
[dual strict gate](../artifacts/phase3_dual_strict_gate_v1.json) 与
[dual strict capacity replay](../artifacts/phase3_dual_strict_capacity_replay_v1.json)。

这个状态不是 `COMPLETE`，也不是 extensional target verdict。diagnostic commitment 不是
formal RFC6962 root；formal roots 仍为 `null`，没有 hidden-sink formal verdict、outside/MDL
certificate 或 ACTIVE authorization。

此外现在可以说：批准的 shrink step 1 已创建 `hegel-old-dsl-v1.1.0` /
`hegel-freeze-p2b-p3-v1.1.0` diagnostic child freeze。它只删除 mean/min/max admission，
保留 numeric IDs 0/1/5，tombstone 2/3/4。Python/Rust child vectors 一致，25,872-source
constructive subset 均产生 25,872 unique AST，commitment 为
`sha256:653fcb9428684cfed11c3f2345ac95ed98ded6e31564c9eeabf97c57ee71a7e9`，无 50,001
witness。准确称谓是 `SHRINK1_SUBSET_QUALIFIED_M3_BLOCKED`；child closure state 是
`NOT_RUN`，不是 `COMPLETE`。

Target/control source binding manifests 已生成，但因历史 split seed commitment、parent
binding manifest、custodian continuity attestation 和 hidden-access ledger 不存在，M3
commitment gates fail-closed。formal roots 全部仍为 null。

v1.1.2 已允许继续 M2.5 deterministic wire/seed/bridge 施工。当前 Python/Rust 已
重现 odd 480 行、sink 85 行以及 amendment 列出的四个 RFC6962 candidate values；
odd `192/96/192` 和 sink `39/20/26` quota allocation 也已实现为不接触真实
secret 的纯函数。Rust 诊断报告绑定实际运行 binary 的 SHA-256，但明确
`binary_source_binding_claim=false`；它不是 ImplementationBinding 或 build attestation。

这类 candidate qualification 仍不是 gate pass。authoritative DAG 实现审计发现 12 组
exact 冲突，涵盖 output-slot count、bridge topology/envelope/domain、actor trust、FD-3/ledger
boundary、absence-audit wire、role/state IDs、nested root preimages、opaque-ID evidence、sink witness
和 signature coverage。外部 start guard 因此必定在 CSPRNG 和 marker 前失败。权威计数
仍是 `14/24`，seed 未实例化，signature/ledger/absence claims 均为 false，formal roots、
M3 execution manifest 与 run outputs 均为 null，child 继续 `NOT_RUN`。不得由
Codex、自签 test key 或固定 test seed 冒充独立 custody。完整待决问题见
[`questions_for_gpt_phase3_m25_wire_completion_errata.md`](questions_for_gpt_phase3_m25_wire_completion_errata.md)。

## v2 SCAR 后续正式负结果

Hegel legacy snapshot 冻结后，Assumption Agent commit `4861b2d8` 记录了一个
protocol-valid SCAR negative：冻结的 fixed extractor/binder、hard structural eligibility
与 length-2 composition arm 显著低于 semantic-only，且主要失败机制是 coverage collapse
与结构信号噪声。该后续证据已由
[`v2_scar_negative_evidence_binding_v1.json`](../artifacts/v2_scar_negative_evidence_binding_v1.json)
append-only 绑定，影响分析见
[`v2_scar_negative_impact_assessment.md`](v2_scar_negative_impact_assessment.md)。

它不推进或阻塞 M2.5/M3 formal gates，不改变第一次 split seed 实例化，也不继承为 v3
effect evidence。它约束的是更强的 Phase-3B/3C 解释：在 old-law competence、recognizer/
extractor 故障隔离和 conservative soft integration controls 通过前，不得把结构前端失败诊断为
`ONTOLOGY_DEFECT`。Phase-3A 的 outside claim 继续只绑定明确版本和 universe 的 frozen
bounded DSL。

## 当前不能说什么

- 没有新关系发现、开放世界本体演化或现实科学发现的效果证据。
- v1 的 framework-growth 分数含公式化 fixture；它们只提供 schema/threshold
  原型，不是 v3 的 PASS 证据。
- v2 的 GSCL controlled corpus 是合成 qualification，不是 downstream efficacy。
- 不能把 `4861b2d8` 的 SCAR operationalization negative 写成“22 条 UAO 或 13 条 legacy
  aliases 已被整体证伪”；SCAR formal arm 没有逐条执行 T01–T22，13 条也不是独立的第二套
  ontology。
- 不能把 SCAR 与 WikiSQL same-v5 的效果大小直接比较成先验优劣。WikiSQL 只开放四个手写
  recipe、选择 T05/T18，overall primary 仍为 false，且 missing-arm completion 是公开披露的
  post-terminal protocol exception；两项结果只能用于形成后续 matched control 的设计动机。
- ARN 已被用于实现后验诊断，不能再叫 untouched。
- 文献或 repo 被归档不等于其结论已在本项目复现。
- benchmark 中反向构造的 semantic-only control 只诊断验收路径没有读该分数；
  其 decoy accuracy/gap 不参与 exit gate，不能作为与真实 embedding 系统的效果
  比较；semantic metadata replacement invariance 则是 anti-leak 硬检查。
- API-blinded selector 的 projection 由冻结 adapter 提供；尚未测从 raw text、table
  或 trajectory 生成完整候选集合。
- 当前结果不是 sealed holdout、正式 Phase-2 exit 或 open-world discovery 证据；
  `controlled_api_selector_qualified` 只是受控机械闭环报告中的状态标签。
- blinded 仅表示 recognizer API 不接收 answer key，并通过一致 ID 重命名不变量；
  source-visible generator 的 case schedule 可被重建，公开 ID 不是保密边界。
- fixture 值由 evaluator case spec 反向构造；family/binding/scale accuracy 是 selector
  mechanics 的功能测试，不是独立 raw evidence 上的能力估计。
- 不能把 isolation argv/dataclass 当作真实隔离证明，也不能把一次性状态机对象当作
  独立 custodian attestation。
- 不能说 Phase-2B 已通过：没有 secret 720-case holdout、外部 baselines、prediction-
  before-reveal 时间证据、consumed score report 或第三方 replay；额外 240 challenge
  和 572 derived pairs 也没有正式运行证据。
- 不能说 parity 或 480-row target 已证明 outside。v1.0.2 已定案无 implicit Bit coercion，
  binary XOR sanity witness 必须显式使用两个 `bit_to_scalar`；但 M2 只判定 syntactic
  budget overflow，没有运行 extensional target comparison，所以 formal verdict 仍为 `null`。
- 不能把 bounded `DSL_TOO_LARGE` 写成 `COMPLETE`、closure cardinality 或 outside evidence。
  64,680 是双 strict replay 接受的 unique canonical AST 数，但 M2 没有 frontier exhaustion、
  output archives、extensional quotient、match set 或 target verdict。
- 不能把 caller-supplied `ClosureEnumerationReceipt` 当执行证明。即使它结构合法地记录
  50,000 个 accepted programs 和第 50,001 个 witness，仍不能取代 M2 dual artifact。
  receipt 必须绑定完整
  `dsl_spec_id`/`operator_semantics_id`，并为 outside target 与 null control 选择各自独立的
  diagnostic universe/truth content IDs；跨 role 复用 IDs 必须 fail closed。正式
  canonical-CBOR/RFC6962 bridge schema 已冻结但未执行，不能从 diagnostic ID 换前缀得到
  formal root。
- 不能把 shrink-1 的 25,872 accepted unique 写成 child closure cardinality 或
  `COMPLETE`；它只是一个预注册 constructive subset，完整 grammar 仍可能超过 budget。
- 不能声称旧 split seed 已复用或未泄漏。仓库没有可验证的 seed commitment、custodian
  attestation 或 access ledger；当前 manifest 的 null 字段是 blocker，不是证明。
- 不能把“certificate wire 已冻结”写成 certificate 已生成或签发。program/output archives、
  formal roots、complete dual closure replay、key-status trust chain、3/3 signature 与 MDL
  dual replay 仍未完成。
- 不能把任何 bounded outside result 简写成 `OUTSIDE_LANGUAGE`。若未来满足全部条件，
  claim 也只能是
  `OUTSIDE_FROZEN_CLOSURE(dsl_version, bounded_universe_root,
  target_truth_table_root, equivalence = exact_extensional)`。

## 权力与数据隔离

| 角色 | 可以读 | 禁止 |
|---|---|---|
| Generator | train/source-only residual、ontology | holdout outcome、最终晋升标签 |
| Formalizer | typed candidate、冻结语言 | 自造证据、决定晋升 |
| Falsifier | candidate、反例预算 | 更改 candidate 以便通过 |
| Evaluator | 预注册 protocol、sealed evidence | semantic score、跨 epoch 偷换标尺 |
| Promoter | 结构化 certificate/receipts | 原始自由文本、自己生成测试结果 |

Evidence priority 为 proof > executable test > physical/simulation >
held-out human > independent LLM。多个相似 LLM judge 不视为独立重复。

## 版本与 split 规则

- Evidence 必须绑定 `theory_version`、`evaluator_epoch`、`probe_version`、
  `data_cutoff` 和 split。
- 预注册预测必须在 outcome 打开前有内容 hash 和早于 receipt cutoff 的时间。
- epoch 内 evaluator 冻结；换 epoch 后旧分数只保留为历史，不能直接相加。
- semantic retrieval 与 structural/predictive validation 分列保存。
- sealed manifest 的五个 split 必须互斥并使用内容寻址 observation ID；
  每个 ID 都解析到带来源 hash、split 和 cutoff 的 `Observation`；manifest
  还绑定 parent、patch、evaluator、probe registry、policy、注册/开启时间和
  独立 custodian。
- 已评估 patch 的 hard negatives、失败 receipts 和完整 replay record 进入当前
  本地、非权威 shadow lifecycle；writer 签名接入前，REJECT 不能作为不可逆
  全局结论。Verifier abstention 的统一持久化仍是后续工作。
- Certificate 必须绑定 patch、receipt、ledger、policy、reduction 和拟议 child；
  非活动记录 API 会从完整输入重算，active 写入在可信根实现前硬禁用。

## 文献访问边界

`references/manifest.json` 逐项区分 full text、author manuscript、metadata /
landing page、访问受限、无官方源码和第三方复现。不会绕过付费墙，也不会把
HTML access page 命名成 PDF。
