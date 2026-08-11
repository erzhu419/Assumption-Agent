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
holdout evidence；custodian/runtime 未 attested，covert audit 未执行。当前已有
root/identity + binary64 absolute-bound envelope mechanics、独立的 bundle-atomic exact
RationalValue-grid uncertainty receipt，以及一个受限的 root/identity exact bridge。
receipt 用 `Fraction.from_float` 绑定 wire 的 binary64 语义，对 numeric bounds 精确扩张并
向冻结的 663 点 grid 向外取整；任何 `standard_error` 或 grid 越界整包拒绝、绝不返回
部分结果。权威 bridge 内部重算该 receipt 与完整 adapter grid，用 exact 自然区间对六类
law 的 residual、tolerance、normalized interval 和 structural margin 作保守包络，不降回
float selector。它还冻结 entity/role/quantity/channel/membership、transform、
observation/scale/edge/vector/component/candidate、adapter-scan、exact-operation 与 Fraction
bit-length 预算，并在哈希前递归拒绝任意嵌套 authority dataclass/enum/tuple/primitive
子类，同时限制 theory 与整棵 authority tree 的节点数、文本量和整数 bit-length。所有
preflight rejection 都发生在任何 authority 内容哈希前，只携带
bundle ID、schema/registry-theory version 与冻结 policy IDs；它不携带内容根或 run ID，
不能作为证据或下游 receipt。通过 preflight 后的执行错误才绑定完整 provenance 并整案
abstain。

当前还实现了 `PublicTransformEvidenceBundleV2` 的八种 wire-operation typed certificate 与
exact sparse/discrete kernel mechanics，以及从 transformed observation 重建 witness inventory、
完整 strict-scope law × binding × scale × support-slice grid、先按 scale 合并所有 slice hull
再选择的 derived bridge。它们不是完整物理语义：unit/aggregation/sampling/split/coarse 各有
明确的窄合同，derived verifier 仍拒绝有量纲 witness，forest 也不支持 multi-root merge。
八种 wire operation 与八类 formal preservation transform 不是一一对应；572 个
preservation pairs 尚未生成或执行。

固定 65,536-byte envelope 的 feature/statistics/32-32-16 invariance mechanics 也已实现，
但 receipt 恒为 non-authoritative。Stage A accepted-JCS profile mechanics 把 binary64/rational
转成 schema-closed 字符串表示，显式记录 frozen-10 与 V2-extension-6 namespace/path，并用
80-byte header + 公开 test padding 构造可重放的 65,536-byte envelope。Stage B keyed batch
mechanics 进一步实施三份 pairwise-distinct 32-byte IKM 的 purpose separation、unbiased
whole-batch shuffle、case-local HMAC UUIDv4、rename 后 recanonicalization、wire-only public
provenance、secret HMAC padding、原子 batch 与 supplied-secret custodian replay。

这些不是 trust/effect evidence：pairwise distinct 不证明独立 custodian key generation，public
decoder 不验证 secret padding，replay 只说明给定 authorities/run/IKM 可重建同一 bytes，且
wire-only provenance 不能由现有 V2 typed validator 重放。因此 strict typed authority decode、
origin authentication、完整 trusted RFC8785 builder、formal 字段/UUID namespace 审计、
720+240 formal 资源合同、recognizer CLI、archive evaluator、sealed data、runtime/custodian 和
C1 exit 仍缺失；宽泛 projection compiler、完整 typed pipeline、formal covert audit 与 formal
holdout 状态继续为 false。入口虽冻结 1..1024 authority cap，最大规模的 wall-time/RSS 尚未
资格化，不能据小批次 mechanics 回归推断 formal-corpus capacity。

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

Target/control source binding manifests 已生成。`parent_binding_manifest_root = null` 是冻结的
规范替代方案，由 legacy payload IDs、absence attestation V2 和完整历史 audit 支撑；当前
缺的是 purpose-4 隔离 technical actor 的 live audit/attestation，不是待定 wire 选择。split seed commitment、
custodian continuity attestation 和 hidden-access ledger 仍不存在，因此 formal roots 全部为
null。

Phase-3A M2.5 的 “external actor” 已由
[`Hegel_Machine_Owner_Accepted_Container_Technical_Actor_Eligibility_Amendment_v1.md`](Hegel_Machine_Owner_Accepted_Container_Technical_Actor_Eligibility_Amendment_v1.md)
精确定义为仓库构建/编排进程之外、purpose-separated 的离线 Docker technical actor；它不要求
独立真人或组织。所有相关 publication 必须完整披露
`same_admin_controller=true`、`organizational_independence=false`、
`independent_human_actors=false`、`technical_role_independence=true`、
`owner_accepted_threat_model=true`、`remote_attestation=false` 和
`hardware_key_nonexportability=false`。这只是 M2.5 threat-model/eligibility 决定，不把容器
说成组织独立，也不修改 formal wire 或 Phase-2B sealed-holdout custody 要求。

Phase-3A M2.5 deterministic wire 现有 81 个唯一 tags/schemas。Python 与 Rust 从 detached
Commit-A snapshot 精确重放 21 个 candidate objects、8 个 candidate record trees 和 15 个
production-validator errors，并与 golden 完全一致。准确状态是
`DUAL_EXACT_WIRE_ERRATA_GOLDEN_PASS`，artifact kind 为
`DETERMINISTIC_CANDIDATE_NON_AUTHORITATIVE`。它只允许开始 owner-accepted technical-actor
external-genesis 流程；
该流程尚未执行，stored JSON/self-hash 单独不能授权。

这次 qualification 没有创建 seed、key、signature、marker、external audit claim、formal root、
Gate 15–24 pass 或 M3 identity。权威计数仍是 `14/24`，formal roots、M3 execution manifest
与 run outputs 均为 null，child 继续 `NOT_RUN`。Codex 可以编排冻结流程，但不得用编排进程
内的自签 test key、固定 test seed 或伪造 receipt 代替隔离 actor 的 live evidence。定案与操作边界见
[`Hegel_Machine_Phase3A_M25_Exact_Wire_Errata_Resolution.md`](Hegel_Machine_Phase3A_M25_Exact_Wire_Errata_Resolution.md)、
[`Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md`](Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md)
和
[`phase3_m25_external_genesis_operator_runbook.md`](phase3_m25_external_genesis_operator_runbook.md)。

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
  canonical-CBOR/RFC6962 candidate wire 已双实现资格化，但 formal roots 仍未实例化；不能
  从 diagnostic ID 换前缀或重命名 candidate root 得到 formal root。
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
