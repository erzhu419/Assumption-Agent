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
规则均不再是 open question。overall contract 是 `hegel-freeze-p2b-p3-v1.0.1`；它以
`2611585425` 作为 domain-separated SHA-256 → uint32 的 sklearn `random_state`，并
保留 `411876909552964556` 作为 master/bootstrap seed。不可执行的 v1.0.0 直接 64-bit
sklearn 绑定已被 implementation-audit amendment supersede。在完整 standard-error
语义实现前，formal selector 只允许
`absolute_bound`。这些数字和限制是 preregistration，不是已生成、已审计或已通过的
holdout evidence；custodian/runtime 未 attested，covert audit 未执行，projection compiler
与完整 pipeline 仍未完成。

此外可以说：Phase-3 的 `hegel-old-dsl-v1.0.0` finite domains、typing、bottom、exact
equivalence、limits、50,000/5,000,000 budgets 和 shrink order 参数已经冻结；首个 generic
odd-cardinality target 的 universe 是 480 行，observed omitted-channel null control 是
85 行，MDL table 是 `hegel-mdl-prefix-v1.0.0`/Q32。capacity preflight 还可以准确报告：
diagnostic tuple-AST/canonical-JSON 表示下有 64,680 个 distinct、typed、limit-conforming
candidate AST，状态为 `CONDITIONAL_CAPACITY_LOWER_BOUND_EXCEEDS_BUDGET`。strict canonical
AST/CBOR acceptance 未冻结，所以 executed closure status 仍为 `NOT_RUN`。这叫
**Phase-3A Bounded Frozen-Closure Adequacy** 的 surface-parameter freeze：
`surface_parameter_freeze_complete=true`，但
`strict_acceptance_contract_complete=false`、`normative_parameter_freeze_complete=false`。
可以说 untrusted receipt wire 会绑定 DSL spec/operator semantics，并依据 target role
分别选择 480-row outside-target roots 或 85-row null-control roots；不能说这些 roots 已由
sealed replay 重算。

## 当前不能说什么

- 没有新关系发现、开放世界本体演化或现实科学发现的效果证据。
- v1 的 framework-growth 分数含公式化 fixture；它们只提供 schema/threshold
  原型，不是 v3 的 PASS 证据。
- v2 的 GSCL controlled corpus 是合成 qualification，不是 downstream efficacy。
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
- 不能说 Phase-3A 已开始正式实验或 parity 已证明 outside。旧 DSL 的
  intended standard numeric semantics 下，`absolute(difference(x,y))` 给出 binary XOR
  truth table；但决定稿把 `Bit` 直接传给要求 `RationalValue` 的 `difference`，是否漏写
  `bit_to_scalar` 仍待消歧。在 executable closure 前它只算
  `TARGET_DESIGN_SANITY_ONLY`。480-row target 与 85-row null control 已冻结，但完整
  finite closure、canonical archives、Python/Rust 双 replay、3/3 signatures 和 MDL
  scorer replay 尚未完成或执行。
- 不能把 64,680 candidate-AST 容量下界称为 `DSL_TOO_LARGE`。它尚未通过 strict
  canonical AST node CBOR schema、normalization/rewrite 和 node-counting rules 的正式
  acceptance；diagnostic canonical JSON 不能替代 canonical CBOR。只有该子集被正式
  canonicalizer 接受，才进入 `DSL_TOO_LARGE → new DSL version → shrink step 1`。
- 不能把 caller-supplied `ClosureEnumerationReceipt` 当执行证明。即使它结构合法地记录
  50,000 个 accepted programs 和第 50,001 个 witness，当前仍是 untrusted claim，不能
  把 executed closure 从 `NOT_RUN` 改成 `DSL_TOO_LARGE`。receipt 必须绑定完整
  `dsl_spec_id`/`operator_semantics_id`，并为 outside target 与 null control 选择各自独立的
  diagnostic universe/truth content IDs；跨 role 复用 IDs 必须 fail closed。正式
  canonical-CBOR/RFC6962 roots 的 preimage/tree bridge 尚未冻结，不能从 diagnostic ID
  换前缀得到。
- 不能说 certificate wire 已完全冻结。program/output archive root identity、match hash、
  exhaustion preimage、certificate/key/revocation envelope 和 MDL cross-language wire 等
  machine-readable schema 仍待消歧。
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
