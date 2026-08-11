# Architecture v0.2 + formal-track contracts

## 1. 一个状态，而不是三套互斥公式

两份主文档先后使用
`(Σ, φ_s, R, V, Ω, P, N, ρ, C)`、`(X, A, Q, P, ~)` 和
`(X, M, Q, V, Ω)`。活动实现统一为：

```text
TheoryState
├── signature / ontology                  Σ, X
├── model classes                         M
├── representation and scale maps         φ_s, π
├── typed relations and composition       R, A
├── hypothesis family                     P
├── probes and evaluator epoch            Q
├── violation functionals                 V
├── scope                                 Ω
├── observational equivalence             ~Q
├── hard negatives and counterfactuals    N
├── preregistered novel predictions
├── reduction maps                        ρ
├── conditional description length       C
└── immutable version / lineage / cutoff
```

每次 scope 修改先产生内容寻址的 proposed version；旧版本和失败分支不覆写、
不删除。当前 active append 关闭，因此 proposed version 只进入 certificate，
不进入活动状态图。

## 2. 四个认知层

- L0 `Observation`：来源、时间、split、数据截止点与可观测量。
- L1 `ObjectHypothesis`：当前语言可表达的对象级机制主张。
- L2 `RelationLaw`：typed roles、arity、scope、scale、violation functional。
- L3 `TheoryState`：允许哪些对象、关系、probe、编译器和 evaluator。

`TreatmentProgram` 属于执行层。它可以由通过检验的 claim 编译而来，但不能
替代 claim，也不能用“action 有效”倒推某个机制已被发现。

## 3. 认知空间对与几何

系统保存对象/假设空间与 probe/evaluator 对：

```text
C_t = (X_t, A_t, Q_t, P_t, ~_t)
```

对 probe `q`，两候选的差异是
`p_q(h1,h2)=D(P_h1^q,P_h2^q)`。一组 probe 给出一族 seminorm /
pseudometric，任务只是在冻结 probe 上加权。若所有 probe 都无法区分两个
表示，它们在当前任务中进入同一 observational equivalence class，而不是
被重复计为“新理论”。

语义检索分数只允许召回候选；`semantic_retrieval_score` 不进入 law
verifier、evidence ledger 聚合或 promotion gate。

## 4. 一轮只改一个主坐标

`TheoryPatch` 必须声明唯一 `coordinate`：

```text
parameter | noise | scope | mixture | composition
representation | robustification | idealization
probe | language | evaluator | revision
```

这实现理论坐标下降：冻结其余主要组件，以最小扩展定位究竟是哪一层不足。
语言扩展只有在本体不足证书通过后才可提出。

## 5. Phase-2A development 数据流

```text
candidate retrieval (semantic allowed)
  → uniform verifier-ready synthetic witness bundle with anonymous labels
  → frozen adapter replay of the complete family/binding/scale grid
  → required-observable and scale checks
  → deterministic law residual
  → hard-negative / binding-counterfactual / scale-counterfactual / sign-flip checks
  → tolerance-normalized boundary-margin selection or policy-bound abstention
  → falsifier receipt
  → evaluator receipt in frozen epoch
  → verified typed law match
```

初版六类 law 是 symmetry/equivariance、monotonicity/order、
conservation/balance、pair complementarity、negative feedback 和
locality/Markov。前四类沿用 GSCL v0.2 exact residual 的核心定义；
GSCL 的第五个 exact kernel 是 composition/path consistency，并未偷换成
当前的 feedback。negative feedback 与 locality 是按《黑格尔机》最新
Phase-2 范围新增的 prototype。feedback 只有在系统诱发响应、严格 margin、
时序、同一受控量和局部稳定窗口均可观测时才运行；locality 需要绑定
Markov blanket 和外部上下文。

v0.1 benchmark 的 observable schema 一一暴露候选 law family，纵切直接提供
typed binding；它只测 executable verification 和反语义控制。v0.2 新增
`SharedEvidenceBundle`、`FrozenProjectionAdapter` 与 `UnboundStructuralEpisode`：
43 个 case 都先形成统一的 verifier-ready synthetic witness bundle，再确定性重放
六族 × 两种 binding × 两种
scale 的 24 个 projections。measurement key 只绑定实际见证该 observable 的角色
实体，因此未变角色的 measurement 会被两个 binding 竞争项共同消费；bundle 拒绝
重复 witness key，adapter 内容同时绑定 theory version、verifier registry 与
evaluator epoch。六族均跨两种 scale；24 个 case 可回答、19 个应 abstain。运行
路径不接收 answer key，并要求唯一 PASS、完整 family coverage、目标 binding/scale
competitor 均完成计算，以及冻结的 tolerance-normalized boundary margin，否则
abstain。

上述结果只资格化 synthetic controlled adapter replay 与候选选择。它没有测 raw
text/table/trajectory extraction、一次性 sealed holdout、正式 Phase-2 exit 或
open-world family discovery。人工 semantic-only control 只做 metadata-invariance
诊断；它的 decoy accuracy 和 structural gap 不进入 exit gate，但“替换 semantic
metadata 后结构决策不变”本身是 anti-leak 硬检查。

`PreservationWitness` 现已检查两个具体 recognition decision 的 law、role map、
scale、evaluator epoch 和 residual drift。当前 24 个 answerable case 构成 12 个
family × scale 的原例/实体改名 pair。entity correspondence 来自 recognition 前已
冻结的 evaluator answer table，scale correspondence 是预注册 identity map，不从
selected output 事后生成。它只是 bounded metamorphic witness，不称 functor 或通用
structural correspondence。

## 6. Phase-2B 正式轨道边界

overall contract 版本是 `hegel-freeze-p2b-p3-v1.0.2`。它继承 v1.0.1 的 seed 修正：
`411876909552964556` 保持 master/bootstrap seed，sklearn 使用 domain-separated
SHA-256 → uint32 的 `2611585425`；同时冻结 Phase-3 strict acceptance/certificate
规范。随后 M1/M2 的通过是 Phase-3 bounded-capacity 证据，不替代 Phase-2B sealed exit。

Phase-2B 不会在 Phase-2A report 外包一层 `sealed=true`。它使用新的 public wire、
进程和一次性状态边界：

```text
PublicEvidenceBundle + TheoryState + Phase2BAdapterRegistry
  ├─ root/identity profile:
  │    exact uncertainty + authoritative adapter replay
  │      → six-law exact rational verifier + exact selector
  └─ transform profile:
       PublicTransformEvidenceBundleV2 authority
         → internally recomputed uncertainty + typed exact transform kernels
         → derived witness inventory + complete support-slice grid
         → per-scale conservative hull + exact selector

Both bounded profiles
  → trusted RFC-8785 wire / CLI / archive                [not implemented]
  → unique family+binding and admissible scale set, or abstain
  → PredictionBundle commitment                         [not implemented]
```

`phase2b_wire.py` 不 import law/verifier、generator、evaluator 或 Phase-2A fixture；
它严格拒绝 law/gold/PASS/rank/margin/candidate-private fields。Adapter 内部才持有
opaque quantity/role/family registry，调用者不能传 projection grid。枚举超过冻结
budget、缺 role/quantity registry、或 scale DAG 到同一节点有多条未消歧 path 时，
adapter 不返回部分 grid。

Public selector 同样不接受 caller-provided grid commitment；它从输入 bundle 与
frozen registry 确定性重跑 adapter，再将每项 evaluation 与完整 hypothesis ID、
family/binding/scale metadata 和 footprint commitment 对齐。自洽但截断的子网格
不能给自己签发“complete”声明。

当前 `phase2b_projection_compiler.py` 已把这一边界推进到一个窄而可执行的
mechanics slice：严格匹配 quantity、role/entity witness 和完整 adapter grid；对
root/identity scale path 做 Boolean 与 binary64 `absolute_bound` 的 outward interval
envelope。只有无量纲、temporal/spatial support 精确对齐、且落在保守 binary64
算术安全域内的退化 point envelope 才进入六类 verifier；非退化 interval 不能只靠
角点声称得到非线性 residual 的保守界，因此在完整 grid 上显式成为 error cell。
非 identity transform、缺失/歧义 witness 或 shape drift 同样 fail-closed，selector
必须 abstain。

`phase2b_uncertainty_compiler.py` 提供独立的、bundle-atomic exact receipt：它把 wire
已规范化的 binary64 用 `Fraction.from_float` 精确提升，对 `NumericValue` 使用
`x +/- radius`、对 `NumericInterval` 使用 `[lower - radius, upper + radius]`，再向
literal-pinned 663 点 RationalValue grid 向外取整。任何 `standard_error` 或越界端点
都拒绝整包且不返回 partial siblings；Boolean 与 missing 保持 typed。

`phase2b_exact_bridge_v1.py` 现在消费这份 receipt，但只声明一个明确受限的
root/identity bridge。权威入口只接收原始 bundle、theory 与 adapter registry；调用者
不能注入 receipt、candidate grid、evaluation、selection 或 policy。入口在任何 bundle、
theory 或 registry 内容根计算前，先递归要求整棵 authority tree 使用 exact frozen
dataclass、enum、tuple 与 primitive 类型，并施加节点数、文本量和整数 bit-length 预算；
再执行 entity/role/quantity/channel/membership、transform、
observation/scale/edge、vector width、总 component、candidate 与 adapter-scan 预算，并
同时检查 transform catalog/path 的 preflight 语义。任何 preflight rejection 都只携
bundle ID、schema/registry-theory version 与冻结 policy IDs；它无内容根、无 run ID，
不可作为证据或进入 selector。通过 preflight 的输入随后再计算 provenance，并在内部
确定性重算 uncertainty receipt 和完整 adapter grid。六类 law residual
均以 `RationalAtom`/`Fraction` 自然区间运算保守包络：加减乘、绝对值、max 与正分母除法；
negative-feedback 的离散分支只在整段输入上可判定时执行，跨过 zero-branch boundary
则生成 error cell 并令 selector abstain。tolerance、normalized interval 和 structural
margin 也不降回 binary64。exact-operation 与 Fraction bit-length 另有执行中预算；
unused/missing transform catalog 项、非唯一 path 或任何定义域不确定性均 fail-closed。

`phase2b_exact_transform_semantics_v1.py` 新增了 content-addressed
`PublicTransformEvidenceBundleV2`、逐 observation/component 的 scale/unit/frame/support
metadata、八类不同的 typed certificate，以及 exact sparse interval/discrete kernels。
`phase2b_exact_derived_witness_bridge_v1.py` 只接收 transform authority、theory 与 registry，
内部重算 uncertainty 和 transform receipt，再重建完整 strict-scope
law × binding × scale × support-slice grid。selector 先把同一 scale 的所有 slice 做保守
normalized hull，禁止挑选有利 slice。缺失、歧义、unused observation、lineage/root 漂移或
任一 error cell 都原子 abstain。

这仍是窄 mechanics：unit conversion 只核声明的正比例及逆，coordinate affine 只给
axis-aligned box hull，aggregation 只核声明权重，sampling v1 仅 single-series
distinct-point scalar subselection，split/merge 只覆盖 extensive 值的一侧代数逆，
coarse-graining 只核声明的 sparse matrix equality，forest 也不支持 multi-root merge。
derived verifier 仍拒绝有量纲 witness；八种 wire operation 与八类 formal
`PreservationTransform` 不是同一 taxonomy，更不等于 496 legal + 76 invalid pairs 已执行。
recognizer CLI、trusted wire、formal covert audit、archive evaluator、sealed data、
runtime/custodian 证据都未完成。因此 `projection_compiler_implemented`、宽泛
`exact_rational_residual_interval_semantics_implemented`、
`uncertainty_semantics_compiler_implemented` 和完整 typed pipeline 状态仍为 false；下一刀
是 trusted RFC-8785 wire builder 与 namespace-aware field/UUID auditor，而不是生成或消费
holdout。

这里的 `family-neutral-shaped` 只表示 schema 没有显式 family/gold 字段。允许的 UUID、
provenance hash、role candidates、missingness 和 unused transforms 仍可能成为 covert
answer channel；正式 run 必须由独立 generator 随机化并全局 shuffle ID，再做重命名
不变量及 allowed-field answer-correlation/side-channel audit。冻结审计使用彼此独立的
shuffle/ID/padding keys、固定 65,536-byte envelope、10,000 次 stratified permutation、
Holm–Bonferroni FWER=0.01 和 32 次 global consistent renaming。当前已有固定 envelope 的
prefix/suffix feature、NMI/LOO balanced-accuracy、单一全局 Holm 和 32/32/16 invariance
mechanics；receipt 恒为 `NON_AUTHORITATIVE_MECHANICS_ONLY`。它没有 trusted RFC-8785
builder 或 namespace-aware field extractor，且未在 formal corpus 上执行，所以正式 covert
audit 仍未实现或通过。

新 selector 使用 residual 和 tolerance 的闭区间。保守 normalized interval 为：

```text
[residual.lower / tolerance.upper,
 residual.upper / tolerance.lower]
```

`upper ≤ 1` 才 PASS，`lower > 1` 才 FAIL，中间区间 fail-closed。选择结果可以是
唯一 family+binding 下一个或多个 preregistered admissible scale；这与当前只在
显式 scale-tagged projection 间选择的 Phase-2A 能力不同。在 frozen Student-t、
Bonferroni 和 outward-grid rounding 语义全部实现并测试前，formal selector 只允许
`absolute_bound`；`standard_error` 返回 `STANDARD_ERROR_UNSUPPORTED`。

正式样本合同是 720 个 main latent cases，加 240 个不进入主 accuracy 分母的 sealed
semantic-conflict challenge。另有 496 个 legal preservation pairs 与 76 个
invalid-transform controls，共 572 个 derived pairs；它们不是 720 的独立 case。

Host runner 只生成 OCI launch-spec contract：read-only input/root、no network、drop all
capabilities、no-new-privileges、ephemeral tmpfs、固定 image digest、资源上限，而且
repo、generator 与 answer manifest 不挂载。该本地 contract 不证明 runtime 真被
这样执行；冻结入口目前只是保留路径，专用 recognizer CLI、严格 main/challenge archive
evaluator、签名 SBOM 与 external attestation verifier 均未实现。

Seal lifecycle 是单向的：

```text
PREREGISTERED
  → GENERATED_SEALED
  → PREDICTIONS_COMMITTED
  → CONSUMED
```

answer manifest 只能在 prediction/audit archive hash 固定后 reveal，并校验 salted
commitment opening。当前 guard 只在同一 Python custodian 进程内原子阻止 parent
分叉；它不是跨重启持久的 append-only CAS ledger，也不是独立 custodian 证明。
quota、统计、baseline config、rerun 与 covert-audit 规则已经精确冻结；当前剩余的是
实现和外部证据 blockers。因此没有生成 720 + 240 sealed artifacts，也没有执行 572
derived pairs、完整 typed evidence → verifier projection compiler 或 formal pipeline。

## 7. 诊断与理论生长

固定诊断顺序：

```text
refit → noise/data → scope → mixture → low-order composition
      → idealize → robustify → add probe → invent language
```

前一步足以解释 residual 时立即停止。只有 persistent、跨 seed、超出不确定性、
有结构、可压缩且对预注册 holdout 有预测性的 residual 才能得到
`ONTOLOGY_DEFECT`。

理论晋升必须独立验证：

1. residual explanation；
2. old-success preservation；
3. reduction under old scope；
4. expressivity gain；
5. compression/MDL gain；
6. preregistered unseen prediction；
7. bounded regression；
8. complexity budget。

这八门另加 hard-negative rejection，只读结构化 receipts。Gate 核验每个
receipt 的 actor、probe/version、split、cutoff、独立性和 preregistration；
Generator 不看 holdout；Promoter 不接受自由文本结论；不同 evaluator epoch
的分数不能聚合。Sealed manifest 还必须冻结互斥的 train、validation、
old-success、holdout 与 hard-negative 分区；分区成员是内容寻址 observation
ID，且 ID 必须解析到带来源 hash、split 与 cutoff 的真实 `Observation`。
Manifest 同时绑定 parent theory、candidate/patch、evaluator epoch/version、
probe registry 和 gate policy；开启晚于注册，custodian 独立于五个执行角色。
即使满足这些结构约束仍只能得到 `CANDIDATE`：当前 kernel 没有外部
custodian 签名可信根，所以 `external_trust_root` 固定 fail-closed，active
graph append 被硬禁用。

Phase 2 只有 `scope` 坐标有活动 compiler。Certificate 绑定 patch hash、
receipt hashes、ledger、policy、reduction、epoch 和 proposed-child hash。
非活动 lifecycle 写入也保存完整 replay inputs 并重新计算 certificate；
`authorize_promotion` 当前只执行校验后拒绝，直到外部可信根接入。

## 8. Phase-3 freeze 与之后的接口

`OntologyInadequacyReport`、`TheoryPatch(language)`、`ReductionMap`、
`IdentifiabilityCertificate`、`RobustificationCandidate` 和
`IdealizationContract` 已作为真实接口存在，但当前 benchmark 不以它们证明
“已发明未知定律”。Phase 3 必须另建 hidden-law benchmark、冻结 validation
并通过 expression/non-equivalence test 后才能提升 claim。当前 Phase-3A 里程碑名为
**Bounded Frozen-Closure Adequacy**；它不是无边界的 outside-language detection。

Phase-3 的父 DSL surface 是 `hegel-old-dsl-v1.0.0`：finite rational/interval/
identifier domains、四 scope、六 aggregate、完整 typing、strict bottom、exact
extensional equivalence 和结构上限。v1.0.2 已冻结 strict AST/CBOR、rewrite/count、
archive/bridge、certificate/key/MDL wire；当前
`surface_parameter_freeze_complete=true`、`strict_acceptance_contract_complete=true`、
`normative_parameter_freeze_complete=true`。规范见
[v1.0.2 strict canonical/certificate freeze](Hegel_Machine_Strict_Canonical_AST_CBOR_Certificate_Bridge_Freeze_v1.0.2.md)，
readiness 状态迁移见
[Phase-3 resolution](Hegel_Machine_Phase3_Freeze_Readiness_Resolution.md)。

M1 的 shared vectors 在 Python 与 Rust 上各 48/48 PASS。M2 两端都把 64,680 个 source
candidates 接受为 64,680 个 unique strict canonical AST，diagnostic set commitment 同为
`sha256:c1a02a66a8d6d8f75204cb3daf03ab0b01c2b3b8e486d0ab3d481ee3be43c930`；canonical
order 中 ordinal 50,001 的 AST hash 同为
`sha256:7c7f786c2cc57d31506b3c61d162d175c7f69a2878a089c72c9d053694cba948`。证据见
[dual strict gate](../artifacts/phase3_dual_strict_gate_v1.json) 与
[dual strict capacity replay](../artifacts/phase3_dual_strict_capacity_replay_v1.json)。
因此 `strict_acceptance_implementation_verified=true`。

因此 `hegel-old-dsl-v1.0.0` 在 50,000 syntactically canonical program budget 下的
bounded status 是 `DSL_TOO_LARGE`。这不是 `COMPLETE`：没有 closure cardinality、
frontier exhaustion、extensional quotient、target match set 或 target verdict。set
commitment 只是 diagnostic commitment，不是 formal RFC6962 archive root；formal roots
仍为 `null`。

`ClosureEnumerationReceipt` 是按单一 `target_role` 提交的 untrusted replay claim。
每份 receipt 必须同时绑定完整 `dsl_spec_id`、`operator_semantics_id`、equivalence、
enumerator 与 budget。outside target 使用自己的 480-row
`bounded_universe_diagnostic_id` 和 `target_table_diagnostic_id`；in-language null control
使用独立的 85-row `hidden_sink_universe_diagnostic_id` 和
`hidden_sink_target_table_diagnostic_id`。这些是 canonical-JSON diagnostic content IDs，
不是正式 canonical-CBOR/RFC6962 roots；两组 diagnostic IDs 也不得复用或互换。
caller-supplied receipt 仍不可信，但 M2 dual artifact 已独立建立 bounded overflow：
50,000 accepted positions 加 ordinal-50,001 strict AST witness，且不声称 closed frontier、
closure cardinality、target match 或 formal archive root。

首个正式 target 是 `TARGET_P3A_GENERIC_ODD_REDUCTION_V1`，在 size 5–8 的完整
480-row universe 上定义，agent-facing split 为 192/96/192。hidden-sink null control
是 `CONTROL_P3A_OBSERVED_OMITTED_SINK_V1`：四个通道全部 observed，仅 auxiliary 被
初始 scope 遗漏，完整 universe 为 85 行。决定稿的 baseline label
`control_volume_primary_only_v1` 已由 v1.0.2 定案为 deprecated source alias；唯一 machine
ID 是 `scope_primary_only_v1`，formal canonicalizer 拒绝 alias，不增加第五个 scope。

v1.0.2 同时冻结“无 implicit Bit coercion”；二元 XOR sanity witness 必须显式包含两个
`bit_to_scalar`。但 M2 没有执行 extensional target replay，所以 binary XOR 与 480-row
generic target 的 formal language verdict 都仍为 `null`，token blacklist 不能替代判断。

MDL code table 已冻结为 `hegel-mdl-prefix-v1.0.0`，所有长度用向上取整 unsigned Q32，
禁止 binary float；formal scorer 必须忽略 caller-supplied length/Fraction/gain 并从 AST、
partition、prediction 与 code table 重算。certificate wire 已由 v1.0.2 冻结；synthetic/candidate
bridge wire 已双实现资格化，但 externally instantiated formal bridge roots、正式
program/output archives、完整 closure replay、key-status
chain、3/3 certificate 和 MDL dual replay 均未执行。因此没有 extensional target verdict、
hidden-sink formal verdict 或 outside/MDL certificate，ACTIVE 也关闭。正式 claim 只能写
`OUTSIDE_FROZEN_CLOSURE(dsl_version, bounded_universe_root,
target_truth_table_root, equivalence = exact_extensional)`，禁止无边界的
`OUTSIDE_LANGUAGE`。

该后继动作现已执行到 diagnostic publication：child
`hegel-old-dsl-v1.1.0` 使用 sparse-preserving AggregateMapId registry，active IDs
为 0/1/5，2/3/4 永久 tombstone，并在 source/formal acceptance 统一返回
`REJECT_REMOVED_AGGREGATE_MAP`。父 Python/Rust 实现保持不变；child admission 由新的
Python layer 与 Rust sibling crate 独立实现，因此 surviving AST bytes/hash 与父版本一致。

Child 的 23 个 strict vectors 双实现一致。预注册 25,872-source subset 在两端均得到
25,872 unique、零拒绝、零 collapse，diagnostic set commitment 为
`sha256:653fcb9428684cfed11c3f2345ac95ed98ded6e31564c9eeabf97c57ee71a7e9`，且无
out-of-budget witness。这只满足 M3 24 gates 的前 14 个，不是 closure exhaustion。

Target/control payload diagnostic IDs 保持内容稳定，并生成新的 child binding manifests。
冻结方案明确令 `parent_binding_manifest_root = null`，以 legacy payload IDs、versioned
absence attestation V2 和完整历史 audit 替代，不追造 retrospective parent manifest。
旧版本从未物化 split seed commitment、custodian continuity attestation 或 hidden-access
ledger；规范替代虽已解决，合格隔离 technical-actor live evidence 尚未生成。formal roots
继续为 null，child
state 为 `NOT_RUN`。

v1.1.2 已把这段工作接管为 **Phase-3A M2.5 Formal Commitment, Seed Genesis
and Bridge Qualification** 的 bit-exact completion amendment。实现依旧严格分层：
Python/Rust 实现 strict CBOR、ContentHash/RFC6962、typed odd/sink rows、HKDF/HMAC rank
与 state/output-null guards；真实 seed/key/signature 不在 import、test 或普通 build 中生成。

M2.5 purpose 1–4 的 actor eligibility 由
[`Hegel_Machine_Owner_Accepted_Container_Technical_Actor_Eligibility_Amendment_v1.md`](Hegel_Machine_Owner_Accepted_Container_Technical_Actor_Eligibility_Amendment_v1.md)
接管：四个 purpose-separated、离线、live-probed Docker runtime 构成
`OWNER_ACCEPTED_CONTAINER_TECHNICAL_ACTORS_V1`，不要求不同真人或组织。该模型必须披露
`same_admin_controller=true`、`organizational_independence=false`、
`independent_human_actors=false`、`technical_role_independence=true`、
`owner_accepted_threat_model=true`、`remote_attestation=false`、
`hardware_key_nonexportability=false`。它不改变 formal CBOR/root DAG，也不改变 Phase-2B
sealed-holdout 的 custody 合同。

当前 deterministic layer 已覆盖 81 个唯一 tags/schemas。E1–E12 的 output-slot、bridge
topology/envelope/domain、actor trust/ledger、absence audit、role/state、nested preimage、
opaque-ID、sink witness 与 signature-coverage 冲突已由 exact-wire resolution 和
implementation addendum 解决。Commit A 的 fresh detached Python/Rust replay 对 21 个
candidate objects、8 个 candidate record trees 和 15 个 negative guard codes 完全一致。
这准确称为 `DUAL_EXACT_WIRE_ERRATA_GOLDEN_PASS`，但仍是 non-authoritative candidate
evidence；checked JSON 单独不能授权 operational use。

因此下一硬门不再是规范选择，而是 fresh qualification 之后由上述 owner-accepted
technical actors 执行 genesis、parent-history audit、opaque-ID persistence 和
purpose-separated signatures。该流程尚未
执行，状态保持 `14/24`、formal roots `null`、`NOT_RUN`。定案和操作边界见
[`Hegel_Machine_Phase3A_M25_Exact_Wire_Errata_Resolution.md`](Hegel_Machine_Phase3A_M25_Exact_Wire_Errata_Resolution.md)、
[`Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md`](Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md)
与
[`phase3_m25_external_genesis_operator_runbook.md`](phase3_m25_external_genesis_operator_runbook.md)。

v2/SCAR commit `4861b2d8` 的 protocol-valid negative 不进入上述 identity/root DAG，故不
改变 M2.5 或 M3 closure gate；它作为 Phase-3B/3C 的 design-risk input 单独绑定。后续
adequacy/synthesis 评估必须用 matched arms 区分 ontology defect、extractor/recognizer
failure 与 hard eligibility coverage collapse，且不得继承 v2 thresholds/weights 为已验证
正先验。
