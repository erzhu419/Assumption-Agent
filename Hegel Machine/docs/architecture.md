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

overall contract 版本是 `hegel-freeze-p2b-p3-v1.0.1`。implementation audit 发现
v1.0.0 把 64-bit seed 直接作为 sklearn `random_state`，因 API 不可执行而被 supersede；
v1.0.1 保留 `411876909552964556` 为 master/bootstrap seed，并冻结
domain-separated SHA-256 → uint32 的 sklearn 值 `2611585425`。

Phase-2B 不会在 Phase-2A report 外包一层 `sealed=true`。它使用新的 public wire、
进程和一次性状态边界：

```text
PublicEvidenceBundle (family-neutral-shaped, UUIDv4 syntax, field allowlist)
  → Phase2BAdapterRegistry
  → complete internal family × injective role binding × scale-path hypotheses
  → verifier projection compiler                         [尚未实现]
  → interval residual / heterogeneous tolerance
  → unique family+binding and admissible scale set, or abstain
  → PredictionBundle commitment
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

这里的 `family-neutral-shaped` 只表示 schema 没有显式 family/gold 字段。允许的 UUID、
provenance hash、role candidates、missingness 和 unused transforms 仍可能成为 covert
answer channel；正式 run 必须由独立 generator 随机化并全局 shuffle ID，再做重命名
不变量及 allowed-field answer-correlation/side-channel audit。冻结审计使用彼此独立的
shuffle/ID/padding keys、固定 65,536-byte envelope、10,000 次 stratified permutation、
Holm–Bonferroni FWER=0.01 和 32 次 global consistent renaming；规范已经确定，但 wire
builder 与审计尚未完成或执行。

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

Phase-3 的 DSL surface 已冻结为 `hegel-old-dsl-v1.0.0`：finite rational/interval/
identifier domains、四 scope、六 aggregate、四个仅供 adapter/preservation 使用的
transform、完整 typing、strict bottom、exact extensional equivalence 和所有结构上限。
这里冻结的是 surface parameters：`surface_parameter_freeze_complete=true`；strict
canonical acceptance 和完整 normative contract 尚未闭合，所以
`strict_acceptance_contract_complete=false`、`normative_parameter_freeze_complete=false`。
50,000 计 extensional quotient 前的 syntactically canonical programs；raw expansion
cap 为 5,000,000。若正式 canonicalizer 接受并由 enumerator 产生第 50,001 个 canonical
program，规则才给出 `DSL_TOO_LARGE`；未耗尽 frontier 而先触发 raw cap 则是
`INCONCLUSIVE_BUDGET`。这两条是状态转换规则，不是当前执行结果。

当前 capacity preflight 只在 diagnostic tuple-AST / canonical-JSON 表示下构造了
64,680 个 distinct、typed、limit-conforming candidate AST。strict canonical AST node
CBOR schema、operator alias 与 algebraic rewrite、以及 aggregate/tolerance/AND 的
node-counting semantics 尚未冻结，所以该结果只能叫
`CONDITIONAL_CAPACITY_LOWER_BOUND_EXCEEDS_BUDGET`；executed closure status 保持
`NOT_RUN`。只有正式 canonicalizer 接受这 64,680 个 witness，才进入
`DSL_TOO_LARGE → new DSL version → frozen shrink step 1`。

`ClosureEnumerationReceipt` 是按单一 `target_role` 提交的 untrusted replay claim。
每份 receipt 必须同时绑定完整 `dsl_spec_id`、`operator_semantics_id`、equivalence、
enumerator 与 budget。outside target 使用自己的 480-row
`bounded_universe_diagnostic_id` 和 `target_table_diagnostic_id`；in-language null control
使用独立的 85-row `hidden_sink_universe_diagnostic_id` 和
`hidden_sink_target_table_diagnostic_id`。这些是 canonical-JSON diagnostic content IDs，
不是正式 canonical-CBOR/RFC6962 roots；两组 diagnostic IDs 也不得复用或互换。
wire 允许 `DSL_TOO_LARGE` 精确表示为 50,000 个 accepted canonical programs 加一个非空
`first_out_of_budget_program_id`（第 50,001 witness），且禁止同时声称 closed frontier、
closure cardinality 或 raw abort。但 sealed verifier 尚未实现，caller-supplied receipt
仍不可信；当前 executed closure status 继续是 `NOT_RUN`。

首个正式 target 是 `TARGET_P3A_GENERIC_ODD_REDUCTION_V1`，在 size 5–8 的完整
480-row universe 上定义，agent-facing split 为 192/96/192。hidden-sink null control
是 `CONTROL_P3A_OBSERVED_OMITTED_SINK_V1`：四个通道全部 observed，仅 auxiliary 被
初始 scope 遗漏，完整 universe 为 85 行。决定稿的 baseline label
`control_volume_primary_only_v1` 不在四成员 scope catalog 中；机器合同暂用
`scope_primary_only_v1` 并保留前者为来源别名，等待规范确认，而不擅自新增
第五个 scope。

二元 XOR 在 executable closure 完成前保持 `TARGET_DESIGN_SANITY_ONLY`；若完整旧 closure
找到任一 480-row extensional match，generic target 自动降为 in-language control，并按
预承诺 registry 选择替代目标。token blacklist 永远不能替代这个判断。

MDL code table 已冻结为 `hegel-mdl-prefix-v1.0.0`，所有长度用向上取整 unsigned Q32，
禁止 binary float；formal scorer 必须忽略 caller-supplied length/Fraction/gain 并从 AST、
partition、prediction 与 code table 重算。certificate 的高层要求是 canonical CBOR、
RFC-6962 Merkle、Python/Rust 双完整 replay 和 3/3 Ed25519，但 canonical CBOR backend /
acceptance、AST/archive/hash/envelope/key/MDL wire 的若干 strict schema 仍需 machine-readable
消歧。当前完整 closure、archive replay、Rust
replay、3/3 certificate 和完整 MDL scorer replay 均未执行，因此 formal outside/MDL gate
硬关闭。正式 claim 只能写
`OUTSIDE_FROZEN_CLOSURE(dsl_version, bounded_universe_root,
target_truth_table_root, equivalence = exact_extensional)`，禁止无边界的
`OUTSIDE_LANGUAGE`。LANGUAGE compiler 和 ACTIVE append 也仍然关闭。
