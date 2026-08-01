# Architecture v0.2

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

## 5. Phase-2 数据流

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

## 6. 诊断与理论生长

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

## 7. Phase-3 之后的接口

`OntologyInadequacyReport`、`TheoryPatch(language)`、`ReductionMap`、
`IdentifiabilityCertificate`、`RobustificationCandidate` 和
`IdealizationContract` 已作为真实接口存在，但当前 benchmark 不以它们证明
“已发明未知定律”。Phase 3 必须另建 hidden-law benchmark、冻结 validation
并通过 expression/non-equivalence test 后才能提升 claim。
