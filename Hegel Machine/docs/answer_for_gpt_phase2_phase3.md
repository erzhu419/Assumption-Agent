# Hegel Machine Phase 2 Exit 与 Phase 3 入口：冻结决策稿

**文档性质**：superseded normative design source（历史保留）
**适用对象**：`Hegel Machine` 当前 Phase-2 typed selector 轨道与即将开始的 bounded Phase-3 hidden-law synthesis
**建议状态**：本稿确认后进入预注册；阈值如需修改，必须在 sealed holdout 生成之前完成并产生新版本号

> **STATUS: `SUPERSEDED_HISTORICAL_DESIGN_SOURCE`**
>
> 本稿正文与其中所有“下一步/施工顺序”保留为历史设计来源，不再是当前执行指令。
> v1.0.2 规范见
> [strict canonical/certificate freeze](Hegel_Machine_Strict_Canonical_AST_CBOR_Certificate_Bridge_Freeze_v1.0.2.md)，
> readiness 见
> [Phase-3 resolution](Hegel_Machine_Phase3_Freeze_Readiness_Resolution.md)。
>
> 当前 M1 Python/Rust shared vectors 各 48/48 PASS；M2 两端均接受 64,680 个 unique
> strict canonical AST，并得到相同 diagnostic set commitment
> `sha256:c1a02a66a8d6d8f75204cb3daf03ab0b01c2b3b8e486d0ab3d481ee3be43c930` 与
> ordinal-50,001 hash
> `sha256:7c7f786c2cc57d31506b3c61d162d175c7f69a2878a089c72c9d053694cba948`。证据为
> [dual strict gate](../artifacts/phase3_dual_strict_gate_v1.json) 和
> [dual strict capacity replay](../artifacts/phase3_dual_strict_capacity_replay_v1.json)。
>
> `hegel-old-dsl-v1.0.0` 在 50,000 syntactic budget 下的 bounded status 是
> `DSL_TOO_LARGE`，不是 `COMPLETE`，没有 extensional target/hidden-sink verdict、formal
> roots、outside/MDL certificate 或 ACTIVE authorization。当前唯一 next action 是发布新
> old-DSL version，按 frozen shrink step 1 删除 `mean_v1`/`min_v1`/`max_v1`，并重建
> target/validation commitments；不得从本稿正文恢复旧施工顺序。

---

## 0. 总结性决定

当前 43-case 系统应准确命名为：

> **Phase-2A Controlled Typed-Selector Mechanics Qualification**

它证明的是：在 verifier-ready、source-visible、人工条件构造的 synthetic witness 上，candidate projection、verifier、binding/scale competitor、normalized margin、abstention 与 preservation 机械闭环能够运行。

它**不是**：

- 正式 Phase-2 exit；
- sealed holdout 结果；
- raw-evidence structural reasoning；
- 从上下文自主推断 scale；
- open-world law discovery；
- meta-prior invention；
- 可进入 ACTIVE 的权威认知结果。

下一步主线应分为三条但不混分：

1. **Phase-2B：Sealed Typed-Evidence Structural Identification Qualification**
   正式验证 typed evidence 上的 family、binding、scale 与 verifier 选择。

2. **Phase-3A/3B：Bounded Language-Inadequacy Detection and Meta-Prior Synthesis**
   在冻结 DSL 中证明旧语言确实不能表达隐藏规律，然后发明最小新关系并保守并入旧语言。

3. **Phase-2R：Raw-Evidence Structuralization Qualification**
   从自然语言、表格或轨迹抽取 family-neutral typed evidence。它可与 Phase 3 并行，不阻塞 typed Phase-3 施工，但阻塞任何 end-to-end raw-evidence claim。

正式 ACTIVE promotion 暂不进入近期认知主线；继续 shadow-only / fail-closed。
大二进制文件今后采用“源码与 manifest 在 Git，二进制在 Release / 对象存储”的分层策略；不重写已经推送的 569 MB 历史。

---

# 1. Phase-2 正式 exit 的输入边界

## 1.1 决定

正式 Phase-2 exit 首先采用 **Typed exit**，但应进一步区分当前能力和正式能力。

### 当前能力的准确名称

> **Explicit-Projection Typed Structural Selection**

含义：

- 输入已经包含若干显式构造的 family × binding × scale candidate projections；
- projection 可以带内部 scale identity；
- selector 负责比较完整候选、调用 verifier、处理 competitor 和决定 answer / abstain；
- 不声称系统从原始上下文自行发现 scale。

### 正式 Phase-2B claim

> **Sealed Typed-Evidence Structural Law Identification and Verification**

正式输入是 family-neutral typed evidence，系统不可获得：

- law family 真值；
- role binding 真值；
- correct scale 真值；
- expected PASS / FAIL；
- candidate-private payload；
- 由 case ID、顺序、容差或字段缺失模式编码的答案。

系统可以获得：

- typed observations；
- opaque entity / role candidates；
- quantity type、单位和测量不确定性；
-时间或空间支持范围；
-任务所要求的预测或决策对象；
-预注册的 aggregation hierarchy / scale transform catalog。

系统必须自行完成：

\[
\text{typed evidence}
\rightarrow
\text{candidate family}
\rightarrow
\text{role binding}
\rightarrow
\text{scale hypothesis}
\rightarrow
\text{law verification}
\rightarrow
\text{answer or abstain}.
\]

## 1.2 Raw-evidence 轨道的准确名称

Raw extractor 单独称为：

> **Phase-2R Raw-Evidence Structuralization Qualification**

若 extractor 与 Phase-2B selector 串联后均通过，可称：

> **End-to-End Raw-Evidence Structural Law Identification**

只有后一名称允许声称系统从自然语言、表格或轨迹开始完成结构识别。

## 1.3 Raw extractor 是否是 Phase 3 的硬前置条件

### 不是以下工作的前置条件

Raw extractor **不阻塞**：

- Phase-3 DSL 设计；
- language-outside certificate；
- parity-like hidden relation；
- typed hidden-law synthesis；
- conservative integration；
- meta-prior invention 的机械验证。

理由是 Phase 3 首先要隔离地回答：

> 在无 extraction noise 的 typed evidence 上，系统能否识别“旧语言不够”并发明新关系？

若先把 raw extraction 混入，会无法判断失败来自抽取还是 invention。

### 是以下 claim 的硬前置条件

Raw extractor 是以下工作的硬前置条件：

- 从真实自然语言/表格/轨迹自主发明关系；
- end-to-end Hegel Machine；
- real-world open-evidence deployment；
- 将 raw residual 自动送入 active meta-prior invention；
- 声称 scale 是从自然上下文中推断，而非从显式 projection 中选择。

## 1.4 一句话边界

\[
\boxed{
\text{Typed Phase 3 可以先施工；raw end-to-end claim 必须后验依赖 Phase-2R。}
}
\]

---

# 2. Phase-2 正式 holdout 的统计协议

## 2.1 数据规模

正式 sealed holdout 建议使用 **720 个独立 latent cases**，不含 preservation pairs。

对于每个：

\[
6\text{ law families}\times2\text{ scales}
\]

的 cell，生成 60 个独立 case：

| Case type | 每个 family × scale |
|---|---:|
| 唯一可识别的 answerable positive | 20 |
| wrong-family / structural hard negative | 8 |
| binding counterfactual | 8 |
| scale counterfactual | 8 |
| sign / direction / invariant-breaking counterfactual | 8 |
| insufficient / genuinely ambiguous，应 abstain | 8 |
| **合计** | **60** |

若某 family 不具有自然 sign，`sign-flip` 必须由预注册的 invariant-breaking transformation 替代，不得删除该 control cell。

独立的含义是：

- 不共享 latent seed；
- 不是同一 case 的简单字段改名；
- 不由同一个 evaluator answer 反向生成；
- pair transformation 只计入 preservation，不重复计入独立 case 数量。

## 2.2 核心指标

### Answerable 子集

必须报告：

- family exact accuracy；
- binding exact accuracy；
- scale set accuracy；
- joint family + binding + scale accuracy；
- correct verifier acceptance；
- strongest competitor rejection。

若一个 case 存在多个理论上等价的 scale，scale 真值必须是预注册 admissible set，不能强迫单一标签。

### Control 子集

必须分别报告：

- hard-negative rejection；
- binding-counterfactual rejection；
- scale-counterfactual rejection；
- sign / invariant-breaking rejection；
- deterministic abstention specificity；
- fail-closed rate。

### Preservation 子集

必须报告：

- pairwise decision preservation；
- score / residual transformation consistency；
- invalid transformation sensitivity。

## 2.3 置信区间

所有 binary proportion 使用：

> **95% one-sided Wilson lower confidence bound**

理由：

- exit gate 关心性能下界；
- 比 Wald interval 稳定；
- 不需要 Bayesian prior 争论；
- 预注册容易。

同时报告 point estimate 和 two-sided 95% interval 作诊断，但 gate 使用 one-sided LCB。

## 2.4 初版冻结阈值

### 总体 gate

| 指标 | 最低点估计 | 95% one-sided LCB |
|---|---:|---:|
| family exact | 0.90 | 0.86 |
| binding exact | 0.90 | 0.86 |
| scale set accuracy | 0.87 | 0.82 |
| joint exact | 0.85 | 0.80 |
| hard-negative rejection | 0.95 | 0.90 |
| binding-CF rejection | 0.95 | 0.90 |
| scale-CF rejection | 0.93 | 0.88 |
| sign/invariant-break rejection | 0.95 | 0.90 |
| abstention specificity | 0.95 | 0.90 |
| fail-closed rate | 1.00 | 不适用 |
| preservation consistency | 0.97 | 0.94 |

### Slice gate

每个 family 和每个 scale 均需：

| 指标 | 最低点估计 | LCB |
|---|---:|---:|
| answerable joint exact | 0.80 | 0.70 |
| all-control rejection | 0.88 | 0.78 |
| abstention specificity | 0.85 | 0.75 |

slice gate 防止一个 family 的失败被其他 family 平均掉。

## 2.5 deterministic abstention 何时升级为概率

Phase-2B 可以保留 deterministic abstention，只要 claim 限于：

> 在冻结规则下决定 answer / abstain。

一旦系统出现以下任一变化，就必须进入概率校准轨道：

- 输出连续 confidence；
- 根据 confidence 调整 coverage；
- 不同风险等级采用不同阈值；
- confidence 被用于 ACTIVE routing；
- 阈值由数据学习而不是固定逻辑产生。

概率轨道至少需要额外的独立 calibration set，建议不少于 500 cases。

### 概率 abstention 的指标优先级

1. **Risk–coverage curve / AURC**：主指标；
2. **Selective risk at frozen coverage**：报告 50%、70%、90% coverage；
3. **Brier score**：用于“最终 joint decision 正确”这一 Bernoulli 事件；
4. **NLL / multiclass Brier**：用于 family posterior；
5. **ECE**：只作诊断，不作唯一 gate，因为受 binning 强烈影响。

正式 gate 应冻结：

- 最大 selective risk；
- 最低 coverage；
- AURC 相对 baseline 的提升；
- calibration split，不得在 sealed test 上选阈值。

## 2.6 是否必须加入真实 semantic / embedding baseline

**必须加入。**

至少包含：

1. 一个真实 embedding cosine / nearest-prototype baseline；
2. 一个冻结 LLM semantic-only classifier；
3. 一个不使用 verifier law 的 flat learned typed baseline，建议 logistic / tree / small MLP 三选一。

它们不是 Phase-2 exit 的 truth oracle，而是用于证明：

> 提升来自结构约束，不只是更强语义模型或更大分类器。

### 结构优势 gate

建立单独的 semantic-conflict challenge subset，包括：

- low-semantic-overlap structural positives；
- high-semantic-overlap structural negatives；
- role-swap；
- sign-flip；
- entity-renaming。

结构系统相对最强 semantic-only baseline 必须满足：

\[
\Delta \text{balanced accuracy}\ge0.15
\]

且 paired bootstrap 的 95% one-sided lower bound：

\[
\operatorname{LCB}(\Delta)\ge0.05.
\]

此外总体 joint accuracy 不得比最强 flat typed baseline 低超过 0.02。

## 2.7 一票否决项

以下任何一项发生，正式 exit 自动无效，不允许用总体 CI 抵消：

- recognizer / selector 读取 answer manifest；
- candidate-private payload；
- case ID、文件名、生成顺序、字段顺序、容差大小可稳定预测答案；
- source-visible schedule 被用于 lookup；
- holdout 打开后修改代码、参数、阈值或 DSL；
- correct competitor 未执行；
- verifier error 时 fail-open；
- deterministic replay 在相同 artifact hash 下产生不同结果；
- evaluator-conditioned PASS/FAIL fixture 进入正式 holdout；
- sealed holdout 被重复打开用于调试；
- exact-preservation transformation 的合法映射出现任何违反；
- metadata replacement 改变结果；
- hidden family 名称或生成模板泄漏到 system prompt。

以下可以用统计 gate：

- 个别 family classification 错误；
- 个别 near-boundary abstain；
- 单个 scale set miss；
- approximate preservation 的少数误差；
- semantic baseline 的个别胜负。

## 2.8 Development、validation、sealed holdout

### Development

- source-visible；
- 可重放；
- 可包含当前 43-case fixtures；
- 可加入极端边界、已知 bug、人工 adversarial；
- 可无限运行；
- 不参与正式 exit 数字。

### Validation

- generator spec 可见，但实例 seed 独立；
- 用于选固定阈值、tolerance normalization 和工程参数；
- 最多允许两轮完整 protocol；
- 每轮必须保存 commit hash、配置和变更原因；
- validation 不得被宣传为 holdout。

### Sealed holdout

- 独立 custodian；
- 在代码、DSL、adapter、threshold、baseline 全部冻结后生成；
- secret master seed、case schedule、role permutation、scale transform 和 answer manifest 不进入 repo；
- 系统只运行一次；
- 输出 hash 后由 custodian 打开答案并评分；
- 失败后不得继续修复再重跑同一 holdout。

建议另保留一个小型 archival reserve，但它不能替代主 sealed holdout。

## 2.9 防 lookup shortcut

custodian 必须保密：

- master seed；
- case family schedule；
- answerable / abstain 比例的局部顺序；
- binding permutation；
- scale transform；
- tolerance stratum；
- hard-negative subtype；
- answer manifest；
- case ID 到 generator state 的映射。

公开输入使用随机 UUID；所有 case 顺序全局 shuffle。
字段顺序 canonicalize 后再随机序列化作 metamorphic test。
任何在所有 case 中恒定或按 family 固定的 witness 不得计入 shared evidence。

---

# 3. Phase-3 “旧语言不能表达”的冻结 DSL

## 3.1 核心决定

第一版 Phase-3 claim 必须是：

> **bounded language-outside relative to a frozen finite DSL closure**

不得声称“绝对不可由旧理论表达”。
所有 non-expressibility 结论只相对于冻结的：

- grammar；
- type system；
- arity；
- AST depth；
- composition depth；
- parameter grid；
- scope budget；
- search budget；
- equivalence tolerance。

## 3.2 Primitive types

第一版 DSL 使用以下 primitive sorts：

```text
Entity
Role
Observation
Event
Index
ScaleContext
EntitySet
Bool
Sign = {-1, 0, +1}
BoundedInt
RationalScalar
IntervalScalar
OrderedCategory
```

`RationalScalar` 必须来自冻结量化网格；正式 hidden holdout 不允许通过任意浮点常数拟合。

## 3.3 Primitive leaves

```text
measurement(entity, quantity_id)
event_value(event, quantity_id)
time_index(event)
space_index(entity)
membership(entity, entity_set)
context_flag(context_id)
task_target(target_id)
uncertainty_interval(measurement)
```

所有 quantity、context 和 target ID 都是 opaque typed IDs，不包含 law 名称。

## 3.4 Operators

### Numeric / order

```text
identity
difference
absolute
sign
sum
mean
count
min
max
affine_combination(coefficients in {-1, 0, +1})
```

### Structural

```text
same_entity
same_role
before
adjacent
subset
aggregate_by(pre-registered scale map)
transform_by(pre-registered transform)
```

### Comparators

```text
approx_equal
less_equal
greater_equal
same_sign
opposite_sign
invariant_equal
within_interval
```

### Logical form

只允许：

```text
top-level conjunction of at most 3 atomic clauses
```

明确禁止：

```text
OR
XOR
modulo
parity
negation of compound expressions
arbitrary truth-table lookup
unbounded recursion
learned neural predicate inside the DSL
case-ID-dependent branch
```

允许 atomic inequality 自带方向，但不允许用 disjunctive normal form 间接编码 parity。

## 3.5 Arity、深度与参数自由度

- ordinary relation arity：最多 3；
- predefined aggregate operator 可接收 bounded `EntitySet`，但不算任意 n-ary predicate invention；
- `EntitySet` 最大大小：8；
- AST depth：最多 4；
- top-level clauses：最多 3；
- old-law composition depth：最多 2；
- fitted scalar parameters：最多 3；
- scope clauses：最多 2；
- scope 每个 clause 只能引用公开 context fields；
- 参数网格：冻结 rational grid；
- 每个 hidden task 最多枚举 50,000 个 canonical old-language programs；
- LLM 可提议程序，但所有提议必须 canonicalize 并计入同一预算。

若完整 closure 超过 50,000，必须缩小 DSL 或把 claim 写为：

> no expression found within the frozen bounded search budget

不能写为 language-outside certificate。

## 3.6 等价与非等价测试

### 1. Syntactic / canonical equivalence

对以下差异归一化：

- alpha-renaming；
- commutative operand order；
- redundant identity；
- duplicate clause；
- equivalent rational normalization；
- scope clause order。

### 2. Observational equivalence

在 development / validation 数据上的输出一致。

它只能说明当前数据不可区分，不能证明语言内等价。

### 3. Extensional equivalence

由于第一版输入域被限制为 finite typed domain，应对整个 bounded universe 枚举 truth table。

若：

\[
P_1(x)=P_2(x),\ \forall x\in\mathcal X_{\mathrm{bounded}}
\]

则在第一版 claim 中视为 extensional equivalent。

### 4. Algebraic equivalence

对 affine、order、interval constraints 使用 canonical symbolic normalization 或 SMT。

只有 verifier 能给出 proof / unsat certificate 时，才可声称 algebraic equivalence / non-equivalence。

## 3.7 参数调整、scope refinement、composition 与 invention 的边界

### Parameter adjustment

- AST 和 relation symbol 不变；
- 只改变已有 parameter；
- 不新增 context field；
- 不改变 arity。

### Scope refinement

- program body 不变；
- 只增加至多 2 个允许的 scope predicates；
- 在旧 scope 交集上预测不变；
- scope support 必须高于预注册最小样本数。

### Low-order composition

- 只使用旧 relation library；
- composition depth ≤ 2；
- AST depth ≤ 4；
- 不引入新 latent object；
- 不引入新 violation functional。

### Invention

只有同时满足以下条件才算 invention：

1. candidate 在旧 DSL closure 中无 extensional equivalent；
2. 不是参数调整；
3. 不是 scope refinement；
4. 不是两层以内旧 law composition；
5. 引入了新 predicate、relation、operator、scale 或 violation functional；
6. 新符号在至少两个独立 case cluster 上复用；
7. 总 MDL 明显降低；
8. 对 preregistered unseen cases 有新预测；
9. 保留旧成功或有明确 conflict boundary。

## 3.8 MDL 编码

### Program length

使用冻结 prefix code：

\[
L(P)=L(\text{AST tokens})+L(\text{identifiers})+L(\text{parameters})+L(\text{scope}).
\]

具体编码：

- 每个 grammar token：\(\lceil \log_2 |V_{\text{token class}}|\rceil\) bits；
- entity / quantity identifier：Elias-delta code；
- rational parameter：sign bit + numerator/denominator grid index；
- scope clause：field index + operator index + value index；
- 新 primitive 额外支付 symbol-definition code。

### Data length

binary deterministic output 使用 enumerative error code：

\[
L(D\mid P)
=
\log_2(n+1)
+
\log_2 {n\choose k}
+
k\log_2(|Y|-1),
\]

其中 \(k\) 为 prediction errors。

总长度：

\[
L_{\mathrm{total}}=L(P)+L(D\mid P).
\]

新关系的最低 compression gain：

\[
\Delta L
\ge
\max(32\text{ bits}, 0.05L(D\mid P_{\mathrm{old}})).
\]

同时必须满足 sealed predictive gate；MDL 不能替代 holdout。

## 3.9 首个 hidden family

**只选择 parity-like 作为第一个真正 language-outside family。**

理由：

- 可以明确冻结为旧 DSL 不包含 XOR / modulo；
- finite domain 下可以穷举 old closure；
- 容易生成低语义、高结构的正负例；
- 可给出严格 outside-language certificate；
- 不易被解释为已有 conservation 参数修正。

### Hidden sink 的角色

hidden sink 不作为第一个 invention target，而作为：

> **in-language refinement / false-invention control**

用途：

- 检查系统是否把 conservation scope/aggregation refinement 错误升格为新关系；
- 检查 ontology inadequacy detector 的 false invention rate。

因此第一轮 Phase 3 应包含：

1. 一个真正 outside-language 的 parity-like target；
2. 一个可以由旧语言 refinement 表达的 hidden-sink null control。

它们不是两个并行 invention targets。

---

# 4. ACTIVE promotion 的可信根

## 4.1 决定

近期选择方向 1：

> Phase 2–3 全部保持 candidate / shadow-only，ACTIVE append 继续硬关闭。

理由：

- 签名只能证明 artifact 来源和不可篡改，不能证明认知正确；
- 引入 key rotation、revocation、custodian authority 会把工程治理与核心认知 benchmark 混合；
- 当前最大科研风险是识别/发明 claim 不成立，而不是 candidate 未被签名；
- 一旦允许 ACTIVE，审稿与安全负担会明显扩大。

## 4.2 近期允许做的最小治理

可以现在加入：

- SHA-256 / BLAKE3 manifest；
- append-only shadow ledger；
- theory version；
- evaluator version；
- data cutoff；
- deterministic replay；
- writer identity 字段保留为空或 `non_authoritative_local`。

不得加入：

- 自动 ACTIVE promotion；
- 无独立 custodian 的正式签名；
- 以本地 key 伪装外部可信根；
- signature-present 即表示 cognitively valid。

## 4.3 何时再启动 ACTIVE 轨道

至少在以下条件满足后：

- Phase-2B sealed typed exit；
- Phase-3B hidden-law invention 通过；
- 独立第三方 replay；
- custodian role 确定；
-撤销与 key rotation 流程存在；
- scoped active rollback 通过；
- cognitive claim 与 governance claim 分开报告。

签名治理应是独立 workstream，不能回填 Phase 2–3 的能力证据。

---

# 5. 文献和 repo 快照的长期归档策略

## 5.1 决定

采用方案 4 的分层版本：

> **源码与文本在普通 Git；大体积不可变二进制在 GitHub Release / 对象存储；仓库保存 checksum manifest。**

不重写已推送的首个 569 MB 历史。

## 5.2 各方案评价

### 普通 Git

优点：

- 固定 commit 离线完整；
- 无额外 hydration；
- 单仓库简单。

缺点：

- 历史永久膨胀；
- clone / fetch 成本持续增加；
- 删除工作树文件也不减少历史；
- 不适合作为长期文献仓库。

结论：只保留当前已经推送的历史，今后不继续。

### Git LFS

适合：

- 需要频繁 checkout 的可变大文件；
- 团队已经接受 LFS 客户端和带宽配额。

问题：

- pointer 与实际 object 分离；
- quota、迁移和长期可用性风险；
- 普通 clone 不是完整离线归档；
- 外部复现环境可能未安装 LFS。

结论：不是默认方案；只用于少数持续变化、开发时必须直接 checkout 的数据。

### GitHub Release / 对象存储

适合：

- 不可变 PDF bundle；
- repo snapshot；
- benchmark binary；
- large result pack；
- rendered artifacts。

仓库保存：

```text
artifact_id
sha256
blake3
byte_length
media_type
origin
license
retrieval_date
related_commit
release_url / object_uri
unpack_command
```

结论：默认大文件策略。

## 5.3 论文与源码分离

### 普通 Git

- source code；
- tests；
- Markdown；
- TeX；
- small JSON / YAML；
- schema；
- checksum manifest；
- small representative fixture。

### Release / object storage

- compiled PDFs；
- external literature snapshots；
- 70–83 MB archives；
- complete synthetic bundles；
- large benchmark outputs；
- environment snapshots；
- container images。

### 论文发表

最终 paper artifact 建议同步到具有 DOI 的归档服务；GitHub Release 作为工程镜像，manifest 记录 DOI 与 hash。

## 5.4 防止再次膨胀

新增：

- repository storage policy；
- pre-commit / CI size gate；
- 10 MB warning；
- 25 MB hard review；
- 50 MB 默认拒绝；
- artifact hydration script；
- `.gitattributes` 只对明确批准的 LFS 类别生效。

不修改当前已推送历史，除非未来 clone 成本已实际阻塞项目并有完整迁移计划。

---

# 6. Phase-2 / Phase-3 里程碑命名与施工顺序

## 6.1 当前成果名称

正式采用：

> **Phase-2A Controlled Typed-Selector Mechanics Qualification**

不使用单独的 “Phase-2 typed selector qualification”，因为容易被理解为正式 Phase-2 exit。

## 6.2 Raw extractor 是否阻塞 Phase 3

不阻塞 Phase-3A / Phase-3B typed synthesis。
作为并行轨道推进。
阻塞 Phase-3C end-to-end raw evidence invention。

## 6.3 Sealed holdout 何时打开

不要把 Phase 2 与 Phase 3 合并成一个 holdout。

### Phase-2B holdout

在以下全部冻结后由 custodian 新生成并打开：

- typed evidence schema；
- adapter；
- candidate generation；
- selector；
- verifier；
- threshold；
- preservation maps；
- baselines；
-统计 protocol。

Phase-3 可以在 Phase-2B holdout 打开之前开始施工。

### Phase-3 holdout

在以下冻结后单独生成：

- old DSL；
- search closure；
- MDL；
- outside-language certificate procedure；
- hidden generator specification；
- null controls；
- conservative integration gate。

原因：两者测试不同 claim，不能共用一个一次性 holdout，也不能因 Phase 2 过早消耗 Phase 3 的隐藏 family。

## 6.4 允许开始 Phase-3 施工的最小条件

允许开始施工，不等于允许发布 Phase-3 claim。

最低条件：

1. Phase-2A mechanics 全部 deterministic replay；
2. fail-closed；
3. correct binding/scale competitors 确实执行；
4. typed evidence interface 冻结为 versioned schema；
5. 当前 43-case 全部降级为 development fixtures；
6. old DSL 设计和 equivalence semantics 已写入预注册草案；
7. Phase-3 hidden generator 与 current selector 代码分离；
8. 不使用 Phase-2 validation/test outcome 设计 hidden answer；
9. ACTIVE 关闭。

不要求 Phase-2B sealed exit。

## 6.5 可以正式宣称 Phase-2 exit 的最小条件

必须同时具备：

- independent typed evidence generator；
- sealed answer manifest；
- untrusted recognizer isolation；
- 一次性 sealed holdout；
- 720-case protocol；
-统计 gates 通过；
- semantic / embedding baselines；
- 强 preservation transforms；
- near-boundary / heterogeneous tolerance；
- anti-lookup audit；
- exact claim naming；
-第三方或独立 custodian replay。

### 允许表述

> The system identifies and verifies a bounded set of structural law families, role bindings, and admissible scales from sealed family-neutral typed evidence, with deterministic abstention and preregistered preservation tests.

### 禁止表述

- understands raw natural language；
- discovers arbitrary laws；
- infers scale from unrestricted context；
- open-world scientific discovery；
- invents universal meta-priors；
- production ACTIVE cognition；
- human-like theory formation。

## 6.6 推荐 5 个里程碑

### M1 — Phase-2A Controlled Typed-Selector Mechanics Qualification

**状态**：当前已完成。
**输入**：source-visible verifier-ready synthetic witness。
**输出**：selector mechanics report、replay artifacts、development fixtures。
**主要失败模式**：fixture-conditioned shortcut、large margin、explicit scale tag。
**Gate**：仅进入下一步施工，不产生正式 exit claim。

### M2 — Phase-2B Sealed Typed-Evidence Structural Identification

**输入**：custodian 生成的 family-neutral typed evidence。
**输出**：

- sealed run manifest；
- predictions；
- verifier residuals；
- baseline comparison；
- CI report；
- preservation report；
- anti-leak report。

**失败模式**：

- answer leakage；
- scale tags；
- family-specific fields；
- semantic shortcut；
- weak boundary；
- no competitor reuse。

**Go**：正式 Phase-2 typed exit。
**No-go**：修复后必须使用新的 holdout，不得重用旧 holdout。

### M3 — Phase-3A Bounded Language-Adequacy and Outside-Language Detection

**输入**：冻结 old DSL、parity-like hidden target、hidden-sink null control。
**输出**：

- DSL closure enumeration；
- equivalence certificates；
- ontology inadequacy report；
- false-invention report；
- old-language best program；
- outside-language certificate。

**失败模式**：

- parity 被旧 DSL 间接表达；
- search budget不足却声称不可表达；
- hidden sink 被错误发明；
- MDL 事后调整。

**Gate**：只有 outside target 识别成功且 null control 不发明，才进入 M4。

### M4 — Phase-3B Bounded Meta-Prior Synthesis and Conservative Integration

**输入**：M3 证明 language-outside 的 cases。
**输出**：

- new relation specification；
- operational semantics；
- violation functional；
- reduction map；
- new predictions；
- old-success replay；
- theory version graph。

**失败模式**：

- verbal-only invention；
- lookup predicate；
-无复用；
- old success 回归；
- no new prediction；
- no MDL gain。

**Gate**：sealed prediction、conservative extension 和 compression 同时通过。

### M5 — Phase-2R / Phase-3C Raw-Evidence End-to-End Qualification

**输入**：natural text / table / trajectory。
**输出**：

- raw evidence；
- structuralized typed bundle；
- provenance map；
- extractor confidence；
- downstream invention result；
- error decomposition。

**失败模式**：

- extraction 与 reasoning 混分；
- hallucinated measurements；
- domain-name leakage；
- semantic pattern lookup；
- uncalibrated scale inference。

**Gate**：分别通过 extraction、typed reasoning、invention，然后才允许 end-to-end claim。

---

# 7. 独立证据合同、scale 语义与 preservation

## 7.1 正式 typed evidence schema

正式 schema 至少包含：

```text
bundle_id: random opaque UUID
observations:
  observation_id
  source_channel
  entity_ids
  role_candidates
  quantity_type
  value or interval
  unit_dimension
  temporal_support
  spatial_support
  measurement_uncertainty
  provenance_hash
context:
  task_target
  available_aggregation_graph
  admissible_transform_catalog
  missingness_mask
```

明确禁止：

```text
law_family
correct_binding
correct_scale
expected_pass
expected_fail
candidate_rank
oracle_margin
family-specific field names
answer-correlated case_id
answer-correlated field order
candidate-private witness
```

## 7.2 信任边界

### Custodian

持有：

- generator master seed；
- answer manifest；
- family schedule；
- role permutation；
- scale truth/admissible set；
- tolerance strata；
- transform mapping；
- final scoring script。

### Evidence generator

只生成 family-neutral evidence 和独立 sealed labels。
不得调用 candidate selector 反向构造“刚好 PASS”的值。

### Frozen adapter

输入 family-neutral evidence，输出统一 candidate-readable typed form。
不得读取 answer manifest。

### Untrusted recognizer / selector

运行于隔离环境：

- read-only input directory；
- no network；
- no repository write；
- no access to generator source；
- no access to answer manifest；
- ephemeral filesystem；
- fixed environment image；
- output only predictions、candidate residuals 和 audit logs。

### Evaluator

在 recognizer 输出 hash 冻结之后，由 custodian 独立执行。
evaluator 不向 recognizer 返回逐例结果。

## 7.3 一次性打开流程

1. 系统 commit、container digest、config hash 冻结；
2. custodian 生成 secret holdout；
3. recognizer offline 执行；
4. 预测和日志 tarball hash；
5. custodian 验证 hash；
6. 打开 sealed answer manifest；
7. 运行评分；
8. 生成不可变 report；
9. holdout 标记 consumed；
10. 任何后续修改使用新 holdout。

## 7.4 当前 scale 能力应如何命名

当前两个显式 scale-tagged projections 中选择，应命名为：

> **Scale-Indexed Candidate Projection Selection**

或：

> **Selection Across Explicitly Declared Scales**

不得称为：

- context-inferred scale；
- autonomous scale discovery；
- scale abstraction learning。

## 7.5 真正 context-conditioned scale selection

输入必须提供：

-原始 observation support；
-单位与 dimension；
- temporal/spatial windows；
- aggregation graph；
- task target；
-允许的 coarse-graining transforms。

禁止提供：

- correct scale ID；
- projection order；
- family-dependent scale field；
- oracle aggregation；
- answer-correlated tolerance。

系统负责：

1. 生成 scale hypotheses；
2. 在各 scale 下构造 candidate measurements；
3. 比较 law residual；
4. 判断唯一 scale、admissible scale set 或 abstain。

### Held-out scale transforms

至少包括：

- temporal aggregation；
- spatial aggregation；
- sampling-resolution change；
- unit conversion；
- coordinate affine transform；
- split/merge equivalent aggregation；
- nontrivial coarse-graining map。

### Scale 指标

- unique-scale exact accuracy；
- admissible-set accuracy；
- cross-scale counterfactual rejection；
- normalized decision regret against oracle scale；
- nonidentifiable-scale abstention；
- scale-transform preservation。

初版 gate：

| 指标 | Point | LCB |
|---|---:|---:|
| admissible-set accuracy | 0.87 | 0.80 |
| cross-scale CF rejection | 0.92 | 0.85 |
| nonidentifiable abstention | 0.90 | 0.82 |
| normalized decision regret | ≤0.05 | bootstrap UCB ≤0.08 |

## 7.6 Preservation 强化

下表中的“family”按 law 的数学性质映射，不依赖名称。

| Transformation | 适用结构 | 最低独立 pairs |
|---|---|---:|
| entity alpha-renaming | 全部 family | 每 family × scale 6 |
| observation reorder | order-invariant / set-based laws | 每适用 family × scale 6 |
| irrelevant entity augmentation | 有明确 scope / locality 的 laws | 每 family × scale 6 |
| unit conversion | numeric laws | 每 numeric family × scale 8 |
| coordinate translation / scaling | invariant/equivariant laws | 每适用 family × scale 8 |
| equivalent aggregation split/merge | conservation/additivity/coverage | 每适用 family × scale 8 |
| nontrivial scale map | 声称跨尺度稳定的 laws | 每适用 family 10 |
| sign convention reparameterization | direction/sign laws | 每适用 family × scale 6 |

所有合法 mapping 在 generator 产生数据前预冻结，不能看结果后决定是否合法。

### 失败判据

对于 exact invariance：

- 任意一个已验证合法 mapping 的 decision flip，正式 exact-preservation gate 失败。

对于 approximate equivariance：

- pair consistency point ≥0.95；
- one-sided LCB ≥0.90；
- transformed residual 必须落入预注册误差带。

Invalid transformation controls 必须导致：

- decision change；
- verifier violation；
- 或 abstain。

否则系统可能只是无条件输出相同答案。

## 7.7 下一版本走哪条轨

下一版本应**直接建立正式轨道**：

> independent shared evidence generator + untrusted recognizer + sealed evaluator

不再把主要资源投入 trusted API selector 的增量强化。

当前 fixtures 可以保留为：

- development regression；
- unit tests；
- numerical stress；
- adapter compatibility；
- public demo；
- known bug reproduction。

当前 fixtures 不得进入：

- threshold calibration；
- formal validation；
- sealed holdout；
- Phase-2 exit CI；
- baseline margin claim；
- Phase-3 hidden-law evidence。

---

# 8. 边界压力与共享证据覆盖

## 8.1 Margin 定义

对每个 candidate \(c\)，先计算 tolerance-normalized violation：

\[
z_c=\frac{v_c}{\tau_c},
\]

其中 \(v_c\) 是 verifier residual，\(\tau_c\) 是 candidate-specific frozen tolerance。

将最优与次优 candidate 排序：

\[
z_{(1)}\le z_{(2)}.
\]

selector margin：

\[
m=z_{(2)}-z_{(1)}.
\]

若 verifier 自身输出 interval，则使用保守 interval margin：

\[
m_{\mathrm{LCB}}
=
\operatorname{LCB}(z_{(2)})
-
\operatorname{UCB}(z_{(1)}).
\]

## 8.2 Holdout margin strata

正式 holdout 预注册比例：

| Stratum | 定义 | 比例 |
|---|---|---:|
| clear interior | \(m_{\mathrm{LCB}}\ge3\) | 35% |
| moderate | \(1\le m_{\mathrm{LCB}}<3\) | 30% |
| near-boundary identifiable | \(0.25\le m_{\mathrm{LCB}}<1\) | 20% |
| structurally ambiguous / insufficient | \(m_{\mathrm{LCB}}<0.25\) 或 interval overlap | 15% |

最后一层 oracle label 必须是 abstain 或 admissible set，不能强制 top-1。

## 8.3 Heterogeneous tolerance

在所有 numeric cases 中：

- 至少 50% 的 strongest candidates tolerance ratio ≥4；
- 至少 20% ratio ≥10；
- tolerance distribution 在 family 之间匹配；
- tolerance 大小不得预测正确 family；
-同一 family 内同时包含 tight 和 loose tolerance；
- adapter 不得看到 tolerance stratum 名称。

## 8.4 Shared evidence footprint

定义 candidate footprint：

\[
F(c)=
\{\text{被 candidate 实际读取的 nonconstant measurement IDs}\}.
\]

不计入：

- 全局常量；
- case metadata；
- family-specific sentinel；
- candidate-private fields；
- 在所有 case 中相同的 measurement。

定义：

\[
\operatorname{shared}(c)
=
\frac{
|F(c)\cap\bigcup_{c'\neq c}F(c')|
}{
|F(c)|
}.
\]

### 每个 case 的 gate

正确 candidate 与 strongest competitor 都必须：

- 共享至少 2 个 nonconstant measurements；
- `shared(c) ≥ 0.60`；
- 至少共享一个 numeric/order/sign measurement；
- footprint size ratio 不超过 3:1；
- candidate-private measurement 数为 0。

### 每个 family × scale × witness-footprint cell 的 gate

- 至少 80% cases 满足 case-level shared gate；
- mean shared fraction ≥0.70；
- 至少 3 种不同 shared footprint template；
- 任何单一 measurement 不得承担超过 50% 的 family discrimination；
- 删除任一 shared measurement 后不得完全暴露答案；
- 不能用一个全局常量 witness 通过 coverage。

## 8.5 三种“边界问题”的区分

### Verifier numerical instability

定义：

- 同一 candidate、同一结构，只改变浮点精度、serialization 或微小数值扰动；
- residual 或 PASS/FAIL 不稳定。

测试：

- float64 / high precision；
- interval arithmetic；
- field order permutation；
- repeated execution；
- value perturbation within measurement uncertainty。

若 fixed candidate 结论改变，是 verifier 问题。

### Selector margin instability

verifier 对每个 candidate 稳定，但：

- strongest candidate 排名随小扰动变化；
- candidate order 改变输出；
- score aggregation 不稳。

若 candidate residual 稳定而选择变化，是 selector 问题。

### Genuine structural ambiguity

- 两个以上 candidate 在高精度下均通过；
- interval residual overlap；
- 当前 evidence 无 probe 可以区分；
- 在合法扰动下都保持成立。

这不是 numerical bug。oracle 必须标为：

```text
admissible_set
或
abstain_due_to_nonidentifiability
```

强迫 top-1 会错误惩罚正确系统。

## 8.6 Development stress 与 sealed boundary

### Development stress suite

允许公开：

- 极端接近 0 的 margin；
- exact tie；
- NaN / overflow / underflow；
- tolerance ratio 100+；
- zero denominator；
- duplicate measurement；
- inconsistent unit；
- known selector bug；
- adversarial field order；
- deliberately malformed evidence。

用途是工程 hardening，不进入正式统计。

### Sealed holdout

必须包含：

- 未见 latent seeds；
- 未见 tolerance combinations；
- 未见 binding permutations；
- 未见 scale transforms；
- 20% near-boundary identifiable；
- 15% genuinely ambiguous；
- representative numerical stress，但不复制公开 bug fixture。

sealed holdout 的边界样本不能在开发集中以相同 generator seed 或相同参数组合出现。

---

# 9. 最终施工优先级

下一轮 Codex 的施工顺序应是：

1. 冻结 `Phase-2B` claim、schema 和 statistical protocol；
2. 建独立 evidence generator / custodian contract；
3. 将 recognizer 置于 untrusted isolated runner；
4. 增加 boundary strata、shared footprint 和 stronger preservation；
5. 加入真实 embedding / semantic baselines；
6. 完成 validation，不打开 sealed holdout；
7. 同时冻结 Phase-3 old DSL、MDL 和 equivalence；
8. 先跑 parity-like outside target + hidden-sink null control；
9. Phase-2B 与 Phase-3 使用独立 sealed holdout；
10. 全程保持 shadow-only。

最重要的顺序原则：

\[
\boxed{
\text{先冻结“什么算旧语言、什么算证据、什么算失败”，再让系统发明新语言。}
}
\]

若这三项未冻结，Phase 3 的“发明”无法与事后修改 evaluator、扩大旧语言或 lookup patch 区分。

---

# 10. 最终 claim ladder

## 当前可说

> Hegel Machine v0.2 has completed a controlled typed-selector mechanics qualification on verifier-ready synthetic fixtures.

## Phase-2B 通过后可说

> The system identifies and verifies a bounded set of structural laws, role bindings, and admissible scales from sealed family-neutral typed evidence, with preregistered abstention, counterfactual, preservation, and anti-leak tests.

## Phase-3A 通过后可说

> The system detects, relative to a frozen bounded DSL, when a hidden relation is outside the old language while avoiding false invention on an in-language refinement control.

## Phase-3B 通过后可说

> The system synthesizes a bounded new relation with executable semantics, improves description length and unseen prediction, preserves prior successes, and supplies a reduction map to the parent theory.

## Phase-2R / Phase-3C 通过后才可说

> The system performs end-to-end bounded meta-prior invention from raw natural-language, tabular, or trajectory evidence.

任何阶段均不得声称：

- arbitrary scientific discovery；
- universal philosophy generation；
- unbounded self-evolution；
- human-level theory formation；
- formal proof of all natural-language relations；
- production ACTIVE cognition。
