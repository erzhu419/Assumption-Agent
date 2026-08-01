# Hegel Machine Phase-2B / Phase-3 精确冻结决策

> **STATUS: `HISTORICAL_V1_0_1_DECISION_SOURCE`**
>
> 本文件的 v1.0.1 正文、状态字段和施工顺序按原样保留，用于审计 v1.0.2 之前的决策；
> 它们不再描述当前 operational state。当前规范由
> [v1.0.2 strict canonical/certificate amendment](Hegel_Machine_Strict_Canonical_AST_CBOR_Certificate_Bridge_Freeze_v1.0.2.md)
> 与 [Phase-3 readiness resolution](Hegel_Machine_Phase3_Freeze_Readiness_Resolution.md)
> supersede。
>
> M1 Python/Rust shared vectors 各 48/48 PASS。M2 两端都把 64,680 个 source candidates
> 接受为 64,680 个 unique strict canonical AST；共同 diagnostic set commitment 是
> `sha256:c1a02a66a8d6d8f75204cb3daf03ab0b01c2b3b8e486d0ab3d481ee3be43c930`，ordinal
> 50,001 AST hash 是
> `sha256:7c7f786c2cc57d31506b3c61d162d175c7f69a2878a089c72c9d053694cba948`。执行证据见
> [dual strict gate](../artifacts/phase3_dual_strict_gate_v1.json) 与
> [dual strict capacity replay](../artifacts/phase3_dual_strict_capacity_replay_v1.json)。
>
> 因此 `hegel-old-dsl-v1.0.0` 在 50,000 syntactic budget 下的 bounded status 已是
> `DSL_TOO_LARGE`，但不是 `COMPLETE`；没有 extensional target/hidden-sink verdict，formal
> roots 仍为 `null`，没有 outside/MDL certificate 或 ACTIVE authorization。当前唯一后继是
> 发布新 old-DSL version，按 frozen shrink step 1 删除 `mean_v1`、`min_v1`、`max_v1`，
> 重建 target/validation commitments，并让新版本从 `NOT_RUN` 开始。下方 v1.0.1 正文
> 中的旧 `NOT_RUN`、conditional capacity 与 next-step 表述必须按历史语境阅读。

**文档类型**：Normative freeze decision<br>
**适用范围**：Phase-2B sealed typed-evidence qualification、Phase-3A frozen-language inadequacy、Phase-3B bounded meta-prior synthesis<br>
**版本**：`hegel-freeze-p2b-p3-v1.0.1`<br>
**状态**：v1.0.1 冻结本文明确列出的 surface/high-level 参数；strict canonical
acceptance 与 certificate wire 仍未闭合，因而 normative freeze 不完整。任何已冻结参数的
改变必须提升版本号并生成全新的 validation / holdout artifact。

> **2026-08-01 implementation-audit amendment**：本文件冻结的是设计层表格与参数，
> 不代表 strict canonical AST / CBOR、完整 certificate wire schema 或跨语言 Q32
> replay 已经冻结或实现。机器状态必须称为 `surface_parameter_freeze`；在文末列出的
> schema blockers 解决前，`normative_parameter_freeze_complete=false`，closure 仍为
> `NOT_RUN`，不得发布 `DSL_TOO_LARGE` 或任何 outside certificate。
>
> 同一 amendment 将 overall freeze 从 v1.0.0 提升到 v1.0.1，并 supersede v1.0.0。
> 原 64-bit 值 `411876909552964556` 继续作为 master/bootstrap seed；由于 sklearn
> `random_state` 只接受 uint32，flat typed baseline 改用冻结的 domain-separated
> SHA-256 → uint32 值 `2611585425`。这不是实现自行截断。

---

## 0. 总体判断：当前方向没有跑偏

根据当前问题暴露出来的实现状态，方向仍然正确，而且比上一版更严谨：

1. 已经主动发现二元 XOR 和低元 parity 可能被旧 DSL 的 `absolute(difference(...))` 表达；
2. 已经不再把“禁止出现 XOR 这个 token”误当成 language-outside 证明；
3. 已经把 covert-answer-channel 从字段 allowlist 问题提升为信息泄漏问题；
4. 已经意识到 self-reported closure receipt 不能支持正式 outside-language claim；
5. 已经把 complete closure、MDL、sealed replay 和独立证书作为 Phase-3 claim 的前置条件。

这说明当前工作正在从：

> “实现一个看起来会发明关系的 agent”

转向：

> “在冻结语言、冻结证据和冻结评价合同下，证明某个关系确实不在旧语言中，并证明新关系不是 lookup、事后扩展或 evaluator 漂移的产物。”

这正是正确方向。

但有三条边界必须继续保持：

- cryptographic integrity 只能证明 artifact 未被替换，不能证明科学结论正确；
- `OUTSIDE_FROZEN_CLOSURE` 只相对于指定 DSL、bounded universe 和等价定义成立；
- 首个 parity target 是 mechanism benchmark，不是“系统已发现深层自然定律”。

---

# 1. 720 case table 与 margin strata 的 12-case 冲突

## 1.1 决定

选择方案 1：

> 每个 `family × scale` cell 的 20 个 positive 中，抽出 1 个改成 set-valued answerable case。

原来的“20 个唯一可识别 positive”改名为：

> **20 answerable structural-positive cases**

并拆成：

- 19 个 `unique-scale answerable`；
- 1 个 `admissible-scale-set answerable`。

## 1.2 冻结后的每 cell 配额

每个 `family × scale` cell 共 60 例：

| `case_type` | 每 cell | 12 cells 总数 |
|---|---:|---:|
| `unique_scale_answerable` | 19 | 228 |
| `admissible_scale_set_answerable` | 1 | 12 |
| `wrong_family_hard_negative` | 8 | 96 |
| `binding_counterfactual` | 8 | 96 |
| `scale_counterfactual` | 8 | 96 |
| `sign_or_invariant_break` | 8 | 96 |
| `insufficient_or_nonidentifiable` | 8 | 96 |
| **总计** | **60** | **720** |

## 1.3 冻结后的 margin strata

每个 cell 的 margin 配额：

| `margin_stratum` | 每 cell | 全局 | 比例 |
|---|---:|---:|---:|
| `clear_interior` | 21 | 252 | 35% |
| `moderate` | 18 | 216 | 30% |
| `near_boundary_identifiable` | 12 | 144 | 20% |
| `nonunique_or_insufficient` | 9 | 108 | 15% |
| **总计** | **60** | **720** | **100%** |

最后一层 9 例由：

- 8 个 `insufficient_or_nonidentifiable`；
- 1 个 `admissible_scale_set_answerable`

组成。

这里的 `nonunique` 不等于 `unanswerable`。<br>
`admissible_scale_set_answerable` 的正确答案是一个预注册集合，而不是 abstain。

## 1.4 指标分母

| 指标 | 是否包含 `admissible_scale_set_answerable` |
|---|---|
| `answerable_count` | 是 |
| `family_exact_accuracy` | 是 |
| `binding_exact_accuracy` | 是 |
| `scale_set_accuracy` | 是 |
| `unique_scale_accuracy` | 否 |
| `joint_exact_accuracy` | 是 |
| `abstention_specificity` | 否 |
| `nonidentifiability_abstention_accuracy` | 否 |
| `set_valued_answer_accuracy` | 是，单独报告 |

### `joint_exact` 的定义

对 set-valued case，只有同时满足：

```text
predicted_family == gold_family
predicted_binding == gold_binding
predicted_scale_set == gold_admissible_scale_set
decision == ANSWER_SET
```

才计为 joint correct。

输出单个 scale、超集、子集或 abstain 均计错。

## 1.5 Machine-readable 配置

```json
{
  "holdout_case_count": 720,
  "cell_axes": {
    "law_family_count": 6,
    "scale_count": 2,
    "cell_count": 12,
    "cases_per_cell": 60
  },
  "case_type_quota_per_cell": {
    "unique_scale_answerable": 19,
    "admissible_scale_set_answerable": 1,
    "wrong_family_hard_negative": 8,
    "binding_counterfactual": 8,
    "scale_counterfactual": 8,
    "sign_or_invariant_break": 8,
    "insufficient_or_nonidentifiable": 8
  },
  "margin_quota_per_cell": {
    "clear_interior": 21,
    "moderate": 18,
    "near_boundary_identifiable": 12,
    "nonunique_or_insufficient": 9
  }
}
```

---

# 2. 首个高元 parity target 的精确定义

## 2.1 决定

首个 target 冻结为：

> **Generic Odd-Cardinality Reduction over Bounded Entity Sets**

内部 ID：

```text
TARGET_P3A_GENERIC_ODD_REDUCTION_V1
```

不再使用模糊的 `parity-like` 作为正式 target 名称。

## 2.2 输入类型

输入为：

```text
EntitySet S
Bit-valued measurement b(e) ∈ {0,1}, for each e ∈ S
|S| ∈ {5,6,7,8}
```

输出为：

\[
y(S)=
\begin{cases}
1,&\sum_{e\in S}b(e)\text{ 为奇数},\\
0,&\sum_{e\in S}b(e)\text{ 为偶数}.
\end{cases}
\]

目标必须满足：

- permutation invariant；
- 对所有 set size 使用同一关系；
- 不允许为不同 set size 编写独立 lookup；
- 不允许读取 entity ID 的数值或 lexical pattern。

## 2.3 完整 bounded universe

\[
\mathcal U_{\text{parity}}
=
\bigcup_{n=5}^{8}\{0,1\}^{n}.
\]

总行数：

\[
2^5+2^6+2^7+2^8=480.
\]

按 set size 的行数：

| Set size | Universe rows |
|---:|---:|
| 5 | 32 |
| 6 | 64 |
| 7 | 128 |
| 8 | 256 |
| **总计** | **480** |

每个 size 内正负标签严格各半。

## 2.4 Discovery / validation / sealed prediction split

每个 size、每个 label 内使用独立 HMAC 排序：

```text
rank_key =
HMAC-SHA256(
  K_split,
  target_id || set_size || label || canonical_bitstring
)
```

然后按下表分配：

| Set size | Discovery train | Validation | Sealed prediction |
|---:|---:|---:|---:|
| 5 | 12 | 6 | 14 |
| 6 | 26 | 12 | 26 |
| 7 | 52 | 26 | 50 |
| 8 | 102 | 52 | 102 |
| **总计** | **192** | **96** | **192** |

每一个 cell 内 label 仍保持 50/50：

| Set size | 每 label train | 每 label validation | 每 label sealed |
|---:|---:|---:|---:|
| 5 | 6 | 3 | 7 |
| 6 | 13 | 6 | 13 |
| 7 | 26 | 13 | 25 |
| 8 | 51 | 26 | 51 |

## 2.5 480 行 truth table 的用途

480 行完整 truth table **不提供给 synthesis agent**。

其用途是：

- 对 frozen old DSL 做 extensional closure comparison；
- 生成 `target_truth_table_root`；
- 判断是否存在旧语言精确等价程序；
- 独立 replay certificate。

Discovery / validation / sealed prediction split 用于评估 invention agent；<br>
完整 480 行用于语言成员资格证明。二者不得混淆。

## 2.6 二元 XOR 的状态

在 executable operator semantics 和完整 DSL 尚未冻结之前：

```text
binary_xor_status =
TARGET_DESIGN_SANITY_ONLY
```

不能签发：

```text
IN_LANGUAGE
```

或：

```text
OUTSIDE_LANGUAGE
```

冻结后，只有以下机器证据可将二元 XOR 判为 `IN_LANGUAGE`：

1. closure enumeration 状态为 `COMPLETE`；
2. 至少一个 canonical old-DSL program：
   ```text
   absolute(difference(bit_at(0), bit_at(1)))
   ```
   或其 canonical equivalent；
3. 在完整四行 universe `{00,01,10,11}` 上输出与 XOR truth table 完全一致；
4. 无 `undefined`；
5. Python 与 Rust replay 得到相同 program hash 和 output root。

## 2.7 Generic target 的成员资格

若 old closure 中存在程序 \(P\)，满足：

\[
P(x)=y(x),\quad\forall x\in\mathcal U_{\text{parity}},
\]

则：

```text
target_language_status = IN_LANGUAGE
```

无论该程序是否包含 `xor`、`parity` 或 `modulo` 字样。

只检查 token 名称是无效的。

## 2.8 若旧 closure 找到等价表达

自动执行：

```text
ABANDON_AS_OUTSIDE_TARGET
```

该 target 可以保留为：

```text
IN_LANGUAGE_POSITIVE_CONTROL
```

不得继续用于 invention claim。

替代 target 必须来自预先承诺的 registry：

| Priority | Hidden target |
|---:|---|
| 1 | odd cardinality: `count mod 2 == 1` |
| 2 | residue class: `count mod 3 == 1` |
| 3 | prime cardinality: `count ∈ {2,3,5,7}`，在 size 5–8 universe 上评估 |

选择规则：

```text
选择 registry 中优先级最低数字、
且在 frozen closure 中 match_set_count == 0、
且 positive prevalence ∈ [0.25, 0.75] 的第一个 target。
```

目标 registry、truth-table commitments 和优先级必须在 synthesis agent 看到任何 hidden outcome 前冻结。

---

# 3. Hidden sink null control 的可观测性

## 3.1 决定

hidden sink 必须是：

> 已经存在于 typed observations 中，但被初始 scope / aggregation 排除的 observed auxiliary channel。

它不是 latent variable，也不是未观测变量。

正式名称：

> **Observed Omitted-Channel Conservation Refinement Control**

内部 ID：

```text
CONTROL_P3A_OBSERVED_OMITTED_SINK_V1
```

## 3.2 数据结构

每个 case 有四个 observed channels：

```text
inflow_a
inflow_b
primary_outflow
auxiliary_outflow
```

所有值均出现在 public typed evidence 中。

生成约束：

\[
a+b-c-d=0.
\]

其中：

```text
a,b,c,d ∈ {0,1,2,3,4}
d = a + b - c
0 <= d <= 4
```

bounded universe 共 85 行。

## 3.3 正确 old-DSL program

```text
approx_equal(
  aggregate_by(
    map_id = signed_balance_v1,
    scope_id = control_volume_all_observed_v1,
    quantity_id = q0
  ),
  0,
  tolerance = 0
)
```

等价展开：

\[
(+a)+(+b)+(-c)+(-d)=0.
\]

baseline scope：

```text
control_volume_primary_only_v1
```

只包含 \(a,b,c\)，故产生 residual：

\[
a+b-c=d.
\]

正确修复是 scope / aggregation refinement，不是新 law。

## 3.4 aggregation map

```json
{
  "map_id": "signed_balance_v1",
  "input_type": "EntitySet",
  "output_type": "Rational",
  "term_rule": "sum(orientation(entity) * measurement(entity, quantity_id))",
  "orientation_domain": [-1, 1],
  "undefined_if": [
    "missing_orientation",
    "missing_measurement",
    "quantity_mismatch"
  ]
}
```

正确 scope：

```json
{
  "scope_id": "control_volume_all_observed_v1",
  "clauses": [
    ["control_volume_id", "eq", "current_control_volume"],
    ["quantity_id", "eq", "q0"]
  ],
  "include_auxiliary": true
}
```

`include_auxiliary` 是 scope catalog 的冻结成员，不是 agent 新增的 primitive。

## 3.5 support 下限

scope refinement 进入可接受候选至少需要：

```text
discovery_support_total >= 16
discovery_support_per_scale >= 8
validation_support_total >= 8
sealed_support_total >= 8
```

并且 discovery 每个 scale 至少包含：

```text
4 cases with d = 0
4 cases with d > 0
```

防止仅根据“始终存在 sink”作 lookup。

## 3.6 no-false-invention gate

null control 通过必须同时满足：

```text
old_closure_exact_match_count >= 1
best_old_program_error == 0
decision == IN_LANGUAGE_REFINEMENT
promoted_new_symbol_count == 0
outside_language_certificate_count == 0
```

在 sealed null controls 中：

```text
false_invention_rate == 0
```

若系统生成新 relation 但最终 promotion gate 阻止它，可以记录为 proposal；<br>
若写入 candidate theory 或签发 outside certificate，则 gate 失败。

---

# 4. 50,000 search budget 的计数口径

## 4.1 决定

50,000 指：

> **syntactically canonical programs before extensional quotient**

即：

```text
max_canonical_program_count = 50000
```

不指 extensional equivalence representatives。

extensional dedup 只能用于：

- 分析；
- archive 压缩；
- equivalent representative 映射；

不能用于把超过 50,000 的 grammar 伪装成完整 bounded closure。

## 4.2 完整 closure 的定义

只有满足：

```text
enumeration_frontier_exhausted == true
all_type_buckets_closed == true
canonical_program_count <= 50000
raw_expansion_limit_not_hit == true
wall_clock_abort_not_hit == true
```

才可标记：

```text
closure_status = COMPLETE
```

若出现第 50,001 个 canonical program：

```text
closure_status = DSL_TOO_LARGE
```

若 raw node expansion cap 被触发：

```text
closure_status = INCONCLUSIVE_BUDGET
```

两种情况都不得签发 outside certificate。

## 4.3 raw node-expansion cap

```text
max_raw_operator_applications = 5,000,000
```

一次 raw expansion 定义为：

> 对一个 operator token 和一个类型合法的 child tuple 尝试构造父 AST，不论最后是否被 canonicalizer 去重或拒绝。

## 4.4 Canonical traversal order

排序键按顺序为：

```text
1. total_ast_depth ascending
2. total_node_count ascending
3. output_sort_id ascending
4. root_operator_id ascending
5. canonical_ast_cbor bytes lexicographically ascending
```

动态规划 bucket：

```text
(output_sort, depth, node_count)
```

所有 commutative operator 的 children 必须按 child canonical hash 排序。

## 4.5 replay 必须绑定的 roots

```text
dsl_spec_root
operator_semantics_root
identifier_registry_root
bounded_universe_root
canonicalizer_source_root
enumerator_source_root
canonical_program_archive_root
program_output_archive_root
target_truth_table_root
chunk_manifest_root
enumeration_exhaustion_receipt_root
container_image_digest
repository_commit_sha
```

---

# 5. Frozen old DSL v1

## 5.1 定位

版本：

```text
hegel-old-dsl-v1.0.0
```

它是：

> Phase-3A mechanism benchmark 的有限旧语言。

它不是通用科学 DSL，也不是最终 Hegel Machine ontology。

## 5.2 Primitive sorts 与 cardinality

| Sort | Domain | Cardinality |
|---|---|---:|
| `Bool` | `{false,true}` | 2 |
| `Bit` | `{0,1}` | 2 |
| `Sign` | `{-1,0,+1}` | 3 |
| `BoundedInt` | integers `[-8,8]` | 17 |
| `RationalValue` | reduced `p/q`, `|p|<=64`, `1<=q<=8` | 663 |
| `RationalParameter` | `{-2,-1,-1/2,0,1/2,1,2}` | 7 |
| `Tolerance` | `{0,1/4,1/2}` | 3 |
| `IntervalEndpoint` | `{-8,-4,-2,-1,0,1,2,4,8}` | 9 |
| `ClosedInterval` | ordered pairs `lo<=hi` | 45 |
| `EntitySlot` | `e0...e7` | 8 |
| `Index` | `0...7` | 8 |
| `QuantityId` | `q0,q1` | 2 |
| `ContextId` | `c0,c1,c2,c3` | 4 |
| `RoleId` | `r0,r1,r2,r3` | 4 |
| `ScaleId` | `s0,s1` | 2 |
| `TaskId` | `t0,t1` | 2 |
| `ScopeId` | frozen catalog below | 4 |
| `AggregateMapId` | frozen catalog below | 6 |
| `TransformId` | adapter-only catalog below | 4 |

`RationalValue` 加一个 internal bottom：

```text
⊥
```

但 `⊥` 不算合法 observable value。

## 5.3 Identifier registries

```json
{
  "entity_slots": ["e0","e1","e2","e3","e4","e5","e6","e7"],
  "quantity_ids": ["q0","q1"],
  "context_ids": ["c0","c1","c2","c3"],
  "role_ids": ["r0","r1","r2","r3"],
  "scale_ids": ["s0","s1"],
  "task_ids": ["t0","t1"]
}
```

这些名称只在 private canonical form 中存在。public wire 使用独立 opaque IDs。

## 5.4 Scope catalog

| Scope ID | Semantics |
|---|---|
| `scope_all_observed_v1` | 全部 observed entities |
| `scope_primary_only_v1` | 排除 `auxiliary=true` |
| `scope_boundary_only_v1` | 仅 `boundary_member=true` |
| `control_volume_all_observed_v1` | 同一 control volume、同 quantity，包含 auxiliary |

最多允许额外 2 个 context clauses，但只能从冻结 `ContextId` registry 中选取。

## 5.5 Aggregate catalog

| Map ID | Type | Semantics |
|---|---|---|
| `sum_v1` | `EntitySet × Quantity -> Rational` | values sum |
| `count_nonzero_v1` | `EntitySet × Quantity -> BoundedInt` | nonzero count |
| `mean_v1` | `EntitySet × Quantity -> Rational` | exact rational mean |
| `min_v1` | `EntitySet × Quantity -> Rational` | minimum |
| `max_v1` | `EntitySet × Quantity -> Rational` | maximum |
| `signed_balance_v1` | `EntitySet × Quantity -> Rational` | `sum(orientation × value)` |

## 5.6 Transform catalog

Transform catalog 只用于 adapter / preservation，不作为 old DSL 可组合 operator：

| Transform ID | Semantics |
|---|---|
| `identity_v1` | \(x\mapsto x\) |
| `negate_v1` | \(x\mapsto -x\) |
| `scale_by_2_v1` | \(x\mapsto2x\) |
| `scale_by_half_v1` | \(x\mapsto x/2\) |

这样可防止 transform composition 暗中扩大旧 closure。

## 5.7 Leaf expressions

```text
scalar_const(parameter) -> Rational
bit_at(index) -> Bit
set_size() -> BoundedInt
aggregate(map_id, scope_id, quantity_id) -> Rational | BoundedInt
context_flag(context_id) -> Bool
task_flag(task_id) -> Bool
```

## 5.8 Unary operators

```text
bit_to_scalar(Bit) -> Rational
int_to_scalar(BoundedInt) -> Rational
absolute(Rational) -> Rational
sign(Rational) -> Sign
```

## 5.9 Binary operators

```text
add(Rational, Rational) -> Rational
difference(Rational, Rational) -> Rational
equal_exact(Rational, Rational) -> Bool
less_equal(Rational, Rational) -> Bool
greater_equal(Rational, Rational) -> Bool
same_sign(Sign, Sign) -> Bool
opposite_sign(Sign, Sign) -> Bool
```

## 5.10 Ternary operator

```text
approx_equal(Rational, Rational, Tolerance) -> Bool
```

## 5.11 Boolean composition

只允许顶层 conjunction：

```text
AND(atom_1)
AND(atom_1, atom_2)
AND(atom_1, atom_2, atom_3)
```

禁止：

```text
OR
XOR
NOT(compound)
modulo
parity
arbitrary lookup table
recursive fold
user-defined reducer
case-ID branch
```

## 5.12 Structural limits

```json
{
  "max_total_ast_depth": 4,
  "max_total_node_count": 7,
  "max_top_level_clauses": 3,
  "max_distinct_bit_slots": 4,
  "max_aggregate_leaves": 1,
  "max_scope_clauses": 2,
  "max_old_law_composition_depth": 2,
  "max_fitted_scalar_parameters": 3
}
```

深度定义：

```text
leaf depth = 0
operator depth = 1 + max(child depth)
```

## 5.13 Undefined semantics

所有 operator 使用 strict bottom propagation：

```text
if any required child == ⊥:
    output = ⊥
```

具体规则：

- `bit_at(i)=⊥` 当 `i >= |S|`；
- empty-set `mean/min/max = ⊥`；
- missing measurement、orientation 或 quantity mismatch -> `⊥`；
- exact rational result 不属于 `RationalValue` -> `⊥`；
- `sign(0)=0`；
- no NaN、no infinity、no implicit float rounding。

一个 program 若在 bounded universe 任一行输出 `⊥`，不得成为 exact extensional match。

## 5.14 Equivalence tolerance

### Boolean / Bit output

```text
exact equality only
tolerance = 0
```

### Rational output

closure extensional equivalence：

```text
exact Fraction equality only
```

`approx_equal` 是 DSL 内 operator，不改变 closure equivalence 的 exactness。

## 5.15 Scope support

```text
scope_min_discovery_support = 16
scope_min_discovery_per_scale = 8
scope_min_validation_support = 8
scope_min_sealed_support = 8
```

## 5.16 若 closure 超过 50,000

不得在同一版本里临时缩小。

必须：

1. 终止为 `DSL_TOO_LARGE`；
2. 发布新的 DSL version；
3. 按下列顺序缩小；
4. 重新生成 target commitments 和所有 validation artifacts。

冻结 shrink order：

```text
1. remove mean_v1, min_v1, max_v1
2. reduce RationalParameter to {-1,0,1}
3. remove add; retain difference
4. reduce max_top_level_clauses from 3 to 2
5. reduce max_total_node_count from 7 to 6
6. reduce max_total_ast_depth from 4 to 3
```

不得在查看 synthesis hidden outcome 后选择 shrink step。

---

# 6. MDL code table

## 6.1 版本

```text
mdl_code_table_id = hegel-mdl-prefix-v1.0.0
fixed_point_unit = 2^-32 bits
```

## 6.2 AST shape prefix code

每个 node 使用 preorder：

| Node shape | Prefix |
|---|---|
| leaf | `00` |
| unary | `01` |
| binary | `10` |
| ternary | `110` |
| top-level AND length 1 | `1110` |
| top-level AND length 2 | `11110` |
| top-level AND length 3 | `111110` |
| reserved | `111111` |

operator arity 由 shape code 冻结；children 紧随其后按 preorder 编码。

## 6.3 Token classes

### Leaf class：3 bits

| ID | Leaf |
|---:|---|
| 0 | `scalar_const` |
| 1 | `bit_at` |
| 2 | `set_size` |
| 3 | `aggregate` |
| 4 | `context_flag` |
| 5 | `task_flag` |
| 6 | `new_symbol_call` |
| 7 | reserved |

### Unary token：2 bits

| ID | Operator |
|---:|---|
| 0 | `bit_to_scalar` |
| 1 | `int_to_scalar` |
| 2 | `absolute` |
| 3 | `sign` |

### Binary token：3 bits

| ID | Operator |
|---:|---|
| 0 | `add` |
| 1 | `difference` |
| 2 | `equal_exact` |
| 3 | `less_equal` |
| 4 | `greater_equal` |
| 5 | `same_sign` |
| 6 | `opposite_sign` |
| 7 | reserved |

### Ternary token：1 bit

| ID | Operator |
|---:|---|
| 0 | `approx_equal` |
| 1 | reserved |

## 6.4 Identifier code

所有 registry index 从 1 开始，使用 Elias-delta code：

\[
L_\delta(n)
=
\lfloor\log_2 n\rfloor
+
2\lfloor\log_2(\lfloor\log_2 n\rfloor+1)\rfloor
+1.
\]

绑定 registry root：

```text
identifier_registry_root
```

名称字符串本身不计长度，只计冻结 index。

## 6.5 Rational parameter code

`RationalParameter` 使用固定 3-bit index：

| Bits | Value |
|---|---:|
| `000` | -2 |
| `001` | -1 |
| `010` | -1/2 |
| `011` | 0 |
| `100` | 1/2 |
| `101` | 1 |
| `110` | 2 |
| `111` | reserved |

Tolerance 使用 2 bits：

| Bits | Value |
|---|---:|
| `00` | 0 |
| `01` | 1/4 |
| `10` | 1/2 |
| `11` | reserved |

## 6.6 Scope code

Clause count：

| Count | Code |
|---:|---|
| 0 | `0` |
| 1 | `10` |
| 2 | `11` |

每个 clause：

```text
ContextId: 2 bits
expected Bool: 1 bit
```

基础 `ScopeId` 使用 2 bits。

## 6.7 Aggregate leaf code

```text
leaf-shape
leaf-class aggregate
AggregateMapId: 3 bits
ScopeId: 2 bits
QuantityId: 1 bit
scope-extension-code
```

## 6.8 New-symbol definition code

Phase-3B v1 只允许一种新 symbol class：

> bounded generic reducer

编码：

```text
header "NEW_REDUCER_V1"                  16 bits
arity                                  Elias-delta
input sort IDs                         4 bits each
output sort ID                         4 bits
reduction scheme:
  left_fold                            0
  balanced_fold                        1
identity RationalParameter             3 bits
binary combiner AST                    ordinary AST prefix code
maximum supported set size             4 bits
scope code                             as above
verifier specification hash reference 256 bits
```

如果 candidate 不能表示为该 reducer class，只能标记：

```text
UNSUPPORTED_INVENTION_FORM
```

不得自由提交自然语言新 symbol。

## 6.9 Data code

第一版 Phase-3 target 为 binary / Bit output，使用 enumerative error code：

\[
L(D\mid P)
=
\log_2(n+1)
+
\log_2 {n\choose k}
+
k\log_2(|Y|-1).
\]

因 \(|Y|=2\)：

\[
k\log_2(1)=0.
\]

其中：

- \(n\)：scoring partition rows；
- \(k\)：program error count；
- `⊥` 计作 error。

## 6.10 四个 MDL 长度

证书必须重算：

```text
L_old_program
L_train_given_old
L_new_model
L_train_given_new
```

其中：

```text
L_new_model =
L_new_symbol_definition
+
L_new_call_program
```

压缩增益：

\[
\Delta L
=
[L_{\text{old program}}+L(D_{\text{train}}\mid P_{\text{old}})]
-
[L_{\text{new model}}+L(D_{\text{train}}\mid P_{\text{new}})].
\]

gate：

\[
\Delta L
\ge
\max\left(32,\;0.05L(D_{\text{train}}\mid P_{\text{old}})\right).
\]

## 6.11 `log2` 精度和比较

- 组合数使用 exact integer；
- `log2` 使用 80-decimal-digit precision；
- 每个长度向上取整到 \(2^{-32}\) bit；
- 内部存储 unsigned Q32 integer；
- 所有 gate 比较直接比较 Q32 integer；
- 禁止 binary float；
- 禁止 epsilon comparison；
- equality 只认 integer equality。

## 6.12 Invention split

| Partition | Rows | Agent access | 用途 |
|---|---:|---|---|
| discovery train | 192 | 可见 | proposal、train MDL |
| validation | 96 | outcome 对 generator 不可见 | frozen candidate selection |
| sealed prediction | 192 | 完全不可见 | final predictive gate |
| full truth table | 480 | synthesis agent 不可见 | old closure certificate |

MDL threshold 只使用 discovery train。<br>
validation 和 sealed prediction 是强制外部 predictive gates，不能被 MDL 替代。

---

# 7. Phase-2B 剩余统计与运行细节

## 7.1 Canonical family IDs

若当前代码使用不同名称，必须通过 frozen mapping 映射到以下六类：

```text
F01_symmetry_equivariance
F02_monotonicity_order
F03_conservation_balance
F04_additivity_complementarity
F05_locality_composition
F06_negative_feedback_stability
```

## 7.2 Preservation applicability matrix

符号：

```text
A = applicable
- = not required in v1
```

| Transformation | F01 | F02 | F03 | F04 | F05 | F06 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| entity alpha-renaming | A | A | A | A | A | A |
| observation reorder | A | - | A | A | A | - |
| irrelevant entity augmentation | A | A | A | A | A | A |
| unit conversion | A | A | A | A | A | A |
| coordinate affine transform | A | A | - | - | A | A |
| equivalent aggregation split/merge | - | - | A | A | A | - |
| nontrivial scale map | A | A | A | A | A | A |
| sign-convention reparameterization | - | A | A | - | - | A |

## 7.3 Legal preservation pair counts

| Transformation | Rule | Legal pairs |
|---|---|---:|
| alpha-renaming | 6 / family / scale | 72 |
| observation reorder | 6 / applicable family / scale | 48 |
| irrelevant augmentation | 6 / family / scale | 72 |
| unit conversion | 8 / family / scale | 96 |
| coordinate affine | 8 / applicable family / scale | 64 |
| aggregation split/merge | 8 / applicable family / scale | 48 |
| nontrivial scale map | 10 / family | 60 |
| sign convention | 6 / applicable family / scale | 36 |
| **合法总数** |  | **496** |

额外 invalid transformation controls：

```text
2 per applicable family × transformation
```

其中 scale-specific transform 的两个 control 分别来自两个 scale。

invalid 总数：

```text
76
```

总 preservation / sensitivity pairs：

```text
496 + 76 = 572
```

这些是 derived pairs，不计入 720 个独立 latent cases。

## 7.4 Embedding baseline

```json
{
  "model_id": "sentence-transformers/all-mpnet-base-v2",
  "revision_policy": "exact_40_hex_commit_required",
  "pooling": "model_default_mean_pooling",
  "normalization": "l2",
  "similarity": "cosine",
  "input": "canonical_public_evidence_text",
  "prototype": "development-family-centroid",
  "no_verifier_access": true
}
```

运行前必须将实际 Hugging Face 40-hex revision 写入 manifest。浮动 `main` 不允许进入正式 run。

## 7.5 LLM semantic-only baseline

```json
{
  "model_id": "Qwen/Qwen2.5-7B-Instruct",
  "revision_policy": "exact_40_hex_commit_required",
  "do_sample": false,
  "temperature": 0.0,
  "top_p": 1.0,
  "max_new_tokens": 128,
  "seed": 0,
  "tool_access": false,
  "verifier_access": false
}
```

冻结 prompt：

```text
You are a semantic-only structural-label baseline.
You receive one canonical evidence description.
Do not execute equations, call a verifier, enumerate candidate programs,
or use hidden metadata.
Return exactly one JSON object:
{
  "family": "<one allowed family id or ABSTAIN>",
  "binding": "<one candidate binding id or ABSTAIN>",
  "scale": ["<zero, one, or multiple allowed scale ids>"],
  "decision": "ANSWER | ANSWER_SET | ABSTAIN"
}
Use only the visible wording and surface associations in the evidence.
```

## 7.6 Flat typed baseline

每个 output head 使用：

```json
{
  "estimator": "sklearn.ensemble.HistGradientBoostingClassifier",
  "learning_rate": 0.05,
  "max_iter": 200,
  "max_leaf_nodes": 15,
  "max_depth": 3,
  "min_samples_leaf": 20,
  "l2_regularization": 1.0,
  "early_stopping": false,
  "random_state": 2611585425
}
```

`random_state` 的冻结 derivation ID 为
`sha256_domain_separated_uint64_be_first32_v1`：对 domain-separated preimage 中的
domain bytes `hegel-machine/phase2b/bootstrap-and-flat-baseline/uint32/v1\0` 与
unsigned-64-bit big-endian master seed `411876909552964556` 取 SHA-256，并把 digest
前 4 bytes 按 unsigned big-endian 解释为 `2611585425`。bootstrap 自身的 `seed` 保持
原 64-bit master value，不把两种字段混为同一个 API 参数。

独立 heads：

```text
family
binding
scale_set_class
answer_vs_abstain
```

joint output 由冻结组合规则产生，不允许在 holdout 后调整。

## 7.7 Bootstrap

```json
{
  "method": "paired_cluster_bootstrap",
  "replicates": 10000,
  "seed": 411876909552964556,
  "resampling_unit": "latent_base_case",
  "cluster_members": [
    "original_case",
    "all_preservation_variants",
    "all_baseline_predictions"
  ],
  "interval": "one_sided_95_percent_percentile"
}
```

## 7.8 Semantic-conflict subset

不从 720 中扣除。

建立额外 sealed challenge：

```text
240 cases
```

每个 family × scale：

```text
10 low-semantic-overlap structural positives
10 high-semantic-overlap structural negatives
```

总数：

```text
12 cells × 20 = 240
```

它与主 720 同一次冻结、同一次 reveal，但：

- 不进入主 accuracy 分母；
- 单独用于 structural-vs-semantic gate；
- 不用于调 threshold。

## 7.9 Shared-footprint taxonomy

每个 case 标注：

```text
P2_PAIR
P3_CHAIN
P4_STAR
PSET_AGGREGATE
```

定义：

| Footprint class | 条件 |
|---|---|
| `P2_PAIR` | 2 个共享 nonconstant measurements |
| `P3_CHAIN` | 3 个 measurement，形成有向或有序链 |
| `P4_STAR` | 4 个以上 measurement，共享中心对象/量 |
| `PSET_AGGREGATE` | 5–8 个 set measurements，经聚合使用 |

每个 `family × scale` 至少出现 3 种 footprint class。

## 7.10 “单一 measurement 不超过 50%”统计量

使用冻结 flat typed baseline 做 grouped permutation importance。

对 measurement group \(j\)：

\[
\Delta_j
=
A_{\text{full}}
-
\frac1{1000}\sum_{b=1}^{1000}A_{\text{perm}(j,b)}.
\]

定义：

\[
s_j=
\frac{\max(\Delta_j,0)}
{\sum_k\max(\Delta_k,0)}.
\]

gate：

```text
max_j s_j <= 0.50
```

并同时要求：

```text
best_single_measurement_balanced_accuracy <= 0.50
```

permutation 必须在 `case_type × scale` strata 内进行。

## 7.11 Answer reveal 前允许的 rerun

允许 rerun 的原因仅限：

```text
CONTAINER_START_FAILURE
HOST_OOM_BEFORE_FIRST_PREDICTION_BYTE
HOST_TERMINATION_BEFORE_FIRST_PREDICTION_BYTE
INPUT_TRANSFER_CHECKSUM_MISMATCH
OUTPUT_UPLOAD_FAILURE_AFTER_OUTPUT_HASH_COMMITTED
```

最后一种不重新执行 recognizer，只重新上传已冻结 output。

不允许 rerun：

```text
MODEL_EXCEPTION
PARSER_FAILURE
MISSING_CASE_OUTPUT
VERIFIER_EXCEPTION
LOW_COVERAGE
NONDETERMINISTIC_OUTPUT
TIMEOUT_AFTER_ANY_PREDICTION_BYTE
```

最大重新执行次数：

```text
2
```

所有 attempt 均永久记录。

如果任何 valid prediction byte 已产生，则该 attempt 是正式 attempt，不得因结果不好重跑。

## 7.12 Validation 两轮都失败

流程：

1. `validation_v1` 最多两轮；
2. 两轮均失败后，写入 immutable failure report；
3. 原 validation 全部降级为公开 development evidence；
4. 修改任何算法/阈值必须提升：
   ```text
   protocol_version
   selector_version
   validation_version
   ```
5. custodian 使用全新独立 seed 生成 `validation_v2`；
6. `validation_v2` 仍最多两轮；
7. 若 v2 也失败，Phase-2B 宣布 NO-GO，必须重新设计协议，不允许继续生成 v3 直到外部审查。

sealed holdout 只能在某一 validation version 通过后生成。

---

# 8. Public wire covert-answer-channel 审计

## 8.1 ID 生成

使用三个相互独立的 secret keys：

```text
K_shuffle
K_id
K_padding
```

它们与 latent generator seed、answer seed 分离。

流程：

1. 生成全部 latent cases；
2. 使用 `K_shuffle` 做 Fisher–Yates global shuffle；
3. shuffle 后才分配 public IDs；
4. ID：
   \[
   \operatorname{HMAC\!-\!SHA256}
   (K_{\text{id}}, run\_id\Vert namespace\Vert counter)
   \]
   截取 128 bits，并设置 UUIDv4 version/variant bits；
5. provenance public commitment 只能哈希 public payload，不可哈希 private answer object。

collision policy：

```text
retry_with_counter_suffix
max_retries = 10
collision_count must be reported
collision_count > 0 triggers audit warning
failure after 10 retries aborts wire generation
```

## 8.2 字段分类

每个 allowlisted field 必须标记：

```text
EVIDENCE_BEARING
CHANNEL_ONLY
```

### Evidence-bearing

例如：

```text
measurement value
unit dimension
temporal support
spatial support
orientation
task target
```

允许与答案相关，但受 shared-footprint、single-field dominance 和 counterfactual tests 约束。

### Channel-only

例如：

```text
UUID bytes
filename
record order
provenance commitment
serialization length
padding
unused transform identifiers
observation slot number
```

不得提供额外预测信息。

## 8.3 Covert-channel tests

对每个 channel-only field group 和每个 target：

```text
family
binding
scale
answerable_vs_abstain
joint_decision_class
```

运行：

1. byte / prefix / suffix feature test；
2. normalized mutual information；
3. one-field predictive auditor；
4. 10,000 label permutations。

### Permutation strata

| Target | Permutation strata |
|---|---|
| family | `case_type × scale` |
| scale | `case_type × family` |
| binding | `case_type × family × scale` |
| answerable | `family × scale` |
| joint class | `case_type` |

multiple testing：

```text
Holm-Bonferroni
family-wise alpha = 0.01
```

channel-only gate：

```text
adjusted_p >= 0.01
normalized_mutual_information <= 0.02
cross_validated_balanced_accuracy_advantage <= 0.05
```

三个条件均须通过。

## 8.4 Unique IDs 的特殊测试

因 unique categorical ID 的 naive empirical MI 无意义，不能直接把 UUID 当类别计算 MI。

冻结 feature family：

```text
128 individual ID bits
first 8/16/32 bits
last 8/16/32 bits
Hamming weight
integer mod 3,5,7,11,13
hex character histogram
```

这些 feature 进入上述 permutation auditor。

## 8.5 Consistent-renaming invariance

覆盖 namespaces：

```text
bundle_id
observation_id
entity_id
role_candidate_id
quantity_id
context_id
task_id
scale_id
aggregate_map_id
transform_id
```

每次 permutation 必须同步修改：

- public evidence；
- catalog references；
- candidate outputs 的 ID mapping；
- evaluator decode map。

最少：

```text
32 independent global bijective renamings
```

要求映射回 canonical IDs 后：

```text
decision identical
family identical
binding identical
scale set identical
candidate residual vector bit-identical
```

任何一例失败即 anti-leak gate 失败。

## 8.6 Order、length、missingness 与 unused-field channels

### 序列顺序

每个正式 run 额外执行：

```text
32 global case-order permutations
16 within-case observation-order permutations
```

decision 必须保持。

### Wire length

trusted wire builder 将每个 case 编码为 RFC-8785 canonical JSON，然后包进：

```text
fixed 65536-byte envelope
```

padding 使用 `K_padding`，与 answer seed 独立。

recognizer 只通过 stdin 获取 envelope，不获得文件名、目录位置或 case sequence index。

### Missingness

所有 schema fields 必须存在；缺失值显式写：

```json
{"available": false, "value": null}
```

missingness pattern 作为 channel-only derived feature 进入 permutation audit。

### Unused transform

public case 不发送 per-case unused transform list。<br>
只发送全局 frozen catalog root：

```text
transform_catalog_root
```

若必须发送 eligibility mask，该 mask 必须：

- 在 family / scale / answer class 之间配额匹配；
- 通过 channel audit；
- 有实际 semantics，不得作为 padding sentinel。

## 8.7 `standard_error` 转 closed interval

### 正式允许条件

`standard_error` 只有在同时提供：

```text
sample_count n >= 3
sampling_unit_id
estimator == sample_mean
independent_replicates == true
finite_variance_assumption == true
distribution_model == student_t_iid
```

时才可转换。

对一个 candidate comparison 内共有 \(M\) 个同时使用的 intervals，使用 family-wise 99% interval：

\[
\bar x
\pm
t_{1-\frac{0.01}{2M},\,n-1}\,SE.
\]

interval endpoints 向外取整到 frozen RationalValue grid；不能向内取整。

若任何语义字段缺失：

```text
STANDARD_ERROR_UNSUPPORTED
```

formal selector 只能接受：

```text
absolute_bound
```

因此，在上述语义完全实现并测试前：

> **只允许 `absolute_bound` 进入正式 selector。**

---

# 9. Closure / MDL receipt 的可信证书

## 9.1 决定

采用：

> **双独立实现完整 replay + 3/3 detached custodian signatures**

不接受：

- 单个实现自报；
- caller-supplied closure cardinality；
- caller-supplied `Fraction`；
- 只有一个签名 custodian；
- 仅凭 proof-looking JSON。

实现 A：

```text
Python reference implementation
```

实现 B：

```text
Rust independent implementation
```

两者不得共享 canonicalizer、enumerator 或 evaluator source。

## 9.2 Canonical serialization 与 hash

所有 record 使用 canonical CBOR。

hash：

```text
SHA-256
```

Merkle 采用 RFC-6962 风格：

```text
leaf_hash = SHA256(0x00 || canonical_cbor(record))
node_hash = SHA256(0x01 || left_hash || right_hash)
```

非 2 的幂的树按 RFC-6962 的 largest-power-of-two split 递归构造，不复制最后 leaf。

## 9.3 Program record schema

```json
{
  "schema_version": "closure-program-record-v1",
  "program_index": 0,
  "canonical_ast": {},
  "canonical_ast_hash": "sha256:...",
  "output_sort": "Bit",
  "depth": 0,
  "node_count": 0,
  "distinct_entity_slot_count": 0,
  "program_code_length_q32": 0,
  "undefined_row_bitmap_hash": "sha256:...",
  "output_vector_hash": "sha256:...",
  "extensional_class_hash": "sha256:...",
  "first_extensional_representative_index": 0,
  "dsl_spec_root": "sha256:...",
  "bounded_universe_root": "sha256:..."
}
```

`output_vector` 存在独立 bit-packed / canonical rational blob 中，由 hash 绑定。

## 9.4 Canonical ordering

program records 依次按：

```text
depth
node_count
output_sort
root_operator_id
canonical_ast_cbor_bytes
```

排序。

`program_index` 必须等于排序后 0-based position。

## 9.5 Chunking

```text
records_per_chunk = 4096
```

最后一块可少于 4096。

chunk manifest：

```json
{
  "chunk_index": 0,
  "first_program_index": 0,
  "last_program_index": 4095,
  "record_count": 4096,
  "record_merkle_root": "sha256:...",
  "compressed_blob_sha256": "sha256:...",
  "uncompressed_byte_length": 0
}
```

archive root 是按 `chunk_index` 排序的 chunk manifest Merkle root。

## 9.6 Bounded universe root

每个 universe row：

```json
{
  "universe_index": 0,
  "input_signature_id": "...",
  "canonical_input": {},
  "canonical_input_hash": "sha256:..."
}
```

按 `universe_index` 排序后构造：

```text
bounded_universe_root
```

## 9.7 Target truth-table root

每个 row：

```json
{
  "universe_index": 0,
  "canonical_input_hash": "sha256:...",
  "target_output": 0
}
```

root：

```text
target_truth_table_root
```

这样 target root 同时绑定：

- bounded universe；
- row order；
- input；
- output。

## 9.8 Exhaustion receipt

每个实现独立输出：

```json
{
  "implementation_id": "...",
  "dsl_spec_root": "...",
  "bucket_counts": [
    {
      "output_sort": "...",
      "depth": 0,
      "node_count": 0,
      "raw_operator_applications": 0,
      "accepted_canonical_programs": 0,
      "canonical_duplicates": 0,
      "type_rejections": 0,
      "limit_rejections": 0
    }
  ],
  "raw_operator_application_count": 0,
  "canonical_program_count": 0,
  "frontier_exhausted": true,
  "program_archive_root": "...",
  "output_archive_root": "...",
  "exhaustion_receipt_root": "..."
}
```

两个实现的 traversal root 不要求相同，因为 traversal 算法可以不同。

但必须相同：

```text
canonical_program_count
program_archive_root
output_archive_root
match_set_count
match_program_hashes
bounded_universe_root
target_truth_table_root
```

## 9.9 Certificate trusted root

三个 offline Ed25519 keys：

```text
K_custodian
K_replay_python
K_replay_rust
```

正式 certificate 要求：

```text
3 of 3 signatures
```

签名对象：

```text
SHA256(canonical_cbor(final_certificate_without_signatures))
```

### Key rotation

key epoch manifest：

```json
{
  "key_epoch": 2,
  "previous_key_epoch": 1,
  "new_public_keys": [],
  "effective_at": "...",
  "reason": "...",
  "invalidate_certificates_before": null
}
```

rotation 需要旧 epoch 2-of-3 signatures。

### Revocation

revocation manifest 也需 2-of-3 signatures。

默认：

- revocation 不使旧证书自动失效；
- 只有显式 `invalidate_certificates_before` / `after` 才改变历史 validity；
- verifier 必须读取最新 key-status manifest。

签名证明的是 replay provenance 和 artifact integrity，不是“关系在真实世界中为真”。

## 9.10 `OUTSIDE_FROZEN_CLOSURE` 的机器条件

全部满足才可签发：

```text
dsl_spec_status == FROZEN
target_commitment_precedes_synthesis == true

python_replay_status == COMPLETE
rust_replay_status == COMPLETE

python_canonical_count == rust_canonical_count
canonical_count <= 50000

python_program_archive_root == rust_program_archive_root
python_output_archive_root == rust_output_archive_root

python_match_set_count == 0
rust_match_set_count == 0

undefined_target_row_count == 0
bounded_universe_root_match == true
target_truth_table_root_match == true

raw_expansion_limit_hit == false
frontier_exhausted_python == true
frontier_exhausted_rust == true

covert_channel_audit_pass == true
signature_count == 3
```

输出 claim 必须包含：

```text
OUTSIDE_FROZEN_CLOSURE(
  dsl_version,
  bounded_universe_root,
  target_truth_table_root,
  equivalence = exact_extensional
)
```

不得简写成：

```text
OUTSIDE_LANGUAGE
```

## 9.11 MDL scorer replay

MDL scorer只接收：

- frozen AST；
- new-symbol definition；
- code table；
- scoring partition；
- prediction vectors；
- target labels。

它忽略 caller-supplied：

```text
length
Fraction
delta_L
threshold_pass
```

独立重算：

```text
L_old_program
L_train_given_old
L_new_symbol_definition
L_new_call_program
L_train_given_new
delta_L
required_delta_L
```

## 9.12 MDL certificate bindings

```text
mdl_code_table_root
dsl_spec_root
identifier_registry_root
discovery_partition_root
validation_partition_root
sealed_partition_root
target_truth_table_root
old_program_ast_hash
new_symbol_definition_hash
new_call_program_ast_hash
old_prediction_vector_root
new_prediction_vector_root
validation_prediction_root
sealed_prediction_root
fixed_point_precision_id
mdl_algorithm_id
repository_commit_sha
container_image_digest
```

MDL certificate 也使用 Python / Rust 双 replay 和 3/3 signatures。

---

# 10. 当前最优先施工顺序

## Priority 1：冻结与实现 DSL closure

```text
dsl_spec
operator_semantics
canonicalizer
bounded universe
Python enumerator
Rust enumerator
```

先运行 public preflight，确认 canonical closure 不超过 50,000。

## Priority 2：冻结 target registry

完成：

```text
target truth-table commitments
priority order
fallback rule
split commitments
```

然后才允许 synthesis agent 接触 discovery train。

## Priority 3：covert channel wire builder

完成：

```text
independent ID/shuffle/padding keys
fixed envelope
consistent renaming tests
metadata predictive audits
standard_error disabled by default
```

## Priority 4：closure / MDL certificate

完成：

```text
program records
chunk roots
archive roots
independent replay
3-key signatures
key status manifests
```

## Priority 5：Phase-2B statistical completion

完成：

```text
720 + 240 challenge
572 preservation/sensitivity pairs
baselines
bootstrap
validation protocol
rerun contract
```

## Priority 6：只在以上冻结后生成 sealed artifacts

不得先生成 holdout 再补 DSL、MDL 或 covert-channel test。

---

# 11. Go / No-Go 总表

| 项目 | 当前决定 |
|---|---|
| Phase-2B infrastructure construction | GO |
| Phase-2B formal holdout generation | NO-GO，等待 wire / quotas / baselines / certificate freeze |
| Phase-3 DSL implementation | GO |
| parity target synthesis run | NO-GO，等待完整 closure preflight 和 target commitments |
| hidden sink null-control implementation | GO |
| `OUTSIDE_FROZEN_CLOSURE` certificate | NO-GO，等待双实现 replay |
| ACTIVE promotion | NO-GO |
| shadow candidate records | GO |

---

# 12. 最终不变原则

\[
\boxed{
\text{旧语言的边界必须在看到 hidden result 之前冻结。}
}
\]

\[
\boxed{
\text{新关系必须在完整旧 closure 上无等价表达，而不是仅在搜索中没找到。}
}
\]

\[
\boxed{
\text{metadata 不含显式答案，不等于不存在 covert answer channel。}
}
\]

\[
\boxed{
\text{证书必须由独立执行重算产生，不能由被评价系统自报。}
}
\]

\[
\boxed{
\text{Phase-3 首个结果只证明 bounded mechanism，不证明开放世界理论发明。}
\]
