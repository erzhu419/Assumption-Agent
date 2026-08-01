# Phase-3 Shrink Step 1 发布前待 GPT / 用户定案的问题

**文档状态**：`RESOLVED_AND_SUPERSEDED`<br>
**用途**：只收集发布 shrink step 1 前仍会改变 wire identity、commitment 或 M3 运行合同的实质决策。<br>
**非规范声明**：本文中的所有“建议默认”均为 **NON-NORMATIVE DEFAULT**；只有 GPT 给出明确方案并经用户批准、随后写入新的 normative amendment 后才生效。<br>
**依据规范**：

- `Hegel_Machine_Strict_Canonical_AST_CBOR_Certificate_Bridge_Freeze_v1.0.2.md`
- `Hegel_Machine_Phase3_Freeze_Readiness_Resolution.md`

> **RESOLUTION:** 本文问题已由
> `Hegel_Machine_Phase3_Shrink_Step1_Freeze_Decisions.md` 完整回答并经用户批准。
> 下文保留为决策来源审计记录，不再是当前施工 blocker。

---

## 0. 已成立的证据边界

以下不再作为开放问题：

```json
{
  "old_dsl_version": "hegel-old-dsl-v1.0.0",
  "freeze_version": "hegel-freeze-p2b-p3-v1.0.2",
  "canonical_cbor_profile_id": "hegel-cbor-det-v1",
  "canonical_ast_schema_id": "hegel-canonical-ast-v1",
  "canonical_program_budget": 50000,
  "source_candidate_count": 64680,
  "python_accepted_unique_count": 64680,
  "rust_accepted_unique_count": 64680,
  "rejected_count": 0,
  "rewrite_collapsed_count": 0,
  "dual_replay_equal": true,
  "executed_closure_status": "DSL_TOO_LARGE"
}
```

当前可审计 evidence：

| Evidence | Identity / commitment |
|---|---|
| dual strict gate | `phase3_dual_strict_gate_06eae23f68536e3f7e80badb46a5b15e0665072f65477608a3f688e54adefad6` |
| dual capacity replay | `phase3_dual_strict_capacity_replay_f75214e75f5fc3812d7375463ba72c347c9c08bc7bae3b68c87a63b484c4e414` |
| shared golden vectors | `sha256:4eed028c6cf4cbf140b8b86a8a8c264f70f67c9ceb206ca3404b196dd44be46e` |
| Python strict source root | `sha256:bb3d9b3ee9b270165f66f0e0d8fcc3c364226b38290ea2bf3b09ebad34fe5c9a` |
| Python capacity execution source root | `sha256:eb8a0b6f6425084c964ebb200ec6eeeb995f0ac4a8909e5d551ad0eb88c0d525` |
| Rust strict source root | `sha256:98fec63ea16d4e5ded2fc09ad8ed57b8cc2f599234c59fbe86868d445401e46f` |
| Rust release binary | `sha256:875ad23b688e1592d357966c3a5895c8c5909b5e088e6003d40ecb1de7b71a31` |
| accepted-set diagnostic commitment | `sha256:c1a02a66a8d6d8f75204cb3daf03ab0b01c2b3b8e486d0ab3d481ee3be43c930` |
| canonical ordinal 50,001 AST hash | `sha256:7c7f786c2cc57d31506b3c61d162d175c7f69a2878a089c72c9d053694cba948` |
| canonical ordinal 50,001 CBOR | `820182048284020383000002830000048402038600030000008083000000` |

上述 set commitment 是有长度 framing 的 diagnostic commitment，不是 RFC6962 archive root。当前仍然：

```json
{
  "complete_closure_enumerated": false,
  "formal_archive_roots_generated": false,
  "formal_bounded_universe_roots": null,
  "formal_target_truth_table_roots": null,
  "target_synthesis_allowed": false,
  "outside_certificate_issued": false,
  "active_promotion_allowed": false
}
```

v1.0.2 已预注册的 shrink step 1 只有：

```text
remove mean_v1, min_v1, max_v1
```

不得借本次发版同时执行 shrink step 2，或修改 budget、typing、rewrite、node/depth、scope、equivalence 与 target role。

---

## 决策 1：新 DSL 与 freeze 的精确版本 ID

需要 GPT 明确批准两个不可再含糊的 machine IDs：

```text
new_dsl_version = ?
new_freeze_version = ?
```

### NON-NORMATIVE DEFAULT 1

建议批准：

```text
new_dsl_version = hegel-old-dsl-v1.1.0
new_freeze_version = hegel-freeze-p2b-p3-v1.1.0
```

理由：这是预注册 shrink order 内的第一个受限 profile，保留 v1 strict AST/CBOR 语义；它不是对 coercion、typing 或 operator semantics 的扩张。`v1.0.3` 会把实质 surface deletion 错写成 patch，而 `v2.0.0` 可保留给未来改变 typing/coercion 或 AST schema 的语言。

请 GPT 回答：

1. 是否批准以上两个精确字符串；若不批准，必须给出替代的两个完整字符串；
2. 是否同时冻结：
   ```text
   canonical_cbor_profile_id = hegel-cbor-det-v1
   canonical_ast_schema_id = hegel-canonical-ast-v1
   ```
   不升版；
3. 新 amendment 是否命名为：
   ```text
   hegel-freeze-p2b-p3-v1.1.0-shrink-step1
   ```
   作为 human document ID，而 machine freeze ID 仍为上面的 `hegel-freeze-p2b-p3-v1.1.0`。

---

## 决策 2：surviving AggregateMap numeric IDs 是 sparse-preserving 还是重新编号

旧 registry：

| ID | Map | Step-1 disposition |
|---:|---|---|
| 0 | `sum_v1` | survive |
| 1 | `count_nonzero_v1` | survive |
| 2 | `mean_v1` | remove |
| 3 | `min_v1` | remove |
| 4 | `max_v1` | remove |
| 5 | `signed_balance_v1` | survive |

必须在以下两种 wire 中二选一：

### A. sparse-preserving

```text
sum_v1            = 0
count_nonzero_v1  = 1
signed_balance_v1 = 5
2,3,4             = tombstoned
```

### B. dense reindex

```text
sum_v1            = 0
count_nonzero_v1  = 1
signed_balance_v1 = 2
```

### NON-NORMATIVE DEFAULT 2

建议选择 **A / sparse-preserving**，并在 registry 中分别记录：

```json
{
  "registry_width": 6,
  "active_map_count": 3,
  "active_ids": [0, 1, 5],
  "tombstoned_ids": [2, 3, 4],
  "id_reuse_allowed": false
}
```

这样 surviving programs 的 aggregate leaf CBOR 不因 shrink 被无意义改写，也避免 `signed_balance_v1` 的历史 AST hash 全部漂移。

请 GPT 明确回答：

1. 选择 A 还是 B；
2. 若选择 A，`registry_width` 与 `active_map_count` 是否必须作为两个不同字段；
3. tombstoned IDs 是否永久禁止在后续 v1.x 中复用。

---

## 决策 3：removed maps 在 source 与 formal decoder 中是 unknown，还是 reserved-and-rejected

仅仅从 active catalog 删除名称还不够。必须冻结以下输入的 exact disposition：

```text
source name mean_v1 / min_v1 / max_v1
formal aggregate_map_id 2 / 3 / 4
future unknown aggregate_map_id >= 6
```

### NON-NORMATIVE DEFAULT 3

建议：

```text
mean_v1 / min_v1 / max_v1 source names
  -> REJECT_REMOVED_AGGREGATE_MAP

formal map IDs 2 / 3 / 4
  -> REJECT_REMOVED_AGGREGATE_MAP

IDs >= 6
  -> REJECT_REGISTRY_INDEX_OUT_OF_RANGE
```

即：removed IDs 是有审计意义的 tombstones，不再 type-check、canonicalize、计数或执行，但与从未定义过的 unknown ID 区分。不得把 removed map 自动迁移成另一个 map，也不得只在 evaluator 阶段返回 bottom；它必须在 strict acceptance 阶段被拒绝。

请 GPT 明确批准：

1. exact rejection code `REJECT_REMOVED_AGGREGATE_MAP`；
2. source parser 与 formal AST decoder 使用同一 code；
3. legacy diagnostic migration 是否只允许读取旧 artifact，绝不把旧 removed-map program 注入新 closure。

---

## 决策 4：canonical AST / hash 的跨版本兼容语义

若采用 sparse-preserving：

- surviving program 的 canonical AST CBOR bytes 可以保持不变；
- `ContentHash("HEGEL/AST/V1", canonical_ast_cbor)` 可以保持不变；
- removed-map AST 在新 DSL 中不再是 admitted program；
- program archive、output archive 和 receipt 必然改变，因为它们必须绑定新的 DSL / operator-semantics roots 与新的 enumeration。

危险点是：相同 `canonical_ast_hash` 不能单独证明两个 DSL version 中的 program semantics 相同。

### NON-NORMATIVE DEFAULT 4

建议冻结：

```text
syntactic_identity = canonical_ast_cbor bytes
cross_dsl_semantic_identity =
  canonical_ast_cbor bytes
  + dsl_spec_root
  + operator_semantics_root
  + identifier_registry_root
```

并采用：

```json
{
  "ast_schema_id": "hegel-canonical-ast-v1",
  "ast_hash_domain": "HEGEL/AST/V1",
  "surviving_ast_hash_stability_required": true,
  "cross_version_archive_root_reuse_allowed": false,
  "cross_version_receipt_reuse_allowed": false,
  "cross_version_certificate_reuse_allowed": false
}
```

新 golden vectors 必须至少证明：

- IDs `0,1,5` 的 surviving AST bytes/hash 与 v1.0.0 相同；
- IDs `2,3,4` 以 exact tombstone code 拒绝；
- ID `5` 未被 reindex；
- removed-map old bytes 仍能被 generic CBOR parser读取，但不能通过新 DSL strict AST acceptance。

请 GPT 明确回答：是否批准这一“syntax hash stable、semantic binding versioned”的分层，而不是给 AST hash 换 domain 或把旧 hash 当新 DSL 证据。

---

## 决策 5：必须重新生成哪些 target / validation / diagnostic commitments

当前 relation/control payload identities 是：

### Odd-reduction outside target

```text
target_id = TARGET_P3A_GENERIC_ODD_REDUCTION_V1
target_spec_id = target_spec_b491c0a9719fb0279fe02798ede026e440c17a539965514145a7818b15387ac3
bounded_universe_diagnostic_id = bounded_universe_2425ded9cbb0f3d2b6cb7c08583ed9d65ce968e0386f53847ad2731262887fae
target_table_diagnostic_id = target_truth_table_40fd713925d3c987c5af005e0be411619d0b3eaeb2c7fe196a8a6b8ca9d0761e
formal_bounded_universe_root = null
formal_target_truth_table_root = null
```

### Hidden-sink in-language null control

```text
control_id = CONTROL_P3A_OBSERVED_OMITTED_SINK_V1
control_spec_id = sink_control_spec_7fd6f9a6e2b4c6eda0c7e1545ad42cb19666743ede8ed87f40d82c0ef46198a0
bounded_universe_diagnostic_id = bounded_universe_2ac0c35cd5ad001eb95d21390c0282adfc9f066ec27357e12f712084c45cb9ef
target_table_diagnostic_id = target_truth_table_f8e41f3dd569cfa23cee44693127c821a407830e8118b649e72c6ce623b19328
hidden_generator_spec_id = hidden_generator_spec_e498b9803e6b0cf02b8a287a0f8756df4081828d1504db8719bf8b6d22f03259
formal_bounded_universe_root = null
formal_target_truth_table_root = null
```

规范要求 DSL version 改变后重新生成 target commitments 和所有 validation artifacts。这里需要区分“row payload identity”与“绑定新 DSL 的 commitment manifest”。

### NON-NORMATIVE DEFAULT 5

建议按以下 disposition 执行：

| Object | Shrink-1 publication action |
|---|---|
| shrunk DSL surface / active+tombstone registry / operator semantics diagnostic objects | 生成全新 content IDs |
| new preregistration / freeze manifest | 生成全新 ID，并把旧 dual-capacity evidence 作为 parent evidence 引用 |
| odd target binding manifest | 生成全新 ID，绑定新 DSL/freeze、target role、fallback priority、split commitments |
| hidden-sink control binding manifest | 生成全新 ID，绑定新 DSL/freeze 与 `IN_LANGUAGE_NULL` role |
| discovery / validation / sealed split artifacts | 使用新独立 seed/version 全部重新生成并 commitment；旧版本不得用于新 DSL scoring |
| hidden generator / custodian manifest | 生成全新 ID，并证明在 synthesis 前已封存 |
| odd 480-row universe与 truth-table JCS payload | 重新构造、重新 JCS 验证；若 bytes 真正相同，content ID 应自然保持相同，不人为加盐制造新 identity |
| sink 85-row universe与 truth-table JCS payload | 同上；相同 payload 允许相同 content ID，但必须由新的 control binding manifest 重新绑定 |
| shrink-1 capacity replay | Python/Rust 重新运行并生成新 gate/report IDs；不得把 v1.0.0 的 64,680 commitment当作新 DSL count |

建议 shrink-1 的保守 witness 预期组合数单独预注册为：

```text
rational aggregate maps = sum_v1, signed_balance_v1 = 2
rational aggregate leaves = 2 * 4 scopes * 2 quantities = 16
constant atoms = 77
mixed atoms = 7 constants * 16 aggregates * 3 comparison orientations = 336
source AND2 candidates = 77 * 336 = 25,872
```

`25,872` 只能是新 M2 subset replay 的 source count；在双实现 strict replay 前不得称 canonical count，更不得推断完整 closure `COMPLETE`。

在上述 publication / preregistration 阶段，建议所有 formal roots 继续为 `null`：

```text
dsl_spec_root
operator_semantics_root
identifier_registry_root
canonical_ast_schema_root
bounded_universe_root
target_truth_table_root
program/archive roots
diagnostic_formal_bridge_root
```

不得把旧 diagnostic ID、删除 prefix 后的 hex，或旧 v1.0.0 evidence commitment填入这些字段。

请 GPT 明确回答：

1. 是否批准“相同 row payload保留相同 content ID，但生成新的 version-binding manifest”；
2. 新 split 是否必须使用独立 seed，还是只需重新封装旧 deterministic split；
3. odd target 与 sink control 是否都必须重新绑定；
4. 旧 validation artifact 的正式 disposition 是 `HISTORICAL_DEVELOPMENT_ONLY` 还是完全封存不可见；
5. publication 时 formal roots 全部保持 `null` 是否批准。

---

## 决策 6：M3 complete-enumerator / archive replay 的精确进入条件

这里的“进入 M3”是允许创建一个全新的 execution manifest 并将新 DSL run 从 `NOT_RUN` 迁移到 `RUNNING`；它不是预先宣布 `COMPLETE`。

### NON-NORMATIVE DEFAULT 6

建议只有以下 checklist 全部为真时才进入 M3：

```json
{
  "shrink1_normative_amendment_user_approved": true,
  "dsl_version": "hegel-old-dsl-v1.1.0",
  "freeze_version": "hegel-freeze-p2b-p3-v1.1.0",
  "aggregate_id_policy_frozen": true,
  "removed_map_rejection_policy_frozen": true,
  "canonical_ast_hash_compatibility_policy_frozen": true,

  "python_strict_implementation_updated": true,
  "rust_strict_implementation_updated": true,
  "new_shared_golden_vectors_frozen": true,
  "all_valid_vectors_equal": true,
  "all_invalid_vectors_rejected_identically": true,

  "shrink1_capacity_source_count": 25872,
  "shrink1_python_rust_accepted_set_equal": true,
  "shrink1_first_out_of_budget_witness": null,
  "shrink1_capacity_replay_has_no_semantic_or_execution_disagreement": true,

  "target_and_control_binding_manifests_committed": true,
  "discovery_validation_sealed_commitments_committed": true,
  "commitments_precede_any_new_synthesis_or_hidden_outcome_access": true,

  "diagnostic_formal_bridge_python_rust_equal": true,
  "formal_dsl_spec_root_non_null": true,
  "formal_operator_semantics_root_non_null": true,
  "formal_identifier_registry_root_non_null": true,
  "formal_ast_schema_root_non_null": true,
  "formal_bounded_universe_root_non_null": true,
  "formal_target_truth_table_root_non_null": true,

  "python_complete_enumerator_implemented": true,
  "rust_complete_enumerator_implemented": true,
  "canonical_traversal_and_bucket_accounting_frozen": true,
  "program_output_and_chunk_archive_emitters_replay_verified": true,
  "records_per_chunk": 4096,

  "new_execution_manifest_root_non_null": true,
  "new_run_id": true,
  "initial_state": "NOT_RUN",
  "prior_v1_0_0_run_state_reuse": false
}
```

其中 formal roots 在 shrink-1 publication 时保持 `null`，但在正式 M3 archive run 开始前必须通过 frozen diagnostic→CBOR→RFC6962 bridge 变为非空并跨实现一致。理由是 ProgramRecord、ProgramOutputRecord、receipt 与 execution manifest 都需要绑定这些 roots；先跑后补 root 会产生不可接受的 orphan replay。

M3 run 还必须沿用：

```text
canonical program budget = 50,000
raw operator-application cap = 5,000,000
equivalence = exact extensional
records_per_chunk = 4096
state begins at NOT_RUN under a new run_id
```

正式运行只允许以下终态：

### `COMPLETE`

```text
frontier_exhausted = true
all_type_buckets_closed = true
closure_cardinality = canonical_program_count <= 50,000
raw cap not hit
wall-clock abort not hit
canonical_program_archive_root != null
program_output_archive_root != null
chunk_manifest_root != null
match_set_count != null
Python/Rust counts, archives, outputs, buckets and match set equal
```

### 再次 `DSL_TOO_LARGE`

```text
canonical_program_count = 50,000
first_out_of_budget_program_hash != null
frontier_exhausted = false
closure_cardinality = null
match_set_count = null
```

随后只能发布又一个新 DSL version 并进入已预注册 shrink step 2。

### `INCONCLUSIVE_BUDGET / INCONCLUSIVE_SEMANTICS / INCONCLUSIVE_EXECUTION`

任何 raw cap、未定义语义、跨实现不一致或执行失败都不得携带正式 match verdict。

建议明确排除以下 M3 entry prerequisites：

- final 3/3 certificate signatures；
- `OUTSIDE_FROZEN_CLOSURE` certificate；
- MDL replay；
- invention synthesis；
- ACTIVE governance。

这些属于 M4/M5 或更晚阶段；但 M3 execution manifest、bridge 与 replay outputs 必须可供随后签名。

请 GPT 明确回答：

1. 是否批准以上 M3 entry checklist；
2. formal roots 是否必须在 M3 run 前非空，还是允许 target-independent enumeration 先跑、随后整次重跑；
3. shrink-1 subset 双 replay `accepted_unique_count <= 50,000` 是否只是进入完整 enumeration 的门槛，而绝不等于 `COMPLETE`；
4. final signatures/key chain 是否确认留在 M4 certificate gate，而不阻塞 M3 execution。

---

## 请 GPT 按此格式给出最终方案

请不要只给原则性建议。请逐项输出可直接转成 machine-readable amendment 的决定：

```yaml
decision_1_versions:
  new_dsl_version: "..."
  new_freeze_version: "..."
  ast_schema_id: "..."
  cbor_profile_id: "..."

decision_2_aggregate_ids:
  policy: "SPARSE_PRESERVING | DENSE_REINDEX"
  active_ids: [...]
  tombstoned_ids: [...]
  registry_width: ...
  active_map_count: ...

decision_3_removed_maps:
  source_disposition: "..."
  formal_ast_disposition: "..."
  exact_error_code: "..."
  future_id_reuse_allowed: false

decision_4_hash_compatibility:
  surviving_ast_bytes_stable: true|false
  surviving_ast_hash_stable: true|false
  required_cross_dsl_binding_roots: [...]

decision_5_commitments:
  regenerate: [...]
  recompute_but_content_stable_if_identical: [...]
  fresh_split_seed_required: true|false
  old_validation_disposition: "..."
  formal_roots_at_publication: null

decision_6_m3_entry:
  required_gates: [...]
  formal_roots_required_before_run: true|false
  allowed_terminal_states: [...]
  certificate_signatures_are_m3_entry_gate: true|false
```

该历史限制已经由正式决策稿解除到 shrink-step-1 publication / dual subset replay
为止。它没有解除 formal roots、M3 run、target synthesis、certificate 或 ACTIVE 的门禁；
当前剩余问题转入 `questions_for_gpt_phase3_formal_bridge_and_seed_continuity.md`。
