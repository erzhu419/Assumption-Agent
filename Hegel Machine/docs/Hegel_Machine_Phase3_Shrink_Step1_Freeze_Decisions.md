# Hegel Machine Phase-3 Shrink Step 1 发布冻结决策

**文档类型**：Normative shrink-transition amendment<br>
**建议 human document ID**：`hegel-freeze-p2b-p3-v1.1.0-shrink-step1`<br>
**新 machine freeze ID**：`hegel-freeze-p2b-p3-v1.1.0`<br>
**新 old-language ID**：`hegel-old-dsl-v1.1.0`<br>
**父版本**：

```text
parent_freeze_version = hegel-freeze-p2b-p3-v1.0.2
parent_dsl_version    = hegel-old-dsl-v1.0.0
```

**触发证据**：commit `fb3a3ee4865a140c558821017ddd3e9a6a99de48` 下，Python/Rust strict canonicalizer 对 64,680 个 source candidates 均接受为 64,680 个不同 strict canonical AST，并确定 canonical ordinal 50,001 witness。由此，父 DSL 在冻结的 50,000 syntactic-program budget 下正式进入有边界的：

```text
DSL_TOO_LARGE
```

该状态只授权执行预注册 shrink step 1：

```text
remove mean_v1, min_v1, max_v1
```

它不授权：

- `COMPLETE`；
- extensional target verdict；
- odd-target synthesis；
- hidden-sink formal verdict；
- outside/MDL certificate；
- Phase-2B formal exit；
- ACTIVE promotion。

## 施工执行覆盖层（2026-08-01）

本规范已由用户批准并落实到 child diagnostic publication：

```json
{
  "dsl_version": "hegel-old-dsl-v1.1.0",
  "freeze_version": "hegel-freeze-p2b-p3-v1.1.0",
  "python_rust_child_vectors_equal": true,
  "shrink1_source_count": 25872,
  "python_accepted_unique_count": 25872,
  "rust_accepted_unique_count": 25872,
  "accepted_set_commitment": "sha256:653fcb9428684cfed11c3f2345ac95ed98ded6e31564c9eeabf97c57ee71a7e9",
  "first_out_of_budget_witness": null,
  "complete_closure_enumerated": false,
  "child_execution_state": "NOT_RUN",
  "formal_roots": null,
  "m3_entry_allowed": false
}
```

实现证据位于：

- `artifacts/phase3_shrink1_dual_strict_gate_v1.json`；
- `artifacts/phase3_shrink1_dual_capacity_replay_v1.json`；
- `artifacts/phase3_shrink1_publication_v1.json`；
- `artifacts/phase3_dsl_shrink_transition_v1.json`。

仓库中没有旧 split seed commitment、parent binding manifest、custodian continuity
attestation 或 hidden-access ledger。实现因此没有生成替代 seed，也没有回填 formal root；
相关 M3 gates 继续 fail-closed。该缺口不影响 child strict/subset qualification，但阻止
`NOT_RUN -> RUNNING`。

---

# 0. 总体判断

当前主线没有跑偏。

从父 DSL 的 strict bounded overflow 进入预注册的 shrink step，是：

\[
\boxed{
\text{冻结旧语言}
\rightarrow
\text{证明预算溢出}
\rightarrow
\text{按预注册顺序收缩语言}
\rightarrow
\text{新版本从 NOT\_RUN 重启}
}
\]

而不是为了让 target 看起来 language-outside，事后修改 grammar。

本轮必须只改变 aggregate catalog 的 admission surface：

```text
remove mean_v1
remove min_v1
remove max_v1
```

以下全部继承且不得修改：

- canonical CBOR profile；
- canonical AST schema；
- explicit typing / no implicit Bit coercion；
- strict bottom semantics；
- rewrite rules；
- node/depth/clause/slot/parameter limits；
- scope catalog；
- equality / exact-extensional contract；
- 50,000 canonical-program budget；
- 5,000,000 raw-application cap；
- target/control semantics；
- target fallback order；
- MDL code table；
- Phase-2B contract。

---

# 1. 最终 machine-readable 决定

```yaml
decision_1_versions:
  parent_dsl_version: "hegel-old-dsl-v1.0.0"
  parent_freeze_version: "hegel-freeze-p2b-p3-v1.0.2"
  new_dsl_version: "hegel-old-dsl-v1.1.0"
  new_freeze_version: "hegel-freeze-p2b-p3-v1.1.0"
  human_amendment_id: "hegel-freeze-p2b-p3-v1.1.0-shrink-step1"
  shrink_step_id: "SHRINK_STEP_1_REMOVE_MEAN_MIN_MAX"
  ast_schema_id: "hegel-canonical-ast-v1"
  cbor_profile_id: "hegel-cbor-det-v1"
  mdl_code_table_id: "hegel-mdl-prefix-v1.0.0"
  equivalence_mode: "EXACT_EXTENSIONAL"
  phase2b_contract_change: false

decision_2_aggregate_ids:
  policy: "SPARSE_PRESERVING"
  registry_namespace: "AggregateMapId/v1"
  registry_width: 6
  active_map_count: 3
  tombstone_count: 3
  active_ids: [0, 1, 5]
  active_entries:
    0: "sum_v1"
    1: "count_nonzero_v1"
    5: "signed_balance_v1"
  tombstoned_ids: [2, 3, 4]
  tombstoned_entries:
    2: "mean_v1"
    3: "min_v1"
    4: "max_v1"
  next_allocatable_id: 6
  id_reuse_allowed: false

decision_3_removed_maps:
  source_disposition: "REJECT_REMOVED_AGGREGATE_MAP"
  formal_ast_disposition: "REJECT_REMOVED_AGGREGATE_MAP"
  exact_error_code: "REJECT_REMOVED_AGGREGATE_MAP"
  future_unknown_id_disposition: "REJECT_REGISTRY_INDEX_OUT_OF_RANGE"
  legacy_artifact_parse_allowed: true
  legacy_artifact_execution_under_parent_dsl_only: true
  legacy_artifact_admission_to_new_closure: false
  automatic_map_migration_allowed: false
  future_id_reuse_allowed: false

decision_4_hash_compatibility:
  syntax_identity: "STRICT_CANONICAL_AST_CBOR_BYTES"
  ast_hash_domain: "HEGEL/AST/V1"
  surviving_ast_bytes_stable: true
  surviving_ast_hash_stable: true
  tombstoned_ast_admitted_in_new_dsl: false
  required_cross_dsl_binding_roots:
    - "dsl_spec_root"
    - "operator_semantics_root"
    - "identifier_registry_root"
    - "canonical_ast_schema_root"
    - "canonical_cbor_profile_root"
  program_semantic_identity_domain: "HEGEL/PROGRAM_SEMANTIC_IDENTITY/V1"
  cross_version_archive_root_reuse_allowed: false
  cross_version_receipt_reuse_allowed: false
  cross_version_certificate_reuse_allowed: false

decision_5_commitments:
  regenerate:
    - "shrunk_dsl_surface_object"
    - "aggregate_registry_active_tombstone_object"
    - "operator_admission_semantics_object"
    - "freeze_preregistration_manifest"
    - "odd_target_dsl_binding_manifest"
    - "hidden_sink_dsl_binding_manifest"
    - "split_binding_manifest"
    - "custodian_binding_manifest"
    - "shrink1_dual_strict_gate_artifact"
    - "shrink1_dual_capacity_replay_artifact"
    - "future_m3_execution_manifest"
  recompute_but_content_stable_if_identical:
    - "odd_target_spec_payload"
    - "odd_480_row_universe_jcs_payload"
    - "odd_480_row_truth_table_jcs_payload"
    - "hidden_sink_spec_payload"
    - "sink_85_row_universe_jcs_payload"
    - "sink_85_row_truth_table_jcs_payload"
    - "discovery_split_payload"
    - "validation_split_payload"
    - "sealed_prediction_split_payload"
    - "canonical_ast_schema_payload"
    - "canonical_cbor_profile_payload"
  fresh_split_seed_required: false
  split_seed_reuse_required_if_uncompromised: true
  fresh_seed_required_on_any_prior_hidden_access_or_key_compromise: true
  odd_target_rebinding_required: true
  hidden_sink_rebinding_required: true
  old_validation_disposition: "HISTORICAL_PRECOMMITMENT_ONLY_SEALED"
  old_validation_becomes_development: false
  formal_roots_at_publication: null

decision_6_m3_entry:
  required_gates:
    - "SHRINK1_NORMATIVE_AMENDMENT_APPROVED"
    - "NEW_DSL_AND_FREEZE_IDS_COMMITTED"
    - "SPARSE_AGGREGATE_REGISTRY_FROZEN"
    - "TOMBSTONE_REJECTION_FROZEN"
    - "CROSS_DSL_HASH_POLICY_FROZEN"
    - "PYTHON_STRICT_IMPLEMENTATION_UPDATED"
    - "RUST_STRICT_IMPLEMENTATION_UPDATED"
    - "SHRINK1_GOLDEN_VECTORS_EQUAL"
    - "REMOVED_MAP_VECTORS_REJECTED_IDENTICALLY"
    - "SURVIVING_AST_HASH_STABILITY_VERIFIED"
    - "SHRINK1_SOURCE_SUBSET_COUNT_25872"
    - "SHRINK1_DUAL_ACCEPTED_SET_EQUAL"
    - "SHRINK1_ACCEPTED_UNIQUE_COUNT_LE_50000"
    - "SHRINK1_FIRST_OUT_OF_BUDGET_WITNESS_NULL"
    - "TARGET_AND_CONTROL_BINDING_MANIFESTS_COMMITTED"
    - "SPLIT_COMMITMENTS_PRECEDE_HIDDEN_ACCESS"
    - "DIAGNOSTIC_FORMAL_BRIDGE_DUAL_REPLAY_EQUAL"
    - "ALL_REQUIRED_FORMAL_SPEC_AND_TARGET_ROOTS_NON_NULL"
    - "PYTHON_COMPLETE_ENUMERATOR_IMPLEMENTED"
    - "RUST_COMPLETE_ENUMERATOR_IMPLEMENTED"
    - "TRAVERSAL_AND_BUCKET_ACCOUNTING_FROZEN"
    - "PROGRAM_OUTPUT_AND_CHUNK_ARCHIVE_EMITTERS_VERIFIED"
    - "NEW_EXECUTION_MANIFEST_ROOT_NON_NULL"
    - "NEW_RUN_ID_WITH_INITIAL_STATE_NOT_RUN"
  formal_roots_required_before_run: true
  target_independent_diagnostic_dry_run_before_roots_allowed: true
  diagnostic_dry_run_is_authoritative: false
  shrink1_subset_is_complete_closure: false
  allowed_terminal_states:
    - "COMPLETE"
    - "DSL_TOO_LARGE"
    - "INCONCLUSIVE_BUDGET"
    - "INCONCLUSIVE_SEMANTICS"
    - "INCONCLUSIVE_EXECUTION"
  certificate_signatures_are_m3_entry_gate: false
  key_status_chain_is_m3_entry_gate: false
  outside_certificate_is_m3_entry_gate: false
  mdl_replay_is_m3_entry_gate: false
  invention_synthesis_is_m3_entry_gate: false
  active_governance_is_m3_entry_gate: false
```

---

# 2. 决策 1：版本 ID

## 2.1 批准的版本

```text
new_dsl_version    = hegel-old-dsl-v1.1.0
new_freeze_version = hegel-freeze-p2b-p3-v1.1.0
```

批准。

理由：

- 删除 3 个 aggregate operators 是语言 surface 的向后不兼容收缩；
- 但 AST schema、typing、coercion、CBOR 和 operator semantics framework 没变；
- 因此应提升 minor version，而不是 patch；
- `v2.0.0` 保留给 AST schema、typing、implicit coercion、equivalence 或 semantic model 的根本变化。

## 2.2 不升版的身份

```text
canonical_cbor_profile_id = hegel-cbor-det-v1
canonical_ast_schema_id   = hegel-canonical-ast-v1
mdl_code_table_id         = hegel-mdl-prefix-v1.0.0
```

保持不变。

## 2.3 Human amendment ID

批准：

```text
hegel-freeze-p2b-p3-v1.1.0-shrink-step1
```

它是文档 ID，不替代 machine freeze ID。

## 2.4 Phase-2B 的继承

`hegel-freeze-p2b-p3-v1.1.0` 必须显式声明：

```json
{
  "phase2b_contract_inherited_from": "hegel-freeze-p2b-p3-v1.0.2",
  "phase2b_contract_changed": false
}
```

避免版本提升被误读为 Phase-2B 的统计合同也发生变化。

---

# 3. 决策 2：AggregateMap ID

## 3.1 选择

选择：

```text
A / SPARSE_PRESERVING
```

冻结：

| ID | State | Entry |
|---:|---|---|
| 0 | ACTIVE | `sum_v1` |
| 1 | ACTIVE | `count_nonzero_v1` |
| 2 | TOMBSTONE | `mean_v1` |
| 3 | TOMBSTONE | `min_v1` |
| 4 | TOMBSTONE | `max_v1` |
| 5 | ACTIVE | `signed_balance_v1` |

## 3.2 `registry_width` 与 `active_map_count`

必须是不同字段：

```text
registry_width   = 6
active_map_count = 3
```

定义：

```text
registry_width = highest allocated numeric ID + 1
```

不是 active entry 数量。

## 3.3 Tombstone 的生命周期

ID 2、3、4 在：

```text
AggregateMapId/v1
```

registry lineage 内永久禁止复用。

这不仅适用于 v1.x；任何继承该 registry namespace 的后续 DSL 也不得复用。

若未来需要新增 map：

```text
next ID = 6
```

若未来 major DSL 希望从 0 重新编号，必须创建新的 registry namespace，不能继续叫 `AggregateMapId/v1`。

## 3.4 为什么不 dense reindex

若把 `signed_balance_v1` 从 5 改为 2：

- 所有 surviving signed-balance AST bytes 改变；
- AST hash 改变；
- parent evidence 无法直接比较；
- shrink 引入与语义无关的大规模 identity drift；
- tombstone 与新 entry 混淆。

这不提供任何 closure 或计算收益。

---

# 4. 决策 3：removed maps

## 4.1 Exact rejection

以下全部使用同一 error code：

```text
REJECT_REMOVED_AGGREGATE_MAP
```

适用：

```text
source name mean_v1
source name min_v1
source name max_v1
formal map ID 2
formal map ID 3
formal map ID 4
```

## 4.2 Future unknown IDs

```text
map ID >= registry_width
```

使用：

```text
REJECT_REGISTRY_INDEX_OUT_OF_RANGE
```

不能与 tombstone 混为一类。

## 4.3 Rejection stage

removed maps 必须在 strict acceptance 阶段拒绝。

不得：

- type-check；
- canonicalize；
- 计入 canonical count；
- 执行后返回 bottom；
- 自动映射到 sum 或 signed-balance；
- 延迟到 evaluator 才拒绝。

## 4.4 Legacy diagnostic migration

允许：

> 读取并验证旧 artifact 在父 DSL 下的历史身份。

不允许：

- 把旧 AST 注入新 closure；
- 自动修改成其他 aggregate；
- 以 migration 后 AST 代表原程序；
- 将旧 archive root 作为新版本 evidence。

建议 legacy read result：

```json
{
  "legacy_program_status": "VALID_UNDER_PARENT_DSL_ONLY",
  "parent_dsl_version": "hegel-old-dsl-v1.0.0",
  "admitted_under_current_dsl": false
}
```

---

# 5. 决策 4：跨版本 AST 与 semantic identity

## 5.1 分层 identity

### Syntactic identity

\[
\operatorname{SyntaxId}(P)
=
\text{strict canonical AST CBOR bytes}.
\]

AST hash：

\[
\operatorname{AstHash}(P)
=
\operatorname{ContentHash}
(\texttt{HEGEL/AST/V1},\operatorname{ASTCBOR}(P)).
\]

surviving programs 必须保持 AST bytes 和 hash 不变。

### Admission identity

定义：

```text
ProgramAdmissionIdentityV1 =
[
  canonical_ast_hash,
  dsl_spec_root,
  identifier_registry_root
]
```

它回答：

> 这个 syntactic program 是否属于该 DSL。

### Semantic identity

定义：

```text
ProgramSemanticIdentityV1 =
[
  canonical_ast_hash,
  dsl_spec_root,
  operator_semantics_root,
  identifier_registry_root
]
```

hash：

```text
ContentHash(
  "HEGEL/PROGRAM_SEMANTIC_IDENTITY/V1",
  ProgramSemanticIdentityV1
)
```

它回答：

> 这个 AST 在哪一套冻结语义下被解释。

## 5.2 必须绑定的 roots

跨 DSL 比较至少绑定：

```text
dsl_spec_root
operator_semantics_root
identifier_registry_root
canonical_ast_schema_root
canonical_cbor_profile_root
```

其中前 3 个决定 admission / semantics；后 2 个证明 bytes 的解释合同没有变化。

## 5.3 Hash 稳定性

批准：

```json
{
  "surviving_ast_bytes_stable": true,
  "surviving_ast_hash_stable": true,
  "ast_hash_domain_changed": false
}
```

不得因为 DSL 版本变更就更换：

```text
HEGEL/AST/V1
```

否则 syntax identity 与 language membership 被错误合并。

## 5.4 不允许跨版本复用

即便 AST hash 相同，以下均不得复用：

```text
canonical_program_archive_root
program_output_archive_root
chunk_manifest_root
closure receipt
match receipt
outside certificate
MDL certificate
```

因为这些对象必须绑定新 DSL / registry / semantics / execution manifest。

## 5.5 新 golden vectors

至少包括：

1. `sum_v1` ID 0：bytes/hash 与父版本一致；
2. `count_nonzero_v1` ID 1：一致；
3. `signed_balance_v1` ID 5：一致；
4. ID 2/3/4：`REJECT_REMOVED_AGGREGATE_MAP`；
5. ID 5 未被重排；
6. old removed-map CBOR 可由 generic CBOR parser 读取；
7. 同一 bytes 不能通过新 DSL strict AST acceptance。

---

# 6. 决策 5：commitment 重建

## 6.1 核心原则

区分：

\[
\boxed{
\text{payload identity}
\neq
\text{version-binding identity}
}
\]

如果 target universe / truth rows 的 canonical JCS bytes 完全相同，它们的 content IDs 应自然保持相同。

不得：

- 人为加盐制造“新 payload”；
- 为了版本号变化篡改 row order；
- 把旧 content ID 填入 formal Merkle root 字段。

但必须生成新的 manifest，将相同或重算的 payload 绑定到新 DSL 和 freeze。

## 6.2 必须生成新 identity 的对象

```text
shrunk DSL surface
active+tombstone aggregate registry
new DSL spec
new admission/operator-semantics specification
new freeze/preregistration manifest
odd target binding manifest
hidden-sink binding manifest
split binding manifest
custodian binding manifest
shrink-1 dual strict gate
shrink-1 capacity replay
future M3 execution manifest
```

## 6.3 可以 content-stable 的对象

如果 canonical bytes 相同：

```text
odd target semantic spec
480-row odd universe JCS
480-row odd truth table JCS
85-row sink universe JCS
85-row sink truth table JCS
discovery split payload
validation split payload
sealed prediction split payload
canonical AST schema payload
canonical CBOR profile payload
```

其 content ID 应保持相同。

新 manifest 必须记录：

```text
payload_content_id
parent_binding_manifest_id
new_dsl_version
new_freeze_version
target_role
split_seed_commitment
```

## 6.4 Split seed

决定：

```text
fresh_split_seed_required = false
```

并冻结：

```text
split_seed_reuse_required_if_uncompromised = true
```

理由：

- target synthesis 尚未开始；
- validation / sealed outcomes 尚未访问；
- shrink 由 target-independent capacity overflow 触发；
- 保留原 secret split 比重新抽一次更符合事前承诺；
- 反复重抽 hidden split 会制造不必要的 researcher degrees of freedom。

### 例外

若发生以下任一情况：

- split key 暴露；
- validation/sealed rows 被 synthesis agent 看见；
- row allocation 可被恢复；
- custodian 无法证明原 payload 未使用；

则不得重用。

必须：

1. 将当前 target version 标记 compromised；
2. 发布新的 target/split version；
3. 使用 fresh independent key；
4. 在任何新 synthesis 前重新 commitment。

## 6.5 Odd target 和 sink control

二者都必须重新绑定：

```text
odd target role = OUTSIDE_TARGET
sink control role = IN_LANGUAGE_NULL
```

不得只重绑 odd target。

## 6.6 旧 validation 的 disposition

冻结：

```text
HISTORICAL_PRECOMMITMENT_ONLY_SEALED
```

它不是：

```text
HISTORICAL_DEVELOPMENT_ONLY
```

因为不应主动打开旧 validation 使其变成 development。

其 payload：

- 继续 sealed；
- 旧 binding superseded；
- 若 split 未受损，可通过新 binding manifest 重新绑定相同 payload；
- 不允许旧 DSL scoring record 进入新版本正式结果。

## 6.7 Publication 时 formal roots

批准：

```text
formal_roots_at_publication = null
```

包括：

```text
dsl_spec_root
operator_semantics_root
identifier_registry_root
canonical_ast_schema_root
bounded_universe_root
target_truth_table_root
program archive roots
diagnostic_formal_bridge_root
```

本轮 publication 只冻结 source/diagnostic identities 与规范。

formal roots 只有在：

- strict CBOR conversion；
- Python/Rust bridge replay；
- RFC6962 root equality；

完成后才可填入。

## 6.8 Shrink-1 subset precommitment

批准 source count：

```text
rational aggregate maps   = 2
scopes                    = 4
quantities                = 2
rational aggregate leaves = 16
constant atoms            = 77
mixed atoms               = 336
source AND2 candidates    = 25,872
```

其中：

\[
7\times16\times3=336,
\qquad
77\times336=25{,}872.
\]

它是：

```text
shrink1_capacity_source_count
```

不是 canonical count，也不是 closure cardinality。

---

# 7. 决策 6：M3 准入

## 7.1 是否批准 checklist

批准问题文档中的 M3 checklist，并增加两个显式 gate：

```text
shrink1_accepted_unique_count <= 50000
shrink1_first_out_of_budget_witness == null
```

若 shrink-1 subset 已经接受 50,001 个：

```text
new DSL status = DSL_TOO_LARGE
```

不得进入 complete M3；必须执行预注册 shrink step 2，并再次发布新 DSL version。

## 7.2 Subset replay 的准确含义

即使：

```text
accepted_unique_count = 25,872
```

也只说明：

> 该预注册 constructive subset 没有单独证明新 DSL 超预算。

它不说明：

```text
closure COMPLETE
closure cardinality <= 50000
target OUTSIDE
```

完整 grammar 仍可能产生第 50,001 个 program。

## 7.3 Formal roots 是否必须在 M3 前非空

决定：

```text
formal_roots_required_before_run = true
```

正式 M3 run 开始前必须非空且 Python/Rust 一致：

```text
dsl_spec_root
operator_semantics_root
identifier_registry_root
canonical_ast_schema_root
canonical_cbor_profile_root
bounded_universe_root
target_truth_table_root
diagnostic_formal_bridge_root
```

理由：

- ProgramRecord 绑定 DSL 与 universe；
- ProgramOutputRecord 绑定 universe；
- match set 绑定 target root；
- execution manifest 绑定所有 specification roots；
- 先跑后补 root 会产生 orphan replay；
- 事后桥接无法证明运行时使用的正是被签名对象。

### 允许的 pre-root dry run

可以运行：

```text
target-independent diagnostic enumerator dry run
```

但必须：

```text
different diagnostic run ID
non-authoritative
no formal state transition
no archive reuse
no receipt reuse
```

正式 M3 必须整次重跑。

## 7.4 Target role 与 archive

每个 formal closure receipt 只绑定一个：

```text
target_role
bounded_universe_root
target_truth_table_root
```

因此至少产生两个 role-specific evaluation receipts：

1. odd outside target；
2. hidden-sink null control。

AST enumeration cache 可以共享用于性能优化，但：

- cache 不是 formal archive；
- 每个 role 的 output archive、match set、receipt独立重放；
- 不得把 odd target output root 用于 sink control。

## 7.5 M3 不需要 final signatures

确认：

```text
certificate_signatures_are_m3_entry_gate = false
key_status_chain_is_m3_entry_gate = false
```

M3 的任务是产生可签名的：

- archive roots；
- execution manifest；
- replay receipts；
- match set。

3/3 signatures 与 key chain 属于 M4 certificate gate。

同样不阻塞 M3：

```text
OUTSIDE_FROZEN_CLOSURE certificate
MDL replay
invention synthesis
ACTIVE governance
```

## 7.6 M3 合法终态

### COMPLETE

```text
frontier_exhausted = true
all_type_buckets_closed = true
closure_cardinality = canonical_program_count <= 50000
raw cap not hit
wall-clock abort not hit
formal archive roots non-null
match_set_count non-null
Python/Rust roots and match sets equal
```

### DSL_TOO_LARGE

```text
canonical_program_count = 50000
first_out_of_budget_program_hash non-null
frontier_exhausted = false
all_type_buckets_closed = false
closure_cardinality = null
match_set_count = null
```

后续只能进入 shrink step 2。

### INCONCLUSIVE_BUDGET

raw cap 或预算阻止闭合，且没有合法 50,001 witness。

不得携带 match verdict。

### INCONCLUSIVE_SEMANTICS

typing、bottom、operator 或 Python/Rust 语义不一致。

不得携带 match verdict。

### INCONCLUSIVE_EXECUTION

非语义执行失败。

不得携带 match verdict。

---

# 8. M3 完整准入清单

```json
{
  "m3_entry_contract_id": "hegel-m3-entry-shrink1-v1",
  "normative": {
    "shrink1_amendment_approved": true,
    "dsl_version": "hegel-old-dsl-v1.1.0",
    "freeze_version": "hegel-freeze-p2b-p3-v1.1.0",
    "ast_schema_id": "hegel-canonical-ast-v1",
    "cbor_profile_id": "hegel-cbor-det-v1",
    "aggregate_policy": "SPARSE_PRESERVING",
    "removed_map_error": "REJECT_REMOVED_AGGREGATE_MAP",
    "hash_compatibility_policy_frozen": true
  },
  "dual_strict_implementation": {
    "python_updated": true,
    "rust_updated": true,
    "shared_golden_vectors_frozen": true,
    "valid_vectors_equal": true,
    "invalid_vectors_rejected_identically": true,
    "surviving_ast_hash_stability_verified": true,
    "tombstone_rejection_verified": true
  },
  "shrink1_subset": {
    "source_count": 25872,
    "python_rust_accepted_set_equal": true,
    "accepted_unique_count_le_50000": true,
    "first_out_of_budget_witness": null,
    "semantic_disagreement": false,
    "execution_disagreement": false,
    "interpreted_as_complete_closure": false
  },
  "commitments": {
    "odd_target_binding_committed": true,
    "hidden_sink_binding_committed": true,
    "split_binding_committed": true,
    "custodian_binding_committed": true,
    "commitments_precede_hidden_access": true
  },
  "formal_bridge": {
    "python_rust_equal": true,
    "dsl_spec_root_non_null": true,
    "operator_semantics_root_non_null": true,
    "identifier_registry_root_non_null": true,
    "ast_schema_root_non_null": true,
    "cbor_profile_root_non_null": true,
    "odd_universe_root_non_null": true,
    "odd_target_root_non_null": true,
    "sink_universe_root_non_null": true,
    "sink_target_root_non_null": true,
    "diagnostic_formal_bridge_root_non_null": true
  },
  "enumeration_and_archives": {
    "python_complete_enumerator_implemented": true,
    "rust_complete_enumerator_implemented": true,
    "canonical_traversal_frozen": true,
    "bucket_accounting_frozen": true,
    "program_archive_emitter_verified": true,
    "output_archive_emitter_verified": true,
    "chunk_manifest_emitter_verified": true,
    "records_per_chunk": 4096
  },
  "execution": {
    "execution_manifest_root_non_null": true,
    "new_run_id": true,
    "initial_state": "NOT_RUN",
    "parent_run_state_reuse": false
  }
}
```

所有字段为真后：

```text
NOT_RUN -> RUNNING
```

才是正式 M3 开始。

---

# 9. Shrink transition artifact

建议生成不可变 transition record：

```text
DslShrinkTransitionV1 =
[
  schema_version,
  parent_dsl_version,
  child_dsl_version,
  parent_freeze_version,
  child_freeze_version,
  triggering_parent_receipt_id,
  parent_status,
  shrink_step_id,
  removed_registry_entries,
  surviving_registry_entries,
  tombstone_policy,
  hash_compatibility_policy_id,
  regenerated_binding_manifest_ids,
  retained_payload_content_ids,
  new_capacity_replay_id,
  child_initial_state
]
```

其中：

```text
parent_status = DSL_TOO_LARGE
child_initial_state = NOT_RUN
```

不得直接继承父 run 的：

```text
RUNNING
DSL_TOO_LARGE
COMPLETE
```

---

# 10. 当前 go/no-go

| Work item | Status |
|---|---|
| 发布 shrink-step 1 normative amendment | GO，按本文件 |
| 创建 `hegel-old-dsl-v1.1.0` | GO，用户批准后 |
| 创建 `hegel-freeze-p2b-p3-v1.1.0` | GO，用户批准后 |
| sparse/tombstone registry implementation | GO |
| Python/Rust golden vectors | GO |
| shrink-1 25,872 subset replay | GO |
| 将 25,872 解释为 complete closure | NO-GO |
| 若 subset >50,000，启动 M3 | NO-GO，进入 shrink step 2 |
| formal bridge/root generation | GO，在 binding commitments 后 |
| M3 complete enumerator run | 条件 GO，满足 §8 后 |
| target synthesis | NO-GO，等待 M3 COMPLETE 与 outside certificate |
| hidden-sink formal verdict | NO-GO，等待 M3 COMPLETE |
| outside/MDL certificate | NO-GO，留在 M4/M5 |
| Phase-2B formal exit | NO-GO |
| ACTIVE | NO-GO |

---

# 11. 最终主线结论

父版本已经获得了一个严格但有限的结论：

```text
hegel-old-dsl-v1.0.0
under the frozen 50,000 syntactic-program budget
=
DSL_TOO_LARGE
```

下一步不是修改 target，也不是放宽 budget，而是按预注册顺序缩小 old language。

本轮最重要的设计原则是：

\[
\boxed{
\text{保留 surviving syntax identity，版本化 semantic/admission identity。}
}
\]

以及：

\[
\boxed{
\text{相同 target payload 可以保持 content identity，
但必须由新的 DSL-binding manifest 重新承诺。}
}
\]

这使 shrink transition 同时具备：

- 历史可追溯性；
- wire stability；
- tombstone 可审计性；
- 新 DSL closure 的独立性；
- target precommitment 连续性；
- 不通过重抽 hidden split 获取额外自由度。

正式执行顺序保持：

\[
\boxed{
\text{shrink amendment}
\rightarrow
\text{dual subset replay}
\rightarrow
\text{formal roots}
\rightarrow
\text{M3 complete enumeration}
\rightarrow
\text{adequacy verdict}
}
\]

而不是从 `DSL_TOO_LARGE` 直接跳到 relation synthesis。
