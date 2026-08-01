# Hegel Machine Phase-3A M2.5 Bit-Exact Wire Completion Amendment

**文档类型**：Normative bit-exact completion amendment
**human document ID**：`hegel-freeze-p2b-p3-v1.1.2-m25-wire-completion`
**machine freeze ID**：`hegel-freeze-p2b-p3-v1.1.2`
**child DSL ID**：`hegel-old-dsl-v1.1.0`（不升版）
**上位规范**：

```text
hegel-freeze-p2b-p3-v1.1.1
hegel-freeze-p2b-p3-v1.1.0
hegel-old-dsl-v1.1.0
hegel-canonical-ast-v1
hegel-cbor-det-v1
hegel-mdl-prefix-v1.0.0
```

**适用范围**：

- Phase-3A M2.5 formal commitment；
- split seed first instantiation；
- custodian / auditor / implementation attester wire；
- diagnostic-to-formal bridge；
- odd/sink universe、truth 和 split roots；
- M3 execution identity、run genesis、state machine；
- complete enumerator / role evaluation / dual replay receipts；
- archive framing；
- M3 gates 15–24；
- Phase-3B 对上游 v2/SCAR 负证据的处理。

---

# 0. 当前证据与总决策

## 0.1 当前已成立

```yaml
branch: "codex/reconstruction-v2-paper"
implementation_commit: "d772b844e7c92b20f1e370244cc88202581fc72a"

strict_vectors:
  total: 20
  positive: 7
  rejection_or_priority: 13
  python_rust_equal: true

short_cbor_differential:
  sample_count: 3031
  acceptance_mismatch: 0
  error_code_mismatch: 0

shrink1:
  source_count: 25872
  accepted_unique_count: 25872
  rejected_count: 0
  rewrite_collapsed_count: 0
  first_out_of_budget_witness: null
  subset_is_complete_closure: false

machine_state:
  status: "SHRINK1_SUBSET_QUALIFIED_M3_BLOCKED"
  m3_gates_satisfied: 14
  m3_gates_total: 24
  child_state: "NOT_RUN"
  formal_roots: null
  real_seed_generated: false
  real_private_key_generated: false
  real_signature_claimed: false
```

## 0.2 对当前工作的判断

当前方向正确。

现有实现已经做到两件最重要的事：

1. 将 deterministic primitive qualification 与 authoritative artifact generation 分开；
2. 在外部 actor、seed、签名和 formal root 不存在时保持 fail-closed。

因此现在可以继续推进，但下一阶段仍是：

> **Phase-3A M2.5 — Formal Commitment, Seed Genesis and Bridge Qualification**

而不是：

- complete closure；
- target synthesis；
- outside certificate；
- MDL certificate；
- ACTIVE。

## 0.3 本 amendment 批准后的状态

```yaml
m25_wire_specification_complete: true
m25_wire_implementation: "PARTIAL_ALLOWED"
authoritative_formal_root_generation: false
split_seed_first_instantiation: false
custodian_signature_claim: false
auditor_attestation_claim: false
m3_gates_satisfied: 14
m3_gates_total: 24
m3_entry_allowed: false
m3_entry_qualified: false
m3_run_started: false
child_state: "NOT_RUN"
```

只有独立 actor 流程完成、Python/Rust formal roots 相同且 24/24 gates 通过后：

```yaml
m3_entry_allowed: true
m3_entry_qualified: true
m3_run_started: false
child_state: "NOT_RUN"
```

随后必须由单独的显式 start action 才能进入：

```text
NOT_RUN -> RUNNING/CANONICAL_ENUMERATION
```

---

# 1. IdDigestV1、digest reference 与 opaque ID

## 1.1 `IdDigestV1`

```yaml
IdDigestV1:
  input_type: "ASCII machine-id text"
  accepted_regex: "^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$"
  normalization: "NONE"
  empty_allowed: false
  nul_allowed: false
  preimage: "ASCII('HEGEL/ID_DIGEST/V1') || 0x00 || exact_ascii_id_bytes"
  hash: "SHA-256"
  output: "32-byte byte string"
  failure_codes:
    non_ascii: "REJECT_MACHINE_ID_NON_ASCII"
    invalid_syntax: "REJECT_MACHINE_ID_SYNTAX"
    length: "REJECT_MACHINE_ID_LENGTH"
```

不得：

- lowercase；
- uppercase；
- Unicode NFC/NFKC；
- trim；
- slash normalization；
- path normalization。

## 1.2 字段命名语义

### `*_id_digest`

一律：

```text
IdDigestV1(完整 human/machine ID ASCII text)
```

包括：

```text
rng_profile_id_digest
kdf_profile_id_digest
commitment_profile_id_digest
legacy_parent_payload_source_id_digest
approving_actor_id_digest
semantic_spec_diagnostic_id_digest
universe_diagnostic_id_digest
truth_diagnostic_id_digest
split_algorithm_id_digest
shrink_step_id_digest
canonical_name_digest
introduced_dsl_version_digest
removed_dsl_version_digest
diagnostic_profile_id_digest
formal_profile_id_digest
```

### `*_root` 或普通 `*_digest`

表示已经计算好的 raw digest，formal wire 中直接保存 32-byte byte string。

JSON rendering 中：

```text
sha256:<64 lowercase hex>
```

进入 formal wire 时解码为 32 raw bytes，不再做 `IdDigestV1`。

## 1.3 `<namespace>_<hex>` diagnostic ID

human diagnostic ID：

```text
<namespace>_<64 lowercase hex>
```

具有两种 formal identity：

```text
diagnostic_namespace_id = numeric enum
diagnostic_digest       = decoded 32-byte suffix
```

同时，若某 manifest 字段名称明确为：

```text
*_diagnostic_id_digest
```

则该字段保存：

```text
IdDigestV1(完整 "<namespace>_<hex>" text)
```

二者用途不同：

- suffix digest绑定 legacy payload；
- full-ID digest绑定 human-visible content-addressed identifier。

禁止仅删除 prefix 后改写成 formal root。

## 1.4 时间戳

wire 类型：

```text
unsigned integer Unix seconds
```

编码允许范围：

```text
0 <= timestamp <= 253402300799
```

其中：

- `0` 仅允许 synthetic/golden-vector profile；
- authoritative manifest/receipt 要求：
  ```text
  timestamp >= 1704067200
  ```
- authoritative timestamp 不得超过 verifier 当前 UTC 时间 300 秒。

错误：

```text
REJECT_TIMESTAMP_OUT_OF_RANGE
FAIL_AUTHORITATIVE_TIMESTAMP_ZERO
FAIL_TIMESTAMP_EXCESSIVELY_FUTURE
FAIL_TIMESTAMP_ORDERING
```

## 1.5 `run_id` / `ledger_id`

```yaml
OpaqueId128V1:
  length_bytes: 16
  generation: "OS CSPRNG"
  all_zero_allowed: false
  uuid_bit_rewriting: false
  text_rendering: "32 lowercase hex"
  wire: "16-byte byte string"
  duplicate_scope: "all project runs/ledgers reachable from the pinned genesis trust anchor"
```

持久化：

```text
AppendOnlyOpaqueIdRegistryV1
```

必须在生成后、任何 manifest 使用前，以 atomic create 记录。

错误：

```text
FAIL_OPAQUE_ID_CSPRNG_UNAVAILABLE
FAIL_OPAQUE_ID_ALL_ZERO
FAIL_OPAQUE_ID_ALREADY_USED
FAIL_OPAQUE_ID_REGISTRY_UNAVAILABLE
```

---

# 2. Numeric enum registries

共同规则：

```yaml
enum_policy:
  unknown_value: "REJECT_UNKNOWN_ENUM_VALUE"
  removed_value: "REJECT_TOMBSTONED_ENUM_VALUE"
  numeric_reuse_allowed: false
  private_extension_range: [32768, 65535]
  private_extension_allowed_in_authoritative_wire: false
```

除明确说明外，`0` 为 `INVALID/UNSPECIFIED`，authoritative object 中拒绝。

## 2.1 InputSignatureId

| Value | Name |
|---:|---|
| 0 | `INVALID` |
| 1 | `ODD_BITSET_ENTITY_SET_V1` |
| 2 | `OBSERVED_OMITTED_SINK_TUPLE_V1` |
| 3–32767 | reserved |

## 2.2 SortId

| Value | Name |
|---:|---|
| 0 | `INVALID` |
| 1 | `BOOL` |
| 2 | `BIT` |
| 3 | `SIGN` |
| 4 | `BOUNDED_INT` |
| 5 | `RATIONAL_VALUE` |
| 6 | `RATIONAL_PARAMETER` |
| 7 | `TOLERANCE` |
| 8 | `CLOSED_INTERVAL` |
| 9 | `ENTITY_SLOT` |
| 10 | `INDEX` |
| 11 | `QUANTITY_ID` |
| 12 | `CONTEXT_ID` |
| 13 | `ROLE_ID` |
| 14 | `SCALE_ID` |
| 15 | `TASK_ID` |
| 16 | `ENTITY_SET` |
| 17 | `SCOPE_ID` |
| 18 | `AGGREGATE_MAP_ID` |
| 19 | `TRANSFORM_ID` |
| 20 | `OBSERVATION` |
| 21 | `EVENT` |
| 22–32767 | reserved |

## 2.3 RegistryKindId

| Value | Name |
|---:|---|
| 0 | `INVALID` |
| 1 | `ENTITY_SLOT` |
| 2 | `QUANTITY` |
| 3 | `CONTEXT` |
| 4 | `ROLE` |
| 5 | `SCALE` |
| 6 | `TASK` |
| 7 | `SCOPE` |
| 8 | `AGGREGATE_MAP` |
| 9 | `TRANSFORM` |
| 10 | `OPERATOR` |
| 11 | `NEW_SYMBOL` |
| 12–32767 | reserved |

## 2.4 RegistryEntryStateId

| Value | Name |
|---:|---|
| 0 | `INVALID` |
| 1 | `ACTIVE` |
| 2 | `TOMBSTONE` |
| 3–32767 | reserved |

## 2.5 OperatorClassId

| Value | Name |
|---:|---|
| 0 | `INVALID` |
| 1 | `LEAF` |
| 2 | `UNARY` |
| 3 | `BINARY` |
| 4 | `TERNARY` |
| 5 | `CONJUNCTION` |
| 6 | `AGGREGATE_MAP` |
| 7 | `ADAPTER_TRANSFORM` |
| 8–32767 | reserved |

## 2.6 OperatorAdmissionStateId

| Value | Name |
|---:|---|
| 0 | `INVALID` |
| 1 | `ACTIVE_DSL` |
| 2 | `TOMBSTONE_REMOVED` |
| 3 | `ADAPTER_ONLY` |
| 4 | `RESERVED_NOT_IMPLEMENTED` |
| 5–32767 | reserved |

## 2.7 UndefinedSemanticsId

| Value | Name |
|---:|---|
| 0 | `INVALID` |
| 1 | `TOTAL_NO_BOTTOM` |
| 2 | `STRICT_BOTTOM_PROPAGATION` |
| 3 | `LEAF_INDEX_OUT_OF_RANGE_BOTTOM` |
| 4 | `EMPTY_AGGREGATE_BOTTOM` |
| 5 | `MISSING_TYPED_MEASUREMENT_BOTTOM` |
| 6 | `RATIONAL_DOMAIN_OVERFLOW_BOTTOM` |
| 7–32767 | reserved |

一个 operator 可在其 semantics entry 中引用一个主 ID；更细条件由 `executable_semantics_root` 绑定。

## 2.8 ArtifactRoleId

| Value | Name |
|---:|---|
| 0 | `INVALID` |
| 1 | `OUTSIDE_TARGET_SPEC` |
| 2 | `OUTSIDE_TARGET_UNIVERSE` |
| 3 | `OUTSIDE_TARGET_TRUTH` |
| 4 | `IN_LANGUAGE_NULL_SPEC` |
| 5 | `IN_LANGUAGE_NULL_UNIVERSE` |
| 6 | `IN_LANGUAGE_NULL_TRUTH` |
| 7 | `CHILD_DSL_SPEC` |
| 8 | `OPERATOR_SEMANTICS` |
| 9 | `IDENTIFIER_REGISTRY` |
| 10 | `CANONICAL_AST_SCHEMA` |
| 11 | `CANONICAL_CBOR_PROFILE` |
| 12 | `SPLIT_CONTRACT` |
| 13 | `DISCOVERY_SPLIT` |
| 14 | `VALIDATION_SPLIT` |
| 15 | `SEALED_PREDICTION_SPLIT` |
| 16 | `SHRINK_TRANSITION` |
| 17 | `M3_EXECUTION` |
| 18 | `NORMATIVE_APPROVAL` |
| 19 | `CUSTODIAN_BINDING` |
| 20 | `PARENT_ABSENCE_ATTESTATION` |
| 21 | `FALLBACK_REGISTRY` |
| 22–32767 | reserved |

## 2.9 DiagnosticNamespaceId

| Value | Name / prefix |
|---:|---|
| 0 | `INVALID` |
| 1 | `target_spec` |
| 2 | `sink_control_spec` |
| 3 | `bounded_universe` |
| 4 | `target_truth_table` |
| 5 | `dsl_spec` |
| 6 | `operator_semantics` |
| 7 | `identifier_registry` |
| 8 | `canonical_ast_schema` |
| 9 | `canonical_cbor_profile` |
| 10 | `split_contract` |
| 11 | `hidden_generator_spec` |
| 12 | `publication` |
| 13 | `replay` |
| 14 | `freeze_document` |
| 15–32767 | reserved |

## 2.10 FormalObjectKindId

| Value | Name |
|---:|---|
| 0 | `INVALID` |
| 1 | `CONTENT_HASH` |
| 2 | `RFC6962_TREE_ROOT` |
| 3 | `SIGNED_MANIFEST_ROOT` |
| 4 | `ARCHIVE_ROOT` |
| 5 | `RECEIPT_ROOT` |
| 6 | `EXECUTION_MANIFEST_ROOT` |
| 7–32767 | reserved |

## 2.11 DiagnosticProfileId

| Value | Name |
|---:|---|
| 0 | `INVALID` |
| 1 | `HEGEL_LEGACY_STABLE_JSON_V1` |
| 2 | `RFC8785_JCS_V1` |
| 3–32767 | reserved |

## 2.12 FormalProfileId

| Value | Name |
|---:|---|
| 0 | `INVALID` |
| 1 | `HEGEL_CBOR_CONTENT_HASH_V1` |
| 2 | `HEGEL_RFC6962_ROW_TREE_V1` |
| 3 | `HEGEL_RFC6962_ARCHIVE_TREE_V1` |
| 4–32767 | reserved |

## 2.13 StratumId

### Odd role

| Value | Set size | Target label |
|---:|---:|---:|
| 1 | 5 | 0 |
| 2 | 5 | 1 |
| 3 | 6 | 0 |
| 4 | 6 | 1 |
| 5 | 7 | 0 |
| 6 | 7 | 1 |
| 7 | 8 | 0 |
| 8 | 8 | 1 |

### Sink role

| Value | Predicate |
|---:|---|
| 9 | `d == 0` |
| 10 | `d == 1` |
| 11 | `d == 2` |
| 12 | `d == 3` |
| 13 | `d == 4` |

`14–32767` reserved。

## 2.14 PartitionId

| Value | Name |
|---:|---|
| 0 | `INVALID` |
| 1 | `DISCOVERY` |
| 2 | `VALIDATION` |
| 3 | `SEALED_PREDICTION` |
| 4–32767 | reserved |

## 2.15 EquivalenceModeId

| Value | Name |
|---:|---|
| 0 | `INVALID` |
| 1 | `EXACT_EXTENSIONAL` |
| 2–32767 | reserved |

## 2.16 ImplementationId

| Value | Name |
|---:|---|
| 0 | `INVALID` |
| 1 | `PYTHON_REFERENCE` |
| 2 | `RUST_INDEPENDENT` |
| 3 | `CUSTODIAN` |
| 4 | `AUDITOR` |
| 5–32767 | reserved |

## 2.17 ParentStatusId

| Value | Name |
|---:|---|
| 0 | `INVALID` |
| 1 | `COMPLETE` |
| 2 | `DSL_TOO_LARGE` |
| 3 | `INCONCLUSIVE_BUDGET` |
| 4 | `INCONCLUSIVE_SEMANTICS` |
| 5 | `INCONCLUSIVE_EXECUTION` |
| 6–32767 | reserved |

## 2.18 ChildInitialStateId

| Value | Name |
|---:|---|
| 0 | `INVALID` |
| 1 | `NOT_RUN` |
| 2–32767 | reserved |

## 2.19 M3TransitionReasonId

| Value | Name |
|---:|---|
| 0 | `INVALID` |
| 1 | `ENTRY_GATES_24_OF_24` |
| 2 | `ENUMERATION_FRONTIER_EXHAUSTED` |
| 3 | `CANONICAL_PROGRAM_50001_ACCEPTED` |
| 4 | `RAW_OPERATOR_CAP_HIT` |
| 5 | `WALL_CLOCK_BUDGET_HIT` |
| 6 | `SEMANTICS_OR_DUAL_REPLAY_MISMATCH` |
| 7 | `EXECUTION_FAILURE` |
| 8 | `ROLE_EVALUATION_COMPLETE` |
| 9–32767 | reserved |

## 2.20 M3ClosureStatusId

这是 receipt status，独立于 M3 state。

| Value | Name |
|---:|---|
| 0 | `NOT_RUN` |
| 1 | `COMPLETE` |
| 2 | `DSL_TOO_LARGE` |
| 3 | `INCONCLUSIVE_BUDGET` |
| 4 | `INCONCLUSIVE_SEMANTICS` |
| 5 | `INCONCLUSIVE_EXECUTION` |
| 6–32767 | reserved |

## 2.21 M3StateId

| Value | Name |
|---:|---|
| 0 | `NOT_RUN` |
| 1 | `RUNNING` |
| 2 | `COMPLETE` |
| 3 | `DSL_TOO_LARGE` |
| 4 | `INCONCLUSIVE_BUDGET` |
| 5 | `INCONCLUSIVE_SEMANTICS` |
| 6 | `INCONCLUSIVE_EXECUTION` |
| 7–32767 | reserved |

因此：

```text
M3ImplementationEnumerationReceiptV1.closure_status_id
```

必须使用 `M3ClosureStatusId`，不是 `M3StateId`。

## 2.22 M3RunningPhaseId

| Value | Name |
|---:|---|
| 0 | `NONE` |
| 1 | `CANONICAL_ENUMERATION` |
| 2 | `ROLE_EVALUATION` |
| 3–32767 | reserved |

## 2.23 RoleAgreementStatusId

| Value | Name |
|---:|---|
| 0 | `NOT_APPLICABLE` |
| 1 | `AGREED` |
| 2 | `DISAGREED` |
| 3–32767 | reserved |

---

# 3. Core root schemas

## 3.1 新增 object tags

| Object | Tag |
|---|---:|
| `NormativeDocumentBlobV1` | `0x3001` |
| `FreezeSpecV1` | `0x3002` |
| `DslSpecV1` | `0x3003` |
| `SplitContractV1` | `0x3004` |
| `TargetBundleV1` | `0x3005` |
| `ApprovalEvidenceBundleV1` | `0x3006` |
| `ReplacementPolicyV1` | `0x3007` |
| `SplitSpecFreezeV1` | `0x3008` |
| `TombstonePolicyV1` | `0x3009` |
| `CrossDslHashPolicyV1` | `0x300A` |
| `FallbackRegistryV1` | `0x300B` |
| `ImplementationBindingV1` | `0x300C` |
| `TraversalContractV1` | `0x300D` |
| `BucketAccountingContractV1` | `0x300E` |
| `ProgramArchiveContractV1` | `0x300F` |
| `OutputArchiveContractV1` | `0x3010` |
| `StateMachineContractV1` | `0x3011` |
| `RowTransformSpecV1` | `0x3012` |
| `InputSignatureSpecV1` | `0x3013` |
| `TargetSpecFormalV1` | `0x3014` |
| `SplitAlgorithmSpecV1` | `0x3015` |
| `ExecutionEnvironmentSpecV1` | `0x3016` |
| `ActorKeyManifestV1` | `0x310C` |
| `AttestationBundleV1` | `0x310D` |
| `CustodianBindingCoreV1` | `0x310B` |
| `M3RunGenesisV1` | `0x3300` |
| `MismatchRecordV1` | `0x320D` |
| `PartialDiagnosticBundleV1` | `0x320E` |
| `OddInputV1` | `0x3401` |
| `SinkInputV1` | `0x3402` |

共同结构：

```text
[1, numeric_tag, schema_id_byte_string, ...fields]
```

## 3.2 `NormativeDocumentBlobV1`

```text
schema_id = b"hegel-normative-document-blob/1"
domain    = HEGEL/NORMATIVE_DOCUMENT/V1
producer  = PUBLIC_BUILD
nullable  = false
```

```text
[
  1,
  0x3001,
  schema_id,
  repository_relative_path_id_digest,
  raw_git_blob_bytes,
  repository_commit_id
]
```

不做换行或 Unicode normalization。

## 3.3 `FreezeSpecV1`

```text
schema_id = b"hegel-freeze-spec/1"
domain    = HEGEL/FREEZE_SPEC/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x3002, schema_id,
  freeze_version_id_digest,
  parent_freeze_root_or_null,
  child_dsl_spec_root,
  phase2b_contract_root,
  canonical_ast_schema_root,
  canonical_cbor_profile_root,
  mdl_code_table_root,
  amendment_document_root,
  effective_repository_commit_id
]
```

## 3.4 `DslSpecV1`

```text
schema_id = b"hegel-dsl-spec/1"
domain    = HEGEL/DSL_SPEC/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x3003, schema_id,
  dsl_version_id_digest,
  parent_dsl_spec_root,
  canonical_ast_schema_root,
  canonical_cbor_profile_root,
  identifier_registry_root,
  operator_semantics_root,
  equivalence_mode_id,
  max_ast_depth,
  max_ast_node_count,
  max_top_level_clauses,
  max_distinct_bit_slots,
  max_aggregate_leaves,
  max_scope_clauses,
  max_composition_depth,
  max_fitted_parameters,
  max_entity_set_size,
  canonical_program_budget,
  raw_operator_application_cap,
  shrink_step_id_digest
]
```

## 3.5 `SplitContractV1`

```text
schema_id = b"hegel-split-contract/1"
domain    = HEGEL/SPLIT_CONTRACT/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x3004, schema_id,
  split_contract_version_id_digest,
  split_algorithm_spec_root,
  hkdf_profile_id_digest,
  rank_hmac_profile_id_digest,
  exhaustive_partition_required,
  odd_stratum_quota_table,
  sink_stratum_quota_table,
  assignment_ordering_rule_id,
  fallback_split_policy_id,
  hidden_artifact_scope_root
]
```

quota row：

```text
[stratum_id, universe_count, discovery_count, validation_count, sealed_count]
```

按 `stratum_id` 升序。

## 3.6 `TargetBundleV1`

```text
schema_id = b"hegel-target-bundle/1"
domain    = HEGEL/TARGET_BUNDLE/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x3005, schema_id,
  outside_target_spec_root,
  outside_target_universe_root,
  outside_target_truth_root,
  null_control_spec_root,
  null_control_universe_root,
  null_control_truth_root,
  fallback_registry_root,
  null_control_required_witness_ast_hash_or_null,
  null_control_claim_level_id
]
```

`null_control_claim_level_id`：

```text
1 = FALSE_INVENTION_NULL_ONLY
2 = MECHANISM_SPECIFIC_RECOVERY
```

当前若 sink truth output cardinality 为 1，必须为 `1`。

## 3.7 `ApprovalEvidenceBundleV1`

```text
schema_id = b"hegel-approval-evidence-bundle/1"
domain    = HEGEL/APPROVAL_EVIDENCE_BUNDLE/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x3006, schema_id,
  amendment_document_root,
  approving_actor_id_digest,
  approval_statement_id_digest,
  parent_normative_decision_root,
  approval_method_id,
  approval_recorded_at_unix_seconds
]
```

当前：

```text
approving_actor_machine_id = "project-owner:erzhu419"
approval_statement_id      = "approve:hegel-freeze-p2b-p3-v1.1.2"
```

这是非密码学 pseudonymous approval。

## 3.8 `ReplacementPolicyV1`

```text
schema_id = b"hegel-replacement-policy/1"
domain    = HEGEL/REPLACEMENT_POLICY/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x3007, schema_id,
  key_rotation_threshold,
  key_revocation_threshold,
  custodian_replacement_requires_new_seed_version,
  actor_key_reuse_across_purposes_allowed,
  secret_material_export_allowed
]
```

冻结：

```text
key_rotation_threshold = 2
key_revocation_threshold = 2
custodian_replacement_requires_new_seed_version = true
actor_key_reuse_across_purposes_allowed = false
secret_material_export_allowed = false
```

## 3.9 `SplitSpecFreezeV1`

```text
schema_id = b"hegel-split-spec-freeze/1"
domain    = HEGEL/SPLIT_SPEC_FREEZE/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x3008, schema_id,
  split_contract_root,
  target_bundle_root,
  child_freeze_root,
  amendment_document_root,
  seed_state_id,
  frozen_at_unix_seconds,
  repository_commit_id
]
```

`seed_state_id`：

```text
1 = SPEC_FROZEN_SEED_NOT_INSTANTIATED
2 = SEED_INSTANTIATED
3 = COMPROMISED_REQUIRES_NEW_VERSION
```

Commit A 中为 1。

## 3.10 `TombstonePolicyV1`

```text
schema_id = b"hegel-tombstone-policy/1"
domain    = HEGEL/TOMBSTONE_POLICY/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x3009, schema_id,
  registry_namespace_id_digest,
  id_reuse_allowed,
  removed_source_name_error_id_digest,
  removed_numeric_id_error_id_digest,
  unknown_numeric_id_error_id_digest
]
```

## 3.11 `CrossDslHashPolicyV1`

```text
schema_id = b"hegel-cross-dsl-hash-policy/1"
domain    = HEGEL/CROSS_DSL_HASH_POLICY/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x300A, schema_id,
  ast_hash_domain_id_digest,
  surviving_ast_bytes_stable,
  surviving_ast_hash_stable,
  semantic_identity_domain_id_digest,
  required_binding_root_role_ids,
  cross_version_archive_reuse_allowed,
  cross_version_receipt_reuse_allowed,
  cross_version_certificate_reuse_allowed
]
```

## 3.12 `FallbackRegistryV1`

```text
schema_id = b"hegel-fallback-registry/1"
domain    = HEGEL/FALLBACK_REGISTRY/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x300B, schema_id,
  [
    [priority, target_machine_id_digest, target_spec_root_or_null],
    ...
  ],
  selection_rule_id_digest,
  requires_new_target_version,
  requires_new_split_first_instantiation
]
```

当前 fallback target 不复用 odd split。

## 3.13 `ImplementationBindingV1`

```text
schema_id = b"hegel-implementation-binding/1"
domain    = HEGEL/IMPLEMENTATION_BINDING/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x300C, schema_id,
  implementation_id,
  source_root,
  binary_digest,
  execution_environment_spec_root,
  compiler_or_interpreter_id_digest,
  compiler_or_interpreter_version_digest,
  dependency_lock_root,
  build_profile_id_digest,
  entrypoint_id_digest,
  golden_vector_root,
  repository_commit_id
]
```

Python/Rust 使用同一 schema，以 `implementation_id` 区分。

## 3.14 `TraversalContractV1`

```text
schema_id = b"hegel-traversal-contract/1"
domain    = HEGEL/TRAVERSAL_CONTRACT/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x300D, schema_id,
  bucket_key_field_ids,
  canonical_sort_key_field_ids,
  commutative_child_ordering_rule_id_digest,
  maximum_canonical_programs,
  maximum_raw_operator_applications,
  frontier_exhaustion_definition_id_digest
]
```

## 3.15 `BucketAccountingContractV1`

```text
schema_id = b"hegel-bucket-accounting-contract/1"
domain    = HEGEL/BUCKET_ACCOUNTING_CONTRACT/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x300E, schema_id,
  bucket_key_field_ids,
  required_counter_field_ids,
  bucket_ordering_rule_id_digest,
  zero_count_bucket_emission_required,
  accounting_sum_invariants
]
```

## 3.16 `ProgramArchiveContractV1`

```text
schema_id = b"hegel-program-archive-contract/1"
domain    = HEGEL/PROGRAM_ARCHIVE_CONTRACT/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x300F, schema_id,
  program_record_schema_tag,
  program_ordering_rule_id_digest,
  records_per_chunk,
  chunk_blob_codec_id,
  chunk_blob_framing_rule_id_digest,
  rfc6962_profile_id_digest,
  target_independent
]
```

## 3.17 `OutputArchiveContractV1`

```text
schema_id = b"hegel-output-archive-contract/1"
domain    = HEGEL/OUTPUT_ARCHIVE_CONTRACT/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x3010, schema_id,
  output_record_schema_tag,
  output_ordering_rule_id_digest,
  records_per_chunk,
  chunk_blob_codec_id,
  chunk_blob_framing_rule_id_digest,
  undefined_bitmap_profile_id_digest,
  role_specific
]
```

## 3.18 `StateMachineContractV1`

```text
schema_id = b"hegel-m3-state-machine-contract/1"
domain    = HEGEL/M3_STATE_MACHINE_CONTRACT/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x3011, schema_id,
  m3_state_registry_root,
  m3_phase_registry_root,
  m3_transition_reason_registry_root,
  legal_transition_table,
  terminal_state_ids,
  reopen_allowed
]
```

## 3.19 `RowTransformSpecV1`

```text
schema_id = b"hegel-row-transform-spec/1"
domain    = HEGEL/ROW_TRANSFORM_SPEC/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x3012, schema_id,
  source_diagnostic_profile_id,
  source_namespace_id,
  target_formal_profile_id,
  target_object_tag,
  transform_rule_id_digest,
  ordering_rule_id_digest,
  expected_row_count_or_null
]
```

## 3.20 `InputSignatureSpecV1`

```text
schema_id = b"hegel-input-signature-spec/1"
domain    = HEGEL/INPUT_SIGNATURE_SPEC/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x3013, schema_id,
  input_signature_id,
  input_object_tag,
  field_sort_ids,
  static_role_metadata,
  canonical_ordering_rule_id_digest
]
```

## 3.21 `TargetSpecFormalV1`

```text
schema_id = b"hegel-target-spec-formal/1"
domain    = HEGEL/TARGET_SPEC_FORMAL/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x3014, schema_id,
  role_id,
  target_machine_id_digest,
  input_signature_spec_root,
  output_sort_id,
  target_rule_id_digest,
  universe_row_count,
  target_output_cardinality,
  required_witness_ast_hash_or_null,
  claim_level_id
]
```

## 3.22 `SplitAlgorithmSpecV1`

```text
schema_id = b"hegel-split-algorithm-spec/1"
domain    = HEGEL/SPLIT_ALGORITHM_SPEC/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x3015, schema_id,
  os_csprng_profile_id_digest,
  hkdf_profile_id_digest,
  rank_hmac_profile_id_digest,
  rank_tie_break_rule_id_digest,
  exhaustive_partition_required,
  assignment_row_schema_tag
]
```

## 3.23 `ExecutionEnvironmentSpecV1`

```text
schema_id = b"hegel-execution-environment-spec/1"
domain    = HEGEL/EXECUTION_ENVIRONMENT_SPEC/V1
producer  = PUBLIC_BUILD
```

```text
[
  1, 0x3016, schema_id,
  os_id_digest,
  architecture_id_digest,
  runtime_id_digest,
  runtime_version_id_digest,
  dependency_lock_root,
  locale_id_digest,
  timezone_id_digest,
  container_or_host_profile_id_digest,
  oci_manifest_digest_or_null
]
```

`environment_image_digest` 字段必须保存该 object root；不是任意 OCI string。

## 3.24 RFC6962 roots

以下不是 ContentHash object，而是按已冻结 row schema直接做 RFC6962：

```text
identifier_registry_root
operator_semantics_root
removed_registry_entry_root
surviving_registry_entry_root
diagnostic_formal_bridge_root
bounded_universe_roots
target_truth_roots
split_partition_roots
bucket_accounting_root
program/output/match/archive roots
```

---

# 4. Odd / sink formal rows

## 4.1 Input objects

### `OddInputV1`

```text
[
  1,
  0x3401,
  b"hegel-odd-input/1",
  set_size,
  [bit_0, ..., bit_(set_size-1)]
]
```

约束：

- `set_size ∈ {5,6,7,8}`；
- bit 为 CBOR uint `0/1`；
- bits 数量必须等于 set_size；
- bits 使用 MSB-first bitstring order。

### `SinkInputV1`

```text
[
  1,
  0x3402,
  b"hegel-sink-input/1",
  a,
  b,
  c,
  d
]
```

约束：

```text
a,b,c,d ∈ {0,1,2,3,4}
d = a + b - c
```

orientation、quantity 和 scope 的静态含义由：

```text
InputSignatureSpecV1(input_signature_id=2)
```

绑定，不重复写入每一 row。

## 4.2 Canonical input hash

```text
canonical_input_hash
=
ContentHash(
  "HEGEL/CANONICAL_INPUT/V1",
  canonical_input_object
)
```

## 4.3 Universe / truth rows

### Universe

```text
[
  1,
  0x3201,
  b"hegel-bounded-universe-row/1",
  universe_index,
  input_signature_id,
  canonical_input_object
]
```

### Truth

```text
[
  1,
  0x3202,
  b"hegel-target-truth-row/1",
  universe_index,
  canonical_input_hash,
  target_output
]
```

`target_output` 为 CBOR uint Bit `0/1`，不是 bool。

## 4.4 Row ordering

### Odd

```text
set_size ascending
then bitstring numeric value ascending
bits encoded MSB-first
```

indices：

```text
size 5: 0..31
size 6: 32..95
size 7: 96..223
size 8: 224..479
```

### Sink

按：

```text
(a,b,c,d) lexicographic ascending
```

过滤满足约束的 85 行，indices `0..84`。

## 4.5 Roots

分别对排序后的 rows 直接做 RFC6962：

```text
outside_target_universe_root  # 480 leaves
outside_target_truth_root     # 480 leaves
null_control_universe_root    # 85 leaves
null_control_truth_root       # 85 leaves
```

失败：

```text
FAIL_UNIVERSE_INDEX_DUPLICATE
FAIL_UNIVERSE_INDEX_GAP
FAIL_CANONICAL_INPUT_HASH_MISMATCH
FAIL_TARGET_OUTPUT_TYPE
FAIL_INPUT_SIGNATURE_MISMATCH
FAIL_ROW_ORDERING
```

## 4.6 Sink control 的 claim boundary

若当前 85-row truth table 对所有 valid rows 均为 `1`：

- 它仍可作为：
  ```text
  FALSE_INVENTION_NULL_ONLY
  ```
- 不能单独声称系统辨认了 conservation mechanism；
- `DslRoleBindingManifestV1.required_witness_ast_hash` 必须绑定预注册的 signed-balance witness；
- formal control pass 还必须要求该 witness 位于 match set。

只有未来新 control version 含 valid/invalid contrast rows、target output cardinality ≥2 时，才可提升到：

```text
MECHANISM_SPECIFIC_RECOVERY
```

本 M2.5 不修改既有 85-row payload。

---

# 5. Exact split contracts

## 5.1 Odd split

allocation 在每个 `StratumId` 内独立 rank 后切分。

| Stratum | Size | Label | Universe | Discovery | Validation | Sealed |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 0 | 16 | 6 | 3 | 7 |
| 2 | 5 | 1 | 16 | 6 | 3 | 7 |
| 3 | 6 | 0 | 32 | 13 | 6 | 13 |
| 4 | 6 | 1 | 32 | 13 | 6 | 13 |
| 5 | 7 | 0 | 64 | 26 | 13 | 25 |
| 6 | 7 | 1 | 64 | 26 | 13 | 25 |
| 7 | 8 | 0 | 128 | 51 | 26 | 51 |
| 8 | 8 | 1 | 128 | 51 | 26 | 51 |
| **Total** |  |  | **480** | **192** | **96** | **192** |

要求：

```text
每 row 恰好分配一次
每 stratum 内先 rank 再切 quota
不得全局 rank 后回填
```

fallback target：

```text
不得共用 odd split
```

因为 label strata 改变。fallback 需新 target/split version，且只能在 closure membership 决定后首次实例化。

## 5.2 Sink split

早期“per scale”文本不适用于当前 85-row control，因为 row 中没有 scale。它被本 amendment 明确替换为：

```text
sink_scale_semantics = NOT_APPLICABLE
sink_stratification = d_value
```

| Stratum | Predicate | Universe | Discovery | Validation | Sealed |
|---:|---|---:|---:|---:|---:|
| 9 | `d == 0` | 15 | 7 | 4 | 4 |
| 10 | `d == 1` | 18 | 8 | 4 | 6 |
| 11 | `d == 2` | 19 | 9 | 4 | 6 |
| 12 | `d == 3` | 18 | 8 | 4 | 6 |
| 13 | `d == 4` | 15 | 7 | 4 | 4 |
| **Total** |  | **85** | **39** | **20** | **26** |

要求：

```text
exhaustive = true
每 row 恰好分配一次
discovery d=0 >= 4
discovery d>0 >= 4
validation >= 8
sealed >= 8
```

## 5.3 SplitAssignmentRowV1

```text
[
  1,
  0x3203,
  b"hegel-split-assignment-row/1",
  role_id,
  universe_index,
  canonical_input_hash,
  stratum_id,
  partition_id,
  rank_digest
]
```

每个 role+partition 独立 RFC6962 root。

leaf ordering：

```text
universe_index ascending
```

虽然 row 内仍包含 role/partition。

六个 roots：

```text
outside_discovery_split_root
outside_validation_split_root
outside_sealed_split_root
null_discovery_split_root
null_validation_split_root
null_sealed_split_root
```

## 5.4 Assignment visibility

M3 前：

```text
all assignment row payloads = custodian sealed
public = roots + counts + signed manifests only
```

Phase-3B synthesis 开始时：

- outside discovery assignment 可释放给 synthesis runner；
- outside validation/sealed保持 sealed；
- null discovery可用于 control runner；
- null validation/sealed保持 sealed。

---

# 6. CustodianBindingCore 与 root DAG

## 6.1 `CustodianBindingCoreV1`

```text
tag       = 0x310B
schema_id = b"hegel-custodian-binding-core/1"
domain    = HEGEL/CUSTODIAN_BINDING_CORE/V1
```

```text
[
  1, 0x310B, schema_id,
  custodian_key_id,
  custodian_public_key_32_bytes,
  custodian_key_epoch,
  responsibility_bitmask,
  valid_from_unix_seconds,
  valid_until_unix_seconds_or_null,
  replacement_policy_root,
  repository_commit_id
]
```

不包含：

- seed commitment；
- ledger；
- seed continuity。

## 6.2 References bind object root, not envelope root

所有 manifest 间引用一律绑定：

```text
enclosed canonical object root
```

签名 envelope root 由：

```text
AttestationBundleV1
```

单独绑定。

## 6.3 `AttestationBundleV1`

```text
tag       = 0x310D
schema_id = b"hegel-attestation-bundle/1"
domain    = HEGEL/ATTESTATION_BUNDLE/V1
```

```text
[
  1, 0x310D, schema_id,
  [
    [purpose_id, enclosed_object_root, signed_envelope_root],
    ...
  ]
]
```

按：

```text
(purpose_id, enclosed_object_root, signed_envelope_root)
```

升序。

## 6.4 Topological construction order

```text
01 commit A
02 normative document / static source blobs
03 CBOR profile / AST schema / enum registries
04 identifier registry rows -> identifier_registry_root
05 operator semantics rows -> operator_semantics_root
06 canonical AST / CBOR profile roots
07 child DslSpec -> child_dsl_spec_root
08 FreezeSpec -> child_freeze_root
09 target specs + odd/sink formal rows -> target/universe/truth roots
10 fallback registry -> fallback_registry_root
11 target bundle -> target_bundle_root
12 split algorithm + split contract -> split_contract_root
13 split spec freeze -> split_spec_freeze_root
14 approval evidence + approval manifest
15 auditor actor key + parent absence attestation
16 custodian actor key + custodian core
17 split seed first instantiation
18 seed commitment manifest + signature envelope
19 hidden-access ledger genesis + signature envelope
20 seed continuity manifest + signature envelope
21 final custodian binding + signature envelope
22 split assignment rows -> six split roots
23 split binding manifest
24 diagnostic-formal bridge records -> bridge root
25 Python/Rust/custodian bridge attestations -> bridge attestation bundle
26 odd role binding manifest
27 sink role binding manifest
28 formal shrink transition
29 implementation/contract roots
30 M3 execution manifest
31 M3 run genesis
32 gate 24 qualification
33 optional explicit start transition
```

---

# 7. Independent custodian genesis

## 7.1 Actor eligibility

Codex 自动运行环境不得充当 independent custodian。

允许：

- 用户本人在独立 one-shot command 中执行；
- 外部指定人员；
- 与 repo-building agent 隔离的本地 OS account/process。

最低要求：

```text
custodian key ID != Python attester key ID
custodian key ID != Rust attester key ID
custodian key ID != auditor key ID
```

## 7.2 Key generation

```yaml
key_algorithm: "Ed25519"
private_key_seed_length: 32
generation_profile: "hegel-os-csprng-v1"
key_epoch_initial_value: 0
```

使用 OS CSPRNG，与 split seed 分别调用。

key ID 继承已冻结规则：

```text
first 16 bytes of SHA256(raw 32-byte Ed25519 public key)
```

## 7.3 Secret storage

研究级最低允许：

```text
repo 外目录
directory mode 0700
private key file mode 0600
split seed file mode 0600
不在 cloud-sync 目录
不写 shell history
不输出 stdout/stderr
磁盘至少使用 OS account-level access control
```

OS key store / hardware token 推荐，但 M2.5 不强制。

禁止：

```text
repo
Git LFS
artifact bundle
argv
environment variable
command history
log
core dump
temporary world-readable file
```

## 7.4 Second invocation guard

在调用 CSPRNG 前：

1. 对 secret-state directory 获取 exclusive lock；
2. 以 `O_CREAT|O_EXCL` 创建：
   ```text
   split_seed_instantiation.marker
   ```
3. marker 中保存：
   ```text
   split version digest
   seed commitment manifest root or PENDING
   custodian key ID
   created timestamp
   ```
4. 已存在则：
   ```text
   FAIL_SPLIT_SEED_ALREADY_INSTANTIATED
   ```
5. seed/manifest完成后 atomic replace marker 的 PENDING 状态；
6. fsync file 与 directory。

PENDING 中断时不得再次生成 seed；需要 external recovery procedure，而不是重抽。

## 7.5 向 Python/Rust 提供 seed

允许：

```text
one-shot inherited anonymous pipe FD
```

冻结 FD：

```text
FD 3
```

流程：

- custodian process 从 secret storage 读 32 bytes；
- 分别启动 Python/Rust split calculator；
- 通过 inherited FD 3 写入 exactly 32 bytes；
- argv/env/stdin 不含 seed；
- child 读取完关闭 FD；
- child 不落盘、不打印；
-内存尽力 mlock 和 zeroize；
-输出仅为 assignment roots / commitment evidence。

## 7.6 Public / sealed outputs

可以公开 commit：

```text
public key manifest
key ID
seed commitment digest
signed seed commitment envelope
ledger genesis root/envelope
split roots/counts
binding manifests
formal roots
```

必须 sealed：

```text
raw private key
raw split seed
derived K_role
row-level split assignments
validation/sealed membership
pre-final role evaluation outputs
```

---

# 8. Auditor 与 parent absence

## 8.1 Independence

auditor 必须：

```text
auditor key ID != custodian key ID
```

且应由不同人员执行。仅使用不同 key、但同一自动 agent，不满足 independent auditor claim。

若无独立 auditor：

```text
FAIL_AUDITOR_ATTESTATION_MISSING
```

## 8.2 Audited parent commit

parent binding absence 的 authoritative commit：

```text
fb3a3ee4865a140c558821017ddd3e9a6a99de48
```

原因：它是父 DSL strict overflow 的 evidence boundary。

另记录：

```text
pre_genesis_repository_commit =
commit A of this amendment
```

用于检查 commit A 不含 secret material，但不替代 parent absence claim。

## 8.3 Audited source tree

不直接使用 Git tree SHA 作为唯一审计根。

`AuditedPathBlobRecordV1`：

```text
[
  1,
  0x3210,
  b"hegel-audited-path-blob-record/1",
  repository_path_utf8_bytes,
  git_object_algorithm_id,
  git_blob_digest,
  file_mode,
  byte_length
]
```

按 path bytes 升序做 RFC6962：

```text
audited_source_tree_root
```

## 8.4 Audited path set

必须覆盖：

```text
Hegel Machine/**
.gitattributes
.gitignore
所有 Hegel Machine manifest 中声明的外部 artifact inventory
父 commit 可达历史中所有触及上述路径的 commits
```

history record root 需绑定：

```text
commit ID
parent IDs
touched path root
```

不能只看 parent snapshot。

## 8.5 Absence reason

bitmask：

| Bit | Meaning |
|---:|---|
| 0 | `TYPED_BINDING_MANIFEST_NEVER_INSTANTIATED` |
| 1 | `LEGACY_ONLY_DIAGNOSTIC_PAYLOAD_EXISTED` |
| 2 | `NO_VERIFIABLE_SPLIT_SEED_COMMITMENT_OR_ALLOCATION_FOUND` |
| 3 | `NO_VERIFIABLE_HIDDEN_ACCESS_LEDGER_FOUND` |

当前要求：

```text
absence_reason_bitmask = 0b1111
```

它只声明“未发现可验证历史 artifact”，不证明任何 off-repo seed 从未被人脑或未记录程序生成过。

## 8.6 Auditor signature

attestation 进入 `SignedManifestEnvelopeV1`。

signature message：

```text
UTF8("HEGEL/AUDITOR_ATTESTATION_SIGNATURE/V1")
|| 0x00
|| attestation_root
```

---

# 9. Approval manifest 与 two-commit choreography

## 9.1 Amendment document root

使用：

```text
NormativeDocumentBlobV1
```

绑定 Git blob raw bytes，不做文本 normalization。

## 9.2 Approval evidence

`approval_method_id=1` 时：

```text
approval_evidence_root =
ApprovalEvidenceBundleV1 root
```

绑定：

- amendment document root；
- project-owner pseudonymous actor；
- approval statement ID；
- parent decision root。

不把当前聊天 UI 当成 cryptographic identity。

## 9.3 Approving actor

```text
approving_actor_machine_id = "project-owner:erzhu419"
approving_actor_id_digest  = IdDigestV1(machine_id)
```

其语义：

```text
NON_CRYPTOGRAPHIC_PROJECT_OWNER_PSEUDONYM
```

## 9.4 Child DSL root

`NormativeApprovalManifestV1.child_dsl_spec_root` 必须 non-null。

## 9.5 Two-commit choreography

批准：

### Commit A

包含：

```text
normative amendment
deterministic implementation
golden vectors
schemas
tests
no real seed
no private key
no authoritative signature
```

### External one-shot

绑定 Commit A：

```text
custodian/auditor/attester key genesis
seed first instantiation
formal root replay
signed envelopes
M3 execution manifest
```

### Commit B

只允许提交：

```text
public manifests
public keys
roots
receipts
signed envelopes
readiness/status artifacts
```

禁止修改 executable implementation。

formal object 中：

```text
repository_commit_id = Commit A
```

Commit B 只是 publication carrier，不产生自引用。

若 A→B 间 implementation 变化：

```text
FAIL_PUBLICATION_COMMIT_CONTAINS_IMPLEMENTATION_CHANGE
```

必须创建新 Commit A2，并重新执行 formal replay；seed是否保留由 compromise policy决定，不能自动重抽。

---

# 10. Diagnostic profile 与 bridge

## 10.1 选择

采用：

```text
LEGACY_DIAGNOSTIC_PROFILE
```

保留现有 diagnostic IDs。

不将当前 `stable_hash` 冒充 RFC8785 JCS。

## 10.2 Exact legacy profile

```yaml
profile_id: "hegel-legacy-stable-json-v1"
canonicalize:
  dataclass: "field order; exclude content_id and version_id"
  enum: "enum.value"
  path: "str(path)"
  mapping: "str(key), sorted by str(key)"
  tuple_list: "JSON array, preserve order"
  set_frozenset: "canonicalize elements, sort by canonical_json"
  primitive: ["null", "string", "integer", "float", "boolean"]
json_dumps:
  ensure_ascii: false
  sort_keys: true
  separators: [",", ":"]
  allow_nan: false
digest:
  hash: "SHA-256"
  domain_prefix: null
  preimage: "UTF-8 JSON bytes"
scope:
  existing_objects_only: true
  new_normative_objects_allowed: false
```

## 10.3 Bridge required records

M2.5 bridge root至少包含 12 records：

1. child DSL spec；
2. operator semantics；
3. identifier registry；
4. AST schema；
5. CBOR profile；
6. odd target spec；
7. odd universe；
8. odd truth；
9. sink control spec；
10. sink universe；
11. sink truth；
12. split contract。

按：

```text
(artifact_role_id, diagnostic_namespace_id, diagnostic_digest)
```

升序直接 RFC6962。

```text
record_count = 12
```

## 10.4 Row transform specs

至少三个：

```text
LEGACY_OBJECT_TO_FORMAL_SPEC_V1
ODD_DIAGNOSTIC_ROWS_TO_FORMAL_V1
SINK_DIAGNOSTIC_ROWS_TO_FORMAL_V1
```

split assignments从 custodian raw rows直接生成 formal rows，不经过 legacy diagnostic profile。

---

# 11. Bridge signatures

## 11.1 选择

选择：

```text
B. bridge-specific 3/3 signatures
```

M2.5 要求：

- custodian bridge signature；
- Python replay attester signature；
- Rust replay attester signature。

这不等于 M4 final certificate signature。

## 11.2 Attester keys

三种 purpose：

| Purpose | ID |
|---|---:|
| `CUSTODIAN_BRIDGE_ATTESTER` | 1 |
| `PYTHON_BRIDGE_ATTESTER` | 2 |
| `RUST_BRIDGE_ATTESTER` | 3 |
| `AUDITOR` | 4 |
| `FINAL_CERTIFICATE_SIGNER` | 5 |

每一 purpose 使用独立 Ed25519 key。不得跨 purpose 复用。

`ActorKeyManifestV1`：

```text
[
  1,
  0x310C,
  b"hegel-actor-key-manifest/1",
  purpose_id,
  key_id,
  public_key_32_bytes,
  key_epoch,
  valid_from_unix_seconds,
  valid_until_unix_seconds_or_null,
  repository_commit_id
]
```

domain：

```text
HEGEL/ACTOR_KEY_MANIFEST/V1
```

## 11.3 Bridge signature

```text
UTF8("HEGEL/BRIDGE_ATTESTATION_SIGNATURE/V1")
|| 0x00
|| diagnostic_formal_bridge_root
|| execution_manifest_candidate_root
```

bridge attestation bundle 要求：

```text
three distinct key IDs
purpose IDs exactly {1,2,3}
all signatures valid
```

缺一：

```text
FAIL_BRIDGE_ATTESTATION_THRESHOLD
```

---

# 12. Missing formal objects

## 12.1 `BucketAccountingRecordV1`

```text
[
  1,
  0x320C,
  b"hegel-bucket-accounting-record/1",
  bucket_index,
  output_sort_id,
  ast_depth,
  ast_node_count,
  raw_operator_applications,
  accepted_canonical_programs,
  syntactic_duplicates,
  type_rejections,
  structural_limit_rejections,
  rewrite_collapses,
  first_program_index_or_null,
  last_program_index_or_null
]
```

ordering：

```text
bucket_index ascending
```

root：RFC6962。

## 12.2 `MismatchRecordV1`

```text
[
  1,
  0x320D,
  b"hegel-mismatch-record/1",
  mismatch_index,
  mismatch_kind_id,
  python_object_root_or_null,
  rust_object_root_or_null,
  affected_program_index_or_null,
  diagnostic_detail_digest
]
```

ordering：`mismatch_index ascending`。

## 12.3 `PartialDiagnosticBundleV1`

```text
[
  1,
  0x320E,
  b"hegel-partial-diagnostic-bundle/1",
  run_id,
  implementation_id,
  terminal_failure_code_id_digest,
  completed_bucket_count,
  partial_bucket_accounting_root_or_null,
  partial_log_digest,
  authoritative_claim_allowed
]
```

必须：

```text
authoritative_claim_allowed = false
```

domain：

```text
HEGEL/PARTIAL_DIAGNOSTIC_BUNDLE/V1
```

## 12.4 Implementation binding

Python/Rust 都使用 `ImplementationBindingV1`；不创建两个不同 schema。

## 12.5 Environment digest

`environment_image_digest` 字段保存：

```text
ExecutionEnvironmentSpecV1 root
```

若使用 OCI，OCI digest只是该 spec中的字段。

## 12.6 INCONCLUSIVE_BUDGET

精确触发：

```text
raw_operator_application_count reaches 5,000,000
before frontier exhaustion and before accepted program 50,001
```

或：

```text
predeclared wall-clock budget sends graceful stop,
receipt is successfully finalized,
and no 50,001 witness exists
```

unexpected kill / OOM / nonzero process exit不是 budget，属于：

```text
INCONCLUSIVE_EXECUTION
```

## 12.7 Process exit

```text
exit_code == 0
```

是 authoritative receipt 必要条件。

任何 nonzero：

```text
INCONCLUSIVE_EXECUTION
```

且正式 archive roots必须为 null。

跨实现语义不一致由 dual validator判：

```text
INCONCLUSIVE_SEMANTICS
```

---

# 13. Archive bit-exact rules

## 13.1 Program MDL length

```text
CanonicalProgramRecordV2.program_mdl_length_q32
```

必须使用：

```text
hegel-mdl-prefix-v1.0.0
```

无法计算：

```text
FAIL_PROGRAM_MDL_LENGTH_UNAVAILABLE
```

整个 run：

```text
INCONCLUSIVE_SEMANTICS
```

## 13.2 Blob codec

v1 使用：

```text
codec_id = 0 = IDENTITY_V1
```

不使用 zstd/gzip。

chunk blob：

```text
for each record in order:
  uint32_be(record_cbor_length)
  record_cbor_bytes
```

无 header、无 trailer、无 padding。

blob digest：

```text
SHA256(
  UTF8("HEGEL/CHUNK_BLOB/V1")
  || 0x00
  || exact_blob_bytes
)
```

## 13.3 Root identities

```text
canonical_program_archive_root
=
RFC6962 over CanonicalProgramRecordV2
```

```text
program_chunk_manifest_root
=
RFC6962 over ProgramChunkManifestV2
```

两者不得互换。

## 13.4 Role output root

```text
program_output_archive_root(role)
=
RFC6962 directly over ProgramOutputRecordV2
```

chunk root另算，不先按 chunk聚合后替代 record root。

## 13.5 Match set

空：

```text
SHA256(empty byte string)
```

非空排序：

```text
canonical_ast_hash
then output_blob_hash
```

## 13.6 Undefined bitmap

继承：

```text
HEGEL/UNDEFINED_BITMAP/V1
```

## 13.7 Hash collision

发现同 digest 对应不同 canonical preimage时，必须比较：

- AST hash：canonical AST CBOR bytes；
- output hash：canonical output blob bytes；
- record leaf：canonical record CBOR bytes；
- diagnostic digest：legacy JSON bytes；
- content root：canonical object CBOR bytes。

任何不同：

```text
FAIL_SHA256_PREIMAGE_COLLISION
```

全流程 abort，不尝试二级 hash。

## 13.8 Odd/sink output roots

`ProgramOutputRecordV2` 必须包含 `role_id` 和 role-specific universe root。

因此即使 output values偶然相同，preimage仍不同。

若 two role archive roots仍数值相同：

```text
FAIL_SHA256_PREIMAGE_COLLISION
```

不得将其视为合法 dedup。

---

# 14. M3 run genesis 与 output null slots

## 14.1 `M3RunGenesisV1`

```text
[
  1,
  0x3300,
  b"hegel-m3-run-genesis/1",
  run_id_16_bytes,
  execution_manifest_root,
  initial_state_id,
  canonical_program_archive_root_or_null,
  program_chunk_manifest_root_or_null,
  bucket_accounting_root_or_null,
  outside_program_output_archive_root_or_null,
  outside_output_chunk_manifest_root_or_null,
  outside_match_set_root_or_null,
  outside_role_evaluation_receipt_root_or_null,
  null_program_output_archive_root_or_null,
  null_output_chunk_manifest_root_or_null,
  null_match_set_root_or_null,
  null_role_evaluation_receipt_root_or_null,
  python_enumeration_receipt_root_or_null,
  rust_enumeration_receipt_root_or_null,
  dual_replay_agreement_root_or_null,
  final_state_record_root_or_null,
  created_at_unix_seconds,
  repository_commit_id
]
```

domain：

```text
HEGEL/M3_RUN_GENESIS/V1
```

## 14.2 Gate 24

pass 要求：

```text
execution_manifest_root non-null
run genesis root non-null
initial_state = NOT_RUN
all 16 run-produced output slots = null
run_id fresh and registered
```

失败：

```text
FAIL_M3_OUTPUT_ROOT_PREPOPULATED
```

## 14.3 Qualified 与 started

M2.5 完成后：

```yaml
m3_entry_qualified: true
m3_entry_allowed: true
m3_run_started: false
child_state: "NOT_RUN"
```

不得自动生成 `NOT_RUN -> RUNNING`。

显式 operator action：

```text
phase3-m3-start --run-genesis-root ...
```

才创建 state record：

```text
NOT_RUN/NONE -> RUNNING/CANONICAL_ENUMERATION
```

## 14.4 Run ID 时点

run ID 在创建 `M3RunGenesisV1` 前生成并写入 append-only registry。

M2.5 qualification不会偶发启动 complete closure。

---

# 15. Hidden artifact scope

## 15.1 Public before M3

对开发者/审稿者公开：

```yaml
public_before_m3:
  - child DSL and freeze specifications
  - odd target machine ID and mathematical definition
  - odd universe generator and parity truth predicate
  - sink generator and balance predicate
  - 480/85 universe cardinalities
  - stratum/quota tables
  - split algorithm specification
  - target/universe/truth diagnostic IDs
  - formal roots after generation
  - public keys
  - seed commitment digest
  - signed manifest envelopes
```

因此：

```text
ledger genesis-only
```

绝不声称 target law/definition 从未被开发者或 Codex看见。

## 15.2 Custodian-sealed before M3

```yaml
custodian_sealed_before_m3:
  - raw Ed25519 private keys
  - raw K_split_master
  - all derived K_role
  - exact odd row-to-partition assignment
  - exact sink row-to-partition assignment
  - validation membership
  - sealed-prediction membership
  - row-level assignment payloads
  - pre-final role evaluation outputs
  - pre-final match-set details
```

## 15.3 Event type 2 subjects

`HIDDEN_ARTIFACT_ACCESS_GRANTED` 用于：

```text
raw seed
derived role key
split assignment payload
validation/sealed membership
pre-final output archive
pre-final match set
```

不用于已经公开的 target definition。

## 15.4 Event type 4 subjects

`HIDDEN_ARTIFACT_REVEALED` 表示上述 sealed object被提供给：

- synthesis agent；
-未授权开发进程；
- public repository；
-非 custodian/auditor actor。

## 15.5 Synthesis agent forbidden inputs

```yaml
synthesis_agent_forbidden_inputs:
  - target machine ID and target name
  - target rule source code
  - full enumerated target truth table
  - validation assignment
  - sealed assignment
  - validation outcomes
  - sealed outcomes
  - raw split seed
  - derived role keys
  - old-closure role output archives
  - match-set contents
  - oracle relation implementation
```

虽然这些信息可对项目开发者公开，synthesis runner 必须是受限输入环境。这是：

```text
agent-facing blindness
```

不是全局 secrecy。

---

# 16. M2.5 failure registry

## 16.1 Categories

### `BLOCKED_SPECIFICATION`

```text
FAIL_M25_WIRE_UNDERSPECIFIED
FAIL_ID_DIGEST_PROFILE_UNFROZEN
FAIL_FORMAL_ROOT_PREIMAGE_UNFROZEN
FAIL_SINK_SPLIT_CONTRACT_UNFROZEN
FAIL_HIDDEN_ARTIFACT_SCOPE_UNFROZEN
FAIL_M3_OUTPUT_SLOT_WIRE_UNFROZEN
```

### `BLOCKED_EXTERNAL_CUSTODY`

```text
FAIL_CUSTODIAN_KEY_MISSING
FAIL_SPLIT_SEED_NOT_INSTANTIATED
FAIL_CUSTODIAN_SIGNATURE_MISSING
FAIL_AUDITOR_ATTESTATION_MISSING
```

### `INCONCLUSIVE_SEMANTICS`

```text
FAIL_DIAGNOSTIC_PROFILE_MISMATCH
FAIL_FORMAL_GOLDEN_VECTOR_MISMATCH
FAIL_FORMAL_BRIDGE_MISMATCH
FAIL_DUAL_REPLAY_MISMATCH
```

### `INCONCLUSIVE_EXECUTION`

```text
FAIL_OS_CSPRNG_RUNTIME
FAIL_SECRET_PIPE_RUNTIME
FAIL_PROCESS_NONZERO_EXIT
FAIL_ARTIFACT_IO
FAIL_HOST_TERMINATION
```

## 16.2 Synthetic materials

synthetic test key/seed：

```text
只能进入 golden/synthetic artifact
```

必须带：

```text
AUTHORITATIVE_USE_PROHIBITED
```

不得：

- 填入 authoritative manifest；
- 签 gate 15–24；
- 写成 seed first instantiation；
- 使 gate status变成 passed。

---

# 17. v2 / SCAR 负证据

## 17.1 Exact status

```yaml
upstream_v2_counterevidence:
  evidence_commit: "4861b2d88ef7e85fb62f32e3d2e1f5c78afe9529"
  evidence_status: "PROTOCOL_VALID_NEGATIVE"
  primary_delta_pair_f1: -0.6728865211
  primary_ci95:
    - -0.7095084627
    - -0.6358874857
  diagnosed_risk: "HARD_STRUCTURAL_ELIGIBILITY_COVERAGE_COLLAPSE"

  m25_formal_wire_gate_effect: "NONE"
  m3_old_dsl_closure_gate_effect: "NONE"
  outside_certificate_effect: "NONE"

  phase3b_design_risk: "HARD_STRUCTURAL_ELIGIBILITY_COVERAGE_COLLAPSE"
  transfer_existing_v2_weights_or_thresholds: false
  treat_v2_priors_as_verified_positive_priors: false
  require_shadow_only_ablation: true
```

该结果不允许改变：

- M2.5 CBOR bytes；
- roots；
- odd/sink target；
- split seed；
- M3 budget；
- closure semantics。

## 17.2 Bounded core Phase-3B controls

对 odd-cardinality bounded invention，v2/semantic RAG arms并不具有同一输入语义，因此首个核心实验使用：

| Arm | Definition |
|---|---|
| `CORE_A_OLD_CLOSURE` | complete old-language best program / abstain |
| `CORE_B_OLD_LANGUAGE_NO_INVENTION_SEARCH` | parameter/scope/composition only |
| `CORE_C_HEGEL_INVENTED_RELATION` | new symbol with sealed evaluation |
| `CORE_D_ORACLE_RELATION_UPPER_BOUND` | preregistered oracle reducer |
| `CORE_E_LOOKUP_MEMORIZATION_CONTROL` | row lookup，必须在 sealed split失效 |

这组回答“是否发明了 language extension”。

## 17.3 Downstream GSCL/RAG transfer controls

在后续真实关系任务中采用：

| Arm | Definition |
|---|---|
| `TRANSFER_A_NO_PRIOR` | old-language / no structural prior |
| `TRANSFER_B_FROZEN_V2_PRIOR` | v2 hard selector，禁止 retune |
| `TRANSFER_C_HEGEL_INVENTED_SOFT` | invented relation作为 soft residual/feature，保留 semantic fallback |
| `TRANSFER_D_SEMANTIC_ONLY` | semantic baseline |
| `TRANSFER_E_HEGEL_HARD_GATE_ABLATION` | hard eligibility风险消融 |

不得默认把 invented relation 作为 hard selector。

## 17.4 WikiSQL UAO P4

选择：

```text
1. historical reference only
```

它属于不同 task/domain/protocol，不做直接数值对照。

若以后需要比较，必须新开独立 preregistered reference arm。

---

# 18. Shared golden vectors

以下为本 amendment 的 normative formal-row vectors。它们不等于当前 authoritative roots；只有 Python/Rust 独立 replay 与外部 manifest完成后，formal fields才可非 null。

## 18.1 IdDigest

```text
input ASCII:
hegel-old-dsl-v1.1.0

preimage hex:
484547454c2f49445f4449474553542f563100686567656c2d6f6c642d64736c2d76312e312e30

digest:
49022ed9fa53522e10dd60ce5da983a4ac0be2d7bc8c7737f6d5ae1dc88c4703
```

## 18.2 Odd row 0

Input：

```text
set_size=5
bits=00000
target=0
```

Odd input CBOR：

```text
850119340151686567656c2d6f64642d696e7075742f3105850000000000
```

canonical input hash：

```text
9b1690571f1a85c3368ad55a112be39c791758acbc3cf17b5a37cfd7ffc06558
```

Universe row CBOR：

```text
8601193201581c686567656c2d626f756e6465642d756e6976657273652d726f772f310001850119340151686567656c2d6f64642d696e7075742f3105850000000000
```

Universe leaf hash：

```text
82d372f5c01c3cb6acbc296e7499bf66c9d69fa96f01dc214a14399ea40300c3
```

Truth row CBOR：

```text
86011932025818686567656c2d7461726765742d74727574682d726f772f310058209b1690571f1a85c3368ad55a112be39c791758acbc3cf17b5a37cfd7ffc0655800
```

Truth leaf hash：

```text
5cd8da3a9e56a7d5d38b160ec4d6715f578c8d90af0a5037db36b4044fe30036
```

## 18.3 Odd row 1

```text
set_size=5
bits=00001
target=1
```

Universe row CBOR：

```text
8601193201581c686567656c2d626f756e6465642d756e6976657273652d726f772f310101850119340151686567656c2d6f64642d696e7075742f3105850000000001
```

Universe leaf hash：

```text
3687134ca503f7d6bfe583be468003df01066ab16cb72664d974561b609fd74e
```

Truth row CBOR：

```text
86011932025818686567656c2d7461726765742d74727574682d726f772f3101582041de3d87149e3d5a9491c856d674c500cdb5dec260cce2c4f4e9c9e7114ee9ea01
```

Truth leaf hash：

```text
17ae2956c4cfb90d94ea54439c0efb1cc8bc8f7fc6e3e95776216b14b29b8990
```

Two-row roots：

```text
odd_universe_two_row_root =
a10e24853c11986ceec4a7167c8dca3a7587261dbc0fcd5df0dfc9f7604acf24

odd_truth_two_row_root =
b6e2a6d9808cb9c0542a0bcf5cd4af398a419e275221733736045fe3de960fd6
```

Full 480-row expected roots：

```text
odd_universe_480_root =
b7e6eed1174ceee1944bd540cbb0ace4c5c7fc0dffea79dd956a51f7a2410a05

odd_truth_480_root =
f5bbdc26bec62f9966e5ef31eaa800190ed52dedc73ee61545e0f9c122a1a506
```

## 18.4 Sink row 0

```text
(a,b,c,d)=(0,0,0,0)
target=1
```

Sink input CBOR：

```text
870119340252686567656c2d73696e6b2d696e7075742f3100000000
```

canonical input hash：

```text
3f72fa41888208bfdaca3a6926003556a5a1375bc0185413c1645f6b2e10471b
```

Universe row CBOR：

```text
8601193201581c686567656c2d626f756e6465642d756e6976657273652d726f772f310002870119340252686567656c2d73696e6b2d696e7075742f3100000000
```

Universe leaf hash：

```text
b505d6646bca49e669c4f26f0ff2fbb73db60d5a0e14f4e21ab60fe607768a88
```

Truth row CBOR：

```text
86011932025818686567656c2d7461726765742d74727574682d726f772f310058203f72fa41888208bfdaca3a6926003556a5a1375bc0185413c1645f6b2e10471b01
```

Truth leaf hash：

```text
6f6e82713aacd2d6f9471d2893810540f887bdfff73ffb0541170f206a787bcb
```

## 18.5 Sink row 1

```text
(a,b,c,d)=(0,1,0,1)
target=1
```

Universe row CBOR：

```text
8601193201581c686567656c2d626f756e6465642d756e6976657273652d726f772f310102870119340252686567656c2d73696e6b2d696e7075742f3100010001
```

Universe leaf hash：

```text
aca75776c48d03ad5738638b465482cca65f303fa88e37812b0d92e17cee7201
```

Truth row CBOR：

```text
86011932025818686567656c2d7461726765742d74727574682d726f772f310158207e00b8d30c05362e63c9f5b4a8217bc8ce90356fcdc9009507aece95d87539f201
```

Truth leaf hash：

```text
188092325daf0ac152753b9518df3358f30cc112ac0249827202069708bf6591
```

Two-row roots：

```text
sink_universe_two_row_root =
2d06e5870c0ea2a67468f814647f8b11b6cd60243ff4c399d7031d99c33a9b13

sink_truth_two_row_root =
bac8bb909d6bf86b097c9a97e3656173cadcda3b1c6a8e7184fc5be256118c32
```

Full 85-row expected roots：

```text
sink_universe_85_root =
1a46f9967ad48df6ec1d9be609de701413ce86171fb0f92495a982bef0f40ff5

sink_truth_85_root =
9c0f5d75ea3c31f6cb1ea9917346a7a3f480ae9ce0ac0cb3bb21aac9d3bd7808
```

若当前 diagnostic sink truth并非上述 valid-row Bit1 profile，bridge必须失败：

```text
FAIL_SINK_TRUTH_PROFILE_MISMATCH
```

不得静默适配。

---

# 19. Gates 15–24

| Gate | Pass predicate | Failure |
|---:|---|---|
| 15 `SPLIT_SEED_FIRST_INSTANTIATION_SIGNED` | real 32-byte seed first-instantiated；commitment envelope由 pinned custodian key验证 | `FAIL_SPLIT_SEED_NOT_INSTANTIATED` / `FAIL_CUSTODIAN_SIGNATURE_MISSING` |
| 16 `HIDDEN_ACCESS_LEDGER_GENESIS_ONLY` | ledger count=1；head=genesis；无 access/reveal event | `FAIL_M3_LEDGER_HEAD_NOT_GENESIS` |
| 17 `PARENT_MANIFEST_ABSENCE_ATTESTED` | independent auditor envelope valid；bitmask=15；parent commit正确 | `FAIL_AUDITOR_ATTESTATION_MISSING` |
| 18 `FORMAL_BINDING_MANIFESTS_CANONICALIZED` | approval、split、custodian、continuity、role、transition strict decode/re-encode通过 | `FAIL_M25_WIRE_UNDERSPECIFIED` |
| 19 `FORMAL_SPEC_AND_REGISTRY_ROOTS_DUAL_EQUAL` | DSL/operator/registry/AST/CBOR roots Python=Rust | `FAIL_FORMAL_GOLDEN_VECTOR_MISMATCH` |
| 20 `ODD_UNIVERSE_AND_TRUTH_ROOTS_DUAL_EQUAL` | roots等于本 amendment 480-row golden roots | `FAIL_FORMAL_BRIDGE_MISMATCH` |
| 21 `SINK_UNIVERSE_AND_TRUTH_ROOTS_DUAL_EQUAL` | roots等于本 amendment 85-row golden roots | `FAIL_FORMAL_BRIDGE_MISMATCH` |
| 22 `SPLIT_PARTITION_ROOTS_DUAL_EQUAL` | 六个 roots Python=Rust；quota/exhaustive checks通过 | `FAIL_SPLIT_CUSTODIAN_BINDING_MISMATCH` |
| 23 `M3_STATE_AND_RECEIPT_WIRE_GOLDEN_TESTS_PASS` | state/receipt/archive/genesis vectors双端 bit-identical | `FAIL_FORMAL_GOLDEN_VECTOR_MISMATCH` |
| 24 `M3_EXECUTION_MANIFEST_ROOT_NON_NULL_AND_OUTPUT_ROOTS_NULL` | execution/genesis roots non-null；所有 output slots null；run ID fresh | `FAIL_M3_OUTPUT_ROOT_PREPOPULATED` |

只有：

```text
24/24
```

才设置：

```yaml
m3_entry_qualified: true
m3_entry_allowed: true
m3_run_started: false
child_state: "NOT_RUN"
```

---

# 20. M3 后续 phase

M2.5 完成后的下一阶段是：

> **Phase-3A M3 — Complete Frozen-Closure Enumeration and Role Evaluation**

任务：

1. 创建显式 start state record；
2. Python/Rust 完整枚举；
3. 生成 target-independent program archive；
4. 若 50,001 witness：`DSL_TOO_LARGE`，进入 shrink step 2；
5. 若 frontier closed ≤50,000：进入 role evaluation；
6. 分别计算 odd/sink output archive与 match set；
7. 双 replay一致后：
   - odd match=0：准备 outside certificate；
   - odd match>0：降为 in-language control并按 fallback registry；
   - sink designated witness出现：null control通过；
8. M3 itself不签 final certificate，签名在 M4。

---

# 21. 最终状态

```yaml
freeze_version: "hegel-freeze-p2b-p3-v1.1.2"
child_dsl_version: "hegel-old-dsl-v1.1.0"

m25_wire_specification_complete: true
m25_deterministic_foundation_qualified: true
m25_external_actor_work_required: true

authoritative_formal_root_generation: false
split_seed_first_instantiation: false
custodian_signature_claim: false
auditor_attestation_claim: false

m3_gates_satisfied: 14
m3_gates_total: 24
m3_entry_allowed: false
m3_entry_qualified: false
m3_run_started: false
child_state: "NOT_RUN"

outside_certificate_allowed: false
mdl_certificate_allowed: false
target_synthesis_allowed: false
phase2b_formal_exit: false
active_promotion_allowed: false

next_phase: "PHASE3A_M25_EXTERNAL_GENESIS_AND_FORMAL_ROOT_QUALIFICATION"
```

---

# 22. 最终主线结论

当前可以继续施工，但必须按以下顺序：

\[
\boxed{
\text{bit-exact spec completion}
\rightarrow
\text{external seed/key genesis}
\rightarrow
\text{formal roots and bridge}
\rightarrow
\text{24/24 M3 qualification}
\rightarrow
\text{explicit M3 start}
}
\]

不能因为：

- 20/20 synthetic vectors；
- 3,031 differential samples；
- 467 Python tests；
- Rust tests全部通过；

就把 synthetic identity qualification替代成外部 custody 或 formal execution evidence。

同时，v2/SCAR 的 decisive negative 不影响 M2.5 和 closure validity，但它对后续 Phase-3B 提供了明确约束：

\[
\boxed{
\text{新关系首先作为 soft evidence 与 semantic fallback 共存，
不得未经新实验直接充当 hard eligibility gate。}
}
\]
