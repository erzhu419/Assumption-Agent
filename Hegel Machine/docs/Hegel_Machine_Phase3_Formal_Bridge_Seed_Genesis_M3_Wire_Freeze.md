# Hegel Machine Phase-3 Formal Bridge、Seed Genesis 与 M3 Run Wire 冻结决策

**文档类型**：Normative formal-commitment amendment
**建议 human document ID**：`hegel-freeze-p2b-p3-v1.1.1-formal-bridge-seed-m3-wire`
**machine freeze ID**：`hegel-freeze-p2b-p3-v1.1.1`
**child DSL ID**：`hegel-old-dsl-v1.1.0`（不升版）
**父规范**：

```text
hegel-freeze-p2b-p3-v1.1.0
hegel-old-dsl-v1.1.0
hegel-canonical-ast-v1
hegel-cbor-det-v1
```

**当前证据边界**：

> Repository transcription note (before first commit/root): the two closing
> display equations contained accidental control bytes from `\\boxed`,
> `\\text`, and `\\rightarrow` escaping. They were repaired to the literal
> LaTeX commands before any `amendment_document_root` was minted.

```json
{
  "strict_vector_count": 23,
  "python_rust_strict_vectors_equal": true,
  "shrink1_source_count": 25872,
  "python_accepted_unique_count": 25872,
  "rust_accepted_unique_count": 25872,
  "rejected_count": 0,
  "rewrite_collapsed_count": 0,
  "first_out_of_budget_witness": null,
  "accepted_set_commitment": "sha256:653fcb9428684cfed11c3f2345ac95ed98ded6e31564c9eeabf97c57ee71a7e9",
  "subset_is_complete_closure": false,
  "m3_gates_satisfied": 14,
  "m3_gates_total": 24,
  "child_state": "NOT_RUN",
  "formal_roots": null,
  "current_status": "SHRINK1_SUBSET_QUALIFIED_M3_BLOCKED"
}
```

该结果证明：

> 预注册 shrink-1 constructive subset 在 Python/Rust strict acceptance 下完全一致，并且该 subset 本身没有触发 50,000-program overflow。

它不证明：

- 完整 old-language closure 小于等于 50,000；
- child closure `COMPLETE`；
- odd-cardinality target 在 closure 外；
- hidden-sink control 在 closure 内；
- outside/MDL certificate 可签发。

---

# 0. 总体决定与下一阶段

## 0.1 当前方向

方向正确，且 shrink-1 子阶段已经完成。

现在不应再次 shrink，也不应直接运行 hidden synthesis。下一子阶段正式命名为：

> **Phase-3A M2.5 — Formal Commitment, Seed Genesis and Bridge Qualification**

其任务是补齐剩余 10 个 M3 gates：

1. 第一次实例化 split seed；
2. 创建 hidden-access ledger genesis；
3. 生成 parent-manifest absence attestation；
4. 将 approval、binding、split、custodian、seed continuity、shrink transition 转为 strict formal CBOR；
5. 生成并双重验证全部 M3 input roots；
6. 冻结 M3 state machine；
7. 冻结 target-independent enumeration receipt；
8. 冻结 odd/sink 两套 role-specific evaluation receipts；
9. 冻结 dual replay agreement；
10. 创建新的 M3 execution manifest。

完成后才允许：

```text
child state: NOT_RUN -> RUNNING
```

## 0.2 当前 go/no-go

| 工作 | 决定 |
|---|---|
| M2.5 formal-wire / seed / bridge 施工 | **GO** |
| split seed 第一次实例化 | **GO，按本文件** |
| retrospective parent binding manifest | **NO-GO** |
| formal roots 生成 | **GO，双实现一致后** |
| non-authoritative enumerator dry run | **GO** |
| formal M3 run | **条件 GO，24/24 gates 后** |
| target synthesis | **NO-GO** |
| outside/MDL certificate | **NO-GO** |
| Phase-2B formal exit | **NO-GO** |
| ACTIVE | **NO-GO** |

---

# 1. 全局 formal-wire 约定

## 1.1 Canonical profile

所有本文件定义的 formal object 使用：

```text
canonical_cbor_profile_id = hegel-cbor-det-v1
hash_algorithm            = SHA-256
```

继承规则：

```text
ContentHash(domain, object)
=
SHA256(
  UTF8(domain)
  || 0x00
  || CanonicalCBOR(object)
)
```

formal core：

- 使用 numeric arrays；
- schema ID 使用 ASCII UTF-8 **byte string**，不是 CBOR text string；
- digest/root 使用 32-byte byte string；
- key ID 使用 16-byte byte string；
- Git SHA-1 使用 `[1, 20-byte digest]`；
- null 使用 CBOR null；
- 禁止 map、float、tag、indefinite length 和 trailing bytes；
- decoder 必须 exact re-encode。

## 1.2 Object numeric tags

| Object | Numeric tag |
|---|---:|
| `NormativeApprovalManifestV1` | `0x3101` |
| `DslRoleBindingManifestV1` | `0x3102` |
| `SplitSeedCommitmentManifestV1` | `0x3103` |
| `SplitBindingManifestV1` | `0x3104` |
| `CustodianBindingManifestV1` | `0x3105` |
| `SeedContinuityManifestV1` | `0x3106` |
| `ParentManifestAbsenceAttestationV1` | `0x3107` |
| `HiddenAccessLedgerRecordV1` | `0x3108` |
| `DslShrinkTransitionFormalV1` | `0x3109` |
| `M3ExecutionManifestV1` | `0x310A` |
| `SignedManifestEnvelopeV1` | `0x31FF` |
| `BoundedUniverseRowV1` | `0x3201` |
| `TargetTruthRowV1` | `0x3202` |
| `SplitAssignmentRowV1` | `0x3203` |
| `IdentifierRegistryEntryV1` | `0x3204` |
| `OperatorSemanticsEntryV1` | `0x3205` |
| `DiagnosticFormalBridgeRecordV1` | `0x3206` |
| `CanonicalProgramRecordV2` | `0x3207` |
| `ProgramOutputRecordV2` | `0x3208` |
| `ProgramChunkManifestV2` | `0x3209` |
| `RoleOutputChunkManifestV2` | `0x320A` |
| `MatchRecordV2` | `0x320B` |
| `BucketAccountingRecordV1` | `0x320C` |
| `M3RunStateRecordV1` | `0x3301` |
| `M3ImplementationEnumerationReceiptV1` | `0x3302` |
| `M3RoleEvaluationReceiptV1` | `0x3303` |
| `M3DualReplayAgreementV1` | `0x3304` |
| `M3RoleAgreementEntryV1` | `0x3305` |

每个 object 以：

```text
[1, numeric_tag, schema_id_bytes, ...fields]
```

开始。

---

# 2. 问题 1：历史 split seed 不存在

## 2.1 唯一决定

将历史状态正式定义为：

```text
SPLIT_SPEC_FROZEN_BUT_SEED_NEVER_INSTANTIATED
```

允许 independent custodian **第一次实例化** split seed。

该动作必须称为：

```text
FIRST_INSTANTIATION
```

不得称为：

```text
REUSE
REDRAW
RESEED
```

不需要发布新的 target version，因为此前从未产生任何 realized split allocation，也没有可被挑选或重抽的 seed。

## 2.2 Machine-readable 决定

```yaml
split_seed_decision:
  historical_state: "SPLIT_SPEC_FROZEN_BUT_SEED_NEVER_INSTANTIATED"
  action: "FIRST_INSTANTIATION"
  require_external_recovery: false
  parent_seed_commitment_root: null
  seed_length_bytes: 32
  rng_profile_id: "hegel-os-csprng-v1"
  kdf_profile_id: "hegel-hkdf-sha256-split-v1"
  commitment_profile_id: "hegel-split-seed-commitment-v1"
  redraw_allowed: false
  second_instantiation_under_same_split_version_allowed: false
```

## 2.3 Seed generation

```text
K_split_master ∈ {0,1}^256
```

由 custodian 使用：

```text
hegel-os-csprng-v1
```

生成。

允许的系统 primitive：

- Linux `getrandom(2)`；
- Windows `BCryptGenRandom`；
- macOS `getentropy` / `SecRandomCopyBytes`。

要求：

- 阻塞直到成功；
- 不允许伪随机 fallback；
- 不允许时间戳、UUID、Git hash、模型 seed 或 master/bootstrap seed派生；
- 失败则：
  ```text
  FAIL_SPLIT_SEED_CSPRNG_UNAVAILABLE
  ```

## 2.4 HKDF

冻结：

```text
salt = UTF8("HEGEL/SPLIT/HKDF/SALT/V1")
PRK  = HKDF-Extract-SHA256(salt, K_split_master)
```

派生：

```text
K_role(role_id)
=
HKDF-Expand-SHA256(
  PRK,
  UTF8("HEGEL/SPLIT/ROLE/V1") || uint16_be(role_id),
  32
)
```

row rank：

```text
rank(row, role_id)
=
HMAC-SHA256(
  K_role(role_id),
  UTF8("HEGEL/SPLIT/RANK/V1")
  || uint16_be(role_id)
  || uint16_be(stratum_id)
  || canonical_input_hash
)
```

按 rank bytes 升序；碰撞时按 `canonical_input_hash` 升序。若二者均相同但 row 不同：

```text
FAIL_SPLIT_RANK_IDENTITY_COLLISION
```

## 2.5 Seed commitment

```text
split_seed_commitment_digest
=
SHA256(
  UTF8("HEGEL/SPLIT_MASTER_SEED_COMMITMENT/V1")
  || 0x00
  || K_split_master
)
```

seed 不进入 repo、manifest、ledger 或 certificate。

## 2.6 SplitSeedCommitmentManifestV1

schema ID：

```text
hegel-split-seed-commitment-manifest/1
```

CBOR array：

```text
[
  1,
  0x3103,
  b"hegel-split-seed-commitment-manifest/1",
  split_contract_root,
  target_bundle_root,
  split_seed_commitment_digest,
  32,
  rng_profile_id_digest,
  kdf_profile_id_digest,
  commitment_profile_id_digest,
  custodian_key_id,
  created_at_unix_seconds,
  repository_commit_id
]
```

hash domain：

```text
HEGEL/SPLIT_SEED_COMMITMENT_MANIFEST/V1
```

## 2.7 Custodian signature

沿用：

```text
SignatureRecordV1 =
[
  key_id_16_bytes,
  signature_64_bytes
]
```

签名输入：

```text
UTF8("HEGEL/CUSTODIAN_MANIFEST_SIGNATURE/V1")
|| 0x00
|| manifest_root
```

`SignedManifestEnvelopeV1`：

```text
[
  1,
  0x31FF,
  b"hegel-signed-manifest-envelope/1",
  enclosed_object_tag,
  enclosed_manifest_root,
  created_at_unix_seconds,
  custodian_key_epoch,
  [SignatureRecordV1]
]
```

seed commitment、seed continuity 和 access-ledger genesis 要求：

```text
signature_count = 1
signature_key_id = pinned custodian_key_id
```

这不是最终 3/3 certificate signature。

## 2.8 Hidden-access ledger genesis

事件 enum：

| Event | ID |
|---|---:|
| `SPLIT_SEED_FIRST_INSTANTIATION` | 1 |
| `HIDDEN_ARTIFACT_ACCESS_GRANTED` | 2 |
| `HIDDEN_ARTIFACT_ACCESS_DENIED` | 3 |
| `HIDDEN_ARTIFACT_REVEALED` | 4 |
| `LEDGER_CLOSED` | 5 |

genesis：

```text
HiddenAccessLedgerRecordV1 =
[
  1,
  0x3108,
  b"hegel-hidden-access-ledger-record/1",
  ledger_id_16_bytes,
  0,
  null,
  1,
  custodian_key_id,
  split_seed_commitment_manifest_root,
  null,
  null,
  created_at_unix_seconds,
  repository_commit_id
]
```

字段顺序：

```text
schema
tag
schema_id
ledger_id
sequence_number
previous_record_root
event_type_id
actor_key_id
subject_manifest_root
revealed_artifact_root
authorization_root
timestamp
repository_commit
```

hash domain：

```text
HEGEL/HIDDEN_ACCESS_LEDGER_RECORD/V1
```

M3 entry 时要求：

```text
ledger_record_count = 1
ledger_head_root = genesis_record_root
no event_type in {2,4}
```

若 hidden artifact 已被访问或 revealed：

```text
FAIL_HIDDEN_ACCESS_BEFORE_M3
```

并必须新建 target/split version与 fresh seed。

---

# 3. 问题 2：parent binding manifest 缺失

## 3.1 唯一决定

选择方案 1：

```text
parent_binding_manifest_root = null
```

并要求：

```text
legacy_parent_payload_source_id != null
parent_manifest_absence_attestation_root != null
```

禁止 retrospective parent manifest。

原因：

- retrospective object 可以记录当前对历史的描述；
- 不能伪装成当时存在的 precommitment；
- 不能回溯修改 parent evidence chronology。

## 3.2 ParentManifestAbsenceAttestationV1

schema ID：

```text
hegel-parent-manifest-absence-attestation/1
```

CBOR：

```text
[
  1,
  0x3107,
  b"hegel-parent-manifest-absence-attestation/1",
  parent_dsl_version_digest,
  parent_freeze_version_digest,
  parent_repository_commit_id,
  audited_source_tree_root,
  audited_path_set_root,
  legacy_parent_payload_source_id_digest,
  absence_reason_bitmask,
  auditor_key_id,
  audited_at_unix_seconds
]
```

bitmask：

| Bit | Meaning |
|---:|---|
| 0 | `TYPED_BINDING_MANIFEST_NEVER_INSTANTIATED` |
| 1 | `LEGACY_ONLY_DIAGNOSTIC_PAYLOAD_EXISTED` |

当前：

```text
absence_reason_bitmask = 0b00000011
```

hash domain：

```text
HEGEL/PARENT_MANIFEST_ABSENCE_ATTESTATION/V1
```

## 3.3 Binding manifest 中的表示

```text
parent_binding_manifest_root = null
legacy_parent_payload_source_id_digest = non-null
parent_manifest_absence_attestation_root = non-null
```

presence guard：

```text
(parent_binding_manifest_root != null)
XOR
(
  legacy_parent_payload_source_id_digest != null
  AND parent_manifest_absence_attestation_root != null
)
```

否则：

```text
FAIL_PARENT_BINDING_PROVENANCE_INCOMPLETE
```

---

# 4. 问题 3：formal manifest wire

## 4.1 NormativeApprovalManifestV1

```text
schema_id = hegel-normative-approval-manifest/1
hash_domain = HEGEL/NORMATIVE_APPROVAL_MANIFEST/V1
```

```text
[
  1,
  0x3101,
  b"hegel-normative-approval-manifest/1",
  amendment_document_root,
  parent_freeze_root,
  child_freeze_root,
  child_dsl_spec_root_or_null,
  approval_status_id,
  approval_method_id,
  approval_evidence_root,
  approving_actor_id_digest,
  recorded_at_unix_seconds,
  repository_commit_id
]
```

enum：

```text
approval_status_id:
  1 = APPROVED
  2 = REJECTED
  3 = SUPERSEDED

approval_method_id:
  1 = USER_DECISION_RECORDED_IN_COMMITTED_NORMATIVE_DOCUMENT
  2 = EXTERNAL_DIGITAL_SIGNATURE
```

当前使用：

```text
approval_status_id = 1
approval_method_id = 1
```

不得声称 user cryptographic signature。

## 4.2 DslRoleBindingManifestV1

role enum：

```text
1 = OUTSIDE_TARGET
2 = IN_LANGUAGE_NULL
```

```text
schema_id = hegel-dsl-role-binding-manifest/1
hash_domain = HEGEL/DSL_ROLE_BINDING_MANIFEST/V1
```

```text
[
  1,
  0x3102,
  b"hegel-dsl-role-binding-manifest/1",
  role_id,
  child_dsl_spec_root,
  child_freeze_root,
  operator_semantics_root,
  identifier_registry_root,
  canonical_ast_schema_root,
  canonical_cbor_profile_root,
  semantic_spec_diagnostic_id_digest,
  semantic_spec_formal_root,
  universe_diagnostic_id_digest,
  truth_diagnostic_id_digest,
  formal_universe_root,
  formal_truth_root,
  split_binding_manifest_root,
  custodian_binding_manifest_root,
  seed_continuity_manifest_root,
  parent_binding_manifest_root_or_null,
  legacy_parent_payload_source_id_digest_or_null,
  parent_manifest_absence_attestation_root_or_null,
  fallback_registry_root_or_null,
  created_at_unix_seconds,
  repository_commit_id
]
```

guards：

```text
FAIL_ROLE_ROOT_BINDING_MISMATCH
FAIL_PARENT_BINDING_PROVENANCE_INCOMPLETE
FAIL_SPLIT_CUSTODIAN_BINDING_MISMATCH
```

## 4.3 SplitBindingManifestV1

```text
schema_id = hegel-split-binding-manifest/1
hash_domain = HEGEL/SPLIT_BINDING_MANIFEST/V1
```

```text
[
  1,
  0x3104,
  b"hegel-split-binding-manifest/1",
  split_contract_root,
  split_seed_commitment_manifest_root,
  seed_continuity_manifest_root,
  split_algorithm_id_digest,
  outside_target_discovery_root,
  outside_target_validation_root,
  outside_target_sealed_root,
  null_control_discovery_root,
  null_control_validation_root,
  null_control_sealed_root,
  hidden_access_ledger_genesis_root,
  hidden_access_ledger_head_root,
  split_instantiation_status_id,
  created_at_unix_seconds,
  repository_commit_id
]
```

status：

```text
1 = FIRST_INSTANTIATION
2 = VERIFIED_REUSE
3 = FRESH_AFTER_COMPROMISE
```

当前为 1。

## 4.4 CustodianBindingManifestV1

```text
schema_id = hegel-custodian-binding-manifest/1
hash_domain = HEGEL/CUSTODIAN_BINDING_MANIFEST/V1
```

```text
[
  1,
  0x3105,
  b"hegel-custodian-binding-manifest/1",
  custodian_key_id,
  custodian_public_key_32_bytes,
  custodian_key_epoch,
  responsibility_bitmask,
  split_seed_commitment_manifest_root,
  hidden_access_ledger_genesis_root,
  seed_continuity_manifest_root,
  valid_from_unix_seconds,
  valid_until_unix_seconds_or_null,
  replacement_policy_root,
  repository_commit_id
]
```

responsibility bits：

| Bit | Responsibility |
|---:|---|
| 0 | seed generation |
| 1 | seed custody |
| 2 | split allocation |
| 3 | hidden-access ledger |
| 4 | formal bridge attestation |
| 5 | answer reveal |

M3 前：

```text
bits 0..4 = 1
bit 5 = 0
```

## 4.5 SeedContinuityManifestV1

```text
schema_id = hegel-seed-continuity-manifest/1
hash_domain = HEGEL/SEED_CONTINUITY_MANIFEST/V1
```

status：

```text
1 = FIRST_INSTANTIATION_AFTER_SPEC_FREEZE
2 = VERIFIED_PARENT_SEED_REUSE
3 = FRESH_VERSION_AFTER_COMPROMISE
```

CBOR：

```text
[
  1,
  0x3106,
  b"hegel-seed-continuity-manifest/1",
  continuity_status_id,
  split_spec_freeze_root,
  parent_seed_commitment_manifest_root_or_null,
  current_seed_commitment_manifest_root,
  parent_manifest_absence_attestation_root,
  hidden_access_ledger_genesis_root,
  custodian_binding_core_root,
  instantiated_at_unix_seconds,
  repository_commit_id
]
```

当前：

```text
continuity_status_id = 1
parent_seed_commitment_manifest_root = null
```

为避免 circular reference：

```text
CustodianBindingCoreV1
→ SeedContinuityManifestV1
→ final CustodianBindingManifestV1
```

## 4.6 DslShrinkTransitionFormalV1

```text
schema_id = hegel-dsl-shrink-transition-formal/1
hash_domain = HEGEL/DSL_SHRINK_TRANSITION/V1
```

```text
[
  1,
  0x3109,
  b"hegel-dsl-shrink-transition-formal/1",
  parent_dsl_spec_root,
  child_dsl_spec_root,
  parent_freeze_root,
  child_freeze_root,
  parent_execution_evidence_root,
  parent_status_id,
  shrink_step_id_digest,
  removed_registry_entry_root,
  surviving_registry_entry_root,
  tombstone_policy_root,
  cross_dsl_hash_policy_root,
  approval_manifest_root,
  outside_target_binding_manifest_root,
  null_control_binding_manifest_root,
  split_binding_manifest_root,
  custodian_binding_manifest_root,
  seed_continuity_manifest_root,
  shrink1_subset_replay_root,
  child_initial_state_id,
  created_at_unix_seconds,
  repository_commit_id
]
```

current：

```text
parent_status_id = DSL_TOO_LARGE
child_initial_state_id = NOT_RUN
```

## 4.7 M3ExecutionManifestV1

```text
schema_id = hegel-m3-execution-manifest/1
hash_domain = HEGEL/M3_EXECUTION_MANIFEST/V1
```

```text
[
  1,
  0x310A,
  b"hegel-m3-execution-manifest/1",
  run_id_16_bytes,
  child_dsl_spec_root,
  child_freeze_root,
  approval_manifest_root,
  shrink_transition_root,
  operator_semantics_root,
  identifier_registry_root,
  canonical_ast_schema_root,
  canonical_cbor_profile_root,
  diagnostic_formal_bridge_root,
  outside_target_binding_manifest_root,
  null_control_binding_manifest_root,
  split_binding_manifest_root,
  custodian_binding_manifest_root,
  seed_continuity_manifest_root,
  hidden_access_ledger_genesis_root,
  hidden_access_ledger_head_root,
  outside_target_universe_root,
  outside_target_truth_root,
  null_control_universe_root,
  null_control_truth_root,
  outside_discovery_split_root,
  outside_validation_split_root,
  outside_sealed_split_root,
  null_discovery_split_root,
  null_validation_split_root,
  null_sealed_split_root,
  canonical_program_budget,
  raw_operator_application_cap,
  records_per_chunk,
  equivalence_mode_id,
  python_implementation_binding_root,
  rust_implementation_binding_root,
  traversal_contract_root,
  bucket_accounting_contract_root,
  program_archive_contract_root,
  output_archive_contract_root,
  state_machine_contract_root,
  created_at_unix_seconds,
  repository_commit_id
]
```

guards：

```text
FAIL_M3_INPUT_ROOT_NULL
FAIL_M3_LEDGER_HEAD_NOT_GENESIS
FAIL_M3_RUN_ID_REUSED
FAIL_M3_BUDGET_CONTRACT_MISMATCH
FAIL_M3_ROLE_BINDING_MISMATCH
```

---

# 5. 问题 4：root 分类与 ordering

## 5.1 M3 pre-run input roots

必须在 `NOT_RUN -> RUNNING` 前非空并双实现一致：

```text
approval_manifest_root
shrink_transition_root
child_dsl_spec_root
child_freeze_root
operator_semantics_root
identifier_registry_root
canonical_ast_schema_root
canonical_cbor_profile_root

outside_target_binding_manifest_root
null_control_binding_manifest_root
split_seed_commitment_manifest_root
split_binding_manifest_root
custodian_binding_manifest_root
seed_continuity_manifest_root
parent_manifest_absence_attestation_root
hidden_access_ledger_genesis_root
hidden_access_ledger_head_root

diagnostic_formal_bridge_root
outside_target_universe_root
outside_target_truth_root
null_control_universe_root
null_control_truth_root
outside_discovery_split_root
outside_validation_split_root
outside_sealed_split_root
null_discovery_split_root
null_validation_split_root
null_sealed_split_root

python_implementation_binding_root
rust_implementation_binding_root
traversal_contract_root
bucket_accounting_contract_root
program_archive_contract_root
output_archive_contract_root
state_machine_contract_root
```

最后生成：

```text
m3_execution_manifest_root
```

## 5.2 Run-produced outputs

run 前必须为 null：

```text
canonical_program_archive_root
program_chunk_manifest_root
bucket_accounting_root

outside_program_output_archive_root
outside_output_chunk_manifest_root
outside_match_set_root
outside_role_evaluation_receipt_root

null_program_output_archive_root
null_output_chunk_manifest_root
null_match_set_root
null_role_evaluation_receipt_root

python_enumeration_receipt_root
rust_enumeration_receipt_root
dual_replay_agreement_root
final_state_record_root
```

预填则：

```text
FAIL_M3_OUTPUT_ROOT_PREPOPULATED
```

## 5.3 Row leaf schemas

### BoundedUniverseRowV1

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

ordering：

```text
universe_index ascending
```

### TargetTruthRowV1

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

ordering：

```text
universe_index ascending
```

### SplitAssignmentRowV1

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

ordering：

```text
(role_id, partition_id, universe_index) ascending
```

### IdentifierRegistryEntryV1

```text
[
  1,
  0x3204,
  b"hegel-identifier-registry-entry/1",
  registry_kind_id,
  numeric_id,
  entry_state_id,
  canonical_name_digest,
  semantics_digest_or_null,
  introduced_dsl_version_digest,
  removed_dsl_version_digest_or_null
]
```

ordering：

```text
(registry_kind_id, numeric_id) ascending
```

### OperatorSemanticsEntryV1

```text
[
  1,
  0x3205,
  b"hegel-operator-semantics-entry/1",
  operator_class_id,
  operator_id,
  admission_state_id,
  input_sort_ids,
  output_sort_id,
  undefined_semantics_id,
  normalization_rule_root_or_null,
  executable_semantics_root
]
```

ordering：

```text
(operator_class_id, operator_id) ascending
```

### DiagnosticFormalBridgeRecordV1

```text
[
  1,
  0x3206,
  b"hegel-diagnostic-formal-bridge-record/1",
  artifact_role_id,
  diagnostic_namespace_id,
  diagnostic_digest,
  formal_object_kind_id,
  formal_digest_or_root,
  row_count_or_null,
  diagnostic_profile_id_digest,
  formal_profile_id_digest,
  row_transform_spec_root,
  source_artifact_digest,
  repository_commit_id
]
```

ordering：

```text
(artifact_role_id, diagnostic_namespace_id, diagnostic_digest) ascending
```

## 5.4 Content roots

以下是 `ContentHash`，不是 RFC6962 row trees：

```text
dsl_spec_root
canonical_ast_schema_root
canonical_cbor_profile_root
traversal_contract_root
bucket_accounting_contract_root
program_archive_contract_root
output_archive_contract_root
state_machine_contract_root
```

---

# 6. Program archive audit amendment

## 6.1 决定

正式拆分：

1. target-independent canonical program archive；
2. role-specific program-output archive。

## 6.2 CanonicalProgramRecordV2

```text
[
  1,
  0x3207,
  b"hegel-canonical-program-record/2",
  program_index,
  canonical_ast_cbor_bytes,
  canonical_ast_hash,
  output_sort_id,
  ast_depth,
  ast_node_count,
  distinct_bit_slot_count,
  program_mdl_length_q32,
  child_dsl_spec_root,
  operator_semantics_root,
  identifier_registry_root
]
```

ordering：

```text
program_index ascending
```

root：

```text
canonical_program_archive_root
```

odd/sink 共享。

## 6.3 ProgramOutputRecordV2

```text
[
  1,
  0x3208,
  b"hegel-program-output-record/2",
  role_id,
  program_index,
  canonical_ast_hash,
  bounded_universe_root,
  operator_semantics_root,
  output_sort_id,
  row_count,
  output_blob_hash,
  undefined_bitmap_hash
]
```

ordering：

```text
program_index ascending
```

每 role 独立 root。

## 6.4 Chunk manifests

Program：

```text
[
  1,
  0x3209,
  b"hegel-program-chunk-manifest/2",
  chunk_index,
  first_program_index,
  last_program_index,
  record_count,
  canonical_program_record_subtree_root,
  compressed_program_blob_hash,
  uncompressed_program_byte_length
]
```

Role output：

```text
[
  1,
  0x320A,
  b"hegel-role-output-chunk-manifest/2",
  role_id,
  chunk_index,
  first_program_index,
  last_program_index,
  record_count,
  output_record_subtree_root,
  compressed_output_blob_hash,
  uncompressed_output_byte_length
]
```

ordering：

```text
chunk_index ascending
(role_id, chunk_index) ascending
```

## 6.5 MatchRecordV2

```text
[
  1,
  0x320B,
  b"hegel-match-record/2",
  role_id,
  canonical_ast_hash,
  output_blob_hash,
  target_truth_table_root
]
```

ordering：

```text
canonical_ast_hash bytes ascending
```

空 match set 使用 RFC6962 empty root。

---

# 7. 问题 5：M3 state machine

## 7.1 State enum

```text
0 = NOT_RUN
1 = RUNNING
2 = COMPLETE
3 = DSL_TOO_LARGE
4 = INCONCLUSIVE_BUDGET
5 = INCONCLUSIVE_SEMANTICS
6 = INCONCLUSIVE_EXECUTION
```

RUNNING phase：

```text
0 = NONE
1 = CANONICAL_ENUMERATION
2 = ROLE_EVALUATION
```

## 7.2 Legal transitions

```text
NOT_RUN/NONE
  -> RUNNING/CANONICAL_ENUMERATION

RUNNING/CANONICAL_ENUMERATION
  -> RUNNING/ROLE_EVALUATION
  -> DSL_TOO_LARGE/NONE
  -> INCONCLUSIVE_BUDGET/NONE
  -> INCONCLUSIVE_SEMANTICS/NONE
  -> INCONCLUSIVE_EXECUTION/NONE

RUNNING/ROLE_EVALUATION
  -> COMPLETE/NONE
  -> INCONCLUSIVE_SEMANTICS/NONE
  -> INCONCLUSIVE_EXECUTION/NONE
```

## 7.3 M3RunStateRecordV1

```text
[
  1,
  1,
  0x3301,
  b"hegel-m3-run-state-record/1",
  run_id_16_bytes,
  transition_index,
  previous_state_record_root_or_null,
  from_state_id,
  from_phase_id,
  to_state_id,
  to_phase_id,
  transition_reason_id,
  execution_manifest_root,
  triggering_receipt_root_or_null,
  recorded_at_unix_seconds
]
```

hash domain：

```text
HEGEL/M3_RUN_STATE_RECORD/V1
```

failure：

```text
FAIL_ILLEGAL_M3_STATE_TRANSITION
FAIL_M3_STATE_CHAIN_BREAK
FAIL_M3_TERMINAL_STATE_REOPEN
```

---

# 8. Implementation enumeration receipt

```text
schema_id = hegel-m3-implementation-enumeration-receipt/1
hash_domain = HEGEL/M3_IMPLEMENTATION_ENUMERATION_RECEIPT/V1
```

```text
[
  1,
  0x3302,
  b"hegel-m3-implementation-enumeration-receipt/1",
  implementation_id,
  run_id_16_bytes,
  execution_manifest_root,
  implementation_source_root,
  implementation_binary_digest,
  environment_image_digest,
  child_dsl_spec_root,
  operator_semantics_root,
  identifier_registry_root,
  canonical_ast_schema_root,
  canonical_cbor_profile_root,
  closure_status_id,
  raw_operator_application_count,
  canonical_program_count,
  closure_cardinality_or_null,
  frontier_exhausted,
  all_type_buckets_closed,
  raw_expansion_limit_hit,
  wall_clock_abort_hit,
  canonical_program_archive_root_or_null,
  program_chunk_manifest_root_or_null,
  bucket_accounting_root_or_null,
  first_out_of_budget_program_hash_or_null,
  partial_diagnostic_bundle_root_or_null,
  started_at_unix_seconds,
  finished_at_unix_seconds,
  process_exit_code
]
```

## COMPLETE

```yaml
canonical_program_count: <= 50000
closure_cardinality: equal to canonical_program_count
frontier_exhausted: true
all_type_buckets_closed: true
raw_expansion_limit_hit: false
wall_clock_abort_hit: false
canonical_program_archive_root: non-null
program_chunk_manifest_root: non-null
bucket_accounting_root: non-null
first_out_of_budget_program_hash: null
process_exit_code: 0
```

## DSL_TOO_LARGE

```yaml
canonical_program_count: 50000
closure_cardinality: null
frontier_exhausted: false
all_type_buckets_closed: false
raw_expansion_limit_hit: false
wall_clock_abort_hit: false
canonical_program_archive_root: non-null
program_chunk_manifest_root: non-null
bucket_accounting_root: non-null
first_out_of_budget_program_hash: non-null
process_exit_code: 0
```

## Inconclusive

formal archive roots必须为 null；可保留：

```text
partial_diagnostic_bundle_root
```

不得携带 role match verdict。

---

# 9. Role-specific evaluation receipt

```text
schema_id = hegel-m3-role-evaluation-receipt/1
hash_domain = HEGEL/M3_ROLE_EVALUATION_RECEIPT/V1
```

```text
[
  1,
  0x3303,
  b"hegel-m3-role-evaluation-receipt/1",
  implementation_id,
  role_id,
  run_id_16_bytes,
  execution_manifest_root,
  enumeration_receipt_root,
  canonical_program_archive_root,
  bounded_universe_root,
  target_truth_table_root,
  program_output_archive_root,
  role_output_chunk_manifest_root,
  match_set_count,
  match_set_root,
  undefined_program_count,
  evaluation_complete,
  started_at_unix_seconds,
  finished_at_unix_seconds,
  process_exit_code
]
```

只有 enumeration dual agreement 为 COMPLETE 后可生成。

failure：

```text
FAIL_ROLE_EVALUATION_BEFORE_COMPLETE_ENUMERATION
FAIL_ROLE_RECEIPT_ROOT_MISMATCH
FAIL_ROLE_OUTPUT_ARCHIVE_REUSED
```

---

# 10. Dual replay agreement

## 10.1 RoleAgreementEntry

```text
[
  1,
  0x3305,
  b"hegel-m3-role-agreement-entry/1",
  role_id,
  python_role_receipt_root,
  rust_role_receipt_root,
  bounded_universe_root,
  target_truth_table_root,
  program_output_archive_root,
  role_output_chunk_manifest_root,
  match_set_count,
  match_set_root,
  agreement
]
```

## 10.2 Agreement

```text
[
  1,
  0x3304,
  b"hegel-m3-dual-replay-agreement/1",
  run_id_16_bytes,
  execution_manifest_root,
  python_enumeration_receipt_root,
  rust_enumeration_receipt_root,
  agreed_closure_status_id,
  canonical_program_count_or_null,
  closure_cardinality_or_null,
  canonical_program_archive_root_or_null,
  program_chunk_manifest_root_or_null,
  bucket_accounting_root_or_null,
  first_out_of_budget_program_hash_or_null,
  [role_agreement_entries],
  enumeration_agreement,
  role_agreement_status_id,
  mismatch_record_root_or_null,
  created_at_unix_seconds
]
```

role status：

```text
0 = NOT_APPLICABLE
1 = AGREED
2 = DISAGREED
```

COMPLETE：

```text
enumeration_agreement = true
role_agreement_status = AGREED
2 role entries
mismatch root = null
```

DSL_TOO_LARGE：

```text
enumeration_agreement = true
role_agreement_status = NOT_APPLICABLE
role entries = []
mismatch root = null
```

跨实现不一致时不生成 authoritative agreement，进入：

```text
INCONCLUSIVE_SEMANTICS
```

---

# 11. Terminal guards

## COMPLETE

必须：

```text
Python/Rust enumeration COMPLETE
same program count/cardinality
same canonical program root
same program chunk root
same bucket root
odd role receipts agree
sink role receipts agree
odd output/match roots agree
sink output/match roots agree
dual agreement root non-null
```

状态迁移：

```text
RUNNING/ROLE_EVALUATION -> COMPLETE/NONE
```

## DSL_TOO_LARGE

必须：

```text
Python/Rust both DSL_TOO_LARGE
canonical_program_count = 50000
same first_out_of_budget_program_hash
same prefix program archive root
same prefix chunk root
same bucket root
closure_cardinality = null
role receipts absent
match roots absent
```

状态迁移：

```text
RUNNING/CANONICAL_ENUMERATION -> DSL_TOO_LARGE/NONE
```

---

# 12. Failure-code registry

| Code | Meaning |
|---|---|
| `FAIL_SPLIT_SEED_CSPRNG_UNAVAILABLE` | OS CSPRNG不可用 |
| `FAIL_SPLIT_SEED_ALREADY_INSTANTIATED` | 同版本重复实例化 |
| `FAIL_SPLIT_RANK_IDENTITY_COLLISION` | deterministic rank冲突 |
| `FAIL_SEED_CONTINUITY_UNVERIFIABLE` | continuity不完整 |
| `FAIL_HIDDEN_ACCESS_BEFORE_M3` | hidden access发生 |
| `FAIL_PARENT_BINDING_PROVENANCE_INCOMPLETE` | parent provenance不完整 |
| `FAIL_ROLE_ROOT_BINDING_MISMATCH` | odd/sink root错绑 |
| `FAIL_SPLIT_CUSTODIAN_BINDING_MISMATCH` | binding roots不一致 |
| `FAIL_M3_INPUT_ROOT_NULL` | required input root为空 |
| `FAIL_M3_OUTPUT_ROOT_PREPOPULATED` | output root被预填 |
| `FAIL_M3_LEDGER_HEAD_NOT_GENESIS` | ledger非genesis-only |
| `FAIL_M3_RUN_ID_REUSED` | run ID复用 |
| `FAIL_M3_BUDGET_CONTRACT_MISMATCH` | budget合同不符 |
| `FAIL_ILLEGAL_M3_STATE_TRANSITION` | 非法状态迁移 |
| `FAIL_M3_STATE_CHAIN_BREAK` | state chain断裂 |
| `FAIL_M3_TERMINAL_STATE_REOPEN` | 终态重开 |
| `FAIL_DUAL_REPLAY_MISMATCH` | Python/Rust不一致 |
| `FAIL_ROLE_EVALUATION_BEFORE_COMPLETE_ENUMERATION` | 提前role评价 |
| `FAIL_ROLE_RECEIPT_ROOT_MISMATCH` | role receipt错绑 |
| `FAIL_ROLE_OUTPUT_ARCHIVE_REUSED` | odd/sink archive复用 |
| `FAIL_RECEIPT_FIELD_PRESENCE` | nullability不符 |
| `FAIL_FORMAL_BRIDGE_MISMATCH` | bridge不一致 |

---

# 13. 剩余 10 个 M3 gates

| # | Gate |
|---:|---|
| 15 | `SPLIT_SEED_FIRST_INSTANTIATION_SIGNED` |
| 16 | `HIDDEN_ACCESS_LEDGER_GENESIS_ONLY` |
| 17 | `PARENT_MANIFEST_ABSENCE_ATTESTED` |
| 18 | `FORMAL_BINDING_MANIFESTS_CANONICALIZED` |
| 19 | `FORMAL_SPEC_AND_REGISTRY_ROOTS_DUAL_EQUAL` |
| 20 | `ODD_UNIVERSE_AND_TRUTH_ROOTS_DUAL_EQUAL` |
| 21 | `SINK_UNIVERSE_AND_TRUTH_ROOTS_DUAL_EQUAL` |
| 22 | `SPLIT_PARTITION_ROOTS_DUAL_EQUAL` |
| 23 | `M3_STATE_AND_RECEIPT_WIRE_GOLDEN_TESTS_PASS` |
| 24 | `M3_EXECUTION_MANIFEST_ROOT_NON_NULL_AND_OUTPUT_ROOTS_NULL` |

只有 24/24 才允许：

```text
NOT_RUN -> RUNNING/CANONICAL_ENUMERATION
```

---

# 14. M2.5 施工顺序

1. 创建 custodian key；
2. 第一次实例化 32-byte split seed；
3. 写 seed commitment；
4. 写 hidden-access ledger genesis；
5. 写 parent-manifest absence attestation；
6. 生成 split allocations与 partition roots；
7. 写 split/custodian/seed continuity manifests；
8. 写 odd/sink binding manifests；
9. formalize shrink transition；
10. Python/Rust 双重生成全部 roots；
11. 运行 formal-wire golden tests；
12. 创建 M3 execution manifest；
13. 检查 input roots non-null、output roots null、ledger genesis-only；
14. gates推进到 24/24；
15. 创建 `NOT_RUN -> RUNNING/CANONICAL_ENUMERATION` state record。

---

# 15. 当前允许的 claim

现在可以说：

> Python and Rust independently agree on all 23 shrink-1 strict vectors and on a 25,872-program constructive subset, with no rejection, collapse, or 50,001st witness. The child DSL remains `NOT_RUN` because formal split custody, binding roots, and M3 execution identities have not yet been instantiated.

M2.5 通过后可以说：

> The shrunk DSL has a replay-equal formal input identity, a first-instantiated sealed split commitment, a genesis-only hidden-access ledger, and a frozen M3 execution contract.

仍不能说：

- closure COMPLETE；
- target outside；
- null control inside；
- relation invention；
- certificate issued。

---

# 16. 最终判断

当前已经可以进入下一子阶段：

\[
\boxed{
\text{Phase-3A M2.5 Formal Commitment, Seed Genesis and Bridge Qualification}
}
\]

它不是形式主义绕路，而是在证明：

- split 是何时首次存在；
- 谁能访问 hidden artifacts；
- odd/sink 与 child DSL 的绑定不可互换；
- Python/Rust 将对同一 formal inputs 执行；
- later certificate 所签对象就是 run 开始时冻结的对象。

主线必须保持：

\[
\boxed{
\text{shrink subset qualification}
\rightarrow
\text{formal commitment/bridge}
\rightarrow
\text{M3 complete closure}
\rightarrow
\text{adequacy verdict}
\rightarrow
\text{synthesis}
}
\]

不能从 25,872 subset 直接跳到 `COMPLETE` 或新关系发明。
