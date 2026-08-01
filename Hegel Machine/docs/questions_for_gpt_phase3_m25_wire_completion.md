# Phase-3A M2.5 bit-exact wire completion 待决问题

**文档状态**：`DRAFT_FOR_NORMATIVE_COMPLETION`
**上位 amendment**：`hegel-freeze-p2b-p3-v1.1.1`
**child DSL**：`hegel-old-dsl-v1.1.0`
**当前机器状态**：`SHRINK1_SUBSET_QUALIFIED_M3_BLOCKED / NOT_RUN`

## 0. 已冻结且不再开放的证据

```yaml
strict_vector_count: 23
python_rust_strict_vectors_equal: true
shrink1_source_count: 25872
python_accepted_unique_count: 25872
rust_accepted_unique_count: 25872
accepted_set_commitment: "sha256:653fcb9428684cfed11c3f2345ac95ed98ded6e31564c9eeabf97c57ee71a7e9"
first_out_of_budget_witness: null
subset_is_complete_closure: false
closure_cardinality: null
formal_roots: null
m3_gates_satisfied: 14
m3_gates_total: 24
child_state: "NOT_RUN"
```

`Hegel_Machine_Phase3_Formal_Bridge_Seed_Genesis_M3_Wire_Freeze.md`
已冻结了主要 object tags、字段列表和 hash domains，但实现审计发现，
下列输入仍无法从现有文档唯一推导。它们会导致 Python 和 Rust 即使
各自“合理实现”，也可能得到不同的 canonical bytes/root。因此在完成下列决策前，
只能实现 wire skeleton 和 synthetic golden vectors，不能生成 authoritative M2.5 roots，
不能把 gate 15–24 报为通过。

---

## 1. 通用 byte-string digest 与 ID 转换规则仍未冻结

新 wire 使用了大量 `*_id_digest`，但没有定义从 machine ID text 到
32-byte digest 的唯一函数，包括：

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

请冻结：

```yaml
IdDigestV1:
  input_type: "ASCII machine-id text"
  normalization: "NONE | NFC | other"
  preimage: "exact byte formula"
  hash_domain: "exact ASCII domain"
  output: "32-byte byte string"
  nul_allowed: false
```

还请明确：

1. 已有 `sha256:<hex>` diagnostic value 进入 formal wire 时，是解码为 32 raw bytes，
   还是先对整个 ID text 做 `IdDigestV1`；
2. `<namespace>_<hex>` 是解码 suffix，还是对完整 human ID 做 digest；
3. Unix timestamp 的允许范围、是否允许 `0`，以及 Python/Rust 越界错误码；
4. 16-byte `run_id` / `ledger_id` 的生成 profile、重复检测持久化范围与错误码。

---

## 2. 缺失的 numeric enum registry

请为下列字段给出完整 numeric registry，包含保留值、未知值处置和
永不复用规则：

```text
InputSignatureId
SortId
RegistryKindId
RegistryEntryStateId
OperatorClassId
OperatorAdmissionStateId
UndefinedSemanticsId
ArtifactRoleId
DiagnosticNamespaceId
FormalObjectKindId
DiagnosticProfileId
FormalProfileId
StratumId
PartitionId
EquivalenceModeId
ImplementationId
ParentStatusId
ChildInitialStateId
M3TransitionReasonId
M3ClosureStatusId
M3StateId
M3RunningPhaseId
RoleAgreementStatusId
```

特别需要解决一个实际冲突：

- M3 state enum 已冻结为 `NOT_RUN=0, RUNNING=1, COMPLETE=2,
  DSL_TOO_LARGE=3, ...`；
- 旧 closure receipt enum 是 `NOT_RUN=0, COMPLETE=1, DSL_TOO_LARGE=2, ...`；
- `M3ImplementationEnumerationReceiptV1.closure_status_id` 未指定使用哪一套。

请明确 `M3StateId` 与 `M3ClosureStatusId` 是两个独立 registry，还是共享编号，
并给出 exact values。

---

## 3. 核心 ContentHash roots 的 canonical preimage 未定义

下列 root 被要求在 M3 entry 前 non-null，但文档只给了 root 名，没有
canonical object schema / numeric tag / domain：

```text
amendment_document_root
parent_freeze_root
child_freeze_root
child_dsl_spec_root
operator_semantics_root
identifier_registry_root
canonical_ast_schema_root
canonical_cbor_profile_root
split_contract_root
target_bundle_root
approval_evidence_root
replacement_policy_root
split_spec_freeze_root
removed_registry_entry_root
surviving_registry_entry_root
tombstone_policy_root
cross_dsl_hash_policy_root
fallback_registry_root
python_implementation_binding_root
rust_implementation_binding_root
traversal_contract_root
bucket_accounting_contract_root
program_archive_contract_root
output_archive_contract_root
state_machine_contract_root
row_transform_spec_root
```

请对每个 root 给出：

```yaml
root_name:
  object_name: "...V1"
  numeric_tag: 0x....
  schema_id_bytes: "b\".../1\""
  canonical_array: ["exact ordered fields"]
  hash_domain: "HEGEL/.../V1"
  nullable: false
  producer: "PUBLIC_BUILD | CUSTODIAN | AUDITOR | RUNNER"
```

若不希望继续增加大量 object tag，请明确哪些可以使用一个已冻结的
`ContentBlobV1 = [1, tag, schema_id, payload_bytes]`，但不得由实现者自行选择。

---

## 4. Odd/sink formal row 的 canonical input 仍不唯一

`BoundedUniverseRowV1` 包含 `input_signature_id` 和 `canonical_input_object`，
`TargetTruthRowV1` 依赖 `canonical_input_hash`，但尚未冻结：

1. odd row 是否编码为 `[set_size, [bits]]`、`[[bits]]`，还是 typed entity records；
2. sink row 是否编码为 `[a,b,c,d]`，是否包含 orientation / quantity / scope metadata；
3. odd 与 sink 的 `input_signature_id` exact numeric value；
4. `canonical_input_hash` 是普通 SHA-256 还是 `ContentHash`，若为后者则 exact domain；
5. `target_output` 使用 CBOR bool，还是 Bit uint `0/1`；
6. `formal_universe_root` / `formal_truth_root` 是否分别对排序后的
   480/85 records 直接做 RFC6962；
7. duplicate index、index gap、input-hash mismatch 的 exact failure codes。

请输出四个至少各2行的 exact CBOR hex golden vectors：

```text
odd universe row
odd truth row
sink universe row
sink truth row
```

---

## 5. Split allocation 尤其是 sink split 尚未闭合

### 5.1 Odd role

现有 odd quota 给出了 set size 5–8 的 discovery/validation/sealed 数量与
per-label 数量，但没有冻结：

- `stratum_id = f(set_size, target_label)` 的 numeric mapping；
- partition IDs；
- quota 是在每个 stratum 内分配，还是先全局 rank 后切分；
- 480 行是否必须恰好分配一次；
- fallback targets 是否共用 odd split。

### 5.2 Sink role

当前 sink universe 共 85 行。旧 freeze 只给出了下限：

```text
discovery_support_total >= 16
discovery_support_per_scale >= 8
validation_support_total >= 8
sealed_support_total >= 8
discovery each scale: d=0 >=4 and d>0 >=4
```

但 85-row generator 没有 `scale` 字段，也没有 exact partition quotas。
实现者无法知道：

- `scale` 到底指什么；
- strata 是 `d=0/d>0`、`d value 0..4`，还是其他 typed scale；
- discovery/validation/sealed 各多少行；
- 剩余 53 行是否仍须分配；
- sink split 是否也必须 exhaustive partition。

请给出 sink 的 exact stratum table 和每 cell quota，例如：

```yaml
sink_split_contract:
  exhaustive: true
  strata:
    - stratum_id: 0
      predicate: "exact machine predicate"
      universe_count: 0
      discovery_count: 0
      validation_count: 0
      sealed_count: 0
```

### 5.3 Split roots

请明确：

1. 六个 partition root 是对 `SplitAssignmentRowV1` 的 role-specific subset 分别做
   RFC6962，还是使用额外 partition-manifest object；
2. leaf ordering 在分开 root 内是 `universe_index` 还是仍包含 role/partition；
3. `target_bundle_root` 精确绑定 odd+sink 哪些 roots；
4. `split_contract_root` 精确绑定 quotas、strata、HKDF/HMAC profile 和 ordering 的哪个 object；
5. split assignment rows 是 public，还是 custodian-sealed，只公开 roots。

---

## 6. `CustodianBindingCoreV1` 未定义，现有依赖图无法执行

文档为解决循环引用了：

```text
CustodianBindingCoreV1
  -> SeedContinuityManifestV1
  -> CustodianBindingManifestV1
```

但 `CustodianBindingCoreV1` 没有 numeric tag、schema ID、fields 或 hash domain。
请给出完整定义，并输出整个 root DAG 的唯一 topological order，至少覆盖：

```text
split contract
target bundle
seed commitment
ledger genesis
custodian core
seed continuity
final custodian binding
split binding
odd role binding
sink role binding
approval
shrink transition
M3 execution manifest
initial state record
```

对每条 edge 请明确是绑定 object root、envelope root，还是 enclosed manifest root。

---

## 7. Custodian key / first seed genesis 必须有独立流程

请冻结可操作的独立 custodian 流程，不要只写“用 CSPRNG”：

```yaml
custodian_genesis:
  actor_eligibility: "who may act"
  key_algorithm: "Ed25519"
  private_key_generation_primitive: "exact allowed OS primitive/profile"
  private_key_storage: "outside repo; exact minimum controls"
  public_key_publication: "exact artifact"
  key_id_derivation: "exact formula"
  key_epoch_initial_value: 0
  split_seed_storage: "outside repo; exact minimum controls"
  second_invocation_guard: "persistent marker/root checked before CSPRNG"
  allowed_outputs: ["public key", "commitment", "signed envelopes", "roots"]
  forbidden_outputs: ["raw private key", "raw split seed", "K_role"]
```

请同时回答：

1. Codex 运行所在的本地机器是否可以充当 independent custodian，还是必须
   由用户/外部人员执行 one-shot command；
2. private key/seed 是否允许保存到 repo 之外、`0600` 权限的目录，
   还是必须 OS key store / hardware token；
3. seed 如何安全同时供 Python 和 Rust 独立计算 split：禁止 argv/env 后，
   是否允许一次性 inherited FD/stdin；
4. 二次运行如何在调用 CSPRNG 前返回
   `FAIL_SPLIT_SEED_ALREADY_INSTANTIATED`；
5. 哪些签名 envelope 可以公开 commit，哪些 split row payload 必须仅在
   custodian custody 中保留。

---

## 8. Auditor / parent absence attestation 的身份、审计根和签名未闭合

`ParentManifestAbsenceAttestationV1` 仍缺：

```text
audited_source_tree_root preimage
audited_path_set_root preimage
legacy_parent_payload_source_id_digest rule
auditor_key_id provenance
auditor signature requirement
```

请冻结：

1. auditor 必须独立于 custodian，还是允许同一 key/person；
2. 审计的 parent commit 是 `fb3a3ee4...`、`405ab525...`，还是其他精确 commit；
3. source tree 是 Git tree object ID、按 path+blob digest 构造的 RFC6962 root，还是
   ContentHash object；
4. audited path set 必须包含哪些目录、Git history 深度与 external artifact locations；
5. attestation 是否必须进入 `SignedManifestEnvelopeV1`，如果是，
   signature domain 是什么；
6. `absence_reason_bitmask=3` 之外是否还需要“无历史 seed realization”的独立 bit/
   attestation。

---

## 9. Approval manifest 与 Git commit 自引用问题

`NormativeApprovalManifestV1` 尚不能 bit-exact 生成，因为下列 preimage 未定：

```text
amendment_document_root
approval_evidence_root
approving_actor_id_digest
child_dsl_spec_root_or_null
repository_commit_id
```

请冻结：

1. Markdown document root 是 `ContentHash(domain, raw_file_bytes)`，还是规范化文本 object；
2. `approval_method_id=1` 时 approval evidence root 绑定用户消息、normative document，
   还是两者的 bundle；
3. `approving_actor_id_digest` 不声称 user cryptographic identity 时的 exact machine value；
4. 当 child DSL 已有 diagnostic spec 时，`child_dsl_spec_root_or_null` 必须 non-null 还是
   允许 null；
5. formal object 中的 `repository_commit_id` 是否必须指向“含规范和实现、
   但不含该 formal object”的 implementation commit。

建议请 GPT 明确 two-commit choreography：

```text
commit A: normative doc + deterministic implementation + golden vectors
external one-shot genesis/replay binds commit A
commit B: public manifests/roots/receipts only; secrets remain outside repo
```

否则“manifest 必须包含容纳自身的 Git commit”会形成不可解自引用。

---

## 10. Diagnostic JCS 与当前 `stable_hash` 的冲突

上位 v1.0.2 规范要求：

```text
diagnostic_digest =
SHA256(UTF8("HEGEL/DIAGNOSTIC_JCS/V1") || 0x00 || RFC8785_JCS_BYTES)
```

但当前 `hegel_machine.hashing.stable_hash` 实际计算：

```text
SHA256(json.dumps(sort_keys=True, separators=(",", ":"), ...))
```

它没有 `HEGEL/DIAGNOSTIC_JCS/V1` domain，也没有声称完整 RFC8785 实现。
现有 odd/sink diagnostic IDs 因此不能直接当作 normative JCS digest。

请三选一：

1. `LEGACY_DIAGNOSTIC_PROFILE` 方案：保留现有 IDs，在 bridge 中冻结独立
   legacy profile，formal rows 按新规范生成；
2. `REGENERATE_NORMATIVE_JCS_IDS` 方案：生成新 diagnostic payload/IDs，显式记录
   legacy→new bridge；
3. 声明当前 `stable_hash` 为该批已冻结 payload 的合法 profile，但必须给出
   exact profile ID 与不可用于新 objects 的范围。

还请冻结 `DiagnosticFormalBridgeRecordV1` 集合的 root 规则、record count、
required roles 和 exact `row_transform_spec_root` objects。

---

## 11. Bridge 签名与“M4 才做 final signatures”的边界

v1.0.2 要求 diagnostic→formal bridge 由 custodian + Python + Rust `3/3` 签名；
新 amendment 又说 final signatures/key chain 留在 M4，不阻塞 M3。

请明确 M2.5 gate 18–24 需要的是：

```text
A. 只需 Python/Rust root equality + custodian seed/ledger signatures
B. 需 bridge-specific 3/3 signatures，但不需 final certificate 3/3 signatures
C. bridge 签名也延后到 M4
```

若选 B，请给出 Python/Rust attester key 的 genesis、key ID、signature message domain、
envelope schema 和 key custody。不得用 implementation source root 冒充签名者身份。

---

## 12. 已列出 tag 但仍缺 schema 的 formal objects

object tag table 包含：

```text
BucketAccountingRecordV1 = 0x320C
```

但正文没有它的 array schema、ordering 或 root rules。另外，下列 M3 必要
objects 也只有 root 名而无 wire：

```text
PythonImplementationBindingV1
RustImplementationBindingV1
TraversalContractV1
BucketAccountingContractV1
ProgramArchiveContractV1
OutputArchiveContractV1
StateMachineContractV1
MismatchRecordV1
PartialDiagnosticBundleV1
```

请为它们给出 numeric tag、schema ID、array、domain/order/nullability。

`M3ImplementationEnumerationReceiptV1` 还需冻结：

- `implementation_id` exact type/value；
- `environment_image_digest` 是 32 raw bytes、OCI digest，还是其他 object root；
- `closure_status_id` enum；
- 两种 inconclusive status 之外，`INCONCLUSIVE_BUDGET` 的 exact 触发条件；
- `partial_diagnostic_bundle_root` 的 schema/domain；
- started/finished timestamp constraints；
- nonzero process exit code 与 formal failure state 的 mapping。

---

## 13. Archive/receipt 的剩余 bit-exact 缺口

请冻结：

1. `CanonicalProgramRecordV2.program_mdl_length_q32` 是否强制使用已冻结
   `hegel-mdl-prefix-v1.0.0`，以及不可计算时的 failure code；
2. compressed program/output blobs 的 codec、version、compression parameters、
   header/trailer 和 deterministic-byte 要求；
3. `canonical_program_archive_root` 是 ProgramRecord tree root，
   `program_chunk_manifest_root` 是 ChunkManifest tree root，两者不得互换；
4. role output root 对 `ProgramOutputRecordV2` 直接做 RFC6962，还是先组 chunk；
5. match-set empty root 与 nonempty ordering；
6. `undefined_bitmap_hash` 是否继承 `HEGEL/UNDEFINED_BITMAP/V1`；
7. hash collision 时必须比较哪些 canonical bytes，以及 exact abort code；
8. odd/sink output archive 必须独立，但若内容偶然完全相同时，
   是允许 root 数值相同，还是必须通过 role-specific domain/tag 强制不同。

---

## 14. `FAIL_M3_OUTPUT_ROOT_PREPOPULATED` 尚无可检查的 formal object

`M3ExecutionManifestV1` 的 array 只包含 pre-run input roots，并不包含任何
run-produced output root slot。但规范又要求 run 前检查所有 output roots 为 null，否则
`FAIL_M3_OUTPUT_ROOT_PREPOPULATED`。

请冻结哪个 object 承载这些 null slots：

```text
M3RunGenesisV1
M3RunPublicationV1
extended M3ExecutionManifestV1
or another exact object
```

并给出 tag/schema/domain。还请明确：

1. gate 24 通过后，M2.5 artifact 是仍保持 `NOT_RUN`，还是必须立即创建
   `NOT_RUN -> RUNNING/CANONICAL_ENUMERATION` state record；
2. “允许进入 RUNNING”与“已经进入 RUNNING”必须如何用不同状态表示；
3. run ID 在哪个时点生成并被持久化为不可复用；
4. M2.5 只做 formal qualification 时，是否不应偶发启动耗时的 complete closure run。

---

## 15. Hidden-access ledger 所谓“hidden artifact”的范围必须明确

当前 repository 已公开：

- odd relation 的 machine description；
- 480-row universe generator；
- `target_output = parity(bits)` 的生成逻辑；
- 85-row sink generator 与 truth predicate。

因此 `ledger genesis-only` 不可以诚实地声称“target relation/truth 从未被 Codex
或开发者看见”。可真正 sealed 的对象至少是第一次实例化后的
discovery/validation/sealed row allocation，可能还包含 role-evaluation outputs。

请冻结：

```yaml
hidden_artifact_scope:
  public_before_m3: ["exact list"]
  custodian_sealed_before_m3: ["exact list"]
  event_type_2_subjects: ["exact list"]
  event_type_4_subjects: ["exact list"]
  synthesis_agent_forbidden_inputs: ["exact list"]
```

若 sealed 只指 split assignments，请明确写出，避免把已公开 target 定义错报为
“从未访问”。

---

## 16. 外部 secret / actor 不存在时的强制 fail-closed 状态

请冻结一个可由 Python/Rust 共享的 M2.5 readiness failure registry，至少包含：

```text
FAIL_M25_WIRE_UNDERSPECIFIED
FAIL_ID_DIGEST_PROFILE_UNFROZEN
FAIL_FORMAL_ROOT_PREIMAGE_UNFROZEN
FAIL_SINK_SPLIT_CONTRACT_UNFROZEN
FAIL_CUSTODIAN_KEY_MISSING
FAIL_SPLIT_SEED_NOT_INSTANTIATED
FAIL_CUSTODIAN_SIGNATURE_MISSING
FAIL_AUDITOR_ATTESTATION_MISSING
FAIL_HIDDEN_ARTIFACT_SCOPE_UNFROZEN
FAIL_DIAGNOSTIC_PROFILE_MISMATCH
FAIL_FORMAL_GOLDEN_VECTOR_MISMATCH
FAIL_M3_OUTPUT_SLOT_WIRE_UNFROZEN
```

请明确哪些属于：

```text
BLOCKED_SPECIFICATION
BLOCKED_EXTERNAL_CUSTODY
INCONCLUSIVE_SEMANTICS
INCONCLUSIVE_EXECUTION
```

当前不得用 synthetic test key/seed 填充 authoritative artifact，也不得把未签名 JSON
写成 gate passed。

---

## 17. v2/GSCL 负结果对 Hegel Machine 的规范性影响

与 Hegel Machine 同仓库的 commit
`4861b2d88ef7e85fb62f32e3d2e1f5c78afe9529` 记录了协议有效的 GSCL/SCAR
负结果：

```yaml
effect_status: "PROTOCOL_VALID_NEGATIVE"
primary_delta_pair_f1: -0.6728865211
primary_ci95: [-0.7095084627, -0.6358874857]
semantic_only_coverage: 1.0
semantic_only_f1: 0.677030
full_structural_composition_coverage: 0.006906
full_structural_composition_f1: 0.004144
diagnosed_risk: "HARD_STRUCTURAL_ELIGIBILITY_COVERAGE_COLLAPSE"
```

这个结果不是 DSL closure 证明，也不逻辑否定“新 relation 可被发明”；
但它否定了已冻结 SCAR hard-selector operationalization 在该 cohort 上优于
semantic-only 的经验声称，对 Phase-3B 的候选生成/选择有直接的设计风险。

请 GPT 冻结下列建议状态是否正确：

```yaml
upstream_v2_counterevidence:
  evidence_commit: "4861b2d88ef7e85fb62f32e3d2e1f5c78afe9529"
  evidence_status: "PROTOCOL_VALID_NEGATIVE"
  m25_formal_wire_gate_effect: "NONE"
  m3_old_dsl_closure_gate_effect: "NONE"
  outside_certificate_effect: "NONE"
  phase3b_design_risk: "HARD_STRUCTURAL_ELIGIBILITY_COVERAGE_COLLAPSE"
  transfer_existing_v2_weights_or_thresholds: false
  treat_v2_priors_as_verified_positive_priors: false
  require_shadow_only_ablation: true
```

请进一步冻结 Phase-3B 首个 adequacy/synthesis experiment 的 control arms，建议至少包含：

```text
A = NO_PRIOR / old-language baseline
B = FROZEN_V2_PRIOR_CONTROL, no retuning after negative result
C = HEGEL_INVENTED_RELATION, untrusted recognizer + sealed evaluator
D = SEMANTIC_ONLY_CONTROL
```

对 `WikiSQL UAO P4 same-v5` 请二选一：

```text
1. 只作为不同 task/domain 的历史参考，不做直接数值对照
2. 在新 protocol 中定义可比的独立 reference arm
```

请明确不得：

- 因 v2 失败而改变 M2.5 formal bytes/root；
- 因 v2 失败而改变已冻结 odd/sink target、split seed 或 M3 closure budget；
- 将 v2 负结果写成 Hegel Machine 已失败；
- 将 v2 的 13/22 条 prior 写成已被验证的正先验。

---

## 18. 请 GPT 输出的形式

请返回一份可直接作为 normative amendment 的 machine-readable Markdown，
不要只给原则性建议。必须包含：

1. 所有新增/changed object 的 numeric tag；
2. exact numeric-array schema 和 schema ID byte string；
3. exact ContentHash domain 或 RFC6962 leaf/root rule；
4. 所有 enum 数值、reserved range 和 tombstone policy；
5. root dependency DAG 与 topological construction order；
6. odd/sink exact split strata/quota table；
7. custodian/auditor 独立流程、signature wires 与 public/sealed artifact boundary；
8. Python/Rust 至少一组共享 golden vectors，包含 exact CBOR hex 和 roots；
9. M3 gate 15–24 每一项的唯一 pass predicate 和 failure code；
10. v2 counterevidence 对 M2.5/M3/Phase-3B 的 exact status 与 control-arm 决策。

在这份 completion amendment 落地前，建议机器保持：

```yaml
m25_wire_implementation: "PARTIAL_ALLOWED"
authoritative_formal_root_generation: false
split_seed_first_instantiation: false
custodian_signature_claim: false
m3_gates_satisfied: 14
m3_gates_total: 24
m3_entry_allowed: false
child_state: "NOT_RUN"
```
