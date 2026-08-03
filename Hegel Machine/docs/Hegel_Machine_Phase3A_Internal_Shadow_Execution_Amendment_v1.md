# Hegel Machine Phase-3A Internal Shadow Execution Amendment v1

**Document type**: owner-authorized internal-research execution amendment
**document ID**: `hegel-phase3a-internal-shadow-execution-amendment-v1`
**shadow track ID**: `hegel-internal-shadow-v1`
**formal machine freeze ID**: `hegel-freeze-p2b-p3-v1.1.2`（不升版）
**formal child DSL ID**: `hegel-old-dsl-v1.1.0`（不升版）
**status**: `INTERNAL_EXECUTION_AUTHORIZED_FORMAL_AUTHORITY_UNCHANGED`

本 amendment 冻结一条与 formal M2.5/M3 完全分离的 internal-shadow 执行轨道。它解决的
是研发不能因暂时缺少 external custodian / attesters / auditor 而永久停止的问题；它不把
Codex、subagent、本机 namespace、临时 key 或同一项目控制者描述成外部独立主体。

本文件受以下既有规范约束：

1. `Hegel_Machine_Phase3A_M25_Bit_Exact_Wire_Completion_Amendment.md`；
2. `Hegel_Machine_Phase3A_M25_Exact_Wire_Errata_Resolution.md`；
3. `Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md`；
4. `phase3_m25_external_genesis_operator_runbook.md`。

发生冲突时，本文件只决定 internal-shadow 行为；它无权修改上述文件的 formal wire、
external actor eligibility、Gate 15–24 或 certificate 条件。

---

## 1. 双轨状态与唯一允许声明

### 1.1 Formal 轨道保持不变

创建、执行或发布任何 shadow artifact 前后，以下 invariants 都必须成立：

```yaml
formal_track:
  freeze_id: hegel-freeze-p2b-p3-v1.1.2
  child_dsl_id: hegel-old-dsl-v1.1.0
  gates_satisfied: 14
  gates_total: 24
  m3_state_id: 0
  m3_state_name: NOT_RUN
  m3_entry_qualified: false
  m3_entry_allowed: false
  m3_run_started: false
  formal_roots: null
  formal_seed_first_instantiated: false
  external_actor_evidence: false
  outside_certificate_allowed: false
  mdl_certificate_allowed: false
  active_promotion_allowed: false
```

Shadow runner 必须在每份状态报告中同时输出该 formal snapshot。只输出 shadow 状态而省略
formal snapshot 属于 `FAIL_SHADOW_FORMAL_STATUS_OMITTED`。

### 1.2 Shadow 轨道的唯一 artifact kind

```yaml
ShadowArtifactKindId:
  0: INVALID
  1: INTERNAL_PURPOSE_SEPARATED_NON_AUTHORITATIVE
```

所有 shadow objects、JSON reports、receipts、envelopes 和 terminal summaries 必须同时含：

```yaml
artifact_kind_id: 1
artifact_kind: INTERNAL_PURPOSE_SEPARATED_NON_AUTHORITATIVE
external_independence_claim: false
formal_evidence_claim: false
```

字符串和 numeric ID 任一不匹配即返回 `FAIL_SHADOW_ARTIFACT_KIND`。禁止使用：

```text
EXTERNAL
INDEPENDENT_ACTOR
AUTHORITATIVE
FORMAL_ROOT
OUTSIDE_LANGUAGE
OUTSIDE_FROZEN_CLOSURE
CERTIFIED
ACTIVE
```

描述 shadow artifact。允许的最强声明是：

> 在一个由同一项目控制者编排、purpose-separated、进程隔离且非权威的本机环境中，
> 指定 Commit snapshot 的 frozen closure mechanics 得到了可重放的 candidate outcome。

### 1.3 禁止使用“internal 24/24”

Shadow admission 使用独立的 `12/12` registry。不得写成 `24/24-INTERNAL`、不得复用
formal Gate 15–24 的名称或 pass bit。状态展示必须采用：

```yaml
formal: "14/24 / NOT_RUN"
shadow: "<n>/12 / <ShadowStateId name>"
```

---

## 2. Canonical encoding、tag namespace 与 domain separation

### 2.1 Shadow-only tag namespace

Shadow tags 为 document-local registry，范围严格为 `0x7A00..0x7AFF`。v1 分配：

| Tag | Object |
|---:|---|
| `0x7A00` | `ShadowPolicyBindingV1` |
| `0x7A01` | `ShadowPurposeWorkerManifestV1` |
| `0x7A02` | `ShadowIsolationManifestV1` |
| `0x7A03` | `ShadowAdmissionReceiptV1` |
| `0x7A04` | `ShadowEnvelopeV1` |
| `0x7A05` | `ShadowRunGenesisV1` |
| `0x7A06` | `ShadowStateRecordV1` |
| `0x7A07` | `ShadowEnumerationReceiptV1` |
| `0x7A08` | `ShadowRoleEvaluationReceiptV1` |
| `0x7A09` | `ShadowDualReplayAgreementV1` |
| `0x7A0A` | `ShadowDisclosureLedgerRecordV1` |
| `0x7A0B` | `ShadowExecutionBundleV1` |
| `0x7A0C` | `ShadowIsolationPlanV1` |
| `0x7A0D` | `ShadowSecurityProbeReceiptV1` |

这些 tags：

- 不得加入 formal `phase3_m25_wire_v1.OBJECT_TAGS`；
- 不得被 formal decoder 接受；formal decoder 必须返回 `REJECT_UNKNOWN_M25_SCHEMA`；
- shadow decoder 遇到 `0x3000..0x34FF` 或 `0x31FF` 必须返回
  `REJECT_SHADOW_FORMAL_TAG_NAMESPACE`；
- 不得通过 tag alias、schema alias 或 profile flag 在两种 decoder 间转换。

### 2.2 Encoding profile

Shadow hashed core 复用 `hegel-cbor-det-v1` 的 canonical numeric-array encoding 限制：

- definite-length arrays / byte strings；
- integers 使用最短编码；
- 允许 `null` 和 strict boolean；
- 禁止 CBOR map、text、float、tag 和 indefinite encoding；
- object prefix 严格为 `[1, numeric_tag, ascii_schema_id_bytes]`。

复用编码器不构成 formal identity。Shadow identity 只由本节的独立 domain 产生。

### 2.3 ShadowDigestV1

所有 32-byte shadow hash identity 字段名以 `_digest` 结尾；shadow wire 和 public JSON 中
禁止把它们命名为 `_root`。

```text
ShadowDigestV1(domain, value) =
SHA256(
  UTF8("HEGEL/INTERNAL_SHADOW/NON_AUTHORITATIVE/")
  || UTF8(domain)
  || UTF8("/V1")
  || 0x00
  || CanonicalCBOR(value)
)
```

`domain` 只能是下表的 ASCII token：

```yaml
shadow_hash_domains:
  POLICY_BINDING: ShadowPolicyBindingV1
  PURPOSE_WORKER: ShadowPurposeWorkerManifestV1
  ISOLATION_MANIFEST: ShadowIsolationManifestV1
  ADMISSION_RECEIPT: ShadowAdmissionReceiptV1
  ENVELOPE: ShadowEnvelopeV1
  RUN_GENESIS: ShadowRunGenesisV1
  STATE_RECORD: ShadowStateRecordV1
  ENUMERATION_RECEIPT: ShadowEnumerationReceiptV1
  ROLE_EVALUATION_RECEIPT: ShadowRoleEvaluationReceiptV1
  DUAL_REPLAY_AGREEMENT: ShadowDualReplayAgreementV1
  DISCLOSURE_LEDGER_RECORD: ShadowDisclosureLedgerRecordV1
  EXECUTION_BUNDLE: ShadowExecutionBundleV1
  ISOLATION_PLAN: ShadowIsolationPlanV1
  SECURITY_PROBE_RECEIPT: ShadowSecurityProbeReceiptV1
```

任何 domain 不含固定 prefix、与 formal ContentHash domain 相同或大小写不精确，返回
`FAIL_SHADOW_DOMAIN_COLLISION`。

### 2.4 ShadowTreeDigestV1

有序 record collection 不使用 formal RFC6962 root。其独立算法为：

```text
prefix = UTF8("HEGEL/INTERNAL_SHADOW/NON_AUTHORITATIVE/TREE/")
         || UTF8(domain)
         || UTF8("/V1")
         || 0x00

empty_digest = SHA256(prefix || 0x02)
leaf_digest  = SHA256(prefix || 0x00 || CanonicalCBOR(record))
node_digest  = SHA256(prefix || 0x01 || left_digest || right_digest)
```

非空 tree 使用与 RFC6962 相同的 largest-power-of-two split shape，但 leaf/node preimage
不同，因此其 digest 不能成为 formal RFC6962 root。`domain` 必须显式包含 collection
role，例如 `CANONICAL_PROGRAM_ARCHIVE`、`ODD_OUTPUT_ARCHIVE`、`SINK_MATCH_SET`。

---

## 3. Numeric registries

### 3.1 ShadowPurposeId

| ID | Name | Internal responsibility |
|---:|---|---|
| 0 | `INVALID` | rejected |
| 1 | `SHADOW_CUSTODIAN_AND_SPLIT_COORDINATOR` | ephemeral shadow seed/key；FD-3 calculators；sealed assignment routing |
| 2 | `SHADOW_PYTHON_REPLAY_WORKER` | Python independent replay and candidate receipt |
| 3 | `SHADOW_RUST_REPLAY_WORKER` | Rust independent replay and candidate receipt |
| 4 | `SHADOW_POLICY_AUDIT_WORKER` | isolation/claim/secret/output audit and adversarial checks |
| 5–32767 | reserved | rejected in v1 |

这些 purpose 是 process responsibilities，不是 external trust identities。Subagent 名称、
模型名称、PID、key label 或不同 prompt 都不能把 `external_independence_claim` 改为 true。

### 3.2 ShadowStateId

| ID | Name | Terminal |
|---:|---|---|
| 0 | `NOT_ADMITTED` | no |
| 1 | `ADMITTED_NOT_STARTED` | no |
| 2 | `RUNNING_CANONICAL_ENUMERATION` | no |
| 3 | `RUNNING_ROLE_EVALUATION` | no |
| 4 | `COMPLETE_CANDIDATE` | yes |
| 5 | `DSL_TOO_LARGE_CANDIDATE` | yes |
| 6 | `INCONCLUSIVE_BUDGET` | yes |
| 7 | `INCONCLUSIVE_SEMANTICS` | yes |
| 8 | `INCONCLUSIVE_EXECUTION` | yes |
| 9 | `ABORTED_POLICY_VIOLATION` | yes |
| 10 | `ABORTED_OPERATOR` | yes |
| 11–32767 | reserved | rejected |

### 3.3 ShadowTransitionReasonId

| ID | Name |
|---:|---|
| 0 | `INVALID` |
| 1 | `SHADOW_ADMISSION_GATES_12_OF_12` |
| 2 | `EXPLICIT_SHADOW_START` |
| 3 | `CANONICAL_FRONTIER_CLOSED` |
| 4 | `SYNTACTIC_CAPACITY_WITNESS_50001` |
| 5 | `SEARCH_BUDGET_HIT` |
| 6 | `SEMANTICS_OR_DUAL_REPLAY_MISMATCH` |
| 7 | `EXECUTION_FAILURE` |
| 8 | `ROLE_EVALUATION_COMPLETE` |
| 9 | `POLICY_VIOLATION` |
| 10 | `EXPLICIT_OPERATOR_ABORT` |

### 3.4 ShadowOutcomeId

| ID | Name |
|---:|---|
| 0 | `NOT_RUN` |
| 1 | `COMPLETE_CANDIDATE` |
| 2 | `DSL_TOO_LARGE_CANDIDATE` |
| 3 | `INCONCLUSIVE_BUDGET` |
| 4 | `INCONCLUSIVE_SEMANTICS` |
| 5 | `INCONCLUSIVE_EXECUTION` |
| 6 | `ABORTED_POLICY_VIOLATION` |
| 7 | `ABORTED_OPERATOR` |

ShadowOutcomeId 不得 decode 为 `M3ClosureStatusId`。

### 3.5 ShadowDisclosureEventTypeId

| ID | Name |
|---:|---|
| 0 | `INVALID` |
| 1 | `LEDGER_GENESIS` |
| 2 | `SEALED_ASSIGNMENT_DELIVERED_TO_ROLE_EVALUATOR` |
| 3 | `TERMINAL_ROLE_SUMMARY_REVEALED_TO_ORCHESTRATOR` |
| 4 | `FORBIDDEN_ARTIFACT_EXPOSED_TO_SYNTHESIS` |
| 5 | `PUBLIC_SHADOW_ARTIFACT_PUBLISHED` |

---

## 4. Four-purpose isolation profile

### 4.1 Exact isolation claim

本 track 的 isolation class 唯一为：

```yaml
ShadowIsolationProfileId:
  1: LOCAL_NAMESPACE_PURPOSE_SEPARATION_V1

claim:
  process_and_artifact_separation: true
  implementation_diversity_python_rust: true
  external_actor_independence: false
  organizational_independence: false
  adversarial_controller_independence: false
```

它证明“编排器按 policy 分离了输入、写路径、进程和临时 keys”，不证明“编排器无法读取或
串通”。同一 Unix owner、同一 host、同一根 agent 和共享项目控制权必须在 receipt 中
明确记录。

### 4.2 Required runtime invariants

四个 purpose 必须分别在不同 process 和不同 Linux namespace set 中运行。Admission 先用
四个不含 key/seed 的 probe workers 验证能力；explicit start 的真实 workers 必须重新验证
同一 profile。Probe identity 不得冒充 runtime worker identity。以下 18 个 predicate 必须全部
为 strict boolean `true`，count 字段必须精确匹配：

```yaml
local_namespace_purpose_separation_v1:
  purpose_count: 4
  purpose_ids_exactly: [1, 2, 3, 4]
  distinct_worker_instance_ids: true
  distinct_process_ids: true
  distinct_mount_namespaces: true
  distinct_pid_namespaces: true
  distinct_network_namespaces: true
  distinct_ipc_namespaces: true
  user_namespace_enabled: true
  no_new_privileges: true
  effective_capability_count_zero: true
  network_interfaces_loopback_only_and_down: true
  live_repository_mount_count_zero: true
  writable_cross_purpose_mount_count_zero: true
  basis_snapshot_read_only: true
  purpose_private_tmpfs: true
  purpose_private_home: true
  environment_allowlist_applied: true
  umask_0077: true
  core_dump_disabled: true
```

`bubblewrap`、`unshare` 或同等机制可以实现这些 observable invariants；工具名称本身不是
证据。任一 invariant 无法验证时不得弱化 profile，返回
`FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE`。

`isolation_invariant_bitset=0x3FFFF` 只编码以上 18 个 namespace/filesystem/process
predicates。Seccomp live probe 使用独立的 `required_security_probe_digest` 绑定，不得占用
第 19 bit 或改变该 bitset 的 v1 身份。

### 4.3 Mandatory seccomp live probe and Landlock disclosure

Gate 8 除 18-bit namespace profile 外，还必须在四个 admission probe workers 中分别执行
真实 syscall probe；只检查 launcher arguments、binary strings 或 host policy 文件不算通过。
四份 `ShadowSecurityProbeReceiptV1` 按 purpose `[1,2,3,4]` 排序后，以
`ShadowTreeDigestV1("SECURITY_PROBE_SET", rows)` 形成 admission
`required_security_probe_digest`。

每个 receipt 必须证明：

```yaml
required_seccomp_live_probe:
  proc_status_seccomp_value: 2
  proc_status_no_new_privs_value: 1
  attack_syscall_errno_required: 1
  attack_syscall_errno_name: EPERM
  attack_syscalls_exactly:
    1: SOCKET_AF_INET_STREAM
    2: SOCKET_AF_INET6_STREAM
    3: MOUNT
    4: PTRACE_TRACEME
    5: BPF_MAP_CREATE
    6: PERF_EVENT_OPEN
```

六个 syscall 必须真实调用并全部返回 `-1`、Linux `errno=1 (EPERM)`。未调用、返回成功、
被参数校验以其他 errno 拒绝、`Seccomp != 2` 或 `NoNewPrivs != 1` 均 fail closed。Explicit
start 后四个真实 workers 必须再次运行 phase-2 probe；runtime probe set 满足同一 predicates，
但因 worker identity/timestamp 不同，不要求其 digest 等于 admission probe set digest。

Landlock 在 v1 是透明的 nonblocking hardening gap，而不是 Gate 8 pass predicate：

```yaml
ShadowLandlockStatusId:
  0: NOT_PROBED
  1: ENFORCED
  2: UNAVAILABLE
  3: PARTIAL

gate_effect:
  ENFORCED: PASS_WITHOUT_GAP
  NOT_PROBED: PASS_ONLY_WITH_NONBLOCKING_GAP_DISCLOSED
  UNAVAILABLE: PASS_ONLY_WITH_NONBLOCKING_GAP_DISCLOSED
  PARTIAL: PASS_ONLY_WITH_NONBLOCKING_GAP_DISCLOSED
```

当 status 不为 `ENFORCED` 时，receipt 必须设置
`landlock_nonblocking_gap_disclosed=true`，terminal summary 必须保留 exact gap code
`HARDENING_GAP_LANDLOCK_NOT_ENFORCED_NONBLOCKING`。该 gap 不允许弱化 seccomp、read-only
snapshot、no-network namespace 或 cross-purpose write isolation。

本 amendment 同时登记施工前已发生的 transient capability probe incident：

```yaml
known_diagnostic_incident:
  incident_id: TRANSIENT_CAPABILITY_PROBE_DURING_SHADOW_PREFLIGHT_V1
  classification: PRE_ADMISSION_NON_AUTHORITATIVE_DIAGNOSTIC
  disposition: SUPERSEDED_BY_REQUIRED_SECCOMP_LIVE_PROBE
  formal_state_effect: NONE
  formal_gate_effect: NONE
  shadow_state_transition: NONE
  shadow_admission_credit: NONE
  external_evidence_effect: NONE
```

该 incident 只说明隔离能力探索中出现过一次 transient capability probe；它不是通过或失败
的 admission evidence，也不产生 run ID、key、seed、formal root 或 state record。第一次
qualifying shadow admission probe set 必须在每个 receipt 中以 count/digest 透明绑定该
incident；此后 fresh run 只需保留同一 incident collection digest，不重复计数为新事件。它
不改变 formal `14/24 / NOT_RUN`，也不把 shadow 从 `NOT_ADMITTED` 推进到其他 state。

### 4.4 Read-only basis Commit snapshot

Shadow run 必须绑定一个已提交的 40-hex SHA-1 `basis_commit_id`：

1. basis commit 必须是当前 repository HEAD 的 ancestor；
2. 从 Git objects materialize 私有 detached snapshot；
3. snapshot 不含 `.git`、live worktree bind mount 或 uncommitted bytes；
4. 所有 bound source、tests、golden、DSL/freeze docs 和 dependency locks 都进入
   `snapshot_manifest_digest`；
5. 四个 worker 只读挂载同一 bytes-equivalent snapshot；
6. 每个 worker 的 snapshot manifest 必须相同；
7. start 前、每个 phase 后和 terminal packaging 前重新计算；
8. pre/post digest 不同即进入 `ABORTED_POLICY_VIOLATION`。

Worker 可以拥有 purpose-private tmpfs，但不得拥有 snapshot 内写权限。任何 live worktree
path 可见即返回 `FAIL_SHADOW_LIVE_WORKTREE_VISIBLE`。

### 4.5 Ephemeral keys and seed

只有达到 shadow admission `12/12` 后，四个 worker 才能分别调用 OS CSPRNG：

- 每个 purpose 生成一把临时 Ed25519 key；
- purpose 1 另行生成一个 32-byte `shadow_split_seed`；
- 五次 CSPRNG 调用相互独立；
- 四个 public keys 与 key IDs 两两不同；
- key epoch 严格为 `0`；
- key ID = `first_16_bytes(SHA256(raw_32_byte_public_key))`；
- private keys 和 raw seed 只存在于 purpose-private tmpfs / locked memory；
- 不写 repo、snapshot、argv、env、stdin、stdout、stderr 或 report；
- terminal 后 zeroize，关闭 namespace，销毁 tmpfs；
- crash 后不得将残留 bytes 收集进 artifact。

Shadow seed 的生成不属于 formal split seed first instantiation。它不得使用 formal marker、
formal seed domain、formal custodian schema或 formal key manifest。它也不能使 formal Gate 15
通过。

### 4.6 FD channel registry

| FD | Name | Producer -> consumer | Payload |
|---:|---|---|---|
| 3 | `SHADOW_SEED_INPUT` | purpose 1 -> its Python/Rust split calculators | exactly 32 raw bytes then EOF |
| 4 | `SHADOW_SEALED_ASSIGNMENT_INPUT` | purpose 1 -> role-evaluation endpoint | one `uint64_be(length) || canonical_cbor(payload)` frame then EOF |
| 5 | `SHADOW_PUBLIC_EVIDENCE_OUTPUT` | each worker -> orchestrator | one allowlisted canonical-CBOR frame then EOF |

Rules：

- FD 3 只在 purpose-1 child calculators 中存在；purpose 2/3/4 和 synthesis process 不得
  继承；
- FD 4 只能在 state 3 `RUNNING_ROLE_EVALUATION` 打开，payload 不含 raw seed/private key，
  synthesis process 不得继承；
- FD 5 payload 必须先通过 recursive secret-field、private-key-header、row-membership 和
  artifact-kind lint；
- stdin 关闭；stdout/stderr 必须为空，或只含固定 failure code；
- 不得用 argv/env/stdin/file fallback 替代 FD 3/4；
- unexpected inherited FD count 必须为零。

违反 channel policy 立即返回 `FAIL_SHADOW_SECRET_CHANNEL_POLICY`，并进入
`ABORTED_POLICY_VIOLATION`。

### 4.7 Temporary shadow signatures

四 purpose 的 signature preimage 严格为：

```text
UTF8("HEGEL/INTERNAL_SHADOW/NON_AUTHORITATIVE/SIGNATURE/V1")
|| 0x00
|| shadow_object_digest_32_bytes
|| uint16_be(shadow_purpose_id)
|| uint64_be(key_epoch)
|| shadow_run_id_16_bytes
```

它们只能进入 `ShadowEnvelopeV1`，不能进入 `SignedManifestEnvelopeV1` 或
`AttestationBundleV1`。它们证明一个 ephemeral key 对 shadow digest 的一致性，不证明
签名主体独立、真实身份或 external custody。

---

## 5. Exact shadow schemas

所有 ID byte lengths：`shadow_run_id=16`、`worker_instance_id=16`、`key_id=16`、digest=32、
Ed25519 public key=32、Ed25519 signature=64。Git ID 使用 `[1, sha1_20_bytes]`。

### 5.1 `ShadowPolicyBindingV1` — `0x7A00`

```text
[
  1, 0x7A00, b"hegel-internal-shadow-policy-binding/1",
  1,
  b"hegel-internal-shadow-v1",
  b"hegel-freeze-p2b-p3-v1.1.2",
  b"hegel-old-dsl-v1.1.0",
  amendment_git_blob_sha256,
  basis_commit_id
]
```

### 5.2 `ShadowPurposeWorkerManifestV1` — `0x7A01`

```text
[
  1, 0x7A01, b"hegel-internal-shadow-purpose-worker/1",
  1,
  shadow_run_id,
  shadow_purpose_id,
  worker_instance_id,
  1,
  basis_commit_id,
  snapshot_manifest_digest,
  executable_manifest_digest,
  environment_manifest_digest,
  namespace_manifest_digest,
  ephemeral_key_id,
  ephemeral_public_key,
  0,
  false
]
```

最后一个字段是 `external_independence_claim`，v1 只接受 `false`。

### 5.3 `ShadowIsolationManifestV1` — `0x7A02`

```text
[
  1, 0x7A02, b"hegel-internal-shadow-isolation-manifest/1",
  1,
  shadow_run_id,
  basis_commit_id,
  snapshot_manifest_digest,
  [purpose_1_worker_digest, purpose_2_worker_digest,
   purpose_3_worker_digest, purpose_4_worker_digest],
  isolation_invariant_bitset,
  required_security_probe_digest,
  fd_policy_digest,
  output_allowlist_digest,
  secret_lint_policy_digest,
  created_at_unix_seconds,
  false
]
```

Worker digests 的顺序严格为 purpose `[1,2,3,4]`。`isolation_invariant_bitset` 的低 18 bits
对应 §4.2 的 18 个 boolean，必须严格等于 `0x3FFFF`；更高 bits 必须为零。

这是 explicit start 后由真实 worker runtime 产生的 manifest，不是 admission 前的 launch
plan。Admission 使用 §5.13，避免在准入前生成 ephemeral keys 的 construction cycle。

### 5.4 `ShadowAdmissionReceiptV1` — `0x7A03`

```text
[
  1, 0x7A03, b"hegel-internal-shadow-admission-receipt/1",
  1,
  shadow_run_id,
  policy_binding_digest,
  isolation_plan_digest,
  basis_commit_id,
  0x0FFF,
  12,
  14,
  24,
  0,
  true,
  false,
  admitted_at_unix_seconds
]
```

字段语义依次为 `shadow_gate_bitset`、`shadow_gate_count`、`formal_gates_satisfied`、
`formal_gates_total`、`formal_m3_state_id`、`formal_roots_all_null`、
`external_actor_evidence`。最后两项必须为 `true,false`。

### 5.5 `ShadowEnvelopeV1` — `0x7A04`

```text
[
  1, 0x7A04, b"hegel-internal-shadow-envelope/1",
  1,
  shadow_run_id,
  enclosed_shadow_object_digest,
  signer_purpose_id,
  signer_key_id,
  0,
  signature_64_bytes,
  false
]
```

### 5.6 `ShadowRunGenesisV1` — `0x7A05`

```text
[
  1, 0x7A05, b"hegel-internal-shadow-run-genesis/1",
  1,
  shadow_run_id,
  policy_binding_digest,
  admission_receipt_digest,
  isolation_manifest_digest,
  basis_commit_id,
  1,
  null, null, null, null, null,
  null, null, null, null, null,
  created_at_unix_seconds,
  false
]
```

`initial_shadow_state_id=1`。恰好 10 个 run-produced candidate slots：

1. canonical program archive digest；
2. program chunk manifest digest；
3. bucket-accounting digest；
4. odd output archive digest；
5. odd match-set digest；
6. odd role receipt digest；
7. sink output archive digest；
8. sink match-set digest；
9. sink role receipt digest；
10. dual-replay agreement digest。

Genesis 时十项必须全部 `null`；最后字段 `formal_run_genesis_claim` 必须为 `false`。

### 5.7 `ShadowStateRecordV1` — `0x7A06`

```text
[
  1, 0x7A06, b"hegel-internal-shadow-state-record/1",
  1,
  shadow_run_id,
  transition_index,
  previous_state_record_digest_or_null,
  from_shadow_state_id,
  to_shadow_state_id,
  transition_reason_id,
  triggering_shadow_receipt_digest_or_null,
  recorded_at_unix_seconds,
  14,
  24,
  0
]
```

后三项在每次 transition 中重申 formal `14/24 / NOT_RUN`。

### 5.8 `ShadowEnumerationReceiptV1` — `0x7A07`

```text
[
  1, 0x7A07, b"hegel-internal-shadow-enumeration-receipt/1",
  1,
  shadow_run_id,
  implementation_id,
  basis_commit_id,
  canonical_program_budget,
  syntactically_canonical_program_count,
  accepted_unique_program_count,
  first_50001_witness_digest_or_null,
  frontier_closed,
  candidate_program_archive_digest_or_null,
  candidate_chunk_manifest_digest_or_null,
  candidate_bucket_accounting_digest,
  shadow_outcome_id,
  started_at_unix_seconds,
  finished_at_unix_seconds,
  false
]
```

`implementation_id` 只接受 `1=PYTHON`、`2=RUST`。最后字段
`formal_enumeration_receipt_claim=false`。

### 5.9 `ShadowRoleEvaluationReceiptV1` — `0x7A08`

```text
[
  1, 0x7A08, b"hegel-internal-shadow-role-evaluation-receipt/1",
  1,
  shadow_run_id,
  implementation_id,
  target_role_id,
  candidate_program_archive_digest,
  candidate_output_archive_digest,
  candidate_match_set_digest,
  match_count,
  undefined_program_count,
  evaluation_complete,
  sealed_assignment_consumed_via_fd4,
  disclosure_ledger_head_digest,
  started_at_unix_seconds,
  finished_at_unix_seconds,
  false
]
```

`target_role_id` 使用 `1=ODD_OUTSIDE_TARGET`、`2=SINK_NULL_CONTROL` 的 shadow-local
registry，不复用 formal `ArtifactRoleId`。最后字段
`formal_role_evaluation_receipt_claim=false`。

### 5.10 `ShadowDualReplayAgreementV1` — `0x7A09`

```text
[
  1, 0x7A09, b"hegel-internal-shadow-dual-replay-agreement/1",
  1,
  shadow_run_id,
  python_enumeration_receipt_digest,
  rust_enumeration_receipt_digest,
  python_odd_receipt_digest_or_null,
  rust_odd_receipt_digest_or_null,
  python_sink_receipt_digest_or_null,
  rust_sink_receipt_digest_or_null,
  canonical_bytes_equal,
  candidate_archive_digests_equal,
  role_result_digests_equal,
  policy_checks_pass,
  false
]
```

`policy_checks_pass` 必须为 strict `true`。最后字段
`external_dual_attestation_claim=false`。Purpose 4 在该 object 构造并审计后签署其 digest，
因此 agreement 不反向引用 purpose-4 envelope，不形成签名循环。

### 5.11 `ShadowDisclosureLedgerRecordV1` — `0x7A0A`

```text
[
  1, 0x7A0A, b"hegel-internal-shadow-disclosure-ledger-record/1",
  1,
  shadow_run_id,
  sequence_number,
  previous_record_digest_or_null,
  event_type_id,
  subject_digest_or_null,
  recipient_purpose_id_or_zero,
  synthesis_visible,
  recorded_at_unix_seconds
]
```

Sequence 从 0 连续。若任一 record 的 `event_type_id=4` 或
`synthesis_visible=true` 且 subject 属于 validation/sealed membership、role outcome 或
match set，则 `holdout_contaminated=true`，不可在后续 external track 中静默清除。

### 5.12 `ShadowExecutionBundleV1` — `0x7A0B`

```text
[
  1, 0x7A0B, b"hegel-internal-shadow-execution-bundle/1",
  1,
  shadow_run_id,
  policy_binding_digest,
  admission_receipt_digest,
  isolation_manifest_digest,
  run_genesis_digest,
  final_state_record_digest,
  dual_replay_agreement_digest_or_null,
  disclosure_ledger_head_digest,
  terminal_shadow_state_id,
  terminal_shadow_outcome_id,
  holdout_contaminated,
  [purpose_1_envelope_digest, purpose_2_envelope_digest,
   purpose_3_envelope_digest, purpose_4_envelope_digest],
  completed_at_unix_seconds,
  14,
  24,
  0,
  false,
  false
]
```

末尾两个 fields 分别是 `formal_evidence_claim=false` 和
`external_independence_claim=false`。

四个 envelope 的 enclosed digest scope 严格为：

```yaml
purpose_1: isolation_manifest_digest
purpose_2: ShadowTreeDigestV1("PYTHON_RECEIPT_SET", [python_enum, python_odd_or_null, python_sink_or_null])
purpose_3: ShadowTreeDigestV1("RUST_RECEIPT_SET", [rust_enum, rust_odd_or_null, rust_sink_or_null])
purpose_4: dual_replay_agreement_digest
```

这样 `ShadowExecutionBundleV1` 在四个 signatures 之后构造，不存在 object 签自己或 bundle
反向进入被签 object 的 cycle。

### 5.13 `ShadowIsolationPlanV1` — `0x7A0C`

```text
[
  1, 0x7A0C, b"hegel-internal-shadow-isolation-plan/1",
  1,
  shadow_run_id,
  basis_commit_id,
  snapshot_manifest_digest,
  [1, 2, 3, 4],
  1,
  worker_launch_plan_digest,
  required_security_probe_digest,
  fd_policy_digest,
  output_allowlist_digest,
  secret_lint_policy_digest,
  false
]
```

最后的 `false` 是 `external_independence_claim`。该 plan 只包含 executable/configuration
bindings 和无 secret 的 namespace probes；不得含 runtime PID、ephemeral public/private key
或 shadow seed。Admission receipt 绑定此 plan。Explicit start 后产生的四个 worker manifests
和 `ShadowIsolationManifestV1` 必须逐项满足该 plan。

### 5.14 `ShadowSecurityProbeReceiptV1` — `0x7A0D`

```text
[
  1, 0x7A0D, b"hegel-internal-shadow-security-probe-receipt/1",
  1,
  shadow_run_id,
  shadow_purpose_id,
  probe_phase_id,
  worker_instance_id,
  basis_commit_id,
  2,
  1,
  [[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1]],
  landlock_status_id,
  landlock_nonblocking_gap_disclosed,
  transient_capability_probe_incident_count,
  transient_capability_probe_incident_digest_or_null,
  observed_at_unix_seconds,
  false
]
```

`probe_phase_id` 严格为 `1=ADMISSION_PROBE` 或 `2=START_RUNTIME_PROBE`。Prefix 后两个固定
integers `2,1` 分别是 `/proc/self/status` 的 `Seccomp` 和 `NoNewPrivs` 值。六行
`[attack_syscall_id, observed_errno]` 必须 exact、ordered 且 errno 都为 `1=EPERM`。

`transient_capability_probe_incident_count` 可以为 0 或正整数；它是透明诊断计数，不是
admission/state bit。Count 为 0 时 digest 必须 `null`；大于 0 时 digest 必须绑定一个只含
incident type、probe command digest、observed capability metadata、timestamp 和 resolution 的
non-secret diagnostic collection，禁止写 raw process memory 或 secret-shaped bytes。最后字段
`external_security_attestation_claim` 必须为 `false`。

---

## 6. Shadow admission gates

Shadow gate registry 与 formal gates 无关。只有以下 12 项全部通过，才能从
`NOT_ADMITTED` 进入 `ADMITTED_NOT_STARTED`：

| Gate | Exact name | Pass predicate | Failure |
|---:|---|---|---|
| 1 | `SHADOW_OWNER_POLICY_BOUND` | 本 amendment 的 committed Git blob 和 policy digest 已绑定 | `FAIL_SHADOW_POLICY_NOT_BOUND` |
| 2 | `SHADOW_BASIS_COMMIT_PINNED` | basis SHA-1 committed、reachable、source equality pass | `FAIL_SHADOW_BASIS_COMMIT_MISMATCH` |
| 3 | `SHADOW_READ_ONLY_SNAPSHOT_VERIFIED` | detached snapshot pre-hash pass 且 live worktree不可见 | `FAIL_SHADOW_SNAPSHOT_NOT_READ_ONLY` |
| 4 | `SHADOW_DETERMINISTIC_DUAL_BASELINE_PASS` | Commit-basis Python/Rust strict qualification pass | `FAIL_SHADOW_BASELINE_DUAL_MISMATCH` |
| 5 | `FORMAL_TRACK_INVARIANTS_UNCHANGED` | exact `14/24 / NOT_RUN / roots null` | `FAIL_SHADOW_FORMAL_STATE_MUTATION` |
| 6 | `SHADOW_TAG_AND_DOMAIN_SEPARATION_PASS` | shadow/formal cross-decoder negative tests pass | `FAIL_SHADOW_DOMAIN_COLLISION` |
| 7 | `FOUR_PURPOSE_LAUNCH_PLAN_EXACT` | purposes exactly `[1,2,3,4]`，worker plans distinct | `FAIL_SHADOW_PURPOSE_SET` |
| 8 | `LOCAL_NAMESPACE_AND_SECCOMP_ISOLATION_AVAILABLE` | §4.2 18-bit profile + §4.3 four-purpose live seccomp probes pass；Landlock gap透明披露 | `FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE` |
| 9 | `SHADOW_FD_POLICY_VERIFIED` | FD 3/4/5 allowlist、all others closed | `FAIL_SHADOW_SECRET_CHANNEL_POLICY` |
| 10 | `SHADOW_SECRET_NONPERSISTENCE_PLAN_VERIFIED` | tmpfs/zeroize/core-dump/cleanup checks pass | `FAIL_SHADOW_SECRET_PERSISTENCE_POLICY` |
| 11 | `SYNTHESIS_BLINDNESS_ROUTE_VERIFIED` | synthesis process无 target oracle、FD3、FD4、sealed outputs | `FAIL_SHADOW_SYNTHESIS_BLINDNESS` |
| 12 | `SHADOW_OUTPUT_AND_CLAIM_LINTER_PASS` | output allowlist、artifact kind、forbidden claims pass | `FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED` |

Admission receipt 的 bitset 必须为 `0x0FFF`。`11/12` 不能降级运行，也不能生成 temporary
key/seed。Admission 成功只产生 state transition index 0：

```text
NOT_ADMITTED
-- SHADOW_ADMISSION_GATES_12_OF_12 -->
ADMITTED_NOT_STARTED
```

它不自动启动 enumeration。

Admission procedure 严格为：先验证 gates 1–6；再生成并 O_EXCL reserve 一个公开、非
formal、非 secret 的 16-byte `shadow_run_id`；随后运行 gates 8–12 的无 secret probes、构造
`ShadowIsolationPlanV1` 并验证 gate 7；最后重新汇总 12 bits 后才构造 admission receipt。
该 run ID 供 isolation plan、receipt 和 state record 共用。此过程不得生成 key 或 split
seed；后续任一 gate 失败时该 run ID 仍永久作废，不得复用。

---

## 7. Explicit start and state machine

### 7.1 Explicit action

唯一 start action 名称为：

```text
phase3-m3-shadow-start
```

该 action 必须重新验证 12/12、basis snapshot 和 formal invariants，随后在一个 transaction
中使用 admission receipt 已保留的 shadow run ID、启动四 workers、生成 ephemeral keys/seed、构造
`ShadowRunGenesisV1`，再创建 transition index 1：

```text
ADMITTED_NOT_STARTED
-- EXPLICIT_SHADOW_START -->
RUNNING_CANONICAL_ENUMERATION
```

调用 formal `phase3-m3-start`、写 formal state record 或构造 formal M3 run ID 均返回
`FAIL_SHADOW_FORMAL_START_PROHIBITED`。

### 7.2 Legal transitions

```yaml
legal_shadow_transitions:
  - [0, 1, 1]
  - [1, 2, 2]
  - [1, 8, 7]
  - [1, 9, 9]
  - [1, 10, 10]
  - [2, 3, 3]
  - [2, 5, 4]
  - [2, 6, 5]
  - [2, 7, 6]
  - [2, 8, 7]
  - [2, 9, 9]
  - [2, 10, 10]
  - [3, 4, 8]
  - [3, 7, 6]
  - [3, 8, 7]
  - [3, 9, 9]
  - [3, 10, 10]
```

Rows 是 `[from_state_id, to_state_id, transition_reason_id]`。不存在其他 transition；terminal
state 无 outgoing edge。

### 7.3 Canonical enumeration semantics

Enumeration 继续使用 frozen child DSL、strict canonicalizer、pre-count rewrites、typing、
bottom semantics、traversal order、search budget 和 50,000 syntax-cap 语义。Shadow 轨道
不改变它们：

- 发现第 50,001 个 syntactically canonical program：
  `DSL_TOO_LARGE_CANDIDATE`；
- budget 用尽但 frontier 未闭合：`INCONCLUSIVE_BUDGET`；
- Python/Rust semantics 或 canonical bytes 不同：`INCONCLUSIVE_SEMANTICS`；
- process/IO/host failure：`INCONCLUSIVE_EXECUTION`；
- 只有完整 frontier closed 且 count `<= 50,000` 才能进入 role evaluation；
- shrink-1 subset replay 即使小于 50,000 也绝不产生 `COMPLETE_CANDIDATE`。

### 7.4 Role evaluation semantics

Role evaluation 只能消费已经冻结的 target-independent candidate program archive。Odd 和
sink 分别产生 shadow-domain output/match digests。Allowed interpretation：

```yaml
odd_match_count_zero:
  allowed: CANDIDATE_NO_MATCH_IN_INTERNAL_SHADOW_ENUMERATION
  prohibited: OUTSIDE_FROZEN_CLOSURE

odd_match_count_positive:
  allowed: CANDIDATE_IN_LANGUAGE_WITNESS_FOUND
  prohibited: FORMAL_IN_LANGUAGE_VERDICT

sink_designated_witness_found:
  allowed: INTERNAL_FALSE_INVENTION_CONTROL_CANDIDATE_PASS
  prohibited: CONSERVATION_MECHANISM_IDENTIFIED
```

Python/Rust role outputs、bytes 和 digests 完全相同，且 purpose-4 policy audit pass 后，才能
进入 `COMPLETE_CANDIDATE`。这仍不产生 formal `COMPLETE`。

---

## 8. Failure registry and fail-closed behavior

### 8.1 Admission / identity failures

```text
FAIL_SHADOW_POLICY_NOT_BOUND
FAIL_SHADOW_ARTIFACT_KIND
FAIL_SHADOW_BASIS_COMMIT_MISMATCH
FAIL_SHADOW_FORMAL_STATUS_OMITTED
FAIL_SHADOW_FORMAL_STATE_MUTATION
FAIL_SHADOW_DOMAIN_COLLISION
FAIL_SHADOW_PURPOSE_SET
FAIL_SHADOW_KEY_REUSE
FAIL_SHADOW_PROCESS_REUSE
```

### 8.2 Isolation / secret failures

```text
FAIL_SHADOW_ISOLATION_PROFILE_UNAVAILABLE
FAIL_SHADOW_RUNTIME_PLAN_MISMATCH
FAIL_SHADOW_SECURITY_PROBE_SET_INCOMPLETE
FAIL_SHADOW_SECCOMP_NOT_FILTER_MODE
FAIL_SHADOW_SECCOMP_ATTACK_SYSCALL_NOT_EPERM
FAIL_SHADOW_LANDLOCK_GAP_NOT_DISCLOSED
FAIL_SHADOW_RUNTIME_SECURITY_PROBE_MISMATCH
FAIL_SHADOW_LIVE_WORKTREE_VISIBLE
FAIL_SHADOW_SNAPSHOT_NOT_READ_ONLY
FAIL_SHADOW_SNAPSHOT_MUTATED
FAIL_SHADOW_CROSS_PURPOSE_WRITABLE_PATH
FAIL_SHADOW_SECRET_CHANNEL_POLICY
FAIL_SHADOW_SECRET_PERSISTENCE_POLICY
FAIL_SHADOW_SECRET_MATERIAL_DETECTED_IN_OUTPUT
FAIL_SHADOW_SYNTHESIS_BLINDNESS
```

### 8.3 Replay / state / output failures

```text
FAIL_SHADOW_BASELINE_DUAL_MISMATCH
FAIL_SHADOW_SIGNATURE_MISMATCH
FAIL_SHADOW_DUAL_REPLAY_MISMATCH
FAIL_SHADOW_ADMISSION_INCOMPLETE
FAIL_SHADOW_START_NOT_EXPLICIT
FAIL_SHADOW_FORMAL_START_PROHIBITED
FAIL_SHADOW_INVALID_TRANSITION
FAIL_SHADOW_OUTPUT_NOT_ALLOWLISTED
FAIL_SHADOW_FORBIDDEN_CLAIM
FAIL_SHADOW_EXECUTION
```

Admission 之前的 failure 保持 state 0。State 1 之后发生 isolation、secret、forbidden claim、
snapshot mutation 或 synthesis-blindness failure 时必须进入 state 9，并销毁 ephemeral
secret state。不得通过 retry、删 ledger 或重写 report 把 state 9 变回 runnable；新尝试必须
使用新的 shadow run ID。

---

## 9. Artifact publication and claim linter

允许写入 repository 的 shadow artifact 仅包括：

- basis commit、snapshot/executable/environment digests；
- ephemeral public keys、key IDs 和 shadow envelopes；
- admission/isolation/state/enumeration/role/dual receipts；
- aggregate counts、candidate digests、failure codes；
- disclosure ledger metadata；
- explicit formal `14/24 / NOT_RUN` snapshot。

禁止写入：

- raw shadow/formal seed、private key/private-key seed；
- FD-4 payload、row-level assignments、validation/sealed membership；
- pre-terminal output rows、match rows、oracle implementation；
- live tmp paths、core dump、process memory；
- formal object bytes masquerading as shadow output；
- 任何 non-null field named `formal_root`、`outside_certificate`、`mdl_certificate`、
  `external_actor_evidence` 或 `active_promotion`。

Artifact path 必须位于：

```text
Hegel Machine/artifacts/phase3_internal_shadow/
```

每个 JSON 顶层必须含：

```yaml
artifact_kind: INTERNAL_PURPOSE_SEPARATED_NON_AUTHORITATIVE
formal_track_status: "14/24 / NOT_RUN"
external_independence_claim: false
formal_evidence_claim: false
```

Artifact publication 不得修改 implementation、tests、DSL、freeze 或 formal operator code。
若需要这些修改，先形成新的 basis commit，重新 admission，再执行新 shadow run。

---

## 10. Future external ratification boundary

### 10.1 What can be reused

未来 external ceremony 可以复用：

- frozen algorithm/specification；
- independently rebuilt deterministic implementation；
- shadow 暴露出的 bug report、performance measurement 和 engineering lesson；
- 作为优化 hint 的 candidate archive bytes，但必须从 formal basis commit 独立重算并逐字节
  验证，不能复制 shadow digest 当 formal root。

### 10.2 What can never be promoted

以下对象不得通过 rename、prefix replacement、bridge record 或 signature wrapping 升格：

- shadow run ID / seed / key / signature / envelope；
- shadow admission gate bits；
- shadow state record / terminal outcome；
- 任何 `ShadowDigestV1` / `ShadowTreeDigestV1`；
- shadow odd/sink verdict；
- 由同一 orchestrator 调度得到的 purpose separation 声明。

Formal ratification 必须重新完成：

1. fresh Commit-A qualification；
2. real external purpose keys and actor trust；
3. formal split seed 的首次实例化；
4. formal roots、bridge envelopes、Gate 15–24；
5. formal `24/24 / NOT_RUN`；
6. 单独 formal `phase3-m3-start`；
7. 完整 formal dual enumeration / role evaluation。

Shadow seed 不算 formal seed 的历史实例化，因此未来 external seed 仍称“formal first
instantiation”，不是 redraw。

### 10.3 Code changes and holdout contamination

```yaml
external_ratification_rules:
  shadow_influenced_executable_or_dsl_change:
    action: NEW_BASIS_COMMIT_AND_FULL_REQUALIFICATION
  shadow_digest_or_signature_reuse:
    action: REJECT
  forbidden_artifact_exposed_to_synthesis:
    action: HOLDOUT_CONTAMINATED
  contaminated_target_claim:
    action: OWNER_AMENDMENT_OR_CLAIM_DOWNGRADE_REQUIRED
```

若 disclosure ledger 出现 event 4，或 validation/sealed membership、role outcome、match set
进入 synthesis/code-selection 输入，则受影响 target 不得被描述为 untouched formal holdout。
未来 formal 流程必须由 owner 明确选择以下之一：

1. 预先冻结且可证明不重叠的 preservation cases；
2. 新的 owner-approved target/split amendment；
3. 将结论降级为 exploratory/mechanics evidence。

禁止删除 shadow artifact、换 seed 或换 run ID 来“恢复”未污染声明。

---

## 11. Required dual-status terminal summary

每个 terminal run 必须输出以下结构，字段不得省略：

```yaml
formal_track:
  gates: "14/24"
  state: NOT_RUN
  formal_roots: null
  external_actor_evidence: false
  outside_certificate: null
  mdl_certificate: null
  active_promotion: false

shadow_track:
  artifact_kind: INTERNAL_PURPOSE_SEPARATED_NON_AUTHORITATIVE
  admission_gates: "12/12"
  state: <ShadowStateId name>
  outcome: <ShadowOutcomeId name>
  purpose_ids: [1, 2, 3, 4]
  external_independence_claim: false
  hardening_gaps: <[] or [HARDENING_GAP_LANDLOCK_NOT_ENFORCED_NONBLOCKING]>
  holdout_contaminated: <strict boolean>
  candidate_execution_bundle_digest: <32-byte digest>
```

这使项目可以从研究意义上的“未运行”推进到真实的 internal candidate enumeration / role
evaluation，同时保持 formal evidence ledger 的事实完全不变。

---

## 12. Final authorization

本 amendment 批准下一施工阶段：

> **Phase-3A Internal Shadow M3 — Purpose-Separated Candidate Closure Enumeration and Role Evaluation**

允许顺序：

```text
shadow policy implementation
-> 12/12 shadow admission
-> explicit phase3-m3-shadow-start
-> canonical enumeration
-> role evaluation when frontier closes <= 50,000
-> candidate terminal outcome
```

同时明确禁止：

```text
shadow evidence -> formal Gate 15–24
shadow signature -> external actor evidence
shadow digest -> formal root
candidate no-match -> outside certificate
candidate terminal state -> ACTIVE
```

因此，external actors 暂时不可得不会再阻塞内部工程和认知能力研究；它只继续阻塞正式
ratification、certificate 和 ACTIVE claim。
