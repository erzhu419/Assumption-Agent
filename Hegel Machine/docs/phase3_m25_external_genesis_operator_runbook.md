# Phase-3A M2.5 External Genesis Operator Runbook

**文档性质**：外部操作交接稿；不是可执行程序、签名请求或授权凭证
**machine freeze ID**：`hegel-freeze-p2b-p3-v1.1.2`
**child DSL ID**：`hegel-old-dsl-v1.1.0`
**当前状态**：`NOT_EXECUTED`
**当前 child state**：`NOT_RUN`
**当前 M3 gates**：执行前仍为 `14/24`

本 runbook 只说明：Commit A 的 checked dual qualification 真正通过后，独立
custodian、Python bridge attester、Rust bridge attester 和 parent-absence auditor
应如何完成首次 external genesis，并把仅含公开材料的结果交给 Commit B 发布。

本文档的创建没有、也不得：

- 调用 OS CSPRNG；
- 创建或读取 split seed、private key 或 secret-state marker；
- 生成真实签名、formal root、external actor evidence 或 M3 execution identity；
- 推进 Gate 15–24；
- 执行 `NOT_RUN -> RUNNING`。

测试和 golden vector 中固定的 key-shaped、seed-shaped、signature-shaped 字节只供
跨实现资格化。它们不得复制、派生或重用为真实 key、seed、ID、signature 或
external-actor evidence。

## 1. 规范与实现边界

操作员必须共同使用以下冻结依据，不得从较早文档或摘要补写字段：

1. [Hegel_Machine_Phase3A_M25_Bit_Exact_Wire_Completion_Amendment.md](Hegel_Machine_Phase3A_M25_Bit_Exact_Wire_Completion_Amendment.md)；
2. [Hegel_Machine_Phase3A_M25_Exact_Wire_Errata_Resolution.md](Hegel_Machine_Phase3A_M25_Exact_Wire_Errata_Resolution.md)；
3. [Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md](Hegel_Machine_Phase3A_M25_Implementation_Closure_Addendum_v1.md)；
4. `src/hegel_machine/phase3_m25_wire_v1.py` 的 strict wire validators；
5. `src/hegel_machine/phase3_m25_external_v1.py` 的 side-effect-free external validators；
6. Commit A 中 checked Python/Rust errata vectors、golden fixture、tests 和 source bindings。

现有 external module 只能做只读或纯验证；它不能生成随机数、key、marker、签名或
formal root。不得把“validator 返回通过”描述成 external genesis 已经发生。

## 2. 开始条件：先锁定 Commit A

### 2.1 唯一准入证据

在创建 marker、生成任何真实 key 或调用 split-seed CSPRNG 之前，操作员必须对
Commit A 运行 fresh dual qualification，并确认十项 guard 全部为严格 JSON boolean
`true`。Checked qualification artifact 只用于选择 Commit A、比较预期结果和归档；
它本身不能授权：

1. `errata_document_in_commit_A`；
2. `python_errata_vectors_pass`；
3. `rust_errata_vectors_pass`；
4. `python_rust_canonical_bytes_equal`；
5. `python_rust_error_codes_equal`；
6. `actor_trust_genesis_schema_frozen`；
7. `append_only_id_registry_schema_frozen`；
8. `parent_audit_bundle_schema_frozen`；
9. `bridge_statement_and_execution_v2_schema_frozen`；
10. `secrets_absent_from_repository`。

准入必须来自 Commit A 自身的 committed sources：fresh empty target、locked/offline
Rust build、构建前后 Commit-A blob equality、Python/Rust exact report equality 和
repository secret-absence receipt 都必须成立。构建和 Python replay 必须来自私有的
detached Commit-A snapshot，而不是 live worktree；Rust 必须使用隔离 Cargo home、
whitelist environment；隔离 Cargo home 只复制与 `Cargo.lock` SHA-256 完全匹配的
`.crate` archives 和 offline index，不复制 ambient unpacked source，并验证所有可见
ancestor/Cargo-home config 均不存在。Cargo/Rust 路径不得由调用者指定；rustup
launcher、实际 cargo/rustc binaries、版本和完整 selected-toolchain directory manifest
必须匹配 Commit A 中的 approved-local-toolchain policy。随后必须对同一 open binary
inode 完成 hash 和 exec。调用者提供的任意
binary、stored JSON/self-hash、工作树结果或口头确认都不能替代 fresh replay。
Python replay 只装载 exact-wire generator 的显式最小模块闭包，不执行宽泛的 package
`__init__`；receipt 必须把这两点分别记录为 true/false，禁止隐式扩大输入闭包。

任一项缺失、类型不为 `bool` 或为 `false`，立即返回：

```text
FAIL_M25_EXACT_ERRATA_REQUIRED
```

此失败必须发生在所有外部副作用之前。

### 2.2 固定两个不同的 Git 身份

操作员从 qualification receipt 记录并离线保留：

```text
COMMIT_A_SHA1 = deterministic implementation basis commit
AUDITED_PARENT_SHA1 = fb3a3ee4865a140c558821017ddd3e9a6a99de48
```

二者语义不同：

- `COMMIT_A_SHA1` 绑定本次确定性实现、规范、golden、tests 和所有随后构造的 external
  formal objects；
- `AUDITED_PARENT_SHA1` 是 parent-manifest absence audit 的历史边界。

在 Commit A 与发布 Commit B 之间构造的每个含 `repository_commit_id` 的 formal object
都必须填 Commit A 的 Git wire ID `[1, COMMIT_A_SHA1_20_BYTES]`。不得填工作树当前
`HEAD`、未来 Commit B 或 audited parent commit。Commit B 只由 publication receipt
和 Git history 绑定，不能让 formal object 自引用 Commit B。

## 3. 人员、进程与四个 purpose

### 3.1 独立性

Codex 自动运行环境和 repo-building agent 不得充当 independent custodian 或
independent auditor。最低可接受安排是：

- 用户本人在隔离的 one-shot process/OS account 中担任 custodian；或
- 用户指定外部人员担任 custodian；
- parent-absence auditor 由不同人员独立执行；
- Python/Rust bridge attester 分别重放同一公开 statement，再使用各自 purpose key
  签名。

只换一把 key、但仍由同一自动 agent 自我审计，不构成 independent auditor claim。

### 3.2 冻结的 purpose registry

| Purpose | 角色 | 本阶段签名范围 |
|---:|---|---|
| 1 | `CUSTODIAN_IDENTITY_AND_BRIDGE_ATTESTER` | 四个 custodian objects；同一 bridge statement |
| 2 | `PYTHON_BRIDGE_ATTESTER` | 同一 bridge statement |
| 3 | `RUST_BRIDGE_ATTESTER` | 同一 bridge statement |
| 4 | `PARENT_ABSENCE_AUDITOR` | parent-absence attestation V2 |
| 5 | `FINAL_CERTIFICATE_SIGNER_RESERVED_FOR_M4` | 本阶段不得出现 |

split calculator 与 bridge attester 不是同一信任角色。Python/Rust split calculators
只是 custodian one-shot process 的受限子进程：它们通过 FD 3 接收 seed、没有独立 actor
key，也不形成 hidden-access ledger actor。Purpose 2/3 是之后对公开 bridge statement
做独立重放和签名的 attesters。

### 3.3 真实 key genesis

只有十项 guard 全部通过后，外部操作员才可分别调用 OS CSPRNG 生成四把真实 Ed25519
key：

```yaml
algorithm: Ed25519
private_key_seed_length: 32
generation_profile: hegel-os-csprng-v1
initial_key_epoch: 0
```

每把 key 的 CSPRNG 调用与 32-byte split-seed 调用相互独立。Key ID 为：

```text
first_16_bytes(SHA256(raw_32_byte_Ed25519_public_key))
```

四个 key ID 必须两两不同，四个 raw public keys 也必须两两不同。Purpose 1 可在不同
domain 下签署多个 purpose-1 objects；这是同 purpose 使用，不是跨 purpose key reuse。
任何 key ID 或 public-key collision 都必须 fail closed；不得通过改 label 掩盖。

`ActorTrustGenesisV1` 必须：

- 只含 purposes `1,2,3,4`，按 purpose ID 升序；
- 绑定四个真实 `ActorKeyManifestV1` roots；
- 将 `purpose_key_policy_root` 绑定冻结的 `ReplacementPolicyV1` ContentHash root；
- 使用 Commit A 的 `repository_commit_id`。

其 root 是 trust anchor。文件名、key label、URL 或实现 source root 都不是 trust
anchor。项目 owner 应在 repo 外独立保存该 anchor；公开副本由 Commit B 发布。

## 4. Repo 外 secret state

### 4.1 目录和文件

选择一个真实、非 symlink、位于 repository 之外且不在 cloud-sync 下的目录。开始前用
只读 validator 验证：

```text
secret-state directory: mode 0700
every private-key file: regular file, inside secret-state, mode 0600
split-seed file: regular file, inside secret-state, mode 0600, exactly 32 bytes
marker/lock state containing custody metadata: mode 0600
```

必须同时满足 OS account-level access control。推荐 OS key store 或 hardware token，
但 M2.5 不强制。禁止将任何 secret 放入：

- repository、Git LFS 或 public artifact bundle；
- argv、environment、stdin 或 shell history；
- stdout/stderr、日志、core dump；
- world/group-readable temporary file。

路径或权限不合格时，分别按 validator fail closed，例如
`FAIL_SECRET_STATE_PATH_INVALID`、`FAIL_SECRET_STATE_INSIDE_REPOSITORY`、
`FAIL_SECRET_STATE_PERMISSIONS` 或 `FAIL_SECRET_FILE_PERMISSIONS`；不得降级继续。

### 4.2 one-shot marker：必须先 PENDING，后 COMPLETE

在 split seed 的 OS CSPRNG 调用之前：

1. 对 secret-state directory 获取 exclusive lock；
2. 确认 `split_seed_instantiation.marker` 不存在；
3. 使用 `O_CREAT|O_EXCL` 原子创建 marker，不得先覆盖或删除旧 marker；
4. fsync marker 和目录；
5. marker 初始状态严格为：

```text
state = PENDING
split_version_digest = exact 32-byte frozen split version digest
seed_commitment_manifest_root = null
custodian_key_id = exact 16-byte purpose-1 key ID
created_at_unix_seconds = nonnegative integer
```

marker 已存在时，在 CSPRNG 前返回：

```text
FAIL_SPLIT_SEED_ALREADY_INSTANTIATED
```

只有 marker 已持久化为 `PENDING` 后，才可首次、且仅首次调用 OS CSPRNG 生成 32-byte
split seed。Seed commitment manifest 和公开 evidence 完成后，以原子替换将 marker
变为：

```text
state = COMPLETE
seed_commitment_manifest_root = exact non-null 32-byte root
```

然后再次 fsync file 和 directory。`PENDING` 状态下发生中断时，不得删除 marker、自动
重试 CSPRNG 或重抽 seed；必须返回
`FAIL_SPLIT_SEED_PENDING_EXTERNAL_RECOVERY_REQUIRED`，等待单独批准的 external
recovery procedure。本文不发明该 recovery procedure。

`COMPLETE` 也永久阻止第二次实例化。Marker 防止 redraw，不是 seed 本身，也不能进入
Commit B。

## 5. FD 3 split-calculator 运行

custodian one-shot process 分别启动冻结的 Python 和 Rust split calculators。两个 child
process 都必须绑定 Commit A allowlisted executable digests，并满足：

1. 使用 inherited anonymous pipe，secret descriptor 严格为 FD `3`；
2. custodian 向每个 child 分别写入 exactly 32 bytes，然后 EOF；
3. argv、environment 和 stdin 不含 seed；
4. child 读取 exactly 32 bytes 后立即关闭 FD 3；
5. child 无网络、无 repository 写权限，不持久化或打印 seed；
6. child 不输出 row-level assignment、validation/sealed membership 或其他 sealed data；
7. child 尽力 mlock 并在退出前 zeroize input buffer；
8. exit code 必须为 `0`；
9. 唯一可交接输出是 allowlisted public commitment evidence、split roots/counts 和后续
   formal construction 所需的公开 roots。

FD 不是 `3`、payload 不为 exact 32-byte bytes、未见 EOF 或 child 非零退出时，分别以
`FAIL_SECRET_PIPE_RUNTIME` 或 `FAIL_PROCESS_NONZERO_EXIT` fail closed。不得把 seed
改放 argv/env/stdin 作为兼容性回退。

两个 calculators 的公开结果必须按冻结算法重放一致。结果不一致时不得签名、不得将
marker 转为 `COMPLETE`、不得发布 Commit B。

## 6. Parent-absence auditor

独立 auditor 使用 purpose-4 key 对固定 parent commit
`fb3a3ee4865a140c558821017ddd3e9a6a99de48` 做完整 replay：

1. 遍历该 commit 可达的全部 Git history；
2. 校验 ordered parent commit IDs 和 commit generations；
3. 为每个 commit 构造按 raw path bytes 排序的 touched-path record tree；
4. 构造去重后的 top-level audited-path union；
5. 构造且只构造两个冻结 legacy source rows，分别对应 outside target 与 sink control；
6. 构造 `ParentAbsenceAuditBundleV1`；
7. 构造 `ParentManifestAbsenceAttestationV2`，其 `absence_reason_bitmask` 严格为
   `0b1111`；
8. 使用 Commit A 作为 formal objects 的 `repository_commit_id`，同时保留上面的固定
   parent SHA-1 作为被审计对象；
9. 使用 purpose 4、epoch 0 和
   `HEGEL/PARENT_ABSENCE_AUDITOR_SIGNATURE/V2` 生成一个真实
   `SignedManifestEnvelopeV1`。

Purpose-4 envelope 必须恰好包含一个签名。Auditor 不得接触 raw split seed 或
purpose-1 private key；custodian 不得代替 auditor 自签 parent-absence claim。

## 7. 五个 external-input envelopes

Purpose 1 使用同一真实 custodian key、但使用四个不同 domain，为四个不同 object roots
各生成一个单签名 `SignedManifestEnvelopeV1`：

| Purpose | Tag | Object | Signature domain |
|---:|---:|---|---|
| 1 | `0x3103` | seed commitment | `HEGEL/CUSTODIAN_SPLIT_SEED_COMMITMENT_SIGNATURE/V1` |
| 1 | `0x3105` | custodian binding | `HEGEL/CUSTODIAN_BINDING_SIGNATURE/V1` |
| 1 | `0x3106` | seed continuity | `HEGEL/CUSTODIAN_SEED_CONTINUITY_SIGNATURE/V1` |
| 1 | `0x3108` | hidden-access ledger genesis | `HEGEL/CUSTODIAN_LEDGER_GENESIS_SIGNATURE/V1` |

再加入 auditor 的：

| Purpose | Tag | Object | Signature domain |
|---:|---:|---|---|
| 4 | `0x3114` | parent-manifest absence attestation V2 | `HEGEL/PARENT_ABSENCE_AUDITOR_SIGNATURE/V2` |

每个 external signature preimage 严格为：

```text
UTF8(domain_for_object_tag)
|| 0x00
|| enclosed_object_root_32_bytes
|| uint16_be(signer_purpose_id)
|| uint64_be(signer_key_epoch)
```

每个 envelope：

- 恰好一个签名；
- 使用 formal field `signer_key_epoch`；
- 将历史名称 `custodian_key_epoch` 仅视为废弃文档别名，不得编码为额外字段；
- root 使用 `HEGEL/SIGNED_MANIFEST_ENVELOPE/V1` ContentHash domain。

五项按 `(purpose_id, enclosed_object_root, signed_envelope_root)` canonical order 进入
`AttestationBundleV1`。该 bundle root 绑定到
`M3ExecutionCandidateV1.custodian_attestation_bundle_root`。字段名为历史保留名，其
精确语义是“四个 purpose-1 custodian envelopes 加一个 purpose-4 parent-auditor
envelope”，不是只有 custodian 的四项集合。

## 8. ID registry、ledger 与 execution candidate

在任何新 ID 被 consumer object 使用前：

1. 为 fresh run ID 和 ledger ID 构造 registration intent；
2. 以 one-file-per-ID `O_EXCL` reservation 防止并发重用；
3. 向 append-only `OpaqueIdRegistryRecordV1` 追加连续 sequence；
4. raw 16-byte ID 必须跨 `RUN_ID`/`LEDGER_ID` kind 全局唯一；
5. 构造 singleton `added_record_root`、完整 `registry_tree_root` 和新 snapshot；
6. replay previous snapshot prefix 和 record count `+1`。

hidden-access ledger 在此阶段只有 genesis：`sequence_number == 0`，且
`ledger_head_root == ledger_genesis_root`；不得存在 access-granted 或 revealed record。
split calculators 的 FD 3 读取不生成 ledger access event，因为它们处于 custodian
process boundary 内。

随后按冻结 root DAG 构造 `M3ExecutionCandidateV1`。至少必须实际绑定：

- Commit-A child DSL/freeze、operator、identifier、AST/CBOR 和 contract roots；
- target/control universe、truth 和 six split roots；
- split/custodian/seed-continuity bindings；
- 五项 external-input attestation bundle；
- parent-absence attestation V2；
- ledger genesis/head；
- append-only opaque-ID snapshot；
- `ActorTrustGenesisV1` root；
- fresh run ID；
- Commit A 的 `repository_commit_id`。

candidate 构造时不得预填任何 run-produced output root。

## 9. 三个 bridge envelopes 与 final execution identity

### 9.1 Acyclic order

严格按以下顺序工作，不能先签一个尚不存在的 final manifest：

1. 完成并 root `M3ExecutionCandidateV1`；
2. 构造 `BridgeReplayStatementV1`，绑定 run ID、diagnostic-formal bridge root、
   execution candidate root、child DSL/freeze、actor trust 和 opaque-ID snapshot；
3. 得到唯一 bridge statement root；
4. purposes `1,2,3` 分别独立重放并签署同一 statement root；
5. 构造三项 bridge attestation bundle；
6. 构造 `M3ExecutionManifestV2`，绑定 candidate、statement、bridge bundle、trust 和
   opaque-ID snapshot；
7. 构造 `M3RunGenesisV1`，绑定 final manifest root、initial state `NOT_RUN=0`，并保持
   exactly 15 个 output slots 全部为 `null`。

### 9.2 bridge signature

三个 attesters 的 exact preimage 都是：

```text
UTF8("HEGEL/BRIDGE_ATTESTATION_SIGNATURE/V1")
|| 0x00
|| bridge_replay_statement_root_32_bytes
|| uint16_be(signer_purpose_id)
|| uint64_be(signer_key_epoch)
```

三项 purposes 必须严格为 `[1,2,3]`，每个 purpose 一个单签名 envelope，三个 key IDs
和 raw public keys 两两不同。Purpose 1 可复用其 custodian key，因为这是 purpose 1
在另一个 domain 下签署；purpose 2/3 不得复用 purpose 1 或 purpose 4 的 key。

Python/Rust attesters 必须签自己独立 replay 后得到的同一 statement root，而不是盲签
custodian 传入的 hex string。任一 replay、root、purpose、epoch 或 signature verification
不一致，都不得构造 final execution manifest。

## 10. Gate 24 与状态边界

external roots 和所有 signatures replay 通过后，再评估剩余 gates。Gate 24 的唯一名称
是：

```text
M3_EXECUTION_MANIFEST_ROOT_NON_NULL_AND_15_OUTPUT_ROOTS_NULL
```

通过条件包括：

- final `M3ExecutionManifestV2` root non-null；
- `M3RunGenesisV1` root non-null；
- initial state 为 `NOT_RUN=0`；
- 15 个 output slots 全部为 `null`；
- run ID 已注册在绑定的 opaque-ID snapshot；
- bridge envelope count 为 3，purposes 严格为 `[1,2,3]`。

达到 `24/24` 的结果仍然是：

```text
m3_entry_qualified = true
child_state = NOT_RUN
m3_run_started = false
```

本 runbook 不执行 `phase3-m3-start`。只有之后一个单独、显式的 operator action，在重新
验证完整 24/24 evidence 后，才可构造唯一 start transition：
`NOT_RUN/NONE -> RUNNING/CANONICAL_ENUMERATION`。External genesis、Gate 24 或 Commit B
都不得自动触发该 transition。

## 11. Commit B public-only publication

### 11.1 发布前 secret lint 与人工审计

Commit B 只可包含公开 manifests、public keys、key IDs、roots、receipts、signed
envelopes 和 readiness/status artifacts。以下材料永远不得进入 Commit B：

- raw private key、private-key seed、raw split seed；
- derived `K_role`；
- row-level split assignments；
- validation/sealed membership；
- sealed predictions；
- pre-final match set 或 output archive。

当前 public-payload validator 明确拒绝这些字段名及其任意嵌套形式：

```text
raw_private_key
private_key
private_key_seed
raw_split_seed
split_master_seed
master_seed_hex
derived_role_key
k_role
assignment_rows
validation_membership
sealed_membership
sealed_prediction_membership
pre_final_match_set
pre_final_output_archive
```

该字段 lint 不是“任意字节绝无秘密”的证明；发布前仍须独立人工审计、检查 private-key
headers、路径、权限、Git staged blob 和生成日志。

### 11.2 A→B changed-path allowlist

对 Commit A 到候选 Commit B 的 changed paths 调用
`validate_commit_b_changed_paths()`。本 runbook 的 publication allowlist 严格为：

```text
allowed_public_prefixes:
  - Hegel Machine/artifacts/phase3_m25_external
  - Hegel Machine/docs/phase3_m25_external_status.md

executable_prefixes:
  - Hegel Machine/src
  - Hegel Machine/rust
  - Hegel Machine/tests
```

因此 Commit B 可新增/更新：

- `Hegel Machine/artifacts/phase3_m25_external/` 下经过审计的公开 evidence；
- 唯一状态交接文档 `Hegel Machine/docs/phase3_m25_external_status.md`。

其他路径一律不在 allowlist。尤其不得在 Commit B 修改 Python/Rust implementation、
tests、golden、normative documents、operator scripts 或依赖锁文件。若出现任何 executable
change 或非 allowlisted path，返回：

```text
FAIL_PUBLICATION_COMMIT_CONTAINS_IMPLEMENTATION_CHANGE
```

此时不得“顺手”扩充 allowlist。必须创建新的 deterministic basis Commit A2，重新做
dual qualification、secret-absence audit 和 formal replay。已有 seed 是否继续保留只能由
compromise/recovery policy 明确决定；绝不能自动重抽。

### 11.3 Commit B 最终检查

发布前由不持有 raw seed/private keys 的 reviewer 确认：

- staged diff 只含上述 allowlist；
- every external formal object 的 `repository_commit_id` 仍为 Commit A；
- public key manifests 与 repo 外留存的 trust anchor 一致；
- 五个 external-input envelopes 和三个 bridge envelopes 都能独立重放；
- marker、seed、private keys、row memberships 和 sealed outputs 均不在 Git index/history；
- Commit B receipt 明确自己只是 publication carrier；
- child 仍为 `NOT_RUN`，run-produced roots 仍为 15 个 `null` slots；
- 没有声称 `COMPLETE`、`DSL_TOO_LARGE`、outside verdict、sink verdict、certificate 或
  ACTIVE promotion。

## 12. Fail-closed operator checklist

### 12.1 handoff 前

- [ ] Commit A SHA-1 来自 checked qualification，而不是未提交工作树。
- [ ] 十项 external-genesis start guard 全为严格 boolean `true`。
- [ ] secret-absence receipt 与 dual report 都从 Commit A 重放。
- [ ] custodian 与 auditor 身份独立。
- [ ] 已准备 repo 外、非 symlink、非 cloud-sync、`0700` secret-state directory。
- [ ] 尚未调用真实 key/seed CSPRNG，尚未创建 marker。

### 12.2 one-shot 内

- [ ] 四把真实 Ed25519 keys 独立生成，epoch 为 0，ID/public key 两两不同。
- [ ] 所有 private-key files 为 `0600`，未进入 argv/env/stdout/stderr/history。
- [ ] exclusive lock 已持有，marker 以 `O_CREAT|O_EXCL` 创建并 fsync 为 `PENDING`。
- [ ] split seed 仅首次生成一次，exactly 32 bytes，文件为 `0600`。
- [ ] Python/Rust calculators 仅通过 FD 3 接收 exactly 32 bytes 后 EOF。
- [ ] calculators 无网络、无 repo 写权限、无 secret output，且公开结果一致。
- [ ] purpose-4 auditor 独立完成固定 parent commit 的完整历史审计。
- [ ] 四个 purpose-1 envelopes、一个 purpose-4 envelope 均为真实单签名。
- [ ] marker 仅在 commitment evidence 完整后原子替换、fsync 为 `COMPLETE`。

### 12.3 execution identity 与发布

- [ ] IDs 已 O_EXCL reservation 并进入 append-only opaque-ID snapshot。
- [ ] actor trust、external-input bundle、candidate 和 bridge statement roots 已重放。
- [ ] purposes 1/2/3 独立签署同一 bridge statement root。
- [ ] final manifest V2 和 `M3RunGenesisV1` 已重放，15 output slots 全为 `null`。
- [ ] 24/24 只产生 qualified `NOT_RUN`，没有 start transition。
- [ ] Commit B changed paths 严格落在 allowlist，executable paths 零变化。
- [ ] staged public payload 通过 secret-field lint 和独立人工审计。
- [ ] Commit B 中没有任何测试 key/seed 冒充真实 external evidence。

## 13. 当前交接结论

截至本文写入 repository 时，本 runbook 的 external one-shot 尚未执行。Repo 内没有由本
runbook 产生的真实 seed、private key、external signature、PENDING/COMPLETE marker、
formal root、Commit B public bundle 或 M3 state transition。

下一动作不是让 Codex 代为生成 secret，而是由项目 owner：

1. 固定并核验真正的 Commit A checked dual qualification；
2. 指定独立 custodian、purpose-2/3 attesters 和 independent auditor；
3. 在 repo 外、经过单独安全审查的 one-shot 环境中按本 runbook 执行；
4. 只把 allowlisted、可重放、无 secret 的公开 evidence 交回 Commit B review。
