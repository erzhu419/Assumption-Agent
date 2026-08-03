# Phase-3A M2.5 External Genesis Operator Runbook

**文档性质**：外部操作交接稿；不是可执行程序、签名请求或授权凭证
**machine freeze ID**：`hegel-freeze-p2b-p3-v1.1.2`
**child DSL ID**：`hegel-old-dsl-v1.1.0`
**当前状态**：`NOT_EXECUTED`
**当前 child state**：`NOT_RUN`
**当前 M3 gates**：执行前仍为 `14/24`

本 runbook 只说明：Commit A 的 checked dual qualification 真正通过后，四个隔离的
technical actors（custodian、Python bridge attester、Rust bridge attester 和
parent-absence auditor）
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
detached Commit-A snapshot，而不是 live worktree。Rust 端必须使用 Commit A 绑定的
digest-pinned OCI image、`env -i` whitelist、独立 runtime/build seccomp，以及
`--pull=never --network=none`。宿主 Cargo cache 只允许作为一次性只读取材来源：每个
`.crate` 必须先与 `Cargo.lock` checksum 比较，再安全解包为 run-private vendor
snapshot；host Cargo home、registry cache/index 和 unpacked source 均不得 mount 进
container。Cargo/Rust 路径不得由调用者指定；container 内 cargo/rustc binaries、版本、
full compiler probe、OCI manifest/image ID、vendor root 和 environment roots 必须匹配
approved OCI-toolchain policy。fresh binary 必须先通过 Python/Rust/golden exact replay，
才能原子写入 `DEFAULT_RUST_BINARY`，并从该持久化路径再次离线重放。调用者提供的任意
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

## 3. 隔离进程与四个 purpose

### 3.1 独立性

> **Owner policy amendment (2026-08-02).** 早期版本的 human-only 最低安排已由
> `Hegel_Machine_Owner_Accepted_Container_Technical_Actor_Eligibility_Amendment_v1.md`
> 接管。满足该 amendment 全部 live probes 的四个离线、digest-pinned container purpose
> actors，可作为本项目 Gate 15–24 的正式 technical actors。它们必须披露同一 admin
> controller 与不存在组织/真人独立性；单纯换 key、启动普通 Docker 或只声明 launch flags
> 仍不合格。Formal wire、purpose IDs、signatures 与 gates 不变。旧版要求不同真人参与的
> 段落属于历史安排，不再是 Gate 15–24 的准入条件。

正式安排是四个同时存活、purpose-private 的 Docker actors。它们由同一 owner/Codex
orchestrator 启动，但分别拥有不同 container ID、PID/mount/network/IPC/UTS namespaces、
只读输入快照、公开输出目录和 daemon-managed local key volume。每次 keygen、seed、签名
或 marker 操作前，同一 worker process（Rust 为实际执行签名命令的 parent shell）必须产生
fresh operation-bound receipt，绑定 purpose-local 递增序号、16-byte nonce、完整请求输入
摘要、container/daemon/profile/run identity；host 必须 strict-JSON 重放 receipt 并重新核对
Docker inspect。四个 actor 的 `qualify-only` receipt 全部通过之前，任何敏感操作均被拒绝。

该安排的权威披露固定为：`same_admin_controller=true`、
`organizational_independence=false`、`independent_human_actors=false`、
`technical_role_independence=true`、`owner_accepted_threat_model=true`、
`remote_attestation=false`、`hardware_key_nonexportability=false`。人工复核可以作为可选治理层，但不得成为 Gate 15–24
或后续恢复流程的阻塞依赖。

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

选择一个真实、非 symlink、位于 repository 之外且不在 cloud-sync 下的持久目录。它
不得位于 `/tmp`、DrvFS/9p、NFS/CIFS/FUSE 或 `/mnt/c`；建议位于
`$HOME/.local/state/hegel-machine/` 下的 Linux-local durable filesystem。开始前 validator
必须执行 ownership/mode、effective mount、file fsync、atomic rename 和 directory fsync
探针。与之相反，container input snapshots、offline Rust build 和 Docker client config
只允许放入经过验证的 Linux-local `/tmp` 私有目录：

```text
secret-state directory: mode 0700
every private-key file: regular file, inside its purpose-private Docker volume, mode 0600
split-seed file: regular file, inside secret-state, mode 0600, exactly 32 bytes
marker/lock/intent/completion/reservation state containing custody metadata: mode 0600
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

然后再次 fsync file 和 directory。`PENDING` 状态下发生中断时，不得删除 marker、由
普通 `execute` 自动重试或重抽 seed。恢复只能走独立的 explicit recovery API：它必须
先重开并锁定 canonical mode-0600 persistent lock，核验同一 run/ledger reservation、三个
output reservation、transaction journal、PENDING marker，以及 run/purpose/basis/profile/
image labels 完全相等的四个 purpose-private volumes；purpose-1 key 不存在时立即终止，
不得重新生成。

Seed recovery 三态冻结为：

1. `PENDING` 已 durable，但 intent、seed inode 和 completion receipt 三者全不存在：这证明
   worker 尚未进入 CSPRNG boundary；**仅 explicit recovery** 可先 durable 写 intent、预建并
   fsync zero-length seed inode，然后进行唯一一次 `getrandom(32)`，报告
   `REAL_FIRST_GENESIS_AFTER_PENDING_NO_INTENT`；
2. exact intent + mode-0600 exact-32 seed + exact mode-0600 completion receipt：只重放两个
   calculators，CSPRNG 调用数为零，报告 `REAL_PENDING_RESUME`；
3. intent 已存在而 completion 缺失/无效，或 seed 缺失、zero/partial/oversized、symlink、
   wrong-mode/wrong-content：永久 `FAIL_M25_SPLIT_SEED_UNRECOVERABLE_NO_REDRAW`，不得
   修补、删除或 redraw。

`split_seed_generation.complete` 只有在 seed file 与 directory 均 fsync 后才以 `O_EXCL`
创建；它绑定 schema、attempt=1、intent SHA-256、seed length 与 domain-separated seed
commitment。Exact 32-byte seed 本身不够，没有 valid completion receipt 就不得恢复。

`COMPLETE` 也永久阻止第二次实例化。Marker 防止 redraw，不是 seed 本身，也不能进入
Commit B。

### 4.3 purpose-private volume 生命周期

四把 private keys 分别保存在四个 persistent local Docker volumes；volume name 与 labels
绑定 run ID、purpose ID、Commit A、actor-profile SHA-256 与 pinned-image SHA-256。新建
volume 先由同一 pinned local image 的一次性 root helper 在 `--pull=never`、
`--network=none`、read-only root、frozen seccomp 下，以 exactly `CAP_CHOWN` 设置
`65534:65534/0700`；随后由 cap-zero 的 `65534:65534` helper 完成 stat/write/remove live
probe。初始化 receipt 是公开元数据，不含 key。

Cleanup 冻结为：marker absent 时 destroy-and-verify（尚无不可逆 seed choice）；PENDING
时 retain-and-verify；marker malformed/unreadable 时 retain-and-fail；COMPLETE 只有在 public
evidence 已 durable stage、从磁盘 reload 并 prospective replay 一致后才可
destroy-and-verify。这里销毁的对象严格只是四个 actor key volumes；purpose-1 custody
中的 raw split seed、intent、completion receipt 与 COMPLETE marker 按 seed continuity
policy 保留，直到另一个明确冻结的 continuity destruction gate。实现和文档不得声称
“全部 private state 已销毁”。任何 container/labelled descendant/key volume 未消失都
必须使 publication 失败。

### 4.4 post-stage fresh-process 恢复

普通 `execute` 永不自动猜测恢复。只要 custody 中已经存在 transaction lock，operator
必须保留原 custody、public output directory 和 Docker volumes，并显式调用：

1. 以 `with FormalCeremonyTransactionV1.rehydrate_post_stage_v1(...) as transaction`
   获取并保证释放同一 persistent lock；
2. 核对 `transaction.recovery_phase`；
3. 使用同一 Commit-A、同一 bound Rust binary 构造 authoritative Docker actors；
4. 在该 context 内调用 `continue_post_stage_transaction_recovery_v1(...)`。

Lock v4 绑定三个 absolute output paths 和 stage directory；三个 reservation 各自绑定
output kind/path。恢复只接受原 run/ledger reservations，绝不生成第二组 opaque IDs。
Stage 中 evidence、promotion、receipt、journal 必须全部是 canonical exact bytes；最终输出若
已存在，只能是 stage bytes 的 write-order prefix。任何 alternate path、extra stage file、
missing reservation、非 prefix publication、marker/receipt/opaque-ID drift 都终止恢复。

允许的 phase 与 action 是：

| `recovery_phase` | 唯一允许动作 |
|---|---|
| `STAGED_PENDING` | 要求完整 intent/seed-inode/completion；恢复已有 purpose-1 key；只允许 `REAL_PENDING_RESUME`；FD-5 frames 与 stage 完全相等后完成 marker |
| `MARKER_COMPLETE_CLEANUP_STATUS_UNKNOWN` | 不再启动 signer；校验并删除仍存在的 exact run-labelled actor/volume subset，证明全部 absent |
| `ACTORS_ABSENT` | 再次证明 actor/volume absent，然后开始 publication |
| `PARTIAL_PUBLICATION` | 校验 existing prefix exact，只以 `O_EXCL` 补齐缺失文件 |
| `ALL_PUBLIC_OUTPUTS_UNJOURNALED` | 校验三文件 exact，清理剩余 exact reservations，fsync 后 journal `PUBLISHED` |
| `PUBLISHED` | 再次 replay exact public bytes；不修改 M3 state |

Post-stage PENDING 不允许 `REAL_FIRST_GENESIS_AFTER_PENDING_NO_INTENT`，也不允许重新 keygen
或重新签名。恢复完成仍然只是 `24/24 / NOT_RUN`；不会隐式执行 `phase3-m3-start`。

### 4.5 pre-seed exact abort 与 output-path retirement

只有 exact `RESERVED`、marker/intent/raw-seed/completion/final-output 全部不存在时，operator
才可显式调用 `abort_preseed_reserved_transaction_v1(...)`。实现先取得 host recovery-anchor、
custody-directory、public-parent 与 persistent-lock 的 liveness locks，然后 durable 安装一个
canonical immutable abort plan；plan 绑定 Commit-A、run/ledger、daemon/profile/audit roots、
全部绝对路径、父目录与目标 inode identity、payload hash、唯一删除顺序和 forbidden seed
state。任何 POSIX 删除都不得先于 plan。

plan durable 后才允许 sealed Docker backend 对 exact run-labelled containers/volumes 执行
inspect/remove/inspect，并 durable 写入 actor-absence receipt。每次重入都重新执行 live
absence 检查。删除进度只可由“已缺失的 exact prefix + 仍精确存在的 suffix”推断；每一步只
使用 plan 内的单路径 `unlink`/`rmdir`，随后立即 fsync parent。Abort plan 倒数第二删除，
persistent lock 最后删除，禁止 glob、递归删除或进度 counter。

在开始删除前，public output parent 会永久保留 mode-0600 canonical terminal tombstone，绑定
run/ledger、plan hash、actor-absence hash 和原 output paths。即使进程在最后 lock unlink 与
parent fsync 之间退出，重入仍可从 tombstone 恢复 identity、重新核对 Docker absence 并完成
fsync。与此同时，evidence、promotion 与 derived publication-receipt 三个物理路径各自具有
role-independent deterministic retirement marker；任一旧路径即使在新组合中改作另一角色也
必须 fail closed。后续正式运行必须选择全新的三路径；不得删除/归档 tombstone 或 marker 后
偷偷复用旧路径。它们不是 formal evidence、不会提升 gate，也不含 seed/key/signature。

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

隔离的 purpose-4 technical auditor 使用自己的 purpose-4 key 对固定 parent commit
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
purpose-1 private key；custodian 不得代替 purpose-4 actor 自签 parent-absence claim。

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

### 11.1 发布前 secret lint 与隔离重放

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

该字段 lint 不是“任意字节绝无秘密”的证明。发布前的强制检查由不持有 raw seed/key 的
purpose-4 isolated actor 与 host strict replay 共同完成：检查 private-key headers、路径、
权限、Git staged blobs、changed paths 和生成日志；host 还必须重放 public payload 的
canonical bytes/self-hash。人工复核是可选治理复核，不是 formal gate blocker。

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

发布前由不持有 raw seed/private keys 的 purpose-4 isolated actor 生成审计 receipt，host
对 receipt 与公开输入做独立 strict replay，确认：

- staged diff 只含上述 allowlist；
- every external formal object 的 `repository_commit_id` 仍为 Commit A；
- public key manifests 与 repo 外留存的 trust anchor 一致；
- 五个 external-input envelopes 和三个 bridge envelopes 都能独立重放；
- marker、seed、private keys、row memberships 和 sealed outputs 均不在 Git index/history；
- Commit B receipt 明确自己只是 publication carrier；
- child 仍为 `NOT_RUN`，run-produced roots 仍为 15 个 `null` slots；
- 没有声称 `COMPLETE`、`DSL_TOO_LARGE`、outside verdict、sink verdict、certificate 或
  ACTIVE promotion。

可执行入口与两阶段自引用处理见
`Hegel_Machine_Phase3A_M25_Commit_B_Staged_Publication_Audit_v1.md`：先用
`render-status` 从其余十项公开角色确定性生成唯一 status（随后仍以 index 为权威），
再用 `prepare` 审计除唯一 audit receipt 之外的完整 Git index 候选；暂存该 receipt 后，
必须再用 fresh purpose-4 actor 重放 receipt-excluded 候选并逐字节匹配 staged receipt，
然后由另一 fresh actor 执行 receipt-inclusive `finalize-index`；最终 receipt 只能写到 repo 外
或 stdout。Commit B 生成后再执行只读 `verify-commit` parent/tree replay。这三步均不
新增 formal gate，也不启动 M3。该证据边界是 owner-controlled self-consistent
transcript，不是 remote attestation，也不主张对同一管理员不可伪造。

## 12. Fail-closed operator checklist

### 12.1 handoff 前

- [ ] Commit A SHA-1 来自 checked qualification，而不是未提交工作树。
- [ ] 十项 external-genesis start guard 全为严格 boolean `true`。
- [ ] secret-absence receipt 与 dual report 都从 Commit A 重放。
- [ ] 四个 purpose actors 已按 owner amendment 在不同容器/命名空间中通过 live
  技术隔离资格化；报告如实披露同一 admin controller 且无组织/真人独立性。
- [ ] 已准备 repo 外、非 symlink、非 cloud-sync、`0700` secret-state directory。
- [ ] 尚未调用真实 key/seed CSPRNG，尚未创建 marker。

### 12.2 one-shot 内

- [ ] 四把真实 Ed25519 keys 独立生成，epoch 为 0，ID/public key 两两不同。
- [ ] 所有 private-key files 为 `0600`，未进入 argv/env/stdout/stderr/history。
- [ ] exclusive lock 已持有，marker 以 `O_CREAT|O_EXCL` 创建并 fsync 为 `PENDING`。
- [ ] intent 与 zero-length seed inode 在唯一 CSPRNG 前已 durable；seed exactly 32 bytes、
  mode `0600`，completion receipt 在 seed+directory fsync 后创建并验证。
- [ ] Python/Rust calculators 仅通过 FD 3 接收 exactly 32 bytes 后 EOF。
- [ ] calculators 无网络、无 repo 写权限、无 secret output，且公开结果一致。
- [ ] purpose-4 auditor 独立完成固定 parent commit 的完整历史审计。
- [ ] 四个 purpose-1 envelopes、一个 purpose-4 envelope 均为真实单签名。
- [ ] public evidence 已 serialized、durable staged、从磁盘 reload 并 prospective replay
  一致；marker 仅在此后原子替换、fsync 为 `COMPLETE`。
- [ ] COMPLETE 后四个 actor containers/labelled descendants/private volumes 全部删除并
  验证 absent，才开始 final publication。

### 12.3 execution identity 与发布

- [ ] IDs 已 O_EXCL reservation 并进入 append-only opaque-ID snapshot。
- [ ] actor trust、external-input bundle、candidate 和 bridge statement roots 已重放。
- [ ] purposes 1/2/3 独立签署同一 bridge statement root。
- [ ] final manifest V2 和 `M3RunGenesisV1` 已重放，15 output slots 全为 `null`。
- [ ] 24/24 只产生 qualified `NOT_RUN`，没有 start transition。
- [ ] Commit B changed paths 严格落在 allowlist，executable paths 零变化。
- [ ] staged public payload 通过 secret-field lint、purpose-4 isolated audit 与 host strict
  public-payload/changed-path replay；人工复核如有，仅作可选治理层。
- [ ] Commit B 中没有任何测试 key/seed 冒充真实 external evidence。

## 13. 当前交接结论

截至本文写入 repository 时，本 runbook 的 external one-shot 尚未执行。Repo 内没有由本
runbook 产生的真实 seed、private key、external signature、PENDING/COMPLETE marker、
formal root、Commit B public bundle 或 M3 state transition。

下一动作由项目 owner 已授权的离线容器技术角色流程执行：

1. 固定并核验真正的 Commit A checked dual qualification；
2. 对 purposes 1–4 运行 digest-pinned、`--pull=never`、`--network=none` 的 live
   隔离资格化，并绑定定制 seccomp 与全部负向探针；
3. 在 repo 外 `0700` secret state 与 purpose-private persistent-volume 容器内执行首次
   seed/key genesis、独立 replay 与签名；
4. 只把 allowlisted、可重放、无 secret 的公开 evidence 交回 purpose-4 + host strict
   Commit B publication replay；可再做非阻塞的人工治理复核。

该流程允许 Codex 作为 owner-authorized orchestrator 启动容器，但 orchestrator
本身不持有 purpose 签名密钥、不读取 raw split seed，也不能以自身输出
代替容器内可重放证据。
