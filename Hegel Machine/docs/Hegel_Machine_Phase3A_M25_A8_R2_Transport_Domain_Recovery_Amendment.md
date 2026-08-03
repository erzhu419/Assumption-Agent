# Phase-3A M2.5 A8 → R1 → R2 Transport-Domain Recovery Amendment

状态：R2 实现及隔离测试验证已完成；source manifest 最终化、R2 提交/推送、clean-tree
preflight 和新授权链仍待完成。在这些门全部通过前，禁止执行正式 recovery。

Formal 对象的 `repository_commit_id` 仍是 A8
`0af65964235390ce2bebefea7379eaa9c50eda24`。R1
`0349131599a688470c15eded51f942eefeded392` 和 R2 都只是诊断与恢复来源，
不进入 formal roots。

## 事故结论

R1 的首次恢复调用在 Docker actors、source admission 和 formal core 之前 fail
closed，错误为：

```text
FAIL_M25_FORMAL_CEREMONY_LOCKED_OR_RESERVED:
transaction-local live qualification bundle differs from intent
```

持久化的 standalone bundle 与 intent 中的 embedded bundle 没有内容漂移。两者的
canonical payload SHA-256 都是
`b1866e49a3d7aa3b4a649f94a5595591576a0d72e25bd844f280953ace643404`。唯一差异是：
standalone JSON 经 `json.loads` 后保持 `list`，intent 经 `_restore` 后把相同数组表示为
`tuple`。208 个 sequence 节点全部是这一类型差异，键、值、长度和非 sequence
字段差异均为 0。

因此 R2 只把 loader 的内存对象比较改为 canonical transport-domain 比较；原有
canonical payload exact equality、expected SHA-256、schema 和 replay checks 全部保留。不修改
`_restore`，不修改 formal CBOR wire、roots、actor workers 或持久化证据。

## 不可变事实

- 唯一 transaction 仍为 run `e4af9f57c38fb298462ec628c4ed8a03`、ledger
  `ec849e2f1e2e1163cfc450370b25b484`。
- marker 必须仍为 `PENDING`，journal 必须仍为 `RESERVED`。
- R1 failure receipt 没有记录三个 seed 文件各自的历史 inode，因此 R2 不声称从 R1
  失败时刻起的逐文件 lifetime inode continuity。R2 冻结 intent/completion 的内容 hash，
  对 raw seed 只做 stat-only metadata，并把三者的 dev/inode 写入 incident；正式入口必须
  byte-exact 重放该 incident，所以可证明的是 prepare → execute 典礼窗口内的 inode
  continuity，加上既有 recovery anchor/custody continuity。orchestrator 禁止 open、read、
  hash、copy 或 redraw raw seed。
- R1 audit directory 永远只读，必须精确包含
  `preflight.json`、`authorization-request.json`、`authorization.json`、`failure.json`；
  `admission.json` 和 `finalize.json` 必须不存在。
- R2 是 R1 的单一直接子提交；R2 的 source manifest、clean tree、changed paths
  和 blob hashes 必须完全一致。
- 只允许 `REAL_PENDING_RESUME`。禁止 ordinary execute、redraw、abort、post-stage
  recovery 和 `phase3-m3-start`。formal identity entropy 仍为 0。

## R1 失败链

R2 必须重放并绑定以下 R1 raw receipts：

| record | raw SHA-256 |
| --- | --- |
| preflight | `42110a9cfd9a5a5d416bf8fd09cebb5dab7fed38cf2d72a40db291b798856a1e` |
| authorization request | `6ed40f5a116bbf98516d003e2761640b8029141e383ae8ea1291cb0307f7af05` |
| authorization | `14a108b28bf7ee4e47c28d292238b62c62a8b302dbabae7f1a57973a63711569` |
| failure | `d4b7be4432b4101de5aab1693e37ae5769d1587155d634b4e746fee60109168a` |

R1 failure 的 self receipt 是
`ce8948da791a1c42d934ec4a3752ba4bbe5484f96add28f9df5e094444ecb658`。旧 R1 入口在发现
`failure.json` 或 `finalize.json` 后必须 terminal reject，不得原地重试。

## Attempt-2 audit lifecycle

R2 是 `recovery_attempt_ordinal = 2`，动作类型为
`CODE_AMENDMENT_RECOVERY_CONTINUATION`，不是 R1 retry。它使用新的固定 repo-external
audit directory：

```text
/home/erzhu419/.local/state/hegel-machine/
phase3-m25-0af65964235390ce2bebefea7379eaa9c50eda24/
recovery-audit-r2-e4af9f57c38fb298462ec628c4ed8a03-attempt-2
```

`prepare-authorization` 以 fail-if-exists 方式创建 mode `0700` 目录，然后按顺序写入
canonical、mode `0600`、`O_EXCL`、file-fsync + directory-fsync 的 records。典礼时序是：

1. `preflight.json` 绑定 clean R2 source 和 manifest；
2. `incident-diagnostic.json` 绑定 R1 failure chain、transport 根因、事故现场 hashes、
   `PENDING / RESERVED` 和 raw-seed stat-only metadata；
3. `authorization-request.json` 绑定上述两个 records 和 attempt ordinal 2；
4. 独立 `authorize-fixed-transaction` invocation 用新 owner confirmation 写
   `authorization.json`；R1 authorization 已消费，不得复用；
5. `recover-fixed-complete-seed` 在 actor construction 和 formal lock 之前重放全部门，
   然后持久化 `attempt-start.json`；
6. `attempt-start.json` 是一次性消费点。一旦存在，attempt-2 无论成功或失败都不得
   再进入；失败后只能另立 R3、ordinal 3、新 audit 和新授权；
7. source admission 在 core 前持久化，core 入口再做一次 exact replay；
8. 成功时 `finalize.json` 必须绑定 R1 failure、incident、attempt-start、admission、
   public evidence/promotion/publication/seed-verification hashes，并重放得到
   `24/24 + NOT_RUN + 15 output roots null`；
9. 任何 attempt-start 后的异常只写一个 `failure.json`，不删除或改写任何旧 record。

## 执行前必须通过的测试

- exact incident golden：旧比较必然拒绝，transport-domain 比较必须通过；
- scalar flip、row reorder/drop/insert、bytes nibble flip、wrong SHA、noncanonical JSON、
  symlink 和 wrong mode 均在 actor 边界前 fail closed；
- R1 audit 在 prepare、授权、所有负例和 finalize 后 bytes/mode/inode 不变；
- `attempt-start` 持久化之前不调用 actor/acquire，持久化后任何失败均永久禁止二次
  invocation；
- raw seed 的 `os.open`、`Path.read_*` 和 hash helpers 在测试中全部设为 fatal，R2
  全链路调用计数必须是 0；
- 最终 public replay 必须为 24/24、`NOT_RUN`、15 个 output roots 全 null、
  actor containers 与四个 key volumes 已清理，且 M3 从未启动。

64 个旧 test files 和 R2 新测试使用四个独立 shard 并行执行；会真正使用 fixed
custody/Docker state 的 complete-only recovery 仍严格串行，且只允许一次。

## 命令形态

以下命令只能在 R2 commit 已推送、side worktree clean、manifest 和全部测试通过后使用。
`prepare-authorization` 和正式 recovery 必须接收相同的 custody/public paths；执行入口
会重算 incident record 并要求逐字段一致。

```text
python3 -m hegel_machine.phase3_m25_a8_recovery_cli_r2_v1 preflight

python3 -m hegel_machine.phase3_m25_a8_recovery_cli_r2_v1 prepare-authorization \
  --audit-directory <fixed-attempt-2-audit-dir> \
  --custody-directory <existing-A8-custody> \
  --public-evidence-output <reserved-evidence-path> \
  --promotion-output <reserved-promotion-path>

python3 -m hegel_machine.phase3_m25_a8_recovery_cli_r2_v1 \
  authorize-fixed-transaction \
  --audit-directory <same-attempt-2-audit-dir> \
  --owner-confirmation \
  AUTHORIZE_A8_R2_ATTEMPT_2_COMPLETE_ONLY_REAL_PENDING_RESUME

python3 -m hegel_machine.phase3_m25_a8_recovery_cli_r2_v1 \
  recover-fixed-complete-seed \
  --custody-directory <existing-A8-custody> \
  --rust-formal-replay-binary <existing-A8-formal-binary> \
  --rust-bridge-dag-replay-binary <existing-A8-bridge-binary> \
  --rust-bridge-dag-qualification-report <existing-A8-report> \
  --public-evidence-output <reserved-evidence-path> \
  --promotion-output <reserved-promotion-path> \
  --audit-directory <same-attempt-2-audit-dir>
```

最后一条命令一旦写入 `attempt-start.json` 就永久消费 attempt-2，禁止由 shell
loop、test retry plugin 或人工重复执行。

R2 不开启 M3。24/24 仅表示 M3 已获准但仍为 `NOT_RUN`。
