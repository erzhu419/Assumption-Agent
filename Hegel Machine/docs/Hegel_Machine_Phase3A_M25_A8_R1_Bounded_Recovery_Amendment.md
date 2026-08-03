# Phase-3A M2.5 A8 → R1 Bounded Recovery Amendment

状态：实现草案；在 R1 被提交、双端测试通过且 manifest hash 填充完成前，禁止运行正式恢复。

本 amendment 只处理 A8 (`0af65964235390ce2bebefea7379eaa9c50eda24`)
首次 formal genesis 在构造第二个 opaque-ID registry snapshot 时 fail closed 的事故。
Python 端错误地把 strict AST CBOR bytes 再作一次 CBOR bytestring 编码；Rust 端在计算
新增 singleton root 时把 sequence `1` 的记录重新当作从 `0` 开始的独立 registry 校验。
R1 分别修正这两个实现，并增加 second-append 回归。

## 不变量

- formal `repository_commit_id`、actor qualification、runtime bindings 和持久化 A8 Rust
  binaries 全部继续绑定 A8；R1 只作为 recovery provenance。
- 唯一允许的 transaction 是 run
  `e4af9f57c38fb298462ec628c4ed8a03`、ledger
  `ec849e2f1e2e1163cfc450370b25b484`。
- R1 必须是 A8 的单一直接子提交，`HEAD` 必须选中 R1，index/worktree 必须 clean，diff
  和所有 changed blobs 必须与 manifest 完全相等。
- 只接受 `PENDING + RESERVED + intent + exact-32-byte seed + completion`，worker mode
  必须是 `REAL_PENDING_RESUME`。禁止
  `REAL_FIRST_GENESIS_AFTER_PENDING_NO_INTENT`。
- formal identity entropy 为 0；container operation nonce 仍可生成，但它不是 formal
  identity。orchestrator 永不打开、读取或 hash raw seed inode。
- 禁止 ordinary `execute`、redraw、preseed abort、post-stage recovery 和
  `phase3-m3-start`。

## Repo-external audit lifecycle

Audit directory 固定为
`/home/erzhu419/.local/state/hegel-machine/phase3-m25-0af65964235390ce2bebefea7379eaa9c50eda24/recovery-audit-e4af9f57c38fb298462ec628c4ed8a03`，
必须位于 repo 外、caller-owned、mode `0700`，且不得与 custody、stage 或 public-output
parent 相同或互为祖先。每个 record 使用 canonical
JSON、`O_EXCL`、mode `0600`、file fsync + directory fsync；`receipt_sha256` 是删除该字段后
canonical bytes 的 SHA-256。

1. `prepare-authorization` 写 `preflight.json` 与 `authorization-request.json`；
2. 独立的 `authorize-fixed-transaction` invocation 在显式 owner confirmation 后写
   `authorization.json`；
3. `recover-fixed-complete-seed` 重新验证 clean R1、authorization、固定 transaction、
   三件套 metadata，并写 `admission.json`；
4. 只通过 recovery-only callback 进入 existing prestage core，并强制调用
   `resume_post_stage_seed_split()` 的 complete-only branch；
5. 成功后独立 replay public evidence，核验 `24/24 + NOT_RUN`、官方 COMPLETE marker、
   seed commitment、custodian key ID 和 publication receipt，再写 `finalize.json`；
6. 任何异常写 `failure.json`（若 slot 尚未占用）并原样 fail closed。

## 操作形态（R1 commit 与 manifest 完成后才可用）

```text
python3 -m hegel_machine.phase3_m25_a8_recovery_cli_v1 preflight
python3 -m hegel_machine.phase3_m25_a8_recovery_cli_v1 prepare-authorization \
  --audit-directory <repo-external-0700-dir>
python3 -m hegel_machine.phase3_m25_a8_recovery_cli_v1 authorize-fixed-transaction \
  --audit-directory <same-dir> \
  --owner-confirmation AUTHORIZE_A8_R1_COMPLETE_ONLY_REAL_PENDING_RESUME
python3 -m hegel_machine.phase3_m25_a8_recovery_cli_v1 recover-fixed-complete-seed \
  --custody-directory <existing-A8-custody> \
  --rust-formal-replay-binary <existing-A8-formal-binary> \
  --rust-bridge-dag-replay-binary <existing-A8-bridge-binary> \
  --rust-bridge-dag-qualification-report <existing-A8-report> \
  --public-evidence-output <reserved-evidence-path> \
  --promotion-output <reserved-promotion-path> \
  --audit-directory <same-dir>
```

这些命令不包含、也不得随后追加 `phase3-m3-start`。
