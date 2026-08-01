# Phase-3 shrink-1 后续 formal bridge / seed continuity 待定问题

**文档状态**：`DRAFT_FOR_DECISION`<br>
**当前 child**：`hegel-old-dsl-v1.1.0` / `hegel-freeze-p2b-p3-v1.1.0`<br>
**当前机器状态**：`SHRINK1_SUBSET_QUALIFIED_M3_BLOCKED`，child execution=`NOT_RUN`

## 已完成且不再开放的事实

```json
{
  "aggregate_policy": "SPARSE_PRESERVING",
  "active_ids": [0, 1, 5],
  "tombstoned_ids": [2, 3, 4],
  "removed_error": "REJECT_REMOVED_AGGREGATE_MAP",
  "surviving_ast_bytes_and_hash_stable": true,
  "shrink1_source_count": 25872,
  "python_rust_accepted_unique_count": 25872,
  "accepted_set_commitment": "sha256:653fcb9428684cfed11c3f2345ac95ed98ded6e31564c9eeabf97c57ee71a7e9",
  "first_out_of_budget_witness": null,
  "subset_is_complete_closure": false,
  "formal_roots": null,
  "m3_entry_allowed": false
}
```

当前 24 个 M3 gates 中前 14 个已满足。以下问题来自实现审计发现的真实缺口，不能
通过代码自行猜测或伪造。

## 问题 1：没有历史 split seed 时，“必须复用”如何落地

仓库中不存在以下任何对象：

- parent `K_split` 或其 domain-separated commitment；
- realized discovery / validation / sealed row-allocation payload；
- parent split binding manifest；
- append-only hidden-access ledger；
- custodian continuity attestation。

因此当前实现既没有重抽 seed，也没有声称 reuse，统一 fail-closed。请确定：

1. 是否把此前状态正式定义为 `SPLIT_SPEC_FROZEN_BUT_SEED_NEVER_INSTANTIATED`，允许
   custodian 首次生成 seed；该行为明确称 first instantiation，而不是 redraw；
2. 还是必须从外部找回历史 seed/commitment，否则发布新的 target/split version；
3. 若允许首次实例化，请冻结 exact seed length、CSPRNG、commitment domain、commitment
   wire、custodian signature wire 与 access-ledger genesis record。

建议：若无法提供可验证的历史 commitment，不应事后声称“同一 seed 被复用”。

## 问题 2：缺失 parent binding manifest 的合法表示

批准稿要求每个新 manifest 记录 `parent_binding_manifest_id`，但父版本从未生成 typed
binding manifest，只有 target/control diagnostic content IDs。请二选一：

1. 允许 `parent_binding_manifest_id = null`，并额外记录
   `legacy_parent_payload_source_id` 与 `parent_manifest_absence_proof`；
2. 允许创建 retrospective parent manifest，但必须明确它不能被写成当时已存在的
   precommitment，也不能回溯改变 parent evidence。

当前实现采用 1 的 fail-closed diagnostic 形式，尚未称为 M3 commitment complete。

## 问题 3：binding / transition artifacts 的 exact formal wire

决策稿冻结了逻辑字段，但没有冻结以下对象的 numeric-tag CBOR schema、hash domain、
字段顺序与 schema-version machine string：

- approval manifest；
- target/control DSL-binding manifests；
- split/custodian/seed-continuity manifests；
- `DslShrinkTransitionV1`；
- M3 execution manifest。

请给出这些对象的 exact numeric array schema 与 ContentHash domain。当前 JSON artifacts
明确只是 diagnostic publication，不能直接充当 formal CBOR root leaf。

## 问题 4：publication roots 与 M3 input/output roots 的完整分类

批准稿 §6.7 的 publication list 漏写 `canonical_cbor_profile_root` 与 sink role-specific
roots，而 §8 又要求它们。请确认 formal bridge exact input-root set至少为：

```text
dsl_spec_root
operator_semantics_root
identifier_registry_root
canonical_ast_schema_root
canonical_cbor_profile_root
diagnostic_formal_bridge_root
outside_target_universe_root
outside_target_truth_root
null_control_universe_root
null_control_truth_root
```

并确认 program/output/chunk/match/receipt roots 是 run-produced outputs，不可能作为
`NOT_RUN -> RUNNING` 的前置 input roots。还需冻结每个 root 的 leaf record schema 与
RFC6962 ordering key。

## 问题 5：M3 状态机与 role-specific receipt wire

当前旧 receipt schema 不能完整表达 `NOT_RUN -> RUNNING`、
`INCONCLUSIVE_EXECUTION`、execution manifest、双独立实现环境及 odd/sink 两套独立
output/match receipts。请冻结：

- M3 run/state schema 与合法单向迁移；
- Python/Rust execution receipt schema；
- role-specific evaluation receipt schema；
- dual replay agreement schema；
- `COMPLETE` 和 `DSL_TOO_LARGE` 的 exact field-presence/nullability rules。

在这些 wire 冻结之前，只能开发非权威 enumerator dry run，不能启动 formal M3。

## 请 GPT 输出

请逐项给出可直接写入下一份 normative amendment 的 machine-readable 决定，不要只给
原则说明。尤其需要 exact schema IDs、numeric tags、hash domains、nullability、failure
codes 与 state-transition guards。
