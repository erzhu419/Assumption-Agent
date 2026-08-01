# Phase-2B / Phase-3 冻结问题记录（historical pre-v1.0.2）

> **STATUS: `SUPERSEDED_FOR_NORMATIVE_DECISIONS`**
>
> 本文件保留 target outcome 之前发现 specification gaps 的审计轨迹。所有旧待消歧项已由
> [v1.0.2 strict canonical / certificate bridge freeze](Hegel_Machine_Strict_Canonical_AST_CBOR_Certificate_Bridge_Freeze_v1.0.2.md)
> 在规范层解决；当前 go/no-go 以
> [Phase-3 readiness resolution](Hegel_Machine_Phase3_Freeze_Readiness_Resolution.md)
> 为准。下方旧问题正文不是当前 normative truth。

## M1/M2 执行后的权威状态

| 层级 | 状态 |
|---|---|
| surface + strict acceptance/certificate specification | 已冻结 |
| Python/Rust strict implementation + shared golden vectors | 两端各 **48/48 PASS** |
| bounded strict capacity replay | `hegel-old-dsl-v1.0.0` 在 50,000 syntactic budget 下为 `DSL_TOO_LARGE`；两端各接受 64,680，且全部 unique |
| complete closure / extensional target verdict | 未得到；当前不是 `COMPLETE` |
| formal roots | `null` |
| outside / MDL certificate | 未签发 |

```json
{
  "freeze_version": "hegel-freeze-p2b-p3-v1.0.2",
  "surface_parameter_freeze_complete": true,
  "strict_acceptance_contract_complete": true,
  "normative_parameter_freeze_complete": true,
  "strict_acceptance_implementation_verified": true,
  "python_shared_vectors_passed": "48/48",
  "rust_shared_vectors_passed": "48/48",
  "accepted_strict_canonical_count_python": 64680,
  "accepted_strict_canonical_count_rust": 64680,
  "accepted_unique_count_python": 64680,
  "accepted_unique_count_rust": 64680,
  "diagnostic_set_commitment": "sha256:c1a02a66a8d6d8f75204cb3daf03ab0b01c2b3b8e486d0ab3d481ee3be43c930",
  "first_out_of_budget_ordinal": 50001,
  "first_out_of_budget_ast_hash": "sha256:7c7f786c2cc57d31506b3c61d162d175c7f69a2878a089c72c9d053694cba948",
  "executed_closure_status": "DSL_TOO_LARGE",
  "complete_closure_enumerated": false,
  "extensional_target_verdict": null,
  "formal_roots": null,
  "hidden_sink_formal_verdict_allowed": false,
  "outside_certificate_issued": false,
  "mdl_certificate_allowed": false,
  "phase2b_formal_exit": false,
  "active_promotion_allowed": false,
  "required_next_action": "PUBLISH_SHRUNK_OLD_DSL_VERSION_USING_FROZEN_STEP_1"
}
```

旧问题的规范结论现为：`control_volume_primary_only_v1` 只是 deprecated source alias，
唯一 machine ID 是 `scope_primary_only_v1`，strict canonicalizer 必须拒绝 alias；v1 不存在
implicit Bit coercion，XOR sanity witness 必须显式使用两个 `bit_to_scalar`；strict AST/CBOR、
normalization/count boundary、九组 certificate/MDL wire 均为 `RESOLVED_IN_SPEC`。其中
strict parser/typechecker/canonicalizer 已通过 M1 双实现验证；certificate/MDL 的正式
archive、bridge、trust chain 与 replay 仍未执行。binary XOR 的正式 language verdict
仍为 `null`，因为 M2 只证明 syntactic budget overflow，没有执行 extensional target replay。

证据见
[dual strict gate artifact](../artifacts/phase3_dual_strict_gate_v1.json) 与
[dual strict capacity replay artifact](../artifacts/phase3_dual_strict_capacity_replay_v1.json)。
共同的 diagnostic set commitment 不是 formal RFC6962 root；第 50,001 个 hash 只证明
50,000 syntactic budget overflow，不证明完整 closure cardinality。

下一动作是 `PUBLISH_SHRUNK_OLD_DSL_VERSION_USING_FROZEN_STEP_1`：发布新 DSL version，
删除 `mean_v1`、`min_v1`、`max_v1`，重建 target/validation commitments，并让新版本从
`NOT_RUN` 开始。当前仍为 NO-GO 的是 extensional target verdict、target synthesis、
hidden-sink formal verdict、outside/MDL certificate、Phase-2B formal exit 与 ACTIVE；
formal roots 必须保持 `null`。

---

## 历史问题记录（pre-v1.0.2）

原先九组实验设计问题已由
`Hegel_Machine_Phase2B_Phase3_Exact_Freeze_Decisions.md` 回答。配额、target、value
grids、typing、budgets、MDL 参数和高层证书路线继续冻结为
`hegel-freeze-p2b-p3-v1.0.1`。v1.0.1 是 implementation-audit amendment：保留
`411876909552964556` 作为 master/bootstrap seed，并把 sklearn `random_state` 冻结为
domain-separated SHA-256 → uint32 的 `2611585425`；不可执行地把 64-bit seed 直接传给
sklearn 的 v1.0.0 已被 supersede。独立实现审计同时发现：strict canonical AST/CBOR
acceptance 与若干 certificate wire identities 没有被决定稿唯一确定。它们不能由实现
自行猜测，也不能在看到 target outcome 后补写。

本文件是对决定稿的 **implementation-audit amendment**。若旧文中的“已经完整冻结”可被
读成 strict acceptance contract 也已完成，应以这里的机器状态为准：当前里程碑名为
**Phase-3A Bounded Frozen-Closure Adequacy**，且只有 surface 参数冻结完成。

```json
{
  "surface_parameter_freeze_complete": true,
  "strict_acceptance_contract_complete": false,
  "normative_parameter_freeze_complete": false
}
```

## 已解决决定

1. Phase-2B main 为 720：每 cell 是 19 个 `unique_scale_answerable` + 1 个
   `admissible_scale_set_answerable`，其余五类各 8，margin 为 `21/18/12/9`。
2. 首个 target 是 `TARGET_P3A_GENERIC_ODD_REDUCTION_V1`，完整 universe 为 480 行，
   split 为 `192/96/192`。二元 XOR 在 executable closure 前只算
   `TARGET_DESIGN_SANITY_ONLY`。
3. hidden sink 是 85-row observed omitted-channel control，不得是 latent sink。
4. 50,000 计 extensional quotient 前的 syntactically canonical programs；
   5,000,000 是 raw operator-application cap。这是预算语义，不是当前 closure 结果。
5. `hegel-old-dsl-v1.0.0` 的 finite domains、registries、operator typing、bottom、
   structural limits、support 和 shrink order 已完成 **surface-parameter freeze**；这不
   等于 strict canonical acceptance 或 normative freeze 已完成。
6. `hegel-mdl-prefix-v1.0.0` 的 code-table 参数、80-digit `log2`、Q32、整数比较和
   invention split 已冻结。
7. Phase-2B 正式规模是 720 main + 240 semantic-conflict challenge；derived suite
   是 496 legal + 76 invalid = 572 pairs。baseline config、bootstrap、footprint、rerun
   与 validation version policy 已确定。
8. covert-channel 高层审计规范已冻结；在完整语义实现前 formal selector 只允许
   `absolute_bound`，`standard_error = STANDARD_ERROR_UNSUPPORTED`。
9. outside/MDL certificate 的高层路线是 canonical CBOR + RFC-6962 + Python/Rust
   双完整 replay + 3/3 Ed25519，但下面列出的 strict schemas/identities 仍待消歧。

正式 claim 只能是：

```text
OUTSIDE_FROZEN_CLOSURE(
  dsl_version,
  bounded_universe_root,
  target_truth_table_root,
  equivalence = exact_extensional
)
```

禁止简写为 `OUTSIDE_LANGUAGE`。

## 当前 capacity evidence 的准确状态

```json
{
  "status": "CONDITIONAL_CAPACITY_LOWER_BOUND_EXCEEDS_BUDGET",
  "constructive_candidate_ast_count": 64680,
  "canonical_program_budget": 50000,
  "diagnostic_representation": "tuple_ast_plus_canonical_json",
  "strict_canonicalizer_acceptance_verified": false,
  "formal_canonical_cbor_archive": false,
  "executed_closure_status": "NOT_RUN",
  "dsl_too_large_claim_allowed": false
}
```

64,680 是 distinct、typed、limit-conforming **candidate AST** 的构造性下界，不是已
接受的 canonical-program count。只有 strict canonicalizer 接受该子集且正式重放保持
超过 50,000，才进入：

```text
DSL_TOO_LARGE
  → new DSL version
  → shrink step 1: remove mean_v1, min_v1, max_v1
  → regenerate target/validation commitments
```

## 当前已冻结的 receipt 结构边界（不是待消歧项）

`ClosureEnumerationReceipt` 当前只是 **untrusted replay-claim wire record**。它已要求绑定
完整 `dsl_spec_id`、`operator_semantics_id`、`equivalence_contract_id`、enumerator 与
50,000 search budget；`target_role` 还必须选择彼此独立的 diagnostic content IDs：

| `target_role` | universe binding | truth binding |
| --- | --- | --- |
| `outside_target` | 480-row `bounded_universe_diagnostic_id` | `target_table_diagnostic_id` |
| `in_language_null_control` | 85-row `hidden_sink_universe_diagnostic_id` | `hidden_sink_target_table_diagnostic_id` |

两组 diagnostic IDs 不能复用或互换。wire 也能结构化表达第 50,001 个 witness：

当前 preregistration/receipt 层使用 `dsl_spec_<hex>`、`bounded_universe_<hex>` 等
canonical-JSON named content IDs，而 certificate record 使用 canonical-CBOR row leaves +
RFC6962 Merkle 的 `sha256:<hex>` roots。两者连 preimage 和聚合算法都不同，不能换前缀或
去前缀互转。正式 bridge 尚未冻结；这属于下方 `CERT_04`。

```json
{
  "closure_status": "DSL_TOO_LARGE",
  "enumerated_canonical_program_count": 50000,
  "first_out_of_budget_program_id": "<content-id-for-program-50001>",
  "frontier_exhausted": false,
  "all_type_buckets_closed": false,
  "closure_cardinality": null,
  "raw_expansion_limit_hit": false,
  "wall_clock_abort_hit": false
}
```

这只是 fail-closed 的字段约束，不是当前执行证据。sealed verifier 尚未实现，调用者提供
任何 receipt 都不能把当前 `executed_closure_status = NOT_RUN` 改成
`DSL_TOO_LARGE`、`COMPLETE`、`IN_LANGUAGE` 或 `OUTSIDE_FROZEN_CLOSURE`。

## 历史待消歧 A：scope source alias（v1.0.2 已在规范层解决）

决定稿 §3.3 使用 `control_volume_primary_only_v1`，§5.4 的四成员 catalog 只有
`scope_primary_only_v1`。当前保守绑定是：

```json
{
  "decision_id": "P3_SCOPE_BASELINE_ALIAS",
  "status": "UNRESOLVED_CONFIRMATION",
  "machine_scope_id": "scope_primary_only_v1",
  "source_document_alias": "control_volume_primary_only_v1",
  "scope_catalog_cardinality": 4,
  "fifth_scope_added": false
}
```

请确认前者只是来源别名/笔误；若必须新增语义不同的第五个 scope，应发布新 DSL
version，不能原地改写 v1。

## 历史待消歧 B：strict canonical AST/CBOR acceptance（v1.0.2 已在规范层解决）

决定稿还有一个直接的 typing 冲突：§2.6 把 executable XOR witness 写为
`absolute(difference(bit_at(0), bit_at(1)))`，但 §5.7–5.9 同时冻结
`bit_at -> Bit`、`difference(RationalValue, RationalValue)`，并提供显式
`bit_to_scalar(Bit) -> RationalValue`。当前机器合同不发明 implicit coercion，保存：

```json
{
  "decision_id": "P3_BIT_RATIONAL_COERCION",
  "status": "UNRESOLVED",
  "source_expression": "absolute(difference(bit_at(0), bit_at(1)))",
  "source_expression_typechecks": false,
  "type_explicit_expression": "absolute(difference(bit_to_scalar(bit_at(0)), bit_to_scalar(bit_at(1))))",
  "implicit_bit_to_rational_coercion_frozen": false,
  "formal_xor_language_verdict": null
}
```

请唯一确认：采用类型显式表达式，还是在新 DSL version 中冻结隐式 coercion。建议保持
v1 无隐式 coercion并把 §2.6 表达式视为漏写两个 `bit_to_scalar`；无论选择哪项，都要在
strict canonicalizer/schema 冻结后再由 Python/Rust 完整 replay 判定 `IN_LANGUAGE`。

请提供下列 amendment 的唯一 machine-readable 值：

```json
{
  "amendment_id": "hegel-old-dsl-canonical-acceptance-v1",
  "status": "UNRESOLVED",
  "required_decisions": [
    {
      "id": "CANON_AST_NODE_CBOR_SCHEMA",
      "must_specify": [
        "node tag and field IDs for every leaf/unary/binary/ternary/AND node",
        "array-versus-map representation",
        "child and registry-reference representation",
        "scope-extension and clause-boundary representation",
        "root_operator_id extraction",
        "decoder rejection rules",
        "whether any implicit Bit-to-RationalValue coercion exists"
      ]
    },
    {
      "id": "SYNTACTIC_NORMALIZATION_AND_REWRITE",
      "must_specify": [
        "commutative child ordering",
        "operator aliases",
        "associative flattening",
        "duplicate or idempotent clause removal",
        "constant folding",
        "greater_equal(a,b) versus less_equal(b,a)",
        "approx_equal(a,b,0) versus equal_exact(a,b)",
        "AND1(atom) versus atom",
        "all permitted algebraic reductions"
      ]
    },
    {
      "id": "CANONICAL_COUNT_BOUNDARY",
      "must_specify": [
        "which syntactic rewrites occur before canonical-program counting",
        "which transformations are archive-only after counting",
        "explicit prohibition on extensional quotient as a completeness shortcut"
      ]
    },
    {
      "id": "AST_NODE_COUNTING",
      "must_specify": [
        "whether aggregate is one leaf or includes map/scope/quantity/extension nodes",
        "whether tolerance is a child node or an inline parameter",
        "whether top-level AND wrapper counts as a node",
        "whether AND clause boundaries or scope clauses count as nodes"
      ]
    }
  ]
}
```

在这些规则冻结前，diagnostic canonical JSON 不能替代 canonical CBOR，64,680 也
不能升级为 `DSL_TOO_LARGE`。

## 历史待消歧 C：certificate / MDL strict wire（v1.0.2 已在规范层解决）

以下九组覆盖 certificate implementation audit 暴露的 specification blockers：

```json
{
  "amendment_id": "hegel-p3-certificate-wire-v1",
  "status": "UNRESOLVED",
  "required_decision_groups": [
    {
      "id": "CERT_01_CANONICAL_CBOR_PROFILE",
      "covers": [
        "canonical CBOR backend and pinned version",
        "exact encoder/decoder acceptance profile"
      ]
    },
    {
      "id": "CERT_02_CANONICAL_AST_IDENTITY",
      "covers": [
        "strict canonical AST schema",
        "root operator extraction"
      ]
    },
    {
      "id": "CERT_03_PROGRAM_OUTPUT_ARCHIVE",
      "covers": [
        "program-output blob record and encoding",
        "output archive and root schema"
      ]
    },
    {
      "id": "CERT_04_PROGRAM_ARCHIVE_ROOT_RELATIONS",
      "covers": [
        "canonical_program_archive_root versus program_archive_root naming",
        "program archive root versus chunk_manifest_root relation",
        "canonical-JSON content ID versus canonical-CBOR/RFC6962 root preimage and algorithm bridge",
        "Merkle preimages for each root"
      ]
    },
    {
      "id": "CERT_05_MATCH_PROGRAM_HASH_IDENTITY",
      "covers": [
        "whether match_program_hash binds AST CBOR, AST hash, record hash, or output identity"
      ]
    },
    {
      "id": "CERT_06_EXHAUSTION_RECEIPT_ROOT",
      "covers": [
        "exhaustion receipt root preimage",
        "self-field and signature-field exclusion rule"
      ]
    },
    {
      "id": "CERT_07_FINAL_ENVELOPE_AND_COMMIT",
      "covers": [
        "final certificate envelope and timestamp schema",
        "repository_commit_sha hash algorithm and wire format"
      ]
    },
    {
      "id": "CERT_08_KEY_STATUS_AND_REVOCATION",
      "covers": [
        "latest key-status manifest discovery and trust anchor",
        "Ed25519 public-key and signature encoding",
        "exact key-revocation manifest fields"
      ]
    },
    {
      "id": "CERT_09_MDL_DUAL_REPLAY_WIRE",
      "covers": [
        "MDL AST and new-symbol canonical wire schema",
        "literal 16-bit NEW_REDUCER_V1 header value",
        "MDL dual-replay receipt and certificate envelope",
        "cross-language Q32 log2 reference algorithm"
      ]
    }
  ]
}
```

## 历史快照中已冻结但尚未执行

- Phase-2B trusted wire builder/covert audit、720 + 240 generation、572 derived
  pairs、baseline pins、custodian、recognizer、evaluator 和 durable ledger；
- Phase-3 strict canonicalizer acceptance、完整 closure、program/output archives、
  exhaustion receipt、Python/Rust 双 replay、3/3 certificate 和完整 MDL scorer replay；
- 当前 receipt 虽已绑定完整 DSL/operator semantics 和按 `target_role` 分离的
  universe/truth roots，仍是 untrusted wire；sealed replay 未执行，closure 保持 `NOT_RUN`；
- Phase-2B/Phase-3 均保持 shadow-only，formal claim 与 ACTIVE promotion 为 NO-GO。
