# Hegel Machine Strict Canonical Acceptance 与 Certificate Bridge 冻结决策

**文档类型**：Normative implementation-audit amendment<br>
**建议版本**：`hegel-freeze-p2b-p3-v1.0.2`<br>
**继承版本**：`hegel-freeze-p2b-p3-v1.0.1`<br>
**适用范围**：

- Phase-2B exact-freeze artifact binding；
- Phase-3A Bounded Frozen-Closure Adequacy；
- strict canonical AST；
- canonical CBOR；
- RFC6962 program/output archives；
- diagnostic JSON content ID → formal root bridge；
- Python/Rust closure replay；
- MDL dual replay；
- 3/3 Ed25519 certificate。

本 amendment **不改变** v1.0.1 已冻结的：

- 720 main + 240 semantic-conflict；
- 496 legal + 76 invalid = 572 derived pairs；
- 每 cell 的 `19 + 1` answerable 配额；
- `21 / 18 / 12 / 9` margin；
- master/bootstrap seed `411876909552964556`；
- sklearn 派生 uint32 seed `2611585425`；
- 480-row odd-cardinality target；
- 85-row observed omitted-sink null control；
- 50,000 canonical-program budget；
- 5,000,000 raw operator-application cap；
- DSL shrink order；
- MDL threshold与 Q32 比较原则；
- shadow-only、formal exit 关闭、ACTIVE 关闭。

> **POST-M2 EXECUTION OVERLAY（2026-08-01）**
>
> 本文的 wire schema、rewrite、typing、count boundary 与 certificate bridge
> 决策仍是 normative。本文中把 strict implementation 写成 `false`、把 64,680
> 写成 conditional、把 closure 写成 `NOT_RUN` 的“当前状态”段落，均是 M1/M2
> 执行前快照，不再代表 checkout 的执行状态。
>
> 当前证据是：Python/Rust shared vectors 各 `48/48`；两端对同一纯生成器定义的
> 64,680 个 source candidates 均接受为 64,680 个 unique strict canonical AST；
> bounded old-DSL 状态为 `DSL_TOO_LARGE`，不是 `COMPLETE`。共同 diagnostic set
> commitment 为
> `sha256:c1a02a66a8d6d8f75204cb3daf03ab0b01c2b3b8e486d0ab3d481ee3be43c930`。
>
> `formal_root_generation_allowed` 在当前 operational artifacts 中仍为 `false`：
> M1 通过只证明 strict identity 实现门，M2 又进入 `DSL_TOO_LARGE` path；正式
> archive/bridge executor 尚未双实现验证，也没有生成任何 root。§14.4 的
> “formal archive roots may bind accepted first 50000”只是 envelope schema 的许可，
> 不是自动状态迁移，也不把 M2 diagnostic commitment 升格为 RFC6962 root。
> 当前唯一下一动作是发布新 DSL version、执行 frozen shrink step 1、重建
> commitments，并让新版本从 `NOT_RUN` 重跑。formal roots、extensional target/
> hidden-sink verdict、outside/MDL certificate 与 ACTIVE 均保持关闭。

---

# 0. 总体定案

## 0.1 当前方向判断

当前方向**没有跑偏**。

尤其正确的是：

1. 将 Phase-3A 更名为 **Bounded Frozen-Closure Adequacy**，避免在 closure 尚未运行时声称 invention；
2. 将 64,680 严格保留为 candidate-AST 条件性容量下界，而不是 canonical closure cardinality；
3. 将：
   ```text
   COMPLETE
   DSL_TOO_LARGE
   INCONCLUSIVE_BUDGET / execution
   ```
   设计成互斥状态；
4. formal roots 保持 `null`，直到 canonical CBOR archive 和 replay 真正执行；
5. diagnostic JSON IDs 与正式 Merkle roots 不做字符串前缀转换；
6. 不允许单实现自报 `OUTSIDE_FROZEN_CLOSURE`。

当前真正的主线是：

\[
\boxed{
\text{冻结 identity}
\rightarrow
\text{运行完整 closure}
\rightarrow
\text{独立 replay}
\rightarrow
\text{才判断 adequacy}
}
\]

而不是继续扩充 benchmark family。

## 0.2 v1.0.2 完成后的机器状态

规范签署后：

```json
{
  "freeze_version": "hegel-freeze-p2b-p3-v1.0.2",
  "surface_parameter_freeze_complete": true,
  "strict_acceptance_specification_complete": true,
  "normative_parameter_freeze_complete": true,
  "strict_acceptance_implementation_verified": false,
  "formal_root_generation_allowed": false,
  "executed_closure_status": "NOT_RUN",
  "phase2b_formal_exit": false,
  "outside_certificate_allowed": false,
  "active_promotion_allowed": false
}
```

只有 Python/Rust strict implementations、golden vectors 和 replay 全部通过后：

```json
{
  "strict_acceptance_implementation_verified": true,
  "formal_root_generation_allowed": true
}
```

这仍不等于 closure 已经执行。

---

# 1. 待消歧 A：scope source alias

## 1.1 唯一决定

`control_volume_primary_only_v1` 只是来源文档别名/笔误。

唯一 machine scope ID 是：

```text
scope_primary_only_v1
```

不增加第五个 scope，不发布新 DSL version。

## 1.2 Machine-readable 决定

```json
{
  "decision_id": "P3_SCOPE_BASELINE_ALIAS",
  "decision_version": 1,
  "status": "RESOLVED",
  "machine_scope_id": "scope_primary_only_v1",
  "deprecated_source_alias": "control_volume_primary_only_v1",
  "semantic_identity": true,
  "scope_catalog_cardinality": 4,
  "fifth_scope_added": false,
  "canonical_ast_accepts_alias": false,
  "diagnostic_migration_layer_may_rewrite_alias": true
}
```

## 1.3 执行边界

alias rewrite 只允许出现在：

```text
legacy diagnostic JSON
→ migration adapter
→ strict machine object
```

formal canonicalizer 输入中出现 alias 时必须拒绝：

```text
REJECT_NONCANONICAL_SCOPE_ALIAS
```

不能让 alias 成为第五个可枚举 token，否则会改变 closure。

---

# 2. 待消歧 B：Bit → RationalValue coercion

## 2.1 唯一决定

v1 **不存在任何 implicit coercion**。

二元 XOR sanity witness 的唯一合法表达为：

```text
absolute(
  difference(
    bit_to_scalar(bit_at(0)),
    bit_to_scalar(bit_at(1))
  )
)
```

原文：

```text
absolute(difference(bit_at(0), bit_at(1)))
```

视为漏写两个显式 coercion，不是另一个合法 program。

## 2.2 Machine-readable 决定

```json
{
  "decision_id": "P3_BIT_RATIONAL_COERCION",
  "decision_version": 1,
  "status": "RESOLVED",
  "implicit_bit_to_rational_coercion": false,
  "canonical_expression": [
    "absolute",
    [
      "difference",
      ["bit_to_scalar", ["bit_at", 0]],
      ["bit_to_scalar", ["bit_at", 1]]
    ]
  ],
  "source_shorthand_is_executable": false,
  "source_shorthand_disposition": "DOCUMENTATION_OMISSION_ONLY",
  "formal_xor_language_verdict": null,
  "verdict_requires_complete_dual_replay": true
}
```

## 2.3 后果

在 strict closure 中：

- `difference(Bit, Bit)` 是 type error；
- type error program 不进入 canonical count；
- canonicalizer 不得自动插入 coercion；
- parser 不得根据 expected output type猜测 coercion；
- 若未来需要 implicit coercion，必须发布：
  ```text
  hegel-old-dsl-v2.*
  ```
  并重新生成 target/validation commitments。

---

# 3. Strict canonical CBOR profile

## 3.1 Profile ID

```text
canonical_cbor_profile_id = hegel-cbor-det-v1
```

它是项目自定义的 RFC8949 deterministic subset。

权威不是某个第三方 library；权威是：

- 本文的编码规则；
- golden vectors；
- Python encoder source root；
- Rust encoder source root。

## 3.2 Backend 决定

```json
{
  "profile_id": "hegel-cbor-det-v1",
  "normative_backend": "PROJECT_MINIMAL_ENCODER",
  "python_implementation": "commit_and_source_root_bound",
  "rust_implementation": "commit_and_source_root_bound",
  "third_party_encoder_is_authoritative": false
}
```

## 3.3 允许的 CBOR 类型

formal hashed core 只允许：

- unsigned integer；
- negative integer；
- byte string；
- array；
- `false`；
- `true`；
- `null`。

禁止：

- float；
- decimal fraction tag；
- bignum tag；
- 任意 CBOR tag；
- text string；
- map；
- indefinite-length item；
- `undefined`；
- 非 canonical integer encoding；
- trailing bytes。

固定 schema 中所有“名称”都通过 integer registry ID 表达。

## 3.4 Decoder acceptance

decoder 必须：

1. 完整 parse；
2. schema/type/range validate；
3. re-encode；
4. 要求：
   ```text
   original_bytes == reencoded_bytes
   ```
5. 否则拒绝：
   ```text
   REJECT_NONCANONICAL_CBOR
   ```

不允许“能读就接受”。

## 3.5 Integer canonicality

- 使用能容纳值的最短 additional-information encoding；
- `0..23` 必须直接编码；
- 不允许 `0` 编码为 `uint8 0`；
- negative integer 按 CBOR 的 `-1-n` 规则；
- registry index 必须是 nonnegative uint；
- fixed-width 16-bit header 是 MDL bit code 概念，不强迫 CBOR 用 16-bit integer encoding。

---

# 4. Strict canonical AST schema

## 4.1 AST envelope

```text
CanonicalAstV1 = [1, RootNode]
```

其中首元素 `1` 是 schema version。

envelope 不计 AST node。

## 4.2 Node tags

| Node kind | Tag |
|---|---:|
| leaf | 0 |
| unary | 1 |
| binary | 2 |
| ternary | 3 |
| conjunction | 4 |

## 4.3 Leaf schema

### `scalar_const`

```text
[0, 0, rational_parameter_index]
```

### `bit_at`

```text
[0, 1, entity_slot_index]
```

### `set_size`

```text
[0, 2]
```

### `aggregate`

```text
[
  0,
  3,
  aggregate_map_id,
  scope_id,
  quantity_id,
  scope_extension
]
```

其中：

```text
scope_extension = [
  [context_id, expected_bool],
  ...
]
```

约束：

- 0–2 clauses；
- 按 `context_id` 升序；
- context ID 不得重复；
- `expected_bool ∈ {false,true}`；
- canonical form 中不得出现 source alias。

### `context_flag`

```text
[0, 4, context_id]
```

### `task_flag`

```text
[0, 5, task_id]
```

### `new_symbol_call`

```text
[0, 6, new_symbol_registry_index]
```

此 leaf：

- 在 old DSL canonicalizer 中拒绝；
- 只在 Phase-3B extended-language / MDL canonicalizer 中接受；
- 对当前输入的整个 typed `EntitySet` 调用 reducer；
- 不允许隐式参数列表或 case-specific argument。

## 4.4 Unary schema

```text
[1, unary_operator_id, child_node]
```

| Operator | ID |
|---|---:|
| `bit_to_scalar` | 0 |
| `int_to_scalar` | 1 |
| `absolute` | 2 |
| `sign` | 3 |

## 4.5 Binary schema

```text
[2, binary_operator_id, left_node, right_node]
```

| Operator | Source ID | Canonical status |
|---|---:|---|
| `add` | 0 | accepted |
| `difference` | 1 | accepted |
| `equal_exact` | 2 | accepted |
| `less_equal` | 3 | accepted |
| `greater_equal` | 4 | source-only；rewritten |
| `same_sign` | 5 | accepted |
| `opposite_sign` | 6 | accepted |
| reserved | 7 | rejected |

canonical AST 不接受 operator ID 4。

## 4.6 Ternary schema

```text
[3, 0, left_node, right_node, tolerance_index]
```

唯一 ternary operator 是：

```text
approx_equal
```

`tolerance_index` 是 inline parameter，不是 AST child。

## 4.7 Conjunction schema

```text
[4, [atom_1, atom_2, ...]]
```

canonical clause count 只能是：

```text
2 or 3
```

`AND1(atom)` canonicalize 为 `atom`。

clause list：

- 按 child canonical CBOR bytes lexicographic ascending；
- exact duplicate 删除；
- 删除后为 1 个 atom 时移除 wrapper；
- 删除后为 0 个 atom 是非法，不生成常量 true。

## 4.8 `root_operator_id`

提取为 uint16 semantic ID：

```text
leaf:        0x0000 + leaf_id
unary:       0x0100 + unary_operator_id
binary:      0x0200 + binary_operator_id
ternary:     0x0300 + ternary_operator_id
conjunction: 0x0400
```

formal CBOR 中作为普通 uint 编码。

## 4.9 Decoder rejection rules

必须拒绝：

- unknown schema version；
- unknown/reserved tag或 operator；
- wrong array length；
- wrong child type；
- implicit coercion requirement；
- registry index out of range；
- unordered commutative children；
- duplicate AND atom；
- unsorted scope clauses；
- duplicate scope context；
- alias scope ID；
- `greater_equal` 出现在 canonical AST；
- `approx_equal(..., tolerance=0)`；
- `AND` length 1；
-违反 node/depth/slot/aggregate/scope limits；
- old DSL 中出现 `new_symbol_call`；
-任何 trailing bytes。

---

# 5. Syntactic normalization 与 rewrite

## 5.1 Normalize-before-count 原则

流程冻结为：

```text
parse source AST
→ resolve legacy documentation aliases
→ type-check source AST
→ apply only the frozen rewrites below
→ serialize strict canonical AST
→ enforce structural limits
→ syntactic deduplicate
→ increment canonical-program count
```

除下列规则外，不允许其他“聪明化简”。

## 5.2 Commutative child ordering

以下 operator 视为 commutative：

```text
add
equal_exact
same_sign
opposite_sign
approx_equal
```

binary/ternary 的前两个 expression children 按：

```text
SHA256(canonical_child_cbor)
then canonical_child_cbor bytes
```

排序。

`difference`、`less_equal` 不排序。

## 5.3 `greater_equal`

```text
greater_equal(a,b)
→
less_equal(b,a)
```

发生在 canonical count 前。

## 5.4 `approx_equal` tolerance 0

```text
approx_equal(a,b,0)
→
equal_exact(a,b)
```

发生在 canonical count 前。

## 5.5 AND normalization

- nested AND flatten；
- atoms 排序；
- exact duplicate remove；
- `AND1(atom) → atom`；
- 最大 3 clauses 在 flatten 后检查；
- 不执行逻辑吸收、resolution、De Morgan 或 truth-table simplification。

## 5.6 Add normalization

只对 `add`：

1. 收集 nested `add` operands；
2. 按 child canonical hash/bytes 排序；
3. 以 **right-associated** 形式重建；
4. 重建后再检查 depth/node limit。

例如：

```text
add(add(a,c),b)
→
add(a,add(b,c))
```

假定 canonical order 为 `a < b < c`。

## 5.7 允许的局部 algebraic rewrite

只允许：

```text
add(x, 0)          → x
add(0, x)          → x
difference(x, 0)   → x
difference(x, x)   → scalar_const(0)
absolute(absolute(x)) → absolute(x)
```

常量折叠仅在结果仍属于冻结 `RationalParameter` grid 时执行：

```text
add(const_a,const_b)
difference(const_a,const_b)
absolute(const_a)
```

若结果不在 grid，不折叠，也不引入新 literal。

## 5.8 明确禁止的 pre-count rewrite

禁止：

- distributivity；
- factorization；
- arbitrary associativity of non-add operators；
- inequality transitive closure；
- comparison negation；
- replacing equivalent aggregates；
- symbolic linear algebra beyond上述规则；
- extensional truth-table quotient；
- SMT-derived equivalence merging；
- target-aware simplification。

这些只能进入 post-count analysis。

---

# 6. Canonical count boundary 与 AST node counting

## 6.1 Count 对象

50,000 统计：

> strict canonical AST bytes 不同的 syntactically canonical programs。

程序经过 §5 的 rewrites 后计数。

## 6.2 Post-count archive-only transformations

以下不减少 canonical count：

- extensional output-vector grouping；
- algebraic equivalence class；
- first extensional representative；
- target match grouping；
- output-archive dedup；
- compression of repeated blobs。

## 6.3 Node counting

| Element | Counts as AST node |
|---|:---:|
| leaf | 是 |
| unary/binary/ternary operator | 是 |
| AND wrapper | 是 |
| each AND atom subtree | 按正常递归 |
| AST envelope | 否 |
| aggregate map ID | 否 |
| scope ID | 否 |
| quantity ID | 否 |
| scope-extension list | 否 |
| each scope clause | 否 |
| tolerance inline parameter | 否 |
| registry reference | 否 |
| clause boundary | 否 |

`aggregate` 是一个 leaf node，无论其 scope extension 有 0、1 还是 2 clauses。

## 6.4 Depth

```text
leaf depth = 0
operator depth = 1 + max(child depth)
AND depth = 1 + max(atom depth)
```

scope clauses 不影响 AST depth，但影响 MDL scope code length。

## 6.5 64,680 capacity result 的状态

```json
{
  "status": "CONDITIONAL_CAPACITY_LOWER_BOUND_EXCEEDS_BUDGET",
  "constructive_candidate_ast_count": 64680,
  "canonical_program_budget": 50000,
  "strict_rewrite_application_pending": true,
  "strict_canonicalizer_acceptance_verified": false,
  "dsl_too_large_claim_allowed": false
}
```

若 strict normalization 后可接受的该构造子集仍产生第 50,001 个不同 canonical AST，才可运行：

```text
DSL_TOO_LARGE
```

否则必须继续完整枚举，不能从 diagnostic count 推断 closure status。

---

# 7. Formal content hash 与 RFC6962 通用规则

## 7.1 Hash algorithm

```text
hash_algorithm_id = 1
hash_algorithm = SHA-256
```

formal wire 内 digest 表示为 32-byte CBOR byte string。
人类 JSON rendering 才使用：

```text
sha256:<lowercase hex>
```

## 7.2 Domain-separated content hash

```text
ContentHash(domain, object)
=
SHA256(
  UTF8(domain)
  || 0x00
  || CanonicalCBOR(object)
)
```

## 7.3 RFC6962 Merkle

```text
LeafHash(record)
=
SHA256(0x00 || CanonicalCBOR(record))

NodeHash(left,right)
=
SHA256(0x01 || left || right)
```

非 2 的幂按 RFC6962 largest-power-of-two split。
禁止 duplicate-last-leaf。

空树 root：

```text
SHA256(empty byte string)
```

每一种 tree 的 record schema tag 不同，防止跨 archive leaf 混用。

---

# 8. CERT_01：Canonical CBOR profile

## 8.1 决定

```json
{
  "decision_group": "CERT_01_CANONICAL_CBOR_PROFILE",
  "status": "RESOLVED",
  "profile_id": "hegel-cbor-det-v1",
  "normative_backend": "PROJECT_MINIMAL_ENCODER",
  "allowed_major_types": [0,1,2,4,7],
  "allowed_simple_values": ["false","true","null"],
  "maps_allowed": false,
  "text_strings_allowed": false,
  "floats_allowed": false,
  "tags_allowed": false,
  "indefinite_lengths_allowed": false,
  "decoder_requires_exact_reencode": true
}
```

Python/Rust source roots和 execution binary digests 必须进入 execution manifest。

---

# 9. CERT_02：Canonical AST identity

## 9.1 AST hash

```text
canonical_ast_hash
=
ContentHash(
  "HEGEL/AST/V1",
  CanonicalAstV1
)
```

## 9.2 Identity

两个 programs 的 syntactic identity 当且仅当：

```text
canonical_ast_cbor bytes identical
```

hash 相等只是索引；安全比较最终以 bytes 相等为准。

## 9.3 Machine-readable 决定

```json
{
  "decision_group": "CERT_02_CANONICAL_AST_IDENTITY",
  "status": "RESOLVED",
  "ast_schema_id": "hegel-canonical-ast-v1",
  "identity_preimage": "canonical_ast_cbor_bytes",
  "hash_domain": "HEGEL/AST/V1",
  "root_operator_extraction": "section_4_8",
  "hash_collision_policy": "COMPARE_CANONICAL_BYTES_AND_ABORT_ON_DISTINCT_PREIMAGES"
}
```

---

# 10. CERT_03：Program-output archive

## 10.1 Output cell

```text
UndefinedCell = [0]
DefinedCell   = [1, CanonicalValue]
```

CanonicalValue：

| Sort | Encoding |
|---|---|
| Bool | `false` / `true` |
| Bit | uint `0` / `1` |
| Sign | int `-1 / 0 / 1` |
| BoundedInt | CBOR integer |
| RationalValue | `[numerator, denominator]`，reduced，denominator > 0 |

## 10.2 Output blob

```text
ProgramOutputBlobV1 =
[
  1,
  output_sort_id,
  row_count,
  [cell_0, cell_1, ..., cell_(row_count-1)]
]
```

output blob hash：

```text
ContentHash("HEGEL/OUTPUT_BLOB/V1", ProgramOutputBlobV1)
```

## 10.3 Undefined bitmap

- universe index 0 对应第一个 bit 的 MSB；
- 每 byte 从 bit 7 到 bit 0；
- `1 = undefined`；
- 最后一 byte unused low bits 必须为 0。

hash：

```text
SHA256(
  UTF8("HEGEL/UNDEFINED_BITMAP/V1")
  || 0x00
  || bitmap_bytes
)
```

## 10.4 Program output record

```text
ProgramOutputRecordV1 =
[
  1,
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

## 10.5 Output archive root

按 `program_index` 升序，对 ProgramOutputRecordV1 做 RFC6962 tree：

```text
program_output_archive_root
```

---

# 11. Program record 与 program archive

## 11.1 Program record

```text
ProgramRecordV1 =
[
  1,
  program_index,
  canonical_ast_cbor_bytes,
  canonical_ast_hash,
  output_sort_id,
  ast_depth,
  ast_node_count,
  distinct_bit_slot_count,
  program_mdl_length_q32,
  undefined_bitmap_hash,
  output_blob_hash,
  extensional_class_hash,
  first_extensional_representative_index,
  dsl_spec_root,
  operator_semantics_root,
  bounded_universe_root
]
```

decoder 必须验证：

- AST bytes strict canonical；
- AST hash正确；
- derived fields重新计算一致；
- output hash 与 output archive record 一致。

## 11.2 Extensional class hash

```text
extensional_class_hash =
SHA256(
  UTF8("HEGEL/EXTENSIONAL_CLASS/V1")
  || 0x00
  || operator_semantics_root
  || bounded_universe_root
  || output_blob_hash
)
```

它用于 grouping，不用于 canonical count。

## 11.3 Canonical program archive root

按全局 canonical order 对 ProgramRecordV1 做 RFC6962 tree：

```text
canonical_program_archive_root
```

formal wire 中不再使用另一个 `program_archive_root` 名称。

legacy JSON 中：

```text
program_archive_root
```

只能作为：

```text
deprecated_alias_of = canonical_program_archive_root
```

formal decoder 看到该字段名不适用，因为 formal core 是 numeric schema。

---

# 12. CERT_04：Archive roots 与 diagnostic bridge

## 12.1 三个不同 roots

```text
canonical_program_archive_root
program_output_archive_root
chunk_manifest_root
```

三者绝不相等，也不能互换。

- program root：ProgramRecord leaves；
- output root：ProgramOutputRecord leaves；
- chunk root：ChunkManifest leaves。

## 12.2 Chunk manifest

```text
ChunkManifestV1 =
[
  1,
  chunk_index,
  first_program_index,
  last_program_index,
  record_count,
  program_record_subtree_root,
  output_record_subtree_root,
  compressed_program_blob_hash,
  compressed_output_blob_hash,
  uncompressed_program_byte_length,
  uncompressed_output_byte_length
]
```

固定：

```text
records_per_chunk = 4096
```

chunk manifests 按 `chunk_index` 做 RFC6962 tree：

```text
chunk_manifest_root
```

`chunk_manifest_root` 是 transport/index root，不是 program root 的别名。

## 12.3 Diagnostic JSON content ID

diagnostic object 使用 RFC8785 JCS：

```text
diagnostic_digest =
SHA256(
  UTF8("HEGEL/DIAGNOSTIC_JCS/V1")
  || 0x00
  || JCS_UTF8_BYTES
)
```

human ID：

```text
<namespace>_<lowercase hex diagnostic_digest>
```

例：

```text
bounded_universe_<hex>
target_table_<hex>
dsl_spec_<hex>
```

prefix 是 namespace，不属于 digest，也不能替换成 `sha256:`。

## 12.4 Formal objects

formal object 分两类：

```text
FORMAL_CONTENT_DIGEST
FORMAL_RFC6962_TREE_ROOT
```

单个 spec：

```text
FORMAL_CONTENT_DIGEST =
ContentHash(domain, canonical_cbor_object)
```

row collection：

```text
FORMAL_RFC6962_TREE_ROOT =
RFC6962(row_records)
```

## 12.5 Diagnostic → formal bridge

```text
DiagnosticFormalBridgeV1 =
[
  1,
  artifact_role_id,
  diagnostic_namespace_id,
  diagnostic_digest,
  formal_object_kind_id,
  formal_digest_or_root,
  row_count_or_null,
  diagnostic_profile_id,
  formal_profile_id,
  row_transform_spec_root,
  source_artifact_digest,
  repository_commit_id
]
```

bridge hash：

```text
ContentHash(
  "HEGEL/DIAGNOSTIC_FORMAL_BRIDGE/V1",
  DiagnosticFormalBridgeV1
)
```

bridge 必须由 custodian + Python replay + Rust replay 3/3 签名。

## 12.6 关键原则

禁止：

```text
remove diagnostic prefix
→ add sha256:
```

禁止：

```text
SHA256(diagnostic JSON)
== RFC6962 formal root
```

正式关系是：

```text
signed bridge binds both distinct preimages
```

---

# 13. CERT_05：match program identity

## 13.1 唯一决定

```text
match_program_hash
=
canonical_ast_hash
```

它绑定 strict canonical AST CBOR，不绑定：

- ProgramRecord hash；
- output blob hash；
- extensional class hash；
- program index。

## 13.2 Match set record

```text
MatchRecordV1 =
[
  1,
  canonical_ast_hash,
  output_blob_hash,
  target_truth_table_root
]
```

match set 按 `canonical_ast_hash` bytes 升序。

certificate 同时绑定：

- AST identity；
- exact output identity；
- target identity。

## 13.3 Outside case

`match_set_count == 0` 时：

```text
match_records = []
```

不能用 null。

---

# 14. CERT_06：Exhaustion receipt root

## 14.1 Closure status enum

| Status | ID |
|---|---:|
| `NOT_RUN` | 0 |
| `COMPLETE` | 1 |
| `DSL_TOO_LARGE` | 2 |
| `INCONCLUSIVE_BUDGET` | 3 |
| `INCONCLUSIVE_EXECUTION` | 4 |

互斥状态由 schema invariants 检查。

## 14.2 Receipt body

```text
ClosureReplayReceiptBodyV1 =
[
  1,
  implementation_id,
  implementation_source_root,
  implementation_binary_digest,
  dsl_spec_root,
  operator_semantics_root,
  equivalence_contract_root,
  canonical_ast_schema_root,
  canonicalizer_source_root,
  enumerator_source_root,
  bounded_universe_root,
  target_truth_table_root,
  target_role_id,
  closure_status_id,
  raw_operator_application_count,
  canonical_program_count,
  closure_cardinality_or_null,
  frontier_exhausted,
  all_type_buckets_closed,
  raw_expansion_limit_hit,
  wall_clock_abort_hit,
  canonical_program_archive_root_or_null,
  program_output_archive_root_or_null,
  chunk_manifest_root_or_null,
  match_set_count_or_null,
  [match_records],
  first_out_of_budget_program_hash_or_null,
  bucket_accounting_root_or_null,
  execution_manifest_root
]
```

## 14.3 Receipt root

```text
exhaustion_receipt_root =
ContentHash(
  "HEGEL/CLOSURE_REPLAY_RECEIPT/V1",
  ClosureReplayReceiptBodyV1
)
```

body 中不包含：

- exhaustion receipt root 自身；
- signatures；
- certificate timestamp；
- human notes。

## 14.4 State invariants

### COMPLETE

```text
frontier_exhausted = true
all_type_buckets_closed = true
closure_cardinality = canonical_program_count
canonical_program_archive_root != null
program_output_archive_root != null
chunk_manifest_root != null
match_set_count != null
first_out_of_budget_program_hash = null
```

### DSL_TOO_LARGE

```text
canonical_program_count = 50000
closure_cardinality = null
frontier_exhausted = false
all_type_buckets_closed = false
first_out_of_budget_program_hash != null
formal archive roots may bind accepted first 50000
match_set_count = null
```

### INCONCLUSIVE_BUDGET / EXECUTION

不得携带正式 match verdict。

---

# 15. CERT_07：Final certificate envelope 与 Git commit

## 15.1 Git commit identity

```text
RepositoryCommitIdV1 =
[
  algorithm_id,
  raw_digest_bytes
]
```

| Algorithm | ID | Length |
|---|---:|---:|
| Git SHA-1 | 1 | 20 bytes |
| Git SHA-256 | 2 | 32 bytes |

当前 GitHub repository 必须使用：

```text
[1, <20 raw bytes decoded from 40 hex chars>]
```

禁止在 formal body 中保存 ASCII hex。

## 15.2 Signature input

```text
certificate_body_hash =
ContentHash(
  "HEGEL/FINAL_CERTIFICATE_BODY/V1",
  FinalCertificateBodyV1
)

signature_message =
UTF8("HEGEL/CERT_SIGNATURE/V1")
|| 0x00
|| certificate_body_hash
```

## 15.3 Signature record

```text
SignatureRecordV1 =
[
  key_id,
  ed25519_signature_64_bytes
]
```

`key_id`：

```text
first 16 bytes of SHA256(raw_ed25519_public_key_32_bytes)
```

signatures 按 key ID bytes 升序。

## 15.4 Envelope

```text
FinalCertificateEnvelopeV1 =
[
  1,
  certificate_kind_id,
  created_at_unix_seconds,
  key_epoch,
  FinalCertificateBodyV1,
  certificate_body_hash,
  [signature_record_1, signature_record_2, signature_record_3]
]
```

- timestamp 为 UTC Unix seconds；
- 不允许 fractional seconds；
- 3/3 active keys；
- envelope 自身不再递归包含 envelope hash。

---

# 16. CERT_08：Key status、rotation 与 revocation

## 16.1 Genesis trust anchor

genesis manifest 的 SHA-256 digest 必须：

- 写入 preregistration；
- 内嵌在独立 verifier binary；
- 至少保存在一个 repo 外渠道。

仅放在同一 mutable repository 中不构成 trust anchor。

## 16.2 Key epoch body

```text
KeyEpochBodyV1 =
[
  1,
  key_epoch,
  previous_epoch_manifest_hash_or_null,
  effective_at_unix_seconds,
  [
    [key_id_1, raw_public_key_1],
    [key_id_2, raw_public_key_2],
    [key_id_3, raw_public_key_3]
  ],
  certificate_threshold,
  transition_threshold,
  reason_code
]
```

冻结：

```text
certificate_threshold = 3
transition_threshold = 2
```

## 16.3 Rotation envelope

新 epoch manifest 必须由旧 epoch 至少 2/3 keys 签名。

## 16.4 Revocation body

```text
KeyRevocationBodyV1 =
[
  1,
  revocation_id,
  issuing_epoch,
  issued_at_unix_seconds,
  revoked_key_ids,
  reason_code,
  invalid_before_unix_seconds_or_null,
  invalid_after_unix_seconds_or_null,
  superseding_epoch_or_null,
  previous_status_manifest_hash
]
```

需当前有效 epoch 2/3 signatures。

## 16.5 Latest manifest discovery

verifier 不从网络“寻找最新”。

输入必须包含一个 manifest bundle。verifier：

1. 从 pinned genesis 开始；
2. 验证连续 epoch chain；
3. 验证每次 2/3 transition；
4. 验证 revocations；
5. 选择最高有效 epoch；
6. 若同一 parent 有两个有效 next epoch，判：
   ```text
   KEY_STATUS_FORK
   ```
   并 fail closed。

---

# 17. CERT_09：MDL dual replay wire

## 17.1 New reducer header

MDL bit-code 中：

```text
NEW_REDUCER_V1_HEADER = 0x4852
```

大端 16 bits：

```text
01001000 01010010
```

`0x4852` 对应 ASCII `HR`，但 ASCII 解释不是语义的一部分。

canonical CBOR 中 header 作为 uint `18514` 编码，仍须最短 CBOR integer form。

## 17.2 New reducer definition

```text
NewReducerDefinitionV1 =
[
  1,
  0x4852,
  new_symbol_registry_index,
  arity,
  [input_sort_ids],
  output_sort_id,
  fold_scheme_id,
  identity_rational_parameter_index,
  binary_combiner_canonical_ast_cbor,
  maximum_supported_set_size,
  scope_id,
  scope_extension,
  verifier_spec_root
]
```

fold scheme：

| Scheme | ID |
|---|---:|
| left fold | 0 |
| balanced fold | 1 |

v1 parity invention 必须声明一种，不能由 runtime 自选。

## 17.3 MDL replay body

```text
MdlReplayReceiptBodyV1 =
[
  1,
  implementation_id,
  mdl_code_table_root,
  dsl_spec_root,
  identifier_registry_root,
  discovery_partition_root,
  validation_partition_root,
  sealed_partition_root,
  target_truth_table_root,
  old_program_ast_hash,
  new_symbol_definition_hash,
  new_call_program_ast_hash,
  old_prediction_vector_root,
  new_prediction_vector_root,
  validation_prediction_root,
  sealed_prediction_root,
  l_old_program_q32,
  l_train_given_old_q32,
  l_new_symbol_definition_q32,
  l_new_call_program_q32,
  l_train_given_new_q32,
  delta_l_q32,
  required_delta_l_q32,
  mdl_gate_pass,
  q32_algorithm_id,
  execution_manifest_root
]
```

## 17.4 Q32 log2 reference algorithm

```text
q32_algorithm_id = hegel-mpfr-log2-q32-v1
MPFR version = 4.2.1
```

对正整数 \(N\)：

\[
Q(N)=
\left\lceil
2^{32}\log_2 N
\right\rceil.
\]

执行算法：

1. 以 exact integer 构造 \(N\)；
2. 初始 MPFR precision：
   ```text
   267 bits
   ```
   对应至少 80 decimal digits；
3. 分别用：
   ```text
   MPFR_RNDD
   MPFR_RNDU
   ```
   计算 \(\log_2 N\) 的区间；
4. 两端乘 exact \(2^{32}\)；
5. 分别取 mathematical ceiling；
6. 若两个 ceiling 相同，该整数即 Q32；
7. 否则 precision 加倍；
8. 上限 4272 bits；
9. 仍不唯一则：
   ```text
   Q32_NUMERIC_INCONCLUSIVE
   ```
   并 fail closed。

对：

\[
\log_2 {n\choose k}
\]

先以 exact integer 算出 binomial，再调用同一算法。

每一个 log component 独立向上 Q32；所有 component 的相加使用 exact uint arithmetic。

Python 和 Rust 可以都链接 MPFR，但：

- 调度逻辑；
- exact binomial；
- code-table traversal；
- receipt construction；

必须独立实现。

所有 golden vector 必须绑定：

```text
q32_golden_vector_root
```

## 17.5 MDL certificate

MDL final certificate 使用与 closure certificate 同一：

- canonical CBOR profile；
- final envelope；
- key epoch；
- 3/3 signatures。

Python/Rust receipt 必须一致：

```text
all roots
all seven length fields
delta
required delta
gate result
```

caller-supplied Fraction、float、length 字段一律忽略。

---

# 18. Certificate body：Outside frozen closure

## 18.1 Certificate kind

```text
certificate_kind_id = 1
```

## 18.2 Body

```text
OutsideFrozenClosureCertificateBodyV1 =
[
  1,
  claim_id,
  freeze_version_id,
  dsl_spec_root,
  operator_semantics_root,
  equivalence_contract_root,
  canonical_ast_schema_root,
  bounded_universe_root,
  target_truth_table_root,
  target_role_id,
  equivalence_mode_id,
  python_exhaustion_receipt_root,
  rust_exhaustion_receipt_root,
  canonical_program_count,
  canonical_program_archive_root,
  program_output_archive_root,
  chunk_manifest_root,
  match_set_count,
  match_set_root,
  diagnostic_formal_bridge_root,
  execution_manifest_root,
  repository_commit_id,
  container_image_digest
]
```

冻结：

```text
equivalence_mode_id = 1  # exact extensional
match_set_count = 0
match_set_root = RFC6962 empty tree root
```

## 18.3 签发条件

只有：

```text
python closure status == COMPLETE
rust closure status == COMPLETE
all strict roots equal
canonical counts equal
frontiers exhausted
all buckets closed
no budget abort
match_set_count == 0
covert-channel audit pass
diagnostic bridge valid
3 active signatures
```

才允许签发。

human claim rendering：

```text
OUTSIDE_FROZEN_CLOSURE(
  dsl_version = hegel-old-dsl-v1.0.0,
  bounded_universe_root = sha256:<...>,
  target_truth_table_root = sha256:<...>,
  equivalence = exact_extensional
)
```

仍然禁止：

```text
OUTSIDE_LANGUAGE
```

---

# 19. Diagnostic JSON → formal roots 的完整 bridge

## 19.1 为什么必须有 bridge

当前两套 identity 的差异是实质性的：

| 层 | Preimage | Aggregation |
|---|---|---|
| diagnostic ID | RFC8785 JCS whole object | single SHA-256 |
| formal content digest | canonical CBOR whole object | domain-separated SHA-256 |
| formal row root | canonical CBOR row leaves | RFC6962 |
| archive root | canonical CBOR program/output records | RFC6962 |

因此不能通过字符串转换连接。

## 19.2 Universe row schema

```text
BoundedUniverseRowV1 =
[
  1,
  universe_index,
  input_signature_id,
  canonical_input_object
]
```

按 `universe_index` 升序做 RFC6962：

```text
bounded_universe_root
```

## 19.3 Target row schema

```text
TargetTruthRowV1 =
[
  1,
  universe_index,
  canonical_input_hash,
  target_output
]
```

按 `universe_index` 升序：

```text
target_truth_table_root
```

## 19.4 Hidden sink 独立 roots

outside target 和 null control 必须分别拥有：

```text
odd_reduction_bounded_universe_root
odd_reduction_target_truth_table_root

hidden_sink_bounded_universe_root
hidden_sink_target_truth_table_root
```

formal root 和 diagnostic ID 均不得复用。

## 19.5 Bridge verification

独立 verifier 必须从 diagnostic artifact：

1. 验证 JCS diagnostic ID；
2. 用冻结 transform spec 转换成 formal rows；
3. strict CBOR encode；
4. 重算 row hashes；
5. 重算 RFC6962 root；
6. 与 bridge 中 formal root 比较；
7. 验证 bridge 3/3 signatures。

bridge 不是“命名表”；它是可重放转换证书。

---

# 20. Execution manifest strong binding

## 20.1 Execution manifest body

```text
ExecutionManifestBodyV1 =
[
  1,
  freeze_version_id,
  repository_commit_id,
  source_tree_digest,
  python_binary_digest,
  rust_binary_digest,
  python_canonicalizer_source_root,
  rust_canonicalizer_source_root,
  python_enumerator_source_root,
  rust_enumerator_source_root,
  cbor_profile_root,
  ast_schema_root,
  dsl_spec_root,
  operator_semantics_root,
  equivalence_contract_root,
  mdl_code_table_root,
  identifier_registry_root,
  q32_golden_vector_root,
  cbor_golden_vector_root,
  rfc6962_golden_vector_root,
  baseline_spec_root,
  phase2b_exact_freeze_root,
  input_artifact_roots,
  environment_image_digests
]
```

manifest root：

```text
ContentHash(
  "HEGEL/EXECUTION_MANIFEST/V1",
  ExecutionManifestBodyV1
)
```

## 20.2 强绑定原则

formal receipt/certificate 中禁止只保存 human-readable IDs。
必须保存 roots。

ID 用于阅读；root 用于 identity。

---

# 21. Strict implementation 的黄金向量

至少冻结以下测试。

## 21.1 CBOR

- 每种 node kind；
- 23/24、255/256 integer boundary；
- negative integer boundary；
- byte string length boundary；
- forbidden float；
- forbidden text；
- forbidden map；
- indefinite array rejection；
- trailing byte rejection；
- non-shortest integer rejection。

## 21.2 AST

- explicit Bit coercion；
- implicit coercion rejection；
- `greater_equal → less_equal`；
- `approx_equal tolerance 0 → equal_exact`；
- commutative reorder；
- add flatten/right-association；
- duplicate AND removal；
- AND1 removal；
- scope alias rejection；
- scope-clause sorting；
- aggregate counts one node；
- tolerance not a child；
- new symbol rejected in old DSL。

## 21.3 RFC6962

- empty tree；
- 1、2、3、4、5、4095、4096、4097 leaves；
- chunk subroot；
- global root；
- prohibit duplicate-last behavior。

## 21.4 Bridge

- same diagnostic payload → same diagnostic ID；
- same formal rows → same formal root；
- diagnostic ID 不等于 formal root；
- one changed row changes formal root；
- wrong namespace / row transform fails。

## 21.5 Q32

至少：

```text
N = 1
2
3
4
5
8
10
480
481
binom(192,0)
binom(192,1)
binom(192,96)
binom(480,240)
```

Python/Rust uint64 outputs必须 bit-identical。

---

# 22. 当前状态与下一步

## 22.1 当前保持关闭

```json
{
  "phase2b_formal_exit": false,
  "phase3a_closure_status": "NOT_RUN",
  "dsl_too_large_claim": false,
  "outside_frozen_closure_certificate": false,
  "formal_cbor_roots": null,
  "formal_rfc6962_roots": null,
  "active_promotion": false
}
```

## 22.2 下一施工顺序

1. 实现 scope alias migration，formal 拒绝 alias；
2. 实现 explicit Bit coercion typing；
3. 实现 Python/Rust strict AST encoders/decoders；
4. 完成 canonical rewrites 和 node count golden tests；
5. 对 64,680 constructive subset 做 strict canonicalizer replay；
6. 仅由该结果决定：
   - 是否触发 `DSL_TOO_LARGE`；
   - 或继续完整 closure；
7. 实现 program/output archives；
8. 实现 diagnostic-formal bridge；
9. 实现 dual exhaustion receipts；
10. 实现 key-status chain和 3/3 certificate；
11. 实现 MDL dual replay；
12. 最后才生成非 null formal roots。

## 22.3 Go / No-Go

| 工作 | 决定 |
|---|---|
| strict canonicalizer施工 | GO |
| diagnostic bridge施工 | GO |
| closure archive施工 | GO |
| hidden sink control施工 | GO |
| 64,680 strict preflight | GO |
| 声称 DSL_TOO_LARGE | NO-GO，等待 preflight |
| 完整 closure运行 | 等 preflight 决定 |
| outside certificate | NO-GO |
| Phase-2B sealed generation | 仍等待全部 exact-freeze infrastructure |
| ACTIVE | NO-GO |

---

# 23. 最终主线判断

当前最重要的成果不是“64,680 大于 50,000”，而是你们已经把它正确降格为：

```text
conditional candidate-AST capacity evidence
```

并拒绝将其升级成 closure verdict。

本 amendment 之后，以下四个对象将拥有唯一 identity：

1. **program**：strict canonical AST CBOR bytes；
2. **program behavior**：在绑定 universe 上的 canonical output blob；
3. **closure**：有序 ProgramRecord / ProgramOutputRecord 的 RFC6962 roots；
4. **claim**：绑定 DSL、universe、target、closure、replay和 trust epoch 的 certificate body。

这正是 Phase-3A 能否成立的可信基础。

只要继续坚持：

\[
\boxed{
\text{先冻结 representation identity，再讨论 language adequacy}
}
\]

当前方向就是正确的。
