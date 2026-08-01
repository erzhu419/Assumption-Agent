# Hegel Machine Phase-3 Freeze Readiness：状态迁移与施工准入决策

**文档类型**：Normative readiness resolution<br>
**建议版本**：`hegel-phase3-freeze-readiness-v1.0.0`<br>
**依赖规范**：

- `hegel-freeze-p2b-p3-v1.0.1`
- `hegel-freeze-p2b-p3-v1.0.2`
- `hegel-old-dsl-v1.0.0`
- `hegel-mdl-prefix-v1.0.0`

**证据基础**：本文件依据 `phase3_freeze_readiness.md` 中记录的 implementation-audit 状态，以及随后冻结的 v1.0.2 strict canonical acceptance / certificate bridge 决策。它是 readiness 与 go/no-go 判定，不替代 source-level 双实现 replay。

> **POST-M2 CURRENT-STATE ADDENDUM（2026-08-01）**
>
> 本文的状态机、触发条件和 fail-closed 规则继续有效；其中把 implementation
> 写成未验证、把 64,680 写成 conditional、把执行状态写成 `NOT_RUN` 的“当前”
> 段落，均保留为 M1/M2 前的 readiness snapshot。
>
> M1 已完成：Python/Rust shared strict vectors 各 `48/48` 且 identity 相同。
> M2 已完成：两端对同一 64,680 source subset 均得到 64,680 个 unique canonical
> AST、零 reject、零 collapse，并共享 diagnostic set commitment
> `sha256:c1a02a66a8d6f75204cb3daf03ab0b01c2b3b8e486d0ab3d481ee3be43c930`。
> 因此 `hegel-old-dsl-v1.0.0` 在 50,000 syntactic budget 下的 bounded 状态已是
> `DSL_TOO_LARGE`，但不是 `COMPLETE`，没有 extensional target verdict。
>
> 当前 `formal_roots=null`；outside/MDL certificate、target synthesis、hidden-sink
> formal verdict、Phase-2B formal exit 与 ACTIVE 全部关闭。下一动作按本文 Path C：
> 发布新 DSL version，执行 frozen shrink step 1（删除 `mean_v1`、`min_v1`、
> `max_v1`），重建 commitments，并让新版本从 `NOT_RUN` 重新运行 closure。

> **POST-SHRINK-1 EXECUTION ADDENDUM（2026-08-01）**
>
> 上述下一动作已执行到 child diagnostic publication。新 IDs 为
> `hegel-old-dsl-v1.1.0` / `hegel-freeze-p2b-p3-v1.1.0`；Python/Rust 对 child
> vectors 与 25,872-source subset 完全一致，subset commitment 为
> `sha256:653fcb9428684cfed11c3f2345ac95ed98ded6e31564c9eeabf97c57ee71a7e9`。
> 该结果没有 50,001 witness，但也不是 full closure。Child execution state 仍为
> `NOT_RUN`，formal roots 仍为 null。由于 parent split seed/binding/custodian evidence
> 未曾物化，M3 entry fail-closed；当前下一硬门是 seed continuity 决策/证据与 dual
> formal bridge root generation。

---

# 0. 结论

`phase3_freeze_readiness.md` 对当时状态的判断是正确的：

```text
surface_parameter_freeze_complete = true
strict_acceptance_contract_complete = false
normative_parameter_freeze_complete = false
```

但在 v1.0.2 strict acceptance amendment 生效后，规范状态应更新为：

```json
{
  "surface_parameter_freeze_complete": true,
  "strict_acceptance_contract_complete": true,
  "normative_parameter_freeze_complete": true,
  "strict_acceptance_implementation_verified": false,
  "formal_root_generation_allowed": false,
  "executed_closure_status": "NOT_RUN",
  "target_synthesis_allowed": false,
  "outside_certificate_allowed": false,
  "active_promotion_allowed": false
}
```

核心区分是：

\[
\boxed{
\text{规范已冻结}
\neq
\text{实现已验证}
\neq
\text{closure 已执行}
}
\]

当前已可以开始 strict implementation 和 replay 施工，但仍不能：

- 将 64,680 升格为 canonical-program count；
- 声称 `DSL_TOO_LARGE`；
- 开始 hidden target synthesis；
- 签发 `OUTSIDE_FROZEN_CLOSURE`；
- 签发 MDL certificate；
- 将任何结果晋升 ACTIVE。

---

# 1. readiness 文档的定位

`phase3_freeze_readiness.md` 应被保留为：

> **pre-v1.0.2 implementation-audit snapshot**

它不再是当前 normative truth。

建议在文件头增加：

```text
STATUS: SUPERSEDED_FOR_NORMATIVE_DECISIONS
SUPERSEDED_BY: hegel-freeze-p2b-p3-v1.0.2
HISTORICAL_VALUE: preserves the audit state that motivated strict acceptance freeze
```

不要删除该文件，因为它证明：

- 64,680 没有被错误宣传成 closure 结果；
- strict acceptance 缺口是在 target outcome 前暴露；
- source aliases、typing 和 certificate roots 没有由实现自行猜测；
- formal roots 在未执行时保持 `null`。

这些属于重要的审计证据。

---

# 2. 待消歧项的最终状态

## 2.1 scope alias

最终状态：

```json
{
  "decision_id": "P3_SCOPE_BASELINE_ALIAS",
  "status": "RESOLVED",
  "machine_scope_id": "scope_primary_only_v1",
  "deprecated_source_alias": "control_volume_primary_only_v1",
  "semantic_identity": true,
  "fifth_scope_added": false,
  "formal_canonicalizer_accepts_alias": false
}
```

约束：

- legacy migration adapter 可以重写 alias；
- strict canonical AST 中出现 alias 必须拒绝；
- closure token vocabulary 仍有 4 个 scope；
- 不发布新 DSL version。

## 2.2 Bit → RationalValue

最终状态：

```json
{
  "decision_id": "P3_BIT_RATIONAL_COERCION",
  "status": "RESOLVED",
  "implicit_coercion": false,
  "type_explicit_xor_witness": "absolute(difference(bit_to_scalar(bit_at(0)), bit_to_scalar(bit_at(1))))",
  "source_shorthand_typechecks": false,
  "formal_xor_language_verdict": null
}
```

约束：

- parser 不补 coercion；
- canonicalizer 不补 coercion；
- `difference(Bit, Bit)` 是 type rejection；
- binary XOR 是否 `IN_LANGUAGE` 仍需完整 executable replay；
- 文档 shorthand 不构成 executable witness。

## 2.3 strict canonical AST / CBOR

状态：

```json
{
  "decision_id": "CANON_AST_NODE_CBOR_SCHEMA",
  "status": "RESOLVED_IN_SPEC",
  "implementation_verified": false
}
```

已冻结：

- numeric-tag array AST；
- no-map/no-text/no-float canonical CBOR subset；
- exact re-encode decoder acceptance；
- node tags；
- operator IDs；
- root operator extraction；
- scope extension ordering；
- AND schema；
- explicit coercion；
- decoder rejection rules。

## 2.4 normalization 与 count boundary

状态：

```json
{
  "decision_id": "SYNTACTIC_NORMALIZATION_AND_REWRITE",
  "status": "RESOLVED_IN_SPEC",
  "implementation_verified": false
}
```

允许的 pre-count rewrite 仅包括：

```text
commutative child ordering
greater_equal(a,b) -> less_equal(b,a)
approx_equal(a,b,0) -> equal_exact(a,b)
AND flatten/sort/deduplicate/AND1 unwrap
add flatten/sort/right-associate
add(x,0) -> x
difference(x,0) -> x
difference(x,x) -> 0
absolute(absolute(x)) -> absolute(x)
limited constant folding inside frozen parameter grid
```

明确禁止：

```text
extensional quotient
target-aware simplification
SMT-derived equivalence merging
distributivity/factorization
arbitrary algebraic normalization
```

50,000 只计经过上述规则后的 distinct strict canonical AST bytes。

## 2.5 node counting

状态：

```json
{
  "decision_id": "AST_NODE_COUNTING",
  "status": "RESOLVED_IN_SPEC"
}
```

规则：

- leaf = 1 node；
- unary/binary/ternary operator = 1 node；
- top-level AND wrapper = 1 node；
- aggregate = 1 leaf；
- map/scope/quantity/scope clauses不计额外 AST node；
- tolerance 是 inline parameter，不计 child；
- scope clauses影响 MDL，不影响 AST depth/node count。

---

# 3. 九组 certificate / MDL item 的 readiness 状态

| Group | 规范状态 | 实现状态 |
|---|---|---|
| `CERT_01_CANONICAL_CBOR_PROFILE` | 已冻结 | 未双实现验证 |
| `CERT_02_CANONICAL_AST_IDENTITY` | 已冻结 | 未双实现验证 |
| `CERT_03_PROGRAM_OUTPUT_ARCHIVE` | 已冻结 | 未生成正式 archive |
| `CERT_04_PROGRAM_ARCHIVE_ROOT_RELATIONS` | 已冻结 | bridge 未执行 |
| `CERT_05_MATCH_PROGRAM_HASH_IDENTITY` | 已冻结 | match replay 未执行 |
| `CERT_06_EXHAUSTION_RECEIPT_ROOT` | 已冻结 | receipt 未重算 |
| `CERT_07_FINAL_ENVELOPE_AND_COMMIT` | 已冻结 | 3/3 envelope 未签发 |
| `CERT_08_KEY_STATUS_AND_REVOCATION` | 已冻结 | trust chain 未部署 |
| `CERT_09_MDL_DUAL_REPLAY_WIRE` | 已冻结 | MDL dual replay 未执行 |

因此：

```json
{
  "certificate_specification_ready": true,
  "certificate_implementation_ready": false,
  "certificate_issuance_ready": false
}
```

---

# 4. 64,680 capacity preflight 的唯一解释

## 4.1 当前状态保持

```json
{
  "status": "CONDITIONAL_CAPACITY_LOWER_BOUND_EXCEEDS_BUDGET",
  "constructive_candidate_ast_count": 64680,
  "canonical_program_budget": 50000,
  "strict_canonicalizer_acceptance_verified": false,
  "executed_closure_status": "NOT_RUN",
  "dsl_too_large_claim_allowed": false
}
```

## 4.2 不能做的推断

不允许：

```text
64,680 tuple/JSON representations
→ 64,680 strict canonical programs
→ DSL_TOO_LARGE
```

原因包括：

- aliases 可能被拒绝；
- typing 可能拒绝；
- canonical rewrites 可能合并；
- structural limits 必须按 strict AST 重算；
- tuple identity 不等于 canonical CBOR identity。

## 4.3 strict preflight 的运行合同

输入：

```text
exact same constructive subset generator
+
v1.0.2 strict parser/typechecker/canonicalizer
```

输出必须包括：

```json
{
  "source_candidate_count": 64680,
  "type_rejected_count": 0,
  "limit_rejected_count": 0,
  "rewrite_collapsed_count": 0,
  "accepted_strict_canonical_count": 0,
  "first_accepted_out_of_budget_ast_hash": null,
  "python_subset_root": null,
  "rust_subset_root": null,
  "dual_replay_equal": false
}
```

实际值由运行填写，不能预置。

## 4.4 触发 `DSL_TOO_LARGE` 的必要充分条件

必须同时满足：

```text
Python and Rust both process all 64,680 source candidates
same strict schema roots
same accepted canonical AST set
same subset Merkle root
accepted_strict_canonical_count >= 50,001
first accepted program after canonical order position 50,000 exists
no raw cap hit
no wall-clock abort
no semantic incompleteness
```

然后 closure 状态可进入：

```text
DSL_TOO_LARGE
```

但这仍不是完整 closure cardinality。

receipt 必须是：

```json
{
  "closure_status": "DSL_TOO_LARGE",
  "enumerated_canonical_program_count": 50000,
  "first_out_of_budget_program_hash": "<strict canonical AST hash>",
  "frontier_exhausted": false,
  "all_type_buckets_closed": false,
  "closure_cardinality": null,
  "match_set_count": null
}
```

## 4.5 若 strict count ≤ 50,000

不得自动得出 `COMPLETE`。

应继续运行完整 old DSL enumeration。

状态保持：

```text
NOT_RUN
```

直到正式 closure execution 开始；执行中若预算/语义失败，进入相应 `INCONCLUSIVE_*`。

---

# 5. closure 状态机

唯一允许状态：

```text
NOT_RUN
RUNNING
COMPLETE
DSL_TOO_LARGE
INCONCLUSIVE_BUDGET
INCONCLUSIVE_SEMANTICS
INCONCLUSIVE_EXECUTION
```

建议 formal enum：

| State | ID |
|---|---:|
| `NOT_RUN` | 0 |
| `RUNNING` | 1 |
| `COMPLETE` | 2 |
| `DSL_TOO_LARGE` | 3 |
| `INCONCLUSIVE_BUDGET` | 4 |
| `INCONCLUSIVE_SEMANTICS` | 5 |
| `INCONCLUSIVE_EXECUTION` | 6 |

## 5.1 合法迁移

```text
NOT_RUN -> RUNNING

RUNNING -> COMPLETE
RUNNING -> DSL_TOO_LARGE
RUNNING -> INCONCLUSIVE_BUDGET
RUNNING -> INCONCLUSIVE_SEMANTICS
RUNNING -> INCONCLUSIVE_EXECUTION
```

终态不能原地改写。

任何 code/spec 修复后必须：

```text
new execution_manifest
new run_id
state starts at NOT_RUN
```

## 5.2 状态互斥

### `DSL_TOO_LARGE`

```text
program 50,001 accepted
frontier_exhausted = false
closure_cardinality = null
match_set_count = null
```

### `INCONCLUSIVE_BUDGET`

```text
raw cap or declared canonical budget mechanism prevents proof
program 50,001 not validly accepted as overflow witness
```

### `INCONCLUSIVE_SEMANTICS`

```text
operator/undefined/typing behavior missing or implementations disagree
```

不得把 `INCONCLUSIVE_SEMANTICS` 当作 `DSL_TOO_LARGE` 或 target outside evidence。

---

# 6. target synthesis readiness

## 6.1 当前 NO-GO

```json
{
  "target_id": "TARGET_P3A_GENERIC_ODD_REDUCTION_V1",
  "target_synthesis_status": "NO_GO",
  "reason": [
    "strict canonical implementation not dual-verified",
    "closure state is NOT_RUN",
    "target old-language status is unresolved",
    "formal universe/target roots are null"
  ]
}
```

## 6.2 何时可以开始 synthesis

必须先得到以下之一：

### 路径 A：closure `COMPLETE`

且：

```text
match_set_count = 0
Python/Rust roots equal
OUTSIDE_FROZEN_CLOSURE certificate issued
```

然后允许 invention synthesis。

### 路径 B：closure `COMPLETE`

且：

```text
match_set_count >= 1
```

则 odd-cardinality target 降级：

```text
IN_LANGUAGE_POSITIVE_CONTROL
```

按 precommitted fallback registry 选择下一个 target，重新生成 commitments。

### 路径 C：`DSL_TOO_LARGE`

不得开始 synthesis。

必须：

```text
publish new DSL version
apply shrink step 1
regenerate target/validation commitments
rerun closure
```

理由：未得到完整 closure，无法判断 target 是否已在旧语言中。

---

# 7. hidden sink null-control readiness

## 7.1 当前可施工但不可 claim

```json
{
  "control_id": "CONTROL_P3A_OBSERVED_OMITTED_SINK_V1",
  "implementation_construction_allowed": true,
  "formal_in_language_verdict_allowed": false
}
```

## 7.2 正式通过条件

只有 old closure `COMPLETE` 后：

```text
old_closure_exact_match_count >= 1
best_old_program_error == 0
correct program uses:
  signed_balance_v1
  control_volume_all_observed_v1
  q0
  zero tolerance
```

系统行为必须：

```text
decision = IN_LANGUAGE_REFINEMENT
promoted_new_symbol_count = 0
outside_certificate_count = 0
false_invention_rate = 0
```

若 old closure 因 DSL_TOO_LARGE 未完成，则 null-control 也不能获得 formal in-language verdict。

---

# 8. diagnostic IDs 与 formal roots readiness

## 8.1 当前状态

```json
{
  "diagnostic_json_ids_available": true,
  "formal_cbor_row_roots_available": false,
  "diagnostic_formal_bridge_available": false
}
```

## 8.2 必须保持的独立 identities

### outside target

```text
bounded_universe_diagnostic_id
target_table_diagnostic_id
odd_reduction_bounded_universe_root
odd_reduction_target_truth_table_root
```

### null control

```text
hidden_sink_universe_diagnostic_id
hidden_sink_target_table_diagnostic_id
hidden_sink_bounded_universe_root
hidden_sink_target_truth_table_root
```

两组均不得复用。

## 8.3 bridge readiness gate

必须完成：

1. JCS diagnostic ID revalidation；
2. frozen row transform；
3. strict CBOR row encoding；
4. RFC6962 root；
5. Python/Rust root equality；
6. 3/3 bridge signature。

在此之前：

```text
formal roots remain null
```

不得以 diagnostic ID 代替 root。

---

# 9. certificate readiness

## 9.1 当前状态

```json
{
  "closure_receipt": "UNTRUSTED_WIRE_ONLY",
  "python_complete_replay": false,
  "rust_complete_replay": false,
  "program_archive_root": null,
  "program_output_archive_root": null,
  "chunk_manifest_root": null,
  "exhaustion_receipt_root": null,
  "key_status_chain_verified": false,
  "signature_count": 0,
  "outside_certificate": null
}
```

## 9.2 certificate issuance gate

全部满足：

```text
strict implementation golden tests pass
diagnostic-formal bridge valid
closure COMPLETE
Python/Rust canonical count equal
program roots equal
output roots equal
chunk roots equal
target roots equal
match set equal
match_set_count == 0
key chain valid from pinned genesis
3/3 active Ed25519 signatures
```

## 9.3 证书不证明什么

即使签发：

```text
OUTSIDE_FROZEN_CLOSURE(...)
```

它也不证明：

- target 不可由任何程序表达；
- target 是自然界新定律；
- Hegel Machine 已经发明成功；
- 新关系比旧关系更有科学价值；
- ACTIVE promotion 合理。

它只证明：

> 在绑定的 bounded universe 上，没有 strict frozen closure 中的 program 与 target exact-extensionally equivalent。

---

# 10. MDL readiness

## 10.1 当前状态

```json
{
  "mdl_surface_parameters_frozen": true,
  "mdl_canonical_ast_wire_frozen": true,
  "new_symbol_wire_frozen": true,
  "q32_reference_algorithm_frozen": true,
  "python_mdl_replay_complete": false,
  "rust_mdl_replay_complete": false,
  "mdl_certificate_allowed": false
}
```

## 10.2 MDL scorer 开始条件

只有 candidate new relation 已产生后才能运行，但 scorer implementation/golden tests 可以先完成。

正式 scorer 必须重算：

```text
L_old_program
L_train_given_old
L_new_symbol_definition
L_new_call_program
L_train_given_new
delta_L
required_delta_L
```

不得接受 caller-supplied：

```text
Fraction
float length
delta
pass flag
```

## 10.3 MDL 不替代 outside certificate

顺序必须是：

```text
old closure exact status
→ outside certificate
→ invention synthesis
→ predictive gates
→ MDL comparison
→ conservative integration
```

不能通过 MDL 变短，反过来推断旧 closure 中没有等价程序。

---

# 11. Phase-2B 与 Phase-3A 的解耦

这份 readiness 主要是 Phase-3A，但不能让 Phase-3 certificate 反向替代 Phase-2B exit。

## 11.1 Phase-2B 仍未 exit

即便 strict canonical closure 完成，Phase-2B 仍需要：

```text
720 main
240 semantic-conflict
572 derived pairs
baselines
covert-channel audit
trusted wire
untrusted recognizer
sealed evaluator
statistical gates
```

因此：

```json
{
  "phase2b_formal_exit": false,
  "phase3a_spec_freeze": true,
  "phase2b_and_phase3a_are_independent_gates": true
}
```

## 11.2 可以并行的施工

可以并行：

- Phase-2B wire/custodian/baseline；
- Phase-3A strict canonicalizer；
- Python/Rust replay；
- bridge；
- key chain；
- MDL scorer。

不得并行提前运行：

- sealed target synthesis；
- sealed Phase-2B holdout；
- outside certificate；
- ACTIVE promotion。

---

# 12. 施工里程碑

## M0 — Normative Strict Freeze

**状态**：v1.0.2 决策后完成。

输出：

```text
canonical AST schema
CBOR profile
rewrite rules
node count
archive schemas
bridge schema
certificate schema
key schema
MDL replay schema
golden vector specification
```

Gate：

```text
no unresolved normative field
```

## M1 — Dual Strict Acceptance Implementation

输出：

```text
Python parser/typechecker/canonicalizer
Rust parser/typechecker/canonicalizer
golden vector results
source roots
binary digests
```

Gate：

```text
all golden vectors equal
all invalid vectors rejected identically
```

## M2 — 64,680 Strict Capacity Replay

输出：

```text
accepted set
rejected set
collapse accounting
strict subset root
first out-of-budget witness if any
```

Gate：

- 若 accepted ≥ 50,001：`DSL_TOO_LARGE`；
- 否则进入 full enumeration。

## M3 — Full Closure and Archive Replay

输出：

```text
canonical_program_archive_root
program_output_archive_root
chunk_manifest_root
closure status
match set
Python/Rust receipts
```

Gate：

```text
COMPLETE or fail-closed terminal status
```

## M4 — Adequacy Certificate

若 `COMPLETE`：

- match = 0 → outside certificate；
- match > 0 → in-language target transition。

若 `DSL_TOO_LARGE`：

- new DSL version + shrink step；
- no adequacy verdict。

## M5 — Synthesis / MDL

仅在 outside certificate 后：

```text
new relation synthesis
unseen prediction
MDL
conservative integration
null-control false-invention test
```

---

# 13. 当前 go/no-go

| Work item | Status |
|---|---|
| v1.0.2 normative strict freeze | GO / complete in specification |
| strict Python implementation | GO |
| strict Rust implementation | GO |
| golden-vector tests | GO |
| diagnostic-formal bridge implementation | GO |
| 64,680 strict replay | GO after M1 |
| `DSL_TOO_LARGE` claim | NO-GO until M2 result |
| full closure | conditional on M2 |
| odd target synthesis | NO-GO |
| hidden sink implementation | GO / shadow |
| hidden sink formal verdict | NO-GO |
| outside certificate | NO-GO |
| MDL certificate | NO-GO |
| Phase-2B formal exit | NO-GO |
| ACTIVE promotion | NO-GO |

---

# 14. 推荐更新到 readiness JSON

```json
{
  "milestone": "Phase-3A Bounded Frozen-Closure Adequacy",
  "freeze_version": "hegel-freeze-p2b-p3-v1.0.2",
  "surface_parameter_freeze_complete": true,
  "strict_acceptance_contract_complete": true,
  "normative_parameter_freeze_complete": true,
  "strict_acceptance_implementation_verified": false,
  "diagnostic_capacity_status": "CONDITIONAL_CAPACITY_LOWER_BOUND_EXCEEDS_BUDGET",
  "constructive_candidate_ast_count": 64680,
  "executed_closure_status": "NOT_RUN",
  "formal_cbor_archive": false,
  "formal_roots": null,
  "dsl_too_large_claim_allowed": false,
  "target_synthesis_allowed": false,
  "outside_certificate_allowed": false,
  "mdl_certificate_allowed": false,
  "phase2b_formal_exit": false,
  "active_promotion_allowed": false,
  "next_gate": "DUAL_STRICT_CANONICAL_IMPLEMENTATION_AND_GOLDEN_VECTOR_VERIFICATION"
}
```

---

# 15. 最终判断

这份 readiness 文档暴露的问题不是研究方向失败，而是项目已经走到一个必须从“概念正确”进入“对象身份可证明”的阶段。

在当前阶段，最重要的不是 target outcome，而是下面四个等式能否跨实现成立：

\[
\text{Python AST bytes}
=
\text{Rust AST bytes},
\]

\[
\text{Python program set}
=
\text{Rust program set},
\]

\[
\text{Python behavior archive}
=
\text{Rust behavior archive},
\]

\[
\text{diagnostic artifact}
\xrightarrow{\text{frozen bridge}}
\text{formal root}.
\]

只有这四项成立：

- 64,680 才能被解释；
- 50,001st witness 才有意义；
- complete closure 才可信；
- outside-language 的 bounded claim 才能签发；
- 后续 invention 与 MDL 才不是建立在自报 identity 上。

因此当前主线应保持：

\[
\boxed{
\text{strict identity}
\rightarrow
\text{closure adequacy}
\rightarrow
\text{invention}
}
\]

而不是跳过前两步直接让 Agent 发明新关系。
