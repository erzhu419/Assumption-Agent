# Hegel Machine Phase-3：Shrink-6 耗尽后的规范方向决策

**文档类型**：Post-capacity normative direction amendment  
**建议 human document ID**：`hegel-phase3-post-shrink6-quotient-direction-v1`  
**当前 DSL**：`hegel-old-dsl-v1.6.0`  
**当前证据基线**：Source Y `5217568303d5c7f902682c092750f637c64f080a`

## 0. 最终决定

```yaml
selected_primary_path: "C_THEN_B_IF_NEEDED"

rationale: >
  选择 C3_HYBRID 作为新的主路径。六步 shrink 全部耗尽后，
  rank-50,001 witness 仍位于 depth 2，并在最后两次结构缩减中保持，
  说明当前主要瓶颈已不是某个高深度 operator 尚未删除，而是
  syntactically canonical AST 被当作 closure identity 所造成的表示乘法。
  科学问题真正关心的是冻结 DSL 能产生哪些不同的可观察行为，而不是
  有多少种不同语法写法。因此先把 closure 改写为 exact quotient 是
  对研究对象的修正，而不是为了 benchmark 过关而缩小语言。
  若 target-blind quotient capacity qualification 仍然不可行，再允许一个
  全新、单独预注册的 B 类 target-neutral language adaptation。
  不允许实现根据 target match 结果决定是否进入 B。

path_A:
  disposition: "REJECT"
  new_exact_canonical_program_budget: null
  new_raw_operator_application_budget: null
  anti_post_hoc_constraints_approved: true

path_B:
  disposition: "CONDITIONAL"
  target_neutrality_rule: >
    B 只能在 C3 的 target-blind quotient capacity/completeness qualification
    无法在预冻结资源界限内完成时重新提出；触发 B 之前不得读取 parity truth labels、
    sink truth labels、hidden split membership、role match result、synthesis trace
    或任何 target-conditioned score。若 C3 已 COMPLETE，则无论 target 是 in-language
    还是 outside，都禁止因为结果不理想而进入 B。
  allowed_high_level_shrink_scope: >
    首选 quotient-preserving shrink：删除或降格那些在公开 qualification universes
    上被 exact proof 证明不增加任何 reachable behavior、且可由更简单构造替代的
    primitive/operator/parameterization；其次才允许基于公开 grammar structure、
    interpretability、resource-signature growth 和 target-free capacity analysis
    缩小 parameter grid、scope family 或 structural forms。任何真正减少 reachable
    behavior set 的变化都必须成为新 DSL version，并承认它是更窄的新语言。
  post_capacity_adaptation_label_required: true

path_C:
  disposition: "PRIMARY"
  quotient_variant: "C3"
  closure_identity_definition: >
    对每个冻结 input signature σ 与 public frozen universe U_σ，
    每个 admitted AST p 的 exact denotation 定义为
    B_U(p) = (output_sort(p), [eval(p,x)]_{x∈U})，其中 output vector
    包含显式 bottom/undefined sentinel。两个 AST 在且仅在 output sort 相同且
    整个 frozen-universe behavior vector bit-exact 相同时属于同一 extensional class。
    Symbolic normalization 只能作为 target-neutral 的预压缩和推导工具；final closure
    identity 由 exact behavior class 决定。为保证 structural limits 下 quotient pruning
    不漏掉未来可组合路径，每个 behavior class 必须维护所有 future-admissibility-relevant
    construction signatures 的 Pareto frontier，而不能只保存一个任意 AST。
    每个 frontier point 保存一个可审计的 admitted representative；MDL 同时维护该 class
    中最短 admitted representative 的 exact code length。COMPLETE 的含义改为：
    在冻结 grammar、structural limits、input signature、universe 和 quotient semantics 下，
    least fixed point 已饱和，并有独立 completeness evidence 证明每个 bounded admitted AST
    的 quotient image 都包含于该 reachable behavior set。
  old_claims_abandoned:
    - "不再要求归档所有 syntactically canonical AST 才能判 closure complete"
    - "canonical_program_count 不再作为 closure cardinality"
    - "AST archive 不再作为 semantic closure identity；只保留 representative/provenance 角色"
    - "不再使用旧的 exact-syntactic COMPLETE 语义作为 Phase-3 exit"
    - "不再把旧 OUTSIDE_FROZEN_CLOSURE 名称用于 quotient 结果"
    - "有限-universe quotient 结果不得外推为所有可能输入上的语言不可表达性"
  replacement_certificate_claim: >
    OUTSIDE_FROZEN_QUOTIENT_CLOSURE(
      dsl_version,
      closure_semantics_version,
      input_signature,
      frozen_universe_root,
      equivalence = exact_behavior_vector_with_bottom,
      target_truth_root
    ):
    在绑定的 DSL、结构约束、input signature、frozen universe 与 exact quotient semantics 下，
    已完整证明所有 bounded admitted AST 的 reachable behavior classes；target 的 exact behavior
    vector 不属于该 reachable set。该证书只证明 frozen universe 上的 extensional non-membership，
    不证明该 relation 在任意输入域上不可由该 DSL 表达。

path_D:
  disposition: "FALLBACK"
  strongest_publishable_negative_claim: >
    在冻结的 typed DSL、canonicalization、50,000 canonical-program boundary、raw guard
    和六步 target-neutral shrink schedule 下，Python/Rust 独立实现与 host replay 均复现
    最终 DSL 的 syntactic capacity overflow；rank-50,001 witness 位于 depth 2，并在最后两步
    node/depth restriction 中保持不变，因而最后两次结构缩减没有消除已经存在于低深度的
    syntactic multiplicity boundary。该结果足以作为可复现的工程/方法学负结果，并应无论 C
    最终成功与否都保留为历史证据。
  permanently_forbidden_claims:
    - "parity relation 已被证明 outside hegel-old-dsl-v1.6.0"
    - "hegel-old-dsl-v1.6.0 全域不可表达 parity"
    - "symbolic/extensional quotient 也必然失败"
    - "2237 residual 等于剩余完整 closure 大小"
    - "把 canonical budget 事后调到 52237 即可宣称 COMPLETE"
    - "六步 shrink 失败否证了 Hegel Machine hypothesis invention 原理"
    - "容量负结果本身构成自主新关系发明证据"

target_and_control:
  retain_generic_odd_cardinality_target: true
  retain_hidden_sink_false_invention_control: true

split_continuity:
  decision: "KEEP_EXISTING"
  reason: >
    closure representation 的改变不构成重抽 hidden split 的理由。
    若 seed/assignment 已权威实例化，则原样保持，只通过新的 closure-semantics binding
    重新绑定；若尚未实例化，则继续沿用已经冻结的 split contract，执行原本的 first
    instantiation，而不是因为 C3 路线重新抽取一个 seed。只有 input membership domain
    本身发生变化时，才允许另行提出 RECOMPUTE_MEMBERSHIP_ONLY；不得为了改善结果使用新 seed。

formal_reset:
  m3_remains_not_run_until_requalified: true
  root_and_gate_reset_principle: >
    所有语义上依赖 closure representation、equivalence、closure cardinality、archive identity、
    termination condition、capacity limit、role membership 或 certificate semantics 的 roots、
    execution manifests、receipts 和 M3 gates 全部失效，必须在 C3 新规范下从头双重资格验证。
    不能把旧 24/24 readiness 计数直接增量继承。与 quotient 语义无关且可证明 canonical bytes
    未变的 DSL grammar/typing、target/control payload、split custody/history、actor trust、seed commitment、
    hidden-access ledger、strict CBOR/AST profile 可作为 historical provenance 通过新 binding 重用，
    但它们不能自动使新的 quotient gate 通过。
  unaffected_genesis_evidence_reuse_rule: >
    只有当 object preimage bytes、actor/key epoch、seed commitment、split assignment 和 access history
    均 bit-identical，且新 manifest 明确绑定其旧 root 时才可重用；任何涉及新 closure/equivalence
    contract 的 object 必须新生成。

minimum_evidence:
  nine_item_baseline_approved: true
  additional_normative_evidence_categories:
    - "QUOTIENT_CONGRUENCE_PROOF: exact behavior equality 必须对所有 admitted operators 构成 congruence；任何 syntax-sensitive operator 必须显式进入 construction signature"
    - "FUTURE_ADMISSIBILITY_SIGNATURE_SUFFICIENCY: 证明 quotient class 保留的 construction signature 包含所有会影响后续 type/admission/structural-limit 的属性"
    - "PARETO_DOMINANCE_SOUNDNESS: 证明删除 dominated representative 不会删除任何未来 admissible composition"
    - "STRUCTURAL_INDUCTION_COMPLETENESS: leaves 完备 + operator closure + resource-signature preservation 推出所有 bounded admitted AST 均映射到 reachable quotient"
    - "FIXPOINT_SATURATION_EVIDENCE: 双实现与 host replay 对完整 reachable class set、frontier、counts 和 roots 一致"
    - "TARGET_LABEL_ISOLATION_AUDIT: quotient design、capacity qualification、equivalence freeze 和 closure completion 在读取 target truth/match 前完成"
    - "CLASS_MDL_MINIMALITY: class MDL 定义为同 class 所有 admitted AST 中的最短 code length，并证明搜索过程不会因 quotient pruning 丢失更短 representative"
    - "SYNTACTIC_TO_SEMANTIC_MULTIPLICITY_CURVES: target-free 报告每 depth/operator 的 syntactic count、behavior-class count、Pareto-frontier count 和 new-class yield"
    - "QUOTIENT_COLLISION_ADVERSARIAL_CONTROLS: 同语义异语法必须合并、不同语义不得合并、bottom/undefined 差异必须保留"
    - "C3_CAPACITY_PREFLIGHT: full run 前用 target-free growth model 冻结 quotient-class、raw-application、wall-time、memory 上限和所有 terminal routes；subset 不得写成 COMPLETE"

web_gpt_scope_acknowledgement:
  direction_and_claims_only: true
  wire_cbor_tags_and_engineering_details_deferred_to_codex: true
```

---

# 1. 为什么 C3 是“修正研究对象”，不是降标准

当前 exact closure 把每个 syntactically canonical AST 都当作不同 closure item。这个定义对“完整归档所有程序”是自然的，但对 Phase-3 的真正问题——**旧语言是否已经能表达某个关系**——过强。

如果两个 AST 在当前可观察域上行为完全相同：

\[
\llbracket p\rrbracket_U=\llbracket q\rrbracket_U,
\]

那么对 bounded extensional expressibility 来说，它们并不是两个不同假设，而只是同一个行为的两个坐标表示。

因此原来的对象是：

\[
\mathcal P_G=\{\text{syntactically canonical programs}\},
\]

而真正与 Phase-3 membership 对应的对象应当是：

\[
\mathcal Q_{G,U}=\mathcal P_G/\sim_U.
\]

这与 Hegel Machine 之前讨论的商空间、可区分性和“只有当前 probes 能区分的差异才应进入 geometry”是同一条数学主线。

---

# 2. C3 的核心状态：behavior class + construction-signature Pareto frontier

单纯把所有 AST 按 behavior vector 去重还不够。

原因是 DSL 有结构资源约束，例如：

- depth；
- node count；
- distinct bit slots；
- aggregate leaves；
- scope clauses；
- composition depth；
- fitted parameter count；
- 其他会影响后续 operator admission 的离散 flags。

两个 AST 即使行为相同，也可能具有不同的未来组合能力。

因此每个 exact behavior class 应维护：

```text
BehaviorClass
  exact_output_sort
  exact_behavior_vector_with_bottom
  ParetoFrontier[
    FutureAdmissibilitySignature
      -> admitted representative
  ]
  minimum_MDL_admitted_representative
```

只有当一个 representative 在**所有未来相关资源维度**都被另一个 representative 支配时，才允许删除。

这一步是 C3 是否保持 exact 的关键。

---

# 3. C3 必须证明的四个数学条件

## 3.1 Congruence

对所有 admitted operator \(F\)：

\[
p_i\sim_Uq_i\;\forall i
\Longrightarrow
F(p_1,\ldots,p_k)\sim_UF(q_1,\ldots,q_k),
\]

前提是两边都合法。

如果某个 operator 会读取 AST identity、字符串、provenance 或其他 behavior vector 中没有的信息，那么它不能被纯 extensional quotient 安全替换。此时必须把那部分未来相关状态加入 construction signature，或者将该 operator 从这个 quotient contract 中排除。

## 3.2 Dominance soundness

若：

\[
\rho(p)\preceq\rho(q),
\]

则用 \(p\) 替换 \(q\) 不能让原来合法的 parent composition 失去合法性。

## 3.3 Structural induction completeness

必须由：

```text
all leaves represented
+ all operator expansions covered
+ admission-signature information sufficient
+ Pareto pruning sound
```

推出所有 bounded admitted AST 的 quotient image 都被生成。

## 3.4 Fixed-point saturation

只有达到真正的 least fixed point：

\[
Q_{t+1}=Q_t
\]

才能报告 quotient `COMPLETE`。

“增长变慢”或“很久没出现新 class”都不能代替这一条件。

---

# 4. Symbolic 层与 extensional 层的关系

选择 C3 而不是 C2，是因为 symbolic normalization 仍然值得保留：

```text
source AST
→ frozen symbolic normalization
→ future-admissibility signature
→ exact behavior vector
→ extensional quotient class
```

但最终 identity 必须是 exact behavior，而不是 symbolic label。

也就是说：

```text
symbolic = compression / proof aid
extensional vector = bounded semantic identity
```

如果 symbolic normalizer 将两个在 frozen universe 上不同的 behavior 合并，formal gate 必须失败。

---

# 5. MDL 应改成 equivalence-class 最短描述

对 class \(C\)：

\[
L(C)=\min_{p\in C,\;p\;admitted}L_{DSL}(p).
\]

因此不能：

- 随便拿第一个 representative 的长度；
- 用一个并非 old DSL admitted AST 的 symbolic normal form 长度；
- 因 quotient pruning 丢掉更短的后续 representative。

C3 engine 必须维护 exact minimum admitted MDL。

---

# 6. 新 certificate 的科学含义

推荐停止使用容易和旧 syntactic closure 混淆的：

```text
OUTSIDE_FROZEN_CLOSURE
```

改成：

```text
OUTSIDE_FROZEN_QUOTIENT_CLOSURE(
  dsl_version,
  closure_semantics_version,
  input_signature,
  frozen_universe_root,
  equivalence = exact_behavior_vector_with_bottom,
  target_truth_root
)
```

它只表示：

> 在该 frozen universe 上，所有 bounded admitted AST 的 exact behavior image 已经完整覆盖，而 target behavior 不在其中。

它**不**表示：

> parity 在任意输入域上都无法由该 DSL 表达。

如果未来希望获得这种更强 claim，需要另外做 C1 symbolic/global proof。

---

# 7. B 只能在“看 target 之前”发生

允许 B 的唯一合理触发是：

```text
C3 exactness obligations 已通过
+ target truth 尚未读取
+ role match 尚未执行
+ target-free quotient capacity 仍不可接受
```

然后单独提交新 normative amendment。

尤其禁止：

```text
发现 parity 在 old quotient closure 中
→ 再 shrink language
→ 直到 parity outside
```

这会把整个 outside benchmark 变成目标导向的 target shopping，必须明确禁止。

---

# 8. B 若发生，优先做 behavior-preserving shrink

未来 B 的优先顺序应为：

## B1：Quotient-preserving

例如删除：

- 从不产生新 behavior class 的 primitive；
- 永远有更便宜等价构造的 syntactic sugar；
- 只产生 dominated construction signatures 的 forms。

若可证明：

\[
Q_{G,U}=Q_{G',U},
\]

这是最干净的 shrink。

## B2：真正缩窄语言

如果 B1 仍不足，才允许真正减少 reachable behavior set。

此时必须承认：

```text
这是新 DSL
不是 v1.6 的完成
certificate 只对新 DSL 有效
```

---

# 9. A 为什么不该继续

直接提高 syntactic budget 的问题不是“花钱多”，而是它继续回答：

> 有多少种不同语法写法？

而不是：

> 旧语言有多少种不同可观察行为？

所以本轮不批准 A 作为主路径，也不批准用更大的旧 syntactic budget 作为救援。

C3 自己当然需要新的 quotient-class / raw / memory / wall-time guards，但那属于新 closure semantics 的 termination contract，而不是对旧 50,000 syntactic budget 的事后放大。

---

# 10. D 的负结果仍然应永久保留

无论 C3 最终是否成功，Source Y 的结果都应该作为独立结果保留。

建议后续 target-free 报告：

- 每个 shrink version 的 canonical syntax growth；
- boundary bucket size；
- first out-of-budget depth/node count；
- raw applications per new syntax；
- shrink-4/5/6 witness preservation；
- 若 C3 完成，再报告 syntax-to-behavior multiplicity ratio。

如果 C3 成功，这个负结果反而会成为“为什么必须取商”的直接实验动机。

---

# 11. Target 与 split 不动

继续冻结：

```text
Generic Odd-Cardinality Reduction
observed omitted-sink false-invention control
```

不换 target，避免 target shopping。

split 也不重抽。representation change 不是换 split 的理由。

---

# 12. Formal reset 层级

## 可继承的历史 provenance

若 preimage bit-identical，可重新绑定：

- DSL grammar/typing；
- strict AST/CBOR；
- target/control payload；
- universe payload；
- custody / trust；
- seed / split history；
- Source-Y syntactic capacity evidence。

## 必须全部重建

- quotient equivalence contract；
- closure semantics；
- construction signature；
- Pareto policy；
- class/archive identity；
- quotient capacity/termination；
- M3 execution manifest；
- completion receipts；
- role match roots；
- certificate semantics。

因此旧 readiness 不能“加一个 quotient gate”继续沿用，必须建立新的 C3 readiness。

---

# 13. 建议的新阶段命名

## Phase-3A-Q0 — Exact Quotient Closure Semantics Qualification

任务：

1. 冻结 C3 equivalence；
2. 冻结 future-admissibility signature；
3. 逐 operator congruence；
4. Pareto soundness；
5. 在小型 closure 上和 exhaustive syntax→quotient oracle 做完全等价；
6. Python/Rust 独立 qualification；
7. host replay；
8. target-blind growth preflight；
9. 冻结 full-run resource envelope。

## Q0 PASS 后：Phase-3A-Q1 — Complete Frozen Quotient Closure

完全生成 reachable quotient closure，仍不读取 target truth。

## Q1 COMPLETE 后：Phase-3A-Q2 — Role Membership Evaluation

```text
quotient closure sealed
→ open target truth
→ parity membership
→ sink false-invention control
```

然后才进入 certificate / synthesis。

---

# 14. 最终科研判断

Codex 的 `C3_HYBRID` 推荐是当前最合理的方向，我批准，而且理由不只是工程成本。

现在暴露的其实是一个数学对象选择问题：

原来我们枚举：

\[
\mathcal P_G,
\]

现在应该研究：

\[
\mathcal P_G/\sim_U.
\]

也就是：

\[
\boxed{\text{先取商，再讨论可表达性。}}
\]

这与 Hegel Machine 之前关于“商空间、可区分性、由观测定义等价关系”的理论方向是完全一致的。

正式路线因此冻结为：

```text
C3 PRIMARY
→ B only after target-blind quotient infeasibility and a new normative review
→ D if exactness/completeness still cannot be certified
```

A 不再作为 old syntactic closure 的救援路线。
