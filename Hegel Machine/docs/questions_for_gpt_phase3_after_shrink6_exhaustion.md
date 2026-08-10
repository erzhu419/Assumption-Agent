# Phase-3 shrink-6 耗尽后的规范方向决策问题

**文档状态**：`OPEN_NORMATIVE_DIRECTION_REVIEW_REQUEST`

**用途**：请网页端 GPT / 用户审阅 Phase-3 在六步预注册 shrink 全部耗尽后的研究方向与声明边界。

**权限边界**：本文只请求规范路径、科学声明和重新预注册范围的决策；具体实现、wire、CBOR、numeric tag、字段顺序、hash domain、容器参数与测试细节由 Codex 根据双实现和工程测试决定。

**当前基线**：Source Y commit `5217568303d5c7f902682c092750f637c64f080a`。

在新的 normative amendment 被明确批准并先于执行冻结以前：

```yaml
automatic_budget_change_allowed: false
automatic_additional_shrink_allowed: false
automatic_dsl_v1_7_allowed: false
m3_start_allowed: false
formal_root_generation_allowed: false
target_role_evaluation_allowed: false
split_seed_redraw_allowed: false
```

---

## 1. 已验证事实，不再作为开放问题

Source Y 的独立 Python/Rust complete-diagnostic 与 host replay 已通过，但
终态仍是 `DSL_TOO_LARGE`，不是 `COMPLETE`：

```yaml
source_y_commit: 5217568303d5c7f902682c092750f637c64f080a
dual_host_replay: PASS
diagnostic_status: DUAL_DSL_TOO_LARGE_HOST_REPLAY_PASS
closure_status: DSL_TOO_LARGE
canonical_program_budget: 50000
first_out_of_budget_program_ordinal: 50001
raw_operator_application_count: 3120719
residual_out_of_budget_canonical_programs: 2237
witness_ast_depth: 2
prefix_preservation: PASS
preregistered_shrink_steps_consumed: 6
preregistered_shrink_steps_total: 6
next_preregistered_shrink_step: null
m3_state: NOT_RUN
formal_roots: null
```

第六步只删除 depth 4，而 rank 50,001 witness 位于 depth 2。Source Y 的正式
preservation 证明绑定 shrink-5→shrink-6；结合既有 shrink-4/5 evidence，当前
witness、50,000/50,001 边界和相应 prefix 在最后两步结构缩减（node 7→6、
depth 4→3）中保持不变。更早步骤曾出现不同 witness，因此这里不声称它贯穿
六步都未改变。这不是实现错误或双端不一致，而是当前 exact syntactic closure
表示在冻结预算下仍然发生容量爆炸。

`residual_out_of_budget_canonical_programs=2237` 只描述已完全关闭的 boundary
bucket 内部 residual，不是整个剩余 closure 的大小；尤其不得据此把新 canonical
budget 事后设成 52,237 并宣称足够。

由此只能得出一个受限结论：

> 已预注册的六步缩减没有使 `hegel-old-dsl-v1.6.0` 的 exact syntactically
> canonical closure 在 50,000 canonical boundary 与冻结 raw guard 下达到
> `COMPLETE`；到该 closed boundary 已观察到 3,120,719 次 raw operator
> application。

目前仍不能声称：

- `OUTSIDE_FROZEN_CLOSURE(...)` 或无边界的 `OUTSIDE_LANGUAGE`；
- parity target 在旧 DSL 中不可表达；
- hidden-sink control 已被正式求值；
- M3 已启动或 formal roots 已生成；
- 系统已完成自主 hypothesis invention。

---

## 2. 本次必须选择的规范路径

请比较并明确选择 A、B、C、D 中的一条主路径；可以批准有严格先后关系的
组合，例如 `C_PRIMARY_THEN_B_IF_QUOTIENT_STILL_TOO_LARGE`，但不能把多条路径
同时写成可由实现自行选择的开放菜单。

| 路径 | 核心改变 | 可保留的主要主张 | 主要风险 |
|---|---|---|---|
| A | 提高 exact canonical/raw budgets | 保留当前 exact syntactic closure 语义 | 事后追逐预算、资源继续爆炸 |
| B | 新的 target-neutral DSL/closure shrink | 保留 exact closure，但对象变成新版本语言 | shrink 被隐藏 target 污染或削弱研究问题 |
| C | symbolic/extensional quotient closure | 改为对冻结等价类或可达行为的 exact closure | 必须重写 certificate 语义并证明 quotient 完备 |
| D | 停止 exact closure 路线 | 发表可复现的容量负结果和方法学边界 | 放弃 outside certificate 与正式 invention exit |

---

## 3. 路径 A：提高 exact canonical/raw budget

如果选择 A，请给出一次性、具体且可执行的两个新上限：

```yaml
new_exact_canonical_program_budget: <positive integer>
new_raw_operator_application_budget: <positive integer>
```

还必须明确批准以下防事后调参约束：

1. 本次提高预算必须公开标记为
   `POST_CAPACITY_ADAPTATION_AFTER_SHRINK6_EXHAUSTION`，不能继续称为原始
   preregistration；
2. 新上限必须由 target-neutral 的容量模型、资源上限或 termination 分析
   支持，不能由 parity truth labels、sink labels、hidden split 或“刚好超过
   当前 witness”决定；
3. 在新 Source commit 冻结后只允许一次 sealed dual execution，不允许看到
   新边界后继续逐级加预算；
4. 新预算再次耗尽时必须回到新的规范决策，不能自动扩容；
5. canonical budget、raw budget、wall-time、memory 和失败语义必须在执行前
   一并冻结。

请回答：A 是否被批准为主路径、仅作为 C 的有限验证辅助，还是明确拒绝？若
批准，必须填写两个具体整数，不能只回答“适当提高”。

---

## 4. 路径 B：新的 target-neutral DSL / closure shrink

路径 B 不是预注册 shrink step 7。它是六步耗尽后新提出的
post-capacity adaptation，必须形成新的 DSL/freeze version 和新的公开
normative amendment。

若选择 B，请明确：

1. 是否允许基于公开的语言结构、可解释性、静态容量和 Source-Y replay
   选择新的 operator、constant、scope 或 structural restriction；
2. 是否冻结以下不可越过的 target-neutrality 条件：设计和资格验证阶段不得
   读取 parity truth labels、sink truth labels、hidden split、match result 或
   synthesis trace；
3. 新版本是否必须公开标记为
   `POST_CAPACITY_TARGET_NEUTRAL_LANGUAGE_ADAPTATION`，并禁止把它描述成原
   v1.6 closure 的完成；
4. 新 shrink 是否仍需保持 parity target 与 sink control 的问题定义不变，
   以避免同时改变语言和评价目标；
5. 若新语言达到 `COMPLETE`，certificate 是否只能针对新冻结语言，而不能
   回溯声明 v1.6 outside。

请给出 B 的高层允许范围和禁止范围。无需给 operator ID、AST bytes 或 decoder
细节；这些由后续工程 freeze 决定。

---

## 5. 路径 C：symbolic / extensional quotient closure

路径 C 改变的不是 target，而是 closure 的表示单位：不再把每个
syntactically canonical AST 都作为独立 closure item，而是计算冻结 universe
上的可达语义类、symbolic normal form 或 exact behavior vector，并为每个类保留
可审计 representative。

若选择 C，请先在以下语义中作明确决策：

1. `C1_SYMBOLIC_EXACT_QUOTIENT`：以冻结 symbolic equivalence 为身份，证明
   每个 admitted AST 都映射到且只映射到一个 quotient class；
2. `C2_FROZEN_UNIVERSE_EXTENSIONAL_QUOTIENT`：以冻结 public universe 上的
   exact output behavior 为身份，完整计算所有可达 behavior；
3. `C3_HYBRID`：symbolic normalization 负责压缩，frozen-universe behavior
   负责最终 extensional identity 和 replay。

同时必须逐项声明哪些旧主张被放弃、哪些被替换：

- 放弃或保留“归档了所有 syntactically canonical AST”的主张；
- `canonical_program_count` 是否改为 quotient-class / reachable-behavior
  count；
- AST archive 是否只作为 representative/provenance，而不再是 closure
  identity；
- `COMPLETE` 是否改为“所有 bounded admitted AST 的 quotient image 已被完整
  覆盖”；
- outside certificate 是否改为只对明确命名的 frozen quotient、universe、
  equivalence 和 DSL version 成立；
- MDL 是否使用每个 equivalence class 的最短 representative，而不把任意代表
  的长度当成 class 的固有长度。

C 只有在以下条件同时成立时才仍可称为 exact：

1. equivalence relation 在接触 target truth labels 前冻结；
2. 有完备性论证，证明没有 admitted AST 的 behavior 被漏掉；
3. quotient 不合并 frozen universe 上可观察不同的行为；
4. Python/Rust 独立实现与 host replay 对完整 reachable set 一致；
5. certificate 名称和 claim text 明确限定 quotient 语义，不冒充旧的 exact
   syntactic archive claim。

若选择基于有限 universe 的 C2/C3，`outside` 只能表示该冻结 universe 上不存在
匹配的 reachable behavior；除非另有证明，不得外推成对所有可能输入的全域
language impossibility。

请回答选择 C1、C2 还是 C3，并给出新的 closure identity 与 certificate claim
的一段精确定义；不要设计 wire。

---

## 6. 路径 D：停止 exact closure，并界定可发表负结果

如果选择 D，Phase-3 不再以 exact closure / outside certificate 为近期 exit。
请明确批准可发表的最强结论与永久禁止的过度主张。

建议可发表负结果限定为：

> 在冻结的 typed DSL、canonicalization、50,000 canonical budget、raw budget
> 和六步 target-neutral shrink schedule 下，两个独立实现与 host replay 均
> 复现最终 DSL 的容量超限；当前 depth-2 boundary witness 在最后两步结构缩减
> （node 7→6、depth 4→3）中保持不变，因而这两步没有消除已经在低深度出现的
> syntactic multiplicity boundary。

不得据此声称 parity relation outside、旧 DSL 不可表达、symbolic quotient 也会
失败，或黑格尔机的 hypothesis invention 原理被否证。

若选择 D，请回答：

1. 上述负结果是否足以作为独立工程/方法学结果；
2. 发表前还需哪些 target-free ablation、复杂度曲线或失败复现实验；
3. Phase-3 是否转为 bounded approximate discovery / recognizer research，且与
   formal outside certificate 明确分轨；
4. 哪些 M3 gates 永久关闭，哪些可作为未来不同 closure semantics 的历史输入。

---

## 7. 四条路径都必须回答的交叉决策

### 7.1 是否保留当前 target 与 control

请明确回答是否继续冻结：

```yaml
outside_target: Generic Odd-Cardinality Reduction
outside_target_role: parity-like relation
null_control: observed auxiliary channel omitted from scope
null_control_claim: false-invention control only
```

如果要换 target 或 control，必须说明为何这不是看到容量负结果后的 target
shopping，并另行冻结评价设计。

### 7.2 holdout / split seed 是否重置

建议默认：**不重抽 seed，不改变既有 split assignment**。预算、语言或 closure
representation 改变本身不构成重抽 hidden split 的理由。请在以下三项中选择：

1. `KEEP_EXISTING_SEED_AND_ASSIGNMENT_WITH_NEW_BINDING`；
2. `KEEP_SEED_RECOMPUTE_ONLY_IF_MEMBERSHIP_DOMAIN_CHANGES`；
3. `NEW_SEED_REQUIRED`，但必须给出不可由现有 seed/assignment 满足的规范理由，
   并禁止利用新 seed 改善结果。

### 7.3 formal roots 与 M3 gates 如何重置

请明确哪些层级失效：

- 所有依赖 DSL、closure representation、budget、equivalence 或 certificate
  semantics 的 formal roots/gates 必须重新生成和双重资格验证；
- 当前 `M3=NOT_RUN` 与 `formal_roots=null` 在新资格完成前保持；
- 与新路径无关且可证明输入字节未变的 custody、trust、seed-history evidence
  是否允许通过新 binding 继承；
- 24/24 readiness 是否必须重新计算，而不是在旧计数上直接加一项。

请给出 gate-reset 的原则和层级，不需要给 root DAG 或 schema 字段。

### 7.4 最小 preregistration 与 dual qualification evidence

无论选择何种继续路径，启动新 sealed execution 前至少应冻结：

1. adaptation 的公开名称、原因、版本与 claim boundary；
2. target-neutral 输入边界，以及禁止访问的 target/split/secrets；
3. exact termination condition、capacity limits 和全部 terminal routes；
4. 若选 C，equivalence、representative、completeness obligation 与 certificate
   semantics；
5. parity target、sink control、holdout/split continuity policy；
6. 哪些旧 roots/gates 作废、哪些只作为历史 provenance；
7. source-only commit 与 observed-evidence commit 分离；
8. Python/Rust 独立生成、host replay、golden/collision/adversarial vectors、
   preservation controls 和 fail-closed negative controls；
9. 在 full run 前完成缩减预算 qualification 与容量可行性预测，但不得把 subset
   replay 当成 `COMPLETE`。

请判断这九项是否足够；若不足，只补充规范证据类别，不补 wire 细节。

---

## 8. Codex 工程判断，供审阅而非既成规范

Codex 当前建议：

```yaml
recommended_primary_path: C
recommended_variant: C3_HYBRID
conditional_secondary_path: B
sequence: REPRESENTATION_FIX_FIRST_THEN_TARGET_NEUTRAL_SHRINK_ONLY_IF_NEEDED
simple_budget_increase_as_primary_path: NOT_RECOMMENDED
retain_parity_target_and_sink_control: true
redraw_split_seed: false
```

理由：当前 depth-2 witness 与 prefix 在最后两步结构缩减中完整保留，说明主要
问题更像 syntactic multiplicity 的表示瓶颈，而不是仅剩某个高深度 operator。
直接加预算可能只是把同构或 extensional-equivalent AST 继续堆入 archive；继续
盲目 shrink 又可能牺牲语言意义，却仍未解决重复表示。

因此优先修复 closure representation：在不读取 target labels 的前提下计算冻结
universe 上的 exact reachable behavior / symbolic quotient，并证明它覆盖所有
bounded admitted AST。若 quotient 后仍超限，再基于公开的 quotient 容量结构提出
单独的 target-neutral B amendment。A 只适合作为已经证明可终止的小范围验证
预算，不建议作为主路线；若 C/B 的 exactness obligation 无法满足，则选择 D，
诚实发表容量负结果，而不是降低 certificate 口径。

这只是更贴近当前实现证据的工程判断。网页端 GPT / 用户应审阅的是方向、科学
含义和声明边界；最终具体算法、版本字符串、schema 与测试门由 Codex 在工程
验证中定案。

---

## 9. 请按此最小格式返回决策

```yaml
selected_primary_path: A | B | C | D | C_THEN_B_IF_NEEDED
rationale: ""

path_A:
  disposition: PRIMARY | AUXILIARY_ONLY | REJECT
  new_exact_canonical_program_budget: null
  new_raw_operator_application_budget: null
  anti_post_hoc_constraints_approved: false

path_B:
  disposition: PRIMARY | CONDITIONAL | REJECT
  target_neutrality_rule: ""
  allowed_high_level_shrink_scope: ""
  post_capacity_adaptation_label_required: false

path_C:
  disposition: PRIMARY | CONDITIONAL | REJECT
  quotient_variant: C1 | C2 | C3 | null
  closure_identity_definition: ""
  old_claims_abandoned: []
  replacement_certificate_claim: ""

path_D:
  disposition: PRIMARY | FALLBACK | REJECT
  strongest_publishable_negative_claim: ""
  permanently_forbidden_claims: []

target_and_control:
  retain_generic_odd_cardinality_target: false
  retain_hidden_sink_false_invention_control: false

split_continuity:
  decision: KEEP_EXISTING | RECOMPUTE_MEMBERSHIP_ONLY | NEW_SEED
  reason: ""

formal_reset:
  m3_remains_not_run_until_requalified: false
  root_and_gate_reset_principle: ""
  unaffected_genesis_evidence_reuse_rule: ""

minimum_evidence:
  nine_item_baseline_approved: false
  additional_normative_evidence_categories: []

web_gpt_scope_acknowledgement:
  direction_and_claims_only: false
  wire_cbor_tags_and_engineering_details_deferred_to_codex: false
```

任何缺少主路径、A 的具体预算（若 A 被批准）、C 的 quotient 语义（若 C 被
批准）、target/split continuity 或 gate-reset 原则的回答，都只算审阅意见，不能
直接授权下一次 execution。
