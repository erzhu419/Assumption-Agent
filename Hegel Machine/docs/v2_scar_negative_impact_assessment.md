# v2 SCAR 负结果对 Hegel Machine 的影响评估

日期：2026-08-01
状态：`EVIDENCE_BOUNDARY_ASSESSMENT`
源结果：Assumption Agent commit
`4861b2d88ef7e85fb62f32e3d2e1f5c78afe9529`

对应的机器可读绑定记录是
[`v2_scar_negative_evidence_binding_v1.json`](../artifacts/v2_scar_negative_evidence_binding_v1.json)。
v2 本体中的完整代码级根因、可恢复信号、component oracle ladder 与 fresh recovery study 方案见
[`GSCL/SCAR v2 负结果：根因审计与恢复路线`](../../reconstruction_v2/docs/gscl_scar_cssm_intrinsic_negative_postmortem_and_recovery_plan_20260801.md)。
本文件是后续证据评估，不修改
[`legacy/source_git_head.txt`](../legacy/source_git_head.txt) 中冻结于
`ae8eb7f6733be38eab2a1e03d3ffa1f8d175e009` 的历史快照，也不把 v2 的结果继承为
v3 的通过或失败证据。

## 1. 结论

`4861b2d8` **不构成 Phase-3A M2.5 或 M3 formal closure 的 blocker**。M2.5
已继续实现 v1.1.2 的 strict CBOR、typed rows、candidate roots 和 split-crypto 纯函数；
但独立于 v2 结果，v1.1.2 authoritative DAG 审计发现 12 组 exact-wire 勘误与
外部 actor blockers，故当前仍必须保持 `14/24` 和 `NOT_RUN`。勘误补齐并取得
独立 actor evidence 后，`24/24` 也只表示 ready-but-`NOT_RUN`；M3 仍需单独显式启动。

但该结果对后续 claim 有实质约束：在证明旧语言识别正常之前，不得把抽取、绑定、识别或
hard integration 的失败诊断成 `ONTOLOGY_DEFECT`，也不得把它包装成“现有 22 条先验不足，
因此系统发明了第 23 条”。这主要约束 Phase-3B 的 synthesis/integration 与 Phase-3C 的
raw-evidence claim，而不是 M2.5/M3 的形式身份和完整枚举。

更准确的判断是：

> SCAR 结果否定了一套冻结的“fixed extractor/binder + hard structural eligibility +
> length-2 composition” operationalization；它没有逐条检验或否定 22 条 UAO，也没有
> 否定 13 条旧假设。13 条本来就是 22 条 ontology 的 legacy aliases，不是另一套 13 个
> 独立先验。

## 2. `4861b2d8` 实际证明了什么

正式 SCAR 执行是 protocol-valid，而不是 implementation/infrastructure-invalid：唯一有效
invocation systemd success、`NRestarts=0`，两 action shards 先于 label/secret barrier
完成，offline scorer 只调用一次，retry/replay/resample 与 online/API evaluator 均为零。

其唯一 confirmatory primary 是：

```text
full_with_length2_composition - semantic_only
```

结果为明确反向效应：

| arm | answer coverage | item-macro pair-F1 |
|---|---:|---:|
| semantic-only | 1.000000 | 0.677030 |
| flat-structural | 0.998619 | 0.487495 |
| full no-composition | 0.011050 | 0.006183 |
| full + length-2 composition | 0.006906 | 0.004144 |
| full + composition + target-color shuffle | 0.009669 | 0.003453 |

主效应是 `-0.6728865210909416`，95% bootstrap CI 为
`[-0.7095084626852585, -0.6358874857493647]`，正式 disposition 为 `FAIL`。
full composition 只在 724 个 variant 中选择 5 个；这 5 个的 typed incidence、structural
origin 和 length-2 path verification 都成立，因此失败不能解释成“结构代码没有执行”。

同时，flat-structural 覆盖 723/724，却仍比 semantic-only 低约 0.190 F1；所以结果也不能
简化为“只要放宽一个 hard gate 就会恢复”。证据支持的机制诊断是：当前抽取出的结构信号
既稀疏又有噪声，hard selector 会删除有用的 semantic mapping，而仅去掉 hard eligibility
仍不足以让结构信号产生增量价值。

## 3. 为什么这不是“22/13 条先验整体失败”

### 3.1 13 条不是额外的 13 个独立对象

v2 中的 13 条旧假设是 UAO v1 的兼容别名，例如单调性映射到 T17、守恒映射到
T15、对称/等变映射到 T14。不能把“22 + 13”当成 35 个独立先验，也不能从一个
SCAR operationalization 的失败推断每个 alias 或 template 都失败。

### 3.2 SCAR formal arm 没有逐条调用 UAO ontology

SCAR action path 使用 narrative extractor、slot graph binder 和一个固定八算子的
slot-set mapping closure。其 verifier 明确不接收 laws，只检查 injectivity、typed
incidence 与 ordered length-2 composition。它没有对 T01–T22 逐条形成 claim、执行
violation functional 或比较每个 template 的 retained utility。

所以 `4861b2d8` 的合适结论是：

```text
PROTOCOL_VALID_PRIMARY_FAIL_GENERALIZED_COUNTERPOINT_OPERATIONALIZATION_NEGATIVE
```

而不是：

```text
ALL_22_OR_13_META_PRIORS_FALSIFIED
```

### 3.3 Hegel Machine 的活动路径没有继承 v2 执行依赖

Hegel Machine 把 v2 放在 `legacy/v2_gscl/` 中作迁移和审计参照；活动
`src/hegel_machine/` 与 Rust crates 不 import 该目录。当前 shrink-1 old DSL 的 active
aggregate IDs 是 `sum_v1`、`count_nonzero_v1`、`signed_balance_v1`，而不是一个声称已完整
编译 T01–T22 的可执行闭包。

因此未来的 outside verdict 只能说明 target 位于指定的 bounded aggregate DSL closure
之外，不能自动改写成“target 位于全部 22 条先验及其所有合理组合之外”。若要支持后者，
必须先另行冻结 ontology-template → executable DSL coverage manifest，并对未编译 template
显式记为 `UNREPRESENTED`。

## 4. WikiSQL UAO P4 same-v5 的正确对照边界

WikiSQL same-v5 的数值确实显著好于 SCAR hard structural arm：

- Agent−patched official HippoRAG aggregate `+34`，EQ/GT/LT=`+14/+6/+14`，该比较通过；
- Agent−RAW aggregate `+30`，GT/LT=`+10/+20`，但 EQ=`0`；
- 因预注册要求对两个 baseline 的三个 family 都严格为正，overall 仍是
  `FAIL_REALITY_PRIMARY`。

该执行还是公开披露的 `user_authorized_post_terminal_same_v5_missing_arm_ext4_continuation`
协议例外，不能当作原始无中断 confirmatory primary。其设计只开放四个手写 recipe
T02/T05/T08/T18，并由 TRAIN 选择两个；设计文件还明确说明该 study 测的是 system-level
double-baseline existence claim，不是 UAO claim selection 的因果效应。

因此 SCAR 与 WikiSQL 不能直接用效果大小排序“哪个先验更真”：数据、任务、endpoint、
candidate construction、fallback 和统计合同均不同。它们可以共同支持一个架构判断：

> WikiSQL 的 frozen no-op / minimum-commitment policy 会在预期增益不足时 byte-exact 保留
> RAW，而 SCAR 的 hard structural selector 发生 coverage collapse。后续 invented law 应作为
> 对 semantic/old-law backbone 的保守增量，经 decision-relevance gate 后才改变 action，
> 不应默认替换 backbone。

## 5. 对各阶段的影响

| 阶段 | 是否 blocker | 影响 |
|---|---|---|
| Phase-3A M2.5 formal commitment/seed/bridge | 否 | gates 15–24 都是 seed、ledger、CBOR、root、state/receipt wire 身份与外部 actor 问题；v2 efficacy 不参与这些 gate |
| Phase-3A M3 complete closure | 否 | M3 回答 frozen bounded DSL 的 extensional closure；继续要求 odd target 与 sink control 分角色重放 |
| Phase-3A claim interpretation | 有边界影响 | 只能称 `OUTSIDE_FROZEN_CLOSURE(...)`，不能称“全部 22 条先验不足”或无边界 `OUTSIDE_LANGUAGE` |
| Phase-3B bounded synthesis | 条件影响 | v2 不增加新的 formal blocker；synthesis execution 仍须等待既有的 M3 COMPLETE/outside 前置条件，且在发明 claim 前必须排除 old-law recognizer/extractor/integration failure |
| Phase-3C raw-evidence end-to-end | 是，针对效果 claim | v2 的 extractor/binder/hard selector 不得作为已资格化的 trusted front end；必须在 fresh evidence 上重新验证 |
| ACTIVE governance | 无新增影响 | 本来就是 shadow-only/NO-GO；SCAR 结果不改变该状态 |

`4861b2d8` 也不构成 split compromise 或 hidden access，不要求重抽 M3 split seed。当前历史状态仍是
`SPLIT_SPEC_FROZEN_BUT_SEED_NEVER_INSTANTIATED`，应按新的 normative freeze 做第一次实例化，
不能利用 v2 结果增加 split 自由度。

## 6. 必须冻结的控制实验

下面控制不应追溯加入当前 M3 的 24 个 formal entry gates；它们属于 Phase-3B/3C 的后续
能力与效果 gate。M2.5/M3 的已冻结合同保持不变。

### 6.1 Old-language alignment audit

在声称“现有 22 条不足”之前发布 machine-readable coverage matrix：

```text
T01..T22
→ executable schema / verifier / old-DSL program family
→ REPRESENTED | PARTIALLY_REPRESENTED | UNREPRESENTED
→ exact equivalence domain and bounds
```

13 个 legacy aliases 只引用该 matrix，不另计为独立 language elements。未表示的 template
不能被 complete closure certificate 默认为已排除。

### 6.2 三层故障隔离

用同一批 in-language cases 依次运行：

1. `oracle typed evidence → frozen old-law verifier`：隔离 law semantics/verifier；
2. `independent typed generator → untrusted recognizer → verifier`：隔离 recognizer；
3. `raw extractor → untrusted recognizer → verifier`：隔离 raw structuralization。

只有第 1、2 层通过而 frozen old closure 仍无法解释 residual，才允许把失败升级为
`ONTOLOGY_DEFECT`。第 3 层失败只阻塞 raw-evidence claim，不能反向否定 typed Phase-3A。

### 6.3 多族 in-language false-invention controls

hidden-sink 是必要的单个 in-language null control，但不足以证明所有已知关系都能被识别。
Phase-3B 前至少应为六个 executable law family 各冻结 in-language positive/null cases，并要求：

- old-law exact recovery；
- false invention count 为 0；
- role/sign/scale counterfactual 正确拒绝；
- entity-renaming 和合法 scale transform preservation；
- abstention 不从 accuracy 分母中消失。

具体配额和阈值应在看 sealed outcomes 之前另行冻结。

### 6.4 Fresh matched downstream arms

在同一个 fresh cohort、相同 candidate pool、预算、模型和 scorer 下预注册：

1. semantic-only backbone；
2. semantic + frozen known-prior soft residual/no-op integration；
3. semantic + invented-law soft residual/no-op integration；
4. hard structural-only selector，作为 SCAR failure-mode control；
5. shuffled-law / role-swap / sign-flip control。

primary 应计入全部 item，至少同时报告 unconditional utility、answer coverage、family-stratified
utility 与 preservation；不能只在 selector 选择的少量 item 上报告 conditional precision/F1。
invented arm 必须满足事前冻结的 coverage floor 和 old-success preservation，不能靠大量 abstain
制造表面高精度。

### 6.5 Invention-specific controls

- odd-cardinality outside target：只有 complete closure match set 为零后才进入 synthesis；
- hidden-sink/in-language targets：必须恢复旧表达，不得发明；
- exact extensional non-equivalence：拒绝旧 program 的改名、语法重写和低阶组合；
- MDL、unseen prediction、reduction map 与 preservation pairs 全部通过后，才称 bounded invention；
- semantic/old-law backbone 的 action 只有在 preregistered incremental-utility gate 通过时才可改变，
  否则 byte-exact no-op。

### 6.6 Freshness

已消费的 SCAR root 禁止重放、改 gate 或按结果定向修候选。WikiSQL same-v5 也只能作为透明披露的
协议例外和设计动机。任何 soft-integration 或 invented-law 正向比较都必须使用新的版本、fresh
cohort 和独立 sealed split。

## 7. 建议新增的 Phase-3B 准入条件

这些条件不改写 M3 24 gates，建议在 Phase-3B preregistration 中另行冻结：

```text
OLD_LANGUAGE_IN_LANGUAGE_COMPETENCE_QUALIFIED
ONTOLOGY_DEFECT_NOT_RECOGNIZER_OR_EXTRACTOR_FAILURE
CONSERVATIVE_INTEGRATION_NO_COVERAGE_COLLAPSE
```

exact quota、coverage floor、统计 endpoint 和是否按六族或 22-template 分层仍是需要在网页端与
GPT 定案的问题。推荐原则是：先证明系统能在 matched conditions 下可靠使用旧语言，再允许它
声明旧语言不足；先证明 invented relation 对 backbone 有增量且保存覆盖，才允许写效果 claim。

## 8. 最终施工决定

1. 保留已通过的 v1.1.2 deterministic candidate foundation，先就 E1–E12 取得新的
   bit-exact errata amendment，再由独立 custodian/auditor 执行；不得按现稿猜测缺失 wire 来凑成 24/24；
2. gates 完成后可执行 M3 complete closure，不因 `4861b2d8` 暂停或重抽 seed；
3. 不把 M3 outside certificate 解释为“22/13 全部失败”；
4. Phase-3B 施工前加入 §6–7 的 old-language competence 与 conservative-integration controls；
5. Phase-3C 在 raw extractor/recognizer fresh qualification 前保持 NO-GO；
6. v2 SCAR hard selector 只保留为后续 failure-mode control，不作为 trusted Hegel front end。
