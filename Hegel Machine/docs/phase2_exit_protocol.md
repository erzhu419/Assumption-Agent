# Phase-2A Controlled Typed-Selector Mechanics Qualification

当前一般能力的冻结名称是 **Explicit-Projection Typed Structural Selection**；
其中 scale 子能力只能称为 **Scale-Indexed Candidate Projection Selection**。
本文档描述 development fixture 协议，不是正式 Phase-2 exit 协议。

## 目的

v0.1 只证明了：在 law family 和 typed binding 已给定时，六个 executable
verifier 能计算 residual 并拒绝结构反例。它不能支持 law-family selection、
binding inference、scale selection 或 abstention policy 的 claim。

v0.2 补的是这四项之间的 synthetic controlled selection/replay 工程闭环：

```text
uniform verifier-ready synthetic witness bundle with anonymous labels
  → frozen adapter replay
  → complete family × binding × scale projection grid
  → six executable residual families
  → unique-pass + tolerance-normalized boundary-margin selection
  → deterministic policy-bound abstention
  → bounded cross-episode preservation witness
```

## 盲化边界

每个 case 先构造一份使用匿名 channel、entity 和 observation ID 的共享 typed
measurement bundle。冻结 adapter 从同一 bundle 重放所有候选 projection；生成物
再进入 `recognize_structural_law()`。Recognizer 的输入只有 theory、unbound
episode 和冻结 policy，没有 gold family、gold role map、gold scale 参数或私有
answer table。每个 episode 都含有相同的 outer schema，并为六个 law family 各提供：

- canonical binding × 两个 scale；
- wrong binding × 两个 scale。

因此每个 case 固定有 24 个同型 `StructuralProjection`。Projection 是冻结 adapter
对共享 evidence 的确定性重放结果，不是 gold label；benchmark 还逐例核验 replay
content ID。Measurement 按 observable 的 witness-role footprint 绑定；没有被
counterfactual 改动的角色值会由 canonical/wrong binding 共同消费，而不是给每个
candidate 私藏一份完整 payload。Bundle 拒绝重复 witness key，adapter registry 又
绑定 theory version、verifier source registry 和 evaluator epoch。Answer key 是
evaluator 中的独立对象，recognizer API 无法读取。

这里的 blinded 是 **API/dataflow 边界**，不是保密承诺。生成器源码与 evaluator
table 都在仓库中，source-aware oracle 可以重建 synthetic schedule；当前 corpus
不是 anti-overfitting benchmark。工程 gate 检查的是实际 recognizer 不读 answer
table，并在 episode、observation、entity、projection ID 全部一致重命名后保持同一
结构决策。只有未来独立 custodian 管理的一次性 sealed holdout 才能提供 secrecy
边界。

此外，PASS/violation/missing fixture 值由 evaluator case spec 反向构造，再通过
role-scoped measurement 和 adapter 形成候选。它们不是从独立 raw evidence 抽取的
观测。因此 family/binding/scale accuracy 是 selector mechanics 的受控功能测试，
不是统计意义上的独立识别能力估计，也不资格化任意不受信 recognizer。

严格 qualification policy 不只要求六族都有已完成的 residual。对唯一 PASS 候选，
同 law、同 scale 的另一 binding，以及同 law、同 binding 的另一 scale 也必须完成
计算；缺少任一 competitor 都会 abstain。不同 verifier 的 raw residual 不直接比较，
而是先除以各自 acceptance tolerance，得到相对边界距离，再应用冻结 margin。

这仍然从 typed projection 开始，**没有资格化 raw text / table / trajectory
extractor**。系统测的是对完整候选集合的结构选择，不是从原始证据发现 family。

## Corpus

冻结 corpus 有 43 个 deterministic synthetic cases：

- 六类 law 在每一种 scale 上各一个 low-semantic positive；
- 六类 law 在每一种 scale 上各一个 entity-renamed positive；
- 六类 high-semantic hard negative；
- 六类 sign/order/constraint flip；
- 六类 missing-evidence abstention；
- 一个 cross-family ambiguity abstention。

前两项合计 24 个 answerable case；后三项和 ambiguity 合计 19 个应 abstain case。
两种 task scale 是 `phase2_dynamic_law` 与 `phase2_causal_mechanism`，每个 family
都在两种 scale 上出现，避免 family–scale 混杂。每个可回答 case 只有一个
family–binding–scale projection 同时通过 verifier 和冻结 normalized margin；错误
binding 和错误 scale 都保留为已完成的显式 counterfactual。24 个正例按
family × scale 组成 12 个原例/实体改名 preservation pair。

## 冻结指标

报告独立列出：

- family classification accuracy；
- binding accuracy；
- scale-selection accuracy；
- hard-negative rejection；
- role-binding counterfactual rejection；
- scale counterfactual rejection；
- sign-flip sensitivity；
- deterministic abstention accuracy；
- cross-candidate shared-measurement reuse；
- frozen-adapter replay integrity；
- identifier-value renaming invariance；
- cross-episode preservation；
- structural exact-decision accuracy；
- semantic-only control accuracy（仅诊断）。

人工构造的 adversarial semantic-only control 不是真实 embedding baseline，也不是
外部 efficacy 对照。报告中的 decoy accuracy 与 structural gap 仅为诊断字段，不
进入 exit gate；但是“替换全部 semantic metadata 后选择不变”是防止 acceptance
路径偷读 metadata 的硬 gate。

## Exit 判定

当前里程碑的唯一名称是 **Phase-2A Controlled Typed-Selector Mechanics
Qualification**。Artifact 为兼容已有机器消费者，仍保留内部状态标签
`controlled_api_selector_qualified`，但它不是正式 Phase-2 exit：

- synthetic；
- validation split，未 sealed；
- 没有独立 custodian signature；
- 样本量不足以支持置信区间；
- 只测 role-scoped synthetic witness bundle 的 frozen-adapter replay 与候选选择，
  不测 family-neutral raw evidence 或 raw extractor；
- 没有真实 embedding / learned structural model baseline。

因此它支持“受控 typed-selector mechanics 工程资格化”，不支持“Phase 2 已以外部证据完成”或
“系统已经发现未知关系”。它也不是 sealed-holdout 结果、raw extraction
qualification 或 open-world discovery。正式 exit 的样本量、置信阈值和
raw-evidence 边界需在 holdout 打开前预注册。

当前从两个显式 scale-tagged projections 中选择的能力不得写成 context-inferred
scale、autonomous scale discovery 或 scale abstraction learning。

## 重放

```bash
cd "Hegel Machine"
PYTHONPATH=src python3 -m hegel_machine phase2-exit \
  --output /tmp/phase2_exit.json
diff -u artifacts/phase2_exit_benchmark_v2.json /tmp/phase2_exit.json
```

完整 policy、theory、shared evidence、adapter、episode、projection、decision 和
preservation witness 都是内容寻址对象。Projection 输入顺序会在 episode 构造时
规范化；adapter replay 必须复现 projection content ID，semantic metadata 替换
不会改变结构选择。Preservation 的 entity map 来自在 recognition 前冻结的 evaluator
table，scale map 固定为 identity；两者都不从 selected output 事后拟合。

## Phase 3 入口

Phase-2B/Phase-3 overall contract 已由 implementation audit 修订为
`hegel-freeze-p2b-p3-v1.0.2`：`411876909552964556` 保持 master/bootstrap seed，
sklearn `random_state` 使用冻结的 domain-separated SHA-256 → uint32 值
`2611585425`；v1.0.2 还冻结 strict AST/CBOR 与 certificate bridge 规范。

下一工程切片的唯一里程碑名称是
**Phase-3A Bounded Frozen-Closure Adequacy**。这里的状态是
`surface_parameter_freeze_complete = true`、
`strict_acceptance_contract_complete = true`、
`normative_parameter_freeze_complete = true`、
`strict_acceptance_implementation_verified = true`。

M1 的 Python/Rust shared vectors 各 48/48 PASS。M2 两端均接受 64,680 个 source
candidates，得到 64,680 个 unique strict canonical AST；共同 diagnostic commitment 为
`sha256:c1a02a66a8d6d8f75204cb3daf03ab0b01c2b3b8e486d0ab3d481ee3be43c930`，ordinal
50,001 的 AST hash 为
`sha256:7c7f786c2cc57d31506b3c61d162d175c7f69a2878a089c72c9d053694cba948`。因此
`hegel-old-dsl-v1.0.0` 在 50,000 syntactic budget 下的 bounded status 是
`DSL_TOO_LARGE`。证据见
[dual strict gate](../artifacts/phase3_dual_strict_gate_v1.json) 和
[dual strict capacity replay](../artifacts/phase3_dual_strict_capacity_replay_v1.json)。

这不是 `COMPLETE`，没有 extensional target verdict、hidden-sink formal verdict、formal
roots、outside/MDL certificate 或 ACTIVE authorization。下一施工顺序已由 frozen transition
决定：

1. 发布新的 old-DSL version；
2. 执行 shrink step 1，删除 `mean_v1`、`min_v1`、`max_v1`；
3. 重建 target/validation commitments，让新版本 closure 从 `NOT_RUN` 开始；
4. 只有新版本未来得到 `COMPLETE`，才允许做 target extensional membership 与
   hidden-sink formal test；
5. 始终保持 candidate/shadow-only，不提前打开 LANGUAGE compiler 或 ACTIVE graph。

当前 untrusted receipt wire 已绑定完整 `dsl_spec_id`、`operator_semantics_id`、
equivalence/enumerator/search-budget，并按 `target_role` 分别绑定 480-row outside target 与
85-row null control 各自独立的 diagnostic universe/truth IDs。v1.0.2 已冻结 formal bridge
schema，但 M2 set commitment 不是 canonical-CBOR/RFC6962 root，正式 roots 仍为 `null`。
caller-supplied receipt 仍不能升级 claim；当前 bounded transition 来自双实现 artifact，也只
授权 shrink，不授权 adequacy 或 certificate。

v1.0.2 规范见
[`Hegel_Machine_Strict_Canonical_AST_CBOR_Certificate_Bridge_Freeze_v1.0.2.md`](Hegel_Machine_Strict_Canonical_AST_CBOR_Certificate_Bridge_Freeze_v1.0.2.md)，
当前 readiness 决议见
[`Hegel_Machine_Phase3_Freeze_Readiness_Resolution.md`](Hegel_Machine_Phase3_Freeze_Readiness_Resolution.md)。
旧 readiness/questions 文件保留为带 superseded banner 的审计快照。
