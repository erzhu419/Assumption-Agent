# 交给网页端 GPT 的方案问题：Phase 2 exit 与 Phase 3 入口

这些问题不会阻塞当前施工。当前已完成的是 verifier-ready synthetic witness 上的
Phase-2 controlled API-blinded selector mechanics qualification，并保持
shadow-only / fail-closed；family-neutral typed measurements 仍是待讨论的正式
exit 输入边界。

## 1. Phase-2 正式 exit 的输入边界

应选择哪一个正式 claim？

1. **Typed exit**：输入已经是 family-neutral 的 typed measurements，但不提供
   law family、role binding 或 scale 真值；系统自行选择并验证。
2. **Raw-evidence exit**：从自然语言、表格或轨迹原始证据开始，同时评价抽取、
   binding、scale selection 与 law verification。

当前默认：先单独资格化 Typed exit；raw extractor 作为后续独立模块，避免把
抽取错误与结构推理错误混成一个分数。需要 GPT 给出两种 claim 的准确命名、
边界，以及 raw-evidence 何时成为进入 Phase 3 的硬前置条件。

## 2. Phase-2 正式 holdout 的统计协议

路线图列出了 family classification、binding、hard-negative rejection、binding /
scale counterfactual rejection、sign-flip、scale selection、deterministic
abstention 和 preservation，尚未冻结样本量、阈值和置信区间。当前人工
semantic-only decoy 的 accuracy/gap 只作诊断、不参与 exit；metadata replacement
invariance 仍是 anti-leak gate。正式协议需另行决定是否加入真实 embedding /
learned baseline。

请 GPT 设计一份可预注册协议，至少回答：

- 每个 law family、control type、scale 至少多少独立 case；
- family / binding / scale 指标的最低点估计和置信下界；
- deterministic abstention 何时升级为可校准概率，以及届时用 selective risk、
  coverage、Brier/ECE 中哪些指标；
- 是否必须加入真实 semantic/embedding baseline；若是，最低胜幅如何定义；
- 哪些失败是一票否决，哪些可以用总体置信区间判断；
- development set、validation 和一次性 sealed holdout 如何分离。

当前 source-visible synthetic generator 的 schedule 可由读源码者重建；请明确正式
协议中哪些组件必须由独立 custodian 保密、何时冻结代码、如何防止在 holdout 打开
前通过公开 ID 或生成顺序形成 lookup shortcut。

## 3. Phase-3 “旧语言不能表达”如何冻结

若不先冻结旧语言的 operator、arity、composition depth、参数自由度、等价容差
和 MDL 编码，任何“新关系”都可能被事后说成旧关系组合，反之亦然。

请 GPT 给出第一版 bounded DSL 的精确定义，包括：

- primitive types 与 operators；
- 最大 arity、程序深度和搜索预算；
- observational / extensional / algebraic non-equivalence tests；
- 参数调整、scope refinement、低阶 composition 与 invention 的分界；
- program 与数据的描述长度编码；
- 首个 hidden family 应选 parity-like、hidden sink，还是两者并行。

当前默认：以 parity-like relation 作为第一个真正的 language-outside 任务；
hidden sink 更容易被合理归入 conservation refinement。

## 4. ACTIVE promotion 的可信根是否进入近期范围

当前本地 lifecycle 是非权威 shadow ledger，active append 在没有独立 custodian、
writer signature 与外部可信根时硬性关闭。

请 GPT 比较两个方向：

1. Phase 2–3 全部保持 candidate/shadow-only，把研究重点放在识别与发明；
2. 现在引入签名 manifest、custodian key、撤销/轮换和可验证 replay，使 scoped
   candidate 可以真实晋升 ACTIVE。

当前默认：选择 1；签名治理单列工程轨道，不让它阻塞认知能力 benchmark。

## 5. 文献和 repo 快照的长期二进制归档策略

当前首个本地提交包含约 569 MB 新对象，单文件都低于 GitHub 100 MB 硬限制，
但普通 Git 会永久携带这些历史对象。请 GPT 比较：

1. 继续普通 Git，换取单仓库、固定 commit 的完整离线复现；
2. 改用 Git LFS；
3. 将大文件放 GitHub Release / 对象存储，仓库只保留 checksum manifest；
4. 论文和源码分别采用不同策略。

首个提交已经按用户授权通过普通 Git 成功 push；GitHub 仅对 4 个 70–83 MB
文件给出 LFS 建议，没有拒绝。后续仍需决定长期策略，但本轮不重写已推送历史。

## 6. Phase-2 / Phase-3 里程碑命名与施工顺序

下面这段可以直接贴给网页端 GPT：

> 我们当前的 Hegel Machine v0.2 有一套 43-case synthetic controlled benchmark：
> 24 个 answerable、19 个应 abstain，每例从统一的 verifier-ready synthetic witness bundle 经
> frozen adapter 重放 24 个 family × binding × scale projections；六个 family 都跨
> 两种 scale，严格要求 binding/scale competitor 完成计算，并用 verifier tolerance
> 归一化的 boundary margin 做选择，另有 12 个 preservation pair。它不测 raw
> text/table/trajectory extraction，未使用一次性 sealed holdout，也不支持
> open-world discovery；fixture 值仍由 evaluator case spec 反向构造，
> `controlled_api_selector_qualified` 只是内部机械闭环状态。请为我们
> 设计不夸大 claim 的 Phase-2/Phase-3 里程碑命名和顺序，明确回答：
>
> 1. 当前成果应叫 “Phase-2 typed selector qualification”、Phase-2A，还是别的名称？
> 2. raw extractor qualification 是否必须在开始 bounded Phase-3 hidden-law synthesis
>    之前完成，还是可以作为并行轨道？
> 3. 一次性 sealed holdout 是否必须在进入 Phase 3 前打开，还是应先完成 Phase-3
>    DSL/benchmark 后再统一封存，避免过早消耗 holdout？
> 4. 请分别给出 “允许开始 Phase-3 施工” 与 “可以正式宣称 Phase-2 exit” 的最小
>    验收条件、证据类型和禁止表述。
> 5. 请给出一个 3–5 个里程碑的推荐路线图，并说明每个里程碑的输入边界、输出
>    artifact、主要失败模式和 go/no-go gate。

## 7. 正式 Phase-2 的独立证据合同、scale 语义与 preservation 强度

当前 benchmark 的 witness bundle 由 source-visible evaluator case spec 条件生成，
recognizer 虽然不接收 answer object，但还没有被当作不可信组件隔离；公开 case ID
和生成顺序也不是 secrecy boundary。请 GPT 对下面三个相互关联的问题给出可实现、
可预注册的方案：

1. **独立 evidence generator → adapter 合同**：正式 typed holdout 是否必须禁止
   evaluator-conditioned PASS/FAIL fixture、candidate-private payload 和任何可反推答案
   的顺序信号？请定义 family-neutral typed evidence schema、生成器与 evaluator 的信任
   边界、adapter 可见字段、untrusted recognizer 的进程/文件/网络隔离，以及 sealed
   answer manifest 的 custodian 和一次性打开流程。
2. **真正的 scale selection**：当前系统是在两个显式 scale-tagged projections 中选择，
   还是已经足以称为“从上下文推断 scale”？若正式 claim 要求后者，请定义 scale 的
   可观测输入、禁止泄漏字段、held-out scale transforms、跨尺度反事实和验收指标；
   同时说明前一种能力应如何准确命名。
3. **更强的 preservation**：当前 12 对样本只验证 entity alpha-renaming，scale map
   是预注册 identity。正式 exit 至少应加入哪些变换（例如观测重排、无关实体扩充、
   单位/坐标变换、等价聚合或非平凡 scale map）？请给出每类变换的适用 law family、
   最小样本数、合法映射的预冻结方式和失败判据。

另请判断：下一版本应继续增强 trusted API selector，还是直接建立“独立 raw/shared
evidence generator + untrusted recognizer + sealed evaluator”的正式 Phase-2 轨道。
如果选择后者，请明确哪些当前 synthetic fixture 可以保留作开发集，哪些不得进入
正式 holdout。

## 8. 边界压力与共享证据覆盖

当前 24 个正例的 tolerance-normalized margin 都远高于冻结阈值，shared-measurement
gate 只要求每个 case 至少有一个 measurement 被多个 candidate 复用。请 GPT 设计：

- near-boundary 与 heterogeneous-tolerance case 的分层构造和预注册比例；
- 按 family × scale × witness footprint 冻结的最低共享覆盖，而不是允许一个全局
  常量 witness 使 gate 通过；
- 如何区分 verifier 数值稳定性、selector margin 稳定性和真正的结构歧义；
- 哪些边界样本应进入 development stress suite，哪些必须保留为 sealed holdout。
