# Evidence and claim boundaries

## 当前可以说什么

当前实现和测试支持两个递进层级：

> 在一个冻结的有限定律库和受控离线样例中，系统能够在已提供 typed law
> binding 的条件下，不用语义相似度作为接受依据，计算 structural residual，
> 拒绝缺观测、虚构实体、角色交换、符号翻转和尺度不兼容，并把测得的证据送入
> 不可变、受 evaluator epoch 约束的保守候选评估流程。

以及：

> 在统一 outer schema、冻结且完整的 family/role/scale projection 集合中，系统在
> 不接收 answer key、不读取 semantic metadata 的条件下，能够从统一的
> verifier-ready synthetic witness bundle 经 frozen adapter 重放 24 个候选，并
> 选择唯一通过的 family–binding–scale 组合；目标 binding/scale counterfactual 必须完成计算，
> 不同 family 的距离按 verifier tolerance 归一化后才比较。遇到多解、缺证据、
> competitor 未完成或全部违反时按冻结 policy abstain，并生成绑定 law、role map、
> scale、epoch 和 residual drift 的跨 episode witness。

第一层叫 verifier/integration qualification；第二层叫 controlled API-blinded
selector qualification。它们都不叫 raw-evidence law-family discovery，也不是
正式 Phase-2 exit evidence。v0.2 corpus 有 43 个 synthetic case（24 个
answerable、19 个应 abstain），每例 24 个 projections，六族均跨两种 scale；
24 个正例形成 12 个 preservation pair。当前 controlled data 只是代码内合成与
adapter replay。Phase-2 selector 报告没有 sealed manifest，输出的是内部工程标签
`controlled_api_selector_qualified`；另一个 governance vertical slice 才输出
`candidate_framework`。两者都不授权 active graph mutation。即使 governance
manifest 结构检查通过，当前版本也没有外部签名可信根，不会晋升 ACTIVE。

## 当前不能说什么

- 没有新关系发现、开放世界本体演化或现实科学发现的效果证据。
- v1 的 framework-growth 分数含公式化 fixture；它们只提供 schema/threshold
  原型，不是 v3 的 PASS 证据。
- v2 的 GSCL controlled corpus 是合成 qualification，不是 downstream efficacy。
- ARN 已被用于实现后验诊断，不能再叫 untouched。
- 文献或 repo 被归档不等于其结论已在本项目复现。
- benchmark 中反向构造的 semantic-only control 只诊断验收路径没有读该分数；
  其 decoy accuracy/gap 不参与 exit gate，不能作为与真实 embedding 系统的效果
  比较；semantic metadata replacement invariance 则是 anti-leak 硬检查。
- API-blinded selector 的 projection 由冻结 adapter 提供；尚未测从 raw text、table
  或 trajectory 生成完整候选集合。
- 当前结果不是 sealed holdout、正式 Phase-2 exit 或 open-world discovery 证据；
  `controlled_api_selector_qualified` 只是受控机械闭环报告中的状态标签。
- blinded 仅表示 recognizer API 不接收 answer key，并通过一致 ID 重命名不变量；
  source-visible generator 的 case schedule 可被重建，公开 ID 不是保密边界。
- fixture 值由 evaluator case spec 反向构造；family/binding/scale accuracy 是 selector
  mechanics 的功能测试，不是独立 raw evidence 上的能力估计。

## 权力与数据隔离

| 角色 | 可以读 | 禁止 |
|---|---|---|
| Generator | train/source-only residual、ontology | holdout outcome、最终晋升标签 |
| Formalizer | typed candidate、冻结语言 | 自造证据、决定晋升 |
| Falsifier | candidate、反例预算 | 更改 candidate 以便通过 |
| Evaluator | 预注册 protocol、sealed evidence | semantic score、跨 epoch 偷换标尺 |
| Promoter | 结构化 certificate/receipts | 原始自由文本、自己生成测试结果 |

Evidence priority 为 proof > executable test > physical/simulation >
held-out human > independent LLM。多个相似 LLM judge 不视为独立重复。

## 版本与 split 规则

- Evidence 必须绑定 `theory_version`、`evaluator_epoch`、`probe_version`、
  `data_cutoff` 和 split。
- 预注册预测必须在 outcome 打开前有内容 hash 和早于 receipt cutoff 的时间。
- epoch 内 evaluator 冻结；换 epoch 后旧分数只保留为历史，不能直接相加。
- semantic retrieval 与 structural/predictive validation 分列保存。
- sealed manifest 的五个 split 必须互斥并使用内容寻址 observation ID；
  每个 ID 都解析到带来源 hash、split 和 cutoff 的 `Observation`；manifest
  还绑定 parent、patch、evaluator、probe registry、policy、注册/开启时间和
  独立 custodian。
- 已评估 patch 的 hard negatives、失败 receipts 和完整 replay record 进入当前
  本地、非权威 shadow lifecycle；writer 签名接入前，REJECT 不能作为不可逆
  全局结论。Verifier abstention 的统一持久化仍是后续工作。
- Certificate 必须绑定 patch、receipt、ledger、policy、reduction 和拟议 child；
  非活动记录 API 会从完整输入重算，active 写入在可信根实现前硬禁用。

## 文献访问边界

`references/manifest.json` 逐项区分 full text、author manuscript、metadata /
landing page、访问受限、无官方源码和第三方复现。不会绕过付费墙，也不会把
HTML access page 命名成 PDF。
