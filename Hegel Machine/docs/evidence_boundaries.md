# Evidence and claim boundaries

## 当前可以说什么

当前实现和测试支持：

> 在一个冻结的有限定律库和受控离线样例中，系统能够在已提供 typed law
> binding 的条件下，不用语义相似度作为接受依据，计算 structural residual，
> 拒绝缺观测、虚构实体、角色交换、符号翻转和尺度不兼容，并把测得的证据送入
> 不可变、受 evaluator epoch 约束的保守候选评估流程。

这叫 verifier/integration qualification，不叫 law-family discovery，也不是
Phase-2 exit evidence。当前 controlled holdout 只是代码内合成重放，manifest
未 sealed，因此 gate 输出 `candidate_framework`，不授权 active graph mutation。
即使 manifest 结构检查通过，当前版本也没有外部签名可信根，不会晋升 ACTIVE。

## 当前不能说什么

- 没有新关系发现、开放世界本体演化或现实科学发现的效果证据。
- v1 的 framework-growth 分数含公式化 fixture；它们只提供 schema/threshold
  原型，不是 v3 的 PASS 证据。
- v2 的 GSCL controlled corpus 是合成 qualification，不是 downstream efficacy。
- ARN 已被用于实现后验诊断，不能再叫 untouched。
- 文献或 repo 被归档不等于其结论已在本项目复现。
- benchmark 中反向构造的 semantic-only baseline 只证明验收路径没有读该分数，
  不能作为与真实 embedding 系统的效果比较。

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
