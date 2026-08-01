# 交给网页端 GPT 的 Phase-2B / Phase-3 精确冻结问题

以下问题不会阻塞合同和隔离基础设施施工，但会硬性阻止生成正式 Phase-2B
holdout，或阻止 Phase-3 签发 outside-language certificate。请基于
`answer_for_gpt_phase2_phase3.md` 给出可直接写入 JSON/schema 的唯一答案。

## 1. 720 case table 与 margin strata 的 12-case 冲突

当前每个 family × scale cell 的 60 例中，`insufficient / genuinely ambiguous`
为 8 例，因此全体只有 96 例；但 15% ambiguous/insufficient margin stratum 要求
108 例，相差 12 例。

请在以下方向中选择并给出精确新表：

1. 每个 family × scale cell 的 20 个 positive 中抽 1 个改为 admissible-scale-set
   case，使最后一层增加 12 例；此时“20 个 unique answerable”应如何改名和计分？
2. 保持 case type 表不变，把 margin strata 改为能对 720 整除且与 96 ambiguous
   一致的比例；请给出四层的新整数计数。
3. 把额外 12 个 admissible-set case 作为独立第七类，同时保持总数 720；请给出
   从其他 case type 扣除的精确数量。

同时请明确 admissible-set case 是否计入 answerable、scale-set accuracy、joint exact
和 abstention specificity 的哪个分母。

## 2. parity-like target 必须重新精确定义

旧 DSL 允许：

```text
absolute(difference(x, y))
```

所以二元 XOR 已经在旧语言内；嵌套 absolute-difference 也能表达低元 parity。
仅禁止名为 XOR/modulo/parity 的 token 不足以证明 language-outside。

请确认并冻结：

- 首个 target 是否改为 size 5–8 bounded `EntitySet` 上的 generic parity reduction；
- 精确集合大小、输入 universe、target truth table 和 train/validation/holdout 分布；
- 在 operator semantics 和 executable DSL 尚未冻结前，二元 XOR 是否只记为
  intended-numeric-semantics target-design sanity；冻结后由何种 executable witness
  才能正式判 `IN_LANGUAGE`；
- 若完整 frozen closure 找到等价表达，是否自动放弃该 target 并在看 hidden result
  前按什么规则选择替代 target。

## 3. hidden sink 的可观测性

请确认 null control 的 sink 是：

> 已作为 opaque typed measurement 出现，但被初始 scope/aggregation 遗漏，旧
> conservation + `aggregate_by` 或 scope refinement 可精确恢复。

而不是真正 latent/unobserved sink。请给出正确 old-DSL program、scope support
下限、aggregation map 和 fail/no-false-invention 判据。

## 4. 50,000 search budget 的计数口径

请明确 50,000 是：

- syntactically canonical programs 的数量；还是
- extensional equivalence representatives 的数量。

建议采用前者：先计 syntactic canonical programs，extensional quotient 只作有证据
的优化；若完整 closure 超过 50,000 或枚举未闭合，只能
`INCONCLUSIVE_BUDGET`。请确认 canonical traversal order、node-expansion 上限和
独立 replay 要绑定的 Merkle/root 字段。

## 5. 把 old DSL 真正变成 finite closure

请给出唯一、完整的：

- rational grid；
- `BoundedInt` 范围和 interval grid；
- 各 primitive sort 的有限 cardinality；
- quantity/context/task identifier vocabulary；
- aggregate/transform catalog；
- equivalence tolerance；
- scope minimum support；
- exact operator typing 与 undefined semantics。

如果完整 grammar 在这些参数下仍超过 50,000，请明确应缩小哪些 operator/arity/
depth，而不是把截断搜索称为 outside proof。

## 6. MDL code table

请冻结：

- AST shape、arity、clause boundary 的 prefix code；
- token class sizes；
- identifier 到 Elias-delta 正整数 index 的 registry；
- rational parameter code；
- scope code；
- new-symbol definition code；
- `log2` 精度、舍入和边界比较规则；
- MDL 的 train/validation invention split。

`ΔL ≥ max(32 bits, 0.05 L(D|P_old))` 已冻结，但没有上述 code table 就不能产生
可重放的 MDL certificate。

## 7. Phase-2B 剩余统计与运行细节

请一次性冻结：

- preservation transformation → law-family 适用矩阵和总 pair 数；
- embedding model、LLM model/prompt、flat typed baseline、bootstrap seed/次数/
  resampling unit；
- semantic-conflict subset 是 720 内切片还是额外 challenge；
- shared-footprint cell taxonomy 和“单一 measurement 承担 family discrimination
  不超过 50%”的精确统计量；
- answer reveal 前，哪些纯基础设施失败允许重跑以及最大次数；
- validation 两轮都失败后怎样生成新 validation version。

## 8. public wire 的 covert-answer-channel 审计

当前 allowlist 只能禁止显式 `law_family/gold/rank` 字段；合法 UUID、provenance hash、
role-candidate 集合、missingness pattern 和 unused transform 仍可编码答案。请冻结：

- UUID/opaque ID 的独立随机生成、全局 shuffle、seed custody 和 collision policy；
- 对所有允许字段分别做的 answer-correlation / mutual-information / permutation test，
  显著性阈值和 multiple-testing correction；
- consistent-renaming invariance 覆盖哪些 ID、至少多少次 permutation；
- unused field/transform、序列顺序、JSON 长度和 missingness 的 side-channel 判据；
- `standard_error` 转 closed interval 时的置信水平、分布假设与 multiple-comparison
  correction；在这些语义未冻结时是否只允许 `absolute_bound` 进入 formal selector。

## 9. closure/MDL receipt 如何从自报记录升级为可信证书

请给出可直接实现的唯一方案：

- closure archive 的逐 program record schema、canonical ordering、chunking 和 Merkle
  root；target truth-table root 如何绑定 bounded universe；
- independent evaluator 如何重放 canonicalizer/enumerator/operator semantics 并重算
  closure cardinality、match set、closure root 与 target root；
- complete enumeration 是双实现一致、proof-carrying enumeration，还是一个签名
  custodian attestation；可信根、key rotation/revocation 和 replay policy 是什么；
- provisional structural receipt 满足哪些机器条件后，才允许签发
  `OUTSIDE_FROZEN_CLOSURE` certificate；
- MDL scorer 如何从冻结 scoring partition、program AST 和 code table 重算四个长度，
  而不是接受 caller-supplied `Fraction`；certificate 应绑定哪些 roots/IDs。

回答时请优先给 machine-readable 表格、枚举值和公式，不要只给原则性描述。
