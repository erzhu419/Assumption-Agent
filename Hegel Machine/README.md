# Hegel Machine

这是 Assumption Agent 的独立 v3 工作区。v1 是已有效果的 Assumption Agent，
v2 是广义对位/GSCL；本目录冻结它们中需要复用的对象与算法，但活动代码不
反向 import 旧目录，也不继承旧实验的效果分数。

当前版本不是“自动发明哲学”的演示，而是可运行的 Phase 1–2 verifier 基线：

```text
L0 observations
  → typed structural episode
  → frozen probe family / task geometry
  → six known-law verifiers
  → contrastive falsification
  → verified typed law match
  → measured evidence receipts
  → conservative theory patch gate
  → immutable theory version
```

已实现：

- 统一、不可变、内容寻址的 `TheoryState`；
- observation / probe / scale / law / candidate / theory patch / evidence schema；
- probe family 诱导的伪度量与 observational quotient；
- symmetry、monotonicity、conservation、complementarity、negative feedback、
  locality 六类确定性 verifier；
- 固定的本体不足诊断阶梯，只有 `ONTOLOGY_DEFECT` 可以打开语言扩展；
- Generator / Formalizer / Falsifier / Evaluator / Promoter 五权分离合同，
  gate 会核验 actor、registered probe、split、cutoff、独立性和 evaluator epoch；
- sealed holdout manifest 的分区、开启时间和独立 custodian 均受内容 hash 约束，
  每条封存 observation 必须带来源 SHA-256 并以真实内容 ID 注册，不得跨 split
  重用；manifest 还绑定 parent、patch、evaluator、probe registry 和 gate policy；
- 不读取 semantic score 的保守晋升门；
- 内容绑定 certificate、单坐标 scope compiler 和原子 promotion API，调用者
  不能把手工构造的 certificate 当授权；当前没有外部签名可信根，因此
  active promotion 明确 fail-closed；
- robustification 与 idealization 的首版候选及合同；
- 一个离线、确定性、含低语义正例和高语义负例的已知定律 verifier
  qualification；它不测 law-family retrieval。

明确未实现或未宣称：

- 开放世界新关系发明成功；
- 自动 ontology / metric / evaluator 联合演化；
- 真实科学发现、下游 efficacy 或人类水平对照；
- 跨 evaluator epoch 的分数可比性；
- 把语义相似、LLM 自评、旧 fixture PASS 当作结构证据。
- 当前 controlled vertical slice 不是 sealed holdout；即使传入结构完整的
  manifest，在独立 custodian 签名验证器实现前也只到
  `candidate_framework`，不会写入 active theory graph；当前 lifecycle 只是
  本地非权威 shadow ledger，writer 签名接入前不产生不可逆全局 REJECT。

## 快速运行

无需联网或模型权重：

```bash
cd "Hegel Machine"
PYTHONPATH=src python3 -m pytest -q -s
PYTHONPATH=src python3 -m hegel_machine benchmark \
  --output artifacts/phase2_benchmark.json
PYTHONPATH=src python3 -m hegel_machine vertical-slice \
  --output artifacts/controlled_vertical_slice_v1.json
PYTHONPATH=src python3 -m hegel_machine demo
```

## 目录

- `markdown/`：五份指定源文档，另含被《黑格尔机》直接引用的范畴/态射对话；
- `references/`：文献、网页、元数据、固定 commit 的关联 repo 快照与校验和；
- `legacy/`：v1/v2 最小冻结快照，仅作迁移与审计；
- `src/hegel_machine/`：完全独立的 v3 活动代码；
- `tests/`：单元、治理和纵向闭环验收；
- `docs/`：架构、证据边界、迁移映射和来源说明。

更严格的 claim 边界见
[`docs/evidence_boundaries.md`](docs/evidence_boundaries.md)。
