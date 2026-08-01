# Hegel Machine

这是 Assumption Agent 的独立 v3 工作区。v1 是已有效果的 Assumption Agent，
v2 是广义对位/GSCL；本目录冻结它们中需要复用的对象与算法，但活动代码不
反向 import 旧目录，也不继承旧实验的效果分数。

当前已完成里程碑的唯一名称是 **Phase-2A Controlled Typed-Selector Mechanics
Qualification**。一般能力称为 **Explicit-Projection Typed Structural Selection**，
其中 scale 子能力只能称为 **Scale-Indexed Candidate Projection Selection**。
活动包版本仍是 v0.2.0；它不是“自动发明哲学”的演示，而是可运行的 Phase 1–2
verifier、Phase-2A development qualification，以及正在施工的 Phase-2B/Phase-3
正式轨道合同：

```text
L0 observations
  → uniform verifier-ready synthetic witness bundle with anonymous labels
  → frozen adapter replay of family / binding / scale projections
  → frozen probe family / task geometry
  → six known-law verifiers
  → unique selection or deterministic policy-bound abstention
  → contrastive falsification
  → verified typed law match
  → measured evidence receipts
  → conservative theory patch gate
  → immutable theory version

Phase-2B formal track（尚未资格化）：

family-neutral-shaped typed evidence wire + independent side-channel audit
  → internal role/scale hypothesis enumeration
  → adapter-grid-committed interval selector with admissible scale sets
  → externally enforced untrusted OCI recognizer             [尚未可运行]
  → prediction commitment before answer reveal
  → one-shot independent-custodian scoring                    [目标，未实现]

Phase-3 freeze track（尚未开始正式 hidden experiment）：

partially frozen old DSL
  → untrusted closure-receipt wire                             [未重放]
  → sealed evaluator IN_LANGUAGE / OUTSIDE / INCONCLUSIVE     [尚未实现]
  → exact Fraction MDL arithmetic precheck                    [formal gate 关闭]
  → shadow-only conservative integration
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
  qualification；
- 一个不向 recognizer 提供 family、binding 或 scale answer key 的受控选择闭环：
  43 个统一 outer-schema case，其中 24 个 answerable、19 个应 abstain；每例从
  uniform verifier-ready synthetic witness bundle 经 frozen adapter 重放出 24 个
  projections，
  六个 family 都与两种 scale 完全交叉；严格策略要求目标的 binding 和 scale
  counterfactual 都完成 verifier 计算，并以 tolerance-normalized boundary margin
  决定唯一选择或 abstention；measurement 按 observable 的 witness-role footprint
  绑定，未改变角色的底层值必须由相互竞争的 binding 共同复用；12 对跨 episode
  preservation witness 覆盖 family × scale。
- Phase-2B family-neutral-shaped public wire allowlist：公开 ID 具有 UUIDv4 语法，typed value /
  interval、单位、时空 support、不确定性、来源 hash、aggregation graph、transform
  catalog 和 missingness 全部规范化；oracle、candidate-private、family-specific 和
  unknown fields 一律 fail-closed；这只证明字段形状，不证明允许字段没有编码答案；
- 从 public bundle 内部枚举完整 role binding 与 scale hypothesis 的 adapter，超预算、
  registry 漂移或非唯一 transform path 时不返回部分候选；
- public API 从 evidence bundle + frozen registry 重新运行 adapter，并绑定其完整
  candidate-grid commitment 的 interval selector core；缺任一 hypothesis、
  candidate metadata 漂移或任一结构 competitor 区间不确定时均 abstain，并支持
  unique structure + preregistered admissible scale set，而不是强迫 scale top-1；
- Phase-2B 720-case 统计合同、95% one-sided Wilson LCB、family/scale marginal slice、
  shared-footprint gate、三类 baseline registry 和 preservation 独立分母；
- seal → prediction commitment → reveal → consumed 的 immutable lifecycle model、
  answer salted-commitment opening 校验和进程内原子防分叉 guard，以及 read-only
  input/root、no network、无 repo/generator/answer mount 的 OCI launch-spec contract；
  这些不等于跨进程持久 one-shot custodian 或可运行 recognizer；
- Phase-3 old DSL 已决定部分的 content-addressed contract、50,000 program fail-closed
  budget、四态 adequacy 类型、untrusted receipt schema、精确 Fraction MDL 算术预检，
  以及按预期标准数值语义成立的 XOR2 target-design sanity；后两者都不是 formal gate
  或 executable closure verdict。

明确未实现或未宣称：

- 开放世界新关系发明成功；
- 自动 ontology / metric / evaluator 联合演化；
- 真实科学发现、下游 efficacy 或人类水平对照；
- 跨 evaluator epoch 的分数可比性；
- 把语义相似、LLM 自评、旧 fixture PASS 当作结构证据；
- 正式 Phase-2 exit：当前 API-blinded corpus 仍是 synthetic validation、未 sealed，
  且只资格化 role-scoped witness 到 frozen-projection 的 replay 与候选选择；报告内
  `controlled_api_selector_qualified` 只是机械闭环工程标签，不是正式里程碑声明；
- open-world family discovery，或从 raw natural-language/table/trajectory 证据
  抽取完整候选集合；
- 对 source-aware 攻击者隐藏答案：当前生成器和 evaluator table 均可见，所谓
  blinded 只指 recognizer API 不接收 answer key，且选择结果必须对所有公开 ID 的
  一致重命名保持不变；ID 不是密码学保密边界；
- independent raw-evidence projection 或不受信 recognizer 隔离：fixture 值由
  evaluator case spec 反向构造，当前 accuracy 只测受控 selector mechanics；
- 真实 semantic/embedding baseline 的效果比较。人工 semantic decoy 的 accuracy /
  gap 不参与 exit gate；semantic metadata replacement invariance 是 anti-leak 硬检查；
- 当前 controlled vertical slice 不是 sealed holdout；即使传入结构完整的
  manifest，在独立 custodian 签名验证器实现前也只到
  `candidate_framework`，不会写入 active theory graph；当前 lifecycle 只是
  本地非权威 shadow ledger，writer 签名接入前不产生不可逆全局 REJECT。
- Phase-2B typed evidence → executable verifier projection 的正式 compiler 和完整
  unsealed pipeline validation；当前只完成 wire、candidate enumeration、interval
  selector 与 runner/state-machine 合同；
- 720-case secret holdout、独立 custodian、真实 OCI attestation、三套冻结外部
  baseline 或 consumed score report；
- allowed UUID/provenance/role-candidate/missingness/transform 字段的 answer-correlation
  与 side-channel audit、随机并全局 shuffle 的 ID 分配证明，以及把 standard error /
  absolute bound 编译成闭区间的冻结 uncertainty semantics；
- 专用 recognizer CLI、严格 720-bundle prediction archive evaluator、签名 SBOM /
  runtime attestation 验证器和跨进程持久的 append-only CAS ledger；
- Phase-3 outside certificate：当前协议仍缺 exact rational grid、bounded universe、
  operator semantics、equivalence/canonicalizer/enumerator、MDL code table 和精确
  high-arity parity/hidden-sink generator；
- sealed closure archive replay/root recomputation verifier 与从冻结 partition/code table
  重算长度的 MDL scorer；当前调用者自报 receipt 不能产生 semantic certificate；
- 把“禁用 XOR/parity 名称”当作不可表达证明。旧 DSL 的
  `absolute(difference(x,y))` 已能表达二元 XOR，因此最终 target 必须由完整 closure
  的 extensional enumeration 决定。

## 快速运行

无需联网或模型权重：

```bash
cd "Hegel Machine"
PYTHONPATH=src python3 -m pytest -q -s
PYTHONPATH=src python3 -m hegel_machine benchmark \
  --output artifacts/phase2_benchmark.json
PYTHONPATH=src python3 -m hegel_machine phase2-exit \
  --output artifacts/phase2_exit_benchmark_v2.json
PYTHONPATH=src python3 -m hegel_machine phase2b-preregister \
  --output artifacts/phase2b_preregistration_v1.json
PYTHONPATH=src python3 -m hegel_machine phase3-preregister \
  --output artifacts/phase3_preregistration_v1.json
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
[`docs/evidence_boundaries.md`](docs/evidence_boundaries.md)，Phase-2A development 协议见
[`docs/phase2_exit_protocol.md`](docs/phase2_exit_protocol.md)。正式轨道施工边界见
[`docs/phase2b_preregistration.md`](docs/phase2b_preregistration.md) 与
[`docs/phase3_freeze_readiness.md`](docs/phase3_freeze_readiness.md)；待网页 GPT
消歧的问题见
[`docs/questions_for_gpt_phase2b_phase3_freeze.md`](docs/questions_for_gpt_phase2b_phase3_freeze.md)。
