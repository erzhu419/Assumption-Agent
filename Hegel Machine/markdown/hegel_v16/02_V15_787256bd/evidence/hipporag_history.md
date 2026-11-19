# HippoRAG 1/2、Assumption-Agent 历史与 V15 最小重接审计

审计对象固定为 `erzhu419/Assumption-Agent@787256bd415515905359626f6a5498174ad5db3c`。本笔记是本次审计的历史来源补充。只读取原论文、作者仓库和当前上传的 V15 源码；没有重新运行付费模型，也没有修改项目源文件。

## 1. 结论先行

用户关于“早期确实用过 HippoRAG / 图记忆，而且有过效果”的记忆有真实材料基础；不应将早期工作统称为无效。但至少存在四种不同证据，必须分别陈述：十题跨域结构匹配演示、十五题真实 QA 的简化 PPR 探针、三千题检索策略选择器、十二题官方 HippoRAG 对照。它们不是同一个实验，也不都支持相同的泛化主张。

当前 V15 Hegel 主研究循环的 `_recall` / `KnowledgeLibrary.retrieve` 没有调用早期 `SimpleAssumptionGraph` 的 PPR 或官方 HippoRAG adapter。当前 `ResearchGraph` 是研究问题与可执行依赖图，不是语义检索图。因此，“重新接上关联检索”是有意义且尚未完成的方向；它能够改进候选发现与排序，不能单独解决新假设生成、范围验证、冲突调和或成本优势。

建议保留当前可执行假设、版本、范围和消费链，在它前面增加一个可替换的候选检索接口。先复用原经历文本、关系和结构图式的联合索引，不搬回旧的任务路由、固定 strategy-ID 加分、预设答案图式或全样本选择器。

## 2. 原论文到底做了什么

### 2.1 HippoRAG 1

官方论文为 NeurIPS 2024 的 *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models*，arXiv `2405.14831`。

论文 §2、Figure 2（正文第 3–5 页）明确描述：LLM 用 OpenIE 从段落抽取主谓宾关系，主/宾短语构成节点，关系构成边；检索编码器另加近义短语边。在线从查询抽取实体、链接到图节点，并以其为 seed 做 Personalized PageRank；通过短语—段落关联矩阵把激活分数汇总为段落排名。节点 specificity 用局部出现频次降低过泛实体影响。

这已经包含关系结构和多步关联，不是只做“相似概念两两连边”。但 PPR 排名本身不是逻辑证明、反事实验证或跨领域机制等价性判定。

### 2.2 HippoRAG 2

上传论文为 *From RAG to Memory: Non-Parametric Continual Learning for Large Language Models*，arXiv `2502.14802v2`（2025-06-19），ICML 2025 / PMLR 267。已抽取全文并视觉检查第 4 页 Figure 2、第 7 页 Tables 2/3、第 8 页 Table 4、Table 5 与 Figure 3。

其关键结构是：

| 层 | 实际操作 | 对 Hegel 的含义 |
|---|---|---|
| 索引 | OpenIE triples；phrase 节点、relation 边与 embedding 近义边 | 存关系与来源，不能只保留一个概念名 |
| 上下文 | 增加 passage 节点及 contains/context 边 | 假设应保留产生它的经历、条件、失败与修订上下文 |
| 查询链接 | 默认 query-to-triple，使用完整查询匹配关系片段 | 检索应包含当前目标、已知失败、角色与缺失机制 |
| Recognition memory | LLM 筛选相关 triples | 筛相关性，不能替代真伪、范围或机制验证 |
| 传播 | phrase 与 passage 双类 seeds，PPR 后排名 passage | 关联路径可提出候选；分数不等于真理置信度 |
| 回退 | 无有效 triple 时用 dense passage 检索 | 应允许无图收益/弃权，保留强普通方法 |

Paper Table 3 的平均 Recall@5 为 NV-Embed-v2 `73.4`、HippoRAG 2 `78.2`。MuSiQue、2Wiki、Hotpot 三个多跳任务平均为约 `80.2 → 87.1`，相差约 6.9 个百分点，对应摘要所说的约 7% associative improvement。Table 2 所报平均答案 F1 为 `57.0 → 59.8`。不要将“约 7%”泛称所有任务端到端质量提升，也不要描述成每个任务、每个指标均获胜。

Table 4 的多跳 Recall@5 消融有较明确的设计含义：完整方法 `87.1`；NER-to-node `74.6`；query-to-node `59.6`；去 passage 节点 `81.0`；去 LLM filter `86.4`。保留段落上下文、用完整问题关联关系，比单纯添加 LLM 筛选更加关键。

论文 §6.3 / Figure 3 把 NQ 与 MuSiQue 语料分成四段，逐步加入更多语料，测试固定问题的表现。它支持语料增长下的检索鲁棒性；MuSiQue 中双方仍随语料扩张下降。它没有验证“系统自主提出—反驳—修订假设，并让同预算未来研究更强”的完整主张。Appendix E 检查的是 100 个 Recall@5<1 的失败样本，其中相关 phrase 在筛选前不匹配为 7%、筛选后为 26%，说明在线筛选也可能误删桥接证据；这些不是全测试集错误率。

## 3. 历史证据逐项核对

### 3.1 2026-06-02 的设计记录确实连接了 HippoRAG 与结构类比

`reconstruction/md/category_morphism_dialogue_20260602.md` 第 17–104 行讨论保留 KG/Assumption Graph，另建 typed diagram / structural pattern 层，比较 role、composition、invariant。楞次—勒夏特列例子被用作“扰动—响应—约束”结构类比。这是设计记录，不是该类比有效域已实测的证据；其中跨领域物理或算法解释也不能因出现在旧讨论里就自动成为事实。

### 3.2 十个跨域结构匹配案例：存在正结果，解释限于预先编码的结构任务

来源：`reconstruction/md/rag_to_memory_baseline_comparison_20260606.md` 第 9–17、27–40、84–90 行；实现 `assumption_os/rag_to_memory_baseline.py`。

- 报告十题结构方法 top-1 `10/10`，RAG-to-Memory-style PPR `3/10`；top-2 为 `10/10` 对 `4/10`。
- 文档明确不是完整官方 HippoRAG 2。该 baseline 只看 label/domain/surface_text/kg_triples；结构方法另获 objects、morphisms、composition_laws、invariants、negative_invariants。
- 源码 `_passage_similarity`（344 行）、`_triple_similarity`（348 行）、`_expanded_tokens`（360 行）、`_add_synonym_edges`（373 行）是词项/固定同义组机制，不能等同官方 NV-Embed-v2 + OpenIE + LLM filter。

因此可以说“结构表示在这十个预先构造的匹配案例上命中了词面图方法漏掉的对应关系”。不能据此声称整体系统比官方 HippoRAG 2 强 3.33 倍，或已自行发现了那些结构不变量。该演示仍可作为回归测试和候选构造诊断。

### 3.3 十五题真实 QA：简化 PPR 没有整体胜 BM25

已读取原始 JSON：`phase four/assumption_graph/paper_readiness_20260604/hipporag_qa_probe_20260606.json`，blob `d53b1c1b11955bf106c74796fe342cf19e5335ae`。`aggregate.overall`：

| top-5 指标，n=15 | BM25 | 简化 PPR |
|---|---:|---:|
| any support recall | 0.8667 | 0.8667 |
| all support recall | 0.1333 | 0.1333 |
| 平均 support fraction | 0.5111 | 0.4889 |
| answer coverage | 0.2667 | 0.2000 |

真实数据来自 HotpotQA、MuSiQue、2Wiki 各五题。直接 structural morphism 在这些 QA 上弃权；safe policy 回退 BM25。六题 live reader 探针报告 BM25 0/6、PPR 1/6，样本过小，不能推导稳定答案优势。这个负结果针对当时简化 PPR 与适配方式，不否定原论文方法，也不否定结构类比的其他用处。

### 3.4 三千题 Meta-QA：有报告收益，但现代码存在选择层数据隔离限制

来源：`reconstruction/md/meta_qa_evolution_20260607.md` 第 80–145 行；`assumption_os/meta_qa_evolution.py`。报告 top-5 all-support recall：BM25 `0.3077` → fixed meta `0.3627` → learned meta `0.3630`；平均 support fraction `0.5988→0.6384`；answer coverage `0.4730→0.5033`。DistilBERT reader300 的 EM `0.1167→0.1267`、F1 `0.1883→0.2039`，fixed 与 learned 在此切片相同。额外 learned 相对 fixed 的 all-support 增益仅约 `.0003`，报告 CI 跨 0。

但 289–300 行先对完整 `rows` 运行 `_evaluate_hypotheses(rows)`，据此选择 accepted policies/priority，并为相同 rows 写入 `retained_policy`；之后才运行 crossfit selector。836–880 行 bucket 模型本身排除了当前 fold，但 `_select_learned_policy_for_row` 第 962 行仍把全样本选择所得 `retained_policy` 当作 base/fallback。

这是选择层可确认的隔离问题：排除当前行 gold 作为运行时特征，不等于整套策略选择独立于评价集。当前结果可以作为有前景的开发集发现，不能直接作为严格未见任务效益证明；并非因此每一个排序结果都无效。八种策略也是源码显式定义的候选集合，不能说任意新策略由系统自行创造。

两份大 JSON 的 connector 返回了 blob SHA 但空内容，本次未逐行重算 3000/reader300 表：`meta_qa_evolution_learned_full3000_20260608.json` 为 `718e847bca7376b952312431e30822c44079ef15`；`meta_qa_evolution_reader300_learned_20260608.json` 为 `6cd5e1b3416e403c61a47a74171b0c1a5e0eae20`。不要把上述报告数字表述成本次独立复算。

### 3.5 后期实际官方 HippoRAG 对照：十二题小样本有正信号，RAW 很弱

此处不能仅说“官方代码接入只有 smoke test”。确有两层材料：

1. `reconstruction_v2/manifests/official_hipporag_runtime_adapter_qualification_v1.json`：官方代码 commit `ef2f14c4f254f11ac29f9395f262466ad1bb4d10`、HippoRAG `2.0.0a4`；四个合成文档、一个问题，index/retrieve/qa 路径通过。回执明确为非计分资格验证、tiny local causal model、不评价答案质量。
2. `reconstruction_v2/manifests/musique_recursive_study_m1_aggregate_report_v1.json`：十二题、29 个 support，36 次 retrieval 终止后统一计分，三臂如下。

| M1 臂 | support hits | Recall@5 | any-support 命中题数 |
|---|---:|---:|---:|
| canonical_RAW | 7/29 | .241379 | 7/12 |
| frozen_P | 14/29 | .482759 | 9/12 |
| official_HippoRAG | 14/29 | .482759 | 10/12 |

`musique_m1_retrieval_runner_v1.py` 第 468–469 行的 RAW 是 **直接取 corpus 前五段**，不是 BM25 或强 dense retriever。`frozen_P` 为有限 DSL 选出的 TF-IDF 词项检索：标题权重 4、正文权重 1、一个 seed document、一跳 entity-token 扩展权重 2。已读取冻结 program 与 pre-run freeze；形成和测量块有不同 hash，代码将官方臂接到实际 runtime adapter。

P 相对 RAW 是 7 gain / 1 harm / 4 tie，并获 generation-one promotion。正确解释为：这个选定小切片上，有限检索程序和实际官方实现均超过极弱的原始顺序基线，二者 support 总数打平。不是超过强检索，也不是证明长期递归改进、最终 QA 或科学新假设效用。公开回执没有 item-level rows；本次只能核对 aggregate、冻结程序与执行代码，不能逐题重算或复核全部私有数据来源。

M2 回执 `musique_recursive_study_m2_infrastructure_failure_disposition_v1.json` 明确环境阻断、未形成 aggregate、未计分，后续效益 unknown。另一个 M3 evaluator 回执是 active 与 incumbent 都 18/29、net_support_hits=0、positive_search_utility=false。后者是另一有限 DSL 的评价器试验，不能拼接为 M1 后递归一直进步。

## 4. 哪些代码可复用，哪些不能原样继承

| 位置 | 可复用部分 | 必须避免的旧耦合 |
|---|---|---|
| `assumption_os/graph_memory.py` `JsonlGraphStore` 50–118 行 | 节点/边/证据分开存储，支持来源与图版本 | 不把旧 confidence / metaproductivity 数值当 Hegel 已验证真值 |
| 同文件 `SimpleAssumptionGraph.reindex/retrieve` 128–219 行 | query seeds、图传播、候选子图返回，适合小规模低成本基线 | 当前是词项 cosine + PPR-like 排序；不是官方 HippoRAG 完整实现；固定反向权重不能定义科学关系逆命题 |
| `assumption_os/retrieval_policy.py` `retrieve_phase2_assumptions` 45–108 行 | 语义子图与 structural patterns 两路召回的接口思路 | 不复制 190–292 行 keyword route 与固定 strategy-ID boost map，也不直接把命中图式变成已确认假设 |
| `assumption_os/rag_to_memory_baseline.py` 223–334 行 | passage/phrase/triple/context 的小型图参考与可解释 seed/path | 不把固定 synonym 词表当完整 embedding 模型；不照搬十题答案图式 |
| `reconstruction_v2/.../adapter_v3.py` `run_official_hipporag_retrieve_only_v3` 28 行起 | 官方实现作为一个可插拔检索器；输入问题+段落、输出段落 indices 的窄契约 | 不搬整个评测/冻结/沙箱治理系统回 Hegel；不要以 tiny local model 对照冒充原论文配置 |
| `reconstruction_v2/.../musique_typed_retriever_formation_v1.py` `retrieve` 498 行起 | 只接受 question/corpus 的有限可执行检索程序接口 | 候选选择与阈值必须在形成集完成；强基线需 BM25/dense，而非前五段 |

早期 first-party consolidation probe 的 `_populate_store`（`full_v3_phase1_first_party_retrieval_audit.py` 95–125 行）直接构造 relevant/negative、confidence、stale/active 标签，是合成机制测试。可保留作回归测试，不能把这些预设标签变成生产知识来源或新增科学效果证据。

没有发现足够证据可以声称旧系统已有一个名为“automatic semantic truth edge”的真实机制。因此准确措辞是：**新接入不能把 synonym/related/analogy proposal 边自动升级为 equivalent/entails/supports_validated**，而不是虚构一个旧漏洞。关系边必须带 type、source、版本与证据状态。

## 5. 当前 Hegel 路径的确切落点

- `Hegel Machine/src/hegel_machine/cognition/unified_research.py` `_recall` 2125–2205 行：遍历 definitions/bindings/concepts/templates，检查 revoked、excluded_tasks、risk_quantile、已有 scope，再按 reuse_history/成功任务/generation/program size 排序，做角色和测量绑定。未调用旧图检索。
- `.../memory.py` `KnowledgeLibrary.retrieve` 523–539 行：按程序调用能力、可观测变量、scope、历史与反驳过滤；`propose` 554 行起还可按 `source_concept_ids` 限制来源，并保留具体 source。它提供了很合适的候选 ID 接口。
- `.../research_graph.py` 1–6、23–37 行描述的是问题节点、可执行回答、依赖、预算和恢复 frontier。不能因为名叫 Graph 就说 HippoRAG 已接入。
- `.../legacy_research_bridge.py` 的桥是 Hegel 旧研究/实验室入口适配，不能从文件名推断它桥接了 `assumption_os`。

限定结论：本次核查当前主召回路径没有旧 PPR 调用；并非证明整仓库所有历史入口都不可达。旧模块与官方 adapter 仍在仓库，是可复用材料。

## 6. 下一施工最小范围：五条接口与验收建议

### A. 先增加候选接口，保持既有研究验证链

新增 `retrieve_candidates(query, memory_snapshot, budget)`，返回 object_id + exact version、source episode/passages、候选关联路径、检索分数及花费。先由当前 `_recall` 的 records 进入统一 selector，或以 `source_concept_ids` 接 `KnowledgeLibrary.propose`；之后仍走现有类型、角色、范围、独立实测和消费事件。查询特征只允许当前可见的目标、观测、失败和资源限制。

“找到过候选”“生成了绑定”“参加了比较”“被最终程序消费”分别记事件。图排名不产生 `adopted_confirmed`，文案引用对象也不算消费。范围未知的对象可以供研究提问，不能直接当已适用程序运行。

### B. 第一版只索引真实经历与现有对象，保留原文上下文

为每个 source episode / 假设版本存一个 passage-like 记录：问题、观测条件、失败、程序、验证范围、来源证据。概念、角色和关系是索引视图，不覆盖原记录。可参考 HippoRAG 2 的 passage/phrase 双图；`related_to` 或 `analogy_candidate` 用于召回，真实 `refines/contradicts/validated_under` 由现有证据链提供。

先用本地词项/BM25或已有 embedding 的轻量图基线评估，不必同时引进完整大模型 OpenIE/recognition/新数据库。当前候选库小时穷举已能触达所有概念；图的首要假设应是更好地分配有限 fit/验证预算，而不是凭空扩展研究能力。

### C. 把结构类比作为有成本的候选映射

输出 typed role map、保持的关系/组合、符号方向、量纲或约束、未知项以及反例请求。词面相似但 sign/role/causal direction 不同的负例必须能弃权。跨领域“同图式”只是一条待测迁移假设；未经范围确认不能合并两个假设的事实身份，也不能共用支持证据计数。

可直接保留原 `search_structural_patterns` 的接口思想，避免把旧十题手写图式作为新实验的隐藏 gold。若人预置图式，应称“给定机制库的迁移选择”，另外测系统是否从经历自己形成图式。

### D. 一个时间顺序冻结实验分离三个收益

冻结形成集、未来任务与索引快照。相同可见历史/文档、相同后端和总预算下比较：当前 selector、BM25/dense selector、图 selector、图+结构映射 selector。普通强方法允许访问同样原经历并重建知识。任何 accepted-policy、路由阈值、图式选择都只在形成集确定；新任务 gold 不能影响全局 base/fallback，再做 crossfit 也不能补救已发生的选择污染。

分开报告：(1) 有用候选是否进入 top-k；(2) 在总预算内是否实际执行并通过；(3) 未来任务质量、达到同等质量的总成本、伤害/弃权率。检索 recall 上涨不等于最终效用上涨。把索引构建、抽取、维护、搜索、额外验证算入成本，并报告按预期复用次数摊销前后的数值。

### E. 先交一个因果清楚的完整案例，再扩大任务分布

一个最小可验收案例应能指出：哪次经历形成对象 H_v；图路径为什么在新任务召回它；结构映射保留/改变了什么；scope 检查和反例测试怎样决定采用/修订/弃权；最终程序实际调用哪个版本；同历史同预算普通方法如何重建，最终效果差多少。若当前强普通方法更便宜，就保留此负结果，定位为召回、绑定、拟合、验证或维护的具体成本，不能只改叙事。

## 7. 来源与复核边界

精确源清单、固定提交 URL、下载副本位置见同目录 `hipporag_sources_manifest.json`。论文主源：

- HippoRAG 1 官方 PDF：https://proceedings.neurips.cc/paper_files/paper/2024/file/6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.pdf
- HippoRAG 2 官方会议页：https://proceedings.mlr.press/v267/gutierrez25a.html
- HippoRAG 2 官方 PDF：https://raw.githubusercontent.com/mlresearch/v267/main/assets/gutierrez25a/gutierrez25a.pdf
- 官方代码：https://github.com/OSU-NLP-Group/HippoRAG

本次方法与表格复核以用户上传的完整PDF为主。没有进行全套历史实验重跑，也没有根据作者qualified/valid字段单独推导科学效果。
