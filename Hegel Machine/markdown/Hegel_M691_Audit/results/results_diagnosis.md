# M6.8 至 M6.9.1 成绩、成本与证据审计

固定提交 `4001bef56ffa77520c248b853e7fed3535ee9fe4`。结论：本轮原生 `17/20` 反题与 `18/20` 合题都能从实际官方回执重算，属于明显进展；但“独立端到端 1x”“同实际成本”“完整独立送审包”“全部旧缺陷闭环”仍需修订。数据包的完整性问题不等于成绩造假；分支继承/成本与可复现性问题也不能从分数本身自动消失。

## 1. 实测成绩的正确口径

| 实验/分支 | 原生总分 | W1 | W2 | TB-Science | 来源 |
|---|---:|---:|---:|---:|---|
| Bare R1 | 12/20 | 7/10 | 5/10 | 8/15 | 旧封存基线复制，非新运行 |
| M6.8 B′ | 9/20 | 4/10 | 5/10 | 5/15 | 原生 |
| M6.8.1 B′ | 11/20 | 6/10 | 5/10 | 7/15 | 原生 |
| M6.9 Thesis / Bare R2 | 8/20 | 4/10 | 4/10 | 4/15 | post-lock 分支 |
| M6.9 Antithesis | 12/20 | 5/10 | 7/10 | 8/15 | post-lock 分支 |
| M6.9 Synthesis | 10/20 | 5/10 | 5/10 | 6/15 | 原生；事后修复回放另列 |
| M6.9.1 Thesis / Bare R3 | 10/20 | 6/10 | 4/10 | 6/15 | post-lock 分支 |
| M6.9.1 Antithesis | 17/20 | 9/10 | 8/10 | 13/15 | 双轨系统中的分支成绩 |
| M6.9.1 Synthesis | 18/20 | 10/10 | 8/10 | 14/15 | 原生、无 Hegel 重评补分 |

最新报告的二十题主矩阵一致；Bare R1∪R3=15、R1∪R2∪R3=16、最新 Thesis∪Antithesis=18，二元选择 regret=0。相对同场 Bare R3，Antithesis 八胜一负；Synthesis 八胜零负。M6.9→M6.9.1 Antithesis 保留旧 12 个成功，新加 baseline-free、navigation、sparse-network、symbolic、traffic 五题。PZT 专项改动不可能仅凭这组多因素新运行就解释全部五题增益。[Report 35](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/markdown/hegel_v16_5_multi_precision_plan/35_M6_9_1_A_Clean_18of20_Breakthrough_and_External_Audit_Closure_Dossier.md#L1)

每个版本每题仍只有一次新运行；三次 Bare 来自不同阶段/脚手架，平均 10/20 是描述统计，不是三个严格同条件随机重复。旧二十题经过多轮结果诊断和代码修改，当前得分应标作“固定开发套件上的原生端到端结果”。不能用这些数直接认定陌生任务总体 90% 或稳态重复胜率 90%。

## 2. 17/20 不等于独立端到端 1x

Wave-2 的 Antithesis 继承了 **Wave-1 双轨合题** 的文本、AST 与 producer manifests。入口将 `w1_m6_9` 的双轨 summary 和顶层 ep_dirs 传入三个 consolidation 函数。实际 AMR、navigation、ODE、sparse-network 的 Wave-1 被选 producer 来自 Thesis。因此 17/20 是合法的反题分支成绩，但没有独立证明“只投入 28 轮/题的完整单分支学习系统也达到 17”。[双轨实验入口](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/scripts/run_m6_9_a_two_branch_study.py#L245)

后继应二选一：A. 报告为“双轨系统中反题分支 score”，完整计入前期知识形成成本；B. 真正重跑 Anti-only Wave-1→Anti-only consolidation→Anti-only Wave-2，或在双方共用的外部训练库上预先冻结知识，明确训练/推理分账。不能从当前一次双轨运行抽一半分支就声明获得完整的独立 1x 实验。

## 3. 实际成本

| 项目 | 实际轮数之和 | Total tokens | 项目定义的 effective tokens | Episode wall 之和（秒） |
|---|---:|---:|---:|---:|
| Bare R1 | 445 | 14,872,858 | 12,752,890 | 16,544.14 |
| 同场 Bare R3 | 441 | 14,800,370 | 12,811,346 | 21,569.68 |
| 最新 Antithesis | 442 | 19,848,910 | 17,648,398 | 35,239.56 |
| 最新 Synthesis（含两分支） | 883 | 34,980,064 | 30,781,120 | 42,360.05 |

Antithesis 对同场 Bare 的 tokens 增加 **34.11%**、项目 effective tokens 增加 **37.76%**；Synthesis 对 `Bare R1+R3` 增加 **17.88% / 20.41%**。因此可以说相同 *turn cap*；不能说实际算力/费用完全相同。Synthesis 另用了 330,784 模型 tokens，已被计入 34,980,064；它不只是机械选择。上述 wall 是多个 episode 的时间和，会混入锁等待且受并行影响，不是 study elapsed，也不是 CPU-seconds。

study manifest 的实际整场 elapsed=10,107.54 秒。BioMystery 本轮三个评分流实际消耗 judge tokens 19,313（Synthesis）+19,294（Thesis）+19,305（Antithesis）=**57,912**；manifest 只列 19,313，漏了 38,599 个 post-lock branch judge tokens。评估预算可与求解预算分栏，但应完整记账。[最新 study_manifest](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/docs/evidence/m6_9_1_a_two_branch_frontier_40ep_20261123/study_manifest.json#L1)

## 4. 成绩与顺序的独立交叉核对

四个新 bundle 共 240 个 DAG 文件出现次数，180 个唯一 DAG 字节摘要：80 个历史 Bare 复制项是同一套 20 个，因此只有 20 个唯一旧基线。每个 JSON canonical hash、prev hash、parent、seq、累计 token/cost 均由审计脚本独立复算通过。每份 summary 与本文件夹原生 official report 保持一致；deferred 分支的实际成绩来自另外的 post-lock report，而非修改旧 summary。

最新 20 个合题 DAG 均先 D3 commit 后 terminal official verification。14/15 TB 原生 pytest 回执有非零时间戳，均显示合题评分结束早于两分支评分开始；sparse 的 timestamp 为 0，无法用该字段独立确认时间，只能依赖代码顺序和 DAG。该证据支持本次选择没有读当前题两分支的官方结果；它不等同于完全独立的外部时间戳见证。

## 5. 校验和与便携送审包不完整

| Bundle | SHA 清单条目 | 当前 checkout 匹配 | 缺失 |
|---|---:|---:|---:|
| M6.8 | 899 | 899 | 0 |
| M6.8.1 | 813 | 813 | 0 |
| M6.9 | 1536 | 1534 | 2 个 pyc |
| M6.9.1 | 2103 | 2099 | 4 个 pyc |

最新缺失的是 SHA256SUMS 第 14、44、52、53 行的 `3ad93579…pyc`、`dae02eab…pyc`、`eec141ff…pyc`、`ef9cfa24…pyc`；git 里也没追踪它们。它们来自 reactor 与 AMR producer 状态。现有文件未发现摘要篡改，但交付不能重现“2103/2103 全绿”。可显式把可重建 pycache 从科研状态契约中排除，重新生成清单并保留勘误；或确实提交缺失 blob。不能静默把校验器改成忽略缺文件。[最新 SHA256SUMS](https://github.com/erzhu419/Assumption-Agent/blob/4001bef56ffa77520c248b853e7fed3535ee9fe4/Hegel%20Machine/docs/evidence/m6_9_1_a_two_branch_frontier_40ep_20261123/SHA256SUMS.txt#L14)

仓库 ZIP 为 612,200 bytes、138 个成员、CRC 完整；但它只包含 SHA 清单中 **111/2103** 个条目，其余 **1992** 个没有打包。只有 40 个顶层 official reports（20 旧 Bare+20 Synthesis），没有 40 个 post-lock branch reports、任何 DAG、JSONL、producer manifests/blobs。六份核心源码也不能覆盖依赖导入。因此它适合作为摘要审阅包，尚不支持其声称的独立完整验证、17/20 分支验证和一键全测试。

## 6. 调用量与知识收益应分开

4141 次运行调用全部有 completed 记录与 JSONL 对齐，是上一轮可观测性的改进；但 4121 次（99.52%）集中于 nano。全部调用中 agent=4104、host_probe=37。6/15 TB 有激活；没有任何 `wave1_sublated_*` 或 `wave1_manifest_*` API 调用回执。这不排除文本阅读或内联利用，只表示“自动形成的新 API 已被后继运行消费”尚未有实证。

nano 的三个 `pi_fold_temporal_regime_selector` 调用只在 `tmp__indent_reduce.py` 第 78 行把返回值打印，未把其输出接入卸载拟合/材料估计。当前证据支持诊断调用，不能据此证明 Mori–Zwanzig/跨尺度迁移贡献。全部 15 个 TB 反题启动时，64 个 staged 模块的 D1C 都是 `qualification_exec_failed`、qualified_count=0，随后仍可运行：模块不一定坏，资格检查或调用协议确实未形成有效门。详见能力审阅报告。

合题 6 个选择 Thesis 的 TB 题，其 summary 的 `producer_skill_manifest_summary` 仍写着 Anti 的 artifact ID，而实际 04b manifest 是被选 Thesis 的正确 ID。需要从被提交的真实 artifact/manifest生成 summary，不可始终复制反题字段。这不改变官方成绩，但会误导自动能力归因。

## 7. 二元零 regret 的局限与两道未解题

**virtual-baseline：**两个分支都没过 20 mm，所以二元 regret=0；但合题选择最大误差 247.487 mm 的 Thesis，丢掉最大误差仅 29.280 mm 的 Anti。连续性能差了 8.45 倍。这是已发生的选择问题，不是抽象担忧。修复公共跨几何探针的 ABI/无结果语义，用相同独立场景比较，不把模型自建、各自不同的模拟分数直接排序；同时保留候选集合和 geometry-conditioned 各分支。见逐题附录。

**hb024：**当前提供的主要是微生物 marker，未获得足以命名具体宿主和组织的证据。不能因为知道 rubric 就硬编码三个物种。保留异步参考检索、host-marker 与技术污染排除、多层社区证据和可识别性判定。预算不足/网络 WAITING 与数学不可识别应分开；官方失败照实登记，另记合理 abstention。

## 8. 后继实验最小闭环

1. 先保留当前 18/20 原生结果，修正独立 1x、实际成本、ZIP、SHA 与身份描述；旧结果不重写。
2. 用已存 20 对候选离线修复/回放公共选择器；覆盖每一种真实 task ABI，错误必须显式为 error，不能落成一个假的分数。开发回放不能冒充新验证。
3. 按真实 ABI 资格检查一个新模块，记录 binding→pilot→adoption→retention→gain；有复杂依赖就保留完整方法和构建步骤，不退化成小纯函数计数。
4. 冻结知识库和运行代码后，做 Anti-only 完整两波路线，或双轨对 Bare 双轨的同期等 tokens/CPU 上限比较。将验证、知识形成和推理成本分栏。
5. 在已曝光二十题之外，用新任务实例/几何/控制历史/噪声生成器验证；先选与最新机制直接有关的一小组，再决定是否扩大，不直接花钱重复刷同一套直至满分。

历史复用原则：仓库已有 `paid_score_history`、`research_episodes`、`retain_evaluated_python_model`、分支支持/confirmation 等路线；此次问题是这些契约在 Frontier 路径中的真实执行与证据绑定，不是必须再建一套新记忆或简化哲学树。`historical_confirmation.json` 是本次只读 registry 查询记录。完整二十题诊断见 `twenty_task_diagnosis.md`；可重算数据与代码见同目录。
