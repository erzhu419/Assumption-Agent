# V15：对象移除、范围重放与缓存复核

审计目标：固定提交 `787256bd415515905359626f6a5498174ad5db3c` 的上传包 `v15_input`。本笔记以包内代码和保存回执为证据，不把报告中的“闭合”判断当作验收依据。

## 1. 结论与证据等级

**V15 对 V14 的三个关键问题作了实际修复：list 形态的 concept/abstract 残留、对象删除后禁止原始经历重建、scope 无法识别时把有独立来源支持的对象全局反驳。共同状态转换器也已接入在线和普通按需重建路径。** 当前 24 对历史重放和 36 对新目标的已记录路由字段一致，应保留这份进展。

但有一项在本轮回执已经出现的身份错误，以及几项受控输入可触发的边界缺陷。它们要求修补和定向验收；不能据此宣称全部 E1/E2 分数无效，也不能用 60 对有限字段一致声称全语义严格等价。

| 类别 | 发现 | 证据状态 | 建议处理 |
|---|---|---|---|
| 已实际发生 | 原始历史重建事件的 semantic_version 与入库 inventory 不一致 | 当前 control 3 回执直接可见 | 本轮交付修复；统一记录、事件、缓存和干预身份 |
| 确定函数缺陷 | 找不到相同实验时回退到首个档案，仍报告 reused_exact_protocol | 实际回调函数的受控输入触发；未证明本轮 E1/E2 命中 | 去掉错误回退，导出实际 match mode，再重验现有匹配 |
| 确定函数缺陷 | revoked 的第二父节点仍可让共享后代可执行 | 实际移除函数和调用可用性检查触发；未证明本轮库存在此形态 | 有效支持根过滤 + 支持证据验收 |
| 接口边界 | 指定版本经过初始检查后，实质按 cid 删除，inventory 也按 cid 折叠 | 同 cid 双版本控制触发；未证明本轮出现并存版本 | 明确并强制当前单版本不变量，或真正按版本组织 |
| 验证范围 | “exact state/semantic parity”比较的是选定状态和路由集合字段 | 脚本直接证明 | 保留结果，收窄措辞；补最小真实状态值检查 |
| 次级依赖缺口 | feature_bounds 影响重放的实验构造，却不进入档案哈希 | 源码依赖链 + 哈希控制；未证明冻结实验中该元数据被修改 | 增补哈希投影，不另建缓存架构 |

## 2. 复核方法与文件

源码均在 `v15_input/bundled_sources/src/hegel_machine/cognition/`。

- `memory.py`：对象清单和移除、可执行载体。
- `branch_reconciliation.py`：范围、重放、原始历史重建、共同转换器。
- `premise_commitment_index.py`：对象版本及档案哈希。
- 测试与交付脚本在 `v15_input/bundled_sources/tests/` 和 `scripts/`。

本次控制：

- `analysis_v15/scope_controls.py` / `scope_controls.json`：调用上传源码的实际 AST 函数、实际 KnowledgeLibrary 类体、Program/模型数据类；最小库通过 `__new__` 初始化。检查实际 definitions、bindings 与 `_calls_available`，不是另写一套模拟删除算法。
- `analysis_v15/scope_fixed_control.json`：用上一轮已有 scope 探针对 V15 重跑；显式固定 alignment summary 和后处理 refinement，隔离原先错误 fallback。
- `analysis_v15/scope_actual_receipt_version_mismatch.json`：原始 control 3 回执的精确摘录。

这些控制**没有重跑完整研究代理或物理环境，没有调用付费 API，也没有修改输入包**。它们证明特定函数分支的行为，不证明每项边界问题在当前 E1/E2 实验发生过。

可复现入口：

```bash
python analysis_v15/scope_controls.py --bundle v15_input --output analysis_v15/scope_controls.json
python v15_input/probes/probe_scope_replay.py --bundle v15_input --output analysis_v15/scope_fixed_control.json
```

## 3. 已修复的部分应完整认可

### 3.1 对象移除已经进入真实 list 结构和所有主要可执行载体

`remove_library_object_version` 在 `memory.py:1579`，现在处理 concepts（1733–1752）、abstracts（1754–1782）、definitions/bindings（1784–1808）、meta_priors（1810–1823）、草稿和覆盖记录（1825–1850），并刷新 definitions。不是上一轮仅对 dict 才执行而跳过实际 list 的代码。

独占子节点移除和具有存活父节点的共享子节点保留，在实际函数控制中均成立。可用第二来源的控制中：focal/exclusive 不再存在，shared 保留，parents 缩为 other，真实 `_calls_available` 返回 true。

`memory.py:1860–1873` 保留原始档案并去掉被删对象的预解 commitment、coefficients、radius；1876–1878 同时打开 family、routing、raw-history 重建。错误版本的请求在 1621–1622 返回不变库。branch-only 模式在 1685–1690 保留知识并开放重建。这些都应记作真实修复。

### 3.2 原始经历重建已经有执行和计费证据

`reconstruct_missing_objects_from_raw_history`（`branch_reconciliation.py:4122–4507`）读取历史 source_episode_batch，从 observations/transitions/schema 构造重建问题，执行普通审计、候选实例化和 premise revision，确认后 `library.learn`。这不是简单恢复已删除的对象副本。

当前交付回执 control 3：concept_0004 删除后从原始经历重建 concept_0005，compute=67053，确认误差约 `1.07609e-16`，新的概念、abstracts、definitions、bindings、meta_prior 可执行。应认可“保留同经历并允许计费重建”已经具备可用实现；下文身份错误不撤销这份功能证据。

### 3.3 scope_unidentified 不再自动变成全局 refuted

`replay_evidence_batches_onto_branches`（3888–4035）在实例失败后先做模板重绑定检查（3957–3959），预算不足明确延期（3960），只有严格模板域失败才构成模板不可行判据（3963–3967）。缩域失败时检查来源支持（4007）；仅无来源支持且模板严格不可行才全局 refute（4008–4010）。有支持但范围不明进入 template/instance scope unknown（4011–4018）。

下游 dispatch 已把这些状态加入 unknown 集合（5174–5180），无 scope_guard 的未知对象进入 unknown pool（5225–5233），没有继续沿用全局剔除。

同一个上一轮隔离控制，在 V15 得到：direct shrink=`scope_unidentified`；replayed old=`instance_scope_unknown`；template checker 被调用 1 次；来源支持仍保留。注意老探针 JSON 键 `old_source_support_preserved_but_globally_refuted` 名称过时，其布尔值只读取 independent_empirical_support；不得把这个旧键名称当成 V15 仍被全局反驳的证据。

### 3.4 共同转换器已接入生产，按历史时间推进并处理预算中断

`apply_reconciliation_step_from_evidence`（4038–4119）统一竞争检查、合并和 refinement。在线路径 6529–6538 调用它；普通重建在 4926–4935 调用它。在线同时保存 source_episode_batch（6568–6594）和具有 episode_seq 的判别实验批次（6487–6515）。普通路径按步骤计划重放（4806–4827），而不是把全部未来来源先做成一个初始通用库；失败的独立经历也进入步骤计划。

预算中断会回滚当前未完成步骤（4957–4962）、保存进度（4983–4992），并返回 partial_pending（5074–5093），没有将中断误写成全部重建成功；完整 memo 仅在未中断时形成（5058–5066）。

证据档案哈希已经补上 radius、coefficients、commitment、fresh rows 等 V14 缺项（index 文件 74–99、102–146）。控制确认 radius/commitment/observations 修改都失效。

## 4. 当前回执直接触发：重建的版本标识不一致

代码 `branch_reconciliation.py:4447` 先计算新 prior 的 semantic version，4448 才把 reconstruction_compute_cost 加入该 prior，4449 入库；事件 4468 使用早先 hash。`premise_commitment_index.py:52–66` 的 fingerprint 把所有非 `_cached_` 字段都作为内容，因此后加成本会改变版本。

不是仅理论控制：

`receipts/v15_true_intervention_and_strategy_20261118/deliverable_2_removal_replay_parity_and_exposed_diagnostics_20261118.json`

JSON 路径 `three_object_removal_controls_sec3_3.control_3_billed_raw_history_reconstruction` 中，同一个 concept_0005：

- event.reconstructed_semantic_version：`d16da439cede2073d663ffc6`
- inventory.semantic_version：`0220d69b8783ff749caf1a41`

影响：事件→对象版本→删除干预/消费者的精确关联不成立。预测模型可以依然正确，不能由此推出其重建损失或全部效用成绩错误。

最小修复：先填完当前 fingerprint 使用的字段，再计算一次版本；事件、inventory、调用和干预使用同一个值。若决定成本是非语义元数据，应通过明确字段 schema 排除，而不是只在事件端改 hash。

验收：完成一次现有 control 3，断言 emitted version == `compute_semantic_version(stored prior)` == inventory version；随后用事件的版本执行移除，确认去掉的正是该对象。

## 5. 受控可触发：档案回调把不匹配实验当成精确协议复用

`_archived_probe_lookup_fn` 在 `branch_reconciliation.py:4881–4924`：

1. requested_probe_id 相同即可匹配（4889–4891），未核对 sequence/reset。
2. 或 sequence 相同即可匹配（4892–4895），仍未核对 reset。
3. 都不相同，就取首个未使用批次（4896–4900）。
4. 三种路径均返回 `reused_exact_protocol`（4914）。
5. 4919–4922 还优先把当前请求的 frozen_program_versions 写作结果前版本，而非保留原档案的实验历史。

实际回调控制：请求 x=1、reset=True；档案仅有另一 probe 的 x=99、reset=False；仍返回 reused_exact_protocol，实际 rows 为 x=99，版本被标作 new_request_version。同 id 但 sequence/reset 不同也被接受。

这是重放边界的确定错误；**尚未证明冻结 E1/E2 中某次实验命中这种回退**。把历史另一批数据作为普通观察重新评估可能合法，但不能宣称它是刚才所请求实验的精确复用，更不能覆盖其历史版本身份。

最小修复：匹配键至少包含实际需要的 sequence、reset、protocol、schema/channel 和实验上下文；id 只作索引提示。无精确匹配返回 unavailable/pending，移除首个未用批次回退。历史证据重新评分另记事件，保留原观察历史与当前被评估版本两个字段。导出 match_mode 和 matched_evidence_id。

验收：同 id 不同 sequence、相同 sequence 不同 reset、完全无匹配三项都不返回 exact；精确匹配仍收费并消费一次。对现有 60 对回放仅重验记录中是否发生非 exact 取证，不需要因此重写整个实验架构。

## 6. 受控可触发：存在父节点不等于仍有独立有效支持

`memory.py:1693–1701` 中 concepts 排除 revoked，但 meta_priors 无 status 过滤。1715 的“valid”只检查父 cid 是否存在于该集合。1727–1729 只在全部 valid 父都被移除时删子节点。abstracts 的 valid_other_pars（1762–1764）甚至只检查名字不在 removed_cids，没有有效状态核验。inventory 的 independent_parent_roots（1535–1548）直接收集 parents 字符串。

实际控制：删除 focal，shared 的另一父 other 的 Concept 和 meta_prior 都设为 revoked；结果 shared 仍保留并可执行，inventory 把 other 报作 independent_parent_roots。它与“other 仍有效”控制得到相同保留结论。

更深的语义边界是 parents 当前也用作程序来源和调用依赖（memory.py:1292–1299），不天然等同于“每个 parent 都可独立证明 shared”。测试 `test_v15_true_removal_and_state_parity.py:125–143` 和交付脚本 759–768 的共享后代是手工造的 parents；手工 program 为 x+常数而传入目标为 x 的行，没有独立确认第二来源就可推出该后代。这些测试证明载体剪枝，不证明两个独立有效支持。

最小修复：先对当前支持根有效性作一致过滤；对声称“独立支持”的第二根附最小可重算 witness/confirmation，不能用 parent 非空替代。如果 parents 是合取依赖，则删除任何必要依赖应失效；只有明确记录的替代支持才可继续保留。无需引入通用证明图框架。

验收三例：有效独立第二根保留；第二根 revoked 或证据失效时不保留；仅有依赖关系、无独立确认时不得报告 independent support。应检查实际 recall/definition 调用，而不是只查一个列表。

## 7. “精确版本”当前仍受单 cid 单版本前提约束

inventory 的 ver_by_cid（1492–1500）和 by_object[cid]（1549–1552）会折叠同 cid 的多个版本。移除接口先检查第一个 matching prior/branch（1605–1618），通过后所有删除都用 removed_cids 集合（1702 起），不是 `(cid, version)`。

双版本控制中，请求删除 v1 后 v1/v2 priors 及两条 branch 都消失。**这是构造的多版本库存边界，不是当前 E1/E2 同 cid 并存版本的证据。**

最小选择：若当前产品就是每 cid 一份当前版本，明确这个 invariant 并在导出/删除时拒绝重复 cid；把接口说成“当前版本守卫下删除对象”。若要历史并存，则载体、清单、父引用和删除都必须用二元键。无需把现有单版本回执一概判错。

## 8. parity 的结果成立，措辞应限定观察字段

D1/D2 脚本 `run_v15_deliverables_1_2_errata_controls_and_replays.py:918–925` 的 source_end_state_exact_match 只比四项：status、scope_claim_level、是否有 scope_guard、是否有 routing_guard。没有比 guard 的具体内容、program/参数/证据身份。

24 对 dispatch 归一化在同脚本 130–143；新 36 对归一化在 `run_v15_prospective_e1_e2_studies.py:214–229`。它把 branch id 映射为 source cid 后**排序**选定列表，并附 scope_claim_level；排序会忽略候选顺序。该设计适合检查集合资格差异，不覆盖相同集合下顺序/预算/预测值的差异。

准确说法：V15 已消除此前在这 60 对检查中可观察的所选路由字段差异。不能因此断言“所有状态、所有实验协议、所有资源预算下在线和延迟语义完全一致”。

最小补强：保留现有报表，增加规范化 scope/routing guard 内容、对象和模板版本、给定 pilot 的预测/判断；需讨论调度公平性时保持候选顺序再比一次。不要把双方成本或全过程动态变化强制成一样，也不要因为未覆盖项否定现有 0 差异结果。

## 9. 次级边界：只需小修，不宜扩大成新项目

### feature_bounds 与缓存依赖

哈希 `premise_commitment_index.py:119–143` 未含 feature_bounds。它不是仅日志字段：重放 4861–4877 把它读入 TaskSpec，4929 传给共同转换器；判别实验构造 1058 读取 bounds，1122–1129 用其决定控制点中值和 clipping。因此运行依赖链成立；本次控制确认只改此字段 hash 不变，但没有证明冻结实验中发生了编辑后命中旧缓存。

最小修复是在实际哈希 canonical projection 加 feature_bounds，并同步语义字段声明。验收对该字段改变的 miss 和无关 wall time 改变的 hit 即够；无需扩造缓存框架。

### 无独立 fresh 行时的确认标签

原始重建 4395 在 fresh_rows 缺失时用训练 obs_rows 尾部；4398–4401 可由其确认，随后记录“fresh holdout confirmation”（4416）、independent_empirical_support（4435）、confirmed_exact（4436）。缺 fresh 数据时这个标签过强。

这里只作静态边界记录；没有证明本轮 67053 成本控制使用了 fallback。最小修复是缺独立 holdout 时保留未确认/待验证，或明确标为 in-sample check；不能把相同训练数据改名 fresh。验收一个无 holdout 输入不得产出 fresh confirmed。

## 10. 下一轮最小交付顺序

1. 修复已经出现在回执的版本计算顺序，并复用 control 3 验收身份闭合。
2. 收紧 archive probe matcher，记录精确匹配来源；对现有重放输出匹配统计，按实际受影响链决定是否重跑。
3. 修复 revoked 根有效性，加入一个真实独立支持保留例和一个失效例；把目前仅结构性控制改名准确。
4. 明确单 cid 单版本不变量或完成双版本键；同时补 feature_bounds 哈希和缺 holdout 的确认标签。
5. 保留 60 对已观察 parity 与本轮研究结果，不用上述边界修复冒充新的效用成功。机制修复后仍须由完整策略对比和未见问题结果判断维护是否值得。

