# Assumption Agent × Red Queen Gödel Machine：架构诊断与 Reconstruction V2 复核

> - 初版日期：2026-07-11
> - 本次复核：2026-07-27
> - 当前 BioASQ P1：独立 study `BIOASQ_P1_TYPED_QUESTION_EVIDENCE_EVALUATOR_L5_V1`
>   已完成公开 P0、source-free coordinate canary、legacy official HippoRAG closure、formal execution
>   freeze 与 311linux 唯一正式启动；311 未联网下载，所有模型均为既有 exact-hash 本地资产。正式
>   source 只打开、哈希和 decode 一次，四块 cohort 与 2,900-passage corpus 已密封；随后 GPU1
>   coordinate initial worker 与 GPU0 HippoRAG global build 各启动一次。17:11:47，最后一个登录
>   session 关闭且 `Linger=no`，用户级 systemd manager 整体触发 `exit.target`，以 SIGTERM 同时停止
>   parent formal service 和两个 child unit。中断前没有 coordinate/Hippo output、A_form action
>   archive、qrel release、evaluator、A_hold、score 或 M_search；没有 outer success/failure terminal，
>   restart/replay 均为 0。该 root 严格终止为 **post-source-selection infrastructure-invalid /
>   efficacy unknown / no replay**，不能在启用 linger 后重启同一 source/study/cohort
> - 当前 DSTC9 P1：全新 study `DSTC9_P1_HIERARCHICAL_KNOWLEDGE_EVALUATOR_L5_V1`
>   已由本机 WSL 对 official fixed commit 完成一次 content-addressed 下载和哈希校验，再一次同步到不通
>   外网的 311linux；311 未向 Hugging Face、GitHub 或其他外部 source 发请求。公开 P0 有效资格化
>   2,900 条 knowledge、TRAIN/VALIDATION eligibility 与 hotel/restaurant/taxi/train 四个 family；
>   v5 source-free 双 GPU canary 随后一次通过，GPU0 official HippoRAG build/reopen 与 GPU1 typed
>   coordinate lane 均真实执行，formal source/API/retry 都为 0。唯一 formal invocation
>   `c29ebcae…9b471` 在 `compile_formal_source_once` 失败：正式 typed venv 未包含 source compiler
>   已冻结使用的 `ijson`。失败发生在 bundle identity/topology 与 knowledge member open 之后、任何
>   action/model/qrel/score 前；safe terminal、空 action runtime 与 `NRestarts=0` 一致。由于 formal
>   source capability 已消费，不能在补装依赖后重跑同一 source/study/cohort；终态为
>   **formal-source runtime-dependency implementation/infrastructure-invalid / efficacy unknown / no replay**。
>   这不是 311 无外网导致的下载失败，也不是 Agent 相对 RAW/HippoRAG 的效果负结果；现实域四-family
>   双基线稳定优势与 A_hold 晋升后改善 untouched M_search 的 L5 仍都缺失
> - 当前 HiTab P1：全新 study `HITAB_P1_DMC1_HIERARCHICAL_SET_EVALUATOR_V1` 已在任何
>   HiTab source body、secret、cohort、model action 或 score 前完成 source custody、DMC1
>   hierarchical evidence-set evaluator、late-qrel controller 与三阶段 production closure。
>   首版 implementation inventory 在 source access=0 时发现外层 SentenceTransformers 5.5.1
>   会从共享 `/tmp` 生成并执行 Python，已作为 implementation-invalid 永久关闭；事前 v2
>   addendum 改用同一 content-addressed MiniLM 的 direct Transformers mean-pooling backend。v2
>   live probe 又发现 Hippo child 保留 cwd 作为 Python import root，仍在 canary/source 前关闭；
>   v3 的 pre-scrub 随后被 `litellm` 再次追加 cwd，亦在 canary/source 前关闭；最后的 v4
>   对整个 child 生命周期封存 `sys.path`。74/74 离线测试及独立终审通过；双 Python filesystem closure、
>   source-free canary
>   的真实 GPU1-overlap→GPU0-formation→cache-release→GPU0-Hippo 顺序、每 GPU 最多一个
>   Hippo process、canary/formal 一次性 claim、promotion 后才可首次 decode TEST，以及
>   `CPUQuota=800% / MemoryMax=40G / TasksMax=64` 均已闭合。双卡空闲后唯一 v4 canary
>   有效启动一次，但 planner/cross-encoder 构造后，direct MiniLM constructor 入口的首个
>   guard 检出 shared-temporary Python；失败早于 direct MiniLM 自己导入或加载 AutoModel，
>   未进入 public item execution，qualified receipt 为 0。HiTab 四个正式文件始终未下载，
>   source/cohort/Hippo action/qrel/score 均为 0；该 study 已按事前规则以
>   **source-free implementation-invalid / efficacy unknown / no replay** 终止，不能支持三臂优势或 L5
> - 最新 AVeriTeC P1 终态：独立 official source/cohort、v4 source-free 双 GPU canary、execution/launch freeze
>   与唯一 formal launch 均有效；systemd 为 success、`NRestarts=0`，全程 0 API/online evaluator、0 retry/replay。
>   A_hold 上 E1 相对 E0/RAW 为 36/36 utility tie、净 0、exact tail 1；虽有 19/36 项实际换 recipe 且改变 top-5，
>   仍未改变任何 qrel recall。E1 相对 official HippoRAG aggregate 净 `+3/2`、exact tail `3/32`，但 causal family
>   为净 0，且这份 aggregate 差不能归因于 typed action，因为 E1 与 RAW 的逐项 utility 完全相同。故 evaluator
>   不晋升、现实三-family双基线 primary=false，M_search 按冻结协议不执行，L5 未测。40 个远端私有 artifact、
>   safe terminal 与 aggregate 已独立离线复算一致；这是 **valid negative**，不是 implementation/infrastructure-invalid
> - 最新 EBM-NLP P1 终态：全新本地 source 上的 v4 source-free canary 已一次完成 2/2 CUDA worker，
>   0 API/network/retry；v1–v3 的 pre-model implementation failure 均未打开 source。GPU 空闲后 v4 formal
>   只启动一次，实际 service cgroup、网络隔离与冻结绑定全部通过；唯一 source epoch 随后在 tar header 资格检查中命中
>   `documents/` 命名空间的未覆盖成员。安全错误摘要精确匹配冻结代码中的
>   `document member does not match the exact frozen path pattern`，但不公开成员名。任何 member payload、
>   model inference、action、gold/label、score 与 online evaluator 均为 0；终态为
>   **source-header schema qualification implementation-invalid / efficacy unknown / replay=false**。这不是 Agent
>   相对 RAW/HippoRAG 的效果负结果；现实域三-family双基线稳定优势与 L5 仍同时缺失
> - 最新 EntailmentBank G1/E1 终态：official Task2 TRAIN/DEV aggregate qualification 通过后，fresh v2 secret 一次形成
>   G/A/F/A_hold/M=`60/36/30/30/30` 的三-family平衡 186-item cohort，F label 从未创建。v1 acquisition 因把同 ID
>   多行变体误当非法而在 selection/action 前 fail-closed；v2 只事前增加私有 source-line identity，未改任何 efficacy contract。
>   E0/E1 在 F 上分别冻结 Q0/Q1；A_hold 总 U 为 Q0/Q1/official HippoRAG/RAW=`96/92/90/19`。
>   Q1−Q0=`−4`、exact p=`57/64`，故 evaluator valid non-promotion，M 与 TEST 未开；Q1−Hippo 仅 `+2`、29/30 tie、
>   p=`1/2`。post-terminal Q0−Hippo 虽为 `+6`，p=`35/256` 且 family net=`0/+6/0`，仍不是跨 family 稳定优势。
>   该结果证明 Agent 明显优于 RAW、与 item-local official HippoRAG 近似持平，但 L5 仍未达到，同源不再补 gate 或换 evaluator
> - 最新 post-terminal synthetic multiseed 结果：全新 v3 `8×64=512` cohort 已由 v5 在单一 detached
>   formal attempt 中完成 RAW / official HippoRAG / Agent_R1 共 1,536 个 action；离线终态为 success。
>   Agent−HippoRAG 的 seed-level U 差为 `[3,8,0,2,1,5,6,2]`，总 U `+27`、7 positive / 1 tie；
>   但 Agent−RAW 总 U 为 `−14`，且 Agent 相对 HippoRAG 的全部 `+27 U` 只来自 definition-positive 两个 family，故只把
>   窄 synthetic mechanism stability 从 unknown 更新为 descriptively supported，不扩张为现实域、L4/L5
>   或 Agent 普遍优于 HippoRAG
> - 最新 FEVEROUS source-epoch v3 终态：原 source qualification 在设计冻结前已记录并规定整组排除 2 个
>   `NFD+casefold` 等价但 exact 不同的 title-context evidence sets；提交 `96234cf5` 将该 typed exclusion
>   接入 production adapter，`ac2919c0` 绑定 runner/resolver/formal-source/adapter/atomic-corpus/acquisition-core、
>   loader callable 与 Unicode DB，`c2c7efb8` 只修正 exact-subset 与 adapter pre-exclusion counter 的会计口径。
>   一次不足绑定的 nonqualifying prepass 在 TRAIN 读完、DB 哈希未完时中止，adapter=0；随后两次 pass 均完整
>   哈希 53.5 GB DB 并穷尽 TRAIN resolver/adapter，但都在形成 receipt 前被同一 aggregate topology 断言拒绝。
>   实际不一致字段/数值从未输出或落盘，v3 root/secret/HMAC/cohort/retrieval/action/score 均为 0。终态
>   `04ee399d…6b46` 禁止 FEVEROUS v3/v4 重跑；这不是 efficacy negative，Agent/RAW/HippoRAG 效果仍 unknown。
> - 代码审计基线 revision：`6224bb5a279f50fbcf1f8b36d19cb4ce6cc6c882`
> - 本次实现复核：receipt/runtime provenance 修复提交 `e43670f6`、`18ff3417`；v3.3 execution-policy 提交 `e0b1a33b`；v3.4 model-only/action-budget 主提交 `e491b0af`，runtime-path 修复 `995e6446`，Ruoli 503 分类修复 `ba0f36cf`，host-readable audit artifact 修复 `1df3092a` / `ad66d5a2`；v3.4 max2 v5 canary 已通过、fresh development 因四并发 429 fail closed；v3.5 将所有在线 phase 版本化为 1 worker，repair identity 修复 `96d53a5d`，malformed proposal/claim binding 修复 `d70562de`；v3.6 contrastive evidence / invalid-evidence lifecycle 实现提交 `01608e1e`；v3.7 六路首批 6/6 收到 429；v3.8 两路在 16 valid 后收到 2 个 503；v3.9 固定为 outer item workers=6 / shared model slot=1，并完成首个 clean full development 负结果；v3.10 exact-three/coverage-first fresh run 将 activation 提高到 2/16 但仍 0 gain，并暴露 semantic-diversity hard reject 与 action lowering 丢 target；v3.11 actionability fresh run 证明 treatment 已改变 trace、PDF 与成本但仍 0 gain，同时暴露 repair response shape 未被 generic system contract 可靠约束；v3.12 显式版本化 singular repair response并完成 56/56 clean development trials，但两代仍各仅激活 1/16、0 gain，无 incumbent；误入的空 freeze/partial controls 已隔离并补上 phase prerequisite；v3.13 complementary program-set 已完成 375/375 离线测试和 76/76 valid live development，三套 bundle 均激活 2/16、6 个 policy-on 全部失败、0 gain/0 harm、无 incumbent，同时暴露 G2 cross-arm raw replay 不一致；v3.14 提交 `2229d7af` 完成 411/411 离线测试及 62/62 attempted live development，selector 成功选中 7/7 三-family set、activation=3/16，但 7 个 policy-on 全失败；一条 recursive raw 超 64 MiB 使 primary non-claim，valid baseline replay=31、invalid key 又跨臂执行一次，两份 archive 仍无 incumbent；v3.15 action-quality / terminal-invalid provenance 实现提交 `696a2954`，453/453 离线测试通过，随后 clean lock、86/86 cache-only prewarm、smoke 与 57/57-valid live development 全部完成，但两臂仍 0 gain/0 harm、`incumbent_id=null`
> - 最新 proposal-only 复核：v3.16 family-slot formation 提交 `6ad5c156`；v3.17 artifact-blueprint formation 提交 `4f94e613`。两轮均失败且未启动 benchmark trial
> - 最新表示层进展：commit `b03c643a` 的 causal action-span extractor、closed typed operator/artifact graph 与 opaque recipe-only selection 已完成唯一一次正式离线 decision，9/9 preregistered predicates PASS，既有 report/event/lock 精确复验 PASS
> - 最新 production integration：提交 `ad6a8314` 首次把 proposer/evolution 接入 receipt-bound opaque `recipe_id`；live smoke 随后发现 ledger 绑定晚于 runner 构造的真实顺序错误。提交 `8caba466` 修复后，正式 integration v2 通过 13/13 predicates、12/12 tamper probes 与 exact replay，0 live model/backend/evaluator call；提交 `9b4623f9` 冻结该结果
> - 最新 live 结果：v3.18r1 以 38 个 item workers / 48 model slots 完成 38/38 TRAIN、16/16 logical paired validation 与两臂两代生命周期，报告均为 3/16 对 3/16、0 gain/0 harm、`incumbent_id=null`。事后非评分因果审计发现 `organize-messy-files-*` 镜像缺少应有的 100 个 PDF，且 stock 的 RAW/G1/G2 实际为 8/10、4/10、3/10 tests，却被二元 success 投影为三次 0；所以本轮只证明 production selection/provenance 的机械闭环，不能作为 clean typed-action utility negative
> - 最新 typed-portable integration：2026-07-14 的唯一一次正式非评分 integration 及其 exact replay 均通过，decision hash `a151ca52916101f0ea31b0d2f11c8fde8407f4410d175b1ac983e013d6e7957e`；真实 Docker canary、v3.20 production authorization loader 与 container cleanup 均通过，model/task-backend/evaluator call 均为 0。该结果只执行 agent-start 前的只读 artifact-evidence sidecar，不声称 write/render/move 等 recipe operator 已由 capability 执行；它允许建立 fresh v3.20 development root，但不是 incumbent 或 promotion
> - 最新 clean development 结果：v3.20 在 fresh root 上以 38 个 item workers / 48 model slots 完成 38/38 valid TRAIN、16/16 valid validation baseline 与 6 个 policy-on；61 次 attempt 中 60 次 valid，唯一 `codex_turn_failed` 以 same-request clean retry 恢复。两代均为 raw/candidate 5/16、activation 3/16、0 gain/0 harm、16 个 binary tie，两臂均 `incumbent_id=null`。非评分因果审计显示两代 recipe 与 action trace 确实不同，但 6 份 candidate trace 没有留下读取或消费 pre-agent sidecar 的可审计证据；因此断点是 capability evidence 未被下游动作消费，不是 selector、并发、budget 或新 gate 缺失。本轮已按预注册停止，不进入 freeze/controls/family-out/HippoRAG/sealed
> - 最新 runtime-delivery integration：2026-07-15 的唯一一次 TRAIN-only、非评分正式 decision 及其 exact no-model Docker replay 已通过 8/8 predicates 与 4/4 固定 tamper probes。三个 exact-image canary 以 3 路并发、`--network none` 完成 production TRAIN compile、canonical profile 生成/回读、真实 Codex run-template 注入与 effective-prompt shell readback；model、`run_task`、evaluator、verifier、score、promotion call 均为 0。该结果证明 profile 已进入 launch input，并不证明模型在认知上使用了 profile，也不产生 incumbent 或 task-utility claim
> - 最新 profile-consumption 诊断：2026-07-15 的预注册 consumed-development diagnostic 只新增 G1/G2 × 3 项共 6 个 policy-on trial，6 路同时启动、无 retry，6/6 valid 且 runtime receipt binding 全部通过；全程复用冻结 RAW 并使用离线 post-agent verifier。G1/G2 均为 0/3，prompt-delivery delta 与相对 RAW utility signal 均为 0/6。该结果 `fresh_validation=false`、`claim_eligible=false`，不创建 incumbent 或 promotion，也不触碰 test trial/sealed scoring；它终止的是“继续强化同一 profile 的 delivery”路线，下一步先改变 task-local execution contract 与 TRAIN-only action-utility 搜索对象，而不是立刻消耗新的 holdout 或增加 gate
> - 最新 execution-contract TRAIN ranking：14 个历史候选已编译为 6 个 typed programs，并在 38 个 TRAIN item 上形成完整的 14×38=532 outcome grid：56 个实际 policy-on、476 个冻结 RAW replay。Plus 首轮以 56 outer workers / 48 model slots 调度，实测 model 峰值并发 34，得到 55 个有效结果与 1 个 provider-capacity terminal；随后只用 Pro 重试该 1 路并得到 valid failure，没有重跑其余 55 路。最终 56/56 active outcome 有效，全部由本地离线 verifier 评价，online judge=0、validation/test=false。首位 `72c5ea9e…cd295` 为 1 recovery / 0 regression；被补跑的 `4033a94b…dedabf` 为 valid failure、0 recovery。该 ranking 未运行 promotion gate，也不授权 freeze 或 downstream
> - 最新泄漏审计与 targeted item-out 结果：上述 56 个 active outcome 的 route source 都包含被评价题本身，strict leave-item-out count=0；因此 `72c5…` 只是 in-sample TRAIN signal。事后看过 ranking 后，先对 recovery 所在的 `organize-messy-files-2` 做 bounded item-out refit/falsification，再以同一固定 workflow 并发补齐 organize-5/-6 两个 family fold；每 fold 都只用另外两题构图和派生 contract，固定 minimum support=2、maximum registered artifacts=6，不新增 gate。三路均 evaluation-valid、各带 37 个冻结 RAW replay，结果为 organize-2 false→true、organize-5 false→false、organize-6 false→false，即 1/3 recovery、0 regression；后两路分别使用 17/100 与 12/100 actions。全程 offline judge、validation/test=false，Codex/verifier 原始文件已持久化。由于整个 family audit 是看过 source ranking 后启动，`unbiased_crossfit=false`；且 1/3 不支持 family-wide transferable signal，因此该候选已被否决，不再补 gate，也不产生 freeze、promotion 或 incumbent
> - 最新 trace-refined organize 负结果：旧三折 trace 将失败分解为 destination nesting、无证据 fallback 与错误分类；在任何新 actual 前，提交 `baa3230a` 预注册 generic v2 contract 和三折 exact hashes，manifest=`da85625c…5483`。v2 只增加“从公开任务派生 destination”“每个 assignment 需正证据且禁用 catch-all”“重开 destination 并核对布局”三个 closed invariant，不含 verifier/test literal。三路 Plus 随后同时启动，3/3 evaluation-valid + 111 frozen-RAW replay，结果改为 organize-2 false→false、organize-5 false→false、organize-6 false→true，仍是 1/3 recovery、0 regression。org2 最终有 10 个 wrong-subject 文件，org5 只剩 `2409.15949v1.pdf` 被从 music_history 错放到 LLM，org6 为 6/6 verifier PASS。也就是说 refinement 改变了成功 fold，却没有增加 family recovery coverage；该分支已否决，停止继续改 prompt/补 gate，仍不产生 freeze、promotion 或 incumbent
> - 最新 typed-assignment organize 负结果：最终提交 `0eba5b7c` 预注册 content evidence→typed plan→post-agent harness apply/reconcile 三折 candidate；Plus canary 401/0 response 后在 task call 前固定 Pro，3 路 actual 最大并发完成，另 replay 111 个冻结 RAW。3/3 evaluation-valid，三路均对 103/103 文件完成证据、assignment、reopen 与 hash reconciliation；organize-2/-5/-6 仍为 false→false、false→false、false→true，即 1/3 recovery，低于事前 2/3。org2 错 1 个，org5 因 trial 内临时宽关键词分类器错 32 个，org6 6/6 PASS。执行 grammar 已闭合，但 semantic assignment operator 不稳定；该 representation 已按预注册停止，不 retry、不补 prompt gate、不进入 fresh development 或 downstream
> - 最新 frozen semantic financial candidate：提交 `c66e3a73` 将声明 TRAIN distribution 中的财务表格 evidence→question→typed answer operator、离线 DistilBERT QA/runtime assets 与 post-agent capability 固定；formation replay 为 candidate 3/3、历史 RAW 0/3，但明确是 `in_sample_formation_replay=true`、`cross_fit=false`、不授权因果或 prospective claim。提交 `2a8ade07` 随后把唯一 fresh treatment、recipe 与 split commitment 在读取 fresh outcome 前冻结
> - 最新 prospective paired 结果：fresh split `9698dd65…a9f3` 中 9 个 RAW 与 `financial-analysis-4` 的 1 个 candidate 共 10 次 Plus 模型调用最大并发启动，评价全为本地 offline verifier；app scheduler 在 agent 完成后丢失，但三次事前冻结的 recovery/continuation 只恢复 post-agent semantic stage 与 verifier，模型 replay=0、operator replay=0、verifier replay=0。最终 active pair 为 RAW=false、candidate=true，`candidate_minus_raw=+1`，两份 observation、typed operator receipt 与 CTRF 均有效；final report=`e6bc247e…d389`。这是协议定义下的 single-item treatment-associated gain；完整 9-item physical audit 因一条 inactive temperature RAW 的冻结 `codex_turn_failed` 标记仍为 invalid（trace 实为 reconnect 后 `turn.completed`，但未覆盖原 observation），所以 1/9→projected 2/9 只作描述，不作 promotion。无新 gate、无 online judge、无 official HippoRAG adapter、sealed 未访问，仍无 incumbent
> - 最新 SEC 13F period-out 多折复验：preregistration=`ab051822…454e6`、execution freeze=`20358ad1…9d3c`，8 个 measurement item 的 RAW/candidate 共 16 个物理调用按冻结上限同时启动，固定 Plus、同镜像/100-step budget、全本地 offline verifier、0 retry、0 online judge，4 个 sealed item 未访问。15 个 observation 完整有效；`financial-period-out-measurement-f1-r0` 的 RAW 虽收到一个 `turn.completed` 且 agent exit=0，却在退出审计中留下 1 个 residual process/TID，action-budget receipt 因而 fail closed。随后 recover-only 对 backend/model/operator/verifier 的调用均为 0，明确拒绝 replay。7 个完整 pair 上 candidate=2/7、RAW=1/7，均值差 `+1/7=+14.29pp`，candidate-only=1、RAW-only=0；但阳性只位于 fold 2，fold 0/1/3 的可比较差均为 0。把缺失 RAW 按最好/最坏情况界定后，完整 8-pair 差值只能落在 `[0,+12.5pp]`。因此本轮不是有效 primary positive，也没有证明多折稳定收益，不 promotion、不跑 controls/family-out/sealed。离线 failure attribution 又发现复用的 parent operator 把 TRAIN stock-class 尾部错误拼成一个 token，和 period-out 公开合同不一致；candidate 的 6 个首个可见失败正好是 2 个 stock-count scalar 与 4 个 increase-ranking。下一步是更换为 contract-derived typed SEC-13F operator 并使用全新 untouched measurement，不是继续补 gate。partial report=`d75d8d4f…ba7`
> - 最新 Replication C promotion：contract-derived SEC-13F typed operator 在 fresh development 上完成 8/8 valid pair，RAW 0/8、candidate 8/8、8 gain/0 harm，四个 fold 各为 +2；固定候选随后被正式 promotion，未再修改 candidate 或 gate
> - 最新 controls / family-out：operator-only 的 8 个输出逐字节匹配 Replication C candidate output，skill-only 因 host verifier leaf 权限与 `--cap-drop ALL` 的已定位 infrastructure defect 得到 7 个有效失败、1 个 unresolved；该批固定为 `executed_incomplete_no_retry`、不补跑，单独 disposition 认定 output-level operator sufficiency 满足 promotion condition。family-out 因没有同构 proxy/adapter 固定为 `not_applicable_scope_mismatch_no_proxy`，不产生 transfer claim
> - 最新 sealed 结果：4 个预提交 sealed item 的 RAW/candidate 共 8 个 Plus 调用以 8 路最大并发一次完成，离线 verifier、0 retry/replay/resample/provider switch/online judge；4/4 pair valid，RAW 0/4、candidate 4/4、4 gain/0 harm。4 个 candidate 均有 post-agent typed-operator evidence，8/8 容器均在 verifier materialize 前断网并在 verifier 后保持 network-none。严格盲化有两个必须披露的程序性事件：授权前曾对 private pack 做过一次只返回既有 digest 的 SHA stream；正式批次启动后，host `ps` 诊断又意外显示 sealed instruction text。两者均发生在 candidate/freeze 不可变之后，未暴露 gold/outcome、未触发 adaptation 或补跑；因此结果可作固定候选的 paired descriptive confirmation，但不能声称严格 blind holdout
> - 最新 MuSiQue generation-one 结果：提交 `aac4ceb3` 先冻结 official HippoRAG v2 filesystem attestation；`8702c369` / `97644e21` 随后在读取 outcome 前预注册并一次形成 official DEV 的 96-item、8×12 block HMAC custody；`fdef9d45` 只在 F1 的 12 项上形成 frozen typed program；`7b38a23d` 再冻结 M1 的 12×3=36 路 one-shot retrieval-only comparison，`4617a976` 只公开 aggregate result。M1 official support recall@5 为 RAW `7/29`、P `14/29`、official HippoRAG `14/29`；P−RAW 净增 7 个 support hit，逐题为 7 gain / 1 harm / 4 tie，按事前 strict-positive policy 产生 `promote_P_to_retained_generation_one` disposition。36/36 retrieval terminal 在评分前 join，0 Ruoli/external-network、0 study-level answer-generator、0 online evaluator、0 retry/replay/resample；official arm 内部保留冻结本地 LLM/OpenIE。postflight fresh filesystem attestation 通过。该 M1 阳性本身不覆盖 F1 cross-fit 稳定性（program 与 behavior 均为 false）、M2 retention、evaluator co-evolution 或 family-out；后续 family-out 见下。第三方 runtime 依赖仍只绑定 distribution metadata tree，而非完整第三方源码/行为。旧 6-item MuSiQue cohort 没有重放；当时只继续形成 Q/M2，没有在 M1 上补 gate
> - 最新 MuSiQue generation-two / evaluator 结果：Q 已在 F2 形成，但 one-shot M2 在授权消费、exact M2 打开及 36/36 unit 启动后，被当前 managed sandbox 禁止 bubblewrap 创建 `NETLINK_ROUTE` socket；24 个 terminal 完成、0 score，M2 以 infrastructure-invalid / efficacy-unknown 终态关闭且不重放。该 stderr digest 已由不含 benchmark/model 的最小 bwrap postdiagnostic 精确复现，P 的 M1 晋升不受影响。独立 L5 链随后完成：A_form 形成 `micro_worst_v1` challenger，但 incumbent/challenger 在 A_form 选择同一 program；A_hold 两者均为 `12/29`、0.9 Wilson 下界均为 `0.304297...`，challenger 不晋升；M3 active 仍为 incumbent，`18/29` 对 `18/29`、净收益 0。因此本轮没有证明 recursive retention，也没有发生 evaluator co-evolution；不再补 gate 或重跑已消费 block
> - 最新 MuSiQue→HotpotQA family-out：提交 `7306f333` 固定 acquisition/runner，`87fd35f7` 在与正式执行相同的 host 权限下验证 bwrap network namespace+writable-bind，`6852f15f` 在打开任何 row 前预注册，`bc6afb6b` 从固定 SHA 的 Hugging Face-hosted HotpotQA distractor-validation conversion 中以私有 HMAC 一次选取 12 项，`86242994` 冻结同一 RAW/P/official-HippoRAG 三臂后只执行一次，`d2981542` 只公开 aggregate result。frozen P 未适配或重形成；36/36 retrieval terminal 由单一 36-party barrier 释放，全部 join 与 fresh runtime postflight 后才离线评分。source-provided support recall@5 为 RAW `11/24`、P `21/24`、official HippoRAG `20/24`；P−RAW 为 +10 support hits / 7 gain / 0 harm / 5 tie，P−official 为 +1 / 4 gain / 2 harm / 6 tie。0 Ruoli/external-network、0 study-level answer-generator、0 online evaluator、0 retry/replay/resample；official arm 内部保留冻结本地 LLM/OpenIE。该结果支持 frozen P 的小样本 cross-family retrieval transfer，不声称 Parquet 与原始 CMU JSON 等价、answer generation、M2 recursive retention 或 L5 evaluator co-evolution
> - 最新 fresh Hotpot L4/L5：提交 `aedc0bd3` 固定六分区 acquisition、P/Q retained-recursion runner 与行为可识别的 evaluator protocol；`126ba352` 在 row access 前预注册 F_Q=36 及 M_L4/A_form/A_hold/F_search/M_search 各 24，`1f4f6b42` 一次取得 156 项且排除旧 Hotpot 12 项。纯合成文本的 24-worker official-HippoRAG capacity diagnostic 为 24/24 terminal + fresh postflight，未访问私有/评分数据。Q 只在 F_Q 形成；其四折 program/behavior 不稳定但不作为 gate。untouched M_L4 的 96/96 terminal + postflight 后，RAW/P/Q/P+Q/official support hit 为 `22/48`、`36/48`、`40/48`、`43/48`、`31/48`；P+Q−Q=+3、P+Q−P=+7、P+Q−official=+12，P support 只遗忘 1 个。故 L4 在这一 retrieval-only fresh cohort 的窄定义上达到，但不声称统计优越、family-out、等算力或端到端 QA。L5 的 A_form/F_search 形成 program 与 observed-action 均不同的 evaluator contrast；untouched A_hold 72/72 terminal 后 challenger `38/48`、incumbent `41/48`、净 −3，exact sign-flip `p=0.96875`，不晋升、不 invalidation，M_search 未授权且未打开。同一 anchor 不重试、不换候选、不补 gate；L5 仍未达到
> - 最新 final Hotpot portfolio acquisition 终止：`b504f8b3` 冻结 two-Q portfolio 机制，`6f373fce` 固定实现，`257d6283` 在任何新 source/private row 打开前预注册 continuation rank window `[156,324)`。第一次带 typo 的调用停在 one-shot marker 前，0 消费、0 private input；纠正后的正式调用先持久化 marker（semantic SHA `bcfed9d9…41f3`），随后才打开旧 12-item exclusion pack 与固定 source，并在内存中确定 168-item window。它在第一次 `os.mkdir(pack_root, 0o700)` 因父目录缺失抛出 `FileNotFoundError`，早于任何 block、locator 或 acquisition receipt 写入。结果为 0 block、0 locator、0 receipt、0 score、0 model/network/online-evaluator call；该 window 仍因 marker 与 post-marker selection 永久烧毁，不 retry/replay/resample，不再启动 Hotpot v4。此前 family-out 阳性、窄 L4 阳性与有效 L5 阴性均不受影响；新 portfolio efficacy 为 unknown。公开终态 semantic SHA 为 `b929ae19…199b`
> - 最新 MuSiQue residual portfolio（备选 A）终止：`6dd53a19` / `c7e20674` 固定 same-source residual two-Q portfolio design 与实现，`0271f9e5` 在 source row、旧 private row 均为 0 次读取时预注册 continuation `[96,264)`，`96d779ce` 一次形成 A_form/F_search/A_hold/M_search 六块共 168 项。A_form 与 F_search 各完成 4080/4080 local terminal，形成 behavior-distinct 的 incumbent/challenger action，并由 `1b9d53c9` 在 A_hold 前提交；`47faa049` 再冻结 48-item A_hold 的 288 路单屏障运行，冻结时 A_hold/M_search 均 0 row/label。正式运行先消费 authorization 并完整反序列化 A_hold 48 行与 labels，但 committed runner 的 terminal list comprehension 对 lazy submit generator 逐项立即调用 `future.result()`：只有第 1 个 work unit 被提交并进入 288-party barrier，其余 287 个尚未提交，180 秒后确定性 `BrokenBarrierError`。结果为 attempted=1、terminal=0、0 ranking/score/private evidence/report/model/network/online evaluator；这是 implementation-invalid，不是 efficacy negative。A_hold 永久烧毁，禁止 replay/retry/resample 或 same-source 新 cohort；无 promotion，M_search 仍未授权且未打开。MuSiQue portfolio efficacy 与 L5 仍为 unknown；公开终态 semantic SHA 为 `f1f51d93…3d2c`。严格终态提交后，`98763f27` 才把 formation/A_hold/M_search 统一为 eager bulk submit→join，并通过 focused 16/16 与 grouped 66/66 tests；该修复只供未来独立 study，不授权重放 A_hold
> - 最新 fresh 2Wiki fixed-action transfer：official archive、历史 1000-row denylist、train/dev/test collision scan 与 exact MuSiQue A/F action hashes 在 private selection 前固定；`3ac92a5d` 提交 corrected eager runner，唯一 public-synthetic diagnostic 完成 384/384 terminal 和两个 192-party barrier。正式 acquisition 一次形成按四 type 均衡的 A_hold=48、M_search=24；A_hold 384/384 terminal + fresh postflight 后离线得到 incumbent/challenger/P/official/RAW=`111/110/110/99/56`（support 总数 120）。唯一 promotion comparison 为 challenger−incumbent=−1、exact p=1，不晋升，M_search 未授权且未打开。预声明 non-gating comparison 中 incumbent−official=+12、16 gain/4 harm/28 tie、exact p=1549/262144；该强阳性只支持 fresh-item item-local retrieval transfer，不等同 official full-corpus 2Wiki、answer generation 或 L5 evaluator co-evolution
> - 最新 QASC evaluator direct-action 终态：known viewer disclosures 后轮换私有 secret，并在正式 row open 前固定 official archive/corpus、NLI runtime、16 个 equal-compute recipes、四块各 64 的 A/F/A_hold/M acquisition 与唯一 promotion policy。row-free probe 暴露 24 路 official HippoRAG 为 0/24、8 路为 8/8，故在 marker 前把最大稳定并发固定为 8；最终 synthetic diagnostic 的 RAW/P/official 全可用。一次性 acquisition 严格得到 TRAIN 7175、DEV 865，16 路两遍 BM25 扫描 16,987,130 行且 TEST 未重开。A/F formation 完成 2048 个 recipe action，并形成 behavior-distinct pair。untouched A_hold 上 incumbent/challenger support hit=`67/128`、`66/128`，总 U=`90/84`，challenger−incumbent=−6、exact p=`1668987/2097152≈0.795835`，不晋升；RAW/P/official HippoRAG support hit=`19/38/103`，official complete=`44/64`。因此 agent 不仅未改善 evaluator，也明显落后 item-local HippoRAG；epoch 不变，M_search 未授权、未打开，不在同源 QASC 上换 objective、补 gate 或重试
> - 最新 ContractNLI 独立法律域终态：在 commit `b018f948`，唯一 clean aggregate source-qualification 已因无 receipt 的隐藏 worker failure 严格终止；正式 marker 已消费，但 TRAIN 是否在失败前被程序性打开、精确失败原因、eligible capacity 与 source feasibility 均未知。selection/四块/action/RAW/P/official HippoRAG/evaluator/model/score 全为 0，故其 typed clause graph 与 evaluator efficacy 仍为 unknown，不构成性能负结果，也不允许同源重放
> - 最新 CUAD direct-acquisition 终态：为避开连续 clean-worker qualification，CUAD 改为 no-prequalification、parent-process direct one-shot。首个 CLI 在 marker、secret、archive/member 前因 design 缺顶层 schema 停止并由 `2cb8718a` 透明记账；唯一 marker-consuming attempt 随后只打开 TRAIN member 一次。commit `3e458d5f` 的公开 receipt 得到 407 个 contract components、232 个 eligible，低于冻结的 4×64=256；主要 aggregate exclusion reason 为 node cardinality 173，另有 exposure 2、gold cardinality 3。0 block/private/model/score，故这是 source-capacity terminal 而非 Agent 对 HippoRAG 的性能负结果；`1b9aaaa5` 已固定 no replay/no smaller block/no TEST/CUADv1
> - 最新 synthetic typed-graph causal 终态：原始设计 `d24dfb96` 后、任何 formal seed/cohort 之前，`b37054e2` 透明修正 TN2 语义、label-free evaluator derangement 与 sign-enumeration 的非随机化解释，并固定 acquisition/runner；41/41 focused tests、`py_compile` 与 diff check 通过。唯一 32-byte seed 一次形成 A_form/F_search/A_hold/M_search 各 64 项，F_search label 从未创建。formation 形成 behavior-distinct 的 real/permuted recipe；untouched A_hold 上 Agent full / official HippoRAG / RAW 的 total U=`168/164/158`、support hit=`108/106/101`（总 112）、complete=`60/58/57`。Agent−HippoRAG 的 matched net U=+4，但只有 2 个 nonzero pair，预注册 reference tail=`1/4=0.25>0.1`，故 valid non-promotion；M_search 未授权且未打开。drop-designated/wrong-type 各回落 4 U，endpoint-permuted 回落 6 U，增益只出现在 `MENTIONS_DEFINITION` positive family；real/permuted evaluator 在 A_hold 的净差为 0。终止后 `6f06464a` 已公开 exact seed 与 256-row cohort，不含 retrieval/model/score output。该结果支持这份 synthetic SCM 内很窄的 typed-action 因果效应，但不支持 evaluator co-evolution、family-out、L4/L5、现实总体效果或 Agent 普遍优于 HippoRAG
> - 最新 FEVER fixed-P 现实域 acquisition 终态：本地候选审计只发现 FEVER 具备继续价值；`543bed23` / `e5d5a7d7` 在内容读取前固定并下载官方 labelled `paper_test`、1.713 GB June-2017 wiki archive 与许可证，`e07cf640` 冻结一个无 promotion/gate 的 128-item gold-injected item-local reranking design，`83e185d7` 的 acquisition/runner 通过 13/13 synthetic tests 与独立审计。唯一 formal run 在 marker 后完整解析 paper_test 并在内存 HMAC 固定 64 SUPPORTS+64 REFUTES；随后在任何 wiki JSONL member content 打开前，central-directory contract 发现至少一个非目录 member 的 suffix 不是 `.jsonl`，以 `source_schema_invalid` 终止。action/label pack、RAW/P/Hippo、model/evaluator/score 均为 0；不事后查看 member 名、不改 allowlist、不重跑。FEVER transfer efficacy 仍 unknown，不是 Agent 的性能负结果
> - 历史 synthetic 8-seed v1 终态（已由 v3→v5 更新）：`dabcbde7` 在 seed 前固定 exact R1、RAW/official-HippoRAG/Agent 三臂、8 个 fresh seed cluster 与纯描述 estimand；两轮审计先修正 official paragraph title、1536-future submission barrier、未声明分析面、重复 grammar regeneration 及 success/failure terminal publication，64 项相关测试通过。`2ecf5ec8` / `5efbb5b1` / `f7d3335b` 依次提交 implementation freeze、8-seed custody 与 512-item acquisition。唯一 v1 runner 在 marker 后、任何 retrieval/action/label/score 前，把 512×(1 question+32 nodes)=16,896 条文本一次交给冻结 MiniLM encoder；该 runtime 的单次上限是 16,384，故以 implementation/infrastructure-invalid 严格终止。该 v1 cohort 的 stability 仍未知，`d185b84a` 只公开 seeds/cohort；它没有被修补后作为正式/评分 efficacy evidence 重放，后续 v2 只做过非评分 integration diagnostic。2026-07-18 的结论已由顶部最新 bullet 与 12.17 的全新 v3 cohort / v5 success 取代
> - 最新 BRIGHT P9 前瞻结果：在已消费 TRAIN45 上形成 relation/mechanism cross-encoder + RAW + HippoRAG 的固定 RRF 候选后，设计与实现均先于剩余 RESERVE 内容访问冻结；随后一次性测量每 family 11 项、共 33 项。P9 / RAW / HippoRAG mean nDCG@10=`0.12338/0.11431/0.09218`。P9−HippoRAG aggregate `+0.03120` 且三 family 全正；P9−RAW aggregate `+0.00907`，但 family integer delta=`−72,732,371 / 0 / +372,156,928`，因此事前固定的双 baseline、三 family 全正 primary 为 false。该 cohort 不调参、不补 gate、不重跑；剩余 4 项不作为 rescue cohort
> - 最新 BRIGHT P17 all-remote 终态：P14/P15/P16 均未产生 efficacy 后，P17 在 311linux 上完成 27/27 candidate-specific HippoRAG terminal，并按冻结顺序 seal 每族前 8 个 complete cases，共 24 个三臂 action；26 个 Qwen generation source-valid、1 个由冻结 totalizer 补全，外网与旧 P14/P15 action reuse 均为 0。但远端回执自报 HippoRAG 峰值进程并发为 9，超过 study design、runtime fingerprint 与 plan 共同冻结的 8；根因是 9-worker shared executor 在 cross-encoder 结束后把第九个 slot 交给 HippoRAG。该偏差在任何 gold/score 前由 archive audit 发现，正式 finalizer 未调用，gold/score 均为 0，P17 efficacy=unknown、同 candidate/cohort 永久不 replay。27-attempt forensic tree 已回传并校验；另透明保留 acquisition receipt 中不参与执行的 `target_terminal_count_per_family=10` 遗留字段，规范 target 始终为 8
> - 最新独立后续 study：TAT-QA P23 按预注册终止后，FRAMES P1 固定 official revision `58d9fb63…22ef`、Git blob `cea20270…025` 与 viewer-exposed rows `[0,100)` exclusion；实现提交 `6552fefb` 的 18/18 tests 与独立 adversarial audit 通过，freeze `8ee6662c…3f20` 绑定 real Git ancestor/四个 commit blobs，并显式披露 freeze 前一次未保存、未解析 row/cell 的 TSV byte stream。正式 source SHA-256=`4255093c…69ff`；唯一资格 marker 随后消费，但 raw TSV header 与预冻结的 public viewer conversion header 不同，故在首行、任何 row content/action/score 前 terminal。FRAMES 不改 parser、不重跑，efficacy/capacity 仍 unknown；这证明 viewer schema 不能代替 raw repository schema contract
> - 最新 FanOutQA P1 独立 study：固定官方 `v1.1.1` commit `ccf127bd…d54` 的 310-row DEV 与官方 1.539 GB revision cache；不透明下载虽因远端缺少 `git` 在收尾阶段退出，但两个完整 `.part` 已按冻结 size/SHA/Git-blob 校验并原地晋升，未重下。安全审计在任何 JSON/tar-member parse 前以不可变 amendment 透明记录 one-shot、cache trust anchor、qrel 隔离和 selection-commitment 加固；32/32 离线测试与最终审计通过。唯一 formal qualification 随后在 DEV item parse 中因官方 `categories` schema 与冻结 exact contract 不同而 `category schema drifted` fail-closed；cache tar member、TEST、candidate、RAW/HippoRAG、evaluator、score 均为 0。FanOutQA P1 不改 parser、不重跑，source capacity 与 efficacy 仍 unknown
> - 最新 MMQA P1 独立 study 终态（source 从未下载或解析）：固定 official MultiModalQA commit `4dd14328…02e3` 的 TRAIN、DEV、tables、texts 四个 gzip，共 69,204,571 bytes；候选、A/F/A_hold/M 与三臂离线评价均在 source 前冻结。311linux 驱动升级后已通过重启恢复为两张 RTX 2080 / `595.84`。第一次 source-free official preflight invocation 因 shell brace expansion 在 builder 前退出，单独 disposition 后，唯一 corrected capability launch 通过 address-family、filesystem 与 runtime inspection并进入 public synthetic official worker；worker exit 1，只留下冻结的 stderr digest，receipt 未生成。事后静态复核定位到确定性的首个 worker 内兼容冲突：两个冻结绝对模型路径被 pinned HippoRAG 转成一个 272-byte working-directory basename，超过该文件系统 `NAME_MAX=255`，且目录创建发生在模型构造、index 与 retrieve 之前；该结论来自 exact code/path-length，不冒充从单向 stderr digest 恢复出的异常文本。formal root/source/item/action/score 与 online evaluator 均为 0。该 study 因 source-free runtime infrastructure-invalid 严格终止，不重跑、不换 runtime/model、不下载 source；不是 Agent 对 RAW/HippoRAG 的效果负结果
> - 最新执行状态与缺口：MAUD extraction P2 的 corrected runtime fingerprint 与唯一 full source-free canary 均一次通过；canary 覆盖并发 MiniLM/CE、同一 22-query corpus、九个 typed recipes、E0 与 official HippoRAG，并由 `CPUQuota=400%`、`MemoryMax=40 GiB`、`TasksMax=64` cgroup 审计，0 API/network/retry。execution freeze 随后在 source access 前绑定 implementation、runtime、canary、fresh secret commitment 与 formal config。唯一下载 unit 以 `Restart=no`、1 CPU、1 GiB、16 tasks 发出冻结的三个 HTTPS GET：TRAIN/DEV 完整通过 size/Git-blob 校验，TEST stream 在约 300 秒后只得到 `403,264/6,169,945` bytes 并提前结束，触发 frozen size mismatch；service exit 1、restart=0。三个 JSON 均未 parse，formal controller/model/action/gold/score 与 online evaluator 均为 0。P2 已严格关闭为 **acquisition-infrastructure-invalid / efficacy unknown**，禁止 retry/resume/mirror switch、partial parse、formal launch 或在 successor 复用这些私有 bytes；因此现实域 Agent 对 RAW+official HippoRAG 的三-family稳定优势和 L5 仍都缺失
> - RQGM 版本：arXiv:2606.26294v2，2026-06-29
> - legacy 代码范围：`assumption_os/`；legacy 报告范围：`reconstruction/md/` 与对应 artifacts
> - v2 范围：`reconstruction_v2/`

本文从已故障的旧任务“继续调试 assumption”的本地完整记录中恢复了 2026-07-10
的 Red Queen 原始诊断，并用当前代码、测试、实验报告和本地论文重新核验。旧任务
中的凭证、网络连接参数和与架构无关的敏感内容没有复制到本文。

本文同时保留两个时间切片：

1. **legacy 诊断**：解释旧 HLE 系统为什么不是由假设学习闭环驱动；
2. **v2 复核**：判断上述缺口哪些已经修复，哪些只是有了接口或测试，哪些仍会
   阻断论文级结论。

除非明确写成当前状态，legacy 的实验数字和行号只描述诊断时的旧实现，不代表
`reconstruction_v2` 的最新性能。

## 一、执行摘要

### 1.1 最准确的当前结论

旧诊断的主结论仍成立：legacy HLE 把最有研究价值的“假设学习”放在了旁路，
真正控制答案的是一个由 prompt、检索、手写规则、verifier、fallback 和 selector
组成的高维控制面。`trace -> transition -> miner` 虽然存在，却没有可靠地改变下一题
的 runtime，也没有用 policy-off/on 反事实估计因果收益。

但这句话不能原样套到 v2。`reconstruction_v2` 已经在**接口和实验 harness 层**接通：

- 三类结构化 `HypothesisProgram`；
- 内部 effectful runtime；
- 递归 proposal repair；
- paired policy-off/on evaluation；
- train/validation/sealed-test guard；
- archive node、evaluator epoch 和 selective invalidation 骨架；
- SkillLearnBench instance-out/family-out 协议。

因此，当前诊断应更新为：

> **学习闭环在 harness 层已接通；promotion 所有权、外部 backend action/fallback
> 边界和 86-item 离线可运行协议已经闭合。contrastive trigger learning 已在 v3.6 的代码、
> manifest、离线测试和 live train 中运行到真实 paired validation，但串行轮只完成 2/16 pairs
> 后因吞吐主动终止。v3.7 把跨题 worker 从 1 改为 6，但首批 6/6 请求均收到 429，
> 熔断后其余 30 条本地跳过。v3.8 两路完成 16 valid 后又同时收到 2 个 503，熔断跳过 20。
> v3.9 的 6 路题级 pipeline / 1 个在线 agent slot 已 clean 完成，但两代 candidate 均只激活
> 1/16 validation、0 gain/0 harm。v3.10 fresh root 随后完成 38/38 valid train、16/16 valid pairs、
> 56/56 actual trials，0 provider/infra/mismatch；exact-three coverage-first 候选覆盖 2 个 train family、
> 6/6 failure precision，并把 validation activation 提到 2/16，但 candidate/raw 仍同为 3/16、
> 0 gain/0 harm。第二代两次 exact-three response 都因三项 activation signature 坍缩为同一组而被
> 旧合同 terminal reject，两臂 report 因 proposal failure non-claim，archive 仍无 incumbent。
> 离线轨迹复核证明 routing/treatment 实际执行并改变了命令和答案；`selection_change_count=0` 只因
> backend 把 answer 投影为 success 布尔。真正 blocker 是 proposer feedback 硬编码 completion check，
> 以及 lowering v1 在非空 value 时丢掉 `execute_step`/`check_condition` target、把 action 降为含糊
> JSON mode blob。v3.11 因此只修 gate 前的 actionable directive、lowering 与 diversity audit；
> fresh root 的 38/38 train 全部 valid（5 success / 33 residual），exact-three 返回三种 distinct
> activation signature，两个 root 静态通过。入选 court policy 在唯一激活题上真实执行：raw/candidate
> action starts 为 66/16，输出 PDF 内容不同，但二者都失败；no-rec 汇总仍为 4/16 对 4/16、0 gain/
> 0 harm。该 arm 又因一个未激活的 raw poster trial 超过冻结 64 MiB 而 non-claim。recursive arm
> 则在 validation 前失败：repair transport/JSON 成功，却返回 batch 字段 `hypotheses`，而 repair parser
> 需要 singular `hypothesis`。事后调用链复核纠正了最初判断：真实 repair payload 并不含
> `proposal_batch_contract`，只是 generic system contract 没有 versioned repair-specific singular
> override，模型仍复用了 root response shape。v3.12 因而保留 train coverage objective，并新增
> top-level one-object/`hypothesis` response contract；proposer 只作防御性 batch-contract 清理。exact-three root、
> train selection 和所有 evaluator/promotion/split/fuse/retry/sealed 合同不变。fresh root 已完成
> 56/56 valid external trials：38 train 为 8 success / 30 residual，0 provider/infra/action/network/
> mismatch；两代 exact-three 均 3/3 static pass、三种 distinct signature，但每个 root 都只覆盖
> 一个 train family。coverage-first 连续选择 court root，held-out 都仅激活 `court-form-filling-5`
> 这 1/16，raw/candidate 均为 4/16、0 gain/0 harm，action starts 从 raw 9 变为 candidate 32/43。
> 两臂以 consecutive non-promotion 停止，archive 字节相同且 `incumbent_id=null`。由于 12 个
> static node 全部通过，本轮 repair request=0；所以 singular scope 未破坏 full development，但
> 尚不能声称 full-run repair path 已验证。随后旧 runner 错把空 archive 写成 frozen receipt 并误入
> controls；该批仅 2/96 record 且全部 interruption-invalid，现已 quarantine，禁止拼接或作 claim，
> sealed/test 未访问。runner/freeze/control consumer 已统一要求真实 recursive incumbent；这是阶段
> 前置条件，不是新增评分 gate。
> v3.13 随后把互补 root 组合成 program set，并在隔离第一次 partial development 后保留同一冻结
> lock、86/86 prewarm 与 smoke，重新执行正式 development event/work tree。正式轮为 76/76 valid
> attempt-one trials：38 train-off、32 validation-off、6 validation-on，0 provider/infra/action/network/
> mismatch；最大 69/100 actions、62,200,000/67,108,864 network bytes。三套实际 bundle 都由 poster
> 与 court 两个 TRAIN-perfect、0 success-FP 成员组成，只激活 `anthropic-poster-design-2` 与
> `court-form-filling-5`。六个 policy-on 与对应 raw 全部失败，四个 generation decision 因而都是
> 2/16 activation、0 gain/0 harm；两臂均 consecutive non-promotion、`incumbent_id=null`。recursive
> repair 已真实执行且无 response/model failure。证据完整不等于性能提升，因此不 freeze、不跑
> controls/family-out/HippoRAG/sealed。另一个机制缺口是 recursive G2 复用 G1 的 16 条 raw，而
> no-recursive G2 重跑 16 条 raw并得到 4/16（前者 2/16）；两臂内部 pair 有效，但该差异不能纯归因
> recursion。v3.14 随后以提交 `2229d7af` 完成两项有限修订和 411/411 离线测试，再通过新的
> claim-eligible lock、86/86 cache-only prewarm、Plus canary 与 clean smoke。正式 development 完成
> 62/62 attempted trials：38 条 train 全 valid（7 success / 31 residual），0 provider/model/slot/action/
> mismatch。新版 selector 在 G1 按冻结 TRAIN objective 真正选中 7/7 failure support、3 families、
> 0/7 success-FP 的三成员 set，并把 held-out activation 提到 3/16；但 recursive/no-recursive G1 的
> 6 个 on 与 no-recursive G2 的 1 个 on 全部失败，所有可比 pair 都是 0 gain/0 harm。recursive G1
> 一条 court policy-off 使用 68,660,000/67,108,864 bytes，硬 fuse 正确抑制同 request retry，primary
> report 因而 non-claim 并停止；no-recursive 机械上 claim-eligible、两代 non-promotion。valid baseline
> cohort 产生 31 次零执行 replay；但 invalid 不入 evidence cache，导致相同 baseline replay key 在
> no-recursive 又执行一次并得到 valid row，所以该单题不能作严格 cross-arm recursion attribution。
> 两份 archive 都是 `incumbent_id=null`，没有 freeze/controls/family-out/HippoRAG/sealed/test。
> 该结果兑现了预先约定的停止条件：不再迭代 selector。下一问题是 action quality——三条 G1 directive
> 分别缺少实际 HEX、可用的离线漏洞数据源或新的表单操作，基本只是重述 task instruction；不能再靠
> trigger coverage 或 promotion gate 修补。Plus/Pro 都是同一 `gpt-5.4-mini` route，本轮 Plus 全程可用。
> v3.15 已在提交 `696a2954` 把该诊断落实为一个有界的 TRAIN-only action-quality 合同，并通过
> 453/453 离线测试：instruction 明确只是 baseline requirement，候选应补充 exact constant/mapping、
> concrete local tool command 或 artifact-internal manipulation 中至少一种 material delta；proposal 只接收
> 经过 allowlist、containment 与敏感信息过滤的 TRAIN public-environment facts 和 policy-off action-trace facts，
> 不读取 validation outcome、test、solution 或 verifier，也不给 proposal 外部工具、网络或运行时安装权限。
> `proposal_action_delta_audited` 只记录 material-delta/restatement 风险诊断，不拒绝 response、不 retry、
> 不触发 repair、不重排候选、不改变 promotion gate。相同 baseline request 在声明的 same-request retry
> 完成后仍 invalid 时，v3 replay policy 只写 run-scoped terminal tombstone；后续 arm/generation 零执行复用
> 同一 invalid，且明确 `promotion_evidence=false`。首代 checkpoint、action-profile count/set hash 同时进入
> 两臂 report，并由 freeze 逐臂及跨臂核验。正式 v3.15 root 随后通过 clean lock、86/86 cache-only
> prewarm 与 smoke，并完成 57/57 valid actual trials：38 TRAIN policy-off、16 shared validation baseline、
> 3 activated policy-on；8/8 proposal/repair model calls 完成，TRAIN 为 6 success / 32 residual，在线 agent
> 最大并发严格为 1，provider/infra/action-budget/network-cap/pair-mismatch 错误均为 0。recursive G1/G2
> 都只激活 1/16，candidate/raw 均为 4/16、0 gain/0 harm；no-recursive G1 static reject，G2 也只有
> 1/16 activation、0 gain/0 harm。共享 cohort 产生 32 次 zero-execution baseline replay。两臂虽均
> claim-eligible，但这里只表示 clean negative result 可用；它们都以 `consecutive_non_promotion_limit` 结束，
> archive 均为 `incumbent_id=null`，sealed/test=false，未进入任何 downstream phase。13 个 candidate audit
> （9 roots + 4 repairs）中 7 个有 material delta、6 个有 restatement risk，但所有 material delta 都仅是
> `exact_constant_or_mapping`，没有 concrete local tool、artifact manipulation 或 environment primitive；
> 9 个 root 又全部坍缩到 `anthropic-poster`，搜索从 v3.14 G1 的 3-family/7-support 退回单-family/2-support。
> v3.16/v3.17 随后没有直接重跑 development，而是复用冻结的 v3.15 TRAIN receipt 做 proposal-only
> feasibility：38 observations、6 success controls、32 failures、31 profiles、0 source-agent re-execution。
> v3.16 三个 logical call 全部成功，但 9 项标准失败 6 项。v3.17 固定 exact family trigger、空 anti-trigger、
> deterministic reusable artifact 与 read→parse→update→serialize→write-back blueprint 后，distinct single-family
> signature、support 2/2/3、3/3 concrete local tool、2/3 artifact manipulation、0 restatement/self-block 等 8/9
> 均通过；第三个 action 仍绑定两个来自失败 TRAIN command 的 primitive，故唯一剩余项
> `failed_profile_primitive_avoidance_passed=false`，整体仍 fail。一次 `RemoteDisconnected` 在同 request retry
> 后恢复，3 个 logical call 均完成，因此不是 credential/provider-capacity 结论。两轮 backend/evaluator/
> validation/test/verifier/sealed access 全为 0，没有 benchmark trial、promotion 或 archive。该 free-text
> family-slot 路线到此停止，不再做 v3.18 prompt/gate/acceptance patch。
> 仍未成立的是 clean development promotion、
> 跨 family 泛化，以及 Red Queen 式多谱系搜索和
> evaluator co-evolution。v3.3 已把 low reasoning/verbosity、32,768-token
> `body_after_prefix` compaction、10,000-token tool-output limit 和 request-compression
> 变成 protocol-owned treatment；`video-object-counting-1` 从 v3.2 的 71.1 MB hard-cap
> failure 降为 19.69 MB valid failure，full train 的最大流量为 40.6 MB，38/38 均未触发
> cap/provider error，说明本次 batch 未再被 fuse 直接阻塞；但 canary/full 的 1.47/19.69 MB
> 波动也说明跨运行稳定性尚未建立。full train 仍只有 37 valid、
> 9 success、1 invalid：`offer-letter-generator-1` 的真实 Codex JSONL 返回了一次
> `web_search` item，违反冻结的 model-only contract。因 `all_valid_before_proposal_v1`，
> proposal/validation 仍为 0。现已定位根因：Codex 0.144.1 会把兼容键
> `tools.web_search=false` 解析后丢弃，未设置顶层 `web_search` 时默认仍为 `cached`；因此
> v3.3 的 38 个请求都暴露 hosted web search，仅一条实际调用。v3.4 改用权威顶层
> `web_search="disabled"`。零模型 loopback 捕获证明，canonical 请求有 7 个本地工具且
> 0 个 `web_search*`，同配置仅换回旧布尔键的阳性对照有 8 个工具并明确包含
> `web_search(external_web_access=true)`。`max_steps=100` 也已定义为可观测的
> `codex_action_start_v1`：每个 `item.started` 都占一单位，由容器内 supervisor
> 在第 100 个 start 终止，并按 task/TID 清理专用 trial 容器基线后新增的所有 live task；它不是 semantic turn。异常退出、畸形
> start、残留 descendant 和 receipt/trace 不一致均 fail closed。v3.4 clean lock、共享 runtime
> 与 v4 86/86 prewarm 已通过；PATH 作用域和 root-owned `0600` audit artifact 两个本地问题
> 已分别修复。max2 v5 随后完成真实模型推理、2-step 受控截断和本地 verifier：action/tool/
> process/receipt 均 valid，0 web/remote tool。fresh development 因而获准从零启动；38 个 train
> request slot 中 17 条形成有效离线评价（3 success），4 条收到 `provider_rate_limit`，熔断后
> 17 条在本地跳过。没有 cap、action、tool 或 verifier violation，但 all-valid-before-proposal
> 正确阻止 proposal/validation/report/archive。由此可知 API/单次 route 可用，尚未满足的是冻结
> 四并发的持续容量。当时的 v3.5 因而只把五个在线 phase 的 worker 从 4 改为 1，其余合同
> 不变。首轮 serial run 取得 38/38 valid train、9 success、0 provider/cap/tool/action/verifier
> invalid，随后在静态递归验证中因两个不同 repair payload 复用同一 model-declared ID 而
> fail closed；没有 validation-split trial、promotion、report/archive 或 sealed。修复改由父分支、
> depth 与规范化内容派生确定性 repair ID，并保留 archive 冲突硬拒绝。修复后 fresh root 再次
> 得到 38/38 valid、9 success、0 provider/cap/action/tool/verifier invalid；root proposal 返回
> 3 个候选，但第一个 repair 的 transport/JSON 成功后没有 mapping-valued `hypothesis`，裸
> `ValueError` 穿透旧异常边界，仍在 validation trial 前退出。现已把 malformed root/repair
> envelope 与 canonical parse 纳入既有 typed failure isolation，并让 report/freeze 从 generation
> rows 绑定非 claim 状态；没有新增评分 gate、响应 retry 或在线 evaluator。两次 38 条都不能跨进程复用。其后第三个 v3.5 fresh root 首次完成 38/38 train、proposal、真实 repair、双臂 paired validation、两代 lifecycle 与四份 report/archive；两臂均未 promotion，第一代 recursive 为 0 gain/2 harm，no-recursive 为 1 gain/0 harm但 LCB 仍小于 0。第二代 no-recursive 又被一次 Ruoli 503、circuit skip、9 个 invalid pair 与 8 个 budget mismatch 污染，旧 lifecycle 错把它计作普通 non-promotion。v3.6 已一次性把成功 train rows 变成无 instruction/context 的 negative controls，按 train activation precision / false positives / failure support / complexity 选择候选，并把 invalid counterfactual evidence 归类为 terminal non-claim。其串行 live root 完成 38/38 valid train（7 success/31 residual），选出的 root 为 26/27 train activation precision、1 个 success false positive，并完成 2/16 validation pairs 后主动停止；无 report/archive/promotion/freeze。v3.7 的六路 agent 首批 6/6 收到 429；v3.8 的两路 agent 在 16 valid 后同时收到 2 个 503。两轮均被既有 circuit 正确阻止、无可复用 bundle/report。v3.9 不改 evaluator、learning/promotion、model、split、预算、retry 或 circuit；只把调度改为 6 个 outer item pipeline 共用 1 个在线 agent slot，本地容器准备和离线 verifier 仍可并行，同题 variants 仍串行。该调度现已 clean 完成 full development，负结果把 blocker 定位到 proposal diversity 与 prospective coverage，而不是继续补 gate；旧 rows 不复用。**

2026-07-14 的 typed-selection integration 与 v3.18r1 又把结论向前推进了一层，但也纠正了
“clean negative”的含义：

- production proposer 现在只能选择已注册 opaque `recipe_id`，snapshot/catalog commitment、跨代
  exclusion、recursive repair、archive round-trip、compiler/runtime installation receipt 已贯穿真实路径；
  formal integration v2 为 13/13 predicates、12/12 tamper probes、exact replay PASS。
- fresh v3.18r1 以最大并行完成 38/38 valid TRAIN、16/16 runner-valid logical pairs、58 个实际
  task trials 和 12/12 model attempts；两臂两代都只激活 `organize-messy-files-3` 与
  `stock-data-visualization-3`，二元结果均为 raw/candidate 3/16、0 gain/0 harm，且无 incumbent。
- 这 0/0 不能解释为行为相同。`organize-messy-files-3` 的缓存镜像只有 3 个 Office 文件，缺 verifier
  要求的 100 个 PDF；三路在不可完成的输入上得到相同 4/6 tests。stock 输入完整，但 RAW/G1/G2
  分别为 8/10、4/10、3/10 tests；全或无 `task_success` 把明显退化压成三个 0，报告遂记为 tie。
- 两个 candidate 的 `fine_grained_action_receipt_available=false`；轨迹证明 skill 已路由和安装，却没有
  证明 agent 读取或执行了 recipe。recipe 还把 TRAIN locator（例如 `.csv` 与单一 Office 文件绝对路径）
  带到 validation，而当前 item 实际是 `.tsv` 或另一个目录；G2 也只是排除 G1 recipe 后枚举新 root，
  没有非空 `parent_recipe_id` / `failed_checks` 来吸收上一代 residual。

因此 v3.18r1 的 `performance_claim_eligible=true` 只表示冻结协议下的运行记录机械有效，不能提升为
benchmark construct-valid 的 action-utility 结论。当前最短路径不是补 promotion gate，而是先恢复
离线 task input/dependency closure，再把 literal locator 改成运行时 artifact role，并让至少一个受限
capability 由 harness 实际执行、产出 effect receipt；TRAIN/search 可保留离线 per-test residual vector，
最终 promotion 的 task-success gate 不变。

这条有界路径现已完成到 fresh-development authorization：v3.19 已冻结 task input/dependency closure；
随后 typed-portable formal integration 只用 stored TRAIN evidence 与 public validation input，完成一次
正式运行及 exact replay。三项真实 Docker canary 在 `--network none` 下由 production pre-agent hook
生成并回读只读 artifact-evidence sidecar，production loader、exact image、cleanup 均通过；model、
task backend、evaluator、verifier、score 与 promotion call 均为 0。decision hash 为
`a151ca52916101f0ea31b0d2f11c8fde8407f4410d175b1ac983e013d6e7957e`。这里的 effect 仅指输入
evidence profile/inventory sidecar，不是 task effect：write、render、move 及其他 recipe operators 仍由
agent plan 承担，不能写成“完整 typed capability 已执行”。该 PASS 允许另行冻结并建立 fresh v3.20
development root；它本身没有 benchmark score、candidate gain、incumbent 或 promotion。

fresh v3.20 已按该授权完成，并给出 clean 但为负的 development 结果。cache-only
prewarm 为 86/86，正式轮为 38/38 valid TRAIN、16/16 valid shared validation baseline 与
6 个 valid policy-on；60/60 valid trials 的 action-budget 与本地 offline verifier receipt 均有效，
最大为 73/100 actions，无 web/remote tool/runtime install、budget truncation 或 sealed/test 访问。
唯一 `codex_turn_failed` 在相同 request 上 clean retry 成功，因此 61 attempts 最终得到 60 valid trials。

两代均以 3/16 activation 得到 baseline/candidate 5/16、0 gain、0 harm 和 16 个 binary tie，
且均因 `insufficient_net_gain_count` 被拒绝。第二代选中了不同 recipe，compile manifest、
treatment hash、observation hash、verifier 子测试与 action cost 也确实改变，所以
`selection_change_count=0` 不能解释为行为不变：该计数虽实现为 `selected_result.answer`
的 hash 比较，但 SkillLearn adapter 把 `answer` 设为 `observation.success`，因而在此路径上只能看到
二元 task-success 改变，不包含文件树、最终消息或 action trace。

| 激活 family | RAW 子测试 | G1 子测试 | G2 子测试 | RAW/G1/G2 actions | 二元 success |
|---|---:|---:|---:|---:|---:|
| stock | 6/10 | 8/10 | 8/10 | 20/23/28 | 0/0/0 |
| organize | 5/6 | 3/6 | 4/6 | 21/14/14 | 0/0/0 |
| temperature | 6/7 | 5/7 | 5/7 | 73/54/26 | 0/0/0 |

这张表说明机制已改变真实执行，但没有形成可晋级净收益：stock 只得到 partial-test 改善，organize 与
temperature 反而退化。更关键的是，6 份 candidate `codex.txt` 都没有留下显式读取或消费已生成
`.assumption-v2/capabilities/portable-*.json`，对应 pair 也全部
`fine_grained_action_receipt_available=false`。因此当前断点是“sidecar 已生成但未被任务动作消费”；继续搜索
同类 read-only-sidecar + prompt-directive recipe，或给 promotion 再加 gate，都不能修复这个因果缺口。
recursive/no-recursive archive 因 `repaired_candidate_count=0` 而字节相同，故本轮也不支持 recursion benefit。
primary validation 已被本轮消费，不能再据这些子测试调 recipe 后复用为 claim evidence；没有 incumbent，
所以 freeze、controls、family-out、HippoRAG transfer 与 sealed test 均正确停止。

2026-07-16 的 frozen financial semantic treatment 第一次改变了上述“始终没有二元 gain”的边界。
候选先在三条声明 TRAIN formation item 上完成 3/3 replay，但该结果被显式限制为 in-sample；随后在任何
fresh outcome 前固定新的 provenance split、唯一 active item `financial-analysis-4`、candidate/recipe/
treatment、Plus route、同镜像/预算和本地离线 verifier。正式 batch 把 9 个 RAW 与 1 个 candidate 共
10 次模型调用同时调度；尽管父 scheduler 在 agent 完成后丢失，预注册恢复没有重放任何模型调用，只在
原容器上完成一次 candidate semantic stage 和并发 verifier。active pair 最终为 RAW=false、
candidate=true，即 `+1`，且两臂 observation、action-budget、typed operator 与 CTRF evidence 均有效。

这项结果支持一个更窄但此前缺失的命题：**包含固定 semantic operator 的预注册 treatment 在一个未用于
候选形成的 fresh item 上出现 task-success unit-level gain**。它仍不支持“稳定优于 RAW”：样本只有一个 active pair，原
runner 不是 pristine completion，完整 9-item physical cohort 又保留一条 inactive transport-audit invalid；
1/9 RAW→2/9 projected candidate 只能描述，不能用于 promotion。`financial-analysis-4` 至此已消费，
不得据其 trace 调候选后再次当作 fresh；`financial-analysis-6` 仍 sealed，且没有 incumbent，所以
freeze、controls、family-out、HippoRAG transfer 与 sealed test 仍不启动。

随后对同一冻结 candidate 进行了真正的 period/source-out 多折复验。两份官方 SEC 13F archive 在正式执行前
下载并做双 oracle 一致性检查；8 个 measurement item 按 4 fold × 2 replicate 固定，另有 4 个 sealed
commitment 不进入 materialized benchmark。执行计划没有 HippoRAG projection：项目没有同构 official adapter，
所以只运行 RAW/candidate 的 `2×8=16` 个物理调用，并把 16 worker/16 inference slot 一次性全部放开。结果
不是“跑得慢所以串行化”，而是约 7 分钟完成模型阶段；唯一缺口来自退出审计 fail closed，而非吞吐或 provider。

16 路都有 model-execution claim，15 路完整进入 observation；缺失的是 fold 1 replicate 0 的 RAW。该路 action
trace 本身有且仅有一个 `turn.completed`、agent exit=0、token usage 完整，但 dedicated container 扫描仍发现
1 个 residual process/TID，且对它完成一次 SIGKILL。冻结的 action-budget contract 要求非 budget-triggered
退出时 residual=0，因此不能把它事后接受为 valid。recover-only receipt 进一步证明没有安全的 post-agent
resume：本次恢复 backend/model/operator/verifier/online-judge call 全为 0、stage transition=0、model replay=0，
并返回 `do_not_replay_model`。这条 RAW 不重跑，也不靠放宽 receipt 补齐。

剩余 7 个完整 pair 的描述性结果为 candidate 2/7、RAW 1/7，candidate-only 1、RAW-only 0、both-fail 5、
both-pass 1，均值差 `+14.29pp`。但唯一正 discordance 位于 fold 2；fold 0、fold 1 的一个可比较 replicate、
fold 3 都是零差。若缺失 RAW 为失败，完整八对差为 `+12.5pp`；若它为成功，差为 0，所以严格缺失值界为
`[0,+12.5pp]`。这保留了“方向非负”的信号，却直接否定了当前数据上“多个 fold 稳定复现正收益”的表述。
primary batch 不完整且下界含 0，不能 promotion，也不能进入 controls、family-out 或 sealed。

事后只读、非评分诊断没有再制造 gate。8/8 candidate 的 semantic operation assignment 都与公开模板一致，
而 15/15 完成路都成功生成文件；candidate 的首个可见 verifier failure 集中在 2 个 stock count 和 4 个
quarter-increase rank。静态代码审计给出一个直接机制解释：冻结 parent candidate 为复现三条 consumed TRAIN
solution，故把相邻的 stock-class string literals 尾部合并成一个值；period-out instruction 却显式声明了完整、
分离的 TITLEOFCLASS 集合。同一旧 operator 还没有把 manager-rank inventory 限制到 latest eligible accessions，
tie-break 也未完全实现新合同。这说明下一候选应更换 evaluator-contract semantics，而不是给 prompt 或 promotion
再加条件。由于该判断发生在 outcome 后，只有全新的 untouched measurement 才能检验修复是否真的带来 task gain。

还有一项证据卫生缺陷：pytest CTRF 的失败 traceback 会保存部分 expected/observed diff，和 measurement view
声明的 `ground_truth_persisted=false` 不完全一致。它发生在 agent 结束、verifier 执行期间，不会回流影响本轮
模型输出，也没有触碰 sealed；但 8 个 measurement item 必须视为彻底消费，后续报告和诊断不得复制其中的答案值。
完整的无答案值离线汇总见
[`financial_semantic_sec13f_period_out_partial_result_v1.json`](../manifests/financial_semantic_sec13f_period_out_partial_result_v1.json)。

### 1.2 结论分层

| 命题 | 当前状态 | 证据层级 |
|---|---|---|
| legacy HLE 是高维手写控制面，学习 policy 没有闭环 | 支持 | 代码审计 + 历史 artifacts |
| v2 的 proposal -> repair -> off/on -> gate -> archive 接口已连通 | 支持 | typed-selection integration v2 13/13 + 12/12；v3.18r1 完成 38/38 TRAIN、双臂两代与四份 report/archive |
| v2 的内部 runtime action 能改变 lane plan | 支持 | 代码 + 单元测试 |
| v2 主 SkillLearn 路径执行了每个 typed action/verifier/fallback 的强语义 | **不支持，且协议已停止这样声称** | 只接受四类显式 prompt/self-check lowering；其余 fail closed |
| promotion threshold 完全由冻结 protocol 所有 | 支持 | protocol-bound spec + 宽松 candidate 对抗测试 |
| 86-item offline-ready runtime 与 affected task-input closure 已预验 | 支持（冻结 closure scope） | v3.19 v5 cache-only prewarm 86/86；11/11 closure-required images 以 immutable ID、无网络 inventory/content receipt 复验 |
| production opaque recipe selection 已接入真实 proposer/evolution | 支持（机械层） | formal integration v2 13/13 predicates、12/12 tamper、exact replay；live v3.18r1 已 materialize/install/route |
| v2 主 SkillLearn 路径实际执行 typed capability | **部分支持，仅限 pre-agent 只读 evidence sidecar** | typed-portable integration 的真实 Docker canary 执行只读 evidence profile/inventory；`task_effect_claimed=false`、`recipe_operator_effect_claimed=false`，write/render/move 未由 capability 执行 |
| frozen financial semantic capability 在 fresh item 上是否出现 paired task gain | **支持存在性，但 period-out 未复现为稳定多折收益** | `financial-analysis-4` 为 0→1；随后 8-pair SEC 13F period-out 最大并发运行得到 15 valid + 1 fail-closed RAW。7 个完整 pair 为 candidate 2/7、RAW 1/7、+14.29pp，但唯一正差只在 fold 2，完整八对缺失值界 `[0,+12.5pp]` |
| typed-portable integration 已授权 fresh v3.20 development | 支持，且授权已使用完毕 | 一次正式 run + exact replay PASS；随后 v3.20 fresh development 已完成，本身仍无 incumbent/promotion |
| v2 已产生可保留的 promoted incumbent | **不支持** | single-item gain 之后的独立 period-out batch 不完整，且稳定多折收益未建立；v3.20 两代 archive 仍为 `incumbent_id=null` |
| v2 稳定优于 raw 或 budget-matched raw | **不支持；方向信号仍非稳定复现** | period-out 7 个完整 pair 无 observed regression、1 个 candidate-only gain，但只有 fold 2 为正；缺失 RAW 使八对差值下界为 0，不能作 promotion claim |
| v3.15 已改善真实 action utility | **不支持；clean live 负结果** | 13 个 candidate audit 中 7 material / 6 restatement-risk；material 仅 exact constant/mapping，且 9 roots 全坍缩为 poster 单-family |
| v3.17 family-slot/artifact-blueprint proposal 已达到 trial-feasible | **不支持；proposal-only 负结果** | 8/9 feasibility 通过，但第三候选绑定 2 个 failed TRAIN primitives；0 benchmark/evaluator call |
| v3.18r1 是 clean typed-action utility negative | **不支持；只能作机械闭环与异常诊断** | organize 输入缺 100 PDF；stock partial-test harm 被 binary success 投影为 tie；literal TRAIN locator 不可移植 |
| v3.20 是 clean typed-portable development | **支持，但是无 incumbent 的负结果** | 60/60 valid trials，offline verifier/action receipts 完整；两代均 3/16 activation、5/16 对 5/16、0 gain/0 harm；sidecar 无下游读取证据 |
| v2 已实现 Red Queen 式多 clade 搜索和 evaluator co-evolution | **不支持** | 目前是单 incumbent；evaluator 路径未接主实验 |

### 1.3 潜力判断

研究问题是连贯且可证伪的。显式 `HypothesisProgram` 可能比整块 workspace mutation
更容易做 lineage、activation 和 off/on attribution；但“更可解释”目前仍是待验证
假设，而不是既成事实。它至少需要以下操作化证据：

- schema fidelity；
- action lowering 成功率；
- lineage completeness；
- prospective activation precision；
- paired gain/harm attribution；
- cross-instance 与 cross-family retention。

在结构重复、可程序验证、能复用操作步骤或约束模式的任务上，超过单次 raw 有现实
潜力；在 broad random HLE 上稳定领先的先验较低，因为知识瓶颈、一次性长尾和 source
availability 会与 policy quality 混杂。HLE 更适合作外部 transfer/stress test，而不应
继续作为唯一开发靶子。

## 二、术语、三种“递归”与证据标签

### 2.1 核心术语

| 术语 | 本文含义 |
|---|---|
| assumption / hypothesis | 可证伪的关系、策略或 evaluator 命题，不等同于任意 prompt 建议 |
| `HypothesisProgram` | trigger、anti-trigger、action graph、expected effect、verifier、fallback、lineage 与 evaluator epoch 的结构化程序 |
| activation | 程序在运行前由可用特征命中，且实际改变 treatment 或 execution plan |
| promotion | 只依据冻结 validation 与预注册 gate，把 candidate 变成未来 runtime incumbent |
| archive node | 一组 active programs、runtime version、evaluator epoch 与证据依赖的完整配置 |
| evaluator epoch | 一个 evaluator、artifact protocol 和 scoring rule 保持不变的时期 |
| selective erasure | evaluator 被替换后，仅使依赖旧 evaluator 的 utility/score records 失效；不是删除失败假设的同义词 |
| clean external evidence | split、provider、预算、runtime、verifier、invalid-row policy 和 protocol lock 都满足预注册约束的外部结果 |

### 2.2 三种“递归”必须分开

1. **同题推理递归**：在一道题内展开 assumption tree 或多轮验证；
2. **假设修复递归**：候选未通过静态/训练检查后，生成有 lineage 的 child；
3. **跨代演化递归**：被 promotion 的程序改变 incumbent，再影响下一代 train residual、
   proposal 和未来题的 runtime。

legacy 主要有第 1 种；v2 已实现第 2 种的机制和第 3 种的 harness，但尚未出现真实
promotion，因此还没有观察到完整的跨代能力积累。RQGM 的核心则是跨任务 archive
tree search，不应被简化成“多调用几次模型”。

### 2.3 证据标签

本文使用以下强度顺序：

- **[CODE]**：源码直接可见的事实；
- **[TEST]**：离线测试验证的 wiring/invariant；
- **[ARTIFACT]**：真实运行留下的报告或 event；
- **[INFERENCE]**：由代码和结果支持、但尚无 controlled ablation 的解释；
- **[PROPOSAL]**：建议或验收标准。

## 三、legacy Assumption Agent 的架构诊断

### 3.1 真实行为链路

```text
Assumption Graph
  -> retrieval / OperatorSpec / morphism
  -> multi-prompt candidate generation
  -> source search / span / comparator / many verifiers
  -> fallback / selector
  -> final answer

final trace
  -> transition dataset
  -> fast-policy miner
  -> candidate/shadow policy
  -X-> did not reliably control the next HLE runtime
```

所以 legacy 中真正产生行为变化的主要是 prompt ensemble、手写 domain rule、source、
verifier、fallback 和 selector，而不是“系统自己提出、验证、保留并复用的假设”。

### 3.2 高维控制面的复杂度证据

**[CODE]** 在审计 revision 上，
[`hle_smoke_eval.py`](../../assumption_os/hle_smoke_eval.py) 共 116,808 行；AST 可见
1,401 个顶层函数定义，包含嵌套定义时为 1,604。源码中出现 770 个唯一的 `HLE_*`
配置名。按顶层函数名统计，126 个含 `verifier`，56 个含 `fallback`。

这些数字是复杂度代理指标，不等于 770 个布尔开关、126 个独立 verifier 或 56 个
独立 fallback 行为。它们能直接证明的是：旧系统有很大的配置面和函数面；“归因困难”
则是由此产生、并与历史反复局部回归相一致的**诊断推断**。若要严格量化，仍需要
调用图、交互覆盖与模块消融。

该控制面带来四个可观察风险：

- 一次提升难以归因到 assumption、source、selector、fallback 或额外预算；
- 局部规则可能覆盖另一个局部规则；
- 同 seed 的模型波动可能被误认为代码改进；
- 每次“下一刀”增加自由度，扩大 adaptive overfitting 与不可复现风险。

### 3.3 同题递归没有形成跨题学习

**[CODE]** HLE 在
[`hle_smoke_eval.py:L2963-L2973`](../../assumption_os/hle_smoke_eval.py#L2963-L2973)
调用
[`build_recursive_assumption_run`](../../assumption_os/recursive_runner.py#L73)
时使用 `writeback=False`。该 runner 会构造可审计的同题 assumption tree，但不会把
一道题中新提出并通过验证的程序写回为下一题可调用的 incumbent。

因此 legacy 的“递归”主要是同题内展开，而不是：

```text
propose hypothesis
  -> validate and repair
  -> estimate benefit, harm, and cost
  -> promote or reject
  -> alter future runtime behavior
```

### 3.4 fast policy 没有 effectful semantics

**[CODE]** 旧代码已有
[`fast_policy_memory.py`](../../assumption_os/fast_policy_memory.py) 和
[`hle_fast_policy_miner.py`](../../assumption_os/hle_fast_policy_miner.py)，但 HLE 主文件
对 [`route_option_lanes`](../../assumption_os/hle_lane_router.py#L105) 的四处调用并未把
完整 `fast_policy_decision` 接成动作控制。router 即使收到 policy，也主要把
`selected_policy_ids` / `selected_actions` 写入 metadata，不会据此启停 candidate、
source、solver、verifier 或 final-selection lane。

因此 policy 当时是可审计的 data object，不是可消融的 behavior program。

### 3.5 miner 学的是故障支持度，不是因果收益

**[CODE]** 旧 miner 在
[`_make_policy`](../../assumption_os/hle_fast_policy_miner.py#L185-L207)
里用 `support_count / wrong_count` 构造 `expected_utility`。这回答的是“某个 failure
bucket 出现得多不多”，而不是：

```text
同一题、同一 evaluator、同一预算下，
policy_on 相比 policy_off 修正了多少题，又伤害了多少题？
```

高频故障可以对应无效修复，也可以对应副作用更大的修复。没有 matched off/on，
故障频率不能被解释为净因果收益。

### 3.6 transition data 可审计，但缺 prospective trigger semantics

**[CODE]** [`hle_transition_dataset.py`](../../assumption_os/hle_transition_dataset.py)
为防止泄漏保存了 hash、label、failure bucket、cost 和 path metadata，这是正确的审计
方向；但当时缺少足够的关系类型、约束结构、输出 schema、可验证条件、候选差异和
反触发条件。数据可以证明“发生过一次 transition”，却很难支持 router 学会“什么
新题应触发哪条 policy”。

## 四、从 self-evolution 文献抽取的项目设计约束

本地材料实际包含
[`21 篇 self-evolution/continual-learning PDF`](../reference/self_evo_continual_20260707/papers/)、
2 个背景页面、21 个相关 repo，以及单独保存的
[`RQGM 论文`](<../reference/The Red Queen Gödel Machine Co-Evolving Agents and Their Evaluators.pdf>)。
下表是面向本项目的机制综合，不表示每篇论文都逐字主张所有约束。

| 文献组 | 可迁移机制 | legacy 缺口 |
|---|---|---|
| SkillLearnBench、Voyager、LifelongAgentBench、Fast/Slow | 经验需要编译成可执行、可复用的 skill/fast state，并在未来任务中调用 | OperatorSpec 最接近，但历史 application evidence 不足 |
| Reflexion、ExpeL、FLEX、EvolveR、AgentEvolver | 成功/失败对照、经验抽象、credit assignment 与生命周期 | 主要从 error bucket 生成修补建议 |
| DSPy、GEPA、TextGrad、OPRO、GPTSwarm | 需要小而明确、可优化和可消融的计算图 | 优化表面是 116k 行隐式分支 |
| MemGPT、MemoryBank、A-MEM、HippoRAG 2 | 记忆组织与检索有价值，但 retrieval 不等于 learning | source coverage 投入没有闭合 policy learning |
| Agent-as-a-Judge、Self-Rewarding LM | 中间轨迹评价有价值，但自评需要外部 anchor 与漂移控制 | 多数 verifier 与被评系统共享模型偏差 |
| AI Agents That Matter | held-out、简单基线、预算匹配、成本与复现性必须同时报告 | harness 逐渐接近，历史主实验仍不完整 |

SkillLearnBench 关于 external feedback 优于纯 self-feedback 的结果，为旧项目的递归漂移
提供了一个与观察一致的候选解释；它不是项目内因果证明，仍需 recursive/no-recursive
和 external/self-feedback controlled ablation。

## 五、与 Red Queen Gödel Machine 的精确对照

两者在“生成候选 -> 固定标准评估 -> 保留有效变体 -> 影响后续搜索”的抽象上同构，
但搜索单位不同：Assumption Agent 试图搜索显式 assumption/operator；RQGM 搜索的是
可修改的多代理 workspace/archive node。不能把 RQGM 说成也在直接搜索显式假设。

### 5.1 archive tree 与 clade search

RQGM 把 self-improvement 表示成增长的 archive tree，并按 clade metaproductivity 进行
Thompson sampling；一个节点是共享的多代理 workspace。见
[`RQGM PDF pp. 4-5`](<../reference/The Red Queen Gödel Machine Co-Evolving Agents and Their Evaluators.pdf#page=4>)。

legacy 更接近同一代码主线上的人工 hill climbing：固定/debug seeds 持续影响代码、
prompt、verifier 和 gate，每次局部胜利都被编译进全局复杂度。

### 5.2 evaluator 在 epoch 内冻结

RQGM 在一个 epoch 内冻结 evaluator、artifact-generation protocol 和 binary scoring rule；
只有 epoch boundary 才允许 utility transition。这使每个 epoch 保持为固定标准的搜索
问题，而不是 agent 与 judge 同时漂移。

### 5.3 产生候选与选择候选的数据隔离

RQGM 明确分开创建 node 的 training feedback、驱动 node selection 的 validation，以及
最终 held-out test。训练结果可以指导 meta-agent 修改 node，但不能直接构成 search
utility。见 [`RQGM PDF p. 5`](<../reference/The Red Queen Gödel Machine Co-Evolving Agents and Their Evaluators.pdf#page=5>)。

legacy 中反复使用的 HLE fixed/debug seeds 只能继续作为 regression cohort，不能再作为
泛化证据。

### 5.4 anchor lower bound 与 selective erasure

RQGM challenger evaluator 在固定 ground-truth anchor 上按保守 best-belief/lower-bound
标准与 incumbent 比较。替换 evaluator 后，只清除依赖旧 evaluator slot 的 utility
records，并按需重排 archive；无关信息保留。见
[`RQGM PDF pp. 5-6`](<../reference/The Red Queen Gödel Machine Co-Evolving Agents and Their Evaluators.pdf#page=5>)。

因此：

- 失败 hypothesis 的降级、停用或归档不是 selective erasure；
- selective erasure 只描述 evaluator/utility dependency invalidation。

### 5.5 对 RQGM 性能主张保持边界

RQGM 是 preliminary empirical investigation；主要结果依赖强模型，理论保证是
epoch-local，anchor 偏差仍可能导致 evaluator 漂移。适合借鉴的是 archive、数据隔离、
冻结标准、anchor promotion 和 dependency-aware invalidation，而不是直接继承其性能
或开放式自我改进主张。

## 六、legacy HLE 效果证据 ledger

下表把点估计与 validity 同时报告。它不是 benchmark 排名表，而是说明为什么历史
结果不能支持“Assumption Agent 优于 raw”的主结论。

| 证据 | 点估计 | validity 与归因边界 |
|---|---:|---|
| [`fixed/debug agent-only 12`](../../reconstruction/md/hle_candidateconflictresolve_baseline12_agentonly_mini_20260629.md) | agent 6/12 | `pass=False`、paper-clean false、无同题 raw/Hippo、无 budget controls、operator application verifier 0 |
| [`所谓 unseen12 agent-only`](../../reconstruction/md/hle_docrank_mathbinding_current_unseen12_cacheonly_mini_20260708.md) | agent 5/12 | paper-clean false、无 controls；operator selected/activated 12/12，但 application verifier 0，不能归因于 operator/assumption |
| [`fresh triad promotion report`](../../reconstruction/md/hle_parallelrun_unseen_mc12_fair_policy_promotion_20260707.md) | agent/raw/HippoRAG 均 2/12 | 12 个 triad 齐全，但 promotion `pass=False`；有 control errors、无 selector gain、低于 24-triad gate，缺 budget-matched controls |
| [`controls-only 12`](../../reconstruction/md/hle_freshunseen12_controls_multiglob_fair_cacheonly_mini_20260709.md) | raw 2/12、Hippo 3/12、budget raw 4/12、budget Hippo 3/12 | 4 个不对称 endpoint errors，无 agent；clean-shared n=10 分别为 2/10、3/10、4/10、3/10 |

本表没有把 fixed cohort 的后续更高点估计当成反例隐藏掉：同一 cohort 后续经过更多
adaptive debugging 后出现过更高分，但这只提高 regression value，不提高 generalization
evidence。固定 cohort 越被反复用于决策，越不能承担 sealed claim。

历史证据只能支持：

- 系统有工程性能力和若干有效局部模块；
- 某些改动能改变单题或固定 cohort 的行为；
- frozen financial semantic treatment 已在一个 fresh same-item pair 上给出 RAW=false→candidate=true 的
  preregistered treatment-associated gain；
- 同一 candidate 的独立 SEC 13F period-out 复验在 7 个完整 pair 上保持 observed non-regression，并得到
  1 个 candidate-only gain；但该 gain 只出现于一个 fold，另有 1 条 RAW 因 residual process fail closed，
  八对缺失值界包含 0；
- 尚无可靠证据证明 agent 在多 item/fold 上稳定优于 raw、HippoRAG 或 budget-matched raw，也尚无
  attribution 证明该单题收益来自可迁移、可保留的 Assumption 机制。

## 七、legacy 缺口到 reconstruction_v2 的 closure delta

### 7.1 已明显改善的部分

| legacy 缺口 | v2 状态 | 证据 | 尚缺 |
|---|---|---|---|
| assumption 没有统一可执行 schema | 已实现三类 `HypothesisProgram`，独立 financial path 又执行了 bounded post-agent typed operator | [`models.py:L221-L275`](../assumption_agent/models.py#L221-L275)；[`financial_semantic_operator_v1.py`](../assumption_agent/benchmarks/financial_semantic_operator_v1.py) | production evolution 中通用、多 family 的 typed lowering |
| policy 不改变 runtime | 内部 `PolicyRuntime` 可启停、排序 lane、设参数和执行 operator step；financial fresh pair 已出现一次 0→1，period-out 又有 1 个 candidate-only discordance | [`runtime.py:L72-L226`](../assumption_agent/runtime.py#L72-L226)；[`financial recovered report`](../artifacts/financial_semantic_fresh_v1_plus_actual01/fresh_paired.recovered.report.json)；[`period-out partial report`](../manifests/financial_semantic_sec13f_period_out_partial_result_v1.json) | 完整有效且跨多个 fold 的稳定复验、promotion 与 retained benefit |
| 无 hypothesis repair lineage | 已实现 failed-check -> child repair tree | [`validation.py`](../assumption_agent/validation.py) | empirical repair benefit |
| utility 来自 failure frequency | promotion 已使用 protocol-owned paired gain/harm/cost/LCB，candidate 只能收紧 | [`evaluation.py`](../assumption_agent/evaluation.py) | 尚缺真实 promotion 与 retained gain |
| train/validation/test 混用 | split guard 与 archive-freeze gate 已实现 | [`splits.py:L220-L267`](../assumption_agent/splits.py#L220-L267) | 一次完整 current-protocol sealed run |
| evaluator 变更无依赖失效 | controller/anchor lower bound/selective invalidation 已实现 | [`archive.py:L291-L370`](../assumption_agent/archive.py#L291-L370) | 尚未接入主 evolution 或真实 challenger |
| HLE 是唯一主战场 | 已转向 86-item offline-ready SkillLearnBench，并增加 project-authored SEC 13F period-out measurement | [`BENCHMARK_PROTOCOL.md`](../BENCHMARK_PROTOCOL.md)；[`period-out freeze`](../manifests/financial_semantic_sec13f_period_out_execution_freeze_v1.json) | v3.20 为 clean negative；financial period-out 只有局部方向信号且 primary 不完整，仍无 replicated stable gain/incumbent，所以下游停止 |

### 7.2 当前证据到哪一层

**[TEST]** v3.13 的 `reconstruction_v2` 离线 suite 为 **375/375 通过**，v3.14 为
**411/411 通过**；v3.15 提交 `696a2954` 当时为 **453/453 通过**；加入 v3.16/v3.17 formation、
proposal-only boundary、typed representation 与 single-decision binding 后，当时完整 suite 为 **540/540 通过**。
后续 production integration v2 的 13/13 predicates、12/12 tamper probes、真实 harness-construction
regression 与 exact replay 另行通过；本文不把不同时点的 suite 数拼成新的总数。新增覆盖包括 TRAIN-only
action profile 的 containment/allowlist/secret isolation、request-local action-quality prompt、audit-only
不改变 retry/selection/promotion、terminal-invalid memo 的 retry identity 与零执行 replay，以及
report/freeze 的首代 checkpoint/profile provenance。此前 shared immutable valid baseline cohort、
legacy replay compatibility 和 family-count/support tie-break 覆盖仍保留。这些测试证明 schema、wiring、guard、
replay、failure handling 和若干 invariant；不证明真实 benchmark improvement。既有覆盖还包括
protocol threshold ownership、candidate 宽松阈值攻击、backend action lowering v2、exact-three
cardinality 与 audit-only signature diversity、真实/声明 fallback 分离、offline-ready split 不重抽样，
以及离线 verifier receipt 必须绑定 proxy 实际执行的 frozen runtime profile/command。

**[ARTIFACT]** 在第三个 v3.5 fresh-root 运行前，对 `reconstruction_v2/artifacts` 中当时
可读的 v1/v2/v3 smoke、diagnostics 和 development runs 做混合扫描得到：

- 23 份 `*.archive.json`；
- 22 份 `*.report.json`；
- 非空 incumbent：0；
- 这些 report 中 `promoted=true`：0。

这 23/22 不是 23 次 current-protocol 独立实验，也不能作为样本量；它只是 available
artifact tree 的状态审计。结果不是说 gate “失败”；恰恰说明现存 artifacts 没有把
诊断信号包装成 incumbent。但它也意味着系统尚未完成“promoted program 改变下一代
runtime”的实证闭环。

**[ARTIFACT]** 曾有一次 full replay-locked development 出现 raw 4/18、candidate 7/18、
3 gain/0 harm、cost ratio 0.914，但一条 baseline trial 无效，gate 正确拒绝；该结果只能
视为 promising but inadmissible diagnostic，见
[`STATUS.md:L96`](../STATUS.md#L96)。后续 pre-network-hardening 的 685a run 第一代是
raw 4/18、candidate 5/18、2 gain/1 harm，LCB 为负并被拒绝；第二代未完整收束，见
[`development_recursive.events.jsonl`](../artifacts/paper_primary_v3_ruoli_gpt54mini/runs/685a4482_full_development_20260711/development_recursive.events.jsonl)。
这些结果都不能形成性能主张。

**[ARTIFACT]** clean commit `e07913f9` 上当时的 v3.1 protocol smoke 已完成机制验收：
两臂均为 2 个有效 pair、0 invalid、0 provider/budget mismatch，且 behavior-identical
validation 被精确 replay；两臂 candidate/raw 都是 0/2，因此没有 promotion。它只证明
运输、lowering、paired replay 和 fail-closed promotion 能协同工作，不是性能证据，见
[`smoke_recursive.report.json`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/smoke_recursive.report.json)
和
[`smoke_no_recursive.report.json`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/smoke_no_recursive.report.json)。

随后第一次 v3.1 full development 在完整 38-item train 上严格中止，见
[`development_recursive.events.jsonl`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/development_recursive.events.jsonl)：

- 26 个本地 verifier 有效 observation，其中 9 pass、17 fail；
- 2 个已启动 trial 收到 `429 Too Many Requests`，provider circuit 随即打开；
- 9 个尚未启动的 trial 按同一 circuit 在本地跳过，没有继续消耗模型请求；
- `court-form-filling-6` 的长轨迹累计模型流量为 33,730,000 bytes，超过冻结的
  33,554,432-byte fuse，作为 hard-budget invalid 处理；
- 因 12/38 training observations 无效，training evidence 没有写入 replay cache，
  proposal、repair、validation、archive 和 promotion 均未执行；report/archive 也没有落盘；
- sealed split 保持未访问。

这次失败没有调用 online evaluator：task payload 与 verifier 均来自冻结的本地
SkillLearnBench checkout，evaluation 仍由 post-agent offline verifier 完成；唯一在线流量
是预注册的 agent model inference。因而缺口不是“再下载一个 evaluator”或“再补一个
readiness gate”，而是恢复预注册 provider transport 后取得一份完整、0-invalid 的
development evidence。当前进程内 training replay 也不能跨失败进程复用这 26 条有效结果，
所以它们只能作为 transport diagnostic，不能与后续 run 拼接成 claim。

全 run 退出后的单题、5-step、非 claim transport canary 已在同一 provider route 上恢复：
模型请求完成、offline verifier 正常执行，observation 为 `evaluation_valid=1`、
`task_success=0`。这说明 429 已冷却；任务失败不等于 transport 失败。canary 没有读取
validation/sealed，也不进入任何性能汇总，见
[`transport recovery canary`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/transport_recovery_canary/report.json)。

29 个实际启动 trial 的 network receipt 显示总流量中位数约 2.40 MB、p90 约 5.53 MB、
p95 约 22.03 MB；唯一超过 32 MiB 的就是被右删失的 `court-form-filling-6`，且只超出
175,568 bytes（0.52%）。因此 v3.1 没有因单个 train diagnostic 原地抬 cap，而获得最多
一次同协议、全新 run-root 的 clean rerun。该 rerun 在同一 item 上再次触发 hard cap，
这次观测到 38,599,999 bytes；进程在 stop condition 已不可逆后主动中止，没有继续烧完
余下 train。v3.1 因此正式判为 execution-infeasible，而不是继续重跑到碰巧通过。

唯一允许的资源修订已版本化为
[`v3.2 protocol`](../manifests/skilllearn_paper_protocol_v3_2_ruoli_gpt54mini.json)：
统一 provider-only fuse 从 32 MiB 提高到 64 MiB；model、86-item subset、offline evaluator、
dependency policy、4 workers、search budget 和 promotion contract 均不变。64 MiB 是
train-only 已观察最大流量向上取下一 2 的幂，不读取 validation/test，也不复用 v3.1
observations。后续不再允许第二次按题调 cap。

v3.2 在 clean commit `748469b2` 上重新得到 claim-eligible lock 和 86/86 cache-only
prewarm，随后 full development 的 64 MiB budget 没有触发；`court-form-filling-6` 本轮以
4.70 MB 正常完成并通过 verifier-validity 检查。然而同一路由在 8 个有效 train
observation（1 pass、7 fail）后发生独立 transport outage：一个 trial 明确以 429 终止，
其余三个在途 trial 的原始 Codex JSONL 先连续报告 `gpt-5.4-mini` 没有可用 distributor
channel（503），随后也以 turn failure/429 收尾；provider circuit 打开后，29 个尚未启动
请求被本地跳过。training evidence 因 30/38 invalid 再次 fail-closed，未进入 proposal、
validation 或 sealed，见
[`v3.2 development events`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini/development_recursive.events.jsonl)。

随后使用用户提供、仅注入进程且未落盘的新 GPT Pro credential 对同一
`https://ruoli.dev` / `gpt-5.4-mini` route 做最小 canary：models、Responses、chat
端点均为 HTTP 200，真实 Codex container 也完成 agent -> offline verifier，证明
`ruoli` 本身并非永久不可用。这里没有切 model/provider contract，也没有使用 online
evaluator。

第一个新渠道 full root `gptpro01` 完成 38/38 train：36 valid、12 success、2 invalid；
provider/circuit 与 64 MiB cap 错误均为 0，最大 trial 流量 62.9 MB。两个 invalid 都来自
`temperature-simulation-2/3` 的确定性 receipt false-negative：proxy 实际执行的离线
profile 已生成完整 CTRF（各 7 tests、5 pass/2 fail、reward 0），但旧 auditor 错误检查
了未被执行的 upstream `test.sh` 是否含 `--ctrf`，把本应是“有效任务失败”的结果标成
`verifier_execution_receipt_unsupported`。这不是增加 gate 的理由，而是 evidence source
绑定错误。

提交 `e43670f6`、`18ff3417` 之后，receipt 直接绑定 proxy 实际执行的 frozen runtime
profile ID/hash/command hash；profile-backed CTRF 缺失或畸形仍 fail closed，reward 0
仍是 valid failure。该历史提交当时的全套 136/136 tests 通过。最终 clean root `gptpro03` 的 lock 绑定
`18ff3417` 且 claim eligible，prewarm 再次为 86/86、0 failed、无 online build。
真实 run 中两项 temperature receipt 均成为 `pytest_ctrf`、`test_count=7`、valid=true，
证明修复生效而未把失败改成成功。

`gptpro03` 最终 38/38 train 返回：37 valid、9 success、1 invalid；provider error 为 0。
唯一 invalid 是 `video-object-counting-1` 的真实 hard-cap：TX 66.5 MB、RX 4.6 MB、总计
71.1 MB，超过冻结 67,108,864-byte limit，容器被监控终止且禁止 retry。因
`all_valid_before_proposal_v1`，proposal/generation/validation/no-recursive/promotion 均为 0，
report/archive 未生成；sealed/test content 仍未访问，见
[`gptpro03 development events`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini_gptpro03/development_recursive.events.jsonl)。

该 71.1 MB 全部发生在只允许模型端点的 egress 域内。trace 只有 23 次本地命令和 10 次
agent message，可见 shell 输出合计约 13 KB，没有 web/image、包安装或视频上传命令；
15.4 MB 本地视频只被 ffmpeg/OpenCV 读取。同族 `video-object-counting-3` 用 16 次命令、
约 8.79 MB 即完成。因此最可信解释不是依赖缺失或数据外泄，而是 Responses 多轮工具
往返反复发送累计 context/envelope，叠加数次无效能力探测和调试循环，造成 TX 放大。

据此，**v3.2 也正式判为 execution-infeasible**。本项目不会第三次抬 cap、删除该题、
降低验收标准或重跑到碰巧通过。本地 Codex 0.144.1 审计确认请求压缩已经默认开启；
`previous_response_id` 没有公开 ConfigToml 开关，固定 mini route 的最小
`/responses/compact` canary 又返回 503 `model_not_found`。可用且必须 protocol-owned 的
节流手段是 `model_reasoning_effort=low`、更早的本地 auto-compaction，以及低 verbosity；
tool-output cap 对本题约 13 KB 可见输出预计是低杠杆。

下一工作流只允许改进 model transport/trajectory 效率：统一冻结上述配置，并把视频类
轨迹收敛为“一次能力探测、一个整合脚本、一次最终校验”；先在 train-only 长轨迹
canary 上证明仍低于现有 64 MiB，再冻结一次新的 execution-policy revision。
promotion/subset/evaluator/cap 不变，也不拼接任何失败 run 的有效 observation。

该工作流已由 v3.3 完整执行。提交 `e0b1a33b` 新增独立
`codex_low_reasoning_early_local_compaction_v1` policy；v3.1/v3.2 仍解析为旧 catalog-default
treatment，不会被静默套用新配置。v3.3 与 v3.2 删除 protocol ID/version 和这一 policy
字段后逐项相同：64 MiB、86-item subset、4 workers、search/promotion/evaluator/sealed
合同均未改变。该 v3.3 提交当时的全套 150/150 tests 及 Codex 0.144.1 `--strict-config` 的断网解析通过。
claim lock 绑定 clean commit `e0b1a33b`、policy hash 和 67,108,864-byte cap，
`claim_eligible=true`；cache-only prewarm 为 86/86、47 images、7 verifier runtimes，
`online_build_attempted=false`。其中历史字段 `test_content_accessed=false` 表示未执行/评分
test split，也未向模型暴露 test bytes；prewarm 的 infrastructure 路径实际会读取并哈希
test task/image/verifier 文件，v3.4 receipt 已改为显式记录这一区别。

train-only `video-object-counting-1` canary 先得到 valid task failure：总流量 1.47 MB、
`error_type=null`、本地 `common-pytest-ctrf-py312-v1` verifier、0 provider/cap/sealed event。
随后同一 lock/prewarm 的 full development 返回 38/38 train：37 valid、9 success、1 invalid；
38 个 verifier execution receipt 全部 valid，37 个 model-only audit valid、1 个 violated；
38 个 network monitor 均 finalized，最大为 `temperature-simulation-3` 的 40.6 MB，
`video-object-counting-1` 为 19.69 MB，provider/circuit 和 hard-cap error 都为 0。因此
**本次 v3.3 batch 已排除 fuse 作为未进入 proposal 的直接原因**。这不等于跨运行稳定性
已经成立：同一 video task 在 canary/full 中为 1.47/19.69 MB，后续仍须报告这一波动。

唯一 invalid 是 `offer-letter-generator-1`。其 trace 明确包含
`item.started(type=web_search)` 和 `item.completed(type=web_search, query=placeholder)`；
auditor 记录 `remote_tool_call_count=1` 并正确产生
`model_remote_tool_policy_violation`。该 trial 已同时使用 `--ignore-user-config`、
`tools.web_search=false`、disabled `standalone_web_search` 和禁止联网工具的 developer
instruction；容器 egress 又只允许模型端点。故它不是 online evaluator，也不是 benchmark
dependency 下载，而是 Codex/Responses execution boundary 出现了禁止的远程工具 item；
现有证据还不能区分该 item 源于 Codex CLI mapping、provider 还是 model。
按现有合同这是不可重试的真实 invalid，不应通过删 trace、白名单 placeholder 或重采样
改写成普通任务失败。`all_valid_before_proposal_v1` 随即阻止 residual/proposal；0 proposal、
0 counterfactual、0 sealed event，recursive/no-recursive report/archive 均未生成。

本轮还暴露一个与评分 gate 无关的执行预算缺口：`max_steps=100` 目前只参与 request/backend
锁定与一致性校验，上游 Codex run template 没有把它传给 `codex exec` 作为可执行限制。
有效的 `temperature-simulation-3` 运行 2,485 秒、累计约 334 万 token，并产生 241 行 JSONL；
其中只有 93 行是 `item.started`，且它们不能直接等同于 protocol 的 semantic step 或 turn，
所以本 run **不能**证明越过了 100-step/turn cap。它证明的是该 cap 在当前执行链中没有
可审计的 enforcement。该结果本身不授权直接启动新 paper run 或添加 ad-hoc gate；若要
声称 budget-matched，必须先定义并离线验证一个可观测的 action budget。以下 v3.4
revision 正是在完成该非评分诊断后建立，而不是对 v3.3 invalid 的重试。

#### v3.4 零模型定位与最小执行修复

上述两个来源不明的问题现已在不调用模型、不评分、不访问 sealed 的条件下定位并实现
最小修复。

首先，精确审阅 Codex 0.144.1 源码与 loopback 出站请求后确认：旧配置
`tools.web_search=false` 是兼容性 no-op；布尔 `false` 被解析成空值，随后未设置的顶层
`web_search` 回落到 `cached`。自定义 Responses provider 支持 hosted web search，因此
v3.3 的 38 个请求都曾把该工具暴露给模型，不能把其中 37 个“未调用” observation 复用为
合规证据。根因在本地 CLI 配置语义，不是 Ruoli 注入，也无需换 provider、升级 Codex、
重试样本或扩充评分 gate。

v3.4 只合并两项 execution-treatment 修复：使用顶层 `web_search="disabled"`，以及把
`max_steps` 落为 `codex_action_start_v1`。独立的
[`codex_model_only_wire_probe_v1`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/diagnostics/codex_model_only_wire.json)
直接复用实际 provider argv，并用本地 canned Responses server 做阴性/阳性对照：

- canonical v3.4：`POST /v1/responses` 一次，`stream=true`，7 个工具，0 个
  `web_search*`/`web.run`，Codex 正常产生 `turn.completed`；
- stale boolean 对照：同一配置只把 canonical 顶层键换回 `tools.web_search=false`，8 个
  工具中出现 `web_search` 且 `external_web_access=true`；
- 两次请求都只到 loopback，模型推理与评分均为 0；raw request、instruction、Authorization
  和 raw trace 均不落盘。

预算单位不是模糊的 turn：每一条可解析的 `item.started` 都计数，缺失 item/id/type 的
malformed start 也保守占一单位并使 evidence invalid，不能靠畸形事件绕过上限。容器内
Node supervisor 为 Codex 建独立进程组；第 N 个 start 先发 TERM，15 秒为 KILL 的上限。
它直接持有并截断当前 attempt 的 trace，写入随机 nonce，并在 verifier 开始前扫描专用 trial
容器的 `/proc/<tgid>/task/<tid>`，清除基线之后新增的全部 live task；因此 `setsid`/new
session，以及 thread-group leader 已为 zombie 但 worker 仍存活的情况也不能逃逸。任何无法
完成的 task scan 都 fail closed。receipt
绑定 supervisor hash、nonce、full trace hash、action projection hash、limit、实际 steps、
spawn/exit/signal、post-trigger count、严格 token usage、process-group 与 container-scope
confirmation。离线注入已覆盖正常完成、无/空 usage 的 `turn.completed`、exit 42、spawn 失败、
同 chunk N+1、旧 receipt 替换当前 trace、background descendant、`setsid` escape、zombie
leader + live pthread worker，以及忽略 TERM 后的 KILL。自然完成若遗留 task 会清理并
invalid；受控 budget truncation 在所有 live task 清空后可保持 valid。完整 trace 中较晚的
429/鉴权/限额等 fatal provider 事件优先于较早的 generic stream error，避免被误分为普通
receipt invalid 并继续请求 provider。

截断通常没有完整 token usage。为避免 promotion cost ratio 把一臂的 token 与另一臂的
action count 静默混算，v3.4 明确对所有 arms 统一使用 action starts 作为 promotion cost；
token usage 继续作为二级报告指标，并逐 trial 持久化 completeness/truncation。64 MiB cap、
subset、workers、retries、evaluator、promotion、statistics、recursive/no-recursive 定义和
sealed policy 均不变。clean commit/lock、新 runtime cache 与 86/86 v4 cache-only prewarm
已经完成。v4 canary 实际到达模型和 verifier，但 root-owned `0600` trace/receipt 使宿主 auditor
无法读取；`1df3092a` / `ad66d5a2` 将这两类 immutable artifact 显式创建为 `0644`，同时保留
temp-file + rename 原子写入、内容 hash 和 nonce 绑定。对应离线/Docker 回归直接断言生产模式，
不再靠测试侧 `chmod` 掩盖回归。

随后同一路由 `max_steps=2` non-claim/train-only v5 canary 通过：observation
`evaluation_valid=1`、`task_success=0`、`steps=2`、`truncated=true`，action receipt、process
cleanup、model-only tool audit 与 post-agent local verifier 均 valid，0 remote/web tool、0 runtime
install。任务未通过不等于执行机制无效；这个结果满足了启动一次 fresh-root development 的
预设条件。

fresh development 最终覆盖 38 个 train request outcome：17 条 valid（3 success）、4 条已启动
trial 因 `provider_rate_limit` invalid、熔断器打开一次后 17 条在本地跳过。21 个实际启动 trial
均完成 network finalize，最大 47.2 MB、0 hard-cap；17 条完成推理的 action/tool/verifier audit
全部 valid。运行以 `all_valid_before_proposal_v1` fail closed，proposal、paired validation、
recursive/no-recursive report/archive 与 sealed event 均为 0。失败运行中的 17 条 valid observation
不能跨进程拼接。切换 online evaluator 对 model inference 429 没有帮助；下一次运行前需要解决
冻结四并发与 provider 持续容量的矛盾，而不是继续增加评分 gate。

v3.5 随后只把五个在线 phase 的 `parallel_workers` 从 4 改为 1。新 clean lock 与 fresh
86/86 prewarm 均通过；serial train 在约 99 分钟内完成 38/38 valid、9 success，38/38
action/tool/offline-verifier audit 有效，0 provider/circuit、0 cap，最大 finalized traffic
60.28 MB。一次 root proposal 在同 request hash 的 `RemoteDisconnected` 后重试成功并返回
3 个候选；两个静态通过，第三个进入两层 repair。depth 1/2 repair 的 model-declared ID 相同，
但 payload hash 分别不同，archive 因而正确 fail closed。碰撞发生在 validation-split trial 前，
所以仍无 paired counterfactual、promotion、recursive/no-recursive report/archive 或 sealed event。

根因不是 archive gate，而是 repair 把模型字符串当成全局主键。修复后的
`parent_content_scoped_repair_id_v1` 忽略模型 ID/status，使用 parent ID、去除可变 status 的规范化
parent-content hash、repair depth 与去掉 ID 的 canonical candidate child 派生 `repair_<sha256>`；
repair 一律由 harness 以 `candidate` 身份进入既有生命周期，事件记录 policy/hash 与被弃 model ID 的 hash。
同 parent/content replay 保持确定，跨 root 和跨 depth 不再 alias；archive 对真正的同 ID 异内容
仍抛错。离线回归覆盖 sibling roots、此次 depth-1/depth-2 复现和 archive 阴性对照。旧进程已死，
JSONL 不是 checkpoint，38 条 observation 不能拼入修复后运行；必须新 lock/prewarm/root。

repair identity 修复后的 fresh root `repairid01` 绑定提交 `96d53a5d`，新 lock 与 86/86
cache-only prewarm 通过。38 个 serial train 再次全部 valid，其中 9 success；trial duration
合计 5,188.542 秒、最长 574.710 秒，38/38 action/tool/offline-verifier audit 有效，38/38
network monitor finalized、0 超限，最大 22.82 MB。训练后形成 29 residual，root proposal
transport 成功并返回 3 个程序，paired checkpoint 冻结。recursive arm 只验证了第一个 root：
`training_support=false` 且 `runtime_action=false`。repair request 的 HTTP、JSON-object parse 和
provider selection 均成功，但 parsed response 没有 mapping-valued `hypothesis`；事件只保存
response hash，raw response 未落盘，因此不能进一步声称具体错误 envelope。旧 proposer 在
`_complete()` 返回后抛裸 `ValueError`，validator 只捕获 typed `HypothesisProposalCallError`，
进程遂在 `hypothesis_repair_proposed` 前 exit 1。16 个 validation ID 仅被 authorize，实际
counterfactual、promotion、archive、generation complete、report/archive 与 sealed event 均为 0。

提交 `d70562de` 在模型响应语义边界做最小修复。root 的所有 consumed rows 先原子化
canonical parse，成功后才 emit/replay；repair 的 envelope/canonical parse failure 进入同一 typed
candidate-local 通道。事件只写 request/response/key-set hash、字段 presence/type/count 与 phase，
不写 raw。validator 不捕获任意 `ValueError`，所以 archive collision/harness invariant 仍 fail loud。
一个 malformed repair 只终止该 branch，其余 root 继续 static audit，但整代不执行 held-out
validation 或 promotion；malformed root 为两臂保留 terminal non-claim report。report 的
failure count/presence/claim/blockers 从 generation rows 派生，paper freeze 再独立重算，诚实
failure 或 top-level 篡改都被拒绝。该修复落实既有 failure policy，没有改变评分阈值或搜索预算。

response-contract 修复后的 fresh root
`paper_primary_v3_5_offline86_ruoli_gpt54mini_repaircontract01` 绑定 clean commit
`4f772e38`，86/86 cache-only prewarm 再次通过。单 worker train 在同一 invocation 内完成
38/38 valid、5 success，trial duration 合计 6,472.468 秒、最长 691.144 秒；38/38
action/tool/offline-verifier audit 有效，network 全部 finalized、0 hard-cap，最大约 50.2 MB。
root proposal 返回两个候选；其中一个静态通过，另一个 depth-1 repair 成功并以新的
content-scoped ID 注册。训练侧仍按最大 failure support 选择 repair child，因此这次真实经过了
前两次运行未能到达的 paired validation、promotion decision、双臂多代停止与 report/archive
落盘，repair identity 与 response-contract 修复均在生产路径上得到验证。

第一代给出了目前最有信息量的因果对照。recursive arm 选择 29-support repair child，16/16
validation pair 全部 valid，但 activation 13、gain 0、harm 2、mean effect -0.125、90% lower
bound -0.2344，因净收益、harm rate 与 lower bound 三项被拒绝。no-recursive arm 在相同 train
observations、residuals 和 roots 上选择 14-support 的 `artifact-completion-guard-v1`；其 16/16
pair 同样全部 valid，activation 6、gain 1、harm 0、mean effect +0.0625、lower bound
-0.0176，只因效应下界未越过 0 而拒绝。这个结果不能算 promotion，但它直接暴露了当前
selection bias：最大化 failure support 的 repair 覆盖更广却造成伤害，较保守的 root 反而产生
唯一净收益且没有 observed harm。递归本身尚未证明有益；成功训练样本没有进入候选排序，
anti-trigger 的所谓 support 也只来自失败 residual，不能表示对已成功区域的保护。

第二代进一步区分了 architecture failure 与 infrastructure failure。recursive arm 的一个新 root
连续两层 repair 后仍未通过 static checks，因此以 `recursive_validation_rejected` 正常结束，未消费
held-out task trial。no-recursive arm 的新候选通过 static checks，并复用同臂 baseline evidence；
前 4 个实际 candidate trial 正常完成，随后 Ruoli 对一个 policy-on request 返回 HTTP 503，分类为
`provider_model_unavailable`。run-scoped circuit 打开后又跳过 8 个 candidate trial，最终该代
9/16 pair invalid、8 个 budget mismatch。promotion gate 正确以 invalid evidence 等 blocker 拒绝，
但 lifecycle 把这一代错误计成第二次普通 non-promotion：top-level report 仍写
`performance_claim_eligible=true`、`evolution_stop_reason=consecutive_non_promotion_limit`，对应
archive score 也仍为 `valid=true`。这不是 hash 损坏，而是 evidence/claim 语义缺口；provider
故障不应消耗科学性的 non-promotion 次数，也不能产生 valid score。

本轮主事件 2,397 行与 prewarm 340 行的 payload hash/event ID 重算错误均为 0；两份 archive
内部 hash、report 引用、node/score 引用均一致，8 个 secret-like 环境值对 574 个 artifact 文件的
exact-literal 扫描为 0 命中，609 次 split access 为 497 train、112 validation、0 test。两臂
`incumbent_id` 都为 `null`，sealed/test true 为 0。因此本轮应作为“机械闭环成立但 learning 未达到、
且第二代被 provider 污染”的负结果收口，不 freeze 空 control，不运行 validation-controls 或 sealed。
下一版不是放宽 evaluator 或继续增加评分 gate，而是一次性版本化为 v3.6：成功 train rows 作为
anti-trigger negative controls；候选只用 train evidence 按 activation precision、success false-positive、
failure support 与复杂度排序；invalid counterfactual evidence 以 terminal non-claim 停止而不增加
non-promotion counter。model、single worker、split、预算、offline evaluator 与 promotion mapping
保持不变，旧 v3.5 rows 不跨协议复用。

**[ARTIFACT]** 2026-07-13 的 v3.9 fresh root
`paper_primary_v3_9_offline86_ruoli_gpt54mini_outer6_model1_plus01` 首先通过 clean lock 和
86/86 cache-only prewarm，随后完整写出 recursive/no-recursive 两组 report/archive。事件账本记录
56 次实际外部 trial、56 次 model-slot acquire/release、观测最大在线 agent 并发为 1；56 次
trial 全部完成，provider failure、circuit open/skip、infrastructure failure、network-budget failure、
invalid pair、provider mismatch 和 budget mismatch 均为 0。唯一在线环节仍是同一 Ruoli 模型
推理；task 数据与 verifier 均为本地，verifier 在 agent 退出后以 `--network none` 执行；sealed/test
访问保持 false。

共享 train evidence 为 38/38 valid、10 success、28 residual。recursive 第一代 root 的初始
family scope 因 anti-scope support 静态失败，depth-1 repair 通过；它在 train 上只激活 2 个 failure、
0 个 success control。held-out 16 pairs 全部有效，但 prospective activation 只有 1/16，candidate/raw
均为 3/16，gain 0、harm 0、effect lower bound 0、cost ratio 1.038462。第二代未修复 root 也只在
held-out 激活 1/16，仍是 3/16 对 3/16、0 gain/0 harm，cost ratio 1.052885。两代均被原冻结
promotion contract 以 zero net gain、6.25% activation 和 zero effect lower bound 拒绝。no-recursive
两代各返回一个 root，但都未通过 train-only static audit，因而没有消费 held-out treatment trial。
两臂最终均为 `consecutive_non_promotion_limit`、`incumbent_id=null`。

这是首份 clean、完整、可解释的 current-protocol development 结果，也是负学习结果。它排除了
“当前只差 API 恢复”与“离线 evaluator 不可用”这两个解释；真正瓶颈前移到了 candidate search：
协议允许每代最多 3 个 proposal，但该 run 每代只返回 1 个；通过静态审计的两个 recursive
candidate 又都收缩到 2/38 train support 和 1/16 prospective activation。继续放宽 promotion gate、
重复同一 root，或冻结空 archive 都不会接近目标。下一次架构工作应限定在 gate 之前：让 proposer
稳定给出多样化 roots，并把多个高精度、低覆盖的局部程序组成一个可审计 candidate configuration，
或用不读取 validation outcome 的 train-only coverage objective 选择可达到 prospective coverage 的
候选。evaluator、split、预算、sealed policy 与现有 promotion thresholds 不应随之改变。

**[ARTIFACT]** 2026-07-13 的 v3.10 fresh root
`paper_primary_v3_10_offline86_ruoli_gpt54mini_outer6_model1_diverse_plus01` 通过 clean lock 与
86/86 cache-only prewarm，随后完成 56/56 actual trials。事件账本为 56 start / 56 complete、
56 slot wait/acquire/release，最大在线 agent active=1；38/38 train 与 16/16 paired validation 全部
evaluation-valid，provider、circuit、infra、timeout、action/network budget、provider mismatch、
budget mismatch 和 invalid pair 都为 0。train 为 6 success / 32 residual。

第一代一次返回 exact 3 且 3/3 静态通过：poster root 覆盖 1 个 family、2/2 failure；template root
覆盖 2 个 family、3/6 precision；retrieval root 覆盖 2 个 family、6/6 precision，coverage-first
正确选择后者。它在 held-out 激活 2/16，candidate/raw 都为 3/16、0 gain、0 harm、effect/LCB=0、
cost ratio 0.948207，仍被既有 `insufficient_net_gain_count` 与
`paired_effect_lower_bound_below_target` 拒绝。第二代 recursive/no-recursive 各自收到一个 transport、
JSON、exact-count 均成功的三候选 response，但 host 事后计算发现每批三项只有一个 distinct
failed-train activation signature；旧 v3.10 semantic response contract 将两批都原子拒绝。两份 report
因此各有一次 proposal failure、`performance_claim_eligible=false`，两份 archive 字节相同、
`incumbent_id=null`；sealed/test 访问仍为 0。

更关键的非评分诊断推翻了“treatment 没执行”的表面解释。两个激活题上 policy-on/off 的 Codex
command trace 和实际答案均不同；financial candidate 还新增了 PUT/CALL 过滤，但 task success 仍为
0→0。当前 `selection_change_count` 比较的是 `selected_result.answer`，而 SkillLearn 把该字段投影为
`observation.success` 布尔，所以 0 只表示 pass/fail 未翻转，不能表示轨迹相同。compiler lowering v1
又在 `execute_step.value` 非空时丢弃 target，在 `check_condition` 也只保留 value；实际 agent skill
因此出现仅含 `{"mode": "evidence_join_then_compute"}` 的含糊步骤。与此同时训练失败 feedback
硬编码“explicit completion check”，直接把 proposal 引向表面完整性而非任务 operator。这些证据将
blocker 从 coverage/bundle 前移到 action representation；v3.11 先一次性修这一层，不新增评分 gate。

**[ARTIFACT]** 2026-07-13 的 v3.11 fresh root
`paper_primary_v3_11_offline86_ruoli_gpt54mini_outer6_model1_actionable_plus01` 绑定 scoped-clean
commit `a0ca50d8`，通过 86/86 cache-only prewarm（47 images / 7 verifier runtimes）。共享 train
执行 38 次且全部 evaluation-valid：5 success / 33 residual。一次 root proposal 原子返回 exact 3；
activation audit 为 3 个 candidate、3 个 distinct signature、group size `[1,1,1]`，0 error、0 retry。
poster root 因 train support=0 进入 repair；court 与 Mario roots 静态通过，分别为 3/3 与 2/2
failure activation precision，均只覆盖 1 个 train family，未达到既有 2-family coverage target，最终
court 因 support=3 入选。

no-recursive arm 运行 16 个 pair slot，只在 `court-form-filling-5` 激活一次。该 pair 的 treatment
确实执行：编译后的 skill 保留了 form-field scope、`/root/sc100-blank.pdf` 和
`/root/sc100-filled.pdf` 三个 target/value 指令；raw/candidate 分别产生 66/16 个 action start，
observation、treatment 和 PDF hash 均不同，两个外部离线 verifier 结果仍都为失败。因此总计
raw/candidate 均为 4/16，0 gain、0 harm、LCB=0、cost ratio=0.820789；这证明 actionability
修复改变了实际行为和成本，但尚未产生 task-success gain。另一个不激活 candidate 的
`anthropic-poster-design-2` raw/policy-off trial 使用 68,400,000 bytes，超过冻结的 67,108,864-byte
hard cap；容器按合同终止且 hard-budget retry 被抑制。该 pair 因而 invalid，report 以
`invalid_counterfactual_evidence` non-claim 停止，archive `incumbent_id=null`。这里不提高 fuse，
也不把 invalid raw row 拼接或改记为普通错误答案。

recursive arm 更早暴露一个与候选质量无关的 request-contract 缺陷。poster repair 的 provider
transport 与 JSON object 解析在 53.402 秒内成功，但 response 顶层唯一字段为 `hypotheses`；repair
envelope/parser 严格期待 singular `hypothesis`，因此 candidate-local `response_envelope` failure 阻止
整代 held-out validation/promotion。事后逐层复核 `ValidationContext -> RecursiveValidationEngine ->
StructuredHypothesisProposer.revise` 发现，真实 repair capabilities 已不含 `proposal_batch_contract`；
最初的“batch contract 泄漏”判断错误。实际缺口是 repair 只在 nested `output_schema` 中声明 singular
field，而共享 system contract 仅明确了 root batch 情况，没有在 versioned policy 存在时显式要求
one-object/`hypothesis`，模型仍返回了 root-shaped envelope。recursive report 以
`proposal_model_failure` non-claim 停止，archive 同样无 incumbent。这个结果把下一步限定为显式、
协议绑定的 singular repair response，而不是添加 gate、放宽 parser、增加 retry 或修改 evaluator。

**[ARTIFACT]** 2026-07-13 的 v3.12 fresh root
`paper_primary_v3_12_offline86_ruoli_gpt54mini_outer6_model1_repairscope01` 绑定 scoped-clean commit
`9c692b2d` 并通过 86/86 prewarm。development 的 56/56 actual external trials 全部 valid：38 train
policy-off、16 validation raw/policy-off、2 个实际激活的 policy-on；train 为 8 success / 30 residual，
最大 finalized network 为 35,070,000 bytes，0 provider、infrastructure、action、network、budget、
invalid-pair 或 mismatch。两臂均完成两代并以 `consecutive_non_promotion_limit` 停止。

G1/G2 均返回 exact 3、三种 distinct activation signature、3/3 static pass。三个候选分别覆盖
poster、poem、court 的互斥单一 family；TRAIN precision 都为 1、success false positive 都为 0，
coverage target=2 却没有单 root 达到。单体 selector 因 support=3 连续选择 court root。两代都只在
`court-form-filling-5` 激活：raw 为 9 action starts，两个 candidate 分别为 32/43，三者都失败。
因此每代都是 activation 1/16、raw/candidate 4/16、0 gain/0 harm、LCB=0，cost ratio 分别为
1.079038 与 1.116838。recursive/no-recursive 共享或重放同一证据，最终 archive 字节相同且
`incumbent_id=null`。全部 12 个 static node 直接通过，所以本轮 repair request=0；v3.12 singular
contract 已绑定并通过 bounded live canary，但 full development 没有提供 repair-path 样本。

**[ARTIFACT / QUARANTINE]** 旧 `all-development` 紧接着无条件执行 freeze/controls，产生
`frozen=true` 但 `selected_candidate_available=false` 的空 receipt；`promoted_v2` 与 no-rec control
均编译为空 program set/raw alias。进程被立即终止。partial validation 只留下 8 starts、2 completes、
2/96 records，且两条都因 `codex_action_budget_receipt_missing` invalid；没有 control report、family-out、
HippoRAG、sealed journal 或 test access。`validation.partial_admissibility.json` 已把该批标成
diagnostic-only、performance-claim inadmissible、row-reuse forbidden。它不反向污染此前完整有效的
development report/archive。随后增加的是单一 phase-transition invariant：无 promoted recursive
candidate 时 runner 正常结束，freeze producer fail closed，controls consumer 也拒绝旧空 receipt；
没有调整任何 promotion 分数或阈值。

**[ARTIFACT / PROPOSAL-ONLY]** v3.16 与 v3.17 在正式 benchmark 之前增加一次有界的
TRAIN-only feasibility screen，并复用受版本控制的 v3.15 source receipt。该路径重新从 38 份
policy-off result/trace/action receipt 与 public TRAIN environment 重建 38 observations（6 success / 32
failure）和 31 action profiles，`source_agent_trials_reexecuted=0`；不读取 v3.15 development report/events，
不访问 validation/test/verifier，也不构造 task backend 或 evaluator。

v3.16 的 3/3 singular family-slot model calls 均成功，但只通过 root count、profile binding 与 schema 三项；
distinct single-family signature、minimum support、anti-trigger self-block、executable delta、restatement absence
和 failed-primitive avoidance 六项失败。v3.17 随后只作一次结构修订：host 固定 exact family trigger 与空
anti-trigger，按冻结优先级为每个 slot 选一个 support≥2 的 reusable artifact，并给出
read→parse→update→serialize→write-back blueprint。新结果的 support 为 2/2/3，3/3 有 concrete local tool，
2/3 有 artifact manipulation，且 distinct signature、self-block、restatement、schema 等八项全部通过；
唯一失败是第三候选的 `failed_primitive_binding_count=2`。一次 transport `RemoteDisconnected` 在有界 retry
后恢复，最终仍为 3 个 logical success，因此整体失败不能归因于 credential tier 或 route outage。

两轮报告都明确 `backend_call_count=0`、`evaluator_call_count=0`、`validation_task_count=0`、sealed/test=false、
raw response/secret 未落盘。`failure_blocks_future_trial_spend_only=true`，既没有 benchmark row，也没有
promotion/archive。这个结果正确阻止了 preflight/lock/prewarm/smoke/development 支出，并冻结该 free-text
family-slot 路线为负结果；后续不再通过新增 prompt、acceptance predicate、retry、selector 或 promotion gate
继续追逐同一表示。

### 7.3 当前 infrastructure/protocol 状态

全 inventory 的
[`offline verifier coverage audit`](skilllearn_offline_verifier_matrix.md)
给出 credential-independent 任务 **86/95 可运行、9 项 blocked**。本次没有继续追逐
大体积、异构依赖，而是在任何新模型调用前冻结了保留原 split assignment 的 86-item
offline-ready subset：

- instance holdout：38 train / 16 validation / 32 sealed test，16 families；
- family out：48 train / 11 validation / 27 sealed test，9/2/5 families；
- 排除 3 个完整 infrastructure-blocked families、缺权威 verifier 的 GDP item 2，以及
  原先需要 `GH_TOKEN` 的 family；
- 不复用相邻 GDP item 的 oracle，也不把 online evaluator 当作替代品。

新的
[`offline86 verifier matrix`](../artifacts/offline_verifier_matrix_offline86_20260711_v1/matrix.json)
实际得到：7/7 active profiles、15/15 train-family representatives、
`blockers=[]`、`manifest_execution_ready=true`、`passed=true`。随后完整本地 preflight
同样得到 `blockers=[]`、`selected_item_count=86`、`ready_for_live_skill_generation=true`。
两次检查均未执行模型，sealed-test 语义也未暴露给模型。该结果另有受版本控制的精简
[`offline readiness receipt`](../manifests/skilllearn_offline_readiness_receipt_v1.json)，
供 protocol/lock 绑定；不再把 `.gitignore` 下的 matrix artifact 当作唯一证据入口。

这里必须区分三层证据。readiness receipt 绑定的是 7/7 profile contract、15/15 train-family
动态代表探针和 86-item 静态 preflight；它不声称逐项执行了 86 个 verifier。独立的
all-manifest runtime prewarm 才覆盖 train、validation、sealed-test 的全部 86 个 image/runtime。
本次第一次 cache-only 检查暴露 14 个未建镜像；它们在独立准备阶段有界构建后，第二次
cache-only 验收为 **86/86 passed、0 failed、47 个唯一镜像、7 个离线 verifier runtime**，
且最终 receipt 记录 `online_build_attempted=false`。这仍只是零模型、零 sealed scoring 的
基础设施证据，不是 86 项任务准确率。

9 项未进入主协议的原因已经分型，而不是统称“缺缓存”：GDP item 2 在当前官方主分支
仍缺权威 `test_outputs.py` 和 solution；Druid 已有零下载 direct-`javac` 参考 patch 路线，
但缺 vulnerable negative control 与 arbitrary-edit generality；Scala 还需要固定 SBT/Maven
闭包和 CLI verifier adapter；NLP 则需要 Python 3.10 CPU runtime 与约 0.5--1.2 GB 的
最小 ML closure。它们是后续独立 infrastructure workstream，不再阻塞主学习实验。

旧 development lock 仍声明 `network_scope_audit=v1`，v3.1 已升级为 hard-egress v2、
offline-verifier v3、32 MiB/题 hard fuse、prompt-action lowering v1 和 protocol-owned
promotion v2；v3.2 只把同一 fuse 版本化为 64 MiB。旧 live 与 v3.1 observations 因此
都只能作诊断，不能与 v3.2 直接合并。

最新 `gptpro03` protocol lock 绑定 clean commit `18ff3417`、`validation_issues=[]`，对应
prewarm 为 86/86。该 run 的 37 条 valid train observation 与 1 条 hard-cap invalid 也不能
被事后修补或跨进程 replay；它只证明 offline receipt 修复和 provider 稳定性，同时否证
当前 64 MiB execution contract 对所有 train trajectory 的可行性。

sealed test 仍未访问，这是正确状态。

## 八、v2 当前最关键的架构缺口

### 8.1 已关闭的 P0：promotion 标准所有权

**[CODE + TEST]** `PromotionGateSpec` 现在是唯一的 evaluator-owned contract。pairs、
confidence、net gain、activation、minimum effect LCB、maximum harm 和 maximum cost
全部由 [`PaperProtocol`](../assumption_agent/benchmarks/paper_protocol.py) 严格解析；实验 CLI
已移除 `--minimum-pairs` 旁路，recursive/no-recursive 两臂共享同一个 immutable spec，
protocol lock 和 freeze report 都复核完整 promotion mapping。

candidate 的 `ExpectedEffect` 仍可表达更保守的自我约束，但 effective threshold 只能收紧：

```text
effective_min_delta = max(protocol_min_delta, candidate_min_delta)
effective_max_harm = min(protocol_max_harm, candidate_max_harm)
effective_max_cost = min(protocol_max_cost, candidate_max_cost)
```

新增对抗测试证明 `minimum_delta=-1`、`maximum_harm_rate=1`、超大 cost ratio 都不能放宽
protocol 及格线；更严格 candidate 则会生效。promotion decision/event 同时记录 protocol、
candidate 和 effective thresholds。这是一次收敛现有判断所有权的修复，不是新增 gate。

### 8.2 已收紧的 P0：外部 action/verifier/fallback contract

内部 `PolicyRuntime` 的 typed lane action 仍保持 effectful。外部 SkillLearn backend 则不再
冒充同等语义：compiler 只接受 `execute_step`、`produce_artifact`、`request_evidence` 三类
`prompt_directive` 和 `check_condition` 这一类 `agent_local_self_check`。`enable_lane`、
`disable_lane`、`prioritize_lane`、`set_parameter`、`require_verifier`、`abstain` 没有外部
lowering，因而 fail closed；proposal/repair capabilities 也只广告这四类。

编译后的 `SKILL.md` 不再暴露 benchmark external verifier、policy-off/on evidence 或
expected-effect 阈值，并明确 external verifier 只在 agent 退出后运行。compile manifest
绑定完整 program set、每项 lowered treatment、实际 `SKILL.md` content hash、
`external_verifier_exposed_to_agent=false` 和 `baseline_on_nonactivation_only_v1`。输出目录
由 staging tree 整体替换，旧 skill 不会残留；action target/value 引用 hidden verifier、
required evidence 或 policy-off/on 时在 validator/compiler 共用的结构检查中 fail closed。

novelty、counterfactual replay、training replay 和 proposal prior context 现在统一使用实际
lowered-treatment identity，而不是 raw program metadata。只改 expected-effect/verifier 元数据
不能获得新行为身份或重新采样；真正改变注入内容才会改变 treatment hash。generation report
同时绑定实际评价的 candidate treatment；freeze 使用 archive program 重算该 hash，并用与
runtime gate 相同的 summary-blocker 函数重算 promotion，防止替换 treatment 或伪造 allowed。

fallback 的伪证据链也已删除：activated candidate 是独立 treatment，
`baseline_preserved=false`；只有 trigger miss 时 candidate observation 直接 alias baseline，
才记为 observed baseline。SkillLearn promotion 依据冻结的 paired harm/LCB/cost contract，
不再要求一个由字符串声明伪造的 post-verifier rollback。

这没有把通用 production prompt directive 变成 typed program；它做的是把宏观“注入 skill”与细粒度
agent instruction 的证据层级说清楚。后续独立 financial measurement path 已实际执行一个 bounded、
post-agent typed semantic operator，并出现一个 fresh treatment-associated 0→1；但它尚未接回通用
production evolution，也没有多 family 或 retained-benefit 证据。因此不能把这次局部 vertical slice
外推成主路径已经具有强类型 operator 语义。

### 8.3 P1：failure-only support bias 已关闭，但 precision-first 收缩为 coverage starvation

v3.5 的真实第一代已经给出反例：最大 failure-support 的 recursive repair 在 held-out
validation 造成 0 gain / 2 harm，而较窄的 no-recursive root 为 1 gain / 0 harm。v3.6
因此没有放宽 promotion gate，而是改变 gate 之前的 train-only proposal selection：

- 每个 valid train failure 仍形成带 sanitized failure context 的正例；
- 每个 valid train success 形成 `baseline_success_control`，只含 runtime features 与 label，
  不含 instruction、feedback 或 execution context；
- root/repair/replay request 与双臂第一代 checkpoint 绑定全部 labeled transition IDs；
- static support pass 仍只由 failed rows 决定，success rows 不会把无支持候选洗成通过；
- 同代候选按精确 `failure activations / all train activations`、success false positives、
  failure support、predicate/action complexity 与 payload hash 排序，不读取 validation。

held-out report 另增加 evidence-valid activation、activated gain/harm、precision、harm rate 与
abstention。其分母排除 evaluator-invalid、provider mismatch 与 budget mismatch 的并集；零
valid activation 时 ratio 为 `null` 且 `defined=false`。这些字段是诊断，不进入既有
`PromotionGateSpec`。v3.9 clean development 已验证这套 selection 不再选择 success false
positive，却暴露相反失败模式：两代入选 candidate 均为 train support 2/38、held-out activation
1/16，0 gain/0 harm。precision-first 排序把搜索收缩成了局部 family policy，无法形成足以检验或
晋级的 prospective coverage。下一步应改变 proposer/search/configuration formation，而不是再给
promotion 增加 blocker 或放宽 minimum activation。

v3.10 已完成这个 coverage 假设的 live 检验：exact-three 机制一次给出 3 个静态可执行 root，
coverage-first 从 1-family/2-failure 候选转而选择 2-family/6-failure 且 0 success false positive 的
retrieval root；held-out activation 也确实从 v3.9 的 1/16 提高到 2/16。因此“只要覆盖更宽就会出现
收益”被否定：candidate/raw 仍同为 3/16，0 gain/0 harm。此时继续扩大 bundle 只会扩大含糊
directive 的覆盖面，尚无依据。轨迹和 compiled skill 复核把下一 blocker 定位为 operator actionability：
training feedback 强推 completion check，lowering 又丢 target 并输出 mode JSON。v3.11 因此保留
coverage objective，只修训练提示、action schema 与 agent-facing lowering。

v3.11 live 已把这个假设拆成两部分。action representation 部分通过：三项 root directive 都是
可读的 target/value 指令，入选 court treatment 在 held-out 激活题上把 action starts 从 66 降到 16，
并生成不同 PDF；因此不能再把 0 gain 解释为 skill 未注入或 lowering 未执行。但 task-success 仍为
0→0，说明单个局部 directive 尚未解决缺失日期/checkbox 等真实 operator 细节。另一方面，recursive
arm 没有检验到 repair quality，因为 generic response contract 未显式绑定 singular repair，模型返回
了 root-shaped batch envelope。v3.12 只显式版本化这一 response scope；它不是新的 performance gate。

v3.12 fresh root 随后完整结束，但没有进入 repair：两代各 3 个 root、共 12 个 recursive/no-recursive
static node 全部通过，`hypothesis_repair_requested=0`。因此这轮能说明 singular contract 没有破坏正常
development，却不能说明 live repair quality 已改善，也不应通过人为收紧 static check 来强迫 repair。
更重要的新证据来自 candidate formation：G1/G2 的 poster、poem、court roots 都是 TRAIN precision=1、
success false positive=0，但每个只覆盖一个 family；selector 连续只保留 support=3 的 court root，
把另两个互补、高精度局部 policy 丢到 shadow。held-out 因而仍只有 1/16 activation、0 gain/0 harm。
v3.13 已把 proposal diversity 形成的互补 roots 用冻结的 TRAIN-only subset objective 组合成单一
program-set treatment，并规定只做一次 paired validation；它不能逐个试 validation 后挑最好，也没有
增加或放宽 promotion gate。live 证明 set-level routing/replay 按设计执行：每套 selected bundle 都是
5/5 failure support、2 families、0/8 success FP，held-out 激活也从单一 family 扩为 poster+court 的
2/16。但 6 个真实 policy-on 全部失败，故它否定的是“精确命中 baseline failure 足以预测 action utility”。
no-recursive G2 还有一个 7/7、3-family、0-FP 的三成员 subset，却因为 family deficit 在 2 后封顶且
`bundle_size_asc` 早于 `failure_support_desc` 而只排第四。v3.14 已一次性修这一 TRAIN-only objective
和跨臂 baseline replay：leading precision/capped-deficit/success-FP/overlap 顺序不变，只在这些项相同
时把实际 family count 与 failure support 提到 bundle size 前。live G1 确实选中 7/7、3-family、0-FP
三成员 set，并把 activation 从 2/16 提到 3/16；这证明 selector 改动生效。结果仍是 6 个 G1 on 与
1 个 no-recursive G2 on 全失败、0 gain/0 harm，所以预先声明的停止条件已经满足：不再迭代 selector，
action quality 单列为下一研究问题。valid policy-off evidence 产生 31 次跨臂/多代 replay；但一条 64 MiB
hard-cap invalid 没有形成 terminal memo，随后被另一 arm 以相同 baseline evidence key 重执行。因此
shared valid cohort 有效，terminal invalid 传播仍不完整；它是 attribution correctness，不是新评分 gate。

v3.15 在提交 `696a2954` 只修这两个 gate 前/归因层缺口。`train_only_material_action_delta_prompt_audit_v1`
把 task instruction 标为 baseline requirement，而不是候选自身的 treatment；request-local prompt 要求每个
hypothesis 至少提出 exact constant/mapping、concrete local tool command 或 artifact-internal manipulation
中的一种 material delta，并允许模型使用其静态知识。它的证据面严格限于 TRAIN failure：public
`environment/` 被归一化为 allowlisted package、task-local path、file/operation labels，policy-off trace
只保留 allowlisted executable、flag、`/root` path、成功/失败和 file-change facts。symlink escape、credential/
network syntax、敏感路径、test/verifier/oracle/solution 引用整行丢弃；model prose、command output、raw trace/
environment 文本均不进入 proposal profile。成功 TRAIN control 仍无 instruction/context，validation outcome、
verifier/test content 均不可见，proposal 也没有外部 tool、网络或 runtime install 权限。

`proposal_action_delta_audited` 只回答候选是否看起来提供 material delta、是否有 instruction-restatement 风险。
即使 audit 输入异常，它也不能 reject/重试 response、触发 recursive repair、改变 candidate selection 或进入
promotion decision；因此这是非评分诊断，不是隐藏 gate。完整 profile 通过 hash 引用，首代 checkpoint、
profile count/set hash 写入 recursive/no-recursive report；freeze 同时核验 plan 与 generation、两臂共享值，
并写入 receipt，防止事后替换 action-design cohort。

`behavior_identical_shared_validation_baseline_terminal_outcome_replay_v3` 则在同一冻结 request 已走完或被
明确抑制的 same-request retry 后，把最终 invalid 记为 run-scoped immutable tombstone。key 绑定 baseline
execution/fairness 与 retry policy identity；后续 arm/generation 复用同一 invalid、增加 0 次 baseline execution，
但 tombstone 永远不是 promotion evidence，pair 仍 terminal non-claim，冲突也不覆写 first outcome。
v1/v2 历史语义不变。以上机制在 453/453 离线测试通过后进入正式 live root：clean lock、86/86
cache-only prewarm 和 smoke 均通过，full development 完成 57/57 valid actual trials（38 TRAIN off、
16 个共享 validation baseline、3 个 activated on）与 8/8 proposal/repair model calls。TRAIN 为
6 success / 32 residual；最大 online-agent concurrency=1；provider、infrastructure、action-budget、
network-cap 与 pair-mismatch failure 均为 0。recursive G1/G2 各只激活 1/16，candidate/raw 都是
4/16、0 gain/0 harm；no-recursive G1 在 TRAIN static audit 被拒，G2 的 1/16 activation 同样
0 gain/0 harm。共享 valid cohort 支持 32 次 zero-execution baseline replay。因本轮无 invalid baseline，
terminal-invalid tombstone 没有被 live 触发，其 retry-identity/非评分语义仍只由离线测试证明。

两臂 report 都是 claim-eligible，但这只表示该 clean negative result 可用于诊断，不表示 promotion：
两臂均以 `consecutive_non_promotion_limit` 停止，archive 的 `incumbent_id` 都是 `null`，sealed/test=false，
freeze、controls、family-out、HippoRAG 与 sealed 均未运行。action audit 给出更直接的失败定位：13 个
candidate（9 roots + 4 repairs）中 7 个被检测为有 material delta，6 个有 restatement risk；然而 7 个
material delta 全部只有 `exact_constant_or_mapping`，concrete local tool、artifact manipulation 和
environment primitive 都是 0。更严重的是 9 个 roots 全部指向 `anthropic-poster`，候选形成从 v3.14 G1
的 3-family/7-support 倒退为单-family/2-support。由此可见 action-quality prompt/profile 的 wiring 生效，
但 proposal 仍没有把 profile grounding 转成多 family、可执行 operator；下一改动必须留在 gate 前。

v3.16/v3.17 已把这个 gate 前假设做成独立 proposal-only 试验，而不是再花一轮 38+16 task trials。
结构化 family slot 成功消除了 root family collapse；v3.17 的 exact trigger、空 anti-trigger 和 artifact
blueprint 又消除了 support=0、self-block 与 restatement，并使 3/3 candidate 具有 concrete local tool。
所以 diversity、support 和表面 actionability 已不再是最后 blocker。唯一失败落在第三候选对两个 failed
TRAIN primitives 的绑定。

这里的更深问题不是再加一条“不要使用失败 primitive”的 prompt。第三个 slot 的本地 failed set 是通用
executable `file` 与 `python`；它们来自失败 command 的共现，而不是证明这些工具本身导致失败。自由文本
action 又很自然会在“解析文件”工作流中提到 file/Python。因此当前 taxonomy 同时混合了“在失败轨迹中出现”
与“因果上应禁止”，而模型侧只收到 count/hash，也不可能可靠满足未知的逐值 exclusion。

新的前置结构复核已从 38 条 receipt-bound V3.15 TRAIN trace 中保留 429 个按时间排序的 allowlisted command
occurrences，其中 70 个失败；另有 208 个 non-allowlisted command 被显式计数后丢弃。38 条 trace 均完整，
总 relevant action starts 为 655，单 trace 最大 61，低于冻结的 100-action budget；失败 span 没有一个是最后
allowlisted span，且 observational inadmissibility 计数为 0。这些数字只证明 allowlisted chronology 未去重、
未截断，不声称覆盖完整 raw-command chronology。closed typed graph 已把 proposal output 收窄为一个 opaque、
已注册的 `recipe_id`，使 primitive、locator 与 free-text action 不进入模型输出域；但 materialization 仍经现有
prompt-directive/self-check compiler 交给通用 agent，capability implementation 未验证，也没有 restricted
runtime executor。正式 operator feasibility 已 9/9 PASS；production integration v1 的 live smoke 继而发现
ledger-binding construction-order bug，修复后的 v2 通过 13/13 predicates、12/12 tamper probes 与 exact replay。
v3.18r1 也已完成真实 development，但其 action receipt 仍止于安装/路由，且暴露 task input/dependency 与
literal locator 问题。继续补 prompt 或放宽 acceptance 都会掩盖这一边界；下一步是 task closure 与
capability-backed portable role，不是新的 gate patch。

### 8.4 P1：prospective runtime features 仍过粗

当前 SkillLearn feature catalog 主要只有
[`family/category/difficulty/tags/environment_file_count/has_container_environment`](../assumption_agent/benchmarks/skilllearnbench.py)。
完整 train instruction 与 v3.15 的 environment/action-trace profile 可以帮助设计 action，但都被正确禁止
作为 runtime trigger。v3.15 live 的九个 roots 仍全部坍缩到 `anthropic-poster`，也说明“提供 profile”本身
不会自动产生结构化 family diversity 或新的 prospective routing feature。

这比 legacy hash-only data 有进步，却仍缺：

- artifact/output schema；
- constraint signature；
- required capability；
- verification class；
- input modality与工具需求；
- 可在未见 family 上计算的语义特征。

当前很多 candidate 因此只能形成宽泛 family/env-count gate，不足以支撑可信 family-out
prospective routing。

### 8.5 P2：archive 是树形数据结构，不是多 clade 搜索算法

[`ArchiveNode`](../assumption_agent/archive.py#L23-L31) 有 parent、active programs、epoch
和 generation；但 [`PolicyArchive`](../assumption_agent/archive.py#L96-L102) 只有一个
`incumbent_id`。v3.12 及以前每代只把 train-only 排名第一的 `eligible[0]` 送入 validation；
v3.13-v3.15 虽可把多个互补 roots 组成同一 candidate node，但仍只选择一个 program set、只从当前
incumbent 扩展：
[`evolution.py:L325-L453`](../assumption_agent/evolution.py#L325-L453)。

因此 active 算法仍是保守的单 incumbent configuration hill climbing，而不是 RQGM 中多 clade 并行保留、
按 metaproductivity 继续扩展的 archive search。

此外，`ScoreRecord` 只存 candidate successes/total 和 item-set hash，未直接绑定完整 pair
bundle、gain/harm/cost、promotion decision 与 protocol hash。archive 的 provenance 还不够
承担跨 epoch、多分支重排。

### 8.6 P2：evaluator co-evolution 还是独立骨架

`EvaluatorEpochController`、anchor lower bound 和 selective invalidation 有代码与测试；
但主 SkillLearn 实验明确只允许 task/policy hypothesis，evaluator hypothesis 不能编译为
agent skill。当前没有真实 evaluator challenger、epoch transition 或 incumbent re-ranking
artifact。

所以 v2 可以声称“有 evaluator-epoch mechanism skeleton”，不能声称“已经实现 Red Queen
式 agent/evaluator co-evolution”。

### 8.7 P1：递归修复被触发过，但没有因果收益证据

v2 recursive validation 主要修复 schema、trigger support、action vocabulary 和 epoch 等
静态/训练检查。v3.9 已得到完整 clean 对照：recursive 第一代的 depth-1 repair 通过静态审计并在
held-out 真正激活 1/16，但相对 raw 为 0 gain/0 harm；no-recursive 同代 root 因 anti-scope
support 未通过静态审计。该对照证明 repair 能把 candidate 从静态失败变成可执行 treatment，却
没有证明它改善 task success，而且激活范围过窄，无法形成 retained incumbent。

v3.10 第一代 3/3 roots 均直接通过静态检查，没有触发 repair；第二代又在 root response contract
处终止。recursive/no-recursive archive 因而字节相同，report 只在 arm/trace/path provenance 上不同。
这轮没有提供新的 repair 因果样本，也不能把两臂相同解释成 repair 无效。

v3.18r1 的 recursive archive 确实比 no-recursive 多出 4 个 repair binding，但最终被验证的 treatment、
nodes 与 score records 完全相同。更关键的是 stock G2 记录 `parent_recipe_id=null`、`failed_checks=[]`、
`repair_depth=0`：它只是排除 G1 recipe 后选择另一个 root，并未把 G1 的 59-row、tooltip/click 或浏览器
residual 编译成 child repair。故本轮只能说明 recursive machinery 扩大了搜索/archive，不能说明它把
上一代执行反馈转化为因果修复。

因此当前可说“递归修复机制会运行并改变候选可执行性”，不能说“递归验证已经改善性能”。

### 8.8 文档与协议漂移正在本次收口

[`ARCHITECTURE.md`](../ARCHITECTURE.md) 和
[`BENCHMARK_PROTOCOL.md`](../BENCHMARK_PROTOCOL.md) 此前曾有段落声称 destination allowlist /
dependency-cache-only 尚未强制；但当前
[`docker_egress.py`](../assumption_agent/benchmarks/docker_egress.py) 和 protocol manifest 已
实现 provider-only hard egress、offline package mode 与 network fuse。本次已同步主
README、benchmark protocol、offline-verifier matrix 和 status 摘要；本轮又把 receipt
runtime provenance、v3.5 serial execution-policy / repair identity / response-contract binding、v3.6 contrastive/invalid-evidence contract、v3.7-v3.9 并发容量/共享 slot、v3.10/v3.11 live 结果，以及 v3.12 singular repair scope、clean 负结果与 empty-incumbent phase prerequisite。历史段落仍
保留为 diagnostic ledger，不能当作当前协议。2026-07-14 又同步 v3.13 clean negative evidence、
v3.14 的 mixed-claim live 结果，以及由此确定的 action-quality 转向；没有把 no-recursive 的机械
claim eligibility 写成 primary performance improvement。当前再同步 v3.15 提交 `696a2954` 的
TRAIN-only action profile、audit-only 边界、terminal-invalid memo/retry identity 与 paired report/freeze
provenance，以及随后 clean 57/57-valid live negative evidence。两臂 claim eligibility 只表明证据可用，
没有被写成 promotion 或 performance improvement。

这种文档漂移本身会破坏 protocol review；重新跑论文实验前必须同步。

### 8.9 P0 已执行：v3.20 clean negative 把断点定位到 sidecar 消费

formal integration v1 的离线 predicates 曾全部通过，但第一次真实 smoke 在 task trial 前暴露
`typed snapshot ledger binding is missing`：harness 在把 ledger 绑定到共享 proposer registry 之前就构造
compiler/runner，后续 `EvolutionKernel` 的绑定永远不可达。提交 `8caba466` 把 binding 移到 auth validation
之后、compiler/runner 之前，并以真实 live-harness construction regression 覆盖。另行冻结的 v2 此后
13/13 predicates、12/12 tamper probes 与 exact replay 全过，0 live model/backend/evaluator call。这个过程
说明非评分 integration diagnostic 有价值，但也明确了它的边界：selection/provenance pass 不等于 task
semantics pass。

v3.18r1 是第一轮真实使用这条 production path 的 development。38/38 TRAIN、16/16 logical pairs、
58 actual trials、12/12 model attempts 均在 38 个 item workers / 48 shared model slots 下完成，0 provider、
budget、pair mismatch。recursive 与 no-recursive 每代都选中 stock + organize，activation=2/16，二元
score 都是 3/16 对 3/16；G1/G2 平均 action cost 分别从 15.8125 变为 15.75/15.0。两臂均停止于
consecutive non-promotion，`incumbent_id=null`，因此 freeze、controls、HippoRAG、family-out、sealed
均正确未运行。

非评分 action-span/CTRF 审计否定了“0 gain/0 harm 即行为相同”：

1. **task input closure 失败。** `organize-messy-files-3` 的 Dockerfile 声明下载 100 PDF 后再复制 3 个
   Office 文件；实际 content-addressed image 的下载 heredoc layer 为 0B，初始 `/root/papers/all` 仅有
   3 个 Office 文件。确定根因是本机 Docker 29.1.3 没有 `buildx`，`docker build` 落到 deprecated legacy
   builder；它把 BuildKit-only `RUN <<'EOF'` 当成无命令 here-doc 重定向，body 中的 `wget`、count assertion
   乃至 `false` 都不执行且返回 0。RAW/G1/G2 最终结构相同、均为 4/6 tests，失败只来自缺失 100 PDF。
   相同模式影响全部 6 个 organize task instance（5 个 unique cached images）；其中 3 个又构成当前 selected
   TRAIN support 5 的多数。prewarm 的 86/86
   只证明 image/runtime/verifier 可启动，没有证明 task input inventory 完整。
2. **离线依赖 closure 失败。** stock 输入本身完整：50 个 logical TSV records、50 个 history CSV、
   5 sectors，三路 data-integrity test 都通过；但题目要求 D3 v6，task image 没有 vendored D3，official
   solution 原本运行时 `npm install d3@6.7.0`，而冻结协议正确禁止 runtime install。三路只能手写 shim。
   下一次离线运行应预下载并 checksum 固化 D3 6.7.0，而不是切到在线 evaluator。
3. **portable role binding 失败。** G1/G2 recipe 把 TRAIN `stock-data-visualization-5` 的
   `/root/data/stock-descriptions.csv` 带到 validation item 3，但后者声明 `.tsv`；organize recipe 同样绑定
   不存在或实例不匹配的单一文档路径。closed grammar 只让模型不能生成 locator，没有把 concrete locator
   提升为 `primary_table` / `input_document_collection` 等由当前 item 解析的 artifact role。
4. **effect semantics 失败。** event 证明 recipe 被编译、安装到 7 个目的地并命中路由，却全部记录
   `fine_grained_action_receipt_available=false`。trace 没有留下 agent 显式读取 `SKILL.md` 的可审计证据，
   也未表现为遵循错误的 `.csv` 指令；
   三条 stock 轨迹共同表现为检查数据、搜索 D3、手写 shim、只做语法检查。prompt materialization 不是
   restricted capability execution，`action_activated=true` 只能解释为 treatment delivered。
5. **搜索反馈被压扁。** stock RAW 实际通过 8/10 tests，只剩 tooltip/click 的共同 pointer-event residual；
   G1 误把 60 个物理行当 59 条数据而降到 4/10，G2 浏览器在 `parentNode` 报错、降到 3/10。全或无
   `task_success` 把 8→4→3 都投影成 0。G2 又是 `parent_recipe_id=null`、`failed_checks=[]` 的另一个 root，
   并非吸收 G1 residual 的 repair child。

因此下一 workstream 不再增加 recipe prompt、selector 或 promotion blocker。先恢复并哈希 task
input/dependency closure；随后只做一条 bounded vertical slice：

`artifact role → current-item resolver → restricted task-local capability → effect receipt`

TRAIN/search 侧可使用不泄露 verifier 内容的离线 per-test status vector，至少保留 8→4→3 的退化并把
上一代 residual 绑定到非空 parent/failed-check child；最终 promotion 仍使用原冻结 task-success contract。

2026-07-14 已完成前半段底座修复。125 个去重 PDF 与 D3 6.7.0 bundle 被放入本地
content-addressed cache；随后 cache-only v2 preparation 以 0 下载冻结 11 项 closure / 126 个实际注入对象。
第一次实现审查又发现 closure 未绑定源 Dockerfile、以及 WSL `/mnt/c` staging 会把 0644 mode 变为 0777；
最终 manifest v2 因而绑定不含 mode、但包含 path/kind/size/content 的 transport-stable source-environment
hash，并在原始 environment 与改写前 staging 各验证一次。protocol v3.19 进一步冻结 preparation file、
receipt、per-item ledger、object set 与 source-environment set，development、controls 和 sealed runtime 都只能
复用 v5 prewarm 中的精确 image ID。

11/11 affected images 已重建；`organize-messy-files-3` 的无网络实测 inventory 为 103 files = 100 PDF +
2 DOCX + 1 PPTX，stock image 内 D3 header 为 v6.7.0。随后 86-worker cache-only v5 prewarm 为 86/86 PASS，
其中 closure-required/verified=11/11、0 online build、0 model/evaluator、0 sealed scoring。

后半段的独立离线 feasibility 先行跑通：冻结 stock recipe 在当前 item 3 上不使用 TRAIN `.csv` locator，
只从 public instruction 唯一解析当前 `.tsv`，固定只读 capability 真实解析 50×14，并产生 input pre/post、
output 与 role binding 的 exact effect receipt。随后唯一一次正式 typed-portable 非评分 integration 已把
同一 role contract 接到 production compiler/runtime 的 agent-start 前 hook：三项真实 Docker canary 在
`--network none` 下生成并回读只读 evidence profile/inventory sidecar，exact image、production v3.20
authorization loader 与 cleanup 均通过；model/task-backend/evaluator/verifier/score/promotion call 均为 0。
exact replay 不产生新 selector call，decision hash 为
`a151ca52916101f0ea31b0d2f11c8fde8407f4410d175b1ac983e013d6e7957e`。

这个 PASS 的边界必须按 receipt 原样保留：pre-agent sidecar 是只读输入证据，不是 task effect，也不覆盖
完整 recipe operator plan。write、render、move 等非 access operators 仍是固定 agent plan，不能声称
它们已由 restricted capability 执行。该 PASS 授权的 fresh v3.20 development 现已执行完毕。

v3.20 的 lock 与 86/86 cache-only prewarm 先行通过；正式轮以 38 个 item workers / 48 model
slots 完成 61 attempts，最终为 38 TRAIN baseline + 16 shared validation baseline + 6 policy-on
= 60 valid trials。全部评价与 verifier 在本地离线完成；60/60 action-budget/verifier receipts
有效，最大 73/100 actions，无 web/remote tool/runtime install 或 sealed/test 访问。两代均为
baseline/candidate 5/16、activation 3/16、0 gain/0 harm，且都被
`insufficient_net_gain_count` 拒绝。终止原因为 `consecutive_non_promotion_limit`，两臂
archive 都没有 incumbent。recursive/no-recursive archive 字节相同，原因是两代的
`repaired_candidate_count=0`，而不是已证明 recursion 无用。

独立非评分审计进一步排除了 selector 卡死：G1/G2 选中不同 recipe，compile manifest、
treatment、action count 与 verifier 子测试结果均发生变化。然而 6 份 policy-on trace 都没有
读取 `.assumption-v2/capabilities/portable-*.json`，所有 pair 也都标记
`fine_grained_action_receipt_available=false`。所以 **v3.20 sidecar 路线在该轮的** 因果断点不是“缺一个 gate”，而是
“生成的 capability profile 没有被下游任务动作强制消费”。同类 read-only-sidecar +
prompt-directive recipe 的 development 搜索到此停止；不新增 recipe prompt、selector 或评分 gate。

### 8.10 P0 已执行：profile 已进入 launch prompt，剩余断点转为 task-local execution semantics

runtime-delivery integration 解决的是 launch wiring，不是 task utility。为避免直接消耗新的无偏
split，2026-07-15 另行预注册了一个严格限额的 consumed-development diagnostic：只运行既有
v3.20 G1/G2 在三个已消费 validation item 上的 6 个新 policy-on trial，不运行新的 policy-off、
proposal 或 training model call，不 retry。六个独立 backend 由 6 workers 同时调度，实际 agent
launch=6、峰值并发=6；6/6 observation valid，全部使用 offline post-agent verifier，且 runtime
profile effect、launch fragment、effective prompt、request/source/typed binding receipt 全部闭合。

正式结果为 G1=0/3、G2=0/3；相对同代未注入 prompt 的冻结历史 observation，delivery delta
signal=0/6；相对冻结 RAW，utility signal=0/6。该 diagnostic 明确
`fresh_validation=false`、`claim_eligible=false`、`semantic_profile_consumption_claimed=false`、
`task_utility_causal_claimed=false`，没有 incumbent、promotion、test trial 或 sealed scoring。运行只
检查了 test infrastructure metadata，没有读取 test task-input bytes，也没有向模型暴露；HippoRAG
没有同构 executable file-task arm，因此本次只用已冻结 RAW 作比较。

这排除了“verified profile 根本没有进入 launch input”这一解释，但不能证明模型语义性读取、遵循或
把 profile 转化为因果 task action。尤其 validation 在预注册前已经消费，本轮又没有同时随机化的新
policy-off；所以 0/6 不是 clean causal null efficacy，只能支持一个 bounded stopping decision：继续
把同一份 schema/inventory profile 更牢地塞进 prompt，不值得直接消耗新的无偏 split。

local-only、非评分的 CTRF/trace 解释进一步缩小了设计缺口，但不属于 formal performance claim：
六份 trace 都没有打开注入的 `portable-*.json`；只有 temperature G2 另行读取过 frozen `SKILL.md`，
其中虽列出 profile locator，后续仍未打开 sidecar。因此 launch receipt 不能替代 semantic-consumption
证据。

| family | G1 / G2 离线子测试 | 精确 residual | 设计含义 |
|---|---:|---|---|
| stock visualization | 9/10、9/10 | 都只剩 non-ETF tooltip interaction | 需要可重放的 interaction/postcondition contract，而不只是表头 profile |
| temperature simulation | 4/7、6/7 | G2 的 final output 已过参数和 RMSE 阈值，只因 `metrics.json` 的自评语义与离线重算不一致 | 需要 final-artifact single-source metric recomputation，而不是 search-time 近似 |
| organize files | 3/6、4/6 | G2 只剩错误分类，G1 另有缺件与 source completion 问题 | 需要 exhaustive mapping、one-to-one move reconciliation 与 completion audit |

共同点不是“还缺一个 promotion gate”，而是当前 profile 主要给出 schema/inventory facts，没有表达
决定成败的 task-local 操作不变量、postcondition 和 self-evaluation semantics。另一个关键资源反例是
temperature G2：66 个 action starts 却消耗 3,000,056 tokens 和 1,763.18 秒，local trace 中一个 shell
action 内调度/尝试 140-point grid search（不能证明 140 点全部完成）。action-start count 不能约束 action-span 内部计算；下一表示应让
typed operator 声明有限候选集、搜索空间 hash 与实际 evaluation count receipt，而不是再叠一个评分 gate。

### 8.11 P0 已执行：execution-contract TRAIN ranking 出现阳性，但证据仍受 in-sample 与 post-selection 限制

closed execution-contract 的候选搜索已经真正接入 production compile 与离线 outcome ranking，而不再停留在
prompt/profile delivery。正式非评分 integration 从 v3.20 的 38 条冻结 TRAIN evidence（9 success / 29
failure）重建 14 个历史候选、6 个 typed programs，以 14 个 compile workers 形成 14×38=532 个完整
candidate/item outcome：56 个 route-matched active execution 与 476 个 exact frozen-RAW replay。integration
本身的 model、evaluator、online judge、network call 均为 0，validation/test content access=false，且不授权
freeze、promotion 或 scoring。

正式 active execution 首轮使用 Plus，以 56 outer workers / 48 model slots 最大化题级调度，实测 source
model 峰值并发为 34。56 路中 55 路有效；唯一异常是 candidate `4033a94b…dedabf` 在
`temperature-simulation-2` 上收到明确的 provider model-capacity terminal，而不是 budget、web、verifier 或
route drift。1356-event source ledger 经 envelope/hash/identity 复核后恢复 55 份有效 observation；随后只用
Pro 对该 exact request 补跑 1 路，得到 evaluation-valid task failure，0 recovery / 0 regression。其余 55 路
没有重跑。因为最初 backend instance hashes 只存在进程内，不能声称恢复了原 run 的 bit-exact ranking hash；
最终 report 对此显式标记 `source_bit_exact_original_ranking_hash_recoverable=false`，并另行冻结排除 backend
identity 的 semantic ranking。

最终 grid 为 56/56 active valid + 476 replay，online judge=0、network fallback=false、
validation/test=false、promotion gate=false。排名首位是 `72c5ea9e…cd295`：总 recovery=1、regression=0，
唯一翻转为 `organize-messy-files-2` 的冻结 RAW failure → candidate success；第二次补跑的 `4033…` 仍为
valid failure。ranking hash 为 `2ec01860…db33`，semantic ranking hash 为 `58b35d34…82e0`。这证明新的
search object 至少不再是全零，但**不能直接建立 incumbent**。

关键原因是 source leakage audit：56 个 active route 的 candidate graph/contract source 都包含当前被评价的
TRAIN item，strict leave-item-out active outcome=0。换言之，`72c5…` 的 +1/0 是 in-sample fit；它既不是
unseen transfer，也不是 clean L2 validation learning。把它直接 freeze、送 controls、family-out、HippoRAG
或 sealed，都会把候选形成证据与效用检验混写。

因此只做了最低成本的 falsification，而没有增加 gate 或重跑全 grid：针对阳性所在的
`organize-messy-files-2`，从 graph 与 execution contract 中同时排除该题，只用 organize-5/-6 两条独立
source，保持 `minimum_independent_support=2`、`maximum_registered_artifacts_per_family=6`，编译
`organize_collection` recipe `recipe_0443d11a27ce50690356` 为新 candidate
`a34f06d0…52634`。非评分 compile 预注册为 1 个 active held-out execution + 37 个 frozen-RAW replay，
model/evaluator/network/online-judge call=0、validation/test access=false；post-run compile audit exact replay
hash 为 `0cb50293…deb74`。

该单路 Plus actual 已完成：`organize-messy-files-2` 从 baseline=false 变为 candidate=true，
evaluation-valid，14/100 actions、182.488 秒；完整 ranking 为 1 active + 37 replay，ranking hash
`2e070ea0…ea10`、active run receipt `a728245b…f5516`、最终 report hash `849bfbc6…b58ec`。actual
worker_state 保留了 Codex trace 与 verifier 原始文件，报告也明确
`raw_candidate_trial_artifacts_persisted=true`。

但这个阳性只能解释为 **post-selection targeted item-out refit/falsification**：item 与 workflow 都是在看过
source TRAIN ranking 后选择，workflow 也没有在不看 heldout 的条件下重新选择。正式 report 因而冻结
`fold_and_workflow_selected_post_source_ranking=true`、`unbiased_crossfit=false`、
`single_fold_incumbent_authorized=false`。它说明原来的 in-sample recovery 在移除本题 graph/contract
evidence 后没有立刻消失，但不估计无偏泛化率。

为避免把单个成功题反复包装成证据，同时也不新增 gate，随后一次性补齐同一 organize family 的另外两个
item-out fold。每个 fold 都保持同一 `organize_collection` workflow、support=2、max artifacts=6，graph 与
contract 同时排除 heldout，只用另外两题；fold 的 exact graph/snapshot/contract/candidate/work hashes 在 actual
前固定。organize-5 与 organize-6 以两路 Plus 并发启动，均无 retry、provider anomaly 或 online judge：

- organize-5：candidate `3fd2c792…114b`，17/100 actions、236,454 tokens，offline-valid false→false，
  ranking=`33cd43f4…47f1`，report=`3201ca6c…6fd8`；
- organize-6：candidate `999b2e5e…3dc3`，12/100 actions、207,469 tokens，offline-valid false→false，
  ranking=`22f69877…9fe2`，report=`a59d6f4e…70da`。

因此这组 post-selected family audit 的完整结果是 3/3 active valid + 111 frozen-RAW replay、1/3 recovery、
0 regression、online judge=0、validation/test=false。它不是 prospective unbiased cross-fit，但已经足够否定
“organize-2 阳性代表 family-wide transferable signal”：另外两道同 family RAW failure 都没有恢复。这里按
候选级停止条件终止，不再给该 workflow 增加规则、gate 或更多 holdout；它不创建 promotion/incumbent，也
不改变“无 incumbent 前禁止 downstream”的结论。下一步必须改变 candidate search/selection objective，
让 out-of-item utility 直接参与 TRAIN 候选形成，而不是继续验证这一个失败候选。

旧三折的 local-only trace 允许进一步做一次不依赖在线评分的失败归因。legacy organize-5 把五个目标目录
建在 source `/root/papers/all` 内部，而不是公开任务所要求的 sibling destination；legacy organize-6 已找到
正确 root，却把明显涉及 DNA binding 的 `0707.3224v1.pdf` 放入 `music_history`。因此下一候选不是新增
promotion gate，而是一个 trace-informed、仍属 TRAIN candidate search 的 generic contract 变体：

1. destination root 与目录名必须从公开 task/environment 派生，默认不得嵌套在待清空的 source 内；
2. 每个文件 assignment 必须有正内容证据，不允许用 catch-all/fallback 把剩余文件塞入任意类别；
3. move 后必须重开 destination，按冻结的 pre-move public manifest 核对一对一覆盖、source empty 与最终布局。

这三个 invariant 在任何 refined actual 前以提交 `baa3230ad25e49e137bb1ba4df63eb79bd410ce5`
预注册；manifest hash=`da85625ca58b63adb82ea0c7dba3ff7403c44160aa734ad224083a4167dc5483`，
trace-design evidence-set hash=`bc365eaf…f9de2`。三折 contract/bundle/candidate/work hashes 均固定，
static complexity 按 `base program 5 + 3 refined invariants = 8` 显式计数。manifest 同时诚实冻结
`prior_item_out_outcomes_used_for_candidate_design=true`、`globally_unbiased_crossfit=false`、
validation/test access=false、promotion/incumbent unauthorized；所以这不是新的无偏实验，也不会通过
“先提交 manifest”洗掉已有的 design leakage。

预注册提交之后，三折以 maximum concurrent model calls=3 的 Plus batch 同时启动，均无 retry、provider
anomaly、network fallback 或 online judge；每折仍是 1 个 active policy-on + 37 个 frozen-RAW replay：

- organize-2：candidate `c6c2521b…9f48a`，19/100 actions、340,315 tokens，offline-valid
  false→false，ranking=`340b664b…075f`，report=`7a9eff32…64e7`。初始 move plan 把
  `0907.4819v1.pdf` 同时列入两个类别，部分 move 失败后虽完成 source/layout 修复，但最终仍有 10 个
  wrong-subject 文件；4/6 verifier 子测试通过；
- organize-5：candidate `aa995536…31da`，8/100 actions、138,160 tokens，offline-valid
  false→false，ranking=`f534fb91…140b`，report=`4aa5f94a…6873`。103/103 manifest exact move、
  source empty 与 sibling layout 都成立，只剩 `2409.15949v1.pdf` 被放入 LLM 而非 music_history；
  4/6 verifier 子测试通过；
- organize-6：candidate `eadd5ab9…b945`，19/100 actions、366,903 tokens，offline-valid
  false→true，ranking=`36287072…960f`，report=`5f64ace5…4928`；五类 PDF 数均为 20，另三份
  office 文件正确归类，source empty，6/6 verifier 子测试通过。

refined aggregate 仍是 3/3 active valid + 111 replay、1/3 recovery、0 regression。相对 legacy contract，
阳性从 organize-2 换成 organize-6，但 family recovery coverage 没有提高。这说明本次三个自然语言
invariant 不足以得到可迁移 organize policy，也显示剩余错误主要位于自由文本 evidence→semantic assignment，
而非 destination、completion 或评分 gate。该 trace-refined 分支因此按候选级停止条件终止：不再围绕这三题
改 prompt、加 verifier-facing rule 或消耗 holdout。下一候选类若继续，必须把 `extract evidence → typed
assignment → one-to-one reconciliation` 变成可执行、可审计的生成对象，并在看 outcome 前固定的多 item/fold
TRAIN cross-fit objective 上提高 recovery coverage；这仍是 candidate formation/selection 的改变，不是新 gate。

### 8.12 P0 局部关闭：fresh 单题首次出现 preregistered task-utility gain

SC-100 role grammar 失败后，下一 candidate class 没有再补 regex、prompt 或 acceptance gate，而是从声明
TRAIN distribution 形成 frozen semantic structured extractor。financial operator 读取 current-item 的表格
内容，按冻结 question templates、DistilBERT QA runtime、typed answer plan 和 deterministic reconciliation
生成 agent 退出后的 capability effect。candidate=`0a6a4b1f…95cb`，operator source=`7d26085b…551c`；
formation 的 financial-1/-3/-5 为 candidate 3/3、历史 RAW 0/3，但正式 report 明确绑定
`in_sample_formation_replay=true`、`cross_fit=false`、`causal_gain_claim_authorized=false`。这一步只证明
候选可执行，不把 3/3 伪装成泛化。

随后在读取 fresh outcome 前，split manifest `9698dd65…a9f3` 固定 63 formation、9 fresh validation 与
14 residual sealed；treatment manifest `0d60ded5…18d` 固定 `financial-analysis-4`、recipe
`ea4b44b8…a781`、treatment `d6043343…dacd5`、Plus route、同镜像/预算以及本地 offline verifier。
9 个 RAW work unit 与该题唯一 1 个 candidate work unit 作为 10 次 physical model call 最大并发启动；
其余 8 个 candidate route 事前定义为 exact RAW projection，不额外消耗模型。

原 app tool session 在 agent 全部完成后丢失父 scheduler：travel 路已经完成 verifier，其余 9 个容器停在
post-agent/pre-verifier。这里没有重跑 agent，也没有把残留结果拼进新 experiment。恢复路径先后冻结
scheduler-loss manifest、source commit/bytes、原 session/work/result hashes 与 exact continuation boundary；
实际只在原 candidate 容器执行一次 semantic stage，再把 9 个离线 verifier 并发完成。最终统计为原始
model calls=10、replayed model calls=0、offline verifier executions=10、replayed verifier=0、online judge=0、
network fallback=false；9 个 orphan container 最终全部清理。

核心 active pair 是有效的：`financial-analysis-4` RAW=false、candidate=true，`candidate_minus_raw=+1`；
RAW/candidate observation hashes 分别为 `007a4886…a6f` 与 `bf78bbd5…943e`，两侧 error 均为空，
candidate 的 typed operator receipt 与 CTRF 闭合。因此此前的关键 P0——“包含 effectful typed operator 的
冻结 treatment 是否可能出现真实二元 task gain”——在**单 item existence** 意义上首次关闭。该结论不依赖 online evaluator，
也没有通过修改 promotion gate 得到；final report 的 `active_pair_evidence_valid` 与
`paired_task_utility_measurement_valid` 均为 true。

归因仍以整个预注册 treatment 为单位：RAW 与 candidate 是两个独立 agent trajectories，且 candidate 没有
保留 operator 执行前的输出快照。因而这是一条 unit-level paired evidence，不是 semantic operator 单独的
平均处理效应，也不估计随机 trajectory 方差。

边界同样重要。temperature-4 RAW 的冻结 observation 仍标记 `codex_turn_failed`；事后 trace 表明它只有
一条 transient reconnect error，随后 `turn.completed`，agent exit=0 且 action-budget receipt valid。
finalizer 没有覆盖该 observation，而把它保留为一条 inactive physical-audit invalid。因此 full batch audit
仍为 false；`execution_completed=true` 只指恢复/终结流程完成，必须与
`pristine_runner_completion=false`、`full_physical_batch_frozen_audit_valid=false` 同时读取。RAW 1/9、
projected candidate 2/9、delta +1 只能作 cohort 描述，不能作 promotion evidence。
official HippoRAG 对这种 million-row file task 没有同构 executable adapter，故明确 N/A 而非伪造代理；
residual sealed（包括 `financial-analysis-6`）未访问。

这使 claim ladder 需要细分：same-item causal utility 的 existence proof 已有；promotion-grade L2 validation
learning、稳定多题净收益、retained incumbent 与 unseen family transfer 仍没有。下一步不是在已消费的
financial-4 上调 question/template，也不是给 gate 加条件，而是冻结当前 operator，建立多个真正 untouched
measurement item/fold 的同一 paired treatment。未来 runner 同时应把 post-agent checkpoint/resume 做成
durable state，并把“transient `error` 后 `turn.completed`”与 terminal `turn.failed` 分开；这属于执行证据
基础设施，不能回溯重评分本次 temperature row。

### 8.13 P0 仍未关闭：SEC 13F period-out 只有局部方向信号，未形成稳定多折收益

单题 existence proof 之后没有把同一题反复调优，而是建立了 project-authored SEC 13F period-out replication。
preregistration `ab051822…454e6` 在 acquisition 前固定两份官方季度 archive、4 fold × 2 replicate
measurement、4 个独立 sealed commitment、双 oracle、RAW/candidate 两臂、Plus、100 action starts、16 路
最大并发、0 retry 和 offline-only evaluator。acquisition `f0770832…df1c` 在两次暂时 403 后从预注册的
exact SEC URL 成功取得原文件；pandas/streaming 两套独立 oracle 对 measurement 与 private sealed gold
逐项一致。execution freeze `20358ad1…9d3c` 又绑定 materialization、共享 image/cache、provider receipt、
prewarm `2fec4bc0…f91` 和 16 个 exact work unit。这里为修复 producer/consumer schema、live sibling receipt、
detached-input 与 `__pycache__` 污染做的是一次有限 integration audit；所有修复在最终 freeze 前收敛，没有
根据 performance outcome 增加 gate。

正式 root `financial_semantic_sec13f_period_out_v1_actual01` 一次启动全部 16 路。16/16 都写入
model-execution claim；15/16 完成 agent→operator→offline verifier→observation。唯一 invalid 是
`financial-period-out-measurement-f1-r0` RAW：trace 有一个 terminal `turn.completed`、agent exit=0、
15/100 actions、token usage 完整，但退出扫描仍发现并 SIGKILL 1 个 residual process/TID。冻结 receipt
要求这一计数为 0，所以 runner 正确 fail closed。随后唯一一次 `--recover-only` 只读取 durable artifacts，
backend/model/operator/verifier/online-judge call、stage transition 与 model replay 全为 0；由于 upstream
没有安全的 post-agent resume API，它将该路固定为 `do_not_replay_model`。不能为补齐样本而重放模型，也
不能把 residual 当成“可能无害”后放宽 contract。

7 个完整 pair 的统计为：candidate success 2、RAW success 1；candidate-only 1、RAW-only 0、both-fail 5、
both-pass 1；平均 paired delta `+1/7=+14.2857pp`。fold 0 为 0/2，fold 1 的唯一完整 pair 为 0，fold 2
为 `+1/2`，fold 3 为 0/2。换言之，唯一净 gain 只在 fold 2 replicate 1，不能被描述为多个 fold 的
稳定复制。完整八对的 candidate 固定为 2/8；缺失 RAW 若失败则 RAW=1/8、delta `+12.5pp`，若成功则
RAW=2/8、delta 0。因此最强诚实结论是“已观察 complete pairs 上无 regression、存在一个局部正 discordance”，
不是 valid primary positive。report `d75d8d4f…ba7` 固定 `promotion_authorized=false`、
`controls_authorized=false`、`family_out_authorized=false` 和 `sealed_test_authorized=false`。

非评分 failure attribution 将二元失败进一步定位到 operator semantics，而不是 routing 或 gate。8/8 candidate
都产生与公开模板完全一致的 operation assignment；15/15 完成路都有 `answers.json`，3/15 通过完整
answer-quality test。candidate 的 6 个首个可见 assertion failure 是 2 个 stock-count scalar 和 4 个
quarter-increase rank value/order，Q1 AUM 没有成为首个失败点。冻结 parent operator 的
`TRAIN_DEFINED_STOCK_CLASSES` 为忠实复现三条 consumed TRAIN solution，把多项相邻 literal 拼成一个长 token；
它与 period-out 公开合同的 25 类只重合 9 类，缺 16 类，另多 1 个拼接伪类别；而本 period-out task 明确给出
分离、规范化的 stock-title class ontology，且 stock count 与 investment-increase 两种 operation 都使用该集合。
operator 对 manager rank 的 latest-eligible-accession filter 和 normalized-manager
tie-break 也没有完整实现新合同。复用 candidate 的 provenance 是正确的，但它证明的是：**原 candidate 在新
evaluator 中可执行，不等于它已经语义适配新 evaluator。**

operator 在 agent exit 后会无条件替换 `answers.json`，但 receipt 没有保存替换前的 output hash；因此现有证据
只证明 candidate treatment 接管了 verifier 输入，不能对每题计算“同一 candidate trajectory 在 operator 前后”的
内容 delta。这个限制和两臂本来就是独立 trajectory 一起，要求归因继续停留在整个 frozen treatment。

这项静态诊断发生在 outcome 后，故不能修改 operator 后继续使用这 8 题作 fresh claim。正确的候选变化是：
从公开 instruction/evaluator contract 编译 NFKC manager identity、完整 stock-class set、latest non-NOTICE
eligible accession、CUSIP/manager aggregate 与确定性 tie-break，形成新的 typed SEC-13F operator；随后只在
全新 period/source 与新 selection commitment 上做 paired measurement。旧 8 题可以用于 deterministic
regression 和 consumed diagnosis，不能参与 selection threshold、promotion 或再评价。

最后，CTRF failure traceback 会持久化 expected/observed diff 片段，违反 measurement view 对
`ground_truth_persisted=false` 的字面强表述。由于 verifier 在 agent 完成后才 materialize，现有 15 个分数
没有 treatment leakage；sealed 也未 materialize 或访问。但 measurement content 已不可再作为秘密 holdout，
未来 verifier 应只持久化结构化 failure category/hash，不保存 assertion value diff。这是 future evidence
hygiene 修复，不改变本轮 success，也不是新的 performance gate。

### 8.14 exact-domain P0 已关闭：Replication C promotion 与 sealed 形成完整正复验

8.13 的结论只适用于已消费的 parent operator。随后没有在同一 measurement 上补关键词或加 gate，而是把
公开 period-out contract 编译为 closed typed operator：完整 stock-class ontology、latest non-NOTICE
eligible accession、NFKC/punctuation-insensitive identity、CUSIP/manager aggregate、deterministic rank/tie-break
均成为可执行语义。候选、recipe、source closure 与 8 个 development pair 在 fresh root 前冻结；正式批次
16 个 Plus 调用一次最大并发完成，8/8 pair valid，RAW 0/8、candidate 8/8，四个 fold 各为 +2，0 harm。
这一次满足了预注册 promotion decision，而不是修改 evaluator 或把旧 incomplete rows 拼接成 positive。

promotion 后的 controls 没有陷入继续补 gate。operator-only 8/8 产生与 development candidate 完全相同的
output SHA，直接支持 output-level operator sufficiency。skill-only arm 的 verifier leaf 由 UID 1000、mode 0755
创建，而容器又 `--cap-drop ALL`；写失败被旧 verifier 的 unconditional exit 0 吞掉，造成 7 个有效失败与
1 个 unresolved。该批按原样固定为 `executed_incomplete_no_retry`，不修后补跑；separate disposition 把它
解释为 control infrastructure defect，并接受 operator-only 的完整因果输出闭包。family-out 同样没有伪造
HippoRAG 或近似 proxy：任务范围没有同构 official adapter，故固定 N/A，不声称 family transfer。

sealed preparation 在单独 post-controls authorization 后才开始。access journal 以 `O_EXCL` 写 claim、fsync
文件与父目录，再读取一次 private pack；pandas/streaming 两个本地 oracle 对 4 项完全一致。执行 freeze
绑定完整 runtime source roots、candidate/provider、payload/gold/materialization/prewarm、4 对/8 calls 与 one-shot
launcher；正式 Plus 批次 8 路同时提交，0 retry/replay/resample/switch。最终 4/4 pair valid，RAW 0/4、
candidate 4/4、4 gain/0 harm；4 份 candidate runtime evidence 证明 operator 真正在 agent exit 后执行，8 份
hash-only network receipt 证明所有容器都先断开 provider network、8 方 barrier 到齐后才 materialize `/tests`，
offline verifier 后仍为 network-none。

这项 sealed 证据有两条必须同时保留的盲化限定。第一，authorization 前曾发生一次只返回已提交 digest、没有
结构化/语义字段的 private-pack SHA stream，故严格 zero-byte claim 已被正式放弃。第二，正式 frozen batch
启动后，监督诊断的 host process-table 检查意外展开 sealed instruction text；gold、expected output、answer、
model response 与 outcome 均未显示。第二事件发生时 candidate、freeze、assignment 与 one-shot batch 已不可变，
之后没有 adaptation、重试或 provider switch，所以它不改变本批 paired treatment contrast；但监督通道的严格
query blindness 已失效。最窄、可审计的结论因此是：**固定 SEC-13F typed treatment 在声明的 Replication C
development 与 sealed cohort 上分别取得 8/8 和 4/4 paired gain；这是 exact-domain L2 与 unseen-instance L3
confirmation，不是严格 blind、family-wide transfer、一般 recursive learning 或 RQGM 结论。**

### 8.15 generation-one P 晋级时，recursive retention 与 evaluator co-evolution 尚未得到证明

旧 6-item MuSiQue formal cohort 因 prelaunch runtime failure 已按 `closed_efficacy_unknown` 封存，本轮没有删除
failure receipt、重放旧题或把旧结果并入新统计。新的研究单元先以提交 `aac4ceb3` 冻结 v2 runtime
attestation，再由 `8702c369` / `97644e21` 对 official DEV 预注册并一次形成 96 项：F1/M1、F2/M2、
F3/M3、A_form/A_hold 各 12 项，八个 block 在任何 formation 或 measurement 前一起确定。selection 使用私有
secret 的 HMAC 顺序，公开层只保留 commitment；acquisition 报告 2,417 个 source row、594 个 eligible row、
96 个 selected row，oracle disagreement 为 0。该新链与旧 official-TRAIN 6-item cohort 分离，不能把两者拼接
成扩大样本的结果。

F1 formation 提交 `fdef9d45` 只打开 12 个 F1 item，model、network、generator、online evaluator 和 measurement
access 均为 0。有限 DSL 的 84 个 type-valid candidate 对应 70 个独立 retrieval behavior；全 F1 winner 为
`TF-IDF(title=4,text=1) → one-hop entity-token expansion(weight=2) → top-5`，命中 21/34 个 support label，
program hash 为 `0e9fea15…0785`。但四折 winner 不是同一个 program，完整 retrieval behavior 也不是同一个；
receipt 明确记录 `selected_program_stable=false` 与 `selected_behavior_stable=false`。因此 F1 只形成了事前允许的
generation-one 候选 P，并没有证明形成算法跨 fold 稳定，更不能据此回到 F1 加关键词、改 prompt 或增加 gate。

M1 在提交 `7b38a23d` 的完整 pre-run freeze 后才开放。冻结合同是同一 12-item、top-k=5、三臂
`canonical_RAW / frozen_P / official_HippoRAG`，36 个 retrieval unit 最大并发提交；前端只向 retrieval callable
传递不含 support/answer 的 gold-free item view，并在全部 terminal join 后才执行
official-support scoring；study-level answer-generator、Ruoli/external-network、online evaluator、retry、replay 和
resample 都固定为 0，official arm 内部仍使用冻结本地 LLM/OpenIE。提交 `4617a976` 的公开 aggregate report
给出：RAW 命中 `7/29`，P 命中 `14/29`，official HippoRAG 也命中
`14/29`；P−RAW 的 support-recall delta 为 `+7/29`，逐 item 是 7 gain、1 harm、4 tie。P 与 official HippoRAG
在总命中数上相同，但这不等于两者逐题行为相同，也不构成 family-out。

事前 promotion policy 只问 `total_support_hits(P)-total_support_hits(RAW)>0`，所以本次产生
`promote_P_to_retained_generation_one` disposition；runner 同时明确 `archive_mutated_by_runner=false`，不能把这份
aggregate report 夸大为完整 archive-retention 实验。正式 postflight 对相同 runtime filesystem binding 做了
fresh attestation，`postflight_binding_sha256=04498dcb…3df6`，且 formal entry 没有追加 executable identity probe。
这支持“本次 official arm 的冻结 filesystem identity 在执行后仍闭合”，不支持“所有第三方依赖源码与运行语义
已被完整 attestation”。v2 对 gritlm、hipporag、litellm、networkx、openai、torch、transformers 等第三方包记录
的是 distribution metadata tree；完整源码树绑定只覆盖 official HippoRAG source，自有 adapter/attestation
implementation 和本地 model assets。这个 metadata boundary 必须继续写进 claim limitation。

因此本轮真正推进的是：在 fresh official-DEV M1 上，frozen P 相对 canonical RAW 取得预注册的 positive
retrieval utility，并按固定规则获得 generation-one promotion disposition；它没有证明 P 在新一代与 Q 共存时
被保留，也没有证明 evaluator 会按 anchor evidence 自主升级。当时下一步只允许在 F2 形成 Q，随后一次冻结并执行
M2 的 `empty / P / Q / P+Q`，分别估计 novelty、retention 与 forgetting；M2 完成前，L4 recursive retention
仍未达到。A_form/A_hold evaluator co-evolution、跨域或同构 family-out 也都尚未执行。这里的正确动作是进入
Q/M2，而不是在已经看过的 F1/M1 上补 gate。

### 8.16 M2 因 sandbox 基础设施失效；L5 有效完成但没有 evaluator transition

提交 `ee8f3a48` 先冻结 stage-specific M2 runner，`2956d5b7` 只打开 F2 并形成 Q，`57d7e601` 再在 M2
仍未打开时冻结 exact root、positive M1 lineage、P/Q program、M2 commitment 与三臂同构 comparison。正式 M2
随后只启动一次：authorization 已消费，exact M2 已打开，36 个 P/Q/official work unit 全部提交；其中 24 个
terminal 完成后 official worker 返回非零。无 benchmark/model 的最小 postdiagnostic 精确复现相同 stderr digest：
当前 managed sandbox 不允许 bubblewrap 为 `--unshare-net` 创建 `NETLINK_ROUTE` socket。该故障发生在 scoring
之前，aggregate report 不存在，任何 P/Q partial terminal 都没有评分或转作结果。`7bb8061e` 因而将 M2 固定为
`closed_invalid_environmental_no_replay_no_score`：retention efficacy 仍为 unknown，不能把 infrastructure failure
解释为 P retained、forgotten 或 Q 无效；也不能在解除 sandbox 后重放同一 M2。M1 对 P 的 generation-one
promotion 是独立已完成证据，不因 M2 invalid 而撤销。

与 M2 独立的 evaluator 链没有使用 official HippoRAG、Ruoli、generator 或在线 evaluator。A_form 与 F3 各对
84 个 fixed typed program × 12 item 执行 1008 路最大本地并发，全部 terminal join 后才形成 official-support
evidence。A_form 选择 `micro_worst_v1` 作为 challenger，但它与 `micro_recall_v1` incumbent 选择同一 program；
untouched A_hold 上两者因此均为 `12/29`，0.9 Wilson lower bound 同为 `0.3042971842`，strict-improvement policy
拒绝晋升。F3 上两条规则原本会选择不同 program，但由于 anchor transition 未发生，M3 依法保持 incumbent；
untouched M3 上 incumbent/active 均为 `18/29`，net support hit 为 0，positive search utility 为 false。故本轮
证明了 evaluator candidate formation、anchor rejection 与未晋升时的 score-dependency preservation 能执行，
但没有证明 evaluator co-evolution。

这一 L5 结果还有明确的完整性边界：实际观察到的命令均由 clean `python -m` 进程执行，formal API 没有
result/callback 参数；但实现不能抵御任意可信主机内的 in-process monkeypatch，measurement implementation set
也不是完整 transitive closure，private evidence 从执行到 public commit 依赖本地 custody。结果因此可作为当前
committed observed offline run，而不是 adversarial-host integrity proof。正确下一步不是修改 A_form/A_hold 或
追加 gate，而是把 M2 retention 标为未测得，并只在新 domain/family 的新 acquisition 上检验 P 的 transfer。

### 8.17 frozen P 在 HotpotQA family-out 上保持正净 retrieval utility

family-out 没有重用 MuSiQue M2/L5，也没有在 HotpotQA 上修改 P。提交 `7306f333` 固定新的 acquisition 与
one-shot runner；source 绑定为 Hugging Face-hosted HotpotQA distractor-validation Parquet conversion 的固定
commit、27,452,575 bytes 与 SHA-256 `c20b638c…7c6`，同时明确不声称它与原始 CMU JSON 逐字节或逐 row 等价。
官方 HotpotQA repository commit 与 README 只用于 schema/original-source declaration。eligibility 纯机械地要求
全局唯一非空 ID、非空问题、至少五个唯一非空 context title、有效 sentence/support index 与恰好两个 gold
title；不按 answer、type、level、文本主题、operator 或 score 过滤。7,405 行中 7,318 行满足条件，私有 HMAC
一次选择 12 行；公开 acquisition 只保存计数、runtime/provenance、commitment 与 hash，不保存题目、ID、正文、
support 或私有路径。

在任何 row 打开前，`6852f15f` 已提交完整 preregistration；acquisition 又先写 one-shot consumption marker，再打开
Parquet。正式执行前，`87fd35f7` 用不含 benchmark/model 的 probe 在同一 host 权限验证 bubblewrap 的
`--unshare-net` 与 actual writable-root `--bind` 拓扑；`86242994` 随后冻结 exact acquisition、F1 P、runtime、
execution root、12×3 work-unit grid 与 36 路最大并发。runner 私有加载 source-provided support labels，但只把
question/corpus 交给三臂，不交付 support、answer、item ID 或 label-derived commitment；单一 36-party barrier
到齐后同时释放全部 work unit。36/36 terminal join 后执行 fresh filesystem postflight，只有 postflight 与
preflight binding 一致时才调用 scorer 使用 labels 做本地 recall@5 评分。没有 Ruoli、study-level
answer-generator、online evaluator、外部网络、retry、replay 或 resample；official arm 内部仍使用冻结本地
LLM/OpenIE。

提交 `d2981542` 的 aggregate report 为 valid：RAW `11/24=0.4583`，frozen P `21/24=0.8750`，official
HippoRAG `20/24=0.8333`。P−RAW 为 `+10/24=+0.4167`，逐项 7 gain / 0 harm / 5 tie；P−official 为
`+1/24=+0.0417`，逐项 4 gain / 2 harm / 6 tie。P 只在 MuSiQue F1 形成并先于 Hotpot acquisition 固定；
Hotpot measurement outcome 又只在 pre-run freeze 后计算，且 P 没有 adaptation/reformation。因此这是一条真实的
cross-family retrieval-only transfer positive。它关闭了本阶段“frozen P 是否只在 MuSiQue 同族有效”的主要
缺口，不需要再为这个结论追加 gate。

结论边界同样重要。样本只有 12 项/24 个 source-provided support title，未预注册显著性阈值或 broad-population
置信结论；RAW 是 canonical context-order top-5，official arm 是本地离线 official HippoRAG core/runtime；primary
metric 只覆盖 retrieval support recall，不覆盖 answer generation、端到端 QA、完整 HotpotQA 或原始 CMU JSON。
该阳性也不能修复已经不可重放的 M2：recursive retention 仍 unknown；L5 challenger 仍未晋升，evaluator
co-evolution 仍未发生。下一研究缺口因此回到真正的 L4/L5，而不是继续重跑 Hotpot、调整 P 或增加 retrieval gate。

### 8.18 fresh Hotpot 多代保留成立；evaluator challenger 被 untouched anchor 否决

这轮没有重用旧 MuSiQue M2/L5，也没有把上一批 12-item Hotpot family-out 当作新 measurement。提交
`aedc0bd3` 先固定 acquisition、L4 与 L5 实现及对抗测试；审计在任何新 private block 打开前发现并修复了
两个会破坏因果解释的问题：Q 必须重新绑定 acquisition/F_Q/P/one-shot marker，evaluator 的 action hash 必须只由
实际 Q 与 P+Q rankings 决定而不能混入 program ID。最终 formal cache 又强制 exact ordered 84-program pool、每个
program exact 24 rows 与 source commitment。38 项相关回归全部通过；另一次只含合成公开文本的 24-worker
official-HippoRAG capacity diagnostic 完成 24/24 terminal 与 fresh postflight，未访问题目、label 或 score，因而
保留 M_L4=24，而不是为规避资源风险把样本缩回 12。这里的 clean-CLI / local-custody 仍不是 adversarial-host
integrity proof，但 action-identifiability 已不再只是 program-ID inequality。

`126ba352` 随后在 source row 与旧 private pack 都未打开时提交六分区 preregistration。固定 HF Parquet 中一次
private-HMAC selection 形成 F_Q=36，M_L4/A_form/A_hold/F_search/M_search 各 24，共 156 项；全部旧 Hotpot 12 项
在新 HMAC 排序前精确排除，公开 acquisition 不保存 ID、question、corpus、support 或私有路径。`1f4f6b42`
提交该 acquisition 后，Q formation 才消费固定 marker 并打开 F_Q。冻结 Q 在 F_Q 上 P+Q 命中 64/72，Q direct
63/72，相对 P 新增 10、相对 Q 保留 4、P forgetting=0；但四折 selected program 与完整 behavior 均不稳定。
预注册明确把 cross-fit 作为描述而非 gate，所以没有重选 Q、调 prompt、补关键词或重采样；`62005707` 直接冻结
第一次形成的 Q。

`2e0da52c` 在 M_L4 打开前冻结 exact P/Q、positive M1 lineage、block commitment、runtime、execution root 与
24×4=96 work-unit single barrier。一次正式执行完成 96/96 terminal；RAW/P/Q/official 是物理 retrieval，P+Q 只在
全部 terminal join 与 fresh runtime postflight 通过后由 P/Q rankings 做固定 RRF，再用 source-provided support
离线评分。`4caa37fc` 的聚合结果为：RAW 22/48，retained P 36/48，novel Q 40/48，P+Q 43/48，official
HippoRAG 31/48。primary retained contribution P+Q−Q 为 +3（4 gain / 1 harm / 19 tie）；novel contribution
P+Q−P 为 +7（6/1/17）；只在 1 个 item 丢失 1 个 P support。因而这个 fresh retrieval-only cohort 同时观察到
旧 P 与新 Q 对组合的正边际贡献，关闭了此前由 MuSiQue M2 infrastructure failure 留下的 L4 efficacy 缺口。
P+Q−official 为 +12（13/3/8），明显强于旧 12-item family-out 的 +1，但协议没有预注册 population-level
significance 或 compute matching；因此只写 fixed-cohort positive net，不写 statistical superiority、广义
family-out、等算力、answer generation 或完整 HotpotQA 优越。

L5 没有因为 L4 阳性而调 evaluator。A_form 对 exact 84-program pool 形成 incumbent `q_direct_micro_v2` 与
challenger `combined_novelty_balanced_v2`；两者在 A_form 的 program 与 observed action 都不同。F_search 又在
A_hold 打开前冻结未来 incumbent/challenger program 与 behavior hash，`measurable_contrast=true`；不存在相同行为
时换 fallback 的路径。`4dd34197` 冻结 A_hold 的 P、两组 A-form Q action 与 24×3=72 barrier；一次执行 72/72
terminal 后才离线读取 support。结果 incumbent P+Q 为 41/48，challenger P+Q 为 38/48，challenger−incumbent
=−3，1 gain / 4 harm / 19 tie，单侧 exact paired sign-flip p=31/32=0.96875。唯一 promotion criterion 因而为
false；epoch 保持 `q_direct_micro_v2`，dependent score 没有被 invalidation，independent-source record 保留。
`18b94c44` 明确记录 M_search_authorized=false、M_search_opened=false。

所以当前 claim ladder 必须非对称更新：**L4 在这一窄的 Hotpot source-support retrieval 定义上达到；L5 仍未
达到。** 后者不是 wiring 或 identifiability 缺失，而是一个有效、可比较、untouched anchor 上的真实负结果。
不得在同一 A_hold 上换 challenger、降低 alpha、追加 gate 或打开 M_search。若继续研究 L5，只能先用已消费的
formation/anchor 作为设计数据，提出实质不同且可执行的 evaluator objective，再从新 acquisition 同时取得新的
A_form/F_search/A_hold/M_search；不能把连续尝试到成功包装成 evaluator co-evolution。更高层目标还缺：成功的
evaluator epoch replacement、selective invalidation 实际发生，以及被晋升 evaluator 在预先冻结的新 search
measurement 上改善后续 candidate search。

### 8.19 final Hotpot portfolio cohort 因 acquisition 持久化故障严格终止

上一节允许的唯一同源后续不是重配旧 single-Q objective，而是一个事前冻结的 two-Q portfolio 机制。
`b504f8b3` 固定 design（file SHA `1cb6d552…ba43`；semantic SHA `3ed3811d…d37d`）；
`6f373fce` 固定 acquisition/portfolio 实现（两个模块 SHA 分别为 `2638c9c7…202`、`822bff25…a88`，
preregistered implementation-set SHA `2f3979ef…f6e0`）；`257d6283` 再提交 preregistration（file SHA
`82b84903…053`；semantic SHA `5e265fd7…697c`）。固定 selection 复用既有 HMAC ordering，在排除原 12 项后只取
与 v2 `[0,156)` 构造性不相交的 continuation `[156,324)`，并事前分配 A_form_0/A_form_1/F_search_0/
F_search_1/A_hold/M_search 共 168 项。旧 v2 acquisition 与 final disposition 的 file/semantic SHA 也分别绑定为
`2d907a72…dd6` / `ebd6e89f…ce4` 与 `631c8091…949` / `487831a0…aef`，没有打开任何旧 v2 private block，
旧 M_search 仍保持 unopened。

执行中有两个必须分开的事件。第一次调用带有 invocation typo，在 marker 前即停止：authorization 未消费，
source/original pack 未打开，也没有形成 cohort。纠正调用先原子持久化 one-shot marker（file SHA
`31b6a9c6…e00`；semantic SHA `bcfed9d9…41f3`）；只有此后才打开原 12-item exclusion pack（既有 file commitment
`bf22661c…6c9`）和固定 source（file SHA `c20b638c…7c6`），完成 exact continuation window 的内存选择。
下一条操作 `os.mkdir(pack_root, 0o700)` 因预定 private root 的父目录不存在而抛出 errno 2
`FileNotFoundError`。故失败位置在 deterministic selection 之后、任何 private-pack root/block/locator/public
acquisition receipt 写入之前；最终只有一个 marker，block、locator、receipt、持久化 selected row 均为 0。

这是 non-scoring local-persistence infrastructure failure，不是 evaluator negative：没有运行 retrieval、RAW、
HippoRAG、agent、answer generator 或 evaluator，没有计算 support score，也没有外网/online evaluator 调用。
但 corrected call 已消费 authorization，并在 post-marker 阶段确定 fixed cohort，所以不能把“没有 block 文件”解释为
可安全重跑。严格终态是：`[156,324)` 的六个 logical slice 全部永久烧毁，禁止 retry/replay/resample、同 window
reconstruction、换 secret 重新抽样或 Hotpot v4；新 portfolio efficacy 保持 unknown。此前独立完成的 Hotpot
family-out、L4 positive 和 L5 negative 不依赖这次失败，因而全部保留。公开 hash-only disposition 为
[`hotpot_evaluator_portfolio_acquisition_infrastructure_failure_disposition_v1.json`](../manifests/hotpot_evaluator_portfolio_acquisition_infrastructure_failure_disposition_v1.json)，
semantic SHA `b929ae19bfac016ed9e34f13e08afd4435d76005445772935845a0122df2199b`。

### 8.20 MuSiQue residual portfolio 的 A_hold 因 committed submission-ordering defect 严格终止

Hotpot final acquisition 严格终止后，备选 A 没有回到 Hotpot 连续抽样，而是使用本地 official MuSiQue DEV
archive 的同一私有 HMAC ordering 做 exact continuation。`6dd53a19` 固定 two-Q portfolio design（file SHA
`82f09dc1…84d2`；semantic SHA `f93d9e67…3cb`），`c7e20674` 固定 acquisition/runner（portfolio runner SHA
`66f44253…24d`；implementation-set SHA `95a57ec5…833`），`54ae45ca` 提交纯合成 infrastructure diagnostic。
随后 `0271f9e5` 在 source row 与旧 private row 都为 0 次读取时提交 preregistration，`96d779ce` 一次取得与旧
`[0,96)` 构造性不相交的 residual `[96,264)`：A_form_0/A_form_1/F_search_0/F_search_1 各 24 项，A_hold
48 项，M_search 24 项。acquisition receipt 的 file/semantic SHA 为 `ade2b90c…25bc` / `bd0b9bd8…cbf7`。

A_form 与 F_search 各运行两个 2040-party barrier、4080/4080 terminal、0 invalid；两者各有 69 个 behavior
class，并形成 program/action 都可区分的 incumbent/challenger portfolio。`1b9d53c9` 在 A_hold 打开前提交两份
public receipt；`47faa049` 再提交 A_hold pre-run freeze（file SHA `b79a1b7f…3c6e`；semantic SHA
`ac8a46b2…246b`）。freeze 明确记录 A_hold rows=0、labels=0、M_search rows=0，并固定 48×6=288 work units、
maximum concurrency=288、single barrier parties=288、0 retry/replay/resample 与 exact sign-flip 唯一晋升标准。

正式 A_hold 先原子消费 authorization（semantic SHA `64f7d29c…1ac9`），随后打开并完整验证 48-item private
block。故这一 cohort 已被正式过程看到，不能再用“尚未评分”来恢复盲性。失败发生在 support scoring 之前的
maximum-width retrieval：committed code 以
`[future.result() for future in (executor.submit(...) for unit in work_units)]` 消费 lazy generator，导致第一个
`submit` 后立刻等待其 result；该 worker 已进入 288-party barrier，而其余 287 个 future 尚未提交。180 秒后屏障
必然破裂。failure receipt 精确记录 attempted=1、terminal=0、stage=`maximum_width_six_component_retrieval`，
与代码根因一致；目录只有 consumption 与 failure，没有 ranking、private evidence、aggregate report 或 score。
模型、外网、online evaluator、answer generator、retry/replay/resample 均为 0。

因此该结果归类为 **implementation-invalid non-scoring**，不能写成 incumbent/challenger negative。A_hold 48 项
永久烧毁，不修复后重放、不换 cohort 补测，也不把未打开的 M_search 改作 backup；design 的 final same-source
terminal policy 同时排除新的 MuSiQue evaluator attempt。没有 A_hold report 就没有 promotion，M_search 必须保持
未授权、未打开。A/F action commitments 仍是有效的 prospective provenance，可在独立新领域被明确标为
zero-shot action transfer 或 grammar transfer，但不能称为 promoted/active evaluator，更不能补写 MuSiQue L5。
严格终态由 `db8db16d` 先行提交。其后 `98763f27` 才将三个 barrier path 统一为先 eager submit 完整
party width、再逐一 join terminal；新 runner/test file SHA 分别为 `6c3bb16c…3f51` / `27194911…a29`，focused
16/16 与相关 grouped 66/66 tests 均通过。该 postmortem 修复不改变旧 implementation binding，也不授权删除
execution root 或重放 A_hold。
公开 disposition 为
[`musique_evaluator_portfolio_a_hold_implementation_failure_disposition_v1.json`](../manifests/musique_evaluator_portfolio_a_hold_implementation_failure_disposition_v1.json)，
semantic SHA `f1f51d9343f1d9d134344dcdb93c9514ea34353a7714664d57a2c1d512803d2c`。

### 8.21 fresh 2Wiki fixed-action transfer 有效完成；agent 明显高于 item-local HippoRAG，但 evaluator 未晋升

MuSiQue residual implementation-invalid 后没有修复并重放，也没有回到 Hotpot 连续抽样。新 study 改用 official
2WikiMultiHopQA archive，archive SHA 固定为 `95df2bf…9eef`。只在 source qualification 阶段做 outcome-blind
aggregate/hash 解析；正式 selection 前又固定 exact canonical identity、私有 HMAC、历史 HippoRAG 1000-row
四字段 denylist，以及 train/dev/test 三 split 的 normalized-question/item-ID collision exclusion。test 只参加
metadata collision scan，从不进入候选；最终 ledger 对 train `167454`、dev `12576`、test `12576` 逐行闭合。
A_hold 从 train 按四种官方 type 各取 12 项，M_search 从 dev 各取 6 项；两块在一次 acquisition 中同时形成，
但 M_search 只有在 A_hold 晋升后才可打开。

这不是新的 2Wiki candidate search。retained P、A incumbent/challenger 与未来 F incumbent/challenger 全部由既有
MuSiQue public action hashes 在任何 2Wiki private row 打开前固定。提交 `3ac92a5d` 锁定 corrected eager-submit
runner；随后唯一一次 public-synthetic diagnostic 完成 384/384 terminal、两个 192-party wave 和 48/48 official
terminal。正式 A_hold freeze 再固定相同的 `2 × (24 items × 8 components)` 结构、official concurrency cap 24、
0 retry/replay/resample、fresh pre-marker bubblewrap probe 与 postflight-before-scoring。正式运行得到 384/384
terminal、barrier parties `[192,192]`、official 48/48，且所有 support score 均在 terminal closure 与 runtime
postflight 后离线计算。

固定 cohort 上，incumbent portfolio 为 `111/120` support hits，challenger `110/120`，retained P `110/120`，
official item-local HippoRAG core `99/120`，canonical RAW `56/120`。唯一 promotion comparison 是 challenger−incumbent：
0 gain / 1 harm / 47 tie，净 `−1`，one-sided exact magnitude-preserving sign-flip `p=1`；所以 challenger 不晋升，
epoch 不变，M_search 没有 freeze、授权、打开或 artifact。不能换 challenger、降低 alpha、把 M_search 当 backup，
也不能继续抽同源 2Wiki cohort。

预先声明为不影响 promotion 的 descriptive control 给出一条更强的 retrieval 结果：incumbent−official 为
`+12` support hits，16 gain / 4 harm / 28 tie，exact `p=1549/262144=0.0059089661`；retained P−official 为
`+11`，15 gain / 4 harm / 29 tie，exact `p=1259/131072=0.0096054077`。因此 frozen agent retrieval 在这组
fresh 2Wiki items 上并非只“略高”于 HippoRAG；但该结论严格限于 48-item、10-paragraph item-local
support-recall@5，不是 official shared-index/full-corpus 2Wiki benchmark、端到端 answer generation、等算力或总体/SOTA
优越。L3 fixed-action/frozen-P transfer 证据因此增强，L5 evaluator replacement 仍未达到。

### 8.22 QASC direct-action evaluator 有效完成；未晋升且显著落后 HippoRAG

2Wiki 的有效 non-promotion 后没有在同一 family 继续换 evaluator。QASC study 使用 official TRAIN/DEV archive 与
16,987,130 行 fact corpus；两次 unsolicited viewer disclosure 后烧毁旧 secret，再生成当前私有 HMAC secret。正式
selection 只用 TRAIN/DEV 的 normalized `question.stem` 做 collision exclusion，严格复现 TRAIN `7175`、DEV `865`，
TEST 不重开。四块 A_form/F_search/A_hold/M_search 各 64 项在任何 formation 前一次取得并封存；每项由 row fact1/fact2
和 full-corpus BM25 top-30 distractors 组成固定 32-document retrieval view。gold、answerKey、identity commitment 均不进入
view；labels 与 views 物理分离。

候选不再是关键词或追加 gate，而是 16 个 equal-compute two-hop recipes：两种 q1、两种 bridge budget、两种 q2 与
两种 pair aggregation 的笛卡尔积。冻结 DistilRoBERTa NLI 在 8 worker × 4 threads 上分两波计算；incumbent 只按
support maximin 形成，challenger 只按 answer counterfactual chain margin 形成。row-free capacity diagnostic 首先发现
24 个 concurrent official model process 为 0/24 terminal，而独立 8-worker probe 与最终 diagnostic 均为 8/8；这一
调整发生在 acquisition marker 与任何 formal QA row 之前。正式 formation 随后完成 A/F 共 128×16=2048 个 action，
两组 incumbent/challenger 的 ordered top-5 behavior 都不同，因此 A_hold 可执行。

untouched A_hold 的唯一 confirmatory comparison 为 item utility `U=support_hits+complete`。incumbent 得到 `67/128`
support hits、23 个 complete、总 U=90；challenger 得到 `66/128`、18 complete、U=84。逐题 challenger−incumbent 为
10 gain / 11 harm / 43 tie、净 U `−6`，one-sided exact magnitude-preserving sign-flip
`p=1668987/2097152≈0.795835`。所以 counterfactual evaluator 不晋升，support-maximin epoch 保留，不发生 selective
invalidation，M_search 没有 freeze、report、execution root 或 row/label open。

三条 descriptive control 全部 64/64 可用：RAW 为 `19/128` support、0 complete、U=19；retained P 为 `38/128`、
2 complete、U=40；official item-local HippoRAG core 为 `103/128`、44 complete、U=147。challenger 相对 official 的
净 U 为 `−63`，6 gain / 36 harm / 22 tie。该结果推翻了“当前 agent 只比 HippoRAG 少赢一点”的乐观解释：在这个
QASC-derived 32-document task 上，主要缺口是 graph/two-fact coverage action 本身，以及 evaluator objective 与实际
support utility 的对齐，不是缺少更多 promotion gate。它仍不是 official full-corpus QASC、端到端 answer generation、
等算力或 SOTA 比较；row facts 直接注入，official HippoRAG 只是 custom item-local control。

本 study 已按预注册终止。不能在已消费 QASC blocks 上换 objective、调 alpha、改 recipe、打开 M_search 或抽 backup。
若论文仍硬追 L5，下一研究对象必须同时改变 action family 与 evaluator mechanism，并在独立新领域/新 custody 上一次性
预注册；若保持当前 action family，最诚实结论是三次 behavior-valid evaluator rejection 与零次 evaluator replacement。

### 8.23 AVeriTeC typed QA-set evaluator 有效完成；typed action 未超过 RAW，L5 未获授权

为同时检验现实域三臂优势与 L5，本轮没有重用 BRIGHT、QASC、EBM-NLP 或既有 scored cohort，而是建立
`AVERITEC_P1_TYPED_QA_SET_EVALUATOR_V1`。official AVeriTeC source 固定在 commit
`7c62d1ec…f81`；公开 P0 只资格化 archive/schema/topology，不评分。正式 projection 是 real-world fact-check
claim 到同一封闭 cohort 内 question-answer evidence pair 的 retrieval，不声称 open-web AVeriTeC。三组所谓 family
是 causal→quote→numerical priority-assigned native label strata，不是互斥 source family。A_form/F_search/A_hold/M_search
分别冻结为 108/36/36/36 项；RAW 与 E0 都是同一 `R0_DIRECT_DENSE` action，official HippoRAG 是本地冻结、
同 corpus/query 的 item-local comparator。

候选空间一次固定为八个 typed QA-slot recipes，MiniLM 产生 DIRECT/CAUSE/EFFECT/QUOTE/SOURCE/NUMBER/COMPARE/CONTEXT
坐标，stable greedy one-to-one assignment 形成 top-5；E1 只在 A_form qrel 于 action seal 后打开时拟合一次 ridge。
F_search 不创建 qrel、永不参与 gate。v1–v3 source-free canary 暴露的 canonical JSON、dictionary-order 与 ephemeral-index
错误均发生在 formal source/secret 为 0 时，各自关闭且不重放；v4 唯一通过两张 RTX 2080 的 production-isomorphic
canary。随后 execution freeze、142-file checksum manifest 与 launch freeze 先提交，311linux 上唯一 formal service
以 `CPUQuota=400%`、`MemoryMax=40 GiB`、`TasksMax=64`、`Restart=no`、`AF_UNIX`/IP deny 启动一次。
终态为 systemd success、`NRestarts=0`、40 个私有 artifact 全部闭合、0 API/online evaluator、0 retry/replay/resample。

A_hold 给出明确的有效负结果。E1−E0 与 E1−RAW 都是 0 gain / 0 harm / 36 tie，净 utility 0、exact tail 1。
这不是 evaluator 没有动作：E1 在 19/36 项选择了非 R0 recipe，并在同样 19 项改变了 top-5；只是这些变化对 qrel
recall 全部为零。E1−official HippoRAG aggregate 为 4 gain / 1 harm / 31 tie，净 `+3/2`、exact tail `3/32`；
但 causal family 为 12/12 tie、净 0，numerical 与 quote 各净 `+3/4`、family tail 均为 `1/4`。更关键的是，
E1 与 RAW 的逐项 utility 完全相同，所以 aggregate 的 HippoRAG 差同样属于 RAW，不能归因于 typed action。
因此 `A_hold_evaluator_promoted=false`、现实三-family双基线 primary=false。formal controller 没有读取 M_search
view/qrel，也没有创建 M_search action/executor/score；`M_search_L5_passed=null`，不是 false，更不是 L5 positive。

安全 terminal 已从远端回传，并由 sealed action/qrel 独立离线重算；40 个私有文件的 canonical hash、mode、worker
stdout/stderr、blocked-network audit 与 evidence binding 全部一致，原始 item/query/document/qrel/逐项 score 继续只留
远端。证据见
[`formal terminal`](../manifests/averitec_p1_formal_terminal_v1.json)、
[`offline finalize`](../manifests/averitec_p1_offline_finalize_v1.json)、
[`execution freeze`](../manifests/averitec_p1_execution_freeze_v1.json) 与
[`launch freeze`](../manifests/averitec_p1_launch_freeze_v1.json)。

这轮终止的是“给同一 dense corpus 增加 typed query prefix/slot assignment 就能产生 utility”的路线，而不是增加新 gate
的理由。若继续总目标，下一独立 study 必须更换 source/cohort，并把 action family 改成会形成不同证据集合的
decomposition→evidence-unit coverage→complementarity-aware set selection；evaluator 也必须评价 set-level marginal
coverage，而不是继续在已消费 AVeriTeC anchor 上调整 ridge、recipe、alpha 或关键词。只有该新机制在独立 A_hold
先真实超过 RAW 与 official HippoRAG、再晋升并改善预冻结 M_search，才能同时关闭现实三臂缺口和 L5。

### 8.24 WiCE P0 一次资格化终止；没有形成 P1 或效果测量

AVeriTeC 后的新候选先按修正后的 study ordering 建立独立、公开、非评分的
`WICE_P0_PUBLIC_SCHEMA_TOPOLOGY_V1`，而不是直接把正式 source epoch 当 schema canary。P0 design、实现和
source-free tests 先在提交 `ace43078` 冻结；随后只在 311linux 并行下载 official WiCE commit
`ddeb6c18…9870f` 的 claim-level TRAIN/DEV/TEST 三个 raw blob，一次成功取得并由冻结实现同时验证 size、
whole-file SHA-256 与 Git blob SHA-1。没有执行 repository code，也没有形成 secret、cohort、action、model、
RAW/HippoRAG、qrel 或 score。

冻结 qualifier 对 TRAIN `1260` 行和 DEV `349` 行各解码一次；TEST 只做 identity/newline observation，
`json_decode_count=0`。official row/meta keyset 全部 exact，但预注册 contract 仍记录 `153` 个 aggregate anomaly：
TRAIN 中 55 个空白 evidence sentence、14 个 not-supported row 带非空 supporting set、31 个重复 alternative；
DEV 对应为 12、13、28。安全回执 self hash `5f1e5a73…15d19` 有效，终态因此是
`not_qualified_public_schema_anomalies`。

这不是 WiCE retrieval 的负结果，也不是 source 本身无效；准确分类是 **pre-efficacy source-contract
incompatible for this program**。但冻结 terminal policy 已事前规定：P0 failure 后不得在同源修改 parser/family/
capacity contract、重跑 P0 或形成 WiCE P1。因此 TEST 继续不解析，WiCE 不进入 A_hold/M_search。这里执行的是一次
资格化终止，不是新增 efficacy gate；下一次总目标尝试必须换独立 source/study/cohort，并复用已经 source-free
证明可执行的 set-marginal mechanism，而不能围绕这 153 个已观察 aggregate anomaly 反向扩规则。

## 九、下一步优先级与硬验收标准

| 优先级 | 工作 | 硬验收标准 |
|---|---|---|
| 完成但无 efficacy（MuSiQue M2） | F2→Q 与 one-shot M2 retention | Q 已冻结；M2 授权消费后因 managed-sandbox/bubblewrap `NETLINK_ROUTE` 权限失败，36 attempted / 24 terminal / 0 score；相同 digest 由 benchmark-free postdiagnostic 复现；终态 no replay，retention efficacy unknown，不撤销 M1 P promotion |
| 完成但未晋升（MuSiQue L5） | A_form/A_hold evaluator transition 与 F3→M3 utility | A_form/F3/A_hold/M3 各 84×12=1008 local terminal；challenger=`micro_worst_v1`，A_hold 12/29 vs 12/29、Wilson lower bound 相同、not promoted；M3 active=incumbent，18/29 vs 18/29、net 0；只可声称 formation/rejection wiring，不可声称 evaluator co-evolution |
| 完成（MuSiQue generation-one M1） | fresh official-DEV P vs RAW vs official HippoRAG one-shot retrieval | 96-item/8-block pack 在 outcome 前一次形成；F1-only P 冻结；M1 36/36 terminal 后离线评分；RAW 7/29、P 14/29、official 14/29，P−RAW 为 +7 support hits / 7 gain / 1 harm / 4 tie；0 Ruoli/generator/online evaluator/retry/replay/resample；promotion disposition 与 postflight attestation 均公开 |
| 完成（cross-family retrieval L3） | frozen P 的 HotpotQA transfer | 固定 SHA 的 HF-hosted distractor-validation conversion 在 row-open 前预注册，7,318 eligible 中 private-HMAC 一次取 12；P 无适配；36-party barrier、36/36 terminal、fresh postflight 后离线评分；RAW 11/24、P 21/24、official 20/24；P−RAW +10 / 7 gain / 0 harm / 5 tie；P−official +1；0 retry/replay/online evaluator。claim 限于 12-item source-support retrieval，不等同端到端 QA/完整 Hotpot/原 CMU JSON |
| 完成（窄 retrieval-only L4） | fresh Hotpot P/Q retained recursion | 新六分区 pack 排除旧 12 项；Q 仅在 F_Q 形成且不因 cross-fit 不稳定而重选；M_L4 96/96 terminal + postflight 后，P/Q/P+Q 为 36/48、40/48、43/48，P+Q−Q=+3、P+Q−P=+7，P forgetting=1；RAW 22/48、official 31/48。只支持 fixed-cohort source-support multi-generation contribution，不扩张为统计/等算力/端到端 QA claim |
| 完成但未晋升（fresh Hotpot L5） | behavior-distinct evaluator transition | A_form/F_search 的 program 与 observed action 均不同；A_hold 72/72 terminal，challenger 38/48、incumbent 41/48、净 −3，exact p=31/32，不 promotion、不 invalidation；M_search 未授权且未打开。同一 anchor 禁止换候选、补 gate 或重试 |
| 严格终止（final Hotpot portfolio acquisition） | genuinely new two-Q evaluator mechanism + continuation cohort | 机制、实现和 `[156,324)` cohort 均先冻结；corrected call 消费 marker 并在 post-marker 内存选择后，因 private-root 父目录缺失在首个 `os.mkdir` 失败；0 block/locator/receipt/score/model/online evaluator。整段 window 永久烧毁，不 retry/replay/resample，不做 Hotpot v4；portfolio efficacy unknown，既有 claims 不变 |
| 严格终止（MuSiQue residual portfolio A） | same-source residual two-Q evaluator test | `[96,264)` 在 row-zero preregistration 后一次 acquisition；A/F 各 4080/4080 terminal 并冻结 behavior-distinct actions；A_hold freeze 固定 288 路。正式调用消费 authorization、打开 48 项后，lazy-submit/early-result 的 committed ordering defect 使 barrier attempted=1/terminal=0；0 ranking/score/model/network。A_hold 烧毁、不 replay；无 promotion，M_search 永久未开；efficacy unknown，不追加同源 cohort |
| 完成但未晋升（fresh 2Wiki fixed-action transfer） | 独立 fresh-item A_hold 与 agent/RAW/official HippoRAG 对照 | official archive + 历史 1000-row denylist + 三 split collision exclusion 在 selection 前固定；A_hold 48 项按四 type 均衡。384/384 terminal 后 incumbent/challenger/P/official/RAW 为 111/110/110/99/56（总 support=120）；challenger−incumbent=−1、exact p=1，不晋升，M_search 未开。incumbent−official=+12、16 gain/4 harm、descriptive exact p=1549/262144；claim 限 item-local retrieval |
| 完成但未晋升（QASC direct-action evaluator） | 新领域的自动 recipe formation、A/F pair 与 RAW/P/official HippoRAG 对照 | 四块各 64 在 formation 前一次 acquisition；16 路 full-corpus BM25 两遍扫描 16,987,130 行，TEST 未开；A/F 2048 actions 后 pair behavior-distinct。A_hold incumbent/challenger support=67/66、U=90/84、净 U −6、exact p=1668987/2097152，不晋升；RAW/P/official support=19/38/103，official 44 complete、U=147。M_search 未授权未开，同源 QASC 终止 |
| 完成但未晋升（AVeriTeC typed QA-set evaluator） | 独立现实 fact-check source 上一次检验 E1/RAW/official HippoRAG 与 evaluator→M lifecycle | A_form/F/A_hold/M=108/36/36/36；v4 source-free 双 GPU canary、142-file launch binding 与唯一 formal service 均通过。A_hold E1−E0/RAW 为 36 tie、净 0、tail 1；19/36 action/top-5 实际改变但 utility 全不变。E1−Hippo aggregate 净 +3/2、tail 3/32，但 causal family 净 0，且该差与 RAW 完全相同，故不能归因 typed action。promotion=false、reality primary=false，formal controller 未读/未执行 M，L5=null；0 API/online evaluator/retry |
| 完成（窄 synthetic multiseed stability） | fresh 8-seed typed-graph Agent_R1 / RAW / official HippoRAG replication | 全新 v3 cohort 为 8×64=512；v5 单次 detached formal 完成 1536/1536 action，official/local 峰值并发 8/64，MiniLM 固定两段 8448。Agent/Hippo/RAW 总 U=1259/1232/1273；Agent−Hippo seed delta `[3,8,0,2,1,5,6,2]`，mean=3.375、7 positive/1 tie；Agent−RAW 总 U −14。+27 全在 DEF_TP1/TP2，其他 family 与 Hippo 完全相同。只支持固定 synthetic distribution 的窄机制稳定性，不是现实域 transfer、promotion、L4/L5 或总体优越性 |
| 完成但未晋升（HoVer joint graph/evaluator） | 新现实 derived closed-corpus 上六 typed actions、RAW、official HippoRAG 与 A/F/A_hold/M lifecycle | TRAIN-only private-HMAC 一次形成 A_form/F/A_hold/M=48/36/30/30 与 609-doc corpus；A_hold E0/RAW/Hippo 均 U=487/12、72 hits、16 complete，30/30 item utility tie；E1−E0=−47/12、2 gain/10 harm/18 tie、exact p=3739/4096。primary=false、promotion=false，M_search 未打开；0 online/network evaluator。claim 不等同 official HoVer/open-domain/family-out |
| 完成但 primary 未通过（BRIGHT fresh RESERVE） | 现实 reasoning-retrieval 上 retained P6 / RAW / candidate-restricted official HippoRAG core 三臂与 E1 counterfactual | fresh 45 项三 family 各 15；45/45 Qwen valid、135 intents 先于 join、45/45 HippoRAG terminal、单 launch 峰值并发 12、late label 仅开一次、0 external network。Agent/Hippo/RAW mean nDCG@10=`0.14538/0.13598/0.14874`；Agent−Hippo aggregate `+0.00939`，但 family delta=`−0.46468/+0.16826/+0.71916`（integer-sum scale）、7 gain/9 harm/29 tie，未跨 family 稳定；Agent−RAW=`−0.00336`。E1−P6=`−0.00495`，既有 non-promotion 被 fresh reserve 再次支持；不是 full-corpus BRIGHT、answer generation、SOTA 或 L5 positive |
| 完成但 primary 未通过（BRIGHT P9 prospective C_confirm） | 固定 semantic P9 对剩余同源 RESERVE 的一次前瞻五臂确认 | 在打开剩余内容前固定每 family rank 15–25（0-based），共 33 项并保留 4 项 untouched；33/33 generation valid、66 external intents、1 cross-encoder + 12 HippoRAG 最大并发、33/33 HippoRAG terminal、late label 一次、0 external network。P9/Hippo/RAW mean=`0.12338/0.09218/0.11431`；P9−Hippo=`+1,029,664,470` 且 family 全正，7 gain/1 harm/25 tie；P9−RAW=`+299,424,557`，但 family=`−72,732,371/0/+372,156,928`，6/1/26，故预注册 primary=false。P9 含 RAW+Hippo+CE，结果只支持额外 ensemble 的同源增量，不支持等算力/SOTA/L5 |
| implementation-invalid（BRIGHT P14→P17 all-remote） | P13 在 fresh Earth Science/Psychology/Sustainable Living complete cases 上相对 RAW 与 candidate-restricted HippoRAG 的方向性三臂确认 | P14 本地在 12 个 HippoRAG terminal 后因机器不可用被用户中止；P15 迁移 gpu1 后无 remote action result；P16 在 HMAC 前因 source capacity 失败。P17 在 311linux 完成 27/27 HippoRAG terminal 并 seal 24 个 action，0 external network/reuse；但实际 HippoRAG 峰值并发=9，违反冻结上限 8。偏差在 prelabel audit 发现，finalizer=0、gold/score=0、primary 未评价、efficacy unknown；禁止 replay/resample、改 candidate 或补 gate |
| 当前剩余（不新增 gate、不重用已评分 cohort 调参） | 同时闭合现实域稳定三臂净收益与 evaluator→untouched-search 因果链 | AVeriTeC P1 已把最后一次 infrastructure-unknown 更新为有效 efficacy negative：typed E1 对 RAW 为 36/36 tie，未晋升，M 未执行；P9 的三个 family 对 Hippo 正向但 Biology/Economics 未严格高于 RAW，仍是此前最接近现实三臂 primary 的结果。下一轮只能用全新 study/source/cohort，把 action 换成真正改变 evidence-set coverage 的机制，并一次冻结 A_hold promotion 与 M_search；不能在 AVeriTeC anchor 上调 ridge/recipe/alpha 或补 gate。若不另立研究，就以“窄 L3/L4 positive、现实域无稳定双基线优势、L5 未达到”收束 |
| 完成（exact-domain L2/L3 instance） | Replication C promotion、controls disposition 与 one-shot sealed | development 8/8 gain、四 fold 各 +2；operator-only output 8/8 exact match；sealed 4/4 gain、8 路最大并发、8/8 network-none verifier receipts、0 retry/replay/online judge；两条盲化事件完整披露，claim 限于固定 SEC-13F treatment |
| 完成 | 冻结 evaluator-owned promotion policy | 已由 protocol 绑定完整 spec；candidate 只能收紧；对抗测试通过 |
| 完成 | 收紧外部 action/fallback contract | 4 类 prompt/self-check lowering；6 类 unsupported op fail closed；observed fallback 不再由字符串伪造 |
| 完成 | 冻结 offline-ready 范围 | 86-item manifests 保留旧 split；readiness matrix/static preflight 均 `blockers=[]`，无模型调用 |
| 完成（v3.3 历史） | 全 manifest runtime prewarm | cache-only 86/86、47 images、7 verifier runtimes；无 agent、无 sealed scoring；不作为 v3.4 receipt |
| 完成（v3.4 历史） | clean commit、lock 与 v4 prewarm | lock 绑定运行时 scoped clean commit `ad66d5a2`、claim-eligible、0 validation issue；cache-only prewarm 86/86、0 model call、0 sealed scoring |
| 本次排除（非稳定性结论） | 64 MiB fuse 作为本 batch 的直接 blocker | v3.3 38/38 均低于 64 MiB；最大 40.6 MB，video-1 为 19.69 MB；0 cap/provider error。canary/full 波动为 1.47/19.69 MB，尚无跨运行稳定性证据 |
| 完成（零模型） | 定位 model-only execution boundary | 根因为 Codex 0.144.1 丢弃 `tools.web_search=false`；canonical 顶层 disabled 的 loopback 为 7 tools / 0 web，旧键阳性对照为 8 tools / 1 hosted web；未调用模型、未评分 |
| 完成（实现与离线注入） | 执行 action budget | `codex_action_start_v1` 在第 N 个 `item.started` 终止 PGID，并按 task/TID 清除 dedicated-container 基线后的 live task；异常退出、malformed、N+1、`setsid`、zombie leader/live worker、残留 descendant 与 evidence tamper 均 fail closed；所有 arms 统一 action-step cost，不再混合 token/step |
| 完成（机制） | v3.4 clean runtime canary | lock/prewarm、PATH、host-readable receipt 均已验证；max2 v5 为 2-step valid truncation，本地 verifier 有效，0 remote tool，全部 agent task 已退出 |
| 完成（协议版本化） | v3.5 serial execution policy | 五个在线 phase 的 `parallel_workers` 全部由 4 改为 1；其余 model/subset/budget/offline evaluator/retry/circuit/search/promotion/sealed 合同不变 |
| 完成（容量验证） | v3.5 pre-fix serial train | clean lock/prewarm，38/38 valid、9 success、0 provider/cap/action/tool/verifier invalid；约 99 分钟；旧 rows 不跨进程复用 |
| 完成（确定性修复） | repair branch identity | parent ID + status-independent parent hash + depth + canonical candidate content 派生 ID；模型 ID/status 不控制主键或生命周期；真实 depth-2 collision 与 archive fail-closed 对照通过 |
| 完成（异常边界） | malformed proposal isolation + claim binding | post-transport envelope/parse failure typed 化；root 原子 replay、repair branch-local、整代 validation/promotion blocked；report/freeze 防失败 claim 篡改 |
| 完成（第二次容量验证） | v3.5 repairid01 serial train | 38/38 valid、9 success、0 provider/cap/action/tool/verifier invalid；5,188.542 trial-seconds；repair malformed response 在 validation 前暴露，旧 rows 不复用 |
| 完成（首次全闭环，非 clean claim） | response-fix fresh-root development | 新 lock/prewarm、38/38 all-valid train、proposal/repair、双臂 paired validation/promotion/report/archive 均实际完成；第一代 32/32 pair valid、sealed=false；第二代 no-rec 被 503/circuit 污染为 9 invalid，故“整轮 0 invalid”硬标准仍未满足 |
| 完成（v3.6 实现，live 待验） | invalid counterfactual lifecycle/claim | invalid/provider/budget mismatch terminal non-claim，不增加 consecutive non-promotion；archive score invalid；report/freeze 独立重算；legacy 污染报告同样拒绝；mismatch bundle 不缓存 |
| P1 | 递归因果归因 | 两臂共享 train evidence 和 roots，唯一差异是 repair；behavior-identical 时 effect 报 N/A，不重采样 |
| 完成（v3.9 clean 负结果） | contrastive trigger learning 首次 full live 验证 | 38/38 valid train、16/16 valid pairs、0 provider/infra/mismatch；两代 candidate 均仅 1/16 activation、0 gain/0 harm，证明 precision-first 已避开 false positive 但发生 coverage starvation |
| 完成（v3.10 clean 负结果） | proposal diversity 与 train-only coverage selection | exact 3 / 3 static pass；选中 2-family、6/6 precision root；activation 2/16 但 0 gain/0 harm；G2 两批 signature collapse 被 terminal reject；无 incumbent |
| 完成（v3.11 机制有效、性能/claim 失败） | actionable directive lowering 与 audit-only diversity | 38/38 valid train；exact 3 / 3 distinct signatures；court treatment 在激活题改变 trace/PDF 且 66→16 actions，但 0→0；no-rec 因未激活 raw poster 超 64 MiB non-claim；recursive repair 返回 batch envelope 而 non-claim；无 incumbent |
| 完成（v3.12 clean 负结果；repair path 未触发） | singular repair scope + candidate-formation 诊断 | 56/56 valid actual trials、0 provider/infra/mismatch；两代 exact3/3 static pass但所有 root 均只覆盖 1 family，selected court 仅激活 1/16、4/16 对 4/16、0 gain/0 harm；repair request=0；两臂无 incumbent |
| 完成（phase prerequisite 修复） | 禁止空 incumbent freeze/control | 误入的 partial controls 仅 2/96 invalid，已标 diagnostic-only/no-reuse；runner 无 incumbent 即结束，freeze producer 与 controls consumer 双向拒绝空候选；不改评分 gate |
| 完成（v3.13 clean 负结果） | train-only complementary policy bundle | 76/76 valid、0 provider/infra/mismatch；三套两成员 bundle 均为 TRAIN 5/5、2-family、0 success-FP，held-out activation 2/16；6 个 policy-on 全失败、0 gain/0 harm、无 incumbent。program-set routing/replay 正常，但 G2 cross-arm raw 未共享，且 7/7 三-family subset 被 capped target + size-first 排到第四 |
| 完成（v3.14 mixed-claim 负结果） | shared baseline cohort + support-aware tied-set ranking | lock/prewarm/smoke clean；62/62 attempted、38/38 valid train；G1 选中 7/7 三-family set并 activation 3/16，但 7 个 on 全失败、0 gain/0 harm。一条 recursive raw 68.66 MB 超 fuse 使 primary non-claim；31 次 valid baseline replay，invalid key 又跨臂执行一次；两份 archive 无 incumbent |
| 完成（v3.15 clean 负结果） | TRAIN-only material action delta + terminal-invalid attribution | commit `696a2954`、453/453 offline 后，clean lock、86/86 prewarm、smoke 与 57/57 valid live 完成；8/8 model calls、max online concurrency=1、0 provider/infra/budget/network/pair error；recursive G1/G2 与 no-rec G2 均 1/16、0 gain/0 harm，32 次 baseline 零执行 replay；两臂 claim-eligible non-promotion、incumbent null、无 downstream |
| 完成（v3.16/v3.17 proposal-only 负结果） | structural family-stratified proposal formation + artifact blueprint | 复用冻结 TRAIN receipt，0 source rerun、0 benchmark/evaluator call；v3.17 通过 8/9 feasibility，解决 family collapse/support/self-block/restatement，但第三候选仍绑定 2 个 failed primitives，故未授权 development |
| 完成（正式离线 PASS） | typed operator/capability grammar + causal action-span evidence | commit `b03c643a` 的唯一一次 decision 通过 9/9 predicates，既有 report/event/lock 精确复验通过。38 条 receipt-bound trace 含 655 个 action starts（max 61/100）、429 个 chronological allowlisted spans、70 failed、63 later scope-matched recoveries 与 208 discarded commands；3/3 graph/program materialize、9/9 tamper probes fail closed、primitive/locator disclosure=0、live model/backend/evaluator call=0。PASS 仅证明闭合表示的离线 feasibility，不授权 development |
| 完成（integration v1 暴露顺序 bug；v2 正式 PASS） | production opaque recipe selection | 提交 `ad6a8314` 接入真实 proposer/evolution；live smoke 发现 ledger 绑定晚于 runner construction。提交 `8caba466` 修复，正式 v2 13/13 predicates、12/12 tamper、exact replay、0 live call；提交 `9b4623f9` 冻结。只证明 selection/provenance，不证明 capability execution |
| 完成（v3.18r1 机械闭环；performance 降级为 mixed-validity diagnostic） | typed selection live development | 38/38 TRAIN、16/16 runner-valid logical pairs、58 actual trials、12/12 model attempts、max agent concurrency=38；两臂两代 3/16 对 3/16、0/0、incumbent null。organize 缺 100 PDF，stock 8/10→4/10→3/10 被 binary success 压成 tie，故不得解释为 clean action-utility negative |
| 完成（v3.19 task closure） | offline task input/dependency closure | manifest v2 绑定 transport-stable source environment；preparation receipt 冻结 11-item ledger / 126 objects；11/11 affected images 重建并以 immutable ID 无网络检查；86-worker v5 cache-only prewarm 86/86、closure 11/11、0 model/evaluator/sealed scoring。evaluator 与 promotion gate 未改变 |
| 完成（typed-portable formal integration PASS） | capability-backed portable artifact role 的 pre-agent 只读 sidecar | 唯一一次正式 run + exact replay PASS，decision `a151ca52…e7957e`；3 项真实 Docker canary、production v3.20 loader、exact image 与 cleanup 通过，0 model/task-backend/evaluator call。只声明 evidence profile/inventory sidecar，write/render/move 不属于 capability effect；该次性 development 授权已使用 |
| 完成（v3.20 clean negative） | 验证 portable treatment 的真实 task utility | 38/38 TRAIN、16/16 baseline、6/6 policy-on valid；两代均 3/16 activation、5/16 对 5/16、0 gain/0 harm、incumbent null。G1/G2 行为不同，但 6 份 trace 都未消费 sidecar；已按停止条件终止，不新增 recipe prompt、selector 或评分 gate |
| 完成（consumed、non-claim diagnostic） | verified profile launch-prompt delivery | 6/6 valid、actual peak concurrency=6、runtime receipt binding PASS；G1/G2 均 0/3，delivery delta=0/6、RAW utility=0/6；明确不声明 semantic consumption 或 task causality |
| 完成（execution-contract TRAIN ranking） | 改变候选搜索对象并按实际离线 outcome 排名 | 14 candidates × 38 TRAIN=532：56 active + 476 frozen-RAW replay；56/56 active valid，online judge=0、validation/test=false。`72c5…` 为 1 recovery / 0 regression；`4033…` capacity exact retry 后为 valid failure。未应用 promotion gate |
| 完成（证据边界审计） | 区分 in-sample ranking 与 transferable signal | 56 个 active route 全部包含被评价题的 source evidence，strict item-out=0；`72c5…` 不是 incumbent、L2 learning 或迁移证据 |
| 完成（targeted family item-out falsification；有 post-selection bias） | 检验 organize-2 阳性是否扩展到同 family 的另外两题 | 三个 fold 都只用其余两题构图/派生 contract，support=2、max artifacts=6；3/3 active valid + 111 RAW replay，结果依次为 false→true、false→false、false→false，即 1/3 recovery、0 regression。`unbiased_crossfit=false`；已否定 family-wide signal并停止该候选 |
| 完成（trace-refined candidate negative；仍有 design leakage） | 检验 destination/evidence/reopen 三个 generic invariant 是否提高 organize family recovery coverage | 提交 `baa3230a` 在 actual 前冻结 manifest=`da85625c…5483` 与三折 hashes；三路 Plus 同时启动，3/3 active valid + 111 RAW replay，结果为 false→false、false→false、false→true，仍是 1/3 recovery、0 regression。阳性 fold 改变但 coverage 未提高，故停止该分支 |
| 完成（typed-assignment representation negative；历史知情） | 检验 content evidence→typed plan→harness reconciliation 是否提高多折 recovery coverage | 最终提交 `0eba5b7c` 后三路 Pro 同时启动，3/3 active valid + 111 RAW replay；每路 103/103 evidence/assignment/reopen/hash-match，但结果仍为 false→false、false→false、false→true，1/3 < 预注册 2/3。org2/org5 wrong-subject=1/32，证明剩余方差在 trial 内 semantic classifier；representation stopped |
| STOP | 重跑相同 organize 题、继续修改同类 prompt/keyword、立即消耗 fresh split、freeze/controls/family-out/HippoRAG/sealed | 三类候选都只有 1/3；无 incumbent。2/5/6 已全部成为设计数据，不得把再调参后的同 cohort 包装成 cross-fit generalization，也不得用更多自然语言 invariant/gate 追逐单折成功 |
| 完成（operator 已冻结；prospective acquisition invalid） | 冻结的本地 semantic assignment operator | commit `2f804812` 固定 MiniLM/115 条 consumed TRAIN/OVR 参数/default/runtime；两次事前注册且 period-disjoint 的 public-OA acquisition 分别在 trapped-ion 40-pool 与 LLM 120-pool 供给不足处、prediction 前 fail closed。合计 operator call=0、outcome=0；efficacy unknown，不是 PASS/FAIL/incumbent |
| STOP | 第三次 public-OA acquisition、拼接 partial/reference PDF 或 synthetic 文献分类替代 task effect | 本地只存在 6 个 organize instance：2/5/6 已作 TRAIN、3 已消费、1/4 sealed；其余 reference/HF cache 不具备五类独立 gold。两批 partial 已成为关闭的 acquisition data。不得换 period/source、扩 pool、读取 sealed 或用人为措辞的分类 PASS 冒充 L2 utility |
| 完成（native verifier false positive；candidate stopped） | SC-100 semantic field compiler | commit `edb88957`、manifest `9388b249…0fb1` 事前冻结 candidate `86319d63…10bb`、三题 plan、镜像与 3+3 最大并发。native offline verifier 报 `1/0/0→1/1/1`，但强制 PDF 渲染审计发现 item4 被告地址实际为 `550184 and lives at 245 Mission St Apt 9`；substring test 漏掉 extra content。语义改判 `1/0/0→1/0/1`，仅 1 gain、0 harm，未达 2 gain；result `1460d8f6…fc1d0`，不在同三题修 parser 或重跑 |
| 完成（instrument qualified；shadow candidate stopped） | 独立 SC-100 synthetic shadow 与离线 oracle | instrument 先通过 2/2 canary + 5/5 mutant；唯一 frozen 24-case shadow 随后为 required 0/12、true-negative 5/6、coverage 0/6，18 个 task-valid case 全在 parser 阶段 reject，oracle call=0。candidate-class 失败，不补 regex、不重跑 |
| 完成（formation only；不是 cross-fit） | frozen financial semantic structured extractor | commit `c66e3a73` 固定 candidate/assets/runtime；financial-1/-3/-5 formation replay 为 3/3 对历史 RAW 0/3，但 `in_sample_formation_replay=true`、`cross_fit=false`。这修订了此前“先通过 deterministic TRAIN cross-fit 才消费 fresh”的 spend-control 路线；不能倒写成已通过 cross-fit |
| 完成（single-item existence evidence） | fresh paired financial efficacy | split/treatment 在 outcome 前冻结；9 RAW + 1 candidate 共 10 次 Plus model call 最大并发，offline only。scheduler-loss 后只恢复 post-agent stage，0 model/operator/verifier replay；financial-4 为 false→true、+1，active pair valid。runner 非 pristine、cohort 有 1 条 inactive audit invalid，无 promotion/incumbent |
| 完成但 primary invalid（不 retry） | 固定 financial candidate 的独立 SEC 13F multi-fold period-out 复验 | 16 路 RAW/candidate 最大并发、Plus、offline-only；15 valid + 1 RAW residual-process fail closed。7 个完整 pair 为 2/7 对 1/7、+14.29pp、0 observed regression，但唯一 gain 只在 fold 2，八对缺失值界 `[0,+12.5pp]`。没有 stable replicated gain，不 promotion；4 sealed 未访问 |
| NEXT（候选变化，不新增 gate） | contract-derived typed SEC-13F operator + 全新 untouched measurement | 用公开 task contract 实现完整 stock-class ontology、NFKC/punctuation-insensitive identity、latest eligible accession、aggregate 与 exact tie-break；当前 8 题只能作 consumed regression。新 candidate、period/source、selection 和 sanitized offline verifier 必须在任何新 outcome 前冻结；通过完整多折 paired batch 后才谈 incumbent/controls/family-out/sealed |
| P0 infrastructure（仅未来运行） | durable post-agent resume + terminal event auditor | runner 持久化 post-agent checkpoint，恢复不得重放模型/operator/verifier；auditor 区分 transient `error` 后 `turn.completed` 与 terminal `turn.failed`。这是 evidence plumbing，不是 performance gate，也不得回改本轮 temperature-4 |
| P2 | 多 clade archive | 同 epoch 至少两个 clade 可继续扩展；node 绑定 protocol/evidence/promotion hashes，并报告 retention 与 branch productivity |
| P2 | evaluator co-evolution | 独立 anchor challenger、epoch transition、selective invalidation 和旧 incumbent re-evaluation 实际执行后再作主张 |

以下为按时间保留的执行记录（当前证据边界见第 74 项）：

1. 已完成：审阅并提交 protocol/action/subset 改动以及 3 个新 manifest/receipt 文件；
2. 已完成：在 clean scoped commit 上重建 claim-eligible lock 和 86-item content-hashed prewarm receipt；
3. 已执行但未形成性能证据：第一次 full development 在 26 个有效 train observation 后，
   被 provider 429/circuit 与一个既有 hard-byte fuse fail-closed；未进入 proposal/validation；
4. 已完成：一个单题、5-step、非 claim transport canary 得到有效 offline-verifier
   observation，确认 provider 已从 429 恢复；
5. 已完成：同协议 fresh-root rerun 再次在 `court-form-filling-6` 超过 32 MiB；按 stop
   rule 中止，v3.1 判 execution-infeasible；
6. 已完成设计：新建 v3.2，仅把统一 fuse 一次性版本化为 64 MiB，其余实验合同不变；
7. 已完成：v3.2 clean lock/prewarm 均通过，64 MiB 未触发；full run 在 8 个有效 train
   observation 后被 provider 的“无可用 distributor channel”503/429 熔断；
8. 已完成：新 GPT Pro credential 的 API/Codex canary 证明同一路由恢复；`gptpro01`
   跑完 38 train，暴露两个 deterministic receipt false-negative，而非 provider/cap 问题；
9. 已完成：receipt auditor 改为绑定实际 runtime profile/command，136/136 tests 通过；
   `gptpro03` 中两项 temperature 均以 7-test CTRF valid failure 完成；
10. 已完成：`gptpro03` 跑完 38 train，但 `video-object-counting-1` 以 71.1 MB 超过
    冻结 64 MiB；37 valid / 1 invalid，proposal 被 fail-closed，v3.2 判 execution-infeasible；
11. 已完成：v3.3 只版本化 Codex execution treatment，v3.1/v3.2 保持旧 mapping；
    150/150 tests、strict-config、claim lock 与 86/86 prewarm 通过；
12. 已完成：video-1 canary 为 1.47 MB valid failure；full run 中 video-1 为 19.69 MB，
    38 个 trial 最大 40.6 MB、0 cap/provider，本次排除 fuse 作为直接 blocker，但尚未证明
    跨运行稳定性；
13. 已执行并 fail-closed：full train 为 37 valid / 1 `web_search` policy invalid；
    proposal/counterfactual/sealed 均为 0，四份 report/archive 未生成。不得重试或通过新 gate
    洗掉 invalid；v3.3 已冻结为不可复用的诊断证据；
    v3.1–v3.4 仅作为 immutable evidence，当前代码仍可按其声明的 schema 验证历史
    receipt，但不承诺这些协议在当前 commit backward-executable；
14. 已完成零模型定位：Codex 0.144.1 把旧 boolean key 丢弃并默认暴露 cached hosted
    search；canonical 顶层 disabled 的真实 wire 捕获无 web，旧键阳性对照稳定检出 web；
15. 已完成 v3.4 最小实现与离线注入：同一 execution policy 同时冻结 model-only tool
    exposure、可执行 action budget、dedicated-container task/TID 清理 receipt、token completeness 和统一 action-step
    promotion cost；未改变 cap/subset/evaluator/promotion/sealed；
16. 已完成：v3.4 clean claim lock、新 shared runtime 与 v4 86/86 cache-only prewarm；receipt
    显式记录 test infrastructure inspected、sealed scoring=false、test bytes exposed to model=false；
17. 已诊断并修复：max2 canary v1 在模型请求前因 shell 的 `PATH=... rm && node` 作用域失败；
    `995e6446` 使用固定 runtime PATH 和 node/codex 绝对路径；
18. 已完成：canary v2/v3 的 no-distributor 503 被 `ba0f36cf` 正确归类；v4 到达模型和 verifier
    后暴露 root-owned `0600` audit artifact，`1df3092a` / `ad66d5a2` 改为显式 `0644` 并补生产断言；
19. 已完成：max2 v5 为 evaluation-valid 的 2-action 截断与本地 verifier failure，0 remote tool，
    action receipt 和 process cleanup 均 valid，因此一次 fresh-root development 获准启动；
20. 已执行并 fail-closed：四并发 development 的 38 个 outcome 为 17 valid（3 success）、4 个
    `provider_rate_limit`、17 个 circuit skip；0 cap/action/tool/verifier violation，未进入 proposal，
    四份 report/archive 未生成，sealed 未触碰，17 条 valid 不得跨 run 拼接；
21. 已完成 v3.5 最小版本化与容量验证：五个在线 phase 的 worker 统一从 4 改为 1；新 lock/
    prewarm 后 serial train 为 38/38 valid、9 success、0 provider/cap/action/tool/verifier invalid；
22. 已执行并 fail-closed：proposal 返回 3 roots，第三个的 depth-1/depth-2 repair 复用同一模型 ID
    但 payload 不同，archive 在 paired validation 前抛 collision；四份 report/archive 未生成，
    sealed 未触碰，38 条 train 不得跨进程拼接；
23. 已完成 repair identity 最小修复：repair ID 改由 parent content/depth/canonical candidate
    content 确定性派生，model ID/status 不控制主键或 lifecycle，archive 冲突保护不放松；
24. 已执行第二个 fresh root：38/38 valid、9 success、0 provider/cap/action/tool/verifier invalid，
    proposal 返回 3 roots；第一个 repair transport 成功但 response envelope malformed，裸 ValueError
    在 validation 前终止；report/archive/sealed 仍为 0，38 条 train 不跨进程复用；
25. 已完成 `d70562de` 与当时的 216/216 离线回归：malformed root/repair 进入既有 typed failure isolation，
    root replay 原子、repair branch-local、整代不 validation/promotion；report/freeze 强制 non-claim。
26. 已执行 response-fix fresh root：clean lock、86/86 prewarm 与 38/38 valid train 后，首次完整
    走通 proposal、真实 repair、recursive/no-recursive paired validation、两代停止以及四份
    report/archive。clean g1 recursive 为 0 gain/2 harm；no-recursive 为 1 gain/0 harm 但 lower bound
    -0.0176，均未 promotion。两臂 incumbent 仍为空，sealed/test 访问为 0；不生成空 control claim。
27. 已诊断 no-rec g2 的 evidence 语义缺口：一次 Ruoli 503 `provider_model_unavailable` 打开 circuit，
    之后 8 个 candidate skip，合计 9 invalid pair，却被旧 lifecycle 计为第二次普通 non-promotion，
    report/score 仍显示 claim-eligible/valid。下一步一次性关闭该分类缺口并版本化 v3.6 contrastive
    trigger learning；不新增 evaluator gate，不改 promotion/split/model/budget，旧 rows 不复用。
    有 retained validation gain 后再做 family-out，最后才增加 multi-clade/evaluator mutation；
28. 已完成 `01608e1e` 与当时的 254/254 离线回归：v3.6 manifest、success controls、exact
    contrastive selection、invalid terminal non-claim、mismatch-safe replay、pair diagnostics、
    legacy/v3.6 report schema 隔离和 freeze 重算均已落地。真实 v3.5 recursive report 可按旧
    schema 解析，污染的 no-recursive report 现因 invalid evidence 明确拒绝；
29. 已运行 v3.6 serial diagnostic：clean lock、86/86 prewarm、38/38 valid train、7 success、
    31 residual；单一 root 的 train activation precision 为 26/27，success false positive 为 1。
    仅完成 2/16 validation pairs（4 个 valid 0→0 trial），第 5 个 trial 中断，无 report/archive/
    promotion/freeze/family-out/sealed。随后一次性版本化 v3.7：五个在线 phase 的跨题 worker
    统一 1→6，invalid retry worker 仍为 1，同题 off/on 仍串行；不新增 gate，不改 evaluator、
    learning/promotion、route、split 或预算，且不复用 v3.6 rows；
30. 已运行 v3.7 six-worker fresh root：clean lock、86/86 prewarm 均通过，首批六个 train
    请求全部返回 `provider_rate_limit`，一次 circuit open 后其余 30 个 request slot 本地跳过。
    无 valid training bundle、proposal、report、archive、family-out 或 sealed access。v3.8 只把五个
    online phase 的 worker 从 6 改为 2，其他字段归一化后完全不变；从新 lock/root 开始，
    不拼接 v3.7 rows，也不新增 gate；
31. 已运行 v3.8 two-worker fresh root：完成 16 个 valid train rows 后，两条同时在途请求均返回
    `provider_model_unavailable`，既有 circuit 跳过其余 20。无 valid bundle/proposal/report/archive/
    sealed。v3.9 采用旧快版真正的两级并发结构：6 个题级 worker 共用 1 个进程级在线 agent
    semaphore，slot 只包围 `docker exec ... codex exec`；docker run、准备、离线 verifier 不取 slot。
    slot policy/count 进入 protocol/lock/plan/fairness/freeze，异常 finally 释放；不新增评分 gate；
32. 已完成 `8d862e8f` 与 289/289 离线回归，v3.9 clean lock 和 86/86 prewarm 均通过。
    随后两个 fresh root 都在第 0 条 benchmark trial 前停止：health probe 的两次 transport retry
    均为 HTTP 503，`skilllearn_trial_started=0`。其后 10 次低频恢复探针仍失败。当前唯一 blocker
    是 Ruoli route availability；不再版本化 worker、gate、retry 或 evaluator。恢复后从新 clean root
    启动，不复用任何 v3.7-v3.9 失败行；
33. 已完成 v3.9 lower-cost credential fresh root：clean lock、86/86 prewarm、38/38 valid train、
    56/56 actual trials、56/56 slot acquire/release，maximum online concurrency=1，0 provider/circuit/
    infra/budget/mismatch failure。recursive 两代均只激活 1/16 validation，candidate/raw 都为 3/16、
    0 gain/0 harm；no-recursive 两代 root 均在 train-only static audit 被拒绝。四份 report/archive
    完整落盘但两臂 `incumbent_id=null`，sealed/test 未访问。当前 blocker 已从 transport 转为
    candidate diversity/configuration coverage；下一步不新增或放宽 gate，不 freeze 空 incumbent；
34. 已实现 v3.10 bounded pre-gate revision 与 309/309 离线回归：单次 root response 必须 exact 3，
    三者在 failed train rows 上的 activation signature 两两不同；少/多/重复均 typed terminal，0 retry。
    proposal response budget 由 protocol 固定为 8,000 tokens。selection 只用 train labels，先最小化
    `ceil(existing minimum_activation_rate × distinct train families)` 的 capped coverage deficit，再比较
    exact precision、success false positives、failure support 与复杂度；同一 objective 也进入 repair
    request。model/provider、6×outer/1×model 调度、offline evaluator、promotion thresholds、split、
    action/network budget、retry、controls 与 sealed policy 均未改变。v3.9 rows 不复用，须新 lock/root。
35. 已完成 v3.10 fresh root：clean lock、86/86 prewarm、38/38 valid train（6 success / 32 residual）、
    56/56 actual trials、16/16 valid pairs、0 provider/infra/budget/mismatch。第一代 exact 3 全部 static
    pass，coverage-first 选中 2-family/6-of-6 failure root并在 validation 激活 2/16，但 candidate/raw
    都为 3/16、0 gain/0 harm。第二代 recursive/no-rec 各自收到 transport/JSON/exact-count 成功的
    三候选 response，却都只有一个 distinct activation signature，旧合同 terminal reject；两臂
    proposal failure non-claim、`incumbent_id=null`、sealed=false；
36. 已实现并运行 v3.11 bounded actionability revision：38/38 valid train（5 success / 33 residual），
    exact-three 得到 3 个 distinct activation signatures；2 个 root 静态通过。选中的 court policy 在
    validation 激活 1/16，实际改变 trace/PDF 并将 action starts 从 66 降到 16，但 task success 仍
    0→0；总计 raw/candidate 均 4/16、0 gain/0 harm。no-rec 又因一个未激活 raw poster trial 超过
    冻结 64 MiB 而 terminal non-claim。recursive repair transport/JSON 成功，却返回 root-shaped
    `hypotheses`；调用链复核确认真实 repair payload 没有 batch contract，但也没有 versioned singular
    response override，整代在 validation 前 non-claim。
    两臂 `incumbent_id=null`、sealed=false，不 freeze、不跑 controls/family-out/HippoRAG；
37. 已实现 v3.12 bounded repair-scope revision：repair 保留 `train_coverage_objective`，并以
    `single_candidate_excludes_root_batch_contract_v1` 绑定 top-level one-object/`hypothesis` response、
    system prompt、plan 与 freeze；若调用方意外提供 root batch contract，`revise()` 防御性删除。
    root exact-three、model/provider、6×outer/1×model 调度、offline evaluator、promotion gate、split、
    action/64 MiB network budget、retry、controls 与 sealed 全部不变；须新 clean lock/root，v3.11 rows
    不复用。
38. 已完成 v3.12 fresh root：clean commit `9c692b2d`、86/86 prewarm、56/56 actual external trials
    全部 valid；38 train 为 8 success / 30 residual，最大网络流量 35,070,000 bytes，0 provider/infra/
    action/network/budget/mismatch。两代 exact-three 均 3/3 static pass且三种 distinct signature，
    但每个 root 都只覆盖一个 train family；selected court treatments 在同一 held-out item 上把 raw 9
    actions 改为 32/43 actions，仍均 0→0。两臂每代都是 1/16 activation、raw/candidate 4/16、
    0 gain/0 harm，最终 archive 字节相同且 `incumbent_id=null`。12 个 static node 全通过，故本轮
    repair request=0，不能声称 full-run singular repair path 已实跑。
39. 同一旧 runner 在 `selected_candidate_available=false` 后仍无条件进入 controls，生成空
    `promoted_v2`/no-rec program set并启动 8 个 trial；立即中止后仅 2/96 records 落盘且两条都因
    缺 action-budget receipt invalid，无 report、family-out、HippoRAG、sealed/test access。该批已有
    machine-readable diagnostic-only/no-reuse marker。现已统一 phase invariant：all-development 无真实
    recursive incumbent 即正常结束，paper freeze 拒绝空 archive，control consumer 也拒绝旧空 receipt。
    345/345 离线回归通过；没有新增或放宽 performance gate。
40. 已实现 v3.13 complementary program-set revision 与 375/375 离线回归。exact-three static-valid
    roots 的最多 7 个非空子集只在 TRAIN 上枚举；排序依次为 union precision、capped family deficit、
    success false positive、overlap、bundle size、failure support、complexity 与 canonical set hash。
    因此低精度成员不会为凑 coverage 被强塞，也没有 minimum bundle size。确定一个 delta set 后才
    进行一次 paired validation。SkillLearn runner 分别绑定 delta/full/per-item matched set，任一新成员
    命中才执行一次 policy-on，否则严格 alias baseline；program-set replay 对顺序不敏感但区分 `{A}`
    与 `{A,B}`。archive rejection 只拒绝 bundle node、成员留 shadow；report/freeze 强校验 selected IDs、
    treatment-set hash、baseline union、node/status。Promotion 仍是
    `evaluator_owned_paired_validation_v2`，成员自约束保守聚合，所有 protocol 数值阈值、model/provider、
    6×outer/1×model、offline evaluator、split、64 MiB fuse、retry、controls 与 sealed 均未改变。
    v3.12 rows 不复用。
41. v3.13 正式 development 已完成：第一次 partial invocation 被独立标为 diagnostic-only，冻结的
    lock、86/86 prewarm 和 smoke 因 Plus/Pro credential 都服务同一 `gpt-5.4-mini` route、model/provider
    identity 不变而保留，正式 development 使用新的
    events/work tree。76/76 trials 全 valid 且均 attempt-1：38 train-off、32 validation-off、6 validation-on，
    0 provider/infra/retry/action/network/mismatch，最大 69/100 actions 与 62,200,000/67,108,864 bytes。
    每套 bundle 都把 poster+court 两个 TRAIN-perfect 成员合并并激活 2/16；6 个 on 与对应 off 全失败，
    四代决策均 0 gain/0 harm。两臂 stop=`consecutive_non_promotion_limit`、`incumbent_id=null`，无 freeze/
    controls/family-out/HippoRAG/sealed/test。recursive repair path 已无错误地实跑。G2 发现 cross-arm
    baseline cache 未共享；no-recursive 的三成员 7/7 subset 又被 capped-family/size-first objective 排第四。
42. 已实现 v3.14 的两项有限修订。`behavior_identical_shared_validation_baseline_arm_replay_v2`
    以 baseline behavior/treatment 与冻结 task/runtime/fairness identity 为 key，去掉 challenger pair 元数据，
    在 recursive/no-recursive 与多代间共享只写一次的 valid policy-off cohort；invalid 不入缓存、冲突不覆写。
    `train_contrastive_complementary_family_support_bundle_precision_first_v2` 保留 precision、capped deficit、
    success false positive、overlap 的 leading order，仅在这些项相同时把 actual family count 与 failure support
    放到 bundle size 前。仍只固定一个 TRAIN set、只做一次 held-out paired validation；model/provider、
    evaluator、promotion thresholds、split、fuse、scheduler、controls/sealed 均不变。由于 protocol/code
    identity 变更，v3.14 使用新 lock、cache-only prewarm、smoke、run root；已有 model/runtime image 与
    依赖没有重下。
43. v3.14 live 已完成并触发停止 selector 的预设条件。commit=`2229d7af`，lock claim eligible，prewarm
    86/86，smoke 8/8 valid；formal development 62/62 attempted、61 valid / 1 hard-budget invalid，0 provider/
    model/slot/action/mismatch，max valid actions=55/100。38 条 train 全 valid（7 success / 31 residual）。G1
    selector 选中 court/dependency/poster 三成员 set，TRAIN 为 7/7、3-family、0/7 success-FP，held-out
    activation=3/16；recursive 两个 valid activation 与 no-recursive 三个 activation 都 0→0，no-recursive
    G2 poster singleton 的一个 activation 也 0→0。合计 7 个 policy-on 全失败。recursive court raw 使用
    68,660,000/67,108,864 bytes，primary report=`invalid_counterfactual_evidence` non-claim；no-recursive
    report 机械上 claim eligible、两代 non-promotion。valid baseline rows 共 replay 31 次，但 invalid key
    因“不入 cache”在 no-recursive 又执行一次，说明 terminal invalid 还没有跨 consumer 传播。两份 archive
    `incumbent_id=null`，无 freeze/controls/family-out/HippoRAG/sealed/test。下一步转向 action content：当前
    directives 没有补充实际 HEX、离线漏洞记录位置或新的表单操作，只是更清楚地复述 instruction。
44. 已实现并完成 clean live：v3.15 提交 `696a2954` 与 453/453 offline regression 通过。它把 material action
    delta 作为 TRAIN-only request-local prompt 与 audit-only 诊断，并以 allowlisted public-environment/
    policy-off trace facts 补充 proposal context；不读取 validation/test/verifier/solution，不允许 proposal
    外部工具，不把 audit 变成 reject/retry/repair/selection/promotion gate。terminal-invalid replay v3 在
    frozen same-request retry identity 完成后共享 non-evidence tombstone，report/freeze 再绑定两臂共同的
    首代 checkpoint 与 action-profile count/set hash。正式 root 通过 clean lock、86/86 cache-only prewarm
    与 smoke，完成 57/57 valid actual trials、8/8 model calls；TRAIN 为 6 success / 32 residual，max online
    concurrency=1，0 provider/infra/action-budget/network-cap/pair-mismatch。recursive G1/G2 都是 1/16、
    candidate/raw 4/16、0 gain/0 harm；no-rec G1 static reject，G2 为 1/16、0 gain/0 harm；32 次 baseline
    replay 零执行。两臂均 claim-eligible negative、`consecutive_non_promotion_limit`、`incumbent_id=null`，
    sealed/test=false，无 downstream。
45. v3.15 的 13 个 candidate audits（9 roots + 4 repairs）中 7 个 material、6 个 restatement-risk；7 个
    material delta 全是 `exact_constant_or_mapping`，concrete local tool、artifact manipulation、environment
    primitive 全为 0。9 个 roots 又全部坍缩到 `anthropic-poster`，从 v3.14 G1 的 3-family/7-support 退回
    单-family/2-support。因此先授权一次 gate 前的 structural family-stratified proposal-only feasibility，
    不直接启动新 development，也不新增/放宽 promotion gate。
46. v3.16 提交 `6ad5c156` 用冻结 v3.15 TRAIN receipt 形成三个 singular family slots。3/3 logical proposal
    calls 完成，0 source-agent/backend/evaluator/validation/test/sealed access，但 9 项 feasibility 中 6 项失败；
    没有 benchmark trial。v3.17 提交 `4f94e613` 只作最后一次结构修订：exact trigger、empty anti-trigger、
    deterministic reusable artifact 和固定 workflow blueprint。新结果通过 8/9：support=2/2/3、3/3 concrete
    tool、2/3 artifact manipulation、0 restatement/self-block；第三候选仍绑定 2 个 failed TRAIN primitives，
    故 `diagnostic_passed=false`。一次 retryable disconnect 已恢复，不改变 semantic negative 结论。
47. free-text family-slot 路线按预设停止。离线重建显示第三 slot 的两个 failed primitives 是通用 `file`/
    `python` executable，暴露了 failed-command 共现不等于 causal inadmissibility，且模型只获 count/hash 无法
    满足未知逐值 exclusion 的表示矛盾。下一 workstream 必须换成 typed operator/capability grammar 或
    artifact-operation graph，并以 causal action-span evidence 定义不可表达项；只允许一次 preregistered
    feasibility decision。没有真实 incumbent 时继续禁止 controls/family-out/HippoRAG/sealed。
48. causal action-span extractor、closed typed operator/artifact graph、opaque recipe-only selection、harness-owned
    materializer 与 single-decision lock/preregistration 已实现并冻结为 commit `b03c643a`。前置结构复核得到
    38/38 complete trace、429 allowlisted occurrences、70 failed、208 discarded、655 action starts、max 61/100。
    该路径使用 stored offline TRAIN outcomes、本地 contract validation 并哈希 unit-test source，但有 0 live
    model/task-backend/evaluator invocation，且未访问 validation/test/sealed split 或 verifier content。该表示只
    闭合 proposal selection；现有 lowering 仍是 prompt directive，非 restricted executor。
49. 唯一一次正式离线 decision 已按预注册命令完成：9/9 predicates PASS，decision hash
    `79acda9b9e393330b8418e5fea15f176236edf8ecf802d310d73862710ba8bfc`，report hash
    `aa1033429980cfc5881aa6b3ccf25609c3d80ce0514c1dc37d1188354789797d`；随后 `--verify-existing`
    对 report、9-event ledger 与 completed lock 精确复验通过。3/3 target graph/program materialize、9/9
    tamper probes fail closed、raw primitive/locator disclosure 均为 0，70 个 failed spans 中 63 个存在后续
    scope-matched recovery。结果收据见
    [`skilllearn_typed_operator_feasibility_result_v1.json`](../manifests/skilllearn_typed_operator_feasibility_result_v1.json)。
    该 PASS 只使 separately frozen typed-selection integration diagnostic freeze-eligible；它不验证 capability
    implementation、restricted executor 或 benchmark gain，也不授权 development。
50. production typed-selection integration v1 已接入真实 proposer/evolution，但 formal predicates 没有覆盖
    live harness construction 顺序；第一次 smoke 在 task trial 前以 `typed snapshot ledger binding is missing`
    fail closed。根因是 compiler/runner 先于 shared proposer registry 的 ledger binding 构造。提交 `8caba466`
    调整顺序并加入真实 harness construction regression；没有修改 evaluator、promotion 或 recipe acceptance。
51. 另行冻结的 integration v2 已通过 13/13 predicates、12/12 tamper probes 与 `--verify-existing` exact replay，
    implementation/decision/report hashes 均绑定，且 0 live model/backend/evaluator call。提交 `9b4623f9` 保留
    v1 失败证据并冻结 v2。result 明示 capability implementation 与 restricted executor 均未验证。
52. v3.18r1 fresh root 已以 38 个 item workers / 48 model slots 完成 38/38 TRAIN、16/16 logical paired
    validation、58 actual trials、12/12 model attempts；两臂两代均 activation=2/16、raw/candidate=3/16、
    gain=0、harm=0、`incumbent_id=null`，因此所有 downstream 正确跳过。recursive 产生更多 repair binding，
    但最终 selected treatment/nodes/scores 与 no-recursive 相同，不能声称 recursion gain。
53. 事后离线 audit 使该 performance 结论降级：全部 organize 镜像的 100-PDF heredoc layer 为 0B，当前
    selected TRAIN support 5 中有 3 条 organize，且 validation organize-3 不可完成；stock 虽输入完整，
    RAW/G1/G2 的 CTRF 为 8/10、4/10、3/10，却被二元 task success 投影为三个 0。报告的 runner-valid
    tie 不是行为 tie，也不是 typed operator efficacy 的 clean 反例。
54. 下一步固定为 task closure 与单条 capability-backed role vertical slice：先离线固化 organize PDFs 和
    D3 6.7.0，再以 `artifact role → current-item resolver → restricted capability → effect receipt` 消除 TRAIN
    literal locator 与 prompt-only activation。TRAIN/search 保留 per-test residual，最终 promotion gate 不变；
    禁止继续补 recipe prompt、selector、minimum-activation 或其他评分 gate。
55. task-input preparation v2 已冻结 11 项 per-item closure ledger 和 126 个实际注入对象；source binding 使用
    transport-stable environment hash，能拒绝 Dockerfile URL 漂移，同时不受 WSL staging 的 0644→0777 mode
    变化影响。冻结 file SHA 为 `73c25f8e…e1b8b06`、receipt 为 `8d1979e5…6771a16`、ledger 为
    `7ad7f448…9ac3a9`；cache-only regeneration 为 126/126 hit、0 download。
56. 11/11 affected images 已重建并经 immutable image ID、`--network none` 精确检查。v3.19 将 preparation
    ledger 同时接入 prewarm、development、controls 与 sealed cache；86-worker v5 cache-only prewarm 为
    86/86 PASS、closure-required/verified=11/11、failed=0、online-build=false、sealed-scoring=false，receipt
    为 `e7851d19…6e7839`。这恢复了实验输入构造，不是 performance claim。
57. 独立 portable-capability feasibility 已让冻结 stock recipe 在 item 3 上从 public instruction 解析当前
    `.tsv`、固定只读解析 50×14，并产生 `c45bd4…6744f8` effect receipt；compiler phase 1 又以显式 opt-in
    生成 locator-free per-item role metadata，并把其 hash 绑定到 treatment/manifest/source/event，默认路径
    byte-compatible。该 evidence 只支持 pre-agent read-only role，不把 write/render/move 计作 capability effect。
58. 2026-07-14 唯一一次正式 typed-portable 非评分 integration 已执行并由 exact replay 精确复验；decision
    hash 为 `a151ca52916101f0ea31b0d2f11c8fde8407f4410d175b1ac983e013d6e7957e`。三项真实 Docker
    canary 在 `--network none` 下通过 exact image、pre-agent sidecar 回读与 container cleanup；production
    v3.20 authorization loader 返回相同 projected ledger，legacy policy/full ledger 均 fail closed。model、
    task-backend、evaluator、verifier 与 score call 全为 0。PASS 允许建立 fresh v3.20 development root，
    但 `task_effect_claimed=false`、`recipe_operator_effect_claimed=false`，没有 incumbent 或 promotion。
59. fresh v3.20 development 已在独立 root 上以 38 item workers / 48 model slots 完成。lock
    `claim_eligible=true`，86/86 cache-only prewarm PASS；61 attempts 中 60 valid，由 38 TRAIN baseline、
    16 shared validation baseline 与 6 policy-on 组成。60/60 action-budget/verifier receipts 有效，
    max 73/100 actions，无 web/remote tool/runtime install、budget truncation 或 sealed/test 访问。两代均
    baseline/candidate 5/16、activation 3/16、0 gain/0 harm、16 binary ties，并因
    `insufficient_net_gain_count` 被拒；两臂 stop=`consecutive_non_promotion_limit`、
    `incumbent_id=null`。recursive/no-recursive archive 因 repair=0 而字节相同。非评分审计确认
    G1/G2 recipe、treatment、action trace 与 verifier 子测试不同，但 6 份 candidate trace 均没有
    留下显式读取或消费 portable sidecar 的可审计证据；`selection_change_count=0` 在该 adapter
    上只是二元 success 不变，不是
    行为不变。该路线按预注册停止，无 incumbent 前不进入 freeze、controls、family-out、
    HippoRAG transfer 或 sealed test。events SHA-256 为 `beb93367…b86e83e6`，recursive report
    SHA-256 为 `98664f6b…870cdf59`，shared archive hash 为 `58d55253…a05edd04`。
60. runtime delivery 已作为一个 bounded vertical slice 完成，而没有增加 performance gate。新 request
    字段默认关闭，旧 v3.20 request byte-compatible；显式启用时，verified profile 以 128 KiB/profile、
    256 KiB/capsule 的固定上界组成 canonical fragment，复制到容器并回读，然后把 runner-local Codex
    run template 中唯一的 `$(cat {instruction_file})` 绑定为同时读取原 instruction 与该 fragment。
    receipt 绑定 request/context/source/typed-binding/effect/profile/fragment/run-template/effective-prompt
    hashes，且明确记录 `semantic_consumption_claimed=false`、`task_effect_attributed=false`。
    2026-07-15 正式 TRAIN-only integration 在 stock、temperature、organize 三个 exact image 上 3 路并发
    PASS：production compile manifest `3ccddcfc…93ab3`，8/8 acceptance predicates、4/4 tamper probes、
    3/3 cleanup 与 exact no-model Docker replay，decision `7120d5a7…e415b0`；model、task-backend
    `run_task`、evaluator、verifier、score、
    promotion 均为 0。它解决了“profile 只在 host sidecar/metadata 中、没有进入模型 launch input”的机制
    断点，但不把 prompt delivery 偷换成模型理解、action causality 或 task utility。
61. 预注册的 consumed-development profile-consumption diagnostic 已严格用完 6 个新 policy-on trial
    的一次性预算：G1/G2 × stock、temperature、organize，0 retry、0 新 policy-off、0 proposal/training
    call，6 workers 的实际 peak agent concurrency=6。6/6 valid，offline verifier 与 runtime receipt
    binding 全部通过；G1/G2 均 0/3，delivery delta=0/6、stored-RAW utility=0/6。结果明确为
    `fresh_validation=false`、`claim_eligible=false`，没有 incumbent/promotion、test trial 或 sealed
    scoring；只检查 test infrastructure metadata，未读取 test task-input bytes。HippoRAG 因没有同构
    executable file-task arm而不运行。formal report hash 为 `6e7620ff…e321e`。
62. delivery-only 路线到此停止，但不把 0/6 写成 clean causal null。local-only CTRF 显示三类失败已
    收缩为 exact interaction contract、文件映射/completion audit 与 final-artifact metric semantics；
    temperature G2 又用 66 个 action starts 隐藏了单 action 调度/尝试的 140-point search，最终达到 3,000,056
    tokens / 1,763.18 秒。所以下一步改变的是候选生成空间和 TRAIN-only objective：task-local invariant、
    bounded operator、postcondition 与 single-source self-evaluation，而不是增加 gate 或立刻用同一候选
    消耗 fresh holdout。只有 deterministic TRAIN cross-fit 先出现 transferable action-utility signal，才
    进行一次新的 paired development。
63. execution-contract candidate grid 已完成 production compile 与最大并行 actual。非评分 integration 从
    v3.20 的 38 TRAIN rows（9 success / 29 failure）形成 14 candidates、6 programs、14×38=532 outcomes，
    其中 56 active / 476 frozen-RAW replay。Plus 首轮以 56 outer workers / 48 model slots 调度，source
    model 峰值并发 34；55 路有效，唯一 `4033…`/temperature-2 为明确 provider-capacity terminal。
64. 1356-event ledger 完整恢复 55 个有效结果后，只以 Pro exact retry 该 1 路并得到 valid failure；最终
    56/56 active valid、476 replay、online judge=0、validation/test=false。`72c5…` 以 organize-2 的
    1 recovery / 0 regression 排名第一，ranking=`2ec01860…db33`；但 56 个 active route 的 source
    evidence 全部包含当前 item，strict leave-item-out=0，所以它仍只是 in-sample signal。未运行 promotion
    gate，未产生 incumbent，也不授权 freeze/controls/family-out/HippoRAG/sealed。
65. 看过上述 ranking 后，完成一次明确标记 selection leakage 的 organize-2 targeted item-out
    refit/falsification。graph 与 contract 均排除 organize-2，仅使用 organize-5/-6，support=2、max
    artifacts=6；candidate `a34f…` 的唯一 active run 以 Plus 在 14 actions / 182.488 秒内得到 offline-valid
    false→true，另 37 outcomes 为冻结 RAW replay。post-run compile exact audit hash=`0cb50293…deb74`，
    final report=`849bfbc6…b58ec`，raw Codex/verifier worker artifacts 已持久化。由于 item/workflow 是
    post-selection，`unbiased_crossfit=false`、single-fold incumbent unauthorized；该结果只说明阳性在移除
    同题 graph/contract evidence 后存活，不是无偏 transfer。下一步若继续，必须事前冻结 multi-fold /
    multi-item TRAIN cross-fit；仍不增加 gate，也不进入 downstream。
66. 为一次性判定这个候选而非围绕成功题继续补 gate，已固定同一 workflow 并以两路 Plus 最大并发补齐
    organize-5/-6 item-out；每路只用另外两题形成 graph/contract，各 1 active + 37 RAW replay，均无 retry。
    两路分别在 17/100 与 12/100 actions 内得到 offline-valid false→false，reports=`3201ca6c…6fd8` /
    `a59d6f4e…70da`。完整 targeted family 结果因此为 3/3 active valid、111 replay、1/3 recovery、0
    regression、online judge=0、validation/test=false。整个 audit 仍有 post-selection bias，但 1/3 已足以
    否定 family-wide transferable signal；该候选在此停止，不 freeze、不进 controls/family-out/HippoRAG/
    sealed。下一步改 candidate search objective，使 out-of-item utility 在 TRAIN candidate formation 时直接
    参与选择，而不是继续验证这个失败候选。
67. 对上述三折原始 trace 的非评分归因显示：legacy organize-5 的主要结构错误是 destination 嵌套在 source，
    organize-6 则只剩一个 semantic misclassification。随后在任何新 actual 前，以提交 `baa3230a` 和
    manifest=`da85625c…5483` 预注册三个 generic refined invariants 及三折 exact hashes；manifest 明确记录
    prior outcomes 已用于设计、`globally_unbiased_crossfit=false`。三路 Plus actual 同时启动，均 valid、
    offline、无 retry/online judge，共 111 RAW replay：organize-2/5/6 分别为 false→false、false→false、
    false→true，19/8/19 actions、340,315/138,160/366,903 tokens，reports=`7a9eff32…64e7` /
    `4aa5f94a…6873` / `5f64ace5…4928`。org2 最终仍有 10 个 wrong-subject 文件，org5 只错 1 个，org6
    6/6 PASS。阳性 identity 改变但 aggregate 仍为 1/3 recovery、0 regression，故 trace-refined prompt
    分支同样停止；下一步只允许新的 executable typed-assignment candidate class 与 prospective TRAIN
    cross-fit objective，不再补 prompt/gate，也不进入任何 downstream。
68. typed-assignment candidate 已在 outcome 前固定 runtime class、三折 work-unit hashes、3/3 valid 与至少
    2/3 recovery 停止条件；执行器经安全审查后改成 agent 退出才注入不可预测 path 的 exact-byte tool，并把
    prepare/reconciliation 完整 host-safe receipt 持久化。Plus canary 401 且无 model response；在 task/evaluator
    call=0 时仅修正 provider-unavailability route，最终提交 `0eba5b7c` 后 Pro 固定完成三路并发 actual。
    3/3 valid + 111 RAW replay，organize-2/5/6 为 false→false、false→false、false→true，1/3 < 2/3；
    每路 103/103 evidence/assignment/reopen/hash match。verifier trace 将失败精确定位为 org2 1 个与 org5
    32 个 semantic wrong-subject；org5 agent 临时宽关键词规则造成大量 LLM false positive，org6 6/6 PASS。
    因此 candidate formation/reconciliation 已闭合而 semantic operator 未稳定，该 representation 按计划停止。
69. frozen MiniLM semantic-assignment operator 的两次 period-disjoint public-OA acquisition 都在 prediction 前
    因类别供给不足 fail closed；operator call/outcome=0。路线按预注册停止，不以第三次下载、partial PDF、
    synthetic 文献或 sealed organize 替代 task effect。
70. SC-100 v1 field compiler 的 native verifier 给出表面 2 gain/0 harm，但 Poppler/PyMuPDF 审计发现
    item4 地址带电话尾号前缀；语义改判仅 1 gain。candidate 停止，没有在同三题修 regex 或 rescore。
71. 独立 SC-100 oracle 先通过 2 canary/5 mutant；唯一 24-case role-v2 shadow 随后为 required 0/12、
    true-negative 5/6、coverage 0/6，18 个 task-valid case 全在 parser 阶段 reject。该 hand-authored grammar
    class 被否决，0 model/Ruoli/online/official-test call。
72. commit `c66e3a73` 冻结 financial semantic structured extractor 与 runtime assets；formation financial-1/-3/-5
    为 candidate 3/3、历史 RAW 0/3，但 report 明确为 in-sample、`cross_fit=false`。commit `2a8ade07`
    随后在 fresh outcome 前固定 split、financial-4 treatment 与唯一一次 paired measurement。此前要求先有
    deterministic TRAIN cross-fit 的 spend-control 没有满足；这不污染后来的事前冻结 pair，但限制其 claim。
73. fresh batch 以 10 路 Plus 模型调用最大并发启动，offline only。父 scheduler 在 agent 后丢失，三次冻结的
    recovery/continuation 只执行未完成的 post-agent stage；model/operator/verifier 均无 replay。active
    financial-4 为 RAW=false→candidate=true、+1 且 evidence valid；完整 cohort 因 temperature-4 inactive
    frozen-audit invalid 仍非 valid，1/9→projected 2/9 仅描述。无 promotion/incumbent/HippoRAG/sealed，
    final report=`e6bc247e…d389`。
74. SEC 13F period-out 以 16 路 RAW/candidate 最大并发执行，固定 Plus、offline-only、0 retry。15 路
    observation valid，1 路 RAW 因 residual process action receipt fail closed；recover-only 为 0 call/
    0 replay。7 个完整 pair 为 2/7 对 1/7、+14.29pp、0 observed regression，但唯一 gain 在 fold 2，
    八对缺失值界 `[0,+12.5pp]`。复用 parent operator 的 stock ontology 与公开合同确定性不一致；本轮
    不 promotion，当前 8 题成为 consumed diagnosis，4 sealed 未访问。partial report=`d75d8d4f…ba7`。

这些负结果、首个 single-item prospective 阳性与 period-out 局部方向信号共同把下一风险定位到 candidate
semantic contract 和完整 replicated utility，而不是继续扩展 gate、archive 或 HLE source span。

## 十、建议的实验协议与 claim ladder

### 10.1 主 benchmark

当前最合适的主战场仍是 SkillLearnBench：

- instance holdout：38 train / 16 validation / 32 sealed test；
- family out：48 train / 11 validation / 27 sealed test；
- HLE：只作冻结的 external transfer/stress slice。

如果选择 86-item infrastructure subset，必须在任何模型调用前冻结新 manifests、重新计算
split counts，并说明 9 项排除只由 verifier/toolchain availability 决定，而非 task outcome。

### 10.2 必要 controls

最低矩阵应包含：

```text
raw_no_skill
static_generic_v2              # fixed, no learning
v2_no_recursive_repair         # same evolution budget, repair disabled
promoted_v2                    # frozen incumbent from recursive loop
skilllearn_b1_sonnet           # upstream static reference
human_authored                 # upper reference, not budget matched
```

raw、static、no-recursive 与 promoted arms 必须共享 model、provider policy、step budget、
runtime、verifier isolation 和 invalid-row policy。外部参考与 human upper reference 不应被
伪装成 budget-matched primary controls。

### 10.3 两个不同的因果问题

1. **same-item paired off/on**：相同 item、runtime 和 evaluator 下，candidate treatment
   是否造成 gain/harm；
2. **prospective transfer**：在未见 instance/family 上，router 是否在看不到 outcome 时
   正确激活，并保持净收益。

第 1 个回答局部因果 effect；第 2 个回答假设是否可复用。只做第 1 个不能证明 continual
learning，只做第 2 个而没有 matched controls 又无法归因。

### 10.4 预注册指标

- task success / executable reward；
- gain、harm、net gain 与 exact McNemar；
- effect LCB 与 item-clustered interval；
- prospective activation rate、evidence-valid precision 与 abstention；
- behavior-changing repair count；
- cost ratio、token、latency 与 model calls；
- invalid/error rate、provider/budget/runtime mismatch；
- archive retention、duplicate rate、forgetting 与 cross-family transfer；
- 多比较 Holm correction 与预注册 early stopping。

“hypothesis proposal precision”不能只按 schema pass 定义；更可操作的定义是：候选先通过
train-only static contract，再在 prospective matched validation 中产生正净效应且不超过
harm/cost gate。train-side selection precision 是 failed activation / 全部 labeled train
activation；held-out causal activation precision 的分母则是 evidence-valid 实际激活，正例
来自独立 paired gain，而不是模型自评。两者不得混写。

### 10.5 claim ladder

| 层级 | 可声明内容 | 当前状态 |
|---|---|---|
| L0 wiring | schema、repair、off/on、guard、archive transition 的机械链路已连接 | 达到：typed operator feasibility 9/9，production selection integration v2 13/13 + 12/12 tamper + exact replay；typed-portable formal integration 又以一次 run + exact replay、3 项真实 Docker canary、production loader/cleanup 闭合 pre-agent 只读 sidecar。它不覆盖 write/render/move task effect |
| L1 mechanism live | 真实外部任务中 proposal/repair/treatment/gate 全链路完成 | 达到：v3.20 完成 60/60 valid receipts；execution-contract/organize 路径闭合实际 action；独立 financial path 又在 agent 后执行 bounded typed operator，并在 fresh active pair 留下 operator/verifier evidence |
| L2 validation learning | clean held-out validation 上有可晋级净收益 | **在 exact SEC-13F workstream 达到**：contract-derived candidate 的 Replication C development 为 8/8 valid gains、四 fold 各 +2、0 harm，并产生正式 promotion；旧 financial-4 与 parent period-out incomplete 只保留为历史诊断，不与本结果拼接 |
| L3 prospective generalization | frozen incumbent 在 unseen instance/family 上保持收益 | **在多个窄 scope 达到**：SEC-13F frozen candidate 在 4 个预提交同域 sealed item 上 4/4 gain；只在 MuSiQue F1 形成的 P 在 12-item Hotpot cohort 为 21/24，相对 RAW +10、相对 official +1；随后 exact frozen actions 在 48-item fresh 2Wiki A_hold 上得到 incumbent/P/official/RAW=111/110/99/56（总 support=120），incumbent−official=+12、16 gain/4 harm、descriptive exact p=1549/262144。2Wiki family 历史上并非从未见过，所以最强表述是 fresh-item no-new-search transfer；全部 QA claim 都只覆盖 item-local retrieval，不覆盖 answer generation、full-corpus benchmark 或 broad Assumption-Agent transfer |
| L4 self-evolution | 多代 retained improvement，且 recursion ablation 有因果贡献 | **在窄 Hotpot retrieval-only scope 达到**：fresh M_L4 上 P+Q=43/48、Q=40/48、P=36/48；P+Q−Q=+3、P+Q−P=+7，只有 1 个 P support 被遗忘。P/Q direct retrieval 与固定 RRF ablation 在同一 24-item cohort 完成 96/96 terminal 后才评分。Q 的 F_Q cross-fit 不稳定，故不能外推为广义自我演化或端到端 QA |
| L5 evaluator co-evolution | anchor-guided evaluator replacement 与 selective erasure改善搜索 | **未达到**。Hotpot、2Wiki、QASC、AVeriTeC、HoVer、ERASER、MAVEN-ERE、EntailmentBank 与 BRIGHT 均给出有效 non-promotion 或 challenger degradation；AVeriTeC E1 虽在 19/36 项改变 top-5，但对 E0/RAW 为 36/36 utility tie，故 M_search 合规保持未执行。BRIGHT A_hold 的 E1−P6 为 `−1,725,169,818` integer nDCG、未晋升，fresh RESERVE counterfactual 又为 `−222,856,829`。HybridQA 首次证明 promotion/authorization/M consumption wiring 可执行，但 promoted E2 在现实域 A_hold 相对 HippoRAG primary 未通过，untouched M 的预注册 L5 也为 false。故现有证据支持“evaluator 能拒绝无收益候选，且偶尔能晋升”，仍不支持“晋升后改善后续 untouched search” |

## 十一、什么才算“真正自我提出并递归验证假设”

以下条件需要同时满足：

1. 候选不是人工预写的唯一答案，而是系统只从 train evidence 提出；
2. candidate selection 同时利用失败与成功对照，不能只奖励 failure support；
3. 假设被编译为当前 backend 能强制或明确审计的程序；
4. activation 在 outcome 前决定，并实际改变 execution treatment；
5. 同一 item/runtime/evaluator 有 policy-off/on paired counterfactual；
6. promotion gate 完全由冻结 protocol/evaluator 所有，candidate 不能放宽阈值；
7. promotion 不读取 sealed test；
8. 通过的程序进入 archive，并在未来未见题上被 prospective router 调用；
9. 失败程序能降级、停用或归档；evaluator epoch 改变时只使旧依赖证据失效；
10. recursive repair 的收益用共享 root/evidence 的 no-recursive arm 做因果消融；
11. 最终提升能归因到该程序，而不是额外预算、fallback、provider 或重采样；
12. 至少一次真实 promotion 改变下一代 incumbent，并在后续任务上保留净收益。

在满足这些条件前，“自我提出并递归验证”仍应被称为研究机制或 harness，而不是已经
证实的 self-evolving capability。

## 十二、最终结论

旧 Assumption Agent 的主要问题不是“没有足够多假设”，而是假设没有稳定编译成可
执行、可路由、可反事实验证并可跨题保留的 policy。legacy 优化的是“怎样更复杂地
回答 HLE”，而不是“哪些假设值得在未来任务中保留，以及它们是否因果性地改善行为”。

`reconstruction_v2` 已经完成了重要转向：它把三层 hypothesis、paired evaluation、split
guard、archive 和 evaluator epoch 做成了清晰的小型系统。这使研究问题第一次真正可
证伪，也比继续给 legacy HLE monolith 加规则更有价值。

本次已经关闭四个会让后续结果先天不可解释的 P0：candidate 不能控制 promotion
及格线；外部 backend 不再把 prompt/verifier/fallback 声明伪装成 typed/observed 事实；
86-item offline-ready manifests 通过 readiness，且 v3.4 v4 prewarm 为 86/86；verifier receipt 现在绑定 proxy
实际执行的 runtime profile/command，完整 CTRF 的任务失败不再被误标成 infrastructure
failure。新 GPT Pro route 证明 Ruoli 模型调用在该 v3.3 batch 中可用，但不能外推持续稳定；离线 evaluator 从未需要
替换为 online evaluator。

v3.3 在本次 batch 中排除了原先的 64 MiB 直接阻塞：38/38 train 的最大流量为 40.6 MB，
`video-object-counting-1` 从 71.1 MB invalid 降为 19.69 MB valid；没有 provider 或 hard-cap
error。代价不是修改 evaluator、子集或 promotion gate，而是把低 reasoning / verbosity
与更早的本地 history compaction 作为 protocol-owned agent treatment。canary/full 的
1.47/19.69 MB 差异表明重复稳定性仍未建立，不能把单次 batch 外推成稳定完成。

最近的两个 execution blocker 已在零模型层定位并修复。`web_search` 不是 provider/model
注入，而是 Codex 0.144.1 对旧 boolean config 的兼容性 no-op；canonical 顶层 disabled 已由
真实 wire 阴性/阳性对照确认。nominal `max_steps` 也不再被称为 semantic turn，而是冻结为
可流式观察的 `codex_action_start_v1`，由容器内 supervisor 终止 PGID、清理 dedicated-container
基线后的全部 live task 并生成 receipt。
这不是通过重试或放松 auditor 把 v3.3 invalid 洗成 performance evidence；v3.3 的 37 条
valid observation 仍全部不可复用，sealed 仍未触碰。

因此当前距离目标的第一段不再是修 gate 或证明单次 API 连通性。v3.4 clean lock、新 runtime
cache、86/86 cache-only prewarm 和 max2 v5 action-budget canary 均已通过；v5 同时证明 actual
wire 无 web、budget receipt valid、全部 agent task 已退出且本地 verifier 在 agent 后执行。
fresh development 也已真实启动，但冻结四并发在 17 条有效离线结果之后触发四个 429，随后
17 个 slot 被 circuit 本地跳过。API credential 和 bounded inference 可用，持续四并发容量不可用；
online evaluator 无法修复该问题。

v3.5 的第三个 fresh root 已在同一 invocation 内取得 38/38 all-valid train，并首次完成 proposal、
真实 repair、双臂 paired validation/promotion decision、两代停止与四份 report/archive。clean g1
给出的最强信号不是 promotion，而是 selection 反例：recursive 最大-support repair 为 0 gain/2 harm，
no-recursive 保守 root 为 1 gain/0 harm但 lower bound 尚为负。g2 随后被一次 Ruoli 503 与 circuit
skip 污染；旧 lifecycle 又把 9 invalid pair 当成普通 non-promotion 并写出 valid score。因此这份
run 只能保留为 L1 机械闭环与 contrastive-learning 动机，不能作为 clean full-development claim，
也不能 freeze 空 incumbent 或进入 sealed。identity/response/invalid-evidence 修复都不是放宽评分 gate；
archive 冲突硬拒绝、promotion contract 和 evaluator 均未改变。v3.6 live 已从零完成 38/38
contrastive train 并进入真实 paired validation，但串行执行只完成 2/16 pairs 后主动终止，因而
没有完整 development claim。v3.7 的固定六路被首批 6/6 429 否决，v3.8 的固定两路又在
16 valid 后被 2 个同时 503 否决。v3.9 随后用 6 个题级 pipeline 配 1 个共享在线 agent slot，
在 2026-07-13 首次完成 clean full development：38/38 valid train、56/56 actual trials、0 provider/
infra/mismatch failure，并写出四份完整 report/archive。这个结果不是 promotion：recursive 两代
都只激活 1/16 validation，candidate/raw 均为 3/16、0 gain/0 harm；no-recursive roots 均在
train-only static audit 被拒绝，两臂 `incumbent_id` 都为空。

因此距离目标最近的缺口不再是 transport、offline evaluator 或更多 promotion gate。v3.10 已证明
exact-three/coverage-first 能把 activation 从 1/16 提高到 2/16，却不能产生 gain；第二代又证明用文本
要求模型满足 host 事后计算的 pairwise signature 不是可靠 response contract。v3.11 随后证明新的
imperative action/lowering 确实改变 agent trace、PDF 和 action cost，但唯一激活仍为 0→0；它同时因
repair model 返回 root-shaped batch envelope 而没有完成 recursive quality 检验。调用链确认 batch
contract 实际未进入 repair，因此 v3.12 只新增 protocol-bound singular response override，保留
exact-three root、8,000-token budget、train-only coverage selection 和 audit-only signature policy，
也没有放宽 evaluator/promotion 或修改 split/fuse/retry/sealed。

v3.12 的 clean 结果排除了“先等 repair 触发”作为下一步：所有 static node 直接通过，repair=0，
而 exact-three 已连续两代产生三个互补、高精度、零 success-FP 的单-family roots。真正的信息损失发生
在 selector 把其中两个丢弃。v3.13 已把这一点一次性版本化为 train-only complementary program-set，
且 live 证实 program-set routing、per-item match、nonactivation alias 和 G1 cross-arm replay 都正确。
正式轮 76/76 valid，但三套 bundle 的 6 个 policy-on 全失败；两臂四代均 activation 2/16、0 gain/0 harm、
无 incumbent。故“精确命中 TRAIN failure”只能证明 trigger precision，不能证明 action utility。

v3.14 的机制修复不是再加 gate：它共享 recursive/no-recursive 与多代的 valid raw baseline cohort，
并在 precision、capped deficit、success-FP、overlap 相同的 TRAIN-only subsets 中把实际 family count 与
failure support 放到 bundle size 前。live 已证明后者按设计选择 7/7、3-family set，activation 也从 2/16
增至 3/16；但 7 个 policy-on 仍全失败、0 gain/0 harm。因此 selector 迭代到此结束。valid baseline
evidence 的 31 次 replay 也证明共享路径有效；一条 hard-cap invalid 因不入 cache 又被另一 arm 执行，
说明未来若还做 cross-arm claim，应共享 terminal invalid memo，而不是重新采样。该修复属于 evidence
identity，不改变任何 promotion threshold。

下一 workstream 必须学习**可执行 action delta**，而不是更宽 trigger 或更多 gate。当前三个 G1 action
program 虽然语法清楚，却分别只说“使用品牌色”“收集权威离线漏洞记录”“只填写必要表单字段”；它们
没有给出任务缺失的实际 HEX、可访问数据源/路径或新操作步骤。TRAIN failure precision 只能证明这些
instruction 出现在失败样本中，不能证明 directive 提供了 baseline agent 原本不知道的知识。v3.15
提交 `696a2954` 已把 instruction-restatement 与 material executable knowledge 的区别写入 request-local
prompt，并用受限 TRAIN environment/policy-off trace profile 提供具体但非 oracle 的设计上下文；audit
仍严格为非评分、不能改变 response lifecycle 或 promotion。terminal-invalid memo 又关闭了 v3.14 的
跨臂重采样归因缺口，paired report/freeze 则绑定共同 action-profile provenance。正式 live root 已通过
clean lock、86/86 cache-only prewarm 与 smoke，并以 57/57 valid actual trials、8/8 model calls、最大
在线 agent 并发 1、0 provider/infra/budget/network/pair error 完成。38 条 TRAIN 中有 6 success / 32
residual；16 条共享 baseline 支撑两臂/两代共 32 次零执行 replay，实际只运行 3 个 candidate-on。
recursive G1/G2 均为 activation 1/16、candidate/raw 4/16、0 gain/0 harm；no-recursive G1 static reject、
G2 仍为 1/16 与 0 gain/0 harm。两臂 claim-eligible 只意味着 clean negative evidence 成立；二者都
`consecutive_non_promotion_limit`、`incumbent_id=null`、sealed/test=false，完全没有 downstream。

action audit 解释了为什么机制更干净却没有更强：13 个 candidates（9 roots + 4 repairs）中 7 个有
material delta、6 个有 restatement risk，但 7 个 material 全部只是 `exact_constant_or_mapping`；concrete
local tool、artifact manipulation 与 environment primitive 均为 0。九个 roots 又全部坍缩为
`anthropic-poster`，从 v3.14 G1 的 3-family/7-support 退到 single-family/2-support。v3.16/v3.17 因而先把
structural family stratification 与 artifact grounding 放进 proposal-only screen，而不是再跑完整 development。
v3.17 确实把三项 proposal 分到不同 family，取得 support 2/2/3、3/3 concrete local tool、2/3 artifact
manipulation、0 restatement/self-block；但第三项仍绑定 `file`/`python` 两个 failed-command primitives，
因此 8/9 pass 仍是整体 fail。0 backend/evaluator/benchmark trial 是正确的 spend-control 结果。

这也终止了“继续写更强 prompt”的路线：failed command 中出现通用 executable 并不证明它在因果上应被
禁用，而只把 count/hash 给模型又无法要求其避开未知具体值。causal action-span taxonomy 与 closed typed
operator/artifact graph 已实现，并在唯一一次正式离线 decision 中 9/9 PASS、精确复验 PASS。production
typed selection 也已经独立冻结并在 v3.18r1 live 接通；旧文中“尚未接入 proposer/evolution”的判断到此失效。
新证据同时给出更严格边界：opaque selection/provenance 成立；locator-free portable role 与 production
pre-agent read-only evidence sidecar 也已由 formal integration 和真实 Docker canary 验证。但 sidecar 只读取
输入 evidence profile/inventory，`task_effect_claimed=false`、`recipe_operator_effect_claimed=false`；当前
materializer 的 write/render/move 等步骤仍由 prompt skill/agent plan 执行，不能把 installation、route 或
sidecar receipt 写成完整 typed operator execution。

v3.18r1 的二元报告虽为 0 gain/0 harm，却不能作 clean efficacy negative：当时 organize 输入缺 100 PDF，
stock 又把 8/10→4/10→3/10 的真实退化压成三个 task-success=0。v3.19 随后恢复并冻结了 task
input/dependency closure，typed-portable formal integration 又以唯一预注册 decision 与 exact replay 授权
fresh v3.20 development。

该 fresh development 现已完成，而且是可审计的 clean negative：38/38 TRAIN、16/16 shared
validation baseline、6/6 policy-on 均 valid；两代均为 3/16 activation、5/16 对 5/16、0 gain/
0 harm，两臂均 `incumbent_id=null`。这不是行为不变：G1/G2 的 recipe、treatment、
action count 和 verifier 子测试都不同；stock 从 6/10 升至 8/10，但 organize 从 5/6 降至
3/6 或 4/6，temperature 从 6/7 降至 5/7。这些都没有翻转二元 task success。

launch-prompt integration 与随后严格限额的 6-trial diagnostic 已把旧断点进一步缩小：问题不再是 profile
缺席于模型输入，而是 delivered profile 没有表达足够 task-local、可执行且可自校验的决定性不变量；
material task action 仍主要由 agent 自由规划。6/6 runtime receipt 虽然闭合，G1/G2 仍各 0/3，且相对
同代历史 treatment 与冻结 RAW 都没有 utility signal。因为这些 validation item 在预注册前已经消费，
这个结果只能作 design evidence，不能作 clean causal null。

closed execution-contract 搜索现已真正跑过 14×38 TRAIN grid：56 active + 476 RAW replay 全部有效，首位
`72c5…` 出现 1 recovery / 0 regression。但 source audit 同时证明 56 个 active 都是 in-sample，strict
item-out=0。随后仅用 organize-5/-6 refit 的 organize-2 targeted item-out 又得到一次离线 valid
false→true，说明该阳性在移除同题 graph/contract evidence 后存活；其 Codex/verifier worker artifacts 已持久化。
然而这道题和 workflow 是看过 ranking 后才选定，`unbiased_crossfit=false`，所以仍不能把 single-fold
survival 写成 transfer、promotion 或 incumbent。现在同一 workflow 的另外两个 item-out fold 也已最大并发
补齐，organize-5/-6 均为 offline-valid false→false；完整 family audit 是 1/3 recovery、0 regression。它仍
有 post-selection bias，却已经足够反驳 family-wide transferable signal，因此该候选已终止。

随后完成的 trace-refined 候选没有改变这一结论。它在 actual 前冻结三折 exact hashes，并以三个 generic
invariant 修复 destination/source nesting、无证据 fallback 和 final-layout reopen；三路 Plus 同时执行且全部
offline-valid。但结果只是把唯一阳性从 organize-2 换到 organize-6：organize-2/5/6 分别 false→false、
false→false、false→true，aggregate 仍为 1/3 recovery、0 regression。org2 的 source/layout 已闭合却有
10 个 wrong-subject 文件，org5 只差一个 semantic assignment，说明当前剩余 blocker 不再是多一个 prompt
条款或评分 gate，而是 evidence 到 assignment 的可执行、可审计表示。由于 refined design 本身使用了先前
outcome，`globally_unbiased_crossfit=false`；它同样不是 incumbent 或 L2 evidence。

所以当前停止条件比“有一个阳性就换 fresh split”更严格：现有 primary validation 不能继续用于无偏 claim，
也不能让 post-selected candidate 直接消耗新 holdout，更不应继续围绕 organize prompt 增加规则或 gate。
上一版诊断要求的 executable typed-assignment candidate 现已实现并跑完：在 agent 前从公开任务输入提取
PDF/DOCX/PPTX 内容证据，agent 只写完整 typed plan，agent 退出后 harness 才注入经精确 SHA 回读的全新
runtime tool，执行一对一 move 并 reopen/reconcile，最后才物化本地离线 verifier。prepare 与
reconciliation 的完整 host-safe receipt 都进入 actual report；agent 期间可篡改的已知 tool path 不会被执行。
这修复的是 candidate representation 和执行可归因性，不是增加评分 gate。

三折 actual 在最终预注册提交 `0eba5b7c` 后一次性以 3 路并发完成，另有 111 个冻结 RAW outcome 直接
离线 replay。Plus 事前 canary 返回 401 且 0 model response；在任何 task/evaluator call 之前，仅把 provider
availability 路由修正并重新提交预注册，随后 Pro canary 通过并固定用于完整 batch。候选、fold、2/3 recovery
阈值和离线 evaluator 都没有根据结果改变，也没有中途切 provider、retry valid failure 或 resample。三路均
`evaluation_valid=true`，每路都对 103/103 文件取得内容证据、103/103 typed assignment、source empty、
reopen count 103 与 content-hash reconciliation；但 organize-2/-5/-6 分别为 false→false、false→false、
false→true，只得到 1/3 recovery，低于事前固定的 2/3，`candidate_search_success=false`。

失败形态把断点进一步收窄到 semantic evidence→label operator，而不是文件发现、抽取、move、布局、receipt
或 verifier。organize-2 只错 1 个文件（`1909.05563v1.pdf` 被放入 trapped-ion，正确为 DNA）；
organize-5 错 32 个文件，trace 显示 agent 临时生成了带 `model`、`attention`、`prompt` 等宽触发词的
keyword classifier，造成大量 LLM false positives，并把多篇 music-history 文档误归 DNA；organize-6 为
6/6 verifier tests 全过。换言之，typed grammar 消除了不完整 plan 和不可审计副作用，却没有消除每个 trial
重新发明 semantic classifier 的方差；它没有增加跨 fold 的二元 recovery coverage。按预注册停止条件，
该 representation 到此终止，不进入 fresh development，不补新的 prompt gate，也不在同一三折上 retry。

若继续候选搜索，下一次必须是新的 operator class：把 trial 内 model-authored keyword heuristic 换成冻结、
可复验的本地 semantic assignment operator（例如离线 embedding/NLI prototype 或显式文献元数据 parser），
并把模型权重、category prototype、tie-break 与 abstention/default 规则在任何新 outcome 前绑定。当前 2/5/6
已经全部成为设计数据；再在它们上调参只能算 TRAIN fitting，不能重新声称 cross-fit generalization。若没有
可用的未消费 TRAIN-derived target 或独立 synthetic holdout，应诚实停止该 benchmark 路线，而不是提前读取
organize-1/-4 sealed 或 consumed organize-3。

该 NEXT 已按边界执行到第一次独立 acquisition。提交 `2f804812` 先冻结了本地
`all-MiniLM-L6-v2` runtime、115 条 consumed-TRAIN PDF、四个 one-vs-rest logistic target operator、
`music_history` 全负分 default、参数字节与全部依赖；2/5/6 上的 115/115 仅记为 resubstitution，
不作 prospective claim。随后预注册 6 strata × 10、60/60 exact-map 的 public-OA feasibility；正式评价
要求 Linux network namespace 只剩 loopback，Ruoli/online judge/Agent/HippoRAG/RAW 均为 0，且没有读取
organize-1/-4 sealed 或复用 organize-3。

第一次目标采集没有形成评价包：在每 stratum 最多 40 个已排序候选的冻结条件下，
`trapped_ion_and_qc` 不足 10 个可下载、可抽取且去重后的 PDF，程序 fail closed；现场只留下 18 个
partial PDF（content-set hash `51845c7c…8bb8`），没有 `pack.lock.json`、decision lock、report、operator
prediction 或 semantic outcome。该失败是 acquisition invalid，不是 classifier 的正/负结果。按照预注册，
不放宽这一个 holdout、不重试、不把 partial PDFs 当评价数据；结果 receipt 为
`semantic_assignment_public_oa_feasibility_result_v1.json`。下一次若继续，只能保持 candidate 完全不变，
事前冻结一个与 2020--2025 查询区间不相交的新 public-OA period/source 及更充足的纯 transport pool；
若第二次仍不能锁定 pack，则停止 public-OA feasibility 路线，而不是继续修 acquisition gate。

第二次 preregistration `74a387b3` 保持同一 candidate/threshold，改用从未查询的 2015--2019 period，
把纯 transport pool 事前扩到每 stratum 120、96 路并发；它仍在任何 prediction 前 fail closed：LLM
stratum 只有 2 个合格 partial PDF，仍无 pack lock/report/decision/outcome。结果 receipt
`semantic_assignment_public_oa_period_out_feasibility_result_v2.json` 因而将
`public_oa_feasibility_route_stopped=true` 固化。至此不再换 period/source、扩 pool 或补 acquisition gate；
local operator 本身仍是“未评价”，不是 PASS、FAIL 或 incumbent。

随后只读审计了全部本地可用 target，结论为 NONE：SkillLearnBench 的 organize family 只有上述 6 题；
约 34 篇 reference PDF 高度偏 LLM/self-evolution，无法组成 trapped-ion/black-hole/DNA/music-history/
LLM 五类平衡且独立的 gold；本地 Hugging Face cache 也没有相应分类数据。因而不把人为 synthetic 文献
分类当作 task-level causal gain 的替身，semantic operator 归档为
`transport-blocked / efficacy-unknown`。离目标最近的下一搜索对象改为非-organize TRAIN 上的有限
heterogeneous typed-action library：候选直接生成可执行 action 与 reopen/reconcile receipt，以 frozen RAW
和本地 verifier 的 item-out task-success 净恢复排序，而不是再修改 recipe prompt 或增加 acceptance gate。

第一个 executable vertical slice 是 SC-100 security-deposit field compiler。commit `edb88957` 在任何新
operator outcome 前冻结了 role/fact parser、固定 24 个 text field + 6 个 button action、原始 `/AP`
保留策略、三题 plan hash、PyMuPDF/Poppler、immutable Docker image、offline verifier 和一次 decision。
三路 generation 完成后才把 tests 交给另外三路 `--network none` verifier；native report
`5e717696…3747c` 给出 RAW `1/0/0` 到 candidate `1/1/1`、2 gain、0 harm。

但这不是有效阳性。按 PDF 交付要求渲染三份 page 2--4 后，item4 被告 street field 肉眼显示并由
PyMuPDF 精确重读为 `550184 and lives at 245 Mission St Apt 9`，而正确值应是
`245 Mission St Apt 9`。原因是未锚定的 address regex 从前一个 10 位电话内部最后 6 位开始匹配；
官方 test 只要求正确地址是 PDF text 的 substring，因此仍给 reward=1。这是 evaluator false positive，
也说明“operator 自己抽 facts、再用同一 facts reconcile”不能证明 source-to-action 正确。post-decision
result `1460d8f6…fc1d0` 将语义 reward 改判为 `1/0/1`，只有 1 gain、0 harm，candidate stopped；不在
同三题补 regex 或重跑。

这条 NEXT 已推进到“测量仪器冻结”，但尚未运行 candidate shadow。独立作者先生成 12 个 required positive、
6 个 task-valid coverage probe 和 6 个 true negative；随后只读审计发现 `corpus_spec.json` 声明按真实 NUL
排序、实际却按字面 `\\0` 排序。该矛盾在任何 successor code/outcome 前修正，24 个 case/prompt/gold
重新自校验为 corpus self-hash `5e16c371…b0660`。strict adapter 逐项验证 role/address/phone/date/amount、
Q9/Q10 边界、venue ZIP、C05 primary phone、C06 Unicode name，以及 reject reason；N02 明确冻结为
`attorney_fee_dispute` 优先于 `unsupported_claim_type`，negative receipt 禁止生成空白副本或 partial PDF。

独立 oracle 的冻结前对抗审计又发现两个会让错误 artifact 假通过的缺口：全页 `pdftotext` 没把显示文本
绑定到具体 widget，且未选 button 的 `/AP` 变化被 target whitelist 豁免。修正后，oracle 逐字段解析
`/AP/N` 并绑定 `/V`、以 Poppler bbox 和局部像素差验证可见位置，同时检查 `/AS`、button AP、DV、
field-tree topology、page resources、XFA/security、未授权字段与非 rasterization；reason/calculation 也改为
显式未返还语义和唯一 `$` currency value。正式 manifest `dfc7f3a1…0058` 在 immutable image、
`--network none`、只读 root、7 路并发下完成唯一 conformance decision：2/2 正 canary、5/5 mutant（含真实
address-prefix contamination、amount、Q9、Q10、swapped gold）全部符合预期，decision
`00ecee97…fdf36`、report `600467dc…b69f4`；shadow case/model/Ruoli/online judge call 全为 0。

该步现已在任何 shadow outcome 前完成冻结，而没有增加新 gate。candidate
`8d5fb7d9…7478b` 是 role-anchored closed grammar：按 plaintiff/defendant 局部证据绑定姓名、地址、primary
phone 和 email；event/demand/signature date、claim currency、Q9/Q10 与 venue basis/ZIP 分别使用 typed
parser；8 类 reject 按固定 precedence 返回自校验 receipt。写入仍限于 24 text + 6 button，先写临时 PDF、
复用 v1 reopen/reconcile，再以同文件系统 no-clobber hard link 原子发布；官方 blank、plan、inner receipt、
source/output hash 与临时文件清理都进入 outer receipt。冻结前独立自造输入审计发现并修复了 phone-prefix
street、`I am suing` name alias、defendant email 冒充 plaintiff email、否定 public/attorney 语句、无标签日期
范围、跨句 rent currency、双 plaintiff precedence、contract venue 和 unsigned-contract 等缺口；这些修正均
发生在读取 24-case outcome 之前。SC-100 相关 100+ 项离线 test 通过，page 2--4 Poppler render 也确认 invented
case 的文本和按钮在正确位置可见。

one-shot runner 同时冻结：24 路 generation 必须全部 join 后才加载 latent gold，随后最多 18 路 immutable
Docker oracle、`--network none`；negative 要 exact reason 且零写入。preregistration
`5cbe4815…3df9` 绑定 candidate、v1 writer、corpus `5e16c371…b0660`、已合格 oracle、host
PyMuPDF/Poppler 与 container image，并预留唯一 decision root。下一步只运行这一次已冻结 24-case shadow；
12/12 required 与 6/6 true-negative 是唯一硬判定，6 个 coverage probe 只原样报告 `coverage_starved`，
不得事后改成负例或据其修补重跑。即使通过，也仅是 synthetic feasibility，只能支持另行预注册新的 paired
development root，不能直接成为 incumbent。

这次唯一 decision 已在 frozen commit `40734632` 上消费，结论不是边界阳性而是明确的 candidate-class
失败。24 路 generation 全部先 join，但 18 个 task-valid case 全在 parser 阶段返回 reject，因而没有生成
任何 candidate PDF，也没有启动 Docker oracle：required positive 为 0/12，true negative exact reason +
zero-write 为 5/6，coverage probe 为 0/6、全部原样记为 `coverage_starved`。N01/N02/N03/N04/N06 的
scope control 正确；N05 应为 `conflicting_claim_amount`，却在 claim-type 阶段先给
`missing_or_ambiguous_required_fact`。report hash `39834131…c916`、decision
`e82ff2b2…0dc7`，result receipt `26843d5e…4b2f`；model/Ruoli/online/official-test call 均为 0。

只读 post-hoc stage attribution 进一步说明这不是 evaluator 或新 gate 的问题，而是 hand-authored grammar
本身没有覆盖任务语言：12 个 required 的首个失败分布为 pre-rejection 4、event date 4、list-style plaintiff
name 2、currency 1、phone 1；6 个 coverage 则为 pre-rejection 2、event date 2、signature 1、interleaved
address 1。`rather than / neither / ... or public entity` 被误当成 positive public-entity evidence；“leave
second-plaintiff fields blank”被误当成存在第二原告；`relevant period`、列表式 Name/Address、自然语言签名和
金额标点也落在 invented tests 之外。candidate 在 outcome 后未修改、未 rescore、未 retry，并按 manifest
永久停止。

因此不能把下一步写成“再补几个 regex”。这条证据否定的是从少量 invented examples 手工枚举 source
surface form 的 candidate class。若继续，只能换成由声明 TRAIN distribution 形成、输出严格 typed facts
并由独立 writer/reconcile 执行的 semantic structured extractor（或等价 capability interface），同时使用
全新的 untouched measurement；本 24-case corpus 只能作为 consumed diagnosis，不能再成为 selection 或
promotion evidence。这是 candidate search 的改变，不是增加 acceptance gate。

这条 NEXT 已由 frozen financial semantic candidate 执行。formation financial-1/-3/-5 为 3/3，而历史 RAW
为 0/3；但 formation report 明确是 in-sample、`cross_fit=false`。这意味着旧文要求“先通过 deterministic
TRAIN cross-fit 再消费 fresh”的 spend-control 并未满足，不能事后改写成已通过。它不使随后 pair 失效：
split、candidate、recipe、treatment、provider 与 offline verifier 都在读取 financial-4 outcome 前冻结；
但它解释了为什么本轮只能支持单题存在性，不能支持候选搜索程序的无偏泛化率。

fresh batch 的 9 个 RAW 和 1 个 candidate 共 10 次 Plus model call 同时调度。父 scheduler 在 agent 完成后
丢失，最终证据来自事前冻结、逐 stage 绑定的 post-agent recovery；没有重放模型、operator 或 verifier。
active financial-4 pair 为 RAW=false、candidate=true、差值 +1，且两份 observation、candidate typed
receipt 与离线 CTRF 有效。这是首个 preregistered treatment-associated unit-level gain，也是 L2 因果
存在性的第一条 prospective 证据。与此同时，RAW 与 candidate 是两个独立 agent trajectory，candidate
没有保存 operator 前输出快照；因此最窄的归因对象是整个冻结 treatment，不能把单样本差值扩写为
semantic stage 的普遍平均因果效应。

整个 physical batch 又不能称为 clean positive：runner 不是 pristine completion，temperature-4 的 inactive
RAW observation 保留 frozen audit invalid；1/9 RAW→2/9 projected candidate 的其余 8 个 candidate 都只是
exact RAW projection。finalizer 没有覆盖 invalid，也没有运行 promotion gate。official HippoRAG 因无同构
adapter 明确 N/A，residual sealed 未访问。

同一 frozen candidate 随后完成 SEC 13F period-out 多折复验的全部 16 个模型 claim。15 路 observation 有效，
1 路 RAW 因 residual process receipt fail closed；recover-only 证明 0 replay 且不能安全补齐。7 个完整 pair
为 candidate 2/7、RAW 1/7、平均 `+14.29pp`、0 observed regression，但唯一 candidate-only gain 只在
fold 2，完整八对缺失值界为 `[0,+12.5pp]`。因此这一轮把证据从“单题阳性”推进为“多个 untouched item
上的局部方向信号”，却没有推进到“多个 fold 稳定正收益”。它是 primary-invalid descriptive result，不能
和旧 single-item gain 拼成 promotion evidence。

离线归因又把下一 blocker 收窄为 candidate/evaluator semantic contract mismatch：parent operator 的 stock
ontology 只覆盖 period-out 25 个合法类中的 9 个，缺 16 个并含一个拼接伪类别；candidate 的首个失败全部落在
依赖该 filter 的 stock count 或 quarter-increase rank。下一步直接改变 operator 本体，在新 untouched
period/source 上复验；不再给 gate、prompt 或同一 8 题增加规则。当前 8 个 measurement item 因已运行且
CTRF traceback 泄漏部分首错 diff，只能作为 consumed diagnosis；4 个 sealed commitment 仍未访问。

因此现在仍不跑 incumbent freeze、完整 controls、family-out、HippoRAG 或 sealed test，更不谈 multi-clade
或 evaluator co-evolution。任何把 primary sealed item 改作 development 的方案都会消耗既有 sealed holdout，
必须另行显式重划并保留新的最终 sealed 集；v3.12 空 freeze/partial-control rows、v3.14 mixed-claim rows、
v3.16/v3.17 proposal-only artifacts、v3.18r1 mixed-validity rows、consumed diagnostics、in-sample
execution-contract grid、post-selected organize family 1/3 与本次 incomplete period-out 都不能拼成
performance evidence。距离目标的状态是：L1 wiring/delivery 已完成；L2 的 single-item causal existence 已证明，
且有一个独立 period-out candidate-only signal，但 replicated、完整、可晋级的稳定净收益仍缺；无 incumbent，
所以 L3 retention/family transfer 尚未开始；L4 recursive retained improvement 未证明，L5 evaluator
co-evolution 未开始。
最诚实的论文级表述是：

> **显式 HypothesisProgram 是一个有希望、可能更易归因的 self-evolution 搜索表示；
> v2 已证明协议所有权、离线 evaluator 和学习环 wiring 可运行，取得一次冻结、untouched、single-item
> prospective success，并在独立 period-out 多题复验中观察到一个局部 candidate-only gain、没有 observed
> regression；但该批次缺一条有效 RAW、完整差值下界为 0，且正差未跨 fold 复制，所以仍未证明冻结候选能
> 产生可晋级的稳定净收益，更未证明 retained self-evolution、Red Queen 式多谱系或 evaluator 共演化。**

### 12.1 2026-07-16 最终状态修订

上面的纵向叙述保留了每个已消费时间切片，但“仍无 incumbent、sealed 未开始”的旧结论已经被 Replication C
取代。当前最强证据链是：全新 contract-derived candidate 在 development 上 8/8 paired gain 并 promotion；
operator-only control 对 8/8 candidate output 做 exact SHA 复现；promotion 后冻结的 candidate 在 4 个 sealed
item 上再次 4/4 paired gain。正式 sealed 批次只有 8 个物理模型调用、峰值并发 8，RAW 0/4、candidate 4/4，
离线 verifier、0 online judge、0 retry/replay/resample/provider switch；所有 verifier 都在 agent/operator 完成且
容器断网后才看到 tests/gold。

这关闭了“typed action 是否只能改变 trace、不能产生稳定 task utility”的 exact-domain P0，也把 claim ladder
推进到 SEC-13F workstream 的 L2 和同域 unseen-instance L3。它没有证明 Assumption Agent 的一般 hypothesis
search、recursive repair 或 archive retention 自身产生了这些收益：本次获益对象是人工收敛并冻结的 typed
domain operator。family-out 与 official HippoRAG 因无同构 adapter 为 N/A，不能用 RAW 之外的伪代理补齐。

两个盲化事件又要求进一步收窄措辞。授权前的 digest-only private-pack stream 违反程序性 zero-byte 边界；
post-launch `ps` 又使监督通道看到 sealed instruction text。前者没有新增 digest 信息，后者发生在 candidate、
freeze、assignment 和唯一批次都不可变之后；两者均未显示 gold/outcome，也没有带来 adaptation 或补跑。
所以 4/4 结果仍是有效的固定 treatment paired observation，但不再是严格 blind holdout。后续不能重跑这 4 项
来“洗掉”事件；若要形成更宽论文主张，应在新的 period/source、重新建立的 sealed cohort 上由不读取 host
process argv 的独立执行器完成，而不是继续给当前候选补 gate。

因此，距离总目标仍缺三件不同层级的东西：第一，把同类 typed operator 的形成过程从人工诊断收敛提升为
TRAIN-only、可重复的 candidate search；第二，在真正同构的 family-out/adapter 上验证跨域保留，而不是把 N/A
写成成功；第三，证明 recursive/no-recursive 的 retained multi-generation 差异与 evaluator co-evolution。
当前 SEC-13F workstream 已完成，应停止在其 sealed item 上继续优化；下一研究单元必须是新的数据与新的
预注册 claim scope，而不是新的 gate。

### 12.2 2026-07-16 后续路线审计：先闭合 archive/epoch，再取得新域数据

三条路线的只读审计进一步收窄了可执行顺序。现有 typed grammar 能从 TRAIN trace 发现 artifact、按文件格式
生成通用 workflow，并让 production proposer 只选择 opaque `recipe_id`；但
`DERIVE_TASK_DELTA` 等节点仍是无参数占位，filter、group、aggregate、missing normalization、unit conversion
和 stable tie-break 不在可生成空间。因此它证明的是“闭合表示与选择”，还不是“自动形成具体语义 operator”。
已有 public development cohort 又已被后续运行消费，不能按已知 RAW outcome 再选 operator 后倒写成 fresh
measurement；14 个 residual-sealed SkillLearn item 继续保持不读。

官方 HippoRAG 2 源码镜像实际已在本地，绑定 commit `ef2f14c4`；此前 N/A 的准确含义是“没有同构且冻结的
benchmark adapter/runtime”，不是“没有源码”。当前没有 graph/index，官方 import/runtime 依赖也未在独立环境
冻结。`enterprise-information-search` 虽然在任务形态上最接近，但其 public TRAIN/validation 已消费，只余一个
residual-sealed item；而 upstream core 不负责层级 JSON 到 documents、复数答案 normalization 和 `answer.json`
写出。故后续若执行，只能称“official HippoRAG core + frozen custom adapter”，且一个 sealed item 不能支撑
family-out。正式 RAW / typed / HippoRAG 三臂必须另建 HippoRAG 原生 knowledge-QA claim scope、共享答案指标与
预算；它不是 SEC-13F transfer control，也不能用计算型文件任务上的伪代理替代。

在消费任何新数据前，最小研究单元改为零模型、全离线的 archive/epoch integration：G1 确定性 promotion 后，
从同一 checkpoint 在 G2 比较 `empty / P / Q / P+Q`，以 `Y(P+Q)-Y(Q)` 单独识别 retention；随后在固定 anchor
上执行 evaluator challenger、只失效旧 epoch 依赖分数、保留 independent objective，并对旧 incumbent 在新
epoch 重评分。它不是新的 performance gate，也不能把 synthetic integration 写成 L4/L5。完成该机械闭环后，
第一个新 efficacy 域采用固定历史 NOAA GSOD 气候 CSV 和闭合 relational DSL；station-out 的
12 TRAIN / 6 development / 6 residual-sealed acquisition 在任何 outcome 前形成，唯一一次 development 后无论
结果正负都停止 relational-DSL v1，不在同 cohort 补 primitive 或 gate。

这两个前置单元现已实际完成。零模型 integration 的 G1 为 4 gain / 0 harm 并产生 synthetic incumbent；
G2 从完全相同的 archive bytes 形成 `empty / P / Q / P+Q` 四臂，`Y(P+Q)-Y(Q)=4/4`，且 repair 没有参与。
两个不同的 synthetic evaluator 又在同一 8-row anchor 上实际计算出 4/8 与 8/8，challenger 随后晋级；分数
不是注入的 60/100 与 90/100。旧 epoch 的 1 条依赖 score 被失效、independent objective 保持有效；旧 node
在新 epoch 下 readiness 为 false，同一行为被显式 clone 到新 epoch node、重新完成 paired validation 和
promotion 后才变为 true，旧 node 同时标成 superseded。13/13 integration invariants 全部通过，archive
write/reload 完全同 hash，model/backend/online-evaluator call 均为 0；result hash 为
`51d71b3c...ef04`。这只把 retention/epoch skeleton 推进到 synthetic L0 integration，不改变 L4/L5 的
“未达到”结论，见 [`integration result`](../manifests/archive_epoch_retention_integration_result_v1.json) 与
[`safe report`](../artifacts/archive_epoch_retention_integration_v1/report.json)。

NOAA acquisition 也已一次性闭合：官方 2020 GSOD 的美国全年 station/file 交集为 2700，固定 hash 顺序检查
36 个 source，24 个满足事前 completeness，12 个只因覆盖不足被排除；划分固定为 12/6/6。stdlib
`csv+Decimal` 与 `sqlite3+integer` 两套 aggregation oracle 在 24/24 项一致，pack hash 为
`ab488057...3fc98`，公开 receipt hash 为 `90daaaf5...3e4a`。候选形成、评分、模型与 online judge call 均为 0；
sealed station identity、raw CSV、oracle output 和 private pack 只保存在 git-ignored acquisition root。该步
结束时的状态是“新数据已取得、typed operator 尚未形成”，因此没有提前消费 development，见
[`NOAA acquisition receipt`](../manifests/noaa_gsod_auto_typed_operator_v1_acquisition.json)。

随后只从 private pack 导出了 12 个 TRAIN item 的匿名、最小三列 CSV、公开 task contract 与 TRAIN oracle
consensus；development/sealed 的 identity、raw、gold 和 commitment mapping 均未进入该视图。导出后两套
oracle 又在匿名输入上完成 24 次一致性复核；TRAIN view hash 为 `8949062c...731a4`、preparation receipt hash
为 `30c38847...d93c9`，network/model/scoring/online-judge call 仍为 0。该视图只作为 candidate formation input，
保留在 git-ignored root，公开仓库只保存不含 TRAIN answer 的
[`preparation receipt`](../manifests/noaa_gsod_auto_typed_operator_v1_train_preparation.json)。

HippoRAG 的 N/A 也已缩窄。官方 commit `ef2f14c4...d10` 的 52 个 Python source 与独立 overlay runtime 的
安装树完全同 hash；在物理隔离的 network namespace 中，本地 tiny causal model 与本地 MiniLM 已实际走通
official core 的 import、initialize、index、retrieve、QA。层级 JSON→documents、multi-answer normalization
和 prediction-only `answer.json` adapter 同时通过；online model/evaluator call 为 0，receipt hash 为
`c2a6b540...a4e4`。但 official `openai==1.91.1` pin 无可满足版本，overlay 使用 1.91.0，vLLM 也未安装；因此
这只关闭“core+adapter 能否离线执行”的 runtime blocker，不是 benchmark、homologous baseline 或性能证据，
见 [`non-scoring qualification`](../manifests/official_hipporag_runtime_adapter_qualification_v1.json)。下一步必须
另取未接触的原生 knowledge-QA corpus，再让 RAW / Assumption treatment / official-core retrieval 共享同一
generator、预算与离线 evaluator；不能把本 synthetic probe 计作 family-out。

MuSiQue-Answerable v1.0 的原生 QA acquisition 随后形成，但尚未授权执行。第一次草案使用公开固定 salt，
即使公开 receipt 没有明文 item ID，任何人仍可从官方数据重建精确成员；该草案因此在模型调用和评分均为 0
时即作废，旧 private root 不再引用，也不能作为 sealed 证据。替代 acquisition 在读取 source row 前生成
32-byte 私有 secret，公开只提交 commitment，并以 HMAC 排序形成新的 12 TRAIN / 6 development / 6
residual-sealed。官方 TRAIN 共 19,938 行；预注册的 answerable、至少 5 段、至少 2 supporting、连续 paragraph
index、非空且双 normalizer 一致的 multi-alias 条件留下 4,589 行，再取固定 24 行。preregistration hash 为
`1ab83845...bce0`，acquisition hash 为 `86cd2881...d742`，private-pack commitment 为
`86d0e4d1...3924`。RAW 已准确命名为 canonical-order top-k=5 context baseline；三臂将共享逐 item candidate
corpus、generator、prompt、document count 与预算，answer/alias/support label 不进入 retriever 或 generator。
该 claim 只覆盖 official TRAIN 的预注册 multi-alias eligible subset，不等同于完整 MuSiQue 或其内部
family-out。acquisition 工具最初把成功的新 private root 留在随机 `mktemp` 路径而没有另存 path receipt；为避免
通过目录遍历暴露 dev/sealed/secret，后续 custodian 只对精确的 `train.jsonl` 候选计算 hash，并以公开 receipt
中的 TRAIN SHA 唯一匹配成功 root。这个 provenance 缺陷没有改变 split 或打开非 TRAIN 分区，但新 acquisition
工具必须把 secret-free root locator 单独持久化，不能再依赖事后定位。

在这条受限 custody 下，Assumption retriever 的真实 TRAIN-only formation 已完成一次，模型、network、online
evaluator 与 development/sealed access 都为 0。有限 grammar 的 84 个 type-valid program 先按完整 TRAIN
retrieval behavior 去重为 66 类，再以固定 support-recall@5、invalid、program length、program hash 排序；选中
program 在 12 个 TRAIN item 上命中 22/30 个 supporting paragraph（按 item 聚合为 11/15），invalid 为 0。
冻结 program 是 `BM25(title=4,text=1) → entity-token one-hop → expansion-weight=1 → top-5`，program hash
为 `a2404490...5119`，formation receipt hash 为 `03731387...1f7e`。四折 held-out 合计分别为 3/8、6/9、
3/6、6/7，且 fold winner 的 program/behavior 均不稳定；这项不稳定性作为形成结果保留，不再回到相同 TRAIN
fold 调 prompt、补关键词或补 gate。全 TRAIN winner 现可被冻结用于唯一一次 prospective development，但仍不是
performance、family-out 或 HippoRAG comparison 证据。见
[`MuSiQue preregistration`](../manifests/musique_official_core_comparison_v1_preregistration.json)、
[`acquisition receipt`](../manifests/musique_official_core_comparison_v1_acquisition.json)、
[`formation result`](../manifests/musique_typed_retriever_formation_result_v1.json) 与
[`frozen program`](../manifests/musique_typed_retriever_program_v1.json)。

official HippoRAG 对照也从 qualification placeholder 收敛为可执行但仍未评分的 retrieve-only adapter。它对
每个 item 建立独立 official-core index，只调用 `index/retrieve`，向下游只返回按 official score、paragraph
idx 稳定裁决的 5 个 idx；question/corpus/index 在隔离网络的私有 work root 中使用后删除。冻结 binding 同时
绑定 official commit 的 52-file source tree、overlay venv/dependency、local causal/embedding assets、8-file
transitive implementation closure 与 config，并如实保留 `openai==1.91.0` 相对 official `1.91.1` pin 的偏差。
本地 synthetic item 已实际走通 official index+retrieve，online model/evaluator 和 benchmark row access 都为
0；binding receipt 为 `522d3192...45d0`。这仍只是 homologous baseline infrastructure，不是 MuSiQue 分数，
见 [`retrieve-only binding`](../manifests/musique_official_hipporag_retrieve_only_binding_v1.json)。

与之配套的 gold-safe custody、launchable freeze 和三臂 formal runner 已在读取真实 development 前闭合。正式
custodian 先钉死 preregistration/acquisition/formation/program/qualification/adapter 六个已发布 trust root，
再把 6 题转成匿名 question+label-free corpus；answers/aliases/support 进入独立 0600 index，freeze 只把其 bytes
按事前 hash opaque-copy，直到 18 个 generation terminal 全部 join 后才允许解析。freeze 固定
`canonical first-5 / formed Assumption retriever / official HippoRAG retrieve-only` 的 6×3 grid、同一
serialization/prompt/model/top-5、18 路 generator concurrency、Plus canary→整批 Pro fallback 和
retry/replay/resample=0；invalid/transport terminal 按 ITT 直接记 answer EM/F1=0，不从 denominator 删除。
旧 acquisition draft 的 `maximum_context_tokens=8192` 没有 tokenizer 或可执行计数语义，因此在任何
development outcome 前显式改为完整 canonical request body 的 65,536 UTF-8 bytes、overflow fail-closed、
no truncation，不再声称 token budget。独立审计复现并关闭了可重算 worker plan、synthetic formal injection、
private-index 换金标、换 output root replay、实际使用点 TOCTOU、缺失 transitive binding 和失真实 failure
receipt 等绕过；相关 acquisition/formation/adapter/custody/freeze/runner 64 项合成与失效测试通过。当前状态仍
是 pre-run protocol：implementation 提交后，formal custodian 已唯一一次打开 6-item development pack，生成
gold-safe source view、独立 private index 与 public receipt；freeze 随后 opaque-copy index bytes 并形成唯一
authorization，但没有解析 gold、没有 task/scoring call，也没有 performance 或 family-out claim。custody、
private-index binding、freeze、worker 与 controller hashes 分别为 `d8ec4968...6e44`、`2b096194...9483`、
`852ea437...78a4`、`9b34ebd5...0f6a` 与 `4b043ed0...29f5`；`formal_execution` 和全局 consumption marker
均尚不存在。这些 hash-only evidence 随后已在任何 outcome 前公开提交。

唯一的 formal 启动随后在进入任何 benchmark 执行前关闭：公开绑定与 credential preflight 通过后，
official HippoRAG overlay venv 的离线 runtime-identity 子进程未启动成功。失败位置在 authorization
consumption、Plus/Pro canary、task/network call、gold parse、prediction terminal 和 scoring 之前；因此相应计数
均为 0，`formal_execution` 与 consumption marker 均没有形成。事后不读数据、不评分的同一探针
5/5 通过，但这不能确定首次启动失败的根因，也不能追溯性恢复 formal validity。根据事前
`retry/replay/resample=0` ，原 6-item cohort 已作为 `closed_efficacy_unknown` 封存：这不是负效果，
也不允许删除 failure receipt 或重启同一批。下一次 measurement 必须先在无数据 synthetic 环境中把
可执行子进程 qualification 与 formal-entry 的确定性 filesystem attestation 分离，再为全新预提交样本建立
新 root；不为这个失败增加 retry gate。见
[`prelaunch failure disposition`](../manifests/musique_formal_development_prelaunch_failure_v1.json)。

NOAA TRAIN-only typed formation 随后实际完成，而没有读取 development/sealed。冻结 grammar 枚举 864 个
七节点 relational program（上限 4096/8），先 type-check，再按 12 个匿名 TRAIN item 的完整输出行为 hash
去重为 103 类；固定 rank 为 invalid、harm、负 exact recovery、program length、program hash。winner 对
12/12 TRAIN exact，四折 anonymous-station-out 均为 held-out exact，并且四折选择同一 program。TRAIN 上存在
4 个行为等价 program，系统没有把 dedup 后的一行伪称为“行为唯一”：冻结 task contract 作为公开语义来源，
把 contract deviation 纳入事前固定的 program-length 项，最终仅 1 个 exact candidate 与 contract 一致；选中
program hash 为 `499aa46a...a3dd`，formation receipt 为 `cda7c209...138d`。这证明的是“在一个闭合、有限、
contract-derived relational DSL 内自动形成可执行 operator”，不是无约束自然语言语义发现，也仍不构成
development performance。见 [`formation result`](../manifests/noaa_gsod_typed_relational_formation_result_v1.json)。

NOAA development 随后在上述边界下完成唯一一次正式执行。monolithic acquisition pack 只由 source
custodian 打开；它只导出 6 份匿名、最小列 `STATION/DATE/PRCP` 输入和一份无 gold/oracle/sealed
mapping 的 private index。后续 freeze/controller 不接受 pack path，只接受该 source view，并形成精确的
6 item ×（RAW model、typed-agent model、local operator-only）=18 work-unit Cartesian grid。pre-run public
freeze 绑定 TRAIN preparation、formed program、source-view tree、worker/controller schema、provider identity、
公开 acquisition receipt 以及固定 23-file executable implementation set（含三个 CLI entrypoint）；runner
在任何 credential、canary 或 task call 前重新计算 live implementation hash。provider route 只允许
`https://ruoli.dev` 上 exact `gpt-5.4-mini`，先以不含任务内容
的 Plus canary 选择整批 Plus 或整批 Pro；task 中途不切换，12 个 model request 在调用前一次性提交 hash，
18 个 claim 先落盘再并发提交，retry/replay/resample 均为 0。全部 terminal join 后才运行两套本地 oracle，
完整 canonical request body 受同一 64 KiB UTF-8 byte budget 约束，而不再伪称未执行的 token budget。
公开报告只保留 arm 总数和逐 item ITT 配对 gain/harm/tie/incomplete 计数；invalid/transport failure 是已分配
terminal 的 incorrect，而不是从 paired denominator 删除。formal 入口也不再接受 transport/oracle 注入，
并分别报告 execution integrity、paired completeness 与 formal evidence validity。当前 source/freeze/runner
通过 36 项隔离与失效关闭测试。

正式 run 中 Plus canary 不可用，事前规则因此在读取任务前把完整 12-call model batch 固定到 Pro；两次 canary
加 12 次 task call 共 14 次 model call，online judge 为 0，12 次 offline oracle 一致，observed model
concurrency 为 12，18/18 terminal 全部 join，retry/replay/resample 仍为 0。三臂 contract-valid 与 exact 都是
6/6；agent-minus-RAW、operator-only-minus-RAW 和 agent-minus-operator-only 均为 0 gain / 0 harm / 6 tie，
paired net gain 全为 0。formal evidence validity 为真只表示执行和比较完整，不表示正效应：这个 saturated
relational task 上 RAW 已经全对，因此没有建立 L2 causal gain，也不授权 promotion 或 sealed。source receipt、
pre-run freeze、formal report 与 result hashes 分别为 `d40d6867...c179`、`de9ba630...43e1`、
`f5dbdc89...c7b3` 与 `50c37a2d...b29`；见 [`source receipt`](../manifests/noaa_gsod_development_source_receipt_v1.json)、
[`pre-run freeze`](../manifests/noaa_gsod_formal_development_pre_run_freeze_v2.json) 和
[`formal result`](../manifests/noaa_gsod_formal_development_result_v1.json)。这条零收益结果终止 NOAA 上继续调
prompt、补关键词或补 gate 的路线；下一次因果 measurement 必须换到有真实 retrieval bottleneck 的原生 QA。

第一次 direct-script source export 还暴露出一个纯启动器缺陷：`scripts/` 作为 `sys.path[0]` 时无法导入项目
package。该进程在 import 阶段、读取任何 private pack 之前即退出，所以没有形成一次被消费的 development
measurement；三个 entrypoint 随后统一显式绑定 project root，并纳入上述 implementation set 与 direct-launch
测试。

还必须限缩 NOAA 后续 sealed 的含义：该 acquisition 的 selection seed、算法与官方源均已公开，第三方可以
离线重建 split；它只有“执行流程没有把 sealed 内容交给 development controller”的程序性 custody，不具有
MuSiQue private-HMAC split 的不可重建盲化。因而本轮 6-item development 仍可作 prospective descriptive
measurement，但若要把 NOAA sealed 用作最终确认性证据，必须在启动前另建私密选择层和 commitment，不能直接
把现有公开-seed sealed 重命名为强盲 holdout。

本次审计还记录两项边界事件。一次 broad file search 意外展开官方 HippoRAG bundled public benchmark 的少量
question/answer/support 样例；这些样例不再属于 untouched acquisition。另一次只读检索看到 SkillLearn
residual-sealed item 的名称列表与少量 evaluator-epoch 元数据，但没有 instruction、gold、expected、answer、
outcome、verifier result 或 worker trace；涉事子任务不参与这些 item 的后续执行或决策。两项事件均未形成
候选或修改 split，但后续新数据必须使用新的 acquisition/split，而不能靠重命名现有 holdout 恢复盲化。

### 12.3 2026-07-16 MuSiQue generation-one 修订

fresh official-DEV 链已经把 MuSiQue 证据从“旧 6 题 prelaunch 失败、efficacy unknown”推进到一次有效的
retrieval-only generation-one measurement。96 项八 block 在形成/测量前预注册并一起取得；F1 只形成 P，M1
只比较 RAW、frozen P 与 official HippoRAG。one-shot 36-way 离线结果为 RAW `7/29`、P `14/29`、official
HippoRAG `14/29`，P−RAW 为 +7 support hit，7 gain / 1 harm / 4 tie；冻结 policy 因而给出 generation-one
promotion disposition，postflight filesystem binding 也保持闭合。全过程 0 Ruoli/external-network、0 study-level
answer-generator、0 online evaluator，official arm 内部保留冻结本地 LLM/OpenIE；旧 6 题没有重放。

最窄的正面结论是：**一个只在 fresh F1 形成并冻结的 typed retrieval program，在 untouched M1 上相对 canonical
RAW 取得预注册的正净 support utility。** 这比 synthetic archive wiring 和 infrastructure qualification 更强，
但还不是 retained recursive improvement。F1 的 cross-fit program 与完整 behavior 均不稳定；P 与 official
HippoRAG 只在总 support hit 上持平；runner 给出 promotion disposition但没有直接修改 archive；runtime v2 对
第三方包只形成 metadata-tree attestation boundary。M2 的 P/Q retention、A_form/A_hold evaluator co-evolution
以及同构 family-out 均未完成。因此论文级状态应写成“generation-one P positive”，不能写成“recursive retention、
evaluator co-evolution 或一般 Red Queen 已证明”。下一步是 F2→Q→一次 M2，不是继续修改 P 或补 gate。

后续实际执行改变了“尚未完成”但没有改变结论上限：F2/Q 已形成；M2 因 sandbox 阻止 bubblewrap network
namespace 而在评分前终止，0 score、no replay，所以 recursive retention 仍是 efficacy unknown。A_form/F3/A_hold/M3
则完整执行，但 challenger 在 A_form 与 incumbent 选择同一 program，A_hold 12/29 对 12/29 未晋升，M3 保持
incumbent 并得到 18/29 对 18/29、net 0。因此 evaluator co-evolution 也没有发生。下一条科学缺口已经不再是
“补 gate”。新的 HotpotQA family-out 随后已完成：P 未适配，12×3=36 路一次最大并发、36/36 terminal 与
fresh postflight 后离线得到 RAW 11/24、P 21/24、official HippoRAG 20/24；P−RAW 为 +10、7 gain/0 harm，
P−official 为 +1。family transfer 因此在 retrieval-only 小样本 scope 获得正证据；原 M2 与 L5 blocks 仍不得
重用，L4 retained recursion 与 L5 successful evaluator transition 仍是下一缺口。

### 12.4 2026-07-16 fresh Hotpot L4/L5 闭合

12.3 末尾所列的 L4/L5 缺口随后在不重用旧 block 的全新六分区 acquisition 上正式检验。L4 得到有效 positive：
P+Q=43/48，同时高于 Q=40/48 与 P=36/48，分别提供 +3 retained contribution 与 +7 novel contribution；
96/96 work unit、24 个 official worker、fresh postflight 与离线 source-support scoring 全部闭合。相对 official
HippoRAG 的 fixed-cohort 差为 +12，但没有据此追加 post-hoc significance 或 broad superiority claim。Q formation
的四折不稳定作为风险保留，不触发重选。

L5 则得到有效 negative：A_form/F_search 先证明 program 与实际 retrieval behavior 都不同，A_hold 再以 72 路
一次执行得到 challenger 38/48、incumbent 41/48、exact p=31/32。challenger 不晋升，archive 不 invalidation，
M_search 不授权、不打开。故最终状态不再是“L4 未测得、L5 wiring 不充分”，而是“窄 retrieval-only L4 positive；
behavior-identifiable L5 negative”。这正是预注册 evaluator 应允许出现的结果；任何在同一 anchor 上继续抽样、换
objective 或降低晋升阈值都会破坏证据。

### 12.5 2026-07-16 final Hotpot portfolio acquisition infrastructure-invalid 闭合

single-Q evaluator 负结果之后冻结的 final two-Q portfolio 机制没有产生新的 efficacy observation。design、实现与
preregistration 依次由 `b504f8b3`、`6f373fce`、`257d6283` 固定；纠正后的 acquisition 先消费 one-shot marker，
再打开 exclusion/source 并在内存选定 `[156,324)`，但在首个 private-root `os.mkdir` 因缺父目录失败。没有 block、
locator、acquisition receipt、retrieval 或 score。该失效已按 hash-only disposition 严格关闭；固定 window 不重建，
不进行 Hotpot v4。它既不能算 portfolio 阴性，也不能升级 L5，只把新机制的 Hotpot efficacy 留为 unknown；此前
family-out、L4 与 L5 结果不变。后续测试必须转到独立、fresh、事前提交的 family/domain，而非在 Hotpot 上补 gate。

### 12.6 2026-07-16 MuSiQue residual portfolio A_hold implementation-invalid 闭合

备选 A 已按原计划完成 residual acquisition 与 A/F formation，但没有产生 A_hold efficacy observation。两份
formation receipt 在 A_hold 前提交且各为 4080/4080 terminal；A_hold freeze 又在 0 row/label 状态固定 288-party
一次运行。正式调用消费 authorization、加载 48 行后，被 committed runner 的 lazy-submit/early-result 顺序错误
锁死在第一个 barrier participant，180 秒后 attempted=1、terminal=0 失败。因为问题发生在 authorization 与 block
open 之后，该 cohort 不能通过修复代码重放；因为没有 terminal、score 或 transition，也不能把失败当成 evaluator
阴性。M_search 没有授权或 artifact，继续保持未打开。

这次闭合把下一步约束得更清楚。终态提交后，`98763f27` 已在纯合成数据上修复 future bulk submission，并将
formation/A_hold/M_search 三个路径统一绑定为完整 bulk submit 后才 join；focused 16/16 与相关 grouped 66/66
tests 通过。该修复不回溯授权；随后若继续，只能使用独立新 family/domain 的 fresh preregistration。
不得新建 MuSiQue continuation、把 M_search 充当 A_hold、或围绕同一 action 追加 gate。既有 MuSiQue M1、Hotpot
family-out/L4/L5 claims 均不受影响；新 portfolio 的 MuSiQue efficacy 与 evaluator co-evolution 仍未知。

### 12.7 A 终止后仍保持干净的备选测试

本地盘点没有发现“零下载且 untouched”的 QA family。现有 HippoRAG reproduction rows 中，HotpotQA、MuSiQue、
2WikiMultiHopQA 各 1000 项都已被历史流程使用；MuSiQue 虽还有未抽取 row，但 final same-source policy 禁止继续，
Hotpot continuation 也已关闭。因此不能为了省下载把本地余量包装成 fresh family。

若目标仍是直接检验 L5 与相对 HippoRAG 的增益，首选是下载 official 2Wiki 的另一个 fresh split，先按公开 ID/hash
排除本地既有 1000 项，再对剩余集合做私有 HMAC selection；exact frozen A/F actions 与 corrected eager-submit runner
必须在任何 row 打开前绑定。这样最贴近现有 source-support recall 与 official Hippo core，但只能声称 fresh-item
family-out，不能声称模型从未接触 2Wiki family。更低污染的第二选择是 QASC：下载 official split 与 fact corpus，
把两条 gold fact 只用于 join 后离线 recall；它是真正的新科学问答领域，但 paragraph/title 映射需事前机械固定，
HippoRAG 只能作为 custom-dataset official-core 对照，不是官方 2Wiki benchmark。fresh NOAA GSOD 时间窗下载成本最低、
时间污染也低，但 HippoRAG 不适用，只能补 typed-operator/retention 证据，不能替代 evaluator co-evolution L5。

因此若继续当前主目标，建议顺序是 **fresh official 2Wiki → QASC → NOAA**。任何选择都只能有一个 row-zero
preregistration、一个 fresh cohort 和一次终态；不得把 A 的 M_search、MuSiQue 余量、Hotpot continuation、SEC13F
或 SkillLearn sealed 数据当作 backup。

### 12.8 2026-07-16 fresh 2Wiki 备选测试闭合

12.7 的首选已经执行完毕，不再是“待下载/待选择”。official archive、历史 1000-row exclusion、test-only
collision scan、fixed MuSiQue A/F action transfer、corrected eager-submit implementation、唯一 synthetic diagnostic、
acquisition、A_hold freeze 与 aggregate report 全部形成 committed custody。A_hold 384/384 terminal，两个 barrier
均为 192 parties；incumbent/P/official/RAW 为 111/110/99/56 support hits（总数 120）。incumbent 相对 official
为 +12，paired descriptive exact p=1549/262144；这给出了目前最强的 fresh-item retrieval transfer 证据。

但 evaluator 主问题仍是有效 negative：challenger 110 对 incumbent 111，净 −1、exact p=1，不 promotion。
因此 M_search 按设计保持未授权、未打开，不能提供“promoted evaluator 改善后续 search”的 L5 证据。当前距离完整
Red Queen 目标只剩但也确实缺少这一层：一次真实 evaluator epoch replacement、selective invalidation，以及在
事前冻结的后续 measurement 上改善 search。不能通过继续 2Wiki、重用 M_search 或新增 gate 来填补。若论文把 L5
视为硬性主张，下一次实验必须转向独立新领域和实质不同 evaluator objective；若不是硬性主张，当前证据已足以支撑
更窄且更稳的结论——typed retrieval 能跨 fresh items/family 转移并超过 local HippoRAG，而 evaluator 更新机制目前
只证明了正确拒绝，没有证明成功自我改写。

### 12.9 2026-07-17 QASC 新领域检验闭合

12.8 所列“转向独立新领域和实质不同 evaluator objective”已经执行，不再是待办。QASC 的 source custody、viewer
exposure limitation、rotated HMAC、NLI asset、16-recipe design、row-free infrastructure diagnostic、四块 acquisition、
A/F formation、A_hold freeze/report 与 terminal disposition 全部形成 committed public chain。正式 source selection
得到事前固定的 TRAIN `7175` / DEV `865`；两遍 full-corpus BM25 完整扫描 16,987,130 行。A/F formation behavior
可识别，故这不是因 candidate collapse 而无法检验。

结果仍然是有效 negative：challenger 总 U=84、incumbent=90，净 −6、exact p≈0.795835，不 promotion。更重要的是，
official HippoRAG 的 U=147、support=103/128，远高于 challenger/incumbent 的 84/90 与 66/67 support。由此可以把
“距离目标还缺什么”进一步收窄：不是第四个 gate，也不是继续换 evaluator scoring key；当前 action family 没有表达
HippoRAG 已表现出的 graph/two-fact coverage，而 counterfactual choice separation 也没有改善 support utility。

因此新的硬前提是 **action–evaluator joint mechanism change**：proposal grammar 必须先能产生 graph-aware retrieval
action；evaluator formation 再使用与未来 search improvement 同构的 coverage/robustness signal，而不能只优化 choice
margin。该机制若没有独立新 custody，就只允许做不评分的 retrospective design analysis；不得重用 QASC M_search、
读取其 labels，或在 Hotpot/2Wiki/QASC 已消费 anchor 上继续试到成功。论文当前最强、可复核的结论应停在：窄 L3/L4
已有 positive，L5 有三次有效 rejection，尚无 evaluator epoch replacement，也没有 promoted evaluator 改善后续 search。

### 12.10 2026-07-17 QASPER / FinQA 独立域 source-qualification 闭合

QASC 之后没有继续补 gate，而是依次转向 scientific-paper graph evidence 与 financial table-text evidence；两条路线都在
任何 item action、retrieval 或评分之前终止，因而不能算 efficacy negative。QASPER 先固定 source custody、公开样例
排除、四块容量合同与 aggregate-only qualifier；唯一正式 clean-worker 调用失败，没有 receipt、selection、model call 或
score。事后只读官方 reader 代码表明它允许 `section_name=null`，而冻结 qualifier 拒绝该分支；这证明实现兼容域比官方
schema 窄，但不能证明隐藏失败正由这一条件触发。按一次性政策没有修复重跑。同期冻结的离线 MiniLM runtime 是纯
asset/canary 结果，可供独立新来源复用，但没有取得 QASPER 性能证据。

随后 FinQA 在 source bytes 下载前提交了固定 commit、公开暴露整页 denylist、private HMAC、TRAIN/DEV 四块容量与
官方 HippoRAG-primary 边界；官方归档只做 whole-archive binding。具体 action/evaluator 也在 source row 打开前固定为
16 个 typed table-text graph swap recipes：cell/header 只能回落到官方 `table_i` parent，official HippoRAG top-5 是唯一
primary incumbent，A/F 使用 gold-free coverage evaluator，A_hold/M 才使用 `hits + complete` 离线 utility，且不存在
runner-up、补阈值或第二 objective。qualifier 与 QASPER 回归合计 45 项 synthetic test 通过。

但 FinQA 唯一正式 aggregate qualification 仍在 clean worker 内非零退出。worker 程序性打开了固定 TRAIN/DEV bytes，
却没有向父进程返回 receipt；无人看到 item ID、question、table、gold、program、answer 或 row，selection secret 未打开，
四个 private block 未选择，RAW/P/Hippo/action/evaluator/model/score 均为 0，TEST/private TEST 未开。事后只打开两个
官方代码 member；`read_mathqa_entry` 的 retrieval contract 只直接消费 `id`、`qa.question`、可缺省 `gold_inds`、
`pre_text`、`post_text` 和 `table`，而冻结 qualifier 还强制非空 `program/program_re` 与更窄的 `exe_ans` 类型。因此可
确认 **qualifier implementation 严于 official retriever**，但隐藏 worker 的精确失败条件仍未知。该 FinQA 来源现已按
预注册记为 implementation-invalid、efficacy unknown，同源不 replay、不缩块、不放宽 schema，也不用 unopened TEST
作 backup。

这两次失效把下一步工程约束进一步收窄，但没有引出新 gate：独立新来源必须直接复用或逐分支等价实现其官方
retrieval reader，只验证 action 真正需要的最小字段；不能把 generator-only annotation 类型伪装成 retrieval source
schema。新来源仍只允许一次 aggregate qualification、一次 fresh cohort 与一次终态。FinQA 的 row-free graph design
可以移植，FinQA 数据与 secret 不可再用；L5 状态保持未证明。

### 12.11 2026-07-17 ContractNLI 独立法律域 source-qualification 终止

ContractNLI 路线在 source row 前固定了 [source custody](../manifests/contractnli_graph_evaluator_source_custody_v1.json)、
[graph/evaluator design](../manifests/contractnli_graph_evaluator_design_v1.json)、
[source-access addendum](../manifests/contractnli_source_access_addendum_v1.json) 与
[TRAIN member binding](../manifests/contractnli_source_member_binding_v1.json)。它的目标不是复用 CUAD prompt，而是从
官方 document spans 形成 definition、exception、list-sibling 与 explicit-cross-reference 四类 typed edges，再用固定
recipe/coverage registry 检验一次 evaluator transition。该机制的 row-free graph core 与 synthetic tests 成立，但不携带
任何 ContractNLI 性能证明。

唯一正式 clean aggregate qualification 在 marker 已持久化后非零退出，没有形成 receipt；stdout、stderr 与 traceback
按冻结隔离协议没有转发。因而 TRAIN member 是否在失败前已被程序性打开未知，精确失败原因、schema/capacity 结果与
eligible content-group 数也未知；DEV、TEST、raw contract member 未打开，selection secret 未被 qualification runtime
读取，HMAC selection、四块 materialization、RAW/P/official HippoRAG、typed action、coverage evaluator、model 与 score
均为 0。commit `b018f948` 的
[terminal disposition](../manifests/contractnli_source_qualification_failure_disposition_v1.json) 因此把它分类为
`implementation_or_infrastructure_invalid`、source feasibility 未建立也未否定、efficacy unknown，而不是有效 negative。

正式 marker 已消费，所以同源 ContractNLI 不 replay、不缩块、不放宽 schema、不轮换 secret，也不把未开的 DEV/TEST
当 backup。typed clause graph core 只可作为 row-free engineering 复用；不能据此声明 ContractNLI retrieval utility、
相对 HippoRAG 优势或 evaluator replacement。此前 2Wiki、QASC、Hotpot、MuSiQue、QASPER 与 FinQA 的结论均不受影响。

### 12.12 2026-07-17 CUAD parent-process direct acquisition 容量终止

为避免 QASPER、FinQA、ContractNLI 所暴露的“clean-worker 无 receipt → 再换 source qualification”循环，CUAD 在任何
真实 row 打开前改为 no-prequalification、parent-process direct one-shot，并固定
[design](../manifests/cuad_graph_evaluator_design_v1.json)、
[source custody](../manifests/cuad_graph_evaluator_source_custody_v1.json) 与
[source-access binding](../manifests/cuad_graph_evaluator_source_access_v1.json)。首个 formal CLI 只读到公开 design manifest，
因其遗漏顶层 `schema` 而在 secret、archive hash/central directory、marker、member、selection 与 output 之前失败。
commit `2cb8718a` 的 [pre-marker incident](../manifests/cuad_pre_marker_invocation_incident_v1.json) 公开绑定了原 design
self/file hash 与唯一允许的 exact compatibility correction；它记录 formal CLI entry=1、marker-consuming attempt=0、
source/member bytes=0，而没有静默把这次 activation failure 擦除。

修正后的唯一 marker-consuming run 在 durable marker 后只打开固定 TRAIN member 一次，其他 member content open=0，
TEST/CUADv1 open=0。commit `3e458d5f` 的
[aggregate acquisition receipt](../manifests/cuad_graph_evaluator_acquisition_v1.json) 显示：408 个 contract records 形成
407 个 components，公开 exposure 排除 2 个，冻结 eligibility 最终只剩 232 个，低于事前 4×64=256，缺口 24。
aggregate parser reason 中 `node_cardinality=173` 是主要容量损失，`gold_cardinality=3`；offset mismatch、schema、
duplicate QA ID 与 omitted alignment 等错误均为 0。这里的 reason counts 不应相加解释为互斥 document 数，但足以定位
固定 node envelope 与长合同分布不相容，而不是 parser 普遍损坏。

容量不足发生在 block materialization 前：selected blocks/items=0、private files=0、model=0、online evaluator=0、
performance score=0。因此它没有运行 Agent、RAW 或 official HippoRAG，**不是性能负结果**，也不能改变 QASC 中
HippoRAG 明显领先的既有事实。commit `1b9aaaa5` 的
[capacity disposition](../manifests/cuad_graph_evaluator_capacity_disposition_v1.json) 已终止该来源：不 replay/resample、
不缩成 3×64、不改 node/gold/exposure/parser 条件、不旋转 secret，也不开 TEST/CUADv1 或其他 archive member。

### 12.13 EvidenceBench 独立科学域 direct acquisition 终止

CUAD 之后没有再写 clean-worker qualifier，而是选择了唯一通过许可、公开来源 pin、至少 256 个候选容量与科学文献
evidence 任务初审的 EvidenceBench。来源固定为 `EvidenceBench/EvidenceBench` commit
`bf1d9633c694381c7b016fd56ee9f95f48593cc3` 的 `datasets/evidencebench_test_set.json`，Git blob
`df380a1...e6513`、12,735,397 bytes；已公开论文示例的 PMCID/DOI/URL 整个 component 事前排除。设计把每篇论文精确
分为 32 个连续 bucket，以原 `hypothesis` 为 query，把每个 aspect 的官方 evidence sentence indices 映射成 alternative
gold bucket set；utility 是 aspect coverage，而不是把 aspect ID 错当自然语言 query。

commit `ebd80d25` 在任何 source bytes 下载前固定了 9-recipe/16-evaluator scientific graph core 的 direct parent-process
acquisition 和 stage-isolated runner；独立审计发现并在正式执行前消除了任意 pack/path、可重复 stage root、未绑定 prior
receipt、A-form/A-hold eager label load 等阻断。修正后每个 stage 只能使用 canonical paths，acquisition→formation→
A_hold→M_search 的公开 receipt 必须先提交并逐级匹配 current HEAD blob；任何 private pack 在 stage marker 前均不打开，
F_search label 永不加载。52 项离线 synthetic/mock test 通过。随后依次提交
[source custody](../manifests/evidencebench_graph_evaluator_source_custody_v1.json)、
[source-access binding](../manifests/evidencebench_graph_evaluator_source_access_v1.json) 与
[implementation freeze](../manifests/evidencebench_implementation_freeze_v1.json)；下载后只做 opaque whole-file SHA256/Git-blob/
size 校验，未在 marker 前 decode JSON 或打开 row。

唯一正式 acquisition 在 commit `a7bfc9d5` 的 actual HEAD 上验证 freeze 与全部 9 个文件后持久化 marker，随后打开并解析
固定 source 一次。它在 `form_paper_disjoint_selection` 的 root contract 立即终止：可确认的精确析取只有“decoded root
不是 list，或其长度不等于冻结的 293”，不能在不重开 source 的情况下进一步区分。commit `58063cc7` 的
[aggregate receipt](../manifests/evidencebench_direct_acquisition_v1.json) 与
[terminal disposition](../manifests/evidencebench_acquisition_terminal_disposition_v1.json) 记录：private blocks=0、selection=0、
model/RAW/official HippoRAG/Agent/score=0，efficacy 与 source feasibility 均未知。该来源不 replay、不改 root schema/293、
不缩 block、不旋转 secret，也不做 post-hoc root-shape diagnostic；它不是 Agent 的性能负结果，更没有缩小 QASC 中
HippoRAG 已有优势。

### 12.14 方案 A 终态：公开 grammar 的 synthetic mechanistic causal stress test

QASPER、FinQA、ContractNLI、CUAD 与 EvidenceBench 已足以否决“再找相似数据集、失败后补 schema/gate”的工作方式。
现实数据主线因此暂停，备选 A 改为**公开 grammar synthetic mechanistic causal stress test**；目的只是把 source/schema
不确定性拿掉后检验 typed graph intervention 是否真的造成可审计的 action utility，而不是把 synthetic 阳性包装成现实效果。

原始 commit `d24dfb96` 在任何 formal seed/cohort 生成前冻结
[公开 grammar](../assumption_agent/benchmarks/synthetic_typed_graph_causal_grammar_v1.py) 与
[causal design](../manifests/synthetic_typed_graph_causal_design_v1.json)。正式实现审计随后发现三项必须在 seed 前透明处理的问题：
TN2 metadata 声称 edge 指向 degree-matched decoy，但实际 DGP 保留 target endpoint；evaluator derangement 的排序标识间接包含
gold hash；DGP 又没有随机 treatment sign，因而完整 magnitude-sign enumeration 不能称为 design-based randomization p-value。
commit `b37054e2` 的
[pre-seed amendment](../manifests/synthetic_typed_graph_causal_preseed_amendment_v1.json) 没有更改 surface、nodes、edges、gold、
block quota 或增加 gate：TN2 被如实描述为“edge present、query/gold 由独立 direct cue 决定”，derangement 改用唯一
label-free commitment，sign enumeration 只保留为事前 one-shot protocol heuristic/reference tail。acquisition/runner、late-label
barrier、递归 receipt/marker/seal/历史 HEAD 验证与终止后复现发布一并固定；41/41 focused tests、`py_compile` 与 diff check 通过。

commit `80c8110f` 的
[implementation freeze](../manifests/synthetic_typed_graph_causal_implementation_freeze_v1.json) 绑定 21 个代码、测试、MiniLM 与
official-HippoRAG runtime 文件。唯一 seed 的
[custody](../manifests/synthetic_typed_graph_causal_seed_custody_v1.json) 在 commit `96364f68` 先公开 commitment；随后 commit
`2f98ce9b` 的 [acquisition receipt](../manifests/synthetic_typed_graph_causal_acquisition_v1.json) 一次形成 4×64=256 项，不存在
candidate pool/filter/retry/replacement。A_form、F_search、A_hold、M_search 的 label-free pack 全部形成；只有 A_form/A_hold/M_search
有独立 late-label pack，F_search label 从未创建。全程 offline judge=0、外部在线网络=0；official HippoRAG 最大 8 并发，
本地计算最大 64 并发。

commit `dd781261` 的 [formation receipt](../manifests/synthetic_typed_graph_causal_formation_v1.json) 得到 real evaluator
`E_DEF_HEAVY_L050`、real recipe `R1_DEFINITION_1SWAP`、permuted recipe `R5_DEFINITION_EXCEPTION_2SWAP` 与固定 E00 recipe
`R1_DEFINITION_1SWAP`。real/permuted recipe 的 observed action 不同，所以 transition 在 formation 的窄定义上可识别并授权
untouched A_hold；formation 分数仅用于形成，不作效果 claim。

commit `808f43a4` 的 [A_hold receipt](../manifests/synthetic_typed_graph_causal_A_hold_v1.json) 给出最终有效结果：

| A_hold arm | total U | support hit / 112 | complete / 64 |
|---|---:|---:|---:|
| Agent full | 168 | 108 | 60 |
| official HippoRAG | 164 | 106 | 58 |
| canonical RAW | 158 | 101 | 57 |
| Agent drop-designated | 164 | 106 | 58 |
| Agent wrong-type | 164 | 106 | 58 |
| Agent endpoint-permuted | 162 | 105 | 57 |

Agent full 相对 official HippoRAG 的 matched net U 是 +4，但 32 个 matched contrast 中只有 2 个 nonzero pair；完整
magnitude-sign reference tail 为 `1/4=0.25`，高于预注册 `0.1`。因此结果是 `valid_nonpromotion`，M_search 未授权且从未打开，
不能在当前 A_hold 上换 recipe、阈值、objective 或追加 seed。机制方向与设计一致：drop-designated 和 wrong-type 各损失 4 U，
endpoint-permuted 损失 6 U；但相应 reference tail 分别为 `1/4`、`1/4`、`1/8`，它们是非 gating 描述而非新的通过条件。
增益全部来自 `MENTIONS_DEFINITION` 的 positive family：该 family 的 U 为 Agent 44、HippoRAG 40、RAW 26，其余三类 Agent 与
HippoRAG 完全相同。real recipe、permuted-evaluator recipe 与 E00 在 A_hold 的总 U 又同为 168，real−permuted matched net U=0；
所以本轮没有证明 evaluator co-evolution。

终止后，commit `6f06464a` 按事前合同发布
[exact seed 与完整 256-row cohort](../published/synthetic_typed_graph_causal_v1/formal_seed_and_cohort.json)，不含 retrieval action、
model output 或 score；self-hash、seed commitment、四个 block commitment 与 exact regeneration 均已复验。结论因此不是
“Agent 没有任何作用”，而是更窄也更准确：**在这份 synthetic SCM 内，typed definition edge 的干预造成了方向正确但稀疏的
小幅收益；它只比 HippoRAG 多 4 U / 2 support hits / 2 complete items，未达到事前晋升规则，也没有 evaluator transition 或
family-out 证据。** 这不能支撑 official benchmark、production utility、现实总体效果或 Agent 普遍优于 HippoRAG。

距离目标仍缺两层证据。第一层是机制稳定性：若继续 synthetic，只能新建明确标为 post-terminal 的多 seed replication，事前
固定所有 seeds 与 pooled estimand，回答这个 definition-only +4 是否稳定；它仍不能补现实效度。第二层也是更关键的一层，是
在一个全新且与现实 reader 等价、许可/容量/隔离均合格的来源上，证明跨 relation family 的稳定 Agent−HippoRAG 净收益。
当前结果不授权回到同一 64-item A_hold 继续调 gate，也不支持消耗已密封的 M_search。

### 12.15 FEVER fixed-P 现实域备选：source-schema terminal

synthetic 终态后，对本地可见来源做了只读排查。GSM8K 与 HumanEval 已存在同题代码、输出或机制污染；唯一接近现实 evidence
retrieval 要求的是 FEVER，但本地只有 reference repo 内的 `paper_dev.jsonl`，缺少匹配的离线 Wikipedia corpus。因而没有把
reference copy 当正式源，而是先依据 FEVER 官方公开的 labelled paper split、JSONL schema、June-2017 preprocessed wiki 与
许可页面建立新的 source chain。

commit `543bed23` 的
[source custody](../manifests/fever_official_fixed_transfer_source_custody_v1.json) 在下载前固定官方 `paper_test.jsonl`
（2,181,168 bytes）、`wiki-pages.zip`（1,713,485,474 bytes）与 `license.html`（670 bytes）的 HTTPS URL、ETag、长度与
Last-Modified；commit `e5d5a7d7` 的
[source-access binding](../manifests/fever_official_fixed_transfer_source_access_v1.json) 随后只做 opaque whole-file hash，得到
SHA256 `fb7b0280...d88e`、`4b06d95d...65f2`、`eaddbd6e...d05`。三文件存入 Git-ignored mode-0600 目录；此时 archive
listing、decompression、JSON decode、row/wiki sentence open、selection 与 score 均为 0。FEVER annotations 继续受官方
Wikipedia/CC-BY-SA 条款约束；该许可链没有被 Apache-2.0 baseline code license 替代。

commit `e07cf640` 的
[fixed-P design](../manifests/fever_fixed_p_itemlocal_reranking_design_v1.json) 没有重新 formation 或新 evaluator：它冻结既有
MuSiQue P `0e9fea15...0785`，只计划一次 64 SUPPORTS + 64 REFUTES 的 HMAC holdout。每题先 HMAC 固定一套 1–5 句完整
evidence set，再把 gold 句注入由 full-wiki frozen BM25 hard negatives 组成的 32-sentence pool；RAW、item-local official
HippoRAG 与 exact frozen P 只做 top-5 reranking。指标只有 selected-set support hits、complete count 与 descriptive paired
delta；没有 p-value、promotion、gate、M_search、retry 或第二 cohort。claim boundary 也事前写死：这不是 full-Wikipedia
FEVER retrieval、label prediction、official FEVER score 或开放域优越性。

commit `83e185d7` 的
[acquisition](../assumption_agent/benchmarks/fever_fixed_p_itemlocal_acquisition_v1.py) 与
[runner](../assumption_agent/benchmarks/fever_fixed_p_itemlocal_runner_v1.py) 把 source/selection/action/late-label 隔离、8 路 official
与 64 路 local eager barrier、fresh postflight 及 aggregate-only receipt 固定下来；13/13 synthetic tests、`py_compile` 与独立
formal audit 通过。commit `406fe880` 的
[implementation freeze](../manifests/fever_official_fixed_transfer_implementation_freeze_v1.json) 绑定 21 个代码、测试、P 与
runtime 文件；commit `7d70cce3` 的
[selection custody](../manifests/fever_official_fixed_transfer_selection_custody_v1.json) 在 source parse 前只公开唯一 secret
commitment。

唯一 acquisition 先持久化 marker，随后完整解析 `paper_test`，仅按 label 和 syntactic 1–5-reference eligibility 在内存中固定
128 rows 与各自 evidence set；没有用 wiki resolvability 或 BM25 coverage 换 row。它进入 wiki pass-1 后先打开 central
directory，在任何 JSONL member content 打开前触发冻结条件：至少一个非目录 member 的 suffix 不是 `.jsonl`。commit
`694d5184` 的 [failure receipt](../manifests/fever_official_fixed_transfer_acquisition_failure_v1.json) 与 commit `2ecb7a4a` 的
[terminal disposition](../manifests/fever_official_fixed_transfer_acquisition_terminal_disposition_v1.json) 将其固定为
`source_schema_invalid / efficacy_unknown`：wiki page/sentence parse=0、action pack=0、label pack=0、RAW/P/Hippo action=0、
model/evaluator/score=0。空 identity ledger 只由冻结控制流推断，未作 post-failure query；offending member 名也没有事后查看或
公开。

因此不把这个失败解释成“只要忽略 README/metadata 就能继续”。正式 cohort 与 selection secret 已消费；同源不 replay、
不添加 member allowlist、不解压后手工挑 JSONL、不换 `paper_dev/shared_task_dev`、不旋转 secret，也不把内存中已选 128 rows
交给修正版 runner。此前 synthetic definition-only +4、QASC 中 HippoRAG 优势及其他现实域结论均不受影响；FEVER 固定 P
相对 RAW/HippoRAG 的效果仍完全未知。

### 12.16 post-terminal synthetic 8-seed replication：action 前 implementation-invalid

单 seed A_hold 的 +4 U 只集中在 definition-positive family，因而下一项没有继续改 recipe、阈值或 promotion gate，而是建立
[post-terminal 8-seed design](../manifests/synthetic_typed_graph_multiseed_replication_design_v1.json)：8 个 fresh seed 各只调用一次
公开 grammar 的 `generate_block(seed, "A_hold")`，每个 seed 固定 64 项；exact `R1_DEFINITION_1SWAP`、RAW、official HippoRAG 与
Agent_R1 三臂不再 formation/search。primary 只描述每个 seed 的 Agent−HippoRAG 总 U 差及八个 cluster 的 mean/median/min/max/
range/K-positive；Agent−RAW 与三臂 overall/by-seed/by-family/by-polarity 是事前声明的 secondary 描述。没有 threshold、p-value、
confidence interval、pass/fail、promotion、gate 或 population inference；原单 seed +4 只作排除在 estimand 外的历史参照。

seed 前的两轮独立审计发现并一次性关闭五类 formal defect：official paragraph title 必须与原 causal run 完全相同；Agent worker
不能在 1,536 个 future 全部 submit 前读取 official future；RAW/三臂分层与 Agent−RAW 必须进入 design；terminal publication 必须
同时支持 success 与 infrastructure-invalid；publication 不能再次调用 grammar、把 8 次生成变成 16 次。因此 commit `dabcbde7` 的
[acquisition](../assumption_agent/benchmarks/synthetic_typed_graph_multiseed_acquisition_v1.py) 在同一批 8 次生成中额外密封 mode-0600、
逐行及整包自哈希的 full-compiled cohort pack，[runner](../assumption_agent/benchmarks/synthetic_typed_graph_multiseed_runner_v1.py)
使用 submission barrier，并把 success/failure 统一写入 canonical result。64 项相关 grammar/formal/runtime/adapter 测试通过；隔离临时
Git root 还按真实 CLI 与逐阶段 commit 顺序验证了 freeze→seed-custody→acquire，没有触碰正式 root。

commit `2ecf5ec8` 的
[implementation freeze](../manifests/synthetic_typed_graph_multiseed_replication_implementation_freeze_v1.json) 绑定 25 个文件，
self-hash=`46aa3baf...fe7`。commit `5efbb5b1` 的
[seed custody](../manifests/synthetic_typed_graph_multiseed_replication_seed_custody_v1.json) 只在 durable marker 后调用一次
`os.urandom(256)`，按顺序切成 8×32 bytes；8 个 commitment 互异且不与原 seed 相撞。commit `f7d3335b` 的
[acquisition receipt](../manifests/synthetic_typed_graph_multiseed_replication_acquisition_v1.json) 随后一次形成 512 个彼此唯一、且与原
64-item A_hold 不重叠的 item commitment；action、late-label 与 full-compiled pack 均为 512 rows、mode 0600，receipt 绑定六个
file/set hash。没有 A_form/F_search/M_search、candidate pool、formation、filter、recipe search、网络或 score。

唯一 formal runner 在 committed marker 后打开并验证 label-free action pack，随后尚未进入 `_execute_all_actions` 就在
`precompute_local_tensors` 终止。实现把 512 个 question 与每题 32 个 node text 合成一次 `encoder.encode`，精确为
`512×33=16,896` texts；冻结的 `OfflineMiniLMEncoder` 单次上限是 `MAXIMUM_TEXTS_PER_CALL=16,384`，因此抛出
`QasperMiniLMError: text count is outside the frozen bound`，再被封装为 `SyntheticTypedGraphMultiseedRunnerError`。正式 work root、
action seal 与 late-label open 均未发生；RAW、official HippoRAG、Agent_R1 action、post-action score 与 seed-level delta 都是 0。
这不是 provider/network/capacity 波动，也不是 Agent 的性能负结果，而是 tests 使用无该 runtime bound 的 fake encoder、正式
preflight 又只实例化资源而没有验证 full-call shape 所遗漏的 implementation defect。

commit `d7d4b86d` 的
[canonical terminal result](../manifests/synthetic_typed_graph_multiseed_replication_result_v1.json) 已把该轮固定为
`terminal_infrastructure_or_implementation_invalid_no_replay`，receipt=`3f7c5e6d...a7ee`；没有修改 chunking 后重放、缩小 seed 数、
替换 seed 或追加 cohort。commit `d185b84a` 随后只执行事前授权的 terminal publication，公开
[exact 8 seeds 与 512-row cohort](../published/synthetic_typed_graph_multiseed_replication_v1/formal_seeds_and_cohort.json)，
reproducibility self-hash=`f54998ce...a13c`，明确不含 retrieval action、model output 或 score。labels 与 cohort 现已公开，故同一
cohort 也不能交给分批修正版 runner。

该 v1 attempt 的当时终态结论是：**8-seed stability 仍为 unknown。** 原单 seed Agent−HippoRAG +4 U 的窄 synthetic mechanism signal 没有被复现，
也没有被否定。若另立 v2，必须使用全新 seeds/cohort，并在 seed 前让真实 frozen encoder 对 exact 16,896-text call shape 或事前
固定的 deterministic chunk schedule 做 executable preflight；但这会是新的独立研究，而不是本轮 repair/replay，而且仍无法补
现实效度。为避免“失败后不断补 gate/contract 再试到成功”，当时应停止对该 v1 cohort 的 synthetic repair；12.17 记录后来另立
全新 cohort 后的独立 v3→v5 研究。论文距离目标仍缺现实域中跨 relation
family 的稳定 Agent−HippoRAG 净收益，以及 evaluator challenger 晋升后对 untouched search 的真实改善。

### 12.17 post-terminal multiseed v2→v5：窄机制稳定性完成，但不构成现实域或 L5

12.16 的“8-seed stability unknown”是 v1 在第一次 action 前因 MiniLM 单次 `16,896` rows 超过冻结上限而终止时的
正确结论；它不是对后续研究的永久禁止。后续工作没有在已公开的 v1 cohort 上改到成功，也没有继续添加 performance
gate，而是把每一次失败严格分成独立、事前声明且不读取 outcome 的 execution-repair 阶段。

首先，commit `3f8a779a` 的
[v2 design](../manifests/synthetic_typed_graph_multiseed_replication_design_v2.json) 与 exact v2 kernel 将 MiniLM 固定为
`8,448+8,448` 两段，并把 official/local 并发上限固定为 8/64。它只在已经公开的 v1 cohort 上做非评分 integration
diagnostic；唯一进程被外部终止，commit `6d60139c` 的
[diagnostic receipt](../manifests/synthetic_typed_graph_multiseed_replication_integration_diagnostic_v2.json) 因而是
`terminal_integration_diagnostic_invalid_fresh_formal_not_authorized`。labels、scores、estimand 与 claim 均为 0，不能把它当成
chunk repair 的性能反证，也没有为 v2 生成正式 seed/cohort。

commit `d1b6771b` 的 [v3 design](../manifests/synthetic_typed_graph_multiseed_replication_design_v3.json) 改用 detached
`systemd --user` custody；row/model-free
[preseed verification](../manifests/synthetic_typed_graph_multiseed_replication_preseed_verification_v3.json)、
[implementation freeze](../manifests/synthetic_typed_graph_multiseed_replication_implementation_freeze_v3.json)、
[seed custody](../manifests/synthetic_typed_graph_multiseed_replication_seed_custody_v3.json) 与一次性
[acquisition](../manifests/synthetic_typed_graph_multiseed_replication_acquisition_v3.json) 随后形成全新 `8×64=512` A_hold。
generated item set 为 `22cdb517…2fcc2f`，action/label/compiled file hash 分别为 `56feacea…0f55`、
`caf548c9…ff78`、`e09d1ac5…b561`。正式 launch 前发现 frozen driver 的 `Path.resolve(strict=True)` 会消除
`venv/bin/python` 的 lexical symlink identity，而 frozen v2 attestation 恰好要求保留该 identity。commit `5b43dd74` 的
[v3 prelaunch closure](../manifests/synthetic_typed_graph_multiseed_replication_prelaunch_terminal_v3.json) 因此在 marker、pack open、
action、label、score 全为 0 时关闭 v3；正式 attempt 未消费，这份 prospectively acquired cohort 仍 untouched。

v4 只预注册 lexical-path transport repair，没有授权修改 kernel/cohort/metric。临时 official-HippoRAG runtime 消失后，
SmolLM2 revision `12fd25f…`, 11 个 payload（272,030,008 bytes）、依赖、52-file source 与 embedding 都能重建；但 11 个
Hugging Face `.metadata` 第三行由 `time.time()` 生成，旧 timestamp bytes 无法恢复，而 v2 raw attestation 又把它们计入
asset/topology identity。故 commit `8842a327` 的
[v4 prefreeze closure](../manifests/synthetic_typed_graph_multiseed_execution_repair_prefreeze_terminal_v4.json) 在 implementation
freeze 之前终止；v4 attempt 仍未消费，也没有正式 v4 implementation code/result。

commit `df96bf93` / `da449cd3` 的
[v5 design](../manifests/synthetic_typed_graph_multiseed_runtime_normalization_design_v5.json) 在实现前只授权四项变化：保留 lexical
symlink；仅从 runtime identity 排除已经验证为 finite nonnegative float 的 download timestamp 行；机械地把 adapter verifier
从 v2 换成 v3；修正 failure receipt 的 pack/label open state 与 systemd invocation provenance。payload path/size/hash、commit、
ETag、`.gitignore`、Python target、dependency/source/embedding bindings 仍严格验证，额外/symlink/lock/temp 文件均拒绝。
commit `8cf78fd5` 的
[runtime attestation v3](../manifests/musique_official_hipporag_runtime_attestation_v3.json) self-hash 为 `23996f9f…2c60`；
normalized LLM、filesystem 与 path-free safe binding 分别为 `8d3cd27a…b4bb`、`3330c67e…e894`、`818f16bc…6038`。

v5 lifecycle 在冻结前经两轮独立审计。后审计关闭了三个 blocker：finalizer 只有在完整、正向的 systemd terminal evidence
下才能写 failure；`systemd-run` 非零不能无证据声称 child/pack 未启动；success readback 必须逐项复验 exact v2 receipt、
`512/1536` counts、8/64 caps、两段 8448、pack commitments、private action seal 与 512 action rows。focused 28/28 tests
通过且复审未发现新的 P0/P1。commit `ef4d918b` 固定实现；commit `e0498961` 的
[v5 implementation freeze](../manifests/synthetic_typed_graph_multiseed_runtime_normalization_implementation_freeze_v5.json)
self-hash=`5f7d3535…162a`，只按 path/mode/size/raw hash 绑定三份 v3 private pack，`semantic_payload_opened=false`，没有新 seed、
cohort、smoke、candidate search 或 gate。

唯一 v5 formal attempt 随后完成。launcher 与 child 的 MiniLM/runtime preflight 都在 action pack 前通过；全部 1,536 futures
先提交，observed official/local peak concurrency 为 8/64。512 次 official retrieval、512 RAW 与 512 Agent_R1 action 全部
terminal；MiniLM observed input/output 均严格为 `[8448,8448]`。fresh official postflight 后先写 private action seal
`f1bdc1d7…2c81`（file `f4a387c7…ce0d`），late-label pack 才恰好打开一次并离线评分。commit `a83f6d54` 的
[canonical v5 result](../manifests/synthetic_typed_graph_multiseed_replication_result_v5.json) 为
`terminal_descriptive_eight_seed_replication_complete`，self-hash=`3e2bb0f9…2084`：

| arm | total U | support hit / 896 | complete / 512 |
|---|---:|---:|---:|
| Agent_R1 | 1259 | 817 | 442 |
| official HippoRAG | 1232 | 801 | 431 |
| RAW | 1273 | 813 | 460 |

primary Agent_R1−official-HippoRAG seed deltas 为 `[3,8,0,2,1,5,6,2]`：mean `3.375`、median `2.5`、
min/max `0/8`、7 positive / 1 tie / 0 negative，总 U `+27`，可分解为 support `+16` 与 complete `+11`。
这把单 seed 的稀疏 `+4` 更新为**同一固定 synthetic distribution 内可跨 seed 保留的窄 positive mechanism signal**。
但它并不广：全部 `+27` 只来自 `DEF_TP1 +10` 与 `DEF_TP2 +17`；其余 14 个 family 的 Agent 与 HippoRAG total U
完全相同，negative polarity 也完全相同。相对 RAW 的 seed deltas 又是 `[0,-5,-6,-2,-5,2,4,-2]`，总 U `−14`；
Agent 虽多 4 个 support hit，却少 18 个 complete item。因此“Agent 已普遍超过 HippoRAG/RAW”仍然不成立。

terminal result 提交后，commit `b9b9989c` 才执行事前授权的
[exact v3 seed/cohort publication](../published/synthetic_typed_graph_multiseed_replication_v5/reproducibility.json)，
reproducibility self-hash=`c9cbf99e…2962`；publication 没有再次调用 grammar，也不含 action、model output 或 score。
协议没有 threshold、p-value、CI、promotion 或 performance gate，所以以上是 fixed-cohort descriptive replication，不能包装成
population inference、official benchmark、端到端 QA、现实 family-out、L4 retained recursion 或 L5 evaluator co-evolution。

因此 synthetic 分支现在应以**有效完成**而不是再次 repair 结束：不再对当前 cohort 补 gate、换 recipe/evaluator 或重跑。
距离总目标仍缺两项没有被这次阳性缩短的核心证据：其一，在全新现实 reader/domain 中跨 relation family 保持稳定的
Agent−HippoRAG 净收益，并同时说明为何不被 RAW 的 complete advantage 反超；其二，一个真正晋升的 evaluator challenger，
在预先冻结的 untouched search cohort 上改善后续搜索。若论文不再追加一个同时改变 graph/two-fact action 与 evaluator
mechanism、且一次性预注册 A/F/A_hold/M 的新现实域 study，最诚实的终稿结论就是：**现实 retrieval-only 的窄 L3/L4 与 synthetic mechanism
stability 已有证据，evaluator replacement/L5 仍未达到。**

### 12.18 2026-07-19 HoVer joint graph/evaluator：正式有效 non-promotion，并定位为候选生成退化

这项新现实域研究没有继续 Hotpot/2Wiki/QASC 的旧 cohort，也没有在结果后追加 gate。official HoVer TRAIN 与其固定
Wikipedia SQLite 先通过 source qualification；随后一次 private-HMAC acquisition 同时形成 A_form/F_search/A_hold/M_search
`48/36/30/30` 项和一个 609-document closed corpus。该任务是 TRAIN-only、transductive、oracle-gold-containing 的
derived retrieval task，不能写成 official HoVer、open-domain benchmark 或 relation-family transfer。公开
[design](../manifests/hover_joint_graph_evaluator_design_v1.json)、
[implementation freeze](../manifests/hover_joint_graph_implementation_freeze_v1.json) 与
[acquisition receipt](../manifests/hover_direct_acquisition_v1_acquisition.json) 分别绑定设计、45-role/36-Python 实现闭包和四块
private pack；M_search 的 utility label 从未创建为 F_search，且只有 promotion 才允许打开 M。

正式 acquisition 前第一次入口暴露了一个 Git subdirectory pathspec bug：repo-root relative path 却从
`reconstruction_v2/` 执行，因而错误查找双重前缀。失败发生在 qualification HEAD 校验、TRAIN/SQLite read 与 acquisition
marker 之前；commit `0b4ef6b6` 只把后续 Git 命令统一到 worktree root，并增加真实 nested-repo regression，49 项相关测试
通过；commit `f0d6f763` 重新冻结实现。之后 acquisition 一次成功并由 commit `48a97596` 提交。controller 的两次环境
preflight 又分别发现 `-I` 看不到 user-site distribution metadata、以及 pinned `sentence-transformers` 导入时由 Torch
生成临时 remote-module path 导致 project 不再位于 `sys.path` 末端；二者都在 lifecycle marker 与 private view/label read
之前终止。最终执行环境只把已经版本锁定的依赖预载到 project path 之前，不改 frozen code、action、evaluator、cohort 或
threshold；正式 runtime preflight 记录 0 inference、0 private-pack decode、0 network，并通过 MiniLM、NER 与 official
HippoRAG attestation。

唯一正式 lifecycle 随后完整执行到 A_hold。official HippoRAG global index 只 build 一次；RAW、HippoRAG 与六个 Agent
action 对每块最大并行，A_form label 在 action seal 与 evaluator freeze 后才打开，F_search 没有 utility label。A_form
label-free E0/E1 选为 P0/P5；独立 F_search 又冻结为 E0=`P0_IND_SUM`、E1=`P3_TOPIC_BRIDGE`。公开
[A_form action seal](../manifests/hover_a_form_action_seal_v1.json)、
[A_form evaluator freeze](../manifests/hover_a_form_evaluator_freeze_v1.json)、
[F policy freeze](../manifests/hover_f_search_policy_freeze_v1.json) 与
[A_hold action seal](../manifests/hover_a_hold_action_seal_v1.json) 把 complete trace matrix、policy 和 late-label 顺序逐级锁定。

A_hold 的冻结 utility 为 `distinct-gold recall + complete bonus`，30 项安全聚合如下：

| arm | total U | distinct gold hits | complete / 30 |
|---|---:|---:|---:|
| RAW | 487/12 | 72 | 16 |
| official HippoRAG | 487/12 | 72 | 16 |
| P0_IND_SUM（E0） | 487/12 | 72 | 16 |
| P1_IND_MAXIMIN | 487/12 | 72 | 16 |
| P2_ENTITY_BRIDGE | 65/6 | 30 | 0 |
| P3_TOPIC_BRIDGE（E1） | 110/3 | 63 | 15 |
| P4_META_ASSIGN | 505/12 | 73 | 17 |
| P5_FAMILY_UNION | 63/2 | 56 | 12 |

E0−HippoRAG 与 E0−RAW 都是 `0`，30/30 item utility tie、exact sign-flip `p=1`，2/3/4-hop 三层 delta 也都是
`0`。更强的非评分 trace audit 表明 P0 与 RAW/official HippoRAG 在 30/30 项选择相同 top-5 集合；12 项只有顺序变化，
而冻结 utility 不计顺序。E1−E0 则为 `−47/12`：2 gain / 10 harm / 18 tie，exact `p=3739/4096`；其中 recall
从 `295/12` 降到 `65/3`（`−35/12`），complete 又从 16 降到 15（`−1`）。P4 虽事后总 U 比 baseline 高
`3/2`，但全部来自单个 2-hop item、1 gain / 0 harm / 29 tie、`p=1/2`；它不是预先冻结政策，不能在看到 A_hold 后
改选 P4、补 threshold 或重跑。

[terminal result](../artifacts/hover_joint_graph_formal_v1/formal_result.json) 因此有效地记录
`valid_A_hold_nonpromotion_M_unopened`：primary=false、promotion=false，M_search view/labels 均未打开；online evaluator 与
external network calls 都是 0。它不是 implementation-invalid，也不是“小幅输给 HippoRAG”：E0 在定义的 utility 上与
HippoRAG/RAW 精确打平，而 challenger 明确变差。这使 L5 更新为第四次 behavior/action-identifiable rejection；仍然没有
evaluator replacement，也没有 promotion 后改善 untouched search 的证据。

静态机制审计进一步解释了退化，而不是再提出新 gate。HoVer adapter 把 source/category 设为每文档唯一的 missing
sentinel、date 为空，query plan 又固定没有 normalized source，因此 E0 的 metadata-coverage 首维对六个 action 恒为 0；
P0 的 pair/extension/tail 都按 dense relevance sum 排序，数学上退化为 RAW dense top-5。P3 则不受 query plan 锚定，先在
全语料最大化 reciprocal-topic 边与连通性，再用 relevance 破平；E1 的 necessary-fraction 首维词典序把“删点易断的稀疏
topic clique”误当作因果必要，后续相关性再高也无法补偿。A_form 的 E1=P5、F_search 却翻为 P3，也显示这个结构键跨 block
不稳定。

当前 cohort 只允许继续做不改变决策的描述性根因诊断，不能选择 P4/P5、调权、追加 relevance gate、重跑 A_hold 或越过
non-promotion 打开 M。若继续总目标，下一最小但实质不同的机制应是 query-anchored residual path：新 action
`P6_QUERY_ANCHORED_RESIDUAL_PATH` 只允许由 claim entity/relation facet 锚定边，逐步补充 baseline top-5 尚未覆盖的 facet；
doc-doc bridge 只有连接 query anchor 时才计分，禁止 query-independent corpus clique。配套 evaluator
`E2_QUERY_INTERVENTION` 使用冻结本地 embedding/NLI 衡量 claim-facet 语义覆盖，并以删除/同型替换后的连续覆盖下降作为
干预效应，不再让 necessary-count 的单个脆弱首维支配全部相关性。P6/E2 必须在任何新 outcome 前冻结，并进入全新现实
custody；HoVer DEV 最多能做同域 fresh test，不能把 hop strata 包装成 relation family。离线资产盘点中，优先的新 domain 是
FEVEROUS（text/table/cell/cross-page/aggregation family），其次是 HybridQA/OTT-QA 的 table-linked-passage task；两者都需要
新下载 official dataset 与对应 corpus。IIRC 已有 public source audit disposition，因 license、规模和缺少原生 relation-family
标签被拒，不再列作候选。若不再承担这一项新 study，当前最诚实的终稿结论仍是：**现实域没有稳定 Agent−HippoRAG 净收益，L5 未达到；
系统已证明能正确拒绝候选，而不是已经实现 evaluator co-evolution。**

### 12.19 2026-07-19 FEVEROUS source epoch v3：adapter 可穷尽，但 aggregate compatibility 未获证明

FEVEROUS 路线没有沿用 v1/v2 的 root、secret 或失败后的 partial state。v1 在 TRAIN decode 后、DB open 前终止；v2 完成
TRAIN decode、53.5 GB SQLite SHA 校验与 partial adapter scan 后，在第一个 content/title exact-page mismatch 处终止。
后续只做 aggregate-only、无评分诊断，确认全 TRAIN 恰有 2 个 reference、2 个 evidence set、2 个 record 命中同一形态；更早的
[Wikipedia source qualification](../manifests/feverous_wikipedia_source_qualification_v1.json) 已在任何 cohort selection 前记录
`casefold_nfd_only_title_context_occurrences=2`，并明确要求整组排除。因此这不是看结果后新增关键词或 gate，而是补回冻结来源协议中
遗漏的 typed exclusion：只允许 exact 不同但 `NFD→casefold` 相等的 content/title page 被分类为 invalid whole-set；不改写 ID、
不 fuzzy resolve，任何无关跨页、非-title context member 跨页、缺/多 title 或非法 ID 仍 fail closed。invalid set 的 content ID
只保留在 distractor 禁入 universe，不进入 canonical evidence set；set/reference/record 三类计数与 family-structure exclusion 分开。

实现与资格闭包分三次提交：`96234cf5` 固定 adapter、aggregate receipt 和 synthetic tests；`ac2919c0` 再把实际 candidate scan
执行到的 qualification runner、Wikipedia resolver、formal source、adapter、atomic corpus、acquisition core、strict JSON decoder、
blank-sentinel predicate 与 `unicodedata` version 全部纳入 receipt；`c2c7efb8` 仅把 source qualification 的 exact-context subset
`75,219 / 338,061` 与 adapter 在 exclusion 前递增的 `official_*` counter 明确分层，不改变 source、candidate、selection 或 evaluator。
相关 synthetic suite 均通过，且没有生成或读取任何 v3 selection secret。

正式 compatibility qualification 没有成功产生 receipt。第一次不足绑定的 prepass 在 TRAIN 已完整读取、SQLite SHA 尚未完成时因
独立审计主动中止；resolver 未打开、adapter 未调用。补齐绑定后的第一次 completed pass 完整哈希 DB 并穷尽 resolver/adapter，
但在最终 aggregate 断言处以 generic `FeverousAdapterCompatibilityQualificationError` 终止。根据预先公开的 source aggregate 与
counter 定义修正会计口径后，最后一次 capped pass 再次完整穷尽 adapter，仍在同一断言处终止。runner 没有打印 actual mismatch
字段或值，也没有把 adapter aggregate、candidate 或 raw identifier/claim/label/evidence 写盘；因此不能继续根据隐藏差异调常数再试。

[terminal failure receipt](../manifests/feverous_p6_e2_adapter_compatibility_qualification_v3_terminal_failure.json)
以 `04ee399d40f475b23a013d8e2eaa1ffd8e70cea29fce38908a35e97c8c066b46` 固定三次边界：两次 completed pass 都是
adapter exhaustion、0 receipt；整个 v3 为 0 root、0 secret、0 HMAC/cohort/corpus、0 RAW/P6/HippoRAG、0 evaluator/score、
0 DEV/TEST/online evaluator。故结论是 **source adapter execution 可完成，但冻结 aggregate compatibility 未被证明**；不能把它写成
Agent 对 HippoRAG 的性能负结果，也不能建立 v3 rollover 或 v4。下一项总目标实验必须转到独立 domain/custody，优先审计
HybridQA/OTT-QA 的 official table-linked-passage source，并直接采用一次性 acquisition + offline evaluation；不得在 FEVEROUS 上
继续补 gate 或用更多全量 pass 追逐 topology 常数。

### 12.20 2026-07-19 HybridQA P6/E2：首次 evaluator promotion，但现实域 primary 与 untouched L5 均未通过

HybridQA 的第一轮 official TRAIN source epoch 已封存为 implementation-invalid，而不是性能负结果。一次性 acquisition
成功形成 144 个三-family、全局 question/table-disjoint item 和 609-unit shared corpus；controller 也已完成
`A_form / F_search / A_hold` 的 label-free action、feature 与 official HippoRAG 计算。但第一份 A_form archive 写盘后，冻结
verifier 错把完整 `ActionTrace.trace_sha256` 与较窄的 `RecipeTrace.behavior_sha256` 要求相等，因而在任何 late-label open
之前必然终止。生产端的 action/type/order 没有漂移，错误只在 durable archive consumer；但按 one-shot 协议，该 root 和
TRAIN cohort 仍不可救援、补评分或重跑，efficacy 保持 unknown。见
[`v1 terminal disposition`](../manifests/hybridqa_p6_e2_formal_v1_implementation_failure_disposition_v1.json)。

修复没有回写 v1，而是新建 v2 controller：从 archive JSON 严格重构 typed `SelectionStep / ActionTrace` 并验证完整 operator
self-hash，同时独立重算 evaluator behavior hash；真实
`execute_item → BlockExecution → persist → durable verify` 合成 roundtrip 已纳入冻结测试。后续 formal study 使用完全独立的
official DEV source epoch、新 root、新 secret 与新 freeze；不读取 v1 TRAIN private packs，不打开 TEST，也不改变 P6/E2、utility、
promotion threshold 或 block size。DEV acquisition 一次完成，候选容量为 DUAL 258、PASSAGE_ONLY 1328、TABLE_ONLY 375，最终仍按
`48 / 36 / 30 / 30` 形成 A_form/F/A_hold/M。整个 v2 为 0 retry/resample、0 online/Ruoli evaluator，四个 block 都在 label open
前形成 durable label-free archive。

F_search 冻结出 behavior-identifiable 的不同 policy：E0 选择 `R1_P6_DIRECT_B2`，E2 选择 `R0_DENSE5`。A_hold 首次产生了真正的
evaluator promotion，因此 M_search 按预注册 capability 正常授权并打开；但最终结果把“evaluator 晋升”和“晋升后改善 untouched
search”清楚地区分开：

| block / comparison | net U | nonzero pairs | exact one-sided p | disposition |
|---|---:|---:|---:|---|
| A_hold E2−E0 | +41/6 | 5 | 1/32 | evaluator promoted |
| A_hold E2−official HippoRAG | −1/2 | 2 | 3/4 | real-domain primary false |
| A_hold E2−RAW | 0 | 0 | 1 | RAW advantage not overcome |
| M_search E2−E0 | +3/2 | 1 | 1/2 | M_L5 false |
| M_search E2−official HippoRAG | +1/2 | 1 | 1/2 | descriptive only |
| M_search E2−RAW | 0 | 0 | 1 | RAW advantage not overcome |

A_hold complete count 为 E0/E2/HippoRAG/RAW=`18/22/22/22`；E2−HippoRAG 的 family sums 是
DUAL=`0`、PASSAGE_ONLY=`+3/2`、TABLE_ONLY=`−2`。M_search complete count 为 `13/14/14/14`；对应 family sums
是 DUAL=`0`、PASSAGE_ONLY=`+1/2`、TABLE_ONLY=`0`。所以这不是“比 HippoRAG 赢得不多”，而是预注册 primary 明确未通过：
A_hold 总差为负且 family 条件失败；M 的正差只来自一个 nonzero pair，exact p=`1/2`。E2 在两个 anchor 上又都与 RAW utility
完全相同。

机制解释也已收敛：监督 evaluator 确实在 A_hold 正确拒绝了表现较差的 P6 direct policy，但它晋升的是 dense `R0`，并没有形成
一个比 RAW/HippoRAG 更强的新 typed action。换言之，首次 promotion 证明了 evaluator replacement、authorization 与 untouched M
消费链路真实可执行；M 未通过则说明 **L5 evaluator co-evolution 仍未达到**，因为晋升没有在后续搜索中产生足够稳定的新收益。
同一结果还再次否定了现实域稳定 Agent−HippoRAG 增益与 RAW advantage overcome。

当前 HybridQA DEV cohort 已终结：不得放大 M、修改 alpha、换 utility、补 gate、改 policy 后再测，TEST 也不用于追结果。若继续总目标，
下一候选必须改变 action generation 本身，使 Agent 能从 dense top-5 之外生成与 query anchor 因果相连的新增证据，而不是再训练一个
evaluator 在 `R0` 与近似退化的 P6 recipes 之间选择；正式 measurement 只能进入独立 domain/source epoch 并重新 freeze。完整安全聚合见
[`v2 result disposition`](../manifests/hybridqa_p6_e2_formal_v2_result_disposition_v1.json)，原始 public terminal receipt 见
[`terminal result`](../artifacts/hybridqa_p6_e2_formal_v2/controller/lifecycle.terminal_result.json)。

### 12.21 2026-07-19 ERASER Evidence Inference R7/E3：恢复有效，但真实域三臂与 evaluator 均明确负

下一项独立现实域使用 ERASER Evidence Inference TRAIN/VAL，事前固定三类临床 effect relation、`48 / 36 / 30 / 30`
四块、完整 article-local sentence corpus、RAW / official HippoRAG / Agent 三臂、`R0_DENSE5` 与
`R7_QUERY_ANCHORED_ATOMIC_PATH_BUNDLE` 两个 action，以及只从 A_form 拟合的八特征无截距 ridge E3。原 v1 已进入
`A_form + F_search` 预定 252 个逻辑任务的首轮、但尚未形成任何 schedule/archive/label/score 时遭主机硬中断；残留恰有
32 个 A_form HippoRAG work root，没有 output/result。由于没有 durable schedule receipt，不能反推全 252 项已提交或任何 future
完成；它因此被封为 infrastructure-invalid，而非性能结果。

结果盲 crash-recovery v2 在任何 label/score access 前单独预注册并冻结：原 499-entry tree 原子归档；只 byte-for-byte 克隆
同一 secret/assignment 的五个 acquisition 基础文件；不重跑 qualifier/acquire、不生成 secret、不 resample，也不复用任何 partial
cache。资源审计发现每个 item-local HippoRAG 进程还会扩张大量内部 OpenIE/Torch 线程，所以物理 pool 从上限 32 固定为 2；这不拆波，
仍由冻结 scheduler 在第一次读取 future result 前一次提交完整 `3×84=252` 和后续 `3×30=90` 个逻辑任务。正式 receipt 确认两次
`all_3n_tasks_submitted_before_first_result=true`，外部网络/online evaluator/test access 均为 0。

A_hold 是有效的强负结果，而不是“比 HippoRAG 赢得不多”：

| comparison | net U | nonzero pairs | exact one-sided p | complete count delta |
|---|---:|---:|---:|---:|
| E3−E0 | −13/2 | 4 | 1 | −3 |
| E3−official HippoRAG | −13/2 | 4 | 1 | −3 |
| E3−RAW | −13/2 | 4 | 1 | −3 |

E0/E3/HippoRAG/RAW complete counts 为 `9/6/9/9`。E3−HippoRAG 的三个事前 relation-family net U 又全部为负：
`NO_SIGNIFICANT_DIFFERENCE=−2`、`SIGNIFICANTLY_DECREASED=−1/2`、`SIGNIFICANTLY_INCREASED=−4`。
E3 在 30 项中选择 R7 12 次、R0 18 次；这 12 次全部 behavior-distinct，全部有 used-edge deletion action change，共引入
40 个 RAW5 之外的 sentence。因此失败不是 typed action 没执行、没有越出 RAW，或 HippoRAG 没跑，而是**可识别、因果活跃的 R7
route 在真实 held anchor 上产生净伤害**。

evaluator 的失败也能在事前 A_form descriptive cross-fit 中看到：四折非零 utility item 共 11 个，held correct-direction 只有 5 个；
冻结 E3 随后仍在 A_hold 激活 12 个 R7 route，最终对 E0、RAW 与 HippoRAG 都是同一个 `−13/2`。所以不应再给同一个 router
补 feature、关键词、threshold 或 promotion gate。A_hold promotion 为 false，M_search 没有 materialize/open/measure，L5 仍为 null。

另有一个不影响 A_hold 的透明元数据边界：外部进度监控在正式 controller 之外做了 15 次 block-phase check，对已知但始终不存在的
HippoRAG `M_search` runtime directory name 调用了 30 次 `Path.is_dir()`；没有枚举目录、没有 stat/open acquisition M view，亦没有
接触 M query/document/label/gold。因此 controller 的零调用 postflight 仍成立，M 数据仍未 materialize 或测量；但全局意义上不能再写
“任何进程从未 stat 过任何含 M_search 的路径”。该事实已单独封存，且不授权 M、recovery retry 或新 gate。

至此总目标的两个缺口均未闭合：现实域稳定 Agent−HippoRAG 增益仍不存在，evaluator 也没有晋升，更没有改善 untouched search。
ERASER 同 cohort、同 source epoch 与 sealed M 全部停止。下一条允许的路线只能进入独立 domain/source epoch，并把学习作用于
evidence-sufficiency / terminal-selection 的 action generator 本身，而不是继续训练一个在 dense R0 与当前有害 R7 之间切换的 gate。
安全聚合见 [`recovery result disposition`](../manifests/eraser_evidence_inference_r7_e3_recovery_v2_result_disposition_v1.json)，
恢复终局见 [`recovery terminal`](../artifacts/eraser_evidence_inference_r7_e3_crash_recovery_v2/controller/recovery.terminal_result.json)。

### 12.22 2026-07-19 DocRED G8/E1：正式源资格终止，efficacy 未知

ERASER 后的新 domain/source epoch 选择 official DocRED TRAIN/DEV，先在任何 source row 打开前冻结三类 relation family、
`96 / 48 / 36 / 30 / 30` 五块、top-3 typed structured-set generator G8、pairwise evaluator E1、完整 dense RAW3 与
official-core HippoRAG 对照。assignment/controller、G8/E1 core 及 synthetic tests 在提交 `ebb606ce` 后由
[`pre-row implementation freeze`](../manifests/docred_structured_set_decoder_pre_row_implementation_freeze_v1.json) 固定；正式 provenance
preflight 通过后才创建唯一 formal root。

正式 one-shot 只各打开一次 `train_annotated.json`、`dev.json` 和 `rel_info.json`，随即在 aggregate source qualification 终止。
唯一非零 schema anomaly 是 `sentence_token`：TRAIN 169 个 invalid document、DEV 50 个，共 219；root、document keyset、label、mention、
endpoint 与 evidence 等其他 anomaly 均为 0。值得区分的是，排除这些 invalid document 后的公开聚合容量仍能同时分配要求的 240 个
document-disjoint item，因此终止原因不是 family/cohort 容量不足，而是冻结的 minimum-schema-equivalent 条件不成立。

该终止发生在 secret generation 之前：没有 HMAC/cohort/private view，没有 G8/E1 formation，没有 RAW、HippoRAG、Agent、utility、
promotion 或 M_search，online/external evaluator 仍为 0。故它既不是 implementation-invalid，也不是 Agent 性能负结果；DocRED efficacy
保持 unknown。一次性协议不允许把 219 项改为 post-hoc exclusion、放宽 token schema 后重跑、重组 cohort 或复用 formal root。
安全聚合已封存在
[`terminal disposition`](../manifests/docred_structured_set_decoder_formal_v1_source_qualification_terminal_disposition_v1.json)，原始公开聚合见
[`qualification receipt`](../artifacts/docred_structured_set_decoder_formal_v1/source_qualification_receipt.json) 与
[`terminal incident`](../artifacts/docred_structured_set_decoder_formal_v1/terminal_incident.json)。总目标仍缺同样两项实证；下一次正式测量只能
换独立 domain/source epoch，并继续让学习直接作用于 evidence sufficiency / terminal selection，而不是给旧候选补 gate。

### 12.23 2026-07-19 MAVEN-ERE G8/E1：generator 有描述性增益，但 evaluator 有效 non-promotion

DocRED 终止后转入 official MAVEN-ERE TRAIN/valid。source qualification 在不选择 cohort 的 aggregate pass 中一次通过：TRAIN/valid
分别为 2,913/710 个 reader-valid document，3,623 个 collision component；按事前优先级 `CAUSAL > SUBEVENT > TEMPORAL` 可完整容纳
`96 / 48 / 36 / 30 / 30` 五块共 240 项。正式设计固定 article-local exact sentence、端点关系边双向删除、RAW3、official-core
HippoRAG top-3、完整 Set3 G8 generator，以及从 A_form 拟合的 pairwise ridge E1；本地线程上限 16、HippoRAG 物理进程上限 2，
每块仍在读取第一个 future 前提交完整 `3×n` 逻辑任务。设计、实现与 acquisition 分别见
[`formal design`](../manifests/maven_ere_g8_e1_formal_design_v1.json)、
[`implementation freeze`](../manifests/maven_ere_g8_e1_implementation_freeze_v1.json) 和
[`acquisition result`](../manifests/maven_ere_g8_e1_acquisition_result_v1.json)。整个链路只使用离线 MiniLM/NLI/HippoRAG；source
TRAIN/valid 各打开一次，online/external evaluator 与 hidden TEST 均为 0。

v1 controller 在 G8 已拟合、48 个 A_form 的三臂动作已全部返回、但 A_form label 尚未打开时终止。原因不是 action/model 输出坏掉，
而是 archive 中 Python tuple 经 JSON 写盘后自然变成 list，旧 `_durable_roundtrip` 却把 decode 后的 list 与写盘前 tuple 直接比较，
因而必然报 semantic drift。v1 efficacy 保持 unknown，原 root 没有原地重跑或补评分；边界见
[`v1 failure disposition`](../manifests/maven_ere_g8_e1_formal_v1_implementation_failure_disposition_v1.json)。

随后单独冻结 result-blind v2 recovery：不重采样、不换 secret、不重新打开 released source row，也不重跑 A_form 三臂；只重算 G/A
label-free semantics，要求 G archive 与 G8 model 和 v1 字节级一致，并在打开 A_form label 前逐项复验已有 archive 的 RAW3、G8 frontier、
E0 behavior、selection shape 与 `3×n` submission receipt。真实 `BlockExecution → archive → JSON normalize → durable readback` 回归测试纳入
30 项 MAVEN suite。所有复验均通过后才拟合 E1，再一次执行 F_search 与 A_hold。见
[`recovery design`](../manifests/maven_ere_g8_e1_result_blind_recovery_design_v2.json)、
[`recovery implementation freeze`](../manifests/maven_ere_g8_e1_result_blind_recovery_implementation_freeze_v2.json) 和
[`A_form validation`](../artifacts/maven_ere_g8_e1_result_blind_recovery_v2/controller/A_form.reused_action.validation.json)。因此 v2 是有效恢复，
不是 implementation-invalid。

A_hold 的正式结果为：

| comparison | net U | nonzero pairs | exact one-sided p | family net `C / S / T` |
|---|---:|---:|---:|---:|
| E1−E0 | −3 | 5 | 31/32 | `0 / −1 / −2` |
| E1−official HippoRAG | +1 | 5 | 1/2 | `0 / +2 / −1` |
| E1−RAW | +1 | 5 | 1/2 | `0 / +2 / −1` |

E0/E1/HippoRAG/RAW correct count 为 `15/12/11/11`。E1 在 30 项中有 28 项与 E0 behavior-distinct，18 项存在 used-edge
deletion action change、累计 304 个 deletion witness；所以 evaluator 确实改变了行为，机制也有因果活性，但它相对 E0 净伤害 3 项，
promotion 明确为 false。E1 对 HippoRAG/RAW 的 `+1` 既不显著，CAUSAL 又为 0、TEMPORAL 为负，real-domain primary 也明确为 false。
M_search 因未晋升而未打开，故 L5 是 **未测量**，不是失败分数。

这里出现了比前几轮更具体的机制线索：固定 G8/E0 的总正确数描述性地比 HippoRAG 和 RAW 各高 4，而监督 E1 把该优势削弱到 `+1`。
这不能事后升级为未预注册的 E0 primary 或显著性结论，却说明当前最有希望的组件是 typed generator，不是 pairwise evaluator。
当前 source epoch 与未打开的 M 均终止，不能在同一 anchor 上修改 threshold、feature 或再加 switch gate。若继续总目标，应在 fresh cohort
中把 G8/E0 冻结为 generator incumbent，挑战者必须换成实质不同的 setwise expected-utility/uncertainty evaluator，并一次性同时检验
promotion、Agent−HippoRAG/RAW 和新的 untouched M。安全聚合见
[`result disposition`](../manifests/maven_ere_g8_e1_result_blind_recovery_result_disposition_v2.json)，原始 aggregate terminal 见
[`recovery terminal`](../artifacts/maven_ere_g8_e1_result_blind_recovery_v2/controller/recovery.terminal_result.json)。

### 12.24 2026-07-19 MAVEN-ERE global-family E2：唯一 TRAIN cross-fit 未授权 formal

为避免把 E1 的负结果变成继续补 switch gate，下一候选改成实质不同的 global-family E2：从全部 sentence、authorized sentence、
head/tail mention sentence 的三族 NLI aggregate 拟合 18 维 multiclass ridge，先预测 item family，再在**完整 typed Set3 空间**中选择
对该 family margin 最大的候选，G8 energy 只作次级 tie-break。该机制、四折划分和唯一 feasibility rule 在任何新 training-pack reopen 前
由 [`TRAIN cross-fit design`](../manifests/maven_ere_global_family_e2_train_crossfit_design_v1.json) 与
[`implementation freeze`](../manifests/maven_ere_global_family_e2_train_crossfit_implementation_freeze_v1.json) 固定；只允许重开原 G_form/A_form
训练 pack，F/A_hold/M 全部禁止，source row、online evaluator 与 TEST 仍为 0。

四折每次使用 G_form 96 项加 A_form 其余 36 项训练，在 held A_form 12 项评价；总计 48 项且每族 16。结果是
E0/E2/RAW correct=`21/22/18`，E2 与 E0 在 32 项 behavior-distinct，但 paired E2−E0 只有 `+1`、17 个 nonzero、exact p=`1/2`；
family net 为 CAUSAL=`+2`、SUBEVENT=`+4`、TEMPORAL=`−5`。E2−RAW 虽为 `+4`，p=`3473/16384≈0.212`，TEMPORAL 仍为 `−1`。
所以总量上的一项提升不能掩盖明显的 family harm；事前 feasibility 不成立，final E2 model 没有 materialize，也没有 fresh formal cohort。

这条线已严格停止：不能在相同四折上改 feature、lambda、family threshold 或加 fallback gate 直到成功。它进一步说明 current NLI
global-family posterior 对 CAUSAL/SUBEVENT 有信号，但 TEMPORAL 的结构不能由同一 aggregate 可靠覆盖。后续最干净的下一步不是 E2 v2，
而是先在完全 fresh、排除既有 240 个 collision component 的 cohort 上对固定 G8/E0 做独立三臂确认；只有其跨三族的
Agent−HippoRAG/RAW 优势真正复现后，才值得在**新 training cohort**发展另一 evaluator。安全聚合见
[`E2 result disposition`](../manifests/maven_ere_global_family_e2_train_crossfit_result_disposition_v1.json)，原始训练结果见
[`cross-fit result`](../artifacts/maven_ere_global_family_e2_train_crossfit_v1/crossfit.result.json)。

### 12.25 2026-07-19 MAVEN-ERE fresh G8/E0：先前描述性优势未复现

E2 停止后，没有继续试 evaluator，而是直接检验上一轮唯一正向线索：固定 G8/E0 在原 A_hold 中比 HippoRAG/RAW 多 4 个 correct。
该确认在 original secret 和 valid source 重开前由
[`fresh confirmation design`](../manifests/maven_ere_g8_e0_fresh_confirmation_design_v1.json) 与
[`implementation freeze`](../manifests/maven_ere_g8_e0_fresh_confirmation_implementation_freeze_v1.json) 固定。runner 只读一次 original secret
和 official valid，不打开任何原 private pack；它从 source+secret 精确重构 original A_hold/M assignment，要求四个 view/label pack 的
公开 file/pack hash 全部相等，然后排除这 60 个 collision component。新 secret 一次分配 60 个 fresh item，每族 20；无 retry、replacement、
TRAIN/TEST/online access。随后固定读取既有 G8 model，不再打开 G/A training pack，并在读取 label 前提交和封存完整 `3×60` 三臂动作。

fresh 结果不是边缘 non-significant，而是方向反转：

| comparison | correct count | net U | nonzero pairs | exact one-sided p | family net `C / S / T` |
|---|---:|---:|---:|---:|---:|
| E0 | 18 | — | — | — | — |
| official HippoRAG | 22 | E0−Hippo=`−4` | 12 | 3797/4096 | `−1 / −4 / +1` |
| RAW | 22 | E0−RAW=`−4` | 12 | 3797/4096 | `−1 / −4 / +1` |

E0 仍不是“没有执行”：60 项中 9 项有 used-edge deletion action change，共 485 个 deletion witness；但因果活跃不等于收益。
原 30 项上的 E0 `+4` 只能认定为不稳定的描述性波动，不能再作为 generator incumbent 的现实域证据。尤其 SUBEVENT 在 fresh cohort
净输 4，CAUSAL 也输 1；只有 TEMPORAL `+1`，完全不满足跨 relation family 稳定性。

因此 MAVEN-ERE source epoch 到此终止：G8、E1、global-family E2 都不能在同 source 上继续改 feature、prompt、threshold 或补 gate。
总目标的两项关键实证仍都缺失：没有稳定 Agent−HippoRAG/RAW 优势，也没有 evaluator promotion→untouched-search improvement。
下一候选若继续，必须换独立 domain 与 fresh training cohort，并让监督直接对应 evidence sufficiency 或下游 task utility；不能再用同一个
NLI family scorer 同时充当生成特征、训练 target proxy 和最终效用来源。安全聚合见
[`fresh result disposition`](../manifests/maven_ere_g8_e0_fresh_confirmation_result_disposition_v1.json)，原始 aggregate terminal 见
[`fresh terminal`](../artifacts/maven_ere_g8_e0_fresh_confirmation_v1/controller/terminal.result.json)。

### 12.26 2026-07-19 SciFact direct-evidence：事前平衡 family 容量不足，未进入 efficacy

MAVEN-ERE 终止后，下一条路线按 12.25 的约束转向独立科学事实核验域，并把 utility 直接定义为“所选 Set3 完整覆盖至少一套官方
gold rationale”，不再复用 NLI family score 充当最终效用。official SciFact repository 固定在 commit
`68b98a56d93e0f9da0d2aab4e6c3294699a0f72e`；归档 SHA256=`11c62128...d76be`、3,115,079 bytes。来源下载后、任何 member
payload 打开前提交了 [`source custody`](../manifests/scifact_direct_evidence_source_custody_v1.json)，如实记录只读过 tar headers 与
official `doc/data.md` 两个示例，并把 claim `123/263` 及相连 document component 事前排除。TEST 只见到 header 名称与大小，payload
未解包、未单独哈希、未打开。

[`source-qualification design`](../manifests/scifact_direct_evidence_source_qualification_design_v1.json) 在 row 前固定三个互斥 family：
`CONTRADICT_SINGLE`、`MULTI_SENTENCE`、`SUPPORT_SINGLE`；固定 TRAIN 的 G/A/F/A_hold 共 `52×3=156` 项和 DEV M_search
`10×3=30` 项，并要求 claim/cited-document 全图 component-disjoint。qualifier 只允许解包 corpus、TRAIN、DEV 三个 member，使用
component→family exact max-flow 检查同时容量；它不是 performance gate，不生成 secret、不选 item、不运行 action/model/evaluator。
六项 synthetic regression 与 provenance preflight 通过后，
[`implementation freeze`](../manifests/scifact_direct_evidence_source_qualification_implementation_freeze_v1.json) 先提交，随后唯一 formal
qualification 才打开三份允许的 source payload。

[`aggregate terminal result`](../manifests/scifact_direct_evidence_source_qualification_result_v1.json) 明确显示该设计不可行，而非 action
负结果。TRAIN 可分配 component 数为 single-contradict `81`、multi `19`、single-support `162`，固定需求各 `52`；DEV 对应
`16/5/42`，固定需求各 `10`。max-flow 只能完成 TRAIN `123/156` 与 DEV `25/30`，总缺口 38，全部由稀缺
`MULTI_SENTENCE` family 造成。来源另有一条 DEV row 的 `cited_doc_ids` 重复，按冻结 minimal schema 计一个 schema error；evidence-to-corpus
映射错误为 0。公开示例 component 排除后，clean candidates 为 TRAIN `100/21/210`、DEV `21/5/43`；跨 TRAIN/DEV component
隔离又分别排除了 `206` 与 `132` 个候选，说明短缺不是 parser 普遍失效。

正式 terminal 的边界是：selection secret/item/block/action/RAW/official HippoRAG/Agent/evaluator/score/online call 全为 0，TEST
payload open=0；因此 SciFact efficacy 完全未知，不能把它计为 Agent 输给 HippoRAG。该 source epoch 也不能通过把 multi family 降到
`19/5`、改成 structured/unstructured family、取消 cross-split component 隔离或忽略重复 cited ID 后重跑。这个结果再次说明，下一来源
必须在公开统计层面就有充足的**多事实/多跳 explanation**，而不是事后把稀有 multi-rationale 子群硬凑成三分之一。后续应转向自带
proof/explanation graph 且 TRAIN/DEV 容量公开充足的独立来源；仍只允许一次 aggregate qualification，不回到 SciFact 补 gate。

### 12.27 2026-07-19 EntailmentBank G1/E1：Agent 与 HippoRAG 近似持平，但 evaluator 反向且未晋升

SciFact 因 source capacity 终止后，下一来源转向 official EntailmentBank Task2（repository commit
`daac2fdb7ab52ec3ef8f2953f59288c1edd7c2f0`）。TRAIN/DEV 文件分别为 10,867,722 / 1,537,951 bytes，SHA256
为 `36cdb362...55a1b` / `3271adc6...a28f7`；TEST payload 始终未打开。事前 source qualification 把 gold proof leaf 数固定成
`TWO_LEAF / THREE_LEAF / FOUR_FIVE_LEAF` 三族，并要求 ID、normalized question、normalized hypothesis 的传递闭包 component-disjoint。
aggregate-only qualification 得到 TRAIN clean candidates `265/294/415`、可分配 components `253/292/405`，DEV clean candidates
`37/44/48`，足以同时满足 TRAIN 每族 52 与 DEV 每族 10；因此这次确实进入了 efficacy，而不是再次用 capacity gate 代替实验。

[`formal design v1`](../manifests/entailmentbank_proof_retrieval_g1_e1_formal_design_v1.json) 在 selection 前固定 186 项：
G/A/F/A_hold=`60/36/30/30` 来自 TRAIN，M=`30` 来自 DEV，所有 block 三族平衡。每项只有 question、answer、hypothesis 与按官方
JSON insertion order 的 25 条 exact fact；utility 是 Set5 命中的 distinct gold proof leaves 数，再加“完整覆盖全部 gold leaves”的 1 分。
G1 是八个冻结离线特征上的 weighted ridge；16 个 action recipe 是四个 seed rank 与四个 graph-bridge α 的笛卡尔积。E0 是固定
label-free proxy；E1 用 A_form 中 16×36 个 frozen action 的真实 utility 拟合 ridge，然后 E0/E1 分别在完全无 label 的 F_search 上选
一个全局 Q0/Q1。A_hold primary 只允许 `Q1−Q0>0` 且 exact one-sided sign-flip `p≤0.10` 时晋升并打开 M；不存在 runner-up、阈值修改或
同源重试。

第一次 acquisition v1 在 source 各读一次后、component 构造与 HMAC selection 之前 fail-closed：实现错误地要求 `split:ID` 行唯一，
而官方数据允许同 ID 多行变体。它没有选 item、没有生成 pack、没有运行 action/model/label/score；旧 secret 永久退役，failure 由
[`v1 terminal receipt`](../manifests/entailmentbank_proof_retrieval_acquisition_failure_v1.json) 固定。随后先提交
[`v2 remediation design`](../manifests/entailmentbank_proof_retrieval_g1_e1_formal_design_v2.json)：唯一改动是把私有 source line ordinal
加入 row identity 与 full-item commitment，component 等价关系、family、quota、HMAC flow、recipe、evaluator、promotion 和 controls 全部不变。
22 项 regression 与三套 row-free 离线 runtime preflight 通过、[`v2 implementation freeze`](../manifests/entailmentbank_proof_retrieval_g1_e1_implementation_freeze_v2.json)
提交后，fresh v2 secret 一次选定 186 项；[`v2 acquisition receipt`](../manifests/entailmentbank_proof_retrieval_acquisition_receipt_v2.json)
复现 qualification 的 1,091 components、15 个 cross-split components 与所有 clean counts，`F_search.labels` 从未创建。

formation 在 A label 打开前已封存 A/F 的全部 16-recipe action matrix。E0 选出 Q0=`R00_G_RIDGE_A0000000`；E1 选出
Q1=`R09_MINILM_HYPOTHESIS_A0250000`。值得警惕的是，A_form 的**真实** aggregate utility 已经是 Q0=`99`、Q1=`97`，即 challenger
在形成样本上也没有超过 incumbent；它只是按 E1 对 label-free F_search 的预测总分胜出。冻结后 A_hold 的正式结果如下：

| arm / comparison | total U | net | nonzero pairs | exact one-sided p | family net `two / three / four-five` |
|---|---:|---:|---:|---:|---:|
| Q0 | 96 | — | — | — | — |
| Q1 | 92 | Q1−Q0=`−4` | 7 | `57/64` | `0 / −4 / 0` |
| official HippoRAG | 90 | Q1−Hippo=`+2` | 1 | `1/2` | `0 / +2 / 0` |
| RAW | 19 | Q1−RAW=`+73` | 26 | `1/67,108,864` | `+25 / +24 / +24` |

因此“看起来没有比 HippoRAG 赢太多”是准确判断：promoted candidate Q1 只多 2 分，29/30 项与 HippoRAG utility 相同，证据远不足以
支持稳定优势。预先固定的 Q0 描述性地比 HippoRAG 多 6 分，但 post-terminal exact audit 为 `35/256≈0.137`，而且净收益
`0/+6/0` 全部集中在 THREE_LEAF；它同样不满足跨 family 稳定正收益。相反，两种 Agent recipe 对 RAW 的优势很大，说明 typed semantic
retrieval 本身有效，但 official HippoRAG 已吸收了绝大部分可得收益。

[`A_hold result`](../manifests/entailmentbank_proof_retrieval_g1_e1_ahold_result_v2.json) 因 `Q1−Q0=−4` 合法 non-promotion；
[`final result`](../manifests/entailmentbank_proof_retrieval_g1_e1_final_result_v2.json) 因而终止，M view/label 与 TEST 均未打开。
这条 source epoch 不能再改 E1 feature、ridge、recipe、threshold 或把 sealed M 改成 generator confirmation。对总目标的更新是：首次在新
proof domain 得到 Agent≈official HippoRAG、显著优于 RAW 的有效现实域证据，但仍没有跨 family 的稳定 Agent−HippoRAG 净收益，更没有
evaluator promotion 后改善 untouched search 的 L5。下一候选必须在**新来源/新 cohort**采用实质不同的 evaluator mechanism，例如直接学习
recipe pairwise utility difference、带 out-of-source calibration 的 conservative selector，并事前允许保留 incumbent；不能继续用同类
absolute-utility ridge 在近乎等价的 recipe 上外推。

### 12.28 2026-07-20 BRIGHT reasoning-retrieval：aggregate 略高 HippoRAG，但 family 不稳定且仍低于 RAW

本轮在 BRIGHT 的 Biology、Economics 与 Robotics 三个现实 retrieval domain 上检验同一条总目标：固定 Agent action 是否能稳定超过 RAW 与 graph-bearing HippoRAG，以及 evaluator challenger 是否能晋升并改善后续 search。源数据绑定 commit `3066d29c9651a576c8aba4832d249807b181ecae`；169,540 篇文档只使用本地 Qwen2.5-1.5B、MiniLM 与 candidate-restricted official HippoRAG core。该 HippoRAG 对照在同一 label-free 32-document pool 内建图并返回 top-10；它不是 full-corpus official BRIGHT leaderboard，也没有 answer generation、在线 evaluator 或外部网络调用。

v2 的 G_form 在 heterogeneous padded batch 上 1,800 秒超时，没有产出 label 或 score。非评分诊断证明最长单项在 batch-1 可于 23.32 秒完成，因此问题是执行器 padding/scheduling，而不是模型不可用。事前提交的 v3 length-bounded executor 固定 ascending token schedule、batch≤8、non-singleton padded-token budget 4,096 与 3,600 秒 timeout；GPU canary 两次 byte-exact。随后 G_form 30/30、A_form 60/60、F_search 45/45 与 A_hold 45/45 均有效完成。F_search 选出的 retained P_base 是 `P6_RELATION_MECHANISM_RRF`：其 F_search sum integer nDCG 为 `5,025,679,677`，RAW 为 `4,860,392,609`。E1=`SAME_FAMILY_ONLY_K09_A01` 在 A_form leave-one-out 上看似达到 `6,030,079,236`，但 untouched A_hold 的 P6/E1 分别为 `5,453,493,905 / 3,728,324,087`，E1−P6=`−1,725,169,818`，4 gain / 4 harm / 37 tie，因此合法 non-promotion，active evaluator 保持 E0/P6。

原 v3 M_search 已完成 45/45 Qwen/local actions 与 24 个 HippoRAG terminal outputs 后收到外部 `SIGTERM`。当时 `three_arm.action.json` 与公开 M result 尚未生成，冻结实现只会在所有 futures join 和 action seal 之后读取 labels，所以 M labels 保持 sealed。该 root 被记为 infrastructure-invalid、efficacy unknown，并永久不 replay；它既不是 P6 的性能负结果，也不能补 L5。

替代 measurement 没有重跑 v3 M，而是在原 HMAC-randomized RESERVE 顺序中事前固定每 family 前 15 项，共 45 项，另留 37 项不消费。active Agent 仍是未看 reserve outcome 的 P6；E1 只作为 counterfactual，三臂 primary 为 P6 / RAW / candidate-restricted official HippoRAG core。第一次 acquisition 在既有 view/label packs 写完后，因公开结果组装中的 Python 小写 `false` 触发 `NameError`。失败收据在只检查文件存在性、大小和 SHA-256 后提交；随后 result-blind recovery design、实现与 freeze 均在解析 packs 前固定。恢复只验证并保留原 attempt/view/labels，不重采样、不覆盖、不读取 source assignments，也不运行模型或评分，最终仅补写缺失的 public acquisition result。因此这次异常是可审计的 implementation-invalid acquisition attempt，不构成 outcome-driven cohort replacement。

正式 reserve prepare 得到 45/45 valid generation，并在任何 HippoRAG join 前持久化 135 个 RAW/Agent/HippoRAG intents。detached action 阶段以冻结上限 12 并发、单 controller launch 完成 45/45 terminal outputs；没有启用第二次 recovery launch，最小 graph 为 41 nodes / 18 edges。`three_arm.action` seal 后才一次打开 label pack 并离线评分。公开 aggregate 如下：

| arm | sum integer nDCG@10 | mean nDCG@10 | Biology / Economics / Robotics family sums |
|---|---:|---:|---:|
| Agent P6 | 6,541,965,406 | 0.145377 | 1,764,593,165 / 2,649,083,995 / 2,128,288,246 |
| E1 counterfactual | 6,319,108,577 | 0.140425 | 1,487,336,222 / 2,455,285,912 / 2,376,486,443 |
| official HippoRAG core | 6,119,227,934 | 0.135983 | 2,229,277,037 / 2,480,822,973 / 1,409,127,924 |
| RAW | 6,693,156,962 | 0.148737 | 1,736,359,743 / 3,349,852,646 / 1,606,944,573 |

Agent−HippoRAG 的 aggregate delta 为 `+422,737,472`，即 mean `+0.009394`；但逐 family 为 Biology `−464,683,872`、Economics `+168,261,022`、Robotics `+719,160,322`，且逐题只有 7 gain / 9 harm / 29 tie。预注册的“三 family 全正”因此为 false。Agent−RAW 为 `−151,191,556`，mean `−0.003360`，5 gain / 7 harm / 33 tie；E1−retained P6 为 `−222,856,829`，3 gain / 4 harm / 38 tie。设计没有授权 population significance test，所以这些差值只作 descriptive evidence。

这组结果排除了两个更强解释。第一，P6 不能被描述为现实域稳定优于 HippoRAG：总和优势由 Robotics 主导，并伴随 Biology 明显负效应和不利的 gain/harm 计数。第二，P6 也没有克服最简单的 RAW baseline；因此不能把 graph/typed recipe 的 aggregate 小幅收益写成总体 retrieval advantage。相反，E1 在 A_hold 与 fresh reserve counterfactual 上均更差，说明 non-promotion 是正确决策，但这只证明 evaluator rejection，不是 evaluator co-evolution。

当前仍缺两项决定性证据：其一，在新的现实 cohort 上，实质不同的 action 同时对 RAW 与 HippoRAG 产生跨 relation family 的稳定正净收益；其二，一个 evaluator challenger 在独立 A_hold 上真实晋升，并在事前冻结且 untouched 的 M_search 上改善 retained search。BRIGHT 已评分 45 项不得用于继续调 P6/E1、补关键词或改 promotion gate；余下数据也不能把同源 post-hoc 调参包装成独立 family-out。若继续，下一机制必须从新 TRAIN distribution 形成能扩展 baseline candidate set 的 action，并一次性冻结 A/F/A_hold/M，而不是继续修改 gate。

### 12.29 2026-07-20 BRIGHT P9：前瞻 aggregate 仍为正，但 RAW 跨 family 稳定性判据失败

本轮没有在上一节已评分的 45 项上继续改 P6/E1 或 promotion gate，而是只把它们作为**已消费的候选形成数据**。冻结的 P9=`P9_RELATION_MECHANISM_CE_RAW_HIPPORAG_RRF` 以 relation/mechanism 两类 query 对完整 32-document pool 做本地 cross-encoder 排序，再把 `cross_encoder_full32`、`RAW_top10` 与 `HippoRAG_top10` 用等权、exact-Fraction RRF（`k=60`）融合并截取 top-10。cross-encoder 资产、deterministic runtime 与 implementation freeze 均先固定；该形成过程没有读取剩余 RESERVE 内容或 label，也没有外部网络调用。

在已消费 TRAIN45 上，P9 的 sum integer nDCG@10 为 `8,245,721,318`（mean `0.183238`），高于 HippoRAG 的 `6,119,227,934`（`0.135983`）和 RAW 的 `6,693,156,962`（`0.148737`）。P9−HippoRAG 的三-family delta 为 `+651,825,052 / +929,763,162 / +544,905,170`，12 gain / 4 harm / 29 tie；P9−RAW 为 `+1,144,742,346 / +60,733,489 / +347,088,521`，11/2/32。单独 cross-encoder relation/mechanism arm 的 mean 为 `0.202240`。这些数字是在查看已消费 TRAIN45 outcome 后从多个候选中形成 P9，只是 historically informed formation evidence，`prospective_confirmation_claim=false`，不能充当独立性能结论。

真正的前瞻检验在任何新 row/label 打开前预注册：从原 HMAC RESERVE 顺序中，对每个 family 取紧接既有前 15 项的 rank 15–25（0-based），即 Biology、Economics、Robotics 各 11 项、共 33 项；另外 4 项保持 untouched。设计与 implementation freeze 都在 acquisition 前提交，且固定只有 P9 一个 candidate、没有 fallback、weight search 或失败后替换。prepare 得到 33/33 valid Qwen generation，并在 join 前持久化 66 个 external action intents。action 以一个 cross-encoder worker 加 12 个 HippoRAG workers 的最大 13 进程并发、单 launch 完成；33/33 HippoRAG terminal，最小 graph 40 nodes / 22 edges。所有 action seal 后才一次打开 labels 并离线评分，external network、online evaluator 与 retry/replay/resample 均为 0。

前瞻 `C_confirm` 的五臂 aggregate 为：

| arm | sum integer nDCG@10 | mean nDCG@10 | Biology / Economics / Robotics family sums |
|---|---:|---:|---:|
| P9 | 4,071,558,132 | 0.123381 | 1,473,774,717 / 1,000,000,000 / 1,597,783,415 |
| CrossEncoder_RM | 3,852,937,636 | 0.116756 | 702,054,217 / 1,000,000,000 / 2,150,883,419 |
| candidate-restricted HippoRAG core | 3,041,893,662 | 0.092179 | 1,433,994,531 / 630,929,754 / 976,969,377 |
| RAW | 3,772,133,575 | 0.114307 | 1,546,507,088 / 1,000,000,000 / 1,225,626,487 |
| retained P6 | 2,409,166,381 | 0.073005 | 788,482,861 / 1,000,000,000 / 620,683,520 |

P9−HippoRAG aggregate 为 `+1,029,664,470`，三-family 为 `+39,780,186 / +369,070,246 / +620,814,038`，逐题 7 gain / 1 harm / 25 tie；因此在这份同源前瞻 cohort 上，“每个 family 均超过 candidate-restricted HippoRAG”得到描述性支持。P9−RAW aggregate 也为正（`+299,424,557`），但三-family 为 `−72,732,371 / 0 / +372,156,928`，逐题 6/1/26。预注册 primary 要求 P9 对 RAW、HippoRAG 的 aggregate 和每个 family 都严格为正，因此 `primary_passed=false`；不能用 overall mean 抹去 Biology 的负差和 Economics 的零差。

这项结果把断点定位得更精确：当前主要障碍不再是是否能在三个 family 都高于这份 HippoRAG core，而是是否能对最简单 RAW baseline 也保持跨 family 稳定收益。与此同时，P9 本身直接融合 RAW top-10 与 HippoRAG top-10，并增加完整 32-document cross-encoder 计算，所以 P9−HippoRAG 只能解释为 fixed ensemble 的增量价值，不能解释为等算力、full-corpus BRIGHT、SOTA 或 Agent 普遍优越性。

本 source epoch 到此停止：不在 TRAIN45 或 `C_confirm` 上改 query、RRF 权重、top-k、family rule 或 gate；剩余 4 项数量太小且 family 不平衡，只保留为审计余量，不作为 rescue cohort。下一条非 gate 主线必须换成能在**新来源/新领域扩展或重写候选集**的 action，例如新检索入口、typed multi-hop evidence synthesis 或跨文档结构生成，而不是继续在同一 32 项上重排。它应从新的 TRAIN distribution 一次形成并冻结，再进入独立 measurement。L5 仍是另一条未闭合链：需要实质不同的 evaluator challenger 在独立 A_hold 上真正晋升，并改善事前冻结且 untouched 的 M_search。若不开展这两项，诚实终态就是窄 L3/L4 positive、现实域无稳定三臂优势、L5 未达到。

### 12.30 2026-07-20～21 FiQA P10/P11：TRAIN 形成了更强 recipe，但 DEV comparator 在评分前失效

为避免继续在 BRIGHT 已评分 cohort 上补规则，后续路线转向 BEIR/FiQA 的独立 TRAIN distribution，并把候选形成与前瞻验证严格分开。P10 的离线 TRAIN 端到端结果覆盖 12 项：P10 mean nDCG@10=`0.297878`，高于 candidate-restricted HippoRAG 的 `0.182360`，但低于 RAW 的 `0.421876`；P10−HippoRAG 为 5 gain / 2 harm / 5 tie，而 P10−RAW 仅 1/3/8。它还生成 1,729 个 base pool 外的唯一 bridge candidates，但没有找回任何“原 base pool 缺失且被 P10 top-10 找回”的 gold document。因此，typed bridge 确实改变了 action space，却尚未证明扩展候选带来标签收益。

在这 12 个已消费 TRAIN item 上，固定候选族的离线 formation 选择 `P11_RAW1_CE_SUM2_EXPANDED_RRF_K60`。其 mean nDCG@10=`0.473354`，相对 RAW 为 3 gain / 0 harm / 9 tie，整数 nDCG 净增 `617,741,586`；相对 HippoRAG 为 8/0/4，净增 `3,491,935,267`，且 leave-one-out 对 RAW 的最小净值仍为 `+235,061,267`。这只是看过 TRAIN outcome 后的 candidate formation，不是 prospective claim。

随后冻结的 FiQA DEV `C_confirm` 在 48/48 Qwen generation、48/48 cross-encoder output 和 48 个 action intents 完成后，official HippoRAG 于 item 2 触发 phrase-weight assertion；当时只有 11 个 comparator terminal outputs，尚无 action seal、label pack 打开或 performance score。该 DEV cohort 因而严格记为 infrastructure-invalid 并永久不 replay，TEST 也未打开。这个结果既不能证明 P11 有效，也不能把 comparator failure 当作 P11 失败；它只证明未加固的 official comparator 不是 complete-case evaluator。

### 12.31 2026-07-21 NanoBEIR P11：新域 acquisition 完成，但 typed generator 不是 total function

在对 official HippoRAG 的已知 nonfinite/missing-phrase failure 做独立、非评分 hardening qualification 后，P11 被原样冻结到三个新 NanoBEIR family：NanoClimateFEVER、NanoDBPedia、NanoHotpotQA。三族均有 50 个 source-valid query；fresh HMAC 在任何 action、model 或 score 前分配 138 个 private items，其中 `C_confirm` 每族 12 项、共 36 项，A/F/A_hold/M 与 reserve 同时密封。

正式 `C_confirm` 没有到达 retrieval。36 个 label-free Qwen completion 中 34 个满足 grammar，NanoDBPedia 的 2 个 completion 无效；运行在 corpus embedding、bridge retrieval、cross-encoder、HippoRAG、action seal 和 label access 前 fail-closed。公开 failure receipt 明确记录 action intent=0、action seal=0、label/score=0，当前 cohort 与同源 P11 均禁止 replay 或 prompt/grammar 搜索。这里暴露的是架构缺口而非性能负结果：若 typed operator 仍依赖自由生成 JSON，那么“不允许的 primitive 不可表达”并未真正成立；generator 必须被 totalize，且 totalization 本身要在新 cohort 前冻结。

### 12.32 2026-07-21 NanoBEIR P12：generator 已 totalize，但 comparator complete-case 仍是必要条件

P12 将 P11 的 source-valid generation 保留，对无效 completion 使用事前固定的本地 totalizer，并在 fresh source/cohort 上重新开始。第一轮 `C_confirm` 的 36 项均形成 action intent：35 项使用 source-valid Qwen，1 项使用 frozen totalization；36/36 cross-encoder 均完成。然而 official HippoRAG 在 NanoNQ item 15 上再次触发 top-k/nonzero-cardinality assertion。此时 35/36 comparator outputs 已完成，但三臂 action 尚未 seal，labels 与 scores 均未打开，因此该 cohort 仍是 comparator-runtime-invalid，而不是 P12 efficacy evidence。

这次 failure 把需求从“再补一个异常分支”收敛为可验证的 complete-case 合同：先对全部 source-valid query 做一次 label-free、single-launch availability screen，只有 terminal、返回 10 个唯一 ordinal 且 graph counts 为正的 query 才进入后续 HMAC population；未来选中的 HippoRAG bytes 必须原样复用，不能在评分 block 再次启动 comparator。该设计不是为 P12 提高分数的 gate，而是防止任一 comparator crash 在 label 打开前摧毁整组 paired comparison。

complete-case fresh source 的首版 NanoSCIDOCS 含 344 个空正文，违反冻结的 nonempty-document contract；在 selection/action/score 前按预注册只允许一次的 compatibility replacement 换为 NanoSciFact。最终 NanoArguAna、NanoFEVER、NanoSciFact 各 50 项的 availability screen 达到 150/150 terminal，零 failure；随后 corrected one-shot acquisition 形成每族 36 个 selected items、共 99 个 runtime items与 14 个 private packs。acquisition v1 曾因错误要求 RAW top-10 等于 ordinal-sorted base pool 的前十项而在 secret 生成前失败；v2 只修正为“ranked unique subset”语义，没有读取 query assignment、label、action 或 score。

但 P12 complete-case `C_confirm` 仍在 action 前终止。30 项中有 2 个 NanoArguAna source generation 无效并触发 totalizer；P12 fallback 的 typed query cap 为 900，而 bridge composer cap 为 768，附加在末尾的 distinct anchor 被整体截断，导致 bridge query 空或重复。没有 action intent、action seal、Hippo reuse、label 或 score。由于 cap=900 属于已冻结 candidate，而不是单纯 runtime bug，当前 P12 cohort 不得修补或 replay。由此形成的下一候选 P13 只做一个候选级变化：把 relation/mechanism 投影到带 role prefix 的 671 字符上限，使 `671 + 1 + 96 = 768`，从生成空间排除 anchor truncation；它必须在全新 source/cohort 上验证，不能继承 P12 的 performance opportunity。

### 12.33 2026-07-21 NanoBEIR P13：bridge 已可执行，但 availability 与 runtime 的 base-pool 合同不一致

P13 在任何新 outcome 打开前，把 P12 totalizer 输出再投影为 role-prefixed、最多 671 字符的 relation/mechanism query。对上一轮失败的 30 个 label-free item 做 formation replay 时，30/30 local plan 可执行，118 个 bridge query 的最大长度为 699，证明 anchor truncation 已从生成空间消失。随后候选被单独冻结，并换到 NanoFiQA2018、NanoNFCorpus、NanoTouche2020 三个 fresh family。NanoFiQA2018 原始 corpus 有 27 个空正文且没有 qrel 指向它们；唯一一次、跨三臂一致的 source projection 在 selection/action/score 前排除这 27 行，其余 family 不变。

single-launch availability screen 对 149 个 source-valid query 运行 candidate-restricted official HippoRAG：NanoFiQA2018 为 36 terminal / 14 failure，NanoNFCorpus 为 40/10，NanoTouche2020 为 40/9，三族均达到事前最低 36。该 screen 没有读取 qrel/label、没有 selection secret、candidate action 或 score。随后 fresh HMAC 一次性形成每族 C10/A8/F5/A_hold5/M5/reserve3，共 99 个 runtime items和 14 个 private packs；选中的 HippoRAG bytes 本应在正式 block 原样复用。

正式 P13 `C_confirm` 在 cached comparator join 处 fail-closed。30 项的 Qwen/totalizer、corpus/typed/bridge embeddings、cross-encoder output 与 action intents 已完成：29 项是 source-valid generation，1 项使用 frozen totalizer。但 label-free audit 证明 availability 与 runtime 计算的不是同一个候选池：screen 以**原问题单向量**构造 MiniLM top-32；runtime 的 frozen Agent pipeline 则以 RAW top-10 为 seed，再从原问题与四个 typed query 的全局 ranking 做 RRF fill 到 32。30/30 ordered set 与 unordered set 都不相等，交集范围 15–31、均值 23.733；虽然 cached RAW top-10 在 30/30 runtime pool 中仍是子集，cached HippoRAG 结果却不能映射到新 pool。于是 0 个 reuse receipt、0 action seal、0 label、0 score，A/F/A_hold/M 均未打开。

这不是 P13 的性能负结果，也不能靠把 cache-check 放宽来恢复：若忽略 pool mismatch，HippoRAG ordinal 将指向不同文档，三臂比较会直接失真。当前 P13 同源 cohort 因此永久终止。下一次测量不再添加一个与 candidate 脱节的 availability gate，而应在 fresh family 上先冻结 HMAC query order，对每项先形成 candidate-specific typed pool，再在该同一 pool 上单次运行 HippoRAG；按事前顺序取得每族固定数量 terminal complete cases 后一次 seal 并离线评分。这样 comparator availability 是直接 measurement 的缺失数据规则，而不是可反复修改的前置 gate。

### 12.34 2026-07-21～22 BRIGHT P14→P17：all-remote action 已 seal，但并发契约偏差使 labels 保持未打开

P13 后续严格改成 candidate-specific same-pool complete-case measurement，没有再增加 availability gate。P14 先从
Earth Science、Psychology、Sustainable Living 建立 source custody 与 HMAC block；本地 direct run 已形成 Qwen、
cross-encoder 与 action intents，并得到 12 个 HippoRAG terminal，但 12 路 CPU comparator 使本机不可用而被用户
中止。其 complete-case selection 与 three-arm action seal 均未形成，direct label pack open count 和 score count 都是
0；P14 因而以 infrastructure-invalid / efficacy-unknown 关闭且不 replay。

P15 把全部 action-producing computation 迁到 gpu1，并另外取得 HMAC positions `[72,92)` 的 20 attempts/family
view-only extension；它最后没有形成 `remote_action.result.json`，也没有任何 action/score 被 P17 复用。P16 再迁到
有线 311linux，预先冻结 8 个 HippoRAG CPU processes，但原计划仍要求从 position 92 起每族再取 20 项。正式
acquisition 在 HMAC ordering 与 selected-item materialization 前发现 source capacity 只有 `116/101/108`，因此以
design-invalid 终止；HMAC selection、remote action、gold 与 score 均为 0。这三次终态都不增加 P13 efficacy sample。

P17 使用新的 study ID，把 feasible window 事前固定为 positions `[92,101)`：每族 9 attempts，并按顺序选择前 8 个
terminal complete cases。corrected runtime、implementation freeze 与 execution plan 共同规定 HippoRAG CPU process
concurrency=8、OMP threads/process=2、cross-encoder 独占物理 GPU 1、MiniLM/Qwen 使用物理 GPU 0、外网/在线 evaluator/
retry/replay/resample=0。唯一 formal action 在 311linux 完成：26 个 Qwen output source-valid、1 个由冻结 totalizer
补全；27/27 HippoRAG terminal，三族各 9 个；selection 与 24-item three-arm action 均 canonical seal；远端 action
result 的 `external_network_call_count=0`、prior P14/P15 output reuse=0。minimal archive 三文件及排除 top-level
`corpus_tensors/hf/home/tmp` 后的 2,972-file forensic tree 均已回传，后者 tree SHA-256 为
`952b8794…7f92`。

但同一远端回执也在 gold 打开前给出了阻断性事实：`HippoRAG_peak_process_concurrency=9`，超过 study design、
runtime fingerprint 与 plan 冻结的 8。冻结 runner 虽定义 `CONCURRENCY=8`，却用
`ThreadPoolExecutor(max_workers=CONCURRENCY + 1)` 同池提交一个 cross future 与全部 27 个 Hippo futures；
cross-encoder 很快结束后，第九个 slot 随即启动额外 HippoRAG process。该 counter 包围真实子进程生命周期，所以 9
不是日志估计或线程数误读。现有 unit test 只断言常量等于 8，finalizer 又只校验 host/network/reuse，没有校验 observed
peak；若继续调用它会错误越过冻结 execution contract 并首次读取 gold。因此正式 offline performance finalizer 没有
调用，gold ID column read、selected label score 与 performance score 全部保持 0，primary 未评价，P17 efficacy
保持 unknown。

另有一个不参与执行但必须透明保留的 receipt 问题：P17 acquisition result 在每族只有 9 attempts 时仍沿用
`target_terminal_count_per_family=10` 字面量；authoritative study design、plan、selection 与 finalizer 全部明确为 8。
该 self-hashed receipt 不回写、不伪装修正；它不是本次 invalidity 的主因，也没有改变 8/family 的实际 selection。
终态 manifest `90c9490e…7df5` 把两项异常、三份 sealed action hashes、forensic tree 与 label/score=0 一并绑定，并使
one-shot result path 关闭。P17 不得 retry、resample、改 candidate、放宽并发口径或补 gate；24 个未评分 action 也
不得被描述成 Agent 对 RAW/HippoRAG 的性能结果。

因此本轮没有把 P9 的断点向 efficacy 方向推进。P9 仍是该现实域分支最后一份有效评分结果：它对
candidate-restricted HippoRAG 的三 family net 都为正，但未在 Biology/Economics 严格高于 RAW。现实域双 baseline
稳定三臂优势与 L5 evaluator→A_hold promotion→untouched M_search 因果链都仍未闭合。若不另立完全独立、事前修正
runtime 的新 study，当前总目标应在这一边界收束，而不是继续启动 P18 或增加 gate。

### 12.35 2026-07-22～23 TAT-QA P18：源外 qualification 在 runtime inventory 终止

P18 是在 P17 之后另立的独立 TAT-QA study。它在任何 official TAT-QA 文件下载、row parse、selection secret、
formal item identity、模型 action 或评分之前，先冻结 typed multi-hop P0/P1、A_form/F_search/A_hold/M_search、
E0/E1 promotion、RAW 与 project-attested HippoRAG 三臂合同。实现提交 `39e0a80c…82` 经 171 项源外测试与
独立 adversarial audit；新增的 worker 证据包括真实子模型 monotonic interval、完整 transport receipt、named
systemd unit、HippoRAG `TasksMax=3`（monitor process 预留 1、worker process 最多 2 threads）以及异常时对全部
unit 的同步 stop/kill/reap/finalize。这里没有通过补 gate 改变候选或 performance 判据。

唯一一次 source-free production qualification 在有线 311linux 启动后约 2 秒，于 `runtime_inventory` fail-closed。
terminal failure self-hash 为 `008f6a11…fae5`；`formal_source_opened=false`，external network、online evaluator、
retry/replay/resample 均为 0。失败发生在 fingerprint 和 public model canary 之前：P17 迁移后的 lexical venv 声明
`include-system-site-packages=true`，但代码按 `home/../lib/python3.10/site-packages` 查找 base root；该目录不存在，
实际活动依赖来自另一个 `lib/python3/dist-packages`/`.pth` topology。因此 P18 按预注册记为
runtime/implementation-invalid、efficacy unknown，qualification root 已烧毁，不重试，也不打开 TAT-QA source。

只读追查还暴露出比路径拼接更深的 capability conflation：P18 把 Qwen/MiniLM 与 HippoRAG 共用一个
`runtime_python`。冻结的 Qasper MiniLM manifest 要求 Python 3.10.12、torch 2.8.0、sentence-transformers 5.5.1；
冻结的 HippoRAG attestation v3 则绑定 Python 3.11.15、base torch 2.5.1、sentence-transformers 5.4.1 与另一份
pyvenv/topology。单一解释器不可能同时满足两套 exact runtime identity。下一 study 不能把同一 P18 root 换路径再跑；
必须使用新 study ID/root，并把 `typed_plan/MiniLM runtime` 与 `HippoRAG runtime` 拆成两个独立、各自 fingerprinted
capability。这个修正属于源外执行语义，不允许顺带修改 P0/P1、evaluator、promotion、cohort 或 gate；公开 synthetic
qualification 通过并提交后，才可首次取得 formal source。

### 12.36 2026-07-23 TAT-QA P19：双 runtime 已冻结，但外层 launch envelope 使 qualification 终止

P19 没有重放 P18。它使用新 study ID/root，把 typed-plan/Qwen 与 exact Qasper MiniLM 固定到一个 Python 3.10.12、
torch 2.8.0+cu128、sentence-transformers 5.5.1 的 lexical venv；official HippoRAG 则继续使用另一 lexical venv，
并由新的 P19 attestation 被动复验 executable、`pyvenv.cfg`、`.pth`、19 项 distribution metadata、active module origin
以及 HippoRAG/SmolLM/MiniLM 三组资产。外层 composite fingerprint 内含两枚 canonical self-hashed subfingerprint，
public canary 也显式交叉绑定两枚 nested self-hash；没有把 runtime 拆分伪装成两个 gate。P0/P1、cohort、E0/E1、
A_hold promotion、M_search 与三臂 performance 判据均沿用事前冻结合同。

源外实现提交 `39215941…120` 经 188/188 测试、`py_compile`、diff check 与 adversarial audit。311linux 上新的 typed
runtime 已验证七项 exact distribution version 与物理 GPU 1 可见；P19 archive 与本地提交逐文件 SHA-256 一致。
official TAT-QA source、selection secret、formal item、模型 action 和 label 在此时仍全部不存在。

唯一一次 qualification 仍在约 2 秒内、模型 canary 与 fingerprint 写出前，于 `systemd_network_preflight` fail-closed。
terminal self-hash 为 `2e4dbf0b…9a21`。原因不是双 runtime 不兼容，而是外层 transient service 使用
`/usr/bin/env -i` 时只保留 HOME/LANG/PATH 等变量，同时删除了 user-systemd client 所需的 `XDG_RUNTIME_DIR` 与
`DBUS_SESSION_BUS_ADDRESS`；冻结代码的 nested `systemd-run --user` 因而无法连接现存 user bus。只读证据同时证明
`/run/user/1001/bus` 与 `/run/user/1001/systemd/private` 存在，而该 unit 的 `ExecStart` 不含两项变量。失败回执记录
`formal_source_opened=false`、external network=0、online evaluator=0；没有 fingerprint、public canary 或模型执行。

P19 root 已烧毁，不能补变量后原地重试。该终态应记为 launch-infrastructure-invalid、efficacy unknown。若继续，只能
建立新 study ID/root，并在源外预注册中把 safe user-bus launch envelope 当作执行能力显式绑定；候选、cohort、metric、
promotion 与 gate 不得改动，也不得把 P19 计入 efficacy sample。修复属于 host launcher contract，不是增加新 gate。

### 12.37 2026-07-23 TAT-QA P20：entry launch 已修复，但 post-inventory 环境断言使 qualification 终止

P20 是 P19 之后的新 study，而不是在 P19 root 上补变量重跑。提交 `a0c9e5ab…862d` 保持 P19 的 typed P0/P1、四个
cohort、E0/E1 promotion、RAW/official HippoRAG 三臂指标与全部 efficacy gate 不变，只新增事前绑定的 safe
user-systemd launch capability。外层 `/usr/bin/env -i` 精确保留 12 个变量，其中包括
`XDG_RUNTIME_DIR=/run/user/1001` 与 `DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1001/bus`；receipt 只记录变量名、
路径/地址哈希和 socket 类型/ownership boolean，不读取或保存任何 provider/API credential。实现同时把两个真实
runtime 的 nested self-hash 交叉绑定到 composite fingerprint、canary、qualification 与 freeze。

源外实现经 206/206 tests、`py_compile`、diff check 和双人 adversarial audit。审计在提交前修复了两个真实缺口：
implementation registry 补绑四个实际 import 的 package/model 文件，worker transient unit 从残留的 `tatqa-p18-*`
改为独立 `tatqa-p20-*` namespace。311linux 上从该提交导出的 37 个 P20 文件与本地 Git object 逐文件 SHA-256 一致；
official TAT-QA source、selection secret、formal item、action、label 与评分仍未创建。

唯一一次 P20 qualification 的 entry capability 已通过，但在约 2 秒后仍于 `systemd_network_preflight` fail-closed。
marker 文件 SHA-256 为 `7eb226c8…1585`，terminal failure 文件 SHA-256 为 `005b7607…2d19`、self-hash 为
`3a7257e3…00c`；回执明确 `formal_source_opened=false`、external network=0、online evaluator=0。只读 journal 证明
底层异常是 `outer launch environment variable allowlist drifted`。随后在相同提交、相同 12 项 entry environment、
相同 user-systemd 网络隔离下做的非 qualification 源外诊断给出唯一变化：`runtime_inventory_snapshot()` 导入并初始化
CUDA-capable torch 后自动新增 `CUDA_MODULE_LOADING=LAZY`，环境名从 12 变为 13，未删除或修改其余 entry 名；因此
后续 `_launcher_environment()` 把可信 runtime 的确定性内部 mutation 误判为外层凭据注入。P19 的 user-bus 缺失已真实
修复，本次是 P20 post-import validation implementation-invalid，不是模型、双 runtime 或 efficacy 失败。

P20 root 已烧毁，不允许删回执、重试 qualification 或把该 run 计入 performance sample。若继续，下一独立 study 只能
把“entry 时 exact 12-name allowlist”与“torch inventory 后唯一允许的 exact safe mutation
`CUDA_MODULE_LOADING=LAZY`”分成两个明确 phase；nested worker 仍只复制固定 safe subset，候选、cohort、metric、
promotion 与 gate 不得变化。在新的 source-free qualification 成功并提交前，仍不得下载或打开 official TAT-QA。

### 12.38 2026-07-23 TAT-QA P21：三阶段 launcher 与 split runtime 已通过，硬件相关 MiniLM byte canary 终止 qualification

P21 使用新 study ID/root，提交 `4a8550fb…911c`；它没有重放 P20，也没有修改 typed P0/P1、144-item cohort、
E0/E1 promotion、三臂指标或 efficacy gate。唯一变更是把 launcher 证据拆成三个闭合 phase：entry 精确 12 个变量，
post-inventory 只允许新增 `CUDA_MODULE_LOADING=LAZY`，post-MiniLM 再只允许
`CUDA_VISIBLE_DEVICES: "1" -> ""`；nested systemd client 始终只复制 DBUS、HOME、LANG、PATH、XDG 五项。
fingerprint 绑定前两个 live receipt，canary/terminal 预注册绑定第三个。实现经 209/209 tests、`py_compile`、builder
idempotence、diff check 与独立 adversarial audit；37 个提交文件在 311linux 与本地 Git object 逐文件 SHA-256 一致。

唯一 qualification 首次越过 P18/P19/P20 的三个前置故障：safe user-bus entry、split runtime inventory、
post-inventory exact environment、nested `systemd_network_preflight`、五组 asset tree 与两枚 nested runtime
subfingerprint 均成功，产生 composite fingerprint 文件 SHA-256 `efd48481…d498`、self-hash `dff7f478…d323`。
这说明 P19 的 dual-runtime 设计与 P21 的 launcher 修复都已得到实际 source-free 证据，而不是只通过 unit test。

终止发生在 `bound_minilm_initialization`，即 Qwen/Hippo public canary 和 official TAT-QA source 之前。marker 文件
SHA-256 为 `e3cb3d7f…fcaf`；terminal failure 文件 SHA-256 为 `b6db424a…e076`、self-hash 为
`d5393a48…9cf2`，并记录 `formal_source_opened=false`、external network=0、online evaluator=0。底层错误是冻结的
Qasper MiniLM 256-sentence startup canary 要求完整 float32 bytes hash `e76f373b…c5746` 与每维乘 `10^6` 后的
integer-matrix hash `f24c3299…ee1b`；311linux 在相同 Python 3.10.12、torch 2.8.0+cu128、
sentence-transformers 5.5.1、模型 tree 与 CPU/float32/eval 设置下得到 `6b0a0498…509f` 和
`86a1a981…16f`，但同机重复逐元素完全相同。进一步 source-free forensic 将 torch CPU dispatch 强制为
`default` 后又得到第三组稳定 hash `9510f133…f395` / `19065a6d…5435`，而 manifest 没有绑定 CPU ISA/dispatch。
因此 exact byte/1e-6 component hash 对 CPU dispatch 敏感，不能作为跨 AMD/Intel 主机的 portable runtime 判据；
这是 canary contract-invalid，而不是模型资产漂移或 semantic/effect failure。

P21 root 同样终止，不允许用 311 的实测 hash 补成“通过 profile”后原地重试。下一步不再直接新立 study 盲跑；应先在
完全公开 synthetic 输入上做一次非评分、非 qualification 的全链 feasibility：模型 tree/runtime identity 仍 exact，
MiniLM canary 改为同机 repeat-exact、shape/finite/L2-normalization 与无正式 row 访问，不再要求跨 CPU 的完整 embedding
bytes；同一次 feasibility 必须实际走通 Qwen typed-plan、MiniLM 与 official HippoRAG 三种 capability。只有该公开源外
全链一次通过，才值得冻结新 study；这属于删除错误硬件假设，不是新增 performance gate。

### 12.39 2026-07-23 TAT-QA P22：portable MiniLM 已越过硬件假设，但源外 feasibility 在 post-model 环境合同终止

P22 不是正式 study，也不替代 P21 qualification；它是事前限定的一次 source-free、nonqualification、non-efficacy
全链 feasibility。实现提交为 `b18656d0…facf`，从该提交导出的最小 51-file Git snapshot commit 为
`8cb0ffd9…774a`。snapshot 同时绑定真实 Git object/clean worktree、完整文件系统 closure、18 个已加载项目模块、四个
空 source-isolation sentinel root，以及 311linux 上本地解包的 Git 2.43.0 executable SHA-256
`2a8c18fb…d668`。独立 adversarial re-audit 通过；P22 focused tests 34/34、继承的 P21 runtime/public-canary/MiniLM
tests 50/50 通过。official TAT-QA source 在整个阶段仍未下载、未打开。

唯一固定 unit `p22-source-free-feasibility-c1-v1.service`、固定 root
`/home/erzhu419/p22_source_free_feasibility_20260723/attempt` 依次越过 entry、split runtime inventory、
post-inventory launcher、network-denied nested systemd preflight、P21 composite fingerprint 复验、portable MiniLM exact
asset/runtime identity 与公开 256-sentence structural startup canary。后者在 311linux 上满足同机两次 byte/elementwise
exact、shape/dtype/finite、向量非退化与 L2 norm 误差上限，且不再把 CPU-dependent embedding bytes 当作 normative
acceptance；因此 P21 的跨 CPU hash contract 缺陷已被真实消除，而不只是单元测试中的假设。

随后 feasibility 在 `p21_post_minilm_launch_envelope` fail-closed，尚未启动 Qwen typed-plan 或 official HippoRAG
public canary。marker 文件 SHA-256 为 `1bd11b8c…5940`、self-hash 为 `c991995f…f19c`；terminal failure 文件
SHA-256 为 `4101aeda…e239`、self-hash 为 `f810a0e9…990b`。两份回执均绑定
`formal_TAT_QA_source_opened=false`、external network=0、online evaluator=0、retry/replay/resample/provider switch=0。
P22 attempt 已永久消耗，不得重试。

在相同 network-denied/source-free 条件下进行的一次非 qualification、非 canary postmortem 只比较变量名与预期安全值，
没有 encode、Qwen、HippoRAG、正式 source 或评分。它把变化精确定位为：inventory 后新增
`CUDA_MODULE_LOADING=LAZY`；MiniLM model 初始化后又新增 `KMP_DUPLICATE_LIB_OK=True`、
`KMP_INIT_AT_FORK=FALSE`，并发生原已预注册的 `CUDA_VISIBLE_DEVICES: "1" -> ""`。所以 P22 终态是继承的
post-model launch-envelope implementation-invalid，不是 portable MiniLM semantic failure，更不是 efficacy 结果。

若继续，只允许另立一次独立编号、固定 root 的最终 source-free feasibility：事前只接受上述两个确定性 KMP mutation，
在形成自哈希 receipt 后把它们从 nested-launch 环境规范化掉，再复验原 P21 post-MiniLM 合同，并实际走通 Qwen、
portable MiniLM、official HippoRAG 与 worker closure。候选、cohort、metric、promotion 和 performance gate 全部不变；
这是一项 launcher implementation correction，而不是继续添加性能 gate。该独立 feasibility 若仍失败，这条 TAT-QA
可行性路线即终止，不再按新异常逐项扩 allowlist。只有它一次通过并提交，才可下载 official TAT-QA 并冻结真正的新
study。

### 12.40 2026-07-23 TAT-QA P23：环境修复通过，但 public typed-action feasibility 为阴性；该路线关闭

P23 严格执行 12.39 预先限定的最后一次 source-free feasibility，而没有重启 P22。主提交
`6a946a29…70b`，最小 55-file snapshot commit `9af74536…d39`；snapshot 把 P23 canonical entry module、normalizer
callable/object/file、六个空 source sentinel root、完整 Git/文件系统/import closure 与固定 Git 2.43.0 binary 一并
绑定。独立 adversarial audit 为 PASS；P22/P23/portable focused tests 42/42、继承的 P21 runtime/MiniLM tests
77/77、`py_compile` 与 diff check 均通过。固定 unit 使用 canonical-import `python -c` 入口，部署前的远端只读 binding
preflight 得到 55/55 files、19 loaded modules、clean worktree；此时 attempt root 仍不存在。

唯一 unit `p23-source-free-feasibility-c1-v1.service` 随后创建固定 attempt。P23 实际越过 P22 的终止点：portable
MiniLM structural canary 完成；`KMP_DUPLICATE_LIB_OK=True` 与 `KMP_INIT_AT_FORK=FALSE` 先按完整变量名集合和精确值
验证、形成自哈希 receipt，再从 nested-launch 环境删除；原 P21 post-MiniLM phase 复验也通过。因此 P22 的
post-model environment invalidity 已被真实修正，而不是继续放宽到任意变量。

终止发生在下一阶段 `p21_public_synthetic_production_path`。真实 Qwen worker 已启动一次并返回 schema-valid typed plan：
`operation=DIFFERENCE`、`relation_query="one nonempty retrieval"`；input/output 文件 SHA-256 分别为
`ec4837a7…d0d` 与 `a3a29f06…862`。该 plan 与 portable MiniLM 在冻结 public fixture 上编译出的 P1 没有任何
P0 之外的 typed residual unit，因而按事前 canary 合同抛出 `P1 introduced no typed residual unit`。这发生在第一
repeat 内，official HippoRAG canary 尚未启动，不能把 P23 写成三种 capability 全链通过。

marker 文件 SHA-256 为 `61393655…860`、self-hash `6bf2ba48…618`；terminal failure 文件 SHA-256 为
`03332b63…460`、self-hash `404cf98f…961`；终态仍为 `formal_TAT_QA_source_opened=false`、external network=0、
online evaluator=0、retry/replay/resample/provider switch=0。源外 postmortem self-hash 为
`cf842b08…da1`，只绑定冻结 control flow、unit journal 与 public synthetic input/output，不运行第二次模型。

该结果不是 TAT-QA efficacy negative，也不是新的 launcher 故障：Qwen transport/schema、portable MiniLM 与环境修复
均已执行；失败的是预注册的集成行为条件。按 12.39 的明确终止规则，P23 不重启、不换 fixture、不改 prompt、不把
`residual unit` 条件删除后重跑，也不再建立 P24 TAT-QA formal study。official TAT-QA source 至此始终未下载、未打开。
若仍追总目标，只能换完全独立的现实 benchmark/source、study ID、candidate/evaluator mechanism 与 cohort；新研究不得
继承这条已失败的 P21 public canary，也不得把 P23 的 partial capability evidence 当作性能样本。

### 12.41 2026-07-23 FRAMES P1：独立现实域 study 在正式持久化下载与资格前冻结

TAT-QA P23 关闭后，下一条路线没有沿用其 fixture、candidate、cohort、study ID 或 gate，而是选择官方
`google/frames-benchmark` 作为完全独立的现实多跳 source。官方 dataset card 声明 Apache-2.0、824 个 test rows、
每题 2～15 个相关 Wikipedia articles，以及 Multiple constraints、Numerical reasoning、Post processing、Tabular
reasoning、Temporal reasoning 五种原生 reasoning types；source repository 固定在 revision
`58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef` 的 `test.tsv`。这一步只建立 source custody 与容量资格，不是
performance gate，也尚未创建 Agent、RAW、HippoRAG action、evaluator 或 score。repository metadata 又把
`test.tsv` 绑定为 Git blob `cea20270…025`、484,887 bytes。

在 custody freeze 前，公开 dataset viewer 曾显示 formation 区域内的部分 rows；后续只用 `[0,90)` 的已暴露区域
核对列名、`reasoning_types` 的 ` | ` grammar 与 `wiki_links` 的字符串形态，没有把 prompt、answer 或 URL 输出到
资格回执。为保守起见，正式 measurement 永久排除整个 row-id `[0,100)`，它们只能用于 public candidate formation，
不能进入 A_form、F_search、A_hold 或 M_search。BRIGHT P17 在终止后发生的另一次只读全文搜索曾展开
`C_confirm.view.json` 的 query/view，但 label、gold、score 输出与语义读取均为 0；P17 本来已经 terminal，现进一步
明确为不再 question-blind，BRIGHT/P17 不得参与这项或任何后续 candidate/measurement 决策。

这里还发生了一次必须单列的程序性 custody 事件：为取得 repository object digest，命令请求了 Hugging Face `raw`
URL，预期得到 Git pointer，却实际收到 TSV 正文字节流。该流只进入“整串是否为 LFS pointer”的正则并失败，没有
持久化、没有按 TSV/row/cell 解析、没有输出 question/answer/URL/row value，也没有据此改变 candidate、metric、quota
或 parser；formal marker、download receipt、selection secret、model/action/score 都仍为 0。故它不构成在线评价或
outcome access，但使“freeze 前从未传输 source bytes”的更强说法无效；custody 与 implementation freeze 都固定记录
`strict_source_download_before_freeze=false`，后续不重试该访问。

FRAMES P1 的唯一 source qualification 在正式持久化下载和 formal row parse 前固定：只接受上述 exact
revision/path/blob/size；同一 `O_NOFOLLOW` file descriptor 在 parse 前后都复验 SHA-256、Git blob、inode 与 metadata；
只检查 824-row exact header、canonical row-id sequence、事前核过的 cell grammar、reasoning-type aggregate、gold-link
aggregate 与每个预声明 family 是否至少有 `4 blocks × 12 = 48` 个 eligible rows。family precedence 固定为
Temporal→temporal，Numerical/Tabular→structured，
其余→constraint_postprocess；measurement row 只接受 2～5 个去 fragment 后仍为唯一 canonical
`https://en.wikipedia.org/wiki/` pages 的 gold links，并先排除与 formation 100 rows 的规范化 prompt、answer 或 gold-page
碰撞，再按固定 public hash order 证明全局 prompt/answer/page-disjoint 的 48×3 pool 容量。输出只允许 schema、计数、
直方图与长度范围，不允许 question、
answer、URL 或单项 row id；online evaluator/API、model action、score 都必须为 0。失败即关闭 FRAMES source 路线，
不得改 quota、family、parser、candidate 或补 gate 后重跑。只有 aggregate qualification 一次通过，才允许另行事前冻结
私有四块 assignment、Wikipedia snapshot、typed multi-hop candidate、三臂 execution 与 L5 promotion/search 合同。

实际 implementation commit 为 `6552fefb24b7bfc07ec0040e375736704a69abc8`，18/18 focused tests 与独立
adversarial audit 均 PASS。freeze self-hash `8ee6662c…3f20` 绑定该 real Git ancestor 下 qualifier、tests、custody、
BRIGHT disposition 四个 blobs，并用固定 Git 的 no-replace-object 路径复验；formal source 下载后 SHA-256 为
`4255093c…69ff`，Git blob/size 与 repository metadata 精确一致。

唯一 formal qualification 随后写入 one-shot marker，但在 `csv.DictReader.fieldnames == EXPECTED_COLUMNS` 的首行断言
终止：raw repository TSV header 与 freeze 前公开 dataset viewer 所显示的 conversion header 不同。程序没有输出或继续
读取任何 row 的 prompt、answer、URL 或 row id；model/action/score、online evaluator、retry/replay/resample 都为 0。
marker 文件 SHA-256=`e7e5d432…0735`、self-hash=`85f7d9a2…9071`；terminal failure 文件 SHA-256=
`1a67a42b…ff23`、self-hash=`8b9a2aec…ca6b`。按本节事前终止规则，不能打开 raw header 后改 parser，也不能改为
viewer-generated conversion 后重跑；FRAMES P1 source capacity 与 efficacy 都保持 unknown，该 route 永久关闭。

### 12.42 2026-07-23 FanOutQA P1：presemantic hardening 后在 DEV category schema fail-closed

FRAMES P1 关闭后，新路线没有修改其 TSV parser 或另取同源 cohort，而是选择官方 FanOutQA release `v1.1.1`：
commit `ccf127bd0b1e1091e98ffb9aff7dc694eaf58d54`、tree `2b6a01b…fac5`、310-row DEV Git blob
`76ad1feb…a67`。正式 scope 固定为官方 revision cache 上的 derived closed-corpus retrieval，不声称 official full-Wikipedia
open-book 或端到端 answer generation；TEST 永久不授权。candidate 也不再是 RRF、关键词或 gate，而预注册为 top-level question
生成 typed DAG 后执行一对一/依赖条件 page assignment 的四个 recipe；三臂为 E3 Agent、同 corpus RAW MiniLM 与 official
HippoRAG，另有独立 A_hold promotion 与只有晋升后才能打开的 M_search。

官方 DEV 与 `wikicache.tar.gz` 只在 311linux 上并发下载一次。下载字节已经完整落盘后，原 shell 因远端没有 `git`
executable 在 Git-blob 收尾检查处以 127 退出；没有第二次网络请求。现有 `.part` 的 size 与 SHA-256 分别复验为
DEV `1,177,174` bytes / `359300b0…1288`、cache `1,538,812,319` bytes / `62059df8…d5f4`，DEV Git blob
又由同一现有字节流离线算得 `76ad1feb…a67`，随后才原子晋升为固定 source path。下载回执明确记录 retry/redownload=0、
semantic parse=0、TEST=0。

这次下载发生在最终 qualification freeze 之前，因此不能声称 strict pre-byte freeze。原 download authorization 又只允许补
hash/receipt/freeze，而独立审计随后发现 one-shot 可重放、cache SHA 没有硬编码 trust anchor、gzip 尾部零填充未计入上限、
sign-test 的“exact”措辞不成立等问题。没有静默改写原 authorization；新增 self-hash
`566befbc…9db7` 的 prequalification hardening amendment，逐项绑定旧/new design hash，并声明当时 JSON parse、tar-member
parse、model/action/score 均为 0。实现提交 `3662c953b3affc577a41e999b3c3fbb569e902db` 后，source-open marker
改为 fixed-path、O_EXCL、逐级 directory fsync，并绑定 DEV/cache、父 marker、download receipt 与 amendment；cache 在 tar
member audit 前先以 `pread` 对硬编码 SHA 做不透明复验。32/32 synthetic focused tests、`py_compile` 与独立 adversarial
audit 均通过。final freeze self-hash `972d819b…cdc7` 精确绑定 qualifier、tests、custody、design、authorization、receipt
与 amendment；同时如实写明 source 已持久化但从未语义打开，四类 marker/result/failure 文件都不存在。

唯一 formal service `fanoutqa-p1-source-qualification-v1.service` 随后消费 qualification marker 与 source-open marker。
DEV exact SHA/blob/size 和 310-row JSON 顶层成立，但在逐 item schema validation 中发现官方 `categories` 字段形态不符合冻结的
exact list contract，抛出 `FanOutQaP1SourceQualificationError: category schema drifted`。失败发生在 cache tar member audit、
任何 cohort/item materialization、candidate/RAW/HippoRAG action、qrel release、evaluator 或 score 之前；TEST 也未下载或打开。
qualification marker、source-open marker、terminal failure 的文件 SHA-256 分别为 `fb8ca663…e09e`、`fcddb1d0…ca32`、
`c83ae7a1…d06`，终态 observation self-hash 为 `7dc7e356…ca5a`。formal attempt=1，retry/replay/resample/parser/quota/
family/candidate/gate change=0。

因此 FanOutQA P1 是 **source-contract terminal / efficacy unknown**，不是 Agent 相对 RAW 或 HippoRAG 的负结果。不能根据已经
看到的 schema 改 `categories` parser 后重跑，也不能缩 quota 或继续同 DEV。现实域三臂稳定优势与 L5 两个总缺口均保持不变；
若继续，只能使用全新 source/domain、study ID、root 与 cohort，并在任何新 source semantic access 前一次性冻结真实 raw
schema acquisition、selection commitment、模型/checkpoint/prompt/corpus projection、三臂与 A_hold→M_search 合同。

### 12.43 2026-07-23 BIRCO P1：全新三域 source qualification 有效通过

FanOutQA P1 关闭后没有修改其 parser 或使用同源 rescue cohort。新 study `BIRCO_P1_TYPED_CONSTRAINT_E4_V1` 选择 BIRCO 的
`doris-mae`、`clinical-trial` 与 `wtb`，明确排除本地历史已使用过的 ArguAna。正式 source 只绑定 Zenodo record
`10850865` 的单一 `BIRCO_dataset.json`，不混用后来 GitHub pickle epoch。Zenodo aggregate metadata 的 CC-BY-SA 与仓库 aggregate
CC-BY-NC、各子任务许可证并不一致，因此 custody 预先采用较严格的 CC-BY-NC 私有研究边界；原始 query/document/qrel 不回传、
不提交，archive 只允许 hash、aggregate、opaque work ID、rank ordinal 与 score。

设计先于 source semantic access 固定：每族 A_form/F_search/A_hold/M_search 各 10 个 query，四块 query-disjoint；candidate
membership 来自 qrel keys，但数值 qrel 始终封存到对应 action archive 完成以后。现实域 primary 使用预先固定的 E0 Agent，不能在
A_hold 看到 E4 是否晋升后偷换 primary。candidate 是 typed required/excluded/eligibility/temporal/relational facet DAG 与
facet-to-evidence capacity matching，不是 RRF、关键词、删除式 gate 或不断追加 gate；E4 是完整四-recipe slate 上的 listwise
distributional policy，不复用此前 lower-quantile HGBR。RAW 使用相同 GPT-5.4 endpoint 的 direct relevance scoring；HippoRAG
只称“pinned official core + frozen candidate-pool adapter”，每 query 对同一 candidate pool fresh index/retrieve，不能声称
official BIRCO baseline 或 full-corpus retrieval。

311linux 上的一次不透明下载由独立 service 完成，network attempt=1、retry=0，固定字节为 20,134,244 bytes、MD5
`548cad5d…fa78`、SHA-256 `0c30d869…1f34`。在 JSON parse 前，custody/design/download authorization/receipt、qualifier 与
23/23 synthetic tests 已冻结；唯一 `birco-p1-source-qualification-v1.service` 随后消费 qualification 与 source-open 两个
O_EXCL marker，在禁止网络的 scope 内完成 aggregate-only parse。result self-hash=`c708a02d…6618`，qualified=true；没有
terminal failure、item selection、model/action、online evaluator 或 score。

正式 aggregate 为：DORIS-MAE query/corpus/pool-entry=`60/5543/6633`，Clinical-Trial=`50/3256/3420`，WTB=
`100/1767/5043`；各族均超过 4×10 query quota。DORIS relevance 含 6,142 个 fractional score，故 primary nDCG@10 保留原始
linear graded gain，不能全局二值化；Recall@5 才按 score≥1。candidate-ID overlap graph 的 DORIS 与 WTB 都只有一个 query
component，Clinical 有 6 个 component 但最大一个含 45/50 query。这一结果没有触发新 gate，反而证实预注册的边界是必要的：
blocks 只能 query-disjoint，不能 document-disjoint；gain-vs-harm tail 只能是描述性 reference，不能包装成总体随机化 p-value。

因此 BIRCO P1 已从“source feasibility unknown”进入 **qualified / implementation pending**。它仍不是 efficacy 结果。下一步只允许
完成 frozen selection commitment、runtime/capacity attestation 与 typed/E4/RAW/HippoRAG adapter 实现，用公开合成数据作一次
非评分 integration diagnostic，然后一次性执行 A_form→F_search→A_hold；只有预注册 E4 promotion 成立才打开 M_search。

### 12.44 2026-07-23 BIRCO P1 正式终态：F_search 不可识别，三臂与 L5 均未进入

后续实现没有修改 candidate 或 performance gate。selector、typed constraint/E0/E4、GPT-5.4 semantic worker、RAW、
candidate-pool-restricted official HippoRAG adapter、offline scorer 与 crash-closed controller 经 113/113 BIRCO tests 后冻结；
Plus route 与 4-slot HippoRAG runtime 只用公开常量/合成数据作非评分 preflight。第一次部署 selector 的 systemd 进程在 Python
package entrypoint import 时因 staging closure 缺少已提交的 `assumption_agent.models` 而退出；当时 selector module/main 尚未进入，
selection root、secret 与 source open 均为 0。该 pre-capability failure 已单独 disposition，补入同一已提交依赖并发布 superseding
implementation freeze；它不是一次被消费 selection 的重试。随后唯一 capability-consuming selection 一次成功，以 fresh 32-byte
HMAC secret 将三个 family 各 `10×4`、共 120 个 query 分配进 query-disjoint A_form/F_search/A_hold/M_search，未公开原始
qid/cid/query/document/qrel。

正式 `birco-p1-formal-v1.service` 在 311linux 单次启动、0 restart、exit 0，最大 API concurrency 实测 64，CPU 41.759 s、
memory peak 850.1 MiB。A_form 的 148/148 semantic requests transport succeeded；118 个 completion 通过 grammar，30 个 planner
输出按事前 totalizer 完成。A_form action archive 先以 0400 封存，随后 qrel capability 只消费一次并离线拟合 30-slate E4；
solver 在 41 iterations 后 `converged=false`，objective=`41.22566591119905`。这不会绕过后续 identifiability gate。

F_search 的冻结 route 随后发生明显 provider degradation：142 个已消费 requests 中只有 10 个 transport succeeded，132 个是
`transport_unavailable`；仅 6 个 generation-valid，另 4 个成功响应由 totalizer 完成。controller 没有 retry、replay、resample、
provider switch 或 online evaluator。由这些冻结 terminal 形成的 E4 与 E0 完整 permutation 仅在 30 项中的 2 项不同，虽跨 2 个
family，但未达到预注册的“至少 3 项且至少 2 family”。因此 formal terminal 正确写为
`terminal_F_search_label_free_unidentifiable`，F_search qrel open=0，并在任何 A_hold action/qrel 前停止；A_hold 与 M_search 均未
materialize，RAW 与 official HippoRAG 也从未在正式 reality block 运行。

该终态应分两层解释。执行层是 **protocol-valid**：freeze/self-hash、只读 archives、attempt/terminal 一一对应、0 failure receipt、
0 retry 与 qrel custody 均通过离线复核；启动时的 runpy duplicate-import warning 来自 package `__init__` 预先导入同一个无
import-time action 的 runner，未改变唯一 `__main__` lifecycle，记为以后修复的非结果性 warning，不据此宣告 implementation-invalid。
实证层则是 **provider-degraded / mechanism-inconclusive**：正式可支持的最窄陈述只有“本次冻结执行中的 E4 action 未通过
label-free identifiability”，不能把它解释为 Agent 输给 RAW/HippoRAG，也没有现实域三臂数据，更没有 evaluator promotion 或
untouched M_search 的 L5 证据。BIRCO 当前 root 永久终止，不得用同一 selection 重跑、补一次 ranking、降低 3-item gate 或事后打开
F/A_hold/M qrel。总目标的两个缺口——跨 family 同时超过 RAW/HippoRAG，以及 promotion 后改善 untouched search——保持原样。

若继续总目标，不能再在 BIRCO terminal 上补 gate。下一项独立 study 应把 semantic action 形成改成 provider-capacity 与正式
measurement 解耦的冻结机制（优先完全本地、或在新 cohort 身份形成前完成可审计的容量资格），使用新 source/cohort/study ID 一次
执行；否则最诚实的当前结论仍是现实域稳定双基线优势与 L5 均未达到。

### 12.45 2026-07-23 MMQA P1：本地 structured-proof study 在 source 解封前冻结

BIRCO P1 没有重跑、降低 F gate 或重用其 selection。新的 `MMQA_P1_LOCAL_PROOF_E5_V1` 改用 official MultiModalQA commit
`4dd14328c6d02a4daa357cc6032915a0b14602e3`，只授权 TRAIN、DEV、tables、texts 四个固定 gzip；size、Git blob SHA-1 与
总字节数 69,204,571 已从 repository metadata 固定，但四个 formal file 的 download/parse/item/label/support open 仍全部为 0。
官方仓库没有可确认的 license file，因此边界是 311linux 私有、研究用途、不再分发，只公开 hashes、capacity 与 aggregate score。

这次没有把 provider prompt 换一种写法。action 形成被改成 query-local typed proof：ROW 是一条带 header/cells/link titles 的 table
row，TEXT 是一段 source text；label-free closure 各最多 48 个 ROW/TEXT node，只用 exact cell-link↔text-title/URL 建边，枚举 2–5
node、最多 256 个 connected row-text bundles。E0 是固定 unsupervised proof energy；唯一 challenger E5 在 A_form 三族各 40 项上用
L2=1、L-BFGS max_iter=256 的 conditional maxent 拟合。exact-positive 必须是 bundle 自身包含一个 late exact gold row-text pair；
top-5 的补齐节点或满 nDCG 不能伪造 positive。若一个 sealed slate 没有 exact-positive，它仍进入全体 bundle feature scaler，但不生成
伪标签、不贡献 conditional gradient。RAW 直接用同一 frozen cross-encoder 排同一 closure；official HippoRAG 只在同一 closure 内建立
fresh per-query index。primary 是 binary exact-support linear-gain nDCG@5，utility 固定为 `floor(1e9×nDCG)`。

四块在 source open 前固定为 TRAIN A_form=`40×3=120`，DEV F_search=`5×3=15`、A_hold=`15×3=45`、M_search=`15×3=45`；
DEV 按共享 table/text connected component 跨块隔离。F_search 永久不打开 gold，也不再充当 identifiability gate。A_hold 的
E5−E0 正净 utility 与 ties-excluded exact reference tail≤0.10 是 M 的唯一 opening rule；E5 同时超过 RAW/HippoRAG、两组 aggregate
tail≤0.10 且每个 family 严格为正，是独立 reality-primary claim。reality-primary 失败会关闭该 claim 并禁止同源 rescue，但只要
E5−E0 已晋升，仍执行事前冻结的 untouched M_search；这提前消除了旧 terminal-list 文句可能造成的歧义。

pre-source adversarial audit 还发现 gold-open capability 不能只相信 caller 提供的 64-hex。冻结实现因此要求在 A_form/A_hold/M 打开
gold 前复验真实 canonical mode-0600 action archive 的 file/self hash、block、action-pack、item count 与完整有序 anonymous work-ID；
M 另复验真实 A_hold score 的 `promoted=true`、`M_search_authorized=true`、A_hold archive 及已消费 authorization/marker。两项都只是
使既有 no-label-leakage 与 promotion-only 合同可执行，不改变 source、cohort、candidate、features、metric、threshold 或 gate。

311linux 当前 AppArmor policy 使 unprivileged bwrap user namespace 不可用，user service 的 `PrivateNetwork` 也不是可依赖的有效边界；
因此 pre-execution disposition 不再声称二者。outer 从启动起固定 `RestrictAddressFamilies=AF_UNIX` 并主动验证 AF_INET/AF_INET6
socket creation 均以 errno 97 失败；唯一 network-enabled sibling 只运行四个固定 HTTPS acquisition，显式限制为
AF_UNIX/AF_INET/AF_INET6、`env -i`、0 retry，完成即退出。正式 child 再独立使用 AF_UNIX-only、NoNewPrivileges、PrivateTmp、
ProtectSystem、ProtectHome、单一 project ReadWritePaths 与 UMask 0077。official comparator 按其已冻结 runtime 改为 CUDA hidden、
最多 4 个 CPU worker；GPU0/GPU1 分别留给 MiniLM 与 cross-encoder。这只是实际 scheduling attestation，不改变三臂语义或 pipeline-
utility（非 equal-compute SOTA）claim。

上述 outer/runtime、gold capability、structured target、source/qualification 与 official adapter 的完整 pre-source 离线套件为
277/277 PASS，`py_compile` 同时通过。该时点仍是 **pre-source implementation freeze pending**，不是 efficacy 结果。下一步只允许先在 source 不存在时完成两套 public-
synthetic runtime canary 与 external filesystem live revalidation，再冻结 implementation/execution manifests；随后一次执行四文件下载、
aggregate-only qualification、private component-atomic selection 与正式 controller。任何 source identity/capacity、runtime、one-shot action 或
valid non-promotion/primary failure都按预注册终止，不 retry、resample、换模型、补 gate 或用在线评价。

2026-07-26 重启 311linux 以使已升级的 NVIDIA `595.84` userspace 与内核模块一致后，第一次 official public-preflight service
invocation 在 `_json_mapping_argument` 内即退出：远端 shell 对未作为单一 argv 元素保护的 inline JSON 做了 brace expansion。
traceback 控制流、正式 root 不存在、receipt 与 canary stage 均不存在共同证明
`build_fresh_comparator_preflight`、runtime inspector、public canary、source、model action 与 score 的消费次数仍全部为 0。该失败已单独
记录为 [`pre-capability launch disposition`](../manifests/mmqa_p1_official_preflight_launch_disposition_v1.json)；只允许换新 unit 将同一
两张 frozen JSON map 各自作为一个完整 argv 元素传入，随后消费唯一一次 public-canary capability。它不授权改 candidate、cohort、
feature、metric、threshold、gate、model/runtime 或 source，也不把一次已消费的 canary 失败重试。

纠正后的 `mmqa-p1-official-preflight-v2.service` 通过 AF_UNIX isolation、实际 filesystem tree 与 package/import-root inspection 后，
确实只消费一次 public synthetic canary；official child 随后 return code 1，stdout 为空，冻结 adapter 只保留
stderr SHA-256 `3f8e3cd8…2cc47`，因此没有 receipt。item-local work 已销毁、stage parent 为空、残留进程为 0；正式 root 仍不存在，
四个 source file 的 download/parse、formal item/action/score 与 online evaluator 仍全部为 0。按照事前 no-retry 合同，MMQA P1
由 [`official preflight terminal`](../manifests/mmqa_p1_official_preflight_terminal_v1.json) 关闭为
**source-free infrastructure-invalid / efficacy unknown**。不得重放 canary、换 runtime/model 后继续该 study 或下载其 source；后续只能
新立 study，并在新 prospective runtime contract 中让 public failure 原因可审计。这是换 workstream，不是补 gate。

终止后只做了不执行 worker/model/index/retrieve 的静态 postmortem。冻结 MuSiQue worker 把两个绝对模型路径分别拼成
`Transformers/<absolute path>`；pinned HippoRAG 对两个名字做 slash→underscore，再以一个下划线连接成单一
working-directory component，随后在模型构造前调用 `os.makedirs`。311linux 上两个 label 的精确字节数为 153 与 118，
连接后为 272，而同一目录的 `NAME_MAX=255`。因此存在一个独立于 benchmark 内容与 GPU 状态的确定性不兼容；由于 P1
只保存 stderr SHA-256，这里严格称为 exact-code/path-length 所定位的 overwhelmingly likely root cause，不声称 digest
本身确认了 `ENAMETOOLONG` 文本。BIRCO 已在相同 HippoRAG 文件身份上用 cwd-local `smollm2`/`minilm` 两个短 alias
通过 public synthetic preflight，投影 basename 仅 40 bytes。新的独立 study 必须同时固定：绝对模型 argv 数为 0、alias
symlink 的 `lstat/readlink/resolve/samefile` 与 target-tree identity、deployed project-first 的 child `PYTHONPATH`/transitive
module-origin 检查，以及 scratch 外 mode-0600 私有 stdout/stderr custody。完整证据见
[`static postmortem`](../manifests/mmqa_p1_official_preflight_static_postmortem_v1.json)；它不修复、不重开 MMQA P1。

### 12.46 2026-07-26 MAUD extraction P1：现实域三臂与 L5 的新 source/design freeze

MMQA P1 不重开。新 study `MAUD_EXTRACTION_P1_TYPED_EVALUATOR_V1` 固定 official
`The-Atticus-Project/maud-extraction` commit `89cc5f6ce210268f170aa019ea204ed4e608c604` 的三个独立
TRAIN/DEV/TEST blob，总计 61,342,921 bytes；重复的 `train_and_dev` 永久禁止。该 supplementary extraction source
并非与 primary MAUD 独立的血缘：论文明确说明 extraction spans 对应 primary main dataset 的 deal-point texts。因此公开
viewer 自动展示过的 13 份合同与论文表格可定位的 3 份合同，均以 normalized title SHA-256 denylist 在三个 split 中按整份
contract 全局排除；不公开 raw title，不替补、不重采样。此时三个 formal JSON 仍未下载或解析。

正式 parser 不复用 upstream 首次缓存需至少 150 GB RAM、训练约需 22 GB GPU 的 baseline pipeline，也不继承它只训练第一条
span 的已知 bug。新的 trusted streaming SQuAD2 parser 完整消费 `answers[]`，保存 exact raw-context character offsets；
同一合同的 22 个 public-template queries 共用一个 gold-independent passage corpus。TRAIN 以 fresh HMAC 按合同分成
A_form 4/5 与永久 label-free 的 F_search 1/5；DEV 是一次 A_hold，TEST 只有 A_hold promotion 后才能 parse 为 M_search。
所有剩余合同的 22 个 query 均先执行，无答案 query 只报告数量，不进入 positive-evidence retrieval utility，也不产生
abstention/no-answer claim。

RAW 是 frozen cross-encoder；official HippoRAG comparator 只在同一合同的相同 passage corpus 上每合同建一次 index；
Agent 的九个固定 recipe 均调用 definition/condition/exception/section-xref typed operator，不把纯 CE 或纯 fused ranking
伪装成 Agent。E0 与 E1 都是同一个 `score(item, recipe) → fixed argmax` 接口；E1 只在所有 A_form actions 封存后，用
recipe utility 相对 E0-selected utility 的 delta 拟合唯一 L2=1 ridge，之后不再重拟合。F_search 只封存 E0/E1 的
recipe/behavior identity，相同也不是 gate。

统计审计修正了同一合同 22 个 query 的伪重复风险：先在 `contract×family` 内求均值，再等权得到 contract utility；
A_hold/M 的 reference tail 只对非零 paired contract deltas 做全部符号枚举，并如实称为 sign-flip reference tail，
不冒充无条件 exact causal p-value。唯一 TEST opening rule 是 A_hold 上 E1−E0 contract net positive 且 tail≤0.10；
family 条件与 RAW/HippoRAG 不混入 promotion。现实域 primary 独立要求 E1-Agent 对 RAW 与 official HippoRAG 的
contract aggregate 均为正、各自 tail≤0.10，且三个事前 public type family 对两条 baseline 的净差全部严格为正。
L5 则只在晋升后用完全冻结的 E1/E0 slate 检验 untouched TEST contracts；任何合法 non-promotion 或负结果都关闭本
source epoch，不 rescue、不补 gate。

311linux 重启后模型资产身份未漂移，两张 RTX 2080 与 driver `595.84` 已恢复；但 kernel/driver runtime identity 已变化，
必须建立新的 source-free fingerprint。旧 HippoRAG tree hash 把可变 `.pyc` 纳入身份，因此本 study 改为绑定排除
`__pycache__`/`.pyc` 的 60-file source tree `342505c3…27b1f`，同时设置 `PYTHONDONTWRITEBYTECODE=1`；模型只通过
study-local `minilm`/`smollm2` 短别名传入，避免 MMQA 的 272-byte basename 冲突。执行上最多两条 GPU lane 与四个
CPU worker，按 stage bulk-submit 后统一 join，全程离线评分、无 fine-tune、无 API/online evaluator。

实现审计又在 source access 前关闭了几处会让结果无效或让 311 再次满载的缺口：HTTP redirect 必须在第二个 GET
发生前拒绝；trusted acquisition 必须逐字节绑定 downloader receipt，并以 title/context 双身份证明
TRAIN/DEV/TEST contract-disjoint；TEST capability 不能仅凭任意 self-hash 与两个布尔值，而要精确重算并绑定
A_hold action/gold/model/contract-tail receipt。A_form 的 unanswerable item 保留全部九个 training rows，训练专用
target 全为 0，但仍不进入 primary metric。typed coordinate runtime 只允许 project+typed-site import closure，
不继承 Ruoli/OpenAI/API/proxy 环境；MiniLM、cross-encoder 与 official HippoRAG 都固定 native/Torch
intra-op/inter-op 线程，OpenIE executor 强制单 worker，并对每个模型进程的 OS-thread peak 设为最多 2。唯一
source-free canary 必须对同一 `build_passages` corpus 依次覆盖并发 MiniLM/CE、coordinate join、九个 typed
recipes、E0 与 official HippoRAG；formal 只验证该已冻结 receipt，不重放 canary。62 个纯 synthetic test
已全部通过。当前状态推进为 **pre-source implementation frozen；只缺新的 runtime fingerprint、唯一 full canary
与 execution freeze**，仍不是 efficacy 结果。

第一次生成 runtime fingerprint 时，official import closure 在 receipt 写入前静态失败：`-S` 正确屏蔽了隐式
system-site，但显式 `PYTHONPATH` 漏列了现有 base Python `dist-packages`，所以先暴露 `distro`、继续只读检查又会暴露
`click`。该阶段没有模型 inference、full canary、source GET/parse、action 或 score；临时下载的 `distro 1.9.0`
probe 从未进入任何 frozen path，随后已删除。implementation freeze v2 仅在所有更具体的 project/overlay/HippoRAG/p16
root 之后追加现有 base root，并精确绑定 `distro 1.7.0` 与 `click 8.0.3` 的版本和 module origin；不改变任何
candidate、evaluator、metric、threshold、cohort 或 gate。

重启验证后，corrected source-free runtime fingerprint 一次通过：kernel、driver、两张 RTX 2080、typed 与 official
两套显式 import closure、模型/source tree 和新增 `click`/`distro` origin 均与冻结身份一致，且该阶段仍为 0 model
inference、0 source/score。随后唯一 full canary 在 `CPUQuota=400%`、`MemoryMax=40 GiB`、`TasksMax=64` 的
user cgroup 中启动；MiniLM 与 cross-encoder 两个 frozen worker 都完成同一 synthetic contract 的 22 个 query，
分别写出 canonical mode-0600 output，0 API/network、0 retry/resample。二者随后都在 output 写入后的同一检查处
退出：observed process OS-thread peak 大于冻结值 2。由于 frozen code 只在阈值通过后才把 peak 数值写入 stdout，
不能事后声称精确 peak；但两份完整 output、模型加载 stderr、无 OOM/timeout/quota termination 以及未创建 official
scratch/custody 共同把失败定位为 thread-count gate，而非 GPU、内存、模型或 source 问题。

这个 gate 在语义上也是错误的：native BLAS/OpenMP 与 Torch intra/inter-op 均已固定为 1，但 CUDA/Transformers 可创建
不等于并行计算核心数的 helper threads；用 `/proc/<pid>/task` 总数≤2 作为资源上限会拒绝正常推理。P1 的唯一 canary
能力已经消费且没有通过 receipt，因此严格终止为 **source-free implementation-invalid / efficacy unknown**。私有
coordinate outputs 只保留作 custody，不作为 passed canary 复用；official HippoRAG、execution freeze、selection secret、
三个 formal JSON 的 GET/parse、action、gold、score 均未发生。下一步不是给 P1 补 gate 或重跑，而是新建独立 successor：
移除无效的 per-process OS-thread-count 判定，保留 native/Torch/OpenIE worker 控制，并把真实资源边界放在预注册的
bounded process count 与外部 cgroup `CPUQuota`、`MemoryMax`、`TasksMax`。

successor `MAUD_EXTRACTION_P2_CGROUP_BOUNDED_EVALUATOR_V1` 已在任何 P2 remote root、runtime fingerprint、model inference、
canary、secret 或 source access 前另行冻结。P2 继承同一 unopened official extraction source、16-contract exposure denylist、
contract-disjoint HMAC split、九个 typed recipes、E0/E1 ridge、RAW/official-HippoRAG 三臂、离线 coverage utility、
A_hold promotion 与 untouched M_search L5；P1 的 private coordinate output 不作为输入。唯一 substantive runtime change
是删除 total OS-thread-count 的 pass/fail 判定，同时保留 native BLAS/OpenMP=1、Torch intra/inter-op=1、OpenIE
max-workers=1、每 GPU 最多一个 official contract worker和两条 GPU lane。outer 必须以 `Restart=no`、
`CPUQuota=400%`、`MemoryMax=40 GiB`、`TasksMax=64` 启动；observed process thread peak 仅保存在私有 diagnostic。
P2 另使用 `maud-p2-*` public-synthetic contract/query seeds，避免把 P1 已消费的 fixture 或 output 冒充 fresh canary。
完整 64/64 synthetic tests、`py_compile` 与 whitespace diff check 已通过。新的 project-root-bound runtime
fingerprint 一次通过；唯一 full canary 随后在实际 cgroup 中一次完成并发 MiniLM/CE、同一个 22-query synthetic
corpus、九个 typed recipes、E0 和 official HippoRAG，service restart=0，0 API/network/retry。canary receipt 与
仍在 active unit 内形成的资源 attestation 都在 execution freeze 前封存。execution freeze
`8b16fd9e…1472` 再绑定 implementation commit `befe6fb4`、runtime/canary、fresh 32-byte secret 的单向
commitment 与 formal config；此时 source GET/parse、formal model/action/gold/score 仍全为 0。

唯一 source download 以独立 `maud-p2-download-v1.service` 启动，实际属性为 `Restart=no`、
`CPUQuota=100%`、`MemoryMax=1 GiB`、`TasksMax=16`、`KillMode=control-group`。它严格发出三个 commit-addressed
HTTPS GET，不跟随 redirect、不解析 JSON。TRAIN 49,039,965 bytes 与 DEV 6,133,011 bytes 完整通过冻结
size/Git-blob 校验；第三个 TEST stream 在约 300 秒后只写入 403,264 / 6,169,945 bytes 就结束，因而由冻结代码
报 `source size does not match frozen metadata`。terminal receipt 只记录两个 complete files、
`MaudDownloadError` 和 `terminal_no_retry_resume_or_mirror_switch`；unit exit 1、restart=0，未生成成功 receipt。
完整与 partial source bytes 均为远端 mode-0600 私有 custody，未公开、未解析，也不会输入 successor。

因此 P2 严格终止为 **acquisition-infrastructure-invalid / efficacy unknown**：不 retry、resume、换 mirror/downloader，
不解析 partial/complete P2 bytes，不启动 P2 formal controller，不产生任何三臂或 L5 分数。这一终态不是
Agent 对 RAW/HippoRAG 的负结果，也不改变既有证据；总目标仍同时缺现实域三个 relation family 对两条 baseline
均稳定为正，以及 evaluator 在独立 A_hold 晋升后改善 untouched M_search。下一步只能使用全新 study/source/cohort，
并优先选择已完整本地冻结、无需临时网络传输的 source artifact；不能把 P2 改名后补跑。

### 12.47 2026-07-26 EBM-NLP P1：source-header schema qualification 终止

MAUD P2 不重开。新 study `EBMNLP_P1_TYPED_PICO_SET_EVALUATOR_V1` 改用已完整驻留 311linux、固定
SHA-256 `b7357503…a44e3f` 的 EBM-NLP 2.0 archive，避免把临时 API 或正式阶段网络容量继续混入效果检验。
候选是 P/I/O 三个 role 的 typed evidence-set operator；A_form/F_search/A_hold/M_search、RAW、item-local official
HippoRAG、E0/E1 promotion 与离线 exact-Fraction scoring 均在正式 source epoch 前冻结。E1 只有在独立 A_hold
胜过 E0 才能打开 M_search；现实域 primary 仍要求 E1-Agent 对 RAW 与 official HippoRAG 的 aggregate 和三个
role family 都为正。这里没有 fine-tune、在线 judge 或 Ruoli/API evaluator。

v1–v3 的 source-free 尝试都在模型或 source 前暴露 implementation defect；每次都按各自 disposition 关闭，没有把
失败能力冒充 passed canary。v4 改正后先形成 project/runtime fingerprint，再以两张 RTX 2080 上各一 worker 完成
2/2 public-synthetic terminal 与 2/2 CUDA attestation；峰值只有两个 worker，external network、API credential、
retry、replay 与 source access 均为 0。execution freeze `cfc32850…79bb1` 随后绑定 implementation、
fingerprint、canary、formal config 与 exact systemd unit，才允许等待 GPU 空闲并启动正式服务一次。

正式启动本身有效：live receipt 在 source 前观测到 `ActiveState=activating`、`SubState=start`，验证
`IPAddressDeny=any`、仅 `AF_UNIX`、两张 GPU、`CPUQuota=800%`、`MemoryMax=40 GiB`、`TasksMax=64` 与
0 provider/API credential；service 最终 `Result=success`、main exit 0、restart 0。这里的 systemd success
只说明 fail-closed controller 正常写出安全终态，不代表 benchmark 成功。正式 source epoch 恰好消费一次，
archive SHA、size 与 mode 在终止后仍与冻结身份一致。

controller 在任何 member payload 打开前只检查 tar header topology。`documents/` 下有一个 regular member
不符合冻结 grammar 允许的 numeric-PMID `.tokens`、`.text` 或 ancillary `.pos` 路径，因而抛出
`EbmNlpP1SourceQualificationError`。终态中的 error-message SHA-256 `ca880135…adf6ad` 与已提交代码第 456 行的
静态 literal `document member does not match the exact frozen path pattern` 精确相等；私有 member 名未输出、
未回传、未写入公开 manifest。这已经足以定位为 frozen path grammar 与实际 public archive header topology
不相容，不需要也不允许事后查看成员名来扩 allowlist。

因此 v4 是 **protocol launch valid，但 source-schema qualification implementation-invalid**：member payload、
block/cohort、MiniLM/official HippoRAG、typed action、gold/label、E0/E1、score receipt 与 stage archive 全部为 0，
efficacy 为 unknown，`primary_evaluated=false`，`replay_permitted=false`。它既不是 Agent 的性能负结果，也不能支持
Agent>RAW、Agent>official HippoRAG、现实域三-family primary、evaluator promotion、M_search improvement 或 L5。
同一 epoch 不 retry、不 resample、不换模型/provider、不补 allowlist/gate。

下一条有效路线不是重开 EBM-NLP P1，而是新 source/study/cohort：在 secret、cohort 和评分 formal 之前，用独立的
public、non-scoring schema study 前瞻性资格化 archive topology，再一次冻结可执行 grammar。这样把公开文件格式
兼容性从密封效果 epoch 中移出，是修正 study ordering，不是不断增加 efficacy gate。总目标仍同时缺少现实域三个
relation family 对 RAW 与 official HippoRAG 的稳定净收益，以及 evaluator challenger 在独立 A_hold 晋升后改善
untouched M_search 的 L5 证据。

### 12.48 2026-07-26 AVeriTeC P1：正式执行有效，但 typed action 对 RAW 无效

EBM-NLP 的 source-header 失败没有通过补 allowlist 重开；后续先用独立公开 P0 一次资格化 official AVeriTeC
TRAIN/DEV 的 archive、schema、三类原生 claim type 容量和 collision topology，再建立
`AVERITEC_P1_TYPED_QA_SET_EVALUATOR_V1`。source-free production canary、execution/launch freeze、142-file
checksum closure 与 311linux 唯一正式 service 均通过；正式终态为 systemd success、restart 0、40 个私有 artifact
完整、0 API/online evaluator/retry/replay/resample，因此这轮是有效 efficacy measurement，不是 implementation-invalid。

A_hold 上 E1 在 19/36 项选择非 RAW recipe 并改变 top-5，但 E1−E0/RAW 仍为 36/36 utility tie、净 0、exact tail 1。
E1−item-local official HippoRAG aggregate 虽为净 `+3/2`、tail `3/32`，causal family 却为净 0；而 E1 与 RAW
逐项完全相同，所以该差不能归因于 typed action。promotion 与现实三-family双基线 primary 均为 false；formal
controller 没有读取或执行预冻结 M_search，故 L5 为 `null`。这关闭的是“同一 dense corpus 上增加 query prefix/slot
assignment 足以提高 evidence recall”的路线。下一 study 必须换 source/cohort，并让 operator 直接优化
decomposition 后的 evidence-set marginal coverage 与互补性；不能在 AVeriTeC 上继续调 ridge、recipe、alpha 或补 gate。

### 12.49 2026-07-26 WiCE P0：公开 topology 一次资格化失败，按冻结规则关闭

下一候选 WiCE 没有直接进入正式评分。`WICE_P0_PUBLIC_SCHEMA_TOPOLOGY_V1` 先在任何 source body 下载前冻结
design、qualifier 与 tests；311linux 随后一次并行取得 official commit `ddeb6c18…9870f` 的三个 claim blob，
且 size、SHA-256、Git blob 全部匹配。TRAIN/DEV 仅产生 schema/topology/capacity/collision 聚合，TEST 只验证
identity 与 raw newline，JSON decode 为 0；全过程没有 secret、cohort、action、model、evaluator、qrel、score、
API 或在线评价。

一次 qualifier 终态为 `not_qualified_public_schema_anomalies`：TRAIN/DEV 共记录 67 个空白 evidence sentence、
27 个 not-supported row 带非空 supporting set、59 个重复 supporting alternative，总计 153。该结果说明
**冻结 contract 与 public source 不相容**，不说明 WiCE source 本身无效，也没有测到 Agent/RAW/HippoRAG
performance。P0 design 已明确规定 failure 后不修改同源 parser/family/capacity、不 replay、不形成 P1，因此 WiCE
在这里严格结束；这避免把观察到的格式特征改写成事后通过规则。总目标仍缺现实域三个可辩护 family 的双基线稳定优势
和 L5，后续只能转向全新 source/study/cohort。

### 12.50 2026-07-26 HiTab P1：直接边际 coverage evaluator 的 pre-source implementation closure

WiCE 没有按观察到的 anomaly 修改 parser 后重开。新的
`HITAB_P1_DMC1_HIERARCHICAL_SET_EVALUATOR_V1` 改用 Microsoft HiTab 固定 commit
`d179602662b490249baf068a76fbe4137029126e`，只把 source-native aggregation token 事前归为
AGGREGATE、COMPARATIVE、SUPERLATIVE 三个 operation family，并永久排除 Wikipedia/ToTTo 与 corporate
financial table source。这里的 family 只用于 selection balance 与结果分组，不能写成跨 semantic relation family
证明；任务也只是 item-local hierarchical table evidence-cell retrieval@5，不是 official HiTab QA 或 execution
accuracy。

候选机制不再增加 recipe gate、fallback 或 RAW retention。DMC1 在同一完整 cell corpus 上从空集合逐步选择五个
evidence unit；E0 使用固定整数势函数，E1 只从 A_form 已密封的 state/action feature archive 学习 exact DNF utility
marginal。每个官方 `[ANSWER]` coordinate 是一个独立 singleton requirement，相同 literal 不合并；corpus commitment、
coordinate-to-ordinal mapping 与 qrel binding 贯穿 registry、model 和四臂 action archive。A_form=`108`、A_hold=`36`、
M_search=`36`，每个 block 三 family 平衡且所有 block 同 table 最多一个 question。A_hold 的 E1−E0 exact sign-flip
promotion 是首次 decode TEST 和 materialize M 的唯一 authority；现实 primary 另要求 E1 对 RAW 与 item-local
official HippoRAG 的 aggregate 及每个 family net 均严格为正，不能反向 gate promotion。

正式实现采用三支独立 user unit：source-free canary、唯一四文件 commit-addressed GET acquisition、以及断网 formal。
外层 typed runtime 与 Hippo child runtime 都从 `python -S -B`、`PYTHONPYCACHEPREFIX=/dev/null` 和显式 ordered
roots 启动，分别绑定完整 Python stdlib/dependency tree、module origin/version 与旧 P17 HippoRAG source/SmolLM2/
MiniLM attestation；claim scope 明确限于 Python filesystem closure，不扩写为 libc/CUDA ABI 的跨机器重现性。
canary 先在零 Hippo 的 synthetic A_form 完成 registry seal→late qrel→E1 fit，再让 GPU1 odd Hippo 的真实
subprocess-launch ACK 与第二次 GPU0 compile/RAW/E0/E1 重叠，empty-cache receipt 后才放行 GPU0 even Hippo。
ACK 缺失/重复、pre-ACK failure、旧 false-overlap 反例、binding failure、重复 canary/formal、source-before-bootstrap、
qrel-before-seal 与 promotion-before-TEST 都有 fail-closed 回归。

implementation inventory 的真实路径资格化在 source access=0 时另外发现三项不能由 synthetic fixture 暴露的
filesystem drift：typed venv 的 copied executable 会把 loader 推到错误 stdlib，P17 base stdlib/dist-packages
含无效 symlink，以及旧 P17 HippoRAG receipt 的 36 个 path-sensitive `pyc` 中已有一个漂移。处理没有放宽
verifier：outer/child 改为两个 study-local lexical venv symlink，共同指向去除无效链接且完整 tree-bound 的
study-local Python base；HippoRAG 执行源改为事前从五个独立副本一致的 60 个 non-bytecode 文件形成的 portable
projection（generic tree `925e2a…3e4c`），保留旧 `a644ab…cdd5` receipt lineage、旧 origin-file hash 与两个
model hash，但不声称 60 文件逐一由已漂移的 legacy bytecode tree 密码学导出。projection 的精确 study-local
relocation、full-root receipt、`src` dependency receipt、无 symlink/special/hardlink/bytecode 及每次 child
launch 前重验均已闭合。

首版 clean-source implementation inventory 的 file SHA-256 为
`95441e20…6dc3`、self SHA-256 为 `30585c5b…3158`，但随后 unitlike live probe 发现外层
SentenceTransformers 5.5.1 导入 `torch.distributed.nn.jit.instantiator`，在随机共享 `/tmp` 下生成并执行
`_remote_module_non_scriptable.py`。最早一次以 script file 启动 verifier 的尝试还会把 `preparation/`
加入 `sys.path`，故单独记为 invocation-invalid；修正为 unitlike `-c` 后复现的 `/tmp` 路径才是首版真正的
source-free implementation-invalid 根因。两者都发生在 source、model action、qrel 与 score 之前，旧 inventory
只作失败证据保留，不能授权 canary 或 formal。

事前冻结的 `direct_transformers_minilm_v2` addendum 没有改 study、family、selection、DMC1、promotion 或
primary contract，也没有增加行为/效果 gate。它只把外层 encoder 换成同一
`sentence-transformers/all-MiniLM-L6-v2@1110a243…4d41` 的 Transformers 5.10.1
`AutoTokenizer+AutoModel`：max length 256、attention-mask mean pooling、单次 L2 normalize、float32，
并在初始化、每次 encode、production binding、完整 public canary 与 formal controller 返回后拒绝任何
`/tmp`、`/var/tmp` Python module/package path 以及外层 `sentence_transformers`。独立 HippoRAG child 仍使用
已封闭且 source-free diagnostic 无 shared-tmp module 的 SentenceTransformers 3.1.1。八条公开合成文本上，
direct 与退役参考的最大绝对差为 `1.4901161193847656e-08`；这只记录 feasibility，不是通过阈值或额外 gate。

v2 implementation freeze 已在远端 source-free inventory 中唯一生成：file SHA-256
`72722ab5…5d47`、self SHA-256 `bf8b2910…8e46`。外层 live closure 通过，但独立 Hippo child probe
fail-closed；一次只读差异诊断确认 child return code 0、invalid cache 0、stderr 为空，唯一未冻结
`sys.path` 是 `/`。它来自 `python -c` 的 cwd injection；同一 bootstrap 在真实 item process 中会把 exclusive
model cwd 留作 Python import root。此时 v2 canary attempt、HiTab source、model action、qrel 与 score 仍全为 0，
故 v2 inventory 保留为 source-free implementation-invalid，不能覆盖或授权 canary。

事前 v3 addendum 只在 import probe 与正式 child bootstrap 中删除空、相对及所有 resolved-cwd alias，并在
`runpy.run_module` 前再次断言 cwd 不可导入；工作目录仍用于解析 content-addressed `smollm2`/`minilm`
相对模型路径，但不再是 Python search root。真实 subprocess 负测同时覆盖 `.`, absolute cwd 与 symlink alias，
并证明相对模型路径语义不变。study、direct MiniLM、family、selection、DMC1、promotion、primary 与唯一
production-isomorphic canary 均未改变，也没有新增 behavior/efficacy gate。

v3 implementation freeze 的 file/self SHA-256 为 `40275c5d…da61` / `308c3bfd…5863`，但 unitlike
child probe 仍 fail-closed。静态定位确认不是另一种 source 或模型问题：冻结的 LiteLLM 顶层导入
`litellm.proxy.proxy_cli`，其第 21 行明确执行 `sys.path.append(os.getcwd())`，把已经移除的 cwd 在 import
过程中重新加入。v3 canary attempt、HiTab source、model action、qrel 与 score 仍全为 0，因此 v3 同样只作为
source-free implementation-invalid 保存。

最后的 v4 不再按库名追加过滤项，而是把 child 的初始 exact path sequence 变成贯穿 import probe 与完整
`runpy` worker 生命周期的 fail-closed list subclass：只对空/cwd alias 与 exact frozen duplicate 做 no-op，
任何其他新增、删除、切片、`+=`/`*=`、重绑或顺序漂移均失败；probe 每次 import 后及输出前复验，
worker 在 `runpy` 前和 `try/finally` 后复验，因此 `SystemExit(0)` 也不能绕过。合成 subprocess 已覆盖
cwd absolute/`.`/symlink alias、unknown absolute path、destructive mutation、rebind、`*=` 与 SystemExit，
并保持相对模型资产解析。这里的 closure claim 只针对已冻结、非对抗依赖使用的常规 `sys.path` API，不声称
list subclass 是恶意 Python sandbox；整体重绑等旁路由紧邻检查点 fail-closed。v4 是同一个 pre-source
import-closure defect 的最终结构性修复，不是新增 efficacy gate；
若它失败，HiTab P1 直接结束，不再形成 v5/v6。

v4 source-free code、design、units 与 74/74 离线测试已通过独立终审；design self hash 为
`9f060347…6dba`，v2/v3/v4 addendum self hash 分别为
`b5cb382e…149`、`fe55b40f…ca506`、`b15c6f80…91c9`。远端唯一 v4 implementation freeze 的
file/self SHA-256 为 `111bf7e6…fdfdc` / `d163f2b4…9cf0`，断网 unitlike 双 runtime live binding
self SHA-256 为 `b12321e7…d3c3`；真实冻结的 LiteLLM import 在这里已经通过。

两张 GPU 空闲后，唯一 canary service 于 06:54:30 有效启动：`NRestarts=0`、40 GiB、8 CPU、
仅 AF_UNIX，attempt file/self SHA-256 为 `c97c9133…c8a0` / `10b4b224…7b90`。Planner 与
cross-encoder 加载后，direct Transformers MiniLM constructor 入口的首个 guard 立即发现
shared-temporary Python path；该失败早于 direct MiniLM 自己导入 `torch`/`AutoModel`，也早于任何 public
canary item execution。未保留具体临时 module path/name，不能再归因给某个前序组件；静态错误 literal SHA-256 为
`f7ac238d…dd51`。service 于 06:56:10 以 exit 1 fail-closed，qualified receipt、Hippo child launch、
source acquisition freeze、四个 GET、HiTab decode/selection、qrel、score、API/online evaluator 均为 0。

因此 HiTab P1 的终态是 **source-free implementation-invalid / efficacy unknown / primary=false**，不是
Agent 相对 RAW 或 official HippoRAG 的性能负结果。按 v4 事前停止规则，不 retry、resample、换模型、
改 family、补 allowlist/gate，也不形成 v5/v6；HiTab P1 永久结束。总目标仍同时缺少现实域三个 family
对 RAW 与 official HippoRAG 的稳定优势，以及 evaluator promotion 后改善 untouched M_search 的 L5。
下一步只能换全新独立 study/source/cohort，并采用不依赖该 shared-temporary Python 行为的 runtime backend。

### 12.51 2026-07-27 LoCoMo P0：公开 source topology 不满足冻结 contract，P1 未形成

HiTab 关闭后没有在同源补 v5/v6。新的 `LOCOMO_P0_PUBLIC_SCHEMA_TOPOLOGY_V1` 选择 LoCoMo 固定
commit `3eb6f2c5…b376`，事前把 category 4/1/2 固定为 SINGLE_HOP/MULTI_HOP/TEMPORAL，要求十个
conversation 均在 qrel cardinality 1–5 后每 family 至少有 12 项，从而无内容选择地支持 conversation-group
disjoint `2/4/4` formation/A_hold/M_search。P0 只允许一次公开、非评分 schema/topology/capacity 资格化；
不允许生成 secret/cohort/action/evaluator/qrel archive/score，也不允许根据 aggregate 修改 parser、quota、
family 或 partition。

第一次 systemd launcher invocation 在 LoCoMo module import 前即因远端最小快照漏带已提交的
`assumption_agent/models.py` 停止：`InvocationID=f54fbb5b…4354`、`NRestarts=0`、CPU 19,977,000 ns，
且 `p0_work_v1`、pre-network attempt marker、source directory 和下载均不存在。它被单独记为
pre-entrypoint implementation-invalid，不计 source attempt。补齐同一提交中 package import 与冻结 verifier
已经绑定的 test 文件后，source-free import/verifier 才通过；unit、qualification module、source identity、
family、quota 与 partition 没有改变。

唯一真实 P0 attempt（`InvocationID=ee573f78…0ad`、`NRestarts=0`）随后下载并验证三份固定官方文件，
data/README/LICENSE 的 size 与 Git blob 全部匹配，data SHA-256 为 `79fa87e9…8ff4`。strict JSON 只 decode
一次，安全 aggregate 得到 10 conversations、5,882 turns、1,986 QA、2,815 evidence links，其中 2,806
可映射。公开 category 总数与论文一致：single-hop 841、multi-hop 282、temporal 321、open-domain 96、
adversarial 446。

冻结资格没有通过。决定性的 capacity 反例是一个 conversation 在 qrel 1–5 后只有 11 个 eligible
multi-hop，故 10 个 conversation 中只有 9 个满足 multi-hop quota=12；此外 strict schema 还记录 1,471
个 aggregate anomaly，包括官方 multimodal turn 的额外/部分字段、空 optional value、一个缺失 session-date
pair、9 个 evidence grammar/映射问题、一个重复 evidence，以及 6 个 category 1–4 非文本 answer。
这些都是事前 contract 与 source 的不相容，不是 Agent/RAW/HippoRAG 性能结果。receipt 已写为
`terminal_not_qualified_no_same_source_revision`，同源不 replay、不把 quota 改成 11、不为观察到的字段补
allowlist，也不形成 LoCoMo P1；typed core 只保留为 source-agnostic implementation artifact，不能算效果证据。

因此两个总缺口仍原样存在：现实域三个可辩护 family 对 RAW 与 official HippoRAG 的稳定优势尚未建立，
L5 的独立 A_hold 晋升后改善 untouched M_search 也尚未建立。下一来源必须同时具备可冻结的公开 payload、
原生或事前可辩护的 family、精确 retrieval qrel 与足够独立 cluster；不能继续以降低同源资格条件的方式寻找
“通过”结果。

### 12.52 2026-07-27 TechQA P0：唯一固定下载在零 payload 处终止

LoCoMo 没有降低 quota 或重开。同日另立的 `TECHQA_P0_PUBLIC_SCHEMA_CAPACITY_V1` 固定 IBM TechQA
repository commit `f0cf8ce1…6faf`、Hugging Face revision `60437bc7…365b` 与唯一
`TechQA.tar.gz`，预期 size=`2,959,973,525`、SHA-256=`6b094ef9…118`。P0 只允许验证三个
whitelisted member、官方 QA/corpus schema、50-document candidate set、gold span、共享 typed/official-HippoRAG
字符边界与事前 operational INFORMATION/PROCEDURE/TROUBLESHOOT capacity；不生成 selection secret、
cohort、action、qrel/evaluator 或 score。若 P0 通过，P1 才会一次冻结 TRAIN A_form/F_search、DEV
A_hold/M_search、Agent/RAW/official HippoRAG 三臂以及 A_hold promotion→untouched M 的 L5 链。
这些 family 是事前 operational query-intent strata，不是 TechQA 原生 gold relation label；即使成功，
最终 claim 也必须保留这一边界。

远端 source-free preflight 首先发现一处真实实现错误：runtime manifest 的 path-list digest 绑定其已冻结
declared order，而 verifier 又用 Python code-point order 对同一集合重排。此时 unit 未安装/启动，
`p0_work_v1`、attempt marker、HF network call 与 source byte 均为 0。该失败以
[`preflight disposition`](../manifests/techqa_p0_public_source_preflight_disposition_v1.json) 单独封存；
唯一修复只是按 byte-bound manifest declared order 重算 path-list，未改 source/parser/classifier/quota/
provider/revision/file 或任何效果条件。修复后 311linux 的 3,249 个 regular runtime files、四个 symlink、
resolved Python 与五个 pinned package version 的 source-free attestation 一次通过；66 项 P0/typed/formal/
official-adapter tests 通过，refreeze self SHA-256=`f7e3db01…ab6b`。

唯一 P0 service 于 09:08:35 启动，`NRestarts=0`，attempt marker 已在网络前 durable 写入。固定
`hf download` invocation 在约 72 秒后 return code 1；archive payload file count=0，source member、
qualification result 与 private eligibility manifest 均未形成。safe terminal 把阶段固定为
`single_pinned_hf_download`、状态为 `implementation_source_or_infrastructure_invalid`，且明确
`formal_P1_capability_consumed=false`。冻结实现只保存了 stderr SHA-256，没有保存原始 stderr，
所以现有证据只能支持 **acquisition-infrastructure-invalid / exact transport subtype unknown**，不能把
失败事后归因到具体 DNS/TLS/provider 子类型。

TechQA 同源现已终止：不 retry/resume、不换 mirror/provider/revision/file，也不以已经完成的 P1 runtime
打开其他 TechQA payload。它没有产生 Agent/RAW/HippoRAG 或 L5 性能样本。总目标仍缺现实域三-family
对两条 baseline 的稳定净收益和 evaluator 晋升后改善 untouched M_search；下一实验必须使用独立 source/
study/cohort，并优先选择已经完整本地冻结或能由单一小型 content-addressed artifact 取得的来源，不能继续
把临时大文件传输当作效果研究的前置不确定性。

### 12.53 2026-07-27 MultiDoc2Dial P0：本机一次下载/中继成功，archive topology 资格终止

TechQA 关闭后另立 `MULTIDOC2DIAL_P1_TYPED_DIALOGUE_RETRIEVAL_L5_V1`，没有复用其 source、cohort、
family 或效果条件。official MultiDoc2Dial commit 固定为 `6b756598…ffe9e0`，唯一 archive Git blob 为
`9d8dd4a2…191672`。由于 311linux 不通外网，获取路径按新的固定边界执行：本机 WSL 对 commit-addressed
official GitHub raw URL 只下载一次，随后验证 size=`6,868,509`、Git blob 与 whole-file SHA-256
`f0c034c2…1ce00`；同一字节流再只通过一次 SCP 中继到 311linux，远端 mode=`0600`，size/SHA-256
复核一致。311linux 没有向 Hugging Face、GitHub 或其他外部 source 发请求。

P0 在 source 打开前已提交 typed core、19 项 qualifier tests、systemd unit 与 execution freeze；P0+typed
合计 28 项离线测试通过。唯一 service invocation 为 `be2f7961…58b2`，`NRestarts=0`。实现先验证完整
archive size、SHA-256 与 Git blob，随后只打开一次 ZIP central directory，并在
`_validate_topology` 立即以 `ZIP regular-member whitelist drifted` 终止。该分支位于任何
`archive.open(member)` 之前，因此 document/TRAIN/VALIDATION/TEST payload open count 均为 0；
`work` 目录为空，没有 private eligibility manifest、safe aggregate、secret/cohort、action、qrel、score
或 evaluator 结果。准确分类是 **pre-efficacy source-topology-contract incompatible / efficacy unknown**，
不是 Agent 输给 RAW/HippoRAG，也不是 L5 negative。

冻结 terminal policy 现已执行：不在观察 central directory 后修改四-member whitelist，不重启、不重放、
不换 revision/file/provider，也不进入同源 P1。MultiDoc2Dial 的 source-agnostic typed core 仅保留为实现
artifact。下一独立来源将把公开 archive topology 身份验证放在 acquisition/custody 阶段一次完成，再冻结
效果 study；这是把已知格式不确定性移出密封效果 epoch，不是新增可反复修改的 efficacy gate。外部 source
仍只允许“本机 content-addressed 下载并校验 → 一次同步 311”，正式 action 与评分继续全离线。

本终态不改变总目标的两个缺口：现实域三个可辩护 family 尚未同时稳定胜过 RAW 与 official HippoRAG；
也仍没有 evaluator 在独立 A_hold 晋升后改善预冻结 untouched M_search。后续必须换独立
source/study/cohort，不能用 MultiDoc2Dial topology 观察反向修同源候选。

### 12.54 2026-07-27 DSTC9 P1：P0 与双 GPU canary 通过，正式 source runtime 依赖终止

MultiDoc2Dial 的 topology 终止后没有反向修改同源 whitelist。新 study
`DSTC9_P1_HIERARCHICAL_KNOWLEDGE_EVALUATOR_L5_V1` 固定 official DSTC9 commit
`7ebb4c76…cc0ae4` 的 TRAIN、VALIDATION 与 knowledge。311linux 不通外网，因此 source 获取严格采用
“本机 WSL 下载 official commit-addressed bytes、逐文件验证 size/Git blob/SHA-256、形成 deterministic
USTAR、只 SCP 一次、311 再验证 mode/size/SHA-256”的路径；远端 bundle 为
`116,961,280` bytes、SHA-256 `6c3efa69…ffb83`。311 没有执行 Hugging Face、GitHub 或其他外部下载，
后续 action 与评价也保持完全离线。

一次公开、非评分 P0 有效通过：knowledge corpus 为 2,900 条，family 为
hotel/restaurant/taxi/train；TRAIN/VALIDATION 最终 eligible rows 分别为 19,184/2,670，四个
family 的 unique dialogue-group capacity 均高于冻结 quota。P0 只写 mode-0600 private eligibility
manifest 与 safe aggregate receipt，没有形成 formal cohort、action 或 score。P1 随后冻结
`A_form/F_search/A_hold/M_search=96/32/48/48`，E1 仅由 A_form 形成，A_hold 只以 E1−E0
净正且 exact one-sided tail≤0.1 才晋升；M_search 只允许在晋升后 materialize。现实域 primary
另要求 E1-Agent 在 A_hold 的 aggregate 及每个 family 都同时净胜 RAW 与 official HippoRAG。

source-free implementation 经 v1/v2/v4 的确定性 infrastructure failure 与 v3 的 prelaunch canonical-order
修正后，均在 formal source=0 时封存且从未 replay。v5 只将已经绝对路径验证的 P17 model roots 映射为
HippoRAG 内部两个短 alias；唯一 v5 service 在 30 秒内 success，`NRestarts=0`，真实完成 GPU0
official HippoRAG build→reopen retrieve、GPU1 coordinate worker 与 query-only MiniLM predictor，
formal source/API/retry 均为 0。execution freeze `8af372cb…fe188`、formal config
`5ff78dd0…d3f` 与 commit `3f28a6ad…f7d` 随后一次部署；preformal hardware/canary/runtime closure
只读复验通过。

唯一 formal service 于 13:31:45 启动，13:31:46 以 exit 1 终止，`NRestarts=0`。safe terminal
`708bab6f…22d08` 固定失败阶段为 `compile_formal_source_once`；journal 与冻结调用顺序共同证明，
P0 receipt/private manifest 已认证、whole-study secret selection 已形成、formal source counter 已递增，
bundle identity/USTAR topology 已验证且 knowledge member 已打开，随后在第一个 streaming JSON event 前
触发 `ModuleNotFoundError: ijson`。原因不是需要联网下载模型，而是 formal unit 使用的 typed venv
没有包含 P0 source compiler 已冻结依赖的 `ijson`。source/action stage 的持久化条目分别为 0/0，
没有 coordinate/Hippo worker、qrel、score、API 或 online evaluator；没有任何 item/query/document/qrel/
action/per-item score 被公开。

尽管 JSON payload 尚未产生一个 parser event，formal source capability 已按事前合同消费，因此不能把它
重分类为 source-free canary，也不能安装 `ijson` 后重跑同一 DSTC9 source/study/cohort。该 root 永久记为
**formal-source runtime-dependency implementation/infrastructure-invalid / efficacy unknown / no replay**。
下一独立 study 必须在任何 private selection 和 formal source 解封前，对 source parser 的完整 Python
runtime closure 做一次 source-free import/version/entrypoint 资格化；这属于把依赖身份移到正确的阶段，
不是新增 efficacy gate。总目标仍缺现实域至少三个可辩护 family 同时稳定胜过 RAW 与 official HippoRAG，
以及 evaluator 在独立 A_hold 晋升后改善预冻结 untouched M_search 的 L5。

### 12.55 2026-07-27 BioASQ P1：正式 source 已形成，但用户级 systemd manager 随登录会话退出

DSTC9 的 parser dependency 终止后没有补装 `ijson` 重跑。新的 BioASQ study
`BIOASQ_P1_TYPED_QUESTION_EVIDENCE_EVALUATOR_L5_V1` 固定本机已校验后中继到 311linux 的
`training11b.json`（37,639,648 bytes，SHA-256 `6df65686…eac98`），先以公开非评分 P0
证明 yesno/factoid/list/summary 四个 family 的 component capacity，再冻结
`A_form/F_search/A_hold/M_search=96/32/48/48`、2,900-passage shared corpus、E0、四个 typed
evidence-set recipe、RAW、official HippoRAG、A_form-only E1、A_hold promotion 与 conditional
untouched M_search。评价完全离线；311 不通外网，正式所需 MiniLM、cross encoder、SmolLM2、
HippoRAG source 和两个 Python runtime 均已驻留并逐项 exact-hash 通过，不需要 Hugging Face 或其他下载。

source-free coordinate canary 只启动一次并成功：GPU1 对 2,900 synthetic passages 和一个 query 完成
MiniLM/CE，constructor/formal encode count 为 2/1，formal source/action/evaluator/score/API/retry 均为
0；同一时刻形成 BioASQ study ID 的 current-hardware receipt，并与旧 DSTC9 official HippoRAG
canary hardware exact 一致。formal execution binding `42e29673…2998f`、freeze
`07726a72…a8a4`、C2 commit `5588051d…f34a` 与 archive `1f31867d…9dc0` 随后封存。唯一 formal
service 于 17:01:26 启动；preflight、source compiler 与 selection 均成功，source open/hash/decode
各为 1，private cohort/corpus 已形成。controller 随后并行启动一次 GPU1 initial-176 coordinate
worker 和一次 GPU0 HippoRAG global build。

17:11:36，远端最后一个 SSH/login session 断开；当时 `loginctl show-user` 的冻结后诊断为
`Linger=no`。17:11:47，用户级 systemd manager 激活 `exit.target`，先停止 default target、DBus
与两个 transient child unit，再以 `status=15/TERM` 停止 formal parent。journal 的因果顺序和
全 user-manager shutdown 证明这不是模型 OOM、provider、311 外网、candidate、source parser 或
formal code 自行退出。service start=1、restart=0；中断时 coordinate score 与 Hippo build receipt
均未写出，A_form action archive、A_form qrel、E1、F_search behavior、A_hold action/qrel、promotion、
score、M_search 与 outer/controller terminal 全都不存在。M_search 从未由 controller materialize。

由于 formal source 和 secret cohort 已消费，这个 root 不能在启用 linger 后恢复、restart 或 replay；
其终态是 **post-source-selection infrastructure-invalid / efficacy unknown / no replay**，不是 Agent
相对 RAW/HippoRAG 的效果负结果。下一次只能使用新的独立 source/study/cohort，而且在任何正式 source
或 secret 前必须先完成与生产同构的持久化资格：`Linger=yes` 或 system-level service、主动关闭所有
SSH session 后 user manager 仍存活、两个真实 child unit 持续运行并自然终止。它是执行容器资格，不是
新增 efficacy gate。

从 BIRCO P1 起按注册 study ID 严格计数，到本次 BioASQ 一共已经切换 **13 个 study**：
BIRCO、MMQA、MAUD P1、MAUD P2、EBM-NLP、AVeriTeC、WiCE P0、HiTab、LoCoMo P0、TechQA P0、
MultiDoc2Dial、DSTC9、BioASQ；其中 MAUD P1/P2 共用一个 source family，所以是 **12 个独立 source
family**。若把紧邻 BIRCO 之前同一轮目标驱动的 FRAMES 与 FanOutQA 也计入，则是 **15 个 study /
14 个 source family**。这 13 个 study 中，只有 AVeriTeC 完整进入有效 A_hold 三臂 efficacy scoring；
BIRCO 停在 F_search identifiability，BioASQ 停在 action output 前，其余大多终止在 source/P0、
source-free runtime、acquisition 或正式 source schema。这个比例说明当前主要损耗不是“科学假设被
13 次否定”，而是 study ordering 与 production-envelope 资格不足；继续随机换数据集不会提高成功率。

总目标在工程上仍可实现，但实证结论不能保证。已有证据分别证明了必要子链：HybridQA 已真实完成
`A_hold evaluator promotion → authorization → untouched M_search`，只是 M_search 增益不显著；BRIGHT
P9 在三个 family 都高于 HippoRAG，却没有在三个 family 都高于 RAW；Hotpot/2Wiki/EntailmentBank
又证明 frozen Agent 在若干 fresh retrieval cohort 能明显高于 RAW，并可接近或高于 item-local
HippoRAG。因此两个目标并非被逻辑或现有数据否定，但还没有在同一现实 study 中同时成立。真正的效果
缺口是 action generator 必须产生 RAW top-5 之外的互补证据，并让 evaluator 在独立错误分布上选择
这种真实增益；基础设施缺口则应一次解决，不再让它消耗新 cohort。下一路线应先暂停 efficacy study
轮换，完成 persistent-service source-free soak，再只选择一个已公开 topology/schema、可本地完整
资格化且 action 机制与 family 划分都有因果含义的新 source，一次冻结并执行。

### 12.56 2026-07-27 persistent user-service P0：先冻结执行容器资格，不消费新 study

BioASQ root 永久关闭后，operator 已在 311linux 单独执行 `loginctl enable-linger erzhu419`；
只读复核得到 `Linger=yes`、user manager `running`。这不会授权恢复 BioASQ，也不改变其
post-source infrastructure-invalid 终态。为避免第三次由登录会话生命周期烧毁 secret cohort，
新的 `RQ_PERSISTENT_USER_SERVICE_P0_V1` 被明确注册为 **source-free infrastructure
qualification，而不是 efficacy study**。

freeze `eb4104e8…643a` 固定 311linux 的 Python 3.12、systemd 255 与
`systemd-run/systemctl/loginctl` 字节身份；唯一 parent unit 必须启动两个独立 transient child
unit，三者均为 `Type=exec / Restart=no / KillMode=control-group / AF_UNIX-only`。parent 与 children
必须分别证明 AF_INET/AF_INET6 被拒绝；启动 SSH session 关闭后，新的 observer session 必须在冻结的
15–55 秒窗口内观察同一 boot ID、同一 user-manager PID 以及三个 active unit，随后两个 child 必须
自然终止，parent 才能形成 success terminal。整个资格的 source/item/query/document/label/qrel/action、
model/GPU/provider/API/online-evaluator count 固定为 0；不得 restart、retry 或改窗口。只有本 P0
通过，才允许为下一个独立效果 study 形成 source/cohort。

唯一执行已通过。parent 于 10:00:25 UTC 启动 coordinate/hipporag 两个 sibling transient service；
原 launch session `13838` 随后关闭。28.31 秒时，新 observer session `13842` 观察到 boot ID 与
user-manager PID 均未改变、parent 与两个 child 全部仍为 `active/running`，且三者均保持
`Type=exec / Restart=no / KillMode=control-group`。两个 child 在 75 秒后自然完成；parent
`Result=success / ExecMainStatus=0 / NRestarts=0`，stdout/stderr 为 0 bytes。terminal
self SHA-256 为 `f6d32092…6098`，离线复核 8 份 canonical receipt、file/self commitment、mode
`0600` 与所有零活动计数均通过。

因此 **user-service 持久化缺口已经关闭**，可以为一个新的独立效果 study 形成设计；这项 P0
本身没有消费效果 source/cohort，也没有增加现实域双基线优势或 L5 证据。BioASQ 及其他已关闭 root
仍不得 replay。

### 12.57 2026-07-27 Spider P1 custody：停止同类 QA 轮换，改测可执行 schema-graph action

下一个候选不再是另一份 text/table QA prompt 变体。新 study
`SPIDER_P1_TYPED_SCHEMA_EXPANSION_EVALUATOR_L5_V1` 选择官方 Spider 1.0 的现实跨域
text-to-SQL schema-linking：公开 release 含 10,181 个人工问题、200 个多表数据库与 138 个 domain，
`tables.json` 原生给出 table/column/type/primary-key/foreign-key graph。这里的 action 不依赖关键词
或 prompt，而是从 RAW semantic top-5 schema seed 出发，执行可审计的
`value anchor → column → table membership → foreign-key path` typed expansion；它会真正把 RAW top-5
之外的 schema evidence 加入生成空间，而不是重新排序同一候选。

source custody `f13ffd48…1021` 固定官方页面、CC-BY-SA-4.0 archive、Google Drive file ID、205,800,266-byte
HEAD、official code commit `b7b5b8c8…ca7c` 与一次本机下载/clone 权限。下载后先只允许 central-directory
inventory，再单独形成 source-access receipt；随后可以公开、非评分地解析 TRAIN/DEV、tables 与 SQLite
schema aggregate，测量 `ONE_FOREIGN_KEY_EDGE / MULTI_FOREIGN_KEY_PATH / NESTED_OR_SET_RELATION`
三类是否同时有足够、gold schema evidence 为 2–5 的 component-disjoint 容量。TEST payload、selection
secret、item action、RAW/HippoRAG、evaluator 与 score 此时都保持未打开。只有 topology/capacity 真实通过
后才冻结四块和效果执行，避免再次拿 secret cohort 做 parser canary。

唯一 archive GET 与 official-repository clone 均已完成：archive 为 205,800,266 bytes、SHA-256
`00636695…121b`，official code 为 commit `b7b5b8c8…ca7c` / tree `7687d1f7…3491`。central
directory 有 2,625 entries、0 duplicate、0 unsafe path；TRAIN/DEV database 侧有 166 个 SQLite
member。source-access receipt `bf53b034…379a` 形成时 archive member payload open 仍为 0，
`test.json/test_tables.json/test_database` 均未打开。下一步只允许在 committed qualifier 中读取
TRAIN/DEV/tables 与对应 database aggregate。

唯一 P0 随后合规终止，原因是 **未见 schema 的 DEV multi-FK family 容量不足**，不是 parser 或
infrastructure failure。TRAIN 中三类分别有 1,823 / 127 / 1,134 个 eligible item，三个冻结 tier
的 database-disjoint TRAIN allocation 全部可行；但 DEV 的 `MULTI_FOREIGN_KEY_PATH` 只有 10 items、
3 databases，低于最小 floor 的 12 items、6 databases，而另外两类分别为 263/20 与 140/20。
因此不能降低 quota、把 train row 填入 M、改变 family 或重跑。terminal
`a98bc12b…f9a5` 记录 effect cohort/secret/action/RAW/HippoRAG/evaluator/score/model/GPU/API 均为
0，TEST 与 SQLite payload 也仍未打开。

这关闭的是 Spider 1.0 对当前跨-schema L5 设计的容量，不是 typed schema expansion 的效果负结果。
机制只允许原样移植到一个独立、公开即可先证明 DEV multi-relation 容量的 source；若没有这样的来源，
本路线应停止，不能继续 source roulette。

### 12.58 2026-07-27 BIRD P1：最后一次 schema-expansion source 审计

Spider P0 没有测到机制效果，因此允许把同一个 typed schema-expansion 机制原样移植一次，但不允许继续
轮换来源。最后一个 source 固定为官方 BIRD train/dev，study ID 为
`BIRD_P1_TYPED_SCHEMA_EXPANSION_EVALUATOR_L5_V1`。BIRD 与 Spider/BioASQ 及近期效果 roots 独立；
action 仍固定为 `semantic schema top-5 seed → table membership → declared foreign-key path`，三类仍为
`ONE_FOREIGN_KEY_EDGE / MULTI_FOREIGN_KEY_PATH / NESTED_OR_SET_RELATION`，没有换 quota、family、
metric、gate 或 prompt。

source custody `b7fbb94f…1449d` 已事前固定官方页面、CC-BY-SA-4.0、train/dev archive URL、HEAD
byte length/ETag/Last-Modified、official code commits 与访问次数。由于官方 train archive 为
8,919,543,554 bytes、dev archive 为 346,207,293 bytes，而资格化只需要公开 annotations 与
`tables.json` schema graph，custody 禁止下载或打开无关数据库值，改为每个 archive 唯一一次 suffix
Range GET 绑定完整 central directory；只有形成 source-access receipt 后，才允许各一次取得四个指定
member。该动作只节省无关 8.9 GB 数据，不改变 source、split 或 cohort。

这是本机制的硬停止点：若公开、非评分 P0 仍不能在 BIRD DEV 中满足同一 floor，则关闭
typed schema-expansion 路线，不再选择第五个同类 source；若通过，才允许冻结 A_form/F_search/A_hold
与 untouched DEV M_search，并一次执行 Agent/RAW/official HippoRAG 和 L5。

唯一 topology inventory 已完成，未下载 8.9 GB/346 MB 的完整 archive，也未解压任何 member。
train/dev central directory 分别为 10/5 entries，均为 0 duplicate、0 unsafe path；四个授权 member
的 central-directory size/CRC/local-header offset 已绑定。source-access receipt
`3b6b70fd…f755` 形成时，semantic member payload、database values、selection/action/baseline/evaluator/
score 的打开计数仍全部为 0。下一步只能用已提交的 acquisition/qualifier 各一次取得并解析这四个
member，形成 aggregate P0 terminal。

central directory 不包含 local-header extra length，因此事前 transport addendum
`59dcda4a…9d42` 仅把每个 member 的单次 combined GET 改为一次 30-byte local-header GET 加一次
精确 filename/extra/compressed-stream GET；source/family/quota/metric/gate 均未改变。P0 freeze
`b26ded14…37e4` 已绑定本机取得的 `sqlglot==30.13.0` wheel `08f87ff7…49b1`、5 个 source-free
tests、保守 alias/table/column parser、四个 member 的唯一打开次序与一个不可 replay 的 output root。
现在才允许唯一一次公开 P0。

第一次 launcher invocation 因冻结 output root 的父目录不存在，在 `Path.mkdir` 即退出；控制流尚未写
`qualification.attempt.json`、尚未进入 `qualify()`，四个 member 的 local-header/stream GET 和
semantic payload open 全部为 0。因此 incident `86932b02…85d9` 将它定性为
`pre_attempt_launch_preparation_invalid`，不是资格化 retry。只允许创建 mode `0700` 的冻结父目录，
随后用同一 commit、arguments 与 output leaf 消费仍剩的一次正式 P0；不得修改代码、source 或设计。

父目录完成后，唯一正式 P0 于 10:41:48 UTC 写入 attempt `2de306fd…3a57`；控制流依次打开了四个
授权 public member，随后在 schema capacity aggregation 前以 `RowContractError` 终止，没有
`qualification.result.json`。只读 post-terminal 诊断定位到冻结 contract 的实质错误：BIRD
`primary_keys` 同时包含 integer 与 composite-key list，而 parser 错误地要求每个 member 都是
integer；非权威 mirror aggregate 中 TRAIN 为 338 integer + 127 list，DEV 为 66 + 6。该诊断没有
打开 question/SQL/database value/score，也没有用于修改或重跑。

terminal `468a5fa5…c9e40` 因此是 **implementation-invalid、capacity/efficacy unknown**，不是负效果。
四个 member 已被正式打开，所以不得修 parser 后 replay、另立 BIRD P2 或再换第五个 source。
`typed schema expansion` 这条路线按事前 hard stop 关闭；它没有增加现实域双基线优势或 L5 证据。
下一步若继续总目标，必须先基于既有全部结果做 architecture-level stop/go 分析，不能立刻再开一轮
source/gate/prompt study。

### 12.59 2026-07-27 architecture stop/go：typed candidate 有真实上界，边际 evaluator 不稳定

BIRD 关闭后没有继续换 source。architecture decision `66040a27…a958` 先把 FRAMES→BIRD 的近期
序列固定为 **17 个 study / 16 个独立 source family**，并把既有有效结果放在同一证据账本中：
HybridQA 已经真实发生过 A_hold evaluator promotion，但 E2 在 A_hold 对 RAW 为 0、对 HippoRAG
为 −1/2，untouched M_search 的 E2−E0 只有 +3/2、exact p=1/2；BRIGHT P9 三个 family 都高于
HippoRAG，却对 RAW 分别负、平、正；EntailmentBank 对 RAW 三 family 都有大幅优势，却只在一个
family 高于 HippoRAG。故当前不是“少跑一个数据集”，而是同一 study 中尚未出现稳定双基线优势，
且 evaluator 的未来动作选择仍没有 L5。

这也排除了把 HybridQA P6 再包装成新机制：P6 本来就保留 RAW top-3，再由 direct/path typed
candidate 填两个 residual slot，并由一个 global recipe evaluator 选择。唯一允许的 retrospective
meta-development 改为实质不同的 action space：对每题枚举所有 RAW top-5 外、由 query anchor 或至多
两条 typed edge 可达的 candidate；每个 action 是替换一个仍在场的原始 RAW slot，显式 no-op 与其
共同竞争，最多替换两个不同 slot。固定的无截距 λ=1 weighted ridge 学习每个具体 action 的边际
utility，而不是选择一个全局 recipe。A_form/A_hold/M_search 三个均已消费且相互不交叠的 block
只做 leave-one-block-out 描述性 cross-fit；它们不能重新成为 fresh efficacy evidence。事前 hard
stop 要求 oracle 在每个 block/family 都正、learned 在每个 held block 和 pooled family 都正、
learned pooled 高于旧 P6 path-2，且 learned−RAW exact p≤0.1；任何一项失败都停止，不改 feature、
fold、λ、threshold、gate 或 prompt，也不再消耗独立 source。

source-free runtime 先发现旧 QASPER CPU canary 在 311 上虽 repeat-exact，但 float/quantized hash
分别为 `6b0a0498…509f` / `86a1a981…16f`，与原硬件冻结值不同。此时 HybridQA pack open=0。
没有放宽成 tolerance，而是使用 311 上 P17 已资格化的同权重 GPU0 runtime：
MiniLM model tree `1514beb6…cfdb`、torch 2.4.1+cu118，256-sentence 两次输出逐字节相同且
hash `62fc4780…5f8c`；5 个 source-free pure test 全部通过。GPU runtime qualification
`0031e086…db22` 因而在任何 pack open 前封存，这只是把已知跨硬件问题固定为同机数值身份，
不改变 candidate、metric、fold 或判定线。

第一次 service invocation 因部署时中间 parent `/formal_v1` 被 `install -d` 以 0755 创建，在
`validate_output_parent` 即退出。`work/`、attempt、pack、model inference 与 action 均不存在，
`NRestarts=0`；incident `0d354da4…13d1` 将它固定为 pre-attempt launch preparation invalid。
只把该已冻结 parent 改为 0700 后，实际唯一 attempt 才开始；这不是 cohort retry。实际 invocation
`338dcaac…aa21` 于 54.539 CPU seconds 后 success，`NRestarts=0`、memory peak 523.7M、swap 0，
运行中采样 GPU memory 2,168 MiB。本次没有新 source、fresh selection、TEST、HippoRAG feature、
API/online evaluator、retry/replay/resample。

108 个已消费 item 的 aggregate 离线结果为：

- RAW `781/6`、complete 56；旧固定 P6 path-2 `344/3`、complete 47；
  cross-fit learned `811/6`、complete 60；oracle `202`、complete 101。
- learned−P6 path-2 为 **+41/2**，14 正 / 1 负 / 93 平，exact p=`3/32768`。这证明逐题 action
  scorer 比旧 global residual recipe 更合理。
- learned−RAW 仅 **+5**，9 正 / 6 负 / 93 平，exact p=`3473/16384`，未达 0.1；
  三个 held block 分别为 `+7/3, −11/6, +9/2`，A_hold 为负；三个 family 分别为
  DUAL `+4`、PASSAGE `+3`、TABLE `−2`，TABLE_ONLY 为负。
- oracle−RAW 为 **+431/6**，45 正 / 0 负 / 63 平，exact p=`1/35184372088832`，而且事前核验的
  每个 block×family 都严格为正。

因此 terminal `404737a0…b5a8` 与 result `78c88290…36f5` 的有效结论是
**`STOP_CURRENT_ARCHITECTURE`**。关键定位已经从“typed grammar 找不到 RAW 外证据”推进为：
RAW 外存在大量具有因果 utility 的 typed action，真正瓶颈是当前 additive marginal ridge evaluator
不能跨 block、尤其不能在 TABLE_ONLY 上稳定识别它们。该结果同时解释了为何继续换 source、补 gate
或微调 residual recipe 不会解决总目标。它不增加现实域 Agent−RAW−HippoRAG 或 L5 fresh evidence，
也不授权独立 confirmatory study。若以后恢复研究，只能先提出能表达 candidate 交互与 set-level
utility 的实质新 evaluator architecture，再另做一次全新的 architecture decision；当前主线按
预注册停止，两个总目标缺口原样保留。

### 12.60 2026-07-28 whole-set interaction 资格化：aggregate 显著改善，但 TABLE_ONLY 终止晋级

按 12.59 的唯一合法后续，没有换 source，也没有再补 gate。architecture decision
`fe9bc18d…421` 事前固定了实质不同的 whole-set hypothesis class：对每题的 RAW top-5 与全部
typed-reachable 外部 candidate，完整枚举 no-op、一次替换与两次替换的最终 top-5 set；48 个固定
feature 同时表达 candidate-candidate、candidate-retained-RAW、typed connectivity、path
complementarity、global coverage 与 deletion loss，再用无截距 λ=1、item/utility-stratum
balanced 的单次 L2 solve，全局 argmax 选择最终 set。最大状态数为每题 20,481；没有 candidate
pruning、sampling、顺序 policy、family label、HippoRAG feature、threshold 或 post-result repair。

在打开既有 private pack 前，311linux 的 source-free numeric canary 已由两个独立进程逐字节复现，
qualification self 为 `aef2539c…e8cce`；pack access、online/API evaluation 与正式 output 均为 0。
实现 freeze `8f5a70fb…e462` 经 30/30 文件、runtime、service、acquisition commitment 与既有完整
fold/family projection 双重校验后提交。正式 service 只启动一次：
InvocationID `5ada9716…b47`，`NRestarts=0`、`Result=success`、CPU 1min 4.014s、
memory peak 621.7M、swap 0；没有 retry/replay/resample、新 source、fresh selection、official
TEST 或外部 evaluator。

108 个已消费 item 的 pooled aggregate 为：

- RAW `781/6`、complete 56；旧 P6 path-2 `344/3`、complete 47；旧 marginal-v1
  `811/6`、complete 60；whole-set learned **153**、complete **74**；complete-set oracle
  **202**、complete 101。
- learned−RAW 为 **+137/6**，24 正 / 10 负 / 74 平，exact p=`70842161/17179869184`；
  learned−marginal-v1 为 **+107/6**，22 正 / 12 负 / 74 平，
  exact p=`397410689/17179869184`；learned−P6 为 **+115/3**。
- 三个 held block 的 learned−RAW 均为正：A_form `+23/2`、A_hold `+16/3`、
  M_search `+6`。这说明 whole-set interaction 确实消除了旧 marginal evaluator 的 block-level
  负迁移。
- 但 pooled family 只有 DUAL_TABLE_PASSAGE `+46/3` 与 PASSAGE_ONLY `+27/2`
  为正；TABLE_ONLY 为 **−6**、complete `−3`。learned 选择了 210 次 replacement，而 oracle
  只选择 51 次，显示当前 fixed linear set energy 仍严重 over-act，尤其不能在 TABLE_ONLY
  保留 RAW。与此同时 oracle 在每个 block×family cell 仍严格为正，所以失败不是 candidate
  state space 没有有益证据。

九项实现级事前 requirement 中只有一项失败：
`set_learned_positive_every_pooled_family=false`；其余包括每个 held block 为正、pooled exact
p≤0.1、超过 P6、超过旧 marginal，以及 legacy full projection 精确复现均通过。由于规则是
all-of-nine，terminal `57a1d1fc…5a7c` 合法给出
**`STOP_SET_INTERACTION_ARCHITECTURE`**，result self 为 `aad912a0…8346`，后验 disposition
为 `b049b407…81e2`。

这是一个比 12.59 更强但仍不足以晋级的重要结果：set-level interaction 不是无效方向，它在已消费
数据上大幅提高 aggregate、三个 held block 与两个 family；但预注册要求的是跨三个 family 的稳定
无伤害，而非用 aggregate 掩盖 TABLE_ONLY 回退。因此不能事后加 TABLE switch、replacement
penalty、family-specific expert、改 λ/stratum/feature，也不能据此打开独立 confirmatory source。
本轮没有 official HippoRAG 三臂，也不是 fresh efficacy 或 L5 测量。

故当前授权研究边界已经耗尽：同一 architecture/cohort/root 不得重跑，当前 architecture 下不得
另换 source，confirmatory study 未获授权。总目标在逻辑上没有被否定，但实证仍缺两项：
现实域至少三个事前 family 同时超过 RAW 与 official HippoRAG，以及 evaluator 在独立 A_hold 晋升
后改善预冻结 untouched M_search 的 L5。若未来重新开放研究，必须先明确扩展 architecture-level
研究边界，并在任何新效果数据前冻结一个不依赖本次 TABLE_ONLY 观察而形成的新训练/测量方案；
不能把本次失败直接改成新 gate 后继续。

## 附录 A：关键证据索引

- HybridQA whole-set interaction consumed-data architecture qualification：
  [`architecture decision`](../manifests/red_queen_set_interaction_architecture_decision_v1.json)；
  [`source-free numeric runtime qualification`](../manifests/hybridqa_set_interaction_numeric_runtime_qualification_v1.json)；
  [`implementation freeze`](../manifests/hybridqa_set_interaction_meta_development_freeze_v1.json)；
  [`attempt`](../artifacts/hybridqa_set_interaction_meta_development_v1/attempt.json)；
  [`safe aggregate result`](../artifacts/hybridqa_set_interaction_meta_development_v1/result.safe.json)；
  [`terminal`](../artifacts/hybridqa_set_interaction_meta_development_v1/terminal.json)；
  [`result disposition`](../manifests/hybridqa_set_interaction_meta_development_result_v1.json)；
  [`implementation`](../assumption_agent/benchmarks/hybridqa_set_interaction_meta_development_v1.py)。

- HybridQA marginal-replacement consumed-data architecture decision chain：
  [`architecture stop/go`](../manifests/red_queen_architecture_stop_go_v1.json)；
  [`GPU runtime qualification`](../manifests/hybridqa_marginal_replacement_gpu_runtime_qualification_v1.json)；
  [`implementation freeze`](../manifests/hybridqa_marginal_replacement_meta_development_freeze_v1.json)；
  [`pre-attempt parent-mode incident`](../manifests/hybridqa_marginal_replacement_pre_attempt_parent_mode_incident_v1.json)；
  [`safe aggregate result`](../artifacts/hybridqa_marginal_replacement_meta_development_v1/result.safe.json)；
  [`terminal`](../artifacts/hybridqa_marginal_replacement_meta_development_v1/terminal.json)；
  [`result disposition`](../manifests/hybridqa_marginal_replacement_meta_development_result_v1.json)；
  [`implementation`](../assumption_agent/benchmarks/hybridqa_marginal_replacement_meta_development_v1.py)。

- BIRD P1 typed schema-expansion source chain：
  [`source custody`](../manifests/bird_p1_typed_schema_expansion_source_custody_v1.json)；
  [`public source access`](../manifests/bird_p1_public_source_access_v1.json)；
  [`ZIP transport addendum`](../manifests/bird_p1_zip_member_transport_addendum_v1.json)；
  [`sqlglot runtime asset`](../manifests/bird_p1_sqlglot_runtime_asset_v1.json)；
  [`P0 qualification freeze`](../manifests/bird_p0_public_source_qualification_freeze_v1.json)；
  [`pre-attempt parent incident`](../manifests/bird_p0_pre_attempt_parent_directory_incident_v1.json)；
  [`P0 terminal`](../manifests/bird_p0_public_source_qualification_terminal_v1.json)；
  [`P0 qualifier`](../assumption_agent/benchmarks/bird_p0_public_source_qualification_v1.py)；
  [`remote ZIP topology inventory`](../scripts/inventory_remote_zip_members_v1.py)

- Spider P1 typed schema-expansion source chain：
  [`source custody`](../manifests/spider_p1_typed_schema_expansion_source_custody_v1.json)；
  [`public source access`](../manifests/spider_p1_public_source_access_v1.json)；
  [`P0 qualification freeze`](../manifests/spider_p0_public_source_qualification_freeze_v1.json)；
  [`P0 terminal`](../manifests/spider_p0_public_source_qualification_terminal_v1.json)

- persistent user-service source-free P0：
  [`qualification freeze`](../manifests/persistent_user_service_p0_qualification_freeze_v1.json)；
  [`one-shot implementation`](../scripts/qualify_persistent_user_service_v1.py)；
  [`qualification result`](../manifests/persistent_user_service_p0_qualification_result_v1.json)

- BioASQ P1 source-free canary、formal freeze 与 post-source infrastructure interruption chain：
  [`study design`](../manifests/bioasq_p1_typed_evidence_set_evaluator_study_design_v1.json)；
  [`P0 aggregate receipt`](../manifests/bioasq_p0_public_source_qualification_receipt_v1.json)；
  [`coordinate canary freeze`](../manifests/bioasq_p1_source_free_coordinate_canary_freeze_v1.json)；
  [`coordinate canary receipt`](../manifests/bioasq_p1_source_free_coordinate_canary_receipt_v1.json)；
  [`formal execution binding`](../manifests/bioasq_p1_formal_execution_binding_v1.json)；
  [`formal execution freeze`](../manifests/bioasq_p1_execution_freeze_v1.json)；
  [`formal runtime`](../replication_runtime/bioasq_p1_formal_v1/runner.py)；
  [`infrastructure interruption disposition`](../manifests/bioasq_p1_formal_infrastructure_interruption_disposition_v1.json)

- DSTC9 P0/source-free canary/formal terminal chain（formal action/qrel/score 均未进入）：
  [`public source custody`](../manifests/dstc9_p1_public_source_custody_v1.json)；
  [`local acquisition and relay receipt`](../manifests/dstc9_p1_public_source_acquisition_receipt_v1.json)；
  [`P0 qualification freeze`](../manifests/dstc9_p0_public_source_qualification_freeze_v1.json)；
  [`P0 safe aggregate receipt`](../manifests/dstc9_p0_public_source_qualification_receipt_v1.json)；
  [`v5 source-free canary binding`](../manifests/dstc9_p1_source_free_canary_binding_v5.json)；
  [`formal execution freeze`](../manifests/dstc9_p1_execution_freeze_v1.json)；
  [`formal config`](../manifests/dstc9_p1_formal_config_v1.json)；
  [`formal runtime`](../replication_runtime/dstc9_p1_formal_v1/runner.py)；
  [`formal source compiler`](../assumption_agent/benchmarks/dstc9_p1_formal_source_v1.py)；
  [`formal controller`](../assumption_agent/benchmarks/dstc9_p1_formal_controller_v1.py)；
  [`runtime-dependency terminal disposition`](../manifests/dstc9_p1_formal_runtime_dependency_failure_disposition_v1.json)

- MultiDoc2Dial P0 local-download/remote-relay terminal chain（member payload/P1/action/score 均为 0）：
  [`source custody`](../manifests/multidoc2dial_p0_public_source_custody_v1.json)；
  [`local acquisition and relay receipt`](../manifests/multidoc2dial_p0_public_source_acquisition_receipt_v1.json)；
  [`source-agnostic typed core`](../assumption_agent/benchmarks/multidoc2dial_p1_typed_core_v1.py)；
  [`public qualification implementation`](../assumption_agent/benchmarks/multidoc2dial_p0_public_source_qualification_v1.py)；
  [`qualification freeze`](../manifests/multidoc2dial_p0_public_source_qualification_freeze_v1.json)；
  [`terminal disposition`](../manifests/multidoc2dial_p0_public_source_qualification_disposition_v1.json)

- TechQA P0 single-download terminal chain（archive payload/P1/action/score 均为 0）：
  [`source custody`](../manifests/techqa_p0_public_source_custody_v1.json)；
  [`qualification freeze`](../manifests/techqa_p0_public_schema_qualification_freeze_v1.json)；
  [`source-free preflight disposition`](../manifests/techqa_p0_public_source_preflight_disposition_v1.json)；
  [`qualification implementation`](../assumption_agent/benchmarks/techqa_p0_public_source_qualification_v1.py)；
  [`source-agnostic typed core`](../assumption_agent/benchmarks/techqa_p1_typed_core_v1.py)；
  [`one-shot P1 runtime`](../assumption_agent/benchmarks/techqa_p1_runtime_v1.py)；
  [`safe terminal result`](../manifests/techqa_p0_public_source_qualification_result_v1.json)

- LoCoMo P0 public non-scoring terminal chain（无 P1/action/score）：
  [`source custody`](../manifests/locomo_p0_public_source_custody_v1.json)；
  [`qualification freeze`](../manifests/locomo_p0_public_schema_qualification_freeze_v1.json)；
  [`qualification implementation`](../assumption_agent/benchmarks/locomo_p0_public_source_qualification_v1.py)；
  [`typed source-agnostic core`](../assumption_agent/benchmarks/locomo_p1_typed_core_v1.py)；
  [`pre-entrypoint disposition`](../manifests/locomo_p0_preentrypoint_deployment_failure_disposition_v1.json)；
  [`snapshot correction addendum`](../manifests/locomo_p0_preentrypoint_deployment_correction_addendum_v2.json)；
  [`safe qualification result`](../manifests/locomo_p0_public_source_qualification_result_v1.json)；
  [`safe terminal`](../manifests/locomo_p0_public_source_qualification_terminal_v1.json)

- HiTab P1 pre-source implementation chain（正式 source/model/action/score 均为 0）：
  [`public source custody`](../manifests/hitab_p1_public_source_custody_v1.json)；
  [`study design`](../manifests/hitab_p1_dmc1_hierarchical_set_evaluator_design_v1.json)；
  [`direct Transformers MiniLM v2 addendum`](../manifests/hitab_p1_direct_transformers_minilm_addendum_v2.json)；
  [`child cwd sanitization v3 addendum`](../manifests/hitab_p1_child_cwd_sanitization_addendum_v3.json)；
  [`sealed child sys.path v4 addendum`](../manifests/hitab_p1_sealed_child_sys_path_addendum_v4.json)；
  [`v4 implementation freeze`](../manifests/hitab_p1_implementation_freeze_v1.json)；
  [`v4 source-free failure disposition`](../manifests/hitab_p1_source_free_canary_v4_failure_disposition.json)；
  [`DMC1 core`](../assumption_agent/benchmarks/hitab_p1_dmc1_core_v1.py)；
  [`runtime`](../assumption_agent/benchmarks/hitab_p1_runtime_v1.py)；
  [`source acquisition`](../assumption_agent/benchmarks/hitab_p1_source_acquisition_v1.py)；
  [`formal controller`](../assumption_agent/benchmarks/hitab_p1_formal_controller_v1.py)；
  [`public canary`](../assumption_agent/benchmarks/hitab_p1_public_canary_v1.py)；
  [`production runner`](../replication_runtime/hitab_p1_formal_v1/runner.py)；
  [`dependency closure`](../replication_runtime/hitab_p1_formal_v1/dependency_closure.py)

- WiCE P0 source-contract-incompatible terminal chain（TEST JSON 未解码；无 P1/效果测量）：
  [`public P0 design`](../manifests/wice_p0_public_schema_qualification_design_v1.json)；
  [`qualification implementation`](../assumption_agent/benchmarks/wice_p0_public_schema_qualification_v1.py)；
  [`source acquisition receipt`](../manifests/wice_p0_public_source_acquisition_receipt_v1.json)；
  [`safe aggregate qualification receipt`](../manifests/wice_p0_public_schema_qualification_receipt_v1.json)；
  [`terminal disposition`](../manifests/wice_p0_public_schema_qualification_failure_disposition_v1.json)

- AVeriTeC P1 valid-negative chain（M_search 未读取/未执行）：
  [`public P0 design`](../manifests/averitec_p0_public_schema_qualification_design_v1.json)；
  [`public P0 receipt`](../manifests/averitec_p0_public_schema_qualification_receipt_v1.json)；
  [`P1 study design`](../manifests/averitec_p1_typed_qa_set_evaluator_design_v1.json)；
  [`execution freeze`](../manifests/averitec_p1_execution_freeze_v1.json)；
  [`launch freeze`](../manifests/averitec_p1_launch_freeze_v1.json)；
  [`formal terminal`](../manifests/averitec_p1_formal_terminal_v1.json)；
  [`offline finalize`](../manifests/averitec_p1_offline_finalize_v1.json)

- EBM-NLP P1 v4 terminal chain（正式 source epoch 已消费；member payload/model/action/gold/score 均为 0）：
  [`source custody`](../manifests/ebmnlp_p1_source_custody_v1.json)；
  [`study design`](../manifests/ebmnlp_p1_typed_pico_set_evaluator_study_design_v1.json)；
  [`v1 failure disposition`](../manifests/ebmnlp_p1_source_free_canary_v1_failure_disposition.json)；
  [`v2 failure disposition`](../manifests/ebmnlp_p1_source_free_canary_v2_failure_disposition.json)；
  [`v3 failure disposition`](../manifests/ebmnlp_p1_source_free_canary_v3_failure_disposition.json)；
  [`implementation freeze v4`](../manifests/ebmnlp_p1_implementation_freeze_v4.json)；
  [`runtime fingerprint v4`](../manifests/ebmnlp_p1_runtime_fingerprint_receipt_v4.json)；
  [`source-free canary live`](../manifests/ebmnlp_p1_source_free_canary_live_receipt_v4.json)；
  [`source-free canary`](../manifests/ebmnlp_p1_source_free_canary_receipt_v4.json)；
  [`execution freeze`](../manifests/ebmnlp_p1_execution_freeze_v4.json)；
  [`formal live receipt`](../manifests/ebmnlp_p1_formal_live_receipt_v4.json)；
  [`safe terminal`](../manifests/ebmnlp_p1_formal_terminal_v4.json)；
  [`aggregate result`](../manifests/ebmnlp_p1_formal_result_v4.json)

- MAUD extraction P2 terminal chain（source JSON 未 parse；formal model/action/gold/score 均为 0）：
  [`source custody`](../manifests/maud_extraction_p2_source_custody_v1.json)；
  [`study design`](../manifests/maud_extraction_p2_cgroup_bounded_evaluator_study_design_v1.json)；
  [`pre-source clarification`](../manifests/maud_extraction_p2_pre_source_clarification_v1.json)；
  [`implementation freeze`](../manifests/maud_extraction_p2_implementation_freeze_v1.json)；
  [`runtime fingerprint`](../manifests/maud_extraction_p2_remote_runtime_fingerprint_v1.json)；
  [`full canary`](../manifests/maud_extraction_p2_full_canary_receipt_v1.json)；
  [`canary unit attestation`](../manifests/maud_extraction_p2_full_canary_unit_attestation_v1.json)；
  [`execution freeze`](../manifests/maud_extraction_p2_execution_freeze_v1.json)；
  [`download attempt`](../manifests/maud_extraction_p2_source_download_attempt_v1.json)；
  [`download terminal`](../manifests/maud_extraction_p2_source_download_terminal_v1.json)；
  [`acquisition failure disposition`](../manifests/maud_extraction_p2_acquisition_failure_disposition_v1.json)

- MAUD extraction P1 pre-source chain（formal source/action/score 仍为 0）：
  [`source custody`](../manifests/maud_extraction_p1_source_custody_v1.json)；
  [`study design`](../manifests/maud_extraction_p1_typed_evaluator_study_design_v1.json)；
  [`pre-source clarification`](../manifests/maud_extraction_p1_pre_source_clarification_v1.json)；
  [`implementation freeze v1`](../manifests/maud_extraction_p1_implementation_freeze_v1.json)；
  [`runtime fingerprint failure disposition`](../manifests/maud_extraction_p1_runtime_fingerprint_failure_disposition_v1.json)；
  [`implementation freeze v2`](../manifests/maud_extraction_p1_implementation_freeze_v2.json)；
  [`source-free runtime fingerprint`](../manifests/maud_extraction_p1_remote_runtime_fingerprint_v1.json)；
  [`full-canary terminal`](../manifests/maud_extraction_p1_full_canary_terminal_v1.json)

- MMQA P1 pre-source frozen chain（formal source/model/action/score 仍为 0）：
  [`source custody`](../manifests/mmqa_p1_source_custody_v1.json)；
  [`study design`](../manifests/mmqa_p1_local_proof_e5_study_design_v1.json)；
  [`download authorization`](../manifests/mmqa_p1_source_download_authorization_v1.json)；
  [`pre-execution runtime disposition`](../manifests/mmqa_p1_preexecution_runtime_disposition_v1.json)；
  [`official pre-capability launch disposition`](../manifests/mmqa_p1_official_preflight_launch_disposition_v1.json)；
  [`official preflight terminal`](../manifests/mmqa_p1_official_preflight_terminal_v1.json)；
  [`static postmortem`](../manifests/mmqa_p1_official_preflight_static_postmortem_v1.json)

- BIRCO P1 qualified source chain（formal item/model/action/score 尚为 0）：
  [`source custody`](../manifests/birco_p1_source_custody_v1.json)；
  [`study design`](../manifests/birco_p1_typed_constraint_e4_study_design_v1.json)；
  [`download authorization`](../manifests/birco_p1_source_download_authorization_v1.json)；
  [`download receipt`](../manifests/birco_p1_source_download_receipt_v1.json)；
  [`qualification freeze`](../manifests/birco_p1_source_qualification_freeze_v1.json)；
  [`qualification result`](../manifests/birco_p1_source_qualification_result_v1.json)；
  [`qualification marker`](../artifacts/birco_p1_source_qualification_v1/qualification.one_shot_marker.json)；
  [`source-open marker`](../artifacts/birco_p1_source_qualification_v1/source_open.one_shot_marker.json)

- BIRCO P1 formal terminal chain（protocol-valid；provider-degraded；A_hold/RAW/HippoRAG/M_search 未进入）：
  [`provider preflight`](../manifests/birco_p1_provider_preflight_selection_v1.json)；
  [`HippoRAG runtime preflight`](../manifests/birco_p1_hipporag_runtime_preflight_v1.json)；
  [`initial implementation freeze`](../manifests/birco_p1_implementation_freeze_v1.json)；
  [`preselection entrypoint failure disposition`](../manifests/birco_p1_preselection_entrypoint_failure_disposition_v1.json)；
  [`superseding implementation freeze`](../manifests/birco_p1_implementation_freeze_v2.json)；
  [`public selection receipt`](../manifests/birco_p1_private_selection_receipt_v1.json)；
  [`execution freeze`](../manifests/birco_p1_execution_freeze_v1.json)；
  [`A_form E4 model receipt`](../manifests/birco_p1_A_form_e4_model_v1.json)；
  [`F_search identifiability receipt`](../manifests/birco_p1_F_search_identifiability_v1.json)；
  [`formal terminal`](../manifests/birco_p1_formal_terminal_v1.json)；
  [`safe aggregate result`](../manifests/birco_p1_formal_result_v1.json)

- FanOutQA P1 source-contract terminal chain（TEST/model/action/score 均为 0）：
  [`source custody`](../manifests/fanoutqa_p1_source_custody_v1.json)；
  [`study design`](../manifests/fanoutqa_p1_typed_fanout_e3_study_design_v1.json)；
  [`download authorization`](../manifests/fanoutqa_p1_source_download_authorization_v1.json)；
  [`download receipt`](../manifests/fanoutqa_p1_source_download_receipt_v1.json)；
  [`prequalification hardening amendment`](../manifests/fanoutqa_p1_prequalification_hardening_amendment_v1.json)；
  [`qualification freeze`](../manifests/fanoutqa_p1_source_qualification_freeze_v1.json)；
  [`terminal observation`](../manifests/fanoutqa_p1_source_qualification_terminal_observation_v1.json)；
  [`qualification marker`](../artifacts/fanoutqa_p1_source_qualification_v1/qualification.one_shot_marker.json)；
  [`source-open marker`](../artifacts/fanoutqa_p1_source_qualification_v1/source_open.one_shot_marker.json)；
  [`terminal failure`](../artifacts/fanoutqa_p1_source_qualification_v1/qualification.terminal_failure.json)
- FiQA TRAIN P10/P11 formation 与 DEV comparator-invalid chain：
  [`TRAIN source integration`](../manifests/fiqa_bridge_expansion_train_integration_result_v2.json)；
  [`P10 TRAIN runtime`](../manifests/fiqa_bridge_expansion_train_runtime_result_v2.json)；
  [`P11 formation`](../manifests/fiqa_p11_formation_result_v1.json)；
  [`DEV acquisition`](../manifests/fiqa_bridge_expansion_dev_acquisition_result_v1.json)；
  [`DEV runtime failure`](../manifests/fiqa_bridge_expansion_dev_runtime_failure_v1.json)；
  [`HippoRAG hardening qualification`](../manifests/hipporag_upstream_hardening_qualification_result_v1.json)
- NanoBEIR P11/P12 与 complete-case comparator chain：
  [`P11 source qualification`](../manifests/nanobeir_p11_source_access_v1.json)；
  [`P11 private acquisition`](../manifests/nanobeir_p11_acquisition_result_v1.json)；
  [`P11 generation failure`](../manifests/nanobeir_p11_c_confirm_runtime_failure_v1.json)；
  [`P12 source qualification`](../manifests/nanobeir_p12_source_access_v2.json)；
  [`P12 private acquisition`](../manifests/nanobeir_p12_acquisition_result_v1.json)；
  [`P12 comparator failure`](../manifests/nanobeir_p12_c_confirm_runtime_failure_v1.json)；
  [`complete-case source qualification`](../manifests/nanobeir_p12_completecase_source_access_v2.json)；
  [`150-item availability result`](../manifests/nanobeir_p12_completecase_availability_result_v1.json)；
  [`corrected complete-case acquisition`](../manifests/nanobeir_p12_completecase_acquisition_result_v1.json)；
  [`P12 bridge-query failure`](../manifests/nanobeir_p12_completecase_c_confirm_runtime_failure_v1.json)；
  [`P13 bridge-safe formation`](../manifests/nanobeir_p13_bridge_safe_formation_v1.json)；
  [`P13 candidate freeze`](../manifests/nanobeir_p13_candidate_freeze_v1.json)；
  [`P13 source qualification`](../manifests/nanobeir_p13_source_access_v2.json)；
  [`P13 availability result`](../manifests/nanobeir_p13_availability_result_v1.json)；
  [`P13 private acquisition`](../manifests/nanobeir_p13_acquisition_result_v1.json)；
  [`P13 base-pool contract failure`](../manifests/nanobeir_p13_c_confirm_runtime_failure_v1.json)
- BRIGHT P14→P17 candidate-specific all-remote terminal chain（P17 labels/scores 未打开）：
  [`P14 source custody`](../manifests/bright_p14_source_custody_v1.json)；
  [`P14 complete-case design`](../manifests/bright_p14_direct_completecase_study_design_v1.json)；
  [`P14 private acquisition`](../manifests/bright_p14_acquisition_result_v1.json)；
  [`P14 direct freeze`](../manifests/bright_p14_direct_c_confirm_freeze_v1.json)；
  [`P14 interruption disposition`](../manifests/bright_p14_direct_c_confirm_interruption_disposition_v1.json)；
  [`P15 all-remote design`](../manifests/bright_p15_all_remote_c_confirm_study_design_v1.json)；
  [`P15 view-only extension`](../manifests/bright_p15_extension_acquisition_result_v1.json)；
  [`P15 runtime fingerprint`](../manifests/bright_p15_remote_runtime_fingerprint_v1.json)；
  [`P16 wired design`](../manifests/bright_p16_all_remote_c_confirm_study_design_v1.json)；
  [`P16 runtime fingerprint`](../manifests/bright_p16_remote_runtime_fingerprint_v1.json)；
  [`P16 source-capacity disposition`](../manifests/bright_p16_extension_acquisition_disposition_v1.json)；
  [`P17 capacity-feasible design`](../manifests/bright_p17_all_remote_c_confirm_study_design_v1.json)；
  [`P17 view-only extension`](../manifests/bright_p17_extension_acquisition_result_v1.json)；
  [`P17 implementation freeze`](../manifests/bright_p17_all_remote_implementation_freeze_v1.json)；
  [`P17 corrected runtime fingerprint`](../manifests/bright_p17_remote_runtime_fingerprint_v1.json)；
  [`P17 prelaunch runtime disposition`](../manifests/bright_p17_prelaunch_runtime_disposition_v1.json)；
  [`P17 remote execution plan`](../artifacts/bright_p17_all_remote_c_confirm_v1/remote_execution.plan.json)；
  [`P17 minimal remote archive`](../artifacts/bright_p17_all_remote_c_confirm_v1/remote_archive/)；
  [`P17 forensic action/receipt/trace archive`](../artifacts/bright_p17_all_remote_c_confirm_v1/remote_forensic_archive/)；
  [`P17 prelabel execution-contract-invalid result`](../manifests/bright_p17_all_remote_c_confirm_result_v1.json)
- TAT-QA P18 source-free qualification-invalid chain（official source 未下载/未打开）：
  [`P18 design`](../manifests/tatqa_p18_typed_evaluator_study_design_v1.json)；
  [`P18 source custody`](../manifests/tatqa_p18_public_source_custody_v1.json)；
  [`P18 qualification marker`](../artifacts/tatqa_p18_runtime_qualification_v1/qualification.one_shot_marker.json)；
  [`P18 terminal qualification failure`](../artifacts/tatqa_p18_runtime_qualification_v1/qualification.terminal_failure.json)
- TAT-QA P19 split-runtime launch-infrastructure-invalid chain（official source 未下载/未打开）：
  [`P19 design`](../manifests/tatqa_p19_typed_evaluator_study_design_v1.json)；
  [`P19 source custody`](../manifests/tatqa_p19_public_source_custody_v1.json)；
  [`P19 HippoRAG runtime attestation`](../manifests/tatqa_p19_hipporag_runtime_attestation_v1.json)；
  [`P19 qualification marker`](../artifacts/tatqa_p19_runtime_qualification_v1/qualification.one_shot_marker.json)；
  [`P19 terminal qualification failure`](../artifacts/tatqa_p19_runtime_qualification_v1/qualification.terminal_failure.json)
- TAT-QA P20 post-inventory environment-validation-invalid chain（official source 未下载/未打开）：
  [`P20 design`](../manifests/tatqa_p20_typed_evaluator_study_design_v1.json)；
  [`P20 source custody`](../manifests/tatqa_p20_public_source_custody_v1.json)；
  [`P20 qualification marker`](../artifacts/tatqa_p20_runtime_qualification_v1/qualification.one_shot_marker.json)；
  [`P20 terminal qualification failure`](../artifacts/tatqa_p20_runtime_qualification_v1/qualification.terminal_failure.json)
- TAT-QA P21 hardware-fragile MiniLM canary-invalid chain（official source 未下载/未打开）：
  [`P21 design`](../manifests/tatqa_p21_typed_evaluator_study_design_v1.json)；
  [`P21 source custody`](../manifests/tatqa_p21_public_source_custody_v1.json)；
  [`P21 successful composite runtime fingerprint`](../manifests/tatqa_p21_composite_runtime_fingerprint_v1.json)；
  [`P21 qualification marker`](../artifacts/tatqa_p21_runtime_qualification_v1/qualification.one_shot_marker.json)；
  [`P21 terminal qualification failure`](../artifacts/tatqa_p21_runtime_qualification_v1/qualification.terminal_failure.json)
- TAT-QA P22 portable-MiniLM feasibility environment-contract-invalid chain（official source 未下载/未打开）：
  [`P22 one-shot feasibility marker`](../artifacts/tatqa_p22_source_free_feasibility_v1/feasibility.one_shot_marker.json)；
  [`P22 terminal feasibility failure`](../artifacts/tatqa_p22_source_free_feasibility_v1/feasibility.terminal_failure.json)
- TAT-QA P23 final source-free typed-action feasibility-negative chain（official source 未下载/未打开）：
  [`P23 one-shot feasibility marker`](../artifacts/tatqa_p23_source_free_feasibility_v1/feasibility.one_shot_marker.json)；
  [`P23 terminal feasibility failure`](../artifacts/tatqa_p23_source_free_feasibility_v1/feasibility.terminal_failure.json)；
  [`P23 source-free postmortem`](../artifacts/tatqa_p23_source_free_feasibility_v1/feasibility.postmortem.json)；
  [`P23 public Qwen input`](../artifacts/tatqa_p23_source_free_feasibility_v1/qwen_public_canary_repeat_1.input.json)；
  [`P23 public Qwen output`](../artifacts/tatqa_p23_source_free_feasibility_v1/qwen_public_canary_repeat_1.output.json)
- BRIGHT P17 post-terminal query/view exposure disposition：
  [`P17 post-terminal view exposure disposition`](../manifests/bright_p17_postterminal_view_exposure_disposition_v1.json)
- FRAMES P1 source-before-download freeze：
  [`FRAMES P1 source custody`](../manifests/frames_p1_source_custody_v1.json)；
  [`FRAMES P1 aggregate-only qualifier`](../assumption_agent/benchmarks/frames_p1_source_qualification_v1.py)；
  [`FRAMES P1 implementation freeze`](../manifests/frames_p1_source_qualification_freeze_v1.json)；
  [`FRAMES P1 source download receipt`](../manifests/frames_p1_source_download_receipt_v1.json)；
  [`FRAMES P1 one-shot marker`](../artifacts/frames_p1_source_qualification_v1/qualification.one_shot_marker.json)；
  [`FRAMES P1 terminal header failure`](../artifacts/frames_p1_source_qualification_v1/qualification.terminal_failure.json)
- BRIGHT reasoning-retrieval v3、terminal M failure 与 fresh RESERVE 三臂链：
  [`v3 executor design`](../manifests/bright_reasoning_retrieval_executor_repair_design_v3.json)；
  [`v3 implementation freeze`](../manifests/bright_reasoning_retrieval_study_implementation_freeze_v3.json)；
  [`G_form`](../manifests/bright_reasoning_retrieval_G_form_v3.json)；
  [`A_form`](../manifests/bright_reasoning_retrieval_A_form_v3.json)；
  [`F_search`](../manifests/bright_reasoning_retrieval_F_search_v3.json)；
  [`A_hold`](../manifests/bright_reasoning_retrieval_A_hold_v3.json)；
  [`M_search infrastructure-invalid receipt`](../manifests/bright_reasoning_retrieval_M_search_failure_v3.json)；
  [`reserve design`](../manifests/bright_reasoning_retrieval_reserve_measurement_design_v1.json)；
  [`reserve freeze`](../manifests/bright_reasoning_retrieval_reserve_measurement_implementation_freeze_v1.json)；
  [`acquisition failure receipt`](../manifests/bright_reasoning_retrieval_reserve_acquisition_failure_v1.json)；
  [`result-blind recovery design`](../manifests/bright_reasoning_retrieval_reserve_acquisition_recovery_design_v1.json)；
  [`recovery freeze`](../manifests/bright_reasoning_retrieval_reserve_acquisition_recovery_implementation_freeze_v1.json)；
  [`recovered acquisition result`](../manifests/bright_reasoning_retrieval_reserve_acquisition_result_v1.json)；
  [`prepare result`](../manifests/bright_reasoning_retrieval_reserve_prepare_result_v1.json)；
  [`action seal`](../manifests/bright_reasoning_retrieval_reserve_actions_result_v1.json)；
  [`final three-arm result`](../manifests/bright_reasoning_retrieval_reserve_final_result_v1.json)；
  [`cross-encoder asset`](../manifests/bright_cross_encoder_runtime_asset_v1.json)；
  [`cross-encoder runtime freeze`](../manifests/bright_cross_encoder_runtime_implementation_freeze_v1.json)；
  [`P9 consumed-TRAIN45 formation result`](../manifests/bright_reasoning_retrieval_cross_encoder_formation_result_v1.json)；
  [`P9 prospective design`](../manifests/bright_reasoning_retrieval_p9_confirmation_design_v1.json)；
  [`P9 implementation freeze`](../manifests/bright_reasoning_retrieval_p9_confirmation_implementation_freeze_v1.json)；
  [`P9 acquisition`](../manifests/bright_reasoning_retrieval_p9_confirmation_acquisition_result_v1.json)；
  [`P9 prepare`](../manifests/bright_reasoning_retrieval_p9_confirmation_prepare_result_v1.json)；
  [`P9 action seal`](../manifests/bright_reasoning_retrieval_p9_confirmation_actions_result_v1.json)；
  [`P9 prospective final result`](../manifests/bright_reasoning_retrieval_p9_confirmation_final_result_v1.json)

- EntailmentBank proof-retrieval qualification、v1 fail-closed、v2 formal terminal chain：
  [`source custody`](../manifests/entailmentbank_proof_retrieval_source_custody_v1.json)；
  [`source access`](../manifests/entailmentbank_proof_retrieval_source_access_v1.json)；
  [`qualification design`](../manifests/entailmentbank_proof_retrieval_source_qualification_design_v1.json)；
  [`qualification implementation freeze`](../manifests/entailmentbank_proof_retrieval_source_qualification_implementation_freeze_v1.json)；
  [`qualification result`](../manifests/entailmentbank_proof_retrieval_source_qualification_result_v1.json)；
  [`formal design v1`](../manifests/entailmentbank_proof_retrieval_g1_e1_formal_design_v1.json)；
  [`implementation freeze v1`](../manifests/entailmentbank_proof_retrieval_g1_e1_implementation_freeze_v1.json)；
  [`selection custody v1`](../manifests/entailmentbank_proof_retrieval_selection_secret_custody_v1.json)；
  [`v1 acquisition failure`](../manifests/entailmentbank_proof_retrieval_acquisition_failure_v1.json)；
  [`v2 remediation design`](../manifests/entailmentbank_proof_retrieval_g1_e1_formal_design_v2.json)；
  [`v2 implementation freeze`](../manifests/entailmentbank_proof_retrieval_g1_e1_implementation_freeze_v2.json)；
  [`selection custody v2`](../manifests/entailmentbank_proof_retrieval_selection_secret_custody_v2.json)；
  [`v2 acquisition receipt`](../manifests/entailmentbank_proof_retrieval_acquisition_receipt_v2.json)；
  [`formation result`](../manifests/entailmentbank_proof_retrieval_g1_e1_formation_result_v2.json)；
  [`A_hold result`](../manifests/entailmentbank_proof_retrieval_g1_e1_ahold_result_v2.json)；
  [`post-terminal Q0 descriptive audit`](../manifests/entailmentbank_proof_retrieval_q0_postterminal_descriptive_audit_v1.json)；
  [`final result`](../manifests/entailmentbank_proof_retrieval_g1_e1_final_result_v2.json)

- SciFact direct-evidence source terminal chain（无 selection/action/score，TEST payload 未打开）：
  [`source custody`](../manifests/scifact_direct_evidence_source_custody_v1.json)；
  [`source qualification design`](../manifests/scifact_direct_evidence_source_qualification_design_v1.json)；
  [`source qualifier`](../assumption_agent/benchmarks/scifact_direct_evidence_source_qualification_v1.py)；
  [`implementation freeze`](../manifests/scifact_direct_evidence_source_qualification_implementation_freeze_v1.json)；
  [`aggregate terminal result`](../manifests/scifact_direct_evidence_source_qualification_result_v1.json)

- MAVEN-ERE G8/E1 formal/recovery chain：
  [`source qualification design`](../manifests/maven_ere_relation_context_source_qualification_design_v1.json)；
  [`source qualification result`](../manifests/maven_ere_relation_context_source_qualification_result_v1.json)；
  [`formal design`](../manifests/maven_ere_g8_e1_formal_design_v1.json)；
  [`formal implementation freeze`](../manifests/maven_ere_g8_e1_implementation_freeze_v1.json)；
  [`acquisition result`](../manifests/maven_ere_g8_e1_acquisition_result_v1.json)；
  [`v1 failure disposition`](../manifests/maven_ere_g8_e1_formal_v1_implementation_failure_disposition_v1.json)；
  [`result-blind recovery design`](../manifests/maven_ere_g8_e1_result_blind_recovery_design_v2.json)；
  [`recovery implementation freeze`](../manifests/maven_ere_g8_e1_result_blind_recovery_implementation_freeze_v2.json)；
  [`recovery result disposition`](../manifests/maven_ere_g8_e1_result_blind_recovery_result_disposition_v2.json)；
  [`recovery terminal`](../artifacts/maven_ere_g8_e1_result_blind_recovery_v2/controller/recovery.terminal_result.json)；
  [`E2 TRAIN design`](../manifests/maven_ere_global_family_e2_train_crossfit_design_v1.json)；
  [`E2 implementation freeze`](../manifests/maven_ere_global_family_e2_train_crossfit_implementation_freeze_v1.json)；
  [`E2 result disposition`](../manifests/maven_ere_global_family_e2_train_crossfit_result_disposition_v1.json)；
  [`E2 cross-fit result`](../artifacts/maven_ere_global_family_e2_train_crossfit_v1/crossfit.result.json)；
  [`fresh G8 confirmation design`](../manifests/maven_ere_g8_e0_fresh_confirmation_design_v1.json)；
  [`fresh G8 implementation freeze`](../manifests/maven_ere_g8_e0_fresh_confirmation_implementation_freeze_v1.json)；
  [`fresh G8 result disposition`](../manifests/maven_ere_g8_e0_fresh_confirmation_result_disposition_v1.json)；
  [`fresh G8 terminal`](../artifacts/maven_ere_g8_e0_fresh_confirmation_v1/controller/terminal.result.json)

- ERASER Evidence Inference R7/E3 formal/recovery chain：
  [`base design`](../manifests/eraser_evidence_inference_r7_e3_design_v1.json)；
  [`v1 full implementation freeze`](../manifests/eraser_evidence_inference_full_implementation_freeze_v1.json)；
  [`hard-interruption incident`](../manifests/eraser_evidence_inference_formal_v1_hard_interruption_incident_v1.json)；
  [`crash-recovery design`](../manifests/eraser_evidence_inference_crash_recovery_design_v2.json)；
  [`recovery implementation freeze`](../manifests/eraser_evidence_inference_crash_recovery_implementation_freeze_v2.json)；
  [`monitoring metadata incident`](../manifests/eraser_evidence_inference_recovery_monitoring_metadata_incident_v1.json)；
  [`recovery result disposition`](../manifests/eraser_evidence_inference_r7_e3_recovery_v2_result_disposition_v1.json)；
  [`recovery terminal`](../artifacts/eraser_evidence_inference_r7_e3_crash_recovery_v2/controller/recovery.terminal_result.json)

- HybridQA P6/E2 formal chain：
  [`v1 implementation failure disposition`](../manifests/hybridqa_p6_e2_formal_v1_implementation_failure_disposition_v1.json)；
  [`v2 DEV design`](../manifests/hybridqa_p6_e2_design_v2.json)；
  [`v2 implementation freeze`](../manifests/hybridqa_p6_e2_implementation_freeze_v2.json)；
  [`v2 acquisition`](../assumption_agent/benchmarks/hybridqa_direct_acquisition_v2.py)；
  [`v2 formal controller`](../assumption_agent/benchmarks/hybridqa_p6_e2_formal_controller_v2.py)；
  [`v2 result disposition`](../manifests/hybridqa_p6_e2_formal_v2_result_disposition_v1.json)；
  [`v2 terminal result`](../artifacts/hybridqa_p6_e2_formal_v2/controller/lifecycle.terminal_result.json)

- FEVEROUS source epoch v3 qualification terminal（无 v3 secret/cohort/action/score）：
  [`typed adapter`](../assumption_agent/benchmarks/feverous_p6_e2_source_adapter_v1.py)；
  [`aggregate-only qualifier`](../assumption_agent/benchmarks/feverous_p6_e2_adapter_compatibility_qualification_v3.py)；
  [`terminal failure receipt`](../manifests/feverous_p6_e2_adapter_compatibility_qualification_v3_terminal_failure.json)

- HoVer joint graph/evaluator formal chain（有效 A_hold non-promotion；M_search 未打开）：
  [`source custody`](../manifests/hover_source_custody_v1.json)；
  [`source access`](../manifests/hover_source_access_v1.json)；
  [`source qualification`](../manifests/hover_source_qualification_v1.json)；
  [`design`](../manifests/hover_joint_graph_evaluator_design_v1.json)；
  [`implementation freeze`](../manifests/hover_joint_graph_implementation_freeze_v1.json)；
  [`acquisition receipt`](../manifests/hover_direct_acquisition_v1_acquisition.json)；
  [`A_form action seal`](../manifests/hover_a_form_action_seal_v1.json)；
  [`A_form evaluator freeze`](../manifests/hover_a_form_evaluator_freeze_v1.json)；
  [`F_search policy freeze`](../manifests/hover_f_search_policy_freeze_v1.json)；
  [`A_hold action seal`](../manifests/hover_a_hold_action_seal_v1.json)；
  [`terminal result`](../artifacts/hover_joint_graph_formal_v1/formal_result.json)

- QASPER source qualification implementation-invalid chain（无 selection/retrieval/score）：
  [`source custody`](../manifests/qasper_graph_evaluator_source_custody_v1.json)；
  [`aggregate qualifier`](../assumption_agent/benchmarks/qasper_fresh_source_qualification_v03.py)；
  [`terminal disposition`](../manifests/qasper_graph_evaluator_source_qualification_failure_disposition_v1.json)；
  [`reusable offline MiniLM asset`](../manifests/qasper_minilm_runtime_asset_v1.json)
- FinQA source qualification implementation-invalid chain（无 receipt/private block/action/score）：
  [`source custody`](../manifests/finqa_graph_evaluator_source_custody_v1.json)；
  [`source archive binding`](../manifests/finqa_source_access_addendum_v1.json)；
  [`frozen graph/evaluator design`](../manifests/finqa_graph_evaluator_design_v1.json)；
  [`aggregate qualifier`](../assumption_agent/benchmarks/finqa_fresh_source_qualification_v1.py)；
  [`terminal disposition`](../manifests/finqa_source_qualification_failure_disposition_v1.json)

- ContractNLI source qualification terminal chain（efficacy/source feasibility unknown）：
  [`source custody`](../manifests/contractnli_graph_evaluator_source_custody_v1.json)；
  [`frozen graph/evaluator design`](../manifests/contractnli_graph_evaluator_design_v1.json)；
  [`source-access addendum`](../manifests/contractnli_source_access_addendum_v1.json)；
  [`TRAIN member binding`](../manifests/contractnli_source_member_binding_v1.json)；
  [`terminal disposition`](../manifests/contractnli_source_qualification_failure_disposition_v1.json)

- CUAD direct-acquisition terminal chain（232/256 capacity shortfall；无 block/action/score）：
  [`frozen design`](../manifests/cuad_graph_evaluator_design_v1.json)；
  [`source custody`](../manifests/cuad_graph_evaluator_source_custody_v1.json)；
  [`source-access binding`](../manifests/cuad_graph_evaluator_source_access_v1.json)；
  [`pre-marker schema incident`](../manifests/cuad_pre_marker_invocation_incident_v1.json)；
  [`aggregate acquisition receipt`](../manifests/cuad_graph_evaluator_acquisition_v1.json)；
  [`terminal capacity disposition`](../manifests/cuad_graph_evaluator_capacity_disposition_v1.json)

- EvidenceBench direct-acquisition terminal chain（root contract invalid；无 block/action/score）：
  [`frozen design`](../manifests/evidencebench_graph_evaluator_design_v1.json)；
  [`source custody`](../manifests/evidencebench_graph_evaluator_source_custody_v1.json)；
  [`source-access binding`](../manifests/evidencebench_graph_evaluator_source_access_v1.json)；
  [`implementation freeze`](../manifests/evidencebench_implementation_freeze_v1.json)；
  [`aggregate terminal receipt`](../manifests/evidencebench_direct_acquisition_v1.json)；
  [`terminal disposition`](../manifests/evidencebench_acquisition_terminal_disposition_v1.json)

- synthetic typed-graph causal study terminal chain（valid non-promotion；M_search 未打开）：
  [`public grammar`](../assumption_agent/benchmarks/synthetic_typed_graph_causal_grammar_v1.py)；
  [`row-free tests`](../tests/test_synthetic_typed_graph_causal_grammar_v1.py)；
  [`frozen causal design`](../manifests/synthetic_typed_graph_causal_design_v1.json)；
  [`pre-seed amendment`](../manifests/synthetic_typed_graph_causal_preseed_amendment_v1.json)；
  [`formal implementation`](../assumption_agent/benchmarks/synthetic_typed_graph_causal_acquisition_v1.py)；
  [`formal runner`](../assumption_agent/benchmarks/synthetic_typed_graph_causal_runner_v1.py)；
  [`implementation freeze`](../manifests/synthetic_typed_graph_causal_implementation_freeze_v1.json)；
  [`seed custody`](../manifests/synthetic_typed_graph_causal_seed_custody_v1.json)；
  [`256-item acquisition receipt`](../manifests/synthetic_typed_graph_causal_acquisition_v1.json)；
  [`formation receipt`](../manifests/synthetic_typed_graph_causal_formation_v1.json)；
  [`A_hold non-promotion receipt`](../manifests/synthetic_typed_graph_causal_A_hold_v1.json)；
  [`terminal exact seed/cohort publication`](../published/synthetic_typed_graph_causal_v1/formal_seed_and_cohort.json)

- synthetic typed-graph post-terminal 8-seed replication（MiniLM full-call bound；action/score=0；stability unknown）：
  [`frozen replication design`](../manifests/synthetic_typed_graph_multiseed_replication_design_v1.json)；
  [`formal acquisition/publication implementation`](../assumption_agent/benchmarks/synthetic_typed_graph_multiseed_acquisition_v1.py)；
  [`formal three-arm runner`](../assumption_agent/benchmarks/synthetic_typed_graph_multiseed_runner_v1.py)；
  [`implementation freeze`](../manifests/synthetic_typed_graph_multiseed_replication_implementation_freeze_v1.json)；
  [`eight-seed custody`](../manifests/synthetic_typed_graph_multiseed_replication_seed_custody_v1.json)；
  [`512-item acquisition receipt`](../manifests/synthetic_typed_graph_multiseed_replication_acquisition_v1.json)；
  [`canonical infrastructure-invalid result`](../manifests/synthetic_typed_graph_multiseed_replication_result_v1.json)；
  [`terminal exact seeds/cohort publication`](../published/synthetic_typed_graph_multiseed_replication_v1/formal_seeds_and_cohort.json)

- FEVER fixed-P real-source acquisition terminal chain（wiki member contract invalid；无 pack/action/score）：
  [`source custody`](../manifests/fever_official_fixed_transfer_source_custody_v1.json)；
  [`source-access binding`](../manifests/fever_official_fixed_transfer_source_access_v1.json)；
  [`fixed-P item-local design`](../manifests/fever_fixed_p_itemlocal_reranking_design_v1.json)；
  [`acquisition implementation`](../assumption_agent/benchmarks/fever_fixed_p_itemlocal_acquisition_v1.py)；
  [`fixed-P runner`](../assumption_agent/benchmarks/fever_fixed_p_itemlocal_runner_v1.py)；
  [`implementation freeze`](../manifests/fever_official_fixed_transfer_implementation_freeze_v1.json)；
  [`selection custody`](../manifests/fever_official_fixed_transfer_selection_custody_v1.json)；
  [`aggregate failure receipt`](../manifests/fever_official_fixed_transfer_acquisition_failure_v1.json)；
  [`terminal disposition`](../manifests/fever_official_fixed_transfer_acquisition_terminal_disposition_v1.json)

- QASC direct-action evaluator public chain（valid non-promotion；M_search 未打开）：
  [`source custody`](../manifests/qasc_fresh_source_custody_v1.json)；
  [`source-access addendum`](../manifests/qasc_source_access_addendum_v2.json)；
  [`outcome-blind source qualification`](../manifests/qasc_fresh_source_qualification_v1.json)；
  [`NLI runtime asset`](../manifests/qasc_nli_runtime_asset_v1.json)；
  [`frozen design`](../manifests/qasc_evaluator_direct_action_coevolution_design_v1.json)；
  [`row-free infrastructure diagnostic`](../manifests/qasc_evaluator_direct_action_infrastructure_diagnostic_v1.json)；
  [`row-zero acquisition preregistration`](../manifests/qasc_evaluator_direct_action_acquisition_v1_preregistration.json)；
  [`256-item acquisition receipt`](../manifests/qasc_evaluator_direct_action_acquisition_v1_acquisition.json)；
  [`formation pre-run freeze`](../manifests/qasc_evaluator_direct_action_formation_pre_run_freeze_v1.json)；
  [`formation receipt`](../manifests/qasc_evaluator_direct_action_formation_receipt_v1.json)；
  [`A_hold pre-run freeze`](../manifests/qasc_evaluator_direct_action_a_hold_pre_run_freeze_v1.json)；
  [`A_hold aggregate report`](../manifests/qasc_evaluator_direct_action_a_hold_aggregate_report_v1.json)；
  [`terminal disposition`](../manifests/qasc_evaluator_direct_action_coevolution_disposition_v1.json)。
  没有 M_search freeze/report/execution root：A_hold 未晋升，M_search 未授权且未打开

- fresh 2Wiki fixed-action transfer public chain（M_search 未打开）：
  [`source custody`](../manifests/twowiki_fresh_source_custody_v1.json)；
  [`outcome-blind source qualification`](../manifests/twowiki_fresh_source_qualification_v2.json)；
  [`source-access addendum`](../manifests/twowiki_source_access_addendum_v3.json)；
  [`fixed transfer design`](../manifests/twowiki_evaluator_zero_shot_transfer_design_v1.json)；
  [`eager 2×192 infrastructure diagnostic`](../manifests/twowiki_evaluator_zero_shot_transfer_infrastructure_diagnostic_v1.json)；
  [`row-zero acquisition preregistration`](../manifests/twowiki_evaluator_zero_shot_transfer_acquisition_v1_preregistration.json)；
  [`72-item acquisition receipt`](../manifests/twowiki_evaluator_zero_shot_transfer_acquisition_v1_acquisition.json)；
  [`A_hold pre-run freeze`](../manifests/twowiki_evaluator_zero_shot_transfer_a_hold_pre_run_freeze_v1.json)；
  [`A_hold aggregate report`](../manifests/twowiki_evaluator_zero_shot_transfer_a_hold_aggregate_report_v1.json)；
  [`terminal disposition`](../manifests/twowiki_evaluator_zero_shot_transfer_disposition_v1.json)

- MuSiQue fresh generation-one public chain（旧 6-item cohort 未重放）：
  [`v2 runtime attestation`](../manifests/musique_official_hipporag_runtime_attestation_v2.json)；
  [`96-item official-DEV preregistration`](../manifests/musique_recursive_evaluator_study_v1_preregistration.json)；
  [`96-item acquisition receipt`](../manifests/musique_recursive_evaluator_study_v1_acquisition.json)；
  [`F1 formation receipt`](../manifests/musique_recursive_study_f1_formation_v1/formation.receipt.json)；
  [`F1 frozen program`](../manifests/musique_recursive_study_f1_formation_v1/frozen_program.json)；
  [`M1 one-shot pre-run freeze`](../manifests/musique_recursive_study_m1_pre_run_freeze_v1.json)；
  [`M1 aggregate result / promotion disposition / postflight`](../manifests/musique_recursive_study_m1_aggregate_report_v1.json)；
  [`M2 infrastructure failure disposition`](../manifests/musique_recursive_study_m2_infrastructure_failure_disposition_v1.json)；
  [`A_form evaluator receipt`](../manifests/musique_recursive_evaluator_a_form_receipt_v1.json)；
  [`F3 search-formation receipt`](../manifests/musique_recursive_evaluator_f3_receipt_v1.json)；
  [`A_hold anchor report`](../manifests/musique_recursive_evaluator_a_hold_report_v1.json)；
  [`M3 prospective utility report`](../manifests/musique_recursive_evaluator_m3_report_v1.json)；
  [`L5 no-transition disposition`](../manifests/musique_recursive_evaluator_coevolution_disposition_v1.json)
- MuSiQue→HotpotQA frozen-P family-out public chain：
  [`bubblewrap capability receipt`](../manifests/hotpot_family_out_bubblewrap_capability_v1.json)；
  [`row-zero preregistration`](../manifests/hotpot_family_out_preregistration_v1.json)；
  [`private-HMAC acquisition receipt`](../manifests/hotpot_family_out_acquisition_v1.json)；
  [`one-shot pre-run freeze`](../manifests/hotpot_family_out_pre_run_freeze_v1.json)；
  [`36/36 aggregate offline result`](../manifests/hotpot_family_out_aggregate_report_v1.json)
- fresh Hotpot retained-recursion / evaluator public chain（旧 12-item cohort 已排除）：
  [`24-worker synthetic capacity diagnostic`](../manifests/hotpot_recursive_official_capacity24_diagnostic_v1.json)；
  [`row-zero six-block preregistration`](../manifests/hotpot_recursive_study_v1_preregistration.json)；
  [`156-item private-HMAC acquisition receipt`](../manifests/hotpot_recursive_study_v1_acquisition.json)；
  [`F_Q formation receipt`](../manifests/hotpot_recursive_study_fq_formation_v1/formation.receipt.json)；
  [`frozen Q program`](../manifests/hotpot_recursive_study_fq_formation_v1/frozen_program.json)；
  [`M_L4 pre-run freeze`](../manifests/hotpot_recursive_study_l4_pre_run_freeze_v1.json)；
  [`M_L4 96/96 aggregate report`](../manifests/hotpot_recursive_study_l4_aggregate_report_v1.json)；
  [`A_form behavior-distinct challenger receipt`](../manifests/hotpot_recursive_evaluator_a_form_receipt_v2.json)；
  [`F_search future-action receipt`](../manifests/hotpot_recursive_evaluator_f_search_receipt_v2.json)；
  [`A_hold pre-run freeze`](../manifests/hotpot_recursive_evaluator_a_hold_pre_run_freeze_v2.json)；
  [`A_hold no-promotion report`](../manifests/hotpot_recursive_evaluator_a_hold_report_v2.json)；
  [`L4-positive / L5-negative final disposition`](../manifests/hotpot_recursive_study_v1_final_disposition.json)。
  M_search 没有 artifact：它未获授权且未打开
- final Hotpot two-Q portfolio acquisition strict-termination chain（不含任何 selected ID/private row）：
  [`portfolio design`](../manifests/hotpot_evaluator_portfolio_design_v1.json)；
  [`portfolio preregistration`](../manifests/hotpot_evaluator_portfolio_preregistration_v1.json)；
  [`one-shot consumption marker`](../artifacts/hotpot_evaluator_robust_acquisition_v1/authorization.consumed.json)；
  [`infrastructure-failure disposition`](../manifests/hotpot_evaluator_portfolio_acquisition_infrastructure_failure_disposition_v1.json)。
  没有 block、locator 或 acquisition receipt artifact；`[156,324)` 永久烧毁且不做 Hotpot v4
- MuSiQue residual two-Q portfolio strict-termination chain（不含 item ID/raw row）：
  [`portfolio design`](../manifests/musique_evaluator_portfolio_design_v1.json)；
  [`row-zero preregistration`](../manifests/musique_evaluator_portfolio_acquisition_v1_preregistration.json)；
  [`168-item acquisition receipt`](../manifests/musique_evaluator_portfolio_acquisition_v1_acquisition.json)；
  [`A-form frozen-action receipt`](../manifests/musique_evaluator_portfolio_a_form_receipt_v1.json)；
  [`F-search frozen-action receipt`](../manifests/musique_evaluator_portfolio_f_search_receipt_v1.json)；
  [`A-hold pre-run freeze`](../manifests/musique_evaluator_portfolio_a_hold_pre_run_freeze_v1.json)；
  [`A-hold authorization consumption`](../artifacts/musique_evaluator_portfolio_v1/a_hold_execution_root/a_hold.authorization.consumed.json)；
  [`A-hold failure receipt`](../artifacts/musique_evaluator_portfolio_v1/a_hold_execution_root/a_hold.failure.json)；
  [`implementation-invalid disposition`](../manifests/musique_evaluator_portfolio_a_hold_implementation_failure_disposition_v1.json)。
  A_hold 已烧毁且不 replay；没有 private evidence/aggregate report/promotion，M_search 未授权且未打开
- Replication C promotion / controls / sealed final chain：
  [`promotion decision`](../manifests/financial_sec13f_contract_v2_replication_c_promotion_decision_v1.json)；
  [`controls disposition`](../manifests/financial_sec13f_contract_v2_controls_disposition_v1.json)；
  [`family-out disposition`](../manifests/financial_sec13f_contract_v2_family_out_applicability_disposition_v1.json)；
  [`sealed preregistration`](../manifests/financial_sec13f_contract_v2_sealed_preregistration_v1.json)；
  [`sealed authorization`](../manifests/financial_sec13f_contract_v2_sealed_authorization_v1.json)；
  [`sealed execution freeze`](../manifests/financial_sec13f_contract_v2_replication_c_sealed_execution_freeze_v1.json)；
  [`sealed report`](../artifacts/financial_sec13f_contract_v2_replication_c_sealed_formal_v1/sealed.report.json)；
  [`sealed result disposition`](../manifests/financial_sec13f_contract_v2_replication_c_sealed_result_v1.json)；
  [`preauthorization digest incident`](../manifests/financial_sec13f_contract_v2_sealed_hash_only_access_incident_v1.json)；
  [`post-launch instruction exposure incident`](../manifests/financial_sec13f_contract_v2_sealed_runtime_instruction_exposure_incident_v1.json)
- legacy 代码：[`assumption_os/`](../../assumption_os/)
- legacy 自我演化评估：
  [`codex_gpt_advice_assessment_20260707.md`](../../reconstruction/md/codex_gpt_advice_assessment_20260707.md)
- self-evolution bundle：
  [`reference/self_evo_continual_20260707/`](../reference/self_evo_continual_20260707/)
- RQGM PDF：
  [`The Red Queen Gödel Machine`](<../reference/The Red Queen Gödel Machine Co-Evolving Agents and Their Evaluators.pdf>)
- v2 architecture：[`ARCHITECTURE.md`](../ARCHITECTURE.md)
- v2 benchmark protocol：[`BENCHMARK_PROTOCOL.md`](../BENCHMARK_PROTOCOL.md)
- v2 current status：[`STATUS.md`](../STATUS.md)
- frozen financial semantic candidate、formation 与 fresh split：
  [`fresh provenance split`](../manifests/skilllearn_fresh_provenance_split_v1.json)；
  [`operator asset`](../manifests/financial_semantic_operator_asset_v1.json)；
  [`DistilBERT QA runtime asset`](../manifests/financial_distilbert_qa_runtime_asset_v1.json)；
  [`operator source`](../assumption_agent/benchmarks/financial_semantic_operator_v1.py)；
  [`formation report`](../artifacts/financial_semantic_train_diagnostic_v1_actual02/financial_semantic_train_diagnostic.report.json)；
  [`formation event ledger`](../artifacts/financial_semantic_train_diagnostic_v1_actual02/financial_semantic_train_diagnostic.events.jsonl)
- financial fresh single-item treatment 与 scheduler-loss recovery（final report `e6bc247e…d389`；非 incumbent）：
  [`treatment freeze`](../manifests/financial_semantic_treatment_freeze_v1.json)；
  [`final verifier-only continuation manifest`](../manifests/financial_semantic_fresh_scheduler_recovery_v1_actual03.json)；
  [`original event ledger`](../artifacts/financial_semantic_fresh_v1_plus_actual01/execution.events.jsonl)；
  [`recovery event ledger`](../artifacts/financial_semantic_fresh_v1_plus_actual01/recovery.events.jsonl)；
  [`finalization event ledger`](../artifacts/financial_semantic_fresh_v1_plus_actual01/recovery.finalization.events.jsonl)；
  [`final recovered report`](../artifacts/financial_semantic_fresh_v1_plus_actual01/fresh_paired.recovered.report.json)；
  [`raw worker artifacts`](../artifacts/financial_semantic_fresh_v1_plus_actual01/worker_state/)
- SEC 13F period-out 多折复验（15 valid + 1 fail-closed RAW；partial descriptive、非 incumbent）：
  [`preregistration`](../manifests/financial_semantic_sec13f_period_out_preregistration_v1.json)；
  [`acquisition receipt`](../manifests/financial_semantic_sec13f_period_out_acquisition_v1.json)；
  [`measurement view`](../manifests/financial_semantic_sec13f_period_out_measurement_view_v1.json)；
  [`execution freeze`](../manifests/financial_semantic_sec13f_period_out_execution_freeze_v1.json)；
  [`offline partial result`](../manifests/financial_semantic_sec13f_period_out_partial_result_v1.json)；
  [`formal failure receipt`](../artifacts/financial_semantic_sec13f_period_out_v1_actual01/measurement.failure.json)；
  [`zero-replay recovery report`](../artifacts/financial_semantic_sec13f_period_out_v1_actual01/recovery_attempts/4c2c8a6018952eadc1d3445cfaabc8d05150e279d557c2ae821e9a8081cbeb27.json)；
  [`raw worker artifacts`](../artifacts/financial_semantic_sec13f_period_out_v1_actual01/worker_state/)
- latest clean negative development protocol：
  [`skilllearn_paper_protocol_v3_20_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_20_ruoli_gpt54mini.json)
- v3.20 development evidence：
  [`protocol lock`](../artifacts/paper_primary_v3_20_offline86_ruoli_gpt54mini_outer38_model48_portable01/protocol_lock.json)；
  [`prewarm receipt`](../artifacts/paper_primary_v3_20_offline86_ruoli_gpt54mini_outer38_model48_portable01/development_prewarm.json)；
  [`recursive report`](../artifacts/paper_primary_v3_20_offline86_ruoli_gpt54mini_outer38_model48_portable01/development_recursive.report.json)；
  [`no-recursive report`](../artifacts/paper_primary_v3_20_offline86_ruoli_gpt54mini_outer38_model48_portable01/development_no_recursive.report.json)；
  [`shared event ledger`](../artifacts/paper_primary_v3_20_offline86_ruoli_gpt54mini_outer38_model48_portable01/development_recursive.events.jsonl)；
  [`recursive archive`](../artifacts/paper_primary_v3_20_offline86_ruoli_gpt54mini_outer38_model48_portable01/development_recursive.archive.json)
- execution-contract candidate compile integration（14 candidates / 6 programs / 14×38；非评分）：
  [`integration report`](../artifacts/train_execution_contract_integration_v2_v320_train_actual01/integration.report.json)
- execution-contract TRAIN actual source run（Plus；55 valid + 1 capacity terminal）：
  [`compile integration report`](../artifacts/train_execution_contract_development_v2_v320_train_plus_actual01/compile_integration/integration.report.json)；
  [`source event ledger`](../artifacts/train_execution_contract_development_v2_v320_train_plus_actual01/execution.events.jsonl)；
  [`ranking failure receipt`](../artifacts/train_execution_contract_development_v2_v320_train_plus_actual01/ranking.failure.json)
- execution-contract TRAIN one-route resume 与最终离线 ranking（Pro；56 active + 476 replay）：
  [`final ranking report`](../artifacts/train_execution_contract_development_v2_v320_train_resume_pro_actual01/ranking.report.json)；
  [`exact retry event ledger`](../artifacts/train_execution_contract_development_v2_v320_train_resume_pro_actual01/retry.execution.events.jsonl)
- organize-2 post-selection targeted item-out refit/falsification（`unbiased_crossfit=false`；非 incumbent）：
  [`compile report`](../artifacts/train_execution_contract_crossfit_v2_v320_train_loo_organize2_plus_actual01/compile_diagnostic/crossfit.compile.report.json)；
  [`post-run compile audit replay`](../artifacts/train_execution_contract_crossfit_v2_v320_train_loo_organize2_plus_actual01/compile_diagnostic_audit_replay/crossfit.compile.report.json)；
  [`actual event ledger`](../artifacts/train_execution_contract_crossfit_v2_v320_train_loo_organize2_plus_actual01/crossfit.execution.events.jsonl)；
  [`final report`](../artifacts/train_execution_contract_crossfit_v2_v320_train_loo_organize2_plus_actual01/crossfit.report.json)；
  [`raw Codex/verifier worker artifacts`](../artifacts/train_execution_contract_crossfit_v2_v320_train_loo_organize2_plus_actual01/worker_state/)
- organize-5/-6 targeted item-out family completion（两路 Plus 并发；均 valid failure）：
  [`organize-5 compile report`](../artifacts/train_execution_contract_crossfit_v2_v320_train_loo_organize5_plus_actual01/compile_diagnostic/crossfit.compile.report.json)；
  [`organize-5 final report`](../artifacts/train_execution_contract_crossfit_v2_v320_train_loo_organize5_plus_actual01/crossfit.report.json)；
  [`organize-5 raw worker artifacts`](../artifacts/train_execution_contract_crossfit_v2_v320_train_loo_organize5_plus_actual01/worker_state/)；
  [`organize-6 compile report`](../artifacts/train_execution_contract_crossfit_v2_v320_train_loo_organize6_plus_actual01/compile_diagnostic/crossfit.compile.report.json)；
  [`organize-6 final report`](../artifacts/train_execution_contract_crossfit_v2_v320_train_loo_organize6_plus_actual01/crossfit.report.json)；
  [`organize-6 raw worker artifacts`](../artifacts/train_execution_contract_crossfit_v2_v320_train_loo_organize6_plus_actual01/worker_state/)
- trace-refined organize candidate（提交 `baa3230a` 后三路 Plus 同时启动；1/3 recovery，仍非 unbiased/incumbent）：
  [`preregistered manifest`](../manifests/train_trace_refined_organize_item_out_v2.json)；
  [`organize-2 compile report`](../artifacts/train_trace_refined_organize_item_out_v2_organize2_plus_actual01/compile_diagnostic/crossfit.compile.report.json)；
  [`organize-2 final report`](../artifacts/train_trace_refined_organize_item_out_v2_organize2_plus_actual01/crossfit.report.json)；
  [`organize-2 raw worker artifacts`](../artifacts/train_trace_refined_organize_item_out_v2_organize2_plus_actual01/worker_state/)；
  [`organize-5 compile report`](../artifacts/train_trace_refined_organize_item_out_v2_organize5_plus_actual01/compile_diagnostic/crossfit.compile.report.json)；
  [`organize-5 final report`](../artifacts/train_trace_refined_organize_item_out_v2_organize5_plus_actual01/crossfit.report.json)；
  [`organize-5 raw worker artifacts`](../artifacts/train_trace_refined_organize_item_out_v2_organize5_plus_actual01/worker_state/)；
  [`organize-6 compile report`](../artifacts/train_trace_refined_organize_item_out_v2_organize6_plus_actual01/compile_diagnostic/crossfit.compile.report.json)；
  [`organize-6 final report`](../artifacts/train_trace_refined_organize_item_out_v2_organize6_plus_actual01/crossfit.report.json)；
  [`organize-6 raw worker artifacts`](../artifacts/train_trace_refined_organize_item_out_v2_organize6_plus_actual01/worker_state/)
- typed-assignment organize candidate（最终预注册提交 `0eba5b7c`；三路 Pro 并发；3/3 valid、1/3 recovery，
  representation stopped，非 incumbent）：
  [`preregistered manifest`](../manifests/train_typed_assignment_organize_crossfit_v3.json)；
  [`final non-scoring compile report`](../artifacts/train_typed_assignment_crossfit_v3_v320_preregister02/typed_assignment_crossfit.compile.report.json)；
  [`Plus 401 event ledger`](../artifacts/train_typed_assignment_crossfit_v3_v320_provider_selection01/plus.canary.events.jsonl)；
  [`Plus unavailability receipt`](../artifacts/train_typed_assignment_crossfit_v3_v320_provider_selection01/plus.failure.json)；
  [`Pro canary report`](../artifacts/train_typed_assignment_crossfit_v3_v320_provider_selection01/pro.canary.json)；
  [`provider selection receipt`](../artifacts/train_typed_assignment_crossfit_v3_v320_provider_selection01/provider.selection.json)；
  [`actual report`](../artifacts/train_typed_assignment_crossfit_v3_v320_pro_actual01/typed_assignment_crossfit.report.json)；
  [`actual event ledger`](../artifacts/train_typed_assignment_crossfit_v3_v320_pro_actual01/typed_assignment_crossfit.execution.events.jsonl)；
  [`raw Codex/verifier worker artifacts`](../artifacts/train_typed_assignment_crossfit_v3_v320_pro_actual01/worker_state/)
- frozen local semantic-assignment operator 与第一次 public-OA acquisition（operator 未被评价）：
  [`consumed TRAIN pack`](../manifests/semantic_assignment_consumed_train_pack_v1.json)；
  [`MiniLM runtime asset`](../manifests/semantic_assignment_minilm_runtime_asset_v1.json)；
  [`frozen operator asset`](../manifests/semantic_assignment_operator_asset_v1.json)；
  [`public-OA preregistration`](../manifests/semantic_assignment_public_oa_feasibility_v1.json)；
  [`acquisition failure receipt`](../manifests/semantic_assignment_public_oa_feasibility_result_v1.json)；
  [`period-out preregistration`](../manifests/semantic_assignment_public_oa_period_out_feasibility_v2.json)；
  [`period-out acquisition failure receipt`](../manifests/semantic_assignment_public_oa_period_out_feasibility_result_v2.json)
- SC-100 v1 false-positive closure、frozen instrument 与已停止的 role-v2 shadow：
  [`v1 semantic result`](../manifests/sc100_typed_train_diagnostic_result_v1.json)；
  [`synthetic corpus`](../reference/synthetic_sc100_shadow_v1/)；
  [`oracle qualification fixtures`](../manifests/sc100_shadow_oracle_qualification_fixtures_v1.json)；
  [`oracle preregistration`](../manifests/sc100_shadow_oracle_qualification_v1.json)；
  [`oracle result receipt`](../manifests/sc100_shadow_oracle_qualification_result_v1.json)；
  [`offline report`](../artifacts/sc100_shadow_oracle_qualification_v1/sc100_shadow_oracle_qualification.report.json)；
  [`decision lock`](../artifacts/sc100_shadow_oracle_qualification_v1/sc100_shadow_oracle_qualification.decision.lock.json)；
  [`role-v2 shadow preregistration`](../manifests/sc100_synthetic_shadow_v1.json)；
  [`role-v2 shadow result receipt`](../manifests/sc100_synthetic_shadow_result_v1.json)；
  [`role-v2 shadow report`](../artifacts/sc100_synthetic_shadow_v1_role_v2/sc100_synthetic_shadow.report.json)；
  [`role-v2 shadow decision lock`](../artifacts/sc100_synthetic_shadow_v1_role_v2/sc100_synthetic_shadow.decision.lock.json)
- previous live-mechanism protocol（performance mixed-validity）：
  [`skilllearn_paper_protocol_v3_18r1_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_18r1_ruoli_gpt54mini.json)
- formal production typed-selection integration v2：
  [`preregistration`](../manifests/skilllearn_typed_selection_integration_v2.json)；
  [`result receipt`](../manifests/skilllearn_typed_selection_integration_result_v2.json)；
  [`report`](../artifacts/typed_selection_integration_v2_v315_train/typed_selection_integration.report.json)；
  [`event ledger`](../artifacts/typed_selection_integration_v2_v315_train/typed_selection_integration.events.jsonl)；
  [`decision lock`](../artifacts/typed_selection_integration_v2_v315_train/typed_selection_integration.decision.lock.json)
- formal typed-portable pre-agent integration v1（已使用的 v3.20 development eligibility；非 incumbent/promotion）：
  [`preregistration`](../manifests/skilllearn_typed_portable_integration_v1.json)；
  [`result receipt`](../manifests/skilllearn_typed_portable_integration_result_v1.json)；
  [`report`](../artifacts/typed_portable_integration_v1_v315_train/typed_portable_integration.report.json)；
  [`event ledger`](../artifacts/typed_portable_integration_v1_v315_train/typed_portable_integration.events.jsonl)；
  [`decision lock`](../artifacts/typed_portable_integration_v1_v315_train/typed_portable_integration.decision.lock.json)
- formal TRAIN-only runtime-profile prompt delivery integration v1（非评分、非 semantic-consumption/task-utility claim）：
  [`preregistration`](../manifests/skilllearn_typed_profile_injection_integration_v1.json)；
  [`result receipt`](../manifests/skilllearn_typed_profile_injection_integration_result_v1.json)；
  [`report`](../artifacts/typed_profile_injection_integration_v1_v320_train/typed_profile_injection.report.json)；
  [`event ledger`](../artifacts/typed_profile_injection_integration_v1_v320_train/typed_profile_injection.events.jsonl)；
  [`decision lock`](../artifacts/typed_profile_injection_integration_v1_v320_train/typed_profile_injection.decision.lock.json)
- formal consumed-development runtime-profile consumption diagnostic v1（非 claim）：
  [`preregistration`](../manifests/skilllearn_typed_profile_consumption_diagnostic_v1.json)；
  [`result receipt`](../manifests/skilllearn_typed_profile_consumption_diagnostic_result_v1.json)；
  [`report`](../artifacts/typed_profile_consumption_diagnostic_v1_v320_consumed_validation/profile_consumption.report.json)；
  [`event ledger`](../artifacts/typed_profile_consumption_diagnostic_v1_v320_consumed_validation/profile_consumption.events.jsonl)；
  [`completion lock`](../artifacts/typed_profile_consumption_diagnostic_v1_v320_consumed_validation/profile_consumption.completion.lock.json)
- latest failed proposal-only diagnostic protocol：
  [`skilllearn_paper_protocol_v3_17_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_17_ruoli_gpt54mini.json)
- immutable v3.16 proposal-only diagnostic protocol：
  [`skilllearn_paper_protocol_v3_16_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_16_ruoli_gpt54mini.json)
- immutable v3.10 proposal-diversity diagnostic protocol：
  [`skilllearn_paper_protocol_v3_10_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_10_ruoli_gpt54mini.json)
- immutable v3.9 clean negative-development protocol：
  [`skilllearn_paper_protocol_v3_9_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_9_ruoli_gpt54mini.json)
- immutable v3.8 two-worker capacity diagnostic protocol：
  [`skilllearn_paper_protocol_v3_8_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_8_ruoli_gpt54mini.json)
- immutable v3.7 six-worker capacity diagnostic protocol：
  [`skilllearn_paper_protocol_v3_7_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_7_ruoli_gpt54mini.json)
- immutable v3.6 contrastive/serial diagnostic protocol：
  [`skilllearn_paper_protocol_v3_6_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_6_ruoli_gpt54mini.json)
- immutable v3.5 execution/learning diagnostic protocol：
  [`skilllearn_paper_protocol_v3_5_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_5_ruoli_gpt54mini.json)
- immutable v3.4 execution diagnostic protocol：
  [`skilllearn_paper_protocol_v3_4_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_4_ruoli_gpt54mini.json)
- immutable v3.3 execution diagnostic protocol：
  [`skilllearn_paper_protocol_v3_3_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_3_ruoli_gpt54mini.json)
- immutable v3.2 diagnostic protocol：
  [`skilllearn_paper_protocol_v3_2_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_2_ruoli_gpt54mini.json)
- immutable v3.1 diagnostic protocol：
  [`skilllearn_paper_protocol_v3_ruoli_gpt54mini.json`](../manifests/skilllearn_paper_protocol_v3_ruoli_gpt54mini.json)
- frozen offline-ready manifests：
  [`instance holdout`](../manifests/skilllearnbench_instance_holdout_offline_ready_v1.json)；
  [`family out`](../manifests/skilllearnbench_family_out_offline_ready_v1.json)
- version-controlled readiness evidence：
  [`skilllearn_offline_readiness_receipt_v1.json`](../manifests/skilllearn_offline_readiness_receipt_v1.json)
- local ignored diagnostics（非 clone 中的主证据）：
  [`offline verifier matrix`](../artifacts/offline_verifier_matrix_offline86_20260711_v1/matrix.json)；
  [`86-item runtime prewarm receipt`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/development_prewarm.json)；
  [`mechanism smoke`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/smoke_recursive.report.json)；
  [`full-development fail-closed events`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini/development_recursive.events.jsonl)；
  [`v3.1 clean-rerun cap recurrence`](../artifacts/paper_primary_v3_1_offline86_ruoli_gpt54mini_rerun01/development_recursive.events.jsonl)；
  [`v3.2 claim lock`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini/protocol_lock.json)；
  [`v3.2 86-item prewarm`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini/development_prewarm.json)；
  [`v3.2 provider-capacity failure`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini/development_recursive.events.jsonl)；
  [`GPT Pro Codex canary`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini/gpt_pro_transport_canary/report.json)；
  [`gptpro01 receipt false-negative run`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini_gptpro01/development_recursive.events.jsonl)；
  [`gptpro03 final 64 MiB hard-cap run`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini_gptpro03/development_recursive.events.jsonl)；
  [`video-object-counting-1 failed trace`](../artifacts/paper_primary_v3_2_offline86_ruoli_gpt54mini_gptpro03/development_recursive/upstream_trials/no_skill/video-object-counting/video-object-counting-1/v2_policy_off_66599fccf924efd4c6/agent/codex.txt)；
  [`v3.3 claim lock`](../artifacts/paper_primary_v3_3_offline86_ruoli_gpt54mini/protocol_lock.json)；
  [`v3.3 86-item prewarm`](../artifacts/paper_primary_v3_3_offline86_ruoli_gpt54mini/development_prewarm.json)；
  [`v3.3 video-1 canary`](../artifacts/paper_execution_policy_v3_3_video_object_counting_1_canary01/report.json)；
  [`v3.3 full-train events`](../artifacts/paper_primary_v3_3_offline86_ruoli_gpt54mini/development_recursive.events.jsonl)；
  [`forbidden web-search trace`](../artifacts/paper_primary_v3_3_offline86_ruoli_gpt54mini/development_recursive/upstream_trials/no_skill/offer-letter-generator/offer-letter-generator-1/v2_policy_off_a99904ddf5496bed16/agent/codex.txt)；
  [`v3.3 video-1 valid trace`](../artifacts/paper_primary_v3_3_offline86_ruoli_gpt54mini/development_recursive/upstream_trials/no_skill/video-object-counting/video-object-counting-1/v2_policy_off_26f7d1bd8b10776c43/agent/codex.txt)；
  [`long temperature-3 trace`](../artifacts/paper_primary_v3_3_offline86_ruoli_gpt54mini/development_recursive/upstream_trials/no_skill/temperature-simulation/temperature-simulation-3/v2_policy_off_970cfa8b6418033bd2/agent/codex.txt)；
  [`v3.4 zero-model wire probe`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/diagnostics/codex_model_only_wire.json)；
  [`v3.4 runtime preparation`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/codex_runtime_preparation.json)；
  [`v3.4 claim lock`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/protocol_lock.json)；
  [`v3.4 86-item prewarm`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/development_prewarm.json)；
  [`max2 v1 pre-model PATH failure`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/diagnostics/max2_offer_letter_canary_v1.json)；
  [`max2 v2 raw 503 diagnosis`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/diagnostics/max2_offer_letter_canary_v2.json)；
  [`max2 v3 classified provider blocker`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/diagnostics/max2_offer_letter_canary_v3.json)；
  [`max2 v4 host-permission diagnostic`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/diagnostics/max2_offer_letter_canary_v4.json)；
  [`max2 v5 passing action-budget canary`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/diagnostics/max2_offer_letter_canary_v5.json)；
  [`v3.4 four-worker provider-capacity failure`](../artifacts/paper_primary_v3_4_offline86_ruoli_gpt54mini/development_recursive.events.jsonl)；
  [`v3.5 38/38 train then repair-ID collision`](../artifacts/paper_primary_v3_5_offline86_ruoli_gpt54mini/development_recursive.events.jsonl)；
  [`v3.5 repairid01 38/38 train then malformed repair envelope`](../artifacts/paper_primary_v3_5_offline86_ruoli_gpt54mini_repairid01/development_recursive.events.jsonl)；
  [`v3.5 repaircontract01 recursive report`](../artifacts/paper_primary_v3_5_offline86_ruoli_gpt54mini_repaircontract01/development_recursive.report.json)；
  [`v3.5 repaircontract01 no-recursive contaminated report`](../artifacts/paper_primary_v3_5_offline86_ruoli_gpt54mini_repaircontract01/development_no_recursive.report.json)；
  [`v3.9 clean recursive report`](../artifacts/paper_primary_v3_9_offline86_ruoli_gpt54mini_outer6_model1_plus01/development_recursive.report.json)；
  [`v3.9 clean recursive archive`](../artifacts/paper_primary_v3_9_offline86_ruoli_gpt54mini_outer6_model1_plus01/development_recursive.archive.json)；
  [`v3.9 clean no-recursive report`](../artifacts/paper_primary_v3_9_offline86_ruoli_gpt54mini_outer6_model1_plus01/development_no_recursive.report.json)；
  [`v3.9 clean no-recursive archive`](../artifacts/paper_primary_v3_9_offline86_ruoli_gpt54mini_outer6_model1_plus01/development_no_recursive.archive.json)；
  [`v3.10 recursive report`](../artifacts/paper_primary_v3_10_offline86_ruoli_gpt54mini_outer6_model1_diverse_plus01/development_recursive.report.json)；
  [`v3.10 recursive archive`](../artifacts/paper_primary_v3_10_offline86_ruoli_gpt54mini_outer6_model1_diverse_plus01/development_recursive.archive.json)；
  [`v3.10 no-recursive report`](../artifacts/paper_primary_v3_10_offline86_ruoli_gpt54mini_outer6_model1_diverse_plus01/development_no_recursive.report.json)；
  [`v3.10 no-recursive archive`](../artifacts/paper_primary_v3_10_offline86_ruoli_gpt54mini_outer6_model1_diverse_plus01/development_no_recursive.archive.json)；
  [`v3.15 claim lock`](../artifacts/paper_primary_v3_15_offline86_ruoli_gpt54mini_outer6_model1_actiondelta01/protocol_lock.json)；
  [`v3.15 86-item prewarm`](../artifacts/paper_primary_v3_15_offline86_ruoli_gpt54mini_outer6_model1_actiondelta01/development_prewarm.json)；
  [`v3.15 recursive report`](../artifacts/paper_primary_v3_15_offline86_ruoli_gpt54mini_outer6_model1_actiondelta01/development_recursive.report.json)；
  [`v3.15 recursive archive`](../artifacts/paper_primary_v3_15_offline86_ruoli_gpt54mini_outer6_model1_actiondelta01/development_recursive.archive.json)；
  [`v3.15 no-recursive report`](../artifacts/paper_primary_v3_15_offline86_ruoli_gpt54mini_outer6_model1_actiondelta01/development_no_recursive.report.json)；
  [`v3.15 no-recursive archive`](../artifacts/paper_primary_v3_15_offline86_ruoli_gpt54mini_outer6_model1_actiondelta01/development_no_recursive.archive.json)；
  [`v3.15 action-audit/event ledger`](../artifacts/paper_primary_v3_15_offline86_ruoli_gpt54mini_outer6_model1_actiondelta01/development_recursive.events.jsonl)；
  [`v3.16 failed proposal-only report`](../artifacts/paper_primary_v3_16_offline86_ruoli_gpt54mini_outer6_model1_familyslots01/train_proposal_diagnostic.report.json)；
  [`v3.16 proposal-only event ledger`](../artifacts/paper_primary_v3_16_offline86_ruoli_gpt54mini_outer6_model1_familyslots01/train_proposal_diagnostic.events.jsonl)；
  [`v3.17 failed proposal-only report`](../artifacts/paper_primary_v3_17_offline86_ruoli_gpt54mini_outer6_model1_familyslots02/train_proposal_diagnostic.report.json)；
  [`v3.17 proposal-only event ledger`](../artifacts/paper_primary_v3_17_offline86_ruoli_gpt54mini_outer6_model1_familyslots02/train_proposal_diagnostic.events.jsonl)；
  [`v3.18r1 claim lock`](../artifacts/paper_primary_v3_18r1_offline86_ruoli_gpt54mini_outer38_model48_typed02/protocol_lock.json)；
  [`v3.18r1 86-item prewarm`](../artifacts/paper_primary_v3_18r1_offline86_ruoli_gpt54mini_outer38_model48_typed02/development_prewarm.json)；
  [`v3.18r1 recursive report`](../artifacts/paper_primary_v3_18r1_offline86_ruoli_gpt54mini_outer38_model48_typed02/development_recursive.report.json)；
  [`v3.18r1 recursive archive`](../artifacts/paper_primary_v3_18r1_offline86_ruoli_gpt54mini_outer38_model48_typed02/development_recursive.archive.json)；
  [`v3.18r1 no-recursive report`](../artifacts/paper_primary_v3_18r1_offline86_ruoli_gpt54mini_outer38_model48_typed02/development_no_recursive.report.json)；
  [`v3.18r1 no-recursive archive`](../artifacts/paper_primary_v3_18r1_offline86_ruoli_gpt54mini_outer38_model48_typed02/development_no_recursive.archive.json)；
  [`v3.18r1 causal event ledger`](../artifacts/paper_primary_v3_18r1_offline86_ruoli_gpt54mini_outer38_model48_typed02/development_recursive.events.jsonl)；
  [`organize-3 missing-input verifier`](../artifacts/paper_primary_v3_18r1_offline86_ruoli_gpt54mini_outer38_model48_typed02/development_recursive/upstream_trials/no_skill/organize-messy-files/organize-messy-files-3/v2_policy_off_9fd24b87613fdf5ce0/verifier/ctrf.json)；
  [`stock-3 RAW 8/10 verifier`](../artifacts/paper_primary_v3_18r1_offline86_ruoli_gpt54mini_outer38_model48_typed02/development_recursive/upstream_trials/no_skill/stock-data-visualization/stock-data-visualization-3/v2_policy_off_a7ed3142643bb27425/verifier/ctrf.json)；
  [`stock-3 G1 4/10 verifier`](../artifacts/paper_primary_v3_18r1_offline86_ruoli_gpt54mini_outer38_model48_typed02/development_recursive/upstream_trials/assumption-agent-v2-challenger/stock-data-visualization/stock-data-visualization-3/v2_policy_on_3828e1d61fc53a4d3a/verifier/ctrf.json)；
  [`stock-3 G2 3/10 verifier`](../artifacts/paper_primary_v3_18r1_offline86_ruoli_gpt54mini_outer38_model48_typed02/development_recursive/upstream_trials/assumption-agent-v2-challenger/stock-data-visualization/stock-data-visualization-3/v2_policy_on_9929872798245bb924/verifier/ctrf.json)

## 附录 B：复杂度统计口径

legacy 数字按以下口径复核：

```text
lines:
  Python source splitlines

functions:
  AST module.body 中 FunctionDef / AsyncFunctionDef
  nested-inclusive count 使用 ast.walk

HLE configuration surface:
  source 中唯一正则 token HLE_[A-Z0-9_]+

verifier / fallback proxy:
  顶层函数名分别包含 verifier / fallback
```

这些统计用于描述控制面规模，不应被当作独立行为数量或性能指标。
