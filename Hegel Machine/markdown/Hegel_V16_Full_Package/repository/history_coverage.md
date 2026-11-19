# GitHub 历史覆盖清单与边界

日期：2026-11-19（Asia/Shanghai）。仓库：`erzhu419/Assumption-Agent`。本次只补充**提交元数据、父链和关键区间的变更文件清单**；源码、文档正文与科学实验的审查深度由其他专项报告说明。

## 1. 实际覆盖到了哪里

以 `branches.json` 保存的 16 个当前分支 head 为起点，通过只读 GitHub API 获得并去重 **1627 个可达提交**。每条保存完整 SHA、作者日期、提交者日期、提交信息、父 SHA、tree SHA 和网页链接。检查结果：16 个 head 均存在；所有已收集提交的父 SHA 均已解析；共同根为 `bae8dd48c8bbeda4a3d60555e40ed69d63b73bb2`。因此，**对这 16 个固定 head 的可达提交元数据，父链已经闭合**。

认知主线 `787256bd415515905359626f6a5498174ad5db3c` 的提交集合有 1618 条，API 分页为 16 页×100 条及末页18条，共17页；记录日期从2026-04-16到2026-11-18。另取 main、R7 recovery、R4 recovery 各1页，用已有共同祖先补足其父链，没有逐分支重复拉取全部祖先。全分支合计只比认知主线多9个不同 SHA。

这里的“闭合”是 Git 提交图的覆盖，不是1627次实验独立发生，也不是全文阅读1627个版本。历史中存在相似提交信息的不同 SHA、合并与文档提交；这些不能当独立实验样本或独立复现。

| 当前分支 | 固定 head（短写） | 可达提交元数据数 | 不在认知主线祖先集中的提交数 |
|---|---:|---:|---:|
| codex/hegel-cognitive-mainline | 787256bd | 1618 | 0 |
| codex/hegel-m3-start-runtime | 3ce99193 | 1585 | 0 |
| codex/hegel-m25-a8-r4-attempt4-recovery | 6b4c0ed9 | 1421 | 5 |
| codex/hegel-m25-a8-r4r2-attempt4-recovery | f24bae3c | 1421 | 5 |
| codex/hegel-m25-a8-r5-attempt5-recovery | 0024f811 | 1422 | 6 |
| codex/hegel-m25-a8-r6-attempt6-recovery | e10fa895 | 1423 | 7 |
| codex/hegel-m25-a8-r7-attempt7-poststage-recovery | 6055eef7 | 1424 | 8 |
| codex/hegel-m25-a8-r31-pre-attempt-erratum | 6c1b7306 | 1420 | 4 |
| codex/hegel-m25-a8-recovery-amendment | 03491315 | 1417 | 1 |
| codex/hegel-m25-a8-recovery-amendment-r2 | ec7c04cf | 1418 | 2 |
| codex/hegel-m25-a8-recovery-amendment-r3 | 52a4a619 | 1419 | 3 |
| codex/hle-generalization-prefetch-fairness-20260628 | 788d3282 | 468 | 0 |
| codex/operator-fidelity-repair-20260621 | c66868a5 | 440 | 0 |
| codex/reconstruction-v2-paper | 0af65964 | 1416 | 0 |
| main | 71e133b4 | 320 | 0 |
| reconstruction-v2 | b05f2902 | 433 | 0 |

各行有大量共同祖先，不能将表中数量相加。`main` 也不是最新认知主线；它的 head 提交日期为2026-06-10。

## 2. 用户指定的连续审查版本均已覆盖

| 版本锚点 | 提交者日期（UTC） | 元数据核对 |
|---|---|---|
| [abe15bea](https://github.com/erzhu419/Assumption-Agent/commit/abe15bead458739d1ef5b83e45e96140c75a4d34) | 2026-11-17 02:01:04 | 此提交本身仅新增两份 ZIP 档案；核心代码状态继承父提交 `e6f9ab4e` |
| [eee787bf](https://github.com/erzhu419/Assumption-Agent/commit/eee787bfdca8b87f38095adff0664955098e7d8a) | 2026-11-17 07:10:50 | 提交信息为 v11，父提交是 abe15bea |
| [6e963206](https://github.com/erzhu419/Assumption-Agent/commit/6e963206fd2822476c894fd2d0f09db4be70223c) | 2026-11-17 13:17:39 | 提交信息为 v12；中间还有 `adedf0d2` 的 v11.1，不能说它直接以 eee787bf 为父提交 |
| [c71e65f7](https://github.com/erzhu419/Assumption-Agent/commit/c71e65f7f10863ea68f2ccecac2cd5a86e1e1ba4) | 2026-11-17 20:04:23 | 提交信息为 v13，父提交是6e963206 |
| [c9b60dc5](https://github.com/erzhu419/Assumption-Agent/commit/c9b60dc525b5d87409619cc5c7b0f904d6761bf8) | 2026-11-18 05:01:17 | 提交信息为 v14，父提交是c71e65f7 |
| [787256bd](https://github.com/erzhu419/Assumption-Agent/commit/787256bd415515905359626f6a5498174ad5db3c) | 2026-11-18 12:22:17 | 提交信息为 v15，父提交是c9b60dc5 |

日期和标题来自 Git 元数据，不能替代实验实际执行时间、结果真实性或提交说明所称“完成”的验收。此次没有因提交说明含有通过率而将其视为独立重算证据。

## 3. 五个区间的关键 diff 覆盖

通过 GitHub compare API 取以下固定 SHA 区间的 changed paths、状态、行数统计、blob SHA 与网页地址。五个区间返回文件数均小于300，未碰到 compare API 常见的文件清单截断上限。部分大文件没有返回 patch；已记录 `patch_present`，没有用“空 patch”推断“没有变更”。

| 比较区间 | 包含提交数 | 返回变更文件数 | 核心源码/脚本路径涉及的范围 |
|---|---:|---:|---|
| abe15bea → eee787bf | 1 | 38 | branch_reconciliation、premise_commitment_index、discriminating_experiment_policy、unified_research，以及v11生命周期和2×2脚本 |
| eee787bf → 6e963206 | 2 | 47 | 含v11.1的候选/区分实验调整，v12 autonomous lifecycle 与 knowledge utility 脚本，研究入口和分支处理 |
| 6e963206 → c71e65f7 | 1 | 37 | 分支处理、竞争程序、commitment index、研究入口，v13 PR-A/B 与 PR-C 脚本 |
| c71e65f7 → c9b60dc5 | 1 | 36 | 分支处理、竞争程序、commitment index、研究入口，v14公平比较和连续对象脚本 |
| c9b60dc5 → 787256bd | 1 | 42 | memory、分支处理、commitment index、研究入口，v15控制/回放和前瞻E1/E2脚本 |

“涉及”只表示 Git 变更路径。部分关键 patch 的 hunk 位置和新增函数/类定义另存 JSON，供定位；本清单不据此声称实现逻辑正确、所有路径连接成功，或全部验收测试通过。

## 4. 早期 reconstruction 与 phase 版本的可追溯锚点

除全量元数据之外，对以下五个早期提交补取了完整的当次 changed-path 清单（每次均少于100个返回文件）：

| 锚点 | 日期（UTC） | 变更文件数 | 可确认的路径联系 |
|---|---|---:|---|
| [41609f5f](https://github.com/erzhu419/Assumption-Agent/commit/41609f5f6872) | 2026-04-17 | 80 | phase two 的 distiller/evaluator/integrator 与 phase three 的 kernel/isomorphism/metrics，以及 phase one/zero |
| [1b312fda](https://github.com/erzhu419/Assumption-Agent/commit/1b312fda80db) | 2026-04-22 | 26 | 早期 `phase2_v15_*`、`phase2_v16_cases_reflect.py` 与答案/裁判缓存 |
| [027e6593](https://github.com/erzhu419/Assumption-Agent/commit/027e659381b1) | 2026-06-01 | 11 | assumption_os 的 evolution_cycle、formal_mapping、recursive_daemon/executor、residual_clusterer、world_model等 |
| [77f47b75](https://github.com/erzhu419/Assumption-Agent/commit/77f47b75ad9f) | 2026-06-06 | 6 | assumption_os/hipporag_qa_probe.py、reconstruction/md 对应报告及测试 |
| [d9bb635f](https://github.com/erzhu419/Assumption-Agent/commit/d9bb635fcfec) | 2026-06-12 | 5 | finite_category_lean_export.py、Lean工件、reconstruction/md执行方案及测试 |

这些路径支持“历史上确实存在这些模块”的溯源。早期提交标题中的效果数字在本子任务中未重新检验；也没有把旧 `phase2_v16` 与当前2026年11月的“V16计划”当作同一版本编号。跨阶段出现同名 vN，须连同目录、SHA 和日期定位。

## 5. ‘所有 branch 和版本’应怎样准确表述

推荐用语：

> 本次列出了当时可见的16个分支head，建立了其全部可达1627个提交的元数据与父链清单；对各head的文件版本清单及相关正文另作专项读取，并核对了用户指定六个版本的五段变化和五个早期关键提交。实现与实验结论以明确读过的固定版本代码、回执及原始文档为限。

不应写“逐行审完仓库全部历史版本”。当前 `all_branch_file_versions.json` 的11960条记录来自多个分支head的文件版本合并；它不是1627个历史提交的全部 blob 版本全集。本次没有对每个历史 tree 全量取文件，也没有审查已删除分支、重写后不可达提交、仅由tag指向而不被这16个head覆盖的历史或用户本地未提交文件。

## 6. 交付索引

- `history_commits_deduplicated.json`：1627条去重元数据。
- `history_coverage_summary.json`：16分支覆盖、共同根与父链闭合检查。
- `history_acquisition_manifest.json`：31个只读请求的固定SHA、分页范围、返回数量与实际时间。
- `history_cognitive_page1.json` 至 `history_cognitive_page17.json`：分页提取的认知主线元数据。
- `history_main_page1.json`、`history_recovery_r7_page1.json`、`history_recovery_r4_page1.json`：额外分支元数据页。
- `history_compare_*.json`：五个固定区间的变更文件统计与部分结构化patch定位。
- `history_abe_commit_details.json`：abe15bea只新增两份ZIP的单提交核对。
- `history_early_changed_paths.json`：五个早期提交的变更路径。

读取接口为 `GET /repos/erzhu419/Assumption-Agent/commits?sha=<固定SHA>&per_page=100&page=<页>`，以及只读 `commits/<SHA>` 和 `compare/<base>...<head>`。没有写入仓库、执行项目或重跑科学实验。
