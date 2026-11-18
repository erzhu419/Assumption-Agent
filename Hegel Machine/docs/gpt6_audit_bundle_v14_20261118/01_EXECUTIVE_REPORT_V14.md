# Hegel Machine v14：生产者—消费者合同收口、去混杂无 H* 源簇比较与连续修复对象端到端报告

## 1. 对核心研究问题的直接回答（§10 首要结论）

- **哪个新假设、因何经历改变、如何改变未见问题：** 在未曝光 `focal_memory` 连续链 `1041000–1046000`（共 `6` 条全部记录，其中 `4/6` 条产出合格 `accumulate` 修复对象）中，系统先由普通马尔可夫任务形成已确认马尔可夫分支，随后在潜变量储层冲突任务上实测双分支均失败（`both_competing_branches_refuted`），触发子问题调用现有 `latent_state_discovery` 生成含 `accumulate` 的具名修复对象（例如 `concept_0004` @ `323e791fc0e2d6e540fbbc1a`），经全新 24 行确认批独立通过后执行局部历史分支重调和。在冻结后的未见 `N=5` 新参数／新布线潜变量目标 `T1` 上，`K_eager_noH` 完成 `5/6`（均值算力 `144,103.5`，观测 `224.0`），而**仅移除该修复版本但保留其余马尔可夫源与完整潜变量 API** 的最小干预臂 `K_minimal_repaired_object_removed_other_sources_kept` 为 `5/6`（均值算力 `140,789.8`，观测 `224.0`）。
- **同样强的普通方法表现如何（严格执行 §9 停止规则）：**
  1. **问题 A（同潜变量 API、同预算父任务修复）：** 不受限直接启用潜变量搜索的普通失败重试 `ordinary_failure_retry_equal_api` 严格通过 `5/6`（均值算力 `228,395.2`，均值观测 `330.7`），子问题门控修复 `subquestion_driven_repair_equal_api` 严格通过 `4/6`（均值算力 `532,292.3`，均值观测 `269.2`）。
  2. **问题 B 与 Deliverable 2（同原始历史＋按需缓存重建 `R_lazy_noH` vs 预调和 `K_eager_noH`，全组关闭 `H*` 前置）：** 在未曝光 8 源簇 `1031000–1038000` 上，`R_lazy_noH` 严格通过 **`23/24`**（假确认 `0`，均值目标算力 `272,877.8`），`K_eager_noH` 严格通过 **`23/24`**（假确认 `0`，均值目标算力 `219,416.0`），源簇配对胜／负／平 `1/1/22`（双侧符号翻转 `p = 1.0`）；计入源期维护费 `C_K_maint` 后，逐链前缀全成本差均值 `$\Delta C_c(1..3) = (-71,171.8, -45,746.0, 37,178.1)$`。在问题 B 上，`R_lazy_noH`（`14/18`，均值算力 `228,402.0`）亦从共享原始经历按需重建出等价复用对象并与 `K_eager_noH`（`15/18`，均值算力 `214,375.2`）正面对比。按 §9 停止规则，我们如实报告普通按需重建的追平／取舍，不做夸大。

---

## 2. V13 五项勘误表（§2）与 Replays A–F 生产路径核验（Deliverable 1）

1. **普通源任务严格通过数更正：** V13 的 32 个普通源任务为 `32/32` 内部确认、**`31/32` 外部严格通过**（`1021000/S_left_dpg_N4` 内部 `1.12e-18`、外部 `8.5874e-4` 属 `1` 例源假确认），`8/8` 共同锚点单独列计；
2. **`-18.92%` 混杂撤回：** 旧 `R_ordinary_optimized_lazy`（`H* ON`）对 `K_routing_no_H_star`（`H* OFF`）混杂了 `H*` 开关，不再写作分支组织单因子净节省；
3. **同开 `H* ON` 对照：** `K_routing_plus_H_star`（`20/24`，`249,564.33`）较 `R_ordinary_optimized_lazy`（`21/24`，`226,943.08`）多耗 `+9.97%` 算力且少通过 `1` 题；
4. **PR-A Choice Point 2 记账更正：** 实有 `1` 条首分支真实 `executed_new` 与 `1` 条次分支 `unavailable`（不变量为 `0` 条 `unavailable` 混入 `paid_probes_executed`）；
5. **边界探针接线与 `H*` 默认关闭：** 本轮已将残差阈值边界探针接入 `UnifiedResearchAgent.run()` 确认前主循环，并默认关闭 `H*` 前置优先级（`_prioritize_generalized_families = False`）。
- **GPT-6 V14 三探针与 Replays A–F：** `all_gpt6_v14_probes_remediated = True`，`all_replays_a_to_f_passed = True`。

---

## 3. Deliverable 2：8 条未曝光源簇（`1031000–1038000`）去混杂无 `H*` 主比较

- **共同标定锚点（单独列计）：** `8/8` 严格通过，假确认 `0`；
- **普通源任务（单独列计）：** `31/32` 严格通过（内部确认 `31/32`，假确认 `0`，未决 `1`，执行阈值边界探针 `5` 次，每链源调和维护算力均值 `123,207.4`）。

| 目标对比臂（全组 `H* OFF`） | 严格通过 (`n=8` 簇, 24题) | 假确认 | 未决 | 错误迁移 | `T1` 新阈值族 | `T2` 旧域保留 | `T3` 少数反向 | 均值目标算力 | 均值观测 | 运行墙钟(不含外评) | 冷/暖缓存重建算力 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: | ---: | :---: |
| `K_cold` | **`23/24`** | `0` | `1` | `0` | `7/8` | `8/8` | `8/8` | `234,553.1` | `291.8` | `25.09s` | `0` / `0` |
| `R_lazy_noH` | **`23/24`** | `0` | `1` | `0` | `8/8` | `8/8` | `7/8` | `272,877.8` | `297.8` | `37.08s` | `35,101` / `5` |
| `K_eager_noH` | **`23/24`** | `0` | `1` | `0` | `7/8` | `8/8` | `8/8` | `219,416.0` | `285.2` | `28.21s` | `0` / `0` |

### 逐链前缀全成本差 $\Delta C_c(k) = C^R_{\mathrm{maint},c} + \sum_{t\le k} C^R_{c,t} - C^K_{\mathrm{maint},c} - \sum_{t\le k} C^K_{c,t}$
- `k = 1`：逐链 `[230695, -124247, -222159, -92770, -104046, -93356, -54195, -109296]`，均值 `-71,171.8`（`K` 全成本更低链数 `1/8`）；
- `k = 2`：逐链 `[498669, -131324, -227587, -99367, -110205, -111235, -68438, -116481]`，均值 `-45,746.0`（`K` 全成本更低链数 `1/8`）；
- `k = 3`：逐链 `[720559, -137431, -230434, -110744, -114378, -125972, 218073, 77752]`，均值 `37,178.1`（`K` 全成本更低链数 `3/8`）。

---

## 4. Deliverable 3：6 条未曝光连续修复对象链（`1041000–1046000`）与首链七句事实示例

| 问题 B 对比臂（全组 `latent_state_discovery=True`） | 总严格通过 (18题) | 假确认 | `T1` 新潜变量参数 (`N=5`) | `T2` 返回旧马尔可夫 (`N=5`) | `T3` 左向条件反例 (`N=5`) | 均值算力 | 均值观测 |
| :--- | :---: | :---: | :---: | :---: | :---: | ---: | ---: |
| `K_cold` | **`14/18`** | `0` | `5/6` (`171,771`) | `5/6` (`197,640`) | `4/6` (`318,086`) | `229,165.6` | `244.1` |
| `R_lazy_noH` | **`14/18`** | `0` | `5/6` (`143,317`) | `5/6` (`189,410`) | `4/6` (`352,479`) | `228,402.0` | `244.1` |
| `K_eager_noH` | **`15/18`** | `0` | `5/6` (`144,104`) | `5/6` (`192,261`) | `5/6` (`306,761`) | `214,375.2` | `240.6` |
| `K_minimal_branch_removed_raw_kept` | **`15/18`** | `0` | `5/6` (`145,120`) | `5/6` (`221,838`) | `5/6` (`306,760`) | `224,572.6` | `241.9` |
| `K_minimal_repaired_object_removed_other_sources_kept` | **`15/18`** | `0` | `5/6` (`140,790`) | `5/6` (`197,305`) | `5/6` (`306,385`) | `214,826.7` | `240.6` |

### 连续链 `unexposed_v14_focal_chain_1041000` 机器生成七句事实账本（全 6 条链见回执 JSON）

- **1_prior_ordinary_experiences_and_versions**: After neutral anchor unexposed_v14_focal_chain_1041000:S_cal_focus_N4 and ordinary Markov episodes unexposed_v14_focal_chain_1041000:S_markov_sym_N4 and unexposed_v14_focal_chain_1041000:S_markov_right_N4, the library held confirmed Markov branches ['branch:concept_0001@063ae9d3f588f615e3db03f2', 'branch:concept_0002@22c468ac219ed47ec5f594ef', 'branch:concept_0003@31ade5b11b94aed1d36203e7'] before encountering unexposed_v14_focal_chain_1041000:S_conflict_focal_memory_N4.
- **2_refuting_real_evidence_and_failed_versions**: On paid episode anonymous_process:d3083f1e29cb (unexposed_v14_focal_chain_1041000:S_conflict_focal_memory_N4, decay=0.54, weight=0.36), paid fine-channel observations refuted both Markov versions ['branch:concept_0001@063ae9d3f588f615e3db03f2', 'branch:concept_0002@22c468ac219ed47ec5f594ef', 'branch:concept_0003@31ade5b11b94aed1d36203e7'] under historical state-dependence.
- **3_reconciliation_choice_rationale**: Because both competing Markov branches failed on the same paid trajectory (`both_competing_branches_refuted`), the adjudicator narrowed their Markov claim status rather than falsely merging them and opened subquestion `reconcile_subq:anonymous_process:d3083f1e29cb:5bd5c0f9c5_vs_0ef430ba94:256676e9d7:both_competing_branches_refuted` requesting `expanded_latent_state`.
- **4_subanswer_parent_mutation**: Subquestion `reconcile_subq:anonymous_process:d3083f1e29cb:5bd5c0f9c5_vs_0ef430ba94:256676e9d7:both_competing_branches_refuted` invoked `latent_state_discovery` on all paid fine rows/transitions and replaced the unconfirmed Markov candidate `world_commitment:0001:world_commitment_instantiation:4` with `accumulate`-bearing candidate `research_composition:39` (`concept_0004` @ `323e791fc0e2d6e540fbbc1a`).
- **5_fresh_confirmation_and_historical_branch_checks**: Repaired object `concept_0004` passed an independent fresh 24-row confirmation probe (loss=1.295985666600592e-16, external48_mse=1.2275163856845478e-16) and ran local post-subresearch re-reconciliation (status=`None`) so Markov and `focal_memory` branches coexist by observable scope.
- **6_unseen_target_consumption_and_action_change**: On unseen N=5 target `unexposed_v14_focal_chain_1041000:T1_unseen_focal_memory_N5` (new decay=0.68, weight=0.42, radius=0.9), `K_eager_noH` invoked `concept_0004`=True via `research_program` (strict_pass=True, compute=115848, obs=234), whereas removing ONLY `concept_0004` (`K_minimal_repaired_object_removed_other_sources_kept`) yielded strict_pass=True, compute=108943, obs=234; meanwhile old Markov `T2` passed=True and mismatched left-conditional `T3` passed=True (wrong_transfer=False).
- **7_equal_capability_ordinary_method_comparison**: On Question A (`S_conflict`), equal-API ordinary failure-retry (`branch_reconciliation=False, latent_state_discovery=True`) achieved strict_pass=True at compute=234841, obs=348 vs subquestion repair strict_pass=True at compute=385386, obs=296; on Question B (`T1..T3`), same-history `R_lazy_noH` achieved T1 strict_pass=True (compute=117711, obs=234) and total 3-target compute=666412 vs `K_eager_noH`=379830.
