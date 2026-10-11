# 校准与关系范围共同研究：0.122

9990 正式批已完成：12 个新根、每根两套装置、六阶段、四臂，288 次初访＋6 次自然回访；全部学习结束后统一完成 288 个任务的初访／最终终测。两个主要新旧对照均没有新增预测收益。学习、终测及汇总进程全部结束，冻结结果未重跑或调参。

| 臂 | 达标／正确完成／误报 | 学习：观测／测量／计算 | 全成本：观测／测量／计算 | 来源＋研究 CPU 秒 |
|---|---:|---:|---:|---:|
| 旧 0.121 有来源 | 72／72／0 | 11584／97260／2893993 | 33438／135156／5703462 | 169.956 |
| 新 0.122 有来源 | 72／72／0 | 11585／97272／2900022 | 33439／135168／5709491 | 171.285 |
| 旧 0.121 空库起步 | 67／49／0 | 12937／128028／9059497 | 31881／171036／9096361 | 618.540 |
| 新 0.122 空库起步 | 67／49／0 | 12914／127752／9107033 | 31858／170760／9143897 | 618.825 |

全成本包含来源、学习和终测。有来源两臂各计同一批真实来源的 3422 观测、37896 测量、2772605 计算、46.803 CPU 秒；空库两臂为零，随后正常积累。终测 CPU 未单独记录。有来源新版本多 6029 计算、1 观测、12 测量；空库新版少 23 观测和 276 测量，但多 47536 计算，不能宣称稳定净收益。每臂 72 个任务的新旧比较均 0 胜、0 负、72 并列（MSE 差阈值 `1e-7`）。

剩余五个失败均为空库的首次完整响应任务，根为 root04/07/09/10/12；旧新均正确保持 unresolved。其余各阶段均 12/12 达标，但空库去事件阶段 12 个任务仍未确认完成。两有来源臂平均 MSE 均为 `3.606723e-17`，两空库臂均为 `4.550616e-5`。具体失败及下一主线保存于 `fresh_replication/next_gap.json`：转向冷启动的非线性局部机制发现与组合，不继续细调本批微小成本差。

开发记录完整保留。初版 `development` 的三次已知误报原样存在；修复 `development_v2` 空库达标 8→11/12、正确完成 5→9/12、误报 3→0。有来源开发仍为 12/12。三次 before/after 和额外费用见两批 `root07_before_after.json` 与 `cost_totals.json`；开发改善不当成正式新增收益。

协议仍为每链 1920 观测、23040 测量、1260000 计算，初访和回访共用未花余池，6 worker，`PYTHONHASHSEED=0`。每任务四条 64 步固定新轨迹终测，不同测量库分别付费。全流程离线，外部 API 为零；空库臂不是普通 LLM。主要配对为 `knowledge_revisit` 对 `baseline`、`empty_start` 对 `baseline_empty`。

复现入口（完整检查点会复用）：

```bash
PYTHONHASHSEED=0 python3 scripts/run_knowledge_update_revisits.py --output artifacts/calibration_scope_revision_20260926/fresh_replication --phase learn
PYTHONHASHSEED=0 python3 scripts/run_knowledge_update_revisits.py --output artifacts/calibration_scope_revision_20260926/fresh_replication --phase evaluate
PYTHONHASHSEED=0 python3 scripts/run_knowledge_update_revisits.py --output artifacts/calibration_scope_revision_20260926/fresh_replication --phase summary
PYTHONHASHSEED=0 python3 scripts/summarize_mechanism_revision_study.py --output artifacts/calibration_scope_revision_20260926/fresh_replication
```

版本边界：正式冻结后，当前生产另补“映射缺字段则未决退出”及“修复证据产生前保持旧 binding 归属”两处退出修复，由独立 focused 测试覆盖；本批保留原冻结 runtime，未覆盖或重跑，实际运行无异常。完整结果在 `fresh_replication/{summary,cost_totals,mechanism_outcomes,next_gap}.json`。
