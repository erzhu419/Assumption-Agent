# Full V3 Phase1: JSONL Memory Sleep Job

## 背景

`GPT_revise_v3.md` 指出 Phase1 memory consolidation 仍是 shadow / fixture validation，没有真正的 graph sleep job。此前 Phase1 能证明“应该如何合并/剪枝”，但没有提供可复用的 JSONL graph dry-run/apply primitive。

这次新增：

- `assumption_os/memory_consolidation_job.py`

它直接作用于 `JsonlGraphStore`：

- dry-run: 只生成 consolidation plan，不写图；
- apply: 归档 stale / duplicate / conflicting nodes，写入 consolidated memory node，并添加 `derived_from` edges；
- 使用 `payload.family` / `family:*` tag / normalized claim 形成 family；
- 默认不删除节点，只改 status，便于 rollback 和 audit。

## Validation

新增单元测试：

```bash
python3 -m unittest \
  tests.test_assumption_os.AssumptionOSTest.test_memory_consolidation_job_dry_run_and_apply_on_jsonl_graph
```

结果：通过。

Phase1 artifact 重新生成：

```bash
python3 -m assumption_os.full_v3_phase1_memory_consolidation \
  --root . \
  --out "phase four/assumption_graph/paper_readiness_20260604/full_v3_phase1_memory_consolidation_20260611.json"
```

关键指标：

- production_sleep_group_count: 3
- production_sleep_planned_archive_count: 6
- production_sleep_planned_consolidated_node_count: 3
- production_sleep_applied_archived_node_count: 6
- production_sleep_applied_consolidated_node_count: 3
- production_sleep_dry_run_mutated: false

Paper evidence 重新通过，并加入 Phase1 production sleep gate：

- required_artifact_count: 26
- required_artifact_pass_rate: 1.0
- phase1_production_sleep_applied_consolidated_node_count: 3
- phase1_production_sleep_dry_run_mutated: false

## 当前边界

这一步补上了真实 JSONL graph sleep job，但还没有把它开成长期后台任务。生产使用仍应先 dry-run，检查 planned archive/consolidated nodes，再 gated apply。下一步如果继续补 daemon，需要把这个 job 接入 scheduler、budget/rate limit、checkpoint 和 rollback。
