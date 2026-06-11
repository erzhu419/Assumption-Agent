# Full V3 Phase0: Production Contract Gate

## 背景

`GPT_revise_v3.md` 指出 Phase0 contract checker 仍偏 wrapper over v2 bypass，缺少真正可复用的 production pre-overlay contract checker。

这次补了一个独立模块：

- `assumption_os/proposal_contract.py`

它在 candidate proposal 进入 overlay 前检查：

- proposal id / type / parent ref；
- candidate node schema；
- scope / measurable effect / risk prediction；
- verifier；
- rollback；
- negative-control / outside-control / regression-harm guard；
- edge parse 和 candidate connection；
- duplicate / conflict；
- manifest completeness。

新增生产入口：

```python
apply_contract_checked_proposal_overlay(store, proposal_payload)
```

只有通过 contract 的 proposal id 会被传给旧 overlay helper；如果 admitted list 为空，会直接返回，不会误触发旧 helper 的“空 filter 等于不过滤”行为。

## Validation

新增单元测试：

```bash
python3 -m unittest \
  tests.test_assumption_os.AssumptionOSTest.test_full_v3_phase0_contract_checker_validates_overlay_admission \
  tests.test_assumption_os.AssumptionOSTest.test_proposal_contract_checked_overlay_quarantines_invalid_candidates
```

结果：通过。

Phase0 artifact 重新生成：

```bash
python3 -m assumption_os.full_v3_phase0_contract_checker \
  --root . \
  --out "phase four/assumption_graph/paper_readiness_20260604/full_v3_phase0_contract_checker_20260611.json"
```

关键指标：

- production_contract_proposal_count: 2
- production_contract_admitted_count: 1
- production_contract_quarantined_count: 1
- production_contract_invalid_admitted_count: 0
- production_contract_applied_count: 1

Paper evidence 也重新通过，并加入 Phase0 production contract gate：

- required_artifact_count: 26
- required_artifact_pass_rate: 1.0
- phase0_production_contract_invalid_admitted_count: 0
- phase0_production_contract_applied_count: 1

## 当前边界

这一步解决的是“真实 pre-overlay contract checker”缺口，但还没有把所有历史脚本默认改成强制 contract mode。为了保持复现实验兼容性，旧 `apply_proposal_overlay` 保持原语义；新生产路径应使用 `apply_contract_checked_proposal_overlay`。
