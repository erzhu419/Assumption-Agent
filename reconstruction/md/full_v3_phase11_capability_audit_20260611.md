# Full V3 Phase11: Capability Audit and Claim Guard

## 背景

`GPT_revise_v3.md` 的一个重要反馈是：当前 V3 有生产级 kernel，也有很多 shadow / fixture / frozen validation；如果不把二者机器可读地区分开，后续写论文或自动 promotion 时容易把“验证 harness”误写成“生产实现”。

Phase11 解决这个问题：给每个 V3 phase 建立 capability matrix，明确：

- artifact 是否通过；
- validation mode 是 live-derived、learned candidate、shadow harness、fixture/frozen harness 还是 mechanism validation；
- 当前是否能作为 production default；
- 允许声称什么；
- 禁止声称什么；
- promotion requirement 是什么。

## 实现

新增：

- `assumption_os/full_v3_phase11_capability_audit.py`
- `phase four/assumption_graph/paper_readiness_20260604/full_v3_phase11_capability_audit_20260611.json`

并接入：

- `assumption_os/full_v3_paper_scale_evidence.py`
- `tests/test_assumption_os.py`

## Validation

命令：

```bash
python3 -m assumption_os.full_v3_phase11_capability_audit \
  --root . \
  --out "phase four/assumption_graph/paper_readiness_20260604/full_v3_phase11_capability_audit_20260611.json"
```

结果：

- `pass`: true
- capability_count: 11
- artifact_pass_rate: 1.0
- outer_shell_count: 5
- outer_shell_production_claim_count: 0
- live_or_live_derived_count: 2
- shadow_or_fixture_count: 9
- blocked_claim_count: 22
- promotion_requirement_count: 11
- phase10_status: `learned_candidate_not_promoted`

Paper-scale evidence 也重新通过：

- required_artifact_count: 26
- required_artifact_pass_rate: 1.0
- v3_mechanism_count: 10
- v3_mechanism_pass_rate: 1.0

## 当前边界

Phase11 明确保留以下限制：

- Phase9 hybrid guard 是 retained gated profile，不是 unconditional default replacement。
- Phase10 discrete world model 是 learned candidate，不是强校准 task-world simulator。
- Phase0 / Phase1 / Phase3 / Phase5 / Phase7 这类 outer-shell phase 仍需要 fresh live promotion 才能算生产主循环能力。

这不是下游性能模块，而是 claim governance / promotion safety 模块。它的作用是让后续重构不会把“已经验证的机制”误标成“已经生产自治”。下一步应该继续把其中的 shadow/fixture phase 逐个替换成真实生产实现。
