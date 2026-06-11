# Full V3 Phase10: Discrete Graph-Action World Model Selector

## 背景

`GPT_revise_v3.md` 指出的核心问题是：现有 Phase3/Phase5/Phase7 有 world-model / scheduler / long-run 的外壳，但真实行为更多是 fixture、shadow validation 或规则 guard。尤其 `world_model.py` 仍像 cheap verifier / budget gate，不是能预测 graph-action transition 的模型。

这一阶段没有直接替换主系统，而是先做一个垂直切片：把 Phase9 的 compact / micro / original V3 真实 judgment outcome 转成 action transition row，用脱敏 Boolean latent state 做 leave-one-out policy selection。

## 文献取舍

参考方向：

- Discrete World Models via Regularization: 使用 Boolean latent state、熵/独立性/局部 action change 正则，适合把 graph/query state 表示成可审计离散 bits。
- DreamerV2 / DreamerV3: world model 不只是评分器，而是用 compact latent transition 支持 planning/search-control。
- Causal world models for language agents: world state/action 应该能用自然语言变量解释，不能只是隐藏 prompt rule。
- Web agents with world models: agent world model 更适合预测 action -> observation/reward transition，而不是重构完整 observation。

因此这里没有做图像/RL 式 Dreamer simulator，也没有声称完整 task-world simulator；先做一个 graph-action transition predictor：state bits -> candidate answer profile action -> observed utility transition。

## 实现

新增：

- `assumption_os/full_v3_phase10_discrete_world_model_selector.py`
- `phase four/assumption_graph/paper_readiness_20260604/full_v3_phase10_discrete_world_model_feature_snapshot_20260611.json`
- `phase four/assumption_graph/paper_readiness_20260604/full_v3_phase10_discrete_world_model_selector_20260611.json`

输入只使用脱敏 artifacts：

- Phase9 hybrid heldout decisions / judgments
- Phase9 selective compact heldout judgments
- Phase9 micro heldout judgments
- Phase9 compact-frame support judgments
- redacted feature snapshot: `problem_id`, `domain`, `pattern_id`, `route_strategy_tag`, `feature_bits`

不读取 raw prompts、answers、reference answers、forensics 或 API secrets。

## Performance Validation

命令：

```bash
python3 -m assumption_os.full_v3_phase10_discrete_world_model_selector \
  --root . \
  --out "phase four/assumption_graph/paper_readiness_20260604/full_v3_phase10_discrete_world_model_selector_20260611.json"
```

结果：

- `pass`: true
- heldout transition rows: 54
- compact support rows: 31
- candidate transition rows: 17
- candidate action coverage: 1.0
- selected arms: compact 3, micro 14
- candidate LOO utility vs V1: 0.7059
- original V3 on same candidate cases vs V1: 0.6471
- candidate V1 lift over V3: +0.0588
- all-heldout policy vs V1: 0.6111
- original V3 all-heldout vs V1: 0.5926
- all-heldout lift over V3: +0.0185
- all-heldout vs original V3 utility: 0.5093
- retained hybrid all-heldout vs V1: 0.6481
- learned gap to retained hybrid: -0.0370

结论：outcome-only discrete world model 已经比 original V3 正向，但还不能替代 retained hybrid guard。它现在应保留为 world-model/search-control candidate。

## 重要限制

这不是完整 simulator。它只能预测当前几类 graph-action profile 的 transition utility，不能替代 live ablation / judge。

校准还不强：

- all-arm MAE: 0.3129
- base-rate MAE: 0.3032
- calibration beats base-rate: false

所以论文里不能写成“世界模型已经可独立模拟任务世界”。更准确表述是：Phase10 把原本的规则 guard 推进到一个离散 graph-action transition model；它能做正收益 search-control，但仍需要更大 live trace 和更强 calibration。

Teacher distillation 只作为 bootstrap upper bound 记录，不计入独立性能：它能复刻 retained hybrid，但这不等于真正学会了 world model。

## 下一步

1. 把 residual cluster -> candidate action profile -> world-model pre-screen 接成闭环。
2. 用更多 first-party live traces 训练/校准 graph-action transition predictor。
3. 做 leave-domain-out / leave-pattern-out validation，防止只记住 Phase9 artifact。
4. 让 selector 输出 uncertainty 和 abstain，而不是只输出 argmax arm。
5. 若 calibration 打过 base-rate，再考虑替换手写 hybrid guard。
