# Recursive Positive Acceptance Report

Date: 2026-06-01

## Why this run exists

The previous recursive closed loop proved that the runner can reject bad or
underpowered candidates. This run proves the positive path as well:

candidate hypothesis -> verification child -> acceptance gate -> parent resume
-> gated graph apply.

## Primary positive example: SE hard-only policy

- Proposal: `prop_50e44c655f61`
- Applied node: `cand_50e44c655f61`
- Candidate: use proposal-forced Assumption Graph answers only on
  `software_engineering` hard rows, and fall back to `phase2_v20` on
  `software_engineering` medium rows.
- Evidence: bidirectional `gpt-5.5` pairwise judgments from the prior hard-only
  policy ablation.

Acceptance result:

- Decision: `accept`
- Trigger split: 4 hard SE rows, judged bidirectionally
- Trigger outcomes: `win=6`, `loss=2`
- Trigger utility: `0.75`
- Trigger LCB90: `0.5540408205773457`
- Control split: 5 medium SE fallback rows, judged bidirectionally
- Control outcomes: `tie=10`
- Control loss UCB90: `0.0`

Recursive propagation:

- Initial runner frontier: one `run_fresh_ablation` verification child
- Resumed runner: verification child `resolved`
- Parent next action: `apply_accepted_candidate_if_requested`
- Gated apply: `graph_mutated=true`, applied `cand_50e44c655f61`

Artifacts:

- `recursive_positive_se_hard_policy_acceptance.json`
- `recursive_positive_se_hard_policy_runner_resumed.json`
- `recursive_positive_se_hard_policy_gated_apply.json`

## Secondary positive-control: math/science bridge

- Proposal: `prop_e61e596b7f98`
- Applied node: `cand_e61e596b7f98`
- Candidate: use intent-aware math/science bypass routing instead of one generic
  math/science hygiene prompt.

Acceptance result:

- Decision: `accept`
- Trigger split: 9 math/science rows, judged bidirectionally
- Trigger outcomes: `win=16`, `loss=2`
- Trigger utility: `0.8888888888888888`
- Trigger LCB90: `0.794074074074074`
- Gated apply: `graph_mutated=true`, applied `cand_e61e596b7f98`

This is a useful domain-scoped positive-control, but it is weaker as a gate
demonstration than the SE hard-only policy because it has no explicit judged
control rows in the acceptance payload. The SE hard-only candidate is the
cleaner proof that the runner can pass both benefit and no-harm checks.

## Conclusion

The recursive assumption runner is now validated in both directions:

- Reject path: the prior 8 leaf candidates were judged, resumed, and rejected
  without mutating the graph.
- Accept path: `prop_50e44c655f61` passed trigger and control gates, resumed to
  the parent, and applied `cand_50e44c655f61` through gated graph mutation.
