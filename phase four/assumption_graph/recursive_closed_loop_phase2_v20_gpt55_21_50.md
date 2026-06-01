# Recursive Closed Loop: phase2 v20 gpt55 21-50

Date: 2026-06-01

## What Ran

The recursive frontier from `recursive_runner_phase2_v20_gpt55_21_50.json`
contained eight `run_fresh_ablation` verification leaves. Gemini
`gemini-3.5-flash-low` was attempted first but the proxy returned a 503
`No available channel` error, so the real execution used the configured
`gpt-5.5` channel.

To avoid wasting calls, each proposal was run on a scoped sample containing
only its trigger and control rows:

- `prop_dfa8c5b146f9`: 12 rows
- `prop_2ec0255facee`: 11 rows
- `prop_69d3d6dd67c7`: 12 rows
- `prop_54db59587ab9`: 11 rows
- `prop_1c34e615945c`: 11 rows
- `prop_fb5fc39a8090`: 11 rows
- `prop_1382d47d213b`: 12 rows
- `prop_2892408c37de`: 12 rows

All scoped answer coverage completed. The scoped ablation log is in
`recursive_scoped_ablation_run_gpt55_21_50.log`.

## Judge Results

Each candidate was judged against `phase2_v20_gpt55` on its scoped sample.

- `prop_dfa8c5b146f9`: 7 wins / 3 losses / 2 ties
- `prop_2ec0255facee`: 4 wins / 4 losses / 3 ties
- `prop_69d3d6dd67c7`: 9 wins / 3 losses / 0 ties
- `prop_54db59587ab9`: 4 wins / 5 losses / 2 ties
- `prop_1c34e615945c`: 3 wins / 5 losses / 3 ties
- `prop_fb5fc39a8090`: 4 wins / 4 losses / 3 ties
- `prop_1382d47d213b`: 7 wins / 3 losses / 2 ties
- `prop_2892408c37de`: 5 wins / 4 losses / 3 ties

The scoped judge log is in `recursive_scoped_judge_run_gpt55_21_50.log`.

## Acceptance Gate

The combined acceptance gate rejected all eight candidates:

- `reject_harm=1`
- `reject_benefit=7`
- `accept=0`

Details:

- `prop_dfa8c5b146f9`: `reject_harm`; trigger utility passed, but control-loss
  upper bound was too high.
- `prop_2ec0255facee`: `reject_benefit`.
- `prop_69d3d6dd67c7`: `reject_benefit`; raw scoped win rate was strong, but
  trigger lower bound did not clear the acceptance threshold.
- `prop_54db59587ab9`: `reject_benefit`.
- `prop_1c34e615945c`: `reject_benefit`.
- `prop_fb5fc39a8090`: `reject_benefit`.
- `prop_1382d47d213b`: `reject_benefit`.
- `prop_2892408c37de`: `reject_benefit`.

The acceptance payload is in
`recursive_candidate_acceptance_phase2_v20_gpt55_21_50.json`.

## Recursive Resume

`recursive_runner` was resumed with the real acceptance payload. The tree no
longer stops at `planned run_fresh_ablation` leaves:

- Frame counts: `root_problem=1`, `candidate_hypothesis=8`,
  `verification_subproblem=8`
- Status counts: `open=1`, `ready_to_act=8`, `resolved=8`
- Next frontier: `reject_or_revise_candidate=7`,
  `reject_or_narrow_scope=1`

The full resumed tree is in
`recursive_runner_phase2_v20_gpt55_21_50_resumed.json`.

## Gated Apply

The gated apply path was executed as a no-op because no proposal passed the
acceptance gate:

- accepted proposals: 0
- applied candidate nodes: 0
- graph mutated: false

The apply record is in `recursive_gated_apply_phase2_v20_gpt55_21_50.json`.
