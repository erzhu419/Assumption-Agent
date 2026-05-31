# Recursive Runner Dry Run: phase2 v20 gpt55 21-50

Date: 2026-06-01

## Command

```bash
python3 -m assumption_os.recursive_runner \
  --graph-dir "phase four/assumption_graph" \
  --problem "Phase2 v20 graph self-evolution has judged losses and skipped math/science/software rows; decide which assumptions should be tested, repaired, or written back next." \
  --goal "Build a recursive assumption tree where each candidate hypothesis has explicit support, objections, falsification tests, and child subproblems that return updates to the parent." \
  --problem-id phase2_v20_gpt55_21_50_recursive \
  --eval-id recursive_runner_phase2_v20_gpt55_21_50 \
  --evolution-payload "phase four/assumption_graph/evolution_cycle_dryrun_phase2_v20_gpt55_21_50.json" \
  --max-children 8 \
  --max-depth 3 \
  --summary-out "phase four/assumption_graph/recursive_runner_phase2_v20_gpt55_21_50.json"
```

No graph mutation was performed.
No acceptance payload was supplied for this dry run, so unresolved verification
children remain in the frontier.

## Result

- Frame counts: `root_problem=1`, `candidate_hypothesis=8`, `verification_subproblem=8`
- Status counts: `open=1`, `ready_to_act=16`
- Depth counts: `0=1`, `1=8`, `2=8`
- Recursion edges: 16
- Leaf next actions: 8

The runner selected the top eight Bayesian-ranked proposal candidates from the
evolution-cycle payload. Each candidate frame contains:

- the parent/candidate assumption ids
- support and objection lists from preflight, falsification, regression, formal
  mapping, and Bayesian policy evidence
- ordered falsification tests
- a child verification subproblem
- a return-update contract describing how the child result updates the parent

All eight leaf frames are `run_fresh_ablation` subproblems. This means the
current recursive frontier is not "invent more rules"; it is to execute the
minimal candidate ablations required to resolve the highest-value parent
hypotheses.

The runner now also supports a second pass with `--acceptance-payload`. In that
mode, child verification results are propagated to the parent candidate:
`accept` becomes `apply_accepted_candidate_if_requested`, `reject_harm` becomes
`reject_or_narrow_scope`, `reject_benefit` becomes
`reject_or_revise_candidate`, and `insufficient_judgments` becomes
`collect_more_judgments`. The next frontier then points at the parent candidate
rather than the already-returned verification child.

## Interpretation

This is the first implementation of the recursive assumption loop:

```text
root problem
  -> candidate hypothesis
    -> verification/evidence/repair subproblem
      -> return update to parent
```

It still uses existing evolution artifacts as input and does not execute
external experiments by itself. The missing next layer is an executor that
consumes the leaf `command_hint`s, runs the ablations, feeds fresh judgments
through `candidate_acceptance`, then calls the runner again automatically with
`--acceptance-payload`.
