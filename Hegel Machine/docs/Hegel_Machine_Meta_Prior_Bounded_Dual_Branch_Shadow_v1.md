# Hegel Machine Meta-Prior Bounded Dual-Branch Shadow v1

Status: `IMPLEMENTED_SHADOW_LOCAL_MECHANICS`

## Purpose

This data-driven shadow-mechanics vertical slice closes one narrow gap between
the diagnostic ladder and the existing robustification/idealization interfaces.
Aggregate gains are not accepted as inputs.  Instead, the implementation derives
them from bounded fixture residual rows, constructs one candidate on each branch,
and compiles each accepted discovery object into an immutable `TheoryPatch`.
These fixture measurements are executable development inputs, not external
observations or scientific evidence.

The fixed order is:

```text
TRAIN residual rows
  -> measured branch diagnosis
  -> candidate or contract
  -> reduction replay
  -> ROBUSTIFICATION or IDEALIZATION TheoryPatch
  -> previously preregistered independent HOLDOUT probe
  -> bounded shadow result
```

The patch is fixed before the holdout rows are evaluated.  Changing a holdout
outcome changes the qualification receipt but not the candidate or patch.

## Robustification branch

Normal rows measure nominal utility and model-description cost.  Stress rows
measure tail utility, safety, and uncertainty coverage.  Zero-radius rows replay
the reduction to the parent prediction.  The bounded positive fixture derives:

- parent/candidate nominal utility: `0.900 / 0.885`;
- parent/candidate tail utility: `0.350 / 0.775`;
- parent/candidate safety: `0.000 / 1.000`;
- candidate coverage: `1.000`;
- maximum zero-radius reduction error: `0.000`.

The independent preregistered stress holdout contains two unseen rows.  Both
must independently satisfy the frozen tail-gain, safety-gain, and coverage
policy.  The rejection control has insufficient tail/safety gain, misses
coverage, and exceeds the permitted nominal loss.

## Idealization branch

Task-scope rows measure full versus quotient probe output, compute units, sample
units, and description units.  A separate boundary-counterexample row binds the
declared failure boundary and is not included in the task-scope discrepancy.
Full-model recovery is replayed directly on every preservation row.  The bounded
positive fixture derives:

- maximum task-probe discrepancy: `0.010` under budget `0.020`;
- compute gain: `0.700`;
- sample gain: `0.500`;
- complexity gain: `0.500`;
- maximum full-model recovery error: `0.000`.

The independent preregistered holdout contains two unseen task-scope rows.  The
rejection control preserves resource gains but moves the task output by `0.100`,
so it is rejected for probe non-preservation rather than rewarded for being
cheaper.

## Files and replay

- policy and measured rows:
  `config/meta_prior_bounded_dual_branch_shadow_v1.json`;
- mechanics:
  `src/hegel_machine/meta_prior_shadow_v1.py`;
- focused replay and controls:
  `tests/test_meta_prior_shadow_v1.py`.

The config is `SHADOW_ONLY_LOCAL_MECHANICS`.  This is not full scientific
completion of the Meta-prior objective.  The slice writes no ledger or
theory graph, performs no adoption or promotion, and makes no Meta-wide,
Q0.5b, Q1, C1, relation-invention, `ACTIVE`, or production claim.  It neither
modifies nor supplies evidence for the Q0.5b attempt-2/v2 paths.  Passing this
bounded fixture is not completion of the broader Meta-prior goal; automatic
operation discovery on non-fixture external tasks and formal downstream
evidence remain separate work.
