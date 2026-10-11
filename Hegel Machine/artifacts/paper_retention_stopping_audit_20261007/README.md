# Frozen stopping and cost audit

`summary.json` diagnoses the original 20 first-revision discard failures using
the unchanged 2026-10-05 runtime. Each replay reproduces the original prefix
costs, challenge, selected kind, status, research costs, confirmation loss and
terminal MSE exactly. Original results remain unchanged; no LLM was called.

All 16 unresolved cases stop because multiple relation candidates survive fresh
confirmation: 14 after the first confirmation and two after the second, below
the six-attempt limit. Fifteen final probe plans report no predicted distinction;
one exhausts its local 6,000-unit generation allowance. The controller then
breaks on `relation_ambiguous`, and no revisit is scheduled. The other four cases
pass short confirmation and fail terminal evaluation.

Deletion removes C, its 17 abstractions/definitions and its meta-prior. Generic
method books, research/action experience and the other recorded collections are
unchanged. These outcomes therefore depend on retained knowledge together with
the frozen search and stopping policy. Further random comparisons are affordable,
but their usefulness has not been tested.

`cost_contract.json` separates billed candidate-row/fit-column work from expanded
program size, primitive execution, CPU time and model costs. For the separate
equal-success 56-task method comparison, two author phases plus the new
counterexample cost 118,797 computation units. Against the observed mean saving
of 7,407.21 units per target, projected payback is 17 comparable targets. This is
a compute-only projection with earlier common acquisition treated as sunk;
other resource dimensions do not establish payback.

Replay: `python3 scripts/audit_scoped_retention_stopping.py --apparatus SEED`.
Aggregate saved traces: append `--summarize` instead of `--apparatus SEED`.
`initial_trace/` preserves the first diagnostic before normalizing Python tuples
to JSON lists for challenge comparison; the corrected replay is the main trace.
