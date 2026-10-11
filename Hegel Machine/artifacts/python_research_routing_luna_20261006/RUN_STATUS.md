# Operator-interrupted development run

This run was stopped before either active target completed; the other two targets
had not started. There are no terminal evaluations or research success/failure
counts. The original source snapshot and partial trajectories remain preserved.

Concurrent implementation tests found a same-path whole-data interpolation route
that bypassed the earlier validation, and a `None`-MSE recommendation error. The
operator stopped PID 2836491 and its children (session exit 143). Source files were
not changed while this run executed. Repairs and any full replay use a new directory.

Seven saved calls completed: three in `trial_000`, four in `trial_001`. They report
178,244 tokens (176,568 input; 1,676 output), including 93,952 cached input tokens
(53.21% of input). All seven logs contain usage; no incomplete call log was saved.
An in-flight call without a saved usage report cannot be counted or priced here.

Latest active checkpoints record 144 paid observations, no meter charges, and 144
scoring units. Both agents were still researching. These are recorded partial
costs, excluding earlier source learning and any unreported work after checkpoints.
Per-call usage and checkpoint costs are in `partial_usage.json`.
