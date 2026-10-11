# First-order linear response

Task 000: 32 observations with steps, zero-control intervals, and alternating controls identify y[t] = 0.65*y[t-1] + 0.45*u[t] + 0.8*sensor_a[t], with y[-1] = 0. Least-squares residuals are at floating-point roundoff. Adding an intercept and sensor_b gives coefficients indistinguishable from zero, so only sensor_a is needed.

linear_memory.py supplies fitting and prediction for a zero-intercept first-order target recurrence with selected current inputs. It reads the previous __target__ from each row's history, so trajectory resets do not require mutable predictor state.

Applicability: use when acquired data support this recurrence; refit coefficients on each new task. This library does not represent higher-order memory, nonlinear responses, or a nonzero reset state. Channel relevance and model structure can change between tasks.

## Task 003: both instruments matter

The 76 acquired observations fit the zero-reset first-order recurrence with coefficients for previous target, current u, sensor_a, sensor_b: [0.6500000000000005, 0.4499999999999994, 0.8000000000000006, 0.6000000000000002].
Training MSE: 4.38e-31; held-out MSE on the last 16 rows after fitting the first 60: 4.29e-31. Omitting sensor_b gives MSE 0.0288223.

The first 32-row probe had sensor_b identically zero and could not identify its coefficient. The subsequent 44-row probe supplied nonzero readings and revealed its contribution. A constant instrument does not establish irrelevance. Refit coefficients and reassess channels on each task.
