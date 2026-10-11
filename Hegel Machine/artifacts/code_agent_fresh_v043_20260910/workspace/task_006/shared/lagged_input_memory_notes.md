# Linear response with a lagged control

Task 002: 32 varied observations identify
y[t] = 0.93*y[t-1] - 0.15*u[t] + 0.45*sensor_a[t] - 0.1395*u[t-1],
with missing pre-trajectory values set to zero. A full first-order ARX fit has MSE 2.01e-31 and an eight-row held-out MSE 9.66e-32. The intercept, sensor_b, and lagged instrument coefficients are numerically zero. The current-input-only shared linear model fails (MSE 0.00805): the previous control matters.

lagged_input_memory.py reuses shared.linear_memory.vector and appends selected one-step input lags. Prediction uses supplied history without mutable state. Refit coefficients and reassess channels on each task.

Applicability: first-order linear target recurrences with current and one-step-lagged inputs, no intercept, and zero reset values. It does not cover nonlinear dynamics, higher-order target memory, or nonzero reset states.

## Task 006: lagged-control recurrence confirmed after refitting

The 48 acquired observations identify y[t] = 0.93*y[t-1] - 0.15*u[t] + 0.45*sensor_a[t] - 0.1395*u[t-1], with zero reset values. Reusing lagged_input_memory.py and refitting gives MSE 8.34e-32; fitting the first 32 rows and predicting the last 16 gives MSE 1.36e-32. The current-input-only model fails (MSE 0.0100004). The full design including both current and lagged instruments and an intercept has rank 8 of 8; the additional coefficients are numerically zero. Only sensor_a is needed. Refit coefficients and reassess channels on future tasks.
