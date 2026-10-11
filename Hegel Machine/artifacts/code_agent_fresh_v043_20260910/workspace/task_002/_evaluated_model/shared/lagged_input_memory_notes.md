# Linear response with a lagged control

Task 002: 32 varied observations identify
y[t] = 0.93*y[t-1] - 0.15*u[t] + 0.45*sensor_a[t] - 0.1395*u[t-1],
with missing pre-trajectory values set to zero. A full first-order ARX fit has MSE 2.01e-31 and an eight-row held-out MSE 9.66e-32. The intercept, sensor_b, and lagged instrument coefficients are numerically zero. The current-input-only shared linear model fails (MSE 0.00805): the previous control matters.

lagged_input_memory.py reuses shared.linear_memory.vector and appends selected one-step input lags. Prediction uses supplied history without mutable state. Refit coefficients and reassess channels on each task.

Applicability: first-order linear target recurrences with current and one-step-lagged inputs, no intercept, and zero reset values. It does not cover nonlinear dynamics, higher-order target memory, or nonzero reset states.
