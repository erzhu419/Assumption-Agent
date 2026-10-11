# Delayed finite-memory cascade

This task changed the instrument's control delay from one step to two:

m[t] = a0 + a1*u[t] + a2*u[t-2] + a3*u[t-2]^2
y[t] = b0 + b1*u[t] + b2*m[t]^2 + b3*m[t-1]

The instrument intercept fitted to numerical zero. Pre-trajectory controls and instruments are zero. Prediction requires the current control and three past controls; neither the paid meter nor past targets are needed.

The refitted mechanism matched 64 measured rows and an independent 21-row unmeasured reset trajectory to floating-point precision. Evaluated MSE across all 85 rows: 3.38e-32. Candidate instrument delays 1, 3, and 4 did not fit.

Reuse shared/delayed_cascade.py with an instrument_delay supported by current observations, and refit all coefficients. Its fitting function requires measured meter rows; its predictor reconstructs the meter from controls. The older shared/memory_cascade.py fixes the delay at one and is not valid for this task. Other response dynamics, stochastic instruments, or additional memory require revising the model.
