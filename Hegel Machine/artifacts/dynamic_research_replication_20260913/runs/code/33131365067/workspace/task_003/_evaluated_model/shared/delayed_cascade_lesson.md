# Two-step delayed finite-memory cascade

The supported mechanism is:

m[t] = a0 + a1*u[t] + a2*u[t-2] + a3*u[t-2]^2
y[t] = b0 + b1*u[t] + b2*m[t]^2 + b3*m[t-1]

Pre-trajectory controls and instruments are zero. The fitted instrument intercept is numerically zero. Prediction needs the current control and three past controls; neither the paid meter nor past targets are needed.

Task 2 identified this delay, rejecting delays 1, 3, and 4, and validated an independent 21-row reset trajectory. Task 3 independently supported the same mechanism with refitted coefficients: 64 measured rows and a separate 22-row unmeasured reset trajectory containing sustained inputs, boundary controls, transitions, and decay to baseline. Both matched to floating-point precision.

Reuse shared/delayed_cascade.py only with a delay supported by current observations, and refit all coefficients. Its fitter requires measured meter rows; its predictor reconstructs the meter from controls. shared/memory_cascade.py supplies the feature accessor but its own predictor fixes delay one. Different response dynamics, stochastic instruments, or additional memory require revising the model.
