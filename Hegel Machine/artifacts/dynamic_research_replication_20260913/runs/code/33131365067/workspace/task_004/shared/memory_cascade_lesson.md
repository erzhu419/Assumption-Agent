# Finite-memory cascade

Current evidence identifies m[t] = a0 + a1*u[t] + a2*u[t-1] + a3*u[t-1]^2 and y[t] = b0 + b1*u[t] + b2*m[t]^2 + b3*m[t-1]. Pre-trajectory controls and the pre-trajectory instrument are zero. The fitted instrument intercept is numerically zero.

The mechanism matched 96 measured rows and a separate 20-row unmeasured reset trajectory to floating-point precision. Consequently prediction needs only u and two past controls, not the paid meter or past targets.

Reuse shared/memory_cascade.py only after current-task evidence supports this mechanism; refit coefficients from current observations. Its fit function requires meter measurements. Different dynamics, stochastic meters, or longer memory require revising the mechanism.
