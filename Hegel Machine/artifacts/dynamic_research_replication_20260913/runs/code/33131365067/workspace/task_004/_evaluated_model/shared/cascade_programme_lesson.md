# Select the delay from current evidence

Task 4 returns to a one-step control delay, unlike tasks 2 and 3. Refitting shared/delayed_cascade.py with delay 1 matched 64 measured rows; the frozen model also matched a separate 22-row unmeasured reset trajectory to floating-point precision. Delays 2–4 failed on the measured data.

The supported mechanism remains m[t] = a0 + a1*u[t] + a2*u[t-d] + a3*u[t-d]^2 and y[t] = b0 + b1*u[t] + b2*m[t]^2 + b3*m[t-1], with zero pre-trajectory controls and instrument. The fitted instrument intercept is numerically zero. For d=1, prediction needs only the current control and two past controls; no paid meter or past targets are needed.

Reuse shared/delayed_cascade.py and its shared/memory_cascade.py feature accessor, but refit coefficients and identify the delay on each task. Historical delay persistence is not reliable. Stochastic instruments, different response dynamics, or additional memory require revising this model.
